"""
End-to-end radar preprocessing pipeline.

Steps
-----
1. Read raw radar data from CARRADA (or configured dataset).
2. Validate samples — skip missing or corrupted frames.
3. Normalize radar tensors (log-minmax / z-score).
4. Resize to the target shape defined in ``config.py``.
5. Assign samples to official train / val / test splits.
6. Optionally augment the training split.
7. Save processed tensors and split manifests to ``dataset/processed/``.

Usage
-----
    python -m preprocessing.preprocess
    python -m preprocessing.preprocess --splits train val --no-augment
    python -m preprocessing.preprocess --modality range_doppler_numpy
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from preprocessing.loader import RadarSample
from preprocessing.splitter import SplitManifest, samples_from_split, summarize_splits
from preprocessing.storage import (
    save_manifest,
    save_preprocessing_config,
    save_processed_sample,
)
from preprocessing.transforms import maybe_augment_sample, transform_sample
from preprocessing.validators import ValidationReport, filter_valid_samples

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessConfig:
    """Runtime options for one preprocessing run."""

    splits: Tuple[str, ...] = config.PREPROCESS_SPLITS
    modality: str = config.DEFAULT_MODALITY
    output_root: Path = config.PROCESSED_DATA_DIR
    dataset_root: Optional[Path] = None
    normalize: bool = True
    resize: bool = True
    target_shape: Optional[Tuple[int, int]] = None
    augment_train: bool = config.AUGMENTATION_ENABLED
    require_annotations: bool = False
    seed: int = config.TRAINING.seed


@dataclass
class SplitResult:
    """Statistics collected while processing one split."""

    split: str
    loaded: int = 0
    valid: int = 0
    saved: int = 0
    skipped: int = 0
    augmented: int = 0
    validation: ValidationReport = field(default_factory=ValidationReport)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "loaded": self.loaded,
            "valid": self.valid,
            "saved": self.saved,
            "skipped": self.skipped,
            "augmented": self.augmented,
            "validation": self.validation.as_dict(),
        }


@dataclass
class PipelineResult:
    """Aggregated result across all processed splits."""

    splits: Dict[str, SplitResult] = field(default_factory=dict)
    output_root: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "output_root": self.output_root,
            "splits": {name: result.as_dict() for name, result in self.splits.items()},
        }


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for reproducible augmentation."""
    np.random.seed(seed)


def read_split_samples(
    split: str,
    modality: str,
    dataset_root: Optional[Path] = None,
) -> List[RadarSample]:
    """
    Step 1 — Read raw radar data for one split.

    Loads ``.npy`` frames and parses CARRADA annotations without applying
    normalization or resize (those happen in later steps).
    """
    logger.info("Reading split=%r modality=%r", split, modality)
    return samples_from_split(
        split=split,
        modality=modality,
        dataset_root=str(dataset_root) if dataset_root else None,
        normalize=False,
    )


def validate_samples(
    samples: List[RadarSample],
    require_annotations: bool = False,
    allow_resize: bool = True,
) -> Tuple[List[RadarSample], ValidationReport]:
    """
    Step 2 — Validate and filter missing or corrupted samples.

    Skips frames that are unreadable, empty, all-zero, or contain NaN/Inf.
    """
    return filter_valid_samples(
        samples,
        require_annotations=require_annotations,
        allow_resize=allow_resize,
    )


def process_sample(
    sample: RadarSample,
    normalize: bool = True,
    resize: bool = True,
    target_shape: Optional[Tuple[int, int]] = None,
) -> RadarSample:
    """
    Steps 3 & 4 — Normalize and optionally resize one sample.
    """
    return transform_sample(
        sample,
        normalize=normalize,
        resize=resize,
        target_shape=target_shape,
    )


def augment_if_training(
    sample: RadarSample,
    split: str,
    augment_train: bool,
) -> List[RadarSample]:
    """
    Step 6 — Apply optional augmentation (train split only).
    """
    if split != "train" or not augment_train:
        return [sample]
    return maybe_augment_sample(sample, enabled=True)


def save_split_outputs(
    samples: List[RadarSample],
    split: str,
    output_root: Path,
    manifest: SplitManifest,
    suffix_map: Optional[Dict[int, str]] = None,
) -> int:
    """
    Step 7 — Persist processed samples and update the split manifest.

    Returns the number of files written.
    """
    saved = 0
    for idx, sample in enumerate(samples):
        suffix = (suffix_map or {}).get(idx, "")
        out_path = save_processed_sample(
            sample,
            split=split,
            output_root=output_root,
            suffix=suffix,
        )
        manifest.add(
            sequence_id=sample.sequence_id,
            frame_id=sample.frame_id,
            processed_path=str(out_path.relative_to(output_root)),
            num_objects=sample.num_objects,
            extra={"metadata": sample.metadata},
        )
        saved += 1
    return saved


def preprocess_split(
    split: str,
    cfg: PreprocessConfig,
) -> Tuple[SplitManifest, SplitResult]:
    """Run the full preprocessing pipeline for a single split."""
    result = SplitResult(split=split)
    target_shape = cfg.target_shape or config.get_target_shape(cfg.modality)

    # Step 1: Read
    raw_samples = read_split_samples(
        split=split,
        modality=cfg.modality,
        dataset_root=cfg.dataset_root,
    )
    result.loaded = len(raw_samples)

    # Step 2: Validate
    valid_samples, validation_report = validate_samples(
        raw_samples,
        require_annotations=cfg.require_annotations,
        allow_resize=cfg.resize,
    )
    result.validation = validation_report
    result.valid = len(valid_samples)
    result.skipped = result.loaded - result.valid

    manifest = SplitManifest(split=split, modality=cfg.modality)
    processed_for_save: List[RadarSample] = []
    suffix_map: Dict[int, str] = {}

    # Steps 3–6: Transform + augment
    for sample in valid_samples:
        transformed = process_sample(
            sample,
            normalize=cfg.normalize,
            resize=cfg.resize,
            target_shape=target_shape,
        )

        augmented_list = augment_if_training(
            transformed,
            split=split,
            augment_train=cfg.augment_train,
        )

        base_idx = len(processed_for_save)
        processed_for_save.append(augmented_list[0])

        for aug_idx, aug_sample in enumerate(augmented_list[1:], start=1):
            idx = base_idx + aug_idx
            processed_for_save.append(aug_sample)
            suffix_map[idx] = f"aug{aug_idx}"
            result.augmented += 1

    # Step 7: Save
    result.saved = save_split_outputs(
        processed_for_save,
        split=split,
        output_root=cfg.output_root,
        manifest=manifest,
        suffix_map=suffix_map,
    )

    save_manifest(manifest, output_root=cfg.output_root)
    logger.info(
        "Split %r done: loaded=%d valid=%d saved=%d skipped=%d augmented=%d",
        split,
        result.loaded,
        result.valid,
        result.saved,
        result.skipped,
        result.augmented,
    )
    return manifest, result


def run_preprocessing(cfg: PreprocessConfig) -> PipelineResult:
    """Execute the full preprocessing pipeline across all requested splits."""
    set_seed(cfg.seed)
    cfg.output_root.mkdir(parents=True, exist_ok=True)

    pipeline_result = PipelineResult(output_root=str(cfg.output_root))

    # Step 5: Process each official split independently
    for split in cfg.splits:
        _, split_result = preprocess_split(split, cfg)
        pipeline_result.splits[split] = split_result

    # Persist run configuration for reproducibility
    run_config = {
        "dataset": config.DATASET_NAME,
        "modality": cfg.modality,
        "splits": list(cfg.splits),
        "normalize": cfg.normalize,
        "normalize_method": config.NORMALIZE_METHOD,
        "resize": cfg.resize,
        "target_shape": cfg.target_shape or config.get_target_shape(cfg.modality),
        "augment_train": cfg.augment_train,
        "augmentation_config": config.AUGMENTATION_CONFIG,
        "seed": cfg.seed,
        "results": pipeline_result.as_dict(),
    }
    save_preprocessing_config(run_config, output_root=cfg.output_root)

    return pipeline_result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preprocess radar dataset for training.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(config.PREPROCESS_SPLITS),
        choices=list(config.PREPROCESS_SPLITS),
        help="Splits to preprocess (default: train val test).",
    )
    parser.add_argument(
        "--modality",
        default=config.DEFAULT_MODALITY,
        help="Radar modality folder to load (default: range_doppler_numpy).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.PROCESSED_DATA_DIR,
        help="Output directory for processed data.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Override raw dataset root path.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalization step.",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Keep native radar map resolution.",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable training augmentation.",
    )
    parser.add_argument(
        "--require-annotations",
        action="store_true",
        help="Skip frames without bounding-box annotations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.TRAINING.seed,
        help="Random seed for augmentation.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print split summary without processing.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.summary_only:
        summary = summarize_splits(
            splits=tuple(args.splits),
            modality=args.modality,
        )
        import json

        print(json.dumps(summary, indent=2))
        return 0

    cfg = PreprocessConfig(
        splits=tuple(args.splits),
        modality=args.modality,
        output_root=args.output,
        dataset_root=args.dataset_root,
        normalize=not args.no_normalize,
        resize=not args.no_resize,
        augment_train=not args.no_augment,
        require_annotations=args.require_annotations,
        seed=args.seed,
    )

    result = run_preprocessing(cfg)

    import json

    print("\nPreprocessing complete.")
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
