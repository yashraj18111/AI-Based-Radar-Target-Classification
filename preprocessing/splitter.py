"""
Train / validation / test split management for radar datasets.

Uses official CARRADA sequence splits from ``config.py`` and builds
per-split manifests that the preprocessing pipeline consumes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from preprocessing.loader import RadarDataset, RadarSample

logger = logging.getLogger(__name__)


@dataclass
class SplitManifest:
    """Manifest describing all samples in one data split."""

    split: str
    modality: str
    entries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.entries)

    def add(
        self,
        sequence_id: str,
        frame_id: str,
        processed_path: str,
        num_objects: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "sequence_id": sequence_id,
            "frame_id": frame_id,
            "processed_path": processed_path,
            "num_objects": num_objects,
        }
        if extra:
            entry.update(extra)
        self.entries.append(entry)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "modality": self.modality,
            "num_samples": self.size,
            "entries": self.entries,
        }


def get_official_splits() -> Dict[str, List[str]]:
    """Return the configured official split -> sequence mapping."""
    if config.DATASET_NAME == "carrada":
        return config.CARRADA_SPLITS
    raise NotImplementedError(
        f"Official splits not defined for dataset {config.DATASET_NAME!r}"
    )


def build_split_index(
    split: str,
    modality: str = config.DEFAULT_MODALITY,
    dataset_root: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """
    Discover raw (sequence_id, frame_id) pairs for a split.

    Delegates to :class:`RadarDataset` index builder without loading tensors.
    """
    dataset = RadarDataset(
        split=split,
        modality=modality,
        dataset_root=dataset_root,
        normalize=False,
        load_annotations=False,
    )
    return list(dataset._samples)


def summarize_splits(
    splits: Tuple[str, ...] = config.PREPROCESS_SPLITS,
    modality: str = config.DEFAULT_MODALITY,
) -> Dict[str, Any]:
    """Return sample counts per split without loading full tensors."""
    summary: Dict[str, Any] = {"modality": modality, "splits": {}}

    for split in splits:
        index = build_split_index(split, modality=modality)
        sequences = get_official_splits().get(split, [])
        summary["splits"][split] = {
            "num_sequences": len(sequences),
            "num_frames": len(index),
        }

    return summary


def samples_from_split(
    split: str,
    modality: str = config.DEFAULT_MODALITY,
    dataset_root: Optional[str] = None,
    normalize: bool = False,
) -> List[RadarSample]:
    """Load all raw samples for a split (no transform applied)."""
    dataset = RadarDataset(
        split=split,
        modality=modality,
        dataset_root=dataset_root,
        normalize=normalize,
        load_annotations=True,
    )

    samples: List[RadarSample] = []
    for idx in range(len(dataset)):
        try:
            samples.append(dataset[idx])
        except (FileNotFoundError, OSError, ValueError) as exc:
            seq_id, frame_id = dataset._samples[idx]
            logger.warning(
                "Skipping unreadable sample %s/%s in split %r: %s",
                seq_id,
                frame_id,
                split,
                exc,
            )
    return samples
