"""
Persistence layer for preprocessed radar data.

Saves individual frames as compressed ``.npz`` archives plus JSON manifests
that downstream training code can consume directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import config
from preprocessing.loader import BoundingBox, RadarSample
from preprocessing.splitter import SplitManifest

logger = logging.getLogger(__name__)


def _sample_filename(sequence_id: str, frame_id: str, suffix: str = "") -> str:
    safe_seq = sequence_id.replace("/", "_").replace("\\", "_")
    safe_frame = frame_id.replace("/", "_").replace("\\", "_")
    tag = f"_{suffix}" if suffix else ""
    ext = config.PROCESSED_FILE_FORMAT
    return f"{safe_seq}__{safe_frame}{tag}.{ext}"


def _bbox_to_array(boxes: List[BoundingBox]) -> np.ndarray:
    """
    Serialize bounding boxes to an ``(N, 5)`` array:

    ``[class_id, range_bin, doppler_bin, width, height]``
    """
    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for bbox in boxes:
        rows.append(
            [
                bbox.class_id,
                bbox.range_bin,
                bbox.doppler_bin,
                bbox.width,
                bbox.height,
            ]
        )
    return np.array(rows, dtype=np.float32)


def get_split_output_dir(
    split: str,
    output_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Return ``dataset/processed/<split>/`` creating it if needed."""
    root = Path(output_root or config.PROCESSED_DATA_DIR)
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / config.PROCESSED_FRAME_DIR).mkdir(exist_ok=True)
    return split_dir


def save_processed_sample(
    sample: RadarSample,
    split: str,
    output_root: Optional[Union[str, Path]] = None,
    suffix: str = "",
) -> Path:
    """
    Save one processed sample to disk.

    Returns the path of the saved file.
    """
    split_dir = get_split_output_dir(split, output_root)
    frame_dir = split_dir / config.PROCESSED_FRAME_DIR
    filename = _sample_filename(sample.sequence_id, sample.frame_id, suffix=suffix)
    out_path = frame_dir / filename

    radar_map = sample.radar_map.astype(np.float32)
    bboxes = _bbox_to_array(sample.annotations)

    if config.PROCESSED_FILE_FORMAT == "npz":
        np.savez_compressed(
            out_path,
            radar_map=radar_map,
            bboxes=bboxes,
            sequence_id=sample.sequence_id,
            frame_id=sample.frame_id,
            modality=sample.modality,
        )
    elif config.PROCESSED_FILE_FORMAT == "npy":
        np.save(out_path, radar_map)
    else:
        raise ValueError(f"Unsupported format: {config.PROCESSED_FILE_FORMAT!r}")

    return out_path


def load_processed_sample(path: Union[str, Path]) -> RadarSample:
    """Load a previously saved processed sample."""
    path = Path(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            radar_map = data["radar_map"]
            bboxes_raw = data["bboxes"]
            sequence_id = str(data["sequence_id"])
            frame_id = str(data["frame_id"])
            modality = str(data["modality"])

        boxes: List[BoundingBox] = []
        for row in bboxes_raw:
            class_id = int(row[0])
            class_name = config.CLASS_NAMES[class_id] if class_id < len(config.CLASS_NAMES) else "background"
            boxes.append(
                BoundingBox(
                    class_name=class_name,
                    range_bin=float(row[1]),
                    doppler_bin=float(row[2]),
                    width=float(row[3]),
                    height=float(row[4]),
                )
            )

        return RadarSample(
            sequence_id=sequence_id,
            frame_id=frame_id,
            modality=modality,
            radar_map=radar_map,
            annotations=boxes,
            metadata={"processed_path": str(path)},
        )

    if path.suffix == ".npy":
        radar_map = np.load(path)
        return RadarSample(
            sequence_id="unknown",
            frame_id=path.stem,
            modality=config.DEFAULT_MODALITY,
            radar_map=radar_map,
            metadata={"processed_path": str(path)},
        )

    raise ValueError(f"Unsupported processed file: {path}")


def save_manifest(
    manifest: SplitManifest,
    output_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Write a split manifest JSON file."""
    split_dir = get_split_output_dir(manifest.split, output_root)
    manifest_path = split_dir / config.PROCESSED_MANIFEST_NAME

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.as_dict(), handle, indent=2)

    logger.info(
        "Saved %s manifest with %d entries to %s",
        manifest.split,
        manifest.size,
        manifest_path,
    )
    return manifest_path


def save_preprocessing_config(
    config_dict: Dict[str, Any],
    output_root: Optional[Union[str, Path]] = None,
) -> Path:
    """Persist the preprocessing configuration alongside processed data."""
    root = Path(output_root or config.PROCESSED_DATA_DIR)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / config.PROCESSED_CONFIG_NAME

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config_dict, handle, indent=2, default=_json_default)

    return config_path


def load_manifest(
    split: str,
    output_root: Optional[Union[str, Path]] = None,
) -> SplitManifest:
    """Load a previously saved split manifest."""
    split_dir = get_split_output_dir(split, output_root)
    manifest_path = split_dir / config.PROCESSED_MANIFEST_NAME

    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    manifest = SplitManifest(
        split=raw["split"],
        modality=raw["modality"],
        entries=raw.get("entries", []),
    )
    return manifest


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
