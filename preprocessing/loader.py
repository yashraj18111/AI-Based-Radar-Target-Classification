"""
Radar dataset loader for CARRADA (primary) and RADDet (extensible).

Provides a unified interface to discover sequences, load radar tensors,
parse annotations, and normalize data for downstream model training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in radar map coordinates."""

    class_name: str
    range_bin: float
    doppler_bin: float
    width: float
    height: float

    @property
    def class_id(self) -> int:
        return config.CLASS_TO_ID.get(self.class_name, 0)

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) in map pixel coordinates."""
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        return (
            self.doppler_bin - half_w,
            self.range_bin - half_h,
            self.doppler_bin + half_w,
            self.range_bin + half_h,
        )


@dataclass
class RadarSample:
    """Single radar frame with optional annotations."""

    sequence_id: str
    frame_id: str
    modality: str
    radar_map: np.ndarray
    annotations: List[BoundingBox] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.radar_map.shape

    @property
    def num_objects(self) -> int:
        return len(self.annotations)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_radar_map(
    radar_map: np.ndarray,
    method: str = config.NORMALIZE_METHOD,
) -> np.ndarray:
    """
    Normalize a raw radar power map for neural network input.

    Parameters
    ----------
    radar_map : np.ndarray
        2-D float array (range x doppler or range x angle).
    method : str
        One of ``log_minmax``, ``zscore``, or ``none``.

    Returns
    -------
    np.ndarray
        Normalized array with the same shape, dtype float32.
    """
    data = radar_map.astype(np.float32)

    if method == "none":
        return data

    if method == "log_minmax":
        data = np.log1p(np.maximum(data, 0.0) + config.LOG_EPSILON)
        low, high = np.percentile(data, config.CLIP_PERCENTILE)
        if high <= low:
            return np.zeros_like(data)
        data = np.clip(data, low, high)
        return (data - low) / (high - low)

    if method == "zscore":
        mean = data.mean()
        std = data.std()
        if std < config.LOG_EPSILON:
            return np.zeros_like(data)
        return (data - mean) / std

    raise ValueError(f"Unknown normalization method: {method!r}")


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_carrada_annotations(
    annotation_path: Path,
    frame_id: str,
) -> List[BoundingBox]:
    """Parse CARRADA ``instances.json`` for a single frame."""
    if not annotation_path.exists():
        logger.debug("No annotation file at %s", annotation_path)
        return []

    with annotation_path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = json.load(handle)

    frame_key = frame_id.zfill(6) if frame_id.isdigit() else frame_id
    frame_data = raw.get(frame_key, {})

    boxes: List[BoundingBox] = []
    for class_name in ("car", "cyclist", "pedestrian"):
        for entry in frame_data.get(class_name, []):
            if len(entry) < 4:
                continue
            range_bin, doppler_bin, width, height = entry[:4]
            boxes.append(
                BoundingBox(
                    class_name=class_name,
                    range_bin=float(range_bin),
                    doppler_bin=float(doppler_bin),
                    width=float(width),
                    height=float(height),
                )
            )
    return boxes


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class RadarDataset:
    """
    Iterable dataset over radar sequences.

    Example
    -------
    >>> dataset = RadarDataset(split="train")
    >>> sample = dataset[0]
    >>> print(sample.radar_map.shape, sample.num_objects)
    """

    def __init__(
        self,
        split: str = config.DEFAULT_SPLIT,
        modality: str = config.DEFAULT_MODALITY,
        dataset_root: Optional[Union[str, Path]] = None,
        normalize: bool = True,
        load_annotations: bool = True,
    ) -> None:
        self.split = split
        self.modality = modality
        self.dataset_root = Path(dataset_root or config.get_dataset_root())
        self.normalize = normalize
        self.load_annotations = load_annotations

        self._samples: List[Tuple[str, str]] = self._build_index()
        if not self._samples:
            logger.warning(
                "No samples found under %s for split=%r. "
                "Download CARRADA and place it in dataset/raw/CARRADA/.",
                self.dataset_root,
                split,
            )

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def _build_index(self) -> List[Tuple[str, str]]:
        """Discover (sequence_id, frame_id) pairs for the configured split."""
        if config.DATASET_NAME == "carrada":
            return self._index_carrada()
        if config.DATASET_NAME == "raddet":
            return self._index_raddet()
        raise ValueError(f"Unsupported dataset: {config.DATASET_NAME!r}")

    def _index_carrada(self) -> List[Tuple[str, str]]:
        sequences = config.get_split_sequences(self.split)
        index: List[Tuple[str, str]] = []

        for seq_id in sequences:
            modality_dir = self.dataset_root / seq_id / self.modality
            if not modality_dir.is_dir():
                logger.debug("Missing modality dir: %s", modality_dir)
                continue

            for frame_path in sorted(modality_dir.glob(f"*{config.FRAME_EXTENSION}")):
                frame_id = frame_path.stem
                index.append((seq_id, frame_id))

        return index

    def _index_raddet(self) -> List[Tuple[str, str]]:
        """Placeholder index builder for RADDet (Phase 2 extension)."""
        split_dir = self.dataset_root / config.RADDET_SPLITS.get(self.split, self.split)
        if not split_dir.is_dir():
            return []

        index: List[Tuple[str, str]] = []
        for cube_path in sorted(split_dir.glob(f"*{config.RADDET_CUBE_SUFFIX}")):
            frame_id = cube_path.name.replace(config.RADDET_CUBE_SUFFIX, "")
            index.append((self.split, frame_id))
        return index

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> RadarSample:
        sequence_id, frame_id = self._samples[index]
        return self.load_sample(sequence_id, frame_id)

    def __iter__(self) -> Iterator[RadarSample]:
        for idx in range(len(self)):
            yield self[idx]

    def load_sample(self, sequence_id: str, frame_id: str) -> RadarSample:
        """Load a single frame by sequence and frame identifier."""
        if config.DATASET_NAME == "carrada":
            return self._load_carrada_sample(sequence_id, frame_id)
        if config.DATASET_NAME == "raddet":
            return self._load_raddet_sample(sequence_id, frame_id)
        raise ValueError(f"Unsupported dataset: {config.DATASET_NAME!r}")

    def _load_carrada_sample(self, sequence_id: str, frame_id: str) -> RadarSample:
        seq_dir = self.dataset_root / sequence_id
        frame_path = seq_dir / self.modality / f"{frame_id}{config.FRAME_EXTENSION}"

        if not frame_path.exists():
            raise FileNotFoundError(f"Radar frame not found: {frame_path}")

        radar_map = np.load(frame_path)
        if self.normalize:
            radar_map = normalize_radar_map(radar_map)

        annotations: List[BoundingBox] = []
        if self.load_annotations:
            ann_path = seq_dir / config.CARRADA_ANNOTATION_DIR / config.CARRADA_ANNOTATION_FILE
            annotations = _parse_carrada_annotations(ann_path, frame_id)

        return RadarSample(
            sequence_id=sequence_id,
            frame_id=frame_id,
            modality=self.modality,
            radar_map=radar_map,
            annotations=annotations,
            metadata={"source_path": str(frame_path)},
        )

    def _load_raddet_sample(self, sequence_id: str, frame_id: str) -> RadarSample:
        split_dir = self.dataset_root / sequence_id
        cube_path = split_dir / f"{frame_id}{config.RADDET_CUBE_SUFFIX}"

        if not cube_path.exists():
            raise FileNotFoundError(f"RADDet cube not found: {cube_path}")

        cube = np.load(cube_path)
        radar_map = cube[..., 0] if cube.ndim == 3 else cube
        if self.normalize:
            radar_map = normalize_radar_map(radar_map)

        return RadarSample(
            sequence_id=sequence_id,
            frame_id=frame_id,
            modality="range_doppler",
            radar_map=radar_map,
            annotations=[],
            metadata={"source_path": str(cube_path)},
        )

    # ------------------------------------------------------------------
    # Batch utilities
    # ------------------------------------------------------------------

    def get_batch(self, indices: List[int]) -> List[RadarSample]:
        """Load multiple samples by index."""
        return [self[i] for i in indices]

    def summary(self) -> Dict[str, Any]:
        """Return a lightweight dataset summary for logging."""
        return {
            "dataset": config.DATASET_NAME,
            "split": self.split,
            "modality": self.modality,
            "root": str(self.dataset_root),
            "num_samples": len(self),
            "normalize": self.normalize,
            "load_annotations": self.load_annotations,
        }


def create_dataset(
    split: str = config.DEFAULT_SPLIT,
    modality: str = config.DEFAULT_MODALITY,
    **kwargs: Any,
) -> RadarDataset:
    """Factory function for constructing a :class:`RadarDataset`."""
    return RadarDataset(split=split, modality=modality, **kwargs)
