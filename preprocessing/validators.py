"""
Sample validation utilities for the radar preprocessing pipeline.

Filters missing files, unreadable arrays, and numerically invalid tensors
before they enter normalization or augmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from preprocessing.loader import BoundingBox, RadarSample

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Outcome of validating a single radar sample."""

    is_valid: bool
    sequence_id: str
    frame_id: str
    reason: str = ""
    sample: Optional[RadarSample] = None


@dataclass
class ValidationReport:
    """Aggregated validation statistics for a batch or split."""

    total: int = 0
    valid: int = 0
    skipped: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def record(self, result: ValidationResult) -> None:
        self.total += 1
        if result.is_valid:
            self.valid += 1
            return

        self.skipped += 1
        key = result.reason or "unknown"
        self.reasons[key] = self.reasons.get(key, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "valid": self.valid,
            "skipped": self.skipped,
            "reasons": self.reasons,
        }


def _expected_native_shape(modality: str) -> Tuple[int, int]:
    if "angle" in modality:
        return config.RA_MAP_SHAPE
    return config.RD_MAP_SHAPE


def validate_radar_array(
    radar_map: np.ndarray,
    modality: str,
    allow_resize: bool = True,
) -> Tuple[bool, str]:
    """
    Check that a raw radar tensor is loadable and numerically sane.

    Returns (is_valid, reason_if_invalid).
    """
    if radar_map is None:
        return False, "null_array"

    if not isinstance(radar_map, np.ndarray):
        return False, "not_ndarray"

    if radar_map.size == 0:
        return False, "empty_array"

    if radar_map.ndim not in (2, 3):
        return False, f"invalid_ndim_{radar_map.ndim}"

    data = radar_map[..., 0] if radar_map.ndim == 3 else radar_map

    if not np.isfinite(data).all():
        return False, "non_finite_values"

    if np.all(data == 0):
        return False, "all_zero"

    native_shape = _expected_native_shape(modality)
    if data.shape != native_shape and not allow_resize:
        return False, f"unexpected_shape_{data.shape}"

    return True, ""


def validate_annotations(annotations: List[BoundingBox]) -> Tuple[bool, str]:
    """Ensure bounding boxes contain positive dimensions."""
    for bbox in annotations:
        if bbox.width <= 0 or bbox.height <= 0:
            return False, "invalid_bbox_dimensions"
        if bbox.class_name not in config.CLASS_TO_ID:
            return False, f"unknown_class_{bbox.class_name}"
    return True, ""


def validate_sample(
    sample: RadarSample,
    require_annotations: bool = False,
    allow_resize: bool = True,
) -> ValidationResult:
    """Validate a loaded :class:`RadarSample`."""
    ok, reason = validate_radar_array(
        sample.radar_map,
        sample.modality,
        allow_resize=allow_resize,
    )
    if not ok:
        return ValidationResult(
            is_valid=False,
            sequence_id=sample.sequence_id,
            frame_id=sample.frame_id,
            reason=reason,
        )

    if require_annotations and not sample.annotations:
        return ValidationResult(
            is_valid=False,
            sequence_id=sample.sequence_id,
            frame_id=sample.frame_id,
            reason="missing_annotations",
        )

    ok, reason = validate_annotations(sample.annotations)
    if not ok:
        return ValidationResult(
            is_valid=False,
            sequence_id=sample.sequence_id,
            frame_id=sample.frame_id,
            reason=reason,
        )

    return ValidationResult(
        is_valid=True,
        sequence_id=sample.sequence_id,
        frame_id=sample.frame_id,
        sample=sample,
    )


def safe_load_frame(path: Path) -> Tuple[Optional[np.ndarray], str]:
    """
    Safely load a ``.npy`` radar frame, catching IO and parsing errors.

    Returns (array_or_none, error_reason).
    """
    if not path.exists():
        return None, "file_missing"

    try:
        data = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        logger.warning("Corrupted file %s: %s", path, exc)
        return None, "corrupted_file"

    return data, ""


def filter_valid_samples(
    samples: List[RadarSample],
    require_annotations: bool = False,
    allow_resize: bool = True,
) -> Tuple[List[RadarSample], ValidationReport]:
    """
    Filter a list of samples, returning only valid entries plus a report.
    """
    report = ValidationReport()
    valid_samples: List[RadarSample] = []

    for sample in samples:
        result = validate_sample(
            sample,
            require_annotations=require_annotations,
            allow_resize=allow_resize,
        )
        report.record(result)
        if result.is_valid and result.sample is not None:
            valid_samples.append(result.sample)

    return valid_samples, report
