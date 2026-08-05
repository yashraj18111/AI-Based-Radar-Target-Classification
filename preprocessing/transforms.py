"""
Radar tensor transforms: normalization, resize, and augmentation.

All transforms that modify spatial layout also update bounding-box
coordinates so annotations stay aligned with the radar map.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import zoom

import config
from preprocessing.loader import BoundingBox, RadarSample, normalize_radar_map

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

def resize_radar_map(
    radar_map: np.ndarray,
    target_shape: Tuple[int, int],
    interpolation: str = config.RESIZE_INTERPOLATION,
) -> np.ndarray:
    """
    Resize a 2-D radar map to ``target_shape`` (range, doppler/angle).
    """
    if radar_map.shape == target_shape:
        return radar_map.astype(np.float32)

    src_h, src_w = radar_map.shape[:2]
    tgt_h, tgt_w = target_shape
    order = 0 if interpolation == "nearest" else 1
    zoom_factors = (tgt_h / src_h, tgt_w / src_w)
    resized = zoom(radar_map.astype(np.float32), zoom_factors, order=order)
    return resized


def _scale_bbox(
    bbox: BoundingBox,
    scale_y: float,
    scale_x: float,
) -> BoundingBox:
    return BoundingBox(
        class_name=bbox.class_name,
        range_bin=bbox.range_bin * scale_y,
        doppler_bin=bbox.doppler_bin * scale_x,
        width=bbox.width * scale_x,
        height=bbox.height * scale_y,
    )


def resize_sample(
    sample: RadarSample,
    target_shape: Optional[Tuple[int, int]] = None,
) -> RadarSample:
    """Resize radar map and scale bounding boxes proportionally."""
    target = target_shape or config.get_target_shape(sample.modality)
    if target is None:
        return sample

    src_h, src_w = sample.radar_map.shape[:2]
    tgt_h, tgt_w = target
    if (src_h, src_w) == (tgt_h, tgt_w):
        return sample

    scale_y = tgt_h / src_h
    scale_x = tgt_w / src_w

    resized_map = resize_radar_map(sample.radar_map, target)
    resized_boxes = [_scale_bbox(b, scale_y, scale_x) for b in sample.annotations]

    updated = copy.copy(sample)
    updated.radar_map = resized_map
    updated.annotations = resized_boxes
    updated.metadata = {
        **sample.metadata,
        "resize_from": (src_h, src_w),
        "resize_to": (tgt_h, tgt_w),
    }
    return updated


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _flip_bboxes_horizontal(
    boxes: List[BoundingBox],
    width: int,
) -> List[BoundingBox]:
    flipped: List[BoundingBox] = []
    for bbox in boxes:
        flipped.append(
            BoundingBox(
                class_name=bbox.class_name,
                range_bin=bbox.range_bin,
                doppler_bin=(width - 1) - bbox.doppler_bin,
                width=bbox.width,
                height=bbox.height,
            )
        )
    return flipped


def horizontal_flip(sample: RadarSample) -> RadarSample:
    """Flip the Doppler/angle axis (horizontal flip of the heatmap)."""
    flipped_map = np.flip(sample.radar_map, axis=1).copy()
    _, width = sample.radar_map.shape[:2]

    updated = copy.copy(sample)
    updated.radar_map = flipped_map
    updated.annotations = _flip_bboxes_horizontal(sample.annotations, width)
    updated.metadata = {**sample.metadata, "augmentation": "horizontal_flip"}
    return updated


def add_gaussian_noise(
    sample: RadarSample,
    std: float = config.AUGMENTATION_CONFIG["gaussian_noise_std"],
) -> RadarSample:
    """Add small Gaussian noise to simulate sensor variation."""
    noisy = sample.radar_map + np.random.normal(0.0, std, sample.radar_map.shape)
    noisy = np.clip(noisy, 0.0, 1.0) if sample.radar_map.max() <= 1.0 else noisy

    updated = copy.copy(sample)
    updated.radar_map = noisy.astype(np.float32)
    updated.metadata = {**sample.metadata, "augmentation": "gaussian_noise"}
    return updated


def random_intensity_scale(
    sample: RadarSample,
    scale_range: Tuple[float, float] = tuple(
        config.AUGMENTATION_CONFIG["intensity_scale_range"]
    ),
) -> RadarSample:
    """Randomly scale intensity to improve robustness."""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    scaled = sample.radar_map * scale

    updated = copy.copy(sample)
    updated.radar_map = scaled.astype(np.float32)
    updated.metadata = {**sample.metadata, "augmentation": "intensity_scale"}
    return updated


def apply_augmentation(
    sample: RadarSample,
    aug_config: Optional[dict] = None,
) -> RadarSample:
    """
    Apply a random combination of augmentations to one sample.

    Augmentations are applied sequentially when their probability triggers.
    """
    cfg = aug_config or config.AUGMENTATION_CONFIG
    augmented = copy.copy(sample)

    if np.random.rand() < cfg.get("horizontal_flip_prob", 0.0):
        augmented = horizontal_flip(augmented)

    if np.random.rand() < 0.5:
        augmented = random_intensity_scale(augmented)

    if np.random.rand() < 0.3:
        augmented = add_gaussian_noise(augmented)

    return augmented


def maybe_augment_sample(
    sample: RadarSample,
    enabled: bool = config.AUGMENTATION_ENABLED,
) -> List[RadarSample]:
    """
    Return the original sample plus optional augmented copies.

    When enabled, saves the base sample and ``extra_augmented_copies``
    randomly augmented variants (train split only).
    """
    if not enabled:
        return [sample]

    cfg = config.AUGMENTATION_CONFIG
    extra = int(cfg.get("extra_augmented_copies", 1))
    results = [sample]

    for _ in range(max(extra, 0)):
        results.append(apply_augmentation(sample))

    return results


# ---------------------------------------------------------------------------
# Full transform pipeline (single sample)
# ---------------------------------------------------------------------------

def transform_sample(
    sample: RadarSample,
    normalize: bool = True,
    resize: bool = True,
    target_shape: Optional[Tuple[int, int]] = None,
) -> RadarSample:
    """
    Apply normalization and optional resize to a raw sample.

    Augmentation is intentionally excluded here; call :func:`maybe_augment_sample`
    separately so it can be gated by split (train only).
    """
    updated = copy.copy(sample)

    if normalize:
        updated.radar_map = normalize_radar_map(
            updated.radar_map,
            method=config.NORMALIZE_METHOD,
        )

    if resize:
        updated = resize_sample(updated, target_shape=target_shape)

    return updated
