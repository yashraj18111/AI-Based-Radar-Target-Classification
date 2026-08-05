"""
Signal Processing & Spectrogram Image Generation Engine.

Converts raw CSV micro-Doppler radar signal matrices into normalized,
color-mapped spectrogram images for visual inspection and deep learning input.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.cm as cm
import numpy as np
import pandas as pd
from PIL import Image

import config

logger = logging.getLogger(__name__)


def read_raw_csv_signal(csv_path: Union[str, Path]) -> np.ndarray:
    """
    Read a raw radar signal CSV file into a 2D float32 NumPy array.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing radar signal amplitude/power matrix.

    Returns
    -------
    np.ndarray
        2D float32 array representing the raw radar signal.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Radar CSV signal file not found: {path}")

    try:
        # Load CSV ignoring trailing empty columns if any
        df = pd.read_csv(path, header=None).dropna(how="all", axis=1)
        matrix = df.to_numpy(dtype=np.float32)
        if matrix.ndim != 2 or matrix.size == 0:
            raise ValueError(f"Invalid matrix dimensions: {matrix.shape}")
        return matrix
    except Exception as exc:
        logger.warning(f"Error reading {path} via pandas: {exc}. Retrying with np.genfromtxt...")
        matrix = np.genfromtxt(path, delimiter=",", dtype=np.float32)
        # Drop NaN columns
        if np.isnan(matrix).any():
            matrix = np.nan_to_num(matrix, nan=0.0)
        return matrix


def convert_signal_to_spectrogram_image(
    signal_matrix: np.ndarray,
    target_shape: Tuple[int, int] = config.SPECTROGRAM_SHAPE,
    colormap_name: str = config.COLORMAP_NAME,
) -> np.ndarray:
    """
    Transform a 2D radar signal matrix into a normalized RGB Spectrogram Image.

    Steps:
    1. Log-magnitude dB scaling: P_dB = 10 * log10(|S| + epsilon)
    2. Min-Max percentile normalization to [0, 1] range.
    3. Apply color-map (e.g., Viridis, Magma, Inferno) to obtain 3-channel RGB image.
    4. Resize to target dimension (e.g., 128x128).

    Parameters
    ----------
    signal_matrix : np.ndarray
        2D radar signal array.
    target_shape : Tuple[int, int]
        (height, width) for output spectrogram image.
    colormap_name : str
        Matplotlib colormap identifier ('viridis', 'magma', etc.).

    Returns
    -------
    np.ndarray
        3-D uint8 array of shape (height, width, 3) representing the RGB spectrogram image.
    """
    data = signal_matrix.astype(np.float32)

    # 1. Log-magnitude scaling if values are non-negative or wide dynamic range
    min_val = np.min(data)
    if min_val >= 0:
        data = 10.0 * np.log10(data + config.LOG_EPSILON)

    # 2. Percentile clipping to suppress extreme noise spikes
    p_low, p_high = config.CLIP_PERCENTILE
    vmin, vmax = np.percentile(data, p_low), np.percentile(data, p_high)
    if vmax > vmin:
        data = np.clip(data, vmin, vmax)
        normalized = (data - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(data)

    # 3. Apply Colormap -> float array in [0, 1] with shape (H, W, 4)
    colormap_fn = getattr(cm, colormap_name, cm.viridis)
    rgb_float = colormap_fn(normalized)[:, :, :3]  # Strip alpha channel

    # Convert to uint8 RGB Image (H, W, 3)
    rgb_uint8 = (rgb_float * 255.0).astype(np.uint8)

    # 4. Resize Image using PIL bicubic interpolation
    pil_img = Image.fromarray(rgb_uint8, mode="RGB")
    h, w = target_shape
    pil_resized = pil_img.resize((w, h), resample=Image.BICUBIC)

    return np.array(pil_resized, dtype=np.uint8)


def save_spectrogram_image(
    spectrogram_rgb: np.ndarray,
    output_path: Union[str, Path],
) -> Path:
    """Save an RGB uint8 array to a PNG image file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(spectrogram_rgb, mode="RGB")
    img.save(path, format="PNG")
    return path


def process_all_raw_signals(
    raw_dir: Union[str, Path] = config.RAW_DATA_DIR,
    output_dir: Union[str, Path] = config.SPECTROGRAM_DIR,
    max_samples_per_class: Optional[int] = None,
) -> List[Dict[str, Union[str, int]]]:
    """
    Scan raw dataset directories (Cars, Drones, People), generate spectrogram images,
    and cache them to disk with a metadata manifest.

    Parameters
    ----------
    raw_dir : Path
        Root directory containing raw CSV signal files.
    output_dir : Path
        Target directory to save generated PNG spectrogram images.
    max_samples_per_class : Optional[int]
        Optional cap on samples per class (useful for fast development/testing).

    Returns
    -------
    List[Dict[str, Union[str, int]]]
        List of manifest entries describing every generated sample.
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, Union[str, int]]] = []

    for folder_name, class_name in config.RAW_FOLDER_TO_CLASS.items():
        class_id = config.CLASS_TO_ID[class_name]
        class_output_dir = out_path / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)

        # Look for CSV files in raw_path/folder_name and raw_path/data/folder_name
        search_paths = [
            raw_path / folder_name,
            raw_path / "data" / folder_name,
        ]

        csv_files: List[Path] = []
        for p in search_paths:
            if p.exists():
                csv_files.extend(p.rglob("*.csv"))

        logger.info(f"Discovered {len(csv_files)} CSV signal files for class '{class_name}'.")

        if max_samples_per_class is not None and len(csv_files) > max_samples_per_class:
            csv_files = csv_files[:max_samples_per_class]

        for idx, csv_file in enumerate(csv_files):
            try:
                # Derive unique filename based on session subfolder
                rel_parts = csv_file.relative_to(raw_path).parts
                sample_name = "_".join(rel_parts).replace(".csv", ".png")
                png_path = class_output_dir / sample_name

                # Check if already processed
                if not png_path.exists():
                    raw_signal = read_raw_csv_signal(csv_file)
                    spectrogram_rgb = convert_signal_to_spectrogram_image(raw_signal)
                    save_spectrogram_image(spectrogram_rgb, png_path)

                manifest.append(
                    {
                        "sample_id": f"{class_name}_{idx:05d}",
                        "class_name": class_name,
                        "class_id": class_id,
                        "csv_path": str(csv_file),
                        "image_path": str(png_path),
                    }
                )

            except Exception as err:
                logger.error(f"Failed to process signal file {csv_file}: {err}")

    # Save manifest JSON
    manifest_path = out_path.parent / config.PROCESSED_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Successfully processed {len(manifest)} radar signal samples. Manifest saved to {manifest_path}.")
    return manifest
