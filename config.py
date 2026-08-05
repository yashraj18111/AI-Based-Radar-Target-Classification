"""
Central configuration for the AI-Based Radar Object Detection and Classification System.

All tunable parameters live here so experiments, paths, and model settings
can be changed in one place without touching pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
RAW_DATA_DIR: Path = DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATASET_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATASET_DIR / "sample_data"

OUTPUT_DIR: Path = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR: Path = OUTPUT_DIR / "models"
LOG_OUTPUT_DIR: Path = OUTPUT_DIR / "logs"
FIGURE_OUTPUT_DIR: Path = OUTPUT_DIR / "figures"


# ---------------------------------------------------------------------------
# Dataset selection & micro-Doppler target classes
# ---------------------------------------------------------------------------

# Supported values: "micro_doppler_csv" | "carrada" | "raddet"
DATASET_NAME: str = "micro_doppler_csv"

SPECTROGRAM_DIR: Path = PROCESSED_DATA_DIR / "spectrogram_images"

# Primary micro-Doppler CSV radar target classes
CLASS_NAMES: List[str] = ["car", "drone", "person"]
CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

# Mapping folder names to standardized class names
RAW_FOLDER_TO_CLASS: Dict[str, str] = {
    "Cars": "car",
    "Drones": "drone",
    "People": "person",
}

# ---------------------------------------------------------------------------
# Signal Processing & Spectrogram Image Settings
# ---------------------------------------------------------------------------

SPECTROGRAM_SHAPE: Tuple[int, int] = (128, 128)  # (height, width)
COLORMAP_NAME: str = "viridis"  # "viridis" | "magma" | "inferno" | "plasma"
INPUT_CHANNELS: int = 3  # RGB spectrogram image

# CARRADA range-doppler maps are 256 x 64 (range x doppler)
RD_MAP_SHAPE: Tuple[int, int] = (256, 64)
RA_MAP_SHAPE: Tuple[int, int] = (256, 256)


# ---------------------------------------------------------------------------
# Preprocessing & Normalization
# ---------------------------------------------------------------------------

NORMALIZE_METHOD: str = "log_minmax"  # "log_minmax" | "zscore" | "none"
LOG_EPSILON: float = 1e-6
CLIP_PERCENTILE: Tuple[float, float] = (1.0, 99.0)

AUGMENTATION_ENABLED: bool = True
AUGMENTATION_CONFIG: Dict[str, Any] = {
    "horizontal_flip_prob": 0.5,
    "gaussian_noise_std": 0.02,
    "intensity_scale_range": (0.9, 1.1),
    "extra_augmented_copies": 1,
}

# Target tensor shape after resize
TARGET_RD_SHAPE: Optional[Tuple[int, int]] = (128, 32)
TARGET_RA_SHAPE: Optional[Tuple[int, int]] = (128, 128)
RESIZE_INTERPOLATION: str = "linear"

MIN_VALID_STD: float = 1e-8
PREPROCESS_SPLITS: Tuple[str, ...] = ("train", "val", "test")

# Processed data layout
PROCESSED_MANIFEST_NAME: str = "manifest.json"
PROCESSED_CONFIG_NAME: str = "preprocessing_config.json"
PROCESSED_FRAME_DIR: str = "frames"
PROCESSED_LABEL_DIR: str = "labels"
PROCESSED_FILE_FORMAT: str = "png"


# ---------------------------------------------------------------------------
# DataLoader behaviour
# ---------------------------------------------------------------------------

DEFAULT_SPLIT: str = "train"
DEFAULT_MODALITY: str = "range_doppler_numpy"
FRAME_EXTENSION: str = ".npy"
NUM_WORKERS: int = 2
SHUFFLE: bool = True
BATCH_SIZE: int = 32


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

VIS_COLORMAP: str = "viridis"
VIS_DPI: int = 150
VIS_FIGSIZE: Tuple[float, float] = (10.0, 6.0)
CLASS_COLORS: Dict[str, str] = {
    "car": "#FF4444",
    "drone": "#44AAFF",
    "person": "#44FF44",
}


# ---------------------------------------------------------------------------
# PyTorch Model Training Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for PyTorch model training."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    checkpoint_name: str = "best_radar_model.pth"
    seed: int = 42
    device: str = "cuda"  # auto-falls back to cpu if cuda unavailable


TRAINING: TrainingConfig = TrainingConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset_root() -> Path:
    """Return the raw dataset root directory for the active dataset."""
    if DATASET_NAME == "carrada":
        return CARRADA_RAW_DIR
    if DATASET_NAME == "raddet":
        return RADDET_RAW_DIR
    raise ValueError(f"Unsupported dataset: {DATASET_NAME!r}")


def get_target_shape(modality: str) -> Optional[Tuple[int, int]]:
    """Return the configured resize target for a radar modality."""
    if "angle" in modality:
        return TARGET_RA_SHAPE
    return TARGET_RD_SHAPE


def get_split_sequences(split: str) -> List[str]:
    """Return sequence identifiers for the given split."""
    if DATASET_NAME == "carrada":
        if split not in CARRADA_SPLITS:
            raise ValueError(
                f"Unknown split {split!r}. Choose from {list(CARRADA_SPLITS)}"
            )
        return CARRADA_SPLITS[split]
    raise NotImplementedError(f"Split lookup not implemented for {DATASET_NAME!r}")
