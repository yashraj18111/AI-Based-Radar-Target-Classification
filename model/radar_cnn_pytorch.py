"""
PyTorch & High-Performance Neural Network Engine for Radar Target Classification.

Features a dual-engine architecture:
1. Native PyTorch 2D CNN model (`RadarTargetCNN`).
2. High-Performance Fallback Deep Neural Classifier (NumPy / SciPy / Scikit-Learn)
   if PyTorch DLL loading encounters OS-level version incompatibilities on Windows (e.g., Python 3.14).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPClassifier

import config

logger = logging.getLogger(__name__)

# Check if PyTorch native DLL loading is functional
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    # Test dummy tensor operation to verify C++ DLL sanity
    _dummy = torch.zeros(1)
    TORCH_AVAILABLE = True
    logger.info("Native PyTorch backend successfully loaded.")
except Exception as _err:
    logger.warning(f"PyTorch DLL loading fallback engaged ({_err}). Using High-Performance Neural Engine.")


if TORCH_AVAILABLE:

    class ConvBlock(nn.Module):
        """Dual Convolutional block with BatchNorm, LeakyReLU, and MaxPool."""

        def __init__(self, in_channels: int, out_channels: int, dropout_rate: float = 0.2):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.act(self.bn1(self.conv1(x)))
            out = self.act(self.bn2(self.conv2(out)))
            out = self.pool(out)
            out = self.dropout(out)
            return out

    class RadarTargetCNN(nn.Module):
        """2D Convolutional Neural Network for Radar Target Classification."""

        def __init__(
            self,
            in_channels: int = config.INPUT_CHANNELS,
            num_classes: int = config.NUM_CLASSES,
        ):
            super().__init__()
            self.block1 = ConvBlock(in_channels, 32, dropout_rate=0.15)
            self.block2 = ConvBlock(32, 64, dropout_rate=0.20)
            self.block3 = ConvBlock(64, 128, dropout_rate=0.25)

            self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
            self.bn4 = nn.BatchNorm2d(256)
            self.act4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

            self.fc1 = nn.Linear(256, 128)
            self.fc_bn = nn.BatchNorm1d(128)
            self.fc_dropout = nn.Dropout(p=0.4)
            self.fc2 = nn.Linear(128, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.act4(self.bn4(self.conv4(x)))
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = F.relu(self.fc_bn(self.fc1(x)))
            x = self.fc_dropout(x)
            return self.fc2(x)


class RadarNeuralClassifier:
    """
    High-Performance Deep Neural Network Classifier for Spectrogram Features.
    Serves as a robust backend for model training & real-time API inference.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64),
        max_iter: int = 200,
        random_state: int = config.TRAINING.seed,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=config.TRAINING.batch_size,
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=5,
            random_state=random_state,
            verbose=True,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> RadarNeuralClassifier:
        """Train the deep neural classifier on spectrogram features."""
        # Flatten (N, H, W, C) -> (N, H*W*C)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        self.model.fit(X_flat, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probability distribution."""
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        return self.model.predict_proba(X_flat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target class IDs."""
        if X.ndim == 3:
            X = np.expand_dims(X, axis=0)
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32) / 255.0
        return self.model.predict(X_flat)


def save_neural_classifier(
    classifier: Union[RadarNeuralClassifier, Any],
    filepath: Union[str, Path] = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl",
):
    """Save trained model instance to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved model classifier checkpoint to {path}.")


def load_neural_classifier(
    filepath: Union[str, Path] = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl",
) -> Optional[RadarNeuralClassifier]:
    """Load model classifier checkpoint from disk."""
    path = Path(filepath)
    if not path.is_file():
        return None
    with open(path, "rb") as f:
        classifier = pickle.load(f)
    logger.info(f"Loaded model classifier checkpoint from {path}.")
    return classifier
