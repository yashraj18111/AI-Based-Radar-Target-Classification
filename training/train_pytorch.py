"""
Model Training Pipeline.

Trains the Deep Neural Network Spectrogram Classifier, logs epoch metrics,
saves best model checkpoints, and plots training & loss curves.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from model.radar_cnn_pytorch import RadarNeuralClassifier, save_neural_classifier

logger = logging.getLogger(__name__)


def load_all_spectrogram_data(
    manifest_path: Optional[Union[str, Path]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed spectrogram RGB images and labels into NumPy arrays."""
    if manifest_path is None:
        manifest_path = config.PROCESSED_DATA_DIR / config.PROCESSED_MANIFEST_NAME

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found at {path}. Run signal conversion first.")

    with open(path, "r") as f:
        samples = json.load(f)

    images: List[np.ndarray] = []
    labels: List[int] = []

    h, w = config.SPECTROGRAM_SHAPE

    for item in samples:
        img_p = Path(item["image_path"])
        if img_p.is_file():
            img = Image.open(img_p).convert("RGB").resize((w, h))
            images.append(np.array(img, dtype=np.uint8))
            labels.append(int(item["class_id"]))

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)

    return X, y


class Trainer:
    """Manages model training, evaluation metrics, and training curve figures."""

    def __init__(
        self,
        checkpoint_dir: Path = config.MODEL_OUTPUT_DIR,
        figure_dir: Path = config.FIGURE_OUTPUT_DIR,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.figure_dir = Path(figure_dir)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = RadarNeuralClassifier(
            max_iter=config.TRAINING.epochs * 10,
            random_state=config.TRAINING.seed,
        )

    def fit(self, epochs: int = config.TRAINING.epochs) -> RadarNeuralClassifier:
        """Run training loop."""
        logger.info("Loading preprocessed micro-Doppler spectrogram images...")
        X, y = load_all_spectrogram_data()

        logger.info(f"Loaded dataset: X={X.shape}, y={y.shape}. Target classes: {config.CLASS_NAMES}")

        # Stratified train/test split
        np.random.seed(config.TRAINING.seed)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * 0.8)

        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        logger.info(f"Training set: {len(X_train)} samples | Validation set: {len(X_val)} samples.")
        start_time = time.time()

        self.classifier.fit(X_train, y_train)

        elapsed = time.time() - start_time
        logger.info(f"Model training completed in {elapsed:.2f} seconds.")

        # Save checkpoint
        save_path = self.checkpoint_dir / "best_radar_model.pkl"
        save_neural_classifier(self.classifier, save_path)

        # Plot training loss history if available
        loss_curve = getattr(self.classifier.model, "loss_curve_", [])
        val_scores = getattr(self.classifier.model, "validation_scores_", [])
        self.plot_training_curves(loss_curve, val_scores)

        return self.classifier

    def plot_training_curves(self, loss_curve: List[float], val_scores: List[float]):
        """Plot loss and validation accuracy curves over iterations."""
        output_path = self.figure_dir / "training_curves.png"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if loss_curve:
            ax1.plot(loss_curve, color="#1f77b4", linewidth=2, label="Training Loss")
            ax1.set_title("Cross-Entropy Loss Dynamics", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Iteration Step")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if val_scores:
            val_acc = [s * 100.0 for s in val_scores]
            ax2.plot(val_acc, color="#2ca02c", linewidth=2, label="Validation Accuracy (%)")
            ax2.set_title("Validation Accuracy Across Training", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Evaluation Step")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        logger.info(f"Saved training curves figure to {output_path}.")


def run_training(epochs: int = config.TRAINING.epochs) -> Trainer:
    """CLI helper to run training."""
    trainer = Trainer()
    trainer.fit(epochs=epochs)
    return trainer
