# train_pytorch.py
# loads spectrogram images from manifest.json, trains the classifier, saves the model

import json
import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from model.radar_cnn_pytorch import RadarNeuralClassifier, save_neural_classifier

logger = logging.getLogger(__name__)


def load_all_spectrogram_data(manifest_path=None):
    """Read all the spectrogram PNGs and their labels from the manifest file."""
    if manifest_path is None:
        manifest_path = config.PROCESSED_DATA_DIR / config.PROCESSED_MANIFEST_NAME

    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"manifest not found at {path} - run 'python main.py convert' first"
        )

    with open(path, "r") as f:
        samples = json.load(f)

    images = []
    labels = []
    h, w = config.SPECTROGRAM_SHAPE

    for item in samples:
        img_path = Path(item["image_path"])
        if img_path.is_file():
            img = Image.open(img_path).convert("RGB").resize((w, h))
            images.append(np.array(img, dtype=np.uint8))
            labels.append(int(item["class_id"]))

    X = np.array(images, dtype=np.uint8)
    y = np.array(labels, dtype=np.int64)
    return X, y


class Trainer:
    """Handles training and saving the model."""

    def __init__(self, checkpoint_dir=None, figure_dir=None):
        self.checkpoint_dir = Path(checkpoint_dir or config.MODEL_OUTPUT_DIR)
        self.figure_dir     = Path(figure_dir     or config.FIGURE_OUTPUT_DIR)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        self.classifier = RadarNeuralClassifier(
            max_iter=config.EPOCHS * 10,
            random_state=config.SEED,
        )

    def fit(self, epochs=config.EPOCHS):
        """Run training."""
        print("loading spectrogram images from manifest...")
        X, y = load_all_spectrogram_data()
        print(f"loaded {len(X)} images, classes: {config.CLASS_NAMES}")

        # 80/20 train/val split
        # thought about doing 70/15/15 but with this dataset size giving the model more
        # training data mattered more. using random permutation (not sequential) so that
        # class ordering in the folder doesn't bias the split
        np.random.seed(config.SEED)
        indices   = np.random.permutation(len(X))
        split_idx = int(len(X) * 0.8)

        train_idx, val_idx = indices[:split_idx], indices[split_idx:]
        X_train, y_train   = X[train_idx], y[train_idx]
        X_val,   y_val     = X[val_idx],   y[val_idx]

        print(f"train: {len(X_train)} samples | val: {len(X_val)} samples")

        start = time.time()
        self.classifier.fit(X_train, y_train)
        print(f"training done in {time.time() - start:.1f}s")

        # save the trained model
        save_path = self.checkpoint_dir / "best_radar_model.pkl"
        save_neural_classifier(self.classifier, save_path)
        print(f"model saved to {save_path}")

        # sklearn stores loss and validation score history internally after training
        # we can just pull those out and plot them, no need for a custom training loop
        loss_curve  = getattr(self.classifier.model, "loss_curve_", [])
        val_scores  = getattr(self.classifier.model, "validation_scores_", [])
        self.plot_training_curves(loss_curve, val_scores)

        return self.classifier

    def plot_training_curves(self, loss_curve, val_scores):
        """Plot training loss and validation accuracy and save as PNG."""
        output_path = self.figure_dir / "training_curves.png"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        if loss_curve:
            ax1.plot(loss_curve, color="#1f77b4", linewidth=2, label="Training Loss")
            ax1.set_title("Loss over training", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if val_scores:
            val_acc = [s * 100.0 for s in val_scores]
            ax2.plot(val_acc, color="#2ca02c", linewidth=2, label="Validation Accuracy (%)")
            ax2.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Evaluation step")
            ax2.set_ylabel("Accuracy (%)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"training curves saved to {output_path}")


def run_training(epochs=config.EPOCHS):
    """Entry point called by main.py."""
    trainer = Trainer()
    trainer.fit(epochs=epochs)
    return trainer
