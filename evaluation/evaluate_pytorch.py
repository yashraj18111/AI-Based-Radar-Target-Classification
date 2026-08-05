"""
Model Diagnostic Evaluation and Report Generator.

Evaluates the trained model on test spectrogram data, computes multi-class metrics,
and renders publication-quality Seaborn Confusion Matrix, ROC curves, and sample prediction grids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

import config
from model.radar_cnn_pytorch import load_neural_classifier
from training.train_pytorch import load_all_spectrogram_data

logger = logging.getLogger(__name__)


def evaluate_model(
    checkpoint_path: Optional[Union[str, Path]] = None,
    figure_dir: Path = config.FIGURE_OUTPUT_DIR,
) -> Dict[str, Union[float, Dict]]:
    """Evaluate trained model on test data split and save visual reports."""
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"

    classifier = load_neural_classifier(checkpoint_path)
    if classifier is None:
        raise FileNotFoundError(f"Classifier checkpoint not found at {checkpoint_path}. Train model first.")

    X, y = load_all_spectrogram_data()

    # Stratified split to evaluate on test set
    np.random.seed(config.TRAINING.seed)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    test_idx = indices[split_idx:]

    X_test, y_test = X[test_idx], y[test_idx]

    logger.info(f"Evaluating model on test set ({len(X_test)} samples)...")

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

    # Calculate overall metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_str = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES)
    print("\n=======================================================")
    print("                 MODEL TEST EVALUATION REPORT           ")
    print("=======================================================")
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Precision        : {prec:.4f}")
    print(f"Recall           : {rec:.4f}")
    print(f"F1-Score         : {f1:.4f}")
    print("\nPer-Class Classification Report:\n")
    print(report_str)
    print("=======================================================\n")

    # Generate Figures
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, figure_dir / "confusion_matrix.png")
    plot_roc_curves(y_test, y_prob, figure_dir / "roc_curves.png")
    plot_prediction_grid(X_test, y_test, y_pred, y_prob, figure_dir / "spectrogram_predictions.png")

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "report": report_str,
    }
    return results


def plot_confusion_matrix(cm: np.ndarray, output_path: Path):
    """Plot Seaborn confusion matrix heatmap."""
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cbar=True,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Radar Target Classification - Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved confusion matrix figure to {output_path}.")


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, output_path: Path):
    """Plot multi-class ROC curves."""
    plt.figure(figsize=(8, 6))

    colors = ["#FF4444", "#44AAFF", "#44FF44"]
    for i, class_name in enumerate(config.CLASS_NAMES):
        y_true_binary = (y_true == i).astype(int)
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(
                fpr,
                tpr,
                color=colors[i % len(colors)],
                lw=2,
                label=f"{class_name.capitalize()} (AUC = {roc_auc:.3f})",
            )

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("Multi-Class Receiver Operating Characteristic (ROC) Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved ROC curves figure to {output_path}.")


def plot_prediction_grid(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path,
):
    """Plot a grid of spectrogram images with prediction labels."""
    if len(images) == 0:
        return

    n_samples = min(9, len(images))
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for idx in range(9):
        ax = axes[idx // 3, idx % 3]
        if idx < n_samples:
            img = images[idx]
            gt_label = config.CLASS_NAMES[y_true[idx]]
            pred_label = config.CLASS_NAMES[y_pred[idx]]
            conf = y_prob[idx, y_pred[idx]] * 100.0

            color = "green" if gt_label == pred_label else "red"

            ax.imshow(img)
            ax.set_title(
                f"True: {gt_label.capitalize()}\nPred: {pred_label.capitalize()} ({conf:.1f}%)",
                fontsize=10,
                color=color,
                fontweight="bold",
            )
        ax.axis("off")

    plt.suptitle("Sample Spectrogram Target Predictions", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved spectrogram prediction grid figure to {output_path}.")


def run_evaluation() -> Dict[str, Union[float, Dict]]:
    """CLI helper to run evaluation."""
    return evaluate_model()
