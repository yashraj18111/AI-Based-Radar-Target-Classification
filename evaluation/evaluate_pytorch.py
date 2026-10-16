# evaluate_pytorch.py
# evaluates the trained model on test data and plots confusion matrix + ROC curves

import logging
from pathlib import Path

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


def evaluate_model(checkpoint_path=None, figure_dir=None):
    """Evaluates the trained model on the test split and saves metrics/plots."""
    if figure_dir is None:
        figure_dir = config.FIGURE_OUTPUT_DIR
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"

    classifier = load_neural_classifier(checkpoint_path)
    if classifier is None:
        raise FileNotFoundError(f"model checkpoint not found at {checkpoint_path}. Train model first!")

    X, y = load_all_spectrogram_data()

    # same 80/20 train/val split using seed so test indices match training setup
    np.random.seed(config.SEED)
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    test_idx = indices[split_idx:]

    X_test, y_test = X[test_idx], y[test_idx]

    print(f"evaluating model on test set ({len(X_test)} samples)...")

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

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

    # save plots
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


def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix heatmap with Seaborn."""
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
    print(f"saved confusion matrix to {output_path}")


def plot_roc_curves(y_true, y_prob, output_path):
    """Plot ROC curves for each target class."""
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
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title("ROC Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"saved ROC curves to {output_path}")


def plot_prediction_grid(images, y_true, y_pred, y_prob, output_path):
    """Plot sample spectrogram grid with predicted labels."""
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
    print(f"saved sample predictions grid to {output_path}")


def run_evaluation():
    """CLI helper."""
    return evaluate_model()
