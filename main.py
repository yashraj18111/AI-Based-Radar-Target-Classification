"""
Main Entry Point for AI-Based Radar Target Classification System.

Provides CLI commands for signal conversion, PyTorch model training, evaluation,
and starting the FastAPI web application server.
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import config

# ---------------------------------------------------------------------------
# GitHub Releases — pre-trained model download
# ---------------------------------------------------------------------------

# Update this URL after creating the GitHub Release and uploading the model file.
# Format: https://github.com/<user>/<repo>/releases/download/<tag>/<filename>
MODEL_DOWNLOAD_URL = (
    "https://github.com/yashraj18111/AI-Based-Radar-Target-Classification"
    "/releases/download/v1.0.0/best_radar_model.pkl"
)

MODEL_PATH = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"


def download_model_if_missing() -> None:
    """Auto-download the pre-trained model from GitHub Releases if not found locally."""
    if MODEL_PATH.exists():
        return  # Already present, skip download

    logger.info("Pre-trained model not found locally. Downloading from GitHub Releases...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  Downloading model... {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH, reporthook=_progress)
        print()  # newline after progress
        logger.info(f"Model downloaded successfully to {MODEL_PATH}")
    except Exception as exc:
        logger.error(
            f"Failed to download pre-trained model: {exc}\n"
            f"  You can manually download it from:\n  {MODEL_DOWNLOAD_URL}\n"
            f"  and place it at: {MODEL_PATH}"
        )
        sys.exit(1)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("main")


def command_convert(max_samples: int = None):
    """Convert raw CSV radar signals into Spectrogram RGB images."""
    logger.info("--- Phase 1: Radar Signal-to-Spectrogram Conversion ---")
    from preprocessing.signal_to_image import process_all_raw_signals

    manifest = process_all_raw_signals(max_samples_per_class=max_samples)
    print(f"\n[SUCCESS] Processed {len(manifest)} radar CSV files into spectrogram images.")


def command_train(epochs: int = config.TRAINING.epochs):
    """Train PyTorch RadarTargetCNN model."""
    logger.info("--- Phase 2: PyTorch Model Training ---")
    from training.train_pytorch import run_training

    trainer = run_training(epochs=epochs)
    print(f"\n[SUCCESS] Training completed over {epochs} epochs. Best model checkpoint saved to {config.MODEL_OUTPUT_DIR}.")


def command_evaluate():
    """Evaluate trained PyTorch model on test data split."""
    logger.info("--- Phase 3: Model Diagnostic Evaluation ---")
    download_model_if_missing()  # Auto-download pre-trained model if needed
    from evaluation.evaluate_pytorch import run_evaluation

    results = run_evaluation()
    print(f"\n[SUCCESS] Test Accuracy: {results['accuracy'] * 100:.2f}% | F1-Score: {results['f1_score']:.4f}")
    print(f"Evaluation figures saved to {config.FIGURE_OUTPUT_DIR}.")


def command_serve(host: str = "127.0.0.1", port: int = 8000):
    """Launch FastAPI web application server."""
    logger.info("--- Phase 4: Launching FastAPI Web Application ---")
    download_model_if_missing()  # Auto-download pre-trained model if needed
    import uvicorn

    print(f"\n[SERVER ONLINE] Starting RadarSense AI Web Dashboard at http://{host}:{port}")
    uvicorn.run("app.api:app", host=host, port=port, reload=False)


def command_run_all(epochs: int = 10, max_samples: int = 500):
    """Run end-to-end pipeline: Convert -> Train -> Evaluate."""
    logger.info("--- Running Full End-to-End Radar Classification Pipeline ---")
    command_convert(max_samples=max_samples)
    command_train(epochs=epochs)
    command_evaluate()  # will skip download since model was just trained
    print("\n[PIPELINE COMPLETE] System ready. Launching FastAPI Server...")
    command_serve()


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Radar Target Classification & Micro-Doppler Visualizer System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Convert
    parser_convert = subparsers.add_parser("convert", help="Convert raw CSV signals to spectrogram images")
    parser_convert.add_argument("--max-samples", type=int, default=None, help="Cap samples per class")

    # Train
    parser_train = subparsers.add_parser("train", help="Train PyTorch CNN classifier")
    parser_train.add_argument("--epochs", type=int, default=config.TRAINING.epochs, help="Number of training epochs")

    # Evaluate
    subparsers.add_parser("evaluate", help="Evaluate model and generate figures")

    # Serve
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI web server")
    parser_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port number")

    # Run All
    parser_all = subparsers.add_parser("run-all", help="Run full pipeline end-to-end")
    parser_all.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser_all.add_argument("--max-samples", type=int, default=500, help="Cap samples per class for training")

    args = parser.parse_args()

    if args.command == "convert":
        command_convert(max_samples=args.max_samples)
    elif args.command == "train":
        command_train(epochs=args.epochs)
    elif args.command == "evaluate":
        command_evaluate()
    elif args.command == "serve":
        command_serve(host=args.host, port=args.port)
    elif args.command == "run-all":
        command_run_all(epochs=args.epochs, max_samples=args.max_samples)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
