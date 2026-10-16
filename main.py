# main.py
# main CLI entry point for the project
# allows running signal conversion, training, evaluation, or starting the API server

import argparse
import logging
import sys
import urllib.request
from pathlib import Path

import config

MODEL_DOWNLOAD_URL = (
    "https://github.com/yashraj18111/AI-Based-Radar-Target-Classification"
    "/releases/download/v1.0.0/best_radar_model.pkl"
)

MODEL_PATH = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"


def download_model_if_missing():
    """Auto-download pre-trained model if not found locally."""
    if MODEL_PATH.exists():
        return

    print("pre-trained model not found locally, downloading...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(downloaded / total_size * 100, 100)
                print(f"\r  downloading model... {pct:.1f}%", end="", flush=True)

        urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, MODEL_PATH, reporthook=_progress)
        print()
        print(f"model downloaded to {MODEL_PATH}")
    except Exception as exc:
        print(f"could not download model automatically: {exc}")
        print(f"you can download manually from {MODEL_DOWNLOAD_URL}")
        sys.exit(1)


# basic logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("main")


def command_convert(max_samples=None):
    """Convert raw CSV radar signals into spectrogram PNG images."""
    print("--- Phase 1: Radar Signal-to-Spectrogram Conversion ---")
    from preprocessing.signal_to_image import process_all_raw_signals

    manifest = process_all_raw_signals(max_samples_per_class=max_samples)
    print(f"\n[SUCCESS] Processed {len(manifest)} radar CSV files into spectrogram images.")


def command_train(epochs=config.EPOCHS):
    """Train the model."""
    print("--- Phase 2: PyTorch Model Training ---")
    from training.train_pytorch import run_training

    trainer = run_training(epochs=epochs)
    print(f"\n[SUCCESS] Training completed. Model saved to {config.MODEL_OUTPUT_DIR}.")


def command_evaluate():
    """Evaluate trained model on test split."""
    print("--- Phase 3: Model Diagnostic Evaluation ---")
    download_model_if_missing()
    from evaluation.evaluate_pytorch import run_evaluation

    results = run_evaluation()
    print(f"\n[SUCCESS] Test Accuracy: {results['accuracy'] * 100:.2f}% | F1-Score: {results['f1_score']:.4f}")
    print(f"evaluation figures saved to {config.FIGURE_OUTPUT_DIR}.")


def command_serve(host="127.0.0.1", port=8000):
    """Launch FastAPI web server."""
    print("--- Phase 4: Launching FastAPI Web Application ---")
    download_model_if_missing()
    import uvicorn

    print(f"\n[SERVER ONLINE] Starting API server at http://{host}:{port}")
    uvicorn.run("app.api:app", host=host, port=port, reload=False)


def command_run_all(epochs=10, max_samples=500):
    """Run full pipeline: Convert -> Train -> Evaluate."""
    print("--- Running Full End-to-End Radar Classification Pipeline ---")
    command_convert(max_samples=max_samples)
    command_train(epochs=epochs)
    command_evaluate()
    print("\n[PIPELINE COMPLETE] System ready. Launching FastAPI Server...")
    command_serve()


def main():
    parser = argparse.ArgumentParser(
        description="AI-Based Radar Target Classification System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # convert
    parser_convert = subparsers.add_parser("convert", help="Convert raw CSV signals to spectrogram images")
    parser_convert.add_argument("--max-samples", type=int, default=None, help="Cap samples per class")

    # train
    parser_train = subparsers.add_parser("train", help="Train PyTorch CNN classifier")
    parser_train.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model and generate figures")

    # serve
    parser_serve = subparsers.add_parser("serve", help="Launch FastAPI web server")
    parser_serve.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser_serve.add_argument("--port", type=int, default=8000, help="Port number")

    # run-all
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
