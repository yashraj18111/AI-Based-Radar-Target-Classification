# ml_engine/predict.py
# Command-line interface for Node.js Express backend to run radar classification inference

import os
import sys
import json
import base64
import io
from pathlib import Path

# Add project root to sys.path so config and modules load cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

import config
from model.radar_cnn_pytorch import load_neural_classifier
from preprocessing.signal_to_image import convert_signal_to_spectrogram_image, read_raw_csv_signal

import urllib.request

MODEL_DOWNLOAD_URL = (
    "https://github.com/yashraj18111/AI-Based-Radar-Target-Classification"
    "/releases/download/v1.0.0/best_radar_model.pkl"
)

def ensure_model_exists(model_path: Path):
    if not model_path.exists():
        print(f"Model not found at {model_path}, auto-downloading from GitHub release...", file=sys.stderr)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, model_path)
            print(f"Downloaded model to {model_path}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to auto-download model: {e}", file=sys.stderr)

_classifier_instance = None

def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        model_path = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"
        ensure_model_exists(model_path)
        _classifier_instance = load_neural_classifier(model_path)
    return _classifier_instance

def run_prediction_on_file(file_path: str):
    path = Path(file_path)
    if not path.is_file():
        return {"success": False, "error": f"File not found: {file_path}"}

    try:
        df = pd.read_csv(path, header=None).dropna(how="all", axis=1)
        signal_matrix = df.to_numpy(dtype=np.float32)

        if signal_matrix.size == 0:
            return {"success": False, "error": "CSV file contains no numeric data"}

        # 1. Convert signal matrix to 128x128 Viridis spectrogram RGB image
        spectrogram_rgb = convert_signal_to_spectrogram_image(signal_matrix)

        # 2. Encode to base64 PNG data URL
        pil_img = Image.fromarray(spectrogram_rgb, mode="RGB")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{img_base64}"

        # 3. Model inference
        classifier = get_classifier()
        if classifier is None:
            return {"success": False, "error": "Trained model checkpoint could not be loaded"}

        probs = classifier.predict_proba(spectrogram_rgb)[0]
        pred_id = int(np.argmax(probs))
        pred_class = config.CLASS_NAMES[pred_id]
        confidence = float(probs[pred_id] * 100.0)

        probabilities = {
            cls_name: round(float(probs[idx] * 100.0), 2)
            for idx, cls_name in enumerate(config.CLASS_NAMES)
        }

        signal_stats = {
            "matrix_shape": list(signal_matrix.shape),
            "min_power": round(float(np.min(signal_matrix)), 2),
            "max_power": round(float(np.max(signal_matrix)), 2),
            "mean_power": round(float(np.mean(signal_matrix)), 2),
        }

        return {
            "success": True,
            "filename": path.name,
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "spectrogram_image": image_data_url,
            "signal_stats": signal_stats,
        }

    except Exception as exc:
        return {"success": False, "error": str(exc)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "Usage: python predict.py <path_to_csv>"}))
        sys.exit(1)

    csv_file = sys.argv[1]
    result = run_prediction_on_file(csv_file)
    print(json.dumps(result))
