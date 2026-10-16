# api.py
# FastAPI server so web apps or frontend clients can upload radar CSVs
# and get back the predicted target class + generated spectrogram image

import base64
import io
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import config
from model.radar_cnn_pytorch import RadarNeuralClassifier, load_neural_classifier
from preprocessing.signal_to_image import convert_signal_to_spectrogram_image, read_raw_csv_signal

logger = logging.getLogger(__name__)

# initialize FastAPI App
app = FastAPI(
    title="AI-Based Radar Target Classification API",
    description="Micro-Doppler Radar Signal to Spectrogram Image Generation and Target Classification",
    version="1.0.0",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# global model reference - lazy loading so server starts instantly without waiting for pkl load
model_instance = None


def get_active_classifier():
    """Lazy load the trained model checkpoint when first request arrives."""
    global model_instance
    if model_instance is None:
        ckpt_path = config.MODEL_OUTPUT_DIR / "best_radar_model.pkl"
        classifier = load_neural_classifier(ckpt_path)
        if classifier is not None:
            model_instance = classifier
            print(f"loaded model from {ckpt_path}")
        else:
            print("no saved model found, creating fresh model instance...")
            model_instance = RadarNeuralClassifier()

    return model_instance


@app.get("/api/health")
def health_check():
    """Simple status check."""
    return {
        "status": "online",
        "target_classes": config.CLASS_NAMES,
        "model_loaded": model_instance is not None or (config.MODEL_OUTPUT_DIR / "best_radar_model.pkl").exists(),
    }


@app.get("/api/info")
def system_info():
    """Dataset & target class metadata for frontend displays."""
    return {
        "dataset_name": config.DATASET_NAME,
        "target_classes": config.CLASS_NAMES,
        "class_colors": config.CLASS_COLORS,
        "spectrogram_shape": config.SPECTROGRAM_SHAPE,
        "colormap": config.COLORMAP_NAME,
    }


@app.post("/api/predict")
async def predict_radar_signal(file: UploadFile = File(...)):
    """
    Accepts uploaded radar signal CSV file:
    1. Reads matrix
    2. Converts to spectrogram image
    3. Encodes to base64 PNG so frontend can display it immediately
    4. Runs inference and returns prediction + probability scores
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload a radar CSV signal file."
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), header=None).dropna(how="all", axis=1)
        signal_matrix = df.to_numpy(dtype=np.float32)

        if signal_matrix.size == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        # 1. signal -> spectrogram image
        spectrogram_rgb = convert_signal_to_spectrogram_image(signal_matrix)

        # 2. encode RGB image to base64 PNG data URL
        pil_img = Image.fromarray(spectrogram_rgb, mode="RGB")
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_data_url = f"data:image/png;base64,{img_base64}"

        # 3. model prediction
        classifier = get_active_classifier()

        if not hasattr(classifier.model, "classes_"):
            from training.train_pytorch import Trainer
            print("fitting model before running inference...")
            trainer = Trainer()
            classifier = trainer.fit(epochs=2)

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
            "min_power": float(np.min(signal_matrix)),
            "max_power": float(np.max(signal_matrix)),
            "mean_power": float(np.mean(signal_matrix)),
        }

        return {
            "success": True,
            "filename": file.filename,
            "predicted_class": pred_class,
            "confidence": round(confidence, 2),
            "probabilities": probabilities,
            "spectrogram_image": image_data_url,
            "signal_stats": signal_stats,
        }

    except Exception as err:
        logger.error(f"Error during prediction for {file.filename}: {err}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")


@app.get("/api/samples")
def get_sample_presets():
    """Return pre-configured sample CSV files for frontend testing."""
    samples = []
    raw_dir = config.RAW_DATA_DIR

    for class_folder, class_name in config.RAW_FOLDER_TO_CLASS.items():
        found = list((raw_dir / class_folder).rglob("*.csv"))
        if not found:
            found = list((raw_dir / "data" / class_folder).rglob("*.csv"))

        if found:
            sample_file = found[0]
            samples.append(
                {
                    "class_name": class_name,
                    "label": f"Sample {class_name.capitalize()} Signal",
                    "path": str(sample_file),
                    "filename": sample_file.name,
                }
            )

    return samples


@app.get("/api/sample-content/{class_name}")
def get_sample_content(class_name: str):
    """Fetch content of a sample CSV for a specific target class."""
    if class_name not in config.CLASS_NAMES:
        raise HTTPException(status_code=404, detail="Class not found.")

    raw_dir = config.RAW_DATA_DIR
    folder_name = [k for k, v in config.RAW_FOLDER_TO_CLASS.items() if v == class_name][0]

    found = list((raw_dir / folder_name).rglob("*.csv"))
    if not found:
        found = list((raw_dir / "data" / folder_name).rglob("*.csv"))

    if not found:
        raise HTTPException(status_code=404, detail=f"No sample CSV found for class {class_name}.")

    sample_path = found[0]
    raw_signal = read_raw_csv_signal(sample_path)
    spectrogram_rgb = convert_signal_to_spectrogram_image(raw_signal)

    pil_img = Image.fromarray(spectrogram_rgb, mode="RGB")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    classifier = get_active_classifier()
    if not hasattr(classifier.model, "classes_"):
        from training.train_pytorch import Trainer
        trainer = Trainer()
        classifier = trainer.fit(epochs=2)

    probs = classifier.predict_proba(spectrogram_rgb)[0]
    pred_id = int(np.argmax(probs))
    pred_class = config.CLASS_NAMES[pred_id]
    confidence = float(probs[pred_id] * 100.0)

    probabilities = {
        cls_name: round(float(probs[idx] * 100.0), 2)
        for idx, cls_name in enumerate(config.CLASS_NAMES)
    }

    return {
        "success": True,
        "filename": sample_path.name,
        "actual_class": class_name,
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
        "spectrogram_image": f"data:image/png;base64,{img_base64}",
        "signal_stats": {
            "matrix_shape": list(raw_signal.shape),
            "min_power": float(np.min(raw_signal)),
            "max_power": float(np.max(raw_signal)),
            "mean_power": float(np.mean(raw_signal)),
        },
    }


# serve static web files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
