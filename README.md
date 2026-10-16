# 🎯 AI-Based Radar Target Classification

An end-to-end deep learning system that classifies radar targets (People, Drones, Cars) from FMCW radar signals using a **CNN (Convolutional Neural Network)** built with PyTorch.

---

## 📌 Project Overview

This project processes raw FMCW radar CSV data, converts signals into **Range-Doppler spectrograms**, and trains a CNN classifier to distinguish between:
- 🚶 **People**
- 🚁 **Drones**
- 🚗 **Cars**

---

## 🗂️ Project Structure

```
AI-Based-Radar-Target-Classification/
├── client/                  # MERN Frontend: React + Vite (Hosted on Vercel)
│   ├── src/
│   │   ├── components/
│   │   │   ├── GoogleDriveBox.jsx    # Dedicated Google Drive input & dataset card
│   │   │   ├── RadarDropzone.jsx     # Drag & drop signal uploader & presets
│   │   │   ├── SpectrogramResults.jsx# Generated spectrogram & target probabilities
│   │   │   ├── ClassificationHistory.jsx # Live MongoDB history feed
│   │   │   └── Navbar.jsx
│   │   └── App.jsx
│   └── vercel.json          # Vercel deployment configuration
├── server/                  # MERN Backend: Node.js + Express + MongoDB (Hosted on Render)
│   ├── config/db.js         # MongoDB connection with in-memory fallback
│   ├── controllers/         # Radar inference & Google Drive link fetcher
│   ├── models/              # Classification history Mongoose schema
│   └── server.js            # Express REST API
├── ml_engine/
│   └── predict.py           # Python CLI inference bridge for PyTorch CNN
├── dataset/
│   └── raw/data/            # Raw micro-Doppler radar CSV signals (Cars, Drones, People)
├── model/
│   └── radar_cnn_pytorch.py # PyTorch 2D CNN architecture
├── preprocessing/           # Signal preprocessing & Viridis spectrogram synthesis
├── output/
│   ├── figures/             # Confusion matrix, ROC, training curves
│   └── models/              # Saved model checkpoints
├── Dockerfile               # Production container for Render deployment
├── render.yaml              # Render Blueprint deployment config
├── DEPLOYMENT.md            # Step-by-step Render + Vercel hosting guide
├── requirements-deploy.txt  # Cloud-optimized CPU PyTorch dependencies
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python **3.9+**
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/yashraj18111/AI-Based-Radar-Target-Classification.git
cd AI-Based-Radar-Target-Classification
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note for GPU users:** For CUDA-enabled PyTorch, visit [pytorch.org](https://pytorch.org/get-started/locally/) and install the appropriate version for your CUDA version.

---

## 🚀 Running the Project

### Train the Model

```bash
python main.py
```

This will:
1. Load and preprocess radar CSV data from `dataset/raw/data/`
2. Convert signals to spectrograms
3. Train the CNN model
4. Save the best model to `output/models/best_radar_model.pkl`
5. Generate evaluation plots in `output/figures/`

### View Results

After training, check `output/figures/` for:
- `training_curves.png` — Loss & accuracy over epochs
- `confusion_matrix.png` — Per-class prediction accuracy
- `roc_curves.png` — ROC curves for each class
- `spectrogram_predictions.png` — Sample predictions

---

## 🌐 MERN Web Application & Live Testing

A full-stack modern dashboard built with **React (Vite)**, **Express (Node.js)**, **MongoDB**, and **PyTorch**:

- 📁 **Dedicated Google Drive Test Hub:** Main page features a dedicated box with a direct link to the sample radar dataset, 1-click sample downloads (`Car`, `Drone`, `Person`), and an instant Google Drive link importer where users can paste shareable file links to test the model on the fly.
- 🎯 **Real-Time Spectrogram Generation:** Automatically renders 128×128 Viridis micro-Doppler heatmaps.
- 📊 **Confidence & Softmax Meters:** Visual probability distributions and matrix properties.
- 🗄️ **MongoDB History:** Logs and persists past user classifications.
- ☁️ **Cloud Deployment Ready:** Pre-configured for hosting the frontend on **Vercel** and the backend on **Render** (see [DEPLOYMENT.md](file:///c:/AI-Based-Radar-Target-Classification/DEPLOYMENT.md)).

### Run MERN App Locally:

```bash
# Terminal 1: Start Express Backend
cd server
npm install
node server.js

# Terminal 2: Start React Frontend
cd client
npm install
npm run dev
```

Visit `http://localhost:3000` in your browser.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Model | CNN (PyTorch) |
| Input | Range-Doppler Spectrograms |
| Classes | People, Drones, Cars |
| Output | `output/figures/` |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch`, `torchvision` | Deep learning framework |
| `numpy`, `scipy`, `pandas` | Data processing |
| `scikit-learn` | Metrics & evaluation |
| `matplotlib`, `seaborn` | Visualization |
| `Pillow` | Image processing |

---

## ⬇️ Pre-Trained Model

A pre-trained model is available on [GitHub Releases](https://github.com/yashraj18111/AI-Based-Radar-Target-Classification/releases/tag/v1.0.0).

**Auto-download (recommended):** The model is **automatically downloaded** when you run `evaluate` or `serve` commands — no manual steps needed!

```bash
# This will auto-download the model if not present, then run evaluation
python main.py evaluate
```

**Manual download:** If auto-download fails, download `best_radar_model.pkl` from the [v1.0.0 release](https://github.com/yashraj18111/AI-Based-Radar-Target-Classification/releases/tag/v1.0.0) and place it in `output/models/`.

---

## ⚠️ Important Notes

- The **trained model file** (`output/models/best_radar_model.pkl`) is hosted on GitHub Releases (144 MB) and will be **auto-downloaded** when needed.
- The dataset (`dataset/raw/data/`) is included in the repo.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **PyTorch** — CNN model training
- **NumPy / SciPy** — Signal processing
- **Scikit-learn** — Evaluation metrics
- **Matplotlib / Seaborn** — Visualization

---

## 👤 Author

**Yashraj** — [github.com/yashraj18111](https://github.com/yashraj18111)