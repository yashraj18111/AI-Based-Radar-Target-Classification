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
├── dataset/
│   └── raw/data/
│       ├── People/          # Raw radar CSV files for people
│       ├── Drones/          # Raw radar CSV files for drones
│       └── Cars/            # Raw radar CSV files for cars
├── model/
│   └── radar_cnn_pytorch.py # CNN model architecture
├── preprocessing/
│   ├── loader.py            # Data loading utilities
│   ├── preprocess.py        # Signal preprocessing
│   ├── signal_to_image.py   # Spectrogram generation
│   ├── splitter.py          # Train/val/test split
│   └── transforms.py        # Data augmentation
├── training/
│   └── train_pytorch.py     # Training pipeline
├── evaluation/
│   └── evaluate_pytorch.py  # Model evaluation & metrics
├── output/
│   ├── figures/             # Training curves, confusion matrix, ROC curves
│   ├── logs/                # Training logs
│   └── models/              # Saved model checkpoints (not tracked by git)
├── main.py                  # Entry point
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

## ⚠️ Important Notes

- The **trained model file** (`output/models/best_radar_model.pkl`) is **not included** in this repo (144 MB — exceeds GitHub's limit). Run `main.py` to train and generate it locally.
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