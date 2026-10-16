# config.py
# all project settings in one place so i dont have to dig through files every time

from pathlib import Path

# ---- paths ----
PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DATA_DIR     = PROJECT_ROOT / "dataset" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "dataset" / "processed"
SPECTROGRAM_DIR  = PROCESSED_DATA_DIR / "spectrogram_images"

MODEL_OUTPUT_DIR  = PROJECT_ROOT / "output" / "models"
LOG_OUTPUT_DIR    = PROJECT_ROOT / "output" / "logs"
FIGURE_OUTPUT_DIR = PROJECT_ROOT / "output" / "figures"

# ---- class labels ----
CLASS_NAMES   = ["car", "drone", "person"]
CLASS_TO_ID   = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES   = len(CLASS_NAMES)
INPUT_CHANNELS = 3  # RGB spectrogram image

# the raw dataset folders have capitalized names, this maps them to what i use internally
RAW_FOLDER_TO_CLASS = {
    "Cars":   "car",
    "Drones": "drone",
    "People": "person",
}

DATASET_NAME = "micro_doppler_csv"

# ---- spectrogram image settings ----
SPECTROGRAM_SHAPE = (128, 128)  # tried 64x64 first but the doppler patterns werent clear enough
COLORMAP_NAME = "viridis"       # tried magma too but viridis gave better separation between classes
LOG_EPSILON   = 1e-6            # prevents log(0) from blowing up to -inf
CLIP_PERCENTILE = (1.0, 99.0)  # cut off top and bottom 1% to remove noise spikes
                                # plain min-max was terrible, one spike ruined the whole image

# ---- training settings ----
EPOCHS     = 20    # 20 was enough, validation loss stopped improving around epoch 15
BATCH_SIZE = 32    # tried 64 but it crashed, 16 was way too slow, 32 is the sweet spot
LEARNING_RATE = 1e-3   # default adam lr, works fine for this
WEIGHT_DECAY  = 1e-4   # small L2 penalty to stop overfitting
EARLY_STOPPING_PATIENCE = 5  # if val score doesnt improve for 5 rounds, just stop
SEED   = 42     # so results are reproducible
DEVICE = "cuda" # falls back to cpu automatically if no gpu is available

# ---- misc ----
PROCESSED_MANIFEST_NAME = "manifest.json"  # train.py reads this to find the spectrogram files

CLASS_COLORS = {
    "car":    "#FF4444",
    "drone":  "#44AAFF",
    "person": "#44FF44",
}
