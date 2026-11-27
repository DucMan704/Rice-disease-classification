"""
config.py
=========
Single source of truth for paths and hyperparameters.
Nothing here needs to be edited to run on Kaggle - paths are auto-detected.
If you DO want to tune something (epochs, batch size, split ratios), this
is the only file you should touch.
"""

import os

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
ON_KAGGLE = os.path.isdir("/kaggle")

# Where raw Kaggle datasets are mounted (read-only). We search under here.
KAGGLE_INPUT_DIR = "/kaggle/input"

# Where we are allowed to write on Kaggle (persists for the session and is
# saved as notebook output when you commit the notebook).
KAGGLE_WORKING_DIR = "/kaggle/working"

# Optional manual override: `export DATA_DIR=/path/to/dataset` before running
# if you ever need to point at a custom location (not required on Kaggle).
DATA_DIR_OVERRIDE = os.environ.get("DATA_DIR", "").strip()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

OUTPUT_ROOT = os.path.join(KAGGLE_WORKING_DIR if ON_KAGGLE else PROJECT_ROOT, "outputs")

# Sub-directories under OUTPUT_ROOT (created automatically by utils.ensure_dirs)
DIRS = {
    "root": OUTPUT_ROOT,
    "logs": os.path.join(OUTPUT_ROOT, "logs"),
    "data_split": os.path.join(OUTPUT_ROOT, "data_split"),
    "models": os.path.join(OUTPUT_ROOT, "models"),
    "history": os.path.join(OUTPUT_ROOT, "history"),
    "metrics": os.path.join(OUTPUT_ROOT, "metrics"),
    "plots_curves": os.path.join(OUTPUT_ROOT, "plots", "training_curves"),
    "plots_cm": os.path.join(OUTPUT_ROOT, "plots", "confusion_matrix"),
    "plots_roc": os.path.join(OUTPUT_ROOT, "plots", "roc_curve"),
    "plots_pred": os.path.join(OUTPUT_ROOT, "plots", "prediction_examples"),
    "gradcam_correct": os.path.join(OUTPUT_ROOT, "plots", "gradcam", "correct"),
    "gradcam_incorrect": os.path.join(OUTPUT_ROOT, "plots", "gradcam", "incorrect"),
    "comparison": os.path.join(OUTPUT_ROOT, "comparison"),
}

# ---------------------------------------------------------------------------
# Data / split settings
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
BATCH_SIZE = 32
SEED = 42

VAL_SPLIT = 0.15   # fraction of full dataset
TEST_SPLIT = 0.15  # fraction of full dataset
# remaining (0.70) goes to training

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG")

# ---------------------------------------------------------------------------
# Training settings (Round 3 tuning: pulled epochs back slightly from Round 2
# since the curves were already converging, and pushed the fine-tune unfreeze
# fraction up further since that was the lever with the most headroom left)
# ---------------------------------------------------------------------------
EPOCHS_HEAD = 16          # phase 1: frozen backbone, train classifier head
EPOCHS_FINE_TUNE = 27     # phase 2: unfreeze top layers, fine-tune
FINE_TUNE_UNFREEZE_FRACTION = 0.80  # unfreeze last 75% of backbone layers

LR_HEAD = 1e-3
LR_FINE_TUNE = 1e-5

EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

DROPOUT_RATE = 0.3

# Grad-CAM
GRADCAM_NUM_CORRECT = 8
GRADCAM_NUM_INCORRECT = 8

# Prediction example grid
PRED_EXAMPLES_GRID = 12

# ---------------------------------------------------------------------------
# Models to train, in execution order
# ---------------------------------------------------------------------------
MODEL_NAMES = [
    "EfficientNetB0",
    "ResNet50",
    "DenseNet121",
    "MobileNetV3Large",
    "ConvNeXtTiny",
]
