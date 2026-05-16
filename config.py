# config.py
# ─────────────────────────────────────────────────────────────────────────────
# All hyperparameters live here. Change things here, not scattered in other files.
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = "data/images"          # folder with your 70 PNGs
ANNOTATIONS_CSV = "data/annotations.csv" # ground truth CSV
OUTPUT_DIR      = "outputs"              # where models + plots are saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data split ────────────────────────────────────────────────────────────────
VAL_SPLIT   = 0.2    # 20% of images go to validation (~14 images)
RANDOM_SEED = 42     # for reproducibility

# ── Image settings ────────────────────────────────────────────────────────────
IMAGE_W = 640        # original image width
IMAGE_H = 480        # original image height

# Each detected tube lid is cropped to this size before going into the CNN
CROP_SIZE = 64       # pixels. 64×64 is a good starting point.

# ── Model settings ────────────────────────────────────────────────────────────
# CNN backbone: "resnet18" or "lenet"
# resnet18 is strongly recommended — pretrained weights help a lot with 70 images
BACKBONE = "resnet18"
PRETRAINED = True    # use ImageNet pretrained weights for the backbone

# Rowdy KNN head settings (from Jagtap et al. 2022)
ROWDY_K      = 4     # number of terms in Rowdy activation (K=4 is a good start)
ROWDY_N_INIT = 1.0   # scaling factor η — paper warns >1 can be sensitive
ROWDY_BASE   = "relu"  # base activation φ_1: "relu" or "tanh"

# ── Detection settings ────────────────────────────────────────────────────────
# For detection we use a simple sliding window / proposal approach.
# IoU threshold: a predicted box is a True Positive if IoU >= this with any GT box
IOU_THRESHOLD = 0.5

# ── Training settings ─────────────────────────────────────────────────────────
BATCH_SIZE    = 16    # number of crops per batch
NUM_EPOCHS    = 60
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
LR_ANNEAL_EPOCH = 30  # halve the LR at this epoch (paper recommends annealing for Rowdy)
LR_ANNEAL_FACTOR = 0.1

# ── Evaluation settings ───────────────────────────────────────────────────────
# How many pixels away can a predicted center be from GT and still count as TP?
CENTER_DIST_THRESHOLD = 30   # pixels

# Angle error tolerance for "close enough"
ANGLE_TOLERANCE_DEG = 15     # degrees
