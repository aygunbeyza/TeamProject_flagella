"""
Shared configuration for ALL models (U-Net, YOLO, RT-DETR).
Everything that must be identical across models lives here: reference voxel spacing,
hit distance (evaluation metric) and the validation split (fixed by prepare_data.py).
"""
import os
from pathlib import Path

# ----------------------------------------------------------------- paths
PROJECT_DIR = Path(__file__).resolve().parent

# Raw data (read-only): <DATA_DIR>/train/<tomo_id>/slice_XXXX.jpg, train_labels.csv, train_ids.txt, val_ids.txt
DATA_DIR = Path(os.environ.get(
    "FLAGELLA_DATA_DIR", "/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data"))
LABELS_CSV = DATA_DIR / "train_labels.csv"
IMAGE_DIR = DATA_DIR / "train"
TRAIN_IDS_TXT = DATA_DIR / "train_ids.txt"
VAL_IDS_TXT = DATA_DIR / "val_ids.txt"

# Output of prepare_data.py (shared by all models)
PROCESSED_DIR = PROJECT_DIR / "processed"
SLICES_CSV = PROCESSED_DIR / "slices.csv"      # one row per image (slice)
GT_CSV = PROCESSED_DIR / "gt.csv"              # one row per ground-truth motor

# Each model has its own folder:  <model>/runs/<run_tag>/  (one folder per experiment)  and  <model>/results/
MODEL_DIRS = {m: PROJECT_DIR / m for m in ("unet", "yolo", "rtdetr")}

# ----------------------------------------------------------------- data / normalisation
REFERENCE_SPACING = 15.6        # median "Voxel spacing"; every image is rescaled to it
BG_SLICES_PER_TOMO = 3          # motor-free slices per motor-free tomogram (train AND val)
BG_REGION = (0.3, 0.7)          # background slices come from the middle 30-70 % of a tomogram
SPLIT_SEED = 0                  # only used if train_ids.txt / val_ids.txt do not exist
VAL_FRACTION = 0.2              # only used if train_ids.txt / val_ids.txt do not exist

# ----------------------------------------------------------------- evaluation (same for ALL models)
HIT_DISTANCE_PX_AT_REF = 24     # hit if closer than 24 px at the reference spacing ...
HIT_DISTANCE_NM = HIT_DISTANCE_PX_AT_REF * REFERENCE_SPACING   # ... = 374.4 physical units
MIN_SCORE = 0.05                # detections below this score are never written to predictions files
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]        # 0.05 ... 0.95
FIXED_THRESHOLD = 0.5           # additionally reported without tuning on the validation set

SEED = 0
