import json
from pathlib import Path

# =====================================================
# Project structure
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
MODELS_DIR = DATA_DIR / "models"

# =====================================================
# Model versioning
# =====================================================

# Active model version, read at import time from data/models/latest/metadata.json.
# Falls back to "unknown" if that file is missing (e.g. before first training run).
try:
    with open(MODELS_DIR / "latest" / "metadata.json") as _f:
        MODEL_VERSION_LABEL = json.load(_f)["version"]
except FileNotFoundError:
    MODEL_VERSION_LABEL = "unknown"

# =====================================================
# Dataset selection (TRAIN vs DEMO / TEST)
# =====================================================

# Which featured snapshot the API should serve
# “train” -> training-era data (2013-2015)
# “test”  -> out-of-time demo data (2016Q1)
ACTIVE_DATASET_MODE = "test"  # <-- switch here for demos

FEATURED_SNAPSHOT_BY_MODE = {
    "train": "favorita_train_featured_2015.parquet",
    "test": "favorita_test_featured_2016Q1.parquet",
}
