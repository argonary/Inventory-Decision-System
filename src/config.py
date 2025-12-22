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

# Active model version used by inference
# Change this to switch models (e.g. "v1", "v2_2025_12_20")
ACTIVE_MODEL_VERSION = "v1"

# =====================================================
# Dataset selection (TRAIN vs DEMO / TEST)
# =====================================================

# Which featured snapshot the API should serve
# "train" → training-era data (2013–2015)
# "test"  → out-of-time demo data (2016Q1)
ACTIVE_DATASET_MODE = "test"  # <-- switch here for demos

FEATURED_SNAPSHOT_BY_MODE = {
    "train": "favorita_train_featured_2015.parquet",
    "test": "favorita_test_featured_2016Q1.parquet",
}
