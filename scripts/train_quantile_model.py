import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import SNAPSHOTS_DIR, MODELS_DIR
from src.ml.feature_config import (
    FEATURES,
    TARGET_COL,
    CATEGORICAL_FEATURES,
)
from src.ml.splits import (
    TRAIN_START,
    TRAIN_END,
    VALID_START,
    VALID_END,
)
from src.ml.trainer import train_lgbm_quantile
from src.features.categorical import extract_category_schemas, save_category_schemas


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LightGBM quantile models with deterministic time split"
    )

    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.90, 0.95],
        help="Quantiles to train (e.g. 0.9 0.95)",
    )

    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version name (e.g. v1, v2_2025_12_20)",
    )

    parser.add_argument(
        "--update-latest",
        action="store_true",
        help="After training, overwrite data/models/latest/ with this version's artifacts.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    quantiles = args.quantiles
    version = args.version

    model_dir = MODELS_DIR / version
    model_dir.mkdir(parents=True, exist_ok=True)

    print("📥 Loading featured training snapshot...")
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_featured_2015.parquet"
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date

    print("✂️ Applying deterministic time split...")
    train_df = df[
        (df["date"] >= TRAIN_START) &
        (df["date"] <= TRAIN_END)
    ].copy()

    valid_df = df[
        (df["date"] >= VALID_START) &
        (df["date"] <= VALID_END)
    ].copy()

    print(
        f"Train rows: {len(train_df):,} | "
        f"Valid rows: {len(valid_df):,}"
    )

    if train_df.empty or valid_df.empty:
        raise RuntimeError("Train/validation split produced empty dataset.")

    print("📦 Extracting and saving category schemas (TRAIN ONLY)...")
    schemas = extract_category_schemas(
        train_df,
        categorical_features=CATEGORICAL_FEATURES,
    )

    schema_path = model_dir / "category_schemas.json"
    save_category_schemas(schemas, schema_path)

    print(f"✅ Category schemas saved to {schema_path}")

    for q in quantiles:
        if not (0 < q < 1):
            raise ValueError(f"Invalid quantile: {q}")

        q_label = int(q * 100)
        model_path = model_dir / f"favorita_lgbm_p{q_label}.txt"

        print(f"🚀 Training P{q_label} quantile model...")

        train_lgbm_quantile(
            df=train_df,
            features=FEATURES,
            target_col=TARGET_COL,
            quantile=q,
            categorical_features=CATEGORICAL_FEATURES,
            model_path=model_path,
        )

        print(f"✅ Saved model to {model_path}")

    print("📝 Writing metadata...")
    metadata = {
        "version": version,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "dataset": "favorita_train_featured_2015.parquet",
        "train_window": [str(TRAIN_START), str(TRAIN_END)],
        "valid_window": [str(VALID_START), str(VALID_END)],
        "quantiles": quantiles,
    }

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata written to {model_dir / 'metadata.json'}")

    if args.update_latest:
        latest_dir = MODELS_DIR / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(model_dir, latest_dir)
        print(f"✅ data/models/latest/ updated → {version}")

    print("🎉 Training complete")


if __name__ == "__main__":
    main()
