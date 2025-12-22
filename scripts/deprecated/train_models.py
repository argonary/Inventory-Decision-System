import argparse
import pandas as pd
from pathlib import Path

from src.config import SNAPSHOTS_DIR, MODELS_DIR
from src.ml.trainer import train_lgbm_quantile
from src.ml.feature_config import FEATURES, TARGET_COL, CATEGORICAL_FEATURES


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LightGBM quantile models"
    )

    parser.add_argument(
        "--quantiles",
        nargs="+",
        type=float,
        default=[0.90, 0.95],
        help="List of quantiles to train (e.g. 0.9 0.95)",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    quantiles = args.quantiles

    print("Loading training table...")
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_model_table_2015.parquet"
    )

    for q in quantiles:
        if not (0 < q < 1):
            raise ValueError(f"Invalid quantile: {q}")

        q_label = int(q * 100)
        print(f"Training P{q_label} model...")

        model_path = MODELS_DIR / f"favorita_lgbm_p{q_label}.txt"

        train_lgbm_quantile(
            df=df,
            features=FEATURES,
            target_col=TARGET_COL,
            quantile=q,
            model_path=model_path,
            categorical_features=CATEGORICAL_FEATURES,
        )

        print(f"✅ Saved model to {model_path}")

    print("🎉 Training complete")


if __name__ == "__main__":
    main()
