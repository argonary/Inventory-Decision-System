import logging

import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR
from src.features.feature_pipeline import apply_all_features
from src.logging_config import configure_logging
from src.validation.feature_validation import (
    validate_base_snapshot,
    validate_featured_snapshot,
)

configure_logging()
logger = logging.getLogger(__name__)

# -----------------------------------------
# Feature configuration
# -----------------------------------------
LAGS = [7, 14, 28]
ROLLS = [7, 14]


def build_featured_snapshot(
    base_snapshot: pd.DataFrame,
    holidays: pd.DataFrame,
    oil: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a fully featured training snapshot from a base snapshot.
    """

    # -----------------------------
    # Validate base snapshot
    # -----------------------------
    validate_base_snapshot(base_snapshot)

    df = base_snapshot.copy()

    # -----------------------------
    # Feature engineering
    # -----------------------------
    logger.info("Applying feature pipeline")
    df = apply_all_features(df, holidays, oil, lags=LAGS, rolls=ROLLS)

    # -----------------------------
    # Validate featured snapshot
    # -----------------------------
    validate_featured_snapshot(df)

    return df


def main():
    logger.info("Building featured training snapshot")

    # -----------------------------------------
    # Load base snapshot
    # -----------------------------------------
    base_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2015.parquet"
    df_base = pd.read_parquet(base_path)

    logger.info(f"Loaded base snapshot: {df_base.shape}")

    # -----------------------------------------
    # Load external tables
    # -----------------------------------------
    holidays = pd.read_csv(
        RAW_DIR / "holidays_events.csv",
        usecols=["date", "description"],
        parse_dates=["date"],
    )

    oil = pd.read_csv(
        RAW_DIR / "oil.csv",
        usecols=["date", "dcoilwtico"],
        parse_dates=["date"],
    )

    # -----------------------------------------
    # Build featured snapshot
    # -----------------------------------------
    df_featured = build_featured_snapshot(
        base_snapshot=df_base,
        holidays=holidays,
        oil=oil,
    )

    # -----------------------------------------
    # Persist
    # -----------------------------------------
    out_path = SNAPSHOTS_DIR / "favorita_train_featured_2015.parquet"
    df_featured.to_parquet(out_path, index=False)

    logger.info(f"Featured snapshot written to {out_path}")
    logger.info(f"Final shape: {df_featured.shape}")


if __name__ == "__main__":
    main()
