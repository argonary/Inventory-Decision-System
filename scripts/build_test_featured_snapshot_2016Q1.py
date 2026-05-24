import logging

import pandas as pd

from src.config import SNAPSHOTS_DIR, RAW_DIR
from src.features.feature_pipeline import apply_all_features
from src.logging_config import configure_logging
from src.validation.feature_validation import (
    validate_base_snapshot,
    validate_featured_snapshot,
)

configure_logging()
logger = logging.getLogger(__name__)


INPUT_SNAPSHOT = "favorita_test_snapshot_2016Q1.parquet"
OUTPUT_SNAPSHOT = "favorita_test_featured_2016Q1.parquet"


def main():
    logger.info("Building featured TEST snapshot (2016Q1)")

    in_path = SNAPSHOTS_DIR / INPUT_SNAPSHOT
    out_path = SNAPSHOTS_DIR / OUTPUT_SNAPSHOT

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input snapshot: {in_path}")

    # --------------------------------------------------
    # Load base snapshot
    # --------------------------------------------------
    logger.info("Loading base test snapshot")
    df = pd.read_parquet(in_path)
    logger.info(f"Base snapshot shape: {df.shape}")

    validate_base_snapshot(df)

    # --------------------------------------------------
    # Load auxiliary tables
    # --------------------------------------------------
    logger.info("Loading auxiliary tables")
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])

    # --------------------------------------------------
    # Apply SAME feature steps as training (explicit)
    # --------------------------------------------------
    logger.info("Applying feature pipeline")
    df = apply_all_features(df, holidays, oil, lags=[7, 14, 28], rolls=[7, 14])

    # --------------------------------------------------
    # Validate (same rules as training)
    # --------------------------------------------------
    logger.info("Validating featured snapshot")
    validate_featured_snapshot(df)

    # --------------------------------------------------
    # Trim pre-history rows used only for lag computation
    # (Jan 4–31 loaded to populate 28-day lags for Feb 1+;
    #  artifact must only expose Feb–Apr.)
    # --------------------------------------------------
    df = df[df["date"] >= "2016-02-01"].reset_index(drop=True)
    logger.info(f"Trimmed to Feb–Apr rows: {df.shape}")

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    df.to_parquet(out_path, index=False)

    logger.info(f"Test featured snapshot written to {out_path}")
    logger.info(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
