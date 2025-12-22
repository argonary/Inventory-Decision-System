import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR

from src.data.snapshot_builder import build_training_snapshot
from src.features.calendar import add_calendar_features
from src.features.holidays import add_holiday_feature
from src.features.oil import add_oil_feature
from src.features.lags import add_lag_features

from src.validation.feature_validation import (
    validate_base_snapshot,
    validate_featured_snapshot,
)


def main():
    print("🚀 Starting end-to-end feature pipeline test")

    # --------------------------------------------------
    # 1. Load raw data
    # --------------------------------------------------
    print("📥 Loading raw data...")

    train = pd.read_csv(
        RAW_DIR / "train.csv",
        parse_dates=["date"],
        low_memory=False,
    )
    items = pd.read_csv(RAW_DIR / "items.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")
    oil = pd.read_csv(
        RAW_DIR / "oil.csv",
        parse_dates=["date"],
    )
    holidays = pd.read_csv(
        RAW_DIR / "holidays_events.csv",
        parse_dates=["date"],
    )

    # --------------------------------------------------
    # 2. Build base snapshot
    # --------------------------------------------------
    print("🧱 Building base snapshot...")

    df = build_training_snapshot(
        train=train,
        items=items,
        stores=stores,
        start_date="2013-01-01",
        end_date="2015-12-31",
    )

    validate_base_snapshot(df)

    print(f"Base snapshot shape: {df.shape}")

    # --------------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------------
    print("🛠 Applying calendar features...")
    df = add_calendar_features(df)

    print("🛢 Applying oil feature...")
    df = add_oil_feature(df, oil)

    print("🎉 Applying holiday feature...")
    df = add_holiday_feature(df, holidays)

    print("⏱ Applying lag features...")
    df = add_lag_features(
        df,
        lags=[7, 14, 28],
        rolls=[7, 14],
    )

    # --------------------------------------------------
    # 4. Validation of featured snapshot
    # --------------------------------------------------
    print("🔍 Validating featured snapshot...")
    validate_featured_snapshot(df)

    # --------------------------------------------------
    # 5. Save output (optional but useful)
    # --------------------------------------------------
    out_path = SNAPSHOTS_DIR / "favorita_featured_snapshot_test.parquet"
    df.to_parquet(out_path, index=False)

    print(f"✅ Featured snapshot written to {out_path}")
    print(f"Final shape: {df.shape}")
    print("🎯 Pipeline test complete")


if __name__ == "__main__":
    main()
