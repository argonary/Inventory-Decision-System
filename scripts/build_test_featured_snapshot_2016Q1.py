import pandas as pd
from pathlib import Path

from src.config import SNAPSHOTS_DIR, RAW_DIR
from src.features.calendar import add_calendar_features
from src.features.holidays import add_holiday_feature
from src.features.oil import add_oil_feature
from src.features.promotion import add_promotion_feature
from src.features.lags import add_lag_features
from src.validation.feature_validation import validate_featured_snapshot


INPUT_SNAPSHOT = "favorita_test_snapshot_2016Q1.parquet"
OUTPUT_SNAPSHOT = "favorita_test_featured_2016Q1.parquet"


def main():
    print("🚀 Building featured TEST snapshot (2016Q1)")

    in_path = SNAPSHOTS_DIR / INPUT_SNAPSHOT
    out_path = SNAPSHOTS_DIR / OUTPUT_SNAPSHOT

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input snapshot: {in_path}")

    # --------------------------------------------------
    # Load base snapshot
    # --------------------------------------------------
    print("📥 Loading base test snapshot")
    df = pd.read_parquet(in_path)
    print(f"Base snapshot shape: {df.shape}")

    # --------------------------------------------------
    # Load auxiliary tables
    # --------------------------------------------------
    print("📦 Loading auxiliary tables")
    holidays = pd.read_csv(RAW_DIR / "holidays_events.csv", parse_dates=["date"])
    oil = pd.read_csv(RAW_DIR / "oil.csv", parse_dates=["date"])

    # --------------------------------------------------
    # Apply SAME feature steps as training (explicit)
    # --------------------------------------------------
    print("➕ Adding calendar features")
    df = add_calendar_features(df)

    print("➕ Adding holiday feature")
    df = add_holiday_feature(df, holidays)

    print("➕ Adding oil feature")
    df = add_oil_feature(df, oil)

    print("➕ Adding promotion feature")
    df = add_promotion_feature(df)

    print("➕ Adding lag & rolling features")
    df = df.sort_values(["store_nbr", "item_nbr", "date"])
    df = add_lag_features(df, lags=[7, 14, 28], rolls=[7, 14])

    # --------------------------------------------------
    # Validate (same rules as training)
    # --------------------------------------------------
    print("🔎 Validating featured snapshot")
    validate_featured_snapshot(df)

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    df.to_parquet(out_path, index=False)

    print(f"✅ Test featured snapshot written to {out_path}")
    print(f"Final shape: {df.shape}")


if __name__ == "__main__":
    main()
