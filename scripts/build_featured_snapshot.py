import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR
from src.features.calendar import add_calendar_features
from src.features.holidays import add_holiday_feature
from src.features.oil import add_oil_feature
from src.features.promotion import add_promotion_feature
from src.features.lags import add_lag_features
from src.validation.feature_validation import (
    validate_base_snapshot,
    validate_featured_snapshot,
)

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
    print("➕ Adding calendar features")
    df = add_calendar_features(df)

    print("➕ Adding holiday feature")
    df = add_holiday_feature(df, holidays)

    print("➕ Adding oil feature")
    df = add_oil_feature(df, oil)

    print("➕ Adding promotion feature")
    df = add_promotion_feature(df)

    # Ensure deterministic order before time-series ops
    df = df.sort_values(
        ["store_nbr", "item_nbr", "date"]
    ).reset_index(drop=True)

    print("➕ Adding lag & rolling features")
    df = add_lag_features(
        df,
        lags=LAGS,
        rolls=ROLLS,
    )

    # -----------------------------
    # Validate featured snapshot
    # -----------------------------
    validate_featured_snapshot(df)

    return df


def main():
    print("🚀 Building featured training snapshot")

    # -----------------------------------------
    # Load base snapshot
    # -----------------------------------------
    base_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2015.parquet"
    df_base = pd.read_parquet(base_path)

    print(f"Loaded base snapshot: {df_base.shape}")

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

    print(f"✅ Featured snapshot written to {out_path}")
    print(f"Final shape: {df_featured.shape}")


if __name__ == "__main__":
    main()
