import pandas as pd
from typing import List

# ============================================================
# Base snapshot validation
# ============================================================

BASE_REQUIRED_COLUMNS = [
    "date",
    "store_nbr",
    "item_nbr",
    "unit_sales",
    "onpromotion",
]


def validate_base_snapshot(df: pd.DataFrame) -> None:
    """
    Validate structural integrity of the base snapshot
    (after raw joins, before feature engineering).
    """

    missing = set(BASE_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Base snapshot missing required columns: {missing}"
        )

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("`date` column must be datetime")

    if not pd.api.types.is_numeric_dtype(df["unit_sales"]):
        raise TypeError("`unit_sales` must be numeric")

    if df.empty:
        raise ValueError("Base snapshot is empty")

    print("✅ Base snapshot validation passed")


# ============================================================
# Featured snapshot validation
# ============================================================

# Columns that must NEVER contain nulls at this stage
NON_NULL_FEATURES = [
    # identifiers
    "date",
    "store_nbr",
    "item_nbr",

    # target
    "unit_sales",

    # calendar
    "year",
    "month",
    "dayofweek",
    "weekofyear",
    "is_weekend",

    # external
    "is_holiday",
    "dcoilwtico",

    # joins / categoricals
    "family",
    "class",
    "city",
    "state",
    "type",
    "perishable",

    # ops
    "onpromotion",
]


def validate_featured_snapshot(
    df: pd.DataFrame,
    lag_prefix: str = "lag_",
    roll_prefix: str = "rolling_",
) -> None:
    """
    Validate engineered snapshot BEFORE lag-row dropping.

    Rules:
    - Lag and rolling columns ARE allowed to have NaNs
    - All other core features must be fully populated
    """

    # -----------------------------
    # Structural checks
    # -----------------------------
    if df.empty:
        raise ValueError("Featured snapshot is empty")

    # Ensure lag / rolling columns exist
    lag_cols = [c for c in df.columns if c.startswith(lag_prefix)]
    roll_cols = [c for c in df.columns if c.startswith(roll_prefix)]

    if not lag_cols:
        raise ValueError("No lag features found")

    if not roll_cols:
        raise ValueError("No rolling features found")

    # -----------------------------
    # Null checks (non-lag only)
    # -----------------------------
    missing_non_null_cols = [
        c for c in NON_NULL_FEATURES
        if c in df.columns and df[c].isna().any()
    ]

    if missing_non_null_cols:
        raise ValueError(
            "Unexpected missing values in non-lag features: "
            f"{missing_non_null_cols}"
        )

    print("✅ Featured snapshot validation passed")
