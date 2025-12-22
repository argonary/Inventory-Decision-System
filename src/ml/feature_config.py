# src/ml/feature_config.py

TARGET_COL = "unit_sales"
TARGET_TRANSFORM = "log1p"  # must be inverted at inference

ID_COLS = [
    "store_nbr",
    "item_nbr",
    "date",
]

CATEGORICAL_FEATURES = [
    "family",
    "class",
    "city",
    "state",
    "type",
    "cluster",
]

NUMERIC_FEATURES = [
    # calendar
    "year",
    "month",
    "weekofyear",
    "dayofweek",

    # binary operational flags
    "onpromotion",
    "is_weekend",
    "is_holiday",
    "perishable",

    # external
    "dcoilwtico",

    # lags & rollings
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7",
    "rolling_14",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
