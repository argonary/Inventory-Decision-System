import pandas as pd
from typing import List, Dict


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: List[str],
) -> None:
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_dtypes(
    df: pd.DataFrame,
    dtype_map: Dict[str, str],
) -> None:
    errors = {}

    for col, expected in dtype_map.items():
        if col not in df.columns:
            continue

        actual = df[col].dtype

        if expected == "datetime" and not pd.api.types.is_datetime64_any_dtype(actual):
            errors[col] = f"expected datetime, got {actual}"

        elif expected == "numeric" and not pd.api.types.is_numeric_dtype(actual):
            errors[col] = f"expected numeric, got {actual}"

        elif expected == "category" and not (
            pd.api.types.is_categorical_dtype(actual)
            or pd.api.types.is_object_dtype(actual)
        ):
            errors[col] = f"expected category/object, got {actual}"

    if errors:
        raise TypeError(f"Dtype validation failed: {errors}")


def validate_missingness(
    df: pd.DataFrame,
    forbidden_missing: List[str],
) -> None:
    bad = {
        col: int(df[col].isna().sum())
        for col in forbidden_missing
        if col in df.columns and df[col].isna().any()
    }

    if bad:
        raise ValueError(f"Unexpected missing values: {bad}")
