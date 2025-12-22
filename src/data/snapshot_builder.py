import pandas as pd
from typing import Optional


def build_base_snapshot(
    train: pd.DataFrame,
    items: pd.DataFrame,
    stores: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build base snapshot from an already-filtered training table.
    """

    required_cols = {"date", "store_nbr", "item_nbr", "unit_sales"}
    missing = required_cols - set(train.columns)
    if missing:
        raise ValueError(
            f"`train` missing required columns: {missing}"
        )

    df = train.copy()

    if start_date is not None:
        df = df[df["date"] >= start_date]

    if end_date is not None:
        df = df[df["date"] <= end_date]

    pre_rows = len(df)

    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")

    if len(df) != pre_rows:
        raise ValueError(
            "Row count changed after joining dimension tables"
        )

    return df
