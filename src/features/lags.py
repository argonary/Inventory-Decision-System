import pandas as pd
from typing import List, Optional


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolls: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = df.copy()

    lags = lags or [7, 14, 28]
    rolls = rolls or [7, 14]

    df = df.sort_values(
        ["store_nbr", "item_nbr", "date"]
    )

    g = df.groupby(["store_nbr", "item_nbr"], sort=False)

    for lag in lags:
        df[f"lag_{lag}"] = g["unit_sales"].shift(lag)

    for window in rolls:
        df[f"rolling_{window}"] = (
            g["unit_sales"]
            .shift(1)
            .rolling(window)
            .mean()
        )

    return df
