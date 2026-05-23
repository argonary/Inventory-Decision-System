import pandas as pd
from typing import List, Optional

from src.features.calendar import add_calendar_features
from src.features.holidays import add_holiday_feature
from src.features.oil import add_oil_feature
from src.features.promotion import add_promotion_feature
from src.features.lags import add_lag_features


def apply_all_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolls: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_holiday_feature(df, holidays_df)
    df = add_oil_feature(df, oil_df)
    df = add_promotion_feature(df)
    df = df.sort_values(
        ["store_nbr", "item_nbr", "date"]
    ).reset_index(drop=True)
    df = add_lag_features(df, lags=lags, rolls=rolls)
    return df
