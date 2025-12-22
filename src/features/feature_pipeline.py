import pandas as pd
from typing import List, Optional

from src.features.calendar import add_calendar_features
from src.features.holidays import add_holiday_features
from src.features.oil import add_oil_features
from src.features.lags import add_lag_features


def apply_all_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    rolls: Optional[List[int]] = None,
) -> pd.DataFrame:
    df = add_calendar_features(df)
    df = add_holiday_features(df, holidays_df)
    df = add_oil_features(df, oil_df)
    df = add_lag_features(df, lags=lags, rolls=rolls)
    return df
