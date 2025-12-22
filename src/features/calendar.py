import pandas as pd


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard calendar-based features derived from `date`.

    Adds:
    - year
    - month
    - dayofweek
    - weekofyear
    - is_weekend
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    return df
