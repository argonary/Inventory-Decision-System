import pandas as pd


def add_promotion_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure onpromotion is a clean binary feature.
    Missing means not on promotion.
    """
    df = df.copy()

    df["onpromotion"] = (
        df["onpromotion"]
        .fillna(0)
        .astype(int)
    )

    return df
