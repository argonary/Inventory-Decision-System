import pandas as pd


def add_oil_feature(
    df: pd.DataFrame,
    oil: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    oil = oil.copy()

    df["date"] = pd.to_datetime(df["date"])
    oil["date"] = pd.to_datetime(oil["date"])

    df = df.merge(
        oil[["date", "dcoilwtico"]],
        on="date",
        how="left",
    )

    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    return df
