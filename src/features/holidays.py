import pandas as pd


def add_holiday_feature(
    df: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    holidays = holidays.copy()

    df["date"] = pd.to_datetime(df["date"])
    holidays["date"] = pd.to_datetime(holidays["date"])

    df = df.merge(
        holidays[["date", "description"]],
        on="date",
        how="left",
    )

    df["is_holiday"] = df["description"].notna().astype(int)
    df = df.drop(columns=["description"])

    return df
