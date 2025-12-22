import pandas as pd
from typing import List, Tuple


def select_store_item_universe(
    train: pd.DataFrame,
    top_n_stores: int = 25,
    min_item_obs: int = 500,
    top_n_items: int = 800,
) -> Tuple[List[int], List[int]]:
    """
    Select a restricted store / item universe based on
    volume and history sufficiency.
    """

    # Top stores by total volume
    store_volume = (
        train.groupby("store_nbr")["unit_sales"]
        .sum()
        .sort_values(ascending=False)
    )
    stores = store_volume.head(top_n_stores).index.tolist()

    # Items with sufficient history
    item_history = (
        train.groupby("item_nbr")
        .agg(
            total_sales=("unit_sales", "sum"),
            n_obs=("unit_sales", "count"),
        )
    )

    items = (
        item_history
        .query("n_obs > @min_item_obs")
        .sort_values("total_sales", ascending=False)
        .head(top_n_items)
        .index.tolist()
    )

    return stores, items


def apply_universe_filter(
    train: pd.DataFrame,
    stores: List[int],
    items: List[int],
) -> pd.DataFrame:
    """
    Filter raw train data to selected store/item universe.
    """

    return train[
        train["store_nbr"].isin(stores)
        & train["item_nbr"].isin(items)
    ].copy()
