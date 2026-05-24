import logging

import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR
from src.data.snapshot_builder import build_base_snapshot
from src.data.sampling import select_store_item_universe, apply_universe_filter
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


# -----------------------------------------
# Configuration (explicit + intentional)
# -----------------------------------------
START_DATE = "2013-01-01"
END_DATE = "2015-12-31"

TOP_N_STORES = 25
MIN_ITEM_OBS = 500
TOP_N_ITEMS = 800


def main():
    logger.info("Building training snapshot with universe selection")

    # -----------------------------------------
    # Load raw training data (minimal columns)
    # -----------------------------------------
    logger.info("Loading raw train.csv")
    train = pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=[
            "date",
            "store_nbr",
            "item_nbr",
            "unit_sales",
            "onpromotion",
        ],
        parse_dates=["date"],
        low_memory=False,
    )

    logger.info(f"Raw train shape: {train.shape}")

    # -----------------------------------------
    # Universe selection (CRITICAL STEP)
    # -----------------------------------------
    logger.info("Selecting store/item universe")

    stores_u, items_u = select_store_item_universe(
        train=train,
        top_n_stores=TOP_N_STORES,
        min_item_obs=MIN_ITEM_OBS,
        top_n_items=TOP_N_ITEMS,
    )

    logger.info(
        f"Selected {len(stores_u)} stores "
        f"and {len(items_u)} items"
    )

    train = apply_universe_filter(train, stores_u, items_u)

    logger.info(f"After universe filter: {train.shape}")

    # -----------------------------------------
    # Load dimension tables
    # -----------------------------------------
    logger.info("Loading dimension tables")
    items = pd.read_csv(RAW_DIR / "items.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    # -----------------------------------------
    # Build base snapshot (date filtering + joins)
    # -----------------------------------------
    logger.info("Building base training snapshot")

    df = build_base_snapshot(
        train=train,
        items=items,
        stores=stores,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    logger.info(f"Base snapshot shape: {df.shape}")

    # -----------------------------------------
    # Persist snapshot (directory-style parquet)
    # -----------------------------------------
    out_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2015.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"Training snapshot written to {out_path}")


if __name__ == "__main__":
    main()
