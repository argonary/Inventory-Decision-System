import logging

import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR
from src.data.sampling import select_store_item_universe
from src.data.snapshot_builder import build_base_snapshot
from src.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

START_DATE = "2016-01-04"
END_DATE = "2016-04-30"

OUTPUT_NAME = "favorita_test_snapshot_2016Q1.parquet"


def main():
    logger.info("🚀 Building 2016Q1 test snapshot")

    # --------------------------------------------------
    # Load raw train.csv (minimal columns)
    # --------------------------------------------------
    logger.info("📥 Loading raw train.csv")
    train_df = pd.read_csv(
        RAW_DIR / "train.csv",
        usecols=["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"],
        parse_dates=["date"],
    )

    logger.info(f"Raw train rows loaded: {len(train_df):,}")

    # --------------------------------------------------
    # Load dimension tables
    # --------------------------------------------------
    logger.info("📦 Loading dimension tables")
    items = pd.read_csv(RAW_DIR / "items.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    # --------------------------------------------------
    # Reuse SAME universe logic as training
    # --------------------------------------------------
    logger.info("🔎 Selecting store/item universe (same as training)")
    store_ids, item_ids = select_store_item_universe(train_df)

    # --------------------------------------------------
    # Filter to 2016 Jan–Apr + universe
    # --------------------------------------------------
    logger.info("✂️ Filtering to Jan–Apr 2016")
    mask = (
        (train_df["date"] >= START_DATE)
        & (train_df["date"] <= END_DATE)
        & (train_df["store_nbr"].isin(store_ids))
        & (train_df["item_nbr"].isin(item_ids))
    )

    df_slice = train_df.loc[mask].copy()
    logger.info(f"Filtered rows: {len(df_slice):,}")

    # --------------------------------------------------
    # Build base snapshot (same logic as training)
    # --------------------------------------------------
    logger.info("🏗️ Building base snapshot")
    snapshot = build_base_snapshot(
    df_slice,
    items,
    stores,
    START_DATE,
    END_DATE,
    )   

    # --------------------------------------------------
    # Write output
    # --------------------------------------------------
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    out_path = SNAPSHOTS_DIR / OUTPUT_NAME

    snapshot.to_parquet(out_path, index=False)

    logger.info(f"✅ Test snapshot written to {out_path}")
    logger.info(f"Final shape: {snapshot.shape}")


if __name__ == "__main__":
    main()
