import pandas as pd

from src.config import RAW_DIR, SNAPSHOTS_DIR
from src.data.snapshot_builder import build_base_snapshot
from src.data.sampling import select_store_item_universe


# -----------------------------------------
# Configuration (explicit + intentional)
# -----------------------------------------
START_DATE = "2013-01-01"
END_DATE = "2015-12-31"

TOP_N_STORES = 25
MIN_ITEM_OBS = 500
TOP_N_ITEMS = 800


def main():
    print("🚀 Building training snapshot with universe selection")

    # -----------------------------------------
    # Load raw training data (minimal columns)
    # -----------------------------------------
    print("📥 Loading raw train.csv")
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

    print(f"Raw train shape: {train.shape}")

    # -----------------------------------------
    # Universe selection (CRITICAL STEP)
    # -----------------------------------------
    print("🔎 Selecting store/item universe")

    stores_u, items_u = select_store_item_universe(
        train=train,
        top_n_stores=TOP_N_STORES,
        min_item_obs=MIN_ITEM_OBS,
        top_n_items=TOP_N_ITEMS,
    )

    print(
        f"Selected {len(stores_u)} stores "
        f"and {len(items_u)} items"
    )

    train = train[
        train["store_nbr"].isin(stores_u)
        & train["item_nbr"].isin(items_u)
    ]

    print(f"After universe filter: {train.shape}")

    # -----------------------------------------
    # Load dimension tables
    # -----------------------------------------
    print("📦 Loading dimension tables")
    items = pd.read_csv(RAW_DIR / "items.csv")
    stores = pd.read_csv(RAW_DIR / "stores.csv")

    # -----------------------------------------
    # Build base snapshot (date filtering + joins)
    # -----------------------------------------
    print("🏗️ Building base training snapshot")

    df = build_base_snapshot(
        train=train,
        items=items,
        stores=stores,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    print(f"Base snapshot shape: {df.shape}")

    # -----------------------------------------
    # Persist snapshot (directory-style parquet)
    # -----------------------------------------
    out_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2015.parquet"
    df.to_parquet(out_path, index=False)

    print(f"✅ Training snapshot written to {out_path}")


if __name__ == "__main__":
    main()
