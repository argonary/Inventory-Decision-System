import pandas as pd
from src.config import SNAPSHOTS_DIR

SNAPSHOT = SNAPSHOTS_DIR / "favorita_test_featured_2016Q1.parquet"

def main():
    print("📥 Loading 2016Q1 featured snapshot...")
    df = pd.read_parquet(SNAPSHOT)
    df["date"] = pd.to_datetime(df["date"])

    # ---- Pick a date with many SKUs ----
    date_counts = (
        df.groupby(["store_nbr", "date"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
    )

    row = date_counts.iloc[0]
    store_id = int(row["store_nbr"])
    decision_date = row["date"]

    print(f"\n✅ Selected store/date:")
    print(f"Store: {store_id}")
    print(f"Date:  {decision_date.date()}")

    # ---- Pull SKUs for that store/date ----
    df_slice = df[
        (df["store_nbr"] == store_id) &
        (df["date"] == decision_date)
    ].sort_values("item_nbr")

    # Keep it small and clean
    df_slice = df_slice.head(15)

    print("\n🧾 Sample SKUs:")
    print(df_slice[["item_nbr", "perishable", "onpromotion"]])

    # ---- Build API payload ----
    payload = {
        "store_nbr": store_id,
        "date": decision_date.strftime("%Y-%m-%d"),
        "service_level": 0.90,
        "capacity_units": 5000,
        "service_floor_ratio": 0.0,
        "items": [
            {
                "item_nbr": int(r.item_nbr),
                "onpromotion": bool(r.onpromotion),
            }
            for r in df_slice.itertuples()
        ],
    }

    print("\n📦 Ready-to-use API payload:\n")
    print(payload)

if __name__ == "__main__":
    main()
