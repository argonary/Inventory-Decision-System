import pandas as pd
from src.config import SNAPSHOTS_DIR, DATA_DIR
from src.data.sampling import load_sampling_universe, apply_sampling_universe


def main():
    print("Loading snapshot...")
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_snapshot_2016Q1_cal.parquet"
    )

    print("Loading sampling universe...")
    stores, items = load_sampling_universe(
        DATA_DIR / "sampling_universe.parquet"
    )

    print("Applying sampling universe...")
    df = apply_sampling_universe(df, stores, items)

    out_path = (
        SNAPSHOTS_DIR
        / "favorita_train_snapshot_2016Q1_cal_sampled.parquet"
    )
    df.to_parquet(out_path, index=False)

    print(f"✅ Sampled snapshot written to {out_path}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
