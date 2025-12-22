import pandas as pd
from src.config import SNAPSHOTS_DIR
from src.features.calendar import add_calendar_features


def main():
    in_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2016Q1.parquet"
    out_path = SNAPSHOTS_DIR / "favorita_train_snapshot_2016Q1_cal.parquet"

    df = pd.read_parquet(in_path)
    df = add_calendar_features(df)
    df.to_parquet(out_path, index=False)

    print(f"✅ Calendar snapshot written to {out_path}")


if __name__ == "__main__":
    main()
