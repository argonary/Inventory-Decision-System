import numpy as np
import pandas as pd

from src.config import SNAPSHOTS_DIR
from src.ml.predictor_factory import build_default_predictor
from src.ml.splits import VALID_START, VALID_END


def pinball_loss(y_true, y_pred, alpha):
    diff = y_true - y_pred
    return np.mean(
        np.maximum(alpha * diff, (alpha - 1) * diff)
    )


def main():
    print("📥 Loading featured snapshot...")
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_featured_2015.parquet"
    )

    df["date"] = pd.to_datetime(df["date"]).dt.date

    valid_df = df[
        (df["date"] >= VALID_START) &
        (df["date"] <= VALID_END)
    ].copy()

    print(f"Validation rows: {len(valid_df):,}")

    predictor = build_default_predictor()

    y_true = valid_df["unit_sales"].values

    for alpha in [0.90, 0.95]:
        print(f"\n📊 Evaluating P{int(alpha * 100)}")

        y_pred = predictor.predict_df(
            valid_df,
            service_level=alpha,
        )

        coverage = np.mean(y_true <= y_pred)
        loss = pinball_loss(y_true, y_pred, alpha)

        print(f"Coverage: {coverage:.3f}")
        print(f"Pinball loss: {loss:.4f}")


if __name__ == "__main__":
    main()
