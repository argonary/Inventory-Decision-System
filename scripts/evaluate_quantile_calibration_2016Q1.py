import pandas as pd
import numpy as np
from pathlib import Path

from src.config import SNAPSHOTS_DIR
from src.ml.predictor_factory import build_predictor
from src.ml.feature_config import FEATURES, TARGET_COL


TEST_SNAPSHOT = "favorita_test_featured_2016Q1.parquet"
QUANTILES = [0.90, 0.95]


def pinball_loss(y, y_hat, alpha):
    """
    Vectorized pinball loss for quantile regression.
    """
    diff = y - y_hat
    return np.mean(
        np.maximum(alpha * diff, (alpha - 1) * diff)
    )


def main():
    print("📥 Loading 2016Q1 featured test snapshot")
    df = pd.read_parquet(SNAPSHOTS_DIR / TEST_SNAPSHOT)

    print(f"Rows: {len(df):,}")

    y_true = df[TARGET_COL].values
    X = df[FEATURES]

    results = []

    for q in QUANTILES:
        print(f"\n📊 Evaluating P{int(q * 100)}")

        predictor = build_predictor(version="latest")

        # Predict
        y_hat = predictor.predict_df(
            df_features=X,
            service_level=q,
            clip_negative=True,
        )

        # Coverage
        coverage = np.mean(y_true <= y_hat)

        # Pinball loss
        loss = pinball_loss(y_true, y_hat, alpha=q)

        results.append(
            {
                "quantile": f"P{int(q * 100)}",
                "target_alpha": q,
                "empirical_coverage": coverage,
                "pinball_loss": loss,
            }
        )

        print(f"Coverage:      {coverage:.3f}")
        print(f"Pinball loss:  {loss:.4f}")

    print("\n✅ Calibration summary")
    print(pd.DataFrame(results))


if __name__ == "__main__":
    main()
