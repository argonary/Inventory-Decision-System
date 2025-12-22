import pandas as pd

from src.config import SNAPSHOTS_DIR
from src.ml.predictor_factory import build_default_predictor

def main():
    predictor = build_default_predictor()

    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_model_table_2016Q1.parquet"
    ).sample(5000, random_state=42)

    p90 = predictor.predict_df(df, service_level=0.90)
    p95 = predictor.predict_df(df, service_level=0.95)

    out = pd.DataFrame({
        "pred_p90": p90,
        "pred_p95": p95,
        "delta": (p95 - p90),
        "delta_pct": (p95 - p90) / (p90 + 1e-6),
        "actual": df["unit_sales"].values,
    })

    print("✅ Risk knob demo (P90 vs P95)")
    print("\nSummary stats:")
    print(out[["pred_p90", "pred_p95", "delta", "delta_pct"]].describe())

    print("\nFirst 10 rows:")
    print(out.head(10).to_string(index=False))

    # Quick sanity: P95 should be >= P90 almost always
    frac = (out["pred_p95"] >= out["pred_p90"]).mean()
    print(f"\nP95 >= P90 fraction: {frac:.4f}")

if __name__ == "__main__":
    main()
