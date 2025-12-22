import pandas as pd

from src.config import SNAPSHOTS_DIR, MODELS_DIR, ACTIVE_MODEL_VERSION
from src.ml.predictor_factory import build_predictor


def main():
    print("🚀 Starting inference sanity check...")

    snapshot_path = SNAPSHOTS_DIR / "favorita_train_featured_2015.parquet"
    model_dir = MODELS_DIR / ACTIVE_MODEL_VERSION

    print(f"Snapshot path: {snapshot_path}")
    print(f"Model dir:     {model_dir}")

    # ---- Load data ----
    print("Loading featured snapshot...")
    df = pd.read_parquet(snapshot_path)

    print(f"Loaded dataframe shape: {df.shape}")

    if df.empty:
        raise ValueError("❌ Loaded dataframe is empty. Cannot run inference.")

    # ---- Sample data ----
    n = min(5_000, len(df))
    df_sample = df.sample(n=n, random_state=1)

    print(f"Sampled {n} rows for inference")

    # ---- Load predictor (VERSIONED) ----
    print("Initializing QuantilePredictor...")
    predictor = build_predictor(version=ACTIVE_MODEL_VERSION)

    # ---- Run inference ----
    print("Running predictions (P90)...")
    preds = predictor.predict_df(
        df_sample,
        service_level=0.90,
    )

    print("Predictions completed.")

    # ---- Sanity checks ----
    preds_series = pd.Series(preds)

    print("\n📊 Prediction summary:")
    print(preds_series.describe())

    print("\n🔍 First 10 predictions:")
    print(preds_series.head(10).to_string(index=False))

    print("\n🔍 Corresponding actual unit_sales:")
    print(
        df_sample["unit_sales"]
        .head(10)
        .reset_index(drop=True)
        .to_string(index=False)
    )

    print("\n✅ Inference sanity check complete.")


if __name__ == "__main__":
    main()
