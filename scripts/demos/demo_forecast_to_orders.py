import pandas as pd
import numpy as np

from src.config import (
    SNAPSHOTS_DIR,
    ACTIVE_MODEL_VERSION,
)
from src.ml.predictor_factory import build_predictor
from src.optimization.optimizer import optimize_orders


def main():
    print("🚀 Demo: Forecast → Capacity-Constrained Orders")

    # -------------------------
    # Configuration
    # -------------------------
    SERVICE_LEVELS = [0.90, 0.95]
    CAPACITY = 5_000
    FLOOR_RATIO = 0.10
    PERISHABLE_WEIGHT = 1.5
    SAMPLE_SIZE = 200

    snapshot_path = SNAPSHOTS_DIR / "favorita_train_featured_2015.parquet"

    print(f"Snapshot: {snapshot_path}")
    print(f"Model version: {ACTIVE_MODEL_VERSION}")
    print(f"Capacity: {CAPACITY}")

    # -------------------------
    # Load data
    # -------------------------
    df = pd.read_parquet(snapshot_path)

    # Pick a realistic slice
    store_id = df["store_nbr"].iloc[0]
    decision_date = df["date"].max()

    df_slice = (
        df[
            (df["store_nbr"] == store_id) &
            (df["date"] == decision_date)
        ]
        .sort_values("item_nbr")
        .head(SAMPLE_SIZE)
        .copy()
    )

    if df_slice.empty:
        raise RuntimeError("No data found for demo slice.")

    print(
        f"Demo slice → store {store_id}, "
        f"date {decision_date}, "
        f"{len(df_slice)} SKUs"
    )

    # -------------------------
    # Build predictor
    # -------------------------
    predictor = build_predictor(version=ACTIVE_MODEL_VERSION)

    # -------------------------
    # Forecasts + Optimization
    # -------------------------
    forecasts = {}
    orders = {}

    for alpha in SERVICE_LEVELS:
        print(f"📈 Predicting P{int(alpha * 100)} demand...")

        preds = predictor.predict_df(
            df_slice,
            service_level=alpha,
        )

        forecasts[alpha] = preds

        print(f"⚙️ Optimizing orders for P{int(alpha * 100)}...")

        orders[alpha] = optimize_orders(
            demand=preds,
            capacity=CAPACITY,
            service_level=alpha,
            perishable_flags=df_slice["perishable"].values,
            perishable_weight=PERISHABLE_WEIGHT,
            floor_ratio=FLOOR_RATIO,
        )

    # -------------------------
    # Assemble comparison table
    # -------------------------
    result = pd.DataFrame(
        {
            "item_nbr": df_slice["item_nbr"].values,
            "actual_sales": df_slice["unit_sales"].values,
            "perishable": df_slice["perishable"].values,
            "P90_forecast": forecasts[0.90],
            "P95_forecast": forecasts[0.95],
            "order_P90": orders[0.90],
            "order_P95": orders[0.95],
        }
    )

    result["delta_orders"] = result["order_P95"] - result["order_P90"]

    # -------------------------
    # Summary metrics
    # -------------------------
    print("\n📊 Summary metrics")
    summary = pd.DataFrame(
        {
            "Total forecast": [
                result["P90_forecast"].sum(),
                result["P95_forecast"].sum(),
            ],
            "Total orders": [
                result["order_P90"].sum(),
                result["order_P95"].sum(),
            ],
            "Avg order per SKU": [
                result["order_P90"].mean(),
                result["order_P95"].mean(),
            ],
        },
        index=["P90", "P95"],
    )

    print(summary.round(2))

    # -------------------------
    # Show top movers
    # -------------------------
    print("\n🔼 SKUs with largest order increase (P95 vs P90)")
    print(
        result.sort_values("delta_orders", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\n🔽 SKUs with smallest / reduced allocation")
    print(
        result.sort_values("delta_orders", ascending=True)
        .head(10)
        .to_string(index=False)
    )

    print("\n✅ Demo complete.")


if __name__ == "__main__":
    main()
