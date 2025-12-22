import pandas as pd

from src.config import SNAPSHOTS_DIR
from src.ml.predictor_factory import build_default_predictor
from src.optimization.optimizer import optimize_proportional_allocation


def run_scenario(
    df_store_day: pd.DataFrame,
    predictor,
    service_level: float,
    service_floor_ratio: float,
    perishable_weight: float,
    capacity: int,
):
    # Forecast
    preds = predictor.predict_df(
        df_store_day,
        service_level=service_level,
    )

    demand = dict(zip(df_store_day["item_nbr"], preds))
    perishable_flags = dict(
        zip(
            df_store_day["item_nbr"],
            df_store_day["perishable"].astype(bool),
        )
    )

    # Optimize
    orders = optimize_proportional_allocation(
        demand=demand,
        capacity=capacity,
        service_floor_ratio=service_floor_ratio,
        perishable_flags=perishable_flags,
        perishable_weight=perishable_weight,
    )

    out = pd.DataFrame({
        "item_nbr": df_store_day["item_nbr"].values,
        "predicted_demand": preds,
        "perishable": df_store_day["perishable"].values,
        "order_qty": [orders[k] for k in df_store_day["item_nbr"]],
    })

    # Metrics
    total_orders = out["order_qty"].sum()
    perishable_orders = out.loc[out["perishable"] == 1, "order_qty"].sum()
    perishable_share = perishable_orders / total_orders

    top_20_share = (
        out.sort_values("order_qty", ascending=False)
        .head(20)["order_qty"]
        .sum()
        / total_orders
    )

    avg_order = out["order_qty"].mean()

    return {
        "service_level": f"P{int(service_level * 100)}",
        "floor_ratio": service_floor_ratio,
        "perishable_weight": perishable_weight,
        "total_units": total_orders,
        "perishable_share": round(perishable_share, 3),
        "top20_share": round(top_20_share, 3),
        "avg_units_per_sku": round(avg_order, 2),
    }


def main():
    print("🚀 Running scenario comparison")

    predictor = build_default_predictor()

    # Load data
    df = pd.read_parquet(
        SNAPSHOTS_DIR / "favorita_train_model_table_2016Q1.parquet"
    )

    # Fix decision context
    store_nbr = df["store_nbr"].value_counts().idxmax()
    df_store = df[df["store_nbr"] == store_nbr]
    date = df_store["date"].value_counts().idxmax()
    df_store_day = df_store[df_store["date"] == date]

    print(
        f"Decision context: store {store_nbr}, "
        f"date {date.date()}, "
        f"{len(df_store_day)} items"
    )

    capacity = 500

    scenarios = [
        (0.90, 0.10, 1.0),
        (0.90, 0.10, 1.5),
        (0.90, 0.20, 1.0),
        (0.90, 0.20, 1.5),
        (0.95, 0.10, 1.0),
        (0.95, 0.10, 1.5),
        (0.95, 0.20, 1.0),
        (0.95, 0.20, 1.5),
    ]

    rows = []
    for sl, floor, pw in scenarios:
        rows.append(
            run_scenario(
                df_store_day=df_store_day,
                predictor=predictor,
                service_level=sl,
                service_floor_ratio=floor,
                perishable_weight=pw,
                capacity=capacity,
            )
        )

    result = pd.DataFrame(rows)

    print("\n📊 Scenario comparison summary:")
    print(result.to_string(index=False))

    print("\n✅ Scenario analysis complete")


if __name__ == "__main__":
    main()
