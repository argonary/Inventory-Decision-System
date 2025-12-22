from fastapi import FastAPI, HTTPException
from typing import List
import pandas as pd
import numpy as np

from src.config import (
    SNAPSHOTS_DIR,
    ACTIVE_MODEL_VERSION,
    ACTIVE_DATASET_MODE,
    FEATURED_SNAPSHOT_BY_MODE,
)
from src.ml.predictor_factory import build_default_predictor
from src.optimization.optimizer import optimize_proportional_allocation
from api.schemas import ForecastToOrdersRequest, ForecastToOrdersResponse

# =====================================================
# App metadata
# =====================================================

app = FastAPI(
    title="Forecast-to-Orders API",
    description="Quantile forecasting with capacity-constrained optimization",
    version="1.0",
)

# =====================================================
# Load predictor (SAFE: never crashes on missing latest/)
# =====================================================

print("📦 Loading predictor...")
predictor = build_default_predictor()

# =====================================================
# Load featured snapshot (mode-aware)
# =====================================================

snapshot_name = FEATURED_SNAPSHOT_BY_MODE[ACTIVE_DATASET_MODE]
FEATURED_SNAPSHOT_PATH = SNAPSHOTS_DIR / snapshot_name

print(f"📦 Loading featured snapshot ({ACTIVE_DATASET_MODE})...")
df_features = pd.read_parquet(FEATURED_SNAPSHOT_PATH)
df_features["date"] = pd.to_datetime(df_features["date"])

print(f"✅ Loaded snapshot {snapshot_name} with shape {df_features.shape}")

# =====================================================
# Health & version endpoints
# =====================================================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {
        "model_version": ACTIVE_MODEL_VERSION,
        "dataset_mode": ACTIVE_DATASET_MODE,
        "snapshot": FEATURED_SNAPSHOT_PATH.name,
    }

# =====================================================
# Forecast → Orders endpoint
# =====================================================

@app.post("/forecast-to-orders", response_model=ForecastToOrdersResponse)
def forecast_to_orders(req: ForecastToOrdersRequest):

    # -----------------------------
    # Parse & validate inputs
    # -----------------------------
    try:
        decision_date = pd.to_datetime(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format")

    store_id = req.store_nbr
    service_level = req.service_level
    capacity = req.capacity_units
    service_floor_ratio = req.service_floor_ratio or 0.0
    perishable_weight = req.perishable_weight or 1.0

    # -----------------------------
    # Slice snapshot (store + date)
    # -----------------------------
    df_slice = df_features[
        (df_features["store_nbr"] == store_id)
        & (df_features["date"] == decision_date)
    ].copy()

    if df_slice.empty:
        raise HTTPException(
            status_code=404,
            detail="No feature data available for store/date",
        )

    # -----------------------------
    # Restrict to requested SKUs
    # -----------------------------
    item_map = {item.item_nbr: item.onpromotion for item in req.items}

    df_slice = df_slice[df_slice["item_nbr"].isin(item_map.keys())]

    if df_slice.empty:
        raise HTTPException(
            status_code=404,
            detail="No matching SKUs found for request",
        )

    # -----------------------------
    # Enforce ONE ROW PER SKU
    # -----------------------------
    df_slice = (
        df_slice
        .sort_values("item_nbr")
        .drop_duplicates(subset=["item_nbr"], keep="last")
        .reset_index(drop=True)
    )

    assert df_slice["item_nbr"].is_unique, "Duplicate SKUs in decision slice"

    # -----------------------------
    # Override onpromotion flags
    # -----------------------------
    df_slice["onpromotion"] = (
        df_slice["item_nbr"]
        .map(item_map)
        .astype(int)
    )

    # -----------------------------
    # Predict quantile demand
    # -----------------------------
    y_hat = predictor.predict_df(
        df_slice,
        service_level=service_level,
    )

    df_slice = df_slice.assign(forecast=y_hat)

    # -----------------------------
    # Build optimizer inputs
    # -----------------------------
    demand = dict(zip(df_slice["item_nbr"], df_slice["forecast"]))
    perishable_flags = dict(
        zip(df_slice["item_nbr"], df_slice["perishable"])
    )

    # -----------------------------
    # Optimize orders
    # Capacity is a CAP, not a target
    # -----------------------------
    orders = optimize_proportional_allocation(
        demand=demand,
        capacity=capacity,
        service_floor_ratio=service_floor_ratio,
        perishable_flags=perishable_flags,
        perishable_weight=perishable_weight,
        fill_capacity=False,
    )

    # -----------------------------
    # Build response rows
    # -----------------------------
    results = [
        {
            "item_nbr": int(item),
            "forecast": round(float(demand[item]), 2),
            "order_qty": int(orders.get(item, 0)),
        }
        for item in df_slice["item_nbr"]
    ]

    # -----------------------------
    # Summary metrics
    # -----------------------------
    total_forecast = float(np.sum(list(demand.values())))
    total_orders = int(np.sum(list(orders.values())))

    return {
        "store_nbr": store_id,
        "date": req.date,
        "service_level": service_level,
        "capacity_units": capacity,
        "fill_capacity": False,
        "model_version": ACTIVE_MODEL_VERSION,
        "dataset_mode": ACTIVE_DATASET_MODE,
        "snapshot": FEATURED_SNAPSHOT_PATH.name,
        "summary": {
            "total_forecast": round(total_forecast, 2),
            "total_orders": total_orders,
        },
        "results": results,
    }
