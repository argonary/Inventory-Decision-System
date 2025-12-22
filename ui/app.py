import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================================
# Config
# =====================================================

API_URL = "http://127.0.0.1:8000/forecast-to-orders"
FEATURE_PATH = "data/snapshots/favorita_test_featured_2016Q1.parquet"

st.set_page_config(page_title="Inventory Decision System", layout="wide")

st.title("📦 Inventory Ordering Decision Tool")
st.caption(
    "Quantile-based inventory decisions with hard capacity constraints "
    "(served via FastAPI)"
)

# =====================================================
# Load feature data (for UI only)
# =====================================================

@st.cache_data
def load_feature_data():
    df = pd.read_parquet(FEATURE_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df

df_features = load_feature_data()

# =====================================================
# Sidebar — Inputs
# =====================================================

with st.sidebar:
    st.header("Decision Inputs")

    date = st.date_input(
        "Order Date",
        value=df_features["date"].min().date(),
        min_value=df_features["date"].min().date(),
        max_value=df_features["date"].max().date(),
    )

    valid_stores = sorted(
        df_features[df_features["date"].dt.date == date]["store_nbr"].unique()
    )
    store_nbr = st.selectbox("Store Number", valid_stores)

    service_level = st.selectbox(
        "Service Level",
        options=[0.90, 0.95],
        format_func=lambda x: f"P{int(x*100)}",
    )

    st.divider()
    st.subheader("Capacity Policy")

    capacity_units = st.number_input(
        "Total Capacity (Units)",
        min_value=1,
        value=300,
        step=25,
    )

    fill_capacity = st.checkbox(
        "Force fill capacity (allow orders > forecast)",
        value=False,
        help="Unchecked = never exceed per-SKU forecast (recommended)",
    )

    st.divider()
    st.subheader("SKU Input")

    uploaded_csv = st.file_uploader("Upload SKU list (CSV)", type=["csv"])

    items = []

    if uploaded_csv:
        csv_df = pd.read_csv(uploaded_csv)
        if not {"item_nbr", "onpromotion"}.issubset(csv_df.columns):
            st.error("CSV must contain columns: item_nbr, onpromotion")
            st.stop()

        csv_df = csv_df.drop_duplicates(subset=["item_nbr"])

        for _, r in csv_df.iterrows():
            items.append(
                {
                    "item_nbr": int(r["item_nbr"]),
                    "onpromotion": bool(r["onpromotion"]),
                }
            )

    else:
        store_day_df = df_features[
            (df_features["date"].dt.date == date)
            & (df_features["store_nbr"] == store_nbr)
        ]

        valid_items = sorted(store_day_df["item_nbr"].unique())
        selected_items = st.multiselect(
            "Items (SKUs)",
            valid_items,
            valid_items[:8],
        )

        for sku in selected_items:
            promo = st.checkbox(
                f"SKU {sku} on promotion",
                key=f"promo_{sku}",
            )
            items.append(
                {
                    "item_nbr": int(sku),
                    "onpromotion": bool(promo),
                }
            )

    run_decision = st.button("Run Decision", type="primary")

# =====================================================
# Run Decision
# =====================================================

if run_decision:
    if not items:
        st.error("No SKUs selected.")
        st.stop()

    payload = {
        "store_nbr": int(store_nbr),
        "date": pd.to_datetime(date).isoformat(),
        "service_level": float(service_level),
        "capacity_units": int(capacity_units),
        "fill_capacity": bool(fill_capacity),
        "items": items,
    }

    with st.spinner("Calling Forecast-to-Orders API..."):
        r = requests.post(API_URL, json=payload)

    if r.status_code != 200:
        st.error("API Error")
        st.code(r.text)
        st.stop()

    response = r.json()
    df = pd.DataFrame(response["results"])

    total_forecast = response["summary"]["total_forecast"]
    total_orders = response["summary"]["total_orders"]

    st.session_state["df"] = df
    st.session_state["total_forecast"] = total_forecast
    st.session_state["total_orders"] = total_orders
    st.session_state["capacity"] = capacity_units
    st.session_state["payload"] = payload

    # ----------------------------
    # Capacity stress test
    # ----------------------------

    cap_grid = np.linspace(
        int(0.3 * total_forecast),
        int(1.2 * total_forecast),
        40,
    ).astype(int)

    coverages = []

    for cap in cap_grid:
        test_payload = payload.copy()
        test_payload["capacity_units"] = int(cap)

        rr = requests.post(API_URL, json=test_payload)
        tmp = pd.DataFrame(rr.json()["results"])
        coverages.append(tmp["order_qty"].sum() / total_forecast)

    st.session_state["cap_grid"] = cap_grid
    st.session_state["coverages"] = coverages

# =====================================================
# Display Results
# =====================================================

if "df" in st.session_state:
    df = st.session_state["df"]
    total_forecast = st.session_state["total_forecast"]
    total_orders = st.session_state["total_orders"]
    current_capacity = st.session_state["capacity"]

    current_coverage = total_orders / total_forecast

    st.subheader("Order Recommendations")

    st.info(
        f"""
**Outcome Summary**
- Total Forecast: **{total_forecast:.1f} units**
- Total Ordered: **{total_orders} units**
- Demand Served: **{current_coverage:.1%}**
"""
    )

    st.dataframe(
        df.rename(
            columns={
                "item_nbr": "Item",
                "forecast": "Forecast",
                "order_qty": "Order Qty",
            }
        ),
        use_container_width=True,
    )

    st.download_button(
        "⬇ Download Order Table (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "order_recommendations.csv",
        "text/csv",
    )

    # =====================================================
    # Capacity Curve
    # =====================================================

    cap_grid = st.session_state["cap_grid"]
    coverages = st.session_state["coverages"]

    target = st.slider("Target Demand Served (%)", 50, 100, 90) / 100

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cap_grid,
            y=coverages,
            mode="lines+markers",
            line=dict(shape="hv"),
            name="Demand Served",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[current_capacity],
            y=[current_coverage],
            mode="markers",
            marker=dict(size=12, color="orange"),
            name="Current Capacity",
        )
    )

    fig.add_hline(
        y=target,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Target ({int(target*100)}%)",
    )

    fig.update_layout(
        title="Capacity vs Demand Served (Stepwise)",
        xaxis_title="Capacity (Units)",
        yaxis_title="Fraction of Demand Served",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # Guaranteed Capacity Metric
    # =====================================================

    required_capacity = next(
        cap for cap, cov in zip(cap_grid, coverages) if cov >= target
    )

    st.metric(
        "Minimum Capacity Required (Guaranteed)",
        f"{required_capacity} units",
    )

    st.caption(
        "Coverage is stepwise due to integer SKU allocations. "
        "The metric shows the smallest capacity that guarantees the target service."
    )
