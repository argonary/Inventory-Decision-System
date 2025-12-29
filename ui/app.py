import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =====================================================
# App Config
# =====================================================

API_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_STORE = 44
DEFAULT_DATE = "2016-04-21"
DEFAULT_SERVICE_LEVEL = 0.90
DEFAULT_CAPACITY = 300

st.set_page_config(
    page_title="Inventory Decision System",
    layout="wide",
)

st.title("📦 Inventory Ordering Decision Tool")
st.caption(
    "Quantile-based inventory decisions with hard capacity constraints "
    "(Streamlit client → FastAPI backend)"
)

# =====================================================
# Sidebar — API Status
# =====================================================

with st.sidebar:
    st.header("API Status")

    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=2).json()
        version = requests.get(f"{API_BASE_URL}/version", timeout=2).json()

        st.success("API is running")
        st.caption(f"Model version: {version['model_version']}")
        st.caption(f"Dataset mode: {version['dataset_mode']}")
        st.caption(f"Snapshot: {version['snapshot']}")

    except Exception:
        st.error("API not reachable")
        st.stop()

# =====================================================
# Sidebar — Decision Inputs
# =====================================================

with st.sidebar:
    st.header("Decision Inputs")

    store_nbr = st.number_input(
        "Store number",
        min_value=1,
        value=DEFAULT_STORE,
    )

    date = st.text_input(
        "Decision date (YYYY-MM-DD)",
        value=DEFAULT_DATE,
    )

    service_level = st.selectbox(
        "Service level",
        options=[0.90, 0.95],
        format_func=lambda x: f"P{int(x*100)}",
        index=0,
    )

    st.divider()
    st.subheader("Capacity Policy")

    capacity_units = st.number_input(
        "Total capacity (units)",
        min_value=1,
        value=DEFAULT_CAPACITY,
        step=25,
    )

    st.divider()
    st.subheader("SKU Input")

    if "items_df" not in st.session_state:
        st.session_state["items_df"] = None

    uploaded_csv = st.file_uploader(
        "Upload SKU payload (CSV)",
        type=["csv"],
        help="CSV must contain columns: item_nbr, onpromotion",
    )

    if st.button("Load demo payload"):
        demo_df = pd.read_csv("ui/test_payload.csv")
        st.session_state["items_df"] = demo_df

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        if not {"item_nbr", "onpromotion"}.issubset(df.columns):
            st.error("CSV must contain columns: item_nbr, onpromotion")
            st.stop()
        st.session_state["items_df"] = df.drop_duplicates("item_nbr")

    items_df = st.session_state.get("items_df")

    run_decision = st.button("Run forecast → orders", type="primary")

# =====================================================
# Show SKU payload
# =====================================================

if items_df is None:
    st.info("Upload a CSV or click **Load demo payload** to begin.")
else:
    st.subheader("SKU Payload")
    st.dataframe(items_df, use_container_width=True)

# =====================================================
# Run Decision
# =====================================================

if run_decision:

    if items_df is None or items_df.empty:
        st.error("No SKUs provided.")
        st.stop()

    payload = {
        "store_nbr": int(store_nbr),
        "date": date,
        "service_level": float(service_level),
        "capacity_units": int(capacity_units),
        "items": items_df.to_dict(orient="records"),
    }

    try:
        with st.spinner("Calling forecasting API..."):
            r = requests.post(
                f"{API_BASE_URL}/forecast-to-orders",
                json=payload,
                timeout=10,
            )

        if r.status_code != 200:
            st.error("API returned an error")
            st.code(r.text)
            st.stop()

        response = r.json()

    except Exception as e:
        st.error("Failed to call API")
        st.exception(e)
        st.stop()

    df = pd.DataFrame(response["results"])
    total_forecast = response["summary"]["total_forecast"]
    total_orders = response["summary"]["total_orders"]

    st.session_state.update(
        {
            "df": df,
            "total_forecast": total_forecast,
            "total_orders": total_orders,
            "capacity": capacity_units,
            "payload": payload,
        }
    )

    # =====================================================
    # Capacity Stress Test (safe demo loop)
    # =====================================================

    cap_grid = np.linspace(
        int(0.3 * total_forecast),
        int(1.2 * total_forecast),
        30,
    ).astype(int)

    coverages = []

    for cap in cap_grid:
        test_payload = payload | {"capacity_units": int(cap)}
        rr = requests.post(
            f"{API_BASE_URL}/forecast-to-orders",
            json=test_payload,
            timeout=10,
        )
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
        "⬇ Download order table (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "order_recommendations.csv",
        "text/csv",
    )

    # =====================================================
    # Capacity Curve
    # =====================================================

    cap_grid = st.session_state["cap_grid"]
    coverages = st.session_state["coverages"]

    target = st.slider(
        "Target demand served (%)",
        50,
        100,
        90,
    ) / 100

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
        title="Capacity vs Demand Served",
        xaxis_title="Capacity (Units)",
        yaxis_title="Fraction of Demand Served",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
    )

    st.plotly_chart(fig, use_container_width=True)

    required_capacity = next(
        cap for cap, cov in zip(cap_grid, coverages) if cov >= target
    )

    st.metric(
        "Minimum Capacity Required (Guaranteed)",
        f"{required_capacity} units",
    )

    st.caption(
        "Coverage is stepwise due to integer SKU allocations. "
        "The metric shows the smallest capacity that guarantees the target service level."
    )
