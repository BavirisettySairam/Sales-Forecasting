import os

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Model Comparison", page_icon="🏆", layout="wide")
st.title("🏆 Model Comparison")


@st.cache_data(ttl=300)
def get_all_models():
    try:
        r = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


def _fmt_large(n) -> str:
    """Format large numbers as 1.23B / 1.23M / 1.23K for readability."""
    if n is None:
        return "N/A"
    n = float(n)
    if abs(n) >= 1e9:
        return f"{n / 1e9:.2f}B"
    if abs(n) >= 1e6:
        return f"{n / 1e6:.2f}M"
    if abs(n) >= 1e3:
        return f"{n / 1e3:.2f}K"
    return f"{n:.2f}"


models = get_all_models()

if not models:
    st.info("No trained models found. Run `make train` first.")
    st.stop()

records = []
for m in models:
    metrics = m.get("metrics") or {}
    records.append(
        {
            "model": m["name"],
            "state": m.get("state") or "National",
            "mape": metrics.get("mape", None),
            "rmse": metrics.get("rmse", None),
            "mae": metrics.get("mae", None),
            "is_champion": m.get("is_champion", False),
            "version": m.get("version", "—"),
        }
    )

df = pd.DataFrame(records)

states = sorted(df["state"].unique())
with st.sidebar:
    st.header("Filters")
    selected_state = st.selectbox("State", ["all states"] + list(states))
    if st.button("Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

if selected_state != "all states":
    view_df = df[df["state"] == selected_state].copy()
else:
    view_df = df.copy()

st.subheader("MAPE by Model")

if view_df.empty or view_df["mape"].isna().all():
    st.info("No metric data available for the selected state.")
else:
    view_df_sorted = view_df.dropna(subset=["mape"]).sort_values("mape")

    colors = [
        "gold" if row["is_champion"] else "#636EFA"
        for _, row in view_df_sorted.iterrows()
    ]
    labels = [
        f"{'👑 ' if row['is_champion'] else ''}{row['model']}"
        for _, row in view_df_sorted.iterrows()
    ]

    fig_bar = go.Figure(
        go.Bar(
            x=labels,
            y=view_df_sorted["mape"].tolist(),
            marker_color=colors,
            text=[f"{v:.2f}%" for v in view_df_sorted["mape"].tolist()],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="MAPE by Model (lower is better) — 👑 = Champion",
        yaxis_title="MAPE %",
        xaxis_title="Model",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.subheader("Ranked Model Table")
table_df = view_df.dropna(subset=["mape"]).sort_values("mape").reset_index(drop=True)
table_df.insert(0, "Rank", range(1, len(table_df) + 1))
table_df["Champion"] = table_df["is_champion"].apply(lambda x: "👑" if x else "")
display_df = table_df[["Rank", "Champion", "model", "state", "mape", "rmse", "mae", "version"]].copy()
display_df = display_df.rename(columns={"model": "Model", "state": "State", "mape": "MAPE %", "version": "Version"})
display_df["RMSE"] = display_df["rmse"].apply(_fmt_large)
display_df["MAE"] = display_df["mae"].apply(_fmt_large)
display_df = display_df.drop(columns=["rmse", "mae"])
st.dataframe(display_df, use_container_width=True)

st.subheader("State × Model MAPE Heatmap")
if len(df["state"].unique()) > 1 and len(df["model"].unique()) > 1:
    pivot = df.pivot_table(
        index="state", columns="model", values="mape", aggfunc="first"
    )
    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        title="MAPE Heatmap — State × Model (darker = higher error)",
        labels={"color": "MAPE %"},
        aspect="auto",
    )
    fig_heat.update_layout(height=max(400, len(pivot) * 20))
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info(
        "Heatmap requires models trained for multiple states and multiple model types."
    )
