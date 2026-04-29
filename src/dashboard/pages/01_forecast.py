import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Forecast", page_icon="📊", layout="wide")
st.title("📊 Sales Forecast")


@st.cache_data(ttl=300)
def get_states():
    """Return list of available forecast targets.
    Models with state=None are national-level and shown as 'National'.
    """
    try:
        r = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            models = r.json().get("data", [])
            states = set()
            for m in models:
                s = m.get("state")
                states.add(s if s else "National")
            return sorted(states)
    except Exception:
        pass
    return ["National"]


@st.cache_data(ttl=86400, show_spinner="Generating forecast...")
def get_forecast(state: str, weeks: int):
    # "National" maps to the champion model which is always national-level
    api_state = "national" if state == "National" else state
    try:
        r = httpx.post(
            f"{API_BASE}/forecast",
            headers=HEADERS,
            json={"state": api_state, "weeks": weeks},
            timeout=60,
        )
        if r.status_code == 200:
            return r.json().get("data", {})
        return {"error": r.json().get("message", f"HTTP {r.status_code}")}
    except Exception as e:
        return {"error": str(e)}


states = get_states()

with st.sidebar:
    st.header("Forecast Settings")
    selected_state = st.selectbox("State / Region", states, index=0)
    weeks = st.slider("Forecast horizon (weeks)", min_value=1, max_value=52, value=8)
    run = st.button("Generate Forecast", type="primary", use_container_width=True)

if (
    run
    or "forecast_data" not in st.session_state
    or st.session_state.get("forecast_state") != selected_state
):
    result = get_forecast(selected_state, weeks)
    st.session_state["forecast_data"] = result
    st.session_state["forecast_state"] = selected_state
    st.session_state["forecast_weeks"] = weeks

result = st.session_state.get("forecast_data", {})

if "error" in result:
    st.error(f"Forecast failed: {result['error']}")
    st.stop()

if not result:
    st.info("Click **Generate Forecast** to see predictions.")
    st.stop()

forecast_list = result.get("forecast", [])
model_used = result.get("model_used", "unknown")
model_mape = result.get("model_mape", 0)

st.markdown(
    f"**Model:** `{model_used}` &nbsp;|&nbsp; **MAPE:** `{model_mape:.2f}%` &nbsp;|&nbsp; **State:** `{result.get('state', selected_state)}`"  # noqa: E501
)

fc_df = pd.DataFrame(forecast_list)
if fc_df.empty:
    st.warning("No forecast data returned.")
    st.stop()

fc_df["date"] = pd.to_datetime(fc_df["date"])

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=fc_df["date"],
        y=fc_df["upper_bound"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper CI",
    )
)
fig.add_trace(
    go.Scatter(
        x=fc_df["date"],
        y=fc_df["lower_bound"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(99, 110, 250, 0.2)",
        line=dict(width=0),
        name="95% CI",
    )
)
fig.add_trace(
    go.Scatter(
        x=fc_df["date"],
        y=fc_df["predicted_value"],
        mode="lines+markers",
        line=dict(color="#636EFA", dash="dash", width=2),
        marker=dict(size=6),
        name="Forecast",
    )
)

display_region = result.get("state", selected_state) or selected_state
fig.update_layout(
    title=f"{display_region.title()} — {weeks}-Week Sales Forecast ({model_used})",
    xaxis_title="Date",
    yaxis_title="Total Sales",
    hovermode="x unified",
    legend=dict(orientation="h", y=-0.15),
    height=500,
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast Table")
display_df = fc_df.copy()
display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
display_df = display_df.rename(
    columns={
        "date": "Date",
        "predicted_value": "Predicted",
        "lower_bound": "Lower (95%)",
        "upper_bound": "Upper (95%)",
    }
)
st.dataframe(display_df.set_index("Date"), use_container_width=True)
