import os
import sys

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theme import (  # noqa: E402
    C_ACCENT,
    C_DANGER,
    C_PRIMARY,
    C_SAGE,
    C_TEXT,
    apply_theme,
    callout,
    fmt_large,
    hex_rgba,
    inject_css,
    kpi,
    page_header,
    section_label,
)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Forecast", page_icon="bar_chart", layout="wide")
inject_css()


@st.cache_data(ttl=300)
def get_regions() -> list[str]:
    try:
        resp = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            return sorted({m["state"] for m in models if m.get("state")})
    except Exception:
        pass
    return []


@st.cache_data(ttl=86400, show_spinner="Loading regional champion...")
def get_forecast(region: str, weeks: int) -> dict:
    try:
        resp = httpx.post(
            f"{API_BASE}/forecast",
            headers=HEADERS,
            json={"state": region, "weeks": weeks},
            timeout=60,
        )
        body = resp.json()
        if resp.status_code == 200:
            return body.get("data", {})
        return {"error": body.get("message") or body.get("detail") or resp.text}
    except Exception as exc:
        return {"error": str(exc)}


regions = get_regions()

with st.sidebar:
    st.markdown("### Forecast Settings")
    if regions:
        selected_region = st.selectbox("Region", regions)
    else:
        selected_region = None
        st.info("No trained regional champions yet.")
    weeks = st.slider("Horizon (weeks)", min_value=1, max_value=10, value=8)
    run = st.button("Generate Forecast", type="primary", width="stretch")
    if st.button("Refresh Registry", width="stretch"):
        st.cache_data.clear()
        st.rerun()

page_header(
    "Forecast Explorer",
    "Next-week sales forecast from the selected region's full-history champion.",
)

if not regions:
    callout(
        "No regional champion models are available yet. Run the all-region "
        "training job first; legacy National models are intentionally ignored.",
        color=C_DANGER,
    )
    st.stop()

should_fetch = (
    run
    or st.session_state.get("fc_region") != selected_region
    or st.session_state.get("fc_weeks") != weeks
    or "fc_data" not in st.session_state
)
if should_fetch:
    data = get_forecast(selected_region, weeks)
    st.session_state.update(fc_data=data, fc_region=selected_region, fc_weeks=weeks)
else:
    data = st.session_state["fc_data"]

if data.get("error"):
    st.error(f"Forecast failed: {data['error']}")
    st.stop()

forecast = pd.DataFrame(data.get("forecast", []))
if forecast.empty:
    st.info("Click Generate Forecast to request predictions from the API.")
    st.stop()

forecast["date"] = pd.to_datetime(forecast["date"])
forecast["width"] = forecast["upper_bound"] - forecast["lower_bound"]
forecast["relative_width"] = (
    forecast["width"] / forecast["predicted_value"].replace(0, pd.NA) * 100
)

model_used = data.get("model_used", "-").upper()
model_mape = data.get("model_mape")
region = data.get("state") or selected_region

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub in [
    (k1, "Region", region, "selected series"),
    (k2, "Champion", model_used, "full-history artifact"),
    (
        k3,
        "CV MAPE",
        f"{model_mape:.3f}%" if model_mape is not None else "-",
        "selection score",
    ),
    (k4, "Horizon", f"{weeks} weeks", "API request"),
]:
    with col:
        st.markdown(kpi(label, value, sub), unsafe_allow_html=True)

st.divider()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=forecast["date"],
        y=forecast["upper_bound"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast["date"],
        y=forecast["lower_bound"],
        mode="lines",
        fill="tonexty",
        fillcolor=hex_rgba(C_PRIMARY, 0.12),
        line=dict(width=0),
        name="95% Interval",
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast["date"],
        y=forecast["predicted_value"],
        mode="lines+markers",
        line=dict(color=C_PRIMARY, width=2.5),
        marker=dict(size=7, color=C_PRIMARY),
        name="Forecast",
        hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
    )
)
apply_theme(
    fig,
    height=460,
    title=f"{region} – {weeks}-Week Forecast",
    xlabel="Week",
    ylabel="Sales",
    showlegend=True,
)
st.plotly_chart(fig, width="stretch")

pred = forecast["predicted_value"]
lo = forecast["lower_bound"]
hi = forecast["upper_bound"]
s1, s2, s3, s4 = st.columns(4)
for col, label, value, sub, color in [
    (s1, "Total forecast", fmt_large(pred.sum(), 1), "cumulative", C_PRIMARY),
    (s2, "Weekly average", fmt_large(pred.mean(), 1), "mean forecast", C_SAGE),
    (s3, "Peak week", fmt_large(pred.max(), 1), "highest prediction", C_ACCENT),
    (s4, "Avg interval", fmt_large(((hi - lo) / 2).mean(), 1), "half-width", C_TEXT),
]:
    with col:
        st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)

st.divider()
tab_table, tab_uncertainty = st.tabs(["Forecast Table", "Uncertainty"])

with tab_table:
    display = forecast[
        ["date", "predicted_value", "lower_bound", "upper_bound", "width"]
    ].copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    for col in ["predicted_value", "lower_bound", "upper_bound", "width"]:
        display[col] = display[col].map("{:,.0f}".format)
    display = display.rename(
        columns={
            "date": "Date",
            "predicted_value": "Predicted",
            "lower_bound": "Lower",
            "upper_bound": "Upper",
            "width": "Interval Width",
        }
    )
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download CSV",
        data=display.to_csv(index=False),
        file_name=f"{region.lower().replace(' ', '_')}_{weeks}w_forecast.csv",
        mime="text/csv",
    )

with tab_uncertainty:
    section_label("Interval Width by Week")
    width_q1 = forecast["width"].quantile(0.33)
    width_q2 = forecast["width"].quantile(0.67)
    colors = [
        C_SAGE if w <= width_q1 else C_ACCENT if w <= width_q2 else C_DANGER
        for w in forecast["width"]
    ]
    fig_width = go.Figure(
        go.Bar(
            x=forecast["date"],
            y=forecast["width"],
            marker_color=colors,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.0f}<extra></extra>",
        )
    )
    apply_theme(
        fig_width,
        height=330,
        xlabel="Week",
        ylabel="Interval Width",
        showlegend=False,
    )
    st.plotly_chart(fig_width, width="stretch")
    callout(
        "Wider intervals mean the champion model is less certain for that week.",
        color=C_ACCENT,
    )
