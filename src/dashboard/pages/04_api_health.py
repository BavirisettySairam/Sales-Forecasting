import os
import sys
import time

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
    apply_theme,
    callout,
    hex_rgba,
    inject_css,
    kpi,
    page_header,
    section_label,
)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="API Health", page_icon="heart", layout="wide")
inject_css()


def timed_get(path: str, auth: bool = False) -> dict:
    headers = HEADERS if auth else {}
    started = time.perf_counter()
    try:
        resp = httpx.get(f"{API_BASE}{path}", headers=headers, timeout=5)
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        is_json = resp.headers.get("content-type", "").startswith("application/json")
        return {
            "path": path,
            "status": resp.status_code,
            "latency_ms": elapsed,
            "ok": resp.status_code < 400,
            "body": resp.json() if is_json else {},
        }
    except Exception as exc:
        return {
            "path": path,
            "status": "error",
            "latency_ms": None,
            "ok": False,
            "error": str(exc),
            "body": {},
        }


with st.sidebar:
    st.markdown("### Controls")
    auto_refresh = st.checkbox("Auto-refresh every 30 seconds", value=False)
    if st.button("Refresh Now", type="primary", width="stretch"):
        st.rerun()
    st.caption(f"API base: {API_BASE}")

if auto_refresh:
    time.sleep(30)
    st.rerun()

page_header("System Monitor", "API health, latency, and endpoint spot checks.")

health_check = timed_get("/health")
health = health_check.get("body", {}).get("data", {})
api_ok = health.get("api") == "ok"
db_ok = health.get("database") == "ok"
latency = health_check.get("latency_ms")

c1, c2, c3 = st.columns(3)
for col, label, value, sub, color in [
    (
        c1,
        "API",
        health.get("api", "down").upper(),
        "FastAPI",
        C_SAGE if api_ok else C_DANGER,
    ),
    (
        c2,
        "Database",
        health.get("database", "-").upper(),
        "SQLAlchemy",
        C_SAGE if db_ok else C_DANGER,
    ),
    (c3, "Latency", f"{latency} ms" if latency else "-", "/health", C_PRIMARY),
]:
    with col:
        st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)

if not api_ok:
    st.error(health_check.get("error") or "API is not healthy.")
    callout(f"Start the API service at {API_BASE} before using the dashboard.")
    st.stop()

st.divider()

if "latency_history" not in st.session_state:
    st.session_state.latency_history = []
if latency is not None:
    st.session_state.latency_history.append(latency)
    st.session_state.latency_history = st.session_state.latency_history[-60:]

section_label("Latency History")
history = st.session_state.latency_history
if len(history) > 1:
    fig = go.Figure()
    fig.add_hline(y=300, line=dict(color=hex_rgba(C_SAGE, 0.5), dash="dot"))
    fig.add_hline(y=800, line=dict(color=hex_rgba(C_ACCENT, 0.5), dash="dot"))
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history) + 1)),
            y=history,
            mode="lines+markers",
            line=dict(color=C_PRIMARY, width=2),
            fill="tozeroy",
            fillcolor=hex_rgba(C_PRIMARY, 0.08),
        )
    )
    apply_theme(fig, height=320, xlabel="Check", ylabel="ms", showlegend=False)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("Refresh or enable auto-refresh to build a latency history.")

st.divider()
section_label("Endpoint Spot Checks")
checks = [
    ("GET /health", timed_get("/health")),
    ("GET /models", timed_get("/models", auth=True)),
]
rows = []
for label, result in checks:
    rows.append(
        {
            "Endpoint": label,
            "Status": result["status"],
            "Latency ms": result["latency_ms"] or "-",
            "Result": "PASS" if result["ok"] else "FAIL",
        }
    )
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

st.divider()
section_label("Configuration")
st.dataframe(
    pd.DataFrame(
        [
            {"Key": "API Base URL", "Value": API_BASE},
            {"Key": "Auth", "Value": "X-API-Key"},
            {"Key": "Forecast Scope", "Value": "Regional Champions Only"},
            {"Key": "Train/Val/Test Split", "Value": "70/15/15 Chronological"},
        ]
    ),
    width="stretch",
    hide_index=True,
)
