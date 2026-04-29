import os
import time

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="API Health", page_icon="❤️", layout="wide")
st.title("❤️ API Health Monitor")

auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    time.sleep(30)
    st.rerun()

if st.sidebar.button("Refresh Now", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


def check_health() -> dict:
    start = time.perf_counter()
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        if r.status_code == 200:
            data = r.json().get("data", {})
            data["latency_ms"] = latency_ms
            data["status_code"] = 200
            return data
        return {"api": "error", "status_code": r.status_code, "latency_ms": latency_ms}
    except Exception as e:
        return {"api": "unreachable", "error": str(e), "latency_ms": None, "status_code": None}


health = check_health()

col1, col2, col3, col4 = st.columns(4)

def status_badge(val: str) -> str:
    return "✅" if val == "ok" else "❌"

with col1:
    api_s = health.get("api", "unknown")
    st.metric("API", f"{status_badge(api_s)} {api_s.upper()}")
with col2:
    db_s = health.get("database", "unknown")
    st.metric("Database", f"{status_badge(db_s)} {db_s.upper()}")
with col3:
    redis_s = health.get("redis", "unknown")
    st.metric("Redis", f"{status_badge(redis_s)} {redis_s.upper()}")
with col4:
    lat = health.get("latency_ms")
    st.metric("Latency", f"{lat} ms" if lat else "N/A")

st.divider()

if health.get("api") == "unreachable":
    st.error(f"API is unreachable: {health.get('error')}")
    st.info(f"Make sure the API is running at `{API_BASE}`")
    st.stop()

st.subheader("Response Time History")

if "latency_history" not in st.session_state:
    st.session_state["latency_history"] = []

lat = health.get("latency_ms")
if lat is not None:
    st.session_state["latency_history"].append(lat)
    if len(st.session_state["latency_history"]) > 50:
        st.session_state["latency_history"] = st.session_state["latency_history"][-50:]

history = st.session_state["latency_history"]
if len(history) > 1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history,
        mode="lines+markers",
        line=dict(color="#636EFA", width=2),
        name="Latency (ms)",
    ))
    fig.update_layout(
        title="API Response Time (last 50 checks)",
        yaxis_title="Latency (ms)",
        xaxis_title="Check #",
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Latency history builds up as you keep this page open or enable auto-refresh.")

st.subheader("Endpoint Spot-Check")

endpoints = [
    ("GET", "/health", None, False),
    ("GET", "/models", None, True),
]

results = []
for method, path, body, auth in endpoints:
    h = HEADERS if auth else {}
    t0 = time.perf_counter()
    try:
        if method == "GET":
            r = httpx.get(f"{API_BASE}{path}", headers=h, timeout=5)
        else:
            r = httpx.post(f"{API_BASE}{path}", headers=h, json=body, timeout=5)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        results.append({"Endpoint": f"{method} {path}", "Status": r.status_code, "Latency (ms)": ms, "OK": "✅" if r.status_code < 400 else "❌"})
    except Exception as e:
        results.append({"Endpoint": f"{method} {path}", "Status": "Error", "Latency (ms)": "—", "OK": "❌"})

st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
