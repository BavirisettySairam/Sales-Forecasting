import os

import httpx
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")

HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Sales Forecasting System")
st.markdown("Production-ready time series forecasting with automatic model selection")

st.sidebar.header("Navigation")
st.sidebar.info(
    "Use the pages in the sidebar to explore forecasts, model comparisons, training history, and API health."  # noqa: E501
)


@st.cache_data(ttl=300)
def fetch_models():
    try:
        r = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def fetch_health():
    try:
        r = httpx.get(f"{API_BASE}/health", timeout=5)
        if r.status_code == 200:
            return r.json().get("data", {})
    except Exception:
        pass
    return {}


models = fetch_models()
health = fetch_health()


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


col1, col2, col3, col4 = st.columns(4)

champion_count = len([m for m in models if m.get("is_champion")])
total_models = len(models)
mape_values = [
    (m.get("metrics") or {}).get("mape")
    for m in models
]
mape_values = [v for v in mape_values if v is not None and v < float("inf")]
avg_mape = round(sum(mape_values) / len(mape_values), 2) if mape_values else None

with col1:
    st.metric("Total Models", total_models)
with col2:
    st.metric("Champion Models", champion_count)
with col3:
    st.metric("Avg MAPE", f"{avg_mape}%" if avg_mape else "N/A")
with col4:
    api_status = health.get("api", "unknown")
    st.metric("API Status", "✅ Healthy" if api_status == "ok" else "⚠️ Degraded")

st.divider()

if not models:
    st.info("No trained models found. Run `make train` to train models, then refresh.")
else:
    st.subheader("Champion Models")
    champions = [m for m in models if m.get("is_champion")]
    if champions:
        champ_data = [
            {
                "Model": m["name"],
                "State": m.get("state") or "National",
                "MAPE %": round((m.get("metrics") or {}).get("mape", 0), 2),
                "RMSE": _fmt_large((m.get("metrics") or {}).get("rmse")),
                "MAE": _fmt_large((m.get("metrics") or {}).get("mae")),
                "Version": m.get("version", "—"),
            }
            for m in champions
        ]
        st.dataframe(champ_data, use_container_width=True)
    else:
        st.info("No champion model selected yet.")
