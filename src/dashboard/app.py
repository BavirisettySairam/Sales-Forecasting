import os
import sys

import httpx
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Auto-rename pages to fix sidebar casing (since Streamlit uses filenames)
_pages_dir = os.path.join(os.path.dirname(__file__), "pages")
_renames = {
    "01_forecast.py": "01_Forecast_Explorer.py",
    "02_model_comparison.py": "02_Model_Comparison.py",
    "03_training_history.py": "03_Training_History.py",
    "04_api_health.py": "04_API_Health.py",
    "05_about.py": "05_About_the_Platform.py"
}
for _old, _new in _renames.items():
    _old_path = os.path.join(_pages_dir, _old)
    if os.path.exists(_old_path):
        try:
            os.rename(_old_path, os.path.join(_pages_dir, _new))
        except Exception:
            pass
from theme import (  # noqa: E402
    C_ACCENT,
    C_BORDER,
    C_DANGER,
    C_PRIMARY,
    C_SAGE,
    C_SURFACE,
    C_TEXT,
    C_TEXT_LITE,
    callout,
    fmt_large,
    inject_css,
    kpi,
    section_label,
)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()


@st.cache_data(ttl=300)
def fetch_models() -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    try:
        resp = httpx.get(f"{API_BASE}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json().get("data", {})
    except Exception:
        pass
    return {}


def region_name(model: dict) -> str:
    return model.get("state") or "Legacy global"


with st.sidebar:
    st.markdown("### Controls")
    st.caption(f"API: {API_BASE}")
    if st.button("Refresh", type="primary", width="stretch"):
        st.cache_data.clear()
        st.rerun()

models = fetch_models()
health = fetch_health()
api_ok = health.get("api") == "ok"
db_ok = health.get("database") == "ok"

regional_models = [m for m in models if m.get("state")]
champions = [m for m in regional_models if m.get("is_champion")]
versions = sorted({m.get("version", "") for m in models if m.get("version")})
regions = sorted({m["state"] for m in regional_models})

st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, {C_PRIMARY} 0%, #1E293B 100%);
    padding: 3rem 2rem; border-radius: 16px; margin-bottom: 2.5rem; color: white;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.1),
    0 8px 10px -6px rgba(0,0,0,0.1);">
        <h1 style="font-family: 'Fraunces', serif; font-size: 2.8rem;
        font-weight: 300; margin: 0 0 0.8rem 0; color: white !important;">
        Sales Intelligence Platform</h1>
        <p style="font-family: 'Inter', sans-serif; font-size: 1.1rem;
        opacity: 0.95; margin: 0; max-width: 650px; line-height: 1.6;">
            Advanced machine learning forecasting with automated champion selection
            across all U.S. regions. Five model families competing per region
            to deliver unparalleled accuracy.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not api_ok:
    st.error(
        "API is unavailable. Start the FastAPI service before using the dashboard."
    )
elif not db_ok:
    st.warning("API is up, but the database is degraded.")

mape_vals = [
    (m.get("metrics") or {}).get("mape")
    for m in regional_models
    if (m.get("metrics") or {}).get("mape") not in (None, float("inf"))
]
test_vals = [
    (m.get("metrics") or {}).get("test_mape")
    for m in regional_models
    if (m.get("metrics") or {}).get("test_mape") not in (None, float("inf"))
]

c1, c2, c3, c4, c5 = st.columns(5)
for col, label, value, sub, color in [
    (c1, "Regions", str(len(regions)), "trained regional series", C_TEXT),
    (c2, "Candidates", str(len(regional_models)), "all region/model metrics", C_TEXT),
    (c3, "Champions", str(len(champions)), "production artifacts", C_PRIMARY),
    (
        c4,
        "Best CV MAPE",
        f"{min(mape_vals):.3f}%" if mape_vals else "-",
        "lower is better",
        C_SAGE,
    ),
    (
        c5,
        "Best Test MAPE",
        f"{min(test_vals):.3f}%" if test_vals else "-",
        "held-out 15%",
        C_ACCENT,
    ),
]:
    with col:
        st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)

st.divider()

if not regional_models:
    callout(
        "No regional models are registered yet. Start the all-region training job; "
        "legacy National entries are ignored by the forecast workflow.",
        color=C_DANGER,
    )
    st.code(
        ".\\gcc_env\\python.exe -m src.pipeline.train --data data.csv "
        "--all-states --cv-splits 3",
        language="powershell",
    )
    st.stop()

section_label("Champion Models By Region")
champ_rows = []
for model in sorted(champions, key=lambda m: region_name(m)):
    metrics = model.get("metrics") or {}
    champ_rows.append(
        {
            "Region": region_name(model),
            "Champion": model["name"].upper(),
            "CV MAPE %": metrics.get("mape"),
            "Test MAPE %": metrics.get("test_mape"),
            "RMSE": metrics.get("rmse"),
            "MAE": metrics.get("mae"),
            "Version": model.get("version"),
            "Artifact": "ready" if model.get("path") else "metrics only",
        }
    )

champ_df = pd.DataFrame(champ_rows)
if champ_df.empty:
    st.info("Metrics exist, but no regional champion has been marked yet.")
else:
    display = champ_df.copy()
    for col in ["CV MAPE %", "Test MAPE %"]:
        display[col] = display[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "-")
    display["RMSE"] = display["RMSE"].map(fmt_large)
    display["MAE"] = display["MAE"].map(fmt_large)
    st.dataframe(display, width="stretch", hide_index=True)

st.divider()
section_label("Registry Summary")

summary_cols = st.columns(4)
summary = [
    ("Model types", ", ".join(sorted({m["name"].upper() for m in regional_models}))),
    ("Training versions", str(len(versions))),
    ("API", "ok" if api_ok else "down"),
]
for col, (label, value) in zip(summary_cols, summary):
    with col:
        st.markdown(
            f"<div style='background:{C_SURFACE};border:1px solid {C_BORDER};"
            f"border-radius:8px;padding:0.85rem 1rem;'>"
            f"<p style='font-size:0.68rem;color:{C_TEXT_LITE};margin:0;'>{label}</p>"
            f"<p style='font-size:0.92rem;color:{C_TEXT};margin:0.25rem 0 0;'>"
            f"{value or '-'}</p></div>",
            unsafe_allow_html=True,
        )
