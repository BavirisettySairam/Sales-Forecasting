import os

import httpx
import pandas as pd
import plotly.express as px
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Training History", page_icon="📋", layout="wide")
st.title("📋 Training History")


@st.cache_data(ttl=60)
def get_models():
    try:
        r = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


models = get_models()

if not models:
    st.info("No training history found. Run `make train` to train models.")
    st.stop()

records = []
for m in models:
    metrics = m.get("metrics", {})
    records.append(
        {
            "Model": m["name"],
            "State": m.get("state") or "all",
            "Version": m.get("version", "—"),
            "MAPE %": round(metrics.get("mape", 0), 2),
            "RMSE": round(metrics.get("rmse", 0), 2),
            "MAE": round(metrics.get("mae", 0), 2),
            "CV Folds": metrics.get("n_folds", "—"),
            "Champion": "👑" if m.get("is_champion") else "",
            "Path": m.get("path", "—"),
        }
    )

df = pd.DataFrame(records)

with st.sidebar:
    st.header("Filters")
    state_options = ["All"] + sorted(df["State"].unique().tolist())
    selected_state = st.selectbox("State", state_options)
    model_options = ["All"] + sorted(df["Model"].unique().tolist())
    selected_model = st.selectbox("Model", model_options)

filtered = df.copy()
if selected_state != "All":
    filtered = filtered[filtered["State"] == selected_state]
if selected_model != "All":
    filtered = filtered[filtered["Model"] == selected_model]

st.subheader(f"Training Runs ({len(filtered)} results)")
st.dataframe(
    filtered.drop(columns=["Path"]).sort_values("MAPE %"),
    use_container_width=True,
    hide_index=True,
)

if len(df) > 1 and df["MAPE %"].notna().any():
    st.subheader("MAPE by Model and State")
    chart_df = df[df["MAPE %"] > 0].copy()
    if not chart_df.empty:
        fig = px.bar(
            chart_df.sort_values("MAPE %"),
            x="Model",
            y="MAPE %",
            color="State",
            barmode="group",
            title="Cross-Validation MAPE by Model",
            labels={"MAPE %": "Avg MAPE (%)"},
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("View model file paths"):
    st.dataframe(
        df[["Model", "State", "Version", "Path"]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("Trigger Retraining")
with st.form("retrain_form"):
    retrain_state = st.text_input("State to retrain (leave blank for all)", value="")
    submitted = st.form_submit_button("Start Retraining", type="primary")

if submitted:
    payload = {"states": [retrain_state.strip()] if retrain_state.strip() else None}
    try:
        r = httpx.post(f"{API_BASE}/retrain", headers=HEADERS, json=payload, timeout=10)
        if r.status_code == 200:
            st.success(
                "Retraining started in background. Refresh this page in a few minutes."
            )
        else:
            st.error(f"Retrain failed: {r.json().get('message', r.status_code)}")
    except Exception as e:
        st.error(f"Request failed: {e}")
