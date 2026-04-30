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
    C_GOLD,
    C_PRIMARY,
    C_SAGE,
    C_TEXT,
    C_TEXT_LITE,
    apply_theme,
    callout,
    fmt_large,
    inject_css,
    kpi,
    page_header,
    section_label,
)

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "forecasting-api-key-2026")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="Training History", page_icon="clipboard", layout="wide")
inject_css()


@st.cache_data(ttl=120)
def get_models() -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


def request_retrain(states: list[str] | None) -> tuple[bool, str]:
    try:
        resp = httpx.post(
            f"{API_BASE}/retrain",
            headers=HEADERS,
            json={"states": states},
            timeout=10,
        )
        body = resp.json()
        if resp.status_code == 200:
            return True, body.get("message", "Retraining queued.")
        return False, body.get("message") or body.get("detail") or resp.text
    except Exception as exc:
        return False, str(exc)


models = [m for m in get_models() if m.get("state")]

page_header(
    "Training History",
    "Regional candidate metrics, champion artifacts, and background retraining.",
)

if not models:
    callout(
        "No regional training runs found. Queue all-region training to create "
        "per-region metrics and champions.",
        color=C_ACCENT,
    )
else:
    records = []
    for model in models:
        metrics = model.get("metrics") or {}
        records.append(
            {
                "Region": model["state"],
                "Model": model["name"].upper(),
                "Version": model.get("version"),
                "CV MAPE %": metrics.get("mape"),
                "Test MAPE %": metrics.get("test_mape"),
                "RMSE": metrics.get("rmse"),
                "MAE": metrics.get("mae"),
                "Folds": metrics.get("n_folds"),
                "Champion": bool(model.get("is_champion")),
                "Artifact": model.get("path") or "",
            }
        )
    df = pd.DataFrame(records)

    with st.sidebar:
        st.markdown("### Filters")
        regions = sorted(df["Region"].dropna().unique().tolist())
        region = st.selectbox("Region", ["All"] + regions)
        champion_only = st.checkbox("Champions only", value=False)
        if st.button("Refresh", width="stretch"):
            st.cache_data.clear()
            st.rerun()

    view = df.copy()
    if region != "All":
        view = view[view["Region"] == region]
    if champion_only:
        view = view[view["Champion"]]

    paired = view.dropna(subset=["CV MAPE %", "Test MAPE %"])
    best_cv = view["CV MAPE %"].min() if view["CV MAPE %"].notna().any() else None
    best_test = (
        view["Test MAPE %"].min() if view["Test MAPE %"].notna().any() else None
    )
    gap = (
        (paired["Test MAPE %"] - paired["CV MAPE %"]).mean()
        if not paired.empty
        else None
    )

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, sub, color in [
        (k1, "Rows", str(len(view)), "current filter", C_TEXT),
        (
            k2,
            "Best CV",
            f"{best_cv:.3f}%" if best_cv is not None else "-",
            "selection",
            C_PRIMARY,
        ),
        (
            k3,
            "Best Test",
            f"{best_test:.3f}%" if best_test is not None else "-",
            "held-out",
            C_SAGE,
        ),
        (
            k4,
            "Avg Gap",
            f"{gap:+.3f}%" if gap is not None else "-",
            "test - cv",
            C_DANGER if gap and gap > 2 else C_ACCENT,
        ),
    ]:
        with col:
            st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)

    st.divider()
    tab_table, tab_gap, tab_retrain = st.tabs(["Runs", "CV vs Test", "Retrain"])

    with tab_table:
        display = view.sort_values(["Region", "CV MAPE %"], na_position="last").copy()
        display["Champion"] = display["Champion"].map(lambda v: "yes" if v else "")
        display["Artifact"] = display["Artifact"].map(lambda v: "ready" if v else "")
        for col in ["CV MAPE %", "Test MAPE %"]:
            display[col] = display[col].map(
                lambda v: f"{v:.3f}" if pd.notna(v) else "-"
            )
        display["RMSE"] = display["RMSE"].map(fmt_large)
        display["MAE"] = display["MAE"].map(fmt_large)
        st.dataframe(display, width="stretch", hide_index=True)

    with tab_gap:
        if paired.empty:
            st.info("No rows with both CV and test metrics.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    name="CV MAPE %",
                    x=paired["Region"] + " / " + paired["Model"],
                    y=paired["CV MAPE %"],
                    marker_color=C_PRIMARY,
                )
            )
            fig.add_trace(
                go.Bar(
                    name="Test MAPE %",
                    x=paired["Region"] + " / " + paired["Model"],
                    y=paired["Test MAPE %"],
                    marker_color=C_ACCENT,
                )
            )
            apply_theme(
                fig,
                height=430,
                title="CV MAPE vs Held-Out Test MAPE",
                ylabel="MAPE %",
                showlegend=True,
            )
            fig.update_layout(barmode="group")
            fig.update_xaxes(tickangle=-30)
            st.plotly_chart(fig, width="stretch")
            callout(
                "Large positive gaps mean the model looked better in CV than on "
                "the newest held-out weeks.",
                color=C_ACCENT,
            )

    with tab_retrain:
        section_label("Queue Background Training")
        st.write(
            "Leave the region blank to train every region. The job trains all five "
            "model families per region, selects the best by CV MAPE, then refits "
            "only that champion on the complete regional history."
        )
        with st.form("retrain_form"):
            region_input = st.text_input("Region", placeholder="California")
            submitted = st.form_submit_button("Start Retraining", type="primary")
        if submitted:
            states = [region_input.strip()] if region_input.strip() else None
            ok, message = request_retrain(states)
            if ok:
                st.success(message)
            else:
                st.error(message)

        st.divider()
        section_label("Pipeline")
        steps = [
            ("1", "Drop constant Category column and clean missing dates/values."),
            ("2", "Create lag, rolling, calendar, holiday, and trend features."),
            ("3", "Split chronologically into 70% train, 15% validation, 15% test."),
            ("4", "Train SARIMA, Prophet, XGBoost, LightGBM, and LSTM candidates."),
            ("5", "Tune candidate params where supported by auto_arima or Optuna."),
            ("6", "Select the best region model by CV MAPE, RMSE tiebreaker."),
            ("7", "Refit the champion on the full regional history and save it."),
        ]
        for number, text in steps:
            st.markdown(
                f"<p style='color:{C_TEXT_LITE};margin:0.25rem 0;'>"
                f"<strong style='color:{C_GOLD};'>{number}.</strong> {text}</p>",
                unsafe_allow_html=True,
            )
