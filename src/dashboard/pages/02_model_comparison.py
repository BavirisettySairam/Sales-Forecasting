import os
import sys

import httpx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theme import (  # noqa: E402
    C_ACCENT,
    C_GOLD,
    C_PRIMARY,
    C_SAGE,
    C_TEXT,
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

st.set_page_config(page_title="Model Comparison", page_icon="trophy", layout="wide")
inject_css()


@st.cache_data(ttl=300)
def get_models() -> list[dict]:
    try:
        resp = httpx.get(f"{API_BASE}/models", headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("data", [])
    except Exception:
        pass
    return []


models = [m for m in get_models() if m.get("state")]

page_header(
    "Model Comparison",
    "Candidate metrics for every regional model family and champion selection.",
)

if not models:
    callout("No regional model metrics are available yet.", color=C_ACCENT)
    st.stop()

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
            "Artifact": bool(model.get("path")),
        }
    )

df = pd.DataFrame(records)

with st.sidebar:
    st.markdown("### Filters")
    regions = sorted(df["Region"].dropna().unique().tolist())
    versions = sorted(df["Version"].dropna().unique().tolist(), reverse=True)
    region = st.selectbox("Region", ["All"] + regions)
    version = st.multiselect("Version", versions, default=versions[:1] or versions)
    metric = st.selectbox("Rank by", ["CV MAPE %", "Test MAPE %", "RMSE", "MAE"])
    champions_only = st.checkbox("Champions only", value=False)
    if st.button("Refresh", width="stretch"):
        st.cache_data.clear()
        st.rerun()

view = df.copy()
if region != "All":
    view = view[view["Region"] == region]
if version:
    view = view[view["Version"].isin(version)]
if champions_only:
    view = view[view["Champion"]]

ranked = view.dropna(subset=[metric]).sort_values(metric)
best = ranked.iloc[0] if not ranked.empty else None
champ_count = int(view["Champion"].sum()) if not view.empty else 0

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub, color in [
    (k1, "Rows in view", str(len(view)), f"{len(df)} total", C_TEXT),
    (k2, "Champions", str(champ_count), "current production picks", C_GOLD),
    (
        k3,
        "Best model",
        best["Model"] if best is not None else "-",
        best["Region"] if best is not None else "",
        C_PRIMARY,
    ),
    (
        k4,
        "Best score",
        f"{best[metric]:.3f}" if best is not None else "-",
        metric,
        C_SAGE,
    ),
]:
    with col:
        st.markdown(kpi(label, value, sub, color), unsafe_allow_html=True)

st.divider()

tab_leader, tab_heatmap, tab_table = st.tabs(["Leaderboard", "Heatmap", "Table"])

with tab_leader:
    if ranked.empty:
        st.info("No rows match the current filters.")
    else:
        labels = [
            f"{row['Region']} / {row['Model']}{' *' if row['Champion'] else ''}"
            for _, row in ranked.iterrows()
        ]
        colors = [
            C_GOLD if row["Champion"] else C_PRIMARY
            for _, row in ranked.iterrows()
        ]
        fig = go.Figure(
            go.Bar(
                x=labels,
                y=ranked[metric],
                marker_color=colors,
                text=[f"{v:.2f}" for v in ranked[metric]],
                textposition="outside",
            )
        )
        apply_theme(
            fig,
            height=430,
            title=f"{metric} leaderboard (lower is better)",
            ylabel=metric,
            showlegend=False,
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, width="stretch")

with tab_heatmap:
    section_label("Region × Model Best CV MAPE")
    pivot = df.pivot_table(
        index="Region",
        columns="Model",
        values="CV MAPE %",
        aggfunc="min",
    )
    # Drop columns and rows with no data at all
    pivot = pivot.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if pivot.empty:
        st.info("No heatmap data available.")
    else:
        # Only show regions that have been evaluated on more than one model family
        multi_model = pivot.notna().sum(axis=1) > 1
        display_pivot = pivot[multi_model] if multi_model.any() else pivot
        if display_pivot.empty:
            st.info(
                "All regions were evaluated on a single model family — "
                "train additional model types to compare."
            )
        else:
            fig = px.imshow(
                display_pivot,
                color_continuous_scale=[
                    [0.0, "#4E7A63"],
                    [0.5, "#B8850A"],
                    [1.0, "#964040"],
                ],
                labels={"color": "CV MAPE %"},
                aspect="auto",
                text_auto=".2f",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#EAE3D6",
                font=dict(color=C_TEXT),
                height=max(420, len(display_pivot) * 28 + 120),
                margin=dict(l=120, r=60, t=45, b=60),
            )
            st.plotly_chart(fig, width="stretch")
        if not multi_model.all():
            n_single = int((~multi_model).sum())
            st.caption(
                f"{n_single} region(s) trained on a single model family are "
                "hidden from the heatmap. Enable more model types in the config "
                "and retrain to include them."
            )

with tab_table:
    display = view.sort_values(["Region", "CV MAPE %"], na_position="last").copy()
    display["Champion"] = display["Champion"].map(lambda value: "yes" if value else "")
    display["Artifact"] = display["Artifact"].map(
        lambda value: "ready" if value else ""
    )
    for pct_col in ["CV MAPE %", "Test MAPE %"]:
        display[pct_col] = display[pct_col].map(
            lambda value: f"{value:.3f}" if pd.notna(value) else "-"
        )
    for abs_col in ["RMSE", "MAE"]:
        display[abs_col] = display[abs_col].map(fmt_large)
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption(
        "Non-champion rows store candidate metrics only. Champion rows also have "
        "the full-history model artifact used by the API."
    )
