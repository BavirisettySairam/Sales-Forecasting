import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from theme import (  # noqa: E402
    C_PRIMARY,
    C_TEXT_MID,
    callout,
    inject_css,
    page_header,
    section_label,
)

st.set_page_config(
    page_title="About the Platform", page_icon="light_bulb", layout="wide"
)
inject_css()

page_header(
    "System Architecture & Design",
    "An inside look at the intelligent forecasting engine.",
)

callout(
    "Our forecasting engine is designed to deliver highly accurate, automated "
    "predictions by running multiple algorithmic models simultaneously and "
    "selecting the champion for each region dynamically.",
    color=C_PRIMARY,
)

c1, c2 = st.columns([2, 1])

with c1:
    st.markdown("### 🧠 The Intelligence Layer")
    st.markdown(
        "The system employs a rigorous tournament-style evaluation process for every "
        "US state. Rather than relying on a single algorithm, we train five distinct "
        "model families:\n\n"
        "1. **SARIMA (Statistical):** Captures seasonal trends and autoregressive "
        "behaviors.\n"
        "2. **Prophet (Additive):** Handles holiday effects and abrupt trend changes "
        "smoothly.\n"
        "3. **XGBoost (Gradient Boosting):** Excels at capturing non-linear "
        "interactions and complex feature relationships.\n"
        "4. **LightGBM (Gradient Boosting):** Optimized for high-efficiency and "
        "handles large feature spaces.\n"
        "5. **LSTM (Deep Learning):** A Recurrent Neural Network designed to "
        "remember long-term sequential dependencies.\n\n"
        "For each state, all 5 models are trained and evaluated using an "
        "**Expanding Window Cross-Validation** approach. The model that achieves the "
        "lowest **Mean Absolute Percentage Error (MAPE)** is automatically crowned "
        'the "Champion" and promoted to production.'
    )

    st.divider()

    st.markdown("### 📊 Robust Feature Engineering")
    st.markdown(
        "To ensure our machine learning models can find meaningful patterns, we "
        "generate 21 robust features from raw sales data:\n"
        "- **Temporal Features:** Month, quarter, week of year.\n"
        "- **Lag Features:** 1-week, 2-week, and 4-week historical lags.\n"
        "- **Rolling Statistics:** 4-week moving averages and standard deviations.\n\n"
        "*Crucially, all engineered features use strict temporal shifting to "
        "mathematically guarantee zero data leakage from the future into the past.*"
    )

with c2:
    st.markdown("### ⚡ Infrastructure")
    st.markdown(
        f"""
        <div style="background:#FFFFFF; border:1px solid #E5E7EB; border-radius:12px;
        padding:1.5rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <h4 style="margin-top:0; color:#111827; font-family:'Inter', sans-serif;">
        Core Tech Stack</h4>
        <ul style="color:{C_TEXT_MID}; font-size:0.9rem; line-height:1.6;
        font-family:'Inter', sans-serif;">
            <li><b>Backend API:</b> FastAPI (Python)</li>
            <li><b>Dashboard:</b> Streamlit & Plotly</li>
            <li><b>Database:</b> PostgreSQL & SQLAlchemy</li>
            <li><b>Data Processing:</b> Pandas & NumPy</li>
            <li><b>Machine Learning:</b> PyTorch, XGBoost, Prophet, Optuna</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### 🔒 Enterprise Security")
    st.markdown(
        "- **Authentication:** All API endpoints are protected by API key validation.\n"
        "- **Rate Limiting:** Built-in memory rate limiters prevent API abuse "
        "(200 req/min for general routes, 5 req/hour for heavy retraining tasks).\n"
        "- **Headers:** Strict Content-Security-Policy, HSTS, and XSS protection "
        "enabled."
    )

st.divider()

st.markdown("### 🚀 Open Source")
st.markdown(
    "This project is entirely open-source. You can review the code, open issues, "
    "or contribute directly on GitHub:  \n"
    "👉 **[BavirisettySairam/Sales-Forecasting]"
    "(https://github.com/BavirisettySairam/Sales-Forecasting)**"
)

st.markdown("<br><br>", unsafe_allow_html=True)
section_label("System Constraints & Upgradations")
st.markdown(
    """
    <span style="color:#6B7280; font-size:0.85rem;">
    <b>Note on Architecture:</b> Redis and Docker were originally part of the
    infrastructure stack for distributed caching and container orchestration. They
    have been temporarily paused from the main deployment workflow due to constraints,
    and are slated for reintegration in future enterprise updates to improve
    high-traffic scaling.
    </span>
    """,
    unsafe_allow_html=True,
)
