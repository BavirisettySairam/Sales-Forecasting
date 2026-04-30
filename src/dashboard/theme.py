"""
Design system for the Sales Forecasting dashboard.

Typography stack
  Display / page titles : Inter (geometric sans-serif, tight tracking)
  UI / body             : Inter (geometric sans-serif)
  Numbers / metrics     : JetBrains Mono (tabular monospace)
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Tokens ────────────────────────────────────────────────────────────────────
C_BG        = "#F9FAFB"
C_SURFACE   = "#FFFFFF"
C_SURFACE_2 = "#F3F4F6"
C_BORDER    = "#E5E7EB"
C_BORDER_DK = "#D1D5DB"

C_TEXT      = "#111827"
C_TEXT_MID  = "#4B5563"
C_TEXT_LITE = "#6B7280"

C_PRIMARY   = "#0F172A"   # deep navy/slate
C_ACCENT    = "#2563EB"   # bright corporate blue
C_BLUE      = "#3B82F6"
C_SAGE      = "#059669"   # deeper emerald
C_GOLD      = "#D97706"   # deeper amber
C_TERRA     = "#E11D48"   # rose
C_DANGER    = "#DC2626"   # red

PALETTE = [
    C_ACCENT, C_PRIMARY, "#0EA5E9", C_SAGE,
    C_GOLD, C_TERRA, "#8B5CF6", "#DB2777"
]


# ── Utilities ─────────────────────────────────────────────────────────────────
def fmt_large(n, decimals: int = 2) -> str:
    if n is None:
        return "—"
    v = float(n)
    if abs(v) >= 1_000_000_000:
        return f"{v / 1e9:.{decimals}f}B"
    if abs(v) >= 1_000_000:
        return f"{v / 1e6:.{decimals}f}M"
    if abs(v) >= 1_000:
        return f"{v / 1e3:.{decimals}f}K"
    return f"{v:.{decimals}f}"


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join([c*2 for c in h])
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── HTML Components ───────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", color: str = C_TEXT) -> str:
    """Return HTML for a KPI card."""
    sub_html = (
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.75rem;"
        f"color:{C_TEXT_LITE};margin:0.3rem 0 0;line-height:1.35;\">{sub}</p>"
        if sub else ""
    )
    return (
        f"<div style=\"background:{C_SURFACE};border:1px solid {C_BORDER};"
        f"border-radius:8px;padding:1.25rem;box-sizing:border-box;height:100%;"
        f"box-shadow:0 1px 2px 0 rgba(0,0,0,0.05);\">"
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:600;"
        f"text-transform:uppercase;letter-spacing:0.05em;color:{C_TEXT_MID};"
        f"margin:0 0 0.5rem;\">{label}</p>"
        f"<p style=\"font-family:'JetBrains Mono',monospace;font-size:1.6rem;"
        f"font-weight:600;color:{color};margin:0;line-height:1.1;\">{value}</p>"
        f"{sub_html}</div>"
    )


def page_header(title: str, subtitle: str = "") -> None:
    """Render a modern page header."""
    import streamlit as st
    sub_html = (
        f"<p style=\"font-family:'Inter',sans-serif;color:{C_TEXT_MID};"
        f"font-size:0.95rem;margin:0.25rem 0 1.5rem;line-height:1.5;\">{subtitle}</p>"
        if subtitle else "<div style=\"margin-bottom:1.5rem;\"></div>"
    )
    st.markdown(
        f"<h1 style=\"font-family:'Inter',sans-serif;"
        f"font-weight:700;font-size:2rem;color:{C_TEXT};margin:0;"
        f"letter-spacing:-0.03em;line-height:1.2;\">{title}</h1>{sub_html}",
        unsafe_allow_html=True,
    )


def section_label(text: str, margin_top: str = "0") -> None:
    """Render a small all-caps section label."""
    import streamlit as st
    st.markdown(
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.65rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.05em;color:{C_TEXT_LITE};"
        f"margin:{margin_top} 0 0.5rem;\"> {text}</p>",
        unsafe_allow_html=True,
    )


def callout(text: str, color: str = C_PRIMARY) -> None:
    """Inline callout / annotation box."""
    import streamlit as st
    bg_color = hex_rgba(color, 0.05)
    st.markdown(
        f"<div style=\"background:{bg_color};border-left:4px solid {color};"
        f"border-radius:0 8px 8px 0;padding:1rem 1.25rem;margin:0.75rem 0;\">"
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.9rem;color:{C_TEXT};"
        f"margin:0;line-height:1.5;font-weight:500;\">{text}</p></div>",
        unsafe_allow_html=True,
    )


# ── Plotly theme ──────────────────────────────────────────────────────────────
_AXIS = dict(
    gridcolor=C_SURFACE_2,
    gridwidth=1,
    linecolor=C_BORDER,
    tickcolor=C_BORDER,
    zeroline=False,
    title_font=dict(size=11, color=C_TEXT_MID, family="Inter, sans-serif"),
    tickfont=dict(size=10, color=C_TEXT_LITE, family="JetBrains Mono, monospace"),
)


def plot_layout(
    height: int = 400,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    showlegend: bool = True,
    legend_pos: str = "bottom",
) -> dict:
    out: dict = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=C_TEXT, size=12, family="Inter, sans-serif"),
        height=height,
        margin=dict(l=58, r=24, t=52, b=50),
        hoverlabel=dict(
            bgcolor=C_SURFACE,
            bordercolor=C_BORDER,
            font_color=C_TEXT,
            font_size=12,
            font_family="Inter, sans-serif",
        ),
        xaxis=dict(**_AXIS, title=xlabel),
        yaxis=dict(**_AXIS, title=ylabel),
        showlegend=showlegend,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=11, color=C_TEXT_MID, family="Inter, sans-serif"),
            orientation="h" if legend_pos == "bottom" else "v",
            y=-0.22 if legend_pos == "bottom" else 1.0,
            x=0 if legend_pos == "bottom" else 1.02,
            xanchor="left",
        ),
    )
    if title:
        out["title"] = dict(
            text=title,
            font=dict(size=14, color=C_TEXT, family="Inter, sans-serif", weight="bold"),
            x=0,
            xanchor="left",
            pad=dict(b=6),
        )
    return out


def apply_theme(fig: go.Figure, **kwargs) -> go.Figure:
    fig.update_layout(**plot_layout(**kwargs))
    return fig


# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&"""
"""family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>

/* ── Reset & global ─────────────────────────────────────────────────────────── */
html, body { background: #F9FAFB !important; }
.stApp { background: #F9FAFB !important; }
*{ font-family: 'Inter', system-ui, sans-serif; box-sizing: border-box; }
.block-container { padding-top: 2rem !important; max-width: 1400px !important; }

/* ── Page headings ──────────────────────────────────────────────────────────── */
h1 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2rem !important;
    color: #111827 !important;
    letter-spacing: -0.03em !important;
    line-height: 1.2 !important;
}
h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.25rem !important;
    color: #111827 !important;
    letter-spacing: -0.02em !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: #4B5563 !important;
}
p { font-family: 'Inter', sans-serif !important; color: #374151 !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
}
[data-testid="stSidebarNav"] { padding-top: 0.5rem !important; }
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #4B5563 !important;
    border-radius: 12px !important;
    padding: 0.7rem 0.6rem !important;
    margin-bottom: 0.3rem !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: 1px solid transparent !important;
}
[data-testid="stSidebarNav"] [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #0F172A !important;
    font-weight: 600 !important;
    border: 1px solid #E5E7EB !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05),
                0 2px 4px -1px rgba(0, 0, 0, 0.03) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stSidebarNav"] a:hover {
    color: #111827 !important;
    background: #F3F4F6 !important;
}

/* Sidebar section headings */
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6B7280 !important;
    margin-top: 1rem !important;
}
[data-testid="stSidebar"] p {
    font-size: 0.85rem !important;
    color: #4B5563 !important;
    line-height: 1.5 !important;
}

/* ── Metric ─────────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    border-radius: 8px !important;
    padding: 1.25rem !important;
    box-shadow: 0 1px 2px 0 rgba(0,0,0,0.05) !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #6B7280 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Buttons ────────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    border-radius: 6px !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.15s ease !important;
    border: 1px solid transparent !important;
}
.stButton > button[kind="primary"] {
    background: #0F172A !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1E293B !important;
    border-color: #1E293B !important;
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    border: 1px solid #E5E7EB !important;
    color: #4B5563 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #F9FAFB !important;
    border-color: #D1D5DB !important;
    color: #111827 !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #E2E8F0 !important;
    gap: 1.5rem !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.75rem 0.25rem !important;
    letter-spacing: 0.01em !important;
    margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: #0F172A !important;
    border-bottom: 2px solid #3B82F6 !important;
    font-weight: 600 !important;
}

/* ── Form inputs ────────────────────────────────────────────────────────────── */
.stSelectbox label,
.stSlider label,
.stTextInput label,
.stCheckbox label,
.stRadio label > p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #64748B !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    color: #0F172A !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
.stTextInput input {
    background: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    color: #0F172A !important;
    font-size: 0.9rem !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #3B82F6 !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3) !important;
}

/* ── Checkbox & radio ───────────────────────────────────────────────────────── */
.stCheckbox [data-baseweb="checkbox"] input:checked ~ span,
.stRadio [data-baseweb="radio"] input:checked ~ span {
    background: #3B82F6 !important;
    border-color: #3B82F6 !important;
}

/* ── Divider ────────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #E2E8F0 !important;
    margin: 1.5rem 0 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    background: #FFFFFF !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    color: #0F172A !important;
    padding: 0.8rem 1rem !important;
}

/* ── DataFrame ──────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.02) !important;
}
[data-testid="stDataFrame"] th {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    background: #F8FAFC !important;
    color: #64748B !important;
    border-bottom: 1px solid #E2E8F0 !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    color: #334155 !important;
    border-bottom: 1px solid #F1F5F9 !important;
}

/* ── Alerts / info ──────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: none !important;
}

/* ── Caption ────────────────────────────────────────────────────────────────── */
.stCaption p, small {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    color: #94A3B8 !important;
}

/* ── Download button ────────────────────────────────────────────────────────── */
.stDownloadButton button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    background: #FFFFFF !important;
    color: #475569 !important;
}
.stDownloadButton button:hover {
    background: #F8FAFC !important;
    border-color: #CBD5E1 !important;
}

/* ── Spinner / progress ─────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: 'Inter', sans-serif !important;
    color: #64748B !important;
}

</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.html(_CSS)
