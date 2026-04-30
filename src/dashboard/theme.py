"""
Design system for the Sales Forecasting dashboard.

Typography stack
  Display / page titles : Fraunces (optical serif, italic)
  UI / body             : Inter (geometric sans-serif)
  Numbers / metrics     : JetBrains Mono (tabular monospace)

Palette (warm cream)
  Background  #F7F4EF     Surface     #EEE9E0     Surface-2   #E4DDD1
  Border      #CEC6B5     Border-dk   #B5AA9C
  Text        #1C1814     Text-mid    #4A4035     Text-lite   #7A7068
  Primary     #6B5744     Accent      #B8865A     Blue        #4A6FA5
  Sage        #4E7A63     Gold        #B8850A     Terra       #9E5238
  Danger      #964040
"""
from __future__ import annotations

import plotly.graph_objects as go

# ── Tokens ────────────────────────────────────────────────────────────────────
C_BG        = "#F7F4EF"
C_SURFACE   = "#EEE9E0"
C_SURFACE_2 = "#E4DDD1"
C_BORDER    = "#CEC6B5"
C_BORDER_DK = "#B5AA9C"

C_TEXT      = "#1C1814"
C_TEXT_MID  = "#4A4035"
C_TEXT_LITE = "#7A7068"

C_PRIMARY   = "#6B5744"   # deep warm brown — brand
C_ACCENT    = "#B8865A"   # cognac / amber — highlights
C_BLUE      = "#4A6FA5"   # slate blue — contrast accent
C_SAGE      = "#4E7A63"   # deep sage — positive / good
C_GOLD      = "#B8850A"   # warm gold — champion
C_TERRA     = "#9E5238"   # terracotta — emphasis
C_DANGER    = "#964040"   # muted crimson — error

PALETTE = [C_PRIMARY, C_ACCENT, C_BLUE, C_SAGE, C_GOLD, C_TERRA, "#7A6A8A", "#3D7A8A"]


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
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── HTML Components ───────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", color: str = C_TEXT) -> str:
    """Return HTML for a KPI card. Render with st.markdown(unsafe_allow_html=True)."""
    sub_html = (
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.75rem;"
        f"color:{C_TEXT_LITE};margin:0.25rem 0 0;line-height:1.35;\">{sub}</p>"
        if sub else ""
    )
    return (
        f"<div style=\"background:{C_SURFACE};border:1px solid {C_BORDER};border-radius:12px;"
        f"padding:1.15rem 1.35rem;box-sizing:border-box;height:100%;\">"
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.63rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.13em;color:{C_TEXT_LITE};margin:0 0 0.5rem;\""
        f">{label}</p>"
        f"<p style=\"font-family:'JetBrains Mono',monospace;font-size:1.55rem;"
        f"font-weight:500;color:{color};margin:0;line-height:1.15;\">{value}</p>"
        f"{sub_html}</div>"
    )


def page_header(title: str, subtitle: str = "") -> None:
    """Render a page header with Fraunces title and Inter subtitle."""
    import streamlit as st
    sub_html = (
        f"<p style=\"font-family:'Inter',sans-serif;color:{C_TEXT_LITE};"
        f"font-size:0.88rem;margin:0.25rem 0 1.5rem;line-height:1.5;\">{subtitle}</p>"
        if subtitle else "<div style=\"margin-bottom:1.5rem;\"></div>"
    )
    st.markdown(
        f"<h1 style=\"font-family:'Fraunces',Georgia,serif;font-style:italic;"
        f"font-weight:400;font-size:1.95rem;color:{C_TEXT};margin:0;"
        f"letter-spacing:-0.025em;line-height:1.1;\">{title}</h1>{sub_html}",
        unsafe_allow_html=True,
    )


def section_label(text: str, margin_top: str = "0") -> None:
    """Render a small all-caps section label."""
    import streamlit as st
    st.markdown(
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.63rem;font-weight:700;"
        f"text-transform:uppercase;letter-spacing:0.13em;color:{C_TEXT_LITE};"
        f"margin:{margin_top} 0 0.5rem;\"> {text}</p>",
        unsafe_allow_html=True,
    )


def callout(text: str, color: str = C_PRIMARY) -> None:
    """Inline callout / annotation box."""
    import streamlit as st
    st.markdown(
        f"<div style=\"background:{C_SURFACE};border-left:3px solid {color};"
        f"border-radius:0 8px 8px 0;padding:0.7rem 1rem;margin:0.6rem 0;\">"
        f"<p style=\"font-family:'Inter',sans-serif;font-size:0.84rem;color:{C_TEXT_MID};"
        f"margin:0;line-height:1.5;\">{text}</p></div>",
        unsafe_allow_html=True,
    )


# ── Plotly theme ──────────────────────────────────────────────────────────────
_AXIS = dict(
    gridcolor="#DDD6C8",
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
        plot_bgcolor="#EAE3D6",
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
            font=dict(size=13, color=C_TEXT, family="Inter, sans-serif"),
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
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;1,9..144,300;1,9..144,400&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>

/* ── Reset & global ─────────────────────────────────────────────────────────── */
html, body { background: #F7F4EF !important; }
.stApp { background: #F7F4EF !important; }
* { font-family: 'Inter', system-ui, sans-serif; box-sizing: border-box; }
.block-container { padding-top: 2rem !important; max-width: 1400px !important; }

/* ── Page headings ──────────────────────────────────────────────────────────── */
/* h1 is driven by page_header() helper via unsafe HTML — but keep st.title fallback */
h1 {
    font-family: 'Fraunces', Georgia, serif !important;
    font-style: italic !important;
    font-weight: 400 !important;
    font-size: 1.95rem !important;
    color: #1C1814 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.1 !important;
}
h2 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: #1C1814 !important;
    letter-spacing: -0.01em !important;
}
h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: #4A4035 !important;
}
p { font-family: 'Inter', sans-serif !important; color: #1C1814 !important; }

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #E8E2D6 !important;
    border-right: 1px solid #CEC6B5 !important;
}
[data-testid="stSidebarNav"] { padding-top: 0.25rem !important; }
[data-testid="stSidebarNav"] a,
[data-testid="stSidebarNav"] span {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.86rem !important;
    font-weight: 500 !important;
    color: #4A4035 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebarNav"] [aria-selected="true"],
[data-testid="stSidebarNav"] a:hover { color: #1C1814 !important; }

/* Sidebar section headings */
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Inter', sans-serif !important;
    font-style: normal !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: #7A7068 !important;
}
[data-testid="stSidebar"] p {
    font-size: 0.84rem !important;
    color: #4A4035 !important;
    line-height: 1.5 !important;
}

/* ── Metric — use CSS, not st.metric, for main KPIs; style fallback here ───── */
[data-testid="stMetric"] {
    background: #EEE9E0 !important;
    border: 1px solid #CEC6B5 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.3rem !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.63rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: #7A7068 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.55rem !important;
    font-weight: 500 !important;
    color: #1C1814 !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem !important;
}
[data-testid="stMetricDelta"] svg { display: none !important; }

/* ── Buttons ────────────────────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.025em !important;
    border-radius: 8px !important;
    transition: background 0.15s, box-shadow 0.15s !important;
}
.stButton > button[kind="primary"] {
    background: #6B5744 !important;
    border-color: #6B5744 !important;
    color: #F7F4EF !important;
}
.stButton > button[kind="primary"]:hover {
    background: #5A4836 !important;
    border-color: #5A4836 !important;
}
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #CEC6B5 !important;
    color: #4A4035 !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #E4DDD1 !important;
    border-color: #B5AA9C !important;
}

/* ── Tabs ───────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #CEC6B5 !important;
    gap: 0 !important;
    padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #7A7068 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 0.55rem 1.1rem !important;
    letter-spacing: 0.01em !important;
    margin-bottom: -1px !important;
}
.stTabs [aria-selected="true"] {
    color: #1C1814 !important;
    border-bottom: 2px solid #6B5744 !important;
    background: transparent !important;
}

/* ── Form inputs ────────────────────────────────────────────────────────────── */
.stSelectbox label,
.stSlider label,
.stTextInput label,
.stCheckbox label,
.stRadio label > p {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.13em !important;
    color: #7A7068 !important;
}
.stSelectbox [data-baseweb="select"] > div {
    background: #E4DDD1 !important;
    border-color: #CEC6B5 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    color: #1C1814 !important;
}
.stTextInput input {
    background: #E4DDD1 !important;
    border-color: #CEC6B5 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    color: #1C1814 !important;
    font-size: 0.88rem !important;
}
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: #6B5744 !important;
    border-color: #6B5744 !important;
}

/* ── Checkbox & radio ───────────────────────────────────────────────────────── */
.stCheckbox [data-baseweb="checkbox"] input:checked ~ span,
.stRadio [data-baseweb="radio"] input:checked ~ span {
    background: #6B5744 !important;
    border-color: #6B5744 !important;
}

/* ── Divider ────────────────────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid #CEC6B5 !important;
    margin: 1.4rem 0 !important;
}

/* ── Expander ───────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #CEC6B5 !important;
    border-radius: 10px !important;
    background: #EEE9E0 !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    color: #4A4035 !important;
    padding: 0.65rem 1rem !important;
}

/* ── DataFrame ──────────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #CEC6B5 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] th {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    background: #E4DDD1 !important;
    color: #4A4035 !important;
}
[data-testid="stDataFrame"] td {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.86rem !important;
    color: #1C1814 !important;
}

/* ── Alerts / info ──────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Caption ────────────────────────────────────────────────────────────────── */
.stCaption p, small {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.76rem !important;
    color: #7A7068 !important;
}

/* ── Form submit ────────────────────────────────────────────────────────────── */
.stForm [data-testid="stFormSubmitButton"] button {
    background: #6B5744 !important;
    border-color: #6B5744 !important;
    color: #F7F4EF !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ── Download button ────────────────────────────────────────────────────────── */
.stDownloadButton button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #CEC6B5 !important;
    background: transparent !important;
    color: #4A4035 !important;
}
.stDownloadButton button:hover {
    background: #E4DDD1 !important;
}

/* ── Spinner / progress ─────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p {
    font-family: 'Inter', sans-serif !important;
    color: #7A7068 !important;
}

</style>
"""


def inject_css() -> None:
    import streamlit as st
    st.markdown(_CSS, unsafe_allow_html=True)
