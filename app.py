"""
app.py — CareSignal Reablement Intelligence
============================================
Operational dashboard for community reablement services, analysing NHS
discharge and A&E data across a configurable set of acute NHS trusts.

Design: Clinical precision dark theme — deep navy/slate, teal accents,
        IBM Plex Sans typography, data-dense but readable.

Pages:
  🏠  Operations      — 5 trust cards, last 3 months, 6+2 metrics
  🔀  Demand Pipeline — NCTR funnel per trust
  📉  Capacity Gap    — P1 capacity gap trend + A&E overlay
  📊  Benchmark       — each trust vs its region + National
  📈  Forecast        — 14-day P1 forecast per trust
  💬  Ask the Data    — AI chat on the loaded NHS data
  🔧  Data Quality    — file status + known issues
"""

from __future__ import annotations

import os, sys
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data_loader import (
    load_all_data, get_quality_report,
    get_provider_monthly, get_daily_metric,
    get_regional_benchmark, get_ecl_ae, get_ecl_sitrep,
    ECL_PROVIDER_CODES, ECL_DISPLAY_NAMES, ECL_AREA,
    ECL_REGION_MAP, ECL_METRICS, REABLEMENT_METRICS,
)
from lag_model import build_forecast_model
from chat_engine import build_data_context, ask_question, is_api_configured

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CareSignal · Reablement Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM — Clinical Precision Dark Theme
# IBM Plex Sans + Roboto Mono | Deep navy + teal/cyan accents
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Root tokens ─────────────────────────────────────── */
:root {
  --bg:        #f0f4f8;
  --surface:   #ffffff;
  --surface2:  #f5f8fb;
  --surface3:  #eaf0f6;
  --border:    rgba(30,80,130,0.09);
  --border2:   rgba(30,80,130,0.16);
  --text:      #0f2137;
  --text2:     #2e5070;
  --text3:     #6a8aaa;
  --teal:      #006fa8;
  --teal-dim:  rgba(0,111,168,0.07);
  --teal-mid:  rgba(0,111,168,0.16);
  --blue:      #1a56a0;
  --amber:     #b86e00;
  --red:       #b52b20;
  --green:     #0a6b50;
  --font:      'IBM Plex Sans', sans-serif;
  --mono:      'IBM Plex Mono', monospace;
}

/* ── Global reset ────────────────────────────────────── */
html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--font) !important;
  color: var(--text) !important;
}
[data-testid="stAppViewContainer"] > .main {
  background: var(--bg) !important;
}
.block-container {
  padding: 1.2rem 1.8rem 2rem !important;
  max-width: 1600px !important;
}
* { box-sizing: border-box; }

/* ── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #0f2137 !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
}
/* Hide Streamlit's sidebar collapse toggle button entirely */
[data-testid="collapsedControl"] {
  display: none !important;
}
/* Also hide the expand/collapse chevron that shows icon names as text */
button[kind="header"],
[data-testid="stSidebarCollapseButton"] {
  display: none !important;
}
/* Fallback: any button in sidebar header area */
[data-testid="stSidebar"] > div:first-child > div:first-child > button {
  display: none !important;
}
[data-testid="stSidebar"] * { font-family: var(--font) !important; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div { color: #c8ddf0 !important; }

/* ── Typography ──────────────────────────────────────── */
h1, h2, h3 { font-family: var(--font) !important; letter-spacing: -0.3px; }
h1 { font-size: 1.5rem !important; font-weight: 700 !important; }
h2 { font-size: 1.1rem !important; font-weight: 600 !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
p, li, label { font-size: 0.85rem !important; }

/* ── Metrics ─────────────────────────────────────────── */
[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  box-shadow: 0 1px 3px rgba(15,33,55,0.07) !important;
}
[data-testid="metric-container"] label {
  color: var(--text3) !important;
  font-size: 10px !important;
  text-transform: uppercase !important;
  letter-spacing: 0.8px !important;
  font-family: var(--mono) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  color: var(--blue) !important;
  font-family: var(--mono) !important;
  font-size: 1.4rem !important;
  font-weight: 500 !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-size: 0.75rem !important;
}

/* ── Divider ─────────────────────────────────────────── */
hr { border-color: var(--border) !important; }

/* ── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  box-shadow: 0 1px 3px rgba(15,33,55,0.06) !important;
}

/* ── Dataframe ───────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

/* ── Buttons ─────────────────────────────────────────── */
.stButton > button {
  background: var(--surface2) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: 6px !important;
  font-family: var(--font) !important;
  font-size: 0.82rem !important;
  transition: all 0.15s !important;
}
.stButton > button:hover {
  border-color: var(--teal) !important;
  color: var(--teal) !important;
  background: var(--teal-dim) !important;
}

/* ── Selectbox / radio ───────────────────────────────── */
.stSelectbox > div > div {
  background: var(--surface2) !important;
  color: var(--text) !important;
}
/* Radio inside sidebar */
[data-testid="stSidebar"] .stRadio > div {
  background: transparent !important;
}
[data-testid="stSidebar"] .stRadio label {
  color: #a8c8e8 !important;
  font-size: 0.85rem !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  padding: 3px 0 !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"],
[data-testid="stSidebar"] [data-baseweb="block"] {
  background: transparent !important;
}
/* Radio dot colour */
[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] div {
  background-color: #5bbfdf !important;
  border-color: #5bbfdf !important;
}

/* ── Chat ────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 10px !important;
  box-shadow: 0 1px 3px rgba(15,33,55,0.06) !important;
}
[data-testid="stChatInputTextArea"] {
  background: var(--surface) !important;
  border: 1px solid var(--border2) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* ── Spinner ─────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--blue) !important; }

/* ── Custom components (defined below) ──────────────── */
.cs-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 0 1.2rem 0;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 1.4rem;
}
.cs-logo {
  font-size: 1.4rem; font-weight: 700; letter-spacing: -0.5px;
  color: var(--text);
}
.cs-logo span { color: var(--teal); font-weight: 800; }
.cs-tagline {
  font-family: var(--mono); font-size: 0.72rem;
  color: var(--text3); letter-spacing: 0.5px; margin-top: 2px;
}
.cs-badge {
  background: var(--teal-dim); border: 1px solid var(--teal-mid);
  color: var(--teal); border-radius: 5px; padding: 4px 12px;
  font-family: var(--mono); font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.5px;
}
.cs-card {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: 10px;
  padding: 18px 20px;
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
  transition: all 0.2s;
  box-shadow: 0 1px 4px rgba(15,33,55,0.07);
}
.cs-card:hover { border-color: var(--teal); box-shadow: 0 2px 10px rgba(0,111,168,0.12); }
.cs-card-accent {
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--teal), transparent);
}
.cs-card-accent.amber { background: linear-gradient(90deg, var(--amber), transparent); }
.cs-card-accent.red   { background: linear-gradient(90deg, var(--red),   transparent); }
.cs-card-accent.green { background: linear-gradient(90deg, var(--green),  transparent); }
.cs-card-title {
  font-size: 0.78rem; font-weight: 600; color: var(--text3);
  text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 2px;
  font-family: var(--mono);
}
.cs-card-name {
  font-size: 1.05rem; font-weight: 700; color: var(--text);
  margin-bottom: 12px;
}
.cs-area-tag {
  display: inline-block;
  background: var(--surface3); border: 1px solid var(--border2);
  color: var(--text2); border-radius: 4px;
  padding: 1px 8px; font-size: 0.7rem; font-family: var(--mono);
  margin-bottom: 12px;
}
.cs-metric-grid {
  display: grid; grid-template-columns: repeat(3,1fr); gap: 8px;
}
.cs-mini-metric {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 7px; padding: 10px 12px;
  transition: background 0.15s;
}
.cs-mini-label {
  font-size: 9px; font-family: var(--mono);
  color: var(--text3); text-transform: uppercase;
  letter-spacing: 0.6px; margin-bottom: 4px;
}
.cs-mini-value {
  font-size: 1.15rem; font-weight: 600; font-family: var(--mono);
  color: var(--text); line-height: 1;
}
.cs-mini-delta {
  font-size: 0.7rem; margin-top: 3px;
  font-family: var(--mono);
}
.cs-mini-delta.up   { color: var(--red); }
.cs-mini-delta.down { color: var(--green); }
.cs-mini-delta.flat { color: var(--text3); }
.cs-alert {
  padding: 10px 14px; border-radius: 7px;
  font-size: 0.82rem; margin-bottom: 10px;
  border-left: 3px solid;
  font-family: var(--font);
}
.cs-alert.red    { background: rgba(181,43,32,0.07);  border-color: var(--red);   color: var(--red); }
.cs-alert.amber  { background: rgba(184,110,0,0.07);  border-color: var(--amber); color: var(--amber); }
.cs-alert.teal   { background: var(--teal-dim);        border-color: var(--teal);  color: var(--teal); }
.cs-alert.green  { background: rgba(10,107,80,0.07);  border-color: var(--green); color: var(--green); }
.cs-section-title {
  font-size: 0.72rem; font-weight: 600; font-family: var(--mono);
  color: var(--teal); text-transform: uppercase; letter-spacing: 1px;
  border-top: 2px solid var(--teal-mid); padding-top: 8px;
  margin-bottom: 12px; margin-top: 4px;
  display: flex; align-items: center; gap: 8px;
}
.cs-section-title::after {
  content: ''; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border2), transparent);
}
.cs-pressure-indicator {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 6px 14px; border-radius: 6px; font-size: 0.78rem;
  font-family: var(--mono); font-weight: 500; border: 1px solid;
}
.pressure-low      { background: rgba(10,107,80,0.08);  border-color: var(--green); color: var(--green); }
.pressure-normal   { background: rgba(26,86,160,0.08); border-color: var(--blue);  color: var(--blue); }
.pressure-elevated { background: rgba(184,110,0,0.08); border-color: var(--amber); color: var(--amber); }
.pressure-high     { background: rgba(181,43,32,0.08); border-color: var(--red);   color: var(--red); }
.pressure-unknown  { background: var(--surface3); border-color: var(--border2); color: var(--text3); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE PER PROVIDER
# ─────────────────────────────────────────────────────────────────────────────

PROVIDER_COLORS = {
    "RAJ": "#00c8d4",   # teal    — Mid & South Essex (largest)
    "RDE": "#3b82f6",   # blue    — East Suffolk & NE Essex
    "RQW": "#a78bfa",   # violet  — Princess Alexandra
    "RF4": "#f59e0b",   # amber   — Barking Havering Redbridge
    "RYR": "#10b981",   # emerald — University Hospitals Sussex
}

PRESSURE_CSS = {
    "low":      ("pressure-low",      "●", "Low"),
    "normal":   ("pressure-normal",   "●", "Normal"),
    "elevated": ("pressure-elevated", "●", "Elevated"),
    "high":     ("pressure-high",     "●", "High"),
    "unknown":  ("pressure-unknown",  "○", "—"),
}


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE THEME
# ─────────────────────────────────────────────────────────────────────────────

def _plotly_layout(**kwargs) -> dict:
    base = dict(
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = dict(family="IBM Plex Sans", color="#4a6585", size=11),
        margin        = dict(l=8, r=8, t=36, b=8),
        xaxis = dict(gridcolor="rgba(30,80,130,0.07)",
                     linecolor="rgba(30,80,130,0.12)",
                     tickfont=dict(size=10, color="#6a8aaa")),
        yaxis = dict(gridcolor="rgba(30,80,130,0.07)",
                     linecolor="rgba(30,80,130,0.12)",
                     tickfont=dict(size=10, color="#6a8aaa")),
    )
    base.update(kwargs)
    return base


def _sparkline(values: list[float], color: str = "#00c8d4",
               height: int = 48) -> go.Figure:
    """Minimal sparkline with gradient fill."""
    x = list(range(len(values)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=values, mode="lines",
        line=dict(color=color, width=1.8),
        fill="tozeroy",
        fillcolor=f"rgba({_hex_rgb(color)},0.12)",
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


def _hex_rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"{r},{g},{b}"


def _hex_rgba(hex_color: str, alpha: float) -> str:
    return f"rgba({_hex_rgb(hex_color)},{alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _alert(text: str, level: str = "teal") -> None:
    st.markdown(
        f'<div class="cs-alert {level}">{text}</div>',
        unsafe_allow_html=True,
    )


def _section(title: str) -> None:
    st.markdown(
        f'<div class="cs-section-title">{title}</div>',
        unsafe_allow_html=True,
    )


def _delta_class(val: float, prev: float, invert: bool = False) -> tuple[str, str]:
    """Returns (css_class, formatted_delta_string)."""
    if prev == 0 or pd.isna(prev) or pd.isna(val):
        return "flat", "—"
    pct = (val - prev) / abs(prev) * 100
    symbol = "▲" if pct > 0 else "▼"
    if invert:
        css = "down" if pct > 0 else "up"
    else:
        css = "up" if pct > 0 else "down"
    return css, f"{symbol} {abs(pct):.1f}%"


def _card_accent(code: str, cap_val: float, remaining_pct: float) -> str:
    """Returns accent colour class based on operational status."""
    if remaining_pct > 0.6 or cap_val > 10:
        return "red"
    elif remaining_pct > 0.4 or cap_val > 5:
        return "amber"
    return "teal"


# ─────────────────────────────────────────────────────────────────────────────
# CACHED DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def _load(sitrep_dir: str, ae_dir: str):
    return load_all_data(sitrep_dir, ae_dir)


@st.cache_data(ttl=3600, show_spinner=False)
def _train(s_hash: int, a_hash: int, sitrep_dir: str, ae_dir: str):
    df_s, df_a, _ = load_all_data(sitrep_dir, ae_dir)
    return build_forecast_model(df_s, df_a)


def _data_hash(df: pd.DataFrame) -> int:
    return hash((len(df), str(df["period"].max()) if "period" in df.columns else ""))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

SITREP_DIR = str(ROOT / "data" / "sitrep")
AE_DIR     = str(ROOT / "data" / "ae")

with st.sidebar:
    # Logo
    st.markdown("""
    <div style="padding:12px 0 16px">
      <div style="font-size:1.3rem;font-weight:700;letter-spacing:-0.5px;color:#e8edf8;
                  font-family:'IBM Plex Sans',sans-serif">
        Care<span style="color:#00c8d4">Signal</span>
      </div>
      <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
                  color:#4d6580;letter-spacing:0.4px;margin-top:2px">
        REABLEMENT INTELLIGENCE
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # Load data
    with st.spinner("Loading data…"):
        try:
            df_s, df_a, rpt = _load(SITREP_DIR, AE_DIR)
            data_ok = len(df_s) > 0
        except Exception as e:
            st.error(f"Data load error: {e}")
            st.stop()

    if not data_ok:
        st.warning("No data found.\nPlace sitrep_YYYY-MM.csv and ae_YYYY-MM.csv in data folders.")
        st.stop()

    # Data status
    st.markdown('<div style="font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;'
                'color:#7aafd4;text-transform:uppercase;letter-spacing:0.7px;'
                'margin-bottom:8px">Data Status</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    latest_s = rpt.sitrep_date_range.split(" - ")[-1].strip() if rpt.sitrep_date_range else "—"
    latest_a = rpt.ae_date_range.split(" - ")[-1].strip()     if rpt.ae_date_range     else "—"
    c1.metric("SitRep", latest_s, f"{len(rpt.sitrep_periods_loaded)} mo")
    c2.metric("A&E",    latest_a, f"{len(rpt.ae_periods_loaded)} mo")

    if rpt.lag_note:
        color = "teal" if "Both sources" in rpt.lag_note else "amber"
        _alert(rpt.lag_note, color)

    n_err  = sum(1 for f in rpt.files if f.status == "error")
    n_warn = sum(1 for f in rpt.files if f.status == "warning")
    if n_err:
        _alert(f"❌ {n_err} file error(s) — see Data Quality", "red")
    elif n_warn:
        _alert(f"⚠ {n_warn} file warning(s)", "amber")
    else:
        _alert(f"✅ All {len(rpt.files)} files OK", "teal")

    st.divider()

    # Navigation — plain st.radio (reliable, no emoji artifacts)
    page = st.radio("", [
        "Operations",
        "Demand Pipeline",
        "Capacity Gap P1",
        "Benchmark",
        "Forecast",
        "Ask the Data",
        "Data Quality",
    ], label_visibility="collapsed")

    st.divider()
    st.caption(f"5 trusts · NHS England\n{rpt.sitrep_date_range}")

# Train model
with st.spinner("Initialising forecast model…"):
    try:
        model    = _train(_data_hash(df_s), _data_hash(df_a), SITREP_DIR, AE_DIR)
        sm       = model.get_summary()
        model_ok = True
    except Exception as e:
        model_ok   = False
        model_error = str(e)

# Derived globals
periods  = sorted(df_s["period"].unique())
latest_p = periods[-1]
prev_p   = periods[-2] if len(periods) > 1 else None
last3    = periods[-3:]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: pressure status widget
# ─────────────────────────────────────────────────────────────────────────────

def _render_pressure_bar() -> None:
    if not model_ok:
        return
    ps = model.get_pressure_status()
    css, dot, label = PRESSURE_CSS.get(ps["level"], PRESSURE_CSS["unknown"])
    pct  = f"{ps['percentile']}th pct" if ps["percentile"] else ""
    val  = f"{ps['value']:,.0f}" if ps["value"] else "—"
    st.markdown(
        f'<div class="cs-pressure-indicator {css}">'
        f'{dot}&nbsp; System Pressure: <strong>{label}</strong>'
        f'&nbsp;·&nbsp; A&E Wait 12h+: {val} total ({pct})'
        f'&nbsp;·&nbsp; Data: {sm.last_ae_period}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

if page == "Operations":

    # Header
    st.markdown(f"""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            OPERATIONS CENTRE
          </span>
        </div>
        <div class="cs-tagline">CareSignal Reablement · Pathway 1 · {str(latest_p).upper()}</div>
      </div>
      <div class="cs-badge">{len(ECL_PROVIDER_CODES)} TRUSTS MONITORED</div>
    </div>
    """, unsafe_allow_html=True)

    _render_pressure_bar()

    # Pre-fetch all metrics for last 3 months
    def _m3(key: str) -> pd.DataFrame:
        df = get_provider_monthly(df_s, key, codes=ECL_PROVIDER_CODES)
        return df[df["period"].isin(last3)].copy()

    m_nctr  = _m3("nctr")
    m_dis   = _m3("discharged")
    m_rem   = _m3("nctr_remaining")
    m_cap   = _m3("cap_gap_p1")
    m_ifc   = _m3("interface_p1")
    m_p1    = _m3("p1_home")
    ae_l3   = get_ecl_ae(df_a)
    ae_l3   = ae_l3[ae_l3["period"].isin(last3)]

    def _get3(df: pd.DataFrame, code: str, col: str = "Value") -> list[float]:
        sub = df[df["Org Code"] == code].sort_values("period")
        return sub[col].fillna(0).tolist()

    def _latest(df: pd.DataFrame, code: str) -> float:
        sub = df[(df["Org Code"] == code) & (df["period"] == latest_p)]
        if len(sub) == 0:
            return 0.0
        return float(sub["Value"].iloc[0])

    def _prev_val(df: pd.DataFrame, code: str) -> float:
        if prev_p is None:
            return 0.0
        sub = df[(df["Org Code"] == code) & (df["period"] == prev_p)]
        if len(sub) == 0:
            return 0.0
        return float(sub["Value"].iloc[0])

    def _ae3(code: str, col: str) -> list[float]:
        sub = ae_l3[ae_l3["Org Code"] == code].sort_values("period")
        return sub[col].fillna(0).tolist()

    # One column per trust (responsive: 3+2 layout)
    row1 = st.columns(3)
    row2_c = st.columns([1, 1, 0.001])   # last cell invisible spacer

    for idx, code in enumerate(ECL_PROVIDER_CODES):
        col = row1[idx] if idx < 3 else row2_c[idx - 3]
        name  = ECL_DISPLAY_NAMES[code]
        area  = ECL_AREA[code]
        color = PROVIDER_COLORS[code]

        # Values
        nctr_v   = _latest(m_nctr, code)
        nctr_p   = _prev_val(m_nctr, code)
        dis_v    = _latest(m_dis, code)
        dis_p    = _prev_val(m_dis, code)
        rem_v    = _latest(m_rem, code)
        rem_p    = _prev_val(m_rem, code)
        cap_v    = _latest(m_cap, code)
        cap_p    = _prev_val(m_cap, code)
        ifc_v    = _latest(m_ifc, code)
        p1_v     = _latest(m_p1, code)
        p1_p_val = _prev_val(m_p1, code)

        rem_pct  = rem_v / nctr_v if nctr_v > 0 else 0
        accent   = _card_accent(code, cap_v, rem_pct)

        nctr_vals  = _get3(m_nctr, code)
        p1_vals    = _get3(m_p1, code)
        ae_admissions = _ae3(code, "emerg_admissions_t1")
        ae_wait    = _ae3(code, "wait_12h_plus")

        _, d_nctr = _delta_class(nctr_v, nctr_p)
        _, d_dis  = _delta_class(dis_v, dis_p)
        _, d_rem  = _delta_class(rem_v, rem_p, invert=True)
        _, d_cap  = _delta_class(cap_v, cap_p, invert=True)
        _, d_p1   = _delta_class(p1_v, p1_p_val)

        dc_nctr, _ = _delta_class(nctr_v, nctr_p)
        dc_dis, _  = _delta_class(dis_v, dis_p)
        dc_rem, _  = _delta_class(rem_v, rem_p, invert=True)
        dc_cap, _  = _delta_class(cap_v, cap_p, invert=True)
        dc_p1, _   = _delta_class(p1_v, p1_p_val)

        with col:
            st.markdown(f"""
            <div class="cs-card">
              <div class="cs-card-accent {accent}"></div>
              <div class="cs-card-title">{area}</div>
              <div class="cs-card-name">{name}</div>
              <div class="cs-metric-grid">
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">NCTR avg/day</div>
                  <div class="cs-mini-value">{nctr_v/latest_p.days_in_month:.0f}</div>
                  <div class="cs-mini-delta {dc_nctr}">{d_nctr}</div>
                </div>
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">Discharged</div>
                  <div class="cs-mini-value">{int(dis_v):,}</div>
                  <div class="cs-mini-delta {dc_dis}">{d_dis}</div>
                </div>
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">Remaining</div>
                  <div class="cs-mini-value">{rem_v/latest_p.days_in_month:.0f}
                    <span style="font-size:0.7rem;color:#4d6580">/day</span>
                  </div>
                  <div class="cs-mini-delta {dc_rem}">{d_rem}</div>
                </div>
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">Cap Gap P1</div>
                  <div class="cs-mini-value" style="color:{'#ef4444' if cap_v>8 else '#f59e0b' if cap_v>3 else '#10b981'}">{cap_v:.0f}</div>
                  <div class="cs-mini-delta {dc_cap}">{d_cap}</div>
                </div>
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">Interface P1</div>
                  <div class="cs-mini-value">{ifc_v:.0f}</div>
                </div>
                <div class="cs-mini-metric">
                  <div class="cs-mini-label">P1 Home</div>
                  <div class="cs-mini-value" style="color:{color}">{int(p1_v):,}</div>
                  <div class="cs-mini-delta {dc_p1}">{d_p1}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Sparklines row
            if len(p1_vals) >= 2 and len(nctr_vals) >= 2:
                sp1, sp2 = st.columns(2)
                with sp1:
                    st.caption("P1 Home — 3 months")
                    st.plotly_chart(
                        _sparkline(p1_vals, color),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                with sp2:
                    st.caption("A&E Wait 12h+")
                    st.plotly_chart(
                        _sparkline(ae_wait, "#f59e0b"),
                        use_container_width=True, config={"displayModeBar": False},
                    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Demand Pipeline":

    st.markdown(f"""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            DEMAND PIPELINE
          </span>
        </div>
        <div class="cs-tagline">NCTR → Discharged vs Remaining → Delay Causes · {str(latest_p).upper()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _render_pressure_bar()

    sel_code = st.selectbox(
        "Select trust",
        ECL_PROVIDER_CODES,
        format_func=lambda c: ECL_DISPLAY_NAMES[c],
    )
    color = PROVIDER_COLORS[sel_code]

    # Build pipeline data
    def _monthly_ecl(key: str) -> pd.DataFrame:
        return get_provider_monthly(df_s, key, codes=[sel_code])

    p_nctr = _monthly_ecl("nctr")
    p_dis  = _monthly_ecl("discharged")
    p_rem  = _monthly_ecl("nctr_remaining")
    p_cap  = _monthly_ecl("cap_gap_p1")
    p_ifc  = _monthly_ecl("interface_p1")
    p_p1   = _monthly_ecl("p1_home")

    # Merge
    pipe = (p_nctr[["period","period_label","Value"]].rename(columns={"Value":"nctr"})
            .merge(p_dis[["period","Value"]].rename(columns={"Value":"discharged"}), on="period", how="left")
            .merge(p_rem[["period","Value"]].rename(columns={"Value":"remaining"}),  on="period", how="left")
            .merge(p_cap[["period","Value"]].rename(columns={"Value":"cap_gap"}),    on="period", how="left")
            .merge(p_ifc[["period","Value"]].rename(columns={"Value":"interface"}),  on="period", how="left")
            .merge(p_p1[["period","Value"]].rename(columns={"Value":"p1_home"}),     on="period", how="left"))
    pipe = pipe.sort_values("period").reset_index(drop=True)
    pipe["period_label"] = pipe["period"].dt.strftime("%b %y")
    pipe = pipe.fillna(0)
    pipe["days"]          = pipe["period"].dt.days_in_month
    pipe["nctr_daily"]    = (pipe["nctr"] / pipe["days"]).round(1)
    pipe["rem_daily"]     = (pipe["remaining"] / pipe["days"]).round(1)
    pipe["dis_daily"]     = (pipe["discharged"] / pipe["days"]).round(1)
    pipe["rem_pct"]       = np.where(pipe["nctr"] > 0, pipe["remaining"] / pipe["nctr"] * 100, 0)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        _section("NCTR · DISCHARGED · REMAINING — avg per day")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pipe["period_label"], y=pipe["dis_daily"],
            name="Discharged /day", marker_color=color,
            marker_line_width=0,
        ))
        fig.add_trace(go.Bar(
            x=pipe["period_label"], y=pipe["rem_daily"],
            name="Remaining /day", marker_color="#ef4444",
            marker_line_width=0,
        ))
        fig.add_trace(go.Scatter(
            x=pipe["period_label"], y=pipe["nctr_daily"],
            name="NCTR avg/day", mode="lines+markers",
            line=dict(color="#8fa3c0", width=1.5, dash="dot"),
            marker=dict(size=5),
        ))
        fig.update_layout(
            barmode="stack", height=300,
            title=dict(text=ECL_DISPLAY_NAMES[sel_code], font=dict(size=12, color="#e8edf8")),
            legend=dict(orientation="h", y=1.08),
            **_plotly_layout(),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        _section("% PATIENTS REMAINING IN HOSPITAL (of NCTR)")
        fig2 = go.Figure()
        fig2.add_hline(y=50, line_dash="dot", line_color="#4d6580", line_width=1)
        fig2.add_trace(go.Scatter(
            x=pipe["period_label"], y=pipe["rem_pct"],
            mode="lines+markers+text",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(239,68,68,0.08)",
            text=[f"{v:.0f}%" for v in pipe["rem_pct"]],
            textposition="top center",
            textfont=dict(size=9, color="#ef4444"),
        ))
        fig2.update_layout(
            height=220, yaxis_ticksuffix="%",
            yaxis_range=[0, max(100, pipe["rem_pct"].max() * 1.2)],
            **_plotly_layout(),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        _section("DELAY CAUSES — P1 PATHWAY")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=pipe["period_label"], y=pipe["cap_gap"],
            name="Capacity Gap P1", marker_color="#ef4444",
            marker_line_width=0,
        ))
        fig3.add_trace(go.Bar(
            x=pipe["period_label"], y=pipe["interface"],
            name="Interface P1", marker_color=color,
            marker_line_width=0,
        ))
        fig3.update_layout(
            barmode="group", height=220,
            legend=dict(orientation="h", y=1.1),
            **_plotly_layout(),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        _section("P1 HOME DISCHARGES")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=pipe["period_label"], y=pipe["p1_home"],
            marker_color=color, marker_line_width=0,
            text=pipe["p1_home"].astype(int).astype(str),
            textposition="outside",
            textfont=dict(size=9, color=color),
        ))
        fig4.update_layout(height=220, **_plotly_layout())
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

        # Summary stats
        _section("LATEST MONTH SUMMARY")
        row = pipe.iloc[-1]
        st.markdown(f"""
        <div class="cs-mini-metric" style="margin-bottom:6px">
          <div class="cs-mini-label">NCTR avg/day</div>
          <div class="cs-mini-value">{row['nctr_daily']:.0f}</div>
        </div>
        <div class="cs-mini-metric" style="margin-bottom:6px">
          <div class="cs-mini-label">Remaining in hospital (%)</div>
          <div class="cs-mini-value" style="color:#ef4444">{row['rem_pct']:.1f}%</div>
        </div>
        <div class="cs-mini-metric">
          <div class="cs-mini-label">P1 Home discharges</div>
          <div class="cs-mini-value" style="color:{color}">{int(row['p1_home'])}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CAPACITY GAP P1
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Capacity Gap P1":

    st.markdown(f"""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            CAPACITY PRESSURE
          </span>
        </div>
        <div class="cs-tagline">Home-based reablement capacity not yet available · Pathway 1 · 14-month trend</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    _render_pressure_bar()

    cap_all = get_provider_monthly(df_s, "cap_gap_p1", codes=ECL_PROVIDER_CODES)
    ifc_all = get_provider_monthly(df_s, "interface_p1", codes=ECL_PROVIDER_CODES)
    ae_ecl  = get_ecl_ae(df_a)

    # Multi-trust capacity gap
    _section("CAPACITY GAP P1 — ALL MONITORED TRUSTS — 14 MONTHS")
    fig = go.Figure()
    for code in ECL_PROVIDER_CODES:
        sub = cap_all[cap_all["Org Code"] == code].sort_values("period")
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub["period_label"], y=sub["Value"],
            name=ECL_DISPLAY_NAMES[code],
            mode="lines+markers",
            line=dict(color=PROVIDER_COLORS[code], width=2),
            marker=dict(size=5),
            hovertemplate=f"<b>{ECL_DISPLAY_NAMES[code]}</b><br>%{{x}}: %{{y:.0f}} patients<extra></extra>",
        ))
    fig.update_layout(
        height=320,
        legend=dict(orientation="h", y=-0.2),
        yaxis_title="Avg patients/day",
        **_plotly_layout(),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    col_a, col_b = st.columns(2)

    with col_a:
        _section("INTERFACE DELAYS P1 — ALL TRUSTS")
        fig2 = go.Figure()
        for code in ECL_PROVIDER_CODES:
            sub = ifc_all[ifc_all["Org Code"] == code].sort_values("period")
            if len(sub) == 0:
                continue
            fig2.add_trace(go.Scatter(
                x=sub["period_label"], y=sub["Value"],
                name=ECL_DISPLAY_NAMES[code],
                mode="lines+markers",
                line=dict(color=PROVIDER_COLORS[code], width=1.8),
                marker=dict(size=4),
            ))
        fig2.update_layout(
            height=280,
            legend=dict(orientation="h", y=-0.25),
            yaxis_title="Avg patients/day",
            **_plotly_layout(),
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with col_b:
        _section("A&E WAIT 12H+ — MONITORED TRUSTS (LEADING INDICATOR)")
        ae_monthly = ae_ecl.groupby(["period","Org Code","display_name"])["wait_12h_plus"].sum().reset_index()
        fig3 = go.Figure()
        for code in ECL_PROVIDER_CODES:
            sub = ae_monthly[ae_monthly["Org Code"] == code].sort_values("period")
            if len(sub) == 0:
                continue
            fig3.add_trace(go.Scatter(
                x=sub["period"].dt.strftime("%b %y"), y=sub["wait_12h_plus"],
                name=ECL_DISPLAY_NAMES[code],
                mode="lines+markers",
                line=dict(color=PROVIDER_COLORS[code], width=1.8),
                marker=dict(size=4),
                hovertemplate="<b>" + ECL_DISPLAY_NAMES[code] + "</b><br>%{x}: %{y:,.0f}<extra></extra>",
            ))
        fig3.update_layout(
            height=280,
            legend=dict(orientation="h", y=-0.25),
            yaxis_title="Patients",
            **_plotly_layout(),
        )
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    # Heatmap — latest month snapshot
    _section("CAPACITY GAP HEATMAP — LATEST MONTH")
    latest_cap = cap_all[cap_all["period"] == latest_p][["Org Code","Value"]].copy()
    latest_cap["Trust"] = latest_cap["Org Code"].map(ECL_DISPLAY_NAMES)
    latest_cap["Area"]  = latest_cap["Org Code"].map(ECL_AREA)
    latest_cap = latest_cap.sort_values("Value", ascending=False)

    fig4 = px.bar(
        latest_cap,
        x="Value", y="Trust",
        orientation="h",
        color="Value",
        color_continuous_scale=[[0, "#10b981"], [0.4, "#f59e0b"], [1, "#ef4444"]],
        text=latest_cap["Value"].apply(lambda v: f"{v:.0f}"),
        labels={"Value": "Avg patients/day blocked", "Trust": ""},
    )
    fig4.update_traces(textposition="outside", textfont=dict(size=10, color="#8fa3c0"))
    fig4.update_coloraxes(showscale=False)
    fig4.update_layout(height=240, **_plotly_layout())
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Benchmark":

    st.markdown(f"""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            BENCHMARK
          </span>
        </div>
        <div class="cs-tagline">Each trust vs its NHS region + National · {str(latest_p).upper()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    sel_metric = st.selectbox(
        "Metric",
        {
            "nctr":           "NCTR avg/day (patients ready to be discharged)",
            "p1_home":        "P1 Home discharges (monthly total)",
            "cap_gap_p1":     "Capacity Gap P1 (patients blocked — avg/day)",
            "nctr_remaining": "NCTR Remaining in hospital (avg/day)",
            "discharged":     "Total patients discharged (monthly)",
        }
    )

    # Per-provider benchmark charts (2 columns)
    bench  = get_regional_benchmark(df_s, sel_metric)
    prov_m = get_provider_monthly(df_s, sel_metric, codes=ECL_PROVIDER_CODES)

    col_pairs = [ECL_PROVIDER_CODES[:2], ECL_PROVIDER_CODES[2:4], [ECL_PROVIDER_CODES[4]]]

    for pair in col_pairs:
        cols = st.columns(len(pair))
        for i, code in enumerate(pair):
            region = ECL_REGION_MAP[code]
            color  = PROVIDER_COLORS[code]
            name   = ECL_DISPLAY_NAMES[code]

            prov_sub  = prov_m[prov_m["Org Code"] == code].sort_values("period")
            reg_sub   = bench[bench["label"] == region.title()].sort_values("period")
            nat_sub   = bench[bench["label"] == "National"].sort_values("period")

            with cols[i]:
                _section(f"{code} · vs {region.title()}")
                fig = go.Figure()

                if len(nat_sub) > 0:
                    # Normalise national to per-trust scale for visual context
                    nat_vals = nat_sub["Value"].values
                    prov_vals = prov_sub["Value"].values if len(prov_sub) > 0 else [0]
                    scale = np.mean(prov_vals) / np.mean(nat_vals) if np.mean(nat_vals) > 0 else 1
                    fig.add_trace(go.Scatter(
                        x=nat_sub["period_label"], y=nat_sub["Value"] * scale,
                        name="National (scaled)", mode="lines",
                        line=dict(color="#4d6580", width=1, dash="dot"),
                    ))

                if len(reg_sub) > 0:
                    reg_vals = reg_sub["Value"].values
                    prov_avg = np.mean(prov_sub["Value"].values) if len(prov_sub) > 0 else 0
                    scale_r  = prov_avg / np.mean(reg_vals) if np.mean(reg_vals) > 0 else 1
                    fig.add_trace(go.Scatter(
                        x=reg_sub["period_label"], y=reg_sub["Value"] * scale_r,
                        name=f"{region.title()} (scaled)", mode="lines",
                        line=dict(color="#8fa3c0", width=1.5, dash="dash"),
                    ))

                if len(prov_sub) > 0:
                    fig.add_trace(go.Scatter(
                        x=prov_sub["period_label"], y=prov_sub["Value"],
                        name=name, mode="lines+markers",
                        line=dict(color=color, width=2.5),
                        marker=dict(size=5, color=color),
                        fill="tozeroy",
                        fillcolor=_hex_rgba(color, 0.07),
                    ))

                fig.update_layout(
                    height=260,
                    title=dict(text=name, font=dict(size=11, color="#e8edf8"), x=0),
                    legend=dict(orientation="h", y=-0.35, font=dict(size=9)),
                    **_plotly_layout(),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # Benchmark delta badge
                if len(prov_sub) > 0 and len(reg_sub) > 0:
                    pv = float(prov_sub[prov_sub["period"] == latest_p]["Value"].sum()) if len(prov_sub[prov_sub["period"] == latest_p]) > 0 else 0
                    rv = float(reg_sub[reg_sub["period"] == latest_p]["Value"].sum()) if len(reg_sub[reg_sub["period"] == latest_p]) > 0 else 0
                    if rv > 0:
                        diff = (pv - rv) / rv * 100
                        direction = "▲" if diff > 0 else "▼"
                        clr = "#ef4444" if diff > 10 else ("#f59e0b" if diff > 0 else "#10b981")
                        st.markdown(
                            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
                            f'color:{clr};margin-top:-8px;margin-bottom:12px">'
                            f'{direction} {abs(diff):.1f}% vs {region.title()} (latest month)</div>',
                            unsafe_allow_html=True,
                        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — FORECAST
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Forecast":

    st.markdown(f"""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            DEMAND FORECAST
          </span>
        </div>
        <div class="cs-tagline">P1 Home Reablement · 14-day projection · {str(latest_p).upper()}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ok:
        st.error(f"Forecast model unavailable: {model_error}")
        st.stop()

    _render_pressure_bar()

    horizon = st.slider("Forecast horizon (days)", 7, 30, 14, key="fc_horizon")
    fc_all  = model.predict_df(horizon_days=horizon)

    LOAD_COLORS = {
        "low":          "#4d6580",
        "below average":"#8fa3c0",
        "normal":       "#00c8d4",
        "above average":"#f59e0b",
        "high":         "#ef4444",
    }

    # Two-column layout: chart + table
    col_ch, col_tb = st.columns([3, 2])

    with col_ch:
        _section("14-DAY P1 FORECAST — ALL MONITORED TRUSTS")
        fig = go.Figure()

        for code in ECL_PROVIDER_CODES:
            fc = fc_all[fc_all["org_code"] == code].copy()
            color = PROVIDER_COLORS[code]

            # CI band
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["date"], fc["date"][::-1]]),
                y=pd.concat([fc["upper"], fc["lower"][::-1]]),
                fill="toself",
                fillcolor=_hex_rgba(color, 0.06),
                line=dict(width=0),
                name=f"{code} CI",
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=fc["date"], y=fc["forecast"],
                name=ECL_DISPLAY_NAMES[code],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=4),
                hovertemplate=(
                    f"<b>{ECL_DISPLAY_NAMES[code]}</b><br>"
                    "%{x|%d %b %a}<br>"
                    "P1 forecast: <b>%{y:.1f}</b><extra></extra>"
                ),
            ))

        fig.update_layout(
            height=360,
            legend=dict(orientation="h", y=-0.18),
            yaxis_title="P1 discharges/day",
            **_plotly_layout(),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_tb:
        _section("WEEKLY SUMMARY BY TRUST")
        # Group by week for each provider
        fc_all["week"] = fc_all["date"].dt.isocalendar().week
        weekly = (fc_all.groupby(["org_code", "week"])
                  .agg(forecast_sum=("forecast", "sum"),
                       n_days=("forecast", "count"))
                  .reset_index())
        weekly["weekly_total"] = (weekly["forecast_sum"]).round(0)

        for code in ECL_PROVIDER_CODES:
            sub = weekly[weekly["org_code"] == code]
            color = PROVIDER_COLORS[code]
            st.markdown(
                f'<div style="font-size:0.78rem;font-family:\'IBM Plex Mono\',monospace;'
                f'color:{color};margin-bottom:4px;margin-top:8px">'
                f'▸ {ECL_DISPLAY_NAMES[code]}</div>',
                unsafe_allow_html=True,
            )
            weeks_text = "  ".join(
                [f"W{int(r.week)}: {r.weekly_total:.0f}" for _, r in sub.iterrows()]
            )
            st.markdown(
                f'<div style="font-size:0.75rem;color:#8fa3c0;'
                f'font-family:\'IBM Plex Mono\',monospace;margin-left:12px">'
                f'{weeks_text}</div>',
                unsafe_allow_html=True,
            )

        _section("MODEL PARAMETERS")
        sm_s = model.get_summary()
        with st.expander("Day-of-week & seasonal factors", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                st.caption("Day-of-week")
                for day, coef in sm_s.dow_coefficients.items():
                    bar_w = int(coef * 80)
                    clr   = "#ef4444" if coef > 1.1 else ("#f59e0b" if coef > 1.0 else "#4d6580")
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
                        f'<div style="font-size:0.7rem;font-family:\'IBM Plex Mono\',monospace;'
                        f'color:#8fa3c0;width:32px">{day[:3]}</div>'
                        f'<div style="height:8px;width:{bar_w}px;background:{clr};border-radius:2px"></div>'
                        f'<div style="font-size:0.7rem;color:{clr};font-family:\'IBM Plex Mono\','
                        f'monospace">{coef:.3f}</div></div>',
                        unsafe_allow_html=True,
                    )
            with d2:
                st.caption("Seasonal index")
                for month, idx in sm_s.seasonal_index.items():
                    d = "▲" if idx > 1.05 else ("▼" if idx < 0.95 else " ")
                    clr = "#ef4444" if idx > 1.05 else ("#10b981" if idx < 0.95 else "#4d6580")
                    st.markdown(
                        f'<div style="font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;'
                        f'margin-bottom:2px">'
                        f'<span style="color:#4d6580;width:30px;display:inline-block">{month}</span>'
                        f'<span style="color:{clr}">{idx:.3f} {d}</span></div>',
                        unsafe_allow_html=True,
                    )

        for note in sm_s.model_notes:
            _alert(f"ℹ {note}", "teal")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ASK THE DATA
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Ask the Data":

    st.markdown("""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            AI ANALYST
          </span>
        </div>
        <div class="cs-tagline">Chat with your reablement data · Powered by OpenAI</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not is_api_configured():
        _alert(
            "OpenAI API key not found. "
            "Add OPENAI_API_KEY to your .env file (local) "
            "or Streamlit Secrets (cloud deployment).",
            "red",
        )
        st.code("OPENAI_API_KEY=sk-your-key-here", language="bash")
        st.stop()

    if not model_ok:
        _alert("Forecast model unavailable — chat requires model to be loaded.", "amber")
        st.stop()

    # Build context
    data_hash_str = str(_data_hash(df_s)) + str(_data_hash(df_a))
    if (
        "chat_data_context" not in st.session_state or
        st.session_state.get("chat_data_hash") != data_hash_str
    ):
        with st.spinner("Preparing data context…"):
            st.session_state.chat_data_context = build_data_context(df_s, df_a, model)
            st.session_state.chat_data_hash    = data_hash_str

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    with st.sidebar:
        st.divider()
        st.markdown('<div style="font-size:0.72rem;font-family:\'IBM Plex Mono\',monospace;'
                    'color:#4d6580;text-transform:uppercase;letter-spacing:0.7px;'
                    'margin-bottom:8px">Chat</div>', unsafe_allow_html=True)
        if st.button("🗑 Clear conversation"):
            st.session_state.chat_messages = []
            st.rerun()
        n = len(st.session_state.chat_messages) // 2
        st.caption(f"{n} exchange{'s' if n != 1 else ''}")

    # Suggested questions
    if len(st.session_state.chat_messages) == 0:
        _section("SUGGESTED QUESTIONS")
        suggestions = [
            "Which trust has the highest % of patients remaining in hospital?",
            "Is the P1 demand growing across all monitored trusts?",
            "Should I be concerned about staffing levels next week?",
            "Give me a briefing on Mid & South Essex NHS Trust for a regional meeting.",
            "How does RYR (West Sussex) compare to the South East region?",
            "What is driving the delay in Barking Havering Redbridge?",
        ]
        c1, c2 = st.columns(2)
        for i, s in enumerate(suggestions):
            col = c1 if i % 2 == 0 else c2
            with col:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role":"user","content":s})
                    with st.spinner("Analysing data…"):
                        answer = ask_question(
                            s,
                            st.session_state.chat_data_context,
                            st.session_state.chat_messages[:-1],
                        )
                    st.session_state.chat_messages.append({"role":"assistant","content":answer})
                    st.rerun()
        st.divider()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about the reablement data…"):
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analysing…"):
                answer = ask_question(
                    prompt,
                    st.session_state.chat_data_context,
                    st.session_state.chat_messages[:-1],
                )
            st.markdown(answer)
        st.session_state.chat_messages.append({"role":"assistant","content":answer})

    st.divider()
    st.caption(
        f"⚠ Answers are based on pre-computed aggregates from NHS SitRep & A&E data. "
        f"Always verify critical figures in the dashboard before acting. "
        f"Data: {rpt.sitrep_date_range}."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — DATA QUALITY
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "Data Quality":

    st.markdown("""
    <div class="cs-header">
      <div>
        <div class="cs-logo">Care<span>Signal</span>
          <span style="font-size:0.75rem;font-weight:400;color:#4d6580;
                       font-family:'IBM Plex Mono',monospace;margin-left:12px">
            DATA QUALITY
          </span>
        </div>
        <div class="cs-tagline">File status · Known issues · NHS publication schedule</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    n_ok   = sum(1 for f in rpt.files if f.status == "ok")
    n_warn = sum(1 for f in rpt.files if f.status == "warning")
    n_err  = sum(1 for f in rpt.files if f.status == "error")
    k1.metric("Total files",  len(rpt.files))
    k2.metric("✅ OK",         n_ok)
    k3.metric("⚠ Warnings",   n_warn)
    k4.metric("❌ Errors",     n_err)

    if rpt.lag_note:
        _alert(rpt.lag_note, "teal" if "Both" in rpt.lag_note else "amber")

    _section("FILE STATUS")
    qdf = get_quality_report(rpt)
    st.dataframe(
        qdf.style.apply(
            lambda col: [
                "background-color:rgba(239,68,68,0.1)"   if "❌" in str(v) else
                "background-color:rgba(245,158,11,0.1)" if "⚠️" in str(v) else ""
                for v in col
            ],
            subset=["Status"],
        ),
        use_container_width=True,
        height=460,
    )

    _section("KNOWN NHS DATA QUIRKS")
    with st.expander("Expand", expanded=False):
        st.markdown("""
**Fixed automatically by CareSignal:**
- `Additional bed days` rows dated in the prior month → reassigned to file period  
  *(NHS pattern: monthly LOS totals carry the last day of the previous month)*
- `Delay reason LOS 7+` rows with a wrong-month date → dropped  
  *(Apr/Jun 2025 files contained preliminary Oct 2025 data)*
- BOM encoding in Feb 2026 → read with `utf-8-sig`
- Non-breaking spaces in metric names → normalised

**Known data anomalies:**
- **RDE Jan 2026**: P1 = 702 vs normal ~240 (3x spike). Excluded from forecast model.
- **RF4**: structural step-change Dec 2025 (P1: ~35 → ~120/month). Forecast baseline uses Dec 2025 onward.

**Metrics available from Aug 2025 only:**
- Total cost of delayed bed days
- Daily Average NCTR remaining

**NHS publication schedule:**
- A&E data: ~4 weeks after reference month
- SitRep: ~6 weeks after reference month
        """)

    _section("DATA FRESHNESS")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**SitRep files loaded:**")
        for p in sorted(rpt.sitrep_periods_loaded):
            st.markdown(f"  ✅ `sitrep_{p}.csv`")
    with c2:
        st.markdown("**A&E files loaded:**")
        for p in sorted(rpt.ae_periods_loaded):
            st.markdown(f"  ✅ `ae_{p}.csv`")
