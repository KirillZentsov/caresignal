"""
chat_engine.py
==============
"Chat with your data" module for the CareSignal Reablement Intelligence
dashboard.

How it works:
  1. build_data_context() compresses 14 months of data into a
     structured one-page summary (~3-4k tokens).
  2. ask_question() sends that summary + the user's question to
     the OpenAI API and returns a plain-English answer.
  3. app.py stores the conversation in st.session_state and
     renders it as a standard chat interface.

The LLM never sees raw CSV rows — only the pre-computed summary.
This keeps costs low (~$0.001 per message) and responses fast.
"""

from __future__ import annotations

import os
import json
from typing import Optional

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — works without it on Streamlit Cloud

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ESSEX_PROVIDER_CODES = [
    "RAJ","RDE","RQW","RWH","RWG","RGT","RGP","RGR","RM1","RCX","RC9","RD8","RGN"
]

PROVIDER_NAMES = {
    "RAJ": "Mid & South Essex NHS FT",
    "RDE": "East Suffolk & NE Essex NHS FT",
    "RQW": "Princess Alexandra Hospital",
    "RWH": "East & North Hertfordshire NHS Trust",
    "RWG": "West Hertfordshire Teaching Hospitals",
    "RGT": "Cambridge University Hospitals",
    "RGP": "James Paget University Hospitals",
    "RGR": "West Suffolk NHS FT",
    "RM1": "Norfolk & Norwich University Hospitals",
    "RCX": "QE Hospital King's Lynn",
    "RC9": "Bedfordshire Hospitals",
    "RD8": "Milton Keynes University Hospital",
    "RGN": "North West Anglia NHS FT",
}

REABLEMENT_METRICS = {
    "p1_home": "Pathway 1: Discharge to a domestic home, hotel, or other temporary accommodation, or hospice at home with rehabilitation, reablement and recovery",
    "p2_bed":  "Pathway 2: Short-term bed/hospice for rehabilitation, reablement and recovery / end of life care",
    "nctr":    "Number of patients who no longer meet the criteria to reside",
    "cap_p1":  "Capacity - Home-based rehabilitation, reablement or recovery services not yet available (Pathway 1)",
    "cap_p2":  "Capacity - Bed-based rehabilitation, reablement or recovery services not yet available (Pathway 2)",
    "iface_p1":"Interface process - Home based rehabilitation, reablement or recovery service arrangements still underway (Pathway 1)",
    "discharges":"Number of patients discharged",
    "los14":   "Number of additional bed days, patients with length of stay of 14 days or over",
    "delay_cost":"Total cost of delayed bed days in the month",
}

# System prompt — sets the assistant's role and constraints
SYSTEM_PROMPT = """You are an intelligent data analyst for a community
reablement services operation in England, working with NHS England open data.

Your job is to help an operational manager understand what is happening across
the monitored acute NHS trusts using NHS England discharge and A&E datasets.

You have been given a structured data context below containing:
- Monthly P1/P2 reablement discharge volumes per provider (Jan 2025 – latest)
- Capacity gap trends (patients blocked because reablement capacity unavailable)
- A&E system pressure (Wait 12h+)
- Demand forecast for the coming days
- Key alerts and anomalies

IMPORTANT RULES:
- Base ALL answers strictly on the data context provided. Do not invent figures.
- When quoting numbers, be specific (e.g. "RAJ averaged 325 NCTR/day in Jan 2026").
- If the data does not contain enough detail to answer precisely, say so clearly and 
  suggest what dashboard page to check for the full detail.
- Keep answers concise and operational — the manager needs to act, not just understand.
- Always end with a clear "Bottom line:" sentence.
- Use plain English, avoid NHS jargon where possible.
- Data covers monitored acute trusts including RAJ (Mid & South Essex),
  RDE (East Suffolk & NE Essex), RQW (Princess Alexandra) and others.

DATA CONTEXT:
{data_context}
"""


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_data_context(
    df_s: pd.DataFrame,
    df_a: pd.DataFrame,
    model,  # ReablementForecastModel instance
) -> str:
    """
    Compresses 14 months of SitRep + A&E data into a structured
    text summary (~3-4k tokens) suitable for passing to the LLM.

    This is the "cheat sheet" — the LLM reads this, not the raw CSV.
    """
    lines: list[str] = []
    lines.append("=== REABLEMENT DATA CONTEXT ===\n")

    periods = sorted(df_s["period"].unique())
    lines.append(f"Data range: {periods[0]} to {periods[-1]}")
    lines.append(f"Providers tracked: {len(ESSEX_PROVIDER_CODES)}\n")

    # ── 1. Provider profiles ─────────────────────────────────────────────────
    lines.append("--- PROVIDER PROFILES ---")

    essex = df_s[df_s["Org Code"].isin(ESSEX_PROVIDER_CODES)].copy()

    for code in ["RAJ", "RDE", "RQW", "RWH", "RWG"]:  # priority 5 first
        name = PROVIDER_NAMES.get(code, code)
        sub  = essex[essex["Org Code"] == code]

        def monthly_sum(metric_key: str) -> dict:
            m = REABLEMENT_METRICS.get(metric_key, "")
            rows = sub[sub["Metric"] == m]
            agg  = rows.groupby("period")["Value"].sum()
            return {str(p): int(v) for p, v in agg.items() if not np.isnan(v)}

        def monthly_mean(metric_key: str) -> dict:
            m = REABLEMENT_METRICS.get(metric_key, "")
            rows = sub[sub["Metric"] == m]
            agg  = rows.groupby("period")["Value"].mean()
            return {str(p): round(float(v), 1) for p, v in agg.items() if not np.isnan(v)}

        def monthly_first(metric_key: str) -> dict:
            m = REABLEMENT_METRICS.get(metric_key, "")
            rows = sub[sub["Metric"] == m]
            agg  = rows.groupby("period")["Value"].first()
            return {str(p): int(v) for p, v in agg.items() if not np.isnan(v)}

        p1     = monthly_sum("p1_home")
        p2     = monthly_sum("p2_bed")
        nctr   = monthly_mean("nctr")
        cap_p1 = monthly_first("cap_p1")
        los14  = monthly_first("los14")

        # Trend calculation
        p1_vals = list(p1.values())
        trend_str = "insufficient data"
        if len(p1_vals) >= 3:
            first3 = sum(p1_vals[:3]) / 3
            last3  = sum(p1_vals[-3:]) / 3
            pct    = (last3 - first3) / first3 * 100 if first3 > 0 else 0
            trend_str = f"{'UP' if pct > 5 else 'DOWN' if pct < -5 else 'STABLE'} {pct:+.0f}% (first 3 vs last 3 months avg)"

        # Cap gap status
        cap_vals = list(cap_p1.values())
        cap_trend = "no data"
        if len(cap_vals) >= 3:
            if cap_vals[-1] > cap_vals[-3] * 1.1:
                cap_trend = "GROWING (pressure increasing)"
            elif cap_vals[-1] < cap_vals[-3] * 0.9:
                cap_trend = "SHRINKING (improving)"
            else:
                cap_trend = "STABLE"

        # A&E wait12h for this provider
        ae_sub = df_a[df_a["Org Code"] == code]
        ae_wait = {}
        if len(ae_sub) > 0:
            ae_grp = ae_sub.groupby("period")["wait_12h_plus"].sum()
            ae_wait = {str(p): int(v) for p, v in ae_grp.items()}

        lines.append(f"\n[{code}] {name}")
        lines.append(f"  P1 home reablement discharges/month: {p1}")
        lines.append(f"  P2 bed reablement discharges/month:  {p2}")
        lines.append(f"  P1+P2 trend: {trend_str}")
        lines.append(f"  NCTR avg/day: {nctr}")
        lines.append(f"  Capacity gap P1 (£/day avg): {cap_p1}")
        lines.append(f"  Capacity gap trend: {cap_trend}")
        lines.append(f"  LOS 14+ bed days: {los14}")
        if ae_wait:
            lines.append(f"  A&E Wait 12h+: {ae_wait}")

    # ── 2. Aggregate monthly across all tracked providers ─────────────────────
    lines.append("\n--- AGGREGATE (all tracked providers combined) ---")

    for mkey in ["p1_home", "p2_bed", "nctr"]:
        mname = REABLEMENT_METRICS[mkey]
        agg = (essex[essex["Metric"] == mname]
               .groupby("period")["Value"]
               .sum()
               .round(0))
        lines.append(f"  {mkey}: { {str(p): int(v) for p, v in agg.items()} }")

    # Total A&E wait12h across tracked providers
    ae_essex = df_a[df_a["Org Code"].isin(ESSEX_PROVIDER_CODES)]
    if len(ae_essex) > 0:
        ae_agg = ae_essex.groupby("period")["wait_12h_plus"].sum()
        lines.append(f"  A&E Wait 12h+ (total): { {str(p): int(v) for p, v in ae_agg.items()} }")

    # ── 3. Forecast & pressure ────────────────────────────────────────────────
    lines.append("\n--- FORECAST & SYSTEM PRESSURE ---")

    try:
        sm = model.get_summary()
        ps = model.get_pressure_status()
        fc = model.predict(horizon_days=7)

        lines.append(f"  Current pressure level: {ps['level'].upper()}")
        lines.append(f"  A&E Wait 12h+ latest: {ps['value']:,.0f} ({ps['percentile']}th percentile historically)")
        lines.append(f"  Pressure message: {ps['message']}")
        lines.append(f"  Daily baseline P1+P2: {sm.monthly_avg_p1p2 / 30:.0f} per day")
        lines.append(f"  Monthly avg P1+P2 (all tracked): {sm.monthly_avg_p1p2:.0f}")

        lines.append("  7-day forecast:")
        for _, row in fc.iterrows():
            lines.append(
                f"    {row['date'].strftime('%a %d %b')}: {row['forecast']:.0f} "
                f"(range {row['lower']:.0f}–{row['upper']:.0f}) — {row['interpretation']}"
            )

        lines.append(f"  Model notes:")
        for note in sm.model_notes:
            lines.append(f"    - {note}")

        lines.append(f"  Seasonal context: {_seasonal_note(sm)}")
    except Exception as e:
        lines.append(f"  Forecast unavailable: {e}")

    # ── 4. Key alerts ─────────────────────────────────────────────────────────
    lines.append("\n--- KEY ALERTS ---")
    alerts = _build_alerts(essex, df_a, periods)
    for alert in alerts:
        lines.append(f"  ! {alert}")
    if not alerts:
        lines.append("  No major alerts at this time.")

    # ── 5. Benchmark context ──────────────────────────────────────────────────
    lines.append("\n--- BENCHMARK CONTEXT ---")
    latest_p = max(periods)

    for level, label in [("National", "National England"), ("Region", "East of England")]:
        sub_level = df_s[df_s["Level"] == level]
        if level == "Region":
            sub_level = sub_level[sub_level["Region"] == "EAST OF ENGLAND"]
        p1n = sub_level[
            (sub_level["Metric"] == REABLEMENT_METRICS["p1_home"]) &
            (sub_level["period"] == latest_p)]["Value"].sum()
        p2n = sub_level[
            (sub_level["Metric"] == REABLEMENT_METRICS["p2_bed"]) &
            (sub_level["period"] == latest_p)]["Value"].sum()
        dn  = sub_level[
            (sub_level["Metric"] == REABLEMENT_METRICS["discharges"]) &
            (sub_level["period"] == latest_p)]["Value"].sum()
        share = (p1n + p2n) / dn * 100 if dn > 0 else 0
        lines.append(f"  {label} reablement share ({latest_p}): {share:.1f}%")

    # Tracked-providers share
    p1e = essex[(essex["Metric"] == REABLEMENT_METRICS["p1_home"]) & (essex["period"] == latest_p)]["Value"].sum()
    p2e = essex[(essex["Metric"] == REABLEMENT_METRICS["p2_bed"])  & (essex["period"] == latest_p)]["Value"].sum()
    de  = essex[(essex["Metric"] == REABLEMENT_METRICS["discharges"]) & (essex["period"] == latest_p)]["Value"].sum()
    share_e = (p1e + p2e) / de * 100 if de > 0 else 0
    lines.append(f"  Tracked providers reablement share ({latest_p}): {share_e:.1f}%")

    return "\n".join(lines)


def _seasonal_note(sm) -> str:
    """Returns a plain-English seasonal note for the current month."""
    import datetime
    current_month = datetime.date.today().strftime("%b")
    idx = sm.seasonal_index.get(current_month, 1.0)
    if idx > 1.05:
        return f"{current_month} is historically {(idx-1)*100:.0f}% ABOVE average — expect elevated demand."
    elif idx < 0.95:
        return f"{current_month} is historically {(1-idx)*100:.0f}% BELOW average — demand typically lower."
    else:
        return f"{current_month} is historically close to average demand."


def _build_alerts(essex: pd.DataFrame, df_a: pd.DataFrame, periods: list) -> list[str]:
    """Identifies key operational alerts from the data."""
    alerts = []

    # Alert: cap gap growing 3+ months for any priority provider
    for code in ["RAJ", "RDE", "RQW"]:
        sub  = essex[essex["Org Code"] == code]
        m    = REABLEMENT_METRICS["cap_p1"]
        rows = sub[sub["Metric"] == m].groupby("period")["Value"].first().sort_index()
        vals = list(rows.values)
        if len(vals) >= 3 and all(vals[i] < vals[i+1] for i in range(len(vals)-3, len(vals)-1)):
            alerts.append(
                f"{PROVIDER_NAMES.get(code, code)}: P1 capacity gap has been growing "
                f"for 3+ consecutive months (latest: £{vals[-1]:,.0f}/day avg)"
            )

    # Alert: A&E Wait 12h+ above 75th percentile
    ae_essex = df_a[df_a["Org Code"].isin(ESSEX_PROVIDER_CODES)]
    if len(ae_essex) > 0:
        ae_agg  = ae_essex.groupby("period")["wait_12h_plus"].sum().sort_index()
        ae_vals = list(ae_agg.values)
        if len(ae_vals) >= 4:
            pct75 = float(pd.Series(ae_vals).quantile(0.75))
            if ae_vals[-1] > pct75:
                alerts.append(
                    f"A&E Wait 12h+ is above 75th percentile: "
                    f"{ae_vals[-1]:,.0f} vs threshold {pct75:,.0f} — elevated referral pressure expected."
                )

    # Alert: P1+P2 declining trend in RAJ (largest provider)
    raj = essex[essex["Org Code"] == "RAJ"]
    p1  = raj[raj["Metric"] == REABLEMENT_METRICS["p1_home"]].groupby("period")["Value"].sum()
    p2  = raj[raj["Metric"] == REABLEMENT_METRICS["p2_bed"]].groupby("period")["Value"].sum()
    p12 = (p1 + p2).sort_index()
    vals = list(p12.values)
    if len(vals) >= 6:
        early = sum(vals[:3]) / 3
        late  = sum(vals[-3:]) / 3
        if late < early * 0.85:
            alerts.append(
                f"RAJ P1+P2 volume down {(1 - late/early)*100:.0f}% vs earlier months "
                f"— review whether this reflects capacity constraints or reporting change."
            )

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# OPENAI CHAT
# ─────────────────────────────────────────────────────────────────────────────

def get_openai_client() -> Optional["OpenAI"]:
    """Returns an OpenAI client using the API key from env or Streamlit secrets."""
    if not OPENAI_AVAILABLE:
        return None

    api_key = None

    # Try Streamlit secrets first (Streamlit Cloud deployment)
    try:
        import streamlit as st
        api_key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        pass

    # Fall back to environment variable (local dev with .env file)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return None

    return OpenAI(api_key=api_key)


def ask_question(
    question: str,
    data_context: str,
    conversation_history: list[dict],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Sends a question to OpenAI with the data context and conversation history.

    Parameters:
        question             : the user's current question
        data_context         : pre-computed data summary from build_data_context()
        conversation_history : list of {"role": "user"|"assistant", "content": "..."}
                               — last N turns of conversation for continuity
        model                : OpenAI model to use

    Returns:
        str : the assistant's answer
    """
    client = get_openai_client()

    if client is None:
        return (
            "⚠️ OpenAI API key not found. "
            "Please set OPENAI_API_KEY in your .env file (local) "
            "or in Streamlit secrets (cloud deployment)."
        )

    # Build messages: system prompt with data context + conversation history + new question
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(data_context=data_context),
        }
    ]

    # Add last 8 turns of history for conversational continuity
    # (keeps token count manageable)
    for turn in conversation_history[-8:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add the new question
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=600,
            temperature=0.3,  # low temperature = factual, consistent answers
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"⚠️ API error: {e}"


def is_api_configured() -> bool:
    """Returns True if the OpenAI API key is available."""
    client = get_openai_client()
    return client is not None
