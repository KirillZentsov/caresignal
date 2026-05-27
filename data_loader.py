"""
data_loader.py
==============
Data loading, cleaning and validation for CareSignal Reablement Intelligence.

Supported sources:
  - NHS England SitRep Discharge monthly CSV  ->  data/sitrep/sitrep_YYYY-MM.csv
  - NHS England A&E Attendances monthly CSV   ->  data/ae/ae_YYYY-MM.csv

Default provider scope — 5 acute NHS trusts across East of England,
London and South East (configurable via the constants below):
  RAJ  Mid & South Essex NHS Trust           (East of England)
  RDE  East Suffolk & NE Essex NHS Trust     (East of England)
  RQW  Princess Alexandra Hospital NHS Trust (East of England)
  RF4  Barking, Havering & Redbridge UH      (London)
  RYR  University Hospitals Sussex NHS Trust (South East)
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Default scope: 5 acute trusts monitored by the dashboard
ECL_PROVIDER_CODES: list[str] = ["RAJ", "RDE", "RQW", "RF4", "RYR"]

# Display names for cards and charts (short, no "NHS Foundation Trust" clutter)
ECL_DISPLAY_NAMES: dict[str, str] = {
    "RAJ": "Mid & South Essex NHS Trust",
    "RDE": "East Suffolk & NE Essex NHS Trust",
    "RQW": "Princess Alexandra Hospital NHS Trust",
    "RF4": "Barking, Havering & Redbridge UH NHS Trust",
    "RYR": "University Hospitals Sussex NHS Trust",
}

# Operational area label per trust
ECL_AREA: dict[str, str] = {
    "RAJ": "Essex Mid/South",
    "RDE": "Essex North East",
    "RQW": "Essex West",
    "RF4": "Havering",
    "RYR": "West Sussex",
}

# Each trust's NHS region — used for regional benchmark comparisons
ECL_REGION_MAP: dict[str, str] = {
    "RAJ": "EAST OF ENGLAND",
    "RDE": "EAST OF ENGLAND",
    "RQW": "EAST OF ENGLAND",
    "RF4": "LONDON",
    "RYR": "SOUTH EAST",
}

# Regions that appear in SitRep and are needed for benchmark comparisons
ECL_BENCHMARK_REGIONS: list[str] = ["EAST OF ENGLAND", "LONDON", "SOUTH EAST"]

# Region org codes in SitRep
REGION_ORG_CODES: dict[str, str] = {
    "EAST OF ENGLAND": "Y61",
    "LONDON":          "Y56",
    "SOUTH EAST":      "Y59",
    "NATIONAL":        "ENG",
}

# ─────────────────────────────────────────────────────────────────────────────
# LEGACY CONSTANTS  (kept for backward compatibility with lag_model)
# ─────────────────────────────────────────────────────────────────────────────

ESSEX_PROVIDER_CODES: list[str] = ECL_PROVIDER_CODES   # alias
ESSEX_ICB_CODES: list[str] = [
    "QH8", "QJG", "QM7", "QMM", "QUE", "QHG",
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE METRICS
# All metric strings normalised (no \xa0 non-breaking spaces).
# ─────────────────────────────────────────────────────────────────────────────

# Core operational metrics (+ P1 discharge volume)
ECL_METRICS: dict[str, str] = {
    # ── Demand pipeline ────────────────────────────────────────────────────
    "nctr": (
        "Number of patients who no longer meet the criteria to reside"
    ),
    "discharged": (
        "Number of patients discharged"
    ),
    "nctr_remaining": (
        "Number of patients remaining in hospital who no longer meet "
        "the criteria to reside"
    ),
    # ── P1 capacity signals ────────────────────────────────────────────────
    "cap_gap_p1": (
        "Capacity – Home-based rehabilitation, reablement or recovery "
        "services not yet available (Pathway 1)"
    ),
    "interface_p1": (
        "Interface process – Home based rehabilitation, reablement or "
        "recovery service arrangements still underway (Pathway 1)"
    ),
    # ── P1 discharge volume (direct reablement output) ─────────────────────
    "p1_home": (
        "Pathway 1: Discharge to a domestic home, hotel, or other "
        "temporary accommodation, or hospice at home with rehabilitation, "
        "reablement and recovery"
    ),
}

# Full metric dictionary (superset — used by lag_model and chat_engine)
REABLEMENT_METRICS: dict[str, str] = {
    **ECL_METRICS,
    "nctr_remain":    ECL_METRICS["nctr_remaining"],   # alias
    "p2_bed": (
        "Pathway 2: Short-term bed/hospice for rehabilitation, reablement "
        "and recovery / end of life care"
    ),
    "cap_p1":         ECL_METRICS["cap_gap_p1"],        # alias
    "cap_p2": (
        "Capacity – Bed-based rehabilitation, reablement or recovery "
        "services not yet available (Pathway 2)"
    ),
    "iface_p1":       ECL_METRICS["interface_p1"],      # alias
    "iface_p2": (
        "Interface process – Bed-based rehabilitation, reablement or "
        "recovery service arrangements still underway (Pathway 2)"
    ),
    "discharges":     ECL_METRICS["discharged"],        # alias
    "delay_cost": (
        "Total cost of delayed bed days in the month"
    ),
    "delay_days": (
        "Total number of delayed bed days in the month"
    ),
    "nctr_daily_avg": (
        "Daily Average - Number of patients remaining in hospital who no "
        "longer meet the criteria to reside"
    ),
    "los7":  "Number of additional bed days, patients with length of stay of 7 days or over",
    "los14": "Number of additional bed days, patients with length of stay of 14 days or over",
    "los21": "Number of additional bed days, patients with length of stay of 21 days or over",
    "wait_therapy": (
        "Hospital process – Awaiting therapy review of need for supported discharge"
    ),
    "wait_cth": (
        "Care transfer hub process – Waiting for confirmation of immediate "
        "care needs and pathway"
    ),
    "wellbeing": (
        "Wellbeing concerns – Patient /family/carer concerns over discharge readiness"
    ),
}

# Metrics that carry one monthly total row (not daily rows)
MONTHLY_TOTAL_METRICS: frozenset[str] = frozenset({
    "delay_cost", "delay_days", "nctr_daily_avg",
    "los7", "los14", "los21",
    "cap_gap_p1", "cap_p1",           # LOS-7+ group in SitRep
    "interface_p1", "iface_p1",
})

# Metrics in the "Additional bed days" group whose date = last day of prior month
ADDITIONAL_BED_DAYS_METRICS: list[str] = [
    "Number of additional bed days, patients with length of stay of 7 days or over",
    "Number of additional bed days, patients with length of stay of 14 days or over",
    "Number of additional bed days, patients with length of stay of 21 days or over",
]

# Metric groups where NHS sometimes embeds wrong-month data (drop those rows)
DROP_IF_OUTSIDE_PERIOD_GROUPS: set[str] = {"Delay reason LOS 7+"}

# Metrics only available from August 2025 onwards
METRICS_FROM_AUG_2025: list[str] = [
    "Daily Average - Number of patients remaining in hospital who no longer meet the criteria to reside",
    "Total cost of delayed bed days in the month",
    "Total number of delayed bed days in the month",
]

# ─────────────────────────────────────────────────────────────────────────────
# QUALITY REPORT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FileQuality:
    """Quality check result for a single file."""
    filename:   str
    period:     Optional[pd.Period]
    file_type:  str          # 'sitrep' | 'ae'
    status:     str          # 'ok' | 'warning' | 'error'
    total_rows: int = 0
    issues:     list[str] = field(default_factory=list)
    warnings:   list[str] = field(default_factory=list)
    info:       list[str] = field(default_factory=list)

    @property
    def emoji(self) -> str:
        return {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(self.status, "❓")


@dataclass
class QualityReport:
    """Aggregated quality report across all loaded files."""
    files:                  list[FileQuality] = field(default_factory=list)
    sitrep_periods_loaded:  list[pd.Period]   = field(default_factory=list)
    ae_periods_loaded:      list[pd.Period]   = field(default_factory=list)
    sitrep_date_range:      Optional[str]     = None
    ae_date_range:          Optional[str]     = None
    lag_note:               str               = ""

    @property
    def has_errors(self) -> bool:
        return any(f.status == "error" for f in self.files)

    @property
    def summary(self) -> str:
        ok   = sum(1 for f in self.files if f.status == "ok")
        warn = sum(1 for f in self.files if f.status == "warning")
        err  = sum(1 for f in self.files if f.status == "error")
        return f"{ok} OK · {warn} warnings · {err} errors"


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_period_from_filename(filename: str) -> Optional[pd.Period]:
    """Extracts YYYY-MM period from filename."""
    m = re.search(r"(\d{4})-(\d{2})\.csv$", filename, re.IGNORECASE)
    if m:
        return pd.Period(f"{m.group(1)}-{m.group(2)}", freq="M")
    return None


def _normalise_metric_name(name: str) -> str:
    """Strips non-breaking spaces and whitespace from metric names."""
    return name.replace("\xa0", " ").strip()


def _short_provider_name(name: str) -> str:
    """Returns a shortened trust name for chart labels."""
    return (
        name.replace(" NHS FOUNDATION TRUST", "")
            .replace(" NHS TRUST", "")
            .replace(" NATIONAL HEALTH SERVICE TRUST", "")
            .replace(" TEACHING HOSPITALS", "")
            .strip()
    )


# ─────────────────────────────────────────────────────────────────────────────
# SITREP LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_sitrep_file(path: Path) -> tuple[pd.DataFrame, FileQuality]:
    """Loads one SitRep file with all quality rules applied."""
    qf = FileQuality(
        filename=path.name,
        period=_parse_period_from_filename(path.name),
        file_type="sitrep",
        status="ok",
    )

    if qf.period is None:
        qf.issues.append(
            f"Cannot extract period from '{path.name}'. "
            "Expected: sitrep_YYYY-MM.csv"
        )
        qf.status = "error"
        return pd.DataFrame(), qf

    # Auto-detect BOM encoding (Feb 2026 uses utf-8-sig)
    df = None
    for enc in ("utf-8-sig", "cp1252", "utf-8"):
        try:
            candidate = pd.read_csv(path, encoding=enc, low_memory=False)
            if "Period" in candidate.columns:
                df = candidate
                if enc != "cp1252":
                    qf.info.append(f"Encoding: {enc} (BOM detected)")
                break
        except Exception:
            continue
    if df is None:
        qf.issues.append("Cannot read file with utf-8-sig, cp1252 or utf-8.")
        qf.status = "error"
        return pd.DataFrame(), qf

    qf.total_rows = len(df)

    # Required columns
    expected = {"Period", "Level", "Region", "ICB", "Org Code",
                "Org Name", "Metric", "Metric Type", "Metric Group", "Value"}
    missing = expected - set(df.columns)
    if missing:
        qf.issues.append(f"Missing columns: {missing}")
        qf.status = "error"
        return pd.DataFrame(), qf

    df["Value"]        = pd.to_numeric(df["Value"], errors="coerce")
    df["Period"]       = pd.to_datetime(df["Period"], dayfirst=True, errors="coerce")
    df["Metric"]       = df["Metric"].apply(_normalise_metric_name)
    df["Metric Group"] = df["Metric Group"].apply(_normalise_metric_name)

    # Outside-period mask
    outside = (
        (df["Period"].dt.month != qf.period.month) |
        (df["Period"].dt.year  != qf.period.year)
    ) & df["Period"].notna()

    # Rule 5a: Additional bed days — reassign to file period
    add_bed = df["Metric"].isin(ADDITIONAL_BED_DAYS_METRICS)
    n = int((add_bed & outside).sum())
    if n:
        correct = pd.Timestamp(qf.period.year, qf.period.month, 1)
        df.loc[add_bed & outside, "Period"] = correct
        qf.info.append(
            f"Reassigned {n} 'Additional bed days' rows -> {correct.strftime('%b %Y')}"
        )

    # Rule 5b: Drop wrong-month Delay reason LOS 7+ rows
    drop_mask = df["Metric Group"].isin(DROP_IF_OUTSIDE_PERIOD_GROUPS) & outside
    n_drop = int(drop_mask.sum())
    if n_drop:
        bad = df.loc[drop_mask, "Period"].dt.to_period("M").unique()
        qf.info.append(
            f"Dropped {n_drop} 'Delay reason LOS 7+' rows "
            f"with wrong-month dates {[str(p) for p in bad]}."
        )
        df = df[~drop_mask].copy()
        outside = (
            (df["Period"].dt.month != qf.period.month) |
            (df["Period"].dt.year  != qf.period.year)
        ) & df["Period"].notna()

    # Rule 6: Reassign remaining out-of-period rows
    remaining_outside = outside & ~df["Metric"].isin(ADDITIONAL_BED_DAYS_METRICS)
    n_r = int(remaining_outside.sum())
    if n_r:
        correct = pd.Timestamp(qf.period.year, qf.period.month, 1)
        df.loc[remaining_outside, "Period"] = correct
        qf.info.append(f"Reassigned {n_r} out-of-period rows -> file period")

    # Rule 7: All 5 monitored providers must be present
    prov_df   = df[df["Level"] == "PROVIDER"]
    found     = set(prov_df["Org Code"].unique()) & set(ECL_PROVIDER_CODES)
    missing_e = set(ECL_PROVIDER_CODES) - found
    if missing_e:
        qf.warnings.append(f"Missing monitored providers: {', '.join(sorted(missing_e))}")
        if qf.status == "ok":
            qf.status = "warning"
    qf.info.append(f"Monitored providers: {len(found)}/5")

    # Rule 8: Flag expected absence of new metrics before Aug 2025
    if qf.period < pd.Period("2025-08", freq="M"):
        if any(m not in df["Metric"].values for m in METRICS_FROM_AUG_2025):
            qf.info.append(
                "Delay cost / daily avg NCTR not available before Aug 2025 — expected."
            )

    # Rule 9: Warn if key providers have all-zero NCTR
    for code in {"RAJ", "RF4", "RYR"}:
        p_data = prov_df[
            (prov_df["Org Code"] == code) &
            (prov_df["Metric"] == ECL_METRICS["nctr"])
        ]
        if len(p_data) > 0 and p_data["Value"].dropna().eq(0).all():
            qf.warnings.append(
                f"All NCTR values are 0 for provider {code} — possible missing data."
            )
            if qf.status == "ok":
                qf.status = "warning"

    # Helper columns
    df["file_period"]    = str(qf.period)
    df["is_ecl"]         = df["Org Code"].isin(ECL_PROVIDER_CODES)
    df["is_essex_icb"]   = df["Org Code"].isin(ESSEX_ICB_CODES)
    df["ecl_area"]       = df["Org Code"].map(ECL_AREA)
    df["ecl_region"]     = df["Org Code"].map(ECL_REGION_MAP)
    df["display_name"]   = df["Org Code"].map(ECL_DISPLAY_NAMES)
    df["short_name"]     = df["Org Name"].apply(_short_provider_name)

    if qf.status == "ok" and qf.warnings:
        qf.status = "warning"

    return df, qf


# ─────────────────────────────────────────────────────────────────────────────
# A&E LOADER
# ─────────────────────────────────────────────────────────────────────────────

def _load_ae_file(path: Path) -> tuple[pd.DataFrame, FileQuality]:
    """Loads one A&E file with quality rules applied."""
    qf = FileQuality(
        filename=path.name,
        period=_parse_period_from_filename(path.name),
        file_type="ae",
        status="ok",
    )

    if qf.period is None:
        qf.issues.append(
            f"Cannot extract period from '{path.name}'. Expected: ae_YYYY-MM.csv"
        )
        qf.status = "error"
        return pd.DataFrame(), qf

    try:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, encoding="cp1252", low_memory=False)
            qf.info.append("Read with encoding=cp1252")
        except Exception as e:
            qf.issues.append(f"File read error: {e}")
            qf.status = "error"
            return pd.DataFrame(), qf
    except Exception as e:
        qf.issues.append(f"File read error: {e}")
        qf.status = "error"
        return pd.DataFrame(), qf

    qf.total_rows = len(df)

    expected = {
        "Org Code", "Org name",
        "A&E attendances Type 1",
        "Emergency admissions via A&E - Type 1",
    }
    missing = expected - set(df.columns)
    if missing:
        qf.issues.append(f"Missing columns: {missing}")
        qf.status = "error"
        return pd.DataFrame(), qf

    df = df[df["Org Code"] != "Total"].copy()

    numeric_cols = [
        "A&E attendances Type 1",
        "A&E attendances Type 2",
        "A&E attendances Other A&E Department",
        "Attendances over 4hrs Type 1",
        "Patients who have waited 4-12 hs from DTA to admission",
        "Patients who have waited 12+ hrs from DTA to admission",
        "Emergency admissions via A&E - Type 1",
        "Emergency admissions via A&E - Type 2",
        "Emergency admissions via A&E - Other A&E department",
        "Other emergency admissions",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived columns
    df["total_attendances"] = (
        df.get("A&E attendances Type 1", 0) +
        df.get("A&E attendances Type 2", 0) +
        df.get("A&E attendances Other A&E Department", 0)
    )
    df["emerg_admissions_t1"] = df.get("Emergency admissions via A&E - Type 1", 0)
    df["wait_12h_plus"]       = df.get("Patients who have waited 12+ hrs from DTA to admission", 0)
    df["wait_4_12h"]          = df.get("Patients who have waited 4-12 hs from DTA to admission", 0)
    df["breaches_4h_t1"]      = df.get("Attendances over 4hrs Type 1", 0)

    # Monitored providers coverage check
    found     = set(df["Org Code"].unique()) & set(ECL_PROVIDER_CODES)
    missing_e = set(ECL_PROVIDER_CODES) - found
    if missing_e:
        qf.warnings.append(f"Missing monitored providers in A&E: {', '.join(sorted(missing_e))}")
        if qf.status == "ok":
            qf.status = "warning"

    # Warn on suspicious zeros for large trusts
    for code in ["RAJ", "RF4", "RYR"]:
        row = df[df["Org Code"] == code]
        if len(row) > 0 and row["emerg_admissions_t1"].iloc[0] == 0:
            qf.warnings.append(
                f"Emergency admissions T1 = 0 for {code} — please verify."
            )
            if qf.status == "ok":
                qf.status = "warning"

    # Helper columns
    df["file_period"]  = str(qf.period)
    df["period"]       = pd.Period(str(qf.period), freq="M")
    df["is_ecl"]       = df["Org Code"].isin(ECL_PROVIDER_CODES)
    df["ecl_area"]     = df["Org Code"].map(ECL_AREA)
    df["ecl_region"]   = df["Org Code"].map(ECL_REGION_MAP)
    df["display_name"] = df["Org Code"].map(ECL_DISPLAY_NAMES)
    df = df.rename(columns={"Org name": "Org Name"})
    df["short_name"]   = df["Org Name"].apply(_short_provider_name)

    qf.info.append(f"Monitored providers: {len(found)}/5")
    if qf.status == "ok" and qf.warnings:
        qf.status = "warning"

    return df, qf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _apply_period_summary(df_sitrep: pd.DataFrame, df_ae: pd.DataFrame, report: QualityReport) -> QualityReport:
    """Populate date ranges and lag note from loaded frames."""
    if len(df_sitrep) > 0 and "period" in df_sitrep.columns:
        report.sitrep_periods_loaded = sorted(
            [p for p in df_sitrep["period"].dropna().unique()]
        )

    if len(df_ae) > 0 and "period" in df_ae.columns:
        report.ae_periods_loaded = sorted(
            [p for p in df_ae["period"].dropna().unique()]
        )

    if report.sitrep_periods_loaded:
        p = sorted(report.sitrep_periods_loaded)
        report.sitrep_date_range = (
            f"{p[0].strftime('%b %Y')} - {p[-1].strftime('%b %Y')}"
        )

    if report.ae_periods_loaded:
        p = sorted(report.ae_periods_loaded)
        report.ae_date_range = (
            f"{p[0].strftime('%b %Y')} - {p[-1].strftime('%b %Y')}"
        )

    if report.sitrep_periods_loaded and report.ae_periods_loaded:
        latest_s = max(report.sitrep_periods_loaded)
        latest_a = max(report.ae_periods_loaded)
        diff = (
            (latest_a.year * 12 + latest_a.month) -
            (latest_s.year * 12 + latest_s.month)
        )
        if diff > 0:
            report.lag_note = (
                f"A&E data current to {latest_a.strftime('%b %Y')}, "
                f"SitRep to {latest_s.strftime('%b %Y')} ({diff} month gap)."
            )
        elif diff < 0:
            report.lag_note = (
                f"SitRep is {abs(diff)} month(s) ahead of A&E. "
                f"A&E data current to {latest_a.strftime('%b %Y')}."
            )
        else:
            report.lag_note = (
                f"Both sources current to {latest_s.strftime('%b %Y')}."
            )

    return report


def _processed_dir_from_raw_dirs(
    sitrep_dir: str | Path,
    processed_dir: str | Path | None = None,
) -> Path:
    """Resolve data/processed location."""
    if processed_dir is not None:
        return Path(processed_dir)
    return Path(sitrep_dir).resolve().parent / "processed"


def _read_processed_pair(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Load cached parquet files if present.

    Supported filenames are intentionally flexible to avoid breaking older
    local builds:
      - sitrep.parquet / ae.parquet
      - df_sitrep.parquet / df_ae.parquet
      - sitrep_processed.parquet / ae_processed.parquet
    """
    sitrep_candidates = [
        "sitrep.parquet",
        "df_sitrep.parquet",
        "sitrep_processed.parquet",
        "sitrep_clean.parquet",
    ]
    ae_candidates = [
        "ae.parquet",
        "df_ae.parquet",
        "ae_processed.parquet",
        "ae_clean.parquet",
    ]

    sitrep_path = next((processed_dir / n for n in sitrep_candidates if (processed_dir / n).exists()), None)
    ae_path = next((processed_dir / n for n in ae_candidates if (processed_dir / n).exists()), None)

    if sitrep_path is None or ae_path is None:
        return None

    df_sitrep = pd.read_parquet(sitrep_path)
    df_ae = pd.read_parquet(ae_path)

    # Ensure period dtype survives parquet round-trip
    if "period" in df_sitrep.columns and not isinstance(df_sitrep["period"].dtype, pd.PeriodDtype):
        df_sitrep["period"] = pd.PeriodIndex(df_sitrep["period"].astype(str), freq="M")
    if "period" in df_ae.columns and not isinstance(df_ae["period"].dtype, pd.PeriodDtype):
        df_ae["period"] = pd.PeriodIndex(df_ae["period"].astype(str), freq="M")

    return df_sitrep, df_ae


def load_all_data(
    sitrep_dir: str | Path = "data/sitrep",
    ae_dir:     str | Path = "data/ae",
    prefer_processed: bool = True,
    processed_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """
    Reads all monthly files from both directories.

    If prefer_processed=True, tries to load prepared parquet files from
    data/processed first. If they are not present, falls back to raw CSV files.

    Returns:
        df_sitrep : combined SitRep DataFrame (all months, all providers)
        df_ae     : combined A&E DataFrame (all months, all providers)
        report    : QualityReport with per-file status
    """
    report = QualityReport()
    # Fast cloud/demo path: read prepared parquet if available.
    print(f"prefer_processed={prefer_processed}")

    # Fast cloud/demo path: read prepared parquet if available.
    if prefer_processed:
        print("Trying to load processed parquet files...")

        pdir = _processed_dir_from_raw_dirs(sitrep_dir, processed_dir)
        processed = _read_processed_pair(pdir)
        if processed is not None:
            df_sitrep, df_ae = processed
            qf_s = FileQuality(
                filename=str(pdir / "sitrep*.parquet"),
                period=None,
                file_type="sitrep",
                status="ok",
                total_rows=len(df_sitrep),
                info=["Loaded from processed parquet"],
            )
            qf_a = FileQuality(
                filename=str(pdir / "ae*.parquet"),
                period=None,
                file_type="ae",
                status="ok",
                total_rows=len(df_ae),
                info=["Loaded from processed parquet"],
            )
            report.files.extend([qf_s, qf_a])
            report = _apply_period_summary(df_sitrep, df_ae, report)
            return df_sitrep, df_ae, report

    # Raw ETL path: read source CSV files.
    sitrep_frames: list[pd.DataFrame] = []
    ae_frames:     list[pd.DataFrame] = []

    for fpath in sorted(Path(sitrep_dir).glob("sitrep_*.csv")):
        df, qf = _load_sitrep_file(fpath)
        report.files.append(qf)
        if qf.status != "error" and len(df) > 0:
            sitrep_frames.append(df)
            if qf.period:
                report.sitrep_periods_loaded.append(qf.period)

    for fpath in sorted(Path(ae_dir).glob("ae_*.csv")):
        df, qf = _load_ae_file(fpath)
        report.files.append(qf)
        if qf.status != "error" and len(df) > 0:
            ae_frames.append(df)
            if qf.period:
                report.ae_periods_loaded.append(qf.period)

    df_sitrep = (
        pd.concat(sitrep_frames, ignore_index=True)
        if sitrep_frames else pd.DataFrame()
    )
    df_ae = (
        pd.concat(ae_frames, ignore_index=True)
        if ae_frames else pd.DataFrame()
    )

    if len(df_sitrep) > 0:
        df_sitrep["period"]       = pd.PeriodIndex(df_sitrep["file_period"], freq="M")
        df_sitrep["period_label"] = df_sitrep["period"].dt.strftime("%b %Y")
        df_sitrep.sort_values(["file_period", "Period", "Org Code"], inplace=True)
        df_sitrep.reset_index(drop=True, inplace=True)

    report = _apply_period_summary(df_sitrep, df_ae, report)

    return df_sitrep, df_ae, report


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE EXTRACTORS
# ─────────────────────────────────────────────────────────────────────────────

def get_ecl_sitrep(df_sitrep: pd.DataFrame) -> pd.DataFrame:
    """Returns only monitored-provider rows."""
    return df_sitrep[df_sitrep["is_ecl"]].copy()


def get_ecl_ae(df_ae: pd.DataFrame) -> pd.DataFrame:
    """Returns only monitored-provider rows from A&E data."""
    return df_ae[df_ae["is_ecl"]].copy()


def get_provider_monthly(
    df_sitrep: pd.DataFrame,
    metric_key: str,
    level: str = "PROVIDER",
    codes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Monthly aggregate for a metric at a given level.

    Parameters:
        df_sitrep  : full SitRep DataFrame
        metric_key : key from REABLEMENT_METRICS
        level      : 'PROVIDER' | 'ICB' | 'Region' | 'National'
        codes      : optional list of Org Codes to filter (None = all)

    Returns DataFrame: period | Org Code | display_name | Value
    """
    metric_name = REABLEMENT_METRICS.get(metric_key)
    if metric_name is None:
        raise ValueError(
            f"Unknown metric_key='{metric_key}'. "
            f"Available: {list(REABLEMENT_METRICS.keys())}"
        )

    mask = (df_sitrep["Level"] == level) & (df_sitrep["Metric"] == metric_name)
    if codes:
        mask &= df_sitrep["Org Code"].isin(codes)
    df = df_sitrep[mask].copy()

    # Monthly totals get first-value aggregation; daily metrics get sum
    if metric_key in MONTHLY_TOTAL_METRICS:
        agg_fn = lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else np.nan
    else:
        agg_fn = "sum"

    group_cols = ["period", "period_label", "Org Code", "Org Name", "short_name"]
    for extra in ["display_name", "ecl_area", "ecl_region", "Region"]:
        if extra in df.columns:
            group_cols.append(extra)
    group_cols = list(dict.fromkeys(c for c in group_cols if c in df.columns))

    if callable(agg_fn):
        # For lambda functions, apply per-group explicitly
        result = (df.groupby(group_cols)["Value"]
                    .apply(agg_fn)
                    .reset_index())
    else:
        result = (df.groupby(group_cols, as_index=False)["Value"]
                    .agg(agg_fn))
    # Rename Value column if needed
    if "Value" not in result.columns and len(result.columns) > len(group_cols):
        result = result.rename(columns={result.columns[-1]: "Value"})
    return result.sort_values("period").reset_index(drop=True)


def get_daily_metric(
    df_sitrep: pd.DataFrame,
    metric_key: str,
    codes: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Daily values for a metric, for given provider codes.
    Defaults to all monitored providers if codes is None.
    """
    if codes is None:
        codes = ECL_PROVIDER_CODES

    metric_name = REABLEMENT_METRICS.get(metric_key)
    if metric_name is None:
        raise ValueError(f"Unknown metric_key='{metric_key}'.")

    mask = (
        (df_sitrep["Level"] == "PROVIDER") &
        (df_sitrep["Org Code"].isin(codes)) &
        (df_sitrep["Metric"] == metric_name)
    )
    return (
        df_sitrep[mask][
            ["Period", "period_label", "period", "Org Code",
             "display_name", "ecl_area", "Value"]
        ]
        .sort_values(["Org Code", "Period"])
        .reset_index(drop=True)
    )


def get_regional_benchmark(
    df_sitrep: pd.DataFrame,
    metric_key: str,
    regions: Optional[list[str]] = None,
    include_national: bool = True,
) -> pd.DataFrame:
    """
    Monthly metric values at Region and National level for benchmark.

    Parameters:
        regions : list of region names (e.g. ['EAST OF ENGLAND', 'LONDON'])
                  Defaults to all benchmark regions.
    """
    if regions is None:
        regions = ECL_BENCHMARK_REGIONS

    metric_name = REABLEMENT_METRICS.get(metric_key)
    if metric_name is None:
        raise ValueError(f"Unknown metric_key='{metric_key}'.")

    frames = []

    # Regional rows
    reg_mask = (
        (df_sitrep["Level"] == "Region") &
        (df_sitrep["Region"].isin(regions)) &
        (df_sitrep["Metric"] == metric_name)
    )
    if metric_key in MONTHLY_TOTAL_METRICS:
        reg_agg = (df_sitrep[reg_mask]
                   .groupby(["period","period_label","Region"])["Value"]
                   .first().reset_index())
    else:
        reg_agg = (df_sitrep[reg_mask]
                   .groupby(["period","period_label","Region"])["Value"]
                   .sum().reset_index())
    reg_agg["label"] = reg_agg["Region"].str.title()
    reg_agg["level"] = "Region"
    frames.append(reg_agg[["period","period_label","label","level","Value"]])

    # National row
    if include_national:
        nat_mask = (
            (df_sitrep["Level"] == "National") &
            (df_sitrep["Metric"] == metric_name)
        )
        if metric_key in MONTHLY_TOTAL_METRICS:
            nat_agg = (df_sitrep[nat_mask]
                       .groupby(["period","period_label"])["Value"]
                       .first().reset_index())
        else:
            nat_agg = (df_sitrep[nat_mask]
                       .groupby(["period","period_label"])["Value"]
                       .sum().reset_index())
        nat_agg["label"] = "National"
        nat_agg["level"] = "National"
        frames.append(nat_agg[["period","period_label","label","level","Value"]])

    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["label","period"])
        .reset_index(drop=True)
    )


def get_quality_report(report: QualityReport) -> pd.DataFrame:
    """Converts QualityReport to a display-ready DataFrame."""
    rows = []
    for f in report.files:
        rows.append({
            "Status":   f.emoji,
            "File":     f.filename,
            "Period":   str(f.period) if f.period else "-",
            "Type":     f.file_type.upper(),
            "Rows":     f.total_rows,
            "Issues":   " | ".join(f.issues)   if f.issues   else "",
            "Warnings": " | ".join(f.warnings) if f.warnings else "",
            "Info":     " | ".join(f.info)     if f.info     else "",
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    sitrep_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sitrep"
    ae_dir     = sys.argv[2] if len(sys.argv) > 2 else "data/ae"

    df_s, df_a, rpt = load_all_data(sitrep_dir, ae_dir)
    print(f"Quality  : {rpt.summary}")
    print(f"SitRep   : {rpt.sitrep_date_range}")
    print(f"A&E      : {rpt.ae_date_range}")
    print(f"Lag note : {rpt.lag_note}")

    ecl = get_ecl_sitrep(df_s)
    print(f"\nMonitored SitRep rows : {len(ecl):,}")
    print(f"Monitored A&E rows    : {len(get_ecl_ae(df_a)):,}")

    print("\nMonitored providers summary (latest period):")
    latest = max(df_s["period"].unique())
    for code in ECL_PROVIDER_CODES:
        sub = ecl[
            (ecl["Org Code"] == code) &
            (ecl["period"] == latest) &
            (ecl["Metric"] == ECL_METRICS["nctr"])
        ]
        nctr = sub["Value"].mean()
        print(f"  {code} {ECL_DISPLAY_NAMES[code]:<50} NCTR avg {nctr:.0f}/day")
