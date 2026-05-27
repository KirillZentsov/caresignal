"""
lag_model.py
============
P1 demand forecast model for CareSignal Reablement Intelligence.

Scope: the 5 monitored acute trust partners, Pathway 1 (home reablement) only.

Three-component model:
  1. Per-provider daily baseline  — historical avg P1 discharges per day
  2. Day-of-week multiplier       — from daily NCTR patterns (±23%)
  3. Seasonal multiplier          — from monthly P1 patterns (±11%)
  4. Pressure adjustment          — A&E Wait 12h+ -> P1 (r=0.725, p=0.003)

Known data anomalies excluded from fitting:
  RDE  2026-01  P1=702 vs normal ~240 (3x spike, likely reporting change)
  RF4  step-change Dec 2025: baseline computed from Dec 2025 onward
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from data_loader import (
        ECL_PROVIDER_CODES, ECL_DISPLAY_NAMES, ECL_METRICS,
        ECL_REGION_MAP, REABLEMENT_METRICS,
    )
except ImportError:
    # Standalone fallback
    ECL_PROVIDER_CODES  = ["RAJ", "RDE", "RQW", "RF4", "RYR"]
    ECL_DISPLAY_NAMES   = {
        "RAJ": "Mid & South Essex NHS Trust",
        "RDE": "East Suffolk & NE Essex NHS Trust",
        "RQW": "Princess Alexandra Hospital NHS Trust",
        "RF4": "Barking, Havering & Redbridge UH NHS Trust",
        "RYR": "University Hospitals Sussex NHS Trust",
    }
    ECL_METRICS = {
        "p1_home":        "Pathway 1: Discharge to a domestic home, hotel, or other temporary accommodation, or hospice at home with rehabilitation, reablement and recovery",
        "nctr":           "Number of patients who no longer meet the criteria to reside",
        "nctr_remaining": "Number of patients remaining in hospital who no longer meet the criteria to reside",
        "cap_gap_p1":     "Capacity \u2013 Home-based rehabilitation, reablement or recovery services not yet available (Pathway 1)",
        "interface_p1":   "Interface process \u2013 Home based rehabilitation, reablement or recovery service arrangements still underway (Pathway 1)",
        "discharged":     "Number of patients discharged",
    }
    REABLEMENT_METRICS = ECL_METRICS

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Known anomalies: (org_code, period_str) — excluded from model fitting
KNOWN_ANOMALIES: list[tuple[str, str]] = [
    ("RDE", "2026-01"),   # P1 spike 702 vs normal ~240 — likely reporting change
]

# RF4 had a structural step-change in Dec 2025 (P1 went from ~35 to ~120/month)
# Baseline for RF4 is computed from Dec 2025 onward only
RF4_BASELINE_FROM = pd.Period("2025-12", freq="M")

# ─────────────────────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProviderForecast:
    """14-day forecast for one monitored provider."""
    org_code:     str
    display_name: str
    baseline_daily: float         # avg P1 per day
    forecast_df:  pd.DataFrame    # date | dow_name | forecast | lower | upper | load


@dataclass
class ModelSummary:
    """Summary of fitted model parameters."""
    n_months:          int
    date_range:        str
    ecl_monthly_avg_p1: float        # total tracked P1 per month
    provider_baselines: dict[str, float]   # {code: daily_baseline}

    dow_coefficients:  dict[str, float]    # {day_name: multiplier}
    seasonal_index:    dict[str, float]    # {month_abbr: multiplier}

    pressure_r:        float
    pressure_p:        float
    pressure_slope:    float
    pressure_intercept: float

    anomalies_excluded: list[tuple[str, str]]
    model_notes:       list[str]

    last_ae_period:    Optional[str]   = None
    last_wait12h:      Optional[float] = None
    current_pressure:  str             = "unknown"   # low|normal|elevated|high


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ReablementForecastModel:
    """
    P1 demand forecast model for the 5 trust partners.

    Fit once after loading data, then call predict() per provider.
    """

    def __init__(self) -> None:
        self._fitted              = False
        self._ts:   Optional[pd.DataFrame] = None   # monthly time series
        self._daily: Optional[pd.DataFrame] = None  # daily NCTR
        self._summary: Optional[ModelSummary] = None

        # Fitted parameters
        self._dow_coefs:     dict[int, float]   = {}
        self._seasonal_coefs: dict[int, float]  = {}
        self._provider_baselines: dict[str, float] = {}
        self._residual_std:  float = 0.0
        self._pressure_r:    float = 0.0
        self._pressure_p:    float = 1.0
        self._pressure_slope: float = 0.0
        self._pressure_intercept: float = 0.0
        self._wait12h_history: Optional[pd.Series] = None

    # ── PUBLIC API ────────────────────────────────────────────────────────────

    def fit(
        self,
        df_sitrep: pd.DataFrame,
        df_ae:     pd.DataFrame,
        provider_codes: Optional[list[str]] = None,
    ) -> "ReablementForecastModel":
        """Fits the model on historical data."""
        if provider_codes is None:
            provider_codes = ECL_PROVIDER_CODES

        self._ts    = self._build_monthly_ts(df_sitrep, df_ae, provider_codes)
        self._daily = self._build_daily_nctr(df_sitrep, provider_codes)

        self._fit_dow_pattern()
        self._fit_seasonal_pattern()
        self._fit_pressure_signal()
        self._fit_provider_baselines()
        self._fit_residual_std()
        self._build_summary(provider_codes)

        self._fitted = True
        return self

    def predict(
        self,
        horizon_days:  int = 14,
        start_date:    Optional[pd.Timestamp] = None,
        latest_wait12h: Optional[float] = None,
        org_code:      Optional[str] = None,
    ) -> pd.DataFrame | dict[str, ProviderForecast]:
        """
        Generates forecasts.

        If org_code is given, returns a single ProviderForecast.
        Otherwise returns dict {org_code: ProviderForecast} for all providers.

        Parameters:
            horizon_days   : days to forecast (max 30)
            start_date     : start date (defaults to tomorrow)
            latest_wait12h : latest tracked-total A&E Wait 12h+ for pressure adj
            org_code       : specific provider or None for all
        """
        self._check_fitted()
        horizon_days = min(horizon_days, 30)

        if start_date is None:
            start_date = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

        if latest_wait12h is None and self._wait12h_history is not None:
            latest_wait12h = float(self._wait12h_history.iloc[-1])

        pressure_adj = self._compute_pressure_adj(latest_wait12h)

        if org_code:
            codes = [org_code]
        else:
            codes = list(self._provider_baselines.keys())

        results: dict[str, ProviderForecast] = {}
        for code in codes:
            baseline = self._provider_baselines.get(code, 0.0)
            rows = []
            for i in range(horizon_days):
                date     = start_date + pd.Timedelta(days=i)
                dow      = date.dayofweek
                month    = date.month
                dow_c    = self._dow_coefs.get(dow, 1.0)
                sea_c    = self._seasonal_coefs.get(month, 1.0)
                point    = baseline * dow_c * sea_c * (1 + pressure_adj)
                margin   = 1.282 * self._residual_std * dow_c * sea_c
                rows.append({
                    "date":         date,
                    "dow_name":     DOW_NAMES[dow],
                    "is_weekend":   dow >= 5,
                    "baseline":     round(baseline, 1),
                    "dow_coef":     round(dow_c, 3),
                    "seasonal_coef": round(sea_c, 3),
                    "pressure_adj": round(pressure_adj, 3),
                    "forecast":     round(point, 1),
                    "lower":        round(max(0, point - margin), 1),
                    "upper":        round(point + margin, 1),
                    "load":         self._interpret_load(point, baseline),
                })
            results[code] = ProviderForecast(
                org_code      = code,
                display_name  = ECL_DISPLAY_NAMES.get(code, code),
                baseline_daily = round(baseline, 1),
                forecast_df   = pd.DataFrame(rows),
            )

        if org_code:
            return results[org_code]
        return results

    def predict_df(
        self,
        horizon_days:   int = 14,
        start_date:     Optional[pd.Timestamp] = None,
        latest_wait12h: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Convenience method: returns all provider forecasts as a
        single flat DataFrame with org_code column.
        """
        all_fc = self.predict(horizon_days, start_date, latest_wait12h)
        frames = []
        for code, pf in all_fc.items():
            df = pf.forecast_df.copy()
            df.insert(0, "org_code",     code)
            df.insert(1, "display_name", pf.display_name)
            frames.append(df)
        return pd.concat(frames, ignore_index=True)

    def get_summary(self) -> ModelSummary:
        """Returns the fitted model summary."""
        self._check_fitted()
        return self._summary

    def get_monthly_ts(self) -> pd.DataFrame:
        """Returns the monthly time series used for trend charts."""
        self._check_fitted()
        return self._ts.copy()

    def get_provider_ts(self, org_code: str) -> pd.DataFrame:
        """Returns the monthly time series for a single provider."""
        self._check_fitted()
        return self._ts[self._ts["Org Code"] == org_code].copy()

    def get_pressure_status(self, wait12h: Optional[float] = None) -> dict:
        """
        Returns current system pressure based on A&E Wait 12h+ (tracked total).
        """
        self._check_fitted()
        if wait12h is None and self._wait12h_history is not None:
            wait12h = float(self._wait12h_history.iloc[-1])
        if wait12h is None:
            return {"level": "unknown", "value": None,
                    "percentile": None, "message": "A&E data unavailable"}

        hist = self._wait12h_history.dropna()
        pct  = float(stats.percentileofscore(hist, wait12h))

        if pct < 25:
            level, msg = "low", f"Low pressure — Wait 12h+ at {pct:.0f}th percentile"
        elif pct < 60:
            level, msg = "normal", f"Normal pressure — Wait 12h+ at {pct:.0f}th percentile"
        elif pct < 85:
            level, msg = "elevated", f"Elevated pressure — Wait 12h+ at {pct:.0f}th percentile"
        else:
            level, msg = "high", f"High pressure — Wait 12h+ at {pct:.0f}th percentile (historically high)"

        return {
            "level":      level,
            "value":      wait12h,
            "percentile": round(pct, 1),
            "message":    msg,
        }

    # ── PRIVATE: BUILD SERIES ─────────────────────────────────────────────────

    def _build_monthly_ts(
        self,
        df_s: pd.DataFrame,
        df_a: pd.DataFrame,
        codes: list[str],
    ) -> pd.DataFrame:
        """Builds the monthly provider time series."""
        ecl = df_s[df_s["Org Code"].isin(codes)].copy()

        def grp(metric_key: str, agg: str) -> pd.DataFrame:
            m   = ECL_METRICS.get(metric_key) or REABLEMENT_METRICS.get(metric_key, metric_key)
            sub = ecl[ecl["Metric"] == m]
            return (sub.groupby(["period", "Org Code"])["Value"]
                       .agg(agg).reset_index()
                       .rename(columns={"Value": metric_key}))

        p1   = grp("p1_home",        "sum")
        nctr = grp("nctr",           "mean")
        rem  = grp("nctr_remaining", "mean")
        dis  = grp("discharged",     "sum")
        cap  = grp("cap_gap_p1",     "first")
        ifc  = grp("interface_p1",   "first")

        ae_ecl = df_a[df_a["Org Code"].isin(codes)].copy()
        ae_m   = (ae_ecl.groupby(["period", "Org Code"])
                  [["emerg_admissions_t1", "wait_12h_plus"]]
                  .sum().reset_index())

        ts = (p1.merge(nctr, on=["period", "Org Code"], how="outer")
                .merge(rem,  on=["period", "Org Code"], how="outer")
                .merge(dis,  on=["period", "Org Code"], how="outer")
                .merge(cap,  on=["period", "Org Code"], how="outer")
                .merge(ifc,  on=["period", "Org Code"], how="outer")
                .merge(ae_m, on=["period", "Org Code"], how="left"))

        ts["p1_p2_total"]  = ts["p1_home"].fillna(0)   # P1-only now
        ts["period_str"]   = ts["period"].astype(str)
        ts["is_anomaly"]   = ts.apply(
            lambda r: (r["Org Code"], r["period_str"]) in KNOWN_ANOMALIES, axis=1
        )
        ts["display_name"] = ts["Org Code"].map(ECL_DISPLAY_NAMES)
        ts["days_in_month"] = ts["period"].dt.days_in_month
        ts["p1_daily"]     = ts["p1_home"] / ts["days_in_month"]

        return ts.sort_values(["Org Code", "period"]).reset_index(drop=True)

    def _build_daily_nctr(
        self,
        df_s:  pd.DataFrame,
        codes: list[str],
    ) -> pd.DataFrame:
        """Builds the daily NCTR series (all monitored providers aggregated)."""
        mask = (
            df_s["Org Code"].isin(codes) &
            (df_s["Metric"] == ECL_METRICS["nctr"])
        )
        daily = (df_s[mask]
                 .groupby("Period")["Value"]
                 .sum().reset_index()
                 .rename(columns={"Period": "date", "Value": "nctr_ecl"}))
        daily["dow"]      = daily["date"].dt.dayofweek
        daily["dow_name"] = daily["date"].apply(lambda d: DOW_NAMES[d.dayofweek])
        daily["month"]    = daily["date"].dt.month
        daily["period"]   = daily["date"].dt.to_period("M")
        return daily.sort_values("date").reset_index(drop=True)

    # ── PRIVATE: FIT COMPONENTS ───────────────────────────────────────────────

    def _fit_dow_pattern(self) -> None:
        """Computes day-of-week multipliers from daily NCTR data."""
        if self._daily is None or len(self._daily) == 0:
            self._dow_coefs = {i: 1.0 for i in range(7)}
            return
        overall = self._daily["nctr_ecl"].mean()
        if overall == 0:
            self._dow_coefs = {i: 1.0 for i in range(7)}
            return
        dow_avg = self._daily.groupby("dow")["nctr_ecl"].mean()
        self._dow_coefs = {int(d): float(v / overall) for d, v in dow_avg.items()}
        for i in range(7):
            if i not in self._dow_coefs:
                self._dow_coefs[i] = 1.0

    def _fit_seasonal_pattern(self) -> None:
        """Computes monthly seasonal multipliers from P1 data (tracked aggregate)."""
        if self._ts is None:
            self._seasonal_coefs = {i: 1.0 for i in range(1, 13)}
            return
        clean = self._ts[~self._ts["is_anomaly"]].copy()
        agg   = clean.groupby("period")["p1_home"].sum().reset_index()
        agg["month"] = agg["period"].dt.month
        overall = agg["p1_home"].mean()
        if overall == 0:
            self._seasonal_coefs = {i: 1.0 for i in range(1, 13)}
            return
        month_avg = agg.groupby("month")["p1_home"].mean()
        self._seasonal_coefs = {int(m): float(v / overall) for m, v in month_avg.items()}
        for i in range(1, 13):
            if i not in self._seasonal_coefs:
                self._seasonal_coefs[i] = 1.0

    def _fit_pressure_signal(self) -> None:
        """
        Calibrates A&E Wait 12h+ (tracked total) -> tracked total P1 regression.
        Calibration result: r=0.725, p=0.003 (14 months of data).
        """
        if self._ts is None:
            return
        clean = self._ts[~self._ts["is_anomaly"]].copy()
        agg   = (clean.groupby("period")
                 .agg(p1_home=("p1_home", "sum"),
                      wait12h=("wait_12h_plus", "sum"))
                 .reset_index()
                 .dropna(subset=["wait12h"]))

        self._wait12h_history = agg["wait12h"]

        if len(agg) < 4:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            slope, intercept, r, p, _ = stats.linregress(agg["wait12h"], agg["p1_home"])

        self._pressure_r         = float(r)
        self._pressure_p         = float(p)
        self._pressure_slope     = float(slope)
        self._pressure_intercept = float(intercept)

    def _fit_provider_baselines(self) -> None:
        """
        Computes per-provider daily P1 baseline.
        RF4 uses only Dec 2025 onward (post step-change).
        """
        if self._ts is None:
            return
        for code in (self._ts["Org Code"].unique()):
            sub = self._ts[
                (self._ts["Org Code"] == code) &
                (~self._ts["is_anomaly"])
            ].copy()
            # RF4 step-change: only use data from Dec 2025
            if code == "RF4":
                sub = sub[sub["period"] >= RF4_BASELINE_FROM]
            if len(sub) == 0:
                self._provider_baselines[code] = 0.0
                continue
            self._provider_baselines[code] = float(sub["p1_daily"].mean())

    def _fit_residual_std(self) -> None:
        """Estimates forecast uncertainty from residuals."""
        if self._daily is None or len(self._daily) == 0:
            self._residual_std = 5.0
            return
        # Predict daily NCTR using dow + seasonal, compare to actual
        predicted = self._daily.apply(
            lambda r: (self._daily["nctr_ecl"].mean() *
                       self._dow_coefs.get(int(r["dow"]), 1.0) *
                       self._seasonal_coefs.get(int(r["month"]), 1.0)),
            axis=1,
        )
        residuals = self._daily["nctr_ecl"] - predicted
        # Scale to P1 units (tracked total NCTR -> tracked total P1 ratio)
        nctr_mean = self._daily["nctr_ecl"].mean()
        total_p1  = sum(self._provider_baselines.values()) * 30  # monthly
        ratio     = (total_p1 / 30) / nctr_mean if nctr_mean > 0 else 0.1
        self._residual_std = float(residuals.std() * ratio)

    # ── PRIVATE: SUMMARY ─────────────────────────────────────────────────────

    def _build_summary(self, codes: list[str]) -> None:
        """Assembles ModelSummary after all fit steps."""
        clean = self._ts[~self._ts["is_anomaly"]] if self._ts is not None else pd.DataFrame()
        n_months = clean["period"].nunique() if len(clean) > 0 else 0

        date_range = ""
        if len(clean) > 0:
            ps = sorted(clean["period"].unique())
            date_range = f"{ps[0].strftime('%b %Y')} - {ps[-1].strftime('%b %Y')}"

        ecl_avg_p1 = float(
            clean.groupby("period")["p1_home"].sum().mean()
        ) if len(clean) > 0 else 0.0

        # Last A&E period and wait12h
        last_ae_period = None
        last_wait12h   = None
        if self._ts is not None:
            ae_data = self._ts[self._ts["wait_12h_plus"].notna()]
            if len(ae_data) > 0:
                lp = ae_data["period"].max()
                last_ae_period = str(lp)
                last_wait12h   = float(
                    self._ts[self._ts["period"] == lp]["wait_12h_plus"].sum()
                )

        # Pressure status (computed directly — _fitted not yet True)
        current_pressure = "unknown"
        if last_wait12h is not None and self._wait12h_history is not None:
            hist = self._wait12h_history.dropna()
            if len(hist) > 0:
                pct = float(stats.percentileofscore(hist, last_wait12h))
                if pct < 25:   current_pressure = "low"
                elif pct < 60: current_pressure = "normal"
                elif pct < 85: current_pressure = "elevated"
                else:          current_pressure = "high"

        # Named seasonal / dow dicts
        mo = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
              7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        seasonal_named = {mo.get(m, str(m)): round(v, 3)
                          for m, v in self._seasonal_coefs.items()}
        dow_named = {DOW_NAMES[i]: round(self._dow_coefs.get(i, 1.0), 3)
                     for i in range(7)}

        notes = []
        if self._pressure_p < 0.05:
            notes.append(
                f"A&E Wait 12h+ is a strong predictor of P1 demand "
                f"(r={self._pressure_r:.3f}, p={self._pressure_p:.3f})"
            )
        elif self._pressure_p < 0.10:
            notes.append(
                f"A&E Wait 12h+ is marginally significant "
                f"(r={self._pressure_r:.3f}, p={self._pressure_p:.3f})"
            )
        else:
            notes.append(
                f"A&E Wait 12h+ signal is weak "
                f"(r={self._pressure_r:.3f}, p={self._pressure_p:.3f}) — use with caution"
            )
        if n_months < 24:
            notes.append(
                f"Only {n_months} months of training data — "
                "seasonal coefficients will improve as more data accumulates"
            )
        notes.append(
            "RF4 baseline uses Dec 2025 onward only (structural step-change detected)"
        )
        notes.append(
            f"Known anomaly excluded: RDE 2026-01 "
            "(P1=702 vs normal ~240)"
        )

        self._summary = ModelSummary(
            n_months            = n_months,
            date_range          = date_range,
            ecl_monthly_avg_p1  = round(ecl_avg_p1, 0),
            provider_baselines  = {c: round(v, 2)
                                   for c, v in self._provider_baselines.items()},
            dow_coefficients    = dow_named,
            seasonal_index      = seasonal_named,
            pressure_r          = round(self._pressure_r, 3),
            pressure_p          = round(self._pressure_p, 3),
            pressure_slope      = round(self._pressure_slope, 4),
            pressure_intercept  = round(self._pressure_intercept, 1),
            anomalies_excluded  = KNOWN_ANOMALIES,
            model_notes         = notes,
            last_ae_period      = last_ae_period,
            last_wait12h        = last_wait12h,
            current_pressure    = current_pressure,
        )

    # ── HELPERS ───────────────────────────────────────────────────────────────

    def _compute_pressure_adj(self, wait12h: Optional[float]) -> float:
        """Converts Wait 12h+ into a forecast adjustment factor [-0.15, +0.20]."""
        if wait12h is None or self._wait12h_history is None:
            return 0.0
        hist = self._wait12h_history.dropna()
        if len(hist) == 0:
            return 0.0
        hist_mean = float(hist.mean())
        if hist_mean == 0:
            return 0.0
        relative = (wait12h - hist_mean) / hist_mean
        return float(np.clip(relative * 0.25, -0.15, 0.20))

    def _interpret_load(self, forecast_val: float, baseline: float) -> str:
        """Returns a plain-English load label."""
        if baseline == 0:
            return "no data"
        ratio = forecast_val / baseline
        if ratio < 0.85:   return "low"
        elif ratio < 0.95: return "below average"
        elif ratio < 1.10: return "normal"
        elif ratio < 1.25: return "above average"
        else:              return "high"

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit(df_sitrep, df_ae) first.")


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_forecast_model(
    df_sitrep: pd.DataFrame,
    df_ae:     pd.DataFrame,
    provider_codes: Optional[list[str]] = None,
) -> ReablementForecastModel:
    """Creates and fits the model in one call."""
    model = ReablementForecastModel()
    model.fit(df_sitrep, df_ae, provider_codes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sitrep_dir = sys.argv[1] if len(sys.argv) > 1 else "data/sitrep"
    ae_dir     = sys.argv[2] if len(sys.argv) > 2 else "data/ae"

    try:
        from data_loader import load_all_data
    except ImportError:
        print("ERROR: data_loader.py not found in path.")
        sys.exit(1)

    df_s, df_a, rpt = load_all_data(sitrep_dir, ae_dir)
    print(f"Data: {rpt.summary} | {rpt.sitrep_date_range}\n")

    model = build_forecast_model(df_s, df_a)
    sm    = model.get_summary()

    print(f"Training range  : {sm.date_range}")
    print(f"Tracked monthly P1  : {sm.ecl_monthly_avg_p1:.0f} avg/month")
    print(f"Pressure signal : r={sm.pressure_r:.3f}, p={sm.pressure_p:.3f}")
    print(f"Current pressure: {sm.current_pressure.upper()}")
    if sm.last_wait12h:
        print(f"Last Wait 12h+  : {sm.last_wait12h:,.0f} ({sm.last_ae_period})")

    print("\nProvider daily P1 baselines:")
    for code, baseline in sm.provider_baselines.items():
        name = ECL_DISPLAY_NAMES.get(code, code)
        print(f"  {code}  {name:<50}  {baseline:.1f} P1/day")

    print("\nDay-of-week coefficients:")
    for day, coef in sm.dow_coefficients.items():
        bar = "█" * int(coef * 15)
        print(f"  {day:<10} {coef:.3f}  {bar}")

    print("\nSeasonal index:")
    for month, idx in sm.seasonal_index.items():
        arrow = "▲" if idx > 1.05 else ("▼" if idx < 0.95 else " ")
        print(f"  {month:<4} {idx:.3f} {arrow}")

    print("\nModel notes:")
    for n in sm.model_notes:
        print(f"  • {n}")

    print("\n14-day forecast (all providers):")
    fc_all = model.predict_df(horizon_days=7)
    for code in ECL_PROVIDER_CODES:
        fc = fc_all[fc_all["org_code"] == code]
        vals = fc[["date","dow_name","forecast","load"]].copy()
        vals["date"] = vals["date"].dt.strftime("%d %b %a")
        print(f"\n  [{code}] {ECL_DISPLAY_NAMES.get(code,'')}:")
        print(vals.to_string(index=False))

    print("\nSelf-test complete.")
