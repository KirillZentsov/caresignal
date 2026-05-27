"""
build_processed_data.py
=======================
Local ETL script for CareSignal.

Run this locally after adding new raw NHS files:
    python build_processed_data.py

It parses:
    data/sitrep/sitrep_YYYY-MM.csv
    data/ae/ae_YYYY-MM.csv

and writes small prepared parquet files for Streamlit Cloud:
    data/processed/sitrep_clean.parquet
    data/processed/ae_clean.parquet

Raw files should stay local and be excluded from GitHub if you want a lightweight repo.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from data_loader import load_all_data


ROOT = Path(__file__).parent
SITREP_DIR = ROOT / "data" / "sitrep"
AE_DIR = ROOT / "data" / "ae"
PROCESSED_DIR = ROOT / "data" / "processed"


def _period_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet is happier with string periods; app restores Period dtype on load."""
    df = df.copy()
    if "period" in df.columns:
        df["period"] = df["period"].astype(str)
    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Reading raw NHS files...")
    df_s, df_a, rpt = load_all_data(SITREP_DIR, AE_DIR, prefer_processed=False)

    if len(df_s) == 0:
        raise RuntimeError(f"No SitRep rows loaded from {SITREP_DIR}")
    if len(df_a) == 0:
        raise RuntimeError(f"No A&E rows loaded from {AE_DIR}")

    print(f"Quality: {rpt.summary}")
    print(f"SitRep : {rpt.sitrep_date_range} ({len(df_s):,} rows)")
    print(f"A&E    : {rpt.ae_date_range} ({len(df_a):,} rows)")
    if rpt.lag_note:
        print(f"Lag    : {rpt.lag_note}")

    # Store raw-cleaned app-ready dataframes. These are not raw NHS files;
    # they are already cleaned, typed, annotated and validated by data_loader.
    sitrep_out = PROCESSED_DIR / "sitrep_clean.parquet"
    ae_out = PROCESSED_DIR / "ae_clean.parquet"

    _period_to_str(df_s).to_parquet(sitrep_out, index=False)
    _period_to_str(df_a).to_parquet(ae_out, index=False)

    print("\nWritten:")
    print(f"  {sitrep_out} ({sitrep_out.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  {ae_out} ({ae_out.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
