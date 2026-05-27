# CareSignal — Reablement Intelligence

An operational analytics dashboard built on NHS England open data
(SitRep Discharge + A&E Attendances). It tracks discharge pressure,
capacity gaps and forecasts Pathway 1 reablement demand across a
configurable set of acute NHS trusts.

> Non-commercial project. Built for learning and demonstration on
> public NHS England data.

---

## Features

- **Operations** — provider cards with last-3-month metrics
- **Demand Pipeline** — NCTR funnel per trust
- **Capacity Gap** — P1 capacity-gap trend with A&E overlay
- **Benchmark** — each trust vs its NHS region and National
- **Forecast** — 14-day P1 forecast (baseline × DOW × seasonal × A&E pressure)
- **Ask the Data** — GPT-4o-mini chat grounded in a pre-computed data summary
- **Data Quality** — per-file ingestion report

## Stack

Python 3.11 · Streamlit · pandas · numpy · pyarrow · plotly · scipy · OpenAI API

---

## Quick start (local)

```bash
git clone https://github.com/<your-account>/caresignal.git
cd caresignal

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Optional: enable the AI chat
echo "OPENAI_API_KEY=sk-..." > .env

streamlit run app.py
```

The repository ships with prepared `data/processed/*.parquet` files so the
app runs immediately. Raw NHS CSV files are not required for runtime.

---

## Updating the data

Raw NHS CSV files are intentionally **not** committed to keep the repo
lightweight and the cloud runtime fast. To refresh the dashboard:

1. Download the latest monthly files from NHS England:
   - **SitRep Discharge** → save as `data/sitrep/sitrep_YYYY-MM.csv`
   - **A&E Attendances** → save as `data/ae/ae_YYYY-MM.csv`
2. Run the ETL:
   ```bash
   python build_processed_data.py
   ```
   This regenerates `data/processed/sitrep_clean.parquet` and
   `data/processed/ae_clean.parquet`.
3. Verify locally: `streamlit run app.py`
4. Commit and push the updated parquet files. Streamlit Cloud
   auto-redeploys on push.

---

## Deployment (Streamlit Community Cloud)

1. Push this repository to GitHub (public).
2. Go to <https://share.streamlit.io> → **New app**.
3. Select the repo, branch `main`, main file `app.py`.
4. **Advanced settings → Python version: 3.11**.
5. **Secrets**: paste your OpenAI key in TOML format:
   ```toml
   OPENAI_API_KEY = "sk-..."
   ```
6. Deploy. First build takes ~3–5 minutes.

The AI chat tab will show a clear "API key not configured" message if
`OPENAI_API_KEY` is missing — the rest of the dashboard works without it.

---

## Project layout

```
caresignal/
├── app.py                    Streamlit entry point (pages, layout, charts)
├── data_loader.py            CSV/parquet ingest, cleaning, quality rules
├── lag_model.py              P1 demand forecast model
├── chat_engine.py            OpenAI-backed chat with pre-computed data context
├── build_processed_data.py   Local ETL: raw NHS CSV → parquet
├── data/
│   ├── sitrep/   (gitignored, local raw files)
│   ├── ae/       (gitignored, local raw files)
│   └── processed/
│       ├── sitrep_clean.parquet
│       └── ae_clean.parquet
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── requirements.txt
├── .python-version
└── .gitignore
```

## Default monitored providers

The dashboard is pre-configured for five acute NHS trusts in
East of England, London and South East. The list lives in
`data_loader.py` (`ECL_PROVIDER_CODES` constant) and is easy to swap:

| Code | NHS Trust                                | NHS Region        |
|------|------------------------------------------|-------------------|
| RAJ  | Mid & South Essex NHS Trust              | East of England   |
| RDE  | East Suffolk & NE Essex NHS Trust        | East of England   |
| RQW  | Princess Alexandra Hospital NHS Trust    | East of England   |
| RF4  | Barking, Havering & Redbridge UH         | London            |
| RYR  | University Hospitals Sussex NHS Trust    | South East        |

---

## Data sources

- [NHS England — Acute Discharge SitRep](https://www.england.nhs.uk/statistics/)
- [NHS England — A&E Attendances & Emergency Admissions](https://www.england.nhs.uk/statistics/)

All raw data is published openly by NHS England under the Open Government Licence.

## License

This project is provided for educational and non-commercial use.
