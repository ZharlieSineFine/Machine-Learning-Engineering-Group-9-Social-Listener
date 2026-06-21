# Dashboard — BrewLeaf Social Listener · Group 9

**Owner:** Amelia  
**Stack:** Streamlit · Plotly · Pandas · WordCloud  
**Entry point:** `dashboard/app.py`

---

## What this folder contains

```
dashboard/
├── app.py              # Marketing dashboard (main page, served at /)
├── data.py             # All data loading and transformation logic
├── pages/
│   └── mlops_monitor.py  # MLOps monitor dashboard (served at /mlops_monitor)
├── requirements.txt    # Python dependencies for this module
└── README.md           # This file
```

---

## The two dashboards

### 🍃 Marketing dashboard (`app.py`)

The brand-facing view. Intended for whoever manages BrewLeaf's social media — not a technical audience. Shows:

| Section | What it does |
|---|---|
| **Header** | BrewLeaf logo, last batch timestamp, spike alert badge (red if negative sentiment ≥ 20%) |
| **KPI tiles** | Total posts analysed, % negative / positive / neutral |
| **Sentiment trend** | Line chart of positive / neutral / negative % over time. Toggle between Daily (14 days) and Weekly (8 weeks) |
| **Alerts panel** | Most recent negative review snippets with source tag (Yelp / TripAdvisor) and date |
| **Word cloud** | Top words from negative reviews, rendered in amber. Falls back to a horizontal bar chart if `wordcloud` is not installed |

### 🔬 MLOps monitor (`pages/mlops_monitor.py`)

The data scientist view. Shows model health and pipeline signals. Streamlit picks this file up automatically as a second page because it lives under `pages/`.

| Section | What it does |
|---|---|
| **Header** | API health badge (green = `/health` responds, red = unreachable) |
| **Production model KPIs** | F1 macro, F1 negative, Recall negative for the current production run |
| **MLflow run history** | Table of all experiment runs with status badges (production / staging / archived). Falls back to hardcoded checkpoint metrics if MLflow is unreachable |
| **Shadow deploy** | Agreement rate between production and staging model predictions. Populated by calls to `POST /predict`. Shows a placeholder until Van promotes a Staging model in MLflow |
| **Drift panel** | Evidently drift score with a visual bar and gate pass/fail badge. Phase 1 runs train-vs-itself as a stub |

---

## Data flow

```
Airflow DAG (Charlie/Ha)
  └─ scrapes reviews → calls POST /predict → writes to Gold parquet / Postgres

Dashboard (reads on page load, cached 5 min)
  └─ load_reviews() in data.py tries sources in order:
       1. PostgreSQL (if POSTGRES_* env vars are set)
       2. Gold parquet files (data/gold/)
       3. CSV fallback → data/demo_data/demo_jun2026_stable.csv
```

The dashboard never calls `/predict` itself. It only reads results that the pipeline has already written. The demo CSV is the current fallback while the live pipeline is being completed by Charlie/Ha.

---

## `data.py` — what's inside

All data loading and transformation is kept out of `app.py` so it can be unit tested without Streamlit's runtime context.

| Function | What it does |
|---|---|
| `load_reviews(dsn, gold_root, csv_path, days)` | Main loader. Tries Postgres → Gold parquet → CSV in order. Returns a DataFrame with `text`, `label`, `review_date` columns |
| `_load_gold_parquet(gold_root, days)` | Reads partitioned parquet files from `gold/feature_store/` and `gold/label_store/`, merges on `review_id` |
| `sentiment_timeline(df, freq, time_col)` | Groups by time period (`D` or `W`) and computes % positive / negative / neutral per bucket |
| `negative_word_counts(df, top_n)` | Returns a `Counter` of the most frequent words in negative reviews, after stopword filtering |
| `list_mlflow_runs(...)` | Pulls experiment run history from MLflow. Returns empty DataFrame on failure |
| `latest_drift_report(dsn)` | Reads the most recent row from `monitoring_reports` in Postgres |
| `fetch_drift_html(report_url, minio_client)` | Downloads an Evidently HTML report from MinIO given an `s3://` URL |

---

## How to run locally

### 1. Install dependencies

```bash
pip install -r dashboard/requirements.txt
```

### 2. Run the marketing dashboard

```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. The MLOps monitor is automatically available at `http://localhost:8501/mlops_monitor`.

### 3. Environment variables (optional)

Without these, the dashboard falls back to the demo CSV automatically — you do not need to set any of these to get it running.

| Variable | Default | Purpose |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI service base URL |
| `GOLD_ROOT` | `data/gold/` | Path to Gold parquet files |
| `DEMO_DATA_ROOT` | `data/demo_data/` | Path to demo CSV files |
| `POSTGRES_HOST` | *(unset)* | Enable live Postgres loading |
| `POSTGRES_USER` | *(unset)* | Postgres credentials |
| `POSTGRES_PASSWORD` | *(unset)* | Postgres credentials |
| `POSTGRES_DB` | *(unset)* | Postgres database name |
| `MLFLOW_TRACKING_URI` | *(unset)* | MLflow server URL |
| `DRIFT_THRESHOLD` | `0.3` | Drift score above which the gate fails |

### 4. Run via Docker Compose (full stack)

```bash
docker compose up -d
```

The dashboard container mounts the repo and serves at port `8501`. The `GOLD_ROOT` and `DEMO_DATA_ROOT` env vars are set in `docker-compose.yml` to point to `/repo_data/` inside the container.

---

## What is and isn't live yet

| Feature | Status |
|---|---|
| Marketing dashboard with demo data | ✅ Working |
| MLOps monitor with checkpoint fallback | ✅ Working |
| Shadow deploy tile | ✅ Working — needs `/predict` calls to populate |
| Live Gold parquet loading | ⏳ Waiting on Charlie/Ha's Airflow pipeline |
| Drift report with real reference dataset | ⏳ Waiting on Charlie/Ha's Evidently Phase 2 |
| MLflow run history (live) | ⏳ Waiting on Van to register and promote models |
| Postgres live loading | ⏳ Waiting on Charlie/Ha's predictions table schema |

---

## Notes for teammates

- **Van** — once you register a DistilBERT model in MLflow and promote one to Staging, the MLOps monitor run history table and shadow deploy tile will populate automatically. No dashboard changes needed.
- **Charlie/Ha** — once Gold parquets land at `data/gold/feature_store/` and `data/gold/label_store/` (partitioned by `review_date=YYYY-MM-DD/part.parquet`), the marketing dashboard will switch off the demo CSV automatically. The swap happens inside `load_reviews()` in `data.py`.
- The demo CSV (`demo_jun2026_stable.csv`) is only used as a last resort when neither Postgres nor Gold parquets are available. It is not shown to users — it is a development fallback only.