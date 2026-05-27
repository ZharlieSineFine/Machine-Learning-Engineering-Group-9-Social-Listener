# Dashboard — Streamlit

**Owner:** Amelia

A single-page Streamlit app that surfaces what the system is doing right now.

## Sections (target)

1. **Overview tiles** — total reviews, % positive, drift status (green/red), last training date
2. **Sentiment over time** — line chart from Postgres `reviews`/`predictions`
3. **Word cloud** — most frequent terms in negative reviews this week
4. **Live probe** — text input → calls `POST /predict` → shows label + confidence
5. **Model comparison** (Phase 2) — baseline vs DistilBERT side-by-side from MLflow runs
6. **Drift report** (Phase 2) — embeds the latest Evidently HTML from MinIO

## Data sources

- Postgres (`reviews`, `predictions`, `monitoring_reports`)
- FastAPI (`/predict` for the live probe)
- MLflow REST API (for model metadata + experiment comparison)

## Phase 1 stub
Three tiles, hardcoded numbers if needed, just to validate that the container builds and serves at `:8501`.
