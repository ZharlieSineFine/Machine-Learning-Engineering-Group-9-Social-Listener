# Architecture — Brand Sentiment Analysis Platform

> **One-line:** An end-to-end MLOps pipeline that ingests product reviews, classifies sentiment with a fine-tuned transformer, and serves predictions through a REST API and a dashboard — all reproducible with `docker compose up`.

---

## 1. Goal

Build a production-style sentiment analysis system for an online brand. The system should be:

- **End-to-end** — ingestion → training → registry → serving → monitoring → UI
- **Locally deployable** — every component runs in Docker on a developer laptop
- **Cloud-portable (stretch)** — swap Postgres for RDS, MinIO for S3 and the same stack lifts to AWS/GCP/Azure
- **Reproducible** — pinned dependencies, CI runs the same image set as local dev
- **Thin-slice first** — get a working skeleton end-to-end before adding depth in any one layer

---

## 2. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Orchestration | **Airflow** | Schedules ingestion, training, evaluation, drift checks |
| Storage (metadata + features) | **Postgres** | Single source of truth for reviews, predictions, run metadata |
| Object storage | **MinIO** | S3-compatible local store for raw datasets, model artifacts |
| Modeling | **HuggingFace Transformers + scikit-learn** | DistilBERT fine-tuning; sklearn baseline + utilities |
| Topic modelling (stretch) | **BERTopic** | Theme extraction from negative reviews |
| Experiment tracking + registry | **MLflow** | Logs runs, metrics, params; promotes models to `Staging` / `Production` |
| Data validation | **Great Expectations** | Schema + value-range checks on ingested reviews |
| Drift + performance monitoring | **Evidently** | Generates data/target drift reports per DAG run |
| Serving | **FastAPI** | REST endpoint that pulls the `Production` model from MLflow registry |
| Dashboard | **Streamlit** | KPI tiles, sentiment timelines, word-cloud, model comparison |
| Tests | **pytest** | Unit + integration tests gating CI |
| CI/CD | **GitHub Actions** | Lint, test, build images, push to registry |
| Containerisation | **Docker / Docker Compose** | One-command spin-up of the entire stack |

---

## 3. High-Level Data Flow

```
                ┌─────────────────────┐
                │   Product reviews   │   (Amazon / Yelp / app store dumps)
                └──────────┬──────────┘
                           │
                  ┌────────▼────────┐
                  │  Airflow: DAG 1 │   ingest_reviews.py
                  │   Ingestion     │── writes raw → MinIO
                  └────────┬────────┘── writes parsed → Postgres
                           │
                  ┌────────▼────────┐
                  │ Great            │  schema + null + range checks
                  │ Expectations     │  fails fast if data is malformed
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Airflow: DAG 2 │   train_model.py
                  │   Training      │── fine-tune DistilBERT on labelled reviews
                  └────────┬────────┘── log metrics/params/artifact → MLflow
                           │
                  ┌────────▼────────┐
                  │  MLflow Model   │   registers to "sentiment-distilbert"
                  │   Registry      │   versions: None → Staging → Production
                  └────────┬────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
      ┌───────▼────────┐        ┌───────▼────────┐
      │   FastAPI      │        │  Airflow: DAG 3│
      │   /predict     │        │  Evaluation +  │
      │                │        │  Drift (Evidently)
      │  loads "Prod"  │        │                │
      │   model        │        │  blocks promote│
      └───────┬────────┘        │  if drift > τ  │
              │                 └────────────────┘
      ┌───────▼────────┐
      │   Streamlit    │  KPI tiles, sentiment over time,
      │   Dashboard    │  word-cloud, model A/B compare
      └────────────────┘
```

---

## 4. Component Map

Each box in image 1 maps to a folder in the repo. See [WORKFLOW.md](./WORKFLOW.md) for ownership.

```
mle_project/
├── airflow/                  # DAGs + Airflow config         (Anh, Charlie, Ha)
│   ├── dags/
│   │   ├── ingest_reviews.py
│   │   ├── train_model.py
│   │   └── evaluate_and_monitor.py
│   └── plugins/
├── api/                      # FastAPI service               (Amelia)
│   ├── app/
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── model_loader.py
│   └── Dockerfile
├── dashboard/                # Streamlit app                 (Amelia)
│   ├── app.py
│   └── Dockerfile
├── models/                   # Training code + experiments   (Van, Amelia)
│   ├── train.py
│   ├── evaluate.py
│   ├── baseline_sklearn.py
│   └── distilbert_finetune.py
├── data/                     # Ingestion + validation        (Charlie, Ha)
│   ├── ingest/
│   ├── expectations/         # Great Expectations suites
│   └── schemas/
├── monitoring/               # Evidently configs + reports   (Charlie, Ha)
│   └── drift_checks.py
├── infra/                    # Docker, compose, CI           (Anh)
│   ├── docker/
│   └── github-actions/
├── tests/                    # pytest suites                 (all)
├── notebooks/                # Exploration (not in CI)       (Van, Amelia)
├── docker-compose.yml
├── ARCHITECTURE.md           # this file
├── WORKFLOW.md               # roles, phases, handoffs
└── README.md
```

---

## 5. Services in `docker-compose`

| Service | Image (base) | Port | Depends on |
|---|---|---|---|
| `postgres` | `postgres:15` | 5432 | — |
| `minio` | `minio/minio` | 9000 / 9001 | — |
| `mlflow` | custom (mlflow + psycopg2 + boto3) | 5000 | postgres, minio |
| `airflow-webserver` | custom (apache/airflow + project deps) | 8080 | postgres |
| `airflow-scheduler` | same image | — | postgres |
| `api` | custom (FastAPI) | 8000 | mlflow |
| `dashboard` | custom (Streamlit) | 8501 | api, postgres |
| `evidently` | runs inside Airflow DAGs (no standalone service) | — | — |

A single `docker compose up` brings the whole stack online.

---

## 6. Model Lifecycle

1. **Train** — Airflow `train_model` DAG kicks off `models/distilbert_finetune.py`, logs run to MLflow.
2. **Register** — Best run (per held-out F1) is registered to MLflow Model Registry as `sentiment-distilbert`, stage `None`.
3. **Validate** — `evaluate_and_monitor` DAG runs Evidently against a hold-out slice; if drift and F1 thresholds pass, the model is promoted to `Staging`.
4. **Promote** — Manual approval (or CI gate) flips `Staging` → `Production`.
5. **Serve** — FastAPI reads `models:/sentiment-distilbert/Production` on startup. A `/reload` admin endpoint can re-pull without restart.
6. **Monitor** — Daily DAG produces Evidently HTML reports stored in MinIO; the dashboard embeds the latest.

---

## 7. Environments

| Env | Where | Used for |
|---|---|---|
| Local dev | Docker Compose on laptop | Day-to-day development |
| CI | GitHub Actions (same images) | Lint, unit + integration tests, build, push |
| Production (stretch) | Any cloud — swap Postgres → managed RDS, MinIO → S3 | Live demo |

Every dependency is pinned (`requirements.txt` + `Dockerfile` base tags). CI uses the same compose stack as local, so "works on my machine" is the same as "works in CI."

---

## 8. Design Principles

- **Thin-slice first.** A trivial sklearn baseline served via FastAPI and shown on the dashboard is more valuable on day 7 than a perfectly-tuned DistilBERT that nobody can call. Depth comes after the skeleton is green end-to-end.
- **Contracts between layers are explicit.** API request/response schemas live in `api/app/schemas.py` and are shared with the dashboard. DB schemas live in `data/schemas/`. Don't pass implicit dicts across team boundaries.
- **Everything observable.** Every model run goes to MLflow; every DAG run leaves a trace; the dashboard surfaces what's live in prod right now.
- **No secret hand-offs.** If two teammates need to share a value (model name, table name, bucket), it goes in `infra/.env.example`, not in chat.

---

## 9. Open Questions (resolve in week 1)

- Which review dataset are we starting with — Amazon Reviews 2023, Yelp Open Dataset, or a smaller curated sample?
- Sentiment labels — binary (pos/neg), ternary (pos/neu/neg), or 5-star?
- Do we need multilingual support, or English-only for v1?
- BERTopic on negative reviews is a stretch goal — confirm in/out of scope before sprint 2.
