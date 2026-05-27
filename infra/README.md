# Infra — Docker, Compose, CI

**Owner:** Anh

This folder owns everything that makes the stack runnable and reproducible.

## What lives here

```
infra/
├── docker/             # Per-service Dockerfiles (api, dashboard, mlflow, airflow)
├── github-actions/     # Reusable composite actions (lint, build, test)
└── .env.example        # Every env var the stack needs (no secrets, just keys)
```

(`.github/workflows/` lives at repo root because GitHub requires it; the reusable actions it calls live here.)

## Env vars (see `.env.example`)

```
POSTGRES_USER=mlops
POSTGRES_PASSWORD=mlops
POSTGRES_DB=sentiment
POSTGRES_HOST=postgres

MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin

AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://mlops:mlops@postgres/airflow

MODEL_NAME=sentiment-baseline       # phase 1
# MODEL_NAME=sentiment-distilbert    # phase 2
MODEL_STAGE=Production
```

## CI matrix (GitHub Actions)

| Job | Trigger | What it does |
|---|---|---|
| `lint` | PR | ruff + black --check |
| `unit` | PR | pytest with `-m "not integration"` |
| `integration` | PR to main | `docker compose up -d`, hit `/health`, run smoke prediction |
| `build-push` | tag | Build images, push to GHCR |

## Phase 1 stub
Lint + unit only. Integration job lands in Phase 2 once the stack stabilises.
