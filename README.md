# Brand Sentiment Analysis — MLOps Platform

End-to-end sentiment analysis for online brand product reviews. Built locally with Docker Compose, cloud-portable when we're ready to scale.

> One command, one stack: `docker compose up`.

---

## Quick start

```bash
# 1. configure
cp infra/.env.example .env

# 2. boot the stack
docker compose up -d

# 3. open the UIs
   Airflow      http://localhost:8080
#    MLflow       http://localhost:5000
#    FastAPI docs http://localhost:8000/docs
#    Streamlit    http://localhost:8501
#    MinIO        http://localhost:9001
```

---

## Repo map

| Path | Owner | What lives here |
|---|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Anh | System architecture, data flow, design principles |
| [`WORKFLOW.md`](./WORKFLOW.md) | Anh | Team roles, RACI, phased milestones |
| [`airflow/`](./airflow/README.md) | Charlie + Ha (DAGs), Anh (infra) | Orchestration DAGs |
| [`api/`](./api/README.md) | Amelia | FastAPI serving |
| [`dashboard/`](./dashboard/README.md) | Amelia | Streamlit dashboard |
| [`models/`](./models/README.md) | Van (+ Amelia) | Training, MLflow integration |
| [`data/`](./data/README.md) | Charlie + Ha | Ingestion, schemas, Great Expectations |
| [`monitoring/`](./monitoring/README.md) | Charlie + Ha | Evidently drift checks |
| [`infra/`](./infra/README.md) | Anh | Docker, env, GitHub Actions |
| [`tests/`](./tests/README.md) | Amelia (lead, all contribute) | Integration tests |
| [`notebooks/`](./notebooks/README.md) | Van, Amelia | Exploration (not in CI) |
| [`scripts/`](./scripts/README.md) | Anh | Bootstrap, demo, reset helpers |

---

## Where to start

- **New here?** Read `ARCHITECTURE.md` (10 min) then `WORKFLOW.md` (10 min).
- **Picking up your folder?** Read its README, find your Phase 1 stub, open a PR.
- **Just want to demo it?** Wait for Phase 3 — `scripts/demo.sh` will do it all.

---

## Tech stack (one-liner)

Airflow · Postgres · MinIO · MLflow · HuggingFace + sklearn · Great Expectations · Evidently · FastAPI · Streamlit · Docker · GitHub Actions
