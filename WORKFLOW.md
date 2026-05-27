# Team Workflow — Brand Sentiment Analysis

This document is the single source of truth for **who does what, in what order, and what "done" looks like** for each phase of the project. Pair it with [ARCHITECTURE.md](./ARCHITECTURE.md) for the technical picture.

---

## 1. Team Roles & Ownership

| Role | Owner(s) | Primary tools | Owns these folders |
|---|---|---|---|
| **The Glue — DevOps / CI/CD / Orchestration** | Anh | Airflow infra, Docker, MinIO, Postgres, GitHub Actions | `infra/`, `airflow/` (infra side), `docker-compose.yml`, `.github/` |
| **The Modeler — ML & Experimentation** | Van *(+ Amelia as second pair)* | DistilBERT, HuggingFace, BERTopic, MLflow | `models/`, MLflow experiment design |
| **Data & Evaluation** | Charlie + Ha | Airflow DAGs (data + eval), Great Expectations, Evidently, MLflow tracking | `airflow/dags/`, `data/`, `monitoring/` |
| **Serving, Monitoring UI & QA** | Amelia | FastAPI, MLflow Registry, Streamlit, pytest | `api/`, `dashboard/`, `tests/`, demo script, README |

### RACI cheat-sheet

| Workstream | R (does the work) | A (approves PR) | C (consulted) | I (informed) |
|---|---|---|---|---|
| Docker / Compose / CI | Anh | Anh | All | All |
| Ingestion DAG + GE checks | Charlie, Ha | Anh | Van | All |
| Training code + MLflow logging | Van *(+ Amelia)* | Van | Charlie, Ha | All |
| Eval DAG + Evidently drift | Charlie, Ha | Charlie | Van | All |
| FastAPI + Registry pull | Amelia | Amelia | Anh, Van | All |
| Streamlit dashboard | Amelia | Amelia | Charlie | All |
| Pytest suite | Amelia (+ each owner for their folder) | Amelia | All | All |
| Architecture / workflow docs | Anh | Anh | All | All |

> **Rule of thumb:** the owner of a folder reviews every PR that touches it.

---

## 2. Build Strategy — Thin-Slice First

We build the **end-to-end skeleton with a trivial model first**, then add depth in each layer. This is non-negotiable: nobody works on DistilBERT fine-tuning until a sklearn baseline is being served, predicted from, and shown on the dashboard.

```
Phase 1 (MVP skeleton)  ──►  Phase 2 (depth)  ──►  Phase 3 (polish + demo)
   "everything works,         "real models,         "ready to present"
    nothing is fancy"          real drift,
                               real UI"
```

---

## 3. Phase 1 — Thin-Slice MVP (Week 1–2)

**Goal:** `docker compose up` brings a working end-to-end system online: ingestion → training (sklearn) → registry → API → dashboard, with one DAG running daily.

### Deliverables per role

**Anh (Glue) — must finish first; unblocks everyone else**
- `docker-compose.yml` with postgres, minio, mlflow, airflow, api, dashboard
- `infra/.env.example` with all shared env vars (DB URLs, MLflow URI, S3 keys)
- Bootstrap script to init Postgres schemas + MinIO buckets
- GitHub Actions: lint + pytest on PR

**Charlie + Ha (Data & Eval)**
- Sample product-review dataset committed (1–10k rows, CSV)
- `airflow/dags/ingest_reviews.py` — reads CSV → Postgres `reviews` table
- Minimal Great Expectations suite (schema + non-null on `text`, `label`)
- `airflow/dags/evaluate_and_monitor.py` — stub that runs Evidently on train vs. held-out

**Van (Modeler)**
- `models/baseline_sklearn.py` — TF-IDF + LogisticRegression
- `models/train.py` callable from Airflow DAG; logs to MLflow; registers as `sentiment-baseline`
- Promote v1 manually to `Production` in MLflow UI

**Amelia (Serving + UI)**
- `api/app/main.py` — FastAPI with `POST /predict` loading `models:/sentiment-baseline/Production`
- `dashboard/app.py` — 3 tiles: total reviews, % positive, latest prediction probe
- Pytest: 1 unit test per folder so CI has something to gate

### Phase 1 Definition of Done
- [ ] Fresh clone → `docker compose up` → all services healthy
- [ ] Trigger ingestion DAG → row count in Postgres goes up
- [ ] Trigger training DAG → run appears in MLflow, model registered
- [ ] `curl -X POST /predict` returns a label
- [ ] Dashboard loads at `localhost:8501` and shows real numbers
- [ ] CI passes on `main`

---

## 4. Phase 2 — Depth (Week 3–5)

**Goal:** Replace the toy parts with the real thing. The skeleton stays the same; each owner deepens their layer.

### Per role

**Van + Amelia (Modeling)**
- `models/distilbert_finetune.py` — HuggingFace Trainer, GPU-optional
- Compare baseline vs. DistilBERT in MLflow; pick winner
- Register as `sentiment-distilbert`, promote via approval gate
- *(Stretch)* BERTopic on negative reviews → top-5 themes per week

**Charlie + Ha (Data & Eval)**
- Real review source wired in (Amazon Reviews 2023 sample, Yelp, or chosen alternative)
- Full Great Expectations suite: length bounds, language detect, label cardinality
- Evidently report uploaded to MinIO per DAG run; URL stored in Postgres
- **Model validation gate**: DAG fails (and blocks promotion) if F1 drops > 3% or drift score > τ

**Anh (Glue)**
- CI matrix: lint, unit, integration (boots compose, hits `/predict`)
- Push images to GitHub Container Registry on tag
- Secrets management via `.env` + GitHub Actions secrets
- *(Stretch)* Terraform stub for AWS lift (RDS + S3 + ECS)

**Amelia (Serving + UI)**
- `/predict` supports batch input
- `/reload` admin endpoint to refresh model without restart
- Streamlit: sentiment timeline (line chart), word cloud of negative reviews, model A/B comparison view
- End-to-end pytest hitting compose stack

### Phase 2 Definition of Done
- [ ] DistilBERT beats baseline on held-out F1 in MLflow
- [ ] Drift report runs daily and is visible from the dashboard
- [ ] A bad data day actually blocks promotion (test this with a poisoned batch)
- [ ] Integration test in CI: bring stack up, train, predict, assert

---

## 5. Phase 3 — Polish & Demo (Week 6)

**Goal:** Anyone can clone the repo and run the demo. The story is presentable.

- README walks a new dev from zero to `/predict` in under 10 minutes
- Demo script (`scripts/demo.sh`) seeds data, trains, predicts, opens the dashboard
- Architecture diagram (ASCII version is in this repo; add a polished SVG)
- Slide-ready 5-minute walkthrough — covers problem, stack, results, what we'd do next
- Postmortem doc: what surprised us, what we'd cut, what's worth doing if we had another month

---

## 6. Handoff Contracts

The points where teammates' work meets — these are the contracts you should agree on **before** writing code.

| Boundary | Owners | Contract (must be written down) |
|---|---|---|
| `reviews` Postgres table | Charlie/Ha → Van | Column names, types, label encoding |
| MLflow run logging | Van → Charlie/Ha | Metric names (`f1`, `precision_neg`, …), tag conventions |
| MLflow registered model name | Van → Amelia | Exactly `sentiment-distilbert`, alias `Production` |
| FastAPI request/response schema | Amelia → dashboard, downstream consumers | Pydantic models in `api/app/schemas.py` |
| Evidently report location | Charlie/Ha → Amelia | MinIO path + Postgres `monitoring_reports` table |
| Env vars | Anh → all | `infra/.env.example` is the canonical list |

When a contract changes, the owner opens a PR that updates the contract file *and* every consumer. No silent drift.

---

## 7. Ways of Working

- **PRs are small.** Aim for < 400 lines changed. If it grows, split.
- **One owner per folder.** Cross-folder PRs need both owners to approve.
- **Tests live next to code.** `models/test_train.py`, `api/app/test_main.py`, etc. The top-level `tests/` is reserved for cross-cutting integration tests.
- **Daily 15-min standup** during phase 1 (most coupling). Async after.
- **Block early.** If you're blocked > 4 hours waiting on a contract, flag in chat; don't build around it.
- **Reproducibility is a feature.** Any PR that introduces a new dependency must pin it in the relevant `requirements.txt` and update the Dockerfile.

---

## 8. Stretch Goals (only after Phase 2 DoD passes)

- BERTopic theme extraction on negative reviews → weekly "what users complain about" widget
- Multilingual sentiment (XLM-R or mDistilBERT)
- Active-learning loop: low-confidence predictions get queued for human review in the dashboard
- Terraform → ECS lift, demo on a tiny cloud instance
- Slack alert from Evidently when drift > τ

---

## 9. Quick Reference

| What | Where |
|---|---|
| Spin up locally | `docker compose up` from repo root |
| Airflow UI | `http://localhost:8080` |
| MLflow UI | `http://localhost:5000` |
| FastAPI docs | `http://localhost:8000/docs` |
| Streamlit dashboard | `http://localhost:8501` |
| MinIO console | `http://localhost:9001` |
| Postgres | `localhost:5432`, see `.env.example` for creds |
