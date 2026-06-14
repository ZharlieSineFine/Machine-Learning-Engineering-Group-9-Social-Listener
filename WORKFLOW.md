# Team Workflow — Brand Sentiment Analysis

Single source of truth for **who does what, in what order, and what "done" looks like**. Pair with [ARCHITECTURE.md](./ARCHITECTURE.md) for the technical picture.

---

## 1. Team Roles & Ownership

| Role | Owner(s) | Primary tools | Owns these folders |
|---|---|---|---|
| **The Glue — DevOps / CI/CD / Orchestration** | Anh | Airflow infra, Docker, MinIO, Postgres, GitHub Actions | `infra/`, `airflow/` (infra side), `docker-compose.yml`, `.github/` |
| **The Modeler — ML & Experimentation** | Van *(+ Amelia as second pair)* | DistilBERT, HuggingFace, BERTopic, MLflow | `models/` (incl. `embeddings.py` used by Gold), MLflow experiment design |
| **Data & Evaluation — Medallion + Drift** | Charlie + Ha | Airflow DAGs, Great Expectations, Evidently, MLflow tracking | `airflow/dags/`, `data/`, `monitoring/` |
| **Serving, Monitoring UI & QA** | Amelia | FastAPI (with shadow deploy), MLflow Registry, Streamlit, pytest | `api/`, `dashboard/`, `tests/`, demo script, README |

### RACI cheat-sheet

| Workstream | R (does the work) | A (approves PR) | C (consulted) | I (informed) |
|---|---|---|---|---|
| Docker / Compose / CI | Anh | Anh | All | All |
| Bronze ingestion + replay simulator | Charlie, Ha | Anh | Van | All |
| Silver refinement + GE gate | Charlie, Ha | Charlie | Van | All |
| Gold (embeddings + labels) | Charlie, Ha (joins with Van for `embeddings.py`) | Charlie | Van | All |
| Training + MLflow logging | Van *(+ Amelia)* | Van | Charlie, Ha | All |
| Evidently drift + retrain trigger | Charlie, Ha | Charlie | Van, Amelia | All |
| FastAPI + Registry pull + **shadow deploy** | Amelia | Amelia | Anh, Van | All |
| Streamlit dashboard | Amelia | Amelia | Charlie | All |
| Pytest suite (incl. smoke container) | Amelia (+ each owner for their folder) | Amelia | All | All |
| Architecture / workflow docs | Anh | Anh | All | All |

> **Rule of thumb:** the owner of a folder reviews every PR that touches it.

---

## 2. Build Strategy — Thin-Slice First

End-to-end skeleton with a trivial model first, then depth. Nobody works on DistilBERT until a sklearn baseline is being served (in production *and* shadow lanes) and shown on the dashboard.

```
Phase 1 (MVP skeleton)  ──►  Phase 2 (depth)  ──►  Phase 3 (polish + demo)
   "everything works,         "real medallion,      "ready to present"
    nothing is fancy"          real drift,
                               shadow deploy,
                               feedback loop"
```

---

## 3. Phase 1 — Thin-Slice MVP (Week 1–2)

**Goal:** `docker compose up` brings an end-to-end skeleton online: sample CSV → Bronze → Silver → Gold → sklearn training → registry → FastAPI → Streamlit. One 6-hour DAG cycle runs successfully.

### Deliverables per role

**Anh (Glue) — must finish first; unblocks everyone**
- `docker-compose.yml` with postgres, minio (+ init), mlflow, airflow, api, dashboard, smoke
- `infra/.env.example` with all shared env vars
- Bootstrap script: init Postgres schemas + MinIO buckets (`bronze`, `silver`, `gold`, `mlflow`, `monitoring`)
- GitHub Actions: lint + pytest + `docker compose run --rm smoke` on PR

**Charlie + Ha (Data & Eval)**
- `data/sample/reviews_sample.csv` committed (few hundred rows)
- `airflow/dags/ingest_bronze.py` — reads sample CSV → MinIO `bronze/`
- `airflow/dags/refine_silver.py` — Bronze → Postgres `reviews_silver`, minimal GE suite (schema + non-null)
- `airflow/dags/build_gold.py` — Silver → Postgres `reviews_gold` (label derived from `stars`, embedding stubbed as random vector)
- `airflow/dags/evaluate_and_monitor.py` — Evidently stub running train-vs-train (always passes) so the wiring exists
- Schemas committed in `data/schemas/` and matching DDL applied at bootstrap

**Van (Modeler)**
- `models/baseline_sklearn.py` — TF-IDF + LogisticRegression
- `models/train.py` callable from the `train_model` DAG; logs to MLflow; registers as `sentiment-baseline`
- Manually promote v1 to `Production` in MLflow UI
- `models/embeddings.py` stub returning a fixed-size random vector (real embeddings land in Phase 2)

**Amelia (Serving + UI)**
- `api/app/main.py` — FastAPI with `POST /predict` loading `models:/sentiment-baseline/Production`
- `api/app/shadow.py` — stub that logs Production predictions to `predictions` table (real shadow lane in Phase 2)
- `dashboard/app.py` — 3 tiles: total reviews, % positive, latest prediction probe
- Smoke container green: pytest covers happy-path `/predict`

### Phase 1 Definition of Done
- [ ] Fresh clone → `cp infra/.env.example .env` → `docker compose up -d` → all services healthy
- [ ] Trigger `ingest_bronze` → object lands in MinIO `bronze/`
- [ ] Trigger `refine_silver` → rows appear in `reviews_silver`
- [ ] Trigger `build_gold` → rows appear in `reviews_gold` with labels
- [ ] Trigger `train_model` → MLflow run appears, model registered
- [ ] `curl -X POST /predict` returns a label
- [ ] Dashboard shows real numbers at `localhost:8501`
- [ ] `docker compose run --rm smoke` exits 0
- [ ] CI passes on `main`

---

## 4. Phase 2 — Depth (Week 3–5)

**Goal:** Replace the stubs with the real thing. The skeleton stays the same.

### Per role

**Van + Amelia (Modeling)**
- `models/distilbert_finetune.py` — HuggingFace Trainer, GPU-optional
- `models/embeddings.py` real implementation (sentence-transformers MiniLM-L6-v2 by default)
- Compare baseline vs. DistilBERT in MLflow; pick winner
- Register as `sentiment-distilbert`; promotion path goes through **Staging (shadow)** first
- *(Stretch)* BERTopic on negative reviews → top-5 themes per week

**Charlie + Ha (Data & Eval)**
- Real sources wired in: `yelp_loader.py`, `malaysia_loader.py`, `replay.py`
- Full Great Expectations suite (length bounds, language detect, duplicate rate, label cardinality)
- Evidently report uploaded to MinIO per DAG run; row written to `monitoring_reports`
- **Model validation gate**: DAG fails (and blocks promotion) if F1 drops > 3% or drift score > τ
- **Retrain trigger**: `monitoring/retrain_trigger.py` calls Airflow REST API to kick off `train_model` when drift > τ; writes `triggered_retrain=true` to the report row

**Anh (Glue)**
- CI matrix: lint, unit, integration (boots compose, hits `/predict`)
- Push images to GitHub Container Registry on tag
- Secrets via `.env` + GitHub Actions secrets
- *(Stretch)* Terraform stub for AWS lift (RDS + S3 + ECS)

**Amelia (Serving + UI)**
- `/predict` supports batch input
- **Shadow deploy**: when MLflow has a `Staging` version, FastAPI loads it alongside Production, runs both, logs both to `predictions` with `stage` set accordingly. Only Production responses are returned to the caller.
- `/admin/reload` to refresh registry without restart
- Streamlit: sentiment timeline, word cloud of negative reviews, model A/B comparison (Production vs Staging from `predictions` table), drift report embed
- `human_corrections` write path: a dashboard tile lets reviewers fix a wrong prediction; row goes to `human_corrections` and next training run picks it up
- End-to-end pytest hitting the live compose stack

### Phase 2 Definition of Done
- [ ] DistilBERT beats baseline on held-out F1 in MLflow
- [ ] Drift report runs every 6h and is visible in the dashboard
- [ ] A poisoned batch (via replay simulator) actually trips the gate and triggers retraining
- [ ] Shadow lane produces N hours of paired Production/Staging predictions before promotion
- [ ] A human correction in the dashboard appears in the next training run's label set
- [ ] Integration test in CI: stack up, ingest, refine, build, train, predict, assert

---

## 5. Phase 3 — Polish & Demo (Week 6)

**Goal:** Anyone can clone the repo and run the demo. The story is presentable.

- README walks a new dev from zero to `/predict` in under 10 minutes
- `scripts/demo.sh` — seeds the replay simulator, triggers DAGs, opens browser tabs
- **Drift demo flow:** replay a clean window → replay a poisoned window → Evidently fires → retraining kicks off → shadow comparison → promotion gate decides
- Polished SVG architecture diagram alongside the ASCII version
- 5-minute walkthrough deck (problem, stack, medallion, results, feedback loop, what's next)
- Postmortem: what surprised us, what we'd cut, what we'd build with another month

---

## 6. Handoff Contracts

| Boundary | Owners | Contract |
|---|---|---|
| Bronze object layout | Charlie/Ha → Charlie/Ha | `s3://datasets/bronze/{source}/{YYYY-MM-DD}/*.json[.gz]` with provenance fields |
| `reviews_silver` table | Charlie/Ha → Van + Amelia | Columns + types in `data/schemas/silver.sql`; PII-masked |
| `reviews_gold` table | Charlie/Ha (with Van for embeddings) → Van | Columns + label derivation from `rating` |
| MLflow run logging | Van → Charlie/Ha | Metric names (`f1_macro`, `precision_neg`, …), tag conventions |
| MLflow registered model name | Van → Amelia | `sentiment-baseline` (P1), `sentiment-distilbert` (P2); aliases `Production`, `Staging` |
| FastAPI request/response schema | Amelia → dashboard, downstream | Pydantic models in `api/app/schemas.py` |
| Evidently report location | Charlie/Ha → Amelia | MinIO path + `monitoring_reports` row |
| `human_corrections` write path | Amelia → Van | Schema in `data/schemas/`; training DAG reads it next run |
| Env vars | Anh → all | `infra/.env.example` is canonical |

When a contract changes, the owner opens a PR that updates the contract file *and* every consumer. No silent drift.

---

## 7. Ways of Working

- **PRs are small.** < 400 lines changed. If it grows, split.
- **One owner per folder.** Cross-folder PRs need both owners to approve.
- **Tests live next to code.** `models/test_train.py`, `api/app/test_main.py`, etc. The top-level `tests/` is for cross-cutting integration tests; `docker compose run --rm smoke` is the lightest gate.
- **Daily 15-min standup** during Phase 1 (most coupling). Async after.
- **Block early.** If you're blocked > 4 hours waiting on a contract, flag in chat; don't build around it.
- **Reproducibility is a feature.** Every new dependency is pinned in the relevant `requirements.txt` and the Dockerfile.

---

## 8. Stretch Goals (after Phase 2 DoD)

- BERTopic theme extraction → weekly "what users complain about" widget
- Multilingual (XLM-R / mDistilBERT) — needed for Malaysia Restaurant Reviews
- Active-learning loop: low-confidence predictions queued for human review in the dashboard
- Terraform → ECS lift, demo on a tiny cloud instance
- Slack alert from Evidently when drift > τ

---

## 9. Quick Reference

| What | Where |
|---|---|
| Spin up locally | `docker compose up` from repo root |
| Run smoke tests | `docker compose run --rm smoke` |
| Airflow UI | `http://localhost:8080` |
| MLflow UI | `http://localhost:5000` |
| FastAPI docs | `http://localhost:8000/docs` |
| Streamlit dashboard | `http://localhost:8501` |
| MinIO console | `http://localhost:9001` |
| Postgres | `localhost:5432`, see `.env.example` for creds |
