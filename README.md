# BrewLeaf — Brand Sentiment Analysis (MLOps Platform)

End-to-end sentiment analysis for online restaurant/brand reviews, packaged as a
reproducible local MLOps stack: a medallion data pipeline (Bronze → Silver → Gold),
a trained sentiment model tracked in MLflow, live serving via FastAPI, a Streamlit
dashboard, and drift monitoring with a human-in-the-loop retrain gate (Airflow + Evidently).

For the system design, see [ARCHITECTURE.md](./ARCHITECTURE.md). **This file is the
single recipe to set up, run, and test the project — follow it top to bottom.**

> **Demo story:** a *normal day* (~20% negative reviews), then a sudden *negative-review
> spike* (~51%) that the backend detects, alerts on, and blocks at the drift gate.

---

## 0. Prerequisites

Install these once:

| Tool | Check | Install |
|------|-------|---------|
| **Docker Desktop** | `docker info` (no error) | https://www.docker.com/products/docker-desktop |
| **Python 3.11 or 3.12** | `python3 --version` | `brew install python@3.12` (3.13/3.14 not supported — Great Expectations needs numpy<2) |
| **Git** | `git --version` | comes with macOS / `brew install git` |

> **Mac note:** macOS Control Center holds port `5000` (AirPlay). The stack already maps
> MLflow to **`localhost:5001`** to dodge this — don't be surprised it's not on 5000.

---

## 1. First-time setup (once)

```bash
# 1.1  Clone
git clone git@github.com:ZharlieSineFine/Machine-Learning-Engineering-Group-9-Social-Listener.git
cd Machine-Learning-Engineering-Group-9-Social-Listener

# 1.2  Env file (the real .env is gitignored — defaults work out of the box)
cp infra/.env.example .env

# 1.3  Python virtual env
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip

# 1.4  Install Python deps (tests + dashboard + API — covers pytest, streamlit, uvicorn)
pip install -r tests/requirements.txt -r dashboard/requirements.txt -r api/requirements.txt
```

> If `pip` warns about `protobuf` conflicts, that's expected — we pin `protobuf<5` for
> MLflow compatibility.

---

## 2. Start the stack

You have two options. **Option A** is the simplest "everything in Docker"; **Option B**
runs the infra in Docker and the app from your venv (handy for development/debugging).

### Option A — full stack in Docker (one command)

```bash
docker info > /dev/null && echo OK     # confirm Docker is running
docker compose build                   # first run only (a few minutes)
docker compose up -d                   # postgres, minio, mlflow, airflow, api, dashboard
docker compose ps                      # all should be "running"/"healthy"
```

Then jump to **§3** to put a model + data in, and **§5** for the URLs.

### Option B — infra in Docker, app from venv

```bash
./scripts/up.sh                        # postgres + minio + mlflow only
```

Expected tail:

```
Postgres : localhost:5432   MinIO : http://localhost:9001   MLflow : http://localhost:5001
```

---

## 3. Create a model + load data (so the dashboard isn't empty)

A fresh database has no model and no reviews. Run these once (venv active, from repo root):

```bash
# Point the tools at the local MLflow + MinIO
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MODEL_NAME=sentiment-baseline
export MLFLOW_EXPERIMENT=manual-demo

# 3.1  Train + register the baseline (TF-IDF + LogisticRegression) on the sample data
python3 models/train.py

# 3.2  Promote the new version to Production so the API can find it
python3 -c "
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri='http://localhost:5001')
latest = c.get_latest_versions('sentiment-baseline', stages=['None'])
if latest:
    v = latest[0].version
    c.transition_model_version_stage('sentiment-baseline', v, 'Production',
                                     archive_existing_versions=True)
    print(f'promoted v{v} -> Production')
else:
    print('no None-stage version found (already promoted?)')
"

# 3.3  Seed the reviews table so the timeline + word cloud have data
python3 -m data.ingest.ingest_reviews \
  --csv data/sample/reviews_sample.csv \
  --dsn postgresql://mlops:mlops@localhost:5432/sentiment
```

Open http://localhost:5001 — you'll see the `manual-demo` experiment and the
`sentiment-baseline` model at stage **Production**.

> Using **Option A** (full Docker stack)? The `api` container already serves the model from
> the registry; after step 3.2, hit `curl -X POST http://localhost:8000/reload -H "X-Admin-Token: demo-token"`
> to pick up the new version, then skip to §5.

---

## 4. Run the app (Option B — from venv)

### 4.1 FastAPI service — in a **new terminal**

```bash
cd Machine-Learning-Engineering-Group-9-Social-Listener && source .venv/bin/activate
export PYTHONPATH=$PWD
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export MODEL_NAME=sentiment-baseline MODEL_STAGE=Production ADMIN_TOKEN=demo-token
uvicorn app.main:app --app-dir api --host 127.0.0.1 --port 8002
```

Open http://localhost:8002/docs (Swagger). *Port 8002 dodges the common 8000 conflict —
use 8003/8088 if it's busy, and update `API_URL` below to match.*

### 4.2 Streamlit dashboard — in **another new terminal**

```bash
cd Machine-Learning-Engineering-Group-9-Social-Listener && source .venv/bin/activate
export API_URL=http://localhost:8002
export POSTGRES_HOST=localhost POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops
export POSTGRES_DB=sentiment POSTGRES_PORT=5432
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_EXPERIMENT=manual-demo
streamlit run dashboard/app.py
```

Open http://localhost:8501 — a Marketing view (KPI tiles, timeline, alerts) and an
MLOps Monitor page (model, shadow deploy, drift).

---

## 5. Service URLs

| Service | URL | Login |
|---|---|---|
| Streamlit dashboard | http://localhost:8501 | — |
| FastAPI docs (Swagger) | http://localhost:8002/docs (Option B) · http://localhost:8000/docs (Option A) | — |
| MLflow UI | http://localhost:5001 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Airflow (Option A) | http://localhost:8080 | `airflow` / `airflow` |
| Postgres | `psql postgresql://mlops:mlops@localhost:5432/sentiment` | — |

> Credentials are the local-dev defaults from `infra/.env.example`. Override them in your
> own `.env` for any non-local deployment. `.env` is gitignored.

---

## 6. Run the tests

```bash
# Self-contained smoke test in an isolated container (no venv needed)
docker compose run --rm smoke

# Unit tests (fast, from the venv)
pytest tests/ --ignore=tests/integration -q

# Integration tests (need the stack from §2 running)
pytest tests/integration/ -q

# Opt-in slow test (downloads ~250 MB DistilBERT weights on first run)
RUN_SLOW=1 pytest tests/test_distilbert_slow.py -q
```

The marquee check is `tests/integration/test_e2e_pipeline.py` — it walks CSV → ingest →
train → register → promote → serve → predict → reload → drift gate in one test.

---

## 7. Things to try once it's running

| Try this | Where | Proves |
|----------|-------|--------|
| Type a review → **Predict** | Dashboard | API ↔ dashboard loop |
| `POST /predict {"text":"awful service"}` | Swagger | API works |
| `POST /reload` **without** `X-Admin-Token` | Swagger | 401 — auth enforced |
| `POST /reload` **with** `X-Admin-Token: demo-token` | Swagger | 200 — hot-reload |
| Browse the `mlflow/` bucket | MinIO console | the pickled model artifact |

---

## 8. Troubleshooting

- **"Cannot connect to the Docker daemon"** — Docker Desktop isn't running; start it and retry.
- **"port is already allocated"** — find the owner with `lsof -nP -iTCP:<port> -sTCP:LISTEN`,
  then kill it or change the port (uvicorn `--port`, `streamlit ... --server.port`).
- **`ModuleNotFoundError: No module named 'models'`** when starting the API — run
  `export PYTHONPATH=$(git rev-parse --show-toplevel)` from the repo root.
- **API `/health` shows `model_loaded: false`** — no Production model yet; run §3.1–3.2.
- **`role "mlops" does not exist`** — `init.sql` only runs on a fresh volume:
  `./scripts/up.sh nuke && ./scripts/up.sh`.
- **`PromotionBlocked: f1_drop=...`** — working as designed: the drift gate blocked a
  regression. Fix the data or train on the new distribution.

---

## 9. Stop / clean up

```bash
# Option B helpers:
./scripts/up.sh down     # stop infra (volumes kept)
./scripts/up.sh nuke     # wipe everything incl. Postgres + MinIO data

# Option A:
docker compose down      # stop the full stack (volumes kept)
```

---

## Tech stack

Airflow · Postgres · MinIO · MLflow · scikit-learn + HuggingFace (DistilBERT) ·
Great Expectations · Evidently · FastAPI · Streamlit · Docker Compose · GitHub Actions
