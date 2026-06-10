# Runbook — How to Run & Test This Project

This is a step-by-step recipe. Follow it top to bottom the first time.
Once you've done it once, you can jump around.

For the code itself, see [CODE_WALKTHROUGH.md](./CODE_WALKTHROUGH.md).

---

## 0. Prerequisites

Install these once, on your laptop:

| Tool | How to check | How to install |
|------|--------------|----------------|
| **Docker Desktop** | `docker info` (should not error) | https://www.docker.com/products/docker-desktop |
| **Python 3.11+** | `python3 --version` | macOS: comes with Xcode tools / `brew install python@3.11` |
| **Git** | `git --version` | Comes with macOS / `brew install git` |

**Mac users only — heads up:** macOS Control Center holds port `5000` (AirPlay
Receiver). Our compose file already maps MLflow to `5001` to dodge this. You
don't need to do anything, but don't be surprised that MLflow is on
`localhost:5001`, not `5000`.

---

## 1. First-time setup (do this once)

```bash
# 1.1  Clone the repo
git clone git@github.com:ZharlieSineFine/Machine-Learning-Engineering-Group-9-Social-Listener.git
cd Machine-Learning-Engineering-Group-9-Social-Listener

# 1.2  Copy the env file (real .env is gitignored)
cp infra/.env.example .env

# 1.3  Create a Python virtual env (recommended — keeps deps out of system Python)
python3 -m venv .venv
source .venv/bin/activate

# 1.4  Install Python test dependencies
pip install --upgrade pip
pip install -r tests/requirements.txt
```

> **Tip:** if `pip install` complains about `protobuf` conflicts, that's
> fine — the test requirements pin `protobuf<5` deliberately for MLflow
> compatibility. The conflict warning is from unrelated system packages.

---

## 2. Start Docker, then the stack

```bash
# 2.1  Make sure Docker Desktop is running. Verify:
docker info > /dev/null && echo OK

# 2.2  Bring up the infra (postgres + minio + mlflow). One command:
./scripts/up.sh
```

What you should see in the last lines:

```
Waiting for services to become healthy...
  postgres   healthy
  minio      healthy
  mlflow     ok

Postgres : localhost:5432  (sentiment / airflow / mlflow)
MinIO    : http://localhost:9001  (console)
MLflow   : http://localhost:5001
```

**Verify in your browser:**
- MinIO console → http://localhost:9001 — login `minioadmin` / `minioadmin`
- MLflow UI → http://localhost:5001 — shows "No experiments yet"

If something didn't come up, see **§6 Troubleshooting**.

---

## 3. Run the tests (four levels)

We have tests at four levels. Start with the fastest, move up.

### Level 1 — Unit + smoke tests (no services needed, ~15s)

These are pure Python. They run against in-memory data + a local pickle.

```bash
pytest tests/ --ignore=tests/integration -v
```

Expected: **~38 passed**, plus 1 skipped (the slow DistilBERT test).

### Level 2 — Integration tests (needs the infra from §2, ~30s)

These hit the live Postgres, MinIO, and MLflow on your laptop.

```bash
pytest tests/integration/ -v
```

Expected: **~17 passed**. The marquee one is
`test_e2e_pipeline.py::test_e2e_pipeline` — that single test walks the
whole pipeline from CSV → ingest → train → register → promote → serve →
predict → reload → drift gate. If it goes green, everything works.

### Level 3 — Slow / opt-in test (downloads DistilBERT weights, ~30s after cache)

```bash
RUN_SLOW=1 pytest tests/test_distilbert_slow.py -v
```

First run downloads ~250 MB of model weights from HuggingFace; subsequent
runs are fast. Skip this if you're in a hurry.

### Level 4 — Manual smoke (full UI — see §4 below)

This is the one your team will actually use day to day.

---

## 4. Run the app for manual testing

The stack from §2 doesn't include the API or the dashboard yet — let's
start those.

### 4.1 — Train + promote a model (one time per fresh DB)

If you wiped the volumes (`./scripts/up.sh nuke`) or are starting cold,
MLflow has no models yet. Run this to create + promote one:

```bash
# Make sure we're pointing at the local MLflow
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MODEL_NAME=sentiment-baseline
export MLFLOW_EXPERIMENT=manual-demo

# Train + register
python3 models/train.py

# Promote the latest version to Production so the API can find it
python3 -c "
import os
from mlflow.tracking import MlflowClient
c = MlflowClient(tracking_uri='http://localhost:5001')
latest = c.get_latest_versions('sentiment-baseline', stages=['None'])
if latest:
    v = latest[0].version
    c.transition_model_version_stage('sentiment-baseline', v,
                                     'Production', archive_existing_versions=True)
    print(f'promoted v{v} -> Production')
else:
    print('no None-stage version found (already promoted?)')
"
```

Open http://localhost:5001 — you should see the `manual-demo` experiment
with one run, and a registered model `sentiment-baseline` at stage
`Production`.

### 4.2 — Start the FastAPI service

In a **new terminal** (keep §2 running):

```bash
cd Machine-Learning-Engineering-Group-9-Social-Listener
source .venv/bin/activate

export PYTHONPATH=$PWD
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MODEL_NAME=sentiment-baseline
export MODEL_STAGE=Production
export ADMIN_TOKEN=demo-token

cd api
uvicorn app.main:app --host 127.0.0.1 --port 8002
```

> **Why port 8002?** On Mac, port 8000 is often held by other dev servers.
> Pick something free. If `8002` is also taken: try `8003`, `8088`, etc.
> Just update `API_URL` in the dashboard step below.

Open http://localhost:8002/docs — Swagger UI lets you click "Try it out"
on every endpoint.

### 4.3 — Start the Streamlit dashboard

In **another new terminal**:

```bash
cd Machine-Learning-Engineering-Group-9-Social-Listener
source .venv/bin/activate

export API_URL=http://localhost:8002
export POSTGRES_HOST=localhost
export POSTGRES_USER=mlops
export POSTGRES_PASSWORD=mlops
export POSTGRES_DB=sentiment
export POSTGRES_PORT=5432
export MLFLOW_TRACKING_URI=http://localhost:5001
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_EXPERIMENT=manual-demo

streamlit run dashboard/app.py
```

Open http://localhost:8501 — you should see 5 sections (tiles, timeline,
word cloud, MLflow A/B, drift report).

### 4.4 — Optional: seed the `reviews` table so the dashboard isn't empty

The dashboard tries Postgres first, falls back to CSV. If Postgres is
empty, the timeline and word cloud will look thin. Ingest the sample:

```bash
python3 -m data.ingest.ingest_reviews \
  --csv data/sample/reviews_sample.csv \
  --dsn postgresql://mlops:mlops@localhost:5432/sentiment
```

Expected output: `Ingested 981 rows into reviews from data/sample/reviews_sample.csv`.
Refresh the dashboard — tiles + timeline now show real numbers.

### 4.5 — Optional: generate a drift report so the bottom section isn't empty

```bash
python3 -c "
import os, pandas as pd, boto3, psycopg2
from botocore.client import Config
from monitoring.drift_checks import evaluate

df = pd.read_csv('data/sample/reviews_sample.csv')[['text','label','rating','source']].dropna()
cut = int(len(df) * 0.8)

s3 = boto3.client('s3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin', aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'), region_name='us-east-1')

conn = psycopg2.connect('postgresql://mlops:mlops@localhost:5432/sentiment')
print(evaluate(df.iloc[:cut], df.iloc[cut:], conn, s3))
conn.commit()
conn.close()
"
```

Refresh the dashboard — the **Latest drift report** section now embeds the
full Evidently HTML.

---

## 5. Things to try once it's running

Open the dashboard at http://localhost:8501 and the Swagger UI at
http://localhost:8002/docs side by side.

| Try this | Where | What it proves |
|----------|-------|----------------|
| Type a review in the "Live prediction probe" → click **Predict** | Dashboard | The whole API ↔ dashboard loop works |
| `POST /predict` from Swagger with `{"text":"awful service"}` | http://localhost:8002/docs | API alone works |
| `POST /predict/batch` with 3 texts | Swagger | Batch endpoint works |
| `POST /reload` **without** `X-Admin-Token` | Swagger | 401 — auth is enforced |
| `POST /reload` **with** `X-Admin-Token: demo-token` | Swagger | 200 — admin endpoint works |
| Browse the `mlflow/` bucket in MinIO console | http://localhost:9001 | See the actual pickled model artifact |
| Open the `sentiment-baseline` registered model in MLflow | http://localhost:5001 | See version history + metrics |
| Train another version → promote → hit `/reload` | terminal + Swagger | Hot-reload works without restart |

**Hot-reload recipe (copy-paste):**
```bash
# Train v_N+1
MLFLOW_TRACKING_URI=http://localhost:5001 \
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin \
MODEL_NAME=sentiment-baseline python3 -c "
from models.train import run
from mlflow.tracking import MlflowClient
r = run()
MlflowClient('http://localhost:5001').transition_model_version_stage(
    'sentiment-baseline', r.mlflow_model_version,
    'Production', archive_existing_versions=True)
print('promoted v', r.mlflow_model_version)
"

# Tell the live API to pick it up
curl -X POST http://localhost:8002/reload -H "X-Admin-Token: demo-token"
```

---

## 6. Troubleshooting

### "Cannot connect to the Docker daemon"
Docker Desktop isn't running. Open the app, wait for the whale icon to
go green, then retry.

### "address already in use" / "port is already allocated"
Some other process owns the port. Find it:
```bash
lsof -nP -iTCP:5001 -sTCP:LISTEN  # replace 5001 with the conflicting port
```
Either kill that process or change our port:
- **Port 5000** (Mac AirPlay): we already moved MLflow to 5001
- **Port 8000** (common dev port): start uvicorn with `--port 8002` and
  set `API_URL=http://localhost:8002` for the dashboard
- **Port 8501** (other Streamlit app): `streamlit run dashboard/app.py --server.port=8502`

### `ModuleNotFoundError: No module named 'models'` when starting the API
You didn't `export PYTHONPATH=$PWD` from the repo root, OR you ran
`cd api && uvicorn ...` without setting it first. The fix:
```bash
export PYTHONPATH=$(git rev-parse --show-toplevel)
```

### API `/health` shows `model_loaded: false`
The API tried MLflow but found no Production model. Either run §4.1, or
unset `MLFLOW_TRACKING_URI` and `MODEL_NAME` to fall back to the pickle.

### `RESTORE FAILED... role "mlops" does not exist`
Postgres `init.sql` only runs on a fresh volume. If you upgraded from an
older version of this repo, wipe the volume:
```bash
./scripts/up.sh nuke
./scripts/up.sh
```

### Tests hang on `TRUNCATE`
The test connection still has an open read transaction. Make sure
`tests/integration/conftest.py` sets `conn.autocommit = True` (it should).
If you customised it, set autocommit back on the read-only fixture.

### Dashboard shows "No MLflow runs found"
`MLFLOW_TRACKING_URI` not exported in the dashboard terminal, or the
`MLFLOW_EXPERIMENT` env var points at an empty experiment. Set both.

### `PromotionBlocked: f1_drop=...` from `evaluate()`
**That's working as designed.** It means the current batch's labels
don't match what the Production model would predict, so the drift gate
correctly blocked promotion. Either fix the data, or train a new model
on the new distribution.

---

## 7. Stopping & cleaning up

```bash
# Stop API + Streamlit: Ctrl-C in their terminals
# (or, if you ran them in background:
#   pkill -f "uvicorn app.main"
#   pkill -f "streamlit run" )

# Stop the docker stack (containers gone, volumes kept):
./scripts/up.sh down

# Wipe EVERYTHING including Postgres + MinIO data (use when you want a
# truly fresh start, e.g. to re-test init.sql):
./scripts/up.sh nuke
```

---

## 8. Quick-reference URLs

| What | URL | Login |
|------|-----|-------|
| Streamlit dashboard | http://localhost:8501 | — |
| FastAPI docs (Swagger) | http://localhost:8002/docs | — |
| MLflow UI | http://localhost:5001 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Postgres (psql) | `psql postgresql://mlops:mlops@localhost:5432/sentiment` | — |

---

## 9. What to do if something is broken end-to-end

Work bottom-up — check the layer below before blaming the layer above.

1. Is Docker daemon up? → `docker info`
2. Are the containers healthy? → `docker ps` (look for "healthy")
3. Can you reach the services? → `curl http://localhost:5001/`, `curl http://localhost:9000/minio/health/live`
4. Do the integration tests pass? → `pytest tests/integration/test_infra.py -v`
5. Does the e2e test pass? → `pytest tests/integration/test_e2e_pipeline.py -v`
6. Does the API `/health` say `model_loaded: true`?
7. Does the dashboard load?

If step 4 fails, the infra is broken. If step 5 fails but 4 passes, a
specific layer is broken — the e2e test's traceback will tell you which.
