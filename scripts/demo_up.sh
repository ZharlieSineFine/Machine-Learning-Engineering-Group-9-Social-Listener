#!/usr/bin/env bash
# scripts/demo_up.sh  -  DEMO STEP 1: full stack (incl MLflow registry + FastAPI) + clean day.
# Bash mirror of demo_up.ps1 (PowerShell is the primary path on Windows).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"

wait_url() { for _ in $(seq 1 40); do [ "$(curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null)" = "200" ] && return 0; sleep 2; done; return 1; }

echo "==> [1/6] Starting services (postgres, minio, mlflow, airflow, dashboard)..."
docker compose up -d postgres minio minio-init mlflow airflow-init airflow-webserver airflow-scheduler dashboard >/dev/null

echo "==> [2/6] Waiting for Postgres, MLflow, Airflow..."
for _ in $(seq 1 40); do docker exec sentiment-postgres pg_isready -U mlops >/dev/null 2>&1 && break; sleep 2; done
wait_url http://localhost:5001/ || true
for _ in $(seq 1 40); do docker exec sentiment-airflow-scheduler airflow version >/dev/null 2>&1 && break; sleep 2; done

echo "==> [3/6] Registering the champion in MLflow (idempotent)..."
MSYS_NO_PATHCONV=1 docker exec sentiment-airflow-scheduler python /opt/project/scripts/register_champion.py 2>&1 | grep -aE "register\]" || true

echo "==> [4/6] Starting the API (serves sentiment-baseline/Production from MLflow)..."
docker compose up -d api >/dev/null

# Host-side Postgres access for the inference below. Set AFTER compose up so it
# cannot leak into the containers' ${POSTGRES_HOST}.
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops POSTGRES_DB=sentiment

echo "==> [5/6] Generating replay + scoring a clean 2-week history -> reviews table..."
"$PY" -m data.ingest.replay --scenario stable >/dev/null
"$PY" -m data.ingest.replay --scenario spike  >/dev/null
"$PY" -m serving.batch_infer --scenario stable --shift-to-today --truncate

echo "==> [6/6] Seeding the shadow-deploy log (API /predict/batch)..."
if wait_url http://localhost:8000/health; then "$PY" scripts/seed_shadow.py; else echo "    (API not ready; skipping shadow seed)"; fi

echo
echo "Stack is up and the dashboard shows a NORMAL day."
echo "  Dashboard : http://localhost:8501   (Marketing + MLOps Monitor pages)"
echo "  API       : http://localhost:8000/docs"
echo "  MLflow    : http://localhost:5001   (sentiment-baseline: Production + Staging)"
echo "  Airflow   : http://localhost:8080   (airflow / airflow)"
echo
echo "Next:  ./scripts/demo_spike.sh   to inject the negative-review spike."
