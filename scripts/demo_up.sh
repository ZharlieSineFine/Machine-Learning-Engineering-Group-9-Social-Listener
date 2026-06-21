#!/usr/bin/env bash
# scripts/demo_up.sh  -  DEMO STEP 1: bring the stack up + seed a clean (normal) day.
# Bash mirror of demo_up.ps1 (PowerShell is the primary path on Windows).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"

echo "==> [1/4] Starting the stack (postgres, minio, mlflow, airflow, dashboard)..."
docker compose up -d postgres minio minio-init mlflow airflow-init airflow-webserver airflow-scheduler dashboard >/dev/null

# Host-side Postgres access for the inference below. Set AFTER compose up so it
# cannot leak into the containers' ${POSTGRES_HOST}.
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops POSTGRES_DB=sentiment

echo "==> [2/4] Waiting for Postgres..."
for _ in $(seq 1 40); do docker exec sentiment-postgres pg_isready -U mlops >/dev/null 2>&1 && break; sleep 2; done

echo "==> [3/4] Generating replay streams (clean + spike windows)..."
"$PY" -m data.ingest.replay --scenario stable >/dev/null
"$PY" -m data.ingest.replay --scenario spike  >/dev/null

echo "==> [4/4] Scoring a clean 2-week history with the champion model -> reviews table..."
"$PY" -m serving.batch_infer --scenario stable --shift-to-today --truncate

echo
echo "Stack is up and the dashboard shows a NORMAL day."
echo "  Dashboard : http://localhost:8501   (Marketing view -> 'Sentiment normal')"
echo "  Airflow   : http://localhost:8080   (airflow / airflow)"
echo "  MLflow    : http://localhost:5001"
echo
echo "Next:  ./scripts/demo_spike.sh   to inject the negative-review spike."
