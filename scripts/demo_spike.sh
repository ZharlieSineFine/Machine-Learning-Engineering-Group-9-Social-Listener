#!/usr/bin/env bash
# scripts/demo_spike.sh  -  DEMO STEP 2: inject the negative spike + trigger the MLOps response.
# Bash mirror of demo_spike.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"

SPIKE_DAY="2026-06-21"                       # spike day baked into demo_jun2026_spike.csv
DS="$(date +%Y-%m-%d)"                       # Airflow logical date for the drift task
export POSTGRES_HOST=localhost POSTGRES_PORT=5432 POSTGRES_USER=mlops POSTGRES_PASSWORD=mlops POSTGRES_DB=sentiment

echo "================================================================"
echo "  NEGATIVE-REVIEW SPIKE  -  simulating a brand crisis hitting today"
echo "================================================================"
echo
echo "==> [1/2] Inference: scoring today's review burst with the champion model..."
"$PY" -m serving.batch_infer --scenario spike --asof "$SPIKE_DAY" --n-recent 1 --as-now --clear-today

echo
echo "==> [2/2] Airflow: evaluate_and_monitor detecting drift (Evidently)..."
docker exec -e DRIFT_REPLAY_SCENARIO=spike -e DRIFT_REPLAY_ASOF="$SPIKE_DAY" -e DRIFT_REPLAY_N_RECENT=1 \
  sentiment-airflow-scheduler airflow tasks test evaluate_and_monitor compute_and_log_drift "$DS" 2>&1 |
  grep -aE "replay-drift:spike|drift_score=|blocked=True|evaluate_and_monitor\]" || true
echo
echo "    Drift blocked the gate -> triggering retrain (medallion_train_cycle)..."
docker exec sentiment-airflow-scheduler airflow dags trigger medallion_train_cycle 2>&1 |
  grep -aE "queued|created|medallion_train_cycle" || true

echo
echo "DONE. The dashboard now shows the spike + the red alert banner."
echo "  Dashboard : http://localhost:8501   (negative % jumps, alert fires)"
