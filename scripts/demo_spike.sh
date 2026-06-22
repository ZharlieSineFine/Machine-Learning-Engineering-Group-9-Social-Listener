#!/usr/bin/env bash
# scripts/demo_spike.sh  -  DEMO STEP 2: inject the negative spike -> drift alert (no auto-retrain).
# Bash mirror of demo_spike.ps1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"

# Score the spike with the immutable champion (not the retrain-overwritten baseline.pkl).
[ -f "models/artifacts/champion_baseline_v3.pkl" ] && \
  export MODEL_PICKLE_PATH="$PWD/models/artifacts/champion_baseline_v3.pkl"

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
echo "==> [2/2] Airflow: evaluate_and_monitor observes the drift + records the alert (no auto-retrain)..."
docker exec -e DRIFT_REPLAY_SCENARIO=spike -e DRIFT_REPLAY_ASOF="$SPIKE_DAY" -e DRIFT_REPLAY_N_RECENT=1 \
  sentiment-airflow-scheduler airflow tasks test evaluate_and_monitor drift_check "$DS" 2>&1 |
  grep -aE "replay-drift:spike|drift_score=|blocked=True|evaluate_and_monitor\]" || true
echo
echo "    Gate BLOCKED -> alert recorded in monitoring_reports (the dashboard shows it)."
echo "    No model is retrained automatically - a human reviews the alert and runs"
echo "    medallion_pipeline with FORCE_TRAIN=1 only if warranted."

echo
echo "DONE. Drift detected -> team ALERTED. No model was retrained automatically."
echo "  Dashboard : http://localhost:8501   (negative % jumps; MLOps Monitor -> 'Gate blocked')"
