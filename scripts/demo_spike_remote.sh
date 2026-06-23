#!/usr/bin/env bash
# scripts/demo_spike_remote.sh  —  DEMO STEP 2 for a REMOTE / shared host.
#
# Same negative-review spike as scripts/demo_spike.sh, but in-container (no host venv,
# no host DB port). Scores today's burst with the champion and runs the Evidently
# drift check, which records the alert in monitoring_reports (no auto-retrain).
set -euo pipefail
cd "$(dirname "$0")/.."

SCHED="sentiment-airflow-scheduler"
SPIKE_DAY="${SPIKE_DAY:-2026-06-21}"           # spike day baked into demo_jun2026_spike.csv
DS="$(date -u +%Y-%m-%d)"                       # UTC logical date — matches reviews' UTC ingested_at (no +tz day-skew)
CHAMPION_CTR="/opt/project/models/artifacts/champion_baseline_v3.pkl"
[ -f "models/artifacts/champion_baseline_v3.pkl" ] && MPP="$CHAMPION_CTR" || MPP="/opt/project/models/artifacts/baseline.pkl"

echo "================================================================"
echo "  NEGATIVE-REVIEW SPIKE  -  simulating a brand crisis hitting today"
echo "================================================================"
echo
echo "==> [1/2] Inference: scoring today's review burst with the champion (in-container)..."
docker exec -e POSTGRES_HOST=postgres -e MODEL_PICKLE_PATH="$MPP" "$SCHED" \
  python -m serving.batch_infer --scenario spike --asof "$SPIKE_DAY" --n-recent 1 --as-now --clear-today

echo
echo "==> [2/2] Airflow: evaluate_and_monitor records the drift alert (Evidently)..."
docker exec -e DRIFT_REPLAY_SCENARIO=spike -e DRIFT_REPLAY_ASOF="$SPIKE_DAY" -e DRIFT_REPLAY_N_RECENT=1 \
  "$SCHED" airflow tasks test evaluate_and_monitor drift_check "$DS" 2>&1 |
  grep -aE "replay-drift:spike|drift_score=|blocked=True|evaluate_and_monitor\]" || true

echo
echo "DONE. Drift detected -> alert recorded in monitoring_reports (dashboard shows it)."
echo "  Reset to a normal day with: bash scripts/demo_up_remote.sh"
