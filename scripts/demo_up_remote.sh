#!/usr/bin/env bash
# scripts/demo_up_remote.sh  —  DEMO STEP 1 for a REMOTE / shared host.
#
# Same "normal day" as scripts/demo_up.sh, but every Python step runs INSIDE the
# airflow-scheduler container (which already has sklearn/pandas/pyarrow/psycopg2 and
# reaches postgres:5432 on the compose network). So it needs NO host venv and NO
# published host DB port — safe to run alongside other projects.
#
# Prereq: `docker compose up -d` is already running (with the remote .env), and the
# champion pickle is in models/artifacts/ (mounted into the container).
set -euo pipefail
cd "$(dirname "$0")/.."

SCHED="sentiment-airflow-scheduler"
CHAMPION_HOST="models/artifacts/champion_baseline_v3.pkl"
CHAMPION_CTR="/opt/project/models/artifacts/champion_baseline_v3.pkl"

# in-container exec helper: in-network Postgres + champion-pinned scoring
dex() { docker exec -e POSTGRES_HOST=postgres -e MODEL_PICKLE_PATH="$MPP" "$SCHED" "$@"; }

# --- pre-demo champion check (file lives on the host, mounted into the container) ---
if [ -f "$CHAMPION_HOST" ]; then
  MPP="$CHAMPION_CTR"; echo "==> [check] champion model found -> $CHAMPION_HOST"
else
  MPP="/opt/project/models/artifacts/baseline.pkl"
  echo "============================================================"
  echo "  WARNING: champion model NOT found: $CHAMPION_HOST"
  echo "  Falling back to baseline.pkl — demo numbers will be OFF."
  echo "  scp champion_baseline_v3.pkl into models/artifacts/ first."
  echo "============================================================"
  [ "${REQUIRE_CHAMPION:-0}" = "1" ] && { echo "REQUIRE_CHAMPION=1 -> aborting."; exit 1; }
fi

# The in-container airflow user must be able to write the bind-mounted data/ +
# monitoring/ dirs (replay output, drift reports). After an rsync deploy these keep
# the sender's ownership, so widen perms or the in-container steps hit Permission denied.
chmod -R a+rwX data monitoring 2>/dev/null || true

echo "==> [1/4] Waiting for Postgres, MLflow, Airflow..."
for _ in $(seq 1 60); do docker exec sentiment-postgres pg_isready -U "${POSTGRES_USER:-mlops}" >/dev/null 2>&1 && break; sleep 2; done
for _ in $(seq 1 60); do docker exec "$SCHED" airflow version >/dev/null 2>&1 && break; sleep 2; done

echo "==> [2/4] Registering the champion in MLflow (idempotent)..."
docker exec -e MODEL_PICKLE_PATH="$MPP" "$SCHED" python /opt/project/scripts/register_champion.py 2>&1 | grep -aE "register\]" || true

echo "==> [3/4] Replay + scoring a clean 2-week history -> reviews (in-container)..."
dex python -m data.ingest.replay --scenario stable >/dev/null
dex python -m data.ingest.replay --scenario spike  >/dev/null
dex python -m serving.batch_infer --scenario stable --shift-to-today --truncate

# Reset the drift signal (mirrors demo_up.sh): seed ~2 weeks of low-PSI rows so the
# drift sparkline has a steady baseline for the spike to stand against, then a clean
# 0 for today. REPLACE (not TRUNCATE-only) so the dashboard panel doesn't fall back
# to a live `import monitoring` check it can't satisfy. Units are label PSI (what
# evaluate_and_monitor now records).
read -r -d '' RESET_SQL <<'SQL' || true
TRUNCATE monitoring_reports;
INSERT INTO monitoring_reports (run_date, report_type, report_url, drift_score, blocked_promotion)
SELECT d::date, 'data_drift', 'baseline backfill (demo seed)',
       round((0.003 + random()*0.025)::numeric, 4), false
FROM generate_series(CURRENT_DATE - INTERVAL '13 days',
                     CURRENT_DATE - INTERVAL '1 day', INTERVAL '1 day') AS d;
INSERT INTO monitoring_reports (run_date, report_type, report_url, drift_score, blocked_promotion)
VALUES (CURRENT_DATE, 'data_drift', 'baseline clean day (demo_up reset)', 0, false);
SQL
docker exec sentiment-postgres psql -U "${POSTGRES_USER:-mlops}" -d "${POSTGRES_DB:-sentiment}" \
  -c "$RESET_SQL" >/dev/null

echo "==> [4/4] Seeding the shadow-deploy log (API /predict/batch, in-network)..."
docker exec -e API_URL=http://api:8000 "$SCHED" python /opt/project/scripts/seed_shadow.py || \
  echo "    (API not ready; skipping shadow seed)"

echo
echo "Stack is up and the dashboard shows a NORMAL day."
echo "  Dashboard is on 127.0.0.1:${DASHBOARD_HOST_PORT:-58501} — front it with your reverse proxy."
echo "Next:  bash scripts/demo_spike_remote.sh   to inject the negative-review spike."
