#!/usr/bin/env bash
# Bring up the infra subset needed for integration tests.
#
# Usage:
#   scripts/up.sh            # postgres + minio + mlflow (Step 1 infra)
#   scripts/up.sh all        # everything in docker-compose.yml
#   scripts/up.sh down       # stop + remove containers (keeps volumes)
#   scripts/up.sh nuke       # stop + remove containers AND volumes
#
# Requires: Docker Desktop running. The script verifies the daemon first.

set -euo pipefail

cd "$(dirname "$0")/.."

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: docker daemon not reachable. Start Docker Desktop and retry." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "First run: copying infra/.env.example -> .env"
  cp infra/.env.example .env
fi

case "${1:-up}" in
  up)
    docker compose up -d postgres minio minio-init mlflow
    echo
    echo "Waiting for services to become healthy..."
    for svc in postgres minio mlflow; do
      printf "  %-10s " "$svc"
      # mlflow has no healthcheck — just wait for the port.
      if [[ "$svc" == "mlflow" ]]; then
        for _ in $(seq 1 30); do
          if curl -sf http://localhost:5001/health >/dev/null 2>&1 || \
             curl -sf http://localhost:5001/ >/dev/null 2>&1; then
            echo "ok"; break
          fi
          sleep 1
        done
      else
        for _ in $(seq 1 30); do
          state=$(docker inspect -f '{{.State.Health.Status}}' "sentiment-$svc" 2>/dev/null || echo "starting")
          if [[ "$state" == "healthy" ]]; then echo "healthy"; break; fi
          sleep 1
        done
      fi
    done
    echo
    echo "Postgres : localhost:5432  (sentiment / airflow / mlflow)"
    echo "MinIO    : http://localhost:9001  (console)"
    echo "MLflow   : http://localhost:5001"
    ;;
  all)
    docker compose up -d
    ;;
  down)
    docker compose down
    ;;
  nuke)
    docker compose down -v
    ;;
  *)
    echo "unknown command: $1" >&2
    exit 2
    ;;
esac
