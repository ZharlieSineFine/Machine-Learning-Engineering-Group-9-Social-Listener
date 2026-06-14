#!/usr/bin/env bash
# Run the medallion pipeline inside the data-pipeline compose service.
#
# Usage (from repo root):
#   scripts/run_daily_docker.sh
#   scripts/run_daily_docker.sh --run-date 2026-06-10 --sources yelp tripadvisor
#   scripts/run_daily_docker.sh --skip-bronze

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  cp infra/.env.example .env
  echo "Created .env from infra/.env.example"
fi

if [[ $# -eq 0 ]]; then
  set -- --run-date "$(date +%Y-%m-%d)" --sources yelp tripadvisor
fi

docker compose --profile pipeline run --rm data-pipeline "$@"
