# Smoke-test runner.
#
# Self-contained image that runs the Phase 1 smoke test (tests/test_smoke.py)
# with NO other services running. Use this locally or in CI as a fast
# pre-flight before bringing the full compose stack up.
#
# Build:
#   docker build -f infra/docker/smoke/Dockerfile -t sentiment-smoke .
# Run:
#   docker run --rm sentiment-smoke
# Via compose:
#   docker compose run --rm smoke
#
# Build context MUST be the project root (the COPY paths assume that).

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY infra/docker/smoke/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy only what the smoke test needs. Keep this list minimal so changes to
# unrelated folders (airflow/, dashboard/, monitoring/, ...) don't bust the
# Docker layer cache and trigger a slow rebuild.
COPY models/        /app/models/
COPY api/           /app/api/
COPY data/sample/   /app/data/sample/
COPY tests/         /app/tests/

# tests/conftest.py adds /app and /app/api to sys.path, so `models.*` and
# `app.*` both resolve.
ENV PYTHONPATH=/app:/app/api

CMD ["pytest", "-v", "tests/test_smoke.py"]
