# Medallion pipeline runner — bronze → silver → GE gate → gold.
#
# Build (from repo root):
#   docker build -f infra/docker/data-pipeline/Dockerfile -t sentiment-data-pipeline .
#
# Run:
#   docker compose run --rm data-pipeline --run-date 2026-06-10 --sources yelp tripadvisor
#
# Mounts raw Yelp / TripAdvisor datasets and writes bronze/silver/gold under ./data/.

FROM python:3.11-slim

WORKDIR /opt/project

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/project

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY infra/docker/data-pipeline/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY data/ /opt/project/data/

ENTRYPOINT ["python", "-m", "data.run_daily"]
CMD ["--help"]
