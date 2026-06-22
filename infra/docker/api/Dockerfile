FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./

# Sklearn pipeline pickles reference models.baseline_sklearn; inference modules
# are shared with batch scoring.
COPY models/__init__.py models/baseline_sklearn.py models/inference.py \
     models/prediction_log.py models/metrics.py models/splits.py ./models/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
