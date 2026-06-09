FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./

# The sklearn pipeline pickles `models.baseline_sklearn._preprocess_text` as the
# TfidfVectorizer preprocessor — this module must exist at load time.
COPY models/__init__.py models/baseline_sklearn.py ./models/
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
