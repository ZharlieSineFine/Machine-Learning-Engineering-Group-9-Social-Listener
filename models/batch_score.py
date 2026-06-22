"""Batch inference — score recent reviews and log to Postgres.

Called by the ``shadow_score`` Airflow DAG on a 6-hour cycle. Loads Production
and optional Staging models, predicts on reviews ingested since the last
lookback window, and writes rows to ``predictions``.

CLI:
    python -m models.batch_score
    python -m models.batch_score --lookback-hours 12

Owner: Amelia (+ Charlie/Ha for DAG wiring).
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

import pandas as pd

from models.inference import (
    ModelSet,
    PredictionResult,
    SentimentModel,
    load_models,
    predict_with_scores,
)
from models.metrics import INFERENCE_BATCH_SIZE
from models.prediction_log import PredictionRow, write_predictions


def _default_dsn() -> str:
    user = os.environ["POSTGRES_USER"]
    pw = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def fetch_recent_reviews(
    conn,
    *,
    lookback_hours: int = 6,
) -> pd.DataFrame:
    """Return reviews ingested within the lookback window."""
    query = """
        SELECT id AS review_id, text
        FROM reviews
        WHERE ingested_at >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY ingested_at
    """
    return pd.read_sql(query, conn, params=(lookback_hours,))


def filter_unscored(
    reviews: pd.DataFrame,
    conn,
    model_names: Sequence[str],
) -> pd.DataFrame:
    """Drop reviews that already have predictions for every requested model."""
    if reviews.empty or not model_names:
        return reviews

    placeholders = ", ".join(["%s"] * len(model_names))
    query = f"""
        SELECT review_id
        FROM predictions
        WHERE review_id IS NOT NULL
          AND model_name IN ({placeholders})
        GROUP BY review_id
        HAVING COUNT(DISTINCT model_name) >= %s
    """
    with conn.cursor() as cur:
        cur.execute(query, list(model_names) + [len(model_names)])
        scored_ids = {row[0] for row in cur.fetchall()}

    if not scored_ids:
        return reviews
    return reviews[~reviews["review_id"].isin(scored_ids)].reset_index(drop=True)


def _score_with_model(
    model: SentimentModel,
    reviews: pd.DataFrame,
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    texts = reviews["text"].astype(str).tolist()
    review_ids = reviews["review_id"].tolist()

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_ids = review_ids[start : start + batch_size]
        results: list[PredictionResult] = predict_with_scores(model.model, batch_texts)
        for review_id, text, result in zip(batch_ids, batch_texts, results):
            rows.append(
                PredictionRow(
                    review_id=int(review_id),
                    text=text,
                    predicted_label=result.label,
                    model_name=model.model_name,
                    model_version=model.model_version,
                    stage=model.stage,
                    score=result.score,
                )
            )
    return rows


def run_batch_score(
    dsn: Optional[str] = None,
    *,
    lookback_hours: int = 6,
    models: Optional[ModelSet] = None,
) -> dict:
    """Score pending reviews with Production (+ Staging when available)."""
    import psycopg2

    model_set = models or load_models()
    if model_set.production is None:
        raise RuntimeError("No Production model available for batch scoring")

    lanes = [model_set.production]
    if model_set.shadow is not None:
        lanes.append(model_set.shadow)

    dsn = dsn or _default_dsn()
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        reviews = fetch_recent_reviews(conn, lookback_hours=lookback_hours)
        model_names = [lane.model_name for lane in lanes]
        pending = filter_unscored(reviews, conn, model_names)

        all_rows: list[PredictionRow] = []
        for lane in lanes:
            all_rows.extend(_score_with_model(lane, pending))

        written = write_predictions(conn, all_rows)
        conn.commit()
        return {
            "reviews_fetched": int(len(reviews)),
            "reviews_scored": int(len(pending)),
            "predictions_written": written,
            "production_model": model_set.production.model_name,
            "shadow_model": None if model_set.shadow is None else model_set.shadow.model_name,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-score recent reviews into predictions")
    ap.add_argument("--lookback-hours", type=int, default=6)
    ap.add_argument("--dsn", default=None, help="Postgres DSN (defaults to env vars)")
    args = ap.parse_args()

    summary = run_batch_score(dsn=args.dsn, lookback_hours=args.lookback_hours)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
