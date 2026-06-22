"""Shadow deploy — run Production + Staging and log to Postgres.

Online inference calls ``predict_with_shadow`` so every ``/predict`` request
returns the Production label while optionally scoring and logging a Staging
candidate alongside it.

Owner: Amelia.
"""
from __future__ import annotations

import os
from typing import Optional, Sequence

from models.inference import ModelSet, PredictionResult, SentimentModel, predict_with_scores

from models.prediction_log import PredictionRow, write_predictions


def postgres_dsn() -> Optional[str]:
    """Build a DSN when Postgres env vars are present."""
    required = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
    if not all(os.getenv(k) for k in required):
        return None
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    return (
        f"postgresql://{os.environ['POSTGRES_USER']}:"
        f"{os.environ['POSTGRES_PASSWORD']}@"
        f"{host}:{port}/{os.environ['POSTGRES_DB']}"
    )


def _should_log_predictions() -> bool:
    if os.getenv("LOG_PREDICTIONS", "1").lower() in {"0", "false", "no"}:
        return False
    return postgres_dsn() is not None


def _rows_for_lane(
    lane: SentimentModel,
    texts: Sequence[str],
    results: Sequence[PredictionResult],
    review_ids: Sequence[Optional[int]],
) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    for text, result, review_id in zip(texts, results, review_ids):
        rows.append(
            PredictionRow(
                review_id=review_id,
                text=text,
                predicted_label=result.label,
                model_name=lane.model_name,
                model_version=lane.model_version,
                stage=lane.stage,
                score=result.score,
            )
        )
    return rows


def predict_with_shadow(
    models: ModelSet,
    texts: Sequence[str],
    *,
    review_ids: Optional[Sequence[Optional[int]]] = None,
) -> list[str]:
    """Return Production labels; log Production + Staging rows when configured."""
    if models.production is None:
        raise RuntimeError("Production model not loaded")

    review_ids = list(review_ids or [None] * len(texts))
    if len(review_ids) != len(texts):
        raise ValueError("review_ids must match texts length")

    prod_results = predict_with_scores(models.production.model, texts)
    log_rows = _rows_for_lane(models.production, texts, prod_results, review_ids)

    if models.shadow is not None:
        shadow_results = predict_with_scores(models.shadow.model, texts)
        log_rows.extend(
            _rows_for_lane(models.shadow, texts, shadow_results, review_ids)
        )

    if _should_log_predictions():
        import psycopg2

        dsn = postgres_dsn()
        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        try:
            write_predictions(conn, log_rows)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    return [r.label for r in prod_results]


def predict_one_with_shadow(
    models: ModelSet,
    text: str,
    *,
    review_id: Optional[int] = None,
) -> str:
    labels = predict_with_shadow(models, [text], review_ids=[review_id])
    return labels[0]
