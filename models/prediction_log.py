"""Postgres prediction logging — shared by online shadow and batch scoring."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass
class PredictionRow:
    review_id: Optional[int]
    text: str
    predicted_label: str
    model_name: str
    model_version: Optional[str]
    stage: str
    score: Optional[float] = None


def write_predictions(conn, rows: Sequence[PredictionRow]) -> int:
    if not rows:
        return 0

    sql = """
        INSERT INTO predictions (
            review_id, text, predicted_label, model_name, model_version, stage, score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    payload = [
        (
            row.review_id,
            row.text,
            row.predicted_label,
            row.model_name,
            row.model_version,
            row.stage,
            row.score,
        )
        for row in rows
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, payload)
    return len(rows)
