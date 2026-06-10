"""Integration test — the GE gate prevents poisoned batches from reaching Postgres.

Builds a deliberately bad CSV, runs `ingest()`, asserts that:
  1. The call raises ValueError (from validate_reviews.raise_for_status).
  2. The reviews table row count did NOT change (no partial write).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from data.expectations.reviews_suite import REQUIRED_COLUMNS
from data.ingest.ingest_reviews import ingest


def _host_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "mlops")
    pw = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST_TEST", "localhost")
    port = os.getenv("POSTGRES_PORT_TEST", "5432")
    db = os.getenv("POSTGRES_DB", "sentiment")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _count(pg_conn) -> int:
    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        return cur.fetchone()[0]


def _write_csv(tmp_path: Path, df: pd.DataFrame) -> Path:
    p = tmp_path / "poisoned.csv"
    df.to_csv(p, index=False)
    return p


def test_poisoned_batch_blocked_before_db_write(tmp_path, pg_conn):
    """A CSV that survives soft-cleaning but violates GE must NOT touch the DB.

    We craft rows that pass `load_and_validate` (well-formed, valid labels)
    but fail the GE rating-range expectation (rating=99). The DAG must
    fail-fast without truncating or inserting.
    """
    # Seed the table so we can detect any unintended truncate/insert.
    ingest(
        Path(__file__).resolve().parents[2] / "data" / "sample" / "reviews_sample.csv",
        _host_dsn(),
        truncate=True,
    )
    baseline = _count(pg_conn)
    assert baseline > 0

    poisoned = pd.DataFrame(
        [
            {"text": "ok", "label": "positive", "rating": 5,  "source": "google",
             "restaurant": "R", "location": "KL"},
            {"text": "bad", "label": "negative", "rating": 99, "source": "google",
             "restaurant": "R", "location": "KL"},
        ],
        columns=REQUIRED_COLUMNS,
    )
    csv_path = _write_csv(tmp_path, poisoned)

    with pytest.raises(ValueError, match="reviews_suite validation failed"):
        ingest(csv_path, _host_dsn(), truncate=True)

    # The table must be unchanged — no truncate, no partial insert.
    assert _count(pg_conn) == baseline


def test_good_batch_still_writes(tmp_path, pg_conn):
    """Sanity: GE in the path doesn't break the happy case.

    Batch must include all 3 labels (cardinality check) and texts long
    enough to pass the length gate.
    """
    good = pd.DataFrame(
        [
            {"text": "great food and friendly service", "label": "positive", "rating": 5,
             "source": "google", "restaurant": "R", "location": "KL"},
            {"text": "an average meal nothing special", "label": "neutral",  "rating": 3,
             "source": "google", "restaurant": "R", "location": "KL"},
            {"text": "terrible service we waited forever", "label": "negative", "rating": 1,
             "source": "google", "restaurant": "R", "location": "KL"},
        ],
        columns=REQUIRED_COLUMNS,
    )
    csv_path = _write_csv(tmp_path, good)
    n = ingest(csv_path, _host_dsn(), truncate=True)
    assert n == 3
    assert _count(pg_conn) == 3
