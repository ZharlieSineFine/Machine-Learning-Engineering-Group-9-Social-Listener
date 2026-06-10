"""Integration test for Step 2 — ingestion DAG against real Postgres.

Calls the pure `ingest()` function (the DAG is a one-line wrapper around it),
verifies row count, schema, label values, and idempotency on a second run.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from data.ingest.ingest_reviews import VALID_LABELS, ingest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


def _host_dsn() -> str:
    """DSN that works from the test runner on the host (not inside docker)."""
    user = os.getenv("POSTGRES_USER", "mlops")
    pw = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST_TEST", "localhost")
    port = os.getenv("POSTGRES_PORT_TEST", "5432")
    db = os.getenv("POSTGRES_DB", "sentiment")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


@pytest.fixture(scope="module")
def dsn() -> str:
    return _host_dsn()


def test_ingest_writes_expected_row_count(dsn, pg_conn):
    n = ingest(SAMPLE_CSV, dsn, truncate=True)
    assert n > 100, "ingest should write a non-trivial number of rows"

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        count = cur.fetchone()[0]
    assert count == n


def test_ingest_row_shape(dsn, pg_conn):
    ingest(SAMPLE_CSV, dsn, truncate=True)
    with pg_conn.cursor() as cur:
        cur.execute("SELECT text, label, rating, source FROM reviews LIMIT 5")
        rows = cur.fetchall()
    assert len(rows) == 5
    for text, label, rating, source in rows:
        assert isinstance(text, str) and text
        assert label in VALID_LABELS
        assert rating is None or 1.0 <= rating <= 5.0
        assert source in {"google", "tripadvisor"}


def test_ingest_is_idempotent(dsn, pg_conn):
    """Running twice with truncate=True yields the same row count, not double."""
    n1 = ingest(SAMPLE_CSV, dsn, truncate=True)
    n2 = ingest(SAMPLE_CSV, dsn, truncate=True)
    assert n1 == n2

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        count = cur.fetchone()[0]
    assert count == n1


def test_dag_file_parses():
    """The Airflow DAG file must be syntactically valid Python.

    We don't import it (that would require Airflow on the test host) — we
    just compile it. CI runs a full Airflow DagBag parse in step 7.
    """
    dag_path = ROOT / "airflow" / "dags" / "ingest_reviews.py"
    src = dag_path.read_text()
    compile(src, str(dag_path), "exec")
