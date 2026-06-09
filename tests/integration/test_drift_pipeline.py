"""Integration test for Step 6 — end-to-end drift report against the live stack.

Verifies:
  1. evaluate() splits, computes drift, uploads HTML to MinIO `monitoring`.
  2. A pointer row lands in `monitoring_reports` with the right shape.
  3. The DAG file is syntactically valid (full Airflow parse comes in step 7).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


def _host_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "mlops")
    pw = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST_TEST", "localhost")
    port = os.getenv("POSTGRES_PORT_TEST", "5432")
    db = os.getenv("POSTGRES_DB", "sentiment")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


@pytest.fixture
def write_conn():
    """Transactional connection — the test rolls back so monitoring_reports stays clean."""
    conn = psycopg2.connect(_host_dsn())
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


def test_evaluate_writes_html_and_pointer(minio_client, write_conn):
    from monitoring.drift_checks import DEFAULT_BUCKET, evaluate

    df = pd.read_csv(SAMPLE_CSV)
    cut = int(len(df) * 0.8)
    ref = df.iloc[:cut][["text", "label", "rating", "source"]].reset_index(drop=True)
    cur = df.iloc[cut:][["text", "label", "rating", "source"]].reset_index(drop=True)

    run_date = date.today()
    report_type = f"data_drift_test_{run_date.isoformat()}"
    result = evaluate(
        ref, cur, write_conn, minio_client,
        run_date=run_date, report_type=report_type,
    )

    # 1. Pointer row inserted (rolled back at teardown)
    assert result["report_id"] > 0
    assert result["s3_url"].startswith(f"s3://{DEFAULT_BUCKET}/{run_date.isoformat()}/")
    assert 0.0 <= result["drift_score"] <= 1.0
    assert isinstance(result["blocked_promotion"], bool)

    with write_conn.cursor() as cur_db:
        cur_db.execute(
            "SELECT report_url, drift_score, blocked_promotion "
            "FROM monitoring_reports WHERE id=%s",
            (result["report_id"],),
        )
        row = cur_db.fetchone()
    assert row is not None
    assert row[0] == result["s3_url"]
    assert abs(row[1] - result["drift_score"]) < 1e-6

    # 2. HTML object exists in MinIO and looks like HTML
    key = result["s3_url"].split(f"s3://{DEFAULT_BUCKET}/", 1)[1]
    obj = minio_client.get_object(Bucket=DEFAULT_BUCKET, Key=key)
    body = obj["Body"].read()
    assert len(body) > 1000
    assert body.lstrip()[:30].lower().startswith((b"<!doctype html", b"<html"))
    # Cleanup so re-runs stay tidy.
    minio_client.delete_object(Bucket=DEFAULT_BUCKET, Key=key)


def test_dag_file_parses():
    dag_path = ROOT / "airflow" / "dags" / "evaluate_and_monitor.py"
    compile(dag_path.read_text(), str(dag_path), "exec")
