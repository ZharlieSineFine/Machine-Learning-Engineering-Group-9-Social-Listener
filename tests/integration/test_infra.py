"""Step 1 — infrastructure bootstrap test.

Verifies that after `docker compose up -d postgres minio minio-init mlflow`:
  * Postgres has the application tables (`reviews`, `predictions`,
    `monitoring_reports`) in the `sentiment` database.
  * Postgres has the side databases (`airflow`, `mlflow`).
  * MinIO has the expected buckets (`mlflow`, `monitoring`, `datasets`).
  * The `reviews.label` CHECK constraint actually rejects bad labels.
"""
from __future__ import annotations

import psycopg2
import pytest

EXPECTED_TABLES = {"reviews", "predictions", "monitoring_reports"}
EXPECTED_DATABASES = {"sentiment", "airflow", "mlflow"}
EXPECTED_BUCKETS = {"mlflow", "monitoring", "datasets"}


def test_postgres_application_tables_exist(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public'"
        )
        tables = {row[0] for row in cur.fetchall()}
    missing = EXPECTED_TABLES - tables
    assert not missing, f"Missing tables in sentiment DB: {missing}"


def test_postgres_side_databases_exist(pg_conn):
    with pg_conn.cursor() as cur:
        cur.execute("SELECT datname FROM pg_database")
        dbs = {row[0] for row in cur.fetchall()}
    missing = EXPECTED_DATABASES - dbs
    assert not missing, f"Missing databases: {missing}"


def test_reviews_label_check_constraint(pg_conn):
    """Bad label values must be rejected at the DB layer (defence in depth)."""
    pg_conn.rollback()  # clear any prior failed transaction
    with pg_conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute(
                "INSERT INTO reviews (text, label, source) VALUES (%s, %s, %s)",
                ("test", "garbage_label", "test"),
            )
    pg_conn.rollback()


def test_minio_buckets_exist(minio_client):
    buckets = {b["Name"] for b in minio_client.list_buckets()["Buckets"]}
    missing = EXPECTED_BUCKETS - buckets
    assert not missing, f"Missing MinIO buckets: {missing}"
