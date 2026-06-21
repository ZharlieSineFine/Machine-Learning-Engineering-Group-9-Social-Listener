"""Integration tests for the MinIO + Postgres storage layer.

These hit real services, so they **skip** unless Postgres / MinIO are reachable. Bring them
up first:

    docker compose up -d postgres minio minio-init

Host-side defaults (localhost + the .env.example creds) are used unless overridden by env.
Each test writes under a throwaway key/source and cleans up after itself.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from data.storage import objectstore, warehouse
from data.storage.config import DATASETS_BUCKET, PostgresConfig, S3Config

_TEST_SOURCE = "_pytest_itest"

_LOCAL_DEFAULTS = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "mlops",
    "POSTGRES_PASSWORD": "mlops",
    "POSTGRES_DB": "sentiment",
    "AWS_ACCESS_KEY_ID": "minioadmin",
    "AWS_SECRET_ACCESS_KEY": "minioadmin",
    "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
}


def _env():
    # Real env wins; otherwise fall back to the local docker-compose defaults.
    return {**_LOCAL_DEFAULTS, **os.environ}


def _pg_or_skip() -> PostgresConfig:
    cfg = PostgresConfig.from_env(_env())
    if cfg is None:
        pytest.skip("Postgres env not configured")
    try:
        import psycopg2

        conn = psycopg2.connect(cfg.dsn, connect_timeout=2)
        conn.close()
    except Exception as exc:  # noqa: BLE001 - any connect failure means "not available"
        pytest.skip(f"Postgres not reachable: {exc}")
    return cfg


def _s3_or_skip() -> S3Config:
    cfg = S3Config.from_env(_env())
    if cfg is None:
        pytest.skip("MinIO env not configured")
    try:
        objectstore.make_client(cfg).list_buckets()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MinIO not reachable: {exc}")
    return cfg


def test_warehouse_silver_gold_roundtrip_is_idempotent():
    cfg = _pg_or_skip()
    silver = pd.DataFrame({
        "source": [_TEST_SOURCE], "source_id": ["pt1"], "text": ["integration latte"],
        "text_len": [17], "rating": [5.0], "restaurant": ["BrewLeaf"], "location": ["KL"],
        "date": ["2026-06-21"], "_ingested_at": ["2026-06-21T00:00:00Z"],
    })
    gold = pd.DataFrame({"review_id": ["pt1"], "review_date": ["2026-06-21"],
                         "text": ["integration latte"], "label": ["positive"]})
    try:
        with warehouse.connection(cfg) as conn:
            warehouse.ensure_schema(conn)
            assert warehouse.upsert_silver(conn, silver) == 1
            assert warehouse.upsert_gold(conn, gold) == 1
        # Re-publish the same rows: upsert, not duplicate.
        with warehouse.connection(cfg) as conn:
            warehouse.upsert_silver(conn, silver)
            warehouse.upsert_gold(conn, gold)
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM reviews_silver WHERE source = %s", (_TEST_SOURCE,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT label FROM reviews_gold WHERE review_id = 'pt1'")
            assert cur.fetchone()[0] == "positive"
    finally:
        with warehouse.connection(cfg) as conn:
            cur = conn.cursor()
            cur.execute("DELETE FROM reviews_silver WHERE source = %s", (_TEST_SOURCE,))
            cur.execute("DELETE FROM reviews_gold WHERE review_id = 'pt1'")


def test_objectstore_put_list_roundtrip():
    cfg = _s3_or_skip()
    s3 = objectstore.make_client(cfg)
    objectstore.ensure_bucket(s3)
    key = f"{_TEST_SOURCE}/probe.txt"
    try:
        s3.put_object(Bucket=DATASETS_BUCKET, Key=key, Body=b"probe")
        assert key in objectstore.list_keys(s3, _TEST_SOURCE)
    finally:
        s3.delete_object(Bucket=DATASETS_BUCKET, Key=key)
