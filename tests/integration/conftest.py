"""Fixtures + connection helpers for integration tests.

Integration tests require the compose stack (or at least postgres + minio +
mlflow) to be running. Bring it up with `scripts/up.sh` or:
    docker compose up -d postgres minio minio-init mlflow

If services aren't reachable, the fixtures fail with a clear message rather
than silently skipping.
"""
from __future__ import annotations

import os
from typing import Iterator

import boto3
import psycopg2
import pytest
from botocore.client import Config

POSTGRES_HOST = os.getenv("POSTGRES_HOST_TEST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT_TEST", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "mlops")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mlops")
POSTGRES_DB = os.getenv("POSTGRES_DB", "sentiment")

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT_TEST", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")


def _connect_postgres():
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB,
            connect_timeout=3,
        )
        # Autocommit so read-only SELECTs from tests don't hold a tx that
        # blocks TRUNCATEs done by ingest() on a separate connection.
        # Tests that need transactional behaviour manage commits explicitly.
        conn.autocommit = True
        return conn
    except psycopg2.OperationalError as exc:
        pytest.fail(
            f"Cannot reach Postgres at {POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}.\n"
            f"Start the stack first: `scripts/up.sh` or "
            f"`docker compose up -d postgres`.\nUnderlying error: {exc}"
        )


def _connect_minio():
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4", connect_timeout=3, retries={"max_attempts": 1}),
        region_name="us-east-1",
    )
    try:
        client.list_buckets()
    except Exception as exc:
        pytest.fail(
            f"Cannot reach MinIO at {MINIO_ENDPOINT}.\n"
            f"Start the stack first: `scripts/up.sh` or "
            f"`docker compose up -d minio minio-init`.\nUnderlying error: {exc}"
        )
    return client


@pytest.fixture(scope="session")
def pg_conn() -> Iterator:
    conn = _connect_postgres()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="session")
def minio_client():
    return _connect_minio()
