"""Integration test for Step 10 — drift + recall_neg gate against real services.

Verifies:
    1. Happy path: stable data, real production model → not blocked, no raise.
    2. Poisoned current: relabel all rows to one class → real model's
       recall_neg drops below the 3% threshold → PromotionBlocked raised,
       but the HTML report IS still uploaded to MinIO (forensics).
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

from monitoring.drift_checks import (
    DEFAULT_BUCKET,
    PromotionBlocked,
    evaluate,
)

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
    conn = psycopg2.connect(_host_dsn())
    conn.autocommit = False
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture(scope="module")
def production_model():
    """Load the Production model registered in Step 5's integration test.

    Sets MLflow + S3 env so the artifact download from MinIO works.
    """
    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5001")
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")
    import mlflow
    import mlflow.sklearn
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        return mlflow.sklearn.load_model("models:/sentiment-baseline/Production")
    except Exception as exc:
        pytest.skip(f"sentiment-baseline/Production not registered yet: {exc}")


def _cleanup_minio_key(minio_client, s3_url: str) -> None:
    key = s3_url.split(f"s3://{DEFAULT_BUCKET}/", 1)[1]
    minio_client.delete_object(Bucket=DEFAULT_BUCKET, Key=key)


def test_gate_does_not_block_on_stable_data(production_model, minio_client, write_conn):
    df = pd.read_csv(SAMPLE_CSV)[["text", "label", "rating", "source"]].dropna()
    cut = int(len(df) * 0.8)
    ref = df.iloc[:cut].reset_index(drop=True)
    cur = df.iloc[cut:].reset_index(drop=True)

    result = evaluate(
        ref, cur, write_conn, minio_client,
        run_date=date.today(),
        report_type=f"drift_gate_stable_{os.getpid()}",
        model=production_model,
        raise_on_block=False,
    )
    assert result["reference_recall_neg"] is not None and result["reference_recall_neg"] > 0
    assert result["current_recall_neg"] is not None and result["current_recall_neg"] > 0
    assert result["blocked_promotion"] is False
    _cleanup_minio_key(minio_client, result["s3_url"])


def test_gate_blocks_on_poisoned_current(production_model, minio_client, write_conn):
    """Relabel all current rows to 'negative' so recall_neg collapses."""
    df = pd.read_csv(SAMPLE_CSV)[["text", "label", "rating", "source"]].dropna()
    cut = int(len(df) * 0.8)
    ref = df.iloc[:cut].reset_index(drop=True)
    cur = df.iloc[cut:].copy().reset_index(drop=True)
    cur["label"] = "negative"  # poison the labels — model will look terrible

    with pytest.raises(PromotionBlocked, match=r"recall_neg_drop|drift_score"):
        evaluate(
            ref, cur, write_conn, minio_client,
            run_date=date.today(),
            report_type=f"drift_gate_poisoned_{os.getpid()}",
            model=production_model,
            raise_on_block=True,
        )

    # The report must have been uploaded BEFORE the raise — find and delete it.
    listing = minio_client.list_objects_v2(
        Bucket=DEFAULT_BUCKET,
        Prefix=f"{date.today().isoformat()}/drift_gate_poisoned_{os.getpid()}",
    )
    keys = [obj["Key"] for obj in listing.get("Contents", [])]
    assert keys, "expected the report to be uploaded before the raise"
    for k in keys:
        minio_client.delete_object(Bucket=DEFAULT_BUCKET, Key=k)
