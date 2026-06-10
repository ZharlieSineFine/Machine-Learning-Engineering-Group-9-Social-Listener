"""End-to-end pipeline test — the marquee integration test.

Walks the full Phase 1+2 happy path against the live compose stack:

    1. INGEST     CSV -> Postgres `reviews`  (via data/ingest)
    2. TRAIN      Postgres -> sklearn model logged + registered in MLflow
    3. PROMOTE    flip the new version to stage 'Production'
    4. SERVE      boot the FastAPI app in-process, /health says 'mlflow'
    5. PREDICT    single + batch predictions return valid labels
    6. RELOAD     /reload picks up a freshly-trained version
    7. MONITOR    evaluate() runs the GE-cleaned data, no drift block

If this test goes green, every layer of the project is wired up correctly.
If CI is green, anyone cloning the repo can run `docker compose up` and
have a working sentiment system.

Pre-reqs (provided by CI or scripts/up.sh):
    postgres + minio + mlflow running
    pytest deps from tests/requirements.txt installed
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pandas as pd
import psycopg2
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


def _host_dsn() -> str:
    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'mlops')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'mlops')}@"
        f"{os.getenv('POSTGRES_HOST_TEST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT_TEST', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'sentiment')}"
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """One-time env setup for the whole pipeline."""
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5001"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"
    os.environ["MODEL_NAME"] = "sentiment-baseline"
    os.environ["MODEL_STAGE"] = "Production"
    os.environ["MLFLOW_EXPERIMENT"] = f"e2e-{uuid.uuid4().hex[:8]}"
    os.environ["ADMIN_TOKEN"] = f"e2e-{uuid.uuid4().hex[:8]}"
    return {"tmp": tmp_path_factory.mktemp("e2e")}


def test_e2e_pipeline(env, pg_conn, minio_client):
    # ---- 1. INGEST -------------------------------------------------------
    from data.ingest.ingest_reviews import ingest as ingest_reviews
    rows = ingest_reviews(SAMPLE_CSV, _host_dsn(), truncate=True)
    assert rows > 100, "ingest should write a substantive batch"

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM reviews")
        assert cur.fetchone()[0] == rows

    # ---- 2. TRAIN + REGISTER --------------------------------------------
    from models.train import run as train_run
    artifact = env["tmp"] / "baseline.pkl"
    result = train_run(data_path=SAMPLE_CSV, out_path=artifact)
    assert result.mlflow_run_id is not None
    assert result.mlflow_model_version is not None
    v1 = result.mlflow_model_version

    # ---- 3. PROMOTE -----------------------------------------------------
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    client.transition_model_version_stage(
        name=os.environ["MODEL_NAME"], version=v1,
        stage="Production", archive_existing_versions=True,
    )

    # ---- 4. SERVE -------------------------------------------------------
    for mod in ["app.main", "app.model_loader"]:
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    from app.main import app  # noqa: WPS433 — deliberate late import
    api = TestClient(app)

    h = api.get("/health").json()
    assert h["model_loaded"] is True
    assert h["model_source"] == "mlflow"

    # ---- 5. PREDICT (single + batch) ------------------------------------
    single = api.post("/predict", json={"text": "amazing food, would return"})
    assert single.status_code == 200
    assert single.json()["label"] in {"negative", "neutral", "positive"}

    batch = api.post("/predict/batch", json={"texts": [
        "amazing food and lovely service",
        "worst meal ever, complete waste of money",
        "it was okay, nothing special",
    ]})
    assert batch.status_code == 200
    labels = batch.json()["labels"]
    assert len(labels) == 3
    assert all(lbl in {"negative", "neutral", "positive"} for lbl in labels)

    # ---- 6. RELOAD picks up a NEW version -------------------------------
    result2 = train_run(data_path=SAMPLE_CSV, out_path=artifact)
    v2 = result2.mlflow_model_version
    assert v2 != v1
    client.transition_model_version_stage(
        name=os.environ["MODEL_NAME"], version=v2,
        stage="Production", archive_existing_versions=True,
    )
    r = api.post("/reload", headers={"X-Admin-Token": os.environ["ADMIN_TOKEN"]})
    assert r.status_code == 200, r.text
    assert r.json()["model_source"] == "mlflow"

    prod = client.get_latest_versions(os.environ["MODEL_NAME"], stages=["Production"])
    assert prod and prod[0].version == v2

    # ---- 7. MONITOR (drift + F1 gate) -----------------------------------
    import mlflow.sklearn
    model = mlflow.sklearn.load_model(f"models:/{os.environ['MODEL_NAME']}/Production")

    from monitoring.drift_checks import evaluate
    df = pd.read_csv(SAMPLE_CSV)[["text", "label", "rating", "source"]].dropna()
    cut = int(len(df) * 0.8)
    ref, cur = df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)

    drift_result = evaluate(
        ref, cur, pg_conn, minio_client,
        model=model,
        report_type=f"e2e_{uuid.uuid4().hex[:8]}",
        raise_on_block=False,
    )
    assert drift_result["blocked_promotion"] is False, (
        f"stable data should not block: {drift_result}"
    )

    # cleanup the report so re-runs don't pile up.
    minio_client.delete_object(
        Bucket="monitoring",
        Key=drift_result["s3_url"].split("s3://monitoring/", 1)[1],
    )
