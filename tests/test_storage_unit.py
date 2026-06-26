"""Unit tests for the storage layer (config, warehouse helpers, objectstore, publish reads).

Pure / mocked only — no Postgres or MinIO required. The live round-trip is covered by
tests/test_storage_integration.py (skipped unless the services are reachable).
"""
from __future__ import annotations

import pandas as pd

from data.storage.config import PostgresConfig, S3Config
from data.storage.objectstore import to_key
from data.storage.warehouse import (
    GOLD_TABLE_COLUMNS,
    SILVER_TABLE_COLUMNS,
    _records,
    _silver_table_frame,
    upsert_silver,
)
import data.publish as publish

def test_postgres_config_from_env_and_dsn():
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d",
           "POSTGRES_HOST": "h", "POSTGRES_PORT": "5433"}
    cfg = PostgresConfig.from_env(env)
    assert cfg.dsn == "postgresql://u:p@h:5433/d"
    assert PostgresConfig.from_env({"POSTGRES_USER": "u"}) is None  # incomplete -> None


def test_postgres_config_defaults_host_port():
    cfg = PostgresConfig.from_env({"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"})
    assert cfg.host == "localhost" and cfg.port == "5432"


def test_s3_config_from_env():
    env = {"AWS_ACCESS_KEY_ID": "a", "AWS_SECRET_ACCESS_KEY": "s",
           "MLFLOW_S3_ENDPOINT_URL": "http://minio:9000"}
    cfg = S3Config.from_env(env)
    assert cfg.endpoint_url == "http://minio:9000" and cfg.access_key == "a"
    assert S3Config.from_env({"AWS_ACCESS_KEY_ID": "a"}) is None


def test_silver_table_frame_renames_to_table_columns():
    df = pd.DataFrame({"source": ["y"], "source_id": ["r"], "text": ["t"], "text_len": [1],
                       "rating": [5.0], "restaurant": ["a"], "location": ["k"],
                       "date": ["2020-01-01"], "_ingested_at": ["t0"]})
    out = _silver_table_frame(df)
    assert "review_date" in out.columns and "ingested_at" in out.columns
    assert set(SILVER_TABLE_COLUMNS).issubset(out.columns)


def test_records_coerces_nan_to_none():
    df = pd.DataFrame({"a": [1.0, float("nan")], "b": ["x", None]})
    recs = _records(df, ["a", "b"])
    assert recs[0] == (1.0, "x")
    assert recs[1][0] is None and recs[1][1] is None


def test_upsert_silver_builds_conflict_sql(monkeypatch):
    captured = {}

    def fake_execute_values(cur, sql, rows, page_size=100):
        captured["sql"] = sql
        captured["rows"] = list(rows)

    import psycopg2.extras
    monkeypatch.setattr(psycopg2.extras, "execute_values", fake_execute_values)

    class _Cur:
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False
        def execute(self, *a):
            pass

    class _Conn:
        def cursor(self):
            return _Cur()

    df = pd.DataFrame({"source": ["yelp"], "source_id": ["r1"], "text": ["hi there"], "text_len": [8],
                       "rating": [5.0], "restaurant": ["A"], "location": ["KL"],
                       "date": ["2020-01-01"], "_ingested_at": ["t"]})
    n = upsert_silver(_Conn(), df)
    assert n == 1
    assert "reviews_silver" in captured["sql"]
    assert "ON CONFLICT (source, source_id) DO UPDATE" in captured["sql"]
    # date -> review_date and _ingested_at -> ingested_at, in table-column order
    assert captured["rows"][0] == ("yelp", "r1", "hi there", 8, 5.0, "A", "KL", "2020-01-01", "t")


def test_table_column_constants_are_stable():
    assert SILVER_TABLE_COLUMNS == ["source", "source_id", "text", "text_len", "rating",
                                    "restaurant", "location", "review_date", "ingested_at"]
    assert GOLD_TABLE_COLUMNS == ["review_id", "review_date", "text", "label", "label_source", "text_len"]


def test_to_key_joins_forward_slashes():
    assert to_key("silver/reviews", "review_date=2020-01-01", "part.parquet") == \
        "silver/reviews/review_date=2020-01-01/part.parquet"
    assert to_key("bronze", "", "x.csv") == "bronze/x.csv"


def test_read_silver_concats_partitions(tmp_path):
    p = tmp_path / "review_date=2020-01-01"
    p.mkdir(parents=True)
    pd.DataFrame({"text": ["x"], "text_len": [1], "rating": [5.0], "source": ["yelp"],
                  "source_id": ["r1"], "restaurant": ["A"], "location": ["KL"],
                  "date": ["2020-01-01"], "_ingested_at": ["t"]}).to_parquet(p / "part.parquet")
    s = publish.read_silver(tmp_path)
    assert len(s) == 1 and s.iloc[0]["source_id"] == "r1"


def test_read_gold_merges_feature_and_label(tmp_path):
    fs = tmp_path / "feature_store" / "review_date=2020-01-01"
    fs.mkdir(parents=True)
    pd.DataFrame({"review_id": ["r1"], "review_date": ["2020-01-01"], "text": ["good"]}).to_parquet(fs / "part.parquet")
    ls = tmp_path / "label_store" / "review_date=2020-01-01"
    ls.mkdir(parents=True)
    pd.DataFrame({"review_id": ["r1"], "review_date": ["2020-01-01"], "label": ["positive"]}).to_parquet(ls / "part.parquet")
    g = publish.read_gold(tmp_path)
    assert len(g) == 1
    row = g.iloc[0]
    assert row["review_id"] == "r1" and row["label"] == "positive" and row["text"] == "good"
    assert row["text_len"] == 4 and row["label_source"] == "derived_from_rating"


def test_publish_run_is_noop_without_env():
    # No POSTGRES_*/AWS_* in the supplied env -> publish_run degrades to a safe no-op
    # (a daily run without services configured still succeeds).
    out = publish.publish_run(["2020-01-01"], "2026-06-21", env={})
    assert out == {"published_minio": None, "published_postgres": None}


class _FakeS3:
    """Minimal stand-in for a boto3 S3 client: records uploads, fakes bucket existence."""

    def __init__(self):
        self.uploaded = []
        self.buckets = set()

    def head_bucket(self, Bucket):
        from botocore.exceptions import ClientError
        if Bucket not in self.buckets:
            raise ClientError({"Error": {"Code": "404"}}, "HeadBucket")

    def create_bucket(self, Bucket):
        self.buckets.add(Bucket)

    def upload_file(self, filename, bucket, key):
        self.uploaded.append((bucket, key))


def test_mirror_tree_uploads_relpaths_under_prefix(tmp_path):
    from data.storage.objectstore import mirror_tree
    part = tmp_path / "review_date=2020-01-01"
    part.mkdir()
    (part / "part.parquet").write_bytes(b"x")
    fake = _FakeS3()
    n = mirror_tree(fake, tmp_path, "silver/reviews", bucket="datasets")
    assert n == 1
    assert fake.uploaded == [("datasets", "silver/reviews/review_date=2020-01-01/part.parquet")]


def test_publish_objects_mirrors_each_layer_with_prefix(tmp_path):
    bronze = tmp_path / "bronze" / "yelp"
    bronze.mkdir(parents=True)
    (bronze / "reviews.csv").write_bytes(b"a")
    silver = tmp_path / "silver" / "review_date=2020-01-01"
    silver.mkdir(parents=True)
    (silver / "part.parquet").write_bytes(b"b")
    gold = tmp_path / "gold" / "feature_store" / "review_date=2020-01-01"
    gold.mkdir(parents=True)
    (gold / "part.parquet").write_bytes(b"c")

    fake = _FakeS3()
    written = publish.publish_objects(
        fake, bronze_root=tmp_path / "bronze", silver_root=tmp_path / "silver", gold_root=tmp_path / "gold"
    )
    assert written == {"bronze": 1, "silver": 1, "gold": 1}
    assert "datasets" in fake.buckets  # ensure_bucket created it
    keys = [k for _, k in fake.uploaded]
    assert "bronze/yelp/reviews.csv" in keys
    assert "silver/reviews/review_date=2020-01-01/part.parquet" in keys
    assert "gold/feature_store/review_date=2020-01-01/part.parquet" in keys
