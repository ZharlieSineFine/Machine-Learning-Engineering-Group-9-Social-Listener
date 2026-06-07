"""Unit tests for the daily incremental driver (data/run_daily.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from data.ingest.yelp_loader import write_bronze_partition
from data.run_daily import DailyRunError, run_daily, validate_silver_partitions
from data.refine.build_silver import read_silver_partition, silver_partition_path
from data.refine.build_gold import feature_store_path, label_store_path


def _seed_yelp_bronze(bronze_root, ingestion_date: str) -> None:
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2"],
            "business_id": ["b", "b"],
            "stars": [5, 2],
            "text": ["wonderful bubble tea experience here", "terrible bitter drink"],
            "date": ["2016-03-09 10:00:00", "2016-03-09 11:00:00"],
            "_ingested_at": [f"{ingestion_date}T00:00:00Z"] * 2,
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Boba House"], "city": ["KL"]})
    write_bronze_partition(reviews, business, bronze_root / "yelp", ingestion_date)


def test_run_daily_end_to_end_with_ge_gate(tmp_path):
    pytest.importorskip("great_expectations")
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver" / "reviews"
    gold_root = tmp_path / "gold"
    _seed_yelp_bronze(bronze_root, "2026-06-06")

    summary = run_daily(
        "2026-06-06",
        ["yelp"],
        bronze_root=bronze_root,
        silver_root=silver_root,
        gold_root=gold_root,
        skip_bronze=True,
    )
    assert summary["review_dates"] == ["2016-03-09"]
    assert summary["silver_row_counts"]["2016-03-09"] == 2
    assert feature_store_path(gold_root, "2016-03-09").exists()
    assert label_store_path(gold_root, "2016-03-09").exists()


def test_run_daily_idempotent_rerun(tmp_path):
    pytest.importorskip("great_expectations")
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver" / "reviews"
    gold_root = tmp_path / "gold"
    _seed_yelp_bronze(bronze_root, "2026-06-06")

    run_daily("2026-06-06", ["yelp"], bronze_root=bronze_root, silver_root=silver_root, gold_root=gold_root, skip_bronze=True)
    run_daily("2026-06-06", ["yelp"], bronze_root=bronze_root, silver_root=silver_root, gold_root=gold_root, skip_bronze=True)

    silver = read_silver_partition(silver_partition_path(silver_root, "2016-03-09"))
    assert len(silver) == 2


def test_ge_gate_fails_on_bad_silver(tmp_path):
    pytest.importorskip("great_expectations")
    silver_root = tmp_path / "silver" / "reviews"
    bad = pd.DataFrame(
        {
            "text": [""],
            "rating": [5.0],
            "source": ["yelp"],
            "source_id": ["r1"],
            "restaurant": ["Cafe"],
            "location": ["KL"],
            "date": ["2020-01-01"],
            "_ingested_at": ["t"],
        }
    )
    from data.refine.build_silver import write_silver_partition

    write_silver_partition(bad, silver_partition_path(silver_root, "2020-01-01"))

    with pytest.raises(DailyRunError):
        validate_silver_partitions(silver_root, ["2020-01-01"])
