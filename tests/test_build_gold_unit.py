#Unit tests for the Silver -> Gold refiner (data/refine/build_gold.py).
from __future__ import annotations

import pandas as pd
import pytest

from data.ingest.ingest_reviews import (
    FEATURE_STORE_COLUMNS,
    GOLD_COLUMNS,
    LABEL_STORE_COLUMNS,
    REVIEW_ID_FIELD,
)
from data.refine.build_gold import (
    build_feature_store,
    build_gold,
    build_label_store,
    label_from_rating,
    process_review_dates,
    feature_store_path,
    label_store_path,
)
from data.refine.build_silver import refine_yelp, silver_partition_path, write_silver_partition


def _silver_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["great food", "decent spot", "awful place"],
            "text_len": [10, 11, 11],
            "rating": [5.0, 3.0, 1.0],
            "source": ["yelp", "yelp", "yelp"],
            "source_id": ["r1", "r2", "r3"],
            "restaurant": ["Cafe", "Cafe", "Cafe"],
            "location": ["KL", "KL", "KL"],
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
        }
    )


def test_label_from_rating_thresholds():
    assert label_from_rating(5) == "positive"
    assert label_from_rating(4) == "positive"
    assert label_from_rating(3) == "neutral"
    assert label_from_rating(2) == "negative"
    assert label_from_rating(1) == "negative"


def test_build_gold_derives_labels_from_rating():
    gold = build_gold(_silver_frame())
    assert list(gold.columns) == GOLD_COLUMNS
    assert list(gold["label"]) == ["positive", "neutral", "negative"]


def test_build_gold_from_silver_refiner_end_to_end():
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "business_id": ["b", "b", "b"],
            "stars": [5, 3, 1],
            "text": ["delicious coffee", "espresso was fine", "cold bitter coffee"],
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "_ingested_at": ["t"] * 3,
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Milktooth"], "city": ["Indianapolis"], "state": ["IN"]})
    silver = refine_yelp(reviews, business)
    gold = build_gold(silver)
    row = gold.iloc[0]
    assert row["rating"] == 5.0
    assert row["label"] == "positive"
    assert row["restaurant"] == "Milktooth"


def test_feature_and_label_stores_keyed_by_review_id():
    silver = _silver_frame()
    features = build_feature_store(silver, "2020-01-01")
    labels = build_label_store(silver, "2020-01-01")
    assert list(features.columns) == FEATURE_STORE_COLUMNS
    assert list(labels.columns) == LABEL_STORE_COLUMNS
    assert list(features[REVIEW_ID_FIELD]) == ["r1", "r2", "r3"]
    assert list(labels[REVIEW_ID_FIELD]) == ["r1", "r2", "r3"]
    assert list(labels["label"]) == ["positive", "neutral", "negative"]


def test_process_review_dates_writes_partitions(tmp_path):
    silver_root = tmp_path / "silver" / "reviews"
    gold_root = tmp_path / "gold"
    silver = _silver_frame()
    write_silver_partition(
        silver.assign(_ingested_at="t"),
        silver_partition_path(silver_root, "2020-01-01"),
    )
    process_review_dates(silver_root, gold_root, {"2020-01-01"})
    assert feature_store_path(gold_root, "2020-01-01").exists()
    assert label_store_path(gold_root, "2020-01-01").exists()


def test_gold_passes_great_expectations_gate():
    pytest.importorskip("great_expectations")
    from data.expectations.reviews_suite import validate_reviews

    gold = build_gold(_silver_frame())
    result = validate_reviews(gold, check_language=False)
    assert result.success, result.failures
