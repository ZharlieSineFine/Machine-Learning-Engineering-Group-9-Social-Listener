"""Unit tests for the pure Yelp source adapter.

No DB, no Airflow. Validates the review.json + business.json join, the
category filter, label derivation, malformed-line handling, and that the
output survives the existing Great Expectations gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.ingest.ingest_reviews import EXPECTED_COLUMNS
from data.ingest.yelp_loader import load_yelp


def _write_jsonl(tmp: Path, name: str, rows: list[dict]) -> Path:
    p = tmp / name
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_happy_path_join(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b_cafe", "name": "Bean There", "city": "Austin", "state": "TX", "categories": "Coffee & Tea, Cafes"},
        {"business_id": "b_gym",  "name": "Iron House", "city": "Austin", "state": "TX", "categories": "Gyms, Fitness"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b_cafe", "stars": 5, "text": "amazing latte and pastries"},
        {"business_id": "b_gym",  "stars": 1, "text": "machines always broken here"},
    ])
    df = load_yelp(rev, biz)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 1                              # only the cafe review survives
    row = df.iloc[0]
    assert row["restaurant"] == "Bean There"
    assert row["location"] == "Austin"
    assert row["source"] == "yelp"
    assert row["text"] == "amazing latte and pastries"


def test_label_thresholds(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "KL", "state": "WP", "categories": "Cafes"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b", "stars": 5, "text": "excellent coffee here"},
        {"business_id": "b", "stars": 4, "text": "pretty good espresso"},
        {"business_id": "b", "stars": 3, "text": "average flat white today"},
        {"business_id": "b", "stars": 2, "text": "weak and watery coffee"},
        {"business_id": "b", "stars": 1, "text": "terrible burnt beans"},
    ])
    df = load_yelp(rev, biz)
    assert list(df["label"]) == ["positive", "positive", "neutral", "negative", "negative"]
    assert df["rating"].tolist() == [5.0, 4.0, 3.0, 2.0, 1.0]   # ints cast to float


def test_category_filter_drops_non_matching(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Steakhouse", "city": "KL", "state": "WP", "categories": "Steakhouses, Restaurants"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b", "stars": 5, "text": "great steak dinner"},
    ])
    assert len(load_yelp(rev, biz)) == 0                          # default café scope -> dropped
    assert len(load_yelp(rev, biz, categories={"Restaurants"})) == 1  # widened scope -> kept


def test_categories_str_list_none(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b_str",  "name": "StrCafe",  "city": "KL", "state": "WP", "categories": "Coffee & Tea, Food"},
        {"business_id": "b_list", "name": "ListCafe", "city": "KL", "state": "WP", "categories": ["Cafes", "Bakeries"]},
        {"business_id": "b_none", "name": "NoneCafe", "city": "KL", "state": "WP", "categories": None},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b_str",  "stars": 5, "text": "good drip coffee"},
        {"business_id": "b_list", "stars": 4, "text": "nice cafe vibe here"},
        {"business_id": "b_none", "stars": 3, "text": "no category at all"},
    ])
    df = load_yelp(rev, biz)
    assert set(df["restaurant"]) == {"StrCafe", "ListCafe"}       # None category dropped


def test_join_miss_dropped(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "KL", "state": "WP", "categories": "Cafes"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b",     "stars": 5, "text": "lovely cafe spot"},
        {"business_id": "ghost", "stars": 1, "text": "this business is unknown"},
    ])
    df = load_yelp(rev, biz)
    assert len(df) == 1
    assert df.iloc[0]["restaurant"] == "Cafe"


def test_malformed_line_skipped(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "KL", "state": "WP", "categories": "Cafes"},
    ])
    rev_path = tmp_path / "review.json"
    rev_path.write_text(
        "\n".join([
            json.dumps({"business_id": "b", "stars": 5, "text": "first valid review"}),
            "this is not json at all {{{",
            json.dumps({"business_id": "b", "stars": 1, "text": "second valid review"}),
        ]),
        encoding="utf-8",
    )
    df = load_yelp(rev_path, biz)
    assert len(df) == 2                                          # garbage line skipped


def test_limit_caps_joined_rows(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "KL", "state": "WP", "categories": "Cafes"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b", "stars": 5, "text": f"review number {i} here"} for i in range(10)
    ])
    assert len(load_yelp(rev, biz, limit=3)) == 3


def test_location_with_state(tmp_path: Path):
    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "Austin", "state": "TX", "categories": "Cafes"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b", "stars": 5, "text": "great spot in austin"},
    ])
    assert load_yelp(rev, biz, location_with_state=False).iloc[0]["location"] == "Austin"
    assert load_yelp(rev, biz, location_with_state=True).iloc[0]["location"] == "Austin, TX"


def test_validate_reviews_passes(tmp_path: Path):
    pytest.importorskip("great_expectations")
    from data.expectations.reviews_suite import validate_reviews

    biz = _write_jsonl(tmp_path, "business.json", [
        {"business_id": "b", "name": "Cafe", "city": "Austin", "state": "TX", "categories": "Cafes"},
    ])
    rev = _write_jsonl(tmp_path, "review.json", [
        {"business_id": "b", "stars": 5, "text": "absolutely delicious coffee and friendly staff"},
        {"business_id": "b", "stars": 3, "text": "the espresso was fine but nothing memorable today"},
        {"business_id": "b", "stars": 1, "text": "cold bitter coffee and very slow rude service"},
    ])
    df = load_yelp(rev, biz)
    # check_language=False: Yelp is English and langdetect isn't needed at unit level.
    result = validate_reviews(df, check_language=False)
    assert result.success, result.failures
