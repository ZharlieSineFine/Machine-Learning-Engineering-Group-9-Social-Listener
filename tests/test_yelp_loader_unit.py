#Unit tests for the RAW Yelp Bronze adapter (data/ingest/yelp_loader.py).
from __future__ import annotations

import pandas as pd

from data.ingest.yelp_loader import (
    BUSINESS_BRONZE_COLUMNS,
    REVIEW_BRONZE_COLUMNS,
    bronze_partition_dir,
    build_business_index,
    collect_reviews,
    extract_bronze_from_records,
    write_bronze_partition,
)

BIZ = [
    {"business_id": "b_cafe", "name": "Bean There", "city": "Austin", "state": "TX",
     "categories": "Coffee & Tea, Cafes", "review_count": 12},
    {"business_id": "b_gym", "name": "Iron House", "city": "Austin", "state": "TX",
     "categories": "Gyms, Fitness", "review_count": 3},
]
REV = [
    {"business_id": "b_cafe", "stars": 5, "text": "amazing latte", "date": "2016-03-09 10:00:00", "review_id": "r1"},
    {"business_id": "b_gym", "stars": 1, "text": "broken machines", "date": "2017-01-02 09:00:00", "review_id": "r2"},
    {"business_id": "b_cafe", "stars": 3, "text": "ok flat white", "review_id": "r3"},  # no date key
]


def test_bronze_columns_and_provenance():
    reviews, business = extract_bronze_from_records(REV, BIZ, ingested_at="2026-06-06T00:00:00Z")
    assert list(reviews.columns) == REVIEW_BRONZE_COLUMNS
    assert list(business.columns) == BUSINESS_BRONZE_COLUMNS
    # Bronze must NOT carry any Silver-derived columns.
    assert "label" not in reviews.columns
    assert "restaurant" not in reviews.columns
    assert (reviews["_source"] == "yelp").all()
    assert (reviews["_ingested_at"] == "2026-06-06T00:00:00Z").all()
    assert (business["_ingested_at"] == "2026-06-06T00:00:00Z").all()


def test_date_is_passed_through_verbatim():
    reviews, _ = extract_bronze_from_records(REV, BIZ, ingested_at="t")
    cafe = reviews[reviews["review_id"] == "r1"].iloc[0]
    assert cafe["date"] == "2016-03-09 10:00:00"  # exact source string, not reformatted
    no_date = reviews[reviews["review_id"] == "r3"].iloc[0]
    assert pd.isna(no_date["date"])  # missing -> null, never invented


def test_category_scope_selects_businesses_and_their_reviews():
    reviews, business = extract_bronze_from_records(REV, BIZ, ingested_at="t")
    # Only the café business is in scope, so only its reviews survive (gym review dropped).
    assert set(business["business_id"]) == {"b_cafe"}
    assert set(reviews["business_id"]) == {"b_cafe"}
    assert set(reviews["review_id"]) == {"r1", "r3"}


def test_raw_rows_are_not_cleaned():
    # A null-text review for an in-scope business is KEPT in Bronze (cleaning is Silver's job).
    recs = [{"business_id": "b_cafe", "stars": 4, "text": None, "date": "2016-01-01", "review_id": "rx"}]
    reviews, _ = extract_bronze_from_records(recs, BIZ, ingested_at="t")
    assert len(reviews) == 1
    assert pd.isna(reviews.iloc[0]["text"])
    assert reviews.iloc[0]["stars"] == 4  # verbatim, not coerced to float/string


def test_build_business_index_returns_ids_and_raw_rows():
    ids, rows = build_business_index(BIZ, {"Coffee & Tea", "Cafes"})
    assert ids == {"b_cafe"}
    assert rows[0]["name"] == "Bean There"  # raw business fields preserved


def test_collect_reviews_respects_scope_and_limit():
    ids = {"b_cafe"}
    assert len(collect_reviews(REV, ids)) == 2
    assert len(collect_reviews(REV, ids, limit=1)) == 1
    assert collect_reviews(REV, set()) == []  # nothing in scope


def test_write_bronze_partition_lands_under_dt(tmp_path):
    reviews, business = extract_bronze_from_records(REV, BIZ, ingested_at="2026-06-06T00:00:00Z")
    out_dir = tmp_path / "yelp"
    reviews_path, business_path = write_bronze_partition(reviews, business, out_dir, "2026-06-06")
    assert reviews_path == bronze_partition_dir(out_dir, "2026-06-06") / "reviews.csv"
    assert business_path.parent == reviews_path.parent
    assert reviews_path.exists() and business_path.exists()
