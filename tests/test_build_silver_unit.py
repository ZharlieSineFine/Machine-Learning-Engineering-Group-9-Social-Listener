"""Unit tests for the Bronze -> Silver refiner (data/refine/build_silver.py).

Silver = join, cleaning, ISO date normalisation, source_id, dedup. No labels (that's Gold).
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.ingest.ingest_reviews import INGESTED_AT_FIELD, SILVER_COLUMNS_WITH_DATE
from data.ingest.yelp_loader import write_bronze_partition as yelp_write_bronze
from data.refine.build_silver import (
    assign_review_date_keys,
    build_silver,
    dedup_silver,
    derive_tripadvisor_source_id,
    filter_recent_years_per_source,
    parse_review_date,
    process_ingestion_to_silver,
    read_silver_partition,
    refine_tripadvisor,
    refine_yelp,
    silver_partition_path,
)


def test_parse_review_date_absolute_and_relative():
    assert parse_review_date("Reviewed 6 February 2022") == "2022-02-06"
    assert parse_review_date("Reviewed 22 January 2022 ") == "2022-01-22"
    assert parse_review_date("6 February 2022") == "2022-02-06"
    assert parse_review_date("Reviewed 3 weeks ago") is None
    assert parse_review_date("Reviewed yesterday") is None
    assert parse_review_date("") is None
    assert parse_review_date(None) is None
    assert parse_review_date(float("nan")) is None


def test_refine_tripadvisor_maps_cleans_and_iso_dates():
    bronze = pd.DataFrame(
        {
            "Review": ["great latte here", "   ", "weak watery kopi", "nice teh tarik"],
            "Rating": [5, 4, 2, "garbage"],
            "Dates": [
                "Reviewed 6 February 2022",
                "Reviewed 1 January 2020",
                "Reviewed 10 March 2021",
                "Reviewed yesterday",
            ],
            "Restaurant": ["Bean Cafe", "Bean Cafe", "Kopi House", "Tealive KLCC"],
            "Location": ["KL", "KL", "Penang", "KL"],
            "_source": ["tripadvisor"] * 4,
            "_ingested_at": ["2026-06-06T00:00:00Z"] * 4,
        }
    )
    out = refine_tripadvisor(bronze)
    assert list(out.columns) == SILVER_COLUMNS_WITH_DATE + [INGESTED_AT_FIELD]
    assert "label" not in out.columns
    assert set(out["restaurant"]) == {"Bean Cafe", "Kopi House"}
    bean = out[out["restaurant"] == "Bean Cafe"].iloc[0]
    assert bean["rating"] == 5.0
    assert bean["date"] == "2022-02-06"
    assert bean["source_id"] == derive_tripadvisor_source_id("Bean Cafe", "great latte here", "2022-02-06")
    assert out[out["restaurant"] == "Kopi House"].iloc[0]["rating"] == 2.0


def test_refine_yelp_joins_with_source_id():
    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "business_id": ["b_cafe", "b_cafe", "ghost"],
            "stars": [5, 2, 4],
            "text": ["amazing latte", "cold bitter coffee", "unknown biz"],
            "date": ["2016-03-09 10:00:00", "2018-01-01 12:00:00", "2019-01-01 00:00:00"],
            "_ingested_at": ["t1", "t1", "t1"],
        }
    )
    business = pd.DataFrame(
        {"business_id": ["b_cafe"], "name": ["Bean There"], "city": ["Austin"], "state": ["TX"]}
    )
    out = refine_yelp(reviews, business)
    assert list(out[SILVER_COLUMNS_WITH_DATE].columns) == SILVER_COLUMNS_WITH_DATE
    assert "label" not in out.columns
    assert len(out) == 2
    first = out.iloc[0]
    assert first["restaurant"] == "Bean There"
    assert first["location"] == "Austin"
    assert first["rating"] == 5.0
    assert first["source_id"] == "r1"
    assert first["date"] == "2016-03-09 10:00:00"


def test_refine_yelp_location_with_state():
    reviews = pd.DataFrame(
        {
            "review_id": ["r1"],
            "business_id": ["b"],
            "stars": [5],
            "text": ["good"],
            "date": ["2020-01-01"],
            "_ingested_at": ["t"],
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"], "state": ["TX"]})
    assert refine_yelp(reviews, business, location_with_state=True).iloc[0]["location"] == "Austin, TX"


def test_build_silver_concatenates_sources():
    a = pd.DataFrame(
        {
            "text": ["x"],
            "text_len": [1],
            "rating": [5.0],
            "source": ["yelp"],
            "source_id": ["r1"],
            "restaurant": ["A"],
            "location": ["KL"],
            "date": ["2020-01-01"],
            INGESTED_AT_FIELD: ["t"],
        }
    )
    b = a.assign(source="tripadvisor", source_id="hash1")
    out = build_silver([a, b])
    assert len(out) == 2
    assert list(out.columns) == SILVER_COLUMNS_WITH_DATE


def test_filter_recent_years_per_source_uses_each_source_max():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e"],
            "text_len": [1, 1, 1, 1, 1],
            "rating": [5.0] * 5,
            "source": ["yelp", "yelp", "yelp", "tripadvisor", "tripadvisor"],
            "source_id": ["y1", "y2", "y3", "t1", "t2"],
            "restaurant": ["C"] * 5,
            "location": ["KL"] * 5,
            "date": [
                "2010-01-01",
                "2021-06-01",
                "2022-01-19",
                "2015-01-01",
                "2022-02-06",
            ],
            INGESTED_AT_FIELD: ["t"] * 5,
        }
    )
    out = filter_recent_years_per_source(df, 3)
    assert set(out["source_id"]) == {"y2", "y3", "t2"}


def test_process_ingestion_drops_pre_2019_yelp_with_default_recent_years(tmp_path):
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver" / "reviews"

    reviews = pd.DataFrame(
        {
            "review_id": ["old", "new"],
            "business_id": ["b", "b"],
            "stars": [4, 5],
            "text": ["stale", "fresh"],
            "date": ["2010-01-01 10:00:00", "2021-06-01 10:00:00"],
            "_ingested_at": ["2026-06-06T00:00:00Z"] * 2,
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"]})
    yelp_write_bronze(reviews, business, bronze_root / "yelp", "2026-06-06")

    affected = process_ingestion_to_silver(bronze_root, silver_root, ["2026-06-06"], ["yelp"])
    assert affected == {"2021-06-01"}
    silver = read_silver_partition(silver_partition_path(silver_root, "2021-06-01"))
    assert len(silver) == 1
    assert silver.iloc[0]["text"] == "fresh"


def test_dedup_keeps_latest_ingested_at():
    df = pd.DataFrame(
        {
            "text": ["old", "new"],
            "text_len": [3, 3],
            "rating": [3.0, 5.0],
            "source": ["yelp", "yelp"],
            "source_id": ["r1", "r1"],
            "restaurant": ["Cafe", "Cafe"],
            "location": ["KL", "KL"],
            "date": ["2020-01-01", "2020-01-01"],
            INGESTED_AT_FIELD: ["2026-06-05T00:00:00Z", "2026-06-06T00:00:00Z"],
        }
    )
    out = dedup_silver(df)
    assert len(out) == 1
    assert out.iloc[0]["text"] == "new"
    assert out.iloc[0]["rating"] == 5.0


def test_partitioned_idempotent_rerun(tmp_path):
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver" / "reviews"

    reviews = pd.DataFrame(
        {
            "review_id": ["r1"],
            "business_id": ["b"],
            "stars": [5],
            "text": ["great coffee"],
            "date": ["2016-03-09 10:00:00"],
            "_ingested_at": ["2026-06-06T00:00:00Z"],
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"]})
    yelp_write_bronze(reviews, business, bronze_root / "yelp", "2026-06-06")

    affected1 = process_ingestion_to_silver(bronze_root, silver_root, ["2026-06-06"], ["yelp"])
    assert affected1 == {"2016-03-09"}

    path = silver_partition_path(silver_root, "2016-03-09")
    assert len(read_silver_partition(path)) == 1

    # Re-run same ingestion date — overwrite bronze partition and reprocess.
    reviews2 = reviews.copy()
    reviews2["_ingested_at"] = "2026-06-06T12:00:00Z"
    yelp_write_bronze(reviews2, business, bronze_root / "yelp", "2026-06-06")
    process_ingestion_to_silver(bronze_root, silver_root, ["2026-06-06"], ["yelp"])
    silver = read_silver_partition(path)
    assert len(silver) == 1
    assert silver.iloc[0][INGESTED_AT_FIELD] == "2026-06-06T12:00:00Z"


def test_late_arrival_updates_old_review_date_partition(tmp_path):
    bronze_root = tmp_path / "bronze"
    silver_root = tmp_path / "silver" / "reviews"

    reviews_day1 = pd.DataFrame(
        {
            "review_id": ["r1"],
            "business_id": ["b"],
            "stars": [4],
            "text": ["good"],
            "date": ["2016-03-09 10:00:00"],
            "_ingested_at": ["2026-06-06T00:00:00Z"],
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"]})
    yelp_write_bronze(reviews_day1, business, bronze_root / "yelp", "2026-06-06")
    process_ingestion_to_silver(bronze_root, silver_root, ["2026-06-06"], ["yelp"])

    # Late arrival: same review_id, updated text, lands on a later ingestion date.
    reviews_day2 = pd.DataFrame(
        {
            "review_id": ["r1"],
            "business_id": ["b"],
            "stars": [5],
            "text": ["excellent updated"],
            "date": ["2016-03-09 10:00:00"],
            "_ingested_at": ["2026-06-07T00:00:00Z"],
        }
    )
    yelp_write_bronze(reviews_day2, business, bronze_root / "yelp", "2026-06-07")
    process_ingestion_to_silver(bronze_root, silver_root, ["2026-06-07"], ["yelp"])

    silver = read_silver_partition(silver_partition_path(silver_root, "2016-03-09"))
    assert len(silver) == 1
    assert silver.iloc[0]["text"] == "excellent updated"
    assert silver.iloc[0]["rating"] == 5.0


def test_assign_review_date_keys():
    df = pd.DataFrame(
        {
            "source": ["yelp", "tripadvisor"],
            "date": ["2016-03-09 10:00:00", "2022-02-06"],
        }
    )
    keys = assign_review_date_keys(df)
    assert list(keys) == ["2016-03-09", "2022-02-06"]


def test_refine_yelp_sets_text_len():
    reviews = pd.DataFrame(
        {
            "review_id": ["r1"],
            "business_id": ["b"],
            "stars": [5],
            "text": ["hello"],
            "date": ["2020-01-01"],
            "_ingested_at": ["t"],
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"]})
    out = refine_yelp(reviews, business)
    assert out.iloc[0]["text_len"] == 5


def test_silver_passes_great_expectations_gate():
    pytest.importorskip("great_expectations")
    from data.expectations.reviews_suite import validate_silver

    reviews = pd.DataFrame(
        {
            "review_id": ["r1", "r2", "r3"],
            "business_id": ["b", "b", "b"],
            "stars": [5, 3, 1],
            "text": ["delicious coffee and friendly staff", "espresso was fine nothing special", "cold bitter slow rude"],
            "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
            "_ingested_at": ["t"] * 3,
        }
    )
    business = pd.DataFrame({"business_id": ["b"], "name": ["Cafe"], "city": ["Austin"], "state": ["TX"]})
    silver = refine_yelp(reviews, business)
    result = validate_silver(silver[SILVER_COLUMNS_WITH_DATE], check_language=False)
    assert result.success, result.failures
