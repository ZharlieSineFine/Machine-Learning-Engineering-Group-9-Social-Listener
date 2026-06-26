#Unit tests for the RAW Malaysia TripAdvisor Bronze adapter.

from __future__ import annotations

import pandas as pd

from data.ingest.malaysia_review_loader import (
    BRONZE_COLUMNS,
    bronze_partition_dir,
    is_beverage_shop,
    load_bronze,
    write_bronze_partition,
)

RAW_ROWS = [
    # restaurant, review, rating, dates
    ("Bean Cafe", "great latte here", 5, "Reviewed 6 February 2022"),
    ("Bean Cafe", "   ", "garbage", "Reviewed yesterday"),       # empty text + bad rating (kept raw in Bronze!)
    ("Kopi House", "weak watery kopi", 2, "Reviewed 1 January 2020"),
    ("The Steakhouse", "good steak", 4, "Reviewed 2 March 2021"),  # non-beverage -> filtered out
    ("Tealive KLCC", "nice teh tarik", 3, "Reviewed 10 March 2021"),
]


def _write_csv(tmp_path):
    df = pd.DataFrame(
        {
            "Author": [f"a{i}" for i in range(len(RAW_ROWS))],
            "Title": ["t"] * len(RAW_ROWS),
            "Review": [r[1] for r in RAW_ROWS],
            "Rating": [r[2] for r in RAW_ROWS],
            "Dates": [r[3] for r in RAW_ROWS],
            "Restaurant": [r[0] for r in RAW_ROWS],
            "Location": ["KL"] * len(RAW_ROWS),
        }
    )
    path = tmp_path / "trip.csv"
    df.to_csv(path, index=False)
    return path


def test_bronze_columns_and_provenance(tmp_path):
    out = load_bronze(_write_csv(tmp_path))
    assert list(out.columns) == BRONZE_COLUMNS  # source columns + _source + _ingested_at
    assert (out["_source"] == "tripadvisor").all()
    assert out["_ingested_at"].notna().all()
    # No Silver-derived columns leaked into Bronze.
    for col in ("label", "text", "date", "rating", "source"):
        assert col not in out.columns


def test_dates_kept_verbatim_not_reformatted(tmp_path):
    out = load_bronze(_write_csv(tmp_path))
    bean = out[out["Restaurant"] == "Bean Cafe"]
    assert "Reviewed 6 February 2022" in set(bean["Dates"])  # literal source string
    assert not any(str(d).startswith("2022-02") for d in bean["Dates"])  # NOT ISO-ified


def test_beverage_selection_keeps_bad_rows_but_drops_non_beverage(tmp_path):
    out = load_bronze(_write_csv(tmp_path))
    # The steakhouse is filtered (non-beverage); the empty-text/garbage-rating café row is
    # KEPT (Bronze does not clean) -> Bean Cafe still has both of its rows.
    assert set(out["Restaurant"]) == {"Bean Cafe", "Kopi House", "Tealive KLCC"}
    assert (out["Restaurant"] == "Bean Cafe").sum() == 2
    assert "garbage" in set(out["Rating"].astype(str))  # bad rating preserved verbatim


def test_no_filter_keeps_everything(tmp_path):
    out = load_bronze(_write_csv(tmp_path), beverage_only=False)
    assert len(out) == len(RAW_ROWS)  # steakhouse included


def test_is_beverage_shop():
    assert is_beverage_shop("Geographer Cafe")
    assert is_beverage_shop("Old Town White Coffee")
    assert is_beverage_shop("Tealive SS15")
    assert not is_beverage_shop("Chambers Grill")
    assert not is_beverage_shop("The Steakhouse")  # 'tea' inside 'steak' must NOT match
    assert not is_beverage_shop(None)


def test_write_bronze_partition_lands_under_dt(tmp_path):
    out = load_bronze(_write_csv(tmp_path))
    out_dir = tmp_path / "tripadvisor"
    path = write_bronze_partition(out, out_dir, "2026-06-06")
    assert path == bronze_partition_dir(out_dir, "2026-06-06") / "reviews.csv"
    assert path.exists()
