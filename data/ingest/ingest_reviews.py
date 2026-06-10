"""Shared schema definitions for the medallion layers.

Layer shapes:

* `SILVER_COLUMNS` — harmonised reviews after Bronze refinement: join, clean, ISO
  `date`, stable `source_id`. **No derived labels** — Silver carries `rating` only.
* `EXPECTED_COLUMNS` — the 6-column **training contract** (includes `label`). Produced
  by the Gold builder (`data/refine/build_gold.py`), not Silver.
* `GOLD_COLUMNS` — training contract + ISO `date`.

The **Bronze** layer has no single fixed schema: each adapter writes its source's own
columns verbatim plus `_source` / `_ingested_at`. Harmonisation to Silver happens in
`data/refine/build_silver.py`; label derivation happens in `data/refine/build_gold.py`.

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List

# Silver: cleaned/joined reviews — rating yes, label no.
SILVER_COLUMNS: List[str] = [
    "text",
    "text_len",
    "rating",
    "source",
    "source_id",
    "restaurant",
    "location",
]

# Temporal column — normalised to ISO in Silver. Nullable. Drives the OOT split.
DATE_COLUMN: str = "date"

SILVER_COLUMNS_WITH_DATE: List[str] = [*SILVER_COLUMNS, DATE_COLUMN]

# Stable per-review key within a source (Yelp review_id; TripAdvisor content hash).
SOURCE_ID_FIELD: str = "source_id"

# Canonical review key in Gold feature/label stores (= source_id).
REVIEW_ID_FIELD: str = "review_id"

# Hive-style partition keys for daily incremental layout.
INGESTION_DATE_FIELD: str = "ingestion_date"
REVIEW_DATE_PARTITION: str = "review_date"
NULL_REVIEW_DATE_PARTITION: str = "__null__"

# Gold / training contract (matches `data/sample/reviews_sample.csv` header).
EXPECTED_COLUMNS: List[str] = [
    "text",
    "label",
    "rating",
    "source",
    "restaurant",
    "location",
]

GOLD_COLUMNS: List[str] = [*EXPECTED_COLUMNS, DATE_COLUMN]

# Per-review Gold store column contracts.
FEATURE_STORE_COLUMNS: List[str] = [
    REVIEW_ID_FIELD,
    REVIEW_DATE_PARTITION,
    "text",
]

LABEL_STORE_COLUMNS: List[str] = [
    REVIEW_ID_FIELD,
    REVIEW_DATE_PARTITION,
    "label",
]

# Provenance columns every Bronze adapter appends to the raw source rows.
SOURCE_FIELD: str = "_source"
INGESTED_AT_FIELD: str = "_ingested_at"

# Carried in Silver parquet for dedup across ingestion batches (not in GE contract).
SILVER_PROVENANCE_COLUMNS: List[str] = [INGESTED_AT_FIELD]

# Silver scope: keep only the last N calendar years per source (each source anchors on its
# own max review date). Bronze retains full history; set to None to disable filtering.
DEFAULT_SILVER_RECENT_YEARS: float = 3.0


def utc_now_iso() -> str:
    """Provenance timestamp for `_ingested_at` (UTC, second precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingestion_partition_name(ingestion_date: str) -> str:
    """Bronze Hive partition directory name, e.g. ``dt=2026-06-06``."""
    return f"dt={ingestion_date}"


def review_date_partition_name(review_date: str | None) -> str:
    """Silver/Gold Hive partition directory name for a review event date."""
    key = review_date if review_date else NULL_REVIEW_DATE_PARTITION
    return f"{REVIEW_DATE_PARTITION}={key}"
