"""Medallion schema contracts and Phase-1 Postgres CSV ingest.

Layer shapes (also see ``data/SCHEMA.md``):

* ``SILVER_COLUMNS`` — harmonised reviews after Bronze refinement (no derived labels).
* ``EXPECTED_COLUMNS`` — 6-column training contract (includes ``label``), produced by Gold.
* ``GOLD_COLUMNS`` — training contract + ISO ``date``.

The **Bronze** layer has no single fixed schema: each adapter writes source-native
columns plus ``_source`` / ``_ingested_at``.

This module also exposes the thin-slice **Postgres ingest** used by the Airflow DAG
and integration tests: read sample CSV → soft-clean → GE gate → ``reviews`` table.

Run from the CLI::

    python -m data.ingest.ingest_reviews \\
        --csv data/sample/reviews_sample.csv \\
        --dsn postgresql://mlops:mlops@localhost:5432/sentiment

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Medallion schema contracts (Bronze → Silver → Gold)
# ---------------------------------------------------------------------------

SILVER_COLUMNS: List[str] = [
    "text",
    "text_len",
    "rating",
    "source",
    "source_id",
    "restaurant",
    "location",
]

DATE_COLUMN: str = "date"
SILVER_COLUMNS_WITH_DATE: List[str] = [*SILVER_COLUMNS, DATE_COLUMN]
SOURCE_ID_FIELD: str = "source_id"
REVIEW_ID_FIELD: str = "review_id"
INGESTION_DATE_FIELD: str = "ingestion_date"
REVIEW_DATE_PARTITION: str = "review_date"
NULL_REVIEW_DATE_PARTITION: str = "__null__"

EXPECTED_COLUMNS: List[str] = [
    "text",
    "label",
    "rating",
    "source",
    "restaurant",
    "location",
]

GOLD_COLUMNS: List[str] = [*EXPECTED_COLUMNS, DATE_COLUMN]

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

SOURCE_FIELD: str = "_source"
INGESTED_AT_FIELD: str = "_ingested_at"
SILVER_PROVENANCE_COLUMNS: List[str] = [INGESTED_AT_FIELD]
DEFAULT_SILVER_RECENT_YEARS: float = 3.0

VALID_LABELS = frozenset({"negative", "neutral", "positive"})

# Match the lower bound enforced by the GE suite (data/expectations/reviews_suite.py).
MIN_TEXT_LEN = 5
HAS_LETTER_RE = r"[A-Za-z\u00c0-\u024f]"


def utc_now_iso() -> str:
    """Provenance timestamp for ``_ingested_at`` (UTC, second precision)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ingestion_partition_name(ingestion_date: str) -> str:
    """Bronze Hive partition directory name, e.g. ``dt=2026-06-06``."""
    return f"dt={ingestion_date}"


def review_date_partition_name(review_date: str | None) -> str:
    """Silver/Gold Hive partition directory name for a review event date."""
    key = review_date if review_date else NULL_REVIEW_DATE_PARTITION
    return f"{REVIEW_DATE_PARTITION}={key}"


# ---------------------------------------------------------------------------
# Phase-1 Postgres ingest (sample CSV → reviews table)
# ---------------------------------------------------------------------------

def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and return a DataFrame with the contract columns only.

    Drops rows with null/empty text, null label/source, or invalid labels.
    Raises ValueError if the file is missing required columns entirely.
    """
    df = pd.read_csv(csv_path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {sorted(missing)}")

    df = df.dropna(subset=["text", "label", "source"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= MIN_TEXT_LEN]
    df = df[df["text"].str.contains(HAS_LETTER_RE, regex=True)]
    df = df[df["label"].astype(str).isin(VALID_LABELS)]
    return df[EXPECTED_COLUMNS].reset_index(drop=True)


def to_records(df: pd.DataFrame) -> List[Tuple]:
    """Convert a validated DataFrame into a list of tuples for executemany."""
    return list(df[EXPECTED_COLUMNS].itertuples(index=False, name=None))


def insert_records(records: Iterable[Tuple], dsn: str, truncate: bool = True) -> int:
    """Insert records into the ``reviews`` table. Returns the number inserted."""
    import psycopg2
    from psycopg2.extras import execute_values

    rows = list(records)
    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                if truncate:
                    cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY CASCADE")
                if rows:
                    execute_values(
                        cur,
                        "INSERT INTO reviews "
                        "(text, label, rating, source, restaurant, location) VALUES %s",
                        rows,
                    )
        return len(rows)
    finally:
        conn.close()


def ingest(csv_path: Path, dsn: str, truncate: bool = True, run_ge: bool = True) -> int:
    """End-to-end: read CSV, soft-clean, run GE gate, write to Postgres."""
    df = load_and_validate(csv_path)

    if run_ge:
        from data.expectations.reviews_suite import validate_reviews

        result = validate_reviews(df)
        result.raise_for_status()

    records = to_records(df)
    return insert_records(records, dsn, truncate=truncate)


def _default_csv_path() -> Path:
    inside_container = Path("/opt/project/data/sample/reviews_sample.csv")
    if inside_container.exists():
        return inside_container
    return Path(__file__).resolve().parents[2] / "data" / "sample" / "reviews_sample.csv"


def _default_dsn() -> str:
    user = os.getenv("POSTGRES_USER", "mlops")
    pw = os.getenv("POSTGRES_PASSWORD", "mlops")
    host = os.getenv("POSTGRES_HOST", "postgres")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "sentiment")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=_default_csv_path())
    ap.add_argument("--dsn", type=str, default=_default_dsn())
    ap.add_argument("--no-truncate", dest="truncate", action="store_false")
    args = ap.parse_args()

    n = ingest(args.csv, args.dsn, truncate=args.truncate)
    print(f"Ingested {n} rows into reviews from {args.csv}")


if __name__ == "__main__":
    main()
