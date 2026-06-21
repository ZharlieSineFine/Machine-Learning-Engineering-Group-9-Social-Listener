"""Postgres warehouse for the Silver/Gold medallion tables.

Carries self-contained DDL (`ensure_schema`) so the layer works even if the container's
`init.sql` wasn't applied, plus idempotent upserts keyed on the pipeline's natural keys:

    reviews_silver   PK (source, source_id)        -- harmonised, deduped; NO labels
    reviews_gold     PK (review_id)                 -- review_id == silver source_id; carries label
    human_corrections                               -- feedback loop (Phase 2)

Upserts use `ON CONFLICT ... DO UPDATE`, so re-publishing a partition is safe.

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, List, Tuple

import pandas as pd

from data.storage.config import PostgresConfig

SILVER_TABLE = "reviews_silver"
GOLD_TABLE = "reviews_gold"

SILVER_DDL = """
CREATE TABLE IF NOT EXISTS reviews_silver (
    source       TEXT NOT NULL,
    source_id    TEXT NOT NULL,
    text         TEXT NOT NULL,
    text_len     INTEGER,
    rating       REAL,
    restaurant   TEXT,
    location     TEXT,
    review_date  TEXT,
    ingested_at  TEXT,
    PRIMARY KEY (source, source_id)
);
"""
SILVER_INDEX_DDL = "CREATE INDEX IF NOT EXISTS reviews_silver_review_date_idx ON reviews_silver (review_date);"

GOLD_DDL = """
CREATE TABLE IF NOT EXISTS reviews_gold (
    review_id    TEXT PRIMARY KEY,
    review_date  TEXT,
    text         TEXT NOT NULL,
    label        TEXT NOT NULL CHECK (label IN ('negative', 'neutral', 'positive')),
    label_source TEXT NOT NULL DEFAULT 'derived_from_rating',
    text_len     INTEGER,
    built_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""
GOLD_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS reviews_gold_review_date_idx ON reviews_gold (review_date);"
    "CREATE INDEX IF NOT EXISTS reviews_gold_label_idx ON reviews_gold (label);"
)

HUMAN_CORRECTIONS_DDL = """
CREATE TABLE IF NOT EXISTS human_corrections (
    id              BIGSERIAL PRIMARY KEY,
    review_id       TEXT NOT NULL,
    corrected_label TEXT NOT NULL CHECK (corrected_label IN ('negative', 'neutral', 'positive')),
    corrected_by    TEXT,
    corrected_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# Map Silver parquet columns -> reviews_silver table columns (table order matters for upsert).
_SILVER_DF_TO_TABLE = {
    "source": "source",
    "source_id": "source_id",
    "text": "text",
    "text_len": "text_len",
    "rating": "rating",
    "restaurant": "restaurant",
    "location": "location",
    "date": "review_date",
    "_ingested_at": "ingested_at",
}
SILVER_TABLE_COLUMNS: List[str] = list(_SILVER_DF_TO_TABLE.values())
GOLD_TABLE_COLUMNS: List[str] = ["review_id", "review_date", "text", "label", "label_source", "text_len"]


def connect(cfg: PostgresConfig):
    """Open a psycopg2 connection (imported lazily so the module loads without psycopg2)."""
    import psycopg2

    return psycopg2.connect(cfg.dsn)


@contextmanager
def connection(cfg: PostgresConfig) -> Iterator["object"]:
    """Connection context manager: commits on success, rolls back on error, always closes."""
    conn = connect(cfg)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(conn) -> None:
    """Create the Silver/Gold/feedback tables + indexes if they don't already exist."""
    with conn.cursor() as cur:
        cur.execute(SILVER_DDL)
        cur.execute(SILVER_INDEX_DDL)
        cur.execute(GOLD_DDL)
        cur.execute(GOLD_INDEX_DDL)
        cur.execute(HUMAN_CORRECTIONS_DDL)


def _records(df: pd.DataFrame, columns: List[str]) -> List[Tuple]:
    """Rows as tuples in `columns` order, with NaN/NaT coerced to None for psycopg2."""
    sub = df.reindex(columns=columns)
    sub = sub.astype(object).where(pd.notnull(sub), None)
    return list(sub.itertuples(index=False, name=None))


def _silver_table_frame(silver_df: pd.DataFrame) -> pd.DataFrame:
    """Rename Silver parquet columns to the reviews_silver table columns."""
    present = {src: dst for src, dst in _SILVER_DF_TO_TABLE.items() if src in silver_df.columns}
    return silver_df.rename(columns=present)


def upsert_silver(conn, silver_df: pd.DataFrame) -> int:
    """Upsert Silver rows into reviews_silver on (source, source_id). Returns rows written."""
    from psycopg2.extras import execute_values

    if silver_df is None or silver_df.empty:
        return 0
    table_df = _silver_table_frame(silver_df)
    rows = _records(table_df, SILVER_TABLE_COLUMNS)
    cols = ", ".join(SILVER_TABLE_COLUMNS)
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in SILVER_TABLE_COLUMNS if c not in ("source", "source_id"))
    sql = (
        f"INSERT INTO {SILVER_TABLE} ({cols}) VALUES %s "
        f"ON CONFLICT (source, source_id) DO UPDATE SET {updates}"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    return len(rows)


def upsert_gold(conn, gold_df: pd.DataFrame) -> int:
    """Upsert Gold rows into reviews_gold on review_id.

    `gold_df` needs `review_id, review_date, text, label`; `label_source` and `text_len`
    are filled in when absent.
    """
    from psycopg2.extras import execute_values

    if gold_df is None or gold_df.empty:
        return 0
    df = gold_df.copy()
    if "label_source" not in df.columns:
        df["label_source"] = "derived_from_rating"
    if "text_len" not in df.columns:
        df["text_len"] = df["text"].astype(str).str.len()
    rows = _records(df, GOLD_TABLE_COLUMNS)
    cols = ", ".join(GOLD_TABLE_COLUMNS)
    updates = ", ".join(
        f"{c}=EXCLUDED.{c}" for c in GOLD_TABLE_COLUMNS if c != "review_id"
    ) + ", built_at=NOW()"
    sql = (
        f"INSERT INTO {GOLD_TABLE} ({cols}) VALUES %s "
        f"ON CONFLICT (review_id) DO UPDATE SET {updates}"
    )
    with conn.cursor() as cur:
        execute_values(cur, sql, rows, page_size=1000)
    return len(rows)


def table_count(conn, table: str) -> int:
    """Row count for a table (used by integration tests / sanity checks)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        return int(cur.fetchone()[0])
