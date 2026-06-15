"""Malaysia TripAdvisor source adapter — RAW Bronze ingestion (no transformation).

Bronze = the export's own columns, verbatim, plus provenance. This adapter does NO label
derivation, NO date reformatting, NO rating coercion, NO text cleaning, NO row dropping for
bad values — all of that is the Silver refiner's job (`data/refine/build_silver.py`).

Rows are *selected* to the beverage segment (a sourcing scope, configurable) but their
values are never modified, so the Bronze `Dates` column keeps the literal
"Reviewed 6 February 2022" string straight from the export, and `Rating`/`Review` are
untouched.

Why a name filter? The cleaned TripAdvisor export has no `categories` field — only
`Author, Title, Review, Rating, Dates, Restaurant, Location`. BrewLeaf is a bubble-tea
brand, so to mirror the Yelp beverage slice we keep reviews whose *restaurant name* looks
like a beverage shop (coffee / tea / café / kopitiam / bubble tea / juice + a few SEA
chains). Override with `--keywords`, or ingest everything with `--no-filter`.

Run:
    python -m data.ingest.malaysia_review_loader \\
        --in "Malaysia Restaurant Review Datasets/data_cleaned/TripAdvisor_data_cleaned.csv" \\
        --out data/bronze/tripadvisor/reviews.csv

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from data.ingest.ingest_reviews import INGESTED_AT_FIELD, SOURCE_FIELD, ingestion_partition_name, utc_now_iso

ROOT = Path(__file__).resolve().parents[2]

SOURCE = "tripadvisor"

# Raw source columns kept in Bronze (verbatim), plus the provenance columns.
SRC_COLUMNS = ["Author", "Title", "Review", "Rating", "Dates", "Restaurant", "Location"]
SRC_RESTAURANT = "Restaurant"
BRONZE_COLUMNS = [*SRC_COLUMNS, SOURCE_FIELD, INGESTED_AT_FIELD]

# Beverage-shop name filter. No category column exists, so the beverage slice is matched on
# the restaurant *name*. Generic tokens use word boundaries (so "tea" doesn't match
# "s-tea-k"); known SEA coffee/bubble-tea chains are matched as substrings.
_GENERIC_BEVERAGE_TERMS = (
    "coffee",
    "cafe",
    "café",
    "kopitiam",
    "kopi",
    "tea house",
    "teahouse",
    "milk tea",
    "bubble tea",
    "boba",
    "juice",
    "smoothie",
    "barista",
    "roastery",
    "roaster",
    "espresso",
)
_BRAND_BEVERAGE_TERMS = (
    "tealive",
    "chatime",
    "gong cha",
    "gongcha",
    "starbucks",
    "zus",
    "coffee bean",
    "oldtown",
    "old town",
    "san francisco coffee",
    "hwc",
    "chagee",
    "heytea",
    "mixue",
)
_BEVERAGE_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(t) for t in _GENERIC_BEVERAGE_TERMS) + r")\b"
    + r"|(?:" + "|".join(re.escape(t) for t in _BRAND_BEVERAGE_TERMS) + r")",
    re.IGNORECASE,
)


def is_beverage_shop(name) -> bool:
    """True if a restaurant name looks like a drinks/beverage shop (sourcing scope)."""
    if not isinstance(name, str):
        return False
    return bool(_BEVERAGE_RE.search(name))


def _beverage_mask(names: pd.Series, keywords: Optional[Set[str]]) -> pd.Series:
    """Boolean row-selection mask over restaurant names (no value changes)."""
    if keywords:
        wanted = {k.lower() for k in keywords}
        return names.map(lambda n: isinstance(n, str) and any(k in n.lower() for k in wanted))
    return names.map(is_beverage_shop)


def load_bronze(
    csv_path: Path,
    beverage_only: bool = True,
    keywords: Optional[Set[str]] = None,
    limit: Optional[int] = None,
    ingested_at: Optional[str] = None,
) -> pd.DataFrame:
    """Read the cleaned TripAdvisor CSV and return RAW Bronze rows + provenance.

    Source columns are copied verbatim — `Dates` keeps its "Reviewed DD Month YYYY" form,
    `Rating`/`Review` are untouched, and no rows are dropped for bad values. Rows are only
    *selected* to the beverage segment (unless `--no-filter`/`beverage_only=False`).
    """
    raw = pd.read_csv(
        csv_path,
        encoding="utf-8",
        encoding_errors="replace",
        usecols=lambda c: c in set(SRC_COLUMNS),
        low_memory=False,
    )
    # Guarantee all source columns exist and in a stable order (missing -> NaN).
    raw = raw.reindex(columns=SRC_COLUMNS)

    if beverage_only or keywords:
        raw = raw[_beverage_mask(raw[SRC_RESTAURANT], keywords)]

    raw = raw.reset_index(drop=True)
    if limit is not None:
        raw = raw.head(limit)

    out = raw.copy()
    out[SOURCE_FIELD] = SOURCE
    out[INGESTED_AT_FIELD] = ingested_at or utc_now_iso()
    return out[BRONZE_COLUMNS]


def bronze_partition_dir(out_dir: Path, ingestion_date: str) -> Path:
    return out_dir / ingestion_partition_name(ingestion_date)


def write_bronze_partition(df: pd.DataFrame, out_dir: Path, ingestion_date: str) -> Path:
    """Write TripAdvisor bronze under ``dt=<ingestion_date>/reviews.csv``."""
    part = bronze_partition_dir(out_dir, ingestion_date)
    part.mkdir(parents=True, exist_ok=True)
    out_path = part / "reviews.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest the Malaysia TripAdvisor beverage segment into RAW Bronze (verbatim source columns + provenance)."
    )
    ap.add_argument("--in", dest="in_path", type=Path, required=True, help="Path to the cleaned TripAdvisor CSV.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Bronze root (writes dt=YYYY-MM-DD/reviews.csv).")
    ap.add_argument("--out", type=Path, default=None, help="Legacy flat output CSV path.")
    ap.add_argument(
        "--ingestion-date",
        type=str,
        default=None,
        help="Landing partition date (YYYY-MM-DD). Default: today UTC.",
    )
    ap.add_argument("--n", type=int, default=None, dest="limit", help="Cap the number of rows.")
    ap.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Override the beverage name filter with these substrings (case-insensitive).",
    )
    ap.add_argument(
        "--no-filter",
        action="store_true",
        help="Ingest all restaurants (disable the beverage-shop name filter).",
    )
    args = ap.parse_args()

    df = load_bronze(
        args.in_path,
        beverage_only=not args.no_filter,
        keywords=set(args.keywords) if args.keywords else None,
        limit=args.limit,
    )

    from datetime import date

    if args.out_dir is not None:
        ingestion_date = args.ingestion_date or date.today().isoformat()
        out_path = write_bronze_partition(df, args.out_dir, ingestion_date)
    elif args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        out_path = args.out
    else:
        raise SystemExit("Provide --out-dir (partitioned) or --out (legacy flat CSV).")

    print(f"Bronze (tripadvisor): {len(df)} raw rows -> {out_path}")


if __name__ == "__main__":
    main()
