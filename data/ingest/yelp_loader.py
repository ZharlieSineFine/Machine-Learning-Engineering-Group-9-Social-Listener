"""Yelp source adapter — RAW Bronze ingestion from the Yelp Open Dataset.

Bronze = the source's own records, verbatim, plus provenance. This adapter does NO join,
NO label derivation, NO cleaning. It emits two raw tables exactly as they appear in the
Yelp JSON, *selected* to the beverage business segment (a sourcing decision — BrewLeaf is a
bubble-tea brand; see the proposal deck):

    reviews.csv   <- review_id, user_id, business_id, stars, useful, funny, cool, text, date
    business.csv  <- business_id, name, address, city, state, postal_code, stars, review_count, categories

Both get `_source` + `_ingested_at` provenance columns. The review `date` is the literal
Yelp timestamp ("2018-07-07 22:09:11") copied straight through — never reformatted. The
join (review.business_id -> business.name/city), label derivation, and any cleaning all live
in the Silver refiner (`data/refine/build_silver.py`).

review.json is multi-GB, so it is streamed line-by-line — or straight out of the ~9 GB tar
via `--from-tar` (`tarfile.extractfile`, nothing hits disk). Only the small beverage-business
index is held in memory.

Run:
    python -m data.ingest.yelp_loader --from-tar /path/to/yelp_dataset --out-dir data/bronze/yelp
    python -m data.ingest.yelp_loader --reviews review.json --business business.json --out-dir data/bronze/yelp

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import pandas as pd

from data.ingest.ingest_reviews import INGESTED_AT_FIELD, SOURCE_FIELD, ingestion_partition_name, utc_now_iso

ROOT = Path(__file__).resolve().parents[2]

SOURCE = "yelp"

# BrewLeaf is a bubble-tea brand, so the default ingestion scope is the beverage segment:
# coffee, tea, cafés, juice bars and bubble tea. Override with --categories to broaden
# (e.g. add "Restaurants") or narrow. Pass --categories with no values to ingest everything.
DEFAULT_CATEGORIES = frozenset({
    "Coffee & Tea",
    "Cafes",
    "Coffeeshops",
    "Coffee Roasteries",
    "Bubble Tea",
    "Juice Bars & Smoothies",
    "Tea Rooms",
})

# Raw source fields kept in each Bronze table (values copied verbatim; provenance appended).
REVIEW_FIELDS = ["review_id", "user_id", "business_id", "stars", "useful", "funny", "cool", "text", "date"]
BUSINESS_FIELDS = ["business_id", "name", "address", "city", "state", "postal_code", "stars", "review_count", "categories"]
REVIEW_BRONZE_COLUMNS = [*REVIEW_FIELDS, SOURCE_FIELD, INGESTED_AT_FIELD]
BUSINESS_BRONZE_COLUMNS = [*BUSINESS_FIELDS, SOURCE_FIELD, INGESTED_AT_FIELD]

# Member names inside the official Yelp Open Dataset tar archive.
YELP_TAR_BUSINESS = "yelp_academic_dataset_business.json"
YELP_TAR_REVIEW = "yelp_academic_dataset_review.json"


# ---------- category scope (pure row selection — no value changes) ----------

def _normalize_categories(raw) -> Set[str]:
    """Coerce Yelp's `categories` field (CSV string or list) into a set of tokens."""
    if raw is None:
        return set()
    if isinstance(raw, str):
        return {c.strip() for c in raw.split(",") if c.strip()}
    if isinstance(raw, (list, tuple, set)):
        return {c.strip() for c in raw if isinstance(c, str) and c.strip()}
    return set()


def _matches_categories(raw, wanted: Set[str]) -> bool:
    """True if the business should be kept. Empty `wanted` means accept all."""
    if not wanted:
        return True
    return bool(_normalize_categories(raw) & wanted)


# ---------- streaming JSON-lines ----------

def _iter_json_records(lines: Iterable[str]) -> Iterator[dict]:
    """Yield one parsed object per line, skipping blank/malformed lines.

    Works on any iterable of strings — a file handle, or a tar member streamed via
    `tarfile.extractfile`, so review.json never hits disk.
    """
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream one parsed object per line from a JSON-lines file on disk."""
    with open(path, "r", encoding="utf-8") as fh:
        yield from _iter_json_records(fh)


# ---------- raw row collection (verbatim values) ----------

def _pick(record: dict, fields: List[str]) -> dict:
    """Copy `fields` out of a record verbatim (missing keys -> None)."""
    return {f: record.get(f) for f in fields}


def build_business_index(
    records: Iterable[dict], categories: Set[str]
) -> Tuple[Set[str], List[dict]]:
    """Return (beverage business_id set, raw business rows) for matching businesses.

    Pure selection by category — every kept business field is copied verbatim.
    """
    ids: Set[str] = set()
    rows: List[dict] = []
    for biz in records:
        if not _matches_categories(biz.get("categories"), categories):
            continue
        bid = biz.get("business_id")
        if not bid:
            continue
        ids.add(bid)
        rows.append(_pick(biz, BUSINESS_FIELDS))
    return ids, rows


def collect_reviews(
    records: Iterable[dict], business_ids: Set[str], limit: Optional[int] = None
) -> List[dict]:
    """Return raw review rows whose business_id is in scope (verbatim — no cleaning).

    Note: rows with null text or unusable stars are intentionally KEPT here — Bronze is a
    faithful copy of the source; dropping invalid rows is Silver's job.
    """
    rows: List[dict] = []
    for review in records:
        if review.get("business_id") not in business_ids:
            continue
        rows.append(_pick(review, REVIEW_FIELDS))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _stamp(rows: List[dict], columns: List[str], ingested_at: str) -> pd.DataFrame:
    """Build a Bronze frame from raw rows and append provenance columns."""
    base = [c for c in columns if c not in (SOURCE_FIELD, INGESTED_AT_FIELD)]
    df = pd.DataFrame(rows, columns=base)
    df[SOURCE_FIELD] = SOURCE
    df[INGESTED_AT_FIELD] = ingested_at
    return df[columns]


def extract_bronze_from_records(
    review_records: Iterable[dict],
    business_records: Iterable[dict],
    categories: Set[str] = DEFAULT_CATEGORIES,
    limit: Optional[int] = None,
    ingested_at: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Core: (reviews_bronze, business_bronze) from parsed review + business records.

    Builds the beverage-business index first (consuming `business_records`), then selects
    reviews for those businesses. `ingested_at` is injectable for deterministic tests.
    """
    ingested_at = ingested_at or utc_now_iso()
    ids, business_rows = build_business_index(business_records, categories)
    review_rows = collect_reviews(review_records, ids, limit)
    reviews_df = _stamp(review_rows, REVIEW_BRONZE_COLUMNS, ingested_at)
    business_df = _stamp(business_rows, BUSINESS_BRONZE_COLUMNS, ingested_at)
    return reviews_df, business_df


def extract_bronze_from_paths(
    review_path: Path,
    business_path: Path,
    categories: Set[str] = DEFAULT_CATEGORIES,
    limit: Optional[int] = None,
    ingested_at: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stream extracted review.json + business.json from disk into Bronze frames."""
    return extract_bronze_from_records(
        _iter_jsonl(review_path), _iter_jsonl(business_path), categories, limit, ingested_at
    )


def extract_bronze_from_tar(
    tar_path: Path,
    categories: Set[str] = DEFAULT_CATEGORIES,
    limit: Optional[int] = None,
    ingested_at: Optional[str] = None,
    business_member: str = YELP_TAR_BUSINESS,
    review_member: str = YELP_TAR_REVIEW,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the Yelp dataset straight from the official ~9 GB tar into Bronze frames.

    business.json (~118 MB) is consumed first to build the beverage index, then review.json
    (~5.3 GB) is streamed past — only the index is held in memory.
    """
    import io
    import tarfile

    ingested_at = ingested_at or utc_now_iso()
    with tarfile.open(tar_path, "r") as tf:
        biz_fp = tf.extractfile(business_member)
        if biz_fp is None:
            raise FileNotFoundError(f"{business_member!r} not found in {tar_path}")
        with biz_fp:
            ids, business_rows = build_business_index(
                _iter_json_records(io.TextIOWrapper(biz_fp, encoding="utf-8")), categories
            )

        rev_fp = tf.extractfile(review_member)
        if rev_fp is None:
            raise FileNotFoundError(f"{review_member!r} not found in {tar_path}")
        with rev_fp:
            review_rows = collect_reviews(
                _iter_json_records(io.TextIOWrapper(rev_fp, encoding="utf-8")), ids, limit
            )

    reviews_df = _stamp(review_rows, REVIEW_BRONZE_COLUMNS, ingested_at)
    business_df = _stamp(business_rows, BUSINESS_BRONZE_COLUMNS, ingested_at)
    return reviews_df, business_df


def bronze_partition_dir(out_dir: Path, ingestion_date: str) -> Path:
    """Return ``<out_dir>/dt=YYYY-MM-DD`` for daily partitioned bronze."""
    return out_dir / ingestion_partition_name(ingestion_date)


def write_bronze_partition(
    reviews_df: pd.DataFrame,
    business_df: pd.DataFrame,
    out_dir: Path,
    ingestion_date: str,
) -> tuple[Path, Path]:
    """Write Yelp bronze CSVs under ``dt=<ingestion_date>``."""
    part = bronze_partition_dir(out_dir, ingestion_date)
    part.mkdir(parents=True, exist_ok=True)
    reviews_path = part / "reviews.csv"
    business_path = part / "business.csv"
    reviews_df.to_csv(reviews_path, index=False)
    business_df.to_csv(business_path, index=False)
    return reviews_path, business_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ingest the Yelp beverage segment into RAW Bronze (reviews.csv + business.csv)."
    )
    ap.add_argument(
        "--from-tar",
        type=Path,
        default=None,
        help="Path to the Yelp Open Dataset tar (streams business.json + review.json out of it).",
    )
    ap.add_argument("--reviews", type=Path, default=None, help="Path to extracted Yelp review.json (JSON lines).")
    ap.add_argument("--business", type=Path, default=None, help="Path to extracted Yelp business.json (JSON lines).")
    ap.add_argument(
        "--categories",
        nargs="*",
        default=sorted(DEFAULT_CATEGORIES),
        help="Business categories to keep (default: the beverage segment). Pass --categories with no values to ingest all.",
    )
    ap.add_argument("--n", type=int, default=None, dest="limit", help="Cap the number of review rows.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Bronze root (writes dt=YYYY-MM-DD/ subdir).")
    ap.add_argument(
        "--ingestion-date",
        type=str,
        default=None,
        help="Landing partition date (YYYY-MM-DD). Default: today UTC.",
    )
    args = ap.parse_args()

    categories = set(args.categories)
    if args.from_tar is not None:
        reviews_df, business_df = extract_bronze_from_tar(args.from_tar, categories, args.limit)
    elif args.reviews is not None and args.business is not None:
        reviews_df, business_df = extract_bronze_from_paths(args.reviews, args.business, categories, args.limit)
    else:
        raise SystemExit("Provide either --from-tar <tar> or both --reviews <json> and --business <json>.")

    from datetime import date

    ingestion_date = args.ingestion_date or date.today().isoformat()
    reviews_path, business_path = write_bronze_partition(
        reviews_df, business_df, args.out_dir, ingestion_date
    )

    n_dated = int(reviews_df["date"].notna().sum())
    print(f"Bronze reviews:  {len(reviews_df)} rows ({n_dated} with a date) -> {reviews_path}")
    print(f"Bronze business: {len(business_df)} rows -> {business_path}")


if __name__ == "__main__":
    main()
