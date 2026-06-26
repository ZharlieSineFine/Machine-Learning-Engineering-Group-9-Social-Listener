"""Bronze -> Silver refinement: harmonise raw source tables into validated partitions.

Daily incremental layout:

    Bronze  data/bronze/<source>/dt=YYYY-MM-DD/reviews.csv
    Silver  data/silver/reviews/review_date=YYYY-MM-DD/part.parquet

Per source:

    * Yelp        — join reviews.business_id -> business.name/city; ``source_id`` = review_id.
    * TripAdvisor — map Review/Rating/Restaurant/Location; ``source_id`` = hash of
                    (restaurant, text, parsed date).

Dedup on ``(source, source_id)`` keeping the row with the latest ``_ingested_at``.
By default only the **last 3 years per source** (by each source's max review date) are kept
in Silver; Bronze retains the full archive. Labels are derived in Gold only.

Run (single review_date partition):
    python -m data.refine.build_silver \\
        --bronze-root data/bronze --review-date 2022-02-06 \\
        --ingestion-dates 2026-06-06

Run (batch / legacy — all bronze under flat dirs):
    python -m data.refine.build_silver \\
        --yelp-dir data/bronze/yelp --tripadvisor data/bronze/tripadvisor/reviews.csv \\
        --out data/silver/reviews.csv

"""
from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import pandas as pd

from data.ingest.ingest_reviews import (
    DATE_COLUMN,
    DEFAULT_SILVER_RECENT_YEARS,
    INGESTED_AT_FIELD,
    NULL_REVIEW_DATE_PARTITION,
    REVIEW_DATE_PARTITION,
    SILVER_COLUMNS_WITH_DATE,
    SILVER_PROVENANCE_COLUMNS,
    ingestion_partition_name,
    review_date_partition_name,
)

_REVIEWED_PREFIX_RE = re.compile(r"(?i)^\s*reviewed\s+")

SILVER_PARQUET = "part.parquet"
SILVER_WRITE_COLUMNS = [*SILVER_COLUMNS_WITH_DATE, *SILVER_PROVENANCE_COLUMNS]


# ---------- pure transforms (unit-testable) ----------

def parse_review_date(raw) -> Optional[str]:
    """Normalise a TripAdvisor "Reviewed DD Month YYYY" stamp to ISO ``YYYY-MM-DD``."""
    if not isinstance(raw, str):
        return None
    cleaned = _REVIEWED_PREFIX_RE.sub("", raw).strip()
    if not cleaned:
        return None
    try:
        return datetime.strptime(cleaned, "%d %B %Y").strftime("%Y-%m-%d")
    except ValueError:
        return None


def yelp_event_date(raw) -> Optional[str]:
    """Extract calendar date from a Yelp timestamp string."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return pd.to_datetime(raw, errors="coerce").strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def partition_key_from_date(iso_date: Optional[str]) -> str:
    """Map an ISO review date (or None) to a Silver/Gold partition key."""
    return iso_date if iso_date else NULL_REVIEW_DATE_PARTITION


def derive_tripadvisor_source_id(restaurant: str, text: str, parsed_date: Optional[str]) -> str:
    """Stable TripAdvisor key — hash of restaurant + text + parsed calendar date."""
    payload = f"{restaurant}|{text}|{parsed_date or ''}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dedup_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest row per ``(source, source_id)`` by ``_ingested_at``."""
    if df.empty:
        return df
    work = df.copy()
    if INGESTED_AT_FIELD not in work.columns:
        work[INGESTED_AT_FIELD] = ""
    work = work.sort_values(INGESTED_AT_FIELD)
    work = work.drop_duplicates(subset=["source", "source_id"], keep="last")
    return work.reset_index(drop=True)


def silver_row_timestamps(df: pd.DataFrame) -> pd.Series:
    """Parse the harmonised Silver ``date`` column to timestamps."""
    if df.empty or DATE_COLUMN not in df.columns:
        return pd.Series(dtype="datetime64[ns]")
    raw = df[DATE_COLUMN]
    parsed = pd.to_datetime(raw, errors="coerce")
    iso_mask = raw.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False)
    if iso_mask.any():
        parsed = parsed.copy()
        parsed.loc[iso_mask] = pd.to_datetime(raw[iso_mask], errors="coerce")
    needs_parse = parsed.isna() & raw.notna()
    if needs_parse.any():
        parsed = parsed.copy()
        parsed.loc[needs_parse] = raw[needs_parse].map(parse_review_date)
        parsed = pd.to_datetime(parsed, errors="coerce")
    return parsed


def filter_recent_years_per_source(df: pd.DataFrame, years: float) -> pd.DataFrame:
    """Keep only rows within the last ``years`` calendar years per ``source``.

    Each source uses its own max review timestamp as the anchor (Yelp and TripAdvisor
    often end on different dates). Rows with unparseable dates are dropped.
    """
    if df.empty or years <= 0:
        return df
    timestamps = silver_row_timestamps(df)
    kept: List[pd.DataFrame] = []
    for source, group in df.groupby("source", sort=False):
        ts = timestamps.loc[group.index]
        dated = ts.notna()
        if not dated.any():
            continue
        cutoff = ts[dated].max() - pd.DateOffset(years=years)
        mask = dated & (ts >= cutoff)
        if mask.any():
            kept.append(group.loc[mask])
    if not kept:
        return df.iloc[0:0].copy()
    return pd.concat(kept, ignore_index=True)


# ---------- per-source refiners ----------

def refine_tripadvisor(bronze: pd.DataFrame) -> pd.DataFrame:
    """Raw TripAdvisor Bronze -> contract + ISO date + source_id."""
    required = ["Review", "Rating", "Restaurant"]
    missing = [c for c in required if c not in bronze.columns]
    if missing:
        raise ValueError(f"tripadvisor bronze is missing columns: {missing}")

    ingested = (
        bronze[INGESTED_AT_FIELD]
        if INGESTED_AT_FIELD in bronze.columns
        else pd.Series([""] * len(bronze), index=bronze.index)
    )

    text = bronze["Review"].map(lambda v: v.strip() if isinstance(v, str) else "")
    rating = pd.to_numeric(bronze["Rating"], errors="coerce")
    keep = (text.str.len() > 0) & rating.between(1, 5)

    sub = bronze[keep]
    text = text[keep]
    rating = rating[keep].astype(float)
    ingested = ingested[keep]
    dates = sub["Dates"] if "Dates" in sub.columns else pd.Series([None] * len(sub), index=sub.index)
    location = sub["Location"] if "Location" in sub.columns else pd.Series([""] * len(sub), index=sub.index)
    parsed_dates = [parse_review_date(d) for d in dates.values]
    restaurants = sub["Restaurant"].astype(str).tolist()

    out = pd.DataFrame(
        {
            "text": text.values,
            "rating": rating.values,
            "source": "tripadvisor",
            "source_id": [
                derive_tripadvisor_source_id(r, t, d)
                for r, t, d in zip(restaurants, text.values, parsed_dates)
            ],
            "restaurant": restaurants,
            "location": location.fillna("").astype(str).values,
            DATE_COLUMN: parsed_dates,
            INGESTED_AT_FIELD: ingested.values,
        },
        columns=SILVER_WRITE_COLUMNS,
    )
    out["text_len"] = out["text"].astype(str).str.len()
    return out.reset_index(drop=True)


def refine_yelp(
    reviews_bronze: pd.DataFrame,
    business_bronze: pd.DataFrame,
    location_with_state: bool = False,
) -> pd.DataFrame:
    """Raw Yelp Bronze (reviews + business) -> contract + date + source_id."""
    for col in ("business_id", "name", "city"):
        if col not in business_bronze.columns:
            raise ValueError(f"yelp business bronze is missing column: {col}")
    for col in ("business_id", "stars", "text", "date"):
        if col not in reviews_bronze.columns:
            raise ValueError(f"yelp reviews bronze is missing column: {col}")

    ingested = (
        reviews_bronze[INGESTED_AT_FIELD]
        if INGESTED_AT_FIELD in reviews_bronze.columns
        else pd.Series([""] * len(reviews_bronze), index=reviews_bronze.index)
    )

    biz = business_bronze
    name = biz["name"].fillna("").astype(str)
    city = biz["city"].fillna("").astype(str)
    state = biz["state"].fillna("").astype(str) if "state" in biz.columns else pd.Series([""] * len(biz))
    if location_with_state:
        location = [f"{c}, {s}" if c and s else c for c, s in zip(city, state)]
    else:
        location = city.tolist()
    lookup = dict(zip(biz["business_id"], zip(name.tolist(), location)))

    rev = reviews_bronze
    rating = pd.to_numeric(rev["stars"], errors="coerce")
    text_ok = rev["text"].map(lambda v: isinstance(v, str) and len(v.strip()) > 0)
    biz_ok = rev["business_id"].map(lambda b: b in lookup)
    keep = text_ok & biz_ok & rating.between(1, 5)

    sub = rev[keep]
    rating = rating[keep].astype(float)
    ingested = ingested[keep]
    names = [lookup[b][0] for b in sub["business_id"]]
    locations = [lookup[b][1] for b in sub["business_id"]]

    texts = sub["text"].astype(str).tolist()
    dates = sub["date"].tolist()
    if "review_id" in sub.columns:
        source_ids = sub["review_id"].astype(str).tolist()
    else:
        source_ids = [
            derive_tripadvisor_source_id(r, t, yelp_event_date(d))
            for r, t, d in zip(names, texts, dates)
        ]

    out = pd.DataFrame(
        {
            "text": texts,
            "rating": rating.values,
            "source": "yelp",
            "source_id": source_ids,
            "restaurant": names,
            "location": locations,
            DATE_COLUMN: dates,
            INGESTED_AT_FIELD: ingested.values,
        },
        columns=SILVER_WRITE_COLUMNS,
    )
    out["text_len"] = out["text"].astype(str).str.len()
    return out.reset_index(drop=True)


def build_silver(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate refined per-source frames, dedup, return the Silver contract."""
    valid = [f for f in frames if f is not None and len(f) > 0]
    if not valid:
        return pd.DataFrame(columns=SILVER_COLUMNS_WITH_DATE)
    combined = pd.concat(valid, ignore_index=True)
    combined = dedup_silver(combined)
    combined["text_len"] = combined["text"].astype(str).str.len()
    return combined[SILVER_COLUMNS_WITH_DATE]


# ---------- partitioned I/O ----------

def silver_partition_dir(silver_root: Path, review_date_key: str) -> Path:
    return silver_root / review_date_partition_name(review_date_key)


def silver_partition_path(silver_root: Path, review_date_key: str) -> Path:
    return silver_partition_dir(silver_root, review_date_key) / SILVER_PARQUET


def read_silver_partition(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=SILVER_WRITE_COLUMNS)
    df = pd.read_parquet(path)
    for col in SILVER_WRITE_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[SILVER_WRITE_COLUMNS]


def write_silver_partition(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in SILVER_WRITE_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out[SILVER_WRITE_COLUMNS].to_parquet(path, index=False)


def refine_bronze_frames(
    yelp_reviews: Optional[pd.DataFrame] = None,
    yelp_business: Optional[pd.DataFrame] = None,
    tripadvisor: Optional[pd.DataFrame] = None,
    *,
    location_with_state: bool = False,
    recent_years: Optional[float] = None,
) -> pd.DataFrame:
    """Refine one or more Bronze tables into a deduped Silver frame (all dates)."""
    frames: List[pd.DataFrame] = []
    if yelp_reviews is not None and yelp_business is not None:
        frames.append(refine_yelp(yelp_reviews, yelp_business, location_with_state))
    if tripadvisor is not None:
        frames.append(refine_tripadvisor(tripadvisor))
    if not frames:
        return pd.DataFrame(columns=SILVER_WRITE_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = dedup_silver(combined)
    combined["text_len"] = combined["text"].astype(str).str.len()
    if recent_years is not None:
        combined = filter_recent_years_per_source(combined, recent_years)
    return combined


def assign_review_date_keys(df: pd.DataFrame) -> pd.Series:
    """Return a partition key per row from the harmonised ``date`` column."""
    if df.empty:
        return pd.Series(dtype=str)
    ts = silver_row_timestamps(df)
    keys = ts.dt.strftime("%Y-%m-%d")
    return keys.where(ts.notna(), NULL_REVIEW_DATE_PARTITION)


def merge_silver_partition(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    review_date_key: str,
) -> pd.DataFrame:
    """Merge new refined rows into a review_date partition and dedup."""
    inc = incoming.copy()
    inc_keys = assign_review_date_keys(inc)
    inc = inc[inc_keys == review_date_key]
    if existing.empty:
        merged = inc
    else:
        merged = pd.concat([existing, inc], ignore_index=True)
    return dedup_silver(merged)


def load_bronze_ingestion(
    bronze_root: Path,
    source: str,
    ingestion_date: str,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Load Yelp (reviews+business) or TripAdvisor bronze for one ingestion partition."""
    part = bronze_root / source / ingestion_partition_name(ingestion_date)
    if source == "yelp":
        reviews_path = part / "reviews.csv"
        business_path = part / "business.csv"
        if not reviews_path.exists():
            legacy_root = bronze_root / source
            legacy_reviews = legacy_root / "reviews.csv"
            if legacy_reviews.exists():
                reviews = pd.read_csv(legacy_reviews)
                legacy_business = legacy_root / "business.csv"
                business = (
                    pd.read_csv(legacy_business) if legacy_business.exists() else pd.DataFrame()
                )
                return reviews, business
            return None, None
        reviews = pd.read_csv(reviews_path)
        business = pd.read_csv(business_path) if business_path.exists() else pd.DataFrame()
        return reviews, business
    if source == "tripadvisor":
        reviews_path = part / "reviews.csv"
        if not reviews_path.exists():
            legacy_reviews = bronze_root / source / "reviews.csv"
            if legacy_reviews.exists():
                return pd.read_csv(legacy_reviews), None
            return None, None
        return pd.read_csv(reviews_path), None
    raise ValueError(f"unknown source: {source}")


def process_ingestion_to_silver(
    bronze_root: Path,
    silver_root: Path,
    ingestion_dates: Sequence[str],
    sources: Sequence[str],
    *,
    location_with_state: bool = False,
    recent_years: Optional[float] = DEFAULT_SILVER_RECENT_YEARS,
) -> Set[str]:
    """Refine bronze ingestion batch(es) and upsert affected Silver review_date partitions."""
    incoming = refine_bronze_from_ingestions(
        bronze_root,
        ingestion_dates,
        sources,
        location_with_state=location_with_state,
        recent_years=recent_years,
    )
    if incoming.empty:
        return set()

    affected: Set[str] = set()
    keys = assign_review_date_keys(incoming)
    incoming = incoming.assign(__partition_key=keys)
    for review_date_key, group in incoming.groupby("__partition_key", sort=True):
        path = silver_partition_path(silver_root, review_date_key)
        existing = read_silver_partition(path)
        partition_rows = group.drop(columns=["__partition_key"])
        if existing.empty:
            merged = partition_rows
        else:
            merged = pd.concat([existing, partition_rows], ignore_index=True)
        merged = dedup_silver(merged)
        write_silver_partition(merged, path)
        affected.add(review_date_key)
    return affected


def refine_bronze_from_ingestions(
    bronze_root: Path,
    ingestion_dates: Sequence[str],
    sources: Sequence[str],
    *,
    location_with_state: bool = False,
    recent_years: Optional[float] = DEFAULT_SILVER_RECENT_YEARS,
) -> pd.DataFrame:
    """Load and refine bronze from one or more ingestion-date partitions."""
    yelp_reviews: List[pd.DataFrame] = []
    yelp_business: List[pd.DataFrame] = []
    tripadvisor: List[pd.DataFrame] = []

    for ingestion_date in ingestion_dates:
        if "yelp" in sources:
            rev, biz = load_bronze_ingestion(bronze_root, "yelp", ingestion_date)
            if rev is not None and not rev.empty:
                yelp_reviews.append(rev)
            if biz is not None and not biz.empty:
                yelp_business.append(biz)
        if "tripadvisor" in sources:
            rev, _ = load_bronze_ingestion(bronze_root, "tripadvisor", ingestion_date)
            if rev is not None and not rev.empty:
                tripadvisor.append(rev)

    yelp_rev_df = pd.concat(yelp_reviews, ignore_index=True) if yelp_reviews else None
    yelp_biz_df = pd.concat(yelp_business, ignore_index=True) if yelp_business else None
    trip_df = pd.concat(tripadvisor, ignore_index=True) if tripadvisor else None

    return refine_bronze_frames(
        yelp_rev_df,
        yelp_biz_df,
        trip_df,
        location_with_state=location_with_state,
        recent_years=recent_years,
    )


def reprocess_review_date(
    silver_root: Path,
    bronze_root: Path,
    review_date_key: str,
    ingestion_dates: Iterable[str],
    sources: Sequence[str],
    *,
    location_with_state: bool = False,
    recent_years: Optional[float] = DEFAULT_SILVER_RECENT_YEARS,
) -> pd.DataFrame:
    """Re-merge bronze batches into one Silver review_date partition (late arrivals)."""
    incoming = refine_bronze_from_ingestions(
        bronze_root,
        list(ingestion_dates),
        sources,
        location_with_state=location_with_state,
        recent_years=recent_years,
    )
    path = silver_partition_path(silver_root, review_date_key)
    existing = read_silver_partition(path)
    merged = merge_silver_partition(existing, incoming, review_date_key)
    write_silver_partition(merged, path)
    return merged[SILVER_COLUMNS_WITH_DATE]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Refine RAW Bronze into Silver (partitioned parquet or legacy CSV)."
    )
    ap.add_argument("--bronze-root", type=Path, default=None, help="Bronze root (partitioned layout).")
    ap.add_argument("--silver-root", type=Path, default=None, help="Silver root for review_date= partitions.")
    ap.add_argument(
        "--ingestion-dates",
        nargs="+",
        default=None,
        help="Bronze dt= partitions to process (partitioned mode).",
    )
    ap.add_argument("--sources", nargs="+", default=["yelp", "tripadvisor"], help="Sources to load.")
    ap.add_argument("--review-date", type=str, default=None, help="Reprocess a single review_date partition.")
    ap.add_argument("--yelp-dir", type=Path, default=None, help="Legacy flat Yelp bronze dir.")
    ap.add_argument("--tripadvisor", type=Path, default=None, help="Legacy flat TripAdvisor bronze CSV.")
    ap.add_argument("--out", type=Path, default=None, help="Legacy output Silver CSV path.")
    ap.add_argument("--location-with-state", action="store_true", help='Yelp location as "City, ST".')
    ap.add_argument(
        "--recent-years",
        type=float,
        default=DEFAULT_SILVER_RECENT_YEARS,
        help="Keep only the last N years per source (default: 3).",
    )
    ap.add_argument(
        "--all-years",
        action="store_true",
        help="Disable the per-source recency filter (retain full Bronze history in Silver).",
    )
    args = ap.parse_args()
    recent_years = None if args.all_years else args.recent_years

    if args.bronze_root and args.silver_root and args.ingestion_dates:
        if args.review_date:
            merged = reprocess_review_date(
                args.silver_root,
                args.bronze_root,
                args.review_date,
                args.ingestion_dates,
                args.sources,
                location_with_state=args.location_with_state,
                recent_years=recent_years,
            )
            print(f"Silver partition {args.review_date}: {len(merged)} rows")
        else:
            affected = process_ingestion_to_silver(
                args.bronze_root,
                args.silver_root,
                args.ingestion_dates,
                args.sources,
                location_with_state=args.location_with_state,
                recent_years=recent_years,
            )
            print(f"Silver: updated {len(affected)} review_date partition(s): {sorted(affected)}")
        return

    if not args.out:
        raise SystemExit("Provide --out (legacy) or --bronze-root + --silver-root + --ingestion-dates.")

    frames: List[pd.DataFrame] = []
    if args.yelp_dir is not None:
        reviews = pd.read_csv(args.yelp_dir / "reviews.csv")
        business = pd.read_csv(args.yelp_dir / "business.csv")
        frames.append(refine_yelp(reviews, business, location_with_state=args.location_with_state))
    if args.tripadvisor is not None:
        frames.append(refine_tripadvisor(pd.read_csv(args.tripadvisor)))

    if not frames:
        raise SystemExit("Provide --yelp-dir and/or --tripadvisor.")

    if recent_years is not None:
        frames = [filter_recent_years_per_source(f, recent_years) for f in frames]
    silver = build_silver(frames)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    silver.to_csv(args.out, index=False)

    n_dated = int(silver[DATE_COLUMN].notna().sum())
    print(f"Silver: {len(silver)} rows ({n_dated} with a date) -> {args.out}")
    print(silver["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
