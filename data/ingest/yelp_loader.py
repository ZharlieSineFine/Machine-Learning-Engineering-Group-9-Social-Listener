"""Yelp source adapter — normalise the Yelp open dataset into the review contract.

This is the Phase-2 Yelp loader referenced by data/README.md, ARCHITECTURE.md and
WORKFLOW.md. Like `scripts/build_sample.py` (the Google/TripAdvisor adapter), it is
*pure* — no DB, no Airflow imports — so it can be unit-tested in isolation and reused
by a CLI, an Airflow DAG, or a notebook. The output is the same 6-column contract every
layer downstream agrees on (see `data/ingest/ingest_reviews.EXPECTED_COLUMNS`).

Yelp ships one JSON object per line. Only two files are needed:
    review.json   — review_id, user_id, business_id, stars, date, text, useful, funny, cool
    business.json — business_id, name, city, state, categories, ... (the join target)

review.json has no restaurant name or location, so each review is JOINED to business.json
on `business_id`. The join is also where we scope to BrewLeaf's segment (coffee/tea/café
businesses) via `--categories`.

Contract mapping:
    text        <- review.text
    rating      <- float(review.stars)
    label       <- _label(rating)            # >=4 positive, ==3 neutral, <=2 negative
    source      <- "yelp"
    restaurant  <- business.name             # join on business_id
    location    <- business.city (+ state)   # join on business_id

Soft-cleaning (length/letter/null) is intentionally left to the existing
`data.ingest.ingest_reviews.load_and_validate` + the Great Expectations gate, exactly as
`build_sample.py` defers final cleaning. NOTE: a narrow category slice can come out skewed
to a single sentiment class; the GE suite requires all three classes to be present, so
widen `--categories` (e.g. add "Restaurants") if the gate complains.

review.json is multi-GB, so it is streamed line-by-line and never loaded whole; only the
(category-filtered) business lookup is held in memory.

Run:
    python -m data.ingest.yelp_loader \\
        --reviews /path/to/yelp/review.json \\
        --business /path/to/yelp/business.json \\
        --n 300 \\
        --out /tmp/yelp_sample.csv

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, Optional, Set, Tuple

import pandas as pd

from data.ingest.ingest_reviews import EXPECTED_COLUMNS

ROOT = Path(__file__).resolve().parents[2]
SEED_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"

SOURCE = "yelp"
# BrewLeaf is a coffee/tea brand, so the default scope is café-like businesses.
# Override with --categories (e.g. add "Restaurants") if the brand's scope is broader.
DEFAULT_CATEGORIES = frozenset({"Coffee & Tea", "Cafes", "Coffeeshops"})


def _label(rating: float) -> str:
    # Canonical rule — keep in sync with scripts/build_sample.py:_label
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


# ---------- pure helpers (unit-testable) ----------

def _normalize_categories(raw) -> Set[str]:
    """Coerce Yelp's `categories` field into a set of category tokens.

    The live dataset stores a comma-separated string ("Coffee & Tea, Cafes, Food");
    the documentation shows an array. Handle both, plus None/NaN.
    """
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


def _iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield one parsed object per line, skipping blank and malformed lines.

    A generator so the multi-GB review.json is streamed, never loaded whole.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_business_lookup(
    business_path: Path,
    categories: Set[str],
    location_with_state: bool = False,
) -> Dict[str, Tuple[str, str]]:
    """Stream business.json and return {business_id: (name, location)} for kept rows.

    Only businesses whose categories intersect `categories` are kept (see
    `_matches_categories`). ~150k businesses fit comfortably in memory.
    """
    lookup: Dict[str, Tuple[str, str]] = {}
    for biz in _iter_jsonl(business_path):
        if not _matches_categories(biz.get("categories"), categories):
            continue
        bid = biz.get("business_id")
        if not bid:
            continue
        name = (biz.get("name") or "").strip()
        city = (biz.get("city") or "").strip()
        state = (biz.get("state") or "").strip()
        location = f"{city}, {state}" if location_with_state and city and state else city
        lookup[bid] = (name, location)
    return lookup


def load_yelp(
    review_path: Path,
    business_path: Path,
    categories: Set[str] = DEFAULT_CATEGORIES,
    limit: Optional[int] = None,
    location_with_state: bool = False,
) -> pd.DataFrame:
    """Join Yelp reviews to businesses and return the 6-column contract DataFrame.

    Streams review.json; drops reviews whose business_id is not in the (filtered)
    business lookup or that lack usable text/stars. `limit` caps the number of *joined*
    rows (applied after the join), so `--n 300` yields 300 real rows even though most
    reviews are filtered out.
    """
    lookup = build_business_lookup(business_path, categories, location_with_state)

    rows = []
    for review in _iter_jsonl(review_path):
        biz = lookup.get(review.get("business_id"))
        if biz is None:
            continue
        text = review.get("text")
        if text is None:
            continue
        try:
            rating = float(review["stars"])
        except (KeyError, TypeError, ValueError):
            continue

        name, location = biz
        rows.append(
            {
                "text": text,
                "label": _label(rating),
                "rating": rating,
                "source": SOURCE,
                "restaurant": name,
                "location": location,
            }
        )
        if limit is not None and len(rows) >= limit:
            break

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Normalise the Yelp dataset (review.json + business.json) into the review contract CSV."
    )
    ap.add_argument("--reviews", type=Path, required=True, help="Path to Yelp review.json (JSON lines).")
    ap.add_argument("--business", type=Path, required=True, help="Path to Yelp business.json (JSON lines).")
    ap.add_argument(
        "--categories",
        nargs="*",
        default=sorted(DEFAULT_CATEGORIES),
        help="Business categories to keep. Pass --categories with no values to disable filtering.",
    )
    ap.add_argument("--n", type=int, default=None, dest="limit", help="Cap the number of joined rows.")
    ap.add_argument("--out", type=Path, required=True, help="Output CSV path (must not be the committed seed).")
    ap.add_argument(
        "--location-with-state",
        action="store_true",
        help='Format location as "City, ST" instead of just "City".',
    )
    args = ap.parse_args()

    if args.out.resolve() == SEED_CSV.resolve():
        raise SystemExit(f"Refusing to overwrite the committed seed CSV: {SEED_CSV}")

    df = load_yelp(
        args.reviews,
        args.business,
        categories=set(args.categories),
        limit=args.limit,
        location_with_state=args.location_with_state,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows -> {args.out}")
    print(df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
