"""Silver -> Gold refinement: per-review feature_store and label_store partitions.

Silver carries harmonised review fields (no labels). Gold attaches ``label`` from
``rating`` and writes two Hive-partitioned parquet stores:

    data/gold/feature_store/review_date=YYYY-MM-DD/part.parquet
    data/gold/label_store/review_date=YYYY-MM-DD/part.parquet

``review_id`` in Gold equals Silver ``source_id`` (canonical per-review key).

Run (partitioned):
    python -m data.refine.build_gold \\
        --silver-root data/silver/reviews --review-date 2022-02-06 \\
        --gold-root data/gold

Run (legacy CSV):
    python -m data.refine.build_gold --silver data/silver/reviews.csv --out data/gold/reviews.csv

"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

import pandas as pd

from data.ingest.ingest_reviews import (
    DATE_COLUMN,
    FEATURE_STORE_COLUMNS,
    GOLD_COLUMNS,
    LABEL_STORE_COLUMNS,
    REVIEW_DATE_PARTITION,
    REVIEW_ID_FIELD,
    SILVER_COLUMNS_WITH_DATE,
    review_date_partition_name,
)
from data.refine.build_silver import (
    read_silver_partition,
    silver_partition_path,
    yelp_event_date,
)

GOLD_PARQUET = "part.parquet"


def label_from_rating(rating: float) -> str:
    """Canonical sentiment rule — keep in sync with scripts/build_sample.py:_label."""
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _event_date_for_row(row: pd.Series, partition_key: str) -> Optional[str]:
    if partition_key == "__null__":
        return None
    raw = row.get(DATE_COLUMN)
    if row.get("source") == "yelp" and isinstance(raw, str):
        return yelp_event_date(raw) or partition_key
    if isinstance(raw, str) and len(raw) >= 10 and raw[4] == "-":
        return raw[:10]
    return partition_key


def build_gold(silver: pd.DataFrame) -> pd.DataFrame:
    """Attach ``label`` to a Silver frame (legacy combined Gold CSV)."""
    missing = [c for c in SILVER_COLUMNS_WITH_DATE if c not in silver.columns]
    if missing:
        raise ValueError(f"silver frame is missing columns: {missing}")

    out = silver[SILVER_COLUMNS_WITH_DATE].copy()

    if "label" in silver.columns:
        out["label"] = silver["label"]
        missing_label = out["label"].isna()
        out.loc[missing_label, "label"] = out.loc[missing_label, "rating"].map(label_from_rating)
    else:
        out["label"] = out["rating"].map(label_from_rating)

    return out[GOLD_COLUMNS]


def build_feature_store(silver: pd.DataFrame, review_date_key: str) -> pd.DataFrame:
    """Build the per-review feature store partition for one review_date."""
    if silver.empty:
        return pd.DataFrame(columns=FEATURE_STORE_COLUMNS)

    rows = []
    for _, row in silver.iterrows():
        rows.append(
            {
                REVIEW_ID_FIELD: row["source_id"],
                REVIEW_DATE_PARTITION: _event_date_for_row(row, review_date_key),
                "text": row["text"],
            }
        )
    return pd.DataFrame(rows, columns=FEATURE_STORE_COLUMNS)


def build_label_store(silver: pd.DataFrame, review_date_key: str) -> pd.DataFrame:
    """Build the per-review label store partition for one review_date."""
    if silver.empty:
        return pd.DataFrame(columns=LABEL_STORE_COLUMNS)

    labels = silver["rating"].map(label_from_rating)
    rows = []
    for idx, row in silver.iterrows():
        rows.append(
            {
                REVIEW_ID_FIELD: row["source_id"],
                REVIEW_DATE_PARTITION: _event_date_for_row(row, review_date_key),
                "label": labels.loc[idx],
            }
        )
    return pd.DataFrame(rows, columns=LABEL_STORE_COLUMNS)


def feature_store_path(gold_root: Path, review_date_key: str) -> Path:
    return gold_root / "feature_store" / review_date_partition_name(review_date_key) / GOLD_PARQUET


def label_store_path(gold_root: Path, review_date_key: str) -> Path:
    return gold_root / "label_store" / review_date_partition_name(review_date_key) / GOLD_PARQUET


def write_gold_partition(
    silver: pd.DataFrame,
    gold_root: Path,
    review_date_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write feature_store and label_store partitions for one review_date."""
    contract = silver[SILVER_COLUMNS_WITH_DATE].copy()
    features = build_feature_store(contract, review_date_key)
    labels = build_label_store(contract, review_date_key)

    fpath = feature_store_path(gold_root, review_date_key)
    lpath = label_store_path(gold_root, review_date_key)
    fpath.parent.mkdir(parents=True, exist_ok=True)
    lpath.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(fpath, index=False)
    labels.to_parquet(lpath, index=False)
    return features, labels


def process_review_dates(
    silver_root: Path,
    gold_root: Path,
    review_date_keys: Set[str],
) -> None:
    """Build Gold stores for each affected Silver review_date partition."""
    for key in sorted(review_date_keys):
        silver_path = silver_partition_path(silver_root, key)
        silver = read_silver_partition(silver_path)
        if silver.empty:
            continue
        write_gold_partition(silver[SILVER_COLUMNS_WITH_DATE], gold_root, key)


def discover_review_dates_from_silver(silver_root: Path) -> Set[str]:
    """List review_date partition keys under a Silver root."""
    keys: Set[str] = set()
    if not silver_root.exists():
        return keys
    for part in silver_root.iterdir():
        if part.is_dir() and part.name.startswith(f"{REVIEW_DATE_PARTITION}="):
            keys.add(part.name.split("=", 1)[1])
    return keys


def main() -> None:
    ap = argparse.ArgumentParser(description="Derive Gold feature/label stores from Silver.")
    ap.add_argument("--silver-root", type=Path, default=None, help="Silver partitioned root.")
    ap.add_argument("--gold-root", type=Path, default=None, help="Gold partitioned root.")
    ap.add_argument("--review-date", type=str, default=None, help="Single review_date partition to build.")
    ap.add_argument("--silver", type=Path, default=None, help="Legacy input Silver CSV path.")
    ap.add_argument("--out", type=Path, default=None, help="Legacy output Gold CSV path.")
    args = ap.parse_args()

    if args.silver_root and args.gold_root:
        if args.review_date:
            keys = {args.review_date}
        else:
            keys = discover_review_dates_from_silver(args.silver_root)
        process_review_dates(args.silver_root, args.gold_root, keys)
        print(f"Gold: wrote {len(keys)} review_date partition(s) under {args.gold_root}")
        return

    if not args.silver or not args.out:
        raise SystemExit("Provide --silver + --out (legacy) or --silver-root + --gold-root.")

    silver = pd.read_csv(args.silver)
    gold = build_gold(silver)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    gold.to_csv(args.out, index=False)

    print(f"Gold: {len(gold)} rows -> {args.out}")
    print(gold["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
