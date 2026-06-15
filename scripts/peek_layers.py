"""Inspect Silver and Gold parquet partitions without re-running the pipeline.

Prints partition inventory, total row counts, column dtypes, and sample rows
from the most recent review_date folders.

Run:
    python scripts/peek_layers.py
    python scripts/peek_layers.py --latest 3 --sample-rows 2
    python scripts/peek_layers.py --silver-root data/silver/reviews --gold-root data/gold
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingest.ingest_reviews import (  # noqa: E402
    FEATURE_STORE_COLUMNS,
    LABEL_STORE_COLUMNS,
    REVIEW_DATE_PARTITION,
    SILVER_COLUMNS_WITH_DATE,
)
from data.refine.build_gold import feature_store_path, label_store_path  # noqa: E402
from data.refine.build_silver import silver_partition_path  # noqa: E402

DEFAULT_SILVER_ROOT = ROOT / "data" / "silver" / "reviews"
DEFAULT_GOLD_ROOT = ROOT / "data" / "gold"
PARQUET = "part.parquet"
NULL_KEY = "__null__"


def discover_partition_keys(root: Path) -> List[str]:
    """List ``review_date`` keys under a partitioned root (sorted chronologically)."""
    if not root.exists():
        return []
    keys: List[str] = []
    prefix = f"{REVIEW_DATE_PARTITION}="
    for part in root.iterdir():
        if part.is_dir() and part.name.startswith(prefix):
            keys.append(part.name.split("=", 1)[1])
    dated = sorted(k for k in keys if k != NULL_KEY)
    if NULL_KEY in keys:
        dated.append(NULL_KEY)
    return dated


def _count_parquet_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return len(pd.read_parquet(path))


def _partition_summary(root: Path, keys: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        path = root / f"{REVIEW_DATE_PARTITION}={key}" / PARQUET
        counts[key] = _count_parquet_rows(path)
    return counts


def _print_schema(title: str, df: pd.DataFrame, expected: Optional[Sequence[str]] = None) -> None:
    print(f"  {title}")
    print(f"    columns: {list(df.columns)}")
    if expected is not None:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        if missing:
            print(f"    missing expected: {missing}")
        if extra:
            print(f"    extra columns: {extra}")
    for col, dtype in df.dtypes.items():
        print(f"    {col}: {dtype}")


def _peek_file(path: Path, *, sample_rows: int, expected: Optional[Sequence[str]] = None) -> int:
    if not path.exists():
        print(f"  (missing) {path}")
        return 0
    df = pd.read_parquet(path)
    n = len(df)
    print(f"  path: {path}")
    print(f"  rows: {n}")
    _print_schema("schema:", df, expected)
    if sample_rows and not df.empty:
        print(f"  sample ({min(sample_rows, n)} row(s)):")
        print(df.head(sample_rows).to_string(index=False))
    return n


def _print_layer_summary(
    layer: str,
    root: Path,
    keys: Sequence[str],
    counts: dict[str, int],
) -> None:
    total = sum(counts.values())
    print(f"=== {layer} ===")
    print(f"  root: {root}")
    if not keys:
        print("  partitions: 0")
        print("  total rows: 0")
        return
    print(f"  partitions: {len(keys)}  ({keys[0]} .. {keys[-1]})")
    print(f"  total rows: {total:,}")
    print()


def peek_silver(
    silver_root: Path,
    *,
    latest: int,
    sample_rows: int,
    verbose: bool,
) -> int:
    keys = discover_partition_keys(silver_root)
    counts = _partition_summary(silver_root, keys)
    _print_layer_summary("Silver", silver_root, keys, counts)

    if not keys:
        return 0

    if verbose:
        print("  per-partition row counts:")
        for key in keys:
            print(f"    review_date={key}: {counts[key]:,}")
        print()

    print(f"--- Silver detail (latest {min(latest, len(keys))} partition(s)) ---")
    for key in keys[-latest:]:
        print(f"\nreview_date={key}")
        path = silver_partition_path(silver_root, key)
        # Silver parquet also carries _ingested_at; show full file schema.
        _peek_file(path, sample_rows=sample_rows, expected=None)

    return sum(counts.values())


def peek_gold(
    gold_root: Path,
    *,
    latest: int,
    sample_rows: int,
    verbose: bool,
) -> int:
    feature_root = gold_root / "feature_store"
    label_root = gold_root / "label_store"
    f_keys = discover_partition_keys(feature_root)
    l_keys = discover_partition_keys(label_root)
    keys = sorted(set(f_keys) | set(l_keys), key=lambda k: (k == NULL_KEY, k))

    f_counts = _partition_summary(feature_root, keys)
    l_counts = _partition_summary(label_root, keys)

    print("=== Gold feature_store ===")
    print(f"  root: {feature_root}")
    print(f"  partitions: {len(f_keys)}  total rows: {sum(f_counts.values()):,}")
    print("=== Gold label_store ===")
    print(f"  root: {label_root}")
    print(f"  partitions: {len(l_keys)}  total rows: {sum(l_counts.values()):,}")
    print()

    if not keys:
        return 0

    if verbose:
        print("  per-partition row counts (feature / label):")
        for key in keys:
            print(
                f"    review_date={key}: "
                f"{f_counts.get(key, 0):,} / {l_counts.get(key, 0):,}"
            )
        print()

    print(f"--- Gold detail (latest {min(latest, len(keys))} partition(s)) ---")
    for key in keys[-latest:]:
        print(f"\nreview_date={key}")
        print("feature_store:")
        _peek_file(
            feature_store_path(gold_root, key),
            sample_rows=sample_rows,
            expected=FEATURE_STORE_COLUMNS,
        )
        print("label_store:")
        _peek_file(
            label_store_path(gold_root, key),
            sample_rows=sample_rows,
            expected=LABEL_STORE_COLUMNS,
        )

    return sum(f_counts.values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Peek at Silver/Gold parquet partitions (schema + row counts)."
    )
    ap.add_argument("--silver-root", type=Path, default=DEFAULT_SILVER_ROOT)
    ap.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    ap.add_argument(
        "--latest",
        type=int,
        default=1,
        help="Show schema/samples for the N most recent review_date partitions (default: 1).",
    )
    ap.add_argument(
        "--sample-rows",
        type=int,
        default=2,
        help="Sample rows to print per partition file (default: 2; 0 to skip).",
    )
    ap.add_argument(
        "--silver-only",
        action="store_true",
        help="Skip Gold inspection.",
    )
    ap.add_argument(
        "--gold-only",
        action="store_true",
        help="Skip Silver inspection.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-partition row counts for every review_date folder.",
    )
    args = ap.parse_args()

    if args.silver_only and args.gold_only:
        raise SystemExit("Use at most one of --silver-only / --gold-only.")

    print("Expected Silver contract columns:", [*SILVER_COLUMNS_WITH_DATE, "_ingested_at"])
    print("Expected Gold feature_store:", list(FEATURE_STORE_COLUMNS))
    print("Expected Gold label_store:", list(LABEL_STORE_COLUMNS))
    print()

    silver_total = 0
    gold_total = 0
    if not args.gold_only:
        silver_total = peek_silver(
            args.silver_root,
            latest=max(1, args.latest),
            sample_rows=max(0, args.sample_rows),
            verbose=args.verbose,
        )
        print()
    if not args.silver_only:
        gold_total = peek_gold(
            args.gold_root,
            latest=max(1, args.latest),
            sample_rows=max(0, args.sample_rows),
            verbose=args.verbose,
        )

    print("=== Summary ===")
    if not args.gold_only:
        print(f"  Silver total rows: {silver_total:,}")
    if not args.silver_only:
        print(f"  Gold feature_store total rows: {gold_total:,}")


if __name__ == "__main__":
    main()
