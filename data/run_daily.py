"""Daily incremental driver for the review medallion pipeline.

Runs bronze landing -> silver refinement -> GE gate -> gold stores for a single
ingestion date. Each layer overwrites only its own partition(s), so re-running the
same ``--run-date`` is idempotent.

    python -m data.run_daily --run-date 2026-06-06 --sources yelp tripadvisor
    python -m data.run_daily --run-date 2026-06-06 --skip-bronze --all-years  # full history

Bronze keeps the full archive; Silver defaults to the last 3 years per source.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence

from data.expectations.reviews_suite import validate_silver
from data.ingest.ingest_reviews import DEFAULT_SILVER_RECENT_YEARS, SILVER_COLUMNS_WITH_DATE
from data.paths import tripadvisor_csv_path, yelp_business_path, yelp_reviews_path, yelp_tar_path
from data.refine.build_gold import process_review_dates
from data.refine.build_silver import (
    read_silver_partition,
    process_ingestion_to_silver,
    silver_partition_path,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRONZE_ROOT = ROOT / "data" / "bronze"
DEFAULT_SILVER_ROOT = ROOT / "data" / "silver" / "reviews"
DEFAULT_GOLD_ROOT = ROOT / "data" / "gold"


class DailyRunError(Exception):
    """Raised when a pipeline stage fails (e.g. GE gate)."""


def validate_silver_partitions(
    silver_root: Path,
    review_date_keys: Sequence[str],
    *,
    check_language: bool = False,
) -> None:
    """Run ``validate_silver`` on each affected partition; fail the run on violation."""
    failures: List[str] = []
    for key in review_date_keys:
        path = silver_partition_path(silver_root, key)
        silver = read_silver_partition(path)
        if silver.empty:
            continue
        result = validate_silver(silver[SILVER_COLUMNS_WITH_DATE], check_language=check_language)
        if not result.success:
            failures.append(f"review_date={key}: {result.failures}")
    if failures:
        raise DailyRunError("Silver GE gate failed:\n" + "\n".join(failures))


def run_daily(
    run_date: str,
    sources: Sequence[str],
    *,
    bronze_root: Path = DEFAULT_BRONZE_ROOT,
    silver_root: Path = DEFAULT_SILVER_ROOT,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    skip_bronze: bool = False,
    skip_ge: bool = False,
    location_with_state: bool = False,
    recent_years: Optional[float] = DEFAULT_SILVER_RECENT_YEARS,
    publish: bool = False,
) -> dict:
    """Execute the daily pipeline for one ingestion date.

  Returns a summary dict with affected review_date keys and row counts.
    """
    if not skip_bronze:
        _run_bronze(run_date, sources, bronze_root)

    affected = process_ingestion_to_silver(
        bronze_root,
        silver_root,
        [run_date],
        sources,
        location_with_state=location_with_state,
        recent_years=recent_years,
    )

    if not skip_ge and affected:
        validate_silver_partitions(silver_root, sorted(affected))

    if affected:
        process_review_dates(silver_root, gold_root, affected)

    publish_summary = None
    if publish and affected:
        from data.publish import publish_run  # lazy: boto3/psycopg2 only needed with --publish

        publish_summary = publish_run(
            sorted(affected),
            run_date,
            bronze_root=bronze_root,
            silver_root=silver_root,
            gold_root=gold_root,
        )

    counts = {}
    for key in affected:
        silver = read_silver_partition(silver_partition_path(silver_root, key))
        counts[key] = len(silver)

    total_silver = sum(counts.values())
    return {
        "run_date": run_date,
        "sources": list(sources),
        "review_dates": sorted(affected),
        "silver_row_counts": counts,
        "total_silver_rows": total_silver,
        "recent_years": recent_years,
        "publish": publish_summary,
    }


def _run_bronze(run_date: str, sources: Sequence[str], bronze_root: Path) -> None:
    """Invoke source adapters to land bronze under ``dt=<run_date>``."""
    for source in sources:
        if source == "yelp":
            _run_yelp_bronze(run_date, bronze_root)
        elif source == "tripadvisor":
            _run_tripadvisor_bronze(run_date, bronze_root)
        else:
            raise DailyRunError(f"unsupported source for daily bronze: {source}")


def _warn_missing_env(name: str, raw: str) -> None:
    print(f"Bronze ingest: ignoring {name}={raw!r} (file not found)")


def _run_yelp_bronze(run_date: str, bronze_root: Path) -> None:
    """Land Yelp bronze if source data is configured (no-op when paths missing)."""
    import os

    tar = yelp_tar_path()
    if os.environ.get("YELP_TAR_PATH") and tar is None:
        _warn_missing_env("YELP_TAR_PATH", os.environ["YELP_TAR_PATH"])
    reviews = yelp_reviews_path()
    if os.environ.get("YELP_REVIEWS_PATH") and reviews is None:
        _warn_missing_env("YELP_REVIEWS_PATH", os.environ["YELP_REVIEWS_PATH"])
    business = yelp_business_path()
    if os.environ.get("YELP_BUSINESS_PATH") and business is None:
        _warn_missing_env("YELP_BUSINESS_PATH", os.environ["YELP_BUSINESS_PATH"])
    out_dir = bronze_root / "yelp"

    if tar:
        from data.ingest.yelp_loader import extract_bronze_from_tar, write_bronze_partition

        reviews_df, business_df = extract_bronze_from_tar(tar)
        write_bronze_partition(reviews_df, business_df, out_dir, run_date)
        return

    if reviews and business:
        from data.ingest.yelp_loader import extract_bronze_from_paths, write_bronze_partition

        reviews_df, business_df = extract_bronze_from_paths(reviews, business)
        write_bronze_partition(reviews_df, business_df, out_dir, run_date)
        return

    # Tests and local dev may pre-seed bronze; skip ingest when no config.
    part = out_dir / f"dt={run_date}"
    if not (part / "reviews.csv").exists():
        print(
            "Yelp bronze: skipped (set YELP_TAR_PATH or YELP_REVIEWS_PATH+YELP_BUSINESS_PATH, "
            "or place reviews.csv under data/bronze/yelp/)"
        )


def _run_tripadvisor_bronze(run_date: str, bronze_root: Path) -> None:
    import os

    csv_path = tripadvisor_csv_path()
    if os.environ.get("TRIPADVISOR_CSV_PATH") and csv_path is None:
        _warn_missing_env("TRIPADVISOR_CSV_PATH", os.environ["TRIPADVISOR_CSV_PATH"])
    out_dir = bronze_root / "tripadvisor"

    if csv_path:
        from data.ingest.malaysia_review_loader import load_bronze, write_bronze_partition

        df = load_bronze(csv_path)
        write_bronze_partition(df, out_dir, run_date)
        return

    part = out_dir / f"dt={run_date}"
    if not (part / "reviews.csv").exists():
        print(
            "TripAdvisor bronze: skipped (set TRIPADVISOR_CSV_PATH or place reviews.csv "
            "under data/bronze/tripadvisor/)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the daily review medallion pipeline.")
    ap.add_argument(
        "--run-date",
        type=str,
        default=date.today().isoformat(),
        help="Ingestion date (YYYY-MM-DD); default today.",
    )
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["yelp", "tripadvisor"],
        help="Sources to process.",
    )
    ap.add_argument("--bronze-root", type=Path, default=DEFAULT_BRONZE_ROOT)
    ap.add_argument("--silver-root", type=Path, default=DEFAULT_SILVER_ROOT)
    ap.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    ap.add_argument("--skip-bronze", action="store_true", help="Assume bronze already landed.")
    ap.add_argument("--skip-ge", action="store_true", help="Skip the Silver GE gate.")
    ap.add_argument("--location-with-state", action="store_true")
    ap.add_argument(
        "--recent-years",
        type=float,
        default=DEFAULT_SILVER_RECENT_YEARS,
        help=f"Silver scope: last N years per source (default: {DEFAULT_SILVER_RECENT_YEARS}).",
    )
    ap.add_argument(
        "--all-years",
        action="store_true",
        help="Keep full Bronze history in Silver (disable per-source recency filter).",
    )
    ap.add_argument(
        "--publish",
        action="store_true",
        help="After building, publish affected partitions to MinIO + Postgres (needs env configured).",
    )
    args = ap.parse_args()
    recent_years = None if args.all_years else args.recent_years

    try:
        summary = run_daily(
            args.run_date,
            args.sources,
            bronze_root=args.bronze_root,
            silver_root=args.silver_root,
            gold_root=args.gold_root,
            skip_bronze=args.skip_bronze,
            skip_ge=args.skip_ge,
            location_with_state=args.location_with_state,
            recent_years=recent_years,
            publish=args.publish,
        )
    except DailyRunError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    scope = "all years" if summary["recent_years"] is None else f"last {summary['recent_years']}y per source"
    print(
        f"Daily run {summary['run_date']} ({scope}): "
        f"{summary['total_silver_rows']} silver rows across {len(summary['review_dates'])} review_date partition(s)"
    )
    for key, n in summary["silver_row_counts"].items():
        print(f"  review_date={key}: {n} silver rows")
    if summary.get("publish"):
        print(f"Published: {summary['publish']}")


if __name__ == "__main__":
    main()
