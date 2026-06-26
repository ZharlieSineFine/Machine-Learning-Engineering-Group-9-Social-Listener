#Publish the local medallion to MinIO (objects) + Postgres (tables).

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from data.ingest.ingest_reviews import review_date_partition_name
from data.storage import objectstore, warehouse
from data.storage.config import DATASETS_BUCKET, PostgresConfig, S3Config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BRONZE_ROOT = ROOT / "data" / "bronze"
DEFAULT_SILVER_ROOT = ROOT / "data" / "silver" / "reviews"
DEFAULT_GOLD_ROOT = ROOT / "data" / "gold"

LAYER_PREFIX = {"bronze": "bronze", "silver": "silver/reviews", "gold": "gold"}


# ---------- read the local medallion ----------

def _read_concat(paths: Sequence[Path], columns: Optional[List[str]] = None) -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=columns) for p in paths if Path(p).exists()]
    if not frames:
        return pd.DataFrame(columns=columns or [])
    return pd.concat(frames, ignore_index=True)


def _silver_partition_files(silver_root: Path, keys: Optional[Iterable[str]]) -> List[Path]:
    if keys is None:
        return [Path(p) for p in sorted(glob.glob(str(silver_root / "review_date=*" / "part.parquet")))]
    return [silver_root / review_date_partition_name(k) / "part.parquet" for k in keys]


def read_silver(silver_root: Path, keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Concatenate Silver partitions (all, or just `keys`)."""
    return _read_concat(_silver_partition_files(silver_root, keys))


def _gold_store_files(gold_root: Path, store: str, keys: Optional[Iterable[str]]) -> List[Path]:
    if keys is None:
        return [Path(p) for p in sorted(glob.glob(str(gold_root / store / "review_date=*" / "part.parquet")))]
    return [gold_root / store / review_date_partition_name(k) / "part.parquet" for k in keys]


def read_gold(gold_root: Path, keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Join Gold feature_store + label_store on review_id -> reviews_gold rows."""
    features = _read_concat(_gold_store_files(gold_root, "feature_store", keys),
                            ["review_id", "review_date", "text"])
    labels = _read_concat(_gold_store_files(gold_root, "label_store", keys), ["review_id", "label"])
    if features.empty:
        return pd.DataFrame(columns=["review_id", "review_date", "text", "label"])
    merged = features.merge(labels, on="review_id", how="left")
    merged["text_len"] = merged["text"].astype(str).str.len()
    merged["label_source"] = "derived_from_rating"
    return merged


# ---------- publish ----------

def publish_objects(
    s3,
    *,
    bronze_root: Path = DEFAULT_BRONZE_ROOT,
    silver_root: Path = DEFAULT_SILVER_ROOT,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    layers: Sequence[str] = ("bronze", "silver", "gold"),
    bucket: str = DATASETS_BUCKET,
) -> Dict[str, int]:
    """Mirror the requested local layer trees to MinIO. Returns objects written per layer."""
    objectstore.ensure_bucket(s3, bucket)
    roots = {"bronze": bronze_root, "silver": silver_root, "gold": gold_root}
    written: Dict[str, int] = {}
    for layer in layers:
        written[layer] = objectstore.mirror_tree(s3, roots[layer], LAYER_PREFIX[layer], bucket=bucket)
    return written


def publish_tables(
    conn,
    *,
    silver_root: Path = DEFAULT_SILVER_ROOT,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Upsert Silver + Gold rows (all, or just `keys`) into Postgres. Returns rows written."""
    warehouse.ensure_schema(conn)
    silver_df = read_silver(silver_root, keys)
    gold_df = read_gold(gold_root, keys)
    return {
        "reviews_silver": warehouse.upsert_silver(conn, silver_df),
        "reviews_gold": warehouse.upsert_gold(conn, gold_df),
    }


def publish_run(
    affected_keys: Iterable[str],
    run_date: str,
    *,
    bronze_root: Path = DEFAULT_BRONZE_ROOT,
    silver_root: Path = DEFAULT_SILVER_ROOT,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    env=None,
) -> Dict[str, object]:
    """Publish just the partitions a daily run touched (called by run_daily --publish).

    No-op-with-warning when the env isn't configured for Postgres/MinIO, so a daily run
    without services still succeeds locally.
    """
    keys = sorted(set(affected_keys))
    summary: Dict[str, object] = {"published_minio": None, "published_postgres": None}

    s3_cfg = S3Config.from_env(env)
    if s3_cfg is not None:
        s3 = objectstore.make_client(s3_cfg)
        # Mirror the run's bronze ingestion dir + the affected silver/gold partitions.
        objectstore.ensure_bucket(s3)
        counts = {"bronze": objectstore.mirror_tree(s3, bronze_root, LAYER_PREFIX["bronze"])}
        for k in keys:
            objectstore.mirror_tree(s3, silver_root / review_date_partition_name(k),
                                    f"{LAYER_PREFIX['silver']}/{review_date_partition_name(k)}")
            for store in ("feature_store", "label_store"):
                objectstore.mirror_tree(
                    s3, gold_root / store / review_date_partition_name(k),
                    f"gold/{store}/{review_date_partition_name(k)}",
                )
        summary["published_minio"] = counts
    else:
        print("[publish] MinIO env not set (AWS_*/MLFLOW_S3_ENDPOINT_URL) — skipping object publish")

    pg_cfg = PostgresConfig.from_env(env)
    if pg_cfg is not None:
        with warehouse.connection(pg_cfg) as conn:
            summary["published_postgres"] = publish_tables(
                conn, silver_root=silver_root, gold_root=gold_root, keys=keys
            )
    else:
        print("[publish] Postgres env not set (POSTGRES_*) — skipping table publish")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish the local medallion to MinIO + Postgres.")
    ap.add_argument("--bronze-root", type=Path, default=DEFAULT_BRONZE_ROOT)
    ap.add_argument("--silver-root", type=Path, default=DEFAULT_SILVER_ROOT)
    ap.add_argument("--gold-root", type=Path, default=DEFAULT_GOLD_ROOT)
    ap.add_argument("--layers", nargs="+", default=["bronze", "silver", "gold"],
                    choices=["bronze", "silver", "gold"])
    ap.add_argument("--to", choices=["minio", "db", "all"], default="all")
    args = ap.parse_args()

    if args.to in ("minio", "all"):
        s3_cfg = S3Config.from_env()
        if s3_cfg is None:
            raise SystemExit("MinIO env not set: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / MLFLOW_S3_ENDPOINT_URL")
        s3 = objectstore.make_client(s3_cfg)
        written = publish_objects(
            s3, bronze_root=args.bronze_root, silver_root=args.silver_root,
            gold_root=args.gold_root, layers=args.layers,
        )
        print(f"MinIO objects written: {written}")

    if args.to in ("db", "all"):
        pg_cfg = PostgresConfig.from_env()
        if pg_cfg is None:
            raise SystemExit("Postgres env not set: POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB")
        with warehouse.connection(pg_cfg) as conn:
            rows = publish_tables(conn, silver_root=args.silver_root, gold_root=args.gold_root)
            print(f"Postgres rows upserted: {rows}")
            print(f"Totals: reviews_silver={warehouse.table_count(conn, 'reviews_silver')}, "
                  f"reviews_gold={warehouse.table_count(conn, 'reviews_gold')}")


if __name__ == "__main__":
    main()
