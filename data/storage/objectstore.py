"""MinIO / S3 object store for the medallion `datasets` bucket.

Mirrors the local medallion trees to S3 under stable prefixes:

    data/bronze/<source>/dt=.../{reviews,business}.csv  -> s3://datasets/bronze/<source>/dt=.../...
    data/silver/reviews/review_date=.../part.parquet     -> s3://datasets/silver/reviews/review_date=.../...
    data/gold/{feature_store,label_store}/review_date=... -> s3://datasets/gold/...

Keys mirror the local relative paths, so the S3 layout matches the on-disk Hive partitions.

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from data.storage.config import DATASETS_BUCKET, S3Config


def make_client(cfg: S3Config):
    """Build a boto3 S3 client pointed at MinIO (imported lazily)."""
    import boto3
    from botocore.client import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key,
        aws_secret_access_key=cfg.secret_key,
        region_name=cfg.region,
        config=BotoConfig(signature_version="s3v4"),
    )


def ensure_bucket(client, bucket: str = DATASETS_BUCKET) -> None:
    """Create the bucket if it isn't there yet (idempotent; tolerates races)."""
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
        return
    except ClientError:
        pass
    try:
        client.create_bucket(Bucket=bucket)
    except ClientError:
        # Already created concurrently, or owned — fine for a local prototype.
        pass


def to_key(*parts: str) -> str:
    """Join path parts into a forward-slash S3 key (Windows-safe)."""
    return "/".join(p.strip("/") for p in parts if p)


def mirror_tree(
    client,
    local_root: Path,
    key_prefix: str,
    bucket: str = DATASETS_BUCKET,
) -> int:
    """Upload every file under `local_root` to `bucket/key_prefix/<relpath>`.

    Returns the number of objects written. Overwrites by key, so it's idempotent.
    """
    local_root = Path(local_root)
    if not local_root.exists():
        return 0
    n = 0
    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(local_root).as_posix()
            client.upload_file(str(path), bucket, to_key(key_prefix, rel))
            n += 1
    return n


def list_keys(client, prefix: str, bucket: str = DATASETS_BUCKET) -> List[str]:
    """List all object keys under a prefix (paginated)."""
    keys: List[str] = []
    token: Optional[str] = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return keys
