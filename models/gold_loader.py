#Gold -> training-frame assembler: join the feature + label stores into a flat text/label/review_date frame.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingest.ingest_reviews import (  # noqa: E402
    FEATURE_STORE_COLUMNS,
    LABEL_STORE_COLUMNS,
    REVIEW_DATE_PARTITION,
    REVIEW_ID_FIELD,
)

DEFAULT_GOLD_ROOT = ROOT / "data" / "gold"
DEFAULT_FALLBACK_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"

TRAINING_COLUMNS = ["text", "label", REVIEW_DATE_PARTITION]

_FEATURE_GLOB = f"feature_store/{REVIEW_DATE_PARTITION}=*/part.parquet"
_LABEL_GLOB = f"label_store/{REVIEW_DATE_PARTITION}=*/part.parquet"


def _read_store(gold_root: Path, glob: str, columns: list[str]) -> pd.DataFrame:
    #Concat every parquet partition of one Gold store into a single frame.
    parts = sorted(gold_root.glob(glob))
    frames = [pd.read_parquet(p) for p in parts]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def load_gold_training_frame(
    gold_root: Path = DEFAULT_GOLD_ROOT,
    fallback_csv: Optional[Path] = DEFAULT_FALLBACK_CSV,
) -> pd.DataFrame:
    #Join the Gold feature + label stores into a flat training frame; falls back to the sample CSV when Gold is empty (unless fallback_csv=None).
    gold_root = Path(gold_root)
    features = _read_store(gold_root, _FEATURE_GLOB, FEATURE_STORE_COLUMNS)
    labels = _read_store(gold_root, _LABEL_GLOB, LABEL_STORE_COLUMNS)

    if features.empty or labels.empty:
        if fallback_csv is not None:
            print(
                f"[gold_loader] Gold store empty under {gold_root} — "
                f"falling back to sample CSV {fallback_csv}"
            )
            return _load_fallback(Path(fallback_csv))
        raise ValueError(
            f"Gold store under {gold_root} is empty and no fallback_csv given. "
            f"Run the medallion pipeline first, or point gold_root at populated Gold."
        )

    merged = features.merge(
        labels[[REVIEW_ID_FIELD, "label"]],
        on=REVIEW_ID_FIELD,
        how="inner",
    )
    merged = merged.dropna(subset=["text", "label"])
    if merged.empty:
        raise ValueError(
            "Gold feature/label stores have no overlapping review_id rows after join. "
            "Check that build_gold wrote both stores for the same partitions."
        )

    print(
        f"[gold_loader] loaded {len(merged)} rows from Gold "
        f"({len(features)} features x {len(labels)} labels) under {gold_root}"
    )
    return merged[TRAINING_COLUMNS].reset_index(drop=True)


def _load_fallback(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Sample CSV has text + label; review_date may be absent (date-less seed data).
    if REVIEW_DATE_PARTITION not in df.columns:
        df[REVIEW_DATE_PARTITION] = pd.NaT
    keep = [c for c in TRAINING_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


def materialize_training_csv(
    out_path: Path,
    gold_root: Path = DEFAULT_GOLD_ROOT,
    fallback_csv: Optional[Path] = DEFAULT_FALLBACK_CSV,
) -> Path:
    #Load the Gold training frame and write it to a CSV that train.run can read; returns the path.
    df = load_gold_training_frame(gold_root, fallback_csv)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[gold_loader] materialized {len(df)} training rows -> {out_path}")
    return out_path
