"""Train / validation / test / out-of-time (OOT) split utilities.

Two split contracts share this module:

1. **Silver / sample CSV** — :func:`train_val_test_oot_split` holds out the most recent
   slice of *dated* rows (Silver ``date`` column) as OOT, then stratifies the older
   in-time pool into train / val / test. Used by ``baseline_sklearn.train()`` on the
   seed CSV and combined Silver exports.

2. **Gold export** — :func:`split_gold` partitions ``reviews_gold`` by fixed
   ``review_date`` cutoffs (train/val/test/oot/demo) matching notebooks 01/02.

Owner: Van (Modeler) with Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

SplitName = Literal["train", "val", "test", "oot", "demo"]

# Cutoffs from notebook 01/02 on the 50k gold export (2021-05-02 → 2022-04-10).
OOT_CUTOFF = "2021-12-11"
DEMO_CUTOFF = "2022-01-09"
SEED = 42

DATE_COLUMN = "date"
LABEL_COLUMN = "label"
GOLD_REQUIRED_COLUMNS = frozenset({"text", "label", "review_date"})


@dataclass(frozen=True)
class GoldSplits:
    """Named partitions for gold-layer model training and evaluation."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    demo: pd.DataFrame

    def as_dict(self) -> dict[SplitName, pd.DataFrame]:
        return {
            "train": self.train,
            "val": self.val,
            "test": self.test,
            "oot": self.oot,
            "demo": self.demo,
        }

    def summary(self) -> pd.DataFrame:
        """Row counts and label proportions per split."""
        rows = []
        for name, split in self.as_dict().items():
            if split.empty:
                rows.append({"split": name, "n": 0, "label_dist": {}})
                continue
            rows.append({
                "split": name,
                "n": len(split),
                "label_dist": split["label"].value_counts(normalize=True).round(3).to_dict(),
            })
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class DataSplit:
    """The four disjoint frames produced by :func:`train_val_test_oot_split`.

    ``cutoff_date`` is the first (earliest) date in the OOT hold-out — everything on or
    after it is OOT. It is None when there is no OOT set (no usable dates).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    cutoff_date: Optional[pd.Timestamp]

    def summary(self) -> dict:
        """Per-split row counts and date ranges — handy for logging / MLflow params."""

        def _range(frame: pd.DataFrame) -> Optional[tuple]:
            if frame.empty or DATE_COLUMN not in frame.columns:
                return None
            dates = pd.to_datetime(frame[DATE_COLUMN], errors="coerce").dropna()
            if dates.empty:
                return None
            return (str(dates.min()), str(dates.max()))

        return {
            "n_train": len(self.train),
            "n_val": len(self.val),
            "n_test": len(self.test),
            "n_oot": len(self.oot),
            "cutoff_date": None if self.cutoff_date is None else str(self.cutoff_date),
            "train_dates": _range(self.train),
            "test_dates": _range(self.test),
            "oot_dates": _range(self.oot),
        }


def _validate_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = GOLD_REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"gold DataFrame missing required columns: {sorted(missing)}. "
            f"Expected {sorted(GOLD_REQUIRED_COLUMNS)}."
        )
    out = df.copy()
    out["review_date"] = pd.to_datetime(out["review_date"])
    return out.sort_values("review_date").reset_index(drop=True)


def split_gold(
    df: pd.DataFrame,
    *,
    oot_cutoff: str = OOT_CUTOFF,
    demo_cutoff: str = DEMO_CUTOFF,
    seed: int = SEED,
) -> GoldSplits:
    """Partition gold reviews into train/val/test/oot/demo.

    Rules (match notebooks 01/02):
        - ``demo``: ``review_date >= demo_cutoff`` (replay simulator only)
        - ``oot``: ``oot_cutoff <= review_date < demo_cutoff`` (temporal holdout)
        - ``train``/``val``/``test``: random stratified 80/10/10 from pre-OOT pool
    """
    df = _validate_gold_df(df)
    oot_cutoff_ts = pd.Timestamp(oot_cutoff)
    demo_cutoff_ts = pd.Timestamp(demo_cutoff)

    demo_df = df[df["review_date"] >= demo_cutoff_ts].copy()
    oot_df = df[(df["review_date"] >= oot_cutoff_ts) & (df["review_date"] < demo_cutoff_ts)].copy()
    rest_df = df[df["review_date"] < oot_cutoff_ts].copy()

    if rest_df.empty:
        raise ValueError(
            "No rows before OOT cutoff for train/val/test. "
            f"Check review_date range vs oot_cutoff={oot_cutoff!r}."
        )

    rest_df = rest_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    train_df, temp_df = train_test_split(
        rest_df,
        test_size=0.2,
        stratify=rest_df["label"],
        random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df["label"],
        random_state=seed,
    )

    return GoldSplits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        oot=oot_df.reset_index(drop=True),
        demo=demo_df.reset_index(drop=True),
    )


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _stratify_values(frame: pd.DataFrame, stratify_col: Optional[str]):
    """Return a usable stratify vector, or None when stratification isn't safe."""
    if not stratify_col or stratify_col not in frame.columns:
        return None
    counts = frame[stratify_col].value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return frame[stratify_col]


def _split_two(frame: pd.DataFrame, test_size: float, stratify_col: Optional[str], seed: int):
    """Split ``frame`` into (larger, smaller) with ``test_size`` going to the second."""
    if test_size <= 0 or len(frame) < 2:
        return frame, _empty_like(frame)
    n_test = int(round(len(frame) * test_size))
    if n_test < 1 or n_test >= len(frame):
        return frame, _empty_like(frame)

    stratify = _stratify_values(frame, stratify_col)
    try:
        a, b = train_test_split(frame, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        a, b = train_test_split(frame, test_size=test_size, random_state=seed, stratify=None)
    return a, b


def _carve_oot(df: pd.DataFrame, date_col: str, oot_frac: float):
    """Split ``df`` into (in_time, oot) by holding out the most recent ``oot_frac`` by date."""
    if oot_frac <= 0 or date_col not in df.columns:
        return df, _empty_like(df), None

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    dated_mask = parsed.notna()
    n_dated = int(dated_mask.sum())
    if n_dated < 2:
        return df, _empty_like(df), None

    n_oot = max(1, int(round(n_dated * oot_frac)))
    n_oot = min(n_oot, n_dated - 1)
    sorted_dates = parsed[dated_mask].sort_values()
    cutoff_date = sorted_dates.iloc[n_dated - n_oot]

    oot_mask = dated_mask & (parsed >= cutoff_date)
    if oot_mask.all():
        return df, _empty_like(df), None

    return df[~oot_mask].copy(), df[oot_mask].copy(), cutoff_date


def train_val_test_oot_split(
    df: pd.DataFrame,
    *,
    date_col: str = DATE_COLUMN,
    stratify_col: Optional[str] = LABEL_COLUMN,
    oot_frac: float = 0.2,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> DataSplit:
    """Split ``df`` into train / validation / test / OOT frames.

    Fractions:
        - ``oot_frac``  — share of the *dated* rows (most recent, by time) held out as OOT.
        - ``val_frac``  — share of the *in-time pool* used for validation.
        - ``test_frac`` — share of the *in-time pool* used for test.
        - train         — whatever remains of the in-time pool.

    Rows whose ``date`` is null join the in-time pool. If no row has a usable date, OOT
    is empty and this degrades to a plain stratified train/val/test split.
    """
    if not (0.0 <= oot_frac < 1.0):
        raise ValueError(f"oot_frac must be in [0, 1), got {oot_frac}")
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac} + {test_frac}")

    work = df.reset_index(drop=True)
    in_time, oot, cutoff = _carve_oot(work, date_col, oot_frac)

    rest, test = _split_two(in_time, test_frac, stratify_col, seed)
    val_rel = 0.0 if (1.0 - test_frac) <= 0 else val_frac / (1.0 - test_frac)
    train, val = _split_two(rest, val_rel, stratify_col, seed)

    reset = lambda f: f.reset_index(drop=True)
    return DataSplit(
        train=reset(train),
        val=reset(val),
        test=reset(test),
        oot=reset(oot),
        cutoff_date=cutoff,
    )


def read_dataset(paths: List[str]) -> pd.DataFrame:
    """Concatenate one or more Silver CSVs (contract + ISO date) into a single frame."""
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
