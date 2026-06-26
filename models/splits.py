#Train / validation / test / out-of-time (OOT) split for the review datasets.
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

DATE_COLUMN = "date"
LABEL_COLUMN = "label"


@dataclass(frozen=True)
class DataSplit:
    #The four disjoint frames produced by train_val_test_oot_split.
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    oot: pd.DataFrame
    cutoff_date: Optional[pd.Timestamp]

    def summary(self) -> dict:
        #Per-split row counts and date ranges, for logging / MLflow params.
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


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    return df.iloc[0:0].copy()


def _stratify_values(frame: pd.DataFrame, stratify_col: Optional[str]):
    #Return a usable stratify vector, or None when stratification isn't safe.
    if not stratify_col or stratify_col not in frame.columns:
        return None
    counts = frame[stratify_col].value_counts(dropna=False)
    if len(counts) < 2 or counts.min() < 2:
        return None
    return frame[stratify_col]


def _split_two(frame: pd.DataFrame, test_size: float, stratify_col: Optional[str], seed: int):
    #Split frame into (larger, smaller) with test_size going to the second; robust to tiny inputs.
    if test_size <= 0 or len(frame) < 2:
        return frame, _empty_like(frame)
    # Need at least one row on each side.
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
    #Split df into (in_time, oot) by holding out the most recent oot_frac by date; returns (in_time, oot, cutoff_date).
    if oot_frac <= 0 or date_col not in df.columns:
        return df, _empty_like(df), None

    parsed = pd.to_datetime(df[date_col], errors="coerce")
    dated_mask = parsed.notna()
    n_dated = int(dated_mask.sum())
    if n_dated < 2:
        return df, _empty_like(df), None

    n_oot = max(1, int(round(n_dated * oot_frac)))
    n_oot = min(n_oot, n_dated - 1)  # keep at least one dated row in-time
    sorted_dates = parsed[dated_mask].sort_values()
    cutoff_date = sorted_dates.iloc[n_dated - n_oot]

    oot_mask = dated_mask & (parsed >= cutoff_date)
    if oot_mask.all():  # degenerate — no usable in-time pool
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
    #Split df into train / validation / test / OOT frames; deterministic for a given seed.
    if not (0.0 <= oot_frac < 1.0):
        raise ValueError(f"oot_frac must be in [0, 1), got {oot_frac}")
    if val_frac < 0 or test_frac < 0 or (val_frac + test_frac) >= 1.0:
        raise ValueError(f"val_frac + test_frac must be < 1, got {val_frac} + {test_frac}")

    work = df.reset_index(drop=True)

    in_time, oot, cutoff = _carve_oot(work, date_col, oot_frac)

    # Carve test off the in-time pool, then validation off what's left. `val_frac` and
    # `test_frac` are defined relative to the in-time pool, so val is rescaled after test
    # has been removed.
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
    #Concatenate one or more Silver CSVs into a single frame.
    frames = [pd.read_csv(p) for p in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
