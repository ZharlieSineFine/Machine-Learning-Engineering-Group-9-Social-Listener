"""Gold training split utilities — shared contract for notebooks and production.

Implements the split strategy from `notebooks/TUNING_NOTEBOOKS_INSTRUCTIONS.md`
so offline tuning and future `train_model` DAG runs use the same partitions.

Input contract (from Charlie/Ha `reviews_gold` export):
    - Required columns: ``text``, ``label``, ``review_date``
    - ``label`` values: ``negative`` | ``neutral`` | ``positive``
    - ``review_date`` parseable as datetime

Owner: Van (Modeler). Consumed by training scripts once gold is wired.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split

SplitName = Literal["train", "val", "test", "oot", "demo"]

# Cutoffs from notebook 01/02 on the 50k gold export (2021-05-02 → 2022-04-10).
OOT_CUTOFF = "2021-12-11"
DEMO_CUTOFF = "2022-01-09"
SEED = 42

REQUIRED_COLUMNS = frozenset({"text", "label", "review_date"})


@dataclass(frozen=True)
class GoldSplits:
    """Named partitions for model training and evaluation."""

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


def _validate_gold_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"gold DataFrame missing required columns: {sorted(missing)}. "
            f"Expected {sorted(REQUIRED_COLUMNS)}."
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
