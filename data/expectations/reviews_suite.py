"""Great Expectations gates for Silver and Gold review tables.

* `validate_silver` — Bronze → Silver gate (no `label` column).
* `validate_reviews` — Silver → Gold / training gate (requires `label`).

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from data.ingest.ingest_reviews import EXPECTED_COLUMNS, SILVER_COLUMNS_WITH_DATE, SOURCE_ID_FIELD

VALID_SOURCES = frozenset({"yelp", "malaysia", "replay", "google", "tripadvisor"})
VALID_LABELS = frozenset({"positive", "neutral", "negative"})


@dataclass
class ValidationResult:
    success: bool
    failures: List[str] = field(default_factory=list)


def validate_silver(df: pd.DataFrame, *, check_language: bool = True) -> ValidationResult:
    """Run the Phase-1 expectation suite on a Silver-shaped DataFrame."""
    from great_expectations.dataset import PandasDataset

    missing = [c for c in SILVER_COLUMNS_WITH_DATE if c not in df.columns]
    if missing:
        return ValidationResult(False, [f"missing columns: {missing}"])
    if "label" in df.columns:
        return ValidationResult(False, ["silver must not contain a label column"])

    contract = df[SILVER_COLUMNS_WITH_DATE].copy()
    ds = PandasDataset(contract)
    failures: List[str] = []

    def _run(name: str, result) -> None:
        if not result.success:
            failures.append(f"{name}: {result.result}")

    _run("text not null", ds.expect_column_values_to_not_be_null("text"))
    _run(
        "text length 1–5000",
        ds.expect_column_value_lengths_to_be_between("text", min_value=1, max_value=5000),
    )
    _run("text_len not null", ds.expect_column_values_to_not_be_null("text_len"))
    _run(
        "text_len 1–5000",
        ds.expect_column_values_to_be_between("text_len", min_value=1, max_value=5000),
    )
    _run(
        "source in allowed set",
        ds.expect_column_values_to_be_in_set("source", value_set=sorted(VALID_SOURCES)),
    )
    _run(
        "rating in [1, 5]",
        ds.expect_column_values_to_be_between("rating", min_value=1, max_value=5),
    )
    _run("restaurant not null", ds.expect_column_values_to_not_be_null("restaurant"))
    _run("location not null", ds.expect_column_values_to_not_be_null("location"))
    _run("source_id not null", ds.expect_column_values_to_not_be_null(SOURCE_ID_FIELD))

    if check_language:
        try:
            import langdetect  # noqa: F401
        except ImportError:
            pass

    return ValidationResult(success=len(failures) == 0, failures=failures)


def validate_reviews(df: pd.DataFrame, *, check_language: bool = True) -> ValidationResult:
    """Run the Phase-1 expectation suite on a Gold / training-contract DataFrame."""
    from great_expectations.dataset import PandasDataset

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return ValidationResult(False, [f"missing columns: {missing}"])

    contract = df[list(EXPECTED_COLUMNS)].copy()
    ds = PandasDataset(contract)
    failures: List[str] = []

    def _run(name: str, result) -> None:
        if not result.success:
            failures.append(f"{name}: {result.result}")

    _run("text not null", ds.expect_column_values_to_not_be_null("text"))
    _run(
        "text length 1–5000",
        ds.expect_column_value_lengths_to_be_between("text", min_value=1, max_value=5000),
    )
    _run(
        "source in allowed set",
        ds.expect_column_values_to_be_in_set("source", value_set=sorted(VALID_SOURCES)),
    )
    _run(
        "rating in [1, 5]",
        ds.expect_column_values_to_be_between("rating", min_value=1, max_value=5),
    )
    _run(
        "label in allowed set",
        ds.expect_column_values_to_be_in_set("label", value_set=sorted(VALID_LABELS)),
    )
    _run("label not null", ds.expect_column_values_to_not_be_null("label"))
    _run("restaurant not null", ds.expect_column_values_to_not_be_null("restaurant"))
    _run("location not null", ds.expect_column_values_to_not_be_null("location"))

    present_labels = set(contract["label"].dropna().unique())
    for lbl in sorted(VALID_LABELS):
        if lbl not in present_labels:
            failures.append(f"missing label class: {lbl}")

    if check_language:
        try:
            import langdetect  # noqa: F401
        except ImportError:
            pass

    return ValidationResult(success=len(failures) == 0, failures=failures)
