"""Great Expectations gates for Silver, Gold, and Phase-1 Postgres ingest.

* ``validate_silver`` — Bronze → Silver gate (no ``label`` column).
* ``validate_reviews`` — Silver → Gold / training / Postgres gate (requires ``label``).

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from data.ingest.ingest_reviews import (
    EXPECTED_COLUMNS,
    SILVER_COLUMNS_WITH_DATE,
    SOURCE_ID_FIELD,
    VALID_LABELS,
)

REQUIRED_COLUMNS = list(EXPECTED_COLUMNS)
VALID_SOURCES = frozenset({"yelp", "malaysia", "replay", "google", "tripadvisor"})

MIN_RATING = 1.0
MAX_RATING = 5.0
MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 10_000
HAS_LETTER_RE = r"[A-Za-z\u00c0-\u024f]"
ACCEPTED_LANGS = {"en", "ms", "id"}
LANG_MIN_SHARE = 0.80
LANG_SAMPLE_SIZE = 200


@dataclass
class ValidationResult:
    """Compact summary the DAG, medallion driver, and tests all consume."""
    success: bool
    n_rows: int = 0
    failures: List[dict] = field(default_factory=list)

    def raise_for_status(self) -> None:
        if not self.success:
            raise ValueError(
                f"reviews_suite validation failed ({len(self.failures)} expectation(s) failed). "
                f"First failure: {self.failures[0] if self.failures else 'n/a'}"
            )


def _failure_dict(expectation: str, *, kwargs=None, result=None, detail: str | None = None) -> dict:
    if detail is not None:
        return {"expectation": expectation, "detail": detail}
    return {
        "expectation": expectation,
        "kwargs": kwargs,
        "result": result,
    }


def validate_silver(df: pd.DataFrame, *, check_language: bool = True) -> ValidationResult:
    """Run the Silver contract gate (no derived labels)."""
    from great_expectations.dataset import PandasDataset

    missing = [c for c in SILVER_COLUMNS_WITH_DATE if c not in df.columns]
    if missing:
        return ValidationResult(
            False,
            n_rows=len(df),
            failures=[_failure_dict("schema", detail=f"missing columns: {missing}")],
        )
    if "label" in df.columns:
        return ValidationResult(
            False,
            n_rows=len(df),
            failures=[_failure_dict("schema", detail="silver must not contain a label column")],
        )

    contract = df[SILVER_COLUMNS_WITH_DATE].copy()
    ds = PandasDataset(contract)
    failures: List[dict] = []

    def _run(name: str, result) -> None:
        if not result.success:
            failures.append(_failure_dict(name, result=result.result))

    _run("text not null", ds.expect_column_values_to_not_be_null("text"))
    _run(
        "text length 1-5000",
        ds.expect_column_value_lengths_to_be_between("text", min_value=1, max_value=5000),
    )
    _run("text_len not null", ds.expect_column_values_to_not_be_null("text_len"))
    _run(
        "text_len 1-5000",
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

    return ValidationResult(success=len(failures) == 0, n_rows=len(df), failures=failures)


def _run_expectations(df: pd.DataFrame) -> List[dict]:
    """Run Phase-1 Gold / training expectations against ``df``."""
    from great_expectations.dataset import PandasDataset

    ds = PandasDataset(df)
    results: List[dict] = []

    for col in REQUIRED_COLUMNS:
        results.append(ds.expect_column_to_exist(col).to_json_dict())

    for col in ["text", "label", "source"]:
        if col in df.columns:
            results.append(ds.expect_column_values_to_not_be_null(col).to_json_dict())

    if "label" in df.columns:
        results.append(
            ds.expect_column_values_to_be_in_set("label", sorted(VALID_LABELS)).to_json_dict()
        )

    if "rating" in df.columns:
        results.append(
            ds.expect_column_values_to_be_between(
                "rating", min_value=MIN_RATING, max_value=MAX_RATING
            ).to_json_dict()
        )

    if "text" in df.columns:
        results.append(
            ds.expect_column_value_lengths_to_be_between(
                "text", min_value=MIN_TEXT_LEN, max_value=MAX_TEXT_LEN
            ).to_json_dict()
        )
        results.append(
            ds.expect_column_values_to_match_regex("text", HAS_LETTER_RE).to_json_dict()
        )

    if "label" in df.columns:
        results.append(
            ds.expect_column_distinct_values_to_contain_set(
                "label", sorted(VALID_LABELS)
            ).to_json_dict()
        )

    return results


def check_language_distribution(
    df: pd.DataFrame,
    accepted: set = ACCEPTED_LANGS,
    min_share: float = LANG_MIN_SHARE,
    sample_size: int = LANG_SAMPLE_SIZE,
) -> dict:
    """Sample-based language check. Returns a GE-shaped result dict."""
    if "text" not in df.columns or len(df) == 0:
        return {
            "success": False,
            "expectation_config": {
                "expectation_type": "expect_language_distribution",
                "kwargs": {"accepted": sorted(accepted), "min_share": min_share},
            },
            "result": {"observed_share": 0.0, "n_sampled": 0},
        }

    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0

    sample = df.sample(min(len(df), sample_size), random_state=42)
    detected = []
    for text in sample["text"].astype(str):
        try:
            detected.append(detect(text))
        except LangDetectException:
            detected.append("unknown")
    n_accepted = sum(1 for code in detected if code in accepted)
    share = n_accepted / max(1, len(detected))

    return {
        "success": share >= min_share,
        "expectation_config": {
            "expectation_type": "expect_language_distribution",
            "kwargs": {"accepted": sorted(accepted), "min_share": min_share},
        },
        "result": {"observed_share": share, "n_sampled": len(detected)},
    }


def validate_reviews(df: pd.DataFrame, check_language: bool = True) -> ValidationResult:
    """Run the Gold / training / Postgres ingest gate."""
    expectations = _run_expectations(df)
    if check_language and "text" in df.columns and len(df) > 0:
        expectations.append(check_language_distribution(df))

    failures = [
        {
            "expectation": e.get("expectation_config", {}).get("expectation_type"),
            "kwargs": e.get("expectation_config", {}).get("kwargs"),
            "result": e.get("result"),
        }
        for e in expectations
        if not e.get("success", False)
    ]
    return ValidationResult(
        success=not failures,
        n_rows=len(df),
        failures=failures,
    )
