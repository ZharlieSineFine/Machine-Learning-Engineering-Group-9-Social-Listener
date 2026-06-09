"""Great Expectations suite for the `reviews` dataset (Phase 1 thin slice).

Validation gate that runs BEFORE rows hit Postgres. If `validate_reviews(df)`
returns a failed result, the ingestion DAG fails the task and no data is
written, so a bad upstream batch can't poison the warehouse.

Phase 1 checks (per WORKFLOW.md):
    * Schema — all 6 contract columns are present.
    * Non-null — `text`, `label`, `source` have no nulls.
    * Cardinality — `label` is in {negative, neutral, positive}.
    * Range — `rating` (when present) is in [1, 5].

Phase 2 expands this — see Step 9 (`# TODO (member)` hooks below).

Owner: Charlie + Ha (Data & Eval).

API note (for future maintainers):
    We're using the *legacy* PandasDataset API. It's tiny and still ships in
    GE 0.18, but is gone in GE 1.x. Migrate to the FluentDatasource pattern
    when bumping the pin.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import pandas as pd
from great_expectations.dataset import PandasDataset

REQUIRED_COLUMNS = ["text", "label", "rating", "source", "restaurant", "location"]
VALID_LABELS = ["negative", "neutral", "positive"]
MIN_RATING = 1.0
MAX_RATING = 5.0

# --- Phase 2 (Step 9) bounds ---
MIN_TEXT_LEN = 5         # below this is probably garbage ("ok", ".", "!")
MAX_TEXT_LEN = 10_000    # above this is probably scraping noise
HAS_LETTER_RE = r"[A-Za-z-￿]"  # latin or extended unicode letter
ACCEPTED_LANGS = {"en", "ms", "id"}  # Indonesian shares vocab with Malay; langdetect often returns 'id' for short MY phrases
LANG_MIN_SHARE = 0.80     # at least 80% of rows must be in ACCEPTED_LANGS
LANG_SAMPLE_SIZE = 200    # cap detection to keep validation fast on big batches


@dataclass
class ValidationResult:
    """Compact, JSON-serialisable summary the DAG and tests both consume."""
    success: bool
    n_rows: int
    failures: List[dict] = field(default_factory=list)

    def raise_for_status(self) -> None:
        if not self.success:
            raise ValueError(
                f"reviews_suite validation failed ({len(self.failures)} expectation(s) failed). "
                f"First failure: {self.failures[0] if self.failures else 'n/a'}"
            )


def _run_expectations(df: pd.DataFrame) -> List[dict]:
    """Run all Phase 1 expectations against `df`. Returns a list of GE results."""
    ds = PandasDataset(df)
    results: List[dict] = []

    # Schema
    for col in REQUIRED_COLUMNS:
        results.append(ds.expect_column_to_exist(col).to_json_dict())

    # Non-null
    for col in ["text", "label", "source"]:
        if col in df.columns:
            results.append(ds.expect_column_values_to_not_be_null(col).to_json_dict())

    # Label cardinality
    if "label" in df.columns:
        results.append(
            ds.expect_column_values_to_be_in_set("label", VALID_LABELS).to_json_dict()
        )

    # Rating range — allow nulls (mostly_<1 lets nulls slide), reject out-of-range
    if "rating" in df.columns:
        results.append(
            ds.expect_column_values_to_be_between(
                "rating", min_value=MIN_RATING, max_value=MAX_RATING
            ).to_json_dict()
        )

    # ---- Phase 2 additions (Step 9) ----
    # Length bounds — catches "ok" / "..." junk and runaway scraped blobs.
    if "text" in df.columns:
        results.append(
            ds.expect_column_value_lengths_to_be_between(
                "text", min_value=MIN_TEXT_LEN, max_value=MAX_TEXT_LEN
            ).to_json_dict()
        )

    # Regex — must contain at least one letter. Rejects pure-emoji / pure-numeric rows.
    if "text" in df.columns:
        results.append(
            ds.expect_column_values_to_match_regex("text", HAS_LETTER_RE).to_json_dict()
        )

    # Cardinality — all 3 sentiment classes must appear. Otherwise training/eval
    # split could be 2-class, which would break F1_macro reporting downstream.
    if "label" in df.columns:
        results.append(
            ds.expect_column_distinct_values_to_contain_set(
                "label", VALID_LABELS
            ).to_json_dict()
        )

    return results


def check_language_distribution(
    df: pd.DataFrame,
    accepted: set = ACCEPTED_LANGS,
    min_share: float = LANG_MIN_SHARE,
    sample_size: int = LANG_SAMPLE_SIZE,
) -> dict:
    """Sample-based language check. Outside the GE suite because it's slow.

    Returns the same shape GE results use so callers can append it uniformly.
    Sample rather than scan-all — per-row language detection is ~1ms.
    """
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
    DetectorFactory.seed = 0  # deterministic

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
    """Run the suite and return a ValidationResult.

    `check_language=False` is an escape hatch for callers that already
    filter by language upstream (or in tests that don't care).
    """
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
