"""Shared review contract for ingest adapters (Yelp, Malaysia, replay).

`EXPECTED_COLUMNS` is the canonical column order every loader must produce.
Keep in sync with `data/sample/reviews_sample.csv`, `scripts/build_sample.py`,
and `data/README.md`.

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

from typing import List

# Column order matches the committed seed CSV header.
EXPECTED_COLUMNS: List[str] = [
    "text",
    "label",
    "rating",
    "source",
    "restaurant",
    "location",
]
