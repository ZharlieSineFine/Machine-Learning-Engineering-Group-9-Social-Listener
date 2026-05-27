"""Build a small, balanced sample CSV for the smoke test.

Reads from `Malaysia Restaurant Review Datasets/data_cleaned/` and writes a
unified file to `data/sample/reviews_sample.csv` with the schema the rest of
the pipeline expects.

Schema (contract — keep in sync with data/schemas/):
    text        : str   — review body
    label       : str   — 'positive' | 'neutral' | 'negative'
    rating      : float — original 1..5 star rating
    source      : str   — 'google' | 'tripadvisor' | 'yelp' (future)
    restaurant  : str
    location    : str

Label rule (ternary):
    rating >= 4  -> positive
    rating == 3  -> neutral
    rating <= 2  -> negative

Run:
    python scripts/build_sample.py --n 1000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Malaysia Restaurant Review Datasets" / "data_cleaned"
OUT = ROOT / "data" / "sample" / "reviews_sample.csv"


def _label(rating: float) -> str:
    if rating >= 4:
        return "positive"
    if rating <= 2:
        return "negative"
    return "neutral"


def _load_google() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "GoogleReview_data_cleaned.csv")
    df = df.rename(columns={"Review": "text", "Rating": "rating",
                            "Restaurant": "restaurant", "Location": "location"})
    df["source"] = "google"
    return df[["text", "rating", "source", "restaurant", "location"]]


def _load_tripadvisor() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "TripAdvisor_data_cleaned.csv")
    df = df.rename(columns={"Review": "text", "Rating": "rating",
                            "Restaurant": "restaurant", "Location": "location"})
    df["source"] = "tripadvisor"
    return df[["text", "rating", "source", "restaurant", "location"]]


def build_sample(n: int, seed: int) -> pd.DataFrame:
    df = pd.concat([_load_google(), _load_tripadvisor()], ignore_index=True)

    df = df.dropna(subset=["text", "rating"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["label"] = df["rating"].apply(_label)

    # Balanced sample across labels so the baseline isn't trivially biased.
    per_class = max(1, n // 3)
    parts = []
    for lbl in ["positive", "neutral", "negative"]:
        sub = df[df["label"] == lbl]
        take = min(per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return out[["text", "label", "rating", "source", "restaurant", "location"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample = build_sample(args.n, args.seed)
    sample.to_csv(OUT, index=False)
    print(f"Wrote {len(sample)} rows -> {OUT}")
    print(sample["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
