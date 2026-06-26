"""One-off: replace competitor brand names in the demo review CSVs with "BrewLeaf".

Safe by construction:
  * whole-word matches only (``\\b``) — never touches "pretty", "interpret", etc.
  * bare "Costa" is intentionally NOT replaced (reviewers say "Costa Rica" for the
    coffee origin); only "Costa Coffee" is.
  * case-insensitive, and idempotent (re-running finds nothing left to change).

Run from the repo root:  python scripts/rebrand_demo_data.py

"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "data" / "demo"
CSVS = ["demo_jun2026_spike.csv", "demo_jun2026_stable.csv", "demo_holdout_full.csv"]
BRAND = "BrewLeaf"

# Brand aliases -> BrewLeaf. Multi-word / longest variants first so e.g.
# "dunkin' donuts" wins over "dunkin". Apostrophes are optional ([']?) to catch
# both "McDonald's" and "McDonalds".
ALIASES = [
    r"dunkin['’]? donuts", r"dunkin['’]?",
    r"(?:my\s?)?mc\s?donald['’]?s",  # also catches the glued "MyMcDonalds" rewards app
    r"star\s?bucks",
    r"dutch bros", r"gong cha",
    r"pret a manger", r"pret",
    r"panera bread", r"panera",
    r"chick[-\s]?fil[-\s]?a",
    r"burger king",
    r"tim hortons",
    r"peet['’]?s coffee", r"peet['’]?s",
    r"caribou coffee",
    r"costa coffee",
    r"wendy['’]?s", r"wendys",
    r"popeye['’]?s", r"popeyes",
    r"chipotle",
    r"subway",
]
_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(ALIASES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def rebrand(text: str) -> str:
    return _PATTERN.sub(BRAND, str(text))


def main() -> None:
    for name in CSVS:
        path = DEMO / name
        df = pd.read_csv(path)
        before = df["text"].astype(str)
        after = before.map(rebrand)
        changed = int((before != after).sum())
        df["text"] = after
        df.to_csv(path, index=False)
        print(f"[rebrand] {name}: {changed} rows rewritten -> {BRAND}")


if __name__ == "__main__":
    main()
