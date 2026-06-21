"""Replay simulator — replays a pre-built demo window as a BrewLeaf operational stream.

The drift / monitoring demo uses purpose-built windows in `demo_data/` (project root). Both
demo versions cover the same June-2026 window (2026-06-07 -> 2026-06-24) with the same 2,056
reviews and identical *overall* label mix — only the **timing of negatives** differs:

    demo_jun2026_stable.csv   negatives steady at ~19-20%/day            -> drift gate PASSES
    demo_jun2026_spike.csv    ~17-18%/day, then a sudden 60% on          -> drift gate FIRES
                              2026-06-21 (a one-day complaint spike)
    demo_holdout_full.csv     the full 2022 holdout the two are drawn from (reference window)

The simulator replays the chosen scenario in `review_date` order (optionally paced) to a
standalone stream that the Evidently / monitoring step consumes as the "current" window.

These files are already in the final contract shape (`text, label, review_date`) and have no
shop-name column, so — unlike the earlier Gold-sourced version — there is **no brand-name
replacement** (nothing to key on) and **no synthetic drift injection** (the spike is baked
into `demo_jun2026_spike.csv`). The text is replayed verbatim.

Output stream columns: review_id (deterministic), review_date, text, label, source="replay",
scenario, brand="BrewLeaf". Written as `<out>/<scenario>/review_date=YYYY-MM-DD/part.parquet`.

Run:
    python -m data.ingest.replay --scenario spike
    python -m data.ingest.replay --scenario stable --speed 4   # ~4 days/sec to watch it arrive

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = ROOT.parent.parent  # project root that holds demo_data/ (and the raw datasets)

BRAND = "BrewLeaf"
REPLAY_SOURCE = "replay"

# The demo scenarios shipped in demo_data/ (override the folder with --demo-dir).
SCENARIOS = {
    "stable": "demo_jun2026_stable.csv",
    "spike": "demo_jun2026_spike.csv",
    "holdout": "demo_holdout_full.csv",
}
DEFAULT_SCENARIO = "spike"

SOURCE_COLUMNS = ["text", "label", "review_date"]  # required in every demo CSV
REPLAY_COLUMNS = ["review_id", "review_date", "text", "label", "source", "scenario", "brand"]

# Prefer an in-repo copy (tracked, available in containers); fall back to the
# external workspace demo_data/ for local runs.
_IN_REPO_DEMO_DIR = ROOT / "data" / "demo"
DEFAULT_DEMO_DIR = _IN_REPO_DEMO_DIR if _IN_REPO_DEMO_DIR.exists() else WORKSPACE_ROOT / "demo_data"
DEFAULT_OUT_ROOT = ROOT / "data" / "replay"
_MAX_SLEEP_SECONDS = 5.0  # cap per-step pacing sleep so a long gap can't stall the run


# ---------- pure transforms (unit-testable) ----------

def _review_id(scenario: str, review_date: str, text: str) -> str:
    """Deterministic id for a replayed review (stable across runs; for dedup/keys)."""
    return hashlib.sha256(f"{scenario}|{review_date}|{text}".encode("utf-8")).hexdigest()[:16]


def build_replay_stream(demo_df: pd.DataFrame, scenario: str, brand: str = BRAND) -> pd.DataFrame:
    """Shape a demo window into the replay stream (pure core).

    `demo_df` needs `text, label, review_date`. Adds a deterministic `review_id` plus the
    `source` / `scenario` / `brand` tags, and sorts by `review_date`. Text is left verbatim.
    """
    missing = set(SOURCE_COLUMNS) - set(demo_df.columns)
    if missing:
        raise ValueError(f"demo_df missing columns: {sorted(missing)}")

    df = demo_df.copy()
    df["review_date"] = df["review_date"].astype(str)
    df["text"] = df["text"].astype(str)
    df["review_id"] = [_review_id(scenario, d, t) for d, t in zip(df["review_date"], df["text"])]
    df["source"] = REPLAY_SOURCE
    df["scenario"] = scenario
    df["brand"] = brand
    return df[REPLAY_COLUMNS].sort_values("review_date", kind="stable").reset_index(drop=True)


def daily_negative_fraction(stream_df: pd.DataFrame) -> pd.Series:
    """Share of negative-labelled reviews per review_date — the stable-vs-spike signal."""
    neg = stream_df["label"].eq("negative")
    return neg.groupby(stream_df["review_date"]).mean()


# ---------- IO ----------

def demo_csv_path(scenario: str, demo_dir: Path = DEFAULT_DEMO_DIR) -> Path:
    """Resolve a scenario name to its CSV path under `demo_dir`."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    return Path(demo_dir) / SCENARIOS[scenario]


def load_demo(scenario: str, demo_dir: Path = DEFAULT_DEMO_DIR) -> pd.DataFrame:
    """Read a demo scenario CSV and verify it carries the contract columns."""
    path = demo_csv_path(scenario, demo_dir)
    if not path.is_file():
        raise FileNotFoundError(f"demo scenario CSV not found: {path}")
    df = pd.read_csv(path)
    missing = set(SOURCE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df


def emit_replay(stream_df: pd.DataFrame, out_root: Path, speed: float = 0.0) -> int:
    """Write the stream as `review_date=YYYY-MM-DD/part.parquet` partitions under out_root.

    `speed` > 0 paces emission as *simulated days per real second* (sleeps between dates),
    so the demo can watch the window arrive; `speed` == 0 writes everything at once (batch).
    Returns the number of partitions written.
    """
    out_root = Path(out_root)
    dates = sorted(stream_df["review_date"].unique())
    prev: Optional[str] = None
    for d in dates:
        part = stream_df[stream_df["review_date"] == d]
        pdir = out_root / f"review_date={d}"
        pdir.mkdir(parents=True, exist_ok=True)
        part.to_parquet(pdir / "part.parquet", index=False)
        if speed and prev is not None:
            gap_days = (pd.Timestamp(d) - pd.Timestamp(prev)).days
            sleep_s = max(0.0, gap_days / speed)
            if sleep_s:
                time.sleep(min(sleep_s, _MAX_SLEEP_SECONDS))
        prev = d
    return len(dates)


def _summarize(stream_df: pd.DataFrame) -> str:
    n = len(stream_df)
    lo, hi = stream_df["review_date"].min(), stream_df["review_date"].max()
    labels = dict(stream_df["label"].value_counts())
    neg = daily_negative_fraction(stream_df)
    return (f"{n} rows | {lo} -> {hi} | labels={labels} | "
            f"peak negative/day {neg.max():.0%} on {neg.idxmax()}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a demo_data scenario as a BrewLeaf operational stream.")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default=DEFAULT_SCENARIO,
                    help="Which demo window to replay (stable=no drift, spike=negative spike).")
    ap.add_argument("--demo-dir", type=Path, default=DEFAULT_DEMO_DIR, help="Folder holding the demo CSVs.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_ROOT, help="Output root (a <scenario>/ subdir is added).")
    ap.add_argument("--speed", type=float, default=0.0,
                    help="Simulated days per real second (0 = batch, no pacing).")
    args = ap.parse_args()

    demo = load_demo(args.scenario, args.demo_dir)
    stream = build_replay_stream(demo, args.scenario)
    out_root = args.out / args.scenario
    n_parts = emit_replay(stream, out_root, speed=args.speed)
    print(f"Replay [{args.scenario}]: {_summarize(stream)}")
    print(f"Wrote {n_parts} review_date partitions -> {out_root}")


if __name__ == "__main__":
    main()
