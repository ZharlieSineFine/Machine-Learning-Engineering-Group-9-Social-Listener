"""Batch inference — score reviews with the champion model and write the predictions
into the Postgres ``reviews`` table the dashboard reads.

Two entry points share the same scoring + write logic:

  * :func:`run_on_silver` — **production**. Scores the medallion's freshly-built
    silver window (the ``batch_inference`` DAG calls this, Dataset-triggered after
    the medallion publishes).
  * :func:`run` — **offline demo / tests**. Scores a replay window from the replay
    simulator. Kept for the demo scripts + unit tests.

The inference step, either way:

    reviews (text)  ->  champion model  ->  predicted label  ->  reviews table  ->  dashboard

The champion (``models/artifacts/baseline.pkl``, TF-IDF + LogReg) was trained on
integer classes (0=negative, 1=neutral, 2=positive). We map those back to the
string labels the dashboard + GE suite expect, and apply the champion's tuned
negative threshold (0.46) so the negative class — the one the brand cares about —
is caught the way the registered champion was tuned to.

CLI:
    # seed a clean 2-week history (dashboard looks normal)
    python -m serving.batch_infer --scenario stable --truncate

    # inject "today's" negative-review burst (dashboard spikes + alerts)
    python -m serving.batch_infer --scenario spike --asof 2026-06-21 --n-recent 1 --as-now

Owner: Charlie + Ha (Data & Eval), wiring Van's champion model into Amelia's dashboard.
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from data.storage.config import PostgresConfig
from data.storage.warehouse import connection
from monitoring.drift_checks import load_replay_window

# The champion pickle was trained on an older scikit-learn; silence the unpickle
# version-mismatch warning so the demo terminal stays clean. The predictions are
# unaffected for this TF-IDF + LogReg pipeline (verified: stable 19.8%, spike 51.3%).
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning

    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass

# Integer-class -> string-label map (champion pickle predicts ints).
LABELS = ["negative", "neutral", "positive"]
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PICKLE = Path(
    os.getenv("MODEL_PICKLE_PATH", str(_REPO_ROOT / "models" / "artifacts" / "baseline.pkl"))
)
DEFAULT_NEG_THRESHOLD = float(os.getenv("NEG_THRESHOLD", "0.46"))
REPLAY_SOURCE = "replay"

# Production path: score the medallion's freshly-built silver instead of a replay
# folder. ``BATCH_SOURCE`` tags those rows in ``reviews.source``; the window size is
# how many recent ``review_date=`` partitions to score per run (default 1 = the day
# the medallion just built).
BATCH_SOURCE = "batch"
SILVER_ROOT = _REPO_ROOT / "data" / "silver" / "reviews"
DEFAULT_SILVER_PARTITIONS = int(os.getenv("BATCH_INFER_SILVER_PARTITIONS", "1"))


# ---------------------------------------------------------------------------
# Model + prediction (pure)
# ---------------------------------------------------------------------------
def load_model(pickle_path: Path = DEFAULT_PICKLE):
    """Load the champion pipeline pickle (raw sklearn Pipeline)."""
    import models.baseline_sklearn  # noqa: F401 — registers the pickled preprocessor symbol

    with open(pickle_path, "rb") as fh:
        return pickle.load(fh)


def _resolve_label(raw) -> str:
    """Map a model output (int class id or string) to a canonical label."""
    if isinstance(raw, str) and raw in LABELS:
        return raw
    try:
        return ID2LABEL[int(raw)]
    except (KeyError, ValueError, TypeError):
        return str(raw)


def _neg_index(classes: list) -> Optional[int]:
    for i, c in enumerate(classes):
        if _resolve_label(c) == "negative":
            return i
    return None


def predict_labels(pipe, texts: List[str], neg_threshold: float = DEFAULT_NEG_THRESHOLD) -> List[str]:
    """Predict string labels, applying the champion's tuned negative threshold.

    If ``predict_proba`` is available, a row is labelled ``negative`` when
    P(negative) >= ``neg_threshold`` (the champion was tuned to 0.46); otherwise
    the argmax of the remaining classes wins. Falls back to plain ``predict`` for
    models without probabilities.
    """
    texts = [str(t) for t in texts]
    classes = list(getattr(pipe, "classes_", []))
    if neg_threshold and hasattr(pipe, "predict_proba") and classes:
        neg_idx = _neg_index(classes)
        proba = pipe.predict_proba(texts)
        out: List[str] = []
        for row in proba:
            if neg_idx is not None and row[neg_idx] >= neg_threshold:
                out.append("negative")
            else:
                best = max(
                    range(len(row)),
                    key=lambda i: row[i] if i != neg_idx else float("-inf"),
                )
                out.append(_resolve_label(classes[best]))
        return out
    return [_resolve_label(p) for p in pipe.predict(texts)]


# ---------------------------------------------------------------------------
# Write to the reviews table (what the dashboard reads)
# ---------------------------------------------------------------------------
def _ingested_at_series(df: pd.DataFrame, as_now: bool) -> List:
    """Resolve each row's ingested_at: 'now' (today's burst) or the review_date."""
    if as_now:
        now = datetime.now(timezone.utc)
        return [now] * len(df)
    return [
        pd.to_datetime(d, errors="coerce").to_pydatetime() if pd.notna(d) else datetime.now(timezone.utc)
        for d in df["review_date"]
    ]


def write_reviews(conn, df: pd.DataFrame, labels: List[str], *, as_now: bool, source: str = REPLAY_SOURCE) -> int:
    """Insert scored rows into ``reviews`` (text, label, source, ingested_at)."""
    from psycopg2.extras import execute_values

    ingested = _ingested_at_series(df, as_now)
    rows = [
        (str(t)[:10000], lbl, source, ts)
        for t, lbl, ts in zip(df["text"], labels, ingested)
        if isinstance(t, str) and t.strip()
    ]
    if not rows:
        return 0
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO reviews (text, label, source, ingested_at) VALUES %s",
            rows,
        )
    conn.commit()
    return len(rows)


def truncate_reviews(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE reviews RESTART IDENTITY CASCADE")
    conn.commit()


def clear_recent_reviews(conn, days: int = 1) -> int:
    """Delete the most recent ``days`` of reviews so a fresh 'today' batch stands alone."""
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM reviews WHERE ingested_at >= NOW() - make_interval(days => %s)",
            (days,),
        )
        n = cur.rowcount
    conn.commit()
    return n


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _shift_to_today(df: pd.DataFrame, end: Optional[date] = None) -> pd.DataFrame:
    """Shift review_date so the window's last day lands on ``end`` (default: today).

    Keeps the day-to-day shape (the demo data lives in mid-2026) but anchors the
    series to "now" so the dashboard's recent-window query and timeline look live.
    """
    end = end or datetime.now(timezone.utc).date()
    out = df.copy()
    dts = pd.to_datetime(out["review_date"], errors="coerce")
    if dts.notna().any():
        delta = end - dts.max().date()
        out["review_date"] = (dts + pd.Timedelta(days=delta.days)).dt.strftime("%Y-%m-%d")
    return out


def run(
    scenario: str,
    *,
    asof: Optional[str] = None,
    n_recent: Optional[int] = None,
    as_now: bool = False,
    shift_to_today: bool = False,
    truncate: bool = False,
    clear_today: bool = False,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    pickle_path: Path = DEFAULT_PICKLE,
    pg: Optional[PostgresConfig] = None,
) -> dict:
    """Score a replay window and write predictions into ``reviews``.

    Returns a summary dict (rows written + predicted label distribution).
    """
    pg = pg or PostgresConfig.from_env()
    if pg is None:
        raise RuntimeError("Postgres not configured — set POSTGRES_* env vars")

    df = load_replay_window(scenario, asof=asof, n_recent=n_recent)
    if shift_to_today:
        df = _shift_to_today(df)

    pipe = load_model(pickle_path)
    labels = predict_labels(pipe, list(df["text"]), neg_threshold=neg_threshold)

    dist = pd.Series(labels).value_counts().to_dict()
    with connection(pg) as conn:
        if truncate:
            truncate_reviews(conn)
        elif clear_today:
            clear_recent_reviews(conn, days=1)
        n = write_reviews(conn, df, labels, as_now=as_now)

    neg_pct = round(100 * labels.count("negative") / max(1, len(labels)), 1)
    summary = {
        "scenario": scenario,
        "rows_written": n,
        "predicted_dist": dist,
        "negative_pct": neg_pct,
        "as_now": as_now,
    }
    print(
        f"[batch_infer:{scenario}] wrote {n} rows -> reviews | "
        f"predicted negative={neg_pct}% | dist={dist}"
    )
    return summary


def run_on_silver(
    *,
    n_partitions: int = DEFAULT_SILVER_PARTITIONS,
    as_now: bool = True,
    clear_today: bool = True,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    pickle_path: Path = DEFAULT_PICKLE,
    silver_root: Optional[Path] = None,
    pg: Optional[PostgresConfig] = None,
) -> dict:
    """Production batch inference — score the medallion's freshly-built silver window.

    Reads the most recent ``n_partitions`` ``review_date=`` silver partitions (the
    real reviews the medallion just ingested), scores them with the champion model,
    and writes the predictions into the ``reviews`` table the dashboard reads. This is
    the production read-side step; it replaces the replay-driven :func:`run`, which
    stays for the offline demo + tests.

    Mirrors :func:`run`'s write semantics: ``clear_today`` refreshes only the most
    recent day and ``as_now`` stamps the rows "now", so multi-day history is preserved
    while today's batch is rescored. Returns the same summary shape.
    """
    pg = pg or PostgresConfig.from_env()
    if pg is None:
        raise RuntimeError("Postgres not configured — set POSTGRES_* env vars")

    # _load_recent_silver is the same helper the medallion gate + drift monitor use.
    from monitoring.drift_checks import _load_recent_silver

    silver_root = silver_root or SILVER_ROOT
    df = _load_recent_silver(silver_root, n_partitions)
    if df is None or df.empty:
        print(f"[batch_infer:silver] no silver partitions under {silver_root}; nothing to score")
        return {"source": "silver", "rows_written": 0, "predicted_dist": {}, "negative_pct": 0.0, "as_now": as_now}

    pipe = load_model(pickle_path)
    labels = predict_labels(pipe, list(df["text"]), neg_threshold=neg_threshold)

    dist = pd.Series(labels).value_counts().to_dict()
    with connection(pg) as conn:
        if clear_today:
            clear_recent_reviews(conn, days=1)
        n = write_reviews(conn, df, labels, as_now=as_now, source=BATCH_SOURCE)

    neg_pct = round(100 * labels.count("negative") / max(1, len(labels)), 1)
    summary = {
        "source": "silver",
        "rows_written": n,
        "predicted_dist": dist,
        "negative_pct": neg_pct,
        "as_now": as_now,
    }
    print(
        f"[batch_infer:silver] wrote {n} rows -> reviews | "
        f"predicted negative={neg_pct}% | dist={dist}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-score replay reviews into the reviews table.")
    ap.add_argument("--scenario", choices=["stable", "spike", "holdout"], default="stable")
    ap.add_argument("--asof", default=None, help="Keep only replay partitions on/before this YYYY-MM-DD.")
    ap.add_argument("--n-recent", type=int, default=None, help="Use only the most recent N replay partitions.")
    ap.add_argument("--as-now", action="store_true", help="Stamp ingested_at = now (today's burst).")
    ap.add_argument("--shift-to-today", action="store_true", help="Anchor the window's last day to today.")
    ap.add_argument("--truncate", action="store_true", help="Clear the reviews table first (fresh seed).")
    ap.add_argument("--clear-today", action="store_true", help="Delete the last day of reviews before writing.")
    ap.add_argument("--neg-threshold", type=float, default=DEFAULT_NEG_THRESHOLD)
    args = ap.parse_args()

    run(
        args.scenario,
        asof=args.asof,
        n_recent=args.n_recent,
        as_now=args.as_now,
        shift_to_today=args.shift_to_today,
        truncate=args.truncate,
        clear_today=args.clear_today,
        neg_threshold=args.neg_threshold,
    )


if __name__ == "__main__":
    main()
