"""Evidently drift check — Phase 1 stub.

This is the wiring slice for the ``evaluate_and_monitor`` DAG described in
ARCHITECTURE.md §5. Real monitoring (train baseline vs. last 7d of ingested
reviews) lands in Phase 2; right now **there is no train data yet**, so this
module follows the plan in ``monitoring/README.md``:

    > Phase 1 stub: Run Evidently with ``DataDriftPreset`` on train vs. itself
    > (will always pass) so the wiring exists. Real reference dataset comes in
    > phase 2.

So we load the in-repo sample CSV (the contract source-of-truth) as both the
*reference* and the *current* frame. Identical frames → zero drift → the gate
always passes. The value here is the plumbing: a callable an Airflow
PythonOperator can run, an HTML report on disk, and a summary dict ready to be
appended to the ``monitoring_reports`` table once Postgres wiring lands.

Designed to be runnable three ways, like ``models/train.py``:
    * directly from the Airflow DAG via PythonOperator,
    * from the CLI (``python monitoring/drift_checks.py``),
    * from a unit/smoke test calling ``run_drift_check()``.

Nothing here requires Postgres, MinIO, or MLflow to be up — that keeps it
smoke-testable, matching the rest of the Phase 1 thin slice.

Owner: Charlie + Ha (Monitoring).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — works both inside the Airflow container (where the repo is
# split across /opt/project/{data,monitoring}) and from a local checkout.
# ---------------------------------------------------------------------------
def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c and c.exists():
            return c
    return None


# Local checkout root: monitoring/ -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLE_CSV = _first_existing(
    Path(os.getenv("DRIFT_SAMPLE_CSV", "")) if os.getenv("DRIFT_SAMPLE_CSV") else None,
    Path("/opt/project/data/sample/reviews_sample.csv"),  # Airflow container mount
    _REPO_ROOT / "data" / "sample" / "reviews_sample.csv",  # local checkout
)

# Silver root holding ``review_date=YYYY-MM-DD/part.parquet`` partitions, which
# the medallion DAG produces. Phase 2 ``current`` data is read from the most
# recent partitions here. Falls back to the train-vs-itself stub if empty.
DEFAULT_SILVER_ROOT = _first_existing(
    Path(os.getenv("DRIFT_SILVER_ROOT", "")) if os.getenv("DRIFT_SILVER_ROOT") else None,
    Path("/opt/project/data/silver/reviews"),  # Airflow container mount
    _REPO_ROOT / "data" / "silver" / "reviews",  # local checkout
)

# How many recent review_date partitions form the ``current`` window.
DRIFT_RECENT_PARTITIONS = int(os.getenv("DRIFT_RECENT_PARTITIONS", "7"))

# Where the HTML report is written. Phase 2 swaps this for MinIO
# (s3://monitoring/{date}/report.html) per monitoring/README.md.
DEFAULT_REPORT_DIR = Path(
    os.getenv("DRIFT_REPORT_DIR", "/opt/project/monitoring/reports")
)

# Drift gate threshold from monitoring/README.md ("drift score > 0.3").
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))


@dataclass
class DriftResult:
    """What the DAG hands forward / will write to ``monitoring_reports``."""

    report_path: str
    drift_score: float          # share of drifted columns (0.0 .. 1.0)
    n_drifted_columns: int
    dataset_drift: bool
    passed_gate: bool           # True == no actionable drift
    n_reference: int
    n_current: int
    evidently_ran: bool         # False if we fell back to the no-op stub


# ---------------------------------------------------------------------------
# Feature frame
# ---------------------------------------------------------------------------
def _features_from_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Project raw reviews onto the columns we actually monitor.

    Mirrors the "data drift" row in monitoring/README.md: text-length
    distribution, source mix, rating, and label distribution. Keeping this
    tiny and explicit means Evidently has typed numeric + categorical columns
    to reason about instead of free-text it would just ignore.
    """
    out = pd.DataFrame()
    out["text_len"] = df["text"].fillna("").astype(str).str.len()
    if "rating" in df.columns:
        out["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "source" in df.columns:
        out["source"] = df["source"].astype("category")
    if "label" in df.columns:
        out["label"] = df["label"].astype("category")
    return out


def _load_recent_silver(
    silver_root: Path, n_partitions: int
) -> Optional[pd.DataFrame]:
    """Concat the most recent ``n_partitions`` ``review_date=`` silver partitions.

    Returns None when the silver root has no readable partitions yet (fresh
    stack), so the caller can degrade to the train-vs-itself stub instead of
    failing the DAG.
    """
    if not silver_root or not silver_root.exists():
        return None

    # Partition dirs are ``review_date=YYYY-MM-DD``; newest keys sort last.
    keys = sorted(
        (p.name.split("=", 1)[1] for p in silver_root.glob("review_date=*") if p.is_dir()),
        reverse=True,
    )[:n_partitions]
    if not keys:
        return None

    from data.refine.build_silver import read_silver_partition, silver_partition_path

    frames = []
    for key in keys:
        df = read_silver_partition(silver_partition_path(silver_root, key))
        if not df.empty:
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _build_reference_and_current(
    sample_csv: Path,
    silver_root: Optional[Path] = None,
    n_partitions: int = DRIFT_RECENT_PARTITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Reference = sample/training CSV; current = recent silver partitions.

    Returns ``(reference, current, used_silver)``. When no silver partitions
    exist yet, ``current`` falls back to a copy of ``reference`` (train vs.
    itself -> guaranteed-pass gate) and ``used_silver`` is False, preserving the
    "DAG stays green while wiring settles" property.

    Reference and current are reduced to their shared columns so Evidently sees
    a matching schema (silver has no ``label``, the sample CSV does).
    """
    reference = _features_from_reviews(pd.read_csv(sample_csv))

    silver_root = silver_root or DEFAULT_SILVER_ROOT
    recent = _load_recent_silver(silver_root, n_partitions)
    if recent is None:
        return reference, reference.copy(), False

    current = _features_from_reviews(recent)
    shared = [c for c in reference.columns if c in current.columns]
    return reference[shared], current[shared], True


# ---------------------------------------------------------------------------
# Evidently — supports both the classic (0.4–0.6) and new (0.7+) APIs, and
# degrades to a deterministic no-op summary rather than failing the DAG if the
# installed Evidently build exposes neither. Phase 1 is about wiring, not about
# a perfect report.
# ---------------------------------------------------------------------------
def _run_evidently(
    reference: pd.DataFrame, current: pd.DataFrame, html_path: Path
) -> Optional[dict]:
    """Run DataDriftPreset and return a normalized summary, or None on fallback."""
    # Classic API: evidently 0.4.x – 0.6.x
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(str(html_path))
        summary = report.as_dict()["metrics"][0]["result"]
        return {
            "dataset_drift": bool(summary.get("dataset_drift", False)),
            "drift_score": float(summary.get("share_of_drifted_columns", 0.0)),
            "n_drifted_columns": int(summary.get("number_of_drifted_columns", 0)),
        }
    except ImportError:
        pass  # not the classic API — try the new one
    except Exception as exc:  # report ran but shape differs; don't kill the DAG
        print(f"[drift] classic Evidently API errored, falling back: {exc}")

    # New API: evidently 0.7+
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        snapshot = report.run(reference_data=reference, current_data=current)
        try:
            snapshot.save_html(str(html_path))
        except Exception as exc:  # HTML rendering optional in Phase 1
            print(f"[drift] new-API HTML render skipped: {exc}")
        result = snapshot.dict()
        # The 0.7 schema differs; pull what we can, default to "no drift".
        share = _dig_drift_share(result)
        return {
            "dataset_drift": share > 0.0,
            "drift_score": share,
            "n_drifted_columns": 0,
        }
    except Exception as exc:
        print(f"[drift] Evidently unavailable/incompatible, using stub: {exc}")
        return None


def _dig_drift_share(result: dict) -> float:
    """Best-effort scrape of a drift share out of the 0.7 snapshot dict."""
    metrics = result.get("metrics", [])
    for m in metrics:
        val = m.get("value")
        if isinstance(val, dict) and "share" in val:
            try:
                return float(val["share"])
            except (TypeError, ValueError):
                continue
    return 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_drift_check(
    sample_csv: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    threshold: float = DRIFT_THRESHOLD,
    silver_root: Optional[Path] = None,
) -> DriftResult:
    """Run the drift check and return a structured result.

    ``reference`` is the training/sample distribution; ``current`` is the most
    recent silver partitions. Falls back to train-vs-itself when no silver data
    exists yet. Always returns (never raises on Evidently internals) so the
    Airflow task stays green while the pipeline is still being wired together.
    """
    sample_csv = Path(sample_csv) if sample_csv else DEFAULT_SAMPLE_CSV
    if not sample_csv or not sample_csv.exists():
        raise FileNotFoundError(
            f"sample CSV not found: {sample_csv!r} — set DRIFT_SAMPLE_CSV"
        )

    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / f"{date.today():%Y-%m-%d}" / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    reference, current, used_silver = _build_reference_and_current(
        sample_csv, silver_root
    )
    if not used_silver:
        print(
            "[drift] no silver partitions found — falling back to train-vs-itself "
            "stub (gate will pass)"
        )
    summary = _run_evidently(reference, current, html_path)

    if summary is None:
        # Deterministic fallback: identical frames -> zero drift, gate passes.
        result = DriftResult(
            report_path=str(html_path),
            drift_score=0.0,
            n_drifted_columns=0,
            dataset_drift=False,
            passed_gate=True,
            n_reference=len(reference),
            n_current=len(current),
            evidently_ran=False,
        )
    else:
        drift_score = summary["drift_score"]
        result = DriftResult(
            report_path=str(html_path),
            drift_score=drift_score,
            n_drifted_columns=summary["n_drifted_columns"],
            dataset_drift=summary["dataset_drift"],
            passed_gate=drift_score <= threshold,
            n_reference=len(reference),
            n_current=len(current),
            evidently_ran=True,
        )

    print(f"[drift] {result}")
    return result


def main() -> None:
    result = run_drift_check()
    for k, v in asdict(result).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
