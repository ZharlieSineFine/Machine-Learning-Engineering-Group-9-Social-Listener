"""Evidently drift report + promotion gate.

Two layers live here, sharing one Evidently code path:

1. **Observational drift** (`run_drift_check`) — reference = training/sample
   distribution, current = the most recent silver partitions. Always returns a
   ``DriftResult`` (never raises on Evidently internals) so the every-6h
   ``evaluate_and_monitor`` DAG stays green while the pipeline settles. Falls
   back to a train-vs-itself stub when no silver data exists yet.

2. **Promotion gate** (`evaluate`) — given two labelled frames + a trained model,
   it computes data drift AND model performance (macro-F1 + negative-class
   recall), uploads the HTML report to MinIO, writes a pointer row to
   ``monitoring_reports``, and blocks promotion when EITHER:

       * drift_score >= drift_threshold, OR
       * f1_macro drops > f1_drop_threshold, OR
       * recall on the ``negative`` class drops > recall_neg_drop_threshold.

   Upload + DB insert happen **before** any raise, so a blocking report stays
   discoverable in the dashboard. Used by ``medallion_train_cycle`` between
   ``train`` and ``promote``.

Runnable three ways, like ``models/train.py``: from an Airflow PythonOperator,
from the CLI (``python monitoring/drift_checks.py``), or from a unit test.
Evidently/sklearn imports are deferred into the functions that need them so the
module imports cleanly even where those deps are absent.

Owner: Charlie + Ha (Monitoring).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
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
# the medallion DAG produces. ``current`` data is read from the most recent
# partitions here. Falls back to the train-vs-itself stub if empty.
DEFAULT_SILVER_ROOT = _first_existing(
    Path(os.getenv("DRIFT_SILVER_ROOT", "")) if os.getenv("DRIFT_SILVER_ROOT") else None,
    Path("/opt/project/data/silver/reviews"),  # Airflow container mount
    _REPO_ROOT / "data" / "silver" / "reviews",  # local checkout
)

# How many recent review_date partitions form the ``current`` window.
DRIFT_RECENT_PARTITIONS = int(os.getenv("DRIFT_RECENT_PARTITIONS", "7"))

# Where the observational HTML report is written before upload to MinIO.
DEFAULT_REPORT_DIR = Path(
    os.getenv("DRIFT_REPORT_DIR", "/opt/project/monitoring/reports")
)

# ---------------------------------------------------------------------------
# Gate configuration
# ---------------------------------------------------------------------------
# Columns the drift report reasons over. Free text is excluded — Evidently would
# treat it as a high-cardinality categorical and add only noise. Built defensively
# (see ``_column_mapping``): only columns present in BOTH frames are used, so the
# gate works whether ``current`` comes from silver (rating/source) or the Gold
# training frame (text/label only).
NUMERICAL_COLS = ["text_len", "rating"]
CATEGORICAL_COLS = ["source"]
TARGET_COL = "label"
NEG_LABEL = "negative"

# Share of monitored columns that must drift to block promotion (README: "> 0.3").
DEFAULT_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
# F1-macro drop that blocks promotion (ARCHITECTURE/WORKFLOW: "> 3%").
DEFAULT_F1_DROP_THRESHOLD = float(os.getenv("DRIFT_F1_DROP_THRESHOLD", "0.03"))
# Negative-class recall drop that blocks promotion (business-critical class).
DEFAULT_RECALL_NEG_DROP_THRESHOLD = float(
    os.getenv("DRIFT_RECALL_NEG_DROP_THRESHOLD", "0.05")
)
DEFAULT_BUCKET = "monitoring"

# Statistical test Evidently uses per column. PSI (Population Stability Index)
# is the industry-standard distribution-shift metric: it bins each column and
# sums (cur% - ref%) * ln(cur% / ref%) across bins. Standard bands are
# < 0.1 no shift, 0.1-0.2 moderate, > 0.2 significant — so a column "drifts"
# when its PSI clears DRIFT_PSI_THRESHOLD. Applied to data features, the target
# label, AND the prediction column, so all three drift types share one boundary.
DRIFT_STATTEST = os.getenv("DRIFT_STATTEST", "psi")
DRIFT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2"))

# Back-compat alias for the env-driven observational threshold.
DRIFT_THRESHOLD = DEFAULT_DRIFT_THRESHOLD


class PromotionBlocked(RuntimeError):
    """Raised by ``evaluate`` (when ``raise_on_block=True``) on a blocking gate.

    The report has already been uploaded and the pointer row written by the time
    this raises, so the Airflow task log + the dashboard both have a pointer to
    the failing report.
    """


@dataclass
class DriftResult:
    """Unified result for both the observational check and the gate.

    The observational path (``run_drift_check``) sets ``report_path`` /
    ``passed_gate`` / ``evidently_ran``; the gate path (``compute_drift``) sets
    ``html`` / ``drifted_columns``. Every field has a default so either path can
    construct it with only the fields it knows.
    """

    html: bytes = b""
    drift_score: float = 0.0            # share of drifted columns (0.0 .. 1.0)
    drifted_columns: list = field(default_factory=list)
    psi_by_column: dict = field(default_factory=dict)  # {column: PSI statistic}
    n_reference: int = 0
    n_current: int = 0
    report_path: Optional[str] = None
    n_drifted_columns: int = 0
    dataset_drift: bool = False
    passed_gate: bool = True            # True == no actionable drift
    evidently_ran: bool = False         # False if we fell back to the no-op stub

    def is_blocking(self, threshold: float = DEFAULT_DRIFT_THRESHOLD) -> bool:
        return self.drift_score >= threshold


# ---------------------------------------------------------------------------
# Model performance — macro-F1 + negative-class recall
# ---------------------------------------------------------------------------
def compute_model_f1(model, df: pd.DataFrame) -> float:
    """Macro-F1 of ``model`` on ``df`` (expects ``text`` + ``label`` columns).

    Works for any model with a ``.predict([text]) -> [label]`` interface —
    sklearn Pipeline, mlflow.sklearn loaded model, or a custom wrapper.
    """
    from sklearn.metrics import f1_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0
    preds = model.predict(df["text"].astype(str).tolist())
    return float(f1_score(df["label"], preds, average="macro", zero_division=0))


def compute_model_recall_neg(model, df: pd.DataFrame) -> float:
    """Recall of the ``negative`` class — the class we most care about catching."""
    from sklearn.metrics import recall_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0
    preds = model.predict(df["text"].astype(str).tolist())
    return float(
        recall_score(
            df["label"], preds, labels=[NEG_LABEL], average="macro", zero_division=0
        )
    )


def _metrics_from_predictions(df: pd.DataFrame) -> tuple[float, float]:
    """Return ``(f1_macro, recall_neg)`` from a frame carrying ``label`` +
    ``prediction`` columns — i.e. predictions already scored by the caller.

    ``evaluate`` scores each side with exactly ONE ``predict`` call and reuses
    those predictions for both the performance metrics here AND the
    prediction-drift column. Re-predicting would corrupt stateful models whose
    behaviour changes after the first call.
    """
    from sklearn.metrics import f1_score, recall_score

    df = df.dropna(subset=["label", "prediction"])
    if len(df) == 0:
        return 0.0, 0.0
    f1 = float(f1_score(df["label"], df["prediction"], average="macro", zero_division=0))
    recall_neg = float(
        recall_score(
            df["label"], df["prediction"], labels=[NEG_LABEL], average="macro",
            zero_division=0,
        )
    )
    return f1, recall_neg


# ---------------------------------------------------------------------------
# Drift — pure compute (no DB, no S3)
# ---------------------------------------------------------------------------
def _column_mapping(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    """Evidently ColumnMapping restricted to columns present in BOTH frames."""
    from evidently import ColumnMapping

    shared = set(reference_df.columns) & set(current_df.columns)
    return ColumnMapping(
        target=TARGET_COL if TARGET_COL in shared else None,
        numerical_features=[c for c in NUMERICAL_COLS if c in shared],
        categorical_features=[c for c in CATEGORICAL_COLS if c in shared],
    )


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    """Build the Evidently report and extract a single drift_score.

    Pure function — no DB, no S3. Reduces both frames to their shared columns so
    Evidently never trips on a column present on one side only.
    """
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    # Drift only over the declared monitored columns that exist on BOTH sides.
    # Free text (``text``) is deliberately excluded — it's kept on the frame for
    # model scoring (``_score_model``) but would only add noise to the report.
    monitored = [
        c
        for c in (NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])
        if c in reference_df.columns and c in current_df.columns
    ]
    ref = reference_df[monitored]
    cur = current_df[monitored]

    metrics = [DataDriftPreset(stattest=DRIFT_STATTEST, stattest_threshold=DRIFT_PSI_THRESHOLD)]
    if TARGET_COL in monitored:
        metrics.append(
            TargetDriftPreset(stattest=DRIFT_STATTEST, stattest_threshold=DRIFT_PSI_THRESHOLD)
        )

    report = Report(metrics=metrics)
    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=_column_mapping(ref, cur),
    )

    payload = report.as_dict()
    drift_score = 0.0
    drifted_columns: list = []
    psi_by_column: dict = {}
    n_drifted = 0
    dataset_drift = False
    for m in payload.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            drift_score = float(res.get("share_of_drifted_columns", 0.0))
            n_drifted = int(res.get("number_of_drifted_columns", 0))
            dataset_drift = bool(res.get("dataset_drift", False))
        if m.get("metric") == "DataDriftTable":
            cols = m.get("result", {}).get("drift_by_columns", {})
            drifted_columns = [c for c, v in cols.items() if v.get("drift_detected")]
            # With stattest="psi", each column's drift_score IS its PSI statistic.
            psi_by_column = {
                c: float(v["drift_score"])
                for c, v in cols.items()
                if v.get("drift_score") is not None
            }

    # Evidently's save_html wants a filesystem path — round-trip via a temp file.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        report.save_html(tmp_path)
        html = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    return DriftResult(
        html=html,
        drift_score=drift_score,
        drifted_columns=drifted_columns,
        psi_by_column=psi_by_column,
        n_reference=len(ref),
        n_current=len(cur),
        n_drifted_columns=n_drifted,
        dataset_drift=dataset_drift,
        evidently_ran=True,
    )


# ---------------------------------------------------------------------------
# Side-effects — MinIO + Postgres
# ---------------------------------------------------------------------------
def upload_html_to_minio(
    minio_client,
    html: bytes,
    run_date: date,
    report_type: str,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Upload report HTML to ``s3://<bucket>/<date>/<type>.html``. Returns the URL."""
    key = f"{run_date.isoformat()}/{report_type}.html"
    minio_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=html,
        ContentType="text/html",
    )
    return f"s3://{bucket}/{key}"


def insert_pointer_row(
    conn,
    run_date: date,
    report_type: str,
    s3_url: str,
    drift_score: float,
    blocked: bool,
) -> int:
    """Insert a row into ``monitoring_reports`` and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO monitoring_reports "
            "(run_date, report_type, report_url, drift_score, blocked_promotion) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (run_date, report_type, s3_url, drift_score, blocked),
        )
        return cur.fetchone()[0]


# ---------------------------------------------------------------------------
# Promotion gate — drift + performance, uploads, gates
# ---------------------------------------------------------------------------
def evaluate(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    conn,
    minio_client,
    run_date: Optional[date] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    f1_drop_threshold: float = DEFAULT_F1_DROP_THRESHOLD,
    recall_neg_drop_threshold: float = DEFAULT_RECALL_NEG_DROP_THRESHOLD,
    report_type: str = "data_drift",
    bucket: str = DEFAULT_BUCKET,
    model=None,
    raise_on_block: bool = False,
) -> dict:
    """End-to-end: compute drift, score the model, upload HTML, gate promotion.

    Blocks promotion when EITHER data drift crosses ``drift_threshold`` OR (when a
    ``model`` is given) macro-F1 drops past ``f1_drop_threshold`` OR negative-class
    recall drops past ``recall_neg_drop_threshold``.

    The uploaded report ALSO covers target + prediction-distribution drift (PSI),
    so the gate report aligns with the observational monitor — but prediction
    drift is informational here and does NOT change the promote/block decision
    (that stays data drift + measured performance).

    Order of operations: upload + insert FIRST, then raise — so the failing report
    is still discoverable in the dashboard. ``raise_on_block=True`` raises
    ``PromotionBlocked`` on a block (used by the DAG so the task fails red);
    default ``False`` lets callers inspect the returned dict.
    """
    run_date = run_date or date.today()

    # Score the model ONCE per side (reference then current); reuse those
    # predictions for both the performance metrics and the prediction-drift
    # column, so stateful models still see exactly one predict() call per side.
    ref = reference_df.copy()
    cur = current_df.copy()
    reference_f1 = current_f1 = f1_drop = None
    reference_recall_neg = current_recall_neg = recall_neg_drop = None
    if model is not None:
        if "text" in ref.columns:
            ref["prediction"] = model.predict(ref["text"].astype(str).tolist())
            reference_f1, reference_recall_neg = _metrics_from_predictions(ref)
        if "text" in cur.columns:
            cur["prediction"] = model.predict(cur["text"].astype(str).tolist())
            current_f1, current_recall_neg = _metrics_from_predictions(cur)
        if reference_f1 is not None and current_f1 is not None:
            f1_drop = reference_f1 - current_f1
            recall_neg_drop = reference_recall_neg - current_recall_neg

    # model=None: predictions are already attached above (or absent), so the
    # report reuses them and never re-scores.
    rpt = compute_drift_report(ref, cur, model=None)
    data_drift_score = rpt["data_drift_score"]

    drift_blocks = data_drift_score >= drift_threshold
    f1_blocks = (f1_drop is not None) and (f1_drop > f1_drop_threshold)
    recall_neg_blocks = (
        recall_neg_drop is not None
    ) and (recall_neg_drop > recall_neg_drop_threshold)
    blocked = drift_blocks or f1_blocks or recall_neg_blocks

    s3_url = upload_html_to_minio(
        minio_client, rpt["html"], run_date, report_type, bucket=bucket
    )
    row_id = insert_pointer_row(
        conn, run_date, report_type, s3_url, data_drift_score, blocked
    )

    result = {
        "report_id": row_id,
        "s3_url": s3_url,
        "report_url": s3_url,
        "drift_score": data_drift_score,
        "drifted_columns": rpt["drifted_columns"],
        "psi_by_column": rpt["psi_by_column"],
        "target_drift_score": rpt["target_drift_score"],
        "target_drift": rpt["target_drift"],
        "prediction_drift_score": rpt["prediction_drift_score"],
        "prediction_drift": rpt["prediction_drift"],
        "reference_f1": reference_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "f1_blocks": f1_blocks,
        "reference_recall_neg": reference_recall_neg,
        "current_recall_neg": current_recall_neg,
        "recall_neg_drop": recall_neg_drop,
        "recall_neg_blocks": recall_neg_blocks,
        "drift_blocks": drift_blocks,
        "blocked_promotion": blocked,
    }

    if blocked and raise_on_block:
        reasons = []
        if drift_blocks:
            reasons.append(
                f"drift_score={data_drift_score:.3f} >= {drift_threshold:.3f}"
            )
        if f1_blocks:
            reasons.append(
                f"f1_drop={f1_drop:.3f} > {f1_drop_threshold:.3f} "
                f"(ref={reference_f1:.3f}, cur={current_f1:.3f})"
            )
        if recall_neg_blocks:
            reasons.append(
                f"recall_neg_drop={recall_neg_drop:.3f} > "
                f"{recall_neg_drop_threshold:.3f} "
                f"(ref={reference_recall_neg:.3f}, cur={current_recall_neg:.3f})"
            )
        raise PromotionBlocked(
            "Model promotion blocked: " + "; ".join(reasons) + f". Report: {s3_url}"
        )

    return result


# ---------------------------------------------------------------------------
# Observational feature frame + silver loading
# ---------------------------------------------------------------------------
def _features_from_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Project raw reviews onto the columns we monitor.

    Mirrors the "data drift" row in monitoring/README.md: text-length
    distribution, rating, source mix, and label distribution. Keeping this tiny
    and explicit gives Evidently typed numeric + categorical columns to reason
    about instead of free text it would just ignore. ``text`` is preserved so the
    gate can also score a model on the same frame.
    """
    out = pd.DataFrame()
    out["text"] = df["text"].fillna("").astype(str) if "text" in df.columns else ""
    out["text_len"] = out["text"].str.len()
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
    exist yet, ``current`` falls back to a copy of ``reference`` (train vs. itself
    -> guaranteed-pass gate) and ``used_silver`` is False, preserving the "DAG
    stays green while wiring settles" property. Both frames are reduced to their
    shared columns so Evidently sees a matching schema.
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
# Observational entry point — never raises on Evidently internals
# ---------------------------------------------------------------------------
def run_drift_check(
    sample_csv: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    silver_root: Optional[Path] = None,
) -> DriftResult:
    """Run the observational drift check and return a structured result.

    ``reference`` is the training/sample distribution; ``current`` is the most
    recent silver partitions. Falls back to train-vs-itself when no silver data
    exists yet. Always returns (never raises on Evidently internals) so the
    every-6h Airflow task stays green while the pipeline is still being wired.
    """
    sample_csv = Path(sample_csv) if sample_csv else DEFAULT_SAMPLE_CSV
    if not sample_csv or not sample_csv.exists():
        raise FileNotFoundError(
            f"sample CSV not found: {sample_csv!r} — set DRIFT_SAMPLE_CSV"
        )

    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
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

    try:
        drift = compute_drift(reference, current)
        html_path.write_bytes(drift.html)
        drift.report_path = str(html_path)
        drift.passed_gate = not drift.is_blocking(threshold)
        drift.html = b""  # report is on disk now; keep the result XCom-light
        print(f"[drift] {_summary_for_log(asdict(drift))}")
        return drift
    except Exception as exc:  # never kill the observational DAG on Evidently
        print(f"[drift] Evidently unavailable/incompatible, using stub: {exc}")
        html_path.write_text("<html><body>drift stub: evidently unavailable</body></html>")
        return DriftResult(
            report_path=str(html_path),
            drift_score=0.0,
            passed_gate=True,
            n_reference=len(reference),
            n_current=len(current),
            evidently_ran=False,
        )


# ---------------------------------------------------------------------------
# Target + prediction drift (observational monitor)
# ---------------------------------------------------------------------------
@dataclass
class MonitorResult:
    """Full observational drift summary: data + target + (optional) prediction.

    ``target_*`` and ``prediction_*`` are None when the relevant column was
    unavailable (no label on current / no model to score). ``blocked`` is True
    when ANY computed drift is actionable — the monitor DAG uses it to decide
    whether to fire a retrain.
    """

    report_path: str
    data_drift_score: float = 0.0
    dataset_drift: bool = False
    psi_by_column: dict = field(default_factory=dict)
    target_drift_score: Optional[float] = None
    target_drift: bool = False
    prediction_drift_score: Optional[float] = None
    prediction_drift: bool = False
    blocked: bool = False
    n_reference: int = 0
    n_current: int = 0
    used_model: bool = False
    evidently_ran: bool = False


def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    model=None,
) -> dict:
    """One Evidently report: data drift + target drift + (optional) prediction drift.

    * **Data drift** — feature columns (text_len / rating / source).
    * **Target drift** — the ground-truth ``label`` distribution (present when
      ``current`` carries a label; the monitor derives it from ``rating``).
    * **Prediction drift** — the model's predicted-label distribution, scored on
      both frames when a ``model`` is given. This is what catches a model whose
      output mix shifts even before labels are available.

    Pure compute (no DB/S3). Returns scores, drift booleans, and ``html`` bytes.
    """
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import ColumnDriftMetric
    from evidently.report import Report

    ref = reference_df.copy()
    cur = current_df.copy()

    # Establish a `prediction` column on both frames for prediction-distribution
    # drift. The current frame may ALREADY carry one (the predictions-table path
    # supplies real logged predictions) — in that case we keep it and only score
    # the reference to get the expected distribution. Otherwise we score the model
    # on both sides (the silver-window fallback). Either way it's one predict call
    # per side, so stateful models compare correctly.
    ref_has_pred = "prediction" in ref.columns and ref["prediction"].notna().any()
    cur_has_pred = "prediction" in cur.columns and cur["prediction"].notna().any()
    has_pred = False
    if model is not None and "text" in ref.columns and "text" in cur.columns:
        # Silver fallback: score the model on both sides (one predict call each).
        try:
            if not ref_has_pred:
                ref["prediction"] = model.predict(ref["text"].astype(str).tolist())
            if not cur_has_pred:
                cur["prediction"] = model.predict(cur["text"].astype(str).tolist())
            has_pred = True
        except Exception as exc:  # scoring is best-effort; degrade to no pred drift
            print(f"[drift] prediction scoring failed, skipping prediction drift: {exc}")
    elif model is not None and "text" in ref.columns and cur_has_pred:
        # Table path: current carries logged predictions; only score the reference.
        try:
            if not ref_has_pred:
                ref["prediction"] = model.predict(ref["text"].astype(str).tolist())
            has_pred = True
        except Exception as exc:
            print(f"[drift] reference scoring failed, skipping prediction drift: {exc}")
    elif ref_has_pred and cur_has_pred:
        # Both frames already carry predictions — no model needed.
        has_pred = True

    # Prediction drift needs the column on BOTH sides; drop a lone one to avoid a
    # mismatched schema in the Evidently report.
    if not has_pred:
        for frame in (ref, cur):
            if "prediction" in frame.columns:
                frame.drop(columns=["prediction"], inplace=True)

    shared = set(ref.columns) & set(cur.columns)
    has_target = TARGET_COL in shared
    num = [c for c in NUMERICAL_COLS if c in shared]
    cat = [c for c in CATEGORICAL_COLS if c in shared]

    mapping = ColumnMapping(
        target=TARGET_COL if has_target else None,
        prediction="prediction" if has_pred else None,
        numerical_features=num,
        categorical_features=cat,
    )

    metrics = [DataDriftPreset(stattest=DRIFT_STATTEST, stattest_threshold=DRIFT_PSI_THRESHOLD)]
    if has_target:
        # covers target AND prediction columns
        metrics.append(
            TargetDriftPreset(stattest=DRIFT_STATTEST, stattest_threshold=DRIFT_PSI_THRESHOLD)
        )
    elif has_pred:
        # No target to anchor TargetDriftPreset, but we still want prediction
        # drift — add an explicit per-column check on the prediction distribution.
        metrics.append(
            ColumnDriftMetric(
                column_name="prediction",
                stattest=DRIFT_STATTEST,
                stattest_threshold=DRIFT_PSI_THRESHOLD,
            )
        )

    keep = (
        num
        + cat
        + ([TARGET_COL] if has_target else [])
        + (["prediction"] if has_pred else [])
    )
    report = Report(metrics=metrics)
    report.run(
        reference_data=ref[keep],
        current_data=cur[keep],
        column_mapping=mapping,
    )

    payload = report.as_dict()
    out = {
        "data_drift_score": 0.0,
        "dataset_drift": False,
        "drifted_columns": [],
        "psi_by_column": {},
        "target_drift_score": None,
        "target_drift": False,
        "prediction_drift_score": None,
        "prediction_drift": False,
        "used_model": has_pred,
    }
    for m in payload.get("metrics", []):
        name = m.get("metric")
        res = m.get("result", {})
        if name == "DatasetDriftMetric":
            out["data_drift_score"] = float(res.get("share_of_drifted_columns", 0.0))
            out["dataset_drift"] = bool(res.get("dataset_drift", False))
        elif name == "DataDriftTable":
            cols = res.get("drift_by_columns", {})
            out["drifted_columns"] = [
                c for c, v in cols.items() if v.get("drift_detected")
            ]
            # With stattest="psi", each column's drift_score IS its PSI statistic.
            out["psi_by_column"] = {
                c: float(v["drift_score"])
                for c, v in cols.items()
                if v.get("drift_score") is not None
            }
        elif name == "ColumnDriftMetric":
            col = res.get("column_name")
            score = res.get("drift_score")
            detected = bool(res.get("drift_detected", False))
            if col == TARGET_COL:
                out["target_drift_score"] = float(score) if score is not None else None
                out["target_drift"] = detected
            elif col == "prediction":
                out["prediction_drift_score"] = (
                    float(score) if score is not None else None
                )
                out["prediction_drift"] = detected

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        report.save_html(tmp_path)
        out["html"] = Path(tmp_path).read_bytes()
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    return out


# How many days of logged predictions form the ``current`` window when the
# predictions table is the source.
DRIFT_PRED_WINDOW_DAYS = int(os.getenv("DRIFT_PRED_WINDOW_DAYS", "7"))


def _load_recent_predictions(
    dsn: str, window_days: int = DRIFT_PRED_WINDOW_DAYS
) -> Optional[pd.DataFrame]:
    """Load logged predictions from the last ``window_days``, joined to reviews.

    Returns a raw reviews-like frame (``text``, ``predicted_label``, and
    ``rating`` / ``source`` recovered via ``review_id`` when present) or ``None``
    when the table is empty / unreachable — the caller then falls back to scoring
    the model on the silver window. Best-effort: any DB error degrades to None so
    the every-6h DAG stays green.
    """
    try:
        import psycopg2
    except Exception as exc:  # driver missing in some test envs
        print(f"[drift] psycopg2 unavailable, skipping predictions table: {exc}")
        return None

    try:
        conn = psycopg2.connect(dsn)
    except Exception as exc:
        print(f"[drift] could not connect for predictions lookup: {exc}")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT p.text, p.predicted_label, r.rating, r.source "
                "FROM predictions p "
                "LEFT JOIN reviews r ON p.review_id = r.id "
                "WHERE p.predicted_at >= NOW() - (%s || ' days')::interval",
                (window_days,),
            )
            rows = cur.fetchall()
    except Exception as exc:
        print(f"[drift] predictions lookup failed: {exc}")
        return None
    finally:
        conn.close()

    if not rows:
        return None
    return pd.DataFrame(rows, columns=["text", "predicted_label", "rating", "source"])


def _current_from_predictions(dsn: str) -> Optional[pd.DataFrame]:
    """Build the monitored ``current`` frame from the predictions table.

    Carries a real ``prediction`` column (logged predicted labels) plus
    rating-derived ``label`` and the usual data-drift features. Returns ``None``
    when no recent predictions exist.
    """
    raw = _load_recent_predictions(dsn)
    if raw is None or raw.empty:
        return None

    current = _features_from_reviews(raw)
    current["prediction"] = raw["predicted_label"].astype(str).values
    if "rating" in current.columns:
        from data.refine.build_gold import label_from_rating

        current["label"] = [
            label_from_rating(r) if pd.notna(r) else None for r in current["rating"]
        ]
        # Keep label only when ratings actually resolved it; rows without a rating
        # (unjoined review_id) can't contribute target drift. If none resolved,
        # drop the column entirely so prediction drift still runs on its own.
        if current["label"].notna().any():
            current = current.dropna(subset=["label"])
        else:
            current = current.drop(columns=["label"])
    return current


def _build_monitor_frames(
    sample_csv: Path,
    silver_root: Optional[Path] = None,
    n_partitions: int = DRIFT_RECENT_PARTITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Reference = sample/training CSV (text+label); current = recent silver with
    ``label`` derived from ``rating`` (silver carries no label of its own).

    Returns ``(reference, current, used_silver)``. Falls back to reference-vs-itself
    when no silver exists yet so the monitor stays green.
    """
    reference = _features_from_reviews(pd.read_csv(sample_csv))

    silver_root = silver_root or DEFAULT_SILVER_ROOT
    recent = _load_recent_silver(silver_root, n_partitions)
    if recent is None:
        return reference, reference.copy(), False

    current = _features_from_reviews(recent)
    if "rating" in current.columns:
        from data.refine.build_gold import label_from_rating

        current["label"] = [
            label_from_rating(r) if pd.notna(r) else None for r in current["rating"]
        ]
        current = current.dropna(subset=["label"])
    return reference, current, True


def run_monitor_drift(
    sample_csv: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    silver_root: Optional[Path] = None,
    model=None,
    dsn: Optional[str] = None,
) -> MonitorResult:
    """Observational monitor: data + target + prediction drift. Writes the HTML
    report to disk and returns a ``MonitorResult``.

    ``current`` source (both/fallback): when ``dsn`` is given and the predictions
    table has recent rows, those logged predictions are the current window;
    otherwise it falls back to the recent silver window, scoring ``model`` to get
    the prediction distribution.

    ``blocked`` is True when data drift crosses ``threshold`` OR target/prediction
    drift is detected — the monitor DAG uses it to fire a retrain. Never raises on
    Evidently internals so the every-6h DAG stays green.
    """
    sample_csv = Path(sample_csv) if sample_csv else DEFAULT_SAMPLE_CSV
    if not sample_csv or not sample_csv.exists():
        raise FileNotFoundError(
            f"sample CSV not found: {sample_csv!r} — set DRIFT_SAMPLE_CSV"
        )

    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    html_path = report_dir / f"{date.today():%Y-%m-%d}" / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer logged predictions when available; otherwise the silver window.
    reference = _features_from_reviews(pd.read_csv(sample_csv))
    current = _current_from_predictions(dsn) if dsn else None
    if current is not None:
        print(f"[drift] current window = {len(current)} logged predictions")
    else:
        reference, current, used_silver = _build_monitor_frames(sample_csv, silver_root)
        if not used_silver:
            print("[drift] no silver partitions — monitoring reference vs itself (no drift)")

    try:
        rpt = compute_drift_report(reference, current, model=model)
        html_path.write_bytes(rpt.pop("html"))
        blocked = (
            rpt["data_drift_score"] >= threshold
            or rpt["target_drift"]
            or rpt["prediction_drift"]
        )
        result = MonitorResult(
            report_path=str(html_path),
            data_drift_score=rpt["data_drift_score"],
            dataset_drift=rpt["dataset_drift"],
            psi_by_column=rpt["psi_by_column"],
            target_drift_score=rpt["target_drift_score"],
            target_drift=rpt["target_drift"],
            prediction_drift_score=rpt["prediction_drift_score"],
            prediction_drift=rpt["prediction_drift"],
            blocked=blocked,
            n_reference=len(reference),
            n_current=len(current),
            used_model=rpt["used_model"],
            evidently_ran=True,
        )
        print(f"[drift] {_summary_for_log(asdict(result))}")
        return result
    except Exception as exc:  # never kill the observational DAG on Evidently
        print(f"[drift] Evidently unavailable/incompatible, using stub: {exc}")
        html_path.write_text("<html><body>drift stub: evidently unavailable</body></html>")
        return MonitorResult(
            report_path=str(html_path),
            n_reference=len(reference),
            n_current=len(current),
            evidently_ran=False,
        )


def _summary_for_log(result: dict) -> str:
    return json.dumps({k: v for k, v in result.items() if k != "html"}, default=str)


def main() -> None:
    result = run_drift_check()
    for k, v in asdict(result).items():
        if k == "html":
            v = f"<{len(v)} bytes>"
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
