"""Evidently drift report — Phase 1 stub.

Pipeline (called from `airflow/dags/evaluate_and_monitor.py`):
    reference_df (training slice)  ─┐
                                    ├── Evidently Report (data + target drift)
    current_df   (recent reviews)  ─┘
                                    │
                                    ├── HTML report  ── uploaded to MinIO bucket
                                    │                   `monitoring/<date>/<type>.html`
                                    │
                                    └── pointer row inserted into Postgres
                                        `monitoring_reports` (s3 url, drift score,
                                         blocked_promotion flag)

Phase 1 only generates and stores the report. Phase 2 (Step 10) adds the
*blocking* logic — if drift_score exceeds the threshold OR negative-class
recall drops > 3%, the DAG fails and the model promotion is blocked.

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Optional

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

# Drift is computed over structured columns only for the thin slice.
# TODO (member, Phase 2): add text-drift via SentenceTransformer embeddings
# and a `TextOverviewPreset` once a real embedding model is registered.
NUMERICAL_COLS = ["rating"]
CATEGORICAL_COLS = ["source"]
TARGET_COL = "label"

DEFAULT_DRIFT_THRESHOLD = 0.5  # >= 50% of monitored columns drifted blocks promotion
DEFAULT_RECALL_NEG_DROP_THRESHOLD = 0.03  # recall_neg drop > 3% blocks promotion
# Kept for callers/tests that still pass the old kwarg name.
DEFAULT_F1_DROP_THRESHOLD = DEFAULT_RECALL_NEG_DROP_THRESHOLD
DEFAULT_BUCKET = "monitoring"


class PromotionBlocked(RuntimeError):
    """Raised by the DAG when drift / F1-drop crosses the threshold.

    The report has already been uploaded by the time this raises, so the
    Airflow task log + the dashboard's monitoring tab will both have a
    pointer to the failing report.
    """


@dataclass
class DriftResult:
    html: bytes
    drift_score: float          # share of monitored columns that drifted
    drifted_columns: list
    n_reference: int
    n_current: int

    def is_blocking(self, threshold: float = DEFAULT_DRIFT_THRESHOLD) -> bool:
        return self.drift_score >= threshold


def _predict_labels(model, df: pd.DataFrame) -> list:
    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return []
    return model.predict(df["text"].astype(str).tolist())


def compute_model_f1(model, df: pd.DataFrame) -> float:
    """Macro-F1 of `model` on `df` (expects `text` + `label` columns).

    Works for any model with a `.predict([text]) -> [label]` interface —
    sklearn Pipeline, mlflow.sklearn loaded model, or a custom wrapper.
    """
    from sklearn.metrics import f1_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0
    preds = _predict_labels(model, df)
    return float(f1_score(df["label"], preds, average="macro", zero_division=0))


def compute_model_recall_neg(model, df: pd.DataFrame) -> float:
    """Negative-class recall — primary gate metric for surge detection."""
    from sklearn.metrics import recall_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0
    preds = _predict_labels(model, df)
    return float(recall_score(df["label"], preds, pos_label="negative", zero_division=0))


# ---------- pure: report + score computation ----------

def _column_mapping() -> ColumnMapping:
    return ColumnMapping(
        target=TARGET_COL,
        numerical_features=NUMERICAL_COLS,
        categorical_features=CATEGORICAL_COLS,
    )


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    """Build the Evidently report and extract a single drift_score.

    Pure function — no DB, no S3. Easy to unit test.
    """
    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data=reference_df,
        current_data=current_df,
        column_mapping=_column_mapping(),
    )

    # `as_dict()` shape: {"metrics": [{...}, ...]} — we want the DataDriftTable.
    payload = report.as_dict()
    drift_score = 0.0
    drifted_columns: list = []
    for m in payload.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            drift_score = float(res.get("share_of_drifted_columns", 0.0))
        if m.get("metric") == "DataDriftTable":
            cols = m.get("result", {}).get("drift_by_columns", {})
            drifted_columns = [c for c, v in cols.items() if v.get("drift_detected")]

    # Evidently's `save_html` wants a filesystem path. Round-trip via a temp file.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        report.save_html(tmp_path)
        with open(tmp_path, "rb") as fh:
            html = fh.read()
    finally:
        import os as _os
        try:
            _os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    return DriftResult(
        html=html,
        drift_score=drift_score,
        drifted_columns=drifted_columns,
        n_reference=len(reference_df),
        n_current=len(current_df),
    )


# ---------- side-effects: MinIO + Postgres ----------

def upload_html_to_minio(
    minio_client,
    html: bytes,
    run_date: date,
    report_type: str,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    """Upload report HTML to `s3://<bucket>/<date>/<type>.html`. Returns s3 URL."""
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
    """Insert a row into monitoring_reports and return its id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO monitoring_reports "
            "(run_date, report_type, report_url, drift_score, blocked_promotion) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (run_date, report_type, s3_url, drift_score, blocked),
        )
        return cur.fetchone()[0]


# ---------- orchestrator ----------

def evaluate(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    conn,
    minio_client,
    run_date: Optional[date] = None,
    drift_threshold: float = DEFAULT_DRIFT_THRESHOLD,
    recall_neg_drop_threshold: float = DEFAULT_RECALL_NEG_DROP_THRESHOLD,
    f1_drop_threshold: float | None = None,
    report_type: str = "data_drift",
    bucket: str = DEFAULT_BUCKET,
    model=None,
    raise_on_block: bool = False,
) -> dict:
    """End-to-end: compute drift, upload HTML, insert pointer row, gate promotion.

    If `model` is provided, also scores reference and current slices. The
    result blocks promotion when EITHER:
        - drift_score >= drift_threshold, OR
        - reference_recall_neg - current_recall_neg > recall_neg_drop_threshold

    Macro-F1 is still computed for observability but does not gate promotion.

    Order of operations: upload + insert FIRST, then raise. That way the
    failing report is still discoverable in the dashboard.

    `raise_on_block=True` raises PromotionBlocked when blocked — used by
    the Airflow DAG so the task fails red. Default is False so unit/integration
    callers can inspect the dict.
    """
    run_date = run_date or date.today()
    drift = compute_drift(reference_df, current_df)

    if f1_drop_threshold is not None:
        recall_neg_drop_threshold = f1_drop_threshold

    reference_f1: Optional[float] = None
    current_f1: Optional[float] = None
    f1_drop: Optional[float] = None
    reference_recall_neg: Optional[float] = None
    current_recall_neg: Optional[float] = None
    recall_neg_drop: Optional[float] = None
    if model is not None:
        reference_f1 = compute_model_f1(model, reference_df)
        current_f1 = compute_model_f1(model, current_df)
        f1_drop = reference_f1 - current_f1
        reference_recall_neg = compute_model_recall_neg(model, reference_df)
        current_recall_neg = compute_model_recall_neg(model, current_df)
        recall_neg_drop = reference_recall_neg - current_recall_neg

    drift_blocks = drift.is_blocking(drift_threshold)
    recall_neg_blocks = (
        (recall_neg_drop is not None) and (recall_neg_drop > recall_neg_drop_threshold)
    )
    blocked = drift_blocks or recall_neg_blocks

    s3_url = upload_html_to_minio(
        minio_client, drift.html, run_date, report_type, bucket=bucket
    )
    row_id = insert_pointer_row(
        conn, run_date, report_type, s3_url, drift.drift_score, blocked
    )

    result = {
        "report_id": row_id,
        "s3_url": s3_url,
        "drift_score": drift.drift_score,
        "drifted_columns": drift.drifted_columns,
        "reference_f1": reference_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "reference_recall_neg": reference_recall_neg,
        "current_recall_neg": current_recall_neg,
        "recall_neg_drop": recall_neg_drop,
        "drift_blocks": drift_blocks,
        "recall_neg_blocks": recall_neg_blocks,
        "blocked_promotion": blocked,
    }

    if blocked and raise_on_block:
        reasons = []
        if drift_blocks:
            reasons.append(
                f"drift_score={drift.drift_score:.3f} >= {drift_threshold:.3f}"
            )
        if recall_neg_blocks:
            reasons.append(
                f"recall_neg_drop={recall_neg_drop:.3f} > {recall_neg_drop_threshold:.3f} "
                f"(ref={reference_recall_neg:.3f}, cur={current_recall_neg:.3f})"
            )
        raise PromotionBlocked(
            "Model promotion blocked: " + "; ".join(reasons) +
            f". Report: {s3_url}"
        )

    return result


def _summary_for_log(result: dict) -> str:
    return json.dumps({k: v for k, v in result.items() if k != "html"}, default=str)
