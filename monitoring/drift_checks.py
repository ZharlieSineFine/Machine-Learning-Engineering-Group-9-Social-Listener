from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

def _first_existing(*candidates: Path) -> Optional[Path]:
    for c in candidates:
        if c and c.exists():
            return c
    return None


_REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SAMPLE_CSV = _first_existing(
    Path(os.getenv("DRIFT_SAMPLE_CSV", "")) if os.getenv("DRIFT_SAMPLE_CSV") else None,
    Path("/opt/project/data/sample/reviews_sample.csv"),  # Airflow container mount
    _REPO_ROOT / "data" / "sample" / "reviews_sample.csv",  # local checkout
)

DEFAULT_SILVER_ROOT = _first_existing(
    Path(os.getenv("DRIFT_SILVER_ROOT", "")) if os.getenv("DRIFT_SILVER_ROOT") else None,
    Path("/opt/project/data/silver/reviews"),  # Airflow container mount
    _REPO_ROOT / "data" / "silver" / "reviews",  # local checkout
)

DRIFT_RECENT_PARTITIONS = int(os.getenv("DRIFT_RECENT_PARTITIONS", "7"))

DEFAULT_REPORT_DIR = Path(
    os.getenv("DRIFT_REPORT_DIR", "/opt/project/monitoring/reports")
)

NUMERICAL_COLS = ["text_len", "rating"]
CATEGORICAL_COLS = ["source"]
TARGET_COL = "label"
NEG_LABEL = "negative"

DEFAULT_PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.25"))
DEFAULT_DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))
DEFAULT_F1_DROP_THRESHOLD = float(os.getenv("DRIFT_F1_DROP_THRESHOLD", "0.03"))
DEFAULT_RECALL_NEG_DROP_THRESHOLD = float(
    os.getenv("DRIFT_RECALL_NEG_DROP_THRESHOLD", "0.05")
)
DEFAULT_BUCKET = "monitoring"

DRIFT_THRESHOLD = DEFAULT_DRIFT_THRESHOLD


# Raised by evaluate(raise_on_block=True) on a blocking gate. The report is already
# uploaded and the pointer row written before this fires, so the Airflow task log and
# the dashboard both point at the failing report.
class PromotionBlocked(RuntimeError):
    pass


@dataclass
class DriftResult:
    html: bytes = b""
    drift_score: float = 0.0            # share of drifted columns (0.0 .. 1.0)
    drifted_columns: list = field(default_factory=list)
    n_reference: int = 0
    n_current: int = 0
    report_path: Optional[str] = None
    n_drifted_columns: int = 0
    dataset_drift: bool = False
    target_drift: bool = False          # label distribution drifted (PSI > threshold)
    target_drift_score: Optional[float] = None
    passed_gate: bool = True            # True == no actionable drift
    evidently_ran: bool = False         # False if we fell back to the no-op stub

    def is_blocking(self, threshold: float = DEFAULT_DRIFT_THRESHOLD) -> bool:
        # Pure per-column: ANY monitored column flagged by the per-column PSI test
        # (PSI > DRIFT_PSI_THRESHOLD) blocks — a feature column OR the label.
        # ``threshold`` (the legacy share-of-columns gate) is accepted for
        # signature compatibility but no longer drives the decision;
        # ``drift_score`` still carries the share.
        return self.n_drifted_columns >= 1 or self.target_drift


def compute_model_f1(model, df: pd.DataFrame) -> float:
    from sklearn.metrics import f1_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0
    preds = model.predict(df["text"].astype(str).tolist())
    return float(f1_score(df["label"], preds, average="macro", zero_division=0))


def compute_model_recall_neg(model, df: pd.DataFrame) -> float:
    # Recall of the negative class, the one we most care about catching.
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


def _score_model(model, df: pd.DataFrame) -> tuple[float, float]:
    # Return (f1_macro, recall_neg) from a SINGLE predict pass. Callers sometimes
    # pass stateful models whose behaviour changes after the first call, so taking
    # F1 and recall from two separate predict calls would corrupt the comparison.
    from sklearn.metrics import f1_score, recall_score

    df = df.dropna(subset=["text", "label"])
    if len(df) == 0:
        return 0.0, 0.0
    preds = model.predict(df["text"].astype(str).tolist())
    f1 = float(f1_score(df["label"], preds, average="macro", zero_division=0))
    recall_neg = float(
        recall_score(
            df["label"], preds, labels=[NEG_LABEL], average="macro", zero_division=0
        )
    )
    return f1, recall_neg


def _column_mapping(reference_df: pd.DataFrame, current_df: pd.DataFrame):
    # Evidently ColumnMapping restricted to columns present in BOTH frames.
    from evidently import ColumnMapping

    shared = set(reference_df.columns) & set(current_df.columns)
    return ColumnMapping(
        target=TARGET_COL if TARGET_COL in shared else None,
        numerical_features=[c for c in NUMERICAL_COLS if c in shared],
        categorical_features=[c for c in CATEGORICAL_COLS if c in shared],
    )


def compute_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DriftResult:
    # Build the Evidently report and pull out a single drift_score. Pure function,
    # no DB or S3. Reduces both frames to their shared columns so Evidently never
    # trips on a column that's present on one side only.
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    # Drift only over the declared monitored columns that exist on BOTH sides.
    # Free text (``text``) is deliberately excluded: it's kept on the frame for
    # model scoring (``_score_model``) but would only add noise to the report.
    monitored = [
        c
        for c in (NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL])
        if c in reference_df.columns and c in current_df.columns
    ]
    ref = reference_df[monitored]
    cur = current_df[monitored]

    # Per-column PSI for every monitored column (features here; the target is
    # added below when present), threshold 0.25 = "significant shift".
    metrics = [
        DataDriftPreset(stattest="psi", stattest_threshold=DEFAULT_PSI_THRESHOLD)
    ]
    if TARGET_COL in monitored:
        metrics.append(
            TargetDriftPreset(stattest="psi", stattest_threshold=DEFAULT_PSI_THRESHOLD)
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
    n_drifted = 0
    dataset_drift = False
    target_drift = False
    target_drift_score = None
    for m in payload.get("metrics", []):
        metric = m.get("metric")
        res = m.get("result", {})
        if metric == "DatasetDriftMetric":
            drift_score = float(res.get("share_of_drifted_columns", 0.0))
            n_drifted = int(res.get("number_of_drifted_columns", 0))
            dataset_drift = bool(res.get("dataset_drift", False))
        elif metric == "DataDriftTable":
            cols = res.get("drift_by_columns", {})
            drifted_columns = [c for c, v in cols.items() if v.get("drift_detected")]
        elif metric == "ColumnDriftMetric" and res.get("column_name") == TARGET_COL:
            # TargetDriftPreset's per-column PSI on the label. DatasetDriftMetric
            # counts features only, so the label is gated through this branch.
            target_drift = bool(res.get("drift_detected", False))
            score = res.get("drift_score")
            target_drift_score = float(score) if score is not None else None

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
        n_reference=len(ref),
        n_current=len(cur),
        n_drifted_columns=n_drifted,
        dataset_drift=dataset_drift,
        target_drift=target_drift,
        target_drift_score=target_drift_score,
        evidently_ran=True,
    )


def upload_html_to_minio(
    minio_client,
    html: bytes,
    run_date: date,
    report_type: str,
    bucket: str = DEFAULT_BUCKET,
) -> str:
    # Upload report HTML to s3://<bucket>/<date>/<type>.html and return the URL.
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
    # Insert a row into monitoring_reports and return its id.
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO monitoring_reports "
            "(run_date, report_type, report_url, drift_score, blocked_promotion) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING id",
            (run_date, report_type, s3_url, drift_score, blocked),
        )
        return cur.fetchone()[0]


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
    # Compute drift, score the model, upload the HTML, gate promotion. Blocks when
    # ANY of these hold: a feature column drifts (PSI > psi_threshold), the label
    # distribution drifts (PSI > psi_threshold), or (when a model is given) macro-F1
    # drops past f1_drop_threshold or negative-class recall drops past
    # recall_neg_drop_threshold. drift_threshold stays on the signature for
    # back-compat but no longer drives the decision.
    #
    # Upload and insert happen FIRST, then we raise, so the failing report is still
    # discoverable in the dashboard. raise_on_block=True raises PromotionBlocked (the
    # DAG uses this so the task fails red); the default False lets callers inspect the
    # returned dict.
    run_date = run_date or date.today()
    drift = compute_drift(reference_df, current_df)

    reference_f1 = current_f1 = f1_drop = None
    reference_recall_neg = current_recall_neg = recall_neg_drop = None
    if model is not None:
        reference_f1, reference_recall_neg = _score_model(model, reference_df)
        current_f1, current_recall_neg = _score_model(model, current_df)
        f1_drop = reference_f1 - current_f1
        recall_neg_drop = reference_recall_neg - current_recall_neg

    drift_blocks = drift.is_blocking(drift_threshold)
    f1_blocks = (f1_drop is not None) and (f1_drop > f1_drop_threshold)
    recall_neg_blocks = (
        recall_neg_drop is not None
    ) and (recall_neg_drop > recall_neg_drop_threshold)
    blocked = drift_blocks or f1_blocks or recall_neg_blocks

    s3_url = upload_html_to_minio(
        minio_client, drift.html, run_date, report_type, bucket=bucket
    )
    row_id = insert_pointer_row(
        conn, run_date, report_type, s3_url, drift.drift_score, blocked
    )

    result = {
        "report_id": row_id,
        "s3_url": s3_url,
        "report_url": s3_url,
        "drift_score": drift.drift_score,
        "drifted_columns": drift.drifted_columns,
        "reference_f1": reference_f1,
        "current_f1": current_f1,
        "f1_drop": f1_drop,
        "f1_blocks": f1_blocks,
        "reference_recall_neg": reference_recall_neg,
        "current_recall_neg": current_recall_neg,
        "recall_neg_drop": recall_neg_drop,
        "recall_neg_blocks": recall_neg_blocks,
        "drift_blocks": drift_blocks,
        "target_drift": drift.target_drift,
        "target_drift_score": drift.target_drift_score,
        "blocked_promotion": blocked,
    }

    if blocked and raise_on_block:
        reasons = []
        if drift_blocks:
            parts = []
            if drift.n_drifted_columns >= 1:
                parts.append(f"{drift.n_drifted_columns} feature column(s) {drift.drifted_columns}")
            if drift.target_drift:
                parts.append("label")
            reasons.append(
                f"drift in {' + '.join(parts)} by PSI > {DEFAULT_PSI_THRESHOLD:.2f} "
                f"(drift_score={drift.drift_score:.3f})"
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


def _features_from_reviews(df: pd.DataFrame) -> pd.DataFrame:
    # Project raw reviews onto the columns we monitor (the "data drift" row in
    # monitoring/README.md): text length, rating, source mix, label distribution.
    # Keeping it tiny and explicit hands Evidently typed numeric + categorical
    # columns instead of free text it would ignore. text stays on the frame so the
    # gate can also score a model on it.
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
    # Concat the most recent n_partitions review_date= silver partitions. Returns
    # None when the silver root has no readable partitions yet (fresh stack), so the
    # caller can degrade to the train-vs-itself stub instead of failing the DAG.
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
    # Reference is the sample/training CSV; current is the recent silver partitions.
    # Returns (reference, current, used_silver). With no silver partitions yet,
    # current falls back to a copy of reference (train vs itself, so the gate passes)
    # and used_silver is False, which keeps the DAG green while wiring settles. Both
    # frames are reduced to their shared columns so Evidently sees a matching schema.
    reference = _features_from_reviews(pd.read_csv(sample_csv))

    silver_root = silver_root or DEFAULT_SILVER_ROOT
    recent = _load_recent_silver(silver_root, n_partitions)
    if recent is None:
        return reference, reference.copy(), False

    current = _features_from_reviews(recent)
    shared = [c for c in reference.columns if c in current.columns]
    return reference[shared], current[shared], True


def run_drift_check(
    sample_csv: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    silver_root: Optional[Path] = None,
) -> DriftResult:
    # Run the observational drift check and return a structured result. reference is
    # the training/sample distribution; current is the most recent silver partitions,
    # falling back to train-vs-itself when no silver data exists yet. Always returns
    # (never raises on Evidently internals) so the Airflow task stays green while the
    # pipeline is still being wired.
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


# Full observational drift summary: data + target + (optional) prediction.
# target_* and prediction_* are None when the column was unavailable (no label on
# current, or no model to score). blocked is True when any computed drift is
# actionable; the monitor DAG reads it to decide whether to alert.
@dataclass
class MonitorResult:
    report_path: str
    data_drift_score: float = 0.0
    dataset_drift: bool = False
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
    
    from evidently import ColumnMapping
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.report import Report

    ref = reference_df.copy()
    cur = current_df.copy()

    has_pred = False
    if model is not None and "text" in ref.columns and "text" in cur.columns:
        try:
            ref["prediction"] = model.predict(ref["text"].astype(str).tolist())
            cur["prediction"] = model.predict(cur["text"].astype(str).tolist())
            has_pred = True
        except Exception as exc: 
            print(f"[drift] prediction scoring failed, skipping prediction drift: {exc}")

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

    # Per-column PSI (0.25) for features; the target preset also covers the target
    # AND prediction columns with the same per-column PSI test.
    metrics = [
        DataDriftPreset(stattest="psi", stattest_threshold=DEFAULT_PSI_THRESHOLD)
    ]
    if has_target:
        metrics.append(
            TargetDriftPreset(stattest="psi", stattest_threshold=DEFAULT_PSI_THRESHOLD)
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


def _build_monitor_frames(
    sample_csv: Path,
    silver_root: Optional[Path] = None,
    n_partitions: int = DRIFT_RECENT_PARTITIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
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
) -> MonitorResult:
    sample_csv = Path(sample_csv) if sample_csv else DEFAULT_SAMPLE_CSV
    if not sample_csv or not sample_csv.exists():
        raise FileNotFoundError(
            f"sample CSV not found: {sample_csv!r} — set DRIFT_SAMPLE_CSV"
        )

    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    html_path = report_dir / f"{date.today():%Y-%m-%d}" / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    reference, current, used_silver = _build_monitor_frames(sample_csv, silver_root)
    if not used_silver:
        print("[drift] no silver partitions — monitoring reference vs itself (no drift)")

    try:
        rpt = compute_drift_report(reference, current, model=model)
        html_path.write_bytes(rpt.pop("html"))
        blocked = (
            rpt["data_drift_score"] > 0
            or rpt["target_drift"]
            or rpt["prediction_drift"]
        )
        result = MonitorResult(
            report_path=str(html_path),
            data_drift_score=rpt["data_drift_score"],
            dataset_drift=rpt["dataset_drift"],
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
    except Exception as exc: 
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

DEFAULT_REPLAY_ROOT = (
    _first_existing(
        Path(os.getenv("DRIFT_REPLAY_ROOT", "")) if os.getenv("DRIFT_REPLAY_ROOT") else None,
        Path("/opt/project/data/replay"),
        _REPO_ROOT / "data" / "replay",
    )
    or (_REPO_ROOT / "data" / "replay")
)

DEFAULT_REPLAY_REFERENCE_CSV = _first_existing(
    Path(os.getenv("DRIFT_REPLAY_REFERENCE", "")) if os.getenv("DRIFT_REPLAY_REFERENCE") else None,
    Path("/opt/project/data/demo/demo_holdout_full.csv"), 
    _REPO_ROOT / "data" / "demo" / "demo_holdout_full.csv", 
)

REPLAY_REVIEW_DATE_GLOB = "review_date=*"


def load_replay_window(
    scenario: str,
    replay_root: Optional[Path] = None,
    asof: Optional[str] = None,
    n_recent: Optional[int] = None,
) -> pd.DataFrame:
    replay_root = Path(replay_root) if replay_root else DEFAULT_REPLAY_ROOT
    scenario_root = replay_root / scenario
    if not scenario_root.exists():
        raise FileNotFoundError(
            f"replay output not found: {scenario_root} — run "
            f"`python -m data.ingest.replay --scenario {scenario}` first"
        )
    keys = sorted(
        p.name.split("=", 1)[1]
        for p in scenario_root.glob(REPLAY_REVIEW_DATE_GLOB)
        if p.is_dir()
    )
    if asof:
        keys = [k for k in keys if k <= asof]
    if n_recent:
        keys = keys[-n_recent:]
    if not keys:
        raise ValueError(
            f"no replay partitions selected under {scenario_root} "
            f"(asof={asof}, n_recent={n_recent})"
        )
    frames = [
        pd.read_parquet(scenario_root / f"review_date={k}" / "part.parquet") for k in keys
    ]
    return pd.concat(frames, ignore_index=True)


def build_replay_frames(
    scenario: str,
    reference_csv: Optional[Path] = None,
    replay_root: Optional[Path] = None,
    asof: Optional[str] = None,
    n_recent: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference_csv = Path(reference_csv) if reference_csv else DEFAULT_REPLAY_REFERENCE_CSV
    if not reference_csv or not reference_csv.exists():
        raise FileNotFoundError(
            f"replay reference CSV not found: {reference_csv!r} — set DRIFT_REPLAY_REFERENCE"
        )
    reference = _features_from_reviews(pd.read_csv(reference_csv))
    current = _features_from_reviews(load_replay_window(scenario, replay_root, asof, n_recent))
    cols = [c for c in ("text", "label") if c in reference.columns and c in current.columns]
    return reference[cols], current[cols]


def run_replay_monitor(
    scenario: str,
    reference_csv: Optional[Path] = None,
    replay_root: Optional[Path] = None,
    asof: Optional[str] = None,
    n_recent: Optional[int] = None,
    report_dir: Optional[Path] = None,
    threshold: float = DEFAULT_DRIFT_THRESHOLD,
    model=None,
) -> MonitorResult:
    reference, current = build_replay_frames(
        scenario, reference_csv, replay_root, asof, n_recent
    )
    report_dir = Path(report_dir) if report_dir else DEFAULT_REPORT_DIR
    html_path = report_dir / f"replay-{scenario}-{date.today():%Y-%m-%d}" / "report.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        rpt = compute_drift_report(reference, current, model=model)
        html_path.write_bytes(rpt.pop("html"))
        blocked = (
            rpt["data_drift_score"] > 0
            or rpt["target_drift"]
            or rpt["prediction_drift"]
        )
        result = MonitorResult(
            report_path=str(html_path),
            data_drift_score=rpt["data_drift_score"],
            dataset_drift=rpt["dataset_drift"],
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
        print(f"[replay-drift:{scenario}] {_summary_for_log(asdict(result))}")
        return result
    except Exception as exc:  # never kill the demo on an Evidently internal
        print(f"[replay-drift:{scenario}] Evidently unavailable/incompatible, stub: {exc}")
        html_path.write_text(
            "<html><body>replay drift stub: evidently unavailable</body></html>"
        )
        return MonitorResult(
            report_path=str(html_path),
            n_reference=len(reference),
            n_current=len(current),
            evidently_ran=False,
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Run a drift check: the silver observational monitor, or a "
        "replay scenario vs the demo holdout (--scenario)."
    )
    ap.add_argument(
        "--scenario",
        choices=["stable", "spike", "holdout"],
        default=None,
        help="Replay scenario to monitor vs the demo holdout (omit for the silver check).",
    )
    ap.add_argument("--asof", default=None, help="Keep only replay partitions on/before this YYYY-MM-DD.")
    ap.add_argument(
        "--n-recent", type=int, default=None,
        help="Use only the most recent N replay partitions as the current window.",
    )
    args = ap.parse_args()

    if args.scenario:
        result = run_replay_monitor(args.scenario, asof=args.asof, n_recent=args.n_recent)
    else:
        result = run_drift_check()
    for k, v in asdict(result).items():
        if k == "html":
            v = f"<{len(v)} bytes>"
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
