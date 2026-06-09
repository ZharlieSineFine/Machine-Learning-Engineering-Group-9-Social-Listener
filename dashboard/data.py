"""Pure data-loading and transformation helpers for the Streamlit app.

Keeping these out of `app.py` makes them unit-testable (Streamlit's runtime
context isn't trivial to mock). All functions either:
  - take their dependencies as args (DSN, MLflow client), or
  - degrade gracefully when the backing service isn't reachable.

Owner: Amelia.
"""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"

# Stopwords for the negative-reviews word cloud. Intentionally tiny — the
# team can swap in NLTK or spaCy's full English stopword list in Phase 3.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at",
    "is", "it", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "this", "that", "these", "those", "i", "me", "we",
    "us", "you", "your", "they", "them", "their", "he", "she", "his", "her",
    "for", "with", "as", "by", "from", "not", "no", "so", "too", "very",
    "just", "only", "also", "really", "out", "than", "then", "there", "here",
    "what", "which", "when", "where", "who", "how", "why", "all", "any",
    "some", "more", "most", "much", "even", "still", "would", "could",
    "should", "will", "can", "my", "our", "its",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")  # 3+ letter words


# ---------- review data ----------

def load_reviews(dsn: Optional[str] = None, csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Postgres first (the "live" source), CSV fallback (offline / demo)."""
    if dsn:
        try:
            import psycopg2  # noqa: F401  (import-side dep advertised in requirements)
            from sqlalchemy import create_engine

            engine = create_engine(dsn)
            with engine.connect() as conn:
                return pd.read_sql(
                    "SELECT text, label, rating, source, ingested_at "
                    "FROM reviews ORDER BY ingested_at",
                    conn,
                )
        except Exception as exc:
            print(f"[dashboard.data] Postgres unavailable ({exc}); falling back to CSV")

    csv_path = csv_path or DEFAULT_SAMPLE_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=["text", "label", "rating", "source"])
    df = pd.read_csv(csv_path)
    return df


# ---------- timeline ----------

def sentiment_timeline(
    df: pd.DataFrame, freq: str = "D", time_col: str = "ingested_at",
) -> pd.DataFrame:
    """Aggregate % positive (and counts) per period. Returns long-form DataFrame.

    Falls back to a synthetic per-row index when `time_col` is absent (CSV
    sample has no timestamps).
    """
    df = df.copy()
    if time_col not in df.columns:
        # No timestamps — fabricate evenly-spaced days so the chart isn't empty.
        df[time_col] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, "label"])

    period = df[time_col].dt.to_period(freq).dt.to_timestamp()
    grp = df.groupby(period)
    out = pd.DataFrame({
        "period": grp.size().index,
        "n_reviews": grp.size().values,
        "pct_positive": grp.apply(lambda g: (g["label"] == "positive").mean() * 100).values,
        "pct_negative": grp.apply(lambda g: (g["label"] == "negative").mean() * 100).values,
    }).reset_index(drop=True)
    return out


# ---------- word cloud ----------

def negative_word_counts(df: pd.DataFrame, top_n: int = 50) -> Counter:
    """Word counts (post-stopword) across rows labelled 'negative'."""
    if "label" not in df.columns or "text" not in df.columns:
        return Counter()
    neg = df.loc[df["label"] == "negative", "text"].astype(str)
    counter: Counter = Counter()
    for review in neg:
        for token in _TOKEN_RE.findall(review.lower()):
            if token not in _STOPWORDS:
                counter[token] += 1
    return Counter(dict(counter.most_common(top_n)))


# ---------- MLflow A/B ----------

def list_mlflow_runs(
    experiment_names: Optional[list[str]] = None,
    tracking_uri: Optional[str] = None,
    max_runs: int = 20,
) -> pd.DataFrame:
    """Return one row per recent MLflow run (across given experiments).

    Returns an empty DataFrame on failure — the UI shows "no runs yet"
    instead of erroring.
    """
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return pd.DataFrame()

    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_names = experiment_names or [os.getenv("MLFLOW_EXPERIMENT", "sentiment-baseline")]
        experiments = [e for e in (client.get_experiment_by_name(n) for n in experiment_names) if e]
        if not experiments:
            return pd.DataFrame()
        runs = client.search_runs(
            experiment_ids=[e.experiment_id for e in experiments],
            order_by=["start_time DESC"],
            max_results=max_runs,
        )
        rows = []
        for r in runs:
            rows.append({
                "run_id": r.info.run_id,
                "experiment": next(
                    (e.name for e in experiments if e.experiment_id == r.info.experiment_id), ""
                ),
                "start_time": pd.to_datetime(r.info.start_time, unit="ms"),
                "model_type": r.data.params.get("model_type"),
                "f1_macro": r.data.metrics.get("f1_macro"),
                "f1_weighted": r.data.metrics.get("f1_weighted"),
                "n_train": r.data.params.get("n_train"),
            })
        return pd.DataFrame(rows)
    except Exception as exc:
        print(f"[dashboard.data] MLflow listing failed ({exc})")
        return pd.DataFrame()


# ---------- drift report ----------

def latest_drift_report(dsn: Optional[str]) -> Optional[dict]:
    """Read the most recent row from `monitoring_reports`. None if table empty."""
    if not dsn:
        return None
    try:
        import psycopg2

        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT report_url, drift_score, blocked_promotion, run_date "
                    "FROM monitoring_reports ORDER BY created_at DESC LIMIT 1"
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if not row:
            return None
        return {
            "report_url": row[0],
            "drift_score": row[1],
            "blocked_promotion": row[2],
            "run_date": row[3],
        }
    except Exception as exc:
        print(f"[dashboard.data] drift report lookup failed ({exc})")
        return None


def fetch_drift_html(report_url: str, minio_client: Any) -> bytes:
    """Download the HTML body from MinIO. `report_url` is an s3:// URL."""
    if not report_url.startswith("s3://"):
        raise ValueError(f"expected s3:// URL, got: {report_url}")
    _, _, rest = report_url.partition("s3://")
    bucket, _, key = rest.partition("/")
    obj = minio_client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()
