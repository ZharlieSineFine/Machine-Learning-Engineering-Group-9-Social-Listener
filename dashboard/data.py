#Pure data-loading and transformation helpers for the Streamlit app.

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_CSV = ROOT / "data" / "demo" / "demo_jun2026_stable.csv"
DEFAULT_GOLD_ROOT = ROOT / "data" / "gold"

# Stopwords for the negative-reviews word cloud. Intentionally tiny — the
# team can swap in NLTK or spaCy's full English stopword list in Phase 3.
_STOPWORDS = {
    # Articles / prepositions / conjunctions
    "the", "a", "an", "and", "or", "but", "if", "to", "of", "in", "on", "at",
    "for", "with", "as", "by", "from", "into", "onto", "upon", "about",
    "above", "below", "between", "around", "through", "during", "before",
    "after", "over", "under", "again", "further", "then", "once", "off",
    "out", "than", "so", "yet", "both", "either", "neither", "nor",

    # Pronouns
    "i", "me", "my", "we", "us", "our", "you", "your", "they", "them",
    "their", "he", "she", "his", "her", "it", "its", "this", "that",
    "these", "those", "who", "which", "what", "where", "when", "why", "how",

    # Auxiliaries / modals
    "is", "are", "was", "were", "be", "been", "being", "am",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "can", "may", "might", "shall",
    "not", "no", "nor",

    # Contractions
    "wasn't", "weren't", "don't", "doesn't", "didn't", "isn't", "aren't",
    "it's", "i'm", "you're", "they're", "we're", "he's", "she's", "that's",
    "i've", "you've", "they've", "we've", "won't", "can't", "couldn't",
    "wouldn't", "shouldn't", "hadn't", "hasn't", "haven't", "i'd", "you'd",
    "they'd", "we'd", "he'd", "she'd", "i'll", "you'll", "they'll", "we'll",

    # Common adverbs / fillers
    "very", "just", "only", "also", "really", "too", "here", "there",
    "still", "even", "much", "more", "most", "some", "any", "all",
    "many", "few", "such", "own", "same", "other", "another", "each",
    "every", "both", "more", "most", "quite", "rather", "always", "never",
    "ever", "already", "now", "back", "away", "else", "well",

    # Generic action verbs
    "said", "say", "told", "tell", "came", "come", "going", "went", "go",
    "get", "got", "gotten", "getting", "give", "gave", "given", "giving",
    "take", "took", "taken", "taking", "put", "putting", "ask", "asked",
    "asking", "know", "knew", "known", "want", "wanted", "wanting",
    "left", "leave", "leaving", "make", "made", "making", "see", "saw",
    "seen", "seeing", "try", "tried", "trying", "eat", "ate", "eaten",
    "eating", "look", "looked", "looking", "think", "thought", "thinking",
    "need", "needed", "use", "used", "using", "seem", "seemed", "call",
    "called", "calling", "let", "keep", "kept", "bring", "brought",
    "drive", "drove", "driven", "sit", "sat", "start", "started",

    # Connector / transitional words
    "because", "while", "although", "though", "unless", "until", "since",
    "however", "therefore", "instead", "otherwise", "meanwhile",

    # Additional:
    "like", "one", "two","place", "good", "bad", "people", "first", "done", 
    "location", "restaurant", "drinks", "waited", "customers", "times", 
    "way", "waiting", "something", "store", "nothing", "orders", "tasted",
    "years", "day", "great",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']{2,}")  # 3+ letter words


# ---------- review data ----------

def load_reviews(
    dsn: Optional[str] = None,
    gold_root: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    days: int = 14,
) -> pd.DataFrame:
    """Postgres first, Gold parquet second, CSV fallback last."""
    if dsn:
        try:
            from sqlalchemy import create_engine, text
            engine = create_engine(dsn)
            with engine.connect() as conn:
                return pd.read_sql(
                    text(
                        "SELECT text, label, source, ingested_at AS review_date "
                        "FROM reviews WHERE ingested_at >= NOW() - (INTERVAL '1 day' * :days) "
                        "ORDER BY ingested_at"
                    ),
                    conn,
                    params={"days": days},
                )
        except Exception as exc:
            print(f"[dashboard.data] Postgres unavailable ({exc}); trying Gold parquet")

    resolved_gold = gold_root or DEFAULT_GOLD_ROOT
    df = _load_gold_parquet(resolved_gold, days=days)
    if df is not None and not df.empty:
        return df

    csv_path = csv_path or DEFAULT_SAMPLE_CSV
    if not csv_path.exists():
        return pd.DataFrame(columns=["text", "label", "review_date"])
    df = pd.read_csv(csv_path)
    if "review_date" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "review_date"})
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
        cutoff = df["review_date"].max() - pd.Timedelta(days=days)
        df = df[df["review_date"] >= cutoff]
    return df

def _load_gold_parquet(gold_root: Path, days: int) -> Optional[pd.DataFrame]:
    feature_root = gold_root / "feature_store"
    label_root = gold_root / "label_store"

    if not feature_root.exists() or not label_root.exists():
        return None

    all_dates = sorted(
        d.name.replace("review_date=", "")
        for d in feature_root.iterdir()
        if d.is_dir() and d.name.startswith("review_date=")
    )
    if not all_dates:
        return None

    latest = date.fromisoformat(all_dates[-1])
    cutoff = (latest - timedelta(days=days - 1)).isoformat()
    target_dates = [d for d in all_dates if d >= cutoff]

    feat_frames, lab_frames = [], []
    for d in target_dates:
        fp = feature_root / f"review_date={d}" / "part.parquet"
        lp = label_root / f"review_date={d}" / "part.parquet"
        if fp.exists() and lp.exists():
            feat_frames.append(pd.read_parquet(fp, columns=["review_id", "review_date", "text"]))
            lab_frames.append(pd.read_parquet(lp, columns=["review_id", "label"]))

    if not feat_frames:
        return None

    feat = pd.concat(feat_frames, ignore_index=True)
    lab = pd.concat(lab_frames, ignore_index=True)
    df = feat.merge(lab[["review_id", "label"]], on="review_id", how="inner")
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    return df

# ---------- latest batch ----------

def latest_batch(df: pd.DataFrame, time_col: str = "review_date") -> pd.DataFrame:
    """Rows from the most recent ingested day — the "last batch" the KPI tiles and the
    spike alert summarise (vs. the multi-day trend the timeline shows). Falls back to
    the whole frame when there's no usable timestamp."""
    if df.empty or time_col not in df.columns:
        return df
    ts = pd.to_datetime(df[time_col], errors="coerce")
    if ts.notna().sum() == 0:
        return df
    last_day = ts.max().normalize()
    return df[ts.dt.normalize() == last_day]


# ---------- timeline ----------

def sentiment_timeline(
    df: pd.DataFrame, freq: str = "D", time_col: str = "ingested_at",
) -> pd.DataFrame:
    df = df.copy()
    if time_col not in df.columns:
        df[time_col] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, "label"])

    period = df[time_col].dt.to_period(freq).dt.to_timestamp()
    df = df.copy()
    df["_period"] = period

    grp = df.groupby("_period")
    counts = grp.size().rename("n_reviews")

    def pct(lbl):
        return grp["label"].apply(lambda s: (s == lbl).mean() * 100)

    out = pd.DataFrame({
        "period":      counts.index,
        "n_reviews":   counts.values,
        "pct_positive": pct("positive").values,
        "pct_negative": pct("negative").values,
        "pct_neutral":  pct("neutral").values,
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
                "recall_neg": r.data.metrics.get("recall_neg"),
                "precision_neg": r.data.metrics.get("precision_neg"),
                "f1_neg": r.data.metrics.get("f1_neg"),
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