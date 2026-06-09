"""Streamlit dashboard — Phase 2 with depth.

Sections (top to bottom):
    1. Three tiles (total reviews, % positive, live prediction probe)
    2. Sentiment timeline (line chart)
    3. Negative-review word cloud
    4. MLflow A/B compare table (recent runs)
    5. Latest Evidently drift report (embedded HTML)

Owner: Amelia.

Run locally:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# Allow `from dashboard.data import ...` from a fresh repo clone where the
# `dashboard` dir isn't yet on the path (Streamlit Cloud, ad-hoc runs).
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.data import (  # noqa: E402
    fetch_drift_html,
    latest_drift_report,
    list_mlflow_runs,
    load_reviews,
    negative_word_counts,
    sentiment_timeline,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")


def _default_dsn() -> Optional[str]:
    user = os.getenv("POSTGRES_USER")
    pw = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB")
    if not (user and pw and host and db):
        return None
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _minio_client():
    endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    key = os.getenv("AWS_ACCESS_KEY_ID")
    secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not (endpoint and key and secret):
        return None
    import boto3
    from botocore.client import Config
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


# ---------- sections ----------

def render_tiles(df: pd.DataFrame) -> None:
    total = len(df)
    pct_pos = (df["label"].eq("positive").mean() * 100) if total else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total reviews", f"{total:,}")
    c2.metric("% positive", f"{pct_pos:.1f}%")
    with c3:
        st.markdown("**Live prediction probe**")
        text = st.text_area("Review text", value="The food was incredible", height=80)
        if st.button("Predict"):
            try:
                r = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)
                r.raise_for_status()
                st.success(f"Label: **{r.json()['label']}**")
            except Exception as exc:
                st.error(f"API call failed: {exc}")


def render_timeline(df: pd.DataFrame) -> None:
    st.subheader("Sentiment over time")
    freq = st.radio("Bucket", ["D", "W"], format_func={"D": "Daily", "W": "Weekly"}.get,
                    horizontal=True, key="timeline_freq")
    timeline = sentiment_timeline(df, freq=freq)
    if timeline.empty:
        st.info("No timestamped data to plot.")
        return
    st.line_chart(
        timeline.set_index("period")[["pct_positive", "pct_negative"]],
        height=240,
    )
    with st.expander("Counts per bucket"):
        st.dataframe(timeline)


def render_word_cloud(df: pd.DataFrame) -> None:
    st.subheader("Top words in negative reviews")
    counts = negative_word_counts(df, top_n=80)
    if not counts:
        st.info("No negative reviews to visualise.")
        return
    try:
        from wordcloud import WordCloud
        wc = WordCloud(width=800, height=300, background_color="white").generate_from_frequencies(counts)
        st.image(wc.to_array())
    except Exception as exc:
        # wordcloud may be missing in some local envs — degrade to a bar chart.
        st.warning(f"WordCloud unavailable ({exc}); showing top 20 as a bar chart.")
        top = pd.DataFrame(counts.most_common(20), columns=["word", "count"])
        st.bar_chart(top.set_index("word"))


def render_mlflow_ab() -> None:
    st.subheader("Model A/B — recent MLflow runs")
    runs = list_mlflow_runs(
        experiment_names=[os.getenv("MLFLOW_EXPERIMENT", "sentiment-baseline")],
    )
    if runs.empty:
        st.info("No MLflow runs found (set MLFLOW_TRACKING_URI and train a model).")
        return
    st.dataframe(
        runs[["start_time", "experiment", "model_type", "f1_macro", "f1_weighted", "n_train"]],
        use_container_width=True,
    )


def render_drift_report(dsn: Optional[str]) -> None:
    st.subheader("Latest drift report")
    rep = latest_drift_report(dsn)
    if not rep:
        st.info("No drift reports yet. Trigger the `evaluate_and_monitor` DAG.")
        return
    cols = st.columns(3)
    cols[0].metric("Drift score", f"{rep['drift_score']:.2f}")
    cols[1].metric("Blocked?", "Yes" if rep["blocked_promotion"] else "No")
    cols[2].metric("Run date", str(rep["run_date"]))
    st.caption(f"Report: `{rep['report_url']}`")

    minio = _minio_client()
    if minio is None:
        st.info("Set MLFLOW_S3_ENDPOINT_URL + AWS creds to embed the HTML report.")
        return
    try:
        html = fetch_drift_html(rep["report_url"], minio)
        st.components.v1.html(html.decode("utf-8", errors="replace"), height=600, scrolling=True)
    except Exception as exc:
        st.error(f"Could not fetch report: {exc}")


# ---------- entry point ----------

def main() -> None:
    st.set_page_config(page_title="Brand Sentiment", layout="wide")
    st.title("Brand Sentiment dashboard")

    dsn = _default_dsn()
    df = load_reviews(dsn=dsn)

    render_tiles(df)
    st.divider()
    render_timeline(df)
    st.divider()
    render_word_cloud(df)
    st.divider()
    render_mlflow_ab()
    st.divider()
    render_drift_report(dsn)


if __name__ == "__main__":
    main()
