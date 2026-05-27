"""Streamlit dashboard — Phase 1 thin slice.

Three tiles (per WORKFLOW.md):
    1. Total reviews
    2. % positive
    3. Live prediction probe (text box that calls the API /predict)

Data source for tiles 1-2 is the sample CSV. In Phase 2 this gets swapped
for a Postgres read so the dashboard reflects what's actually been ingested.

Owner: Amelia.

Run locally:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
SAMPLE_CSV = Path(os.getenv(
    "DASHBOARD_DATA_PATH",
    str(Path(__file__).resolve().parents[1] / "data" / "sample" / "reviews_sample.csv"),
))


@st.cache_data
def load_reviews() -> pd.DataFrame:
    # TODO (member, Phase 2): replace CSV read with a Postgres query against
    # the `reviews` table populated by the ingestion DAG. Connection string
    # comes from POSTGRES_HOST env var (see docker-compose.yml).
    if not SAMPLE_CSV.exists():
        return pd.DataFrame(columns=["text", "label", "rating", "source"])
    return pd.read_csv(SAMPLE_CSV)


def main() -> None:
    st.set_page_config(page_title="Brand Sentiment — MVP", layout="wide")
    st.title("Brand Sentiment — MVP dashboard")

    df = load_reviews()

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

    # TODO (member, Phase 2): add the deeper views from WORKFLOW.md:
    #   - Sentiment timeline (line chart of % positive over time)
    #   - Word cloud of negative reviews (`wordcloud` is already in requirements.txt)
    #   - Model A/B comparison (pull two runs from MLflow, show metric deltas)
    #   - Latest Evidently drift report (embed the MinIO-hosted HTML)
    st.divider()
    st.caption("Phase 2: timeline, word cloud, A/B compare, embedded drift report.")


if __name__ == "__main__":
    main()
