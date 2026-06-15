"""Streamlit dashboard — Phase 2 with depth.

Sections (top to bottom):
    1. 4 KPI tiles  (total posts, % positive, % negative, % neutral)
    2. Sentiment timeline (line chart)
    3. Alerts panel  (recent negative posts, left-bordered cards)
    4. Topic breakdown  (placeholder — Phase 2 stretch goal)
    5. Word cloud  (top words in negative reviews)
    6. Live prediction probe (for testing if dashboard works)

Owner: Amelia.

Run locally:
    streamlit run dashboard/app.py
"""
from __future__ import annotations
 
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD_ROOT = Path(os.getenv("GOLD_ROOT", str(ROOT / "data" / "gold")))

DEFAULT_DEMO_CSV = Path(os.getenv("DEMO_CSV", str(ROOT / "data" / "sample" / "reviews_sample.csv")))

from data import (
    load_reviews,
    negative_word_counts,
    sentiment_timeline,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
 
API_URL = os.getenv("API_URL", "http://localhost:8000")
NEGATIVE_THRESHOLD = 20   # % — spike warning triggers above this
 
# How many days to load per view mode.
# Daily → 14 days gives a clean 2-week spread of individual points.
# Weekly → 60 days gives ~8 weekly points which makes a meaningful trend line.
DAYS_BY_FREQ = {"D": 14, "W": 60}
 
# Palette (matches prototype)
TEAL   = "#1D9E75"
RED    = "#E24B4A"
AMBER  = "#EF9F27"
BROWN  = "#5C3A21"
CARD_BG    = "#1E2128"
PAGE_BG    = "#16181D"
BORDER     = "#2E3039"
TEXT_PRI   = "#E8E6DF"
TEXT_SEC   = "#888780"
 
 
# ---------------------------------------------------------------------------
# Page config + global CSS
# ---------------------------------------------------------------------------
 
st.set_page_config(
    page_title="BrewLeaf · Brand Sentiment",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)
 
st.markdown(f"""
<style>
/* ── Page background ── */
.stApp {{ background-color: {PAGE_BG}; }}
.block-container {{ padding: 1.5rem 2rem 2rem; max-width: 100%; }}
 
/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
 
/* ── Metric tiles ── */
[data-testid="stMetric"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 1rem 1.2rem;
}}
[data-testid="stMetricLabel"] p {{
    color: {TEXT_SEC} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRI} !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
}}
[data-testid="stMetricDelta"] {{
    font-size: 11px !important;
}}
 
/* ── Dividers ── */
hr {{ border-color: {BORDER} !important; margin: 0.5rem 0; }}
 
/* ── Radio buttons ── */
.stRadio label {{ color: {TEXT_SEC} !important; font-size: 12px; }}
.stRadio [data-testid="stMarkdownContainer"] p {{
    color: {TEXT_SEC} !important; font-size: 12px;
}}
 
/* ── Text area / button ── */
.stTextArea textarea {{
    background: {CARD_BG} !important;
    color: {TEXT_PRI} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px;
}}
.stButton > button {{
    background: {TEAL} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 0.4rem 1rem !important;
}}
.stButton > button:hover {{ opacity: 0.85; }}
 
/* ── Expander ── */
.streamlit-expanderHeader {{
    background: {CARD_BG} !important;
    color: {TEXT_SEC} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}}
 
/* ── General text ── */
p, li, span {{ color: {TEXT_PRI}; }}
h1, h2, h3 {{ color: {TEXT_PRI} !important; }}
</style>
""", unsafe_allow_html=True)
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
def _pct(df: pd.DataFrame, label: str) -> float:
    if df.empty:
        return 0.0
    return round(df["label"].eq(label).mean() * 100, 1)
 
 
def _last_batch_time() -> str:
    now = datetime.now(timezone.utc)
    hour = (now.hour // 6) * 6
    return now.replace(hour=hour, minute=0, second=0, microsecond=0).strftime(
        "%a %d %b %Y · %H:%M UTC"
    )
 
 
def _card(content_html: str, padding: str = "1rem 1.2rem") -> None:
    st.markdown(
        f"""<div style="background:{CARD_BG};border:1px solid {BORDER};
            border-radius:10px;padding:{padding};">{content_html}</div>""",
        unsafe_allow_html=True,
    )
 
 
def _section_title(icon: str, title: str) -> str:
    return (
        f'<p style="margin:0 0 0.8rem;font-size:12px;font-weight:600;'
        f'color:{TEXT_SEC};text-transform:uppercase;letter-spacing:0.05em;">'
        f'{icon}&nbsp;&nbsp;{title}</p>'
    )

# ---------------------------------------------------------------------------
# Load data (cached so Streamlit doesn't reload on every interaction)
# ---------------------------------------------------------------------------
 
@st.cache_data(ttl=300)   # refresh every 5 minutes
def _load(days: int, csv_path: str = "") -> pd.DataFrame:
    dsn = None
    user = os.getenv("POSTGRES_USER")
    pw   = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB")
    if all([user, pw, host, db]):
        dsn = f"postgresql://{user}:{pw}@{host}:{port}/{db}"
        
    return load_reviews(
        dsn=dsn,
        gold_root=DEFAULT_GOLD_ROOT if not csv_path else None,
        csv_path=Path(csv_path) if csv_path else None,
        days=days,
    )
 
# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
 
def render_header(pct_neg: float) -> None:
    left, right = st.columns([3, 1])
 
    with left:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;padding:4px 0 2px;">
                <div style="width:36px;height:36px;border-radius:50%;
                            background:{BROWN};display:flex;align-items:center;
                            justify-content:center;font-size:18px;">🍃</div>
                <div>
                    <p style="margin:0;font-size:18px;font-weight:700;color:{TEXT_PRI};">
                        BrewLeaf Social Listener
                    </p>
                    <p style="margin:0;font-size:12px;color:{TEXT_SEC};">
                        Morning digest &nbsp;·&nbsp; {_last_batch_time()}
                        &nbsp;·&nbsp; Last batch 06:00
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
 
    with right:
        if pct_neg >= NEGATIVE_THRESHOLD:
            st.markdown(
                f"""<div style="margin-top:8px;padding:7px 14px;background:#2E1A1A;
                    border-radius:8px;border:1px solid #6B2222;text-align:center;">
                    <span style="color:#F09595;font-size:12px;font-weight:600;">
                        ⚠ &nbsp;{pct_neg:.0f}% negative — above threshold
                    </span></div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div style="margin-top:8px;padding:7px 14px;background:#162414;
                    border-radius:8px;border:1px solid #2A5C2A;text-align:center;">
                    <span style="color:#6FCF6F;font-size:12px;font-weight:600;">
                        ✓ &nbsp;Sentiment normal
                    </span></div>""",
                unsafe_allow_html=True,
            )
 
 
def render_kpi_tiles(df: pd.DataFrame) -> None:
    total    = len(df)
    pct_pos  = _pct(df, "positive")
    pct_neg  = _pct(df, "negative")
    pct_neu  = _pct(df, "neutral")
 
    c1, c2, c3, c4 = st.columns(4)
 
    c1.metric("Posts analysed (batch)", f"{total:,}")
 
    # Colour the negative tile red when above threshold
    neg_label = f"{pct_neg:.1f}%"
    c2.metric("Negative sentiment", neg_label,
              delta=f"threshold {NEGATIVE_THRESHOLD}%",
              delta_color="inverse" if pct_neg >= NEGATIVE_THRESHOLD else "off")
    if pct_neg >= NEGATIVE_THRESHOLD:
        st.markdown(
            f"""<style>
            [data-testid="stMetric"]:nth-child(2) [data-testid="stMetricValue"] {{
                color: {RED} !important;
            }}</style>""",
            unsafe_allow_html=True,
        )
 
    c3.metric("Positive sentiment", f"{pct_pos:.1f}%")
    c4.metric("Neutral sentiment", f"{pct_neu:.1f}%")
 
 
def render_timeline(df: pd.DataFrame, csv_path: str = "") -> None:
    freq = st.radio(
        "Group by",
        ["D", "W"],
        format_func={"D": "Daily", "W": "Weekly"}.get,
        horizontal=True,
        key="timeline_freq",
        label_visibility="collapsed",
    )

    days = DAYS_BY_FREQ[freq]
    label = "14 days" if freq == "D" else "8 weeks"
    inner = _section_title("📈", f"Sentiment trend — {label}")

    df = _load(days, csv_path=csv_path)
    timeline = sentiment_timeline(df, freq=freq, time_col="review_date")
 
    fig = go.Figure()
 
    if not timeline.empty:
        fig.add_trace(go.Scatter(
            x=timeline["period"], y=timeline["pct_positive"],
            name="Positive", mode="lines+markers",
            line=dict(color=TEAL, width=2),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=timeline["period"], y=timeline["pct_neutral"],
            name="Neutral", mode="lines+markers",
            line=dict(color=AMBER, width=2, dash="dot"),
            marker=dict(size=4),
        ))
        fig.add_trace(go.Scatter(
            x=timeline["period"], y=timeline["pct_negative"],
            name="Negative", mode="lines+markers",
            line=dict(color=RED, width=2, dash="dash"),
            marker=dict(size=4, symbol="diamond"),
        ))
        # Threshold reference line
        fig.add_hline(
            y=NEGATIVE_THRESHOLD,
            line_dash="dash", line_color=RED,
            line_width=1, opacity=0.35,
        )
 
    fig.update_layout(
        height=260,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SEC, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=TEXT_SEC, size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        yaxis=dict(
            range=[0, 100], ticksuffix="%",
            gridcolor=BORDER, gridwidth=1,
            color=TEXT_SEC,
        ),
        xaxis=dict(showgrid=False, color=TEXT_SEC),
        hovermode="x unified",
    )
 
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:1rem 1.2rem;">'
        f'{inner}',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)
 
 
def render_alerts(df: pd.DataFrame, n: int = 4) -> None:
    inner = _section_title("🔔", "Alerts (last batch)")
 
    neg = df[df["label"] == "negative"].copy()
 
    if neg.empty:
        body = f'<p style="color:{TEXT_SEC};font-size:13px;">No negative posts in this batch.</p>'
        _card(inner + body)
        return
 
    # Sort by date descending if available
    if "review_date" in neg.columns:
        neg["review_date"] = pd.to_datetime(neg["review_date"], errors="coerce")
        neg = neg.sort_values("review_date", ascending=False)
    neg = neg.head(n)
 
    cards_html = ""
    for _, row in neg.iterrows():
        text = str(row.get("text", ""))
        snippet = text[:200] + ("…" if len(text) > 200 else "")
        date_str = ""
        if "review_date" in row and pd.notna(row["review_date"]):
            date_str = pd.to_datetime(row["review_date"]).strftime("%d %b %Y")
 
        # source tag — use source column if present, otherwise derive from review_id
        source = str(row.get("source", "")).title()
        if not source or source == "Nan":
            rid = str(row.get("review_id", ""))
            source = "TripAdvisor" if len(rid) == 64 else "Yelp"
 
        cards_html += f"""
        <div style="border-left:3px solid {RED};border-radius:0 8px 8px 0;
                    background:#23171A;padding:10px 14px;margin-bottom:8px;">
            <p style="margin:0;font-size:13px;color:{TEXT_PRI};line-height:1.5;">
                "{snippet}"
            </p>
            <p style="margin:4px 0 0;font-size:11px;color:{TEXT_SEC};">
                <span style="background:#3D1A1A;color:#F09595;padding:1px 8px;
                             border-radius:99px;font-size:10px;margin-right:6px;">
                    negative
                </span>
                {source}{' · ' + date_str if date_str else ''}
            </p>
        </div>"""
 
    _card(inner + cards_html)
 
 
def render_word_cloud(df: pd.DataFrame) -> None:
    inner = _section_title("A", "Top words in negative posts")
    counts = negative_word_counts(df, top_n=15)

    if not counts:
        _card(inner + f'<p style="color:{TEXT_SEC};font-size:13px;">No negative reviews.</p>')
        return

    try:
        from wordcloud import WordCloud
        import io

        wc = WordCloud(
            width=900, height=280,
            background_color="#1E2128",
            color_func=lambda *args, **kwargs: AMBER,
            max_words=15,
            collocations=False,
        ).generate_from_frequencies(counts)
 
        buf = io.BytesIO()
        wc.to_image().save(buf, format="PNG")
        buf.seek(0)
 
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
            f'border-radius:10px;padding:1rem 1.2rem;">{inner}',
            unsafe_allow_html=True,
        )
        st.image(buf, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
 
    except Exception:
        # Graceful fallback: plotly horizontal bar chart
        top = pd.DataFrame(counts.most_common(20), columns=["word", "count"])
        fig = go.Figure(go.Bar(
            x=top["count"], y=top["word"],
            orientation="h",
            marker_color=RED,
        ))
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=4, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SEC, size=11),
            xaxis=dict(showgrid=False, color=TEXT_SEC),
            yaxis=dict(autorange="reversed", color=TEXT_SEC),
        )
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
            f'border-radius:10px;padding:1rem 1.2rem;">{inner}',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
 
 
def render_topic_breakdown() -> None:
    inner = _section_title("🏷", "Topic breakdown (24h)")
    placeholder = f"""
    <div style="border:1px dashed {BORDER};border-radius:8px;
                padding:24px;text-align:center;margin-top:4px;">
        <p style="margin:0;font-size:14px;color:{TEXT_SEC};font-weight:600;">
            🚧 &nbsp;Stretch goal — Phase 2
        </p>
        <p style="margin:8px 0 0;font-size:12px;color:{TEXT_SEC};line-height:1.6;">
            Topic clustering (K-Means / BERTopic) will populate this panel
            once the models layer is complete. Each bar will show what % of
            negative posts mention a topic such as service, wait time, or
            drink quality.
        </p>
    </div>"""
    _card(inner + placeholder)


def render_mlflow_ab() -> None:
    st.subheader("Model A/B — recent MLflow runs")
    runs = list_mlflow_runs(
        experiment_names=[os.getenv("MLFLOW_EXPERIMENT", "sentiment-baseline")],
    )
    if runs.empty:
        st.info("No MLflow runs found (set MLFLOW_TRACKING_URI and train a model).")
        return
    st.dataframe(
        runs[[
            "start_time", "experiment", "model_type",
            "recall_neg", "f1_neg", "precision_neg", "f1_macro", "n_train",
        ]],
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

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    demo_root = Path(os.getenv("DEMO_DATA_ROOT", str(ROOT / "data" / "demo_data")))
    demo_files = {
        "Stable (normal sentiment)": str(demo_root / "demo_jun2026_stable.csv"),
        "Spike (Jun 21 crisis)":     str(demo_root / "demo_jun2026_spike.csv"),
        "Original sample (2022)":    str(demo_root / "demo_holdout_full.csv"),
    }
    chosen = st.radio(
        "🎬 Demo dataset",
        list(demo_files.keys()),
        horizontal=True,
        index=0,
        key="demo_choice",
    )
    csv_path = demo_files[chosen]

    # Default load uses the daily window (14 days).
    # render_timeline() will call _load() again with 60 days if weekly is selected —
    # _load is cached so the extra call costs nothing on repeated toggles.
    df = _load(DAYS_BY_FREQ["W"], csv_path=csv_path)
    pct_neg = _pct(df, "negative")
 
    # ── Header ──────────────────────────────────────────────────────────────
    render_header(pct_neg)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
 
    # ── KPI tiles ───────────────────────────────────────────────────────────
    render_kpi_tiles(df)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
 
    # ── Timeline  |  Alerts ─────────────────────────────────────────────────
    # Note: render_timeline() manages its own data loading based on the toggle.
    # render_alerts() stays on the 14-day df — showing recent negatives is fine
    # on a 14-day window regardless of the chart toggle.
    col_left, col_right = st.columns([1.45, 1])
    with col_left:
        render_timeline(df, csv_path=csv_path)
    with col_right:
        render_alerts(df, n=4)
 
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
 
    # ── Topic breakdown  |  Word cloud ──────────────────────────────────────
    col_a, col_b = st.columns([1, 1])
    with col_a:
        render_topic_breakdown()
    with col_b:
        render_word_cloud(df)
 
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
 
    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="margin-top:1.5rem;padding-top:1rem;
            border-top:1px solid {BORDER};font-size:11px;
            color:{TEXT_SEC};text-align:center;">
            BrewLeaf Social Listener &nbsp;·&nbsp; Group 9 &nbsp;·&nbsp;
            Batch inference every 6 hours &nbsp;·&nbsp;
            Negative spike alert when sentiment &gt; {NEGATIVE_THRESHOLD}%
        </div>""",
        unsafe_allow_html=True,
    )
 
 
if __name__ == "__main__":
    main()