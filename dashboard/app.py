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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_GOLD_ROOT = Path(os.getenv("GOLD_ROOT", str(ROOT / "data" / "gold")))
 
from data import (
    load_reviews,
    negative_word_counts,
    sentiment_timeline,
)

st.write("GOLD_ROOT:", os.getenv("GOLD_ROOT", "NOT SET"))
st.write("Gold path exists:", Path(os.getenv("GOLD_ROOT", "/repo_data/gold")).exists())

test_df = load_reviews(gold_root=Path(os.getenv("GOLD_ROOT")), days=14)
st.write("Rows loaded:", len(test_df))
st.write("Columns:", test_df.columns.tolist())
st.write(test_df.head(2))

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
def _load(days: int) -> pd.DataFrame:
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
        gold_root=ROOT / "data" / "gold",
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
 
 
def render_timeline(
    df: pd.DataFrame, freq: str = "D", time_col: str = "ingested_at",
) -> pd.DataFrame:
    df = df.copy()
    if time_col not in df.columns:
        df[time_col] = pd.date_range(end=pd.Timestamp.utcnow(), periods=len(df), freq="h")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col, "label"])

    df["period"] = df[time_col].dt.to_period(freq).dt.to_timestamp()
    grp = df.groupby("period")

    total    = grp["label"].count()
    positive = grp["label"].apply(lambda s: (s == "positive").sum())
    negative = grp["label"].apply(lambda s: (s == "negative").sum())
    neutral  = grp["label"].apply(lambda s: (s == "neutral").sum())

    out = pd.DataFrame({
        "period":       total.index,
        "n_reviews":    total.values,
        "pct_positive": (positive / total * 100).values,
        "pct_negative": (negative / total * 100).values,
        "pct_neutral":  (neutral  / total * 100).values,
    }).reset_index(drop=True)
    return out
 
 
def render_alerts(df: pd.DataFrame, n: int = 6) -> None:
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
 
    counts = negative_word_counts(df, top_n=80)
 
    if not counts:
        _card(inner + f'<p style="color:{TEXT_SEC};font-size:13px;">No negative reviews.</p>')
        return
 
    try:
        from wordcloud import WordCloud
        import numpy as np
        from PIL import Image
        import io
 
        wc = WordCloud(
            width=900, height=280,
            background_color="#1E2128",
            colormap=None,
            color_func=lambda *args, **kwargs: _wc_color(kwargs.get("word", "")),
            max_words=60,
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
 
 
def _wc_color(word: str) -> str:
    """Colour words red/amber based on rough severity ranking."""
    high = {"terrible", "awful", "worst", "horrible", "disgusting", "rude", "wrong", "cold"}
    if word.lower() in high:
        return RED
    return AMBER
 
 
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
 
 
def render_prediction_probe() -> None:
    st.markdown(
        f'<p style="font-size:12px;font-weight:600;color:{TEXT_SEC};'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.5rem;">'
        f'🔍 &nbsp;&nbsp;LIVE PREDICTION PROBE</p>',
        unsafe_allow_html=True,
    )
 
    text = st.text_area(
        "Review text",
        value="The brown sugar milk tea tasted watered down today.",
        height=90,
        key="probe_text",
        label_visibility="collapsed",
    )
 
    if st.button("Run prediction", type="primary"):
        if not text.strip():
            st.warning("Please enter some text first.")
            return
        try:
            resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)
            resp.raise_for_status()
            label = resp.json().get("label", "unknown")
            colour = {
                "positive": TEAL, "negative": RED, "neutral": AMBER
            }.get(label, TEXT_SEC)
            st.markdown(
                f"""<div style="background:{colour}22;border:1px solid {colour}66;
                    border-radius:8px;padding:12px 16px;margin-top:8px;">
                    <p style="margin:0;font-size:20px;font-weight:700;color:{colour};">
                        {label.upper()}
                    </p>
                    <p style="margin:2px 0 0;font-size:11px;color:{TEXT_SEC};">
                        from {API_URL}/predict
                    </p></div>""",
                unsafe_allow_html=True,
            )
        except requests.exceptions.ConnectionError:
            st.error(f"Cannot reach the API at `{API_URL}`. Run `docker compose up api`.")
        except Exception as exc:
            st.error(f"Error: {exc}")
 
    with st.expander("Health check"):
        if st.button("Ping /health"):
            try:
                h = requests.get(f"{API_URL}/health", timeout=3)
                h.raise_for_status()
                data = h.json()
                ca, cb = st.columns(2)
                ca.metric("Status", data.get("status", "—"))
                cb.metric("Model loaded", "Yes" if data.get("model_loaded") else "No")
            except Exception as exc:
                st.error(f"Health check failed: {exc}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    # Default load uses the daily window (14 days).
    # render_timeline() will call _load() again with 60 days if weekly is selected —
    # _load is cached so the extra call costs nothing on repeated toggles.
    df = _load(DAYS_BY_FREQ["D"])
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
        render_timeline(df)
    with col_right:
        render_alerts(df, n=6)
 
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
 
    # ── Topic breakdown  |  Word cloud ──────────────────────────────────────
    col_a, col_b = st.columns([1, 1])
    with col_a:
        render_topic_breakdown()
    with col_b:
        render_word_cloud(df)
 
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
 
    # ── Developer tools (collapsed — not part of marketing view) ────────────
    with st.expander("🔧 Developer tools", expanded=False):
        st.caption(
            "For demo and grading purposes only — "
            "verify the API is live and test predictions directly."
        )
        render_prediction_probe()
 
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