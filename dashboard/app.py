"""Streamlit dashboard — Marketing view (BrewLeaf design).

Sections: top bar (view switch + light/dark) · header + status banner · 4 KPI tiles ·
sentiment timeline | alerts · negative-word cloud (inline text) · footer.

Data: reviews via the serving API (`GET /reviews`), with a Gold-parquet/CSV offline
fallback — the Marketing page does not read Postgres directly.

Owner: Amelia.  Run locally: streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# dashboard/ dir on path for sibling modules (data, theme)
sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_GOLD_ROOT = Path(os.getenv("GOLD_ROOT", str(ROOT / "data" / "gold")))

from data import (
    latest_batch,
    load_reviews,
    load_reviews_via_api,
    negative_word_counts,
    sentiment_timeline,
)
import theme
from theme import (
    AMBER, BORDER, BROWN, CARD2, CARD_BG, RED, TEAL, TEXT_PRI, TEXT_SEC,
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
NEGATIVE_THRESHOLD = int(os.getenv("NEG_THRESHOLD", "25"))  # % — spike alert threshold
DAYS_BY_FREQ = {"D": 14, "W": 60}

st.set_page_config(
    page_title="BrewLeaf · Brand Sentiment", page_icon="🍃",
    layout="wide", initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(df: pd.DataFrame, label: str) -> float:
    if df.empty:
        return 0.0
    return round(df["label"].eq(label).mean() * 100, 1)


def _digest_line() -> str:
    now = datetime.now(timezone.utc)
    hour = (now.hour // 6) * 6
    stamp = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    return f"Morning digest&nbsp; ·&nbsp; {stamp:%a %d %b %Y · %H:%M} UTC&nbsp; ·&nbsp; Last batch {hour:02d}:00"


def _card(content_html: str, padding: str = "18px 20px") -> None:
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:12px;padding:{padding};">{content_html}</div>',
        unsafe_allow_html=True,
    )


def _section_title(svg: str, title: str) -> str:
    return (
        f'<div style="font-size:11px;font-weight:600;color:{TEXT_SEC};text-transform:uppercase;'
        f'letter-spacing:.06em;display:flex;align-items:center;gap:8px;margin-bottom:12px;">'
        f'{svg}{title}</div>'
    )


# small inline icons (stroke=currentColor inherits the muted title colour)
_IC_TREND = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 17 9 11 13 15 21 6"></polyline><polyline points="14 6 21 6 21 13"></polyline></svg>'
_IC_BELL = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>'
_IC_CLOUD = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7V5a1 1 0 0 1 1-1h14a1 1 0 0 1 1 1v2"></path><path d="M9 20h6"></path><path d="M12 4v16"></path></svg>'


def _auto_refresh(interval_ms: int = 5000) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key="demo_autorefresh")
    except Exception:
        pass


@st.cache_data(ttl=5)
def _load(days: int) -> pd.DataFrame:
    # Primary path: the serving API owns DB access (GET /reviews).
    try:
        df = load_reviews_via_api(API_URL, days=days)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        print(f"[dashboard] API /reviews unavailable ({exc}); falling back to local files")
    # Offline fallback (Gold parquet → CSV) — no direct DB from the Marketing page.
    return load_reviews(dsn=None, gold_root=DEFAULT_GOLD_ROOT, days=days)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def render_header(pct_neg: float) -> None:
    if pct_neg >= NEGATIVE_THRESHOLD:
        banner = (
            f'<div style="display:flex;align-items:center;gap:8px;padding:9px 16px;'
            f'background:var(--bad-bg);border:1px solid var(--bad-bd);border-radius:9px;">'
            f'<span style="color:var(--bad-fg);font-size:13px;font-weight:600;">'
            f'⚠&nbsp; {pct_neg:.1f}% negative — above threshold</span></div>'
        )
    else:
        banner = (
            f'<div style="display:flex;align-items:center;gap:8px;padding:9px 16px;'
            f'background:var(--good-bg);border:1px solid var(--good-bd);border-radius:9px;">'
            f'<span style="color:var(--good-fg);font-size:13px;font-weight:600;">'
            f'✓&nbsp; Sentiment normal</span></div>'
        )
    st.markdown(
        f'<div style="display:flex;align-items:center;justify-content:space-between;gap:18px;'
        f'margin:6px 0 20px;flex-wrap:wrap;">'
        f'<div style="display:flex;align-items:center;gap:14px;">'
        f'<div style="width:42px;height:42px;border-radius:50%;background:{BROWN};display:flex;'
        f'align-items:center;justify-content:center;font-size:21px;">🍃</div>'
        f'<div><div style="font-family:\'Space Grotesk\';font-size:21px;font-weight:600;">'
        f'BrewLeaf Social Listener</div>'
        f'<div style="font-size:12.5px;color:{TEXT_SEC};margin-top:2px;">{_digest_line()}</div></div>'
        f'</div>{banner}</div>',
        unsafe_allow_html=True,
    )


def _tile(label: str, value: str, sub: str, color: str | None = None) -> str:
    val_color = f"color:{color};" if color else ""
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:18px 20px;">'
        f'<div style="font-size:10.5px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:.07em;'
        f'font-weight:600;">{label}</div>'
        f'<div style="font-family:\'Space Grotesk\';font-size:32px;font-weight:600;margin-top:10px;'
        f'font-variant-numeric:tabular-nums;{val_color}">{value}</div>'
        f'<div style="font-size:11.5px;color:{TEXT_SEC};margin-top:8px;">{sub}</div></div>'
    )


def render_kpi_tiles(batch: pd.DataFrame, window: pd.DataFrame) -> None:
    total = len(batch)
    win_total = len(window)
    pos, neg, neu = _pct(batch, "positive"), _pct(batch, "negative"), _pct(batch, "neutral")
    n_pos = int(round(pos / 100 * total)); n_neu = int(round(neu / 100 * total))
    neg_color = RED if neg >= NEGATIVE_THRESHOLD else None
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:18px;">'
        + _tile("Posts analysed · batch", f"{total:,}", f"{win_total:,} across 14-day window")
        + _tile("Negative sentiment", f"{neg:.1f}%", f"threshold {NEGATIVE_THRESHOLD}%", neg_color)
        + _tile("Positive sentiment", f"{pos:.1f}%", f"{n_pos} of {total} posts", TEAL)
        + _tile("Neutral sentiment", f"{neu:.1f}%", f"{n_neu} of {total} posts", AMBER)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_timeline() -> None:
    c = theme.palette()
    head_l, head_r = st.columns([3, 1])
    with head_r:
        freq = st.radio("freq", ["Daily", "Weekly"], horizontal=True,
                        label_visibility="collapsed", key="timeline_freq")
    code = "D" if freq == "Daily" else "W"
    days = DAYS_BY_FREQ[code]
    df = _load(days)
    timeline = sentiment_timeline(df, freq=code, time_col="review_date")

    fig = go.Figure()
    if not timeline.empty:
        for col, color, dash, name in [
            ("pct_positive", c["teal"], None, "Positive"),
            ("pct_neutral", c["amber"], "dot", "Neutral"),
            ("pct_negative", c["red"], "dash", "Negative"),
        ]:
            fig.add_trace(go.Scatter(
                x=timeline["period"], y=timeline[col], name=name, mode="lines+markers",
                line=dict(color=color, width=2.2, dash=dash), marker=dict(size=4),
            ))
        fig.add_hline(y=NEGATIVE_THRESHOLD, line_dash="dash", line_color=c["red"],
                      line_width=1, opacity=0.4)
    fig.update_layout(
        height=250, margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"], size=11, family="Hanken Grotesk"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=c["muted"], size=11), bgcolor="rgba(0,0,0,0)"),
        yaxis=dict(range=[0, 100], ticksuffix="%", gridcolor=c["grid"], color=c["muted"]),
        xaxis=dict(showgrid=False, color=c["muted"]),
        hovermode="x unified",
    )
    with head_l:
        st.markdown(
            f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px 12px 0 0;'
            f'border-bottom:none;padding:18px 20px 0;">'
            + _section_title(_IC_TREND, f"Sentiment trend — {'14 days' if code == 'D' else '8 weeks'}"),
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-top:none;'
        f'border-radius:0 0 12px 12px;padding:0 14px 10px;">', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def render_alerts(df: pd.DataFrame, n: int = 4) -> None:
    neg = df[df["label"] == "negative"].copy()
    if "review_date" in neg.columns:
        neg["review_date"] = pd.to_datetime(neg["review_date"], errors="coerce")
        neg = neg.sort_values("review_date", ascending=False)
    cards = ""
    for _, row in neg.head(n).iterrows():
        text = str(row.get("text", ""))
        snippet = text[:200] + ("…" if len(text) > 200 else "")
        date_str = ""
        if "review_date" in row and pd.notna(row.get("review_date")):
            date_str = pd.to_datetime(row["review_date"]).strftime("%d %b %Y")
        source = str(row.get("source", "")).title()
        if not source or source == "Nan":
            source = "Yelp"
        cards += (
            f'<div style="border-left:3px solid {RED};border-radius:0 8px 8px 0;'
            f'background:var(--alert-bg);padding:11px 14px;margin-bottom:9px;">'
            f'<div style="font-size:12.5px;line-height:1.5;color:{TEXT_PRI};">“{snippet}”</div>'
            f'<div style="margin-top:7px;display:flex;align-items:center;gap:8px;font-size:11px;color:{TEXT_SEC};">'
            f'<span style="background:var(--red-soft);color:{RED};padding:2px 9px;border-radius:99px;'
            f'font-size:10px;font-weight:600;">negative</span>{source}{" · " + date_str if date_str else ""}'
            f'</div></div>'
        )
    if not cards:
        cards = f'<div style="color:{TEXT_SEC};font-size:13px;">No negative posts in this batch.</div>'
    _card(_section_title(_IC_BELL, "Alerts · last batch") + cards)


def render_word_cloud(df: pd.DataFrame) -> None:
    counts = negative_word_counts(df, top_n=17)
    if not counts:
        _card(_section_title(_IC_CLOUD, "Top words in negative posts")
              + f'<div style="color:{TEXT_SEC};font-size:13px;">No negative reviews.</div>')
        return
    items = counts.most_common(17)
    hi, lo = items[0][1], items[-1][1]
    spans = ""
    for i, (word, ct) in enumerate(items):
        size = 18 + (ct - lo) / (hi - lo + 1e-9) * (46 - 18)
        if size < 21:
            color = TEXT_SEC
        elif i % 5 == 2:
            color = RED          # red accents
        else:
            color = AMBER
        weight = 600 if size > 34 else 500
        spans += (f'<span style="font-family:\'Space Grotesk\';font-size:{size:.0f}px;'
                  f'font-weight:{weight};color:{color};">{word}</span>')
    _card(_section_title(_IC_CLOUD, "Top words in negative posts")
          + f'<div style="display:flex;flex-wrap:wrap;align-items:baseline;gap:7px 20px;'
            f'line-height:1.15;">{spans}</div>', padding="18px 22px 22px")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _auto_refresh()
    theme.inject(theme.active_theme())
    theme.topbar(active="marketing")

    window = _load(DAYS_BY_FREQ["W"])
    batch = latest_batch(window)
    pct_neg = _pct(batch, "negative")

    render_header(pct_neg)
    render_kpi_tiles(batch, window)

    col_l, col_r = st.columns([1.5, 1])
    with col_l:
        render_timeline()
    with col_r:
        render_alerts(window, n=4)

    render_word_cloud(window)

    st.markdown(
        f'<div style="border-top:1px solid {BORDER};margin-top:18px;padding-top:14px;'
        f'font-size:11px;color:{TEXT_SEC};text-align:center;">'
        f'BrewLeaf Social Listener&nbsp; ·&nbsp; Group 9&nbsp; ·&nbsp; Batch inference every 6 hours'
        f'&nbsp; ·&nbsp; Negative spike alert when sentiment &gt; {NEGATIVE_THRESHOLD}%</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
