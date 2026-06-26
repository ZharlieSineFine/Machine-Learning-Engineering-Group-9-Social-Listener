"""MLOps Monitor — Social Listener · Group 9.

Data scientist view: model health, shadow deploy comparison, MLflow run history,
the retrain F1-negative trend, and drift signals.

Shares the BrewLeaf design system with the Marketing page via ``dashboard/theme.py``
— the light/dark toggle (``theme.topbar``) and palette (``theme.palette`` / CSS vars)
are identical across both views. HTML references CSS variables so the toggle
re-themes everything live; Plotly reads concrete hex from ``theme.palette()``.

Data sources:
    - MLflow registry/runs  → scripts/compare_mlflow_models.py helpers
    - Shadow neg-volume     → predictions table (Production vs Staging, 14-day trend)
    - Shadow agreement KPI  → predictions table / GET /shadow/log fallback
    - Drift signal + history→ monitoring_reports table (evaluate_and_monitor)
    - Sentiment timeline    → reviews table via dashboard.data, demo fallback

Owner: Amelia.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Path wiring — works from a local checkout (repo root) and the docker image
# (dashboard/ copied to /app with scripts/ as a sibling).
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "scripts").exists():
    ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# dashboard/ dir on path so `import theme` / `from data import ...` resolve from pages/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import theme  # noqa: E402
from theme import (  # noqa: E402
    AMBER, BLUE, BORDER, BROWN, CARD2, CARD_BG, FAINT, PAGE_BG, RED, TEAL,
    TEXT_PRI, TEXT_SEC,
)

# Semantic CSS-var aliases (themed by theme.inject) for status colours.
GOOD_FG, GOOD_BG, GOOD_BD = "var(--good-fg)", "var(--good-bg)", "var(--good-bd)"
BAD_FG, BAD_BG, BAD_BD = "var(--bad-fg)", "var(--bad-bg)", "var(--bad-bd)"
TEAL_SOFT, RED_SOFT, BLUE_SOFT = "var(--teal-soft)", "var(--red-soft)", "var(--blue-soft)"
ROW_HL = "var(--row-hl)"

API_URL = os.getenv("API_URL", "http://localhost:8000")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.25"))
NEGATIVE_THRESHOLD = int(os.getenv("NEGATIVE_THRESHOLD", "25"))
DAYS_BY_FREQ = {"D": 14, "W": 60}

st.set_page_config(
    page_title="MLOps Monitor · Social Listener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------

def _panel_title(title: str, dot: str = "") -> str:
    dot_html = (
        f'<span style="width:8px;height:8px;border-radius:50%;'
        f'background:{dot};display:inline-block;margin-right:8px;"></span>'
        if dot else ""
    )
    return (
        f'<div style="font-size:11px;font-weight:700;letter-spacing:0.07em;'
        f'color:{TEXT_SEC};text-transform:uppercase;margin-bottom:12px;'
        f'display:flex;align-items:center;">{dot_html}{title}</div>'
    )


def _badge(label: str, color: str, bg: str) -> str:
    return (
        f'<span style="display:inline-block;padding:2px 10px;border-radius:99px;'
        f'font-size:10px;font-weight:700;letter-spacing:0.04em;'
        f'background:{bg};color:{color};">{label}</span>'
    )


def _plotly_layout(height: int, c: dict) -> dict:
    return dict(
        height=height,
        margin=dict(l=4, r=8, t=8, b=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=c["muted"], size=11, family="Hanken Grotesk"),
        xaxis=dict(showgrid=False, color=c["muted"], zeroline=False),
        yaxis=dict(gridcolor=c["grid"], color=c["muted"], zeroline=False),
    )


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _runs_from_checkpoints() -> pd.DataFrame:
    """Fallback: static metrics from trainer_state.json checkpoints (Van's runs)."""
    _CHECKPOINT_RUNS = [
        ("distilbert-baseline",            0.8991, 0.7870, "8604", "production"),
        ("distilbert-weighted-loss",       0.8934, 0.7843, "6453", "archived"),
        ("distilbert-lr-5e-05",            0.8894, 0.7903, "2151", "archived"),
        ("distilbert-lr-2e-05",            0.8818, 0.7802, "2151", "archived"),
        ("distilbert-weighted-loss-oversample", 0.8781, 0.7700, "9988", "archived"),
        ("distilbert-lr-5e-06",            0.8773, 0.7644, "2151", "archived"),
    ]
    rows = []
    for run_name, best_metric, f1_macro, checkpoint, status in _CHECKPOINT_RUNS:
        rows.append({
            "run_id": checkpoint, "model": "DistilBERT", "run_name": run_name,
            "f1_macro": f1_macro, "f1_neg": None, "recall_neg": None,
            "status": status, "source": "checkpoint",
        })
    return pd.DataFrame(rows)


_MODEL_FAMILIES = {"sentiment-baseline": "TF-IDF LR", "sentiment-distilbert": "DistilBERT"}
_STAGE_ORDER = {"production": 0, "staging": 1, "archived": 2, "none": 3}


def _metric(metrics: dict, *keys):
    for k in keys:
        val = metrics.get(k)
        if val is not None:
            return float(val)
    return None


@st.cache_data(ttl=60)
def _fetch_mlflow_runs() -> pd.DataFrame:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        try:
            from scripts.compare_mlflow_models import resolve_tracking_uri
            uri = resolve_tracking_uri(ROOT)
        except Exception:
            uri = None

    if uri:
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient(uri)

            rows = []
            for name, family in _MODEL_FAMILIES.items():
                try:
                    versions = client.search_model_versions(f"name='{name}'")
                except Exception:
                    continue
                for v in versions:
                    stage = (v.current_stage or "None").lower()
                    try:
                        run = client.get_run(v.run_id)
                        metrics = run.data.metrics
                        run_name = run.data.tags.get("mlflow.runName", "")
                    except Exception:
                        metrics, run_name = {}, ""
                    rows.append({
                        "run_id": str(v.run_id)[:7],
                        "model": family,
                        "run_name": run_name or f"v{v.version}",
                        "f1_macro": _metric(metrics, "f1_macro", "test_f1_macro"),
                        "f1_neg": _metric(metrics, "f1_neg", "test_f1_negative", "f1_negative"),
                        "recall_neg": _metric(metrics, "recall_neg", "test_recall_negative"),
                        "status": stage if stage != "none" else "archived",
                        "source": "mlflow",
                        "_version": int(v.version),
                    })

            if rows:
                df = pd.DataFrame(rows)
                df["_ord"] = df["status"].map(lambda s: _STAGE_ORDER.get(s, 9))
                df = (
                    df.sort_values(["_ord", "_version"], ascending=[True, False])
                      .drop_duplicates(subset=["run_id"], keep="first")
                      .drop(columns=["_ord"])
                      .reset_index(drop=True)
                )
                return df
        except Exception:
            pass

    return _runs_from_checkpoints()


@st.cache_data(ttl=30)
def _fetch_shadow_log() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/shadow/log", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []


@st.cache_data(ttl=30)
def _shadow_from_predictions(days: int = 7) -> Optional[dict]:
    dsn = _pg_dsn()
    if not dsn:
        return None
    try:
        import psycopg2
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT text, stage, predicted_label, predicted_at "
                    "FROM predictions WHERE stage IN ('Production','Staging') "
                    "AND predicted_at >= NOW() - make_interval(days => %s) "
                    "ORDER BY predicted_at",
                    (days,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[mlops_monitor] predictions read failed ({exc})")
        return None

    if not rows:
        return None

    latest: dict[str, dict[str, str]] = {}
    for text, stage, label, _ in rows:
        if text == _SHADOW_BACKFILL_TEXT:
            continue
        latest.setdefault(text, {})[stage] = label
    paired = [
        (v["Production"], v["Staging"])
        for v in latest.values()
        if "Production" in v and "Staging" in v
    ]
    if not paired:
        return None
    agree = sum(1 for p, s in paired if p == s)
    prod = pd.Series([p for p, _ in paired]).value_counts().to_dict()
    stag = pd.Series([s for _, s in paired]).value_counts().to_dict()
    return {
        "n": len(paired),
        "agree_pct": agree / len(paired) * 100,
        "prod": prod,
        "stag": stag,
        "source": "predictions",
    }


_SHADOW_BACKFILL_TEXT = "shadow backfill (demo seed)"


@st.cache_data(ttl=5)
def _fetch_shadow_neg_volume(days: int = 14) -> pd.DataFrame:
    dsn = _pg_dsn()
    if not dsn:
        return pd.DataFrame()
    try:
        import psycopg2
        conn = psycopg2.connect(dsn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT date(predicted_at) AS d, stage, count(*) AS n_negative "
                    "FROM predictions "
                    "WHERE stage IN ('Production','Staging') "
                    "  AND predicted_label = 'negative' "
                    "  AND predicted_at >= NOW() - make_interval(days => %s) "
                    "GROUP BY 1, 2 ORDER BY 1",
                    (days,),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        print(f"[mlops_monitor] shadow neg-volume read failed ({exc})")
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["period", "stage", "n_negative"])
    df["period"] = pd.to_datetime(df["period"])
    wide = df.pivot_table(
        index="period", columns="stage", values="n_negative",
        aggfunc="sum", fill_value=0,
    )
    for stage in ("Production", "Staging"):
        if stage not in wide.columns:
            wide[stage] = 0
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full, fill_value=0).rename_axis("period").reset_index()
    return wide[["period", "Production", "Staging"]]


def _pg_dsn() -> Optional[str]:
    user = os.getenv("POSTGRES_USER")
    pw   = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB")
    if all([user, pw, host, db]):
        return f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    return None


def _recorded_drift(dsn: str) -> Optional[dict]:
    import psycopg2
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT drift_score, blocked_promotion, run_date, report_url "
                "FROM monitoring_reports ORDER BY created_at DESC LIMIT 1"
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        return None
    return {
        "mode": "recorded",
        "drift_score": float(row[0]) if row[0] is not None else 0.0,
        "passed_gate": not row[1],
        "run_date": str(row[2]),
        "report_url": row[3],
    }


@st.cache_data(ttl=5)
def _fetch_drift() -> Optional[dict]:
    dsn = _pg_dsn()
    if not dsn:
        return None
    try:
        return _recorded_drift(dsn)
    except Exception as exc:
        print(f"[mlops_monitor] monitoring_reports read failed ({exc})")
        return None


def _drift_history(dsn: str, days: int = 21) -> list[tuple]:
    import psycopg2
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT run_date, drift_score, blocked_promotion FROM ("
                "  SELECT DISTINCT ON (run_date) run_date, drift_score, blocked_promotion "
                "  FROM monitoring_reports WHERE report_type = 'data_drift' "
                "  ORDER BY run_date DESC, created_at DESC LIMIT %s"
                ") t ORDER BY run_date ASC",
                (days,),
            )
            return [
                (str(d), float(s) if s is not None else 0.0, bool(b))
                for d, s, b in cur.fetchall()
            ]
    finally:
        conn.close()


@st.cache_data(ttl=5)
def _fetch_drift_history() -> list[tuple]:
    dsn = _pg_dsn()
    if not dsn:
        return []
    try:
        return _drift_history(dsn)
    except Exception as exc:
        print(f"[mlops_monitor] drift history read failed ({exc})")
        return []


_DEMO_F1_NEG_TREND: dict[str, list[float]] = {
    "TF-IDF LR": [0.860, 0.865, 0.868, 0.869, 0.870],
    "DistilBERT": [0.872, 0.881, 0.888, 0.895, 0.903],
}


@st.cache_data(ttl=60)
def _fetch_f1_neg_trend(n_runs: int = 5) -> pd.DataFrame:
    from scripts.compare_mlflow_models import (
        fetch_retrain_f1_trend, resolve_tracking_uri, trend_x_labels,
    )
    try:
        uri = resolve_tracking_uri(ROOT)
        df = fetch_retrain_f1_trend(uri, n_runs=n_runs, root=ROOT)
        if not df.empty:
            return df
    except Exception:
        pass

    rows: list[dict] = []
    for model, values in _DEMO_F1_NEG_TREND.items():
        tail = values[-n_runs:]
        labels = trend_x_labels(len(tail))
        for i, f1 in enumerate(tail):
            rows.append({"model": model, "x": labels[i], "x_ord": i, "f1_neg": f1})
    return pd.DataFrame(rows)


def _demo_timeline_14d() -> pd.DataFrame:
    base = pd.Timestamp.utcnow().normalize()
    neg = [19, 21, 18, 22, 20, 23, 19, 21, 20, 22, 24, 21, 23, 20]
    neu = [8] * 14
    rows = []
    for i in range(14):
        d = base - pd.Timedelta(days=13 - i)
        rows.append({
            "period": d,
            "pct_negative": neg[i],
            "pct_neutral": neu[i],
            "pct_positive": 100 - neg[i] - neu[i],
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=5)
def _fetch_sentiment_timeline(days: int = 14, freq: str = "D") -> pd.DataFrame:
    dsn = _pg_dsn()
    try:
        try:
            from dashboard.data import load_reviews, sentiment_timeline
        except Exception:
            from data import load_reviews, sentiment_timeline

        df = load_reviews(dsn=dsn, days=days)
        if df is not None and not df.empty:
            tl = sentiment_timeline(df, freq=freq, time_col="review_date")
            if not tl.empty:
                tl = tl.copy()
                if "pct_neutral" not in tl.columns:
                    tl["pct_neutral"] = (
                        100 - tl["pct_positive"] - tl["pct_negative"]
                    ).clip(lower=0)
                tl["period"] = pd.to_datetime(tl["period"])
                return tl[["period", "pct_positive", "pct_negative", "pct_neutral"]]
    except Exception as exc:
        print(f"[mlops_monitor] sentiment timeline unavailable ({exc}); demo fallback")
    return _demo_timeline_14d()


@st.cache_data(ttl=30)
def _fetch_api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return {"status": "unreachable", "model_loaded": False, "model_source": "none"}


# ---------------------------------------------------------------------------
# Derived view-model
# ---------------------------------------------------------------------------

def _prod_row(runs_df: pd.DataFrame) -> Optional[pd.Series]:
    if runs_df.empty:
        return None
    prod = runs_df[runs_df["status"] == "production"]
    return prod.iloc[0] if not prod.empty else runs_df.iloc[0]


def _shadow_summary(log: list[dict]) -> dict:
    shadow = [e for e in log if e.get("staging_label") is not None]
    if not shadow:
        return {"n": 0, "agree_pct": None, "prod": {}, "stag": {}, "source": "api"}
    agree = sum(1 for e in shadow if e["production_label"] == e["staging_label"])
    prod_counts = pd.Series([e["production_label"] for e in shadow]).value_counts().to_dict()
    stag_counts = pd.Series([e["staging_label"] for e in shadow]).value_counts().to_dict()
    return {
        "n": len(shadow),
        "agree_pct": agree / len(shadow) * 100,
        "prod": prod_counts,
        "stag": stag_counts,
        "source": "api",
    }


def _shadow_view() -> dict:
    db = _shadow_from_predictions(7)
    if db:
        return db
    return _shadow_summary(_fetch_shadow_log())


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_header(runs_df: pd.DataFrame) -> None:
    health = _fetch_api_health()
    api_ok = health.get("status") == "ok"
    src = health.get("model_source", "?")
    now = datetime.now(timezone.utc).strftime("%a %d %b %Y · %H:%M UTC")

    prod = _prod_row(runs_df)
    prod_name = prod["run_name"] if prod is not None else "logreg-final"
    stag = runs_df[runs_df["status"] == "staging"] if not runs_df.empty else pd.DataFrame()
    stag_name = stag.iloc[0]["run_name"] if not stag.empty else "distilbert-final"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;padding:2px 0;">
              <div style="width:36px;height:36px;border-radius:50%;background:{BROWN};
                          display:flex;align-items:center;justify-content:center;
                          font-size:18px;">🔬</div>
              <div>
                <p style="margin:0;font-family:'Space Grotesk';font-size:18px;
                          font-weight:700;color:{TEXT_PRI};">
                  MLOps Monitor &nbsp;·&nbsp; Social Listener</p>
                <p style="margin:0;font-size:12px;color:{TEXT_SEC};">
                  Data scientist view &nbsp;·&nbsp; {now} &nbsp;·&nbsp; API: {src}
                  &nbsp;·&nbsp; <span style="color:{TEAL};font-weight:600;">{prod_name}</span> Prod
                  &nbsp;·&nbsp; <span style="color:{AMBER};font-weight:600;">{stag_name}</span> Staging</p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        color, bg, bd, label = (
            (GOOD_FG, GOOD_BG, GOOD_BD, "✓ &nbsp;API healthy") if api_ok
            else (BAD_FG, BAD_BG, BAD_BD, "⚠ &nbsp;API unreachable")
        )
        st.markdown(
            f"""<div style="margin-top:8px;padding:7px 14px;background:{bg};
                border-radius:8px;border:1px solid {bd};text-align:center;">
                <span style="color:{color};font-size:12px;font-weight:600;">{label}</span></div>""",
            unsafe_allow_html=True,
        )


def _kpi_card(label: str, value: str, value_color: str, sub: str, sub_color: str) -> str:
    return (
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:10px;'
        f'padding:14px 16px;">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:0.07em;'
        f'text-transform:uppercase;color:{TEXT_SEC};margin-bottom:6px;">{label}</div>'
        f'<div style="font-family:\'Space Grotesk\';font-size:28px;font-weight:700;'
        f'line-height:1;font-variant-numeric:tabular-nums;color:{value_color};">{value}</div>'
        f'<div style="font-size:11px;color:{sub_color};margin-top:4px;">{sub}</div>'
        f'</div>'
    )


def render_kpi_row(runs_df: pd.DataFrame, drift: Optional[dict],
                   trend: pd.DataFrame) -> None:
    prod = _prod_row(runs_df)

    def fmt(v, nd=2):
        return f"{v:.{nd}f}" if v is not None and pd.notna(v) else "—"

    f1_neg = fmt(prod.get("f1_neg")) if prod is not None else "—"
    recall = fmt(prod.get("recall_neg")) if prod is not None else "—"
    f1_macro = fmt(prod.get("f1_macro")) if prod is not None else "—"
    prod_name = prod["run_name"] if prod is not None else "—"

    recall_sub, recall_sub_col = f"{prod_name} · test set", TEXT_SEC
    if not trend.empty:
        lr = trend[trend["model"] == "TF-IDF LR"].sort_values("x_ord")
        if len(lr) >= 2:
            delta = lr.iloc[-1]["f1_neg"] - lr.iloc[-2]["f1_neg"]
            arrow = "↑" if delta >= 0 else "↓"
            recall_sub = f"{arrow} {delta:+.2f} F1-neg vs last run"
            recall_sub_col = GOOD_FG if delta >= 0 else RED

    if drift and "error" not in drift:
        ds = drift.get("drift_score", 0.0)
        gate_ok = drift.get("passed_gate", True)
        drift_val = f"{ds:.3f}"
        drift_col = TEAL if ds < 0.15 else (AMBER if ds < DRIFT_THRESHOLD else RED)
        drift_sub = (f"Gate passed (τ = {DRIFT_THRESHOLD})" if gate_ok
                     else f"↑ Gate blocked (τ = {DRIFT_THRESHOLD})")
        drift_sub_col = GOOD_FG if gate_ok else RED
    else:
        drift_val, drift_col, drift_sub, drift_sub_col = "—", TEXT_SEC, "no monitor run yet", TEXT_SEC

    cards = "".join([
        _kpi_card("F1 Negative (Prod)", f1_neg, AMBER, f"{prod_name} · test set", TEXT_SEC),
        _kpi_card("Recall Negative", recall, TEAL, recall_sub, recall_sub_col),
        _kpi_card("F1 Macro", f1_macro, TEXT_PRI, "reference metric only", TEXT_SEC),
        _kpi_card("Drift Score", drift_val, drift_col, drift_sub, drift_sub_col),
    ])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'
        f'margin:14px 0;">{cards}</div>',
        unsafe_allow_html=True,
    )


def render_sentiment_timeline() -> None:
    c = theme.palette()
    with st.container(border=True):
        head_l, head_r = st.columns([3, 1])
        with head_r:
            freq = st.radio("timeline_freq", ["Daily", "Weekly"], horizontal=True,
                            label_visibility="collapsed", key="timeline_freq")
        code = "D" if freq == "Daily" else "W"
        tl = _fetch_sentiment_timeline(DAYS_BY_FREQ[code], code)
        label = "14 days" if code == "D" else "8 weeks"

        with head_l:
            st.markdown(
                f'<div style="font-size:11px;font-weight:700;letter-spacing:0.07em;'
                f'color:{TEXT_SEC};text-transform:uppercase;padding-top:6px;'
                f'display:flex;align-items:center;">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{c["teal"]};'
                f'display:inline-block;margin-right:8px;"></span>Sentiment trend — {label}</div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure()
        for col, color, dash, name in [
            ("pct_positive", c["teal"], None, "Positive"),
            ("pct_neutral", c["amber"], "dot", "Neutral"),
            ("pct_negative", c["red"], "dash", "Negative"),
        ]:
            fig.add_trace(go.Scatter(
                x=tl["period"], y=tl[col], name=name, mode="lines+markers",
                line=dict(color=color, width=2.2, dash=dash), marker=dict(size=4),
            ))
        fig.add_hline(y=NEGATIVE_THRESHOLD, line_dash="dash", line_color=c["red"],
                      line_width=1, opacity=0.4)

        layout = _plotly_layout(250, c)
        layout["yaxis"].update(range=[0, 100], ticksuffix="%")
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                                x=0, font=dict(color=c["muted"], size=11), bgcolor="rgba(0,0,0,0)")
        layout["hovermode"] = "x unified"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _sparkline_svg(points: list, c: dict, *, w: int = 320, h: int = 64, pad: int = 8) -> str:
    if len(points) < 2:
        return ""
    scores = [p[1] for p in points]
    ymax = max(0.5, max(scores) * 1.1)
    n, iw, ih = len(points), w - 2 * pad, h - 2 * pad

    def xy(i: int, s: float) -> tuple:
        return pad + iw * i / (n - 1), pad + ih * (1 - min(s, ymax) / ymax)

    pts = [xy(i, s) for i, s in enumerate(scores)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{pad},{h - pad} " + line + f" {w - pad},{h - pad}"
    _, ty = xy(0, DRIFT_THRESHOLD)
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" '
        f'fill="{c["red"] if (points[i][2] or scores[i] >= DRIFT_THRESHOLD) else c["teal"]}"/>'
        for i, (x, y) in enumerate(pts)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}" preserveAspectRatio="none" '
        f'style="display:block;overflow:visible;margin-top:4px;">'
        f'<polyline points="{area}" fill="{c["teal"]}" fill-opacity="0.08" stroke="none"/>'
        f'<line x1="{pad}" y1="{ty:.1f}" x2="{w - pad}" y2="{ty:.1f}" stroke="{c["amber"]}" '
        f'stroke-width="1" stroke-dasharray="4 3" opacity="0.7"/>'
        f'<polyline points="{line}" fill="none" stroke="{c["teal"]}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round"/>{dots}</svg>'
    )


def render_drift_panel(drift: Optional[dict]) -> None:
    c = theme.palette()
    with st.container(border=True):
        st.markdown(_panel_title("Sentiment drift (label PSI)", RED), unsafe_allow_html=True)

        if not drift or "error" in drift or "drift_score" not in drift:
            st.markdown(
                f'<div style="border:1px dashed {BORDER};border-radius:8px;padding:18px;'
                f'text-align:center;color:{TEXT_SEC};font-size:13px;">🚧 No drift run recorded yet — '
                f'run <code>scripts/demo_spike.sh</code> to record a drift score.</div>',
                unsafe_allow_html=True,
            )
            return

        score = drift.get("drift_score", 0.0)
        gate_ok = drift.get("passed_gate", True)
        score_col = TEAL if score < 0.1 else (AMBER if score < DRIFT_THRESHOLD else RED)
        badge = (_badge("Gate passed", TEAL, TEAL_SOFT) if gate_ok
                 else _badge("Gate blocked", BAD_FG, RED_SOFT))

        history = _fetch_drift_history()
        spark = _sparkline_svg(history, c)
        if spark:
            trend = (
                spark
                + f'<p style="margin:10px 0 8px;font-size:12px;color:{TEXT_SEC};">'
                f'{len(history)}-day trend &nbsp;·&nbsp; dashed = threshold {DRIFT_THRESHOLD}'
                f' &nbsp;·&nbsp; PSI ≲0.1 stable · 0.1–0.25 moderate · &gt;0.25 significant</p>'
            )
        else:
            trend = (
                f'<p style="margin:6px 0 0;font-size:11px;color:{TEXT_SEC};">'
                f'threshold {DRIFT_THRESHOLD} — trend appears once a few daily points land.</p>'
            )

        body = f"""
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:16px;">
            <div>
                <p style="margin:0;font-size:11px;color:{TEXT_SEC};text-transform:uppercase;
                          letter-spacing:0.05em;">Label PSI</p>
                <p style="margin:2px 0 0;font-family:'Space Grotesk';font-size:2rem;font-weight:700;
                          color:{score_col};">{score:.3f}</p>
            </div>
            <div>{badge}</div>
        </div>
        {trend}
        """
        st.markdown(body, unsafe_allow_html=True)


def render_mlflow_run_history(runs_df: pd.DataFrame, trend: pd.DataFrame) -> None:
    with st.container(border=True):
        st.markdown(_panel_title("MLflow run history", TEAL), unsafe_allow_html=True)

        if runs_df.empty:
            st.markdown(
                f'<div style="border:1px dashed {BORDER};border-radius:8px;padding:18px;'
                f'text-align:center;color:{TEXT_SEC};font-size:13px;">🚧 MLflow not reachable — '
                f'start the stack with docker compose up -d mlflow</div>',
                unsafe_allow_html=True,
            )
            return

        STATUS = {
            "production": (TEAL, TEAL_SOFT),
            "staging":    (AMBER, "var(--amber)"),
            "archived":   (TEXT_SEC, CARD2),
        }

        rows_html = ""
        for _, r in runs_df.head(6).iterrows():
            status = str(r.get("status", "archived")).lower()
            badge_col, badge_bg = STATUS.get(status, STATUS["archived"])
            if status == "staging":
                badge_bg = BLUE_SOFT
                badge_col = AMBER
            model_col = TEAL if r["model"] == "TF-IDF LR" else AMBER
            badge = _badge(status, badge_col, badge_bg)
            is_prod = status == "production"
            row_bg = ROW_HL if is_prod else "transparent"

            f1m = f"{r['f1_macro']:.2f}" if pd.notna(r.get("f1_macro")) else "—"
            f1n = f"{r['f1_neg']:.3f}".rstrip("0").rstrip(".") if pd.notna(r.get("f1_neg")) else "—"
            rec = f"{r['recall_neg']:.2f}" if pd.notna(r.get("recall_neg")) else "—"
            f1n_disp = f"<strong>{f1n}</strong>" if is_prod or status == "staging" else f1n

            rows_html += f"""
            <tr style="background:{row_bg};">
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  font-family:'Spline Sans Mono';color:{TEXT_SEC};font-size:11px;">{r['run_id']}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  color:{model_col};font-weight:700;font-size:12px;">{r['model']}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  color:{TEXT_PRI};font-size:12px;">{r.get('run_name','')}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  color:{TEXT_PRI};font-size:12px;text-align:right;">{f1m}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  color:{TEXT_PRI};font-size:12px;text-align:right;">{f1n_disp}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};
                  color:{TEXT_PRI};font-size:12px;text-align:right;">{rec}</td>
              <td style="padding:8px 8px;border-bottom:1px solid {BORDER};text-align:center;">{badge}</td>
            </tr>"""

        th = (f"text-align:left;font-size:10px;font-weight:700;letter-spacing:0.05em;"
              f"text-transform:uppercase;color:{TEXT_SEC};padding:6px 8px;"
              f"border-bottom:1px solid {BORDER};")
        st.markdown(
            f"""
            <div style="font-size:12px;color:{TEXT_SEC};margin-bottom:10px;">
              <strong style="color:{TEAL};">TF-IDF LR</strong> · Production &nbsp;|&nbsp;
              <strong style="color:{AMBER};">DistilBERT</strong> · Staging (shadow)</div>
            <table style="width:100%;border-collapse:collapse;font-size:12px;">
              <thead><tr>
                <th style="{th}">Run ID</th><th style="{th}">Model</th><th style="{th}">Run name</th>
                <th style="{th};text-align:right;">F1 macro</th><th style="{th};text-align:right;">F1 neg</th>
                <th style="{th};text-align:right;">Recall neg</th><th style="{th};text-align:center;">Status</th>
              </tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            <div style="font-size:10px;color:{FAINT};text-transform:uppercase;
                        letter-spacing:0.06em;margin:12px 0 2px;">F1-negative trend (last 5 retrain runs)</div>
            """,
            unsafe_allow_html=True,
        )

        if trend.empty:
            st.markdown(f'<div style="font-size:11px;color:{FAINT};">no retrain history yet</div>',
                        unsafe_allow_html=True)
            return

        c = theme.palette()
        fig = go.Figure()
        styles = {"TF-IDF LR": (c["teal"], "solid"), "DistilBERT": (c["amber"], "dash")}
        for model, group in trend.groupby("model", sort=False):
            group = group.sort_values("x_ord")
            col, dash = styles.get(model, (c["muted"], "solid"))
            y_vals = group["f1_neg"].tolist()
            fig.add_trace(go.Scatter(
                x=group["x"], y=y_vals, name=model, mode="lines+markers+text",
                line=dict(color=col, width=2, dash=dash), marker=dict(size=6, color=col),
                text=[f"{v:.2f}" if i == len(y_vals) - 1 else "" for i, v in enumerate(y_vals)],
                textposition="middle right", textfont=dict(color=col, size=11),
            ))
        y_min, y_max = trend["f1_neg"].min(), trend["f1_neg"].max()
        pad = max((y_max - y_min) * 0.25, 0.01)
        layout = _plotly_layout(110, c)
        layout["yaxis"].update(range=[y_min - pad, y_max + pad], tickformat=".2f")
        layout["legend"] = dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left",
                                x=0, font=dict(color=c["muted"], size=10), bgcolor="rgba(0,0,0,0)")
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_shadow_panel(shadow: dict, drift: Optional[dict]) -> None:
    c = theme.palette()
    with st.container(border=True):
        st.markdown(_panel_title("Shadow deploy — production vs staging", BLUE), unsafe_allow_html=True)

        vol = _fetch_shadow_neg_volume(14)

        if vol.empty:
            st.markdown(
                f'<div style="border:1px dashed {BORDER};border-radius:8px;padding:18px;'
                f'text-align:center;color:{TEXT_SEC};font-size:13px;">🚧 No shadow predictions yet — '
                f'promote a candidate to Staging and send traffic to /predict.</div>',
                unsafe_allow_html=True,
            )
            return

        if shadow.get("n"):
            summary = (
                f'<strong style="color:{TEXT_PRI};">{shadow["n"]}</strong> paired predictions'
            )
        else:
            summary = "Staging runs alongside Production · responses not served to users"

        prod_total = int(vol["Production"].sum())
        stag_total = int(vol["Staging"].sum())

        st.markdown(
            f'<div style="font-size:12px;color:{TEXT_SEC};margin-bottom:2px;">{summary}</div>'
            f'<div style="font-size:11px;color:{FAINT};margin-bottom:10px;">'
            f'Daily volume of <strong>negative</strong> predictions · last 14 days '
            f'&nbsp;·&nbsp; Production {prod_total} · Staging {stag_total} (window total)</div>'
            f'<div style="display:flex;gap:16px;margin-bottom:4px;font-size:11px;color:{TEXT_SEC};">'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{TEAL};'
            f'border-radius:2px;margin-right:4px;"></span>Production</span>'
            f'<span><span style="display:inline-block;width:10px;height:10px;background:{AMBER};'
            f'border-radius:2px;margin-right:4px;"></span>Staging</span></div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        for col, color, dash, name in [
            ("Production", c["teal"], None, "Production"),
            ("Staging", c["amber"], "dash", "Staging"),
        ]:
            fig.add_trace(go.Scatter(
                x=vol["period"], y=vol[col], name=name, mode="lines+markers",
                line=dict(color=color, width=2.2, dash=dash), marker=dict(size=4),
            ))
        layout = _plotly_layout(180, c)
        layout["yaxis"].update(rangemode="tozero", title=None)
        layout["hovermode"] = "x unified"
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        if drift and "error" not in drift and not drift.get("passed_gate", True):
            gate_html = (
                f'<div style="margin-top:10px;margin-bottom:16px;padding:10px 14px;background:{BAD_BG};'
                f'border-radius:6px;border:1px solid {BAD_BD};">'
                f'<div style="font-size:11px;font-weight:700;color:{BAD_FG};margin-bottom:3px;">'
                f'🔒 Promotion gate blocked</div>'
                f'<div style="font-size:11px;color:{BAD_FG};">Drift score '
                f'({drift.get("drift_score", 0):.3f}) exceeds τ. Staging shadow window paused '
                f'until drift resolves.</div></div>'
            )
        else:
            gate_html = (
                f'<div style="margin-top:10px;margin-bottom:16px;padding:10px 14px;background:{GOOD_BG};'
                f'border-radius:6px;border:1px solid {GOOD_BD};">'
                f'<div style="font-size:11px;font-weight:700;color:{GOOD_FG};">✓ Promotion gate open</div></div>'
            )

        st.markdown(gate_html, unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        f'<div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid {BORDER};'
        f'font-size:11px;color:{TEXT_SEC};text-align:center;">'
        f'MLOps Monitor &nbsp;·&nbsp; Group 9 &nbsp;·&nbsp; MLflow &nbsp;·&nbsp; '
        f'Evidently &nbsp;·&nbsp; Airflow</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _auto_refresh(interval_ms: int = 600000) -> None:
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_ms, key="demo_autorefresh")
    except Exception:
        pass


def main() -> None:
    theme.inject(theme.active_theme())
    st.markdown(
        '<style>div[role="radiogroup"]{flex-direction:row !important;'
        'flex-wrap:nowrap !important;justify-content:flex-end;}</style>',
        unsafe_allow_html=True,
    )
    _auto_refresh()
    theme.topbar(active="mlops")

    runs_df = _fetch_mlflow_runs()
    drift = _fetch_drift()
    shadow = _shadow_view()
    trend = _fetch_f1_neg_trend(5)

    render_header(runs_df)
    render_kpi_row(runs_df, drift, trend)

    left, right = st.columns([1, 1])
    with left:
        render_sentiment_timeline()
    with right:
        render_drift_panel(drift)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    left2, right2 = st.columns([1, 1])
    with left2:
        render_mlflow_run_history(runs_df, trend)
    with right2:
        render_shadow_panel(shadow, drift)

    render_footer()


if __name__ == "__main__":
    main()