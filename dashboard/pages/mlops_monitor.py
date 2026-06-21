"""MLOps Monitor — Social Listener · Group 9.

Data scientist view: model health, shadow deploy comparison,
MLflow run history, pipeline stats, and drift signals.

Sits alongside dashboard/app.py. Add it as a Streamlit page by placing
this file under dashboard/pages/mlops_monitor.py — Streamlit's multi-page
routing picks it up automatically.

Data sources:
    - MLflow registry/runs  → scripts/compare_mlflow_models.py helpers
    - Shadow log            → GET /shadow/log  (api/app/shadow.py)
    - Drift signal          → monitoring_reports table (written by evaluate_and_monitor),
                              live monitoring/drift_checks.run_drift_check() fallback
    - Pipeline stats        → TODO: Airflow XCom / summary JSON from Charlie/Ha
    - Correction queue      → TODO: predictions table from Charlie/Ha

Owner: Amelia.
"""
from __future__ import annotations

import json
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
# Path wiring — same pattern as app.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]   # repo root when file is at dashboard/pages/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants — palette matches the marketing dashboard exactly
# ---------------------------------------------------------------------------
TEAL      = "#1D9E75"
RED       = "#E24B4A"
AMBER     = "#EF9F27"
BROWN     = "#5C3A21"
CARD_BG   = "#1E2128"
PAGE_BG   = "#16181D"
BORDER    = "#2E3039"
TEXT_PRI  = "#E8E6DF"
TEXT_SEC  = "#888780"
BLUE      = "#4A90D9"   # extra accent for "shadow" badge

API_URL   = os.getenv("API_URL", "http://localhost:8000")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MLOps Monitor · Social Listener",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
.stApp {{ background-color: {PAGE_BG}; }}
.block-container {{ padding: 1.5rem 2rem 2rem; max-width: 100%; }}
#MainMenu, footer, header {{ visibility: hidden; }}

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
[data-testid="stMetricDelta"] {{ font-size: 11px !important; }}

hr {{ border-color: {BORDER} !important; margin: 0.5rem 0; }}

.stDataFrame {{ background: {CARD_BG} !important; }}

p, li, span {{ color: {TEXT_PRI}; }}
h1, h2, h3  {{ color: {TEXT_PRI} !important; }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Shared HTML helpers  (same style as app.py)
# ---------------------------------------------------------------------------

def _card(html: str, padding: str = "1rem 1.2rem") -> None:
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:{padding};">{html}</div>',
        unsafe_allow_html=True,
    )


def _section_title(icon: str, title: str) -> str:
    return (
        f'<p style="margin:0 0 0.8rem;font-size:12px;font-weight:600;'
        f'color:{TEXT_SEC};text-transform:uppercase;letter-spacing:0.05em;">'
        f'{icon}&nbsp;&nbsp;{title}</p>'
    )


def _badge(label: str, color: str, bg: str) -> str:
    return (
        f'<span style="background:{bg};color:{color};padding:2px 10px;'
        f'border-radius:99px;font-size:10px;font-weight:600;">{label}</span>'
    )


def _todo_placeholder(message: str) -> str:
    return (
        f'<div style="border:1px dashed {BORDER};border-radius:8px;'
        f'padding:20px;text-align:center;margin-top:4px;">'
        f'<p style="margin:0;font-size:13px;color:{TEXT_SEC};">🚧 &nbsp;{message}</p>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Data fetchers
# ---------------------------------------------------------------------------

def _runs_from_checkpoints() -> pd.DataFrame:
    """Fallback: static metrics extracted from trainer_state.json checkpoint files.
    Used when MLflow is unreachable or has no registered runs yet.
    These are Van's real training runs — best checkpoint per run, sorted by accuracy.
    F1 neg / recall neg are not available in trainer_state (MLflow only).

    To update: re-extract from checkpoints/distilbert-sentiment/*/trainer_state.json.
    """
    _CHECKPOINT_RUNS = [
        # run_name                         best_metric  f1_macro   checkpoint  status
        ("distilbert-baseline",            0.8991,      0.7870,    "8604",     "production"),
        ("distilbert-weighted-loss",       0.8934,      0.7843,    "6453",     "archived"),
        ("distilbert-lr-5e-05",            0.8894,      0.7903,    "2151",     "archived"),
        ("distilbert-lr-2e-05",            0.8818,      0.7802,    "2151",     "archived"),
        ("distilbert-weighted-loss-oversample", 0.8781, 0.7700,    "9988",     "archived"),
        ("distilbert-lr-5e-06",            0.8773,      0.7644,    "2151",     "archived"),
    ]
    rows = []
    for run_name, best_metric, f1_macro, checkpoint, status in _CHECKPOINT_RUNS:
        rows.append({
            "run_id":     checkpoint,
            "model":      "DistilBERT",
            "run_name":   run_name,
            "f1_macro":   f1_macro,
            "f1_neg":     None,
            "recall_neg": None,
            "status":     status,
            "source":     "checkpoint",
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def _fetch_mlflow_runs() -> pd.DataFrame:
    """Pull run history from MLflow via compare_mlflow_models helpers.
    Falls back to checkpoint trainer_state.json files if MLflow is unreachable
    or has no runs yet (e.g. docker compose up but Van hasn't registered models)."""
    try:
        from scripts.compare_mlflow_models import (
            fetch_experiment_runs,
            fetch_registry_versions,
            LOGREG_RUNS,
            DISTILBERT_RUNS,
            resolve_tracking_uri,
        )
        uri = resolve_tracking_uri(ROOT)

        logreg     = fetch_experiment_runs("sentiment-tfidf-logreg", LOGREG_RUNS, uri)
        distilbert = fetch_experiment_runs("sentiment-distilbert", DISTILBERT_RUNS, uri)

        rows = []
        for df, model_type in [(logreg, "TF-IDF LR"), (distilbert, "DistilBERT")]:
            if df.empty:
                continue
            for _, r in df.iterrows():
                rows.append({
                    "run_id":     str(r.get("run_id", ""))[:7],
                    "model":      model_type,
                    "run_name":   r.get("run_name", ""),
                    "f1_macro":   r.get("test_f1_macro"),
                    "f1_neg":     r.get("test_f1_negative"),
                    "recall_neg": r.get("test_recall_negative"),
                    "status":     "archived",
                    "source":     "mlflow",
                })

        # Overlay stage from registry
        try:
            from scripts.compare_mlflow_models import REGISTERED_MODELS
            for model_name in REGISTERED_MODELS:
                versions = fetch_registry_versions(model_name, uri)
                for _, v in versions.iterrows():
                    rid_short = str(v["run_id"])[:7]
                    stage = str(v.get("stage", "archived")).lower()
                    for row in rows:
                        if row["run_id"] == rid_short:
                            row["status"] = stage
        except Exception:
            pass

        # MLflow reachable but no runs logged yet → fall back to checkpoints
        if not rows:
            st.caption("MLflow reachable but no runs found — showing checkpoint metrics.")
            return _runs_from_checkpoints()

        return pd.DataFrame(rows)

    except Exception:
        # MLflow unreachable → fall back to checkpoints
        return _runs_from_checkpoints()


@st.cache_data(ttl=30)
def _fetch_shadow_log() -> list[dict]:
    """GET /shadow/log from the FastAPI service."""
    try:
        r = requests.get(f"{API_URL}/shadow/log", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return []


def _pg_dsn() -> Optional[str]:
    """Postgres DSN from the same env the stack injects (None when unconfigured)."""
    user = os.getenv("POSTGRES_USER")
    pw   = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT", "5432")
    db   = os.getenv("POSTGRES_DB")
    if all([user, pw, host, db]):
        return f"postgresql://{user}:{pw}@{host}:{port}/{db}"
    return None


def _recorded_drift(dsn: str) -> Optional[dict]:
    """Latest row the monitor wrote to ``monitoring_reports`` (None if empty)."""
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


@st.cache_data(ttl=30)   # short TTL so the spike alert shows within seconds during the demo
def _fetch_drift() -> Optional[dict]:
    """Drift signal for the panel.

    Source of truth is the latest ``monitoring_reports`` row the ``evaluate_and_monitor``
    DAG wrote — so the dashboard reflects what actually fired the alert (including the
    demo spike), not a recomputed value. Falls back to a live Evidently check only when
    no monitor run has landed yet (fresh DB).
    """
    dsn = _pg_dsn()
    if dsn:
        try:
            rep = _recorded_drift(dsn)
            if rep:
                return rep
        except Exception as exc:
            print(f"[mlops_monitor] monitoring_reports read failed ({exc}); live fallback")

    try:
        from monitoring.drift_checks import run_drift_check
        result = run_drift_check()
        return {
            "mode": "live",
            "drift_score":       result.drift_score,
            "n_drifted_columns": result.n_drifted_columns,
            "dataset_drift":     result.dataset_drift,
            "passed_gate":       result.passed_gate,
            "evidently_ran":     result.evidently_ran,
            "n_reference":       result.n_reference,
        }
    except Exception as exc:
        return {"error": str(exc)}


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
# Section renderers
# ---------------------------------------------------------------------------


def render_nav(active: str = "mlops") -> None:
    """Top navigation bar — shared between marketing and MLOps pages."""
    marketing_style = (
        f"background:{TEAL};color:#fff;"
        if active == "marketing"
        else f"background:transparent;color:{TEXT_SEC};border:1px solid {BORDER};"
    )
    mlops_style = (
        f"background:{TEAL};color:#fff;"
        if active == "mlops"
        else f"background:transparent;color:{TEXT_SEC};border:1px solid {BORDER};"
    )
    st.markdown(
        f"""<div style="display:flex;gap:8px;margin-bottom:12px;">
            <span style="font-size:11px;color:{TEXT_SEC};
                         align-self:center;margin-right:4px;
                         text-transform:uppercase;letter-spacing:0.05em;">View</span>
            <a href="/" target="_self"
               style="text-decoration:none;padding:5px 16px;border-radius:8px;
                      font-size:12px;font-weight:600;{marketing_style}">
               🍃 Marketing
            </a>
            <a href="/mlops_monitor" target="_self"
               style="text-decoration:none;padding:5px 16px;border-radius:8px;
                      font-size:12px;font-weight:600;{mlops_style}">
               🔬 MLOps Monitor
            </a>
        </div>""",
        unsafe_allow_html=True,
    )

def render_header() -> None:
    health = _fetch_api_health()
    api_ok = health.get("status") == "ok"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:12px;padding:4px 0 2px;">
                <div style="width:36px;height:36px;border-radius:50%;
                            background:{BROWN};display:flex;align-items:center;
                            justify-content:center;font-size:18px;">🔬</div>
                <div>
                    <p style="margin:0;font-size:18px;font-weight:700;color:{TEXT_PRI};">
                        MLOps Monitor &nbsp;·&nbsp; Social Listener
                    </p>
                    <p style="margin:0;font-size:12px;color:{TEXT_SEC};">
                        Data scientist view &nbsp;·&nbsp;
                        {datetime.now(timezone.utc).strftime("%a %d %b %Y · %H:%M UTC")}
                        &nbsp;·&nbsp; API: {health.get("model_source", "?")}
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        color  = "#6FCF6F" if api_ok else "#F09595"
        bg     = "#162414" if api_ok else "#2E1A1A"
        border = "#2A5C2A" if api_ok else "#6B2222"
        label  = "✓ &nbsp;API healthy" if api_ok else "⚠ &nbsp;API unreachable"
        st.markdown(
            f"""<div style="margin-top:8px;padding:7px 14px;background:{bg};
                border-radius:8px;border:1px solid {border};text-align:center;">
                <span style="color:{color};font-size:12px;font-weight:600;">
                    {label}
                </span></div>""",
            unsafe_allow_html=True,
        )


def render_prod_model_kpis(runs_df: pd.DataFrame) -> None:
    """Top KPI row — pull the prod run if we have it, otherwise best available."""
    inner = _section_title("📊", "Production model — overview")

    prod_row = None
    if not runs_df.empty:
        prod_rows = runs_df[runs_df["status"] == "production"]
        prod_row  = prod_rows.iloc[0] if not prod_rows.empty else runs_df.iloc[0]

    f1_macro  = f"{prod_row['f1_macro']:.2f}"  if prod_row is not None and pd.notna(prod_row.get("f1_macro"))  else "—"
    f1_neg    = f"{prod_row['f1_neg']:.2f}"    if prod_row is not None and pd.notna(prod_row.get("f1_neg"))    else "—"
    recall    = f"{prod_row['recall_neg']:.2f}" if prod_row is not None and pd.notna(prod_row.get("recall_neg")) else "—"
    model_lbl = prod_row["model"] if prod_row is not None else "—"
    run_lbl   = prod_row["run_name"] if prod_row is not None else "—"

    html = f"""
    {inner}
    <p style="margin:0 0 0.6rem;font-size:13px;color:{TEXT_SEC};">
        <span style="color:{TEXT_PRI};font-weight:600;">{model_lbl}</span>
        &nbsp;·&nbsp; run: <code style="color:{AMBER};font-size:11px;">{run_lbl}</code>
    </p>
    """
    _card(html)

    c1, c2, c3 = st.columns(3)
    c1.metric("F1 macro",        f1_macro)
    c2.metric("F1 negative",     f1_neg)
    c3.metric("Recall negative", recall)


def render_mlflow_run_history(runs_df: pd.DataFrame) -> None:
    inner = _section_title("🗂", "MLflow run history")

    if runs_df.empty:
        _card(inner + _todo_placeholder(
            "MLflow not reachable — start the stack with docker compose up -d mlflow"
        ))
        return

    STATUS_COLOR = {
        "production": (TEAL,  "#0D2E21"),
        "staging":    (BLUE,  "#111E2E"),
        "failed":     (RED,   "#2E1111"),
        "archived":   (TEXT_SEC, CARD_BG),
    }

    rows_html = ""
    for _, r in runs_df.head(8).iterrows():
        status = str(r.get("status", "archived")).lower()
        col, bg = STATUS_COLOR.get(status, (TEXT_SEC, CARD_BG))
        badge = _badge(status, col, bg)

        f1m = f"{r['f1_macro']:.2f}"    if pd.notna(r.get("f1_macro"))  else "—"
        f1n = f"{r['f1_neg']:.2f}"      if pd.notna(r.get("f1_neg"))    else "—"
        rec = f"{r['recall_neg']:.2f}"  if pd.notna(r.get("recall_neg")) else "—"

        rows_html += f"""
        <tr style="border-bottom:1px solid {BORDER};">
            <td style="padding:8px 6px;color:{TEXT_SEC};font-family:monospace;
                       font-size:11px;">{r['run_id']}</td>
            <td style="padding:8px 6px;color:{TEXT_PRI};font-size:12px;">{r['model']}</td>
            <td style="padding:8px 6px;color:{TEXT_SEC};font-size:11px;">{r.get('run_name','')}</td>
            <td style="padding:8px 6px;color:{TEXT_PRI};font-size:12px;text-align:right;">{f1m}</td>
            <td style="padding:8px 6px;color:{TEXT_PRI};font-size:12px;text-align:right;">{f1n}</td>
            <td style="padding:8px 6px;color:{TEXT_PRI};font-size:12px;text-align:right;">{rec}</td>
            <td style="padding:8px 6px;text-align:center;">{badge}</td>
        </tr>"""

    header_style = f"padding:6px 6px;font-size:10px;color:{TEXT_SEC};text-transform:uppercase;letter-spacing:0.05em;border-bottom:1px solid {BORDER};"
    table_html = f"""
    {inner}
    <table style="width:100%;border-collapse:collapse;">
        <thead>
            <tr>
                <th style="{header_style}">Run ID</th>
                <th style="{header_style}">Model</th>
                <th style="{header_style}">Run name</th>
                <th style="{header_style};text-align:right;">F1 macro</th>
                <th style="{header_style};text-align:right;">F1 neg</th>
                <th style="{header_style};text-align:right;">Recall neg</th>
                <th style="{header_style};text-align:center;">Status</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>"""

    _card(table_html)


def render_shadow_panel() -> None:
    inner = _section_title("🔀", "Shadow deploy — production vs staging")
    log   = _fetch_shadow_log()

    if not log:
        _card(inner + _todo_placeholder(
            "No shadow predictions yet — send requests to /predict to populate this panel. "
            "Staging model logs appear once Van promotes a candidate to Staging in MLflow."
        ))
        return

    shadow_entries = [e for e in log if e.get("staging_label") is not None]
    total    = len(log)
    n_shadow = len(shadow_entries)

    if n_shadow == 0:
        _card(
            inner +
            f'<p style="color:{TEXT_SEC};font-size:13px;">'
            f'{total} production predictions logged. '
            f'No staging model loaded yet — shadow comparison will appear here once Van '
            f'promotes a model to Staging.</p>'
        )
        return

    # Agreement rate
    agree = sum(
        1 for e in shadow_entries
        if e["production_label"] == e["staging_label"]
    )
    agree_pct = agree / n_shadow * 100

    # Per-class breakdown
    prod_counts = pd.Series([e["production_label"] for e in shadow_entries]).value_counts()
    stag_counts = pd.Series([e["staging_label"]    for e in shadow_entries]).value_counts()
    labels = ["positive", "neutral", "negative"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Production",
        x=labels,
        y=[prod_counts.get(l, 0) for l in labels],
        marker_color=TEAL,
    ))
    fig.add_trace(go.Bar(
        name="Staging",
        x=labels,
        y=[stag_counts.get(l, 0) for l in labels],
        marker_color=AMBER,
        opacity=0.8,
    ))
    fig.update_layout(
        barmode="group",
        height=220,
        margin=dict(l=0, r=0, t=4, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SEC, size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=TEXT_SEC, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(showgrid=False, color=TEXT_SEC),
        yaxis=dict(gridcolor=BORDER, color=TEXT_SEC),
    )

    agree_color = TEAL if agree_pct >= 80 else (AMBER if agree_pct >= 60 else RED)
    summary_html = f"""
    {inner}
    <p style="margin:0 0 0.8rem;font-size:12px;color:{TEXT_SEC};">
        {n_shadow} shadow predictions &nbsp;·&nbsp;
        agreement rate:
        <span style="color:{agree_color};font-weight:600;">{agree_pct:.1f}%</span>
    </p>
    """
    st.markdown(
        f'<div style="background:{CARD_BG};border:1px solid {BORDER};'
        f'border-radius:10px;padding:1rem 1.2rem;">{summary_html}',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)


def render_drift_panel() -> None:
    inner  = _section_title("📡", "Evidently drift scores")
    result = _fetch_drift()

    if result is None or "error" in result:
        err = result.get("error", "unknown") if result else "unknown"
        _card(inner + _todo_placeholder(f"Drift check failed: {err}"))
        return

    drift_score = result["drift_score"]
    gate_ok     = result["passed_gate"]
    mode        = result.get("mode", "live")

    # Gate status badge. "Blocked" matches the promotion-gate / alert semantics:
    # a blocked gate is exactly what evaluate_and_monitor alerts on.
    gate_color  = TEAL if gate_ok else RED
    gate_bg     = "#0D2E21" if gate_ok else "#2E1111"
    gate_label  = "Gate passed" if gate_ok else "Gate blocked"
    gate_badge  = _badge(gate_label, gate_color, gate_bg)

    score_bar_pct = min(drift_score * 100 / 0.5, 100)   # 0.5 = visual max
    bar_color = TEAL if drift_score < 0.15 else (AMBER if drift_score < DRIFT_THRESHOLD else RED)

    # Provenance line: a recorded monitor run vs the live Evidently fallback.
    if mode == "recorded":
        meta = (
            f'<p style="margin:8px 0 0;font-size:11px;color:{TEXT_SEC};">'
            f'📥 Recorded by <code>evaluate_and_monitor</code> · run '
            f'<span style="color:{TEXT_PRI};">{result.get("run_date", "?")}</span>'
            f'&nbsp;·&nbsp; report <span style="color:{TEXT_PRI};">'
            f'{result.get("report_url", "—")}</span></p>'
        )
    else:
        evidently = result.get("evidently_ran")
        meta = (
            f'<p style="margin:8px 0 0;font-size:12px;color:{TEXT_SEC};">'
            f'Columns drifted: <span style="color:{TEXT_PRI};">{result.get("n_drifted_columns", "?")}</span>'
            f'&nbsp;·&nbsp; Reference rows: <span style="color:{TEXT_PRI};">{result.get("n_reference", "?")}</span></p>'
        )
        meta += (
            f'<p style="margin:6px 0 0;font-size:11px;color:{AMBER};">'
            f'⚠ Live fallback — no recorded monitor run yet'
            + ('; Evidently stub (train-vs-itself, always passes).' if not evidently else '.')
            + '</p>'
        )

    html = f"""
    {inner}
    <div style="display:flex;align-items:center;gap:16px;margin-bottom:10px;">
        <div>
            <p style="margin:0;font-size:11px;color:{TEXT_SEC};text-transform:uppercase;
                      letter-spacing:0.05em;">Drift score</p>
            <p style="margin:2px 0 0;font-size:2rem;font-weight:700;color:{bar_color};">
                {drift_score:.3f}
            </p>
            <p style="margin:0;font-size:11px;color:{TEXT_SEC};">
                threshold {DRIFT_THRESHOLD}
            </p>
        </div>
        <div style="flex:1;">
            <div style="height:8px;background:{BORDER};border-radius:99px;overflow:hidden;">
                <div style="height:100%;width:{score_bar_pct:.1f}%;
                            background:{bar_color};border-radius:99px;"></div>
            </div>
        </div>
        <div>{gate_badge}</div>
    </div>
    {meta}
    """
    _card(html)






# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    render_nav(active="mlops")
    render_header()
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # ── Production model KPIs + run history ─────────────────────────────────
    runs_df = _fetch_mlflow_runs()
    render_prod_model_kpis(runs_df)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    render_mlflow_run_history(runs_df)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Shadow panel | Drift panel ───────────────────────────────────────────
    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        render_shadow_panel()
    with col_right:
        render_drift_panel()


    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="margin-top:1.5rem;padding-top:1rem;
            border-top:1px solid {BORDER};font-size:11px;
            color:{TEXT_SEC};text-align:center;">
            MLOps Monitor &nbsp;·&nbsp; Group 9 &nbsp;·&nbsp;
            MLflow at localhost:5000 &nbsp;·&nbsp;
            Evidently at localhost:8080 &nbsp;·&nbsp;
            Airflow at localhost:8082
        </div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()