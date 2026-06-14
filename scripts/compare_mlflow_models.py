"""Pull MLflow tuning runs and registry versions for model comparison.

Used by notebooks/03_mlflow_model_comparison.ipynb and runnable standalone:

    python scripts/compare_mlflow_models.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

# Canonical tuning runs from notebooks 01 and 02 (see TUNING_NOTEBOOKS_INSTRUCTIONS.md).
LOGREG_RUNS = {
    "logreg-baseline",
    "logreg-class-weights",
    "logreg-oversample-neutral",
    "logreg-gridsearch",
    "logreg-final",
}
DISTILBERT_RUNS = {
    "distilbert-baseline",
    "distilbert-weighted-loss",
    "distilbert-weighted-loss-oversample",
    "distilbert-lr-probe-5e-06",
    "distilbert-lr-probe-2e-05",
    "distilbert-lr-probe-5e-05",
    "distilbert-final",
}

METRIC_COLS = [
    "val_f1_negative",
    "val_recall_negative",
    "val_precision_negative",
    "val_f1_macro",
    "test_f1_negative",
    "test_recall_negative",
    "test_precision_negative",
    "test_f1_macro",
    "oot_f1_negative",
    "oot_recall_negative",
    "oot_precision_negative",
    "oot_f1_macro",
]

DISPLAY_METRICS = [
    "val_f1_negative",
    "test_f1_negative",
    "oot_f1_negative",
    "test_f1_macro",
    "test_recall_negative",
    "test_precision_negative",
]

REGISTERED_MODELS = {
    "sentiment-baseline": ("sentiment-tfidf-logreg", "logreg-final"),
    "sentiment-distilbert": ("sentiment-distilbert", "distilbert-final"),
}


def resolve_tracking_uri(root: Optional[Path] = None) -> str:
    """Try Docker MLflow (host :5001), then file store under repo root."""
    import requests

    root = root or Path.cwd()
    if not (root / "data").exists() and (root.parent / "data").exists():
        root = root.parent

    candidates = []
    if os.getenv("MLFLOW_TRACKING_URI"):
        candidates.append(os.environ["MLFLOW_TRACKING_URI"])
    candidates.extend(["http://localhost:5001", "http://127.0.0.1:5001"])

    seen: set[str] = set()
    for tracking_uri in candidates:
        if tracking_uri in seen or tracking_uri.startswith("file:"):
            continue
        seen.add(tracking_uri)
        try:
            resp = requests.get(tracking_uri, timeout=5)
            if resp.status_code < 500:
                return tracking_uri
        except Exception:
            continue

    fallback = f"file:{(root / 'mlruns').as_posix()}"
    print(
        "WARNING: Docker MLflow not reachable — falling back to local file store.\n"
        f"  tried: {', '.join(seen) or '(none)'}\n"
        f"  using: {fallback}\n"
        "  Start the server with `docker compose up -d mlflow` and re-run, or set\n"
        "  MLFLOW_TRACKING_URI=http://localhost:5001 before connecting."
    )
    return fallback


def _metric_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map short metric names to search_runs column names."""
    return {m: f"metrics.{m}" for m in METRIC_COLS if f"metrics.{m}" in df.columns}


def fetch_experiment_runs(
    experiment_name: str,
    allowed_run_names: Optional[set[str]] = None,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """Return one row per run name, keeping the row with the richest metrics."""
    import mlflow

    tracking_uri = tracking_uri or resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)

    raw = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"],
    )
    if raw.empty:
        return pd.DataFrame()

    name_col = "tags.mlflow.runName"
    if name_col not in raw.columns:
        return pd.DataFrame()

    if allowed_run_names:
        raw = raw[raw[name_col].isin(allowed_run_names)]

    metric_map = _metric_columns(raw)
    score_col = metric_map.get("test_f1_negative")
    if score_col:
        raw["_completeness"] = raw[list(metric_map.values())].notna().sum(axis=1)
        raw = raw.sort_values(["_completeness", score_col], ascending=[False, False])
    raw = raw.drop_duplicates(subset=[name_col], keep="first")

    rows = []
    for _, r in raw.iterrows():
        row = {
            "run_name": r[name_col],
            "run_id": r["run_id"],
            "experiment": experiment_name,
            "start_time": pd.to_datetime(r["start_time"], unit="ms", errors="coerce"),
        }
        for short, col in metric_map.items():
            row[short] = r.get(col)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty and "test_f1_negative" in out.columns:
        out = out.sort_values("test_f1_negative", ascending=False, na_position="last")
    return out.reset_index(drop=True)


def format_comparison_table(
    df: pd.DataFrame,
    metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Pretty table with rank column; NaN shown as em dash."""
    metrics = list(metrics or DISPLAY_METRICS)
    if df.empty:
        return df

    cols = ["run_name"] + [m for m in metrics if m in df.columns]
    display = df[cols].copy()
    display.index = [
        "* BEST" if i == 0 else f"  #{i + 1}"
        for i in range(len(display))
    ]
    for m in metrics:
        if m in display.columns and pd.api.types.is_numeric_dtype(display[m]):
            display[m] = display[m].map(
                lambda x: f"{x:.4f}" if pd.notna(x) else "-"
            )
    return display.rename(columns={"run_name": "Run"})


def fetch_registry_versions(
    model_name: str,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """List registered versions with stage and linked run metrics."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri or resolve_tracking_uri())
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return pd.DataFrame()

    rows = []
    for v in sorted(versions, key=lambda x: int(x.version)):
        run = client.get_run(v.run_id)
        m = run.data.metrics
        rows.append({
            "model": model_name,
            "version": int(v.version),
            "stage": v.current_stage or "None",
            "run_id": v.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "test_f1_negative": m.get("test_f1_negative"),
            "oot_f1_negative": m.get("oot_f1_negative"),
            "test_f1_macro": m.get("test_f1_macro"),
            "created": pd.to_datetime(v.creation_timestamp, unit="ms"),
        })
    return pd.DataFrame(rows)


def cross_family_leaderboard(
    logreg_df: pd.DataFrame,
    distilbert_df: pd.DataFrame,
) -> pd.DataFrame:
    """Rank final + best tuning runs across both model families."""
    frames = []
    for family, df in [("tfidf-logreg", logreg_df), ("distilbert", distilbert_df)]:
        if df.empty:
            continue
        subset = df.copy()
        subset["family"] = family
        frames.append(subset)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    # Prefer registered final runs when present; otherwise all runs with test scores.
    final_names = {"logreg-final", "distilbert-final"}
    with_test = combined[combined["test_f1_negative"].notna()]
    finals = with_test[with_test["run_name"].isin(final_names)]
    candidates = finals if not finals.empty else with_test
    return candidates.sort_values(
        "test_f1_negative", ascending=False, na_position="last"
    ).reset_index(drop=True)


@dataclass
class PromotionRecommendation:
    model_name: str
    version: int
    run_name: str
    test_f1_negative: float
    reason: str


def recommend_promotions(
    registry: dict[str, pd.DataFrame],
    leaderboard: pd.DataFrame,
    tie_margin: float = 0.01,
) -> list[PromotionRecommendation]:
    """Suggest which registry version to move to Staging per model family."""
    recs: list[PromotionRecommendation] = []

    if leaderboard.empty:
        return recs

    best = leaderboard.iloc[0]
    best_family = best["family"]
    best_score = best["test_f1_negative"]

    for model_name, (_, final_run) in REGISTERED_MODELS.items():
        versions = registry.get(model_name, pd.DataFrame())
        if versions.empty:
            continue

        final_versions = versions[versions["run_name"] == final_run]
        if final_versions.empty:
            # Fall back to latest version.
            pick = versions.sort_values("version", ascending=False).iloc[0]
        else:
            pick = final_versions.sort_values("version", ascending=False).iloc[0]

        family = "tfidf-logreg" if model_name == "sentiment-baseline" else "distilbert"
        if family == best_family:
            reason = (
                f"Highest test F1-negative ({best_score:.4f}) on the cross-family leaderboard. "
                "Promote to Staging for shadow deploy."
            )
        elif abs(pick["test_f1_negative"] - best_score) <= tie_margin:
            reason = (
                f"Within {tie_margin:.2f} of the leaderboard winner; keep registered as fallback baseline."
            )
        else:
            reason = (
                f"Registered for shadow comparison; leaderboard winner is {best_family} "
                f"({best_score:.4f} vs {pick['test_f1_negative']:.4f})."
            )

        recs.append(
            PromotionRecommendation(
                model_name=model_name,
                version=int(pick["version"]),
                run_name=str(pick["run_name"]),
                test_f1_negative=float(pick["test_f1_negative"])
                if pd.notna(pick["test_f1_negative"])
                else float("nan"),
                reason=reason,
            )
        )

    return recs


def main() -> None:
    uri = resolve_tracking_uri()
    print(f"MLflow tracking URI: {uri}\n")

    logreg = fetch_experiment_runs("sentiment-tfidf-logreg", LOGREG_RUNS, uri)
    distilbert = fetch_experiment_runs("sentiment-distilbert", DISTILBERT_RUNS, uri)

    print("=" * 72)
    print("TF-IDF + LogReg experiments (ranked by test F1-negative)")
    print("=" * 72)
    print(format_comparison_table(logreg).to_string())
    print()

    print("=" * 72)
    print("DistilBERT experiments (ranked by test F1-negative)")
    print("=" * 72)
    print(format_comparison_table(distilbert).to_string())
    print()

    leaderboard = cross_family_leaderboard(logreg, distilbert)
    print("=" * 72)
    print("Cross-family leaderboard (final runs)")
    print("=" * 72)
    if leaderboard.empty:
        print("(no runs with test_f1_negative)")
    else:
        lb = format_comparison_table(
            leaderboard, metrics=["family"] + DISPLAY_METRICS
        )
        print(lb.to_string())
    print()

    registry = {
        name: fetch_registry_versions(name, uri) for name in REGISTERED_MODELS
    }
    print("=" * 72)
    print("MLflow Model Registry")
    print("=" * 72)
    for name, df in registry.items():
        print(f"\n--- {name} ---")
        if df.empty:
            print("  (no versions)")
            continue
        show = df[
            ["version", "stage", "run_name", "test_f1_negative", "oot_f1_negative", "created"]
        ].copy()
        for c in ("test_f1_negative", "oot_f1_negative"):
            show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        print(show.to_string(index=False))

    print()
    print("=" * 72)
    print("Promotion recommendations")
    print("=" * 72)
    for rec in recommend_promotions(registry, leaderboard):
        print(f"\n{rec.model_name} v{rec.version} ({rec.run_name})")
        print(f"  test F1-neg: {rec.test_f1_negative:.4f}")
        print(f"  -> {rec.reason}")


if __name__ == "__main__":
    main()
