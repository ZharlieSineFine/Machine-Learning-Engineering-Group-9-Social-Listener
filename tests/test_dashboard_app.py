"""Headless smoke test for the Streamlit app.

Uses streamlit.testing.v1.AppTest to render the whole script in-process.

One test that asserts everything (instead of three) because `AppTest.run()`
takes 30+s — re-running it per assertion blows the suite budget.
"""
from __future__ import annotations

from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renders_with_sections_and_tiles(monkeypatch):
    # Force CSV fallback / no MLflow / no MinIO.
    for var in [
        "POSTGRES_HOST", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB",
        "MLFLOW_TRACKING_URI", "MLFLOW_S3_ENDPOINT_URL",
    ]:
        monkeypatch.delenv(var, raising=False)

    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(APP_PATH))
    at.run(timeout=60)

    # 1. No exception bubbled up to Streamlit.
    assert not at.exception, at.exception

    # 2. All four Phase 2 sections rendered.
    headers = [h.value for h in at.subheader]
    for expected in [
        "Sentiment over time",
        "Top words in negative reviews",
        "Model A/B — recent MLflow runs",
        "Latest drift report",
    ]:
        assert expected in headers, f"missing section: {expected!r} (got {headers})"

    # 3. Total reviews tile shows a non-zero count from the CSV fallback.
    totals = [m for m in at.metric if m.label == "Total reviews"]
    assert totals, "expected a Total reviews metric"
    value = totals[0].value.replace(",", "")
    assert int(value) > 0
