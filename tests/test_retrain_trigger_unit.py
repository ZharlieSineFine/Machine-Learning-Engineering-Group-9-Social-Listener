"""Unit tests for monitoring/retrain_trigger.py — no live Airflow needed.

The HTTP call is injected (``opener=``) so the decision logic is exercised
without a running Airflow REST API.
"""
from __future__ import annotations

import urllib.error

import monitoring.retrain_trigger as rt


def test_build_payload_unique_id_and_reason():
    a = rt.build_dag_run_payload("drift=0.6", {"scenario": "spike"})
    b = rt.build_dag_run_payload("drift=0.6")
    assert a["conf"]["reason"] == "drift=0.6"
    assert a["conf"]["scenario"] == "spike"
    assert a["dag_run_id"].startswith("retrain__")
    assert a["dag_run_id"] != b["dag_run_id"]  # unique per call (no 409 clashes)


def test_trigger_noop_when_unconfigured(monkeypatch):
    monkeypatch.delenv("AIRFLOW_API_USERNAME", raising=False)
    monkeypatch.delenv("AIRFLOW_API_PASSWORD", raising=False)
    r = rt.trigger_retrain("t", api_url="http://x:8080")
    assert r["triggered"] is False
    assert r["status"] == "skipped_unconfigured"


def test_trigger_posts_with_basic_auth():
    seen = {}

    class FakeResp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"dag_run_id":"x"}'

    def fake_open(req, timeout=None):
        seen["url"] = req.full_url
        seen["method"] = req.method
        seen["auth"] = req.headers.get("Authorization")
        return FakeResp()

    r = rt.trigger_retrain(
        "drift",
        dag_id="medallion_train_cycle",
        api_url="http://air:8080",
        username="airflow",
        password="airflow",
        opener=fake_open,
    )
    assert r["triggered"] is True and r["status"] == 200
    assert seen["url"].endswith("/api/v1/dags/medallion_train_cycle/dagRuns")
    assert seen["method"] == "POST"
    assert seen["auth"].startswith("Basic ")


def test_trigger_handles_http_error():
    def boom(req, timeout=None):
        raise urllib.error.URLError("down")

    r = rt.trigger_retrain(
        "t", api_url="http://x", username="u", password="p", opener=boom
    )
    assert r["triggered"] is False and r["status"] == "error"


def test_mark_triggered_retrain_with_fake_conn():
    executed = []

    class Cur:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params=None):
            executed.append(sql)

    class Conn:
        def cursor(self):
            return Cur()

        def commit(self):
            executed.append("COMMIT")

        def rollback(self):
            executed.append("ROLLBACK")

    ok = rt.mark_triggered_retrain(Conn(), report_id=5)
    assert ok is True
    assert any("ADD COLUMN IF NOT EXISTS triggered_retrain" in s for s in executed)
    assert any("UPDATE monitoring_reports" in s for s in executed)
    assert "COMMIT" in executed
