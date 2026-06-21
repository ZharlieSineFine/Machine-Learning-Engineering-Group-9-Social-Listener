"""Retrain trigger — fire ``medallion_train_cycle`` via the Airflow REST API.

``evaluate_and_monitor`` already chains ``should_retrain`` (ShortCircuit) ->
``trigger_retrain`` (a ``TriggerDagRunOperator``) to kick a retrain *inside*
Airflow when drift blocks the gate. This module is the documented external/CLI
equivalent (WORKFLOW.md, task 2): given a blocking drift result, POST a DAG run
to the Airflow REST API.

It is **env-gated** — a no-op (``triggered=False``) when the API isn't
configured — and never raises on a transient HTTP error, so a monitor that calls
it never dies just because the trigger is unavailable. The HTTP call is injected
(``opener=``) so the decision logic is unit-testable without a live Airflow.

CLI:
    python -m monitoring.retrain_trigger --reason "drift_score=0.62"

Env:
    AIRFLOW_API_URL        base URL, e.g. http://airflow-webserver:8080
    AIRFLOW_API_USERNAME   basic-auth user (Airflow default: airflow)
    AIRFLOW_API_PASSWORD   basic-auth password
    RETRAIN_DAG_ID         DAG to trigger (default: medallion_train_cycle)

Owner: Charlie + Ha (Monitoring).
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

DEFAULT_RETRAIN_DAG_ID = "medallion_train_cycle"
DEFAULT_API_URL = "http://airflow-webserver:8080"


def _retrain_dag_id() -> str:
    return os.getenv("RETRAIN_DAG_ID", DEFAULT_RETRAIN_DAG_ID)


def _basic_auth_header(user: str, password: str) -> str:
    import base64

    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def build_dag_run_payload(
    reason: str,
    conf: Optional[dict] = None,
    run_id: Optional[str] = None,
) -> dict:
    """Shape the Airflow ``POST /dagRuns`` body. Pure + unit-testable.

    A unique ``dag_run_id`` avoids 409 clashes when drift fires twice in a window;
    ``conf.reason`` lands in the triggered run's context for traceability.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return {
        "dag_run_id": run_id or f"retrain__{stamp}__{uuid.uuid4().hex[:8]}",
        "conf": {"reason": reason, "source": "retrain_trigger", **(conf or {})},
    }


def trigger_retrain(
    reason: str = "drift detected",
    *,
    dag_id: Optional[str] = None,
    api_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    conf: Optional[dict] = None,
    opener=None,
    timeout: float = 30.0,
) -> dict:
    """POST a DAG run to Airflow REST to retrain. Env-gated no-op when unconfigured.

    Returns ``{triggered, dag_id, dag_run_id, status, detail}``. Never raises on a
    config miss or transient HTTP error — a monitor shouldn't die because the
    trigger is down; it logs and moves on.
    """
    dag_id = dag_id or _retrain_dag_id()
    api_url = (api_url or os.getenv("AIRFLOW_API_URL") or DEFAULT_API_URL).rstrip("/")
    username = username or os.getenv("AIRFLOW_API_USERNAME")
    password = password or os.getenv("AIRFLOW_API_PASSWORD")

    payload = build_dag_run_payload(reason, conf)
    result = {
        "triggered": False,
        "dag_id": dag_id,
        "dag_run_id": payload["dag_run_id"],
        "status": None,
        "detail": None,
    }

    if not username or not password:
        print(
            "[retrain_trigger] AIRFLOW_API_USERNAME/PASSWORD unset — "
            "skipping trigger (no-op)"
        )
        result["status"] = "skipped_unconfigured"
        return result

    import urllib.error
    import urllib.request

    url = f"{api_url}/api/v1/dags/{dag_id}/dagRuns"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": _basic_auth_header(username, password),
        },
        method="POST",
    )
    _open = opener or urllib.request.urlopen
    try:
        with _open(req, timeout=timeout) as resp:  # noqa: S310 (trusted internal URL)
            status = int(getattr(resp, "status", None) or resp.getcode())
            result["detail"] = resp.read().decode("utf-8", "replace")
        result["status"] = status
        result["triggered"] = 200 <= status < 300
        print(
            f"[retrain_trigger] POST {url} -> {status} "
            f"(dag_run_id={payload['dag_run_id']})"
        )
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(f"[retrain_trigger] trigger failed ({exc}); not retraining this cycle")
        result["status"] = "error"
        result["detail"] = str(exc)
    return result


def mark_triggered_retrain(conn, report_id: Optional[int] = None) -> bool:
    """Best-effort: set ``triggered_retrain=true`` on a ``monitoring_reports`` row.

    Adds the column if the deployed schema predates it (idempotent ALTER), then
    updates the given report (or the most recent row). Returns True on update.
    Guarded so a monitor never dies on a schema/permission hiccup.
    """
    try:
        with conn.cursor() as cur:
            cur.execute(
                "ALTER TABLE monitoring_reports "
                "ADD COLUMN IF NOT EXISTS triggered_retrain BOOLEAN NOT NULL DEFAULT FALSE"
            )
            if report_id is not None:
                cur.execute(
                    "UPDATE monitoring_reports SET triggered_retrain = TRUE WHERE id = %s",
                    (report_id,),
                )
            else:
                cur.execute(
                    "UPDATE monitoring_reports SET triggered_retrain = TRUE "
                    "WHERE id = (SELECT id FROM monitoring_reports ORDER BY id DESC LIMIT 1)"
                )
        conn.commit()
        return True
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[retrain_trigger] could not mark triggered_retrain: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Trigger the retrain DAG via the Airflow REST API."
    )
    ap.add_argument("--reason", default="manual trigger")
    ap.add_argument("--dag-id", default=None)
    args = ap.parse_args()

    result = trigger_retrain(args.reason, dag_id=args.dag_id)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
