"""End-to-end drift demo (task 4): clean vs poisoned window through the gate.

Flow:
  1. Generate the replay streams (stable + spike) from the demo windows.
  2. Run the drift gate for each scenario's spike-day batch vs the holdout baseline.
  3. stable -> passes; spike -> blocked -> fire the retrain trigger (a no-op when no
     Airflow REST API is configured, so the demo runs anywhere).

This is the closed loop from WORKFLOW.md Phase 2/3: replay a clean window, replay a
poisoned window, Evidently fires, retraining is triggered.

Run (needs Evidently; e.g. the monitoring venv):
    python scripts/drift_demo.py
    python scripts/drift_demo.py --asof 2026-06-21 --n-recent 1

Owner: Charlie + Ha (Data & Eval).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.ingest.replay import (  # noqa: E402
    DEFAULT_OUT_ROOT,
    build_replay_stream,
    emit_replay,
    load_demo,
)
from monitoring.drift_checks import run_replay_monitor  # noqa: E402
from monitoring.retrain_trigger import trigger_retrain  # noqa: E402


def _generate(scenario: str) -> tuple[Path, int]:
    """Materialise one replay scenario's partitions from its demo window."""
    stream = build_replay_stream(load_demo(scenario), scenario)
    out = DEFAULT_OUT_ROOT / scenario
    return out, emit_replay(stream, out)


def main() -> int:
    ap = argparse.ArgumentParser(description="End-to-end replay drift demo.")
    ap.add_argument("--asof", default="2026-06-21", help="Current window ends on this date.")
    ap.add_argument("--n-recent", type=int, default=1, help="Recent partitions as the current batch.")
    args = ap.parse_args()

    print("=" * 68)
    print("DRIFT DEMO  -  clean (stable) vs poisoned (spike) window")
    print("=" * 68)

    results = {}
    for scenario in ("stable", "spike"):
        out, n = _generate(scenario)
        mon = run_replay_monitor(scenario, asof=args.asof, n_recent=args.n_recent)
        results[scenario] = mon
        verdict = "BLOCKED" if mon.blocked else "passed"
        print(
            f"\n[{scenario:6}] {n} partitions -> {out.name}/  | "
            f"current={mon.n_current} vs holdout={mon.n_reference}\n"
            f"          data_drift={mon.data_drift_score:.2f} "
            f"target_drift={mon.target_drift} "
            f"(score={mon.target_drift_score}) -> gate {verdict}"
        )

    print("\n" + "-" * 68)
    spike = results["spike"]
    if spike.blocked:
        print("Spike tripped the gate -> firing retrain trigger ...")
        r = trigger_retrain(
            reason=f"replay spike drift on {args.asof}",
            conf={"scenario": "spike", "asof": args.asof},
        )
        print(
            f"retrain_trigger: triggered={r['triggered']} status={r['status']} "
            f"dag={r['dag_id']} run_id={r['dag_run_id']}"
        )
    else:
        print("Spike did NOT trip the gate (unexpected for this demo).")

    ok = (not results["stable"].blocked) and spike.blocked
    print("\nDEMO RESULT:", "PASS (stable clean, spike blocked)" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
