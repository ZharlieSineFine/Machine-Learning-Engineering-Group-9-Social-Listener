"""Populate the API's shadow-deploy log so the MLOps-monitor 'Production vs Staging'
panel has data. POSTs a sample of demo reviews to the API's /predict/batch, which
scores both the Production and Staging models and records the pair.

The shadow log is in-memory, so run this after each API (re)start.

Run: python scripts/seed_shadow.py        (API at http://localhost:8000)
"""
from __future__ import annotations

import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_URL = os.getenv("API_URL", "http://localhost:8000")
DEMO_CSV = Path(os.getenv("SHADOW_CSV", str(ROOT / "data" / "demo" / "demo_jun2026_spike.csv")))
N = int(os.getenv("SHADOW_SAMPLE", "60"))  # API caps batch at 256


def main() -> int:
    texts: list[str] = []
    with open(DEMO_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
            if len(texts) >= N:
                break
    if not texts:
        print(f"[seed_shadow] no texts found in {DEMO_CSV}")
        return 1

    body = json.dumps({"texts": texts}).encode("utf-8")
    req = urllib.request.Request(
        f"{API_URL}/predict/batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted localhost)
            out = json.loads(resp.read())
        print(
            f"[seed_shadow] sent {len(texts)} reviews -> {len(out.get('labels', []))} "
            f"predictions; shadow log populated (see {API_URL}/shadow/log)"
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[seed_shadow] failed ({exc}); is the API up at {API_URL}?")
        return 1


if __name__ == "__main__":
    sys.exit(main())
