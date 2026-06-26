#Unit tests for the replay -> drift wiring in monitoring/drift_checks.py.

from __future__ import annotations

import pandas as pd
import pytest

import monitoring.drift_checks as d


def _write_replay(root, scenario, by_date):
    base = root / scenario
    for day, rows in by_date.items():
        pdir = base / f"review_date={day}"
        pdir.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(pdir / "part.parquet", index=False)


def test_load_replay_window_selects_asof_and_recent(tmp_path):
    _write_replay(
        tmp_path,
        "spike",
        {
            "2026-06-19": {"text": ["a"], "label": ["positive"], "review_date": ["2026-06-19"]},
            "2026-06-20": {"text": ["b"], "label": ["positive"], "review_date": ["2026-06-20"]},
            "2026-06-21": {"text": ["c"], "label": ["negative"], "review_date": ["2026-06-21"]},
            "2026-06-22": {"text": ["d"], "label": ["positive"], "review_date": ["2026-06-22"]},
        },
    )
    # asof keeps partitions <= cutoff; n_recent keeps the last of those
    df = d.load_replay_window("spike", replay_root=tmp_path, asof="2026-06-21", n_recent=1)
    assert list(df["review_date"]) == ["2026-06-21"]
    assert list(df["label"]) == ["negative"]
    # asof without n_recent keeps everything up to the cutoff (drops 06-22)
    upto = d.load_replay_window("spike", replay_root=tmp_path, asof="2026-06-21")
    assert len(upto) == 3
    # no filters -> all four partitions
    assert len(d.load_replay_window("spike", replay_root=tmp_path)) == 4


def test_load_replay_window_missing_scenario_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        d.load_replay_window("nope", replay_root=tmp_path)


def test_build_replay_frames_keeps_text_and_label(tmp_path):
    _write_replay(
        tmp_path,
        "spike",
        {
            "2026-06-21": {
                "text": ["c", "e"],
                "label": ["negative", "negative"],
                "review_date": ["2026-06-21", "2026-06-21"],
            }
        },
    )
    ref_csv = tmp_path / "holdout.csv"
    pd.DataFrame(
        {"text": ["x", "y"], "label": ["positive", "neutral"], "review_date": ["2022-01-01", "2022-01-01"]}
    ).to_csv(ref_csv, index=False)
    ref, cur = d.build_replay_frames(
        "spike", reference_csv=ref_csv, replay_root=tmp_path, asof="2026-06-21", n_recent=1
    )
    # text_len dropped (artifact for re-dated identical reviews); monitor the label
    assert set(ref.columns) == {"text", "label"}
    assert set(cur.columns) == {"text", "label"}
    assert len(cur) == 2 and (cur["label"] == "negative").all()


def test_run_replay_monitor_spike_blocks_stable_passes(tmp_path):
    pytest.importorskip("evidently")
    import numpy as np

    rng = np.random.default_rng(0)

    def frame(neg_frac, n, day):
        labels = rng.choice(["negative", "positive"], size=n, p=[neg_frac, 1 - neg_frac])
        return {"text": [f"review {i}" for i in range(n)], "label": list(labels), "review_date": [day] * n}

    _write_replay(tmp_path, "spike", {"2026-06-21": frame(0.6, 120, "2026-06-21")})
    _write_replay(tmp_path, "stable", {"2026-06-21": frame(0.2, 120, "2026-06-21")})
    ref_csv = tmp_path / "holdout.csv"
    pd.DataFrame(frame(0.2, 400, "2022-01-01")).to_csv(ref_csv, index=False)

    common = dict(reference_csv=ref_csv, replay_root=tmp_path, asof="2026-06-21", n_recent=1, report_dir=tmp_path / "rep")
    spike = d.run_replay_monitor("spike", **common)
    stable = d.run_replay_monitor("stable", **common)

    assert spike.evidently_ran and stable.evidently_ran
    assert spike.blocked is True, f"spike should trip the gate: {spike}"
    assert stable.blocked is False, f"stable should pass: {stable}"
