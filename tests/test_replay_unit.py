"""Unit tests for the replay simulator (data/ingest/replay.py).

The simulator replays a pre-built demo_data scenario (stable / spike) as a BrewLeaf
operational stream. Pure cores only — no demo CSVs on disk required (tests write their own).
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.ingest.replay import (
    BRAND,
    REPLAY_COLUMNS,
    SCENARIOS,
    build_replay_stream,
    daily_negative_fraction,
    demo_csv_path,
    emit_replay,
    load_demo,
)


def _demo(rows):
    return pd.DataFrame(rows, columns=["text", "label", "review_date"])


def test_build_replay_stream_shapes_tags_and_sorts():
    demo = _demo([
        ["spike day review", "negative", "2026-06-21"],
        ["normal day review", "positive", "2026-06-07"],
    ])
    out = build_replay_stream(demo, "spike")
    assert list(out.columns) == REPLAY_COLUMNS
    assert (out["source"] == "replay").all()
    assert (out["scenario"] == "spike").all()
    assert (out["brand"] == BRAND).all()
    # sorted by review_date (2026-06-07 first)
    assert out.iloc[0]["review_date"] == "2026-06-07"
    assert out.iloc[0]["text"] == "normal day review"  # text verbatim (no rebrand)


def test_build_replay_stream_requires_contract_columns():
    with pytest.raises(ValueError):
        build_replay_stream(pd.DataFrame({"text": ["hi"], "label": ["positive"]}), "spike")


def test_review_id_is_deterministic():
    demo = _demo([["same text", "positive", "2026-06-07"]])
    a = build_replay_stream(demo, "spike").iloc[0]["review_id"]
    b = build_replay_stream(demo, "spike").iloc[0]["review_id"]
    assert a == b
    # scenario is part of the id, so the same review under a different scenario differs
    c = build_replay_stream(demo, "stable").iloc[0]["review_id"]
    assert a != c


def test_daily_negative_fraction_detects_spike():
    demo = _demo([
        ["a", "positive", "2026-06-07"],
        ["b", "negative", "2026-06-07"],   # 1/2 = 0.5 on the 7th
        ["c", "negative", "2026-06-21"],
        ["d", "negative", "2026-06-21"],   # 2/2 = 1.0 on the 21st (spike)
    ])
    frac = daily_negative_fraction(build_replay_stream(demo, "spike"))
    assert frac["2026-06-07"] == 0.5
    assert frac["2026-06-21"] == 1.0
    assert frac.idxmax() == "2026-06-21"


def test_load_demo_reads_scenario(tmp_path):
    path = tmp_path / SCENARIOS["spike"]
    _demo([["x", "positive", "2026-06-07"]]).to_csv(path, index=False)
    df = load_demo("spike", tmp_path)
    assert list(df.columns) == ["text", "label", "review_date"]
    assert demo_csv_path("spike", tmp_path) == path


def test_load_demo_unknown_scenario_raises(tmp_path):
    with pytest.raises(ValueError):
        load_demo("nope", tmp_path)


def test_emit_replay_writes_partitions(tmp_path):
    stream = build_replay_stream(
        _demo([
            ["good day", "positive", "2026-06-07"],
            ["bad day", "negative", "2026-06-21"],
        ]),
        "spike",
    )
    n = emit_replay(stream, tmp_path, speed=0.0)
    assert n == 2
    p = tmp_path / "review_date=2026-06-21" / "part.parquet"
    assert p.exists()
    back = pd.read_parquet(p)
    assert back.iloc[0]["label"] == "negative"
    assert back.iloc[0]["scenario"] == "spike"
