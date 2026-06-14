"""Unit tests for models.inference and batch scoring helpers."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from models.baseline_sklearn import LABELS, train
from models.inference import (
    SentimentModel,
    load_from_pickle,
    predict_labels,
    predict_with_scores,
    prepare_texts,
)
from models.batch_score import filter_unscored, _score_with_model

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def trained_model():
    df = pd.read_csv(SAMPLE_CSV)
    model, _ = train(df)
    return model


def test_prepare_texts_lowercases():
    assert prepare_texts(["Hello WORLD!"]) == ["hello world"]


def test_predict_labels_match_training_labels(trained_model):
    labels = predict_labels(trained_model, [
        "amazing food and lovely staff",
        "worst meal ever do not recommend",
    ])
    assert all(lbl in LABELS for lbl in labels)


def test_predict_with_scores_returns_scores(trained_model):
    results = predict_with_scores(trained_model, ["great experience"])
    assert len(results) == 1
    assert results[0].label in LABELS
    assert results[0].score is not None
    assert 0.0 <= results[0].score <= 1.0


def test_load_from_pickle_roundtrip(tmp_path, trained_model):
    import pickle

    path = tmp_path / "model.pkl"
    with open(path, "wb") as f:
        pickle.dump(trained_model, f)

    loaded = load_from_pickle(path)
    assert loaded.source == "pickle"
    assert predict_labels(loaded.model, ["okay food"]) == predict_labels(
        trained_model, ["okay food"]
    )


def test_filter_unscored_excludes_existing_ids():
    reviews = pd.DataFrame({"review_id": [1, 2, 3], "text": ["a", "b", "c"]})
    conn = MagicMock()
    cursor = MagicMock()
    cursor.fetchall.return_value = [(1,), (3,)]
    conn.cursor.return_value.__enter__.return_value = cursor

    pending = filter_unscored(reviews, conn, ["sentiment-baseline"])
    assert pending["review_id"].tolist() == [2]


def test_score_with_model_writes_rows(trained_model):
    reviews = pd.DataFrame({
        "review_id": [10, 11],
        "text": ["loved it", "hated it"],
    })
    lane = SentimentModel(
        model=trained_model,
        model_name="sentiment-baseline",
        model_version="1",
        stage="Production",
        source="pickle",
    )
    rows = _score_with_model(lane, reviews, batch_size=8)
    assert len(rows) == 2
    assert rows[0].review_id == 10
    assert rows[0].stage == "Production"
    assert rows[0].predicted_label in LABELS
