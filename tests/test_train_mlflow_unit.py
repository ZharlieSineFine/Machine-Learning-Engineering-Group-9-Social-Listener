"""Unit tests for the MLflow logging path in models/train.py.

We mock `mlflow` and `mlflow.sklearn` to verify the right calls happen with
the right shape, without needing a tracking server.
"""
from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd

from models.baseline_sklearn import train
from models.train import _log_to_mlflow, run


def _tiny_df() -> pd.DataFrame:
    rows = []
    for i in range(30):
        rows.append({
            "text": f"good review {i}" if i % 3 == 0 else f"bad food {i}",
            "label": ["positive", "negative", "neutral"][i % 3],
        })
    return pd.DataFrame(rows)


def test_log_to_mlflow_calls_register_with_model_name(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake:5000")
    monkeypatch.setenv("MODEL_NAME", "sentiment-baseline")

    pipe, metrics = train(_tiny_df())

    mock_mlflow = MagicMock()
    mock_sklearn = MagicMock()
    # `mlflow.sklearn.log_model(...)` accesses mlflow.sklearn as an attribute,
    # so they must be the same object — sys.modules["mlflow.sklearn"] is
    # only consulted by the `import mlflow.sklearn` statement itself.
    mock_mlflow.sklearn = mock_sklearn
    info = MagicMock()
    info.registered_model_version = 3
    mock_sklearn.log_model.return_value = info
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "abc123"

    with patch.dict(
        "sys.modules", {"mlflow": mock_mlflow, "mlflow.sklearn": mock_sklearn}
    ):
        run_id, version = _log_to_mlflow(pipe, metrics)

    assert run_id == "abc123"
    assert version == "3"
    mock_mlflow.set_tracking_uri.assert_called_once_with("http://fake:5000")
    mock_mlflow.set_experiment.assert_called_once()
    mock_mlflow.log_metrics.assert_called_once()
    logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
    for key in (
        "f1_macro", "f1_weighted", "accuracy",
        "f1_neg", "precision_neg", "recall_neg",
    ):
        assert key in logged_metrics
    mock_sklearn.log_model.assert_called_once()
    kwargs = mock_sklearn.log_model.call_args.kwargs
    assert kwargs["registered_model_name"] == "sentiment-baseline"
    assert kwargs["artifact_path"] == "model"


def test_run_skips_mlflow_when_uri_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    csv = tmp_path / "data.csv"
    _tiny_df().to_csv(csv, index=False)
    out = tmp_path / "model.pkl"

    result = run(data_path=csv, out_path=out)
    assert result.mlflow_run_id is None
    assert result.mlflow_model_version is None
    assert out.exists()


def test_run_propagates_mlflow_failure_when_uri_set(tmp_path, monkeypatch):
    """If logging is configured but the server is unreachable, training fails."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://does-not-exist.invalid:9999")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")  # don't hang on retries

    csv = tmp_path / "data.csv"
    _tiny_df().to_csv(csv, index=False)
    out = tmp_path / "model.pkl"

    import pytest
    with pytest.raises(Exception):
        run(data_path=csv, out_path=out)
