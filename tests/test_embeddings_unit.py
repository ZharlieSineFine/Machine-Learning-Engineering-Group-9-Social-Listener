"""Unit tests for models/embeddings.py."""
from __future__ import annotations

import numpy as np
import pytest

from models.embeddings import (
    DEFAULT_MODEL,
    EMBEDDING_DIM,
    clear_model_cache,
    embed,
)


@pytest.fixture(autouse=True)
def _reset_model_cache():
    clear_model_cache()
    yield
    clear_model_cache()


def test_embed_empty():
    out = embed([])
    assert out.shape == (0, EMBEDDING_DIM)
    assert out.dtype == np.float32


def test_embed_stub_shape_and_dim(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STUB", "1")
    out = embed(["great food", "terrible service", "okay"])
    assert out.shape == (3, EMBEDDING_DIM)
    assert out.dtype == np.float32


def test_embed_stub_is_deterministic(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STUB", "1")
    a = embed(["same text"])
    b = embed(["same text"])
    np.testing.assert_array_equal(a, b)


def test_embed_stub_differs_for_different_text(monkeypatch):
    monkeypatch.setenv("EMBEDDING_STUB", "1")
    a = embed(["text one"])
    b = embed(["text two"])
    assert not np.allclose(a, b)


@pytest.mark.slow
def test_embed_minilm_real_vectors(monkeypatch):
    """Downloads MiniLM on first run (~80 MB); skipped unless RUN_SLOW=1."""
    monkeypatch.delenv("EMBEDDING_STUB", raising=False)
    out = embed(["amazing meal", "worst service ever"])
    assert out.shape == (2, EMBEDDING_DIM)
    # Semantically similar food reviews should be closer than unrelated pairs.
    sim_close = np.dot(out[0], out[1]) / (np.linalg.norm(out[0]) * np.linalg.norm(out[1]))
    unrelated = embed(["quantum physics paper"])
    sim_far = np.dot(out[0], unrelated[0]) / (
        np.linalg.norm(out[0]) * np.linalg.norm(unrelated[0])
    )
    assert sim_close > sim_far


def test_default_model_constant():
    assert "MiniLM" in DEFAULT_MODEL
