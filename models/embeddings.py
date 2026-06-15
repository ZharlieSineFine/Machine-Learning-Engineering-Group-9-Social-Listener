"""Embedding helpers for the Gold layer.

Uses ``sentence-transformers/all-MiniLM-L6-v2`` (384-d) for semantic embeddings
stored in ``reviews_gold.embedding`` by the ``build_gold`` DAG.

This module is consumed by the Gold/build_gold DAG (Charlie/Ha), not by
sentiment training (`train.py`, `baseline_sklearn.py`, `distilbert_finetune.py`).
See `models/README.md` for the handoff contract.

Environment:
    EMBEDDING_MODEL   — HuggingFace model id (default: all-MiniLM-L6-v2)
    EMBEDDING_STUB=1  — return deterministic hash-seeded vectors (tests/CI)

Owner: Van (Modeler). Wired by: Charlie/Ha (`build_gold` DAG).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, Sequence, Union

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64


def _use_stub() -> bool:
    return os.getenv("EMBEDDING_STUB", "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer

    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
    return SentenceTransformer(model_name)


def embed(
    texts: Union[Sequence[str], Iterable[str]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Return an (n, EMBEDDING_DIM) float32 array of embeddings.

    Empty input returns shape ``(0, EMBEDDING_DIM)``.
    Set ``EMBEDDING_STUB=1`` for fast deterministic vectors without downloading weights.
    """
    rows = [str(t) for t in texts]
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    if _use_stub():
        return np.stack([_stub_vector_for_text(t) for t in rows])

    model = _load_model()
    vectors = model.encode(
        rows,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    out = np.asarray(vectors, dtype=np.float32)
    if out.ndim != 2 or out.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"expected embeddings shape (n, {EMBEDDING_DIM}), got {out.shape}"
        )
    return out


def _stub_vector_for_text(text: str) -> np.ndarray:
    """Deterministic stub — same text always maps to the same vector."""
    rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)


def clear_model_cache() -> None:
    """Drop the cached SentenceTransformer (useful in tests)."""
    _load_model.cache_clear()
