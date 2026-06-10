"""Embedding helpers for the Gold layer.

Phase 1 stub: returns fixed-size random vectors so `build_gold` can wire
through without loading sentence-transformers. Phase 2 swaps in MiniLM-L6-v2
(same 384-d output) for real semantic embeddings.

Owner: Van (Modeler).
"""
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

# Matches sentence-transformers/all-MiniLM-L6-v2 — keeps the Gold schema
# stable when we replace the stub in Phase 2.
EMBEDDING_DIM = 384


def embed(texts: Union[Sequence[str], Iterable[str]]) -> np.ndarray:
    """Return an (n, EMBEDDING_DIM) float32 array of stub embeddings.

    Deterministic per text (hash-seeded) so the same review always gets the
    same vector — handy for local debugging. Not semantically meaningful
    until Phase 2.
    """
    rows = [str(t) for t in texts]
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    return np.stack([_vector_for_text(t) for t in rows])


def _vector_for_text(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
