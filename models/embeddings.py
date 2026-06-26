#Embedding helpers for the Gold layer
from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np

#Matches sentence-transformers/all-MiniLM-L6-v2 so the Gold schema stays stable when the stub is replaced in Phase 2.
EMBEDDING_DIM = 384


def embed(texts: Union[Sequence[str], Iterable[str]]) -> np.ndarray:
    #Return an (n, EMBEDDING_DIM) float32 array of stub embeddings (deterministic per text, hash-seeded).
    rows = [str(t) for t in texts]
    if not rows:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    return np.stack([_vector_for_text(t) for t in rows])


def _vector_for_text(text: str) -> np.ndarray:
    rng = np.random.default_rng(hash(text) & 0xFFFFFFFF)
    return rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
