"""FAISS wrapper with a NumPy fallback for environments without FAISS."""

from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


class InnerProductIndex:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors: np.ndarray | None = None
        self._index = faiss.IndexFlatIP(dimension) if faiss is not None else None

    @property
    def size(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (n, dimension)")
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self._index is not None:
            return self._index.search(queries, k)
        if self._vectors is None:
            raise RuntimeError("index is empty")
        queries = np.asarray(queries, dtype="float32")
        scores = queries @ self._vectors.T
        order = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, order, axis=1), order
