"""FAISS wrapper with a NumPy fallback for environments without FAISS.

Supports flat inner-product, flat L2 and IVF-flat inner-product indexes.
Provides helpers to save/load an index to disk and to batch-encode
passages through a DPR passage encoder.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None


# ---------------------------------------------------------------------------
# Flat inner-product index
# ---------------------------------------------------------------------------

class InnerProductIndex:
    """FAISS IndexFlatIP (or NumPy fallback) for maximum inner-product search."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors: np.ndarray | None = None
        self._index = faiss.IndexFlatIP(dimension) if faiss is not None else None

    @property
    def size(self) -> int:
        """Return the number of vectors stored in the index."""
        if self._index is not None:
            return int(self._index.ntotal)
        return 0 if self._vectors is None else int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        """Add vectors to the index.

        Parameters
        ----------
        vectors : np.ndarray
            2-D array of shape ``(n, dimension)`` with dtype ``float32``.
        """
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (n, dimension)")
        if self._index is not None:
            self._index.add(vectors)
        else:
            self._vectors = (
                vectors if self._vectors is None
                else np.vstack([self._vectors, vectors])
            )

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k results for each query.

        Returns
        -------
        scores : np.ndarray of shape ``(n_queries, k)``
        indices : np.ndarray of shape ``(n_queries, k)``
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if self._index is not None:
            return self._index.search(
                np.asarray(queries, dtype="float32"), k
            )
        if self._vectors is None:
            raise RuntimeError("index is empty")
        queries = np.asarray(queries, dtype="float32")
        scores = queries @ self._vectors.T
        # Clamp k to the number of indexed vectors to avoid out-of-bounds.
        actual_k = min(k, self._vectors.shape[0])
        order = np.argsort(-scores, axis=1)[:, :actual_k]
        top_scores = np.take_along_axis(scores, order, axis=1)
        # Pad with -inf / -1 when fewer vectors than k.
        if actual_k < k:
            pad_width = k - actual_k
            top_scores = np.pad(
                top_scores,
                ((0, 0), (0, pad_width)),
                constant_values=-np.inf,
            )
            order = np.pad(
                order,
                ((0, 0), (0, pad_width)),
                constant_values=-1,
            )
        return top_scores, order

    # ---- Persistence --------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index to disk.

        When FAISS is available the native ``write_index`` is used; otherwise
        vectors are saved with ``np.save``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            faiss.write_index(self._index, str(path))
        elif self._vectors is not None:
            np.save(str(path), self._vectors)
        else:
            raise RuntimeError("index is empty – nothing to save")

    def load(self, path: str | Path) -> None:
        """Load vectors from a previously saved index file.

        Automatically detects whether the file was saved by FAISS or NumPy.
        """
        path = Path(path)
        if not path.exists():
            # Try .npy extension for NumPy fallback files.
            npy_path = path.with_suffix(".npy")
            if npy_path.exists():
                path = npy_path
            else:
                raise FileNotFoundError(f"index file not found: {path}")

        if faiss is not None and path.suffix != ".npy":
            self._index = faiss.read_index(str(path))
            self.dimension = self._index.d
        else:
            self._vectors = np.load(str(path)).astype("float32")
            self.dimension = self._vectors.shape[1]


# ---------------------------------------------------------------------------
# Flat L2 index
# ---------------------------------------------------------------------------

class L2Index:
    """FAISS IndexFlatL2 (or NumPy fallback) for Euclidean distance search."""

    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors: np.ndarray | None = None
        self._index = faiss.IndexFlatL2(dimension) if faiss is not None else None

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
            self._vectors = (
                vectors if self._vectors is None
                else np.vstack([self._vectors, vectors])
            )

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return top-k nearest by L2 distance (lower is better)."""
        if k <= 0:
            raise ValueError("k must be positive")
        if self._index is not None:
            return self._index.search(
                np.asarray(queries, dtype="float32"), k
            )
        if self._vectors is None:
            raise RuntimeError("index is empty")
        queries = np.asarray(queries, dtype="float32")
        # Squared L2 distances via broadcasting.
        diff = queries[:, None, :] - self._vectors[None, :, :]
        dists = (diff ** 2).sum(axis=2)
        actual_k = min(k, self._vectors.shape[0])
        order = np.argsort(dists, axis=1)[:, :actual_k]
        top_dists = np.take_along_axis(dists, order, axis=1)
        if actual_k < k:
            pad_width = k - actual_k
            top_dists = np.pad(
                top_dists,
                ((0, 0), (0, pad_width)),
                constant_values=np.inf,
            )
            order = np.pad(
                order,
                ((0, 0), (0, pad_width)),
                constant_values=-1,
            )
        return top_dists, order


# ---------------------------------------------------------------------------
# IVF inner-product index (for large-scale retrieval)
# ---------------------------------------------------------------------------

class IVFInnerProductIndex:
    """IVF-Flat inner-product index for approximate large-scale search.

    Falls back to :class:`InnerProductIndex` when FAISS is unavailable.
    """

    def __init__(self, dimension: int, nlist: int = 100, nprobe: int = 10):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if nlist <= 0:
            raise ValueError("nlist must be positive")
        self.dimension = dimension
        self.nlist = nlist
        self.nprobe = nprobe
        self._trained = False
        self._fallback: InnerProductIndex | None = None

        if faiss is not None:
            quantizer = faiss.IndexFlatIP(dimension)
            self._index: Any = faiss.IndexIVFFlat(
                quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._index.nprobe = nprobe
        else:
            # Fallback: just use the brute-force index.
            self._fallback = InnerProductIndex(dimension)
            self._trained = True
            self._index = None

    @property
    def size(self) -> int:
        if self._index is not None:
            return int(self._index.ntotal)
        return self._fallback.size if self._fallback else 0

    def train(self, vectors: np.ndarray) -> None:
        """Train the quantizer on a representative set of vectors."""
        vectors = np.asarray(vectors, dtype="float32")
        if self._index is not None:
            self._index.train(vectors)
        self._trained = True

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("vectors must have shape (n, dimension)")
        if not self._trained:
            raise RuntimeError("index must be trained before adding vectors")
        if self._index is not None:
            self._index.add(vectors)
        else:
            assert self._fallback is not None
            self._fallback.add(vectors)

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._index is not None:
            return self._index.search(
                np.asarray(queries, dtype="float32"), k
            )
        assert self._fallback is not None
        return self._fallback.search(queries, k)


# ---------------------------------------------------------------------------
# Batch encoding helper
# ---------------------------------------------------------------------------

def encode_passages_batched(
    encoder_fn,
    passages: Sequence[str],
    tokenizer,
    batch_size: int = 64,
    device: str = "cpu",
    max_length: int = 256,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode passages in batches and return a NumPy matrix.

    Parameters
    ----------
    encoder_fn : callable
        A function that takes keyword arguments from the tokenizer and returns
        a ``torch.Tensor`` of shape ``(batch, dim)``.  Typically
        ``model.encode_passages``.
    passages : sequence of str
        The raw passage texts to encode.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer compatible with the encoder.
    batch_size : int
        Encoding batch size.
    device : str
        PyTorch device string.
    max_length : int
        Maximum token length for the tokenizer.
    show_progress : bool
        Whether to display a ``tqdm`` progress bar.
    """
    import torch  # local import to keep module importable without torch

    all_embeddings: list[np.ndarray] = []
    iterator: Any = range(0, len(passages), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Encoding passages", unit="batch")
        except ImportError:
            pass

    with torch.no_grad():
        for start in iterator:
            batch_texts = passages[start : start + batch_size]
            inputs = tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            embeddings = encoder_fn(**inputs)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


# ---------------------------------------------------------------------------
# Build-index convenience function
# ---------------------------------------------------------------------------

def build_index(
    vectors: np.ndarray,
    index_type: str = "flat_ip",
    nlist: int = 100,
    nprobe: int = 10,
) -> InnerProductIndex | L2Index | IVFInnerProductIndex:
    """Create, train (if needed) and populate an index.

    Parameters
    ----------
    vectors : np.ndarray
        Passage embeddings of shape ``(n, dim)``.
    index_type : str
        One of ``"flat_ip"``, ``"flat_l2"``, or ``"ivf_ip"``.
    nlist : int
        Number of Voronoi cells (only used by ``"ivf_ip"``).
    nprobe : int
        Number of cells to probe at query time (only used by ``"ivf_ip"``).

    Returns
    -------
    InnerProductIndex | L2Index | IVFInnerProductIndex
    """
    vectors = np.asarray(vectors, dtype="float32")
    dim = vectors.shape[1]

    if index_type == "flat_ip":
        idx = InnerProductIndex(dim)
        idx.add(vectors)
        return idx
    elif index_type == "flat_l2":
        idx = L2Index(dim)
        idx.add(vectors)
        return idx
    elif index_type == "ivf_ip":
        idx = IVFInnerProductIndex(dim, nlist=nlist, nprobe=nprobe)
        idx.train(vectors)
        idx.add(vectors)
        return idx
    else:
        raise ValueError(
            f"unknown index_type {index_type!r}; expected one of "
            "'flat_ip', 'flat_l2', 'ivf_ip'"
        )
