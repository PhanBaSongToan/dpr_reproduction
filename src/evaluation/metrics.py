"""Metrics used by the DPR reproduction.

Contains standard information-retrieval metrics:

* **Recall@K** – fraction of queries with at least one relevant document in
  the top K results.
* **MRR** (Mean Reciprocal Rank) – average of 1/rank-of-first-relevant.
* **Top-K Accuracy** – alias for Recall@1 generalized to any K.
* **NDCG@K** – Normalized Discounted Cumulative Gain.
* **Latency** – wall-clock search time measurement.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Core retrieval metrics
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Fraction of queries where at least one relevant doc appears in top-k.

    Parameters
    ----------
    retrieved : list[list[str]]
        Retrieved document IDs per query, ordered by score (best first).
    relevant : list[set | list of str]
        Gold-standard relevant document IDs per query.
    k : int
        Cutoff depth.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(retrieved) != len(relevant):
        raise ValueError(
            f"length mismatch: {len(retrieved)} retrieved vs "
            f"{len(relevant)} relevant"
        )
    hits = sum(
        bool(set(results[:k]) & set(gold))
        for results, gold in zip(retrieved, relevant)
    )
    return hits / max(1, len(retrieved))


def mean_reciprocal_rank(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
) -> float:
    """Mean Reciprocal Rank (MRR).

    For each query the reciprocal rank is 1/rank of the first relevant
    document.  Queries with no relevant document in the results contribute 0.
    """
    if len(retrieved) != len(relevant):
        raise ValueError(
            f"length mismatch: {len(retrieved)} retrieved vs "
            f"{len(relevant)} relevant"
        )
    total = 0.0
    for results, gold in zip(retrieved, relevant):
        gold_set = set(gold)
        for rank, item in enumerate(results, start=1):
            if item in gold_set:
                total += 1.0 / rank
                break
    return total / max(1, len(retrieved))


def top_k_accuracy(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Top-K accuracy (equivalent to Recall@K with binary relevance)."""
    return recall_at_k(retrieved, relevant, k)


def _dcg(relevances: Sequence[float], k: int) -> float:
    """Discounted Cumulative Gain at depth *k*."""
    return sum(
        rel / np.log2(i + 2)  # i+2 because rank starts at 1
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(
    retrieved: Sequence[Sequence[str]],
    relevant: Sequence[Iterable[str]],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain (NDCG) at depth *k*.

    Uses binary relevance: 1 if the document is in the gold set, 0 otherwise.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if len(retrieved) != len(relevant):
        raise ValueError(
            f"length mismatch: {len(retrieved)} retrieved vs "
            f"{len(relevant)} relevant"
        )

    total_ndcg = 0.0
    for results, gold in zip(retrieved, relevant):
        gold_set = set(gold)
        binary_rels = [1.0 if doc in gold_set else 0.0 for doc in results[:k]]
        dcg = _dcg(binary_rels, k)
        # Ideal: all relevant docs at the top.
        ideal_rels = sorted(binary_rels, reverse=True)
        idcg = _dcg(ideal_rels, k)
        total_ndcg += dcg / idcg if idcg > 0 else 0.0
    return total_ndcg / max(1, len(retrieved))


# ---------------------------------------------------------------------------
# Latency measurement
# ---------------------------------------------------------------------------

@dataclass
class LatencyStats:
    """Container for search latency statistics (in milliseconds)."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    total_ms: float = 0.0
    num_queries: int = 0
    raw_ms: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "num_queries": self.num_queries,
        }


def measure_search_latency(
    index: Any,
    queries: np.ndarray,
    k: int,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> LatencyStats:
    """Measure search latency by running multiple queries.

    Parameters
    ----------
    index : InnerProductIndex | L2Index | IVFInnerProductIndex
        An index that supports ``.search(queries, k)``.
    queries : np.ndarray
        Query vectors of shape ``(n, dim)``.
    k : int
        Top-k to retrieve.
    num_runs : int
        Number of timed runs per query; the median is kept.
    warmup_runs : int
        Number of un-timed warm-up runs.

    Returns
    -------
    LatencyStats
    """
    queries = np.asarray(queries, dtype="float32")

    # Warm-up (uncached / JIT / branch prediction).
    for _ in range(warmup_runs):
        index.search(queries, k)

    per_query_ms: list[float] = []
    for _ in range(num_runs):
        for i in range(queries.shape[0]):
            q = queries[i : i + 1]
            t0 = time.perf_counter()
            index.search(q, k)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            per_query_ms.append(elapsed_ms)

    arr = np.array(per_query_ms)
    return LatencyStats(
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        total_ms=float(arr.sum()),
        num_queries=len(per_query_ms),
        raw_ms=per_query_ms,
    )


# ---------------------------------------------------------------------------
# Aggregate results container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """Holds evaluation metrics for a single experiment configuration."""

    method: str
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    latency: LatencyStats | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for logging or a DataFrame row."""
        row: dict[str, Any] = {"method": self.method, "mrr": round(self.mrr, 4)}
        for k, v in sorted(self.recall.items()):
            row[f"recall@{k}"] = round(v, 4)
        for k, v in sorted(self.ndcg.items()):
            row[f"ndcg@{k}"] = round(v, 4)
        if self.latency is not None:
            row.update(
                {f"latency_{k}": v for k, v in self.latency.as_dict().items()}
            )
        row.update(self.extra)
        return row
