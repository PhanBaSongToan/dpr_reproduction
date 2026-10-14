"""End-to-end retrieval evaluator.

Connects a FAISS (or fallback) index with gold annotations and computes
all reproduction metrics in a single call.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.evaluation.metrics import (
    EvaluationResult,
    LatencyStats,
    mean_reciprocal_rank,
    measure_search_latency,
    ndcg_at_k,
    recall_at_k,
)
from src.indexing.faiss_index import InnerProductIndex, build_index


class RetrievalEvaluator:
    """Orchestrates FAISS indexing, search, and metric computation.

    Usage
    -----
    >>> evaluator = RetrievalEvaluator(
    ...     passage_vectors=passage_vecs,      # (N, dim)
    ...     passage_ids=passage_ids,            # list[str] of length N
    ...     query_vectors=query_vecs,           # (Q, dim)
    ...     gold_relevant=gold_ids,             # list[set[str]] of length Q
    ... )
    >>> result = evaluator.evaluate(top_k=[5, 20, 100], method_name="dpr")
    >>> print(result.summary())
    """

    def __init__(
        self,
        passage_vectors: np.ndarray,
        passage_ids: Sequence[str],
        query_vectors: np.ndarray,
        gold_relevant: Sequence[set[str] | list[str]],
        index_type: str = "flat_ip",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        self.passage_vectors = np.asarray(passage_vectors, dtype="float32")
        self.passage_ids = list(passage_ids)
        self.query_vectors = np.asarray(query_vectors, dtype="float32")
        self.gold_relevant = [set(g) for g in gold_relevant]

        if self.passage_vectors.shape[0] != len(self.passage_ids):
            raise ValueError("passage_vectors and passage_ids must match")
        if self.query_vectors.shape[0] != len(self.gold_relevant):
            raise ValueError("query_vectors and gold_relevant must match")

        # Build the index once.
        self.index = build_index(
            self.passage_vectors,
            index_type=index_type,
            nlist=nlist,
            nprobe=nprobe,
        )

    def _retrieve(self, k: int) -> list[list[str]]:
        """Run search and translate integer indices to passage IDs."""
        _, indices = self.index.search(self.query_vectors, k)
        retrieved: list[list[str]] = []
        for row in indices:
            ids = [
                self.passage_ids[int(i)] for i in row if int(i) >= 0
            ]
            retrieved.append(ids)
        return retrieved

    def evaluate(
        self,
        top_k: Sequence[int] = (5, 20, 100),
        method_name: str = "dpr",
        measure_latency: bool = True,
        latency_runs: int = 3,
    ) -> EvaluationResult:
        """Run full evaluation and return an :class:`EvaluationResult`.

        Parameters
        ----------
        top_k : sequence of int
            List of K values for Recall@K and NDCG@K.
        method_name : str
            Label for this experiment (e.g. ``"dpr"`` or ``"bm25"``).
        measure_latency : bool
            Whether to benchmark single-query latency.
        latency_runs : int
            Number of timed runs for latency measurement.
        """
        max_k = max(top_k)
        retrieved = self._retrieve(max_k)

        recall_scores: dict[int, float] = {}
        ndcg_scores: dict[int, float] = {}
        for k in sorted(top_k):
            recall_scores[k] = recall_at_k(retrieved, self.gold_relevant, k)
            ndcg_scores[k] = ndcg_at_k(retrieved, self.gold_relevant, k)

        mrr = mean_reciprocal_rank(retrieved, self.gold_relevant)

        latency: LatencyStats | None = None
        if measure_latency:
            latency = measure_search_latency(
                self.index,
                self.query_vectors,
                k=max_k,
                num_runs=latency_runs,
            )

        return EvaluationResult(
            method=method_name,
            recall=recall_scores,
            mrr=mrr,
            ndcg=ndcg_scores,
            latency=latency,
        )


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_methods(
    results: Sequence[EvaluationResult],
) -> list[dict[str, Any]]:
    """Return a list of flat dicts ready for ``pandas.DataFrame`` or printing.

    >>> rows = compare_methods([bm25_result, dpr_result])
    >>> import pandas as pd; print(pd.DataFrame(rows))
    """
    return [r.summary() for r in results]


def save_results(
    results: Sequence[EvaluationResult],
    path: str | Path,
) -> None:
    """Save evaluation results to a JSON file for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = compare_methods(results)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_results(path: str | Path) -> list[dict[str, Any]]:
    """Load previously saved evaluation results."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
