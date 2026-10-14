"""Retrieval metrics and evaluation pipeline.

Re-exports the main classes and functions for convenient access::

    from src.evaluation import recall_at_k, RetrievalEvaluator
"""

from .evaluator import (
    RetrievalEvaluator,
    compare_methods,
    load_results,
    save_results,
)
from .metrics import (
    EvaluationResult,
    LatencyStats,
    mean_reciprocal_rank,
    measure_search_latency,
    ndcg_at_k,
    recall_at_k,
    top_k_accuracy,
)

__all__ = [
    "EvaluationResult",
    "LatencyStats",
    "RetrievalEvaluator",
    "compare_methods",
    "load_results",
    "mean_reciprocal_rank",
    "measure_search_latency",
    "ndcg_at_k",
    "recall_at_k",
    "save_results",
    "top_k_accuracy",
]
