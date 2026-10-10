"""Metrics used by the DPR reproduction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def recall_at_k(retrieved: Sequence[Sequence[str]], relevant: Sequence[Iterable[str]], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(bool(set(results[:k]) & set(gold)) for results, gold in zip(retrieved, relevant))
    return hits / max(1, len(retrieved))


def mean_reciprocal_rank(retrieved: Sequence[Sequence[str]], relevant: Sequence[Iterable[str]]) -> float:
    total = 0.0
    for results, gold in zip(retrieved, relevant):
        gold = set(gold)
        for rank, item in enumerate(results, start=1):
            if item in gold:
                total += 1.0 / rank
                break
    return total / max(1, len(retrieved))
