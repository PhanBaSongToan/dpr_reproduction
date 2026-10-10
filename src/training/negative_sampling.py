"""Negative sampling utilities used by the DPR training pipeline."""

from __future__ import annotations

from collections.abc import Sequence


def sample_bm25_hard_negatives(
    candidate_passage_ids: Sequence[Sequence[str]],
    positive_passage_ids: Sequence[str | set[str]],
    num_negatives: int = 1,
) -> list[list[str]]:
    """Select top BM25 candidates that are not known positive passages.

    ``candidate_passage_ids[i]`` must already be ordered by BM25 score for
    question ``i``. The function only selects IDs; passage text lookup stays
    outside the sampler so it can be backed by a dataset or a database.
    """
    if len(candidate_passage_ids) != len(positive_passage_ids):
        raise ValueError("candidate and positive collections must have equal length")
    if num_negatives <= 0:
        raise ValueError("num_negatives must be positive")

    selected: list[list[str]] = []
    for candidates, positives in zip(candidate_passage_ids, positive_passage_ids):
        positive_set = {positives} if isinstance(positives, str) else set(positives)
        negatives = [pid for pid in candidates if pid not in positive_set]
        selected.append(negatives[:num_negatives])
    return selected


def flatten_one_hard_negative(samples: Sequence[Sequence[str]]) -> list[str | None]:
    """Return one hard-negative ID per query, using ``None`` when unavailable."""
    return [row[0] if row else None for row in samples]
