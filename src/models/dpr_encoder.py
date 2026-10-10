"""Dual encoder interface. The Hugging Face implementation is kept small and testable."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DPRDualEncoder(nn.Module):
    """Independent question and passage encoders using CLS representations."""

    def __init__(self, question_encoder: nn.Module, passage_encoder: nn.Module):
        super().__init__()
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder

    @staticmethod
    def _cls(outputs) -> Tensor:
        hidden = getattr(outputs, "last_hidden_state", outputs)
        return hidden[:, 0, :]

    def encode_questions(self, **inputs) -> Tensor:
        return self._cls(self.question_encoder(**inputs))

    def encode_passages(self, **inputs) -> Tensor:
        return self._cls(self.passage_encoder(**inputs))


def dot_product_scores(question_embeddings: Tensor, passage_embeddings: Tensor) -> Tensor:
    """Return the BxB score matrix S = QP^T."""
    if question_embeddings.ndim != 2 or passage_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if question_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("embedding dimensions must match")
    return question_embeddings @ passage_embeddings.T
