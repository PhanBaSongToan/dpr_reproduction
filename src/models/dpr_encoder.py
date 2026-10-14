"""Dual encoder interface. The Hugging Face implementation is kept small and testable."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn


class DPRDualEncoder(nn.Module):
    """Independent question and passage encoders using CLS representations."""

    def __init__(self, question_encoder: nn.Module, passage_encoder: nn.Module):
        super().__init__()
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "bert-base-uncased",
        **kwargs: Any,
    ) -> "DPRDualEncoder":
        """Create two independently loaded Transformer encoders.

        ``kwargs`` are passed to both ``AutoModel.from_pretrained`` calls.
        The import is intentionally lazy so tests can use lightweight dummy
        encoders without requiring the Transformers package or network access.
        """
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "DPRDualEncoder.from_pretrained requires the transformers package"
            ) from exc

        question_encoder = AutoModel.from_pretrained(model_name, **kwargs)
        passage_encoder = AutoModel.from_pretrained(model_name, **kwargs)
        return cls(question_encoder, passage_encoder)

    @staticmethod
    def _cls(outputs: Any) -> Tensor:
        """Extract the first-token representation from encoder output."""
        hidden = getattr(outputs, "last_hidden_state", outputs)
        if not isinstance(hidden, Tensor) or hidden.ndim != 3:
            raise ValueError(
                "encoder output must contain a three-dimensional last_hidden_state"
            )
        if hidden.shape[1] == 0:
            raise ValueError("encoder output must contain at least one token")
        return hidden[:, 0, :]

    def encode_questions(self, **inputs: Any) -> Tensor:
        """Encode tokenized questions and return their CLS embeddings."""
        return self._cls(self.question_encoder(**inputs))

    def encode_passages(self, **inputs: Any) -> Tensor:
        """Encode tokenized passages and return their CLS embeddings."""
        return self._cls(self.passage_encoder(**inputs))


def dot_product_scores(question_embeddings: Tensor, passage_embeddings: Tensor) -> Tensor:
    """Return the question-by-passage inner-product matrix ``S = QP^T``."""
    if question_embeddings.ndim != 2 or passage_embeddings.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if question_embeddings.shape[1] != passage_embeddings.shape[1]:
        raise ValueError("embedding dimensions must match")
    return question_embeddings @ passage_embeddings.T
