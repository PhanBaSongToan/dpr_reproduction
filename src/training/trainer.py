"""A small, explicit DPR training loop for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.models.dpr_encoder import DPRDualEncoder, dot_product_scores
from src.training.loss import nll_loss_with_hard_negative


class DPRTrainer:
    """Train a dual encoder with in-batch negatives and optional BM25 negatives.

    Each batch must contain ``question_inputs`` and ``positive_passage_inputs``.
    Their values are dictionaries accepted by the corresponding Hugging Face
    encoder. ``hard_negative_inputs`` is optional and contains one passage per
    question selected by BM25.
    """

    def __init__(self, model: DPRDualEncoder, optimizer: torch.optim.Optimizer, device: str = "cpu"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = torch.device(device)

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if isinstance(value, Tensor) else value
            for key, value in inputs.items()
        }

    def train_step(
        self,
        question_inputs: dict[str, Any],
        positive_passage_inputs: dict[str, Any],
        hard_negative_inputs: dict[str, Any] | None = None,
    ) -> float:
        """Run one update and return the scalar loss."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        questions = self.model.encode_questions(**self._move_inputs(question_inputs))
        positives = self.model.encode_passages(**self._move_inputs(positive_passage_inputs))
        scores = dot_product_scores(questions, positives)

        if hard_negative_inputs is None:
            loss = torch.nn.functional.cross_entropy(
                scores, torch.arange(scores.shape[0], device=self.device)
            )
        else:
            hard_negatives = self.model.encode_passages(**self._move_inputs(hard_negative_inputs))
            hard_scores = (questions * hard_negatives).sum(dim=1)
            loss = nll_loss_with_hard_negative(scores, hard_scores)

        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def save_checkpoint(self, path: str | Path, epoch: int, loss: float) -> None:
        """Save model, optimizer and run metadata for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def fit(
        self,
        dataloader,
        epochs: int,
        checkpoint_dir: str | Path | None = None,
    ) -> list[float]:
        """Train for several epochs and optionally save one checkpoint per epoch.

        The dataloader must yield dictionaries accepted by ``train_step``. The
        returned list contains the mean training loss for each epoch.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")

        epoch_losses: list[float] = []
        for epoch in range(1, epochs + 1):
            losses = []
            for batch in dataloader:
                losses.append(self.train_step(**batch))
            if not losses:
                raise ValueError("dataloader must yield at least one batch")
            mean_loss = sum(losses) / len(losses)
            epoch_losses.append(mean_loss)
            if checkpoint_dir is not None:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
                self.save_checkpoint(checkpoint_path, epoch, mean_loss)
        return epoch_losses
