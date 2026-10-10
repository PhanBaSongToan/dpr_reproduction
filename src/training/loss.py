"""DPR in-batch negative objective."""

import torch
from torch import Tensor
from torch.nn import functional as F


def in_batch_nll_loss(scores: Tensor) -> Tensor:
    """Diagonal labels identify each question's matching passage."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square B x B matrix")
    labels = torch.arange(scores.shape[0], device=scores.device)
    return F.cross_entropy(scores, labels)


def nll_loss_with_hard_negative(scores: Tensor, hard_negative_scores: Tensor) -> Tensor:
    """Add one BM25 hard-negative score to each row before cross entropy."""
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores must be a square B x B matrix")
    if hard_negative_scores.shape != (scores.shape[0],):
        raise ValueError("hard_negative_scores must have shape (B,)")
    logits = torch.cat([scores, hard_negative_scores[:, None]], dim=1)
    labels = torch.arange(scores.shape[0], device=scores.device)
    return F.cross_entropy(logits, labels)
