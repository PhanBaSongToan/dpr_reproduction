import torch
from src.training.loss import in_batch_nll_loss


def test_in_batch_loss_prefers_diagonal():
    scores = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    assert in_batch_nll_loss(scores).item() < 0.1
