from pathlib import Path

import torch
from torch import nn

from src.models.dpr_encoder import DPRDualEncoder
from src.training.trainer import DPRTrainer


class DummyEncoder(nn.Module):
    def __init__(self, vocab_size=16, embedding_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_ids, attention_mask=None):
        return self.embedding(input_ids)


def batch():
    inputs = {"input_ids": torch.tensor([[1, 2], [3, 4]])}
    hard = {"input_ids": torch.tensor([[5, 6], [7, 8]])}
    return {
        "question_inputs": inputs,
        "positive_passage_inputs": inputs,
        "hard_negative_inputs": hard,
    }


def test_fit_updates_model_and_saves_checkpoint(tmp_path: Path):
    torch.manual_seed(7)
    model = DPRDualEncoder(DummyEncoder(), DummyEncoder())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    trainer = DPRTrainer(model, optimizer)

    losses = trainer.fit([batch(), batch()], epochs=2, checkpoint_dir=tmp_path)

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert (tmp_path / "checkpoint_epoch_1.pt").exists()
    assert (tmp_path / "checkpoint_epoch_2.pt").exists()
