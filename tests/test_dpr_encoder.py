from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.models.dpr_encoder import DPRDualEncoder, dot_product_scores


class RecordingEncoder(nn.Module):
    def __init__(self, offset: float = 0.0, raw_tensor_output: bool = False):
        super().__init__()
        self.offset = offset
        self.raw_tensor_output = raw_tensor_output
        self.calls = []

    def forward(self, input_ids, attention_mask=None):
        self.calls.append((input_ids.clone(), attention_mask))
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 2) + self.offset
        if self.raw_tensor_output:
            return hidden
        return SimpleNamespace(last_hidden_state=hidden)


def test_question_and_passage_encoders_are_called_independently():
    question_encoder = RecordingEncoder(offset=1.0)
    passage_encoder = RecordingEncoder(offset=10.0)
    model = DPRDualEncoder(question_encoder, passage_encoder)
    question_ids = torch.tensor([[2, 9], [3, 8]])
    passage_ids = torch.tensor([[4, 7], [5, 6]])

    question_embeddings = model.encode_questions(input_ids=question_ids)
    passage_embeddings = model.encode_passages(input_ids=passage_ids)

    assert len(question_encoder.calls) == 1
    assert len(passage_encoder.calls) == 1
    assert torch.equal(question_encoder.calls[0][0], question_ids)
    assert torch.equal(passage_encoder.calls[0][0], passage_ids)
    assert torch.equal(question_embeddings, torch.tensor([[3.0, 3.0], [4.0, 4.0]]))
    assert torch.equal(passage_embeddings, torch.tensor([[14.0, 14.0], [15.0, 15.0]]))


def test_cls_representation_is_extracted_from_last_hidden_state():
    encoder = RecordingEncoder()
    model = DPRDualEncoder(encoder, RecordingEncoder())
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    embeddings = model.encode_questions(input_ids=input_ids)

    expected = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    assert torch.equal(embeddings, expected)


def test_raw_tensor_outputs_are_supported():
    encoder = RecordingEncoder(raw_tensor_output=True)
    model = DPRDualEncoder(encoder, RecordingEncoder())
    input_ids = torch.tensor([[7, 8, 9]])

    embeddings = model.encode_questions(input_ids=input_ids)

    assert torch.equal(embeddings, torch.tensor([[7.0, 7.0]]))


def test_dot_product_scores_are_numerically_correct_and_have_expected_shape():
    questions = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    passages = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    scores = dot_product_scores(questions, passages)

    assert scores.shape == (2, 2)
    assert torch.equal(scores, torch.tensor([[17.0, 23.0], [39.0, 53.0]]))


def test_dot_product_scores_rejects_invalid_tensor_ranks():
    with pytest.raises(ValueError, match="two-dimensional"):
        dot_product_scores(torch.ones(2), torch.ones(2, 2))

    with pytest.raises(ValueError, match="two-dimensional"):
        dot_product_scores(torch.ones(2, 2), torch.ones(2))


def test_dot_product_scores_rejects_mismatched_embedding_dimensions():
    with pytest.raises(ValueError, match="embedding dimensions must match"):
        dot_product_scores(torch.ones(2, 3), torch.ones(2, 4))


def test_end_to_end_encoding_produces_a_batch_score_matrix():
    model = DPRDualEncoder(RecordingEncoder(offset=1.0), RecordingEncoder(offset=2.0))
    question_ids = torch.tensor([[1, 0], [2, 0]])
    passage_ids = torch.tensor([[3, 0], [4, 0]])

    question_embeddings = model.encode_questions(input_ids=question_ids)
    passage_embeddings = model.encode_passages(input_ids=passage_ids)
    scores = dot_product_scores(question_embeddings, passage_embeddings)

    assert question_embeddings.shape == (2, 2)
    assert passage_embeddings.shape == (2, 2)
    assert scores.shape == (2, 2)
    assert torch.equal(scores, torch.tensor([[20.0, 24.0], [30.0, 36.0]]))
