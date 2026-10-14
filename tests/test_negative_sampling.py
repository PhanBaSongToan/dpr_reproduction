from src.training.negative_sampling import (
    flatten_one_hard_negative,
    sample_bm25_hard_negatives,
)


def test_hard_negative_sampler_filters_positive_passage():
    candidates = [["positive", "hard-1", "hard-2"], ["hard-3", "positive-2"]]
    positives = ["positive", {"positive-2"}]
    assert sample_bm25_hard_negatives(candidates, positives) == [["hard-1"], ["hard-3"]]


def test_flatten_sampler_handles_missing_candidates():
    assert flatten_one_hard_negative([["p1"], []]) == ["p1", None]
