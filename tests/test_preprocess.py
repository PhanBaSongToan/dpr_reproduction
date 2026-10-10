from src.data.preprocess import split_into_passages


def test_split_into_disjoint_passages():
    assert split_into_passages("a b c d e", 2) == ["a b", "c d", "e"]
