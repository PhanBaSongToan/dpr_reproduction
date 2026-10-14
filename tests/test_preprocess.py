
import pytest

from src.data.preprocess import (
    normalize_text,
    split_into_passages,
    build_passage_records,
)


def test_normalize_text():
    assert normalize_text("  Hello   world \n DPR  ") == "Hello world DPR"


def test_split_into_disjoint_passages():
    assert split_into_passages("a b c d e", 2) == ["a b", "c d", "e"]


def test_split_default_100_words():
    text = " ".join(["word"] * 250)
    passages = split_into_passages(text)

    assert len(passages) == 3
    assert len(passages[0].split()) == 100
    assert len(passages[1].split()) == 100
    assert len(passages[2].split()) == 50


def test_empty_text():
    assert split_into_passages("") == []


def test_invalid_passage_size():
    with pytest.raises(ValueError):
        split_into_passages("hello world", 0)


def test_build_passage_records():
    documents = [
        {
            "id": "doc1",
            "title": "  Python  Programming ",
            "text": "a b c d e",
        }
    ]

    records = list(build_passage_records(documents, 2))

    assert len(records) == 3
    assert records[0] == {
        "passage_id": "doc1_0",
        "title": "Python Programming",
        "text": "a b",
    }
    assert records[1]["passage_id"] == "doc1_1"
    assert records[2]["text"] == "e"

