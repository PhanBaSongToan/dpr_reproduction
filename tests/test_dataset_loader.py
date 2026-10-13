
import pytest

from src.data.dataset_loader import (
    load_jsonl,
    save_jsonl,
    normalize_qa_record,
)


def test_save_and_load_jsonl(tmp_path):
    records = [
        {"id": "1", "text": "First passage"},
        {"id": "2", "text": "Second passage"},
    ]

    output = tmp_path / "sample.jsonl"

    save_jsonl(records, str(output))
    loaded = list(load_jsonl(str(output)))

    assert loaded == records


def test_normalize_qa_record():
    record = {
        "id": "q1",
        "question": "Who invented the telephone?",
        "answers": ["Alexander Graham Bell"],
    }

    result = normalize_qa_record(record)

    assert result["question_id"] == "q1"
    assert result["question"] == "Who invented the telephone?"
    assert result["answers"] == ["Alexander Graham Bell"]


def test_normalize_single_answer():
    record = {
        "id": 10,
        "question": "What is DPR?",
        "answers": "Dense Passage Retrieval",
    }

    result = normalize_qa_record(record)

    assert result["question_id"] == "10"
    assert result["answers"] == ["Dense Passage Retrieval"]


def test_missing_question():
    with pytest.raises(ValueError):
        normalize_qa_record({
            "id": "q1",
            "question": "",
            "answers": [],
        })


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(load_jsonl(str(tmp_path / "missing.jsonl")))


def test_invalid_jsonl(tmp_path):
    path = tmp_path / "invalid.jsonl"
    path.write_text("{invalid json}", encoding="utf-8")

    with pytest.raises(ValueError):
        list(load_jsonl(str(path)))
