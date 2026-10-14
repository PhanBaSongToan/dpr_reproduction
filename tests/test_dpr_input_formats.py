
"""Test compatibility with official DPR JSON and GZIP inputs."""

import gzip
import json

import pytest

from scripts.prepare_dpr_subset import (
    is_gzip_file,
    read_questions,
    resolve_input_path,
)


@pytest.fixture
def sample_records():
    return [
        {
            "question": "Who wrote Hamlet?",
            "answers": ["William Shakespeare"],
            "positive_ctxs": [],
            "negative_ctxs": [],
            "hard_negative_ctxs": [],
        }
    ]


def test_read_official_plain_json(tmp_path, sample_records):
    path = tmp_path / "biencoder-nq-train.json"
    path.write_text(
        json.dumps(sample_records),
        encoding="utf-8",
    )

    assert is_gzip_file(path) is False
    records = list(read_questions(path, 1))
    assert len(records) == 1
    assert records[0][1]["question"] == "Who wrote Hamlet?"


def test_read_gzip_tmp(tmp_path, sample_records):
    path = tmp_path / "nq-train.tmp"

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(sample_records, f)

    assert is_gzip_file(path) is True
    records = list(read_questions(path, 1))
    assert len(records) == 1
    assert records[0][1]["answers"] == ["William Shakespeare"]


def test_read_official_gzip_json(tmp_path, sample_records):
    path = tmp_path / "biencoder-nq-train.json.gz"

    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(sample_records, f)

    assert is_gzip_file(path) is True
    assert len(list(read_questions(path, 1))) == 1


def test_prefer_official_json(tmp_path):
    (tmp_path / "nq-train.tmp").write_bytes(b"unused")

    official = tmp_path / "biencoder-nq-train.json"
    official.write_text("[]", encoding="utf-8")

    assert resolve_input_path(tmp_path, "train") == official


def test_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_input_path(tmp_path, "dev")
