
"""Dataset loading utilities for DPR experiments."""

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(file_path: str) -> Iterator[dict]:
    """Read records from a JSONL file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected object at line {line_number}"
                )

            yield record


def save_jsonl(records, file_path: str) -> None:
    """Write dataset records in JSONL format."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_qa_record(record: dict) -> dict:
    """Normalize a QA example to a common schema."""
    question_id = record.get("question_id", record.get("id"))
    question = record.get("question", "")
    answers = record.get("answers", [])

    if question_id is None:
        raise ValueError("Missing question ID")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Missing or invalid question")

    if isinstance(answers, str):
        answers = [answers]

    if not isinstance(answers, list) or not all(
        isinstance(answer, str) for answer in answers
    ):
        raise ValueError("Answers must be a list of strings")

    return {
        "question_id": str(question_id),
        "question": question.strip(),
        "answers": answers,
    }
