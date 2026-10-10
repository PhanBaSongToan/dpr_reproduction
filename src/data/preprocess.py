"""Minimal, deterministic passage preprocessing for DPR experiments."""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Mapping


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving the original word order."""
    return re.sub(r"\s+", " ", text).strip()


def split_into_passages(text: str, words_per_passage: int = 100) -> list[str]:
    """Split text into disjoint fixed-length word passages."""
    if words_per_passage <= 0:
        raise ValueError("words_per_passage must be positive")
    words = normalize_text(text).split()
    return [" ".join(words[i : i + words_per_passage]) for i in range(0, len(words), words_per_passage) if words[i : i + words_per_passage]]


def build_passage_records(documents: Iterable[Mapping[str, str]], words_per_passage: int = 100) -> Iterator[dict]:
    """Yield stable passage records from {id, title, text} documents."""
    for document in documents:
        title = normalize_text(document.get("title", ""))
        for offset, passage in enumerate(split_into_passages(document.get("text", ""), words_per_passage)):
            yield {"passage_id": f"{document['id']}_{offset}", "title": title, "text": passage}
