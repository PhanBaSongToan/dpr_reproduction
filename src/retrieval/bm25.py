"""Small, deterministic BM25Okapi implementation using Python standard library.

Paper reference: Lucene BM25, b=0.4, k1=0.9. This implementation is not
Lucene and scores can differ with tokenization/IDF details.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


class BM25Retriever:
    def __init__(self, passages: list[str], passage_ids: list[str] | None = None,
                 *, k1: float = 0.9, b: float = 0.4):
        if not passages:
            raise ValueError("Empty passage corpus")
        if not 0 <= b <= 1 or k1 <= 0:
            raise ValueError("Require 0<=b<=1, k1>0")
        self.ids = list(passage_ids) if passage_ids is not None else [str(i) for i in range(len(passages))]
        if len(self.ids) != len(passages) or len(set(self.ids)) != len(self.ids):
            raise ValueError("Passage IDs must be unique and have same length as passages")
        self.k1, self.b, self.passages = k1, b, passages
        docs = [Counter(tokenize(x)) for x in passages]
        self.lengths = [sum(c.values()) for c in docs]
        self.avgdl = sum(self.lengths) / len(docs)
        df = Counter(term for doc in docs for term in doc)
        self.idf = {t: math.log1p((len(docs) - n + .5) / (n + .5)) for t,n in df.items()}
        self.postings = defaultdict(list)
        for i, doc in enumerate(docs):
            for term, tf in doc.items():
                self.postings[term].append((i, tf))

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        if k <= 0:
            raise ValueError("k must be positive")
        terms = set(tokenize(query))
        if not terms:
            return []
        scores = defaultdict(float)
        for term in terms:
            idf = self.idf.get(term, 0.0)
            for i, tf in self.postings.get(term, []):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avgdl,1e-9))
                scores[i] += idf * (tf * (self.k1 + 1) / (tf + norm))
        # Return actual matches only, avoiding arbitrary zero-score nonmatches.
        order = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [(self.ids[i], round(scores[i], 9)) for i in order]

    def hard_negatives(self, query: str, positive_ids: set[str], answers: list[str], k: int = 5) -> list[str]:
        """Exclude both known positive IDs and passages with an answer string.

        Text matching is a heuristic and may exclude true negatives or miss paraphrases.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        texts_by_id = dict(zip(self.ids, self.passages))
        lowered_answers = [a.casefold().strip() for a in answers if a.strip()]
        results = []
        for pid, _ in self.search(query, k=len(self.ids)):
            text = texts_by_id[pid].casefold()
            if pid not in positive_ids and not any(a in text for a in lowered_answers):
                results.append(pid)
            if len(results) == k:
                break
        return results
