"""Small BM25 baseline wrapper."""

from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, passages: list[str]):
        self.passages = passages
        self._tokens = [text.lower().split() for text in passages]
        self._index = BM25Okapi(self._tokens)

    def search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self._index.get_scores(query.lower().split())
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [(i, float(scores[i])) for i in order]
