"""Dense retrieval wrapper connecting a query encoder to an inner-product index."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.indexing.faiss_index import InnerProductIndex


class DenseRetriever:
    def __init__(self, index: InnerProductIndex, passage_ids: Sequence[str]):
        if len(passage_ids) != index.size:
            raise ValueError("passage_ids must match the indexed vectors")
        self.index = index
        self.passage_ids = list(passage_ids)

    def search(self, query_vectors: np.ndarray, k: int = 5):
        scores, indices = self.index.search(query_vectors, k)
        return [
            [
                {"passage_id": self.passage_ids[int(i)], "score": float(score)}
                for score, i in zip(row_scores, row_indices)
                if int(i) >= 0
            ]
            for row_scores, row_indices in zip(scores, indices)
        ]
