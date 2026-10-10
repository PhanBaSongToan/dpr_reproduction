import numpy as np
from src.indexing.faiss_index import InnerProductIndex
from src.evaluation.metrics import recall_at_k


def test_inner_product_index_and_recall():
    index = InnerProductIndex(2)
    index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
    _, ids = index.search(np.array([[0.9, 0.1]], dtype=np.float32), 1)
    assert ids.tolist() == [[0]]
    assert recall_at_k([["p0"]], [["p0"]], 1) == 1.0
