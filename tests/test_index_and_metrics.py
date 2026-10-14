"""Comprehensive tests for FAISS indexing and retrieval evaluation.

Covers:
- InnerProductIndex: add, search, edge cases, save/load
- L2Index: basic search correctness
- IVFInnerProductIndex: train → add → search workflow
- build_index helper for all index types
- Recall@K, MRR, NDCG@K, Top-K accuracy
- LatencyStats and measure_search_latency
- EvaluationResult summary serialization
- RetrievalEvaluator end-to-end pipeline
- compare_methods / save_results / load_results
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.indexing.faiss_index import (
    InnerProductIndex,
    IVFInnerProductIndex,
    L2Index,
    build_index,
)
from src.evaluation.metrics import (
    EvaluationResult,
    LatencyStats,
    mean_reciprocal_rank,
    measure_search_latency,
    ndcg_at_k,
    recall_at_k,
    top_k_accuracy,
)
from src.evaluation.evaluator import (
    RetrievalEvaluator,
    compare_methods,
    load_results,
    save_results,
)


# ===================================================================
#  INDEXING TESTS
# ===================================================================


class TestInnerProductIndex:
    """Tests for InnerProductIndex (flat inner-product search)."""

    def test_add_and_search_basic(self):
        idx = InnerProductIndex(3)
        vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        idx.add(vecs)
        assert idx.size == 3

        query = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)
        scores, ids = idx.search(query, k=2)
        # Best match should be vector 0 ([1,0,0]).
        assert ids[0, 0] == 0
        assert scores.shape == (1, 2)
        assert ids.shape == (1, 2)

    def test_search_returns_correct_order(self):
        idx = InnerProductIndex(2)
        idx.add(np.array([[1, 0], [0, 1], [0.7, 0.7]], dtype=np.float32))
        query = np.array([[1.0, 1.0]], dtype=np.float32)
        scores, ids = idx.search(query, k=3)
        # [0.7, 0.7] has highest inner product with [1, 1].
        assert ids[0, 0] == 2

    def test_add_incremental(self):
        idx = InnerProductIndex(2)
        idx.add(np.array([[1, 0]], dtype=np.float32))
        assert idx.size == 1
        idx.add(np.array([[0, 1]], dtype=np.float32))
        assert idx.size == 2

    def test_dimension_mismatch_raises(self):
        idx = InnerProductIndex(3)
        with pytest.raises(ValueError, match="shape"):
            idx.add(np.array([[1, 0]], dtype=np.float32))

    def test_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="positive"):
            InnerProductIndex(0)

    def test_search_empty_index_raises(self):
        idx = InnerProductIndex(2)
        with pytest.raises(RuntimeError, match="empty"):
            idx.search(np.array([[1, 0]], dtype=np.float32), k=1)

    def test_invalid_k_raises(self):
        idx = InnerProductIndex(2)
        idx.add(np.array([[1, 0]], dtype=np.float32))
        with pytest.raises(ValueError, match="k must be positive"):
            idx.search(np.array([[1, 0]], dtype=np.float32), k=0)

    def test_k_larger_than_index(self):
        """When k > number of indexed vectors, results should be padded."""
        idx = InnerProductIndex(2)
        idx.add(np.array([[1, 0]], dtype=np.float32))
        scores, ids = idx.search(np.array([[1, 0]], dtype=np.float32), k=5)
        assert scores.shape == (1, 5)
        assert ids.shape == (1, 5)
        # Only first result should be valid.
        assert ids[0, 0] == 0

    def test_multiple_queries(self):
        idx = InnerProductIndex(2)
        idx.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
        queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
        scores, ids = idx.search(queries, k=1)
        assert ids[0, 0] == 0
        assert ids[1, 0] == 1

    def test_save_and_load(self, tmp_path):
        idx = InnerProductIndex(4)
        vecs = np.random.randn(10, 4).astype(np.float32)
        idx.add(vecs)
        save_path = tmp_path / "test_index"
        idx.save(save_path)

        idx2 = InnerProductIndex(4)
        # Find the actual saved file (could be .npy or original path).
        if save_path.exists():
            idx2.load(save_path)
        else:
            idx2.load(save_path.with_suffix(".npy"))

        assert idx2.size == 10
        query = np.random.randn(1, 4).astype(np.float32)
        s1, i1 = idx.search(query, k=3)
        s2, i2 = idx2.search(query, k=3)
        np.testing.assert_array_equal(i1, i2)

    def test_save_empty_raises(self, tmp_path):
        idx = InnerProductIndex(3)
        with pytest.raises(RuntimeError, match="empty"):
            idx.save(tmp_path / "empty_index")

    def test_load_missing_raises(self, tmp_path):
        idx = InnerProductIndex(3)
        with pytest.raises(FileNotFoundError):
            idx.load(tmp_path / "nonexistent")


class TestL2Index:
    """Tests for L2Index (Euclidean distance search)."""

    def test_basic_search(self):
        idx = L2Index(2)
        vecs = np.array([[0, 0], [3, 4], [1, 0]], dtype=np.float32)
        idx.add(vecs)
        query = np.array([[0, 0]], dtype=np.float32)
        dists, ids = idx.search(query, k=2)
        # Nearest to origin is [0,0] itself, then [1,0].
        assert ids[0, 0] == 0
        assert ids[0, 1] == 2

    def test_dimension_mismatch_raises(self):
        idx = L2Index(3)
        with pytest.raises(ValueError):
            idx.add(np.array([[1, 0]], dtype=np.float32))

    def test_invalid_dimension(self):
        with pytest.raises(ValueError, match="positive"):
            L2Index(-1)

    def test_search_empty_raises(self):
        idx = L2Index(2)
        with pytest.raises(RuntimeError, match="empty"):
            idx.search(np.array([[1, 0]], dtype=np.float32), k=1)

    def test_size_property(self):
        idx = L2Index(2)
        assert idx.size == 0
        idx.add(np.array([[1, 0]], dtype=np.float32))
        assert idx.size == 1


class TestIVFInnerProductIndex:
    """Tests for IVFInnerProductIndex (approximate search)."""

    def test_train_add_search(self):
        dim = 8
        np.random.seed(42)
        train_vecs = np.random.randn(200, dim).astype(np.float32)
        idx = IVFInnerProductIndex(dim, nlist=4, nprobe=2)
        idx.train(train_vecs)
        idx.add(train_vecs)
        assert idx.size == 200

        query = np.random.randn(1, dim).astype(np.float32)
        scores, ids = idx.search(query, k=5)
        assert scores.shape == (1, 5)
        assert ids.shape == (1, 5)

    def test_add_before_train_raises(self):
        idx = IVFInnerProductIndex(4, nlist=2)
        # Without FAISS, the fallback is always "trained", so this test
        # only verifies behavior when the real FAISS is present.
        if idx._index is not None:
            with pytest.raises(RuntimeError):
                idx.add(np.random.randn(5, 4).astype(np.float32))

    def test_invalid_params(self):
        with pytest.raises(ValueError, match="positive"):
            IVFInnerProductIndex(0)
        with pytest.raises(ValueError, match="nlist"):
            IVFInnerProductIndex(4, nlist=0)


class TestBuildIndex:
    """Tests for the build_index convenience function."""

    def test_flat_ip(self):
        vecs = np.random.randn(20, 4).astype(np.float32)
        idx = build_index(vecs, index_type="flat_ip")
        assert isinstance(idx, InnerProductIndex)
        assert idx.size == 20

    def test_flat_l2(self):
        vecs = np.random.randn(20, 4).astype(np.float32)
        idx = build_index(vecs, index_type="flat_l2")
        assert isinstance(idx, L2Index)
        assert idx.size == 20

    def test_ivf_ip(self):
        vecs = np.random.randn(200, 8).astype(np.float32)
        idx = build_index(vecs, index_type="ivf_ip", nlist=4)
        assert isinstance(idx, IVFInnerProductIndex)
        assert idx.size == 200

    def test_unknown_type_raises(self):
        vecs = np.random.randn(5, 2).astype(np.float32)
        with pytest.raises(ValueError, match="unknown"):
            build_index(vecs, index_type="xyz")


# ===================================================================
#  METRICS TESTS
# ===================================================================


class TestRecallAtK:
    """Tests for recall_at_k."""

    def test_perfect_recall(self):
        retrieved = [["a", "b", "c"], ["d", "e", "f"]]
        relevant = [{"a"}, {"d"}]
        assert recall_at_k(retrieved, relevant, k=1) == 1.0

    def test_zero_recall(self):
        retrieved = [["x", "y"], ["z", "w"]]
        relevant = [{"a"}, {"b"}]
        assert recall_at_k(retrieved, relevant, k=2) == 0.0

    def test_partial_recall(self):
        retrieved = [["a", "b"], ["x", "y"]]
        relevant = [{"a"}, {"b"}]
        assert recall_at_k(retrieved, relevant, k=1) == 0.5

    def test_recall_at_k_cutoff(self):
        retrieved = [["x", "a"]]
        relevant = [{"a"}]
        assert recall_at_k(retrieved, relevant, k=1) == 0.0  # a not in top-1
        assert recall_at_k(retrieved, relevant, k=2) == 1.0  # a in top-2

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="positive"):
            recall_at_k([["a"]], [{"a"}], k=0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            recall_at_k([["a"]], [{"a"}, {"b"}], k=1)

    def test_empty_lists(self):
        assert recall_at_k([], [], k=5) == 0.0


class TestMRR:
    """Tests for mean_reciprocal_rank."""

    def test_perfect_mrr(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"c"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_second_position(self):
        retrieved = [["x", "a"]]
        relevant = [{"a"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.5

    def test_no_relevant(self):
        retrieved = [["x", "y"]]
        relevant = [{"a"}]
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_mixed(self):
        retrieved = [["a", "b"], ["x", "c"]]
        relevant = [{"a"}, {"c"}]
        mrr = mean_reciprocal_rank(retrieved, relevant)
        assert mrr == pytest.approx((1.0 + 0.5) / 2)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            mean_reciprocal_rank([["a"]], [{"a"}, {"b"}])


class TestTopKAccuracy:
    """Tests for top_k_accuracy (alias of recall_at_k)."""

    def test_same_as_recall(self):
        retrieved = [["a", "b"], ["c", "d"]]
        relevant = [{"a"}, {"x"}]
        assert top_k_accuracy(retrieved, relevant, k=1) == recall_at_k(
            retrieved, relevant, k=1
        )


class TestNDCG:
    """Tests for ndcg_at_k."""

    def test_perfect_ndcg(self):
        retrieved = [["a", "b"]]
        relevant = [{"a"}]
        assert ndcg_at_k(retrieved, relevant, k=1) == 1.0

    def test_ndcg_second_position(self):
        retrieved = [["x", "a"]]
        relevant = [{"a"}]
        result = ndcg_at_k(retrieved, relevant, k=2)
        # DCG = 0/log2(2) + 1/log2(3), IDCG = 1/log2(2) + 0/log2(3)
        expected = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_no_relevant(self):
        retrieved = [["x", "y"]]
        relevant = [{"a"}]
        assert ndcg_at_k(retrieved, relevant, k=2) == 0.0

    def test_invalid_k(self):
        with pytest.raises(ValueError, match="positive"):
            ndcg_at_k([["a"]], [{"a"}], k=0)


class TestLatencyStats:
    """Tests for LatencyStats dataclass."""

    def test_as_dict(self):
        stats = LatencyStats(
            mean_ms=1.5, median_ms=1.2, p95_ms=2.5,
            p99_ms=3.0, total_ms=15.0, num_queries=10,
        )
        d = stats.as_dict()
        assert d["mean_ms"] == 1.5
        assert d["num_queries"] == 10
        assert "raw_ms" not in d  # raw data excluded from dict


class TestMeasureSearchLatency:
    """Tests for measure_search_latency."""

    def test_basic_latency(self):
        idx = InnerProductIndex(4)
        idx.add(np.random.randn(50, 4).astype(np.float32))
        queries = np.random.randn(5, 4).astype(np.float32)
        stats = measure_search_latency(idx, queries, k=3, num_runs=2, warmup_runs=1)
        assert stats.num_queries == 10  # 5 queries × 2 runs
        assert stats.mean_ms > 0
        assert stats.median_ms > 0
        assert len(stats.raw_ms) == 10


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_summary(self):
        result = EvaluationResult(
            method="dpr",
            recall={5: 0.65, 20: 0.82},
            mrr=0.55,
            ndcg={5: 0.60},
        )
        s = result.summary()
        assert s["method"] == "dpr"
        assert s["recall@5"] == 0.65
        assert s["recall@20"] == 0.82
        assert s["mrr"] == 0.55
        assert s["ndcg@5"] == 0.6

    def test_summary_with_latency(self):
        latency = LatencyStats(
            mean_ms=1.0, median_ms=0.8, p95_ms=2.0,
            p99_ms=3.0, total_ms=5.0, num_queries=5,
        )
        result = EvaluationResult(method="test", latency=latency)
        s = result.summary()
        assert "latency_mean_ms" in s


# ===================================================================
#  EVALUATOR TESTS
# ===================================================================


class TestRetrievalEvaluator:
    """Tests for the end-to-end RetrievalEvaluator."""

    @pytest.fixture()
    def simple_setup(self):
        """Create a simple 5-passage, 2-query scenario."""
        np.random.seed(0)
        dim = 8
        passage_vecs = np.eye(dim, dtype=np.float32)[:5]  # 5 basis vectors
        passage_ids = [f"p{i}" for i in range(5)]
        # Queries that are close to specific passages.
        query_vecs = np.array([
            [0.9, 0.1, 0, 0, 0, 0, 0, 0],   # closest to p0
            [0, 0, 0, 0.1, 0.9, 0, 0, 0],     # closest to p4
        ], dtype=np.float32)
        gold = [{"p0"}, {"p4"}]
        return passage_vecs, passage_ids, query_vecs, gold

    def test_evaluate_basic(self, simple_setup):
        passage_vecs, passage_ids, query_vecs, gold = simple_setup
        evaluator = RetrievalEvaluator(
            passage_vectors=passage_vecs,
            passage_ids=passage_ids,
            query_vectors=query_vecs,
            gold_relevant=gold,
        )
        result = evaluator.evaluate(
            top_k=[1, 5],
            method_name="test_dpr",
            measure_latency=False,
        )
        assert result.method == "test_dpr"
        assert result.recall[1] == 1.0  # Both queries find their doc at rank 1
        assert result.recall[5] == 1.0
        assert result.mrr == 1.0

    def test_evaluate_with_latency(self, simple_setup):
        passage_vecs, passage_ids, query_vecs, gold = simple_setup
        evaluator = RetrievalEvaluator(
            passage_vectors=passage_vecs,
            passage_ids=passage_ids,
            query_vectors=query_vecs,
            gold_relevant=gold,
        )
        result = evaluator.evaluate(
            top_k=[1],
            measure_latency=True,
            latency_runs=1,
        )
        assert result.latency is not None
        assert result.latency.mean_ms > 0

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="passage_vectors"):
            RetrievalEvaluator(
                passage_vectors=np.zeros((5, 4), dtype=np.float32),
                passage_ids=["a", "b"],  # wrong length
                query_vectors=np.zeros((1, 4), dtype=np.float32),
                gold_relevant=[{"a"}],
            )
        with pytest.raises(ValueError, match="query_vectors"):
            RetrievalEvaluator(
                passage_vectors=np.zeros((2, 4), dtype=np.float32),
                passage_ids=["a", "b"],
                query_vectors=np.zeros((3, 4), dtype=np.float32),
                gold_relevant=[{"a"}],  # wrong length
            )

    def test_evaluate_l2_index(self, simple_setup):
        passage_vecs, passage_ids, query_vecs, gold = simple_setup
        evaluator = RetrievalEvaluator(
            passage_vectors=passage_vecs,
            passage_ids=passage_ids,
            query_vectors=query_vecs,
            gold_relevant=gold,
            index_type="flat_l2",
        )
        result = evaluator.evaluate(
            top_k=[1], measure_latency=False,
        )
        assert result.recall[1] == 1.0


class TestCompareAndSave:
    """Tests for compare_methods, save_results, load_results."""

    def test_compare_methods(self):
        r1 = EvaluationResult(method="bm25", recall={5: 0.5}, mrr=0.4)
        r2 = EvaluationResult(method="dpr", recall={5: 0.8}, mrr=0.7)
        rows = compare_methods([r1, r2])
        assert len(rows) == 2
        assert rows[0]["method"] == "bm25"
        assert rows[1]["method"] == "dpr"

    def test_save_and_load(self, tmp_path):
        r = EvaluationResult(method="dpr", recall={5: 0.8, 20: 0.9}, mrr=0.7)
        path = tmp_path / "results.json"
        save_results([r], path)
        assert path.exists()
        loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0]["method"] == "dpr"
        assert loaded[0]["recall@5"] == 0.8


# ===================================================================
#  BACKWARD-COMPATIBILITY: Original minimal test preserved
# ===================================================================


def test_inner_product_index_and_recall():
    """Original test from the repository – kept for backward compatibility."""
    index = InnerProductIndex(2)
    index.add(np.array([[1, 0], [0, 1]], dtype=np.float32))
    _, ids = index.search(np.array([[0.9, 0.1]], dtype=np.float32), 1)
    assert ids.tolist() == [[0]]
    assert recall_at_k([["p0"]], [["p0"]], 1) == 1.0
