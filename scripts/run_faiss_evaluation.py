"""Run FAISS indexing and retrieval evaluation script.

Implements Issue #5 acceptance criteria:
- Builds FAISS index (or NumPy fallback)
- Performs Top-k search
- Evaluates Recall@K, MRR, NDCG@K and Latency
- Writes results table to CSV and JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Sequence

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.evaluator import RetrievalEvaluator, compare_methods, save_results
from src.evaluation.metrics import EvaluationResult


def run_evaluation(
    passage_vectors: np.ndarray,
    passage_ids: Sequence[str],
    query_vectors: np.ndarray,
    gold_relevant: Sequence[set[str] | list[str]],
    output_dir: str | Path,
    top_k: Sequence[int] = (1, 5, 20, 100),
    method_name: str = "dpr_faiss",
    index_type: str = "flat_ip",
) -> EvaluationResult:
    """Run evaluation and save results table."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    evaluator = RetrievalEvaluator(
        passage_vectors=passage_vectors,
        passage_ids=passage_ids,
        query_vectors=query_vectors,
        gold_relevant=gold_relevant,
        index_type=index_type,
    )

    result = evaluator.evaluate(
        top_k=top_k,
        method_name=method_name,
        measure_latency=True,
    )

    summary = result.summary()

    # Save JSON
    save_results([result], out_path / "evaluation_results.json")

    # Save CSV table
    csv_file = out_path / "evaluation_metrics.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print(f"Results saved to {out_path}")
    return result


def run_toy_demo(output_dir: str = "output/faiss_evaluation") -> EvaluationResult:
    """Run a reproducible toy demo satisfying Issue #5 criteria."""
    np.random.seed(42)
    dim = 64
    num_passages = 100
    num_queries = 20

    # Create synthetic normalized embeddings
    passage_vectors = np.random.randn(num_passages, dim).astype(np.float32)
    passage_vectors /= np.linalg.norm(passage_vectors, axis=1, keepdims=True)
    passage_ids = [f"passage_{i}" for i in range(num_passages)]

    # Queries with known nearest passage targets
    query_vectors = []
    gold_relevant = []
    for i in range(num_queries):
        target_idx = i % num_passages
        # Query is noisy version of the target passage
        noise = np.random.randn(dim).astype(np.float32) * 0.1
        q_vec = passage_vectors[target_idx] + noise
        q_vec /= np.linalg.norm(q_vec)
        query_vectors.append(q_vec)
        gold_relevant.append({passage_ids[target_idx]})

    query_vectors = np.array(query_vectors, dtype=np.float32)

    return run_evaluation(
        passage_vectors=passage_vectors,
        passage_ids=passage_ids,
        query_vectors=query_vectors,
        gold_relevant=gold_relevant,
        output_dir=output_dir,
        top_k=[1, 5, 20],
        method_name="toy_dpr_faiss",
    )


def main():
    parser = argparse.ArgumentParser(description="FAISS Indexing and Retrieval Evaluation")
    parser.add_argument("--output-dir", default="output/faiss_evaluation", help="Directory to save evaluation results")
    parser.add_argument("--toy", action="store_true", default=True, help="Run on toy dataset")
    args = parser.parse_args()

    result = run_toy_demo(args.output_dir)
    print("\n--- Evaluation Summary ---")
    print(json.dumps(result.summary(), indent=2))


if __name__ == "__main__":
    main()
