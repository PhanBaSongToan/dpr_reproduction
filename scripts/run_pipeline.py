"""Run the reproducible DPR pipeline.

The pipeline uses a small JSONL interchange format:

``passages.jsonl``: ``{"passage_id": "p1", "text": "..."}``
``qa.jsonl``: ``{"question_id": "q1", "question": "...", "answers": ["..."], "positive_passage_id": "p1"}``

Use ``--demo`` for a deterministic offline run through preprocessing, BM25,
training, FAISS indexing, and evaluation. A real experiment can provide one
JSON object per line with passage and QA fields via ``--input-jsonl``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_loader import load_jsonl, save_jsonl
from src.data.preprocess import normalize_text
from src.evaluation.evaluator import RetrievalEvaluator
from src.models.dpr_encoder import DPRDualEncoder
from src.retrieval.bm25 import BM25Retriever
from src.training.trainer import DPRTrainer


@dataclass
class PipelinePaths:
    root: Path

    @property
    def passages(self) -> Path:
        return self.root / "passages.jsonl"

    @property
    def qa(self) -> Path:
        return self.root / "qa.jsonl"

    @property
    def checkpoint(self) -> Path:
        return self.root / "checkpoints" / "dpr.pt"

    @property
    def vectors(self) -> Path:
        return self.root / "index" / "vectors.npz"

    @property
    def results(self) -> Path:
        return self.root / "results.json"


class HashEncoder(nn.Module):
    """Small offline encoder used only by the deterministic demo."""

    def __init__(self, vocab_size: int = 4096, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None):
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            hidden = hidden.clone()
            hidden[:, 0, :] = pooled
        return SimpleNamespace(last_hidden_state=hidden)


class SimpleTokenizer:
    """Stable whitespace tokenizer for the offline demo."""

    def __init__(self, vocab_size: int = 4096, max_length: int = 32):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __call__(self, texts: Iterable[str]) -> dict[str, Tensor]:
        rows, masks = [], []
        for text in texts:
            tokens = re.findall(r"\w+", text.casefold())[: self.max_length]
            ids = [self._token_id(token) for token in tokens] or [1]
            mask = [1] * len(ids)
            ids += [0] * (self.max_length - len(ids))
            mask += [0] * (self.max_length - len(mask))
            rows.append(ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def _token_id(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return 1 + int.from_bytes(digest[:4], "big") % (self.vocab_size - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def demo_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    topics = [
        ("python", "Python is a programming language used for software development.", "What is Python?"),
        ("faiss", "FAISS is a library for efficient similarity search over vectors.", "What is FAISS used for?"),
        ("bm25", "BM25 is a lexical ranking algorithm for information retrieval.", "What is BM25?"),
        ("dpr", "DPR uses question and passage encoders for dense retrieval.", "What does DPR use?"),
        ("wikipedia", "Wikipedia is a free online encyclopedia with many articles.", "What is Wikipedia?"),
        ("transformer", "A transformer is a neural architecture based on attention.", "What is a transformer?"),
    ]
    passages = [{"passage_id": f"p{i}", "text": text} for i, (_, text, _) in enumerate(topics)]
    passages.extend(
        {"passage_id": f"p{len(passages) + i}", "text": f"General information about {topic} and research."}
        for i, (topic, _, _) in enumerate(topics)
    )
    qa = [
        {"question_id": f"q{i}", "question": question, "answers": [topic], "positive_passage_id": f"p{i}"}
        for i, (topic, _, question) in enumerate(topics)
    ]
    return passages, qa


def load_records(paths: PipelinePaths) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    passages = list(load_jsonl(str(paths.passages)))
    qa = list(load_jsonl(str(paths.qa)))
    if not passages or not qa:
        raise ValueError("Both passages.jsonl and qa.jsonl must contain records")
    return passages, qa


def preprocess(args: argparse.Namespace, paths: PipelinePaths) -> None:
    if args.demo:
        passages, qa = demo_records()
    elif args.input_jsonl:
        passages, qa = [], []
        for record in load_jsonl(args.input_jsonl):
            if "passage_id" in record and "text" in record:
                passages.append({"passage_id": str(record["passage_id"]), "text": normalize_text(str(record["text"]))})
            if "question" in record and ("question_id" in record or "id" in record):
                item = dict(record)
                item["question_id"] = str(item.get("question_id", item.get("id")))
                item["question"] = str(item["question"]).strip()
                item["answers"] = item.get("answers", [])
                if "positive_passage_id" not in item:
                    raise ValueError("QA records require positive_passage_id")
                qa.append(item)
        if not passages or not qa:
            raise ValueError("Input JSONL must contain passage and QA fields")
    else:
        raise ValueError("Use --demo or provide --input-jsonl")
    passages = passages[: args.max_passages]
    qa = qa[: args.max_queries]
    valid_ids = {str(p["passage_id"]) for p in passages}
    qa = [q for q in qa if str(q["positive_passage_id"]) in valid_ids]
    save_jsonl(passages, str(paths.passages))
    save_jsonl(qa, str(paths.qa))
    print(f"Preprocessed {len(passages)} passages and {len(qa)} QA pairs")


def _retrieval_metrics(retrieved: list[list[str]], relevant: list[set[str]], ks: tuple[int, ...]) -> dict[str, Any]:
    recalls = {f"recall@{k}": sum(bool(set(row[:k]) & gold) for row, gold in zip(retrieved, relevant)) / max(len(relevant), 1) for k in ks}
    reciprocal = []
    for row, gold in zip(retrieved, relevant):
        ranks = [i + 1 for i, pid in enumerate(row) if pid in gold]
        reciprocal.append(1.0 / min(ranks) if ranks else 0.0)
    return {**recalls, "mrr": sum(reciprocal) / max(len(reciprocal), 1)}


def bm25_stage(paths: PipelinePaths, top_k: int = 100) -> dict[str, Any]:
    passages, qa = load_records(paths)
    ids = [str(p["passage_id"]) for p in passages]
    retriever = BM25Retriever([str(p["text"]) for p in passages], ids)
    retrieved = [[pid for pid, _ in retriever.search(str(item["question"]), k=top_k)] for item in qa]
    relevant = [{str(item["positive_passage_id"])} for item in qa]
    hard_negatives = [
        retriever.hard_negatives(str(item["question"]), {str(item["positive_passage_id"])}, item.get("answers", []), 1)
        for item in qa
    ]
    result = {"method": "bm25", **_retrieval_metrics(retrieved, relevant, (5, 20, 100)), "hard_negative_count": sum(bool(x) for x in hard_negatives)}
    paths.root.mkdir(parents=True, exist_ok=True)
    (paths.root / "bm25_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def _build_model(args: argparse.Namespace) -> tuple[DPRDualEncoder, Any]:
    if args.backend == "hf":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        return DPRDualEncoder.from_pretrained(args.model_name), tokenizer
    tokenizer = SimpleTokenizer(max_length=args.max_length)
    model = DPRDualEncoder(HashEncoder(hidden_size=args.hidden_size), HashEncoder(hidden_size=args.hidden_size))
    return model, tokenizer


def _tokenize(tokenizer: Any, texts: list[str]) -> dict[str, Tensor]:
    if tokenizer.__class__.__module__.startswith("transformers"):
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    else:
        encoded = tokenizer(texts)
    return {key: value for key, value in encoded.items() if isinstance(value, Tensor)}


def train_stage(args: argparse.Namespace, paths: PipelinePaths) -> list[float]:
    passages, qa = load_records(paths)
    by_id = {str(p["passage_id"]): str(p["text"]) for p in passages}
    model, tokenizer = _build_model(args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    trainer = DPRTrainer(model, optimizer, device=args.device)
    retriever = BM25Retriever(list(by_id.values()), list(by_id.keys()))
    batches = []
    for start in range(0, len(qa), args.batch_size):
        rows = qa[start : start + args.batch_size]
        questions = [str(row["question"]) for row in rows]
        positives = [by_id[str(row["positive_passage_id"])] for row in rows]
        hard_texts = []
        for i, row in enumerate(rows):
            candidates = retriever.hard_negatives(str(row["question"]), {str(row["positive_passage_id"])}, row.get("answers", []), 1)
            hard_texts.append(by_id[candidates[0]] if candidates else positives[i])
        batches.append({
            "question_inputs": _tokenize(tokenizer, questions),
            "positive_passage_inputs": _tokenize(tokenizer, positives),
            "hard_negative_inputs": _tokenize(tokenizer, hard_texts),
        })
    losses = trainer.fit(batches, epochs=args.epochs, checkpoint_dir=paths.checkpoint.parent)
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "losses": losses}, paths.checkpoint)
    print(json.dumps({"losses": losses, "checkpoint": str(paths.checkpoint)}, indent=2))
    return losses


def encode_texts(model: DPRDualEncoder, tokenizer: Any, texts: list[str], *, questions: bool, batch_size: int = 32) -> np.ndarray:
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            inputs = _tokenize(tokenizer, texts[start : start + batch_size])
            encoded = model.encode_questions(**inputs) if questions else model.encode_passages(**inputs)
            vectors.append(encoded.cpu().numpy().astype("float32"))
    return np.concatenate(vectors, axis=0)


def index_stage(args: argparse.Namespace, paths: PipelinePaths) -> tuple[np.ndarray, np.ndarray]:
    passages, qa = load_records(paths)
    model, tokenizer = _build_model(args)
    checkpoint = torch.load(paths.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    passage_vectors = encode_texts(model, tokenizer, [str(p["text"]) for p in passages], questions=False)
    query_vectors = encode_texts(model, tokenizer, [str(q["question"]) for q in qa], questions=True)
    passage_ids = np.asarray([str(p["passage_id"]) for p in passages])
    paths.vectors.parent.mkdir(parents=True, exist_ok=True)
    np.savez(paths.vectors, passage_vectors=passage_vectors, query_vectors=query_vectors, passage_ids=passage_ids)
    print(f"Indexed {len(passage_ids)} passages with dimension {passage_vectors.shape[1]}")
    return passage_vectors, query_vectors


def evaluate_stage(args: argparse.Namespace, paths: PipelinePaths) -> dict[str, Any]:
    passage_vectors, query_vectors = index_stage(args, paths)
    passages, qa = load_records(paths)
    ids = [str(p["passage_id"]) for p in passages]
    gold = [{str(q["positive_passage_id"])} for q in qa]
    evaluator = RetrievalEvaluator(passage_vectors, ids, query_vectors, gold, index_type=args.index_type)
    result = evaluator.evaluate(top_k=tuple(args.top_k), method_name="dpr_faiss")
    output: dict[str, Any] = {"dpr_faiss": result.summary()}
    bm25_file = paths.root / "bm25_results.json"
    if bm25_file.exists():
        output["bm25"] = json.loads(bm25_file.read_text(encoding="utf-8"))
    paths.results.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the end-to-end DPR reproduction pipeline.")
    parser.add_argument("--stage", choices=["preprocess", "bm25", "train", "index", "evaluate", "all"], default="all")
    parser.add_argument("--demo", action="store_true", help="Use a deterministic offline toy dataset")
    parser.add_argument("--input-jsonl", help="JSONL input containing passages and QA records")
    parser.add_argument("--output-dir", default="output/pipeline")
    parser.add_argument("--max-passages", type=int, default=10000)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--backend", choices=["toy", "hf"], default="toy")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--index-type", choices=["flat_ip", "ivf_ip"], default="flat_ip")
    parser.add_argument("--top-k", type=int, nargs="+", default=[5, 20, 100])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.stage in {"preprocess", "all"} and not args.demo and not args.input_jsonl:
        parser.error("provide --demo for an offline run or --input-jsonl for a real dataset")
    if args.backend == "hf" and args.demo:
        parser.error("--demo uses the toy backend; choose --backend hf with a real input dataset")
    seed_everything(args.seed)
    paths = PipelinePaths(Path(args.output_dir))
    if args.stage in {"preprocess", "all"}:
        preprocess(args, paths)
    if args.stage in {"bm25", "all"}:
        bm25_stage(paths, top_k=max(args.top_k))
    if args.stage in {"train", "all"}:
        train_stage(args, paths)
    if args.stage in {"index", "evaluate", "all"}:
        evaluate_stage(args, paths)
    print(f"Pipeline completed: {paths.root}")


if __name__ == "__main__":
    main()
