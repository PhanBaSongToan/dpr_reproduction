
"""Validate a generated DPR subset."""

from pathlib import Path

from src.data.dataset_loader import load_jsonl


def main():
    data_dir = Path("data/processed")

    corpus = list(load_jsonl(data_dir / "corpus.jsonl"))
    train = list(load_jsonl(data_dir / "train.jsonl"))
    eval_records = list(load_jsonl(data_dir / "eval.jsonl"))

    passage_ids = [p["passage_id"] for p in corpus]
    corpus_ids = set(passage_ids)

    assert len(corpus) == 10000
    assert len(train) == 500
    assert len(eval_records) == 100

    assert len(corpus_ids) == len(corpus), "Duplicate passage IDs"

    for passage in corpus:
        assert isinstance(passage["title"], str)
        assert isinstance(passage["text"], str)
        assert passage["text"].strip()

    train_ids = {q["question_id"] for q in train}
    eval_ids = {q["question_id"] for q in eval_records}

    assert len(train_ids) == len(train)
    assert len(eval_ids) == len(eval_records)
    assert train_ids.isdisjoint(eval_ids)

    for record in train + eval_records:
        assert record["question"].strip()
        assert record["answers"]
        assert record["positive_passage_ids"]

        for pid in record["positive_passage_ids"]:
            assert pid in corpus_ids, (
                f"Missing positive passage {pid}"
            )

    print("=== DPR SUBSET VALIDATION ===")
    print(f"Corpus: {len(corpus)} passages")
    print(f"Train: {len(train)} questions")
    print(f"Evaluation: {len(eval_records)} questions")
    print("Unique passage IDs: PASS")
    print("Question ID checks: PASS")
    print("Positive passage mappings: PASS")
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
