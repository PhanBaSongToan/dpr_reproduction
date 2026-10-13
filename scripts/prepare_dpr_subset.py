

"""Create a DPR subset while preserving positive passage mappings."""

import gzip
import random
import unicodedata
from pathlib import Path

import ijson

from src.data.dataset_loader import save_jsonl


SEED = 42
MAX_PASSAGES = 10000
TRAIN_QUESTIONS = 500
DEV_QUESTIONS = 100


def read_questions(path, limit):
    """Stream DPR questions from a GZIP-compressed JSON array."""
    with gzip.open(path, "rb") as file:
        for index, record in enumerate(ijson.items(file, "item")):
            if index >= limit:
                break
            yield index, record


def make_passage(ctx):
    """Convert a DPR context into a standardized passage record."""
    return {
        "passage_id": str(ctx["passage_id"]),
        "title": ctx["title"],
        "text": ctx["text"],
    }


def normalize_unicode(text):
    """Normalize Unicode to NFC for consistent comparison."""
    return unicodedata.normalize("NFC", text)


def passages_equivalent(first, second):
    """Compare passage content after Unicode normalization."""
    return (
        first["passage_id"] == second["passage_id"]
        and normalize_unicode(first["title"])
        == normalize_unicode(second["title"])
        and normalize_unicode(first["text"])
        == normalize_unicode(second["text"])
    )


def add_passage(store, passage, source):
    """Add a passage while preserving official DPR passage IDs."""
    pid = passage["passage_id"]

    if pid in store:
        previous = store[pid]

        if passages_equivalent(previous, passage):
            return

        print("\n=== REAL PASSAGE ID CONFLICT ===")
        print("Source:", source)
        print("Passage ID:", pid)
        print(
            "Same normalized title:",
            normalize_unicode(previous["title"])
            == normalize_unicode(passage["title"]),
        )
        print(
            "Same normalized text:",
            normalize_unicode(previous["text"])
            == normalize_unicode(passage["text"]),
        )

        raise ValueError(
            f"Conflicting passage ID after Unicode normalization: {pid}"
        )

    store[pid] = passage


def main():
    rng = random.Random(SEED)
    raw_dir = Path("data/raw")

    paths = {}

    for split in ("train", "dev"):
        matches = sorted(raw_dir.rglob(f"nq-{split}.tmp"))

        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one nq-{split}.tmp, "
                f"found {len(matches)}"
            )

        paths[split] = matches[0]

    selected = {}
    positives = {}
    candidates = {}

    for split, limit in (
        ("train", TRAIN_QUESTIONS),
        ("dev", DEV_QUESTIONS),
    ):
        qa_records = []

        print(f"\nProcessing {split}...", flush=True)

        for index, example in read_questions(paths[split], limit):
            positive_ids = []

            for ctx in example.get("positive_ctxs", []):
                if not ctx.get("text", "").strip():
                    continue

                passage = make_passage(ctx)

                add_passage(
                    positives,
                    passage,
                    f"{split} positive question {index}",
                )

                positive_ids.append(passage["passage_id"])

            positive_ids = list(dict.fromkeys(positive_ids))

            if not positive_ids:
                continue

            qa_records.append({
                "question_id": f"nq_{split}_{index}",
                "question": example["question"],
                "answers": example["answers"],
                "positive_passage_ids": positive_ids,
            })

            for key in ("negative_ctxs", "hard_negative_ctxs"):
                for ctx in example.get(key, []):
                    if not ctx.get("text", "").strip():
                        continue

                    passage = make_passage(ctx)

                    add_passage(
                        candidates,
                        passage,
                        f"{split} {key} question {index}",
                    )

        selected[split] = qa_records

        print(
            f"{split}: {len(qa_records)} valid questions",
            flush=True,
        )

    if len(positives) > MAX_PASSAGES:
        raise ValueError(
            f"Too many positive passages: {len(positives)} "
            f"> {MAX_PASSAGES}"
        )

    corpus = dict(positives)

    candidate_ids = list(candidates.keys())
    rng.shuffle(candidate_ids)

    for pid in candidate_ids:
        if len(corpus) >= MAX_PASSAGES:
            break

        add_passage(
            corpus,
            candidates[pid],
            "corpus candidate",
        )

    if len(corpus) < MAX_PASSAGES:
        raise ValueError(
            f"Only {len(corpus)} unique passages available; "
            f"required {MAX_PASSAGES}"
        )

    corpus_ids = set(corpus.keys())

    for split, records in selected.items():
        for record in records:
            positive_ids = set(record["positive_passage_ids"])

            if not positive_ids.issubset(corpus_ids):
                raise ValueError(
                    "Missing positive passages for "
                    f"{record['question_id']}"
                )

    train_ids = {
        record["question_id"]
        for record in selected["train"]
    }

    dev_ids = {
        record["question_id"]
        for record in selected["dev"]
    }

    if train_ids.intersection(dev_ids):
        raise ValueError("Train and dev question IDs overlap")

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(
        corpus.values(),
        str(output_dir / "corpus.jsonl"),
    )

    save_jsonl(
        selected["train"],
        str(output_dir / "train.jsonl"),
    )

    save_jsonl(
        selected["dev"],
        str(output_dir / "eval.jsonl"),
    )

    print("\n=== DPR SUBSET SUMMARY ===")
    print(f"Seed: {SEED}")
    print(f"Train questions: {len(selected['train'])}")
    print(f"Dev questions: {len(selected['dev'])}")
    print(f"Positive passages: {len(positives)}")
    print(f"Corpus passages: {len(corpus)}")
    print("All positive passage mappings validated.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
