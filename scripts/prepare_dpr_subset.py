
"""Prepare a reproducible 10K DPR Natural Questions development subset.

Supported inputs:
- Official DPR decompressed JSON:
  biencoder-nq-train.json / biencoder-nq-dev.json
- Compressed official-style JSON (.json.gz)
- GZIP temporary downloads:
  nq-train.tmp / nq-dev.tmp
"""

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

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")


def resolve_input_path(raw_dir, split):
    """Find the DPR dataset file, preferring official output names."""
    filenames = [
        f"biencoder-nq-{split}.json",
        f"biencoder-nq-{split}.json.gz",
        f"nq-{split}.tmp",
    ]

    for filename in filenames:
        matches = sorted(raw_dir.rglob(filename))

        if len(matches) > 1:
            raise ValueError(
                f"Multiple copies found for {filename}: {matches}"
            )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"DPR {split} dataset not found under {raw_dir}. "
        f"Expected one of: {', '.join(filenames)}"
    )


def is_gzip_file(path):
    """Detect GZIP compression using its magic bytes."""
    with open(path, "rb") as file:
        return file.read(2) == b"\x1f\x8b"


def read_questions(path, limit):
    """Stream JSON array records without loading the entire file."""
    if limit <= 0:
        raise ValueError("Question limit must be positive")

    opener = gzip.open if is_gzip_file(path) else open

    with opener(path, "rb") as file:
        for index, record in enumerate(ijson.items(file, "item")):
            if index >= limit:
                break

            yield index, record


def normalize_unicode(value):
    """Normalize text to Unicode NFC for comparison."""
    return unicodedata.normalize("NFC", value)


def make_passage(ctx):
    """Convert DPR context to a standard passage record."""
    return {
        "passage_id": str(ctx["passage_id"]),
        "title": ctx["title"],
        "text": ctx["text"],
    }


def passages_equivalent(first, second):
    """Check whether two records represent the same passage."""
    return (
        first["passage_id"] == second["passage_id"]
        and normalize_unicode(first["title"])
        == normalize_unicode(second["title"])
        and normalize_unicode(first["text"])
        == normalize_unicode(second["text"])
    )


def add_passage(store, passage, source):
    """Deduplicate by ID without silently accepting content conflicts."""
    pid = passage["passage_id"]

    if pid in store:
        previous = store[pid]

        if passages_equivalent(previous, passage):
            return

        raise ValueError(
            f"Conflicting passage ID after NFC normalization: "
            f"{pid} (source: {source})"
        )

    store[pid] = passage


def build_subset(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR):
    """Create train/eval QA files and a positive-preserving corpus."""
    rng = random.Random(SEED)

    paths = {
        split: resolve_input_path(raw_dir, split)
        for split in ("train", "dev")
    }

    selected = {}
    positives = {}
    candidates = {}

    for split, limit in (
        ("train", TRAIN_QUESTIONS),
        ("dev", DEV_QUESTIONS),
    ):
        print(
            f"\nProcessing {split}: {paths[split]}",
            flush=True,
        )

        qa_records = []

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
            f"Positive passages exceed corpus limit: "
            f"{len(positives)} > {MAX_PASSAGES}"
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

    if len(corpus) != MAX_PASSAGES:
        raise ValueError(
            f"Insufficient unique passages: "
            f"{len(corpus)} / {MAX_PASSAGES}"
        )

    corpus_ids = set(corpus)

    for split, records in selected.items():
        for record in records:
            positive_ids = set(record["positive_passage_ids"])

            if not positive_ids.issubset(corpus_ids):
                raise ValueError(
                    f"Missing positive passage for "
                    f"{record['question_id']}"
                )

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


def main():
    build_subset()


if __name__ == "__main__":
    main()
