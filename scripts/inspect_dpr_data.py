
"""Inspect DPR Natural Questions data using streaming JSON."""

import gzip
from pathlib import Path

import ijson


def inspect_file(path):
    print(f"\nInspecting: {path.name}", flush=True)

    with gzip.open(path, "rb") as file:
        example = next(ijson.items(file, "item"), None)

    if example is None:
        print("No records found.")
        return

    print("Top-level keys:", list(example.keys()))
    print("Question:", example.get("question"))
    print("Answers:", example.get("answers"))

    for key in [
        "positive_ctxs",
        "negative_ctxs",
        "hard_negative_ctxs",
    ]:
        contexts = example.get(key, [])
        print(f"{key}: {len(contexts)}")

        if contexts:
            print("Context keys:", list(contexts[0].keys()))
            print("Title:", contexts[0].get("title"))
            print("Passage ID:", contexts[0].get("passage_id"))


def main():
    paths = sorted(Path("data/raw").rglob("nq-*.tmp"))

    for path in paths:
        inspect_file(path)


if __name__ == "__main__":
    main()
