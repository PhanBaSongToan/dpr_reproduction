# DPR Reproduction Project

Reproduction project for **Dense Passage Retrieval for Open-Domain Question Answering** (Karpukhin et al., EMNLP 2020).

## Assignment 2 scope

The team will implement and evaluate:

1. A BM25 baseline.
2. A DPR dual encoder with dot-product similarity.
3. In-batch negatives and one BM25 hard negative.
4. FAISS vector indexing and Top-k retrieval.
5. Retrieval evaluation against the paper's protocol.

The original paper uses English Wikipedia passages and open-domain QA datasets. Start with a reproducible subset of Natural Questions for development, then run the agreed final experiment on the full feasible split. Record any deviation from the paper in `docs/reproduction_log.md`.

## Team work allocation

| Member | Issue | Ownership |
|---|---:|---|
| Bảo Trân | #1 | Dataset and passage preprocessing |
| Quang Long | #2 | BM25 baseline |
| Thành Tài | #3 | DPR dual encoder |
| Song Toàn | #4 | Training and negative sampling |
| Bình An | #5 | FAISS indexing and evaluation |

Every contribution must include code, a test or validation command, and at least one focused commit.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run checks

```bash
pytest -q
python scripts/run_pipeline.py --help
```

## Run the integrated pipeline

The pipeline connects preprocessing, BM25 hard-negative generation, DPR
training, vector encoding, FAISS indexing, and retrieval evaluation. Start
with the deterministic offline smoke test:

```bash
python scripts/run_pipeline.py --demo --output-dir output/demo --epochs 2
```

It writes the prepared JSONL data, checkpoint, encoded vectors, BM25 results,
and the final comparison to `output/demo/results.json`.

For a real experiment, provide a JSONL file containing passage and QA records.
Each record may contain passage fields (`passage_id`, `text`), QA fields
(`question_id`, `question`, `answers`, `positive_passage_id`), or both:

```bash
python scripts/run_pipeline.py --input-jsonl data/nq_subset.jsonl \
  --output-dir output/nq --backend hf --model-name bert-base-uncased \
  --epochs 1 --batch-size 8
```

The `--demo` mode is for integration testing only; its metrics must not be
reported as the paper's Natural Questions results. Record the real dataset
version, checkpoint, split, and metric definition in
`docs/reproduction_log.md` before making a final claim.

## Development workflow

```bash
git checkout -b feature/<issue-name>
git add .
git commit -m "feat: <focused change>"
git push -u origin feature/<issue-name>
```

Open a Pull Request and link it to the issue with `Closes #<number>`.

## Reproduction metrics

- Recall@5, Recall@20, Recall@100.
- Top-k retrieval accuracy: whether a retrieved passage contains an answer.
- MRR where ranked relevance labels are available.
- Query latency and index build time.

Do not claim exact reproduction unless the dataset version, preprocessing, split, checkpoint, and metric definition match the paper.

## Paper reference

Karpukhin et al. 2020. *Dense Passage Retrieval for Open-Domain Question Answering*. EMNLP 2020.
