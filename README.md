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
| Member 1 | #1 | Dataset and passage preprocessing |
| Member 2 | #2 | BM25 baseline |
| Member 3 | #3 | DPR dual encoder |
| Member 4 | #4 | Training and negative sampling |
| Member 5 | #5 | FAISS indexing and evaluation |

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
