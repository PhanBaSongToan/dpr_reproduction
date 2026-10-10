# GitHub issue drafts

Create each section as a GitHub Issue and assign it to one member.

## Issue #1 - Dataset and passage preprocessing

Implement deterministic document cleaning, 100-word passage splitting, passage IDs, positive passage mapping and validation tests.

Acceptance criteria: processed sample is reproducible, schema is documented, and tests cover empty text and passage boundaries.

## Issue #2 - BM25 baseline

Implement BM25 retrieval, Top-k output and hard-negative export.

Acceptance criteria: baseline runs on the development subset and reports Recall@5/20/100.

## Issue #3 - DPR dual encoder

Implement independent question/passage encoders and dot-product score matrix.

Acceptance criteria: tensor shapes are validated and a small forward pass produces a BxB similarity matrix.

## Issue #4 - DPR training

Implement positive passages, in-batch negatives, one BM25 hard negative, NLL loss, checkpoint saving and a small training run.

Acceptance criteria: loss decreases on a controlled toy batch and the training configuration is saved.

## Issue #5 - FAISS and evaluation

Implement passage embedding indexing, Top-k search, Recall@K, MRR, latency measurement and result export.

Acceptance criteria: a toy index returns the expected nearest passage and the evaluation script writes a results table.

## Issue #6 - Reproduction run

Run BM25 and DPR on the agreed dataset split and compare with the paper.

Acceptance criteria: results, runtime, hardware, seed and deviations are recorded in `docs/reproduction_log.md`.
