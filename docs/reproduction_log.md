# Reproduction log

Fill this file during the experiment. Do not leave unknown values implicit.

| Field | Value |
|---|---|
| Dataset version | TODO |
| Wikipedia version | TODO |
| Passage length | 100 words in the paper |
| Train/dev/test split | TODO |
| Encoder checkpoint | TODO |
| Batch size | TODO |
| Learning rate | TODO |
| Number of epochs | TODO |
| Hardware | TODO |
| Random seed | TODO |
| Index type | TODO |

## Results

| Method | Recall@5 | Recall@20 | Recall@100 | Latency |
|---|---:|---:|---:|---:|
| BM25 | TODO | TODO | TODO | TODO |
| DPR | TODO | TODO | TODO | TODO |

## Deviations from the paper

TODO: document every dataset, preprocessing, model, hardware or metric difference.

## Issue #1 - Dataset and Passage Preprocessing

### Dataset source

- Paper: Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., EMNLP 2020).
- Source: Official Facebook Research DPR repository.
- Dataset: Natural Questions (NQ), DPR preprocessed retriever data.
- Splits: nq-train and nq-dev.
- Data format: GZIP-compressed JSON.
- Parsing: Python gzip + ijson (streaming).

### Subset configuration

- Random seed: 42 (used to shuffle candidate passage IDs).
- Training questions: 500.
- Evaluation questions: 100.
- Corpus size: 10,000 unique passages.
- Positive passages: 4,302.
- Additional candidate passages: 5,698.
- Question selection: First 500 training and first 100 development examples.
- Candidate passages: Sampled from available negative_ctxs and hard_negative_ctxs.

### Preprocessing

1. Load official DPR-preprocessed NQ records.
2. Extract question, answers, and positive contexts.
3. Preserve original passage IDs, titles, and text.
4. Normalize Unicode to NFC when comparing duplicate passage IDs.
5. Deduplicate passages by passage ID.
6. Include all selected positive passages in the corpus.
7. Add shuffled negative/hard-negative candidate passages until the corpus contains 10,000 unique passages.
8. Export standardized JSONL files.

### Output

- data/processed/corpus.jsonl: 10,000 passages.
- data/processed/train.jsonl: 500 questions.
- data/processed/eval.jsonl: 100 questions.

### Validation

- Corpus record count: PASS.
- Unique passage IDs: PASS.
- Question ID uniqueness: PASS.
- Train/eval question ID disjointness: PASS.
- Positive passage ID mapping: PASS.
- Dataset/preprocessing unit tests: 12 PASSED.

### Deviations from the original DPR paper

- Development corpus contains 10,000 passages instead of approximately 21 million Wikipedia passages.
- Corpus is assembled from DPR QA positive, negative, and hard-negative contexts, not randomly sampled from the full Wikipedia corpus.
- Only 500 training questions and 100 development questions are used.
- Source passages are already preprocessed by DPR authors.
- The implemented passage splitting function is tested separately and is not reapplied to official DPR passages.
- Retrieval results on this subset are not directly comparable to the paper's full-corpus benchmark.
- No BM25/DPR retrieval accuracy results have been established by Issue #1.

