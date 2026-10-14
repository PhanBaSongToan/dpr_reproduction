
# Reproduction log

Fill this file during the experiment. Do not leave unknown values implicit.

| Field | Value |
|---|---|
| Dataset version | DPR-preprocessed Natural Questions (NQ), train/dev |
| Wikipedia version | TODO - To be confirmed by the team |
| Passage length | 100 words in the original DPR paper |
| Train/dev/test split | Development subset: 500 train / 100 dev questions; test not used |
| Encoder checkpoint | TODO - Model team |
| Batch size | TODO - Training team |
| Learning rate | TODO - Training team |
| Number of epochs | TODO - Training team |
| Hardware | TODO - To be recorded during experiments |
| Random seed | 42 for subset candidate sampling |
| Index type | TODO - Indexing team |

## Results

| Method | Recall@5 | Recall@20 | Recall@100 | Latency |
|---|---:|---:|---:|---:|
| BM25 | TODO | TODO | TODO | TODO |
| DPR | TODO | TODO | TODO | TODO |

Results will be updated after BM25 and DPR retrieval experiments.

## Deviations from the paper

The project uses a reduced development dataset for implementation,
testing, and debugging. Results on this subset must not be claimed
as an exact reproduction of the original DPR benchmark.

See the Issue #1 section below for dataset-specific deviations.

## Issue #1 - Dataset and Passage Preprocessing

### Dataset source

- Paper: Dense Passage Retrieval for Open-Domain Question Answering (Karpukhin et al., EMNLP 2020).
- Source: Official Facebook Research DPR repository.
- Dataset: Natural Questions (NQ), DPR-preprocessed retriever data.
- Splits: NQ train and NQ dev.
- Data format: Official DPR JSON; GZIP-compressed JSON is also supported.
- Parsing: Python gzip and ijson for streaming input.

### Subset configuration

- Random seed: 42 (used to shuffle candidate passage IDs).
- Training questions: 500.
- Evaluation questions: 100.
- Corpus size: 10,000 unique passages.
- Positive passages: 4,302.
- Additional candidate passages: 5,698.
- Question selection: First 500 training and first 100 development examples.
- Candidate passages: Selected from available negative_ctxs and hard_negative_ctxs.
- Purpose: Development, debugging, and comparison on a controlled subset.

### Preprocessing

1. Load official DPR-preprocessed Natural Questions records.
2. Extract questions, answers, and positive contexts.
3. Preserve original passage IDs, titles, and text.
4. Normalize Unicode to NFC when comparing records with duplicate passage IDs.
5. Deduplicate passages using passage IDs.
6. Include all positive passages from selected questions in the corpus.
7. Shuffle candidate passage IDs using seed 42.
8. Add negative and hard-negative candidate passages until the corpus contains 10,000 unique passages.
9. Validate positive passage mappings.
10. Export standardized JSONL files.

The original DPR passage splitting is not repeated because the
downloaded DPR retriever data already contains preprocessed passages.

The passage splitting utility is implemented and tested separately.

### Output

The generated development dataset contains:

- data/processed/corpus.jsonl: 10,000 unique passages.
- data/processed/train.jsonl: 500 training questions.
- data/processed/eval.jsonl: 100 evaluation questions.

Generated datasets are excluded from Git tracking.

### Validation

The following checks were completed successfully:

- Corpus record count: PASS.
- Unique passage IDs: PASS.
- Question ID uniqueness: PASS.
- Train/eval question ID disjointness: PASS.
- Positive passage ID mappings: PASS.
- Original dataset/preprocessing unit tests: 12 PASSED.
- Additional input-format compatibility tests: 5 PASSED.

Total: 17 relevant tests, consisting of 12 previously passing
tests and 5 newly passing tests.

The combined test suite should also be executed before final
submission to confirm there are no regressions.

### Deviations from the original DPR paper

- The development corpus contains 10,000 passages instead of approximately 21 million Wikipedia passages.
- The corpus is assembled from DPR QA positive, negative, and hard-negative contexts rather than sampled directly from the complete Wikipedia corpus.
- Only 500 training questions and 100 development questions are used.
- Source passages are already preprocessed by the DPR authors.
- The implemented passage splitting function is tested separately and is not reapplied to official DPR passages.
- Retrieval results on the 10K subset are not directly comparable to the paper's full-corpus benchmark.
- No BM25/DPR retrieval accuracy results have been established by Issue #1.

## PR #3 - Reproducibility Fix

### Problem identified during review

The initial subset preparation script expected compressed temporary files:

- nq-train.tmp
- nq-dev.tmp

These files were opened using gzip.open().

However, the official DPR downloader produces decompressed JSON files:

- biencoder-nq-train.json
- biencoder-nq-dev.json

As a result, the original script could fail on a clean setup
using the official DPR downloader.

### Implemented solution

Updated the subset preparation script to support both official
decompressed JSON files and compressed input formats.

Supported input filenames:

- biencoder-nq-train.json
- biencoder-nq-dev.json
- biencoder-nq-train.json.gz
- biencoder-nq-dev.json.gz
- nq-train.tmp
- nq-dev.tmp

The script:

1. Searches for supported input filenames.
2. Prioritizes official decompressed DPR JSON files.
3. Detects GZIP compression using file magic bytes.
4. Streams JSON records using ijson.
5. Preserves the existing subset configuration.
6. Maintains positive passage mappings and validation.

This allows users to reproduce the dataset preparation workflow
using the official DPR downloader output.

### Added input-format tests

Five tests were added in:

tests/test_dpr_input_formats.py

The tests cover:

1. Reading official uncompressed JSON.
2. Reading GZIP-compressed temporary files.
3. Reading official-style GZIP JSON.
4. Preferring official JSON filenames when available.
5. Raising FileNotFoundError when input files are missing.

Individual input-format test result:

5 PASSED.

### Reproduction commands

Install project dependencies:

    python -m pip install -r requirements.txt

Download the official DPR NQ train/dev data into data/raw/
using the official DPR downloader.

Prepare the development subset:

    python -m scripts.prepare_dpr_subset

Validate the generated dataset:

    python -m scripts.validate_dpr_subset

Run the dataset and preprocessing tests:

    python -m pytest tests/test_preprocess.py tests/test_dataset_loader.py tests/test_dpr_input_formats.py -q

### Development subset configuration

- Random seed: 42.
- Training questions: 500.
- Evaluation questions: 100.
- Total corpus: 10,000 passages.
- Positive passages: 4,302.

The reproducibility fix changes only input compatibility.
It does not change the intended subset size or evaluation scope.

### Current status

- Dataset preparation: COMPLETED.
- Dataset validation: PASSED.
- Input-format compatibility tests: 5 PASSED.
- Combined regression test run: 17 PASSED.
- Pull Request: #3.
- Review status: Changes requested; awaiting updated commit and re-review.
