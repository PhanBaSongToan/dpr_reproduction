# Issue #2 — BM25 Baseline với dữ liệu chính thức DPR Natural Questions

## Nguồn dữ liệu (KHÔNG tự tạo dữ liệu thực nghiệm)

Tác giả bài báo **Dense Passage Retrieval for Open-Domain Question Answering** công bố dữ liệu NQ ở:
https://github.com/facebookresearch/DPR

Resource map chính thức:
https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py

- NQ development: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
- NQ training: https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz
- Full Wikipedia passages: https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz (RẤT LỚN)

**Không có dữ liệu NQ nằm bên trong file PDF paper**. File PDF chỉ giải thích dataset và kết quả nghiên cứu.

## Cài đặt trên Windows PowerShell

Từ **thư mục gốc repository nhóm** sau khi chép 4 file `.py` vào đúng vị trí:

```powershell
python -m pip install ijson pytest
python -m pytest -q tests/test_bm25_official.py
python -m scripts.download_nq_official --resource nq-dev
python -m scripts.prepare_official_nq --input data/raw/biencoder-nq-dev.json.gz --max-queries 100 --max-passages 10000
python -m scripts.run_bm25_official --data-dir data/processed/nq_bm25
```

Chú ý: bộ code này tự triển khai BM25 trong Python chuẩn; **không cần rank-bm25**. Bộ lọc tokens và công thức IDF có thể khác Lucene paper.

## Kết quả sinh ra

- `data/processed/nq_bm25/passages.jsonl` và `queries.jsonl` dùng **nội dung NQ thật** từ resource chính thức.
- `data/processed/nq_bm25/manifest.json` ghi xuất xứ và số mẫu.
- `outputs/bm25_nq/topk.jsonl` xếp hạng top-100 passages.
- `outputs/bm25_nq/hard_negatives.jsonl` loại positive IDs và đoạn có answer string.
- `outputs/bm25_nq/metrics.csv` báo `PositiveHit@K`, `AnswerHit@K`, MRR và latency.

### Quan trọng khi báo cáo

1. Corpus 10K lấy từ các context pools thuộc NQ-dev, **không phải 21,015,324 Wikipedia passages** của bài báo.
2. Các query dùng để lấy corpus cũng dùng để tính metric, nghĩa là corpus *có thiên lệch lựa chọn*. Không được coi đây là kiểm chứng trên unseen corpus / kết quả so sánh trực tiếp với paper.
3. PositiveHit@K chỉ tính positive passage đã được gán; AnswerHit@K tính answer string ở retrieved passages, là heuristic.
4. Bài báo dùng **Lucene** BM25 (k1=0.9,b=0.4), dùng corpus Wikipedia đầy đủ. Code minh họa này dùng Python BM25 với cùng k1/b nhưng không đảm bảo parity với Lucene.
5. Không commit `data/raw/`, `data/processed/`, `outputs/` hay model weights lên GitHub. Chỉ push code/tests/docs.
6. Để evaluation mạnh hơn, dùng **corpus 10K do nhóm chia từ Wikipedia**, với QA queries độc lập và kiểm tra ID mapping.

## Tránh ghi đè code đồng đội

Các đường dẫn trong ZIP là patch tương ứng repository; chỉ thay `src/retrieval/bm25.py`, thêm `scripts/*.py`, `tests/test_bm25_official.py`, và README. Không xóa hoặc thay thế `src/data/`, `src/models/`, `src/training/`.

## GitHub

```powershell
git fetch origin
git switch feature/bm25-baseline
python -m pytest -q tests/test_bm25_official.py
git add src/retrieval/bm25.py scripts/download_nq_official.py scripts/prepare_official_nq.py scripts/run_bm25_official.py tests/test_bm25_official.py README_BM25_NQ.md
git commit -m "feat: run BM25 baseline on official DPR NQ data"
git push -u origin feature/bm25-baseline
```

Sau push, mở Pull Request vào `main`. Chỉ báo cáo số liệu thực tế sau khi chạy chương trình.
