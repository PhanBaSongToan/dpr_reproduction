"""Convert the original authors' biencoder NQ dev JSON into a local passage pool.

The source file has ``question``, ``answers``, ``positive_ctxs``,
``negative_ctxs`` and ``hard_negative_ctxs``. Source values are preserved.
A subset of context pools is NOT the paper's 21M-passage evaluation corpus.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def iter_records(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as stream:
        try:
            import ijson
        except ImportError:
            # Small fixture fallback only. The official dataset needs streaming ijson.
            if Path(path).stat().st_size > 2_000_000:
                raise RuntimeError('Install ijson: python -m pip install ijson')
            yield from json.load(stream)
        else:
            yield from ijson.items(stream, 'item')


def prepare(source: str, out_dir: str, limit_queries: int = 100,
            limit_passages: int = 10000) -> dict:
    if limit_queries < 1 or limit_passages < 1:
        raise ValueError("Limits must be positive")
    output = Path(out_dir); output.mkdir(parents=True, exist_ok=True)
    # Stable content-derived IDs. The paper's Wikipedia passage IDs, if present,
    # are not assumed to map across different preprocessing variants.
    passages = {}  # id -> content
    queries = []
    for rec in iter_records(source):
        if len(queries) >= limit_queries:
            break
        q = rec.get('question')
        positives = rec.get('positive_ctxs') or []
        if not q or not positives:
            continue
        pids = []
        for ctx in positives:
            title = str(ctx.get('title') or '')
            text = str(ctx.get('text') or '')
            if not text:
                continue
            pid = hashlib.sha256((title+'\n'+text).encode('utf-8')).hexdigest()[:24]
            passages.setdefault(pid, {'passage_id': pid, 'title': title, 'text': text,
                                      'source_passage_id': ctx.get('passage_id')})
            pids.append(pid)
        if not pids:
            continue
        if len(passages) > limit_passages:
            raise ValueError('limit_passages is smaller than unique positive passages; increase it')
        for typ in ('negative_ctxs', 'hard_negative_ctxs'):
            for ctx in rec.get(typ) or []:
                title = str(ctx.get('title') or '')
                text = str(ctx.get('text') or '')
                if not text:
                    continue
                pid = hashlib.sha256((title+'\n'+text).encode('utf-8')).hexdigest()[:24]
                if len(passages) < limit_passages or pid in passages:
                    passages.setdefault(pid, {'passage_id': pid, 'title': title, 'text': text,
                                              'source_passage_id': ctx.get('passage_id')})
        queries.append({'query_id': f'nq-dev-{len(queries)+1}', 'question': q,
                        'answers': list(rec.get('answers') or []),
                        'positive_passage_ids': list(dict.fromkeys(pids))})
    if not queries:
        raise ValueError('No questions with positive passages found')
    for path, entries in [('passages.jsonl', passages.values()), ('queries.jsonl', queries)]:
        with (output / path).open('w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    manifest = {'source_file': str(source), 'source_kind': 'Official DPR biencoder NQ-dev',
                'corpus_kind': 'Pooled contexts from selected questions; NOT full Wikipedia',
                'questions': len(queries), 'passages': len(passages),
                'limit_queries': limit_queries, 'limit_passages': limit_passages,
                'passage_ids': 'content-hash; not original Wikipedia passage IDs'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/raw/biencoder-nq-dev.json.gz')
    ap.add_argument('--output-dir', default='data/processed/nq_bm25')
    ap.add_argument('--max-queries', type=int, default=100)
    ap.add_argument('--max-passages', type=int, default=10000)
    a = ap.parse_args()
    print(json.dumps(prepare(a.input,a.output_dir,a.max_queries,a.max_passages), indent=2))

if __name__ == '__main__':
    main()
