"""Evaluate BM25 on preprocessed DPR NQ data, output ranked passages & metrics.

All reported subset metrics are for the local pooled corpus, not paper reproduction.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
from src.retrieval.bm25 import BM25Retriever


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(folder: str, out_dir: str, k1: float = .9, b: float = .4):
    inp = Path(folder); out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    docs = read_jsonl(inp/'passages.jsonl'); qs = read_jsonl(inp/'queries.jsonl')
    if not docs or not qs:
        raise ValueError('Dataset is empty')
    ids = [str(d['passage_id']) for d in docs]
    content = [(str(d.get('title') or '')+' '+str(d['text'])).strip() for d in docs]
    index = BM25Retriever(content, ids, k1=k1,b=b)
    hit_pos = {5:0,20:0,100:0}; hit_ans = {5:0,20:0,100:0}
    rr = 0.; latency = 0.
    gold_ids_present = 0
    with (out/'topk.jsonl').open('w',encoding='utf-8') as ranks, (out/'hard_negatives.jsonl').open('w',encoding='utf-8') as negs:
        texts = dict(zip(ids, content))
        for row in qs:
            gold = {str(x) for x in row['positive_passage_ids']}
            answers = [str(a).casefold().strip() for a in row.get('answers',[]) if str(a).strip()]
            gold_ids_present += bool(gold & set(ids))
            start=time.perf_counter()
            results = index.search(row['question'], min(100,len(docs)))
            latency += time.perf_counter()-start
            found_gold = [i+1 for i,(pid,_) in enumerate(results) if pid in gold]
            answer_positions = [i+1 for i,(pid,_) in enumerate(results)
                                if answers and any(a in texts[pid].casefold() for a in answers)]
            if found_gold: rr += 1/found_gold[0]
            for k in hit_pos:
                hit_pos[k] += bool(found_gold and found_gold[0]<=k)
                hit_ans[k] += bool(answer_positions and answer_positions[0]<=k)
            ranks.write(json.dumps({'query_id':row['query_id'],'question':row['question'],
                    'results':[{'passage_id':pid,'score':score} for pid,score in results]},ensure_ascii=False)+'\n')
            negative_ids=index.hard_negatives(row['question'],gold,row.get('answers',[]),k=5)
            negs.write(json.dumps({'query_id':row['query_id'],'hard_negative_passage_ids':negative_ids},ensure_ascii=False)+'\n')
    n=len(qs)
    metrics={'queries':n,'passages':len(docs),'k1':k1,'b':b,
             'gold_in_corpus_rate':gold_ids_present/n,
             **{f'PositiveHit@{k}':hit_pos[k]/n for k in hit_pos},
             **{f'AnswerHit@{k}':hit_ans[k]/n for k in hit_ans},
             'MRR@100_positive':rr/n,'average_search_ms':1000*latency/n,
             'note':'Subset pooled contexts: NOT the paper full-Wikipedia scores'}
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['metric','value']);w.writerows(metrics.items())
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data-dir',default='data/processed/nq_bm25')
    ap.add_argument('--output-dir',default='outputs/bm25_nq')
    ap.add_argument('--k1',type=float,default=.9)
    ap.add_argument('--b',type=float,default=.4)
    a=ap.parse_args()
    print(json.dumps(evaluate(a.data_dir,a.output_dir,a.k1,a.b),indent=2))

if __name__ == '__main__':main()
