import gzip
import json
from pathlib import Path

import pytest
from scripts.prepare_official_nq import prepare
from scripts.run_bm25_official import evaluate
from src.retrieval.bm25 import BM25Retriever, tokenize


def test_bm25():
    m=BM25Retriever(['football match team','machine learning model', 'football tactics'],['a','b','c'])
    assert tokenize('Machine-learning!') == ['machine','learning']
    assert m.search('machine learning')[0][0]=='b'
    assert 'b' not in m.hard_negatives('machine learning',{'b'},['machine learning'])
    assert m.search('???') == []
    with pytest.raises(ValueError): m.search('football',0)


def test_official_schema_pipeline(tmp_path):
    # A synthetic schema fixture tests the official format; it is NOT official data.
    source=tmp_path/'nq.json.gz'
    rows=[{'question':'Which planet has a ring?','answers':['Saturn'],
           'positive_ctxs':[{'title':'Saturn','text':'Saturn is a planet with rings.'}],
           'negative_ctxs':[{'title':'Football','text':'Football match on a field.'}],
           'hard_negative_ctxs':[{'title':'Neptune','text':'Neptune is an icy planet.'}]}]
    with gzip.open(source,'wt',encoding='utf-8') as f:json.dump(rows,f)
    prep=tmp_path/'processed'
    stats=prepare(str(source),str(prep),1,10)
    assert stats['questions']==1 and stats['passages']==3
    result=evaluate(str(prep),str(tmp_path/'output'))
    assert result['PositiveHit@5']==1.0
    assert result['AnswerHit@5']==1.0
    assert (tmp_path/'output'/'hard_negatives.jsonl').exists()
