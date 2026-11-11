"""Zero-shot pool scorers: BM25, bge-large-en-v1.5 (dense, also gives embeddings for MMR/DPP),
bge-reranker-v2-m3 (cross-encoder). Output: results/scores/{ds}_{method}.pkl = list of {'rho': (N,)}"""
import json, os, pickle, sys, re
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

ds = sys.argv[1]
methods = sys.argv[2].split(',') if len(sys.argv) > 2 else ['bm25', 'dense', 'rerank']
recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
os.makedirs('results/scores', exist_ok=True)
dev = 'cuda'


def dstr(d):
    return f"{d['title']}\n{d['text']}"


def toks(s):
    return re.findall(r'\w+', s.lower())


if 'bm25' in methods:
    out = []
    for r in recs:
        bm = BM25Okapi([toks(dstr(d)) for d in r['docs']])
        out.append({'rho': np.asarray(bm.get_scores(toks(r['question'])), dtype=np.float32)})
    pickle.dump(out, open(f'results/scores/{ds}_bm25.pkl', 'wb'))
    print('bm25 done', flush=True)

if 'dense' in methods:
    name = 'BAAI/bge-large-en-v1.5'
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()

    @torch.no_grad()
    def emb(texts):
        outs = []
        for i in range(0, len(texts), 128):
            b = tk(texts[i:i + 128], truncation=True, max_length=512, padding=True, return_tensors='pt').to(dev)
            h = m(**b).last_hidden_state[:, 0]
            outs.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu())
        return torch.cat(outs).numpy()
    qe = emb(['Represent this sentence for searching relevant passages: ' + r['question'] for r in recs])
    flat = [dstr(d) for r in recs for d in r['docs']]
    de = emb(flat)
    out, embs, j = [], [], 0
    for qi, r in enumerate(recs):
        e = de[j:j + len(r['docs'])]; j += len(r['docs'])
        out.append({'rho': (e @ qe[qi]).astype(np.float32)})
        embs.append(e.astype(np.float16))
    pickle.dump(out, open(f'results/scores/{ds}_dense.pkl', 'wb'))
    pickle.dump(embs, open(f'results/scores/{ds}_emb.pkl', 'wb'))
    print('dense done', flush=True)
    del m

if 'rerank' in methods:
    name = 'BAAI/bge-reranker-v2-m3'
    tk = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.float16).to(dev).eval()
    out = []
    with torch.no_grad():
        for r in recs:
            b = tk([r['question']] * len(r['docs']), [dstr(d) for d in r['docs']], truncation='only_second',
                   max_length=512, padding=True, return_tensors='pt').to(dev)
            out.append({'rho': m(**b).logits.squeeze(-1).float().cpu().numpy()})
    pickle.dump(out, open(f'results/scores/{ds}_rerank.pkl', 'wb'))
    print('rerank done', flush=True)
