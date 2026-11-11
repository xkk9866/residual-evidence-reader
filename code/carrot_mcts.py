"""CARROT-style budgeted chunk-combination search (port of wang0702/CARROT core/mcts.py):
reward of a combination = reranker(query, concatenation of its chunks); UCB with cost penalty;
best node over all depths that satisfies the token budget. Defaults follow CARROT's fallback config
(C=2.4, lambda=0.1, 10 iterations). Reranker: bge-reranker-v2-m3 (zero-shot) or a fine-tuned ckpt.
usage: python carrot_mcts.py DS B1,B2,... [ckpt]
"""
import json, math, pickle, sys, os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ds, budgets = sys.argv[1], [int(x) for x in sys.argv[2].split(',')]
ckpt = sys.argv[3] if len(sys.argv) > 3 else ''
tag = 'carrotft' if ckpt else 'carrot'
N_ITER, C, LAM = 10, 2.4, 0.1
name = 'BAAI/bge-reranker-v2-m3'
tk = AutoTokenizer.from_pretrained(name)
m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1)
if ckpt:
    sd = torch.load(ckpt, map_location='cpu')
    sd = {k[len('cls.'):]: v.float() for k, v in sd.items() if k.startswith('cls.')}
    m.load_state_dict(sd, strict=False)
m = m.half().cuda().eval()
recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]


class Node:
    def __init__(self, state, parent=None):
        self.state, self.parent, self.children = state, parent, []
        self.visits, self.total, self.score = 0, 0.0, None


@torch.no_grad()
def score(q, docs, states):
    texts = [' '.join(f"{docs[i]['title']}\n{docs[i]['text']}" for i in s) for s in states]
    out = []
    for i in range(0, len(texts), 16):
        b = tk([q] * len(texts[i:i + 16]), texts[i:i + 16], truncation='only_second', max_length=1024,
               padding=True, return_tensors='pt').to('cuda')
        out += m(**b).logits.view(-1).float().cpu().tolist()
    return out


def search(r, B):
    n = len(r['docs']); cost = r['ntok']
    root = Node([])
    allnodes = []

    def c_of(s):
        return sum(cost[i] for i in s)

    def ucb(nd, pv):
        if nd.visits == 0:
            return float('inf')
        return nd.total / nd.visits + C * math.sqrt(math.log(max(pv, 1)) / nd.visits) - LAM * c_of(nd.state) / B

    for _ in range(N_ITER):
        node = root
        while node.children and len(node.children) >= n - len(node.state):
            valid = [c for c in node.children if c_of(c.state) <= B] or node.children
            node = max(valid, key=lambda c: ucb(c, node.visits))
        avail = [i for i in range(n) if i not in node.state and c_of(node.state + [i]) <= B]
        new = [Node(node.state + [i], node) for i in avail if not any(c.state == node.state + [i] for c in node.children)]
        if not new:
            break
        node.children += new
        rw = score(r['question'], r['docs'], [x.state for x in new])
        for x, v in zip(new, rw):
            x.score = v
            allnodes.append(x)
            cur = x
            while cur is not None:
                cur.visits += 1; cur.total += v; cur = cur.parent
    ok = [x for x in allnodes if c_of(x.state) <= B]
    if not ok:
        return []
    return max(ok, key=lambda x: (x.score, x.visits, len(x.state))).state


res = {}
for B in budgets:
    res[f't{B}'] = [search(r, B) for r in recs]
    print(ds, tag, B, 'avg docs', np.mean([len(s) for s in res[f't{B}']]), flush=True)
os.makedirs('results/scores', exist_ok=True)
pickle.dump(res, open(f'results/scores/{ds}_{tag}.pkl', 'wb'))
