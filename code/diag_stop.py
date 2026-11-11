"""diagnose self-termination of a residual reader on MuSiQue dev"""
import json, pickle, sys, collections
import numpy as np
R = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
L = pickle.load(open(sys.argv[1], 'rb'))
c = collections.Counter(); byhop = collections.defaultdict(collections.Counter)
for o, r in zip(L, R):
    G = set(r['gold']); m = len(G); T = len(o['top']) - 1
    t = o['halt'] if 'halt' in o else next((t for t in range(1, T + 1) if o['top'][t] > o['dmax'][t]), None)
    if 'verdict' in o:
        c['verdict_bot'] += o['verdict'] == 0
    # first step at which the greedy trajectory errs
    err = next((s for s in range(m) if o['order'][s] not in G), None)
    if t is None: k = 'nostop'
    elif G <= set(o['order'][:t]): k = 'ok_exact' if t == m else 'ok_over'
    elif err is not None and err < t: k = 'wrongpick'
    else: k = 'early'
    c[k] += 1; byhop[m][k] += 1
    c['traj_ok_m'] += G <= set(o['order'][:m])
    c['topmax_at_m'] += (o['top'][m] > o['dmax'][m]) if G <= set(o['order'][:m]) else 0
n = len(R)
print({k: round(100 * v / n, 1) for k, v in c.items()})
for m in sorted(byhop): print(m, {k: round(100 * v / sum(byhop[m].values()), 1) for k, v in byhop[m].items()})
