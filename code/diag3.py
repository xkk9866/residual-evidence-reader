"""Per-step diagnostics of residual completion vs a static ranking.
usage: python diag3.py DS lac_pred.pkl static_pred.pkl"""
import json, pickle, sys
import numpy as np

ds, lp, sp = sys.argv[1:4]
recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
L = pickle.load(open(lp, 'rb'))
St = pickle.load(open(sp, 'rb'))
stat = {}
for o, s, r in zip(L, St, recs):
    G = set(r['gold']); h = r['type'][:4] if r['type'][:1].isdigit() else str(len(G))
    so = list(np.argsort(-np.asarray(s['rho'])))
    d = stat.setdefault(h, {'p1_lac': [], 'p1_sta': [], 'exact_lac': [], 'exact_sta': [], 'step_acc': [], 'after_err': [],
                            'stop_ok': [], 'stop_len': []})
    d['p1_lac'].append(o['order'][0] in G); d['p1_sta'].append(so[0] in G)
    d['exact_lac'].append(G <= set(o['order'][:len(G)])); d['exact_sta'].append(G <= set(so[:len(G)]))
    S = []
    err = False
    for t in range(len(o['top']) - 1):
        nxt = o['order'][t]
        if not G <= set(S):
            ok = nxt in G
            d['step_acc'].append(ok)
            if err:
                d['after_err'].append(ok)
            err = err or not ok
        S.append(nxt)
    t = next((t for t in range(1, len(o['top'])) if max(o['top'][t], o['bot'][t]) > o['dmax'][t]), None)
    d['stop_ok'].append(t is not None and set(o['order'][:t]) == G)
    d['stop_len'].append(t if t is not None else -1)
for h, d in sorted(stat.items()):
    print(h, {k: round(100 * np.mean(v), 1) if k != 'stop_len' else round(float(np.mean(v)), 2) for k, v in d.items()}, len(d['p1_lac']))
