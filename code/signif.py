"""Paired significance (bootstrap CI of the CER difference, exact McNemar p) and held-out evaluation.
usage: python signif.py DS A B [keys]     uses results/sel/{DS}_{name}.json
Held-out: questions NOT in the 400-question checkpoint-selection sample (random.Random(0).sample(dev, 400))."""
import json, random, sys
import numpy as np
from scipy.stats import binomtest

ds, A, B = sys.argv[1:4]
keys = sys.argv[4].split(',') if len(sys.argv) > 4 else ['k2', 'k3', 'k4', 'k5', 't384', 't512', 't768']
recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
sel = random.Random(0).sample(recs, 400)
selid = {r['id'] for r in sel}
held = np.array([r['id'] not in selid for r in recs])
SA = json.load(open(f'results/sel/{ds}_{A}.json'))
SB = json.load(open(f'results/sel/{ds}_{B}.json'))
rng = np.random.default_rng(0)
out = {}
for k in keys:
    a = np.array([set(r['gold']) <= set(s) for s, r in zip(SA[k], recs)], dtype=float)
    b = np.array([set(r['gold']) <= set(s) for s, r in zip(SB[k], recs)], dtype=float)
    d = a - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    out[k] = {'diff': round(100 * d.mean(), 2), 'ci95': [round(100 * np.percentile(boot, 2.5), 2), round(100 * np.percentile(boot, 97.5), 2)],
              'p_mcnemar': float(f'{p:.2g}'), 'heldout_A': round(100 * a[held].mean(), 2), 'heldout_B': round(100 * b[held].mean(), 2)}
    print(k, out[k], flush=True)
json.dump(out, open(f'results/metrics/signif_{ds}_{A}_vs_{B}.json', 'w'), indent=1)
