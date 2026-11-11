"""Paired comparison of two closed-loop pipelines on the same questions (each on its own final pool):
python code/loop_signif.py DS A_tag B_tag [key]   e.g. musique lac_adaptive static_adaptive k4
CER indicator per question (final pool contains every gold passage and the k-budget set contains them),
answer EM per question (Qwen3-8B), 95% paired bootstrap CI, exact McNemar p."""
import json, re, string, sys
import numpy as np
from scipy.stats import binomtest

ds, A, B = sys.argv[1:4]
key = sys.argv[4] if len(sys.argv) > 4 else 'k4'


def norm(s):
    s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def ind(tag):
    pool = f'loop_{ds}_{tag}_r2'
    R = [json.loads(l) for l in open(f'data/proc/{pool}.jsonl')]
    S = json.load(open(f'results/sel/{pool}_{tag}.json'))[key]
    cer = np.array([r['gold_missing'] == 0 and set(r['gold']) <= set(s) for r, s in zip(R, S)], dtype=float)
    try:
        q = json.load(open(f'results/qa/{pool}_{tag}_{key}_qwen3_8b.json'))
    except Exception:
        q = None
    em = np.array([max(float(norm(p) == norm(g)) for g in r['answers']) for p, r in zip(q['preds'], R)]) if q else None
    return cer, em


rng = np.random.default_rng(0)
out = {}
(ca, ea), (cb, eb) = ind(A), ind(B)
for lab, a, b in [('CER', ca, cb), ('EM', ea, eb)]:
    if a is None or b is None:
        continue
    d = a - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
    out[lab] = {'diff': round(100 * d.mean(), 2), 'ci95': [round(100 * np.percentile(boot, 2.5), 2), round(100 * np.percentile(boot, 97.5), 2)], 'p': float(f'{p:.2g}')}
print(ds, A, B, out)
json.dump(out, open(f'results/metrics/loopsig_{ds}_{A}_vs_{B}.json', 'w'))
