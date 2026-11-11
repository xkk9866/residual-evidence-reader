"""Event-level calibration of the residual distribution on the states that decoding visits (held-out MuSiQue dev).
For every state S_t (t >= 1) of the completion trajectory of an answerable question, the event 'complete' has
probability p_top(S_t) and label 1[G subset S_t]. Reports ECE (10 equal-width bins), Brier score, reliability bins,
and ECE by hop count.  Also the paired 4-hop CER@5 differences of LACUNA seeds vs static seeds.
  python code/calib.py name=pred.pkl ...  -> results/metrics/calib.json"""
import json, pickle, random, sys
import numpy as np

R = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
sel = set(random.Random(0).sample(range(len(R)), 400))
ho = [i for i in range(len(R)) if i not in sel]


def ece(p, y, nb=10):
    p, y = np.asarray(p), np.asarray(y, float)
    b = np.minimum((p * nb).astype(int), nb - 1)
    e, rel = 0.0, []
    for k in range(nb):
        m = b == k
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
            rel.append([round(float(p[m].mean()), 3), round(float(y[m].mean()), 3), int(m.sum())])
    return 100 * e, rel


out = {}
for spec in sys.argv[1:]:
    name, path = spec.split('=')
    P = pickle.load(open(path, 'rb'))
    ps, ys, yx, hop = [], [], [], []
    for i in ho:
        o, G = P[i], set(R[i]['gold'])
        two = 'states' in o          # two-sided walk: the states are recorded (the walk can release passages)
        for t in range(1, len(o['top'])):
            ps.append(float(np.exp(o['top'][t])))
            St = set(o['states'][t] if two else o['order'][:t])
            ys.append(int(G <= St)); yx.append(int(G == St))
            hop.append(len(G))
    ps, ys, yx, hop = np.array(ps), np.array(ys), np.array(yx), np.array(hop)
    e, rel = ece(ps, ys)
    res = {'ECE': round(e, 2), 'Brier': round(100 * float(np.mean((ps - ys) ** 2)), 2), 'mean_p': round(100 * float(ps.mean()), 2),
           'rate': round(100 * float(ys.mean()), 2), 'reliability': rel,
           # TOP of the two-sided reader asserts S = G exactly; its calibration against that event
           'ECE_exact': round(ece(ps, yx)[0], 2), 'Brier_exact': round(100 * float(np.mean((ps - yx) ** 2)), 2), 'rate_exact': round(100 * float(yx.mean()), 2)}
    for h in (2, 3, 4):
        m = hop == h
        res[f'ECE_{h}hop'] = round(ece(ps[m], ys[m])[0], 2)
        res[f'overconf_{h}hop'] = round(100 * float(ps[m].mean() - ys[m].mean()), 2)
    out[name] = res
    print(name, {k: v for k, v in res.items() if k != 'reliability'}, flush=True)

# paired 4-hop CER@5 over seed pairs (LACUNA seed s vs static seed s)
def c5(path, typ):
    P = pickle.load(open(path, 'rb'))
    idx = [i for i in range(len(R)) if len(R[i]['gold']) == 4]
    if typ == 'order':
        return np.array([set(R[i]['gold']) <= set(P[i]['order'][:5]) for i in idx], float)
    return np.array([set(R[i]['gold']) <= set(np.argsort(-np.asarray(P[i]['rho']))[:5]) for i in idx], float)


pairs = json.loads(open('results/metrics/seedpairs.json').read()) if False else None
rng = np.random.default_rng(0)
seed_res = {}
for tag, A, B in [('s42', ('ckpt/mq_lacr/pred_musique_dev.pkl', 'order'), ('ckpt/mq_jbce3/pred_musique_dev.pkl', 'score')),
                  ('s43', ('ckpt/mq_lacr_s43/pred_musique_dev.pkl', 'order'), ('ckpt/mq_jbce3_s43/pred_musique_dev.pkl', 'score')),
                  ('s44', ('ckpt/mq_lacr_s44/pred_musique_dev.pkl', 'order'), ('ckpt/mq_jbce3_s44/pred_musique_dev.pkl', 'score')),
                  ('f43', ('ckpt/mq_lacr_f43/pred_musique_dev.pkl', 'order'), ('ckpt/mq_jbce3_f43/pred_musique_dev.pkl', 'score'))]:
    try:
        a, b = c5(*A), c5(*B)
    except FileNotFoundError:
        continue
    d = a - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    seed_res[tag] = {'diff': round(100 * d.mean(), 2), 'ci95': [round(100 * np.percentile(boot, 2.5), 2), round(100 * np.percentile(boot, 97.5), 2)]}
print('4hop paired', seed_res)
out['fourhop_seed_pairs'] = seed_res
json.dump(out, open('results/metrics/calib.json', 'w'), indent=1)
