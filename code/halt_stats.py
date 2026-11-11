"""Self-terminated evidence sets on the 2,017 MuSiQue dev questions not used for model selection, + verdict
consistency on MuSiQue-Full. writes results/metrics/halt_stats.json and results/sel/musique_dev_<name>.json['dec']"""
import json, pickle, random
import numpy as np
R = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
sel = {r['id'] for r in random.Random(0).sample(R, 400)}
ho = np.array([r['id'] not in sel for r in R])
T = 5


def stats(sets, reads):
    m = ho
    out = {}
    cer = [set(r['gold']) <= set(s) for s, r in zip(sets, R)]
    em = [set(r['gold']) == set(s) for s, r in zip(sets, R)]
    f1 = []
    for s, r in zip(sets, R):
        tp = len(set(s) & set(r['gold']))
        p, c = (tp / len(s) if s else 0), tp / len(r['gold'])
        f1.append(2 * p * c / (p + c) if tp else 0)
    under = [len(set(s) & set(r['gold'])) < len(r['gold']) and len(s) < len(r['gold']) for s, r in zip(sets, R)]
    a = lambda v: round(100 * float(np.mean(np.array(v)[m])), 2)
    out = {'CER': a(cer), 'SupEM': a(em), 'SupF1': a(f1), 'Premature': a(under),
           'Size': round(float(np.mean([len(s) for s in np.array(sets, dtype=object)[m]])), 2),
           'Tok': round(float(np.mean([sum(r['ntok'][i] for i in s) for s, r, k in zip(sets, R, m) if k])), 1),
           'Reads': round(float(np.mean(np.array(reads)[m])), 2)}
    for h in (2, 3, 4):
        idx = [i for i, r in enumerate(R) if m[i] and len(r['gold']) == h]
        out[f'CER_{h}hop'] = round(100 * float(np.mean([cer[i] for i in idx])), 2)
    return out


res, dec = {}, {}
L = pickle.load(open('ckpt/mq_lacr/pred_musique_dev.pkl', 'rb'))
h = [next((t for t in range(1, T + 1) if o['top'][t] > o['dmax'][t]), T) for o in L]
dec['lacg'] = [o['order'][:t] for o, t in zip(L, h)]
res['lacg'] = stats(dec['lacg'], [t + 1 for t in h])
P = pickle.load(open('ckpt/mq_lacp/pred_musique_dev.pkl', 'rb'))
dec['lacp'] = [o['order'][:o['halt']] for o in P]
res['lacp'] = stats(dec['lacp'], [o['halt'] + 1 for o in P])
B = pickle.load(open('results/scores/musique_dev_beam.pkl', 'rb'))
dec['beam'] = [list(s) for s in (B['k5'] if isinstance(B, dict) else B)]
res['beam'] = stats(dec['beam'], [20 * len(s) for s in dec['beam']])     # pair reads (beam: one per candidate per hop)
J = pickle.load(open('ckpt/mq_jbce3/pred_musique_dev.pkl', 'rb'))
def thr_sets(th):
    out = []
    for o in J:
        p = 1 / (1 + np.exp(-o['rho']))
        s = [int(i) for i in np.argsort(-p) if p[i] > th]
        out.append(s or [int(np.argmax(p))])
    return out
grid = np.linspace(0.05, 0.95, 91)
selm = ~ho
best = max(grid, key=lambda th: np.mean([set(r['gold']) == set(s) for s, r, k in zip(thr_sets(th), R, selm) if k]))
dec['jbce3tuned'] = thr_sets(best)
res['jbce3tuned'] = dict(stats(dec['jbce3tuned'], [1] * len(R)), thr=round(float(best), 2))
for k, v in dec.items():
    json.dump({'dec': [list(map(int, s)) for s in v]}, open(f'results/sel/musique_dev_{k}_dec.json', 'w'))
# verdict consistency on MuSiQue-Full (LACUNA, Algorithm 1)
F = [json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
Q = pickle.load(open('ckpt/mq_lacr/pred_musique_full_dev.pkl', 'rb'))
agree, cert_ok, ncert = [], [], 0
for o, r in zip(Q, F):
    t = next((t for t in range(1, T + 1) if o['top'][t] > o['dmax'][t]), T)
    v = any(o['top'][u] >= max(o['bot'][u], o['dmax'][u]) for u in range(1, T + 1))
    agree.append(v == (o['top'][t] >= o['bot'][t]))
    if v:
        ncert += 1
        cert_ok.append(bool(r['answerable']) and set(r['gold']) <= set(o['order'][:t]))
res['consistency'] = {'verdict_matches_halt_state': round(100 * float(np.mean(agree)), 2),
                      'certified_rate': round(100 * ncert / len(F), 2),
                      'certified_set_is_complete_proof': round(100 * float(np.mean(cert_ok)), 2)}
json.dump(res, open('results/metrics/halt_stats.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
