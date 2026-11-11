"""Fair self-termination comparison on the 2,017 held-out MuSiQue questions.
static (tuned)   : static joint reader, threshold on sigmoid(p) chosen on the 400 selection questions to maximise Sup-EM
static (matched) : static joint reader, per-question top-n with a global threshold chosen so that its mean reader tokens
                   equal LACUNA's on the selection questions
LACUNA           : first state at which TOP beats every open passage (no tuning)
writes results/metrics/stop_fair.json"""
import json, pickle, random
import numpy as np

R = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
sel = {r['id'] for r in random.Random(0).sample(R, 400)}
is_sel = np.array([r['id'] in sel for r in R])
J = pickle.load(open('ckpt/mq_jbce3/pred_musique_dev.pkl', 'rb'))
L = pickle.load(open('ckpt/mq_lacr/pred_musique_dev.pkl', 'rb'))


def stats(sets, mask):
    rs = [r for r, m in zip(R, mask) if m]
    ss = [s for s, m in zip(sets, mask) if m]
    tok = np.mean([sum(r['ntok'][i] for i in s) for s, r in zip(ss, rs)])
    cer = 100 * np.mean([set(r['gold']) <= set(s) for s, r in zip(ss, rs)])
    em = 100 * np.mean([set(r['gold']) == set(s) for s, r in zip(ss, rs)])
    f1 = []
    for s, r in zip(ss, rs):
        tp = len(set(s) & set(r['gold']))
        p, c = (tp / len(s) if s else 0), tp / len(r['gold'])
        f1.append(2 * p * c / (p + c) if tp else 0)
    return {'tok': round(float(tok), 1), 'CER': round(float(cer), 2), 'SupEM': round(float(em), 2), 'SupF1': round(100 * float(np.mean(f1)), 2)}


def thr_sets(th):
    out = []
    for o in J:
        p = 1 / (1 + np.exp(-o['rho']))
        s = [int(i) for i in np.where(p > th)[0]]
        out.append(s or [int(np.argmax(p))])
    return out


lac = []
for o in L:
    t = next((t for t in range(1, len(o['top'])) if o['top'][t] > o['dmax'][t]), len(o['top']) - 1)
    lac.append(o['order'][:t])
grid = np.linspace(0.05, 0.95, 91)
best = max(grid, key=lambda th: stats(thr_sets(th), is_sel)['SupEM'])
lt = stats(lac, is_sel)['tok']
match = min(grid, key=lambda th: abs(stats(thr_sets(th), is_sel)['tok'] - lt))
ho = ~is_sel
res = {'heldout_n': int(ho.sum()), 'lacuna': stats(lac, ho), 'static_p0.5': stats(thr_sets(0.5), ho),
       'static_tuned': dict(stats(thr_sets(best), ho), thr=round(float(best), 2)),
       'static_tokmatched': dict(stats(thr_sets(match), ho), thr=round(float(match), 2))}
json.dump(res, open('results/metrics/stop_fair.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
