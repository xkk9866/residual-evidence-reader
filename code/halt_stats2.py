"""Self-terminated evidence sets (no budget) on the 2,017 held-out MuSiQue dev questions, plus three-way verdicts on
MuSiQue-Full.  usage: python code/halt_stats2.py   ->  results/metrics/halt_stats2.json, results/sel/musique_dev_{name}_dec.json

Rows
  lact / lach / abl_*   : LACUNA decisions as decoded (Bayes event rule: close when pi(TOP) > pi(a passage is missing))
  *_cal                 : the same reader with ONE log-offset on TOP tuned on the 400 selection questions (the calibration
                          the static baselines receive), to separate model quality from the decision rule
  jbce3tuned/jbce4tuned : static reader, probability threshold tuned on the selection questions
  chaintuned            : Chain-CE, first m passages of its best chain, m tuned on the selection questions
  beam                  : official Beam Retrieval (returns |G| passages, i.e. uses the gold hop count)
"""
import json, os, pickle, random
import numpy as np

recs = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
sel_idx = set(random.Random(0).sample(range(len(recs)), 400))
ho = np.array([i not in sel_idx for i in range(len(recs))])
selm = ~ho
GRID_B = np.linspace(-4, 4, 81)


def stats(sets, reads, mask=ho):
    R = [r for r, m in zip(recs, mask) if m]
    S = [s for s, m in zip(sets, mask) if m]
    rd = [x for x, m in zip(reads, mask) if m]
    G = [set(r['gold']) for r in R]
    cer = [g <= set(s) for g, s in zip(G, S)]
    em = [g == set(s) for g, s in zip(G, S)]
    f1 = []
    for g, s in zip(G, S):
        tp = len(g & set(s))
        f1.append(2 * tp / (len(s) + len(g)) if tp else 0.0)
    hop = {}
    for r, c in zip(R, cer):
        hop.setdefault(len(r['gold']), []).append(c)
    out = {'CER': 100 * np.mean(cer), 'SupEM': 100 * np.mean(em), 'SupF1': 100 * np.mean(f1),
           'Premature': 100 * np.mean([not c and set(s) < g for c, s, g in zip(cer, S, G)]),
           'Size': float(np.mean([len(s) for s in S])), 'Tok': float(np.mean([sum(r['ntok'][i] for i in s) for r, s in zip(R, S)])),
           'Reads': float(np.mean(rd))}
    out.update({f'CER_{h}hop': 100 * np.mean(v) for h, v in sorted(hop.items())})
    return {k: round(v, 2) for k, v in out.items()}


def lac_sets(P, b=0.0):
    """close at the first t >= 1 with log pi(TOP) + b > log pi(a passage of the pool is missing)"""
    sets, reads = [], []
    for o in P:
        T = len(o['top']) - 1
        pt, pb = np.exp(o['top']), np.exp(o['bot'])
        pm = np.clip(1 - pt - pb, 1e-12, 1)
        t = next((t for t in range(1, T + 1) if np.log(max(pt[t], 1e-12)) + b > np.log(pm[t])), T)
        sets.append(o['order'][:t]); reads.append(t + 1)
    return sets, reads


def em_sel(sets):
    return np.mean([set(s) == set(recs[i]['gold']) for i, s in enumerate(sets) if selm[i]])


res, dec = {}, {}
LAC = {'lacc': 'ckpt/mq_lacc', 'lacc_s43': 'ckpt/mq_lacc_s43', 'lacc_s44': 'ckpt/mq_lacc_s44', 'laccr': 'ckpt/mq_laccr', 'lace': 'ckpt/mq_lace_s42', 'lace_s43': 'ckpt/mq_lace_s43', 'lace_s44': 'ckpt/mq_lace_s44', 'lach': 'ckpt/mq_lacr_ev', 'lach_s43': 'ckpt/mq_lacr_s43', 'lach_s44': 'ckpt/mq_lacr_s44', 'lacr_f43': 'ckpt/mq_lacr_f43',
       'lact': 'ckpt/mq_lact', 'lact_s43': 'ckpt/mq_lact_s43',
       'abl_ce1': 'ckpt/abl_ce1', 'abl_noroll': 'ckpt/abl_noroll', 'abl_noworld': 'ckpt/abl_noworld', 'abl_mark': 'ckpt/abl_mark', 'abl_prefix': 'ckpt/abl_prefix'}
LAC.update({f'lact_s43_q{q}': f'ckpt/mq_lact_s43_q{q}' for q in range(1, 5)})
for name, d in LAC.items():
    p = f'{d}/pred_musique_dev.pkl'
    if not os.path.exists(p):
        continue
    P = pickle.load(open(p, 'rb'))
    s, rd = lac_sets(P)
    dec[name], res[name] = s, stats(s, rd)
    b = float(max(GRID_B, key=lambda b: em_sel(lac_sets(P, b)[0])))
    s, rd = lac_sets(P, b)
    dec[name + '_cal'], res[name + '_cal'] = s, dict(stats(s, rd), offset=round(b, 2))
if os.path.exists('ckpt/mq_lacp/pred_musique_dev.pkl'):
    P = pickle.load(open('ckpt/mq_lacp/pred_musique_dev.pkl', 'rb'))
    dec['lacp'] = [o['order'][:o['halt']] for o in P]
    res['lacp'] = stats(dec['lacp'], [o['halt'] + 1 for o in P])


def thr_sets(J, th):
    out = []
    for o in J:
        p = 1 / (1 + np.exp(-np.asarray(o['rho'])))
        out.append([int(i) for i in np.argsort(-p) if p[i] > th] or [int(np.argmax(p))])
    return out


for name, d in [('jbce3tuned', 'ckpt/mq_jbce3'), ('jbce4tuned', 'ckpt/mq_jbce4'), ('jbce3_s43tuned', 'ckpt/mq_jbce3_s43'),
                ('jbce3_s44tuned', 'ckpt/mq_jbce3_s44'), ('jbce3_f43tuned', 'ckpt/mq_jbce3_f43')]:
    if not os.path.exists(f'{d}/pred_musique_dev.pkl'):
        continue
    J = pickle.load(open(f'{d}/pred_musique_dev.pkl', 'rb'))
    th = float(max(np.linspace(0.05, 0.95, 91), key=lambda t: em_sel(thr_sets(J, t))))
    dec[name] = thr_sets(J, th)
    res[name] = dict(stats(dec[name], [1] * len(recs)), thr=round(th, 2))
C = pickle.load(open('ckpt/mq_chain/pred_musique_dev.pkl', 'rb'))
m = max(range(1, 6), key=lambda m: em_sel([o['order'][:m] for o in C]))
dec['chaintuned'] = [o['order'][:m] for o in C]
res['chaintuned'] = dict(stats(dec['chaintuned'], [20 * 4 * 2] * len(recs)), m=m)
B = pickle.load(open('results/scores/musique_dev_beam.pkl', 'rb'))
dec['beam'] = [list(s) for s in (B['k5'] if isinstance(B, dict) else B)]
res['beam'] = stats(dec['beam'], [20 * len(s) for s in dec['beam']])
for k, v in dec.items():
    json.dump({'dec': [list(map(int, s)) for s in v]}, open(f'results/sel/musique_dev_{k}_dec.json', 'w'))

# three-way verdicts on MuSiQue-Full: complete (TOP) / missing outside the pool (BOT) / undetermined
F = [json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
for name, d in [('lacc', 'ckpt/mq_lacc'), ('lacc_s43', 'ckpt/mq_lacc_s43'), ('lacc_s44', 'ckpt/mq_lacc_s44'), ('laccr', 'ckpt/mq_laccr'), ('lach_s43', 'ckpt/mq_lacr_s43'), ('lach_s44', 'ckpt/mq_lacr_s44'), ('lace', 'ckpt/mq_lace_s42'), ('lact', 'ckpt/mq_lact'), ('lach', 'ckpt/mq_lacr_ev'), ('lacr_f43', 'ckpt/mq_lacr_f43')]:
    p = f'{d}/pred_musique_full_dev.pkl'
    if not os.path.exists(p):
        continue
    Q = pickle.load(open(p, 'rb'))
    if 'verdict3' not in Q[0]:
        continue
    ans = np.array([bool(r.get('answerable', r.get('gold_missing', 0) == 0)) for r in F])
    v = np.array([o['verdict3'] for o in Q])
    out = {}
    for lab, msk in [('answerable', ans), ('unanswerable', ~ans)]:
        out[lab] = {k: round(100 * float(np.mean(v[msk] == c)), 2) for k, c in [('top', 1), ('bot', 0), ('undetermined', -1)]}
    out['bot_precision'] = round(100 * float(np.mean(~ans[v == 0])), 2) if (v == 0).any() else 0.0
    out['top_precision'] = round(100 * float(np.mean(ans[v == 1])), 2) if (v == 1).any() else 0.0
    out['certified_complete'] = round(100 * float(np.mean([set(r['gold']) <= set(o['order'][:o['halt_any']]) for o, r, a in zip(Q, F, ans) if a and o['verdict3'] == 1])), 2)
    res[f'verdict3_{name}'] = out
json.dump(res, open('results/metrics/halt_stats2.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
