"""Answerability on MuSiQue-Full: does the pool contain a complete proof?
usage: python ans_eval.py name=path:type ...   type: order (LACUNA) | circuit (PROVE-v2) | score (relevance reader)
  order   : max_t sigma(S_t) along the residual completion trajectory (S_t certified complete)
  circuit : pool-level log-sufficiency log Phi(C)
  score   : log of the product of the two most relevant passages' probabilities (relevance proxy)
Also reports the joint "certified answerable AND decision set contains the full proof" rate on answerable pools.
"""
import json, pickle, sys
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from select_alg import pool_logphi


def logsig(x):
    return -np.logaddexp(0, -np.asarray(x, dtype=np.float64))


recs = [json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
y = np.array([r['answerable'] for r in recs], dtype=int)
out = {}
for spec in sys.argv[1:]:
    name, rest = spec.split('=')
    path, typ = rest.split(':')
    preds = pickle.load(open(path, 'rb'))
    if typ == 'order':
        # the trajectory reaches a certified-complete state: max_t log pi(TOP | S_t); decision = TOP is the argmax
        s = np.array([max(o['top'][1:]) for o in preds])
        thr = None
        if 'verdict' in preds[0]:    # point-process reader: verdict of the halting state
            dec = np.array([o['verdict'] for o in preds])
        else:
            dec = np.array([any(o['top'][t] >= max(o['bot'][t], o['dmax'][t]) for t in range(1, len(o['top'])))
                            for o in preds])
    elif typ == 'cls':     # answerability classifier: TOP-vs-BOT logit at the empty state
        s = np.array([o['top'][0] - o['bot'][0] for o in preds]); thr = 0.0
    elif typ == 'ns':      # NS-certified decoding: best sufficiency-necessity score of a TOP-terminated vs BOT-terminated proof
        s = np.array(preds['ns_top']) - np.array(preds['ns_bot_best']); thr = 0.0
    elif typ == 'circuit':
        s = np.array([pool_logphi(o['gam'], o['lp']) for o in preds]); thr = np.log(0.5)
    else:
        s = np.array([np.sort(logsig(o['rho']))[-2:].sum() for o in preds]); thr = np.log(0.25)
    res = {'AUROC': round(100 * roc_auc_score(y, s), 2), 'AUPRC': round(100 * average_precision_score(y, s), 2),
           'MacroF1': round(100 * f1_score(y, (dec if thr is None else s > thr).astype(int), average='macro'), 2)}
    if typ == 'ns':
        cert = np.array(preds['ns_top']) > np.array(preds['ns_bot_best'])
        good = np.array([bool(r['answerable']) and set(r['gold']) <= set(S) for S, r in zip(preds['dec'], recs)])
        res['CertPrec'] = round(100 * good[cert].mean(), 2)          # certified pools whose certified set is a true proof
        res['CertCov'] = round(100 * cert.mean(), 2)
        res['CertProof'] = round(100 * np.mean([r['answerable'] and set(r['gold']) <= set(S) and t > b for S, t, b, r in zip(preds['dec'], preds['ns_top'], preds['ns_bot_best'], recs) if r['answerable']]), 2)
    if typ == 'order':
        ok = []
        for o, r in zip(preds, recs):
            if not r['answerable']:
                continue
            if 'verdict' in o:
                t = o['halt'] if o['verdict'] else None
            else:
                t = next((t for t in range(1, len(o['top'])) if o['top'][t] >= max(o['bot'][t], o['dmax'][t])), None)
            ok.append(t is not None and set(r['gold']) <= set(o['order'][:t]))
        res['CertProof'] = round(100 * np.mean(ok), 2)
    out[name] = res
    print(f'{name:16s}', res, flush=True)
json.dump(out, open('results/metrics/musique_full_ans_' + '_'.join(out) + '.json', 'w'), indent=1)
