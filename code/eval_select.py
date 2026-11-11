"""Budgeted selection + retrieval metrics.
usage: python eval_select.py DS name=path:type[:strategy] ...
  type: circuit (pkl with gam,lg) | score (pkl with rho) | sets (pkl list of index lists)
  strategy for score: topk | mmr | dpp ; for circuit: greedy (default) | fill (rank by max fill, ablation) | noenum
"""
import json, os, pickle, sys
import numpy as np
from select_alg import greedy_select, topk_budget, mmr_budget, dpp_budget, map_proof, doc_marginal

DOC_BUDGETS = [2, 3, 4, 5]
TOK_BUDGETS = [256, 384, 512, 768, 1024]


def hop_of(r):
    t = r.get('type', '')
    return t[:4] if t[:1].isdigit() else str(len(r['gold']))


def select(o, r, strat, typ, B, unit, emb=None):
    n = len(r['docs'])
    cost = np.ones(n) if unit else np.asarray(r['ntok'], dtype=float)
    if typ == 'sets':
        S, sp = [], 0
        for i in o:
            if sp + cost[i] <= B:
                S.append(i); sp += cost[i]
        return S
    if typ == 'order':
        rank = np.empty(n); rank[o['order']] = -np.arange(n)
        return topk_budget(rank, cost, B)
    if typ == 'circuit':
        if strat == 'fill':
            return topk_budget(doc_marginal(o['gam'], o['lp']), cost, B)
        return greedy_select(o['gam'], o['lp'], cost, B, enum=0 if (unit or strat == 'noenum') else 2)
    if strat == 'mmr':
        return mmr_budget(o['rho'], emb.astype(np.float32), cost, B)
    if strat == 'dpp':
        return dpp_budget(o['rho'], emb.astype(np.float32), cost, B)
    return topk_budget(o['rho'], cost, B)


def prf(S, G):
    S, G = set(S), set(G)
    tp = len(S & G)
    p = tp / len(S) if S else 0
    rcl = tp / len(G)
    return p, rcl, (2 * p * rcl / (p + rcl) if tp else 0)


def evaluate(ds, name, path, typ, strat='topk', save_sets=True):
    recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
    preds = pickle.load(open(path, 'rb'))
    emb = pickle.load(open(f'results/scores/{ds}_emb.pkl', 'rb')) if strat in ('mmr', 'dpp') else [None] * len(recs)
    assert typ in ('bsets', 'pv', 'ns') or len(preds) == len(recs), (len(preds), len(recs))
    res = {}
    sets = {}
    for unit, budgets in [(True, DOC_BUDGETS), (False, TOK_BUDGETS)]:
        for B in budgets:
            key = f'k{B}' if unit else f't{B}'
            if typ == 'bsets' and key not in preds:
                continue
            cer, rec, tokused, hop = [], [], [], {}
            sel_all = []
            if typ in ('pv', 'ns'):      # propose-and-verify: verified sets for document budgets, ranking for token budgets
                items = preds[key] if unit else preds['order']
            else:
                items = preds[key] if typ == 'bsets' else preds
            for o, r, e in zip(items, recs, emb):
                S = select(o, r, strat, 'sets' if typ in ('bsets', 'pv', 'ns') else typ, B, unit, e)
                sel_all.append([int(i) for i in S])
                miss = r.get('gold_missing', 0)
                c = miss == 0 and set(r['gold']) <= set(S)
                cer.append(c); rec.append(len(set(S) & set(r['gold'])) / (len(r['gold']) + miss))
                tokused.append(sum(r['ntok'][i] for i in S))
                hop.setdefault(hop_of(r), []).append(c)
            res[key] = {'CER': round(100 * np.mean(cer), 2), 'Recall': round(100 * np.mean(rec), 2),
                        'tok': round(float(np.mean(tokused)), 1),
                        'CER_hop': {h: round(100 * np.mean(v), 2) for h, v in sorted(hop.items())}}
            sets[key] = sel_all
    # parameter-free decision rule -> Sup-EM/F1
    if typ == 'ns':
        dec_sets = [list(map(int, S)) for S in preds['dec']]
        sets['dec'] = dec_sets
        res['dec_tok'] = round(float(np.mean([sum(r['ntok'][i] for i in S) for S, r in zip(dec_sets, recs)])), 1)
        res['dec_CER'] = round(100 * np.mean([set(r['gold']) <= set(S) for S, r in zip(dec_sets, recs)]), 2)
        res['decision'] = {'SupEM': round(100 * np.mean([set(S) == set(r['gold']) for S, r in zip(dec_sets, recs)]), 2),
                           'SupF1': round(100 * np.mean([prf(S, r['gold'])[2] for S, r in zip(dec_sets, recs)]), 2)}
    if typ in ('circuit', 'score') or (typ == 'order' and 'top' in preds[0]):
        em, f1, dec_sets = [], [], []
        for o, r in zip(preds, recs):
            if typ == 'order':
                # selection is decoded in the answerable world: pi restricted to open passages + TOP;
                # stop at the first state whose most probable outcome is TOP (answerability is read from TOP vs BOT)
                if 'halt' in o:    # point-process reader: the state at which it halts
                    t = o['halt']
                else:
                    t = next((t for t in range(1, len(o['top'])) if o['top'][t] > o['dmax'][t]), len(o['top']) - 1)
                S = o['order'][:t]
            elif typ == 'circuit':
                S = map_proof(o['gam'], o['lp'])
            else:
                S = [i for i, v in enumerate(o['rho']) if v > 0] or [int(np.argmax(o['rho']))]
            dec_sets.append([int(i) for i in S])
            if r.get('gold_missing', 0) or not r['gold']:
                em.append(False); f1.append(0.0); continue
            em.append(set(S) == set(r['gold'])); f1.append(prf(S, r['gold'])[2])
        sets['dec'] = dec_sets
        res['dec_tok'] = round(float(np.mean([sum(r['ntok'][i] for i in S) for S, r in zip(dec_sets, recs)])), 1)
        res['dec_CER'] = round(100 * np.mean([set(r['gold']) <= set(S) for S, r in zip(dec_sets, recs)]), 2)
        res['decision'] = {'SupEM': round(100 * np.mean(em), 2), 'SupF1': round(100 * np.mean(f1), 2)}
    os.makedirs('results/sel', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    if save_sets:
        json.dump(sets, open(f'results/sel/{ds}_{name}.json', 'w'))
    json.dump(res, open(f'results/metrics/{ds}_{name}.json', 'w'), indent=1)
    return res


if __name__ == '__main__':
    ds = sys.argv[1]
    for spec in sys.argv[2:]:
        name, rest = spec.split('=')
        parts = rest.split(':')
        path, typ = parts[0], parts[1]
        strat = parts[2] if len(parts) > 2 else ('greedy' if typ == 'circuit' else 'topk')
        r = evaluate(ds, name, path, typ, strat)
        line = ' '.join(f"{k}:{v['CER']}" for k, v in r.items() if isinstance(v, dict) and 'CER' in v)
        print(f'{name:22s}', line, r.get('decision', ''), flush=True)
