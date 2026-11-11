"""Exact decomposition of self-terminated completion failures by the first violated state (Proposition 1):
along the decoded trajectory of an answerable MuSiQue question, the first state t at which the reader's decision is
not an element of the residual R(S_t) is either a premature TOP (TOP chosen while a gold passage is missing) or a
wrong pick (a distractor added while R(S_t) contains passages); a trajectory without violation recovers G exactly.
  python code/first_error.py name=pred.pkl ...   ->  results/metrics/first_error.json (held-out questions)"""
import json, pickle, random, sys
import numpy as np

recs = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
sel = set(random.Random(0).sample(range(len(recs)), 400))
out = {}
for spec in sys.argv[1:]:
    name, path = spec.split('=')
    path, dec = (path.split('@') + [None])[:2]
    P = pickle.load(open(path, 'rb'))
    if dec:          # decisions of another rule (e.g. the calibrated evidence set): the halting state is the set size
        DS = json.load(open(dec))['dec']
        for o, S in zip(P, DS):
            assert list(o['order'][:len(S)]) == list(S)
            o['halt'] = len(S)
    cnt = {'exact': 0, 'premature': 0, 'wrongpick': 0, 'late': 0}
    by_t = {}
    rank_first = []
    n = 0
    for i, (o, r) in enumerate(zip(P, recs)):
        if i in sel:
            continue
        n += 1
        G = set(r['gold']); h = o['halt']
        S = o['order'][:h]
        err = None
        for t in range(h + 1):
            have = set(S[:t])
            if t == h:                                   # the decision at the halting state is TOP
                if not G <= have:
                    err = ('premature', t)
                    rest = [d for d in o['order'][t:]]
                    rank_first.append(min(rest.index(g) for g in G - have))
                break
            if S[t] not in G:                            # a distractor added while something is missing
                if G <= have:
                    err = ('late', t)
                else:
                    err = ('wrongpick', t)
                break
        k = 'exact' if err is None else err[0]
        cnt[k] += 1
        if err:
            by_t.setdefault(err[1], {}).setdefault(err[0], 0)
            by_t[err[1]][err[0]] += 1
    res = {k: round(100 * v / n, 2) for k, v in cnt.items()}
    res['by_step'] = {int(t): {k: round(100 * v / n, 2) for k, v in d.items()} for t, d in sorted(by_t.items())}
    # premature stops in which the missing gold passage was the reader's own top-ranked open passage
    res['premature_gold_ranked_first'] = round(100 * np.mean([x == 0 for x in rank_first]), 2) if rank_first else 0.0
    out[name] = res
    print(name, res, flush=True)
json.dump(out, open('results/metrics/first_error.json', 'w'), indent=1)
