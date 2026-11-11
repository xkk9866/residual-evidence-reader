"""Statistics of the two-sided residual reader (greedy walk on the subset lattice), MuSiQue.
  python code/delta_stats.py name=ckpt_dir ...   ->  results/metrics/delta_stats.json, results/sel/musique_dev_{name}_dec.json

For every reader (ckpt_dir/pred_musique_dev.pkl, pred_musique_full_dev.pkl):
  name       : self-terminated evidence set as decoded (stop = first state with pi(TOP) > total flip mass), 2,017 held-out questions
  name_cal   : the same walk with ONE log-offset on TOP tuned on the 400 selection questions (analysis only)
  walk_name  : what the walk does before it stops: exact / premature / over-complete / unhalted; wrong picks that were
               released again (recovered) and whether those questions end exact; precision of the releases; cycles
  verdict3_name : three-way verdicts on MuSiQue-Full, with the share of 'complete' verdicts whose state contains G
"""
import json, os, pickle, random, sys
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
    hopem = {}
    for r, e in zip(R, em):
        hopem.setdefault(len(r['gold']), []).append(e)
    out = {'CER': 100 * np.mean(cer), 'SupEM': 100 * np.mean(em), 'SupF1': 100 * np.mean(f1),
           'Premature': 100 * np.mean([not c and set(s) < g for c, s, g in zip(cer, S, G)]),
           'Size': float(np.mean([len(s) for s in S])), 'Tok': float(np.mean([sum(r['ntok'][i] for i in s) for r, s in zip(R, S)])),
           'Reads': float(np.mean(rd))}
    out.update({f'CER_{h}hop': 100 * np.mean(v) for h, v in sorted(hop.items())})
    out.update({f'SupEM_{h}hop': 100 * np.mean(v) for h, v in sorted(hopem.items())})
    return {k: round(v, 2) for k, v in out.items()}


def walk_sets(P, b=0.0):
    """stop at the first t >= 1 with log pi(TOP) + b > log(total flip mass); the set is the recorded state"""
    sets, reads = [], []
    for o in P:
        T = len(o['top']) - 1
        t = next((t for t in range(1, T + 1) if o['top'][t] + b > o['pin'][t]), None)
        reads.append((T if t is None else t) + 1)
        if t is None:       # no certified state: the visited state the reader holds most probably exact
            t = 1 + int(np.argmax(o['top'][1:]))
        sets.append(list(o['states'][t]))
    return sets, reads


def em_sel(sets):
    return np.mean([set(s) == set(recs[i]['gold']) for i, s in enumerate(sets) if selm[i]])


def walk_analysis(P):
    n = cnt_exact = cnt_prem = cnt_over = cnt_unh = unh_exact = visited = 0
    recovered = rec_exact = wrong_added = cycles = 0
    rem_total = rem_bad = 0
    steps_to_stop, removals = [], []
    for i, (o, r) in enumerate(zip(P, recs)):
        if i in sel_idx:
            continue
        n += 1
        G = set(r['gold']); h = o['halt_t']; T = len(o['top']) - 1
        S = set(o['states'][h])
        mv = o['moves'][:h]
        visited += any(set(s) == G for s in o['states'][1:])
        added = [m for m in mv if m >= 0]
        rem = [-m - 1 for m in mv if m < 0]
        rem_total += len(rem); rem_bad += sum(x in G for x in rem)
        removals.append(len(rem)); steps_to_stop.append(h)
        wrong = {m for m in added if m not in G}
        if wrong:
            wrong_added += 1
        rel = wrong & set(rem)
        if rel:
            recovered += 1
            rec_exact += S == G
        if len(added) != len(set(added)):
            cycles += 1
        if not o.get('stopped', np.exp(o['top'][h]) > np.exp(o['pin'][h])):
            cnt_unh += 1; unh_exact += S == G
        elif S == G:
            cnt_exact += 1
        elif not G <= S:
            cnt_prem += 1
        else:
            cnt_over += 1
    pc = lambda x: round(100 * x / n, 2)
    return {'exact': pc(cnt_exact), 'premature': pc(cnt_prem), 'over': pc(cnt_over), 'unhalted': pc(cnt_unh), 'unhalted_exact': pc(unh_exact), 'visited': pc(visited),
            'ident_given_visited': round(100 * (cnt_exact + unh_exact) / max(visited, 1), 2),      # the exact state was returned when the walk had visited it
            'ident_given_visited': round(100 * (cnt_exact + unh_exact) / max(visited, 1), 2),      # the exact state was returned when the walk had visited it
            'wrong_pick_any': pc(wrong_added), 'recovered': pc(recovered), 'recovered_exact': pc(rec_exact),
            'release_precision': round(100 * (1 - rem_bad / rem_total), 2) if rem_total else 0.0,
            'releases_per_q': round(float(np.mean(removals)), 3), 'cycles': pc(cycles),
            'steps_to_stop': round(float(np.mean(steps_to_stop)), 2), 'n': n}


res, dec = {}, {}
for spec in sys.argv[1:]:
    name, d = spec.split('=')
    p = f'{d}/pred_musique_dev.pkl'
    if not os.path.exists(p):
        print('missing', p); continue
    P = pickle.load(open(p, 'rb'))
    if 'states' not in P[0]:
        # a one-sided reader decoded with leave-one-out release: only the stop-set statistics (no walk to analyse)
        s = [list(o['order'][:o['halt']]) for o in P]
        dec[name], res[name] = s, stats(s, [o.get('reads', len(o['top']) + o.get('reads_loo', 0)) for o in P])
        res['size_' + name] = stats([o['order'][:len(r['gold'])] for o, r in zip(P, recs)], [0] * len(P), mask=np.ones(len(recs), bool))
        if 'released_loo' in P[0]:      # leave-one-out release: how many passages were released, and how often a distractor
            rel = [o['order'][o['halt']:o['halt'] + o.get('released_loo', 0)] for o in P]
            tot = sum(len(x) for x, m in zip(rel, ho) if m)
            good = sum(sum(e not in set(r['gold']) for e in x) for x, r, m in zip(rel, recs, ho) if m)
            res['walk_' + name] = {'releases_per_q': round(tot / int(ho.sum()), 3), 'release_precision': round(100 * good / max(tot, 1), 2),
                                   'reads_loo': round(float(np.mean([o.get('reads_loo', 0) for o, m in zip(P, ho) if m])), 2)}
        print(name, res[name], flush=True); continue
    s = [list(o['order'][:o['halt']]) for o in P]
    dec[name], res[name] = s, stats(s, [o['reads'] for o in P])
    b = float(max(GRID_B, key=lambda b: em_sel(walk_sets(P, b)[0])))
    s, rd = walk_sets(P, b)
    dec[name + '_cal'], res[name + '_cal'] = s, dict(stats(s, rd), offset=round(b, 2))
    res['walk_' + name] = walk_analysis(P)
    # equal-size comparison: the first |G| passages of the ranking (necessity order of the closed set, then forced additions)
    res['size_' + name] = stats([o['order'][:len(r['gold'])] for o, r in zip(P, recs)], [o['reads'] for o in P], mask=np.ones(len(recs), bool))
    pf = f'{d}/pred_musique_full_dev.pkl'
    if os.path.exists(pf):
        F = [json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
        Q = pickle.load(open(pf, 'rb'))
        ans = np.array([bool(r.get('answerable', r.get('gold_missing', 0) == 0)) for r in F])
        v = np.array([o['verdict3'] for o in Q])
        out = {}
        for lab, msk in [('answerable', ans), ('unanswerable', ~ans)]:
            out[lab] = {k: round(100 * float(np.mean(v[msk] == c)), 2) for k, c in [('top', 1), ('bot', 0), ('undetermined', -1)]}
        out['bot_precision'] = round(100 * float(np.mean(~ans[v == 0])), 2) if (v == 0).any() else 0.0
        out['top_precision'] = round(100 * float(np.mean(ans[v == 1])), 2) if (v == 1).any() else 0.0
        cc = [set(r['gold']) <= set(o['set_any']) for o, r, a in zip(Q, F, ans) if a and o['verdict3'] == 1]
        out['certified_complete'] = round(100 * float(np.mean(cc)), 2) if cc else 0.0
        ce = [set(r['gold']) == set(o['set_any']) for o, r, a in zip(Q, F, ans) if a and o['verdict3'] == 1]
        out['certified_exact'] = round(100 * float(np.mean(ce)), 2) if ce else 0.0
        res['verdict3_' + name] = out
    print(name, json.dumps({k: res[k] for k in res if k.endswith(name) or k.startswith(name)}, indent=1), flush=True)
os.makedirs('results/sel', exist_ok=True)
for k, v in dec.items():
    json.dump({'dec': [list(map(int, s)) for s in v]}, open(f'results/sel/musique_dev_{k}_dec.json', 'w'))
old = json.load(open('results/metrics/delta_stats.json')) if os.path.exists('results/metrics/delta_stats.json') else {}
old.update(res)
json.dump(old, open('results/metrics/delta_stats.json', 'w'), indent=1)
