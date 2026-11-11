"""Propose-and-verify decoding with a residual-proof reader (no retraining).

The residual distribution pi(. | q, C, S) is used twice:
  propose : expand each evidence state by its most probable missing passages (beam over states, path score
            = sum_t log pi(d_t | S_t), states are sets so permutations are merged)
  verify  : among all visited states that fit the budget, keep the one the reader itself certifies as
            complete, argmax log pi(TOP | S); the remaining budget is filled by that state's residual ranking.
usage: python decode3.py --ckpt ckpt/mq_lac/model.pt --out ckpt/mq_lac --infer data/proc/musique_dev.jsonl --width 3
output: pred_{name}_pv.pkl  (dict of budget -> list of selected index lists; 'order' for token budgets)
"""
import sys, os, json, pickle, time
import numpy as np
import torch

W = 3
if '--width' in sys.argv:
    i = sys.argv.index('--width'); W = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train3.py')).read().split("def main():")[0])


@torch.no_grad()
def read(model, r, n, S):
    x, spans = build(r['question'], r['docs'], list(S))
    lp = run(model, x, spans, [i in S for i in range(n)]).log_softmax(0).cpu().numpy()
    return lp


@torch.no_grad()
def propose_verify(model, r, T, width):
    n = len(r['docs'])
    beam = [((), 0.0)]                      # (ordered state, path log-prob)
    seen = {}                               # frozenset -> (order, path, top, lp)
    for t in range(min(T, n) + 1):
        cand = {}
        for S, path in beam:
            lp = read(model, r, n, S)
            seen[frozenset(S)] = (S, path, float(lp[n]), lp)
            if t == min(T, n):
                continue
            free = [i for i in np.argsort(-lp[:n]) if i not in S][:width]
            for i in free:
                key = frozenset(S + (int(i),))
                v = path + float(lp[i])
                if key not in cand or cand[key][1] < v:
                    cand[key] = (S + (int(i),), v)
        beam = sorted(cand.values(), key=lambda z: -z[1])[:width]
        if not beam:
            break
    return seen


def fill(state, lp, n):
    S = list(state)
    return S + [int(i) for i in np.argsort(-lp[:n]) if i not in S]


def main3():
    model = ResidualReader(args.enc).to(dev)
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    for path in args.infer.split(','):
        recs = load(path, args.max_infer)
        t0 = time.time()
        out = {'order': [], 'k2': [], 'k3': [], 'k4': [], 'k5': [], 'top': []}
        for r in recs:
            n = len(r['docs'])
            seen = propose_verify(model, r, args.steps, W)
            for k in [2, 3, 4, 5]:
                S, _, _, lp = max((v for key, v in seen.items() if len(key) <= k), key=lambda v: (v[2], v[1]))
                out[f'k{k}'].append(fill(S, lp, n)[:k])
            S, _, top, lp = max(seen.values(), key=lambda v: (v[2], v[1]))
            out['order'].append(fill(S, lp, n))
            out['top'].append(top)
        name = os.path.basename(path).replace('.jsonl', '')
        pickle.dump(out, open(f'{args.out}/pred_{name}_pv{W}.pkl', 'wb'))
        hit = {k: round(100 * np.mean([set(r['gold']) <= set(s) for s, r in zip(out[k], recs)]), 2) for k in ['k2', 'k3', 'k4', 'k5']}
        print('pv', W, path, len(recs), 'sec', round(time.time() - t0, 1), hit, flush=True)


main3()
