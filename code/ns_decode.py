"""Necessity-sufficiency (NS) certified decoding with a residual-proof reader (no extra parameters).

The residual reader pi(. | q, C, S) answers "what is still missing from S?".  A set S is a *proof* iff
  sufficiency : nothing is missing                       -> pi(TOP | S) high
  necessity   : every member is missing without it       -> pi(e | S \\ {e}) high for every e in S
NS(S) = log pi(TOP | S) + sum_{e in S} log pi(e | S \\ {e})          (order-free pseudo-likelihood of the proof)

Candidates come from residual completion with a small beam (width W over evidence states, states are sets);
all reads are cached per state, so NS needs only the leave-one-out states that the beam did not visit.
Outputs (per question):
  k{2..5} : best NS set with |S| <= k, filled with that state's residual ranking (document budgets)
  order   : best NS set overall, then its residual ranking (token budgets)
  dec     : best NS set overall  (self-terminated evidence; no threshold)
  ans     : NS score of the best set minus that of its BOT-terminated counterpart (answerability score)
usage: python ns_decode.py --ckpt ckpt/X/model.pt --out ckpt/X --infer a.jsonl,b.jsonl [--width 3] [--steps 5]
"""
import sys, os, json, pickle, time
import numpy as np
import torch

W = 3
if '--width' in sys.argv:
    i = sys.argv.index('--width'); W = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'train3.py')).read().split("def main():")[0])


class Reader:
    def __init__(self, model, r):
        self.m, self.r, self.n, self.cache = model, r, len(r['docs']), {}

    @torch.no_grad()
    def __call__(self, S):
        """S: tuple (discovery order). cached by set; returns log pi over [docs..., TOP, BOT]"""
        key = frozenset(S)
        if key not in self.cache:
            x, spans = build(self.r['question'], self.r['docs'], list(S))
            self.cache[key] = run(self.m, x, spans, [i in key for i in range(self.n)]).log_softmax(0).cpu().numpy()
        return self.cache[key]


def ns_score(read, S):
    n = read.n
    lp = read(S)
    nec = sum(float(read(tuple(x for x in S if x != e))[e]) for e in S)
    return float(lp[n]) + nec, float(lp[n + 1]) + nec


@torch.no_grad()
def decode(model, r, T, width):
    read = Reader(model, r)
    n = read.n
    beam, cands = [((), 0.0)], {}
    for t in range(min(T, n)):
        nxt = {}
        for S, path in beam:
            lp = read(S)
            for i in [int(i) for i in np.argsort(-lp[:n]) if i not in S][:width]:
                key = frozenset(S + (i,))
                v = path + float(lp[i])
                if key not in nxt or nxt[key][1] < v:
                    nxt[key] = (S + (i,), v)
        # keep the `width` most probable partial completions (states are sets: permutations merged)
        beam = sorted(nxt.values(), key=lambda z: -z[1])[:width]
        for S, _ in beam:
            cands[frozenset(S)] = S
    scored = []
    for S in cands.values():
        top, bot = ns_score(read, S)
        scored.append((top, bot, S))
    best = max(scored, key=lambda z: z[0])
    out = {}
    for k in [2, 3, 4, 5]:
        sub = [z for z in scored if len(z[2]) <= k]
        b = max(sub, key=lambda z: z[0])
        S = list(b[2])
        out[f'k{k}'] = (S + [int(i) for i in np.argsort(-read(b[2])[:n]) if i not in S])[:k]
    S = list(best[2])
    out['order'] = S + [int(i) for i in np.argsort(-read(best[2])[:n]) if i not in S]
    out['dec'] = S
    out['ns_top'] = best[0]
    out['ns_bot_best'] = max(z[1] for z in scored)
    out['reads'] = len(read.cache)
    return out


def main_ns():
    model = ResidualReader(args.enc).to(dev)
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict({k: v.float() for k, v in sd.items()})
    model.eval()
    for path in args.infer.split(','):
        recs = load(path, args.max_infer)
        t0 = time.time()
        res = [decode(model, r, args.steps, W) for r in recs]
        out = {k: [o[k] for o in res] for k in res[0]}
        name = os.path.basename(path).replace('.jsonl', '')
        pickle.dump(out, open(f'{args.out}/pred_{name}_ns{W}.pkl', 'wb'))
        msg = ''
        if recs[0].get('gold'):
            msg = {k: round(100 * np.mean([set(r['gold']) <= set(s) for s, r in zip(out[k], recs)]), 2) for k in ['k2', 'k3', 'k4', 'k5', 'dec']}
            msg['dec_len'] = round(float(np.mean([len(s) for s in out['dec']])), 2)
            msg['SupEM'] = round(100 * np.mean([set(s) == set(r['gold']) for s, r in zip(out['dec'], recs)]), 2)
        print('ns', W, path, len(recs), 'sec', round(time.time() - t0, 1), 'reads/q', round(float(np.mean(out['reads'])), 1), msg, flush=True)


main_ns()
