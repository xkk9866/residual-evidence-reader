"""Equal-opportunity closed-loop retrieval for ANY reader (file based, one round at a time).

  python loop_build.py DS METHOD MODE ROUND

Reads the round-ROUND pool  data/proc/loop_{DS}_{METHOD}_{MODE}_r{ROUND}.jsonl  (round 0 = openpool_{DS}_N20) and the
reader's predictions on it, decides per question whether to retrieve again, and writes round ROUND+1.
Every method gets the same query construction (question + its secured passages, evidence-as-query / pseudo-relevance
feedback with bge-large) and the same number of rounds; only what each reader secures and when it asks differ:
  lac*   : LACUNA. secured = evidence set closed by the reader (halt_any); adaptive: retrieve unless verdict = TOP
  static : joint reader +1. secured = passages with p > threshold tuned on the 400 selection questions
           adaptive: retrieve unless the same-backbone answerability classifier says answerable
  chain  : Chain-CE. secured = first m passages of its best chain, m tuned on the selection questions
           adaptive: as static (answerability classifier)
  dense  : no reader. secured = top-m dense passages, m tuned on the selection questions (iterative PRF)
  SEL-ASK: cross-swap of the two decisions, e.g. static-lacc = static reader secures, LACUNA's verdict asks;
           lacc-anscls = LACUNA secures, the answerability classifier asks (separates selecting from stopping)
fixed mode re-retrieves for every question in every round and secures at least one passage.
"""
import json, os, pickle, random, sys
import numpy as np
import torch

sys.path.insert(0, 'code')
from open_pipeline import BGE_Enc, make_rec, dstr, QP

DS, METHOD, MODE, ROUND = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
POOL_N = 20
TAG = f'loop_{DS}_{METHOD}_{MODE}'
PRED = {'lac': 'ckpt/mq_lact', 'lach': 'ckpt/mq_lacr_ev', 'lacc': 'ckpt/mq_lacc', 'delta': 'ckpt/mq_delta', 'static': 'ckpt/mq_jbce3', 'chain': 'ckpt/mq_chain'}
ANS = 'ckpt/mq_anscls'
RES = ('lac', 'delta')      # residual readers (one-sided LACUNA variants, two-sided delta): verdict-driven, secured = closed set


def closed_set(o):
    """the secured passages at the verdict state: two-sided readers record the state itself (the walk can release
    passages, so it is not a prefix of the ranking); one-sided readers secure the first halt_any picks"""
    return list(o['set_any']) if 'set_any' in o else list(o['order'][:o['halt_any']])


def pool_name(r):
    return f'openpool_{DS}_N20' if r == 0 else f'{TAG}_r{r}'


def tuned_cut():
    """selection-split tuning (400 MuSiQue dev questions, never the open-corpus questions) of the static threshold
    and of the size m used by Chain-CE / dense: the value maximising exact supporting-set match"""
    path = 'results/metrics/loop_tuned.json'
    if os.path.exists(path):
        return json.load(open(path))
    recs = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
    sel = random.Random(0).sample(range(len(recs)), 400)
    J = pickle.load(open('ckpt/mq_jbce3/pred_musique_dev.pkl', 'rb'))
    C = pickle.load(open('ckpt/mq_chain/pred_musique_dev.pkl', 'rb'))
    D = pickle.load(open('results/scores/musique_dev_dense.pkl', 'rb'))
    em = lambda sets: np.mean([set(sets[i]) == set(recs[i]['gold']) for i in sel])
    def thr_set(o, th):
        p = 1 / (1 + np.exp(-o['rho']))
        return [int(i) for i in np.argsort(-p) if p[i] > th] or [int(np.argmax(p))]
    grid = np.linspace(0.05, 0.95, 91)
    th = float(max(grid, key=lambda t: em([thr_set(o, t) for o in J])))
    mc = max(range(1, 6), key=lambda m: em([o['order'][:m] for o in C]))
    md = max(range(1, 6), key=lambda m: em([list(np.argsort(-o['rho'])[:m]) for o in D]))
    res = {'static_thr': round(th, 2), 'chain_m': mc, 'dense_m': md}
    json.dump(res, open(path, 'w'))
    return res


def secured_and_ask(o, r, a, cut):
    """(secured positions in the pool, ask for more evidence?)"""
    if '-' in METHOD:
        sel, how = METHOD.split('-')
        S = (closed_set(o) if sel.startswith(RES) else
             [int(i) for i in np.argsort(-o['rho']) if 1 / (1 + np.exp(-o['rho'][i])) > cut['static_thr']])
        ask = a['verdict3'] != 1 if how.startswith(RES) else a['top'][0] < a['bot'][0]
    elif METHOD.startswith(RES):
        S = closed_set(o)
        ask = o['verdict3'] != 1
    elif METHOD == 'static':
        p = 1 / (1 + np.exp(-o['rho']))
        S = [int(i) for i in np.argsort(-p) if p[i] > cut['static_thr']]
        ask = a is None or a['top'][0] < a['bot'][0]
    elif METHOD == 'chain':
        S = o['order'][:cut['chain_m']]
        ask = a is None or a['top'][0] < a['bot'][0]
    else:
        S = list(range(cut['dense_m']))
        ask = True
    S = S or ([o['order'][0]] if 'order' in o else [int(np.argmax(o['rho']))] if 'rho' in o else [0])
    if MODE == 'fixed':
        ask = True
    return [int(i) for i in S], ask


def main():
    cut = tuned_cut()
    qs = {q['id']: q for q in (json.loads(l) for l in open(f'data/proc/open_{DS}.jsonl'))}
    corpus = [json.loads(l) for l in open(f'data/proc/open_{DS}_corpus.jsonl')]
    recs = [json.loads(l) for l in open(f'data/proc/{pool_name(ROUND)}.jsonl')]
    if METHOD == 'dense':
        P = [{} for _ in recs]
    else:
        P = pickle.load(open(f"{PRED[METHOD.split('-')[0]]}/pred_{pool_name(ROUND)}.pkl", 'rb'))
    A = pickle.load(open(f'{ANS}/pred_{pool_name(ROUND)}.pkl', 'rb')) if (METHOD in ('static', 'chain') and MODE == 'adaptive') else [None] * len(recs)
    if '-' in METHOD:
        how = METHOD.split('-')[1]
        A = pickle.load(open(f"{PRED[how] if how in PRED else ANS}/pred_{pool_name(ROUND)}.pkl", 'rb'))
    ce = torch.tensor(np.load(f'data/proc/open_{DS}_cemb.npy').astype(np.float32), device='cuda')
    bge = BGE_Enc()
    asks = [secured_and_ask(o, r, a, cut) for o, r, a in zip(P, recs, A)]
    todo = [i for i, (S, ask) in enumerate(asks) if ask]
    texts = [QP + recs[i]['question'] + ' ' + ' '.join(dstr(recs[i]['docs'][j]) for j in asks[i][0]) for i in todo]
    qv = torch.tensor(bge(texts), device='cuda') if texts else None
    out, stat = [], json.load(open(f'results/metrics/{TAG}_stats.json')) if ROUND else {'asked': []}
    tset = set(todo)
    for i, r in enumerate(recs):
        if i in tset:
            S = [r['cid'][j] for j in asks[i][0]]
            v = qv[todo.index(i)]
            new = [int(c) for c in torch.topk(ce @ v, POOL_N + len(S)).indices.tolist() if int(c) not in S]
            out.append(make_rec(qs[r['id']], corpus, S + new[:POOL_N - len(S)]))
        else:
            out.append(r)
    stat['asked'] = stat['asked'][:ROUND] + [len(todo) / len(recs)]
    stat['pool_complete'] = stat.get('pool_complete', [])[:ROUND] + [float(np.mean([r['gold_missing'] == 0 for r in recs]))]
    stat['pool_complete_next'] = float(np.mean([r['gold_missing'] == 0 for r in out]))
    # how often a request for more evidence was issued on a pool that really lacked a gold passage
    stat.setdefault('ask_precision', [])
    stat['ask_precision'] = stat['ask_precision'][:ROUND] + [float(np.mean([recs[i]['gold_missing'] > 0 for i in todo])) if todo else 0.0]
    stat['secured_size'] = stat.get('secured_size', [])[:ROUND] + [float(np.mean([len(S) for S, _ in asks]))]
    with open(f'data/proc/{TAG}_r{ROUND + 1}.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r) + chr(10))
    json.dump(stat, open(f'results/metrics/{TAG}_stats.json', 'w'))
    print(TAG, 'round', ROUND, 'asked', round(100 * len(todo) / len(recs), 1), 'complete-pool next', round(100 * stat['pool_complete_next'], 1), flush=True)


if __name__ == '__main__':
    main()
