"""Equal-size comparison with Beam Retrieval: for every question, each ranker returns exactly as many passages as
Beam's own (self-stopped) chain. writes results/metrics/beam_matched.json"""
import json, pickle, sys
import numpy as np
R = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
B = pickle.load(open('results/scores/musique_dev_beam.pkl', 'rb'))
own = B['k5'] if isinstance(B, dict) else B
size = [len(s) for s in own]
def ranking(path):
    P = pickle.load(open(path, 'rb'))
    if 'order' in P[0]:
        return [o['order'] for o in P]
    return [list(np.argsort(-o['rho'])) for o in P]
res = {'beam': {}}
def score(sets, name):
    out = {'CER': 100 * np.mean([set(r['gold']) <= set(s) for s, r in zip(sets, R)]),
           'SupEM': 100 * np.mean([set(r['gold']) == set(s) for s, r in zip(sets, R)])}
    for h in '234':
        idx = [i for i, r in enumerate(R) if len(r['gold']) == int(h)]
        out[f'CER_{h}hop'] = 100 * np.mean([set(R[i]['gold']) <= set(sets[i]) for i in idx])
    res[name] = {k: round(float(v), 2) for k, v in out.items()}
score(own, 'beam')
import os
for spec in sys.argv[1:]:
    name, path = spec.split('=')
    if not os.path.exists(path):
        print('skip', name, path); continue
    rk = ranking(path)
    score([list(o[:b]) for o, b in zip(rk, size)], name)
res['mean_size'] = round(float(np.mean(size)), 3)
json.dump(res, open('results/metrics/beam_matched.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
