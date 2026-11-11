import json, pickle, numpy as np, sys
recs = [json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
P = pickle.load(open('ckpt/musique_prove/pred_musique_dev.pkl', 'rb'))
sig = lambda x: 1 / (1 + np.exp(-x))
act, nact, same, cover, nullm = [], [], [], [], []
for o, r in list(zip(P, recs))[::7][:300]:
    g = sig(o['gam']); p = np.exp(o['lp']); N = p.shape[1] - 1
    a = g > 0.5
    act.append(a.sum()); nact.append(len(r['gold']))
    w = p.argmax(1)
    same.append(len(set(w[a])) < a.sum())
    cover.append(len(set(w[a]) & set(r['gold'])) / len(r['gold']))
    nullm.append(p[:, -1].mean())
print('mean active req', np.mean(act), 'mean gold', np.mean(nact))
print('active>gold frac', np.mean(np.array(act) > np.array(nact)), 'active<gold', np.mean(np.array(act) < np.array(nact)))
print('dup witnesses among active', np.mean(same), 'gold covered by MAP', np.mean(cover), 'null mass', np.mean(nullm))
for o, r in list(zip(P, recs))[1300:1303]:
    g = sig(o['gam']); p = np.exp(o['lp'])
    print('gold', r['gold'], r['type'])
    for k in range(len(g)):
        top = np.argsort(-p[k])[:3]
        print(f'  k{k} g={g[k]:.2f} top', [(int(i) if i < p.shape[1]-1 else 'NULL', round(float(p[k, i]), 2)) for i in top])
