"""Paired bootstrap of AUROC / Macro-F1 differences on MuSiQue-Full: LACUNA vs other answerability scorers.
writes results/metrics/ans_sig.json"""
import json, pickle
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

R = [json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
y = np.array([r['answerable'] for r in R], dtype=int)


def logsig(x):
    return -np.logaddexp(0, -np.asarray(x, dtype=np.float64))


L = pickle.load(open('ckpt/mq_lacr/pred_musique_full_dev.pkl', 'rb'))
A = pickle.load(open('ckpt/mq_anscls/pred_musique_full_dev.pkl', 'rb'))
J = pickle.load(open('ckpt/mq_jbce3/pred_musique_full_dev.pkl', 'rb'))
s_l = np.array([max(o['top'][1:]) for o in L])
d_l = np.array([any(o['top'][t] >= max(o['bot'][t], o['dmax'][t]) for t in range(1, len(o['top']))) for o in L]).astype(int)
s_a = np.array([o['top'][0] - o['bot'][0] for o in A]); d_a = (s_a > 0).astype(int)
s_j = np.array([np.sort(logsig(o['rho']))[-2:].sum() for o in J]); d_j = (s_j > np.log(0.25)).astype(int)
rng = np.random.default_rng(0)
out = {}
for name, s2, d2 in [('anscls', s_a, d_a), ('jbce3', s_j, d_j)]:
    da, df = [], []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        da.append(roc_auc_score(y[i], s_l[i]) - roc_auc_score(y[i], s2[i]))
        df.append(f1_score(y[i], d_l[i], average='macro') - f1_score(y[i], d2[i], average='macro'))
    out[name] = {'dAUROC': round(100 * (roc_auc_score(y, s_l) - roc_auc_score(y, s2)), 2),
                 'dAUROC_ci': [round(100 * np.percentile(da, 2.5), 2), round(100 * np.percentile(da, 97.5), 2)],
                 'p_AUROC_le0': float(np.mean(np.array(da) <= 0)),
                 'dF1_ci': [round(100 * np.percentile(df, 2.5), 2), round(100 * np.percentile(df, 97.5), 2)]}
json.dump(out, open('results/metrics/ans_sig.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
