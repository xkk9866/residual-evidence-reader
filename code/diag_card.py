import json, pickle, numpy as np
from sklearn.metrics import roc_auc_score
R=[json.loads(l) for l in open('data/proc/musique_dev.jsonl')]
P=pickle.load(open('ckpt/mq_lacp/pred_musique_dev.pkl','rb'))
def halt_size(o):
    return next((t for t,c in enumerate(o['card']) if int(np.argmax(c))==0), len(o['card'])-1)
def n0(o): return int(np.argmax(o['card'][0]))
for name,f in [('size-only',halt_size),('size@0 plan',lambda o: min(max(n0(o),1),5))]:
    S=[o['order'][:f(o)] for o in P]
    print(name,'CER %.1f EM %.1f size %.2f'%(100*np.mean([set(r['gold'])<=set(s) for s,r in zip(S,R)]),100*np.mean([set(r['gold'])==set(s) for s,r in zip(S,R)]),np.mean([len(s) for s in S])))
print('hop-count acc at S=empty', np.mean([n0(o)==len(r['gold']) for o,r in zip(P,R)]))
F=[json.loads(l) for l in open('data/proc/musique_full_dev.jsonl')]
Q=pickle.load(open('ckpt/mq_lacp/pred_musique_full_dev.pkl','rb'))
y=[r['answerable'] for r in F]
print('AUROC max p(N=0)', roc_auc_score(y,[max(c[0] for c in o['card'][1:]) for o in Q]), 'min logpi(OUT)', roc_auc_score(y,[-max(o['bot'][1:]) for o in Q]))
