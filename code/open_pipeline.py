"""Open-corpus setting (HippoRAG 1000-question benchmarks) and Unmet-Requirement Retrieval (URR).

  embed  DS                 : bge-large-en-v1.5 embeddings of corpus + questions
  pool   DS N               : dense top-N pools  -> data/proc/openpool_{DS}_N{N}.jsonl
  urrdata TRAIN_DS CKPT n   : cache (unmet requirement vector, question emb, missing-doc emb, pool embs) from
                              world-counterfactual pools C\\{d} of the distractor train set
  urrtrain TRAIN_DS         : train the requirement->query projection (InfoNCE)
  loop   DS CKPT N M T TAG  : closed loop: PROVE on pool -> if Phi(pool)<1/2, fetch top-M docs for the most
                              unmet requirement; T rounds -> data/proc/openloop_{DS}_{TAG}.jsonl + preds
"""
import json, os, pickle, sys, random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

BGE = 'BAAI/bge-large-en-v1.5'
QP = 'Represent this sentence for searching relevant passages: '


class BGE_Enc:
    def __init__(self):
        self.tk = AutoTokenizer.from_pretrained(BGE)
        self.m = AutoModel.from_pretrained(BGE, torch_dtype=torch.float16).cuda().eval()

    @torch.no_grad()
    def __call__(self, texts, bs=128):
        out = []
        for i in range(0, len(texts), bs):
            b = self.tk(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors='pt').to('cuda')
            out.append(F.normalize(self.m(**b).last_hidden_state[:, 0].float(), dim=-1).cpu())
        return torch.cat(out).numpy()


def dstr(d):
    return f"{d['title']}\n{d['text']}"


def load_open(ds):
    qs = [json.loads(l) for l in open(f'data/proc/open_{ds}.jsonl')]
    corpus = [json.loads(l) for l in open(f'data/proc/open_{ds}_corpus.jsonl')]
    return qs, corpus


def make_rec(q, corpus, ids):
    gold_set = q['gold_corpus']
    return {'id': q['id'], 'question': q['question'], 'answers': q['answers'], 'type': q['type'],
            'docs': [{'title': corpus[i]['title'], 'text': corpus[i]['text']} for i in ids],
            'ntok': [corpus[i]['ntok'] for i in ids], 'cid': [int(i) for i in ids],
            'gold': [ids.index(g) for g in gold_set if g in ids],
            'gold_missing': sum(g not in ids for g in gold_set)}


def cmd_embed(ds):
    qs, corpus = load_open(ds)
    enc = BGE_Enc()
    np.save(f'data/proc/open_{ds}_cemb.npy', enc([dstr(c) for c in corpus]).astype(np.float16))
    np.save(f'data/proc/open_{ds}_qemb.npy', enc([QP + q['question'] for q in qs]).astype(np.float16))
    print('embedded', ds, len(corpus), len(qs))


def cmd_pool(ds, N):
    qs, corpus = load_open(ds)
    ce = np.load(f'data/proc/open_{ds}_cemb.npy').astype(np.float32)
    qe = np.load(f'data/proc/open_{ds}_qemb.npy').astype(np.float32)
    top = np.argsort(-(qe @ ce.T), 1)[:, :N]
    with open(f'data/proc/openpool_{ds}_N{N}.jsonl', 'w') as f:
        for q, t in zip(qs, top):
            f.write(json.dumps(make_rec(q, corpus, [int(x) for x in t])) + '\n')
    print('pool', ds, N)


class URRHead(torch.nn.Module):
    """query = normalize(e_q + W r_k*): starts as the plain question query (W=0) and learns to shift
    toward the evidence demanded by the unmet requirement."""
    def __init__(self, d=512, D=1024):
        super().__init__()
        self.W = torch.nn.Linear(d, D, bias=False)
        torch.nn.init.zeros_(self.W.weight)

    def forward(self, r, eq):
        return F.normalize(eq + self.W(r), dim=-1)


def unmet_k(o):
    """the requirement most likely needed AND witnessed by nobody in the pool: argmax_k g_k p_k(null)"""
    logg = -np.logaddexp(0, -o['gam'])
    return int(np.argmax(logg + o['lp'][:, -1]))


def cmd_urrdata(train_ds, ckpt, n):
    from infer_lib import load_prove, Runner
    recs = [json.loads(l) for l in open(f'data/proc/{train_ds}_train.jsonl')]
    random.seed(1); random.shuffle(recs); recs = recs[:n]
    run = Runner(load_prove(ckpt))
    enc = BGE_Enc()
    out = []
    for j, r in enumerate(recs):
        res = run.run(r['question'], r['docs'], drop_variants=r['gold'], want_r=True)
        de = enc([dstr(d) for d in r['docs']])
        eq = enc([QP + r['question']])[0]
        for gi, g in enumerate(r['gold']):
            o = res[1 + gi]
            k = unmet_k(o)
            out.append({'r': o['r'][k].astype(np.float16), 'eq': eq.astype(np.float16), 'pos': g,
                        'pool': de.astype(np.float16)})
        if j % 500 == 0:
            print('urrdata', j, flush=True)
    pickle.dump(out, open(f'ckpt/urr_{train_ds}_data.pkl', 'wb'))


def cmd_urrtrain(train_ds, epochs=3, tau=0.02):
    data = pickle.load(open(f'ckpt/urr_{train_ds}_data.pkl', 'rb'))
    random.seed(0); random.shuffle(data)
    nval = 500
    val, tr = data[:nval], data[nval:]
    head = URRHead().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.0)

    def batch_loss(items):
        r = torch.tensor(np.stack([x['r'] for x in items]), dtype=torch.float32, device='cuda')
        eq = torch.tensor(np.stack([x['eq'] for x in items]), dtype=torch.float32, device='cuda')
        q = head(r, eq)
        loss, acc = 0.0, 0.0
        for i, x in enumerate(items):
            P = torch.tensor(x['pool'], dtype=torch.float32, device='cuda')        # hard negatives = rest of pool
            others = torch.cat([torch.tensor(y['pool'][y['pos']], dtype=torch.float32, device='cuda')[None]
                                for j, y in enumerate(items) if j != i])          # in-batch negatives
            cand = torch.cat([P, others])
            s = cand @ q[i] / tau
            loss = loss + F.cross_entropy(s[None], torch.tensor([x['pos']], device='cuda'))
            acc += float(s.argmax().item() == x['pos'])
        return loss / len(items), acc / len(items)

    def evaluate():
        with torch.no_grad():
            return batch_loss(val[:256])[1]
    print('val acc@1 before', evaluate(), flush=True)
    for ep in range(epochs):
        random.shuffle(tr)
        for i in range(0, len(tr), 64):
            loss, _ = batch_loss(tr[i:i + 64])
            opt.zero_grad(); loss.backward(); opt.step()
        print('epoch', ep, 'val acc@1', evaluate(), flush=True)
    torch.save(head.state_dict(), f'ckpt/urr_{train_ds}_head.pt')


def cmd_loop(ds, ckpt, N, M, T, tag, train_ds):
    from infer_lib import load_prove, Runner
    from select_alg import pool_logphi
    qs, corpus = load_open(ds)
    ce = torch.tensor(np.load(f'data/proc/open_{ds}_cemb.npy').astype(np.float32), device='cuda')
    qe = np.load(f'data/proc/open_{ds}_qemb.npy').astype(np.float32)
    run = Runner(load_prove(ckpt))
    head = URRHead().cuda()
    head.load_state_dict(torch.load(f'ckpt/urr_{train_ds}_head.pt'))
    recs, preds, stats = [], [], []
    for qi, q in enumerate(qs):
        ids = [int(x) for x in torch.topk(ce @ torch.tensor(qe[qi], device='cuda'), N).indices.tolist()]
        rounds = 0
        while True:
            rec = make_rec(q, corpus, ids)
            o = run.run(q['question'], rec['docs'], want_r=True)[0]
            phi = float(np.exp(pool_logphi(o['gam'], o['lp'])))
            if phi >= 0.5 or rounds >= T:
                break
            k = unmet_k(o)
            with torch.no_grad():
                qv = head(torch.tensor(o['r'][k], device='cuda'), torch.tensor(qe[qi], device='cuda'))
                sc = ce @ qv
                sc[ids] = -1e9
                new = torch.topk(sc, M).indices.tolist()
            ids = ids + [int(x) for x in new]
            rounds += 1
        o.pop('r', None)
        recs.append(rec); preds.append(o); stats.append(rounds)
        if qi % 200 == 0:
            print(ds, qi, 'avg rounds', np.mean(stats), flush=True)
    with open(f'data/proc/openloop_{ds}_{tag}.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + '\n')
    os.makedirs(f'ckpt/open_{tag}', exist_ok=True)
    pickle.dump(preds, open(f'ckpt/open_{tag}/pred_openloop_{ds}_{tag}.pkl', 'wb'))
    print('loop done', ds, 'avg rounds', np.mean(stats), 'avg pool', np.mean([len(r['docs']) for r in recs]))


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'embed':
        cmd_embed(sys.argv[2])
    elif c == 'pool':
        cmd_pool(sys.argv[2], int(sys.argv[3]))
    elif c == 'urrdata':
        cmd_urrdata(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif c == 'urrtrain':
        cmd_urrtrain(sys.argv[2])
    elif c == 'loop':
        cmd_loop(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7], sys.argv[8])
