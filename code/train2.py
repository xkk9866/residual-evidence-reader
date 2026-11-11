"""Train / infer PROVE v2 (joint reader + transport witnesses + sufficiency circuit) and its
same-reader per-document BCE ablation.

Each training question yields ONE pool, drawn 1:1 (as in MuSiQue-Full) between
  * the factual pool C            -> L = -log Phi(S*) - mean_d log(1 - Phi(C \\ {d}))     (lattice extremes)
  * a world-deleted pool C\\{d}   -> L = -log(1 - Phi_pool(C\\{d}))                         (missing evidence)
"""
import argparse, json, math, os, random, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model2 import ProveJoint, Circuit, log1mexp
from select_alg import greedy_select

p = argparse.ArgumentParser()
p.add_argument('--data', default='musique')
p.add_argument('--head', default='circuit', help='circuit | bce')
p.add_argument('--enc', default='BAAI/bge-reranker-v2-m3')
p.add_argument('--out', default='ckpt/tmp2')
p.add_argument('--epochs', type=float, default=2)
p.add_argument('--max_train', type=int, default=0)
p.add_argument('--accum', type=int, default=8)
p.add_argument('--lr_enc', type=float, default=1e-5)
p.add_argument('--lr_head', type=float, default=1e-4)
p.add_argument('--K', type=int, default=6)
p.add_argument('--doclen', type=int, default=160)
p.add_argument('--no_transport', action='store_true')
p.add_argument('--no_world', action='store_true')
p.add_argument('--no_sel', action='store_true')
p.add_argument('--loss', default='proof', help='proof (Viterbi proof likelihood) | lattice (set-level extremes)')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--infer', default='')
p.add_argument('--ckpt', default='')
p.add_argument('--dev_n', type=int, default=400)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(args.enc)
MASK, BOS, SEP = tok.mask_token_id, tok.bos_token_id, tok.sep_token_id


def load(path, n=0):
    out = []
    for l in open(path):
        out.append(json.loads(l))
        if n and len(out) >= n:
            break
    return out


_cache = {}


def doc_ids(d):
    key = (d['title'], d['text'][:64])
    if key not in _cache:
        _cache[key] = tok(f"{d['title']}\n{d['text']}", add_special_tokens=False, truncation=True,
                          max_length=args.doclen)['input_ids']
    return _cache[key]


def build(q, docs):
    ids = [BOS] + [MASK] * args.K + tok(q, add_special_tokens=False, truncation=True, max_length=64)['input_ids'] + [SEP]
    spans = []
    for d in docs:
        t = doc_ids(d)
        spans.append((len(ids), len(ids) + len(t)))
        ids += t + [SEP]
    x = torch.tensor([ids], device=dev)
    return x, torch.ones_like(x), spans


def forward(model, q, docs):
    x, a, spans = build(q, docs)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        r, h, spans = model.read(x, a, spans)
        return model.witness(r, h, spans)


def loss_for(model, r):
    n = len(r['docs'])
    order = list(range(n)); random.shuffle(order)
    gold = set(r['gold'])
    world = (args.head == 'circuit' and not args.no_world and random.random() < 0.5)
    if world:
        drop = random.choice(r['gold'])
        order = [i for i in order if i != drop]
    docs = [r['docs'][i] for i in order]
    y = torch.tensor([float(i in gold) for i in order], device=dev)
    out = forward(model, r['question'], docs)
    if args.head == 'bce':
        return F.binary_cross_entropy_with_logits(out, y)
    gam, lp = out
    if args.loss == 'proof':
        # Viterbi likelihood of the observed minimal proof under the latent witness assignment:
        # targets = gold docs in the pool (+ the null item when a gold doc was deleted from the world)
        tgt = [j for j, i in enumerate(order) if i in gold] + ([lp.shape[1] - 1] if world else [])
        lg, l1g = F.logsigmoid(gam), F.logsigmoid(-gam)
        cost = -(lg.unsqueeze(1) + lp[:, tgt]) + l1g.unsqueeze(1)          # matched minus unmatched
        rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
        rows_t = torch.tensor(rows, device=dev)
        loss = -l1g.sum() + cost[rows_t, torch.tensor(cols, device=dev)].sum()
        return loss / len(tgt)
    if world:
        return -log1mexp(Circuit.log_phi_pool(gam, lp))
    terms = [-Circuit.log_phi(gam, lp, y)]
    if not args.no_sel:
        g = [j for j, i in enumerate(order) if i in gold]
        ms = torch.ones(len(g), len(order), device=dev)
        ms[torch.arange(len(g)), torch.tensor(g)] = 0
        lphi = Circuit.log_phi(gam.unsqueeze(0).expand(len(g), -1), lp.unsqueeze(0).expand(len(g), -1, -1), ms)
        terms.append(-log1mexp(lphi).mean())
    return sum(terms)


@torch.no_grad()
def predict(model, recs):
    model.eval()
    res = []
    for r in recs:
        out = forward(model, r['question'], r['docs'])
        if args.head == 'bce':
            res.append({'rho': out.cpu().numpy()})
        else:
            res.append({'gam': out[0].cpu().numpy(), 'lp': out[1].cpu().numpy()})
    model.train()
    return res


def quick_eval(outs, recs):
    hit = {2: 0, 3: 0, 4: 0, 5: 0}
    for o, r in zip(outs, recs):
        n = len(r['docs'])
        for k in hit:
            S = greedy_select(o['gam'], o['lp'], np.ones(n), k) if 'gam' in o else list(np.argsort(-o['rho'])[:k])
            hit[k] += set(r['gold']) <= set(S)
    return {f'CER@{k}': round(100 * v / len(recs), 2) for k, v in hit.items()}


def main():
    model = ProveJoint(args.enc, K=args.K, transport=not args.no_transport, head=args.head).to(dev)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    os.makedirs(args.out, exist_ok=True)
    if args.infer:
        for path in args.infer.split(','):
            recs = load(path)
            t = time.time()
            outs = predict(model, recs)
            name = os.path.basename(path).replace('.jsonl', '')
            pickle.dump(outs, open(f'{args.out}/pred_{name}.pkl', 'wb'))
            print('infer', path, len(recs), 'sec', round(time.time() - t, 1),
                  quick_eval(outs, recs) if recs[0].get('gold') else '', flush=True)
        return
    model.enc.gradient_checkpointing_enable()
    train = load(f'data/proc/{args.data}_train.jsonl', args.max_train)
    devset = random.Random(0).sample(load(f'data/proc/{args.data}_dev.jsonl'), args.dev_n)
    enc_p = list(model.enc.parameters())
    eid = {id(x) for x in enc_p}
    head_p = [x for x in model.parameters() if id(x) not in eid]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': args.lr_enc}, {'params': head_p, 'lr': args.lr_head}], weight_decay=0.01, foreach=False)
    steps = int(math.ceil(len(train) * args.epochs / args.accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'))
    print('train', len(train), 'opt steps', steps, flush=True)
    order = []
    while len(order) < steps * args.accum:
        perm = list(range(len(train))); random.shuffle(perm); order += perm
    model.train()
    t0, run, best = time.time(), 0.0, -1
    for s in range(steps):
        for a in range(args.accum):
            loss = loss_for(model, train[order[s * args.accum + a]]) / args.accum
            loss.backward()
            run += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % 100 == 0:
            print(f'step {s + 1}/{steps} loss {run / 100:.4f} {time.time() - t0:.0f}s', flush=True); run = 0.0
        if (s + 1) % max(1, steps // 4) == 0 or s + 1 == steps:
            ev = quick_eval(predict(model, devset), devset)
            print('dev', s + 1, ev, flush=True)
            if sum(ev.values()) > best:
                best = sum(ev.values())
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f'{args.out}/model.pt')
                print('saved', flush=True)


if __name__ == '__main__':
    main()
