"""Chain-CE: a same-backbone reproduction of Beam Retrieval (Zhang et al., NAACL'24) as a controlled baseline.

hop t   : score(q, chain_<t, d) = cross-encoder( "<q> </s> e_1 </s> ... e_{t-1}" , d )   for every candidate d
train   : softmax over {gold next hop} U {sampled negatives} (teacher forcing on the annotated chain order)
infer   : beam search over chains (width 2), T hops; ranking = best chain, then last-hop scores
init    : FT-Pointwise checkpoint (same backbone, same data) -> the pairwise analogue of LACUNA's stage 1
usage   : python chain_ce.py --data musique --init ckpt/musique_point/model.pt --out ckpt/mq_chain --epochs 1
          python chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer data/proc/musique_dev.jsonl
"""
import argparse, json, math, os, random, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument('--data', default='musique')
p.add_argument('--enc', default='BAAI/bge-reranker-v2-m3')
p.add_argument('--out', default='ckpt/chain')
p.add_argument('--epochs', type=float, default=1)
p.add_argument('--max_train', type=int, default=0)
p.add_argument('--accum', type=int, default=8)
p.add_argument('--lr', type=float, default=1e-5)
p.add_argument('--neg', type=int, default=5)
p.add_argument('--beam', type=int, default=2)
p.add_argument('--hops', type=int, default=4)
p.add_argument('--doclen', type=int, default=160)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--init', default='')
p.add_argument('--ckpt', default='')
p.add_argument('--infer', default='')
p.add_argument('--dev_n', type=int, default=200)
p.add_argument('--max_infer', type=int, default=0)
args = p.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(args.enc)
BOS, SEP, PAD = tok.bos_token_id, tok.sep_token_id, tok.pad_token_id
_c = {}


def ids_of(text, n):
    k = (text[:80], n)
    if k not in _c:
        _c[k] = tok(text, add_special_tokens=False, truncation=True, max_length=n)['input_ids']
    return _c[k]


def dtext(d):
    return f"{d['title']}\n{d['text']}"


def pair(q, chain, d):
    a = [BOS] + ids_of(q, 64)
    for e in chain:
        a += [SEP] + ids_of(dtext(e), args.doclen)
    return a + [SEP, SEP] + ids_of(dtext(d), args.doclen) + [SEP]


def score(model, seqs):
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), PAD, dtype=torch.long)
    m = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.view(-1).float()


def load(path, n=0):
    out = []
    for l in open(path):
        out.append(json.loads(l))
        if n and len(out) >= n:
            break
    return out


def loss_for(model, r):
    chain_ids = r.get('gold_order') or r['gold']
    docs = r['docs']
    t = random.randrange(len(chain_ids))                      # one hop per example (prefix state)
    pre = [docs[i] for i in chain_ids[:t]]
    tgt = chain_ids[t]
    negs = [i for i in range(len(docs)) if i not in chain_ids[:t + 1] and i != tgt]
    negs = random.sample(negs, min(args.neg, len(negs)))
    cand = [tgt] + negs
    lg = score(model, [pair(r['question'], pre, docs[i]) for i in cand])
    return -F.log_softmax(lg, 0)[0]


@torch.no_grad()
def chain_search(model, r):
    docs, n = r['docs'], len(r['docs'])
    beams = [((), 0.0)]
    last = None
    for t in range(min(args.hops, n)):
        nxt = []
        for ch, sc in beams:
            free = [i for i in range(n) if i not in ch]
            lp = F.log_softmax(score(model, [pair(r['question'], [docs[j] for j in ch], docs[i]) for i in free]), 0).cpu().numpy()
            for i, v in zip(free, lp):
                nxt.append((ch + (i,), sc + float(v), dict(zip(free, lp))))
        nxt.sort(key=lambda z: -z[1])
        beams = [(c, s) for c, s, _ in nxt[:args.beam]]
        last = nxt[0][2]
    best = list(beams[0][0])
    rest = sorted([i for i in range(n) if i not in best], key=lambda i: -last.get(i, -1e9))
    return {'order': best + rest, 'beams': [list(c) for c, _ in beams]}


def main():
    model = AutoModelForSequenceClassification.from_pretrained(args.enc, attn_implementation='sdpa').to(dev)
    if args.init:
        sd = torch.load(args.init, map_location='cpu')
        model.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith('cls.')})
        print('init from', args.init, flush=True)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    os.makedirs(args.out, exist_ok=True)
    if args.infer:
        model.eval()
        for path in args.infer.split(','):
            recs = load(path, args.max_infer)
            t0 = time.time()
            outs = [chain_search(model, r) for r in recs]
            name = os.path.basename(path).replace('.jsonl', '')
            pickle.dump(outs, open(f'{args.out}/pred_{name}.pkl', 'wb'))
            hit = {k: round(100 * np.mean([set(r['gold']) <= set(o['order'][:k]) for o, r in zip(outs, recs)]), 2) for k in [2, 3, 4, 5]}
            print('infer', path, len(recs), 'sec', round(time.time() - t0, 1), hit, flush=True)
        return
    model.gradient_checkpointing_enable()
    train = load(f'data/proc/{args.data}_train.jsonl', args.max_train)
    devset = random.Random(0).sample(load(f'data/proc/{args.data}_dev.jsonl'), args.dev_n)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, foreach=False)
    steps = int(math.ceil(len(train) * args.epochs / args.accum))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'))
    order = []
    while len(order) < steps * args.accum:
        perm = list(range(len(train))); random.shuffle(perm); order += perm
    model.train()
    t0, run_l, best = time.time(), 0.0, -1
    print('train', len(train), 'opt steps', steps, flush=True)
    for s in range(steps):
        for a in range(args.accum):
            loss = loss_for(model, train[order[s * args.accum + a]]) / args.accum
            loss.backward(); run_l += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % 100 == 0:
            print(f'step {s + 1}/{steps} loss {run_l / 100:.4f} {time.time() - t0:.0f}s', flush=True); run_l = 0.0
        if (s + 1) % max(1, steps // 2) == 0 or s + 1 == steps:
            model.eval()
            outs = [chain_search(model, r) for r in devset]
            model.train()
            ev = {k: round(100 * np.mean([set(r['gold']) <= set(o['order'][:k]) for o, r in zip(outs, devset)]), 2) for k in [2, 3, 4, 5]}
            print('dev', s + 1, ev, flush=True)
            if sum(ev.values()) > best:
                best = sum(ev.values())
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f'{args.out}/model.pt')
                print('saved', flush=True)


if __name__ == '__main__':
    main()
