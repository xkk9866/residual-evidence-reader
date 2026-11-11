"""Train / infer PROVE and same-backbone baselines on distractor-style pools.

heads:
  circuit   : PROVE (sufficiency circuit + lattice-extreme counterfactual training)
  setbce    : same requirement decoder, trained with per-document BCE  (objective ablation)
  pointwise : fine-tuned cross-encoder, BCE                             (FT-Pointwise baseline)
"""
import argparse, json, math, os, random, time, pickle
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from model import PROVE, Circuit, log1mexp
from select_alg import greedy_select

p = argparse.ArgumentParser()
p.add_argument('--data', default='musique')
p.add_argument('--head', default='circuit')
p.add_argument('--enc', default='BAAI/bge-reranker-v2-m3')
p.add_argument('--out', default='ckpt/tmp')
p.add_argument('--epochs', type=float, default=2)
p.add_argument('--max_train', type=int, default=0)
p.add_argument('--pool', type=int, default=12, help='max docs per training pool (gold + sampled distractors)')
p.add_argument('--qpb', type=int, default=2, help='questions per micro-batch')
p.add_argument('--accum', type=int, default=4)
p.add_argument('--lr_enc', type=float, default=1e-5)
p.add_argument('--lr_head', type=float, default=1e-4)
p.add_argument('--K', type=int, default=6)
p.add_argument('--layers', type=int, default=3)
p.add_argument('--maxlen', type=int, default=256)
p.add_argument('--no_world', action='store_true')
p.add_argument('--no_sel', action='store_true')
p.add_argument('--no_suff', action='store_true')
p.add_argument('--no_rho', action='store_true')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--infer', default='', help='comma list of jsonl files to run inference on')
p.add_argument('--ckpt', default='')
p.add_argument('--dev_n', type=int, default=400)
args = p.parse_args()

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device('cuda')
tok = AutoTokenizer.from_pretrained(args.enc)


def load(path, n=0):
    out = []
    with open(path) as f:
        for l in f:
            out.append(json.loads(l))
            if n and len(out) >= n:
                break
    return out


def tokenize(q, docs):
    a = tok([q] * len(docs), [f"{d['title']}\n{d['text']}" for d in docs], truncation='only_second',
            max_length=args.maxlen, padding=True, return_tensors='pt')
    b = tok([q], truncation=True, max_length=64, return_tensors='pt')
    return a, b


def sample_pool(r):
    n = len(r['docs'])
    gold = list(r['gold'])
    neg = [i for i in range(n) if i not in gold]
    random.shuffle(neg)
    keep = gold + neg[:max(0, args.pool - len(gold))]
    random.shuffle(keep)
    return keep


def run_batch(model, items, train=True):
    """items: list of (question, docs, gold_local). returns loss (train) or outputs."""
    toks = [tokenize(q, docs) for q, docs, _ in items]
    maxL = max(a['input_ids'].shape[1] for a, _ in toks)
    ids, att, qids, qatt = [], [], [], []
    for a, b in toks:
        ids.append(F.pad(a['input_ids'], (0, maxL - a['input_ids'].shape[1]), value=tok.pad_token_id))
        att.append(F.pad(a['attention_mask'], (0, maxL - a['attention_mask'].shape[1])))
    qL = maxL
    for _, b in toks:
        qids.append(F.pad(b['input_ids'], (0, qL - b['input_ids'].shape[1]), value=tok.pad_token_id))
        qatt.append(F.pad(b['attention_mask'], (0, qL - b['attention_mask'].shape[1])))
    nd = [x.shape[0] for x in ids]
    ids = torch.cat(ids + qids).to(dev)
    att = torch.cat(att + qatt).to(dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        hs, rho = model.encode(ids, att)
    hs, rho = hs.float(), rho.float()
    nq = len(items)
    D = sum(nd)
    outs, losses = [], []
    off = 0
    for j, (q, docs, gold) in enumerate(items):
        h = hs[off:off + nd[j]]
        r = rho[off:off + nd[j]]
        tm = att[off:off + nd[j]].bool()
        # mask the question part + special tokens of the pair? keep all tokens: question-aware states
        qh = hs[D + j]
        qm = att[D + j].bool()
        off += nd[j]
        y = torch.zeros(nd[j], device=dev)
        y[gold] = 1
        if args.head == 'pointwise':
            if train:
                losses.append(F.binary_cross_entropy_with_logits(r, y))
            else:
                outs.append({'rho': r.cpu().numpy()})
            continue
        if args.no_rho:
            r = torch.zeros_like(r)
        world = gold if (train and args.head == 'circuit' and not args.no_world) else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            (gam, lp), docmask = model.forward_question(h, r, qh, qm, tm, world_variants=world)
        if not train:
            outs.append({'rho': r.cpu().numpy(), 'gam': gam[0].cpu().numpy(), 'lp': lp[0].cpu().numpy()})
            continue
        if args.head == 'setbce':
            # pi_i = P(doc i witnesses some needed requirement); same architecture, per-document objective
            lnot = log1mexp(F.logsigmoid(gam[0]).unsqueeze(-1) + lp[0, :, :-1])     # log(1 - g_k p_ki)
            logit_pi = torch.logit(-torch.expm1(lnot.sum(0)).clamp(1e-6, 1 - 1e-6))
            losses.append(F.binary_cross_entropy_with_logits(logit_pi, y))
            continue
        full = torch.ones(nd[j], device=dev)
        terms = []
        if not args.no_suff:
            terms.append(-Circuit.log_phi(gam[0], lp[0], y))                     # minimal sufficient set
        if not args.no_sel:
            ms = full.unsqueeze(0).repeat(len(gold), 1)
            ms[torch.arange(len(gold)), torch.tensor(gold)] = 0
            lphi = Circuit.log_phi(gam[0].unsqueeze(0).expand(len(gold), -1), lp[0].unsqueeze(0).expand(len(gold), -1, -1), ms)
            terms.append(-log1mexp(lphi).mean())                                 # within-pool deletion
        if world:
            lphi_w = Circuit.log_phi_pool(gam[1:], lp[1:])
            terms.append(-log1mexp(lphi_w).mean())                               # world deletion
        losses.append(sum(terms))
    if train:
        return torch.stack(losses).mean()
    return outs


@torch.no_grad()
def predict(model, recs, bs_docs=40):
    model.eval()
    res = []
    batch = []
    for r in recs:
        batch.append((r['question'], r['docs'], r['gold']))
        if sum(len(b[1]) for b in batch) >= bs_docs:
            res += run_batch(model, batch, train=False); batch = []
    if batch:
        res += run_batch(model, batch, train=False)
    model.train()
    return res


def quick_eval(outs, recs):
    hit = {2: 0, 3: 0, 4: 0, 5: 0}
    for o, r in zip(outs, recs):
        n = len(r['docs'])
        for k in hit:
            if 'gam' in o:
                S = greedy_select(o['gam'], o['lp'], np.ones(n), k)
            else:
                S = list(np.argsort(-o['rho'])[:k])
            hit[k] += set(r['gold']) <= set(S)
    return {f'CER@{k}': round(100 * v / len(recs), 2) for k, v in hit.items()}


def main():
    model = PROVE(args.enc, K=args.K, layers=args.layers, head=args.head).to(dev)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict({k: v.float() for k, v in sd.items()})
    if args.infer:
        for path in args.infer.split(','):
            recs = load(path)
            t = time.time()
            outs = predict(model, recs)
            name = os.path.basename(path).replace('.jsonl', '')
            os.makedirs(args.out, exist_ok=True)
            pickle.dump(outs, open(f'{args.out}/pred_{name}.pkl', 'wb'))
            print('infer', path, len(recs), 'sec', round(time.time() - t, 1), quick_eval(outs, recs) if 'gold' in recs[0] else '', flush=True)
        return
    model.enc.gradient_checkpointing_enable()
    train = load(f'data/proc/{args.data}_train.jsonl', args.max_train)
    devset = load(f'data/proc/{args.data}_dev.jsonl')
    devset = random.Random(0).sample(devset, min(args.dev_n, len(devset)))
    enc_params = list(model.enc.parameters()) + list(model.cls.classifier.parameters())
    eid = {id(x) for x in enc_params}
    head_params = [x for x in model.parameters() if id(x) not in eid]
    opt = torch.optim.AdamW([{'params': enc_params, 'lr': args.lr_enc}, {'params': head_params, 'lr': args.lr_head}], weight_decay=0.01)
    steps = int(math.ceil(len(train) * args.epochs / (args.qpb * args.accum)))
    sch = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    print('train', len(train), 'opt steps', steps, flush=True)
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'))
    model.train()
    order = []
    while len(order) < steps * args.qpb * args.accum:
        perm = list(range(len(train))); random.shuffle(perm); order += perm
    t0 = time.time(); run = 0.0; best = -1
    for s in range(steps):
        for a in range(args.accum):
            base = (s * args.accum + a) * args.qpb
            items = []
            for qi in order[base:base + args.qpb]:
                r = train[qi]
                keep = sample_pool(r)
                docs = [r['docs'][i] for i in keep]
                gold = [keep.index(g) for g in r['gold']]
                items.append((r['question'], docs, gold))
            loss = run_batch(model, items) / args.accum
            loss.backward()
            run += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % 50 == 0:
            print(f'step {s + 1}/{steps} loss {run / 50:.4f} {time.time() - t0:.0f}s', flush=True); run = 0.0
        if (s + 1) % max(1, steps // 4) == 0 or s + 1 == steps:
            ev = quick_eval(predict(model, devset), devset)
            print('dev', s + 1, ev, flush=True)
            score = sum(ev.values())
            if score > best:
                best = score
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f'{args.out}/model.pt')
                print('saved', flush=True)


if __name__ == '__main__':
    main()
