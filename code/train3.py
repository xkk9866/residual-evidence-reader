"""LACUNA: learning what is missing.  A residual-proof reader for proof-complete multi-hop evidence selection.

One network, one object.  For a question q, a pool C and a partial evidence state S (any subset of C) the reader
returns the *residual proof* of S:
    delta_i(S) = P(d_i belongs to the evidence still missing from S)          (per-document deficiency)
    sigma(S)   = P(S already contains a complete proof)                       (sufficiency certificate)
The state enters the reader as a single learned "in-evidence" embedding added to the tokens of the documents of S,
so every attention layer re-reads the pool *in the light of the evidence already secured* (bridge entities of S
re-weight the remaining documents).  Input = question + pool + one membership bit per document; nothing else.

Training samples states from the down-set lattice of the gold proof (order-free), mixes in learner-visited states
(DAgger: one roll-out step of the current reader) and, 1:1 as in MuSiQue-Full, world-deleted pools whose proof can
never be completed (sigma must stay 0 although every remaining gold document is in S).
Inference = residual completion: S <- S + argmax_i delta_i(S); sigma(S) certifies completion.

Two-sided residual (--objective delta): the residual of a state is its symmetric difference with the supporting set,
R(S) = (G cap C) xor S. The distribution ranges over every passage of the pool (an open passage: "it is missing",
a secured passage: "it does not belong") and the two terminals; TOP is the target only when S equals G exactly, so
TOP certifies sufficiency AND necessity. Decoding is a greedy walk on the subset lattice (apply the most probable
flip) that stops when TOP beats the total flip mass; a wrong pick is no longer final, it is released in hindsight.
"""
import argparse, json, math, os, random, time, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

p = argparse.ArgumentParser()
p.add_argument('--data', default='musique')
p.add_argument('--enc', default='BAAI/bge-reranker-v2-m3')
p.add_argument('--out', default='ckpt/tmp3')
p.add_argument('--epochs', type=float, default=2)
p.add_argument('--max_train', type=int, default=0)
p.add_argument('--accum', type=int, default=8)
p.add_argument('--lr_enc', type=float, default=1e-5)
p.add_argument('--lr_head', type=float, default=1e-4)
p.add_argument('--doclen', type=int, default=160)
p.add_argument('--states', default='lattice', help='lattice (order-free down-set) | prefix (gold-chain prefixes)')
p.add_argument('--no_roll', action='store_true', help='expert states only (no learner-visited states)')
p.add_argument('--no_world', action='store_true', help='no world-deleted (incompletable) pools')
p.add_argument('--world_fill', default='distract', help='distract: replace the deleted proof passage by a passage of another question (pool size kept, as in MuSiQue-Full) | drop')
p.add_argument('--warmup', type=int, default=0, help='warm-up steps (default 5%% of training)')
p.add_argument('--objective', default='residual', help='residual | closure (on-policy closure: half of the states are reached by greedy residual completion of the current reader; loss = decoupled residual likelihood + event likelihood) | closure_rank (ablation: on-policy states, decoupled loss only) | point (residual as a finite point process: size p(|R|) x location pi over open passages + OUT) | ce1 (ablation: one sampled residual element, softmax CE) | anscls (baseline: same reader, S = empty, binary answerable-vs-deleted classifier on TOP-vs-BOT) | delta (TWO-SIDED residual = symmetric difference S xor G: the distribution ranges over every passage, open ones to add and secured ones to remove, plus TOP/BOT; on-policy closure training; greedy walk decoding) | delta_rank (delta, decoupled loss only) | delta_lat (delta without on-policy walk states: lattice + one learner step)')
p.add_argument('--walk', type=int, default=8, help='two-sided residual: number of flips of the greedy walk at inference (2 x max |G|, so every hop can be corrected once)')
p.add_argument('--fill', type=int, default=0, help='two-sided residual: secured passages after which the ranking is completed by forced additions (default: --steps)')
p.add_argument('--avoid', default='state', help='two-sided walk: state (self-avoiding: the most probable flip into a state not visited before) | undo (a flip is not undone by the next flip) | none (plain greedy)')
p.add_argument('--ident', type=float, default=0.0, help='proof identification (weight of the term; 0 = off, 1 = on): on the reader\'s own walk, the terminal event at the exact state (TOP, or BOT when the pool is incompletable) must carry more mass than the terminal mass at the hardest other visited state. Trained on the full bounded walk; no new parameters.')
p.add_argument('--stop', default='', help='decision rule for the evidence set / verdict: bayes (first state with p_TOP > p_missing; the closure rule) | ident (the visited state with the largest terminal probability; default when --ident > 0)')
p.add_argument('--rem_init', default='zero', help='initialisation of the release readout of the two-sided reader: zero | neg (the negated relevance head: a secured passage starts with the case for release equal to its lack of relevance)')
p.add_argument('--verify', default='', help="leave-one-out necessity at the stop state: for every secured passage e, read E minus e and keep e only if the residual there names it again (pi(e | E-e) >= pi(TOP | E-e)); released passages follow the kept ones in the ranking. 'loo' enables it (one read per secured passage)")
p.add_argument('--roll', default='', help='on-policy state of the closure objective: depth (t ~ U{1..min(|G cap C|+1, 5)} greedy steps; the one-sided recipe) | walk (two-sided default: the reader runs its own walk until a terminal event wins or the bound, and a uniformly random state of that walk is the training state)')
p.add_argument('--lattice', default='', help='expert (lattice) states: level (m ~ U{0..|G cap C|}, a uniform m-subset of the gold; the one-sided recipe) | top (the terminal state G cap C with probability 1/2, otherwise a uniform proper subset; the two-sided residual has one terminal state per question against 2^|G|-1 others, and the stop decision is taken only there) | sym (two-sided default: top, and the non-terminal state is sampled from BOTH sides of G cap C, a uniform subset of the gold plus j ~ U{0..|G cap C|} uniformly drawn distractors, so that the release side of the residual is supervised as broadly as the missing side)')
p.add_argument('--loc', default='set', help='point objective, location loss: set (every residual element vs the non-residual) | ce1 (ablation: one sampled residual element, plain softmax CE)')
p.add_argument('--layout', default='query', help='query (evidence joins the question segment) | mark (in place)')
p.add_argument('--term', default='head', help='head: TOP/BOT logits from a linear map of <s> | text: TOP/BOT are two verbalized hypothesis candidates appended to the pool and scored by the same passage scorer as every passage')
p.add_argument('--save_all', action='store_true', help='also keep the checkpoint of every quarter epoch (model_q1..q4.pt)')
p.add_argument('--steps', type=int, default=5, help='completion steps at inference')
p.add_argument('--seed', type=int, default=42)
p.add_argument('--infer', default='')
p.add_argument('--ckpt', default='')
p.add_argument('--init', default='', help='stage-1 reader (S = empty, bottom of the lattice) to start from')
p.add_argument('--dev_n', type=int, default=400)
p.add_argument('--max_infer', type=int, default=0)
p.add_argument('--openloop', default='', help='open-corpus closed loop on DS (HippoRAG 1000-question corpus)')
p.add_argument('--loop_mode', default='adaptive', help='adaptive: re-retrieve only when the reader halts on OUT | fixed: always re-retrieve')
p.add_argument('--rounds', type=int, default=2)
p.add_argument('--pool_n', type=int, default=20)
args = p.parse_args()
OBJECTIVE = args.objective
# two-sided residual: the label space is the symmetric difference S xor (G cap C); everything else (states, losses,
# event decisions) is the closure recipe, so the flip variants map onto the corresponding one-sided objectives
FLIP = args.objective.startswith('delta')
if FLIP:
    args.objective = {'delta': 'closure', 'delta_rank': 'closure_rank', 'delta_lat': 'residual_ev'}[args.objective]
if not args.fill:
    args.fill = args.steps
if not args.roll:
    args.roll = 'walk' if FLIP else 'depth'
if not args.lattice:
    args.lattice = 'sym' if FLIP else 'level'
if not args.stop:
    args.stop = 'ident' if args.ident > 0 else 'bayes'

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device(os.environ.get('DEV', 'cuda'))     # DEV=cpu only for smoke tests
tok = AutoTokenizer.from_pretrained(args.enc)
KMAX = 5
BOS, SEP = tok.bos_token_id, tok.sep_token_id
# the two terminal outcomes stated as candidate passages (the definitions of TOP and BOT in plain words)
HYP = [tok('Nothing is missing: the passages already given answer the question.', add_special_tokens=False)['input_ids'],
       tok('Something is missing: a passage needed to answer the question is not among the candidates.', add_special_tokens=False)['input_ids']]


class ResidualReader(nn.Module):
    """Reads (question, pool, evidence state S) jointly and returns the residual-proof distribution
        pi(. | q, C, S) over  {d_i not in S}  U  {TOP: nothing is missing}  U  {BOT: what is missing is not in C}."""
    def __init__(self, enc_name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_name, attn_implementation='sdpa', add_pooling_layer=False)
        H = self.enc.config.hidden_size
        self.inS = nn.Parameter(0.02 * torch.randn(H))   # the evidence-state embedding
        self.doc = nn.Linear(H, 1)                        # "d_i is (part of) what is missing"
        self.term = nn.Linear(H, 2)                       # TOP / BOT read at the question's <s>  (point: BOT = OUT)
        self.card = nn.Linear(H, KMAX + 1)                # point process: size of the residual |R| in {0..KMAX}
        # two-sided residual: the same pooled state of a SECURED passage is read by a second linear map,
        # "e_i does not belong to the proof" (removing it is the correction). Zero-initialised: at the start of
        # stage 2 no secured passage is suspected, the flip side of the distribution is learned from the states.
        self.rem = nn.Linear(H, 1)
        nn.init.zeros_(self.rem.weight); nn.init.zeros_(self.rem.bias)

    def forward(self, ids, spans, member):
        """returns logits over [docs..., TOP, BOT]. One-sided residual: docs in S are masked to -inf.
        Two-sided residual (FLIP): every doc has a logit, 'add' for an open passage, 'remove' for a secured one."""
        emb = self.enc.get_input_embeddings()(ids)
        m = torch.zeros(ids.shape[1], device=ids.device, dtype=emb.dtype)
        for (s, e), b in zip(spans, member):
            if b:
                m[s:e + 1] = 1        # the document and its closing separator
        emb = emb + m[None, :, None] * self.inS.to(emb.dtype)
        h = self.enc(inputs_embeds=emb, attention_mask=torch.ones_like(ids)).last_hidden_state[0]
        N, L = len(spans), max(e - s for s, e in spans)
        idx = torch.zeros(N, L, dtype=torch.long, device=h.device)
        msk = torch.zeros(N, L, dtype=h.dtype, device=h.device)
        for i, (s, e) in enumerate(spans):
            idx[i, :e - s] = torch.arange(s, e, device=h.device)
            msk[i, :e - s] = 1
        pooled = (h[idx] * msk.unsqueeze(-1)).sum(1) / msk.sum(1, keepdim=True)
        dl = self.doc(pooled).squeeze(-1).float()
        mem = torch.tensor(list(member) + [False] * (N - len(member)), device=h.device)
        if FLIP:
            dl = torch.where(mem, self.rem(pooled).squeeze(-1).float(), dl)
        else:
            dl = dl.masked_fill(mem, float('-inf'))
        if args.term == 'text':
            # [passages..., TOP hypothesis, BOT hypothesis]: one scorer ranks evidence and outcomes alike
            return dl
        if args.objective == 'point':
            return torch.cat([dl, self.term(h[0]).float()]), self.card(h[0]).float()
        return torch.cat([dl, self.term(h[0]).float()])


def load(path, n=0):
    out = []
    for l in open(path):
        out.append(json.loads(l))
        if n and len(out) >= n:
            break
    return out


_cache = {}
FILL = []          # questions whose passages fill world-deleted pools
ID_STAT = [0.0, 0]  # running sum / count of the identification loss (logged with the main loss)


def doc_ids(d):
    key = (d['title'], d['text'][:64])
    if key not in _cache:
        _cache[key] = tok(f"{d['title']}\n{d['text']}", add_special_tokens=False, truncation=True,
                          max_length=args.doclen)['input_ids']
    return _cache[key]


def build(q, docs, state=()):
    """layout 'query': the secured evidence (positions `state`, in order) joins the question segment,
    <s> q </s> e_1 </s> ... e_m </s> </s> d_1 </s> ... ; layout 'mark': evidence stays in place (marked only).
    returns ids and the span of every document of `docs` (in the order of `docs`)."""
    if args.layout == 'mark':
        state = ()
    ids = [BOS] + tok(q, add_special_tokens=False, truncation=True, max_length=64)['input_ids'] + [SEP]
    spans = [None] * len(docs)
    for j in state:
        t = doc_ids(docs[j])
        spans[j] = (len(ids), len(ids) + len(t))
        ids += t + [SEP]
    if state:
        ids += [SEP]
    for j, d in enumerate(docs):
        if spans[j] is None:
            t = doc_ids(d)
            spans[j] = (len(ids), len(ids) + len(t))
            ids += t + [SEP]
    if args.term == 'text':
        for t in HYP:
            spans.append((len(ids), len(ids) + len(t)))
            ids += t + [SEP]
    return torch.tensor([ids], device=dev), spans


def run(model, x, spans, member):
    with torch.autocast(dev.type, dtype=torch.bfloat16):
        return model(x, spans, member)


def sample_state(r, present, pool=()):
    """expert state. level: a subset of the gold proof present in the pool. top/sym: the terminal state with probability
    1/2; otherwise a proper subset of the gold (top), or a subset of the gold plus up to |G cap C| distractors (sym):
    the two-sided residual names what is missing AND what does not belong, so its states must show both."""
    if args.states == 'prefix':
        chain = [g for g in (r.get('gold_order') or r['gold']) if g in present]
        return set(chain[:random.randint(0, len(chain))])
    if args.lattice in ('top', 'sym') and present and random.random() < 0.5:
        return set(present)                                   # the terminal state: the only state where TOP (or BOT) is the target
    m = random.randint(0, len(present) - (1 if args.lattice == 'top' and present else 0))
    S = set(random.sample(sorted(present), m))
    if args.lattice == 'sym':
        junk = [i for i in pool if i not in present]
        j = min(random.randint(0, len(present)), len(junk))
        S |= set(random.sample(junk, j))
        if S == set(present):                                 # (m = |G cap C|, j = 0): the terminal state is already drawn above
            S = set(random.sample(sorted(present), len(present) - 1)) if present else S
    return S


def pick_flip(lg, Sl, visited, last):
    """the flip of the greedy walk at a state: the most probable passage whose flip is allowed by --avoid.
    Sl: secured positions; visited: set of frozensets of positions already visited; last: position flipped last.
    Under an exact residual no constraint ever binds (the oracle walk never revisits a state); the constraint only
    resolves the cycles of an imperfect reader (secure x, release x, secure x, ...)."""
    n = lg.shape[0] if torch.is_tensor(lg) else len(lg)
    scores = (lg if torch.is_tensor(lg) else torch.as_tensor(lg)).float().cpu()
    cur = frozenset(Sl)
    for j in torch.argsort(scores, descending=True).tolist():
        if args.avoid == 'undo' and j == last:
            continue
        if args.avoid == 'state' and (cur ^ {j}) in visited:
            continue
        return j
    return int(scores.argmax())


def learner_step(S, Sl, order, lg, visited=None, last=None):
    """one greedy step of the current reader on state S (original indices) / Sl (positions in `order`):
    one-sided: add the most probable open passage; two-sided: apply the most probable allowed flip (add or remove)"""
    if FLIP:
        j = pick_flip(lg[:len(order)], Sl, visited if visited is not None else set(), last)
        if order[j] in S:
            return S - {order[j]}, [x for x in Sl if x != j], j
        return S | {order[j]}, Sl + [j], j
    j = int(lg[:len(order)].argmax())
    return S | {order[j]}, Sl + [j], j


def loss_for(model, r):
    n = len(r['docs'])
    order = list(range(n)); random.shuffle(order)
    gold = set(r['gold'])
    world = not args.no_world and random.random() < 0.5
    sub = None
    if world:
        drop = random.choice(r['gold'])
        if args.world_fill == 'distract' and FILL:
            # keep the pool size: the deleted proof passage is replaced by a passage of another question
            o = random.choice(FILL)
            sub = random.choice(o['docs'])
            order = [(-1 if i == drop else i) for i in order]
        else:
            order = [i for i in order if i != drop]
    present = gold & set(order)
    docs = [(sub if i == -1 else r['docs'][i]) for i in order]
    if args.objective == 'anscls':
        x, spans = build(r['question'], docs, [])
        lg = run(model, x, spans, [False] * len(order))
        n = len(order)
        return F.binary_cross_entropy_with_logits(lg[n] - lg[n + 1], torch.tensor(0.0 if world else 1.0, device=dev))
    S = sample_state(r, present, order)
    Sl = [j for j, i in enumerate(order) if i in S]            # random (order-free) arrangement of the state
    ident_loss = None
    if args.objective in ('closure', 'closure_rank') and random.random() < 0.5:
        # on-policy closure: the state is one the decoder itself reaches. Greedy residual completion from the empty
        # state for t steps, t uniform over the depths decoding can stop at (|G cap C| + 1, so the state just past
        # the proof is visited as often as the ones before it). These are exactly the states on which the stop
        # event is decided, including the false-complete ones that follow a wrong pick.
        S, Sl, visited, last = set(), [], {frozenset()}, None
        with torch.no_grad():
            if args.roll == 'walk':
                # the reader's own walk, stopped where a terminal event first wins (its verdict state) or at the
                # bound; the training state is a uniformly random state of this trajectory (t >= 1). Every state
                # on which the deployed decoder takes a decision is therefore a training state, including the
                # exact state reached after a correction, which lies deeper than |G cap C| + 1.
                # With proof identification the walk always runs to the bound: its states are the candidates among
                # which the deployed decoder identifies the proof, so they are the candidates the term compares.
                traj, term = [], []
                n0 = len(order)
                for t in range(args.walk):
                    x, spans = build(r['question'], docs, Sl)
                    lg = run(model, x, spans, [i in S for i in order])
                    if t >= 1:
                        lp = lg.log_softmax(0)
                        pt, pb = math.exp(float(lp[n0])), math.exp(float(lp[n0 + 1]))
                        term.append((float(lp[n0]), float(lp[n0 + 1])))
                        if args.ident <= 0 and max(pt, pb) > 1 - pt - pb:
                            break
                    S, Sl, last = learner_step(S, Sl, order, lg, visited, last)
                    visited.add(frozenset(Sl)); traj.append((S, Sl))
                if args.ident > 0 and len(traj) >= 2:
                    # the last state's terminal mass has not been read yet
                    x, spans = build(r['question'], docs, traj[-1][1])
                    lp = run(model, x, spans, [i in traj[-1][0] for i in order]).log_softmax(0)
                    term.append((float(lp[n0]), float(lp[n0 + 1])))
                    pos = [k for k, (St, _) in enumerate(traj) if St == present]
                    neg = [k for k, (St, _) in enumerate(traj) if St != present]
                    if pos and neg:
                        kp = pos[0]
                        kn = max(neg, key=lambda k: np.logaddexp(*term[k]))
                        a = term[kp][1 if world else 0]                   # log pi(e* | S*), e* = BOT if the pool is incompletable else TOP
                        b = float(np.logaddexp(*term[kn]))                # log terminal mass at the hardest wrong state
                        ident_loss = (traj[kp][1], traj[kn][1], float(1 / (1 + math.exp(-(b - a)))))   # sigma(b - a): the weight of both gradients
                S, Sl = random.choice(traj)
            else:
                for _ in range(random.randint(1, min(len(present) + 1, KMAX, len(order)))):
                    x, spans = build(r['question'], docs, Sl)
                    lg = run(model, x, spans, [i in S for i in order])
                    S, Sl, last = learner_step(S, Sl, order, lg, visited, last)
                    visited.add(frozenset(Sl))
    elif not args.no_roll and random.random() < 0.5 and len(S) < len(order):
        # learner-visited state: one roll-out step of the current reader from the expert state
        with torch.no_grad():
            x, spans = build(r['question'], docs, Sl)
            lg = run(model, x, spans, [i in S for i in order])
            if isinstance(lg, tuple):
                lg = lg[0]
            S, Sl, _ = learner_step(S, Sl, order, lg, {frozenset(Sl)}, None)
    if ident_loss is not None:
        # proof identification: -log sigma(a - b) with a = log pi(e* | S*) and b = log terminal mass at the hardest
        # wrong state. Its gradient is -sigma(b-a) da + sigma(b-a) db; the weight is taken from the no-grad walk, so
        # the two reads are back-propagated one at a time (no activations of two reads in memory at once).
        Slp, Sln, w = ident_loss
        Sp, Sn, n0 = {order[j] for j in Slp}, {order[j] for j in Sln}, len(order)
        x, spans = build(r['question'], docs, Slp)
        lp = run(model, x, spans, [i in Sp for i in order]).log_softmax(0)
        (-w * lp[n0 + (1 if world else 0)] * args.ident / args.accum).backward()
        x, spans = build(r['question'], docs, Sln)
        lp = run(model, x, spans, [i in Sn for i in order]).log_softmax(0)
        (w * torch.logaddexp(lp[n0], lp[n0 + 1]) * args.ident / args.accum).backward()
        ID_STAT[0] += -math.log(max(1 - w, 1e-12)); ID_STAT[1] += 1          # the identification loss -log sigma(a - b) = -log(1 - w)
    x, spans = build(r['question'], docs, Sl)
    lg = run(model, x, spans, [i in S for i in order])
    n = len(order)
    if args.objective == 'point':
        # the residual R(S) is a finite point process: its size N = |G \ S| (the deleted passage counts as missing)
        # and its location: the missing passages of the pool, or OUT when what is missing lies outside the pool
        lg, cl = lg
        inres = [j for j, i in enumerate(order) if i in present and i not in S]
        size = len(inres) + (1 if world else 0)
        loss = F.cross_entropy(cl[None], torch.tensor([min(size, KMAX)], device=dev))
        lg = lg[:n + 2].clone(); lg[n] = float('-inf')            # the location space: open passages + OUT
        if size and args.loc == 'ce1':
            loss = loss + F.cross_entropy(lg[None], torch.tensor([random.choice(inres or [n + 1])], device=dev))
        elif size:
            pos = torch.zeros(n + 2, dtype=torch.bool, device=dev)
            pos[inres or [n + 1]] = True
            neg = ~pos & torch.isfinite(lg)
            lneg = torch.logsumexp(lg[neg], 0)
            loss = loss - (lg[pos] - torch.logaddexp(lg[pos], lneg)).mean()
        return loss
    if args.objective == 'ce1':
        # ablation: same states/terminals, single sampled residual element, plain softmax cross-entropy
        R = [j for j, i in enumerate(order) if i in present and i not in S] or [n + 1 if world else n]
        return F.cross_entropy(lg[None], torch.tensor([random.choice(R)], device=dev))
    if args.objective == 'rbce':
        # ablation: same state-conditioned input, per-passage BCE on "is missing" (no terminals, no competition)
        y = torch.tensor([float(i in present and i not in S) for i in order], device=dev)
        free = torch.isfinite(lg[:n])
        return F.binary_cross_entropy_with_logits(lg[:n][free], y[free])
    # residual R = missing gold in the pool, or the terminal (TOP if the proof is complete, BOT if it cannot be).
    # two-sided residual: R = S xor (G cap C) = missing gold (to add) + secured non-gold (to remove); a terminal only
    # once no correction is left in the pool
    pos = torch.zeros(n + 2, dtype=torch.bool, device=dev)
    R = [j for j, i in enumerate(order) if (i in present and i not in S) or (FLIP and i in S and i not in gold)]
    pos[R or [n + 1 if world else n]] = True
    neg = ~pos & torch.isfinite(lg)
    # every element of the residual must outrank everything that is not missing (terminals included)
    lneg = torch.logsumexp(lg[neg], 0)
    rank = -(lg[pos] - torch.logaddexp(lg[pos], lneg)).mean()
    if args.objective not in ('residual_ev', 'closure'):
        return rank
    # event likelihood: the probability that the state's residual EVENT occurs (some missing passage / TOP / BOT),
    # i.e. -log sum_{y in R(S)} pi(y). It calibrates the event masses that decoding compares; the decoupled term
    # orders the elements within the event. Both are factors of the same distribution.
    fin = torch.isfinite(lg)
    lp = lg[fin].log_softmax(0)
    return rank - torch.logsumexp(lp[pos[fin]], 0)


@torch.no_grad()
def complete(model, r, steps):
    """residual completion. returns full ranking (completion order, then last-state residual), and per state
    log pi(TOP|S_t), log pi(BOT|S_t), log max_i pi(d_i|S_t)"""
    n = len(r['docs'])
    if args.objective == 'anscls':
        steps = 0
    S, top, bot, dmax, card = [], [], [], [], []
    halt = verdict = None
    for t in range(min(steps, n) + 1):
        x, spans = build(r['question'], r['docs'], S)
        out = run(model, x, spans, [i in S for i in range(n)])
        if isinstance(out, tuple):
            # point process: halt when the most probable residual size is 0 (complete) or the most probable
            # location is OUT (what is missing is not in the pool); verdict and returned set come from that state
            lg, cl = out
            lg = lg.clone(); lg[n] = float('-inf')
            lp, lc = lg.log_softmax(0), cl.log_softmax(0)
            card.append(lc.cpu().tolist()); top.append(float(lc[0])); bot.append(float(lp[n + 1]))
            if halt is None and int(lc.argmax()) == 0:
                halt, verdict = t, 1
            elif halt is None and int(lp.argmax()) == n + 1:
                halt, verdict = t, 0
        else:
            lp = out.log_softmax(0)
            top.append(float(lp[n])); bot.append(float(lp[n + 1]))
        dmax.append(float(lp[:n].max()))
        if t == min(steps, n):
            break
        S.append(int(lp[:n].argmax()))
    rest = [int(i) for i in np.argsort(-lp[:n].cpu().numpy()) if i not in S]
    o = {'order': S + rest, 'top': top, 'bot': bot, 'dmax': dmax}
    if card:
        o.update(card=card, halt=len(top) - 1 if halt is None else halt, verdict=0 if verdict is None else verdict)
    elif args.objective in ('residual', 'ce1', 'residual_ev', 'closure', 'closure_rank'):
        # Algorithm 1: the evidence set closes when TOP beats every open passage; verdict = TOP is the overall argmax
        T = len(top) - 1
        o['halt_am'] = next((t for t in range(1, T + 1) if top[t] > dmax[t]), T)
        o['verdict_am'] = int(any(top[t] >= max(bot[t], dmax[t]) for t in range(1, T + 1)))
        # Bayes decisions on the three events of a state: TOP, BOT, and "a passage of the pool is missing"
        # (its probability is the remaining mass 1 - pi(TOP) - pi(BOT)).
        pt, pb = np.exp(top), np.exp(bot)
        pm = np.clip(1 - pt - pb, 0, 1)
        # evidence set (answerable world): close at the first state where "complete" is more probable than "a passage is missing"
        o['halt'] = next((t for t in range(1, T + 1) if pt[t] > pm[t]), T)
        # three-way verdict: the first state at which a terminal event is the most probable event decides it;
        # 1 = complete (TOP), 0 = missing evidence lies outside the pool (BOT), -1 = undetermined within T steps
        ts = next((t for t in range(1, T + 1) if max(pt[t], pb[t]) > pm[t]), None)
        o['halt_any'] = T if ts is None else ts
        o['verdict3'] = -1 if ts is None else (1 if pt[ts] >= pb[ts] else 0)
        o['verdict'] = int(o['verdict3'] == 1)
        # proof identification: the visited state with the largest terminal probability. In the answerable world
        # the evidence set is the state with the largest p(TOP); the verdict is the largest terminal event overall.
        o['halt_bayes'], o['verdict3_bayes'] = o['halt'], o['verdict3']
        t_top = 1 + int(np.argmax(top[1:]))
        t_any = 1 + int(np.argmax(np.maximum(top[1:], bot[1:])))
        o['halt_id'], o['halt_any_id'], o['verdict3_id'] = t_top, t_any, int(top[t_any] >= bot[t_any])
        if args.stop == 'ident':
            o['halt'], o['halt_any'], o['verdict3'] = t_top, t_any, o['verdict3_id']
            o['verdict'] = int(o['verdict3'] == 1)
    return o


@torch.no_grad()
def complete_flip(model, r, walk, fill):
    """Two-sided residual decoding: a greedy walk on the subset lattice. At every state the most probable flip is
    applied (an open passage is secured, or a secured passage is released). The evidence set is the state at which
    TOP first beats the total flip mass ("something is still wrong"); the three-way verdict is the first state at
    which a terminal beats it. The walk is recorded for `walk` flips so that alternative stop offsets can be
    analysed; the deployed cost is reads up to the stop plus the forced additions that complete the budget ranking.
    Output fields are compatible with the one-sided reader: order[:halt] is the evidence set."""
    n = len(r['docs'])
    S, top, bot, pin, states, moves, lps = [], [], [], [], [], [], []
    visited, last = {frozenset()}, None
    for t in range(min(walk, 2 * n) + 1):
        x, spans = build(r['question'], r['docs'], S)
        lp = run(model, x, spans, [i in S for i in range(n)]).log_softmax(0)
        pt, pb = math.exp(float(lp[n])), math.exp(float(lp[n + 1]))
        top.append(float(lp[n])); bot.append(float(lp[n + 1])); pin.append(math.log(max(1 - pt - pb, 1e-12)))
        states.append(list(S)); lps.append(lp[:n].float().cpu().numpy())
        if t == min(walk, 2 * n):
            break
        j = pick_flip(lp[:n], S, visited, last); last = j
        if j in S:
            S = [i for i in S if i != j]; moves.append(-j - 1)
        else:
            S = S + [j]; moves.append(j)
        visited.add(frozenset(S))
    T = len(top) - 1
    pt, pb, pm = np.exp(top), np.exp(bot), np.exp(pin)
    # Bayes rule (closure): the walk stops at the first state where TOP beats the total correction mass; if no state
    # is certified within the bound, the evidence set is the visited state with the largest p_TOP
    stop_t = next((t for t in range(1, T + 1) if pt[t] > pm[t]), None)
    halt_t = stop_t if stop_t is not None else 1 + int(np.argmax(pt[1:]))
    ts = next((t for t in range(1, T + 1) if max(pt[t], pb[t]) > pm[t]), None)
    any_t = halt_t if ts is None else ts
    verdict3 = -1 if ts is None else (1 if pt[ts] >= pb[ts] else 0)
    bayes = dict(halt_t=halt_t, any_t=any_t, verdict3=verdict3, stopped=stop_t is not None)
    reads_walk = (halt_t if stop_t is not None else T) + 1       # reads up to the Bayes stop (or the whole walk)
    if args.stop == 'ident':
        # proof identification: the whole bounded walk is the candidate set; the evidence set is the visited state
        # with the largest p(TOP), the verdict the largest terminal event (TOP or BOT) over the walk
        halt_t = 1 + int(np.argmax(pt[1:]))
        any_t = 1 + int(np.argmax(np.maximum(pt[1:], pb[1:])))
        verdict3 = int(pt[any_t] >= pb[any_t])
        stop_t, reads_walk = T, T + 1               # the whole walk is read
    # the evidence set: secured passages of the stop state, the most necessary first (lowest flip probability)
    ev = sorted(states[halt_t], key=lambda i: float(lps[halt_t][i]))
    set_any = sorted(states[any_t], key=lambda i: float(lps[any_t][i]))
    # ranking for budgets: forced additions from the stop state, re-reading after each, then the last read's order
    Sf, cur, extra = list(ev), lps[halt_t], 0
    while len(Sf) < min(fill, n):
        Sf.append(max((i for i in range(n) if i not in Sf), key=lambda i: float(cur[i])))
        x, spans = build(r['question'], r['docs'], Sf)
        cur = run(model, x, spans, [i in Sf for i in range(n)]).log_softmax(0)[:n].float().cpu().numpy(); extra += 1
    rest = sorted((i for i in range(n) if i not in Sf), key=lambda i: -float(cur[i]))
    return {'order': Sf + rest, 'halt': len(ev), 'top': top, 'bot': bot, 'pin': pin, 'dmax': [float(l.max()) for l in lps],
            'states': states, 'moves': moves, 'halt_t': halt_t, 'halt_any_t': any_t, 'set_any': set_any, 'halt_any': len(set_any),
            'verdict3': verdict3, 'verdict': int(verdict3 == 1), 'bayes': bayes, 'rule': args.stop,
            'stopped': stop_t is not None, 'reads': reads_walk + extra, 'removals': sum(m < 0 for m in moves[:halt_t])}


_complete_add = complete


@torch.no_grad()
def verify_loo(model, r, o):
    """leave-one-out necessity of the stop state E = order[:halt]: e is necessary iff, at the state E minus e, the
    residual still names e (its add log-probability >= log pi(TOP)). The evidence set becomes the necessary members
    (at least one); released members follow them in the ranking, then everything else. Costs |E| reads."""
    n = len(r['docs'])
    E = list(o['order'][:o['halt']])
    if len(E) <= 1:
        o.update(loo=[True] * len(E), loo_gap=[0.0] * len(E), released_loo=0, reads_loo=0)
        return o
    nec, gap = [], []
    for e in E:
        S = [x for x in E if x != e]
        x, spans = build(r['question'], r['docs'], S)
        lp = run(model, x, spans, [i in S for i in range(n)]).log_softmax(0)
        g = float(lp[e] - lp[n]); gap.append(g); nec.append(g >= 0)
    if not any(nec):
        nec[int(np.argmax(gap))] = True
    keep = sorted([e for e, k in zip(E, nec) if k], key=lambda e: -gap[E.index(e)])
    drop = sorted([e for e, k in zip(E, nec) if not k], key=lambda e: -gap[E.index(e)])
    rest = [i for i in o['order'] if i not in E]
    o.update(order=keep + drop + rest, halt=len(keep), loo=nec, loo_gap=gap, released_loo=len(drop),
             reads_loo=len(E), reads=o.get('reads', len(o['top'])) + len(E))
    return o


def complete(model, r, steps):
    o = complete_flip(model, r, args.walk, args.fill) if FLIP else _complete_add(model, r, steps)
    if args.verify == 'loo' and 'halt' in o:
        o = verify_loo(model, r, o)
    return o


def predict(model, recs, steps):
    model.eval()
    out = [complete(model, r, steps) for r in recs]
    model.train()
    return out


def quick_eval(outs, recs):
    hit = {2: 0, 3: 0, 4: 0, 5: 0}
    for o, r in zip(outs, recs):
        for k in hit:
            hit[k] += set(r['gold']) <= set(o['order'][:k])
    res = {f'CER@{k}': round(100 * v / len(recs), 2) for k, v in hit.items()}
    if 'halt' in outs[0]:      # the self-terminated set: complete, and exactly the gold set
        res['halt_CER'] = round(100 * np.mean([set(r['gold']) <= set(o['order'][:o['halt']]) for o, r in zip(outs, recs)]), 2)
        res['halt_EM'] = round(100 * np.mean([set(r['gold']) == set(o['order'][:o['halt']]) for o, r in zip(outs, recs)]), 2)
    if 'removals' in outs[0]:  # two-sided residual: reads to the stop (+ forced additions) and corrections per question
        res['reads'] = round(float(np.mean([o['reads'] for o in outs])), 2)
        res['rm'] = round(float(np.mean([o['removals'] for o in outs])), 3)
    return res


def openloop(model, ds):
    """the residual drives retrieval: when the reader halts because what is missing lies OUTside the pool, the
    secured evidence joins the question as the next dense query (evidence-as-query, the reader's own principle)"""
    import sys
    sys.path.insert(0, 'code')
    from open_pipeline import BGE_Enc, make_rec, dstr, QP
    qs = [json.loads(l) for l in open(f'data/proc/open_{ds}.jsonl')]
    corpus = [json.loads(l) for l in open(f'data/proc/open_{ds}_corpus.jsonl')]
    ce = torch.tensor(np.load(f'data/proc/open_{ds}_cemb.npy').astype(np.float32), device=dev)
    qe = torch.tensor(np.load(f'data/proc/open_{ds}_qemb.npy').astype(np.float32), device=dev)
    bge = BGE_Enc()
    model.eval()
    recs, outs, t0 = [], [], time.time()
    for qi, q in enumerate(qs):
        ids = [int(i) for i in torch.topk(ce @ qe[qi], args.pool_n).indices.tolist()]
        calls, reads = 1, 0
        for rnd in range(args.rounds + 1):
            rec = make_rec(q, corpus, ids)
            o = complete(model, rec, args.steps)
            reads += o.get('reads', len(o['top']))
            last = rnd == args.rounds or (args.loop_mode == 'adaptive' and o['verdict'] == 1)
            if last:
                break
            S = [rec['cid'][j] for j in o['order'][:max(o['halt'], 1 if args.loop_mode == 'fixed' else 0)]]
            qv = torch.tensor(bge([QP + q['question'] + ' ' + ' '.join(dstr(corpus[i]) for i in S)])[0], device=dev)
            new = [int(i) for i in torch.topk(ce @ qv, args.pool_n + len(S)).indices.tolist() if int(i) not in S]
            ids = S + new[:args.pool_n - len(S)]
            calls += 1
        o.update(calls=calls, rounds=rnd, reads=reads)
        recs.append(rec); outs.append(o)
    tag = f'openloop_{ds}_{args.loop_mode}{args.rounds}'
    with open(f'data/proc/{tag}.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r) + chr(10))
    pickle.dump(outs, open(f'{args.out}/pred_{tag}.pkl', 'wb'))
    print(tag, 'sec', round(time.time() - t0, 1), 'pool-complete', round(100 * np.mean([r['gold_missing'] == 0 for r in recs]), 2),
          'calls', round(float(np.mean([o['calls'] for o in outs])), 3), 'reads', round(float(np.mean([o['reads'] for o in outs])), 2), flush=True)


def main():
    model = ResidualReader(args.enc).to(dev)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location='cpu')
        print(model.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False))
    if args.init:
        # stage 1 = the bottom of the lattice (S = empty): a static joint reader trained with per-document targets
        sd = torch.load(args.init, map_location='cpu')
        model.enc.load_state_dict({k[4:]: v.float() for k, v in sd.items() if k.startswith('enc.')})
        model.doc.load_state_dict({'weight': sd['doc_cls.weight'].float(), 'bias': sd['doc_cls.bias'].float()})
        if FLIP and args.rem_init == 'neg':
            model.rem.load_state_dict({'weight': -sd['doc_cls.weight'].float(), 'bias': -sd['doc_cls.bias'].float()})
        print('init from', args.init, 'rem_init', args.rem_init, flush=True)
    os.makedirs(args.out, exist_ok=True)
    if args.openloop:
        return openloop(model, args.openloop)
    if args.infer:
        for path in args.infer.split(','):
            recs = load(path, args.max_infer)
            t = time.time()
            outs = predict(model, recs, args.steps)
            name = os.path.basename(path).replace('.jsonl', '')
            pickle.dump(outs, open(f'{args.out}/pred_{name}.pkl', 'wb'))
            print('infer', path, len(recs), 'sec', round(time.time() - t, 1),
                  quick_eval(outs, recs) if recs[0].get('gold') else '', flush=True)
        return
    model.enc.gradient_checkpointing_enable()
    train = load(f'data/proc/{args.data}_train.jsonl', args.max_train)
    FILL[:] = train
    devset = random.Random(0).sample(load(f'data/proc/{args.data}_dev.jsonl'), args.dev_n)
    enc_p = list(model.enc.parameters())
    eid = {id(x) for x in enc_p}
    head_p = [x for x in model.parameters() if id(x) not in eid]
    opt = torch.optim.AdamW([{'params': enc_p, 'lr': args.lr_enc}, {'params': head_p, 'lr': args.lr_head}], weight_decay=0.01, foreach=False)
    steps = int(math.ceil(len(train) * args.epochs / args.accum))
    sch = get_linear_schedule_with_warmup(opt, args.warmup or int(0.05 * steps), steps)
    json.dump(dict(vars(args), objective=OBJECTIVE), open(f'{args.out}/args.json', 'w'))
    print('train', len(train), 'opt steps', steps, flush=True)
    order = []
    while len(order) < steps * args.accum:
        perm = list(range(len(train))); random.shuffle(perm); order += perm
    model.train()
    t0, run_l, best = time.time(), 0.0, -1
    for s in range(steps):
        for a in range(args.accum):
            loss = loss_for(model, train[order[s * args.accum + a]]) / args.accum
            loss.backward()
            run_l += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        if (s + 1) % 100 == 0:
            idl = f' ident {ID_STAT[0] / ID_STAT[1]:.4f} (n={ID_STAT[1]})' if ID_STAT[1] else ''
            print(f'step {s + 1}/{steps} loss {run_l / 100:.4f}{idl} {time.time() - t0:.0f}s', flush=True); run_l = 0.0; ID_STAT[0] = 0.0; ID_STAT[1] = 0
        if (s + 1) % max(1, steps // 4) == 0 or s + 1 == steps:
            ev = quick_eval(predict(model, devset, 5), devset)
            print('dev', s + 1, ev, flush=True)
            if args.save_all:
                q = min(4, round(4 * (s + 1) / steps))
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f'{args.out}/model_q{q}.pt')
            if sum(v for k, v in ev.items() if k.startswith('CER') or k == 'halt_CER') > best:
                best = sum(v for k, v in ev.items() if k.startswith('CER') or k == 'halt_CER')
                torch.save({k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}, f'{args.out}/model.pt')
                print('saved', flush=True)


if __name__ == '__main__':
    main()
