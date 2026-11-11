"""Budgeted evidence-set selection.

circuit objective F(S) = sum_k log(1 - g_k prod_{i in S}(1-a_ki))  (monotone submodular, Prop. 3)
"""
import itertools
import numpy as np


def _logsig(x):
    return -np.logaddexp(0, -x)


def _prep(gam, lp):
    gam = np.asarray(gam, dtype=np.float64)
    p = np.exp(np.asarray(lp, dtype=np.float64))          # (K, N+1), last column = null witness
    g = np.exp(_logsig(gam))
    c = np.exp(_logsig(-gam))                              # 1-g, computed stably
    return g, c, p[:, :-1], p[:, -1]


def circuit_logphi(gam, lp, S):
    g, c, p, _ = _prep(gam, lp)
    m = p[:, list(S)].sum(1) if len(S) else np.zeros(len(g))
    return np.log(c + g * m).sum()


def pool_logphi(gam, lp):
    g, c, p, p0 = _prep(gam, lp)
    return np.log(1 - g * p0 + 1e-300).sum()


def doc_marginal(gam, lp):
    """P(doc i is the witness of some needed requirement) = 1 - prod_k (1 - g_k p_ki)"""
    g, c, p, _ = _prep(gam, lp)
    return 1 - np.prod(1 - g[:, None] * p, 0)


def greedy_select(gam, lp, cost, B, enum=0, return_obj=False):
    """Cost-benefit greedy on log-sufficiency F(S)=sum_k log((1-g_k)+g_k sum_{i in S} p_ki)
    (monotone submodular: concave of modular); optional partial enumeration of seeds up to size
    `enum` (enum=3 -> (1-1/e) guarantee under a knapsack, Sviridenko 2004)."""
    g, c, p, _ = _prep(gam, lp)
    cost = np.asarray(cost, dtype=np.float64)
    K, N = p.shape
    seeds = [()]
    for e in range(1, enum + 1):
        seeds += [s for s in itertools.combinations(range(N), e) if cost[list(s)].sum() <= B]
    M = len(seeds)
    chosen = np.zeros((M, N), dtype=bool)
    for m_, s in enumerate(seeds):
        chosen[m_, list(s)] = True
    mass = chosen.astype(np.float64) @ p.T                   # (M,K)
    spent = chosen.astype(np.float64) @ cost
    order = [list(s) for s in seeds]
    active = np.ones(M, dtype=bool)
    unit = np.allclose(cost, cost[0])
    tie = 1e-9 * doc_marginal(gam, lp)
    while active.any():
        cur = np.log(c[None] + g[None] * mass)               # (M,K)
        new = np.log(c[None, :, None] + g[None, :, None] * (mass[:, :, None] + p[None]))
        gain = (new - cur[:, :, None]).sum(1)                 # (M,N)
        feas = (~chosen) & (spent[:, None] + cost[None] <= B + 1e-9) & active[:, None]
        if not feas.any():
            break
        score = (gain if unit else gain / np.maximum(cost[None], 1e-9)) + tie[None]
        score = np.where(feas, score, -np.inf)
        j = score.argmax(1)
        ok = np.isfinite(score[np.arange(M), j])
        for m_ in np.where(ok)[0]:
            chosen[m_, j[m_]] = True
            order[m_].append(int(j[m_]))
        mass[ok] += p[:, j[ok]].T
        spent[ok] += cost[j[ok]]
        active &= ok
    obj = np.log(c[None] + g[None] * mass).sum(1)
    best = int(obj.argmax())
    if return_obj:
        return order[best], obj[best]
    return order[best]


def map_proof(gam, lp):
    """Parameter-free proof decoding: every needed requirement (g>1/2) contributes its MAP witness;
    a null witness means the required evidence is missing from the pool."""
    lp = np.asarray(lp)
    N = lp.shape[1] - 1
    S = []
    for k in range(len(gam)):
        if gam[k] > 0:
            j = int(np.argmax(lp[k]))
            if j < N and j not in S:
                S.append(j)
    if not S:
        S = [int(np.argmax(doc_marginal(gam, lp)))]
    return S


def topk_budget(score, cost, B):
    """Score-ordered filling under a token budget (standard reranker usage)."""
    S, spent = [], 0.0
    for i in np.argsort(-np.asarray(score)):
        if spent + cost[i] <= B + 1e-9:
            S.append(int(i)); spent += cost[i]
    return S


def mmr_budget(score, emb, cost, B, lam=0.7):
    """Maximal marginal relevance under budget; score: relevance logits, emb: (N,d) normalized."""
    s = 1 / (1 + np.exp(-np.asarray(score)))
    sim = emb @ emb.T
    S, spent = [], 0.0
    cand = set(range(len(s)))
    while cand:
        best, bv = None, -1e9
        for i in cand:
            if spent + cost[i] > B + 1e-9:
                continue
            red = max(sim[i, S]) if S else 0.0
            v = lam * s[i] - (1 - lam) * red
            if v > bv:
                best, bv = i, v
        if best is None:
            break
        S.append(best); spent += cost[best]; cand.remove(best)
    return S


def dpp_budget(score, emb, cost, B):
    """Greedy MAP of a quality-diversity DPP, L = diag(q) S diag(q), q = exp(score/2) (Chen et al. 2018)."""
    q = np.exp(np.clip(np.asarray(score), -20, 20) / 2)
    Lk = q[:, None] * (emb @ emb.T) * q[None] + 1e-6 * np.eye(len(q))
    S, spent = [], 0.0
    while True:
        best, bv = None, -np.inf
        for i in range(len(q)):
            if i in S or spent + cost[i] > B + 1e-9:
                continue
            T = S + [i]
            sign, ld = np.linalg.slogdet(Lk[np.ix_(T, T)])
            v = ld if sign > 0 else -np.inf
            if v > bv:
                best, bv = i, v
        if best is None:
            break
        S.append(best); spent += cost[best]
    return S
