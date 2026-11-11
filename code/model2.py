"""PROVE v2: a joint pool reader with requirement tokens, an optimal-transport witness layer and the
AND-OR sufficiency circuit.

input   : <s> [R_1]..[R_K] question </s> d_1 </s> d_2 </s> ... d_N </s>   (one sequence per pool)
reader  : the full cross-encoder (bge-reranker-v2-m3 backbone); requirement tokens attend to and are attended
          by every evidence token in every layer -> requirements and bridges are inferred jointly
witness : requirement x evidence affinities -> Sinkhorn partial matching with dustbins
          (each requirement is certified by exactly one item: a document or the null "missing" item;
           each document certifies at most one requirement)  -> witness probabilities p_ki, p_k0
circuit : Phi(S) = prod_k [1 - g_k (1 - sum_{i in S} p_ki)]
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def log1mexp(x):
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def log_sinkhorn(Z, log_mu, log_nu, iters):
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(0), 1)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(1), 0)
    return Z + u.unsqueeze(1) + v.unsqueeze(0)


class Circuit:
    @staticmethod
    def log_phi(gam, lp, m):
        """gam (...,K) need logits; lp (...,K,N+1) witness log-probs (last = null); m (...,N) membership"""
        lpd = lp[..., :-1] + torch.log(m.clamp(min=1e-30)).unsqueeze(-2)
        return torch.logaddexp(F.logsigmoid(-gam), F.logsigmoid(gam) + torch.logsumexp(lpd, -1)).sum(-1)

    @staticmethod
    def log_phi_pool(gam, lp):
        return log1mexp(F.logsigmoid(gam) + lp[..., -1]).sum(-1)


class ProveJoint(nn.Module):
    def __init__(self, enc_name, K=6, d=256, sink_iters=20, transport=True, head='circuit'):
        super().__init__()
        self.enc = AutoModel.from_pretrained(enc_name, attn_implementation='sdpa', add_pooling_layer=False)
        H = self.enc.config.hidden_size
        self.K, self.d, self.iters, self.transport, self.head = K, d, sink_iters, transport, head
        we = self.enc.get_input_embeddings().weight
        mask_id = 250001  # <mask> in the XLM-R vocabulary
        self.req = nn.Parameter(we[mask_id].detach().clone().unsqueeze(0).repeat(K, 1) + 0.02 * torch.randn(K, H))
        self.Wa = nn.Linear(H, d, bias=False)   # requirement -> evidence-token attention
        self.Wk = nn.Linear(H, d, bias=False)
        self.Wv = nn.Linear(H, d, bias=False)
        self.Wb = nn.Linear(H, d, bias=False)   # requirement -> witness matching
        self.null = nn.Linear(H, 1)             # requirement is certified by nothing in the pool
        self.gate = nn.Linear(H, 1)             # requirement is needed
        self.bin = nn.Parameter(torch.tensor(0.0))   # dustbin score (document certifies no requirement)
        self.doc_cls = nn.Linear(H, 1)          # only used by the per-document BCE ablation

    def read(self, ids, att, spans):
        """ids/att: (1,L); spans: list of (start,end) per document. returns r (K,H), doc token states list"""
        emb = self.enc.get_input_embeddings()(ids)
        emb = torch.cat([emb[:, :1], self.req.unsqueeze(0).to(emb.dtype), emb[:, 1 + self.K:]], 1)
        h = self.enc(inputs_embeds=emb, attention_mask=att).last_hidden_state[0]
        return h[1:1 + self.K], h, spans

    def witness(self, r, h, spans):
        K, d = self.K, self.d
        N = len(spans)
        L = max(e - s for s, e in spans)
        idx = torch.zeros(N, L, dtype=torch.long, device=h.device)
        msk = torch.zeros(N, L, dtype=torch.bool, device=h.device)
        for i, (s, e) in enumerate(spans):
            idx[i, :e - s] = torch.arange(s, e, device=h.device)
            msk[i, :e - s] = True
        X = h[idx]                                                     # (N,L,H)
        if self.head == 'bce':
            pooled = (X * msk.unsqueeze(-1)).sum(1) / msk.sum(1, keepdim=True)
            return self.doc_cls(pooled).squeeze(-1).float()
        q = self.Wa(r)                                                 # (K,d)
        att = torch.einsum('kd,nld->knl', q, self.Wk(X)) / math.sqrt(d)
        att = att.float().masked_fill(~msk.unsqueeze(0), -1e4).softmax(-1)
        z = torch.einsum('knl,nld->knd', att.to(X.dtype), self.Wv(X))   # requirement-specific evidence summary
        S = torch.einsum('kd,knd->kn', self.Wb(r), z).float() / math.sqrt(d)
        nul = self.null(r).float()                                     # (K,1)
        gam = self.gate(r).squeeze(-1).float()
        if not self.transport:                                         # ablation: independent softmax rows
            return gam, torch.cat([S, nul], 1).log_softmax(-1)
        Z = torch.cat([torch.cat([S, nul], 1),
                       torch.cat([self.bin.expand(1, N), self.bin.view(1, 1)], 1).float()], 0)   # (K+1,N+1)
        log_mu = torch.cat([torch.zeros(K, device=h.device), torch.tensor([math.log(N)], device=h.device)])
        log_nu = torch.cat([torch.zeros(N, device=h.device), torch.tensor([math.log(K)], device=h.device)])
        P = log_sinkhorn(Z, log_mu, log_nu, self.iters)[:K]            # requirement rows
        return gam, P.log_softmax(-1)
