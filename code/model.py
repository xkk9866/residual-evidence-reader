"""PROVE: Evidence Sufficiency Circuit network.

encoder  : question-aware cross-encoder (bge-reranker-v2-m3) -> token states H_i for each (q, d_i)
decoder  : K latent evidence requirements inferred jointly over the whole pool (DETR-style)
circuit  : each requirement k is needed with prob. g_k and has a *witness* distribution p_k over
           pool documents plus a null witness (the required evidence is absent from the pool).
           Phi(S) = prod_k [ 1 - g_k (1 - sum_{i in S} p_ki) ]
           (AND over requirements, OR over alternative witnesses)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification


def log1mexp(x):
    """log(1 - exp(x)) for x <= 0, numerically stable."""
    x = x.clamp(max=-1e-7)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


class Circuit:
    """Closed-form log-sufficiency.  gam: (..., K) need logits ; lp: (..., K, N+1) witness log-probs
    (last column = null witness) ; m: (..., N) 0/1 membership of S."""

    @staticmethod
    def log_phi(gam, lp, m):
        lpd = lp[..., :-1] + torch.log(m.clamp(min=1e-30)).unsqueeze(-2)     # -inf for i not in S
        logmass = torch.logsumexp(lpd, -1)                                     # log sum_{i in S} p_ki
        return torch.logaddexp(F.logsigmoid(-gam), F.logsigmoid(gam) + logmass).sum(-1)

    @staticmethod
    def log_phi_pool(gam, lp):
        """sufficiency of the whole (variant) pool: prod_k (1 - g_k p_k0)"""
        return log1mexp(F.logsigmoid(gam) + lp[..., -1]).sum(-1)


class DecoderLayer(nn.Module):
    def __init__(self, d, h, ff, p=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ca = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dp = nn.Dropout(p)

    def forward(self, r, mem, mem_pad):
        x = self.n1(r)
        r = r + self.dp(self.sa(x, x, x, need_weights=False)[0])
        x = self.n2(r)
        r = r + self.dp(self.ca(x, mem, mem, key_padding_mask=mem_pad, need_weights=False)[0])
        return r + self.dp(self.ff(self.n3(r)))


class PROVE(nn.Module):
    def __init__(self, enc_name, K=6, d=512, layers=3, heads=8, head='circuit'):
        super().__init__()
        self.cls = AutoModelForSequenceClassification.from_pretrained(enc_name, num_labels=1)
        self.enc = self.cls.roberta
        H = self.enc.config.hidden_size
        self.head = head
        self.K, self.d = K, d
        if head == 'pointwise':
            return
        self.proj = nn.Sequential(nn.Linear(H, d), nn.LayerNorm(d))
        self.slots = nn.Parameter(torch.randn(K, d) * 0.02)
        self.qproj = nn.Linear(d, d)
        self.layers = nn.ModuleList([DecoderLayer(d, heads, 4 * d) for _ in range(layers)])
        self.nout = nn.LayerNorm(d)
        self.Wa = nn.Linear(d, d, bias=False)    # requirement -> token attention query
        self.Wb = nn.Linear(d, d, bias=False)    # requirement -> witness matching
        self.Wz = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self.null = nn.Linear(d, 1)              # "required evidence is not in the pool"
        self.alpha = nn.Parameter(torch.tensor(1.0))   # weight of the question-relevance prior

    # ---------------- encoder ----------------
    def encode(self, ids, att):
        hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        rho = self.cls.classifier(hs).squeeze(-1)
        return hs, rho

    # ---------------- requirement inference + circuit readout ----------------
    def infer(self, X, tmask, qvec, rho, docmask, return_r=False):
        """X: (N,L,d) token memory, tmask: (N,L) bool valid, qvec: (d,), rho: (N,),
        docmask: (V,N) bool -> which docs exist in each of V pool variants.
        returns gam (V,K), lp (V,K,N+1) witness log-probs (last = null)."""
        V = docmask.shape[0]
        N, L, d = X.shape
        mem = X.reshape(1, N * L, d).expand(V, -1, -1)
        valid = (tmask.unsqueeze(0) & docmask.unsqueeze(-1)).reshape(V, N * L)
        r = (self.slots + self.qproj(qvec)).unsqueeze(0).expand(V, -1, -1).contiguous()
        for lyr in self.layers:
            r = lyr(r, mem, ~valid)
        r = self.nout(r)                                                  # (V,K,d)
        att = torch.einsum('vkd,nld->vknl', self.Wa(r), X) / math.sqrt(d)
        att = att.masked_fill(~tmask.view(1, 1, N, L), -1e4).softmax(-1)
        z = torch.einsum('vknl,nld->vknd', att, self.Wz(X))
        lg = torch.einsum('vkd,vknd->vkn', self.Wb(r), z) / math.sqrt(d) + self.alpha * rho.view(1, 1, N)
        lg = lg.float().masked_fill(~docmask.unsqueeze(1), -1e4)          # absent docs cannot witness
        lg = torch.cat([lg, self.null(r).float()], -1)                    # (V,K,N+1)
        lp = lg.log_softmax(-1)
        gam = self.gate(r).squeeze(-1).float()
        if return_r:
            return gam, lp, r
        return gam, lp

    def forward_question(self, hs, rho, qhs, qmask, tmask, world_variants=None, return_r=False):
        """hs: (N,L,H) doc-pair token states; qhs: (Lq,H) question-only states."""
        X = self.proj(hs)
        qX = self.proj(qhs)
        qvec = (qX * qmask.unsqueeze(-1)).sum(0) / qmask.sum()
        N = X.shape[0]
        variants = [torch.ones(N, dtype=torch.bool, device=X.device)]
        for dd in (world_variants or []):
            v = torch.ones(N, dtype=torch.bool, device=X.device)
            v[dd] = False
            variants.append(v)
        docmask = torch.stack(variants)
        return self.infer(X, tmask, qvec, rho, docmask, return_r), docmask
