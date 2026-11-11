"""Inference helpers shared by the open-domain pipeline and the unmet-requirement retrieval head."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import PROVE

ENC = 'BAAI/bge-reranker-v2-m3'


def load_prove(ckpt, head='circuit', K=6, layers=3):
    m = PROVE(ENC, K=K, layers=layers, head=head)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict({k: v.float() for k, v in sd.items()})
    return m.cuda().eval()


class Runner:
    def __init__(self, model, maxlen=256):
        self.m = model
        self.tok = AutoTokenizer.from_pretrained(ENC)
        self.maxlen = maxlen

    @torch.no_grad()
    def encode_question(self, q, docs):
        tk = self.tok
        a = tk([q] * len(docs), [f"{d['title']}\n{d['text']}" for d in docs], truncation='only_second',
               max_length=self.maxlen, padding=True, return_tensors='pt')
        b = tk([q], truncation=True, max_length=64, padding='max_length', return_tensors='pt')
        L = a['input_ids'].shape[1]
        if b['input_ids'].shape[1] < L:
            b = {k: F.pad(v, (0, L - v.shape[1]), value=(tk.pad_token_id if k == 'input_ids' else 0)) for k, v in b.items()}
        else:
            b = {k: v[:, :L] for k, v in b.items()}
        ids = torch.cat([a['input_ids'], b['input_ids']]).cuda()
        att = torch.cat([a['attention_mask'], b['attention_mask']]).cuda()
        hs_all, rho_all = [], []
        for i in range(0, ids.shape[0], 48):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                hs, rho = self.m.encode(ids[i:i + 48], att[i:i + 48])
            hs_all.append(hs.float()); rho_all.append(rho.float())
        hs, rho = torch.cat(hs_all), torch.cat(rho_all)
        n = len(docs)
        return hs[:n], rho[:n], att[:n].bool(), hs[n], att[n].bool()

    @torch.no_grad()
    def run(self, q, docs, drop_variants=None, want_r=False):
        """returns dict per variant: gam (K,), lg (K,N), rho (N,), r (K,d)"""
        h, rho, tm, qh, qm = self.encode_question(q, docs)
        if self.m.head == 'pointwise':
            return [{'rho': rho.cpu().numpy()}]
        X = self.m.proj(h); qX = self.m.proj(qh)
        qvec = (qX * qm.unsqueeze(-1)).sum(0) / qm.sum()
        N = X.shape[0]
        variants = [torch.ones(N, dtype=torch.bool, device='cuda')]
        for dd in (drop_variants or []):
            v = torch.ones(N, dtype=torch.bool, device='cuda'); v[dd] = False; variants.append(v)
        docmask = torch.stack(variants)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            gam, lp, r = self.m.infer(X, tm, qvec, rho, docmask, return_r=True)
        res = []
        for v in range(len(variants)):
            d = {'rho': rho.cpu().numpy(), 'gam': gam[v].cpu().numpy(), 'lp': lp[v].cpu().numpy()}
            if want_r:
                d['r'] = r[v].float().cpu().numpy()
            res.append(d)
        return res
