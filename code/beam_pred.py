"""Inference with a locally trained Beam Retrieval checkpoint (official model code, ext/beam_retriever).

Writes two prediction files in our per-question format (list of passage indices, in order):
  results/scores/{ds}_beamL.pkl  chain of exactly |G| passages (the official protocol, which passes the hop count)
  results/scores/{ds}_beamK.pkl  no hop count: the chain of K hops (K=5 for 20-passage pools, 4 otherwise),
                                 followed by the remaining passages in order of the first-hop score
usage: python code/beam_pred.py <ds> <ckpt> <encoder_dir> [tag]   (tag: output name suffix, default 'beam' -> {ds}_beamL/K.pkl)
"""
import json, pickle, sys
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
sys.path.insert(0, 'ext/beam_retriever')
from retrieval.retriever_model import Retriever
from utils.utils import load_saved

ds, ckpt, enc = sys.argv[1:4]
TAG = sys.argv[4] if len(sys.argv) > 4 else 'beam'
tok = AutoTokenizer.from_pretrained(enc)
cfg = AutoConfig.from_pretrained(enc)
cfg.cls_token_id, cfg.sep_token_id = tok.cls_token_id, tok.sep_token_id
model = Retriever(cfg, enc, encoder_class=AutoModel, mean_passage_len=120, beam_size=2, gradient_checkpointing=False)
model = load_saved(model, ckpt).cuda().eval()

recs = [json.loads(l) for l in open(f'data/beam_in/{ds}_dev.jsonl')]
L, K = [], []
for r in recs:
    q = r['question'][:-1] if r['question'].endswith('?') else r['question']
    qc = tok.encode(q, add_special_tokens=False, return_tensors='pt', truncation=True, max_length=512).squeeze(0).cuda()
    cc = [tok.encode(p['title'] + '.' + p['paragraph_text'], add_special_tokens=False, return_tensors='pt',
                     truncation=True, max_length=512 - qc.shape[-1]).squeeze(0).cuda() for p in r['paragraphs']]
    n, hk = len(cc), (5 if len(cc) > 10 else 4)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        L.append(list(model([qc], [cc], [], hop=len(r['question_decomposition']))['current_preds'][0]))
        chain = list(model([qc], [cc], [], hop=hk)['current_preds'][0])
        # first-hop scores of the official first-hop classifier, to order the passages after the chain
        s = []
        for c in cc:
            x = torch.cat([torch.tensor([cfg.cls_token_id], device='cuda'), qc, c, torch.tensor([cfg.sep_token_id], device='cuda')])[:512]
            h = model.encoder(input_ids=x[None], attention_mask=torch.ones_like(x)[None])[0][:, 0, :]
            s.append(float(model.hop_classifier_layer(h)[0, 1]))
    # sanitise: in-range, no duplicates (the official decoder can repeat an index on pools with fewer than `hop` candidates)
    chain = [int(x) for j, x in enumerate(chain) if 0 <= x < n and x not in chain[:j]]
    L[-1] = [int(x) for j, x in enumerate(L[-1]) if 0 <= x < n and x not in L[-1][:j]]
    K.append(chain + [i for i in sorted(range(n), key=lambda i: -s[i]) if i not in chain])
pickle.dump(L, open(f'results/scores/{ds}_{TAG}L.pkl', 'wb'))
pickle.dump(K, open(f'results/scores/{ds}_{TAG}K.pkl', 'wb'))
g = [set(p['idx'] for p in r['paragraphs'] if p['is_supporting']) for r in recs]
print(ds, 'SupEM@|G|', round(100 * sum(set(a) == b for a, b in zip(L, g)) / len(g), 2),
      'CER@4', round(100 * sum(b <= set(a[:4]) for a, b in zip(K, g)) / len(g), 2))
