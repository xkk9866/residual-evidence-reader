"""Zero-shot Qwen3-Reranker-4B (2025) baseline, int8 (bitsandbytes) weights, official yes/no scoring prompt.
usage: python q3rr.py DS [max_questions]   -> results/scores/{DS}_q3rr.pkl  (list of {'rho': logit(yes)-logit(no)})"""
import json, pickle, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

M = '/data/yaokang/models/Qwen3-Reranker-4B'
ds = sys.argv[1]
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 0
tok = AutoTokenizer.from_pretrained(M, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(M, quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                                             device_map={'': 0}, torch_dtype=torch.float16).eval()
yes, no = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
pre = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. '
       'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
suf = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INS = 'Given a multi-hop question, judge whether the passage provides evidence needed to answer it'


@torch.no_grad()
def scores(q, docs):
    texts = [f"{pre}<Instruct>: {INS}\n<Query>: {q}\n<Document>: {d['title']}\n{d['text']}{suf}" for d in docs]
    enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(0)
    lg = model(**enc).logits[:, -1, :]
    return (lg[:, yes] - lg[:, no]).float().cpu().numpy()


recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
if nmax:
    recs = recs[:nmax]
t0 = time.time()
out = [{'rho': scores(r['question'], r['docs'])} for r in recs]
pickle.dump(out, open(f'results/scores/{ds}_q3rr.pkl', 'wb'))
print('q3rr', ds, len(recs), 'sec', round(time.time() - t0, 1), flush=True)
