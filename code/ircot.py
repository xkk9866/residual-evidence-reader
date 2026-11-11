"""IRCoT (Trivedi et al., ACL 2023) on the open corpora, run locally with the same dense retriever and answer LLM.

Interleaving retrieval with chain-of-thought: retrieve K passages for the question; then repeatedly let the LLM write
the next reasoning sentence given the question, all passages collected so far and the reasoning so far, and retrieve
K passages with that sentence as the query. Stops when a sentence contains "answer is", after MAX_STEPS reasoning
steps, or when MAX_PARAS passages are collected (the paper's settings: K tuned in {2..6}, at most 15 paragraphs).
The retriever is the bge-large-en-v1.5 index used by every other open-corpus pipeline; the LLM is Qwen3-8B-AWQ.

usage: python code/ircot.py DS   -> data/proc/ircot_{DS}.jsonl, results/metrics/ircot_{DS}.json
"""
import json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, 'code')
from open_pipeline import BGE_Enc, QP, dstr
from llm_tools import chat, clean, em_f1, get_llm, qa_prompt, QA_SYS
from vllm import SamplingParams

DS = sys.argv[1]
K, MAX_STEPS, MAX_PARAS = int(os.environ.get('IRCOT_K', 4)), 5, 15
SYS = ('You answer a multi-hop question by reasoning step by step over Wikipedia passages. '
       'Write ONLY the next single sentence of the reasoning. When the reasoning reaches the answer, '
       'write a sentence of the form "So the answer is: <short answer>."')


def main():
    qs = [json.loads(l) for l in open(f'data/proc/open_{DS}.jsonl')]
    corpus = [json.loads(l) for l in open(f'data/proc/open_{DS}_corpus.jsonl')]
    ce = torch.tensor(np.load(f'data/proc/open_{DS}_cemb.npy').astype(np.float32), device='cuda')
    qe = torch.tensor(np.load(f'data/proc/open_{DS}_qemb.npy').astype(np.float32), device='cuda')
    enc = BGE_Enc()
    llm = get_llm()
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=64, stop=['\n'])


    def retrieve(qv, have, k=K):
        out = []
        for i in torch.topk(ce @ qv, k + len(have)).indices.tolist():
            if i not in have and i not in out:
                out.append(int(i))
            if len(out) == k:
                break
        return out


    def prompt(q, cids, cot):
        # passages are capped at 1,500 characters each so that 15 of them fit the model length (MAXLEN=8192 is used)
        ctx = '\n\n'.join(f"Wikipedia Title: {corpus[i]['title']}\n{corpus[i]['text'][:1500]}" for i in cids)
        return chat(tok, f"{ctx}\n\nQ: {q['question']}\nA: {' '.join(cot)}".rstrip(), SYS)


    st = [{'cids': retrieve(qe[i], []), 'cot': [], 'done': False, 'retr': 1} for i in range(len(qs))]
    for step in range(MAX_STEPS):
        act = [i for i, s in enumerate(st) if not s['done']]
        if not act:
            break
        outs = llm.generate([prompt(qs[i], st[i]['cids'], st[i]['cot']) for i in act], sp)
        sents = [o.outputs[0].text.strip() for o in outs]
        need = []
        for i, t in zip(act, sents):
            st[i]['cot'].append(t)
            if 'answer is' in t.lower() or len(st[i]['cids']) >= MAX_PARAS or step == MAX_STEPS - 1:
                st[i]['done'] = True
            else:
                need.append(i)
        if need:
            qv = torch.tensor(enc([QP + st[i]['cot'][-1] for i in need]), device='cuda')
            for j, i in enumerate(need):
                st[i]['cids'] += retrieve(qv[j], st[i]['cids'])[:MAX_PARAS - len(st[i]['cids'])]
                st[i]['retr'] += 1

    # answers: IRCoT's own reasoning answer; otherwise (and as a second protocol) the shared QA prompt on its passages
    qa_outs = llm.generate([chat(tok, qa_prompt({'question': q['question'], 'docs': [dict(corpus[c], text=corpus[c]['text'][:1500]) for c in s['cids']]},
                                                range(len(s['cids']))), QA_SYS) for q, s in zip(qs, st)],
                           SamplingParams(temperature=0, max_tokens=32))
    rows, M = [], {'cot': [], 'qa': [], 'rec': [], 'n': [], 'retr': []}
    for q, s, o in zip(qs, st, qa_outs):
        m = re.search(r'answer is[:\s]*(.+)', ' '.join(s['cot']), flags=re.I)
        a_cot = clean(m.group(1)) if m else clean(o.outputs[0].text)
        a_qa = clean(o.outputs[0].text)
        M['cot'].append(em_f1(a_cot, q['answers'])); M['qa'].append(em_f1(a_qa, q['answers']))
        M['rec'].append(set(q['gold_corpus']) <= set(s['cids'])); M['n'].append(len(s['cids'])); M['retr'].append(s['retr'])
        rows.append({'id': q['id'], 'cids': s['cids'], 'cot': s['cot'], 'answer_cot': a_cot, 'answer_qa': a_qa, 'retr': s['retr']})
    with open(f'data/proc/ircot_{DS}.jsonl', 'w') as f:
        f.writelines(json.dumps(r) + '\n' for r in rows)
    res = {'EM': 100 * np.mean([e for e, _ in M['cot']]), 'F1': 100 * np.mean([f for _, f in M['cot']]),
           'EM_qa': 100 * np.mean([e for e, _ in M['qa']]), 'F1_qa': 100 * np.mean([f for _, f in M['qa']]),
           'recall_all': 100 * np.mean(M['rec']), 'n_passages': float(np.mean(M['n'])), 'retrievals': float(np.mean(M['retr']))}
    res = {k: round(float(v), 2) for k, v in res.items()}
    json.dump(res, open(f'results/metrics/ircot_{DS}.json', 'w'))
    print(DS, res)


if __name__ == '__main__':
    main()
