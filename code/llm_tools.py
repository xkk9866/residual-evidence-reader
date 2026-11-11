"""LLM utilities on int4 (AWQ) models via vLLM:
  qa   : read the selected evidence and answer -> EM / F1
  setr : SetR-style (ACL'25) set selection with information-requirement CoT (zero-shot baseline)
usage:
  python llm_tools.py qa DS name1:k3,t512 name2:t512 ...      (sets from results/sel/{DS}_{name}.json)
  python llm_tools.py setr DS
"""
import json, os, re, string, sys, pickle, collections
from vllm import LLM, SamplingParams

READER = os.environ.get('READER', os.path.expanduser('~/models/Qwen3-8B-AWQ'))
RTAG = os.environ.get('RTAG', 'qwen3_8b')


def norm(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def em_f1(pred, golds):
    best_em, best_f1 = 0.0, 0.0
    p = norm(pred)
    for g in golds:
        g = norm(g)
        best_em = max(best_em, float(p == g))
        pt, gt = p.split(), g.split()
        common = collections.Counter(pt) & collections.Counter(gt)
        ns = sum(common.values())
        if ns:
            pr, rc = ns / len(pt), ns / len(gt)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
    return best_em, best_f1


def chat(tok, user, system):
    return tok.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


QA_SYS = ('You are a precise question answering assistant. Answer the question using the given passages. '
          'Respond with only the short answer (an entity, date, number, or yes/no), without explanation.')


def qa_prompt(r, S):
    ctx = '\n\n'.join(f"Passage {j + 1}: {r['docs'][i]['title']}\n{r['docs'][i]['text']}" for j, i in enumerate(S))
    return f"{ctx}\n\nQuestion: {r['question']}\nAnswer:"


def clean(a):
    a = a.strip().split('\n')[0]
    a = re.sub(r'^(answer|the answer is)\s*[:\-]?\s*', '', a, flags=re.I)
    return a.strip().rstrip('.')


_LLM = []


def get_llm():
    if not _LLM:
        _LLM.append(LLM(READER, quantization='awq_marlin', max_model_len=int(os.environ.get('MAXLEN', '6144')), gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')), enforce_eager=False))
    return _LLM[0]


def run_qa(ds, specs):
    recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
    nmax = int(os.environ.get('QA_N', '0'))
    if nmax:
        recs = recs[:nmax]
    llm = get_llm()
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=32)
    os.makedirs('results/qa', exist_ok=True)
    allres = {}
    for spec in specs:
        name, keys = spec.split(':')
        if name in ('oracle', 'fullpool', 'noctx'):
            S = {'oracle': [r['gold_order'] if r.get('gold_order') else r['gold'] for r in recs],
                 'fullpool': [list(range(len(r['docs']))) for r in recs],
                 'noctx': [[] for r in recs]}[name]
            sets = {k: S for k in keys.split(',')}
        else:
            sets = json.load(open(f'results/sel/{ds}_{name}.json'))
            sets = {k: v[:len(recs)] for k, v in sets.items()}
        for key in keys.split(','):
            prompts = [chat(tok, qa_prompt(r, S), QA_SYS) for r, S in zip(recs, sets[key])]
            outs = llm.generate(prompts, sp, use_tqdm=False)
            ems, f1s, preds = [], [], []
            for r, o in zip(recs, outs):
                a = clean(o.outputs[0].text)
                e, f = em_f1(a, r['answers'])
                ems.append(e); f1s.append(f); preds.append(a)
            res = {'EM': round(100 * sum(ems) / len(ems), 2), 'F1': round(100 * sum(f1s) / len(f1s), 2)}
            allres[f'{name}@{key}'] = res
            json.dump({'metrics': res, 'preds': preds}, open(f'results/qa/{ds}_{name}_{key}_{RTAG}.json', 'w'))
            print(f'{ds} {name}@{key} {RTAG}', res, flush=True)
    return allres


SETR_SYS = 'You are a helpful assistant that selects passages for answering questions.'


def setr_prompt(r):
    ctx = '\n'.join(f"[{i + 1}] {d['title']}: {d['text']}" for i, d in enumerate(r['docs']))
    q = r['question']
    return (f"I will provide you with {len(r['docs'])} passages, each indicated by a numerical identifier []. "
            f"Select the passages based on their relevance to the search query: {q}.\n\n{ctx}\n\n"
            f"Search Query: {q}\n\nPlease follow the steps below:\n"
            "Step 1. Please list up the information requirements to answer the query.\n"
            "Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.\n"
            "Step 3. Choose the passages that mostly covers clear and diverse information to answer the query. "
            "Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', "
            "e.g., ### Final Selection: [2] [1].")


def run_setr(ds):
    recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
    nmax = int(os.environ.get('MAX_Q', '0'))
    if nmax:
        recs = recs[:nmax]
    out_path = f'results/scores/{ds}_setr_{RTAG}.pkl'
    if nmax:
        out_path = f'/tmp/setr_smoke_{ds}.pkl'
    elif os.path.exists(out_path):
        print('exists', out_path, flush=True)
        return
    llm = LLM(READER, quantization='awq_marlin', max_model_len=8192, gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=768)
    outs = llm.generate([chat(tok, setr_prompt(r), SETR_SYS) for r in recs], sp, use_tqdm=False)
    sets = []
    for r, o in zip(recs, outs):
        t = o.outputs[0].text
        m = t.split('Final Selection')[-1]
        ids = []
        for x in re.findall(r'\[(\d+)\]', m):
            i = int(x) - 1
            if 0 <= i < len(r['docs']) and i not in ids:
                ids.append(i)
        sets.append(ids)
    os.makedirs('results/scores', exist_ok=True)
    pickle.dump(sets, open(out_path, 'wb'))
    print('setr done', ds, 'avg selected', sum(map(len, sets)) / len(sets), flush=True)


RANK_SYS = 'You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.'


def rank_prompt(r):
    ctx = '\n'.join(f"[{i + 1}] {d['title']}: {d['text']}" for i, d in enumerate(r['docs']))
    q = r['question']
    return (f"I will provide you with {len(r['docs'])} passages, each indicated by number identifier []. "
            f"Rank the passages based on their relevance to the query: {q}. Passages that are needed to answer the "
            f"(multi-hop) query should be ranked first.\n\n{ctx}\n\nSearch Query: {q}\n"
            f"Rank all {len(r['docs'])} passages above in descending order of relevance. The output format should be "
            "[] > [], e.g., [1] > [2]. Only respond with the ranking results, do not say any word or explain.")


def run_rank(ds):
    """RankGPT-style listwise LLM reranking (Sun et al., EMNLP 2023)."""
    recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
    nmax = int(os.environ.get('MAX_Q', '0'))
    if nmax:
        recs = recs[:nmax]
    out_path = f'results/scores/{ds}_rankgpt_{RTAG}.pkl'
    if nmax:
        out_path = f'/tmp/rank_smoke_{ds}.pkl'
    elif os.path.exists(out_path):
        print('exists', out_path, flush=True)
        return
    llm = LLM(READER, quantization='awq_marlin', max_model_len=8192, gpu_memory_utilization=float(os.environ.get('GPU_UTIL', '0.85')))
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=200)
    outs = llm.generate([chat(tok, rank_prompt(r), RANK_SYS) for r in recs], sp, use_tqdm=False)
    sets = []
    for r, o in zip(recs, outs):
        ids = []
        for x in re.findall(r'\[(\d+)\]', o.outputs[0].text):
            i = int(x) - 1
            if 0 <= i < len(r['docs']) and i not in ids:
                ids.append(i)
        sets.append(ids + [i for i in range(len(r['docs'])) if i not in ids])
    pickle.dump(sets, open(out_path, 'wb'))
    print('rankgpt done', ds, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'qa':
        run_qa(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == 'qamulti':       # one model load for many pools: "DS|name:keys" ...
        for a in sys.argv[2:]:
            d, spec = a.split('|')
            if not os.path.exists(f'data/proc/{d}.jsonl'):
                print('skip missing pool', d, flush=True); continue
            if os.path.exists(f'results/qa/{d}_' + spec.split(':')[0] + '_' + spec.split(':')[1].split(',')[-1] + f'_{RTAG}.json'):
                continue
            run_qa(d, [spec])
    elif sys.argv[1] == 'rank':
        run_rank(sys.argv[2])
    elif sys.argv[1] == 'setr':
        run_setr(sys.argv[2])
