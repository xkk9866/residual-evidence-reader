"""Unify MuSiQue / HotpotQA / 2Wiki (distractor setting), MuSiQue-Full, and the
HippoRAG-style open corpora into one jsonl format.

record = {id, question, answers, type, docs:[{title,text}], gold:[idx], gold_order:[idx],
          ntok:[reader tokens per doc], (answerable, missing)}
"""
import json, os, random, ast, sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer

R = 'data/raw'
O = 'data/proc'
os.makedirs(O, exist_ok=True)
tok = AutoTokenizer.from_pretrained(os.path.expanduser('~/models/Qwen2.5-7B-Instruct-AWQ'))


def doc_str(d):
    return f"{d['title']}\n{d['text']}"


def add_ntok(recs):
    flat = [doc_str(d) for r in recs for d in r['docs']]
    lens = []
    for i in range(0, len(flat), 4096):
        lens += [len(x) for x in tok(flat[i:i + 4096], add_special_tokens=False)['input_ids']]
    j = 0
    for r in recs:
        r['ntok'] = lens[j:j + len(r['docs'])]
        j += len(r['docs'])
    return recs


def dump(recs, name):
    recs = add_ntok(recs)
    with open(f'{O}/{name}.jsonl', 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    ng = sum(len(r['gold']) for r in recs) / max(1, len(recs))
    print(name, len(recs), 'avg_docs', sum(len(r['docs']) for r in recs) / len(recs), 'avg_gold', round(ng, 2),
          'avg_ntok', round(sum(sum(r['ntok']) for r in recs) / sum(len(r['ntok']) for r in recs), 1), flush=True)


def musique(path, full=False):
    out = []
    for l in open(path):
        d = json.loads(l)
        docs = [{'title': p['title'], 'text': p['paragraph_text']} for p in d['paragraphs']]
        gold = [i for i, p in enumerate(d['paragraphs']) if p['is_supporting']]
        order = []
        for s in d['question_decomposition']:
            k = s.get('paragraph_support_idx')
            if k is not None and k not in order:
                order.append(k)
        r = {'id': d['id'], 'question': d['question'], 'answers': [d['answer']] + d.get('answer_aliases', []),
             'type': d['id'].split('__')[0], 'docs': docs, 'gold': gold, 'gold_order': order or gold}
        if full:
            r['answerable'] = d['answerable']
        out.append(r)
    return out


def hotpot_like(path, wiki=False):
    out = []
    t = pq.read_table(path).to_pylist()
    for d in t:
        titles = d['context']['title']
        docs = [{'title': ti, 'text': ''.join(s) if not wiki else ' '.join(s)} for ti, s in
                zip(titles, d['context']['sentences'])]
        sf = []
        for ti in d['supporting_facts']['title']:
            if ti not in sf:
                sf.append(ti)
        order = [titles.index(ti) for ti in sf if ti in titles]
        gold = sorted(set(order))
        if len(gold) < 2:
            continue
        out.append({'id': d['id'], 'question': d['question'], 'answers': [d['answer']], 'type': d['type'],
                    'docs': docs, 'gold': gold, 'gold_order': order})
    return out


def open_set(name):
    data = json.load(open(f'{R}/open/{name}.json'))
    corpus = json.load(open(f'{R}/open/{name}_corpus.json'))
    key = {(c['title'], c['text']): i for i, c in enumerate(corpus)}
    out = []
    for d in data:
        if name == 'musique':
            gold_docs = [(p['title'], p['paragraph_text']) for p in d['paragraphs'] if p['is_supporting']]
            ans = [d['answer']] + d.get('answer_aliases', [])
            q, qid = d['question'], d['id']
            order = []
            for s in d['question_decomposition']:
                k = s.get('paragraph_support_idx')
                if k is not None:
                    p = d['paragraphs'][k]
                    if (p['title'], p['paragraph_text']) not in order:
                        order.append((p['title'], p['paragraph_text']))
            gold_docs = order or gold_docs
            typ = qid.split('__')[0]
        else:
            ctx = ast.literal_eval(d['context']) if isinstance(d['context'], str) else d['context']
            sfs = ast.literal_eval(d['supporting_facts']) if isinstance(d['supporting_facts'], str) else d['supporting_facts']
            tmap = {c[0]: c[1] for c in ctx}
            gold_docs = []
            for ti, _ in sfs:
                sep = ' ' if name.startswith('2wiki') else ''
                g = (ti, sep.join(tmap[ti]))
                if g not in gold_docs:
                    gold_docs.append(g)
            ans, q, qid, typ = [d['answer']], d['question'], d['_id'], d['type']
        gid = []
        for g in gold_docs:
            if g in key:
                gid.append(key[g])
            else:  # fall back to title match
                cand = [i for i, c in enumerate(corpus) if c['title'] == g[0]]
                if cand:
                    gid.append(cand[0])
        out.append({'id': qid, 'question': q, 'answers': ans, 'type': typ, 'gold_corpus': gid})
    print(name, 'open', len(out), 'gold matched', sum(len(o['gold_corpus']) for o in out))
    with open(f'{O}/open_{name}.jsonl', 'w') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    rc = [{'title': c['title'], 'text': c['text']} for c in corpus]
    lens = [len(x) for x in tok([doc_str(c) for c in rc], add_special_tokens=False)['input_ids']]
    with open(f'{O}/open_{name}_corpus.jsonl', 'w') as f:
        for c, n in zip(rc, lens):
            c['ntok'] = n
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    return out


def musique_full_pairs():
    """Pair each unanswerable MuSiQue-Full instance with its answerable twin to recover which gold
    paragraph(s) were removed -> ground truth for 'which missing item causes unanswerability'."""
    rec = musique(f'{R}/musique_full/musique_full_v1.0_dev.jsonl', full=True)
    ans = {}
    for r in rec:
        if r['answerable']:
            ans[r['id']] = r
    for r in rec:
        if not r['answerable'] and r['id'] in ans:
            a = ans[r['id']]
            texts = {(d['title'], d['text']) for d in r['docs']}
            r['missing_hops'] = [h for h, k in enumerate(a['gold_order']) if
                                 (a['docs'][k]['title'], a['docs'][k]['text']) not in texts]
            r['present_gold'] = [i for i, d in enumerate(r['docs']) if
                                 (d['title'], d['text']) in {(a['docs'][k]['title'], a['docs'][k]['text']) for k in a['gold']}]
    return rec


if __name__ == '__main__':
    which = sys.argv[1:] or ['musique', 'hotpot', 'wiki', 'full', 'open']
    random.seed(0)
    if 'musique' in which:
        dump(musique(f'{R}/musique/musique_ans_v1.0_train.jsonl'), 'musique_train')
        dump(musique(f'{R}/musique/musique_ans_v1.0_dev.jsonl'), 'musique_dev')
    if 'hotpot' in which:
        tr = hotpot_like(f'{R}/hotpotqa/train-00000-of-00002.parquet') + hotpot_like(f'{R}/hotpotqa/train-00001-of-00002.parquet')
        dump(tr, 'hotpotqa_train')
        dump(hotpot_like(f'{R}/hotpotqa/validation-00000-of-00001.parquet'), 'hotpotqa_dev')
    if 'wiki' in which:
        tr = hotpot_like(f'{R}/2wiki/train-00000-of-00002.parquet', True) + hotpot_like(f'{R}/2wiki/train-00001-of-00002.parquet', True)
        random.shuffle(tr)
        dump(tr[:60000], '2wiki_train')
        dump(hotpot_like(f'{R}/2wiki/validation-00000-of-00001.parquet', True), '2wiki_dev')
    if 'full' in which:
        dump(musique_full_pairs(), 'musique_full_dev')
    if 'open' in which:
        for n in ['musique', 'hotpotqa', '2wikimultihopqa']:
            open_set(n)
