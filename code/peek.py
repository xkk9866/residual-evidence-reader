import json
c=json.load(open("data/raw/open/musique_corpus.json")); print(type(c), len(c)); print(c[0])
d=json.load(open("data/raw/open/hotpotqa.json")); print(len(d)); print({k:(str(v)[:200]) for k,v in d[0].items()})
d=json.load(open("data/raw/open/2wikimultihopqa.json")); print(len(d)); print({k:(str(v)[:300]) for k,v in d[0].items()})
print(len(json.load(open("data/raw/open/hotpotqa_corpus.json"))), len(json.load(open("data/raw/open/2wikimultihopqa_corpus.json"))))
for i,l in enumerate(open("data/raw/musique_full/musique_full_v1.0_dev.jsonl")):
    d=json.loads(l)
    if not d["answerable"]:
        print(d["id"], d["question"], [(p["idx"],p["is_supporting"]) for p in d["paragraphs"] if p["is_supporting"]], d["question_decomposition"]); break
