"""Convert our processed pools into the input format of the official Beam Retrieval code (type 'musique').

Same questions, same candidate pools, same passage order and same training subset as our readers:
python code/beam_prep.py musique 0 ; python code/beam_prep.py hotpotqa 20000 ; python code/beam_prep.py 2wiki 20000
"""
import json, sys, os, random

ds, n = sys.argv[1], int(sys.argv[2])
os.makedirs('data/beam_in', exist_ok=True)
for split in ('train', 'dev'):
    out = open(f'data/beam_in/{ds}_{split}.jsonl', 'w')
    for i, l in enumerate(open(f'data/proc/{ds}_{split}.jsonl')):
        if split == 'train' and n and i >= n:
            break
        r = json.loads(l)
        gold = [int(g) for g in r['gold_order']]
        paras = [{'idx': j, 'title': d['title'], 'paragraph_text': d['text'], 'is_supporting': j in gold}
                 for j, d in enumerate(r['docs'])]
        out.write(json.dumps({'id': r['id'], 'question': r['question'], 'paragraphs': paras,
                              'question_decomposition': [{'paragraph_support_idx': g} for g in gold]}) + '\n')
    out.close()
    print(ds, split, 'written')
# checkpoint selection on the same 400 development questions as every other trained model
dev = [l for l in open(f'data/beam_in/{ds}_dev.jsonl')]
open(f'data/beam_in/{ds}_sel400.jsonl', 'w').writelines(random.Random(0).sample(dev, 400))
