"""Convert official Beam Retrieval (NAACL'24) dev predictions into our per-question set format."""
import json, pickle
for ds, f in [('musique_dev', 'musique_pred_best.json'), ('hotpotqa_dev', 'hotpot_pred_best_retr.json'),
              ('2wiki_dev', '2wiki_pred_best.json')]:
    p = json.load(open('data/beam_pred/' + f))
    recs = [json.loads(l) for l in open(f'data/proc/{ds}.jsonl')]
    sets = [list(p[r['id']]) for r in recs]
    pickle.dump(sets, open(f'results/scores/{ds}_beam.pkl', 'wb'))
    print(ds, len(sets), 'missing', sum(r['id'] not in p for r in recs), 'avg len', sum(map(len, sets)) / len(sets))
