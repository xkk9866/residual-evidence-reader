"""Round q50: the two-sided residual reader (symmetric difference; ckpt/mq_delta) and every experiment of the paper
that depends on it, as one dependency graph for code/sched2.py on GPU 2 (shared card, memory-budgeted admission).
  python code/q50_jobs.py && nohup .venv/bin/python code/sched2.py sched/q50_jobs.json > logs/sched2.out 2>&1 &
"""
import json, os

PY = '.venv/bin/python'
E = f'{PY} code/eval_select.py'
MQ = 'data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl'
DSS = ['musique', 'hotpotqa', '2wikimultihopqa']
OP = ','.join(f'data/proc/openpool_{d}_N20.jsonl' for d in DSS)
N40 = ','.join(f'data/proc/openpool_{d}_N40.jsonl' for d in DSS)
TR = f'{PY} code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1 --objective delta'
J = []
TRAIN_MEM, INF_MEM, LLM_MEM, BEAM_MEM = 13500, 5500, 9500, 4000


PFX = 'q50_'      # job names are prefixed so that the state files of earlier rounds (sched/*.ok) cannot shadow them


def job(name, cmd, need, deps=(), prio=50, llm=False, soft=False):
    J.append({'name': PFX + name, 'cmd': cmd, 'need': need, 'deps': [PFX + d for d in deps], 'prio': prio, 'llm': llm, 'soft': soft})


def inf(ck, out, data, extra='--objective delta'):
    return f'{PY} code/train3.py {extra} --ckpt {ck} --out {out} --infer {data}'


D = 'ckpt/mq_delta/model.pt'
# ---------------- main model: train, MuSiQue dev + MuSiQue-Full, first evaluation
job('tr_delta', f'{TR} --out ckpt/mq_delta > logs/mq_delta.log 2>&1 && {inf(D, "ckpt/mq_delta", MQ)}', TRAIN_MEM, prio=1)
job('eval_delta', f'{E} musique_dev delta=ckpt/mq_delta/pred_musique_dev.pkl:order && {PY} code/ans_eval.py delta=ckpt/mq_delta/pred_musique_full_dev.pkl:order && '
    f'{PY} code/delta_stats.py delta=ckpt/mq_delta && {PY} code/beam_matched.py delta=ckpt/mq_delta/pred_musique_dev.pkl lacc=ckpt/mq_lacc/pred_musique_dev.pkl lach=ckpt/mq_lacr/pred_musique_dev.pkl jbce3=ckpt/mq_jbce3/pred_musique_dev.pkl jbce4=ckpt/mq_jbce4/pred_musique_dev.pkl chain=ckpt/mq_chain/pred_musique_dev.pkl && '
    + ' && '.join(f'{PY} code/signif.py musique_dev delta {b} k2,k3,k4,k5,t384,t512,t768,t1024' for b in ['lacc', 'jbce3', 'jbce4', 'chain', 'lach']), 0, ['tr_delta'], prio=1)
# open pools (round 0 of every loop), larger pools, longer passages, walk-length variants
job('inf0_delta', inf(D, 'ckpt/mq_delta', OP), INF_MEM, ['tr_delta'], prio=2)
job('inf40_delta', inf(D, 'ckpt/mq_delta', N40), 7500, ['tr_delta'], prio=6)
job('infL320_delta', inf(D, 'ckpt/mq_delta_L320', 'data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl', '--objective delta --doclen 320'), 7500, ['tr_delta'], prio=7)
for W in (2, 3, 4, 6):
    job(f'infW{W}_delta', inf(D, f'ckpt/mq_delta_W{W}', 'data/proc/musique_dev.jsonl', f'--objective delta --walk {W}') + f' > logs/mq_delta_W{W}.log 2>&1', INF_MEM, ['tr_delta'], prio=7)

# ---------------- seeds (stage-2 seeds 43/44 from the same stage-1 reader, as for every other variant)
# run by the wrapper /tmp/q50_seed.sh (sequentially, outside the scheduler, while the scheduler was paused); the wrapper
# touches sched/q50_tr_delta_s4x.ok, so the jobs below are skipped once done and act as dependencies only
for s in (43, 44):
    if not os.environ.get('SKIP_SEEDS'):
        job(f'tr_delta_s{s}', f'{TR} --seed {s} --out ckpt/mq_delta_s{s} > logs/mq_delta_s{s}.log 2>&1 && {inf(f"ckpt/mq_delta_s{s}/model.pt", f"ckpt/mq_delta_s{s}", MQ)}', TRAIN_MEM, prio=3)
# ---------------- ablations of the two-sided reader: without on-policy walk states; without the event term
job('tr_delta_lat', f'{TR.replace("--objective delta", "--objective delta_lat")} --out ckpt/mq_delta_lat > logs/mq_delta_lat.log 2>&1 && {inf("ckpt/mq_delta_lat/model.pt", "ckpt/mq_delta_lat", MQ, "--objective delta_lat")}', TRAIN_MEM, prio=8)
job('tr_delta_rank', f'{TR.replace("--objective delta", "--objective delta_rank")} --out ckpt/mq_delta_rank > logs/mq_delta_rank.log 2>&1 && {inf("ckpt/mq_delta_rank/model.pt", "ckpt/mq_delta_rank", MQ, "--objective delta_rank")}', TRAIN_MEM, prio=9)
# recipe ablations of the two-sided reader (the main recipe is --roll walk --lattice top --avoid state):
#   delta_depth: on-policy depth capped at |G|+1 as in the one-sided recipe (trained first in this round; ckpt/mq_delta_depth)
#   delta_wl   : level-uniform lattice states (no terminal balancing)
#   delta_plain: roll-outs with the plain greedy walk (may cycle); decoded with the self-avoiding walk
job('tr_delta_wl', f'{TR} --lattice level --out ckpt/mq_delta_wl > logs/mq_delta_wl.log 2>&1 && {inf("ckpt/mq_delta_wl/model.pt", "ckpt/mq_delta_wl", MQ)}', TRAIN_MEM, prio=9)
job('tr_delta_plain', f'{TR} --avoid none --out ckpt/mq_delta_plain > logs/mq_delta_plain.log 2>&1 && {inf("ckpt/mq_delta_plain/model.pt", "ckpt/mq_delta_plain", MQ)}', TRAIN_MEM, prio=10)
# decoding ablations of the main model: plain walk and no-undo walk at inference
for A in ('none', 'undo'):
    job(f'infA{A}_delta', inf(D, f'ckpt/mq_delta_A{A}', MQ, f'--objective delta --avoid {A}'), INF_MEM, ['tr_delta'], prio=7)

# ---------------- HotpotQA / 2Wiki (same recipe as the one-sided reader: stage-1 reader of each dataset, 20k questions, 1 epoch)
for d, ds in [('hotpotqa', 'hotpotqa'), ('2wiki', '2wiki')]:
    job(f'tr_{d}_delta', f'{PY} code/train3.py --data {ds} --init ckpt/{d}_jbce/model.pt --epochs 1 --max_train 20000 --objective delta --out ckpt/{d}_delta > logs/{d}_delta.log 2>&1 && '
        f'{inf(f"ckpt/{d}_delta/model.pt", f"ckpt/{d}_delta", f"data/proc/{d}_dev.jsonl")} && {E} {d}_dev delta=ckpt/{d}_delta/pred_{d}_dev.pkl:order', TRAIN_MEM, prio=5)

# ---------------- Beam Retrieval retrained locally (official code; checkpoints of q40): predictions on the three dev sets
B = {'musique': 'ckpt/beam_musique/11-03-2026/beam_musique-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/checkpoint_best.pt',
     'hotpotqa': 'ckpt/beam_hotpotqa/11-05-2026/beam_hotpotqa-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/checkpoint_best.pt',
     '2wiki': 'ckpt/beam_2wiki/11-05-2026/beam_2wiki-seed42-bsz8-fp16True-lr1e-05-decay0.0-warm0.1-valbsz1/checkpoint_best.pt'}
for d, ck in B.items():
    job(f'beam_{d}', f'{PY} code/beam_pred.py {d} {ck} /data/yaokang/models/dv3/deberta-v3-base > logs/beam_pred_{d}.log 2>&1', BEAM_MEM, prio=4)
# beam_pred.py writes results/scores/{d}_beamL.pkl / {d}_beamK.pkl; the official predictions are kept as *_beam_official.pkl
job('eval_beam', ' && '.join(f'([ -f results/scores/{d}_dev_beam_official.pkl ] || cp results/scores/{d}_dev_beam.pkl results/scores/{d}_dev_beam_official.pkl) && cp results/scores/{d}_beamL.pkl results/scores/{d}_dev_beam.pkl && cp results/scores/{d}_beamK.pkl results/scores/{d}_dev_beamK.pkl && '
                             f'{E} {d}_dev beam=results/scores/{d}_dev_beam.pkl:sets beamk=results/scores/{d}_dev_beamK.pkl:sets beamoff=results/scores/{d}_dev_beam_official.pkl:sets' for d in B), 0, [f'beam_{d}' for d in B], prio=4)

# ---------------- equal-opportunity closed loops: the two-sided reader as selector and/or stopper
LOOPS = [('delta', 'adaptive'), ('delta', 'fixed'), ('delta-anscls', 'adaptive'), ('static-delta', 'adaptive')]
for d in DSS:
    for m, mode in LOOPS:
        job(f'loop_{d}_{m}_{mode}', f'bash code/loop_run.sh {d} {m} {mode}', INF_MEM, ['inf0_delta'], prio=3)
    job(f'loopeval_{d}', f'bash code/loop_eval.sh {d} > logs/loopeval50_{d}.txt 2>&1 && ' +
        ' && '.join(f'{PY} code/loop_signif.py {d} delta_fixed {b}' for b in ['static_fixed', 'chain_fixed', 'dense_fixed', 'lacc_fixed']) + ' && ' +
        ' && '.join(f'{PY} code/loop_signif.py {d} delta_adaptive {b}' for b in ['static_adaptive', 'static_fixed', 'lacc_adaptive', 'delta-anscls_adaptive', 'static-delta_adaptive']),
        0, [f'loop_{d}_{m}_{mode}' for m, mode in LOOPS], prio=3)

# ---------------- IRCoT (published open-corpus baseline, run locally with the same retriever and LLM)
# run by the wrapper /tmp/q50_ircot.sh outside the scheduler (marks sched/q50_ircot_<ds>.ok); SKIP_IRCOT=1 drops the jobs here
for d in DSS:
    if not os.environ.get('SKIP_IRCOT'):
        job(f'ircot_{d}', f'{PY} code/ircot.py {d} > logs/ircot_{d}.log 2>&1', 11000, prio=5, llm=True)

# ---------------- CPU evaluation needed before QA (sets of the open pools and of the self-terminated reader)
job('eval_open_delta', ' && '.join(f'{E} openpool_{d}_N20 delta=ckpt/mq_delta/pred_openpool_{d}_N20.pkl:order' for d in DSS), 0, ['inf0_delta'], prio=2)

# ---------------- answer accuracy, int4 Qwen3-8B (main reader) and int4 Qwen3-4B (second reader)
QA = f'{PY} code/llm_tools.py qamulti'
job('qa_mq', f'RTAG=qwen3_8b {QA} "musique_dev|delta:k2,k4,t512,k5" "musique_dev|delta_dec:dec" "musique_dev|delta_cal_dec:dec" '
    + ' '.join(f'"openpool_{d}_N20|delta:k4"' for d in DSS), LLM_MEM, ['eval_delta', 'eval_open_delta'], prio=4, llm=True)
job('qa_beam', f'RTAG=qwen3_8b {QA} "musique_dev|beam:k2,k4,t512,k5" "musique_dev|beamk:k2,k4,t512,k5" "musique_dev|beam_dec:dec" && '
    f'QA_N=2000 RTAG=qwen3_8b {QA} "hotpotqa_dev|beam:k2,t384,k5" "hotpotqa_dev|beamk:k2,t384" "2wiki_dev|beam:k2,t384,k5" "2wiki_dev|beamk:k2,t384"', LLM_MEM, ['eval_beam'], prio=5, llm=True)
for d in DSS:
    job(f'qa_loop_{d}', f'RTAG=qwen3_8b {QA} ' + ' '.join(f'"loop_{d}_{m}_{mode}_r2|{m}_{mode}:k4"' for m, mode in LOOPS), LLM_MEM, [f'loopeval_{d}'], prio=5, llm=True)
job('qa_xds', f'QA_N=2000 RTAG=qwen3_8b {QA} "hotpotqa_dev|delta:k2,t384" "2wiki_dev|delta:k2,t384"', LLM_MEM, ['tr_hotpotqa_delta', 'tr_2wiki_delta'], prio=6, llm=True)
L4 = 'RTAG=qwen3_4b READER=$HOME/models/Qwen3-4B-AWQ'
job('qa4_mq', f'{L4} {QA} "musique_dev|delta:k4" "musique_dev|beam:k4" "musique_dev|beamk:k4" ' + ' '.join(f'"loop_{d}_delta_fixed_r2|delta_fixed:k4" "loop_{d}_delta_adaptive_r2|delta_adaptive:k4"' for d in DSS),
    LLM_MEM, ['eval_delta', 'eval_beam'] + [f'loopeval_{d}' for d in DSS], prio=7, llm=True)

# ---------------- final CPU pass: every evaluation again, then the tables
J.append({'name': PFX + 'final50', 'cmd': 'bash code/final50.sh > logs/final50.log 2>&1', 'need': 0, 'deps': [j['name'] for j in J], 'prio': 99, 'llm': False, 'soft': True})
os.makedirs('sched', exist_ok=True)
json.dump(J, open('sched/q50_jobs.json', 'w'), indent=1)
print(len(J), 'jobs')
