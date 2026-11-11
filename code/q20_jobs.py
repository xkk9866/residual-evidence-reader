"""Writes sched/q20_jobs.json: the complete experiment queue of this round (run by code/sched.py on GPUs 6/7).
Main model: LACUNA with the terminal head (ckpt/mq_lacr, decoded with Bayes event decisions -> ckpt/mq_lacr_ev);
the verbalized-outcome variant (ckpt/mq_lact*) is kept as an analysis of the outcome parameterisation."""
import json, os

PY = '.venv/bin/python'
MQ = 'data/proc/musique_dev.jsonl,data/proc/musique_full_dev.jsonl'
TR = f'{PY} code/train3.py --data musique --init ckpt/mq_jbce/model.pt --epochs 1'
DSS = ['musique', 'hotpotqa', '2wikimultihopqa']
TT = '--term text'
J = []


def job(name, cmd, need, deps=(), prio=50, llm=False):
    J.append({'name': name, 'cmd': cmd, 'need': need, 'deps': list(deps), 'prio': prio, 'llm': llm})


def inf3(ck, out, data, extra=''):
    return f'{PY} code/train3.py {extra} --ckpt {ck} --out {out} --infer {data}'


# ---------------- outcome-hypothesis variant (analysis): seed 42 and seed 43 with every quarter checkpoint
job('tr_lact', f'{TR} {TT} --out ckpt/mq_lact > logs/mq_lact.log 2>&1 && {inf3("ckpt/mq_lact/model.pt", "ckpt/mq_lact", MQ, TT)}', 13500, prio=1)
job('tr_lact_s43', f'{TR} {TT} --seed 43 --save_all --out ckpt/mq_lact_s43 > logs/mq_lact_s43.log 2>&1 && {inf3("ckpt/mq_lact_s43/model.pt", "ckpt/mq_lact_s43", MQ, TT)}', 13500, prio=1)
for q in range(1, 5):
    c = f'ckpt/mq_lact_s43/model_q{q}.pt'
    job(f'qeval{q}', f'{inf3(c, f"ckpt/mq_lact_s43_q{q}", "data/proc/musique_dev.jsonl", TT)} && {inf3(c, f"ckpt/mq_lact_s43_q{q}", "data/proc/musique_full_dev.jsonl", TT + " --max_infer 1600")}',
        5000, ['tr_lact_s43'], prio=1)

# ---------------- main model (terminal head): new decisions, accuracy-latency, robustness
H = 'ckpt/mq_lacr/model.pt'
job('inf_lach', inf3(H, 'ckpt/mq_lacr_ev', MQ), 5000, prio=2)
OP = ','.join(f'data/proc/openpool_{d}_N20.jsonl' for d in DSS)
job('inf0_lac', inf3('ckpt/mq_lact/model.pt', 'ckpt/mq_lact', OP, TT), 5000, ['tr_lact'], prio=2)
job('inf0_lach', inf3(H, 'ckpt/mq_lacr_ev', OP), 5000, prio=2)
job('inf0_anscls', inf3('ckpt/mq_anscls/model.pt', 'ckpt/mq_anscls', OP, '--objective anscls'), 5000, prio=2)
for T in range(5):          # accuracy-latency: T completion steps = T+1 pool reads
    job(f'inf_steps{T}', inf3('ckpt/mq_lact/model.pt', f'ckpt/mq_lact_T{T}', 'data/proc/musique_dev.jsonl', f'{TT} --steps {T}') + f' > logs/mq_lact_T{T}.log 2>&1', 5000, ['tr_lact'], prio=6)
    job(f'infh_steps{T}', inf3(H, f'ckpt/mq_lacr_T{T}', 'data/proc/musique_dev.jsonl', f'--steps {T}') + f' > logs/mq_lacr_T{T}.log 2>&1', 5000, prio=3)
job('inf_L320', inf3('ckpt/mq_lact/model.pt', 'ckpt/mq_lact_L320', 'data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl', f'{TT} --doclen 320') +
    f' && {PY} code/train2.py --head bce --doclen 320 --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3_L320 --infer data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl', 7000, ['tr_lact'], prio=6)
job('infh_L320', inf3(H, 'ckpt/mq_lacr_L320', 'data/proc/musique_dev.jsonl,data/proc/openpool_musique_N20.jsonl', '--doclen 320'), 7000, prio=3)
N40 = ','.join(f'data/proc/openpool_{d}_N40.jsonl' for d in DSS)
job('pool40', ' && '.join(f'{PY} code/open_pipeline.py pool {d} 40' for d in DSS), 3000, prio=2)
job('inf40', inf3('ckpt/mq_lact/model.pt', 'ckpt/mq_lact', N40, TT) + f' && {PY} code/train2.py --head bce --ckpt ckpt/mq_jbce3/model.pt --out ckpt/mq_jbce3 --infer {N40}'
    f' && {PY} code/chain_ce.py --ckpt ckpt/mq_chain/model.pt --out ckpt/mq_chain --infer {N40}', 7500, ['tr_lact', 'pool40'], prio=6)
job('infh40', inf3(H, 'ckpt/mq_lacr_ev', N40), 7500, ['pool40'], prio=3)

# ---------------- second stage-1 reader (seed 43): LACUNA and the static +1 control retrained from it
job('tr_jbce_s43', f'{PY} code/train2.py --data musique --head bce --out ckpt/mq_jbce_s43 --epochs 2 --seed 43 > logs/mq_jbce_s43.log 2>&1', 13500, prio=4)
job('tr_lacr_f43', f'{PY} code/train3.py --data musique --init ckpt/mq_jbce_s43/model.pt --epochs 1 --seed 43 --out ckpt/mq_lacr_f43 > logs/mq_lacr_f43.log 2>&1 && '
    f'{inf3("ckpt/mq_lacr_f43/model.pt", "ckpt/mq_lacr_f43", MQ)}', 13500, ['tr_jbce_s43'], prio=4)
job('tr_jbce3_f43', f'{PY} code/train2.py --data musique --head bce --ckpt ckpt/mq_jbce_s43/model.pt --out ckpt/mq_jbce3_f43 --epochs 1 --seed 43 > logs/mq_jbce3_f43.log 2>&1 && '
    f'{PY} code/train2.py --head bce --ckpt ckpt/mq_jbce3_f43/model.pt --out ckpt/mq_jbce3_f43 --infer {MQ}', 13500, ['tr_jbce_s43'], prio=4)

# ---------------- equal-opportunity closed loops: every reader, same query construction, same rounds
LOOPS = [('lac', 'adaptive'), ('lac', 'fixed'), ('lach', 'adaptive'), ('static', 'adaptive'), ('static', 'fixed'),
         ('chain', 'adaptive'), ('chain', 'fixed'), ('dense', 'fixed')]
DEP0 = {'lac': ['inf0_lac'], 'lach': ['inf0_lach'], 'static': [], 'chain': [], 'dense': []}
for d in DSS:
    for m, mode in LOOPS:
        deps = DEP0[m] + (['inf0_anscls'] if m in ('static', 'chain') and mode == 'adaptive' else [])
        job(f'loop_{d}_{m}_{mode}', f'bash code/loop_run.sh {d} {m} {mode}', 5000, deps, prio=3)
    job(f'loopeval_{d}', f'bash code/loop_eval.sh {d}', 0, [f'loop_{d}_{m}_{mode}' for m, mode in LOOPS], prio=3)
    job(f'loop_{d}_lach_fixed', f'bash code/loop_run.sh {d} lach fixed', 5000, ['inf0_lach'], prio=2)
    job(f'loopeval2_{d}', f'bash code/loop_eval.sh {d}', 0, [f'loopeval_{d}', f'loop_{d}_lach_fixed'], prio=2)
ALL_LOOPS = LOOPS + [('lach', 'fixed')]

# ---------------- evaluation (CPU) needed by QA
E = 'code/eval_select.py'
job('eval_mq', f'{PY} {E} musique_dev lact=ckpt/mq_lact/pred_musique_dev.pkl:order lach=ckpt/mq_lacr_ev/pred_musique_dev.pkl:order && '
    f'{PY} code/ans_eval.py lact=ckpt/mq_lact/pred_musique_full_dev.pkl:order lach=ckpt/mq_lacr_ev/pred_musique_full_dev.pkl:order && {PY} code/halt_stats2.py', 0, ['tr_lact', 'inf_lach'], prio=2)
job('eval_open', f'{PY} {E} openpool_musique_N20 lact=ckpt/mq_lact/pred_openpool_musique_N20.pkl:order && '
    f'{PY} {E} openpool_hotpotqa_N20 lact=ckpt/mq_lact/pred_openpool_hotpotqa_N20.pkl:order && '
    f'{PY} {E} openpool_2wikimultihopqa_N20 lact=ckpt/mq_lact/pred_openpool_2wikimultihopqa_N20.pkl:order', 0, ['inf0_lac'], prio=2)
job('eval_halt', f'{PY} code/halt_stats2.py > /dev/null', 0, ['eval_mq'], prio=2)

# ---------------- answer accuracy: Qwen3-8B-AWQ (int4) and Llama-3.1-8B-Instruct-AWQ (int4); each job runs beside a training job
QW = 'RTAG=qwen3_8b'
LL = 'READER=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 RTAG=llama31_8b'
QA = f'{PY} code/llm_tools.py qamulti'
loop_specs = lambda d, L: ' '.join(f'"loop_{d}_{m}_{mode}_r2|{m}_{mode}:k4"' for m, mode in L)
job('qa_mq', f'{QW} {QA} "musique_dev|lach_dec:dec" "musique_dev|lach_cal_dec:dec" "musique_dev|lact_dec:dec" "musique_dev|lact_cal_dec:dec" '
    f'"musique_dev|jbce4tuned_dec:dec" "musique_dev|chaintuned_dec:dec" "musique_dev|lact:k2,k4,t512" "openpool_musique_N20|lact:k4" "openpool_hotpotqa_N20|lact:k4" "openpool_2wikimultihopqa_N20|lact:k4"',
    9500, ['eval_halt', 'eval_open'], prio=2, llm=True)
for d in DSS:
    job(f'qa_loop_{d}', f'{QW} {QA} {loop_specs(d, ALL_LOOPS)}', 9500, [f'loopeval2_{d}'], prio=2, llm=True)
L_SUB = [('static', 'fixed'), ('static', 'adaptive'), ('chain', 'fixed'), ('dense', 'fixed'), ('lach', 'fixed'), ('lach', 'adaptive')]
MQ_SETS = ['lacg:k4', 'jbce3:k4', 'jbce4:k4', 'chain:k4', 'setr:k4', 'rankgpt:k4', 'beam:k4', 'oracle:k4', 'fullpool:k4', 'lach_cal_dec:dec', 'jbce3tuned_dec:dec']
job('qa2_mq', f'{LL} {QA} ' + ' '.join(f'"musique_dev|{s}"' for s in MQ_SETS), 9500, ['eval_halt'], prio=3, llm=True)
for d in DSS:
    job(f'qa2_loop_{d}', f'{LL} {QA} "openpool_{d}_N20|lacg:k4" "openpool_{d}_N20|jbce3:k4" ' + loop_specs(d, L_SUB), 9500, [f'loopeval2_{d}'], prio=3, llm=True)
job('qa2_xds', f'QA_N=2000 {LL} {QA} "hotpotqa_dev|lacg:k2" "hotpotqa_dev|jbce:k2" "2wiki_dev|lacg:k2" "2wiki_dev|jbce:k2" "hotpotqa_dev|oracle:k2" "2wiki_dev|oracle:k2"',
    9500, [], prio=3, llm=True)

# ---------------- everything else is CPU: final evaluation once every GPU job has finished
J.append({'name': 'final20', 'cmd': 'bash code/final20.sh > logs/final20.log 2>&1', 'need': 0, 'deps': [j['name'] for j in J], 'prio': 99, 'llm': False, 'soft': True})
os.makedirs('sched', exist_ok=True)
json.dump(J, open('sched/q20_jobs.json', 'w'), indent=1)
print(len(J), 'jobs')
