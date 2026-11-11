"""Complete the mainstream-baseline comparison on every dev set.

The published selectors from 2023--2025 that the main table uses are
bge-large-en-v1.5, bge-reranker-v2-m3, Qwen3-Reranker-4B, RankGPT,
SetR and Beam Retrieval. Dense, reranker, Qwen3-Reranker and Beam
Retrieval already have scores on HotpotQA and 2Wiki. RankGPT and SetR
were only run on MuSiQue. This queue fills those two datasets, evaluates
complete-evidence recall, then reads the selected sets with Qwen3-8B.

GPUs: SCHED_GPUS=0,1,2,3 only. Cards 4--7 are left untouched.
"""
import json

PY = '.venv/bin/python'
E = f'{PY} code/eval_select.py'
QA = f'{PY} code/llm_tools.py qamulti'
PFX = 'q52_'
J = []


def job(name, cmd, need, deps=(), prio=50, llm=False):
    J.append({'name': PFX + name, 'cmd': cmd, 'need': need, 'deps': [PFX + d for d in deps],
              'prio': prio, 'llm': llm, 'soft': False})


for ds in ('hotpotqa_dev', '2wiki_dev'):
    job(f'rank_{ds}', f'{PY} code/llm_tools.py rank {ds}', 16000, prio=1, llm=True)
    job(f'setr_{ds}', f'{PY} code/llm_tools.py setr {ds}', 16000, prio=2, llm=True)
    job(f'eval_{ds}',
        f'{E} {ds} '
        f'rankgpt=results/scores/{ds}_rankgpt_qwen3_8b.pkl:sets '
        f'setr=results/scores/{ds}_setr_qwen3_8b.pkl:sets',
        2000, deps=(f'rank_{ds}', f'setr_{ds}'), prio=3)
    job(f'qa_{ds}',
        f'RTAG=qwen3_8b {QA} "{ds}|rankgpt:k2,k4" "{ds}|setr:k2,k4"',
        16000, deps=(f'eval_{ds}',), prio=4, llm=True)

json.dump(J, open('sched/q52_jobs.json', 'w'), indent=1)
print(len(J), 'jobs -> sched/q52_jobs.json')
