"""Memory-budgeted job scheduler for GPUs 6 and 7 (shared with other users).

  python code/sched.py JOBS.json

Each job: {"name", "need" (MiB of GPU memory at peak), "deps": [names], "cmd": bash string, "llm": bool, "prio": int,
"soft": bool (start once deps finished, whatever their outcome)}.
Admission: a job starts on GPU g only if  total(g) - other_users(g) - sum(need of my running jobs on g) >= need + MARGIN,
where other_users(g) is the memory held by processes of other users (nvidia-smi compute apps). So my jobs never
over-commit a card, whatever their momentary usage. Each job records its own exit code in sched/<name>.rc; a failed job
is retried up to RETRY times (OOM caused by other users growing). State: sched/<name>.{run,ok,fail}; logs: logs/sched/.
Re-running the scheduler adopts jobs still marked .run whose process is alive, and skips jobs marked ok.
"""
import json, os, subprocess, sys, time, getpass

GPUS = [6, 7]
MARGIN, COOL, RETRY = 600, 90, 2
ME = getpass.getuser()
os.makedirs('sched', exist_ok=True); os.makedirs('logs/sched', exist_ok=True)
jobs = json.load(open(sys.argv[1]))
attempt = {j['name']: 0 for j in jobs}
running = {}          # name -> (gpu, need)
last = {g: 0.0 for g in GPUS}


def log(msg):
    with open('logs/sched/_sched.log', 'a') as f:
        f.write(time.strftime('%m-%d %H:%M:%S ') + msg + '\n')


def state(n):
    if os.path.exists(f'sched/{n}.ok'):
        return 'ok'
    if os.path.exists(f'sched/{n}.fail'):
        return 'fail'
    return 'run' if n in running else 'wait'


def gpu_info():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    tot, uid = {}, {}
    for line in q.strip().splitlines():
        i, u, t = [x.strip() for x in line.split(',')]
        if int(i) in GPUS:
            tot[int(i)] = int(t); uid[u] = int(i)
    other = {g: 0 for g in GPUS}
    a = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    for line in a.strip().splitlines():
        try:
            u, pid, mem = [x.strip() for x in line.split(',')]
        except ValueError:
            continue
        if u not in uid:
            continue
        user = subprocess.run(['ps', '-o', 'user=', '-p', pid], capture_output=True, text=True).stdout.strip()
        if user != ME:
            other[uid[u]] += int(mem)
    return tot, other


# adopt jobs launched by an earlier run of this scheduler that are still alive
for j in jobs:
    n = j['name']
    if os.path.exists(f'sched/{n}.run') and state(n) == 'wait':
        g, pid = open(f'sched/{n}.run').read().split()
        if os.path.exists(f'/proc/{pid}') or os.path.exists(f'sched/{n}.rc'):
            running[n] = (int(g), j['need']); log(f'ADOPT {n} gpu{g} pid{pid}')
log(f'start with {len(jobs)} jobs')
while True:
    for n in list(running):
        if os.path.exists(f'sched/{n}.rc'):
            rc = int(open(f'sched/{n}.rc').read().strip() or 1)
            g = running.pop(n)[0]
            os.remove(f'sched/{n}.rc')
            if rc == 0:
                open(f'sched/{n}.ok', 'w').write('0'); log(f'END {n} gpu{g} rc=0')
            elif attempt[n] < RETRY:
                attempt[n] += 1; log(f'RETRY {n} gpu{g} rc={rc} attempt={attempt[n]}')
            else:
                open(f'sched/{n}.fail', 'w').write(str(rc)); log(f'FAIL {n} gpu{g} rc={rc}')
    for j in jobs:
        if state(j['name']) == 'wait' and not j.get('soft') and any(state(d) == 'fail' for d in j.get('deps', [])):
            open(f"sched/{j['name']}.fail", 'w').write('dep'); log(f"SKIP {j['name']} (dependency failed)")
    ready = lambda j: all(state(d) in (('ok', 'fail') if j.get('soft') else ('ok',)) for d in j.get('deps', []))
    pend = sorted([j for j in jobs if state(j['name']) == 'wait' and ready(j)], key=lambda j: j.get('prio', 50))
    if not pend and not running and all(state(j['name']) in ('ok', 'fail') for j in jobs):
        log('ALL DONE'); open('sched/ALL_DONE', 'w').write('1'); break
    if pend:
        tot, other = gpu_info()
        for j in pend:
            avail = {g: tot[g] - other[g] - sum(nd for (gg, nd) in running.values() if gg == g) for g in GPUS}
            cand = [g for g in GPUS if time.time() - last[g] > COOL and avail[g] >= j['need'] + MARGIN]
            if not cand:
                continue
            g = max(cand, key=lambda g: avail[g])
            n = j['name']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            if j.get('llm'):
                env['GPU_UTIL'] = f"{min(0.9, j['need'] / tot[g]):.2f}"
            wrapped = f"( {j['cmd']} ) > logs/sched/{n}.log 2>&1; echo $? > sched/{n}.rc"
            p = subprocess.Popen(['setsid', 'bash', '-c', wrapped], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            open(f'sched/{n}.run', 'w').write(f'{g} {p.pid}')
            running[n] = (g, j['need']); last[g] = time.time()
            log(f"START {n} gpu{g} need={j['need']} avail={avail[g]}")
    time.sleep(30)
