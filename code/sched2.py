"""Memory-budgeted job scheduler for ONE shared GPU (default: GPU 2; override with SCHED_GPUS=2,3).

  python code/sched2.py JOBS.json

Each job: {"name", "need" (MiB at peak), "deps": [names], "cmd": bash string, "llm": bool, "prio": int,
"soft": bool (start once deps finished, whatever their outcome)}.
Admission: a job starts on GPU g only if  total(g) - foreign(g) - sum(need of my running jobs on g) >= need + MARGIN,
where foreign(g) is the memory held by every process that this scheduler did not start (other users AND the owner's
other projects: they are identified by process session id, not by user name). My jobs never over-commit the card.
Each job writes its exit code to sched/<name>.rc; a failed job is retried up to RETRY times (OOM caused by foreign
processes growing). State: sched/<name>.{run,ok,fail}; logs: logs/sched/. Re-running adopts live jobs, skips ok ones.
LLM jobs (vLLM) get GPU_UTIL = (foreign + need) / total, because vLLM budgets the whole card, foreign memory included.
"""
import json, os, subprocess, sys, time

GPUS = [int(x) for x in os.environ.get('SCHED_GPUS', '2').split(',')]
MARGIN, COOL, RETRY = int(os.environ.get('SCHED_MARGIN', '500')), 120, 3
# a job that was killed from outside (SIGTERM/SIGKILL: other projects on this shared server free "their" card by killing
# foreign processes) or that hit a CUDA OOM is not a failure of the job: it is retried without counting, up to HARD_RETRY
# attempts, and the card it was killed on is avoided for AVOID_S seconds
KILLED, HARD_RETRY, AVOID_S = (137, 143), 12, 2700
avoid = {}            # gpu -> time until which it is avoided
os.makedirs('sched', exist_ok=True); os.makedirs('logs/sched', exist_ok=True)
# single instance: a second scheduler on the same state directory would launch every waiting job twice
LOCK = 'sched/_sched2.lock'
if os.path.exists(LOCK):
    old = open(LOCK).read().strip()
    if old and os.path.exists(f'/proc/{old}'):
        print(f'another sched2.py (pid {old}) is running; exiting', file=sys.stderr); sys.exit(0)
open(LOCK, 'w').write(str(os.getpid()))
jobs = json.load(open(sys.argv[1]))
attempt = {j['name']: 0 for j in jobs}
hard = {j['name']: 0 for j in jobs}
running = {}          # name -> (gpu, need, sid)


OOM_MARKS = ('CUDA out of memory', 'No available memory for the cache blocks', 'Error in memory profiling', 'OutOfMemoryError')


def oom_in_log(n):
    """the scheduler's log of the job, plus any log file the job's command redirects into"""
    cmd = next((j['cmd'] for j in jobs if j['name'] == n), '')
    files = [f'logs/sched/{n}.log'] + [t for t in cmd.replace('>>', '>').split('>')[1:] for t in [t.strip().split()[0]] if t.startswith('logs/')]
    for f in files:
        try:
            tail = open(f, 'rb').read()[-30000:].decode('utf8', 'ignore')
        except OSError:
            continue
        if any(m in tail for m in OOM_MARKS):
            return True
    return False
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


def sid_of(pid):
    try:
        return int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[3])
    except Exception:
        return -1


def gpu_info():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    tot, used, uid = {}, {}, {}
    for line in q.strip().splitlines():
        i, u, t, m = [x.strip() for x in line.split(',')]
        if int(i) in GPUS:
            tot[int(i)] = int(t); used[int(i)] = int(m); uid[u] = int(i)
    mine = {g: 0 for g in GPUS}
    my_sids = {s for (_, _, s) in running.values()}
    a = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    for line in a.strip().splitlines():
        try:
            u, pid, mem = [x.strip() for x in line.split(',')]
        except ValueError:
            continue
        if u in uid and sid_of(pid) in my_sids:
            mine[uid[u]] += int(mem)
    foreign = {g: max(0, used[g] - mine[g]) for g in GPUS}
    return tot, foreign


# adopt jobs launched by an earlier run of this scheduler that are still alive
for j in jobs:
    n = j['name']
    if os.path.exists(f'sched/{n}.run') and state(n) == 'wait':
        g, pid = open(f'sched/{n}.run').read().split()
        if os.path.exists(f'/proc/{pid}') or os.path.exists(f'sched/{n}.rc'):
            running[n] = (int(g), j['need'], int(pid)); log(f'ADOPT {n} gpu{g} pid{pid}')
log(f'start with {len(jobs)} jobs on GPUs {GPUS}')
while True:
    for n in list(running):
        if os.path.exists(f'sched/{n}.rc'):
            rc = int(open(f'sched/{n}.rc').read().strip() or 1)
            g = running.pop(n)[0]
            os.remove(f'sched/{n}.rc')
            if os.path.exists(f'sched/{n}.run'):
                os.remove(f'sched/{n}.run')
            if rc == 0:
                open(f'sched/{n}.ok', 'w').write('0'); log(f'END {n} gpu{g} rc=0')
            elif (rc in KILLED or oom_in_log(n)) and hard[n] < HARD_RETRY:
                hard[n] += 1; avoid[g] = time.time() + AVOID_S
                log(f'KILLED/OOM {n} gpu{g} rc={rc} (external, not counted; gpu{g} avoided for {AVOID_S // 60} min) hard={hard[n]}')
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
        tot, foreign = gpu_info()
        for j in pend:
            avail = {g: tot[g] - foreign[g] - sum(nd for (gg, nd, _) in running.values() if gg == g) for g in GPUS}
            cand = [g for g in GPUS if time.time() - last[g] > COOL and avail[g] >= j['need'] + MARGIN and time.time() > avoid.get(g, 0)]
            if not cand:
                continue
            g = max(cand, key=lambda g: avail[g])
            n = j['name']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), HF_HUB_OFFLINE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            if j.get('llm'):
                env['GPU_UTIL'] = f"{min(0.97, (foreign[g] + sum(nd for (gg, nd, _) in running.values() if gg == g) + j['need']) / tot[g]):.2f}"
            wrapped = f"( {j['cmd']} ) > logs/sched/{n}.log 2>&1; echo $? > sched/{n}.rc"
            # start_new_session: the job (and every process it spawns, vLLM workers included) carries session id = its pid
            p = subprocess.Popen(['bash', '-c', wrapped], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            open(f'sched/{n}.run', 'w').write(f'{g} {p.pid}')
            running[n] = (g, j['need'], p.pid); last[g] = time.time()
            log(f"START {n} gpu{g} need={j['need']} avail={avail[g]} foreign={foreign[g]}")
            break          # one admission per round: let the memory of the new job appear before admitting more
    time.sleep(30)
