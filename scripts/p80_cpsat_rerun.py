"""P80: re-solve the CP-SAT cells of the regimes without setups with a COMPLETE
warm-start hint, and run the policy-seeded hybrid at the production batches.

WHY. Without setups the hint used to fix only the machine choices and the
processing starts. CP-SAT reports such a hint as incomplete, and its
hint-guided first incumbent can lie far above the warm start (80x25 N,
instance_000: 826 against 523). The search then never starts from the schedule
it was handed, and at the production batches the credited value was the warm
start itself on 64 of 80 solves. constraint_family/cpsat_family.py now hints
every model variable in that branch (verified: CP-SAT reports the hint complete
and feasible, and its first incumbent equals the warm-start makespan at 0.07 s
on 10x25 and 0.7 s on 80x25). The setup-bearing regimes already received a
hint that CP-SAT completes by propagation, so their cells are not re-solved.

WHAT IS RE-SOLVED. Every campaign folder whose cells feed the manuscript
(mapping of 2026-10-14): 10x25_fairwarm{1,2,5,10,30,300}s, 10x25_fairwarmR{5,30}s,
20x25 and 30x25_fairwarm{1,2,5,10,30,60}s, and 50x25/80x25_fairwarm1800s, on
N, L, W and LW. Exactly one thing changes, the hint completeness: the budget,
the worker count, the anytime flag and the warm-start source are read from the
superseded record of the same cell, and the warm-start makespan in the cache
must equal the one that record used, or the run refuses.

HYBRID. The released policy's own schedule (eval_family.py --save_schedules)
seeds CP-SAT at 600 s with the trajectory recorded, on the first 10 instances
per regime of 50x25 and 80x25, all eight regimes, into
or_solution/FAMILY/<size>_fairwarm600s_pactwarm. A second hybrid seeds CP-SAT
with the best of the three released checkpoints' schedules per instance
(p82_best3.py --cache), into <size>_fairwarm600s_pactbest3.

TIMING DISCIPLINE. Every solve is a timing measurement. The host has eight
performance cores (0-7) and sixteen efficiency cores (8-23). The superseded
campaigns ran two solves at a time, eight search workers each, under one
affinity mask of cores 0-15, so the default here is two slots that share that
same mask; the re-solved cells then run under the hardware condition of the
setup-bearing cells they are compared with. No other job may use cores 0-15
while the run is live. Each record carries
cpu_share = CPU seconds / (wall seconds x workers), the affinity mask and the
load average and its peak memory; --phase audit applies the validity rule of
timing_validity.py and moves invalid cells aside for re-solving. Each solve
runs in a fresh child process so its memory returns to the host, an 80x25
setup-regime model (about 30 GB) holds an exclusive lock, and every solve waits
for enough free memory before its timed window starts. A slot pauses before its next task while the file
or_solution/FAMILY/_rerun_control/STOP_<slot> exists.

Usage:
  python scripts/p80_cpsat_rerun.py --phase quarantine     # move superseded cells
  python scripts/p80_cpsat_rerun.py --phase run --slots 0-15 0-15
  python scripts/p80_cpsat_rerun.py --phase audit          # list or drop low-share cells
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

OR = 'or_solution/FAMILY'
QUAR = f'{OR}/_superseded_incomplete_hint'
CTRL = f'{OR}/_rerun_control'
LOG = 'train_log/cpsat_rerun.log'
NOSETUP = ['N', 'L', 'W', 'LW']
R8 = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']

# folder -> (data size, warm-start cache)
RERUN = {}
for b in (1, 2, 5, 10, 30, 300):
    RERUN[f'10x25_fairwarm{b}s'] = ('10x25', f'{OR}/warmstart_cache')
for b in (5, 30):
    RERUN[f'10x25_fairwarmR{b}s'] = ('10x25', f'{OR}/warmstart_cache_repaired')
for size in ('20x25', '30x25'):
    for b in (1, 2, 5, 10, 30, 60):
        RERUN[f'{size}_fairwarm{b}s'] = (size, f'{OR}/warmstart_cache_{size}')
for size in ('50x25', '80x25'):
    RERUN[f'{size}_fairwarm1800s'] = (size, f'{OR}/warmstart_cache_{size}')
HYBRID = {}
for s in ('50x25', '80x25'):
    HYBRID[f'{s}_fairwarm600s_pactwarm'] = (s, f'{OR}/pactwarm_cache_{s}')
    # the best of the three released checkpoints' schedules (p82_best3.py --cache)
    HYBRID[f'{s}_fairwarm600s_pactbest3'] = (s, f'{OR}/pactbest3_cache_{s}')
SETUP_R = {'S', 'LS', 'SW', 'LSW'}
BIG_GB, SMALL_GB = 36, 20      # MemAvailable a solve waits for before it starts
HYBRID_BUDGET, HYBRID_LIMIT = 600, 10


def log(msg):
    with open(LOG, 'a') as f:
        f.write(f'{time.strftime("%m-%d %H:%M:%S")} {msg}\n')


def quarantine():
    moved = 0
    for folder in RERUN:
        for r in NOSETUP:
            src = f'{OR}/{folder}/{r}'
            dst = f'{QUAR}/{folder}/{r}'
            if not os.path.isdir(src):
                sys.exit(f'REFUSING: {src} does not exist')
            if os.path.isdir(dst):
                if os.listdir(src):
                    sys.exit(f'REFUSING: both {src} and {dst} hold files')
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            os.makedirs(src)
            moved += len(os.listdir(dst))
    print(f'quarantined {moved} superseded cells under {QUAR}')


def build_tasks():
    tasks = []
    for folder, (size, cache) in RERUN.items():
        for r in NOSETUP:
            for old_p in sorted(glob.glob(f'{QUAR}/{folder}/{r}/*.json')):
                old = json.load(open(old_p))
                name = old['instance']
                out_p = f'{OR}/{folder}/{r}/{name}.json'
                if os.path.exists(out_p):
                    continue
                tasks.append(dict(folder=folder, size=size, regime=r, name=name,
                                  cache=cache, budget=int(old['budget_s']),
                                  workers=int(old['workers']),
                                  anytime='anytime' in old,
                                  old_warm=float(old['warm_makespan']),
                                  out=out_p))
    for folder, (size, cache) in HYBRID.items():
        for r in R8:
            descs = sorted(glob.glob(f'data/FAMILY/{size}/desc/{r}/*.desc.json'))[:HYBRID_LIMIT]
            for dp in descs:
                name = os.path.basename(dp).replace('.desc.json', '')
                out_p = f'{OR}/{folder}/{r}/{name}.json'
                if os.path.exists(out_p):
                    continue
                tasks.append(dict(folder=folder, size=size, regime=r, name=name,
                                  cache=cache, budget=HYBRID_BUDGET, workers=8,
                                  anytime=True, old_warm=None, out=out_p))
    # longest first, so the slots finish together
    tasks.sort(key=lambda t: -t['budget'])
    return tasks


def is_big(t):
    """An 80x25 model with setups takes about 30 GB; only one may run at a time."""
    return t['size'] == '80x25' and t['regime'] in SETUP_R


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return 0.0


def solve_task(t, slot_cores):
    """One solve in a fresh child process, so that its memory is returned to
    the host when it ends; annotates the record with the timing evidence."""
    import numpy as np
    import resource
    os.sched_setaffinity(0, slot_cores)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import p12_cpsat_fairwarm as fw
    os.makedirs(os.path.dirname(t['out']), exist_ok=True)
    fw.WARM_CACHE = t['cache']
    stem = f"data/PPVC/{t['size']}+ppvc-mixed/{t['name']}"
    dp = f"data/FAMILY/{t['size']}/desc/{t['regime']}/{t['name']}.desc.json"
    w0 = time.time()
    fw.solve_one((stem, dp, t['out'], t['budget'], t['workers'], t['anytime']))
    rec = json.load(open(t['out']))
    rec['cpu_share'] = round(rec['cpu_per_wall'] / t['workers'], 3)
    rec['affinity'] = sorted(os.sched_getaffinity(0))
    rec['loadavg'] = [round(x, 2) for x in os.getloadavg()]
    rec['hint'] = 'complete'
    rec['wall_total'] = round(time.time() - w0, 2)
    rec['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)
    with open(t['out'] + '.tmp', 'w') as f:
        json.dump(rec, f)
    os.replace(t['out'] + '.tmp', t['out'])


def worker(slot_cores, queues):
    """Takes tasks from the first non-empty queue in its own priority order.
    An 80x25 setup-regime solve holds an exclusive lock, and every solve waits
    for enough free host memory before it starts (outside its timed window)."""
    import fcntl
    import numpy as np
    os.sched_setaffinity(0, slot_cores)
    tag = f'{min(slot_cores)}-{max(slot_cores)}'
    ctx = mp.get_context('fork')
    while True:
        t = None
        for q in queues:
            try:      # a timeout, not get_nowait: a fast run of skipped cells can drain the
                t = q.get(timeout=30)      # pipe before the feeder thread refills it
                break
            except Exception:
                continue
        if t is None:
            log(f'slot {tag} queue empty, exiting')
            return
        while os.path.exists(f'{CTRL}/STOP_{tag}'):
            time.sleep(30)
        if os.path.exists(t['out']):
            continue
        cp = f"{t['cache']}/{t['regime']}/{t['name']}.npz"
        while not os.path.exists(cp):      # the hybrid cache is built separately
            time.sleep(60)
        z = np.load(cp, allow_pickle=True)
        if t['old_warm'] is not None and abs(float(z['ms']) - t['old_warm']) > 1e-6:
            log(f'REFUSED {t["folder"]}/{t["regime"]}/{t["name"]}: cache warm '
                f'{float(z["ms"])} != superseded record warm {t["old_warm"]}')
            continue
        lock = None
        if is_big(t):
            lock = open(f'{CTRL}/big.lock', 'w')
            fcntl.flock(lock, fcntl.LOCK_EX)
        need = BIG_GB if is_big(t) else SMALL_GB
        while mem_available_gb() < need:
            time.sleep(30)
        p = ctx.Process(target=solve_task, args=(t, slot_cores))
        p.start()
        p.join()
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        if p.exitcode != 0 or not os.path.exists(t['out']):
            log(f'slot {tag} FAILED {t["folder"]}/{t["regime"]}/{t["name"]} exit={p.exitcode}')
            continue
        rec = json.load(open(t['out']))
        log(f'slot {tag} {t["folder"]}/{t["regime"]}/{t["name"]} {rec["status"]} '
            f'obj={rec["objective"]} warm={rec["warm_makespan"]} '
            f'share={rec["cpu_share"]} load={rec["loadavg"][0]} rss={rec["peak_rss_gb"]}')


def run(slots, order='longest'):
    tasks = build_tasks()
    if order == 'shortest':      # a second launcher drains the queue from the other end
        tasks.reverse()
    hours = sum(t['budget'] for t in tasks) / 3600
    log(f'{len(tasks)} solves queued, {hours:.1f} slot-hours of budget, slots {slots}')
    print(f'{len(tasks)} solves queued, {hours:.1f} slot-hours of budget')
    q_big, q_small = mp.Queue(), mp.Queue()
    for t in tasks:
        (q_big if is_big(t) else q_small).put(t)
    time.sleep(1)
    procs = []
    for k, s in enumerate(slots):
        a, b = map(int, s.split('-'))
        # the first slot works through the large models first, the others last
        qs = [q_big, q_small] if k == 0 else [q_small, q_big]
        p = mp.Process(target=worker, args=(list(range(a, b + 1)), qs))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    log('ALL DONE')


def audit(drop):
    """Apply the validity rule of timing_validity.py; --drop moves invalid
    cells to _superseded_share_outlier/ so the run phase re-solves them."""
    from timing_validity import classify_folders
    res = classify_folders(OR, list(RERUN) + list(HYBRID))
    pending = [p for p, (ok, _) in res.items() if ok is None]
    res = {p: v for p, v in res.items() if v[0] is not None}
    if pending:
        print(f'{len(pending)} cells pending (their type bar is not frozen yet)')
    bad = [(p, r) for p, (ok, r) in res.items() if not ok]
    from collections import Counter
    print(f'{len(res)} re-solved cells, {len(bad)} invalid; valid by',
          dict(Counter(r for ok, r in res.values() if ok)))
    for p, r in bad[:50]:
        print('  ', p, r)
    if not bad:
        # the rule's constants and the counts, for the solver-isolation paragraph
        from macro_io import emit
        from timing_validity import SHARE_MIN, TYPE_SLACK
        c = Counter(r for ok, r in res.values() if ok)
        emit('paper/macros_timing.tex', [
            '% auto-generated by scripts/p80_cpsat_rerun.py --phase audit -- do not hand-edit',
            ('CpuShareMin', f'{SHARE_MIN:.1f}'),
            ('CpuShareTol', f'{TYPE_SLACK:.2f}'),
            ('CpuSharePct', '5'),
            ('CpReservedCores', '16'),
            ('CpuShareCells', f'{len(res):,}'.replace(',', '{,}')),
            ('CpuShareByShare', f'{c["share"]:,}'.replace(',', '{,}')),
            ('CpuShareByOpt', str(c['optimal'])),
            ('CpuShareByType', str(c['type'])),
        ])
    if drop:
        for p, _ in bad:
            dst = p.replace(f'{OR}/', f'{OR}/_superseded_share_outlier/')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(p, dst)
        print(f'moved {len(bad)} to _superseded_share_outlier/; re-run --phase run to re-solve them')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True, choices=['quarantine', 'run', 'audit', 'plan'])
    ap.add_argument('--slots', nargs='+', default=['0-15', '0-15'])
    ap.add_argument('--drop', action='store_true')
    ap.add_argument('--order', choices=['longest', 'shortest'], default='longest')
    a = ap.parse_args()
    if a.phase == 'quarantine':
        quarantine()
    elif a.phase == 'run':
        os.makedirs(CTRL, exist_ok=True)
        run(a.slots, a.order)
    elif a.phase == 'plan':
        ts = build_tasks()
        from collections import Counter
        c = Counter((t['folder'], t['budget'], t['anytime']) for t in ts)
        for k, v in sorted(c.items()):
            print(k, v)
        print(len(ts), 'solves', round(sum(t['budget'] for t in ts) / 3600, 1), 'slot-hours')
    else:
        audit(a.drop)
