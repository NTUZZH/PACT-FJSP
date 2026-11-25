r"""P85: alternative CP-SAT arms (constraint_family/cpsat_arms.py) on the
production-size family instances.

One record per (arm, regime, instance), in the format of the fair-warm
campaign (scripts/p12_cpsat_fairwarm.py, p44, p80) plus the arm's anytime
list, the timing evidence and the final schedule:

  test_results/FAMILY/<size>_cparms/<out>/<arm>/<regime>/<instance>.json
  test_results/FAMILY/<size>_cparms/<out>/<arm>/<regime>/<instance>.schedule.npz

so that timing_validity.classify_folders(root=<size>_cparms/<out>,
folders=[<arm>, ...]) reads them with its folder/regime/instance layout.

Hint (warm start). --hint rule (default) is the best-of-four dispatching-rule
schedule of the size-namespaced cache or_solution/FAMILY/warmstart_cache_<size>
(the comparator of the 30-minute campaign); --hint pact is the released
policy's schedule (pactwarm_cache_<size>); any other value is a directory
holding <regime>/<instance>.npz with arrays amch and ct. The hint must pass
both validators, or the solve is refused.

Timing. Every solve is a timing measurement and runs in a fresh child
process pinned to its slot's cores (--slots, one slot per parallel job;
default: the runner's own affinity mask). Each record carries
cpu_share = CPU seconds / (wall seconds x workers) over the whole arm (model
build included), the CPU seconds of the build alone, the affinity mask, the
load average before and after, and the peak memory. A model that needs more
memory than the host has free waits before its timed window starts; full
setup-regime models at 80x25 also take an exclusive lock.

Records are written atomically and only after the returned schedule passed
both validators (cpsat_arms.validate_all); a failure goes to _failures.log in
the run folder and writes no record. Existing records are skipped
(resumable); --force recomputes and overwrites them.

Usage:
  python scripts/p85_cpsat_arms.py --arm lns --size 50x25 \
      --regimes S SW LSW N --instances 0-2 --budget 600 --workers 8 \
      --jobs 2 --slots 8-15 16-23 --out pilot
  python scripts/p85_cpsat_arms.py --phase plan  ... (same arguments)
  python scripts/p85_cpsat_arms.py --phase size --size 50x25 --regimes S N \
      --instances 0 --arms ref compact [--presolve] --out sizes
"""
import argparse
import fcntl
import json
import multiprocessing as mp
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

from constraint_family import cpsat_arms as A  # noqa: E402

OR = 'or_solution/FAMILY'
SETUP_R = {'S', 'LS', 'SW', 'LSW'}
FULL_KINDS = {'ref', 'full'}


def parse_instances(spec):
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def parse_cores(spec):
    a, b = (spec.split('-') + [spec])[:2]
    return list(range(int(a), int(b) + 1))


def hint_path(hint, size, regime, name):
    if hint == 'rule':
        # the training-size cache predates the size namespacing (p12, p44)
        base = (f'{OR}/warmstart_cache' if size == '10x25'
                else f'{OR}/warmstart_cache_{size}')
    elif hint == 'pact':
        base = f'{OR}/pactwarm_cache_{size}'
    else:
        base = hint
    return f'{base}/{regime}/{name}.npz'


RESULTS = 'test_results/FAMILY'


def out_dir(size, out, arm, regime):
    return f'{RESULTS}/{size}_cparms/{out}/{arm}/{regime}'


def mem_available_gb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) / 2**20
    return 0.0


def mem_state():
    """(MemAvailable, swap in use) in GB, stored at the start and the end of a
    solve so that a solve hit by memory pressure can be told apart."""
    v = {}
    with open('/proc/meminfo') as f:
        for line in f:
            k, _, rest = line.partition(':')
            if k in ('MemAvailable', 'SwapTotal', 'SwapFree'):
                v[k] = int(rest.split()[0]) / 2**20
    return (round(v.get('MemAvailable', 0.0), 2),
            round(v.get('SwapTotal', 0.0) - v.get('SwapFree', 0.0), 2))


def code_commit():
    """Short commit of the working tree's HEAD (None outside a repository)."""
    import subprocess
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


PEER_MIN_GB = float(os.environ.get('PACT_PEER_MIN_GB', 30))
# pause after a large solve, so that a peer that polls this runner's state
# file sees the IDLE line before the next large solve announces itself
PEER_GAP_S = float(os.environ.get('PACT_PEER_GAP_S', 45))


def peer_hold():
    """True while another project on the host announces a short memory peak:
    the last line of the file named by PACT_PEER_MEMORY_STATE starts with one
    of the words of PACT_PEER_HOLD_WORDS (default: LOADING). Without the
    variable or the file there is never a hold."""
    path = os.environ.get('PACT_PEER_MEMORY_STATE')
    if not path:
        return False
    words = tuple(os.environ.get('PACT_PEER_HOLD_WORDS', 'LOADING').split())
    try:
        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError:
        return False
    # a NOTE line is a remark and does not change the state before it
    lines = [ln for ln in lines if not ln.startswith('NOTE')]
    return bool(lines) and lines[-1].startswith(words)


def announce(state):
    """Append this runner's memory state to the file named by
    PACT_MEMORY_STATE (last line = now), for other projects on the host.
    Written only around solves that wait for PACT_PEER_MIN_GB or more, so run
    such solves from one slot at a time. Without the variable nothing is
    written."""
    path = os.environ.get('PACT_MEMORY_STATE')
    if not path:
        return
    with open(path, 'a') as f:
        f.write(f"{state} | written {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def needs_gb(arm, size, regime):
    """Memory a solve waits for (outside its timed window). The current
    model peaked at 37 GB on 80x25 setup regimes and 22 GB on 50x25 with
    eight workers (train_log/cpsat_rerun.log); the LNS and the restrictions
    stay below 1 GB at 50x25 in development runs."""
    kind = A.ARMS[arm]['kind']
    big = kind in FULL_KINDS and arm not in ('fixassign', 'radius') \
        and regime in SETUP_R
    if big and size.startswith('80'):
        return 40.0, True
    if big and size.startswith('50'):
        return 24.0, False
    return 4.0, False


def build_tasks(a):
    tasks = []
    for regime in a.regimes:
        for idx in parse_instances(a.instances):
            name = f'instance_{idx:03d}'
            od = out_dir(a.size, a.out, a.arm, regime)
            rec = f'{od}/{name}.json'
            if os.path.exists(rec) and not a.force:
                continue
            tasks.append({'size': a.size, 'regime': regime, 'name': name,
                          'arm': a.arm, 'budget': a.budget,
                          'workers': a.workers, 'seed': a.seed,
                          'hint': a.hint, 'out': rec, 'log': a.solver_log,
                          'dev': a.dev,
                          'lns': json.loads(a.lns_options) if a.lns_options
                          else None})
    return tasks


def _fail(t, msg):
    root = os.path.dirname(os.path.dirname(os.path.dirname(t['out'])))
    os.makedirs(root, exist_ok=True)
    with open(f'{root}/_failures.log', 'a') as f:
        f.write(f"{time.strftime('%m-%d %H:%M:%S')} {t['arm']} {t['regime']} "
                f"{t['name']}: {msg}\n")


def solve_task(t, cores):
    """One solve in this (child) process; writes the record and schedule."""
    import resource
    if cores:
        os.sched_setaffinity(0, cores)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    case = A.load_case(t['size'], t['regime'], t['name'])
    hp = hint_path(t['hint'], t['size'], t['regime'], t['name'])
    if not os.path.exists(hp):
        _fail(t, f'no hint at {hp}')
        return
    hint, hmeta = A.load_hint_npz(hp, case)
    bad = A.validate_all(case, *hint)
    if bad:
        _fail(t, f'hint fails validation: {bad[:2]}')
        return
    load0 = [round(x, 2) for x in os.getloadavg()]
    mem0 = mem_state()
    log = [] if t['log'] else None
    w0 = time.time()
    try:
        res = A.run_arm(t['arm'], case, hint, t['budget'], t['workers'],
                        seed=t['seed'], log=log, lns_options=t['lns'])
    except Exception as e:                  # recorded, never silently dropped
        _fail(t, f'{type(e).__name__}: {e}')
        return
    wall = time.time() - w0
    amch, st, ct = res['schedule']
    viol = A.validate_all(case, amch, st, ct)
    if viol:
        _fail(t, f'returned schedule fails validation: {viol[:2]}')
        return
    od = os.path.dirname(t['out'])
    os.makedirs(od, exist_ok=True)
    sched_name = f"{t['name']}.schedule.npz"
    tmp = f'{od}/.{sched_name}.tmp.npz'
    np.savez(tmp, amch=amch, st=st, ct=ct, ms=float(np.max(ct)))
    os.replace(tmp, f'{od}/{sched_name}')
    import ortools
    rec = {
        'instance': t['name'], 'regime': t['regime'], 'size': t['size'],
        'arm': t['arm'], 'exactness': res['exactness'],
        'budget_s': t['budget'], 'workers': t['workers'], 'seed': t['seed'],
        'params': res.get('params'),
        'hint_source': t['hint'], 'hint_file': os.path.relpath(hp),
        'warm_rule': hmeta.get('rule'), 'warm_makespan': hmeta['makespan'],
        'warm_all_rules': hmeta.get('all_rules'),
        'status': res['status'], 'objective': res['objective'],
        'bound': res['bound'], 'model_bound': res.get('model_bound'),
        'from_warmstart': res['from_warmstart'],
        'solve_time': res['solve_s'], 'build_s': res['build_s'],
        'wall_total': round(wall, 2),
        'cpu_time': res['cpu_s'], 'cpu_build_s': res['cpu_build_s'],
        'cpu_per_wall': round(res['cpu_s'] / max(wall, 1e-9), 3),
        'cpu_share': round(res['cpu_s'] / max(wall * t['workers'], 1e-9), 3),
        'affinity': sorted(os.sched_getaffinity(0)),
        'loadavg': load0, 'loadavg_end': [round(x, 2) for x in os.getloadavg()],
        'peak_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                             / 2**20, 3),
        'mem_available_gb': [mem0[0], mem_state()[0]],
        'swap_used_gb': [mem0[1], mem_state()[1]],
        'started': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(w0)),
        'commit': t.get('commit'),
        'anytime': res['anytime'], 'hint': 'complete',
        'validated': True, 'violations': [],
        'schedule_file': sched_name,
        'ortools': ortools.__version__, 'python': platform.python_version(),
        'development_run': bool(t.get('dev')),
    }
    for k in ('lns_stats', 'n_arcs'):
        if k in res:
            rec[k] = res[k]
    if log is not None:
        rec['solver_log_head'] = log[:120]
    tmp = t['out'] + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, t['out'])


def worker(cores, q, lock_path):
    ctx = mp.get_context('fork')
    tag = f'{min(cores)}-{max(cores)}' if cores else 'inherit'
    while True:
        try:
            t = q.get(timeout=5)
        except Exception:
            return
        if os.path.exists(t['out']) and not t.get('force_run'):
            # another slot finished it meanwhile
            continue
        need, exclusive = needs_gb(t['arm'], t['size'], t['regime'])
        lock = None
        if exclusive:
            lock = open(lock_path, 'w')
            fcntl.flock(lock, fcntl.LOCK_EX)
        large = need >= PEER_MIN_GB
        waited = 0
        while mem_available_gb() < need or (large and peer_hold()):
            if waited % 600 == 0:
                print(f"[slot {tag}] {t['arm']} {t['regime']} {t['name']} waits: "
                      f"{mem_available_gb():.0f} GB available, {need:.0f} GB needed"
                      f"{', the peer is loading' if large and peer_hold() else ''}",
                      flush=True)
            time.sleep(20)
            waited += 20
        if large:
            # build, presolve and validation come on top of the budget
            end = time.strftime('%H:%M', time.localtime(time.time() + t['budget'] + 300))
            announce(f"BIG up to {need:.0f} GB, {t['arm']} {t['size']} "
                     f"{t['regime']} {t['name']}, until about {end}")
        t_start = time.time()
        p = ctx.Process(target=solve_task, args=(t, cores))
        p.start()
        p.join()
        if large:
            announce('IDLE no large solve')
            if os.environ.get('PACT_PEER_MEMORY_STATE'):
                time.sleep(PEER_GAP_S)
        if lock is not None:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
        fresh = os.path.exists(t['out']) and os.path.getmtime(t['out']) >= t_start
        status = 'ok' if fresh else f'NO RECORD (exit {p.exitcode}, see _failures.log)'
        if status == 'ok':
            r = json.load(open(t['out']))
            status = (f"{r['status']} obj={r['objective']} warm={r['warm_makespan']} "
                      f"share={r['cpu_share']}")
        print(f"[slot {tag}] {t['arm']} {t['regime']} {t['name']} {status}",
              flush=True)


def run(a):
    tasks = build_tasks(a)
    commit = code_commit()
    for t in tasks:
        t['force_run'] = a.force
        t['commit'] = commit
    slots = [parse_cores(s) for s in a.slots] if a.slots else \
        [sorted(os.sched_getaffinity(0))]
    if len(slots) < a.jobs:
        sys.exit(f'REFUSING: {a.jobs} jobs need {a.jobs} --slots')
    slots = slots[:a.jobs]
    used = [c for s in slots for c in s]
    if len(set(used)) != len(used):
        sys.exit('REFUSING: slots overlap')
    for s in slots:
        if a.workers > len(s) and not a.dev:
            sys.exit(f'REFUSING: {a.workers} workers on a {len(s)}-core slot '
                     f'(--dev allows it for development runs, which are not '
                     f'measurements)')
    print(f'{len(tasks)} solves of arm {a.arm} ({A.ARMS[a.arm]["exactness"]}), '
          f'budget {a.budget}s, {a.workers} workers, slots {a.slots or "inherit"}, '
          f'hint {a.hint}, out {a.out}', flush=True)
    if not tasks:
        return
    root = f'{RESULTS}/{a.size}_cparms/{a.out}'
    os.makedirs(root, exist_ok=True)
    q = mp.Queue()
    for t in tasks:
        q.put(t)
    time.sleep(0.5)
    procs = [mp.Process(target=worker, args=(s, q, f'{root}/_big.lock'))
             for s in slots]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print('done', flush=True)


def plan(a):
    tasks = build_tasks(a)
    h = sum(t['budget'] for t in tasks) / 3600
    print(f'{len(tasks)} solves of {a.arm}: {h:.2f} slot-hours of budget, '
          f'{h * a.workers:.1f} core-hours at {a.workers} workers')
    for t in tasks[:5]:
        print('  ', t['regime'], t['name'], '->', t['out'])


def size_phase(a):
    od = f'{RESULTS}/{a.size}_cparms/{a.out}/_model_size'
    os.makedirs(od, exist_ok=True)
    for regime in a.regimes:
        for idx in parse_instances(a.instances):
            name = f'instance_{idx:03d}'
            p = f'{od}/{regime}_{name}.json'
            if os.path.exists(p) and not a.force:
                continue
            case = A.load_case(a.size, regime, name)
            hint, _ = A.load_hint_npz(hint_path(a.hint, a.size, regime, name),
                                      case)
            rec = A.size_report(case, hint, arms=a.arms, presolve=a.presolve)
            rec['instance'], rec['regime'], rec['size'] = name, regime, a.size
            rec['n_ops'], rec['n_machines'] = case.N, case.M
            rec['affinity'] = sorted(os.sched_getaffinity(0))
            with open(p, 'w') as f:
                json.dump(rec, f, indent=1)
            print(regime, name, {k: (v['variables'], v['constraints'],
                                     v['circuit_arcs'], v['build_s'],
                                     v.get('presolve_s_1worker'))
                                 for k, v in rec.items() if isinstance(v, dict)},
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', default='solve', choices=['solve', 'plan', 'size'])
    ap.add_argument('--arm', choices=sorted(A.ARMS))
    ap.add_argument('--arms', nargs='+', default=['ref', 'compact'],
                    help='size phase: arms whose full model is measured')
    ap.add_argument('--size', required=True)
    ap.add_argument('--regimes', nargs='+',
                    default=['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW'])
    ap.add_argument('--instances', default='0-9')
    ap.add_argument('--budget', type=float, default=1800)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--slots', nargs='+', default=None,
                    help='one core range per parallel job, e.g. 8-15 16-23')
    ap.add_argument('--hint', default='rule')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--lns-options', default=None,
                    help='JSON dict overriding cpsat_arms.DEFAULT_LNS')
    ap.add_argument('--solver-log', action='store_true',
                    help='keep the head of the CP-SAT log in the record')
    ap.add_argument('--presolve', action='store_true',
                    help='size phase: also time a one-worker presolve')
    ap.add_argument('--out', required=True)
    ap.add_argument('--root', default=None,
                    help='results root (default test_results/FAMILY)')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dev', action='store_true',
                    help='allow more workers than slot cores; the records are '
                         'marked development_run and are not measurements')
    a = ap.parse_args()
    global RESULTS
    if a.root:
        RESULTS = os.path.abspath(a.root)
    if a.phase in ('solve', 'plan') and not a.arm:
        sys.exit('--arm is required')
    if a.phase == 'plan':
        plan(a)
    elif a.phase == 'size':
        size_phase(a)
    else:
        run(a)


if __name__ == '__main__':
    main()
