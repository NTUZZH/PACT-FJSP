r"""P136: the memetic algorithm on the public Brandimarte instances without rules, against the
best-known makespans.

QUESTION
  The memetic algorithm of the timed comparison is the authors' implementation
  (eval_memetic_family.py, baselines/memetic). This check runs it on the ten Brandimarte
  instances Mk01 to Mk10 as published, where best-known makespans exist, and reports the gap
  of its makespan to the best-known value at 60 s and 600 s of search.

INSTANCES
  data/PUBLIC/family/pub_Brandimarte/desc/N/<name>.desc.json, regime N of the public family:
  no rule is active (the descriptor's 'active' is all false and the memetic Instance has no
  lag, release date, setup or outage; both checked for every cell), so the problem is the
  instance as published. Base instance data/PUBLIC/Brandimarte/<name>.fjs through
  constraint_family.public_adapter.load_base_instance; the descriptor through
  ConstraintDescriptor.from_json(setup_convention='attached'), as p90 reads the public family.
  Best-known makespans: column best_known_makespan of the table arm_registry.PUB_SETS
  ['pub_Brandimarte'] (data/PUBLIC/Brandimarte/best_known.csv), the table the public
  evaluation reads (p89 check_best_known). A run below a best-known value gives a negative
  gap, reported as it is.

METHOD (the protocol of eval_memetic_family.solve_one, by import of baselines.memetic)
  1. load the instance and its descriptor;
  2. start schedule, --start rule (default, the start of the campaign arm memetic_rule): the
     four masked dispatching rules FIFO, MOR, SPT, MWKR rolled out by
     p86_extend_scale_sets.masked_rollout (the procedure of the rule cache of the timed
     campaign), each validated with baselines.memetic.validate.validate; the valid rule with
     the lowest makespan, equal makespans to the earlier rule, enters the initial population
     as the one seed schedule. --start none: no start schedule;
  3. the numba kernels compiled or loaded on the instance (as solve_one);
  4. baselines.memetic.search.Memetic with DEFAULT_CONFIG, seed k (1 to 3), one thread, for
     --budget seconds of wall clock (default 600); steps 1 to 3 come before the timed window;
  5. the returned schedule validated; a schedule that fails is never written.
  Readings: the incumbent at 60 s and 600 s of the search window
  (baselines.memetic.records.incumbent_at, wall clock, origin search) of one run of 600 s.

CELLS AND RUN
  One cell = one instance and one seed: 10 x 3 = 30 cells. `run` takes the cores of its
  affinity mask (set by taskset; e.g. ten cores) and runs the cells in batches of one
  single-thread process per core, each process pinned to its own core (NUMBA, OMP and MKL
  threads 1, no GPU), as a memetic batch of the timed campaign runs. Before the first batch
  the cores are checked as the timed campaign's prelaunch checks them
  (timed_campaign_plan.busy_on), and before every batch with p75_deploy_cost.busy_processes;
  a process above 5% of a core on them refuses the batch. The stop file
  train_log/MEMPUB_HOLD ends the run after the current batch. Launch notes go to
  train_log/memetic_public_launch_notes.txt.
  Record: or_solution/PUBFAMILY/memetic_p136/pub_Brandimarte_N_<budget>s_<start>/
  <name>_seed<k>.json, the fields of eval_memetic_family.py's records (anytime in its
  four-column format, search_start_wall_s, cpu_share = CPU s / wall s of the timed window,
  threads 1, affinity, load average), plus best_known, idle_before, development_run, valid and
  its reason. Written atomically and never overwritten; an invalid record moves to invalid/
  and its cell runs again at the next start.
  Validity: timing_validity.classify_single_thread (validated, wall budget clock, one thread,
  affinity one core of the run's cores, stored as run_cores, cpu_share >= 0.9), the cores idle
  before the batch, not a development run. plan and infer classify every record again.

ANALYSIS (infer)
  Per instance and seed: gap = 100 (makespan - best known) / best known at 60 s and 600 s,
  and of the start schedule; per instance the mean and the smallest gap over the seeds; the
  mean over the instances of the per-instance mean gap; the number of instances on which a
  seed reaches the best-known value. Outputs (arm_registry.out under PACT_OUT; the full run
  needs every cell valid): results/memetic_public_check.json, results/memetic_public_check.md,
  manuscript/macros_memetic_public.tex (macro_io.emit, names Mpc plus letters, one comment per
  macro, the prefix Mpc used by no other .tex file under manuscript/). --out DIR (outside the
  repository's paper, result and record folders) writes them under DIR, marked partial when a
  cell is missing.

Usage (repository root):
  python scripts/p136_memetic_public_check.py plan
  taskset -c 8-17 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES= python scripts/p136_memetic_public_check.py run
  PACT_OUT=out/exact python scripts/p136_memetic_public_check.py infer [--out DIR]
  Development (a shared core, a short budget): run --development --root <folder outside the
  repository> --budget 3 --instances Mk01 --seeds 1
"""
import time

_T_MODULE = time.monotonic()

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
if __name__ == '__main__':
    for _v in THREAD_VARS:
        os.environ[_v] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
_CWD = os.getcwd()
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]           # params.py parses sys.argv at import time

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import timed_campaign_plan as TCP  # noqa: E402
import timing_validity as TV  # noqa: E402

GENERATOR = 'scripts/p136_memetic_public_check.py'
SET = 'pub_Brandimarte'
FAMILY = os.path.join('data', 'PUBLIC', 'family', SET)
REGIME = 'N'
NAMES = tuple(f'Mk{k:02d}' for k in range(1, 11))
SEEDS = (1, 2, 3)
DEFAULT_BUDGET = 600.0
READINGS = (60, 600)
STARTS = ('rule', 'none')
RULE_ORDER = ('FIFO', 'MOR', 'SPT', 'MWKR')     # p86_extend_scale_sets.RULE_ORDER, checked
OUT = os.path.join('or_solution', 'PUBFAMILY', 'memetic_p136')
HOLD = os.path.join('train_log', 'MEMPUB_HOLD')
NOTES = os.path.join('train_log', 'memetic_public_launch_notes.txt')
OVERHEAD_S = 5.0          # per batch beyond the budget: core sample, load, rules, numba
PREFIX = 'Mpc'
OUT_FILES = {'macros': 'manuscript/macros_memetic_public.tex',
             'json': 'results/memetic_public_check.json',
             'md': 'results/memetic_public_check.md'}
METHOD = ('memetic algorithm (baselines/memetic: GA over operation sequence and machine '
          'assignment, offspring improved by critical-path tabu search), DEFAULT_CONFIG, one '
          'thread')


def rel(p):
    a = os.path.abspath(p)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else a


def folder(root, budget, start):
    return os.path.join(root, OUT, f'{SET}_{REGIME}_{TCP.bnum(budget)}s_{start}')


def record_path(root, budget, start, name, seed):
    return os.path.join(folder(root, budget, start), f'{name}_seed{int(seed)}.json')


def best_known():
    """{name: best-known makespan} from the table the public evaluation reads."""
    path = os.path.join(ROOT, REG.PUB_SETS[SET])
    with open(path) as f:
        rows = {r['instance']: r for r in csv.DictReader(f)}
    miss = [x for x in NAMES if x not in rows or not rows[x]['best_known_makespan']]
    if miss:
        raise SystemExit(f'REFUSING: {REG.PUB_SETS[SET]} has no best-known value for {miss}')
    return {x: float(rows[x]['best_known_makespan']) for x in NAMES}


def descriptor_path(name):
    return os.path.join(ROOT, FAMILY, 'desc', REGIME, f'{name}.desc.json')


def load_cell(name):
    """(job lengths, processing times, descriptor, memetic Instance) of regime N, refused
    when a rule is active."""
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.public_adapter import load_base_instance
    from baselines.memetic.instance import Instance
    with open(descriptor_path(name)) as f:
        drec = json.load(f)
    if drec.get('regime') != REGIME or any(drec.get('active') or []):
        raise SystemExit(f'REFUSING: {rel(descriptor_path(name))} has regime '
                         f'{drec.get("regime")} and active rules {drec.get("active")}')
    desc = ConstraintDescriptor.from_json(drec, setup_convention='attached')
    jl, pt, _ = load_base_instance(os.path.join(ROOT, drec['base_instance']))
    inst = Instance(jl, pt, desc)
    if int(inst.lag.sum()) or int(inst.rel_op.sum()) or int(np.asarray(inst.S).sum()) \
            or int(np.asarray(inst.nw).sum()):
        raise SystemExit(f'REFUSING: {name}: the instance carries lags, release dates, setups '
                         'or outages')
    return jl, pt, desc, inst, drec


def rule_start(jl, pt, desc, inst):
    """The best of the four masked rules (module docstring, METHOD 2): (amch, ct, rule, per-rule
    makespans), the makespans of invalid rules None."""
    import p86_extend_scale_sets as p86
    from baselines.memetic.validate import validate
    if tuple(p86.RULE_ORDER) != RULE_ORDER:
        raise SystemExit(f'REFUSING: p86 rule order {p86.RULE_ORDER}, expected {RULE_ORDER}')
    best, per = None, {}
    for rule in RULE_ORDER:
        amch, ct, ms = p86.masked_rollout(rule, jl, pt, desc)
        if not np.all(ct == np.round(ct)):
            per[rule] = None
            continue
        ct = np.round(ct).astype(np.int64)
        amch = np.asarray(amch, dtype=np.int64)
        st = ct - inst.pt[np.arange(inst.N), amch]
        ok, _, vms = validate(inst, amch, st, ct)
        per[rule] = float(vms) if ok else None
        if ok and (best is None or vms < best[3]):
            best = (amch, ct, rule, float(vms))
    if best is None:
        raise SystemExit('REFUSING: no valid rule schedule')
    return best[0], best[1], best[2], per


def solve_cell(a, name, seed, core, idle, out_path):
    """The process of one cell (module docstring, METHOD); writes the record."""
    origin = (time.perf_counter(), time.process_time())
    os.sched_setaffinity(0, {core})
    os.chdir(ROOT)
    from params import configs
    configs.device = 'cpu'
    from baselines.memetic.search import Memetic
    from baselines.memetic.validate import validate
    from baselines.memetic.records import incumbent_at
    jl, pt, desc, inst, drec = load_cell(name)
    t = time.perf_counter()
    seeds, start = [], None
    if a.start == 'rule':
        amch, ct, rule, per = rule_start(jl, pt, desc, inst)
        seeds.append((amch, ct))
        start = dict(source='best of the four masked rules', rule=rule, makespan=per[rule],
                     rule_makespans=per)
    t_seed = time.perf_counter() - t
    t = time.perf_counter()
    Memetic(inst, seed=0, config=dict(pop_size=2, n_offspring=1),
            max_evals=3000).run(seeds[:1])
    w = Memetic(inst, seed=0, max_evals=10)
    w._load_chromosome(inst.base_os.copy(), inst.min_pt_mach.copy())
    t_jit = time.perf_counter() - t
    mm = Memetic(inst, seed=int(seed), budget_s=float(a.budget), clock='wall', origin=origin)
    aff = sorted(os.sched_getaffinity(0))
    load0 = os.getloadavg()
    cpu0, w0 = time.process_time(), time.perf_counter()
    res = mm.run(seeds)
    wall, cpu = time.perf_counter() - w0, time.process_time() - cpu0
    load1 = os.getloadavg()
    ok, viol, vms = validate(inst, res['op_mch'], res['op_st'], res['op_ct'])
    if not ok or int(vms) != res['makespan']:
        print(f'REFUSING to write {rel(out_path)}: schedule fails validation ({len(viol)} '
              f'violations; first {viol[:1]}; makespan {vms} vs {res["makespan"]})', flush=True)
        return 2
    import numba
    bk = best_known()[name]
    rec = {
        'set': SET, 'instance': name, 'regime': REGIME, 'base_instance': drec['base_instance'],
        'seed': int(seed), 'start': a.start, 'start_schedule': start,
        'budget_s': float(a.budget), 'budget_clock': 'wall', 'method': METHOD,
        'config': mm.cfg, 'makespan': res['makespan'],
        'anytime_columns': ['search_s', 'wall_s_from_process_start',
                            'cpu_s_from_process_start', 'makespan'],
        'anytime': res['anytime'],
        'search_start_wall_s': res['search_start_wall_s'],
        'search_start_cpu_s': res['search_start_cpu_s'],
        'n_evaluations': res['n_evaluations'], 'n_generations': res['n_generations'],
        'n_restarts': res['n_restarts'],
        'wall_s': round(wall, 3), 'cpu_s': round(cpu, 3), 'threads': 1, 'workers': 1,
        'cpu_share': round(cpu / max(wall, 1e-9), 4), 'affinity': aff, 'core': core,
        'run_cores': list(a.cores_allowed),
        'loadavg_start': [round(x, 2) for x in load0], 'loadavg_end': [round(x, 2) for x in load1],
        'idle_before': idle, 'development_run': bool(a.development),
        'seed_time_s': round(t_seed, 3), 'jit_time_s': round(t_jit, 3),
        'validated': True, 'violations': [],
        'schedule': {'op_mch': [int(v) for v in res['op_mch']],
                     'op_st': [int(v) for v in res['op_st']],
                     'op_ct': [int(v) for v in res['op_ct']]},
        'best_known': bk, 'best_known_table': REG.PUB_SETS[SET],
        'readings': {str(r): incumbent_at({'anytime': res['anytime'],
                                           'search_start_wall_s': res['search_start_wall_s']},
                                          float(r), 'wall', 'search')
                     for r in READINGS if r <= float(a.budget) + 1e-9},
        'numba': numba.__version__, 'commit': a._commit,
        'started': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    rec['valid'], rec['valid_reason'] = classify(rec)
    if os.path.exists(out_path):
        print(f'NOT WRITTEN: {rel(out_path)} exists', flush=True)
        return 2
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    try:
        os.link(tmp, out_path)           # fails when the record exists
    finally:
        os.unlink(tmp)
    print(f'{name} seed {seed} core {core}: makespan {res["makespan"]} (best known {bk:g}), '
          f'readings {rec["readings"]}, cpu_share {rec["cpu_share"]}, '
          f'{"valid" if rec["valid"] else "INVALID"} ({rec["valid_reason"]})', flush=True)
    return 0


def classify(rec, cores=None):
    """(valid, reason): timing_validity.classify_single_thread with the cores of the run (the
    record's run_cores unless cores is given), the cores idle before the batch, not a
    development run."""
    import tempfile
    if rec.get('development_run', True):
        return False, 'development run'
    idle = rec.get('idle_before') or {}
    if not idle.get('idle'):
        return False, f'cores not idle before the batch: {idle.get("busy_processes", [])[:2]}'
    cores = list(cores) if cores else list(rec.get('run_cores') or [])
    if not cores:
        return False, 'no run_cores in the record'
    fd, tmp = tempfile.mkstemp(suffix='.json')
    with os.fdopen(fd, 'w') as f:
        json.dump({k: rec.get(k) for k in ('validated', 'budget_clock', 'threads', 'workers',
                                           'affinity', 'cpu_share')}, f)
    try:
        ok, why = TV.classify_single_thread([tmp], slots=[cores])[tmp]
    finally:
        os.unlink(tmp)
    return bool(ok), why


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def cells(a):
    out = []
    for name in a.names:
        for s in a.seeds:
            p = record_path(a.root, a.budget, a.start, name, s)
            if os.path.exists(p):
                rec = json.load(open(p))
                state = 'valid' if rec.get('valid') and classify(rec, a.cores_allowed)[0] \
                    else 'invalid'
            elif TCP.n_invalid(p) >= a.max_invalid:
                state = 'held'
            else:
                state = 'missing'
            out.append(dict(name=name, seed=s, record=p, state=state))
    return out


def plan(a):
    cs = cells(a)
    pending = [c for c in cs if c['state'] in ('missing', 'invalid')]
    n = {s: sum(c['state'] == s for c in cs) for s in ('valid', 'missing', 'invalid', 'held')}
    plan_cores = a.cores_allowed or list(range(8, 18))
    k = len(plan_cores)
    rounds = -(-len(pending) // k) if pending else 0
    print(f'memetic on {SET} regime {REGIME}: {len(a.names)} instances x seeds '
          f'{",".join(map(str, a.seeds))} = {len(cs)} cells; start {a.start}; budget '
          f'{TCP.bnum(a.budget)} s; readings at {", ".join(map(str, READINGS))} s')
    print('  ' + ', '.join(f'{s} {v}' for s, v in n.items()))
    print(f'records: {rel(folder(a.root, a.budget, a.start))}/<name>_seed<k>.json')
    print(f'on {k} cores ({TCP.compress(plan_cores)}): {rounds} batch(es) x '
          f'({TCP.bnum(a.budget)} + {OVERHEAD_S:g}) s = '
          f'{rounds * (a.budget + OVERHEAD_S) / 60:.1f} min')
    print('the run is a timing measurement: its cores carry no other job for its whole '
          'duration, including the other timed campaigns')
    print('command (repository root): taskset -c 8-17 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
          'NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= '
          f'{sys.executable} -B scripts/p136_memetic_public_check.py run')
    print('plan only: nothing started, nothing written')
    return 0


def run(a):
    cores = sorted(os.sched_getaffinity(0))
    if a.development and os.path.abspath(a.root) == ROOT:
        print('REFUSING: --development needs --root outside the repository')
        return 1
    best_known()
    pending = [c for c in cells(a) if c['state'] in ('missing', 'invalid')]
    print(f'{len(pending)} cells to run on cores {cores}', flush=True)
    if not pending:
        return 0
    busy, every = TCP.busy_on(cores, 2.0)
    commit = 'not read (development run)' if a.development else TCP.git_commit()
    note = [f'=== memetic public check start {time.strftime("%Y-%m-%d %H:%M:%S")}, commit '
            f'{commit}, cores {cores}, {len(pending)} cells, budget {TCP.bnum(a.budget)} s, start '
            f'{a.start}' + (' (DEVELOPMENT: not a measurement)' if a.development else ''),
            '1. Purpose: the gap of the memetic algorithm to the best-known makespans of the '
            'Brandimarte instances without rules, at 60 s and 600 s of one run per seed.',
            '2. Expected: small gaps on the instances whose best-known values are optimal; if the '
            'gaps are large, the memetic baseline of the timed comparison is a weak search.',
            '3. Contamination: new records only (atomic, never overwritten, invalid ones moved to '
            'invalid/); one single-thread process per core; the cores are checked before every '
            'batch; cpu_share, affinity and load are in every record.',
            f'4. Data: {FAMILY}/desc/{REGIME}/<name>.desc.json, base instances '
            f'data/PUBLIC/Brandimarte/<name>.fjs, best-known {REG.PUB_SETS[SET]}.',
            'processes above 5% CPU, with the core they last ran on:'] + [f'  {x}' for x in every]
    if busy:
        note += [f'{"DEVELOPMENT, not refused" if a.development else "REFUSED"}: cores are not '
                 'idle:'] + [f'  {x}' for x in busy]
    p = os.path.join(a.root, NOTES)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a') as f:
        f.write('\n'.join(note) + '\n')
    if busy and not a.development:
        print('REFUSING: cores not idle:\n  ' + '\n  '.join(busy))
        return 1
    import p75_deploy_cost as P75
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', **{v: '1' for v in THREAD_VARS})
    bad = 0
    for k in range(0, len(pending), len(cores)):
        if os.path.exists(os.path.join(a.root, HOLD)):
            print(f'held by {rel(os.path.join(a.root, HOLD))}', flush=True)
            return 0
        batch = pending[k:k + len(cores)]
        for c in batch:
            if c['state'] == 'invalid':
                TCP.move_invalid(c['record'], 'invalid')
        b = P75.busy_processes(cores)
        idle = {'idle': not b, 'busy_processes': b, 'loadavg': P75.loadavg(),
                'time': time.strftime('%Y-%m-%d %H:%M:%S')}
        if b and not a.development:
            print('REFUSING batch: cores not idle:\n  ' + '\n  '.join(b), flush=True)
            return 1
        procs = []
        for core, c in zip(cores, batch):
            cmd = [sys.executable, '-B', os.path.abspath(__file__), '_cell', '--_cell',
                   f'{c["name"]}:{c["seed"]}:{core}', '--_idle', json.dumps(idle), '--_commit',
                   commit, '--budget', TCP.bnum(a.budget), '--start', a.start, '--root', a.root,
                   '--cores', TCP.compress(a.cores_allowed)]
            if a.development:
                cmd.append('--development')
            procs.append((c, subprocess.Popen(cmd, cwd=ROOT, env=env)))
        for c, pr in procs:
            pr.wait()
        for c, _ in procs:
            ok = os.path.exists(c['record']) and json.load(open(c['record'])).get('valid')
            if not ok:
                bad += 1
                if os.path.exists(c['record']):
                    TCP.move_invalid(c['record'], 'invalid')
    left = [c for c in cells(a) if c['state'] != 'valid']
    print(f'{len(cells(a)) - len(left)} of {len(cells(a))} cells valid')
    return 1 if left else 0


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def out_refusals(out):
    import p94_timed_inference as T
    banned = [os.path.join(ROOT, x) for x in T.RECORD_DIRS + ('results', 'out')] + \
             [os.path.join(ROOT, x) for x in os.listdir(ROOT) if x.startswith('paper')]
    if os.environ.get('PACT_OUT'):
        banned.append(os.path.abspath(os.environ['PACT_OUT']))
    return [f'--out {out} lies inside or around {rel(d)}' for d in banned
            if T._inside(out, d) or T._inside(d, out)]


def name_word(name):
    import p94_timed_inference as T
    return 'Mk' + T.number_word(int(name[2:]))


def analyse(a, recs, bk):
    """Gaps per instance and seed, per instance and over the instances (module docstring)."""
    per = {}
    for name in a.names:
        rows = {}
        for s in a.seeds:
            r = recs.get((name, s))
            if r is None:
                continue
            g = {b: 100.0 * (r['readings'][str(b)] - bk[name]) / bk[name]
                 for b in READINGS if r['readings'].get(str(b)) is not None}
            st = (r.get('start_schedule') or {}).get('makespan')
            rows[str(s)] = dict(makespan={str(b): r['readings'].get(str(b)) for b in READINGS},
                                gap={str(b): v for b, v in g.items()},
                                start_gap=None if st is None else 100.0 * (st - bk[name]) / bk[name])
        x = dict(best_known=bk[name], seeds=rows)
        for b in READINGS:
            v = [rows[s]['gap'][str(b)] for s in rows if str(b) in rows[s]['gap']]
            if v:
                x[f'mean_gap_{b}'] = float(np.mean(v))
                x[f'min_gap_{b}'] = float(np.min(v))
                x[f'n_{b}'] = len(v)
                x[f'min_makespan_{b}'] = min(rows[s]['makespan'][str(b)] for s in rows
                                             if str(b) in rows[s]['gap'])
                x[f'runs_above_{b}'] = sum(g > 1e-9 for g in v)
        per[name] = x
    pooled = {}
    for b in READINGS:
        v = [x[f'mean_gap_{b}'] for x in per.values() if f'mean_gap_{b}' in x]
        if v:
            pooled[str(b)] = dict(mean_gap=float(np.mean(v)), n_instances=len(v),
                                  at_best_known=sum(x[f'min_gap_{b}'] <= 1e-9
                                                    for x in per.values() if f'min_gap_{b}' in x),
                                  below_best_known=sum(x[f'min_gap_{b}'] < -1e-9
                                                       for x in per.values()
                                                       if f'min_gap_{b}' in x),
                                  runs_above=sum(x[f'runs_above_{b}'] for x in per.values()
                                                 if f'runs_above_{b}' in x))
    sg = [rows['start_gap'] for x in per.values() for rows in x['seeds'].values()
          if rows['start_gap'] is not None]
    return dict(instances=per, pooled=pooled,
                start_mean_gap=float(np.mean(sg)) if sg else None)


def macro_lines(res):
    import p94_timed_inference as T
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["run"]} run; {METHOD}; {SET} regime {REGIME} (no rule active); start '
         f'{res["start"]}; seeds {",".join(map(str, res["seeds"]))}; one run of '
         f'{TCP.bnum(res["budget_s"])} s per instance and seed, read at 60 s and 600 s of the '
         f'search; gap = 100 (makespan - best known) / best known, best known from '
         f'{REG.PUB_SETS[SET]}.']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}InstN', len(res['names']), 'Brandimarte instances')
    nc(f'{PREFIX}SeedN', len(res['seeds']), 'seeds of the memetic algorithm per instance')
    nc(f'{PREFIX}CellValidN', res['cells_valid'], f'valid runs of {res["cells_total"]}')
    for b in READINGS:
        bw = T.number_word(b)
        x = res['results']['pooled'].get(str(b))
        if x is None:
            continue
        nc(f'{PREFIX}MeanGap{bw}', f'{x["mean_gap"]:.2f}', f'{b} s: mean over {x["n_instances"]} '
           'instances of the mean gap over the seeds, percent of the best known')
        nc(f'{PREFIX}AtBestKnown{bw}N', x['at_best_known'], f'{b} s: instances on which a seed '
           'reaches the best-known makespan')
        nc(f'{PREFIX}BelowBestKnown{bw}N', x['below_best_known'], f'{b} s: instances on which a '
           'seed is below the best-known makespan')
        nc(f'{PREFIX}RunsAbove{bw}N', x['runs_above'], f'{b} s: runs above the best-known '
           'makespan, over all instances and seeds')
        for name, y in res['results']['instances'].items():
            if f'mean_gap_{b}' not in y:
                continue
            w = name_word(name)
            nc(f'{PREFIX}{w}MeanGap{bw}', f'{y[f"mean_gap_{b}"]:.2f}',
               f'{name}, {b} s: mean gap over {y[f"n_{b}"]} seeds, percent of the best known '
               f'{y["best_known"]:g}')
            nc(f'{PREFIX}{w}MinGap{bw}', f'{y[f"min_gap_{b}"]:.2f}',
               f'{name}, {b} s: smallest gap over the seeds')
            nc(f'{PREFIX}{w}MinMakespan{bw}', f'{y[f"min_makespan_{b}"]:g}',
               f'{name}, {b} s: shortest makespan over the seeds')
    for name, y in res['results']['instances'].items():
        nc(f'{PREFIX}{name_word(name)}BestKnown', f'{y["best_known"]:g}',
           f'{name}: best-known makespan of the table the public evaluation reads')
    if res['results']['start_mean_gap'] is not None:
        nc(f'{PREFIX}StartMeanGap', f'{res["results"]["start_mean_gap"]:.2f}',
           'mean gap of the start schedule (best of the four masked rules) over the runs')
    return L


def markdown(res):
    L = [f'# Memetic algorithm on {SET}, regime {REGIME} ({res["run"]} run)', '',
         f'Generated by {GENERATOR}. {METHOD}; start {res["start"]}; seeds '
         f'{",".join(map(str, res["seeds"]))}; one run of {TCP.bnum(res["budget_s"])} s, read at '
         f'60 s and 600 s of the search. Valid runs: {res["cells_valid"]} of '
         f'{res["cells_total"]}.', '',
         '| instance | best known | ' + ' | '.join(f'mean gap {b} s | min gap {b} s'
                                                   for b in READINGS) + ' |',
         '|---|---|' + '---|---|' * len(READINGS)]
    for name, y in res['results']['instances'].items():
        cols = []
        for b in READINGS:
            cols += [f'{y[f"mean_gap_{b}"]:.2f}' if f'mean_gap_{b}' in y else '-',
                     f'{y[f"min_gap_{b}"]:.2f}' if f'min_gap_{b}' in y else '-']
        L.append(f'| {name} | {y["best_known"]:g} | ' + ' | '.join(cols) + ' |')
    L.append('')
    for b, x in res['results']['pooled'].items():
        L.append(f'- {b} s: mean gap {x["mean_gap"]:.2f}% over {x["n_instances"]} instances; '
                 f'best known reached on {x["at_best_known"]}')
    return '\n'.join(L) + '\n'


def infer(a):
    if a.out:
        bad = out_refusals(a.out)
        if bad:
            raise SystemExit('REFUSING: ' + '; '.join(bad))
    elif os.environ.get('PACT_OUT') is None:
        raise SystemExit('REFUSING: set PACT_OUT or give --out')
    bk = best_known()
    cs = cells(a)
    recs = {(c['name'], c['seed']): json.load(open(c['record'])) for c in cs
            if c['state'] == 'valid'}
    full = len(recs) == len(cs) and float(a.budget) >= max(READINGS)
    if not full and not a.out:
        raise SystemExit(f'REFUSING the full-run write: {len(cs) - len(recs)} of {len(cs)} '
                         'cells without a valid record; give --out for a partial run')
    paths = ({k: os.path.join(a.out, v) for k, v in OUT_FILES.items()} if a.out
             else {k: REG.out(v) for k, v in OUT_FILES.items()})
    res = dict(generator=GENERATOR, run='full' if full else 'partial', set=SET, regime=REGIME,
               start=a.start, seeds=list(a.seeds), names=list(a.names),
               budget_s=float(a.budget), cells_total=len(cs), cells_valid=len(recs),
               best_known_table=REG.PUB_SETS[SET],
               records=rel(folder(a.root, a.budget, a.start)) + '/<name>_seed<k>.json',
               results=analyse(a, recs, bk))
    lines = macro_lines(res)
    if not full:
        lines.insert(1, '% PARTIAL: not every run is present; not for the paper')
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    if any(not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x) for x in names) or \
            len(names) != len(set(names)):
        raise SystemExit('REFUSING: macro names are not unique Mpc plus letters')
    import p124_memetic_parallel as P124
    taken = sorted(x for x in P124.defined_macros(paths['macros']) if x.startswith(PREFIX))
    if taken:
        raise SystemExit(f'REFUSING: the prefix {PREFIX} is used by \\{taken[0]}')
    from macro_io import emit
    emit(paths['macros'], lines)
    for k in ('json', 'md'):
        os.makedirs(os.path.dirname(paths[k]) or '.', exist_ok=True)
    with open(paths['json'], 'w') as f:
        json.dump(res, f, indent=1)
    with open(paths['md'], 'w') as f:
        f.write(markdown(res))
    print(f'p136: {res["run"]} run, {len(recs)} of {len(cs)} runs; wrote '
          + ', '.join(rel(p) for p in paths.values()))
    return 0


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--budget', type=float, default=DEFAULT_BUDGET)
    common.add_argument('--start', choices=STARTS, default='rule')
    common.add_argument('--instances', default=','.join(NAMES), help='names, e.g. Mk01,Mk06')
    common.add_argument('--seeds', default='1-3')
    common.add_argument('--root', default=ROOT)
    common.add_argument('--cores', default=None, help='the cores a valid run may use '
                        '(default the affinity mask of the run)')
    common.add_argument('--max_invalid', type=int, default=2)
    common.add_argument('--development', action='store_true')
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('plan', parents=[common])
    sub.add_parser('run', parents=[common])
    q = sub.add_parser('infer', parents=[common])
    q.add_argument('--out', default=None)
    q = sub.add_parser('_cell', parents=[common])
    q.add_argument('--_cell', required=True)
    q.add_argument('--_idle', default=None)
    q.add_argument('--_commit', default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(_ARGV if argv is None else argv)
    a.root = os.path.abspath(os.path.join(_CWD, a.root))
    if getattr(a, 'out', None):
        a.out = os.path.abspath(os.path.join(_CWD, a.out))
    a.names = [x.strip() for x in a.instances.split(',') if x.strip()]
    if any(x not in NAMES for x in a.names):
        raise SystemExit(f'unknown instance in {a.names}; want names of {NAMES}')
    a.seeds = TCP.parse_instances(a.seeds)
    # the cores a valid run may use: --cores, else for run the affinity mask of the run, else
    # (plan, infer) the run_cores stored in each record
    a.cores_allowed = (TCP.parse_cores(a.cores) if a.cores
                       else sorted(os.sched_getaffinity(0)) if a.cmd == 'run' else None)
    if a.cmd == '_cell':
        name, seed, core = a._cell.split(':')
        out = record_path(a.root, a.budget, a.start, name, int(seed))
        return solve_cell(a, name, int(seed), int(core),
                          json.loads(a._idle) if a._idle else None, out)
    if a.cmd == 'plan':
        return plan(a)
    if a.cmd == 'run':
        return run(a)
    return infer(a)


if __name__ == '__main__':
    sys.exit(main())
