"""Memetic-algorithm baseline on the FAMILY benchmark.

A genetic algorithm over (operation sequence, machine assignment)
chromosomes whose offspring are improved by a critical-path tabu search
(baselines/memetic, method and references in its README.md). Each instance
is solved in its own child process on one thread; the parent only builds the
task list and waits.

Protocol per instance
  1. load the instance and its descriptor (data/FAMILY/<size>/desc/<regime>/,
     any size folder, e.g. 50x25 or 50x25n30; the descriptor names its base
     instance file); a descriptor that carries setup_mult is refused;
  2. seed schedules, all validated (configuration check C6 and the
     attached-setup presence check) before they enter the population:
     --seed_schedules with --warm_cache F: the schedule stored for the
       instance in F (F/<regime>/<instance>.npz, the dispatching-rule warm
       start that CP-SAT receives as its hint);
     --seed_schedules without --warm_cache: the four dispatching rules of
       the CP-SAT warm start (FIFO, MOR, SPT, MWKR) rolled out through the
       family environment as scripts/p12_cpsat_fairwarm.py does, compared
       with the default cache of the size when it exists;
     --seed_from F1 [F2 ...]: in addition, the schedule stored for the
       instance in each folder (same npz format, e.g. the policy's
       schedules in or_solution/FAMILY/pactwarm_cache_<size>); a missing or
       invalid file stops the run;
  3. compile or load the numba kernels on the instance;
  4. run the search for --budget seconds of the budget clock (or until
     --max_decodes schedule evaluations, which makes the run deterministic
     in --seed); steps 1 to 3 run before the timed window, as the CP-SAT
     warm start is built before its budget starts;
  5. validate the returned schedule; a schedule that fails is never written.

One JSON record per (regime, instance, seed) at
<out>/<regime>/<instance>_seed<seed>.json. Existing records are skipped, and
never overwritten without --force. Every record carries the timing evidence
of the run: wall and CPU seconds of the timed window, cpu_share = CPU s /
(wall s x 1 thread), the affinity mask and the 1-minute load average at the
start and at the end. The anytime record holds every improvement of the
incumbent as [budget-clock s since the search start, wall s and CPU s since
the start of the solving process, makespan]; search_start_wall_s and
search_start_cpu_s give the span before the timed window, jit_time_s the
numba warm-up inside it (baselines/memetic/records.py reads the incumbent at
a given time).

The budget is wall-clock, so a result is a timing measurement: run each
process on a core that nothing else uses for its whole duration, e.g.
  taskset -c 8 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \\
    CUDA_VISIBLE_DEVICES='' python eval_memetic_family.py --size 50x25n30 \\
    --regimes LSW --instances 0-29 --budget 1800 --seed 1 --seed_schedules \\
    --warm_cache or_solution/FAMILY/warmstart_cache_xmask_50x25n30 \\
    --out or_solution/FAMILY/50x25n30_memetic1800s
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--size', required=True,
                    help='a folder under data/FAMILY with desc/<regime>/, e.g. '
                         '10x25, 50x25, 80x25, 50x25n30, 80x25n30')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--instances', default='0-9',
                    help='indices, e.g. 0-9 or 0,2,5-7')
    ap.add_argument('--budget', type=float, default=None,
                    help='wall-clock seconds of search per instance')
    ap.add_argument('--max_decodes', type=int, default=None,
                    help='stop after this many schedule evaluations '
                         '(deterministic in --seed)')
    ap.add_argument('--clock', choices=('wall', 'cpu'), default='wall',
                    help="budget clock: 'wall' for real runs; 'cpu' (process "
                         "CPU seconds) only for development runs on a shared core")
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--seed_schedules', action='store_true',
                    help='insert the dispatching-rule warm start into the '
                         'initial population')
    ap.add_argument('--warm_cache', default=None,
                    help='folder of the dispatching-rule warm-start cache '
                         '(<folder>/<regime>/<instance>.npz with amch, ct); with '
                         '--seed_schedules its schedule is the rule seed')
    ap.add_argument('--seed_from', nargs='+', default=[],
                    help='folders of further stored schedules (same npz format) '
                         'added to the initial population')
    ap.add_argument('--out', required=True)
    ap.add_argument('--config', default=None,
                    help='JSON object overriding entries of DEFAULT_CONFIG')
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing records')
    a = ap.parse_args(argv)
    if a.budget is None and a.max_decodes is None:
        ap.error('give --budget or --max_decodes')
    if a.warm_cache and not a.seed_schedules:
        ap.error('--warm_cache names the rule seed; give it with --seed_schedules')
    if a.warm_cache:
        a.warm_cache = os.path.abspath(a.warm_cache)
    a.seed_from = [os.path.abspath(f) for f in a.seed_from]
    for f in ([a.warm_cache] if a.warm_cache else []) + a.seed_from:
        if not os.path.isdir(f):
            ap.error(f'no folder {f}')
    for r in a.regimes:
        if r not in REGIMES:
            ap.error(f'unknown regime {r}')
    return a


def parse_indices(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def code_fingerprint():
    """SHA-256 of the package sources, so a record names the code that made it."""
    h = hashlib.sha256()
    pkg = os.path.join(ROOT, 'baselines', 'memetic')
    for f in ('instance.py', 'decoder.py', 'search.py', 'validate.py'):
        with open(os.path.join(pkg, f), 'rb') as fh:
            h.update(fh.read())
    with open(os.path.abspath(__file__), 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()[:16]


def _warm_cache_path(size, regime, name):
    base = ('or_solution/FAMILY/warmstart_cache' if size == '10x25'
            else f'or_solution/FAMILY/warmstart_cache_{size}')
    return os.path.join(base, regime, f'{name}.npz')


def rule_seed_schedules(jl, pt, desc, size, regime, name):
    """The four dispatching-rule schedules of the CP-SAT warm start, each
    validated with the reference validator. Returns (schedules, per-rule
    makespans, record of the comparison with the CP-SAT warm-start cache)."""
    import numpy as np
    import p12_cpsat_fairwarm as fw
    from constraint_family.reference_sim import validate_schedule
    seeds, per_rule = [], {}
    for rule in fw.RULES:
        (amch, ct), ms = fw._rollout(rule, jl, pt, desc)
        st = ct - pt[np.arange(pt.shape[0]), amch]
        if validate_schedule(jl, pt, desc, st, ct, amch):
            per_rule[rule] = None           # ineligible, recorded as such
            continue
        per_rule[rule] = float(ms)
        seeds.append((np.asarray(amch), np.asarray(ct)))
    cache = {}
    cp = _warm_cache_path(size, regime, name)
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        best = min(v for v in per_rule.values() if v is not None)
        cache = {'warm_cache_rule': str(z['rule']),
                 'warm_cache_makespan': float(z['ms']),
                 'warm_cache_match': bool(abs(float(z['ms']) - best) < 1e-9)}
    return seeds, per_rule, cache


def _rel(path):
    """A path inside the repository as written relative to its root."""
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def load_stored_schedule(folder, regime, name, inst):
    """The schedule stored for one instance in a cache folder
    (<folder>/<regime>/<name>.npz with amch and ct, as in
    or_solution/FAMILY/warmstart_cache_<size> and pactwarm_cache_<size>),
    validated with C6 and the attached-setup presence check. Returns
    (amch, ct, info); raises when the file is missing, does not fit the
    instance or fails validation."""
    import numpy as np
    from baselines.memetic.validate import validate
    path = os.path.join(folder, regime, f'{name}.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f'no stored schedule {path}')
    z = np.load(path, allow_pickle=True)
    amch = np.asarray(z['amch'], dtype=np.int64)
    ct = np.asarray(z['ct'], dtype=float)
    if amch.shape != (inst.N,) or ct.shape != (inst.N,):
        raise ValueError(f'{path}: {amch.shape[0]} operations, the instance has {inst.N}')
    if not np.all(ct == np.round(ct)):
        raise ValueError(f'{path}: end times are not integer hours')
    ct = np.round(ct).astype(np.int64)
    st = ct - inst.pt[np.arange(inst.N), amch]
    ok, viol, ms = validate(inst, amch, st, ct)
    if not ok:
        raise ValueError(f'{path}: stored schedule fails validation '
                         f'({len(viol)} violations; first: {viol[:1]})')
    if 'ms' in z.files and abs(float(z['ms']) - ms) > 1e-9:
        raise ValueError(f'{path}: stored makespan {float(z["ms"])} != max end {ms}')
    info = {'folder': _rel(folder), 'rule': str(z['rule']) if 'rule' in z.files else None,
            'makespan': int(ms)}
    if 'rule_names' in z.files and 'rule_ms' in z.files:
        info['rule_makespans'] = {str(r): float(v)
                                  for r, v in zip(z['rule_names'], z['rule_ms'])}
    return amch, ct, info


def solve_one(task):
    """Child process: solve one instance and write its record."""
    import numpy as np
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    from baselines.memetic.instance import Instance
    from baselines.memetic.search import Memetic
    from baselines.memetic.validate import validate

    a, regime, idx, out_path = task
    origin = (time.perf_counter(), time.process_time())   # start of this process
    name = f'instance_{idx:03d}'
    dp = os.path.join('data/FAMILY', a.size, 'desc', regime, f'{name}.desc.json')
    with open(dp) as f:
        drec = json.load(f)
    jl, pt, _ = load_instance(drec['base_instance'])
    # benchmark descriptors were drawn under the attached setup convention
    desc = ConstraintDescriptor.from_json(drec, setup_convention='attached')
    inst = Instance(jl, pt, desc)       # refuses a descriptor with setup_mult
    config = json.loads(a.config) if a.config else {}

    seeds, per_rule, cache, t_seed, seed_info = [], {}, {}, 0.0, []
    t = time.perf_counter()
    if a.seed_schedules and a.warm_cache:
        amch, ct, info = load_stored_schedule(a.warm_cache, regime, name, inst)
        seeds.append((amch, ct))
        seed_info.append(dict(info, source='warm_cache'))
        per_rule = info.get('rule_makespans') or {info['rule']: float(info['makespan'])}
        cache = {'warm_cache_rule': info['rule'],
                 'warm_cache_makespan': float(info['makespan']),
                 'warm_cache_match': True}
        best_rule, seed_best = info['rule'], float(info['makespan'])
    elif a.seed_schedules:
        seeds, per_rule, cache = rule_seed_schedules(jl, pt, desc, a.size,
                                                     regime, name)
        for (amch, ct), r in zip(seeds, [r for r in per_rule if per_rule[r] is not None]):
            seed_info.append({'source': 'rule rollout', 'rule': r,
                              'makespan': int(per_rule[r])})
        valid = [r for r in per_rule if per_rule[r] is not None]
        best_rule = min(valid, key=lambda r: per_rule[r]) if valid else None
        seed_best = per_rule[best_rule] if valid else None
    else:
        best_rule, seed_best = None, None
    for folder in a.seed_from:
        amch, ct, info = load_stored_schedule(folder, regime, name, inst)
        seeds.append((amch, ct))
        seed_info.append(dict(info, source='seed_from'))
    t_seed = time.perf_counter() - t

    # compile (or load from the numba cache) every kernel before the window
    t = time.perf_counter()
    Memetic(inst, seed=0, config=dict(config, pop_size=2, n_offspring=1),
            max_evals=3000).run(seeds[:1])
    w = Memetic(inst, seed=0, max_evals=10)
    w._load_chromosome(inst.base_os.copy(), inst.min_pt_mach.copy())
    t_jit = time.perf_counter() - t

    mm = Memetic(inst, seed=a.seed, config=config, budget_s=a.budget,
                 max_evals=a.max_decodes, clock=a.clock, origin=origin)
    aff = sorted(os.sched_getaffinity(0))
    load0 = os.getloadavg()[0]
    cpu0, w0 = time.process_time(), time.perf_counter()
    res = mm.run(seeds)
    wall, cpu = time.perf_counter() - w0, time.process_time() - cpu0
    load1 = os.getloadavg()[0]
    for info, ms in zip(seed_info, res['seed_makespans']):
        info['makespan_semi_active'] = int(ms)

    ok, viol, vms = validate(inst, res['op_mch'], res['op_st'], res['op_ct'])
    if not ok or int(vms) != res['makespan']:
        print(f'REFUSING to write {out_path}: schedule fails validation '
              f'({len(viol)} violations; first: {viol[:1]}; makespan '
              f'{vms} vs reported {res["makespan"]})', flush=True)
        return 2
    import numba
    rec = {
        'instance': name, 'regime': regime, 'size': a.size, 'seed': a.seed,
        'budget_s': a.budget, 'budget_clock': a.clock,
        'max_decodes': a.max_decodes,
        'seed_schedules': bool(a.seed_schedules),
        'warm_cache': _rel(a.warm_cache) if a.warm_cache else None,
        'seed_from': [_rel(f) for f in a.seed_from],
        'seed_from_makespans': {i['folder']: i['makespan'] for i in seed_info
                                if i['source'] == 'seed_from'},
        'seeds': seed_info,
        'method': 'memetic (GA + critical-path tabu search)',
        'setup_convention': desc.setup_convention,
        'setup_convention_declared': bool(desc.setup_convention_declared),
        'config': mm.cfg,
        'makespan': res['makespan'],
        'anytime_columns': ['search_s', 'wall_s_from_process_start',
                            'cpu_s_from_process_start', 'makespan'],
        'anytime': res['anytime'],
        'search_start_wall_s': res['search_start_wall_s'],
        'search_start_cpu_s': res['search_start_cpu_s'],
        'n_decodes': res['n_decodes'],
        'n_evaluations': res['n_evaluations'],
        'n_generations': res['n_generations'],
        'n_restarts': res['n_restarts'],
        'n_ts_iterations': res['n_ts_iterations'],
        'n_its_kicks': res['n_its_kicks'],
        'n_its_accepted': res['n_its_accepted'],
        'n_its_improved': res['n_its_improved'],
        'ga_start_fraction': res['ga_start_fraction'],
        'ga_start_s': res['ga_start_s'],
        'wall_s': round(wall, 3), 'cpu_s': round(cpu, 3), 'threads': 1,
        'cpu_share': round(cpu / max(wall, 1e-9), 4),
        'affinity': aff,
        'loadavg_1min_start': round(load0, 2), 'loadavg_1min_end': round(load1, 2),
        'seed_rule_makespans': per_rule,
        'seed_best': seed_best,
        'seed_time_s': round(t_seed, 2), 'jit_time_s': round(t_jit, 2),
        **cache,
        'validated': True, 'violations': [],
        'schedule': {'op_mch': [int(v) for v in res['op_mch']],
                     'op_st': [int(v) for v in res['op_st']],
                     'op_ct': [int(v) for v in res['op_ct']]},
        'numba': numba.__version__, 'code_sha256_16': code_fingerprint(),
    }
    # the key names of the CP-SAT records (scripts/p12_cpsat_fairwarm.py,
    # scripts/p80_cpsat_rerun.py), so that one aggregator reads both
    rec.update({'objective': float(res['makespan']), 'status': 'FEASIBLE',
                'workers': 1, 'solve_time': rec['wall_s'],
                'cpu_time': rec['cpu_s'], 'cpu_per_wall': rec['cpu_share'],
                'warm_rule': best_rule, 'warm_makespan': seed_best,
                'warm_all_rules': per_rule,
                'loadavg': [round(x, 2) for x in os.getloadavg()],
                'peak_rss_gb': round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    os.replace(tmp, out_path)
    print(f'{regime:4s} {name}  makespan={res["makespan"]}  seed_best='
          f'{rec["seed_best"]}  seeds={[i["makespan"] for i in seed_info]}  '
          f'wall={wall:.1f}s  cpu_share={rec["cpu_share"]}  '
          f'evals={res["n_evaluations"]}  gens={res["n_generations"]}  '
          f'kicks={res["n_its_kicks"]}', flush=True)
    return 0


def _child(task, q):
    try:
        q.put(solve_one(task))
    except Exception as e:      # report and let the parent continue
        import traceback
        traceback.print_exc()
        q.put(f'error: {e!r}')


def main():
    a = parse_args(sys.argv[1:])
    sys.argv = sys.argv[:1]         # params.py parses sys.argv at import time
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[v] = '1'
    from params import configs
    configs.device = 'cpu'

    tasks = []
    for regime in a.regimes:
        for idx in parse_indices(a.instances):
            dp = os.path.join('data/FAMILY', a.size, 'desc', regime,
                              f'instance_{idx:03d}.desc.json')
            if not os.path.exists(dp):
                sys.exit(f'no descriptor {dp}')
            out_path = os.path.join(a.out, regime,
                                    f'instance_{idx:03d}_seed{a.seed}.json')
            if os.path.exists(out_path) and not a.force:
                print(f'skip {out_path} (exists)', flush=True)
                continue
            tasks.append((a, regime, idx, out_path))
    print(f'{len(tasks)} runs queued, one process each, affinity '
          f'{sorted(os.sched_getaffinity(0))}', flush=True)
    ctx = mp.get_context('fork')
    failed = 0
    for t in tasks:
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(t, q))
        p.start()
        p.join()
        status = q.get() if not q.empty() else f'exit code {p.exitcode}'
        if status != 0:
            failed += 1
            print(f'FAILED {t[1]} instance_{t[2]:03d}: {status}', flush=True)
    if failed:
        sys.exit(f'{failed} run(s) failed')


if __name__ == '__main__':
    main()
