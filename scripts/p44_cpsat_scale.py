"""P44: the anytime CP-SAT-vs-policy crossover at LARGER instance sizes.

At 10x25 the 30 s solver has passed the policy on all eight regimes (supplement
Sec. S-XII, scripts/p30_anytime.py). The question here is whether that budget
survives scale: at 20x25 (440 ops) and 30x25 (660 ops) the solver's model grows
while the policy stays sub-second, so the crossover may move far beyond 30 s.

This runner is the 10x25 fair-warm anytime protocol (p12_cpsat_fairwarm.py,
p30_anytime.py) with EXACTLY ONE thing changed: the instance size. It reuses
p12_cpsat_fairwarm.solve_one unchanged, so the warm-start construction
(best-of-four dispatching rules, validated, cached), the crediting rule
(min(hint, search), applied later by the aggregator), the worker count, and the
record format are all identical to the shipped 10x25 campaign.

THE ONE TRAP THIS SCRIPT AVOIDS. The 10x25 warm-start cache is keyed only by
`{regime}/{name}.npz`, and instance names repeat across sizes (instance_000 ...).
Pointed at the shipped cache, a 20x25 solve would silently be handed a 10x25 warm
start. So the cache is namespaced by size here: or_solution/FAMILY/warmstart_cache_<size>.

The solver's budget is WALL-CLOCK, so it must run on dedicated cores. Pin the
process (taskset -c 0-15) and keep concurrent solves x workers-per-solve <= 16.

Usage:
  # phase 1: build the size-namespaced warm-start cache at full parallelism
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-15 \
    python scripts/p44_cpsat_scale.py --size 30x25 --phase cache --procs 16
  # phase 2: solve one budget rung
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 0-15 \
    python scripts/p44_cpsat_scale.py --size 30x25 --phase solve \
      --budget 30 --workers 8 --jobs 2
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

# p12 parses sys.argv-free; strip our flags before importing anything that might
# read them, mirroring the sibling scripts.
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np  # noqa: E402

import p12_cpsat_fairwarm as fw  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402


def _paths(size):
    return (f'data/FAMILY/{size}', f'data/PPVC/{size}+ppvc-mixed',
            f'or_solution/FAMILY/warmstart_cache_{size}')


def _build_one(task):
    stem, dp, regime, name, cache = task
    fw.WARM_CACHE = cache                 # re-set inside each worker process
    if os.path.exists(f'{cache}/{regime}/{name}.npz'):
        return None
    jl, pt, _ = load_instance(stem)
    with open(dp) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    _, ms, rule, _ = fw._warm_start(stem, jl, pt, desc, regime, name)
    return f'{regime}/{name}: {rule} {ms:.0f}'


def build_cache(size, regimes, procs):
    data, base, cache = _paths(size)
    # Namespace the cache so a 20x25/30x25 solve can never be handed a 10x25 hint.
    fw.WARM_CACHE = cache
    tasks = []
    for r in regimes:
        for dp in sorted(glob.glob(f'{data}/desc/{r}/*.desc.json')):
            name = os.path.basename(dp).replace('.desc.json', '')
            if os.path.exists(f'{cache}/{r}/{name}.npz'):
                continue
            tasks.append((os.path.join(base, name), dp, r, name, cache))
    print(f'[{size}] {len(tasks)} warm starts to build on {procs} procs',
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for k, out in enumerate(ex.map(_build_one, tasks), 1):
            if k % 25 == 0 or k == len(tasks):
                print(f'  [{k}/{len(tasks)}] {(time.time()-t0)/60:.1f} min',
                      flush=True)
    print(f'[{size}] warm-start cache complete', flush=True)


def _solve_one(task):
    # Reuse p12's solve_one verbatim, but pin the namespaced cache first so the
    # warm-start lookup inside it cannot resolve to the 10x25 cache.
    task, cache = task
    fw.WARM_CACHE = cache
    return fw.solve_one(task)


def solve_rung(size, regimes, budget, workers, jobs, anytime=False, limit=0,
               out_suffix='', warm_cache=None):
    data, base, cache = _paths(size)
    # --warm-cache repoints the hint source (e.g. the policy's own schedules for
    # the policy-seeded hybrid); everything else in the protocol is unchanged.
    cache = warm_cache or cache
    fw.WARM_CACHE = cache
    tasks = []
    for regime in regimes:
        out_dir = f'or_solution/FAMILY/{size}_fairwarm{budget}s{out_suffix}/{regime}'
        os.makedirs(out_dir, exist_ok=True)
        descs = sorted(glob.glob(f'{data}/desc/{regime}/*.desc.json'))
        if limit:
            # The FIRST n, fixed in advance, not a sample: a prefix cannot be
            # tuned after the fact the way a chosen subset can.
            descs = descs[:limit]
        for dp in descs:
            name = os.path.basename(dp).replace('.desc.json', '')
            out_path = os.path.join(out_dir, f'{name}.json')
            if os.path.exists(out_path):
                continue
            cp = f'{cache}/{regime}/{name}.npz'
            if not os.path.exists(cp):
                raise SystemExit(
                    f'REFUSING: no warm start at {cp}. Build the cache first '
                    f'(--phase cache). A missing hint would silently change the '
                    f'protocol.')
            tasks.append(((os.path.join(base, name), dp, out_path, budget,
                           workers, anytime), cache))
    print(f'[{size} {budget}s] {len(tasks)} solves queued  ({workers} workers '
          f'x {jobs} jobs, warm start from {cache}'
          f'{", anytime trajectory recorded" if anytime else ""})', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, out in enumerate(ex.map(_solve_one, tasks), 1):
            inst, reg, st, obj, rule, wms = out
            print(f'[{k}/{len(tasks)}] {reg:4s} {inst}  {st:9s} obj={obj}  '
                  f'warm={rule}({wms})  ({(time.time()-t0)/60:.1f} min)',
                  flush=True)
    print(f'[{size} {budget}s] rung complete', flush=True)


def main():
    a = list(_ARGV)
    size, phase = '30x25', 'solve'
    budget, workers, jobs, procs = 30, 8, 2, 16
    anytime = False
    limit = 0          # 0 = every instance; n = the first n
    # A replicate of cells already in the campaign goes to its own directory:
    # this solver is NOT deterministic (eight workers racing under a wall-clock
    # budget, no fixed seed), so re-running an instance is a second draw, not a
    # correction, and it must not overwrite the first.
    out_suffix = ''
    warm_cache = None
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
    i = 0
    while i < len(a):
        if a[i] == '--size':
            size = a[i + 1]; i += 2
        elif a[i] == '--phase':
            phase = a[i + 1]; i += 2
        elif a[i] == '--budget':
            budget = int(a[i + 1]); i += 2
        elif a[i] == '--workers':
            workers = int(a[i + 1]); i += 2
        elif a[i] == '--jobs':
            jobs = int(a[i + 1]); i += 2
        elif a[i] == '--procs':
            procs = int(a[i + 1]); i += 2
        elif a[i] == '--anytime':
            anytime = True; i += 1
        elif a[i] == '--limit':
            limit = int(a[i + 1]); i += 2
        elif a[i] == '--out-suffix':
            out_suffix = a[i + 1]; i += 2
        elif a[i] == '--warm-cache':
            warm_cache = a[i + 1]; i += 2
        elif a[i] == '--regimes':
            regimes = []
            i += 1
            while i < len(a) and not a[i].startswith('--'):
                regimes.append(a[i]); i += 1
        else:
            sys.exit(f'unknown arg {a[i]}')

    if phase == 'cache':
        build_cache(size, regimes, procs)
    elif phase == 'solve':
        assert workers * jobs <= 16, \
            f'{workers} workers x {jobs} jobs > 16 cores: would starve the solver'
        if warm_cache:
            assert out_suffix, 'a repointed hint source needs its own --out-suffix'
        solve_rung(size, regimes, budget, workers, jobs, anytime, limit,
                   out_suffix, warm_cache)
    else:
        sys.exit(f'unknown phase {phase}')


if __name__ == '__main__':
    main()
