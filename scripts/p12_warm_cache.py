"""Pre-build the best-of-four warm-start cache used by p12_cpsat_fairwarm.

Rollout and solve phases are kept separate. The four dispatching-rule rollouts
per (regime, instance) (~19 s of single-threaded Python each) are pure
heuristic computation with NO wall-clock sensitivity, so they run at full
parallelism here; the solve tiers then read the cache and spend their entire
wall-clock budget on search. Running the rollouts inside the solve workers
would leave solver cores idle during the rollout phase, and raising --jobs to
compensate would oversubscribe the dedicated cores (CP-SAT gets 8 search
workers per job) and starve the solver -- and a starved solver's short-budget
results measure CPU contention, not the solver.

Usage: python scripts/p12_warm_cache.py [--procs 16] [--regimes N L S W LS LW SW LSW]
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor
from ppvc_instance_generator import load_instance

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from p12_cpsat_fairwarm import _warm_start, WARM_CACHE  # noqa: E402


def one(task):
    stem, dp, regime, name = task
    if os.path.exists(f'{WARM_CACHE}/{regime}/{name}.npz'):
        return None
    jl, pt, _ = load_instance(stem)
    with open(dp) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    _, ms, rule, _ = _warm_start(stem, jl, pt, desc, regime, name)
    return f'{regime}/{name}: {rule} {ms:.0f}'


def main():
    a = list(_ARGV)
    procs = int(a[a.index('--procs') + 1]) if '--procs' in a else 16
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
    if '--regimes' in a:
        i = a.index('--regimes') + 1
        regimes = []
        while i < len(a) and not a[i].startswith('--'):
            regimes.append(a[i]); i += 1

    tasks = []
    for r in regimes:
        for dp in sorted(glob.glob(f'data/FAMILY/10x25/desc/{r}/*.desc.json')):
            name = os.path.basename(dp).replace('.desc.json', '')
            if os.path.exists(f'{WARM_CACHE}/{r}/{name}.npz'):
                continue
            tasks.append((os.path.join('data/PPVC/10x25+ppvc-mixed', name),
                          dp, r, name))
    print(f'{len(tasks)} warm starts to build on {procs} procs', flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procs) as ex:
        for k, out in enumerate(ex.map(one, tasks), 1):
            if k % 25 == 0 or k == len(tasks):
                print(f'  [{k}/{len(tasks)}] {(time.time()-t0)/60:.1f} min',
                      flush=True)
    print('warm-start cache complete', flush=True)


if __name__ == '__main__':
    main()
