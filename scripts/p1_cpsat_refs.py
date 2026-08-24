"""CP-SAT reference campaign for the FAMILY benchmark (this paper, P1/E5).

Solves every (instance, regime) with the family model at a fixed budget and
writes one JSON per solve under or_solution/FAMILY/<size>/<regime>/.
Idempotent: existing result files are skipped, so the campaign can be
resumed/parallelized by regime.

Usage:
  python scripts/p1_cpsat_refs.py --regimes N L S W LS LW SW LSW \
         [--data data/FAMILY/10x25] [--budget 300] [--workers 8] [--jobs 3]
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# params.py parses sys.argv at import time (via the env import chain), so our
# private flags must be captured and stripped BEFORE any project import.
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor, REGIMES_ALL
from constraint_family.cpsat_family import family_solver
from constraint_family.reference_sim import validate_schedule
from ppvc_instance_generator import load_instance


def solve_one(task):
    stem, desc_path, out_path, budget, workers, warm = task
    jl, pt, meta = load_instance(stem)
    with open(desc_path) as f:
        d = json.load(f)
    desc = ConstraintDescriptor.from_json(d)

    # deterministic MWKR rollout through the family env as the warm start:
    # guarantees a feasible incumbent on every (instance, regime) and lets
    # the 300 s budget go into IMPROVING it (hints don't change the model)
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from common_utils import heuristic_select_action
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    np.random.seed(50)
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        a = heuristic_select_action('MWKR', env)
        job, mch = a // pt.shape[1], a % pt.shape[1]
        amch[env.candidate[0, job]] = mch
        env.step(np.array([a]))
    ws = (amch, env.true_op_ct[0])

    res = family_solver(jl, pt, desc, time_limit=budget, num_workers=workers,
                        warmstart=ws if warm else None)
    rec = {'instance': os.path.basename(stem), 'regime': d['regime'],
           'budget_s': budget, 'status': res['status'],
           'objective': res['objective'], 'bound': res['bound'],
           'solve_time': round(res['solve_time'], 2)}
    if res['objective'] is not None:
        viol = validate_schedule(jl, pt, desc, res['op_st'], res['op_ct'],
                                 res['assigned_mch'])
        rec['validated'] = not viol
        rec['violations'] = viol[:5]
        rec['assigned_mch'] = res['assigned_mch'].tolist()
        rec['op_st'] = res['op_st'].tolist()
        rec['op_ct'] = res['op_ct'].tolist()
    with open(out_path, 'w') as f:
        json.dump(rec, f)
    return (rec['instance'], rec['regime'], rec['status'], rec['objective'],
            rec.get('validated'))


def main():
    args = list(_ARGV)
    data = 'data/FAMILY/10x25'
    base = 'data/PPVC/10x25+ppvc-mixed'
    budget, workers, jobs = 300, 8, 3
    regimes = list(REGIMES_ALL)
    out_tag, warm = '', True
    i = 0
    while i < len(args):
        if args[i] == '--data':
            data = args[i + 1]; i += 2
        elif args[i] == '--base':
            base = args[i + 1]; i += 2
        elif args[i] == '--budget':
            budget = int(args[i + 1]); i += 2
        elif args[i] == '--workers':
            workers = int(args[i + 1]); i += 2
        elif args[i] == '--jobs':
            jobs = int(args[i + 1]); i += 2
        elif args[i] == '--out-tag':
            out_tag = args[i + 1]; i += 2
        elif args[i] == '--no-warmstart':
            warm = False; i += 1
        elif args[i] == '--regimes':
            regimes = []
            i += 1
            while i < len(args) and not args[i].startswith('--'):
                regimes.append(args[i]); i += 1
        else:
            sys.exit(f'unknown arg {args[i]}')

    size_tag = os.path.basename(data.rstrip('/'))
    tasks = []
    for regime in regimes:
        out_dir = f'or_solution/FAMILY/{size_tag}{out_tag}/{regime}'
        os.makedirs(out_dir, exist_ok=True)
        descs = sorted(glob.glob(os.path.join(data, 'desc', regime,
                                              '*.desc.json')))
        for dp in descs:
            name = os.path.basename(dp).replace('.desc.json', '')
            stem = os.path.join(base, name)
            out_path = os.path.join(out_dir, f'{name}.json')
            if os.path.exists(out_path):
                continue
            tasks.append((stem, dp, out_path, budget, workers, warm))

    print(f'{len(tasks)} solves queued  (budget {budget}s, {workers} workers '
          f'x {jobs} parallel jobs)', flush=True)
    t0 = time.time()
    n_done = 0
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(solve_one, tasks):
            n_done += 1
            print(f'[{n_done}/{len(tasks)}] {r[1]:<4} {r[0]:<14} {r[2]:<8} '
                  f'obj={r[3]}  valid={r[4]}  '
                  f'({(time.time()-t0)/60:.1f} min elapsed)', flush=True)
    print('campaign complete', flush=True)


if __name__ == '__main__':
    main()
