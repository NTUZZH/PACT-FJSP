"""CP-SAT references with a best-of-four-rule warm start.

An anytime CP-SAT run is only as good as the incumbent it starts from, so the
choice of seeding heuristic is part of the baseline's configuration, not an
implementation detail. Seeding from a single fixed rule can understate the
solver badly when that rule is weak on the regime at hand (rule quality varies
widely across regimes, and CP-SAT need not recover the difference within the
budget).

This script therefore rolls out all four dispatching rules (FIFO, MOR, SPT,
MWKR) on every (instance, regime), warm-starts CP-SAT from whichever gives the
lowest makespan, and solves at the documented budget and worker count. That is
the strongest start the solver can be given from heuristics alone, and it is the
configuration the reported comparison is made against.

The solver's budget is WALL-CLOCK, so it must run on dedicated cores (pin the
process with taskset and size --jobs/--workers to the pinned set); a solver
starved of CPU returns worse solutions without reporting any error. Idempotent:
existing results are skipped.

ANYTIME MODE (--anytime, borrowed from the scale-up arm of the companion
transport study). With the flag the solve additionally records its whole
upper-bound trajectory: every improving incumbent lands in the record as
[wall_s, objective, bound]. One long solve then answers the entire budget
ladder, because the makespan the solver would have returned at any smaller
budget is its last incumbent before that instant. At the sizes where a ladder
would cost the sum of its rungs this is the only affordable way to ask how long
exact search needs to reach the policy's quality. The model, the parameters and
the warm start are untouched; the flag only attaches a callback.

Usage:
  python scripts/p12_cpsat_fairwarm.py --regimes S LS SW LSW \
         [--budget 300] [--workers 8] [--jobs 2] [--out-tag _fairwarm]
         [--anytime]

  Other instance sizes go through scripts/p44_cpsat_scale.py, which namespaces
  the warm-start cache by size; this script's cache is not namespaced, so
  pointing it at another size would hand a 10x25 warm start to a larger
  instance.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# params.py parses sys.argv at import time, so strip our flags before importing.
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor
from constraint_family.cpsat_family import family_solver
from constraint_family.reference_sim import validate_schedule
from ppvc_instance_generator import load_instance

RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')


def _rollout(rule, jl, pt, desc):
    """Deterministic dispatching-rule rollout through the family env.

    Forced onto the CPU: a dispatching rule needs no network, so a GPU context
    here would only take memory away from the training lanes (and, at the
    parallelism the cache builder uses, exhaust the card outright).
    """
    from params import configs
    configs.device = 'cpu'
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from common_utils import heuristic_select_action
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    np.random.seed(50)
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        a = heuristic_select_action(rule, env)
        job, mch = a // pt.shape[1], a % pt.shape[1]
        amch[env.candidate[0, job]] = mch
        env.step(np.array([a]))
    ct = env.true_op_ct[0]
    return (amch, ct), float(np.max(ct))


WARM_CACHE = 'or_solution/FAMILY/warmstart_cache'


def _warm_start(stem, jl, pt, desc, regime, name):
    """Best-of-four-rules warm start, cached on disk.

    The four rollouts cost ~19 s per instance and depend only on (regime,
    instance), not on the solve budget. Recomputing them for every budget in
    the ladder would burn hours of single-threaded time while the solver cores
    sit idle, so the result is memoized.

    Every candidate is validated against the reference simulator before it is
    eligible. This is not a formality: the solver RETAINS its warm start as an
    incumbent (see cpsat_family.family_solver), so an infeasible hint would be
    reported as CP-SAT's own answer at budgets where the search returns nothing.
    A rule that violates the regime does not get to be the warm start.
    """
    os.makedirs(f'{WARM_CACHE}/{regime}', exist_ok=True)
    cp = f'{WARM_CACHE}/{regime}/{name}.npz'
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return ((z['amch'], z['ct']), float(z['ms']), str(z['rule']),
                {k: float(v) for k, v in zip(z['rule_names'], z['rule_ms'])})
    best_ws, best_ms, best_rule = None, float('inf'), None
    per_rule = {}
    for r in RULES:
        ws, ms = _rollout(r, jl, pt, desc)
        amch, ct = ws
        st = np.array([ct[i] - pt[i, amch[i]] if amch[i] >= 0 else -1.0
                       for i in range(pt.shape[0])])
        viol = validate_schedule(jl, pt, desc, st, ct, amch)
        if viol:
            per_rule[r] = float('inf')       # ineligible, and recorded as such
            continue
        per_rule[r] = ms
        if ms < best_ms:
            best_ws, best_ms, best_rule = ws, ms, r
    if best_ws is None:
        raise RuntimeError(f'no feasible warm start for {regime}/{name}')
    np.savez(cp, amch=best_ws[0], ct=best_ws[1], ms=best_ms, rule=best_rule,
             rule_names=list(per_rule), rule_ms=[per_rule[k] for k in per_rule])
    return best_ws, best_ms, best_rule, per_rule


def solve_one(task):
    stem, desc_path, out_path, budget, workers, anytime = task
    jl, pt, meta = load_instance(stem)
    with open(desc_path) as f:
        d = json.load(f)
    desc = ConstraintDescriptor.from_json(d)

    # BEST of the four rules, per instance -- the strongest fair incumbent.
    best_ws, best_ms, best_rule, per_rule = _warm_start(
        stem, jl, pt, desc, d['regime'], os.path.basename(stem))

    # Wall time is the budget, but CPU time is what says whether the budget was
    # honoured: eight workers on eight dedicated cores should burn about eight
    # CPU-seconds per wall second, and a ratio well below that is starvation,
    # which a search reports by returning a worse answer rather than an error.
    # Recorded per solve so a contested run can be audited after the fact
    # instead of relying on someone having sampled ps at the time.
    _cpu0 = time.process_time()
    res = family_solver(jl, pt, desc, time_limit=budget, num_workers=workers,
                        warmstart=best_ws, anytime=anytime)
    _cpu = time.process_time() - _cpu0
    rec = {'instance': os.path.basename(stem), 'regime': d['regime'],
           'budget_s': budget, 'workers': workers,
           'warm_rule': best_rule, 'warm_makespan': best_ms,
           'warm_all_rules': per_rule,
           'status': res['status'], 'objective': res['objective'],
           'bound': res['bound'], 'solve_time': round(res['solve_time'], 2),
           'cpu_time': round(_cpu, 2),
           'cpu_per_wall': round(_cpu / max(res['solve_time'], 1e-9), 2)}
    if anytime:
        rec['anytime'] = res.get('anytime', [])
    if res['objective'] is not None:
        viol = validate_schedule(jl, pt, desc, res['op_st'], res['op_ct'],
                                 res['assigned_mch'])
        rec['validated'] = not viol
        rec['violations'] = viol[:5]
    with open(out_path, 'w') as f:
        json.dump(rec, f)
    return (rec['instance'], rec['regime'], rec['status'], rec['objective'],
            best_rule, round(best_ms, 1))


def main():
    args = list(_ARGV)
    data, base = 'data/FAMILY/10x25', 'data/PPVC/10x25+ppvc-mixed'
    budget, workers, jobs = 300, 8, 2
    anytime = False
    out_tag = '_fairwarm'
    limit = 0          # 0 = all instances; N = the FIRST N (a fixed prefix, not
                       # a sampled subset, so the choice cannot be tuned)
    regimes = ['S', 'LS', 'SW', 'LSW']   # setup-bearing: where the claim lives
    i = 0
    while i < len(args):
        if args[i] == '--budget':
            budget = int(args[i + 1]); i += 2
        elif args[i] == '--workers':
            workers = int(args[i + 1]); i += 2
        elif args[i] == '--jobs':
            jobs = int(args[i + 1]); i += 2
        elif args[i] == '--limit':
            limit = int(args[i + 1]); i += 2
        elif args[i] == '--out-tag':
            out_tag = args[i + 1]; i += 2
        elif args[i] == '--anytime':
            anytime = True; i += 1
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
        if limit:
            descs = descs[:limit]
        for dp in descs:
            name = os.path.basename(dp).replace('.desc.json', '')
            out_path = os.path.join(out_dir, f'{name}.json')
            if os.path.exists(out_path):
                continue
            tasks.append((os.path.join(base, name), dp, out_path, budget,
                          workers, anytime))

    print(f'{len(tasks)} solves queued  (budget {budget}s, {workers} workers '
          f'x {jobs} parallel jobs, best-of-{len(RULES)} PDR warm start'
          f'{", anytime trajectory recorded" if anytime else ""})',
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for k, out in enumerate(ex.map(solve_one, tasks), 1):
            inst, reg, st, obj, rule, wms = out
            print(f'[{k}/{len(tasks)}] {reg:4s} {inst}  {st:9s} obj={obj}  '
                  f'warm={rule}({wms})  ({(time.time()-t0)/60:.1f} min)',
                  flush=True)
    print('fair-warm campaign complete', flush=True)


if __name__ == '__main__':
    main()
