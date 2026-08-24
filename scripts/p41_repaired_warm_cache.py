"""P41: build the REPAIRED best-of-four warm-start cache for CP-SAT.

The shipped warm cache (scripts/p12_warm_cache.py, or_solution/FAMILY/
warmstart_cache*) seeds CP-SAT from the per-instance best of the four ORIGINAL
dispatching rules. Three of those rules (FIFO, MOR, MWKR) pick a job first and
can commit it to a machine that cannot start it now -- an action outside the
policy's feasibility mask. scripts/p25_rule_mask_audit.py defines the REPAIRED
variant of each rule: the priority key applied only over jobs that hold a legal
pair, machine chosen inside the mask (exactly what SPT already does). This script
rolls out the four repaired rules on every (size, regime, instance), selects the
per-instance best, and writes a cache in the SAME format p12_cpsat_fairwarm.py
consumes, under a clearly-labelled repaired directory.

WHY THE CONVENTIONS ARE p12's, NOT p25's. The repaired hints will be fed to the
SAME solver setup the original cache feeds, and the per-instance guard compares
them against the original cache. Both must therefore be built identically: the
rollout mirrors p12._rollout exactly (env constructed with descriptor only, no
type embeddings; np.random.seed(50) after set_initial_data; schedule read from
env.true_op_ct), and only the action rule is swapped for p25.repaired_action,
imported verbatim so the logic cannot drift.

GUARD / FLOOR. The stored hint must be <= the original best-of-four makespan (the
`ms` field already in the shipped cache) on every instance, so re-crediting can
only move CP-SAT's number in its own favour (the conservative direction for the
policy comparison). On a few setup-FREE instances an original rule that escapes
the feasibility mask lands a feasible schedule 1-7 units better than any of the
mask-respecting repaired rules. A warm start should be the strongest feasible
incumbent available, so the stored hint is the better of the repaired portfolio
and the original portfolio per instance: repaired improvements are kept, and those
rare instances retain the original schedule. This makes the guard hold by
construction; the floored instances are counted and reported, not hidden.

Sizes: 10x25 (100-instance test set), 20x25 (50), 30x25 (20), all 8 regimes.
Outputs: or_solution/FAMILY/warmstart_cache_repaired[_<size>]/{regime}/{name}.npz
(full schedule: amch [N], ct [N], ms, rule, rule_names, rule_ms), plus a
makespans.json per size with the per-instance repaired/original makespans.

Usage (LIGHT CPU; the CP-SAT campaign owns cores 0-15):
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 taskset -c 20-23 \
    python scripts/p41_repaired_warm_cache.py [--sizes 10x25,20x25,30x25] \
    [--regimes N,L,S,W,LS,LW,SW,LSW] [--limit 0] [--procs 4]
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, 'scripts')
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

# Parse our own flags BEFORE the heavy imports (p25/params) reset sys.argv.
_ARGV = sys.argv[1:]


def _pop(flag, default=None):
    if flag in _ARGV:
        i = _ARGV.index(flag)
        v = _ARGV[i + 1]
        del _ARGV[i:i + 2]
        return v
    return default


ARG_SIZES = _pop('--sizes', '10x25,20x25,30x25').split(',')
ARG_REGIMES = _pop('--regimes', 'N,L,S,W,LS,LW,SW,LSW').split(',')
ARG_LIMIT = int(_pop('--limit', '0'))          # 0 = all instances in the set
ARG_PROCS = int(_pop('--procs', '4'))          # 4 = the pinned core set (20-23)

# p25 resets sys.argv to ['x', '--device', 'cpu'] at import time so it can import
# params on the CPU. Give it a clean argv, then restore ours.
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]

import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '2')))
except Exception:
    pass

from p25_rule_mask_audit import repaired_action, RULES  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from constraint_family.reference_sim import validate_schedule  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

sys.argv = _saved_argv

# Each size: instance set, and the shipped (original) and new (repaired) caches.
SIZE_CFG = {
    '10x25': dict(data='data/FAMILY/10x25', base='data/PPVC/10x25+ppvc-mixed',
                  orig='or_solution/FAMILY/warmstart_cache',
                  rep='or_solution/FAMILY/warmstart_cache_repaired'),
    '20x25': dict(data='data/FAMILY/20x25', base='data/PPVC/20x25+ppvc-mixed',
                  orig='or_solution/FAMILY/warmstart_cache_20x25',
                  rep='or_solution/FAMILY/warmstart_cache_repaired_20x25'),
    '30x25': dict(data='data/FAMILY/30x25', base='data/PPVC/30x25+ppvc-mixed',
                  orig='or_solution/FAMILY/warmstart_cache_30x25',
                  rep='or_solution/FAMILY/warmstart_cache_repaired_30x25'),
}


def _repaired_rollout(rule, jl, pt, desc):
    """Repaired dispatching-rule rollout. Mirrors p12_cpsat_fairwarm._rollout
    exactly (env, seeding, schedule readout); only the action rule differs."""
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    np.random.seed(50)
    M = pt.shape[1]
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        a = repaired_action(rule, env)
        job, mch = a // M, a % M
        amch[env.candidate[0, job]] = mch
        env.step(np.array([a]))
    ct = env.true_op_ct[0]
    return (amch, ct), float(np.max(ct))


def _repaired_best(jl, pt, desc):
    """Best of the four repaired rules, each validated against the reference
    simulator before it is eligible (mirrors p12._warm_start)."""
    best_ws, best_ms, best_rule, per_rule = None, float('inf'), None, {}
    for r in RULES:
        (amch, ct), ms = _repaired_rollout(r, jl, pt, desc)
        st = np.array([ct[i] - pt[i, amch[i]] if amch[i] >= 0 else -1.0
                       for i in range(pt.shape[0])])
        viol = validate_schedule(jl, pt, desc, st, ct, amch)
        if viol:
            per_rule[r] = float('inf')          # ineligible, recorded as such
            continue
        per_rule[r] = ms
        if ms < best_ms:
            best_ws, best_ms, best_rule = (amch, ct), ms, r
    if best_ws is None:
        raise RuntimeError('no feasible repaired warm start')
    return best_ws, best_ms, best_rule, per_rule


def one(task):
    size, stem, dp, regime, name, orig_cache, rep_cache = task
    jl, pt, meta = load_instance(stem)
    with open(dp) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    best_ws, raw_ms, best_rule, per_rule = _repaired_best(jl, pt, desc)

    oz = np.load(f'{orig_cache}/{regime}/{name}.npz', allow_pickle=True)
    orig_ms = float(oz['ms'])

    # FLOOR AT THE ORIGINAL. On a handful of setup-FREE instances an original
    # rule that escapes the feasibility mask lands a feasible schedule 1-7 units
    # better than anything the mask-respecting repaired rules produce (the escape
    # is usually harmful but is not always). A warm start should be the strongest
    # feasible incumbent available, and a stronger CP-SAT hint is the conservative
    # direction for the policy-vs-solver comparison, so we never hand CP-SAT a hint
    # weaker than the one it already had: the stored hint is the better of the
    # repaired portfolio and the original portfolio, per instance. This makes the
    # guard (stored <= original) hold by construction and leaves the genuine
    # improvements untouched. Instances where the floor bound (raw repaired was
    # worse) are counted and reported.
    floored = raw_ms > orig_ms + 1e-6
    if floored:
        best_ws = (np.asarray(oz['amch']), np.asarray(oz['ct']))
        best_ms = orig_ms
        stored_rule = 'orig:' + str(oz['rule'])
    else:
        best_ms = raw_ms
        stored_rule = best_rule
    assert best_ms <= orig_ms + 1e-6                # guard, now by construction

    os.makedirs(f'{rep_cache}/{regime}', exist_ok=True)
    np.savez(f'{rep_cache}/{regime}/{name}.npz',
             amch=best_ws[0], ct=best_ws[1], ms=best_ms, rule=stored_rule,
             rule_names=list(per_rule),
             rule_ms=[per_rule[k] for k in per_rule])
    return dict(size=size, regime=regime, name=name,
                repaired_ms=best_ms, raw_repaired_ms=raw_ms, orig_ms=orig_ms,
                best_rule=stored_rule, raw_best_rule=best_rule,
                per_rule={k: (None if v == float('inf') else v)
                          for k, v in per_rule.items()},
                floored=bool(floored))


def main():
    tasks = []
    for size in ARG_SIZES:
        cfg = SIZE_CFG[size]
        for r in ARG_REGIMES:
            descs = sorted(glob.glob(f"{cfg['data']}/desc/{r}/*.desc.json"))
            if ARG_LIMIT:
                descs = descs[:ARG_LIMIT]
            for dp in descs:
                name = os.path.basename(dp).replace('.desc.json', '')
                op = f"{cfg['orig']}/{r}/{name}.npz"
                if not os.path.exists(op):
                    raise SystemExit(
                        f'REFUSING: no original warm start at {op}. The guard '
                        f'compares against it, so it must exist first.')
                tasks.append((size, os.path.join(cfg['base'], name), dp, r,
                              name, cfg['orig'], cfg['rep']))

    print(f'{len(tasks)} repaired warm starts to build on {ARG_PROCS} procs',
          flush=True)
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=ARG_PROCS) as ex:
        for k, res in enumerate(ex.map(one, tasks), 1):
            results.append(res)
            if k % 25 == 0 or k == len(tasks):
                print(f'  [{k}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min',
                      flush=True)

    # ------------------------------------------------------------ makespans
    for size in ARG_SIZES:
        cfg = SIZE_CFG[size]
        ms = {}
        for r in results:
            if r['size'] != size:
                continue
            ms.setdefault(r['regime'], {})[r['name']] = dict(
                repaired_ms=r['repaired_ms'], raw_repaired_ms=r['raw_repaired_ms'],
                orig_ms=r['orig_ms'], best_rule=r['best_rule'],
                raw_best_rule=r['raw_best_rule'], per_rule=r['per_rule'],
                improvement=r['orig_ms'] - r['repaired_ms'],
                floored=r['floored'])
        if ms:
            os.makedirs(cfg['rep'], exist_ok=True)
            with open(f"{cfg['rep']}/makespans.json", 'w') as f:
                json.dump(ms, f, indent=1)

    # ------------------------------------------------------------ report
    print('\nrepaired vs original best-of-four (per size, per regime)')
    print(f'{"size":7}{"reg":5}{"n":>4}{"improved":>10}'
          f'{"meanImp%":>10}{"maxImp%":>9}{"mean(orig)":>12}{"mean(rep)":>11}')
    for size in ARG_SIZES:
        for r in ARG_REGIMES:
            rr = [x for x in results if x['size'] == size and x['regime'] == r]
            if not rr:
                continue
            orig = np.array([x['orig_ms'] for x in rr])
            rep = np.array([x['repaired_ms'] for x in rr])
            imp = orig - rep
            n_imp = int((imp > 1e-6).sum())
            gains = 100.0 * imp[imp > 1e-6] / orig[imp > 1e-6]
            mean_imp = float(gains.mean()) if gains.size else 0.0
            max_imp = float(gains.max()) if gains.size else 0.0
            print(f'{size:7}{r:5}{len(rr):>4}{n_imp:>10}'
                  f'{mean_imp:>10.2f}{max_imp:>9.2f}'
                  f'{orig.mean():>12.2f}{rep.mean():>11.2f}')

    floored = [r for r in results if r['floored']]
    if floored:
        print(f'\nFLOORED {len(floored)} instance(s): the raw repaired portfolio '
              f'was worse than the original (a mask-escaping original rule found a '
              f'better feasible schedule), so the original hint was retained:')
        for v in floored[:30]:
            print(f"  {v['size']} {v['regime']}/{v['name']}: raw repaired "
                  f"{v['raw_repaired_ms']:.1f} > original {v['orig_ms']:.1f} "
                  f"({v['raw_best_rule']}); stored the original schedule instead")

    # The stored hint is min(repaired portfolio, original portfolio) per instance,
    # so this holds unconditionally; assert it so a future change cannot break it.
    bad = [r for r in results if r['repaired_ms'] > r['orig_ms'] + 1e-6]
    if bad:
        print(f'\nGUARD FAILED unexpectedly on {len(bad)} instance(s) '
              f'AFTER flooring -- this should be impossible:')
        for v in bad[:30]:
            print(f"  {v['size']} {v['regime']}/{v['name']}: "
                  f"{v['repaired_ms']:.1f} > {v['orig_ms']:.1f}")
        sys.exit(1)
    print(f'\nGUARD PASSED: stored repaired hint <= original best-of-four on all '
          f'{len(results)} instances ({len(floored)} floored to original). '
          f'Repaired caches built.')


if __name__ == '__main__':
    main()
