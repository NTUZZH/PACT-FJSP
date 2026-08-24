r"""Which scale-up cell earns the solver's time.

Two axes were opened at once and only the cheap arms have been run on them:
shop size (50 and 80 modules against the 10-module training size) and machine
contention (the generator's capacity-halved 15-station factory against the
published 25-station one). A CP-SAT campaign at a literature-standard budget
costs days per cell, so this script reads the policy and the dispatching rules
on all four and reports the one quantity that should decide where those days go:
the policy's lead over the best rule, per cell, per regime, with a paired
interval.

WHY THE LEAD OVER THE RULES IS THE RIGHT PROXY. The solver is credited with the
better of its search and its warm start, and the warm start is built from the
same rules. A cell where the policy is far ahead of the rules is therefore a
cell where the solver starts far behind and has to close the whole distance by
search; a cell where the rules are already close is one where it starts close.
The proxy cannot tell us whether the search succeeds, which is the experiment
itself, but it does order the cells by how much work the search has to do.

Estimator and conventions are p11_equivalence.py's throughout: the mean
per-instance relative makespan gap, paired over instances, 20k-resample
bootstrap on rng seed 7, policy pooled over its three seeds and refused if the
seeds are not all present.

Usage: python scripts/p58_scale_scout.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']
# The RELEASED policy, and the one every published scale result keys on
# (p18_e6_table.py, p53_load_axis.py). joint-v1 is the CONDITIONED control arm,
# which the paper reports as transferring worse as the shop grows, so reading it
# as the protagonist reproduces that known deficit and calls it a scale result.
POLICY = 'greedy+10x25+family+jointblind-v1'
COND = 'greedy+10x25+family+joint-v1'
SEED_SFX = ('', '-s302', '-s303')
# (cell, modules, stations) -- the published sizes first, as the reference the
# new cells are read against.
CELLS = [('10x25', 10, 25), ('20x25', 20, 25), ('30x25', 30, 25),
         ('50x25', 50, 25), ('80x25', 80, 25),
         ('50x15', 50, 15), ('80x15', 80, 15)]
N_BOOT = 20000
DATA = 'results/scale_scout.json'

# Guard, not a comment: the published scale analyses must agree with this one
# about which policy is the protagonist.
for _p in ('scripts/p18_e6_table.py', 'scripts/p53_load_axis.py'):
    if 'jointblind-v1' not in open(_p).read():
        sys.exit(f'REFUSING: {_p} no longer names jointblind-v1, so this script '
                 f'and the published scale results may disagree about which '
                 f'policy is being reported.')

rng = np.random.default_rng(7)


def arr(cell, tag, r):
    p = f'test_results/FAMILY/{cell}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def policy(cell, r):
    xs = [arr(cell, f'{POLICY}{s}', r) for s in SEED_SFX]
    if any(x is None for x in xs):
        return None, sum(x is not None for x in xs)
    return np.mean(xs, axis=0), len(xs)


def best_rule(cell, r):
    """Per-instance best of the four rules is NOT what the paper compares
    against; the paper's 'best rule' is the single strongest rule, which is SPT
    in every published regime. Both are reported here: the single rule keeps
    this comparable with the published cells, and the per-instance portfolio is
    the stronger floor a solver would actually be warm-started from."""
    xs = {k: arr(cell, k, r) for k in RULES}
    if any(v is None for v in xs.values()):
        return None, None
    single = min(xs.values(), key=lambda v: v.mean())
    port = np.min(np.stack(list(xs.values())), axis=0)
    return single, port


def ci(d):
    n = len(d)
    bs = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


rows, missing = [], []
for cell, nj, nm in CELLS:
    gaps_single, gaps_port, per_regime, nwin, nlose = [], [], {}, 0, 0
    for r in REGIMES:
        pol, nseed = policy(cell, r)
        single, port = best_rule(cell, r)
        if pol is None or single is None:
            missing.append(f'{cell}/{r} (policy seeds {nseed}, rules '
                           f'{"ok" if single is not None else "missing"})')
            continue
        n = min(len(pol), len(single))
        g = 100 * (single[:n] - pol[:n]) / single[:n]      # + = policy ahead
        gp = 100 * (port[:n] - pol[:n]) / port[:n]
        gaps_single.append(g)
        gaps_port.append(gp)
        m, lo, hi = ci(g)
        per_regime[r] = dict(mean=m, lo=lo, hi=hi)
        nwin += lo > 0
        nlose += hi < 0
    if len(gaps_single) != len(REGIMES):
        continue
    m, lo, hi = ci(np.concatenate(gaps_single))
    mp, lop, hip = ci(np.concatenate(gaps_port))
    rows.append(dict(cell=cell, jobs=nj, stations=nm, mean=m, lo=lo, hi=hi,
                     port_mean=mp, port_lo=lop, port_hi=hip,
                     win=nwin, lose=nlose, per_regime=per_regime,
                     ops=None))

if not rows:
    sys.exit('no complete cell yet' + (f' -- missing: {missing[0]}' if missing
                                       else ''))

print(f'{"cell":8s}{"jobs":>5s}{"mch":>5s}{"vs best rule":>16s}'
      f'{"vs portfolio":>16s}{"  ahead/behind":>15s}')
print('-' * 66)
for d in rows:
    print(f'{d["cell"]:8s}{d["jobs"]:5d}{d["stations"]:5d}'
          f'{d["mean"]:+8.2f}%[{d["lo"]:+5.2f},{d["hi"]:+5.2f}]'
          f'{d["port_mean"]:+8.2f}%[{d["port_lo"]:+5.2f},{d["port_hi"]:+5.2f}]'
          f'{d["win"]:>8d}/{d["lose"]:<6d}')
if missing:
    print(f'\nincomplete ({len(missing)}): ' + '; '.join(missing[:6])
          + (' ...' if len(missing) > 6 else ''))

# The decision this script exists to make, stated as a comparison rather than
# left to the eye: at each size, does halving the factory widen the lead?
print()
by = {d['cell']: d for d in rows}
for nj in (50, 80):
    a, b = by.get(f'{nj}x25'), by.get(f'{nj}x15')
    if a and b:
        d = b['mean'] - a['mean']
        print(f'{nj} modules: 25 stations {a["mean"]:+.2f}%  ->  15 stations '
              f'{b["mean"]:+.2f}%  ({d:+.2f} points '
              f'{"WIDER" if d > 0 else "narrower"} under contention)')

os.makedirs('results', exist_ok=True)
with open(DATA, 'w') as f:
    json.dump(dict(rows=rows, regimes=REGIMES, rules=RULES), f, indent=1)
print(f'\nwrote {DATA}')
