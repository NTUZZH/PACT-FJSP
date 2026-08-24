"""P30: when does the exact solver actually overtake the policy?

The main paper says that where setups are active, "an exact solve below roughly ten
seconds returns nothing better than the rule it started from". Only 5 s and 30 s were
ever measured, so the ten-second threshold was an interpolation across a gap in the
evidence rather than a measurement. The gap turns out to be exactly where the
interesting thing happens.

This reads the fair-warm CP-SAT campaigns at 1, 2, 5, 10 and 30 seconds and
reports, per regime and per budget. The warm start is the repaired best-of-four
hint (scripts/p41_repaired_warm_cache.py): the 5s and 30s rungs were re-solved from
it (10x25_fairwarmR{5,30}s), while the 1s/2s/10s rungs, not re-solved, have their
credited value floored at the repaired hint. Per regime and per budget it reports:

  * how many of the instances the SEARCH improved on the warm start it was handed,
    which is the quantity the claim is actually about; and
  * the policy's lead over the solver's credited makespan, under the Protocol's
    min(warm start, search) rule.

WHAT THE MEASUREMENT MUST NOT BE. A wall-clock budget on a contended machine is not
a budget: a starved solver returns a worse schedule and reports no error, which
would make the solver look weak and the policy look strong. Every rung here ran
pinned to cores that no other job could enter, and the isolation was PROVED rather
than assumed: rerunning the published 5 s rung on those cores reproduced all 100 of
its objectives and all 100 of its warm starts exactly, instance by instance. A rung
that cannot show that is not admissible evidence and is not reported.

Emits macros_anytime.tex.

Usage: python scripts/p30_anytime.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


OR = 'or_solution/FAMILY'
RES = 'test_results/FAMILY/10x25'
# PROTAGONIST SWAP (2026-08-11): the anytime frontier is measured for the
# RELEASED token-blind joint, the same policy Table I and E1/E2 now show.
JOINT = 'greedy+10x25+family+jointblind-v1'
SEEDS = ['', '-s302', '-s303']
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SETUP = ['S', 'LS', 'SW', 'LSW']          # the regimes the claim is about
RUNGS = [1, 2, 5, 10, 30]                 # 300 s is a 30-instance prefix; excluded
RERUN = (5, 30)                           # tiers re-solved from the repaired hint

# The repaired best-of-four warm-start cache (scripts/p41_repaired_warm_cache.py):
# per instance, the best of the four mask-respecting dispatching rules, floored at
# the original best-of-four. The 5s and 30s rungs were re-solved from this hint
# (10x25_fairwarmR{5,30}s); the 1s/2s/10s rungs were not, so their credited value
# is floored at the repaired hint, which can only lower it (the conservative
# direction for the policy comparison).
_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def _rung_dir(budget):
    """Directory for a budget: the re-solved repaired-hint campaign for the tiers
    that were rerun, the original campaign for the rest."""
    return (f'{OR}/10x25_fairwarmR{budget}s' if budget in RERUN
            else f'{OR}/10x25_fairwarm{budget}s')


def cells(budget, regime):
    """[(credited makespan, improved-on-warm-start?)] for one (budget, regime),
    credited against the repaired hint."""
    out = []
    rerun = budget in RERUN
    for p in sorted(glob.glob(f'{_rung_dir(budget)}/{regime}/*.json')):
        d = json.load(open(p))
        x = d[0] if isinstance(d, list) else d
        obj, warm = x['objective'], x.get('warm_makespan')
        if warm is None:
            raise SystemExit(f'{p} has no warm_makespan: this is not a fair-warm '
                             f'run and must not be mixed with the ones that are')
        if rerun:
            # warm_makespan IS the repaired hint; credit the better of hint and search.
            out.append((min(obj, warm), obj < warm))
        else:
            # Not rerun: floor the tier's original credited value at the repaired
            # hint, and count the search as improving only when it beat that hint.
            hint = _REPAIRED[regime][x['instance']]['repaired_ms']
            out.append((min(hint, min(obj, warm)), obj < hint))
    return out


def policy(regime):
    a = [np.load(f'{RES}/Result_{JOINT}{s}_{regime}.npy') for s in SEEDS]
    a = [(x[:, 0] if x.ndim > 1 else x).astype(float) for x in a]
    return np.mean(a, axis=0)


have = [b for b in RUNGS
        if len(glob.glob(f'{_rung_dir(b)}/*/*.json')) == 800]
missing = [b for b in RUNGS if b not in have]
# Not a warning. The supplement's table hard-codes all five budgets as columns and
# the prose names them, so a rung that quietly vanishes here does not shrink the
# table; it fills a column with the ?? placeholder, or worse, leaves a stale one. And
# the frontier itself (AnyDeadSec, AnyAliveSec) is defined by WHICH rungs were run:
# drop the 10 s rung and "the search improves on nothing at 5 s or less" would still
# print, having lost the very rung that bounds it from above.
if missing:
    raise SystemExit(
        f'REFUSING: rung(s) {missing}s are incomplete (a full rung is 800 cells: '
        f'8 regimes x 100 instances). The frontier this script reports is defined by '
        f'the rungs that were run, so it may not be computed from a subset of the '
        f'rungs the paper names.')

print('SEARCH IMPROVED ON ITS WARM START, instances out of 100')
print(f'{"regime":7}' + ''.join(f'{b:>7}s' for b in have))
print('-' * (7 + 8 * len(have)))
imp = {}
for r in REGIMES:
    row = []
    for b in have:
        c = cells(b, r)
        imp[(b, r)] = sum(1 for _, i in c if i)
        row.append(imp[(b, r)])
    tag = '  <- setup' if r in SETUP else ''
    print(f'{r:7}' + ''.join(f'{v:>8}' for v in row) + tag)

rng = np.random.default_rng(7)          # the paper's bootstrap seed


def paired_gap(pol, other):
    """THE ESTIMATOR. Copied from p13_e5_table_fair.py, which states the Protocol
    every percentage in this paper is held to: the mean of the PER-INSTANCE relative
    gaps, paired instance by instance. Not a ratio of means.

    This function replaces exactly that mistake. The first version of this script
    computed 100*(solver.mean() - policy.mean())/policy.mean(), which is a ratio of
    means: a different quantity that prints in the same units, and one the Protocol
    explicitly disowns. It also carried no interval, while the same Protocol says
    that a point estimate with no interval is not a comparison.

    Positive = `other` (the solver) is slower than the policy, i.e. the policy wins.
    """
    d = 100.0 * (other - pol) / pol                  # per instance, paired
    idx = rng.integers(0, len(d), (20000, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


print()
print('POLICY LEAD over the credited solver makespan (%), + = policy ahead')
print('paired per-instance gaps; a lead whose CI covers zero is NOT a lead')
print(f'{"regime":7}' + ''.join(f'{b:>16}s' for b in have))
print('-' * (7 + 17 * len(have)))
lead, lead_ci = {}, {}
for r in REGIMES:
    p = policy(r)
    row = []
    for b in have:
        s = np.array([m for m, _ in cells(b, r)], dtype=float)
        m, lo, hi = paired_gap(p, s)
        lead[(b, r)] = m
        lead_ci[(b, r)] = (lo, hi)
        row.append(f'{m:+6.2f}[{lo:+.2f},{hi:+.2f}]')
    print(f'{r:7}' + ''.join(f'{v:>17}' for v in row))

# The claim the paper makes is about the SETUP regimes, so the macro has to be too.
# The largest budget at which the search improved on its warm start on ZERO instances
# of EVERY setup-bearing regime is the largest budget at which the paper's sentence
# is literally true.
dead = [b for b in have if all(imp[(b, r)] == 0 for r in SETUP)]
alive = [b for b in have if any(imp[(b, r)] > 0 for r in SETUP)]
last_dead = max(dead) if dead else None
first_alive = min(alive) if alive else None

print()
if last_dead is not None:
    print(f'On every setup-bearing regime, the search improves on its warm start on')
    print(f'  ZERO of 100 instances at {last_dead}s or less.')
if first_alive is not None:
    frac = [imp[(first_alive, r)] for r in SETUP]
    print(f'  At {first_alive}s it improves on {min(frac)}-{max(frac)} of 100.')
    print(f'  The transition therefore lies between {last_dead}s and {first_alive}s,')
    print(f'  and the paper may say "{last_dead}s", not "roughly ten seconds",')
    print('  unless the 10s rung says otherwise.')

# "Ahead" means the paired interval clears zero, not that the point estimate happens
# to be positive. Counting a lead whose interval covers zero as a win is the thing the
# Protocol forbids everywhere else in this paper, and the sentence these macros feed
# ("it trails on N of the 4 setup-bearing regimes") would be asserting a difference
# the data does not carry.
def is_ahead(b, r):
    return b is not None and lead_ci[(b, r)][0] > 0


ahead = [r for r in SETUP if is_ahead(last_dead, r)]
# The claim that survives the correction, and the one worth printing: the search
# starts working between AnyDeadSec and AnyAliveSec, but it has still not caught the
# policy at AnyAliveSec. The paper was wrong about the mechanism and right about the
# frontier, and both halves have to be stated.
ahead_alive = [r for r in SETUP if is_ahead(first_alive, r)]
if first_alive:
    print(f'  But at {first_alive}s the policy still leads the credited solver, with '
          f'the interval clear of zero, on {len(ahead_alive)} of the {len(SETUP)} '
          f'setup regimes.')
    for r in SETUP:
        lo, hi = lead_ci[(first_alive, r)]
        print(f'    {r:4s} {lead[(first_alive, r)]:+.2f}% [{lo:+.2f},{hi:+.2f}]'
              f'{"  AHEAD" if lo > 0 else "  not significant"}')
    print(f'  So: "returns nothing better than its warm start" is true at '
          f'{last_dead}s, not at {first_alive}s;')
    print(f'      "the policy leads it" is still true at {first_alive}s on '
          f'{len(ahead_alive)}/{len(SETUP)}.')

out = [
    f'\\newcommand{{\\AnyDeadSec}}{{{last_dead}}}'
    '  % largest budget at which CP-SAT improved on its warm start on ZERO instances'
    ' of every setup-bearing regime (s)',
    f'\\newcommand{{\\AnyAliveSec}}{{{first_alive}}}'
    '  % ... and the smallest at which it improved on any',
    f'\\newcommand{{\\AnyRungs}}{{{len(have)}}}'
    f'  % budgets measured: {", ".join(f"{b}s" for b in have)}',
    f'\\newcommand{{\\AnyDeadAhead}}{{{len(ahead)}}}'
    f'\\newcommand{{\\AnySetupN}}{{{len(SETUP)}}}'
    '  % setup-bearing regimes where the policy leads the credited solver at AnyDeadSec',
    f'\\newcommand{{\\AnyAliveAhead}}{{{len(ahead_alive)}}}'
    '  % ... and where it STILL leads at AnyAliveSec, after the search has begun to bite',
]
if ahead_alive:
    out.append(
        f'\\newcommand{{\\AnyAliveLeadLo}}'
        f'{{{min(lead[(first_alive, r)] for r in ahead_alive):+.2f}}}'
        f'\\newcommand{{\\AnyAliveLeadHi}}'
        f'{{{max(lead[(first_alive, r)] for r in ahead_alive):+.2f}}}'
        '  % the policy\'s remaining lead at AnyAliveSec, min and max (%)')
if first_alive is not None:
    lo = min(imp[(first_alive, r)] for r in SETUP)
    hi = max(imp[(first_alive, r)] for r in SETUP)
    out.append(f'\\newcommand{{\\AnyAliveLo}}{{{lo}}}'
               f'\\newcommand{{\\AnyAliveHi}}{{{hi}}}'
               f'  % instances improved at AnyAliveSec, min and max over the setup regimes')
# A LaTeX control sequence is letters only, so the budget cannot go into the name as
# a digit: \Any1NImp is not a command, it is \Any followed by the text "1NImp", and it
# takes a hundred errors with it. Spell the budget.
WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty'}
for b in have:
    for r in REGIMES:
        out.append(f'\\newcommand{{\\Any{WORD[b]}{r}Imp}}{{{imp[(b, r)]}}}'
                   f'\\newcommand{{\\Any{WORD[b]}{r}Lead}}{{{lead[(b, r)]:+.2f}}}'
                   f'  % {r} at {b}s: instances improved, policy lead (%)')

with open('paper/macros_anytime.tex', 'w') as f:
    f.write('% auto-generated by scripts/p30_anytime.py -- do not hand-edit\n')
    f.write('% Fair-warm CP-SAT at every budget, on pinned cores whose isolation was\n')
    f.write('% proved by reproducing the published 5s rung instance by instance.\n')
    f.write('\n'.join(out) + '\n')
print('\nwrote paper/macros_anytime.tex')
