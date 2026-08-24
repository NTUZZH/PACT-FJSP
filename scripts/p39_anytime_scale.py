"""P39: where CP-SAT overtakes the policy as instances grow.

At 10x25 the fair-warm solver has passed the policy on all eight regimes by 30 s
(Sec. S-XII, p30_anytime.py). This asks the same question at 20x25 (440 ops) and
30x25 (660 ops): how far up the budget ladder does the crossover move as the
instances get larger?

For each (size, regime) it reports the CROSSOVER RUNG -- the smallest wall-clock
budget at which the credited CP-SAT mean makespan reaches or beats the policy
mean, credited by the Protocol's min(warm start, search) exactly as p30 does. If
no measured rung crosses, it reports 'not by <largest rung>'.

Everything about the CP-SAT campaign is the 10x25 protocol with only the instance
size changed: same best-of-four warm start, same crediting, same 8 workers x 2
jobs, same policy (greedy+10x25+family+joint-v1, three seeds, makespan column,
averaged). The policy and dispatching-rule numbers are REUSED from the E6
scale-transfer campaign; the policy is not re-evaluated here.

Emits paper/macros_anytime_scale.tex.

Usage: python scripts/p39_anytime_scale.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

OR = 'or_solution/FAMILY'
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SEEDS = ['', '-s302', '-s303']
# PROTAGONIST SWAP (2026-08-11): the scale frontier is the released blind joint's
JOINT = 'greedy+10x25+family+jointblind-v1'
RUNGS = [1, 2, 5, 10, 30, 60, 120]           # the ladder, low to high
# size tag -> (result dir, instances per regime, LaTeX word for the macro name)
SIZES = [
    ('10x25', 'test_results/FAMILY/10x25', 100, 'Ten'),
    ('20x25', 'test_results/FAMILY/20x25', 50, 'Twenty'),
    ('30x25', 'test_results/FAMILY/30x25', 20, 'Thirty'),
]
WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty', 60: 'Sixty',
        120: 'OneTwenty'}


# The repaired best-of-four warm start (scripts/p41_repaired_warm_cache.py),
# one cache per size. p30 credits the 10x25 solver against this hint, and p13
# reports the E5 table from the same convention; computing this ladder against
# the weaker textbook hint instead made the two disagree about the training
# size in print, E5 saying the 30 s solver had passed the policy on seven of
# eight regimes and E6 saying six. One convention, every size.
_REPAIRED = {
    size: json.load(open(f'{OR}/{d}/makespans.json'))
    for size, d in (('10x25', 'warmstart_cache_repaired'),
                    ('20x25', 'warmstart_cache_repaired_20x25'),
                    ('30x25', 'warmstart_cache_repaired_30x25'))
    if os.path.exists(f'{OR}/{d}/makespans.json')
}
# Tiers re-solved from the repaired hint rather than floored at it; only 10x25
# has them, and there warm_makespan already IS the repaired hint.
RESOLVED = {('10x25', 5), ('10x25', 30)}


def _rung_dir(size, budget):
    return (f'{OR}/{size}_fairwarmR{budget}s' if (size, budget) in RESOLVED
            else f'{OR}/{size}_fairwarm{budget}s')


def credited(size, budget, regime):
    """Per-instance credited CP-SAT makespan, exactly p30's rule: the better of
    the search result and the hint it was given, with the hint being the
    repaired warm start wherever a repaired cache exists for the size."""
    rep = _REPAIRED.get(size, {}).get(regime)
    out = []
    for p in sorted(glob.glob(f'{_rung_dir(size, budget)}/{regime}/*.json')):
        d = json.load(open(p))
        x = d[0] if isinstance(d, list) else d
        warm = x.get('warm_makespan')
        if warm is None:
            raise SystemExit(f'{p} has no warm_makespan: not a fair-warm run')
        best = min(x['objective'], warm)
        if (size, budget) not in RESOLVED and rep is not None:
            best = min(best, rep[x['instance']]['repaired_ms'])
        out.append(best)
    return np.array(out, dtype=float)


def policy(res_dir, regime):
    a = []
    for s in SEEDS:
        p = f'{res_dir}/Result_{JOINT}{s}_{regime}.npy'
        if not os.path.exists(p):
            return None
        x = np.load(p)
        a.append((x[:, 0] if x.ndim > 1 else x).astype(float))
    return np.mean(a, axis=0)


rng = np.random.default_rng(7)          # the paper's bootstrap seed


def paired_gap(pol, other):
    """p30's estimator: mean of the per-instance relative gaps, paired.
    Positive = solver slower than the policy (policy ahead)."""
    d = 100.0 * (other - pol) / pol
    idx = rng.integers(0, len(d), (20000, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def complete_rungs(size, n_inst):
    """Rungs where every regime has all n_inst solves on disk."""
    have = []
    for b in RUNGS:
        counts = [len(glob.glob(f'{_rung_dir(size, b)}/{r}/*.json'))
                  for r in REGIMES]
        if counts and all(c == n_inst for c in counts):
            have.append(b)
    return have


out = []
summary = {}                            # size_word -> dict of pooled facts
print(f'{"size":7}{"regime":7}{"crossover":>12}{"  policy":>10}'
      f'{"  cred@cross":>12}{"   gap%@cross":>14}')
print('-' * 64)
for size, res_dir, n_inst, sw in SIZES:
    have = complete_rungs(size, n_inst)
    if not have:
        print(f'{size}: no complete rungs yet -- skipping')
        continue
    maxr = max(have)
    crossed_by = {b: 0 for b in have}      # regimes crossed at or below rung b
    per_regime_cross = {}
    for r in REGIMES:
        pol = policy(res_dir, r)
        if pol is None:
            raise SystemExit(f'no policy results for {size}/{r} in {res_dir}')
        cross = None
        cred_at = gap_at = None
        for b in have:
            cred = credited(size, b, r)
            if len(cred) != len(pol):
                raise SystemExit(
                    f'{size}/{r} @ {b}s: {len(cred)} solves vs {len(pol)} '
                    f'policy instances -- refusing to compare misaligned sets')
            if cred.mean() <= pol.mean() and cross is None:
                cross = b
                cred_at = cred.mean()
                gap_at = paired_gap(pol, cred)[0]
        per_regime_cross[r] = cross
        # every rung at or above the crossover counts as crossed
        for b in have:
            if cross is not None and b >= cross:
                crossed_by[b] += 1
        ctxt = str(cross) if cross is not None else f'not by {maxr}'
        pol_m = pol.mean()
        ca = f'{cred_at:.1f}' if cred_at is not None else '   -'
        ga = f'{gap_at:+.2f}' if gap_at is not None else '     -'
        print(f'{size:7}{r:7}{ctxt:>12}{pol_m:>10.1f}{ca:>12}{ga:>14}')

        # per (size, regime) macros
        cross_val = cross if cross is not None else 0
        out.append(
            f'\\newcommand{{\\ScaleCross{sw}{r}}}{{{cross_val}}}'
            f'\\newcommand{{\\ScaleCross{sw}{r}Txt}}{{{ctxt}}}'
            f'  % {size} {r}: smallest budget (s) at which credited CP-SAT '
            f'reaches the policy; 0/"not by" = never within measured ladder')

    # pooled facts for this size
    n_beyond30 = sum(1 for r in REGIMES
                     if per_regime_cross[r] is None or per_regime_cross[r] > 30)
    n_crossed_max = sum(1 for r in REGIMES if per_regime_cross[r] is not None)
    crossed_vals = [per_regime_cross[r] for r in REGIMES
                    if per_regime_cross[r] is not None]
    all_cross = (max(crossed_vals) if n_crossed_max == len(REGIMES) else None)
    summary[sw] = dict(size=size, maxr=maxr, have=have,
                       n_beyond30=n_beyond30, n_crossed_max=n_crossed_max,
                       all_cross=all_cross)
    out.append(
        f'\\newcommand{{\\Scale{sw}MaxRung}}{{{maxr}}}'
        f'\\newcommand{{\\Scale{sw}NBeyondThirty}}{{{n_beyond30}}}'
        f'\\newcommand{{\\Scale{sw}NCrossedByMax}}{{{n_crossed_max}}}'
        f'  % {size}: rungs to {maxr}s; regimes not crossed by 30s; regimes '
        f'crossed by {maxr}s (of {len(REGIMES)})')
    all_txt = str(all_cross) if all_cross is not None else f'not by {maxr}'
    out.append(
        f'\\newcommand{{\\Scale{sw}AllCrossTxt}}{{{all_txt}}}'
        f'  % {size}: smallest budget by which ALL {len(REGIMES)} regimes are '
        f'crossed, or "not by" the largest rung')

print()
for sw, s in summary.items():
    allc = f'{s["all_cross"]}s' if s['all_cross'] is not None else f'> {s["maxr"]}s'
    print(f'{s["size"]}: rungs {s["have"]}; {s["n_beyond30"]}/8 regimes NOT '
          f'crossed by 30s; {s["n_crossed_max"]}/8 crossed by {s["maxr"]}s; '
          f'all-8 crossover = {allc}')

hdr = ('% auto-generated by scripts/p39_anytime_scale.py -- do not hand-edit\n'
       '% Fair-warm CP-SAT vs the joint policy at 20x25 and 30x25, same protocol\n'
       '% as the 10x25 anytime frontier (p30) with only the instance size changed.\n'
       '% Policy/rule numbers reused from the E6 scale-transfer campaign.\n')
with open('paper/macros_anytime_scale.tex', 'w') as f:
    f.write(hdr)
    f.write('\n'.join(out) + '\n')
print('\nwrote paper/macros_anytime_scale.tex')
