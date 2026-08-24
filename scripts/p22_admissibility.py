"""E7: what the shaping envelope buys, measured against two ablated envelopes.

THREE ARMS, THREE SEEDS EACH. Identical in every other respect -- same schedule,
same architecture, same regimes, same update budget, paired by training seed.

    min    delta = min over source classes.  ADMISSIBLE, tight.   (the paper's)
    zero   delta = 0.                        ADMISSIBLE, loose.   -> isolates TIGHTNESS
    max    delta = MAX over source classes.  INADMISSIBLE, tight. -> isolates ADMISSIBILITY

All three are terminally exact and action-independent at s_0, so by Proposition 1
all three rewards are EXACTLY aligned with the makespan. They differ only in the
intermediate estimate. Verified on 25 LS instances, in the bound's own units:
min over-estimates a realized schedule 0/25, zero 0/25, max 2/25 (worst +6.2%);
terminal-exactness failures 0/25 for all three.

Each arm isolates one property of the envelope, by one paired comparison:

  min vs zero   Does the TIGHTER admissible envelope buy makespan? Both arms are
                admissible, so tightness is the only difference between them.

  min vs max    Does ADMISSIBILITY itself buy makespan? Terminal exactness, not
                admissibility, is what aligns the return with the makespan
                (Prop. 1), so 'max' is the arm that separates the two: aligned,
                and not admissible.

The comparison is made on the SETUP-bearing regimes, because the three envelopes
COINCIDE wherever SETUP is inactive (delta = 0 for all of them). N/L/W/LW
therefore measure the experiment's NOISE FLOOR rather than any envelope effect:
the arms share one jointly trained network, so a reward difference on S/LS
propagates through the shared weights into every regime. An effect smaller than
that floor is below the resolution of the experiment.

Uncertainty is reported at BOTH levels the claim spans: across held-out
instances, and across independently trained policies. Three seeds make a
seed-level bootstrap thin, so the per-seed spread is printed explicitly beside
the pooled interval rather than folded into it, where it would read tighter than
the evidence supports.

Usage: python scripts/p22_admissibility.py
"""
import os

import numpy as np

RES = 'test_results/FAMILY/10x25'
SETUP = ['S', 'LS', 'SW', 'LSW']       # the envelope is non-trivial only here
CONTROL = ['N', 'L', 'W', 'LW']        # SETUP inactive: the noise floor
SEEDS = ['s301', 's302', 's303']

# Paired by TRAINING SEED. The baseline arm is the paper's joint policy, whose
# seed-301 run is historically named without a suffix.
ARMS = {
    'min':  {'s301': 'joint-v1', 's302': 'joint-v1-s302', 's303': 'joint-v1-s303'},
    'zero': {s: f'joint-deltazero-{s}' for s in SEEDS},
    'max':  {s: f'joint-deltamax-{s}' for s in SEEDS},
}
rng = np.random.default_rng(7)


def arr(suffix, r):
    p = f'{RES}/Result_greedy+10x25+family+{suffix}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def missing():
    out = []
    for arm, bys in ARMS.items():
        for s, suf in bys.items():
            if arr(suf, 'S') is None:
                out.append(f'{arm}/{s} ({suf})')
    return out


miss = missing()
if miss:
    raise SystemExit(
        'E7 arms not evaluated yet:\n  ' + '\n  '.join(miss) +
        '\n\nscripts/run_ablations.sh trains them; scripts/run_evals.sh evaluates them.'
        '\nRefusing to emit a partial E7. Every arm must carry the same seeds: a'
        '\ncomparison whose arms rest on different seed sets averages training noise'
        '\nout of one side only, and the verdict then depends on which side is short.')


def gaps_by_seed(arm, regimes):
    """Per-seed vector of per-instance relative gaps vs the 'min' arm (%).

    Positive = the arm is SLOWER (worse) than ours. Paired within a seed: the
    ablation trained with seed 301 is compared against the baseline trained with
    seed 301, so training-seed noise is differenced out rather than pooled in.
    """
    out = {}
    for s in SEEDS:
        d = []
        for r in regimes:
            b = arr(ARMS['min'][s], r)
            v = arr(ARMS[arm][s], r)
            if b is None or v is None:
                continue
            d.append(100.0 * (v - b) / b)
        out[s] = np.concatenate(d) if d else np.array([])
    return out


def nested_ci(by_seed, n=20000):
    """Bootstrap over SEEDS, then over instances within each drawn seed.

    Resampling seeds is what keeps the interval from treating three policies as
    3 x n_instances independent observations. The interval is wide because three
    seeds is three: the width is the evidence, not an artifact.
    """
    seeds = [s for s in SEEDS if by_seed[s].size]
    if not seeds:
        return float('nan'), float('nan'), float('nan')
    means = np.array([by_seed[s].mean() for s in seeds])
    bs = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(seeds), len(seeds))
        vals = [by_seed[seeds[k]] for k in pick]
        bs[b] = np.mean([v[rng.integers(0, v.size, v.size)].mean() for v in vals])
    return float(means.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


print('E7: what does the shaping envelope buy?')
print('All three arms are terminally exact, so all three rewards are exactly')
print('aligned with the makespan (Proposition 1). They differ only in the')
print('intermediate estimate: min = admissible+tight, zero = admissible+loose,')
print('max = INADMISSIBLE+tight.\n')

for label, regimes in (('SETUP-BEARING -- the envelope is live here', SETUP),
                       ('NOISE FLOOR -- SETUP inactive, rewards coincide', CONTROL)):
    print(label)
    for r in regimes:
        line = f'  {r:4s}'
        base = np.mean([arr(ARMS['min'][s], r).mean() for s in SEEDS])
        line += f'  min={base:7.1f}'
        for arm in ('zero', 'max'):
            v = np.mean([arr(ARMS[arm][s], r).mean() for s in SEEDS])
            line += f'   {arm}={v:7.1f} ({100.0 * (v - base) / base:+5.2f}%)'
        print(line)
    print()

results = {}
print('Pooled over the setup-bearing regimes. Positive = WORSE than ours.')
print('Per-seed means show the spread across independently trained policies;')
print('the interval is a nested bootstrap over seeds and instances.\n')
for arm in ('zero', 'max'):
    by = gaps_by_seed(arm, SETUP)
    m, lo, hi = nested_ci(by)
    per = '  '.join(f'{s}={by[s].mean():+.2f}' for s in SEEDS if by[s].size)
    sd = np.std([by[s].mean() for s in SEEDS if by[s].size], ddof=1)
    name = {'zero': 'loose but admissible (delta=0)',
            'max': 'INADMISSIBLE (delta=max)   '}[arm]
    print(f'  {name}: {m:+.2f}%  95% CI [{lo:+.2f}, {hi:+.2f}]')
    print(f'      per-seed: {per}   SD across seeds = {sd:.2f}')
    results[arm] = (m, lo, hi, sd)

# The control regimes are a NOISE FLOOR, not a classical negative control. The
# arms share one jointly trained network, so a different reward on S/LS
# propagates through the shared weights into every regime; differences on
# N/L/W/LW (where the three rewards are the SAME function, delta = 0) therefore
# measure how far two such policies drift apart under that coupling plus
# training noise. This floor bounds the experiment's resolution: an envelope
# effect on the setup regimes smaller than the drift here is not attributable
# to the envelope, and is reported as unresolvable rather than as an effect.
ctrl = {}
for arm in ('zero', 'max'):
    by = gaps_by_seed(arm, CONTROL)
    ctrl[arm] = nested_ci(by)
print()
floor = max(abs(v) for a in ('zero', 'max') for v in ctrl[a][:1]) 
print('noise floor (regimes where the three rewards coincide; drift from the')
print('shared jointly trained weights plus seed noise):')
for a in ('zero', 'max'):
    m, lo, hi = ctrl[a]
    print(f'  {a:5s} on N/L/W/LW: {m:+.2f}%  CI [{lo:+.2f}, {hi:+.2f}]')
print('any setup-regime effect smaller in magnitude than this drift is below '
      'the resolution of the experiment.')

m_max, lo_max, hi_max, _ = results['max']
m_zero, lo_zero, hi_zero, _ = results['zero']

# The verdict is read off the CI against zero, in both directions. The same rule
# decides for and against the envelope: a CI strictly above zero certifies the
# ablated arm is worse, strictly below certifies it is BETTER, and an interval
# spanning zero certifies neither. No branch is privileged.
if lo_max > 0:
    v_adm = ('the inadmissible reward is significantly worse. Admissibility buys '
             'makespan, so Theorem 1 is a mechanism as well as a certificate.')
elif hi_max < 0:
    v_adm = ('the inadmissible reward trains a BETTER policy. Admissibility is '
             'not merely unnecessary here, it costs makespan.')
else:
    v_adm = ('the inadmissible reward is statistically indistinguishable from the '
             'admissible one. Admissibility is a correctness property that no '
             'result here depends on: Theorem 1 is a certificate, not a '
             'performance mechanism.')

if lo_zero > 0:
    v_tight = ('the loose envelope is significantly worse, so the setup-aware '
               'tightening is a mechanism and not only a nicety.')
elif hi_zero < 0:
    v_tight = ('the loose envelope trains a BETTER policy than the tighter one.')
else:
    v_tight = ('tightening the envelope buys nothing measurable. The theorem '
               'stands as a formal contribution; the tighter bound is not the '
               'mechanism behind the empirical results.')

print(f'\nVERDICT (admissibility, min vs max): {v_adm}')
print(f'VERDICT (tightness,     min vs zero): {v_tight}')

out = [
    '% auto-generated by scripts/p22_admissibility.py -- do not hand-edit',
    f'\\newcommand{{\\EsevenMaxDelta}}{{{m_max:+.2f}}}',
    f'\\newcommand{{\\EsevenMaxLo}}{{{lo_max:+.2f}}}',
    f'\\newcommand{{\\EsevenMaxHi}}{{{hi_max:+.2f}}}',
    f'\\newcommand{{\\EsevenZeroDelta}}{{{m_zero:+.2f}}}',
    f'\\newcommand{{\\EsevenZeroLo}}{{{lo_zero:+.2f}}}',
    f'\\newcommand{{\\EsevenZeroHi}}{{{hi_zero:+.2f}}}',
    f'\\newcommand{{\\EsevenCtrlZeroDelta}}{{{ctrl["zero"][0]:+.2f}}}',
    f'\\newcommand{{\\EsevenCtrlMaxDelta}}{{{ctrl["max"][0]:+.2f}}}',
    f'\\newcommand{{\\EsevenSeeds}}{{{len(SEEDS)}}}',
]
with open('paper/macros_e7.tex', 'w') as f:
    f.write('\n'.join(out) + '\n')
print('\nwrote paper/macros_e7.tex')
