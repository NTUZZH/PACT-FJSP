"""P27: what does a NEWLY trained joint policy do, as opposed to the three we trained?

E1 reports paired bootstrap intervals over the 100 test instances, with both sides
pooled over the same three training seeds. That interval answers a real question:
for the policies we trained, is the gap to the specialist inside the margin on this
benchmark? It does NOT answer the question a plant actually asks, which is whether
the ONE policy it trains tomorrow will land inside the margin. The first question
treats the test instance as the unit of inference. The second treats the TRAINING
RUN as the unit, and we have three of those, not a hundred.

This script computes the second one and refuses to dress it up. It reports, per
regime:

  * the gap under each individual seed, joint-s vs specialist-s, SYMMETRICALLY --
    seed 301's joint against seed 301's specialist, never against a pooled one.
    Pooling one side and not the other is the exact error that put a false SW
    result in an earlier figure, and it is not repeated here.
  * a two-level bootstrap interval: resample the three seeds with replacement
    (outer), then the hundred instances with replacement (inner). This propagates
    training-run variability into the interval.
  * the same margin verdicts at +-0.5%, +-1% and +-2%, because a verdict that only
    holds at the margin its authors chose is a verdict about the margin.

WHAT THIS SCRIPT MAY NOT PRETEND. A bootstrap over three points is a bootstrap over
three points. The outer resample can only ever draw from {s301, s302, s303}, so the
interval it produces is a coarse, conservative gesture at seed-level uncertainty and
not a proper estimate of it. Three seeds cannot certify a seed-level claim, and the
paper must not say they can. What the three seeds CAN do is show the reader the
spread directly, which is why the per-seed gaps are emitted as macros and printed as
a table: three honest numbers beat one dishonest interval. If the seed-level interval
is wide, the correct conclusion is not "the policy is unreliable" and not "the claim
survives" -- it is that the instance-level verdict is scoped to the policies we
trained, and the seed-level question is open at n=3.

Emits macros_seed.tex.

Usage: python scripts/p27_seed_inference.py     (CPU only; safe beside GPU training)
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

rng = np.random.default_rng(11)
RES = 'test_results/FAMILY/10x25'
SEEDS = ['', '-s302', '-s303']          # s301 is the unsuffixed one
SEED_NAMES = ['s301', 's302', 's303']
E1 = ['N', 'L', 'S', 'W', 'LS', 'LW']   # the six the joint trains on
E2 = ['SW', 'LSW']                      # held out; specialist exists for both
# PROTAGONIST SWAP (2026-08-11): seed-level inference for the released blind joint
JOINT = 'greedy+10x25+family+jointblind-v1'
SPEC = 'greedy+10x25+family+spec-'
MARGINS = [0.5, 1.0, 2.0]
B = 20000


def arr(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def per_seed_gaps(regime):
    """[(seed_name, gap_vector_over_instances)] -- joint-s vs specialist-s.

    The pairing is by seed AND by instance. Both sides move together or the
    comparison is not paired, and an unpaired comparison here would be measuring
    seed noise plus instance noise while claiming to measure only the first.
    """
    out = []
    for suf, name in zip(SEEDS, SEED_NAMES):
        j = arr(f'{JOINT}{suf}', regime)
        s = arr(f'{SPEC}{regime}{suf}', regime)
        if j is None or s is None:
            continue
        out.append((name, 100.0 * (j - s) / s))
    return out


def verdict(lo, hi, margin):
    if hi < 0:
        return 'SUPERIOR'
    if -margin < lo and hi < margin:
        return 'EQUIVALENT'
    if hi < margin:
        return 'non-inf'
    if lo > margin:
        return 'INFERIOR'
    return 'INCONCL'


MACROS = re.compile(r'\\newcommand\{\\(Eq(?:Eone|Sw|Lsw)\w*?)\}\{([-+0-9.]+)\}')


def paper_intervals():
    """E1/E2's published intervals, READ from the macros the paper prints.

    Not recomputed. p11_equivalence.py already owns this number, and a second
    script that recomputes it will disagree with the first in the second decimal --
    a different bootstrap seed is enough (-0.28 here against the paper's -0.29).
    A supplement table that contradicts the main text by 0.01 is a defect a
    reader will find and an author cannot explain. So the estimator lives in one
    place, and this script reads it.

    p11 pools the three seeds' MAKESPANS per instance and then takes the ratio,
    treating the pooled policy as one artifact. The seed-level analysis below
    cannot do that: it needs each training run's own gap, so it takes the ratio
    first and averages after. mean(ratios) != ratio(means), and on W they differ
    (+1.42 against +1.46). That difference is real and is stated, not smoothed.
    """
    src = {}
    for f in ('paper/macros_eq.tex',):
        src.update(dict(MACROS.findall(open(f).read())))
    out = {}
    for r in E1 + E2:
        key = 'Eone' + r if r in E1 else ('Sw' if r == 'SW' else 'Lsw')
        d, lo, hi = (f'Eq{key}Delta', f'Eq{key}Lo', f'Eq{key}Hi')
        if d in src and lo in src and hi in src:
            out[r] = (float(src[d]), float(src[lo]), float(src[hi]))
    return out


def seed_level(gaps):
    """Two-level: resample SEEDS (outer), then instances within each (inner)."""
    k, n = len(gaps), len(gaps[0][1])
    mat = np.stack([g for _, g in gaps])                    # [k, n]
    draws = np.empty(B)
    for b in range(B):
        si = rng.integers(0, k, k)                          # outer: which seeds
        ii = rng.integers(0, n, n)                          # inner: which instances
        draws[b] = mat[si][:, ii].mean()
    return float(mat.mean()), *np.percentile(draws, [2.5, 97.5])


PAPER = paper_intervals()
rows, macros = [], []
print(f'{"reg":4} {"s301":>8} {"s302":>8} {"s303":>8} | '
      f'{"instance-level 95% CI":>26} | {"seed-level 95% CI":>24}')
print(f'{"":4} {"per-seed gap, joint - specialist (%)":^26} | '
      f'{"the paper (E1)":^26} | {"a NEW training run":^24}')
print('-' * 100)

for r in E1 + E2:
    gaps = per_seed_gaps(r)
    # A skipped regime does not announce itself in the macros. It just makes
    # \SeedEqInstance and \SeedEqSeedLevel smaller, while the sentence they feed still
    # says "of the six E1 equivalences", and the table quietly loses a row. Refuse.
    if len(gaps) < 3:
        raise SystemExit(
            f'REFUSING: regime {r} has only {len(gaps)} of the 3 seeds on disk. The '
            f'counts this script emits are counts OUT OF the full set of regimes; '
            f'dropping one silently shrinks the numerator and the denominator both.')
    if r not in PAPER:
        raise SystemExit(
            f'REFUSING: regime {r} has no published interval in macros_eq.tex. The '
            f'instance-level column is read from the paper rather than recomputed, '
            f'precisely so the two cannot drift; a regime missing from it cannot be '
            f'silently omitted from the comparison.')
    means = [float(g.mean()) for _, g in gaps]
    im, ilo, ihi = PAPER[r]          # the paper's own number, read not recomputed
    sm, slo, shi = seed_level(gaps)
    rows.append((r, means, (im, ilo, ihi), (sm, slo, shi)))
    print(f'{r:4} ' + ' '.join(f'{m:+8.2f}' for m in means) +
          f' | {im:+6.2f} [{ilo:+6.2f}, {ihi:+6.2f}] {verdict(ilo, ihi, 1.0):>10}'
          f' | {sm:+6.2f} [{slo:+6.2f}, {shi:+6.2f}] {verdict(slo, shi, 1.0):>10}')

print('-' * 100)
print('\nMargin sensitivity (instance-level, the paper\'s analysis):')
print(f'{"margin":>8} ' + ' '.join(f'{r:>9}' for r, *_ in rows))
for m in MARGINS:
    line = f'{m:>7.1f}% '
    for _, _, (_, lo, hi), _ in rows:
        line += f'{verdict(lo, hi, m):>9} '
    print(line)

print('\nMargin sensitivity (seed-level -- a new training run):')
print(f'{"margin":>8} ' + ' '.join(f'{r:>9}' for r, *_ in rows))
for m in MARGINS:
    line = f'{m:>7.1f}% '
    for _, _, _, (_, lo, hi) in rows:
        line += f'{verdict(lo, hi, m):>9} '
    print(line)

# ---- macros -----------------------------------------------------------------
n_eq_inst = sum(1 for r, _, (_, lo, hi), _ in rows
                if r in E1 and verdict(lo, hi, 1.0) == 'EQUIVALENT')
n_eq_seed = sum(1 for r, _, _, (_, lo, hi) in rows
                if r in E1 and verdict(lo, hi, 1.0) == 'EQUIVALENT')
widest = max((hi - lo) / (ihi - ilo)
             for _, _, (_, ilo, ihi), (_, lo, hi) in rows if ihi > ilo)

out = [
    '% auto-generated by scripts/p27_seed_inference.py -- do not hand-edit',
    '% The seed-level analysis that matters when the deployable artifact is ONE',
    '% training run, not the mean of three. Outer bootstrap over the three seeds, inner',
    '% over the hundred instances, both sides of every comparison pooled symmetrically.',
    '% n=3 cannot certify a seed-level claim; these macros exist so the paper can say so',
    '% with a number instead of a shrug.',
    f'\\newcommand{{\\SeedN}}{{{len(SEEDS)}}}'
    '  % independent training runs behind every E1/E2 verdict',
    f'\\newcommand{{\\SeedEqInstance}}{{{n_eq_inst}}}'
    f'\\newcommand{{\\SeedEqSeedLevel}}{{{n_eq_seed}}}'
    '  % E1 regimes EQUIVALENT at +-1%: instance-level, then seed-level',
    f'\\newcommand{{\\SeedWidenX}}{{{widest:.1f}}}'
    '  % how much wider the seed-level interval is than the instance-level one, worst regime',
]
TEX_VERDICT = {'SUPERIOR': r'\textsc{sup}', 'EQUIVALENT': r'\textsc{eq}',
               'non-inf': r'\textsc{n-inf}', 'INFERIOR': r'\textsc{inf}',
               'INCONCL': r'\textsc{inc}'}

# The VERDICT is emitted, not typeset from the interval in LaTeX. Computing it with
# an \ifdim on the macro would be elegant and could never drift from the interval
# printed beside it -- but a missing generator renders its macros as \textbf{??},
# and \ifdim\textbf{??}pt is a "Missing number" error. The paper's guard exists so
# that an un-run generator shows a conspicuous ?? instead of breaking the build, and
# a table that crashes the build instead would defeat it. Emitting the verdict keeps
# both properties: it cannot drift from its interval, because the same function
# computed both, and it degrades to ?? like everything else.
for r, means, (im, ilo, ihi), (sm, slo, shi) in rows:
    out.append(
        f'\\newcommand{{\\Sd{r}One}}{{{means[0]:+.2f}}}'
        f'\\newcommand{{\\Sd{r}Two}}{{{means[1]:+.2f}}}'
        f'\\newcommand{{\\Sd{r}Three}}{{{means[2]:+.2f}}}'
        f'\\newcommand{{\\Sd{r}Lo}}{{{slo:+.2f}}}'
        f'\\newcommand{{\\Sd{r}Hi}}{{{shi:+.2f}}}'
        f'  % {r}: the three per-seed gaps, then the seed-level interval (%)')
    for tag, (lo, hi) in (('Inst', (ilo, ihi)), ('Seed', (slo, shi))):
        for m, mtag in ((0.5, 'Half'), (1.0, 'One'), (2.0, 'Two')):
            out.append(
                f'\\newcommand{{\\Vd{r}{tag}{mtag}}}'
                f'{{{TEX_VERDICT[verdict(lo, hi, m)]}}}'
                f'  % {r}, {tag.lower()}-level verdict at +-{m}%')

with open('paper/macros_seed.tex', 'w') as f:
    f.write('\n'.join(out) + '\n')
print('\nwrote paper/macros_seed.tex')
print(f'\nE1 equivalences at +-1%: {n_eq_inst}/6 instance-level, '
      f'{n_eq_seed}/6 seed-level.')
print(f'The seed-level interval is up to {widest:.1f}x wider. That is what three '
      'training\nruns buy, and the paper should say which question each interval '
      'answers.')
