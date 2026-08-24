"""P34: composition when the policy has never seen a composition.

THE OBJECTION THIS ANSWERS. The paper's joint policy trains on six regimes
(N, L, S, W, LS, LW) and is read zero-shot on two (SW, LSW). That is a thinner test
than it sounds: every member of SW and of LSW appears during training both alone and
inside a pair, so the policy has already practised the pairwise interaction it needs,
and n=2 held-out points cannot support an interval on the composition axis anyway.
The hard version is this:

    train on {N, L, S, W} only, and test on every composition.

joint-nlsw is that policy. It has seen each rule ONLY IN ISOLATION. LS, LW, SW and
LSW are all genuinely unseen, so the held-out set doubles from two to four and every
member of it is a real extrapolation rather than a recombination of practised pairs.

WHAT WOULD FALSIFY THE PAPER'S CLAIM. If the ability to compose came from having
trained on LS and LW rather than from the construction (one descriptor, one bound
that is valid on every member and every composition), then joint-nlsw should fall
apart on the four compositions: behind the dispatching rules, behind the
specialists, far behind joint-v1. That is a real risk and this script is written to
report it if it happens.

SYMMETRY. Three seeds on BOTH sides, always. This project has already had to retract
a headline that came from three seeds on one side and one on the other; the
comparison is refused rather than run if a seed is missing.

Emits paper/macros_single.tex.

Usage: python scripts/p34_single_member.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402

rng = np.random.default_rng(7)          # the paper's bootstrap seed
MARGIN = 1.0                            # the paper's +-1% equivalence margin
B = 20000

# ONE CAMPAIGN, NOT TWO. joint-nlsw finished today; the joint policy and the
# specialists it is compared against were evaluated on 2026-07-12. The greedy rollout
# is deterministic in the instance, the descriptor and the evaluation environment, and
# it is NOT bit-reproducible across environments: an arg-max over near-equal action
# scores breaks the other way and the rollout goes elsewhere. On regime N that was
# worth up to eight hours on 32 of 100 instances, on identical inputs.
#
# Reading joint-nlsw from today's run and joint-v1 from July's would fold that
# difference into the measured effect of training on the single members only. So every
# artefact this script compares is read from 10x25_cpuref, in which all of them were
# evaluated by the same command. The published 10x25 results are not touched; they are
# simply not what a same-campaign comparison may be built from.
RES = 'test_results/FAMILY/10x25_cpuref'
SEEDS = ['', '-s302', '-s303']
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']

TRAINED = ['N', 'L', 'S', 'W']          # what joint-nlsw saw
HELDOUT = ['LS', 'LW', 'SW', 'LSW']     # what it did not
ALL = TRAINED + HELDOUT

NLSW = 'greedy+10x25+family+joint-nlsw'   # trained on the single members only
JOINT = 'greedy+10x25+family+joint-v1'    # the paper's joint: also saw LS and LW
WORD = {'LS': 'LS', 'LW': 'LW', 'SW': 'SW', 'LSW': 'LSW',
        'N': 'N', 'L': 'L', 'S': 'S', 'W': 'W'}


def arr(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def pooled(base, r):
    xs = [arr(f'{base}{s}', r) for s in SEEDS]
    if any(x is None for x in xs):
        missing = [s or '301' for s, x in zip(SEEDS, xs) if x is None]
        raise SystemExit(
            f'REFUSING TO REPORT: {base} on {r} is missing seed(s) {missing}. '
            f'The paper pools three seeds on both sides of every comparison. A '
            f'result computed from fewer seeds on one side than the other is the '
            f'exact asymmetry this project has already had to retract once.')
    return np.mean(xs, axis=0)


def best_rule(r):
    xs = []
    for rule in RULES:
        a = arr(rule, r)
        if a is None:
            raise SystemExit(f'REFUSING: no {rule} on {r}; the baseline is the '
                             f'per-instance best of {RULES}, not of whatever is '
                             f'on disk.')
        xs.append(a)
    return np.min(np.stack(xs), axis=0)


def ci(d):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(lo, hi):
    if hi < 0:
        return 'SUPERIOR'
    if -MARGIN < lo and hi < MARGIN:
        return 'EQUIVALENT'
    if hi < MARGIN:
        return 'non-inferior'
    if lo > MARGIN:
        return 'INFERIOR'
    return 'INCONCLUSIVE'


lines = []

# --- 1. the headline: does a policy that never saw a pair still beat the rules? ---
print('joint-nlsw vs the per-instance best of FIFO/MOR/SPT/MWKR')
print('  (%, negative = the policy is ahead)')
beat = 0
for r in ALL:
    p, b = pooled(NLSW, r), best_rule(r)
    m, lo, hi = ci(100.0 * (p - b) / b)
    if hi < 0:
        beat += 1
    tag = 'HELD OUT' if r in HELDOUT else 'trained '
    print(f'  {tag} {r:4s}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]'
          f'  {"AHEAD" if hi < 0 else "not ahead"}')
    lines.append(f'\\newcommand{{\\Nlsw{WORD[r]}Rule}}{{{m:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}RuleLo}}{{{lo:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}RuleHi}}{{{hi:+.2f}}}'
                 f'  % joint-nlsw vs best rule on {r} (%), CI')
held_beat = sum(1 for r in HELDOUT
                if ci(100.0 * (pooled(NLSW, r) - best_rule(r)) / best_rule(r))[2] < 0)
print(f'  ahead of the rules on {beat} of {len(ALL)} regimes, '
      f'{held_beat} of {len(HELDOUT)} of them held out\n')

# --- 2. the price of never having seen a pair: nlsw vs the paper's joint ---
# joint-v1 trained on LS and LW, so on those two it has an advantage that is NOT
# zero-shot. On SW and LSW both policies are zero-shot, and that is the fair cell.
print('joint-nlsw vs joint-v1 (%, positive = nlsw is worse)')
gaps = {}
for r in ALL:
    a, b = pooled(NLSW, r), pooled(JOINT, r)
    m, lo, hi = ci(100.0 * (a - b) / b)
    gaps[r] = (m, lo, hi)
    note = ''
    if r in ('LS', 'LW'):
        note = '  <- joint-v1 TRAINED on this; not a zero-shot comparison'
    elif r in ('SW', 'LSW'):
        note = '  <- both zero-shot: the fair cell'
    print(f'  {r:4s}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]  {verdict(lo, hi):12s}{note}')
    lines.append(f'\\newcommand{{\\Nlsw{WORD[r]}Joint}}{{{m:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}JointLo}}{{{lo:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}JointHi}}{{{hi:+.2f}}}'
                 f'  % joint-nlsw vs joint-v1 on {r} (%), CI')

# --- 3. vs the per-regime specialist, which is the paper's parity claim ---
print('\njoint-nlsw vs the per-regime specialist (%, positive = nlsw is worse)')
spec_equiv = 0
for r in ALL:
    a, b = pooled(NLSW, r), pooled(f'greedy+10x25+family+spec-{r}', r)
    m, lo, hi = ci(100.0 * (a - b) / b)
    v = verdict(lo, hi)
    if v in ('EQUIVALENT', 'SUPERIOR', 'non-inferior'):
        spec_equiv += 1
    print(f'  {r:4s}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]  {v}')
    lines.append(f'\\newcommand{{\\Nlsw{WORD[r]}Spec}}{{{m:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}SpecLo}}{{{lo:+.2f}}}'
                 f'\\newcommand{{\\Nlsw{WORD[r]}SpecHi}}{{{hi:+.2f}}}'
                 f'  % joint-nlsw vs the {r} specialist (%), CI')

zs = [gaps[r][0] for r in ('SW', 'LSW')]
print(f'\non the two cells where BOTH policies are zero-shot, joint-nlsw is '
      f'{np.mean(zs):+.2f}% from joint-v1 on average')

# How far the LSW interval actually reaches past the margin. A verdict of
# INCONCLUSIVE says only that the interval crosses the line; it does not say by how
# much, and the difference between crossing it by a hair and crossing it by a mile is
# the difference between a result and a failure. The number is emitted rather than
# described so that a rerun moves the sentence instead of contradicting it.
_lsw_hi = ci(100.0 * (pooled(NLSW, 'LSW') - pooled(f'greedy+10x25+family+spec-LSW',
                                                   'LSW'))
             / pooled('greedy+10x25+family+spec-LSW', 'LSW'))[2]
lines += [
    f'\\newcommand{{\\NlswLSWOver}}{{{_lsw_hi - MARGIN:.2f}}}'
    '  % how far past the +-1% margin the LSW interval reaches (percentage points)',
    f'\\newcommand{{\\NlswBeat}}{{{beat}}}'
    f'\\newcommand{{\\NlswRegimes}}{{{len(ALL)}}}'
    '  % regimes where joint-nlsw is ahead of the best dispatching rule',
    f'\\newcommand{{\\NlswHeldBeat}}{{{held_beat}}}'
    f'\\newcommand{{\\NlswHeldN}}{{{len(HELDOUT)}}}'
    '  % ... of them held out (LS, LW, SW, LSW: every composition)',
    f'\\newcommand{{\\NlswSpecEquiv}}{{{spec_equiv}}}'
    '  % regimes where joint-nlsw is equivalent to or better than the specialist',
]

emit('paper/macros_single.tex', lines,
     header=('% auto-generated by scripts/p34_single_member.py -- do not hand-edit\n'
             '% joint-nlsw saw each rule ONLY ALONE. Every composition is unseen.'))
print('\nwrote paper/macros_single.tex')
