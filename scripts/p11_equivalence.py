"""Formal equivalence testing for the parity claims (E1, E2).

Paired bootstrap (20k resamples, rng seed 7) of the mean per-instance relative
makespan gap, against a +-1% margin. Emits every Eq* macro the manuscript prints.

SYMMETRY IS THE INVARIANT. Both sides of every comparison are pooled over the
SAME seed set, per instance, and the script REFUSES to certify a comparison whose
two sides had different seed pools. This is not a formality. Pooling three seeds
on one side against one seed on the other averages the noise out of that side
only, and the verdicts move: it is enough to turn a breach of the equivalence
margin into an apparent pass. A verdict that depends on the estimator is not a
verdict. The script prints how many seeds each side actually had, so an
asymmetric run is visible rather than silent.

THE SUPERIORITY VERDICTS ARE ONE FAMILY. An equivalence verdict asserts a
pre-specified margin; a SUPERIOR verdict is a discovery, and the paper makes
more than one. E3 Holm-corrects its fifteen Wilcoxon tests, and the headline
claims cannot be held to a lower standard than the mechanism probes: every
certified SUPERIOR verdict carries a two-sided bootstrap p from the same draws
as its interval, Holm-adjusted across the family of such claims. The
equivalence verdicts are margin assertions, not discoveries, and stay
uncorrected.

Pooling three seeds narrows every interval toward the pooled conclusion, so
each certified verdict is also recomputed one seed at a time, single seed on
BOTH sides, and the seed least favorable to the claim is emitted next to it.
The pooled verdict stays the verdict of record; the worst seed shows how much
of it the pooling bought.

Usage: python scripts/p11_equivalence.py
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

rng = np.random.default_rng(7)
RES = 'test_results/FAMILY/10x25'
MARGIN = 1.0
SEED_SUFFIXES = ['', '-s302', '-s303']


def arr(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def pooled(base, r):
    """Per-instance mean over every seed replicate of `base` that exists.
    Returns (vector, n_seeds_used)."""
    arrs = [arr(f'{base}{s}', r) for s in SEED_SUFFIXES]
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None, 0
    return np.mean(arrs, axis=0), len(arrs)


def ci(d, B=20000):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    # Two-sided bootstrap p for the superiority family, from the SAME draws as
    # the interval so the p and the CI cannot disagree about the data. Add-one
    # in the Davison-Hinkley convention: a resampling p of exactly zero would
    # assert more than B draws can know, so the floor is 2/(B+1).
    p = 2.0 * (1 + min(int((bs <= 0).sum()), int((bs >= 0).sum()))) / (B + 1)
    return float(d.mean()), float(lo), float(hi), min(1.0, p)


def verdict(lo, hi):
    # Order matters: SUPERIOR must be tested BEFORE non-inferiority. hi < 0
    # implies hi < MARGIN, so testing non-inferiority first would make SUPERIOR
    # unreachable and downgrade a strict win (the whole CI below zero) to a mere
    # "non-inferior".
    if hi < 0:
        return 'SUPERIOR'
    if -MARGIN < lo and hi < MARGIN:
        return 'EQUIVALENT'
    if hi < MARGIN:
        return 'non-inferior'
    # A CI lying entirely beyond the margin is not "inconclusive": the
    # comparison is decided, against the first policy, and the verdict must
    # say so rather than hide behind uncertainty language.
    if lo > MARGIN:
        return 'INFERIOR'
    return 'INCONCLUSIVE'


RESULTS = {}
BASES = {}
ASYM = []


def compare(label, a_base, b_base, r, key=None):
    a, na = pooled(a_base, r)
    b, nb = pooled(b_base, r)
    if a is None or b is None:
        print(f'  {label:22s} {r:4s}  MISSING DATA')
        ASYM.append(f'{label} {r}: MISSING')
        return
    sym = 'symmetric' if na == nb else f'*** ASYMMETRIC {na} vs {nb} seeds ***'
    if na != nb:
        ASYM.append(f'{label} {r}: {na} vs {nb} seeds')
    m, lo, hi, p = ci(100 * (a - b) / b)
    v = verdict(lo, hi)
    print(f'  {label:22s} {r:4s} {m:+6.2f}%  CI [{lo:+5.2f},{hi:+5.2f}]  '
          f'{v:13s} ({na}v{nb} seeds, {sym})')
    RESULTS[key or (label, r)] = (m, lo, hi, v, na, nb, p)
    BASES[key or (label, r)] = (a_base, b_base, r)


# PROTAGONIST SWAP (2026-08-11, author-approved): the measured policy of
# E1/E2 is the RELEASED token-blind joint, so the tables show the artifact a
# plant would deploy. The conditioned joint remains the evaluated pathway:
# the E2blind rows below keep their published sign convention (conditioned
# minus released, positive = the conditioned variant is worse).
J = 'greedy+10x25+family+jointblind-v1'   # the released, measured policy
C = 'greedy+10x25+family+joint-v1'        # the conditioned evaluated pathway
E1_REG = ['N', 'L', 'S', 'W', 'LS', 'LW']

print(f'Paired-bootstrap 95% CIs of the relative makespan gap, margin +-{MARGIN:g}%')
print('Negative = the first policy is FASTER. Both sides pooled over the same seeds.\n')

print('E1: joint vs per-regime specialist')
for r in E1_REG:
    compare('joint vs spec', J, f'greedy+10x25+family+spec-{r}', r, ('E1', r))

print('\nE2: zero-shot joint on held-out compositions')
for r in ['SW', 'LSW']:
    compare('joint vs spec', J, f'greedy+10x25+family+spec-{r}', r, ('E2', r))
for r in ['SW', 'LSW']:
    compare('conditioned vs released', C, J, r, ('E2blind', r))

# The dispatching-rule gap uses the SAME estimator as everything above: the mean
# of the per-instance relative gaps. A paired mean-of-ratios and a ratio-of-means
# are different quantities, and quoting them in one sentence with the same units
# would make the comparisons incommensurable.
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
for r in ['SW', 'LSW']:
    j, nj = pooled(J, r)
    if j is None:
        continue
    pdrs = {k: arr(k, r) for k in RULES}
    pdrs = {k: v for k, v in pdrs.items() if v is not None}
    if not pdrs:
        continue
    best = min(pdrs, key=lambda k: pdrs[k].mean())
    m, lo, hi, p = ci(100 * (pdrs[best] - j) / j)   # positive = the rule is WORSE
    print(f'  {"joint vs best PDR":22s} {r:4s} {m:+6.2f}%  CI [{lo:+5.2f},{hi:+5.2f}]  '
          f'(best rule = {best})')
    RESULTS[('E2pdr', r)] = (m, lo, hi, best, nj, 1, p)

# Per-seed robustness of the SW superiority verdict. The pooled bootstrap is the
# estimator of record, but a verdict resting on one seed of three is worth
# knowing about, so the per-seed counts are reported alongside it.
from scipy.stats import wilcoxon                            # noqa: E402
PERSEED = {}
for r in ('SW', 'LSW'):
    spec, _ = pooled(f'greedy+10x25+family+spec-{r}', r)
    if spec is None:
        continue
    wins = sig = 0
    worst_p = 0.0
    for s in SEED_SUFFIXES:
        a = arr(f'{J}{s}', r)
        if a is None:
            continue
        if a.mean() < spec.mean():
            wins += 1
        p = wilcoxon(a, spec, alternative='less').pvalue
        if p < 0.05:
            sig += 1
        worst_p = max(worst_p, p)
    PERSEED[r] = (wins, sig, worst_p)
    print(f'  per-seed {r}: joint beats specialist on {wins}/3 seeds in mean; '
          f'{sig}/3 clear per-seed significance (worst p = {worst_p:.2f})')


# ------------------------------------------------- superiority family (Holm)
# One family, every certified SUPERIOR verdict a member, whichever comparison
# produced it. Correcting E3's fifteen probes while each headline claim keeps
# its own unadjusted interval would grant the strongest claims the weakest
# test. Holm rather than Bonferroni for the same reason as p17: identical
# family-wise control, uniformly more powerful, no independence assumption
# (the claims share the joint policy's rollouts, so they are correlated).
def holm(ps):
    """Holm-Bonferroni adjusted p-values, order preserved (mirrors p17)."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


SUPER_KEYS = [k for k, x in RESULTS.items()
              if k[0] in ('E1', 'E2', 'E2blind') and x[3] == 'SUPERIOR']
HOLM = dict(zip(SUPER_KEYS, holm([RESULTS[k][6] for k in SUPER_KEYS])))
if SUPER_KEYS:
    print(f'\nSuperiority family: {len(SUPER_KEYS)} certified claim(s), '
          f'Holm-Bonferroni as one family')
    for k in SUPER_KEYS:
        m, lo, hi = RESULTS[k][:3]
        ok = 'survives' if HOLM[k] < 0.05 else '*** DOES NOT SURVIVE ***'
        print(f'  {k[0]:8s} {k[1]:4s} CI [{lo:+5.2f},{hi:+5.2f}]  '
              f'raw p = {RESULTS[k][6]:.1e}  Holm p = {HOLM[k]:.1e}  '
              f'{ok} at 0.05')

# ---------------------------------------------------------- per-seed verdicts
# Single seed on BOTH sides, matched by seed index: pooling either side alone
# would average the noise out of that side only, the exact asymmetry this
# script refuses everywhere else. These draws come AFTER every pooled ci()
# call so the pooled intervals are bit-identical whether or not this block
# exists. The seed least favorable to the pooled claim is marked: for a
# SUPERIOR claim the binding bound is hi (it must stay below zero), for a
# containment claim it is the larger excursion toward either margin.
WORSTROWS = {}
print('\nPer-seed verdicts (single seed on both sides; worst = least '
      'favorable to the pooled claim)')
for k in RESULTS:
    if k not in BASES:
        continue
    ab, bb, r = BASES[k]
    rows = []
    for s in SEED_SUFFIXES:
        a, b = arr(f'{ab}{s}', r), arr(f'{bb}{s}', r)
        if a is None or b is None:
            continue
        m, lo, hi, _ = ci(100 * (a - b) / b)
        rows.append(((s or '-s301').lstrip('-'), m, lo, hi, verdict(lo, hi)))
    if not rows:
        continue
    WORSTROWS[k] = rows
    if RESULTS[k][3] == 'SUPERIOR':
        w = max(rows, key=lambda t: t[3])
    else:
        w = max(rows, key=lambda t: max(t[3], -t[2]))
    for s, m, lo, hi, v in rows:
        mark = '   <- worst seed' if s == w[0] else ''
        print(f'  {k[0]:8s} {k[1]:4s} {s:5s} {m:+6.2f}%  '
              f'CI [{lo:+5.2f},{hi:+5.2f}]  {v:13s}{mark}')

# ---------------------------------------------------------------- macros
# The manuscript stores none of these numbers; it \input's what this bootstrap
# emits. That is the only way the printed verdict and the test behind it cannot
# drift apart.
e1 = [RESULTS[('E1', r)] for r in E1_REG if ('E1', r) in RESULTS]
out = ['% auto-generated by scripts/p11_equivalence.py -- do not hand-edit']
if len(e1) == len(E1_REG):
    n_eq = sum(1 for x in e1 if x[3] == 'EQUIVALENT')
    n_sup = sum(1 for x in e1 if x[3] == 'SUPERIOR')
    n_ni = sum(1 for x in e1 if x[3] in ('EQUIVALENT', 'non-inferior', 'SUPERIOR'))
    worst = max(x[2] for x in e1)                 # worst (highest) upper bound
    out += [f'\\newcommand{{\\EqEoneEquiv}}{{{n_eq}}}'
            f'  % E1 regimes formally EQUIVALENT, of {len(E1_REG)}',
            f'\\newcommand{{\\EqEoneSuperior}}{{{n_sup}}}'
            '  % E1 regimes where the JOINT beats its specialist',
            f'\\newcommand{{\\EqEoneAtLeast}}{{{n_eq + n_sup}}}'
            '  % E1 regimes where the joint matches OR beats its specialist',
            f'\\newcommand{{\\EqEoneNonInf}}{{{n_ni}}}'
            f'  % E1 regimes non-inferior, of {len(E1_REG)}',
            f'\\newcommand{{\\EqEoneTotal}}{{{len(E1_REG)}}}',
            f'\\newcommand{{\\EqEoneWorstHi}}{{{worst:+.2f}}}'
            '  % worst E1 upper CI bound, %']
    for r, x in zip(E1_REG, e1):
        # Signed for a table column; unsigned for a sentence that already says
        # "ahead by" or "trails by", which would otherwise print "ahead by
        # -0.41%". The sign is flipped in exactly one place, here.
        out += [f'\\newcommand{{\\EqEone{r}Delta}}{{{x[0]:+.2f}}}'
                f'\\newcommand{{\\EqEone{r}DeltaAbs}}{{{abs(x[0]):.2f}}}'
                f'\\newcommand{{\\EqEone{r}Lo}}{{{x[1]:+.2f}}}'
                f'\\newcommand{{\\EqEone{r}Hi}}{{{x[2]:+.2f}}}  % {x[3]}']
# Macro names follow the manuscript's existing casing (\EqSwLo, \EqLswBlindHi).
# Every one of these is the SAME estimator: the mean per-instance relative gap.
NAME = {('E2', 'SW'): 'EqSw', ('E2', 'LSW'): 'EqLsw',
        ('E2blind', 'SW'): 'EqSwBlind', ('E2blind', 'LSW'): 'EqLswBlind',
        ('E2pdr', 'SW'): 'EqSwPdr', ('E2pdr', 'LSW'): 'EqLswPdr'}
for k, mac in NAME.items():
    if k in RESULTS:
        m, lo, hi, v = RESULTS[k][:4]
        out += [f'\\newcommand{{\\{mac}Delta}}{{{abs(m):.1f}}}'
                f'\\newcommand{{\\{mac}Lo}}{{{lo:+.2f}}}'
                f'\\newcommand{{\\{mac}Hi}}{{{hi:+.2f}}}  % {v}']

# The superiority family. Both the unadjusted and the Holm-adjusted p are
# emitted, so the manuscript can print the correction next to the claim
# instead of asking the reader to trust that one was done. The family size
# is a result too: it moves when a rerun moves a verdict across zero.
out += [f'\\newcommand{{\\EqSuperFamilyN}}{{{len(SUPER_KEYS)}}}'
        '  % certified SUPERIOR verdicts, corrected as ONE family']
if SUPER_KEYS:
    out += [f'\\newcommand{{\\EqSuperHolmSig}}{{{sum(1 for k in SUPER_KEYS if HOLM[k] < 0.05)}}}'
            '  % superiority claims surviving Holm at 0.05']
    for k in SUPER_KEYS:
        base = f'EqEone{k[1]}' if k[0] == 'E1' else NAME[k]
        out += [f'\\newcommand{{\\{base}P}}{{{RESULTS[k][6]:.1e}}}'
                f'\\newcommand{{\\{base}HolmP}}{{{HOLM[k]:.1e}}}'
                f'  % two-sided bootstrap p, raw and Holm over '
                f'{len(SUPER_KEYS)} claim(s)']

# Worst-seed robustness of the verdicts the paper leans on. Containment is
# checked on the bounds rather than through verdict(): verdict() reports a CI
# below zero as SUPERIOR before it tests the margin, and a seed whose CI is
# below zero yet inside the band still upholds the equivalence.
e1_eq = [('E1', r) for r in E1_REG
         if ('E1', r) in RESULTS and RESULTS[('E1', r)][3] == 'EQUIVALENT']
if e1_eq and all(k in WORSTROWS for k in e1_eq):
    n_ws = sum(1 for k in e1_eq
               if all(-MARGIN < lo and hi < MARGIN
                      for _, _, lo, hi, _ in WORSTROWS[k]))
    out += [f'\\newcommand{{\\EqEoneWorstSeedEquiv}}{{{n_ws}}}'
            f'  % of the {len(e1_eq)} EQUIVALENT E1 regimes, how many stay '
            'inside the margin under their least favorable single seed']
if ('E2', 'SW') in WORSTROWS:
    # Least favorable to the SW superiority claim specifically, so the worst
    # seed here is always the one with the highest upper bound.
    w = max(WORSTROWS[('E2', 'SW')], key=lambda t: t[3])
    out += [f'\\newcommand{{\\EqSwWorstSeedLo}}{{{w[2]:+.2f}}}'
            f'\\newcommand{{\\EqSwWorstSeedHi}}{{{w[3]:+.2f}}}'
            f'  % SW vs its specialist under the least favorable single seed '
            f'({w[0]}): {w[4]}']
for r, (wins, sig, worst_p) in PERSEED.items():
    tag = 'Sw' if r == 'SW' else 'Lsw'
    out += [f'\\newcommand{{\\{tag}WinSeeds}}{{{wins}}}'
            f'\\newcommand{{\\{tag}SigSeeds}}{{{sig}}}'
            f'\\newcommand{{\\{tag}WorstSeedP}}{{{worst_p:.2f}}}'
            f'  % per-seed robustness of the {r} verdict']

# Every number below is a RESULT, so it is emitted here rather than typed into
# the manuscript: a rerun moves it, and nothing else would recompute it.

# 1. The joint's 3-seed means on the held-out compositions (Table III footnote).
for r in ('SW', 'LSW'):
    v, n = pooled(J, r)
    if v is not None and n == 3:
        tag = 'Esw' if r == 'SW' else 'Elsw'
        out += [f'\\newcommand{{\\{tag}Joint}}{{{v.mean():.1f}}}'
                f'  % {r} joint, mean over {n} seeds']

# 2. The worst cross-seed spread. This is load-bearing: the protocol justifies the
#    +-\EqMargin% equivalence margin by saying it sits just above the policy's own
#    cross-seed spread. If a rerun widens the spread past the margin, that
#    sentence stops being true, so the number it rests on must come from the data.
#    The margin is a PERCENT, so the spread it is measured against is emitted as
#    a percent of the same regime's mean; the hours figure stays for the
#    replanning-granularity argument, which really is about hours.
spreads, spreads_pct = [], []
for r in ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW'):
    ms = [arr(f'{J}{s}', r) for s in SEED_SUFFIXES]
    ms = [a.mean() for a in ms if a is not None]
    if len(ms) == 3:
        sd = float(np.std(ms, ddof=1))
        spreads.append(sd)
        spreads_pct.append(100.0 * sd / float(np.mean(ms)))
if spreads:
    out += [f'\\newcommand{{\\SeedStdMax}}{{{max(spreads):.1f}}}'
            f'  % worst cross-seed sd of the joint, h (over 8 regimes)',
            f'\\newcommand{{\\SeedStdMaxPct}}{{{max(spreads_pct):.2f}}}'
            f'  % worst cross-seed sd of the joint, % of that regime mean']

# 3. The non-best dispatching rules' deficit on LSW (E2 prose quotes the range).
lsw_j, nj = pooled(J, 'LSW')
if lsw_j is not None and nj == 3:
    rest = []
    pdr = {m: arr(m, 'LSW') for m in ('FIFO', 'MOR', 'SPT', 'MWKR')}
    pdr = {m: a.mean() for m, a in pdr.items() if a is not None}
    if pdr:
        best = min(pdr.values())
        rest = sorted(100 * (v - lsw_j.mean()) / lsw_j.mean()
                      for v in pdr.values() if v > best)
    if rest:
        out += [f'\\newcommand{{\\ElswPdrRestLo}}{{{rest[0]:.0f}}}'
                f'\\newcommand{{\\ElswPdrRestHi}}{{{rest[-1]:.0f}}}'
                f'  % LSW: the non-best rules trail the joint by '
                f'{rest[0]:.1f}-{rest[-1]:.1f}%']

# REFUSE BEFORE WRITING. refuse() must run before emit(), never after: a
# generator that declines to certify has to leave NOTHING behind. A file
# containing only a header is still a file -- every \Eq* macro in it falls
# through to the \providecommand '??' fallback, so the paper renders '?? of ??
# regimes' and still compiles, whatever exit code this script returned.
from macro_io import emit, refuse
if ASYM:
    refuse('paper/macros_eq.tex',
           'asymmetric or missing comparisons:\n   ' + '\n   '.join(ASYM))
emit('paper/macros_eq.tex', out)
