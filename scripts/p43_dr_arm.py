"""P43: does domain randomization buy anything the descriptor did not already?

The out-of-distribution study (P33) located a boundary: hold the constraint FORM
fixed, push its NUMBERS past the training range, and the policy's lead over the
dispatching rules shrinks (worst on the denser outage calendars, the W regime).
The natural counter-move is to train the policy on a WIDER slice of the same
family, so the shifted arms are no longer out of distribution. That is the DR arm:

  joint-dr-s{301,302,303}   trained with widened member-parameter ranges,
                            setup cross-magnitudes drawn asymmetric on (0.5, 2.0)
                            and outage density on (0.05, 0.35); everything else
                            (architecture, data, budget, seeds' role) identical to
                            joint-v1. It is evaluated into the SAME three campaigns
                            the OOD study reads.

Three questions, and each is a paired, per-instance, 3-seed-pooled comparison read
by the SAME conventions P33 uses (arr column-0, pooled() that refuses on a missing
seed, best-of-four rules, ci() with default_rng(7) and B=20000):

  1. Out where the shift bites, does the DR policy beat the best of four rules?
     Per arm, per regime the shift can reach: DR vs best rule, mean and CI.

  2. THE ANCHOR, asserted not hoped for. Recomputing joint-v1 vs the best rule from
     the same files must reproduce the published macros_ood cells to the printed
     precision. If it does not, this script is not reading the campaigns the way
     the OOD table did, and nothing below can be compared to it. Two cells are
     asserted: \\OodSetupLS and \\OodWindowW. The MEAN is rng-independent, so this
     anchor is exact regardless of bootstrap-draw order.

  3. THE PRICE. Widening the training distribution is not free: a policy asked to
     be good everywhere is rarely as good in the middle as one trained only there.
     DR minus joint-v1, paired per instance, both sides pooled over their three
     seeds, as a percent of joint-v1. Positive means DR costs more. The seed-301
     pilot put the in-distribution W-regime price near +1%.

Feasibility is counted from the validator's summary tables exactly as P33 does:
the .npy carries makespan and inference time, never feasibility, so a check of the
makespan column is a gate that cannot fail.

A lead whose bootstrap interval covers zero is not a lead: the verdict wording is
P33's, unchanged.

REFUSAL. Every cell is pooled over the same three seeds on both sides. If any DR
seed's files are missing the script refuses loudly and writes NOTHING, exactly as
the OOD table refuses: a table pooled over two seeds on one side and three on the
other is the asymmetry this project has already had to retract once.

Emits paper/macros_dr.tex.

Usage:
  python3 scripts/p43_dr_arm.py            once seeds 302 and 303 have landed
  python3 scripts/p43_dr_arm.py --pilot    seed 301 only; writes to /tmp, marks
                                            every line PILOT, never touches paper/
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402

PILOT = '--pilot' in sys.argv[1:]

rng = np.random.default_rng(7)          # the paper's bootstrap seed, unchanged
B = 20000
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']  # the same four best-of-four the OOD table uses

# joint-v1's three seeds are named the way the published campaigns name them: the
# first replicate carries the bare tag, the other two an explicit -s302/-s303.
JV1 = 'greedy+10x25+family+joint-v1'
JV1_SEEDS = ['', '-s302', '-s303']

# The DR arm names ALL THREE seeds explicitly: joint-dr-s301/-s302/-s303. There is
# no bare joint-dr tag, so its seed list is not joint-v1's.
DR = 'greedy+10x25+family+joint-dr'
DR_SEEDS = ['-s301', '-s302', '-s303']

# The pilot pools DR over seed 301 alone, and to keep the in-distribution price a
# MATCHED paired comparison it pools joint-v1 over its seed 301 alone too (the bare
# tag). This is a diagnostic, not a result: it goes to /tmp and every line says so.
if PILOT:
    DR_SEEDS = ['-s301']
    JV1_PAIR_SEEDS = ['']
else:
    JV1_PAIR_SEEDS = JV1_SEEDS

# (name, results dir, summary-file suffix, the regimes the arm's shift can reach).
# eval_family.py appends --tag to the SUMMARY name but not to the .npy name, so on
# the in-distribution rerun the arrays are Result_<model>_<regime>.npy while the
# validator tables are summary_<model>_cpuref.md. The OOD arms carry no suffix.
#   ind      : the in-distribution rerun (10x25_cpuref), all 8 regimes.
#   oodsetup : the setup bit is only active on S, LS, SW, LSW.
#   oodwindow: the window bit is only active on W, LW, SW, LSW.
ARMS = [
    ('ind',       'test_results/FAMILY/10x25_cpuref',    '_cpuref',
     ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']),
    ('oodsetup',  'test_results/FAMILY/10x25_oodsetup',  '',
     ['S', 'LS', 'SW', 'LSW']),
    ('oodwindow', 'test_results/FAMILY/10x25_oodwindow', '',
     ['W', 'LW', 'SW', 'LSW']),
]
WORD = {'ind': 'Ind', 'oodsetup': 'Setup', 'oodwindow': 'Window'}

# The two published OOD cells this script must reproduce exactly, as the anchor that
# proves it reads the campaigns the way the OOD table did. Both are MEANS, which do
# not depend on bootstrap-draw order, so the check is exact. Values are the printed
# contents of \OodSetupLS and \OodWindowW in paper/macros_ood.tex.
ANCHORS = [('oodsetup', 'LS', -0.17, 'OodSetupLS'),
           ('oodwindow', 'W', -3.10, 'OodWindowW')]


def arr(res, tag, r):
    p = f'{res}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def pooled(res, base, seeds, r, label):
    """Per-instance mean over the seed replicates. A missing seed is a hard error,
    not a thing to skip: pooling one side over fewer seeds than the other is the
    asymmetry this project has already had to retract once. Refuse loudly and,
    because emit() is the last thing this script does, leave nothing partial."""
    xs = [arr(res, f'{base}{s}', r) for s in seeds]
    if any(x is None for x in xs):
        missing = [s or '(bare)' for s, x in zip(seeds, xs) if x is None]
        raise SystemExit(
            f'REFUSING TO REPORT: {label} on {res} regime {r} is missing seed '
            f'file(s) {missing} for base {base!r}. Every cell in macros_dr.tex is '
            f'pooled over the same seeds on both sides; a partial pool is exactly '
            f'the asymmetry the OOD table refuses. No macros were written. '
            f'{"Drop --pilot only once " if not PILOT else ""}'
            f'seeds 302 and 303 have finished evaluating into all three campaigns, '
            f'then rerun: python3 scripts/p43_dr_arm.py')
    return np.mean(xs, axis=0)


def best_rule(res, r):
    """Per-instance best of the four dispatching rules. Missing any one is a hard
    error: the baseline the paper names is the best of four, and a baseline built
    from fewer rules is a weaker opponent than the one it claims to be."""
    xs = []
    for rule in RULES:
        a = arr(res, rule, r)
        if a is None:
            raise SystemExit(
                f'REFUSING TO REPORT: {res} has no {rule} on regime {r}. The '
                f'baseline is the best of {RULES}; a baseline built from fewer '
                f'rules would overstate the policy. Run the missing rules first.')
        xs.append(a)
    return np.min(np.stack(xs), axis=0)


def infeasible(res, tag, r, sfx=''):
    """The count of schedules the INDEPENDENT VALIDATOR rejected, read from the
    summary table, never from the .npy (which carries makespan and inference time,
    not feasibility). Refuse if the table or its row is missing."""
    p = f'{res}/summary_{tag.split("+", 1)[1]}{sfx}.md'
    if not os.path.exists(p):
        raise SystemExit(
            f'REFUSING TO REPORT feasibility: {p} does not exist, and the '
            f'validator\'s verdict lives nowhere else. The .npy carries makespan '
            f'and inference time, not feasibility.')
    for line in open(p):
        cells = [c.strip() for c in line.split('|')]
        if len(cells) > 5 and cells[1] == r and cells[2] == tag:
            ok, total = cells[5].split('/')          # the "feasible" column, "100/100"
            return int(total) - int(ok)
    raise SystemExit(f'REFUSING: no row for regime {r}, method {tag} in {p}')


def ci(d):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(lo, hi):
    """P33's wording, unchanged: a lead whose interval covers zero is not a lead."""
    if hi < 0:
        return 'policy AHEAD'
    if lo > 0:
        return 'policy BEHIND'
    return 'tied'


tag = 'PILOT (seed 301 only, NOT FOR PUBLICATION)' if PILOT else \
    'full 3-seed pool'
print(f'=== P43 domain-randomized arm, {tag} ===')
print(f'DR seeds: {[s or "(bare)" for s in DR_SEEDS]}   '
      f'joint-v1 seeds (anchor): {[s or "(bare)" for s in JV1_SEEDS]}\n')

# ---------------------------------------------------------------------------
# THE ANCHOR, ASSERTED. Recompute two published OOD cells from the same files by
# the same pipeline. joint-v1 is always on disk for all three seeds, so this runs
# in both modes and is the one thing the pilot verifies before we trust anything.
# ---------------------------------------------------------------------------
print('--- anchor: recomputing published OOD cells (joint-v1, 3-seed) ---')
for arm, r, want, name in ANCHORS:
    res = dict((a[0], a[1]) for a in ARMS)[arm]
    p = pooled(res, JV1, JV1_SEEDS, r, f'joint-v1 anchor {name}')
    b = best_rule(res, r)
    got = round(float((100.0 * (p - b) / b).mean()), 2)
    status = 'OK' if got == want else 'MISMATCH'
    print(f'  \\{name}: macros_ood prints {want:+.2f}, recomputed {got:+.2f}  [{status}]')
    if got != want:
        raise SystemExit(
            f'ANCHOR FAILED: \\{name} recomputes to {got:+.2f}, but macros_ood.tex '
            f'prints {want:+.2f}. This script is not reading the {arm} campaign the '
            f'way p33_ood_table.py did; no DR number below can be compared to the '
            f'OOD table. Stop and reconcile before emitting anything.')
print('  anchor holds: this script reads the campaigns as the OOD table does\n')

# ---------------------------------------------------------------------------
# ITEM 1: DR vs the best of four rules, per arm, per regime the shift can reach.
# ITEM 3: the price. DR minus joint-v1, paired per instance, as a percent of
#         joint-v1. Positive = DR costs more.
# ITEM 4: feasibility, summed over the DR seeds, from the validator's tables.
# ---------------------------------------------------------------------------
dr_vs_rule = {}      # (arm, r) -> (mean, lo, hi)
dr_price = {}        # (arm, r) -> (mean, lo, hi)
dr_infeas = {}       # (arm, r) -> int
for arm, res, sfx, regimes in ARMS:
    print(f'--- {arm}: DR vs best-of-four, and DR minus joint-v1 (paired, %) ---')
    for r in regimes:
        dr = pooled(res, DR, DR_SEEDS, r, 'DR')
        b = best_rule(res, r)
        m, lo, hi = ci(100.0 * (dr - b) / b)          # DR vs rules
        dr_vs_rule[(arm, r)] = (m, lo, hi)

        jv = pooled(res, JV1, JV1_PAIR_SEEDS, r, 'joint-v1 (paired)')
        pm, plo, phi = ci(100.0 * (dr - jv) / jv)     # DR minus joint-v1
        dr_price[(arm, r)] = (pm, plo, phi)

        dr_infeas[(arm, r)] = sum(
            infeasible(res, f'{DR}{s}', r, sfx) for s in DR_SEEDS)
        print(f'  {r:4s}  vs-rule {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}] {verdict(lo, hi):12s}'
              f'   price {pm:+5.2f} [{plo:+5.2f},{phi:+5.2f}]'
              f'   infeas={dr_infeas[(arm, r)]}')
    print()

n_infeas = sum(dr_infeas.values())
n_ahead = sum(1 for m, lo, hi in dr_vs_rule.values() if hi < 0)
n_priced = sum(1 for m, lo, hi in dr_price.values() if lo > 0)   # DR strictly costs more
print(f'DR is statistically ahead of the best rule on {n_ahead}/{len(dr_vs_rule)} '
      f'(arm, regime) cells the shift reaches')
print(f'DR pays a statistically nonzero in-distribution/shift price on '
      f'{n_priced}/{len(dr_price)} cells')
print(f'infeasible schedules across every DR arm, regime and seed: {n_infeas}\n')

# ---------------------------------------------------------------------------
# EMIT. Prefix \Dr..., no collision with any existing macros_*.tex (checked). The
# joint-v1 anchor cells are NOT emitted: they already live in macros_ood.tex, and
# re-emitting them would trip p9's "defined twice" gate.
# ---------------------------------------------------------------------------
lines = [
    f'\\newcommand{{\\DrInfeas}}{{{n_infeas}}}'
    '  % infeasible DR schedules over every arm, regime and seed',
    f'\\newcommand{{\\DrAhead}}{{{n_ahead}}}'
    f'\\newcommand{{\\DrCells}}{{{len(dr_vs_rule)}}}'
    '  % (arm,regime) cells where DR is statistically ahead of the best rule / total',
    f'\\newcommand{{\\DrPriced}}{{{n_priced}}}'
    '  % cells where DR pays a statistically nonzero cost vs joint-v1',
]
for arm, _, _, regimes in ARMS:
    w = WORD[arm]
    for r in regimes:
        m, lo, hi = dr_vs_rule[(arm, r)]
        lines.append(
            f'\\newcommand{{\\Dr{w}{r}}}{{{m:+.2f}}}'
            f'\\newcommand{{\\Dr{w}{r}Lo}}{{{lo:+.2f}}}'
            f'\\newcommand{{\\Dr{w}{r}Hi}}{{{hi:+.2f}}}'
            f'  % {arm} {r}: DR policy vs best rule (%), CI')
        pm, plo, phi = dr_price[(arm, r)]
        lines.append(
            f'\\newcommand{{\\DrPrice{w}{r}}}{{{pm:+.2f}}}'
            f'\\newcommand{{\\DrPrice{w}{r}Lo}}{{{plo:+.2f}}}'
            f'\\newcommand{{\\DrPrice{w}{r}Hi}}{{{phi:+.2f}}}'
            f'  % {arm} {r}: DR minus joint-v1 (%), CI; + = DR costs more')

header = ('auto-generated by scripts/p43_dr_arm.py -- do not hand-edit\n'
          '% Domain-randomized policy on the three OOD campaigns. DR vs the\n'
          '% per-instance best of FIFO/MOR/SPT/MWKR, and DR minus joint-v1 paired.')

if PILOT:
    out = '/tmp/macros_dr_PILOT.tex'
    lines = [f'% *** PILOT: seed 301 only, NOT FOR PUBLICATION -- do not \\input ***'] \
        + lines
    emit(out, lines, header='PILOT seed-301 ONLY -- NOT FOR PUBLICATION. ' + header)
    print(f'\nPILOT: wrote {out} (scratch only; paper/ untouched)')
else:
    emit('paper/macros_dr.tex', lines, header=header)
    print('\nwrote paper/macros_dr.tex')
