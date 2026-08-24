"""P33: what happens when the constraint parameters move outside training range.

Where does the policy stop working? E4 answers one version of that: leave the FAMILY
(blocking, no-wait) and the reward stops being admissible. This is the other version,
and the harder one to dismiss: STAY inside the family, keep the descriptor honest, and
move only the numbers.

  10x25_oodsetup   setup magnitudes drawn from a larger range, and asymmetric,
                   so S[a,b] != S[b,a]. Training saw symmetric, smaller setups.
  10x25_oodwindow  outage calendars drawn denser than any the policy trained on.

THE BASELINE HAS TO BE THE SAME ON BOTH SIDES. In distribution the paper compares
the policy against the per-instance BEST of four dispatching rules. The first OOD
sweep ran SPT alone. Measuring the policy against one rule out of distribution and
against the best of four in distribution would not be the same measurement, and
the quantity of interest is precisely how the policy's lead CHANGES when the
parameters move. So FIFO, MOR and MWKR were run on the OOD arms too, and the
baseline here is best-of-four everywhere.

WHAT WE EXPECT TO FIND, AND WHAT WE MUST REPORT EVEN IF IT IS UNFLATTERING. The
admissibility argument is about the FORM of the constraint, not the size of its
numbers, so feasibility should survive a parameter shift. Quality is a different
claim and has no such protection. If the lead over the rules shrinks, or reverses,
that is the boundary this study exists to locate, and it is reported as found.

Emits paper/macros_ood.tex.

Usage: python scripts/p33_ood_table.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402

rng = np.random.default_rng(7)          # the paper's bootstrap seed, unchanged
MARGIN = 1.0                            # the paper's +-1% equivalence margin
B = 20000
SEEDS = ['', '-s302', '-s303']
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']  # the same four the fair warm start uses
JOINT = 'greedy+10x25+family+joint-v1'
BLIND = 'greedy+10x25+family+jointblind-v1'

# The in-distribution arm is read from 10x25_cpuref, NOT from the published
# 10x25 results, and the reason is the whole methodology of this study.
#
# The three arms share the same 100 base instances. On a regime where the shifted
# constraint is inactive, their descriptors are byte-identical field for field, so
# the dispatching rules return bit-identical makespans, as they must. The published
# policy results do NOT: evaluated two days earlier on a different device and thread
# count, they differ from the shifted arms on 32 of 100 instances of regime N, by up
# to 8 hours, on inputs that are identical. Arg-max ties in the policy break the
# other way and the greedy rollout goes somewhere else.
#
# Comparing that column against the shifted arms would fold our own evaluation
# nondeterminism into the measured effect of the shift, and the "noise floor" it
# produced (0.02 points) was measuring the pipeline, not the benchmark. So the
# in-distribution arm is re-evaluated by the SAME command as the shifted arms
# (ind_cpuref.sh), and the placebo cells then come out bit-identical: their drift is
# exactly zero, by construction, and is asserted to be.
# (name, results dir, data dir, summary-file suffix). eval_family.py appends --tag to
# the SUMMARY name but not to the .npy name, so the in-distribution rerun's validator
# tables are summary_<model>_cpuref.md while its arrays are Result_<model>_<regime>.npy.
ARMS = [
    ('ind',    'test_results/FAMILY/10x25_cpuref',    'data/FAMILY/10x25',           '_cpuref'),
    ('setup',  'test_results/FAMILY/10x25_oodsetup',  'data/FAMILY/10x25_oodsetup',  ''),
    ('window', 'test_results/FAMILY/10x25_oodwindow', 'data/FAMILY/10x25_oodwindow', ''),
]
WORD = {'ind': 'Ind', 'setup': 'Setup', 'window': 'Window'}


def arr(res, tag, r):
    p = f'{res}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def pooled(res, base, r):
    """Per-instance mean over the three seed replicates. A missing seed is an error,
    not a thing to skip: pooling one side over two seeds and the other over three is
    the asymmetry this project has already had to retract once."""
    xs = [arr(res, f'{base}{s}', r) for s in SEEDS]
    if any(x is None for x in xs):
        missing = [s or '301' for s, x in zip(SEEDS, xs) if x is None]
        raise SystemExit(f'REFUSING TO REPORT: {res}/{base} on {r} is missing seed(s) '
                         f'{missing}. Every cell in this table is pooled over the '
                         f'same three seeds on both sides.')
    return np.mean(xs, axis=0), len(xs)


def infeasible(res, tag, r, sfx=''):
    """The count of schedules the INDEPENDENT VALIDATOR rejected.

    Not from the .npy. The .npy holds the makespan in column 0 and the per-instance
    inference time in column 1; feasibility is nowhere in it, so a check of the form
    `sum(~isfinite(makespan))` is structurally zero for every input that has ever
    existed. It is a gate that cannot fail, and a gate that cannot fail is worse than
    no gate, because it prints a zero that looks like evidence.

    eval_family.py runs constraint_family.reference_sim.validate_schedule on every
    realized schedule and records the survivors in the summary table. That table is
    the only place the validator's verdict is written down, so it is where the number
    has to come from."""
    p = f'{res}/summary_{tag.split("+", 1)[1]}{sfx}.md'
    if not os.path.exists(p):
        raise SystemExit(f'REFUSING TO REPORT feasibility: {p} does not exist, and '
                         f'the validator\'s verdict lives nowhere else. The .npy '
                         f'carries makespan and inference time, not feasibility.')
    for line in open(p):
        cells = [c.strip() for c in line.split('|')]
        if len(cells) > 5 and cells[1] == r and cells[2] == tag:
            ok, total = cells[5].split('/')          # the "feasible" column, "100/100"
            return int(total) - int(ok)
    raise SystemExit(f'REFUSING: no row for regime {r}, method {tag} in {p}')


def best_rule(res, r):
    """Per-instance best of the four dispatching rules. Missing any one of them
    is a hard error: the baseline the paper names is the best of four, and a baseline
    built from fewer rules is a weaker opponent than the one it claims to be."""
    xs = []
    for rule in RULES:
        a = arr(res, rule, r)
        if a is None:
            raise SystemExit(
                f'REFUSING TO REPORT: {res} has no {rule} on regime {r}. The '
                f'in-distribution baseline is the best of {RULES}; a baseline '
                f'built from fewer rules is a weaker opponent and would overstate '
                f'the policy. Run the missing rules first.')
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
        return 'policy AHEAD'
    if lo > 0:
        return 'policy BEHIND'
    return 'tied'


def shift_magnitudes():
    """How far out of distribution did we actually go? Read it from the
    descriptors, not from the arguments we think we passed."""
    out = {}
    ref, oods, oodw = ('data/FAMILY/10x25', 'data/FAMILY/10x25_oodsetup',
                       'data/FAMILY/10x25_oodwindow')

    def setups(d):
        v = []
        for p in sorted(glob.glob(f'{d}/desc/S/*.desc.json'))[:100]:
            v += [x for row in json.load(open(p))['setup'] for x in row]
        return np.mean(v)

    def density(d):
        v = []
        for p in sorted(glob.glob(f'{d}/desc/W/*.desc.json'))[:100]:
            j = json.load(open(p))
            v.append(sum(e - s for m in j['windows'] for s, e in m)
                     / (25 * j['horizon']))
        return np.mean(v)

    out['setup_x'] = setups(oods) / setups(ref)
    out['window_x'] = density(oodw) / density(ref)
    A = np.array(json.load(open(f'{oods}/desc/S/instance_000.desc.json'))['setup'])[:-1]
    out['asym'] = not np.allclose(A, A.T)
    return out


sh = shift_magnitudes()
print(f'SHIFT: setups {sh["setup_x"]:.2f}x and '
      f'{"ASYMMETRIC" if sh["asym"] else "symmetric"}; '
      f'outage density {sh["window_x"]:.2f}x\n')

lead, infeas = {}, {}
for name, res, _, sfx in ARMS:
    print(f'--- {name}: joint policy vs the per-instance best of {"/".join(RULES)} '
          f'(%, negative = policy ahead) ---')
    for r in REGIMES:
        p, ns = pooled(res, JOINT, r)
        b = best_rule(res, r)
        g = 100.0 * (p - b) / b                     # paired, per instance
        m, lo, hi = ci(g)
        lead[(name, r)] = (m, lo, hi)
        # From the validator, per seed, summed. Not from the makespan column.
        infeas[(name, r)] = sum(
            infeasible(res, f'greedy+{JOINT.split("+", 1)[1]}{s}', r, sfx)
            for s in SEEDS)
        print(f'  {r:4s}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]  {verdict(lo, hi):13s}'
              f'  seeds={ns}  infeas={infeas[(name, r)]}')
    print()

# ---------------------------------------------------------------------------
# THE CONTROL, ASSERTED RATHER THAN HOPED FOR.
#
# On a regime where the shifted constraint is inactive, the three arms hold the same
# instances and byte-identical descriptors. The policy is deterministic given its
# input and the evaluation command, and all three arms were evaluated by the same
# command. The makespans must therefore be BIT-IDENTICAL, and the placebo drift must
# be exactly zero.
#
# If it is not, the arms were not evaluated the same way, the "effect" of the shift
# then the difference between the arms is not the shift alone, and no number below
# means what it says. So this is an assertion, not a diagnostic print: the table is
# not written unless the control holds bit for bit.
# ---------------------------------------------------------------------------
ACTIVE = {'setup': 'S', 'window': 'W'}     # the constraint bit each arm shifts

for name, res, _, sfx in ARMS[1:]:
    for r in REGIMES:
        if ACTIVE[name] in r:
            continue                        # treated, not placebo
        a, _ = pooled(ARMS[0][1], JOINT, r)
        b, _ = pooled(res, JOINT, r)
        if not np.array_equal(a, b):
            n = int((a != b).sum())
            raise SystemExit(
                f'CONTROL FAILED: on the {name} arm, regime {r} has no {ACTIVE[name]} '
                f'active, so the shift cannot reach it and the policy must return the '
                f'same schedule it returns in distribution. It does not: {n} of '
                f'{len(a)} instances differ, by up to {np.abs(a - b).max():.1f} h.\n'
                f'The two arms were not produced by the same evaluation. Any "effect" '
                f'this script reports would be that difference, not the shift. '
                f'Re-evaluate the in-distribution arm with ind_cpuref.sh and rerun.')
print(f'CONTROL PASSED: on every regime the shift cannot reach, the policy returns '
      f'bit-identical schedules across arms.\n')

# ---------------------------------------------------------------------------
# PLACEBO AND TREATED CELLS.
#
# The three arms hold the SAME instances. A regime in which the shifted constraint is
# inactive cannot be reached by the shift at all: in the setup arm the setup bit is
# off in N, L, W and LW, and in the window arm the window bit is off in N, L, S and
# LS. Those are the placebo cells, their drift is zero by construction, and the
# assertion above has already proved it bit for bit.
#
# The floor against which a treated cell is read is therefore not an estimate of
# noise. It is zero. Every movement below is the shift and nothing else.
# ---------------------------------------------------------------------------
placebo, treated = [], []
for name in ('setup', 'window'):
    for r in REGIMES:
        drift = lead[(name, r)][0] - lead[('ind', r)][0]   # + = lead shrank
        bucket = treated if ACTIVE[name] in r else placebo
        bucket.append((name, r, lead[('ind', r)][0], lead[(name, r)][0], drift))

floor = max(abs(d) for *_, d in placebo) if placebo else 0.0
print(f'PLACEBO CELLS (the shift cannot reach them; drift must be zero)')
for name, r, a, b_, d in placebo:
    print(f'  {name:7s} {r:4s}  {a:+.2f} -> {b_:+.2f}   drift {d:+.2f}')
print(f'  largest placebo drift: {floor:.2f} points')
if floor > 0.005:
    raise SystemExit(
        f'CONTROL FAILED: a placebo cell moved by {floor:.2f} points. The schedules '
        f'were asserted bit-identical, so a nonzero drift here means the RULE '
        f'baseline differs between arms on a regime the shift cannot reach. Stop.')

print(f'\nTREATED CELLS (the shifted constraint is ACTIVE)')
hurt, helped = [], []
for name, r, a, b_, d in treated:
    tag = '  LEAD SHRANK' if d > 0 else ('  LEAD GREW' if d < 0 else '')
    (hurt if d > 0 else helped).append((name, r, a, b_, d))
    print(f'  {name:7s} {r:4s}  {a:+.2f} -> {b_:+.2f}   drift {d:+.2f}{tag}')

# A cell counts as "the policy stopped being ahead" only if it WAS ahead in
# distribution and is not ahead now. Testing only the shifted interval would also
# catch a cell that was never ahead to begin with (S is one), and would report a
# loss the shift did not cause.
reversed_ = [t for t in treated if t[2] < 0 <= t[3]]
tied = [t for t in treated
        if lead[('ind', t[1])][2] < 0 <= lead[(t[0], t[1])][2]]
print(f'\nof the {len(treated)} treated cells, the lead shrank on {len(hurt)} and grew '
      f'on {len(helped)}')
if tied:
    print('the policy is no longer statistically ahead of the dispatching rules on:')
    for name, r, a, b_, _ in tied:
        m, lo, hi = lead[(name, r)]
        print(f'  {name:7s} {r:4s}  {m:+.2f}%  CI [{lo:+.2f},{hi:+.2f}] '
              f'(was {a:+.2f}% in distribution)')
if reversed_:
    print('THE RULES OVERTOOK THE POLICY on:')
    for name, r, a, b_, _ in reversed_:
        print(f'  {name:7s} {r:4s}  {a:+.2f}% -> {b_:+.2f}%')
else:
    print('the mean lead never reverses sign: the policy is not overtaken anywhere')

n_infeas = sum(infeas.values())
print(f'\ninfeasible schedules across every arm and regime: {n_infeas}')

# ---------------------------------------------------------------------------
# One explanation for the descriptor's null result (E2, E3) is that this benchmark
# makes the active regime easy to read off ordinary features, so the description tells
# the network only what it can already see. That explanation is falsifiable: if it is
# right, a shift which makes the regime harder to read should hurt the TOKEN-BLIND
# policy more than the conditioned one. We have both policies, so we test it rather
# than argue about it.
#
# The quantity is the conditioned policy's lead over the token-blind one,
# 100*(blind - joint)/joint, per instance. Negative means the BLIND policy is
# ahead, which is what E2 found in distribution. If the sign moves toward the
# conditioned policy as the parameters shift, that explanation holds.
# ---------------------------------------------------------------------------
print('\n--- conditioned vs token-blind, 100*(blind - joint)/joint (%) ---')
print('    negative = the token-blind policy is ahead (the E2 finding)')
tok = {}
have_blind = True
for name, res, _, sfx in ARMS:
    row = []
    for r in REGIMES:
        j, _ = pooled(res, JOINT, r)
        b, _ = pooled(res, BLIND, r)
        if j is None or b is None:
            have_blind = False
            continue
        g = 100.0 * (b - j) / j
        m, lo, hi = ci(g)
        tok[(name, r)] = (m, lo, hi)
        row.append(m)
    if row:
        print(f'  {name:7s} ' + ' '.join(f'{v:+6.2f}' for v in row)
              + f'   mean {np.mean(row):+.2f}')
if not have_blind:
    print('  token-blind results are not on disk for every arm yet; rerun after')
    print('  ood_blind.sh finishes. The macros below omit what is missing.')

lines = [
    f'\\newcommand{{\\OodSetupX}}{{{sh["setup_x"]:.2f}}}'
    '  % mean setup magnitude, shifted arm / published arm',
    f'\\newcommand{{\\OodWindowX}}{{{sh["window_x"]:.2f}}}'
    '  % mean outage density, shifted arm / published arm',
    f'\\newcommand{{\\OodInfeas}}{{{n_infeas}}}'
    '  % infeasible schedules over both shifted arms and all 8 regimes',
    f'\\newcommand{{\\OodFloor}}{{{floor:.2f}}}'
    '  % largest drift on a PLACEBO cell: the resampling noise floor (pp)',
    f'\\newcommand{{\\OodPlacebo}}{{{len(placebo)}}}'
    f'\\newcommand{{\\OodTreated}}{{{len(treated)}}}'
    '  % cells where the shifted constraint is inactive / active',
    f'\\newcommand{{\\OodHurt}}{{{len(hurt)}}}'
    f'\\newcommand{{\\OodHelped}}{{{len(helped)}}}'
    '  % treated cells whose lead shrank / grew by more than the noise floor',
    f'\\newcommand{{\\OodTied}}{{{len(tied)}}}'
    '  % treated cells where the policy is no longer statistically ahead',
    f'\\newcommand{{\\OodReversed}}{{{len(reversed_)}}}'
    '  % treated cells where the rules actually overtook the policy',
]
for name, _, _, _ in ARMS:
    for r in REGIMES:
        if (name, r) not in lead:
            continue
        m, lo, hi = lead[(name, r)]
        w = WORD[name]
        lines.append(
            f'\\newcommand{{\\Ood{w}{r}}}{{{m:+.2f}}}'
            f'\\newcommand{{\\Ood{w}{r}Lo}}{{{lo:+.2f}}}'
            f'\\newcommand{{\\Ood{w}{r}Hi}}{{{hi:+.2f}}}'
            f'  % {name} {r}: policy vs best rule (%), CI')

if tok:
    for name, _, _, _ in ARMS:
        vals = [tok[(name, r)][0] for r in REGIMES if (name, r) in tok]
        if not vals:
            continue
        w = WORD[name]
        lines.append(
            f'\\newcommand{{\\Tok{w}Mean}}{{{np.mean(vals):+.2f}}}'
            f'  % {name}: mean over regimes of 100*(blind-joint)/joint; '
            f'negative = the token-blind policy is ahead')
        for r in REGIMES:
            if (name, r) not in tok:
                continue
            m, lo, hi = tok[(name, r)]
            lines.append(
                f'\\newcommand{{\\Tok{w}{r}}}{{{m:+.2f}}}'
                f'\\newcommand{{\\Tok{w}{r}Lo}}{{{lo:+.2f}}}'
                f'\\newcommand{{\\Tok{w}{r}Hi}}{{{hi:+.2f}}}'
                f'  % {name} {r}: token-blind minus conditioned (%), CI')

emit('paper/macros_ood.tex', lines,
     header=('% auto-generated by scripts/p33_ood_table.py -- do not hand-edit\n'
             '% Policy vs the per-instance best of FIFO/MOR/SPT/MWKR, the same\n'
             '% baseline in distribution and out of it.'))
print('\nwrote paper/macros_ood.tex')
