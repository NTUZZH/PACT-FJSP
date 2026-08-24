"""Rebuild the E5 table (Table III) and every E5 macro from one source.

Two conventions make the exact-solver comparison fair, and both move numbers
AGAINST the learned policy:

  * CP-SAT is warm-started from the BEST of the four dispatching rules on each
    instance, not from a fixed rule. Seeding it from MWKR alone would understate
    it badly, because MWKR is weak precisely on the setup-bearing regimes. The
    four rules are the mask-respecting (repaired) variants: each rule picks only
    among jobs that hold a legal machine pair, and the stored hint is floored at
    the original best-of-four so it can only be stronger (p41_repaired_warm_cache).
  * CP-SAT is CREDITED with that warm start. AddHint is a search suggestion, not
    a solution: at a short budget the solver can report UNKNOWN, or an incumbent
    worse than the complete feasible schedule it was handed, while that schedule
    is still in hand. Each cell is therefore min(hint, search), which is what a
    practitioner would hold when the budget expires. Cells where the search
    improved on the warm start on NO instance carry a dagger: there the solver's
    reported value IS the repaired best-of-four portfolio.

The policy's own latency comes from results/policy_latency_cpu.json, written by
p16_policy_latency.py on an unloaded machine. It is deliberately NOT taken from
column 1 of the evaluation result files: those timings were recorded during bulk
evaluation, and on a workload where every instance has the same operation count a
wide spread in them measures contention, not computation.

Writes results/e5_table_fair.md, paper/macros_e5.tex, paper/tables/e5_table.tex.
Usage: python scripts/p13_e5_table_fair.py
"""
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

RES = 'test_results/FAMILY/10x25'
# The repaired best-of-four warm-start cache (scripts/p41_repaired_warm_cache.py):
# per instance, the best of the four mask-respecting dispatching rules, floored at
# the original best-of-four so the hint can only be stronger. The 5s and 30s tiers
# were re-solved from this hint (the 10x25_fairwarmR{5,30}s directories, whose
# warm_makespan field IS the repaired hint). The 1s/2s/10s/300s tiers were not
# re-solved, so their credited value is recomputed as min(repaired hint, the tier's
# original credited value): a stronger hint can only lower the number CP-SAT is
# credited with, which is the conservative direction for the policy comparison.
REPAIRED_CACHE = 'or_solution/FAMILY/warmstart_cache_repaired/makespans.json'
_REPAIRED = json.load(open(REPAIRED_CACHE))


def _ga_budget():
    """The GA's per-instance wall-clock budget, read from the GA itself.

    Typed into the paper as a literal, this would be a number that quietly becomes
    false the first time somebody reruns the GA with a different --budget. It is one
    line to parse it out of the default, so it is parsed.
    """
    import re as _re
    src = open('eval_ga_family.py').read()
    m = _re.search(r'budget,\s*pop,\s*limit,\s*jobs\s*=\s*(\d+)', src)
    if not m:
        raise SystemExit('cannot find the GA budget default in eval_ga_family.py; '
                         'refusing to guess it')
    return int(m.group(1))


GA_BUDGET_S = _ga_budget()
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
TRAIN = ['N', 'L', 'S', 'W', 'LS', 'LW']
BUDGETS = [5, 30, 300]
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')


def npy(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def policy(r):
    """The measured policy, per-instance mean over whatever seeds exist.

    PROTAGONIST SWAP (2026-08-11, author-approved): the table shows the
    RELEASED token-blind joint, the artifact a plant would deploy; the
    conditioned joint remains the evaluated pathway of E2--E3."""
    base = 'greedy+10x25+family+jointblind-v1'
    arrs = [npy(f'{base}{s}', r) for s in ('', '-s302', '-s303')]
    arrs = [a for a in arrs if a is not None]
    return np.mean(arrs, axis=0) if arrs else None


def cp(r, b):
    """Fair-warm CP-SAT at budget b, credited against the REPAIRED warm start.

    Returns (mean, n_opt, n, n_search_improved). The last is the count of
    instances on which the SEARCH beat the warm start it was handed; when it is
    zero the cell is exactly the repaired best-of-four dispatching portfolio and
    gets a dagger.

    The 5s and 30s tiers were re-solved from the repaired hint, so they are read
    from the 10x25_fairwarmR{b}s directories, where warm_makespan is that hint and
    the credited value is min(warm start, search) directly. The 300s tier was not
    re-solved: its credited value is recomputed as min(repaired hint, the tier's
    original min(warm start, search)), and 'improved' means the 300s search beat
    the repaired hint.
    """
    objs, nopt, n, improved = [], 0, 0, 0
    rerun = b in (5, 30)
    src = (f'or_solution/FAMILY/10x25_fairwarmR{b}s/{r}/*.json' if rerun
           else f'or_solution/FAMILY/10x25_fairwarm{b}s/{r}/*.json')
    for f in glob.glob(src):
        d = json.load(open(f))
        n += 1
        obj = d.get('objective')
        if obj is None:
            continue
        if rerun:
            # warm_makespan IS the repaired hint; the solve retained its incumbent,
            # so min(warm start, search) is the credited value the Protocol asks for.
            hint = d['warm_makespan']
            objs.append(min(obj, hint))
            if obj < hint - 1e-9:
                improved += 1
        else:
            # Not re-solved: floor the tier's original credited value at the
            # repaired hint (which is <= the original hint by construction).
            hint = _REPAIRED[r][d['instance']]['repaired_ms']
            objs.append(min(hint, min(obj, d['warm_makespan'])))
            if obj < hint - 1e-9:
                improved += 1
        if d.get('status') == 'OPTIMAL':
            nopt += 1
    if n == 0:
        return None, 0, 0, 0
    return (np.mean(objs) if objs else None), nopt, n, improved


# TWO RULE BASELINES, BOTH REPORTED, because the table needs both:
#
#   best single rule : the rule with the lowest per-regime MEAN. The strongest
#                      of the four as a STANDALONE dispatching policy.
#   portfolio        : the per-instance MINIMUM over the four rules. A deployable
#                      heuristic in its own right (run all four, keep the best),
#                      strictly stronger than any single rule, and the estimator
#                      CP-SAT is warm-started from -- the same min-over-four that
#                      builds the hint.
#
# Both columns must appear because the daggered CP cells ARE the portfolio: a
# dagger means the search improved on the warm start on no instance, so the cell
# equals the portfolio and NOT the best single rule. Reporting only one rule
# column would make a daggered cell look equal to a column it is not.
rows, missing = [], []
for r in REGIMES:
    pol = policy(r)                                   # [n] per-instance
    per_rule = {m: npy(m, r) for m in RULES}
    per_rule = {m: a for m, a in per_rule.items() if a is not None}
    best_single = (min(per_rule, key=lambda m: per_rule[m].mean())
                   if per_rule else None)
    pdr_vec = per_rule[best_single] if best_single else None
    port_vec = (np.min(np.stack(list(per_rule.values())), axis=0)
                if per_rule else None)               # per-instance best-of-four
    ga = npy('GA', r)
    cells = {b: cp(r, b) for b in BUDGETS}
    for b in BUDGETS:
        if cells[b][2] == 0:
            missing.append(f'{r}@{b}s')
    rows.append((r, pol, pdr_vec, port_vec, ga, cells, best_single))

if missing:
    print(f'INCOMPLETE (campaign still running): {", ".join(missing)}\n')


def num(x):
    if x is None:
        return '--'
    v = x.mean() if hasattr(x, 'mean') else x
    return f'{v:.1f}'


# ---------------------------------------------------------------- markdown
lines = [('| regime | greedy | best single PDR | PDR portfolio | GA-30s '
          '| CP 5s | CP 30s | CP 300s | opt | n(300s) |'), '|' + '---|' * 10]
for r, p, pdr, port, ga, cells, rule in rows:
    def fmt(b):
        m, _, n, imp = cells[b]
        if n == 0:
            return 'pending'
        if m is None:
            return f'none (0/{n})'
        return f'{m:.1f}' + ('+' if imp == 0 else '')
    opt = cells[300][1] if cells[300][2] else '--'
    n300 = cells[300][2] or '--'
    lines.append(f'| {r} | {num(p)} | {num(pdr)} ({rule}) | {num(port)} | '
                 f'{num(ga)} | {fmt(5)} | {fmt(30)} | {fmt(300)} | {opt} | {n300} |')

# ---------------------------------------------------------------- latex body
# BOLD RULE (stated in the table notes, applied mechanically to every row).
# The columns do not share a decision budget, so a single row minimum would
# compare a policy that answers in under a second against a solver given 30,
# and the bold would mean "was given the most time" rather than "is the better
# option". The comparison is therefore made inside each budget class:
#   fast  -- greedy, rule, pool: every one of them returns within a second
#   slow  -- GA, CP 5s, CP 30s: every one of them is given 5s or more
# and the shortest makespan in each class is bolded. The 300s column is excluded
# because it covers a 30-instance prefix, not the full set (footnote c), so it
# is not comparable with anything in its own row.
#
# The rule is applied whatever it bolds. It happens to bold the policy in the
# fast class of every row, which is the paper's claim; in the slow class it
# bolds a rival in most rows, and that is left standing: the rule is uniform
# across rows and classes.
FAST, SLOW = 'fast', 'slow'


def bold_min(values):
    """Indices of the shortest value, ignoring cells with no number."""
    live = [(i, v) for i, v in enumerate(values) if v is not None]
    if not live:
        return set()
    lo = min(v for _, v in live)
    return {i for i, v in live if v == lo}


tex = []
for r, p, pdr, port, ga, cells, _ in rows:
    def cell(b):
        m, _, n, imp = cells[b]
        if n == 0 or m is None:
            return '--'
        return f'{m:.1f}' + (r'$^\dagger$' if imp == 0 else '')
    opt = cells[300][1] if cells[300][2] else '--'
    if r == 'SW':
        tex.append(r'\midrule')
    # The 300s tier runs a 30-instance prefix while every other column covers all
    # 100, so its cells (and the opt count read from them) are set in italics to
    # keep a row from being scanned as like-for-like (footnote c states why).
    c300 = cell(300)
    c300 = c300 if c300 == '--' else rf'\textit{{{c300}}}'
    opt = opt if opt == '--' else rf'\textit{{{opt}}}'

    def val(x):
        """The number as the READER sees it: rounded to the printed precision,
        so two cells that print the same value are either both bolded or
        neither. Comparing the full-precision means would bold one of two
        columns showing 195.6 and leave the reader looking for the difference."""
        if x is None:
            return None
        v = float(x.mean()) if hasattr(x, 'mean') else float(x)
        return round(v, 1)

    def cval(b):
        m, _, n, _ = cells[b]
        return None if (n == 0 or m is None) else round(float(m), 1)

    def mark(text, on):
        """Bold the number, never a dagger or an em-rule riding on it."""
        if not on or text == '--':
            return text
        head, sep, tail = text.partition('$')
        return rf'\textbf{{{head}}}' + sep + tail

    fast = bold_min([val(p), val(pdr), val(port)])
    slow = bold_min([val(ga), cval(5), cval(30)])
    cs = [mark(num(p), 0 in fast), mark(num(pdr), 1 in fast),
          mark(num(port), 2 in fast), mark(num(ga), 0 in slow),
          mark(cell(5), 1 in slow), mark(cell(30), 2 in slow)]
    tex.append(f'{r} & ' + ' & '.join(cs) + f' & {c300} & {opt} \\\\')

# ---------------------------------------------------------------- macros
# THE ESTIMATOR. The Protocol's definition, which every percentage in the paper
# is held to: the MEAN OF THE PER-INSTANCE RELATIVE GAPS, paired instance by
# instance. Not a ratio of means, and not a mean of per-regime ratios -- those
# are different quantities that print in the same units. Each gap carries a
# paired-bootstrap CI, because a point estimate with no interval is not a
# comparison.
rng = np.random.default_rng(7)


def paired_gap(pol, other):
    """Mean per-instance relative gap, + a paired-bootstrap 95% CI.

    Positive = `other` is SLOWER than the policy, i.e. the policy wins.
    """
    d = 100.0 * (other - pol) / pol                  # per instance
    idx = rng.integers(0, len(d), (20000, len(d)))
    bs = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def pooled_gap(get):
    """Per-instance gaps pooled across the training regimes, then bootstrapped."""
    ds = []
    for r, p, pdr, port, ga, _, _ in rows:
        if r not in TRAIN:
            continue
        o = get(pdr, port, ga)
        if o is None or p is None:
            continue
        ds.append(100.0 * (o - p) / p)
    if not ds:
        return None
    d = np.concatenate(ds)
    idx = rng.integers(0, len(d), (20000, len(d)))
    bs = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


g_pdr = pooled_gap(lambda pdr, port, ga: pdr)
g_port = pooled_gap(lambda pdr, port, ga: port)
g_ga = pooled_gap(lambda pdr, port, ga: ga)

# COUNTS. A quantifier ("beats the best rule on all six") is a claim about the
# data that no rerun can falsify unless the count is printed. Emit them.
def wins(get):
    n = 0
    for r, p, pdr, port, ga, _, _ in rows:
        if r not in TRAIN:
            continue
        o = get(pdr, port, ga)
        if o is not None and p is not None and p.mean() < o.mean():
            n += 1
    return n


n_pdr = wins(lambda pdr, port, ga: pdr)
n_port = wins(lambda pdr, port, ga: port)
n_ga = wins(lambda pdr, port, ga: ga)

# PER-REGIME significance of the policy's lead over the best single rule.
#
# \EfiveBeatsPdr counts the regimes where the policy wins IN THE MEAN. That is a
# point estimate with no interval, and the supplement's own standard ("a lead
# whose interval covers zero is not a lead") holds it to more: the count that can
# be stated as significant is the count of regimes whose paired-bootstrap CI
# excludes zero. Emit one interval per regime so the reader can see which lead is
# certified and which is not.
#
# SAME estimator as \EfivePdrDelta above: the mean of the per-instance relative
# gaps 100*(rule - policy)/policy, positive = the rule is SLOWER, i.e. the policy
# wins; paired bootstrap, B=20000. A dedicated rng seeded exactly as the spec asks
# (np.random.default_rng(7)), independent of the pooled draws above so those
# macros are byte-for-byte unchanged.
_reg_rng = np.random.default_rng(7)


def per_regime_gap(pol, pdr):
    """Mean per-instance relative gap of the best single rule over the policy,
    with a 95% paired-bootstrap CI. Positive = the policy is ahead."""
    d = 100.0 * (pdr - pol) / pol                     # per instance
    idx = _reg_rng.integers(0, len(d), (20000, len(d)))
    bs = d[idx].mean(axis=1)
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


reg_gap = {}
for r, p, pdr, port, ga, _, _ in rows:
    if p is None or pdr is None:
        continue
    reg_gap[r] = per_regime_gap(p, pdr)
# A CI excludes zero when it lies entirely on one side of it. Counted over the
# TRAINING regimes only, to sit next to \EfiveBeatsPdr (also a count of six).
n_pdr_sig = sum(1 for r in TRAIN
                if r in reg_gap and (reg_gap[r][1] > 0 or reg_gap[r][2] < 0))

# THE GA IS THE BASELINE THE DISCUSSION HAS TO ANSWER, so the discussion needs its
# numbers, over all eight regimes and not only the six the policy trained on. The GA
# needs no training at all -- its decoder IS the family dynamics -- so it covers the
# held-out compositions for free, and a reader who notices that will go looking for
# exactly these numbers. Emit them, including the one regime where the GA WINS.
ga_rows = [(r, 100.0 * (ga.mean() - p.mean()) / p.mean())
           for r, p, _, _, ga, _, _ in rows if p is not None and ga is not None]
ga_loss = [(r, -d) for r, d in ga_rows if d < 0]        # regimes where the GA is ahead
ga_win = [d for _, d in ga_rows if d > 0]               # ... and where the policy is
n_ga_all = len(ga_win)
# regimes where the 5s search improved on its warm start on ZERO instances
n_dagger = sum(1 for _, _, _, _, _, c, _ in rows if c[5][2] and c[5][3] == 0)

# The 300s tier must hold the SAME number of instances in every regime, or the
# column is not one column and the footnote's single n is false. Take the size
# from the agreeing set and refuse a ragged tier; a max() or any other reduction
# over regimes would let one regime print a 10-instance mean under a footnote
# claiming 30.
n300_by_regime = {r: c[300][2] for r, _, _, _, _, c, _ in rows if c[300][2]}
n300_set = set(n300_by_regime.values())
if len(n300_set) > 1:
    raise SystemExit(
        'RAGGED 300s tier: ' +
        ', '.join(f'{r}={n}' for r, n in sorted(n300_by_regime.items())) +
        '\nThe footnote claims a fixed prefix of the same size for every regime. '
        'Finish the tier, or report per-regime n. Refusing to emit a false '
        'footnote.')
n300 = n300_set.pop() if n300_set else 0

# The E5 prose makes two counting claims about the solver, and a counting claim
# is a number: emit it rather than type it. At 5 s, on the regimes whose search
# improved on the warm start on zero instances, is the policy ahead of the
# solver? At 30 s, on how many regimes has the solver passed the policy?
n_five_ahead = sum(1 for _, p, _, _, _, c, _ in rows
                   if c[5][2] and c[5][3] == 0 and p.mean() < c[5][0])
n_thirty_solver = sum(1 for _, p, _, _, _, c, _ in rows
                      if c[30][2] and c[30][0] < p.mean())
n_all = len(rows)

def gap_macros(name, g, what):
    if g is None:
        return []
    m, lo, hi = g
    return [f'\\newcommand{{\\{name}Delta}}{{{abs(m):.1f}}}'
            f'\\newcommand{{\\{name}Lo}}{{{lo:+.2f}}}'
            f'\\newcommand{{\\{name}Hi}}{{{hi:+.2f}}}'
            f'  % {what}: mean per-instance relative gap, 95% paired-bootstrap CI']


macros = ['% auto-generated by scripts/p13_e5_table_fair.py -- do not hand-edit']
macros += gap_macros('EfivePdr', g_pdr, 'policy vs BEST SINGLE rule')
macros += gap_macros('EfivePort', g_port, 'policy vs the per-instance PORTFOLIO')
macros += gap_macros('EfiveGa', g_ga, 'policy vs GA-30s')

# Per-regime policy-vs-best-single-rule leads, one signed triple per regime, so a
# claim of significance can be checked regime by regime rather than in the mean.
for r in REGIMES:
    if r not in reg_gap:
        continue
    m, lo, hi = reg_gap[r]
    _excl = 'excludes 0' if (lo > 0 or hi < 0) else 'COVERS 0'
    macros.append(
        f'\\newcommand{{\\EfiveReg{r}}}{{{m:+.2f}}}'
        f'\\newcommand{{\\EfiveReg{r}Lo}}{{{lo:+.2f}}}'
        f'\\newcommand{{\\EfiveReg{r}Hi}}{{{hi:+.2f}}}'
        f'  % {r}: policy lead over best single rule, mean per-instance relative'
        f' gap, 95% paired-bootstrap CI ({_excl})')
macros.append(
    f'\\newcommand{{\\EfiveBeatsPdrSig}}{{{n_pdr_sig}}}'
    f'  % training regimes (of {len(TRAIN)}) whose per-regime CI over the best'
    f' single rule EXCLUDES zero')
macros += [
    f'\\newcommand{{\\EfiveBeatsPdr}}{{{n_pdr}}}'
    f'  % training regimes (of {len(TRAIN)}) where the policy beats the best single rule',
    f'\\newcommand{{\\EfiveBeatsPort}}{{{n_port}}}'
    f'  % ... where it beats the per-instance best-of-four PORTFOLIO',
    f'\\newcommand{{\\EfiveBeatsGa}}{{{n_ga}}}'
    f'  % ... where it beats the GA',
    f'\\newcommand{{\\GaPolicyAhead}}{{{n_ga_all}}}'
    f'\\newcommand{{\\GaRegimes}}{{{len(ga_rows)}}}'
    f'  % regimes (of all {len(ga_rows)}, held-out included) where the policy beats the GA',
    f'\\newcommand{{\\GaAheadN}}{{{len(ga_loss)}}}'
    + (f'\\newcommand{{\\GaAheadReg}}{{{ga_loss[0][0]}}}'
       f'\\newcommand{{\\GaAheadDelta}}{{{ga_loss[0][1]:.2f}}}' if len(ga_loss) == 1 else '')
    + f'  % ... and where the GA beats the policy'
    + (f' ({ga_loss[0][0]}, by {ga_loss[0][1]:.2f}%)' if len(ga_loss) == 1
       else f' ({[r for r, _ in ga_loss]})'),
    f'\\newcommand{{\\GaPolicyLeadLo}}{{{min(ga_win):.2f}}}'
    f'\\newcommand{{\\GaPolicyLeadHi}}{{{max(ga_win):.2f}}}'
    f'  % the policy\'s lead over the GA on the regimes it leads, min and max (%)',
    f'\\newcommand{{\\EfiveTrainN}}{{{len(TRAIN)}}}',
    f'\\newcommand{{\\EfiveDaggerN}}{{{n_dagger}}}'
    '  % regimes where the 5s search improved on its warm start on ZERO instances',
    f'\\newcommand{{\\EfiveCpFiveAhead}}{{{n_five_ahead}}}'
    f'  % of the {n_dagger} daggered regimes, how many the policy leads at 5s',
    f'\\newcommand{{\\EfiveCpThirtySolver}}{{{n_thirty_solver}}}'
    f'\\newcommand{{\\EfiveAllN}}{{{n_all}}}'
    f'  % regimes (of {n_all}) where the 30s solver has passed the policy',
    f'\\newcommand{{\\CpThreeHundredN}}{{{n300}}}'
    '  % instances per regime in the 300s tier',
]

# The optimality counts come from the same JSONs as the table, so they can only
# ever describe the run the table reports.
#
# Optimality is a COUNT whose denominator is the 300s TIER (n300 instances), not
# the test set (\nTest = 100). The two denominators differ by more than 3x, so a
# count read against the wrong one is not a rounding error. Emit the count and
# its own denominator as separate macros and let the sentence quote both, so the
# denominator cannot be substituted in prose.
for r in ('L', 'LW'):
    row = next((x for x in rows if x[0] == r), None)
    if row and row[5][300][2]:
        n_opt, n = row[5][300][1], row[5][300][2]
        macros.append(
            f'\\newcommand{{\\Ref{r}Opt}}{{{n_opt}}}'
            f'  % instances proved optimal at 300s, of {n} (regime {r})')

# ---------------------------------------------------------------------------
# The MWKR counterfactual: what an MWKR-SEEDED solver reads on S at 300s.
#
# INVARIANT: this number is credited under min(hint, search), the same accounting
# every other solver number in the paper uses. A solver is never charged for a
# search result worse than the incumbent it was handed.
#
# The legacy JSONs predate the incumbent field and carry no warm_makespan, so the
# hint is reconstructed from the MWKR rule's own per-instance makespans -- which
# is exactly what that campaign handed the solver. On 4 of 100 instances the hint
# beats the search, and those 4 are credited to the hint.
_legacy = {}
for _f in glob.glob('or_solution/FAMILY/10x25/S/*.json'):
    _d = json.load(open(_f))
    if _d.get('objective') is not None:
        _legacy[_d['instance']] = float(_d['objective'])
_mwkr = npy('MWKR', 'S')
if _legacy and _mwkr is not None and len(_legacy) == len(_mwkr):
    _search = np.array([_legacy[k] for k in sorted(_legacy)], dtype=float)
    _credited = np.minimum(_search, _mwkr)
    macros.append(
        f'\\newcommand{{\\CpMwkrSthreehundred}}{{{_credited.mean():.1f}}}'
        f'  % MWKR-seeded CP-SAT @300s on S, credited min(hint, search) as'
        f' everywhere else; raw search alone reads {_search.mean():.1f}')
else:
    print('WARNING: cannot recompute \\CpMwkrSthreehundred from '
          'or_solution/FAMILY/10x25/S -- it will be undefined.')

# Only the LATENCY macros need a measurement taken on an unloaded machine. The
# table and the makespan macros come from the result files and are valid whatever
# else the machine was doing, so the refusal below is narrow: it withholds the
# three timing macros and emits everything else. The measured artifact follows
# the protagonist swap: the released blind model's own 4-core measurement, not
# the conditioned joint's (results/policy_latency_cpu.json, kept for record).
lat_path = 'results/policy_latency_cpu_4core_blind.json'
have_lat = os.path.exists(lat_path)
if have_lat:
    lat = json.load(open(lat_path))
    psec = lat['worst_regime_median_s']
    cores = int(lat['cores']) if str(lat['cores']).isdigit() else 4
    macros += [
        f'\\newcommand{{\\PolicySec}}{{{psec:.2f}}}'
        f'  % worst-regime median s/instance, {cores} CPU cores, model resident',
        f'\\newcommand{{\\PolicyCores}}{{{cores}}}',
        f'\\newcommand{{\\CpFiveVsPolicy}}{{{5.0 / psec:.0f}}}'
        '  % the 5s rung, in units of the policy runtime',
        # The GA is the baseline the discussion has to answer: it needs no training
        # at all, and at 30 s it is within about a percent of the policy. The one
        # axis on which the policy is not close is latency, so that ratio has to
        # arrive through a macro like every other number, not be typed as "roughly
        # fifty times" and left to rot when the latency is remeasured.
        f'\\newcommand{{\\GaSec}}{{{GA_BUDGET_S}}}'
        '  % the GA wall-clock budget per instance (eval_ga_family.py, --budget)',
        f'\\newcommand{{\\GaVsPolicy}}{{{GA_BUDGET_S / psec:.0f}}}'
        '  % the GA budget, in units of the policy runtime',
    ]
else:
    print('WARNING: no results/policy_latency_cpu.json, so \\PolicySec, '
          '\\PolicyCores and \\CpFiveVsPolicy are NOT emitted and the paper will '
          'fail to build on them.\nRun scripts/p16_policy_latency.py --device cpu '
          'ON A QUIET BOX.\nDo NOT fall back to column 1 of the result files: '
          'that timing was taken under contention and is ~10x too slow.')

os.makedirs('results', exist_ok=True)
with open('results/e5_table_fair.md', 'w') as f:
    f.write('# E5: mean makespan per regime (fair-warm, incumbent-retaining CP-SAT)\n\n'
            'CP columns: best-of-four-PDR warm start, 8 workers on dedicated cores,\n'
            'value = min(warm start, search). "+" = the search improved on the warm\n'
            'start on ZERO instances, so the cell is the best dispatching rule.\n\n')
    f.write('\n'.join(lines) + '\n\n## LaTeX body\n\n```\n' +
            '\n'.join(tex) + '\n```\n')
# emit(), not open(): emit() refuses to write a non-finite macro. A mean over an
# empty result set is nan, and nan is number-shaped -- it reaches LaTeX and
# renders as a result ('leads the best dispatching rule by nan%') rather than as
# the conspicuous '??' of a missing macro. Every number this script puts in the
# manuscript goes through emit(), so a macro that no data produced cannot ship.
from macro_io import emit
emit('paper/macros_e5.tex', macros)

# The table goes straight into the manuscript, which \inputs it. Nothing is
# pasted by hand, so the printed table cannot drift from the data behind it.
#
# The WHOLE tabular is emitted, header and column spec included, and the
# manuscript \inputs it from OUTSIDE any alignment. \input inside a tabular does
# not work: LaTeX's \input leaves a token after the file contents, which opens a
# fresh cell, so the \bottomrule that follows lands inside it. Emitting the
# tabular entire also keeps the column spec next to the code that knows how many
# columns there are.
head = [r'\begin{tabular}{lcccccccc}', r'\toprule',
        r'regime & greedy\tnote{a} & rule & pool\tnote{b} & GA & '
        # No \tnote{c}: the 300s column's prefix caveat lives in the caption
        # (italics + "note c" was an orphan marker with no note behind it).
        r'CP\,\CpFiveSec{}s & CP\,30s & CP\,\CpsatCap{}s & opt \\',
        r'\midrule']
foot = [r'\bottomrule', r'\end{tabular}']
os.makedirs('paper/tables', exist_ok=True)
with open('paper/tables/e5_table.tex', 'w') as f:
    f.write('% auto-generated by scripts/p13_e5_table_fair.py -- do not hand-edit\n')
    f.write('\n'.join(head + tex + foot) + '\n')

print('\n'.join(lines))
for nm, g, n in (('best single rule', g_pdr, n_pdr),
                 ('per-instance portfolio', g_port, n_port),
                 ('GA-30s', g_ga, n_ga)):
    if g:
        m, lo, hi = g
        print(f'\npolicy vs {nm:24s}: {m:+.2f}%  CI [{lo:+.2f},{hi:+.2f}]  '
              f'wins on {n}/{len(TRAIN)} training regimes')
print(f'\n5s search improved on its warm start on ZERO instances in '
      f'{n_dagger} regime(s)')
if have_lat:
    print(f'policy latency: {psec:.2f} s on {cores} CPU cores '
          f'(5s rung = {5.0/psec:.1f}x the policy budget)')
else:
    print('policy latency: NOT MEASURED (see warning above)')
print('\nwrote results/e5_table_fair.md, paper/macros_e5.tex, '
      'paper/tables/e5_table.tex')
if any(p is None for _, p, _, _, _, _, _ in rows):
    print('\nWARNING: the greedy column is EMPTY -- no evaluation results for '
          f'{policy.__doc__.splitlines()[0]}\nThe joint policy must be evaluated '
          '(scripts/run_post_retrain.sh) before this table means anything.')
raise SystemExit(0 if have_lat else 1)
