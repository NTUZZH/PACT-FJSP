r"""P138: the paired difference behind the lowest entry of each row of the decision table.

QUESTION
  The decision table of the paper prints, per row, the mean difference of six columns from the
  policy's greedy schedule, in percent of it, and marks the lowest printed value in bold. A mean
  does not show whether the lowest entry is shorter than the second lowest on the same cells.
  This script computes, for every row of the table, two paired differences per cell with a 95%
  interval: the lowest column minus the second lowest, and the best column that uses the policy
  minus the best column that does not.

ROWS (20)
  50 and 80 modules (12 rows): combinations without setups (NoSetup) and with setups (Setup) at
  60, 300 and 600 s since the request, the decision table on eight cores. The rows at 600 s are
  those of scripts/p124_memetic_parallel.py (both memetic columns the best of eight 600-s runs).
  The rows at 60 s are those of scripts/p137_memetic_one_minute.py after both memetic columns take
  the best of eight 60-s runs, the rows at 300 s those of its five-minute reading (--budget 300)
  after both take the best of eight 300-s runs; their other four columns are p124's. With
  --no-five-minute the rows at 300 s are p124's (both memetic columns the 600-s runs read at
  300 s).
  The training size (8 rows): NoSetup and Setup at 10, 60, 300 and 600 s, the table of
  scripts/p94_timed_inference.py on the instances common to every method of the size
  (results.<size>.decision_common of its record); the memetic algorithm is one run and the two
  columns started from the policy's schedule have no number.

WHAT IS READ (every reader and every statistic by import)
  50 and 80 modules
    p124.read_inputs with p124's own options (budgets 60, 300 and 600 s, --ref-slots as its
    --slots): the eight 600-s runs from each start, CP-SAT from each start, the rule and policy
    records of the request clock and the arrays. p137.read_all with p137's own options (--slots,
    --pol-slots, --ref-slots): the two sets of eight 60-s runs. The six columns of a row come from
    p124.decision_size on these arrays (p124.best_of over all eight runs of a start), at 300 and
    600 s on p124's arrays, at 60 s on p137's arrays with the 60-s best of eight in both memetic
    columns, at 300 s on the arrays of p137.read_all at budget 300 (--five-slots,
    --five-pol-slots, --ref-slots) with the 300-s best of eight in both.
  training size
    p94.collect on the campaign of the training size as p94's record states it (request clock,
    p94's defaults, the budgets of the size), and the columns of p94.analyse_request's table on
    the common instances: the same p94.compare and p94.compare_at calls on the same cells, and
    p94.decision_table over them.
  records  p124's results/memetic_parallel.json, p137's results/memetic_one_minute.json and
           results/memetic_five_minute.json (arm_registry.out), p94's
           results/timed_inference.json of the repository (the record of step timed). Each must
           be a full run on the setting read here (selected training run, CP-SAT configuration,
           B, alpha, seeds, instances, core sets, for the five-minute record budget and reading
           time 300 s), neither a preview nor a dry run (except the records of this script's own
           dry mode, and the five-minute record in a preview).

PER ROW
  values   per cell and column d = 100 (method - greedy) / greedy on the cells of the row, the
           form of the table (p94.gap; the rules against the greedy array, the other columns
           against the greedy schedule where it exists at the time). Every column with a number
           must be a mean over the same instances and combinations, and its mean must print (two
           decimals, p94._printed) as the record prints it; the recomputed row (p124.decision_size
           or p94.decision_table: values, bold columns, cells) must equal the record's row.
  L, R     the lowest column or columns of the printed values (p94.lowest; a tie of printed values
           puts every tied column in L) and the lowest column or columns of the others (R).
  P, O     the lowest column among those that use the policy (policy_sample, memetic_policy,
           cpsat_policy) and among those that do not (rules, memetic_rule, cpsat_rule), by
           p94.lowest over their printed values. At the training size only policy_sample has a
           number among the first three, so P is policy_sample there.
  paired differences
           L minus R and P minus O per cell, in points of the table (percent of the greedy
           schedule); where L, R, P or O holds two tied columns, the column with the lower
           unrounded mean (the first in the order of the table on an exact tie). Mean, 95%
           interval and direction by p94.stat_instance, the estimator of p94.compare_at
           (stats_family.instance_boot: the cluster bootstrap over base instances, all
           combinations of a drawn instance kept together, a fresh default_rng(7) per quantity, B
           of the records; stats_family.direction on the unrounded interval: ahead when the
           interval lies below zero, behind above, neither when it contains zero).
  decided  the interval of L minus R lies below zero and L is one column. A row that is not
           decided is a tie of its two lowest entries.
  L and R are chosen from the same data on which their difference is tested, so the interval of
  L minus R describes the row and is no test of a hypothesis fixed in advance; the same holds for
  P and O. No multiplicity correction is applied over the rows.

COUNTS  over the 12 rows at 50 and 80 modules, over the 4 rows of each reading time (60, 300 and
  600 s) and, apart, over the 8 rows of the training size: rows decided and tied; rows where P is
  ahead of O, behind O, neither; rows whose lowest column is a search started from the policy's
  schedule (memetic_policy or cpsat_policy) and, among them, the rows where P is ahead of O; rows
  whose lowest column uses the policy (policy_sample too). Every row at 60 s equals the row of
  p137's one-minute record after both replacements, every row at 300 s the row of its five-minute
  record after both replacements, every row at 600 s p124's row (values, lowest columns, cells);
  per reading time the count of rows whose lowest column uses the policy equals that record's.

REFUSALS
  A record of p124, p137 (either reading) or p94 that is missing, a preview, a partial or a dry
  run (outside this script's dry mode), or a run on another setting; records of different
  training runs, instance counts or bootstrap counts; an input of p124 or p137 that is missing or
  rejected; a row whose columns are means over different cells; a recomputed row or printed value
  that differs from the record (the refusal names the row and the column). The five-minute
  record is required unless --no-five-minute is given (the rows at 300 s are then p124's, the
  record and the markdown say so in their first lines, and no macro file is written). A refusal
  removes the outputs of the run. --check lists every problem and exits 1 while one exists.

PREVIEW (--min-seeds n --out DIR)
  The rows at 300 s from the preview of the five-minute record (p137 --budget 300 --min-seeds n):
  each 300-s set read on its seeds whose every cell is valid (at least n, seed 1 among those of
  the rule start), which must be the seeds that record names; the record and the markdown say
  preview in their first lines, no macro file is written.

OUTPUTS (arm_registry.out under PACT_OUT, or under --out)
  results/decision_rows.json and .md, manuscript/macros_decision_rows.tex (macro_io.emit; names
  Drow plus letters, one comment per macro, checked against every other macro file under
  manuscript/). Per row, size word S (Fifty, Eighty, Ten), time word T (TenSec, Sixty,
  ThreeHundred, SixHundred) and group G (NoSetup, Setup): \Drow<S><T><G>Lowest, ...RunnerUp,
  ...BestWith, ...BestWithout (the column words of p137.lowest_word), ...LowestVsRunnerUp with
  ...Lo, ...Hi and ...Dir, ...WithVsWithout with ...Lo, ...Hi and ...Dir, ...Tie (a dagger in
  superscript when the row is not decided, empty when it is) and ...LowestIsPolStart (1 when
  the lowest column is a search started from the policy's schedule, 0 otherwise). Counts
  \DrowProd...N (with \DrowProdLowestUsesPolicyN), \DrowProd<T>...N per reading time T in
  Sixty, ThreeHundred, SixHundred (RowN, DecidedN, TiedN, LowestPolStartN, LowestUsesPolicyN,
  WithAheadN, WithNeitherN, WithBehindN) and \DrowTrain...N, and \DrowBoot. The chain step
  decision_rows of scripts/run_analysis.sh brings the macro file to manuscript/.

DRY (--dry [DIR]): p137's synthetic tree with p124's (p137.dry_run at 60 and at 300 s, which
write p124's record and both records of p137),
a training-size world in the same tree (p124.build_world at 10x25 with one memetic run, CP-SAT
on fewer instances than the other methods) and p94's own full run on it, then the whole path of
this script on these records.

Usage (repository root, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p138_decision_rows.py \
      --ref-slots 16-23 8-15 --pol-slots 16-23 8-15 --five-slots 16-23 --five-pol-slots 8-15
  ... --check
  ... --min-seeds 1 --out <folder> --p137-five-json <the preview record of p137 --budget 300>
  ... --no-five-minute --out <folder>
  python scripts/p138_decision_rows.py --dry [DIR]
"""
import argparse
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
_ARGV = list(sys.argv)          # p94 imports p97, which empties sys.argv at import
_CWD = os.getcwd()              # a relative --out is read against it

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p137_memetic_one_minute as P137  # noqa: E402
from macro_io import emit  # noqa: E402

P124 = P137.P124
T = P124.T
P = T.P
SF = P124.SF
Refusal = T.Refusal

GENERATOR = 'scripts/p138_decision_rows.py'
PREFIX = 'Drow'
OUT_FILES = {'macros': 'manuscript/macros_decision_rows.tex',
             'json': 'results/decision_rows.json', 'md': 'results/decision_rows.md'}
P124_JSON = P137.P124_JSON                       # results/memetic_parallel.json
P137_JSON = P137.READINGS[60]['out']['json']     # results/memetic_one_minute.json
FIVE = 300                                       # the reading of p137 with a budget of 300 s
P137_FIVE_JSON = P137.READINGS[FIVE]['out']['json']   # results/memetic_five_minute.json
TIMED_JSON = P124.TIMED_JSON                     # results/timed_inference.json, the repository's
COLUMNS = tuple(c for c, _ in T.DECISION_COLUMNS)
WITH_POLICY = P124.USES_POLICY_COLUMNS           # policy_sample, memetic_policy, cpsat_policy
WITHOUT_POLICY = tuple(c for c in COLUMNS if c not in WITH_POLICY)
POLICY_START = P124.POLICY_START_COLUMNS         # memetic_policy, cpsat_policy
PROD_SIZES = tuple(P124.SIZES)
TRAIN_SIZE = P.TRAINING_SIZE
ENTRIES = tuple(T.group_keys())                  # NoSetup, Setup
ONE_MINUTE = P137.DEFAULT_BUDGET                 # 60
SHORT_BUDGETS = (ONE_MINUTE, FIVE)               # rows whose memetic columns read p137
PROD_BUDGETS = tuple(T.REQUEST_BUDGETS)          # 60, 300, 600
TRAIN_BUDGETS = tuple(T.TRAINING_BUDGETS)        # 10, 60, 300, 600
TIME_WORD = {10: 'TenSec', 60: 'Sixty', 300: 'ThreeHundred', 600: 'SixHundred'}
TIE_MARK = r'\textsuperscript{$\dagger$}'
NO_FIVE_WARNING = ("WITHOUT THE FIVE-MINUTE RECORD: the rows at 300 s at 50 and 80 modules read "
                   "both memetic columns from p124's 600-s runs; not for the paper")
DRY_BOOT = P124.DRY_BOOT
DRY_TRAIN_INST = '0-11'
DRY_TRAIN_CAMPAIGN = (f'{TRAIN_SIZE} = memetic_rule@600:{DRY_TRAIN_INST} cpsat_rule@600:0-5 '
                      f'{T.POLICY_ARM}@{P124.POLICY_BUDGET:g}:{DRY_TRAIN_INST}')
RULES = (
    'rows: NoSetup and Setup at 60, 300 and 600 s at 50x25n30 and 80x25n30 (the decision table on '
    'eight cores; at 60 s both memetic columns the best of eight 60-s runs, at 600 s the best of '
    'eight 600-s runs, at 300 s the runs that the last line of the counts names), and at 10, 60, '
    '300 and 600 s at the training size (the table on '
    'the instances common to every method of the size)',
    'values: per cell 100 (method - greedy) / greedy on the cells of the row; every column with a '
    'number a mean over the same cells; the recomputed row equal to the record\'s row and every '
    'mean printed as the record prints it',
    'L: the lowest column or columns of the printed values (p94.lowest); R: the lowest of the '
    'other columns; P: the lowest of policy_sample, memetic_policy and cpsat_policy; O: the lowest '
    'of rules, memetic_rule and cpsat_rule; of tied columns the one with the lower unrounded mean '
    'enters the paired difference',
    'paired differences L minus R and P minus O per cell, in points of the table; mean, 95% '
    'interval and direction by p94.stat_instance (stats_family.instance_boot over base instances, '
    'combinations of a drawn instance together, fresh default_rng(7) per quantity; '
    'stats_family.direction on the unrounded interval)',
    'decided: the interval of L minus R lies below zero and L is one column; otherwise the row is '
    'a tie of its two lowest entries',
    'L and R (P and O) are chosen from the same data on which their difference is tested: the '
    'interval describes the row and is no test of a hypothesis fixed in advance; no multiplicity '
    'correction over the rows')


# ----------------------------------------------------------------- small helpers
def shown(path):
    return P124.shown(path)


def load(path):
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path)) as f:
        return json.load(f)


def lowest_word(cols):
    return P137.lowest_word(cols)


def stat(d, B, alpha):
    """Mean, 95% interval and direction of the per-cell differences d [instances, combinations]
    by p94.stat_instance, the estimator p94.compare_at applies to an entry."""
    x = T.stat_instance(np.asarray(d, dtype=float), B, alpha)
    return dict(mean=x['mean'], lo=x['lo'], hi=x['hi'], n=x['n'], c=x['c'],
                direction=x['direction'], rng=x['rng'])


def pick(cols, means):
    """The column of a tie of printed values with the lowest unrounded mean (first in the order
    of the table on an exact tie)."""
    return min(cols, key=lambda c: (means[c], COLUMNS.index(c)))


def entry_of(S, col, b, key):
    """The entry of p94.compare or p94.compare_at behind a column of a row, as p94.decision_table
    maps it: rules (iv), policy_sample (vii), the searches from the rule schedule (vi) and from
    the policy's schedule (ii)."""
    if col == 'rules':
        return S['iv'].get('selected', {}).get(key)
    if col == 'policy_sample':
        return S['vii'].get(str(b), {}).get(key)
    method = 'memetic' if col.startswith('memetic') else 'cpsat'
    if col.endswith('_rule'):
        return S['vi'].get(method, {}).get(str(b), {}).get('selected', {}).get(key)
    return S.get('ii', {}).get(method, {}).get(str(b), {}).get(key)


def row_cells(row, cols):
    """{column: (sorted instances, sorted combinations)} of the columns cols of a decision row."""
    return {c: (sorted(int(i) for i in (row.get('instances') or {}).get(c) or []),
                sorted((row.get('combinations_used') or {}).get(c) or [])) for c in cols}


# ----------------------------------------------------------------- one row
def per_cell(first, second, pres, rows, cols):
    """100 (first - second) / second on the cells rows x cols ([instances, combinations], the
    order of p94.compare), or None when a cell lies outside pres or holds no value."""
    sub = np.ix_(rows, cols)
    if not pres[sub].all():
        return None
    d = T.gap(first[sub], second[sub])
    return d if np.isfinite(d).all() else None


def analyse_row(mats, B, alpha):
    """L, R, P and O of one row and the paired differences L minus R and P minus O (module
    docstring, PER ROW). mats: {column: per-cell values [instances, combinations] on the cells of
    the row, or None without a number}."""
    means = {c: float(m.mean()) for c, m in mats.items() if m is not None}
    values = {c: (T.f2(means[c]) if c in means else None) for c in COLUMNS}
    low = T.lowest(values)
    rest = {c: v for c, v in values.items() if c not in low}
    run = T.lowest(rest)
    with_p = T.lowest({c: values[c] for c in WITH_POLICY})
    without = T.lowest({c: values[c] for c in WITHOUT_POLICY})
    out = dict(values=values, means=means, lowest=low, runner_up=run, best_with=with_p,
               best_without=without)
    if low and run:
        lc, rc = pick(low, means), pick(run, means)
        out.update(lowest_column=lc, runner_up_column=rc,
                   lowest_vs_runner_up=stat(mats[lc] - mats[rc], B, alpha))
    else:
        out.update(lowest_column=None, runner_up_column=None, lowest_vs_runner_up=None)
    if with_p and without:
        pc, oc = pick(with_p, means), pick(without, means)
        out.update(with_column=pc, without_column=oc,
                   with_vs_without=stat(mats[pc] - mats[oc], B, alpha))
    else:
        out.update(with_column=None, without_column=None, with_vs_without=None)
    lvr = out['lowest_vs_runner_up']
    out['decided'] = bool(lvr is not None and len(low) == 1 and lvr['direction'] == 'ahead')
    out['printed_tie'] = len(low) > 1
    return out


def row_matrices(what, S, b, key, arrays, inst, combos):
    """The per-cell values of every column with a number of one recomputed row (S: the
    recomputed table, p124.decision_size or the table on the common instances; arrays: {column:
    (first, second, pres)} at budget b; inst, combos: the instances and combinations of the
    arrays). Returns (mats, (instances, combinations) of the row, problems)."""
    row = S['decision'][str(b)][key]
    have = [c for c in COLUMNS if row['values'][c] is not None]
    cells = row_cells(row, have)
    sets = {(tuple(v[0]), tuple(v[1])) for v in cells.values()}
    if len(sets) != 1:
        return None, None, [f'{what}: the columns with a number are means over different cells ('
                            + '; '.join(f'{c} {P.compress(v[0])} x {",".join(v[1])}'
                                        for c, v in cells.items()) + ')']
    ids, comb = next(iter(sets))
    rows = [inst.index(i) for i in ids]
    cols = [j for j, r in enumerate(combos) if r in comb]
    mats, bad = {}, []
    for c in COLUMNS:
        if c not in have:
            mats[c] = None
            continue
        if c not in arrays:
            bad.append(f'{what}, {c}: a number in the row and no array here')
            continue
        m = per_cell(*arrays[c], rows, cols)
        x = entry_of(S, c, b, key)
        if m is None:
            bad.append(f'{what}, {c}: a cell of the row holds no value here')
        elif x is None or 'mean' not in x or abs(float(m.mean()) - x['mean']) > 1e-9 \
                or T.f2(float(m.mean())) != row['values'][c]:
            bad.append(f'{what}, {c}: per-cell mean {float(m.mean()):.6f} here, '
                       f'{None if x is None else x.get("mean")} in the recomputed entry, printed '
                       f'{row["values"][c]}')
        mats[c] = m
    return mats, (list(ids), sorted(comb)), bad


def record_row_problems(what, mine, rec_values, rec_bold, rec_cells, where):
    """Texts of every difference between a recomputed row (values, lowest, cells) and the row of
    a record; empty when they agree."""
    bad = []
    for c in COLUMNS:
        if mine['values'][c] != rec_values.get(c):
            bad.append(f'{what}, {c}: {mine["values"][c] or "no number"} here, '
                       f'{rec_values.get(c) or "no number"} in {where}')
    if rec_bold is not None and sorted(T.lowest(mine['values'])) != sorted(rec_bold):
        bad.append(f'{what}: lowest {T.lowest(mine["values"])} here, bold {rec_bold} in {where}')
    if rec_cells is not None and mine['cells'] != rec_cells:
        bad.append(f'{what}: cells {mine["cells"]} here, {rec_cells} in {where}')
    return bad


# ----------------------------------------------------------------- 50 and 80 modules
def prod_arrays(D, k, t, best, use, bestp, use2, cp):
    """{column: (first, second, pres)} at budget index k (time t) as p124.decision_size passes
    them to p94.compare and p94.compare_at."""
    pol_sel = D['pol_sel'] if D['pol_sel'] is not None else np.full(use.shape, np.nan)
    with np.errstate(invalid='ignore'):
        pol_pres = np.isfinite(pol_sel) & np.isfinite(D['o_pol'])
        g = np.where(D['o_pol'] <= t + T.EPS, pol_sel, np.nan)
    out = dict(rules=(D['rule'], pol_sel, np.isfinite(D['rule']) & np.isfinite(pol_sel)),
               policy_sample=(D['smp'][k], g, D['sp'] & pol_pres),
               memetic_rule=(best[k], g, use & pol_pres),
               cpsat_rule=(D['CP'][k], g, D['CPv'] & pol_pres))
    if bestp is not None:
        out['memetic_policy'] = (bestp[k], g, use2 & pol_pres)
    if cp is not None:
        out['cpsat_policy'] = (cp[0][k], g, cp[1] & pol_pres)
    return out


def prod_namespaces(a):
    """p124's options as its chain step gives them (--ref-slots as its --slots, its default
    budgets) and p137's options (--slots, --pol-slots, --ref-slots)."""
    argv = ['--slots', *a.ref_slots, '--boot', str(a.boot), '--alpha', str(a.alpha),
            '--max-invalid', str(a.max_invalid), '--data-root', a.data_root]
    for opt, val in (('--result-root', a.result_root), ('--bars', a.bars),
                     ('--timed-json', a.timed_json)):
        if val:
            argv += [opt, val]
    pa = P124.build_parser().parse_args(argv)
    bad = P124.check_options(pa)
    argv = ['--slots', *a.slots, '--pol-slots', *a.pol_slots, '--ref-slots', *a.ref_slots,
            '--boot', str(a.boot), '--alpha', str(a.alpha), '--max-invalid', str(a.max_invalid),
            '--data-root', a.data_root]
    for opt, val in (('--result-root', a.result_root), ('--bars', a.bars),
                     ('--p124-json', a.p124_json)):
        if val:
            argv += [opt, val]
    a137 = P137.build_parser().parse_args(argv)
    bad += P137.check_options(a137)
    a5 = None
    if not a.no_five_minute:
        argv = ['--budget', str(FIVE), '--slots', *a.five_slots, '--pol-slots', *a.five_pol_slots,
                '--ref-slots', *a.ref_slots, '--boot', str(a.boot), '--alpha', str(a.alpha),
                '--max-invalid', str(a.max_invalid), '--data-root', a.data_root]
        for opt, val in (('--result-root', a.result_root), ('--bars', a.bars),
                         ('--p124-json', a.p124_json)):
            if val:
                argv += [opt, val]
        with P137.at_budget(FIVE):
            a5 = P137.build_parser().parse_args(argv)
            bad += P137.check_options(a5)
    return pa, a137, a5, bad


def read_five(a, sel, cpsat_arm):
    """The two sets of 300-s runs: p137.read_all with p137's options at budget 300 (--five-slots,
    --five-pol-slots, --ref-slots). Returns (inp, {set name: the seeds read}, problems): every
    seed in the full run, every complete seed of each set in a preview (--min-seeds n, at least n
    in each set, seed 1 among those of the rule start)."""
    a5 = a.a5
    with P137.at_budget(FIVE):
        inp = P137.read_all(a5, sel, cpsat_arm)
        bad = list(inp['missing'])
        if inp['rejected']:
            x = inp['rejected'][0]
            bad.append(f'{len(inp["rejected"])} records read by p137 at {FIVE} s rejected (first '
                       f'{x["id"]}: {x["state"]})')
        if a.min_seeds is None:
            bad += P137.full_gaps(a5, inp)
            used = {name: list(a5.seeds) for name in P137.SETS}
        else:
            used = {name: P137.seed_state(a5, inp, name)[1] for name in P137.SETS}
            if len(used['rule']) < a.min_seeds or P137.SINGLE not in used['rule'] or \
                    len(used['policy']) < a.min_seeds:
                bad.append(f'--min-seeds {a.min_seeds}: seeds with every cell valid '
                           f'{P.compress(used["rule"]) or "none"} ({P137.TOKEN}), '
                           f'{P.compress(used["policy"]) or "none"} ({P137.POL_TOKEN}); the '
                           f'preview needs {a.min_seeds} of them in each set, seed {P137.SINGLE} '
                           f'among those of {P137.TOKEN}')
    return inp, used, bad


def read_production(a, sel, cpsat_arm):
    """The arrays of both production batches: p124.read_inputs at 60, 300 and 600 s,
    p137.read_all at 60 s and, unless --no-five-minute, at 300 s (read_five). Returns
    (dict(data, inp, five, five_seeds, files), problems)."""
    pa, a137 = a.pa124, a.a137
    data, rejected, missing, files, second = P124.read_inputs(pa, sel, cpsat_arm)
    bad = list(missing)
    if rejected:
        bad.append(f'{len(rejected)} records of p124 rejected (first {rejected[0]["id"]}: '
                   f'{rejected[0]["state"]} ({rejected[0]["reason"]}))')
    if second['state'] != 'complete':
        bad.append(f"p124's runs started from the policy's schedule: {second['state']}"
                   + ''.join(f'; {x}' for x in second['problems'][:3]))
    bad += [f'second set of p124: {x}' for x in second['missing'] + [
        f'{y["id"]}: {y["state"]}' for y in second['rejected']]]
    for size, D in data.items():
        inc = P124.incomplete_cells(pa, D)
        if inc:
            bad.append(f'{size}: {len(inc)} cells with fewer than {len(pa.seeds)} valid 600-s runs '
                       f'(first {inc[0][0]}/{inc[0][1]:03d})')
    inp = P137.read_all(a137, sel, cpsat_arm)
    bad += list(inp['missing'])
    if inp['rejected']:
        x = inp['rejected'][0]
        bad.append(f'{len(inp["rejected"])} records read by p137 rejected (first {x["id"]}: '
                   f'{x["state"]})')
    bad += P137.full_gaps(a137, inp)
    five, used = None, None
    if a.a5 is not None:
        five, used, more = read_five(a, sel, cpsat_arm)
        bad += more
    return dict(data=data, inp=inp, five=five, five_seeds=used, files=files), bad


def short_table(pa, inp, size, b, seeds, used):
    """(S, arrays at b, source) of one size: p124.decision_size on p137's arrays read at b s with
    the best of the runs of budget b of the seeds used (used: {set name: seeds}; seeds: the seeds
    of the arrays) in both memetic columns."""
    D = inp['data'][size]
    X = inp['sets']['rule']['arrays'][size]
    XP = inp['sets']['policy']['arrays'][size]
    q = [list(seeds).index(k) for k in used['rule']]
    qp = [list(seeds).index(k) for k in used['policy']]
    use, _, best = P124.best_of(X['M'][q], X['V'][q], len(q))
    usep, _, bestp = P124.best_of(XP['M'][qp], XP['V'][qp], len(qp))
    Q = D['policy']
    cp = (Q['CP'], Q['CPv'])
    S = P124.decision_size(pa, D, best, use, bestp, usep, cp)
    return S, prod_arrays(D, 0, float(b), best, use, bestp, usep, cp), f'p137 at {b} s'


def production_tables(a, prod):
    """{size: {budget: (S, arrays per budget)}}: p124.decision_size at 300 and 600 s on p124's
    arrays (and at 60 s, for the check of p124's record), at 60 s on p137's arrays with the 60-s
    best of eight in both memetic columns ('minute') and, unless --no-five-minute, at 300 s on
    p137's arrays at budget 300 with the 300-s best of the seeds read in both ('five')."""
    pa, a137 = a.pa124, a.a137
    out = {}
    for size in PROD_SIZES:
        D = prod['data'][size]
        use, _, best = P124.best_of(D['M'], D['V'], pa.min_seeds)
        Q = D['policy']
        use2, _, bestp = P124.best_of(Q['M'], Q['V'], len(pa.seeds))
        cp = (Q['CP'], Q['CPv'])
        S = P124.decision_size(pa, D, best, use, bestp, use2, cp)
        tab = {}
        for k, b in enumerate(pa.budgets):
            tab[int(b)] = (S, prod_arrays(D, k, float(b), best, use, bestp, use2, cp), 'p124')
        every = {name: list(a137.seeds) for name in P137.SETS}
        tab['minute'] = short_table(a137.pa, prod['inp'], size, ONE_MINUTE, a137.seeds, every)
        if prod['five'] is not None:
            tab['five'] = short_table(a.a5.pa, prod['five'], size, FIVE, a.a5.seeds,
                                      prod['five_seeds'])
        out[size] = tab
    return out


def production_rows(a, tables, J124, short, w124):
    """The twelve rows of 50 and 80 modules (module docstring, PER ROW) and the problems.
    short: {budget: (p137's record of that budget, its shown path)} of the rows whose memetic
    columns read p137 (60 s, and 300 s unless --no-five-minute); each such row must equal the
    record's row after both replacements, the other rows p124's record."""
    pa = a.pa124
    rows, bad = [], []
    tab_of = {ONE_MINUTE: 'minute', FIVE: 'five'}
    for size in PROD_SIZES:
        tab = tables[size]
        for b in PROD_BUDGETS:
            for key in ENTRIES:
                what = f'{size}, {T.entry_text(key)}, {b} s'
                S, arrays, source = tab[tab_of[b]] if b in short else tab[b]
                mats, cells, prob = row_matrices(what, S, b, key, arrays, list(pa.inst),
                                                 list(pa.combos))
                bad += prob
                if mats is None:
                    continue
                r = analyse_row(mats, pa.boot, pa.alpha)
                mine = dict(values=r['values'], cells=cells)
                rec = J124['decision_table']['sizes'][size]['rows'][str(b)][key]
                # the cells of p124's row: every column with a number over the same cells
                rc = row_cells(rec, [c for c in COLUMNS if rec['values'].get(c) is not None])
                sets = {(tuple(v[0]), tuple(v[1])) for v in rc.values()}
                rcells = (list(next(iter(sets))[0]), list(next(iter(sets))[1])) \
                    if len(sets) == 1 else None
                if len(sets) != 1:
                    bad.append(f'{what}: the columns of the row of {w124} are means over '
                               'different cells')
                if b in short:
                    J, w = short[b]
                    one = J['decision']['rows'][size][key]
                    bad += record_row_problems(what, mine, one['after_values'], one['after'],
                                               (one['instances'], sorted(one['combinations'])),
                                               w)
                    # the four columns p137 takes from p124's record
                    for c in P124.P94_COLUMNS:
                        if r['values'][c] != rec['values'].get(c):
                            bad.append(f'{what}, {c}: {r["values"][c]} here, '
                                       f'{rec["values"].get(c)} in {w124}')
                    # p124's own row at this time, with the 600-s runs, on p124's arrays
                    S6, arr6, _ = tab[b]
                    m6, c6, p6 = row_matrices(what + ' (600-s runs)', S6, b, key, arr6,
                                              list(pa.inst), list(pa.combos))
                    bad += p6
                    if m6 is not None:
                        v6 = {c: (T.f2(float(m.mean())) if m is not None else None)
                              for c, m in m6.items()}
                        bad += record_row_problems(what + ' (600-s runs)',
                                                   dict(values=v6, cells=c6), rec['values'],
                                                   rec.get('bold'), rcells, w124)
                else:
                    bad += record_row_problems(what, mine, rec['values'], rec.get('bold'),
                                               rcells, w124)
                rows.append(dict(size=size, budget=b, entry=key, part='prod', cells=cells,
                                 source=source, **r))
    return rows, bad


# ----------------------------------------------------------------- the training size
def training_campaign(J94, where):
    """The campaign of the training size in p94's record, and problems."""
    camps = [c for c in J94.get('campaigns') or [] if T.parse_campaign(c)[0] == [TRAIN_SIZE]]
    if len(camps) != 1:
        return None, [f'{where}: {len(camps)} campaigns of {TRAIN_SIZE} alone '
                      f'({J94.get("campaigns")})']
    return camps[0], []


def p94_namespace(a, campaign):
    """p94's options as its chain step gives them (its defaults), for the campaign of the
    training size."""
    argv = ['--campaign', campaign, '--primary', str(T.PRIMARY), '--boot', str(a.boot),
            '--alpha', str(a.alpha), '--max-invalid', str(a.max_invalid),
            '--data-root', a.data_root]
    for opt, val in (('--result-root', a.result_root), ('--bars', a.bars)):
        if val:
            argv += [opt, val]
    a94 = T.build_parser().parse_args(argv)
    bad = T.check_options(a94)
    for opt, default in (('policy_batch', T.P97.DEFAULT_BATCH),
                         ('policy_workers', T.P97.DEFAULT_WORKERS),
                         ('policy_base_seed', T.P97.DEFAULT_BASE_SEED)):
        if getattr(a94, opt) is None:
            setattr(a94, opt, default)
    return a94, bad


def read_training(a, campaign, sel, cpsat_arm):
    """p94.collect on the campaign of the training size, request clock, at the budgets of the
    size. Returns the data of the size."""
    a94, bad = p94_namespace(a, campaign)
    if bad:
        raise Refusal(bad)
    specs = [T.parse_campaign(campaign)]
    budgets = sorted(set(a94.training_budgets))
    grid = [float(b) for b in budgets]
    data, gaps, _, files, _, _ = T.collect(a94, budgets, 'request', specs, sel, cpsat_arm, False,
                                           None, False, grid, size_budgets={TRAIN_SIZE: budgets})
    D = data[TRAIN_SIZE]
    D['files'] = files
    return D, a94


def training_table(D, combos, B, alpha):
    """The table on the instances common to every method of the training size, by the calls of
    p94.analyse_request (the block decision_common): returns (Sc with decision, {budget: arrays},
    common instances), or (None, None, common) when the methods share their instances."""
    A, Ap = D['A'], D['A_present']
    rule, pol_sel = D['rule'], D['pol_sel']
    o_rule, o_pol, smp, sp = D['o_rule'], D['o_pol'], D['S'], D['S_present']
    sbud, budgets = D['policy_budget_s'], D['budgets']
    grid = [float(b) for b in budgets]
    gi = {t: k for k, t in enumerate(grid)}
    has_pol = np.isfinite(o_pol)
    pol_pres = has_pol & np.isfinite(pol_sel)
    design = D['design']
    want = T.column_design(D['instances'], design)
    common = T.common_instances(D['instances'], design)
    if not any(v != common for v in want.values()):
        return None, None, common
    cr = np.isin(np.asarray(D['instances']), common)[:, None]
    methods = [m for m in T.METHOD_WORD if any(P.ARMS[x][0] == m for x in A)]
    Sc = dict(instances=D['instances'], budgets=list(budgets), iv={}, vii={}, vi={}, ii={})
    uc = {}
    rule_c = np.where(cr, rule, np.nan)
    Sc['iv']['selected'] = T.compare(rule_c, pol_sel, combos, B, alpha,
                                     used=uc.setdefault(('rules', None), {}))
    arrays = {}
    for b in budgets:
        t = float(b)
        k = gi[t]
        with np.errstate(invalid='ignore'):
            g_sel = np.where(o_pol <= t + T.EPS, pol_sel, np.nan)
        held = sbud is not None and t > sbud + T.EPS
        arr = dict(rules=(rule_c, pol_sel, np.isfinite(rule_c) & np.isfinite(pol_sel)))
        if sp.any():
            pres = sp & pol_pres & cr
            Sc['vii'][str(b)] = T.mark_held(T.compare_at(
                smp[k], g_sel, pres, combos, B, alpha,
                used=uc.setdefault(('policy_sample', str(b)), {})), held, sbud)
            arr['policy_sample'] = (smp[k], g_sel, pres)
        for m in methods:
            ra, pa_ = T.ARM_OF[(m, 'rule')], T.ARM_OF[(m, 'policy')]
            if ra in A:
                pres = Ap[ra] & pol_pres & cr
                Sc['vi'].setdefault(m, {})[str(b)] = dict(selected=T.compare_at(
                    A[ra][k], g_sel, pres, combos, B, alpha, used=uc.setdefault((ra, str(b)), {})))
                arr[ra] = (A[ra][k], g_sel, pres)
            if pa_ in A:
                pres = Ap[pa_] & pol_pres & cr
                Sc['ii'].setdefault(m, {})[str(b)] = T.compare_at(
                    A[pa_][k], g_sel, pres, combos, B, alpha, used=uc.setdefault((pa_, str(b)), {}))
                arr[pa_] = (A[pa_][k], g_sel, pres)
        arrays[int(b)] = arr
    Sc['decision'] = T.decision_table(Sc, np.where(cr, o_rule, np.nan), combos, uc)
    return Sc, arrays, common


def training_rows(a, D, J94, where, combos):
    """The eight rows of the training size (module docstring, PER ROW) and the problems."""
    B, alpha = a.boot, a.alpha
    Sc, arrays, common = training_table(D, combos, B, alpha)
    dc = ((J94.get('results') or {}).get(TRAIN_SIZE) or {}).get('decision_common')
    if Sc is None or dc is None:
        return [], [f'{TRAIN_SIZE}: the table on the common instances is '
                    f'{"absent here" if Sc is None else "absent from " + where}']
    bad = []
    if list(dc.get('instances') or []) != list(common):
        bad.append(f'{TRAIN_SIZE}: common instances {P.compress(common)} here, '
                   f'{dc.get("instances")} in {where}')
    rows = []
    for b in TRAIN_BUDGETS:
        for key in ENTRIES:
            what = f'{TRAIN_SIZE}, {T.entry_text(key)}, {b} s'
            rec = (dc.get('rows') or {}).get(str(b), {}).get(key)
            if rec is None or b not in arrays:
                bad.append(f'{what}: no row in {where}' if rec is None
                           else f'{what}: no budget here')
                continue
            mats, cells, prob = row_matrices(what, Sc, b, key, arrays[b], list(D['instances']),
                                             list(combos))
            bad += prob
            if mats is None:
                continue
            r = analyse_row(mats, B, alpha)
            rc = row_cells(rec, [c for c in COLUMNS if rec['values'].get(c) is not None])
            sets = {(tuple(v[0]), tuple(v[1])) for v in rc.values()}
            if len(sets) != 1:
                bad.append(f'{what}: the columns of the row of {where} are means over different '
                           'cells')
            rcells = (list(next(iter(sets))[0]), list(next(iter(sets))[1])) if len(sets) == 1 \
                else None
            bad += record_row_problems(what, dict(values=r['values'], cells=cells),
                                       rec['values'], rec.get('bold'), rcells, where)
            rows.append(dict(size=TRAIN_SIZE, budget=b, entry=key, part='train', cells=cells,
                             source='p94', **r))
    return rows, bad


# ----------------------------------------------------------------- records
def record_path(opt, rel_path, through_out=True):
    if opt:
        return opt
    return REG.out(rel_path) if through_out else os.path.join(ROOT, rel_path)


def read_five_record(a, sel, cpsat_arm, dry):
    """(p137's record at 300 s or None, its shown path, problems): a full run on the setting
    read here (budget and reading time 300 s, --five-slots, --five-pol-slots, --ref-slots) with
    its rows of the decision table, or with --min-seeds a preview."""
    path = record_path(a.p137_five_json, P137_FIVE_JSON)
    where = shown(path)
    if not os.path.exists(path):
        return None, where, [f'{where}: no record of p137 at {FIVE} s (step memetic_five_minute; '
                             f'--no-five-minute reads the rows at {FIVE} s from p124\'s record)']
    J = load(path)
    a5, pa = a.a5, a.pa124
    preview = a.min_seeds is not None
    want = (('run', 'preview' if preview else 'full'), ('dry', bool(dry)), ('selected_run', sel),
            ('cpsat_configuration', cpsat_arm), ('boot', a.boot), ('alpha', a.alpha),
            ('budget_s', FIVE), ('read_at_s', FIVE), ('ref_seeds', list(pa.seeds)),
            ('instances', list(pa.inst)), ('slots', list(a5.slots)),
            ('pol_slots', list(a5.pol_slots)), ('ref_slots', list(a5.ref_slots)))
    if not preview:
        want += (('seeds', list(a5.seeds)), ('pol_seeds', list(a5.seeds)), ('warning', None))
    bad = [f'{where}: {k} {J.get(k)!r}, this run reads {v!r}' for k, v in want if J.get(k) != v]
    dec = J.get('decision') or {}
    if not dec.get('rows'):
        bad.append(f'{where}: no rows of the decision table at {FIVE} s')
    else:
        for size in PROD_SIZES:
            lack = [k for k in ENTRIES if k not in (dec['rows'].get(size) or {})]
            if lack:
                bad.append(f'{where}: no row {", ".join(lack)} at {FIVE} s at {size}')
    return J, where, bad


def read_records(a, sel, cpsat_arm, dry):
    """(J124, J137, J5, J94, their shown paths, problems): each record a full run on the setting
    read here (J5, p137's record at 300 s: None with --no-five-minute, a preview with
    --min-seeds), and the records on one training run, one B and one set of instances."""
    a137, pa = a.a137, a.pa124
    J124, w124, bad = P137.read_record(a137, a137.pa, sel, cpsat_arm, dry)
    if J124 is not None:
        if J124.get('budgets') != list(pa.budgets):
            bad.append(f'{w124}: budgets {J124.get("budgets")}, this run reads {list(pa.budgets)}')
        if J124.get('warning'):
            bad.append(f'{w124}: {J124["warning"]}')
        if (J124.get('decision_table') or {}).get('warning'):
            bad.append(f'{w124}: decision table {J124["decision_table"]["warning"]}')
    p137 = record_path(a.p137_json, P137_JSON)
    w137 = shown(p137)
    J137 = None
    if not os.path.exists(p137):
        bad.append(f'{w137}: no record of p137 (step memetic_one_minute)')
    else:
        J137 = load(p137)
        want = (('run', 'full'), ('dry', bool(dry)), ('selected_run', sel),
                ('cpsat_configuration', cpsat_arm), ('boot', a.boot), ('alpha', a.alpha),
                ('seeds', list(a137.seeds)), ('pol_seeds', list(a137.seeds)),
                ('ref_seeds', list(pa.seeds)), ('instances', list(pa.inst)),
                ('slots', list(a137.slots)), ('pol_slots', list(a137.pol_slots)),
                ('ref_slots', list(a137.ref_slots)), ('warning', None))
        bad += [f'{w137}: {k} {J137.get(k)!r}, this run reads {v!r}' for k, v in want
                if J137.get(k) != v]
        dec = J137.get('decision') or {}
        if not dec.get('rows') or not dec.get('all_rows'):
            bad.append(f'{w137}: no one-minute rows of the decision table')
        else:
            for size in PROD_SIZES:
                lack = [k for k in ENTRIES if k not in (dec['rows'].get(size) or {})]
                if lack:
                    bad.append(f'{w137}: no one-minute row {", ".join(lack)} at {size}')
    p94 = record_path(a.timed_json, TIMED_JSON, through_out=False)
    w94 = shown(p94)
    J94 = None
    if not os.path.exists(p94):
        bad.append(f'{w94}: no record of p94 (step timed)')
    else:
        J94 = load(p94)
        want = (('run', 'full'), ('partial', False), ('stand_in', False), ('origin', 'request'),
                ('boot', a.boot), ('alpha', a.alpha), ('cpsat_arm', cpsat_arm), ('tag', None))
        bad += [f'{w94}: {k} {J94.get(k)!r}, this run reads {v!r}' for k, v in want
                if J94.get(k) != v]
        if (J94.get('selection') or {}).get('seed') != sel:
            bad.append(f'{w94}: selected run {(J94.get("selection") or {}).get("seed")}, this run '
                       f'reads s{sel}')
        if (J94.get('budgets_per_size') or {}).get(TRAIN_SIZE) != list(TRAIN_BUDGETS):
            bad.append(f'{w94}: budgets of {TRAIN_SIZE} '
                       f'{(J94.get("budgets_per_size") or {}).get(TRAIN_SIZE)}, the table reads '
                       f'{list(TRAIN_BUDGETS)}')
        if (((J94.get('results') or {}).get(TRAIN_SIZE) or {}).get('decision_common') is None):
            bad.append(f'{w94}: no table on the common instances at {TRAIN_SIZE}')
    J5, w5 = None, None
    if a.a5 is not None:
        J5, w5, more = read_five_record(a, sel, cpsat_arm, dry)
        bad += more
    for w, J in ((w137, J137), (w5, J5)):
        if J124 is not None and J is not None and J124.get('instances') != J.get('instances'):
            bad.append(f'{w124} and {w}: instances {J124.get("instances")} and '
                       f'{J.get("instances")}')
    runs = {w: J.get('selected_run') for w, J in ((w124, J124), (w137, J137), (w5, J5))
            if J is not None}
    if J94 is not None:
        runs[w94] = (J94.get('selection') or {}).get('seed')
    if len(set(runs.values())) > 1:
        bad.append('records of different training runs: ' + ', '.join(f'{w} s{v}'
                                                                      for w, v in runs.items()))
    boots = {w: J.get('boot') for w, J in ((w124, J124), (w137, J137), (w5, J5), (w94, J94))
             if J is not None}
    if len(set(boots.values())) > 1:
        bad.append('records of different bootstrap counts: ' + ', '.join(f'{w} B = {v}'
                                                                        for w, v in boots.items()))
    return J124, J137, J5, J94, (w124, w137, w5, w94), bad


# ----------------------------------------------------------------- counts
def count_rows(rows):
    """The counts of the module docstring over one list of rows."""
    def p_dir(r):
        x = r['with_vs_without']
        return None if x is None else x['direction']
    pol_start = [r for r in rows if set(r['lowest']) & set(POLICY_START)]
    return dict(rows=len(rows), decided=sum(r['decided'] for r in rows),
                tied=sum(not r['decided'] for r in rows),
                printed_ties=sum(r['printed_tie'] for r in rows),
                with_ahead=sum(p_dir(r) == 'ahead' for r in rows),
                with_neither=sum(p_dir(r) == 'neither' for r in rows),
                with_behind=sum(p_dir(r) == 'behind' for r in rows),
                without_pair=sum(p_dir(r) is None for r in rows),
                lowest_pol_start=len(pol_start),
                lowest_pol_start_ahead=sum(p_dir(r) == 'ahead' for r in pol_start),
                lowest_uses_policy=sum(bool(set(r['lowest']) & set(WITH_POLICY)) for r in rows))


def count_problems(rows, J124, short, w124):
    """Per reading time, the count of the four rows at 50 and 80 modules whose lowest column
    uses the policy against the record each row equals: at 60 s and (unless --no-five-minute)
    300 s p137's record of that budget (decision.counts.uses_policy_after over its four rows),
    otherwise p124's record (its bold columns). short: {budget: (record, shown path)}. Returns
    the problems."""
    bad = []
    for b in PROD_BUDGETS:
        mine = count_rows([r for r in rows if r['budget'] == b])['lowest_uses_policy']
        if b in short:
            J, w = short[b]
            want = ((J.get('decision') or {}).get('counts') or {}).get('uses_policy_after')
        else:
            w = w124
            want = sum(bool(set(J124['decision_table']['sizes'][size]['rows'][str(b)][key]
                                .get('bold') or []) & set(WITH_POLICY))
                       for size in PROD_SIZES for key in ENTRIES)
        if mine != want:
            bad.append(f'rows at {b} s whose lowest column uses the policy: {mine} here, {want} '
                       f'in {w}')
    return bad


# ----------------------------------------------------------------- outputs
def f2(x):
    return T.f2(x)


def macro_lines(res):
    B = res['boot']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["setting"]}; selected training run s{res["selected_run"]}; CP-SAT configuration '
         f'{res["cpsat_configuration"]}; B = {B}, alpha = {res["alpha"]}.',
         '% Rows of the decision table: per cell 100 (method - greedy) / greedy on the cells of '
         'the row. L: the lowest column of the printed values; R: the lowest of the other columns; '
         "P: the lowest of the columns that use the policy (its sampled schedules, a search "
         "started from its schedule); O: the lowest of the other three. Paired differences in "
         'points of the table, mean and 95% interval by p94.stat_instance (instance-level '
         f'cluster bootstrap over base instances, fresh default_rng({T.INST_SEED}) per quantity), '
         'direction from the unrounded interval (ahead: below zero).',
         '% L and R (P and O) are chosen from the same data on which their difference is tested, '
         'so an interval describes its row and is no test of a hypothesis fixed in advance; no '
         'multiplicity correction over the rows.']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}Boot', B, 'bootstrap resamples of every interval of this file')
    for r in res['rows']:
        base = f'{PREFIX}{T.SIZE_WORD[r["size"]]}{TIME_WORD[r["budget"]]}{r["entry"]}'
        what = (f'{r["size"]}, {T.entry_text(r["entry"])}, {r["budget"]} s since the request '
                f'({P.compress(r["cells"][0])} x {",".join(r["cells"][1])})')
        shown_row = '; '.join(f'{c} {r["values"][c] or "no number"}' for c in COLUMNS)
        nc(base + 'Lowest', lowest_word(r['lowest']), f'{what}: the lowest column (row: '
           f'{shown_row})')
        nc(base + 'RunnerUp', lowest_word(r['runner_up']), f'{what}: the second lowest column')
        nc(base + 'BestWith', lowest_word(r['best_with']), f'{what}: the lowest column that uses '
           'the policy')
        nc(base + 'BestWithout', lowest_word(r['best_without']), f'{what}: the lowest column '
           'that does not use the policy')
        for word, key, first, second in (
                ('LowestVsRunnerUp', 'lowest_vs_runner_up', 'lowest_column', 'runner_up_column'),
                ('WithVsWithout', 'with_vs_without', 'with_column', 'without_column')):
            x = r[key]
            if x is None:
                continue
            text = f'{what}: {r[first]} minus {r[second]}, points of the table'
            nc(base + word, f2(x['mean']), f'{text}: mean over {x["n"]} instances x {x["c"]} '
               'combinations')
            nc(base + word + 'Lo', f2(x['lo']), f'{text}: lower end of the 95% interval')
            nc(base + word + 'Hi', f2(x['hi']), f'{text}: upper end of the 95% interval')
            nc(base + word + 'Dir', SF.TEX[x['direction']],
               f'{text}: direction of [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
        nc(base + 'Tie', '' if r['decided'] else TIE_MARK,
           f'{what}: a dagger when the interval of the lowest minus the second lowest column does '
           f'not lie below zero (a tie of the two lowest entries), empty when it does '
           f'({"decided" if r["decided"] else "not decided"})')
        nc(base + 'LowestIsPolStart', int(bool(set(r['lowest']) & set(POLICY_START))),
           f"{what}: 1 when the lowest column is a search started from the policy's schedule, 0 "
           'otherwise')
    for part, word, text in (('prod', 'Prod', 'rows at 50 and 80 modules (NoSetup and Setup at '
                              '60, 300 and 600 s)'),
                             ('train', 'Train', 'rows at the training size (NoSetup and Setup at '
                              '10, 60, 300 and 600 s)')):
        c = res['counts'][part]
        nc(f'{PREFIX}{word}RowN', c['rows'], text)
        nc(f'{PREFIX}{word}DecidedN', c['decided'], f'{text} whose lowest minus second lowest '
           'column has its interval below zero')
        nc(f'{PREFIX}{word}TiedN', c['tied'], f'{text} whose two lowest entries are tied')
        nc(f'{PREFIX}{word}WithAheadN', c['with_ahead'], f'{text} where the best column that uses '
           'the policy is ahead of the best that does not')
        nc(f'{PREFIX}{word}WithNeitherN', c['with_neither'], f'{text} where that interval '
           'contains zero')
        nc(f'{PREFIX}{word}WithBehindN', c['with_behind'], f'{text} where the best column that '
           'uses the policy is behind the best that does not')
        nc(f'{PREFIX}{word}LowestPolStartN', c['lowest_pol_start'], f'{text} whose lowest column '
           "is a search started from the policy's schedule")
        nc(f'{PREFIX}{word}LowestPolStartAheadN', c['lowest_pol_start_ahead'], f'{text} whose '
           "lowest column is a search started from the policy's schedule and where the best "
           'column that uses the policy is ahead of the best that does not')
        if part == 'prod':
            nc(f'{PREFIX}{word}LowestUsesPolicyN', c['lowest_uses_policy'], f'{text} whose lowest '
               "column uses the policy (its sampled schedules, or a search started from its "
               'schedule)')
    for b, c in res['counts'].get('prod_at', {}).items():
        word = f'{PREFIX}Prod{TIME_WORD[int(b)]}'
        text = f'rows at 50 and 80 modules at {b} s since the request (NoSetup and Setup)'
        for name, key, tail in (
                ('RowN', 'rows', ''),
                ('DecidedN', 'decided', ' whose lowest minus second lowest column has its '
                 'interval below zero'),
                ('TiedN', 'tied', ' whose two lowest entries are tied'),
                ('LowestPolStartN', 'lowest_pol_start', " whose lowest column is a search started "
                 "from the policy's schedule"),
                ('LowestUsesPolicyN', 'lowest_uses_policy', ' whose lowest column uses the policy '
                 "(its sampled schedules, or a search started from its schedule)"),
                ('WithAheadN', 'with_ahead', ' where the best column that uses the policy is ahead '
                 'of the best that does not'),
                ('WithNeitherN', 'with_neither', ' where that interval contains zero'),
                ('WithBehindN', 'with_behind', ' where the best column that uses the policy is '
                 'behind the best that does not')):
            nc(word + name, c[key], text + tail)
    return L


def check_names(lines, own):
    """Every name Drow plus letters, unique, the prefix used by no other macro file under
    manuscript/ (p124.defined_macros), no name defined elsewhere."""
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x)]
    if bad:
        raise Refusal([f'macro names that are not {PREFIX} plus letters: {bad[:5]}'])
    if len(names) != len(set(names)):
        raise Refusal([f'duplicate macro names: '
                       f'{sorted({x for x in names if names.count(x) > 1})[:5]}'])
    have = P124.defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refusal([f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})'])
    clash = sorted(set(names) & set(have))
    if clash:
        raise Refusal([f'macro names defined in another file: {clash[:5]}'])
    return names


def diff_text(x):
    if x is None:
        return '-'
    return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}] {x["direction"]}'


def markdown(res):
    md = [f'# Paired differences behind the lowest entry of each row of the decision table '
          f'({res["run"]} run)', '']
    if res.get('warning'):
        md += [f'**{res["warning"]}**', '']
    md += [f'Generated by {GENERATOR}. {res["setting"]}; selected training run '
          f's{res["selected_run"]}; CP-SAT configuration {res["cpsat_configuration"]}; B = '
          f'{res["boot"]}. Records: {", ".join(res["records"].values())}.', '', '## Rules', '']
    md += [f'- {x}' for x in res['rules']] + ['']
    md += ['## Rows', '',
           'Values: difference from the greedy schedule in % of it (the table). L minus R and P '
           'minus O in points of the table, mean [95% interval] direction.', '',
           '| size | entry | time, s | ' + ' | '.join(COLUMNS) + ' | L | R | L minus R | P | O | '
           'P minus O | decided |', '|---|---|---|' + '---|' * (len(COLUMNS) + 7)]
    for r in res['rows']:
        md.append(f'| {r["size"]} | {T.entry_text(r["entry"])} | {r["budget"]} | '
                  + ' | '.join(r['values'][c] or '-' for c in COLUMNS)
                  + f' | {lowest_word(r["lowest"])} | {lowest_word(r["runner_up"])} | '
                  f'{diff_text(r["lowest_vs_runner_up"])} | {lowest_word(r["best_with"])} | '
                  f'{lowest_word(r["best_without"])} | {diff_text(r["with_vs_without"])} | '
                  f'{"yes" if r["decided"] else "no (tie)"} |')
    md += ['', '## Counts', '']
    for part, text in (('prod', '50 and 80 modules'), ('train', 'training size')):
        c = res['counts'][part]
        md.append(f'- {text}: rows {c["rows"]}; decided {c["decided"]}, tied {c["tied"]} '
                  f'(printed ties {c["printed_ties"]}); P against O ahead {c["with_ahead"]}, '
                  f'neither {c["with_neither"]}, behind {c["with_behind"]}; lowest column a '
                  f"search started from the policy's schedule {c['lowest_pol_start']} (P ahead "
                  f'of O in {c["lowest_pol_start_ahead"]}); lowest column uses the policy '
                  f'{c["lowest_uses_policy"]}')
    for b, c in res['counts']['prod_at'].items():
        md.append(f'- 50 and 80 modules at {b} s: rows {c["rows"]}; decided {c["decided"]}, tied '
                  f'{c["tied"]}; lowest column a search started from the policy\'s schedule '
                  f'{c["lowest_pol_start"]}, uses the policy {c["lowest_uses_policy"]}; P against '
                  f'O ahead {c["with_ahead"]}, neither {c["with_neither"]}, behind '
                  f'{c["with_behind"]}')
    md.append(f'- rows at 300 s: memetic columns from {res["five_minute"]}')
    return '\n'.join(md) + '\n'


def output_paths(a):
    if a.out:
        return {k: os.path.join(a.out, v) for k, v in OUT_FILES.items()}
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(paths):
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)


# ----------------------------------------------------------------- the run
def check_options(a):
    """Option problems (empty when none); sets a.pa124, a.a137 and a.a5 (p137's options at
    300 s, None with --no-five-minute)."""
    bad = []
    if a.out:
        bad += P137.out_refusals(a.out)
    if a.min_seeds is not None:
        if not 1 <= a.min_seeds <= len(P137.SEEDS):
            bad.append(f'--min-seeds must lie between 1 and {len(P137.SEEDS)}')
        if a.no_five_minute:
            bad.append('--min-seeds previews the five-minute record and --no-five-minute reads '
                       'none; give one')
        if a.check:
            bad.append('--check reads only and --min-seeds writes a preview; give one')
        elif not a.out:
            bad.append('--min-seeds needs --out (a preview is not for the paper)')
    for s in list(a.slots) + list(a.pol_slots) + list(a.ref_slots) + list(a.five_slots) + \
            list(a.five_pol_slots):
        try:
            P.parse_cores(s)
        except (ValueError, SystemExit):
            bad.append(f'core set {s!r}')
    if bad:
        return bad
    a.pa124, a.a137, a.a5, more = prod_namespaces(a)
    return bad + more


def analyse(a, sel, cpsat_arm, dry=False):
    """Records, inputs, the 20 rows and the counts; raises Refusal with every problem."""
    J124, J137, J5, J94, (w124, w137, w5, w94), bad = read_records(a, sel, cpsat_arm, dry)
    if bad:
        raise Refusal(bad)
    campaign, bad = training_campaign(J94, w94)
    if bad:
        raise Refusal(bad)
    prod, bad = read_production(a, sel, cpsat_arm)
    if bad:
        raise Refusal(bad)
    if J5 is not None:
        for name, key in (('rule', 'seeds'), ('policy', 'pol_seeds')):
            if J5.get(key) != prod['five_seeds'][name]:
                bad.append(f'{w5}: {key} {J5.get(key)}, the seeds of {P137.SETS[name][2]} with '
                           f'every cell valid here {prod["five_seeds"][name]}')
        if bad:
            raise Refusal(bad)
    D, a94 = read_training(a, campaign, sel, cpsat_arm)
    tables = production_tables(a, prod)
    short = {ONE_MINUTE: (J137, w137)}
    if J5 is not None:
        short[FIVE] = (J5, w5)
    rows_p, bad = production_rows(a, tables, J124, short, w124)
    rows_t, bad2 = training_rows(a, D, J94, w94, list(a94.combos))
    if bad or bad2:
        raise Refusal(bad + bad2)
    if len(rows_p) != len(PROD_SIZES) * len(PROD_BUDGETS) * len(ENTRIES) or \
            len(rows_t) != len(TRAIN_BUDGETS) * len(ENTRIES):
        raise Refusal([f'{len(rows_p)} rows at 50 and 80 modules and {len(rows_t)} at the '
                       'training size'])
    counts = dict(prod=count_rows(rows_p), train=count_rows(rows_t),
                  prod_at={str(b): count_rows([r for r in rows_p if r['budget'] == b])
                           for b in PROD_BUDGETS})
    bad = count_problems(rows_p, J124, short, w124)
    if bad:
        raise Refusal(bad)
    records = dict(p124=w124, p137=w137, p94=w94)
    if w5 is not None:
        records['p137_five'] = w5
    return dict(rows=rows_p + rows_t, counts=counts, records=records,
                campaign=campaign, common=list(rows_t[0]['cells'][0]),
                five_minute=(f'the 300-s runs, {w5}' + (
                    f' (preview on seeds {P.compress(J5["seeds"])} and '
                    f'{P.compress(J5["pol_seeds"])})' if a.min_seeds is not None else '')
                    if J5 is not None else "the 600-s runs of p124's record (--no-five-minute)"),
                five_seeds=prod['five_seeds'],
                files=sorted(set(prod['files']) | set(D['files'])))


def run(a, sel=None, cpsat_arm=None, chosen=True, dry=False):
    """Selections, inputs, analysis and the outputs; a refusal removes the outputs. Returns
    (result, macro names, paths)."""
    paths = output_paths(a)
    remove_outputs(paths)
    try:
        if chosen:
            sel, cpsat_arm, problems = P124.selections(a.pa124)
            if problems or sel is None or cpsat_arm is None:
                raise Refusal(problems or ['no selected training run or no CP-SAT configuration'])
        got = analyse(a, sel, cpsat_arm, dry)
        full = a.min_seeds is None and not a.no_five_minute
        res = dict(warning=None if full else (
            NO_FIVE_WARNING if a.no_five_minute else
            f'PREVIEW: the rows at 300 s read the 300-s runs of seeds '
            f'{P.compress(got["five_seeds"]["rule"])} from the rule schedule and seeds '
            f'{P.compress(got["five_seeds"]["policy"])} from the policy\'s schedule (--min-seeds '
            f'{a.min_seeds}); not for the paper'))
        if full:
            del res['warning']
        res.update(generator=GENERATOR,
                   run='full' if full else ('preview' if a.min_seeds is not None
                                            else 'without the five-minute record'),
                   dry=bool(dry), setting=P124.setting_text(),
                   selected_run=sel, cpsat_configuration=cpsat_arm, boot=a.boot, alpha=a.alpha,
                   origin='request', columns=dict(T.DECISION_COLUMNS),
                   with_policy=list(WITH_POLICY), without_policy=list(WITHOUT_POLICY),
                   rules=list(RULES),
                   statistics=('p94.stat_instance (stats_family.instance_boot, fresh '
                               f'default_rng({T.INST_SEED}) per quantity; stats_family.direction), '
                               'the estimator of p94.compare_at; p94.lowest'),
                   slots=list(a.slots), pol_slots=list(a.pol_slots), ref_slots=list(a.ref_slots),
                   five_slots=list(a.five_slots), five_pol_slots=list(a.five_pol_slots),
                   seeds=list(a.a137.seeds), five_seeds=got['five_seeds'],
                   instances=list(a.pa124.inst),
                   training_campaign=got['campaign'], training_common_instances=got['common'],
                   records=got['records'], five_minute=got['five_minute'],
                   arrays=got['files'], rows=got['rows'], counts=got['counts'])
        names = None
        if full:
            lines = macro_lines(res)
            names = check_names(lines, paths['macros'])
            emit(paths['macros'], lines)
        for k in ('json', 'md'):
            os.makedirs(os.path.dirname(paths[k]), exist_ok=True)
        with open(paths['json'], 'w') as f:
            json.dump(T.jsonable(res), f, indent=1)
        with open(paths['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        remove_outputs(paths)
        raise
    return res, names, paths


def check(a, sel=None, cpsat_arm=None, chosen=True, dry=False):
    """--check: the records, every input and every comparison with the records, without writing;
    exit 1 while a problem exists."""
    problems = []
    if chosen:
        sel, cpsat_arm, problems = P124.selections(a.pa124)
    for x in problems:
        print(f'p138: missing: {x}')
    if problems:
        print('p138: incomplete')
        return 1
    try:
        got = analyse(a, sel, cpsat_arm, dry)
    except Refusal as e:
        for x in e.args[0]:
            print(f'p138: {x}')
        print('p138: incomplete')
        return 1
    c = got['counts']
    print(f'p138: {len(got["rows"])} rows; at 50 and 80 modules {c["prod"]["decided"]} of '
          f'{c["prod"]["rows"]} decided, at the training size {c["train"]["decided"]} of '
          f'{c["train"]["rows"]}')
    print('p138: every input present')
    return 0


# ----------------------------------------------------------------- dry mode
def dry_training_world(tree, boot):
    """The training-size world of the dry tree (p124.build_world at 10x25: one memetic run,
    CP-SAT, rule and policy records, arrays) and p94's own full run on it; returns the path of
    p94's record. Runs inside p124.dry_setting."""
    ns = P124.build_world(tree, [TRAIN_SIZE], P.parse_instances(DRY_TRAIN_INST), (1,),
                          P124.SLOTS, policy_seeds=(), cp_policy=False)
    out_root = os.path.join(tree, 'p94')
    keep = (T.select_policy_seed, T.select_cpsat_arm)
    T.select_policy_seed = lambda: (P124.DRY_SEL, {P124.DRY_SEL: (0.5, 1)})
    T.select_cpsat_arm = lambda *x: (P124.DRY_CPSAT, [])
    try:
        rc = T.main(['--campaign', DRY_TRAIN_CAMPAIGN, '--primary', str(T.PRIMARY),
                     '--boot', str(boot), '--result-root', ns.result_root,
                     '--data-root', ns.data_root, '--bars', ns.bars, '--out-root', out_root])
    finally:
        T.select_policy_seed, T.select_cpsat_arm = keep
    if rc != 0:
        raise Refusal([f'p94 on the training-size world of the dry tree returned {rc}'])
    return os.path.join(out_root, TIMED_JSON)


def dry_run(tree, boot=DRY_BOOT, check_only=False, before=None, five=None, extra=(),
            build=True):
    """The whole path on a synthetic tree (module docstring, DRY); returns (res, names, paths,
    options), or with check_only (rc of --check, None, None, options). before: a function of
    the tree called after p124, p137 and p94 have written their records and before this script
    reads them; five: keyword arguments of p137's dry run at 300 s (p137.dry_run, for example
    min_seeds and pol_drop), None for its full run; extra: further options of this script;
    build: False reads the records of an earlier dry run of the same tree again."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refusal([f'the dry-run folder {tree} lies inside the repository'])
    if build:
        P137.dry_run(tree, boot=boot)
        with P137.at_budget(FIVE):
            P137.dry_run(tree, boot=boot, build=False, **(five or {}))
    with P124.dry_setting(tree):
        timed = dry_training_world(tree, boot) if build else \
            os.path.join(tree, 'p94', TIMED_JSON)
        if before is not None:
            before(tree)
        ns = P137.dry_namespace(tree, P137.READINGS[ONE_MINUTE]['slots'])
        argv = ['--result-root', ns.result_root, '--data-root', ns.data_root, '--bars', ns.bars,
                '--boot', str(boot), '--timed-json', timed] + list(extra)
        if (five or {}).get('min_seeds') is not None:
            argv += ['--p137-five-json', os.path.join(tree, 'preview', P137_FIVE_JSON)]
        if check_only:
            argv.append('--check')
        a = build_parser().parse_args(argv)
        bad = check_options(a)
        if bad:
            raise Refusal(bad)
        if check_only:
            return (check(a, sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False, dry=True),
                    None, None, a)
        res, names, paths = run(a, sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False,
                                dry=True)
    return res, names, paths, a


def dry_verify(res):
    """The dry rows against the values built into the tree: every value of a row at 50 and 80
    modules is the mean of the built differences (p124.dry_built at 600 s, and at 300 s with
    --no-five-minute; at 60 and 300 s otherwise the built best of the runs of that budget of
    p137.dry_expected in both memetic columns, over the seeds res names for the rows at 300 s),
    every lowest column is p94.lowest of the values, and every paired difference is the
    difference of the means of its columns."""
    bad = []
    inst = P.parse_instances(P124.INSTANCES)
    members = dict(T.pools(P124.COMBOS))
    for r in res['rows']:
        what = f'{r["size"]} {r["budget"]} s {r["entry"]}'
        if r['part'] == 'prod':
            short = r['source'].startswith('p137')
            used = (res['five_seeds'] if r['budget'] == FIVE and res.get('five_seeds')
                    else {'rule': P137.SEEDS, 'policy': P137.SEEDS})
            for c in COLUMNS:
                vals = []
                for j in members[r['entry']]:
                    for i in inst:
                        if short and c in ('memetic_rule', 'memetic_policy'):
                            with P137.at_budget(r['budget']):
                                e = P137.dry_expected(r['size'], i, j, used['rule'],
                                                      used['policy'])
                            x = e['best60' if c == 'memetic_rule' else 'bestp60']
                            vals.append(100.0 * (x - e['greedy']) / e['greedy'])
                        else:
                            vals.append(P124.dry_built(c, i, j, float(r['budget'])))
                if r['values'][c] != f2(np.mean(vals)):
                    bad.append(f'{what} {c}: {r["values"][c]}, built {f2(np.mean(vals))}')
        if sorted(r['lowest']) != sorted(T.lowest(r['values'])):
            bad.append(f'{what}: lowest {r["lowest"]}')
        for key, first, second in (('lowest_vs_runner_up', 'lowest_column', 'runner_up_column'),
                                   ('with_vs_without', 'with_column', 'without_column')):
            x = r[key]
            if x is not None and abs(x['mean'] - (r['means'][r[first]] - r['means'][r[second]])) \
                    > 1e-9:
                bad.append(f'{what} {key}: {x["mean"]}')
    return bad


# ----------------------------------------------------------------- command line
def build_parser():
    ap = argparse.ArgumentParser(description='the paired difference behind the lowest entry of '
                                             'each row of the decision table')
    ap.add_argument('--check', action='store_true', help='read only: list every problem')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on a synthetic tree in DIR (default: a temporary folder)')
    ap.add_argument('--out', default=None,
                    help='root of manuscript/ and results/ (default PACT_OUT through '
                         'arm_registry.out); outside the repository\'s paper, result and record '
                         'folders and outside PACT_OUT')
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--slots', nargs='+', default=list(P137.READINGS[ONE_MINUTE]['slots']),
                    help="p137's --slots: core sets of the 60-s runs started from the rule "
                         'schedule (default 16-23)')
    ap.add_argument('--pol-slots', nargs='+',
                    default=list(P137.READINGS[ONE_MINUTE]['pol_slots']),
                    help="p137's --pol-slots: core sets of the 60-s runs started from the "
                         "policy's schedule (default 16-23 8-15)")
    ap.add_argument('--five-slots', nargs='+', default=list(P137.READINGS[FIVE]['slots']),
                    help="p137's --slots at --budget 300: core sets of the 300-s runs started "
                         'from the rule schedule (default 16-23)')
    ap.add_argument('--five-pol-slots', nargs='+', default=list(P137.READINGS[FIVE]['pol_slots']),
                    help="p137's --pol-slots at --budget 300: core sets of the 300-s runs started "
                         "from the policy's schedule (default 8-15)")
    ap.add_argument('--no-five-minute', action='store_true',
                    help="the rows at 300 s from p124's 600-s runs, without p137's record at "
                         '300 s; the outputs say so, no macro file')
    ap.add_argument('--min-seeds', type=int, default=None,
                    help="a preview: the rows at 300 s from the preview of p137's record at 300 s "
                         '(every complete seed of each 300-s set, at least n); needs --out, no '
                         'macro file')
    ap.add_argument('--ref-slots', nargs='+', default=list(P124.SLOTS),
                    help="p124's --slots and p137's --ref-slots: core sets of the 600-s runs "
                         '(default 16-23 8-15)')
    ap.add_argument('--bars', default=None)
    ap.add_argument('--max-invalid', type=int, default=2)
    ap.add_argument('--result-root', default='')
    ap.add_argument('--data-root', default='data/FAMILY')
    ap.add_argument('--p124-json', default=None, help=f"p124's record (default {P124_JSON} "
                    'through arm_registry.out)')
    ap.add_argument('--p137-json', default=None, help=f"p137's record (default {P137_JSON} "
                    'through arm_registry.out)')
    ap.add_argument('--p137-five-json', default=None, help="p137's record at 300 s (default "
                    f'{P137_FIVE_JSON} through arm_registry.out)')
    ap.add_argument('--timed-json', default=None, help=f"p94's record (default {TIMED_JSON} in "
                    'the repository)')
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(_ARGV[1:] if argv is None else argv)
    for k in ('out', 'p124_json', 'p137_json', 'p137_five_json', 'timed_json'):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(os.path.join(_CWD, getattr(a, k))))
    os.chdir(ROOT)
    if a.dry is not None:
        if a.check:
            ap.error('--dry builds its own records; --check reads the real ones')
        keep = bool(a.dry)
        tree = os.path.abspath(os.path.join(_CWD, a.dry)) if keep else \
            tempfile.mkdtemp(prefix='p138_dry_')
        try:
            boot = DRY_BOOT if a.boot == SF.B_DEFAULT else a.boot
            res, names, paths, _ = dry_run(tree, boot)
            bad = dry_verify(res)
            if bad:
                raise Refusal(bad)
        except Refusal as e:
            sys.exit('p138 self-test on synthetic records FAILED: ' + '; '.join(e.args[0][:6]))
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
        c = res['counts']
        print(f'p138 self-test on synthetic records passed: {len(res["rows"])} rows, B = '
              f'{res["boot"]}, {len(names)} macros; at 50 and 80 modules {c["prod"]["decided"]} '
              f'of {c["prod"]["rows"]} rows decided, at the training size {c["train"]["decided"]} '
              f'of {c["train"]["rows"]}; every value equal to the record and to the built one'
              + (f'; outputs in {os.path.dirname(os.path.dirname(paths["json"]))}' if keep else ''))
        return 0
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    if REG.env() != 'exact':
        sys.exit(f'REFUSING: PACT_ENV={REG.env()}: this analysis reads the exact environment')
    bad = check_options(a)
    if bad:
        sys.exit('REFUSING: ' + '; '.join(bad))
    if a.check:
        return check(a)
    if not a.out and os.environ.get('PACT_OUT') is None:
        sys.exit('REFUSING: set PACT_OUT (the output root of the chain, e.g. out/exact)')
    REG.banner()
    try:
        res, names, paths = run(a)
    except Refusal as e:
        print(f'*** REFUSING TO CERTIFY *** {len(e.args[0])} problem(s)', file=sys.stderr)
        for x in e.args[0][:40]:
            print(f'  {x}', file=sys.stderr)
        return 1
    c = res['counts']
    if res.get('warning'):
        print(f'p138: WARNING: {res["warning"]}; no macro file is written')
    print(f'p138: {res["run"]}: {len(res["rows"])} rows; at 50 and 80 modules '
          f'{c["prod"]["decided"]} of {c["prod"]["rows"]} decided, P ahead of O in '
          f'{c["prod"]["with_ahead"]}; at the training size {c["train"]["decided"]} of '
          f'{c["train"]["rows"]} decided; '
          + (f'{len(names)} macros' if names else 'no macro file') + '; wrote '
          + ', '.join(shown(p) for p in paths.values() if os.path.exists(p)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
