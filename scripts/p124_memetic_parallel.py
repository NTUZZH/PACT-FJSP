r"""P124: the memetic algorithm with eight independent runs, read as one parallel method.

QUESTION
  CP-SAT in the timed comparison solves with 8 workers on 8 cores; the memetic algorithm runs
  one thread. Eight independent runs of the memetic algorithm (seeds 1 to 8 of the timed
  campaign's arm memetic_rule@600, one run per core) are read as one method on the same eight
  cores: its incumbent at wall-clock time t is the shortest of the eight incumbents at t. Seed
  1 is the run of the existing comparison (scripts/p94_timed_inference.py); seeds 2 to 8 repeat
  it on the instances 003-011 of 50x25n30 and 80x25n30, all eight combinations.

WHAT IS READ (every reader is p94's or the planner's own; nothing here opens a folder by name)
  memetic runs  per seed k a planner campaign memetic_rule@600:<instances> with --memetic-seed k
                (scripts/timed_campaign_plan.py): record
                or_solution/FAMILY/<size>_memetic600s/<R>/instance_<i>_seed<k>.json; state by
                timed_campaign_plan.cell_states (timing_validity.classify_single_thread: one
                thread on one core of a slot, cpu_share >= 0.9) with --slots (default 16-23 and
                8-15, the two slots the seeds ran on); trajectory by p94.read_cells
                (timed_campaign_plan.record_rows, baselines.memetic.records.incumbent_at).
  CP-SAT        cpsat_rule@600 with the pilot's configuration (p94.select_cpsat_arm), 8 workers,
                read in the same way (timing_validity.classify with the frozen bars).
  request clock the rule records and the selected run's policy records of
                scripts/p97_policy_anytime.py (p94.request_records, p97.classify and
                p97.classify_rules): o_rule (t_all) and o_pol (time of the greedy schedule) of
                each cell, and the policy's sampled schedules (policy_sample@120, p94's token
                at the production batches).
  arrays        p94.check_descriptors and p94.load_arrays: the best of the four dispatching
                rules and the policy's greedy makespans of the exact environment; the training
                run is arm_registry.select_seed (p94.select_policy_seed).
  A record counts only when the planner calls it valid. The script prints one line for every
  record it rejects, with the planner's reason, and nothing for a record it uses.

CLOCK (--origin, p94's three clocks; default request, the clock of p94's chain step)
  request: a memetic run and CP-SAT have their incumbent at run time t - o_rule, the greedy
  schedule exists from o_pol on, the sampled schedules are the policy record's best so far
  (after its budget of 120 s the last value is held, which favours the other method; every
  entry that uses a held value says so). start and search: as p94; no sampled schedules.

QUANTITIES (per size, per combination, pooled over the eight and over the combinations without
and with setups, at every budget p94 reports on the clock up to 600 s: 60, 300, 600 on the
request clock, 1, 10, 60, 300, 600 otherwise)
  best of eight  per cell, min over the eight seeds of the incumbent at t (all eight records
                 valid); single run: seed 1.
  d = 100 (first - second) / second per cell (p94.gap), negative = the first-named method is
  shorter; p94.compare_at (p94.compare, p94.stat_instance: stats_family.instance_boot, the
  cluster bootstrap over base instances, a fresh default_rng(7) per quantity, B = --boot;
  stats_family.direction), on the cells where both methods have a record:
    pol_vs_best  the policy's greedy schedule minus the best of eight
    smp_vs_best  the policy, sampled schedules, minus the best of eight (request clock)
    cp_vs_best   CP-SAT (8 workers) started from the rule schedule minus the best of eight
    best_vs_one  the best of eight minus the run of seed 1 (what the seven further cores add)
  and, on the same cells, the comparisons with the run of seed 1 alone (pol_vs_one,
  smp_vs_one, cp_vs_one), so that each pair of numbers sits side by side.
  Verdicts at the stated budget (--primary, 600 s) for pol_vs_best and smp_vs_best: p94's rule
  of row Sel (p94.verdict_at: one-sided bootstrap p from the draws of the interval, Holm over
  the eight combinations per direction, p94.selected_verdict).
  Spread across the eight seeds at the stated budget: per cell the smallest, median and largest
  makespan; per entry the mean and the largest over the cells of 100 (largest - smallest) /
  smallest.

SECOND SET (eight runs started from the policy's schedule)
  The seeds of --seeds of the arm memetic_policy@600 on the same cells, started from the rule
  schedule and from the greedy schedule of the selected training run (record
  or_solution/FAMILY/<size>_memetic600s_pactwarm_s<run>/<R>/instance_<i>_seed<k>.json; the
  planner's state with the same --slots; on the request clock a run is read at run time
  t - max(o_pol, o_rule), p94.read_cells), are read as a second method on eight cores (best of
  eight as above), with CP-SAT started from the policy's schedule (cpsat_policy@600, the
  pilot's configuration, 8 workers, read at t - o_pol). The set is complete when every cell of
  every seed has a valid record with a reading at every budget. Only then the script also
  writes, at the same budgets, with p94.compare_at on the cells where both methods have a
  record:
    smp_vs_bestp   the policy, sampled schedules, minus the best of the eight runs started from
                   the policy's schedule (request clock)
    cpp_vs_bestp   CP-SAT started from the policy's schedule minus that best of eight
    bestp_vs_best  that best of eight minus the best of the eight runs started from the rule
                   schedule (as p94's comparison (iii))
    bestp_vs_pol   that best of eight minus the policy's greedy schedule (p94's (ii))
    best_vs_pol    the best of the eight runs started from the rule schedule minus the policy's
                   greedy schedule (p94's (vi))
  The last two are the form of the decision table: d in percent of the greedy schedule, on the
  request clock from its time o_pol on. These numbers come after everything the first set
  writes (macros, a JSON block policy_start, a Markdown section); no number or macro of the
  first set changes. While the set is not complete, the three output files are the ones of the
  first set alone, byte for byte, and the run and --check print the valid runs per seed. A seed
  whose every cell has a record of which one or more is rejected makes the run refuse and
  --check exit 1; so does a start schedule that differs from the rule or policy array, once the
  set is complete.

DECISION TABLE ON EIGHT CORES (request clock; written with the second set, after its numbers)
  The rows of p94's decision table (p94.decision_table: per size, pooled entry Pool, NoSetup and
  Setup, and budget, each column's difference from the greedy schedule of the selected run in
  percent of it as p89's f2 prints it; bold: the lowest printed value of the row over the six
  columns, p94.lowest) at 50x25n30 and 80x25n30, with the two memetic columns replaced by the best
  of eight: memetic_rule the best of the eight runs started from the rule schedule (best_vs_pol),
  memetic_policy the best of the eight started from the policy's schedule (bestp_vs_pol). The
  other four columns are computed by p94's comparisons on the same records (rules: p94.compare of
  the best rule against the greedy schedule, comparison (iv); policy_sample, cpsat_rule and
  cpsat_policy: p94.compare_at as comparisons (vii), (vi) and (ii)) and must equal the values of
  p94's record (--timed-json, default results/timed_inference.json, a full run on the same clock
  with the same selected run) to the printed digit. A difference, or a row whose columns are means
  over different cells (p94.decision_mismatches), makes the script refuse.
  Macros: value and bold marker per size, entry, budget and column, named by p94.decision_names
  with Mpar in the place of Tmd (MparFiftyMemRuleStartVsPolSixHundredNoSetupSel and its ...Bold;
  the rule column has one value per size and entry, MparFiftyRuleVsPolNoSetupSel). Counts over the
  rows the table prints (entries NoSetup and Setup at every budget), per size (Fifty, Eighty) and
  over both sizes (Both):
    Mpar<S>DecisionRowN                      rows
    Mpar<S>DecisionLowest<Col>N              rows whose lowest printed value is in column <Col>
                                             (Rule, PolSmp, MemRuleStart, CpRuleStart, MemPolStart,
                                             CpPolStart); a tie counts in each tied column
    Mpar<S>DecisionTieN                      rows with two or more lowest columns
    Mpar<S>DecisionFiveTenMinRowN            rows at 300 and 600 s
    Mpar<S>DecisionFiveTenMinLowestPolStartN those of them whose lowest value is a search started
                                             from the policy's schedule
    Mpar<S>DecisionLowestUsesPolicyN         rows whose lowest value uses the policy (its sampled
                                             schedules or a search started from its schedule)
  A JSON block decision_table and a Markdown section come after the second set's.
  Curves (results/memetic_parallel_curves.csv, p94.write_csv with the columns of p94's
  results/timed_curves.csv): row Sel, methods memetic_rule_best8 and memetic_policy_best8, n_runs
  8, on p94's curve grid (grid.curve of its record). Per cell the shortest incumbent of the eight
  runs at grid time t (p94.read_record with the cell's offset; the runs that have an incumbent at
  t), minus the greedy makespan of the selected run, p94.compare per combination, Pool and group;
  each method's rows start at the first grid time at which every cell has a schedule (the rows of
  a group: every cell of its combinations). p94 builds its curve rows inside analyse_request;
  curve_rows repeats that loop for one series. Rows of every training run (row All) are not
  written, since n_runs here counts memetic runs.

PREVIEW (--preview-first-set; needs --out-root outside the repository's paper and result folders
and outside PACT_OUT)
  The decision table and the curves with the first set alone: memetic_policy has no value and no
  bold marker, the bold rule runs over the five other columns, and the records of CP-SAT started
  from the policy's schedule are read for their column whether or not the second set is complete.
  The four columns of p94 are checked as above. Every output says preview and not for the paper
  (in the CSV the column run); no macro file is written.

VALIDITY AND CHECKS
  A cell enters the best-of-eight numbers only with all eight seeds valid. --check lists, per
  size, the cells with fewer valid seeds and every other missing input, and exits 1 while any
  is incomplete. --min-seeds k (default 8) computes on the cells with at least k valid seeds,
  prints a warning that the output is not for the paper, writes no macro file and needs
  --out-root outside the repository's paper and result folders and outside PACT_OUT. The rule
  start schedule of every memetic and CP-SAT record must equal the best of the four rule
  arrays, and the greedy makespan of every policy record the selected run's array (p94's
  checks); otherwise the script refuses and removes its outputs.

OUTPUTS (arm_registry.out under PACT_OUT, or under --out-root)
  manuscript/macros_memetic_parallel.tex  macro_io.emit; names Mpar<letters>; the prefix Mpar
                                         must be used by no other .tex file under manuscript/
                                         (checked at every run); the second set's macros at
                                         the end, after its own header comments
  results/memetic_parallel.json          every number, the records' states and core sets (the
                                         second set under policy_start)
  results/memetic_parallel.md            tables (the second set in its own section at the end)
  results/memetic_parallel_curves.csv    the curves of the decision table on eight cores (with the
                                         second set, or with --preview-first-set)

DRY (--dry [DIR]): the whole path on a synthetic tree (descriptors, arrays, eight seeds of
memetic records from each start, CP-SAT records from each start, rule and policy records, and a
record of p94 with the four columns of the decision table built from the same values) in DIR or a
temporary folder, with B = --boot (default 500).

Usage (repository root, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p124_memetic_parallel.py
  ... --check
  ... --min-seeds 1 --out-root <folder outside the repository>
  ... --preview-first-set --out-root <folder outside the repository>
  python scripts/p124_memetic_parallel.py --dry [DIR]
"""
import argparse
import contextlib
import csv
import json
import math
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
_ARGV = list(sys.argv)          # p94 imports p97, which empties sys.argv at import

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p94_timed_inference as T  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

P, P97 = T.P, T.P97
GENERATOR = 'scripts/p124_memetic_parallel.py'
PREFIX = 'Mpar'
OUT_FILES = {'macros': 'manuscript/macros_memetic_parallel.tex',
             'json': 'results/memetic_parallel.json', 'md': 'results/memetic_parallel.md',
             'curves': 'results/memetic_parallel_curves.csv'}
SIZES = tuple(P.SIZES)
INSTANCES = '3-11'
SEEDS = tuple(range(1, 9))
SINGLE = 1                      # the run of the existing comparison
ARM_BUDGET = 600                # memetic_rule@600 and cpsat_rule@600
POLICY_BUDGET = 120             # policy_sample@120, p94's token at the production batches
SLOTS = ('16-23', '8-15')
COMBOS = tuple(T.COMBOS)
EPS = T.EPS
DRY_BOOT = 500
Refusal = T.Refusal

# (key, macro word, text, request clock only, shares in the macros, verdict at the stated budget,
#  macros per combination as well as pooled)
COMPARISONS = (
    ('pol_vs_best', 'PolVsBest', "the policy's greedy schedule minus the best of the eight "
     'memetic runs', False, True, True, True),
    ('smp_vs_best', 'PolSmpVsBest', 'the policy, sampled schedules, minus the best of the eight '
     'memetic runs', True, True, True, True),
    ('cp_vs_best', 'CpVsBest', 'CP-SAT (8 workers) started from the rule schedule minus the best '
     'of the eight memetic runs', False, False, False, True),
    ('best_vs_one', 'BestVsOne', 'the best of the eight memetic runs minus the run of seed 1',
     False, True, False, True),
    ('pol_vs_one', 'PolVsOne', "the policy's greedy schedule minus the memetic run of seed 1",
     False, False, False, False),
    ('smp_vs_one', 'PolSmpVsOne', 'the policy, sampled schedules, minus the memetic run of seed 1',
     True, False, False, False),
    ('cp_vs_one', 'CpVsOne', 'CP-SAT (8 workers) started from the rule schedule minus the memetic '
     'run of seed 1', False, False, False, False))
REJECTED = ('invalid', 'pending', 'foreign', 'held')

# the second set (module docstring, SECOND SET)
POLICY_SET_ARM = 'memetic_policy'
CP_POLICY_ARM = 'cpsat_policy'
SET2_TEXT = "eight runs started from the policy's schedule"
# (key, macro word, text, request clock only, shares in the macros, (first, second) as keys of
#  run_texts, macros per combination at the stated budget as well as pooled)
COMPARISONS_POLICY = (
    ('smp_vs_bestp', 'PolSmpVsBestPolStart', 'the policy, sampled schedules, minus the best of the '
     "eight memetic runs started from the policy's schedule", True, True, ('smp', 'bestp'), True),
    ('cpp_vs_bestp', 'CpPolStartVsBestPolStart', "CP-SAT (8 workers) started from the policy's "
     "schedule minus the best of the eight memetic runs started from the policy's schedule",
     False, False, ('cpp', 'bestp'), True),
    ('bestp_vs_best', 'BestPolStartVsBest', "the best of the eight memetic runs started from the "
     "policy's schedule minus the best of the eight started from the rule schedule", False, True,
     ('bestp', 'best'), True),
    ('bestp_vs_pol', 'BestPolStartVsPol', "the best of the eight memetic runs started from the "
     "policy's schedule minus the policy's greedy schedule, in percent of the greedy schedule "
     '(the form of the decision table)', False, False, ('bestp', 'pol'), False),
    ('best_vs_pol', 'BestVsPol', 'the best of the eight memetic runs started from the rule '
     "schedule minus the policy's greedy schedule, in percent of the greedy schedule (the form "
     'of the decision table)', False, False, ('best', 'pol'), False))

# the decision table on eight cores and its curves (module docstring, DECISION TABLE ON EIGHT
# CORES); the columns are p94's DECISION_COLUMNS
TIMED_JSON = 'results/timed_inference.json'          # p94's record of the request clock
P94_COLUMNS = ('rules', 'policy_sample', 'cpsat_rule', 'cpsat_policy')   # equal to p94's values
BEST_COLUMNS = ('memetic_rule', 'memetic_policy')    # the best of eight from each start
POLICY_START_COLUMNS = ('memetic_policy', 'cpsat_policy')
USES_POLICY_COLUMNS = ('policy_sample',) + POLICY_START_COLUMNS
COLUMN_WORD = {'rules': 'Rule', 'policy_sample': T.SAMPLE_WORD, 'memetic_rule': 'MemRuleStart',
               'cpsat_rule': 'CpRuleStart', 'memetic_policy': 'MemPolStart',
               'cpsat_policy': 'CpPolStart'}
DECISION_TEXT = dict(T.DECISION_COLUMNS, memetic_rule=(
    'best of the eight memetic runs started from the rule schedule'), memetic_policy=(
    "best of the eight memetic runs started from the policy's schedule"))
TABLE_ENTRIES = T.group_keys()                       # the rows the table prints: NoSetup, Setup
LATE_BUDGETS = (300, 600)                            # 5 and 10 minutes since the request
BOTH_WORD = 'Both'                                   # the counts over both sizes
CURVE_METHOD = {'memetic_rule': 'memetic_rule_best8', 'memetic_policy': 'memetic_policy_best8'}
PREVIEW_WARNING = ('PREVIEW: the decision table and the curves with the first set alone (no run '
                   "started from the policy's schedule); not for the paper")
PREVIEW_RUN = 'preview of the first set alone, not for the paper'   # column run of the CSV
DECISION_RULE = (
    "the rows of p94's decision table (p94.decision_table) at the production batches, request "
    'clock, row Sel: per size, pooled entry and budget the difference of each column from the '
    'greedy schedule of the selected run in percent of it, as p89 f2 prints it; memetic_rule and '
    'memetic_policy the best of the eight memetic runs from each start (best_vs_pol and '
    'bestp_vs_pol of the second set); rules, policy_sample, cpsat_rule and cpsat_policy computed '
    "by p94's comparisons (iv), (vii), (vi) and (ii) on the same records and equal to p94's "
    'record to the printed digit; bold: the column or columns with the lowest printed value of '
    'the row (p94.lowest), over the columns with a number; every column of a row a mean over one '
    'set of cells (p94.decision_mismatches)')


# ----------------------------------------------------------------- small helpers

def setting_text():
    """The environment and the training seeds, in the words of the outputs."""
    import arm_registry as REG
    s = REG.seeds()
    return f'environment {REG.env()}, training seeds {",".join(map(str, s))}'

def shown(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def bnum(b):
    return f'{float(b):g}'


def entries(combos):
    return list(combos) + [k for k, _ in T.pools(combos)]


def default_budgets(origin):
    return [b for b in (T.REQUEST_BUDGETS if origin == 'request' else T.BUDGETS)
            if b <= ARM_BUDGET]


def p94_args(a, seed=None):
    """p94's own namespace for its readers (planner options passed through)."""
    argv = ['--result-root', a.result_root, '--max-invalid', str(a.max_invalid),
            '--data-root', a.data_root, '--slots', *a.slots, '--combos', *a.combos,
            '--policy-batch', str(a.policy_batch), '--policy-workers', str(a.policy_workers),
            '--policy-base-seed', str(a.policy_base_seed)]
    if a.bars:
        argv += ['--bars', a.bars]
    if seed is not None:
        argv += ['--memetic-seed', str(seed)]
    return T.build_parser().parse_args(argv)


def slot_of(core, slots):
    for s in slots:
        if core in P.parse_cores(s):
            return s
    return None


# ----------------------------------------------------------------- reading
def read_inputs(a, sel, cpsat_arm):
    """Every record and array, through p94's readers, before any statistic. Returns (data per
    size, rejected records, missing inputs, files read, the second set: read_policy_set). Raises
    Refusal on a record or array that contradicts another input (p94's checks; for the second
    set once it is complete)."""
    sizes, combos, inst = list(a.sizes), list(a.combos), list(a.inst)
    spec = P.compress(inst)
    grid = [float(b) for b in a.budgets]
    request = a.origin == 'request'
    rejected, missing, refusals = [], [], []
    root = a.result_root or ROOT
    base = p94_args(a)
    offsets, new = None, {}
    if request:
        if sel is None:
            missing.append('no selected training run: no policy record can be read')
        plan = [(s, r, i) for s in sizes for i in inst for r in combos]
        pol_tokens = {s: (float(a.policy_budget), None, f'{T.POLICY_ARM}@{bnum(a.policy_budget)}')
                      for s in sizes}
        new, _, _, gaps, refs = T.request_records(base, plan, pol_tokens, sel, grid, root)
        refusals += refs
        for g in gaps:
            m = re.match(r'^(\S+): (\w+)(?: \((.*)\))? \[(.*)\]$', g)
            if m and m.group(2) in REJECTED:
                rejected.append(dict(id=m.group(1), state=m.group(2), reason=m.group(3) or '',
                                     record=m.group(4)))
            elif m and m.group(2) == 'missing':
                missing.append(f'{m.group(1)}: no record [{m.group(4)}]')
            elif not g.startswith(f'{T.POLICY_ARM}: no selected'):
                missing.append(g)
        offsets = {k: {start: v[key][field] for start, key, field in
                       (('rule', 'rule', 't_all'), ('policy', 'policy', 't')) if key in v}
                   for k, v in new.items()}
    camps = [(k, T.make_campaign(sizes, [f'memetic_rule@{ARM_BUDGET}:{spec}'], p94_args(a, k),
                                 sel, cpsat_arm or 'ref_fullhint')) for k in a.seeds]
    if cpsat_arm is None:
        missing.append("no CP-SAT configuration (the pilot's selection is unavailable): no "
                       'CP-SAT record is read')
    else:
        camps.append(('cpsat', T.make_campaign(sizes, [f'cpsat_rule@{ARM_BUDGET}:{spec}'], base,
                                               sel, cpsat_arm)))
    cells = {}
    for key, camp in camps:
        got, _, _ = T.read_cells([camp], grid, a.origin, offsets)
        for (_, size, combo, idx), x in got.items():
            cid = f'{"seed " + str(key) if key != "cpsat" else "cpsat_rule"}/{size}/{combo}/{idx:03d}'
            if x['state'] in REJECTED:
                rejected.append(dict(id=cid, state=x['state'], reason=x['reason'],
                                     record=x['record']))
            elif x['state'] == 'missing' and key == 'cpsat':
                missing.append(f'{cid}: no record [{x["record"]}]')
            if x['state'] == 'valid':
                if request and x.get('offset_s') is None:
                    missing.append(f'{cid}: no valid rule record, so no offset on the request '
                                   'clock')
                else:
                    lack = [t for t in grid if x['reading'].get(t) is None]
                    if lack:
                        missing.append(f'{cid}: no reading at {len(lack)} of the budgets (first '
                                       f'{lack[0]:g} s) [{x["record"]}]')
                if key != 'cpsat':
                    rec = json.load(open(os.path.join(ROOT, x['record'])
                                         if not os.path.isabs(x['record']) else x['record']))
                    x['affinity'] = rec.get('affinity')
            cells[(key, size, combo, idx)] = x
    second = read_policy_set(a, sel, cpsat_arm, offsets)
    preview = None
    if a.preview_first_set:
        # the column of CP-SAT started from the policy's schedule, whatever the second set's state
        preview = dict(rejected=[], missing=[], cells={})
        if sel is None or cpsat_arm is None:
            missing.append("no selected training run or no CP-SAT configuration: no record of "
                           "CP-SAT started from the policy's schedule is read")
        else:
            read_cp_policy(a, sel, cpsat_arm, offsets, preview)
            rejected += preview['rejected']
            missing += preview['missing']
    data, files = {}, []
    for size in sizes:
        n_desc = T.check_descriptors(a.data_root, size, combos, inst)
        gaps = []
        rule, pol, fs = T.load_arrays(size, combos, inst, n_desc, list(REG.seeds()), False,
                                      refusals, gaps)
        files += fs
        missing += gaps
        pol_sel = pol.get(sel) if sel is not None else None
        if sel is not None and pol_sel is None:
            missing.append(f'{size}: the policy arrays of the selected run s{sel} are incomplete')
        data[size] = assemble(a, size, cells, new, rule, pol_sel, refusals)
        # the valid memetic records of the first set, for the curves
        data[size]['cells'] = {(key, combo, idx): x for (key, s, combo, idx), x in cells.items()
                               if s == size and key != 'cpsat' and x['state'] == 'valid'}
        if second['state'] == 'complete':
            data[size]['policy'] = assemble_policy(a, size, second['cells'], rule, pol_sel,
                                                   refusals)
            data[size]['policy']['cells'] = {
                (key, combo, idx): x for (key, s, combo, idx), x in second['cells'].items()
                if s == size and key != 'cpsat' and x['state'] == 'valid'}
        if preview is not None:
            data[size]['preview_cp'] = assemble_policy(a, size, preview['cells'], rule, pol_sel,
                                                       refusals)
    if refusals:
        raise Refusal(refusals)
    return data, rejected, missing, sorted(set(files)), second


def read_policy_set(a, sel, cpsat_arm, offsets):
    """The second set (module docstring): the state of every memetic_policy@600 record of every
    seed and, once the set is complete, the records of CP-SAT started from the policy's schedule,
    through p94's readers. Returns dict(state: complete, incomplete, unavailable or refused;
    counts {size: {seed: valid, rejected, missing, cores}}; rejected and missing as for the
    first set; problems; cells {(seed or 'cpsat', size, combination, instance): reading}). Reads
    nothing else and changes nothing the first set reads."""
    sizes, combos, inst = list(a.sizes), list(a.combos), list(a.inst)
    spec = P.compress(inst)
    grid = [float(b) for b in a.budgets]
    request = a.origin == 'request'
    out = dict(state='incomplete', run=sel, seeds=list(a.seeds), n_cells=len(sizes) * len(combos)
               * len(inst), counts={s: {str(k): dict(valid=0, rejected=0, missing=0, cores={})
                                        for k in a.seeds} for s in sizes},
               rejected=[], missing=[], problems=[], cells={})
    if sel is None:
        out.update(state='unavailable', problems=['no selected training run, so the folders of '
                                                  "the runs started from the policy's schedule "
                                                  'are not known'])
        return out

    def usable(x):
        return usable_reading(x, grid, request)
    try:
        for k in a.seeds:
            camp = T.make_campaign(sizes, [f'{POLICY_SET_ARM}@{ARM_BUDGET}:{spec}'],
                                   p94_args(a, k), sel, cpsat_arm or 'ref_fullhint')
            got, _, _ = T.read_cells([camp], grid, a.origin, offsets)
            for (_, size, combo, idx), x in got.items():
                cid = f'{POLICY_SET_ARM} seed {k}/{size}/{combo}/{idx:03d}'
                cnt = out['counts'][size][str(k)]
                state, why = x['state'], x['reason']
                if state == 'valid':
                    why = usable(x)
                    state = 'valid' if why is None else 'unusable'
                if state == 'valid':
                    rec = json.load(open(os.path.join(ROOT, x['record'])
                                         if not os.path.isabs(x['record']) else x['record']))
                    x['affinity'] = aff = rec.get('affinity')
                    slot = slot_of(aff[0], a.slots) if aff and len(aff) == 1 else None
                    cnt['valid'] += 1
                    cnt['cores'][slot or str(aff)] = cnt['cores'].get(slot or str(aff), 0) + 1
                    out['cells'][(k, size, combo, idx)] = x
                elif state == 'missing':
                    cnt['missing'] += 1
                else:
                    cnt['rejected'] += 1
                    out['rejected'].append(dict(id=cid, state=state, reason=why or '',
                                                record=x['record']))
        per_seed = {k: {f: sum(out['counts'][s][str(k)][f] for s in sizes)
                        for f in ('valid', 'rejected', 'missing')} for k in a.seeds}
        partly = [k for k, v in per_seed.items() if v['missing'] == 0 and v['rejected']]
        if partly:
            out['state'] = 'refused'
            out['problems'] += [f'seed {k}: every one of its {out["n_cells"]} cells has a record, '
                                f'{per_seed[k]["rejected"]} of them not usable (first '
                                + next(x['id'] for x in out['rejected']
                                       if x['id'].startswith(f'{POLICY_SET_ARM} seed {k}/'))
                                + ')' for k in partly]
            return out
        if not all(v['valid'] == out['n_cells'] for v in per_seed.values()):
            return out
        out['state'] = 'complete'
        if cpsat_arm is None:
            out['missing'].append("no CP-SAT configuration (the pilot's selection is "
                                  "unavailable): no record of CP-SAT started from the policy's "
                                  'schedule is read')
            return out
        read_cp_policy(a, sel, cpsat_arm, offsets, out)
    except Refusal as e:
        out.update(state='refused', problems=out['problems'] + list(e.args[0]))
    return out


def usable_reading(x, grid, request):
    """None for a valid record with a reading at every budget, else the reason."""
    if request and x.get('offset_s') is None:
        return 'no valid rule or policy record, so no offset on the request clock'
    lack = [t for t in grid if x['reading'].get(t) is None]
    return (f'no reading at {len(lack)} of the budgets (first {lack[0]:g} s)' if lack
            else None)


def read_cp_policy(a, sel, cpsat_arm, offsets, out):
    """The records of CP-SAT started from the policy's schedule (cpsat_policy@600, the pilot's
    configuration, 8 workers) through p94's readers, into out: rejected and missing as for the
    second set, cells {('cpsat', size, combination, instance): reading}. Raises Refusal as
    p94.read_cells does."""
    sizes, inst = list(a.sizes), list(a.inst)
    grid = [float(b) for b in a.budgets]
    camp = T.make_campaign(sizes, [f'{CP_POLICY_ARM}@{ARM_BUDGET}:{P.compress(inst)}'],
                           p94_args(a), sel, cpsat_arm)
    got, _, _ = T.read_cells([camp], grid, a.origin, offsets)
    for (_, size, combo, idx), x in got.items():
        cid = f'{CP_POLICY_ARM}/{size}/{combo}/{idx:03d}'
        if x['state'] in REJECTED:
            out['rejected'].append(dict(id=cid, state=x['state'], reason=x['reason'],
                                        record=x['record']))
        elif x['state'] == 'missing':
            out['missing'].append(f'{cid}: no record [{x["record"]}]')
        elif x['state'] == 'valid':
            why = usable_reading(x, grid, a.origin == 'request')
            if why is not None:
                out['missing'].append(f'{cid}: {why} [{x["record"]}]')
        out['cells'][('cpsat', size, combo, idx)] = x
    return out


def assemble(a, size, cells, new, rule, pol_sel, refusals):
    """The arrays of one size: memetic readings [K, T, n, C] and their validity, CP-SAT, the
    sampled schedules and the offsets; p94's start-schedule checks."""
    combos, inst, seeds = list(a.combos), list(a.inst), list(a.seeds)
    grid = [float(b) for b in a.budgets]
    request = a.origin == 'request'
    n, C, Tn = len(inst), len(combos), len(grid)
    ii, ci = {i: k for k, i in enumerate(inst)}, {r: j for j, r in enumerate(combos)}
    M = np.full((len(seeds), Tn, n, C), np.nan)
    V = np.zeros((len(seeds), n, C), dtype=bool)
    CP, CPv = np.full((Tn, n, C), np.nan), np.zeros((n, C), dtype=bool)
    cores = {k: {} for k in seeds}
    for (key, s, combo, idx), x in cells.items():
        if s != size or x['state'] != 'valid' or (request and x.get('offset_s') is None):
            continue
        i, j = ii[idx], ci[combo]
        vals = [np.nan if x['reading'].get(t) is None else x['reading'][t] for t in grid]
        want = rule[i, j]
        if np.isfinite(want) and (x['rule_start'] is None or abs(x['rule_start'] - want) > 1e-6):
            refusals.append(f'{x["record"]}: rule start schedule {x["rule_start"]}, best of the '
                            f'four rule arrays {want:g} (instance {idx}, {combo})')
        if key == 'cpsat':
            CP[:, i, j], CPv[i, j] = vals, np.isfinite(vals).all()
            continue
        k = seeds.index(key)
        M[k, :, i, j], V[k, i, j] = vals, np.isfinite(vals).all()
        aff = x.get('affinity') or []
        slot = slot_of(aff[0], a.slots) if len(aff) == 1 else None
        cores[key][slot or str(aff)] = cores[key].get(slot or str(aff), 0) + 1
    o_rule, o_pol = np.full((n, C), np.nan), np.full((n, C), np.nan)
    smp, sp, sbud = np.full((Tn, n, C), np.nan), np.zeros((n, C), dtype=bool), set()
    for (s, r, idx), v in new.items():
        if s != size or idx not in ii or r not in ci:
            continue
        i, j = ii[idx], ci[r]
        if 'rule' in v:
            o_rule[i, j] = v['rule']['t_all']
            want = rule[i, j]
            if np.isfinite(want) and abs(v['rule']['best_makespan'] - want) > 1e-6:
                refusals.append(f'{v["rule"]["record"]}: best rule makespan '
                                f'{v["rule"]["best_makespan"]:g}, best of the four rule arrays '
                                f'{want:g} (instance {idx}, {r})')
        if 'policy' in v:
            p = v['policy']
            o_pol[i, j] = p['t']
            smp[:, i, j] = [np.nan if p['reading'][t] is None else p['reading'][t] for t in grid]
            sp[i, j] = True
            sbud.add(p['budget_s'])
            if pol_sel is not None and np.isfinite(pol_sel[i, j]) \
                    and abs(p['greedy_makespan'] - pol_sel[i, j]) > 1e-6:
                refusals.append(f'{p["record"]}: greedy makespan {p["greedy_makespan"]:g}, policy '
                                f'array of the selected run {pol_sel[i, j]:g} (instance {idx}, {r})')
    if len(sbud) > 1:
        refusals.append(f'{size}: policy records with budgets {sorted(sbud)}; one budget per size')
    return dict(M=M, V=V, CP=CP, CPv=CPv, rule=rule, pol_sel=pol_sel, o_rule=o_rule, o_pol=o_pol,
                smp=smp, sp=sp, policy_budget_s=min(sbud) if sbud else None, cores=cores)


def assemble_policy(a, size, cells, rule, pol_sel, refusals):
    """The arrays of the second set at one size (read_policy_set's cells): memetic readings
    [K, T, n, C] and their validity, CP-SAT started from the policy's schedule; p94's
    start-schedule checks (the rule start of a memetic record, the policy start of every
    record)."""
    combos, inst, seeds = list(a.combos), list(a.inst), list(a.seeds)
    grid = [float(b) for b in a.budgets]
    request = a.origin == 'request'
    n, C, Tn = len(inst), len(combos), len(grid)
    ii, ci = {i: k for k, i in enumerate(inst)}, {r: j for j, r in enumerate(combos)}
    M = np.full((len(seeds), Tn, n, C), np.nan)
    V = np.zeros((len(seeds), n, C), dtype=bool)
    CP, CPv = np.full((Tn, n, C), np.nan), np.zeros((n, C), dtype=bool)
    for (key, s, combo, idx), x in cells.items():
        if s != size or x['state'] != 'valid' or (request and x.get('offset_s') is None):
            continue
        i, j = ii[idx], ci[combo]
        vals = [np.nan if x['reading'].get(t) is None else x['reading'][t] for t in grid]
        want = rule[i, j]
        if key != 'cpsat' and np.isfinite(want) and (x['rule_start'] is None
                                                    or abs(x['rule_start'] - want) > 1e-6):
            refusals.append(f'{x["record"]}: rule start schedule {x["rule_start"]}, best of the '
                            f'four rule arrays {want:g} (instance {idx}, {combo})')
        if pol_sel is not None and np.isfinite(pol_sel[i, j]) and (
                x['policy_start'] is None or abs(x['policy_start'] - pol_sel[i, j]) > 1e-6):
            refusals.append(f'{x["record"]}: policy start schedule {x["policy_start"]}, policy '
                            f'array of the selected run {pol_sel[i, j]:g} (instance {idx}, '
                            f'{combo})')
        if key == 'cpsat':
            CP[:, i, j], CPv[i, j] = vals, np.isfinite(vals).all()
            continue
        q = seeds.index(key)
        M[q, :, i, j], V[q, i, j] = vals, np.isfinite(vals).all()
    return dict(M=M, V=V, CP=CP, CPv=CPv)


def incomplete_cells(a, D):
    """[(combination, instance, valid seeds, seeds without a valid record)] of one size."""
    out = []
    for j, r in enumerate(a.combos):
        for i, idx in enumerate(a.inst):
            ok = [k for q, k in enumerate(a.seeds) if D['V'][q, i, j]]
            if len(ok) < len(a.seeds):
                out.append((r, idx, ok, [k for k in a.seeds if k not in ok]))
    return out


# ----------------------------------------------------------------- analysis
def best_of(M, V, min_seeds):
    """(cells used [n, C]: at least min_seeds valid runs; the valid readings [K, T, n, C], NaN
    elsewhere; the best of the valid runs [T, n, C] on the cells used, NaN elsewhere)."""
    use = V.sum(axis=0) >= min_seeds
    Mv = np.where(V[:, None], M, np.nan)
    with np.errstate(all='ignore'):
        best = np.where(use[None], np.nanmin(np.where(use[None, None], Mv, np.inf), axis=0),
                        np.nan)
    best = np.where(np.isfinite(best), best, np.nan)
    return use, Mv, best


def analyse(a, data):
    """Every quantity of every size (module docstring)."""
    combos, seeds = list(a.combos), list(a.seeds)
    grid = [float(b) for b in a.budgets]
    gi = {t: k for k, t in enumerate(grid)}
    request = a.origin == 'request'
    eight = sorted(combos) == sorted(COMBOS)
    one = seeds.index(SINGLE)
    out = {}
    for size, D in data.items():
        M, V, pol_sel = D['M'], D['V'], D['pol_sel']
        use, Mv, best = best_of(M, V, a.min_seeds)
        M1, V1 = M[one], V[one]
        if pol_sel is None:
            pol_sel = np.full(use.shape, np.nan)
        pol_pres = np.isfinite(pol_sel) & (np.isfinite(D['o_pol']) if request else True)
        sbud = D['policy_budget_s']
        S = dict(instances=list(a.inst), combinations=combos, seeds=seeds,
                 cells=int(use.size), cells_used=int(use.sum()),
                 valid_seeds_per_cell={str(k): int((V.sum(axis=0) == k).sum())
                                       for k in range(len(seeds) + 1)},
                 core_sets={str(k): D['cores'][k] for k in seeds}, policy_budget_s=sbud,
                 comparisons={}, verdicts={}, means={})
        for b in a.budgets:
            t = float(b)
            k = gi[t]
            g = (np.where(D['o_pol'] <= t + EPS, pol_sel, np.nan) if request else pol_sel)
            pairs = dict(pol_vs_best=(g, best[k], use & pol_pres),
                         cp_vs_best=(D['CP'][k], best[k], use & D['CPv']),
                         best_vs_one=(best[k], M1[k], use & V1),
                         pol_vs_one=(g, M1[k], use & pol_pres & V1),
                         cp_vs_one=(D['CP'][k], M1[k], use & D['CPv'] & V1))
            if request:
                pairs.update(smp_vs_best=(D['smp'][k], best[k], use & D['sp']),
                             smp_vs_one=(D['smp'][k], M1[k], use & D['sp'] & V1))
            held = request and sbud is not None and t > sbud + EPS
            for key, *_ in COMPARISONS:
                if key not in pairs:
                    continue
                first, second, pres = pairs[key]
                rows = T.compare_at(first, second, pres, combos, a.boot, a.alpha)
                if key.startswith('smp'):
                    rows = T.mark_held(rows, held, sbud)
                S['comparisons'].setdefault(key, {})[str(b)] = rows
                if b == a.primary and eight and key in ('pol_vs_best', 'smp_vs_best'):
                    fam = T.verdict_at(first, second, pres, combos, a.boot, a.alpha)
                    if fam is not None and key == 'smp_vs_best':
                        fam = dict(fam, held=held, held_after_s=sbud)
                    S['verdicts'][key] = fam
            S['means'][str(b)] = means_at(combos, use, best[k], M1[k], g, D['CP'][k], D['CPv'],
                                          D['smp'][k] if request else None, D['sp'])
        S['spread'] = spread(combos, list(a.inst), Mv[:, gi[float(a.primary)]], V, use)
        out[size] = S
    return out


def means_at(combos, use, best, one, greedy, cp, cpv, smp, sp):
    """Mean makespans per combination over the cells used (descriptive)."""
    out = {}
    for j, r in enumerate(combos):
        m = use[:, j]
        if not m.any():
            continue
        x = dict(n=int(m.sum()), best_of_eight=float(best[m, j].mean()),
                 seed_one=float(np.nanmean(one[m, j])), policy_greedy=float(np.nanmean(greedy[m, j])))
        if cpv[m, j].all():
            x['cpsat'] = float(cp[m, j].mean())
        if smp is not None and sp[m, j].all():
            x['policy_sampled'] = float(smp[m, j].mean())
        out[r] = x
    return out


def spread(combos, inst, Mp, V, use):
    """Smallest, median and largest makespan over the valid seeds per cell at the stated budget;
    per entry the mean and the largest of 100 (largest - smallest) / smallest."""
    cells = []
    rng_pct = np.full(use.shape, np.nan)
    for j, r in enumerate(combos):
        for i, idx in enumerate(inst):
            if not use[i, j]:
                continue
            v = Mp[V[:, i, j], i, j]
            lo, med, hi = float(v.min()), float(np.median(v)), float(v.max())
            rng_pct[i, j] = 100.0 * (hi - lo) / lo
            cells.append(dict(combination=r, instance=idx, seeds=int(v.size), smallest=lo,
                              median=med, largest=hi, range_pct=float(rng_pct[i, j])))
    ent = {}
    for key, cols in [(r, [j]) for j, r in enumerate(combos)] + T.pools(combos):
        x = rng_pct[:, cols]
        x = x[np.isfinite(x)]
        if x.size and (len(cols) == 1 or len({c['combination'] for c in cells
                                              if c['combination'] in [combos[q] for q in cols]}) >= 2):
            ent[key] = dict(mean_range_pct=float(x.mean()), max_range_pct=float(x.max()),
                            n_cells=int(x.size), combinations=[combos[q] for q in cols])
    return dict(entries=ent, cells=cells)


def analyse_policy(a, data):
    """The quantities of the second set at every size (module docstring, SECOND SET): its own
    cells (all its runs valid), the first set's cells as analyse uses them, p94.compare_at at
    every budget. Changes nothing analyse computes."""
    combos, seeds = list(a.combos), list(a.seeds)
    grid = [float(b) for b in a.budgets]
    gi = {t: k for k, t in enumerate(grid)}
    request = a.origin == 'request'
    out = {}
    for size, D in data.items():
        Q = D.get('policy')
        if Q is None:
            continue
        use, _, best = best_of(D['M'], D['V'], a.min_seeds)
        use2, _, bestp = best_of(Q['M'], Q['V'], len(seeds))
        pol_sel = D['pol_sel'] if D['pol_sel'] is not None else np.full(use.shape, np.nan)
        pol_pres = np.isfinite(pol_sel) & (np.isfinite(D['o_pol']) if request else True)
        sbud = D['policy_budget_s']
        S = dict(instances=list(a.inst), combinations=combos, seeds=seeds, cells=int(use2.size),
                 cells_used=int(use2.sum()),
                 valid_seeds_per_cell={str(k): int((Q['V'].sum(axis=0) == k).sum())
                                       for k in range(len(seeds) + 1)},
                 comparisons={}, means={})
        for b in a.budgets:
            t = float(b)
            k = gi[t]
            g = (np.where(D['o_pol'] <= t + EPS, pol_sel, np.nan) if request else pol_sel)
            pairs = dict(cpp_vs_bestp=(Q['CP'][k], bestp[k], use2 & Q['CPv']),
                         bestp_vs_best=(bestp[k], best[k], use2 & use),
                         bestp_vs_pol=(bestp[k], g, use2 & pol_pres),
                         best_vs_pol=(best[k], g, use & pol_pres))
            if request:
                pairs['smp_vs_bestp'] = (D['smp'][k], bestp[k], use2 & D['sp'])
            held = request and sbud is not None and t > sbud + EPS
            for key, *_ in COMPARISONS_POLICY:
                if key not in pairs:
                    continue
                first, second, pres = pairs[key]
                rows = T.compare_at(first, second, pres, combos, a.boot, a.alpha)
                if key.startswith('smp'):
                    rows = T.mark_held(rows, held, sbud)
                S['comparisons'].setdefault(key, {})[str(b)] = rows
            m = {}
            for j, r in enumerate(combos):
                c = use2[:, j]
                if c.any():
                    x = dict(n=int(c.sum()),
                             best_of_eight_policy_start=float(bestp[k][c, j].mean()))
                    if Q['CPv'][c, j].all():
                        x['cpsat_policy_start'] = float(Q['CP'][k][c, j].mean())
                    m[r] = x
            S['means'][str(b)] = m
        out[size] = S
    return out


# ----------------------------------------------------------------- decision table, eight cores
def decision_size(a, D, best, use, bestp=None, use2=None, cp_policy=None):
    """The decision table of one size (module docstring, DECISION TABLE ON EIGHT CORES): the
    comparisons of its six columns as p94's analyse_request computes them, on this script's
    arrays, and p94.decision_table over them. best, use: the best of the eight runs started from
    the rule schedule [T, n, C] and its cells; bestp, use2: the same from the policy's schedule
    (None: that column has no value); cp_policy: (readings [T, n, C], cells [n, C]) of CP-SAT
    started from the policy's schedule, or None. Returns dict(instances, budgets, iv, vii, vi,
    ii, decision); decision is p94.decision_table's {budget: {entry: row}}."""
    combos = list(a.combos)
    gi = {float(b): k for k, b in enumerate(a.budgets)}
    B, alpha = a.boot, a.alpha
    pol_sel = D['pol_sel'] if D['pol_sel'] is not None else np.full(use.shape, np.nan)
    pol_pres = np.isfinite(pol_sel) & np.isfinite(D['o_pol'])
    sbud = D['policy_budget_s']
    used = {}
    S = dict(instances=list(a.inst), budgets=list(a.budgets), iv={}, vii={}, vi={}, ii={})
    if np.isfinite(D['rule']).any() and D['pol_sel'] is not None:
        S['iv']['selected'] = T.compare(D['rule'], pol_sel, combos, B, alpha,
                                        used=used.setdefault(('rules', None), {}))
    for b in a.budgets:
        t = float(b)
        k = gi[t]
        with np.errstate(invalid='ignore'):
            g = np.where(D['o_pol'] <= t + EPS, pol_sel, np.nan)
        held = sbud is not None and t > sbud + EPS

        def at(first, pres, col):
            return T.compare_at(first, g, pres & pol_pres, combos, B, alpha,
                                used=used.setdefault((col, str(b)), {}))
        if D['sp'].any():
            S['vii'][str(b)] = T.mark_held(at(D['smp'][k], D['sp'], 'policy_sample'), held, sbud)
        S['vi'].setdefault('memetic', {})[str(b)] = dict(selected=at(best[k], use,
                                                                     'memetic_rule'))
        S['vi'].setdefault('cpsat', {})[str(b)] = dict(selected=at(D['CP'][k], D['CPv'],
                                                                   'cpsat_rule'))
        if bestp is not None:
            S['ii'].setdefault('memetic', {})[str(b)] = at(bestp[k], use2, 'memetic_policy')
        if cp_policy is not None:
            S['ii'].setdefault('cpsat', {})[str(b)] = at(cp_policy[0][k], cp_policy[1],
                                                         'cpsat_policy')
    S['decision'] = T.decision_table(S, D['o_rule'], combos, used)
    return S


def timed_path(a):
    return a.timed_json or os.path.join(ROOT, TIMED_JSON)


def p94_check(a, sel, tables, J, path):
    """Texts of every difference between the record of p94 (J, read from path) and this
    script's decision tables ({size: {budget: {entry: row}}}) in the four columns P94_COLUMNS,
    compared as printed texts; empty when every value is equal. The record must be a full run of
    p94 on the same clock with the same selected run."""
    bad = []
    where = shown(path)
    if J.get('run') != 'full' or J.get('stand_in'):
        bad.append(f'{where}: run {J.get("run")!r}, not a full run of p94 with the policy')
    if J.get('origin') != a.origin:
        bad.append(f'{where}: clock {J.get("origin")!r}, this run reads the {a.origin} clock')
    if (J.get('selection') or {}).get('seed') != sel:
        bad.append(f'{where}: selected run {(J.get("selection") or {}).get("seed")}, this run '
                   f'reads s{sel}')
    for size, rows in tables.items():
        dec = ((J.get('results') or {}).get(size) or {}).get('decision')
        if dec is None:
            bad.append(f'{where}: no decision table at {size}')
            continue
        for b, brow in rows.items():
            for key, row in brow.items():
                want = (dec.get(b) or {}).get(key)
                if want is None:
                    bad.append(f'{where}: no row {size}, {b} s, {T.entry_text(key)}')
                    continue
                for c in P94_COLUMNS:
                    if row['values'][c] != want['values'].get(c):
                        bad.append(f'{size}, {b} s, {T.entry_text(key)}, {c}: '
                                   f'{row["values"][c] or "no number"} here, '
                                   f'{want["values"].get(c) or "no number"} in {where}')
    return bad


def decision_counts(tables, budgets):
    """{size, and 'both' at two or more sizes: counts} over the rows the table prints (the
    entries TABLE_ENTRIES at every budget): rows, rows per lowest column (a tie counts in each
    tied column), ties, rows without a number, rows at LATE_BUDGETS and those of them whose
    lowest value is a search started from the policy's schedule, rows whose lowest value uses the
    policy (USES_POLICY_COLUMNS)."""
    per = {size: [(int(b), set(row['bold'])) for b, brow in rows.items()
                  for key, row in brow.items() if key in TABLE_ENTRIES]
           for size, rows in tables.items()}
    if len(per) > 1:
        per['both'] = [x for size in tables for x in per[size]]
    late = [b for b in LATE_BUDGETS if b in budgets]
    out = {}
    for name, rows in per.items():
        out[name] = dict(
            rows=len(rows),
            lowest={c: sum(c in bold for _, bold in rows) for c, _ in T.DECISION_COLUMNS},
            ties=sum(len(bold) > 1 for _, bold in rows),
            without_number=sum(not bold for _, bold in rows),
            late_budgets=late, late_rows=sum(b in late for b, _ in rows),
            late_lowest_policy_start=sum(b in late and bool(bold & set(POLICY_START_COLUMNS))
                                         for b, bold in rows),
            lowest_uses_policy=sum(bool(bold & set(USES_POLICY_COLUMNS)) for _, bold in rows))
    return out


def curve_series(a, cells, V, use, grid):
    """[len(grid), n, C]: per cell of use the shortest incumbent of its valid runs (V [K, n, C],
    the runs best_of takes) at each grid time, NaN where no run has an incumbent yet. A run is
    read by p94.read_record, on the request clock at run time t - the cell's offset, as
    p94.read_cells reads it at the budgets."""
    combos, inst, seeds = list(a.combos), list(a.inst), list(a.seeds)
    request = a.origin == 'request'
    out = np.full((len(grid), len(inst), len(combos)), np.nan)
    for j, r in enumerate(combos):
        for i, idx in enumerate(inst):
            if not use[i, j]:
                continue
            runs = []
            for q, k in enumerate(seeds):
                if not V[q, i, j]:
                    continue
                x = cells[(k, r, idx)]
                path = x['record'] if os.path.isabs(x['record']) else os.path.join(ROOT,
                                                                                   x['record'])
                rec = json.load(open(path))
                got = (T.read_record(rec, 'memetic', grid, a.origin, offset=x['offset_s'])
                       if request else T.read_record(rec, 'memetic', grid, a.origin))
                runs.append([np.nan if got.get(t) is None else got[t] for t in grid])
            v = np.array(runs, dtype=float)
            ok = np.isfinite(v)
            out[:, i, j] = np.where(ok.any(axis=0), np.where(ok, v, np.inf).min(axis=0), np.nan)
    return out


def curve_rows(size, method, solver, start, vals, pres, pol, combos, grid, B, alpha, n_runs):
    """The curve rows of one series, row Sel, as p94's analyse_request writes them for an arm
    (its loop over the series, which is not a function of its own): vals [len(grid), n, C]
    makespans on the grid, pres [n, C] the cells of the series, pol [n, C] the greedy makespans
    of the selected run. From the first grid time at which every cell of pres has a schedule
    (the rows of a group: every cell of its combinations), p94.compare of the values against pol
    at each grid time. Returns (rows per combination and Pool, rows of the groups, first time,
    {group: first time})."""
    gi = {t: k for k, t in enumerate(grid)}
    groups = [(key, members) for key, members in T.pools(combos) if key != T.POOL]
    csv_group = {g[0]: g[3] for g in T.GROUPS}

    def first_full(pres):
        full = [t for t in grid if pres.any() and np.isfinite(vals[gi[t]][pres]).all()]
        return full[0] if full else None
    first = first_full(pres)
    g_start = {}
    for key, members in groups:
        gp = np.zeros_like(pres)
        gp[:, members] = pres[:, members]
        g_start[key] = first_full(gp)
    begin = [t for t in (first, *g_start.values()) if t is not None]
    curves, group_curves = [], []
    for t in (grid if begin else ()):
        if t < min(begin):
            continue
        v = np.where(pres, vals[gi[t]], np.nan)
        for r, row in T.compare(v, pol, combos, B, alpha).items():
            t0 = g_start.get(r) if r in csv_group else first
            if t0 is None or t < t0:
                continue
            c = dict(row='Sel', size=size, combination=r, method=method, solver=solver,
                     start=start, t_s=t, mean_pct=row['mean'], lo_pct=row['lo'], hi_pct=row['hi'],
                     n_instances=row['n'], n_combinations=row['c'], n_runs=n_runs, held='no')
            if r in csv_group:
                group_curves.append(dict(c, group=csv_group[r]))
            else:
                curves.append(c)
    return curves, group_curves, first, g_start


def first_times(D, combos):
    """p94's (ix) of the rule and policy records that its CSV reads (the median first-schedule
    times of a size and of each group): p94.first_stats of o_rule and o_pol."""
    o_rule, o_pol = D['o_rule'], D['o_pol']
    ix = dict(rules=dict(t_all=T.first_stats(o_rule, np.isfinite(o_rule))),
              policy=dict(t=T.first_stats(o_pol, np.isfinite(o_pol))))
    for key, members in T.pools(combos):
        if key == T.POOL:
            continue
        g = np.zeros(o_rule.shape, dtype=bool)
        g[:, members] = True
        ix[key] = dict(rules=dict(t_all=T.first_stats(o_rule, np.isfinite(o_rule) & g)),
                       policy=dict(t=T.first_stats(o_pol, np.isfinite(o_pol) & g)))
    return ix


def decision_tables(a, data, sel, preview=False):
    """The decision table on eight cores at every size, its check against p94's record, the
    counts and the curves (module docstring, DECISION TABLE ON EIGHT CORES; with preview the
    first set alone). Returns (the JSON block, {size: the part p94.write_csv reads}). Raises
    Refusal on a row whose columns are means over different cells, on a difference from p94's
    record, and when the record or its curve grid is missing."""
    combos = list(a.combos)
    path = timed_path(a)
    if not os.path.exists(path):
        raise Refusal([f"{shown(path)}: no record of p94, so the four columns it shares with the "
                       'decision table on eight cores cannot be checked'])
    J = json.load(open(path))
    grid = [float(t) for t in (J.get('grid') or {}).get('curve') or []]
    lack = [b for b in a.budgets if float(b) not in grid]
    refusals = [f'{shown(path)}: the curve grid lacks the budgets {lack}'] if lack else []
    tables, rule_values, keep = {}, {}, {}
    for size, D in data.items():
        if D['pol_sel'] is None:
            refusals.append(f'{size}: no policy array of the selected run')
            continue
        use, _, best = best_of(D['M'], D['V'], a.min_seeds)
        Q = None if preview else D.get('policy')
        use2 = bestp = cp = None
        if Q is not None:
            use2, _, bestp = best_of(Q['M'], Q['V'], len(a.seeds))
            cp = (Q['CP'], Q['CPv'])
        elif D.get('preview_cp') is not None:
            cp = (D['preview_cp']['CP'], D['preview_cp']['CPv'])
        S = decision_size(a, D, best, use, bestp, use2, cp)
        tables[size] = S['decision']
        rule_values[size] = {key: T._printed(S['iv'].get('selected', {}).get(key))
                             for key, _ in T.pools(combos)}
        keep[size] = (D, Q, use, best, use2, bestp)
    refusals += T.decision_mismatches({s: dict(decision=t) for s, t in tables.items()})
    if not refusals:
        refusals += p94_check(a, sel, tables, J, path)
    if refusals:
        raise Refusal(refusals)
    # the curves: the best of eight on p94's curve grid
    parts, starts, group_starts, n_rows = {}, {}, {}, 0
    for size, (D, Q, use, best, use2, bestp) in keep.items():
        series = [('memetic_rule', D['cells'], D['V'], use, best)]
        if bestp is not None:
            series.append(('memetic_policy', Q['cells'], Q['V'], use2, bestp))
        curves, group_curves = [], []
        for col, cells, V, pres, table_best in series:
            vals = curve_series(a, cells, V, pres, grid)
            # at the budgets the series is the best of eight of the table
            for q, b in enumerate(a.budgets):
                x, y = vals[grid.index(float(b))][pres], table_best[q][pres]
                if x.shape != y.shape or not np.allclose(x, y, rtol=0, atol=1e-9):
                    refusals.append(f'{size}, {col}: the curve at {b} s differs from the best of '
                                    'eight of the table')
            method = CURVE_METHOD[col]
            c, gc, t0, gt0 = curve_rows(size, method, 'memetic', col.split('_')[1], vals, pres,
                                        D['pol_sel'], combos, grid, a.boot, a.alpha,
                                        len(a.seeds))
            curves += c
            group_curves += gc
            starts.setdefault(size, {})[method] = t0
            group_starts.setdefault(size, {})[method] = gt0
        n_rows += len(curves) + len(group_curves)
        parts[size] = dict(curves=curves, group_curves=group_curves, ix=first_times(D, combos),
                           policy_budget_s=D['policy_budget_s'])
    if refusals:
        raise Refusal(refusals)
    block = dict(
        text=("p94's decision table with the memetic algorithm as the best of the eight runs from "
              'each start, on eight cores' + (', the first set alone' if preview else '')),
        rule=DECISION_RULE, columns=DECISION_TEXT, p94_record=shown(path),
        p94_columns=list(P94_COLUMNS),
        p94_check='every value of these columns equal to the printed value of the record of p94',
        clock=a.origin, budgets=list(a.budgets), table_entries=list(TABLE_ENTRIES),
        sizes={size: dict(rows=rows, rule_values=rule_values[size])
               for size, rows in tables.items()},
        counts=decision_counts(tables, a.budgets),
        curves=dict(file=OUT_FILES['curves'], row='Sel', methods=[CURVE_METHOD[c] for c in
                                                                    BEST_COLUMNS
                                                                    if not preview or c ==
                                                                    'memetic_rule'],
                    n_runs=len(a.seeds), grid=grid, grid_source=f'{shown(path)} grid.curve',
                    start_s=starts, group_start_s=group_starts, n_rows=n_rows,
                    reference='the greedy makespan of the selected run (p94 row Sel)'))
    if preview:
        block['warning'] = PREVIEW_WARNING
    return block, parts


# ----------------------------------------------------------------- outputs
def macro_lines(res):
    sizes = res['results']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["setting"]}; selected training run s{res["selected_run"]}; CP-SAT configuration '
         f'{res["cpsat_configuration"]}; B = {res["boot"]}, alpha = {res["alpha"]}; instance-level '
         f'intervals (p94.compare_at, stats_family.instance_boot, fresh default_rng({T.INST_SEED}) '
         f'per quantity).',
         f'% Clock origin {res["origin"]}: {T.ORIGIN_TEXT[res["origin"]]}.',
         f'% Best of eight: per cell the shortest of the incumbents of the eight memetic runs '
         f'(seeds {", ".join(map(str, res["seeds"]))}, memetic_rule@{ARM_BUDGET}, one thread '
         f'each) at the same time; instances {P.compress(res["instances"])}, every cell with all '
         f'eight records valid. Differences in %: first-named minus second over second '
         f'(negative = first-named shorter); direction from the unrounded interval.']
    if res['origin'] == 'request':
        L.append(f'% The policy, sampled schedules: the best schedule so far of the policy record '
                 f'(budget {res["policy_budget_s"]}), its last value held after the budget; a '
                 f'macro that uses a held value says so.')
    L.append(f'% Verdicts (names ending in Verdict) at the stated budget of {res["primary"]} s by '
             f'the rule of p94 row Sel: {T.VERDICT_TEXT["Sel"]}')

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}SeedN', len(res['seeds']), 'independent memetic runs per cell')
    nc(f'{PREFIX}Boot', res['boot'], 'bootstrap resamples')
    nc(f'{PREFIX}InstN', len(res['instances']),
       f'instances per combination ({P.compress(res["instances"])})')
    for size, S in sizes.items():
        sw = T.SIZE_WORD[size]
        nc(f'{PREFIX}{sw}CellN', S['cells_used'], f'{size}: cells with all eight runs valid')
        for key, word, text, _, with_shares, _, per_combo in COMPARISONS:
            for b, rows in S['comparisons'].get(key, {}).items():
                bw = T.number_word(int(b))
                for r, x in rows.items():
                    if 'mean' not in x or (r in COMBOS and (not per_combo
                                                            or int(b) != res['primary'])):
                        continue
                    base = f'{PREFIX}{sw}{word}{bw}{r}'
                    what = f'{size}, {T.entry_text(r)}, {b} s: {text}' + T.held_note(x)
                    n = f'{x["n"]} instances x {x["c"]} combinations'
                    nc(base, T.f2(x['mean']), f'{what}: mean difference, {n}')
                    nc(base + 'Lo', T.f2(x['lo']), f'{what}: lower end of the 95% interval')
                    nc(base + 'Hi', T.f2(x['hi']), f'{what}: upper end of the 95% interval')
                    nc(base + 'Dir', SF.TEX[x['direction']],
                       f'{what}: direction of [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
                    if with_shares:
                        for k2, w2 in (('shorter', 'Shorter'), ('equal', 'Equal'),
                                       ('longer', 'Longer')):
                            nc(base + w2, f'{x[k2]:.1f}',
                               f'{what}: percent of cells on which the first-named method is {k2}')
        for key, fam in S['verdicts'].items():
            if fam is None or 'no_schedule_yet' in fam:
                continue
            word = next(c[1] for c in COMPARISONS if c[0] == key)
            base = f'{PREFIX}{sw}{word}{T.number_word(int(res["primary"]))}'
            note = T.held_note(fam)
            for r, x in fam['rows'].items():
                nc(f'{base}{r}Verdict', x['verdict'],
                   f'{size}, {r}: verdict; interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; Holm p '
                   f'shorter {x["p_boot_less_holm"]:.2g}, Holm p longer '
                   f'{x["p_boot_greater_holm"]:.2g} (bootstrap){note}')
            for w in T.VERDICT_WORDS:
                nc(f'{base}Verdict{T.COUNT_WORD[w]}', fam['counts'][w],
                   f'{size}: combinations (of {len(fam["rows"])}) {w}{note}')
        for r, x in S['spread']['entries'].items():
            nc(f'{PREFIX}{sw}Spread{r}', f'{x["mean_range_pct"]:.2f}',
               f'{size}, {T.entry_text(r)}, {res["primary"]} s: mean over {x["n_cells"]} cells of '
               f'100 (largest - smallest) / smallest makespan over the eight runs')
            nc(f'{PREFIX}{sw}SpreadMax{r}', f'{x["max_range_pct"]:.2f}',
               f'{size}, {T.entry_text(r)}, {res["primary"]} s: largest of that range over the '
               f'cells')
    if res.get('policy_start'):
        policy_macro_lines(res, L.append, nc)
    if res.get('decision_table'):
        decision_macro_lines(res, L.append, nc)
    return L


def run_texts(res):
    """The runs behind each method of the second set's comparisons, in the macro comments."""
    seeds, sel = P.compress(res['seeds']), res['selected_run']
    return dict(
        bestp=(f'best of seeds {seeds} of {POLICY_SET_ARM}@{ARM_BUDGET} (one thread each, started '
               f'from the rule schedule and the greedy schedule of run s{sel})'),
        best=f'best of seeds {seeds} of memetic_rule@{ARM_BUDGET} (one thread each)',
        smp=f'the policy records of run s{sel} ({T.POLICY_ARM}, budget {res["policy_budget_s"]})',
        cpp=(f'{CP_POLICY_ARM}@{ARM_BUDGET} ({res["cpsat_configuration"]}, 8 workers, started from '
             f'the greedy schedule of run s{sel})'),
        pol=f'the greedy schedule of run s{sel}')


def clock_text(res):
    if res['origin'] == 'request':
        return ('request clock: a memetic run started from the policy\'s schedule at run time '
                't - max(o_pol, o_rule), CP-SAT started from it at t - o_pol, the greedy schedule '
                'from o_pol on, the runs started from the rule schedule as above')
    return f'{res["origin"]} clock as above'


def policy_macro_lines(res, comment, nc):
    """The second set's header comments and macros (module docstring, SECOND SET), after the
    first set's lines; comment(text) adds a comment line, nc(name, value, comment) a macro."""
    p = res['policy_start']
    inst = P.compress(res['instances'])
    rt = run_texts(res)
    comment(f'% Second set, {SET2_TEXT}: per cell the shortest of the incumbents of the eight '
            f'memetic runs (seeds {", ".join(map(str, res["seeds"]))}, '
            f'{POLICY_SET_ARM}@{ARM_BUDGET}, one thread each, started from the rule schedule and '
            f'the greedy schedule of run '
            f's{res["selected_run"]}) at the same time; instances {inst}, every cell with all '
            f'eight records valid; CP-SAT started from the policy\'s schedule: {CP_POLICY_ARM}@'
            f'{ARM_BUDGET}, configuration {res["cpsat_configuration"]}, 8 workers.')
    comment(f'% Clock of the second set: {clock_text(res)}. Differences in % as above '
            f'(first-named minus second over second, negative = first-named shorter); BestVsPol '
            f'and BestPolStartVsPol are the form of the decision table (in percent of the '
            f'policy\'s greedy schedule).')
    for size, S in p['results'].items():
        sw = T.SIZE_WORD[size]
        nc(f'{PREFIX}{sw}PolStartCellN', S['cells_used'],
           f'{size}: cells with all eight runs started from the policy\'s schedule valid (seeds '
           f'{P.compress(res["seeds"])} of {POLICY_SET_ARM}@{ARM_BUDGET}, instances {inst}, eight '
           f'combinations)')
        for key, word, text, _, with_shares, roles, per_combo in COMPARISONS_POLICY:
            for b, rows in S['comparisons'].get(key, {}).items():
                bw = T.number_word(int(b))
                for r, x in rows.items():
                    if 'mean' not in x or (r in COMBOS and (not per_combo
                                                            or int(b) != res['primary'])):
                        continue
                    base = f'{PREFIX}{sw}{word}{bw}{r}'
                    what = (f'{size}, {T.entry_text(r)}, {b} s on the {res["origin"]} clock: '
                            f'{text}' + T.held_note(x))
                    tail = (f'{rt[roles[0]]} against {rt[roles[1]]}; instances {inst}, {x["n"]} '
                            f'instances x {x["c"]} combinations')
                    nc(base, T.f2(x['mean']), f'{what}: mean difference; {tail}')
                    nc(base + 'Lo', T.f2(x['lo']), f'{what}: lower end of the 95% interval; {tail}')
                    nc(base + 'Hi', T.f2(x['hi']), f'{what}: upper end of the 95% interval; {tail}')
                    nc(base + 'Dir', SF.TEX[x['direction']],
                       f'{what}: direction of [{x["lo"]:+.4f}, {x["hi"]:+.4f}]; {tail}')
                    if with_shares:
                        for k2, w2 in (('shorter', 'Shorter'), ('equal', 'Equal'),
                                       ('longer', 'Longer')):
                            nc(base + w2, f'{x[k2]:.1f}',
                               f'{what}: percent of cells on which the first-named method is '
                               f'{k2}; {tail}')


def decision_names(sw, bw, key):
    """{column: (value macro, bold macro)} of a row of the decision table on eight cores:
    p94.decision_names with this script's prefix in the place of Tmd."""
    return {c: tuple(PREFIX + x[len('Tmd'):] for x in v)
            for c, v in T.decision_names(sw, bw, key).items()}


def decision_macro_lines(res, comment, nc):
    """The decision table's header comments, value and bold macros and counts (module
    docstring, DECISION TABLE ON EIGHT CORES), after the second set's lines; comment(text) adds a
    comment line, nc(name, value, comment) a macro."""
    d = res['decision_table']
    sel, inst, clock = res['selected_run'], P.compress(res['instances']), res['origin']
    rt = run_texts(res)
    source = dict(memetic_rule=rt['best'], memetic_policy=rt['bestp'])
    comment(f'% Decision table on eight cores (sections_new/tables/tq3_decision_parallel.tex): '
            f"the rows of p94's decision table at {', '.join(d['sizes'])} on the {clock} clock, "
            f'each value the mean difference of a column from the greedy schedule of run s{sel} '
            f'in percent of it (instances {inst}, the combinations of the row), with the memetic '
            f'algorithm as the best of the eight runs from each start; the columns rules, '
            f"policy_sample, cpsat_rule and cpsat_policy equal p94's values (checked against "
            f'{d["p94_record"]}). Names: p94.decision_names with {PREFIX} in the place of Tmd; '
            f'...Bold is \\bfseries when the column holds the lowest printed value of its row.')
    for size, part in d['sizes'].items():
        sw = T.SIZE_WORD[size]
        rows = part['rows']
        for key, value in part['rule_values'].items():
            row = next((r[key] for r in rows.values() if r[key]['values']['rules'] is not None),
                       None)
            if value is None or row is None:
                continue
            name = decision_names(sw, '', key)['rules'][0]
            nc(name, value,
               f'{size}, {T.entry_text(key)}, every budget of the table (the rule schedules exist '
               f'on every cell from {row["rules_from_s"]:.2f} s on the {clock} clock): '
               f'{DECISION_TEXT["rules"]} minus the greedy schedule of run s{sel}, in percent of '
               f'it; mean over instances {P.compress(row["instances"]["rules"])} x '
               f'combinations {",".join(row["combinations_used"]["rules"])}; equal to '
               f'\\{T.decision_names(sw, "", key)["rules"][0]} of p94')
        for b, brow in rows.items():
            bw = T.number_word(int(b))
            for key, row in brow.items():
                names = decision_names(sw, bw, key)
                tmd = T.decision_names(sw, bw, key)
                what = f'{size}, {T.entry_text(key)}, {b} s on the {clock} clock'
                for col in COLUMN_WORD:
                    v = row['values'][col]
                    if col == 'rules' or v is None:
                        continue
                    tail = (f'; {source[col]}' if col in BEST_COLUMNS
                            else f'; equal to \\{tmd[col][0]} of p94')
                    nc(names[col][0], v,
                       f'{what}: {DECISION_TEXT[col]} minus the greedy schedule of run s{sel}, in '
                       f'percent of it; mean over instances {P.compress(row["instances"][col])} '
                       f'x combinations {",".join(row["combinations_used"][col])}{tail}')
                shown_row = '; '.join(f'{c} {row["values"][c] or "no number"}'
                                      for c in COLUMN_WORD) + f'; instances {inst}'
                for col in COLUMN_WORD:
                    nc(names[col][1], T.BOLD if col in row['bold'] else '',
                       f'{what}: bold marker of \\{names[col][0]} ({DECISION_TEXT[col]}), '
                       f'\\bfseries when its printed value is the lowest of the row, empty '
                       f'otherwise (row: {shown_row})')
    budgets = ', '.join(map(str, d['budgets']))
    for name, c in d['counts'].items():
        sw = BOTH_WORD if name == 'both' else T.SIZE_WORD[name]
        where = (f'{" and ".join(d["sizes"])} together' if name == 'both' else name)
        rows = (f'rows of the table (combinations without and with setups at {budgets} s on the '
                f'{clock} clock; instances {inst})')
        late = ' and '.join(map(str, c['late_budgets']))
        base = f'{PREFIX}{sw}Decision'
        nc(f'{base}RowN', c['rows'], f'{where}: {rows}')
        for col in COLUMN_WORD:
            nc(f'{base}Lowest{COLUMN_WORD[col]}N', c['lowest'][col],
               f'{where}: {rows} whose lowest printed value is in the column '
               f'{DECISION_TEXT[col]} (a tie counts in each tied column)')
        nc(f'{base}TieN', c['ties'], f'{where}: {rows} with two or more lowest columns')
        nc(f'{base}FiveTenMinRowN', c['late_rows'], f'{where}: {rows} at {late} s')
        nc(f'{base}FiveTenMinLowestPolStartN', c['late_lowest_policy_start'],
           f'{where}: {rows} at {late} s whose lowest value is a search started from the '
           "policy's schedule (memetic algorithm or CP-SAT)")
        nc(f'{base}LowestUsesPolicyN', c['lowest_uses_policy'],
           f'{where}: {rows} whose lowest value uses the policy (its sampled schedules, or a '
           "search started from the policy's schedule)")


def decision_markdown(res):
    """The section of the decision table on eight cores, after the second set's."""
    d = res['decision_table']
    md = ['## Decision table on eight cores' + (' (preview, not for the paper)'
                                                if d.get('warning') else ''), '']
    if d.get('warning'):
        md += [f'**{d["warning"]}**', '']
    md += [f'- {d["text"]}; {d["clock"]} clock; difference from the greedy schedule of run '
           f's{res["selected_run"]} in % of it',
           '- the columns of the memetic algorithm (started from the rule schedule, started from '
           "the policy's schedule) hold the best of the eight runs from that start; the columns "
           f'{", ".join(d["p94_columns"])} equal the values of {d["p94_record"]}',
           '- counts over the rows the table prints ('
           + ', '.join(T.entry_text(k) for k in d['table_entries'])
           + f' at {", ".join(map(str, d["budgets"]))} s)',
           f'- curves: {d["curves"]["file"]}, methods {", ".join(d["curves"]["methods"])}, '
           f'{d["curves"]["n_rows"]} rows', '']
    for size, part in d['sizes'].items():
        md += T.markdown_decision(None, table=part['rows'], title=(
            f'### {size}: difference from the greedy schedule in % (row Sel); bold = the lowest '
            'printed value of the row; (n) = instances of the value'))
    heads = [t for _, t in T.DECISION_COLUMNS]
    md += ['### Rows by lowest column', '',
           '| sizes | rows | ' + ' | '.join(heads) + ' | ties | rows at '
           + ' and '.join(map(str, LATE_BUDGETS)) + " s | of these, lowest a search from the "
           "policy's schedule | lowest uses the policy |",
           '|---|---|' + '---|' * (len(heads) + 4)]
    for name, c in d['counts'].items():
        md.append(f'| {name} | {c["rows"]} | ' + ' | '.join(str(c['lowest'][k])
                                                          for k, _ in T.DECISION_COLUMNS)
                  + f' | {c["ties"]} | {c["late_rows"]} | {c["late_lowest_policy_start"]} | '
                  f'{c["lowest_uses_policy"]} |')
    md.append('')
    return md


def policy_status(second, a):
    """The lines of the second set's state, for the run and --check: the state, the valid runs
    per seed and size (with their core sets and rejected records), every rejected record, every
    missing input of a complete set, every problem."""
    word = {'complete': 'complete; read into the outputs after the first set',
            'incomplete': 'not complete; the outputs are those of the first set alone',
            'unavailable': 'not read; the outputs are those of the first set alone',
            'refused': 'REFUSED'}[second['state']]
    if second['state'] == 'complete' and second['missing']:
        word = ("the memetic runs are complete; inputs of CP-SAT started from the policy's "
                'schedule are missing')
    run = f', run s{second["run"]}' if second['run'] is not None else ''
    L = [f'p124: second set, {SET2_TEXT} ({POLICY_SET_ARM}@{ARM_BUDGET}, seeds '
         f'{P.compress(second["seeds"])}{run}): {word}']
    n = len(a.inst) * len(a.combos)
    for size, by in second['counts'].items():
        parts = []
        for k, v in by.items():
            extra = [f'{c} {m}' for c, m in v['cores'].items()] + \
                    ([f'{v["rejected"]} rejected'] if v['rejected'] else [])
            parts.append(f'seed {k} {v["valid"]}' + (f' ({"; ".join(extra)})' if extra else ''))
        L.append(f'p124:   second set {size}: valid runs per seed (of {n} cells): '
                 + ', '.join(parts))
    L += [f'p124: second set: rejected {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]'
          for x in second['rejected']]
    L += [f'p124: second set: missing: {x}' for x in second['missing']]
    L += [f'p124: second set: {x}' for x in second['problems']]
    return L


def defined_macros(own):
    """{name: file} of every macro defined in a .tex file under manuscript/ (the repository's and
    the output root's), the output file left out."""
    import glob
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    roots = {os.path.join(ROOT, 'manuscript'), os.path.dirname(os.path.abspath(own))}
    out = {}
    for d in sorted(roots):
        for f in sorted(glob.glob(os.path.join(d, '**', '*.tex'), recursive=True)):
            if os.path.basename(f) == os.path.basename(own):
                continue
            text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                             for x in open(f, errors='replace').read().splitlines())
            for x, y in rx.findall(text):
                out.setdefault(x or y, shown(f))
    return out


def check_names(lines, own):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x)]
    if bad:
        raise Refusal([f'macro names that are not {PREFIX} plus letters: {bad[:5]}'])
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise Refusal([f'duplicate macro names: {dup[:5]}'])
    have = defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refusal([f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})'])
    return names


def cell_text(x):
    if 'no_schedule_yet' in x:
        return f'no schedule yet ({x["no_schedule_yet"]} of {x["n_cells"]} cells)'
    return (f'{T.f2(x["mean"])} [{T.f2(x["lo"])}, {T.f2(x["hi"])}] {x["direction"]}'
            + (' (held)' if x.get('held') else ''))


def markdown(res):
    md = [f'# Eight independent runs of the memetic algorithm read as one method ({res["run"]})', '']
    if res['run'] != 'full':
        md += [f'**{res["warning"]}**', '']
    md += [f'- {res["setting"]}; selected training run s{res["selected_run"]}; CP-SAT '
           f'configuration {res["cpsat_configuration"]}; B = {res["boot"]}',
           f'- clock origin {res["origin"]}: {T.ORIGIN_TEXT[res["origin"]]}',
           f'- best of eight: the shortest incumbent of seeds {", ".join(map(str, res["seeds"]))} '
           f'at the same time, cells with at least {res["min_seeds"]} valid runs; instances '
           f'{P.compress(res["instances"])}',
           '- differences in %: first-named minus second over second, negative = first-named '
           'shorter; instance-level 95% interval (p94.compare_at); (held): a value of the sampled '
           'schedules held after the policy record\'s budget', '']
    if res['rejected']:
        md += [f'## Rejected records ({len(res["rejected"])})', '']
        md += [f'- {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]' for x in res['rejected']]
        md.append('')
    if res['missing']:
        md += [f'## Missing inputs ({len(res["missing"])})', '']
        md += [f'- {x}' for x in res['missing'][:200]]
        md.append('')
    for size, S in res['results'].items():
        md += [f'## {size}', '', f'Cells used (at least {res["min_seeds"]} of '
               f'{len(res["seeds"])} runs valid): {S["cells_used"]} of '
               f'{S["cells"]}; cells by number of valid runs: '
               + ', '.join(f'{k}: {v}' for k, v in S['valid_seeds_per_cell'].items() if v),
               '', 'Core sets of the valid records per seed: '
               + '; '.join(f'seed {k}: ' + (', '.join(f'{c} {n}' for c, n in v.items()) or 'none')
                           for k, v in S['core_sets'].items()), '']
        heads = [f'{b} s' for b in res['budgets']]
        for key, _, text, *_ in COMPARISONS:
            tab = S['comparisons'].get(key)
            if not tab:
                continue
            md += [f'### {text}', '', '| entry | ' + ' | '.join(heads) + ' |',
                   '|---|' + '---|' * len(heads)]
            for r in entries(S['combinations']):
                if not any(r in tab.get(str(b), {}) for b in res['budgets']):
                    continue
                md.append(f'| {T.entry_text(r)} | ' + ' | '.join(
                    cell_text(tab[str(b)][r]) if r in tab.get(str(b), {}) else '-'
                    for b in res['budgets']) + ' |')
            md.append('')
        for key, fam in S['verdicts'].items():
            text = next(c[2] for c in COMPARISONS if c[0] == key)
            md += [f'### Verdict at {res["primary"]} s: {text}', '']
            if fam is None or 'no_schedule_yet' in fam:
                md += ['No verdict (a combination has no number).', '']
                continue
            md += ['| combination | difference | 95% interval | Holm p shorter | Holm p longer | '
                   'verdict |', '|---|---|---|---|---|---|']
            md += [f'| {r} | {T.f2(x["mean"])} | [{T.f2(x["lo"])}, {T.f2(x["hi"])}] | '
                   f'{x["p_boot_less_holm"]:.2g} | {x["p_boot_greater_holm"]:.2g} | {x["verdict"]} |'
                   for r, x in fam['rows'].items()]
            md += ['', ', '.join(f'{w} {fam["counts"][w]}' for w in T.VERDICT_WORDS)
                   + (' (uses held values)' if fam.get('held') else ''), '']
        sp = S['spread']['entries']
        md += [f'### Spread over the eight runs at {res["primary"]} s', '',
               '| entry | cells | mean of (largest - smallest) / smallest, % | largest, % |',
               '|---|---|---|---|']
        md += [f'| {T.entry_text(r)} | {x["n_cells"]} | {x["mean_range_pct"]:.2f} | '
               f'{x["max_range_pct"]:.2f} |' for r, x in sp.items()]
        md += ['', '### Mean makespans (cells used)', '',
               '| budget | combination | cells | best of eight | seed 1 | policy greedy | '
               'policy sampled | CP-SAT |', '|---|---|---|---|---|---|---|---|']
        for b, rows in S['means'].items():
            for r, x in rows.items():
                md.append(f'| {b} | {r} | {x["n"]} | {x["best_of_eight"]:.2f} | '
                          f'{x["seed_one"]:.2f} | {x["policy_greedy"]:.2f} | '
                          + (f'{x["policy_sampled"]:.2f}' if 'policy_sampled' in x else '-')
                          + ' | ' + (f'{x["cpsat"]:.2f}' if 'cpsat' in x else '-') + ' |')
        md.append('')
    if res.get('policy_start'):
        md += policy_markdown(res)
    if res.get('decision_table'):
        md += decision_markdown(res)
    return '\n'.join(md) + '\n'


def policy_markdown(res):
    """The second set's section of the Markdown, after the first set's."""
    p = res['policy_start']
    md = [f'## Second set: {SET2_TEXT}', '',
          f'- seeds {", ".join(map(str, res["seeds"]))} of {POLICY_SET_ARM}@{ARM_BUDGET}, one '
          f'thread each, started from the rule schedule and the greedy schedule of run '
          f's{res["selected_run"]}; best of eight as above, cells with all eight runs valid; '
          f'CP-SAT: {CP_POLICY_ARM}@{ARM_BUDGET} ({res["cpsat_configuration"]}, 8 workers)',
          f'- {clock_text(res)}',
          '- the last two tables are the form of the decision table: difference from the '
          "policy's greedy schedule, in percent of it", '']
    if p['rejected']:
        md += [f'### Rejected records of the second set ({len(p["rejected"])})', '']
        md += [f'- {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]' for x in p['rejected']]
        md.append('')
    if p['missing']:
        md += [f'### Missing inputs of the second set ({len(p["missing"])})', '']
        md += [f'- {x}' for x in p['missing'][:200]]
        md.append('')
    heads = [f'{b} s' for b in res['budgets']]
    for size, S in p['results'].items():
        md += [f'### {size}, second set', '', f'Cells used (all {len(res["seeds"])} runs valid): '
               f'{S["cells_used"]} of {S["cells"]}; valid runs per seed: '
               + '; '.join(f'seed {k}: {v["valid"]} ('
                           + (', '.join(f'{c} {n}' for c, n in v['cores'].items()) or 'none') + ')'
                           for k, v in p['counts'][size].items()), '']
        for key, _, text, *_ in COMPARISONS_POLICY:
            tab = S['comparisons'].get(key)
            if not tab:
                continue
            md += [f'#### {text}', '', '| entry | ' + ' | '.join(heads) + ' |',
                   '|---|' + '---|' * len(heads)]
            for r in entries(S['combinations']):
                if not any(r in tab.get(str(b), {}) for b in res['budgets']):
                    continue
                md.append(f'| {T.entry_text(r)} | ' + ' | '.join(
                    cell_text(tab[str(b)][r]) if r in tab.get(str(b), {}) else '-'
                    for b in res['budgets']) + ' |')
            md.append('')
        md += ['#### Mean makespans of the second set (cells used)', '',
               '| budget | combination | cells | best of eight started from the policy\'s schedule '
               '| CP-SAT started from the policy\'s schedule |', '|---|---|---|---|---|']
        for b, rows in S['means'].items():
            for r, x in rows.items():
                md.append(f'| {b} | {r} | {x["n"]} | {x["best_of_eight_policy_start"]:.2f} | '
                          + (f'{x["cpsat_policy_start"]:.2f}' if 'cpsat_policy_start' in x
                             else '-') + ' |')
        md.append('')
    return md


def output_paths(a):
    if a.out_root:
        return {k: os.path.join(a.out_root, v) for k, v in OUT_FILES.items()}
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(paths):
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)


# ----------------------------------------------------------------- the run
def check_options(a):
    bad = []
    a.inst = P.parse_instances(a.instances)
    a.seeds = tuple(sorted(set(P.parse_instances(a.seeds_spec))))
    if a.min_seeds is None:
        a.min_seeds = len(a.seeds)
    if SINGLE not in a.seeds:
        bad.append(f'the seeds must include seed {SINGLE}, the run of the existing comparison')
    if not 1 <= a.min_seeds <= len(a.seeds):
        bad.append(f'--min-seeds must lie between 1 and {len(a.seeds)}')
    if a.budgets is None:
        a.budgets = default_budgets(a.origin)
    a.budgets = sorted(set(int(b) for b in a.budgets))
    if any(b <= 0 or b > ARM_BUDGET for b in a.budgets):
        bad.append(f'budgets must lie in (0, {ARM_BUDGET}] s')
    if a.primary not in a.budgets:
        bad.append(f'--primary {a.primary} is not one of the budgets {a.budgets}')
    unknown = [s for s in a.sizes if s not in T.SIZE_WORD]
    if unknown:
        bad.append(f'sizes {unknown} have no macro word')
    full = a.min_seeds == len(a.seeds)
    if full and sorted(a.combos) != sorted(COMBOS):
        bad.append('the full run reads the eight combinations')
    if a.preview_first_set:
        if a.check:
            bad.append('--preview-first-set writes outputs and --check reads only; give one')
        if a.origin != 'request':
            bad.append("--preview-first-set reads the request clock (the clock of p94's decision "
                       'table)')
    if (not full and not a.check) or a.preview_first_set:
        if not a.out_root:
            bad.append('--min-seeds below the number of seeds needs --out-root (the output is not '
                       'for the paper)' if not full else
                       '--preview-first-set needs --out-root (the output is not for the paper)')
        else:
            banned = [os.path.join(ROOT, x) for x in T.RECORD_DIRS + ('results', 'out')] + \
                     [os.path.join(ROOT, x) for x in os.listdir(ROOT) if x.startswith('paper')]
            if os.environ.get('PACT_OUT'):
                banned.append(os.path.abspath(os.environ['PACT_OUT']))
            for d in banned:
                if T._inside(a.out_root, d) or T._inside(d, a.out_root):
                    bad.append(f'--out-root {a.out_root} lies inside or around {shown(d)}')
    return bad


def selections(a):
    """(training run, CP-SAT configuration, problems)."""
    problems = []
    sel = None
    try:
        sel, _ = T.select_policy_seed()
    except (REG.MissingResult, SystemExit) as e:
        problems.append(f'training-run selection unavailable: {e}')
    tr_root = (os.path.join(a.result_root, 'test_results', 'FAMILY') if a.result_root
               else 'test_results/FAMILY')
    bars = a.bars or P.build_parser().parse_args(['table']).bars
    chosen, why = T.select_cpsat_arm(tr_root, [P.parse_cores(s) for s in a.slots], bars)
    if chosen is None:
        problems.append(f"pilot selection unavailable ({len(why)} problem(s), first: "
                        f"{why[0] if why else '-'})")
    elif a.cpsat_arm and a.cpsat_arm != chosen:
        problems.append(f"--cpsat-arm {a.cpsat_arm} differs from the pilot's choice {chosen}")
    return sel, chosen if not a.cpsat_arm or a.cpsat_arm == chosen else None, problems


def print_rejected(rejected):
    for x in rejected:
        print(f'p124: rejected {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]')


def check(a, sel=None, cpsat_arm=None, chosen=True):
    """--check: every missing input, the cells with fewer valid runs; exit 1 while any is
    incomplete."""
    problems = []
    if chosen:
        sel, cpsat_arm, problems = selections(a)
    try:
        data, rejected, missing, _, second = read_inputs(a, sel, cpsat_arm)
    except Refusal as e:
        print('p124: REFUSING: ' + '; '.join(e.args[0][:10]))
        return 1
    print_rejected(rejected)
    for x in problems + missing:
        print(f'p124: missing: {x}')
    bad = bool(problems or missing or rejected)
    for size, D in data.items():
        inc = incomplete_cells(a, D)
        print(f'p124: {size}: {len(a.inst) * len(a.combos)} cells (instances '
              f'{P.compress(a.inst)}, {len(a.combos)} combinations), {len(a.seeds)} seeds; cells '
              f'with all {len(a.seeds)} runs valid: {len(a.inst) * len(a.combos) - len(inc)}; '
              f'incomplete: {len(inc)}')
        for q, k in enumerate(a.seeds):
            n_ok = int(D['V'][q].sum())
            print(f'p124:   seed {k}: {n_ok} valid of {D["V"][q].size}'
                  + (f' (cores: {", ".join(f"{c} {n}" for c, n in D["cores"][k].items())})'
                     if n_ok else ''))
        groups = {}
        for r, idx, ok, miss in inc:
            groups.setdefault((tuple(ok), tuple(miss)), {}).setdefault(r, []).append(idx)
        for (ok, miss), by in groups.items():
            print(f'p124:   valid seeds {P.compress(ok) or "none"}, without a valid record '
                  f'{P.compress(miss)}: ' + '; '.join(f'{r} {P.compress(v)}' for r, v in by.items()))
        bad = bad or bool(inc)
    for x in policy_status(second, a):
        print(x)
    # a complete second set writes the decision table on eight cores, checked against p94's record
    no_timed = (second['state'] == 'complete' and a.origin == 'request'
                and not os.path.exists(timed_path(a)))
    if no_timed:
        print(f'p124: second set: missing: {shown(timed_path(a))}, the record of p94 that the '
              'decision table on eight cores is checked against')
    # the second set changes the exit code only when it is refused, or complete with a rejected
    # or missing input; while it is not complete the check is the first set's
    bad2 = second['state'] == 'refused' or (second['state'] == 'complete' and bool(
        second['rejected'] or second['missing'] or no_timed))
    print('p124: ' + ('incomplete' if bad else
                      'every input of the first set present; the second set is not usable'
                      if bad2 else 'every input present'))
    return 1 if bad or bad2 else 0


def run(a, sel=None, cpsat_arm=None, chosen=True, dry=False):
    """Selections, inputs, analysis and the outputs; a refusal removes the outputs. Returns
    (result, macro names or None, paths)."""
    paths = output_paths(a)
    remove_outputs(paths)
    full = a.min_seeds == len(a.seeds)
    preview = a.preview_first_set
    names = None
    try:
        problems = []
        if chosen:
            sel, cpsat_arm, problems = selections(a)
        if problems and full:
            raise Refusal(problems)
        data, rejected, missing, files, second = read_inputs(a, sel, cpsat_arm)
        print_rejected(rejected)
        for x in policy_status(second, a):
            print(x)
        inc = {s: incomplete_cells(a, D) for s, D in data.items()}
        if full and (missing or any(inc.values())):
            raise Refusal(missing + [f'{s}: {len(v)} cells with fewer than {len(a.seeds)} valid '
                                     f'runs (first {v[0][0]}/{v[0][1]:03d}, without a valid record '
                                     f'for seeds {P.compress(v[0][3])})'
                                     for s, v in inc.items() if v])
        if full and second['state'] == 'refused':
            raise Refusal([f'second set: {x}' for x in second['problems']])
        if full and second['state'] == 'complete' and second['missing']:
            raise Refusal([f'second set: {x}' for x in second['missing']])
        if not full:
            print(f'p124: WARNING: --min-seeds {a.min_seeds}: computed on the cells with at least '
                  f'{a.min_seeds} valid runs; NOT FOR THE PAPER; no macro file is written')
        if preview:
            print('p124: WARNING: --preview-first-set: the decision table and the curves with the '
                  'first set alone; NOT FOR THE PAPER; no macro file is written')
        R = analyse(a, data)
        R2 = (analyse_policy(a, data) if second['state'] == 'complete' and not preview
              else None)
        dec = parts = None
        if a.origin == 'request' and (R2 is not None or preview):
            dec, parts = decision_tables(a, data, sel, preview=preview)
        sb = {D['policy_budget_s'] for D in data.values()} - {None}
        warning = None if full else (f'PARTIAL: cells with at least {a.min_seeds} valid runs; '
                                     f'not for the paper')
        res = dict(generator=GENERATOR,
                   run='preview' if preview else 'full' if full else 'partial', dry=bool(dry),
                   warning=(PREVIEW_WARNING + ('; ' + warning if warning else '') if preview
                            else warning),
                   setting=setting_text(), selected_run=sel, cpsat_configuration=cpsat_arm,
                   origin=a.origin, budgets=a.budgets, primary=a.primary, boot=a.boot,
                   alpha=a.alpha, seeds=list(a.seeds), min_seeds=a.min_seeds,
                   instances=list(a.inst), sizes=list(a.sizes), slots=list(a.slots),
                   policy_budget_s=(f'{min(sb):g} s' if sb else None),
                   statistics=('p94.compare_at and p94.compare (p94.stat_instance: '
                               'stats_family.instance_boot, fresh default_rng(7) per quantity; '
                               'stats_family.direction; p94.shares), p94.mark_held, p94.verdict_at '
                               '(p94.verdict_family: instance_boot draws, '
                               'stats_family.boot_p_one_sided, stats_family.holm, '
                               'p94.selected_verdict)'),
                   record_names=('or_solution/FAMILY/<size>_memetic600s/<R>/instance_<i>_seed<k>'
                                 '.json (timed_campaign_plan.Campaign.record, --memetic-seed k)'),
                   rejected=rejected, missing=missing, problems=problems, arrays=files,
                   results=R)
        if R2 is not None:
            res['policy_start'] = dict(
                text=SET2_TEXT, arm=f'{POLICY_SET_ARM}@{ARM_BUDGET}',
                cpsat=f'{CP_POLICY_ARM}@{ARM_BUDGET}', state=second['state'],
                seeds=second['seeds'], counts=second['counts'], rejected=second['rejected'],
                missing=second['missing'], clock=clock_text(res),
                record_names=('or_solution/FAMILY/<size>_memetic600s_pactwarm_s<run>/<R>/'
                              'instance_<i>_seed<k>.json (timed_campaign_plan.Campaign.record, '
                              '--memetic-seed k, --policy-seed run)'),
                comparisons={c[0]: c[2] for c in COMPARISONS_POLICY}, results=R2)
        if dec is not None:
            res['decision_table'] = dec
        lines = None
        if full and not preview:
            lines = macro_lines(res)
            names = check_names(lines, paths['macros'])
        os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
        if lines is not None:
            emit(paths['macros'], lines)
        with open(paths['json'], 'w') as f:
            json.dump(T.jsonable(res), f, indent=1)
        os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
        with open(paths['md'], 'w') as f:
            f.write(markdown(res))
        if parts is not None:
            os.makedirs(os.path.dirname(paths['curves']), exist_ok=True)
            T.write_csv(paths['curves'], parts, dict(
                run=PREVIEW_RUN if preview else res['run'], origin=a.origin,
                policy_text=f'selected training run s{sel}', all_text=''))
    except BaseException:
        remove_outputs(paths)
        raise
    return res, (names if lines is not None else None), paths


# ----------------------------------------------------------------- dry mode
DRY_SEL, DRY_CPSAT = 302, 'lns'
DRY_N = 12
DRY_O_RULE, DRY_O_POL = 2.0, 8.0


def r0(i, j):
    """Best-rule makespan of instance i in combination j (SPT)."""
    return 500.0 + i + 3 * j


def dry_memetic(i, j, k):
    """Incumbents of seed k: the rule schedule at the start, r0 - 10 - k after 30 s of search,
    r0 - 20 - k after 300 s (seed 8 is the shortest)."""
    R = r0(i, j)
    return [(0.0, R), (30.0, R - 10 - k), (300.0, R - 20 - k)]


def dry_policy(s, i, j):
    return r0(i, j) - 15.0 + (s - DRY_SEL)


def dry_memetic_policy(i, j, k):
    """Incumbents of seed k started from the policy's schedule (the greedy schedule of the
    selected run, shorter than the rule schedule): g at the start, g - 10 - k after 30 s of
    search, g - 20 - k after 300 s."""
    g = dry_policy(DRY_SEL, i, j)
    return [(0.0, g), (30.0, g - 10 - k), (300.0, g - 20 - k)]


RULE_OFF = {'FIFO': 40.0, 'MOR': 60.0, 'SPT': 0.0, 'MWKR': 20.0}


def build_world(tree, sizes, inst, seeds, slots, combos=COMBOS, policy_seeds=(), cp_policy=None):
    """Descriptors, arrays and every record of the dry run under tree (module docstring); with
    policy_seeds, the memetic records of these seeds started from the policy's schedule and the
    records of CP-SAT started from it (cp_policy True or False: those records whatever
    policy_seeds is)."""
    for size in sizes:
        for j, r in enumerate(COMBOS):
            d = os.path.join(tree, 'data', size, 'desc', r)
            os.makedirs(d)
            for i in range(DRY_N):
                with open(os.path.join(d, f'instance_{i:03d}.desc.json'), 'w') as f:
                    json.dump({'base_instance': f'base/instance_{i:03d}', 'regime': r}, f)
        out = os.path.join(tree, 'res', f'{size}_xmask')
        os.makedirs(out)
        for j, r in enumerate(COMBOS):
            col = np.array([r0(i, j) for i in range(DRY_N)])
            for q, off in RULE_OFF.items():
                np.save(os.path.join(out, f'Result_{q}_{r}.npy'),
                        np.stack([col + off, np.full(DRY_N, 0.1)], axis=1))
            for s in REG.seeds():
                np.save(os.path.join(out, f'Result_greedy+10x25+family+xmask-v1-s{s}_{r}.npy'),
                        np.stack([[dry_policy(s, i, j) for i in range(DRY_N)],
                                  np.full(DRY_N, 0.5)], axis=1))
    rr = os.path.join(tree, 'rr')
    ns = argparse.Namespace(result_root=rr, max_invalid=2, data_root=os.path.join(tree, 'data'),
                            slots=list(slots), combos=list(combos), policy_batch=P97.DEFAULT_BATCH,
                            policy_workers=P97.DEFAULT_WORKERS,
                            policy_base_seed=P97.DEFAULT_BASE_SEED,
                            bars=os.path.join(tree, 'bars.json'))
    spec = P.compress(inst)
    cores_a, cores_b = P.parse_cores(slots[0]), P.parse_cores(slots[-1])
    for k in seeds:
        camp = T.make_campaign(list(sizes), [f'memetic_rule@{ARM_BUDGET}:{spec}'],
                               p94_args(ns, k), DRY_SEL, DRY_CPSAT)
        for seg in camp.segments:
            for c in camp.cells(seg):
                j = COMBOS.index(c.combo)
                pts = dry_memetic(c.idx, j, k)
                core = (cores_a if k <= 5 else cores_b)[j % len(cores_a)]
                rec = dict(camp.expected_fields(c), validated=True, threads=1, workers=1,
                           cpu_share=0.999, affinity=[core], search_start_wall_s=1.0,
                           search_start_cpu_s=1.0,
                           seeds=[dict(source='warm_cache', makespan=r0(c.idx, j))],
                           makespan=pts[-1][1],
                           anytime=[[t, t + 1.0, t + 1.0, v] for t, v in pts])
                _dump(camp.record(c), rec)
    camp = T.make_campaign(list(sizes), [f'cpsat_rule@{ARM_BUDGET}:{spec}'], p94_args(ns),
                           DRY_SEL, DRY_CPSAT)
    for seg in camp.segments:
        for c in camp.cells(seg):
            R = r0(c.idx, COMBOS.index(c.combo))
            rec = dict(camp.expected_fields(c), hint='complete', validated=True, status='FEASIBLE',
                       cpu_share=0.95, affinity=cores_b, build_s=5.0, warm_makespan=R,
                       objective=R - 25, anytime=[[3.0, R - 5, None], [50.0, R - 25, None]])
            _dump(camp.record(c), rec)
    for k in policy_seeds:
        camp = T.make_campaign(list(sizes), [f'{POLICY_SET_ARM}@{ARM_BUDGET}:{spec}'],
                               p94_args(ns, k), DRY_SEL, DRY_CPSAT)
        for seg in camp.segments:
            for c in camp.cells(seg):
                j = COMBOS.index(c.combo)
                pts = dry_memetic_policy(c.idx, j, k)
                core = (cores_b if k <= 4 else cores_a)[j % len(cores_a)]
                rec = dict(camp.expected_fields(c), validated=True, threads=1, workers=1,
                           cpu_share=0.999, affinity=[core], search_start_wall_s=1.0,
                           search_start_cpu_s=1.0,
                           seeds=[dict(source='warm_cache', makespan=r0(c.idx, j)),
                                  dict(source='seed_from', makespan=dry_policy(DRY_SEL, c.idx, j))],
                           makespan=pts[-1][1],
                           anytime=[[t, t + 1.0, t + 1.0, v] for t, v in pts])
                _dump(camp.record(c), rec)
    if bool(policy_seeds) if cp_policy is None else cp_policy:
        camp = T.make_campaign(list(sizes), [f'{CP_POLICY_ARM}@{ARM_BUDGET}:{spec}'],
                               p94_args(ns), DRY_SEL, DRY_CPSAT)
        for seg in camp.segments:
            for c in camp.cells(seg):
                g = dry_policy(DRY_SEL, c.idx, COMBOS.index(c.combo))
                rec = dict(camp.expected_fields(c), hint='complete', validated=True,
                           status='FEASIBLE', cpu_share=0.95, affinity=cores_a, build_s=5.0,
                           warm_makespan=g, objective=g - 30,
                           anytime=[[3.0, g - 5, None], [50.0, g - 30, None]])
                _dump(camp.record(c), rec)
    model = REG.PREFIX + REG.model('released', None, DRY_SEL)
    for size in sizes:
        for j, r in enumerate(COMBOS):
            for i in inst:
                _dump(P97.rule_record_path(rr, size, r, i), dry_rule_record(size, r, i, j))
                _dump(P97.record_path(rr, size, POLICY_BUDGET, P97.DEFAULT_BATCH, f's{DRY_SEL}',
                                      r, i), dry_policy_record(size, r, i, j, model))
    return ns


def _dump(path, rec):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(rec, f)


def _machine(n, first):
    cores = list(range(first, first + n))
    return dict(development_run=False, validated=True, idle_before={'idle': True},
                per_worker=[dict(worker=w, core=c, affinity=[c], cpu_share=0.99)
                            for w, c in enumerate(cores)],
                affinity=cores, cpu_share=0.99, workers=n, budget_clock='wall')


def dry_rule_record(size, r, i, j):
    R = r0(i, j)
    return dict(_machine(P97.RULE_WORKERS, 20), arm=P97.RULE_ARM, size=size, combination=r,
                instance=P97.instance_name(i), best_rule='SPT', best_makespan=R, makespan=R,
                t_all=DRY_O_RULE, rules=[dict(rule=q, makespan=R + RULE_OFF[q])
                                         for q in P97.RULE_ORDER])


def dry_policy_record(size, r, i, j, model):
    """Greedy schedule at DRY_O_POL, a sampled one 5 h shorter at 30 s, budget 120 s."""
    g = dry_policy(DRY_SEL, i, j)
    return dict(_machine(P97.DEFAULT_WORKERS, 8), arm=T.POLICY_ARM, size=size, combination=r,
                instance=P97.instance_name(i), budget_s=float(POLICY_BUDGET),
                batch=P97.DEFAULT_BATCH, base_seed=P97.DEFAULT_BASE_SEED, model=model,
                training_seed=DRY_SEL, environment=REG.env(), greedy_makespan=g,
                greedy_t=DRY_O_POL, schedules_columns=list(P97.SCHEDULE_COLUMNS),
                schedules=[[0, 0, 'greedy', None, DRY_O_POL, DRY_O_POL - 1, g, DRY_O_POL, 0],
                           [1, 0, 'sampled', 1, 30.0, 29.0, g - 5, 30.0, 0]],
                anytime=[[DRY_O_POL - 1, DRY_O_POL, DRY_O_POL, g], [29.0, 30.0, 30.0, g - 5]],
                search_start_wall_s=1.0, makespan=g - 5, n_schedules=2)


@contextlib.contextmanager
def dry_setting(tree):
    """The dry tree's arrays and outputs, and the stored values p97's validity compares with."""
    keep = {k: os.environ.get(k) for k in ('PACT_ENV', 'PACT_SEEDS', 'PACT_RESULTS', 'PACT_OUT')}
    old = (P97.stored_greedy, P97.rule_cache_entry)
    os.environ.update(PACT_ENV='exact', PACT_SEEDS='301-305',
                      PACT_RESULTS=os.path.join(tree, 'res'), PACT_OUT=os.path.join(tree, 'out'))
    P97.stored_greedy = lambda model, seed, size, combo, idx: (
        {'synthetic greedy array': dry_policy(DRY_SEL, idx, COMBOS.index(combo))}, [])
    P97.rule_cache_entry = lambda size, combo, idx: dict(
        file='synthetic rule cache', rule='SPT', ms=r0(idx, COMBOS.index(combo)),
        rule_ms={q: r0(idx, COMBOS.index(combo)) + RULE_OFF[q] for q in P97.RULE_ORDER})
    try:
        yield
    finally:
        P97.stored_greedy, P97.rule_cache_entry = old
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def dry_expected(i, j, t, origin='request'):
    """The values the dry run must produce at time t (request clock): best of eight, seed 1,
    greedy, sampled, CP-SAT."""
    R = r0(i, j)
    run = t - DRY_O_RULE - 1.0          # search time of the memetic run (search starts at 1 s)
    best = min(v for s, v in dry_memetic(i, j, 8) if s <= run + 1e-9)
    one = min(v for s, v in dry_memetic(i, j, 1) if s <= run + 1e-9)
    cp = min([R] + [R - 5 for x in (3.0,) if x <= t - DRY_O_RULE] +
             [R - 25 for x in (50.0,) if x <= t - DRY_O_RULE])
    g = dry_policy(DRY_SEL, i, j)
    smp = g - 5 if t >= 30.0 else g
    # the second set: a run started from the policy's schedule is read at t - max(o_pol, o_rule),
    # CP-SAT started from it at t - o_pol
    run_p = t - max(DRY_O_POL, DRY_O_RULE) - 1.0
    bestp = min(v for s, v in dry_memetic_policy(i, j, 8) if s <= run_p + 1e-9)
    cpp = min([g] + [g - 5 for x in (3.0,) if x <= t - DRY_O_POL] +
              [g - 30 for x in (50.0,) if x <= t - DRY_O_POL])
    return dict(best=best, one=one, greedy=g, sampled=smp, cpsat=cp, bestp=bestp, cpsat_p=cpp)


DRY_BUILT = dict(rules=None, policy_sample='sampled', memetic_rule='best', cpsat_rule='cpsat',
                 memetic_policy='bestp', cpsat_policy='cpsat_p')    # column -> key of dry_expected


def dry_built(col, i, j, t):
    """The built difference of a decision-table column at a cell of the dry tree, in percent of
    the greedy schedule."""
    e = dry_expected(i, j, t)
    x = r0(i, j) if DRY_BUILT[col] is None else e[DRY_BUILT[col]]
    return 100.0 * (x - e['greedy']) / e['greedy']


def dry_timed_record(sizes, inst, budgets):
    """A record of p94 for the dry tree: a full run on the request clock with the selected run,
    p94's curve grid (log_grid with the budgets and the policy record's budget), and the four
    columns of its decision table that this script checks, built from the values of the tree."""
    out = dict(generator='synthetic record of p94', run='full', stand_in=False, origin='request',
               selection=dict(seed=DRY_SEL),
               grid=dict(curve=T.log_grid(ARM_BUDGET, extra=list(budgets) + [POLICY_BUDGET])),
               results={})
    for size in sizes:
        dec = {str(b): {key: dict(values={c: T.f2(np.mean([dry_built(c, i, j, float(b))
                                                            for j in members for i in inst]))
                                          for c in P94_COLUMNS})
                        for key, members in T.pools(COMBOS)} for b in budgets}
        out['results'][size] = dict(decision=dec)
    return out


def dry_run(tree, boot=DRY_BOOT, seeds=SEEDS, min_seeds=None, drop=None, before=None,
            check_only=False, policy_seeds=SEEDS, policy_drop=None, timed=None, preview=False,
            cp_policy=None):
    """The whole path on a synthetic tree; returns (res, names, paths, options), or with
    check_only (rc of --check, None, None, options). drop: a list of (seed, size, combination,
    instance) whose memetic record is removed before the run; policy_seeds: the seeds of the
    second set in the tree (none: no record of the second set and, unless cp_policy, of CP-SAT
    started from the policy's schedule); cp_policy: the records of CP-SAT started from the
    policy's schedule in the tree (default: with a seed of the second set); policy_drop: as
    drop, for the second set; before: a function of the
    record root called after the tree is built; timed: a function that changes the synthetic
    record of p94 (dry_timed_record) before it is written; preview: --preview-first-set into
    <tree>/preview."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refusal([f'the dry-run folder {tree} lies inside the repository'])
    with dry_setting(tree):
        ns = build_world(tree, SIZES, P.parse_instances(INSTANCES), seeds, SLOTS,
                         policy_seeds=policy_seeds, cp_policy=cp_policy)
        rec = dry_timed_record(SIZES, P.parse_instances(INSTANCES),
                               default_budgets('request'))
        if timed is not None:
            timed(rec)
        timed_file = os.path.join(tree, 'p94', TIMED_JSON)
        _dump(timed_file, rec)
        for k, size, r, i in (drop or ()):
            os.remove(memetic_path(ns.result_root, size, r, i, k))
        for k, size, r, i in (policy_drop or ()):
            os.remove(memetic_path(ns.result_root, size, r, i, k, run=DRY_SEL))
        if before is not None:
            before(ns.result_root)
        argv = ['--result-root', ns.result_root, '--data-root', ns.data_root,
                '--bars', ns.bars, '--boot', str(boot), '--seeds', P.compress(seeds),
                '--timed-json', timed_file]
        if min_seeds is not None:
            argv += ['--min-seeds', str(min_seeds), '--out-root', os.path.join(tree, 'partial')]
        if preview:
            argv += ['--preview-first-set', '--out-root', os.path.join(tree, 'preview')]
        if check_only:
            argv.append('--check')
        a = build_parser().parse_args(argv)
        bad = check_options(a)
        if bad:
            raise Refusal(bad)
        if check_only:
            return check(a, sel=DRY_SEL, cpsat_arm=DRY_CPSAT, chosen=False), None, None, a
        res, names, paths = run(a, sel=DRY_SEL, cpsat_arm=DRY_CPSAT, chosen=False, dry=True)
    return res, names, paths, a


def memetic_path(result_root, size, combo, idx, seed, run=None):
    """The planner's record name of a memetic_rule@600 cell (Campaign.record), or with run (a
    training run) of a memetic_policy@600 cell started from that run's schedule."""
    folder = f'{size}_memetic{ARM_BUDGET}s' + ('' if run is None else f'_pactwarm_s{run}')
    return os.path.join(result_root, 'or_solution', 'FAMILY', folder, combo,
                        f'instance_{idx:03d}_seed{seed}.json')


def dry_verify(res, curves=None):
    """The dry run's numbers against the values built into the tree; with curves (the path of
    the curves file) its Pool rows from 10 s on as well."""
    bad = dry_verify_first(res)
    inst = P.parse_instances(INSTANCES)
    dec = res.get('decision_table')
    if dec is not None:
        preview = bool(dec.get('warning'))
        members = dict(T.pools(COMBOS))
        for size, part in dec['sizes'].items():
            for b, brow in part['rows'].items():
                for key, row in brow.items():
                    for c in DRY_BUILT:
                        want = (None if preview and c == 'memetic_policy' else
                                T.f2(np.mean([dry_built(c, i, j, float(b)) for j in members[key]
                                              for i in inst])))
                        if row['values'][c] != want:
                            bad.append(f'{size} {b} s {key} decision {c}: {row["values"][c]}, '
                                       f'built {want}')
                    if row['bold'] != T.lowest(row['values']) or not row['same_instances']:
                        bad.append(f'{size} {b} s {key} decision: bold {row["bold"]}, same '
                                   f'instances {row["same_instances"]}')
    if curves is not None:
        col = {v: k for k, v in CURVE_METHOD.items()}
        for x in csv.DictReader(open(curves)):
            t = float(x['t_s'])
            if x['combination'] != T.POOL or t < 10:
                continue
            want = np.mean([dry_built(col[x['method']], i, j, t) for j in range(len(COMBOS))
                            for i in inst])
            if abs(float(x['mean_pct']) - want) > 1e-6:
                bad.append(f'{x["size"]} {x["method"]} {t:g} s curve: {x["mean_pct"]}, built '
                           f'{want:.6f}')
    return bad


def dry_verify_first(res):
    """The numbers of the first and second set against the values built into the tree."""
    bad = []
    inst = P.parse_instances(INSTANCES)
    pairs = (('pol_vs_best', 'greedy', 'best'), ('smp_vs_best', 'sampled', 'best'),
             ('cp_vs_best', 'cpsat', 'best'), ('best_vs_one', 'best', 'one'),
             ('pol_vs_one', 'greedy', 'one'), ('smp_vs_one', 'sampled', 'one'),
             ('cp_vs_one', 'cpsat', 'one'))
    for size, S in res['results'].items():
        for b in res['budgets']:
            t = float(b)
            for key, first, second in pairs:
                rows = S['comparisons'][key][str(b)]
                for j, r in enumerate(COMBOS):
                    e = [dry_expected(i, j, t) for i in inst]
                    want = np.mean([100 * (x[first] - x[second]) / x[second] for x in e])
                    if abs(rows[r]['mean'] - want) > 1e-9:
                        bad.append(f'{size} {r} {b} s {key}: {rows[r]["mean"]}, built {want}')
            if S['comparisons']['best_vs_one'][str(b)]['Pool']['hi'] > 1e-12:
                bad.append(f'{size} {b} s: the best of eight is longer than seed 1 somewhere')
        for c in S['spread']['cells']:
            R = r0(c['instance'], COMBOS.index(c['combination']))
            want = (R - 20 - max(res['seeds']), R - 20 - min(res['seeds']))
            if (c['smallest'], c['largest']) != want \
                    or abs(c['range_pct'] - 100 * (want[1] - want[0]) / want[0]) > 1e-9:
                bad.append(f'{size} {c["combination"]}/{c["instance"]}: spread {c}, built {want}')
    second = (res.get('policy_start') or {}).get('results', {})
    pairs = (('smp_vs_bestp', 'sampled', 'bestp'), ('cpp_vs_bestp', 'cpsat_p', 'bestp'),
             ('bestp_vs_best', 'bestp', 'best'), ('bestp_vs_pol', 'bestp', 'greedy'),
             ('best_vs_pol', 'best', 'greedy'))
    for size, S in second.items():
        if S['cells_used'] != len(inst) * len(COMBOS):
            bad.append(f'{size} second set: {S["cells_used"]} cells used')
        for b in res['budgets']:
            t = float(b)
            for key, first, second_ in pairs:
                rows = S['comparisons'][key][str(b)]
                for j, r in enumerate(COMBOS):
                    e = [dry_expected(i, j, t) for i in inst]
                    want = np.mean([100 * (x[first] - x[second_]) / x[second_] for x in e])
                    if abs(rows[r]['mean'] - want) > 1e-9:
                        bad.append(f'{size} {r} {b} s {key}: {rows[r]["mean"]}, built {want}')
    return bad


# ----------------------------------------------------------------- command line
def build_parser():
    ap = argparse.ArgumentParser(description='eight memetic runs read as one parallel method')
    ap.add_argument('--check', action='store_true', help='read only: list every missing input')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on a synthetic tree in DIR (default: a temporary folder)')
    ap.add_argument('--min-seeds', type=int, default=None,
                    help='cells with at least this many valid runs (default: all seeds); below '
                         'that, not for the paper and no macro file')
    ap.add_argument('--seeds', dest='seeds_spec', default=P.compress(SEEDS))
    ap.add_argument('--sizes', nargs='+', default=list(SIZES))
    ap.add_argument('--instances', default=INSTANCES)
    ap.add_argument('--combos', nargs='+', default=list(COMBOS))
    ap.add_argument('--origin', choices=('request', 'start', 'search'), default='request')
    ap.add_argument('--budgets', nargs='+', type=int, default=None,
                    help='default p94\'s budgets on the clock up to 600 s')
    ap.add_argument('--primary', type=int, default=T.PRIMARY)
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--policy-budget', type=float, default=POLICY_BUDGET)
    ap.add_argument('--policy-batch', type=int, default=P97.DEFAULT_BATCH)
    ap.add_argument('--policy-workers', type=int, default=P97.DEFAULT_WORKERS)
    ap.add_argument('--policy-base-seed', type=int, default=P97.DEFAULT_BASE_SEED)
    ap.add_argument('--cpsat-arm', default=None, help="must equal the pilot's choice")
    ap.add_argument('--slots', nargs='+', default=list(SLOTS))
    ap.add_argument('--bars', default=None)
    ap.add_argument('--max-invalid', type=int, default=2)
    ap.add_argument('--result-root', default='')
    ap.add_argument('--data-root', default='data/FAMILY')
    ap.add_argument('--out-root', default=None,
                    help='root of manuscript/ and results/ (default PACT_OUT through arm_registry.out)')
    ap.add_argument('--timed-json', default=None,
                    help=f"p94's record that the decision table on eight cores is checked against "
                         f'(default {TIMED_JSON} in the repository)')
    ap.add_argument('--preview-first-set', action='store_true',
                    help='the decision table and the curves with the first set alone, into '
                         '--out-root outside the repository; not for the paper, no macro file')
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(_ARGV[1:] if argv is None else argv)
    if a.out_root:
        a.out_root = os.path.abspath(a.out_root)
    if a.timed_json:
        a.timed_json = os.path.abspath(a.timed_json)
    os.chdir(ROOT)
    if a.dry is not None:
        if a.check:
            ap.error('--check and --dry exclude each other')
        if a.preview_first_set:
            ap.error('--preview-first-set reads the records; --dry builds its own')
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p124_dry_')
        try:
            boot = DRY_BOOT if a.boot == SF.B_DEFAULT else a.boot
            res, names, paths, _ = dry_run(tree, boot)
            bad = dry_verify(res, paths['curves'])
            if bad:
                raise Refusal(bad)
        except Refusal as e:
            sys.exit('p124 self-test on synthetic records FAILED: ' + '; '.join(e.args[0][:6]))
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
        S = res['results'][SIZES[0]]
        x = S['comparisons']['pol_vs_best'][str(T.PRIMARY)]['Pool']
        y = res['policy_start']['results'][SIZES[0]]['comparisons']['bestp_vs_best'][
            str(T.PRIMARY)]['Pool']
        n_dec = sum(len(r) for p in res['decision_table']['sizes'].values()
                    for r in p['rows'].values())
        print(f'p124 self-test on synthetic records passed: {len(res["seeds"])} seeds, '
              f'{len(SIZES)} sizes, B = {res["boot"]}, {len(names)} macros; {SIZES[0]} pooled at '
              f'{T.PRIMARY} s: policy minus best of eight {T.f2(x["mean"])}% '
              f'[{T.f2(x["lo"])}, {T.f2(x["hi"])}]; second set, best of eight started from the '
              f"policy's schedule minus best of eight started from the rule schedule "
              f'{T.f2(y["mean"])}% [{T.f2(y["lo"])}, {T.f2(y["hi"])}]; decision table on eight '
              f'cores {n_dec} rows, {res["decision_table"]["curves"]["n_rows"]} curve rows; every '
              f'value equal '
              f'to the built one'
              + (f'; outputs in {os.path.dirname(os.path.dirname(paths["json"]))}' if keep else ''))
        return 0
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    if REG.env() != 'exact':
        sys.exit(f'REFUSING: PACT_ENV={REG.env()}: this analysis reads the exact environment')
    if a.min_seeds is None:
        a.min_seeds = len(P.parse_instances(a.seeds_spec))
    bad = check_options(a)
    if bad:
        sys.exit('REFUSING: ' + '; '.join(bad))
    if a.check:
        return check(a)
    if not a.out_root and os.environ.get('PACT_OUT') is None:
        sys.exit('REFUSING: set PACT_OUT (the output root of the chain, e.g. out/exact)')
    REG.banner()
    try:
        res, names, paths = run(a)
    except Refusal as e:
        print(f'*** REFUSING TO CERTIFY *** {len(e.args[0])} problem(s)', file=sys.stderr)
        for x in e.args[0][:40]:
            print(f'  {x}', file=sys.stderr)
        return 1
    print(f'p124: {res["run"]}: ' + '; '.join(
        f'{s}: {S["cells_used"]} cells' for s, S in res['results'].items())
        + (f'; {len(names)} macros' if names else '; no macro file')
        + '; wrote ' + ', '.join(shown(p) for k, p in paths.items()
                                 if os.path.exists(p)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
