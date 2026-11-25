r"""P137: the memetic algorithm with a budget of 60 s, read one minute after the request.

QUESTION
  The decision table of the paper reads every method 60, 300 and 600 s after the request for a
  schedule. At 50 and 80 modules its two memetic columns are the best of eight single-thread runs
  with a budget of 600 s, one column started from the rule schedule (memetic_rule@600) and one
  started from the policy's greedy schedule (memetic_policy@600), both read by
  scripts/p124_memetic_parallel.py. The frozen configuration builds its initial population during
  the first tenth of the budget (init_frac 0.1 in baselines/memetic/search.py), so a 600-s run
  ends that phase at one minute. Two sets of eight runs with a budget of 60 s repeat the same
  cells with nothing else changed (initial phase 6 s): memetic_rule@60 and memetic_policy@60,
  seeds 1 to 8, instances 003 to 011 of 50x25n30 and 80x25n30, eight combinations. This script
  reads them one minute after the request beside the 600-s runs, the policy and CP-SAT, and puts
  both into the one-minute rows of the decision table.

THE FIVE-MINUTE READING (--budget 300)
  Two sets of eight runs with a budget of 300 s on the same cells (memetic_rule@300 and
  memetic_policy@300, initial phase 30 s; records or_solution/FAMILY/<size>_memetic300s/<R>/ and
  <size>_memetic300s_pactwarm_s<run>/<R>/instance_<i>_seed<k>.json; default core sets 16-23 for
  the rule start and 8-15 for the policy start) are read 300 s after the request in the same way:
  every name below with 60 takes 300 (the keys m300_vs_m600, ..., mp300_vs_pol; the macro words
  ThreeHundred in the place of Sixty), the 600-s runs and p124's record are read at 300 s, and the
  four five-minute rows of the decision table take the 300-s values in both memetic columns.
  Prefix Mfiv; outputs results/memetic_five_minute.json and .md and
  manuscript/macros_memetic_five.tex (step memetic_five_minute). The counts over the twelve rows
  of both sizes are left out of this record (scripts/p138_decision_rows.py counts the rows of
  every reading time); the checks of p124's twelve rows are made as at 60 s. configure(budget)
  sets the names of the module for one reading; at_budget(budget) sets them inside a block.

WHAT IS READ (every reader by import; no record folder is opened by a written-out name)
  60-s runs from the rule schedule
                per seed k the planner campaign memetic_rule@60:3-11 with --memetic-seed k
                (scripts/timed_campaign_plan.py: cells, record paths
                or_solution/FAMILY/<size>_memetic60s/<R>/instance_<i>_seed<k>.json, cell_states with
                timing_validity.classify_single_thread: one thread on one core of --slots, default
                16-23, cpu_share >= 0.9); the reading by p94.read_cells on the request clock; the
                arrays and the check of the rule start schedule by p124.assemble.
  60-s runs from the policy's schedule
                per seed k the planner campaign memetic_policy@60:3-11 with --memetic-seed k and
                --policy-seed the selected training run (p124.selections, arm_registry.select_seed);
                records or_solution/FAMILY/<size>_memetic60s_pactwarm_s<run>/<R>/
                instance_<i>_seed<k>.json, validity by the planner on the core sets of --pol-slots
                (default 16-23 8-15); the arrays and the checks of the rule and policy start
                schedules by p124.assemble_policy.
  600-s runs, request clock, policy, rules, CP-SAT
                p124.read_inputs with p124's own options (budget 60 s, primary 60 s, --ref-slots as
                its --slots, default 16-23 8-15): the eight runs memetic_rule@600, the eight runs
                memetic_policy@600 (p124's second set, which must be complete), CP-SAT
                cpsat_rule@600 with the pilot's configuration (8 workers), the rule and policy
                records of the request clock (o_rule, o_pol, the policy's sampled schedules), the
                best of the four rule arrays and the greedy arrays of the selected training run.
  p124's record results/memetic_parallel.json (arm_registry.out): its smp_vs_best at 60 s and the
                twelve rows of its decision table on eight cores (NoSetup and Setup at 60, 300 and
                600 s at both sizes) with their counts.
  A record counts only when the planner calls it valid; every rejected record is printed with the
  planner's reason. Each 60-s record must carry the configuration (config), the code hash
  (code_sha256_16) and the start caches (warm_cache, seed_from) of the 600-s record of the same
  start, seed and cell; the initial phase is its init_frac times its budget.

CLOCK (p124's request clock, p94.read_cells)
  At time t since the request a run started from the rule schedule holds its incumbent at run
  time t - o_rule (o_rule: t_all of the cell's rule record), a run started from the policy's
  schedule at run time t - o_pol (o_pol: the time of the greedy schedule; the run also holds the
  rule schedule, so p94.read_cells reads it at t - max(o_pol, o_rule)); the greedy schedule exists
  from o_pol on; the sampled schedules are the policy record's best so far. Reading time 60 s. A
  60-s run that has not reached the end of its budget at that time is read at its incumbent.

QUANTITIES at 60 s, per size, per combination and pooled (Pool, NoSetup, Setup: p94.pools)
  best of eight  per cell the shortest incumbent of the eight runs of one set (p124.best_of, all
  eight valid)
  d = 100 (first - second) / second per cell, negative = the first-named method is shorter;
  p94.compare_at (p94.stat_instance: stats_family.instance_boot over base instances, a fresh
  default_rng(7) per quantity, B and alpha of p124's record; stats_family.direction) on the cells
  where both methods have a record:
    m60_vs_m600      best of eight 60-s runs minus best of eight 600-s runs, both from the rule
                     schedule (what the shorter initial phase changes)
    one60_vs_one600  the 60-s run of seed 1 minus the 600-s run of seed 1, from the rule schedule
    smp_vs_m60       the policy's sampled schedules minus the best of eight 60-s runs from the rule
                     schedule; verdict per combination by p94.verdict_at, the rule p124 applies to
                     smp_vs_best (one-sided bootstrap p from the draws of the interval, Holm over
                     the eight combinations per direction, p94.selected_verdict)
    pol_vs_m60       the policy's greedy schedule minus the best of eight 60-s runs from the rule
                     schedule
    cp_vs_m60        CP-SAT started from the rule schedule minus the best of eight 60-s runs from
                     the rule schedule
    m60_vs_pol       the best of eight 60-s runs from the rule schedule minus the greedy schedule,
                     in percent of the greedy schedule (p124's best_vs_pol, the form of the
                     decision table)
    mp60_vs_mp600    best of eight 60-s runs minus best of eight 600-s runs, both from the
                     policy's schedule
    smp_vs_mp60      the policy's sampled schedules minus the best of eight 60-s runs from the
                     policy's schedule; verdict per combination as for smp_vs_m60
    mp60_vs_m60      the best of eight 60-s runs from the policy's schedule minus the best of eight
                     60-s runs from the rule schedule (what the start changes at this budget)
    mp60_vs_pol      the best of eight 60-s runs from the policy's schedule minus the greedy
                     schedule, in percent of it (p124's bestp_vs_pol, the column MemPolStart of
                     the decision table)
  Beside smp_vs_m60: p124's smp_vs_best at 60 s, computed by p124.analyse on the same records and
  required to equal p124's record to the printed digit (mean, interval, direction, cells).
  Spread across the eight 60-s runs of each set at 60 s: p124.spread (per entry the mean and the
  largest of 100 (largest - smallest) / smallest over the cells). The valid records and the
  smallest cpu_share of each set.

DECISION TABLE AT ONE MINUTE
  The rows NoSetup and Setup at 60 s of p124's decision table (its record, block decision_table)
  at both sizes, four rows. The record's two memetic columns must equal the 600-s best of eight
  against the greedy schedule computed here from each start, on the same cells, and the bold
  columns of each of its twelve rows must be p94.lowest of its values. Per one-minute row: the two
  60-s values (m60_vs_pol in the column started from the rule schedule, mp60_vs_pol in the column
  started from the policy's schedule, printed as p94 prints them), each required to be a mean over
  the cells of every other value of the row; the lowest column or columns (p94.lowest) before,
  after the column started from the rule schedule alone takes its 60-s value (rule only), and
  after both memetic columns take their 60-s values. Counts over the four rows: rows whose lowest
  column changes (after both; rule only), rows whose 60-s value from the rule schedule lies below
  the row's lowest value, rows whose lowest column is the policy's sampled schedules before and
  after (after both; rule only), rows whose lowest column uses the policy (the sampled schedules,
  or a search started from the policy's schedule: p124.USES_POLICY_COLUMNS) before and after.
  Over the twelve rows of both sizes (the eight rows at 300 and 600 s unchanged from p124's
  record): rows whose lowest column uses the policy before, which must equal the count of p124's
  record (counts.both, the value of \MparBothDecisionLowestUsesPolicyN), and after.

REFUSALS
  The full run refuses unless every cell of both sets of 60-s runs has a valid record with a
  reading at 60 s (2 x 1,152 records), every input of p124 is present (its second set complete),
  p124's record is a full run on the same setting (selected run, CP-SAT configuration, B, alpha,
  seeds, instances, slots) with a decision table, and every check above holds. --check lists what
  is missing in each set and exits 1 while anything is. --min-seeds n computes a preview once each
  set has n or more seeds whose every cell is valid (seed 1 among those of the rule start); each
  set is read on all of its complete seeds, which the outputs name, and each side of a comparison
  takes the seeds of its own set (the 600-s runs of a start take the seeds of the 60-s runs of the
  same start); every output says preview, no macro file is written, and --out (outside the
  repository's paper, result and record folders and outside PACT_OUT) is required. A refusal
  removes the outputs of the run.

OUTPUTS (arm_registry.out under PACT_OUT, or under --out; at --budget 300 the files of the
  five-minute reading above)
  results/memetic_one_minute.json and .md, manuscript/macros_memetic_minute.tex (macro_io.emit;
  names Mmin plus letters, one comment per macro; the prefix Mmin must be used by no other .tex
  file under manuscript/). The one-minute values and bold markers of the decision table carry
  p94.decision_names with Mmin in the place of Tmd (as p124 names them with Mpar), so that the
  table can read its one-minute rows here by a change of prefix. The chain step
  memetic_one_minute of scripts/run_analysis.sh brings the macro file to manuscript/.

DRY (--dry [DIR]): p124's synthetic tree and p124's record of it (p124.dry_run), eight seeds of
60-s records and of 300-s records of each start added side by side through the planner's record
names, the whole path of the chosen budget on them, and every number checked against the values
built into the tree.

Usage (repository root, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p137_memetic_one_minute.py
  ... --check
  ... --min-seeds 1 --out <folder outside the repository>
  ... --budget 300 --slots 16-23 --pol-slots 8-15 --ref-slots 16-23 8-15
  python scripts/p137_memetic_one_minute.py [--budget 300] --dry [DIR]
"""
import argparse
import contextlib
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
import p124_memetic_parallel as P124  # noqa: E402
from macro_io import emit  # noqa: E402

T = P124.T
P = T.P
SF = P124.SF
Refusal = T.Refusal

GENERATOR = 'scripts/p137_memetic_one_minute.py'
P124_JSON = 'results/memetic_parallel.json'      # p124's record, through arm_registry.out
ARM = 'memetic_rule'
POL_ARM = P124.POLICY_SET_ARM                    # memetic_policy
REF_BUDGET = P124.ARM_BUDGET    # 600, the runs p124 reads
SEEDS = tuple(P124.SEEDS)
SINGLE = P124.SINGLE
SIZES = tuple(P124.SIZES)
COMBOS = tuple(P124.COMBOS)
INSTANCES = P124.INSTANCES
TABLE_ENTRIES = tuple(T.group_keys())          # the rows of the decision table: NoSetup, Setup
EPS = T.EPS
COUNT_WORD = dict(zip(T.VERDICT_WORDS, ('AheadN', 'BehindN', 'NdN')))
DRY_BOOT = P124.DRY_BOOT

# the outputs of the reading at 60 s (OUT_FILES, the names configure sets for --budget 60) and of
# the reading at 300 s
OUT_FILES = {'macros': 'manuscript/macros_memetic_minute.tex',
             'json': 'results/memetic_one_minute.json', 'md': 'results/memetic_one_minute.md'}
FIVE_OUT_FILES = {'macros': 'manuscript/macros_memetic_five.tex',
                  'json': 'results/memetic_five_minute.json',
                  'md': 'results/memetic_five_minute.md'}
# the two readings (--budget): budget of the runs of both sets in seconds, equal to the reading
# time since the request -> prefix of the macros, word of the rows, outputs, default core sets of
# the runs started from the rule schedule and from the policy's schedule
READINGS = {
    60: dict(prefix='Mmin', minute='one-minute', out=dict(OUT_FILES),
             slots=('16-23',), pol_slots=tuple(P124.SLOTS)),           # 16-23 8-15
    300: dict(prefix='Mfiv', minute='five-minute', out=dict(FIVE_OUT_FILES),
              slots=('16-23',), pol_slots=('8-15',))}
BUDGETS = tuple(READINGS)
DEFAULT_BUDGET = 60

# the fields a record of a set shares with the 600-s record of the same start, seed and cell
SAME_FIELDS = (('config', 'configuration'), ('code_sha256_16', 'code'),
               ('warm_cache', 'rule start cache'), ('seed_from', 'policy start cache'))
P124_TEXT = ("p124's smp_vs_best: the policy's sampled schedules minus the best of the eight 600-s "
             "runs started from the rule schedule (p124.analyse on the same records, equal to "
             "p124's record)")
LOWEST_WORD = {'rules': 'rules', 'policy_sample': 'policy samples',
               'memetic_rule': 'memetic from rule', 'cpsat_rule': 'solver from rule',
               'memetic_policy': 'memetic from policy', 'cpsat_policy': 'solver from policy'}


def comparison_keys(b):
    """The keys of the comparisons at budget b, by role: m and mp the best of the runs of budget
    b from the rule schedule and from the policy's schedule, one the run of seed 1, smp the
    policy's sampled schedules, pol its greedy schedule, cp CP-SAT, ref the 600-s runs."""
    r = REF_BUDGET
    return dict(m_ref=f'm{b}_vs_m{r}', one_ref=f'one{b}_vs_one{r}', smp_m=f'smp_vs_m{b}',
                pol_m=f'pol_vs_m{b}', cp_m=f'cp_vs_m{b}', m_pol=f'm{b}_vs_pol',
                mp_ref=f'mp{b}_vs_mp{r}', smp_mp=f'smp_vs_mp{b}', mp_m=f'mp{b}_vs_m{b}',
                mp_pol=f'mp{b}_vs_pol')


def configure(budget):
    """Set the names of the module for the runs with a budget of `budget` s (READINGS), read
    `budget` s after the request: budget, reading time, prefix, outputs, default core sets,
    tokens, the two sets, the comparisons and their keys, the rules of the outputs and the values
    of the dry tree. Every function reads these names when it is called."""
    global BUDGET, READ_AT, PREFIX, OUT_FILES, MINUTE, SLOTS, POL_SLOTS, TOKEN, POL_TOKEN, SETS
    global KEY, COMPARISONS, VERDICT_KEYS, VERDICT_SECOND, TABLE_KEY, ROW_M, ROW_MP, RULES
    global DRY_LATE_S, DRY_POL_SLOT, DRY_PAIRS, DRY_COUNTS, DRY_ALL_ROWS
    if budget not in READINGS:
        raise Refusal([f'--budget {budget}: one of {", ".join(map(str, BUDGETS))} s'])
    R = READINGS[budget]
    BUDGET = budget                 # the budget of the runs of both sets read here
    READ_AT = budget                # seconds since the request
    PREFIX, OUT_FILES, MINUTE = R['prefix'], dict(R['out']), R['minute']
    SLOTS, POL_SLOTS = tuple(R['slots']), tuple(R['pol_slots'])
    TOKEN = f'{ARM}@{BUDGET}'
    POL_TOKEN = f'{POL_ARM}@{BUDGET}'
    b, w, s = BUDGET, T.number_word(BUDGET), f'{BUDGET}-s'
    # the two sets of runs: name -> (token, option of the core sets, text, word of the macros)
    SETS = {'rule': (TOKEN, 'slots', f'the {s} runs started from the rule schedule', ''),
            'policy': (POL_TOKEN, 'pol_slots', f"the {s} runs started from the policy's schedule",
                       'PolStart')}
    KEY = comparison_keys(b)
    K = KEY
    # (key, macro word, text, entries with macros: 'all' every combination and pooled entry,
    #  'pooled' Pool, NoSetup and Setup, 'table' the value of NoSetup and Setup only)
    COMPARISONS = (
        (K['m_ref'], f'M{w}VsMSixHundred', f'the best of the {s} runs minus the best of the 600-s '
         f'runs, both started from the rule schedule, both at {b} s', 'all'),
        (K['one_ref'], f'One{w}VsOneSixHundred', f'the {s} run of seed 1 minus the 600-s run of '
         'seed 1, both started from the rule schedule', 'pooled'),
        (K['smp_m'], f'SmpVsM{w}', f"the policy's sampled schedules minus the best of the {s} "
         'runs started from the rule schedule', 'all'),
        (K['pol_m'], f'PolVsM{w}', f"the policy's greedy schedule minus the best of the {s} runs "
         'started from the rule schedule', 'pooled'),
        (K['cp_m'], f'CpVsM{w}', 'CP-SAT (8 workers) started from the rule schedule minus the best '
         f'of the {s} runs started from the rule schedule', 'pooled'),
        (K['m_pol'], f'M{w}VsPol', f'the best of the {s} runs started from the rule schedule minus '
         "the policy's greedy schedule, in percent of the greedy schedule (the form of the "
         'decision table)', 'table'),
        (K['mp_ref'], f'Mp{w}VsMpSixHundred', f"the best of the {s} runs minus the best of the "
         f"600-s runs, both started from the policy's schedule, both at {b} s", 'all'),
        (K['smp_mp'], f'SmpVsMp{w}', f"the policy's sampled schedules minus the best of the {s} "
         "runs started from the policy's schedule", 'all'),
        (K['mp_m'], f'Mp{w}VsM{w}', f"the best of the {s} runs started from the policy's "
         f'schedule minus the best of the {s} runs started from the rule schedule', 'all'),
        (K['mp_pol'], f'Mp{w}VsPol', f"the best of the {s} runs started from the policy's schedule "
         "minus the policy's greedy schedule, in percent of the greedy schedule (the form of the "
         'decision table)', 'table'))
    VERDICT_KEYS = (K['smp_m'], K['smp_mp'])
    VERDICT_SECOND = {K['smp_m']: f'the best of the {s} runs started from the rule schedule',
                      K['smp_mp']: f"the best of the {s} runs started from the policy's schedule"}
    TABLE_KEY = {'memetic_rule': K['m_pol'], 'memetic_policy': K['mp_pol']}   # column -> value
    ROW_M, ROW_MP = f'm{b}', f'mp{b}'           # the fields of the two values in a row
    RULES = (
        'a record counts when the planner calls it valid (timed_campaign_plan.cell_states, '
        'timing_validity.classify_single_thread: validated, wall-clock budget, one thread on one '
        'core of the slots, cpu_share >= 0.9); a cell enters with all its runs valid',
        f'a {s} record carries the configuration, the code hash and the start caches (warm_cache, '
        'seed_from) of the 600-s record of the same start, seed and cell',
        'request clock of p124: a run started from the rule schedule at run time t - o_rule, a run '
        "started from the policy's schedule at run time t - max(o_pol, o_rule) (p94.read_cells), "
        "the greedy schedule from o_pol on, the sampled schedules the policy record's best so far; "
        f'read at {b} s',
        'best of the runs: per cell the shortest incumbent of the runs of one set at the same time '
        '(p124.best_of); each side of a comparison takes the seeds of its own set',
        'difference: 100 (first - second) / second per cell, negative = the first-named method is '
        'shorter; mean and 95% interval by p94.compare_at (instance-level cluster bootstrap over '
        'base instances, fresh default_rng(7) per quantity); direction from the unrounded interval',
        f'verdicts of {" and ".join(VERDICT_KEYS)} per combination: {T.VERDICT_TEXT["Sel"]} (here '
        "the first-named method is the policy's sampled schedules)",
        f'decision table: the {s} values take the places of the two memetic columns (rule only: '
        f'the column started from the rule schedule alone); each {s} value a mean over the cells '
        'of every other value of its row; lowest column by p94.lowest over the printed values of '
        'the six columns, a tie counting in each tied column',
        f"p124's smp_vs_best at {b} s, the two memetic columns of its decision table at {b} s, the "
        'bold columns of its twelve rows and its count of rows whose lowest column uses the '
        'policy: recomputed here and equal to its record, or the script refuses')
    # the dry tree: search time of a step of the runs started from the policy's schedule that the
    # request clock hides at the reading time with o_pol and would show with o_rule; core set of
    # the runs of each seed started from the policy's schedule; the comparisons against the
    # values built into the tree; the counts of the decision rows the full dry run builds
    DRY_LATE_S = float(READ_AT) - 6.0
    DRY_POL_SLOT = {k: (POL_SLOTS[0] if k <= 4 else POL_SLOTS[-1]) for k in SEEDS}
    DRY_PAIRS = ((K['m_ref'], 'best60', 'best600'), (K['one_ref'], 'one60', 'one600'),
                 (K['smp_m'], 'sampled', 'best60'), (K['pol_m'], 'greedy', 'best60'),
                 (K['cp_m'], 'cpsat', 'best60'), (K['m_pol'], 'best60', 'greedy'),
                 (K['mp_ref'], 'bestp60', 'bestp600'), (K['smp_mp'], 'sampled', 'bestp60'),
                 (K['mp_m'], 'bestp60', 'best60'), (K['mp_pol'], 'bestp60', 'greedy'))
    DRY_COUNTS, DRY_ALL_ROWS = DRY_ROWS[budget]


@contextlib.contextmanager
def at_budget(budget):
    """The names of the module for the reading of `budget` (configure) inside the block, the
    previous ones after it."""
    keep = BUDGET
    configure(budget)
    try:
        yield
    finally:
        configure(keep)


# ----------------------------------------------------------------- small helpers
def shown(path):
    return P124.shown(path)


def load(path):
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path)) as f:
        return json.load(f)


def entries(combos):
    return list(combos) + [k for k, _ in T.pools(combos)]


def p124_namespace(a):
    """p124's own options for its readers: request clock, budget and primary READ_AT, its
    instances, B and alpha; --ref-slots as its --slots."""
    argv = ['--budgets', str(READ_AT), '--primary', str(READ_AT), '--instances', INSTANCES,
            '--boot', str(a.boot), '--alpha', str(a.alpha), '--max-invalid', str(a.max_invalid),
            '--data-root', a.data_root, '--slots', *a.ref_slots]
    if a.result_root:
        argv += ['--result-root', a.result_root]
    if a.bars:
        argv += ['--bars', a.bars]
    pa = P124.build_parser().parse_args(argv)
    bad = P124.check_options(pa)
    if bad:
        raise Refusal(bad)
    return pa


def out_refusals(out):
    """--out must lie outside the repository's paper, result and record folders and PACT_OUT."""
    banned = [os.path.join(ROOT, x) for x in T.RECORD_DIRS + ('results', 'out')] + \
             [os.path.join(ROOT, x) for x in os.listdir(ROOT) if x.startswith('paper')]
    if os.environ.get('PACT_OUT'):
        banned.append(os.path.abspath(os.environ['PACT_OUT']))
    return [f'--out {out} lies inside or around {shown(d)}' for d in banned
            if T._inside(out, d) or T._inside(d, out)]


def lowest_word(cols):
    """The words of the lowest column or columns of a row (a tie joined by 'and')."""
    return ' and '.join(LOWEST_WORD[c] for c in cols) or 'none'


def uses_policy(cols):
    return bool(set(cols) & set(P124.USES_POLICY_COLUMNS))


# ----------------------------------------------------------------- reading
def read_set(pa, name, slots, seeds, sel, cpsat_arm, offsets):
    """The runs of one set (SETS[name]): per seed k the planner campaign <token>:<instances>
    with --memetic-seed k on the core sets slots, read by p94.read_cells on the request clock with
    p124's offsets. Returns (the set's namespace for p124's assemblers, cells {(seed, size,
    combination, instance): reading}, rejected, missing)."""
    token = SETS[name][0]
    ns = argparse.Namespace(**vars(pa))
    ns.slots, ns.seeds = list(slots), tuple(seeds)
    grid = [float(READ_AT)]
    spec = P.compress(pa.inst)
    cells, rej, miss = {}, [], []
    for k in seeds:
        camp = T.make_campaign(list(pa.sizes), [f'{token}:{spec}'], P124.p94_args(ns, k), sel,
                               cpsat_arm or 'ref_fullhint')
        got, _, _ = T.read_cells([camp], grid, 'request', offsets)
        for (_, size, combo, idx), x in got.items():
            cid = f'{token} seed {k}/{size}/{combo}/{idx:03d}'
            if x['state'] in P124.REJECTED:
                rej.append(dict(id=cid, state=x['state'], reason=x['reason'], record=x['record']))
            elif x['state'] == 'valid':
                why = P124.usable_reading(x, grid, True)
                if why is not None:
                    miss.append(f'{cid}: {why} [{x["record"]}]')
                rec = load(x['record'])
                x.update(affinity=rec.get('affinity'), config=rec.get('config'),
                         fields={f: rec.get(f) for f, _ in SAME_FIELDS})
            cells[(k, size, combo, idx)] = x
    return ns, cells, rej, miss


def cores_of(ns, size, cells):
    """{seed: {core set: valid records}} of one size, as p124.assemble counts them."""
    cores = {k: {} for k in ns.seeds}
    for (k, s, _, _), x in cells.items():
        if s != size or x['state'] != 'valid' or x.get('offset_s') is None:
            continue
        aff = x.get('affinity') or []
        slot = P124.slot_of(aff[0], ns.slots) if len(aff) == 1 else None
        cores[k][slot or str(aff)] = cores[k].get(slot or str(aff), 0) + 1
    return cores


def record_differences(x, ref):
    """Texts of every field of SAME_FIELDS in which the record of reading x differs from the
    600-s record of reading ref (same start, seed and cell); empty when they agree."""
    r6 = load(ref['record'])
    bad = []
    for f, what in SAME_FIELDS:
        mine, theirs = x['fields'].get(f), r6.get(f)
        if mine == theirs:
            continue
        if f == 'config':
            bad.append(f'{x["record"]}: configuration differs from {ref["record"]}')
        else:
            bad.append(f'{x["record"]}: {what} {f} {mine!r}, {ref["record"]}: {f} {theirs!r}')
    return bad


def read_all(a, sel, cpsat_arm):
    """Every record and array (module docstring, WHAT IS READ). Returns dict(pa, data:
    p124.read_inputs' arrays per size, rejected and missing: p124's (its second set included),
    files, second: the state of p124's second set, sets: {name: dict(ns, cells, rejected,
    missing, arrays {size: dict(M [K, 1, n, C], V [K, n, C], cores)})}). Raises Refusal on a record
    that contradicts another input: a start schedule that differs from the rule or policy array,
    a configuration, code hash or start cache that differs from the 600-s record of the same
    start, seed and cell."""
    pa = a.pa
    data, rejected, missing, files, second = P124.read_inputs(pa, sel, cpsat_arm)
    rejected = rejected + second['rejected']
    missing = missing + second['missing']
    if second['state'] != 'complete':
        missing.append(f"p124's runs started from the policy's schedule ({POL_ARM}@{REF_BUDGET}): "
                       f'{second["state"]}' + ''.join(f'; {x}' for x in second['problems'][:3]))
    # the request clock of p124: o_rule for a run started from the rules, o_pol (with o_rule)
    # for a run started from the policy's schedule
    offsets = {}
    for size, D in data.items():
        for j, r in enumerate(pa.combos):
            for i, idx in enumerate(pa.inst):
                o = {}
                if np.isfinite(D['o_rule'][i, j]):
                    o['rule'] = float(D['o_rule'][i, j])
                if np.isfinite(D['o_pol'][i, j]):
                    o['policy'] = float(D['o_pol'][i, j])
                if o:
                    offsets[(size, r, idx)] = o
    refusals = []
    sets = {}
    for name in SETS:
        if name == 'policy' and sel is None:
            sets[name] = dict(ns=argparse.Namespace(**dict(vars(pa), slots=list(a.pol_slots),
                                                           seeds=tuple(a.seeds))),
                              cells={}, rejected=[],
                              missing=[f'{POL_TOKEN}: no selected training run, so its record '
                                       'folders are not known'])
        else:
            ns, cells, rej, miss = read_set(pa, name, getattr(a, SETS[name][1]), a.seeds, sel,
                                            cpsat_arm, offsets)
            sets[name] = dict(ns=ns, cells=cells, rejected=rej, missing=miss)
        S = sets[name]
        S['arrays'] = {}
        for size in pa.sizes:
            D = data[size]
            if name == 'rule':
                A = P124.assemble(S['ns'], size, S['cells'], {}, D['rule'], D['pol_sel'], refusals)
                S['arrays'][size] = dict(M=A['M'], V=A['V'], cores=A['cores'])
            else:
                A = P124.assemble_policy(S['ns'], size, S['cells'], D['rule'], D['pol_sel'],
                                         refusals)
                S['arrays'][size] = dict(M=A['M'], V=A['V'],
                                         cores=cores_of(S['ns'], size, S['cells']))
        for (k, size, combo, idx), x in sorted(S['cells'].items(), key=lambda kv: str(kv[0])):
            if x['state'] != 'valid':
                continue
            refs = (data[size]['cells'] if name == 'rule'
                    else (data[size].get('policy') or {}).get('cells') or {})
            ref = refs.get((k, combo, idx))
            if ref is not None:
                refusals += record_differences(x, ref)
    if refusals:
        raise Refusal(refusals)
    return dict(pa=pa, data=data, rejected=rejected, missing=missing, files=files,
                second=second['state'], sets=sets)


def seed_state(a, inp, name):
    """{seed: dict(valid, cells, cores)} of one set over both sizes, and its complete seeds
    (every cell valid with a reading at READ_AT), in seed order."""
    X = inp['sets'][name]['arrays']
    per = {}
    for q, k in enumerate(a.seeds):
        V = [X[s]['V'][q] for s in inp['pa'].sizes]
        cores = {}
        for s in inp['pa'].sizes:
            for c, m in X[s]['cores'][k].items():
                cores[c] = cores.get(c, 0) + m
        per[k] = dict(valid=int(sum(v.sum() for v in V)), cells=int(sum(v.size for v in V)),
                      cores=cores)
    done = [k for k, v in per.items() if v['valid'] == v['cells']]
    return per, done


def read_record(a, pa, sel, cpsat_arm, dry=False):
    """(p124's record or None, its shown path, problems): the record must be a full run of p124
    on this setting with its decision table."""
    path = REG.out(P124_JSON) if not a.p124_json else a.p124_json
    where = shown(path)
    if not os.path.exists(path):
        return None, where, [f'{where}: no record of p124 (step memetic_parallel)']
    J = load(path)
    want = (('run', 'full'), ('origin', 'request'), ('dry', bool(dry)), ('selected_run', sel),
            ('cpsat_configuration', cpsat_arm), ('boot', pa.boot), ('alpha', pa.alpha),
            ('seeds', list(pa.seeds)), ('min_seeds', len(pa.seeds)),
            ('instances', list(pa.inst)), ('slots', list(pa.slots)))
    bad = [f'{where}: {k} {J.get(k)!r}, this run reads {v!r}' for k, v in want if J.get(k) != v]
    dec = J.get('decision_table') or {}
    if not dec:
        bad.append(f"{where}: no decision table (p124 writes it once its eight runs started from "
                   "the policy's schedule are complete)")
    elif 'both' not in (dec.get('counts') or {}):
        bad.append(f'{where}: no counts of the decision table over both sizes')
    for size in pa.sizes:
        rows = ((dec.get('sizes') or {}).get(size) or {}).get('rows', {}).get(str(READ_AT), {})
        lack = [k for k in TABLE_ENTRIES if k not in rows]
        if dec and lack:
            bad.append(f'{where}: no row {", ".join(lack)} at {READ_AT} s, {size}')
        got = (((J.get('results') or {}).get(size) or {}).get('comparisons') or {}).get(
            'smp_vs_best', {}).get(str(READ_AT))
        if got is None:
            bad.append(f'{where}: no smp_vs_best at {READ_AT} s, {size}')
    return J, where, bad


# ----------------------------------------------------------------- analysis
def analyse(a, inp, used):
    """Every quantity at the reading time of every size (module docstring, QUANTITIES; the
    names with 60 hold the runs of the chosen budget); used: {set name: the seeds read}. Returns
    ({size: quantities}, p124.analyse's result)."""
    pa, data = inp['pa'], inp['data']
    combos, inst = list(pa.combos), list(pa.inst)
    B, alpha = pa.boot, pa.alpha
    t = float(READ_AT)
    seeds_r, seeds_p = used['rule'], used['policy']
    q60 = [list(a.seeds).index(s) for s in seeds_r]
    q600 = [list(pa.seeds).index(s) for s in seeds_r]
    qp60 = [list(a.seeds).index(s) for s in seeds_p]
    qp600 = [list(pa.seeds).index(s) for s in seeds_p]
    o60, o600 = list(a.seeds).index(SINGLE), list(pa.seeds).index(SINGLE)
    R124 = P124.analyse(pa, data)
    out = {}
    for size in pa.sizes:
        D = data[size]
        Q = D['policy']
        X, XP = inp['sets']['rule']['arrays'][size], inp['sets']['policy']['arrays'][size]
        V60, VP60 = X['V'][q60], XP['V'][qp60]
        use60, Mv60, best60 = P124.best_of(X['M'][q60], V60, len(seeds_r))
        use6, _, best6 = P124.best_of(D['M'][q600], D['V'][q600], len(seeds_r))
        use8, _, best8 = P124.best_of(D['M'], D['V'], len(pa.seeds))
        usep60, Mvp60, bestp60 = P124.best_of(XP['M'][qp60], VP60, len(seeds_p))
        usep6, _, bestp6 = P124.best_of(Q['M'][qp600], Q['V'][qp600], len(seeds_p))
        usep8, _, bestp8 = P124.best_of(Q['M'], Q['V'], len(pa.seeds))
        pol_sel = D['pol_sel'] if D['pol_sel'] is not None else np.full(use60.shape, np.nan)
        with np.errstate(invalid='ignore'):
            pol_pres = np.isfinite(pol_sel) & np.isfinite(D['o_pol'])
            g = np.where(D['o_pol'] <= t + EPS, pol_sel, np.nan)
        sbud = D['policy_budget_s']
        held = sbud is not None and t > sbud + EPS
        K = KEY
        pairs = {
            K['m_ref']: (best60[0], best6[0], use60 & use6),
            K['one_ref']: (X['M'][o60][0], D['M'][o600][0], X['V'][o60] & D['V'][o600]),
            K['smp_m']: (D['smp'][0], best60[0], use60 & D['sp']),
            K['pol_m']: (g, best60[0], use60 & pol_pres),
            K['cp_m']: (D['CP'][0], best60[0], use60 & D['CPv']),
            K['m_pol']: (best60[0], g, use60 & pol_pres),
            K['mp_ref']: (bestp60[0], bestp6[0], usep60 & usep6),
            K['smp_mp']: (D['smp'][0], bestp60[0], usep60 & D['sp']),
            K['mp_m']: (bestp60[0], best60[0], usep60 & use60),
            K['mp_pol']: (bestp60[0], g, usep60 & pol_pres)}
        S = dict(instances=inst, combinations=combos, seeds=list(seeds_r),
                 pol_seeds=list(seeds_p), cells=int(use60.size), cells_used=int(use60.sum()),
                 pol_cells_used=int(usep60.sum()), comparisons={}, cells_of={})

        def cells_text(got):
            return {r: dict(instances=sorted(int(inst[i]) for i in u[0]), combinations=list(u[1]))
                    for r, u in got.items()}
        for key, *_ in COMPARISONS:
            first, second, pres = pairs[key]
            got = {}
            rows = T.compare_at(first, second, pres, combos, B, alpha, used=got)
            if key.startswith('smp'):
                rows = T.mark_held(rows, held, sbud)
            S['comparisons'][key] = rows
            S['cells_of'][key] = cells_text(got)
        S['verdicts'] = {}
        for key in VERDICT_KEYS:
            fam = (T.verdict_at(*pairs[key], combos, B, alpha)
                   if sorted(combos) == sorted(COMBOS) else None)
            S['verdicts'][key] = dict(fam, held=held, held_after_s=sbud) if fam is not None \
                else None
        S['p124_smp_vs_best'] = R124[size]['comparisons']['smp_vs_best'][str(READ_AT)]
        S['p124_smp_vs_best_verdict'] = R124[size]['verdicts'].get('smp_vs_best')
        # the two memetic columns of p124's decision table at READ_AT (its decision_size): the best
        # of eight 600-s runs from each start against the greedy schedule on the cells of both
        for col, best, use in (('memetic_rule', best8, use8), ('memetic_policy', bestp8, usep8)):
            got = {}
            S[f'p124_{col}'] = T.compare_at(best[0], g, use & pol_pres, combos, B, alpha,
                                            used=got)
            S[f'p124_{col}_cells'] = cells_text(got)
        S['spread'] = P124.spread(combos, inst, Mv60[:, 0], V60, use60)
        S['pol_spread'] = P124.spread(combos, inst, Mvp60[:, 0], VP60, usep60)
        for name, seeds_used, V, Xs, rk in (('rule', seeds_r, V60, X, 'records'),
                                            ('policy', seeds_p, VP60, XP, 'pol_records')):
            shares, fracs = [], set()
            for (k, s, combo, idx), x in inp['sets'][name]['cells'].items():
                if s == size and k in seeds_used and x['state'] == 'valid':
                    shares.append(float(x['cpu_share']))
                    fracs.add((x.get('config') or {}).get('init_frac'))
            S[rk] = dict(
                valid=int(V.sum()), cells=int(V.size),
                per_seed={str(k): int(Xs['V'][list(a.seeds).index(k)].sum()) for k in seeds_used},
                cores={str(k): Xs['cores'][k] for k in seeds_used},
                share_min=min(shares) if shares else None, init_frac=sorted(fracs, key=str))
        out[size] = S
    return out, R124


def printed_entry(x):
    """The printed form of an entry of p94.compare_at: (mean, lo, hi as p94 prints them,
    direction, instances, combinations), or the cells without a schedule."""
    if x is None:
        return None
    if 'mean' not in x:
        return (T.NO_SCHEDULE, x.get('no_schedule_yet'), x.get('n_cells'))
    return (T.f2(x['mean']), T.f2(x['lo']), T.f2(x['hi']), x['direction'], int(x['n']),
            int(x['c']))


def smp_best_mismatches(results, J, where):
    """Texts of every entry of p124's smp_vs_best at READ_AT whose printed form here differs from
    p124's record; empty when every entry is equal."""
    bad = []
    for size, S in results.items():
        rec = J['results'][size]['comparisons']['smp_vs_best'][str(READ_AT)]
        mine = S['p124_smp_vs_best']
        for r in dict.fromkeys(list(mine) + list(rec)):
            x, y = printed_entry(mine.get(r)), printed_entry(rec.get(r))
            if x != y:
                bad.append(f'{size}, {T.entry_text(r)}: smp_vs_best at {READ_AT} s {x} here, {y} '
                           f'in {where}')
    return bad


def row_change(values, m60, mp60=None):
    """One row of the decision table: values {column: printed value or None} of p124's record,
    m60 and mp60 the printed values of the runs of the chosen budget started from the rule and
    from the policy's schedule (mp60 None: that column keeps the record's value), stored in the
    fields ROW_M and ROW_MP. Returns the lowest columns before, after the column started from
    the rule schedule alone takes its value (rule only) and after both memetic columns take
    theirs; whether each value lies below the lowest of the row's values; whether the lowest
    columns change; whether they use the policy."""
    before = T.lowest(values)
    nums = [float(v) for v in values.values() if v is not None]
    rule_only_values = dict(values, memetic_rule=m60)
    after_values = dict(rule_only_values) if mp60 is None else dict(rule_only_values,
                                                                    memetic_policy=mp60)
    rule_only = T.lowest(rule_only_values)
    after = T.lowest(after_values)

    def below(v):
        return bool(v is not None and nums and float(v) < min(nums))
    out = {'values': dict(values), ROW_M: m60, ROW_MP: mp60}
    out.update(rule_only_values=rule_only_values,
               after_values=after_values, before=before, rule_only_after=rule_only, after=after,
               below_lowest=below(m60), below_lowest_policy=below(mp60),
               changed_rule_only=set(before) != set(rule_only), changed=set(before) != set(after),
               uses_policy_before=uses_policy(before), uses_policy_after=uses_policy(after),
               lowest_word=lowest_word(after))
    return out


def row_counts(rows):
    """Counts over the rows {size: {entry: row_change}}."""
    flat = [r for by in rows.values() for r in by.values()]
    return dict(rows=len(flat), changed=sum(r['changed'] for r in flat),
                changed_rule_only=sum(r['changed_rule_only'] for r in flat),
                below_lowest=sum(r['below_lowest'] for r in flat),
                below_lowest_policy=sum(r['below_lowest_policy'] for r in flat),
                pol_smp_before=sum('policy_sample' in r['before'] for r in flat),
                pol_smp_after=sum('policy_sample' in r['after'] for r in flat),
                pol_smp_after_rule_only=sum('policy_sample' in r['rule_only_after']
                                            for r in flat),
                uses_policy_before=sum(r['uses_policy_before'] for r in flat),
                uses_policy_after=sum(r['uses_policy_after'] for r in flat))


def row_cells(row, col):
    """(sorted instances, sorted combinations) of the value of a column of a row of p124's
    record."""
    return (sorted(int(i) for i in (row.get('instances') or {}).get(col) or []),
            sorted((row.get('combinations_used') or {}).get(col) or []))


def decision(J, results, where):
    """The rows at READ_AT of p124's decision table with the two values of the budget (module
    docstring, DECISION TABLE AT ONE MINUTE). Returns (rows {size: {entry: row}}, problems)."""
    rows, bad = {}, []
    for size, S in results.items():
        brow = J['decision_table']['sizes'][size]['rows'][str(READ_AT)]
        for key in TABLE_ENTRIES:
            row = brow[key]
            values = {c: row['values'].get(c) for c, _ in T.DECISION_COLUMNS}
            what = f'{size}, {T.entry_text(key)}, {READ_AT} s'
            for col, start in (('memetic_rule', 'the rule schedule'),
                               ('memetic_policy', "the policy's schedule")):
                m600 = T._printed(S[f'p124_{col}'].get(key))
                if m600 != values[col]:
                    bad.append(f'{what}: memetic column started from {start} {m600} here, '
                               f'{values[col]} in {where}')
            others = {c: row_cells(row, c) for c, v in values.items() if v is not None}
            for col, k60 in TABLE_KEY.items():
                cells = S['cells_of'][k60].get(key)
                mine = None if cells is None else (cells['instances'],
                                                   sorted(cells['combinations']))
                diff = [c for c, w in others.items() if mine != w]
                if mine is None or diff:
                    bad.append(f'{what}: the {BUDGET}-s value of column {col} is a mean over '
                               f'{cells}, the row of {where} over '
                               + '; '.join(f'{c} {others[c]}' for c in diff or others))
            m60 = T._printed(S['comparisons'][KEY['m_pol']].get(key))
            mp60 = T._printed(S['comparisons'][KEY['mp_pol']].get(key))
            inst, comb = row_cells(row, 'memetic_rule')
            rows.setdefault(size, {})[key] = dict(row_change(values, m60, mp60), instances=inst,
                                                  combinations=comb)
    return rows, bad


def all_rows(J, rows, where):
    """The twelve rows the table prints at both sizes (entries NoSetup and Setup at every budget
    of p124's record): the rows at READ_AT after both replacements (rows), the others unchanged.
    Every row's bold columns must be p94.lowest of its values, and the count of rows whose lowest
    column uses the policy before must equal p124's record (counts.both). Returns (dict(rows,
    budgets, uses_policy_before, uses_policy_after, record_rows, record_uses_policy, per_row),
    problems)."""
    dec = J['decision_table']
    bad, per_row = [], []
    for size, part in dec['sizes'].items():
        for b, brow in part['rows'].items():
            for key, row in brow.items():
                if key not in TABLE_ENTRIES:
                    continue
                values = {c: row['values'].get(c) for c, _ in T.DECISION_COLUMNS}
                low = T.lowest(values)
                if sorted(row.get('bold') or []) != sorted(low):
                    bad.append(f'{where}: {size}, {T.entry_text(key)}, {b} s: bold '
                               f'{row.get("bold")}, lowest of its values {low}')
                one = rows.get(size, {}).get(key) if b == str(READ_AT) else None
                after = one['after'] if one is not None else low
                per_row.append(dict(size=size, budget=int(b), entry=key, before=low, after=after))
    rec = (dec.get('counts') or {}).get('both') or {}
    before = sum(uses_policy(x['before']) for x in per_row)
    if rec.get('rows') != len(per_row) or rec.get('lowest_uses_policy') != before:
        bad.append(f'{where}: decision table over both sizes: {rec.get("lowest_uses_policy")} of '
                   f'{rec.get("rows")} rows whose lowest column uses the policy in the record, '
                   f'{before} of {len(per_row)} recomputed from its rows')
    return dict(rows=len(per_row), budgets=sorted({x['budget'] for x in per_row}),
                uses_policy_before=before,
                uses_policy_after=sum(uses_policy(x['after']) for x in per_row),
                record_rows=rec.get('rows'), record_uses_policy=rec.get('lowest_uses_policy'),
                per_row=per_row), bad


# ----------------------------------------------------------------- outputs
def macro_lines(res):
    B, alpha = res['boot'], res['alpha']
    seeds = P.compress(res['seeds'])
    pseeds = P.compress(res['pol_seeds'])
    inst = P.compress(res['instances'])
    sel = res['selected_run']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["setting"]}; selected training run s{sel}; CP-SAT configuration '
         f'{res["cpsat_configuration"]}; B = {B}, alpha = {alpha}; instance-level intervals '
         f'(p94.compare_at, stats_family.instance_boot, fresh default_rng({T.INST_SEED}) per '
         f'quantity).',
         f'% Read at {READ_AT} s since the request (request clock of p124: a memetic run started '
         f'from the rule schedule at run time t - o_rule, one started from the policy\'s schedule '
         f'at run time t - max(o_pol, o_rule), the greedy schedule from o_pol on, the policy\'s '
         f'sampled schedules its best so far).',
         f'% Best of eight: per cell the shortest incumbent of the eight memetic runs of one start '
         f'(seeds {seeds} from the rule schedule, seeds {pseeds} from the greedy schedule of run '
         f's{sel}, one thread each) at the same time; {BUDGET}-s runs {TOKEN} and {POL_TOKEN} '
         f'(initial '
         f'phase {res["init_phase_s"]:g} s), 600-s runs {ARM}@{REF_BUDGET} and '
         f'{POL_ARM}@{REF_BUDGET} (initial phase {res["ref_init_phase_s"]:g} s); instances {inst}, '
         f'eight combinations, every cell with all eight records valid. Differences in %: '
         f'first-named minus second over second (negative = first-named shorter); direction from '
         f'the unrounded interval.',
         f'% Verdicts (names ending in Verdict, and the counts) at {READ_AT} s by the rule of p94 '
         f'row Sel: {T.VERDICT_TEXT["Sel"]} (here the first-named method is the policy\'s sampled '
         f'schedules)']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}SeedN', len(res['seeds']), f'independent {BUDGET}-s runs per cell and start '
       f'(seeds {seeds} of {TOKEN} and of {POL_TOKEN}, one thread each)')
    nc(f'{PREFIX}BudgetS', BUDGET, f'budget of each {BUDGET}-s run, seconds of wall clock')
    nc(f'{PREFIX}InitPhaseS', f'{res["init_phase_s"]:g}', 'seconds of the initial phase of a '
       f'{BUDGET}-s run: init_frac {res["init_frac"]:g} of the frozen configuration (every '
       f'record) times the budget')
    nc(f'{PREFIX}RefBudgetS', REF_BUDGET, 'budget of each run of the existing eight, seconds of '
       'wall clock')
    nc(f'{PREFIX}RefInitPhaseS', f'{res["ref_init_phase_s"]:g}', 'seconds of the initial phase of '
       f'a 600-s run: init_frac {res["init_frac"]:g} times the budget')
    nc(f'{PREFIX}InstN', len(res['instances']), f'instances per combination ({inst})')
    nc(f'{PREFIX}Boot', B, 'bootstrap resamples')
    for name, (token, _, text, word) in SETS.items():
        st = res['set_states'][name]
        n_seeds = len(res['seeds'] if name == 'rule' else res['pol_seeds'])
        nc(f'{PREFIX}{word}CellN', st['cells_total'], f'runs of the plan of {text} ({token}): '
           f'{n_seeds} seeds x {len(res["sizes"])} sizes x 8 combinations x '
           f'{len(res["instances"])} instances')
        nc(f'{PREFIX}{word}CellValidN', st['cells_valid'], f'runs of {text} with a record the '
           f'planner calls valid and a reading at {READ_AT} s')
        nc(f'{PREFIX}{word}ShareMin', f'{st["share_min"]:.4f}', 'smallest cpu_share of the valid '
           f'records of {text}')
    for size, S in res['results'].items():
        sw = T.SIZE_WORD[size]
        for name, (_, _, text, word) in SETS.items():
            rk = 'records' if name == 'rule' else 'pol_records'
            nc(f'{PREFIX}{sw}{word}CellValidN', S[rk]['valid'], f'{size}: runs of {text} with a '
               'valid record')
        for key, word, text, which in COMPARISONS:
            for r, x in S['comparisons'][key].items():
                if 'mean' not in x:
                    continue
                if which == 'pooled' and r in COMBOS or which == 'table' and r not in TABLE_ENTRIES:
                    continue
                base = f'{PREFIX}{sw}{word}{r}'
                what = f'{size}, {T.entry_text(r)}, {READ_AT} s since the request: {text}' + \
                       T.held_note(x)
                n = f'{x["n"]} instances x {x["c"]} combinations'
                if which == 'table':
                    nc(base, T.f2(x['mean']), f'{what}: mean difference, {n}; the value of the '
                       'decision-table row')
                    continue
                nc(base, T.f2(x['mean']), f'{what}: mean difference, {n}')
                nc(base + 'Lo', T.f2(x['lo']), f'{what}: lower end of the 95% interval')
                nc(base + 'Hi', T.f2(x['hi']), f'{what}: upper end of the 95% interval')
                nc(base + 'Dir', SF.TEX[x['direction']],
                   f'{what}: direction of [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
        for key in VERDICT_KEYS:
            fam = S['verdicts'][key]
            if fam is None or 'no_schedule_yet' in fam:
                continue
            word = next(c[1] for c in COMPARISONS if c[0] == key)
            base = f'{PREFIX}{sw}{word}'
            note = T.held_note(fam)
            for r, x in fam['rows'].items():
                nc(f'{base}{r}Verdict', x['verdict'],
                   f'{size}, {r}: verdict of the policy\'s sampled schedules against '
                   f'{VERDICT_SECOND[key]}; interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; Holm p '
                   f'shorter {x["p_boot_less_holm"]:.2g}, Holm p longer '
                   f'{x["p_boot_greater_holm"]:.2g} (bootstrap){note}')
            for w in T.VERDICT_WORDS:
                nc(f'{base}Verdict{COUNT_WORD[w]}', fam['counts'][w],
                   f'{size}: combinations (of {len(fam["rows"])}) on which the policy\'s sampled '
                   f'schedules are {w} against {VERDICT_SECOND[key]}{note}')
        for name, (_, _, text, word) in SETS.items():
            sp = S['spread'] if name == 'rule' else S['pol_spread']
            for r, x in sp['entries'].items():
                nc(f'{PREFIX}{sw}{word}Spread{r}', f'{x["mean_range_pct"]:.2f}',
                   f'{size}, {T.entry_text(r)}, {READ_AT} s: mean over {x["n_cells"]} cells of 100 '
                   f'(largest - smallest) / smallest makespan over {text}')
                nc(f'{PREFIX}{sw}{word}SpreadMax{r}', f'{x["max_range_pct"]:.2f}',
                   f'{size}, {T.entry_text(r)}, {READ_AT} s: largest of that range over the cells')
    # the rows of the decision table at the reading time after both replacements, named as p124
    # names its rows (p94.decision_names with the prefix in the place of Tmd)
    d = res['decision']
    bw = T.number_word(READ_AT)
    for size, by in d['rows'].items():
        sw = T.SIZE_WORD[size]
        for key, row in by.items():
            names = {c: tuple(PREFIX + x[len('Tmd'):] for x in v)
                     for c, v in T.decision_names(sw, bw, key).items()}
            what = (f'{size}, {T.entry_text(key)}, {READ_AT} s on the request clock, decision '
                    f'table with both memetic columns from the {BUDGET}-s runs')
            cells = (f'mean over instances {P.compress(row["instances"])} x combinations '
                     f'{",".join(row["combinations"])}')
            for col, v in (('memetic_rule', row[ROW_M]), ('memetic_policy', row[ROW_MP])):
                if v is None:
                    continue
                text = P124.DECISION_TEXT[col].replace('eight', f'eight {BUDGET}-s')
                nc(names[col][0], v, f'{what}: {text} minus the greedy schedule of run s{sel}, in '
                   f'percent of it; {cells}')
            shown_row = '; '.join(f'{c} {row["after_values"][c] or "no number"}'
                                  for c, _ in T.DECISION_COLUMNS)
            for col, _ in T.DECISION_COLUMNS:
                nc(names[col][1], T.BOLD if col in row['after'] else '',
                   f'{what}: bold marker of the column {col}, \\bfseries when its printed value is '
                   f'the lowest of the row, empty otherwise (row: {shown_row})')
            nc(f'{PREFIX}{sw}{bw}{key}Lowest', row['lowest_word'],
               f'{what}: the lowest column (lowest before: {lowest_word(row["before"])})')
    c = d['counts']
    rows = (f'{MINUTE} rows of the decision table on eight cores (combinations without and with '
            f'setups at {", ".join(res["sizes"])})')
    both = f'with the {BUDGET}-s runs in both memetic columns'
    rule_only = (f'with the {BUDGET}-s runs in the memetic column started from the rule schedule '
                 'alone')
    nc(f'{PREFIX}RowN', c['rows'], rows)
    nc(f'{PREFIX}RowLowestChangeN', c['changed'], f'{rows} whose lowest column changes {both}')
    nc(f'{PREFIX}RuleOnlyRowLowestChangeN', c['changed_rule_only'], f'{rows} whose lowest column '
       f'changes {rule_only}')
    nc(f'{PREFIX}RowBelowLowestN', c['below_lowest'], f'{rows} in which the value of the '
       f'{BUDGET}-s runs started from the rule schedule lies below the lowest value of the row')
    nc(f'{PREFIX}RowLowestPolSmpBeforeN', c['pol_smp_before'], f'{rows} whose lowest column is '
       "the policy's sampled schedules, with the 600-s runs in both memetic columns")
    nc(f'{PREFIX}RowLowestPolSmpAfterN', c['pol_smp_after'], f'{rows} whose lowest column is the '
       f"policy's sampled schedules, {both}")
    nc(f'{PREFIX}RuleOnlyRowLowestPolSmpAfterN', c['pol_smp_after_rule_only'], f'{rows} whose '
       f"lowest column is the policy's sampled schedules, {rule_only}")
    uses = "uses the policy (its sampled schedules, or a search started from the policy's schedule)"
    nc(f'{PREFIX}RowUsesPolicyBeforeN', c['uses_policy_before'], f'{rows} whose lowest column '
       f'{uses}, with the 600-s runs in both memetic columns')
    nc(f'{PREFIX}RowUsesPolicyAfterN', c['uses_policy_after'], f'{rows} whose lowest column '
       f'{uses}, {both}')
    a12 = d.get('all_rows')
    if a12 is None:             # the counts over the twelve rows belong to the reading at 60 s
        return L
    rows12 = (f'rows of the decision table on eight cores (combinations without and with setups at '
              f'{", ".join(map(str, a12["budgets"]))} s, {" and ".join(res["sizes"])} together; '
              f'the rows at {READ_AT} s {both}, the others as in p124\'s record)')
    nc(f'{PREFIX}BothDecisionRowN', a12['rows'], rows12)
    nc(f'{PREFIX}BothDecisionLowestUsesPolicyN', a12['uses_policy_after'],
       f'{rows12} whose lowest column {uses}; with the 600-s runs {a12["uses_policy_before"]} '
       f'(p124\'s record)')
    return L


def check_names(lines, own):
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


def cell_text(x):
    if x is None:
        return '-'
    if 'no_schedule_yet' in x:
        return f'no schedule yet ({x["no_schedule_yet"]} of {x["n_cells"]} cells)'
    return (f'{T.f2(x["mean"])} [{T.f2(x["lo"])}, {T.f2(x["hi"])}] {x["direction"]}'
            + (' (held)' if x.get('held') else ''))


def records_text(rc):
    return (f'valid records {rc["valid"]} of {rc["cells"]}; per seed: '
            + '; '.join(f'seed {k}: {v} (' + ', '.join(f'{c} {m}' for c, m in
                                                       rc['cores'][k].items()) + ')'
                        for k, v in rc['per_seed'].items())
            + f'; smallest cpu_share {rc["share_min"]:.4f}')


def markdown(res):
    md = [f'# The memetic algorithm with a budget of {BUDGET} s, read {READ_AT} s after the '
          f'request ({res["run"]} run)', '']
    if res.get('warning'):
        md += [f'**{res["warning"]}**', '']
    st = res['set_states']
    md += [f'Generated by {GENERATOR}. {res["setting"]}; selected training run '
           f's{res["selected_run"]}; CP-SAT configuration {res["cpsat_configuration"]}; B = '
           f'{res["boot"]}.', '', '## What is read', '',
           f'- {BUDGET}-s runs from the rule schedule: {TOKEN}, seeds {P.compress(res["seeds"])}, '
           f'instances {P.compress(res["instances"])}, eight combinations, core sets '
           f'{" ".join(res["slots"])}; records {res["record_names"]}; valid records '
           f'{st["rule"]["cells_valid"]} of {st["rule"]["cells_total"]}; smallest cpu_share '
           f'{st["rule"]["share_min"]:.4f}',
           f"- {BUDGET}-s runs from the policy's schedule: {POL_TOKEN}, seeds "
           f'{P.compress(res["pol_seeds"])}, the same cells, core sets '
           f'{" ".join(res["pol_slots"])}; '
           f'records {res["pol_record_names"]}; valid records {st["policy"]["cells_valid"]} of '
           f'{st["policy"]["cells_total"]}; smallest cpu_share {st["policy"]["share_min"]:.4f}',
           f'- initial phase of a {BUDGET}-s run {res["init_phase_s"]:g} s, of a 600-s run '
           f'{res["ref_init_phase_s"]:g} s',
           f'- 600-s runs: {ARM}@{REF_BUDGET} and {POL_ARM}@{REF_BUDGET}, seeds '
           f'{P.compress(res["ref_seeds"])}, core sets {" ".join(res["ref_slots"])}; with the '
           "request clock, the policy's greedy and sampled schedules, the rules and CP-SAT "
           'through p124.read_inputs',
           f'- p124\'s record: {res["p124_record"]}', '', '## Rules', '']
    md += [f'- {x}' for x in res['rules']] + ['']
    for size, S in res['results'].items():
        md += [f'## {size}', '',
               f'Cells used: {S["cells_used"]} of {S["cells"]} (rule start), {S["pol_cells_used"]} '
               f'of {S["cells"]} (policy start).', '',
               f'- rule start: {records_text(S["records"])}',
               f'- policy start: {records_text(S["pol_records"])}', '',
               f'### Differences at {READ_AT} s, in % of the second-named method', '']
        md += [f'- {key}: {text}' for key, _, text, _ in COMPARISONS] + [f'- {P124_TEXT}', '']
        for part in (COMPARISONS[:6], COMPARISONS[6:]):
            heads = [key for key, *_ in part] + (['smp_vs_best (p124)'] if part[0][0] ==
                                                 KEY['m_ref'] else [])
            md += ['| entry | ' + ' | '.join(heads) + ' |', '|---|' + '---|' * len(heads)]
            for r in entries(S['combinations']):
                cols = [cell_text(S['comparisons'][k].get(r)) for k, *_ in part]
                if part[0][0] == KEY['m_ref']:
                    cols.append(cell_text(S['p124_smp_vs_best'].get(r)))
                md.append(f'| {T.entry_text(r)} | ' + ' | '.join(cols) + ' |')
            md.append('')
        fam124 = S['p124_smp_vs_best_verdict']
        for key in VERDICT_KEYS:
            fam = S['verdicts'][key]
            beside = key == KEY['smp_m']
            md += [f'### Verdicts at {READ_AT} s: the policy\'s sampled schedules against '
                   f'{VERDICT_SECOND[key]}' + (', beside the best of the 600-s runs (p124)'
                                               if beside else ''), '']
            if fam is None or 'no_schedule_yet' in fam:
                md += ['No verdict (a combination has no number).', '']
                continue
            md += ['| combination | difference | 95% interval | Holm p shorter | Holm p longer | '
                   'verdict |' + (' verdict against the 600-s runs |' if beside else ''),
                   '|---|---|---|---|---|---|' + ('---|' if beside else '')]
            for r, x in fam['rows'].items():
                v124 = ((fam124 or {}).get('rows') or {}).get(r, {}).get('verdict', '-')
                md.append(f'| {r} | {T.f2(x["mean"])} | [{T.f2(x["lo"])}, {T.f2(x["hi"])}] | '
                          f'{x["p_boot_less_holm"]:.2g} | {x["p_boot_greater_holm"]:.2g} | '
                          f'{x["verdict"]} |' + (f' {v124} |' if beside else ''))
            md += ['', 'Counts: ' + ', '.join(f'{w} {fam["counts"][w]}' for w in T.VERDICT_WORDS)
                   + ('; against the 600-s runs: ' + ', '.join(
                       f'{w} {fam124["counts"][w]}' for w in T.VERDICT_WORDS)
                      if beside and fam124 and 'counts' in fam124 else ''), '']
        for name, (_, _, text, _) in SETS.items():
            sp = S['spread'] if name == 'rule' else S['pol_spread']
            md += [f'### Spread over {text} at {READ_AT} s', '',
                   '| entry | cells | mean of (largest - smallest) / smallest, % | largest, % |',
                   '|---|---|---|---|']
            md += [f'| {T.entry_text(r)} | {x["n_cells"]} | {x["mean_range_pct"]:.2f} | '
                   f'{x["max_range_pct"]:.2f} |' for r, x in sp['entries'].items()]
            md.append('')
    d = res['decision']
    cols = [c for c, _ in T.DECISION_COLUMNS]
    md += [f'## Decision table on eight cores, {READ_AT} s since the request', '',
           f'Values of the record of p124 ({res["p124_record"]}): difference from the greedy '
           f"schedule in % of it; the {BUDGET}-s values take the places of memetic_rule and "
           'memetic_policy (rule only: memetic_rule alone).', '',
           '| size | entry | ' + ' | '.join(cols) + f' | {BUDGET}-s from rule | {BUDGET}-s from '
           'policy | lowest before | lowest, rule only | lowest after | changes |',
           '|---|---|' + '---|' * (len(cols) + 6)]
    for size, by in d['rows'].items():
        for key, r in by.items():
            md.append(f'| {size} | {T.entry_text(key)} | '
                      + ' | '.join(r['values'][c] or '-' for c in cols)
                      + f' | {r[ROW_M] or "-"} | {r[ROW_MP] or "-"} | '
                      + ' | '.join(', '.join(r[k]) or '-'
                                   for k in ('before', 'rule_only_after', 'after'))
                      + f' | {"yes" if r["changed"] else "no"} |')
    c, a12 = d['counts'], d.get('all_rows')
    md += ['', f'Counts over the {c["rows"]} rows: lowest column changes {c["changed"]} (rule only '
           f'{c["changed_rule_only"]}); the {BUDGET}-s value from the rule schedule below the '
           f'lowest of '
           f'the row {c["below_lowest"]}, from the policy\'s schedule {c["below_lowest_policy"]}; '
           f'lowest column the policy\'s sampled schedules before {c["pol_smp_before"]}, after '
           f'{c["pol_smp_after"]} (rule only {c["pol_smp_after_rule_only"]}); lowest column uses '
           f'the policy before {c["uses_policy_before"]}, after {c["uses_policy_after"]}.', '']
    if a12 is not None:
        md += [f'Over the {a12["rows"]} rows at {", ".join(map(str, a12["budgets"]))} s of both '
               f'sizes (the rows at {READ_AT} s after both replacements): lowest column uses the '
               f'policy before {a12["uses_policy_before"]} (p124\'s record: '
               f'{a12["record_uses_policy"]} of {a12["record_rows"]}), after '
               f'{a12["uses_policy_after"]}.', '']
    if res.get('rejected'):
        md += [f'## Rejected records ({len(res["rejected"])})', '']
        md += [f'- {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]'
               for x in res['rejected'][:200]]
        md.append('')
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
    """Option problems (empty when none); sets a.seeds and a.pa (p124's namespace)."""
    bad = []
    a.seeds = SEEDS
    if getattr(a, 'budget', BUDGET) != BUDGET:
        bad.append(f'--budget {a.budget}: the names of the module are those of the reading at '
                   f'{BUDGET} s (configure)')
    if a.min_seeds is not None:
        if not 1 <= a.min_seeds <= len(SEEDS):
            bad.append(f'--min-seeds must lie between 1 and {len(SEEDS)}')
        if a.check:
            bad.append('--check reads only and --min-seeds writes a preview; give one')
        elif not a.out:
            bad.append('--min-seeds needs --out (a preview is not for the paper)')
    if a.out:
        bad += out_refusals(a.out)
    for s in list(a.slots) + list(a.pol_slots) + list(a.ref_slots):
        try:
            P.parse_cores(s)
        except (ValueError, SystemExit):
            bad.append(f'core set {s!r}')
    try:
        a.pa = p124_namespace(a)
    except Refusal as e:
        bad += list(e.args[0])
    return bad


def print_rejected(inp):
    for x in inp['rejected'] + [y for S in inp['sets'].values() for y in S['rejected']]:
        print(f'p137: rejected {x["id"]}: {x["state"]} ({x["reason"]}) [{x["record"]}]')


def selections(a, sel, cpsat_arm, chosen):
    if not chosen:
        return sel, cpsat_arm, []
    return P124.selections(a.pa)


def check(a, sel=None, cpsat_arm=None, chosen=True, dry=False):
    """--check: every missing input, the incomplete cells of each set of runs, the state of
    p124's record; exit 1 while anything is missing."""
    sel, cpsat_arm, problems = selections(a, sel, cpsat_arm, chosen)
    try:
        inp = read_all(a, sel, cpsat_arm)
    except Refusal as e:
        print('p137: REFUSING: ' + '; '.join(e.args[0][:10]))
        return 1
    print_rejected(inp)
    sets = inp['sets']
    for x in problems + inp['missing'] + [y for S in sets.values() for y in S['missing']]:
        print(f'p137: missing: {x}')
    bad = bool(problems or inp['missing'] or inp['rejected']
               or any(S['missing'] or S['rejected'] for S in sets.values()))
    for name, S in sets.items():
        token = SETS[name][0]
        ns = S['ns']
        for size in inp['pa'].sizes:
            X = S['arrays'][size]
            inc = P124.incomplete_cells(ns, X)
            n = len(ns.inst) * len(ns.combos)
            print(f'p137: {size}: {token}, {n} cells (instances {P.compress(ns.inst)}, '
                  f'{len(ns.combos)} combinations), {len(a.seeds)} seeds, core sets '
                  f'{" ".join(ns.slots)}; cells with all {len(a.seeds)} runs valid: '
                  f'{n - len(inc)}; incomplete: {len(inc)}')
            for q, k in enumerate(a.seeds):
                n_ok = int(X['V'][q].sum())
                print(f'p137:   seed {k}: {n_ok} valid of {X["V"][q].size}'
                      + (f' (cores: {", ".join(f"{c} {m}" for c, m in X["cores"][k].items())})'
                         if n_ok else ''))
            groups = {}
            for r, idx, ok, miss in inc:
                groups.setdefault((tuple(ok), tuple(miss)), {}).setdefault(r, []).append(idx)
            for (ok, miss), by in groups.items():
                print(f'p137:   valid seeds {P.compress(ok) or "none"}, without a valid record '
                      f'{P.compress(miss)}: ' + '; '.join(f'{r} {P.compress(v)}'
                                                          for r, v in by.items()))
            bad = bad or bool(inc)
    for name in SETS:
        _, done = seed_state(a, inp, name)
        print(f'p137: {SETS[name][0]}: seeds with every cell valid: {P.compress(done) or "none"}')
    print('p137: a preview with --min-seeds n reads every complete seed of each set once each set '
          'has n of them (seed 1 among those of the rule start)')
    J, where, rec_bad = read_record(a, inp['pa'], sel, cpsat_arm, dry)
    for x in rec_bad:
        print(f'p137: p124 record: {x}')
    bad = bad or bool(rec_bad)
    print('p137: ' + ('incomplete' if bad else 'every input present'))
    return 1 if bad else 0


def full_gaps(a, inp):
    """Every reason the full run cannot read both sets: missing readings, rejected records, cells
    with fewer valid runs than seeds."""
    gaps = []
    for name, S in inp['sets'].items():
        text = SETS[name][2]
        gaps += list(S['missing'])
        if S['rejected']:
            x = S['rejected'][0]
            gaps.append(f'{len(S["rejected"])} records of the {BUDGET}-s runs rejected, {text} '
                        f'(first '
                        f'{x["id"]}: {x["state"]} ({x["reason"]}))')
        for size in inp['pa'].sizes:
            inc = P124.incomplete_cells(S['ns'], S['arrays'][size])
            if inc:
                gaps.append(f'{size}: {len(inc)} cells with fewer than {len(a.seeds)} valid '
                            f'{BUDGET}-s '
                            f'runs, {text} (first {inc[0][0]}/{inc[0][1]:03d}, without a valid '
                            f'record for seeds {P.compress(inc[0][3])})')
    return gaps


def run(a, sel=None, cpsat_arm=None, chosen=True, dry=False):
    """Selections, inputs, analysis and the outputs; a refusal removes the outputs. Returns
    (result, macro names or None, paths)."""
    paths = output_paths(a)
    remove_outputs(paths)
    full = a.min_seeds is None
    names = None
    try:
        sel, cpsat_arm, problems = selections(a, sel, cpsat_arm, chosen)
        if problems or sel is None or cpsat_arm is None:
            raise Refusal(problems or ['no selected training run or no CP-SAT configuration'])
        inp = read_all(a, sel, cpsat_arm)
        print_rejected(inp)
        pa = inp['pa']
        if inp['missing'] or inp['rejected']:
            raise Refusal([f'an input of p124 is missing: {x}' for x in inp['missing']]
                          + [f'{len(inp["rejected"])} records of p124 rejected (first '
                             f'{inp["rejected"][0]["id"]}: {inp["rejected"][0]["state"]})']
                          * bool(inp['rejected']))
        states = {name: seed_state(a, inp, name) for name in SETS}
        J, where, rec_bad = read_record(a, pa, sel, cpsat_arm, dry)
        if rec_bad:
            raise Refusal(rec_bad)
        if full:
            gaps = full_gaps(a, inp)
            if gaps:
                raise Refusal(gaps)
            used = {name: list(a.seeds) for name in SETS}
        else:
            done_r, done_p = states['rule'][1], states['policy'][1]
            if len(done_r) < a.min_seeds or SINGLE not in done_r or len(done_p) < a.min_seeds:
                raise Refusal([f'--min-seeds {a.min_seeds}: seeds with every cell valid '
                               f'{P.compress(done_r) or "none"} ({TOKEN}), '
                               f'{P.compress(done_p) or "none"} ({POL_TOKEN}); the preview needs '
                               f'{a.min_seeds} of them in each set, seed {SINGLE} among those of '
                               f'{TOKEN}'])
            used = dict(rule=done_r, policy=done_p)
            print(f'p137: WARNING: --min-seeds {a.min_seeds}: preview on seeds '
                  f'{P.compress(done_r)} of {TOKEN} and seeds {P.compress(done_p)} of {POL_TOKEN}; '
                  'NOT FOR THE PAPER; no macro file is written')
        R, _ = analyse(a, inp, used)
        bad = smp_best_mismatches(R, J, where)
        rows, bad2 = decision(J, R, where)
        a12, bad3 = all_rows(J, rows, where)
        fracs = {f for S in R.values() for f in S['records']['init_frac'] +
                 S['pol_records']['init_frac']}
        if len(fracs) != 1 or not isinstance(next(iter(fracs)), (int, float)):
            bad.append(f'init_frac of the {BUDGET}-s records {sorted(fracs, key=str)}; one value '
                       'expected')
        if bad or bad2 or bad3:
            raise Refusal(bad + bad2 + bad3)
        frac = float(next(iter(fracs)))
        set_states = {}
        for name, (per, done) in states.items():
            rk = 'records' if name == 'rule' else 'pol_records'
            set_states[name] = dict(
                cells_total=sum(v['cells'] for v in per.values()),
                cells_valid=sum(v['valid'] for v in per.values()),
                seed_states={str(k): v for k, v in per.items()}, complete_seeds=done,
                seeds_used=used[name], share_min=min(S[rk]['share_min'] for S in R.values()))
        res = dict(
            generator=GENERATOR, run='full' if full else 'preview', dry=bool(dry),
            warning=None if full else (
                f'PREVIEW: seeds {P.compress(used["rule"])} of {TOKEN} and seeds '
                f'{P.compress(used["policy"])} of {POL_TOKEN}, the best of the seeds of each set '
                'on both budgets of its start; not for the paper'),
            setting=P124.setting_text(), selected_run=sel, cpsat_configuration=cpsat_arm,
            origin='request', read_at_s=READ_AT, budget_s=BUDGET, ref_budget_s=REF_BUDGET,
            init_frac=frac, init_phase_s=frac * BUDGET, ref_init_phase_s=frac * REF_BUDGET,
            boot=pa.boot, alpha=pa.alpha, seeds=list(used['rule']),
            pol_seeds=list(used['policy']), ref_seeds=list(pa.seeds),
            instances=list(pa.inst), sizes=list(pa.sizes), combinations=list(pa.combos),
            slots=list(a.slots), pol_slots=list(a.pol_slots), ref_slots=list(pa.slots),
            set_states=set_states,
            cells_total=sum(s['cells_total'] for s in set_states.values()),
            cells_valid=sum(s['cells_valid'] for s in set_states.values()),
            record_names=(f'or_solution/FAMILY/<size>_memetic{BUDGET}s/<R>/instance_<i>_seed<k>'
                          '.json (timed_campaign_plan.Campaign.record, --memetic-seed k)'),
            pol_record_names=(f'or_solution/FAMILY/<size>_memetic{BUDGET}s_pactwarm_s{sel}/<R>/'
                              'instance_<i>_seed<k>.json (timed_campaign_plan.Campaign.record, '
                              f'--memetic-seed k, --policy-seed {sel})'),
            p124_record=where, rules=list(RULES),
            statistics=('p94.compare_at (p94.stat_instance: stats_family.instance_boot, fresh '
                        'default_rng(7) per quantity; stats_family.direction), p94.mark_held, '
                        'p94.verdict_at (p94.verdict_family: instance_boot draws, '
                        'stats_family.boot_p_one_sided, stats_family.holm, '
                        'p94.selected_verdict), p94.lowest'),
            comparisons={k: t for k, _, t, _ in COMPARISONS}, p124_comparison=P124_TEXT,
            rejected=[x for S in inp['sets'].values() for x in S['rejected']],
            missing=[x for S in inp['sets'].values() for x in S['missing']], arrays=inp['files'],
            results=R, decision=dict(rows=rows, counts=row_counts(rows), all_rows=a12,
                                     entries=list(TABLE_ENTRIES)))
        if READ_AT != DEFAULT_BUDGET:
            # the counts over the twelve rows of both sizes are the reading at 60 s and p138's
            del res['decision']['all_rows']
        lines = None
        if full:
            lines = macro_lines(res)
            names = check_names(lines, paths['macros'])
        if lines is not None:
            emit(paths['macros'], lines)
        os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
        with open(paths['json'], 'w') as f:
            json.dump(T.jsonable(res), f, indent=1)
        os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
        with open(paths['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        remove_outputs(paths)
        raise
    return res, names, paths


# ----------------------------------------------------------------- dry mode
DRY_CONFIG = {'init_frac': 0.1, 'pop_size': 20}
DRY_CODE = 'synthetic'
# DRY_LATE_S (search time of a step of the runs started from the policy's schedule that the
# request clock hides at the reading time with o_pol and would show with o_rule: 54 s at 60 s,
# 294 s at 300 s) and DRY_POL_SLOT (core set of each seed of the runs started from the policy's
# schedule) are set by configure


def dry_m60(size, i, j, k):
    """Incumbents of seed k of a run from the rule schedule on the search clock. Budget 60 s: the
    rule schedule at the start, r0 - 13 - k after 20 s; at the first size with setups also
    r0 - 50 - k after 40 s. Budget 300 s: r0 at the start, r0 - 13 - k after 20 s, r0 - 18 - k
    after 150 s (later than any reading at 60 s); at the second size also r0 - 60 - k after
    200 s."""
    R = P124.r0(i, j)
    pts = [(0.0, R), (20.0, R - 13 - k)]
    if BUDGET != DEFAULT_BUDGET:
        pts.append((150.0, R - 18 - k))
        if size == SIZES[1]:
            pts.append((200.0, R - 60 - k))
        return pts
    if size == SIZES[0] and COMBOS[j] in T.GROUPS[1][2]:
        pts.append((40.0, R - 50 - k))
    return pts


def dry_mp60(size, i, j, k):
    """Incumbents of seed k of a run from the policy's schedule on the search clock. Budget 60 s:
    the greedy schedule g of the selected run at the start, g - 12 - k after 20 s; at the second
    size with setups also g - 45 - k after 30 s. Budget 300 s: g at the start, g - 12 - k after
    20 s, g - 14 - k after 100 s; at the first size with setups also g - 40 - k after 120 s. Both:
    g - 100 - k after DRY_LATE_S, which the request clock reaches only after the reading
    time."""
    g = P124.dry_policy(P124.DRY_SEL, i, j)
    pts = [(0.0, g), (20.0, g - 12 - k)]
    if BUDGET != DEFAULT_BUDGET:
        pts.append((100.0, g - 14 - k))
        if size == SIZES[0] and COMBOS[j] in T.GROUPS[1][2]:
            pts.append((120.0, g - 40 - k))
    elif size == SIZES[1] and COMBOS[j] in T.GROUPS[1][2]:
        pts.append((30.0, g - 45 - k))
    pts.append((DRY_LATE_S, g - 100 - k))
    return pts


def dry_namespace(tree, slots):
    """The options the planner's campaigns of the dry tree read (p124.build_world's)."""
    return argparse.Namespace(
        result_root=os.path.join(tree, 'rr'), max_invalid=2, data_root=os.path.join(tree, 'data'),
        slots=list(slots), combos=list(COMBOS), policy_batch=P124.P97.DEFAULT_BATCH,
        policy_workers=P124.P97.DEFAULT_WORKERS, policy_base_seed=P124.P97.DEFAULT_BASE_SEED,
        bars=os.path.join(tree, 'bars.json'))


def dry_share(name, k):
    """cpu_share of the dry records of seed k: the smallest at seed 8 (rule start) or at seed 1
    (policy start)."""
    return round(0.999 - 0.001 * k, 4) if name == 'rule' else round(0.989 + 0.001 * k, 4)


def build_m60(tree, seeds, pol_seeds, ref=True):
    """The records of the dry tree of the chosen budget under the planner's record names: from
    the rule schedule on the cores of SLOTS[0], from the policy's schedule on the cores of
    DRY_POL_SLOT[k], combination j on core j of the set; and, with ref, the configuration and code
    hash in the 600-s records of p124's tree."""
    spec = INSTANCES
    for name, ss, token, pts_of in (('rule', seeds, TOKEN, dry_m60),
                                    ('policy', pol_seeds, POL_TOKEN, dry_mp60)):
        ns = dry_namespace(tree, SLOTS if name == 'rule' else POL_SLOTS)
        for k in ss:
            cores = P.parse_cores(SLOTS[0] if name == 'rule' else DRY_POL_SLOT[k])
            camp = T.make_campaign(list(SIZES), [f'{token}:{spec}'], P124.p94_args(ns, k),
                                   P124.DRY_SEL, P124.DRY_CPSAT)
            for seg in camp.segments:
                for c in camp.cells(seg):
                    j = COMBOS.index(c.combo)
                    pts = pts_of(c.size, c.idx, j, k)
                    starts = [dict(source='warm_cache', makespan=P124.r0(c.idx, j))]
                    if name == 'policy':
                        starts.append(dict(source='seed_from',
                                           makespan=P124.dry_policy(P124.DRY_SEL, c.idx, j)))
                    rec = dict(camp.expected_fields(c), validated=True, threads=1, workers=1,
                               cpu_share=dry_share(name, k), affinity=[cores[j % len(cores)]],
                               search_start_wall_s=1.0, search_start_cpu_s=1.0, seeds=starts,
                               makespan=pts[-1][1],
                               anytime=[[t, t + 1.0, t + 1.0, v] for t, v in pts],
                               config=dict(DRY_CONFIG), code_sha256_16=DRY_CODE)
                    P124._dump(camp.record(c), rec)
    if not ref:
        return
    ns6 = dry_namespace(tree, P124.SLOTS)
    for arm in (ARM, POL_ARM):
        for k in P124.SEEDS:
            camp = T.make_campaign(list(SIZES), [f'{arm}@{REF_BUDGET}:{spec}'],
                                   P124.p94_args(ns6, k), P124.DRY_SEL, P124.DRY_CPSAT)
            for seg in camp.segments:
                for c in camp.cells(seg):
                    p = camp.record(c)
                    rec = load(p)
                    rec.update(config=dict(DRY_CONFIG), code_sha256_16=DRY_CODE)
                    P124._dump(p, rec)


def path60(result_root, size, combo, idx, seed, policy=False):
    """The planner's record name of a cell of the runs of the chosen budget (Campaign.record),
    from the rule schedule or, with policy, from the policy's schedule of the dry tree's selected
    run."""
    ns = dry_namespace(os.path.dirname(result_root), POL_SLOTS if policy else SLOTS)
    ns.result_root = result_root
    camp = T.make_campaign([size], [f'{POL_TOKEN if policy else TOKEN}:{idx}'],
                           P124.p94_args(ns, seed), P124.DRY_SEL, P124.DRY_CPSAT)
    return next(camp.record(c) for seg in camp.segments for c in camp.cells(seg)
                if c.combo == combo)


def dry_expected(size, i, j, seeds=SEEDS, pol_seeds=SEEDS):
    """The values at the reading time of the dry tree: the best of the runs of the chosen budget
    of seeds (best60) and of the 600-s runs of the same seeds (best600) from the rule schedule, the
    same from the policy's schedule over pol_seeds (read at t - max(o_pol, o_rule)), the runs of
    seed 1, and p124's greedy, sampled and CP-SAT values."""
    t = float(READ_AT)
    run = t - P124.DRY_O_RULE - 1.0          # search time of a run from the rule schedule
    run_p = t - max(P124.DRY_O_POL, P124.DRY_O_RULE) - 1.0   # from the policy's schedule

    def at(pts, r):
        return min(v for s, v in pts if s <= r + 1e-9)
    e = P124.dry_expected(i, j, t)
    return dict(best60=min(at(dry_m60(size, i, j, k), run) for k in seeds),
                best600=min(at(P124.dry_memetic(i, j, k), run) for k in seeds),
                one60=at(dry_m60(size, i, j, SINGLE), run),
                one600=at(P124.dry_memetic(i, j, SINGLE), run),
                bestp60=min(at(dry_mp60(size, i, j, k), run_p) for k in pol_seeds),
                bestp600=min(at(P124.dry_memetic_policy(i, j, k), run_p) for k in pol_seeds),
                greedy=e['greedy'], sampled=e['sampled'], cpsat=e['cpsat'], best8=e['best'],
                bestp8=e['bestp'])


# DRY_PAIRS (comparison key, first and second value of dry_expected) is set by configure


def dry_verify(res):
    """The dry run's numbers against the values built into the tree."""
    bad = []
    inst = P.parse_instances(INSTANCES)
    seeds, pseeds = res['seeds'], res['pol_seeds']
    for size, S in res['results'].items():
        for key, first, second in DRY_PAIRS:
            rows = S['comparisons'][key]
            for j, r in enumerate(COMBOS):
                e = [dry_expected(size, i, j, seeds, pseeds) for i in inst]
                want = np.mean([100 * (x[first] - x[second]) / x[second] for x in e])
                if abs(rows[r]['mean'] - want) > 1e-9:
                    bad.append(f'{size} {r} {key}: {rows[r]["mean"]}, built {want}')
        for sk, value, ss in (('spread', 'best60', seeds), ('pol_spread', 'bestp60', pseeds)):
            for c in S[sk]['cells']:
                j = COMBOS.index(c['combination'])
                pts = [dry_expected(size, c['instance'], j, [k], [k])[value] for k in ss]
                if (c['smallest'], c['largest']) != (min(pts), max(pts)):
                    bad.append(f'{size} {c["combination"]}/{c["instance"]}: {sk} {c}, built '
                               f'{(min(pts), max(pts))}')
        for rk, name, ss in (('records', 'rule', seeds), ('pol_records', 'policy', pseeds)):
            want_share = min(dry_share(name, k) for k in ss)
            if abs(S[rk]['share_min'] - want_share) > 1e-12:
                bad.append(f'{size}: smallest cpu_share of {name} {S[rk]["share_min"]}, built '
                           f'{want_share}')
    return bad


def dry_run(tree, boot=DRY_BOOT, seeds=SEEDS, pol_seeds=SEEDS, min_seeds=None, drop=None,
            pol_drop=None, before=None, check_only=False, record=None, build=True):
    """The whole path of the chosen budget on a synthetic tree; returns (res, names, paths,
    options), or with check_only (rc of --check, None, None, options). build: p124's tree and the
    records of both budgets (BUDGETS) side by side (False: the tree of an earlier dry run, read
    again). seeds, pol_seeds: the seeds of the records of each start in the tree; drop, pol_drop:
    (seed, size, combination, instance) of the records of the chosen budget of each start removed
    before the run; before: a function of the record root called after the tree is built;
    record: a function that changes p124's record of the tree (a dict) before the run; min_seeds:
    a preview into <tree>/preview."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refusal([f'the dry-run folder {tree} lies inside the repository'])
    if build:
        P124.dry_run(tree, boot=boot)
    with P124.dry_setting(tree):
        if build:
            for b in BUDGETS:
                with at_budget(b):
                    build_m60(tree, seeds, pol_seeds, ref=b == BUDGETS[0])
        ns = dry_namespace(tree, SLOTS)
        for policy, dropped in ((False, drop), (True, pol_drop)):
            for k, size, r, i in (dropped or ()):
                os.remove(path60(ns.result_root, size, r, i, k, policy))
        if before is not None:
            before(ns.result_root)
        if record is not None:
            p = REG.out(P124_JSON)
            J = load(p)
            record(J)
            P124._dump(p, J)
        argv = ['--result-root', ns.result_root, '--data-root', ns.data_root, '--bars', ns.bars,
                '--boot', str(boot)]
        if min_seeds is not None:
            argv += ['--min-seeds', str(min_seeds), '--out', os.path.join(tree, 'preview')]
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


# the counts the full dry run builds, per budget: (counts of the four rows, counts of the twelve
# rows or None). At 60 s: the lowest column changes in two rows with both 60-s values (one with the
# 60-s value from the rule schedule alone), the 60-s value from the rule schedule below the row's
# lowest in one; every one of the twelve rows uses the policy before (CP-SAT started from the
# policy's schedule) and eleven after (the row at 50x25n30 with setups goes to the 60-s runs from
# the rule schedule). At 300 s: the row at 50x25n30 with setups goes to the 300-s runs from the
# policy's schedule and both rows at 80x25n30 to the 300-s runs from the rule schedule, so the
# lowest column changes in three rows (two with the rule start alone) and two of the four use the
# policy after; the twelve rows are not counted.
DRY_ROWS = {
    60: (dict(rows=4, changed=2, changed_rule_only=1, below_lowest=1, below_lowest_policy=1,
              pol_smp_before=0, pol_smp_after=0, pol_smp_after_rule_only=0,
              uses_policy_before=4, uses_policy_after=3),
         dict(rows=12, uses_policy_before=12, uses_policy_after=11)),
    300: (dict(rows=4, changed=3, changed_rule_only=2, below_lowest=2, below_lowest_policy=1,
               pol_smp_before=0, pol_smp_after=0, pol_smp_after_rule_only=0,
               uses_policy_before=4, uses_policy_after=2),
          None)}
configure(DEFAULT_BUDGET)


# ----------------------------------------------------------------- command line
def build_parser():
    """The options of the reading of the chosen budget (configure): --budget defaults to it, and
    --slots and --pol-slots to the core sets of its two sets."""
    ap = argparse.ArgumentParser(description='the memetic algorithm with a budget of 60 s or 300 '
                                             's, read that many seconds after the request')
    ap.add_argument('--budget', type=int, choices=BUDGETS, default=BUDGET,
                    help='budget of the runs of both sets and reading time since the request, '
                         'seconds (default 60; 300: the five-minute reading)')
    ap.add_argument('--check', action='store_true', help='read only: list every missing input')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on a synthetic tree in DIR (default: a temporary folder)')
    ap.add_argument('--min-seeds', type=int, default=None,
                    help='a preview once each set has n seeds whose every cell is valid, on every '
                         'complete seed of each set; not for the paper, no macro file, needs --out')
    ap.add_argument('--out', default=None,
                    help='root of manuscript/ and results/ (default PACT_OUT through '
                         'arm_registry.out); outside the repository\'s paper, result and record '
                         'folders and outside PACT_OUT')
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--slots', nargs='+', default=list(SLOTS),
                    help='core sets of the runs of the budget started from the rule schedule '
                         '(default 16-23)')
    ap.add_argument('--pol-slots', nargs='+', default=list(POL_SLOTS),
                    help="core sets of the runs of the budget started from the policy's schedule "
                         '(default 16-23 8-15 at 60 s, 8-15 at 300 s)')
    ap.add_argument('--ref-slots', nargs='+', default=list(P124.SLOTS),
                    help="core sets of the 600-s runs, p124's --slots (default 16-23 8-15)")
    ap.add_argument('--bars', default=None)
    ap.add_argument('--max-invalid', type=int, default=2)
    ap.add_argument('--result-root', default='')
    ap.add_argument('--data-root', default='data/FAMILY')
    ap.add_argument('--p124-json', default=None, help=f"p124's record (default {P124_JSON} "
                    'through arm_registry.out)')
    return ap


def main(argv=None):
    argv = _ARGV[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--budget', type=int, choices=BUDGETS, default=DEFAULT_BUDGET)
    configure(pre.parse_known_args(argv)[0].budget)
    ap = build_parser()
    a = ap.parse_args(argv)
    for k in ('out', 'p124_json'):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(os.path.join(_CWD, getattr(a, k))))
    os.chdir(ROOT)
    if a.dry is not None:
        if a.check or a.min_seeds is not None:
            ap.error('--dry builds its own records; --check and --min-seeds read the real ones')
        keep = bool(a.dry)
        tree = os.path.abspath(os.path.join(_CWD, a.dry)) if keep else \
            tempfile.mkdtemp(prefix='p137_dry_')
        try:
            boot = DRY_BOOT if a.boot == SF.B_DEFAULT else a.boot
            res, names, paths, _ = dry_run(tree, boot)
            bad = dry_verify(res)
            c, a12 = res['decision']['counts'], res['decision'].get('all_rows')
            if c != DRY_COUNTS:
                bad.append(f'decision rows {c}, built {DRY_COUNTS}')
            if (a12 is None) != (DRY_ALL_ROWS is None) or a12 is not None and \
                    {k: a12[k] for k in DRY_ALL_ROWS} != DRY_ALL_ROWS:
                bad.append(f'twelve rows {a12}, built {DRY_ALL_ROWS}')
            if bad:
                raise Refusal(bad)
        except Refusal as e:
            sys.exit('p137 self-test on synthetic records FAILED: ' + '; '.join(e.args[0][:6]))
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
        x = res['results'][SIZES[0]]['comparisons'][KEY['m_ref']]['Pool']
        y = res['results'][SIZES[0]]['comparisons'][KEY['mp_ref']]['Pool']
        print(f'p137 self-test on synthetic records passed: {len(res["seeds"])} seeds of each '
              f'start, {len(SIZES)} sizes, B = {res["boot"]}, {len(names)} macros; {SIZES[0]} '
              f'pooled at {READ_AT} s: best of the {BUDGET}-s runs minus best of the 600-s runs '
              f'{T.f2(x["mean"])}% [{T.f2(x["lo"])}, {T.f2(x["hi"])}] from the rule schedule, '
              f'{T.f2(y["mean"])}% [{T.f2(y["lo"])}, {T.f2(y["hi"])}] from the policy\'s '
              f'schedule; decision rows {c["rows"]}, lowest column changes in {c["changed"]} '
              f'({c["changed_rule_only"]} with the rule start alone); '
              + (f'twelve rows using the policy {a12["uses_policy_before"]} before, '
                 f'{a12["uses_policy_after"]} after; ' if a12 is not None else '')
              + 'every value equal to the built one'
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
    c, a12 = res['decision']['counts'], res['decision'].get('all_rows')
    st = res['set_states']
    print(f'p137: {res["run"]}: seeds {P.compress(res["seeds"])} of {TOKEN} '
          f'({st["rule"]["cells_valid"]} of {st["rule"]["cells_total"]} records valid), seeds '
          f'{P.compress(res["pol_seeds"])} of '
          f'{POL_TOKEN} ({st["policy"]["cells_valid"]} of {st["policy"]["cells_total"]}); decision '
          f'rows {c["rows"]}, lowest column changes in {c["changed"]}'
          + (f'; twelve rows using the policy {a12["uses_policy_before"]} before, '
             f'{a12["uses_policy_after"]} after' if a12 is not None else '')
          + (f'; {len(names)} macros' if names else '; no macro file')
          + '; wrote ' + ', '.join(shown(p) for p in paths.values() if os.path.exists(p)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
