r"""P120: a lower-bound reference for the makespans of every method (exact environment, factory
benchmark), from stored results only.

BOUNDS (per test set, rule combination c and test instance)
  (i)   chain   the makespan estimate at the empty schedule C^(s_0;c) of the environment:
                p113_instance_bound.estimate (FJSPEnvFamily.set_initial_data, init_quality on the
                normalized axis divided by the slope), with the reward settings and the mask
                options of the selected policy run's configuration (p113_instance_bound.run_config;
                its envelope must resolve to 'min'). It follows each job chain alone.
  (ii)  load    the machine-load bound of machine_load_bound(): for a machine set M' the
                operations whose eligible machines all lie in M' give
                  LB(M') = min_o h_o + ceil(sum_o d_o / |M'|) + min_o q_o,
                maximized over the station types (and over the distinct eligible sets when these
                do not partition the operations). The ceiling is taken only when every time of
                the instance and the descriptor is an integer. The function's docstring holds the
                derivation. As a check of the reading, max_o (h_o + d_o + q_o) must equal bound (i).
  (iii) stored  the largest field 'bound' of the stored CP-SAT records of the full model,
                <solver root>/<set>[_<arm>]/<c>/instance_NNN.json. A record counts when
                  - its fields 'instance' and 'regime' (and 'size', when present) name the file's
                    instance and combination;
                  - its instance is today's: for a record of 50x25 or 80x25 the descriptor of
                    data/FAMILY/<set> and the two base-instance files equal those of the 30-instance
                    set (ALIAS; instance_000 to instance_019 are copies);
                  - its schedule (assigned_mch, op_st, op_ct), when stored, passes
                    constraint_family.reference_sim.validate_schedule on today's instance and
                    descriptor and its largest completion equals 'objective'; a record without a
                    schedule names a warm start (warm_rule, warm_makespan, warm_all_rules) that is
                    found in its cache (CACHE_OF) with the same rule and makespans and whose
                    schedule passes the same validation with the stored makespan; a record with
                    neither is not verifiable and is left out;
                  - its file was written on or after the date of --stored-since (default
                    2026-07-08, file modification time, local time). The window horizon of the
                    constraint model (constraint_family/cpsat_family.py) was enlarged on
                    2026-07-07; every other constraint of the model has been the same since,
                    for descriptors without release dates. Records written before the date are
                    left out and counted; their validated schedules remain feasible makespans;
                  - its descriptor carries neither release dates nor a per-machine setup
                    multiplier, which the model of the stored records does not describe
                    (model_mismatch; none of the five test sets carries either);
                  - its bound does not exceed the best known makespan of the cell (below). A
                    larger bound is excluded, counted and listed (up to ten examples).
                The large-neighborhood configuration of the timed comparison stores no bound.
                Per test set the JSON and the Markdown list the folders whose bounds are used,
                with their budgets (field budget_s), warm starts and modification times.
  The reference bound is the largest of the three.

FEASIBLE MAKESPANS (the self-check and the best known makespan of a cell)
  - every Result_<tag>_<c>.npy of the exact environment's folders of the set, read through
    arm_registry.res_dir (10x25 also the folder of the off-combination evaluation; the
    30-instance sets also the 20-instance set's folder for the copied instances), column 0, one
    row per descriptor: the policy, the specialists, the single-rule control, the dispatching
    rules and every further trained network stored there;
  - the records of the timed comparison and of its start schedules with validated true and no
    violations: <set>_memetic*, <set>_rulestart, <set>_polsample* under or_solution/FAMILY and
    <set>_cparms/**/<c>/ under test_results/FAMILY;
  - the stored solver schedules and warm starts that pass validate_schedule above.
  SELF-CHECK: bounds (i) and (ii) never exceed any of these makespans of the same cell (relative
  tolerance TOL). A violation is a fault of a bound or of the reading: the script refuses and
  writes nothing, printing the first cases.

GAPS (per test set and combination, and pooled over the eight)
  d = 100 (C - LB) / LB per instance, LB the reference bound; the mean over the instances with
  the instance-level cluster bootstrap stats_family.instance_boot (p88's estimator for one
  checkpoint and for the seed-mean makespans, fresh default_rng(p88.INST_SEED) per quantity,
  B = --boot, alpha 0.05; a pooled row keeps the eight combinations of an instance together and
  uses the instances present in all eight). Methods:
    policy       greedy schedule of the training run with the lowest validation score
                 (arm_registry.select_seed('released'))
    policy_mean  the greedy makespans averaged over the training seeds (PACT_SEEDS)
    specialist   the specialist of the combination, makespans averaged over the training seeds
                 (stored at 10x25, 20x25 and 30x25)
    best_rule    per instance the shortest of FIFO, MOR, SPT and MWKR
    memetic      the memetic algorithm started from the best dispatching rule, 600 s
    solver       CP-SAT, the configuration the pilot selected, started from the best dispatching
                 rule, 600 s
  The last two are the cells of the timed comparison that timed_campaign_plan.cell_states calls
  valid, read through p94_timed_inference (make_campaign, read_cells), the record's makespan at
  the end of its budget. The cells are those of the comparison as it was launched: one
  --campaign "<sizes> = <tokens>" per launch, in the planner's token syntax
  name[@budget[:instances]], parsed by p94_timed_inference.build_parser and parse_campaign and
  turned into the planner's cells by p94_timed_inference.make_campaign. The tokens
  memetic_rule@600 and cpsat_rule@600 (with their instances, or each size's evidence instances
  when none are given) define the expected cells; the other tokens of a launch (the
  policy-started searches, policy_sample) are read past. A test set without such a token has no
  column for that search.
  Also: the share of instances on which each bound is the largest (ties counted for every bound
  that attains the reference), the mean of 100 bound / best known makespan for each bound, and the
  policy's gap on the instances of each search (the same cells as the search's mean).

TEST SETS  10x25 (100 instances), 20x25 (50), 30x25 (20), 50x25n30 and 80x25n30 (evidence
           instances 3 to 29).

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_lbref.tex           macro_io.emit; names Lbr<quantity>, letters only, checked
                                       against every macro defined under manuscript/
  results/lower_bound_reference.json   every cell with its bounds, makespans and sources
  results/lower_bound_reference.md     tables
  manuscript/tables/lbref_table.tex     tabular (booktabs) of the gaps: the training size per
                                       combination and pooled, the other test sets pooled; the
                                       reference bound in % of the best known makespan and the
                                       cells proved optimal (best known makespan equal to the
                                       reference within the self-check tolerance); no bold
  --partial writes the four files into --out DIR (required), never to the paths above; the first
  line of each says that it is partial. A refusal removes them.

CHECK (--check, read only): exit 0 when every input group is complete: the descriptors, the
  arrays of the policy (every seed, eight combinations), of the specialists (own combination, at
  10x25, 20x25, 30x25) and of the four dispatching rules with one row per descriptor, the
  selected run's configuration, and every cell of the launched 600 s tokens of both
  rule-started searches valid. Every mode except --dry refuses without a --campaign.
DRY MODE (--dry [DIR]): the whole path on synthetic arrays, records and timed cells in DIR
  (default a temporary folder, removed afterwards), with the real descriptors of two test sets
  and the campaign description DRY_CAMPAIGNS through the same parser.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p120_lower_bound_reference.py \
           --campaign "50x25n30 80x25n30 = memetic_rule@600 cpsat_rule@600" \
           --campaign "10x25 = memetic_rule@600 cpsat_rule@600:0-9"
       ... python scripts/p120_lower_bound_reference.py --check --campaign ... --campaign ...
       ... python scripts/p120_lower_bound_reference.py --partial --out DIR --campaign ...
       python scripts/p120_lower_bound_reference.py --dry [DIR]
"""
import argparse
import contextlib
import datetime
import glob
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402  (removes the --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
import numpy as np  # noqa: E402

import p113_instance_bound as IB  # noqa: E402  (sets sys.argv for params.py)
import params  # noqa: E402,F401  (parsed now, with the device option p113 set)
import p88_family_inference as F88  # noqa: E402
import stats_family as SF  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from constraint_family.fjsp_env_family import resolve_delta_mode  # noqa: E402
from constraint_family.reference_sim import simulate, validate_schedule  # noqa: E402
from macro_io import emit  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

GENERATOR = 'scripts/p120_lower_bound_reference.py'
PREFIX = 'Lbr'
OUT_FILES = {'macros': 'manuscript/macros_lbref.tex',
             'json': 'results/lower_bound_reference.json',
             'md': 'results/lower_bound_reference.md',
             'table': 'manuscript/tables/lbref_table.tex'}
ENV = 'exact'
R8 = tuple(REG.COMBOS)
DATA_ROOT = 'data/FAMILY'
# test set -> (macro word, index of the first evidence instance)
SETS = {'10x25': ('Ten', 0), '20x25': ('Twenty', 0), '30x25': ('Thirty', 0),
        '50x25n30': ('Fifty', 3), '80x25n30': ('Eighty', 3)}
SPECIALIST_SETS = ('10x25', '20x25', '30x25')
# planner tokens of the two rule-started searches -> method of this analysis
TIMED_ARMS = {'memetic_rule': 'memetic', 'cpsat_rule': 'solver'}
BUDGET = 600
# the description of the timed comparison as it was launched (printed when --campaign is absent)
AS_LAUNCHED = ('50x25n30 80x25n30 = memetic_rule@600 cpsat_rule@600',
               '10x25 = memetic_rule@600 cpsat_rule@600:0-9')
# a set whose instance_000 to instance_019 are byte copies of these instances of the target
ALIAS = {'50x25': '50x25n30', '80x25': '80x25n30'}
# further exact-environment folders whose rows are instances of a test set
EXTRA_FOLDERS = {'10x25': (REG.CROSS_FOLDER,), '50x25n30': ('50x25',), '80x25n30': ('80x25',)}
# folder of a stored solver record -> folder of its warm-start cache (first match)
CACHE_OF = ((re.compile(r'^10x25_fairwarmR\d+s$'), 'warmstart_cache_repaired'),
            (re.compile(r'^10x25_fairwarm\d+s$'), 'warmstart_cache'),
            (re.compile(r'^(?P<s>\d+x25)_fairwarm\d+s_pactwarm$'), 'pactwarm_cache_{s}'),
            (re.compile(r'^(?P<s>\d+x25)_fairwarm\d+s_pactbest3$'), 'pactbest3_cache_{s}'),
            (re.compile(r'^(?P<s>\d+x25)_(?:fairwarm|probe)\d+s(?:_rep2|_repB)?$'),
             'warmstart_cache_{s}'))
_FOLDER = re.compile(r'^(?P<set>\d+x\d+(?:n30)?)(?:_(?P<arm>.+))?$')
_RECORD = re.compile(r'^(?P<name>instance_\d{3})(?:_seed\d+)?\.json$')
_ARRAY = re.compile(r'^Result_(?P<tag>.+)_(?P<c>' + '|'.join(sorted(R8, key=len, reverse=True))
                    + r')\.npy$')
TOL = 1e-6
ALPHA = 0.05
BOUNDS = ('chain', 'load', 'stored')
BOUND_TEXT = {'chain': 'instance bound (job chains)', 'load': 'machine-load bound',
              'stored': 'stored solver bound'}
METHODS = ('policy', 'policy_mean', 'specialist', 'best_rule', 'memetic', 'solver')
METHOD_TEXT = {'policy': 'policy, run with the lowest validation score',
               'policy_mean': 'policy, mean over the training seeds',
               'specialist': 'specialist of the combination, mean over the training seeds',
               'best_rule': 'best dispatching rule',
               'memetic': f'memetic algorithm, {BUDGET} s',
               'solver': f'CP-SAT, {BUDGET} s'}
METHOD_WORD = {'policy': 'Pol', 'policy_mean': 'PolMean', 'specialist': 'Spec',
               'best_rule': 'Rule', 'memetic': 'Mem', 'solver': 'Solver'}
MAX_EXAMPLES = 10
# Stored bounds are read only from record files modified on or after this date (local time).
# The window horizon of the constraint model (constraint_family/cpsat_family.py) was enlarged on
# 2026-07-07, and records written before 2026-07-08 are left out (--stored-since).
STORED_SINCE = '2026-07-08'


class Refused(Exception):
    """A setting, an input or a value that this script does not accept."""


# ----------------------------------------------------------------- where the inputs are read
def _select_policy():
    seed, rec = REG.select_seed('released')
    return int(seed), {int(s): [float(v[0]), int(v[1])] for s, v in rec.items()}


class Inputs:
    """Where the analysis reads. The defaults are the stored results; the dry mode and the tests
    replace fields."""

    def __init__(self):
        self.sets = dict(SETS)
        self.limit = None                  # analysis instances per set (None: every evidence one)
        self.solver_root = 'or_solution/FAMILY'
        self.record_roots = ('or_solution/FAMILY', 'test_results/FAMILY')
        # the --campaign descriptions of the timed comparison as launched
        self.campaigns = None
        # (parsed campaigns, seed) -> (cells, counts, solver configuration, launched segments)
        self.timed = lambda parsed, seed: read_timed(parsed, seed)
        self.select_policy = _select_policy
        self.config = IB.run_config
        self.dry = False
        self.stored_since = STORED_SINCE  # date (or date and time) of the stored-bound rule


INPUTS = Inputs()


def since_time(text):
    """'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM' (local time) -> POSIX time."""
    for fmt in ('%Y-%m-%d', '%Y-%m-%dT%H:%M'):
        try:
            return datetime.datetime.strptime(str(text), fmt).timestamp()
        except ValueError:
            pass
    raise Refused(f'--stored-since {text!r}: want YYYY-MM-DD or YYYY-MM-DDTHH:MM')


def model_mismatch(desc):
    """A reason why the solver model of the stored records does not describe the descriptor, or
    None: the model of the records carries neither release dates nor a per-machine setup
    multiplier, so its bound is not used for a descriptor that has either."""
    if getattr(desc, 'release', None) is not None:
        return 'release dates'
    if getattr(desc, 'setup_mult', None) is not None:
        return 'a per-machine setup multiplier'
    return None


@contextlib.contextmanager
def inputs(**kv):
    old = {k: getattr(INPUTS, k) for k in kv}
    for k, v in kv.items():
        setattr(INPUTS, k, v)
    try:
        yield INPUTS
    finally:
        for k, v in old.items():
            setattr(INPUTS, k, v)


@contextlib.contextmanager
def environment(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the analysis reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')
    return list(REG.seeds())


# ----------------------------------------------------------------- data
_BASE = {}


def base_instance(stem):
    """(job_length, op_pt, meta) of a base instance, read once."""
    if stem not in _BASE:
        _BASE[stem] = load_instance(stem)
    return _BASE[stem]


def descriptor_names(size):
    """Sorted instance names of data/FAMILY/<size>; every combination must name the same."""
    names = sorted(os.path.basename(p)[:-len('.desc.json')]
                   for p in glob.glob(f'{DATA_ROOT}/{size}/desc/N/*.desc.json'))
    if not names:
        raise Refused(f'no descriptors under {DATA_ROOT}/{size}/desc')
    for c in R8:
        other = sorted(os.path.basename(p)[:-len('.desc.json')]
                       for p in glob.glob(f'{DATA_ROOT}/{size}/desc/{c}/*.desc.json'))
        if other != names:
            raise Refused(f'the descriptors of {size}/{c} do not name the instances of {size}/N')
    return names


def analysis_instances(size):
    """(all descriptor names, [(index, name)] of the analysis instances)."""
    names = descriptor_names(size)
    first = INPUTS.sets[size][1]
    idx = list(range(first, len(names)))
    if INPUTS.limit is not None:
        idx = idx[:INPUTS.limit]
    return names, [(i, names[i]) for i in idx]


def descriptor_record(size, combo, name):
    return json.load(open(f'{DATA_ROOT}/{size}/desc/{combo}/{name}.desc.json'))


_CELL = {}


def cell_data(size, combo, name):
    """(job_length, op_pt, meta, descriptor, descriptor record) of one cell, read once."""
    key = (DATA_ROOT, size, combo, name)
    if key in _CELL:
        return _CELL[key]
    rec = descriptor_record(size, combo, name)
    if rec.get('regime') not in (None, combo):
        raise Refused(f'{size}/{combo}/{name}: the descriptor names the combination '
                      f'{rec.get("regime")}')
    desc = ConstraintDescriptor.from_json(rec, setup_convention='attached')
    jl, pt, meta = base_instance(rec['base_instance'])
    _CELL[key] = (np.asarray(jl, dtype=int), np.asarray(pt, dtype=float), meta, desc, rec)
    return _CELL[key]


def integral_times(pt, desc):
    """True when every processing time, lag, setup, window end and release date is an integer."""
    vals = [np.asarray(pt, dtype=float).ravel(), np.asarray(desc.lag, dtype=float).ravel()]
    if desc.has_setup and desc.setup is not None:
        vals.append(np.asarray(desc.setup, dtype=float).ravel())
    if desc.has_window and desc.windows is not None:
        vals.append(np.asarray([x for w in desc.windows for s in w for x in s], dtype=float))
    if getattr(desc, 'release', None) is not None:
        vals.append(np.asarray(desc.release, dtype=float).ravel())
    v = np.concatenate([x for x in vals if x.size] or [np.zeros(0)])
    return bool(np.all(np.isfinite(v)) and np.all(v == np.round(v)))


def descriptor_faults(desc, n_ops):
    """Fields the bounds rely on: lags all zero when the lag rule is inactive (the environment
    reads the lag vector in every combination), and the vector lengths."""
    out = []
    lag = np.asarray(desc.lag, dtype=float)
    if lag.shape != (n_ops,):
        out.append(f'lag has shape {lag.shape}, want ({n_ops},)')
    elif not desc.has_lag and np.any(lag != 0):
        out.append('nonzero lags with the lag rule inactive')
    if np.asarray(desc.op_class).shape != (n_ops,):
        out.append('op_class has the wrong length')
    return out


# ----------------------------------------------------------------- bound (i)
@contextlib.contextmanager
def _p113_data(size):
    old = IB.DATA
    IB.DATA = f'{DATA_ROOT}/{size}'
    try:
        yield
    finally:
        IB.DATA = old


def instance_bound(size, combo, name, cfg):
    """Bound (i): p113_instance_bound.estimate on the descriptors of data/FAMILY/<size>."""
    with _p113_data(size):
        return float(IB.estimate(name, combo, cfg))


# ----------------------------------------------------------------- bound (ii)
def shortest_blocks(op_pt, desc):
    """d_o = min over the eligible machines k of p_ok + u_k delta_o, where delta_o is the smallest
    setup into the class of o from any source class, the cold-start row included (zero when the
    setup rule is inactive) and u_k the per-machine setup multiplier (one when absent)."""
    pt = np.asarray(op_pt, dtype=float)
    N, M = pt.shape
    delta = np.zeros(N)
    if desc.has_setup and desc.setup is not None:
        delta = np.asarray(desc.setup, dtype=float).min(axis=0)[np.asarray(desc.op_class)]
    u = getattr(desc, 'setup_mult', None)
    u = np.ones(M) if u is None else np.asarray(u, dtype=float)
    cand = np.where(pt > 0, pt + u[None, :] * delta[:, None], np.inf)
    d = cand.min(axis=1)
    if not np.all(np.isfinite(d)):
        raise Refused('an operation has no eligible machine')
    return d


def heads_tails(job_length, block, lag):
    """h_o: the sum over the predecessors i of o in its job of (d_i + lag_i), the earliest start
    of the block of o; q_o: the sum over the successors j of o of (lag of the operation before j
    + d_j), the shortest time from the completion of o to the completion of its job."""
    block = np.asarray(block, dtype=float)
    lag = np.asarray(lag, dtype=float)
    h, q = np.zeros(len(block)), np.zeros(len(block))
    first = 0
    for L in np.asarray(job_length, dtype=int):
        if L <= 0:
            continue
        b, g = block[first:first + L], lag[first:first + L]
        h[first:first + L] = np.concatenate([[0.0], np.cumsum(b[:-1] + g[:-1])])
        q[first:first + L] = np.concatenate([np.cumsum((g[:-1] + b[1:])[::-1])[::-1], [0.0]])
        first += L
    if first != len(block):
        raise Refused(f'job lengths sum to {first}, the instance has {len(block)} operations')
    return h, q


def station_structure(op_pt, mch_type):
    """Whether each operation's eligible set lies in one station type and is the whole type."""
    el = np.asarray(op_pt) > 0
    mt = np.asarray(mch_type)
    one = full = 0
    for o in range(el.shape[0]):
        types = set(mt[el[o]].tolist())
        if len(types) == 1:
            one += 1
            if np.array_equal(el[o], mt == next(iter(types))):
                full += 1
    n = int(el.shape[0])
    return dict(ops=n, one_type=one, whole_type=full, partition=bool(full == n),
                types=sorted(set(mt.tolist())),
                machines_per_type={int(t): int((mt == t).sum()) for t in sorted(set(mt.tolist()))})


def machine_load_bound(job_length, op_pt, desc, mch_type, structure=None):
    r"""Bound (ii), the machine-load bound, in hours, with the parts it is built from.

    For a set M' of machines let O(M') be the operations whose eligible machines all lie in M'.
    Every operation of O(M') is processed on a machine of M', because no other machine is
    eligible for it. Under the attached setup convention the block of an operation o, its setup
    followed by its processing, occupies one machine without interruption. The setup charged to o
    is at least the smallest entry of the setup matrix into the class of o over all source
    classes, the cold-start row included, whatever the machine processed before, so the block of
    o lasts at least d_o = min_k (p_ok + delta_o) (shortest_blocks). The block of o starts no
    earlier than h_o (heads_tails), because the setup of o starts only when its job is ready, each
    predecessor of o in its job occupies a machine for at least its own shortest block, and the
    active lag after each predecessor must elapse before the next operation is ready. After o
    completes, its job needs at least q_o more: the lag after o and the shortest blocks of the
    successors with the lags between them. Machine unavailability windows only add idle time and
    never shorten a block or advance a start, so ignoring them keeps every term a lower bound.
    In a feasible schedule let t0 be the earliest block start and t1 the latest completion among
    the operations of O(M'). Their blocks lie in [t0, t1] on the |M'| machines without overlap,
    so |M'| (t1 - t0) >= sum_o d_o, and t0 >= min_o h_o. The operation that completes at t1 still
    needs at least min_o q_o, so the makespan is at least
        LB(M') = min_o h_o + sum_o d_o / |M'| + min_o q_o.
    When every processing time, lag, setup and window end is an integer, the blocks of a
    semi-active schedule start at a job-ready time, a machine-free time or the end of a window,
    so some optimal schedule has integer times and the optimal makespan is an integer. The terms
    min_o h_o and min_o q_o are then integers, and the middle term may be rounded up.
    The bound is the largest LB(M') over the machine sets of the station types and, when the
    eligible sets do not partition the operations into whole station types, over the distinct
    eligible sets as well.

    Returns dict(value, group, kind, integral, chain, groups): the bound; the station type or
    eligible set that attains it; 'type' or 'eligible set'; whether the ceiling was used;
    max_o (h_o + d_o + q_o), the longest job chain of shortest blocks (bound (i) under the 'min'
    envelope); and LB(M') of every machine set evaluated. structure is station_structure() of
    the instance when it is already known."""
    pt = np.asarray(op_pt, dtype=float)
    N, M = pt.shape
    el = pt > 0
    lag = np.asarray(desc.lag, dtype=float) if desc.has_lag else np.zeros(N)
    d = shortest_blocks(pt, desc)
    h, q = heads_tails(job_length, d, lag)
    integral = integral_times(pt, desc)
    mt = np.asarray(mch_type)
    groups = [('type', int(t), frozenset(np.where(mt == t)[0].tolist()))
              for t in sorted(set(mt.tolist()))]
    if not (structure or station_structure(pt, mt))['partition']:
        seen = {g for _, _, g in groups}
        for o in range(N):
            g = frozenset(np.where(el[o])[0].tolist())
            if g not in seen:
                seen.add(g)
                groups.append(('eligible set', sorted(g), g))
    best, out = None, []
    for kind, label, g in groups:
        inside = np.array(sorted(g), dtype=int)
        outside = np.setdiff1d(np.arange(M), inside)
        ops = np.where(el[:, inside].any(axis=1) & ~el[:, outside].any(axis=1))[0] \
            if len(outside) else np.arange(N)
        if not len(ops):
            continue
        load = float(d[ops].sum()) / len(g)
        mid = math.ceil(load - 1e-9) if integral else load
        val = float(h[ops].min() + mid + q[ops].min())
        out.append(dict(kind=kind, group=label, machines=len(g), operations=int(len(ops)),
                        value=val))
        if best is None or val > best[0]:
            best = (val, kind, label)
    if best is None:
        raise Refused('no machine set holds an operation')
    return dict(value=best[0], kind=best[1], group=best[2], integral=integral,
                chain=float((h + d + q).max()), groups=out)


# ----------------------------------------------------------------- feasible makespans
class Feasible:
    """{(size, combo, index): [(source, makespan)]} of validated feasible makespans."""

    def __init__(self):
        self.by_cell = {}
        self.notes = []

    def add(self, cell, source, value):
        if value is None:
            return
        v = float(value)
        if not math.isfinite(v):
            return
        self.by_cell.setdefault(cell, []).append((source, v))

    def best(self, cell):
        x = self.by_cell.get(cell)
        if not x:
            return None, None
        s, v = min(x, key=lambda t: (t[1], t[0]))
        return v, s

    def count(self, cells):
        return sum(len(self.by_cell.get(c, ())) for c in cells)


def _alias_identical(src, dst, name):
    """True when the descriptors of every combination and both base-instance files of instance
    `name` are the same in data/FAMILY/<src> and data/FAMILY/<dst>."""
    key = (src, dst, name)
    if key in _ALIAS_OK:
        return _ALIAS_OK[key]
    ok = True
    for c in R8:
        try:
            a, b = descriptor_record(src, c, name), descriptor_record(dst, c, name)
        except OSError:
            ok = False
            break
        if a != b:
            ok = False
            break
        for ext in ('.fjs', '.meta.json'):
            fa, fb = a['base_instance'] + ext, b['base_instance'] + ext
            if not (os.path.exists(fa) and os.path.exists(fb)) or _sha(fa) != _sha(fb):
                ok = False
        if not ok:
            break
    _ALIAS_OK[key] = ok
    return ok


_ALIAS_OK = {}
_SHA = {}


def _sha(path):
    if path not in _SHA:
        with open(path, 'rb') as f:
            _SHA[path] = hashlib.sha256(f.read()).hexdigest()
    return _SHA[path]


def exact_arrays(size, names, feas, cells):
    """Column 0 of every Result_<tag>_<c>.npy of the exact environment's folders of the set, as
    feasible makespans of the analysis cells. Returns {folder: number of arrays read}."""
    want = {i for (_, _, i) in cells}
    read = {}
    # (folder, data set whose descriptors the rows follow)
    folders = [(size, size)] + [(f, f if f in ALIAS else size) for f in EXTRA_FOLDERS.get(size, ())]
    seen = set()
    for folder, data_set in folders:
        d = REG.res_dir(folder)
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        alias = data_set if data_set != size else None
        n_rows = len(names)
        if alias is not None:
            try:
                n_rows = len(descriptor_names(alias))
            except Refused:
                continue
        k = 0
        for p in sorted(glob.glob(f'{d}/Result_*.npy')):
            m = _ARRAY.match(os.path.basename(p))
            if not m:
                continue
            try:
                col = REG.read_path(p)
            except Exception as e:  # noqa: BLE001  (an array being written by a running job)
                feas.notes.append(f'{p}: unreadable ({type(e).__name__}), left out')
                continue
            if len(col) != n_rows:
                feas.notes.append(f'{p}: {len(col)} rows, the folder has {n_rows} instances; '
                                  'left out')
                continue
            c = m.group('c')
            for i in range(n_rows):
                if i not in want:
                    continue
                if alias is not None and not _alias_identical(alias, size, names[i]):
                    continue
                feas.add((size, c, i), f'{os.path.basename(d)}/{m.group("tag")}', col[i])
            k += 1
        read[os.path.basename(d)] = k
    return read


def _record_values(rec):
    vals = []
    for k in ('makespan', 'objective', 'best_makespan', 'greedy_makespan'):
        v = rec.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            vals.append((k, float(v)))
    for r in rec.get('rules') or []:
        if isinstance(r, dict) and r.get('eligible') and not r.get('violations') \
                and isinstance(r.get('makespan'), (int, float)):
            vals.append((f'rule {r.get("rule")}', float(r['makespan'])))
    return vals


def timed_records(size, names, feas, cells):
    """Validated makespans of the records of the timed comparison and of its start schedules.
    Returns {pattern: number of records read}."""
    want = {(c, i) for (_, c, i) in cells}
    index = {n: i for i, n in enumerate(names)}
    orr, trr = INPUTS.record_roots
    pats = {'memetic': f'{orr}/{size}_memetic*/*/instance_*.json',
            'rule start': f'{orr}/{size}_rulestart/*/instance_*.json',
            'policy samples': f'{orr}/{size}_polsample*/*/instance_*.json',
            'CP-SAT': f'{trr}/{size}_cparms/**/instance_*.json'}
    read = {}
    for what, pat in pats.items():
        k = 0
        for p in sorted(glob.glob(pat, recursive=True)):
            m = _RECORD.match(os.path.basename(p))
            combo = os.path.basename(os.path.dirname(p))
            if not m or combo not in R8 or m.group('name') not in index:
                continue
            if (combo, index[m.group('name')]) not in want:
                continue
            try:
                rec = json.load(open(p))
            except (OSError, ValueError):
                feas.notes.append(f'{p}: unreadable record, left out')
                continue
            if not isinstance(rec, dict) or rec.get('instance') != m.group('name') or \
                    rec.get('regime', rec.get('combination')) != combo or \
                    rec.get('size', size) != size:
                feas.notes.append(f'{p}: the record names another cell, left out')
                continue
            if rec.get('validated') is not True or rec.get('violations'):
                continue
            base = orr if what != 'CP-SAT' else trr
            src = os.path.relpath(os.path.dirname(os.path.dirname(p)), base)
            for key, v in _record_values(rec):
                feas.add((size, combo, index[m.group('name')]), f'{src}:{key}', v)
            k += 1
        read[what] = k
    return read


# ----------------------------------------------------------------- bound (iii)
def cache_folder(folder):
    for rx, tpl in CACHE_OF:
        m = rx.match(folder)
        if m:
            return tpl.format(**m.groupdict())
    return None


def _schedule_ok(jl, pt, desc, op_st, op_ct, op_mch, makespan):
    op_st, op_ct, op_mch = (np.asarray(x, dtype=float) for x in (op_st, op_ct, op_mch))
    if op_st.shape != (pt.shape[0],) or op_ct.shape != op_st.shape or op_mch.shape != op_st.shape:
        return 'the schedule has the wrong length'
    if np.any(op_mch < 0) or np.any(op_mch >= pt.shape[1]):
        return 'the schedule names a machine outside the instance'
    v = validate_schedule(jl, pt, desc, op_st, op_ct, op_mch.astype(int))
    if v:
        return f'the schedule violates today\'s instance or descriptor ({v[0]})'
    if abs(float(op_ct.max()) - float(makespan)) > TOL * max(1.0, abs(float(makespan))):
        return f'the largest completion {op_ct.max():g} differs from the makespan {makespan:g}'
    return None


_CACHE_OK = {}


def _warm_ok(rec, folder, combo, name, jl, pt, desc, cell_key):
    """None when the warm start the record names is in its cache and passes validation."""
    cf = cache_folder(folder)
    if cf is None:
        return f'no warm-start cache is known for the folder {folder}', None
    path = f'{INPUTS.solver_root}/{cf}/{combo}/{name}.npz'
    if not os.path.exists(path):
        return f'its warm start {path} is absent', None
    try:
        z = np.load(path, allow_pickle=True)
        ms, rule = float(z['ms']), str(z['rule'])
        names = [str(x) for x in z['rule_names']]
        rms = [float(x) for x in z['rule_ms']]
        amch, ct = np.asarray(z['amch'], dtype=int), np.asarray(z['ct'], dtype=float)
    except (OSError, KeyError, ValueError) as e:
        return f'its warm start {path} is unreadable ({e})', None
    if rec.get('warm_rule') != rule or rec.get('warm_makespan') is None or \
            abs(float(rec['warm_makespan']) - ms) > TOL * max(1.0, ms):
        return (f'its warm start ({rec.get("warm_rule")}, {rec.get("warm_makespan")}) differs '
                f'from {path} ({rule}, {ms:g})'), None
    war = rec.get('warm_all_rules')
    if isinstance(war, dict):
        mine = dict(zip(names, rms))
        if set(war) != set(mine) or any(
                not ((math.isinf(float(war[k])) and math.isinf(mine[k])) or
                     abs(float(war[k]) - mine[k]) <= TOL * max(1.0, abs(mine[k]))) for k in war):
            return f'its warm-start makespans differ from {path}', None
    key = (path, cell_key)
    if key not in _CACHE_OK:
        if amch.shape != (pt.shape[0],) or np.any(amch < 0) or np.any(amch >= pt.shape[1]):
            _CACHE_OK[key] = 'the cached warm start does not fit today\'s instance'
        else:
            st = ct - pt[np.arange(pt.shape[0]), amch]
            _CACHE_OK[key] = _schedule_ok(jl, pt, desc, st, ct, amch, ms)
    why = _CACHE_OK[key]
    if why:
        return f'its warm start {path}: {why}', None
    return None, (f'{cf} (warm start)', ms)


def stored_bounds(analysis, feas):
    """Records of the solver root with a bound, verified against today's data; the validated
    schedules and warm starts go into feas. A verified record written before the date of the
    rule (INPUTS.stored_since, file modification time), or of a cell whose descriptor the
    records' solver model does not describe (model_mismatch), keeps its validated schedule in
    feas but gives no bound. Returns (candidates, counts, examples)."""
    root = INPUTS.solver_root
    since = since_time(INPUTS.stored_since)
    counts = dict(records_with_bound=0, wrong_fields=0, other_data=0, outside=0,
                  unverifiable=0, verified=0, before_date=0, model_mismatch=0,
                  stored_since=str(INPUTS.stored_since), by_folder={}, by_method={})
    examples = dict(wrong_fields=[], other_data=[], unverifiable=[])
    cand = []
    folders = sorted(f for f in (os.listdir(root) if os.path.isdir(root) else [])
                     if not f.startswith('_') and os.path.isdir(os.path.join(root, f)))
    for folder in folders:
        m = _FOLDER.match(folder)
        if not m:
            continue
        src = m.group('set')
        dst = src if src in analysis else ALIAS.get(src)
        if dst not in analysis:
            continue
        names, idx_name = analysis[dst]['names'], analysis[dst]['index']
        for combo in R8:
            for p in sorted(glob.glob(f'{root}/{folder}/{combo}/instance_*.json')):
                mm = _RECORD.match(os.path.basename(p))
                if not mm or '_seed' in os.path.basename(p):
                    continue
                try:
                    rec = json.load(open(p))
                except (OSError, ValueError):
                    continue
                if not isinstance(rec, dict) or rec.get('bound') is None:
                    continue
                name = mm.group('name')
                counts['records_with_bound'] += 1
                fc = counts['by_folder'].setdefault(folder, dict(read=0, verified=0, used=0,
                                                                 excluded=0, unverifiable=0,
                                                                 outside=0, rejected=0,
                                                                 before_date=0,
                                                                 model_mismatch=0))
                fc['read'] += 1
                rel = os.path.relpath(p, root)
                if rec.get('instance') != name or rec.get('regime', combo) != combo or \
                        rec.get('size', src) != src:
                    counts['wrong_fields'] += 1
                    fc['rejected'] += 1
                    if len(examples['wrong_fields']) < MAX_EXAMPLES:
                        examples['wrong_fields'].append(f'{rel}: the record names '
                                                        f'{rec.get("instance")}, combination '
                                                        f'{rec.get("regime")}')
                    continue
                if name not in idx_name:
                    counts['outside'] += 1
                    fc['outside'] += 1
                    continue
                if src != dst and not _alias_identical(src, dst, name):
                    counts['other_data'] += 1
                    fc['rejected'] += 1
                    if len(examples['other_data']) < MAX_EXAMPLES:
                        examples['other_data'].append(f'{rel}: {src}/{name} is not a copy of '
                                                      f'{dst}/{name}')
                    continue
                i = idx_name[name]
                jl, pt, meta, desc, _ = cell_data(dst, combo, name)
                why, got = None, None
                has_sched = all(rec.get(k) is not None for k in ('assigned_mch', 'op_st', 'op_ct'))
                if rec.get('objective') is not None and rec.get('validated') is False:
                    why = 'the record\'s own validation failed'
                elif has_sched and rec.get('objective') is not None:
                    why = _schedule_ok(jl, pt, desc, rec['op_st'], rec['op_ct'],
                                       rec['assigned_mch'], rec['objective'])
                    if why is None:
                        got = (f'{folder} (schedule)', float(rec['objective']))
                        method = 'schedule'
                elif rec.get('warm_rule') is not None:
                    why, got = _warm_ok(rec, folder, combo, name, jl, pt, desc,
                                        (dst, combo, i))
                    method = 'warm start'
                else:
                    why = 'the record holds neither a schedule nor a warm start'
                if why is not None:
                    counts['unverifiable'] += 1
                    fc['unverifiable'] += 1
                    if len(examples['unverifiable']) < MAX_EXAMPLES:
                        examples['unverifiable'].append(f'{rel}: {why}')
                    continue
                counts['verified'] += 1
                fc['verified'] += 1
                counts['by_method'][method] = counts['by_method'].get(method, 0) + 1
                feas.add((dst, combo, i), got[0], got[1])
                # The date rule. A record solved with the shorter window horizon H would in fact
                # still carry a valid bound whenever its cell has a validated schedule with a
                # makespan of at most H. Every variable of the model (block, start, end, block
                # length, makespan) is at most the makespan, so every schedule with a makespan of
                # at most H lies inside the variable domains. The other constraints are those of
                # today's model, which admits every schedule the validator accepts (contiguous
                # setup and processing block, setup from the previous class on the machine,
                # block after the job is ready, no overlap with an outage), and an optimal
                # schedule has integer times under integer data. The optimal schedule (makespan
                # at most that of the validated one, hence at most H) is therefore feasible in
                # the truncated model, so the optimum of the truncated model equals the true
                # optimum and the proved bound lies below it. The rule leaves such records out
                # all the same, so that the provenance of every bound used can be checked from
                # the file dates alone.
                mtime = os.path.getmtime(p)
                bs = counts.setdefault('by_set', {}).setdefault(
                    dst, dict(before_date=0, model_mismatch=0))
                if mtime < since:
                    counts['before_date'] += 1
                    fc['before_date'] += 1
                    bs['before_date'] += 1
                    continue
                if model_mismatch(desc):
                    counts['model_mismatch'] += 1
                    fc['model_mismatch'] += 1
                    bs['model_mismatch'] += 1
                    continue
                cand.append(dict(cell=(dst, combo, i), folder=folder, record=rel,
                                 bound=float(rec['bound']), status=rec.get('status'),
                                 objective=rec.get('objective'), budget_s=rec.get('budget_s'),
                                 verified_by=method, warm_start=rec.get('warm_rule') is not None,
                                 written=datetime.datetime.fromtimestamp(mtime)
                                 .strftime('%Y-%m-%d %H:%M')))
    return cand, counts, examples


def apply_exclusion(cand, feas, counts):
    """Stored bounds above the best known makespan of their cell are excluded."""
    used, excluded = [], []
    for x in cand:
        best, src = feas.best(x['cell'])
        fc = counts['by_folder'][x['folder']]
        if best is not None and x['bound'] > best + TOL * max(1.0, abs(best)):
            excluded.append(dict(x, best_known=best, best_known_source=src))
            fc['excluded'] += 1
        else:
            used.append(x)
            fc['used'] += 1
    counts['used'] = len(used)
    counts['excluded'] = len(excluded)
    return used, excluded


def stored_folders(used, attaining):
    """Per folder of the stored records used: the records, their budgets (field budget_s), how
    many started from a warm start, the oldest and newest file modification time, how they were
    verified, and the cells on which the folder's record holds the largest stored bound and
    that bound is the reference (attaining, one record per cell)."""
    out = {}
    for x in used:
        f = out.setdefault(x['folder'], dict(records=0, budgets={}, warm_start=0, oldest=None,
                                             newest=None, verified_by={}, reference_cells=0))
        f['records'] += 1
        b = f'{x["budget_s"]:g}' if isinstance(x['budget_s'], (int, float)) else str(x['budget_s'])
        f['budgets'][b] = f['budgets'].get(b, 0) + 1
        f['warm_start'] += int(bool(x.get('warm_start')))
        f['oldest'] = min(f['oldest'] or x['written'], x['written'])
        f['newest'] = max(f['newest'] or x['written'], x['written'])
        f['verified_by'][x['verified_by']] = f['verified_by'].get(x['verified_by'], 0) + 1
    for x in attaining:
        if x['folder'] in out:
            out[x['folder']]['reference_cells'] += 1
    return dict(sorted(out.items()))


# ----------------------------------------------------------------- the timed comparison
SEARCH_TEXT = {'memetic': 'memetic algorithm', 'solver': 'CP-SAT'}


def _p94():
    """p94_timed_inference; its imports change the folder and the command line, both restored."""
    cwd, argv = os.getcwd(), list(sys.argv)
    import p94_timed_inference as T  # noqa: E402
    os.chdir(cwd)
    sys.argv[:] = argv
    return T


def parse_campaigns(specs):
    """The --campaign descriptions, "<sizes> = <tokens>" per launch, through
    p94_timed_inference.parse_campaign. Returns [dict(spec, sizes, tokens, ignored)]: tokens are
    those of the two rule-started searches (any budget; the 600 s cells are selected once the
    planner has resolved the budgets), ignored the other tokens the planner or p94 knows."""
    if not specs:
        raise Refused('the timed comparison must be described as it was launched, one --campaign '
                      'per launch:\n  ' + ' '.join(f'--campaign "{s}"' for s in AS_LAUNCHED))
    T = _p94()
    known = set(T.P.ARMS) | {T.POLICY_ARM}
    out = []
    for spec in specs:
        try:
            sizes, toks = T.parse_campaign(spec)
        except T.Refusal as e:
            raise Refused('; '.join(e.args[0]))
        bad = [s for s in sizes if s not in INPUTS.sets]
        if bad:
            raise Refused(f'campaign {spec!r}: {", ".join(bad)} is not a test set of the analysis '
                          f'({", ".join(INPUTS.sets)})')
        keep, ignored = [], []
        for t in toks:
            name = t.partition('@')[0]
            if name in TIMED_ARMS:
                keep.append(t)
            elif name in known:
                ignored.append(t)
            else:
                raise Refused(f'campaign {spec!r}: unknown token {t} (known: '
                              f'{", ".join(sorted(known))})')
        out.append(dict(spec=spec, sizes=sizes, tokens=keep, ignored=ignored))
    return out


def launched(parsed, seed, chosen):
    """The planner's Campaign of every launch (p94_timed_inference.build_parser and
    make_campaign, with the planner's defaults), each reduced to the segments of the two
    rule-started searches at BUDGET s. Returns (campaigns, segment descriptions)."""
    T = _p94()
    a = T.build_parser().parse_args(sum((['--campaign', p['spec']] for p in parsed), []))
    camps, segs = [], []
    for p in parsed:
        if not p['tokens']:
            continue
        try:
            camp = T.make_campaign(p['sizes'], p['tokens'], a, seed, chosen)
        except SystemExit as e:
            raise Refused(f'campaign {p["spec"]!r}: {e}')
        camp.segments = [s for s in camp.segments if abs(s.budget - BUDGET) < 1e-9]
        if not camp.segments:
            continue
        camps.append(camp)
        for s in camp.segments:
            for size, insts in s.by_size.items():
                segs.append(dict(token=s.label, method=TIMED_ARMS[s.arm], size=size,
                                 instances=T.P.compress(insts), n_instances=len(insts),
                                 cells=len(insts) * len(camp.combos), launch=p['spec']))
    return camps, segs


def read_timed(parsed, seed):
    """{(token name, size, combo, index): cell} of the launched 600 s cells of the two
    rule-started searches through p94_timed_inference.make_campaign and read_cells (p113's
    call), the counts of cell states, the CP-SAT configuration the pilot selected and the
    launched segments."""
    T = _p94()
    P = T.P
    table = P.build_parser().parse_args(['table'])
    slots = [P.parse_cores(s) for s in table.slots]
    chosen, why = T.select_cpsat_arm('test_results/FAMILY', slots, table.bars)
    if chosen is None:
        raise Refused(f'the pilot selection of the CP-SAT configuration is unavailable: '
                      f'{why[0] if why else "-"}')
    camps, segs = launched(parsed, seed, chosen)
    if not camps:
        return {}, {}, chosen, segs
    try:
        cells, counts, _ = T.read_cells(camps, (float(BUDGET),), 'start')
    except T.Refusal as e:
        raise Refused(f'the timed records cannot be read: {"; ".join(e.args[0])}')
    return cells, counts, chosen, segs


def segment_lines(segs, counts):
    """One line per launched segment: the search, the set, the instances and the cell states."""
    out = []
    for s in segs:
        tok = s['token'].partition('@')[0]
        st = {}
        for key, v in counts.items():
            arm, size, _ = key.split('|')
            if arm == tok and size == s['size']:
                for k, n in v.items():
                    st[k] = st.get(k, 0) + n
        out.append(f'{SEARCH_TEXT[s["method"]]} started from the best dispatching rule, '
                   f'{BUDGET} s, {s["size"]}, instances {s["instances"]}: {s["cells"]} cells'
                   + (', ' + ', '.join(f'{n} {k}' for k, n in sorted(st.items())) if st else ''))
    return out


# ----------------------------------------------------------------- arrays of the methods
def read_column(path, n):
    if not os.path.exists(path):
        return None, f'missing {path}'
    try:
        a = REG.read_path(path)
    except Exception as e:  # noqa: BLE001
        return None, f'unreadable {path} ({type(e).__name__})'
    if len(a) != n:
        return None, f'{path} holds {len(a)} rows, the set has {n} instances'
    return a, None


def method_columns(size, n, seeds, sel):
    """{method: {combo: column over all descriptors}} of the array methods, and the missing."""
    out, miss = {m: {} for m in ('policy', 'policy_mean', 'specialist', 'best_rule')}, []
    for c in R8:
        cols = {}
        for s in seeds:
            a, why = read_column(REG.result_path('released', None, s, size, c), n)
            if why:
                miss.append(why)
            else:
                cols[s] = a
        if sel in cols:
            out['policy'][c] = cols[sel]
        if len(cols) == len(seeds):
            out['policy_mean'][c] = np.mean([cols[s] for s in seeds], axis=0)
        if size in SPECIALIST_SETS:
            sc = []
            for s in seeds:
                a, why = read_column(REG.result_path('specialist', c, s, size, c), n)
                if why:
                    miss.append(why)
                else:
                    sc.append(a)
            if len(sc) == len(seeds):
                out['specialist'][c] = np.mean(sc, axis=0)
        rc = []
        for q in F88.RULES:
            a, why = read_column(f'{REG.rule_dir(size)}/Result_{q}_{c}.npy', n)
            if why:
                miss.append(why)
            else:
                rc.append(a)
        if len(rc) == len(F88.RULES):
            out['best_rule'][c] = np.min(rc, axis=0)
    return out, miss


# ----------------------------------------------------------------- statistics
def gap_summary(d, combos, B):
    """d [n, len(combos)] in percent, nan where a method has no schedule. Per combination the
    mean over its instances with stats_family.instance_boot; pooled over the instances present
    in every combination, the combinations of an instance kept together."""
    out = {}
    for j, c in enumerate(combos):
        ok = np.isfinite(d[:, j])
        if not ok.any():
            continue
        x = SF.instance_boot(d[ok, j], B=B, seed=F88.INST_SEED, alpha=ALPHA)
        out[c] = dict(mean=x['mean'], lo=x['lo'], hi=x['hi'], n=int(ok.sum()),
                      rows=np.where(ok)[0].tolist())
    rows = np.all(np.isfinite(d), axis=1)
    if rows.any() and len(out) == len(combos):
        x = SF.instance_boot(d[rows], B=B, seed=F88.INST_SEED, alpha=ALPHA)
        out['All'] = dict(mean=x['mean'], lo=x['lo'], hi=x['hi'], n=int(rows.sum()),
                          c=len(combos), rows=np.where(rows)[0].tolist())
    return out


def bound_shares(rows):
    """Percent of cells on which each bound attains the reference (ties counted for each) and
    on which it alone does."""
    n = len(rows)
    out = dict(n=n)
    for b in BOUNDS:
        at = [r for r in rows if r[b] is not None and
              r[b] >= r['reference'] - TOL * max(1.0, abs(r['reference']))]
        alone = [r for r in at if all(r[o] is None or r[o] < r['reference'] -
                                      TOL * max(1.0, abs(r['reference']))
                                      for o in BOUNDS if o != b)]
        out[b] = dict(largest=100.0 * len(at) / n if n else None, largest_n=len(at),
                      alone=100.0 * len(alone) / n if n else None, alone_n=len(alone))
    return out


def is_optimal(r):
    """A cell is proved optimal when its best known makespan equals its reference bound within
    the tolerance of the self-check (relative TOL)."""
    return (r['best_known'] is not None and
            abs(r['best_known'] - r['reference']) <= TOL * max(1.0, abs(r['reference'])))


def proved_optimal(rows):
    """Cells proved optimal among rows: their count, the number of cells, and on how many of
    them each bound attains the reference (ties counted for each bound)."""
    opt = [r for r in rows if is_optimal(r)]
    return dict(count=len(opt), cells=len(rows),
                attained={b: sum(1 for r in opt if b in r['largest']) for b in BOUNDS})


def bound_over_best(rows):
    """Mean of 100 bound / best known makespan for each bound and the reference."""
    out = {}
    for b in BOUNDS + ('reference',):
        v = [100.0 * r[b] / r['best_known'] for r in rows
             if r[b] is not None and r['best_known']]
        out[b] = dict(mean=float(np.mean(v)) if v else None, n=len(v),
                      min=float(np.min(v)) if v else None, max=float(np.max(v)) if v else None)
    return out


# ----------------------------------------------------------------- analysis
def expected_missing(analysis, cols_miss, timed_counts):
    """The incomplete input groups: missing arrays and timed cells that are not valid."""
    miss = list(cols_miss)
    for key, states in sorted(timed_counts.items()):
        bad = {s: k for s, k in states.items() if s != 'valid'}
        if bad:
            arm, size, combo = key.split('|')
            what = SEARCH_TEXT.get(TIMED_ARMS.get(arm), arm)
            miss.append(f'timed comparison at {BUDGET} s, {what} started from the best '
                        f'dispatching rule, {size}/{combo}: cells '
                        + ', '.join(f'{k} {s}' for s, k in sorted(bad.items())))
    return miss


def setting_words(res):
    """The setting for a printed header: environment and training seeds."""
    return f'environment {ENV}, training seeds {",".join(map(str, res["seeds"]))}'


def slack_text(s):
    """A slack in hours; float rounding of the estimate (about 1e-13 h) prints as 0."""
    return '0' if abs(s) <= TOL else f'{s:g}'


def analyse(partial=False, boot=SF.B_DEFAULT):
    seeds = check_setting()
    parsed = parse_campaigns(INPUTS.campaigns)
    try:
        sel, sel_scores = INPUTS.select_policy()
        cfg = INPUTS.config(sel)
    except (IB.Refused, REG.MissingResult, OSError, ValueError) as e:
        raise Refused(f'the selected policy run or its configuration is unavailable: {e}')
    mode = resolve_delta_mode(argparse.Namespace(delta_mode=cfg.get('delta_mode'),
                                                 reward_delta_env=cfg.get('reward_delta_env',
                                                                          True)))
    if mode != 'min':
        raise Refused(f'the selected run\'s envelope is {mode!r}; the instance bound needs the '
                      'minimum-setup term (min)')
    analysis = {}
    for size in INPUTS.sets:
        names, idx = analysis_instances(size)
        analysis[size] = dict(names=names, idx=idx, index={n: i for i, n in idx})
    # bounds (i) and (ii), data checks
    rows, struct, faults, nonint = {}, {}, [], []
    for size, A in analysis.items():
        st = dict(ops=0, one_type=0, whole_type=0, partition_instances=0, instances=0)
        for i, name in A['idx']:
            bases = {cell_data(size, c, name)[4]['base_instance'] for c in R8}
            if len(bases) != 1:
                faults.append(f'{size}/{name}: the combinations name {len(bases)} base instances')
                continue
            jl, pt, meta, _, _ = cell_data(size, R8[0], name)
            s = station_structure(pt, meta['mch_type'])
            for k in ('ops', 'one_type', 'whole_type'):
                st[k] += s[k]
            st['partition_instances'] += int(s['partition'])
            st['instances'] += 1
            st['types'] = s['types']
            st['machines_per_type'] = s['machines_per_type']
            for c in R8:
                jl, pt, meta, desc, rec = cell_data(size, c, name)
                f = descriptor_faults(desc, pt.shape[0])
                if f:
                    faults.append(f'{size}/{c}/{name}: {"; ".join(f)}')
                    continue
                lb2 = machine_load_bound(jl, pt, desc, meta['mch_type'], s)
                lb1 = instance_bound(size, c, name, cfg)
                if abs(lb2['chain'] - lb1) > TOL * max(1.0, lb1) and \
                        getattr(desc, 'setup_mult', None) is None:
                    faults.append(f'{size}/{c}/{name}: the longest chain of shortest blocks '
                                  f'{lb2["chain"]:.6f} differs from the instance bound '
                                  f'{lb1:.6f}')
                if not lb2['integral']:
                    nonint.append(f'{size}/{c}/{name}')
                rows[(size, c, i)] = dict(size=size, combo=c, index=i, instance=name,
                                          chain=lb1, load=lb2['value'],
                                          load_group=[lb2['kind'], lb2['group']],
                                          integral=lb2['integral'],
                                          release=getattr(desc, 'release', None) is not None)
        struct[size] = st
    if faults:
        raise Refused(f'{len(faults)} cells with a descriptor the bounds cannot read or a '
                      f'bound reading that disagrees with the environment; first: {faults[0]}')
    cells = list(rows)
    # feasible makespans
    feas = Feasible()
    arrays_read, records_read = {}, {}
    for size, A in analysis.items():
        mine = [k for k in cells if k[0] == size]
        arrays_read[size] = exact_arrays(size, A['names'], feas, mine)
        records_read[size] = timed_records(size, A['names'], feas, mine)
    # methods
    cols, cols_miss = {}, []
    for size, A in analysis.items():
        cols[size], m = method_columns(size, len(A['names']), seeds, sel)
        cols_miss += m
    tcells, tcounts, solver_arm, tsegs = INPUTS.timed(parsed, sel)
    for (arm, size, c, i), x in tcells.items():
        if arm in TIMED_ARMS and x.get('state') == 'valid' and (size, c, i) in rows:
            feas.add((size, c, i), f'timed {arm}', x.get('final_makespan'))
    missing = expected_missing(analysis, cols_miss, tcounts)
    if missing and not partial:
        raise Refused(f'{len(missing)} inputs are incomplete (use --partial --out DIR to report '
                      'the others):\n  ' + '\n  '.join(missing[:20])
                      + (f'\n  ... and {len(missing) - 20} more' if len(missing) > 20 else ''))
    # bound (iii)
    cand, scounts, sexamples = stored_bounds(analysis, feas)
    # self-check of (i) and (ii)
    check = self_check(rows, feas)
    if check['violations']:
        raise Refused(f'{len(check["violations"])} feasible makespans lie below bound (i) or (ii); '
                      'nothing is written. First cases:\n  '
                      + '\n  '.join(check['violations'][:10]))
    used, excluded = apply_exclusion(cand, feas, scounts)
    best_stored = {}
    for x in used:
        k = x['cell']
        if k not in best_stored or x['bound'] > best_stored[k]['bound']:
            best_stored[k] = x
    for k, r in rows.items():
        s = best_stored.get(k)
        r['stored'] = None if s is None else s['bound']
        r['stored_record'] = None if s is None else s['record']
        r['reference'] = max(v for v in (r['chain'], r['load'], r['stored']) if v is not None)
        tol = TOL * max(1.0, r['reference'])
        r['largest'] = [b for b in BOUNDS if r[b] is not None and r[b] >= r['reference'] - tol]
        r['best_known'], r['best_known_source'] = feas.best(k)
        r['feasible_n'] = len(feas.by_cell.get(k, ()))
    # makespans of the methods per cell
    for (size, c, i), r in rows.items():
        for m in ('policy', 'policy_mean', 'specialist', 'best_rule'):
            col = cols[size][m].get(c)
            r[m] = None if col is None else float(col[i])
        for arm, m in TIMED_ARMS.items():
            x = tcells.get((arm, size, c, i))
            r[m] = (float(x['final_makespan']) if x and x.get('state') == 'valid' else None)
            r[f'{m}_record'] = x.get('record') if x else None
    # per set: gaps, shares, bound against the best known
    per = {}
    for size, A in analysis.items():
        idx = [i for i, _ in A['idx']]
        S = dict(n=len(idx), instances=[n for _, n in A['idx']], gaps={}, matched={},
                 shares={}, over_best={}, stored=dict(used=0, excluded=0, cells=0))
        for m in METHODS:
            d = np.full((len(idx), len(R8)), np.nan)
            for a, i in enumerate(idx):
                for j, c in enumerate(R8):
                    r = rows[(size, c, i)]
                    if r[m] is not None:
                        d[a, j] = 100.0 * (r[m] - r['reference']) / r['reference']
            g = gap_summary(d, R8, boot)
            if g:
                S['gaps'][m] = g
            if m in ('memetic', 'solver') and g:
                dp = np.full_like(d, np.nan)
                for a, i in enumerate(idx):
                    for j, c in enumerate(R8):
                        r = rows[(size, c, i)]
                        if np.isfinite(d[a, j]) and r['policy'] is not None:
                            dp[a, j] = 100.0 * (r['policy'] - r['reference']) / r['reference']
                gm = gap_summary(dp, R8, boot)
                if gm:
                    S['matched'][m] = gm
        S['optimal'] = {}
        for c in R8 + ('All',):
            sel_rows = [rows[(size, cc, i)] for i in idx for cc in R8 if c in ('All', cc)]
            S['shares'][c] = bound_shares(sel_rows)
            S['over_best'][c] = bound_over_best(sel_rows)
            S['optimal'][c] = proved_optimal(sel_rows)
        S['stored']['used'] = sum(1 for x in used if x['cell'][0] == size)
        S['stored']['excluded'] = sum(1 for x in excluded if x['cell'][0] == size)
        S['stored']['cells'] = sum(1 for k in best_stored if k[0] == size)
        bs = scounts.get('by_set', {}).get(size, {})
        S['stored']['before_date'] = bs.get('before_date', 0)
        S['stored']['model_mismatch'] = bs.get('model_mismatch', 0)
        S['stored']['folders'] = stored_folders([x for x in used if x['cell'][0] == size],
                                                [x for k, x in best_stored.items()
                                                 if k[0] == size and rows[k]['stored'] is not None
                                                 and 'stored' in rows[k]['largest']])
        pol = S['gaps'].get('policy', {})
        pc = {c: pol[c]['mean'] for c in R8 if c in pol}
        if pc:
            hi = max(pc, key=lambda c: (pc[c], -R8.index(c)))
            lo = min(pc, key=lambda c: (pc[c], R8.index(c)))
            S['policy_extremes'] = dict(largest=dict(combo=hi, mean=pc[hi]),
                                        smallest=dict(combo=lo, mean=pc[lo]))
        per[size] = S
    states = {}
    for key, v in tcounts.items():
        states[key] = v
    return dict(generator=GENERATOR, setting=REG.describe(), seeds=seeds, dry=INPUTS.dry,
                partial=bool(missing), missing=missing, boot=int(boot), alpha=ALPHA,
                inst_seed=F88.INST_SEED, bootstrap='stats_family.instance_boot',
                policy_seed=sel, policy_scores=sel_scores, delta_mode=mode,
                solver_arm=solver_arm, budget_s=BUDGET,
                campaign=dict(descriptions=[p['spec'] for p in parsed],
                              read_past=[t for p in parsed for t in p['ignored']],
                              segments=tsegs, lines=segment_lines(tsegs, tcounts)),
                sets={s: dict(word=INPUTS.sets[s][0], first=INPUTS.sets[s][1],
                              n=len(analysis[s]['idx'])) for s in analysis},
                station_types=struct, non_integer_cells=nonint,
                feasible_sources=dict(arrays=arrays_read, records=records_read,
                                      notes=feas.notes[:50], n_notes=len(feas.notes)),
                timed_states=states,
                stored=dict(counts=scounts, examples=sexamples,
                            excluded=[dict(record=x['record'], bound=x['bound'],
                                           best_known=x['best_known'],
                                           best_known_source=x['best_known_source'])
                                      for x in excluded],
                            excluded_examples=[f'{x["record"]}: bound {x["bound"]:g} above the '
                                               f'best known makespan {x["best_known"]:g} '
                                               f'({x["best_known_source"]})'
                                               for x in excluded[:MAX_EXAMPLES]]),
                self_check=check, per=per,
                cells=[rows[k] for k in sorted(rows, key=lambda k: (list(analysis).index(k[0]),
                                                                     R8.index(k[1]), k[2]))],
                definitions=dict(
                    gap='100 (makespan - reference bound) / reference bound per instance',
                    reference='the largest of the instance bound, the machine-load bound and the '
                              'stored solver bound',
                    chain=BOUND_TEXT['chain'] + ': p113_instance_bound.estimate',
                    load=BOUND_TEXT['load'] + ': machine_load_bound()',
                    stored=BOUND_TEXT['stored'] + ': verified records of the solver root',
                    best_known='the shortest validated feasible makespan of the cell',
                    shares='percent of cells on which a bound attains the reference, ties '
                           'counted for every bound that attains it',
                    interval='stats_family.instance_boot, 95% percentile, fresh '
                             f'default_rng({F88.INST_SEED}) per quantity'))


def self_check(rows, feas):
    """Bounds (i) and (ii) against every feasible makespan of the same cell."""
    pairs = {'chain': 0, 'load': 0}
    small = {'chain': None, 'load': None}
    viol = []
    for k, r in rows.items():
        for src, v in feas.by_cell.get(k, ()):
            for b in ('chain', 'load'):
                pairs[b] += 1
                s = v - r[b]
                if small[b] is None or s < small[b]['slack']:
                    small[b] = dict(slack=s, relative=s / v if v else None,
                                    cell=f'{k[0]}/{k[1]}/{r["instance"]}', source=src,
                                    makespan=v, bound=r[b])
                if r[b] > v + TOL * max(1.0, abs(v)):
                    viol.append(f'{k[0]}/{k[1]}/{r["instance"]}: {BOUND_TEXT[b]} {r[b]:.6f} '
                                f'above the makespan {v:g} of {src}')
    no = [f'{k[0]}/{k[1]}/{r["instance"]}' for k, r in rows.items() if not feas.by_cell.get(k)]
    return dict(pairs=pairs, pairs_total=sum(pairs.values()), smallest=small,
                cells=len(rows), cells_without_makespan=len(no),
                cells_without_makespan_examples=no[:MAX_EXAMPLES], violations=viol)


# ----------------------------------------------------------------- macros
def defined_macros():
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.basename(f) == own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, os.path.relpath(f, ROOT))
    return out


def f1(x):
    """One decimal; a value within 1e-9 of zero (rounding of the bounds, about 1e-14 %) prints
    0.0, never -0.0."""
    x = float(x)
    return f'{0.0 if abs(x) < 1e-9 else x:.1f}'


TRAIN_SET = '10x25'
# methods of the per-combination macros and of the table, in column order
COMB_METHODS = ('policy', 'specialist', 'best_rule', 'memetic', 'solver')
NAME_SCHEME = (
    '% Names: Lbr<Set><Quantity> per test set, <Set> one of Ten Twenty Thirty Fifty Eighty; at '
    'the training size also LbrTenComb<R><Quantity> per combination, <R> one of N L S W LS LW SW '
    'LSW and always followed by a word that starts with G or O, never with a combination letter: '
    'Gap<M> (mean gap, %), Gap<M>Lo, Gap<M>Hi (95% interval), Gap<M>N (instances), Opt (cells '
    'proved optimal), OptCells (cells), OverBestRef, OverBestChain, OverBestLoad (mean of 100 '
    'bound / best known makespan: reference, instance bound, machine-load bound); <M> one of Pol '
    '(policy, run with the lowest validation score), Spec, Rule, Mem, Solver.')


def macro_lines(res, partial_line=None):
    L = []
    if partial_line:
        L.append(f'% {partial_line}')
    L.append(f'% Generated by {GENERATOR} (environment {ENV}, training seeds '
             f'{",".join(map(str, res["seeds"]))}) -- do not hand-edit')
    L.append('% Gap = 100 (makespan - LB) / LB, LB the largest of the instance bound, the '
             'machine-load bound and the verified stored solver bounds; mean over the instances '
             f'with the 95% interval of stats_family.instance_boot (B = {res["boot"]}); pooled '
             'rows keep the eight combinations of an instance together; values with one decimal.')
    L.append(NAME_SCHEME)
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    sc = res['stored']['counts']
    ck = res['self_check']
    L.append(f'\\newcommand{{\\{PREFIX}PolSeed}}{{{res["policy_seed"]}}}'
             f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(res["seeds"])}}}'
             f'\\newcommand{{\\{PREFIX}Budget}}{{{res["budget_s"]}}}'
             '  % training seed of the run with the lowest validation score; training seeds; '
             'budget of the searches in s')
    L.append(f'\\newcommand{{\\{PREFIX}StoredUsed}}{{{sc["used"]}}}'
             f'\\newcommand{{\\{PREFIX}StoredExcluded}}{{{sc["excluded"]}}}'
             f'\\newcommand{{\\{PREFIX}StoredUnverified}}{{{sc["unverifiable"]}}}'
             '  % stored solver bounds of the analyzed cells: used; excluded (above a validated '
             'feasible makespan); left out (record not verifiable against today\'s data)')
    L.append(f'\\newcommand{{\\{PREFIX}StoredBefore}}{{{sc["before_date"]}}}'
             f'\\newcommand{{\\{PREFIX}StoredSince}}{{{sc["stored_since"]}}}'
             f'  % stored solver records of the analyzed cells written before '
             f'{sc["stored_since"]} (file modification time), whose bounds are not used, and the '
             f'date: the window horizon of the constraint model was enlarged on 2026-07-07')
    L.append(f'\\newcommand{{\\{PREFIX}CheckPairs}}{{{ck["pairs_total"]}}}'
             f'\\newcommand{{\\{PREFIX}CheckCells}}{{{ck["cells"]}}}'
             '  % pairs of a bound (instance bound or machine-load bound) and a stored feasible '
             'makespan of the same cell checked; cells')
    for size, S in res['per'].items():
        w = PREFIX + res['sets'][size]['word']
        L.append(f'\\newcommand{{\\{w}N}}{{{S["n"]}}}  % {size}: instances per combination')
        for m in METHODS:
            g = S['gaps'].get(m, {}).get('All')
            if g is None:
                continue
            k = w + METHOD_WORD[m] + 'Gap'
            line = (f'\\newcommand{{\\{k}}}{{{f1(g["mean"])}}}'
                    f'\\newcommand{{\\{k}Lo}}{{{f1(g["lo"])}}}'
                    f'\\newcommand{{\\{k}Hi}}{{{f1(g["hi"])}}}'
                    f'\\newcommand{{\\{w}{METHOD_WORD[m]}N}}{{{g["n"]}}}')
            L.append(line + f'  % {size}, {METHOD_TEXT[m]}: pooled gap, %, 95% interval, and the '
                     'instances the mean rests on'
                     + (' (a valid record in every combination)'
                        if m in ('memetic', 'solver') else ''))
        for m in ('memetic', 'solver'):
            g = S['matched'].get(m, {}).get('All')
            if g is None:
                continue
            k = f'{w}PolOn{METHOD_WORD[m]}'
            L.append(f'\\newcommand{{\\{k}Gap}}{{{f1(g["mean"])}}}'
                     f'\\newcommand{{\\{k}GapLo}}{{{f1(g["lo"])}}}'
                     f'\\newcommand{{\\{k}GapHi}}{{{f1(g["hi"])}}}'
                     f'\\newcommand{{\\{k}N}}{{{g["n"]}}}'
                     f'  % {size}, {METHOD_TEXT["policy"]}, on the instances of the '
                     f'{METHOD_TEXT[m]}: pooled gap, %, 95% interval, instances')
        ex = S.get('policy_extremes')
        if ex:
            L.append(f'\\newcommand{{\\{w}PolGapMax}}{{{f1(ex["largest"]["mean"])}}}'
                     f'\\newcommand{{\\{w}PolGapMaxCombo}}{{{ex["largest"]["combo"]}}}'
                     f'\\newcommand{{\\{w}PolGapMin}}{{{f1(ex["smallest"]["mean"])}}}'
                     f'\\newcommand{{\\{w}PolGapMinCombo}}{{{ex["smallest"]["combo"]}}}'
                     f'  % {size}, {METHOD_TEXT["policy"]}: largest and smallest gap of a '
                     'combination, %, and the combination')
        sh = S['shares']['All']
        L.append(f'\\newcommand{{\\{w}LoadShare}}{{{f1(sh["load"]["largest"])}}}'
                 f'\\newcommand{{\\{w}ChainShare}}{{{f1(sh["chain"]["largest"])}}}'
                 f'\\newcommand{{\\{w}StoredShare}}{{{f1(sh["stored"]["largest"])}}}'
                 f'  % {size}: percent of cells on which the machine-load bound, the instance '
                 'bound, the stored solver bound attains the largest bound (ties counted for '
                 'each)')
        L.append(f'\\newcommand{{\\{w}StoredUsed}}{{{S["stored"]["used"]}}}'
                 f'\\newcommand{{\\{w}StoredExcluded}}{{{S["stored"]["excluded"]}}}'
                 f'\\newcommand{{\\{w}StoredCells}}{{{S["stored"]["cells"]}}}'
                 f'  % {size}: stored solver bounds used, excluded; cells with a stored bound')
        L.append(f'\\newcommand{{\\{w}StoredBefore}}{{{S["stored"]["before_date"]}}}'
                 f'  % {size}: stored solver records written before {sc["stored_since"]}, whose '
                 'bounds are not used')
        ob = S['over_best']['All']['reference']
        if ob['mean'] is not None:
            L.append(f'\\newcommand{{\\{w}RefOverBest}}{{{f1(ob["mean"])}}}'
                     f'  % {size}: mean of 100 LB / best known makespan, %')
        L += optimal_lines(w, size, S['optimal']['All'], 'all eight combinations')
        ob = S['over_best']['All']
        if ob['chain']['mean'] is not None and ob['load']['mean'] is not None:
            L.append(f'\\newcommand{{\\{w}ChainOverBest}}{{{f1(ob["chain"]["mean"])}}}'
                     f'\\newcommand{{\\{w}LoadOverBest}}{{{f1(ob["load"]["mean"])}}}'
                     f'  % {size}: mean of 100 bound / best known makespan, %, of the instance '
                     'bound alone and of the machine-load bound alone')
        if size == TRAIN_SET:
            L += combination_lines(w, size, S)
    return L


def optimal_lines(k, size, o, where):
    """The count of cells proved optimal, the number of cells, and in the comment the bounds
    that attain the reference on them."""
    at = ', '.join(f'{BOUND_TEXT[b]} {o["attained"][b]}' for b in BOUNDS)
    return [f'\\newcommand{{\\{k}Opt}}{{{o["count"]}}}'
            f'\\newcommand{{\\{k}OptCells}}{{{o["cells"]}}}'
            f'  % {size}, {where}: cells proved optimal (best known makespan equal to the '
            f'reference bound within the self-check tolerance) and cells; the reference is '
            f'attained there by: {at} (ties counted for each)']


def combination_lines(w, size, S):
    """Per combination at the training size: the gap of each method with its interval and
    instances, the cells proved optimal, and the bounds in percent of the best known makespan."""
    L = []
    for c in R8:
        k = f'{w}Comb{c}'
        for m in COMB_METHODS:
            g = S['gaps'].get(m, {}).get(c)
            if g is None:
                continue
            q = f'{k}Gap{METHOD_WORD[m]}'
            L.append(f'\\newcommand{{\\{q}}}{{{f1(g["mean"])}}}'
                     f'\\newcommand{{\\{q}Lo}}{{{f1(g["lo"])}}}'
                     f'\\newcommand{{\\{q}Hi}}{{{f1(g["hi"])}}}'
                     f'\\newcommand{{\\{q}N}}{{{g["n"]}}}'
                     f'  % {size}, combination {c}, {METHOD_TEXT[m]}: gap, %, 95% interval, '
                     'instances')
        L += optimal_lines(k, size, S['optimal'][c], f'combination {c}')
        ob = S['over_best'][c]
        if all(ob[b]['mean'] is not None for b in ('reference', 'chain', 'load')):
            L.append(f'\\newcommand{{\\{k}OverBestRef}}{{{f1(ob["reference"]["mean"])}}}'
                     f'\\newcommand{{\\{k}OverBestChain}}{{{f1(ob["chain"]["mean"])}}}'
                     f'\\newcommand{{\\{k}OverBestLoad}}{{{f1(ob["load"]["mean"])}}}'
                     f'  % {size}, combination {c}: mean of 100 bound / best known makespan, %, '
                     'of the reference bound, the instance bound alone, the machine-load bound '
                     'alone')
    return L


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


# ----------------------------------------------------------------- markdown
def _ci(g):
    return f'{f1(g["mean"])} [{f1(g["lo"])}, {f1(g["hi"])}]'


def markdown(res, partial_line=None):
    md = []
    if partial_line:
        md += [f'**{partial_line}**', '']
    md += [f'# Lower-bound reference for the makespans ({setting_words(res)})', '',
           f'Reference bound LB per cell: the largest of the {BOUND_TEXT["chain"]}, the '
           f'{BOUND_TEXT["load"]} and the {BOUND_TEXT["stored"]}. Gap = '
           f'{res["definitions"]["gap"]}, %; mean over the instances with the 95% interval of '
           f'stats_family.instance_boot (B = {res["boot"]}, default_rng({res["inst_seed"]})). '
           f'Policy run: seed {res["policy_seed"]} (lowest validation score); CP-SAT '
           f'configuration {res["solver_arm"]}; searches at {res["budget_s"]} s.'
           + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '']
    ck = res['self_check']
    sm = ck['smallest']
    md += [f'Self-check: {ck["pairs"]["chain"]} pairs (instance bound, feasible makespan) and '
           f'{ck["pairs"]["load"]} pairs (machine-load bound, feasible makespan) over '
           f'{ck["cells"]} cells, no makespan below a bound; smallest slack '
           + '; '.join(f'{BOUND_TEXT[b]} {slack_text(sm[b]["slack"])} h at {sm[b]["cell"]} '
                       f'({sm[b]["source"]})' for b in ('chain', 'load') if sm[b]) + '.', '']
    sc = res['stored']['counts']
    how = ', '.join(f'{k} {v}' for k, v in sc['by_method'].items())
    md += [f'Stored solver bounds: {sc["records_with_bound"]} records with a bound read; '
           f'{sc["verified"]} verified ({how}), '
           f'{sc["used"]} used, {sc["excluded"]} excluded (above a validated feasible makespan), '
           f'{sc["unverifiable"]} not verifiable, {sc["outside"]} outside the analyzed instances, '
           f'{sc["wrong_fields"]} naming another cell, {sc["other_data"]} of other data.', '',
           f'Records written before {sc["stored_since"]} (file modification time) give no bound, '
           'since the window horizon of the constraint model was enlarged on 2026-07-07: '
           f'{sc["before_date"]} verified records left out (their validated schedules remain '
           f'feasible makespans); {sc["model_mismatch"]} records left out because the descriptor '
           'carries release dates or a per-machine setup multiplier.', '']
    for e in res['stored']['excluded_examples']:
        md.append(f'- excluded: {e}')
    for e in res['stored']['examples']['unverifiable']:
        md.append(f'- not verifiable: {e}')
    cp = res['campaign']
    md += ['', 'Timed comparison as launched: '
           + '; '.join(f'"{s}"' for s in cp['descriptions'])
           + (f' (read past: {", ".join(cp["read_past"])})' if cp['read_past'] else '') + '.', '']
    md += [f'- {x}' for x in cp['lines']] or ['- no 600 s cell of a rule-started search']
    if res['missing']:
        md += ['', f'Incomplete inputs ({len(res["missing"])}):', '']
        md += [f'- {m}' for m in res['missing'][:30]]
    for size, S in res['per'].items():
        st = res['station_types'][size]
        md += ['', f'## {size} ({S["n"]} instances per combination)', '',
               f'Station types: {st["one_type"]} of {st["ops"]} operations have their eligible '
               f'machines in one station type, {st["whole_type"]} the whole type; the eligible '
               f'sets partition the operations on {st["partition_instances"]} of '
               f'{st["instances"]} instances.', '',
               '| comb. | ' + ' | '.join(METHOD_TEXT[m] for m in METHODS) + ' |',
               '|---|' + '---|' * len(METHODS)]
        for c in R8 + ('All',):
            cells = []
            for m in METHODS:
                g = S['gaps'].get(m, {}).get(c)
                cells.append('-' if g is None else f'{_ci(g)} (n {g["n"]})')
            md.append(f'| {"pooled" if c == "All" else c} | ' + ' | '.join(cells) + ' |')
        if S['matched']:
            md += ['', 'Policy on the instances of each search (pooled; per combination in the '
                   'JSON):', '']
            for m, g in S['matched'].items():
                x, y = g.get('All'), S['gaps'][m].get('All')
                if x and y:
                    md.append(f'- instances of the {METHOD_TEXT[m]} (n {x["n"]}): policy {_ci(x)}, '
                              f'{SEARCH_TEXT[m]} {_ci(y)}')
        md += ['', '| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % '
               '| 100 LB / best known: chain / load / stored / reference |',
               '|---|---|---|---|']
        for c in R8 + ('All',):
            sh, ob = S['shares'][c], S['over_best'][c]
            lg = ' / '.join(f'{sh[b]["largest"]:.1f}' for b in BOUNDS)
            al = ' / '.join(f'{sh[b]["alone"]:.1f}' for b in BOUNDS)
            ov = ' / '.join('-' if ob[b]['mean'] is None else f'{ob[b]["mean"]:.1f}'
                            for b in BOUNDS + ('reference',))
            md.append(f'| {"pooled" if c == "All" else c} | {lg} | {al} | {ov} |')
        md += ['', f'Stored solver bounds: {S["stored"]["used"]} used on {S["stored"]["cells"]} '
               f'cells, {S["stored"]["excluded"]} excluded, {S["stored"]["before_date"]} written '
               f'before {sc["stored_since"]} not used.']
        if S['stored']['folders']:
            md += ['', '| folder of the records used | records | budget_s (records) | warm start '
                   '| verified by | written, oldest to newest | cells where it gives the '
                   'reference |', '|---|---|---|---|---|---|---|']
            for f, x in S['stored']['folders'].items():
                md.append(f'| {f} | {x["records"]} | '
                          + ', '.join(f'{b} s ({k})' for b, k in x['budgets'].items())
                          + f' | {x["warm_start"]} of {x["records"]} | '
                          + ', '.join(f'{k} {v}' for k, v in x['verified_by'].items())
                          + f' | {x["oldest"]} to {x["newest"]} | {x["reference_cells"]} |')
        o = S['optimal']['All']
        per = ', '.join(f'{c} {S["optimal"][c]["count"]}' for c in R8 if S['optimal'][c]['count'])
        md += ['', f'Cells proved optimal (best known makespan equal to the reference bound): '
               f'{o["count"]} of {o["cells"]}' + (f' ({per})' if per else '')
               + ('; the reference is attained there by: '
                  + ', '.join(f'{BOUND_TEXT[b]} {o["attained"][b]}' for b in BOUNDS)
                  + ' (ties counted for each)' if o['count'] else '') + '.']
    return md


# ----------------------------------------------------------------- table
# methods printed with their interval; the other columns print the mean alone. All five fit the
# single-column text width at \footnotesize; ('policy', 'memetic', 'solver') is the narrower form.
TABLE_INTERVALS = ('policy', 'specialist', 'best_rule', 'memetic', 'solver')
TABLE_WORD = {'memetic': 'memetic algorithm', 'solver': 'solver'}
TABLE_NAME = {'policy': 'policy', 'specialist': 'specialists', 'best_rule': 'best dispatching rule',
              'memetic': 'memetic algorithm', 'solver': 'solver'}
# three heading rows per column: the method columns, then the bound ratio and the optimal count
TABLE_HEAD = {'policy': ('', '', 'policy'), 'specialist': ('', '', 'specialists'),
              'best_rule': ('', 'best dispatching', 'rule'),
              'memetic': ('', 'memetic', 'algorithm'), 'solver': ('', '', 'solver'),
              'ratio': ('reference', r'bound, \% of', 'best known'),
              'optimal': ('cells', 'proved', 'optimal')}


def tex_num(x):
    """A number in math mode with a math minus sign, one decimal (f1)."""
    return f'${f1(x)}$'


def table_cell(g, m, n_row=None):
    """mean [lo, hi] (math numbers) for a method of TABLE_INTERVALS, the mean alone otherwise,
    \\textendash without a value; ($n = k$) when the method rests on other instances than the
    row (n_row)."""
    if g is None:
        return r'\textendash'
    s = tex_num(g['mean'])
    if m in TABLE_INTERVALS:
        s += f' [{tex_num(g["lo"])}, {tex_num(g["hi"])}]'
    if n_row is not None and g['n'] != n_row:
        s += f' ($n = {g["n"]}$)'
    return s


def _and(words):
    return words[0] if len(words) == 1 else ', '.join(words[:-1]) + ' and ' + words[-1]


def _n_text(values):
    v = sorted(set(values))
    return f'{v[0]}' if len(v) == 1 else f'{v[0]} to {v[-1]}'


def table_lines(res, partial_line=None):
    """The tabular of the gaps to the reference bound: the training size per combination and
    pooled, then the other test sets pooled; no bold, since the columns of a row do not rest on
    the same instances."""
    cols = len(COMB_METHODS) + 3
    out = []
    if partial_line:
        out.append(f'% {partial_line}')
    out += [f'% Generated by {GENERATOR} -- do not hand-edit',
            f'% {setting_words(res)}. Gap to the reference bound in %, mean over the instances '
            f'with the 95% interval of stats_family.instance_boot (B = {res["boot"]}) for the '
            + _and([TABLE_NAME[m] for m in COMB_METHODS if m in TABLE_INTERVALS])
            + (('; the mean alone for the '
                + _and([TABLE_NAME[m] for m in COMB_METHODS if m not in TABLE_INTERVALS]))
               if set(COMB_METHODS) - set(TABLE_INTERVALS) else '')
            + '. Policy: the run with the '
            f'lowest validation score (training seed {res["policy_seed"]}); specialists: mean '
            'over the training seeds; solver: CP-SAT, configuration '
            f'{res["solver_arm"]}; searches at {res["budget_s"]} s. No bold: the columns of a '
            'row do not rest on the same instances.' + (' DRY RUN on synthetic arrays.'
                                                         if res['dry'] else ''),
            r'\begin{tabular}{@{}l ' + ' '.join('l' if m in TABLE_INTERVALS else 'r'
                                                   for m in COMB_METHODS) + r' rr@{}}',
            r'\toprule']
    nm = len(COMB_METHODS)
    out += [f' & \\multicolumn{{{nm}}}{{c}}{{gap to the reference bound (\\%)}} & '
            + ' & '.join(TABLE_HEAD[m][0] for m in ('ratio', 'optimal')) + r' \\',
            f'\\cmidrule(lr){{2-{nm + 1}}}']
    for k in (1, 2):
        out.append(('combination' if k == 2 else '') + ' & '
                   + ' & '.join(TABLE_HEAD[m][k] for m in COMB_METHODS + ('ratio', 'optimal'))
                   + r' \\')
    T = res['per'].get(TRAIN_SET)
    if T is not None:
        learned = [T['gaps'][m][c]['n'] for m in ('policy', 'specialist', 'best_rule')
                   for c in R8 + ('All',) if c in T['gaps'].get(m, {})]
        extra = []
        for m in ('memetic', 'solver'):
            ns = [g['n'] for g in T['gaps'].get(m, {}).values()]
            if ns:
                extra.append(f'{TABLE_WORD[m]} $n = {_n_text(ns)}$')
        words = (f'{int(TRAIN_SET.split("x")[0])} modules, $n = {_n_text(learned or [T["n"]])}$ '
                 'instances per combination' + (', ' + ', '.join(extra) if extra else ''))
        out += [r'\midrule', f'\\multicolumn{{{cols}}}{{@{{}}l}}{{\\textit{{{words}}}}} \\\\']
        for c in R8 + ('All',):
            cells = [table_cell(T['gaps'].get(m, {}).get(c), m) for m in COMB_METHODS]
            ob = T['over_best'][c]['reference']['mean']
            cells += [r'\textendash' if ob is None else tex_num(ob),
                      f'{T["optimal"][c]["count"]}']
            head = 'pooled' if c == 'All' else f'${c}$'
            out.append(f'{head} & ' + ' & '.join(cells) + r' \\')
    others = [s for s in res['per'] if s != TRAIN_SET]
    if others:
        n_rows = {s: (res['per'][s]['gaps']['policy']['All']['n']
                      if 'All' in res['per'][s]['gaps'].get('policy', {}) else res['per'][s]['n'])
                  for s in others}
        mods = [str(int(s.split('x')[0])) for s in others]
        words = ('pooled over the eight combinations, '
                 + _and([f'${n_rows[s]}$' for s in others]) + ' instances at '
                 + _and(mods) + ' modules')
        out += [r'\midrule', f'\\multicolumn{{{cols}}}{{@{{}}l}}{{\\textit{{{words}}}}} \\\\']
        for size, mod in zip(others, mods):
            S = res['per'][size]
            cells = [table_cell(S['gaps'].get(m, {}).get('All'), m, n_rows[size])
                     for m in COMB_METHODS]
            ob = S['over_best']['All']['reference']['mean']
            cells += [r'\textendash' if ob is None else tex_num(ob),
                      f'{S["optimal"]["All"]["count"]}']
            out.append(f'{mod} modules & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    return out


# ----------------------------------------------------------------- outputs
def outputs(out_dir=None):
    if out_dir is None:
        return {k: REG.out(v) for k, v in OUT_FILES.items()}
    return {k: os.path.join(out_dir, os.path.basename(v)) for k, v in OUT_FILES.items()}


def real_paths():
    roots = ['']
    if os.environ.get('PACT_OUT') not in (None, '', '.'):
        roots.append(os.environ['PACT_OUT'])
    return {os.path.abspath(os.path.join(ROOT, r, v)) for r in roots for v in OUT_FILES.values()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def run(partial=False, out_dir=None, boot=SF.B_DEFAULT):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, outs, md)."""
    check_setting()
    parse_campaigns(INPUTS.campaigns)          # refuses before any output is touched
    if partial:
        if not out_dir:
            raise Refused('--partial writes to the folder given by --out; give --out DIR')
        out_dir = os.path.abspath(out_dir)
        if {os.path.abspath(p) for p in outputs(out_dir).values()} & real_paths():
            raise Refused(f'--out {out_dir} is a folder of the real outputs; a partial run '
                          'writes elsewhere')
        os.makedirs(out_dir, exist_ok=True)
        outs = outputs(out_dir)
    else:
        if out_dir:
            raise Refused('--out is for --partial only; the full run writes through '
                          'arm_registry.out under PACT_OUT')
        if os.environ.get('PACT_OUT') is None:
            raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
        outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(partial, boot)
        note = None
        if partial:
            note = ('PARTIAL: ' + (f'{len(res["missing"])} inputs incomplete'
                                   if res['missing'] else 'no input is missing')
                    + '; not for the manuscript')
        lines = macro_lines(res, note)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(jsonable(res), f, indent=1)
        md = markdown(res, note)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
        with open(outs['table'], 'w') as f:
            f.write('\n'.join(table_lines(res, note)))
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, outs, md


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    seeds = check_setting()
    parsed = parse_campaigns(INPUTS.campaigns)
    why = []
    try:
        sel, _ = INPUTS.select_policy()
        INPUTS.config(sel)
        print(f'p120: policy run of seed {sel} (lowest validation score), configuration read')
    except (Refused, IB.Refused, SystemExit, REG.MissingResult, OSError, ValueError) as e:
        print(f'p120: the selected policy run or its configuration is unavailable: {e}')
        return 1
    miss = []
    sizes = []
    for size in INPUTS.sets:
        try:
            names, idx = analysis_instances(size)
        except Refused as e:
            why.append(str(e))
            continue
        sizes.append(size)
        _, m = method_columns(size, len(names), seeds, sel)
        miss += m
    try:
        _, counts, arm, segs = INPUTS.timed(parsed, sel)
    except Refused as e:
        why.append(str(e))
        counts, arm, segs = {}, None, []
    miss = why + expected_missing({}, miss, counts)
    for p in parsed:
        print(f'p120: campaign "{p["spec"]}"'
              + (f' (read past: {", ".join(p["ignored"])})' if p['ignored'] else ''))
    for x in segment_lines(segs, counts):
        print(f'p120:   {x}')
    print(f'p120: test sets {", ".join(sizes)}; CP-SAT configuration {arm}; '
          + ('every input present' if not miss else f'{len(miss)} inputs incomplete'))
    for m in miss[:8]:
        print(f'p120:   {m}')
    if len(miss) > 8:
        print(f'p120:   ... and {len(miss) - 8} more')
    return 0 if not miss else 1


# ----------------------------------------------------------------- dry mode
DRY_SETS = {'10x25': ('Ten', 0), '50x25n30': ('Fifty', 3)}
DRY_LIMIT = 3
DRY_BOOT = 200
DRY_SEED = 304
# planted gaps in %: the policy seeds 301..305 at 10..14 (mean 12), the specialists at 7..11
# (mean 9), the best of the four rules at 20 (the others above it), the searches at 4 and 6
DRY_GAP = dict(policy=13.0, policy_mean=12.0, specialist=9.0, best_rule=20.0, memetic=4.0,
               solver=6.0)
# the dry campaign: CP-SAT on two of the three analysis instances at 10x25, a token at another
# budget and tokens of other searches that the analysis reads past
DRY_CAMPAIGNS = ('50x25n30 = memetic_rule@600 cpsat_rule@600 memetic_policy@600 policy_sample@120',
                 '10x25 = memetic_rule@600 cpsat_rule@600:0-1 memetic_rule@60:30-31')
DRY_SOLVER = 'lns'
# instances each pooled mean rests on in the dry run
DRY_N = {('10x25', 'memetic'): 3, ('10x25', 'solver'): 2, ('50x25n30', 'memetic'): 3,
         ('50x25n30', 'solver'): 3}


def dry_paths(tree):
    return dict(PACT_ENV=ENV, PACT_SEEDS='301-305', PACT_RESULTS=os.path.join(tree, 'results'),
                PACT_OUT=os.path.join(tree, 'out'))


def _dispatch_schedule(jl, pt, desc):
    """A feasible schedule of the reference simulator: jobs in turn, each operation on its
    fastest eligible machine."""
    left = list(np.asarray(jl, dtype=int))
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    nxt = list(first)
    acts = []
    while any(left):
        for j in range(len(left)):
            if left[j]:
                o = nxt[j]
                row = np.where(pt[o] > 0, pt[o], np.inf)
                acts.append((j, int(np.argmin(row))))
                nxt[j] += 1
                left[j] -= 1
    s = simulate(jl, pt, desc, acts)
    return s['op_st'], s['op_ct'], s['op_mch'], float(s['makespan'])


def build_synthetic(tree):
    """Records under <tree>/or_solution, arrays under <tree>/results (PACT_RESULTS) and the
    makespans of both searches on every analysis cell of the dry sets, built from the bounds of
    the real descriptors. Returns ({(token name, size, combo, index): makespan}, the planted
    reference bound of every cell)."""
    seeds = check_setting()
    sroot = os.path.join(tree, 'or_solution')
    cfg = {}
    ref, table = {}, {}
    for size in INPUTS.sets:
        names, idx = analysis_instances(size)
        n = len(names)
        alias_src = {v: k for k, v in ALIAS.items()}.get(size)
        folder = f'{alias_src}_dry600s' if alias_src else f'{size}_dry600s'
        lbs = {}
        for i, name in idx:
            for c in R8:
                jl, pt, meta, desc, _ = cell_data(size, c, name)
                b1 = instance_bound(size, c, name, cfg)
                b2 = machine_load_bound(jl, pt, desc, meta['mch_type'])['value']
                st, ct, mch, ms = _dispatch_schedule(jl, pt, desc)
                base = max(b1, b2)
                k = idx.index((i, name))
                os.makedirs(f'{sroot}/{folder}/{c}', exist_ok=True)
                if k == 0:
                    # a verified schedule whose bound is the largest of the three
                    stored = min(math.floor(base) + 3.0, ms)
                    rec = dict(instance=name, regime=c, budget_s=600, status='FEASIBLE',
                               objective=ms, bound=stored, validated=True, violations=[],
                               assigned_mch=[int(x) for x in mch], op_st=[float(x) for x in st],
                               op_ct=[float(x) for x in ct])
                    json.dump(rec, open(f'{sroot}/{folder}/{c}/{name}.json', 'w'))
                    base = max(base, stored)
                elif k == 1:
                    # a verified schedule with a bound above every feasible makespan: excluded
                    rec = dict(instance=name, regime=c, budget_s=600, status='FEASIBLE',
                               objective=ms, bound=ms + 50.0, validated=True, violations=[],
                               assigned_mch=[int(x) for x in mch], op_st=[float(x) for x in st],
                               op_ct=[float(x) for x in ct])
                    json.dump(rec, open(f'{sroot}/{folder}/{c}/{name}.json', 'w'))
                else:
                    # a bound without a schedule or a warm start: not verifiable
                    rec = dict(instance=name, regime=c, budget_s=600, status='UNKNOWN',
                               objective=None, bound=10 * ms)
                    json.dump(rec, open(f'{sroot}/{folder}/{c}/{name}.json', 'w'))
                lbs[(c, i)] = base
                ref[(size, c, i)] = base
        # arrays: every descriptor row (rows outside the analysis hold large makespans)
        folder_r = REG.res_dir(size)
        os.makedirs(folder_r, exist_ok=True)
        for c in R8:
            lb = np.full(n, 1e5)
            for (cc, i), v in lbs.items():
                if cc == c:
                    lb[i] = v
            for k, s in enumerate(seeds):
                g = DRY_GAP['policy_mean'] + (s - 303)
                np.save(REG.result_path('released', None, s, size, c),
                        np.stack([lb * (1 + g / 100.0), np.zeros(n)], axis=1))
                if size in SPECIALIST_SETS:
                    g = DRY_GAP['specialist'] + (s - 303)
                    np.save(REG.result_path('specialist', c, s, size, c),
                            np.stack([lb * (1 + g / 100.0), np.zeros(n)], axis=1))
            for q, extra in zip(F88.RULES, (0.0, 1.0, 2.0, 3.0)):
                np.save(f'{REG.rule_dir(size)}/Result_{q}_{c}.npy',
                        np.stack([lb * (1 + (DRY_GAP['best_rule'] + extra) / 100.0),
                                  np.zeros(n)], axis=1))
            for arm, m in TIMED_ARMS.items():
                for i, name in idx:
                    table[(arm, size, c, i)] = lbs[(c, i)] * (1 + DRY_GAP[m] / 100.0)
    return table, ref


def synthetic_timed(table):
    """The reader of the dry mode: the launched cells of the planner's campaigns (launched(),
    the same parser and planner as the real reader), a cell of table valid with its makespan,
    one whose value is None missing, one outside table (an instance outside the analysis) valid
    without a makespan."""
    def read(parsed, seed):
        camps, segs = launched(parsed, seed, DRY_SOLVER)
        cells, counts = {}, {}
        for camp in camps:
            for seg in camp.segments:
                for size, insts in seg.by_size.items():
                    for c in camp.combos:
                        cnt = counts.setdefault(f'{seg.arm}|{size}|{c}', {})
                        for i in insts:
                            key = (seg.arm, size, c, i)
                            if key in cells:
                                raise Refused(f'{seg.label}/{size}/{c}/{i:03d}: the cell appears '
                                              'in two campaigns')
                            ms = table.get(key, False)
                            state = 'missing' if ms is None else 'valid'
                            cells[key] = dict(state=state,
                                              final_makespan=ms if ms else None,
                                              record=f'synthetic/{seg.arm}/{size}/{c}/{i:03d}')
                            cnt[state] = cnt.get(state, 0) + 1
        return cells, counts, DRY_SOLVER, segs
    return read


@contextlib.contextmanager
def dry_context(tree, built=None):
    """The settings of the dry mode on the folder tree, outside the repository, with the
    campaign description DRY_CAMPAIGNS. Builds the synthetic inputs unless built=(makespans of
    the searches, planted bounds) is given (a copy of a tree built before). Yields (makespans of
    the searches, planted bounds)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    os.makedirs(tree, exist_ok=True)
    with environment(**dry_paths(tree)), inputs(sets=dict(DRY_SETS), limit=DRY_LIMIT,
                                                 solver_root=os.path.join(tree, 'or_solution'),
                                                 record_roots=(os.path.join(tree, 'none'),
                                                               os.path.join(tree, 'none')),
                                                 select_policy=lambda: (DRY_SEED, {}),
                                                 config=lambda seed: {}, dry=True,
                                                 campaigns=list(DRY_CAMPAIGNS)):
        table, ref = built if built is not None else build_synthetic(tree)
        with inputs(timed=synthetic_timed(table)):
            yield table, ref


def dry_run(tree, boot=DRY_BOOT, partial=False, out_dir=None):
    with dry_context(tree) as (_, ref):
        res, names, outs, md = run(partial=partial, out_dir=out_dir, boot=boot)
    return res, names, outs, md, ref


def dry_ok(res, ref):
    """The planted gaps are recovered exactly, each search's mean rests on the instances of its
    launched token (DRY_N) and so does the policy's gap beside it, the reference equals the
    planted bound on every cell, one third of the stored records are used, one third excluded,
    one third unverifiable."""
    ok = True
    for size, S in res['per'].items():
        for m, g in DRY_GAP.items():
            if m == 'specialist' and size not in SPECIALIST_SETS:
                continue
            x = S['gaps'].get(m, {}).get('All')
            ok &= x is not None and abs(x['mean'] - g) < 1e-6
            if m in TIMED_ARMS.values() and x is not None:
                y = S['matched'].get(m, {}).get('All')
                ok &= x['n'] == DRY_N[(size, m)]
                ok &= y is not None and y['n'] == x['n'] and abs(y['mean'] - DRY_GAP['policy']) < 1e-6
    for r in res['cells']:
        ok &= abs(r['reference'] - ref[(r['size'], r['combo'], r['index'])]) < 1e-9
    n = len(res['cells'])
    sc = res['stored']['counts']
    ok &= sc['used'] == sc['excluded'] == sc['unverifiable'] == n // 3
    ok &= not res['self_check']['violations']
    return bool(ok)


# ----------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description='a lower-bound reference for the makespans')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--partial', action='store_true',
                    help='report what the complete inputs allow; writes to --out only')
    ap.add_argument('--out', default=None, help='output folder of --partial')
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary '
                         'folder, removed afterwards)')
    ap.add_argument('--campaign', action='append', default=None,
                    help='"<sizes> = <tokens>" of the timed comparison as launched, once per '
                         'launch, in the planner\'s token syntax name[@budget[:instances]] '
                         '(p94_timed_inference); required except with --dry. As launched: '
                         + ' '.join(f'--campaign "{s}"' for s in AS_LAUNCHED))
    ap.add_argument('--stored-since', default=STORED_SINCE, metavar='YYYY-MM-DD[THH:MM]',
                    help='stored solver bounds are read only from record files modified on or '
                         f'after this local date (default {STORED_SINCE}: the window horizon of '
                         'the constraint model was enlarged on 2026-07-07)')
    a = ap.parse_args(_ARGV if argv is None else argv)
    out_dir = os.path.abspath(a.out) if a.out else None
    os.chdir(ROOT)
    try:
        since_time(a.stored_since)
        INPUTS.stored_since = a.stored_since
        if a.dry is not None and a.campaign:
            raise Refused('--dry reads its own campaign description (DRY_CAMPAIGNS); drop '
                          '--campaign')
        if a.check:
            with inputs(campaigns=a.campaign):
                code = check()
            sys.exit(code)
        if a.dry is None:
            REG.banner()
            with inputs(campaigns=a.campaign):
                res, names, outs, md = run(a.partial, out_dir, a.boot)
            print('\n'.join(md))
            print(f'\n{len(names)} macros; wrote ' + ', '.join(outs.values()))
            return
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p120_dry_')
        try:
            res, names, outs, md, ref = dry_run(tree, boot=min(a.boot, DRY_BOOT))
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    ok = dry_ok(res, ref)
    if keep:
        print('\n'.join(md))
    print(f'p120 dry run on synthetic arrays {"passed" if ok else "FAILED"}: {len(names)} macros, '
          f'{len(res["cells"])} cells, pooled gaps '
          + ', '.join(f'{m} {res["per"]["10x25"]["gaps"][m]["All"]["mean"]:.3f}'
                      for m in DRY_GAP if m in res['per']['10x25']['gaps'])
          + f'; stored records used {res["stored"]["counts"]["used"]}, excluded '
            f'{res["stored"]["counts"]["excluded"]}, not verifiable '
            f'{res["stored"]["counts"]["unverifiable"]}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
