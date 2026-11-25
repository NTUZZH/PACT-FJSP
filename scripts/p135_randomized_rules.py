r"""P135: randomized dispatching rules within a time budget, on the eight cores of one slot.

The time-budget comparison gives the policy's sampled schedules (scripts/p97_policy_anytime.py)
eight single-thread workers on eight cores for the budget, while the dispatching rules enter
with one schedule each, the best of four. This runner gives the rules the same cores and the
same budget: a greedy randomized construction over the four rules, sampled on W workers until
the budget ends. One cell is one size, one combination and one instance.

METHOD
  Environment, mask and descriptor: those of the deterministic rules of the timed campaign.
  The worker loads the instance and its descriptor with p97.RuleRoller
  (p86_extend_scale_sets.load_instance and load_stored_descriptor, the attached setup
  convention) and builds, for every schedule, FJSPEnvFamily(family_mode=True,
  first_mask='exact', mask_ties='exact') with the instance and the descriptor and no type
  lists, as p86_extend_scale_sets.masked_rollout builds it; the exact mask is checked through
  env.mask_setting before every construction.
  Rules: p97.RULE_ORDER, (FIFO, MOR, SPT, MWKR), the rules of the best-of-four start schedule
  of the search arms; the cell's process refuses when p86_extend_scale_sets.RULE_ORDER differs.
  Scores. At each dispatch the legal pairs are those of ~env.dynamic_pair_mask[0], and each
  legal pair gets the score common_utils.masked_rule_action ranks it by:
    SPT   the processing time of the pair (env.candidate_pt), lower is better;
    FIFO  the ready time of the pair's job (env.candidate_free_time), lower is better;
    MOR   the operations left in the pair's job (env.op_match_job_left_op_nums), higher is
          better;
    MWKR  the work left in the pair's job (env.op_match_job_remain_work), higher is better.
  Restricted candidate list. With b the best score and r = max - min the range of the scores
  of the legal pairs at that dispatch, the list holds the candidates whose score is within
  alpha * r of b: score - b <= alpha r when lower is better, b - score <= alpha r when higher
  is better (the distance to b is computed first, so alpha = 0 keeps exactly the scores equal
  to b and alpha = 1 keeps every legal candidate). It always holds the best-scored
  candidates, so it is never empty.
  Choice. SPT draws one pair of the list uniformly. FIFO, MOR and MWKR score a pair by its job,
  so their list is the list of the jobs whose score is within the bound: one job is drawn
  uniformly, and the job goes to its legal machine with the shortest processing time, equal
  times drawn uniformly, which is the second stage of masked_rule_action. With alpha = 0 the
  list holds the best-scored pairs or jobs only and the draws are those of masked_rule_action
  in the same order, so with numpy's global stream seeded 50 (the seed of masked_rollout) as
  the random source the construction returns masked_rollout's schedule
  (tests/test_p135_randomized_rules.py; `check` repeats it on the cells already run).
  Samples. One sample is one complete schedule from one rule and one alpha. Sample j of worker
  w (j = 0, 1, ...) uses rule RULE_ORDER[(w + j) mod 4]. Sample 0 of worker w, w = 0 to 3, is
  the deterministic rule RULE_ORDER[w], rolled out by p86_extend_scale_sets.masked_rollout
  itself (numpy seed 50 right before it), so the four deterministic rules run in parallel and
  the arm holds the best-of-four start schedule of the search arms after one rollout per
  worker. With fewer than four workers, worker 0 takes the rules of the missing workers,
  RULE_ORDER[W] to RULE_ORDER[3], as its samples 1 to 4 - W, so every rule is produced once
  (deterministic_rule, deterministic_layout). Every other sample draws alpha uniformly from
  ALPHAS = (0.05, 0.1, 0.2, 0.3) and then constructs a schedule.
  Random numbers. One generator per worker, numpy.random.default_rng([seed, w, cell hash]),
  seed 1 by default (--seed, recorded), cell hash the first 32 bits of the SHA-256 of
  '<size>:<combination>:instance_<iii>' (recorded). In the order of the worker's randomized
  samples the generator draws the sample's alpha and then every choice of its construction.
  The deterministic samples draw from numpy's global stream only.

PROTOCOL (the policy arm of p97, by import: p97.start_workers, machine_fields, machine_reason,
validate_real, write_record, plan_state, common_refusals, idle_sample, campaign_loop)
  W worker processes (default 8), worker w pinned to core w of the runner's affinity mask
  (the launcher sets the mask with taskset; the runner never widens it), one thread each
  (OMP, MKL and OpenBLAS threads 1, no GPU).
  Clock: time.monotonic(); t in seconds from the start of the process that runs the cell (the
  campaign takes the time just before it starts the cell's process, so the interpreter start,
  the imports and the instance load are inside t, as p97 counts them); t_resident from the
  moment the worker's instance was loaded. The budget (default 60 s) ends at t = budget. The
  clock is checked at every decision of a randomized construction, and a construction still
  running at the budget is discarded; a deterministic rollout is started only before the
  budget, and its schedule is discarded when it ends after the budget.
  Validation: every completed schedule is validated by its worker after its rollout window
  (outside the window) with p97.validate_real (constraint_family.cpsat_arms.validate_all:
  validate_strict and the C6 check, the validators of the CP-SAT runner), and every schedule
  of the best-so-far trajectory again by the cell's process before the record is written.

RECORD  test_results/FAMILY/rndrules_p135/<size>_rnd<budget>s_w<W>_s<seed>/<combination>/
  instance_<i>.json (under --root when given), the best schedule beside it as
  instance_<i>.schedule.npz (amch, st, ct, ms). Fields: the cell, arm randomized_rules, the
  method, rule_order, alphas, seed, cell_hash, the rng rule, budget_s, budget_clock wall, the
  clock, workers; 'schedules' (columns in 'schedules_columns': worker, index, kind
  deterministic or randomized, rule, alpha, t, t_resident, makespan, CPU seconds of the
  worker) for every completed schedule; 'anytime', every improvement of the best schedule so
  far over all workers, in the four-column format of p97 [t - search_start_wall_s, t, CPU
  seconds of the producing worker, makespan], read by baselines.memetic.records.incumbent_at
  on the wall clock from the process start; 'anytime_from' (worker, index of each
  improvement); 'deterministic' (per rule the worker and sample index that produced it, its
  makespan, t and t_resident), 'deterministic_layout' (the worker and sample index each rule
  must come from), n_deterministic, t_deterministic_all (the time t at which all four exist,
  from then on the best of four is held) and t_best_rule_reached (the first t at which the
  best schedule so far is at or below the best of the four);
  n_schedules, n_randomized, per rule the number of schedules and their best makespan; the
  validation counts and violations; and the machine evidence of p97.machine_fields: per worker
  its core, affinity, CPU and wall seconds over its rollout window, cpu_share = CPU / wall,
  peak memory and thread settings; the mean and the smallest share, the affinity mask, the
  load average at start and end, the core sample before the cell (idle_before, with the
  processes seen on the slot's cores), the commit, development_run; the stored rule start
  schedule it is compared with; valid and its reason.

VALIDITY (classify)
  p97.machine_reason: validated (every schedule), not a development run, the slot's cores idle
  before the cell, one worker per core and every core inside the mask, each worker pinned to
  its core, every worker's cpu_share over its rollout window at least 0.9; the wall budget
  clock; exactly four deterministic schedules, one per rule, each from the worker and sample
  of deterministic_layout (sample 0 of workers 0 to 3; worker 0 for the rules of missing
  workers), each within the budget, each makespan equal to rule_ms of
  or_solution/FAMILY/warmstart_cache_xmask_<size>/<combination>/instance_<i>.npz
  (p97.rule_cache_entry), the start schedule of the search arms; the best makespan at or below
  the cache's best rule. A cell is valid under the same machine rule as a cell of p97.

CAMPAIGN (run; the launcher is scripts/run_randomized_rules.sh)
  Cells: --sizes (default the production sets of timed_campaign_plan.SIZES, 50x25n30 and
  80x25n30), the eight combinations, --instances (default p124_memetic_parallel.INSTANCES,
  3-11, the nine instances per combination that p94 and p124 read at 600 s); every instance
  must be an evidence instance of timed_campaign_plan.DEFAULT_INSTANCES (000-002 selected the
  search configurations and stay out). 2 x 8 x 9 = 144 cells, in the order size, instance,
  combination. --share i/n runs the cells whose position k in that order has k mod n = i - 1,
  so that two slots split the cells (1/2 and 2/2). Before the first cell the slot's cores are
  sampled as the timed campaign's prelaunch samples them (timed_campaign_plan.busy_on: a
  thread above 5% of a core over 2 s, or a process above 5% lifetime share that last ran
  there); the launch note lists every process above 5% with its core and the start is refused
  when the slot is not idle. Then p97.campaign_loop: before each cell the core sample of p97
  (refused above 5% of a core), one process per cell, the record classified, an invalid record
  moved to <combination>/invalid/ (its cell runs again at the next start; a cell with
  --max_invalid such files is held), two cells in a row without a valid record stop the run;
  the stop file train_log/RNDRULE_HOLD ends the run after the current cell; launch notes go to
  train_log/randomized_rules_launch_notes.txt. A record of another plan (budget, workers,
  seed, alphas, rule order) refuses the start. The run exits 1 when a cell of its share is
  without a valid record at the end (unless it was held).
  --development lifts the refusals of a busy core and of more workers than cores, needs --root
  outside the repository, and marks every record development_run, never valid.

PLAN (plan; reads only): the cells with their state (valid, missing, invalid, held, foreign),
  the cells per share, the expected wall-clock per slot (cells to run x (budget + OVERHEAD_S))
  and the command lines for two slots (cores 8-15 and 16-23). --strict: exit 1 when a start
  must be refused. --slot: the refusals for that core list.

CHECK (check; reads only, one thread): completeness (valid cells of the plan), validity by
  slot (the affinity of each record), and on every cell with a record (or the first
  --alpha0-cells of them) the four rules rebuilt by the randomized construction at alpha = 0
  with numpy's global stream seeded 50, compared with the stored rule_ms of the rule cache and
  with the record's deterministic makespans. Exit 1 when a cell lacks a valid record or a
  makespan differs.

ANALYSIS (infer)
  Reads the valid cells of this arm and, read only through p124's own readers
  (p124_memetic_parallel.read_inputs with p124.check_options and p124.selections: the policy's
  greedy arrays of the selected training run, the rule arrays, the rule and policy records of
  p97, the eight memetic runs memetic_rule@600 (seeds 1-8), CP-SAT cpsat_rule@600 with the
  pilot's configuration), on the request clock at 10, 30 and 60 s: this arm at its own t (the
  best schedule so far, incumbent_at on the wall clock from the process start), the policy's
  greedy schedule from o_pol on, the best of the four deterministic rules from o_rule on, the
  policy's sampled schedules at their own t, a search started from the rules at run time
  t - o_rule. Every difference is p94's: 100 (first - second) / second per cell, p94.compare_at
  (p94.stat_instance: stats_family.instance_boot, the cluster bootstrap over base instances, a
  fresh default_rng(7) per quantity, B = --boot; stats_family.direction on the unrounded
  interval), on the cells where both methods have a record; per size, per combination, Pool
  (the eight combinations), NoSetup and Setup:
    (i)   rnd_vs_pol   the randomized rules minus the policy's greedy schedule, in percent of
                       the greedy schedule (the form of the columns of the decision table)
    (ii)  smp_vs_rnd   the policy's sampled schedules minus the randomized rules
    (iii) rnd_vs_rule  the randomized rules minus the best deterministic rule
    (iv)  mem_vs_rnd   the best of the eight memetic runs started from the rule schedule
                       (p124.best_of, all eight valid) minus the randomized rules
          cp_vs_rnd    CP-SAT started from the rule schedule minus the randomized rules
    (v)   schedules    the mean number of schedules per cell of the randomized rules and of
                       the policy's sampled arm completed within 60 s (rows of its record
                       with t <= 60 s), on the cells with both records
    (vi)  (i) to (iv) at 10 s and 30 s: mean and interval, no direction and no verdict.
  At 60 s: interval and direction per entry; for (ii) per combination the verdict of p94's
  rule of row Sel (p94.verdict_at: one-sided bootstrap p from the draws of the interval, Holm
  over the eight combinations per direction, p94.selected_verdict; ahead = the policy's
  sampled schedules are shorter), with the counts. A completeness count: valid cells of the
  plan.
  Outputs (arm_registry.out under PACT_OUT; the full run needs every cell valid and every
  stored input present, and refuses otherwise):
    results/randomized_rules.json, results/randomized_rules.md,
    manuscript/macros_randomized_rules.tex (macro_io.emit; names Rnd plus letters, one comment
    per macro; the prefix Rnd must be used by no other .tex file under manuscript/)
  --out DIR (outside the repository's paper, result and record folders and outside PACT_OUT)
  writes the same three files under DIR, marked partial when a cell or an input is missing.

Usage (repository root):
  SLOT=8-15 SHARE=1/2 scripts/run_randomized_rules.sh        (the launcher; DRY=1: the plan)
  PACT_ENV=exact python scripts/p135_randomized_rules.py plan [--strict --slot 8-15]
  taskset -c 8-15 env PACT_ENV=exact python scripts/p135_randomized_rules.py run --share 1/2
  PACT_ENV=exact python scripts/p135_randomized_rules.py check [--alpha0-cells N]
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p135_randomized_rules.py infer
  ... infer --out <folder outside the repository>
  Development (another core count, a busy core): run --development --root <folder outside
  the repository> [--workers 1 --budget 5 --sizes 50x25n30 --combos N --instances 3].
"""
import time

_T_MODULE = time.monotonic()      # first statement: the start of a cell's process

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMPY_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
if __name__ == '__main__':
    # one numpy/OpenMP thread and no GPU, before numpy is imported
    for _v in NUMPY_VARS:
        os.environ[_v] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
_CWD = os.getcwd()                # relative --root and --out are read against it

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402  (removes the --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
import p97_policy_anytime as P97  # noqa: E402  (p75_deploy_cost empties sys.argv at import)
import timed_campaign_plan as TCP  # noqa: E402
from baselines.memetic.records import incumbent_at  # noqa: E402

ARM = 'randomized_rules'
GENERATOR = 'scripts/p135_randomized_rules.py'
RULE_ORDER = tuple(P97.RULE_ORDER)          # the best-of-four rules, FIFO, MOR, SPT, MWKR
ALPHAS = (0.05, 0.1, 0.2, 0.3)
LOWER_IS_BETTER = {'FIFO': True, 'MOR': False, 'SPT': True, 'MWKR': False}
RULE_NUMPY_SEED = 50                        # the numpy seed of p86 masked_rollout
METHOD = ('greedy randomized construction over the dispatching rules FIFO, MOR, SPT and MWKR '
          'under the exact mask: at each dispatch the candidates whose rule score is within '
          'alpha times the range of the scores of the legal pairs of the best score, one drawn '
          'uniformly (SPT: a pair; FIFO, MOR, MWKR: a job, which goes to its legal machine with '
          'the shortest processing time); worker w takes rule (w + j) mod 4 for its sample j '
          'and alpha uniformly from 0.05, 0.1, 0.2, 0.3; sample 0 of workers 0 to 3 is the '
          'deterministic rule FIFO, MOR, SPT, MWKR in turn (p86_extend_scale_sets.masked_rollout'
          '), worker 0 taking the rules of missing workers; W single-thread workers')
RNG_RULE = ('one generator per worker, numpy.random.default_rng([seed, worker, cell_hash]); in '
            'the order of the worker\'s randomized samples it draws the sample\'s alpha and then '
            'every choice of its construction; the deterministic samples draw from numpy\'s '
            'global stream (seed 50 before each, inside masked_rollout); cell_hash = first 32 '
            'bits of SHA-256 of "<size>:<combination>:instance_<iii>"')
DEFAULT_WORKERS = 8
DEFAULT_BUDGET = 60.0
DEFAULT_SEED = 1
OVERHEAD_S = 12.0       # slot seconds per cell beyond the budget: the core sample (3 s), the
#                         imports and instance load, the validation of every schedule, the
#                         record
MAX_INVALID = 2
RESULTS = os.path.join('test_results', 'FAMILY', 'rndrules_p135')
HOLD = os.path.join('train_log', 'RNDRULE_HOLD')
NOTES = os.path.join('train_log', 'randomized_rules_launch_notes.txt')
SLOTS = ('8-15', '16-23')
SIZES = tuple(TCP.SIZES)
COMBOS = tuple(REG.COMBOS)
SCHEDULE_COLUMNS = ['worker', 'index', 'kind', 'rule', 'alpha', 't', 't_resident', 'makespan',
                    'cpu_s_worker']
C = {c: k for k, c in enumerate(SCHEDULE_COLUMNS)}
ANYTIME_COLUMNS = list(P97.ANYTIME_COLUMNS)
VALIDATOR = P97.VALIDATOR
EPS = 1e-9

# analysis (infer)
PREFIX = 'Rnd'
BUDGETS = (10, 30, 60)
PRIMARY = 60
OUT_FILES = {'macros': 'manuscript/macros_randomized_rules.tex',
             'json': 'results/randomized_rules.json', 'md': 'results/randomized_rules.md'}
# (key, macro word, text): first-named minus second-named, in percent of the second
COMPARISONS = (
    ('rnd_vs_pol', 'RndVsPol', "the randomized rules minus the policy's greedy schedule, in "
     'percent of the greedy schedule (the form of the decision table)'),
    ('smp_vs_rnd', 'PolSmpVsRnd', "the policy's sampled schedules minus the randomized rules"),
    ('rnd_vs_rule', 'RndVsRule', 'the randomized rules minus the best of the four deterministic '
     'rules'),
    ('mem_vs_rnd', 'MemBestVsRnd', 'the best of the eight memetic runs started from the rule '
     'schedule minus the randomized rules'),
    ('cp_vs_rnd', 'CpRuleStartVsRnd', 'CP-SAT (8 workers) started from the rule schedule minus '
     'the randomized rules'))
VERDICT_KEY = 'smp_vs_rnd'


class Refusal(SystemExit):
    """A check failed before any rollout."""


# ---------------------------------------------------------------------------
# names, folders, cells
# ---------------------------------------------------------------------------

def cell_hash(size, combo, idx):
    """The first 32 bits of the SHA-256 of '<size>:<combination>:instance_<iii>'."""
    text = f'{size}:{combo}:{P97.instance_name(idx)}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def sample_rule(worker, index):
    """The rule of sample `index` of worker `worker`: RULE_ORDER[(worker + index) mod 4]."""
    return RULE_ORDER[(int(worker) + int(index)) % len(RULE_ORDER)]


def deterministic_rule(worker, index, workers):
    """The deterministic rule of sample `index` of worker `worker` in a cell of `workers`
    workers, or None when the sample is randomized. Sample 0 of worker w is RULE_ORDER[w] for
    w = 0 to 3; with W < 4 workers, worker 0 takes the rules of the missing workers,
    RULE_ORDER[W] to RULE_ORDER[3], as its samples 1 to 4 - W. Every rule is produced once."""
    w, j, n_w, n_r = int(worker), int(index), int(workers), len(RULE_ORDER)
    if n_w < 1 or not 0 <= w < n_w:
        raise ValueError(f'worker {w} of {n_w} workers')
    if j == 0 and w < n_r:
        return RULE_ORDER[w]
    if w == 0 and n_w < n_r and 1 <= j <= n_r - n_w:
        return RULE_ORDER[n_w + j - 1]
    return None


def deterministic_layout(workers):
    """{rule: [worker, index]} of the four deterministic samples of a cell of `workers`."""
    n_r = len(RULE_ORDER)
    out = {}
    for w in range(min(int(workers), n_r)):
        for j in range(n_r):
            r = deterministic_rule(w, j, workers)
            if r is not None:
                out[r] = [w, j]
    if sorted(out) != sorted(RULE_ORDER):
        raise ValueError(f'layout of {workers} workers misses a rule: {out}')
    return {r: out[r] for r in RULE_ORDER}


def folder(root, size, budget, workers, seed):
    return os.path.join(root, RESULTS,
                        f'{size}_rnd{P97.bnum(budget)}s_w{int(workers)}_s{int(seed)}')


def record_path(root, size, budget, workers, seed, combo, idx):
    return os.path.join(folder(root, size, budget, workers, seed), combo,
                        f'{P97.instance_name(idx)}.json')


def parse_share(text):
    m = re.fullmatch(r'\s*(\d+)\s*/\s*(\d+)\s*', str(text))
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        raise ValueError(f'share {text!r}: want i/n with 1 <= i <= n')
    return int(m.group(1)), int(m.group(2))


def in_share(position, share):
    i, n = share
    return position % n == i - 1


def default_instances():
    """The nine instances per combination that p94 and p124 read (p124's INSTANCES)."""
    import p124_memetic_parallel as P124
    return P124.INSTANCES


def evidence_refusals(sizes, inst):
    """Instances that are not evidence instances of timed_campaign_plan.DEFAULT_INSTANCES."""
    out = []
    for size in sizes:
        ok = set(TCP.parse_instances(TCP.DEFAULT_INSTANCES.get(size, TCP.OTHER_DEFAULT_INSTANCES)))
        bad = [i for i in inst if i not in ok]
        if bad:
            out.append(f'{size}: instances {TCP.compress(bad)} are not evidence instances '
                       f'({TCP.DEFAULT_INSTANCES.get(size, TCP.OTHER_DEFAULT_INSTANCES)})')
    return out


# ---------------------------------------------------------------------------
# the randomized construction
# ---------------------------------------------------------------------------

def exact_env(jl, pt, desc):
    """The environment of p86_extend_scale_sets.masked_rollout: FJSPEnvFamily under the
    exact mask stated as constructor options, the instance and its descriptor, no type
    lists; the mask is checked."""
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        first_mask='exact', mask_ties='exact')
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    if not (env.mask_setting('first_mask') == 'exact'
            and env.mask_setting('mask_ties') == 'exact' and env._first_mask):
        raise RuntimeError('environment is not under the exact mask')
    return env


def candidates(rule, env, alpha):
    """The restricted candidate list at the current dispatch: flat pair indices (job *
    machines + machine) for SPT, job indices for FIFO, MOR and MWKR, in increasing order.
    The scores are those of common_utils.masked_rule_action over the legal pairs; the list
    holds the candidates whose distance to the best score is at most alpha times the range
    of the scores."""
    legal = ~env.dynamic_pair_mask[0]
    if not legal.any():
        raise RuntimeError('the mask admits no pair')
    a = float(alpha)
    if rule == 'SPT':
        pt = env.candidate_pt[0]
        s = pt[legal]
        best, span = s.min(), s.max() - s.min()
        return np.where((legal & (pt - best <= a * span)).reshape(-1))[0]
    legal_jobs = np.where(legal.any(axis=1))[0]
    ops = env.candidate[0][legal_jobs]
    if rule == 'FIFO':
        key = env.candidate_free_time[0][legal_jobs]
    elif rule == 'MOR':
        key = env.op_match_job_left_op_nums[0][ops]
    elif rule == 'MWKR':
        key = env.op_match_job_remain_work[0][ops]
    else:
        raise ValueError(f'undefined rule {rule}')
    span = key.max() - key.min()
    gap = (key - key.min()) if LOWER_IS_BETTER[rule] else (key.max() - key)
    return legal_jobs[gap <= a * span]


def rcl_action(rule, env, alpha, rng):
    """One action: a uniform draw from the candidate list (SPT: the pair; FIFO, MOR, MWKR:
    the job, then its legal machine with the shortest processing time, equal times drawn
    uniformly, as masked_rule_action's second stage). rng: anything with choice(), a
    numpy Generator or numpy.random itself."""
    cand = candidates(rule, env, alpha)
    if rule == 'SPT':
        return int(rng.choice(cand))
    M = env.number_of_machines
    legal = ~env.dynamic_pair_mask[0]
    j = int(rng.choice(np.atleast_1d(cand).ravel()))
    pt = np.where(legal[j], env.candidate_pt[0][j], np.inf)
    return j * M + int(rng.choice(np.where(pt == pt.min())[0]))


def construct(rule, alpha, rng, jl, pt, desc, deadline=math.inf, clock=time.monotonic):
    """(machine per operation, completion times, makespan) of one schedule of the randomized
    rule, stepped as p86_extend_scale_sets.masked_rollout steps the deterministic rule; None
    when the clock reaches `deadline` before the schedule is complete."""
    env = exact_env(jl, pt, desc)
    M = pt.shape[1]
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        if clock() >= deadline:
            return None
        a = rcl_action(rule, env, alpha, rng)
        job, mch = a // M, a % M
        amch[env.candidate[0, job]] = mch
        env.step(np.array([a]))
    ct = np.asarray(env.true_op_ct[0], dtype=float).copy()
    return amch, ct, float(np.max(ct))


def deterministic_by_construction(jl, pt, desc, rules=RULE_ORDER):
    """{rule: (amch, ct, makespan)} of the construction at alpha = 0 with numpy's global
    stream seeded RULE_NUMPY_SEED before each rule, as masked_rollout seeds it."""
    out = {}
    for rule in rules:
        np.random.seed(RULE_NUMPY_SEED)
        out[rule] = construct(rule, 0.0, np.random, jl, pt, desc)
    return out


class RandomizedRoller(P97.RuleRoller):
    """The instance loaded once in a worker, as p97's rule arm loads it; deterministic(rule)
    is p86_extend_scale_sets.masked_rollout, randomized() the construction above,
    validate() p97.validate_real (the CP-SAT runner's validators)."""

    def deterministic(self, rule):
        ms, amch, ct = self.rollout(rule)
        return amch, ct, ms

    def randomized(self, rule, alpha, rng, deadline=math.inf, clock=time.monotonic):
        return construct(rule, alpha, rng, self.jl, self.pt, self.desc, deadline, clock)

    def validate(self, spec, amch, ct):
        return P97.validate_real(spec, amch, ct)


# ---------------------------------------------------------------------------
# the workers of one cell
# ---------------------------------------------------------------------------

def worker_loop(roller, worker, seed, chash, t0, budget, *, workers, clock=time.monotonic):
    """Samples of one worker of a cell of `workers` workers from now until t0 + budget (module
    docstring, METHOD). A sample that is incomplete or ends after the budget is discarded.
    Returns the rows (SCHEDULE_COLUMNS), the samples that improved this worker's best, every
    kept schedule (index, makespan, amch, ct) for the validation after the window, and the
    clock readings."""
    deadline = t0 + budget
    rng = np.random.default_rng([int(seed), int(worker), int(chash)])
    alphas = np.asarray(ALPHAS, dtype=float)
    t_ready = clock()
    rows, improving, kept = [], [], []
    best, j, discarded = math.inf, 0, False
    while clock() < deadline:
        det_rule = deterministic_rule(worker, j, workers)
        det = det_rule is not None
        rule = det_rule if det else sample_rule(worker, j)
        if det:
            alpha = 0.0
            res = roller.deterministic(rule)
        else:
            alpha = float(rng.choice(alphas))
            res = roller.randomized(rule, alpha, rng, deadline, clock)
        tc = clock()
        if res is None or tc > deadline:
            discarded = True
            break
        amch, ct, ms = res
        ms = float(ms)
        cpu = round(time.process_time(), 3)
        rows.append([int(worker), j, 'deterministic' if det else 'randomized', rule, alpha,
                     round(tc - t0, 4), round(tc - t_ready, 4), ms, cpu])
        kept.append((j, ms, np.asarray(amch), np.asarray(ct, dtype=float)))
        if ms < best:
            best = ms
            improving.append({'index': j, 'makespan': ms, 'amch': np.asarray(amch),
                              'ct': np.asarray(ct, dtype=float)})
        j += 1
    return dict(rows=rows, improving=improving, kept=kept, discarded=discarded,
                t_ready=t_ready, t_stop=clock(), n_started=j + int(discarded))


def check_schedules(roller, spec, kept):
    """[[index, violations]] of the kept schedules that fail the validators or whose
    makespan is not their latest completion."""
    bad = []
    for j, ms, amch, ct in kept:
        viol, sched = roller.validate(spec, amch, ct)
        if viol:
            bad.append([int(j), [str(v) for v in list(viol)[:3]]])
        elif abs(float(np.max(sched[2])) - ms) > 1e-9:
            bad.append([int(j), [f'makespan {ms:g} != latest completion '
                                 f'{float(np.max(sched[2])):g}']])
    return bad


def worker_main(spec, worker, core, q, factory):
    """One worker process: pin to its core, load, sample until the budget, validate every
    schedule after the window, report through q."""
    try:
        os.sched_setaffinity(0, {core})
        for v in NUMPY_VARS:
            os.environ[v] = '1'
        roller = factory(spec)
        rss_load = P97.maxrss_mb()
        c0 = time.process_time()
        out = worker_loop(roller, worker, spec['seed'], spec['cell_hash'], spec['t0'],
                          spec['budget'], workers=int(spec['workers']))
        c1 = time.process_time()
        wall = out['t_stop'] - out['t_ready']
        tk = time.monotonic()
        bad = check_schedules(roller, spec, out['kept'])
        check_s = time.monotonic() - tk
        msg = dict(worker=worker, core=core, affinity=sorted(os.sched_getaffinity(0)),
                   last_core=P97.last_core(), ready_s=round(out['t_ready'] - spec['t0'], 4),
                   stop_s=round(out['t_stop'] - spec['t0'], 4),
                   cpu_s_window=round(c1 - c0, 4), wall_s_window=round(wall, 4),
                   cpu_share=round((c1 - c0) / wall, 4) if wall > 0 else 0.0,
                   cpu_s_total=round(c1, 3), rss_after_load_mb=round(rss_load, 1),
                   peak_rss_mb=round(P97.maxrss_mb(), 1), discarded_rollout=out['discarded'],
                   n_schedules=len(out['rows']), n_started=out['n_started'],
                   schedules_validated=len(out['kept']), schedule_violations=bad,
                   check_s=round(check_s, 4),
                   threads={v: os.environ.get(v) for v in NUMPY_VARS},
                   info=roller.info(), rows=out['rows'], improving=out['improving'])
        q.put(('ok', worker, msg))
    except BaseException as e:          # reported, never silently dropped
        q.put(('error', worker, f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))


def merge(per):
    """Every completed schedule of every worker, ordered by completion time."""
    rows = [r for m in per for r in m['rows']]
    return sorted(rows, key=lambda r: (r[C['t']], r[C['worker']], r[C['index']]))


def trajectory(rows, search_start):
    """Improvements of the best schedule so far: (anytime rows in p97's four-column format,
    [[worker, index]] of each)."""
    anytime, src, best = [], [], math.inf
    for r in rows:
        if r[C['makespan']] < best:
            best = r[C['makespan']]
            anytime.append([round(r[C['t']] - search_start, 4), r[C['t']],
                            r[C['cpu_s_worker']], r[C['makespan']]])
            src.append([r[C['worker']], r[C['index']]])
    return anytime, src


def per_rule(rows):
    """{rule: {n, best}} over the rows (deterministic and randomized together)."""
    out = {}
    for r in rows:
        x = out.setdefault(r[C['rule']], dict(n=0, n_randomized=0, best=math.inf))
        x['n'] += 1
        x['n_randomized'] += r[C['kind']] == 'randomized'
        x['best'] = min(x['best'], r[C['makespan']])
    return {k: out[k] for k in RULE_ORDER if k in out}


def run_cell(spec, factory=RandomizedRoller, validator=P97.validate_real, timeout_s=300.0):
    """Start the workers of one cell, wait for them, merge and validate. Returns (record or
    None, best schedule (amch, st, ct) or None, errors)."""
    la0 = P97.P75.loadavg()
    mask = list(spec['affinity'])
    W = int(spec['workers'])
    msgs, errors, cores = P97.start_workers(spec, worker_main, factory, timeout_s)
    la1 = P97.P75.loadavg()
    if errors:
        return None, None, errors
    per = [msgs[w] for w in range(W)]
    rows = merge(per)
    search_start = min(m['ready_s'] for m in per)
    anytime, src = trajectory(rows, search_start)
    schedules = {(m['worker'], s['index']): s for m in per for s in m['improving']}
    best_sched, n_checked = None, 0
    for (w, j), a in zip(src, anytime):
        s = schedules.get((w, j))
        if s is None or s['makespan'] != a[3]:
            return None, None, [f'worker {w} schedule {j}: the improving schedule '
                                f'{None if s is None else s["makespan"]} is not the '
                                f'trajectory value {a[3]:g}']
        viol, sched = validator(spec, s['amch'], s['ct'])
        n_checked += 1
        if viol:
            return None, None, [f'schedule of worker {w} index {j} (makespan {a[3]:g}) fails '
                                f'validation: {list(viol)[:3]}']
        if abs(float(np.max(sched[2])) - a[3]) > 1e-9:
            return None, None, [f'schedule of worker {w} index {j}: makespan {a[3]:g} != '
                                f'latest completion {float(np.max(sched[2])):g}']
        best_sched = sched
    det_rows = [r for r in rows if r[C['kind']] == 'deterministic']
    det = {r[C['rule']]: dict(worker=r[C['worker']], index=r[C['index']],
                              makespan=r[C['makespan']], t=r[C['t']],
                              t_resident=r[C['t_resident']])
           for r in det_rows}
    t_all = max(v['t'] for v in det.values()) if sorted(det) == sorted(RULE_ORDER) else None
    t_reach = None
    if t_all is not None:
        best_rule = min(v['makespan'] for v in det.values())
        t_reach = next((a[1] for a in anytime if a[3] <= best_rule + EPS), None)
    violations = [f'worker {m["worker"]} schedule {j}: {v}' for m in per
                  for j, v in m['schedule_violations']]
    best = anytime[-1] if anytime else None
    best_row = None if best is None else next(
        x for x in rows if [x[C['worker']], x[C['index']]] == src[-1])
    rec = {
        'arm': ARM, 'method': METHOD.replace('W single', f'{W} single'),
        'size': spec['size'], 'combination': spec['combination'],
        'regime': spec['combination'], 'instance': P97.instance_name(spec['instance']),
        'environment_active': spec.get('environment'),
        'mask': {'first_mask': 'exact', 'mask_ties': 'exact'},
        'mask_check': ('every environment built under the exact mask stated as constructor '
                       'options and checked through env.mask_setting (as '
                       'p86_extend_scale_sets.masked_rollout)'),
        'rule_order': list(RULE_ORDER), 'alphas': list(ALPHAS), 'seed': int(spec['seed']),
        'cell_hash': int(spec['cell_hash']), 'rng_rule': RNG_RULE,
        'budget_s': float(spec['budget']), 'budget_clock': 'wall',
        'clock': ('time.monotonic; t from the start of the cell\'s process (imports and the '
                  'instance load included), t_resident from the moment the worker\'s instance '
                  'was loaded'),
        'workers': W, 'schedules_columns': SCHEDULE_COLUMNS, 'schedules': rows,
        'n_schedules': len(rows),
        'n_randomized': sum(r[C['kind']] == 'randomized' for r in rows),
        'n_started': sum(m['n_started'] for m in per),
        'per_rule': per_rule(rows), 'deterministic': det,
        'deterministic_layout': deterministic_layout(W), 'n_deterministic': len(det_rows),
        't_deterministic_all': t_all, 't_best_rule_reached': t_reach,
        'anytime_columns': ANYTIME_COLUMNS, 'anytime': anytime, 'anytime_from': src,
        'search_start_wall_s': search_start,
        'makespan': None if best is None else best[3],
        'objective': None if best is None else float(best[3]),
        'best_from': None if best_row is None else dict(zip(SCHEDULE_COLUMNS, best_row)),
        'validated': not violations and bool(rows), 'violations': violations[:20],
        'n_violations': len(violations), 'validator': VALIDATOR,
        'schedules_validated': sum(m['schedules_validated'] for m in per),
        'trajectory_schedules_validated': n_checked,
        'check_s_max': max(m['check_s'] for m in per),
        'schedule_file': None,
    }
    rec.update(P97.machine_fields(spec, per, mask, cores, la0, la1))
    return rec, best_sched, []


def classify(rec, cache=None):
    """(valid, reason) of a record (module docstring, VALIDITY). cache: the entry of
    p97.rule_cache_entry (default: read now from the repository)."""
    ok, why = P97.machine_reason(rec)
    if not ok:
        return ok, why
    if rec.get('arm') != ARM:
        return False, f'arm {rec.get("arm")}'
    if rec.get('budget_clock') != 'wall':
        return False, f'budget clock {rec.get("budget_clock")}'
    det = rec.get('deterministic') or {}
    if sorted(det) != sorted(RULE_ORDER):
        return False, f'deterministic rules within the budget: {sorted(det)}'
    if rec.get('n_deterministic', len(det)) != len(RULE_ORDER):
        return False, f'{rec.get("n_deterministic")} deterministic schedules, want one per rule'
    if not isinstance(rec.get('workers'), int) or rec['workers'] < 1:
        return False, f'workers {rec.get("workers")}'
    layout = deterministic_layout(rec['workers'])
    for k in RULE_ORDER:
        where = [det[k].get('worker'), det[k].get('index')]
        if where != layout[k]:
            return False, (f'{k}: deterministic schedule of worker {where[0]} sample {where[1]}, '
                           f'want worker {layout[k][0]} sample {layout[k][1]}')
        if det[k].get('t') is None or det[k]['t'] > float(rec['budget_s']) + EPS:
            return False, f'{k}: deterministic schedule at t {det[k].get("t")} after the budget'
    if cache is None:
        cache = P97.rule_cache_entry(rec['size'], rec['combination'],
                                     int(rec['instance'].split('_')[1]))
    if cache is None:
        return False, 'no stored rule start schedule to compare'
    for k in RULE_ORDER:
        want = cache['rule_ms'].get(k)
        if want is None or abs(det[k]['makespan'] - want) > 1e-9:
            return False, (f'{k} makespan {det[k]["makespan"]:g} differs from the stored '
                           f'{want} ({cache["file"]})')
    if rec.get('makespan') is None or rec['makespan'] > cache['ms'] + 1e-9:
        return False, f'best makespan {rec.get("makespan")} above the best rule {cache["ms"]:g}'
    return True, f'share; four deterministic rule makespans equal to {cache["file"]}'


def cell_main(a):
    """The process of one cell: build the spec, run, write the record."""
    size, combo, idx = P97.parse_cell(a._cell)
    t0 = float(a._t0) if a._t0 else _T_MODULE
    path = record_path(a.root, size, a.budget, a.workers, a.seed, combo, idx)
    if os.path.exists(path):
        print(f'skip {P97.rel(path)} (exists)', flush=True)
        return 2
    P97.cpu_configs()
    # imported here, before the workers start, so that they inherit them (p86 also sets the
    # working folder to the repository root, where its paths are relative to)
    import p86_extend_scale_sets as p86
    import common_utils  # noqa: F401
    from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: F401
    if tuple(p86.RULE_ORDER) != RULE_ORDER:
        raise Refusal(f'REFUSING: p86 rule order {p86.RULE_ORDER}, expected {RULE_ORDER}')
    spec = dict(size=size, combination=combo, instance=idx, workers=a.workers,
                budget=float(a.budget), seed=int(a.seed), cell_hash=cell_hash(size, combo, idx),
                t0=t0, environment=REG.env(), process_start_s=round(_T_MODULE - t0, 4),
                affinity=sorted(os.sched_getaffinity(0)), development=a.development,
                idle_before=json.loads(a._idle) if a._idle else None, commit=a._commit,
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    signal.signal(signal.SIGTERM, P97.on_term_children)
    rec, sched, errors = run_cell(spec)
    if errors:
        print(f'NO RECORD {size} {combo} {P97.instance_name(idx)}:\n  ' + '\n  '.join(errors),
              flush=True)
        return 3
    cache = P97.rule_cache_entry(size, combo, idx)
    rec['stored'] = None if cache is None else {k: v for k, v in cache.items()
                                                if k not in ('amch', 'ct')}
    rec['valid'], rec['valid_reason'] = classify(rec, cache)
    if not P97.write_record(path, rec, sched):
        print(f'NOT WRITTEN: {P97.rel(path)} appeared during the cell', flush=True)
        return 2
    print(describe(rec, size, combo, idx) + f' -> {P97.rel(path)}', flush=True)
    return 0


def describe(rec, size=None, combo=None, idx=None):
    head = '' if size is None else f'{size} {combo} {P97.instance_name(idx)}: '
    det = rec.get('deterministic') or {}
    return (head + f'{rec["n_schedules"]} schedules ({rec["n_randomized"]} randomized), '
            'deterministic ' + ', '.join(f'{k} {det[k]["makespan"]:g} (worker '
                                         f'{det[k].get("worker")}, t {det[k].get("t")})'
                                         for k in RULE_ORDER if k in det)
            + f'; best of four at t {rec.get("t_deterministic_all")}'
            + f'; best {rec["makespan"]}, shares {[w["cpu_share"] for w in rec["per_worker"]]}, '
            + ('valid' if rec.get('valid') else 'INVALID') + f' ({rec.get("valid_reason")})')


# ---------------------------------------------------------------------------
# plan and campaign
# ---------------------------------------------------------------------------

def cell_list(a):
    """(size, instance, combination) of every cell of the plan, in run order."""
    return [(size, idx, combo) for size in a.sizes for idx in a.inst for combo in a.combos]


def plan_want(a):
    return dict(arm=ARM, budget_s=float(a.budget), workers=a.workers, seed=int(a.seed),
                alphas=list(ALPHAS), rule_order=list(RULE_ORDER))


def plan_cells(a):
    """Every cell of the plan with its record path, position, share and state
    (p97.plan_state: done = valid, run = missing, invalid, held, foreign)."""
    want = plan_want(a)
    n = a.share[1]
    cells = []
    for k, (size, idx, combo) in enumerate(cell_list(a)):
        p = record_path(a.root, size, a.budget, a.workers, a.seed, combo, idx)
        c = dict(size=size, combination=combo, instance=idx, record=p, position=k,
                 share=k % n + 1)
        cells.append(P97.plan_state(c, p, want, classify, a.max_invalid))
    return cells


STATE_WORD = {'done': 'valid', 'run': 'missing', 'invalid': 'invalid', 'held': 'held',
              'foreign': 'foreign'}


def refusals_for(a, cores, cells):
    out = list(evidence_refusals(a.sizes, a.inst))
    out += P97.common_refusals(a, cores, cells)
    no_cache = [c for c in cells if P97.rule_cache_entry(c['size'], c['combination'],
                                                         c['instance']) is None]
    if no_cache and not a.development:
        c = no_cache[0]
        out.append(f'{len(no_cache)} cell(s) without a stored rule start schedule to compare, '
                   f'first {c["size"]} {c["combination"]} {P97.instance_name(c["instance"])}')
    return out


def runner_command(slot, share, a=None):
    """The runner's command line of one slot (as the launcher runs it)."""
    extra = []
    if a is not None:
        if float(a.budget) != DEFAULT_BUDGET:
            extra += ['--budget', P97.bnum(a.budget)]
        if a.workers != DEFAULT_WORKERS:
            extra += ['--workers', str(a.workers)]
        if int(a.seed) != DEFAULT_SEED:
            extra += ['--seed', str(a.seed)]
    return (f'taskset -c {slot} env PACT_ENV=exact PACT_SEEDS=301-305 OMP_NUM_THREADS=1 '
            f'MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= {sys.executable} -B '
            f'scripts/p135_randomized_rules.py run --share {share[0]}/{share[1]}'
            + (' ' + ' '.join(extra) if extra else ''))


def print_plan(a, cells, cores):
    count = {s: sum(c['state'] == s for c in cells) for s in STATE_WORD}
    print(f'randomized rules: {", ".join(RULE_ORDER)} (alphas {", ".join(map(str, ALPHAS))}), '
          f'budget {P97.bnum(a.budget)} s, {a.workers} workers, seed {a.seed}'
          + ('; DEVELOPMENT (records are not measurements)' if a.development else ''))
    print(f'cells: {len(cells)} (sizes {" ".join(a.sizes)}; combinations {" ".join(a.combos)}; '
          f'instances {TCP.compress(a.inst)})')
    print('  ' + ', '.join(f'{STATE_WORD[s]} {count[s]}' for s in STATE_WORD))
    print(f'records: {P97.rel(folder(a.root, "<size>", a.budget, a.workers, a.seed))}/'
          '<combination>/instance_<i>.json')
    for c in [c for c in cells if c['state'] in ('invalid', 'held', 'foreign')][:10]:
        print(f'  {STATE_WORD[c["state"]]} {c["size"]} {c["combination"]} '
              f'{P97.instance_name(c["instance"])}: {c["reason"]}')
    n = a.share[1]
    for i in range(1, n + 1):
        mine = [c for c in cells if c['share'] == i]
        pending = [c for c in mine if c['state'] in ('run', 'invalid')]
        print(f'share {i}/{n}: {len(mine)} cells, {sum(c["state"] == "done" for c in mine)} '
              f'valid, {len(pending)} to run; expected wall-clock {len(pending)} x '
              f'({P97.bnum(a.budget)} + {OVERHEAD_S:g}) s = '
              f'{len(pending) * (a.budget + OVERHEAD_S) / 3600:.2f} h')
    if n != 2:
        for i in (1, 2):
            pending = [c for c in cells if c['state'] in ('run', 'invalid')
                    and in_share(c['position'], (i, 2))]
            print(f'as two slots, share {i}/2: {len(pending)} to run, '
                  f'{len(pending) * (a.budget + OVERHEAD_S) / 3600:.2f} h')
    print('launch commands (repository root), one per slot:')
    for i, slot in enumerate(SLOTS, 1):
        print(f'  SLOT={slot} SHARE={i}/{len(SLOTS)} scripts/run_randomized_rules.sh')
    print('the runner command each launcher starts:')
    for i, slot in enumerate(SLOTS, 1):
        print(f'  {runner_command(slot, (i, len(SLOTS)), a)}')


def plan(a):
    cores = TCP.parse_cores(a.slot) if a.slot else sorted(os.sched_getaffinity(0))
    cells = plan_cells(a)
    print_plan(a, cells, cores)
    refusals = refusals_for(a, cores, cells)
    for r in refusals:
        print(f'REFUSAL: {r}')
    print('plan only: nothing started, nothing written')
    return 1 if (refusals and a.strict) else 0


@contextlib.contextmanager
def campaign_files():
    """p97.campaign_loop with this arm's hold file and launch notes (p97.HOLD and p97.NOTES
    hold them in this process for the loop, and are restored after it)."""
    old = P97.HOLD, P97.NOTES
    P97.HOLD, P97.NOTES = HOLD, NOTES
    try:
        yield
    finally:
        P97.HOLD, P97.NOTES = old


PRE_RUN = (
    '1. Purpose: the matched dispatching-rule baseline of the one-minute row of the time-budget '
    'comparison: the four rules randomized and sampled on the eight single-thread workers of '
    'the slot for the budget, as the policy\'s sampled schedules are; the best schedule over '
    'time is compared with the policy\'s sampled schedules, the policy\'s greedy schedule, the '
    'best deterministic rule, the eight memetic runs and CP-SAT at 10, 30 and 60 s.',
    '2. Expected: the best randomized schedule is at or below the best deterministic rule (the '
    'first schedule of workers 0 to 3 is a deterministic rule, so the best of four is held '
    'after one rollout) and falls in the first seconds; whether '
    'it is shorter or longer than the policy\'s sampled schedules is the question, and either '
    'outcome is reported.',
    '3. Contamination: new records only (written atomically, an existing record is never '
    'overwritten; an invalid one moves to invalid/ and its cell runs again); instances 000-002 '
    'of the production sets selected the search configurations and are not in the plan; the '
    'slot is checked for every process above 5% of a core before the start and before every '
    'cell; every record stores each worker\'s core and CPU share (valid at 0.9 or more for every '
    'worker), the mask and the load average.',
)


def data_line(a):
    return (f'4. Data: data/FAMILY/{",".join(a.sizes)}/desc/<combination>/instance_<i> and the '
            'base instance each names, loaded as p97\'s rule arm loads them; the exact mask '
            f'checked in every environment; the deterministic makespans compared with '
            f'{P97.RULE_CACHE}/<combination>/instance_<i>.npz; every schedule validated; records '
            f'under {P97.rel(folder(a.root, "<size>", a.budget, a.workers, a.seed))}.')


def cell_command(a, c, t0=None, idle=None, commit=None):
    cmd = [sys.executable, '-B', os.path.abspath(__file__), '_cell', '--_cell',
           f'{c["size"]}:{c["combination"]}:{c["instance"]}', '--budget', P97.bnum(a.budget),
           '--workers', str(a.workers), '--seed', str(a.seed), '--root', a.root]
    if a.development:
        cmd.append('--development')
    if t0 is not None:
        cmd += ['--_t0', repr(t0), '--_idle', json.dumps(idle), '--_commit', str(commit)]
    return cmd


def campaign(a):
    REG.banner()
    cores = sorted(os.sched_getaffinity(0))
    cells = plan_cells(a)
    mine = [c for c in cells if c['share'] == a.share[0]]
    refusals = refusals_for(a, cores, cells)
    to_run = [c for c in mine if c['state'] in ('run', 'invalid')]
    print_plan(a, cells, cores)
    print(f'this run: share {a.share[0]}/{a.share[1]} on cores {cores}, {len(to_run)} cells to run')
    if to_run:
        print('first cell command: ' + ' '.join(cell_command(a, to_run[0])))
    for r in refusals:
        print(f'REFUSAL: {r}')
    if refusals:
        return 1
    if not to_run:
        print('nothing to run')
        return finish(a)
    hold = os.path.join(a.root, HOLD)
    if os.path.exists(hold):
        print(f'held by {P97.rel(hold)}; remove it to start')
        return 0
    # the prelaunch check of the timed campaign: every process above 5%, the busy ones on the slot
    busy, every = TCP.busy_on(cores, 2.0)
    a._commit_now = ('not read (development run)' if a.development else TCP.git_commit())
    sizes = sorted({c['size'] for c in to_run})
    note = ([f'=== randomized rules start {time.strftime("%Y-%m-%d %H:%M:%S")}, commit '
             f'{a._commit_now}, cores {cores}, share {a.share[0]}/{a.share[1]}, {a.workers} '
             f'workers, budget {P97.bnum(a.budget)} s, seed {a.seed}, {len(to_run)} of '
             f'{len(mine)} cells of the share to run (sizes {", ".join(sizes)})'
             + (' (DEVELOPMENT: not a measurement)' if a.development else '')]
            + list(PRE_RUN) + [data_line(a)]
            + ['processes above 5% CPU, with the core they last ran on:']
            + [f'  {x}' for x in every])
    if busy:
        note += [f'{"DEVELOPMENT, not refused" if a.development else "REFUSED"}: cores {cores} '
                 'are not idle:'] + [f'  {x}' for x in busy]
    note += P97.machine_lines(cores, to_run)
    sys.stdout.flush()
    with campaign_files():
        P97.append_note(a, note)
        if busy and not a.development:
            print(f'REFUSING: cores {cores} are not idle:\n  ' + '\n  '.join(busy))
            return 1
        rc = P97.campaign_loop(
            a, cores, to_run, lambda c, t0, idle, commit: cell_command(a, c, t0, idle, commit),
            classify, lambda rec: describe(rec))
    if rc != 0:
        return rc
    if os.path.exists(hold):
        print(f'held by {P97.rel(hold)}; the cells left run at the next start')
        return 0
    return finish(a)


def finish(a):
    """Exit status of a run: 1 when a cell of the share has no valid record."""
    cells = [c for c in plan_cells(a) if c['share'] == a.share[0]]
    bad = [c for c in cells if c['state'] != 'done']
    for c in bad[:20]:
        moved = TCP.n_invalid(c['record'])
        word = (f'missing, {moved} invalid record(s) moved to invalid/, runs again at the next '
                'start' if c['state'] == 'run' and moved else STATE_WORD[c['state']])
        print(f'without a valid record: {c["size"]} {c["combination"]} '
              f'{P97.instance_name(c["instance"])} ({word}'
              + (f': {c["reason"]}' if c.get('reason') else '') + ')')
    print(f'share {a.share[0]}/{a.share[1]}: {len(cells) - len(bad)} of {len(cells)} cells valid')
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

def slot_name(aff):
    for s in SLOTS:
        if aff and set(aff) <= set(TCP.parse_cores(s)):
            return s
    return str(aff)


def check(a):
    """Completeness, validity by slot, and the alpha = 0 construction against the stored
    rule makespans on the cells already run."""
    cells = plan_cells(a)
    count = {s: sum(c['state'] == s for c in cells) for s in STATE_WORD}
    print(f'cells: {len(cells)}; ' + ', '.join(f'{STATE_WORD[s]} {count[s]}' for s in STATE_WORD))
    by_slot = {}
    for c in cells:
        if os.path.exists(c['record']):
            rec = json.load(open(c['record']))
            k = slot_name(rec.get('affinity'))
            x = by_slot.setdefault(k, dict(valid=0, invalid=0))
            x['valid' if c['state'] == 'done' else 'invalid'] += 1
        n_inv = TCP.n_invalid(c['record'])
        if n_inv:
            by_slot.setdefault('moved to invalid/', dict(valid=0, invalid=0))['invalid'] += n_inv
    for k, x in by_slot.items():
        print(f'  records on {k}: valid {x["valid"]}, invalid {x["invalid"]}')
    run = [c for c in cells if os.path.exists(c['record'])]
    if a.alpha0_cells is not None:
        run = run[:a.alpha0_cells]
    bad = []
    P97.cpu_configs()
    for c in run:
        rec = json.load(open(c['record']))
        roller = RandomizedRoller(dict(size=c['size'], combination=c['combination'],
                                       instance=c['instance']))
        got = deterministic_by_construction(roller.jl, roller.pt, roller.desc)
        cache = P97.rule_cache_entry(c['size'], c['combination'], c['instance'])
        det = rec.get('deterministic') or {}
        layout = deterministic_layout(rec.get('workers') or a.workers)
        name = f'{c["size"]} {c["combination"]} {P97.instance_name(c["instance"])}'
        for k in RULE_ORDER:
            ms = got[k][2]
            want = None if cache is None else cache['rule_ms'].get(k)
            have = det.get(k, {}).get('makespan')
            if want is None or abs(ms - want) > 1e-9 or have is None or abs(ms - have) > 1e-9:
                bad.append(f'{name} {k}: alpha 0 {ms:g}, stored {want}, record {have}')
            where = [det.get(k, {}).get('worker'), det.get(k, {}).get('index')]
            if have is not None and where != layout[k]:
                bad.append(f'{name} {k}: from worker {where[0]} sample {where[1]}, want worker '
                           f'{layout[k][0]} sample {layout[k][1]}')
        print(f'  {name}: best of four at t {rec.get("t_deterministic_all")}, best rule reached '
              f'at t {rec.get("t_best_rule_reached")}; ' + ', '.join(
                  f'{k} worker {det[k].get("worker")} t {det[k].get("t")}'
                  for k in RULE_ORDER if k in det))
    print(f'alpha = 0 against the stored rule makespans: {len(run)} cells x {len(RULE_ORDER)} '
          f'rules, {len(bad)} differ')
    for x in bad[:20]:
        print(f'  DIFFERENT {x}')
    ok = not bad and count['done'] == len(cells)
    print('complete and reproduced' if ok else 'incomplete or different')
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def p124_namespace(a):
    """p124's own options for its readers: the request clock at BUDGETS, primary PRIMARY."""
    import p124_memetic_parallel as P124
    argv = ['--budgets', *map(str, BUDGETS), '--primary', str(PRIMARY), '--instances',
            TCP.compress(a.inst), '--sizes', *a.sizes, '--boot', str(a.boot),
            '--alpha', str(a.alpha)]
    pa = P124.build_parser().parse_args(argv)
    bad = P124.check_options(pa)
    if bad:
        raise Refusal('REFUSING: ' + '; '.join(bad))
    return P124, pa


def read_stored(a):
    """The stored campaign records, read only through p124's readers (module docstring,
    ANALYSIS). Returns dict(sizes: {size: arrays}, selected_run, cpsat_configuration,
    problems, rejected, missing, files)."""
    P124, pa = p124_namespace(a)
    T = P124.T
    if [float(b) for b in pa.budgets] != [float(b) for b in BUDGETS] or list(pa.inst) != list(a.inst) \
            or list(pa.combos) != list(a.combos):
        raise Refusal(f'REFUSING: p124 reads budgets {pa.budgets}, instances {pa.inst}, '
                      f'combinations {pa.combos}; this arm {BUDGETS}, {a.inst}, {a.combos}')
    sel, cpsat_arm, problems = P124.selections(pa)
    data, rejected, missing, files, _ = P124.read_inputs(pa, sel, cpsat_arm)
    grid = [float(b) for b in pa.budgets]
    plan_ = [(s, r, i) for s in pa.sizes for i in pa.inst for r in pa.combos]
    pol_tokens = {s: (float(pa.policy_budget), None,
                      f'{T.POLICY_ARM}@{P124.bnum(pa.policy_budget)}') for s in pa.sizes}
    new = {}
    if sel is not None:
        new, _, _, _, _ = T.request_records(P124.p94_args(pa), plan_, pol_tokens, sel, grid,
                                            pa.result_root or ROOT)
    out = {}
    ii = {i: k for k, i in enumerate(pa.inst)}
    ci = {r: j for j, r in enumerate(pa.combos)}
    for size, D in data.items():
        use, _, best = P124.best_of(D['M'], D['V'], pa.min_seeds)
        pol_n = np.full((len(pa.inst), len(pa.combos)), np.nan)
        for (s, r, i), v in new.items():
            if s != size or 'policy' not in v:
                continue
            rec = json.load(open(os.path.join(ROOT, v['policy']['record'])))
            k = rec['schedules_columns'].index('t')
            pol_n[ii[i], ci[r]] = sum(x[k] <= PRIMARY + EPS for x in rec['schedules'])
        pol_sel = D['pol_sel'] if D['pol_sel'] is not None else np.full(use.shape, np.nan)
        out[size] = dict(pol_sel=pol_sel, o_pol=D['o_pol'], rule=D['rule'], o_rule=D['o_rule'],
                         smp=D['smp'], sp=D['sp'], best=best, use=use, CP=D['CP'],
                         CPv=D['CPv'], pol_n=pol_n, policy_budget_s=D['policy_budget_s'])
    return dict(sizes=out, selected_run=sel, cpsat_configuration=cpsat_arm, problems=problems,
                rejected=rejected, missing=missing, files=files,
                memetic_seeds=list(pa.seeds), boot=pa.boot, alpha=pa.alpha)


def read_rnd(a):
    """This arm's valid records on the grid: per size rnd [len(BUDGETS), n, C] (the best
    schedule so far at each time, NaN without one), rv [n, C] (a valid record), ns [n, C]
    (schedules completed within the budget), and the states of the cells."""
    cells = plan_cells(a)
    n, Cn = len(a.inst), len(a.combos)
    ii = {i: k for k, i in enumerate(a.inst)}
    ci = {r: j for j, r in enumerate(a.combos)}
    out = {s: dict(rnd=np.full((len(BUDGETS), n, Cn), np.nan), rv=np.zeros((n, Cn), dtype=bool),
                   ns=np.full((n, Cn), np.nan)) for s in a.sizes}
    for c in cells:
        if c['state'] != 'done':
            continue
        rec = json.load(open(c['record']))
        R = out[c['size']]
        i, j = ii[c['instance']], ci[c['combination']]
        R['rv'][i, j] = True
        for k, t in enumerate(BUDGETS):
            v = incumbent_at(rec, float(t), 'wall', 'process')
            R['rnd'][k, i, j] = np.nan if v is None else float(v)
        tk = rec['schedules_columns'].index('t')
        R['ns'][i, j] = sum(x[tk] <= float(rec['budget_s']) + EPS for x in rec['schedules'])
    return out, cells


def entry_means(x, pres, combos):
    """Mean of x over the cells of pres per combination and per pooled entry of p94."""
    import p94_timed_inference as T
    out = {}
    for j, r in enumerate(combos):
        m = pres[:, j]
        if m.any():
            out[r] = dict(mean=float(np.mean(x[m, j])), n_cells=int(m.sum()))
    for key, cols in T.pools(combos):
        m = pres[:, cols]
        if m.any():
            out[key] = dict(mean=float(np.mean(x[:, cols][m])), n_cells=int(m.sum()),
                            combinations=[combos[q] for q in cols])
    return out


def analyse(stored, rnd, combos, B, alpha):
    """Every quantity of every size (module docstring, ANALYSIS)."""
    import p94_timed_inference as T
    eight = sorted(combos) == sorted(COMBOS)
    res = {}
    for size, S in stored.items():
        R = rnd[size]
        sbud = S['policy_budget_s']
        pol_pres = np.isfinite(S['pol_sel']) & np.isfinite(S['o_pol'])
        rule_pres = np.isfinite(S['rule']) & np.isfinite(S['o_rule'])
        Z = dict(comparisons={}, verdicts={}, cells_valid=int(R['rv'].sum()),
                 cells=int(R['rv'].size), policy_budget_s=sbud)
        for k, b in enumerate(BUDGETS):
            t = float(b)
            g = np.where(S['o_pol'] <= t + EPS, S['pol_sel'], np.nan)
            rl = np.where(S['o_rule'] <= t + EPS, S['rule'], np.nan)
            x = R['rnd'][k]
            pairs = dict(rnd_vs_pol=(x, g, R['rv'] & pol_pres),
                         smp_vs_rnd=(S['smp'][k], x, R['rv'] & S['sp']),
                         rnd_vs_rule=(x, rl, R['rv'] & rule_pres),
                         mem_vs_rnd=(S['best'][k], x, R['rv'] & S['use']),
                         cp_vs_rnd=(S['CP'][k], x, R['rv'] & S['CPv']))
            held = sbud is not None and t > sbud + EPS
            for key, *_ in COMPARISONS:
                first, second, pres = pairs[key]
                rows = T.compare_at(first, second, pres, combos, B, alpha)
                if key.startswith('smp'):
                    rows = T.mark_held(rows, held, sbud)
                Z['comparisons'].setdefault(key, {})[str(b)] = rows
                if b == PRIMARY and key == VERDICT_KEY and eight:
                    fam = T.verdict_at(first, second, pres, combos, B, alpha)
                    if fam is not None:
                        fam = dict(fam, held=held, held_after_s=sbud)
                    Z['verdicts'][key] = fam
        both = R['rv'] & S['sp'] & np.isfinite(S['pol_n']) & np.isfinite(R['ns'])
        Z['schedules'] = dict(randomized_rules=entry_means(R['ns'], both, combos),
                              policy_sample=entry_means(S['pol_n'], both, combos),
                              within_s=PRIMARY)
        res[size] = Z
    return res


def out_paths(a):
    if a.out:
        return {k: os.path.join(a.out, v) for k, v in OUT_FILES.items()}
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def out_refusals(out):
    """--out must lie outside the repository's paper, result and record folders and PACT_OUT."""
    import p94_timed_inference as T
    banned = [os.path.join(ROOT, x) for x in T.RECORD_DIRS + ('results', 'out')] + \
             [os.path.join(ROOT, x) for x in os.listdir(ROOT) if x.startswith('paper')]
    if os.environ.get('PACT_OUT'):
        banned.append(os.path.abspath(os.environ['PACT_OUT']))
    return [f'--out {out} lies inside or around {P97.rel(d)}' for d in banned
            if T._inside(out, d) or T._inside(d, out)]


def macro_lines(res):
    import p94_timed_inference as T
    import stats_family as SF
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["run"]} run; {res["setting"]}; selected training run s{res["selected_run"]}; '
         f'CP-SAT configuration {res["cpsat_configuration"]}; B = {res["boot"]}, alpha = '
         f'{res["alpha"]}; instance-level intervals (p94.compare_at, stats_family.instance_boot, '
         f'fresh default_rng({T.INST_SEED}) per quantity).',
         '% Request clock: each method from the request for a schedule, cold start; the '
         'randomized rules and the policy\'s sampled schedules at their own t (best so far), the '
         'greedy schedule from its time on, the best deterministic rule from t_all of the rule '
         'record on, a search started from the rules at run time t - t_all.',
         f'% Randomized rules: {res["method"]}; budget {P97.bnum(res["budget_s"])} s, seed '
         f'{res["seed"]}. Differences in %: first-named minus second over second (negative = '
         'first-named shorter); direction from the unrounded interval; at 10 s and 30 s the '
         'mean only.',
         f'% Verdicts (names ending in Verdict) at {PRIMARY} s by the rule of p94 row Sel: '
         f'{T.VERDICT_TEXT["Sel"]} (here the first-named method is the policy\'s sampled '
         'schedules)']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}CellN', res['cells_total'], 'cells of the plan: sizes x combinations x instances')
    nc(f'{PREFIX}CellValidN', res['cells_valid'], 'cells with a valid record of the randomized '
       'rules')
    nc(f'{PREFIX}WorkerN', res['workers'], 'single-thread workers per cell, one per core')
    nc(f'{PREFIX}BudgetS', P97.bnum(res['budget_s']), 'seconds per cell from the request')
    nc(f'{PREFIX}InstN', len(res['instances']),
       f'instances per combination ({TCP.compress(res["instances"])})')
    nc(f'{PREFIX}AlphaN', len(ALPHAS), 'values of alpha, ' + ', '.join(map(str, ALPHAS)))
    nc(f'{PREFIX}Boot', res['boot'], 'bootstrap resamples')
    for size, S in res['results'].items():
        sw = T.SIZE_WORD[size]
        nc(f'{PREFIX}{sw}CellValidN', S['cells_valid'], f'{size}: cells with a valid record')
        for key, word, text in COMPARISONS:
            for b, rows in S['comparisons'].get(key, {}).items():
                bw = T.number_word(int(b))
                for r, x in rows.items():
                    if 'mean' not in x:
                        continue
                    if r in COMBOS and not (key == VERDICT_KEY and int(b) == PRIMARY):
                        continue
                    base = f'{PREFIX}{sw}{word}{bw}{r}'
                    what = f'{size}, {T.entry_text(r)}, {b} s since the request: {text}'
                    n = f'{x["n"]} instances x {x["c"]} combinations'
                    nc(base, T.f2(x['mean']), f'{what}: mean difference in percent, {n}')
                    if int(b) == PRIMARY:
                        nc(base + 'Lo', T.f2(x['lo']), f'{what}: lower end of the 95% interval')
                        nc(base + 'Hi', T.f2(x['hi']), f'{what}: upper end of the 95% interval')
                        nc(base + 'Dir', SF.TEX[x['direction']],
                           f'{what}: direction of [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
        fam = S['verdicts'].get(VERDICT_KEY)
        if fam is not None and 'no_schedule_yet' not in fam:
            word = next(c[1] for c in COMPARISONS if c[0] == VERDICT_KEY)
            base = f'{PREFIX}{sw}{word}{T.number_word(PRIMARY)}'
            for r, x in fam['rows'].items():
                nc(f'{base}{r}Verdict', x['verdict'],
                   f'{size}, {r}: verdict; interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; Holm p '
                   f'shorter {x["p_boot_less_holm"]:.2g}, Holm p longer '
                   f'{x["p_boot_greater_holm"]:.2g} (bootstrap)')
            for w in T.VERDICT_WORDS:
                nc(f'{base}Verdict{T.COUNT_WORD[w]}', fam['counts'][w],
                   f'{size}: combinations (of {len(fam["rows"])}) {w}')
        sch = S['schedules']
        for r, x in sch['randomized_rules'].items():
            if r in COMBOS:
                continue
            y = sch['policy_sample'][r]
            nc(f'{PREFIX}{sw}SchedMean{r}', f'{x["mean"]:.0f}',
               f'{size}, {T.entry_text(r)}: mean number of schedules of the randomized rules per '
               f'cell within {sch["within_s"]} s, {x["n_cells"]} cells')
            nc(f'{PREFIX}{sw}PolSmpSchedMean{r}', f'{y["mean"]:.0f}',
               f'{size}, {T.entry_text(r)}: mean number of the policy\'s sampled schedules per '
               f'cell within {sch["within_s"]} s, {y["n_cells"]} cells')
    return L


def check_names(lines, own):
    import p124_memetic_parallel as P124
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x)]
    if bad:
        raise Refusal(f'REFUSING: macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refusal(f'REFUSING: duplicate macro names: '
                      f'{sorted({x for x in names if names.count(x) > 1})[:5]}')
    have = P124.defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refusal(f'REFUSING: the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def cell_text(x, interval=True, direction=True):
    import p94_timed_inference as T
    if 'no_schedule_yet' in x:
        return f'no schedule yet ({x["no_schedule_yet"]} of {x["n_cells"]} cells)'
    s = T.f2(x['mean'])
    if interval:
        s += f' [{T.f2(x["lo"])}, {T.f2(x["hi"])}]'
    if direction:
        s += f' {x["direction"]}'
    return s + (' (held)' if x.get('held') else '')


def markdown(res):
    import p94_timed_inference as T
    L = [f'# Randomized dispatching rules within {P97.bnum(res["budget_s"])} s ({res["run"]} run)',
         '', f'Generated by {GENERATOR}. {res["setting"]}; selected training run '
         f's{res["selected_run"]}; CP-SAT configuration {res["cpsat_configuration"]}; B = '
         f'{res["boot"]}.', '', f'Method: {res["method"]}.', '',
         f'Cells with a valid record: {res["cells_valid"]} of {res["cells_total"]}.', '',
         'Differences in percent of the second-named method (negative: the first-named is '
         f'shorter), request clock; at {PRIMARY} s mean [95% interval] direction, at 10 s and '
         '30 s mean [interval], descriptive.', '']
    if res.get('warning'):
        L[1:1] = ['', f'**{res["warning"]}**']
    for size, S in res['results'].items():
        L += [f'## {size}', '', f'Cells with a valid record: {S["cells_valid"]} of {S["cells"]}.',
              '']
        for key, word, text in COMPARISONS:
            L += [f'### {key}: {text}', '', '| entry | ' + ' | '.join(f'{b} s' for b in BUDGETS)
                  + ' |', '|---|' + '---|' * len(BUDGETS)]
            rows = S['comparisons'][key]
            entries = list(rows[str(BUDGETS[-1])].keys())
            for r in entries:
                cells = []
                for b in BUDGETS:
                    x = rows[str(b)].get(r)
                    cells.append('-' if x is None else cell_text(x, True, b == PRIMARY))
                L.append(f'| {T.entry_text(r)} | ' + ' | '.join(cells) + ' |')
            L.append('')
        fam = S['verdicts'].get(VERDICT_KEY)
        if fam:
            if 'no_schedule_yet' in fam:
                L += [f'Verdicts at {PRIMARY} s: no schedule yet on {fam["no_schedule_yet"]} '
                      'cells.', '']
            else:
                L += [f'### Verdicts at {PRIMARY} s, the policy\'s sampled schedules minus the '
                      'randomized rules (Holm over the eight combinations)', '',
                      '| combination | mean | interval | Holm p shorter | Holm p longer | '
                      'verdict |', '|---|---|---|---|---|---|']
                for r, x in fam['rows'].items():
                    L.append(f'| {r} | {T.f2(x["mean"])} | [{T.f2(x["lo"])}, {T.f2(x["hi"])}] | '
                             f'{x["p_boot_less_holm"]:.3g} | {x["p_boot_greater_holm"]:.3g} | '
                             f'{x["verdict"]} |')
                L += ['', 'Counts: ' + ', '.join(f'{w} {fam["counts"][w]}'
                                                 for w in T.VERDICT_WORDS), '']
        sch = S['schedules']
        L += [f'### Schedules per cell within {sch["within_s"]} s (cells with both records)', '',
              '| entry | randomized rules | policy, sampled | cells |', '|---|---|---|---|']
        for r, x in sch['randomized_rules'].items():
            y = sch['policy_sample'][r]
            L.append(f'| {T.entry_text(r)} | {x["mean"]:.1f} | {y["mean"]:.1f} | {x["n_cells"]} |')
        L.append('')
    if res.get('rejected') or res.get('missing') or res.get('problems'):
        L += ['## Inputs not used', '']
        L += [f'- rejected {x["id"]}: {x["state"]} ({x["reason"]})' for x in res['rejected'][:50]]
        L += [f'- missing: {x}' for x in (res['problems'] + res['missing'])[:50]]
        L.append('')
    return '\n'.join(L) + '\n'


def infer(a, stored=None):
    """Read, analyse, write (module docstring, ANALYSIS). stored: the stored inputs as
    read_stored returns them (default: read now). Returns (result, paths)."""
    import p94_timed_inference as T
    if a.out:
        bad = out_refusals(a.out)
        if bad:
            raise Refusal('REFUSING: ' + '; '.join(bad))
    elif os.environ.get('PACT_OUT') is None:
        raise Refusal('REFUSING: set PACT_OUT (the output root of the chain, e.g. out/exact) or '
                      'give --out')
    if stored is None:
        stored = read_stored(a)
    rnd, cells = read_rnd(a)
    n_valid = sum(c['state'] == 'done' for c in cells)
    gaps = [c for c in cells if c['state'] != 'done']
    full = (not gaps and not stored['missing'] and not stored['rejected']
            and not stored['problems'] and stored['selected_run'] is not None
            and stored['cpsat_configuration'] is not None)
    if not full and not a.out:
        raise Refusal(f'REFUSING the full-run write: {len(gaps)} of {len(cells)} cells without a '
                      f'valid record, {len(stored["missing"])} missing and '
                      f'{len(stored["rejected"])} rejected stored inputs, '
                      f'{len(stored["problems"])} problems; give --out for a partial run')
    paths = out_paths(a)
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    R = analyse(stored['sizes'], rnd, list(a.combos), stored['boot'], stored['alpha'])
    res = dict(generator=GENERATOR, run='full' if full else 'partial',
               warning=None if full else (f'PARTIAL: {n_valid} of {len(cells)} cells valid; '
                                          f'{len(stored["missing"]) + len(stored["rejected"])} '
                                          'stored inputs missing or rejected; not for the paper'),
               setting=(f'environment {REG.env()}, training seeds '
                        f'{",".join(map(str, REG.seeds()))}'),
               method=METHOD.replace('W single', f'{a.workers} single'), seed=int(a.seed),
               alphas=list(ALPHAS), rule_order=list(RULE_ORDER), budget_s=float(a.budget),
               workers=a.workers, budgets=list(BUDGETS), primary=PRIMARY,
               boot=stored['boot'], alpha=stored['alpha'], instances=list(a.inst),
               sizes=list(a.sizes), combinations=list(a.combos),
               cells_total=len(cells), cells_valid=n_valid,
               cells_without_valid_record=[dict(size=c['size'], combination=c['combination'],
                                                instance=c['instance'],
                                                state=STATE_WORD[c['state']],
                                                reason=c.get('reason', ''))
                                           for c in gaps][:200],
               selected_run=stored['selected_run'],
               cpsat_configuration=stored['cpsat_configuration'],
               memetic_seeds=stored.get('memetic_seeds'),
               records=P97.rel(folder(a.root, '<size>', a.budget, a.workers, a.seed))
               + '/<combination>/instance_<i>.json',
               statistics=('p94.compare_at and p94.compare (p94.stat_instance: '
                           'stats_family.instance_boot, fresh default_rng(7) per quantity; '
                           'stats_family.direction), p94.mark_held, p94.verdict_at '
                           '(p94.verdict_family: instance_boot draws, '
                           'stats_family.boot_p_one_sided, stats_family.holm, '
                           'p94.selected_verdict)'),
               comparisons={k: t for k, _, t in COMPARISONS},
               rejected=stored['rejected'], missing=stored['missing'],
               problems=stored['problems'], arrays=stored.get('files', []), results=R)
    try:
        lines = macro_lines(res)
        if not full:
            lines.insert(1, '% PARTIAL: not every cell or input is present; not for the paper')
        check_names(lines, paths['macros'])
        from macro_io import emit
        emit(paths['macros'], lines)
        os.makedirs(os.path.dirname(paths['json']) or '.', exist_ok=True)
        with open(paths['json'], 'w') as f:
            json.dump(T.jsonable(res), f, indent=1)
        os.makedirs(os.path.dirname(paths['md']) or '.', exist_ok=True)
        with open(paths['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
        raise
    return res, paths


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--budget', type=float, default=DEFAULT_BUDGET,
                        help='wall seconds per cell, from the start of the cell\'s process')
    common.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    common.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='first entry of the seed of every worker\'s generator')
    common.add_argument('--sizes', nargs='+', default=list(SIZES))
    common.add_argument('--combos', nargs='+', default=list(COMBOS))
    common.add_argument('--instances', default=None,
                        help="e.g. '3-11' (default p124_memetic_parallel.INSTANCES)")
    common.add_argument('--share', default='1/1', help='i/n: the cells at positions k with '
                        'k mod n = i - 1')
    common.add_argument('--root', default=ROOT, help='folder under which test_results/FAMILY '
                        'and train_log are written (default the repository root)')
    common.add_argument('--max_invalid', type=int, default=MAX_INVALID)
    common.add_argument('--development', action='store_true',
                        help='lift the refusals of a busy core and of more workers than cores; '
                             'needs --root outside the repository; records are development_run')
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('plan', parents=[common], help='the cells, their states, the commands')
    q.add_argument('--strict', action='store_true', help='exit 1 when a start must be refused')
    q.add_argument('--slot', default=None, help='the core list the refusals are checked for')
    sub.add_parser('run', parents=[common], help='the cells of --share on the runner\'s mask')
    q = sub.add_parser('check', parents=[common], help='completeness, validity by slot, alpha 0')
    q.add_argument('--alpha0-cells', dest='alpha0_cells', type=int, default=None,
                   help='rebuild the four rules on the first N cells with a record only')
    q = sub.add_parser('infer', parents=[common], help='the numbers: JSON, Markdown, macros')
    q.add_argument('--out', default=None, help='write under this folder (outside the '
                   'repository\'s paper, result and record folders)')
    q.add_argument('--boot', type=int, default=None, help='bootstrap resamples (default '
                   'stats_family.B_DEFAULT)')
    q.add_argument('--alpha', type=float, default=0.05)
    q = sub.add_parser('_cell', parents=[common])
    q.add_argument('--_cell', required=True)
    q.add_argument('--_t0', default=None)
    q.add_argument('--_idle', default=None)
    q.add_argument('--_commit', default=None)
    return p


def resolve(a):
    a.root = os.path.abspath(os.path.join(_CWD, a.root))
    if getattr(a, 'out', None):
        a.out = os.path.abspath(os.path.join(_CWD, a.out))
    a.share = parse_share(a.share)
    for c in a.combos:
        if c not in COMBOS:
            raise SystemExit(f'unknown combination {c}')
    if a.workers < 1 or a.workers > 2 ** P97.W_BITS:
        raise SystemExit(f'--workers between 1 and {2 ** P97.W_BITS}')
    if a.budget <= 0:
        raise SystemExit('--budget must be positive')
    if a.seed < 0:
        raise SystemExit('--seed must be non-negative')
    if a.cmd != '_cell':
        a.inst = TCP.parse_instances(a.instances or default_instances())
    if a.cmd == 'infer' and a.boot is None:
        import stats_family as SF
        a.boot = SF.B_DEFAULT
    return a


def main(argv=None):
    a = resolve(build_parser().parse_args(_ARGV if argv is None else argv))
    if a.cmd == '_cell':
        return cell_main(a)
    if a.cmd == 'plan':
        return plan(a)
    if a.cmd == 'run':
        return campaign(a)
    if a.cmd == 'check':
        return check(a)
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    if REG.env() != 'exact':
        raise SystemExit(f'REFUSING: PACT_ENV={REG.env()}: this analysis reads the exact '
                         'environment')
    REG.banner()
    res, paths = infer(a)
    print(f'p135: {res["run"]} run: {res["cells_valid"]} of {res["cells_total"]} cells valid; '
          'wrote ' + ', '.join(P97.rel(p) for p in paths.values() if os.path.exists(p)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
