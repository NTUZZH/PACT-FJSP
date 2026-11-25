r"""P97: the policy's sampled schedules within a time budget, on the eight cores of one slot.

The timed campaign (scripts/run_timed_campaign.sh) gives the CP-SAT comparator eight
workers on eight reserved cores for its budget, while the policy enters with one greedy
schedule from one core. This runner gives the policy the same cores and a budget. One
cell is one size, one combination and one instance. The runner starts W worker processes
(default 8), worker w pinned to core w of the runner's affinity mask (the launcher sets
the mask with taskset; the runner never widens it). Every worker loads the policy and
rolls out complete schedules of the instance until the budget ends: worker 0 starts with
the greedy schedule (argmax at every decision), every other rollout samples a batch of B
schedules from the policy's action distribution. The best schedule so far, over all
workers, is recorded over time. A second arm (--arm rules, section RULE ARM) times the
dispatching-rule start schedule of the search arms on the same cores.

PRE-RUN RECORD (the launch note of every start repeats it with the list of processes on
the slot's cores).

1. PURPOSE. The policy at the computing effort of the CP-SAT arm: eight single-thread
   workers on eight cores for the budget. The best schedule over time is the policy's
   curve in the time-budget figure (from the first second to the budget) and its value
   at 10 s and at 60 s is the policy row of the decision table at those budgets.
2. EXPECTED RESULT. The best sampled schedule is at or below the greedy schedule and
   falls fastest in the first minute. If it never falls below the greedy schedule, the
   policy's curve is flat at the greedy value, and the comparison at equal effort keeps
   the greedy schedule as the policy's result.
3. CONTAMINATION. A record is written atomically (temporary file, then a link that
   fails when the record exists) and an existing record is never overwritten. An
   invalid record moves to <combination>/invalid/ and its cell runs again at the next
   start; a cell with --max-invalid such files is held. Instances 000-002 of the
   production sets selected the configurations of the search methods and are left out
   by default (they are the default of --summary, which selects the batch size). Before
   each cell the slot's cores are sampled (a thread of another process above 5% of one
   core refuses the cell), and every record stores each worker's CPU share, its core,
   the affinity mask and the load average. A record made with --development is marked
   development_run and is never valid.
4. DATA ACCURACY. data/FAMILY/<size>/desc/<combination>/instance_<i>.desc.json and the
   base instance it names; checkpoint, configuration and environment through
   scripts/arm_registry.py, refused when the checkpoint's training configuration does
   not carry the mask options of the active environment. The greedy makespan is
   compared with the stored greedy array of the checkpoint and with its stored
   schedule (where they exist); every schedule on the best-so-far trajectory is
   validated with the two validators of the CP-SAT runner before the record is written.

ROLLOUTS (the code the stored evaluations run, with their settings)
  Model, configuration restore and environment as scripts/p90_public_rule_comparison.py
  rollout and eval_family.py build them: the configuration keys of p90's ARCH_KEYS
  restored from the training configuration, PPO_initialize, the state dict on the CPU,
  FJSPEnvFamily with the restored mask_mode and pair_extra and the mask options of the
  active environment, the restored shaping envelope (only 'min' is accepted: the
  sampling code of p90 leaves the environment at its default 'min', so another envelope
  would be sampled on another observation), the instance's types and token.
  Greedy: common_utils.greedy_select_action at every decision (eval_family.py greedy,
  p90 trace). Sampled: common_utils.sample_action (Categorical sampling) at every
  decision (eval_family.py and p90 sampling).
  Batches (--batch B, default 1). A sampled rollout steps B copies of the instance in one
  environment, built as p90 sample builds its K copies (the instance, the descriptor and
  the type lists repeated B times, one token per copy), and returns B schedules. Worker 0
  starts with the greedy schedule, a batch of one; every later rollout of every worker is
  a batch of B. A batch that is incomplete at the budget is discarded whole.
  Seeds. Before sampled batch j of worker w, torch.manual_seed(seed) with
      seed = base_seed * 2**26 + w * 2**20 + j     (0 <= w < 64, 0 <= j < 2**20),
  one seed per batch and the only random source of a rollout (sample_action draws from
  torch's global generator). The greedy rollout draws nothing and has no seed.
  --base_seed defaults to 50, the seed_test of params.py with which the stored sampled
  schedules were drawn. p90 seeds once per batch of K copies, so a batch here equals
  p90's draw with K = B and seed_test = seed. --reproduce re-runs listed batches of a
  record and compares.

CLOCK
  time.monotonic(), shared by every process on Linux. t: seconds from the start of the
  process that runs the cell (the campaign takes the time just before it starts the
  cell's process, so interpreter start, imports, the checkpoint load and the
  environment construction are inside t, as the memetic runner counts its loading and
  the CP-SAT runner its model build). The cell's process imports numpy, torch and the
  environment code and then starts the workers, which inherit these imports; each
  worker loads the checkpoint and the instance itself and builds its first environment.
  t_resident: seconds from the moment the worker's policy and first environment were
  ready. The budget ends at t = budget; a batch still running then is discarded (each
  worker checks the clock at every decision).

RECORD  or_solution/FAMILY/<size>_polsample<budget>s_b<batch>_s<seed>/<combination>/
  instance_<i>.json (under --root when given) with the best schedule beside it as
  instance_<i>.schedule.npz (amch, st, ct, ms: the format of scripts/p85_cpsat_arms.py).
  Fields: the cell, the checkpoint and its training seed, the environment and mask,
  budget, workers, batch, base seed; 'schedules' (columns in 'schedules_columns':
  worker, batch index, kind greedy or sampled, seed, t, t_resident, makespan, CPU
  seconds of the worker, row inside the batch) for every completed schedule, all rows of
  a batch sharing t, t_resident and the seed; 'anytime' (every improvement of the best
  schedule so far, at most one per batch: the batch's first row with its lowest
  makespan) in the four-column format of eval_memetic_family.py, [t -
  search_start_wall_s, t, CPU seconds of the worker that produced it, makespan], with
  search_start_wall_s the earliest moment a worker was ready, so that
  baselines.memetic.records.incumbent_at and timed_campaign_plan.record_rows (runner
  'memetic') read it unchanged on the wall clock (the third column is the producing
  worker's own CPU, not the arm's total, so the CPU clock of that reader does not
  apply); 'anytime_from' (worker, batch index, row of each improvement); the greedy
  makespan, its two times and the stored values it was compared with; per worker its
  core, CPU seconds and wall seconds over its rollout window (from ready to stop),
  cpu_share = CPU / wall, peak resident memory and thread settings; the affinity mask,
  load average at start and end, the core sample before the cell, the commit,
  development_run, valid and its reason.

VALIDITY (classify below)
  validated; not a development run; the slot's cores idle before the cell; one worker
  per core, every core inside the mask; every worker's cpu_share over its rollout
  window at least 0.9; a greedy schedule within the budget whose makespan equals every
  stored greedy result of the checkpoint for this cell that exists:
    <arm_registry.res_dir(size)>/Result_greedy+<model>_<combination>.npy, the row of the
      instance in the sorted descriptor list (eval_family.py), and
    <arm_registry.schedule_cache(seed, size)>/<combination>/instance_<i>.npz, field ms
      (the policy schedules the policy-started timed arms start from).
  The comparison is made again whenever a record is classified, so a stored array
  written after the record is still checked. The exact-environment arrays and schedules
  are produced on the CPU with one torch thread (scripts/eval_newseeds_plan.py), as
  here. The legacy arrays at 50x25 and 80x25 were produced on the GPU
  (scripts/run_scale_scout.sh), which a CPU rollout does not reproduce bit for bit, so
  in the legacy environment a difference there can come from the device.

CAMPAIGN
  --sizes, --combos, --instances (default: each size's evidence instances, 3-29 at the
  production sets 50x25n30 and 80x25n30, timed_campaign_plan.DEFAULT_INSTANCES). Cells
  run one after the other on the runner's slot, each in its own process; valid records
  are skipped; a record of another plan (budget, workers, batch, base seed, model or
  environment) refuses the start; the stop file train_log/POLSAMPLE_HOLD ends the run
  after the current cell (both arms); a launch note goes to
  train_log/policy_anytime_launch_notes.txt; two invalid records in a row stop the run.
  --dry prints the plan (cells, done, to run, slot-hours) and writes nothing.

SUMMARY (--summary; reads records, writes nothing)
  For the folders of --sizes, --budget, --seed (or --model) and each batch size of
  --batches, over --instances (default 0-2, the selection instances) and --combos
  (default all): per batch size and size the number of valid records, the mean over
  cells of 100 * (best - greedy) / greedy at the budget and at 10, 30, 60 and 120 s where
  the budget covers them (incumbent_at on the wall clock from the process start), and
  the mean number of schedules. Only cells with a valid record for every batch size are
  compared; at a time where a compared cell has no schedule yet for some batch size, the
  cell is left out at that time for every batch size; the output gives the counts. The
  batch size with the lowest mean at the budget, pooled over the sizes, is named (ties
  go to the smaller batch). With --development, records count when they are validated
  and of the plan, whatever their validity (for trying the summary on development
  records, never for a selection).

RULE ARM (--arm rules)
  The rule-started search arms start from the best of four dispatching rules, FIFO,
  MOR, SPT and MWKR, read from or_solution/FAMILY/warmstart_cache_xmask_<size>; that
  schedule was computed before their timed runs, so no record holds its time. This arm
  computes it again on the campaign's cores with the code that produced the cache.
  PRE-RUN RECORD
  1. PURPOSE. The time at which the rule start schedule of the search arms exists on the
     campaign's cores, so that every arm's clock can count its start schedule.
  2. EXPECTED RESULT. Every rule makespan equals the stored one, and the start schedule
     exists within a few seconds of the process start (the slowest of four rollouts run
     in parallel, after the imports). A makespan that differs makes the record invalid:
     the stored start schedules are then not what this code produces, and the arms that
     start from them cannot take this time until the difference is explained.
  3. CONTAMINATION. As the policy arm: atomic records that are never overwritten,
     invalid records moved to invalid/, the core sample before every cell, each worker's
     core and CPU share in the record, development runs never valid.
  4. DATA ACCURACY. The descriptor and the base instance it names, loaded as
     scripts/p92_warmstart_xmask_training_size.py and scripts/p86_extend_scale_sets.py
     load them (p86.load_instance, p86.load_stored_descriptor: the attached setup
     convention); each rule rolled out by p86.masked_rollout (common_utils.
     masked_rule_action through FJSPEnvFamily with first_mask='exact' and
     mask_ties='exact' stated as constructor options and checked, numpy seed 50 right
     before each rollout for the random tie breaks, no type lists); the eligibility test
     of the cache builder (p86._eligible) and the two validators of the CP-SAT runner
     on every schedule; every makespan compared with the stored rule_ms of
     warmstart_cache_xmask_<size>/<combination>/instance_<i>.npz.
  ROLLOUTS. Four workers, worker k pinned to core k of the mask, roll out rule k of
  FIFO, MOR, SPT, MWKR once (with --development fewer workers take the rules in turn).
  No checkpoint, no budget; the only seed is the numpy seed 50 of masked_rollout. The
  start schedule is the eligible rule with the lowest makespan, equal makespans going to
  the earlier rule in that order (the tie rule of the cache, p12_cpsat_fairwarm).
  RECORD  or_solution/FAMILY/<size>_rulestart/<combination>/instance_<i>.json (and the
  start schedule as instance_<i>.schedule.npz): per rule its makespan, t, t_resident,
  the CPU and wall seconds of its rollout, its worker, core and the worker's CPU share;
  the best rule and its makespan; t_all, the moment all four rollouts were complete (the
  first moment the start schedule is known); check_s, the seconds of the eligibility
  check after the rollouts (outside t); the stored values and whether the stored
  schedule of the best rule is the same schedule; and the machine evidence of the
  policy record.
  VALIDITY (classify_rules). Validated; not a development run; idle cores before the
  cell; one worker per core inside the mask, each pinned; every worker's share at least
  0.9; all four rules present; every rule makespan, the best rule and its makespan equal
  to the stored cache file.

Usage (repository root; the slot is the taskset mask):
  taskset -c 8-15 env PACT_ENV=exact PACT_SEEDS=301-305 \
      python scripts/p97_policy_anytime.py --budget 600 --seed select --batch 4
  PACT_ENV=exact PACT_SEEDS=301-305 python scripts/p97_policy_anytime.py \
      --budget 600 --seed select --batch 4 --dry
  python scripts/p97_policy_anytime.py --summary --budget 120 --seed select \
      --batches 1 2 4 8 16 32
  taskset -c 8-11 env PACT_ENV=exact python scripts/p97_policy_anytime.py --arm rules
  python scripts/p97_policy_anytime.py --reproduce <record> [--rows 0:0,1:3:2]
  Development (another core count, a busy core, a checkpoint of the other environment):
  --development --root <folder outside the repository's result folders>.
"""
import time

_T_MODULE = time.monotonic()      # first statement: the start of a cell's process

import argparse  # noqa: E402
import glob  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import multiprocessing as mp  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import queue  # noqa: E402
import re  # noqa: E402
import resource  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMPY_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
if __name__ == '__main__':
    # one numpy/OpenMP thread and no GPU, before numpy and torch are imported
    for _v in NUMPY_VARS:
        os.environ[_v] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402  (removes the --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
import p75_deploy_cost as P75  # noqa: E402  (sets sys.argv to the program name only)
import timed_campaign_plan as TCP  # noqa: E402
from p90_public_rule_comparison import ARCH_KEYS  # noqa: E402

ARM = 'policy_sample'
METHOD = ('policy: the greedy schedule, then batches of B schedules sampled from the '
          'policy\'s action distribution, on W single-thread workers')
COMBOS = tuple(REG.COMBOS)
SIZES = tuple(TCP.SIZES)
SHARE_MIN = P75.SHARE_MIN               # 0.9, every worker
BUSY_PCT = P75.BUSY_PCT                 # 5% of a core
IDLE_WINDOW = P75.IDLE_WINDOW           # seconds of the core sample before a cell
DEFAULT_WORKERS = 8
DEFAULT_BATCH = 1
DEFAULT_BASE_SEED = 50                  # params.py seed_test
W_BITS, J_BITS = 6, 20                  # seed = base * 2**26 + w * 2**20 + j
OVERHEAD_S = 10.0                       # per cell beyond the budget: sample, load, validate
MAX_INVALID = 2
STOP_AFTER_INVALID = 2                  # invalid records in a row that stop a run
OUT = os.path.join('or_solution', 'FAMILY')
HOLD = os.path.join('train_log', 'POLSAMPLE_HOLD')
NOTES = os.path.join('train_log', 'policy_anytime_launch_notes.txt')
SCHEDULE_COLUMNS = ['worker', 'index', 'kind', 'seed', 't', 't_resident', 'makespan',
                    'cpu_s_worker', 'row']
ANYTIME_COLUMNS = ['search_s', 'wall_s_from_process_start', 'cpu_s_of_producing_worker',
                   'makespan']
VALIDATOR = ('constraint_family.cpsat_arms.validate_all (validate_strict and the C6 check '
             'constraint_family.checks.check_schedule_validation), as '
             'scripts/p85_cpsat_arms.py')
SUMMARY_TIMES = (10.0, 30.0, 60.0, 120.0)
SUMMARY_INSTANCES = '0-2'

RULE_ARM = 'rule_start'
RULE_METHOD = ('the best of the dispatching rules FIFO, MOR, SPT and MWKR under the exact '
               'mask (p86_extend_scale_sets.masked_rollout), one rule per single-thread '
               'worker')
RULE_ORDER = ('FIFO', 'MOR', 'SPT', 'MWKR')     # p86_extend_scale_sets.RULE_ORDER
RULE_WORKERS = len(RULE_ORDER)
RULE_CACHE = os.path.join('or_solution', 'FAMILY', 'warmstart_cache_xmask_{size}')
RULE_SEED_NOTE = ('numpy.random.seed(50) right before each rollout, inside '
                  'p86_extend_scale_sets.masked_rollout (random tie breaks of '
                  'common_utils.masked_rule_action); no other seed')
RULE_CELL_S = 15.0      # slot seconds per rule cell assumed by --dry: the core sample (3 s),
#                         the imports (0.7 s), the slowest rule (1.3-1.7 CPU s at 50x25n30,
#                         4.2-4.4 at 80x25n30), the four validations and the record


class Refusal(SystemExit):
    """A check failed before any rollout."""


# ---------------------------------------------------------------------------
# seeds, names, folders
# ---------------------------------------------------------------------------

def rollout_seed(base_seed, worker, index):
    """Seed of sampled batch `index` of worker `worker`:
    base_seed * 2**26 + worker * 2**20 + index, a pure and invertible function."""
    base_seed, worker, index = int(base_seed), int(worker), int(index)
    if base_seed < 0 or not 0 <= worker < 2 ** W_BITS or not 0 <= index < 2 ** J_BITS:
        raise ValueError(f'seed of (base {base_seed}, worker {worker}, index {index}) out '
                         f'of range: base >= 0, worker < {2 ** W_BITS}, index < {2 ** J_BITS}')
    return (base_seed << (W_BITS + J_BITS)) | (worker << J_BITS) | index


def seed_parts(seed):
    """(base seed, worker, index) of a rollout seed (the inverse of rollout_seed)."""
    seed = int(seed)
    return (seed >> (W_BITS + J_BITS), (seed >> J_BITS) & (2 ** W_BITS - 1),
            seed & (2 ** J_BITS - 1))


def bnum(b):
    return f'{float(b):g}'


def instance_name(idx):
    return f'instance_{int(idx):03d}'


def descriptor_path(size, combo, idx):
    return os.path.join(ROOT, 'data', 'FAMILY', size, 'desc', combo,
                        f'{instance_name(idx)}.desc.json')


def instance_names(size, combo):
    """Instance names of a combination in the sorted order of every evaluation."""
    d = os.path.join(ROOT, 'data', 'FAMILY', size, 'desc', combo)
    return [os.path.basename(p)[:-len('.desc.json')]
            for p in sorted(glob.glob(os.path.join(d, '*.desc.json')))]


def seed_tag(seed, model):
    """The s<seed> part of the folder name: the training seed, or the name's stem."""
    if seed is not None:
        return f's{int(seed)}'
    m = re.search(r'-s(\d+)$', model)
    return f's{m.group(1)}' if m else 'm' + re.sub(r'[^A-Za-z0-9.-]+', '-', model)


def folder(root, size, budget, batch, tag):
    return os.path.join(root, OUT, f'{size}_polsample{bnum(budget)}s_b{int(batch)}_{tag}')


def record_path(root, size, budget, batch, tag, combo, idx):
    return os.path.join(folder(root, size, budget, batch, tag), combo,
                        f'{instance_name(idx)}.json')


def rule_folder(root, size):
    return os.path.join(root, OUT, f'{size}_rulestart')


def rule_record_path(root, size, combo, idx):
    return os.path.join(rule_folder(root, size), combo, f'{instance_name(idx)}.json')


def schedule_path(rec_path):
    return rec_path[:-len('.json')] + '.schedule.npz'


def rel(path):
    p = os.path.abspath(path)
    return os.path.relpath(p, ROOT) if p.startswith(ROOT + os.sep) else p


def parse_cell(cell):
    size, combo, idx = cell.split(':')
    return size, combo, int(idx)


# ---------------------------------------------------------------------------
# checkpoint, environment, stored greedy results
# ---------------------------------------------------------------------------

def cpu_configs():
    """params.configs with the CPU as device. The environment module reads the device
    when it is first imported, so this runs before that import in every process."""
    from params import configs
    configs.device = 'cpu'
    return configs


def env_masks():
    """Mask options of the active environment: {'first_mask': .., 'mask_ties': ..}."""
    cpu_configs()
    from constraint_family.fjsp_env_family import resolve_mask_option
    flags = REG.MASK_OPTIONS[REG.env()]
    out = {'first_mask': 'legacy', 'mask_ties': 'legacy'}
    out.update({f.lstrip('-'): resolve_mask_option(f.lstrip('-'), v)
                for f, v in zip(flags[0::2], flags[1::2])})
    return out


def resolve_run(model=None, seed='301', development=False):
    """Checkpoint of the cell's policy (p75_deploy_cost.resolve_checkpoint) and the mask
    options of the active environment, refused when the training configuration does not
    carry them (p75_deploy_cost.mask_options) unless `development`."""
    cpu_configs()
    s = None
    if not model:
        try:
            s = REG.select_seed('released')[0] if str(seed) == 'select' else int(seed)
        except (REG.MissingResult, ValueError) as e:
            raise Refusal(f'REFUSING ({REG.describe()}): {e}')
    name, ckpt, cfg = P75.resolve_checkpoint(model, s)
    with open(os.path.join(ROOT, cfg)) as f:
        snap = json.load(f)
    try:
        P75.mask_options(snap, name)
        check = 'the training configuration carries the mask options of the environment'
    except SystemExit as e:
        if not development:
            raise
        check = f'refusal lifted by --development: {e}'
    return dict(model=name, checkpoint=ckpt, config=cfg, training_seed=s,
                environment=REG.env(), mask=env_masks(), mask_check=check)


def nominal_run(model=None, seed='301'):
    """The run a plan names when its checkpoint cannot be resolved (shown by --dry with
    the refusal; a start is refused)."""
    s, tag, name = None, None, model
    if not model:
        if str(seed) == 'select':
            name, tag = f'{REG.PREFIX}{REG._STEM[REG.env()]["released"]}-s<selected>', 's<selected>'
        else:
            s = int(seed)
            try:
                name = REG.PREFIX + REG.model('released', None, s)
            except ValueError:
                name = f'{REG.PREFIX}<released seed {s}>'
    return dict(model=name, checkpoint='(not resolved)', config='(not resolved)',
                training_seed=s, environment=REG.env(), mask=env_masks(),
                mask_check='not checked: the checkpoint is not resolved', tag=tag)


def run_tag(run):
    return run.get('tag') or seed_tag(run['training_seed'], run['model'])


def stored_greedy(model, training_seed, size, combo, idx):
    """{file: makespan} of every stored greedy result of this checkpoint for the cell,
    and a list of notes on files that exist but cannot be used."""
    name = instance_name(idx)
    out, notes = {}, []
    names = instance_names(size, combo)
    p = os.path.join(ROOT, REG.res_dir(size), f'Result_greedy+{model}_{combo}.npy')
    if os.path.exists(p):
        a = np.load(p)
        if name not in names or a.ndim != 2 or a.shape[0] != len(names):
            notes.append(f'{rel(p)} has shape {a.shape} for {len(names)} descriptors')
        else:
            out[rel(p)] = float(a[names.index(name), 0])
    if training_seed is not None:
        try:
            d = REG.schedule_cache(training_seed, size)
        except ValueError as e:
            notes.append(str(e))
            d = None
        if d is not None:
            z = os.path.join(ROOT, d, combo, f'{name}.npz')
            if os.path.exists(z):
                out[rel(z)] = float(np.load(z, allow_pickle=True)['ms'])
    return out, notes


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

class PolicyRoller:
    """The policy and the instance, loaded once in a worker; rollout_batch() returns one
    batch of complete schedules. The model and environment are built as p90 rollout and
    eval_family.py build them (module docstring, ROLLOUTS). spec['first_batch'] sizes
    the environment built before the worker is ready (1 for the greedy rollout)."""

    def __init__(self, spec):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        import torch
        P75.set_torch_threads(torch, 1)         # the first torch calls of the process
        self.torch = torch
        configs = cpu_configs()
        from fjsp_env_same_op_nums import EnvState
        if str(EnvState.device) != 'cpu':
            raise SystemExit(f'REFUSING: the environment module was imported with device '
                             f'{EnvState.device}; the rollouts run on the CPU')
        from constraint_family.public_types import refuse_mismatch, snapshot_source, type_lists
        with open(os.path.join(ROOT, spec['config'])) as f:
            snap = json.load(f)
        refuse_mismatch(snapshot_source(snap), configs.pub_type_source, spec['model'])
        for k in ARCH_KEYS:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v in ('True', 'False'):
                    v = v == 'True'
                setattr(configs, k, v)
        from model.PPO import PPO_initialize
        from common_utils import greedy_select_action, sample_action
        from constraint_family.descriptor import ConstraintDescriptor
        from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
        from constraint_family.public_adapter import load_base_instance
        self.greedy_select_action, self.sample_action = greedy_select_action, sample_action
        self.FJSPEnvFamily = FJSPEnvFamily
        self.dm = resolve_delta_mode(configs)
        if self.dm != 'min':
            raise SystemExit(f'REFUSING: {spec["model"]} was trained with the shaping envelope '
                             f'{self.dm!r}; the stored sampling leaves the environment at its '
                             "default 'min', so this rollout would not be the stored one")
        self.mm = getattr(configs, 'mask_mode', 'nondelay')
        self.pe = getattr(configs, 'pair_extra', 'none')
        self.fm, self.mt = spec['mask']['first_mask'], spec['mask']['mask_ties']
        ppo = PPO_initialize()
        sd = torch.load(os.path.join(ROOT, spec['checkpoint']), map_location=torch.device('cpu'))
        ppo.policy.load_state_dict(sd)
        ppo.policy.eval()
        self.ppo = ppo
        self.n_param = int(sum(p.numel() for p in ppo.policy.parameters()))
        with open(descriptor_path(spec['size'], spec['combination'], spec['instance'])) as f:
            drec = json.load(f)
        self.desc = ConstraintDescriptor.from_json(drec, setup_convention='attached')
        self.stem = drec['base_instance']
        self.jl, self.pt, meta = load_base_instance(os.path.join(ROOT, self.stem))
        self.N, self.M = self.pt.shape
        if meta['op_type'] is not None:
            self.otl, self.mtl = [meta['op_type']], [meta['mch_type']]
        else:
            self.otl, self.mtl = type_lists(configs.pub_type_source, [self.desc], self.M)
        self._k = int(spec.get('first_batch', 1))
        self._env = self.new_env(self._k)       # the first rollout's environment

    def new_env(self, K=1):
        """An environment of K copies of the instance (p90 sample: [instance] * K, the
        descriptor and the type lists repeated, one token per copy)."""
        env = self.FJSPEnvFamily(self.jl.shape[0], self.M, family_mode=True,
                                 mask_mode=self.mm, pair_extra=self.pe,
                                 first_mask=self.fm, mask_ties=self.mt)
        env.delta_mode = self.dm
        env.reward_delta_env = (self.dm != 'zero')
        env.set_initial_data([self.jl] * K, [self.pt] * K, descriptor_list=[self.desc] * K,
                             op_type_list=None if self.otl is None else self.otl * K,
                             mch_type_list=None if self.mtl is None else self.mtl * K)
        env._attach_token()
        return env

    def _forward(self, st):
        with self.torch.no_grad():
            pi, _ = self.ppo.policy(
                fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor, candidate=st.candidate_tensor,
                fea_m=st.fea_m_tensor, mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
                dynamic_pair_mask=st.dynamic_pair_mask_tensor, fea_pairs=st.fea_pairs_tensor,
                op_type=st.op_type_tensor, mch_type=st.mch_type_tensor, token=st.token_tensor)
        return pi

    def rollout_batch(self, greedy, seed, batch=1, deadline=math.inf, clock=time.monotonic):
        """(makespans [B], machine per operation [B, N], completion times [B, N]) of one
        batch: B copies of the instance stepped in one environment after one
        torch.manual_seed(seed), as p90 sample steps its K copies. The greedy rollout is a
        batch of one without a seed. None when the clock reaches `deadline` before the
        batch is complete."""
        B = 1 if greedy else int(batch)
        env = self._env if (self._env is not None and self._k == B) else self.new_env(B)
        self._env = None
        if not greedy:
            self.torch.manual_seed(int(seed))
        state = env.state
        rows = np.arange(B)
        amch = np.full((B, self.N), -1, dtype=int)
        steps = 0
        while not env.done().all():
            if clock() >= deadline:
                return None
            pi = self._forward(state)
            a = (self.greedy_select_action(pi) if greedy else self.sample_action(pi)[0])
            a = a.cpu().numpy()
            amch[rows, env.candidate[rows, a // self.M]] = a % self.M
            state, _, _ = env.step(a)
            steps += 1
        if steps != self.N:
            raise RuntimeError(f'{steps} decisions for {self.N} operations')
        ms = np.asarray(env.current_makespan, dtype=float).reshape(-1)[:B].copy()
        ct = np.stack([np.asarray(env.true_op_ct[k], dtype=float) for k in range(B)])
        return ms, amch, ct

    def info(self):
        return dict(parameter_numel=self.n_param, base_instance=self.stem,
                    operations=int(self.N), machines=int(self.M),
                    env_mask_mode=self.mm, env_pair_extra=self.pe, env_delta_mode=self.dm,
                    torch_version=self.torch.__version__)

    def thread_record(self):
        return P75.thread_record(self.torch)


def worker_loop(roller, worker, base_seed, t0, budget, clock=time.monotonic, batch=1):
    """Rollouts of one worker from now until t0 + budget. Worker 0 starts with the
    greedy schedule (a batch of one); every other rollout is a batch of `batch` schedules
    sampled after one seed, rollout_seed(base_seed, worker, index). A batch that is
    incomplete or completes after the budget is discarded whole. Returns rows
    (SCHEDULE_COLUMNS), the batches that improved this worker's best (the schedule of the
    batch's first row with its lowest makespan) and the clock readings."""
    deadline = t0 + budget
    t_ready = clock()
    rows, improving = [], []
    best, j, discarded = math.inf, 0, False
    while clock() < deadline:
        greedy = worker == 0 and j == 0
        seed = None if greedy else rollout_seed(base_seed, worker, j)
        res = roller.rollout_batch(greedy, seed, 1 if greedy else batch, deadline, clock)
        tc = clock()
        if res is None or tc > deadline:
            discarded = True
            break
        ms, amch, ct = res
        ms = [float(m) for m in ms]
        cpu = round(time.process_time(), 3)
        kind = 'greedy' if greedy else 'sampled'
        for r, m in enumerate(ms):
            rows.append([worker, j, kind, seed, round(tc - t0, 4), round(tc - t_ready, 4), m,
                         cpu, r])
        r = int(np.argmin(ms))                  # the first row with the lowest makespan
        if ms[r] < best:
            best = ms[r]
            improving.append({'index': j, 'row': r, 'makespan': ms[r],
                              'amch': np.asarray(amch[r]),
                              'ct': np.asarray(ct[r], dtype=float)})
        j += 1
    return dict(rows=rows, improving=improving, discarded=discarded, t_ready=t_ready,
                t_stop=clock(), n_batches=j)


def maxrss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0     # KiB on Linux


def last_core():
    """The processor this process last ran on (/proc/self/stat field 39)."""
    with open('/proc/self/stat') as f:
        return int(f.read().rsplit(')', 1)[1].split()[36])


def worker_main(spec, worker, core, q, factory):
    """One worker process: pin to its core, load, roll out, report through q."""
    try:
        os.sched_setaffinity(0, {core})
        for v in NUMPY_VARS:
            os.environ[v] = '1'
        batch = int(spec.get('batch', 1))
        roller = factory(dict(spec, first_batch=1 if worker == 0 else batch))
        rss_load = maxrss_mb()
        c0 = time.process_time()
        out = worker_loop(roller, worker, spec['base_seed'], spec['t0'], spec['budget'],
                          batch=batch)
        c1 = time.process_time()
        wall = out['t_stop'] - out['t_ready']
        msg = dict(worker=worker, core=core, affinity=sorted(os.sched_getaffinity(0)),
                   last_core=last_core(), ready_s=round(out['t_ready'] - spec['t0'], 4),
                   stop_s=round(out['t_stop'] - spec['t0'], 4),
                   cpu_s_window=round(c1 - c0, 4), wall_s_window=round(wall, 4),
                   cpu_share=round((c1 - c0) / wall, 4) if wall > 0 else 0.0,
                   cpu_s_total=round(c1, 3), rss_after_load_mb=round(rss_load, 1),
                   peak_rss_mb=round(maxrss_mb(), 1), discarded_rollout=out['discarded'],
                   n_schedules=len(out['rows']), n_batches=out['n_batches'],
                   threads=roller.thread_record(), info=roller.info(), rows=out['rows'],
                   improving=out['improving'])
        q.put(('ok', worker, msg))
    except BaseException as e:          # reported, never silently dropped
        q.put(('error', worker, f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))


# ---------------------------------------------------------------------------
# worker processes of one cell (both arms)
# ---------------------------------------------------------------------------

def start_workers(spec, target, factory, timeout_s):
    """Start spec['workers'] processes of target(spec, worker, core, q, factory), worker
    w on core w of the mask, and collect their reports. Returns (reports by worker,
    errors, cores of the workers)."""
    mask = list(spec['affinity'])
    W = int(spec['workers'])
    cores = [mask[w % len(mask)] for w in range(W)]
    ctx = mp.get_context('fork')
    q = ctx.Queue()
    procs = [ctx.Process(target=target, args=(spec, w, cores[w], q, factory), daemon=True)
             for w in range(W)]
    for p in procs:
        p.start()
    msgs, errors = {}, []
    give_up = spec['t0'] + spec.get('budget', 0.0) + timeout_s
    while len(msgs) + len(errors) < W:
        try:
            kind, w, m = q.get(timeout=1.0)
        except queue.Empty:
            if time.monotonic() > give_up:
                errors.append(f'no report from workers {sorted(set(range(W)) - set(msgs))} '
                              f'{timeout_s:g} s after the budget')
                break
            if not any(p.is_alive() for p in procs) and q.empty():
                errors.append(f'workers {sorted(set(range(W)) - set(msgs))} ended without a '
                              f'report (exit codes {[p.exitcode for p in procs]})')
                break
            continue
        if kind == 'ok':
            msgs[w] = m
        else:
            errors.append(f'worker {w}: {m}')
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    return msgs, errors, cores


def machine_fields(spec, per, mask, cores, la0, la1):
    """The machine evidence every record of this runner stores."""
    shares = [m['cpu_share'] for m in per]
    return {
        'per_worker': [{k: v for k, v in m.items() if k not in ('rows', 'improving', 'rules')}
                       for m in per],
        'cpu_share': round(float(np.mean(shares)), 4), 'cpu_share_min': min(shares),
        'affinity': mask, 'cores_of_workers': cores,
        'loadavg_start': la0, 'loadavg_end': la1, 'idle_before': spec.get('idle_before'),
        'peak_rss_mb_worker_max': max(m['peak_rss_mb'] for m in per),
        'process_start_s': spec.get('process_start_s'),
        'commit': spec.get('commit'), 'python': platform.python_version(),
        'numpy_version': np.__version__,
        'development_run': bool(spec.get('development')),
        'started': spec.get('started'), 'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
    }


def machine_reason(rec):
    """(False, reason) when the machine evidence of a record fails, else (True, '')."""
    if rec.get('development_run', True):
        return False, 'development run'
    if not rec.get('validated'):
        return False, 'not validated'
    idle = rec.get('idle_before')
    if not idle:
        return False, 'no core sample before the cell'
    if not idle.get('idle'):
        return False, f'cores not idle before the cell: {idle.get("busy_processes", [])[:2]}'
    per = rec.get('per_worker') or []
    mask = rec.get('affinity') or []
    cores = [w.get('core') for w in per]
    if len(per) != rec.get('workers') or len(set(cores)) != len(cores) \
            or any(c not in mask for c in cores):
        return False, f'workers {rec.get("workers")} on cores {cores} of the mask {mask}'
    if any(w.get('affinity') != [w.get('core')] for w in per):
        return False, 'a worker was not pinned to one core'
    low = [(w['worker'], w['cpu_share']) for w in per if w.get('cpu_share', 0) < SHARE_MIN]
    if low:
        return False, f'cpu_share {low[0][1]} < {SHARE_MIN} (worker {low[0][0]})'
    return True, ''


# ---------------------------------------------------------------------------
# one cell of the policy arm
# ---------------------------------------------------------------------------

def merge(msgs):
    """Every completed schedule of every worker, ordered by completion time (the rows of
    one batch stay together, in row order)."""
    rows = [r for m in msgs for r in m['rows']]
    return sorted(rows, key=lambda r: (r[4], r[0], r[1], r[8] if len(r) > 8 else 0))


def trajectory(rows, search_start):
    """Improvements of the best schedule so far, at most one per batch (its first row
    with its lowest makespan): (anytime rows, [[worker, index, row]])."""
    anytime, src, best = [], [], math.inf
    for _, grp in itertools.groupby(rows, key=lambda r: (r[0], r[1])):
        grp = list(grp)
        r = min(grp, key=lambda x: (x[6], x[8] if len(x) > 8 else 0))
        if r[6] < best:
            best = r[6]
            anytime.append([round(r[4] - search_start, 4), r[4], r[7], r[6]])
            src.append([r[0], r[1], r[8] if len(r) > 8 else 0])
    return anytime, src


def validate_real(spec, amch, ct):
    """(violations, (amch, st, ct) in integer hours) with the CP-SAT runner's validators."""
    from constraint_family import cpsat_arms as A
    if spec.get('_case') is None:
        spec['_case'] = A.load_case(spec['size'], spec['combination'],
                                    instance_name(spec['instance']), root=ROOT)
    case = spec['_case']
    ct = np.asarray(ct, dtype=float)
    if not np.all(ct == np.rint(ct)):
        return ['completion times are not integer hours'], None
    sched = A.schedule_from_hint(case, amch, ct)
    return A.validate_all(case, *sched), sched


def run_cell(spec, factory=PolicyRoller, validator=validate_real, timeout_s=300.0):
    """Start the workers of one cell, wait for them, merge and validate. Returns
    (record or None, best schedule (amch, st, ct) or None, errors)."""
    la0 = P75.loadavg()
    mask = list(spec['affinity'])
    W = int(spec['workers'])
    B = int(spec.get('batch', 1))
    msgs, errors, cores = start_workers(spec, worker_main, factory, timeout_s)
    la1 = P75.loadavg()
    if errors:
        return None, None, errors
    per = [msgs[w] for w in range(W)]
    rows = merge(per)
    search_start = min(m['ready_s'] for m in per)
    anytime, src = trajectory(rows, search_start)
    schedules = {(m['worker'], s['index']): s for m in per for s in m['improving']}
    best_sched, n_checked = None, 0
    for (w, j, r), a in zip(src, anytime):
        s = schedules[(w, j)]
        if s['row'] != r or s['makespan'] != a[3]:
            return None, None, [f'worker {w} batch {j}: stored row {s["row"]} '
                                f'({s["makespan"]:g}), trajectory row {r} ({a[3]:g})']
        viol, sched = validator(spec, s['amch'], s['ct'])
        n_checked += 1
        if viol:
            return None, None, [f'schedule of worker {w} batch {j} row {r} (makespan '
                                f'{a[3]:g}) fails validation: {viol[:3]}']
        if abs(float(np.max(sched[2])) - a[3]) > 1e-9:
            return None, None, [f'schedule of worker {w} batch {j} row {r}: makespan {a[3]:g} '
                                f'!= latest completion {float(np.max(sched[2])):g}']
        best_sched = sched
    greedy = next((r for r in rows if r[2] == 'greedy'), None)
    best = anytime[-1] if anytime else None
    best_row = None if best is None else next(x for x in rows if [x[0], x[1], x[8]] == src[-1])
    rec = {
        'arm': ARM,
        'method': METHOD.replace('W single', f'{W} single').replace('of B ', f'of {B} '),
        'size': spec['size'], 'combination': spec['combination'],
        'regime': spec['combination'], 'instance': instance_name(spec['instance']),
        'model': spec['model'], 'checkpoint': spec['checkpoint'], 'config': spec['config'],
        'training_seed': spec['training_seed'], 'environment': spec['environment'],
        'mask': spec['mask'], 'mask_check': spec['mask_check'],
        'budget_s': float(spec['budget']), 'budget_clock': 'wall',
        'clock': ('time.monotonic; t from the start of the cell\'s process (imports, '
                  'checkpoint load and environment construction included), t_resident from '
                  'the moment the worker\'s policy and first environment were ready'),
        'workers': W, 'batch': B, 'base_seed': int(spec['base_seed']),
        'seed_rule': ('one seed per batch: seed of batch index j of worker w = base_seed * '
                      '2**26 + w * 2**20 + j, torch.manual_seed before the batch of B copies '
                      '(p90 sample with K = B and seed_test = seed); greedy: a batch of one, '
                      'no seed'),
        'schedules_columns': SCHEDULE_COLUMNS, 'schedules': rows,
        'n_schedules': len(rows), 'n_sampled': sum(r[2] == 'sampled' for r in rows),
        'n_batches': sum(m['n_batches'] for m in per),
        'anytime_columns': ANYTIME_COLUMNS, 'anytime': anytime, 'anytime_from': src,
        'search_start_wall_s': search_start,
        'makespan': None if best is None else best[3],
        'objective': None if best is None else float(best[3]),
        'best_from': None if best_row is None else dict(zip(SCHEDULE_COLUMNS, best_row)),
        'greedy_makespan': None if greedy is None else greedy[6],
        'greedy_t': None if greedy is None else greedy[4],
        'greedy_t_resident': None if greedy is None else greedy[5],
        'seeds': [], 'seed_best': None,     # no start schedule (timed_campaign_plan reader)
        'validated': True, 'violations': [], 'validator': VALIDATOR,
        'trajectory_schedules_validated': n_checked,
        'schedule_file': None,
    }
    rec.update(machine_fields(spec, per, mask, cores, la0, la1))
    return rec, best_sched, []


def classify(rec, stored=None):
    """(valid, reason) of a record (module docstring, VALIDITY). stored: {file:
    makespan} of the stored greedy results (default: read now from the repository)."""
    ok, why = machine_reason(rec)
    if not ok:
        return ok, why
    if rec.get('budget_clock') != 'wall':
        return False, f'budget clock {rec.get("budget_clock")}'
    g = rec.get('greedy_makespan')
    if g is None:
        return False, 'no greedy schedule within the budget'
    if stored is None:
        stored, _ = stored_greedy(rec['model'], rec.get('training_seed'), rec['size'],
                                  rec['combination'], int(rec['instance'].split('_')[1]))
    diff = {k: v for k, v in stored.items() if abs(v - g) > 1e-9}
    if diff:
        return False, f'greedy makespan {g:g} differs from the stored {diff}'
    return True, ('share' + (f'; greedy equal to {len(stored)} stored value(s)' if stored
                             else '; no stored greedy result to compare'))


def write_record(path, rec, sched=None):
    """Write the schedule file and the record atomically; never overwrite a record.
    Returns False when the record exists."""
    if os.path.exists(path):
        return False
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    if sched is not None:
        sp = schedule_path(path)
        tmp = os.path.join(d, '.' + os.path.basename(sp) + '.tmp.npz')
        amch, st, ct = sched
        np.savez(tmp, amch=amch, st=st, ct=ct, ms=float(np.max(ct)))
        os.replace(tmp, sp)
        rec = dict(rec, schedule_file=os.path.basename(sp))
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(rec, f)
    try:
        os.link(tmp, path)          # fails when the record exists
    except FileExistsError:
        os.unlink(tmp)
        return False
    os.unlink(tmp)
    return True


def on_term_children(signum, frame):
    for p in mp.active_children():
        p.terminate()
    sys.exit(143)


def cell_main(a):
    """The process of one cell: build the spec, run, write the record."""
    size, combo, idx = parse_cell(a._cell)
    t0 = float(a._t0) if a._t0 else _T_MODULE
    run = resolve_run(a.model, a.seed, a.development)
    path = record_path(a.root, size, a.budget, a.batch, run_tag(run), combo, idx)
    if os.path.exists(path):
        print(f'skip {rel(path)} (exists)', flush=True)
        return 2
    spec = dict(run, size=size, combination=combo, instance=idx, budget=float(a.budget),
                workers=a.workers, batch=a.batch, base_seed=a.base_seed, t0=t0,
                process_start_s=round(_T_MODULE - t0, 4),
                affinity=sorted(os.sched_getaffinity(0)), development=a.development,
                idle_before=json.loads(a._idle) if a._idle else None, commit=a._commit,
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    signal.signal(signal.SIGTERM, on_term_children)
    rec, sched, errors = run_cell(spec)
    if errors:
        print(f'NO RECORD {size} {combo} {instance_name(idx)}:\n  ' + '\n  '.join(errors),
              flush=True)
        return 3
    stored, notes = stored_greedy(run['model'], run['training_seed'], size, combo, idx)
    rec['greedy_stored'], rec['greedy_stored_notes'] = stored, notes
    rec['valid'], rec['valid_reason'] = classify(rec, stored=stored)
    if not write_record(path, rec, sched):
        print(f'NOT WRITTEN: {rel(path)} appeared during the cell', flush=True)
        return 2
    print(f'{size} {combo} {instance_name(idx)}: {rec["n_schedules"]} schedules in '
          f'{rec["n_batches"]} batches of up to {rec["batch"]}, greedy '
          f'{rec["greedy_makespan"]} at t {rec["greedy_t"]}, best {rec["makespan"]}, '
          f'shares {[w["cpu_share"] for w in rec["per_worker"]]}, '
          f'{"valid" if rec["valid"] else "INVALID"} ({rec["valid_reason"]}) -> {rel(path)}',
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# rule arm: the dispatching-rule start schedule of the search arms
# ---------------------------------------------------------------------------

def rule_cache_entry(size, combo, idx):
    """The stored rule start schedule of the cell (warmstart_cache_xmask_<size>):
    {file, rule_ms {rule: makespan}, rule, ms, amch, ct}, or None."""
    p = os.path.join(ROOT, RULE_CACHE.format(size=size), combo, f'{instance_name(idx)}.npz')
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    return dict(file=rel(p),
                rule_ms={str(k): float(v) for k, v in zip(z['rule_names'], z['rule_ms'])},
                rule=str(z['rule']), ms=float(z['ms']), amch=np.asarray(z['amch']),
                ct=np.asarray(z['ct'], dtype=float))


class RuleRoller:
    """The instance loaded once in a worker, as p92 and p86 load it; rollout(rule) is
    p86_extend_scale_sets.masked_rollout and eligible() the cache builder's test. The
    cell's process imports p86 before the workers start."""

    def __init__(self, spec):
        import p86_extend_scale_sets as p86
        with open(descriptor_path(spec['size'], spec['combination'], spec['instance'])) as f:
            rec = json.load(f)
        self.stem = rec['base_instance']
        self.jl, self.pt, _ = p86.load_instance(os.path.join(ROOT, self.stem))
        self.desc = p86.load_stored_descriptor(rec)
        self.p86 = p86

    def rollout(self, rule):
        amch, ct, ms = self.p86.masked_rollout(rule, self.jl, self.pt, self.desc)
        return float(ms), np.asarray(amch), np.asarray(ct, dtype=float)

    def eligible(self, amch, ct):
        ok, c6 = self.p86._eligible(self.jl, self.pt, self.desc, amch, ct)
        return bool(ok), str(c6)

    def info(self):
        return dict(base_instance=self.stem, operations=int(self.pt.shape[0]),
                    machines=int(self.pt.shape[1]))


def rule_worker_main(spec, worker, core, q, factory):
    """One worker of the rule arm: pin, load the instance, roll out its rules (rule k for
    worker k; with fewer workers the rules in turn), then check them."""
    try:
        os.sched_setaffinity(0, {core})
        for v in NUMPY_VARS:
            os.environ[v] = '1'
        roller = factory(spec)
        rss_load = maxrss_mb()
        t0, W = spec['t0'], int(spec['workers'])
        t_ready = time.monotonic()
        c0 = time.process_time()
        out = []
        for rule in list(spec['rules'])[worker::W]:
            wa, ca = time.monotonic(), time.process_time()
            ms, amch, ct = roller.rollout(rule)
            tc, cb = time.monotonic(), time.process_time()
            out.append(dict(rule=rule, makespan=ms, t=round(tc - t0, 4),
                            t_resident=round(tc - t_ready, 4), cpu_s=round(cb - ca, 4),
                            wall_s=round(tc - wa, 4),
                            share=round((cb - ca) / (tc - wa), 4) if tc > wa else 0.0,
                            worker=worker, core=core, cpu_s_worker=round(cb, 3),
                            amch=amch, ct=ct))
        t_stop = time.monotonic()
        c1 = time.process_time()
        wall = t_stop - t_ready
        tk = time.monotonic()
        for r in out:                    # after the timed window: the cache builder's test
            r['eligible'], r['c6'] = roller.eligible(r['amch'], r['ct'])
        check_s = time.monotonic() - tk
        msg = dict(worker=worker, core=core, affinity=sorted(os.sched_getaffinity(0)),
                   last_core=last_core(), ready_s=round(t_ready - t0, 4),
                   stop_s=round(t_stop - t0, 4), cpu_s_window=round(c1 - c0, 4),
                   wall_s_window=round(wall, 4),
                   cpu_share=round((c1 - c0) / wall, 4) if wall > 0 else 0.0,
                   cpu_s_total=round(c1, 3), rss_after_load_mb=round(rss_load, 1),
                   peak_rss_mb=round(maxrss_mb(), 1), check_s=round(check_s, 4),
                   info=roller.info(), rules=out)
        q.put(('ok', worker, msg))
    except BaseException as e:
        q.put(('error', worker, f'{type(e).__name__}: {e}\n{traceback.format_exc()}'))


def best_rule(rules):
    """The start schedule's rule: the eligible rule with the lowest makespan, equal
    makespans going to the earlier rule of RULE_ORDER (the cache's tie rule)."""
    by = {r['rule']: r for r in rules}
    ms = [by[k]['makespan'] if k in by and by[k]['eligible'] else math.inf
          for k in RULE_ORDER]
    k = int(np.argmin(ms))
    return (None, None) if not np.isfinite(ms[k]) else (RULE_ORDER[k], ms[k])


def run_rule_cell(spec, factory=RuleRoller, validator=validate_real, timeout_s=300.0):
    """The four rule rollouts of one cell on the workers. Returns (record or None, start
    schedule (amch, st, ct) or None, errors, {rule: (amch, ct)})."""
    la0 = P75.loadavg()
    mask = list(spec['affinity'])
    W = int(spec['workers'])
    msgs, errors, cores = start_workers(spec, rule_worker_main, factory, timeout_s)
    la1 = P75.loadavg()
    if errors:
        return None, None, errors, {}
    per = [msgs[w] for w in range(W)]
    share = {m['worker']: m['cpu_share'] for m in per}
    raw, rules = {}, []
    for m in per:
        for r in m['rules']:
            raw[r['rule']] = (r['amch'], r['ct'])
            rules.append({k: v for k, v in r.items() if k not in ('amch', 'ct')})
    order = {k: i for i, k in enumerate(RULE_ORDER)}
    rules.sort(key=lambda r: order.get(r['rule'], len(order)))
    sched_of, violations = {}, []
    for r in rules:
        viol, sched = validator(spec, *raw[r['rule']])
        r['violations'] = list(viol)[:3]
        r['worker_share'] = share[r['worker']]
        if not viol and abs(float(np.max(sched[2])) - r['makespan']) > 1e-9:
            r['violations'].append(f'makespan {r["makespan"]:g} != latest completion '
                                   f'{float(np.max(sched[2])):g}')
        sched_of[r['rule']] = sched
    rule, ms = best_rule(rules)
    if rule is None:
        violations.append('no eligible rule schedule')
    else:
        violations += [f'{rule}: {v}' for v in next(r for r in rules
                                                    if r['rule'] == rule)['violations']]
    if sorted(r['rule'] for r in rules) != sorted(spec['rules']):
        violations.append(f'rules {[r["rule"] for r in rules]} for {list(spec["rules"])}')
    search_start = min(m['ready_s'] for m in per)
    anytime, cur = [], math.inf
    for r in sorted(rules, key=lambda r: (r['t'], order.get(r['rule'], 0))):
        if r['eligible'] and r['makespan'] < cur:
            cur = r['makespan']
            anytime.append([round(r['t'] - search_start, 4), r['t'], r['cpu_s_worker'], cur])
    rec = {
        'arm': RULE_ARM, 'method': RULE_METHOD,
        'size': spec['size'], 'combination': spec['combination'],
        'regime': spec['combination'], 'instance': instance_name(spec['instance']),
        'environment_active': spec.get('environment'),
        'mask': {'first_mask': 'exact', 'mask_ties': 'exact'},
        'mask_check': 'p86_extend_scale_sets.masked_rollout constructs and checks the exact mask',
        'rule_seed': RULE_SEED_NOTE, 'rule_order': list(RULE_ORDER),
        'tie_rule': ('the eligible rule with the lowest makespan; equal makespans go to the '
                     'earlier rule of FIFO, MOR, SPT, MWKR'),
        'budget_s': None, 'budget_clock': 'wall',
        'clock': ('time.monotonic; t from the start of the cell\'s process (imports and the '
                  'instance load included), t_resident from the moment the worker\'s '
                  'instance was loaded'),
        'workers': W, 'rules': rules,
        'best_rule': rule, 'best_makespan': ms, 'makespan': ms, 'objective': ms,
        't_all': max(r['t'] for r in rules) if rules else None,
        't_best': next((r['t'] for r in rules if r['rule'] == rule), None),
        'check_s': max(m['check_s'] for m in per),
        'anytime_columns': ANYTIME_COLUMNS, 'anytime': anytime,
        'search_start_wall_s': search_start,
        'validated': not violations, 'violations': violations,
        'validator': (VALIDATOR + '; eligibility: p86_extend_scale_sets._eligible '
                      '(reference_sim.validate_schedule and the C6 status)'),
        'schedule_file': None,
    }
    rec.update(machine_fields(spec, per, mask, cores, la0, la1))
    return rec, (sched_of.get(rule) if not violations else None), [], raw


def classify_rules(rec, cache=None):
    """(valid, reason) of a rule-arm record (module docstring, RULE ARM). cache: the
    stored entry of rule_cache_entry (default: read now from the repository)."""
    ok, why = machine_reason(rec)
    if not ok:
        return ok, why
    have = {r['rule']: r for r in rec.get('rules') or []}
    if sorted(have) != sorted(RULE_ORDER):
        return False, f'rules {sorted(have)}'
    if cache is None:
        cache = rule_cache_entry(rec['size'], rec['combination'],
                                 int(rec['instance'].split('_')[1]))
    if cache is None:
        return False, 'no stored rule start schedule to compare'
    for k in RULE_ORDER:
        want = cache['rule_ms'].get(k)
        if want is None or abs(have[k]['makespan'] - want) > 1e-9:
            return False, (f'{k} makespan {have[k]["makespan"]:g} differs from the stored '
                           f'{want} ({cache["file"]})')
    if rec.get('best_rule') != cache['rule'] or rec.get('best_makespan') is None \
            or abs(rec['best_makespan'] - cache['ms']) > 1e-9:
        return False, (f'best {rec.get("best_rule")} {rec.get("best_makespan")} differs from '
                       f'the stored {cache["rule"]} {cache["ms"]:g}')
    return True, f'share; four rule makespans and the best rule equal to {cache["file"]}'


def rule_cell_main(a):
    """The process of one rule-arm cell."""
    size, combo, idx = parse_cell(a._cell)
    t0 = float(a._t0) if a._t0 else _T_MODULE
    path = rule_record_path(a.root, size, combo, idx)
    if os.path.exists(path):
        print(f'skip {rel(path)} (exists)', flush=True)
        return 2
    cpu_configs()
    # imported here, before the workers start, so that they inherit them (p86 also sets
    # the working folder to the repository root, where its paths are relative to)
    import p86_extend_scale_sets as p86
    import common_utils  # noqa: F401
    from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: F401
    if tuple(p86.RULE_ORDER) != RULE_ORDER:
        raise Refusal(f'REFUSING: p86 rule order {p86.RULE_ORDER}, expected {RULE_ORDER}')
    spec = dict(size=size, combination=combo, instance=idx, workers=a.workers,
                rules=RULE_ORDER, t0=t0, budget=0.0, environment=REG.env(),
                process_start_s=round(_T_MODULE - t0, 4),
                affinity=sorted(os.sched_getaffinity(0)), development=a.development,
                idle_before=json.loads(a._idle) if a._idle else None, commit=a._commit,
                started=time.strftime('%Y-%m-%d %H:%M:%S'))
    signal.signal(signal.SIGTERM, on_term_children)
    rec, sched, errors, raw = run_rule_cell(spec)
    if errors:
        print(f'NO RECORD {size} {combo} {instance_name(idx)}:\n  ' + '\n  '.join(errors),
              flush=True)
        return 3
    cache = rule_cache_entry(size, combo, idx)
    rec['stored'] = None if cache is None else {k: v for k, v in cache.items()
                                                if k not in ('amch', 'ct')}
    b = rec['best_rule']
    rec['stored_schedule_equal'] = (None if cache is None or b not in raw else bool(
        np.array_equal(np.asarray(raw[b][0]), cache['amch'])
        and np.array_equal(np.asarray(raw[b][1], dtype=float), cache['ct'])))
    rec['valid'], rec['valid_reason'] = classify_rules(rec, cache)
    if not write_record(path, rec, sched):
        print(f'NOT WRITTEN: {rel(path)} appeared during the cell', flush=True)
        return 2
    print(f'{size} {combo} {instance_name(idx)}: '
          + ', '.join(f'{r["rule"]} {r["makespan"]:g} at t {r["t"]}' for r in rec['rules'])
          + f'; best {b} {rec["best_makespan"]}, t_all {rec["t_all"]}, shares '
          f'{[w["cpu_share"] for w in rec["per_worker"]]}, '
          f'{"valid" if rec["valid"] else "INVALID"} ({rec["valid_reason"]}) -> {rel(path)}',
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# campaign
# ---------------------------------------------------------------------------

def cell_list(a):
    """(size, instance, combination) of every cell of the plan, in run order."""
    out = []
    for size in a.sizes:
        spec = a.instances or TCP.DEFAULT_INSTANCES.get(size, TCP.OTHER_DEFAULT_INSTANCES)
        for idx in TCP.parse_instances(spec):
            for combo in a.combos:
                out.append((size, idx, combo))
    return out


def plan_state(c, p, want, check, max_invalid):
    """The state of one cell: done (valid record), invalid (moved aside and run again),
    held, foreign (a record of another plan), run."""
    if os.path.exists(p):
        rec = json.load(open(p))
        bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
        if bad:
            c.update(state='foreign', reason=f'{bad}, plan {want}')
        else:
            ok, why = check(rec)
            c.update(state='done' if ok else 'invalid', reason=why)
    elif TCP.n_invalid(p) >= max_invalid:
        c.update(state='held', reason=f'{TCP.n_invalid(p)} invalid records in '
                                      f'{rel(TCP.invalid_dir(p))}')
    else:
        c.update(state='run', reason='')
    return c


def plan_cells(a, run):
    """Every cell of the policy arm with its record path and state (plan_state)."""
    tag = run_tag(run)
    want = dict(budget_s=float(a.budget), workers=a.workers, batch=a.batch,
                base_seed=a.base_seed, model=run['model'], environment=run['environment'])
    cells = []
    for size, idx, combo in cell_list(a):
        p = record_path(a.root, size, a.budget, a.batch, tag, combo, idx)
        cells.append(plan_state(dict(size=size, combination=combo, instance=idx, record=p),
                                p, want, classify, a.max_invalid))
    return cells


def plan_rule_cells(a):
    """Every cell of the rule arm with its record path and state (plan_state)."""
    want = dict(arm=RULE_ARM, workers=a.workers)
    cells = []
    for size, idx, combo in cell_list(a):
        p = rule_record_path(a.root, size, combo, idx)
        cells.append(plan_state(dict(size=size, combination=combo, instance=idx, record=p),
                                p, want, classify_rules, a.max_invalid))
    return cells


def check_cells(cells):
    """Descriptors and base-instance files of every cell; list of missing files."""
    miss = []
    for c in cells:
        dp = descriptor_path(c['size'], c['combination'], c['instance'])
        if not os.path.exists(dp):
            miss.append(rel(dp))
            continue
        stem = os.path.join(ROOT, json.load(open(dp))['base_instance'])
        miss += [rel(stem + e) for e in ('.fjs', '.meta.json') if not os.path.exists(stem + e)]
    return miss


def slot_processes(cores):
    """Lines of `ps` for the processes that last ran on one of the cores with a lifetime
    CPU share of at least 1%, and for every process above BUSY_PCT anywhere (this
    process and its ancestors left out)."""
    out = subprocess.run(['ps', '-eo', 'pid,psr,pcpu,etime,args', '--no-headers'],
                         capture_output=True, text=True).stdout.splitlines()
    mine = P75._own_pids()
    rows = []
    for line in out:
        f = line.split(None, 4)
        if len(f) < 5 or int(f[0]) in mine:
            continue
        if (int(f[1]) in cores and float(f[2]) >= 1.0) or float(f[2]) > BUSY_PCT:
            rows.append(line[:160])
    return rows


def idle_sample(cores):
    """The core sample of p75_deploy_cost before a measurement (busy_processes, core_busy,
    load average), as stored in each record."""
    la = P75.loadavg()
    busy = P75.busy_processes(cores, IDLE_WINDOW)
    per_core = P75.core_busy(cores, window=1.0)
    return {'loadavg': la, 'busy_processes': busy, 'core_busy': per_core,
            'idle': not busy, 'window_s': IDLE_WINDOW,
            'time': time.strftime('%Y-%m-%d %H:%M:%S')}


def cell_command(a, run, c, t0=None, idle=None, commit=None):
    cmd = [sys.executable, os.path.abspath(__file__), '--_cell',
           f'{c["size"]}:{c["combination"]}:{c["instance"]}', '--budget', bnum(a.budget),
           '--workers', str(a.workers), '--batch', str(a.batch), '--base_seed',
           str(a.base_seed), '--root', a.root]
    # the cell's process resolves the same checkpoint: the selected seed, or the name
    cmd += (['--seed', str(run['training_seed'])] if run['training_seed'] is not None
            else ['--model', run['model']])
    if a.development:
        cmd.append('--development')
    if t0 is not None:
        cmd += ['--_t0', repr(t0), '--_idle', json.dumps(idle), '--_commit', str(commit)]
    return cmd


def rule_cell_command(a, c, t0=None, idle=None, commit=None):
    cmd = [sys.executable, os.path.abspath(__file__), '--arm', 'rules', '--_cell',
           f'{c["size"]}:{c["combination"]}:{c["instance"]}', '--workers', str(a.workers),
           '--root', a.root]
    if a.development:
        cmd.append('--development')
    if t0 is not None:
        cmd += ['--_t0', repr(t0), '--_idle', json.dumps(idle), '--_commit', str(commit)]
    return cmd


def machine_lines(cores, to_run):
    return [f'load average {" ".join(map(str, P75.loadavg()))}; processes on cores {cores} '
            f'(lifetime share at least 1%) and above {BUSY_PCT:g}% anywhere:'] + [
        f'  {x}' for x in slot_processes(set(cores))] + ['cells to run:'] + [
        f'  {c["size"]} {c["combination"]} {instance_name(c["instance"])}' for c in to_run]


def launch_note(a, run, cells, to_run, cores, commit):
    tag = run_tag(run)
    sizes = sorted({c['size'] for c in cells})
    lines = [
        f'=== policy sampling start {time.strftime("%Y-%m-%d %H:%M:%S")}, commit {commit}, '
        f'cores {cores}, {a.workers} workers, batch {a.batch}, budget {bnum(a.budget)} s, '
        f'base seed {a.base_seed}, {len(to_run)} of {len(cells)} cells to run'
        + (' (DEVELOPMENT: not a measurement)' if a.development else ''),
        '1. Purpose: the policy at the computing effort of the CP-SAT arm, eight single-thread '
        'workers on the slot for the budget: worker 0 returns the greedy schedule first, then '
        f'every worker samples batches of {a.batch} schedules from the policy; the best '
        'schedule over time is the policy curve of the time-budget figure and its rows at '
        '10 s and 60 s of the decision table.',
        '2. Expected: the best sampled schedule is at or below the greedy schedule and falls '
        'fastest in the first minute; if it never falls below the greedy schedule, the policy '
        'curve is flat at the greedy value and the comparison at equal effort keeps the '
        'greedy schedule.',
        '3. Contamination: new records only (written atomically, an existing record is never '
        'overwritten; an invalid one moves to invalid/ and its cell runs again); instances '
        '000-002 of the production sets selected the search configurations and are not '
        'evidence; the cores are sampled before every cell (refused above 5% of a core); '
        'every record stores each worker\'s core and CPU share (valid at 0.9 or more for '
        'every worker), the mask and the load average.',
        f'4. Data: data/FAMILY/{",".join(sizes)}/desc/<combination>/instance_<i> and the base '
        f'instance each names; checkpoint {run["checkpoint"]} ({run["environment"]} '
        f'environment, masks {run["mask"]}; {run["mask_check"]}); records under '
        f'{rel(folder(a.root, "<size>", a.budget, a.batch, tag))}; the greedy makespan is '
        'compared with the stored greedy results of the checkpoint; every schedule on the '
        'trajectory is validated before the record.',
    ]
    return lines + machine_lines(cores, to_run)


def rule_launch_note(a, cells, to_run, cores, commit):
    sizes = sorted({c['size'] for c in cells})
    lines = [
        f'=== rule start schedules start {time.strftime("%Y-%m-%d %H:%M:%S")}, commit '
        f'{commit}, cores {cores}, {a.workers} workers, {len(to_run)} of {len(cells)} cells '
        'to run' + (' (DEVELOPMENT: not a measurement)' if a.development else ''),
        '1. Purpose: the time at which the rule start schedule of the search arms exists on '
        'the campaign\'s cores, so that every arm\'s clock can count its start schedule; four '
        'single-thread workers, one dispatching rule each (FIFO, MOR, SPT, MWKR).',
        '2. Expected: every rule makespan equals the stored one and the start schedule exists '
        'within a few seconds of the process start; a differing makespan makes the record '
        'invalid, and the stored start schedules then cannot take this time until the '
        'difference is explained.',
        '3. Contamination: new records only (atomic, never overwritten, invalid ones moved to '
        'invalid/); the cores are sampled before every cell; every record stores each '
        'worker\'s core and CPU share (valid at 0.9 or more), the mask and the load average.',
        f'4. Data: data/FAMILY/{",".join(sizes)}/desc/<combination>/instance_<i> and the base '
        'instance each names, loaded as p92 and p86 load them; p86_extend_scale_sets.'
        'masked_rollout under the exact mask with numpy seed 50 before each rollout; every '
        f'makespan compared with {RULE_CACHE}/<combination>/instance_<i>.npz; records under '
        f'{rel(rule_folder(a.root, "<size>"))}.',
    ]
    return lines + machine_lines(cores, to_run)


def append_note(a, lines):
    p = os.path.join(a.root, NOTES)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a') as f:
        f.write('\n'.join(lines) + '\n')


def common_refusals(a, cores, cells):
    out = []
    if a.workers > len(cores) and not a.development:
        out.append(f'{a.workers} workers need {a.workers} cores, the mask has {cores}')
    if a.development and os.path.abspath(a.root) == ROOT:
        out.append('--development needs --root outside the repository root, so that a '
                   'development record never reaches the result folders')
    miss = check_cells(cells)
    if miss:
        out.append(f'{len(miss)} missing input file(s), first {miss[0]}')
    foreign = [c for c in cells if c['state'] == 'foreign']
    if foreign:
        out.append(f'{len(foreign)} record(s) of another plan, first '
                   f'{rel(foreign[0]["record"])}: {foreign[0]["reason"]}')
    return out


def print_counts(a, cells, to_run):
    count = {s: sum(c['state'] == s for c in cells)
             for s in ('done', 'invalid', 'held', 'foreign', 'run')}
    print(f'cells: {len(cells)} (sizes {" ".join(a.sizes)}; combinations {" ".join(a.combos)}; '
          f'instances {a.instances or "each size default"})')
    print(f'  done {count["done"]}, invalid (moved aside, run again) {count["invalid"]}, held '
          f'{count["held"]}, foreign {count["foreign"]}, to run {len(to_run)}')


def campaign_loop(a, cores, to_run, command, check, describe):
    """The cells one after the other (both arms): the hold file, the core sample, one
    process per cell, classification, invalid records moved aside, the stop after
    STOP_AFTER_INVALID cells in a row without a valid record."""
    hold = os.path.join(a.root, HOLD)
    child = {'p': None}

    def on_term(signum, frame):
        if child['p'] is not None and child['p'].poll() is None:
            child['p'].terminate()
            child['p'].wait()
        append_note(a, [f'--- stopped by signal {signum} {time.strftime("%Y-%m-%d %H:%M:%S")}; '
                        'the running cell wrote no record'])
        sys.exit(143)
    signal.signal(signal.SIGTERM, on_term)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', **{v: '1' for v in NUMPY_VARS})
    commit = a._commit_now
    bad_row = 0
    for c in to_run:
        if os.path.exists(hold):
            append_note(a, [f'--- held by {rel(hold)} {time.strftime("%Y-%m-%d %H:%M:%S")}'])
            print(f'held by {rel(hold)}')
            return 0
        if c['state'] == 'invalid':
            TCP.move_invalid(c['record'], c['reason'])
        idle = idle_sample(cores)
        cid = f'{c["size"]} {c["combination"]} {instance_name(c["instance"])}'
        if not idle['idle'] and not a.development:
            append_note(a, [f'--- REFUSED {cid} {idle["time"]}: cores {cores} not idle:']
                        + [f'  {x}' for x in idle['busy_processes']])
            print(f'REFUSING {cid}: cores not idle ({idle["busy_processes"][:3]})')
            return 1
        t0 = time.monotonic()
        child['p'] = subprocess.Popen(command(c, t0, idle, commit), cwd=ROOT, env=env)
        rc = child['p'].wait()
        child['p'] = None
        p = c['record']
        if rc != 0 or not os.path.exists(p):
            append_note(a, [f'--- {cid}: no record (exit {rc})'])
            bad_row += 1
        else:
            rec = json.load(open(p))
            ok, why = check(rec)
            append_note(a, [f'--- {cid}: {"valid" if ok else "invalid"} ({why}), '
                            + describe(rec)])
            if ok:
                bad_row = 0
            else:
                TCP.move_invalid(p, why)
                # a development record is invalid by construction: not a reason to stop
                bad_row = 0 if (a.development and why == 'development run') else bad_row + 1
        if bad_row >= STOP_AFTER_INVALID:
            append_note(a, [f'--- STOP: {bad_row} cells in a row without a valid record'])
            print(f'STOP: {bad_row} cells in a row without a valid record')
            return 1
    append_note(a, [f'--- finished {time.strftime("%Y-%m-%d %H:%M:%S")}'])
    return 0


def start_or_stop(a, refusals, to_run):
    """None when the campaign may start, else its exit code (dry run, refusal, nothing to
    run, hold file)."""
    for r in refusals:
        print(f'REFUSAL: {r}')
    if a.dry:
        print('dry run: nothing started, nothing written')
        return 1 if refusals else 0
    if refusals:
        return 1
    if not to_run:
        print('nothing to run')
        return 0
    hold = os.path.join(a.root, HOLD)
    if os.path.exists(hold):
        print(f'held by {rel(hold)}; remove it to start')
        return 0
    return None


def campaign(a):
    REG.banner()
    cores = sorted(os.sched_getaffinity(0))
    refusals = []
    try:
        run = resolve_run(a.model, a.seed, a.development)
    except SystemExit as e:
        refusals.append(str(e))
        run = nominal_run(a.model, a.seed)
    if a.workers > 2 ** W_BITS:
        refusals.append(f'at most {2 ** W_BITS} workers')
    cells = plan_cells(a, run)
    refusals += common_refusals(a, cores, cells)
    to_run = [c for c in cells if c['state'] in ('run', 'invalid')]
    tag = run_tag(run)
    print(f'policy sampling with a time budget: {run["model"]} ({run["environment"]} '
          f'environment, masks {run["mask"]}; {run["mask_check"]}), budget {bnum(a.budget)} s, '
          f'{a.workers} workers on cores {cores}, batch {a.batch}, base seed {a.base_seed}'
          + ('; DEVELOPMENT (records are not measurements)' if a.development else ''))
    print_counts(a, cells, to_run)
    print(f'slot-hours to run: {len(to_run)} x ({bnum(a.budget)} + {OVERHEAD_S:g}) s = '
          f'{len(to_run) * (a.budget + OVERHEAD_S) / 3600:.2f}')
    for size in a.sizes:
        for combo in a.combos:
            cs = [c for c in cells if c['size'] == size and c['combination'] == combo]
            have = [c for c in cs if stored_greedy(run['model'], run['training_seed'], size,
                                                   combo, c['instance'])[0]]
            print(f'  stored greedy to compare at {size} {combo}: {len(have)} of {len(cs)} cells')
    print(f'records: {rel(folder(a.root, "<size>", a.budget, a.batch, tag))}/<combination>/'
          f'instance_<i>.json')
    for c in [c for c in cells if c['state'] in ('invalid', 'held', 'foreign')][:10]:
        print(f'  {c["state"]} {c["size"]} {c["combination"]} {instance_name(c["instance"])}: '
              f'{c["reason"]}')
    if to_run:
        print('first cell command: ' + ' '.join(cell_command(a, run, to_run[0])))
    stop = start_or_stop(a, refusals, to_run)
    if stop is not None:
        return stop
    a._commit_now = TCP.git_commit()
    append_note(a, launch_note(a, run, cells, to_run, cores, a._commit_now))
    return campaign_loop(
        a, cores, to_run, lambda c, t0, idle, commit: cell_command(a, run, c, t0, idle, commit),
        classify, lambda rec: (f'{rec["n_schedules"]} schedules, best {rec["makespan"]}, shares '
                               f'{[w["cpu_share"] for w in rec["per_worker"]]}'))


def rule_campaign(a):
    REG.banner()
    cores = sorted(os.sched_getaffinity(0))
    refusals = []
    if a.workers != RULE_WORKERS and not a.development:
        refusals.append(f'{a.workers} workers: the rule arm runs one rule per worker, '
                        f'{RULE_WORKERS} workers')
    cells = plan_rule_cells(a)
    refusals += common_refusals(a, cores, cells)
    no_cache = [c for c in cells if rule_cache_entry(c['size'], c['combination'],
                                                     c['instance']) is None]
    if no_cache and not a.development:
        refusals.append(f'{len(no_cache)} cell(s) without a stored rule start schedule to '
                        f'compare, first {no_cache[0]["size"]} {no_cache[0]["combination"]} '
                        f'{instance_name(no_cache[0]["instance"])}')
    to_run = [c for c in cells if c['state'] in ('run', 'invalid')]
    print(f'rule start schedules: {", ".join(RULE_ORDER)} (p86_extend_scale_sets.'
          f'masked_rollout, exact mask, {RULE_SEED_NOTE}), {a.workers} workers on cores '
          f'{cores}' + ('; DEVELOPMENT (records are not measurements)' if a.development else ''))
    print_counts(a, cells, to_run)
    print(f'slot-hours to run (assumed {RULE_CELL_S:g} s per cell): {len(to_run)} x '
          f'{RULE_CELL_S:g} s = {len(to_run) * RULE_CELL_S / 3600:.2f}')
    for size in a.sizes:
        cs = [c for c in cells if c['size'] == size]
        print(f'  stored rule start schedules to compare at {size}: '
              f'{len(cs) - sum(c["size"] == size for c in no_cache)} of {len(cs)} cells '
              f'({RULE_CACHE.format(size=size)})')
    print(f'records: {rel(rule_folder(a.root, "<size>"))}/<combination>/instance_<i>.json')
    for c in [c for c in cells if c['state'] in ('invalid', 'held', 'foreign')][:10]:
        print(f'  {c["state"]} {c["size"]} {c["combination"]} {instance_name(c["instance"])}: '
              f'{c["reason"]}')
    if to_run:
        print('first cell command: ' + ' '.join(rule_cell_command(a, to_run[0])))
    stop = start_or_stop(a, refusals, to_run)
    if stop is not None:
        return stop
    a._commit_now = TCP.git_commit()
    append_note(a, rule_launch_note(a, cells, to_run, cores, a._commit_now))
    return campaign_loop(
        a, cores, to_run, lambda c, t0, idle, commit: rule_cell_command(a, c, t0, idle, commit),
        classify_rules,
        lambda rec: (', '.join(f'{r["rule"]} {r["makespan"]:g}' for r in rec['rules'])
                     + f'; best {rec["best_rule"]}, t_all {rec["t_all"]}, shares '
                     f'{[w["cpu_share"] for w in rec["per_worker"]]}'))


# ---------------------------------------------------------------------------
# summary over batch sizes (reads records, writes nothing)
# ---------------------------------------------------------------------------

def summary_record(a, run, size, batch, tag, combo, idx):
    """The record of a cell for the summary, or None: the valid record of the plan (with
    --development: the record, or the newest one moved to invalid/, when validated and
    of the plan)."""
    p = record_path(a.root, size, a.budget, batch, tag, combo, idx)
    paths = [p] if os.path.exists(p) else []
    if not paths and a.development:
        paths = sorted(glob.glob(os.path.join(TCP.invalid_dir(p), f'*_{os.path.basename(p)}')))[-1:]
    if not paths:
        return None
    rec = json.load(open(paths[0]))
    want = dict(budget_s=float(a.budget), batch=int(batch), workers=a.workers,
                base_seed=a.base_seed, model=run['model'], environment=run['environment'])
    if any(rec.get(k) != v for k, v in want.items()):
        return None
    if a.development:
        return rec if rec.get('validated') and rec.get('greedy_makespan') else None
    return rec if classify(rec)[0] else None


def gap_pct(rec, t):
    """100 * (best at t - greedy) / greedy, or None when no schedule exists at t."""
    from baselines.memetic.records import incumbent_at
    best = incumbent_at(rec, t, 'wall', 'process')
    g = rec['greedy_makespan']
    return None if best is None else 100.0 * (best - g) / g


def summary(a):
    """Per batch size and size: valid records, the mean gap to the greedy schedule at the
    budget and at SUMMARY_TIMES, and the mean number of schedules, over the cells with a
    record for every batch size (module docstring, SUMMARY)."""
    try:
        run = resolve_run(a.model, a.seed, development=True)
    except SystemExit:
        run = nominal_run(a.model, a.seed)
    tag = run_tag(run)
    batches = sorted(set(int(b) for b in a.batches))
    inst = TCP.parse_instances(a.instances or SUMMARY_INSTANCES)
    times = sorted({t for t in SUMMARY_TIMES if t < float(a.budget)} | {float(a.budget)})
    budget = float(a.budget)
    print(f'summary: {run["model"]} ({run["environment"]} environment), budget {bnum(budget)} s, '
          f'{a.workers} workers, base seed {a.base_seed}, batch sizes '
          f'{" ".join(map(str, batches))}, instances {a.instances or SUMMARY_INSTANCES}, '
          f'combinations {" ".join(a.combos)}; '
          + ('DEVELOPMENT: validated records of the plan, valid or not' if a.development
             else 'valid records only'))
    head = '  '.join([f'{"batch":>5}', f'{"valid":>7}'] + [f'{bnum(t) + " s":>8}' for t in times]
                     + [f'{"schedules":>10}'])
    pooled = {b: [] for b in batches}
    pooled_cells = 0
    for size in a.sizes:
        recs = {}
        for combo in a.combos:
            for idx in inst:
                for b in batches:
                    rec = summary_record(a, run, size, b, tag, combo, idx)
                    if rec is not None:
                        recs[(combo, idx, b)] = rec
        n_all = len(a.combos) * len(inst)
        cells = [(c, i) for c in a.combos for i in inst
                 if all((c, i, b) in recs for b in batches)]
        print(f'{size}: {len(cells)} of {n_all} cells compared (a record for every batch size); '
              f'mean of 100 * (best - greedy) / greedy at each time')
        print(head)
        at = {}
        for t in times:
            vals = {(c, i): [gap_pct(recs[(c, i, b)], t) for b in batches] for c, i in cells}
            at[t] = {k: v for k, v in vals.items() if all(x is not None for x in v)}
        for k, b in enumerate(batches):
            n_valid = sum((c, i, b) in recs for c in a.combos for i in inst)
            cols = []
            for t in times:
                v = [x[k] for x in at[t].values()]
                cols.append(f'{np.mean(v):8.3f}' if v else f'{"-":>8}')
            ns = [recs[(c, i, b)]['n_schedules'] for c, i in cells]
            print('  '.join([f'{b:>5}', f'{n_valid:>3}/{n_all:<3}'] + cols
                            + [f'{np.mean(ns):10.1f}' if ns else f'{"-":>10}']))
            pooled[b] += [x[k] for x in at[budget].values()]
        short = [f'{bnum(t)} s: {len(at[t])}' for t in times if len(at[t]) != len(cells)]
        if short:
            print('  cells with a schedule for every batch size, where fewer than compared: '
                  + ', '.join(short))
        pooled_cells += len(at[budget])
    if not pooled_cells:
        print('no cell has a record for every batch size: no batch size named')
        return 1
    means = {b: float(np.mean(v)) for b, v in pooled.items()}
    pick = min(batches, key=lambda b: (round(means[b], 9), b))
    print(f'pooled at the budget ({pooled_cells} cells over {len(a.sizes)} size(s)): '
          + ', '.join(f'batch {b} {means[b]:.3f}' for b in batches))
    print(f'lowest pooled mean at the budget: batch {pick} (ties go to the smaller batch)')
    return 0


# ---------------------------------------------------------------------------
# reproduction of listed batches of a record
# ---------------------------------------------------------------------------

def reproduce(path, rows_spec=None):
    """Re-run batches of a record in this process (one thread, no budget) and compare
    the makespans. rows_spec: 'w:j[:row],...' (default: the greedy rollout and the
    best row)."""
    rec = json.load(open(path))
    if rec['environment'] != REG.env():
        raise Refusal(f'REFUSING: the record was made in the {rec["environment"]} environment, '
                      f'{REG.env()} is active')
    B = int(rec.get('batch', 1))

    def row_of(r):
        return r[8] if len(r) > 8 else 0
    rows = {(r[0], r[1], row_of(r)): r for r in rec['schedules']}
    if rows_spec:
        want = []
        for s in rows_spec.split(','):
            x = [int(v) for v in s.split(':')]
            want.append((x[0], x[1], x[2] if len(x) > 2 else 0))
    else:
        want = [(r[0], r[1], row_of(r)) for r in rec['schedules'] if r[2] == 'greedy']
        if rec.get('best_from'):
            bf = rec['best_from']
            want.append((bf['worker'], bf['index'], bf.get('row', 0)))
    want = list(dict.fromkeys(want))
    spec = dict(model=rec['model'], checkpoint=rec['checkpoint'], config=rec['config'],
                mask=rec['mask'], size=rec['size'], combination=rec['combination'],
                instance=int(rec['instance'].split('_')[1]))
    roller = PolicyRoller(spec)
    same, again = True, {}
    for w, j, r in want:
        x = rows[(w, j, r)]
        greedy = x[2] == 'greedy'
        if not greedy and x[3] != rollout_seed(rec['base_seed'], w, j):
            raise Refusal(f'REFUSING: seed {x[3]} of ({w}, {j}) is not the seed rule')
        if (w, j) not in again:
            ms = roller.rollout_batch(greedy, x[3], 1 if greedy else B)[0]
            again[(w, j)] = ms
            stored = [rows[(w, j, k)][6] for k in range(len(ms)) if (w, j, k) in rows]
            whole = len(stored) == len(ms) and list(map(float, ms)) == stored
            same = same and whole
            print(f'worker {w} batch {j} ({x[2]}, seed {x[3]}, {len(ms)} row(s)): every row '
                  f'{"equal" if whole else "DIFFERENT"} (recorded {stored}, again '
                  f'{[float(m) for m in ms]})', flush=True)
        m = float(again[(w, j)][r])
        same = same and m == x[6]
        print(f'  row {r}: recorded {x[6]:g}, again {m:g} {"equal" if m == x[6] else "DIFFERENT"}',
              flush=True)
    return 0 if same else 1


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--arm', choices=('policy', 'rules'), default='policy',
                   help="policy: sampled schedules within --budget; rules: the dispatching-rule "
                        'start schedule of the search arms (no budget, no checkpoint)')
    p.add_argument('--budget', type=float, default=None, help='wall seconds per cell, from '
                   'the start of the cell\'s process')
    p.add_argument('--workers', type=int, default=None,
                   help=f'default {DEFAULT_WORKERS} (policy) or {RULE_WORKERS} (rules)')
    p.add_argument('--batch', type=int, default=DEFAULT_BATCH,
                   help='schedules per sampled rollout, one seed per batch')
    p.add_argument('--base_seed', type=int, default=DEFAULT_BASE_SEED)
    p.add_argument('--seed', default='301', help="training seed of the released policy, or "
                   "'select' (arm_registry.select_seed)")
    p.add_argument('--model', default=None, help='checkpoint name instead of the released '
                   'policy of the active environment')
    p.add_argument('--sizes', nargs='+', default=list(SIZES))
    p.add_argument('--combos', nargs='+', default=list(COMBOS))
    p.add_argument('--instances', default=None, help="e.g. '3-29' (default: each size's "
                   f'evidence instances; {SUMMARY_INSTANCES} with --summary)')
    p.add_argument('--root', default=ROOT, help='folder under which or_solution/FAMILY and '
                   'train_log are written (default the repository root)')
    p.add_argument('--max_invalid', type=int, default=MAX_INVALID)
    p.add_argument('--dry', action='store_true', help='print the plan, write nothing')
    p.add_argument('--development', action='store_true',
                   help='lift the refusals of a busy core, of more workers than cores and of '
                        'a checkpoint without the mask options; records are development_run')
    p.add_argument('--summary', action='store_true', help='compare the batch sizes of '
                   '--batches on existing records; writes nothing')
    p.add_argument('--batches', type=int, nargs='+', default=None,
                   help='with --summary: the batch sizes to compare')
    p.add_argument('--reproduce', default=None, help='a record: re-run batches and compare')
    p.add_argument('--rows', default=None, help="with --reproduce: 'w:j[:row],...'")
    p.add_argument('--_cell', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_t0', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_idle', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_commit', default=None, help=argparse.SUPPRESS)
    return p


def main(argv=None):
    p = build_parser()
    a = p.parse_args(_ARGV if argv is None else argv)
    if a.reproduce:
        return reproduce(a.reproduce, a.rows)
    for c in a.combos:
        if c not in COMBOS:
            p.error(f'unknown combination {c}')
    a.root = os.path.abspath(a.root)
    if a.arm == 'rules':
        if a.budget is not None or a.summary or a.batch != DEFAULT_BATCH:
            p.error('--arm rules takes no --budget, --batch or --summary')
        a.workers = RULE_WORKERS if a.workers is None else a.workers
        if not 1 <= a.workers <= RULE_WORKERS:
            p.error(f'--arm rules: 1 to {RULE_WORKERS} workers')
        return rule_cell_main(a) if a._cell else rule_campaign(a)
    if a.budget is None or a.budget <= 0:
        p.error('--budget (seconds, positive) is required')
    a.workers = DEFAULT_WORKERS if a.workers is None else a.workers
    if a.workers < 1:
        p.error('--workers must be at least 1')
    if a.batch < 1:
        p.error('--batch must be at least 1')
    if a.summary:
        if not a.batches or min(a.batches) < 1:
            p.error('--summary needs --batches (positive batch sizes)')
        return summary(a)
    if a._cell:
        return cell_main(a)
    return campaign(a)


if __name__ == '__main__':
    sys.exit(main())
