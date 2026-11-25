r"""P131: the policy trained on reinforced-concrete instances alone, evaluation and inference.

Subcommands plan, evaluate, infer and check. CPU only, exact mask.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. The policy and the specialists train on instances that mix reinforced-concrete
   and steel modules (--ppvc_mix mixed). On the test sets of reinforced-concrete modules alone
   (level mix_rc of scripts/p129_project_mix.py at 10, 50 and 80 modules) the policy is not
   distinguishable from the best of the four classical rules at any size (pooled -0.24, +0.42
   and -0.49%), while on the mixed test sets it is 1.4 to 3.9% ahead. This study asks whether
   the lead over the best rule returns on the reinforced-concrete sets when the policy is
   trained on reinforced-concrete instances alone: the runs xmask-rc-s301 to -s303 of
   scripts/run_rcmix_queue.sh, the released recipe with --ppvc_mix rc_project. The numbers
   land in the sensitivity subsection of the supplement and in one sentence of the Results
   subsection on the limits of the policy.
2. Expected result. Comparison (i) below decides the text. If the arm is ahead of the best
   rule on mix_rc where the policy is not, the missing lead is a property of the training mix,
   and a project with one shell system retrains on its own mix; comparison (ii) on the mixed
   sets then states what that retraining costs on a mixed load (the arm never sees a steel
   module, so it is expected to be behind the policy there). If the arm is still not
   distinguishable from the best rule on mix_rc, the reinforced-concrete sets leave the policy
   no margin over the rules at this budget, and the production claim stays bounded to mixed
   loads. Either outcome is reported.
3. Contamination. Results go under --root (default test_results/FAMILY/rc_trained_p131), a
   folder no existing script reads (tests/test_p131_rc_trained_policy.py checks every pattern
   below test_results/FAMILY). The stored results of the policy, the specialists and the
   rules are read only, from the folders p129 reads (mix_rc: p129's results folder; default:
   the exact campaign's folders), and never recomputed. No training or validation instance of
   the three runs may equal a test instance (check --contamination, below; the training queue
   runs it before its probe). Rollouts are deterministic, so results are not timing
   measurements and parts may share cores; the recorded times are planning figures.
4. Data accuracy. evaluate and infer refuse a run whose configuration does not record
   ppvc_mix rc_project and the exact mask, or differs from the policy's configuration of the
   same seed in any key other than model_suffix and ppvc_mix, or whose network does not have
   32786 parameters. Cells, rollout, schedule check (check C6 and the makespan against the last
   completion), result files, estimators and verdict rules are p129's, imported. Controls of
   infer: (a) this script's rollout of the policy's seed 301 equals the exact campaign's stored
   greedy makespans bit for bit on the first three instances of N and LSW at each size of the
   level default (the arm and the stored arrays it is compared with there come from two
   evaluation paths); (b) in a full run, every per-combination mean makespan of the seven
   rules equals p129's record (results/project_mix.json).

RUNS
  arm     xmask-rc-s301, -s302, -s303 (10x25+family+xmask-rc-s<seed>), --ppvc_mix rc_project
  policy  xmask-v1 of the same seeds, specialists xmask-solo-<R> of the same seeds, the rules
          FIFO, MOR, SPT, MWKR, ECT, SSPT and LLF (stored results, read only)

LEVELS AND SIZES (10, 50 and 80 modules on the default factory of 25 stations; combinations
N, L, S, W, LS, LW, SW, LSW)
  mix_rc   reinforced-concrete modules alone: 50 instances at 10 modules (p87's level), 30 at
           50 and 80 (p129's sets); the stored policy, specialist and rule results are p129's.
  default  the mixed test sets of the paper: 100 instances at 10 modules, 30 at 50 and 80; the
           stored arrays are those of the exact campaign, read through p129's default_arrays.
  The arm is evaluated by this script on both levels (p129's evaluate_unit: rollout of
  scripts/p101_mask_controls.py PolicyRunner, one torch thread, schedule check). At the level
  default this replaces a standard campaign evaluation, which would need the arm registered in
  scripts/arm_registry.py and scripts/eval_newseeds_plan.py and would write into the
  campaign's folders.

UNITS (evaluate)
  (level, size, combination, 'rct', seed, greedy+<run>): 2 levels x 3 sizes x 8 combinations
  x 3 seeds = 144 result files, and 6 control units (level control: the policy's seed 301 on
  the first 3 instances of N and LSW per size of the level default). Results:
  <root>/results/exact/<level>/<size>/<combination>/<label>.npz and .json (p129's layout).

OUTCOMES (infer), per level and size, per combination and pooled over the eight, seed s
against seed s over seeds 301-303, d = 100 (a - b) / b, B = 20000 (p129's estimators:
scripts/p102_scale_sensitivity.py component and superiority, stats_family.hier_boot with a
fresh default_rng(11) per quantity, p89_public_family.superiority_verdict)
  (i)   rct_rule, rct_rule7  the arm against the per-instance best of the four classical rules
                             and of the seven rules: interval, direction, one-sided p-values,
                             Holm over the eight combinations, combined verdict
  (ii)  rct_pol              the arm against the policy: interval, verdicts at the margins
                             0.5% and 1%, direction
  (iii) rct_spec             the arm against the specialist of the combination, as (ii)
  (iv)  pol_rule, pol_rule7  the policy against the best rule on the same three seeds and
                             cells, as (i), so that (i) is read beside a number of the same
                             seeds
  Outputs: results/rc_trained_policy.json and .md, manuscript/macros_rctrained.tex (prefix
  Rct, letters only, one comment per macro), in place for a full run, otherwise under --out.

CONTAMINATION CHECK (check --contamination; also part of infer)
  The trainer seeds numpy with the training seed at the start of FamilyTrainer.train and draws
  two integers per environment of each batch in FamilyTrainer.sample_batch (instance seed,
  descriptor seed), a batch of num_envs every reset_env_timestep updates; nothing else draws
  from that stream during training. The check replays the stream, verifies the replay against
  the trainer's own sample_batch on the first batches (a subprocess on the CPU), and refuses if
  a training instance seed or a validation seed (seed_train_vali_datagen + i, i < 20) lies in
  the seed block of a test set with the same module count, or if a generated training or
  validation base instance equals a 10-module test instance in content. Test sets at 50 and 80
  modules hold 1100 and 1760 operations against 220 in training, so none can be equal.

USAGE (repository root; CPU only; one core per part)
  python scripts/p131_rc_trained_policy.py plan     --parts K
  CUDA_VISIBLE_DEVICES= python scripts/p131_rc_trained_policy.py evaluate --part i --parts K
  python scripts/p131_rc_trained_policy.py infer
  python scripts/p131_rc_trained_policy.py check [--contamination]          (read only)
  Selection: --levels mix_rc,default --sizes 10x25,50x25n30,80x25n30 --combos ... --seeds
  301-303; smoke: --root <folder> --n 2, infer --partial --out <folder>. evaluate waits while
  the hold file (--hold, default train_log/EVAL_HOLD) exists, before each unit.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
os.environ.setdefault('PACT_ENV', 'exact')     # the registry resolves the exact-mask folders

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

import params  # noqa: E402,F401  (parsed now, with the argv above; load_instance imports it)
import p129_project_mix as P129  # noqa: E402

P101, P102, P103, P87 = P129.P101, P129.P102, P129.P103, P129.P87

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

STEM = 'xmask-rc'
SEEDS = (301, 302, 303)
EXPECT_PARAMS = 32786
EXPECT_MIX = 'rc_project'
POLICY_MIX = 'mixed'
CONFIG_KEYS_ALLOWED = ('model_suffix', 'ppvc_mix')   # the arm against the policy, same seed
LEVELS = ('mix_rc', 'default')
SIZES = P129.SIZES
R8 = P129.R8
RULES4 = P129.RULES4
RULES7 = P129.RULES7
MODULES = P129.MODULES
N_CELL = {('mix_rc', '10x25'): P129.N_STORED_MIX, ('mix_rc', '50x25n30'): P129.N_NEW,
          ('mix_rc', '80x25n30'): P129.N_NEW}
N_CELL.update({('default', s): P101.SIZES[s]['n'] for s in SIZES})
CONTROL_COMBOS = ('N', 'LSW')
CONTROL_N = 3
CONTROL_SEED = 301
DEFAULT_ROOT = 'test_results/FAMILY/rc_trained_p131'
STORED_ROOT = P129.DEFAULT_ROOT
STORED_DATA = P129.DEFAULT_DATA
RECORD_P129 = P129.OUT_FILES['json']
OUT_FILES = {'json': 'results/rc_trained_policy.json', 'md': 'results/rc_trained_policy.md',
             'macros': 'manuscript/macros_rctrained.tex'}
GENERATOR = 'scripts/p131_rc_trained_policy.py infer'
PREFIX = 'Rct'
B = P129.B
MARGINS = P129.MARGINS
MW = P129.MW
HOLD = P129.HOLD
COST_S = P129.COST_S
# where the runs' files are (repository-relative; tests point them elsewhere)
RUN_DIRS = {'config': 'train_log/FAMILY', 'checkpoint': 'trained_network/FAMILY',
            'log': 'train_log'}
# comparison key -> (first-named method, second, macro word, kind)
ARM_TEXT = 'policy trained on reinforced-concrete instances'
COMPARISONS = {
    'rct_rule': (ARM_TEXT, 'best classical rule', 'RcRule', 'superiority'),
    'rct_rule7': (ARM_TEXT, 'best of the seven rules', 'RcRuleSeven', 'superiority'),
    'rct_pol': (ARM_TEXT, 'policy', 'RcPol', 'equivalence'),
    'rct_spec': (ARM_TEXT, 'specialist', 'RcSpec', 'equivalence'),
    'pol_rule': ('policy', 'best classical rule', 'PolRule', 'superiority'),
    'pol_rule7': ('policy', 'best of the seven rules', 'PolRuleSeven', 'superiority'),
}
SUPERIORITY = tuple(k for k, v in COMPARISONS.items() if v[3] == 'superiority')
EQUIVALENCE = tuple(k for k, v in COMPARISONS.items() if v[3] == 'equivalence')
LEVEL_WORDS = {'mix_rc': 'Concrete', 'default': 'Mixed'}
LEVEL_TEXT = {'mix_rc': 'reinforced-concrete modules alone', 'default': 'mixed test sets'}
SIZE_WORDS = P129.SIZE_WORDS
SPEC_WORDS = P129.SPEC_WORDS
RULE_WORDS = P129.RULE_WORDS
DIR_WORDS = {'ahead': 'Ahead', 'behind': 'Behind', 'neither': 'Neither'}
TRAINER_BATCHES = 2          # batches of each run on which the replay is checked against the trainer
INT_HI = 2 ** 31 - 1         # FamilyTrainer.sample_batch: np.random.randint(0, 2 ** 31 - 1)

Refused = P129.Refused
refuse = P129.refuse
write_json_atomic = P129.write_json_atomic
jsonable = P129.jsonable
f2, u1 = P129.f2, P129.u1


# ---------------------------------------------------------------------------
# names and units
# ---------------------------------------------------------------------------

def run_name(seed):
    return f'{STEM}-s{int(seed)}'


def rct_model(seed):
    return REG.PREFIX + run_name(seed)


def rct_label(seed):
    return 'greedy+' + rct_model(seed)


def policy_model(seed):
    return P103.model_name('pact', None, seed)


def control_label():
    return 'greedy+' + policy_model(CONTROL_SEED)


def cell_n(level, size, n=None):
    return min(n, N_CELL[(level, size)]) if n else N_CELL[(level, size)]


def all_units(levels, sizes, combos, seeds, control=True):
    """(level, size, combo, arm, seed, label) of every unit; with control and the level
    default, the control units of the sizes."""
    out = [(lv, sz, c, 'rct', s, rct_label(s))
           for lv in levels for sz in sizes for c in combos for s in seeds]
    if control and 'default' in levels:
        out += [('control', sz, c, 'pact', CONTROL_SEED, control_label())
                for sz in sizes for c in CONTROL_COMBOS]
    return out


def unit_n(u, n=None):
    if u[0] == 'control':
        return min(n, CONTROL_N) if n else CONTROL_N
    return cell_n(u[0], u[1], n)


def unit_cost(u, n=None):
    return COST_S[u[1]]['policy'] * unit_n(u, n)


def assign_parts(units, parts, n=None):
    """p129's partition (longest unit first onto the lightest part) with this study's costs."""
    order = sorted(range(len(units)), key=lambda i: (-unit_cost(units[i], n), i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += unit_cost(units[i], n)
    return [sorted(x) for x in out], loads


def unit_done(root, u):
    return P129.unit_done(root, u)


# ---------------------------------------------------------------------------
# the runs
# ---------------------------------------------------------------------------

def run_paths(model):
    return {'config': os.path.join(ROOT_DIR, RUN_DIRS['config'], f'config_{model}.json'),
            'checkpoint': os.path.join(ROOT_DIR, RUN_DIRS['checkpoint'], f'{model}.pth')}


def train_log_path(seed):
    return os.path.join(ROOT_DIR, RUN_DIRS['log'], f'FAMILY_{run_name(seed)}.log')


def config_differences(snap, ref):
    """Keys whose values differ between two training configurations."""
    return sorted(k for k in set(snap) | set(ref) if snap.get(k, '<absent>') != ref.get(k, '<absent>'))


def check_config(seed, snap, ref):
    """Refuses a run configuration of the arm that does not record ppvc_mix rc_project and
    the exact mask, or that differs from the policy's configuration of the same seed in any
    key other than CONFIG_KEYS_ALLOWED. Returns the differing keys."""
    name = run_name(seed)
    if snap.get('ppvc_mix') != EXPECT_MIX:
        refuse(f'{name} records ppvc_mix={snap.get("ppvc_mix")!r}, the study needs {EXPECT_MIX!r}')
    if ref.get('ppvc_mix') != POLICY_MIX:
        refuse(f'the policy of seed {seed} records ppvc_mix={ref.get("ppvc_mix")!r}, '
               f'want {POLICY_MIX!r}')
    for k, v in P101.MASK.items():
        if snap.get(k) != v:
            refuse(f'{name} records {k}={snap.get(k)!r}, the study needs {v!r}')
    if int(snap.get('seed_train', -1)) != int(seed):
        refuse(f'{name} records seed_train={snap.get("seed_train")!r}')
    diffs = config_differences(snap, ref)
    bad = [k for k in diffs if k not in CONFIG_KEYS_ALLOWED]
    if bad:
        refuse(f'{name}: configuration differs from {run_name(seed).replace(STEM, "xmask-v1")} '
               f'in ' + ', '.join(f'{k} ({ref.get(k)!r} there, {snap.get(k)!r} here)'
                                  for k in bad[:4]))
    return diffs


def check_params(model, n_params):
    if int(n_params) != EXPECT_PARAMS:
        refuse(f'{model} has {n_params} parameters, the study needs {EXPECT_PARAMS}')


def run_record(seed):
    """The configuration checks of one run of the arm; refuses a missing or wrong run."""
    model = rct_model(seed)
    p = run_paths(model)
    ref_p = run_paths(policy_model(seed))['config']
    for q in (p['config'], p['checkpoint'], ref_p):
        if not os.path.exists(q):
            refuse(f'{P129.rel(q)} does not exist (the run is not trained yet)')
    with open(p['config']) as f:
        snap = json.load(f)
    with open(ref_p) as f:
        ref = json.load(f)
    diffs = check_config(seed, snap, ref)
    return dict(model=model, ppvc_mix=snap['ppvc_mix'], differs_from_policy=diffs,
                snapshot_sha256=P101.sha256_file(p['config']),
                checkpoint_sha256=P101.sha256_file(p['checkpoint']))


def verify_runs(seeds, runner):
    """Before any rollout: the configuration checks, the completion line of the training log
    and the parameter count of the loaded network, for every run."""
    out = {}
    for s in seeds:
        rec = run_record(s)
        lp = train_log_path(s)
        if not (os.path.exists(lp) and 'done in' in open(lp, errors='replace').read()):
            refuse(f'{P129.rel(lp)} has no completion line (the run is not finished)')
        _, loaded = runner.load(rct_model(s))
        check_params(rec['model'], loaded['params'])
        out[s] = dict(rec, params=loaded['params'])
    return out


# ---------------------------------------------------------------------------
# evaluate, plan
# ---------------------------------------------------------------------------

def load_unit_cell(u, data_root, n=None):
    level = 'default' if u[0] == 'control' else u[0]
    return P129.load_cell(level, u[1], u[2], data_root=data_root, n=unit_n(u, n))


def run_units(args, units, runner):
    """Every unit through p129's evaluate_unit (rollout, schedule check, npz and record);
    a schedule that fails the check stops the run before anything of its unit is written."""
    cache = {}
    t_all = time.perf_counter()
    print(f'[p131 eval] {len(units)} units ({sum(unit_done(args.root, u) for u in units)} '
          f'finished); results under {args.root}', flush=True)
    for k, u in enumerate(units):
        if os.path.exists(args.hold):
            print(f'[p131 eval] hold file {args.hold} present; waiting', flush=True)
            while os.path.exists(args.hold):
                time.sleep(60)
        key = u[:3]
        if key not in cache:
            cache.clear()
            cache[key] = load_unit_cell(u, args.data_root, args.n)
        st, c = P129.evaluate_unit(args.root, u, cache[key], runner, args.force)
        print(f'[p131 eval {k + 1}/{len(units)}] {u[0]:8s} {u[1]:9s} {u[2]:4s} {u[5]:40s} '
              f'{st} cpu={c:7.1f}s elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)


def evaluate(args):
    units = all_units(args.levels, args.sizes, args.combos, args.seeds)
    if args.part is not None:
        parts, _ = assign_parts(units, args.parts, args.n)
        units = [units[i] for i in parts[args.part]]
    runner = P101.PolicyRunner()
    seeds = sorted({u[4] for u in units if u[3] == 'rct'})
    runs = verify_runs(seeds, runner)
    for s, r in runs.items():
        print(f'[p131 eval] {r["model"]}: {r["params"]} parameters, ppvc_mix {r["ppvc_mix"]}, '
              f'configuration equal to the policy of seed {s} except {r["differs_from_policy"]}',
              flush=True)
    run_units(args, units, runner)


def plan(args):
    py = sys.executable
    units = all_units(args.levels, args.sizes, args.combos, args.seeds)
    parts, loads = assign_parts(units, args.parts, args.n)
    cost = sum(unit_cost(u, args.n) for u in units)
    done = [unit_done(args.root, u) for u in units]
    left = sum(unit_cost(u, args.n) for u, d in zip(units, done) if not d)
    print(f'[p131 plan] {len(units)} result files ({sum(done)} finished), estimated '
          f'{cost / 3600:.2f} core-hours of CPU time in all and {left / 3600:.2f} left '
          f'(p129 COST_S, policy rollout with the schedule check); {args.parts} parts of '
          f'{min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h')
    groups = {}
    for u in units:
        groups.setdefault((u[0], u[1]), []).append(u)
    for (lv, sz), us in groups.items():
        what = ('the policy of seed 301, the first instances of ' + ' and '.join(CONTROL_COMBOS)
                if lv == 'control' else
                f'{len({u[2] for u in us})} combinations x {len({u[4] for u in us})} seeds')
        print(f'[p131 plan] {lv:8s} {sz:9s} {len(us):3d} units ({what}), {unit_n(us[0], args.n)} '
              f'instances each at {COST_S[sz]["policy"]} s, {unit_cost(us[0], args.n):.0f} s per '
              f'unit, {sum(unit_cost(u, args.n) for u in us) / 3600:.2f} core-hours')
    extra = ''
    if list(args.seeds) != list(SEEDS):
        extra += f' --seeds {",".join(map(str, args.seeds))}'
    for opt, val, full in (('--levels', args.levels, LEVELS), ('--sizes', args.sizes, SIZES),
                           ('--combos', args.combos, R8)):
        if list(val) != list(full):
            extra += f' {opt} {",".join(val)}'
    if args.root != DEFAULT_ROOT:
        extra += f' --root {args.root}'
    if args.n:
        extra += f' --n {args.n}'
    for i in range(args.parts):
        mine = [units[j] for j in parts[i]]
        rest = sum(unit_cost(u, args.n) for u in mine if not unit_done(args.root, u))
        print(f'# part {i}: {len(mine)} units, {loads[i] / 3600:.2f} core-hours '
              f'({rest / 3600:.2f} left)')
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c <CORE_{i}> '
              f'{py} -B scripts/p131_rc_trained_policy.py evaluate --part {i} --parts {args.parts}'
              f'{extra}')
    print(f'[p131 plan] total {cost / 3600:.2f} core-hours')
    return dict(core_h=cost / 3600, parts=[x / 3600 for x in loads], units=len(units))


# ---------------------------------------------------------------------------
# contamination: training and validation instances against the test sets
# ---------------------------------------------------------------------------

def stream_settings(seed):
    """The settings that fix the instances a run trains and validates on: from the run's
    configuration once it exists, otherwise from the policy's configuration of the same seed
    (the queue's probe refuses an arm whose configuration differs from it elsewhere)."""
    own = run_paths(rct_model(seed))['config']
    src = own if os.path.exists(own) else run_paths(policy_model(seed))['config']
    with open(src) as f:
        c = json.load(f)
    return dict(source=P129.rel(src), seed=int(seed), updates=int(c['max_updates']),
                reset=int(c['reset_env_timestep']), envs=int(c['num_envs']),
                vseed=int(c['seed_train_vali_datagen']), vali_n=min(int(c['vali_size']), 20),
                modules=int(c['n_j']), factory=c['ppvc_factory'], regimes=c['family_regimes'],
                ppvc_size_mix=c.get('ppvc_size_mix', ''))


def stream_seeds(seed, updates, reset, envs):
    """(instance seeds, descriptor seeds) the trainer draws: np.random.seed(seed) at the start
    of FamilyTrainer.train, then a sample_batch at every update i with i % reset == 0, two
    draws np.random.randint(0, 2 ** 31 - 1) per environment."""
    rs = np.random.RandomState(int(seed))
    inst, desc = [], []
    for _ in range(0, updates, reset):
        for _ in range(envs):
            inst.append(int(rs.randint(0, INT_HI)))
            desc.append(int(rs.randint(0, INT_HI)))
    return inst, desc


TRAINER_PROBE = r'''
import json, os, random, sys
from types import SimpleNamespace
spec = json.loads(sys.argv[1])
sys.argv = ['train_family.py', '--device', 'cpu', '--device_id', '', '--data_source', 'FAMILY',
            '--n_j', str(spec['modules']), '--ppvc_mix', spec['mix'],
            '--family_regimes', spec['regimes'], '--num_envs', str(spec['envs']),
            '--ppvc_factory', spec['factory']]
import numpy as np
import train_family as T
calls = []
gen, mkd = T.ppvc_instance_generator, T.make_descriptor
def g(**kw):
    calls.append(['inst', int(kw['n_modules']), kw['class_mix'], int(kw['seed'])])
    return gen(**kw)
def d(*a, **kw):
    calls.append(['desc', int(kw['seed'])])
    return mkd(*a, **kw)
T.ppvc_instance_generator, T.make_descriptor = g, d
factory = {'default': T.DEFAULT_FACTORY, 'small': T.SMALL_FACTORY,
           'tight': T.TIGHT_FACTORY}[spec['factory']]
stub = SimpleNamespace(num_envs=T.configs.num_envs, n_j=T.configs.n_j, config=T.configs,
                       regimes=[r.strip().upper() for r in spec['regimes'].split(',') if r.strip()],
                       family_base='ppvc', ppvc_factory=factory,
                       n_m=len(T.build_factory(factory)[0]), desc_shift=None)
out = {}
for s in spec['seeds']:
    calls.clear()
    np.random.seed(s)
    random.seed(s)
    for _ in range(spec['batches']):
        T.FamilyTrainer.sample_batch(stub)
    out[str(s)] = list(calls)
print('P131TRAINER ' + json.dumps(out))
'''


def trainer_draws(seeds, batches, settings):
    """The (n_modules, class_mix, seed) of every instance and the seed of every descriptor
    that the trainer's own FamilyTrainer.sample_batch draws on its first batches after
    np.random.seed(seed), in a subprocess on the CPU (no file written)."""
    s0 = settings[seeds[0]]
    spec = dict(seeds=[int(s) for s in seeds], batches=int(batches), modules=s0['modules'],
                mix=EXPECT_MIX, regimes=s0['regimes'], envs=s0['envs'], factory=s0['factory'])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTORCH_NVML_BASED_CUDA_CHECK='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    r = subprocess.run([sys.executable, '-B', '-c', TRAINER_PROBE, json.dumps(spec)], cwd=ROOT_DIR,
                       env=env, capture_output=True, text=True, timeout=600)
    line = [x for x in r.stdout.splitlines() if x.startswith('P131TRAINER ')]
    if r.returncode != 0 or not line:
        refuse(f'the trainer probe failed (exit {r.returncode}): {r.stderr[-400:]}')
    return {int(k): v for k, v in json.loads(line[-1][len('P131TRAINER '):]).items()}


def heldout_blocks(data_root):
    """{name: dict(seed0, n, modules, mix, base)} of the six test sets of the study, from
    their dataset_meta.json (mix_rc at 50 and 80 modules: also p129's SEED0)."""
    out = {}
    for lv in LEVELS:
        for sz in SIZES:
            if lv == 'default':
                base = P101.SIZES[sz]['base']
            elif sz == '10x25':
                base = P87.LEVELS[lv]['base']
            else:
                base = P129.base_dir(data_root, lv, sz)
            mp = os.path.join(ROOT_DIR, base, 'dataset_meta.json')
            if not os.path.exists(mp):
                refuse(f'{P129.rel(mp)} is missing; the seed block of {lv} {sz} is unknown')
            with open(mp) as f:
                m = json.load(f)
            rec = dict(seed0=int(m['seed0']), n=int(m['n_instances']), modules=int(m['n_modules']),
                       mix=m['ppvc_mix'], base=P129.rel(os.path.join(ROOT_DIR, base)))
            if rec['modules'] != MODULES[sz] or rec['n'] != N_CELL[(lv, sz)]:
                refuse(f'{P129.rel(mp)}: {rec["n"]} instances of {rec["modules"]} modules, the '
                       f'cell {lv} {sz} has {N_CELL[(lv, sz)]} of {MODULES[sz]}')
            if lv == 'mix_rc' and sz != '10x25' and rec['seed0'] != P129.SEED0[(lv, sz)]:
                refuse(f'{P129.rel(mp)}: seed0 {rec["seed0"]}, p129 SEED0 says '
                       f'{P129.SEED0[(lv, sz)]}')
            out[f'{lv} {sz}'] = rec
    return out


def instance_key(jl, pt):
    h = hashlib.sha256()
    for a in (np.asarray(jl), np.asarray(pt)):
        h.update(str(a.shape).encode())
        h.update(np.ascontiguousarray(a, dtype='<i8').tobytes())
    return h.hexdigest()


def stored_instance_keys(blocks, modules):
    """{content key: 'set/instance'} of every stored test instance with this module count."""
    from ppvc_instance_generator import load_instance
    out = {}
    for name, b in blocks.items():
        if b['modules'] != modules:
            continue
        files = sorted(glob.glob(os.path.join(ROOT_DIR, b['base'], 'instance_*.fjs')))
        if len(files) != b['n']:
            refuse(f'{b["base"]} holds {len(files)} instances, its dataset_meta.json says {b["n"]}')
        for f in files:
            jl, pt, _ = load_instance(f[:-len('.fjs')])
            out[instance_key(jl, pt)] = f'{name}/{os.path.basename(f)[:-4]}'
    return out


def contamination(seeds=SEEDS, data_root=STORED_DATA, batches=TRAINER_BATCHES, content=True,
                  verbose=False):
    """Every training and validation instance of the runs against the six test sets; refuses on
    a hit. Returns the record."""
    from ppvc_instance_generator import (DEFAULT_FACTORY, SMALL_FACTORY, TIGHT_FACTORY,
                                         ppvc_instance_generator)
    say = print if verbose else (lambda *a, **k: None)
    settings = {s: stream_settings(s) for s in seeds}
    for s, c in settings.items():
        if c['ppvc_size_mix']:
            refuse(f'{run_name(s)}: ppvc_size_mix {c["ppvc_size_mix"]!r} trains at several sizes')
    blocks = heldout_blocks(data_root)
    streams = {s: stream_seeds(s, c['updates'], c['reset'], c['envs']) for s, c in settings.items()}
    vali = {s: [c['vseed'] + i for i in range(c['vali_n'])] for s, c in settings.items()}
    rec = {'settings': settings, 'blocks': blocks, 'trainer_check': None, 'seed_hits': [],
           'content_hits': [], 'compared': {}}
    c0 = settings[seeds[0]]
    say(f'p131: contamination: seeds {",".join(map(str, seeds))}, {c0["updates"]} updates, a batch '
        f'of {c0["envs"]} every {c0["reset"]} updates: {len(streams[seeds[0]][0])} training '
        f'instances per run ({sum(len(v[0]) for v in streams.values())} in all), validation seeds '
        f'{min(vali[seeds[0]])}-{max(vali[seeds[0]])} ({c0["vali_n"]} instances), preset '
        f'{EXPECT_MIX}, {c0["modules"]} modules; settings from {c0["source"]}')
    if batches:
        got = trainer_draws(list(seeds), batches, settings)
        n_chk = 0
        for s in seeds:
            inst, desc = streams[s]
            k = batches * settings[s]['envs']
            want = []
            for i in range(k):
                want += [['inst', settings[s]['modules'], EXPECT_MIX, inst[i]], ['desc', desc[i]]]
            if got[s] != want:
                refuse(f'the replayed seed stream of seed {s} differs from the trainer\'s '
                       f'sample_batch on its first {batches} batches')
            n_chk += k
        rec['trainer_check'] = dict(batches=batches, instances=n_chk, equal=True)
        say(f'p131: contamination: the trainer\'s own sample_batch draws the replayed seeds '
            f'(instance, descriptor, preset {EXPECT_MIX}, {c0["modules"]} modules) on its first '
            f'{batches} batches of each run ({n_chk} instances)')
    for name, b in blocks.items():
        lo, hi = b['seed0'], b['seed0'] + b['n']
        n_tr = {s: sum(lo <= x < hi for x in streams[s][0]) for s in seeds}
        n_va = {s: sum(lo <= x < hi for x in vali[s]) for s in seeds}
        same = b['modules'] == c0['modules']
        say(f'p131: contamination: {name:16s} seeds [{lo}, {hi}) {b["modules"]:2d} modules '
            f'{b["mix"]:10s}: {sum(n_tr.values())} training and {sum(n_va.values())} validation '
            f'seeds inside' + ('' if same else ' (other module count: no equal instance possible)'))
        if same and (sum(n_tr.values()) or sum(n_va.values())):
            rec['seed_hits'].append(dict(block=name, training=n_tr, validation=n_va))
    if content:
        factory = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY,
                   'tight': TIGHT_FACTORY}[c0['factory']]
        keys = stored_instance_keys(blocks, c0['modules'])
        n_gen = 0
        for s in seeds:
            for kind, seq in (('training', streams[s][0]), ('validation', vali[s])):
                for x in seq:
                    jl, pt, _ = ppvc_instance_generator(n_modules=settings[s]['modules'],
                                                        class_mix=EXPECT_MIX,
                                                        station_counts=factory, seed=x)
                    n_gen += 1
                    hit = keys.get(instance_key(jl, pt))
                    if hit:
                        rec['content_hits'].append(dict(run=run_name(s), kind=kind, seed=x,
                                                        equals=hit))
        rec['compared'] = dict(generated=n_gen, stored=len(keys))
        say(f'p131: contamination: content: {len(rec["content_hits"])} of {n_gen} generated '
            f'training and validation base instances equal one of the {len(keys)} stored '
            f'{c0["modules"]}-module test instances')
    if rec['seed_hits'] or rec['content_hits']:
        refuse('a training or validation instance may equal a test instance: '
               + json.dumps(rec['seed_hits'][:2] + rec['content_hits'][:2]))
    say('p131: contamination: none')
    return rec


# ---------------------------------------------------------------------------
# reading results
# ---------------------------------------------------------------------------

def read_array(root, level, size, combo, label, names, tally):
    """p129's read_unit (record, instance list, zero violations) and the length of the array."""
    a = P129.read_unit(root, level, size, combo, label, names, tally)
    if len(a) != len(names):
        refuse(f'{P129.result_path(root, level, size, combo, label)} holds {len(a)} makespans, '
               f'the cell has {len(names)} instances')
    if not (np.all(np.isfinite(a)) and np.all(a > 0)):
        refuse(f'{P129.result_path(root, level, size, combo, label)} holds a makespan that is '
               f'not finite and positive')
    return a


def read_rct(root, level, size, combo, seed, names, tally, runs):
    """The arm's makespans of one cell; refuses a unit written by another checkpoint or
    configuration, or by a network without the expected parameter count."""
    label = rct_label(seed)
    a = read_array(root, level, size, combo, label, names, tally)
    with open(P129.result_path(root, level, size, combo, label)[:-4] + '.json') as f:
        pol = json.load(f).get('policy') or {}
    check_params(label, pol.get('params', -1))
    run = runs[seed]
    for k in ('snapshot_sha256', 'checkpoint_sha256'):
        if pol.get(k) != run[k]:
            refuse(f'{level}/{size}/{combo}/{label} was written by another {k.split("_")[0]} of '
                   f'the run (re-evaluate with --force)')
    return a


def stored_cell(level, size, combo, seeds, names, stored_root, tally):
    """(rules {R: [n]}, {'pact': [k, n], 'spec': [k, n]}) of the stored results, read where
    p129 reads them."""
    if level == 'default':
        return P129.default_arrays(size, combo, seeds, len(names))
    rules = {r: read_array(stored_root, level, size, combo, r, names, tally) for r in RULES7}
    pol = {a: np.stack([read_array(stored_root, level, size, combo,
                                   'greedy+' + P103.model_name(a, combo, s), names, tally)
                        for s in seeds]) for a in P129.POLICY_ARMS}
    return rules, pol


def control_check(root, sizes, n=None, require=True):
    """The evaluation path of this script against the exact campaign: the policy's seed 301 on
    the first instances of CONTROL_COMBOS at the level default, bit for bit."""
    out = {}
    for size in sizes:
        k = unit_n(('control', size), n)
        names = P101.names_of(size, k)
        for combo in CONTROL_COMBOS:
            label = control_label()
            p = P129.result_path(root, 'control', size, combo, label)
            if not (os.path.exists(p) and os.path.exists(p[:-4] + '.json')):
                if require:
                    refuse(f'the control unit {size}/{combo}/{label} is missing ({p})')
                out[f'{size} {combo}'] = 'not evaluated'
                continue
            mine = read_array(root, 'control', size, combo, label, names, {'validated': 0})
            sp = REG.result_path('released', None, CONTROL_SEED, size, combo)
            if not os.path.exists(sp):
                refuse(f'the campaign array {sp} is missing')
            stored = REG.read_path(sp)[:k]
            eq = int((mine == stored).sum())
            if eq != k:
                refuse(f'the control {size}/{combo}: {k - eq} of {k} makespans of this script\'s '
                       f'rollout of {policy_model(CONTROL_SEED)} differ from {sp}')
            out[f'{size} {combo}'] = [eq, k]
    return out


def record_control(res, path=None):
    """In a full run: every per-combination mean makespan of the seven rules, and of the best
    of four and of seven, equals p129's record."""
    path = path or os.path.join(ROOT_DIR, RECORD_P129)
    rec = P129.read_json(path, P129.GENERATOR, P129.SEEDS, None, {'partial': False})
    bad, n = [], 0
    for size, S in res['sizes'].items():
        for lv, Lv in S['levels'].items():
            X = rec['sizes'][size]['levels'][lv]
            if X['n'] != Lv['n']:
                bad.append(f'{size} {lv}: {Lv["n"]} instances here, {X["n"]} in the record')
            for r, C in Lv['combos'].items():
                theirs = dict(X['combos'][r]['rule_ms'], best=X['combos'][r]['best_rule_ms'],
                              best7=X['combos'][r]['best_rule7_ms'])
                mine = dict(C['rule_ms'], best=C['best_rule_ms'], best7=C['best_rule7_ms'])
                for k, v in mine.items():
                    n += 1
                    if abs(float(v) - float(theirs[k])) > P129.REPRO_TOL:
                        bad.append(f'{size} {lv} {r} {k}: {v} here, {theirs[k]} in the record')
    if bad:
        refuse(f'{len(bad)} rule means differ from {RECORD_P129} (first: {bad[0]})')
    return dict(record=RECORD_P129, values=n)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def comparison_block(st, G, combos, nboot):
    """Rows per combination, pooled rows and verdict counts of the six comparisons of one level
    and size (p129's estimators: p102 component and superiority)."""
    SF = st[0]
    rows = {key: {r: P102.component(SF, G[r][key], nboot)[0] for r in combos}
            for key in EQUIVALENCE}
    for key in SUPERIORITY:
        rows[key] = P102.superiority(st, {r: G[r][key] for r in combos}, nboot)
    pooled = {key: P102.component(SF, np.stack([G[r][key] for r in combos], axis=2), nboot)[0]
              for key in COMPARISONS}
    counts = {}
    for key in EQUIVALENCE:
        for mg in MARGINS:
            counts[f'{key}|{mg}'] = {v: [rows[key][r][f'v{mg}'] for r in combos].count(v)
                                     for v in SF.VERDICTS}
        counts[f'{key}|direction'] = {v: [rows[key][r]['direction'] for r in combos].count(v)
                                      for v in DIR_WORDS}
    for key in SUPERIORITY:
        counts[key] = {v: [rows[key][r]['verdict'] for r in combos].count(v) for v in st[1]}
    return dict(rows=rows, pooled=pooled, counts=counts)


def infer(args):
    """Every outcome of the module doc. Returns the result dictionary."""
    partial = bool(args.partial)
    full_set = (list(args.levels) == list(LEVELS) and list(args.sizes) == list(SIZES)
                and list(args.combos) == list(R8) and list(args.seeds) == list(SEEDS)
                and not args.n)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if not partial and not full_set:
        refuse('every level, size, combination and seed is required; a subset needs --partial')
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    if len(args.seeds) < 2 and not partial:
        refuse('at least two training seeds are required (the unit of the hierarchical bootstrap)')
    st = P102._stats()
    runs = {s: run_record(s) for s in args.seeds}
    res = {'generator': GENERATOR, 'partial': partial, 'seeds': list(args.seeds),
           'mask': P101.MASK, 'B': args.boot, 'levels': list(args.levels),
           'arm': f'{STEM}-s<seed>: the released recipe with --ppvc_mix {EXPECT_MIX}',
           'runs': {str(s): r for s, r in runs.items()},
           'convention': 'd = 100 (first - second) / second, seed s against seed s, per base '
                         'instance; negative means the first-named method is shorter',
           'best_rule': 'per-instance minimum of ' + ', '.join(RULES4),
           'best_rule7': 'per-instance minimum of ' + ', '.join(RULES7),
           'holm_family': 'the eight combinations of a level and size',
           'comparisons': {k: list(v[:2]) + [v[3]] for k, v in COMPARISONS.items()},
           'sizes': {}, 'controls': {}}
    res['controls']['contamination'] = jsonable(contamination(args.seeds, args.data_root))
    if 'default' in args.levels:
        res['controls']['evaluation_path'] = control_check(args.root, args.sizes, args.n,
                                                           require=not partial)
    for size in args.sizes:
        S = {'modules': MODULES[size], 'levels': {}, 'validated': 0}
        tally = {'validated': 0}
        blocks = {}
        for lv in args.levels:
            names = P129.level_names(lv, size, args.data_root, args.n)
            if len(names) != cell_n(lv, size, args.n):
                refuse(f'{lv} {size} lists {len(names)} instances, the study needs '
                       f'{cell_n(lv, size, args.n)}')
            Lv = {'n': len(names), 'combos': {}}
            G = {}
            for combo in args.combos:
                rules, P = stored_cell(lv, size, combo, args.seeds, names, args.stored_root, tally)
                R = np.stack([read_rct(args.root, lv, size, combo, s, names, tally, runs)
                              for s in args.seeds])
                b4 = np.min(np.stack([rules[r] for r in RULES4]), axis=0)
                b7 = np.min(np.stack([rules[r] for r in RULES7]), axis=0)
                G[combo] = {'rct_rule': 100.0 * (R - b4[None]) / b4[None],
                            'rct_rule7': 100.0 * (R - b7[None]) / b7[None],
                            'rct_pol': 100.0 * (R - P['pact']) / P['pact'],
                            'rct_spec': 100.0 * (R - P['spec']) / P['spec'],
                            'pol_rule': 100.0 * (P['pact'] - b4[None]) / b4[None],
                            'pol_rule7': 100.0 * (P['pact'] - b7[None]) / b7[None]}
                Lv['combos'][combo] = dict(
                    rule_ms={r: float(rules[r].mean()) for r in RULES7},
                    best_rule_ms=float(b4.mean()), best_rule7_ms=float(b7.mean()),
                    policy_ms={'rct': float(R.mean()), 'pact': float(P['pact'].mean()),
                               'spec': float(P['spec'].mean())},
                    source='stored arrays of the campaign' if lv == 'default'
                    else "p129's results folder")
            blocks[lv] = comparison_block(st, G, list(args.combos), args.boot)
            S['levels'][lv] = Lv
        S['comparisons'] = blocks
        S['validated'] = tally['validated']
        res['sizes'][size] = S
    res['controls']['record'] = ('not compared (partial run)' if partial else record_control(res))
    lines = macro_lines(res)
    names_m = check_macro_names(lines, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    write_json_atomic(paths['json'], jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md', 'macros'):
        print('wrote', P129.rel(paths[k]))
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def macro_lines(res):
    """(name, value, comment) per macro, after header comment lines."""
    mark = 'PARTIAL ' if res['partial'] else ''
    seeds = ','.join(map(str, res['seeds']))
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% The policy trained on reinforced-concrete instances alone ({STEM}-s<seed>, '
         f'--ppvc_mix {EXPECT_MIX}) against the best rules, the policy and the specialists; '
         f'exact mask, 10, 50 and 80 modules on 25 stations. Levels: Concrete (reinforced-'
         f'concrete modules alone), Mixed (the mixed test sets).',
         f'% Gaps in %: d = 100 (first - second) / second; negative means the first-named method '
         f'is shorter; training seed s against seed s over seeds {seeds}; B = {res["B"]}; '
         f'combined verdicts with Holm over the eight combinations of a level and size.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of levels, sizes, combinations, seeds or instances; not '
                    'for the manuscript.')
    L.append((f'{PREFIX}SeedN', len(res['seeds']),
              f'{mark}training seeds of every arm of this comparison ({seeds})'))
    for size, S in res['sizes'].items():
        w = SIZE_WORDS[size]
        for lv, Lv in S['levels'].items():
            lw = LEVEL_WORDS[lv]
            where = f'{mark}level {lv}, {size}, seeds {seeds}'
            L.append((f'{PREFIX}N{lw}{w}', Lv['n'], f'instances; {where}'))
            for r, C in Lv['combos'].items():
                for word, v, what in (('Rc', C['policy_ms']['rct'], ARM_TEXT),
                                      ('Policy', C['policy_ms']['pact'], 'policy'),
                                      ('Spec', C['policy_ms']['spec'], 'specialist'),
                                      ('Rule', C['best_rule_ms'], 'best classical rule'),
                                      ('RuleSeven', C['best_rule7_ms'], 'best of the seven rules')):
                    L.append((f'{PREFIX}Ms{word}{lw}{w}{r}', u1(v),
                              f'mean makespan (h) of the {what}, {r}; {where}'))
            B_ = S['comparisons'][lv]
            for key, (first, second, cw, kind) in COMPARISONS.items():
                for r in list(Lv['combos']) + ['Pool']:
                    c = B_['pooled'][key] if r == 'Pool' else B_['rows'][key][r]
                    tag = 'pooled over the combinations' if r == 'Pool' else r
                    what = f'{first} vs {second}, {tag}'
                    ci = f'{c["scheme"]} 95% interval over {c["k"]} runs and {c["n"]} instances'
                    L.append((f'{PREFIX}{cw}{lw}{w}{r}', f2(c['mean']),
                              f'mean gap (%), {what}; {where}'))
                    L.append((f'{PREFIX}{cw}Lo{lw}{w}{r}', f2(c['lo']),
                              f'lower end of the {ci}, {what}; {where}'))
                    L.append((f'{PREFIX}{cw}Hi{lw}{w}{r}', f2(c['hi']),
                              f'upper end of the {ci}, {what}; {where}'))
                    if kind == 'equivalence':
                        for mg in MARGINS:
                            L.append((f'{PREFIX}{cw}V{MW[mg]}{lw}{w}{r}', P129._tex(c[f'v{mg}']),
                                      f'verdict at the margin {mg}%, {what}; {where}'))
                    elif r != 'Pool':
                        L.append((f'{PREFIX}{cw}Verdict{lw}{w}{r}', c['verdict'],
                                  f'combined verdict with Holm over the eight combinations, '
                                  f'{what}; {where}'))
                    L.append((f'{PREFIX}{cw}Dir{lw}{w}{r}', P129._tex(c['direction']),
                              f'direction of the interval, {what}; {where}'))
            cnt = B_['counts']
            n_c = len(Lv['combos'])
            for key in EQUIVALENCE:
                first, second, cw, _ = COMPARISONS[key]
                for mg in MARGINS:
                    for v, vw in SPEC_WORDS.items():
                        L.append((f'{PREFIX}{cw}V{MW[mg]}{vw}{lw}{w}', cnt[f'{key}|{mg}'][v],
                                  f'combinations of {n_c} with the verdict {v} at the margin '
                                  f'{mg}%, {first} vs {second}; {where}'))
                for v, vw in DIR_WORDS.items():
                    L.append((f'{PREFIX}{cw}Dir{vw}{lw}{w}', cnt[f'{key}|direction'][v],
                              f'combinations of {n_c} with the direction {v}, {first} vs '
                              f'{second}; {where}'))
            for key in SUPERIORITY:
                first, second, cw, _ = COMPARISONS[key]
                for v, vw in RULE_WORDS.items():
                    L.append((f'{PREFIX}{cw}{vw}{lw}{w}', cnt[key][v],
                              f'combinations of {n_c} with the combined verdict {v}, {first} vs '
                              f'{second}; {where}'))
    return L


def macro_names(lines):
    return [x[0] for x in lines if isinstance(x, tuple)]


def check_macro_names(lines, out_paths):
    """Refuses names that are not the prefix plus letters, duplicates, a clash with a macro
    defined under manuscript/, and a prefix in use there."""
    names = macro_names(lines)
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    have = P129.defined_macros([out_paths['macros'], os.path.join(ROOT_DIR, OUT_FILES['macros'])])
    clash = sorted(x for x in names if x in have)
    if clash:
        refuse('macro names already defined: ' + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        refuse(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def markdown(res):
    L = [f'# Policy trained on reinforced-concrete instances alone ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of levels, sizes, combinations, seeds or instances. Not for the '
              'manuscript.', '']
    L += [f'Arm: {res["arm"]}. Exact mask; training seeds {",".join(map(str, res["seeds"]))} for '
          f'every arm; B = {res["B"]}. Levels: mix_rc (reinforced-concrete modules alone), default '
          f'(the mixed test sets). {res["convention"]}. Best rule: {res["best_rule"]}; best of '
          f'seven: {res["best_rule7"]}. Holm over {res["holm_family"]}. Verdicts: (i) and (iv) the '
          f'combined superiority verdict; (ii) and (iii) the verdicts at the margins 0.5% / 1%.', '']
    c = res['controls']
    con = c.get('contamination') or {}
    if con:
        cmp_, tchk = con.get('compared') or {}, con.get('trainer_check') or {}
        L += [f'Contamination: {cmp_.get("generated", 0)} training and validation base '
              f'instances compared with {cmp_.get("stored", 0)} stored 10-module test '
              f'instances, {len(con["seed_hits"])} seed-block hits and {len(con["content_hits"])} '
              f'equal instances; the replayed seed stream equals the trainer\'s sampler on '
              f'{tchk.get("instances", 0)} instances.', '']
    if isinstance(c.get('evaluation_path'), dict):
        L += ['Evaluation path (the policy\'s seed 301, this script against the campaign): '
              + '; '.join(f'{k} ' + (f'{v[0]} of {v[1]} equal' if isinstance(v, list) else v)
                          for k, v in c['evaluation_path'].items()) + '.', '']
    if isinstance(c.get('record'), dict):
        L += [f'Rule means reproduced: {c["record"]["values"]} values of {c["record"]["record"]}.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'

    def tail(key, x):
        if COMPARISONS[key][3] == 'equivalence':
            return f' {x["v0.5"]} / {x["v1.0"]}, {x["direction"]}'
        return f' {x["verdict"]}' if 'verdict' in x else f' direction {x["direction"]}'

    heads = {'rct_rule': '(i) arm - best rule', 'rct_rule7': '(i) arm - best of seven',
             'rct_pol': '(ii) arm - policy', 'rct_spec': '(iii) arm - specialist',
             'pol_rule': '(iv) policy - best rule', 'pol_rule7': '(iv) policy - best of seven'}
    for size, S in res['sizes'].items():
        for lv, Lv in S['levels'].items():
            B_ = S['comparisons'][lv]
            combos = list(Lv['combos'])
            L += [f'## {size}, {lv} ({S["modules"]} modules, {Lv["n"]} instances, '
                  f'{LEVEL_TEXT[lv]})', '',
                  '| combination | ' + ' | '.join(heads[k] for k in COMPARISONS) + ' |',
                  '|' + '---|' * (1 + len(COMPARISONS))]
            for r in combos + ['pooled']:
                cells = []
                for key in COMPARISONS:
                    x = B_['pooled'][key] if r == 'pooled' else B_['rows'][key][r]
                    cells.append(g(x) + tail(key, x))
                L.append(f'| {r} | ' + ' | '.join(cells) + ' |')
            L += ['', '| mean makespan (h) | ' + ' | '.join(combos) + ' |',
                  '|' + '---|' * (1 + len(combos))]
            for k, word in (('rct', 'arm'), ('pact', 'policy'), ('spec', 'specialist')):
                L.append(f'| {word} | ' + ' | '.join(u1(Lv['combos'][r]['policy_ms'][k])
                                                     for r in combos) + ' |')
            for k, word in (('best_rule_ms', 'best rule'), ('best_rule7_ms', 'best of seven')):
                L.append(f'| {word} | ' + ' | '.join(u1(Lv['combos'][r][k]) for r in combos) + ' |')
            L.append('')
    return L


# ---------------------------------------------------------------------------
# check (read only)
# ---------------------------------------------------------------------------

def log_params(seed):
    lp = train_log_path(seed)
    if not os.path.exists(lp):
        return None, False
    txt = open(lp, errors='replace').read()
    m = re.search(r'\[model\] params total=(\d+)', txt)
    return (int(m.group(1)) if m else None), 'done in' in txt


def check(args):
    if args.contamination:
        try:
            contamination(SEEDS, args.data_root, verbose=True)
        except Refused as e:
            print(f'p131: contamination: {e}')
            return 1
        return 0
    ok = True
    for s in args.seeds:
        model = rct_model(s)
        p = run_paths(model)
        n_par, done = log_params(s)
        if not (os.path.exists(p['checkpoint']) and done):
            ok = False
            print(f'p131: run {run_name(s)}: NOT TRAINED (checkpoint '
                  f'{"present" if os.path.exists(p["checkpoint"]) else "absent"}, completion line '
                  f'{"present" if done else "absent"})')
            continue
        try:
            r = run_record(s)
            check_params(model, n_par if n_par is not None else -1)
            print(f'p131: run {run_name(s)}: trained, ppvc_mix {r["ppvc_mix"]}, {n_par} parameters '
                  f'(training log), configuration equal to the policy of seed {s} except '
                  f'{r["differs_from_policy"]}')
        except Refused as e:
            ok = False
            print(f'p131: run {run_name(s)}: REFUSED: {e}')
    need, have = 0, 0
    for sz in SIZES:
        for c in R8:
            for r in RULES7:
                need += 1
                have += unit_done(args.stored_root, ('mix_rc', sz, c, 'rule', None, r))
            for a in P129.POLICY_ARMS:
                for s in args.seeds:
                    need += 1
                    have += unit_done(args.stored_root, ('mix_rc', sz, c, a, s,
                                                         'greedy+' + P103.model_name(a, c, s)))
    ok &= have == need
    print(f'p131: stored mix_rc results (p129\'s results folder, seeds '
          f'{",".join(map(str, args.seeds))}: policy, specialist, seven rules): {have} of {need} '
          f'present')
    need, have = 0, 0
    for sz in SIZES:
        for c in R8:
            for p in P129.default_paths(sz, c, args.seeds).values():
                need += 1
                have += os.path.exists(p)
    ok &= have == need
    print(f'p131: stored default arrays (the exact campaign\'s folders): {have} of {need} present')
    for sz in P129.NEW_SIZES:
        mp = P129.manifest_path(args.data_root, 'mix_rc', sz)
        ok &= os.path.exists(mp)
        print(f'p131: p129 set mix_rc {sz}: ' + ('built' if os.path.exists(mp) else 'NOT BUILT'))
    units = all_units(LEVELS, SIZES, R8, args.seeds)
    done = [u for u in units if unit_done(args.root, u)]
    ok &= len(done) == len(units)
    print(f'p131: {len(done)} of {len(units)} result files of the design under {args.root}')
    for lv in LEVELS + ('control',):
        for sz in SIZES:
            k = sum(1 for u in units if u[:2] == (lv, sz))
            h = sum(1 for u in done if u[:2] == (lv, sz))
            print(f'p131:   {lv:8s} {sz:9s} {h:3d} of {k}')
    try:
        ctl = control_check(args.root, SIZES, require=False)
        print('p131: evaluation-path control: ' + '; '.join(
            f'{k} ' + (f'{v[0]} of {v[1]} equal' if isinstance(v, list) else v)
            for k, v in ctl.items()))
    except Refused as e:
        ok = False
        print(f'p131: evaluation-path control: DIFFERENCE: {e}')
    try:
        contamination(args.seeds, args.data_root, verbose=True)
    except Refused as e:
        ok = False
        print(f'p131: contamination: {e}')
    return 0 if ok else 1


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['plan', 'evaluate', 'infer', 'check'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='results folder of this study')
    ap.add_argument('--stored_root', default=STORED_ROOT,
                    help="p129's results folder (stored mix_rc results, read only)")
    ap.add_argument('--data_root', default=STORED_DATA, help="folder of p129's new sets")
    ap.add_argument('--levels', default=','.join(LEVELS))
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--seeds', default='301-303')
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--boot', type=int, default=B)
    ap.add_argument('--hold', default=HOLD, help='evaluate waits while this file exists')
    ap.add_argument('--contamination', action='store_true',
                    help='check only: the training and validation instances against the test sets')
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()

    def canon(text, allowed, what):
        got = P101.parse_list(text, allowed, what)
        return [x for x in allowed if x in got]
    args.levels = canon(args.levels, LEVELS, 'levels')
    args.sizes = canon(args.sizes, SIZES, 'sizes')
    args.combos = canon(args.combos, R8, 'combinations')
    args.seeds = list(REG.parse_seeds(args.seeds))
    for s in args.seeds:
        if s not in SEEDS:
            sys.exit(f'REFUSING: seed {s} is not a training seed of {STEM} ({SEEDS})')
    args.n = args.n or None
    if args.parts < 1 or (args.part is not None and not 0 <= args.part < args.parts):
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    if args.contamination and args.phase != 'check':
        sys.exit('--contamination belongs to the phase check')
    os.chdir(ROOT_DIR)
    try:
        r = {'plan': plan, 'evaluate': evaluate, 'infer': infer, 'check': check}[args.phase](args)
    except Refused as e:
        sys.exit(f'*** REFUSING ***\n{e}')
    if args.phase == 'check':
        sys.exit(r)


if __name__ == '__main__':
    main()
