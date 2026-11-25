"""Three constraint-aware dispatching rules beside the four textbook ones.

Why. FIFO, MOR, SPT and MWKR rank jobs or pairs by processing time and work
counts only. None of them reads a changeover, an outage or a post-operation
lag, so a reader can ask whether PACT's margin is a margin over rules that are
blind to the constraints, or over any rule at all. This script answers with
three rules that read exactly the quantities the textbook rules ignore, run
through the same environment, seed and feasibility check as p14.

The rules (one decision = one (job, machine) pair, as in SPT):

  ECT   earliest completion time. Among the COMPATIBLE pairs (every unfinished
        job's candidate on every machine that can process it), choose the one
        with the earliest completion = processing start + processing time, where
        the processing start is the environment's own pair start: block start
        max(job ready, machine free), pushed past every outage the block
        [block, block + setup + p] would overlap, plus the attached setup from
        the machine's last routing class. This is the Giffler-Thompson
        earliest-completion pair, so its action set is the active-schedule set
        of mask_mode='active', a superset of the policy's non-delay set.
        Why not the non-delay set: next_schedule_time is the minimum pair start
        and the mask removes every pair starting later, so every legal pair
        starts at the same instant and min(start + p) over legal pairs is
        min(p) over legal pairs, i.e. SPT with the same tie set and the same
        random draws. That variant is kept as ECTND for the identity check.
        Times: the normalized pair start of env._compute_pair_free_time()
        (recomputed, not env.pair_free_time, which initial_vars leaves at zero
        before the first step of the legacy environment and so ignores the
        cold-start setup and outages at the first decision) plus
        env.candidate_pt, divided by env._slope and rounded to 1e-6 h. All
        instance times are integer hours, so ties are exact ties in hours. In
        the exact environment env.pair_free_time holds the same table at every
        decision, the first included (_first_mask_tables), and the smoke test
        checks that the two agree. ECT reads neither the mask nor the decision
        instant, and the transition does not depend on the mask, so its
        rollouts are the same in both environments; the exact smoke test checks
        this against the stored legacy ECT arrays.

  SSPT  shortest setup plus processing. Among the LEGAL pairs (dynamic_pair_mask
        false, the policy's action set), minimize the setup into the candidate
        on that machine plus its processing time. Setup from
        env._setup_into_candidates(true_side=True), processing time from
        env.true_op_pt, both in true hours. With SETUP inactive the setup term
        is exactly 0, the key is the true processing time, and the rule is SPT
        with the same tie set and the same random draws.

  LLF   longest post-operation lag first, ties by shortest processing time.
        Among the LEGAL pairs, keep those whose candidate operation carries the
        longest post-operation lag (env.true_op_lag, true hours), then the
        shortest true processing time among them. Pair-level, the definition of
        the sister project's rule (FJSP-DRL-2/common_utils.py). With LAG
        inactive every lag is 0 and the rule is SPT. Pair-level instead of
        job-level on purpose: a job-level rule takes its machine from
        available_mch_list_for_job, whose earliest-start tiebreak selects
        nothing under the non-delay mask (p25), so the machine would be drawn
        at random.

Remaining ties are drawn with np.random.choice under np.random.seed(50), set
after set_initial_data exactly as p14 and eval_family do. Every rollout ends
on env.done() and its realized schedule is re-checked by
constraint_family.reference_sim.validate_schedule; a group with any violation
is not saved and the run exits non-zero.

Environment (scripts/arm_registry.py: PACT_ENV or --pact-env, default legacy).
  legacy  FJSPEnvFamily is built without mask options, as for the stored
          arrays; the run is refused when PACT_FIRST_MASK or PACT_MASK_TOL asks
          for an exact correction, since its arrays would land in the legacy
          folders.
  exact   FJSPEnvFamily(..., first_mask='exact', mask_ties='exact'), the options
          arm_registry.MASK_OPTIONS gives eval_family.py, so SPT, ECTND, SSPT
          and LLF choose among exactly the pairs the policy sees at every
          decision, the first included (resolve_mask_option refuses a variable
          that contradicts them). The rule folder is arm_registry.rule_dir, the
          <size>_xmask folder of the masked textbook rules.

Instances: every descriptor per regime at 10x25, 20x25 and 30x25. At the
production batches the environments differ. Legacy: the first --n_prod (default
10) descriptors in sorted order at 50x25 and 80x25, the production-batch
convention of p70. Exact: the 30-instance production sets 50x25n30 and 80x25n30,
every descriptor of data/FAMILY/<size>n30 per regime (30), on the instances of
data/PPVC/<size>+ppvc-mixed-n30, whose first 20 files are those of
data/PPVC/<size>+ppvc-mixed; the policy arrays there are those of the evaluation
group n30 and the classical rules those of the group rules
(test_results/FAMILY/<size>n30_xmask). Result files are
<rule folder>/Result_<RULE>_<regime>.npy (test_results/FAMILY/<size> in the
legacy environment, test_results/FAMILY/<size>_xmask in the exact one), 1-D
makespans where Result_SPT_N.npy of that folder is 1-D (p14 sizes) and
[makespan, decision-loop seconds] where it is 2-D (eval_family sizes). The
seconds are bookkeeping, not a latency measurement. A group (size, regime,
rule) is written when all its rollouts are done, through a temporary file and a
rename, so an interrupted run leaves no partial array. The script refuses to
start if any target file exists, unless --force (overwrite) or --resume (skip
the groups already on disk and run the rest).

Cores: --cores 16-23 starts one worker per listed core, each worker pinned to
its own core with one thread (OMP, MKL and torch); without it --procs workers
(default 8) run unpinned.

Summary (after a run, or alone with --summary_only): per size and regime the
mean makespan of each rule, of SPT and of PACT (per-instance mean over the
seed files of the active training seeds, scripts/arm_registry.py, in the sorted
order of their file names; in the submitted analysis
Result_greedy+10x25+family+jointblind-v1{,-s302,-s303}_<regime>.npy; a missing
seed file is an error), and the mean per-instance gap 100*(rule - ref)/ref, positive =
the rule is WORSE, with a paired-bootstrap 95% CI (20000 resamples, one
default_rng(7) stream drawn in the fixed order size, regime, rule, SPT then
PACT, then the pooled rows per size; p11's estimator). SPT and PACT rows are
sliced to the instances the rule ran on. SPT, the rule arrays and PACT are read
from the folders of the active environment (in the exact environment PACT is
xmask-v1-s<seed>). Writes results/aware_rules.json, results/aware_rules.md
and, when 10x25 and 80x25 (exact: 80x25n30) are both complete,
paper/macros_aware.tex, each through arm_registry.out (in the exact environment
under PACT_OUT, which must be set; the run checks this before any rollout). In
the exact environment the summary is refused, with the missing files named, while
a PACT seed file is missing; the rule arrays of a finished campaign stay on disk.

Inference step (after the summary, with the same sizes; the protocol of the
manuscript, see the comment block above inference()): the policy first against
ECT, SSPT, LLF, the best of the three and the best of all seven rules, per
training run of the policy (hierarchical over the runs and the base instances,
combined verdict with Holm over the eight combinations of each size; the pooled
row gives the interval and its direction), and the best of the three further
rules against the best of the four classical rules (bootstrap over base
instances). Writes manuscript/macros_aware_rules.tex, results/
aware_rules_inference.json and .md, manuscript/tables/aware_rules_table.tex (10,
20 and 30 modules) and aware_rules_table_prod.tex (50 and 80 modules; in the
exact environment the 30-instance sets): in place in the exact environment, or under
--inference_out <folder>, which the earlier environment requires (outside the
repository's paper and results folders); without it the earlier environment skips
the step, so its summary outputs are those of the earlier analysis.

Smoke test (--smoke, in this process, nothing written), at 10x25 on the first
--smoke_n (default 2) instances of each regime of --smoke_regimes (default N,LSW
in the legacy environment, all eight in the exact one):
  both     SPT through this script equals Result_SPT_<regime>.npy of the rule
           folder; every schedule passes validate_schedule.
  legacy   ECTND, SSPT and LLF equal SPT under N.
  exact    ECTND equals SPT in every regime (every legal pair starts at the
           decision instant, the first decision included); SSPT equals SPT
           without S and LLF without L; ECT equals the stored legacy ECT
           array; and at every decision of every rollout: the mask the rules
           read equals the policy's state tensor, it equals the pairs whose
           processing start in hours is the decision instant,
           env.pair_free_time equals env._compute_pair_free_time(), and the
           dispatched pair starts processing on the true axis at the start the
           rule saw.

Usage:
  python scripts/p72_aware_rules.py [--sizes 10x25,20x25,30x25,50x25,80x25]
      [--rules ECT,SSPT,LLF] [--procs 8 | --cores 16-23] [--n_prod 10]
      [--force | --resume] [--inference_out <folder>]
      (exact environment: sizes 10x25,20x25,30x25,50x25n30,80x25n30 by default
      and allowed; --n_prod has no effect there)
  python scripts/p72_aware_rules.py --summary_only [--inference_out <folder>]
  python scripts/p72_aware_rules.py --smoke [--smoke_regimes N,S] [--smoke_n 3]
  Exact environment: PACT_ENV=exact PACT_OUT=<folder> before any of these.
"""
import os

# one thread per process, set before numpy is imported so forked workers
# inherit it; no GPU is ever needed by a dispatching rule
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import glob
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
import arm_registry as REG  # noqa: E402  (pops --pact-* options from sys.argv)
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

NEW_RULES = ('ECT', 'SSPT', 'LLF')
ALLOWED = NEW_RULES + ('ECTND',)
REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
SIZES = ('10x25', '20x25', '30x25', '50x25', '80x25')      # the legacy environment
PROD = ('50x25', '80x25')       # legacy: the first --n_prod descriptors only
N30 = ('50x25n30', '80x25n30')  # exact: the 30-instance production sets, all 30
SIZES_EXACT = ('10x25', '20x25', '30x25') + N30
SEED = 50                       # the seed every published rule rollout used
ENV = REG.env()                 # 'legacy' or 'exact' (PACT_ENV, --pact-env)


def active_sizes():
    """Sizes of the active environment (module doc, Instances)."""
    return SIZES if ENV == 'legacy' else SIZES_EXACT


def _policy_tag():
    """Method tag of the released policy without its seed suffix:
    greedy+10x25+family+jointblind-v1 (legacy), ...+xmask-v1 (exact)."""
    m = REG.model('released', None, 301)
    sfx = REG.suffix('released', 301)
    return f'greedy+{REG.PREFIX}{m[:len(m) - len(sfx)] if sfx else m}'


POLICY = _policy_tag()
N_BOOT = 20000
MACROS = 'paper/macros_aware.tex'
JSON_PATH = 'results/aware_rules.json'
MD_PATH = 'results/aware_rules.md'
SIZEW = {'10x25': 'Ten', '80x25': 'Eighty'}
SIZEW_EXACT = {'10x25': 'Ten', '80x25n30': 'Eighty'}


def size_words():
    """Sizes of the summary's macros: 10 and 80 modules of the active environment."""
    return SIZEW if ENV == 'legacy' else SIZEW_EXACT


def inst_base(size):
    """Instance folder of a size; a 30-instance production set (<size>n30) reads
    data/PPVC/<size>+ppvc-mixed-n30."""
    if size in N30:
        return f'data/PPVC/{size[:-len("n30")]}+ppvc-mixed-n30'
    return f'data/PPVC/{size}+ppvc-mixed'


def rule_dir(size):
    """Rule folder of the active environment (arm_registry.rule_dir):
    test_results/FAMILY/<size> (legacy) or test_results/FAMILY/<size>_xmask."""
    d = REG.rule_dir(size)
    if ENV != 'legacy' and not d.endswith(REG.FOLDER_SUFFIX[ENV]):
        sys.exit(f'REFUSING: rule folder {d} of the {ENV} environment does not '
                 f'end in {REG.FOLDER_SUFFIX[ENV]}')
    return d


def result_path(size, tag, regime):
    return f'{rule_dir(size)}/Result_{tag}_{regime}.npy'


def legacy_result_path(size, tag, regime):
    """The stored legacy array of a rule (the exact smoke test compares ECT)."""
    return f'{REG.fixed_dir(size)}/Result_{tag}_{regime}.npy'


def mask_options():
    """FJSPEnvFamily options of the active environment (module doc).

    legacy: {} (the constructor call of the stored arrays); refused when
    PACT_FIRST_MASK or PACT_MASK_TOL asks for an exact correction.
    exact: the options arm_registry.MASK_OPTIONS gives eval_family.py, each
    through resolve_mask_option, which refuses a contradicting variable."""
    from params import configs
    configs.device = 'cpu'      # before the environment import (class attribute)
    from constraint_family.fjsp_env_family import (mask_option_from_env,
                                                    resolve_mask_option)
    if ENV == 'legacy':
        asked = {n: mask_option_from_env(n) for n in ('first_mask', 'mask_ties')}
        if 'exact' in asked.values():
            sys.exit(f'REFUSING: the legacy environment is active but the mask '
                     f'variables ask for {asked}; the arrays would land in the '
                     f'legacy folders. Unset PACT_FIRST_MASK and PACT_MASK_TOL, or '
                     f'set PACT_ENV=exact.')
        return {}
    flags = REG.MASK_OPTIONS[ENV]
    return {f.lstrip('-'): resolve_mask_option(f.lstrip('-'), v)
            for f, v in zip(flags[0::2], flags[1::2])}


def output_paths():
    """(json, md, macros) through arm_registry.out; outside the default setting
    this refuses unless PACT_OUT is set, so it runs before any rollout."""
    return REG.out(JSON_PATH), REG.out(MD_PATH), REG.out(MACROS)


def parse_cores(text):
    """'16-23' or '16,17,20-23' -> [16, 17, 20, 21, 22, 23]."""
    out = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or len(set(out)) != len(out):
        sys.exit(f'REFUSING: core list {text!r} must be non-empty and without repeats')
    return out


# ---------------------------------------------------------------------------
# the rules
# ---------------------------------------------------------------------------

def _draw(key):
    """Uniform draw among the minima of a flattened [J*M] key (SPT's draw)."""
    key = key.reshape(-1)
    return int(np.random.choice(np.where(key == key.min())[0]))


def select_action(rule, env):
    """Action index job*M + machine for one decision of `rule` (see module doc)."""
    if rule == 'SPT':
        from common_utils import heuristic_select_action
        return int(heuristic_select_action('SPT', env))
    legal = ~env.dynamic_pair_mask[0]                              # [J, M]
    if rule in ('ECT', 'ECTND'):
        start = env._compute_pair_free_time()[0]                   # normalized
        done_h = np.round((start + env.candidate_pt[0]) / env._slope, 6)
        allowed = (~env.candidate_process_relation[0] if rule == 'ECT'
                   else legal)
        return _draw(np.where(allowed, done_h, np.inf))
    tp = env.true_op_pt[0][env.candidate[0]].astype(float)         # [J, M] h
    if rule == 'SSPT':
        s = env._setup_into_candidates(true_side=True)[0]          # [J, M] h
        return _draw(np.where(legal, s + tp, np.inf))
    if rule == 'LLF':
        lag = env.true_op_lag[0][env.candidate[0]].astype(float)   # [J] h
        lagp = np.where(legal, lag[:, None], -np.inf)
        longest = legal & (lagp == lagp.max())
        return _draw(np.where(longest, tp, np.inf))
    raise ValueError(f'unknown rule {rule}')


# ---------------------------------------------------------------------------
# one rollout
# ---------------------------------------------------------------------------

def _init_worker(core_queue=None):
    """One thread per worker; with a core queue (--cores) the worker also pins
    itself to the next free core of the list."""
    if core_queue is not None:
        core = core_queue.get(timeout=60)
        os.sched_setaffinity(0, {core})
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass                # already set in this process
    except ImportError:
        pass


def _decision_view(env):
    """What the smoke checks read before a decision (exact environment)."""
    slope = env._slope
    start = env._compute_pair_free_time()[0]                       # normalized
    compat = ~env.candidate_process_relation[0].astype(bool)
    legal = ~env.dynamic_pair_mask[0].astype(bool)
    seen = ~env.state.dynamic_pair_mask_tensor[0].cpu().numpy().astype(bool)
    hours = np.round(start / slope).astype(np.int64)               # integer hours
    instant = int(hours[compat].min())
    fails = []
    if not np.array_equal(legal, seen):
        fails.append('mask read by the rules differs from the policy state tensor')
    if not np.array_equal(legal, compat & (hours == instant)):
        fails.append('mask differs from the pairs starting at the decision instant')
    if not np.array_equal(start, env.pair_free_time[0]):
        fails.append('env.pair_free_time differs from _compute_pair_free_time()')
    if int(round(float(env.next_schedule_time[0]) / slope)) != instant:
        fails.append('decision instant differs from the minimum pair start')
    return hours, instant, fails


def _realized_start(env, op, m):
    """Processing start of a dispatched operation on the true (hour) axis."""
    return float(env.true_op_ct[0, op] - env.true_op_pt[0, op, m])


def rollout(task, check=None):
    """task = (size, regime, rule, k, desc_path, instance_stem, mask_options).
    Returns (size, regime, rule, k, makespan, seconds, violations).
    check: None, or a dict {'decisions': int, 'fail': list} that the smoke test
    of the exact environment passes to check every decision (module doc)."""
    size, regime, rule, k, dp, stem, masks = task
    from params import configs
    configs.device = 'cpu'
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from constraint_family.reference_sim import validate_schedule
    from ppvc_instance_generator import load_instance
    jl, pt, _ = load_instance(stem)
    with open(dp) as f:
        desc = ConstraintDescriptor.from_json(json.load(f), setup_convention='attached')
    M = pt.shape[1]
    env = FJSPEnvFamily(jl.shape[0], M, family_mode=True, **masks)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    for name, want in masks.items():
        if env.mask_setting(name) != want:
            raise RuntimeError(f'{stem}: environment reads {name}='
                               f'{env.mask_setting(name)}, want {want}')
    if masks.get('first_mask') == 'exact' and not env._first_mask:
        raise RuntimeError(f'{stem}: the exact first-decision mask was not applied')
    amch = np.full(pt.shape[0], -1, dtype=int)
    np.random.seed(SEED)
    t0 = time.perf_counter()
    while not env.done().all():
        if check is not None:
            hours, instant, fails = _decision_view(env)
        a = select_action(rule, env)
        j, m = a // M, a % M
        op = int(env.candidate[0, j])
        amch[op] = m
        env.step(np.array([a]))
        if check is not None:
            check['decisions'] += 1
            want = int(hours[j, m])
            if rule != 'ECT' and want != instant:
                fails.append(f'{rule} chose a pair that starts after the instant')
            if _realized_start(env, op, m) != want:
                fails.append(f'{rule}: realized start {_realized_start(env, op, m)} '
                             f'differs from the start the rule saw ({want})')
            check['fail'].extend(f'{stem} step {check["decisions"]}: {x}'
                                 for x in fails)
    dt = time.perf_counter() - t0
    viol = []
    if (amch < 0).any():
        viol.append('an operation was never assigned a machine')
    else:
        op_ct = env.true_op_ct[0]
        op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
        viol = validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    return size, regime, rule, k, float(env.current_makespan[0]), dt, viol[:3]


# ---------------------------------------------------------------------------
# task list, smoke test, campaign
# ---------------------------------------------------------------------------

def descriptors(size, regime, n_prod):
    descs = sorted(glob.glob(f'data/FAMILY/{size}/desc/{regime}/*.desc.json'))
    if size in PROD:
        descs = descs[:n_prod]
    if not descs:
        sys.exit(f'no descriptors under data/FAMILY/{size}/desc/{regime}')
    out = []
    for dp in descs:
        stem = os.path.join(inst_base(size),
                            os.path.basename(dp).replace('.desc.json', ''))
        if not os.path.exists(stem + '.fjs'):
            sys.exit(f'instance missing for {dp}: {stem}.fjs')
        out.append((dp, stem))
    return out


def stored(size, tag, regime):
    p = result_path(size, tag, regime)
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def smoke(regimes=None, n=2):
    """First n instances of each regime at 10x25, in this process, nothing
    written (module doc)."""
    _init_worker()
    masks = mask_options()
    exact = ENV != 'legacy'
    regimes = regimes or (REGIMES if exact else ('N', 'LSW'))
    rules = ('SPT', 'ECTND') + NEW_RULES
    ok = True
    t_all = time.perf_counter()
    check = {'decisions': 0, 'fail': []} if exact else None
    if exact:
        print(f'environment {ENV}: FJSPEnvFamily options {masks}; stored arrays '
              f'from {rule_dir("10x25")}', flush=True)
    for regime in regimes:
        ref = stored('10x25', 'SPT', regime)
        ms = {}
        for rule in rules:
            ms[rule] = []
            for k, (dp, stem) in enumerate(descriptors('10x25', regime, 0)[:n]):
                _, _, _, _, m, dt, viol = rollout(
                    ('10x25', regime, rule, k, dp, stem, masks), check=check)
                ms[rule].append(m)
                print(f'{regime:4s} {rule:6s} inst {k}: makespan {m:7.1f}  '
                      f'{dt:6.2f} s  feasible={not viol}', flush=True)
                if viol:
                    ok = False
                    print(f'   VIOLATION: {viol}')
        # the driver must reproduce the published SPT numbers
        if ref is None or not np.array_equal(np.array(ms['SPT']), ref[:n]):
            ok = False
            print(f'   FAIL {regime}: SPT {ms["SPT"]} vs stored '
                  f'{None if ref is None else ref[:n].tolist()}')
        else:
            print(f'   ok {regime}: SPT reproduces Result_SPT_{regime}.npy[:{n}]')
        # identities that must hold exactly
        if exact:
            must = (['ECTND'] + (['SSPT'] if 'S' not in regime else []) +
                    (['LLF'] if 'L' not in regime else []))
        else:
            must = ['ECTND', 'SSPT', 'LLF'] if regime == 'N' else []
        for rule in must:
            if ref is not None and np.array_equal(np.array(ms[rule]), ref[:n]):
                print(f'   ok {regime}: {rule} == stored SPT')
            else:
                ok = False
                print(f'   FAIL {regime}: {rule} {ms[rule]} != stored SPT')
        if exact:
            p = legacy_result_path('10x25', 'ECT', regime)
            leg = REG.read_path(p) if os.path.exists(p) else None
            if leg is not None and np.array_equal(np.array(ms['ECT']), leg[:n]):
                print(f'   ok {regime}: ECT == stored legacy ECT ({p})')
            else:
                ok = False
                print(f'   FAIL {regime}: ECT {ms["ECT"]} vs stored legacy '
                      f'{None if leg is None else leg[:n].tolist()}')
    if exact:
        print(f'decision checks: {check["decisions"]} decisions, '
              f'{len(check["fail"])} failures')
        for x in check['fail'][:10]:
            print(f'   {x}')
        if check['fail']:
            ok = False
    print(f'smoke total {time.perf_counter() - t_all:.1f} s')
    if not ok:
        sys.exit('SMOKE TEST FAILED')
    print('SMOKE TEST PASSED (no files written)')


def save_atomic(path, arr):
    """np.save through a hidden temporary name in the same folder, then a
    rename, so no reader ever sees a partial array."""
    d, b = os.path.split(path)
    tmp = os.path.join(d, f'.partial_{os.getpid()}_{b}')
    np.save(tmp, arr)                       # tmp ends in .npy: saved as named
    os.replace(tmp, path)


def campaign(sizes, rules, procs, n_prod, force, resume=False, cores=None):
    masks = mask_options()
    groups, two_col = [], {}
    for size in sizes:
        spt_n = result_path(size, 'SPT', 'N')
        if not os.path.exists(spt_n):
            sys.exit(f'{spt_n} missing: cannot fix the output convention')
        two_col[size] = np.load(spt_n).ndim == 2
        for regime in REGIMES:
            for rule in rules:
                groups.append((size, regime, rule))
    clash = [result_path(s, r, g) for (s, g, r) in groups
             if os.path.exists(result_path(s, r, g))]
    if clash and not (force or resume):
        sys.exit('REFUSING: result files already exist (pass --force to '
                 'overwrite or --resume to skip them):\n  ' + '\n  '.join(clash))
    if resume and not force:
        groups = [(s, g, r) for (s, g, r) in groups
                  if not os.path.exists(result_path(s, r, g))]
        print(f'resume: {len(clash)} groups on disk skipped, {len(groups)} to run',
              flush=True)
    tasks, expected = [], defaultdict(int)
    for size, regime, rule in groups:
        for k, (dp, stem) in enumerate(descriptors(size, regime, n_prod)):
            tasks.append((size, regime, rule, k, dp, stem, masks))
            expected[(size, regime, rule)] += 1
    if ENV != 'legacy':
        print(f'environment {ENV}: FJSPEnvFamily options {masks}; rule folders '
              + ', '.join(sorted({rule_dir(s) for s in sizes})), flush=True)
    # largest instances first so the long rollouts do not trail the pool
    tasks.sort(key=lambda t: -active_sizes().index(t[0]))
    core_queue = None
    if cores:
        import multiprocessing as mp
        core_queue = mp.Queue()
        for c in cores:
            core_queue.put(c)
        procs = len(cores)
        print(f'{len(tasks)} rollouts on {procs} workers, one per core '
              f'{",".join(map(str, cores))}, one thread each', flush=True)
    else:
        print(f'{len(tasks)} rollouts on {procs} procs', flush=True)
    if not tasks:
        print('nothing to run')
        return

    got = defaultdict(dict)
    bad = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=procs, initializer=_init_worker,
                             initargs=(core_queue,)) as ex:
        futs = [ex.submit(rollout, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs), 1):
            size, regime, rule, k, ms, dt, viol = fu.result()
            key = (size, regime, rule)
            got[key][k] = (ms, dt, viol)
            if viol:
                bad.append((key, k, viol))
            if len(got[key]) == expected[key]:
                rows = [got[key][j] for j in range(expected[key])]
                if any(r[2] for r in rows):
                    print(f'  NOT SAVED {key}: infeasible schedule', flush=True)
                else:
                    ms_arr = np.array([r[0] for r in rows])
                    arr = (np.stack([ms_arr, np.array([r[1] for r in rows])], 1)
                           if two_col[size] else ms_arr)
                    save_atomic(result_path(size, rule, regime), arr)
            if i % 200 == 0 or i == len(tasks):
                print(f'  [{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min',
                      flush=True)
    if bad:
        for key, k, viol in bad[:20]:
            print(f'VIOLATION {key} inst {k}: {viol}')
        sys.exit(f'{len(bad)} infeasible rollouts; affected groups not saved')
    print(f'campaign done in {(time.time() - t0) / 60:.1f} min')


# ---------------------------------------------------------------------------
# summary, markdown, macros
# ---------------------------------------------------------------------------

rng = np.random.default_rng(7)


def ci(d, B=N_BOOT):
    """Paired-bootstrap mean and 95% percentile interval (p11's estimator)."""
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def pact(size, regime):
    """Per-instance mean over the PACT seed files of the active seeds for (size,
    regime), in the sorted order of the file names (the stored order: the bare
    seed-301 name sorts after -s302 and -s303). A missing file is an error."""
    paths = sorted((REG.result_path('released', None, s, size, regime)
                    for s in REG.seeds()), key=os.path.basename)
    names = [os.path.basename(q) for q in paths]
    arrs = [REG.read_path(q) for q in paths]
    if len({len(a) for a in arrs}) != 1:
        sys.exit(f'PACT seed files disagree in length at {size}/{regime}: {names}')
    return np.mean(arrs, axis=0), names


def summarize(rules, inference_out=None):
    """Returns False when the exact-environment summary is refused for a missing
    PACT seed file (nothing written), None otherwise. After the summary's own
    outputs, the inference step (inference(); in the earlier environment only with
    inference_out) runs on the same sizes when the three further rules are all
    summarized."""
    REG.banner()
    JSON_OUT, MD_OUT = REG.out(JSON_PATH), REG.out(MD_PATH)
    rules = [r for r in rules if r in NEW_RULES] or list(NEW_RULES)
    sizes = [s for s in active_sizes()
             if all(os.path.exists(result_path(s, r, g))
                    for r in rules for g in REGIMES)]
    if not sizes:
        print('summary: no size has every rule and regime on disk; nothing written')
        return
    if ENV != 'legacy':
        missing = [REG.result_path('released', None, s, size, g)
                   for size in sizes for g in REGIMES for s in REG.seeds()]
        missing = [p for p in missing if not os.path.exists(p)]
        if missing:
            print(f'REFUSING the summary ({REG.describe()}): {len(missing)} PACT seed '
                  f'files missing, nothing written:\n  ' + '\n  '.join(missing[:12])
                  + ('\n  ...' if len(missing) > 12 else ''), file=sys.stderr)
            return False
    out = {'convention': 'gap = mean over instances of 100*(rule - ref)/ref, '
                         'positive = rule worse; paired bootstrap 95% CI, '
                         f'{N_BOOT} resamples, default_rng(7)',
           'policy': POLICY, 'rules': rules, 'sizes': {}}
    if ENV != 'legacy':
        flags = REG.MASK_OPTIONS[ENV]
        out['environment'] = ENV
        out['mask'] = {f.lstrip('-'): v for f, v in zip(flags[0::2], flags[1::2])}
        out['seeds'] = list(REG.seeds())
        out['rule_folders'] = {s: rule_dir(s) for s in sizes}
    identity_fail = []
    for size in sizes:
        S = {'regimes': {}, 'pooled': {}}
        pooled = {r: {'spt': [], 'pact': []} for r in rules}
        for regime in REGIMES:
            R = {}
            spt = stored(size, 'SPT', regime)
            pol, names = pact(size, regime)
            vals = {r: stored(size, r, regime) for r in rules}
            n = len(vals[rules[0]])
            if spt is None or len(spt) < n:
                sys.exit(f'SPT missing or short at {size}/{regime}')
            if pol is None or len(pol) < n:
                sys.exit(f'PACT seeds missing or short at {size}/{regime}')
            spt, pol = spt[:n], pol[:n]
            R.update(n=n, pact_files=names, mean_SPT=float(spt.mean()),
                     mean_PACT=float(pol.mean()))
            for r in rules:
                v = vals[r]
                assert len(v) == n, (size, regime, r)
                gs = 100 * (v - spt) / spt
                gp = 100 * (v - pol) / pol
                m, lo, hi = ci(gs)
                mp, lop, hip = ci(gp)
                R[f'mean_{r}'] = float(v.mean())
                R[f'gap_SPT_{r}'] = [m, lo, hi]
                R[f'gap_PACT_{r}'] = [mp, lop, hip]
                pooled[r]['spt'].append(gs)
                pooled[r]['pact'].append(gp)
                # identities the definitions force
                must = ((r == 'SSPT' and 'S' not in regime) or
                        (r == 'LLF' and 'L' not in regime))
                if must:
                    same = bool(np.array_equal(v, spt))
                    R[f'identity_{r}_eq_SPT'] = same
                    if not same:
                        identity_fail.append((size, regime, r))
            S['regimes'][regime] = R
        for r in rules:
            gs = np.concatenate(pooled[r]['spt'])
            gp = np.concatenate(pooled[r]['pact'])
            S['pooled'][r] = {'gap_SPT': list(ci(gs)), 'gap_PACT': list(ci(gp)),
                              'n': int(len(gs))}
        best = min(rules, key=lambda r: S['pooled'][r]['gap_SPT'][0])
        S['best_rule'] = best
        S['best_beats_SPT_regimes'] = [
            g for g in REGIMES if S['regimes'][g][f'gap_SPT_{best}'][0] < 0]
        S['best_beats_SPT_regimes_ci'] = [
            g for g in REGIMES if S['regimes'][g][f'gap_SPT_{best}'][2] < 0]
        out['sizes'][size] = S
    out['identity_failures'] = identity_fail

    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, 'w') as f:
        json.dump(out, f, indent=1)
    print(f'wrote {JSON_OUT}')
    write_md(out, MD_OUT)
    if identity_fail:
        print('IDENTITY FAILURES (rule should equal SPT here):', identity_fail)
    write_macros(out, refuse_reason=('identity checks failed: '
                                     f'{identity_fail}' if identity_fail else None))
    if list(rules) == list(NEW_RULES):
        inference(sizes, inference_out)
    else:
        print(f'[p72] inference step skipped: it needs all of {NEW_RULES}', file=sys.stderr)


def _g(t):
    return f'{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]'


def write_md(out, MD_OUT):
    rules = out['rules']
    L = ['# Constraint-aware dispatching rules (scripts/p72_aware_rules.py)', '',
         f'Gap convention: {out["convention"]}. PACT = per-instance mean over '
         f'the {POLICY} seed files found. Rules: ECT = earliest completion '
         'over compatible pairs (active set); SSPT = shortest setup plus '
         'processing over legal pairs; LLF = longest post-operation lag first, '
         'ties by SPT, over legal pairs.', '']
    if out.get('environment'):
        L += [f'Environment: {out["environment"]} (FJSPEnvFamily options '
              + ', '.join(f'{k}={v}' for k, v in out['mask'].items())
              + f'; training seeds {",".join(map(str, out["seeds"]))}). Rule and '
              'SPT arrays from ' + ', '.join(sorted(set(out['rule_folders'].values())))
              + '.', '']
    for size, S in out['sizes'].items():
        L += [f'## {size}', '',
              '| regime | n | SPT | PACT | ' + ' | '.join(rules) + ' |',
              '|---|---|---|---|' + '---|' * len(rules)]
        for g in REGIMES:
            R = S['regimes'][g]
            L.append(f'| {g} | {R["n"]} | {R["mean_SPT"]:.1f} | '
                     f'{R["mean_PACT"]:.1f} | ' +
                     ' | '.join(f'{R[f"mean_{r}"]:.1f}' for r in rules) + ' |')
        L += ['', '| regime | ' + ' | '.join(f'{r} vs SPT' for r in rules) +
              ' | ' + ' | '.join(f'{r} vs PACT' for r in rules) + ' |',
              '|---|' + '---|' * (2 * len(rules))]
        for g in REGIMES:
            R = S['regimes'][g]
            L.append(f'| {g} | ' +
                     ' | '.join(_g(R[f'gap_SPT_{r}']) for r in rules) + ' | ' +
                     ' | '.join(_g(R[f'gap_PACT_{r}']) for r in rules) + ' |')
        P = S['pooled']
        L.append('| pooled | ' + ' | '.join(_g(P[r]['gap_SPT']) for r in rules) +
                 ' | ' + ' | '.join(_g(P[r]['gap_PACT']) for r in rules) + ' |')
        L += ['', f'Best new rule (lowest pooled gap to SPT): {S["best_rule"]}; '
              f'mean gap to SPT below zero in {len(S["best_beats_SPT_regimes"])}/8 '
              f'regimes {S["best_beats_SPT_regimes"]}, CI below zero in '
              f'{len(S["best_beats_SPT_regimes_ci"])}/8.', '']
    if out['identity_failures']:
        L += ['IDENTITY FAILURES: ' + str(out['identity_failures']), '']
    with open(MD_OUT, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'wrote {MD_OUT}')


def write_macros(out, refuse_reason=None):
    from macro_io import emit, refuse
    MACROS_OUT = REG.out(MACROS)
    if refuse_reason:
        refuse(MACROS_OUT, refuse_reason)
    missing = [s for s in size_words() if s not in out['sizes']]
    if missing:
        print(f'macros not written: sizes {missing} incomplete')
        return
    lines = []
    for size, w in size_words().items():
        S = out['sizes'][size]
        b = S['best_rule']
        gs, gp = S['pooled'][b]['gap_SPT'], S['pooled'][b]['gap_PACT']
        lines += [
            (f'AwareBest{w}', b, f'{size}: lowest pooled gap to SPT'),
            (f'AwareGapSpt{w}', f'{gs[0]:.1f}', f'{size}: pooled gap of {b} to SPT, %'),
            (f'AwareGapSpt{w}Lo', f'{gs[1]:.1f}', '95% CI low'),
            (f'AwareGapSpt{w}Hi', f'{gs[2]:.1f}', '95% CI high'),
            (f'AwareGapPact{w}', f'{gp[0]:.1f}', f'{size}: pooled gap of {b} to PACT, %'),
            (f'AwareGapPact{w}Lo', f'{gp[1]:.1f}', '95% CI low'),
            (f'AwareGapPact{w}Hi', f'{gp[2]:.1f}', '95% CI high'),
            (f'AwareBeatsSpt{w}', str(len(S['best_beats_SPT_regimes'])),
             f'{size}: regimes (of 8) where {b} has a mean gap to SPT below 0'),
            (f'AwareBeatsSptCI{w}', str(len(S['best_beats_SPT_regimes_ci'])),
             f'{size}: regimes (of 8) where the CI of that gap lies below 0'),
        ]
    folder = f'{REG.results_root()}/<size>{REG.FOLDER_SUFFIX[ENV]}'
    emit(MACROS_OUT, lines, header='Generated by scripts/p72_aware_rules.py from '
                               f'{folder}/Result_{{ECT,SSPT,LLF}}_*.npy')
    with open(MACROS_OUT) as f:
        print(f.read())


# ---------------------------------------------------------------------------
# inference step: the statistical protocol of the manuscript (stats_family)
# ---------------------------------------------------------------------------
#
# Beside the summary above, which it leaves unchanged (the module-level stream rng
# is not drawn from here). Per size the summary covers, with the policy first,
# d = 100 (policy - X) / X per training run of the policy and per instance, X one
# of ECT, SSPT, LLF, the best of the three (per-instance minimum) and the best of
# all seven rules (with FIFO, MOR, SPT and MWKR of the same rule folder); the rules
# are deterministic, so run s of the policy is paired with the one rule schedule.
# Per combination and pooled over the eight (a drawn base instance brings its eight
# combinations): crossed hierarchical bootstrap over the policy's runs and the base
# instances (stats_family.hier_boot, a fresh default_rng(11) per quantity); per
# combination also the one-sided seed-level t and bootstrap p-values in both
# directions, Holm over the eight combinations of the size for each kind, and the
# combined verdict of p89_public_family.superiority_verdict; the pooled row gives the
# interval and its direction only.
# Then whether reading the timing data helps a rule: e = 100 (best3 - best4) /
# best4 per instance, best4 the best of the four classical rules, with the
# cluster bootstrap over base instances (stats_family.instance_boot, a fresh
# default_rng(7) per quantity), one-sided bootstrap p-values, Holm per direction
# over the eight combinations, and the verdict of a fixed comparison (ahead: the
# interval below zero and the adjusted p for "shorter" below 0.05; behind:
# mirrored; otherwise not distinguishable); the pooled row again the interval and
# its direction only.
# Outputs: manuscript/macros_aware_rules.tex (prefix Awr), results/
# aware_rules_inference.json and .md, and the tabulars manuscript/tables/
# aware_rules_table.tex (10, 20, 30 modules) and aware_rules_table_prod.tex (50, 80;
# the 30-instance sets 50x25n30 and 80x25n30 in the exact environment).
# Exact environment: in place (the repository root), or under --inference_out.
# Earlier environment: only under --inference_out <folder>, outside the repository's
# paper and results folders; without it the step is skipped.

INF_GENERATOR = 'scripts/p72_aware_rules.py'
INF_FILES = {'macros': 'manuscript/macros_aware_rules.tex',
             'json': 'results/aware_rules_inference.json',
             'md': 'results/aware_rules_inference.md',
             'table': 'manuscript/tables/aware_rules_table.tex',
             'table_prod': 'manuscript/tables/aware_rules_table_prod.tex'}
INF_GUARDED = ('results', 'test_results', 'or_solution', 'trained_network', 'train_log',
               'data')
CLASSICAL = ('FIFO', 'MOR', 'SPT', 'MWKR')
INF_SIZE_WORDS = {'10x25': 'Ten', '20x25': 'Twenty', '30x25': 'Thirty', '50x25': 'Fifty',
                  '80x25': 'Eighty', '50x25n30': 'Fifty', '80x25n30': 'Eighty'}
# quantity key -> (text, macro word)
INF_QUANTITIES = {'ECT': ('ECT', 'Ect'), 'SSPT': ('SSPT', 'Sspt'), 'LLF': ('LLF', 'Llf'),
                  'best3': ('the best of ECT, SSPT and LLF', 'BestThree'),
                  'best7': ('the best of the seven rules', 'BestSeven')}
INF_HIER_SEED, INF_INST_SEED, INF_ALPHA = 11, 7, 0.05
INF_POOL = 'pool'


def inf_tables():
    """Sizes of the two tabulars in the active environment."""
    return {'table': ('10x25', '20x25', '30x25'),
            'table_prod': PROD if ENV == 'legacy' else N30}


class InferenceRefusal(SystemExit):
    """A precondition of the inference step failed; its outputs are removed."""


def inf_refuse(msg):
    raise InferenceRefusal(f'REFUSING (inference step): {msg}')


def _inf_stats():
    import stats_family as SF
    from p89_public_family import VERDICT_WORDS, f2, jsonable, superiority_verdict
    return SF, VERDICT_WORDS, f2, jsonable, superiority_verdict


def inf_guard(out):
    """An output root other than the repository root lies outside the repository's
    paper folders and its folders of results, data, checkpoints and logs."""
    o = os.path.realpath(out)
    rel = os.path.relpath(o, os.path.realpath(ROOT))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return
    first = rel.split(os.sep)[0]
    if rel == os.curdir or first in INF_GUARDED or first.startswith('paper'):
        inf_refuse(f'--inference_out {out} would put the outputs in the repository root or '
                   f'in one of its paper, results, data, checkpoint or log folders; give a '
                   f'folder outside them')


def inf_out_root(inference_out):
    """Output root of the step, or None when the step does not run."""
    if inference_out is None:
        return ROOT if ENV != 'legacy' else None
    o = os.path.abspath(inference_out)
    if ENV == 'legacy' or os.path.realpath(o) != os.path.realpath(ROOT):
        inf_guard(o)
    return o


def inf_bases(size, n):
    """The first n descriptors of every combination are the same base instances, in
    the same order (the cluster unit of the pooled rows)."""
    ref = None
    for g in REGIMES:
        ps = sorted(glob.glob(f'data/FAMILY/{size}/desc/{g}/*.desc.json'))[:n]
        if len(ps) < n:
            inf_refuse(f'data/FAMILY/{size}/desc/{g} holds {len(ps)} descriptors, the rules '
                       f'ran on {n}')
        key = [(os.path.basename(p), json.load(open(p)).get('base_instance')) for p in ps]
        if ref is None:
            ref = key
        elif key != ref:
            inf_refuse(f'data/FAMILY/{size}: combination {g} does not share file names and '
                       f'base instances with {REGIMES[0]}')
    return [k[0][:-len('.desc.json')] for k in ref]


def inf_arrays(size):
    """Per combination: the policy [k, n] in seed order, the seven rules [n], n the
    instances the three further rules ran on (equal over rules and combinations)."""
    out, n = {}, None
    for g in REGIMES:
        new = {r: stored(size, r, g) for r in NEW_RULES}
        old = {r: stored(size, r, g) for r in CLASSICAL}
        miss = [result_path(size, r, g) for r, v in {**new, **old}.items() if v is None]
        if miss:
            inf_refuse(f'rule arrays missing: {miss[:4]}')
        ns = {len(v) for v in new.values()}
        if len(ns) != 1 or (n is not None and ns != {n}):
            inf_refuse(f'{size}/{g}: the three further rules cover {sorted(ns)} instances, '
                       f'expected one count over every combination ({n})')
        n = ns.pop()
        short = [r for r, v in old.items() if len(v) < n]
        if short:
            inf_refuse(f'{size}/{g}: rule arrays {short} hold fewer than {n} instances')
        pol = []
        for s in REG.seeds():
            p = REG.result_path('released', None, s, size, g)
            try:
                a = REG.read_path(p)
            except REG.MissingResult as e:
                inf_refuse(str(e))
            if len(a) < n:
                inf_refuse(f'{p} holds {len(a)} instances, fewer than {n}')
            pol.append(a[:n])
        rules = {**new, **{r: v[:n] for r, v in old.items()}}
        out[g] = dict(P=np.stack(pol), rules=rules)
    return out, n


def inf_component(SF, d, B, instance=False):
    """Interval of one comparison. instance=False: d [k, n] or [k, n, c], crossed
    hierarchical bootstrap (fresh default_rng(11)) and the seed-level t interval;
    with one run, or instance=True (d [n] or [n, c], fixed rules), the bootstrap over
    base instances (fresh default_rng(7)). Returns (package, draws)."""
    x = np.asarray(d, dtype=float)
    if not instance and x.shape[0] >= 2:
        hb = SF.hier_boot(x, B=B, seed=INF_HIER_SEED, alpha=INF_ALPHA, return_draws=True)
        t = SF.seed_t(x, INF_ALPHA)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                   seed_means=t['seed_means'], k=int(hb['k']), n=int(hb['n']),
                   c=int(hb['c']), scheme='hierarchical',
                   stream=f'default_rng({INF_HIER_SEED})')
        draws = hb['draws']
    else:
        y = x if instance else x[0]
        ib = SF.instance_boot(y, B=B, seed=INF_INST_SEED, alpha=INF_ALPHA, return_draws=True)
        out = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], k=0 if instance else 1,
                   n=int(ib['n']), c=int(ib['c']), scheme='instance',
                   stream=f'default_rng({INF_INST_SEED})')
        draws = ib['draws']
    out['direction'] = SF.direction(out['lo'], out['hi'])
    return out, draws


def inf_policy_rows(st, D, B):
    """Policy against one rule quantity: per combination with Holm over the
    combinations of D and the combined verdict, then the pooled row (interval and
    direction only)."""
    SF, _, _, _, sv = st
    rows = {}
    for g, d in D.items():
        c, draws = inf_component(SF, d, B)
        pl, pg = SF.seed_t_p(d, 'less'), SF.seed_t_p(d, 'greater')
        c.update(p_t=1.0 if np.isnan(pl) else pl, p_boot=SF.boot_p_one_sided(draws, 'less'),
                 p_t_greater=1.0 if np.isnan(pg) else pg,
                 p_boot_greater=SF.boot_p_one_sided(draws, 'greater'))
        rows[g] = c
    keys = list(rows)
    for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
        for g, a in zip(keys, SF.holm([rows[g][col] for g in keys])):
            rows[g][col + '_holm'] = a
    for g in keys:
        x = rows[g]
        x['verdict'] = sv(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                          x['p_t_greater_holm'], x['p_boot_greater_holm'], INF_ALPHA)
        x['holm_family'] = keys
    rows[INF_POOL] = inf_component(SF, np.stack([D[g] for g in keys], axis=2), B)[0]
    return rows


def inf_rule_rows(st, E, B):
    """Best of the three further rules against the best of the four classical rules
    (fixed rules): instance-level interval, one-sided bootstrap p-values, Holm per
    direction over the combinations of E, verdict of a fixed comparison; pooled row
    interval and direction only."""
    SF, VW = st[0], st[1]
    rows = {}
    for g, e in E.items():
        c, draws = inf_component(SF, e, B, instance=True)
        c.update(p_boot=SF.boot_p_one_sided(draws, 'less'),
                 p_boot_greater=SF.boot_p_one_sided(draws, 'greater'),
                 ties=100.0 * float(np.mean(e == 0)))
        rows[g] = c
    keys = list(rows)
    for col in ('p_boot', 'p_boot_greater'):
        for g, a in zip(keys, SF.holm([rows[g][col] for g in keys])):
            rows[g][col + '_holm'] = a
    for g in keys:
        x = rows[g]
        if x['hi'] < 0 and x['p_boot_holm'] < INF_ALPHA:
            x['verdict'] = VW[0]
        elif x['lo'] > 0 and x['p_boot_greater_holm'] < INF_ALPHA:
            x['verdict'] = VW[1]
        else:
            x['verdict'] = VW[2]
        x['holm_family'] = keys
    e2 = np.stack([E[g] for g in keys], axis=1)
    c = inf_component(SF, e2, B, instance=True)[0]
    c['ties'] = 100.0 * float(np.mean(e2 == 0))
    rows[INF_POOL] = c
    return rows


def inf_size(st, size, B):
    A, n = inf_arrays(size)
    names = inf_bases(size, n)
    D = {q: {} for q in INF_QUANTITIES}
    E, means = {}, {}
    for g in REGIMES:
        P, R = A[g]['P'], A[g]['rules']
        best3 = np.min([R[r] for r in NEW_RULES], axis=0)
        best4 = np.min([R[r] for r in CLASSICAL], axis=0)
        best7 = np.minimum(best3, best4)
        X = {'ECT': R['ECT'], 'SSPT': R['SSPT'], 'LLF': R['LLF'], 'best3': best3,
             'best7': best7}
        for q, v in X.items():
            D[q][g] = 100.0 * (P - v[None, :]) / v[None, :]
        E[g] = 100.0 * (best3 - best4) / best4
        means[g] = {**{r: float(R[r].mean()) for r in NEW_RULES + CLASSICAL},
                    'policy': float(P.mean()), 'best3': float(best3.mean()),
                    'best4': float(best4.mean()), 'best7': float(best7.mean())}
    means[INF_POOL] = {k: float(np.mean([means[g][k] for g in REGIMES]))
                       for k in means[REGIMES[0]]}
    vs = {q: inf_policy_rows(st, D[q], B) for q in INF_QUANTITIES}
    tb = inf_rule_rows(st, E, B)
    VW = st[1]
    counts = {q: {w: sum(vs[q][g]['verdict'] == w for g in REGIMES) for w in VW}
              for q in INF_QUANTITIES}
    counts['best3_vs_best4'] = {w: sum(tb[g]['verdict'] == w for g in REGIMES) for w in VW}
    return dict(n=n, k=int(A[REGIMES[0]]['P'].shape[0]), instances=names, means=means,
                policy_vs=vs, best3_vs_best4=tb, counts=counts,
                rule_folder=rule_dir(size),
                policy_files=[os.path.basename(REG.result_path('released', None, s, size, g))
                              for g in REGIMES for s in REG.seeds()])


def inf_check_names(names, paths):
    """Every name is Awr plus letters, unique, and absent from every other macro file
    of the repository (its own path and the output path are skipped)."""
    bad = [x for x in names if not re.fullmatch(r'Awr[A-Za-z]+', x)]
    if bad:
        inf_refuse(f'macro names that are not Awr plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        inf_refuse('duplicate macro names: '
                   f'{sorted({x for x in names if names.count(x) > 1})[:5]}')
    own = {os.path.abspath(paths['macros']),
           os.path.abspath(os.path.join(ROOT, INF_FILES['macros']))}
    files = sorted(glob.glob(os.path.join(ROOT, 'manuscript', 'macros_*.tex')) +
                   glob.glob(os.path.join(ROOT, 'paper', 'macros_*.tex')) +
                   [os.path.join(ROOT, d, 'macros.tex') for d in ('manuscript', 'paper')])
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            inf_refuse(f'macro name collision with {os.path.relpath(f, ROOT)}: '
                       f'{sorted(clash)[:5]}')
    return len(files)


def _inf_scheme(c):
    if c['scheme'] == 'hierarchical':
        return (f'hierarchical 95% CI over {c["k"]} runs and {c["n"]} base instances'
                + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))
    return (f'95% CI over {c["n"]} base instances'
            + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))


def inf_macro_lines(st, res):
    SF, VW, f2 = st[0], st[1], st[2]
    seeds = res['seeds']
    L = [f'% Generated by {INF_GENERATOR} (inference step) -- do not hand-edit',
         f'% Three further dispatching rules (ECT, SSPT, LLF), {res["environment"]} '
         f'environment, training seeds {",".join(map(str, seeds))}; B = {res["B"]}; '
         f'hierarchical intervals fresh default_rng({INF_HIER_SEED}), rule-against-rule '
         f'intervals fresh default_rng({INF_INST_SEED}), one per quantity.',
         '% Gaps in %: first-named method minus second over second (negative = first-named '
         'method shorter), per training run of the policy and per instance; best of the '
         'three = per-instance minimum of ECT, SSPT, LLF; best of the four = of FIFO, MOR, '
         'SPT, MWKR; best of the seven = of all seven.']
    if res['environment'] == 'legacy':
        L.insert(1, '% EARLIER ENVIRONMENT: a check of the code path; not for the manuscript.')
    L.append(f'\\newcommand{{\\AwrBoot}}{{{res["B"]}}}\\newcommand{{\\AwrSeedN}}{{{len(seeds)}}}'
             f'  % bootstrap replicates; training runs of the policy')
    for size, S in res['sizes'].items():
        w = INF_SIZE_WORDS[size]
        L.append(f'\\newcommand{{\\Awr{w}N}}{{{S["n"]}}}  % {size}: instances per combination')
        for q, (qt, qw) in INF_QUANTITIES.items():
            for g in list(REGIMES) + [INF_POOL]:
                c = S['policy_vs'][q][g]
                row = 'Pool' if g == INF_POOL else g
                name = f'Awr{w}Pol{qw}{row}'
                s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
                     f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
                     f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
                     f'\\newcommand{{\\{name}Dir}}{{{SF.TEX[c["direction"]]}}}')
                if g == INF_POOL:
                    L.append(s + f'  % {size}, policy vs {qt}, pooled over the eight '
                             f'combinations: mean, {_inf_scheme(c)}, direction')
                    continue
                L.append(
                    s + f'\\newcommand{{\\{name}Verdict}}{{{c["verdict"]}}}'
                    f'  % {size}, policy vs {qt}, {g}: mean, '
                    f'{_inf_scheme(c)}, direction, combined verdict; Holm p shorter '
                    f'{c["p_t_holm"]:.2g} (t), {c["p_boot_holm"]:.2g} (bootstrap), Holm p '
                    f'longer {c["p_t_greater_holm"]:.2g} (t), {c["p_boot_greater_holm"]:.2g} '
                    f'(bootstrap)')
            cc = S['counts'][q]
            L.append(f'\\newcommand{{\\Awr{w}Pol{qw}AheadCount}}{{{cc[VW[0]]}}}'
                     f'\\newcommand{{\\Awr{w}Pol{qw}BehindCount}}{{{cc[VW[1]]}}}'
                     f'  % {size}, policy vs {qt}: combinations (of 8) ahead, behind')
        for g in list(REGIMES) + [INF_POOL]:
            c = S['best3_vs_best4'][g]
            row = 'Pool' if g == INF_POOL else g
            name = f'Awr{w}ThreeFour{row}'
            s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
                 f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
                 f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
                 f'\\newcommand{{\\{name}Dir}}{{{SF.TEX[c["direction"]]}}}')
            what = f'{size}, best of ECT, SSPT, LLF vs best of FIFO, MOR, SPT, MWKR'
            if g == INF_POOL:
                L.append(s + f'  % {what}, pooled over the eight combinations: mean, '
                         f'{_inf_scheme(c)}, direction; ties {c["ties"]:.1f}% of instances')
                continue
            L.append(
                s + f'\\newcommand{{\\{name}Verdict}}{{{c["verdict"]}}}'
                f'  % {what}, {g}: mean, '
                f'{_inf_scheme(c)}, direction, verdict; Holm p shorter {c["p_boot_holm"]:.2g}, '
                f'longer {c["p_boot_greater_holm"]:.2g} (bootstrap); ties {c["ties"]:.1f}% of '
                f'instances')
        cc = S['counts']['best3_vs_best4']
        L.append(f'\\newcommand{{\\Awr{w}ThreeFourAheadCount}}{{{cc[VW[0]]}}}'
                 f'\\newcommand{{\\Awr{w}ThreeFourBehindCount}}{{{cc[VW[1]]}}}'
                 f'  % {size}, best of the three further rules vs best of the four classical '
                 f'rules: combinations (of 8) ahead, behind')
    return L


def _inf_cell(c, f2):
    s = f'{f2(c["mean"])} [{f2(c["lo"])}, {f2(c["hi"])}]'
    return f'{s} {c["verdict"]}' if 'verdict' in c else f'{s} direction {c["direction"]}'


def inf_markdown(st, res):
    f2 = st[2]
    L = [f'# Three further dispatching rules: the protocol of the manuscript '
         f'({INF_GENERATOR}, inference step)', '']
    if res['environment'] == 'legacy':
        L += ['EARLIER ENVIRONMENT: a check of the code path; not for the manuscript.', '']
    L += [f'Environment {res["environment"]}, training seeds '
          f'{",".join(map(str, res["seeds"]))}, B = {res["B"]}. Gaps in %: first-named '
          'method minus second over second; negative = first-named method shorter. Policy '
          'against a rule quantity: crossed hierarchical bootstrap over the policy\'s runs '
          f'and the base instances, fresh default_rng({INF_HIER_SEED}); combined verdict '
          'with Holm over the eight combinations of the size (pooled row: the interval and '
          'its direction). Best of the three further rules against the best of the four '
          f'classical rules: bootstrap over base instances, fresh '
          f'default_rng({INF_INST_SEED}), Holm per direction over the eight combinations '
          '(pooled row: the interval and its direction).', '']
    for size, S in res['sizes'].items():
        L += [f'## {size} ({S["n"]} instances per combination, {S["k"]} runs of the policy)',
              '', '| combination | ' + ' | '.join(['ECT', 'SSPT', 'LLF', 'FIFO', 'MOR', 'SPT',
                                                   'MWKR', 'policy']) + ' |',
              '|---|' + '---|' * 8]
        for g in list(REGIMES) + [INF_POOL]:
            m = S['means'][g]
            L.append(f'| {g} | ' + ' | '.join(f'{m[k]:.1f}' for k in
                                              ('ECT', 'SSPT', 'LLF', 'FIFO', 'MOR', 'SPT',
                                               'MWKR', 'policy')) + ' |')
        L += ['', '| combination | ' + ' | '.join(f'policy vs {INF_QUANTITIES[q][0]}'
                                                  for q in INF_QUANTITIES)
              + ' | best3 vs best4 |', '|---|' + '---|' * (len(INF_QUANTITIES) + 1)]
        for g in list(REGIMES) + [INF_POOL]:
            L.append(f'| {g} | ' + ' | '.join(_inf_cell(S['policy_vs'][q][g], f2)
                                              for q in INF_QUANTITIES)
                     + f' | {_inf_cell(S["best3_vs_best4"][g], f2)} |')
        L.append('')
    if res['tables_skipped']:
        L += ['Tables not written (sizes incomplete): ' + ', '.join(res['tables_skipped']), '']
    return L


def inf_table(st, res, sizes):
    """Compact tabular: per size the eight combinations and the pooled row; mean
    makespan (h) of ECT, SSPT, LLF and the policy (bold: lowest in the row at printed
    precision), then the policy's gap to each of the three rules."""
    f2 = st[2]
    VW = st[1]

    def num(x):
        return f'${f2(x)}$'
    mark = {VW[0]: '$^{*}$', VW[1]: '$^{\\dagger}$', VW[2]: ''}
    out = [f'% Generated by {INF_GENERATOR} (inference step) -- do not hand-edit',
           f'% {res["environment"]} environment, training seeds '
           f'{",".join(map(str, res["seeds"]))}. Mean makespan in hours (policy: mean over '
           'its training runs and the instances); bold: lowest mean makespan in the row. '
           'Gap in %: policy minus rule over rule, 95% hierarchical interval over the runs '
           'and the base instances; * ahead, \\dagger behind (combined verdict, Holm over '
           'the eight combinations; the pooled row carries no verdict).',
           r'\begin{tabular}{@{}l rrrr lll@{}}', r'\toprule',
           r' & \multicolumn{4}{c}{mean makespan (h)} & \multicolumn{3}{c}{policy against '
           r'the dispatching rule (\%)} \\',
           r'\cmidrule(lr){2-5}\cmidrule(l){6-8}',
           r'combination & ECT & SSPT & LLF & policy & ECT & SSPT & LLF \\']
    for size in sizes:
        S = res['sizes'][size]
        out += [r'\midrule',
                r'\multicolumn{8}{@{}l}{\textit{' + f'{int(size.split("x")[0])} modules, '
                f'$n = {S["n"]}$ instances per combination' + r'}} \\']
        for g in list(REGIMES) + [INF_POOL]:
            m = S['means'][g]
            shown = {k: f'{m[k]:.1f}' for k in ('ECT', 'SSPT', 'LLF', 'policy')}
            best = min(float(v) for v in shown.values())
            cells = [(r'\textbf{' + v + '}' if float(v) == best else v)
                     for v in shown.values()]
            for q in NEW_RULES:
                c = S['policy_vs'][q][g]
                cells.append(f'{num(c["mean"])} [{num(c["lo"])}, {num(c["hi"])}]'
                             f'{mark.get(c.get("verdict"), "")}')
            head = 'pooled' if g == INF_POOL else f'${g}$'
            out.append(f'{head} & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(out)


def inference(sizes, inference_out=None, B=N_BOOT):
    """The inference step (comment block above). Returns the result dictionary, or
    None when the step does not run in this environment."""
    out_root = inf_out_root(inference_out)
    if out_root is None:
        print('[p72] inference step skipped in the earlier environment (it writes only '
              'under --inference_out <folder>)', file=sys.stderr)
        return None
    paths = {k: os.path.join(out_root, v) for k, v in INF_FILES.items()}
    for p in paths.values():                # a refusal below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    st = _inf_stats()
    jsonable = st[3]
    res = {'generator': INF_GENERATOR, 'environment': ENV, 'seeds': list(REG.seeds()),
           'B': int(B), 'policy': _policy_tag(),
           'convention': ('d = 100 (first - second) / second; policy against a rule per '
                          'training run and instance; negative = first-named method shorter'),
           'sizes': {}}
    if ENV != 'legacy':
        flags = REG.MASK_OPTIONS[ENV]
        res['mask'] = {f.lstrip('-'): v for f, v in zip(flags[0::2], flags[1::2])}
    for size in sizes:
        res['sizes'][size] = inf_size(st, size, B)
    res['sizes_missing'] = [s for s in active_sizes() if s not in res['sizes']]
    tables = inf_tables()
    res['tables'] = {k: list(ss) for k, ss in tables.items()}
    res['tables_skipped'] = [k for k, ss in tables.items()
                             if not all(s in res['sizes'] for s in ss)]
    lines = inf_macro_lines(st, res)
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    res['macro_files_checked'] = inf_check_names(names, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
    with open(paths['json'], 'w') as f:
        json.dump(jsonable(res), f, indent=1)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(inf_markdown(st, res)) + '\n')
    for k, ss in tables.items():
        if k in res['tables_skipped']:
            continue
        os.makedirs(os.path.dirname(paths[k]), exist_ok=True)
        with open(paths[k], 'w') as f:
            f.write(inf_table(st, res, ss))
    for k, p in paths.items():           # emit() has reported the macro file
        if k != 'macros' and os.path.exists(p):
            print(f'wrote {os.path.relpath(p, ROOT) if out_root == ROOT else p}')
    print(f'inference step: {len(names)} macros, sizes {list(res["sizes"])}')
    return res


# ---------------------------------------------------------------------------

def main():
    a = list(_ARGV)
    if '--smoke' in a:
        regs = (a[a.index('--smoke_regimes') + 1].split(',')
                if '--smoke_regimes' in a else None)
        bad = [g for g in (regs or []) if g not in REGIMES]
        if bad:
            sys.exit(f'unknown regimes {bad}; allowed {REGIMES}')
        smoke(regs, int(a[a.index('--smoke_n') + 1]) if '--smoke_n' in a else 2)
        return
    rules = (a[a.index('--rules') + 1].split(',') if '--rules' in a
             else list(NEW_RULES))
    bad = [r for r in rules if r not in ALLOWED]
    if bad:
        sys.exit(f'unknown rules {bad}; allowed {ALLOWED}')
    inf_out = a[a.index('--inference_out') + 1] if '--inference_out' in a else None
    if inf_out is not None:
        inf_out_root(inf_out)   # refuse a guarded folder before any rollout
    if '--summary_only' in a:
        if summarize(rules, inf_out) is False:
            sys.exit(2)
        return
    sizes = (a[a.index('--sizes') + 1].split(',') if '--sizes' in a
             else list(active_sizes()))
    bad = [s for s in sizes if s not in active_sizes()]
    if bad:
        sys.exit(f'unknown sizes {bad}; allowed in the {ENV} environment {active_sizes()}')
    if '--cores' in a and '--procs' in a:
        sys.exit('give --cores (one pinned worker per core) or --procs, not both')
    procs = int(a[a.index('--procs') + 1]) if '--procs' in a else 8
    cores = parse_cores(a[a.index('--cores') + 1]) if '--cores' in a else None
    n_prod = int(a[a.index('--n_prod') + 1]) if '--n_prod' in a else 10
    output_paths()              # refuse a missing PACT_OUT before any rollout
    campaign(sizes, rules, procs, n_prod, '--force' in a, '--resume' in a, cores)
    if summarize(rules, inf_out) is False:
        print('campaign complete; summary deferred until the PACT seed files of '
              'this environment exist (run --summary_only then)', flush=True)


if __name__ == '__main__':
    main()
