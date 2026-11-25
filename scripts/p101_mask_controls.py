r"""P101: controls for the non-delay mask, window coverage and shop load.

Evaluation only, CPU only, exact mask. Subcommands run, infer and plan.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. Under the non-delay mask a decision may choose only the operation-machine
   pairs whose processing, after any setup, starts at the decision instant. This
   script measures four quantities of this mask and of the factory benchmark.
   (a) A random control that chooses uniformly among the legal pairs tells how much
   of a schedule's quality the mask fixes by itself. (b) The share of decisions with
   exactly one legal pair tells how often the mask leaves no choice. (c) The stored
   unavailability windows lie on the job-chain horizon H (about 190 h), so their
   coverage of a schedule of 50 or 80 modules (360 to 830 h) is needed as a number.
   (d) The load of the shop per size describes the benchmark. The numbers land in the
   supplement (one table per size) and in the benchmark and method sections.
2. Expected result. The mean of the random rollouts is longer than SPT and than the
   best classical rule at every size. Single-pair decisions are more frequent under
   setups and windows than without them. The stored windows end before the makespan
   of every production schedule. If the random control comes within about 1% of SPT,
   the mask fixes most of each schedule by itself. The paper then states this and
   narrows the claim that the policy's choices carry its results.
3. Contamination. Instances, descriptors and the stored rule arrays are read only.
   Arrays go to a new folder, test_results/FAMILY/maskctl_p101 (--root), which no
   existing script reads (tests/test_p101_mask_controls.py checks every glob over
   test_results/FAMILY). A cell is written through a temporary file and a rename,
   so an interrupted run leaves no partial array, and a stored cell whose design
   differs from the current one is refused, never overwritten (unless --force).
   Rollouts are deterministic. The recorded wall and CPU times are planning figures
   and no paper number depends on them, so the run may share cores.
4. Data accuracy. Instances are those of the stored exact-mask rule campaigns
   (eval_family.py --rule_set masked --first_mask exact --mask_ties exact) at
   10x25 (100 test instances, data/PPVC/10x25+ppvc-mixed), 50x25n30 and 80x25n30
   (30 instances each, data/PPVC/<size>+ppvc-mixed-n30). Each SPT rollout of this
   script must reproduce the stored array test_results/FAMILY/<size>_xmask/
   Result_SPT_<combination>.npy bit for bit, or infer refuses.

LAUNCH NOTE OF THE ARM WITH TEN SCHEDULES OF THE POLICY (run --trace_arm polten)

1. Purpose. At the training size under setups and a second rule, the best of ten
   random rollouts ties the best classical rule, so the policy's lead over the rules
   there may be what ten tries of any chooser give. This arm gives the policy the same
   ten tries: per trained run and instance, the greedy schedule and nine schedules
   sampled from its action distribution, all under the exact mask. Does the policy's
   lead survive when the control gets the same ten tries? The numbers land in the
   supplement (one table per size) and in the sentence of the results section that
   compares the policy with the random control. Runs (--ten_runs): every run of
   --policy_seeds at 10x25, where the question arises and the pooled interval over
   runs is needed, and the selected run only at 50x25n30 and 80x25n30, the run that
   the paper deploys at the production batches: 17.6 core-hours, where every run at
   every size would take 62.7.
2. Expected result. The policy's best of ten is shorter than the random best of ten in
   every combination at every size, with a 95% interval that excludes zero. If a
   combination ties at the training size, the paper reports that boundary and restricts
   the claim to the sizes where the lead holds.
3. Contamination. Checkpoints, instances, descriptors and the stored arrays are read
   only. Cells go to <root>/<size>/<combination>/polten_<model>.npz and .json beside
   the other cells of this script (written through a temporary file and a rename; a
   cell of another design, checkpoint or instance list is refused, never overwritten,
   unless --force). The sampled schedules are a function of fixed seeds and the
   rollouts run on one torch thread, so the cells are not timing measurements and the
   run may share cores; the recorded times are planning figures.
4. Data accuracy. Instances, descriptors and the exact mask as in question 4 above.
   Schedule 0 of every instance must equal the stored greedy array of the same run
   (test_results/FAMILY/<size>_xmask/Result_greedy+<model>_<combination>.npy) and the
   greedy trace cell of the same run where they exist, or run and infer refuse. Every
   schedule passes the validator of this script (reference_sim.validate_schedule and
   the makespan against the last completion), or run refuses. The parameter count of
   every checkpoint is asserted (32786) and a checkpoint trained with other mask
   options is refused.

QUANTITIES (per size and combination, every rollout in FJSPEnvFamily with
first_mask='exact', mask_ties='exact', mask_mode='nondelay', as eval_family.py)

  (a) random control   Ten rollouts per instance. At each decision one legal pair is
                       drawn uniformly from the pairs the mask admits. Draw k of
                       instance i uses numpy default_rng(SeedSequence([20261113,
                       size code, i, k])), size code 10, 50 or 80; the stream does
                       not depend on the combination. Reported per instance as the
                       mean and the best (minimum) makespan of the ten rollouts, and
                       as the relative difference 100 (x - ref) / ref to SPT, to the
                       per-instance best of FIFO, MOR, SPT and MWKR (stored masked
                       rule arrays) and, when its arrays exist, to the released
                       policy (per-instance mean over its training seeds). Mean over
                       instances with a 95% percentile bootstrap over instances
                       (20,000 resamples, a fresh default_rng(7) per interval).
  (b) legal pairs      Along the rollout of the trace arm (SPT now, the policy's
                       greedy rollout once its checkpoints exist), before every
                       decision, the number of pairs the mask admits. Reported as the
                       share of decisions with exactly one legal pair and the mean and
                       median number of legal pairs per decision, each over all
                       decisions of all instances of the cell.
  (c) window coverage  For W, LW, SW and LSW, per instance, the last window end over
                       the makespan of the trace arm's rollout, and the machine time
                       inside windows before the makespan over machines times
                       makespan. Mean over instances, with the range.
  (d) load             Per instance, the total processing time of the chosen
                       assignments (setups excluded) over machines times makespan,
                       and the makespan over the makespan estimate at the empty
                       schedule (max_i of the environment's op_ct_lb at the first
                       state, the job-chain bound with lags and the minimum setup into
                       each operation where active). Mean over instances. The
                       utilization of the busiest station type is stored beside them.
  (e) ten schedules    Per trained run of the released policy and instance, ten
      of the policy    complete schedules under the exact mask, loaded and drawn by
                       p97_policy_anytime.PolicyRoller.rollout_batch (the code of the
                       paper's arm with sampled schedules; imported, not copied).
                       Schedule 0 is the greedy schedule (argmax at every decision).
                       Schedule k = 1..9 samples every decision from the policy's
                       action distribution over the legal pairs (common_utils.
                       sample_action) in an environment of its own, after
                       torch.manual_seed(t_k), t_k = SeedSequence([20261113, 2, size
                       code, instance index, combination index in N, L, S, W, LS,
                       LW, SW, LSW, training seed, k]).generate_state(1, uint64) // 2.
                       The stream is a function of the size, the instance, the
                       combination and the trained run, and differs from the stream
                       of (a). Reported per trained run, for the run that the
                       selection rule picks (arm_registry.select_seed: the lowest
                       validation score of the exact-mask runs of --policy_seeds),
                       and pooled over the runs, each as 100 (x - ref) / ref per
                       instance, negative = the policy shorter:
                         best of the ten policy schedules against the best of the
                           ten random rollouts, with the share of instances on which
                           the policy's best is shorter, equal and longer;
                         best of the ten against the best classical rule, and against
                           the per-instance best of the seven rules FIFO, MOR, SPT,
                           MWKR, ECT, SSPT and LLF when the stored arrays of all seven
                           exist;
                         the greedy schedule against the mean of the ten random
                           rollouts (the comparison without repeated tries);
                         the mean of the nine sampled schedules against the mean of the
                           ten random rollouts.
                       Per run and for the selected run: mean over instances and the
                       95% percentile bootstrap of (a). Pooled: the crossed
                       hierarchical bootstrap over runs and instances
                       (stats_family.hier_boot, the same B, fresh default_rng(11)), with
                       the seed-level t interval in the JSON. A size holds the cells
                       of every run of --policy_seeds or of the selected run only
                       (--ten_runs); the outputs state which, and a size with the
                       selected run only has its rows and no pooled row.

FOLDERS
  <root>/<size>/<combination>/random.npz, .json      control (a)
  <root>/<size>/<combination>/trace_<arm>.npz, .json  (b), (c), (d) of one trace arm;
                                                      arm SPT, or the model name of a
                                                      policy run
  <root>/<size>/<combination>/polten_<model>.npz, .json  (e) of one policy run:
                                                      makespan [instances, 10], column 0
                                                      greedy; torch_seed [instances, 10]
                                                      (column 0 unused)
  infer writes results/mask_controls.json, results/mask_controls.md and
  manuscript/macros_maskctl.tex (macro prefix Mctl), or the same paths under --out. The
  quantities (e) enter these files only when the cells of (e) exist (--ten_arm auto),
  so without them the files are those of the arms (a) to (d) alone.

USAGE (repository root; CPU only)
  python scripts/p101_mask_controls.py plan  [--parts K] [--trace_arm SPT|policy|polten]
  python scripts/p101_mask_controls.py run   [--sizes ...] [--combos ...] [--part i --parts K]
  python scripts/p101_mask_controls.py run   --trace_arm policy [--policy_seeds 301-305]
  python scripts/p101_mask_controls.py run   --trace_arm polten [--policy_seeds 301-305]
                                             [--ten_runs RUNS] [--part i --parts K]
  python scripts/p101_mask_controls.py infer [--trace_arms SPT,policy] [--policy_seeds 301-305]
                                             [--trace_seeds 301] [--ten_arm auto|require|off]
                                             [--ten_runs RUNS]
  RUNS is all, selected, or one choice per size, size=all|selected separated by commas;
  plan and run default to 10x25=all,50x25n30=selected,80x25n30=selected. infer accepts
  at each size the cells of every run or of the selected run only, and with --ten_runs
  requires the form it names.
  Smoke: --root <folder> --n 2 on run, and infer --partial --root <folder> --out <folder>.
  The arm (e) of a stored run of another environment: run --trace_arm polten
  --policy_models <model> --root <folder> [--draws 2], then infer --partial
  --policy_models <model> (the selected run is then the first model named).
"""
import argparse
import copy
import glob
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
sys.path.insert(0, ROOT_DIR)
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

R8 = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
WIN_COMBOS = ('W', 'LW', 'SW', 'LSW')
SETUP_COMBOS = ('S', 'LS', 'SW', 'LSW')
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
RULE_SEED = 50             # np.random.seed after set_initial_data (eval_family.py, seed_test)
N_DRAWS = 10
RANDOM_SEED = 20261113
SIZES = {
    '10x25': dict(data='data/FAMILY/10x25', base='data/PPVC/10x25+ppvc-mixed', n=100,
                  word='Ten', code=10, modules=10),
    '50x25n30': dict(data='data/FAMILY/50x25n30', base='data/PPVC/50x25+ppvc-mixed-n30',
                     n=30, word='Fifty', code=50, modules=50),
    '80x25n30': dict(data='data/FAMILY/80x25n30', base='data/PPVC/80x25+ppvc-mixed-n30',
                     n=30, word='Eighty', code=80, modules=80),
}
MASK = {'first_mask': 'exact', 'mask_ties': 'exact'}
RESULTS_ROOT = 'test_results/FAMILY'
DEFAULT_ROOT = 'test_results/FAMILY/maskctl_p101'
OUT_FILES = {'json': 'results/mask_controls.json', 'md': 'results/mask_controls.md',
             'macros': 'manuscript/macros_maskctl.tex'}
GENERATOR = 'scripts/p101_mask_controls.py infer'
PREFIX = 'Mctl'
B = 20000
CI_SEED = 7
POLICY_SEEDS = (301, 302, 303, 304, 305)
EXPECT_PARAMS = 32786
# eval_family.py ARCH_KEYS: the network and observation settings restored from a run's
# training configuration
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode',
             'mask_mode', 'pair_extra')
# planning cost, CPU seconds per instance and cell kind. Random and trace are the
# process CPU times of the LSW smoke cells (2026-11-13, one shared core). Policy is the
# exact campaign's per-instance rollout time (scripts/eval_newseeds_plan.py SPEED) and,
# at 10x25, the smoke cell.
COST_S = {'10x25': {'random': 0.55, 'trace': 0.20, 'policy': 0.60},
          '50x25n30': {'random': 11.4, 'trace': 1.6, 'policy': 16.3},
          '80x25n30': {'random': 33.6, 'trace': 3.3, 'policy': 33.3}}
# the arm with ten schedules of the policy (QUANTITIES (e)): process CPU seconds per
# instance of the LSW smoke cells with a stored checkpoint of the same architecture
# (2026-11-13, one shared core): 10x25 three instances with ten schedules, 50x25n30 one
# instance with ten, 80x25n30 one instance with two schedules times five
COST_S['10x25']['polten'] = 5.7
COST_S['50x25n30']['polten'] = 54.0
COST_S['80x25n30']['polten'] = 115.0
POLTEN_COST_NOTE = 'process CPU of the LSW smoke cells, 2026-11-13'
TEN_TAG = 2                # second word of the stream of (e); the random control has none
AWARE = ('ECT', 'SSPT', 'LLF')          # stored exact-mask arrays of scripts/p72_aware_rules.py
HIER_SEED = 11             # fresh default_rng of the crossed bootstrap (stats_family, p88)
TEN_QUANTITIES = ('randbest', 'rule', 'rule7', 'greedy_rand', 'samp_rand')
TEN_RUN_CHOICES = ('all', 'selected')    # per size: every run of --policy_seeds, or the
#                                          selected run only
DEFAULT_TEN_RUNS = '10x25=all,50x25n30=selected,80x25n30=selected'
TEN_RUN_TEXT = {'all': 'every run of the arm', 'selected': 'the selected run only'}


def check_mask_options():
    """MASK must equal the options arm_registry gives eval_family.py in the exact
    environment."""
    flags = REG.MASK_OPTIONS['exact']
    reg = {f.lstrip('-'): v for f, v in zip(flags[0::2], flags[1::2])}
    if reg != MASK:
        sys.exit(f'REFUSING: mask options {MASK} differ from arm_registry {reg}')
    if os.environ.get('PACT_ENV', 'exact') != 'exact':
        sys.exit('REFUSING: PACT_ENV is set to another environment; this study runs in '
                 'the exact-mask environment (unset PACT_ENV or set PACT_ENV=exact)')


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def write_json_atomic(path, obj, indent=1):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def save_npz_atomic(path, **arrays):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path[:-4] + '.__tmp.npz'
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT_DIR,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def parse_list(text, allowed, what):
    vals = [x for x in str(text).replace(' ', '').split(',') if x]
    bad = [x for x in vals if x not in allowed]
    if bad:
        sys.exit(f'unknown {what} {bad}; known: {list(allowed)}')
    return vals


def parse_ten_runs(text):
    """{size: 'all' | 'selected'} of --ten_runs: one word for every size, or one
    size=choice item per size separated by commas."""
    text = str(text).replace(' ', '')
    if text in TEN_RUN_CHOICES:
        return {s: text for s in SIZES}
    out = {}
    for item in [x for x in text.split(',') if x]:
        size, sep, choice = item.partition('=')
        if not sep or size not in SIZES or choice not in TEN_RUN_CHOICES or size in out:
            sys.exit(f'--ten_runs {text}: want all, selected, or one size=all|selected item '
                     f'for each of {", ".join(SIZES)}')
        out[size] = choice
    miss = [s for s in SIZES if s not in out]
    if miss:
        sys.exit(f'--ten_runs {text} names no choice for {", ".join(miss)}')
    return {s: out[s] for s in SIZES}


def ten_runs_text(ten_runs):
    """The --ten_runs value of a parsed choice (one word when every size has the same)."""
    vals = set(ten_runs.values())
    return vals.pop() if len(vals) == 1 else ','.join(f'{s}={ten_runs[s]}' for s in SIZES)


def names_of(size, n=None):
    """Sorted instance names of a size (the order of eval_family.py and of the stored
    arrays), optionally the first n."""
    spec = SIZES[size]
    names = sorted(os.path.basename(p)[:-len('.desc.json')] for p in
                   glob.glob(os.path.join(ROOT_DIR, spec['data'], 'desc', 'N',
                                          '*.desc.json')))
    if len(names) != spec['n']:
        sys.exit(f'REFUSING: expected {spec["n"]} descriptors under {spec["data"]}/desc/N, '
                 f'found {len(names)}')
    return names[:n] if n else names


def load_cell(size, combo, n=None):
    """[(name, jl, pt, meta, desc)] of one cell, read only. The base instance is
    <base>/<name>, the convention of eval_family.py --base."""
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    spec = SIZES[size]
    out = []
    for name in names_of(size, n):
        with open(os.path.join(ROOT_DIR, spec['data'], 'desc', combo,
                               f'{name}.desc.json')) as f:
            rec = json.load(f)
        jl, pt, meta = load_instance(os.path.join(ROOT_DIR, spec['base'], name))
        out.append((name, jl, pt, meta,
                    ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    return out


def instance_index(name):
    return int(name.split('_')[-1])


def draw_seeds(size, name):
    """SeedSequence entropy of the N_DRAWS random rollouts of one instance."""
    return [[RANDOM_SEED, SIZES[size]['code'], instance_index(name), k]
            for k in range(N_DRAWS)]


def model_seed(model):
    """Training seed of a run name: its -s<seed> suffix, or 301 for a name without one
    (the legacy seed-301 runs of arm_registry)."""
    m = re.search(r'-s(\d+)$', model)
    return int(m.group(1)) if m else 301


def ten_entropy(size, name, combo, seed, k):
    """SeedSequence entropy of sampled schedule k (1..9) of one instance, combination
    and trained run in the arm with ten schedules of the policy."""
    return [RANDOM_SEED, TEN_TAG, SIZES[size]['code'], instance_index(name), R8.index(combo),
            int(seed), int(k)]


def ten_torch_seed(entropy):
    """The torch.manual_seed value of one sampled schedule: the first 64-bit word of the
    SeedSequence state, halved so that it is a non-negative 63-bit integer."""
    return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint64)[0]) >> 1


# ---------------------------------------------------------------------------
# environment and rollouts
# ---------------------------------------------------------------------------

def _cpu_torch():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        sys.exit('REFUSING: CUDA is visible; run with CUDA_VISIBLE_DEVICES= (CPU only)')
    from params import configs
    configs.device = 'cpu'
    import fjsp_env_same_op_nums
    if str(fjsp_env_same_op_nums.EnvState.device) != 'cpu':
        sys.exit('REFUSING: the environment state was built for another device')
    return torch


def make_env(jl, pt, meta, descs, mask_mode='nondelay', pair_extra='none',
             delta_mode='min'):
    """A batch of len(descs) copies of one instance, built as eval_family.py builds the
    evaluation environment (mask options of the exact environment, delta mode set
    before the data, type lists when the instance carries types). Copies of one
    instance share the processing-time normalization, so each copy behaves as the
    single-instance environment."""
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    E = len(descs)
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True, blind_channels=(),
                        mask_mode=mask_mode, pair_extra=pair_extra, **MASK)
    env.delta_mode = delta_mode
    env.reward_delta_env = (delta_mode != 'zero')
    typed = meta.get('op_type') is not None
    env.set_initial_data([jl] * E, [pt] * E, descriptor_list=list(descs),
                         op_type_list=[meta['op_type']] * E if typed else None,
                         mch_type_list=[meta['mch_type']] * E if typed else None)
    for k, v in MASK.items():
        if env.mask_setting(k) != v:
            raise RuntimeError(f'the environment reads {k}={env.mask_setting(k)}')
    if not env._first_mask:
        raise RuntimeError('the exact first-decision mask was not applied')
    return env


def legal_counts(env):
    """Number of pairs the mask admits in each environment of the batch, [E]."""
    E = env.dynamic_pair_mask.shape[0]
    return (~np.asarray(env.dynamic_pair_mask, dtype=bool)).reshape(E, -1).sum(axis=1)


def lb0_hours(env):
    """Makespan estimate at the empty schedule, in hours, per environment [E]."""
    return np.max(env.op_ct_lb, axis=1) / env._slope


def validate(jl, pt, desc, amch, op_ct, makespan):
    """Violations of one finished schedule (reference_sim.validate_schedule, the check
    eval_family.py runs), plus the makespan against the completion times."""
    from constraint_family.reference_sim import validate_schedule
    if (amch < 0).any():
        return ['an operation was never assigned a machine']
    op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
    v = list(validate_schedule(jl, pt, desc, op_st, op_ct, amch))
    if abs(float(np.max(op_ct)) - float(makespan)) > 1e-6:
        v.append(f'makespan {makespan} differs from the last completion {np.max(op_ct)}')
    return v


def rule_rollout(rule, jl, pt, meta, desc):
    """One rollout of a classical rule restricted to the legal pairs
    (common_utils.masked_rule_action, the rule set of the stored exact-mask arrays),
    with the legal-pair count before every decision. Returns a dict."""
    from common_utils import masked_rule_action
    M = pt.shape[1]
    env = make_env(jl, pt, meta, [desc])
    lb0 = float(lb0_hours(env)[0])
    amch = np.full(pt.shape[0], -1, dtype=int)
    counts = []
    np.random.seed(RULE_SEED)
    while not env.done().all():
        c = int(legal_counts(env)[0])
        if c < 1:
            raise RuntimeError('the mask admits no pair')
        counts.append(c)
        a = int(masked_rule_action(rule, env))
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    return dict(makespan=float(env.current_makespan[0]), amch=amch,
                op_ct=env.true_op_ct[0].copy(), lb0=lb0,
                counts=np.asarray(counts, dtype=np.int32))


def random_rollouts(jl, pt, meta, desc, seeds):
    """len(seeds) rollouts of the random control in one batch: at every decision each
    copy draws one legal pair uniformly with its own generator. Returns makespans [D],
    machine assignments [D, N] and completion times [D, N]."""
    M = pt.shape[1]
    D = len(seeds)
    env = make_env(jl, pt, meta, [desc] * D)
    rngs = [np.random.default_rng(np.random.SeedSequence(s)) for s in seeds]
    amch = np.full((D, pt.shape[0]), -1, dtype=int)
    rows = np.arange(D)
    while not env.done().all():
        legal = ~np.asarray(env.dynamic_pair_mask, dtype=bool).reshape(D, -1)
        acts = np.empty(D, dtype=np.int64)
        for k in range(D):
            idx = np.flatnonzero(legal[k])
            if idx.size == 0:
                raise RuntimeError('the mask admits no pair')
            acts[k] = idx[int(rngs[k].integers(idx.size))]
        amch[rows, env.candidate[rows, acts // M]] = acts % M
        env.step(acts)
    return (env.current_makespan.astype(float).copy(), amch,
            env.true_op_ct.astype(float).copy())


class PolicyRunner:
    """Greedy rollouts of a trained run, loaded and rolled out as eval_family.py does
    (configuration restored from the training snapshot, one torch thread on the CPU,
    the true constraint token)."""

    def __init__(self):
        self.torch = _cpu_torch()
        from params import configs
        self.configs = configs
        self.defaults = copy.deepcopy(vars(configs))
        self.cache = {}

    def _reset(self):
        d = vars(self.configs)
        d.clear()
        d.update(copy.deepcopy(self.defaults))
        self.configs.device = 'cpu'

    def load(self, model):
        if model in self.cache:
            return self.cache[model]
        from constraint_family.fjsp_env_family import resolve_delta_mode
        from model.PPO import PPO_initialize
        snap_p = os.path.join(ROOT_DIR, 'train_log', 'FAMILY', f'config_{model}.json')
        ckpt_p = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY', f'{model}.pth')
        for p in (snap_p, ckpt_p):
            if not os.path.exists(p):
                sys.exit(f'REFUSING: {p} does not exist (the run is not trained yet)')
        with open(snap_p) as f:
            snap = json.load(f)
        for k in MASK:
            if snap.get(k) not in (None, MASK[k]):
                sys.exit(f'REFUSING: {model} was trained with {k}={snap[k]}; this study '
                         f'evaluates with {k}={MASK[k]}')
        self._reset()
        for k in ARCH_KEYS:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v in ('True', 'False'):
                    v = v == 'True'
                setattr(self.configs, k, v)
        self.configs.device = 'cpu'
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(self.torch.load(ckpt_p, map_location='cpu'))
        ppo.policy.eval()
        n_par = sum(p.numel() for p in ppo.policy.parameters())
        if n_par != EXPECT_PARAMS:
            sys.exit(f'REFUSING: {model} has {n_par} parameters, expected {EXPECT_PARAMS}')
        rec = dict(model=model, params=int(n_par),
                   delta_mode=resolve_delta_mode(self.configs),
                   mask_mode=getattr(self.configs, 'mask_mode', 'nondelay'),
                   pair_extra=getattr(self.configs, 'pair_extra', 'none'),
                   snapshot_sha256=sha256_file(snap_p), checkpoint_sha256=sha256_file(ckpt_p))
        self.cache[model] = (ppo, rec)
        return self.cache[model]

    def rollout(self, model, jl, pt, meta, desc):
        """One greedy rollout with the legal-pair count before every decision."""
        from common_utils import greedy_select_action
        from constraint_family.descriptor import compute_token
        ppo, rec = self.load(model)
        M = pt.shape[1]
        env = make_env(jl, pt, meta, [desc], mask_mode=rec['mask_mode'],
                       pair_extra=rec['pair_extra'], delta_mode=rec['delta_mode'])
        lb0 = float(lb0_hours(env)[0])
        env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()), M)])
        env._attach_token()
        state = env.state
        amch = np.full(pt.shape[0], -1, dtype=int)
        counts = []
        torch = self.torch
        while not env.done().all():
            counts.append(int(legal_counts(env)[0]))
            with torch.no_grad():
                pi, _ = ppo.policy(
                    fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                    mch_type=state.mch_type_tensor, token=state.token_tensor)
            a = int(greedy_select_action(pi).cpu().numpy()[0])
            amch[env.candidate[0, a // M]] = a % M
            state, _, _ = env.step(np.array([a]))
        return dict(makespan=float(env.current_makespan[0]), amch=amch,
                    op_ct=env.true_op_ct[0].copy(), lb0=lb0,
                    counts=np.asarray(counts, dtype=np.int32))


# ---------------------------------------------------------------------------
# the arm with ten schedules of the policy (QUANTITIES (e))
# ---------------------------------------------------------------------------

def _p97():
    """scripts/p97_policy_anytime.py, imported on first use. It imports the modules of
    the timed campaign and sets the working directory to the repository root, which the
    other arms of this script and the scripts that import this one do not need."""
    import p97_policy_anytime as P97
    return P97


def run_files(model):
    """(checkpoint, training configuration) of a run, relative to the repository root."""
    return (os.path.join('trained_network', 'FAMILY', f'{model}.pth'),
            os.path.join('train_log', 'FAMILY', f'config_{model}.json'))


def ten_roller(runner, model, size, combo, name):
    """p97's PolicyRoller of one run and one instance under the exact mask, with the
    checks of PolicyRunner.load: both files exist, the training configuration holds no
    other mask option, the network has EXPECT_PARAMS parameters, and the environment
    applies the exact first-decision mask. The configuration is reset to the defaults
    of params.py first, so a run restores its own settings only."""
    ckpt, cfg = run_files(model)
    for p in (ckpt, cfg):
        if not os.path.exists(os.path.join(ROOT_DIR, p)):
            sys.exit(f'REFUSING: {p} does not exist (the run is not trained yet)')
    with open(os.path.join(ROOT_DIR, cfg)) as f:
        snap = json.load(f)
    for k in MASK:
        if snap.get(k) not in (None, MASK[k]):
            sys.exit(f'REFUSING: {model} was trained with {k}={snap[k]}; this study '
                     f'evaluates with {k}={MASK[k]}')
    P97 = _p97()
    if P97.instance_name(instance_index(name)) != name:
        sys.exit(f'REFUSING: p97 names instance {instance_index(name)} '
                 f'{P97.instance_name(instance_index(name))}, this cell {name}')
    runner._reset()
    r = P97.PolicyRoller(dict(model=model, checkpoint=ckpt, config=cfg, size=size,
                              combination=combo, instance=instance_index(name),
                              first_batch=1, mask=dict(MASK)))
    if r.n_param != EXPECT_PARAMS:
        sys.exit(f'REFUSING: {model} has {r.n_param} parameters, expected {EXPECT_PARAMS}')
    for k, v in MASK.items():
        if r._env.mask_setting(k) != v:
            raise RuntimeError(f'the environment reads {k}={r._env.mask_setting(k)}')
    if not r._env._first_mask:
        raise RuntimeError('the exact first-decision mask was not applied')
    return r


def ten_rollouts(roller, size, name, combo, seed, draws=N_DRAWS):
    """The draws schedules of one instance: schedule 0 greedy, schedule k sampled after
    torch.manual_seed(ten_torch_seed(ten_entropy(..., k))), each in an environment of
    its own (rollout_batch with a batch of one). Returns makespans [draws], machine
    assignments [draws, N], completion times [draws, N] and the torch seeds [draws]
    (0 for the greedy schedule, which draws nothing)."""
    out, seeds = [roller.rollout_batch(True, None, 1)], [0]
    for k in range(1, draws):
        s = ten_torch_seed(ten_entropy(size, name, combo, seed, k))
        out.append(roller.rollout_batch(False, s, 1))
        seeds.append(s)
    ms = np.array([float(np.asarray(o[0]).reshape(-1)[0]) for o in out])
    amch = np.stack([np.asarray(o[1])[0] for o in out])
    ct = np.stack([np.asarray(o[2], dtype=float)[0] for o in out])
    return ms, amch, ct, np.array(seeds, dtype=np.uint64)


def stored_greedy_path(size, combo, model):
    return os.path.join(ROOT_DIR, RESULTS_ROOT, f'{size}_xmask',
                        f'Result_greedy+{model}_{combo}.npy')


def greedy_references(root, size, combo, model, names, stop=sys.exit):
    """{source: makespans over names} of every stored greedy result of the run for this
    cell: the exact campaign's array and the greedy trace cell of this script. A source
    that exists but cannot cover the instances stops through `stop`."""
    out = {}
    p = stored_greedy_path(size, combo, model)
    if os.path.exists(p):
        a = np.load(p)
        a = (a[:, 0] if a.ndim > 1 else a).astype(float)
        if len(a) < len(names):
            stop(f'{p} has {len(a)} rows, the study needs {len(names)}')
        out[os.path.relpath(p, ROOT_DIR)] = a[:len(names)]
    npz, js = cell_paths(root, size, combo, trace_label(model))
    if os.path.exists(npz) and os.path.exists(js):
        with open(js) as f:
            inst = json.load(f).get('instances', [])
        if any(nm not in inst for nm in names):
            stop(f'{js} does not cover the instances of the cell')
        with np.load(npz) as z:
            tm = np.asarray(z['makespan'], dtype=float)
        out[f'trace cell {trace_label(model)}'] = tm[[inst.index(nm) for nm in names]]
    return out


def ten_label(model):
    return f'polten_{model}'


# ---------------------------------------------------------------------------
# coverage and load
# ---------------------------------------------------------------------------

def window_coverage(desc, makespan, n_machines):
    """(last window end / makespan, machine time inside windows before the makespan /
    (machines x makespan), last window end in hours); NaN without windows."""
    if not desc.has_window or desc.windows is None:
        return float('nan'), float('nan'), float('nan')
    ms = float(makespan)
    spans = [(float(s), float(e)) for w in desc.windows for (s, e) in w]
    last = max((e for _, e in spans), default=0.0)
    inside = sum(max(0.0, min(e, ms) - min(s, ms)) for s, e in spans)
    return last / ms, inside / (n_machines * ms), last


def shop_load(pt, amch, makespan, lb0, mch_type=None):
    """(processing time of the chosen assignments / (machines x makespan),
    makespan / estimate at the empty schedule, utilization of the busiest station
    type or NaN without station types)."""
    N, M = pt.shape
    busy = pt[np.arange(N), amch].astype(float)
    util = float(busy.sum() / (M * makespan))
    worst = float('nan')
    if mch_type is not None:
        mt = np.asarray(mch_type)
        per = [busy[np.isin(amch, np.flatnonzero(mt == t))].sum() / ((mt == t).sum() * makespan)
               for t in np.unique(mt)]
        worst = float(max(per))
    return util, float(makespan / lb0), worst


# ---------------------------------------------------------------------------
# cells: design record, paths, resumability
# ---------------------------------------------------------------------------

def design(kind, size, combo, model=None, draws=N_DRAWS):
    """The fields that define a cell's content; a stored cell with other fields is
    refused. `draws` applies to the arm with ten schedules of the policy only."""
    d = dict(kind=kind, size=size, combo=combo, mask=MASK, mask_mode='nondelay',
             data=SIZES[size]['data'], base=SIZES[size]['base'])
    if kind == 'random':
        d.update(draws=N_DRAWS, stream=f'default_rng(SeedSequence([{RANDOM_SEED}, '
                                       f'{SIZES[size]["code"]}, instance index, draw]))',
                 choice='uniform over the legal pairs')
    elif kind == 'polten':
        ckpt = os.path.join(ROOT_DIR, run_files(model)[0])
        d.update(model=model, checkpoint_sha256=(sha256_file(ckpt) if os.path.exists(ckpt)
                                                 else None),
                 trained_seed=model_seed(model), draws=int(draws),
                 schedules=('0: greedy, common_utils.greedy_select_action at every decision; '
                            f'1 to {int(draws) - 1}: common_utils.sample_action at every '
                            'decision after torch.manual_seed(t_k), one schedule per '
                            'environment'),
                 route='p97_policy_anytime.PolicyRoller.rollout_batch',
                 stream=(f't_k = SeedSequence([{RANDOM_SEED}, {TEN_TAG}, '
                         f'{SIZES[size]["code"]}, instance index, {R8.index(combo)}, '
                         f'{model_seed(model)}, k]).generate_state(1, uint64) // 2'))
    elif model is None:
        d.update(rule='SPT', rule_set='masked', rule_seed=RULE_SEED)
    else:
        ckpt = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY', f'{model}.pth')
        d.update(model=model, checkpoint_sha256=(sha256_file(ckpt) if os.path.exists(ckpt)
                                                 else None))
    return d


def trace_label(model=None):
    return 'trace_SPT' if model is None else f'trace_{model}'


def cell_paths(root, size, combo, label):
    stem = os.path.join(root, size, combo, label)
    return stem + '.npz', stem + '.json'


def cell_state(root, size, combo, label, want, names):
    """'done' when the cell's files hold this design and these instances, 'absent'
    when missing; any other stored content is refused."""
    npz, js = cell_paths(root, size, combo, label)
    if not (os.path.exists(npz) and os.path.exists(js)):
        return 'absent'
    with open(js) as f:
        rec = json.load(f)
    if rec.get('design') != jsonable(want) or rec.get('instances') != names:
        return 'other'
    return 'done'


def run_record(want, names, t0, c0):
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(design=jsonable(want), instances=names, n=len(names), wall_s=wall,
                cpu_s=cpu, cpu_share=cpu / max(wall, 1e-9),
                affinity=sorted(os.sched_getaffinity(0)), loadavg=list(os.getloadavg()),
                host=socket.gethostname(), python=platform.python_version(),
                omp=os.environ.get('OMP_NUM_THREADS'), commit=git_commit(),
                finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                timing_note='planning figures only; rollouts are deterministic')


def run_random(root, size, combo, n, force):
    names = names_of(size, n)
    want = design('random', size, combo)
    st = cell_state(root, size, combo, 'random', want, names)
    if st == 'done' and not force:
        return 'skip', 0.0
    if st == 'other' and not force:
        sys.exit(f'REFUSING: {cell_paths(root, size, combo, "random")[1]} holds another '
                 f'design or instance list; use --force or another --root')
    t0, c0 = time.perf_counter(), time.process_time()
    cell = load_cell(size, combo, n)
    ms = np.zeros((len(cell), N_DRAWS))
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        m, amch, op_ct = random_rollouts(jl, pt, meta, desc, draw_seeds(size, name))
        for k in range(N_DRAWS):
            v = validate(jl, pt, desc, amch[k], op_ct[k], m[k])
            if v:
                sys.exit(f'REFUSING: random rollout {k} of {size}/{combo}/{name} is '
                         f'infeasible: {v[:2]}')
        ms[i] = m
    npz, js = cell_paths(root, size, combo, 'random')
    save_npz_atomic(npz, makespan=ms)
    rec = run_record(want, names, t0, c0)
    rec['seeds'] = {name: draw_seeds(size, name) for name in names}
    write_json_atomic(js, rec)
    return 'done', rec['cpu_s']


def run_trace(root, size, combo, n, force, runner=None, model=None):
    names = names_of(size, n)
    label = trace_label(model)
    want = design('trace', size, combo, model)
    st = cell_state(root, size, combo, label, want, names)
    if st == 'done' and not force:
        return 'skip', 0.0
    if st == 'other' and not force:
        sys.exit(f'REFUSING: {cell_paths(root, size, combo, label)[1]} holds another '
                 f'design or instance list; use --force or another --root')
    t0, c0 = time.perf_counter(), time.process_time()
    cell = load_cell(size, combo, n)
    n_i, N, M = len(cell), cell[0][2].shape[0], cell[0][2].shape[1]
    out = {k: np.zeros(n_i) for k in ('makespan', 'lb0', 'util', 'ms_over_lb0',
                                      'util_max_type', 'cov_end', 'cov_time',
                                      'last_window_end', 'horizon')}
    out['n_decisions'] = np.zeros(n_i, dtype=np.int64)
    out['n_one_pair'] = np.zeros(n_i, dtype=np.int64)
    out['amch'] = np.zeros((n_i, N), dtype=np.int16)
    out['op_ct'] = np.zeros((n_i, N))
    counts = []
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        r = (rule_rollout('SPT', jl, pt, meta, desc) if model is None
             else runner.rollout(model, jl, pt, meta, desc))
        v = validate(jl, pt, desc, r['amch'], r['op_ct'], r['makespan'])
        if v:
            sys.exit(f'REFUSING: the {label} schedule of {size}/{combo}/{name} is '
                     f'infeasible: {v[:2]}')
        if len(r['counts']) != N:
            sys.exit(f'REFUSING: {len(r["counts"])} decisions for {N} operations')
        ce, ct, le = window_coverage(desc, r['makespan'], M)
        u, ml, w = shop_load(pt, r['amch'], r['makespan'], r['lb0'], meta.get('mch_type'))
        for k, val in (('makespan', r['makespan']), ('lb0', r['lb0']), ('util', u),
                       ('ms_over_lb0', ml), ('util_max_type', w), ('cov_end', ce),
                       ('cov_time', ct), ('last_window_end', le),
                       ('horizon', float(desc.horizon or 0))):
            out[k][i] = val
        out['n_decisions'][i] = len(r['counts'])
        out['n_one_pair'][i] = int((r['counts'] == 1).sum())
        out['amch'][i] = r['amch']
        out['op_ct'][i] = r['op_ct']
        counts.append(r['counts'])
    out['legal_counts'] = np.concatenate(counts).astype(np.int32)
    npz, js = cell_paths(root, size, combo, label)
    save_npz_atomic(npz, **out)
    rec = run_record(want, names, t0, c0)
    if model is not None:
        rec['policy'] = runner.load(model)[1]
    write_json_atomic(js, rec)
    return 'done', rec['cpu_s']


def run_polten(root, size, combo, n, force, runner, model, draws=N_DRAWS):
    """One cell of the arm with ten schedules of the policy (QUANTITIES (e))."""
    names = names_of(size, n)
    label = ten_label(model)
    want = design('polten', size, combo, model, draws)
    st = cell_state(root, size, combo, label, want, names)
    if st == 'done' and not force:
        return 'skip', 0.0
    if st == 'other' and not force:
        sys.exit(f'REFUSING: {cell_paths(root, size, combo, label)[1]} holds another '
                 f'design, checkpoint or instance list; use --force or another --root')
    if draws < 2:
        sys.exit('REFUSING: the arm needs the greedy schedule and at least one sampled one')
    t0, c0 = time.perf_counter(), time.process_time()
    cell = load_cell(size, combo, n)
    seed = model_seed(model)
    refs = greedy_references(root, size, combo, model, names,
                             stop=lambda m: sys.exit(f'REFUSING: {m}'))
    ms = np.zeros((len(cell), draws))
    seeds = np.zeros((len(cell), draws), dtype=np.uint64)
    info = None
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        r = ten_roller(runner, model, size, combo, name)
        # p97 reads the base instance the descriptor names (at 50 and 80 modules partly the
        # folder of the first instances, which holds the same files); the arrays must be
        # those of this cell
        same = (np.array_equal(r.jl, jl) and np.array_equal(r.pt, pt)
                and (meta.get('op_type') is None or (
                    np.array_equal(r.otl[0], meta['op_type'])
                    and np.array_equal(r.mtl[0], meta['mch_type']))))
        if not same:
            sys.exit(f'REFUSING: p97 loads {r.stem} for {size}/{combo}/{name}, whose arrays '
                     f'differ from those of {SIZES[size]["base"]}/{name}')
        m, amch, ct, s = ten_rollouts(r, size, name, combo, seed, draws)
        for k in range(draws):
            v = validate(jl, pt, desc, amch[k], ct[k], m[k])
            if v:
                sys.exit(f'REFUSING: schedule {k} of {model} on {size}/{combo}/{name} is '
                         f'infeasible: {v[:2]}')
        for src, a in refs.items():
            if a[i] != m[0]:
                sys.exit(f'REFUSING: the greedy schedule of {model} on {size}/{combo}/{name} '
                         f'has makespan {m[0]}, {src} holds {a[i]}')
        ms[i], seeds[i] = m, s
        info = info or r.info()
    npz, js = cell_paths(root, size, combo, label)
    save_npz_atomic(npz, makespan=ms, torch_seed=seeds)
    rec = run_record(want, names, t0, c0)
    ckpt, cfg = run_files(model)
    rec['policy'] = dict(model=model, params=int(info['parameter_numel']),
                         delta_mode=info['env_delta_mode'], mask_mode=info['env_mask_mode'],
                         pair_extra=info['env_pair_extra'],
                         snapshot_sha256=sha256_file(os.path.join(ROOT_DIR, cfg)),
                         checkpoint_sha256=sha256_file(os.path.join(ROOT_DIR, ckpt)),
                         torch_version=info['torch_version'],
                         route='p97_policy_anytime.PolicyRoller.rollout_batch')
    rec['seeds'] = {name: [ten_entropy(size, name, combo, seed, k) for k in range(1, draws)]
                    for name in names}
    rec['greedy_check'] = ({src: 'equal' for src in refs} if refs else
                           'no stored greedy array or trace cell of this run')
    write_json_atomic(js, rec)
    return 'done', rec['cpu_s']


# ---------------------------------------------------------------------------
# plan and run
# ---------------------------------------------------------------------------

def policy_model(seed):
    """Released exact-mask run of one training seed (arm_registry, exact environment)."""
    return REG.PREFIX + REG.model('released', None, seed, 'exact')


def cell_list(sizes, combos, trace_arm, seeds, models=None, ten_runs=None, selected=None):
    """(size, combo, kind, model) of every cell of a run, in a fixed order. For the arm
    with ten schedules, a size whose ten_runs choice is 'selected' takes the run
    `selected` only."""
    out = []
    for size in sizes:
        for combo in combos:
            if trace_arm == 'SPT':
                out.append((size, combo, 'random', None))
                out.append((size, combo, 'trace', None))
            else:
                kind = 'polten' if trace_arm == 'polten' else 'policy'
                ms = models or [policy_model(s) for s in seeds]
                if kind == 'polten' and ten_runs and ten_runs[size] == 'selected':
                    ms = [selected]
                for m in ms:
                    out.append((size, combo, kind, m))
    return out


def ten_needs_selection(args):
    return args.trace_arm == 'polten' and any(args.ten_runs[s] == 'selected'
                                              for s in args.sizes)


def cell_cost(c, n):
    size, _, kind, _ = c
    return COST_S[size][kind] * (n or SIZES[size]['n'])


def assign_parts(cells, parts, n):
    """Longest-processing-time assignment of the cells to parts."""
    order = sorted(range(len(cells)), key=lambda i: (-cell_cost(cells[i], n), i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += cell_cost(cells[i], n)
    return [sorted(x) for x in out], loads


def plan(args):
    sel, sel_note = None, ''
    if ten_needs_selection(args):
        try:
            sel = ten_select(args)['model']
        except Refused as e:
            # the parts depend on the number of cells and their cost only, so they are the
            # parts of run whichever run the selection rule picks later
            sel = 'the selected run'
            sel_note = (f'; the selected run cannot be determined yet '
                        f'({str(e).splitlines()[0][:160]}), the parts do not depend on it')
    cells = cell_list(args.sizes, args.combos, args.trace_arm, args.policy_seeds,
                      args.policy_models, getattr(args, 'ten_runs', None), sel)
    parts, loads = assign_parts(cells, args.parts, args.n)
    total = sum(cell_cost(c, args.n) for c in cells)
    print(f'[p101 plan] {len(cells)} cells, estimated {total / 3600:.2f} core-hours of '
          f'CPU time (COST_S, measured on smoke cells), {args.parts} parts of '
          f'{min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h')
    py = sys.executable
    seeds = (f' --policy_seeds {",".join(map(str, args.policy_seeds))}'
             if args.trace_arm in ('policy', 'polten') else '')
    out = dict(cells=len(cells), core_h=total / 3600, part_h=[x / 3600 for x in loads])
    if args.trace_arm == 'polten':
        per = {s: sum(cell_cost(c, args.n) for c in cells if c[0] == s) / 3600
               for s in args.sizes}
        runs = {s: len({c[3] for c in cells if c[0] == s}) for s in args.sizes}
        print('[p101 plan] ten schedules of the policy, core-hours per size: '
              + ', '.join(f'{s} {h:.2f} ({runs[s]} run{"s" if runs[s] != 1 else ""}, '
                          f'{args.ten_runs[s]})' for s, h in per.items())
              + '; per instance '
              + ', '.join(f'{s} {COST_S[s]["polten"]:g} s' for s in args.sizes)
              + f' ({POLTEN_COST_NOTE}){sel_note}')
        out.update(size_h=per, size_runs=runs, ten_runs=dict(args.ten_runs))
        py += ' -B'
        if args.policy_models:
            seeds = f' --policy_models {",".join(args.policy_models)}'
        seeds += f' --ten_runs {ten_runs_text(args.ten_runs)}'
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c '
              f'<CORE_{i}> {py} scripts/p101_mask_controls.py run --trace_arm '
              f'{args.trace_arm}{seeds} --part {i} --parts {args.parts}')
    return out


def run(args):
    sel = ten_select(args)['model'] if ten_needs_selection(args) else None
    if sel:
        print(f'[p101 run] the selected run {sel} at '
              + ', '.join(s for s in args.sizes if args.ten_runs[s] == 'selected'), flush=True)
    cells = cell_list(args.sizes, args.combos, args.trace_arm, args.policy_seeds,
                      args.policy_models, getattr(args, 'ten_runs', None), sel)
    if args.part is not None:
        parts, _ = assign_parts(cells, args.parts, args.n)
        cells = [cells[i] for i in parts[args.part]]
    runner = PolicyRunner() if args.trace_arm in ('policy', 'polten') else None
    if runner is None:
        _cpu_torch()
    t_all = time.perf_counter()
    for k, (size, combo, kind, model) in enumerate(cells):
        if kind == 'random':
            st, c = run_random(args.root, size, combo, args.n, args.force)
        elif kind == 'trace':
            st, c = run_trace(args.root, size, combo, args.n, args.force)
        elif kind == 'polten':
            st, c = run_polten(args.root, size, combo, args.n, args.force, runner, model,
                               args.draws)
        else:
            st, c = run_trace(args.root, size, combo, args.n, args.force, runner, model)
        print(f'[p101 run {k + 1}/{len(cells)}] {size:9s} {combo:4s} {kind:7s} '
              f'{model or "":34s} {st} cpu={c:8.1f}s '
              f'elapsed={time.perf_counter() - t_all:8.0f}s', flush=True)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

class Refused(SystemExit):
    """A precondition of infer failed; the outputs are removed."""


def refuse(msg):
    raise Refused(f'refused: {msg}')


def read_cell(root, size, combo, label, names):
    npz, js = cell_paths(root, size, combo, label)
    if not (os.path.exists(npz) and os.path.exists(js)):
        refuse(f'cell {size}/{combo}/{label} is missing ({npz})')
    with open(js) as f:
        rec = json.load(f)
    if rec.get('instances') != names:
        refuse(f'cell {size}/{combo}/{label} covers {len(rec.get("instances", []))} '
               f'instances, the study needs {len(names)}')
    with np.load(npz) as z:
        arr = {k: np.asarray(z[k]) for k in z.files}
    return arr, rec


def stored_rule(size, rule, combo, n):
    p = os.path.join(ROOT_DIR, RESULTS_ROOT, f'{size}_xmask', f'Result_{rule}_{combo}.npy')
    if not os.path.exists(p):
        refuse(f'stored rule array {p} is missing')
    a = np.load(p)
    a = (a[:, 0] if a.ndim > 1 else a).astype(float)
    if len(a) < n:
        refuse(f'{p} has {len(a)} rows, the study needs {n}')
    return a[:n]


def stored_policy(size, combo, seeds, n):
    """[k, n] makespans of the released exact-mask runs, or None when no seed file
    exists; refused when only some exist."""
    paths = [os.path.join(ROOT_DIR, RESULTS_ROOT, f'{size}_xmask',
                          f'Result_greedy+{policy_model(s)}_{combo}.npy') for s in seeds]
    have = [os.path.exists(p) for p in paths]
    if not any(have):
        return None, []
    if not all(have):
        refuse(f'{size}/{combo}: policy arrays exist for some training seeds only '
               f'({[p for p, h in zip(paths, have) if not h][:2]} missing)')
    a = []
    for p in paths:
        x = np.load(p)
        x = (x[:, 0] if x.ndim > 1 else x).astype(float)
        if len(x) < n:
            refuse(f'{p} has {len(x)} rows, the study needs {n}')
        a.append(x[:n])
    return np.stack(a), [os.path.relpath(p, ROOT_DIR) for p in paths]


def ci(x):
    """Mean and 95% percentile bootstrap interval over instances, fresh
    default_rng(CI_SEED), B resamples."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(CI_SEED)
    idx = rng.integers(0, len(x), (B, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), n=int(len(x)))


def gap(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def trace_stats(arrs):
    """(b), (c), (d) of one trace arm over its runs (one array set per run); each value
    is the mean over the runs, with the range over the runs."""
    def agg(f):
        vals = [f(a) for a in arrs]
        return dict(value=float(np.mean(vals)), run_min=float(min(vals)),
                    run_max=float(max(vals)))

    def nanmean(a, k):
        return float(np.mean(a[k])) if np.all(np.isfinite(a[k])) else float('nan')
    out = dict(
        one_pair_share=agg(lambda a: 100.0 * a['n_one_pair'].sum() / a['n_decisions'].sum()),
        legal_mean=agg(lambda a: float(a['legal_counts'].mean())),
        legal_median=agg(lambda a: float(np.median(a['legal_counts']))),
        decisions=int(arrs[0]['n_decisions'].sum()),
        util=agg(lambda a: 100.0 * float(a['util'].mean())),
        ms_over_lb0=agg(lambda a: float(a['ms_over_lb0'].mean())),
        util_max_type=agg(lambda a: 100.0 * nanmean(a, 'util_max_type')),
        makespan=agg(lambda a: float(a['makespan'].mean())),
        lb0=float(arrs[0]['lb0'].mean()))
    if np.all(np.isfinite(arrs[0]['cov_end'])):
        out.update(
            cov_end=agg(lambda a: float(a['cov_end'].mean())),
            cov_end_min=float(min(a['cov_end'].min() for a in arrs)),
            cov_end_max=float(max(a['cov_end'].max() for a in arrs)),
            cov_time=agg(lambda a: 100.0 * float(a['cov_time'].mean())),
            last_window_end=float(arrs[0]['last_window_end'].mean()),
            horizon=float(arrs[0]['horizon'].mean()))
    return out


def ten_models(args):
    """(run names, training seeds) of the arm with ten schedules: the runs of
    --policy_models, or the released exact-mask runs of --policy_seeds."""
    if args.policy_models:
        return list(args.policy_models), [model_seed(m) for m in args.policy_models]
    return [policy_model(s) for s in args.policy_seeds], list(args.policy_seeds)


def ten_present(args, size, combo):
    """True when a cell of the arm exists for some run of this size and combination."""
    return any(all(os.path.exists(p) for p in cell_paths(args.root, size, combo, ten_label(m)))
               for m in ten_models(args)[0])


def ten_cells_present(root, size, combos, models):
    """{combination: the runs among `models` whose cell of the arm exists there}."""
    return {c: [m for m in models
                if all(os.path.exists(p) for p in cell_paths(root, size, c, ten_label(m)))]
            for c in combos}


def ten_form_of(present, models, selected):
    """What one size holds of the arm, from ten_cells_present: ('all', '') when every
    combination holds every run, ('selected', '') when every combination holds the
    selected run and no other, (None, '') when no combination holds a cell, and
    ('other', reason) for any other state."""
    if not any(present.values()):
        return None, ''
    if all(sorted(v) == sorted(models) for v in present.values()):
        return 'all', ''
    if selected is not None and all(v == [selected] for v in present.values()):
        return 'selected', ''
    counts = ', '.join(f'{c} {len(v)}' for c, v in present.items())
    return 'other', (f'cells of the ten schedules for some runs or combinations only, neither '
                     f'every one of the {len(models)} runs nor the selected run alone in every '
                     f'combination (runs per combination: {counts})')


def _exact_record(arm, combo, seed):
    """arm_registry.validation_record of an exact-mask run, whatever PACT_ENV says."""
    return REG.validation_record(arm, combo, seed,
                                 paths=(REG.vali_log(arm, combo, seed, 'exact'),
                                        REG.train_log(arm, combo, seed, 'exact')))


def ten_select(args):
    """The run the selection rule picks among the runs of the arm: arm_registry.
    select_seed over the exact-mask runs of --policy_seeds (lowest validation score,
    ties to the lower seed). With --policy_models, the first run named."""
    models, seeds = ten_models(args)
    if args.policy_models:
        return dict(model=models[0], seed=seeds[0], scores=None,
                    rule='the first run of --policy_models (development only)')
    try:
        s, sc = REG.select_seed('released', seed_list=tuple(args.policy_seeds),
                                record=_exact_record)
    except (REG.MissingResult, SystemExit) as e:
        refuse(f'the selected run among seeds {",".join(map(str, args.policy_seeds))} cannot '
               f'be determined: {str(e).splitlines()[0] if str(e) else type(e).__name__}')
    return dict(model=policy_model(s), seed=int(s),
                scores={str(k): {'score': float(v[0]), 'update': int(v[1])}
                        for k, v in sc.items()},
                rule='arm_registry.select_seed over the exact-mask runs of the released '
                     'recipe: the lowest validation score of the kept checkpoint, ties to the '
                     'lower seed')


def _stats_family():
    import stats_family as SF
    return SF


def ten_row(d):
    """One run: mean over instances, the 95% bootstrap interval of ci() and the
    direction of that interval (stats_family.direction)."""
    out = ci(d)
    out['direction'] = _stats_family().direction(out['lo'], out['hi'])
    return out


def ten_pool(d):
    """Pooled over the runs, d [k, n]: the crossed hierarchical bootstrap over runs and
    instances (stats_family.hier_boot, B resamples, fresh default_rng(HIER_SEED)) and the
    seed-level t interval; with one run the bootstrap of ci()."""
    SF = _stats_family()
    d = np.asarray(d, dtype=float)
    if d.shape[0] < 2:
        out = ci(d[0])
        out.update(k=1, scheme='bootstrap over instances (one run)')
    else:
        hb = SF.hier_boot(d, B=B, seed=HIER_SEED)
        t = SF.seed_t(d)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], n=int(d.shape[1]),
                   k=int(d.shape[0]), t_lo=t['lo'], t_hi=t['hi'], seed_means=t['seed_means'],
                   scheme='crossed hierarchical bootstrap over runs and instances')
    out['direction'] = SF.direction(out['lo'], out['hi'])
    return out


def ten_shares(pol_best, rnd_best):
    """Counts and percentages of instances (pairs of run and instance when pooled) on
    which the policy's best of ten is shorter than, equal to and longer than the random
    best of ten."""
    a, b = np.asarray(pol_best, float).ravel(), np.broadcast_to(rnd_best, np.shape(pol_best)).ravel()
    n = int(a.size)
    c = dict(shorter=int((a < b).sum()), equal=int((a == b).sum()), longer=int((a > b).sum()))
    return dict(n=n, **c, **{f'{k}_pct': 100.0 * v / n for k, v in c.items()})


def stored_seven(size, combo, n):
    """Per-instance best of the seven rules (the four classical rules and ECT, SSPT, LLF
    from the stored exact-mask arrays), or None when an array of ECT, SSPT or LLF is
    absent."""
    paths = [os.path.join(ROOT_DIR, RESULTS_ROOT, f'{size}_xmask', f'Result_{r}_{combo}.npy')
             for r in AWARE]
    if not all(os.path.exists(p) for p in paths):
        return None
    return np.min(np.stack([stored_rule(size, r, combo, n) for r in RULES + AWARE]), axis=0)


TEN_TEXT = {'randbest': 'best of the ten policy schedules against the best of the ten '
                        'random rollouts',
            'rule': 'best of the ten policy schedules against the best classical rule',
            'rule7': 'best of the ten policy schedules against the best of the seven rules',
            'greedy_rand': 'greedy schedule of the policy against the mean of the ten random '
                           'rollouts',
            'samp_rand': 'mean of the nine sampled policy schedules against the mean of the '
                         'ten random rollouts'}


def ten_cell(args, size, combo, names, r_mean, r_best, best, sel, form='all'):
    """QUANTITIES (e) of one size and combination, or None when the size holds no cell
    of the arm (form None). form 'all' reads every run of the arm, 'selected' the
    selected run only (rows, no pooled row). Refused when the cells exist for some runs
    only, hold another number of schedules or another run, or when a greedy schedule
    differs from a stored greedy result of its run."""
    if form is None:
        return None
    models, seeds = ten_models(args)
    if form == 'selected':
        models, seeds = [sel['model']], [sel['seed']]
    labels = [ten_label(m) for m in models]
    have = [all(os.path.exists(p) for p in cell_paths(args.root, size, combo, lb))
            for lb in labels]
    if not any(have):
        return None
    if not all(have):
        refuse(f'{size}/{combo}: cells of the ten schedules exist for some runs only '
               f'({[lb for lb, h in zip(labels, have) if not h][:2]} missing)')
    if sel is None or sel['model'] not in models:
        refuse(f'the selected run {sel and sel["model"]} is not among the runs of the arm')
    X, checks = [], {}
    for m, lb in zip(models, labels):
        arr, rec = read_cell(args.root, size, combo, lb, names)
        x = np.asarray(arr['makespan'], dtype=float)
        dz = rec.get('design', {})
        if x.shape != (len(names), N_DRAWS) or dz.get('draws') != N_DRAWS:
            refuse(f'{size}/{combo}/{lb} holds {x.shape[1] if x.ndim == 2 else x.shape} '
                   f'schedules per instance, the study needs {N_DRAWS}')
        if dz.get('model') != m or dz.get('kind') != 'polten':
            refuse(f'{size}/{combo}/{lb} records the run {dz.get("model")}, not {m}')
        if not np.all(np.isfinite(x)) or not np.all(x > 0):
            refuse(f'{size}/{combo}/{lb} holds a makespan that is not finite and positive')
        refs = greedy_references(args.root, size, combo, m, names, stop=refuse)
        for src, a in refs.items():
            if not np.array_equal(a, x[:, 0]):
                refuse(f'{size}/{combo}: the greedy schedule of {m} differs from {src} on '
                       f'{int((a != x[:, 0]).sum())} instances')
        checks[m] = sorted(refs) or 'no stored greedy array or trace cell of this run'
        X.append(x)
    X = np.stack(X)
    p_best, p_greedy, p_samp = X.min(axis=2), X[:, :, 0], X[:, :, 1:].mean(axis=2)
    best7 = stored_seven(size, combo, len(names))
    D = {'randbest': gap(p_best, r_best[None]), 'rule': gap(p_best, best[None]),
         'greedy_rand': gap(p_greedy, r_mean[None]), 'samp_rand': gap(p_samp, r_mean[None])}
    if best7 is not None:
        D['rule7'] = gap(p_best, best7[None])
    D = {q: D[q] for q in TEN_QUANTITIES if q in D}
    rows = {}
    for k, (m, s) in enumerate(zip(models, seeds)):
        rows[m] = dict(seed=int(s), **{q: ten_row(d[k]) for q, d in D.items()},
                       shares=ten_shares(p_best[k], r_best),
                       best_ms=float(p_best[k].mean()), greedy_ms=float(p_greedy[k].mean()),
                       sampled_ms=float(p_samp[k].mean()))
    pooled = form == 'all'
    return dict(form=form, runs=labels, models=models, seeds=[int(s) for s in seeds],
                selected=dict(model=sel['model'], seed=sel['seed']),
                best_ms=float(p_best.mean()), greedy_ms=float(p_greedy.mean()),
                sampled_ms=float(p_samp.mean()),
                best_rule7=('per-instance minimum of ' + ', '.join(RULES + AWARE)
                            if best7 is not None else
                            'not computed: a stored array of ECT, SSPT or LLF is absent'),
                rows=rows, pooled=({q: ten_pool(d) for q, d in D.items()} if pooled else {}),
                pooled_shares=ten_shares(p_best, r_best[None]) if pooled else None,
                seed_range=({q: dict(min=float(min(r[q]['mean'] for r in rows.values())),
                                     max=float(max(r[q]['mean'] for r in rows.values())),
                                     ahead=sum(r[q]['direction'] == 'ahead'
                                               for r in rows.values()),
                                     behind=sum(r[q]['direction'] == 'behind'
                                                for r in rows.values()))
                             for q in D} if pooled else {}),
                greedy_check=checks)


def ten_summary(S):
    """Per quantity over the combinations of a size: for the selected run, the
    combinations whose interval lies below zero (ahead) and above zero (behind) and the
    smallest and largest mean; the same for the pooled rows when the size holds every
    run."""
    out = {}
    for q in TEN_QUANTITIES:
        Ts = [(c, C['ten']) for c, C in S['combos'].items()
              if 'ten' in C and q in C['ten']['rows'][C['ten']['selected']['model']]]
        if not Ts:
            continue
        R = [T['rows'][T['selected']['model']][q] for _, T in Ts]
        Y = dict(combos=[c for c, _ in Ts], n=len(Ts),
                 selected_ahead=sum(x['direction'] == 'ahead' for x in R),
                 selected_behind=sum(x['direction'] == 'behind' for x in R),
                 selected_min=float(min(x['mean'] for x in R)),
                 selected_max=float(max(x['mean'] for x in R)))
        P = [T['pooled'][q] for _, T in Ts if q in T['pooled']]
        if P and len(P) == len(Ts):
            Y.update(pooled_ahead=sum(x['direction'] == 'ahead' for x in P),
                     pooled_behind=sum(x['direction'] == 'behind' for x in P),
                     pooled_min=float(min(x['mean'] for x in P)),
                     pooled_max=float(max(x['mean'] for x in P)))
        out[q] = Y
    return out


def ten_description(args, sel, forms):
    models, seeds = ten_models(args)
    return {'cells': 'polten_<model> of every run (or of the selected run, see runs_per_size), '
                     'size and combination',
            'runs': models, 'seeds': [int(s) for s in seeds], 'schedules': N_DRAWS,
            'selected': sel,
            'runs_per_size': {s: f for s, f in forms.items()},
            'runs_per_size_text': {s: TEN_RUN_TEXT[f] for s, f in forms.items()},
            'schedule_0': 'greedy (argmax at every decision)',
            'schedules_1_to_9': 'one complete schedule sampled from the policy\'s action '
                                'distribution under the exact mask per seed, '
                                'p97_policy_anytime.PolicyRoller.rollout_batch with a '
                                'batch of one',
            'stream': f'torch.manual_seed(SeedSequence([{RANDOM_SEED}, {TEN_TAG}, size code, '
                      f'instance index, combination index, training seed, k])'
                      f'.generate_state(1, uint64) // 2), k = 1..{N_DRAWS - 1}',
            'quantities': TEN_TEXT,
            'per_run': f'mean over instances and the 95% bootstrap of the other gaps '
                       f'({B} resamples, fresh default_rng({CI_SEED}))',
            'pooled': f'at a size that holds every run: crossed hierarchical bootstrap over '
                      f'runs and instances, stats_family.hier_boot ({B} resamples, fresh '
                      f'default_rng({HIER_SEED})); seed-level t interval as t_lo, t_hi; no '
                      f'pooled row at a size that holds the selected run only',
            'sign': '100 (policy - reference) / reference; negative means the policy is '
                    'shorter'}


def infer_outputs(out_root):
    return {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}


def guard_out(out):
    """--out must lie outside the repository's paper, results, data and checkpoint
    folders (it may lie outside the repository)."""
    rel = os.path.relpath(os.path.realpath(out), os.path.realpath(ROOT_DIR))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return
    first = rel.split(os.sep)[0]
    if rel == os.curdir or first in ('results', 'test_results', 'data', 'or_solution',
                                     'trained_network', 'train_log') \
            or first.startswith('paper'):
        refuse(f'--out {out} is the repository root or one of its paper, results, data '
               f'or checkpoint folders; give a folder outside them')


def infer(args):
    """Every quantity of the module doc from the cells of --root and the stored rule
    and policy arrays. Returns the result dictionary."""
    partial = bool(args.partial)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if args.policy_models and not partial:
        refuse('--policy_models (runs other than the released ones) needs --partial')
    if args.out is None:
        if partial:
            refuse('--partial writes only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        guard_out(out_root)
    paths = infer_outputs(out_root)
    for p in paths.values():                # a refusal leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if not partial and (args.sizes != list(SIZES) or args.combos != list(R8)):
        refuse('every size and combination is required; a subset needs --partial')
    arms = args.trace_arms
    res = {'generator': GENERATOR, 'partial': partial, 'mask': MASK,
           'rule_set': 'masked (common_utils.masked_rule_action)', 'draws': N_DRAWS,
           'random_stream': f'default_rng(SeedSequence([{RANDOM_SEED}, size code, '
                            f'instance index, draw])), size codes 10, 50, 80',
           'bootstrap': f'{B} resamples over instances, fresh default_rng({CI_SEED}) per '
                        f'interval',
           'gap': '100 (x - ref) / ref per instance, mean over instances; positive means '
                  'x is longer',
           'trace_arms': arms, 'policy_seeds': list(args.policy_seeds),
           'trace_seeds': list(args.trace_seeds), 'sizes': {}}
    have_policy = None
    ten_mode = getattr(args, 'ten_arm', 'auto')
    ten_want = getattr(args, 'ten_runs', None)       # None: either form at each size
    have_ten, ten_sel, ten_forms = None, None, {}
    for size in args.sizes:
        n_want = args.n or SIZES[size]['n']
        names = names_of(size, args.n)
        if len(names) != n_want:
            refuse(f'{size}: {len(names)} instances, expected {n_want}')
        S = {'n': len(names), 'modules': SIZES[size]['modules'], 'combos': {}}
        form = None
        if ten_mode != 'off':
            present = ten_cells_present(args.root, size, args.combos, ten_models(args)[0])
            if ten_sel is None and any(present.values()):
                ten_sel = ten_select(args)
            form, why = ten_form_of(present, ten_models(args)[0], ten_sel and ten_sel['model'])
            if form == 'other':
                refuse(f'{size}: {why}')
            if form is not None and ten_want is not None and form != ten_want[size]:
                refuse(f'{size} holds the ten schedules of {TEN_RUN_TEXT[form]}, --ten_runs '
                       f'asks for {TEN_RUN_TEXT[ten_want[size]]}')
            if form is not None:
                ten_forms[size] = form
        for combo in args.combos:
            rnd, rnd_rec = read_cell(args.root, size, combo, 'random', names)
            ms = rnd['makespan']
            if ms.shape != (len(names), N_DRAWS):
                refuse(f'{size}/{combo}/random has shape {ms.shape}')
            rules = {r: stored_rule(size, r, combo, len(names)) for r in RULES}
            best = np.min(np.stack([rules[r] for r in RULES]), axis=0)
            pol, pol_files = stored_policy(size, combo, args.policy_seeds, len(names))
            if have_policy is None:
                have_policy = pol is not None
            elif have_policy != (pol is not None):
                refuse(f'{size}/{combo}: policy arrays exist for some cells only')
            r_mean, r_best = ms.mean(axis=1), ms.min(axis=1)
            C = {'random_mean_ms': float(r_mean.mean()), 'random_best_ms': float(r_best.mean()),
                 'spt_ms': float(rules['SPT'].mean()), 'best_rule_ms': float(best.mean()),
                 'best_rule_instances': {r: int((np.argmin(np.stack([rules[q] for q in RULES]),
                                                           axis=0) == j).sum())
                                         for j, r in enumerate(RULES)},
                 'gaps': {'rand_spt': ci(gap(r_mean, rules['SPT'])),
                          'randbest_spt': ci(gap(r_best, rules['SPT'])),
                          'rand_rule': ci(gap(r_mean, best)),
                          'randbest_rule': ci(gap(r_best, best))},
                 'trace': {}}
            if pol is not None:
                p = pol.mean(axis=0)
                C['policy_ms'] = float(p.mean())
                C['policy_files'] = pol_files
                C['gaps']['rand_pol'] = ci(gap(r_mean, p))
                C['gaps']['randbest_pol'] = ci(gap(r_best, p))
            for arm in arms:
                labels = (['trace_SPT'] if arm == 'SPT' else
                          [trace_label(m) for m in (args.policy_models or
                                                    [policy_model(s) for s in args.trace_seeds])])
                arrs = [read_cell(args.root, size, combo, lb, names)[0] for lb in labels]
                if arm == 'SPT' and not np.array_equal(arrs[0]['makespan'], rules['SPT']):
                    d = arrs[0]['makespan'] - rules['SPT']
                    refuse(f'{size}/{combo}: the SPT rollouts differ from the stored array '
                           f'on {int((d != 0).sum())} instances; the pipelines diverge')
                if arm == 'policy' and pol is not None and args.policy_models is None:
                    for k, s in enumerate(args.trace_seeds):
                        if s not in args.policy_seeds:
                            continue
                        ref = pol[list(args.policy_seeds).index(s)]
                        if not np.array_equal(arrs[k]['makespan'], ref):
                            refuse(f'{size}/{combo}: the policy rollout of seed {s} differs '
                                   f'from its stored array on '
                                   f'{int((arrs[k]["makespan"] != ref).sum())} instances')
                C['trace'][arm] = trace_stats(arrs)
                C['trace'][arm]['runs'] = labels
            if ten_mode != 'off':
                T = ten_cell(args, size, combo, names, r_mean, r_best, best, ten_sel, form)
                if T is None and ten_mode == 'require':
                    refuse(f'{size}/{combo}: no cell of the ten schedules of the policy '
                           f'(--ten_arm require)')
                if have_ten is None:
                    have_ten = T is not None
                elif have_ten != (T is not None):
                    refuse(f'{size}/{combo}: cells of the ten schedules of the policy exist '
                           f'for some cells only')
                if T is not None:
                    C['ten'] = T
            S['combos'][combo] = C
        if have_ten:
            S['ten_runs'] = form
            S['ten_summary'] = ten_summary(S)
        res['sizes'][size] = S
    res['policy_comparator'] = bool(have_policy)
    if have_ten:
        res['ten_arm'] = ten_description(args, ten_sel, ten_forms)
    lines = macro_lines(res)
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    res['macro_files_checked'] = check_macro_names(names_m, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    write_json_atomic(paths['json'], jsonable(res))
    os.makedirs(os.path.dirname(paths['md']), exist_ok=True)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    for k in ('json', 'md'):
        print('wrote', paths[k])
    print(f'{len(names_m)} macros' + ('; PARTIAL' if partial else ''))
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def f2(x):
    x = float(x)
    return f'{0.0 if x == 0 else x:+.2f}'


def u1(x):
    return f'{float(x):.1f}'


def u2(x):
    return f'{float(x):.2f}'


ARM_WORDS = {'SPT': 'Spt', 'policy': 'Pol'}
GAP_WORDS = {'rand_spt': 'RandSpt', 'randbest_spt': 'RandBestSpt', 'rand_rule': 'RandRule',
             'randbest_rule': 'RandBestRule', 'rand_pol': 'RandPol',
             'randbest_pol': 'RandBestPol'}
GAP_TEXT = {'rand_spt': 'mean of the ten random rollouts against SPT',
            'randbest_spt': 'best of the ten random rollouts against SPT',
            'rand_rule': 'mean of the ten random rollouts against the best classical rule',
            'randbest_rule': 'best of the ten random rollouts against the best classical rule',
            'rand_pol': 'mean of the ten random rollouts against the released policy',
            'randbest_pol': 'best of the ten random rollouts against the released policy'}


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Controls for the non-delay mask under the exact mask: random choice among the '
         f'legal pairs ({N_DRAWS} rollouts per instance), legal pairs per decision, window '
         f'coverage and shop load; sizes 10, 50 and 80 modules on 25 stations.',
         '% Gaps in %: 100 (x - ref) / ref per instance, mean over instances, 95% bootstrap '
         'over instances; positive means x is longer. Best classical rule: per-instance '
         'minimum of FIFO, MOR, SPT and MWKR restricted to the legal pairs.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of sizes, combinations or instances; not for the '
                    'manuscript.')
    L.append(f'\\newcommand{{\\{PREFIX}DrawN}}{{{N_DRAWS}}}'
             f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}'
             f'  % {mark}random rollouts per instance; bootstrap resamples')
    if 'ten_arm' in res:
        X = res['ten_arm']
        L.append(f'\\newcommand{{\\{PREFIX}PtenDrawN}}{{{X["schedules"]}}}'
                 f'\\newcommand{{\\{PREFIX}PtenSeedN}}{{{len(X["runs"])}}}'
                 f'\\newcommand{{\\{PREFIX}PtenSelSeed}}{{{X["selected"]["seed"]}}}'
                 f'  % {mark}ten schedules of the policy: schedules per instance (the greedy '
                 f'one and nine sampled), trained runs of the arm (held at a size with every '
                 f'run), training seed of the selected run (lowest validation score). Gaps of '
                 f'this arm: 100 (policy - reference) / reference per instance, negative means '
                 f'the policy is shorter.')
    for size, S in res['sizes'].items():
        w = SIZES[size]['word']
        L.append(f'\\newcommand{{\\{PREFIX}{w}N}}{{{S["n"]}}}'
                 f'  % {mark}{size}: instances per combination')
        for combo, C in S['combos'].items():
            L.append(f'\\newcommand{{\\{PREFIX}MsRand{w}{combo}}}{{{u1(C["random_mean_ms"])}}}'
                     f'\\newcommand{{\\{PREFIX}MsRandBest{w}{combo}}}'
                     f'{{{u1(C["random_best_ms"])}}}'
                     f'\\newcommand{{\\{PREFIX}MsSpt{w}{combo}}}{{{u1(C["spt_ms"])}}}'
                     f'\\newcommand{{\\{PREFIX}MsRule{w}{combo}}}{{{u1(C["best_rule_ms"])}}}'
                     + (f'\\newcommand{{\\{PREFIX}MsPol{w}{combo}}}{{{u1(C["policy_ms"])}}}'
                        if 'policy_ms' in C else '')
                     + f'  % {mark}{size} {combo}: mean makespan (h) of the random rollouts, '
                       f'of their best, of SPT, of the best classical rule'
                     + (', of the released policy' if 'policy_ms' in C else ''))
            for key, g in C['gaps'].items():
                q = GAP_WORDS[key]
                L.append(f'\\newcommand{{\\{PREFIX}{q}{w}{combo}}}{{{f2(g["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{q}Lo{w}{combo}}}{{{f2(g["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{q}Hi{w}{combo}}}{{{f2(g["hi"])}}}'
                         f'  % {mark}{size} {combo}: {GAP_TEXT[key]}, %, mean and 95% '
                         f'interval over {g["n"]} instances')
            for arm, T in C['trace'].items():
                a = ARM_WORDS[arm]
                s = (f'\\newcommand{{\\{PREFIX}{a}OnePair{w}{combo}}}'
                     f'{{{u1(T["one_pair_share"]["value"])}}}'
                     f'\\newcommand{{\\{PREFIX}{a}LegalMean{w}{combo}}}'
                     f'{{{u1(T["legal_mean"]["value"])}}}'
                     f'\\newcommand{{\\{PREFIX}{a}LegalMedian{w}{combo}}}'
                     f'{{{u1(T["legal_median"]["value"])}}}'
                     f'\\newcommand{{\\{PREFIX}{a}Util{w}{combo}}}{{{u1(T["util"]["value"])}}}'
                     f'\\newcommand{{\\{PREFIX}{a}MsLb{w}{combo}}}'
                     f'{{{u2(T["ms_over_lb0"]["value"])}}}')
                txt = ('share of decisions with one legal pair (%), mean and median legal '
                       'pairs per decision, processing time over machines times makespan '
                       '(%), makespan over the estimate at the empty schedule')
                if 'cov_end' in T:
                    s += (f'\\newcommand{{\\{PREFIX}{a}CovEnd{w}{combo}}}'
                          f'{{{u2(T["cov_end"]["value"])}}}'
                          f'\\newcommand{{\\{PREFIX}{a}CovTime{w}{combo}}}'
                          f'{{{u1(T["cov_time"]["value"])}}}')
                    txt += (', last window end over the makespan, window time before the '
                            'makespan over machines times makespan (%)')
                L.append(s + f'  % {mark}{size} {combo}, {arm} rollout '
                             f'({len(T["runs"])} run(s), mean over runs): {txt}')
        for arm in res['trace_arms']:
            a = ARM_WORDS[arm]
            for q, key, combos, fmt in (('OnePair', 'one_pair_share', R8, u1),
                                        ('OnePairSetup', 'one_pair_share', SETUP_COMBOS, u1),
                                        ('Util', 'util', R8, u1),
                                        ('CovEnd', 'cov_end', WIN_COMBOS, u2)):
                vals = [S['combos'][c]['trace'][arm][key]['value'] for c in combos
                        if c in S['combos'] and key in S['combos'][c]['trace'][arm]]
                if vals:
                    L.append(f'\\newcommand{{\\{PREFIX}{a}{q}Min{w}}}{{{fmt(min(vals))}}}'
                             f'\\newcommand{{\\{PREFIX}{a}{q}Max{w}}}{{{fmt(max(vals))}}}'
                             f'  % {mark}{size}, {arm} rollout: smallest and largest '
                             f'{key} over {len(vals)} combinations ({",".join(c for c in combos if c in S["combos"])})')
        for key in ('rand_spt', 'rand_rule'):
            vals = [C['gaps'][key]['mean'] for C in S['combos'].values()]
            q = GAP_WORDS[key]
            L.append(f'\\newcommand{{\\{PREFIX}{q}Min{w}}}{{{f2(min(vals))}}}'
                     f'\\newcommand{{\\{PREFIX}{q}Max{w}}}{{{f2(max(vals))}}}'
                     f'  % {mark}{size}: smallest and largest mean gap, {GAP_TEXT[key]}, '
                     f'over {len(vals)} combinations')
        L += ten_macro_lines(size, S, mark)
    return L


TEN_WORDS = {'randbest': 'RandBest', 'rule': 'Rule', 'rule7': 'RuleSeven',
             'greedy_rand': 'GreedyRand', 'samp_rand': 'SampRand'}


def ten_macro_lines(size, S, mark):
    """Macros of QUANTITIES (e) for one size; none when the arm has no cell there.
    Names: Mctl Pten <quantity> <row> [Lo|Hi] <size word> <combination>, row Pool (the
    crossed bootstrap over runs and instances) or Sel (the selected run). A size that
    holds the selected run only has the Sel rows and no Pool or Seed macro."""
    w = SIZES[size]['word']
    L = []
    form = S.get('ten_runs')
    if form is None:
        return L
    tcs = [(c, C['ten']) for c, C in S['combos'].items() if 'ten' in C]
    T0 = tcs[0][1]
    L.append(f'\\newcommand{{\\{PREFIX}PtenRunN{w}}}{{{len(T0["runs"])}}}'
             f'  % {mark}{size}: trained runs whose ten schedules this size holds: '
             + ('every run of the arm, with the pooled rows' if form == 'all' else
                f'the selected run only (training seed {T0["selected"]["seed"]}), no pooled '
                f'row'))
    for combo, T in tcs:
        sel = T['rows'][T['selected']['model']]
        k, n = len(T['runs']), S['n']
        L.append(f'\\newcommand{{\\{PREFIX}MsPtenBest{w}{combo}}}{{{u1(T["best_ms"])}}}'
                 f'\\newcommand{{\\{PREFIX}MsPtenGreedy{w}{combo}}}{{{u1(T["greedy_ms"])}}}'
                 f'\\newcommand{{\\{PREFIX}MsPtenSamp{w}{combo}}}{{{u1(T["sampled_ms"])}}}'
                 f'  % {mark}{size} {combo}: mean makespan (h) over {k} trained '
                 f'run{"s" if k != 1 else ""} and {n} instances of the policy\'s best of ten '
                 f'schedules, of its greedy schedule, and of the mean of its nine sampled '
                 f'schedules')
        for q in [x for x in TEN_QUANTITIES if x in sel]:
            a, R = f'Pten{TEN_WORDS[q]}', sel[q]
            if form == 'all':
                P, G = T['pooled'][q], T['seed_range'][q]
                L.append(f'\\newcommand{{\\{PREFIX}{a}Pool{w}{combo}}}{{{f2(P["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}PoolLo{w}{combo}}}{{{f2(P["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}PoolHi{w}{combo}}}{{{f2(P["hi"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}Sel{w}{combo}}}{{{f2(R["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SelLo{w}{combo}}}{{{f2(R["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SelHi{w}{combo}}}{{{f2(R["hi"])}}}'
                         f'  % {mark}{size} {combo}: {TEN_TEXT[q]}, %, negative = policy '
                         f'shorter; mean and 95% interval pooled over {k} runs and {n} instances '
                         f'(crossed bootstrap, [{P["lo"]:+.4f}, {P["hi"]:+.4f}]), then for the '
                         f'selected run (bootstrap over instances, [{R["lo"]:+.4f}, '
                         f'{R["hi"]:+.4f}])')
                L.append(f'\\newcommand{{\\{PREFIX}{a}SeedMin{w}{combo}}}{{{f2(G["min"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SeedMax{w}{combo}}}{{{f2(G["max"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SeedAheadN{w}{combo}}}{{{G["ahead"]}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SeedBehindN{w}{combo}}}{{{G["behind"]}}}'
                         f'  % {mark}{size} {combo}: {TEN_TEXT[q]}: smallest and largest mean '
                         f'over the {k} runs, and the number of runs whose interval lies below '
                         f'zero (policy shorter) and above zero (policy longer)')
            else:
                L.append(f'\\newcommand{{\\{PREFIX}{a}Sel{w}{combo}}}{{{f2(R["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SelLo{w}{combo}}}{{{f2(R["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{a}SelHi{w}{combo}}}{{{f2(R["hi"])}}}'
                         f'  % {mark}{size} {combo}: {TEN_TEXT[q]}, %, negative = policy '
                         f'shorter; mean and 95% interval for the selected run, the only run '
                         f'this size holds (bootstrap over {n} instances, [{R["lo"]:+.4f}, '
                         f'{R["hi"]:+.4f}])')
        Ss = sel['shares']
        if form == 'all':
            Ps = T['pooled_shares']
            L.append(f'\\newcommand{{\\{PREFIX}PtenShorterPool{w}{combo}}}'
                     f'{{{u1(Ps["shorter_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenEqualPool{w}{combo}}}{{{u1(Ps["equal_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenLongerPool{w}{combo}}}'
                     f'{{{u1(Ps["longer_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenShorterSel{w}{combo}}}'
                     f'{{{u1(Ss["shorter_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenEqualSel{w}{combo}}}{{{u1(Ss["equal_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenLongerSel{w}{combo}}}'
                     f'{{{u1(Ss["longer_pct"])}}}'
                     f'  % {mark}{size} {combo}: share (%) of the {Ps["n"]} pairs of run and '
                     f'instance, then of the {Ss["n"]} instances of the selected run, on which '
                     f'the policy\'s best of ten is shorter than, equal to and longer than the '
                     f'random best of ten')
        else:
            L.append(f'\\newcommand{{\\{PREFIX}PtenShorterSel{w}{combo}}}'
                     f'{{{u1(Ss["shorter_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenEqualSel{w}{combo}}}{{{u1(Ss["equal_pct"])}}}'
                     f'\\newcommand{{\\{PREFIX}PtenLongerSel{w}{combo}}}'
                     f'{{{u1(Ss["longer_pct"])}}}'
                     f'  % {mark}{size} {combo}: share (%) of the {Ss["n"]} instances of the '
                     f'selected run on which the policy\'s best of ten is shorter than, equal '
                     f'to and longer than the random best of ten')
    for q, Y in S.get('ten_summary', {}).items():
        a = f'Pten{TEN_WORDS[q]}'
        sel_part = (f'\\newcommand{{\\{PREFIX}{a}SelAheadN{w}}}{{{Y["selected_ahead"]}}}'
                    f'\\newcommand{{\\{PREFIX}{a}SelBehindN{w}}}{{{Y["selected_behind"]}}}'
                    f'\\newcommand{{\\{PREFIX}{a}SelMin{w}}}{{{f2(Y["selected_min"])}}}'
                    f'\\newcommand{{\\{PREFIX}{a}SelMax{w}}}{{{f2(Y["selected_max"])}}}'
                    f'\\newcommand{{\\{PREFIX}{a}CombN{w}}}{{{Y["n"]}}}')
        sel_txt = ('for the selected run the number of combinations whose interval lies '
                   'below zero (policy shorter) and above zero, and the smallest and largest '
                   'mean (%); the number of combinations')
        if 'pooled_ahead' in Y:
            L.append(f'\\newcommand{{\\{PREFIX}{a}PoolAheadN{w}}}{{{Y["pooled_ahead"]}}}'
                     f'\\newcommand{{\\{PREFIX}{a}PoolBehindN{w}}}{{{Y["pooled_behind"]}}}'
                     f'\\newcommand{{\\{PREFIX}{a}PoolMin{w}}}{{{f2(Y["pooled_min"])}}}'
                     f'\\newcommand{{\\{PREFIX}{a}PoolMax{w}}}{{{f2(Y["pooled_max"])}}}'
                     + sel_part +
                     f'  % {mark}{size}: {TEN_TEXT[q]}: of the {Y["n"]} combinations '
                     f'({",".join(Y["combos"])}), the number whose pooled interval lies below '
                     f'zero (policy shorter) and above zero, the smallest and largest pooled '
                     f'mean (%); ' + sel_txt)
        else:
            L.append(sel_part
                     + f'  % {mark}{size}: {TEN_TEXT[q]}: of the {Y["n"]} combinations '
                       f'({",".join(Y["combos"])}), ' + sel_txt
                     + ' (the selected run only, no pooled row)')
    return L


def check_macro_names(names, out_paths):
    """Every name is the prefix plus letters, unique, and absent from every other macro
    file of the repository's paper folders."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    own = {os.path.abspath(out_paths['macros']),
           os.path.abspath(os.path.join(ROOT_DIR, OUT_FILES['macros']))}
    files = sorted(set(glob.glob(os.path.join(ROOT_DIR, 'paper*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'paper*', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'sections_new', '*.tex'))))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, ROOT_DIR)}: '
                   f'{sorted(clash)[:5]}')
    return len(files)


def markdown(res):
    L = [f'# Controls for the non-delay mask ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of sizes, combinations or instances. Not for the manuscript.',
              '']
    L += [f'Exact mask (first_mask and mask_ties exact). Random control: {N_DRAWS} rollouts '
          f'per instance, one legal pair drawn uniformly at every decision, stream '
          f'{res["random_stream"]}. Gaps: {res["gap"]}; 95% interval: {res["bootstrap"]}. '
          'Best rule: per-instance minimum of FIFO, MOR, SPT and MWKR restricted to the legal '
          'pairs (stored arrays test_results/FAMILY/<size>_xmask). Trace arms: '
          f'{", ".join(res["trace_arms"])}; legal pairs counted before every decision, over '
          'all decisions of all instances. Coverage: last window end over the makespan, and '
          'window time before the makespan over machines times makespan. Load: processing '
          'time of the chosen assignments over machines times makespan, and the makespan '
          'over the estimate at the empty schedule.', '']
    if not res['policy_comparator']:
        L += ['The released policy has no exact-mask arrays yet, so the random control is '
              'compared with the rules only.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for size, S in res['sizes'].items():
        L += [f'## {size} ({S["modules"]} modules, {S["n"]} instances)', '',
              '| combination | random mean (h) | random best (h) | SPT (h) | best rule (h) | '
              'random mean vs SPT (%) | random best vs SPT (%) | random mean vs best rule (%) '
              '| random best vs best rule (%) |' +
              (' random mean vs policy (%) |' if res['policy_comparator'] else ''),
              '|' + '---|' * (9 + int(res['policy_comparator']))]
        for combo, C in S['combos'].items():
            G = C['gaps']
            L.append(f'| {combo} | {u1(C["random_mean_ms"])} | {u1(C["random_best_ms"])} | '
                     f'{u1(C["spt_ms"])} | {u1(C["best_rule_ms"])} | {g(G["rand_spt"])} | '
                     f'{g(G["randbest_spt"])} | {g(G["rand_rule"])} | {g(G["randbest_rule"])} |'
                     + (f' {g(G["rand_pol"])} |' if 'rand_pol' in G else ''))
        for arm in res['trace_arms']:
            L += ['', f'{arm} rollout:', '',
                  '| combination | decisions | one legal pair (%) | legal pairs, mean | '
                  'legal pairs, median | utilization (%) | busiest type (%) | makespan / '
                  'estimate | last window end / makespan [min, max] | window time (%) |',
                  '|' + '---|' * 10]
            for combo, C in S['combos'].items():
                T = C['trace'][arm]
                cov = (f'{u2(T["cov_end"]["value"])} [{u2(T["cov_end_min"])}, '
                       f'{u2(T["cov_end_max"])}] | {u1(T["cov_time"]["value"])}'
                       if 'cov_end' in T else '-- | --')
                L.append(f'| {combo} | {T["decisions"]} | {u1(T["one_pair_share"]["value"])} | '
                         f'{u1(T["legal_mean"]["value"])} | {u1(T["legal_median"]["value"])} | '
                         f'{u1(T["util"]["value"])} | {u1(T["util_max_type"]["value"])} | '
                         f'{u2(T["ms_over_lb0"]["value"])} | {cov} |')
        L += ten_markdown(S, g)
        L.append('')
    return L


def ten_markdown(S, g):
    """The table of QUANTITIES (e) of one size; empty when the arm has no cell there."""
    tcs = [(c, C['ten']) for c, C in S['combos'].items() if 'ten' in C]
    if not tcs:
        return []
    sel = tcs[0][1]['selected']
    if S.get('ten_runs') == 'selected':
        L = ['', f'Ten schedules of the policy (greedy and nine sampled) against ten random '
                 f'rollouts. This size holds the selected run only: seed {sel["seed"]} '
                 f'({sel["model"]}); no pooled row. Gaps in %, 100 (policy - reference) / '
                 f'reference, mean and 95% bootstrap interval over instances, negative = '
                 f'policy shorter.',
             '',
             '| combination | policy best of ten (h) | random best of ten (h) | best vs random '
             'best | shorter / equal / longer (%) | best vs best rule | best vs best of seven '
             '| greedy vs random mean | sampled mean vs random mean |',
             '|' + '---|' * 9]
        for combo, T in tcs:
            C = S['combos'][combo]
            R = T['rows'][T['selected']['model']]
            sh = R['shares']
            L.append(f'| {combo} | {u1(T["best_ms"])} | {u1(C["random_best_ms"])} | '
                     f'{g(R["randbest"])} | {u1(sh["shorter_pct"])} / {u1(sh["equal_pct"])} / '
                     f'{u1(sh["longer_pct"])} | {g(R["rule"])} | '
                     f'{g(R["rule7"]) if "rule7" in R else "--"} | {g(R["greedy_rand"])} | '
                     f'{g(R["samp_rand"])} |')
        return L
    L = ['', f'Ten schedules of the policy (greedy and nine sampled) against ten random '
             f'rollouts. This size holds every run of the arm: {len(tcs[0][1]["runs"])} '
             f'trained runs; pooled: crossed bootstrap over runs and instances; selected run: '
             f'seed {sel["seed"]} ({sel["model"]}). Gaps in %, 100 (policy - reference) / '
             f'reference, negative = policy shorter.',
         '',
         '| combination | policy best of ten (h) | random best of ten (h) | best vs random '
         'best, pooled | best vs random best, selected | runs below / above zero | shorter / '
         'equal / longer, pooled (%) | best vs best rule, pooled | best vs best of seven, '
         'pooled | greedy vs random mean, pooled | sampled mean vs random mean, pooled |',
         '|' + '---|' * 11]
    for combo, T in tcs:
        C = S['combos'][combo]
        P, R = T['pooled'], T['rows'][T['selected']['model']]
        sh, G = T['pooled_shares'], T['seed_range']['randbest']
        L.append(f'| {combo} | {u1(T["best_ms"])} | {u1(C["random_best_ms"])} | '
                 f'{g(P["randbest"])} | {g(R["randbest"])} | {G["ahead"]} / {G["behind"]} | '
                 f'{u1(sh["shorter_pct"])} / {u1(sh["equal_pct"])} / {u1(sh["longer_pct"])} | '
                 f'{g(P["rule"])} | {g(P["rule7"]) if "rule7" in P else "--"} | '
                 f'{g(P["greedy_rand"])} | {g(P["samp_rand"])} |')
    return L


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['run', 'infer', 'plan'])
    ap.add_argument('--root', default=DEFAULT_ROOT, help='folder of the cell arrays')
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--combos', default=','.join(R8))
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--trace_arm', default='SPT', choices=['SPT', 'policy', 'polten'],
                    help='run: SPT computes the random control and the SPT trace; policy '
                         'the greedy trace of each released exact-mask run; polten the ten '
                         'schedules of each released exact-mask run (QUANTITIES (e))')
    ap.add_argument('--draws', type=int, default=N_DRAWS,
                    help='run --trace_arm polten: schedules per instance (smoke only; the '
                         'design records the value and infer needs 10)')
    ap.add_argument('--ten_arm', default='auto', choices=['auto', 'require', 'off'],
                    help='infer: the ten schedules of the policy, reported where their '
                         'cells exist (auto), required for every cell (require), or left '
                         'out (off)')
    ap.add_argument('--ten_runs', default=None,
                    help='runs of the ten schedules per size: all (every run of '
                         '--policy_seeds), selected (the selected run only), or '
                         'size=all|selected for each size, separated by commas; plan and run '
                         f'default to {DEFAULT_TEN_RUNS}; infer accepts either form at each '
                         'size unless this option names one')
    ap.add_argument('--trace_arms', default='SPT',
                    help='infer: trace arms to report, SPT and/or policy')
    ap.add_argument('--policy_seeds', default='301-305',
                    help='run: seeds of the policy trace; infer: seeds of the released '
                         'policy that the random control is compared with')
    ap.add_argument('--trace_seeds', default='',
                    help='infer: seeds of the policy trace (default: --policy_seeds)')
    ap.add_argument('--policy_models', default='',
                    help='full model names in place of the released runs (smoke checks of '
                         'the policy path only; infer then skips the stored-array check)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true',
                    help='infer: a subset of sizes, combinations or instances, under --out')
    ap.add_argument('--out', default=None, help='infer: root of the outputs')
    args = ap.parse_args(_ARGV)
    check_mask_options()
    args.sizes = parse_list(args.sizes, SIZES, 'sizes')
    args.combos = parse_list(args.combos, R8, 'combinations')
    args.trace_arms = parse_list(args.trace_arms, ('SPT', 'policy'), 'trace arms')
    args.policy_seeds = list(REG.parse_seeds(args.policy_seeds))
    args.trace_seeds = (list(REG.parse_seeds(args.trace_seeds)) if args.trace_seeds
                        else list(args.policy_seeds))
    args.policy_models = [x for x in args.policy_models.split(',') if x] or None
    if args.ten_runs is None and args.phase in ('plan', 'run'):
        args.ten_runs = DEFAULT_TEN_RUNS
    args.ten_runs = parse_ten_runs(args.ten_runs) if args.ten_runs is not None else None
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    {'run': run, 'infer': infer, 'plan': plan}[args.phase](args)


if __name__ == '__main__':
    main()
