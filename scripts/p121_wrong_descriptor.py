r"""P121: schedules of the policy under a wrong constraint descriptor, in the exact environment.

A descriptor that omits a rule in force (or names a rule that is not in force) configures the
environment with the wrong rules. This script measures what the policy's schedules then do
under the true rules, which validator call catches them, and which configuration check of
constraint_family/checks.py can see the fault. It ports part (ii) of scripts/p40_misconfig.py
to the exact environment and the policy of the paper.

LAUNCH NOTE (the four questions, answered before the rollouts)

1. Purpose. The configuration checks C1 to C7 and the validator are the safeguards of a
   descriptor change. The numbers say, for the two (rule combination in force, combination
   configured) pairs of p40, how far the makespan moves, how many schedules the validator
   rejects against the true descriptor and against the configured one, and which checks
   fail on the wrong descriptor and on six malformed descriptors. They land in the supplement
   section on the validator and the checks and in one sentence of the main text.
2. Expected result. A descriptor that omits a rule in force yields schedules that violate
   the true rules on every instance. The validator rejects them when it is given the true
   descriptor and accepts them when it is given the configured one, and every computed
   check passes on the wrong descriptor, so only the reconciliation with the plant records
   (C7) can find the fault. If some schedules pass under the true rules, the paper reports
   the share and the rule that the schedules happened to respect.
3. Contamination. Checkpoints, descriptors, base instances and the stored arrays are read
   only. Rollout records go to a folder of their own, arm_registry.res_dir('10x25_wrongdesc')
   (test_results/FAMILY/10x25_wrongdesc_xmask in the exact environment), which no other
   script reads. A record is written through a temporary file and a link that fails when
   the record exists, so a stored record is never overwritten, and a record of another
   design is refused. Rollouts are deterministic greedy rollouts on one torch thread, so
   they are not timing measurements and may share a core; their recorded times are planning
   figures only.
4. Data accuracy. The 100 test instances of data/FAMILY/10x25 (base instances
   data/PPVC/10x25+ppvc-mixed) and their stored descriptors of the four combinations LSW, N,
   SW and LS. Every rollout must reproduce the stored array of the same run and combination,
   test_results/FAMILY/10x25_xmask/Result_greedy+10x25+family+xmask-v1-s<seed>_<R>.npy, bit
   for bit, or the record is not written. The checkpoint's parameter count (32786), its mask
   options (first_mask exact, mask_ties exact, as arm_registry.MASK_OPTIONS of the exact
   environment) and its configuration (non-delay mask, no extra pair channels, no network
   input of the descriptor vector) are asserted before the first rollout.

THE ROLLOUTS (subcommand rollout)
  Policy xmask-v1-s<seed> (arm_registry released arm, seeds PACT_SEEDS, default 301 to 305),
  loaded by eval_family.load_model, evaluated as eval_family.py evaluates it with the options
  --device cpu --torch_threads 1 --first_mask exact --mask_ties exact: one instance at a
  time, FJSPEnvFamily(family_mode=True, mask_mode and pair_extra of the run, first_mask and
  mask_ties exact), the shaping envelope of the run (resolve_delta_mode), the descriptor
  vector of the environment's descriptor attached, greedy decisions. One record per training
  seed and combination holds, per instance, the makespan, the machine and the completion time
  of every operation and the decision sequence.

THE CASES (p40's pairs and cases, read from its source; see load_p40)
  pairs   (LSW in force, N configured): every rule omitted; (SW in force, LS configured): the
          window rule omitted and a lag rule added.
  A       environment and inputs from the true descriptor (the record of the true
          combination).
  C       environment and inputs from the configured descriptor (the record of the configured
          combination of the same base instance): what a plant with the wrong descriptor gets.
  B, D    p40 gives the network the descriptor vector of one descriptor while the
          environment simulates the other. The policy of the paper does not read the vector
          (use_film False and token_to_heads False in its configuration; asserted). It reads
          the descriptor only through the environment: the non-delay mask, the lag channels,
          the minimum-setup term of the makespan-estimate channel, the setup and window
          channels of the machines, and every time-based channel, all computed from the
          simulated state of the one descriptor the environment holds. So p40's B gives the
          schedules of A and its D those of C, decision for decision; the rollouts check this
          by evaluating the network with the other descriptor's vector (every decision of the
          first VEC_FULL instances of each record, the first decision of the others) and
          requiring identical outputs. Inputs computed from one descriptor while the
          environment simulates another would need an environment that no plant can
          configure. B and D are therefore not separate faults for this policy, and only A
          and C are run.

PER PAIR, CASE AND TRAINING SEED (and pooled over the seeds)
  makespan    mean over the 100 instances (arm_registry.load / pooled: the per-instance mean
              over the seeds), and its change against case A, p88_family_inference.rel of the
              two means, 100 (case - A) / A, as p40 computes it.
  rejected    the number of schedules that check C6 (checks.check_schedule_validation, the
              two independent validator modules) rejects, against the TRUE descriptor and
              against the CONFIGURED descriptor.
  violations  against the true descriptor, the messages of reference_sim.validate_schedule
              bucketed by p40's classify (lag, setup, window, other), summed over the
              schedules, with the number of schedules that hold at least one of each; and the
              messages of schedule_validator.validate_schedule by their prefix. One message
              kind is counted apart (presence, function buckets): a setup block that starts
              before its job is present ('job j: setup of op b starts before pred ct+lag'),
              which classify would count as a lag violation because it contains 'pred
              ct+lag'. The lag bucket then counts processing starts before the predecessor's
              completion plus its lag, as the precedence check of schedule_validator does; the
              lag count of classify on all messages is kept in the JSON (lag_p40_classify).

THE CHECKS ON THE WRONG DESCRIPTOR (part 2)
  checks.run_all on each configured descriptor of the 100 instances (op_pt and job_length of
  the base instance, finite_downstream_buffers=False as declared for the factory), and C6 on
  the case-C schedules against the configured descriptor, with the true descriptors as the
  control. Each check is named with its function, file and line, and with its basis: a
  computation, a computation on a plant declaration, or a sign-off.

FAULT INJECTION (part 3)
  checks.run_all on six malformed copies of the LSW descriptor of instance_000: a negative lag
  (the first nonzero lag negated), a setup matrix with one negative entry (setup[0][1]
  negated), two overlapping windows on one machine (a copy of the first window of the first
  machine that has one, shifted by 1 h), a window whose end precedes its start (that window
  reversed), the setup matrix doubled (every entry at twice its value, so the entries lie at
  twice the trained range), and an asymmetric setup matrix (setup[1][0] moved to another value
  inside the trained cross-class range).

OUTPUTS (arm_registry.out under PACT_OUT, default out/exact)
  manuscript/macros_wrongdesc.tex  macro_io.emit; names Wrd<quantity>, letters only, checked
                                  against every macro defined in a .tex file under manuscript/
  results/wrong_descriptor.json   every number, the case definitions, the functions called
  results/wrong_descriptor.md     the tables
  A refusal removes all three.

CHECK (--check, read only)
  exit 0 when every record of the active seeds and the four combinations exists with the
  design of this run (model, checkpoint hash, mask options, instances) and reproduces the
  stored array; lists the inputs of the rollouts and the missing records otherwise (exit 1).

DRY MODE (--dry [DIR])
  The analysis on synthetic records in DIR (default a temporary folder, removed at the end):
  the first DRY_N instances, schedules built by reference_sim.simulate from a fixed dispatch
  order under each combination's stored descriptor, synthetic stored arrays equal to them;
  the known answers (no rejection in case A, none against the configured descriptor, every
  LSW schedule built under N rejected) are asserted.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 python scripts/p121_wrong_descriptor.py rollout \
           [--seeds 301] [--combos LSW,N] [--budget-min 8]
       PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p121_wrong_descriptor.py
       ... python scripts/p121_wrong_descriptor.py --check
       python scripts/p121_wrong_descriptor.py --dry [DIR]
       Run every command with CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1.
"""
import argparse
import ast
import copy
import glob
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402  (removes the --pact-* options from sys.argv)
import p88_family_inference as F88  # noqa: E402
from macro_io import emit  # noqa: E402
from constraint_family import checks as CHK  # noqa: E402
from constraint_family import reference_sim as RS  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402

GENERATOR = 'scripts/p121_wrong_descriptor.py'
PREFIX = 'Wrd'
OUT_FILES = {'macros': 'manuscript/macros_wrongdesc.tex', 'json': 'results/wrong_descriptor.json',
             'md': 'results/wrong_descriptor.md'}
DEFAULT_OUT = 'out/exact'
ENV = 'exact'
DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
FOLDER = '10x25_wrongdesc'          # arm_registry.res_dir: <results>/10x25_wrongdesc_xmask
STORED = '10x25'                    # arm_registry.res_dir: <results>/10x25_xmask
N_TEST = 100
PARAMS = 32786                      # parameters of the policy
VEC_FULL = 5                        # instances per record checked at every decision
DRY_N = 10
ARM = 'released'
P40 = os.path.join(ROOT, 'scripts', 'p40_misconfig.py')
P40_FUNCTIONS = ('classify',)
P40_CONSTANTS = ('PAIRS', 'CASES')
CASES_RUN = ('A', 'C')
MASK_NAMES = ('first_mask', 'mask_ties')
PARAMS_ARGV = ['--device', 'cpu', '--device_id', '']
SV_PREFIXES = ('precedence', 'overlap', 'bound', 'compat', 'start', 'release')
CHECK_IDS = ('C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7')
CHECK_FUNCS = {'C1': 'check_schema_and_values', 'C2': 'check_setup_convention',
               'C3': 'check_family_membership', 'C4': 'check_magnitude_range',
               'C5': 'check_applicability', 'C6': 'check_schedule_validation',
               'C7': 'check_source_reconciliation'}
BASIS = {
    'C1': 'computation: shapes, types and values of the descriptor fields',
    'C2': 'computation on a plant declaration: it reads the declared setup convention; '
          'whether the declaration is true cannot be computed',
    'C3': 'computation: the active rule types of the descriptor against the registry',
    'C4': 'computation: rule magnitudes against the training ranges (warns, never fails)',
    'C5': 'plant declaration: finite downstream buffers, passed as an argument; the code '
          'checks only that it is declared and false',
    'C6': 'computation: the schedule against the descriptor it is given',
    'C7': 'sign-off: an engineer compares the descriptor with the plant source records; the '
          'code records the fingerprint and returns MANUAL'}
FAULTS = (('neglag', 'a negative lag', 'NegLag'),
          ('negsetup', 'a setup matrix with a negative entry', 'NegSetup'),
          ('overlap', 'two overlapping windows on one machine', 'Overlap'),
          ('reversed', 'a window whose end precedes its start', 'Reversed'),
          ('twice', 'the setup matrix at twice the trained range', 'Twice'),
          ('asym', 'an asymmetric setup matrix', 'Asym'))
CASE_TEXT = {
    'A': 'environment and inputs from the true descriptor',
    'B': "p40: environment from the true descriptor, descriptor vector from the configured "
         "one. The policy does not read the descriptor vector, so B gives the schedules of A "
         "decision for decision (checked in the rollouts); inputs computed from the "
         "configured descriptor while the environment simulates the true one would need an "
         "environment that no plant can configure. Not run.",
    'C': 'environment and inputs from the configured descriptor: what a plant with the wrong '
         'descriptor gets',
    'D': "p40: environment from the configured descriptor, descriptor vector from the true "
         "one. For the same reason D gives the schedules of C. Not run."}


class Refused(Exception):
    """A setting, an input or a record that this script does not accept."""


class Incomplete(Refused):
    """A record that is absent."""


# ----------------------------------------------------------------- p40's definitions
def load_p40():
    """{name: object} of p40's PAIRS, CASES and classify, from its source; nothing else of
    p40 is executed (its module level changes sys.argv, the thread count and the device)."""
    tree = ast.parse(open(P40).read(), P40)
    keep, seen = [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in P40_FUNCTIONS:
            keep.append(node)
            seen.add(node.name)
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & set(P40_CONSTANTS):
                keep.append(node)
                seen |= names & set(P40_CONSTANTS)
    missing = set(P40_FUNCTIONS + P40_CONSTANTS) - seen
    if missing:
        raise Refused(f'{P40} defines no {sorted(missing)}')
    ns = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), P40, 'exec'), ns)
    return {k: ns[k] for k in P40_FUNCTIONS + P40_CONSTANTS}


def pairs_of(p40):
    """[(true combination, configured combination, label)] of p40, checked."""
    pairs = [tuple(p) for p in p40['PAIRS']]
    for t, f, lab in pairs:
        if t not in REG.COMBOS or f not in REG.COMBOS or not re.fullmatch(r'[A-Za-z]+', lab):
            raise Refused(f'p40 pair {(t, f, lab)} is not two combinations and a label')
    if not set(CASES_RUN) <= set(p40['CASES']):
        raise Refused(f'p40 cases {p40["CASES"]} lack {CASES_RUN}')
    return pairs


def combos_of(pairs):
    out = []
    for t, f, _ in pairs:
        out += [c for c in (t, f) if c not in out]
    return out


def partner_of(pairs):
    """{combination: the other combination of its pair} (each combination in one pair)."""
    out = {}
    for t, f, _ in pairs:
        for a, b in ((t, f), (f, t)):
            if a in out:
                raise Refused(f'combination {a} belongs to two pairs')
            out[a] = b
    return out


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the rollouts and the analysis use the {ENV} environment; the active '
                      f'environment is {REG.env()} (set PACT_ENV={ENV})')
    return active_seeds()


def active_seeds():
    """PACT_SEEDS, or every planned seed of the exact environment (301 to 305)."""
    if os.environ.get('PACT_SEEDS'):
        return REG.seeds()
    return tuple(REG.PLANNED_SEEDS[ENV])


def environment_mask_options():
    o = list(REG.MASK_OPTIONS[ENV])
    opts = {o[i].lstrip('-'): o[i + 1] for i in range(0, len(o), 2)}
    if set(opts) != set(MASK_NAMES):
        raise Refused(f'arm_registry.MASK_OPTIONS of {ENV} is {o}, want both of {MASK_NAMES}')
    return opts


def instance_names(combo, n=None):
    names = sorted(os.path.basename(p)[:-len('.desc.json')]
                   for p in glob.glob(os.path.join(ROOT, DATA, 'desc', combo, '*.desc.json')))
    return names[:n] if n else names


def read_desc(combo, name):
    with open(os.path.join(ROOT, DATA, 'desc', combo, f'{name}.desc.json')) as f:
        return ConstraintDescriptor.from_json(json.load(f), setup_convention='attached')


def params_argv():
    """The options params.py parses here: those of the paper's evaluations (CPU, the mask
    options of the exact environment)."""
    m = environment_mask_options()
    return PARAMS_ARGV + ['--first_mask', m['first_mask'], '--mask_ties', m['mask_ties']]


def ensure_params():
    """params.configs, parsed from params_argv() and not from this script's command line
    (the instance loader imports data_utils, which imports params.py)."""
    if 'params' not in sys.modules:
        saved = sys.argv
        sys.argv = [saved[0]] + params_argv()
        try:
            import params  # noqa: F401
        finally:
            sys.argv = saved
    from params import configs
    if configs.device != 'cpu':
        raise Refused(f'params.py was parsed with device {configs.device}; this script runs on '
                      'the CPU')
    return configs


def read_instance(name):
    ensure_params()
    from constraint_family.public_adapter import load_base_instance
    return load_base_instance(os.path.join(ROOT, BASE, name))


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def light_design(seed):
    """The design every record of this seed carries, from the configuration and the
    checkpoint file (no network is loaded)."""
    model = REG.PREFIX + REG.model(ARM, None, seed, ENV)
    ck = REG.checkpoint(ARM, None, seed, ENV)
    cfg = REG.config_path(ARM, None, seed, ENV)
    for p in (ck, cfg):
        if not os.path.exists(os.path.join(ROOT, p)):
            raise Incomplete(f'missing {p}')
    snap = json.load(open(os.path.join(ROOT, cfg)))
    want = environment_mask_options()
    rec = {k: snap.get(k) for k in MASK_NAMES}
    if rec != want:
        raise Refused(f'{cfg} records the mask options {rec}, the stored arrays of the {ENV} '
                      f'environment were evaluated with {want}')
    shape = {k: snap.get(k) for k in ('mask_mode', 'pair_extra', 'use_film', 'token_to_heads')}
    if shape != {'mask_mode': 'nondelay', 'pair_extra': 'none', 'use_film': False,
                 'token_to_heads': False}:
        raise Refused(f'{cfg} records {shape}; the policy of the paper has the non-delay mask, '
                      'no extra pair channels and no network input of the descriptor vector')
    return dict(environment=ENV, data=DATA, base=BASE, model=model, checkpoint=ck,
                checkpoint_sha256=sha256(os.path.join(ROOT, ck)), config=cfg,
                mask_options=want, parameter_count=PARAMS, torch_threads=1, device='cpu')


def dry_design(seed):
    model = REG.PREFIX + REG.model(ARM, None, seed, ENV)
    return dict(environment=ENV, data=DATA, base=BASE, model=model, checkpoint='dry',
                checkpoint_sha256='dry', config='dry', mask_options=environment_mask_options(),
                parameter_count=PARAMS, torch_threads=1, device='cpu')


# ----------------------------------------------------------------- records
def record_path(seed, combo):
    tag = REG.tag(ARM, None, seed, environment=ENV)
    return os.path.join(REG.res_dir(FOLDER), f'record_{tag}_{combo}.json')


def array_path(seed, combo):
    return REG.result_path(ARM, None, seed, FOLDER, combo)


def stored_array(seed, combo, n):
    """The stored makespans of the run on the combination (10x25_xmask), first n rows; the
    file must hold one row per test instance."""
    a = REG.load(ARM, None, seed, STORED, combo)
    if len(a) < n or (n == N_TEST and len(a) != N_TEST):
        raise Refused(f'{REG.result_path(ARM, None, seed, STORED, combo)} holds {len(a)} rows')
    return a[:n]


def link_new(tmp, path):
    """Move tmp to path; refuses when path exists (a stored record is never overwritten)."""
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise Refused(f'{path} exists; a stored record is never overwritten')
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_record(seed, combo, rec):
    """Write the makespan array, then the record (the record marks the pair complete)."""
    js, npy = record_path(seed, combo), array_path(seed, combo)
    for p in (js, npy):
        if os.path.exists(p):
            raise Refused(f'{p} exists; a stored record is never overwritten (remove a partial '
                          'pair by hand)')
    os.makedirs(os.path.dirname(js), exist_ok=True)
    ms = np.array([r['makespan'] for r in rec['rows']], dtype=float)
    tmp = f'{npy[:-4]}.tmp{os.getpid()}.npy'
    np.save(tmp, ms)
    link_new(tmp, npy)
    rec = dict(rec, array=os.path.relpath(npy, ROOT) if npy.startswith(ROOT) else npy,
               array_sha256=sha256(npy))
    tmp = f'{js}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(F88.jsonable(rec), f)
    link_new(tmp, js)
    return js


def read_record(seed, combo, design, names, partner):
    """The record of one training seed and combination, checked against the design, the
    instance list, its array, its rows and the stored array; Incomplete when absent."""
    js, npy = record_path(seed, combo), array_path(seed, combo)
    if not os.path.exists(js):
        raise Incomplete(f'missing {js}')
    rec = json.load(open(js))
    if rec.get('design') != design:
        diff = sorted(k for k in set(design) | set(rec.get('design') or {})
                      if (rec.get('design') or {}).get(k) != design.get(k))
        raise Refused(f'{js} is of another design (fields {", ".join(diff)})')
    if (rec.get('seed'), rec.get('combination'), rec.get('partner')) != (int(seed), combo, partner):
        raise Refused(f'{js} holds seed {rec.get("seed")}, combination {rec.get("combination")} '
                      f'and partner {rec.get("partner")}')
    if rec.get('instances') != list(names) or len(rec.get('rows', [])) != len(names):
        raise Refused(f'{js} does not hold the {len(names)} instances of this run')
    if not os.path.exists(npy) or sha256(npy) != rec.get('array_sha256'):
        raise Refused(f'{npy} is absent or differs from the array of {js}')
    ms = REG.load(ARM, None, seed, FOLDER, combo)
    mine = np.array([r['makespan'] for r in rec['rows']], dtype=float)
    if not np.array_equal(ms, mine):
        raise Refused(f'{npy} differs from the makespans of {js}')
    for name, r in zip(names, rec['rows']):
        ct = np.asarray(r['op_ct'], dtype=float)
        if r['instance'] != name or float(ct.max()) != float(r['makespan']) \
                or min(r['amch']) < 0 or len(r['actions']) != len(ct):
            raise Refused(f'{js}: the row of {name} is not a complete schedule')
    ref = stored_array(seed, combo, len(names))
    if not np.array_equal(mine, ref):
        i = int(np.flatnonzero(mine != ref)[0])
        raise Refused(f'{js}: {names[i]} has makespan {mine[i]:g}, the stored array '
                      f'{REG.result_path(ARM, None, seed, STORED, combo)} {ref[i]:g}')
    vc = rec.get('vector_check') or {}
    if not vc.get('all_equal') or vc.get('partner') != partner or not vc.get('decisions'):
        raise Refused(f'{js}: the network output changed with the descriptor vector of '
                      f'{partner}, or the check is missing')
    return rec


# ----------------------------------------------------------------- rollouts
def import_eval_family():
    """eval_family, imported with the options of the paper's evaluations: params.py parsed
    from params_argv() (CPU, exact mask options) and --torch_threads 1, which eval_family
    takes from the command line at import."""
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    ensure_params()
    saved = sys.argv
    sys.argv = [saved[0]] + params_argv() + ['--torch_threads', '1']
    try:
        import eval_family as EF
    finally:
        sys.argv = saved
    return EF


class Roller:
    """The policy of one training seed, loaded as eval_family.py loads it; roll() returns the
    greedy schedule of one instance as eval_family.py's greedy branch builds it."""

    def __init__(self, seed):
        self.seed = int(seed)
        self.design = light_design(seed)
        EF = import_eval_family()
        import torch
        from params import configs
        from fjsp_env_same_op_nums import EnvState
        from constraint_family.fjsp_env_family import FJSPEnvFamily, resolve_delta_mode
        from constraint_family.descriptor import compute_token
        from common_utils import greedy_select_action
        if configs.device != 'cpu' or str(EnvState.device) != 'cpu':
            raise Refused(f'device {configs.device} / {EnvState.device}; the rollouts run on '
                          'the CPU')
        if torch.get_num_threads() != 1:
            raise Refused(f'torch runs {torch.get_num_threads()} threads, want 1')
        EF.ARG_MODEL = self.design['model']
        ppo, snap = EF.load_model()
        n = int(sum(p.numel() for p in ppo.policy.parameters()))
        if n != PARAMS:
            raise Refused(f'{self.design["model"]} has {n} parameters, want {PARAMS}')
        if getattr(configs, 'use_film', False) or getattr(configs, 'token_to_heads', False):
            raise Refused(f'{self.design["model"]} reads the descriptor vector')
        masks = EF.mask_options(snap)
        got = {k: v[0] for k, v in masks.items()}
        if got != self.design['mask_options']:
            raise Refused(f'eval_family.mask_options gives {masks}, want '
                          f'{self.design["mask_options"]}')
        self.mask_sources = {k: v[1] for k, v in masks.items()}
        self.dm = resolve_delta_mode(configs)
        self.mm = getattr(configs, 'mask_mode', 'nondelay')
        self.pe = getattr(configs, 'pair_extra', 'none')
        if (self.dm, self.mm, self.pe) != ('min', 'nondelay', 'none'):
            raise Refused(f'delta_mode {self.dm}, mask_mode {self.mm}, pair_extra {self.pe}')
        self.torch, self.ppo = torch, ppo
        self.FJSPEnvFamily, self.compute_token = FJSPEnvFamily, compute_token
        self.greedy = greedy_select_action

    def _forward(self, st, token):
        with self.torch.no_grad():
            pi, _ = self.ppo.policy(
                fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor, candidate=st.candidate_tensor,
                fea_m=st.fea_m_tensor, mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
                dynamic_pair_mask=st.dynamic_pair_mask_tensor, fea_pairs=st.fea_pairs_tensor,
                op_type=st.op_type_tensor, mch_type=st.mch_type_tensor, token=token)
        return pi

    def roll(self, desc, other, jl, pt, meta, full_check):
        """Greedy schedule of one instance under `desc`; at every decision (full_check) or at
        the first, the network is also evaluated with the descriptor vector of `other`."""
        if meta['op_type'] is None:
            raise Refused('a factory instance carries its types; this one has none')
        N, M = pt.shape
        fm, mt = self.design['mask_options']['first_mask'], self.design['mask_options']['mask_ties']
        env = self.FJSPEnvFamily(jl.shape[0], M, family_mode=True, blind_channels=(),
                                 mask_mode=self.mm, pair_extra=self.pe,
                                 first_mask=fm, mask_ties=mt)
        env.delta_mode = self.dm
        env.reward_delta_env = (self.dm != 'zero')
        env.set_initial_data([jl], [pt], descriptor_list=[desc],
                             op_type_list=[meta['op_type']], mch_type_list=[meta['mch_type']])
        mp = float(pt[pt > 0].mean())
        env.token_g = np.asarray([self.compute_token(desc, mp, M)])
        env._attach_token()
        state = env.state
        tok_other = self.torch.from_numpy(
            np.asarray([self.compute_token(other, mp, M)])).float().to(state.token_tensor.device)
        amch = np.full(N, -1, dtype=int)
        actions, compared, equal = [], 0, True
        while not env.done().all():
            pi = self._forward(state, state.token_tensor)
            if full_check or not actions:
                compared += 1
                equal = equal and bool(self.torch.equal(pi, self._forward(state, tok_other)))
            a = self.greedy(pi)
            a0 = int(a.cpu().numpy()[0])
            amch[env.candidate[0, a0 // M]] = a0 % M
            actions.append(a0)
            state, _, _ = env.step(a.cpu().numpy())
        if len(actions) != N:
            raise Refused(f'{len(actions)} decisions for {N} operations')
        ct = np.asarray(env.true_op_ct[0], dtype=float).copy()
        return dict(makespan=float(env.current_makespan[0]), amch=amch.tolist(),
                    op_ct=ct.tolist(), actions=actions), compared, equal


def say(*a):
    print(*a, flush=True)


def roll_record(roller, combo, partner, names, log=say):
    """Rows of one training seed and combination, the descriptor-vector check and the times."""
    rows, compared, equal = [], 0, True
    t0, c0 = time.time(), time.process_time()
    for i, name in enumerate(names):
        jl, pt, meta = read_instance(name)
        row, k, eq = roller.roll(read_desc(combo, name), read_desc(partner, name), jl, pt, meta,
                                 full_check=i < VEC_FULL)
        rows.append(dict(instance=name, **row))
        compared += k
        equal = equal and eq
        if (i + 1) % 25 == 0:
            log(f'[p121 rollout] seed {roller.seed} {combo:4s} {i + 1}/{len(names)} '
                f'{time.time() - t0:6.0f}s')
    return rows, dict(partner=partner, decisions=compared, all_equal=equal,
                      full_instances=min(VEC_FULL, len(names))), \
        dict(wall_s=time.time() - t0, cpu_s=time.process_time() - c0)


def rollout(seeds, combos, budget_min=None, log=say):
    """Roll out every missing record of the given seeds and combinations; records that exist
    are checked and kept. Stops before a record that would end after the budget."""
    check_setting()
    p40 = load_p40()
    pairs = pairs_of(p40)
    partner = partner_of(pairs)
    bad = [c for c in combos if c not in partner]
    if bad:
        raise Refused(f'combinations {bad} belong to no pair of p40 ({pairs})')
    names = instance_names(combos[0])
    if len(names) != N_TEST:
        raise Refused(f'{DATA}/desc/{combos[0]} holds {len(names)} descriptors, want {N_TEST}')
    t_all, last, counts = time.time(), None, dict(written=0, existed=0, left=0)
    for seed in seeds:
        roller = None
        for combo in combos:
            design = light_design(seed)
            try:
                read_record(seed, combo, design, names, partner[combo])
                counts['existed'] += 1
                continue
            except Incomplete:
                pass
            for p in (array_path(seed, combo),):
                if os.path.exists(p):
                    raise Refused(f'{p} exists without its record; remove it by hand')
            if budget_min and last and time.time() - t_all + 1.1 * last > 60.0 * budget_min:
                counts['left'] += 1
                log(f'[p121 rollout] budget: seed {seed} {combo} left for the next command')
                continue
            ref = stored_array(seed, combo, N_TEST)
            if roller is None:
                roller = Roller(seed)
                log(f'[p121 rollout] {design["model"]}: {PARAMS} parameters, mask options '
                    f'{design["mask_options"]} ({roller.mask_sources}), delta {roller.dm}, '
                    f'sha256 {design["checkpoint_sha256"][:12]}')
            t0 = time.time()
            rows, vc, times = roll_record(roller, combo, partner[combo], names, log)
            mine = np.array([r['makespan'] for r in rows], dtype=float)
            if not np.array_equal(mine, ref):
                i = int(np.flatnonzero(mine != ref)[0])
                raise Refused(f'seed {seed} {combo} {names[i]}: makespan {mine[i]:g}, the stored '
                              f'array {ref[i]:g}; nothing written')
            if not vc['all_equal']:
                raise Refused(f'seed {seed} {combo}: the network output changed with the '
                              f'descriptor vector of {partner[combo]}; nothing written')
            rec = dict(generator=GENERATOR, design=design, seed=int(seed), combination=combo,
                       partner=partner[combo], instances=names, rows=rows, vector_check=vc,
                       stored=dict(array=REG.result_path(ARM, None, seed, STORED, combo),
                                   identical=True),
                       mask_sources=roller.mask_sources,
                       times=dict(times, note='planning figures on a shared core'),
                       written=time.strftime('%Y-%m-%d %H:%M:%S'))
            path = write_record(seed, combo, rec)
            last = time.time() - t0
            counts['written'] += 1
            log(f'[p121 rollout] wrote {path} ({last:.0f}s; {vc["decisions"]} decisions with '
                f'the descriptor vector of {partner[combo]}, output unchanged)')
    log(f'[p121 rollout] {counts["written"]} records written, {counts["existed"]} existed, '
        f'{counts["left"]} left for the next command')
    return counts


# ----------------------------------------------------------------- validation and checks
PRESENCE = re.compile(r'^job \d+: setup of (first )?op \d+ starts before ')


def buckets(messages, classify):
    """reference_sim messages by rule: p40's classify (lag, setup, window, other) on every
    message except those of a setup block that starts before its job is present ('job j:
    setup of op b starts before pred ct+lag', and the release form), which classify would put
    into lag because they contain 'pred ct+lag'; they are counted as presence."""
    pres = [x for x in messages if PRESENCE.match(x)]
    out = dict(classify([x for x in messages if not PRESENCE.match(x)]))
    out['presence'] = len(pres)
    return out


def validate(desc, jl, pt, row, classify):
    """C6 of one schedule against `desc`: rejected or not, reference_sim messages by rule
    (buckets), schedule_validator messages by prefix."""
    r = CHK.check_schedule_validation(desc, jl, pt, np.asarray(row['amch'], dtype=int),
                                      np.asarray(row['op_ct'], dtype=float))
    v = r.details['violations']
    rs = [x[len('[reference_sim] '):] for x in v if x.startswith('[reference_sim] ')]
    sv = [x[len('[schedule_validator] '):] for x in v if x.startswith('[schedule_validator] ')]
    if len(rs) + len(sv) != len(v):
        raise Refused(f'a C6 message of unknown origin: {v[:2]}')
    by_check = {k: 0 for k in SV_PREFIXES + ('other',)}
    for x in sv:
        m = re.match(r'\[(\w+)\]', x)
        by_check[m.group(1) if m and m.group(1) in SV_PREFIXES else 'other'] += 1
    return dict(rejected=r.status == CHK.FAIL, status=r.status, by_rule=buckets(rs, classify),
                p40_lag=classify(rs)['lag'], by_module=by_check, reference_sim=len(rs),
                schedule_validator=len(sv))


def cell_counts(vals):
    """Counts over the schedules of one case and seed."""
    out = dict(schedules=len(vals), rejected=sum(v['rejected'] for v in vals),
               rejected_reference_sim=sum(v['reference_sim'] > 0 for v in vals))
    for k in ('lag', 'setup', 'window', 'presence', 'other'):
        out[k] = int(sum(v['by_rule'][k] for v in vals))
        out[f'{k}_schedules'] = int(sum(v['by_rule'][k] > 0 for v in vals))
    out['lag_p40_classify'] = int(sum(v['p40_lag'] for v in vals))
    out['schedule_validator'] = {k: int(sum(v['by_module'][k] for v in vals))
                                 for k in SV_PREFIXES + ('other',)}
    out['bound_schedules'] = int(sum(v['by_module']['bound'] > 0 for v in vals))
    return out


def function_location(fn):
    path = inspect.getsourcefile(fn)
    return f'{os.path.relpath(path, ROOT)}:{inspect.getsourcelines(fn)[1]}'


def check_locations():
    out = {cid: dict(function=f'constraint_family.checks.{CHECK_FUNCS[cid]}',
                     location=function_location(getattr(CHK, CHECK_FUNCS[cid])),
                     basis=BASIS[cid]) for cid in CHECK_IDS}
    out['run_all'] = dict(function='constraint_family.checks.run_all',
                          location=function_location(CHK.run_all))
    return out


def descriptor_checks(desc, jl, pt):
    """{check id: (status, messages)} of checks.run_all without a schedule (C6 deferred)."""
    return {r.id: (r.status, list(r.messages))
            for r in CHK.run_all(desc, pt, False, job_length=jl)}


def status_counts(reports, cid):
    out = {}
    for rep in reports:
        out[rep[cid][0]] = out.get(rep[cid][0], 0) + 1
    return dict(sorted(out.items()))


def faulty_descriptors(desc, pt):
    """{fault key: malformed copy of desc} (module docstring, FAULT INJECTION)."""
    out = {}
    d = copy.deepcopy(desc)
    nz = np.flatnonzero(d.lag > 0)
    if not len(nz):
        raise Refused('the base descriptor has no nonzero lag')
    d.lag = d.lag.copy()
    d.lag[nz[0]] = -d.lag[nz[0]]
    out['neglag'] = d
    d = copy.deepcopy(desc)
    d.setup = d.setup.copy()
    d.setup[0, 1] = -d.setup[0, 1]
    out['negsetup'] = d
    m = next((k for k, w in enumerate(desc.windows) if w), None)
    if m is None:
        raise Refused('the base descriptor has no window')
    s, e = desc.windows[m][0]
    if e - s < 2:
        raise Refused(f'window {(s, e)} is too short to overlap a 1 h shifted copy')
    d = copy.deepcopy(desc)
    d.windows[m] = [(s, e), (s + 1, e + 1)] + list(desc.windows[m][1:])
    out['overlap'] = d
    d = copy.deepcopy(desc)
    d.windows[m] = [(e, s)] + list(desc.windows[m][1:])
    out['reversed'] = d
    d = copy.deepcopy(desc)
    d.setup = 2.0 * d.setup
    out['twice'] = d
    pbar = float(np.asarray(pt, dtype=float)[np.asarray(pt) > 0].mean())
    lo, hi = CHK.SETUP_CROSS_RANGE
    lo_h, hi_h = max(1, int(np.ceil(lo * pbar))), int(np.floor(hi * pbar))
    v = int(desc.setup[0, 1])
    alt = [x for x in range(lo_h, hi_h + 1) if abs(x - v) >= 1]
    if not alt:
        raise Refused(f'no other integer setup inside [{lo_h}, {hi_h}] h')
    d = copy.deepcopy(desc)
    d.setup = d.setup.copy()
    d.setup[1, 0] = float(alt[0])
    out['asym'] = d
    return out


def fault_table(desc, jl, pt):
    """[{fault, text, statuses {id: status}, caught_by [ids that fail or warn], messages}]"""
    rows = []
    faults = faulty_descriptors(desc, pt)
    for key, text, word in FAULTS:
        rep = descriptor_checks(faults[key], jl, pt)
        caught = [c for c in CHECK_IDS if rep[c][0] in (CHK.FAIL, CHK.WARN)
                  and not (c == 'C4' and rep[c][1] == ['not evaluated because C1 failed'])]
        rows.append(dict(fault=key, text=text, word=word,
                         statuses={c: rep[c][0] for c in CHECK_IDS},
                         caught_by=caught, messages={c: rep[c][1] for c in caught}))
    return rows


# ----------------------------------------------------------------- analysis
def collect(seeds, pairs, names, design_of):
    """{(seed, combination): record}; Refused naming every record that does not count."""
    partner = partner_of(pairs)
    recs, miss = {}, []
    for seed in seeds:
        try:
            design = design_of(seed)
        except Refused as e:
            miss.append(str(e))
            continue
        for combo in combos_of(pairs):
            try:
                recs[(seed, combo)] = read_record(seed, combo, design, names, partner[combo])
            except Refused as e:
                miss.append(str(e))
    if miss:
        raise Incomplete(f'{len(miss)} of {len(seeds) * len(combos_of(pairs))} records do not '
                         'count:\n  ' + '\n  '.join(miss))
    return recs


def analyse(seeds, names, design_of):
    p40 = load_p40()
    pairs = pairs_of(p40)
    classify = p40['classify']
    recs = collect(seeds, pairs, names, design_of)
    inst = {nm: read_instance(nm) for nm in names}
    part1, part2 = [], {}
    for true_c, conf_c, lab in pairs:
        tdesc = {nm: read_desc(true_c, nm) for nm in names}
        cdesc = {nm: read_desc(conf_c, nm) for nm in names}
        env_of = {'A': true_c, 'C': conf_c}
        conf_of = {'A': tdesc, 'C': cdesc}
        cases = {}
        for case in CASES_RUN:
            per_seed, sched_true, sched_conf = {}, [], []
            for seed in seeds:
                rows = recs[(seed, env_of[case])]['rows']
                vt = [validate(tdesc[r['instance']], *inst[r['instance']][:2], r, classify)
                      for r in rows]
                vc = [validate(conf_of[case][r['instance']], *inst[r['instance']][:2], r,
                               classify) for r in rows]
                ms = REG.load(ARM, None, seed, FOLDER, env_of[case])
                per_seed[seed] = dict(mean=float(ms.mean()), true=cell_counts(vt),
                                      configured=cell_counts(vc))
                sched_true += vt
                sched_conf += vc
            pooled = REG.pooled(ARM, None, FOLDER, env_of[case], seed_list=list(seeds))
            cases[case] = dict(environment=env_of[case], per_seed=per_seed,
                               pooled=dict(mean=float(pooled.mean()),
                                           true=cell_counts(sched_true),
                                           configured=cell_counts(sched_conf)))
        for case in CASES_RUN:
            c = cases[case]
            for seed in seeds:
                c['per_seed'][seed]['change'] = float(F88.rel(
                    c['per_seed'][seed]['mean'], cases['A']['per_seed'][seed]['mean']))
            c['pooled']['change'] = float(F88.rel(c['pooled']['mean'], cases['A']['pooled']['mean']))
            ch = [c['per_seed'][s]['change'] for s in seeds]
            rj = [c['per_seed'][s]['true']['rejected'] for s in seeds]
            c['seed_range'] = dict(change_min=min(ch), change_max=max(ch),
                                   rejected_true_min=min(rj), rejected_true_max=max(rj))
        part1.append(dict(label=lab, true=true_c, configured=conf_c, cases=cases))
        # part 2: the checks on the configured descriptors, the true ones as the control
        rep_conf = [descriptor_checks(cdesc[nm], *inst[nm][:2]) for nm in names]
        rep_true = [descriptor_checks(tdesc[nm], *inst[nm][:2]) for nm in names]
        rows = {}
        for cid in CHECK_IDS:
            if cid == 'C6':
                cc, ct = cases['C']['pooled'], cases['A']['pooled']
                rows[cid] = dict(
                    configured={'PASS': cc['configured']['schedules'] - cc['configured']['rejected'],
                                'FAIL': cc['configured']['rejected']},
                    configured_against_true={'PASS': cc['true']['schedules'] - cc['true']['rejected'],
                                             'FAIL': cc['true']['rejected']},
                    true={'PASS': ct['true']['schedules'] - ct['true']['rejected'],
                          'FAIL': ct['true']['rejected']},
                    unit='schedules of case C (configured) and case A (true), all seeds')
            else:
                rows[cid] = dict(configured=status_counts(rep_conf, cid),
                                 true=status_counts(rep_true, cid), unit='descriptors')
        part2[lab] = rows
    d0 = read_desc(pairs[0][0], names[0])
    faults = fault_table(d0, *inst[names[0]][:2])
    vec = {f'{s}:{c}': recs[(s, c)]['vector_check'] for s in seeds for c in combos_of(pairs)}
    return dict(generator=GENERATOR, setting=f'environment {ENV}, training seeds '
                + ','.join(map(str, seeds)), seeds=list(seeds), instances=len(names),
                pairs=[list(p) for p in pairs], cases_run=list(CASES_RUN),
                case_definitions=CASE_TEXT, part1=part1, part2=part2,
                checks=check_locations(), faults=dict(base=f'{pairs[0][0]} descriptor of '
                                                      f'{names[0]}', rows=faults),
                vector_check=dict(records=vec,
                                  decisions=int(sum(v['decisions'] for v in vec.values())),
                                  all_equal=all(v['all_equal'] for v in vec.values())),
                reproduction=dict(compared=len(recs) * len(names), identical=len(recs) * len(names),
                                  folder=REG.res_dir(STORED)),
                functions=dict(
                    rollout='eval_family.load_model, eval_family.mask_options, the greedy '
                            'branch of eval_family.main (FJSPEnvFamily, greedy_select_action)',
                    validator='constraint_family.checks.check_schedule_validation (C6: '
                              'schedule_validator.validate_schedule and '
                              'reference_sim.validate_schedule)',
                    rule_buckets='scripts/p40_misconfig.py classify (loaded from its source)',
                    change='scripts/p88_family_inference.py rel: 100 (a - b) / b of the means',
                    pooled='arm_registry.pooled: per-instance mean over the seeds'),
                folder=REG.res_dir(FOLDER))


# ----------------------------------------------------------------- outputs
def mac(name, val, note=None):
    return (name, val, note) if note else (name, val)


def macro_lines(res):
    f2 = F88.f2
    seeds = res['seeds']
    L = [f'% Generated by {GENERATOR}, training seeds ' + ', '.join(map(str, seeds)),
         '% Schedules of the policy under a wrong constraint descriptor; case A environment and '
         'inputs from the true descriptor, case C from the configured one; C6 is the validator',
         mac(f'{PREFIX}Seeds', len(seeds), 'training seeds of the policy'),
         mac(f'{PREFIX}NTest', res['instances'], 'test instances per seed'),
         mac(f'{PREFIX}Schedules', len(seeds) * res['instances'], 'schedules per pair and case')]
    for p in res['part1']:
        w = PREFIX + p['label']
        L.append(mac(f'{w}True', p['true'], 'combination in force'))
        L.append(mac(f'{w}False', p['configured'], 'combination configured'))
        for case in CASES_RUN:
            c = p['cases'][case]
            P, T, K = c['pooled'], c['pooled']['true'], c['pooled']['configured']
            L += [mac(f'{w}{case}Ms', f'{P["mean"]:.2f}', f'case {case}: mean makespan, h, '
                      'pooled over the seeds'),
                  mac(f'{w}{case}dMs', f2(P['change']), f'case {case}: change of the mean '
                      'makespan against case A, %'),
                  mac(f'{w}{case}dMsMin', f2(c['seed_range']['change_min']),
                      f'case {case}: smallest change over the seeds, %'),
                  mac(f'{w}{case}dMsMax', f2(c['seed_range']['change_max']),
                      f'case {case}: largest change over the seeds, %'),
                  mac(f'{w}{case}RejTrue', T['rejected'], f'case {case}: schedules the validator '
                      'rejects against the true descriptor, all seeds'),
                  mac(f'{w}{case}RejTrueMin', c['seed_range']['rejected_true_min'],
                      f'case {case}: fewest rejected against the true descriptor in one seed'),
                  mac(f'{w}{case}RejConf', K['rejected'], f'case {case}: schedules the validator '
                      'rejects against the configured descriptor, all seeds'),
                  mac(f'{w}{case}Lag', T['lag'], f'case {case}: lag violations, all schedules'),
                  mac(f'{w}{case}Set', T['setup'], f'case {case}: setup violations'),
                  mac(f'{w}{case}Win', T['window'], f'case {case}: window violations'),
                  mac(f'{w}{case}Pres', T['presence'], f'case {case}: setups that start before '
                      'their job is present'),
                  mac(f'{w}{case}LagSched', T['lag_schedules'], f'case {case}: schedules with a '
                      'lag violation'),
                  mac(f'{w}{case}SetSched', T['setup_schedules'], f'case {case}: schedules with '
                      'a setup violation'),
                  mac(f'{w}{case}WinSched', T['window_schedules'], f'case {case}: schedules with '
                      'a window violation'),
                  mac(f'{w}{case}PresSched', T['presence_schedules'], f'case {case}: schedules '
                      'with a setup that starts before its job is present'),
                  mac(f'{w}{case}Bound', T['bound_schedules'], f'case {case}: schedules whose '
                      'makespan is below the job-chain bound of the true lags')]
        fails = sum(1 for cid in ('C1', 'C2', 'C3', 'C4', 'C5')
                    for st in res['part2'][p['label']][cid]['configured']
                    if st in (CHK.FAIL, CHK.WARN))
        L.append(mac(f'{w}CheckAlarms', fails, 'checks C1 to C5 that fail or warn on any '
                     'configured descriptor'))
    for r in res['faults']['rows']:
        w = f'{PREFIX}Fault{r["word"]}'
        L.append(mac(f'{w}Check', ', '.join(r['caught_by']) or 'none',
                     f'checks that fail or warn on {r["text"]}'))
        L.append(mac(f'{w}Status', ', '.join(r['statuses'][c] for c in r['caught_by']) or 'PASS',
                     'their statuses'))
    L.append(mac(f'{PREFIX}VecDecisions', res['vector_check']['decisions'], 'decisions at which '
                 'the network was also evaluated with the other descriptor vector, output '
                 'unchanged at every one'))
    return L


def defined_macros(own):
    """{name: file} of every macro defined in a .tex file under manuscript/, the files named
    like the outputs `own` left out (the output and its copy in manuscript/)."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    out = {}
    own = {os.path.basename(p) for p in own}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.basename(f) in own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, os.path.relpath(f, ROOT))
    return out


def check_names(lines, own):
    names = [x[0] for x in lines if isinstance(x, tuple)]
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def esc(text):
    return str(text).replace('|', '\\|')


def markdown(res):
    seeds = res['seeds']
    L = [f'# Schedules under a wrong constraint descriptor ({res["setting"]})', '',
         f'Generated by {GENERATOR}. {res["instances"]} test instances of {DATA} per seed, the '
         'policy xmask-v1-s<seed> under the exact non-delay mask (first_mask exact, mask_ties '
         'exact), one torch thread. Every schedule was re-checked by check C6 (the two '
         'validator modules).', '', '## Cases', '']
    for k in ('A', 'B', 'C', 'D'):
        L.append(f'- {k}: {res["case_definitions"][k]}')
    vc = res['vector_check']
    L += ['', f'Descriptor vector: at {vc["decisions"]} decisions the network was evaluated with '
          'the descriptor vector of the other combination of the pair; its output was '
          + ('identical at every one.' if vc['all_equal'] else 'NOT identical.'),
          f'Reproduction: all {res["reproduction"]["compared"]} makespans equal the stored arrays '
          f'of {res["reproduction"]["folder"]}.', '',
          '## Part 1: makespan and validator, per pair, case and training seed', '',
          '| pair (in force / configured) | case | seed | mean makespan (h) | change vs A (%) | '
          'rejected, true descriptor | lag | setup | window | setup before job present | '
          'rejected, configured descriptor |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for p in res['part1']:
        for case in CASES_RUN:
            c = p['cases'][case]
            rows = [(str(s), c['per_seed'][s]) for s in seeds] + [('pooled', c['pooled'])]
            for lab, x in rows:
                T, K = x['true'], x['configured']
                L.append(f'| {p["true"]} / {p["configured"]} | {case} | {lab} | {x["mean"]:.2f} | '
                         f'{x["change"]:+.2f} | {T["rejected"]}/{T["schedules"]} | {T["lag"]} | '
                         f'{T["setup"]} | {T["window"]} | {T["presence"]} | '
                         f'{K["rejected"]}/{K["schedules"]} |')
    L += ['', 'Violation counts are the messages of reference_sim.validate_schedule against the '
          'true descriptor, summed over the schedules, in the rule buckets of p40 (classify), '
          'except the messages of a setup that starts before its job is present ("setup of op '
          '... starts before pred ct+lag"), which classify would count as lag and which are '
          'counted in their own column. Pooled: the mean of the per-instance means over the '
          'seeds; counts summed.', '',
          '| pair | case | schedules with a lag / setup / window / setup-before-job violation | '
          'lag bucket of classify on all messages | schedule_validator messages (precedence, '
          'overlap, bound) | schedules below the true job-chain bound | change range over seeds '
          '(%) |', '|---|---|---|---|---|---|---|']
    for p in res['part1']:
        for case in CASES_RUN:
            c = p['cases'][case]
            T = c['pooled']['true']
            sv = T['schedule_validator']
            L.append(f'| {p["true"]} / {p["configured"]} | {case} | {T["lag_schedules"]} / '
                     f'{T["setup_schedules"]} / {T["window_schedules"]} / '
                     f'{T["presence_schedules"]} | {T["lag_p40_classify"]} | {sv["precedence"]}, '
                     f'{sv["overlap"]}, {sv["bound"]} | {T["bound_schedules"]} | '
                     f'{c["seed_range"]["change_min"]:+.2f} to {c["seed_range"]["change_max"]:+.2f} |')
    L += ['', '## Part 2: the configuration checks on the wrong descriptor', '',
          '| check | function (file:line) | basis | ' + ' | '.join(
              f'{p["true"]} in force, {p["configured"]} configured' for p in res['part1'])
          + ' | true descriptors (control) |', '|---|---|---|' + '---|' * (len(res['part1']) + 1)]
    for cid in CHECK_IDS:
        loc = res['checks'][cid]
        cells = []
        for p in res['part1']:
            r = res['part2'][p['label']][cid]
            if cid == 'C6':
                cells.append(f'against configured {r["configured"]}; against true '
                             f'{r["configured_against_true"]}')
            else:
                cells.append(str(r['configured']))
        ctl = '; '.join(f'{p["true"]} {res["part2"][p["label"]][cid]["true"]}'
                        for p in res['part1'])
        L.append(f'| {cid} | {loc["function"].split(".")[-1]} ({loc["location"]}) | '
                 f'{esc(loc["basis"])} | ' + ' | '.join(cells) + f' | {ctl} |')
    L += ['', f'All C1 to C7 statuses come from {res["checks"]["run_all"]["function"]} '
          f'({res["checks"]["run_all"]["location"]}), with finite_downstream_buffers=False '
          'declared; C6 from the schedules of part 1.', '',
          f'## Part 3: fault injection on the {res["faults"]["base"]}', '',
          '| fault | ' + ' | '.join(CHECK_IDS) + ' | caught by | first message |',
          '|---|' + '---|' * (len(CHECK_IDS) + 2)]
    for r in res['faults']['rows']:
        first = next(iter(r['messages'].values()), ['-'])[0] if r['messages'] else '-'
        L.append(f'| {r["text"]} | ' + ' | '.join(r['statuses'][c] for c in CHECK_IDS)
                 + f' | {", ".join(r["caught_by"]) or "none"} | {esc(first)} |')
    L += ['', 'C4 reports WARN "not evaluated because C1 failed" when C1 fails; that status is '
          'not counted as catching the fault. C6 needs a schedule and is deferred (MANUAL) on a '
          'descriptor alone; C7 is always MANUAL.', '', '## Functions called', '']
    for k, v in res['functions'].items():
        L.append(f'- {k}: {v}')
    return '\n'.join(L) + '\n'


def outputs():
    os.environ.setdefault('PACT_OUT', DEFAULT_OUT)
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(names=None, design_of=light_design):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, outs)."""
    seeds = check_setting()
    names = names or instance_names(combos_of(pairs_of(load_p40()))[0])
    if design_of is light_design and len(names) != N_TEST:
        raise Refused(f'{DATA} holds {len(names)} test instances, want {N_TEST}')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(seeds, names, design_of)
        lines = macro_lines(res)
        mnames = check_names(lines, [outs['macros']])
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
        with open(outs['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        remove_outputs(outs)
        raise
    return res, mnames, outs


def table(res):
    rows = ['  pair       case  mean (h)  change %  rejected true  lag/setup/window   rejected '
            'configured']
    for p in res['part1']:
        for case in CASES_RUN:
            P = p['cases'][case]['pooled']
            T, K = P['true'], P['configured']
            rows.append(f'  {p["true"]:>3}/{p["configured"]:<4}  {case}     {P["mean"]:8.2f}  '
                        f'{P["change"]:+8.2f}  {T["rejected"]:5d}/{T["schedules"]:<5d}    '
                        f'{T["lag"]}/{T["setup"]}/{T["window"]:<10}  {K["rejected"]}/{K["schedules"]}')
    return rows


# ----------------------------------------------------------------- check and dry
def check(log=print, n=None, design_of=light_design):
    """Read-only precondition of the analysis; returns the exit code. n and design_of select
    the first n instances and another design (the self-test's)."""
    seeds = check_setting()
    p40 = load_p40()
    pairs = pairs_of(p40)
    combos = combos_of(pairs)
    names = instance_names(combos[0], n)
    want = n or N_TEST
    miss = []
    for c in combos:
        nm = instance_names(c, n)
        if nm != names or len(nm) != want:
            miss.append(f'{DATA}/desc/{c} holds {len(nm)} descriptors, want the {want} of '
                        f'{combos[0]}')
    for nm in names:
        if not os.path.exists(os.path.join(ROOT, BASE, nm + '.meta.json')):
            miss.append(f'missing base instance {BASE}/{nm}')
    for s in seeds:
        for c in combos:
            p = REG.result_path(ARM, None, s, STORED, c)
            if not os.path.exists(p):
                miss.append(f'missing stored array {p}')
    try:
        recs = collect(seeds, pairs, names, design_of)
        n_ok = len(recs)
        rec_miss = []
    except Incomplete as e:
        n_ok, rec_miss = None, str(e).split('\n  ')[1:]
    except Refused as e:
        n_ok, rec_miss = None, [str(e)]
    total = len(seeds) * len(combos)
    log(f'p121: seeds {",".join(map(str, seeds))}, combinations {",".join(combos)}: '
        f'{total - len(rec_miss) if n_ok is None else n_ok} of {total} records present with the '
        f'design of this run and the stored makespans')
    for m in (miss + rec_miss)[:12]:
        log(f'p121:   {m}')
    if len(miss + rec_miss) > 12:
        log(f'p121:   ... and {len(miss + rec_miss) - 12} more')
    return 0 if not miss and not rec_miss else 1


def dry_paths(tree):
    return {'PACT_RESULTS': os.path.join(tree, 'results'), 'PACT_OUT': os.path.join(tree, 'out')}


def dry_order(jl, pt, offset):
    """A fixed dispatch order: jobs in cyclic order from `offset`, each operation on its
    fastest eligible machine (lowest index on a tie)."""
    J = len(jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    left = np.asarray(jl, dtype=int).copy()
    acts = []
    while left.sum():
        for q in range(J):
            j = (q + offset) % J
            if left[j]:
                op = first[j] + jl[j] - left[j]
                m = int(np.argmin(np.where(pt[op] > 0, pt[op], np.inf)))
                acts.append((j, m))
                left[j] -= 1
    return acts


def build_dry(tree, seeds, names):
    """Synthetic records and stored arrays under tree (dry_paths)."""
    for k, v in dry_paths(tree).items():
        os.environ[k] = v
    pairs = pairs_of(load_p40())
    partner = partner_of(pairs)
    for q, seed in enumerate(seeds):
        for combo in combos_of(pairs):
            rows = []
            for nm in names:
                jl, pt, meta = read_instance(nm)
                acts = dry_order(jl, pt, q)
                sim = RS.simulate(jl, pt, read_desc(combo, nm), acts)
                rows.append(dict(instance=nm, makespan=float(sim['makespan']),
                                 amch=[int(x) for x in sim['op_mch']],
                                 op_ct=[float(x) for x in sim['op_ct']],
                                 actions=[int(j) * pt.shape[1] + int(m) for j, m in acts]))
            st = REG.result_path(ARM, None, seed, STORED, combo)
            os.makedirs(os.path.dirname(st), exist_ok=True)
            np.save(st, np.array([r['makespan'] for r in rows]))
            write_record(seed, combo, dict(
                generator=GENERATOR, design=dry_design(seed), seed=int(seed), combination=combo,
                partner=partner[combo], instances=list(names), rows=rows,
                vector_check=dict(partner=partner[combo], decisions=len(names), all_equal=True,
                                  full_instances=0),
                stored=dict(array=st, identical=True), written='dry'))


def dry(tree=None, log=print):
    """The whole analysis on synthetic records; asserts the known answers. Returns 0."""
    own = tree is None
    tree = tree or tempfile.mkdtemp(prefix='p121_dry_')
    os.makedirs(tree, exist_ok=True)
    saved = {k: os.environ.get(k) for k in ('PACT_RESULTS', 'PACT_OUT', 'PACT_ENV', 'PACT_SEEDS')}
    try:
        os.environ['PACT_ENV'] = ENV
        seeds = active_seeds()
        pairs = pairs_of(load_p40())
        names = instance_names(combos_of(pairs)[0], DRY_N)
        build_dry(tree, seeds, names)
        res, mnames, outs = run(names, dry_design)
        bad = []
        for p in res['part1']:
            A, C = p['cases']['A'], p['cases']['C']
            n = len(seeds) * len(names)
            if A['pooled']['true']['rejected'] or A['pooled']['configured']['rejected']:
                bad.append(f'{p["label"]}: case A rejected')
            if C['pooled']['configured']['rejected']:
                bad.append(f'{p["label"]}: case C rejected against the configured descriptor')
            if p['true'] == 'LSW' and p['configured'] == 'N' and C['pooled']['true']['rejected'] != n:
                bad.append(f'{p["label"]}: {C["pooled"]["true"]["rejected"]} of {n} LSW schedules '
                           'built under N rejected')
            ma = np.mean([np.mean([r['makespan'] for r in json.load(open(record_path(s, p['true'])))['rows']])
                          for s in seeds])
            mc = np.mean([np.mean([r['makespan'] for r in json.load(open(record_path(s, p['configured'])))['rows']])
                          for s in seeds])
            if abs(C['pooled']['change'] - 100.0 * (mc - ma) / ma) > 1e-9:
                bad.append(f'{p["label"]}: pooled change {C["pooled"]["change"]} differs from the '
                           'direct computation')
        caught = {r['fault']: r['caught_by'] for r in res['faults']['rows']}
        want = {'neglag': ['C1'], 'negsetup': ['C1'], 'overlap': ['C1'], 'reversed': ['C1'],
                'twice': ['C4'], 'asym': ['C4']}
        if caught != want:
            bad.append(f'fault table {caught}, want {want}')
        if check(log=lambda *_: None, n=DRY_N, design_of=dry_design) != 0:
            bad.append('the check refused the complete synthetic records')
        if check(log=lambda *_: None) != 1:
            bad.append('the check accepted the synthetic records as the real ones')
        if bad:
            raise Refused('dry run: ' + '; '.join(bad))
        log(f'p121 dry: {len(mnames)} macros, known answers reproduced; outputs in '
            f'{os.path.dirname(outs["json"])}')
        return 0
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if own:
            shutil.rmtree(tree, ignore_errors=True)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='schedules under a wrong constraint descriptor')
    ap.add_argument('command', nargs='?', default='analyse', choices=('analyse', 'rollout'))
    ap.add_argument('--check', action='store_true', help='read-only precondition')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='self-test on synthetic records (default a temporary folder)')
    ap.add_argument('--seeds', default=None, help='rollout: training seeds (default the active)')
    ap.add_argument('--combos', default=None, help='rollout: combinations (default the four)')
    ap.add_argument('--budget-min', type=float, default=None,
                    help='rollout: start no record that would end after this many minutes')
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.dry is None:
        os.environ.setdefault('PACT_SEEDS', '301-305')
    try:
        if a.dry is not None:
            sys.exit(dry(os.path.abspath(a.dry) if a.dry else None))
        if a.check:
            sys.exit(check())
        if a.command == 'rollout':
            seeds = check_setting()
            if a.seeds:
                want = REG.parse_seeds(a.seeds)
                if not set(want) <= set(seeds):
                    raise Refused(f'--seeds {a.seeds} is not a subset of the active seeds {seeds}')
                seeds = want
            combos = a.combos.split(',') if a.combos else combos_of(pairs_of(load_p40()))
            rollout(seeds, combos, a.budget_min)
            return
        REG.banner()
        res, names, outs = run()
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    print(f'p121: {res["setting"]}, {res["instances"]} instances per seed')
    print('\n'.join(table(res)))
    for r in res['faults']['rows']:
        print(f'  fault: {r["text"]:42s} caught by {", ".join(r["caught_by"]) or "none"}')
    print(f'\n{len(names)} macros; wrote ' + ', '.join(outs.values()))


if __name__ == '__main__':
    main()
