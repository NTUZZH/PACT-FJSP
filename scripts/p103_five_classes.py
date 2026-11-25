r"""P103: a five-class evaluation level, evaluation only.

Subcommands build, plan, evaluate, infer and check. CPU only, exact mask.

LAUNCH NOTE (the four questions, answered before the run)

1. Purpose. Every experiment of the paper holds four setup classes, and the rule change
   that motivates the Introduction is a new module type that adds a changeover. Level
   five gives a quarter of the modules of every instance a fifth setup class and
   re-measures the comparisons of the sensitivity study (scripts/p87_sensitivity.py)
   at the training size and at both production sizes. The policy and the specialists
   keep the weights they learned with four classes. The numbers land in one row group
   of the sensitivity table of the supplement and in one sentence of the discussion.
2. Expected result. The fifth class changes the changeover between some pairs of
   consecutive modules. The rules' makespans moved by a few percent in both directions
   in the smoke runs, so the outcome of interest is the shift of each comparison. The
   policy reads the setup matrix through three inputs that accept any class count (WHAT
   THE POLICY READS below), so its gaps to the specialist and to the best rule are
   expected to stay within the margins of the four-class level. If a verdict changes,
   the paper reports the measured shift next to the claim about new module types.
3. Contamination. The frozen folders data/FAMILY/10x25, 50x25n30 and 80x25n30, the base
   instances and the stored result folders are read only, and no shared module file is
   edited. New descriptors and results go under --root (default
   test_results/FAMILY/five_p103), a folder no existing script reads
   (tests/test_p103_five_classes.py checks every pattern over test_results/FAMILY).
   build refuses to overwrite a file whose content differs. Rollouts are deterministic,
   so results are not timing measurements and may share cores; the recorded times are
   planning figures.
4. Data accuracy. Level default is the stored descriptors, evaluated by the same code
   path as level five with the class count set to 4. infer refuses when a default-level
   rule array differs from the stored exact-mask array of the same rule
   (test_results/FAMILY/<size>_xmask), and when a default-level policy array differs from
   the exact campaign's array of the same run where that array exists. build checks
   that every field of the stored descriptor outside the fifth class is unchanged.

MECHANISM (the class count of one process; no shared file is edited)
  constraint_family/descriptor.py fixes the class count N_CLASSES = 4 and the index of
  the cold-start row IDLE_CLASS = 4. The environment, checks C1 and C4 and the
  reference validator read both names from their own module globals at call time, and
  each module binds them by `from constraint_family.descriptor import ...`.
  set_class_count(K) imports these four modules and rebinds both names in each of them
  to K before any environment or check is built, so the setup matrix has shape
  [K + 1, K] and every machine starts in the cold-start row K. It then refuses when any
  other loaded module binds either name with another value, or when a module that
  captured the count at import time is loaded (public_rules, public_types). A probe
  environment and a descriptor check with K classes confirm the shape, the cold-start
  row and the agreement of the environment and the validator. The new values live in
  the memory of one process, so a new process imports four classes again. One process
  evaluates one level: default with K = 4, five with K = 5.

LEVELS (sizes 10x25 (100 instances), 50x25n30 and 80x25n30 (30 each); combinations S,
LS, SW, LSW)
  default   the stored descriptors of data/FAMILY/<size>/desc/<R>, four classes.
  five      per instance, floor(J / 4) modules (2 of 10, 12 of 50, 20 of 80) form class
            4, drawn without replacement from numpy default_rng(SeedSequence([20261113,
            5, size code, instance index])) by rng.choice(J, J // 4, replace=False). The
            setup matrix keeps the stored block of classes 0 to 3 and the stored
            cold-start row, which moves to row 5. The same stream then draws the new
            column and row, symmetric, in the generator's ranges and rounding (max(1,
            round(U[lo, hi] p_bar)), p_bar the mean compatible processing time): the
            changeovers between class 4 and classes 0 to 3 in this order from the
            cross-class range U[0.50, 1.00] p_bar, then the same-class reset of class 4
            and its cold start from U[0.25, 0.50] p_bar. Lags, windows, horizon,
            activation bits and every other field stay as stored. The stream does not
            depend on the combination, so an instance has one five-class matrix in S,
            LS, SW and LSW, as the stored matrix is shared.

ARMS (exact-mask names of scripts/arm_registry.py)
  pact    released policy xmask-v1, training seeds --seeds (default 301-305)
  spec    specialist of the combination, xmask-solo-<R>, the same seeds
  rules   FIFO, MOR, SPT, MWKR restricted to the legal pairs (common_utils.
          masked_rule_action) and the descriptor-reading rules ECT, SSPT, LLF
          (scripts/p72_aware_rules.py), each with np.random.seed(50) after the data are
          set, as the stored arrays
  Every policy run is loaded with the configuration of its training snapshot and its
  parameter count is asserted (32786). Every schedule passes check C6 (the independent
  schedule validator and constraint_family/reference_sim.py, which charges each setup
  from the machine's previous class) under the class count of its level, and its
  makespan equals the last completion time. A violation stops the run, and each result
  records the number of validated schedules.

WHAT THE POLICY READS THAT DEPENDS ON THE CLASS COUNT
  The network never receives a class index. Its type inputs are the PPVC operation and
  station types of the base instance. The setup matrix reaches it through three inputs
  of fixed size: the token entries g[6..8] (mean and maximum setup over the mean
  processing time, and the coefficient of variation, over all matrix entries); the
  machine channel holding the cheapest setup from the machine's last class into a
  compatible candidate; and the bound term of each operation, its cheapest processing
  time plus the cheapest setup into its class from any row (cold-start row included).
  A fifth class changes these values and leaves every input shape unchanged.

OUTCOMES (infer)
  Rules (available today): mean makespan of each rule; the best of the four classical
  rules and the best of all seven, per instance; the change of SPT, of the best
  classical rule and of the best of seven from level default to level five, 100
  (x_five - x_default) / x_default per instance with a 95% bootstrap over instances
  (20,000 resamples, fresh default_rng(7)); the changeover hours of the SPT schedule
  and the share that involves class 4; the number of validated schedules and
  violations.
  Policy arms (once trained): per level, size and combination, and pooled over the four
  combinations of a size, the gap of the policy to the specialist with verdicts at the
  margins 0.5% and 1%, and of the policy to the best classical rule and to the best of
  the seven rules with the combined verdict (Holm over the four combinations of a
  size); d = 100 (a - b) / b, seed s against seed s, crossed hierarchical bootstrap over
  training runs and base instances (stats_family.hier_boot, B = 20000, fresh
  default_rng(11)), the estimators of scripts/p102_scale_sensitivity.py. The shift of
  each gap from level default to level five, paired by seed and instance, with the same
  bootstrap; the verdicts that change. The default-level policy arrays are this
  script's rollouts (--default_policy run). With --default_policy stored, a run whose
  exact-campaign array exists (test_results/FAMILY/<size>_xmask) is read from that array,
  and plan and evaluate leave its unit out; the other runs are rolled out as before.
  Outputs: results/five_classes.json and .md, manuscript/macros_fiveclass.tex (prefix
  Fcls), in place for --arms rules or for every arm, otherwise under --out.

USAGE (repository root; CPU only; one level per process)
  python scripts/p103_five_classes.py build
  python scripts/p103_five_classes.py plan     --parts K [--arms rules]
  python scripts/p103_five_classes.py evaluate --level default --arms rules [--part i --parts K]
  python scripts/p103_five_classes.py evaluate --level five    --arms rules [--part i --parts K]
  python scripts/p103_five_classes.py evaluate --level five    --arms pact,spec [--seeds 301-305]
  python scripts/p103_five_classes.py infer    --arms rules
  python scripts/p103_five_classes.py infer    [--default_policy stored]       (every arm)
  (plan, evaluate and infer of one campaign take the same --default_policy)
  python scripts/p103_five_classes.py check    --classes 5 [--policy <run name>]
  Smoke: --root <folder> --n 2, infer --partial --out <folder>; --environment legacy
  names the legacy runs (jointblind-v1, spec-<R>) and is allowed only under --root.
"""
import argparse
import contextlib
import hashlib
import importlib
import json
import os
import re
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

import p101_mask_controls as P101  # noqa: E402
import p102_scale_sensitivity as P102  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

LEVELS = ('default', 'five')
LEVEL_K = {'default': 4, 'five': 5}
BASE_K = 4
NEW_CLASS = 4                       # index of the fifth class
SIZES = ('10x25', '50x25n30', '80x25n30')
COMBOS = ('S', 'LS', 'SW', 'LSW')
RULES4 = P101.RULES
AWARE = P102.AWARE
RULES7 = RULES4 + AWARE
POLICY_ARMS = ('pact', 'spec')
ARM_CHOICES = POLICY_ARMS + ('rules',)
ENVIRONMENTS = ('exact', 'legacy')
FIVE_SEED = 20261113
FIVE_TAG = 5
DEFAULT_ROOT = 'test_results/FAMILY/five_p103'
OUT_FILES = {'json': 'results/five_classes.json', 'md': 'results/five_classes.md',
             'macros': 'manuscript/macros_fiveclass.tex'}
GENERATOR = 'scripts/p103_five_classes.py infer'
PREFIX = 'Fcls'
MARGINS = P102.MARGINS
B = 20000
LEVEL_WORDS = {'default': 'Four', 'five': 'Five'}
MW = P102.MW
# comparison key -> (first-named method, second, macro word)
COMPARISONS = {'pol_spec': ('policy', 'specialist', 'PolSpec'),
               'pol_rule': ('policy', 'best classical rule', 'PolRule'),
               'pol_rule7': ('policy', 'best of the seven rules', 'PolRuleSeven')}
# planning cost, CPU seconds per instance, each including the C6 check. The rules and,
# at 10x25, the policy are the process CPU times of this script's smoke cells
# (2026-11-13, shared core; 80x25n30 rules from scripts/p102_scale_sensitivity.py plus
# the check). The policies at 50 and 80 modules are the exact campaign's rollout times
# (scripts/eval_newseeds_plan.py SPEED) plus the check.
COST_S = {'10x25': {'rule': 0.3, 'policy': 0.75},
          '50x25n30': {'rule': 2.5, 'policy': 16.8},
          '80x25n30': {'rule': 5.5, 'policy': 34.3}}

# the modules that bind the class count and the names each binds, rebound by
# set_class_count
PATCHED = {'constraint_family.descriptor': ('N_CLASSES', 'IDLE_CLASS'),
           'constraint_family.fjsp_env_family': ('N_CLASSES', 'IDLE_CLASS'),
           'constraint_family.checks': ('N_CLASSES', 'IDLE_CLASS'),
           'constraint_family.reference_sim': ('IDLE_CLASS',)}
# modules that copy the class count into defaults at import time; refused at K != 4
CAPTURED = ('constraint_family.public_rules', 'constraint_family.public_types')
CLASS_NAMES = ('N_CLASSES', 'IDLE_CLASS')
_SAVED = {}

Refused = P101.Refused
refuse = P101.refuse
sha256_file = P101.sha256_file
write_json_atomic = P101.write_json_atomic
jsonable = P101.jsonable
f2, u1, u2 = P101.f2, P101.u1, P101.u2


# ---------------------------------------------------------------------------
# class count of the process
# ---------------------------------------------------------------------------

def class_holders():
    """{module name: {name: value}} of every loaded module that binds N_CLASSES or
    IDLE_CLASS at module level."""
    out = {}
    for name, mod in list(sys.modules.items()):
        try:
            d = vars(mod)
        except TypeError:
            continue
        hit = {k: d[k] for k in CLASS_NAMES if k in d}
        if hit:
            out[name] = hit
    return out


def set_class_count(K):
    """Rebind N_CLASSES and IDLE_CLASS to K in every module of PATCHED, then verify the
    process (assert_class_count with the probe). Returns the verification record."""
    if K not in (BASE_K, BASE_K + 1):
        sys.exit(f'REFUSING: class count {K}; this study holds 4 or 5 classes')
    mods = {m: importlib.import_module(m) for m in PATCHED}
    for m, mod in mods.items():
        for k in PATCHED[m]:
            if k not in vars(mod):
                sys.exit(f'REFUSING: {m} no longer binds {k}; the override does not cover '
                         f'it')
            _SAVED.setdefault((m, k), getattr(mod, k))
    if any(v != BASE_K for v in _SAVED.values()):
        sys.exit(f'REFUSING: the stored class constants are {_SAVED}, expected {BASE_K}')
    for m, mod in mods.items():
        for k in PATCHED[m]:
            setattr(mod, k, K)
    return assert_class_count(K, probe=True)


def restore_class_count():
    for (name, k), v in _SAVED.items():
        setattr(sys.modules[name], k, v)
    return assert_class_count(BASE_K)


@contextlib.contextmanager
def class_count(K):
    try:
        yield set_class_count(K)
    finally:
        restore_class_count()


def assert_class_count(K, probe=False):
    """Every loaded module that binds the class names holds K, and no module that
    captured the count at import time is loaded when K differs from 4. With probe, a
    K-class environment and check C1 confirm the shape [K + 1, K], the cold start in row
    K and a validated schedule."""
    holders = class_holders()
    bad = {n: h for n, h in holders.items() if any(v != K for v in h.values())}
    if bad:
        sys.exit(f'REFUSING: modules bind another class count than {K}: {bad}')
    loaded = [m for m in CAPTURED if m in sys.modules]
    if K != BASE_K and loaded:
        sys.exit(f'REFUSING: {loaded} copied the class count at import time; they cannot '
                 f'run with {K} classes')
    out = {'class_count': K, 'holders': sorted(holders)}
    if probe:
        out['probe'] = probe_class_count(K)
    return out


def hand_case(K, cross_new=5):
    """Three one-operation jobs on two machines with K setup classes (0 = incompatible):
       op0 (job 0, class 0):     m0 2 h only
       op1 (job 1, class K - 1): m0 3 h only
       op2 (job 2, class K - 1): m1 1 h only
    Setups are attached to the operation. A changeover takes 1 h within a class,
    cross_new h between class 0 and class K - 1, and 3 h between any other two classes.
    A cold start takes 1 h, or 2 h into class K - 1. With K = 5, class 4 is the fifth
    class."""
    from constraint_family.descriptor import ConstraintDescriptor, job_chain_horizon
    jl = np.array([1, 1, 1])
    pt = np.array([[2, 0], [3, 0], [0, 1]])
    S = np.full((K + 1, K), 3.0)
    for a in range(K):
        S[a, a] = 1.0
    S[0, K - 1] = S[K - 1, 0] = float(cross_new)
    S[K, :] = 1.0
    S[K, K - 1] = 2.0
    desc = ConstraintDescriptor(active=(False, True, False), lag=np.zeros(3),
                                op_class=np.array([0, K - 1, K - 1]), setup=S,
                                horizon=job_chain_horizon(jl, pt, None))
    return jl, pt, {'op_type': None, 'mch_type': None}, desc


def probe_class_count(K):
    from constraint_family import checks
    P101._cpu_torch()
    jl, pt, meta, desc = hand_case(K)
    env = P101.make_env(jl, pt, meta, [desc])
    shape = tuple(env.true_setup.shape[1:])
    if shape != (K + 1, K) or not (env.mch_last_class == K).all():
        sys.exit(f'REFUSING: the environment holds setups {shape} and starts machines in '
                 f'row {np.unique(env.mch_last_class).tolist()}, expected {(K + 1, K)} and '
                 f'{K}')
    c1 = checks.check_schema_and_values(desc, pt, jl)
    other = checks.check_schema_and_values(hand_case(K + 1)[3], pt, jl)
    if c1.status == checks.FAIL or other.status != checks.FAIL:
        sys.exit(f'REFUSING: check C1 reads another class count than {K} '
                 f'({c1.status}, {other.status})')
    r = P101.rule_rollout('SPT', jl, pt, meta, desc)
    v = schedule_check(jl, pt, desc, r)
    if v:
        sys.exit(f'REFUSING: the environment and the validator disagree at {K} classes: '
                 f'{v[:2]}')
    return {'setup_shape': list(shape), 'cold_start_row': K, 'c1': c1.status,
            'c1_other_count': other.status, 'spt_makespan': r['makespan'], 'violations': 0}


# ---------------------------------------------------------------------------
# validation and changeover hours
# ---------------------------------------------------------------------------

def schedule_check(jl, pt, desc, r):
    """Violations of one finished schedule: check C6 (schedule_validator and
    reference_sim under the class count of the process) and the makespan against the
    last completion time."""
    from constraint_family import checks
    amch = np.asarray(r['amch'], dtype=int)
    op_ct = np.asarray(r['op_ct'], dtype=float)
    if (amch < 0).any():
        return ['an operation was never assigned a machine']
    op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
    res = checks.check_schedule_validation(desc, jl, pt, amch, op_ct, op_st)
    v = list(res.details.get('violations', []))
    if res.status == checks.FAIL and not v:
        v = list(res.messages)
    if abs(float(op_ct.max()) - float(r['makespan'])) > 1e-6:
        v.append(f'makespan {r["makespan"]} differs from the last completion {op_ct.max()}')
    return v


def changeover_hours(desc, pt, amch, op_ct):
    """(changeover hours of a schedule, the part into or out of class 4), each machine
    starting in the last row of the setup matrix (the cold-start row)."""
    if not desc.has_setup or desc.setup is None:
        return 0.0, 0.0
    S = np.asarray(desc.setup, dtype=float)
    idle = S.shape[0] - 1
    new = NEW_CLASS if S.shape[1] > NEW_CLASS else None
    amch = np.asarray(amch, dtype=int)
    st = np.asarray(op_ct, dtype=float) - pt[np.arange(pt.shape[0]), amch]
    total = part = 0.0
    for m in np.unique(amch):
        ops = np.flatnonzero(amch == m)
        last = idle
        for o in ops[np.argsort(st[ops], kind='stable')]:
            c = int(desc.op_class[o])
            s = float(S[last, c])
            total += s
            if new is not None and new in (c, last):
                part += s
            last = c
    return total, part


# ---------------------------------------------------------------------------
# level five descriptors
# ---------------------------------------------------------------------------

def setup_ranges():
    """(same-class, cross-class, cold-start) ranges of the generator as fractions of
    p_bar (draw_setup_matrix defaults, as check C4 reads them)."""
    from constraint_family import checks
    return (tuple(checks.SETUP_SAME_RANGE), tuple(checks.SETUP_CROSS_RANGE),
            tuple(checks.SETUP_COLD_RANGE))


def five_stream(size, name):
    return [FIVE_SEED, FIVE_TAG, P101.SIZES[size]['code'], P101.instance_index(name)]


def five_class_descriptor(desc, jl, pt, size, name):
    """(level-five descriptor, record of the new fields) of one stored descriptor. Uses
    no class constant, so it runs at any class count."""
    from constraint_family.descriptor import ConstraintDescriptor
    same, cross, cold = setup_ranges()
    S4 = np.asarray(desc.setup, dtype=float)
    if S4.shape != (BASE_K + 1, BASE_K):
        sys.exit(f'REFUSING: {size}/{name} holds a setup matrix of shape {S4.shape}')
    rng = np.random.default_rng(np.random.SeedSequence(five_stream(size, name)))
    J = len(jl)
    pick = np.sort(rng.choice(J, size=J // 4, replace=False))
    job_of_op = np.repeat(np.arange(J), jl)
    oc = np.asarray(desc.op_class, dtype=int).copy()
    oc[np.isin(job_of_op, pick)] = NEW_CLASS
    p_bar = float(pt[pt > 0].mean())

    def draw(rg):
        return float(max(1, int(round(rng.uniform(rg[0] * p_bar, rg[1] * p_bar)))))
    S5 = np.zeros((BASE_K + 2, BASE_K + 1))
    S5[:BASE_K, :BASE_K] = S4[:BASE_K]
    for a in range(BASE_K):
        S5[a, NEW_CLASS] = S5[NEW_CLASS, a] = draw(cross)
    S5[NEW_CLASS, NEW_CLASS] = draw(same)
    S5[BASE_K + 1, :BASE_K] = S4[BASE_K]
    S5[BASE_K + 1, NEW_CLASS] = draw(cold)
    d5 = ConstraintDescriptor(active=desc.active, lag=desc.lag, op_class=oc, setup=S5,
                              windows=desc.windows, horizon=desc.horizon,
                              setup_convention=desc.setup_convention,
                              release=desc.release, setup_mult=desc.setup_mult,
                              setup_convention_declared=desc.setup_convention_declared)
    info = {'new_class_jobs': [int(x) for x in pick], 'n_jobs': int(J),
            'new_class_ops': int(np.isin(job_of_op, pick).sum()),
            'cross_new_h': [float(x) for x in S5[:BASE_K, NEW_CLASS]],
            'same_new_h': float(S5[NEW_CLASS, NEW_CLASS]),
            'cold_new_h': float(S5[BASE_K + 1, NEW_CLASS]), 'p_bar_h': p_bar,
            'stream': five_stream(size, name)}
    return d5, info


def check_unchanged(d4, d5, info):
    """Every field of the stored descriptor outside class 4 is unchanged."""
    S4, S5 = np.asarray(d4.setup, float), np.asarray(d5.setup, float)
    keep = d5.op_class != NEW_CLASS
    ok = (np.array_equal(S5[:BASE_K, :BASE_K], S4[:BASE_K])
          and np.array_equal(S5[BASE_K + 1, :BASE_K], S4[BASE_K])
          and np.array_equal(S5[:BASE_K + 1, :BASE_K + 1], S5[:BASE_K + 1, :BASE_K + 1].T)
          and np.array_equal(d5.op_class[keep], d4.op_class[keep])
          and int((~keep).sum()) == info['new_class_ops']
          and d5.active == d4.active and np.array_equal(d5.lag, d4.lag)
          and d5.windows == d4.windows and d5.horizon == d4.horizon)
    if not ok:
        sys.exit('REFUSING: the level-five descriptor changes a stored field')


def five_desc_path(root, size, combo, name):
    return os.path.join(root, 'data', 'five', size, 'desc', combo, f'{name}.desc.json')


def manifest_path(root, size):
    return os.path.join(root, 'data', 'five', size, 'manifest.json')


def build(args):
    """Level-five descriptors under --root, each checked by C1 to C5 at five classes."""
    from constraint_family import checks
    set_class_count(BASE_K + 1)
    same, cross, cold = setup_ranges()
    for size in args.sizes:
        names = P101.names_of(size, args.n)
        digest = hashlib.sha256()
        per_inst, c4 = {}, {c: 0 for c in args.combos}
        for combo in args.combos:
            for name, jl, pt, meta, d4 in P101.load_cell(size, combo, args.n):
                d5, info = five_class_descriptor(d4, jl, pt, size, name)
                check_unchanged(d4, d5, info)
                src = P102.stored_desc_path(size, combo, name)
                with open(src) as f:
                    rec = json.load(f)
                out = d5.to_json()
                for k in ('base_instance', 'regime', 'desc_seed'):
                    if k in rec:
                        out[k] = rec[k]
                out['level'] = {
                    'level': 'five', 'n_classes': BASE_K + 1, 'new_class': NEW_CLASS,
                    'rule': (f'floor(J/4) modules form class {NEW_CLASS}; the stored block '
                             f'and the stored cold-start row are kept, the cold-start row '
                             f'last; the new column and row are symmetric, cross-class '
                             f'U[{cross[0]}, {cross[1]}] p_bar, same-class and cold start '
                             f'U[{same[0]}, {same[1]}] p_bar, max(1, round(.)) h'),
                    'source_desc': os.path.relpath(src, ROOT_DIR),
                    'source_sha256': sha256_file(src), **info}
                txt = json.dumps(out)
                P102.write_once(five_desc_path(args.root, size, combo, name), txt)
                digest.update(txt.encode())
                rep = {r.id: r for r in checks.run_all(d5, pt, False, job_length=jl)}
                bad = [c for c in ('C1', 'C2', 'C3', 'C5') if rep[c].status == checks.FAIL]
                if bad:
                    sys.exit(f'REFUSING: {size}/{combo}/{name} fails {bad} at five classes: '
                             f'{[rep[c].messages[:1] for c in bad]}')
                c4[combo] += rep['C4'].status == checks.WARN
                key = {k: info[k] for k in ('new_class_jobs', 'cross_new_h', 'same_new_h',
                                            'cold_new_h')}
                key['setup'] = d5.setup.astype(int).tolist()
                if name in per_inst and per_inst[name] != key:
                    sys.exit(f'REFUSING: {size}/{name} has another five-class matrix in '
                             f'{combo}; the stored matrices differ between combinations')
                per_inst[name] = key
        vals = list(per_inst.values())
        man = {'level': 'five', 'size': size, 'n_instances': len(names), 'instances': names,
               'combinations': list(args.combos), 'n_classes': BASE_K + 1,
               'new_class': NEW_CLASS, 'stream': f'[{FIVE_SEED}, {FIVE_TAG}, size code, '
                                                  f'instance index]',
               'ranges_frac_p_bar': {'same': list(same), 'cross': list(cross),
                                     'cold': list(cold)},
               'new_class_modules': int(len(vals[0]['new_class_jobs'])),
               'modules': int(P101.SIZES[size]['modules']),
               'cross_new_mean_h': float(np.mean([v['cross_new_h'] for v in vals])),
               'same_new_mean_h': float(np.mean([v['same_new_h'] for v in vals])),
               'cold_new_mean_h': float(np.mean([v['cold_new_h'] for v in vals])),
               'c4_warn_share': {c: k / len(names) for c, k in c4.items()},
               'per_instance': per_inst, 'content_sha256': digest.hexdigest(),
               'commit': P101.git_commit()}
        mp = manifest_path(args.root, size)
        if os.path.exists(mp):
            with open(mp) as f:
                if json.load(f).get('content_sha256') != man['content_sha256']:
                    sys.exit(f'REFUSING: {mp} records different content')
        write_json_atomic(mp, man)
        print(f'[p103 build] {size}: {len(names)} instances x {len(args.combos)} '
              f'combinations at five classes; {man["new_class_modules"]} of '
              f'{man["modules"]} modules in class {NEW_CLASS}; new changeovers: cross '
              f'{man["cross_new_mean_h"]:.2f} h, same {man["same_new_mean_h"]:.2f} h, cold '
              f'{man["cold_new_mean_h"]:.2f} h on average; C4 warnings '
              + ', '.join(f'{c} {v:.2f}' for c, v in man['c4_warn_share'].items()),
              flush=True)


# ---------------------------------------------------------------------------
# units, rollouts, evaluate
# ---------------------------------------------------------------------------

def model_name(arm, combo, seed, environment='exact'):
    reg_arm = {'pact': 'released', 'spec': 'specialist'}[arm]
    return REG.PREFIX + REG.model(reg_arm, combo if arm == 'spec' else None, seed,
                                  environment)


def units_for(level, size, combo, arms, seeds, environment='exact'):
    out = []
    for arm in arms:
        if arm == 'rules':
            out += [(level, size, combo, 'rule', None, r) for r in RULES7]
        else:
            out += [(level, size, combo, arm, s,
                     'greedy+' + model_name(arm, combo, s, environment)) for s in seeds]
    return out


def all_units(levels, sizes, combos, arms, seeds, environment='exact'):
    return [u for lv in levels for sz in sizes for c in combos
            for u in units_for(lv, sz, c, arms, seeds, environment)]


def result_path(root, environment, level, size, combo, label):
    """Result file of one unit. The rules evaluate in the exact-mask environment
    whatever the run names of the policy arms, so their files sit under 'exact'."""
    env_dir = 'exact' if label in RULES7 else environment
    return os.path.join(root, 'results', env_dir, level, size, combo, f'{label}.npz')


def load_level_cell(root, level, size, combo, n):
    """[(name, jl, pt, meta, desc)] of one cell of a level, each descriptor checked for
    the setup shape of its level."""
    if level == 'default':
        cell = P101.load_cell(size, combo, n)
    else:
        from constraint_family.descriptor import ConstraintDescriptor
        from ppvc_instance_generator import load_instance
        cell = []
        for name in P101.names_of(size, n):
            p = five_desc_path(root, size, combo, name)
            if not os.path.exists(p):
                sys.exit(f'REFUSING: {p} is missing; run the build phase first')
            with open(p) as f:
                rec = json.load(f)
            jl, pt, meta = load_instance(os.path.join(ROOT_DIR, P101.SIZES[size]['base'],
                                                      name))
            cell.append((name, jl, pt, meta,
                         ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    K = LEVEL_K[level]
    for name, _, _, _, d in cell:
        if np.shape(d.setup) != (K + 1, K):
            sys.exit(f'REFUSING: {size}/{combo}/{name} holds setups {np.shape(d.setup)}, '
                     f'level {level} needs {(K + 1, K)}')
    return cell


def rollout(arm, label, jl, pt, meta, desc, runner):
    if arm == 'rule' and label in RULES4:
        return P101.rule_rollout(label, jl, pt, meta, desc)
    if arm == 'rule':
        return P102.aware_rollout(label, jl, pt, meta, desc)
    return runner.rollout(label[len('greedy+'):], jl, pt, meta, desc)


def evaluate_unit(root, environment, unit, cell, runner, force):
    level, size, combo, arm, seed, label = unit
    K = LEVEL_K[level]
    path = result_path(root, environment, level, size, combo, label)
    names = [c[0] for c in cell]
    if os.path.exists(path) and os.path.exists(path[:-4] + '.json') and not force:
        with open(path[:-4] + '.json') as f:
            old = json.load(f)
        if old.get('instances') != names or old.get('class_count') != K:
            sys.exit(f'REFUSING: {path} covers other instances or another class count; '
                     f'use --force or another --root')
        if arm != 'rule':
            ckpt = os.path.join(ROOT_DIR, 'trained_network', 'FAMILY',
                                f'{label[len("greedy+"):]}.pth')
            if old.get('policy', {}).get('checkpoint_sha256') != sha256_file(ckpt):
                sys.exit(f'REFUSING: {path} was written by another checkpoint of the same '
                         f'run; use --force or another --root')
        return 'skip', 0.0
    assert_class_count(K)
    t0, c0 = time.perf_counter(), time.process_time()
    n, N = len(cell), cell[0][2].shape[0]
    arr = {k: np.zeros(n) for k in ('makespan', 'lb0', 'util', 'ms_over_lb0', 'setup_h',
                                    'setup_new_h')}
    arr['amch'] = np.zeros((n, N), dtype=np.int16)
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        r = rollout(arm, label, jl, pt, meta, desc, runner)
        v = schedule_check(jl, pt, desc, r)
        if v:
            sys.exit(f'REFUSING: {label} on {level}/{size}/{combo}/{name} is infeasible at '
                     f'{K} classes: {v[:2]}')
        u, ml, _ = P101.shop_load(pt, r['amch'], r['makespan'], r['lb0'])
        sh, sn = changeover_hours(desc, pt, r['amch'], r['op_ct'])
        for k, val in (('makespan', r['makespan']), ('lb0', r['lb0']), ('util', u),
                       ('ms_over_lb0', ml), ('setup_h', sh), ('setup_new_h', sn)):
            arr[k][i] = val
        arr['amch'][i] = r['amch']
    assert_class_count(K)
    P101.save_npz_atomic(path, **arr)
    rr = P101.run_record({'unit': list(unit)}, names, t0, c0)
    rr.update({'level': level, 'size': size, 'regime': combo, 'arm': arm, 'seed': seed,
               'label': label, 'mask': P101.MASK, 'environment': environment,
               'class_count': K, 'validated': n, 'violations': 0,
               'validator': f'checks.check_schedule_validation (C6) and the makespan, at '
                            f'{K} classes'})
    if runner is not None and arm != 'rule':
        rr['policy'] = runner.load(label[len('greedy+'):])[1]
    write_json_atomic(path[:-4] + '.json', rr)
    return 'done', rr['cpu_s']


def stored_available(unit, environment):
    """True for a default-level policy unit whose exact-campaign array exists."""
    level, size, combo, arm, seed, label = unit
    return (level == 'default' and arm != 'rule' and environment == 'exact'
            and os.path.exists(stored_policy_path(size, combo, label, environment)))


def units_needed(units, args):
    """The units to evaluate. With --default_policy stored, a default-level policy unit
    whose exact-campaign array exists is read from that array by infer and is left out."""
    if args.default_policy != 'stored':
        return units
    return [u for u in units if not stored_available(u, args.environment)]


def unit_cost(u, n):
    return COST_S[u[1]]['rule' if u[3] == 'rule' else 'policy'] * (n or P101.SIZES[u[1]]['n'])


def assign_parts(units, parts, n):
    order = sorted(range(len(units)), key=lambda i: (-unit_cost(units[i], n), i))
    loads, out = [0.0] * parts, [[] for _ in range(parts)]
    for i in order:
        p = int(np.argmin(loads))
        out[p].append(i)
        loads[p] += unit_cost(units[i], n)
    return [sorted(x) for x in out], loads


def plan(args):
    py = sys.executable
    total = 0.0
    for lv in args.levels:
        every = all_units([lv], args.sizes, args.combos, args.arms, args.seeds,
                          args.environment)
        units = units_needed(every, args)
        if not units:
            print(f'[p103 plan] level {lv}: nothing to evaluate')
            continue
        parts, loads = assign_parts(units, args.parts, args.n)
        cost = sum(unit_cost(u, args.n) for u in units)
        total += cost
        print(f'[p103 plan] level {lv}: {len(units)} result files ({len(every) - len(units)} '
              f'read from stored arrays), estimated '
              f'{cost / 3600:.2f} core-hours of CPU time (COST_S), {args.parts} parts of '
              f'{min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h')
        for i in range(args.parts):
            print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c '
                  f'<CORE_{i}> {py} -B scripts/p103_five_classes.py evaluate --level {lv} '
                  f'--arms {",".join(args.arms)} --seeds {",".join(map(str, args.seeds))} '
                  f'--default_policy {args.default_policy} --part {i} --parts {args.parts}')
    print(f'[p103 plan] total {total / 3600:.2f} core-hours')
    return dict(core_h=total / 3600)


def evaluate(args):
    if len(args.levels) != 1:
        sys.exit('REFUSING: one level per process (--level default or --level five)')
    level = args.levels[0]
    info = set_class_count(LEVEL_K[level])
    print(f'[p103 eval] class count {info["class_count"]} in {", ".join(PATCHED)}; probe '
          f'{info["probe"]}', flush=True)
    units = units_needed(all_units([level], args.sizes, args.combos, args.arms, args.seeds,
                                   args.environment), args)
    if args.part is not None:
        parts, _ = assign_parts(units, args.parts, args.n)
        units = [units[i] for i in parts[args.part]]
    runner = P101.PolicyRunner() if any(a in POLICY_ARMS for a in args.arms) else None
    cache = {}
    t_all = time.perf_counter()
    for k, u in enumerate(units):
        key = u[:3]
        if key not in cache:
            cache.clear()
            cache[key] = load_level_cell(args.root, *key, args.n)
        st, c = evaluate_unit(args.root, args.environment, u, cache[key], runner, args.force)
        print(f'[p103 eval {k + 1}/{len(units)}] {u[0]:7s} {u[1]:9s} {u[2]:4s} {u[5]:40s} '
              f'{st} cpu={c:7.1f}s elapsed={time.perf_counter() - t_all:7.0f}s', flush=True)


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def read_unit(root, environment, level, size, combo, label, names, tally):
    p = result_path(root, environment, level, size, combo, label)
    if not (os.path.exists(p) and os.path.exists(p[:-4] + '.json')):
        refuse(f'result {level}/{size}/{combo}/{label} is missing ({p})')
    with open(p[:-4] + '.json') as f:
        rec = json.load(f)
    want = {'level': level, 'size': size, 'regime': combo, 'label': label,
            'class_count': LEVEL_K[level]}
    bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
    if bad or rec.get('instances') != names:
        refuse(f'{p}: record {bad or "instance list"} does not match the study')
    if rec.get('violations') != 0 or rec.get('validated') != len(names):
        refuse(f'{p}: {rec.get("violations")} violations over {rec.get("validated")} '
               f'validated schedules, the study needs 0 over {len(names)}')
    tally['validated'] += rec['validated']
    tally['violations'] += rec['violations']
    with np.load(p) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def stored_policy_path(size, combo, label, environment):
    return os.path.join(ROOT_DIR, P101.RESULTS_ROOT,
                        f'{size}{REG.FOLDER_SUFFIX[environment]}',
                        f'Result_{label}_{combo}.npy')


def stored_policy_array(size, combo, label, environment, n):
    p = stored_policy_path(size, combo, label, environment)
    if not os.path.exists(p):
        return None
    a = np.load(p)
    a = (a[:, 0] if a.ndim > 1 else a).astype(float)
    if len(a) < n:
        refuse(f'{p} has {len(a)} rows, the study needs {n}')
    return a[:n]


def policy_makespans(args, level, size, combo, arm, names, tally):
    """[k, n] makespans of one policy arm and the source of each run's array."""
    rows, src = [], []
    for s in args.seeds:
        label = 'greedy+' + model_name(arm, combo, s, args.environment)
        # the legacy runs' stored arrays come from the legacy mask, so only the exact
        # campaign's arrays are a reference of this study
        stored = (stored_policy_array(size, combo, label, args.environment, len(names))
                  if level == 'default' and args.environment == 'exact' else None)
        p = result_path(args.root, args.environment, level, size, combo, label)
        have_run = os.path.exists(p) and os.path.exists(p[:-4] + '.json')
        if args.default_policy == 'stored' and stored is not None and not have_run:
            rows.append(stored)
            src.append('stored')
            continue
        a = read_unit(args.root, args.environment, level, size, combo, label, names,
                      tally)['makespan']
        if stored is not None and not np.array_equal(stored, a):
            refuse(f'{size}/{combo}: the default level of {label} differs from the stored '
                   f'array on {int((stored != a).sum())} instances; the pipelines diverge')
        rows.append(a)
        src.append('run, equal to the stored array' if stored is not None else 'run')
    return np.stack(rows), src


def policy_block(st, gaps, levels, combos, nboot):
    """Comparisons per level and combination, pooled over the combinations, and the
    shift from level default to level five, with verdict counts and changes."""
    SF = st[0]
    out = {}
    for lv in levels:
        G = gaps[lv]
        rows = {'pol_spec': {r: P102.component(SF, G[r]['pol_spec'], nboot)[0]
                             for r in combos}}
        for key in ('pol_rule', 'pol_rule7'):
            rows[key] = P102.superiority(st, {r: G[r][key] for r in combos}, nboot)
        pooled = {key: P102.component(SF, np.stack([G[r][key] for r in combos], axis=2),
                                      nboot)[0] for key in COMPARISONS}
        counts = {f'pol_spec|{mg}': {v: [rows['pol_spec'][r][f'v{mg}']
                                         for r in combos].count(v) for v in SF.VERDICTS}
                  for mg in MARGINS}
        for key in ('pol_rule', 'pol_rule7'):
            counts[key] = {v: [rows[key][r]['verdict'] for r in combos].count(v)
                           for v in st[1]}
        out[lv] = {'rows': rows, 'pooled': pooled, 'counts': counts}
    if 'five' in levels:
        sh = {key: {r: P102.component(SF, gaps['five'][r][key] - gaps['default'][r][key],
                                      nboot)[0] for r in combos} for key in COMPARISONS}
        for key in COMPARISONS:
            sh[key]['pooled'] = P102.component(SF, np.stack(
                [gaps['five'][r][key] - gaps['default'][r][key] for r in combos], axis=2),
                nboot)[0]
        out['shift'] = sh
        ch = {}
        for key in COMPARISONS:
            cols = [f'v{mg}' for mg in MARGINS] if key == 'pol_spec' else ['verdict']
            for col in cols:
                ch[f'{key}|{col}'] = [r for r in combos
                                      if out['five']['rows'][key][r][col]
                                      != out['default']['rows'][key][r][col]]
        out['changed'] = ch
    return out


def infer(args):
    """Every outcome of the module doc. Returns the result dictionary."""
    partial = bool(args.partial)
    arms = list(args.arms)
    full_set = (args.levels == list(LEVELS) and args.sizes == list(SIZES)
                and args.combos == list(COMBOS) and not args.n)
    if args.n and not partial:
        refuse('--n (a subset of the instances) needs --partial')
    if not partial and not full_set:
        refuse('every level, size and combination is required; a subset needs --partial')
    if 'default' not in args.levels:
        refuse('the default level is the reference of every shift; it is required')
    in_place_ok = (not partial and args.environment == 'exact'
                   and (arms == ['rules'] or sorted(arms) == sorted(ARM_CHOICES)))
    if args.out is None:
        if not in_place_ok:
            refuse('--partial, --environment legacy, or arms other than "rules" or every '
                   'arm, write only to --out <folder>')
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(args.out)
        P101.guard_out(out_root)
    paths = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
    with_policy = all(a in arms for a in POLICY_ARMS)
    if any(a in arms for a in POLICY_ARMS) and not with_policy:
        refuse('the policy comparisons need the arms pact and spec together')
    if with_policy and 'rules' not in arms:
        refuse('the policy comparisons against the rules need --arms with rules')
    if args.default_policy == 'stored' and args.environment != 'exact':
        refuse('--default_policy stored reads the exact campaign\'s arrays; it needs '
               '--environment exact')
    if with_policy and len(args.seeds) < 2 and not partial:
        refuse('at least two training seeds are required (the unit of the hierarchical '
               'bootstrap)')
    st = P102._stats() if with_policy else None
    res = {'generator': GENERATOR, 'partial': partial, 'arms': arms,
           'seeds': list(args.seeds), 'environment': args.environment, 'mask': P101.MASK,
           'B': args.boot, 'levels': list(args.levels),
           'class_count': {lv: LEVEL_K[lv] for lv in args.levels},
           'default_policy': args.default_policy,
           'convention': 'd = 100 (first - second) / second, seed s against seed s, per '
                         'base instance; negative means the first-named method is shorter',
           'best_rule': 'per-instance minimum of ' + ', '.join(RULES4),
           'best_rule7': 'per-instance minimum of ' + ', '.join(RULES7),
           'sizes': {}}
    for size in args.sizes:
        names = P101.names_of(size, args.n)
        tally = {'validated': 0, 'violations': 0}
        S = {'n': len(names), 'combos': {}}
        if 'five' in args.levels:
            mp = manifest_path(args.root, size)
            if not os.path.exists(mp):
                refuse(f'{mp} is missing; run the build phase first')
            with open(mp) as f:
                man = json.load(f)
            if man.get('instances', [])[:len(names)] != names:
                refuse(f'{mp} covers other instances')
            S['manifest'] = {k: v for k, v in man.items()
                             if k not in ('instances', 'per_instance')}
        gaps = {lv: {} for lv in args.levels}
        for combo in args.combos:
            C = {'levels': {}}
            for lv in args.levels:
                R = ({r: read_unit(args.root, args.environment, lv, size, combo, r, names,
                                   tally) for r in RULES7} if 'rules' in arms else {})
                L = {}
                if R:
                    ms = {r: R[r]['makespan'] for r in RULES7}
                    b4 = np.min(np.stack([ms[r] for r in RULES4]), axis=0)
                    b7 = np.min(np.stack([ms[r] for r in RULES7]), axis=0)
                    sp = R['SPT']
                    L.update(rule_ms={r: float(ms[r].mean()) for r in RULES7},
                             best_rule_ms=float(b4.mean()), best_rule7_ms=float(b7.mean()),
                             setup_spt_h=float(sp['setup_h'].mean()),
                             setup_new_share_spt=100.0 * float(
                                 sp['setup_new_h'].sum() / max(sp['setup_h'].sum(), 1e-9)),
                             util_spt=100.0 * float(sp['util'].mean()))
                    L['_ms'], L['_b4'], L['_b7'] = ms, b4, b7
                    if lv == 'default':
                        L['stored_check'] = P102.check_stored(size, combo, ms, len(names))
                if with_policy:
                    P, src = {}, {}
                    for a in POLICY_ARMS:
                        P[a], src[a] = policy_makespans(args, lv, size, combo, a, names,
                                                        tally)
                    d = {'pol_spec': 100.0 * (P['pact'] - P['spec']) / P['spec'],
                         'pol_rule': 100.0 * (P['pact'] - L['_b4'][None]) / L['_b4'][None],
                         'pol_rule7': 100.0 * (P['pact'] - L['_b7'][None]) / L['_b7'][None]}
                    gaps[lv][combo] = d
                    L['policy_ms'] = {a: float(P[a].mean()) for a in POLICY_ARMS}
                    L['policy_source'] = src
                C['levels'][lv] = L
            if 'rules' in arms and 'five' in args.levels:
                D0, D1 = C['levels']['default'], C['levels']['five']
                C['rule_shift'] = {
                    'spt': P101.ci(P101.gap(D1['_ms']['SPT'], D0['_ms']['SPT'])),
                    'best_rule': P101.ci(P101.gap(D1['_b4'], D0['_b4'])),
                    'best_rule7': P101.ci(P101.gap(D1['_b7'], D0['_b7']))}
            for L in C['levels'].values():
                for k in ('_ms', '_b4', '_b7'):
                    L.pop(k, None)
            S['combos'][combo] = C
        if with_policy:
            S['policy'] = policy_block(st, gaps, args.levels, args.combos, args.boot)
        if tally['violations']:
            refuse(f'{size}: {tally["violations"]} violations')
        S['validated'], S['violations'] = tally['validated'], tally['violations']
        res['sizes'][size] = S
    lines = macro_lines(res)
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    check_macro_names(names_m, paths)
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

def _tex_verdict(v):
    import stats_family as SF
    return SF.TEX.get(v, v)


def macro_lines(res):
    mark = 'PARTIAL ' if res['partial'] else ''
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% Five setup classes (level Five) against the stored four (level Four), exact '
         'mask, at 10, 50 and 80 modules with 25 stations, combinations S, LS, SW, LSW; '
         'a quarter of the modules (rounded down) of every instance form the fifth class.',
         '% Gaps in %: first-named method minus second over second, negative means the '
         'first-named method is shorter; best classical rule: per-instance minimum of '
         'FIFO, MOR, SPT and MWKR restricted to the legal pairs.']
    if res['partial']:
        L.insert(1, '% PARTIAL: subset of levels, sizes, combinations or instances; not '
                    'for the manuscript.')
    for size, S in res['sizes'].items():
        w = P101.SIZES[size]['word']
        L.append(f'\\newcommand{{\\{PREFIX}N{w}}}{{{S["n"]}}}'
                 f'\\newcommand{{\\{PREFIX}Validated{w}}}{{{S["validated"]}}}'
                 f'\\newcommand{{\\{PREFIX}Violations{w}}}{{{S["violations"]}}}'
                 f'  % {mark}{size}: instances; schedules validated at their class count '
                 f'and the violations found')
        man = S.get('manifest')
        if man:
            L.append(f'\\newcommand{{\\{PREFIX}NewMods{w}}}{{{man["new_class_modules"]}}}'
                     f'\\newcommand{{\\{PREFIX}Mods{w}}}{{{man["modules"]}}}'
                     f'\\newcommand{{\\{PREFIX}CrossNew{w}}}{{{u1(man["cross_new_mean_h"])}}}'
                     f'\\newcommand{{\\{PREFIX}SameNew{w}}}{{{u1(man["same_new_mean_h"])}}}'
                     f'\\newcommand{{\\{PREFIX}ColdNew{w}}}{{{u1(man["cold_new_mean_h"])}}}'
                     f'  % {mark}{size}: modules in the fifth class and modules per '
                     f'instance; mean new changeover between the fifth class and another '
                     f'class, same-class reset and cold start of the fifth class (h)')
        for combo, C in S['combos'].items():
            for lv, D in C['levels'].items():
                lw = LEVEL_WORDS[lv]
                if 'rule_ms' in D:
                    s = (f'\\newcommand{{\\{PREFIX}MsSpt{lw}{w}{combo}}}'
                         f'{{{u1(D["rule_ms"]["SPT"])}}}'
                         f'\\newcommand{{\\{PREFIX}MsRule{lw}{w}{combo}}}'
                         f'{{{u1(D["best_rule_ms"])}}}'
                         f'\\newcommand{{\\{PREFIX}MsRuleSeven{lw}{w}{combo}}}'
                         f'{{{u1(D["best_rule7_ms"])}}}'
                         f'\\newcommand{{\\{PREFIX}SetupSpt{lw}{w}{combo}}}'
                         f'{{{u1(D["setup_spt_h"])}}}')
                    if lv == 'five':
                        s += (f'\\newcommand{{\\{PREFIX}SetupNewShare{w}{combo}}}'
                              f'{{{u1(D["setup_new_share_spt"])}}}')
                    L.append(s + f'  % {mark}{size} {combo}, level {lv}: mean makespan (h) '
                                 f'of SPT, of the best classical rule, of the best of seven '
                                 f'rules; changeover hours of the SPT schedule'
                             + (' and the share involving the fifth class (%)'
                                if lv == 'five' else ''))
            if 'rule_shift' in C:
                for key, word in (('spt', 'Spt'), ('best_rule', 'Rule'),
                                  ('best_rule7', 'RuleSeven')):
                    g = C['rule_shift'][key]
                    L.append(f'\\newcommand{{\\{PREFIX}{word}Shift{w}{combo}}}{{{f2(g["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{word}ShiftLo{w}{combo}}}{{{f2(g["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{word}ShiftHi{w}{combo}}}{{{f2(g["hi"])}}}'
                             f'  % {mark}{size} {combo}: change of the {key.replace("_", " ")} '
                             f'makespan from four to five classes, %, mean and 95% interval '
                             f'over {g["n"]} instances')
        if 'policy' in S:
            P = S['policy']
            for lv in res['levels']:
                lw = LEVEL_WORDS[lv]
                for key, (first, second, cw) in COMPARISONS.items():
                    for r in list(S['combos']) + ['Pool']:
                        c = P[lv]['pooled'][key] if r == 'Pool' else P[lv]['rows'][key][r]
                        s = (f'\\newcommand{{\\{PREFIX}{cw}{lw}{w}{r}}}{{{f2(c["mean"])}}}'
                             f'\\newcommand{{\\{PREFIX}{cw}Lo{lw}{w}{r}}}{{{f2(c["lo"])}}}'
                             f'\\newcommand{{\\{PREFIX}{cw}Hi{lw}{w}{r}}}{{{f2(c["hi"])}}}')
                        if key == 'pol_spec':
                            s += ''.join(f'\\newcommand{{\\{PREFIX}{cw}V{MW[mg]}{lw}{w}{r}}}'
                                         f'{{{_tex_verdict(c[f"v{mg}"])}}}' for mg in MARGINS)
                        elif r != 'Pool':
                            s += (f'\\newcommand{{\\{PREFIX}{cw}Verdict{lw}{w}{r}}}'
                                  f'{{{c["verdict"]}}}')
                        s += (f'\\newcommand{{\\{PREFIX}{cw}Dir{lw}{w}{r}}}'
                              f'{{{_tex_verdict(c["direction"])}}}')
                        L.append(s + f'  % {mark}{size} {r}, level {lv}: {first} vs {second}, '
                                     f'{c["scheme"]} 95% interval over {c["k"]} runs and '
                                     f'{c["n"]} instances' + (f', {c["c"]} combinations'
                                                              if c['c'] > 1 else ''))
            if 'shift' in P:
                for key, (first, second, cw) in COMPARISONS.items():
                    for r in list(S['combos']) + ['Pool']:
                        c = P['shift'][key]['pooled' if r == 'Pool' else r]
                        L.append(f'\\newcommand{{\\{PREFIX}{cw}Shift{w}{r}}}{{{f2(c["mean"])}}}'
                                 f'\\newcommand{{\\{PREFIX}{cw}ShiftLo{w}{r}}}{{{f2(c["lo"])}}}'
                                 f'\\newcommand{{\\{PREFIX}{cw}ShiftHi{w}{r}}}{{{f2(c["hi"])}}}'
                                 f'  % {mark}{size} {r}: change of the {first} vs {second} gap '
                                 f'from four to five classes, percentage points')
                s, what = '', []
                for k2, lst in P['changed'].items():
                    key, col = k2.split('|')
                    cw = COMPARISONS[key][2]
                    mw = MW[float(col[1:])] if col != 'verdict' else ''
                    s += f'\\newcommand{{\\{PREFIX}{cw}Changed{mw}{w}}}{{{len(lst)}}}'
                    what.append(f'{cw}{mw}: {",".join(lst) or "none"}')
                L.append(s + f'  % {mark}{size}: combinations whose verdict differs between '
                             f'four and five classes ({"; ".join(what)})')
    return L


def check_macro_names(names, out_paths):
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    import glob
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
    L = [f'# Five setup classes ({GENERATOR})', '']
    if res['partial']:
        L += ['PARTIAL: a subset of levels, sizes, combinations or instances. Not for the '
              'manuscript.', '']
    L += [f'Exact mask, {res["environment"]} run names. Arms {", ".join(res["arms"])}; '
          f'training seeds {",".join(map(str, res["seeds"]))}. Level default: the stored '
          'descriptors, four classes. Level five: a quarter of the modules (rounded down) '
          'form a fifth class whose changeovers are drawn from the generator\'s ranges. '
          'Best rule: ' + res['best_rule'] + '. Shift: 100 (five - default) / default per '
          'instance, 95% bootstrap over instances.', '']

    def g(x):
        return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'
    for size, S in res['sizes'].items():
        L += [f'## {size} ({S["n"]} instances; {S["validated"]} schedules validated, '
              f'{S["violations"]} violations)', '']
        man = S.get('manifest')
        if man:
            L += [f'Fifth class: {man["new_class_modules"]} of {man["modules"]} modules; '
                  f'new changeovers on average cross {u2(man["cross_new_mean_h"])} h, same '
                  f'{u2(man["same_new_mean_h"])} h, cold start {u2(man["cold_new_mean_h"])} h; '
                  f'C4 warning share ' + ', '.join(f'{c} {v:.2f}' for c, v in
                                                   man['c4_warn_share'].items()) + '.', '']
        L += ['| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | '
              'SPT changeover (h) | fifth-class share (%) | default level equals stored '
              'arrays |', '|' + '---|' * 8]
        for combo, C in S['combos'].items():
            for lv, D in C['levels'].items():
                if 'rule_ms' not in D:
                    continue
                chk = D.get('stored_check')
                L.append(f'| {combo} | {lv} | {u1(D["rule_ms"]["SPT"])} | '
                         f'{u1(D["best_rule_ms"])} | {u1(D["best_rule7_ms"])} | '
                         f'{u1(D["setup_spt_h"])} | {u1(D["setup_new_share_spt"])} | '
                         + (', '.join(f'{r}: {v}' for r, v in chk.items()) if chk else '')
                         + ' |')
        if any('rule_shift' in C for C in S['combos'].values()):
            L += ['', '| combination | SPT shift (%) | best classical rule shift (%) | '
                  'best of seven shift (%) |', '|---|---|---|---|']
            for combo, C in S['combos'].items():
                if 'rule_shift' in C:
                    L.append(f'| {combo} | {g(C["rule_shift"]["spt"])} | '
                             f'{g(C["rule_shift"]["best_rule"])} | '
                             f'{g(C["rule_shift"]["best_rule7"])} |')
        if 'policy' in S:
            P = S['policy']
            L += ['', '| level | combination | policy - specialist | '
                  'policy - best classical rule | policy - best of seven |', '|' + '---|' * 5]
            for lv in res['levels']:
                for r in list(S['combos']) + ['pooled']:
                    def cell(key):
                        c = P[lv]['pooled'][key] if r == 'pooled' else P[lv]['rows'][key][r]
                        tail = (f' {c["v0.5"]} / {c["v1.0"]}' if key == 'pol_spec'
                                else (f' {c["verdict"]}' if 'verdict' in c
                                      else f' direction {c["direction"]}'))
                        return g(c) + tail
                    L.append(f'| {lv} | {r} | ' + ' | '.join(cell(k) for k in COMPARISONS) + ' |')
            if 'shift' in P:
                L += ['', 'Shift of each gap from four to five classes (percentage points):',
                      '', '| combination | ' + ' | '.join(
                          f'{a} - {b}' for a, b, _ in COMPARISONS.values()) + ' |',
                      '|' + '---|' * (1 + len(COMPARISONS))]
                for r in list(S['combos']) + ['pooled']:
                    L.append(f'| {r} | ' + ' | '.join(g(P['shift'][k][r]) for k in COMPARISONS)
                             + ' |')
                L += ['', 'Verdicts that change: ' + '; '.join(
                    f'{k}: {",".join(v) or "none"}' for k, v in P['changed'].items())]
            src = {f'{c}/{lv}': D.get('policy_source') for c, C in S['combos'].items()
                   for lv, D in C['levels'].items() if D.get('policy_source')}
            if src:
                L += ['', 'Policy arrays: ' + '; '.join(
                    f'{k}: {", ".join(sorted(set(x for v in s.values() for x in v)))}'
                    for k, s in src.items())]
        L.append('')
    return L


# ---------------------------------------------------------------------------
# check: the hand case and the policy input at a given class count
# ---------------------------------------------------------------------------

def replay(jl, pt, meta, desc, seq):
    """Environment after the actions seq, the machine of each operation, and whether
    the schedule is complete."""
    M = pt.shape[1]
    env = P101.make_env(jl, pt, meta, [desc])
    amch = np.full(pt.shape[0], -1, dtype=int)
    for a in seq:
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    return env, amch


def reachable_schedules(jl, pt, meta, desc):
    """Every complete schedule the mask admits on a small instance, by depth-first
    replay: [(machine-0 order, makespan, violations, changeover hours, record)]."""
    out, stack = [], [[]]
    while stack:
        seq = stack.pop()
        env, amch = replay(jl, pt, meta, desc, seq)
        if env.done().all():
            r = dict(makespan=float(env.current_makespan[0]), amch=amch,
                     op_ct=env.true_op_ct[0].astype(float).copy())
            st = r['op_ct'] - pt[np.arange(pt.shape[0]), amch]
            m0 = [int(o) for o in np.flatnonzero(amch == 0)[np.argsort(st[amch == 0])]]
            out.append((m0, r['makespan'], schedule_check(jl, pt, desc, r),
                        changeover_hours(desc, pt, amch, r['op_ct']), r))
            continue
        legal = np.flatnonzero(~np.asarray(env.dynamic_pair_mask, dtype=bool)[0].reshape(-1))
        stack += [seq + [int(a)] for a in legal[::-1]]
    return out


def hand_report(K):
    """The hand case at the class count of the process. At K = 5 the reachable
    schedules with their makespans, violations and changeover hours, the same for a
    cheaper fifth-class changeover, and the validator on a schedule that shortens the
    fifth-class changeover. At K = 4 the five-class hand descriptor offered to check
    C1 and to the environment."""
    from constraint_family import checks
    out = {'class_count': K}
    if K == BASE_K:
        jl, pt, meta, d5 = hand_case(BASE_K + 1)
        out['c1_five'] = checks.check_schema_and_values(d5, pt, jl).status
        try:
            P101.make_env(jl, pt, meta, [d5])
            out['env_five'] = 'accepted'
        except Exception as e:  # noqa: BLE001
            out['env_five'] = f'refused: {type(e).__name__}'
        return out
    for c in (5, 2):
        jl, pt, meta, desc = hand_case(K, cross_new=c)
        rows = reachable_schedules(jl, pt, meta, desc)
        out[f'cross{c}'] = [{'m0_order': m0, 'makespan': ms, 'violations': len(v),
                             'changeover_h': list(h),
                             'op_ct': [float(x) for x in r['op_ct']]}
                            for m0, ms, v, h, r in rows]
        # machine 0 in the order op1, op0, built by hand: op1 ends at 2 + 3 = 5, op0 at
        # 5 + c + 2; op2 on machine 1 ends at 3
        amch = np.array([0, 0, 1])
        rev = dict(amch=amch, op_ct=np.array([7.0 + c, 5.0, 3.0]), makespan=7.0 + c)
        out[f'reverse{c}'] = {'makespan': rev['makespan'],
                              'violations': schedule_check(jl, pt, desc, rev),
                              'changeover_h': list(changeover_hours(desc, pt, amch,
                                                                    rev['op_ct']))}
        if c == 5:
            best = min(rows, key=lambda x: x[1])
            r = dict(best[4])
            r['op_ct'] = r['op_ct'].copy()
            r['op_ct'][1] -= 2.0                   # op1 processes [6, 9) after op0 ends at 3
            r['makespan'] = float(r['op_ct'].max())
            out['shortened'] = schedule_check(jl, pt, desc, r)
            rev['op_ct'] = rev['op_ct'] - np.array([1.0, 0.0, 0.0])
            rev['makespan'] = float(rev['op_ct'].max())
            out['reverse_shortened'] = schedule_check(jl, pt, desc, rev)
    out['setup_shape'] = list(P101.make_env(jl, pt, meta, [desc]).true_setup.shape[1:])
    return out


def policy_report(model, size, combo, name, K):
    """The policy on one instance at the class count of the process (the stored
    descriptor at 4 classes, its level-five descriptor built in memory at 5): parameter
    count, input shapes at the first decision, the token, the machine setup channel,
    the makespan and the violations of the greedy schedule."""
    from constraint_family.descriptor import compute_token
    runner = P101.PolicyRunner()
    ppo, rec = runner.load(model)
    cell = [c for c in P101.load_cell(size, combo, P101.instance_index(name) + 1)
            if c[0] == name]
    _, jl, pt, meta, desc = cell[0]
    if K != BASE_K:
        desc = five_class_descriptor(desc, jl, pt, size, name)[0]
    M = pt.shape[1]
    env = P101.make_env(jl, pt, meta, [desc], mask_mode=rec['mask_mode'],
                        pair_extra=rec['pair_extra'], delta_mode=rec['delta_mode'])
    env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()), M)])
    env._attach_token()
    s = env.state
    keys = ('fea_j_tensor', 'op_mask_tensor', 'candidate_tensor', 'fea_m_tensor',
            'mch_mask_tensor', 'comp_idx_tensor', 'dynamic_pair_mask_tensor',
            'fea_pairs_tensor', 'op_type_tensor', 'mch_type_tensor', 'token_tensor')
    shapes = {k: (None if getattr(s, k, None) is None else list(getattr(s, k).shape))
              for k in keys}
    r = runner.rollout(model, jl, pt, meta, desc)
    v = schedule_check(jl, pt, desc, r)
    return {'model': model, 'params': rec['params'], 'instance': f'{size}/{combo}/{name}',
            'setup_shape': list(env.true_setup.shape[1:]), 'shapes': shapes,
            'token': [float(x) for x in env.token_g[0]],
            'mch_setup_channel': [float(x) for x in env._mch_min_next_setup()[0]],
            'makespan': r['makespan'], 'violations': len(v)}


def check(args):
    K = args.classes
    info = set_class_count(K)
    out = {'override': info,
           'constants': {m: {k: getattr(sys.modules[m], k) for k in names}
                         for m, names in PATCHED.items()},
           'hand': hand_report(K)}
    if args.policy:
        out['policy'] = policy_report(args.policy, '10x25', 'LSW', 'instance_000', K)
    assert_class_count(K)
    print('CHECK ' + json.dumps(jsonable(out)))
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['build', 'plan', 'evaluate', 'infer', 'check'])
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--level', default=None, choices=LEVELS,
                    help='evaluate: the level of this process')
    ap.add_argument('--levels', default=','.join(LEVELS))
    ap.add_argument('--sizes', default=','.join(SIZES))
    ap.add_argument('--combos', default=','.join(COMBOS))
    ap.add_argument('--arms', default=','.join(ARM_CHOICES))
    ap.add_argument('--seeds', default='301-305')
    ap.add_argument('--environment', default='exact', choices=ENVIRONMENTS,
                    help='run names of the policy arms; legacy only under --root')
    ap.add_argument('--default_policy', default='run', choices=('run', 'stored'))
    ap.add_argument('--n', type=int, default=0, help='first n instances (smoke only)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true')
    ap.add_argument('--out', default=None)
    ap.add_argument('--boot', type=int, default=B)
    ap.add_argument('--classes', type=int, default=BASE_K + 1, help='check: class count')
    ap.add_argument('--policy', default=None, help='check: run name of a policy')
    args = ap.parse_args(_ARGV)
    P101.check_mask_options()
    args.levels = P101.parse_list(args.levels, LEVELS, 'levels')
    if args.phase == 'evaluate':
        if args.level is None:
            sys.exit('REFUSING: evaluate needs --level default or --level five')
        args.levels = [args.level]
    args.sizes = P101.parse_list(args.sizes, SIZES, 'sizes')
    args.combos = P101.parse_list(args.combos, COMBOS, 'combinations')
    args.arms = P101.parse_list(args.arms, ARM_CHOICES, 'arms')
    args.seeds = list(REG.parse_seeds(args.seeds))
    for s in args.seeds:
        if s not in REG.PLANNED_SEEDS[args.environment]:
            sys.exit(f'REFUSING: seed {s} is not planned in the {args.environment} '
                     f'environment')
    if args.environment != 'exact' and os.path.abspath(os.path.join(ROOT_DIR, args.root)) \
            == os.path.abspath(os.path.join(ROOT_DIR, DEFAULT_ROOT)) \
            and args.phase in ('evaluate', 'infer'):
        sys.exit('REFUSING: --environment legacy runs only under another --root')
    args.n = args.n or None
    if args.part is not None and not 0 <= args.part < args.parts:
        sys.exit(f'--part {args.part} outside 0..{args.parts - 1}')
    os.chdir(ROOT_DIR)
    {'build': build, 'plan': plan, 'evaluate': evaluate, 'infer': infer,
     'check': check}[args.phase](args)


if __name__ == '__main__':
    main()
