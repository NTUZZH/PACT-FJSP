r"""P87: sensitivity of the main comparisons to the estimated benchmark parameters.

No retraining, evaluation only. Phases: build, plan, evaluate, analyze, compare, timing,
infer, archive-rules.

LAUNCH NOTE (the four questions, answered before the pilot)

1. Purpose. Two first-touch reviews asked whether the conclusions depend on the
   estimated parameters of the synthetic PPVC benchmark (durations estimated, setups
   and windows generated, one setup matrix with K = 4 classes charged on every
   station including same-class successions, low station contention at 10 modules
   on 25 stations). No plant data exist, so this study perturbs each estimated or
   generated parameter one at a time around its default and re-measures the three
   comparisons the paper's conclusions rest on. The numbers land in one sensitivity
   table of the supplement (one row per level, one column block per outcome) and in
   one sentence each in the benchmark section (Section 6.1) and the limitations.
2. Expected result. At levels inside the generator's range the verdicts of the
   default level hold: the policy stays within the equivalence margin of the
   specialists on the training combinations and ahead of the rule pool, and keeps
   its held-out results on SW and LSW. Outside the range the gaps may move, as the
   magnitude-shift study (Supplementary Section on larger rule magnitudes) found for
   larger, asymmetric setups on SW. Conclusion if the result comes back otherwise:
   the level at which a verdict changes is the boundary of the claim, stated as a
   number (for example "parity holds for setup scales 0.75 to 1.25 and fails at
   1.5 on LS"); a verdict that changes inside the generator range would mean the
   conclusion depends on the estimate, and the claim would be narrowed to the
   tested parameter values.
3. Contamination. All inputs are read from the frozen test set (data/FAMILY/10x25,
   data/PPVC/10x25+ppvc-mixed) and never written. Perturbed instances and
   descriptors are written under --root only (default test_results/FAMILY/sens_p87),
   never into an existing folder under data/; build refuses to overwrite a file
   whose content differs. Checkpoints are read-only. Results are new files under
   --root/results/<mask>/; no existing result file is touched, and no analysis
   script reads --root. The study evaluates the stored checkpoints (three seeds per
   arm) with the legacy mask they were trained with. Policy rollouts and dispatching
   rules are deterministic, so the results are not timing measurements; the recorded
   wall and CPU times are planning figures only. The default level is evaluated by
   the same command as every perturbed level, so that no comparison mixes two
   evaluation pipelines (Supplementary Section on the inactive-rule controls: the
   greedy rollout is deterministic within one environment but not bit-reproducible
   across devices or thread counts).
4. Data. 100 stored test instances per combination at the training size (10
   modules, 25 stations), eight combinations. The build regenerates every stored
   setup and window field with the default generator arguments and asserts that it
   is byte-identical to the file on disk before any shifted field is drawn; it
   asserts that each placebo cell (a combination the perturbation cannot reach) is
   unchanged field for field; it runs checks C1 to C5 on every perturbed descriptor
   (C1, C2, C3, C5 must pass; C4 warnings are counted as the measured share outside
   the training range); and it round-trips every perturbed instance file.

DESIGN (fixed before any result was looked at)

One-at-a-time perturbations of the 100 stored test instances, each perturbed
instance and descriptor derived from the stored one by the rule and seed recorded in
its file. Levels:

  default            the stored instances and descriptors (same-campaign reference)
  durations          (all eight combinations reached)
    dur_cv10/20/30   mean-preserving lognormal noise per (operation, machine) on every
                     compatible processing time, coefficient of variation 0.1, 0.2,
                     0.3: p' = max(1, floor(p * f + 0.5)), f = exp(sigma Z - sigma^2/2),
                     sigma^2 = ln(1 + cv^2), Z standard normal from
                     SeedSequence([20261111, 1, instance index]); the same Z at every
                     level (common random numbers), so the levels differ only in size
    dur_long_m25/p25 processing times of the two station types with the longest
                     processing scaled by 0.75 and 1.25: finishing (D) and assembly (F),
                     the two types with the largest mean operation duration (5.6 h and
                     5.2 h) and the largest work per station (56 h and 52 h per
                     instance, against at most 46 h elsewhere), so they are the
                     bottleneck types; p' = max(1, floor(p * s + 0.5))
  lags               (L, LS, LW, LSW reached; N, S, W, SW placebo)
    lag_x050/075/125/150  every nonzero post-operation lag scaled, floor(l * s + 0.5);
                     the outage calendar and its horizon are kept, so only the lag moves
  setups             (S, LS, SW, LSW reached; N, L, W, LW placebo)
    setup_x050/075/125/150  the setup matrix redrawn by descriptor.make_descriptor
                     with the stored descriptor seed and the scaled ranges same =
                     (0.25 s, 0.50 s), cross = (0.50 s, 1.00 s) (the cold-start row
                     uses the same-class range); same rng stream, so each entry is the
                     stored unrounded draw times s, rounded, floor 1 h
    setup_same0      same-class entries S[a, a] set to 0 (cross-class entries and the
                     cold-start row kept)
    setup_sub_strict setups charged only on the shell stations: formwork (A), concrete
                     pour (B), steel fabrication (H), where a mould, casting bed or
                     welding jig is exchanged between module types; 7 of 25 machines
    setup_sub_wide   setups charged on every production station: A, B, H plus MEP
                     fit-out (C), finishing (D), painting (E), assembly (F); not on
                     quality control (Q) and dispatch (G), where an inspection or a
                     wrap-and-ship step does not change tooling with the module type;
                     21 of 25 machines (descriptor field setup_mult)
  windows            (W, LW, SW, LSW reached; N, L, S, LS placebo)
    win_x050/150/200 the outage calendar redrawn by make_descriptor with the stored
                     seed and density_range = (0.05 s, 0.15 s); durations U[2, 8] h kept
  contention         (all eight reached)
    tight15          the generator's capacity-halved factory (TIGHT_FACTORY, 15
                     stations, at least one per type) derived from each stored
                     instance by keeping the first stations of each type; durations of
                     the kept stations, routes, lags, setup matrix and the calendars of
                     the kept stations unchanged. The inventory showed the 15-station
                     factory measured only at 50 and 80 modules and only for the
                     released policy against the rule pool; this level adds the
                     training size and the specialists, paired with the default level.
  project mix        (all eight reached; an addition to the brief, see below)
    mix_rc, mix_steel  the stored single-shell project sets
                     (data/PPVC/10x25+ppvc-rc_project, -steel_project, 50 instances
                     each, generator seeds 90000 and 95000), descriptors drawn by
                     make_descriptor with the default arguments and the seed
                     convention of every stored family set (20260706 + index). A real
                     project uses one shell system (guidebook), so only two of the
                     four route classes, and so two of the four setup classes, occur.
                     These sets were never evaluated with the timing rules. Not paired
                     with the default level (other base instances), so no shift is
                     computed; the within-level comparisons are paired.

Arms: the released policy jointblind-v1 (seeds 301, 302, 303), the specialist of the
combination spec-<R> (three seeds), the single-rule control jointblind-nlsw (three
seeds), and FIFO, MOR, SPT, MWKR. Every policy arm is loaded with the configuration
eval_family.py restores from its training snapshot, and its parameter count is
asserted (32786).

Rules. With --mask exact (the environment of the paper) the four rules are those of
every other analysis of the paper, common_utils.masked_rule_action (eval_family.py
--rule_set masked and its stored arrays in test_results/FAMILY/10x25_xmask;
scripts/p101_mask_controls.py rule_rollout, which p102 and p103 call): the priority key
(earliest candidate ready time, most operations left, most work left) ranks the jobs
that hold a pair the mask admits, and the chosen job goes to its admitted machine with
the shortest processing time; SPT takes the admitted pair with the shortest processing
time. With --mask legacy they stay the rules of the earlier study,
common_utils.heuristic_select_action: the key ranks the ready jobs, and the chosen job
goes to a machine that can start it with the earliest start, ties at random, whatever
its processing time (a job without an admitted pair falls back to a compatible machine
that cannot start it now); SPT is the same rule in both. Ties are drawn from
numpy's global stream, np.random.seed(50) after the data of each instance are set
(eval_family.py, params.py seed_test). Each rule record states its rule set (rule_set:
masked or classical, with rule_function and rule_seed); evaluate, analyze and infer
refuse a rule file of the other rule set, and evaluate never overwrites one. The rule
files of the exact mask written before 2026-11-16 came from heuristic_select_action:
archive-rules moves them to results/exact_rules_classical/<level>/<R>/ and
plan --only_rules plans the rule units again.

Outcomes per level and combination, each a mean of per-instance relative gaps with a
95% percentile paired bootstrap over instances (20,000 resamples, a fresh
numpy default_rng(7) per interval, the paper's estimator), policy arms averaged over
their three seeds per instance (both sides pooled over the same seeds):
  (a) policy minus specialist, 100 (P - S) / S; verdicts at margins 0.5% and 1%
      with the paper's rule (SUPERIOR, EQUIVALENT, non-inferior, INFERIOR,
      INCONCLUSIVE);
  (b) the policy's lead over the rule pool (per-instance best of the four rules),
      100 (pool - P) / P, positive when the policy is ahead; also over the best
      single rule by mean;
  (c) the same on the held-out combinations SW and LSW (reported as their own block);
  (d) policy minus single-rule control, 100 (P - C) / C;
  and the shift of (a) and (b) against the default level, paired by base instance.
Each level is marked inside or outside the training range in two ways: a nominal
mark (the mean of the perturbed parameter's distribution inside the generator range,
"boundary" when it falls on an endpoint, "outside" for structures the generator never
produces) and the measured share of descriptors that check C4 flags (or, for
durations, of processing times outside the generator's per-step envelope).

Phase infer: the statistical protocol of the manuscript (stats_family), beside the
analyze phase above, which it leaves unchanged. It reads the per-seed result files
of the evaluate phase, results/<mask>/<level>/<regime>/<method>.npz (makespan per
instance; one file per arm and training seed and one per rule) with their .json
records; a combination a level does not reach reads the default level.
Gaps are d = 100 (a - b) / b with a the first-named method, seed s of a against
seed s of b, per base instance; "the best dispatching rule" is the per-instance
minimum of FIFO, MOR, SPT and MWKR of the mask's rule set (a rule file whose record
states another rule set is refused; see Rules above). Per level and combination, pooled over the
eight combinations (a drawn base instance brings all its combinations), and, for
a level that reaches fewer than eight, pooled over the combinations it reaches:
  policy against the specialist of the combination and the single-rule control
    against that specialist: crossed hierarchical bootstrap over training runs
    and base instances (stats_family.hier_boot, B = 20000, a fresh
    default_rng(11) per quantity), verdicts at the margins 0.5% and 1%
    (stats_family.verdict) and the direction of the interval;
  policy against the best dispatching rule, on each combination the level
    reaches: the same interval, one-sided seed-level t and bootstrap p-values in
    both directions, Holm's correction over the combinations the level reaches
    for each kind, and the combined verdict of
    p89_public_family.superiority_verdict (ahead, behind, not distinguishable);
    on a combination the level does not reach (a copy of the default level) and
    on the pooled rows, the interval and its direction only.
Per level it also counts the verdicts over the combinations the level reaches and,
for the levels on the default level's instances, the verdicts that differ from the
default level's on the same combination (not computed for the project-mix levels,
which use other base instances). With one training run the intervals are the
bootstrap over base instances (default_rng(7)). When every result file carries the
counts of evaluate --trace_counts, it adds per level and combination two measured
variables, for the policy (mean over its training runs) and for the schedule of the
per-instance best rule: the share of decisions at which the mask admits exactly one
(job, machine) pair, over all decisions of the instances, and the setup share of
machine capacity, per instance the total setup time over the machines used times
the makespan, averaged over the instances (the definitions of
scripts/p93_rule_comparison.py trace_stats); otherwise it says they are not computed.
Outputs: manuscript/macros_sensitivity.tex (prefix Sens: the pooled rows, the
verdict counts per level, the study-level counts, the per-combination rows of the
default level, and the measured variables of the default level, the setup scales
and the shares of stations with setups), manuscript/tables/sensitivity_table.tex
(one row per level, for the supplement), results/sensitivity_inference.json, .md
and .csv (one row per level, comparison and combination or pool, for a figure). The
phase refuses, and removes these outputs, when a level, arm, training seed or
instance is missing, when the arms of a combination cover different instances, or
when a schedule failed check C6. In place only with --mask exact and every level;
otherwise, and always with --partial (the complete combinations only, Holm over
the reached combinations present, pools only when complete), --out <folder>
outside the repository's paper and results folders is required.

Usage (from the repository root; CPU only):
  python scripts/p87_sensitivity.py build    [--root R] [--levels a,b] [--n 100]
  python scripts/p87_sensitivity.py plan     [--root R] --parts K [--n 100]
  python scripts/p87_sensitivity.py plan     [--root R] --parts K --trace_counts
  python scripts/p87_sensitivity.py plan     [--root R] --parts K --trace_counts
                                             --only_rules [--dry]   # rule units alone
  python scripts/p87_sensitivity.py evaluate [--root R] --part i --parts K
                                             [--trace_counts] [--only_rules]
  python scripts/p87_sensitivity.py archive-rules [--root R] --mask exact [--dry]
  python scripts/p87_sensitivity.py evaluate [--root R] --levels a,b --regimes S,LS
                                             --arms pact,spec,single,rules --n 5
  python scripts/p87_sensitivity.py analyze  [--root R]
  python scripts/p87_sensitivity.py infer    [--root R] --mask exact --arm_names xmask
                                             --seeds 301,302,303,304,305 [--boot B]
  python scripts/p87_sensitivity.py infer    [--root R] --partial --out <folder>
                                             [--levels a,b]
  python scripts/p87_sensitivity.py infer    [--root R] --check --mask exact ...
                                             (inputs complete? writes nothing)
  python scripts/p87_sensitivity.py compare  [--root R]   # default level vs stored
  python scripts/p87_sensitivity.py timing   [--root R]   # planning figures
Options common to the phases: --seeds 301,302,303 (304,305 once trained);
--mask legacy|exact with --arm_names legacy|xmask (the exact-mask retraining,
names xmask-v1, xmask-solo-<R>, xmask-ctl); a checkpoint whose recorded mask
differs from --mask is refused; results of the two masks live in separate
folders, results/<mask>/<level>/<regime>/<method>.npz with a .json record.
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
sys.path.insert(0, ROOT_DIR)

_ARGV = sys.argv[1:]
# params parses argv at import time and EnvState reads the device then: CPU only
sys.argv = sys.argv[:1] + ['--device', 'cpu']

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# design constants
# ---------------------------------------------------------------------------

STORED_DESC = 'data/FAMILY/10x25'
STORED_BASE = 'data/PPVC/10x25+ppvc-mixed'
STORED_RES = 'test_results/FAMILY/10x25'
STORED_CPU = 'test_results/FAMILY/10x25_cpuref'
DEFAULT_ROOT = 'test_results/FAMILY/sens_p87'
R8 = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
HELD_OUT = ('SW', 'LSW')
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
# The rule set of each mask (module doc, "Rules"): the exact environment of the paper
# uses the rules restricted to the pairs the mask admits, as eval_family.py --rule_set
# masked and the stored arrays of STORED_XMASK; the legacy environment keeps the rules of
# the earlier study. A rule record without the field rule_set was written by
# heuristic_select_action (every rule file before the field existed), so 'classical'.
RULE_SET = {'legacy': 'classical', 'exact': 'masked'}
RULE_FUNCTION = {'masked': 'common_utils.masked_rule_action',
                 'classical': 'common_utils.heuristic_select_action'}
# np.random.seed before the first decision of every rule rollout, after the data are set:
# params.py seed_test, which eval_family.py uses for the stored arrays (no --seed_test is
# passed by scripts/run_eval_newseeds.sh)
RULE_SEED = 50
STORED_XMASK = 'test_results/FAMILY/10x25_xmask'
# archive-rules: the folder under <root>/results/ that receives the earlier rule files
ARCHIVE_RULES = 'exact_rules_classical'
DEFAULT_ARMS = 'pact,spec,single,rules'
SEEDS = (301, 302, 303)
EXPECT_PARAMS = 32786
NOISE_SEED = 20261111
B = 20000
MARGINS = (0.5, 1.0)

# station types: A formwork, B pour, C MEP, D finishing, E paint, F assembly,
# G dispatch, H steel fabrication, Q quality control (ppvc_instance_generator)
LONG_TYPES = ('D', 'F')
SUB_STRICT = ('A', 'B', 'H')
SUB_WIDE = ('A', 'B', 'C', 'D', 'E', 'F', 'H')

LEVELS = {
    'default':          dict(family='default'),
    'dur_cv10':         dict(family='dur', kind='cv', cv=0.10),
    'dur_cv20':         dict(family='dur', kind='cv', cv=0.20),
    'dur_cv30':         dict(family='dur', kind='cv', cv=0.30),
    'dur_long_m25':     dict(family='dur', kind='type_scale', scale=0.75, types=LONG_TYPES),
    'dur_long_p25':     dict(family='dur', kind='type_scale', scale=1.25, types=LONG_TYPES),
    'lag_x050':         dict(family='lag', scale=0.50),
    'lag_x075':         dict(family='lag', scale=0.75),
    'lag_x125':         dict(family='lag', scale=1.25),
    'lag_x150':         dict(family='lag', scale=1.50),
    'setup_x050':       dict(family='setup', kind='scale', scale=0.50),
    'setup_x075':       dict(family='setup', kind='scale', scale=0.75),
    'setup_x125':       dict(family='setup', kind='scale', scale=1.25),
    'setup_x150':       dict(family='setup', kind='scale', scale=1.50),
    'setup_same0':      dict(family='setup', kind='same_zero'),
    'setup_sub_strict': dict(family='setup', kind='subset', types=SUB_STRICT),
    'setup_sub_wide':   dict(family='setup', kind='subset', types=SUB_WIDE),
    'win_x050':         dict(family='win', scale=0.50),
    'win_x150':         dict(family='win', scale=1.50),
    'win_x200':         dict(family='win', scale=2.00),
    'tight15':          dict(family='contention', factory='tight'),
    'mix_rc':           dict(family='mix', base='data/PPVC/10x25+ppvc-rc_project'),
    'mix_steel':        dict(family='mix', base='data/PPVC/10x25+ppvc-steel_project'),
}
REACHED = {
    'default': R8, 'dur': R8, 'contention': R8, 'mix': R8,
    'lag': ('L', 'LS', 'LW', 'LSW'),
    'setup': ('S', 'LS', 'SW', 'LSW'),
    'win': ('W', 'LW', 'SW', 'LSW'),
}


def reached(level):
    return REACHED[LEVELS[level]['family']]


def placebo(level):
    return tuple(r for r in R8 if r not in reached(level))


def round_half_up(x):
    return np.floor(np.asarray(x, dtype=float) + 0.5)


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def write_json_atomic(path, obj, indent=None):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp, path)


def write_once(path, text):
    """Write text, or verify an existing file holds exactly that text."""
    if os.path.exists(path):
        with open(path) as f:
            if f.read() != text:
                sys.exit(f'REFUSING: {path} exists with different content; the build '
                         f'is deterministic, so a difference means the design or the '
                         f'inputs changed. Use a new --root.')
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT_DIR,
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return 'unknown'


def stored_names(n=None):
    names = sorted(os.path.basename(p).replace('.desc.json', '') for p in
                   glob.glob(os.path.join(STORED_DESC, 'desc', 'N', '*.desc.json')))
    if len(names) != 100:
        sys.exit(f'REFUSING: expected 100 stored test descriptors per regime, found '
                 f'{len(names)} under {STORED_DESC}/desc/N')
    return names[:n] if n else names


def level_names(level, n=None):
    """Instance names of a level: the 100 stored test instances, or for a
    project-mix level the stored base set of that mix (50 instances)."""
    spec = LEVELS[level]
    if spec['family'] != 'mix':
        return stored_names(n)
    names = sorted(os.path.basename(p)[:-len('.fjs')] for p in
                   glob.glob(os.path.join(spec['base'], 'instance_*.fjs')))
    if len(names) != 50:
        sys.exit(f'REFUSING: expected 50 instances under {spec["base"]}, '
                 f'found {len(names)}')
    return names[:n] if n else names


def paired_with_default(level):
    """Levels derived from the stored test instances (paired by instance with
    the default level); the project-mix levels use other base instances."""
    return LEVELS[level]['family'] not in ('mix',)


def stored_record(regime, name):
    with open(os.path.join(STORED_DESC, 'desc', regime, f'{name}.desc.json')) as f:
        return json.load(f)


CORE_KEYS = ('active', 'lag', 'op_class', 'setup', 'windows', 'horizon')


def placebo_unchanged(desc, rec):
    """True when the descriptor of a placebo cell equals its stored record:
    every core field identical and no further field, except the setup
    convention that to_json writes for a record with setups, provided it is
    the convention the stored record states (attached when the record leaves
    the field out, as every stored benchmark record does)."""
    a = desc.to_json()
    if any(json.dumps(a[k]) != json.dumps(rec[k]) for k in CORE_KEYS):
        return False
    extra = set(a) - set(CORE_KEYS)
    if a.get('setup_convention', 'attached') == rec.get('setup_convention', 'attached'):
        extra.discard('setup_convention')
    return not extra


# ---------------------------------------------------------------------------
# perturbation rules
# ---------------------------------------------------------------------------

def perturb_base(level, idx, jl, pt, meta):
    """Perturbed (jl, pt, meta) of base instance idx under `level`, or None when
    the level keeps the stored base instance."""
    spec = LEVELS[level]
    from ppvc_instance_generator import (DEFAULT_FACTORY, STATION_TYPES,
                                         TIGHT_FACTORY, build_factory)
    pt = np.asarray(pt)
    if spec['family'] == 'dur':
        out = pt.astype(float).copy()
        comp = pt > 0
        if spec['kind'] == 'cv':
            cv = spec['cv']
            sig = float(np.sqrt(np.log(1.0 + cv * cv)))
            rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, 1, idx]))
            Z = rng.standard_normal(pt.shape)        # one draw per (op, machine)
            f = np.exp(sig * Z - 0.5 * sig * sig)
            out[comp] = np.maximum(1.0, round_half_up(pt[comp] * f[comp]))
        else:
            st = np.asarray(meta['op_station'])
            tids = [STATION_TYPES.index(t) for t in spec['types']]
            rows = np.isin(st, tids)
            sel = comp & rows[:, None]
            out[sel] = np.maximum(1.0, round_half_up(pt[sel] * spec['scale']))
        meta2 = copy.deepcopy(meta)
        return jl, out.astype(int), meta2
    if spec['family'] == 'contention':
        mch_type, ranges = build_factory(DEFAULT_FACTORY)
        assert np.array_equal(np.asarray(meta['mch_type']), mch_type), \
            'stored instance is not on the default 25-station factory'
        keep = [m for t in STATION_TYPES for m in ranges[t][:TIGHT_FACTORY[t]]]
        meta2 = copy.deepcopy(meta)
        meta2['mch_type'] = np.asarray(mch_type)[keep]
        meta2['station_counts'] = dict(TIGHT_FACTORY)
        out = pt[:, keep]
        assert (out > 0).any(axis=1).all(), 'an operation lost every station'
        return jl, out.astype(int), meta2
    return None


def kept_machines(level):
    """Indices of the stored machines a contention level keeps (else None)."""
    if LEVELS[level]['family'] != 'contention':
        return None
    from ppvc_instance_generator import (DEFAULT_FACTORY, STATION_TYPES,
                                         TIGHT_FACTORY, build_factory)
    _, ranges = build_factory(DEFAULT_FACTORY)
    return [m for t in STATION_TYPES for m in ranges[t][:TIGHT_FACTORY[t]]]


def perturb_desc(level, regime, rec, jl, pt, meta):
    """Perturbed descriptor of a stored record (stored base instance arrays
    jl, pt, meta), as a ConstraintDescriptor."""
    from constraint_family.descriptor import ConstraintDescriptor, make_descriptor
    spec = LEVELS[level]
    fam = spec['family']
    # the stored benchmark draws attached setups (generator convention)
    desc = ConstraintDescriptor.from_json(rec, setup_convention='attached')
    if fam in ('default', 'dur'):
        return desc
    if fam == 'lag':
        if desc.has_lag:
            desc.lag = round_half_up(desc.lag * spec['scale'])
        return desc
    if fam == 'setup':
        if not desc.has_setup:
            return desc
        if spec['kind'] == 'scale':
            s = spec['scale']
            d2 = make_descriptor(regime, jl, pt, meta, pt.shape[1],
                                 seed=rec['desc_seed'],
                                 shift={'same': (0.25 * s, 0.50 * s),
                                        'cross': (0.50 * s, 1.00 * s)})
            desc.setup = d2.setup
        elif spec['kind'] == 'same_zero':
            from constraint_family.descriptor import N_CLASSES
            S = desc.setup.copy()
            for a in range(N_CLASSES):
                S[a, a] = 0.0
            desc.setup = S
        else:
            from ppvc_instance_generator import STATION_TYPES
            mt = np.asarray(meta['mch_type'])
            tids = [STATION_TYPES.index(t) for t in spec['types']]
            desc.setup_mult = np.isin(mt, tids).astype(float)
        return desc
    if fam == 'win':
        if not desc.has_window:
            return desc
        s = spec['scale']
        d2 = make_descriptor(regime, jl, pt, meta, pt.shape[1], seed=rec['desc_seed'],
                             shift={'density_range': (0.05 * s, 0.15 * s)})
        assert d2.horizon == desc.horizon
        desc.windows = d2.windows
        return desc
    if fam == 'contention':
        keep = kept_machines(level)
        if desc.has_window and desc.windows is not None:
            desc.windows = [desc.windows[k] for k in keep]
        return desc
    raise ValueError(level)


def nominal_mark(level):
    """Inside / boundary / outside by the mean of the perturbed distribution
    against the generator range (structures the generator never draws are
    outside)."""
    spec = LEVELS[level]
    fam = spec['family']
    if fam == 'default':
        return 'inside'
    if fam == 'contention':
        return 'outside (factory never trained on)'
    if fam == 'mix':
        return ('outside (single-shell project mix, two route classes; training '
                'draws the multi-project mix of four)')
    if fam == 'dur':
        if spec['kind'] == 'cv':
            # the generator's across-machine variation: U[0.8, 1.2] -> cv 0.115
            return ('inside (mean kept; spread at most the generator\'s cv 0.115)'
                    if spec['cv'] <= 0.2 / np.sqrt(3) + 1e-9 else
                    'outside in spread (mean kept; generator cv 0.115)')
        from ppvc_instance_generator import ROUTES
        s = spec['scale']
        ok = all(0.8 * lo <= s * (lo + hi) / 2 <= 1.2 * hi
                 for r in ROUTES.values() for (_, t, _, (lo, hi), _) in r
                 if t in spec['types'])
        return 'inside (scaled step means within the per-step envelope)' if ok \
            else 'outside'
    if fam == 'lag':
        s = spec['scale']
        bands = [(24, 48), (12, 24)]
        ins = [lo <= s * (lo + hi) / 2 <= hi for lo, hi in bands]
        edge = [np.isclose(s * (lo + hi) / 2, lo) or np.isclose(s * (lo + hi) / 2, hi)
                for lo, hi in bands]
        return 'boundary' if any(edge) and all(ins) else ('inside' if all(ins)
                                                           else 'outside')
    if fam == 'setup':
        if spec['kind'] != 'scale':
            return 'outside (structure never drawn in training)'
        s = spec['scale']
        rngs = [(0.25, 0.50), (0.50, 1.00)]
        ins = [lo <= s * (lo + hi) / 2 <= hi for lo, hi in rngs]
        return 'inside' if all(ins) else 'outside'
    if fam == 'win':
        m = spec['scale'] * 0.10
        if np.isclose(m, 0.05) or np.isclose(m, 0.15):
            return 'boundary'
        return 'inside' if 0.05 < m < 0.15 else 'outside'
    raise ValueError(level)


def entries_outside(desc, pt, jl=None, meta=None):
    """(outside, total) counts of the perturbed rule's entries against the
    generator ranges check C4 uses (constraint_family/checks.py), entry by
    entry: nonzero lags against the band of their own route step when jl and
    meta are given (C4 compares with the union of the two bands, so a scaled
    paint-drying lag of 30 h would pass as a curing lag), else against the
    union; setup entries (same-class,
    cross-class, cold start) against their ranges in multiples of p_bar,
    per-machine outage densities against the density range (plus one outage of
    overshoot). Same 0.5 h rounding tolerance as C4. For setups the interval is
    the one the generator can reach, [max(1, lo) - 0.5, max(1, hi) + 0.5] h
    with lo, hi the range in hours. C4's lower limit is max(lo - 0.5, 1) h,
    which admits the same integer setups as this interval, so both report a
    scale-down below the range."""
    from constraint_family import checks
    from constraint_family.descriptor import IDLE_CLASS, N_CLASSES
    out = {}
    if desc.has_lag and jl is not None and meta is not None:
        from ppvc_instance_generator import ROUTES, ROUTING_CLASSES
        bad = tot = row = 0
        cls = np.asarray(meta['routing_class'])
        for j, L in enumerate(np.asarray(jl, dtype=int)):
            route = ROUTES[ROUTING_CLASSES[int(cls[j])]]
            for r in range(L):
                lo, hi = route[r][4]
                v = desc.lag[row]
                row += 1
                if hi == 0:
                    continue
                tot += 1
                bad += not (lo - checks.ROUND_TOL_H <= v <= hi + checks.ROUND_TOL_H)
        out['lag'] = (int(bad), int(tot))
    elif desc.has_lag:
        nz = desc.lag[desc.lag > 0]
        bad = [v for v in nz if not any(lo - checks.ROUND_TOL_H <= v
                                        <= hi + checks.ROUND_TOL_H
                                        for lo, hi in checks.LAG_RANGES)]
        out['lag'] = (len(bad), int(nz.size))
    if desc.has_setup:
        pbar = float(pt[pt > 0].mean())
        S = desc.setup
        C = N_CLASSES
        n_bad = n_tot = 0
        for vals, (lo, hi) in (([S[a, a] for a in range(C)], checks.SETUP_SAME_RANGE),
                               ([S[a, b] for a in range(C) for b in range(C) if a != b],
                                checks.SETUP_CROSS_RANGE),
                               (list(S[IDLE_CLASS]), checks.SETUP_COLD_RANGE)):
            lo_v = max(1.0, lo * pbar) - checks.ROUND_TOL_H
            hi_v = max(1.0, hi * pbar) + checks.ROUND_TOL_H
            n_bad += sum(1 for v in vals if v < lo_v or v > hi_v)
            n_tot += len(vals)
        out['setup'] = (n_bad, n_tot)
    if desc.has_window:
        H = float(desc.horizon)
        dlo, dhi = checks.WINDOW_DENSITY_RANGE
        dens = [sum(e - s for s, e in w) / H for w in desc.windows]
        lo_ok = dlo - checks.ROUND_TOL_H / H
        hi_ok = dhi + (checks.WINDOW_DUR_RANGE[1] + checks.ROUND_TOL_H) / H
        out['window'] = (sum(1 for d in dens if not lo_ok <= d <= hi_ok), len(dens))
    return out


def duration_envelope_share(jl, pt, meta):
    """Share of compatible processing times outside the generator's per-step
    envelope [0.8 lo, 1.2 hi] (0.5 h rounding tolerance)."""
    from ppvc_instance_generator import ROUTES, ROUTING_CLASSES
    cls = np.asarray(meta['routing_class'])
    out = tot = 0
    row = 0
    for j, L in enumerate(np.asarray(jl, dtype=int)):
        route = ROUTES[ROUTING_CLASSES[int(cls[j])]]
        for r in range(L):
            lo, hi = route[r][3]
            v = pt[row][pt[row] > 0]
            out += int(((v < 0.8 * lo - 0.5) | (v > 1.2 * hi + 0.5)).sum())
            tot += v.size
            row += 1
    return out, tot


# ---------------------------------------------------------------------------
# phase: build
# ---------------------------------------------------------------------------

def level_data_dir(root, level):
    if level == 'default':
        return STORED_DESC
    return os.path.join(root, 'data', level)


def guard_generator(names):
    """Refuse to build if the default generator arguments no longer reproduce the
    stored setup and window fields (every regime that carries them)."""
    from constraint_family.descriptor import make_descriptor
    from ppvc_instance_generator import load_instance
    checked = 0
    for regime in R8:
        if 'S' not in regime and 'W' not in regime:
            continue
        for name in names:
            rec = stored_record(regime, name)
            jl, pt, meta = load_instance(rec['base_instance'])
            d = make_descriptor(regime, jl, pt, meta, pt.shape[1],
                                seed=rec['desc_seed']).to_json()
            for k in CORE_KEYS:
                if json.dumps(d[k]) != json.dumps(rec[k]):
                    sys.exit(f'REFUSING: the default draw no longer reproduces '
                             f'{regime}/{name} field {k}')
            checked += 1
    expect = 6 * len(names)
    if checked != expect:
        sys.exit(f'REFUSING: guard compared {checked} descriptors, expected {expect}')
    return checked


def build_mix(args, level, n_guard):
    """Descriptors of a project-mix level: the stored single-shell base set
    (50 instances) with the eight descriptors drawn by make_descriptor under
    the convention of every stored family set, descriptor seed 20260706 + the
    instance index, default generator arguments (verified by the guard)."""
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor, make_descriptor
    from ppvc_instance_generator import ROUTING_CLASSES, load_instance
    spec = LEVELS[level]
    names = level_names(level, args.n)
    out_dir = level_data_dir(args.root, level)
    digest = hashlib.sha256()
    c4 = {r: 0 for r in R8}
    classes, setup_mean = set(), []
    for name in names:
        stem = os.path.join(spec['base'], name)
        jl, pt, meta = load_instance(stem)
        seed = 20260706 + int(name.split('_')[-1])
        classes |= {ROUTING_CLASSES[int(c)] for c in meta['routing_class']}
        for regime in R8:
            d = make_descriptor(regime, jl, pt, meta, pt.shape[1], seed=seed)
            out = d.to_json()
            out['base_instance'] = stem
            out['regime'] = regime
            out['desc_seed'] = seed
            out['sens'] = {'level': level, 'rule': dict(spec),
                           'source_base': stem, 'source_desc': None,
                           'note': 'descriptor drawn by make_descriptor with the '
                                   'default arguments'}
            txt = json.dumps(out)
            write_once(os.path.join(out_dir, 'desc', regime, f'{name}.desc.json'), txt)
            digest.update(txt.encode())
            dd = ConstraintDescriptor.from_json(out)
            rep = {r.id: r for r in checks.run_all(dd, pt, False, job_length=jl)}
            if any(rep[c].status == checks.FAIL for c in ('C1', 'C2', 'C3', 'C5')):
                sys.exit(f'REFUSING: {level} {regime} {name} fails a C1-C5 check')
            c4[regime] += rep['C4'].status == checks.WARN
            if dd.has_setup:
                setup_mean.append(float(dd.setup.mean()))
    man = {'level': level, 'rule': dict(spec), 'reached': list(R8), 'placebo': [],
           'n_instances': len(names), 'instances': names,
           'nominal_mark': nominal_mark(level),
           'c4_warn_share': {r: v / len(names) for r, v in c4.items()},
           'entries_outside_share': None, 'entries_outside_what': None,
           'route_classes_present': sorted(classes),
           'setup_mean_h': float(np.mean(setup_mean)),
           'paired_with_default': False,
           'content_sha256': digest.hexdigest(), 'generator_guard_checked': n_guard,
           'commit': git_commit()}
    mp = os.path.join(out_dir, 'manifest.json')
    if os.path.exists(mp) and json.load(open(mp)).get('content_sha256') \
            != man['content_sha256']:
        sys.exit(f'REFUSING: {mp} records different content')
    write_json_atomic(mp, man, indent=1)
    print(f'[build] {level:17s} {len(names)} instances of {spec["base"]}, classes '
          f'{sorted(classes)}, C4 warn=' + ' '.join(
              f'{r}:{v:.2f}' for r, v in man['c4_warn_share'].items()), flush=True)


def build(args):
    from constraint_family import checks
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance, save_instance
    names = stored_names(args.n)
    levels = args.levels or [k for k in LEVELS if k != 'default']
    n_guard = guard_generator(names)
    print(f'[build] guard: {n_guard} stored setup/window descriptors regenerated with '
          f'the default arguments, all byte-identical', flush=True)
    # the same entry-level measures on the stored (default) test set, as the
    # reference the perturbed levels are read against
    ref = {'lag': [0, 0], 'setup': [0, 0], 'window': [0, 0], 'dur': [0, 0]}
    for name in names:
        for regime, key in (('L', 'lag'), ('S', 'setup'), ('W', 'window')):
            rec = stored_record(regime, name)
            jl, pt, meta = load_instance(rec['base_instance'])
            o, t = entries_outside(ConstraintDescriptor.from_json(
                rec, setup_convention='attached'), pt, jl, meta)[key]
            ref[key][0] += o
            ref[key][1] += t
        o, t = duration_envelope_share(jl, pt, meta)
        ref['dur'][0] += o
        ref['dur'][1] += t
    dman = {'level': 'default', 'nominal_mark': 'inside',
            'entries_outside_share': {k: v[0] / v[1] for k, v in ref.items()},
            'stored_desc': STORED_DESC, 'stored_base': STORED_BASE,
            'n_instances': len(names), 'commit': git_commit()}
    write_json_atomic(os.path.join(args.root, 'data', 'default_manifest.json'), dman,
                      indent=1)
    print('[build] default level, entries outside the generator ranges: ' + ' '.join(
        f'{k}={v:.3f}' for k, v in dman['entries_outside_share'].items()), flush=True)
    for level in levels:
        if level == 'default':
            continue
        if LEVELS[level]['family'] == 'mix':
            build_mix(args, level, n_guard)
            continue
        spec = LEVELS[level]
        out_dir = level_data_dir(args.root, level)
        fam_key = {'lag': 'lag', 'setup': 'setup', 'win': 'window'}.get(
            spec['family'])
        stats = {'c4_warn': {r: 0 for r in reached(level)},
                 'entries_out': 0, 'entries_tot': 0,
                 'c1235_fail': 0, 'placebo_checked': 0,
                 'dur_outside': 0, 'dur_total': 0,
                 'setup_mean_ratio': [], 'window_density': [],
                 'lag_mean_nonzero': [], 'setup_machines': []}
        digest = hashlib.sha256()
        for idx, name in enumerate(names):
            rec0 = stored_record('N', name)
            stem0 = rec0['base_instance']
            jl, pt, meta = load_instance(stem0)
            idx_global = int(name.split('_')[-1])
            assert rec0['desc_seed'] == 20260706 + idx_global
            nb = perturb_base(level, idx_global, jl, pt, meta)
            if nb is not None:
                jl2, pt2, meta2 = nb
                stem = os.path.join(out_dir, 'base', name)
                os.makedirs(os.path.dirname(stem), exist_ok=True)
                tmp = stem + '.__tmp'
                save_instance(tmp, jl2, pt2, meta2)
                for ext in ('.fjs', '.meta.json'):
                    with open(tmp + ext) as f:
                        txt = f.read()
                    os.remove(tmp + ext)
                    write_once(stem + ext, txt)
                    digest.update(txt.encode())
                jlr, ptr, metar = load_instance(stem)
                assert np.array_equal(ptr, pt2) and np.array_equal(jlr, jl2)
                assert np.array_equal(np.asarray(metar['mch_type']),
                                      np.asarray(meta2['mch_type']))
                if spec['family'] == 'dur':
                    o, t = duration_envelope_share(jl2, pt2, meta2)
                    stats['dur_outside'] += o
                    stats['dur_total'] += t
                base_stem, pt_eval, jl_eval = stem, pt2, jl2
            else:
                base_stem, pt_eval, jl_eval = stem0, pt, jl
            for regime in R8:
                rec = stored_record(regime, name)
                d = perturb_desc(level, regime, rec, jl, pt, meta)
                if regime not in reached(level):
                    # placebo: the rule must leave the descriptor unchanged
                    if not placebo_unchanged(d, rec):
                        sys.exit(f'REFUSING: {level} changed placebo cell '
                                 f'{regime}/{name}')
                    stats['placebo_checked'] += 1
                    continue
                out = d.to_json()
                out['base_instance'] = os.path.relpath(base_stem, ROOT_DIR) \
                    if os.path.abspath(base_stem).startswith(ROOT_DIR) else base_stem
                out['regime'] = regime
                out['desc_seed'] = rec['desc_seed']
                out['sens'] = {'level': level, 'rule': {k: (list(v) if isinstance(
                    v, tuple) else v) for k, v in spec.items()},
                    'source_desc': os.path.join(STORED_DESC, 'desc', regime,
                                                f'{name}.desc.json'),
                    'source_sha256': sha256_file(os.path.join(
                        STORED_DESC, 'desc', regime, f'{name}.desc.json')),
                    'source_base': stem0,
                    'noise_seed': ([NOISE_SEED, 1, idx_global]
                                   if spec.get('kind') == 'cv' else None)}
                txt = json.dumps(out)
                write_once(os.path.join(out_dir, 'desc', regime, f'{name}.desc.json'),
                           txt)
                digest.update(txt.encode())
                # checks C1..C5 on the descriptor as it will configure the env
                dd = ConstraintDescriptor.from_json(out)
                rep = checks.run_all(dd, pt_eval, False, job_length=jl_eval)
                by = {r.id: r for r in rep}
                if any(by[c].status == checks.FAIL for c in ('C1', 'C2', 'C3', 'C5')):
                    stats['c1235_fail'] += 1
                    print(f'  C-check FAIL {level} {regime} {name}: ' + '; '.join(
                        f'{c}: {by[c].messages}' for c in ('C1', 'C2', 'C3', 'C5')
                        if by[c].status == checks.FAIL))
                if by['C4'].status == checks.WARN:
                    stats['c4_warn'][regime] += 1
                for k, (o, t) in entries_outside(dd, pt_eval, jl, meta).items():
                    if k == fam_key:
                        stats['entries_out'] += o
                        stats['entries_tot'] += t
                if dd.has_setup:
                    s0 = np.asarray(rec['setup'], float)
                    stats['setup_mean_ratio'].append(float(dd.setup.mean() / s0.mean()))
                    if dd.setup_mult is not None:
                        stats['setup_machines'].append(int(dd.setup_mult.sum()))
                if dd.has_window:
                    H = float(dd.horizon)
                    stats['window_density'].append(float(np.mean(
                        [sum(e - s for s, e in w) / H for w in dd.windows])))
                if dd.has_lag:
                    stats['lag_mean_nonzero'].append(float(dd.lag[dd.lag > 0].mean()))
        if stats['c1235_fail']:
            sys.exit(f'REFUSING: {stats["c1235_fail"]} descriptors of {level} fail '
                     f'C1, C2, C3 or C5')
        man = {'level': level, 'rule': {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in spec.items()},
               'reached': list(reached(level)), 'placebo': list(placebo(level)),
               'n_instances': len(names), 'instances': names,
               'nominal_mark': nominal_mark(level),
               'c4_warn_share': {r: v / len(names) for r, v in stats['c4_warn'].items()},
               'entries_outside_share': (stats['entries_out'] / stats['entries_tot']
                                         if stats['entries_tot'] else None),
               'entries_outside_what': fam_key,
               'placebo_cells_checked': stats['placebo_checked'],
               'duration_outside_envelope_share': (
                   stats['dur_outside'] / stats['dur_total'] if stats['dur_total']
                   else None),
               'setup_mean_ratio_to_stored': (float(np.mean(stats['setup_mean_ratio']))
                                              if stats['setup_mean_ratio'] else None),
               'setup_machines_charged': (sorted(set(stats['setup_machines']))
                                          if stats['setup_machines'] else None),
               'window_density_mean': (float(np.mean(stats['window_density']))
                                       if stats['window_density'] else None),
               'lag_mean_nonzero_h': (float(np.mean(stats['lag_mean_nonzero']))
                                      if stats['lag_mean_nonzero'] else None),
               'content_sha256': digest.hexdigest(),
               'stored_desc': STORED_DESC, 'stored_base': STORED_BASE,
               'generator_guard_checked': n_guard, 'commit': git_commit()}
        mp = os.path.join(out_dir, 'manifest.json')
        if os.path.exists(mp):
            old = json.load(open(mp))
            if old.get('content_sha256') != man['content_sha256']:
                sys.exit(f'REFUSING: {mp} records different content')
        write_json_atomic(mp, man, indent=1)
        print(f'[build] {level:17s} reached={",".join(reached(level)):15s} '
              f'mark={man["nominal_mark"][:30]:30s} C4 warn='
              + ' '.join(f'{r}:{v:.2f}' for r, v in man['c4_warn_share'].items())
              + (f' entries_out={man["entries_outside_share"]:.3f}'
                 if man['entries_outside_share'] is not None else '')
              + (f' dur_out={man["duration_outside_envelope_share"]:.3f}'
                 if man['duration_outside_envelope_share'] is not None else '')
              + (f' setup_ratio={man["setup_mean_ratio_to_stored"]:.3f}'
                 if man['setup_mean_ratio_to_stored'] is not None else '')
              + (f' win_density={man["window_density_mean"]:.3f}'
                 if man['window_density_mean'] is not None else '')
              + (f' lag_mean={man["lag_mean_nonzero_h"]:.1f}'
                 if man['lag_mean_nonzero_h'] is not None else ''), flush=True)


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------

def seed_sfx(seed):
    return '' if seed == 301 else f'-s{seed}'


# Checkpoint naming. 'legacy': the stored arms (legacy mask, seeds 301-303, and
# 304-305 of scripts/run_seeds45_queue.sh). 'xmask': the exact-mask arms of
# scripts/run_exactmask_queue.sh, to be evaluated with --mask exact.
ARM_NAMES = {'scheme': 'legacy'}


def model_name(arm, regime, seed):
    if ARM_NAMES['scheme'] == 'xmask':
        return {'pact': f'10x25+family+xmask-v1-s{seed}',
                'spec': f'10x25+family+xmask-solo-{regime}-s{seed}',
                'single': f'10x25+family+xmask-ctl-s{seed}'}[arm]
    if arm == 'pact':
        return f'10x25+family+jointblind-v1{seed_sfx(seed)}'
    if arm == 'spec':
        return f'10x25+family+spec-{regime}{seed_sfx(seed)}'
    if arm == 'single':
        return f'10x25+family+jointblind-nlsw-s{seed}'
    raise ValueError(arm)


def rule_function(mask):
    """The function that rolls out FIFO, MOR, SPT and MWKR under `mask` (RULE_SET),
    imported at the call."""
    from common_utils import heuristic_select_action, masked_rule_action
    return masked_rule_action if RULE_SET[mask] == 'masked' else heuristic_select_action


def record_rule_set(path):
    """The rule set the record of a rule result file states (the .json beside the .npz):
    'masked' or 'classical' ('classical' when the record has no field rule_set); None
    when there is no record."""
    jp = path[:-4] + '.json'
    if not os.path.exists(jp):
        return None
    with open(jp) as f:
        return json.load(f).get('rule_set', 'classical')


def other_rule_set_message(path, found, mask, root):
    return (f'REFUSING: {path} is a rule result of the rule set {found} '
            f'({RULE_FUNCTION.get(found, "no record")}); the {mask} mask uses the rule set '
            f'{RULE_SET[mask]} ({RULE_FUNCTION[RULE_SET[mask]]}). A rule file is never '
            f'overwritten: move the files of the earlier rule set first with '
            f'"scripts/p87_sensitivity.py archive-rules --root {root} --mask exact", then '
            f'evaluate the rule units (plan --only_rules).')


def units_for(level, regime, arms, seeds):
    """(arm, seed, method label) tuples evaluated on one (level, regime) cell."""
    out = []
    for arm in arms:
        if arm == 'rules':
            out += [('rule', None, r) for r in RULES]
        else:
            out += [(arm, s, 'greedy+' + model_name(arm, regime, s)) for s in seeds]
    return out


# ---------------------------------------------------------------------------
# phase: evaluate
# ---------------------------------------------------------------------------

# Per-instance counts of the evaluate option --trace_counts, with the definitions of
# the trace mode of scripts/p90_public_rule_comparison.py that
# scripts/p93_rule_comparison.py trace_stats reads:
#   n_decisions    decisions of the rollout (one operation dispatched per decision);
#   n_one_pair     decisions at which the mask admits exactly one (job, machine) pair,
#                  counted before the action from ~env.dynamic_pair_mask[0];
#   setup_sum      summed setup time charged to the dispatched operations, in hours:
#                  for each dispatch, read before the step, the true setup from the
#                  class the machine last held (the cold-start row when it is idle)
#                  into the operation's class, times the setup bit of the combination,
#                  times the machine's setup factor when the descriptor charges setups
#                  only on some machines (as FJSPEnvFamily._earliest_start; the
#                  factor is 1 on every set p90 traces);
#   machines_used  distinct machines that received an operation.
TRACE_FIELDS = ('n_decisions', 'n_one_pair', 'setup_sum', 'machines_used')


def count_decision(env, counts):
    """One decision, before the action; a single-pair decision when the mask
    admits exactly one pair."""
    legal = ~np.asarray(env.dynamic_pair_mask[0]).astype(bool)
    counts['decisions'] += 1
    counts['one_pair'] += int(legal.sum() == 1)


def count_setup(env, j, m, counts):
    """The setup the environment charges when the candidate of job j is dispatched
    on machine m, read before the step."""
    op = int(env.candidate[0, j])
    src = int(env.mch_last_class[0, m])
    s = float(env.true_setup[0, src, env.op_class[0, op]] * env.c_bits[0, 1])
    if getattr(env, '_has_setup_mult', False):
        s *= float(env.setup_mult[0, m])
    counts['setup'] += s


class Evaluator:
    """Rollouts with the configuration eval_family.py would use, one process."""

    ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
                 'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
                 'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
                 'layer_fea_output_dim', 'num_mlp_layers_actor',
                 'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
                 'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
                 'token_to_heads', 'reward_delta_env', 'delta_mode',
                 'mask_mode', 'pair_extra')

    def __init__(self, mask):
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cpu')
        if torch.cuda.is_available():
            sys.exit('REFUSING: CUDA is visible; run with CUDA_VISIBLE_DEVICES= '
                     '(this study is CPU only)')
        from params import configs
        import fjsp_env_same_op_nums
        assert str(fjsp_env_same_op_nums.EnvState.device) == 'cpu'
        if configs.seed_test != RULE_SEED:
            sys.exit(f'REFUSING: params.py seed_test is {configs.seed_test}; the stored rule '
                     f'arrays were drawn with {RULE_SEED}')
        self.torch = torch
        self.configs = configs
        self.defaults = copy.deepcopy(vars(configs))
        self.mask = mask
        self.policies = {}

    @property
    def rule_set(self):
        return RULE_SET[self.mask]

    def _reset_configs(self):
        d = vars(self.configs)
        d.clear()
        d.update(copy.deepcopy(self.defaults))
        self.configs.device = 'cpu'

    def policy(self, model):
        if model in self.policies:
            return self.policies[model]
        from constraint_family.fjsp_env_family import resolve_delta_mode
        from model.PPO import PPO_initialize
        with open(os.path.join(ROOT_DIR, 'train_log', 'FAMILY',
                               f'config_{model}.json')) as f:
            snap = json.load(f)
        for k in ('first_mask', 'mask_ties'):
            if snap.get(k) not in (None, self.mask):
                sys.exit(f'REFUSING: {model} trained with {k}={snap[k]}, this run '
                         f'evaluates with --mask {self.mask}')
        self._reset_configs()
        for k in self.ARCH_KEYS:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v in ('True', 'False'):
                    v = v == 'True'
                setattr(self.configs, k, v)
        self.configs.device = 'cpu'
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(self.torch.load(
            os.path.join(ROOT_DIR, 'trained_network', 'FAMILY', f'{model}.pth'),
            map_location='cpu'))
        ppo.policy.eval()
        n_par = sum(p.numel() for p in ppo.policy.parameters())
        if n_par != EXPECT_PARAMS:
            sys.exit(f'REFUSING: {model} has {n_par} parameters, expected '
                     f'{EXPECT_PARAMS}')
        dm = resolve_delta_mode(self.configs)
        rec = {'model': model, 'params': int(n_par), 'delta_mode': dm,
               'mask_mode': getattr(self.configs, 'mask_mode', 'nondelay'),
               'pair_extra': getattr(self.configs, 'pair_extra', 'none'),
               'snapshot_sha256': sha256_file(os.path.join(
                   ROOT_DIR, 'train_log', 'FAMILY', f'config_{model}.json')),
               'checkpoint_sha256': sha256_file(os.path.join(
                   ROOT_DIR, 'trained_network', 'FAMILY', f'{model}.pth')),
               'arch': {k: getattr(self.configs, k, None) for k in self.ARCH_KEYS}}
        self.policies[model] = (ppo, rec)
        return self.policies[model]

    def rollout(self, method, jl, pt, meta, desc, policy=None, counts=None):
        """One schedule. Returns (makespan, amch, op_ct, lb0_hours). counts: None, or
        a dict {'decisions': 0, 'one_pair': 0, 'setup': 0.0} that count_decision and
        count_setup fill at every decision (evaluate --trace_counts). A rule (policy
        None) is rolled out with the function of the mask's rule set (rule_function),
        np.random.seed(RULE_SEED) after the data are set, as eval_family.py does."""
        from common_utils import greedy_select_action
        from constraint_family.descriptor import compute_token
        from constraint_family.fjsp_env_family import FJSPEnvFamily
        fm = mt = self.mask
        if policy is not None:
            ppo, rec = policy
            mm, pe, dm = rec['mask_mode'], rec['pair_extra'], rec['delta_mode']
        else:
            mm, pe, dm = 'nondelay', 'none', 'min'
        env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                            blind_channels=(), mask_mode=mm, pair_extra=pe,
                            first_mask=fm, mask_ties=mt)
        env.delta_mode = dm
        env.reward_delta_env = (dm != 'zero')
        typed = meta['op_type'] is not None
        state = env.set_initial_data(
            [jl], [pt], descriptor_list=[desc],
            op_type_list=[meta['op_type']] if typed else None,
            mch_type_list=[meta['mch_type']] if typed else None)
        lb0 = float(np.max(env.op_ct_lb[0]) / env._slope)
        amch = np.full(pt.shape[0], -1, dtype=int)
        M = pt.shape[1]
        if policy is not None:
            env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()), M)])
            env._attach_token()
            state = env.state
            torch = self.torch
            while not env.done().all():
                if counts is not None:
                    count_decision(env, counts)
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                        mch_type=state.mch_type_tensor, token=state.token_tensor)
                a = greedy_select_action(pi)
                av = int(a.cpu().numpy()[0])
                if counts is not None:
                    count_setup(env, av // M, av % M, counts)
                amch[env.candidate[0, av // M]] = av % M
                state, _, _ = env.step(a.cpu().numpy())
        else:
            pick = rule_function(self.mask)
            np.random.seed(RULE_SEED)
            while not env.done().all():
                if counts is not None:
                    count_decision(env, counts)
                a = int(pick(method, env))
                if counts is not None:
                    count_setup(env, a // M, a % M, counts)
                amch[env.candidate[0, a // M]] = a % M
                env.step(np.array([a]))
        return float(env.current_makespan[0]), amch, env.true_op_ct[0].copy(), lb0


def load_cell(root, level, regime, n):
    """[(name, jl, pt, meta, desc)] of one (level, regime) cell; a placebo cell
    reads the stored descriptors (identical by the build's assertion)."""
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import load_instance
    names = level_names(level, n)
    if level != 'default' and regime in reached(level):
        ddir = os.path.join(level_data_dir(root, level), 'desc', regime)
    else:
        ddir = os.path.join(STORED_DESC, 'desc', regime)
    out = []
    for name in names:
        p = os.path.join(ddir, f'{name}.desc.json')
        if not os.path.exists(p):
            sys.exit(f'REFUSING: {p} missing; run the build phase first')
        with open(p) as f:
            rec = json.load(f)
        if level != 'default' and regime not in reached(level) \
                and LEVELS[level]['family'] in ('dur', 'contention'):
            raise AssertionError('dur/contention levels reach every regime')
        jl, pt, meta = load_instance(os.path.join(ROOT_DIR, rec['base_instance']))
        out.append((name, jl, pt, meta,
                    ConstraintDescriptor.from_json(rec, setup_convention='attached')))
    return out


def result_path(root, mask, level, regime, label):
    return os.path.join(root, 'results', mask, level, regime, f'{label}.npz')


def _has_trace(path):
    with np.load(path) as z:
        return all(k in z.files for k in TRACE_FIELDS)


def evaluate_unit(ev, root, level, regime, arm, seed, label, cell, force=False,
                  trace=False):
    """One method on one cell. trace=True (evaluate --trace_counts) also stores the
    per-instance counts of TRACE_FIELDS; a stored file without them is then redone.
    A rule file is skipped (resume) only when its record states the mask's rule set; a
    rule file of the other rule set is refused, with --force as well."""
    from constraint_family import checks
    from ppvc_instance_generator import STATION_TYPES
    path = result_path(root, ev.mask, level, regime, label)
    names = [c[0] for c in cell]
    if arm == 'rule' and (os.path.exists(path) or os.path.exists(path[:-4] + '.json')):
        found = record_rule_set(path)
        if found != ev.rule_set:
            sys.exit(other_rule_set_message(path, found, ev.mask, root))
    if os.path.exists(path) and not force:
        old = json.load(open(path[:-4] + '.json'))
        if old['instances'] == names and (not trace or _has_trace(path)):
            return 'skip', 0.0, 0.0
    t_unit, c_unit = time.perf_counter(), time.process_time()
    policy = None
    meta_rec = {'arm': arm, 'seed': seed, 'label': label}
    if arm != 'rule':
        policy = ev.policy(model_name(arm, regime, seed))
        meta_rec.update(policy[1])
    else:
        meta_rec.update(rule_set=ev.rule_set, rule_function=RULE_FUNCTION[ev.rule_set],
                        rule_seed=RULE_SEED)
    n = len(cell)
    ms = np.zeros(n)
    wall = np.zeros(n)
    cpu = np.zeros(n)
    lb0 = np.zeros(n)
    util = np.zeros(n)
    util_max_type = np.zeros(n)
    c6_ok = np.zeros(n, dtype=bool)
    amchs = np.zeros((n, cell[0][2].shape[0]), dtype=np.int8)
    if trace:
        tr = {k: np.zeros(n, dtype=float if k == 'setup_sum' else np.int64)
              for k in TRACE_FIELDS}
    for i, (name, jl, pt, meta, desc) in enumerate(cell):
        t0, c0 = time.perf_counter(), time.process_time()
        counts = {'decisions': 0, 'one_pair': 0, 'setup': 0.0} if trace else None
        if trace:
            m, amch, op_ct, lb = ev.rollout(label if arm == 'rule' else 'greedy',
                                            jl, pt, meta, desc, policy, counts=counts)
        else:
            m, amch, op_ct, lb = ev.rollout(label if arm == 'rule' else 'greedy',
                                            jl, pt, meta, desc, policy)
        wall[i] = time.perf_counter() - t0
        cpu[i] = time.process_time() - c0
        op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
        c6 = checks.check_schedule_validation(desc, jl, pt, amch, op_ct, op_st=op_st)
        c6_ok[i] = c6.status == checks.PASS
        if not c6_ok[i]:
            print(f'  C6 FAIL {level} {regime} {label} {name}: {c6.messages[:1]}',
                  flush=True)
        if abs(c6.details['makespan'] - m) > 1e-6:
            sys.exit(f'REFUSING: env makespan {m} != validator {c6.details["makespan"]}')
        ms[i], lb0[i], amchs[i] = m, lb, amch
        busy = pt[np.arange(pt.shape[0]), amch].astype(float)
        util[i] = busy.sum() / (pt.shape[1] * m)
        mt = np.asarray(meta['mch_type'])
        per_type = [busy[np.isin(amch, np.flatnonzero(mt == t))].sum()
                    / ((mt == t).sum() * m)
                    for t in range(len(STATION_TYPES)) if (mt == t).any()]
        util_max_type[i] = max(per_type)
        if trace:
            tr['n_decisions'][i] = counts['decisions']
            tr['n_one_pair'][i] = counts['one_pair']
            tr['setup_sum'][i] = counts['setup']
            tr['machines_used'][i] = len(np.unique(amch))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + '.__tmp.npz'
    if trace:
        np.savez(tmp, makespan=ms, wall=wall, cpu=cpu, lb0=lb0, util=util,
                 util_max_type=util_max_type, c6_ok=c6_ok, amch=amchs, **tr)
    else:
        np.savez(tmp, makespan=ms, wall=wall, cpu=cpu, lb0=lb0, util=util,
                 util_max_type=util_max_type, c6_ok=c6_ok, amch=amchs)
    os.replace(tmp, path)
    meta_rec.update({'level': level, 'regime': regime, 'mask': ev.mask,
                     'instances': names, 'n': n, 'c6_fail': int((~c6_ok).sum()),
                     'mean_makespan': float(ms.mean()),
                     'wall_s': float(wall.sum()), 'cpu_s': float(cpu.sum()),
                     'cpu_share': float(cpu.sum() / max(wall.sum(), 1e-9)),
                     'affinity': sorted(os.sched_getaffinity(0)),
                     'loadavg': list(os.getloadavg()),
                     'host': socket.gethostname(), 'python': platform.python_version(),
                     'torch': ev.torch.__version__,
                     'torch_threads': ev.torch.get_num_threads(),
                     'omp': os.environ.get('OMP_NUM_THREADS'),
                     'commit': git_commit(),
                     'unit_elapsed_s': time.perf_counter() - t_unit,
                     'unit_cpu_s': time.process_time() - c_unit,
                     'finished': time.strftime('%Y-%m-%d %H:%M:%S')})
    if trace:
        meta_rec['trace_counts'] = list(TRACE_FIELDS)
    write_json_atomic(path[:-4] + '.json', meta_rec, indent=1)
    return 'done', float(wall.sum()), float(cpu.sum())


def all_units(levels, arms, seeds):
    out = []
    for level in levels:
        for regime in reached(level):
            for (arm, seed, label) in units_for(level, regime, arms, seeds):
                out.append((level, regime, arm, seed, label))
    return out


def stale_rule_files(root, mask, units):
    """Rule result files of `units` on disk whose record does not state the rule set of
    `mask` (or that have no record)."""
    out = []
    for (level, regime, arm, seed, label) in units:
        if arm != 'rule':
            continue
        p = result_path(root, mask, level, regime, label)
        if (os.path.exists(p) or os.path.exists(p[:-4] + '.json')) \
                and record_rule_set(p) != RULE_SET[mask]:
            out.append(p)
    return out


def evaluate(args):
    ev = Evaluator(args.mask)
    only_rules = bool(getattr(args, 'only_rules', False))
    if args.part is not None:
        ppath = plan_path(args.root, only_rules)
        if not os.path.exists(ppath):
            sys.exit(f'REFUSING: {ppath} not found; run the plan phase'
                     + (' with --only_rules' if only_rules else '') + ' first')
        plan = json.load(open(ppath))
        if bool(plan.get('only_rules', False)) != only_rules:
            sys.exit(f'REFUSING: {ppath} does not match --only_rules')
        if plan['parts'] != args.parts:
            sys.exit(f'REFUSING: plan has {plan["parts"]} parts, not {args.parts}')
        if plan['mask'] != args.mask or plan.get('arm_names', 'legacy') \
                != args.arm_names:
            sys.exit('REFUSING: plan was made for another mask or arm naming')
        if bool(plan.get('trace_counts', False)) != bool(args.trace_counts):
            sys.exit('REFUSING: the plan was made with' + ('' if plan.get('trace_counts')
                                                          else 'out') +
                     ' --trace_counts; launch every part the same way')
        units = [tuple(u) for u in plan['assignment'][str(args.part)]]
        n = plan['n']
    else:
        levels = args.levels or list(LEVELS)
        units = [u for u in all_units(levels, ['rules'] if only_rules else args.arms,
                                      args.seeds)
                 if not args.regimes or u[1] in args.regimes]
        n = args.n
    stale = stale_rule_files(args.root, args.mask, units)
    if stale:
        sys.exit(f'REFUSING: {len(stale)} rule result files of this run hold another rule '
                 f'set than {RULE_SET[args.mask]}; nothing was evaluated. First of them:\n'
                 + other_rule_set_message(stale[0], record_rule_set(stale[0]), args.mask,
                                          args.root))
    cell_cache = {}
    t_start = time.perf_counter()
    for k, (level, regime, arm, seed, label) in enumerate(units):
        key = (level, regime)
        if key not in cell_cache:
            cell_cache.clear()
            cell_cache[key] = load_cell(args.root, level, regime, n)
        if args.trace_counts:
            st, w, c = evaluate_unit(ev, args.root, level, regime, arm, seed, label,
                                     cell_cache[key], force=args.force, trace=True)
        else:
            st, w, c = evaluate_unit(ev, args.root, level, regime, arm, seed, label,
                                     cell_cache[key], force=args.force)
        print(f'[eval {k + 1}/{len(units)}] {level:17s} {regime:4s} {label:45s} {st} '
              f'wall={w:7.1f}s cpu={c:7.1f}s '
              f'elapsed={time.perf_counter() - t_start:8.0f}s', flush=True)


# ---------------------------------------------------------------------------
# phase: timing (planning figures from finished units; not a measurement)
# ---------------------------------------------------------------------------

def timing(args):
    recs = [json.load(open(p)) for p in glob.glob(os.path.join(
        args.root, 'results', args.mask, '*', '*', '*.json'))]
    if not recs:
        sys.exit(f'no finished units under {args.root}/results/{args.mask}')
    out = {}
    for kind, sel in (('policy', lambda r: r['arm'] != 'rule'),
                      ('rule', lambda r: r['arm'] == 'rule')):
        rs_all = [r for r in recs if sel(r)]
        # units that shared their core with another process (cpu_share < 0.9)
        # would inflate the planning figure; use the others when there are any
        rs = [r for r in rs_all if r['cpu_share'] >= 0.9] or rs_all
        out[f'{kind}_units_excluded_contended'] = len(rs_all) - len(rs)
        if not rs:
            continue
        n = sum(r['n'] for r in rs)
        el = sum(r.get('unit_elapsed_s', r['wall_s']) for r in rs)
        out[f'{kind}_s_per_instance'] = el / n
        out[f'{kind}_rollout_wall_s_per_instance'] = sum(r['wall_s'] for r in rs) / n
        out[f'{kind}_cpu_share'] = (sum(r['cpu_s'] for r in rs)
                                    / max(sum(r['wall_s'] for r in rs), 1e-9))
        out[f'{kind}_units'] = len(rs)
        out[f'{kind}_instances'] = n
    out['by_regime_policy_s'] = {
        reg: (sum(r.get('unit_elapsed_s', r['wall_s']) for r in recs
                  if r['arm'] != 'rule' and r['regime'] == reg)
              / max(1, sum(r['n'] for r in recs
                           if r['arm'] != 'rule' and r['regime'] == reg)))
        for reg in R8 if any(r['regime'] == reg for r in recs)}
    out['affinity'] = sorted({c for r in recs for c in r['affinity']})
    out['loadavg_seen'] = [min(r['loadavg'][0] for r in recs),
                           max(r['loadavg'][0] for r in recs)]
    out['note'] = ('planning figures: unit elapsed time per instance (rollout, C6 '
                   'validation, model load), on the cores listed; rollouts are '
                   'deterministic, so these are not paper measurements')
    write_json_atomic(os.path.join(args.root, 'timing.json'), out, indent=1)
    print(json.dumps(out, indent=1))


# ---------------------------------------------------------------------------
# phase: plan
# ---------------------------------------------------------------------------

def plan_path(root, only_rules):
    """The plan of the evaluate parts: plan.json, or plan_rules.json for the rule units
    alone (plan --only_rules), so that the two plans never replace each other."""
    return os.path.join(root, 'plan_rules.json' if only_rules else 'plan.json')


def plan(args):
    """Phase plan: the units in --parts parts (longest first), written to plan_path and
    printed as evaluate commands, one per part, each pinned by the caller (<CORE_i>).
    --only_rules: the rule units alone (FIFO, MOR, SPT, MWKR on every combination of the
    levels), in plan_rules.json, logs rules_part_<i>.log. --dry: print, write nothing."""
    only_rules = bool(getattr(args, 'only_rules', False))
    dry = bool(getattr(args, 'dry', False))
    levels = args.levels or list(LEVELS)
    arms = list(args.arms)
    if only_rules:
        if arms not in (DEFAULT_ARMS.split(','), ['rules']):
            sys.exit(f'REFUSING: --only_rules plans the rule units alone; --arms {arms}')
        arms = ['rules']
    units = all_units(levels, arms, args.seeds)
    tpath = args.timing or os.path.join(args.root, 'timing.json')
    if os.path.exists(tpath):
        tm = json.load(open(tpath))
        per = {'policy': tm['policy_s_per_instance'], 'rule': tm['rule_s_per_instance']}
        src = f'measured ({tpath})'
    else:
        # rule: 20.8 s per unit of 100 instances, the rule units of 2026-11-15
        per = {'policy': 1.0, 'rule': 0.21}
        src = 'default guess'
    n = args.n or 100

    n_of = {lv: len(level_names(lv, n)) for lv in levels}

    def cost(u):
        return n_of[u[0]] * (per['rule'] if u[2] == 'rule' else per['policy'])
    # longest-processing-time assignment, units of one cell kept adjacent
    order = sorted(units, key=lambda u: -cost(u))
    loads = [0.0] * args.parts
    assign = {str(i): [] for i in range(args.parts)}
    for u in order:
        i = int(np.argmin(loads))
        assign[str(i)].append(list(u))
        loads[i] += cost(u)
    for i in assign:
        assign[i].sort(key=lambda u: (list(LEVELS).index(u[0]), R8.index(u[1]),
                                      u[4]))
    total = sum(cost(u) for u in units)
    out = {'parts': args.parts, 'n': n, 'mask': args.mask,
           'arm_names': args.arm_names, 'levels': levels,
           'arms': arms, 'seeds': args.seeds, 'n_units': len(units),
           'rule_set': RULE_SET[args.mask],
           'cost_source': src, 'per_instance_s': per,
           'est_total_core_h': total / 3600,
           'est_part_h': [x / 3600 for x in loads], 'assignment': assign}
    if only_rules:
        out['only_rules'] = True
    if getattr(args, 'trace_counts', False):
        out['trace_counts'] = True
    ppath = plan_path(args.root, only_rules)
    if not dry:
        write_json_atomic(ppath, out, indent=1)
    n_pol = sum(1 for u in units if u[2] != 'rule')
    n_cells = len({(u[0], u[1]) for u in units})
    print(f'[plan] {len(levels)} levels, {n_cells} combinations, {len(units)} result files '
          f'({n_pol} policy, {len(units) - n_pol} rule, rule set {RULE_SET[args.mask]}), '
          f'{n} instances each (the project-mix levels: their 50)')
    print(f'[plan] per-instance cost {per} from {src}')
    print(f'[plan] estimated {total / 3600:.2f} core-hours; {args.parts} parts of '
          f'{min(loads) / 3600:.2f} to {max(loads) / 3600:.2f} h')
    print('[plan] ' + (f'--dry: nothing written; without --dry the plan goes to {ppath}'
                       if dry else f'wrote {ppath}'))
    py = '$HOME/miniconda3/envs/fjsp/bin/python'
    if not dry:
        os.makedirs(os.path.join(args.root, 'logs'), exist_ok=True)
    log = 'rules_part' if only_rules else 'part'
    for i in range(args.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 '
              f'taskset -c <CORE_{i}> {py} scripts/p87_sensitivity.py evaluate '
              f'--root {args.root} --mask {args.mask} --arm_names {args.arm_names} '
              f'--part {i} --parts {args.parts} '
              + ('--only_rules ' if only_rules else '')
              + ('--trace_counts ' if getattr(args, 'trace_counts', False) else '')
              + f'> {args.root}/logs/{log}_{i}.log 2>&1 &')


# ---------------------------------------------------------------------------
# phase: archive-rules (the rule files of the earlier rule set, exact mask)
# ---------------------------------------------------------------------------

def archive_rules(args):
    """Phase archive-rules, --mask exact only. MOVES every rule result file of
    <root>/results/exact/<level>/<R>/ (<RULE>.npz and <RULE>.json, RULE in RULES) whose
    record does not state the rule set masked, that is the files evaluate wrote with
    common_utils.heuristic_select_action, to <root>/results/exact_rules_classical/
    <level>/<R>/, and writes there manifest.json (each file with its size and SHA-256,
    paths relative to the root). Nothing is deleted or overwritten: when a target file or
    the manifest exists the phase refuses before moving anything. --dry prints the same
    counts and moves nothing. Returns dict(moved=[(from, to)], kept_masked, dry)."""
    if args.mask != 'exact':
        sys.exit('REFUSING: archive-rules is for --mask exact (the legacy mask keeps the '
                 'classical rules of the earlier study)')
    dry = bool(getattr(args, 'dry', False))
    src = os.path.join(args.root, 'results', 'exact')
    dst = os.path.join(args.root, 'results', ARCHIVE_RULES)
    moves, kept, per_level = [], 0, {}
    for level in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        for regime in sorted(os.listdir(os.path.join(src, level))):
            d = os.path.join(src, level, regime)
            if not os.path.isdir(d):
                continue
            for r in RULES:
                files = [os.path.join(d, r + e) for e in ('.npz', '.json')
                         if os.path.exists(os.path.join(d, r + e))]
                if not files:
                    continue
                if record_rule_set(os.path.join(d, r + '.npz')) == 'masked':
                    kept += 1
                    continue
                pl = per_level.setdefault(level, {'combinations': set(), 'rules': 0})
                pl['combinations'].add(regime)
                pl['rules'] += 1
                moves += [(f, os.path.join(dst, level, regime, os.path.basename(f)))
                          for f in files]
    manifest = os.path.join(dst, 'manifest.json')
    exist = [t for _, t in moves if os.path.exists(t)]
    if moves and os.path.exists(manifest):
        exist.append(manifest)
    word = 'would move' if dry else 'moved'
    if exist:
        sys.exit(f'REFUSING (archive-rules): {len(exist)} target files exist, e.g. '
                 f'{exist[:3]}; nothing was moved')
    by_rule = {r: sum(1 for f, _ in moves if os.path.basename(f) == r + '.npz')
               for r in RULES}
    for level, pl in per_level.items():
        print(f'  {level:17s} {len(pl["combinations"])} combinations, {pl["rules"]} rule '
              f'results ({", ".join(sorted(pl["combinations"], key=R8.index))})')
    print(f'[archive-rules] {word} {len(moves)} files ('
          + ', '.join(f'{r} {k}' for r, k in by_rule.items())
          + f' rule results; {sum(len(p["combinations"]) for p in per_level.values())} '
          f'combinations of {len(per_level)} levels) from {src} to {dst}; kept {kept} rule '
          f'results that state the rule set masked')
    if dry or not moves:
        return dict(moved=moves, kept_masked=kept, dry=dry)
    entries = [{'from': os.path.relpath(f, args.root), 'to': os.path.relpath(t, args.root),
                'bytes': os.path.getsize(f), 'sha256': sha256_file(f)} for f, t in moves]
    for f, t in moves:
        os.makedirs(os.path.dirname(t), exist_ok=True)
        os.rename(f, t)
    write_json_atomic(manifest, {
        'generator': 'scripts/p87_sensitivity.py archive-rules',
        'moved': time.strftime('%Y-%m-%d %H:%M:%S'),
        'reason': ('rule results of the exact mask whose record does not state the rule set '
                   'masked: rolled out with common_utils.heuristic_select_action; the exact '
                   'environment uses common_utils.masked_rule_action'),
        'kept_masked': kept, 'files': entries}, indent=1)
    print(f'[archive-rules] wrote {manifest}')
    return dict(moved=moves, kept_masked=kept, dry=dry)


# ---------------------------------------------------------------------------
# phase: analyze
# ---------------------------------------------------------------------------

def ci(x):
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(x), (B, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(x.mean()), float(lo), float(hi)


def verdict(lo, hi, margin):
    """The paper's rule (scripts/p11_equivalence.py), at a given margin."""
    if hi < 0:
        return 'SUPERIOR'
    if -margin < lo and hi < margin:
        return 'EQUIVALENT'
    if hi < margin:
        return 'non-inferior'
    if lo > margin:
        return 'INFERIOR'
    return 'INCONCLUSIVE'


def lead_word(lo, hi):
    return 'ahead' if lo > 0 else ('behind' if hi < 0 else 'unresolved')


def load_result(root, mask, level, regime, label):
    lv = level if (level == 'default' or regime in reached(level)) else 'default'
    p = result_path(root, mask, lv, regime, label)
    if not os.path.exists(p):
        return None
    if label in RULES and record_rule_set(p) != RULE_SET[mask]:
        sys.exit(other_rule_set_message(p, record_rule_set(p), mask, root))
    z = np.load(p)
    meta = json.load(open(p[:-4] + '.json'))
    return {'ms': z['makespan'].astype(float), 'c6_fail': int((~z['c6_ok']).sum()),
            'instances': meta['instances'], 'lb0': z['lb0'], 'util': z['util'],
            'util_max_type': z['util_max_type'], 'placebo': lv != level}


def pooled(root, mask, level, regime, arm, seeds):
    rs = [load_result(root, mask, level, regime,
                      'greedy+' + model_name(arm, regime, s)) for s in seeds]
    if any(r is None for r in rs):
        return None
    inst = rs[0]['instances']
    if any(r['instances'] != inst for r in rs):
        sys.exit(f'REFUSING: seeds of {arm} on {level}/{regime} cover different '
                 f'instances')
    return {'ms': np.mean([r['ms'] for r in rs], axis=0), 'instances': inst,
            'c6_fail': sum(r['c6_fail'] for r in rs), 'n_seeds': len(rs),
            'lb0': rs[0]['lb0'], 'util': np.mean([r['util'] for r in rs], axis=0),
            'util_max_type': np.mean([r['util_max_type'] for r in rs], axis=0),
            'placebo': rs[0]['placebo']}


def analyze_cell(root, mask, level, regime, seeds):
    P = pooled(root, mask, level, regime, 'pact', seeds)
    S = pooled(root, mask, level, regime, 'spec', seeds)
    C = pooled(root, mask, level, regime, 'single', seeds)
    R = {r: load_result(root, mask, level, regime, r) for r in RULES}
    if P is None or S is None or any(v is None for v in R.values()):
        return None
    inst = P['instances']
    for x in [S] + ([C] if C else []) + list(R.values()):
        if x['instances'] != inst:
            sys.exit(f'REFUSING: arms of {level}/{regime} cover different instances')
    if P['n_seeds'] != S['n_seeds']:
        sys.exit('REFUSING: asymmetric seed pools')
    p = P['ms']
    out = {'level': level, 'regime': regime, 'n': len(inst),
           'placebo': P['placebo'], 'held_out': regime in HELD_OUT,
           'seeds': list(seeds), 'pact_mean': float(p.mean()),
           'spec_mean': float(S['ms'].mean())}
    m, lo, hi = ci(100 * (p - S['ms']) / S['ms'])
    out['a'] = {'mean': m, 'lo': lo, 'hi': hi,
                **{f'verdict_{mg:g}': verdict(lo, hi, mg) for mg in MARGINS}}
    pool = np.min(np.stack([R[r]['ms'] for r in RULES]), axis=0)
    m, lo, hi = ci(100 * (pool - p) / p)
    out['b'] = {'mean': m, 'lo': lo, 'hi': hi, 'reading': lead_word(lo, hi)}
    best = min(RULES, key=lambda r: R[r]['ms'].mean())
    m, lo, hi = ci(100 * (R[best]['ms'] - p) / p)
    out['b_best_rule'] = {'rule': best, 'mean': m, 'lo': lo, 'hi': hi,
                          'reading': lead_word(lo, hi)}
    if C is not None:
        m, lo, hi = ci(100 * (p - C['ms']) / C['ms'])
        out['d'] = {'mean': m, 'lo': lo, 'hi': hi,
                    **{f'verdict_{mg:g}': verdict(lo, hi, mg) for mg in MARGINS}}
    out['c6_fail'] = (P['c6_fail'] + S['c6_fail'] + (C['c6_fail'] if C else 0)
                      + sum(R[r]['c6_fail'] for r in RULES))
    out['pact_over_lb0'] = float(np.mean(p / P['lb0']))
    out['util_mean'] = float(P['util'].mean())
    out['util_max_type'] = float(P['util_max_type'].mean())
    out['_vec'] = {'a': 100 * (p - S['ms']) / S['ms'], 'b': 100 * (pool - p) / p}
    return out


def analyze(args):
    levels = args.levels or list(LEVELS)
    rows = []
    base = {}
    for level in levels:
        for regime in R8:
            c = analyze_cell(args.root, args.mask, level, regime, args.seeds)
            if c is None:
                continue
            if level == 'default':
                base[regime] = c
            rows.append(c)
    for c in rows:
        b = base.get(c['regime'])
        if b is not None and c['level'] != 'default' \
                and paired_with_default(c['level']) and not c['placebo']:
            if b['n'] != c['n']:
                sys.exit(f'REFUSING: {c["level"]} {c["regime"]} has {c["n"]} '
                         f'instances, the default level {b["n"]}; a shift needs the '
                         f'same instances')
            for k in ('a', 'b'):
                m, lo, hi = ci(c['_vec'][k] - b['_vec'][k])
                c[f'shift_{k}'] = {'mean': m, 'lo': lo, 'hi': hi}
    for c in rows:
        c.pop('_vec', None)
    marks = {}
    for level in levels:
        mp = os.path.join(level_data_dir(args.root, level), 'manifest.json')
        marks[level] = (json.load(open(mp)) if level != 'default' and os.path.exists(mp)
                        else {'nominal_mark': 'inside', 'c4_warn_share': {}})
    adir = os.path.join(args.root, 'analysis', args.mask)
    os.makedirs(adir, exist_ok=True)
    write_json_atomic(os.path.join(adir, 'cells.json'),
                      {'rows': rows, 'marks': {k: {kk: v[kk] for kk in v
                                                   if kk != 'instances'}
                                               for k, v in marks.items()}}, indent=1)

    def f(x):
        return f'{x["mean"]:+.2f} [{x["lo"]:+.2f},{x["hi"]:+.2f}]'
    L = ['# P87 sensitivity: summary', '',
         f'mask = {args.mask}; seeds = {args.seeds}; estimator: per-instance relative '
         f'gap, policy arms averaged over seeds, 95% percentile bootstrap '
         f'({B} resamples, default_rng(7) per interval).', '',
         '(a) policy minus specialist, % (negative = policy faster), verdicts at '
         '0.5% / 1%; (b) lead over the rule pool, % (positive = policy ahead); '
         '(d) policy minus single-rule control, %. "placebo": the level cannot reach '
         'this combination, so the cell is the default cell. Shift: change of (a) '
         'and (b) against the default level, paired by instance. Mark: the '
         'nominal mark of the level (mean of the perturbed distribution against the '
         'generator range). Outside share: entries of the perturbed rule outside the '
         'generator range (lags, setup entries, per-machine outage densities), or '
         'processing times outside the per-step envelope for the duration levels. '
         'ms / LB0: policy makespan over the makespan estimate at the start (the '
         'job-chain bound with lags and minimum setups); util max type: utilization '
         'of the busiest station type under the policy.', '',
         '| level | mark | outside share | regime | n | (a) policy - specialist | '
         'verdict 0.5 / 1 | shift (a) | (b) lead over rule pool | shift (b) | '
         '(d) policy - single-rule | C6 fail | ms / LB0 | util max type |',
         '|' + '---|' * 14]
    for c in rows:
        mk = marks[c['level']]
        share = mk.get('entries_outside_share')
        if share is None:
            share = mk.get('duration_outside_envelope_share')
        L.append(
            f'| {c["level"]} | '
            f'{"placebo" if c["placebo"] else mk["nominal_mark"].split(" (")[0]} | '
            f'{"" if share is None or c["placebo"] else f"{share:.2f}"} | {c["regime"]}'
            f'{" (held out)" if c["held_out"] else ""}'
            f'{" placebo" if c["placebo"] else ""} | {c["n"]} | {f(c["a"])} | '
            f'{c["a"]["verdict_0.5"]} / {c["a"]["verdict_1"]} | '
            f'{f(c["shift_a"]) if "shift_a" in c else ""} | '
            f'{f(c["b"])} {c["b"]["reading"]} | '
            f'{f(c["shift_b"]) if "shift_b" in c else ""} | '
            f'{f(c["d"]) if "d" in c else ""} | {c["c6_fail"]} | '
            f'{c["pact_over_lb0"]:.3f} | {c["util_max_type"]:.2f} |')
    # verdict stability against the default level
    L += ['', '## Verdict changes against the default level', '']
    for c in rows:
        b = base.get(c['regime'])
        if c['level'] == 'default' or b is None or c['placebo']:
            continue
        ch = [f'(a)@{mg:g}: {b["a"][f"verdict_{mg:g}"]} -> {c["a"][f"verdict_{mg:g}"]}'
              for mg in MARGINS if b['a'][f'verdict_{mg:g}'] != c['a'][f'verdict_{mg:g}']]
        if b['b']['reading'] != c['b']['reading']:
            ch.append(f'(b): {b["b"]["reading"]} -> {c["b"]["reading"]}')
        if ch:
            L.append(f'- {c["level"]} {c["regime"]}: ' + '; '.join(ch))
    with open(os.path.join(adir, 'summary.md'), 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {adir}/summary.md and cells.json')


# ---------------------------------------------------------------------------
# phase: compare (default level against the stored arrays)
# ---------------------------------------------------------------------------

def compare(args):
    """The default level against the stored arrays of the same environment: the legacy
    arrays (test_results/FAMILY/10x25 and its CPU reference) for --mask legacy, the arrays
    of eval_family.py with the exact mask (STORED_XMASK; rules: --rule_set masked) for
    --mask exact."""
    L = []
    refs = (STORED_XMASK,) if args.mask == 'exact' else (STORED_RES, STORED_CPU)
    for regime in R8:
        for (arm, seed, label) in units_for('default', regime, args.arms, args.seeds):
            p = result_path(args.root, args.mask, 'default', regime, label)
            if not os.path.exists(p):
                continue
            z = np.load(p)['makespan']
            n = len(z)
            for ref in refs:
                rp = os.path.join(ref, f'Result_{label}_{regime}.npy')
                if not os.path.exists(rp):
                    continue
                a = np.load(rp)
                a = (a[:, 0] if a.ndim > 1 else a)[:n].astype(float)
                d = z - a
                L.append(f'{regime:4s} {label:45s} vs {os.path.basename(ref):12s} '
                         f'n={n:3d} equal={int((d == 0).sum()):3d} '
                         f'differ={int((d != 0).sum()):3d} max|d|={np.abs(d).max():5.1f} h '
                         f'mean d={d.mean():+6.2f} h  mean ours={z.mean():6.1f} '
                         f'stored={a.mean():6.1f}')
    out = os.path.join(args.root, 'analysis', args.mask, 'compare_stored.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'wrote {out}')


# ---------------------------------------------------------------------------
# phase: infer (the statistical protocol of the manuscript; module doc)
# ---------------------------------------------------------------------------

INFER_GENERATOR = 'scripts/p87_sensitivity.py infer'
INFER_FILES = {'macros': 'manuscript/macros_sensitivity.tex',
               'json': 'results/sensitivity_inference.json',
               'md': 'results/sensitivity_inference.md',
               'csv': 'results/sensitivity_inference.csv',
               'table': 'manuscript/tables/sensitivity_table.tex'}
INFER_GUARDED = ('results', 'test_results', 'or_solution', 'trained_network',
                 'train_log', 'data')
INFER_ARMS = ('pact', 'spec', 'single')
HIER_SEED, INST_SEED, ALPHA = 11, 7, 0.05
MW = {0.5: 'Half', 1.0: 'One'}
# comparison key -> (first-named method, second, macro word)
COMPARISONS = {'pol_spec': ('policy', 'specialist', 'PolSpec'),
               'ctl_spec': ('single-rule control', 'specialist', 'CtlSpec'),
               'pol_rule': ('policy', 'best dispatching rule', 'PolRule')}
POOL_ALL, POOL_REACHED = 'pool', 'reached'
ROW_WORDS = {POOL_ALL: 'Pool', POOL_REACHED: 'Reached'}
LEVEL_WORDS = {
    'default': 'Default', 'dur_cv10': 'DurCvTen', 'dur_cv20': 'DurCvTwenty',
    'dur_cv30': 'DurCvThirty', 'dur_long_m25': 'DurLongDown', 'dur_long_p25': 'DurLongUp',
    'lag_x050': 'LagHalf', 'lag_x075': 'LagThreeQuarters', 'lag_x125': 'LagFiveQuarters',
    'lag_x150': 'LagThreeHalves', 'setup_x050': 'SetupHalf',
    'setup_x075': 'SetupThreeQuarters', 'setup_x125': 'SetupFiveQuarters',
    'setup_x150': 'SetupThreeHalves', 'setup_same0': 'SetupSameZero',
    'setup_sub_strict': 'SetupShell', 'setup_sub_wide': 'SetupProduction',
    'win_x050': 'WinHalf', 'win_x150': 'WinThreeHalves', 'win_x200': 'WinDouble',
    'tight15': 'Tight', 'mix_rc': 'MixConcrete', 'mix_steel': 'MixSteel'}
# stations whose type is charged with setups (module doc: 7 and 21 of 25)
SUBSET_STATIONS = {'setup_sub_strict': 7, 'setup_sub_wide': 21}
MIX_WORDS = {'mix_rc': 'reinforced concrete', 'mix_steel': 'steel'}
# the two measured variables (infer_measured), for the policy and the best rule's
# schedule; macros for the default level, the setup scales and the shares of stations
# with setups
MEASURED = ('pol_one_pair', 'pol_setup_share', 'rule_one_pair', 'rule_setup_share')
MEASURED_WORDS = {'pol_one_pair': 'PolOnePair', 'pol_setup_share': 'PolSetupShare',
                  'rule_one_pair': 'RuleOnePair', 'rule_setup_share': 'RuleSetupShare'}
MEASURED_MACRO_LEVELS = ('default', 'setup_x050', 'setup_x075', 'setup_x125',
                         'setup_x150', 'setup_sub_strict', 'setup_sub_wide')
CSV_FIELDS = ('level', 'family', 'parameter', 'value', 'mark', 'outside_share',
              'comparison', 'first', 'second', 'row', 'reached', 'k', 'n', 'c', 'mean',
              'lo', 'hi', 't_lo', 't_hi', 'direction', 'verdict_0.5', 'verdict_1',
              'verdict', 'p_t_holm', 'p_boot_holm', 'p_t_greater_holm',
              'p_boot_greater_holm') + MEASURED
MIX_NOTE = ('not computed: the level uses other base instances than the default level, '
            'so its verdicts are not on the same instances')


class InferRefusal(SystemExit):
    """A precondition of the infer phase failed; its outputs are removed."""


def infer_refuse(msg):
    raise InferRefusal(f'REFUSING (infer): {msg}')


def _stats():
    """stats_family and the verdict rule and number formats of p89_public_family
    (scripts/ is appended to the search path, so the repository's own modules keep
    precedence)."""
    d = os.path.join(ROOT_DIR, 'scripts')
    if d not in sys.path:
        sys.path.append(d)
    import stats_family as SF
    from p89_public_family import VERDICT_WORDS, f2, jsonable, superiority_verdict, u1
    return SF, VERDICT_WORDS, f2, jsonable, superiority_verdict, u1


def level_parameter(level):
    """(parameter, value) the level sets, in plain words, for the tables."""
    spec = LEVELS[level]
    fam = spec['family']
    if fam == 'default':
        return 'none', ''
    if fam == 'dur':
        if spec['kind'] == 'cv':
            return 'duration coefficient of variation', spec['cv']
        return 'duration scale of the finishing and assembly stations', spec['scale']
    if fam == 'lag':
        return 'lag scale', spec['scale']
    if fam == 'setup':
        if spec['kind'] == 'scale':
            return 'setup scale', spec['scale']
        if spec['kind'] == 'same_zero':
            return 'same-class setup scale', 0.0
        return 'stations charged with setups', SUBSET_STATIONS[level]
    if fam == 'win':
        return 'outage density scale', spec['scale']
    if fam == 'contention':
        return 'stations in the factory', 15
    if fam == 'mix':
        return 'project mix', MIX_WORDS[level]
    raise ValueError(level)


def level_mark(root, level):
    """Nominal mark word (inside, boundary, outside) and, when the level's manifest
    is on disk, the measured share outside the generator range."""
    word = nominal_mark(level).split(' (')[0]
    share = None
    mp = os.path.join(level_data_dir(root, level), 'manifest.json')
    if level != 'default' and os.path.exists(mp):
        with open(mp) as f:
            mk = json.load(f)
        share = mk.get('entries_outside_share')
        if share is None:
            share = mk.get('duration_outside_envelope_share')
    return word, share


def infer_guard(out):
    """An output root other than the repository root lies outside the repository's
    paper folders and its folders of results, data, checkpoints and logs."""
    o = os.path.realpath(out)
    rel = os.path.relpath(o, os.path.realpath(ROOT_DIR))
    if rel == os.pardir or rel.startswith(os.pardir + os.sep):
        return
    first = rel.split(os.sep)[0]
    if rel == os.curdir or first in INFER_GUARDED or first.startswith('paper'):
        infer_refuse(f'--out {out} would put the outputs in the repository root or in one '
                     f'of its paper, results, data, checkpoint or log folders; give a '
                     f'folder outside them')


def infer_read(root, mask, level, regime, label):
    """(result or None, path) of one method on one cell; a combination the level does
    not reach reads the default level. The record must name this mask, level,
    combination and method. trace: the per-instance counts of TRACE_FIELDS when the
    file carries them (evaluate --trace_counts), else None."""
    lv = level if (level == 'default' or regime in reached(level)) else 'default'
    p = result_path(root, mask, lv, regime, label)
    if not (os.path.exists(p) and os.path.exists(p[:-4] + '.json')):
        return None, p
    with np.load(p) as z:
        ms = z['makespan'].astype(float)
        c6_fail = int((~z['c6_ok']).sum())
        tr = ({k: np.asarray(z[k]) for k in TRACE_FIELDS}
              if all(k in z.files for k in TRACE_FIELDS) else None)
    with open(p[:-4] + '.json') as f:
        rec = json.load(f)
    want = {'mask': mask, 'level': lv, 'regime': regime, 'label': label}
    bad = {k: rec.get(k) for k, v in want.items() if rec.get(k) != v}
    if bad or len(rec.get('instances', [])) != len(ms):
        infer_refuse(f'{p}: record {bad or "instance list"} does not match the file '
                     f'({want}, {len(ms)} makespans)')
    found = rec.get('rule_set', 'classical')
    if label in RULES and found != RULE_SET[mask]:
        infer_refuse(f'{p}: a rule file of the rule set {found} '
                     f'({RULE_FUNCTION.get(found, "unknown")}); the {mask} mask uses the rule '
                     f'set {RULE_SET[mask]} ({RULE_FUNCTION[RULE_SET[mask]]}). Move the earlier '
                     f'rule files with archive-rules and evaluate the rule units again '
                     f'(plan --only_rules)')
    if tr is not None and any(len(v) != len(ms) for v in tr.values()):
        infer_refuse(f'{p}: the trace counts do not cover the {len(ms)} instances')
    return dict(ms=ms, instances=list(rec['instances']), c6_fail=c6_fail,
                seed=rec.get('seed'), placebo=lv != level, path=p, trace=tr), p


def infer_cell(root, mask, level, regime, seeds, missing):
    """Arrays of one cell: P, S, C [k, n] in seed order and R [4, n] (FIFO, MOR,
    SPT, MWKR); trace: the counts of the policy runs [k, n] and of the rules [4, n]
    when every file of the cell carries them. None when a file is missing (its path
    is added to missing)."""
    got, absent = {}, []
    for arm in INFER_ARMS:
        for s in seeds:
            x, p = infer_read(root, mask, level, regime,
                              'greedy+' + model_name(arm, regime, s))
            if x is None:
                absent.append(p)
            elif x['seed'] != s:
                infer_refuse(f'{p}: record seed {x["seed"]}, expected {s}')
            got[(arm, s)] = x
    for r in RULES:
        x, p = infer_read(root, mask, level, regime, r)
        if x is None:
            absent.append(p)
        got[r] = x
    if absent:
        missing.update(absent)
        return None
    xs = list(got.values())
    names = xs[0]['instances']
    off = [x['path'] for x in xs if x['instances'] != names]
    if off:
        infer_refuse(f'{level}/{regime}: the arms cover different instances ({off[:3]})')

    def stack(arm):
        return np.stack([got[(arm, s)]['ms'] for s in seeds])
    with_trace = sum(x['trace'] is not None for x in xs)
    trace = None
    if with_trace == len(xs):
        trace = {'P': {k: np.stack([got[('pact', s)]['trace'][k] for s in seeds])
                       for k in TRACE_FIELDS},
                 'R': {k: np.stack([got[r]['trace'][k] for r in RULES])
                       for k in TRACE_FIELDS}}
    return dict(names=names, n=len(names), P=stack('pact'), S=stack('spec'),
                C=stack('single'), R=np.stack([got[r]['ms'] for r in RULES]),
                c6_fail=sum(x['c6_fail'] for x in xs), placebo=xs[0]['placebo'],
                files=sorted({x['path'] for x in xs}), trace=trace,
                trace_files=with_trace, n_files=len(xs))


def infer_component(SF, d, B):
    """Interval package of one comparison d [k, n] or [k, n, c] (a drawn base
    instance brings its c combinations): crossed hierarchical bootstrap, fresh
    default_rng(HIER_SEED), and the seed-level t interval; with one training run the
    bootstrap over base instances, fresh default_rng(INST_SEED). Returns (package,
    draws)."""
    x = np.asarray(d, dtype=float)
    k, n = x.shape[0], x.shape[1]
    c = x.shape[2] if x.ndim == 3 else 1
    if k >= 2:
        hb = SF.hier_boot(x, B=B, seed=HIER_SEED, alpha=ALPHA, return_draws=True)
        t = SF.seed_t(x, ALPHA)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                   seed_means=t['seed_means'], scheme='hierarchical',
                   stream=f'default_rng({HIER_SEED})')
        draws = hb['draws']
    else:
        ib = SF.instance_boot(x[0], B=B, seed=INST_SEED, alpha=ALPHA, return_draws=True)
        out = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], t_lo=float('nan'),
                   t_hi=float('nan'), seed_means=[float(x[0].mean())], scheme='instance',
                   stream=f'default_rng({INST_SEED})')
        draws = ib['draws']
    out.update(k=int(k), n=int(n), c=int(c), direction=SF.direction(out['lo'], out['hi']),
               **{f'v{mg}': SF.verdict(out['lo'], out['hi'], mg) for mg in MARGINS})
    return out, draws


def infer_superiority(st, D, B):
    """Per combination of D (the combinations the level reaches) the interval, the
    one-sided p-values for H1: d < 0 and H1: d > 0 (seed-level t, undefined -> 1, and
    bootstrap from the draws of the interval), Holm over the combinations of D for
    each kind, and the combined verdict."""
    SF, sv = st[0], st[4]
    rows = {}
    for r, d in D.items():
        c, draws = infer_component(SF, d, B)
        pl, pg = SF.seed_t_p(d, 'less'), SF.seed_t_p(d, 'greater')
        c.update(p_t=1.0 if np.isnan(pl) else pl, p_boot=SF.boot_p_one_sided(draws, 'less'),
                 p_t_greater=1.0 if np.isnan(pg) else pg,
                 p_boot_greater=SF.boot_p_one_sided(draws, 'greater'))
        rows[r] = c
    keys = list(rows)
    for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
        for r, a in zip(keys, SF.holm([rows[r][col] for r in keys])):
            rows[r][col + '_holm'] = a
    for r in keys:
        x = rows[r]
        x['holm_family'] = keys
        x['verdict'] = sv(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                          x['p_t_greater_holm'], x['p_boot_greater_holm'], ALPHA)
    return rows


def infer_gaps(cell):
    """d [k, n] of the three comparisons, first-named method first, seed s against
    seed s; the best dispatching rule is the per-instance minimum of the four rules."""
    best = cell['R'].min(axis=0)
    return {'pol_spec': 100.0 * (cell['P'] - cell['S']) / cell['S'],
            'ctl_spec': 100.0 * (cell['C'] - cell['S']) / cell['S'],
            'pol_rule': 100.0 * (cell['P'] - best[None, :]) / best[None, :]}


def _one_pair(dec, one):
    return 100.0 * float(np.sum(one)) / float(np.sum(dec))


def _setup_share(setup, used, ms):
    return 100.0 * float(np.mean(np.asarray(setup, float) / (np.asarray(used, float) * ms)))


def infer_measured(cell):
    """The two measured variables of one combination, in percent, with the definitions
    of scripts/p93_rule_comparison.py trace_stats: the share of single-pair decisions,
    over all decisions of the instances, and the setup share of machine capacity, per
    instance the total setup time over the machines used times the makespan, averaged
    over the instances. Policy: per training run, then the mean (and the range) over
    the runs. Best dispatching rule: per instance the schedule of the rule with the
    shortest makespan (a tie goes to the first of FIFO, MOR, SPT, MWKR)."""
    T, P, R = cell['trace'], cell['P'], cell['R']
    TP, TR = T['P'], T['R']
    op = [_one_pair(TP['n_decisions'][k], TP['n_one_pair'][k]) for k in range(len(P))]
    ss = [_setup_share(TP['setup_sum'][k], TP['machines_used'][k], P[k])
          for k in range(len(P))]
    b = np.argmin(R, axis=0)
    i = np.arange(R.shape[1])
    return dict(pol_one_pair=float(np.mean(op)), pol_one_pair_range=[min(op), max(op)],
                pol_setup_share=float(np.mean(ss)), pol_setup_share_range=[min(ss), max(ss)],
                rule_one_pair=_one_pair(TR['n_decisions'][b, i], TR['n_one_pair'][b, i]),
                rule_setup_share=_setup_share(TR['setup_sum'][b, i],
                                              TR['machines_used'][b, i], R[b, i]),
                best_rule_instances={r: int((b == j).sum()) for j, r in enumerate(RULES)})


def infer_level(st, root, level, cells, B, measured):
    """Every comparison of one level: per combination present (the verdicts against
    the best dispatching rule, with Holm, over the combinations the level reaches),
    pooled over the eight and over the reached combinations when all of them are
    present (interval and direction only against the rule)."""
    SF = st[0]
    combos = [r for r in R8 if cells.get(r) is not None]
    rset = [r for r in combos if r in reached(level)]
    gaps = {r: infer_gaps(cells[r]) for r in combos}
    out = {'combos': combos, 'reached': list(reached(level)),
           'not_reached': [r for r in combos if cells[r]['placebo']],
           'n': {r: cells[r]['n'] for r in combos}, 'rows': {}, 'pools': {},
           'holm_family': rset}
    for key in ('pol_spec', 'ctl_spec'):
        out['rows'][key] = {r: infer_component(SF, gaps[r][key], B)[0] for r in combos}
    sup = infer_superiority(st, {r: gaps[r]['pol_rule'] for r in rset}, B)
    out['rows']['pol_rule'] = {r: sup[r] if r in sup else
                               infer_component(SF, gaps[r]['pol_rule'], B)[0]
                               for r in combos}
    pools = {POOL_ALL: list(R8)}
    if tuple(reached(level)) != R8:
        pools[POOL_REACHED] = list(reached(level))
    for pname, regs in pools.items():
        if not all(r in combos for r in regs):
            continue
        if any(cells[r]['names'] != cells[regs[0]]['names'] for r in regs):
            infer_refuse(f'{level}: the combinations {regs} do not share their instances, '
                         f'so the base instances of a pool are not defined')
        out['pools'][pname] = {'combos': regs}
        for key in COMPARISONS:
            d3 = np.stack([gaps[r][key] for r in regs], axis=2)
            out['pools'][pname][key] = infer_component(SF, d3, B)[0]
    counts = {}
    for key in ('pol_spec', 'ctl_spec'):
        for mg in MARGINS:
            vs = [out['rows'][key][r][f'v{mg}'] for r in rset]
            counts[f'{key}|{mg}'] = {v: vs.count(v) for v in SF.VERDICTS}
    vs = [out['rows']['pol_rule'][r]['verdict'] for r in rset]
    counts['pol_rule'] = {v: vs.count(v) for v in st[1]}
    out['counts'] = counts
    out['counted_over'] = rset
    out['parameter'], out['value'] = level_parameter(level)
    out['mark'], out['outside_share'] = level_mark(root, level)
    out['family'] = LEVELS[level]['family']
    out['c6_fail'] = sum(cells[r]['c6_fail'] for r in combos)
    if measured:
        out['measured'] = {r: infer_measured(cells[r]) for r in combos}
    return out


def _verdict_cols(key):
    return [f'v{mg}' for mg in MARGINS] if key != 'pol_rule' else ['verdict']


def infer_changes(res):
    """Per perturbed level on the default level's instances and per comparison, the
    reached combinations whose verdict differs from the default level's verdict on
    the same combination; the project-mix levels are not compared (MIX_NOTE)."""
    base = res['levels'].get('default')
    if base is None:
        return
    for level, L in res['levels'].items():
        if level == 'default':
            continue
        if not paired_with_default(level):
            L['changed'], L['changed_note'] = None, MIX_NOTE
            continue
        ch = {}
        for key in COMPARISONS:
            for col in _verdict_cols(key):
                ch[f'{key}|{col}'] = [
                    r for r in L['counted_over'] if r in base['counted_over']
                    and L['rows'][key][r][col] != base['rows'][key][r][col]]
        L['changed'] = ch


def infer_outputs(out_root):
    return {k: os.path.join(out_root, v) for k, v in INFER_FILES.items()}


def infer_check_names(names, out_paths):
    """Every name is Sens plus letters, unique, and absent from every other macro
    file of the repository (its own path and the output path are skipped)."""
    bad = [x for x in names if not re.fullmatch(r'Sens[A-Za-z]+', x)]
    if bad:
        infer_refuse(f'macro names that are not Sens plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        infer_refuse(f'duplicate macro names: {dup[:5]}')
    own = {os.path.abspath(out_paths['macros']),
           os.path.abspath(os.path.join(ROOT_DIR, INFER_FILES['macros']))}
    files = sorted(glob.glob(os.path.join(ROOT_DIR, 'manuscript', 'macros_*.tex')) +
                   glob.glob(os.path.join(ROOT_DIR, 'paper', 'macros_*.tex')) +
                   [os.path.join(ROOT_DIR, d, 'macros.tex') for d in ('manuscript', 'paper')])
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            infer_refuse(f'macro name collision with {os.path.relpath(f, ROOT_DIR)}: '
                         f'{sorted(clash)[:5]}')
    return len(files)


def _scheme_text(c):
    if c['scheme'] == 'hierarchical':
        return (f'hierarchical 95% CI over {c["k"]} seed pairs and {c["n"]} base instances'
                + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))
    return (f'one training run: 95% CI over {c["n"]} base instances'
            + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))


def _comp_macros(st, name, c, key, with_verdict):
    """Macros of one comparison: mean, interval and, against a specialist, the
    verdicts at both margins, then the direction, and against the rule the combined
    verdict when with_verdict."""
    SF, f2 = st[0], st[2]
    s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
         f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
         f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}')
    if key != 'pol_rule':
        s += ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{SF.TEX[c[f"v{mg}"]]}}}'
                     for mg in MARGINS)
    s += f'\\newcommand{{\\{name}Dir}}{{{SF.TEX[c["direction"]]}}}'
    if key == 'pol_rule' and with_verdict:
        s += f'\\newcommand{{\\{name}Verdict}}{{{c["verdict"]}}}'
    return s


def _comp_comment(c, key, with_verdict):
    if key != 'pol_rule':
        tail = ', verdicts at 0.5 and 1'
    elif with_verdict:
        tail = (f', combined verdict with Holm over {",".join(c["holm_family"])}; Holm p '
                f'shorter {c["p_t_holm"]:.2g} (t), {c["p_boot_holm"]:.2g} (bootstrap), Holm '
                f'p longer {c["p_t_greater_holm"]:.2g} (t), {c["p_boot_greater_holm"]:.2g} '
                f'(bootstrap)')
    else:
        tail = ''
    return f'mean, {_scheme_text(c)}{tail}, direction of [{c["lo"]:+.4f}, {c["hi"]:+.4f}]'


def infer_macro_lines(st, res):
    """Macro lines: the pooled rows of every level (over the eight combinations and
    over the reached ones), its verdict counts and verdict changes, the study-level
    counts, the per-combination rows (all eight for the default level, the reached
    combinations for a changed level), and the measured variables of
    MEASURED_MACRO_LEVELS when they are computed."""
    VW, u1 = st[1], st[5]
    mark = 'PARTIAL ' if res['partial'] else ''
    sd = res['seeds']
    L = [f'% Generated by {INFER_GENERATOR} -- do not hand-edit',
         f'% Sensitivity of the factory benchmark (10 modules, 25 stations; '
         f'{len(LEVELS) - 1} levels, one parameter at a time), mask {res["mask"]}, training '
         f'seeds {",".join(map(str, sd))}; B = {res["B"]}; hierarchical intervals fresh '
         f'default_rng({HIER_SEED}) per quantity.',
         '% Gaps in %: first-named method minus second over second (negative = first-named '
         'method shorter), seed s against seed s; best dispatching rule = per-instance '
         'minimum of FIFO, MOR, SPT and MWKR'
         + (', each acting on the pairs the mask admits.' if res.get('rule_set') == 'masked'
            else '.'),
         '% Each level is pooled over the eight combinations and over the combinations it '
         'reaches; a combination a level does not reach takes the default level\'s results. '
         'Counts and Holm\'s correction are over the combinations the level reaches; pooled '
         'rows against the best dispatching rule give the interval and the direction.']
    if res['partial']:
        L.insert(1, '% PARTIAL: complete combinations only; not for the manuscript.')
    L.append(f'\\newcommand{{\\SensBoot}}{{{res["B"]}}}'
             f'\\newcommand{{\\SensSeedN}}{{{len(sd)}}}'
             f'\\newcommand{{\\SensLevelN}}{{{len(LEVELS) - 1}}}'
             f'  % {mark}bootstrap replicates; training seeds per arm; perturbed levels')
    for level, x in res['levels'].items():
        w = LEVEL_WORDS[level]
        ns = sorted(set(x['n'].values()))
        nn = ns[0] if len(ns) == 1 else '/'.join(map(str, ns))
        L.append(f'\\newcommand{{\\Sens{w}N}}{{{nn}}}'
                 f'\\newcommand{{\\Sens{w}ReachedN}}{{{len(x["reached"])}}}'
                 f'  % {mark}{level}: instances per combination; combinations the level '
                 f'reaches ({",".join(x["reached"])})')
        # Per-combination rows: all eight for the default level, the reached ones for a
        # changed level (an unreached combination holds the default level's results).
        for key in COMPARISONS:
            first, second, cw = COMPARISONS[key]
            for r in (x['combos'] if level == 'default' else x['reached']):
                if r not in x['rows'][key]:
                    continue
                c = x['rows'][key][r]
                L.append(_comp_macros(st, f'Sens{w}{cw}{r}', c, key, True)
                         + f'  % {mark}{level}, {first} vs {second}, {r}: '
                         + _comp_comment(c, key, True))
        for pname, P in x['pools'].items():
            for key in COMPARISONS:
                first, second, cw = COMPARISONS[key]
                c = P[key]
                L.append(_comp_macros(st, f'Sens{w}{cw}{ROW_WORDS[pname]}', c, key, False)
                         + f'  % {mark}{level}, {first} vs {second}, pooled over '
                         f'{",".join(P["combos"])}: ' + _comp_comment(c, key, False))
        cnt = x['counts']
        for key in ('pol_spec', 'ctl_spec'):
            first, second, cw = COMPARISONS[key]
            s = ''
            for mg in MARGINS:
                cc = cnt[f'{key}|{mg}']
                for v, vw in (('equivalent', 'Eq'), ('non-inferior only', 'NonInf'),
                              ('inferior', 'Inf'), ('inconclusive', 'Inc')):
                    s += f'\\newcommand{{\\Sens{w}{cw}{vw}{MW[mg]}Count}}{{{cc[v]}}}'
            L.append(s + f'  % {mark}{level}, {first} vs {second}: combinations (of '
                     f'{len(x["counted_over"])} reached) equivalent, non-inferior only, '
                     f'inferior, inconclusive at 0.5, then at 1')
        cc = cnt['pol_rule']
        L.append(f'\\newcommand{{\\Sens{w}PolRuleAheadCount}}{{{cc[VW[0]]}}}'
                 f'\\newcommand{{\\Sens{w}PolRuleBehindCount}}{{{cc[VW[1]]}}}'
                 f'  % {mark}{level}, policy vs best dispatching rule: combinations (of '
                 f'{len(x["counted_over"])} reached) with the combined verdict ahead, behind')
        if x.get('changed'):
            s, what = '', []
            for key in COMPARISONS:
                cw = COMPARISONS[key][2]
                for col in _verdict_cols(key):
                    mw = MW[float(col[1:])] if col != 'verdict' else ''
                    lst = x['changed'][f'{key}|{col}']
                    s += f'\\newcommand{{\\Sens{w}{cw}Changed{mw}Count}}{{{len(lst)}}}'
                    what.append(f'{cw}{mw}: {",".join(lst) or "none"}')
            L.append(s + f'  % {mark}{level}: reached combinations whose verdict differs '
                     f'from the default level ({"; ".join(what)})')
        elif x.get('changed_note'):
            L.append(f'% {mark}{level}: verdict changes from the default level '
                     f'{x["changed_note"]}')
        if level in MEASURED_MACRO_LEVELS and 'measured' in x:
            for r in x['counted_over']:
                m = x['measured'][r]
                L.append(''.join(f'\\newcommand{{\\Sens{w}{MEASURED_WORDS[k]}{r}}}'
                                 f'{{{u1(m[k])}}}' for k in MEASURED)
                         + f'  % {mark}{level}, {r}: share of decisions with exactly one '
                         f'admitted pair, setup share of machine capacity, in %; policy '
                         f'(mean over the runs), then the best dispatching rule\'s schedule')
    if 'changed_levels' in res:
        s, what = '', []
        for key in COMPARISONS:
            cw = COMPARISONS[key][2]
            for col in _verdict_cols(key):
                mw = MW[float(col[1:])] if col != 'verdict' else ''
                lst = res['changed_levels'][f'{key}|{col}']
                s += f'\\newcommand{{\\Sens{cw}Changed{mw}Levels}}{{{len(lst)}}}'
                what.append(f'{cw}{mw}: {",".join(lst) or "none"}')
        L.append(f'\\newcommand{{\\SensChangedOverN}}{{{len(res["changed_over"])}}}' + s
                 + f'  % {mark}levels compared with the default level (the levels on its '
                 f'instances), then the levels with a verdict that differs from the default '
                 f'level on a reached combination ({"; ".join(what)})')
    if not res['measured_variables']['computed']:
        L.append(f'% {mark}The two measured variables are not computed: '
                 f'{res["measured_variables"]["reason"]}')
    return L


def _plain_value(level, value):
    if level == 'default':
        return '--'
    if level in SUBSET_STATIONS:
        return f'{value} of 25'
    if isinstance(value, float):
        return f'{value:.2f}'
    return str(value)


def infer_table(st, res):
    """Compact tabular for the supplement, one row per level: the policy against the
    specialists and against the best dispatching rule, pooled over the combinations
    the level reaches, with the counts over those combinations. No bold: no row-wise
    best applies."""
    SF, f2 = st[0], st[2]
    VW = st[1]

    def gap(c):
        return f'${f2(c["mean"])}$ $[{f2(c["lo"])},\\,{f2(c["hi"])}]$'
    out = [f'% auto-generated by {INFER_GENERATOR} -- do not hand-edit',
           f'% Mask {res["mask"]}, training seeds {",".join(map(str, res["seeds"]))}. One '
           'row per level. Gap in %: policy minus the specialists, or minus the best '
           'dispatching rule, over them, pooled over the combinations the level reaches, '
           '95% hierarchical interval over the training runs and the base instances; '
           'counts over the combinations the level reaches.']
    if res['partial']:
        out.insert(1, '% PARTIAL: complete combinations only; not for the manuscript.')
    out += [r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.2cm} l ll r ll r@{}}',
            r'\toprule',
            r' & & \multicolumn{3}{c}{Policy against the specialists} & '
            r'\multicolumn{3}{c}{Policy against the best dispatching rule} \\',
            r'\cmidrule(lr){3-5}\cmidrule(l){6-8}',
            r'Parameter & Value & Gap (\%) [95\% interval] & Verdict at 1\% & '
            r'Equivalent at 1\% & Gap (\%) [95\% interval] & Direction & Ahead \\',
            r'\midrule']
    for level, x in res['levels'].items():
        par = 'default benchmark' if level == 'default' else x['parameter']
        val = _plain_value(level, x['value'])
        P = x['pools'].get(POOL_REACHED) or x['pools'].get(POOL_ALL)
        m = len(x['counted_over'])
        eq = x['counts']['pol_spec|1.0']['equivalent']
        ah = x['counts']['pol_rule'][VW[0]]
        if P is None:
            spec, vtxt, rule, dtxt = '--', '--', '--', '--'
        else:
            a, b = P['pol_spec'], P['pol_rule']
            spec, vtxt, rule = gap(a), SF.TEX[a['v1.0']], gap(b)
            dtxt = SF.TEX[b['direction']]
        out.append(f'{par} & {val} & {spec} & {vtxt} & {eq} of {m} & {rule} & {dtxt} & '
                   f'{ah} of {m}' + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', '']
    return '\n'.join(out)


def _cell(c, key, f2):
    s = f'{f2(c["mean"])} [{f2(c["lo"])}, {f2(c["hi"])}]'
    if key == 'pol_rule':
        return f'{s} {c["verdict"]}' if 'verdict' in c else f'{s} direction {c["direction"]}'
    return f'{s} {c["v0.5"]} / {c["v1.0"]}'


def infer_markdown(st, res):
    f2, u1 = st[2], st[5]
    sd = res['seeds']
    L = ['# Sensitivity of the factory benchmark: the protocol of the manuscript '
         f'({INFER_GENERATOR})', '']
    if res['partial']:
        L += [f'PARTIAL: complete combinations only; {len(res["missing"])} result files '
              'missing. Not for the manuscript.', '']
    L += [f'Mask {res["mask"]}, training seeds {",".join(map(str, sd))}, root '
          f'{res["root"]}. Gaps in %: first-named method minus second over second, seed s '
          f'against seed s; negative = first-named method shorter. Intervals: crossed '
          f'hierarchical bootstrap over training runs and base instances, B = {res["B"]}, '
          f'fresh default_rng({HIER_SEED}) per quantity (one training run: bootstrap over '
          f'base instances, default_rng({INST_SEED})). Verdicts against the specialist at '
          'the margins 0.5% / 1%; against the best dispatching rule (per-instance minimum '
          f'of FIFO, MOR, SPT, MWKR; rule set {res.get("rule_set")}, '
          f'{res.get("rule_function")}) the combined verdict with Holm over the combinations '
          'the level reaches; pooled rows and combinations a level does not reach give the '
          'interval and the direction. A combination a level does not reach takes the '
          'default level\'s results (marked "not reached").', '']
    mv = res['measured_variables']
    L += [('Measured variables: ' + mv['definition']) if mv['computed'] else
          ('The two measured variables (share of decisions with one admitted pair, setup '
           f'share of machine capacity) are not computed: {mv["reason"]}'), '',
          '| level | parameter | mark | combination | n | policy - specialist | '
          'control - specialist | policy - best rule |', '|' + '---|' * 8]
    for level, x in res['levels'].items():
        par = '' if x['parameter'] == 'none' else f'{x["parameter"]} = {x["value"]}'
        mk = x['mark'] + ('' if x['outside_share'] is None
                          else f' ({x["outside_share"]:.2f} outside)')
        rows = [(r + (' (not reached)' if r in x['not_reached'] else ''), x['n'][r],
                 {k: x['rows'][k][r] for k in COMPARISONS}) for r in x['combos']]
        rows += [(f'pooled {",".join(P["combos"])}', P['pol_spec']['n'],
                  {k: P[k] for k in COMPARISONS}) for P in x['pools'].values()]
        for row, n, cs in rows:
            L.append(f'| {level} | {par} | {mk} | {row} | {n} | ' +
                     ' | '.join(_cell(cs[k], k, f2) for k in COMPARISONS) + ' |')
    L += ['', '## Verdict counts over the reached combinations', '',
          '| level | reached | policy - specialist at 0.5 (eq/ni/inf/inc) | at 1 | '
          'control - specialist at 0.5 | at 1 | policy - best rule (ahead/behind/nd) |',
          '|' + '---|' * 7]
    for level, x in res['levels'].items():
        cnt = x['counts']

        def q(key, mg):
            c = cnt[f'{key}|{mg}']
            return '/'.join(str(c[v]) for v in st[0].VERDICTS)
        cr = cnt['pol_rule']
        L.append(f'| {level} | {len(x["counted_over"])} | {q("pol_spec", 0.5)} | '
                 f'{q("pol_spec", 1.0)} | {q("ctl_spec", 0.5)} | {q("ctl_spec", 1.0)} | '
                 + '/'.join(str(cr[v]) for v in st[1]) + ' |')
    L += ['', '## Verdicts that differ from the default level (reached combinations)', '']
    for level, x in res['levels'].items():
        if x.get('changed_note'):
            L.append(f'- {level}: {x["changed_note"]}')
            continue
        ch = {k: v for k, v in (x.get('changed') or {}).items() if v}
        if ch:
            L.append(f'- {level}: ' + '; '.join(f'{k}: {",".join(v)}' for k, v in ch.items()))
    if mv['computed']:
        L += ['', '## Measured variables (%, reached combinations)', '',
              '| level | combination | policy: single-pair decisions [range over runs] | '
              'policy: setup share [range] | best rule: single-pair decisions | best rule: '
              'setup share |', '|' + '---|' * 6]
        for level, x in res['levels'].items():
            for r in x['counted_over']:
                m = x['measured'][r]
                L.append(f'| {level} | {r} | {u1(m["pol_one_pair"])} '
                         f'[{u1(m["pol_one_pair_range"][0])}, {u1(m["pol_one_pair_range"][1])}]'
                         f' | {u1(m["pol_setup_share"])} [{u1(m["pol_setup_share_range"][0])}, '
                         f'{u1(m["pol_setup_share_range"][1])}] | {u1(m["rule_one_pair"])} | '
                         f'{u1(m["rule_setup_share"])} |')
    if res['partial']:
        L += ['', '## Missing result files', ''] + [f'- {p}' for p in res['missing'][:200]]
        if len(res['missing']) > 200:
            L.append(f'- ... ({len(res["missing"]) - 200} more)')
    return L


def infer_csv_rows(res):
    rows = []
    for level, x in res['levels'].items():
        base = dict(level=level, family=x['family'], parameter=x['parameter'],
                    value=x['value'], mark=x['mark'],
                    outside_share='' if x['outside_share'] is None else x['outside_share'])
        items = [(r, x['rows'][k][r], k, int(r in x['reached'])) for k in COMPARISONS
                 for r in x['combos']]
        items += [(p, P[k], k, '') for p, P in x['pools'].items() for k in COMPARISONS]
        for row, c, key, rch in items:
            first, second, _ = COMPARISONS[key]
            rec = dict(base, comparison=key, first=first, second=second, row=row,
                       reached=rch, k=c['k'], n=c['n'], c=c['c'])
            for f in ('mean', 'lo', 'hi', 't_lo', 't_hi'):
                rec[f] = '' if not np.isfinite(c[f]) else f'{c[f]:.6f}'
            rec['direction'] = c['direction']
            if key == 'pol_rule':
                rec.update({'verdict_0.5': '', 'verdict_1': '',
                            'verdict': c.get('verdict', '')})
            else:
                rec.update({'verdict_0.5': c['v0.5'], 'verdict_1': c['v1.0'], 'verdict': ''})
            for f in ('p_t_holm', 'p_boot_holm', 'p_t_greater_holm', 'p_boot_greater_holm'):
                rec[f] = f'{c[f]:.6g}' if f in c else ''
            m = x.get('measured', {}).get(row)
            for f in MEASURED:
                rec[f] = '' if m is None else f'{m[f]:.4f}'
            rows.append(rec)
    return rows


def infer_check(root, mask, levels, seeds):
    """--check: every result file of every level, arm, seed and rule is present, its
    record matches it, the arms of a cell cover the level's instances, and no
    schedule failed check C6. Reads only. Returns dict(complete, missing, c6_fail,
    other_instances, trace_files, n_files)."""
    missing, c6, short = set(), [], []
    tr = nf = 0
    for level in levels:
        want = level_names(level)
        for regime in R8:
            c = infer_cell(root, mask, level, regime, seeds, missing)
            if c is None:
                continue
            tr, nf = tr + c['trace_files'], nf + c['n_files']
            if c['names'] != want:
                short.append(f'{level}/{regime}')
            if c['c6_fail']:
                c6.append(f'{level}/{regime}')
    ok = not (missing or c6 or short)
    print(f'[p87 infer --check] {"complete" if ok else "NOT complete"}: mask {mask}, seeds '
          f'{",".join(map(str, seeds))}, {len(levels)} levels; {len(missing)} result files '
          f'missing, {len(short)} combinations on other instances than the level\'s, '
          f'{len(c6)} combinations with a schedule that failed check C6; trace counts in '
          f'{tr} of {nf} files read')
    for p in sorted(missing)[:12]:
        print(f'  missing <root>/{os.path.relpath(p, root)}')
    for x in (short + c6)[:12]:
        print(f'  combination {x}')
    return dict(complete=ok, missing=sorted(missing), c6_fail=c6, other_instances=short,
                trace_files=tr, n_files=nf)


def _rel_root(root):
    """The results root as written into the outputs: relative to the repository, or
    a neutral word for a folder outside it (no machine path)."""
    rel = os.path.relpath(root, ROOT_DIR)
    return '<a folder outside the repository>' if rel.startswith(os.pardir) else rel


MEASURED_DEFINITION = (
    'per level and combination, the share of decisions at which the mask admits '
    'exactly one (job, machine) pair, over all decisions of the instances, and the '
    'setup share of machine capacity, per instance the total setup time over the '
    'machines used times the makespan, averaged over the instances (the definitions of '
    'scripts/p93_rule_comparison.py trace_stats); the policy as the mean over its '
    'training runs, the best dispatching rule as the schedule of the per-instance best '
    'rule (a tie goes to the first of FIFO, MOR, SPT, MWKR); from the counts of '
    'evaluate --trace_counts')


def infer(args):
    """Phase infer (module doc). Returns the result dictionary."""
    import csv
    st = _stats()
    jsonable = st[3]
    partial = bool(getattr(args, 'partial', False))
    check = bool(getattr(args, 'check', False))
    nboot = int(getattr(args, 'boot', None) or B)
    out_arg = getattr(args, 'out', None)
    root = os.path.abspath(args.root)
    seeds = list(args.seeds)
    if check:
        if partial or out_arg is not None:
            infer_refuse('--check reads only; it takes neither --partial nor --out')
        out_root = None
    elif out_arg is None:
        if partial or args.mask != 'exact':
            infer_refuse('--partial and --mask legacy write only to --out <folder> outside '
                         "the repository's paper and results folders")
        out_root = ROOT_DIR
    else:
        out_root = os.path.abspath(out_arg)
        if os.path.realpath(out_root) != os.path.realpath(ROOT_DIR) \
                or partial or args.mask != 'exact':
            infer_guard(out_root)
    levels = list(args.levels) if args.levels else list(LEVELS)
    if not partial:
        if levels != list(LEVELS):
            infer_refuse('every level is required; --levels needs --partial')
        if len(seeds) < 2:
            infer_refuse('at least two training seeds are required (the unit of the '
                         'hierarchical bootstrap); one seed needs --partial')
    if len(set(seeds)) != len(seeds):
        infer_refuse(f'repeated seeds {seeds}')
    if 'default' not in levels:
        levels = ['default'] + levels       # the reference of every verdict change
    paths = infer_outputs(out_root) if out_root else {}
    for p in paths.values():                # a refusal below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if check:
        return infer_check(root, args.mask, levels, seeds)
    missing = set()
    cells = {}
    for level in levels:
        want = None if partial else level_names(level)
        for regime in R8:
            c = infer_cell(root, args.mask, level, regime, seeds, missing)
            if c is not None and want is not None and c['names'] != want:
                infer_refuse(f'{level}/{regime}: the results cover {c["n"]} instances '
                             f'{c["names"][:2]}..., the level has {len(want)}')
            cells[(level, regime)] = c
    if missing and not partial:
        ms = sorted(missing)
        infer_refuse(f'{len(ms)} result files missing (levels, arms or seeds), e.g.\n  '
                     + '\n  '.join(ms[:12]))
    bad = sorted({(lv, r) for (lv, r), c in cells.items() if c and c['c6_fail']})
    if bad and not partial:
        infer_refuse(f'schedules that failed check C6 in {bad[:6]}')
    have = [c for c in cells.values() if c is not None]
    n_tr, n_all = sum(c['trace_files'] for c in have), sum(c['n_files'] for c in have)
    measured = bool(have) and n_tr == n_all
    mv = ({'computed': True, 'definition': MEASURED_DEFINITION, 'files': n_all}
          if measured else
          {'computed': False,
           'reason': (f'{n_all - n_tr} of the {n_all} result files read carry no trace '
                      'counts (the evaluate phase was run without --trace_counts)')})
    res = {'generator': INFER_GENERATOR, 'partial': partial, 'mask': args.mask,
           'arm_names': ARM_NAMES['scheme'], 'seeds': seeds, 'B': nboot,
           'root': _rel_root(root),
           'convention': ('d = 100 (first - second) / second, seed s against seed s, '
                          'per base instance; negative = first-named method shorter'),
           'best_rule': 'per-instance minimum of ' + ', '.join(RULES),
           'rule_set': RULE_SET[args.mask],
           'rule_function': RULE_FUNCTION[RULE_SET[args.mask]],
           'estimators': {'hierarchical': f'stats_family.hier_boot, fresh '
                                          f'default_rng({HIER_SEED}) per quantity',
                          'one run': f'stats_family.instance_boot, default_rng({INST_SEED})',
                          'verdicts': 'stats_family.verdict at 0.5 and 1; combined verdict '
                                      'p89_public_family.superiority_verdict, Holm over the '
                                      'combinations a level reaches; pooled rows against '
                                      'the rule: interval and direction only'},
           'measured_variables': mv,
           'missing': sorted(os.path.relpath(p, root) for p in missing),
           'c6_fail_combinations': [list(x) for x in bad],
           'levels': {}}
    for level in levels:
        lc = {r: cells[(level, r)] for r in R8}
        present = [r for r in R8 if lc[r] is not None]
        if not present or (level != 'default' and not set(present) & set(reached(level))):
            continue                        # partial: nothing of this level is on disk
        res['levels'][level] = infer_level(st, root, level, lc, nboot, measured)
        res['levels'][level]['files'] = sorted({f for v in lc.values() if v
                                                for f in v['files']})
    infer_changes(res)
    if 'default' in res['levels'] and not partial:
        cl = {}
        res['changed_over'] = [lv for lv, x in res['levels'].items()
                               if lv != 'default' and x.get('changed') is not None]
        for key in COMPARISONS:
            for col in _verdict_cols(key):
                cl[f'{key}|{col}'] = [lv for lv in res['changed_over']
                                      if res['levels'][lv]['changed'][f'{key}|{col}']]
        res['changed_levels'] = cl
        res['changed_not_compared'] = {lv: MIX_NOTE for lv, x in res['levels'].items()
                                       if x.get('changed_note')}
    lines = infer_macro_lines(st, res)
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    res['macro_files_checked'] = infer_check_names(names, paths)
    from macro_io import emit
    emit(paths['macros'], lines)
    rel = {lv: [os.path.relpath(f, root) for f in x.pop('files')]
           for lv, x in res['levels'].items()}
    res['inputs'] = {lv: len(v) for lv, v in rel.items()}
    os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
    write_json_atomic(paths['json'], jsonable(res), indent=1)
    with open(paths['md'], 'w') as f:
        f.write('\n'.join(infer_markdown(st, res)) + '\n')
    with open(paths['csv'], 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(infer_csv_rows(res))
    os.makedirs(os.path.dirname(paths['table']), exist_ok=True)
    with open(paths['table'], 'w') as f:
        f.write(infer_table(st, res))
    for k in ('json', 'md', 'csv', 'table'):     # emit() has reported the macro file
        print('wrote', paths[k])
    print(f'{len(names)} macros, {len(res["levels"])} levels'
          + ('' if measured else '; measured variables not computed: ' + mv['reason'])
          + (f'; PARTIAL, {len(missing)} result files missing' if partial else ''))
    return res



# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['build', 'plan', 'evaluate', 'analyze',
                                      'compare', 'timing', 'infer', 'archive-rules'])
    ap.add_argument('--timing', default='',
                    help='plan: timing.json to cost the units with (default '
                         '<root>/timing.json)')
    ap.add_argument('--root', default=DEFAULT_ROOT)
    ap.add_argument('--levels', default='')
    ap.add_argument('--regimes', default='')
    ap.add_argument('--arms', default=DEFAULT_ARMS)
    ap.add_argument('--seeds', default='301,302,303')
    ap.add_argument('--n', type=int, default=0, help='first n instances (0 = all 100)')
    ap.add_argument('--mask', default='legacy', choices=['legacy', 'exact'],
                    help='first_mask and mask_ties of every rollout')
    ap.add_argument('--arm_names', default='legacy', choices=['legacy', 'xmask'],
                    help='checkpoint naming of the policy arms (see ARM_NAMES)')
    ap.add_argument('--part', type=int, default=None)
    ap.add_argument('--parts', type=int, default=1)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--partial', action='store_true',
                    help='infer: the complete combinations only, written under --out')
    ap.add_argument('--out', default=None,
                    help='infer: root of the outputs (default: the repository, --mask '
                         'exact and every level only)')
    ap.add_argument('--boot', type=int, default=B, help='infer: bootstrap replicates')
    ap.add_argument('--trace_counts', action='store_true',
                    help='evaluate: also store per instance the decisions, the '
                         'single-pair decisions, the summed setup and the machines used '
                         '(TRACE_FIELDS); plan: put the option into the printed commands')
    ap.add_argument('--check', action='store_true',
                    help='infer: only check that every input is present (writes nothing; '
                         'exit status 1 when not)')
    ap.add_argument('--only_rules', action='store_true',
                    help='plan: the rule units alone, in plan_rules.json; evaluate: the '
                         'rule units (with --part: of plan_rules.json)')
    ap.add_argument('--dry', action='store_true',
                    help='plan and archive-rules: print what would be written or moved, '
                         'write and move nothing')
    args = ap.parse_args(_ARGV)
    args.levels = [x for x in args.levels.split(',') if x]
    bad = [x for x in args.levels if x not in LEVELS]
    if bad:
        sys.exit(f'unknown level(s) {bad}; known: {list(LEVELS)}')
    args.regimes = [x for x in args.regimes.split(',') if x]
    args.arms = [x for x in args.arms.split(',') if x]
    args.seeds = [int(x) for x in args.seeds.split(',') if x]
    args.n = args.n or None
    ARM_NAMES['scheme'] = args.arm_names
    # archive-rules moves rule files only, so the checkpoint naming does not apply
    if (args.arm_names == 'xmask') != (args.mask == 'exact') \
            and args.phase != 'archive-rules':
        sys.exit('REFUSING: --arm_names xmask goes with --mask exact, legacy with '
                 'legacy')
    os.chdir(ROOT_DIR)
    if not os.path.isabs(args.root):
        args.root = os.path.normpath(args.root)
    r = {'build': build, 'plan': plan, 'evaluate': evaluate, 'analyze': analyze,
         'compare': compare, 'timing': timing, 'infer': infer,
         'archive-rules': archive_rules}[args.phase](args)
    if args.phase == 'infer' and args.check and not r['complete']:
        sys.exit(1)


if __name__ == '__main__':
    main()
