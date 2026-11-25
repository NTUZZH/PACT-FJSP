r"""P33: the policy against the best dispatching rule when the constraint magnitudes leave the
training ranges, at the training size (10x25).

THE SETS (exact environment, training seeds of arm_registry, PACT_SEEDS=301-305)
  ind     data/FAMILY/10x25, arrays in arm_registry.res_dir('10x25_cpuref'), which is
          10x25_xmask in the exact environment, the folder of FamRule<R> (asserted)
  setup   data/FAMILY/10x25_oodsetup: cross-class setups drawn from a larger range of the mean
          processing time and the matrix asymmetric; reaches S, LS, SW and LSW
  window  data/FAMILY/10x25_oodwindow: per-machine outage density drawn from a larger share of
          the horizon; reaches W, LW, SW and LSW
  scripts/p29_ood.py builds both from the base instances and descriptor seeds of 10x25, so on a
  combination a shift does not reach a set holds the constraint fields of 10x25. The drawn
  ranges are read from each set's dataset_meta.json (ood_shift) and the trained ranges from the
  defaults of constraint_family.descriptor (draw_setup_matrix cross, draw_windows
  density_range).

THE PROCEDURE: the procedure of FamRule<R> (scripts/p88_family_inference.py, section B, and
the protocol of manuscript/supp_new/sn07_statistics.tex), by p88's own functions
  d        100 (policy_s - best) / best per training seed s and test instance, best the
           per-instance minimum of FIFO, MOR, SPT and MWKR on that instance (p88.seed_pairs with
           p88.released and p88.pool_reader); negative = the policy is shorter
  interval p88.superiority_block: per combination the crossed hierarchical bootstrap over
           training seeds and test instances, B = 20000 resamples from a fresh
           default_rng(11) per combination (p88.fresh), 2.5th to 97.5th percentile
  tests    one-sided seed-level t over the per-seed means (4 df) and one-sided bootstrap p from
           the same resamples, both directions; Holm's correction over the family, each kind
           and each direction its own family
  verdict  ahead / behind / not distinguishable (p89_public_family.superiority_verdict)
  families inside the ranges the eight combinations (the family of FamRule); a set that changes
           one benchmark parameter has the combinations it changes: S, LS, SW, LSW for setup,
           W, LW, SW, LSW for window
  Because every combination draws its own fresh stream, an interval does not depend on its
  family; the Holm-adjusted p-values and therefore the verdict do.
  On a combination a shift does not reach, the arrays of the policy and of the four rules
  equal those inside the ranges bit for bit (asserted; the script refuses otherwise), so d,
  the interval and the verdict there are those inside the ranges, and the largest change of the
  mean there (Floor) is exactly zero. These combinations are reported, not counted.
  pooled   per shifted set, d of the combinations the set changes (the arrays of the verdicts
           above) pooled by p88.pooled_block: the crossed hierarchical bootstrap of
           stats_family.hier_boot over training seeds and test instances, a drawn test instance
           bringing all its combinations, B = 20000 from a fresh default_rng(11), 2.5th to
           97.5th percentile, and the direction of the interval (ahead, behind, neither); the
           routine of the pooled rows of scripts/p102_scale_sensitivity.py and
           scripts/p103_five_classes.py

THE POLICY AGAINST THE SPECIALISTS (the set of SPEC_SETS, the denser windows; the larger setups
are compared with the specialists by scripts/p119_asym_setups.py)
  d = 100 (policy_s - specialist_s) / specialist_s, seed s against seed s, the specialist of
  each combination on that combination (p88.seed_pairs with p88.released and p88.specialist),
  pooled over the combinations the set changes by p88.pooled_block (as above), with the
  direction and the verdicts at the margins 0.5% and 1% (stats_family.verdict). On the
  combinations the set does not reach, the specialists' arrays equal those inside the ranges bit
  for bit (asserted).

CHECK AGAINST FamRule (refusal on any difference)
  The rows inside the ranges equal the record of step family for the same environment, seeds,
  B and alpha: results/family_inference.json (rule_pool: mean, lo, hi, the four Holm-adjusted
  p-values and the verdict, exact floats) and paper/macros_family.tex (FamRule<R>, Lo, Hi and
  Verdict as printed), both read through arm_registry.out.

FURTHER QUANTITIES
  shift        mean setup entry of the S descriptors of 10x25_oodsetup over that of 10x25,
               whether the matrix of instance_000 is asymmetric, and the mean outage density of
               the W descriptors of 10x25_oodwindow over that of 10x25 (shift_magnitudes)
  infeasible   schedules the validator rejected, from the policy's summaries (infeasible), per
               set and combination over the seeds; Infeas sums the three sets

OUTPUTS (arm_registry.out under PACT_OUT)
  paper/macros_ood.tex   names Ood<quantity>, letters only, checked against every macro defined
                         in a .tex file under manuscript/ (other than a macros_ood.tex)
  results/ood.json       every number
  results/ood.md         one table per shifted set
  A refusal removes all three.

CHECK (--check, read only): exit 0 when every array and summary the run reads exists, every
array with one row per test instance (the specialists' arrays included), and the record of step
family exists.

DRY MODE (--dry [DIR]): the whole path on synthetic arrays in DIR (default a temporary folder,
removed at the end): the three sets with known differences (DRY_LEAD) and known gaps of the
policy to the specialists (DRY_SPEC), arrays equal to those inside the ranges where a shift does
not reach, a FamRule record computed from the synthetic arrays by the same functions,
B = DRY_BOOT.

THE EARLIER ENVIRONMENT (PACT_ENV=legacy): legacy_main(), the analysis of the earlier
submission, unchanged (its one-level bootstrap over the test instances of the seed-averaged
difference, one default_rng(7) stream, and the comparison with the descriptor-input variant,
which exists in that environment only). The paper reports the exact environment.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p33_ood_table.py
       ... python scripts/p33_ood_table.py --check
       python scripts/p33_ood_table.py --dry [DIR]
"""
import argparse
import contextlib
import glob
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, 'scripts'), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402
import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
from p89_public_family import VERDICT_WORDS  # noqa: E402

GENERATOR = 'scripts/p33_ood_table.py'
ENV = 'exact'
PREFIX = 'Ood'
OUT_FILES = {'macros': 'paper/macros_ood.tex', 'json': 'results/ood.json', 'md': 'results/ood.md'}
# the record of step family (p88), read through arm_registry.out
FAMILY_FILES = {'json': 'results/family_inference.json', 'macros': 'paper/macros_family.tex'}
FAMILY_FOLDER = '10x25'           # the folder p88 reads for FamRule<R>
FAMILY_KEYS = ('mean', 'lo', 'hi', 'p_t_holm', 'p_boot_holm', 'p_t_greater_holm',
               'p_boot_greater_holm', 'verdict')
ALPHA = 0.05
R8 = list(REG.COMBOS)
# (name, results folder key, data folder)
SETS = (('ind', '10x25_cpuref', 'data/FAMILY/10x25'),
        ('setup', '10x25_oodsetup', 'data/FAMILY/10x25_oodsetup'),
        ('window', '10x25_oodwindow', 'data/FAMILY/10x25_oodwindow'))
SHIFTED = ('setup', 'window')
SPEC_SETS = ('window',)           # the sets compared with the specialists here
REACHED = {'ind': R8, 'setup': [r for r in R8 if 'S' in r], 'window': [r for r in R8 if 'W' in r]}
FAMILY = {name: list(REACHED[name]) for name in REACHED}      # Holm family of each set
TEXT = {'ind': 'inside the training ranges (10x25)',
        'setup': 'larger asymmetric setups (10x25_oodsetup)',
        'window': 'denser machine unavailability windows (10x25_oodwindow)'}
VERDICT_KEY = {VERDICT_WORDS[0]: 'Ahead', VERDICT_WORDS[2]: 'Nd', VERDICT_WORDS[1]: 'Behind'}
DRY_BOOT = 400

# ---- the constants and functions of the earlier analysis (legacy_main uses them; infeasible,
# shift_magnitudes and the lists are also used by the procedure above)
rng = np.random.default_rng(7)          # the earlier analysis' bootstrap seed, unchanged
MARGIN = 1.0                            # the paper's +-1% equivalence margin
B = 20000
SEEDS = REG.seeds()                     # the active seeds at import (the earlier analysis)
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
RULES = ['FIFO', 'MOR', 'SPT', 'MWKR']  # the same four the fair warm start uses
# The main arm is the released PACT policy (token-blind, 32786 parameters); the
# conditioned variant (71838 parameters) is the comparison arm of the last block.
JOINT = 'released'                            # the released PACT policy (arm_registry)
BLIND = 'joint-v1'                            # the conditioned variant (legacy only)
WORD = {'ind': 'Ind', 'setup': 'Setup', 'window': 'Window'}


class Refused(Exception):
    """A setting or an input that this script does not accept."""


def arr(res, tag, r):
    p = f'{res}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def seed_tag(base, s):
    return REG.tag(base, None, s) if base == JOINT else REG.other_tag(base, s)


def pooled(res, base, r, seed_list=None):
    """Per-instance mean over the seed replicates (default: every active seed). A
    missing seed is an error, not a thing to skip: pooling one side over two seeds
    and the other over three is the asymmetry this project has already had to
    retract once."""
    seed_list = seed_list or SEEDS
    xs = [arr(res, seed_tag(base, s), r) for s in seed_list]
    if any(x is None for x in xs):
        missing = [s for s, x in zip(seed_list, xs) if x is None]
        raise SystemExit(f'REFUSING TO REPORT: {res}/{base} on {r} is missing seed(s) '
                         f'{missing}. Every cell in this table is pooled over the '
                         f'same three seeds on both sides.')
    return np.mean(xs, axis=0), len(xs)


def infeasible(res, tag, r, sfx=''):
    """The count of schedules the INDEPENDENT VALIDATOR rejected.

    Not from the .npy. The .npy holds the makespan in column 0 and the per-instance
    inference time in column 1; feasibility is nowhere in it, so a check of the form
    `sum(~isfinite(makespan))` is structurally zero for every input that has ever
    existed. It is a gate that cannot fail, and a gate that cannot fail is worse than
    no gate, because it prints a zero that looks like evidence.

    eval_family.py runs constraint_family.reference_sim.validate_schedule on every
    realized schedule and records the survivors in the summary table. That table is
    the only place the validator's verdict is written down, so it is where the number
    has to come from."""
    p = f'{res}/summary_{tag.split("+", 1)[1]}{sfx}.md'
    if not os.path.exists(p):
        raise SystemExit(f'REFUSING TO REPORT feasibility: {p} does not exist, and '
                         f'the validator\'s verdict lives nowhere else. The .npy '
                         f'carries makespan and inference time, not feasibility.')
    for line in open(p):
        cells = [c.strip() for c in line.split('|')]
        if len(cells) > 5 and cells[1] == r and cells[2] == tag:
            ok, total = cells[5].split('/')          # the "feasible" column, "100/100"
            return int(total) - int(ok)
    raise SystemExit(f'REFUSING: no row for regime {r}, method {tag} in {p}')


def best_rule(res, r):
    """Per-instance best of the four dispatching rules. Missing any one of them
    is a hard error: the baseline the paper names is the best of four, and a baseline
    built from fewer rules is a weaker opponent than the one it claims to be."""
    xs = []
    for rule in RULES:
        a = arr(res, rule, r)
        if a is None:
            raise SystemExit(
                f'REFUSING TO REPORT: {res} has no {rule} on regime {r}. The '
                f'in-distribution baseline is the best of {RULES}; a baseline '
                f'built from fewer rules is a weaker opponent and would overstate '
                f'the policy. Run the missing rules first.')
        xs.append(a)
    return np.min(np.stack(xs), axis=0)


def ci(d):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def verdict(lo, hi):
    if hi < 0:
        return 'policy AHEAD'
    if lo > 0:
        return 'policy BEHIND'
    return 'tied'


def shift_magnitudes(ref='data/FAMILY/10x25', oods='data/FAMILY/10x25_oodsetup',
                     oodw='data/FAMILY/10x25_oodwindow'):
    """How far out of distribution did we actually go? Read it from the
    descriptors, not from the arguments we think we passed. setup_x: mean setup entry
    of the S descriptors of oods over that of ref; window_x: mean outage density of the W
    descriptors of oodw over that of ref; asym: the matrix of instance_000 of oods is not
    symmetric."""
    out = {}

    def setups(d):
        v = []
        for p in sorted(glob.glob(f'{d}/desc/S/*.desc.json'))[:100]:
            v += [x for row in json.load(open(p))['setup'] for x in row]
        return np.mean(v)

    def density(d):
        v = []
        for p in sorted(glob.glob(f'{d}/desc/W/*.desc.json'))[:100]:
            j = json.load(open(p))
            v.append(sum(e - s for m in j['windows'] for s, e in m)
                     / (25 * j['horizon']))
        return np.mean(v)

    out['setup_x'] = setups(oods) / setups(ref)
    out['window_x'] = density(oodw) / density(ref)
    A = np.array(json.load(open(f'{oods}/desc/S/instance_000.desc.json'))['setup'])[:-1]
    out['asym'] = not np.allclose(A, A.T)
    return out


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the procedure reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV}; the earlier environment runs '
                      f'legacy_main)')
    if REG.res_dir(SETS[0][1]) != REG.res_dir(FAMILY_FOLDER):
        raise Refused(f'the arrays inside the ranges ({REG.res_dir(SETS[0][1])}) are not the '
                      f'folder of FamRule ({REG.res_dir(FAMILY_FOLDER)})')
    return list(REG.seeds())


def n_test(data):
    return len(glob.glob(os.path.join(data, 'desc', 'N', '*.desc.json')))


def summary_path(key, s):
    """The validator summary infeasible() reads for the policy of seed s."""
    t = REG.tag(JOINT, None, s)
    return f'{REG.res_dir(key)}/summary_{t.split("+", 1)[1]}{REG.summary_tag(key)}.md'


def family_paths():
    return {k: REG.out(v) for k, v in FAMILY_FILES.items()}


def inputs(seeds, sets=SETS):
    """[(path, rows or None)] of every array and summary the run reads."""
    out = []
    for _, key, data in sets:
        res, n = REG.res_dir(key), n_test(data)
        for s in seeds:
            for r in R8:
                out.append((REG.result_path(JOINT, None, s, key, r), n))
            out.append((summary_path(key, s), None))
        for q in RULES:
            for r in R8:
                out.append((f'{res}/Result_{q}_{r}.npy', n))
    return out + spec_inputs(seeds, sets)


def spec_inputs(seeds, sets=SETS):
    """[(path, rows)] of the specialists' arrays the comparison with the specialists reads: for
    each set of SPEC_SETS among sets, the specialist of every combination on it, and inside the
    ranges (the first of sets) the specialists of the combinations the set does not reach (the
    control)."""
    by = {name: (key, data) for name, key, data in sets}
    ind_key, out = sets[0][1], []
    for name in SPEC_SETS:
        if name not in by:
            continue
        key, data = by[name]
        n = n_test(data)
        for s in seeds:
            for r in R8:
                out.append((REG.result_path('specialist', r, s, key, r), n))
                if r not in REACHED[name]:
                    out.append((REG.result_path('specialist', r, s, ind_key, r), n))
    return out


def summary_gaps(path, tag):
    """The combinations without a feasible count of method tag in a validator summary."""
    have = set()
    for line in open(path):
        c = [x.strip() for x in line.split('|')]
        if len(c) > 5 and c[2] == tag and re.fullmatch(r'\d+/\d+', c[5]):
            have.add(c[1])
    return [r for r in R8 if r not in have]


def input_problems(seeds, sets=SETS, record=True):
    """Every array with one row per test instance, every validator summary with a feasible count
    for each combination, and (record) the record of step family."""
    bad = []
    for _, key, data in sets:
        if n_test(data) == 0:
            bad.append(f'{data}/desc/N holds no descriptor')
        for s in seeds:
            p = summary_path(key, s)
            gone = summary_gaps(p, REG.tag(JOINT, None, s)) if os.path.exists(p) else []
            if gone:
                bad.append(f'{p}: no feasible count for {", ".join(gone)}')
    for p, n in inputs(seeds, sets):
        if not os.path.exists(p):
            bad.append(f'missing {p}')
        elif n is not None and len(REG.read_path(p)) != n:
            bad.append(f'{p}: {len(REG.read_path(p))} rows, the test set has {n}')
    if record:
        for p in family_paths().values():
            if not os.path.exists(p):
                bad.append(f'missing {p} (the record of step family, '
                           f'scripts/p88_family_inference.py)')
    return bad


# ----------------------------------------------------------------- the procedure
def differences(key, combos, seeds):
    """{R: d [k, n]}: d = 100 (policy_s - best) / best, seed s against the per-instance best of
    the four rules (p88.seed_pairs, p88.released, p88.pool_reader)."""
    return {r: F88.seed_pairs(F88.released, F88.pool_reader, key, r, seeds=list(seeds))
            for r in combos}


def family_rows(D, family):
    """p88.superiority_block over the combinations of family: interval (fresh draws per
    combination), one-sided tests in both directions, Holm over the family, verdict."""
    rows = F88.superiority_block({r: D[r] for r in family})
    for r in family:
        rows[r]['holm_family'] = list(family)
        rows[r]['reached'] = True
    return rows


def control(seeds, sets=SETS, reached=None):
    """On every combination a shifted set does not reach, the policy's array of every seed and
    the arrays of the four rules equal those inside the ranges bit for bit (makespan column).
    sets: (name, results folder key, data folder), the set inside the ranges first; reached:
    {name: combinations the set changes} (default REACHED). Returns the number of arrays
    compared."""
    reached = reached or REACHED
    ind_key = sets[0][1]
    n = 0
    for name, key, _ in sets[1:]:
        for r in R8:
            if r in reached[name]:
                continue
            pairs = [(REG.result_path(JOINT, None, s, ind_key, r),
                      REG.result_path(JOINT, None, s, key, r)) for s in seeds]
            pairs += [(f'{REG.res_dir(ind_key)}/Result_{q}_{r}.npy',
                       f'{REG.res_dir(key)}/Result_{q}_{r}.npy') for q in RULES]
            for a, b in pairs:
                x, y = REG.read_path(a), REG.read_path(b)
                if not np.array_equal(x, y):
                    raise Refused(
                        f'CONTROL FAILED: {name}, combination {r} is not reached by the shift, '
                        f'but {b} differs from {a} on {int((x != y).sum())} of {len(x)} '
                        f'instances. The sets were not evaluated by the same command; no '
                        f'number of this script would be the shift alone.')
                n += 1
    return n


def set_rows(seeds, sets=SETS, reached=None):
    """Rows {(name, R): row} of every set and combination, the floor and the number of arrays
    compared by the control. The set inside the ranges (first of sets) has the family of the
    eight combinations; a shifted set has the family of the combinations it reaches (reached,
    default REACHED), and on the others its rows are those inside the ranges (reached False),
    after the control and an exact comparison of the differences."""
    reached = reached or REACHED
    n_ctl = control(seeds, sets, reached)
    rows, floor = {}, 0.0
    ind = sets[0][0]
    D_ind = differences(sets[0][1], R8, seeds)
    for r, x in family_rows(D_ind, R8).items():
        rows[(ind, r)] = x
    for name, key, _ in sets[1:]:
        fam = list(reached[name])
        rest = [r for r in R8 if r not in fam]
        for r, x in family_rows(differences(key, fam, seeds), fam).items():
            rows[(name, r)] = x
        for r, d in differences(key, rest, seeds).items():
            if not np.array_equal(d, D_ind[r]):
                raise Refused(f'CONTROL FAILED: {name}, combination {r}: the differences '
                              f'differ from those inside the ranges')
            floor = max(floor, abs(float(d.mean()) - rows[(ind, r)]['mean']))
            rows[(name, r)] = dict(rows[(ind, r)], reached=False)
    if floor != 0.0:
        raise Refused(f'CONTROL FAILED: a combination no shift reaches moved by {floor} points')
    return rows, floor, n_ctl


def pooled_rows(seeds, sets=SETS, reached=None):
    """{name: row} of every shifted set (sets after the first): the differences of the
    combinations the set changes (reached, default REACHED; the arrays of its per-combination
    verdicts, p33.differences), pooled by p88.pooled_block: mean, crossed hierarchical 95%
    interval with all combinations of a drawn test instance kept together (fresh
    default_rng(11)), seed-level t interval, direction, and the list of the pooled
    combinations."""
    reached = reached or REACHED
    out = {}
    for name, key, _ in sets[1:]:
        fam = list(reached[name])
        row = F88.pooled_block(differences(key, fam, seeds), {name: fam})[name]
        out[name] = dict(row, combos=fam)
    return out


def spec_control(seeds, sets=SETS):
    """On every combination a set of SPEC_SETS does not reach, the specialist's array of every
    seed equals that inside the ranges bit for bit. Returns the number of arrays compared."""
    by = {name: key for name, key, _ in sets}
    ind_key, n = sets[0][1], 0
    for name in SPEC_SETS:
        if name not in by:
            continue
        for r in R8:
            if r in REACHED[name]:
                continue
            for s in seeds:
                a = REG.result_path('specialist', r, s, ind_key, r)
                b = REG.result_path('specialist', r, s, by[name], r)
                x, y = REG.read_path(a), REG.read_path(b)
                if not np.array_equal(x, y):
                    raise Refused(
                        f'CONTROL FAILED: {name}, combination {r} is not reached by the shift, '
                        f'but the specialist {b} differs from {a} on {int((x != y).sum())} of '
                        f'{len(x)} instances. The sets were not evaluated by the same command.')
                n += 1
    return n


def spec_block(seeds, sets=SETS):
    """{name: row} of every set of SPEC_SETS among sets: the policy against the specialist of
    each combination the set changes, d = 100 (policy_s - specialist_s) / specialist_s seed
    against seed (p88.seed_pairs, p88.released, p88.specialist), pooled over those combinations
    by p88.pooled_block (mean, hierarchical 95% interval, direction, verdicts at the margins),
    after the control of the specialists' arrays (spec_control)."""
    n_ctl = spec_control(seeds, sets)
    by = {name: key for name, key, _ in sets}
    out = {}
    for name in SPEC_SETS:
        if name not in by:
            continue
        fam = list(REACHED[name])
        D = {r: F88.seed_pairs(F88.released, F88.specialist, by[name], r, seeds=list(seeds))
             for r in fam}
        out[name] = dict(F88.pooled_block(D, {name: fam})[name], combos=fam,
                         control_arrays=n_ctl)
    return out


def family_record(seeds):
    """(rule_pool, macros, paths) of the record of step family for this setting."""
    paths = family_paths()
    for p in paths.values():
        if not os.path.exists(p):
            raise Refused(f'{p} is missing; run step family (scripts/p88_family_inference.py) '
                          f'for this setting first')
    rec = json.load(open(paths['json']))
    want = dict(seeds=list(seeds), boot=F88.B, alpha=F88.ALPHA)
    got = {k: (list(rec.get(k)) if k == 'seeds' and rec.get(k) is not None else rec.get(k))
           for k in want}
    if got != want:
        raise Refused(f'{paths["json"]} records {got}, this run uses {want}')
    if sorted(rec.get('rule_pool') or {}) != sorted(R8):
        raise Refused(f'{paths["json"]} holds no rule_pool row for every combination')
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(paths['macros']).read()))
    return rec['rule_pool'], mac, paths


def check_family(rows, seeds):
    """The rows inside the ranges against FamRule<R>: the record's numbers exactly, the macros
    as printed. Returns the comparison record; refuses on any difference."""
    pool, mac, paths = family_record(seeds)
    bad, n = [], 0
    for r in R8:
        x, y = rows[('ind', r)], pool[r]
        for k in FAMILY_KEYS:
            n += 1
            if x[k] != y.get(k):
                bad.append(f'{r} {k}: {paths["json"]} {y.get(k)!r}, here {x[k]!r}')
        for sfx, v in (('', F88.f2(x['mean'])), ('Lo', F88.f2(x['lo'])), ('Hi', F88.f2(x['hi'])),
                       ('Verdict', x['verdict'])):
            n += 1
            if mac.get(f'FamRule{r}{sfx}') != v:
                bad.append(f'\\FamRule{r}{sfx}: {paths["macros"]} {mac.get(f"FamRule{r}{sfx}")!r}, '
                           f'here {v!r}')
    if bad:
        raise Refused(f'the values inside the training ranges differ from FamRule in {len(bad)} '
                      f'of {n} comparisons:\n  ' + '\n  '.join(bad[:12]))
    return dict(json=paths['json'], macros=paths['macros'], compared=n, identical=True,
                keys=list(FAMILY_KEYS), macro_suffixes=['', 'Lo', 'Hi', 'Verdict'])


def drawn_ranges():
    """The ranges the generator drew from: the shifted sets' dataset_meta.json (ood_shift) and
    the trained ranges, the defaults of constraint_family.descriptor."""
    import inspect
    from constraint_family.descriptor import draw_setup_matrix, draw_windows
    trained = {'cross': list(inspect.signature(draw_setup_matrix).parameters['cross'].default),
               'density_range': list(inspect.signature(draw_windows)
                                     .parameters['density_range'].default)}
    meta = {}
    for name, _, data in SETS:
        p = os.path.join(data, 'dataset_meta.json')
        if not os.path.exists(p):
            raise Refused(f'{p} is missing; the drawn ranges are read from it')
        meta[name] = json.load(open(p)).get('ood_shift')
    if meta['ind'] is not None:
        raise Refused('data/FAMILY/10x25/dataset_meta.json records a shift')
    s, w = meta['setup'] or {}, meta['window'] or {}
    if set(s) != {'cross', 'symmetric'} or s['symmetric'] is not False:
        raise Refused(f'10x25_oodsetup records the shift {s}, want cross and symmetric false')
    if set(w) != {'density_range'}:
        raise Refused(f'10x25_oodwindow records the shift {w}, want density_range')
    return {'setup': dict(quantity='cross-class setup', unit='share of the mean processing time '
                          'of the instance', drawn=[float(v) for v in s['cross']],
                          trained=[float(v) for v in trained['cross']], symmetric=False,
                          source='data/FAMILY/10x25_oodsetup/dataset_meta.json (ood_shift cross); '
                                 'trained: default cross of draw_setup_matrix'),
            'window': dict(quantity='outage density per machine', unit='share of the horizon',
                           drawn=[float(v) for v in w['density_range']],
                           trained=[float(v) for v in trained['density_range']],
                           source='data/FAMILY/10x25_oodwindow/dataset_meta.json (ood_shift '
                                  'density_range); trained: default density_range of '
                                  'draw_windows')}


def verdict_lists(rows, name):
    """{verdict word: [reached combinations]} of a set, in the order of the combinations."""
    return {w: [r for r in REACHED[name] if rows[(name, r)]['verdict'] == w]
            for w in VERDICT_WORDS}


def analyse(dry=False, boot=None):
    seeds = check_setting()
    bad = input_problems(seeds)
    if bad:
        raise Refused(f'{len(bad)} inputs do not count; first: {bad[0]}')
    F88.configure(boot=boot if boot is not None else F88.SF.B_DEFAULT, alpha=ALPHA)
    for _, _, data in SETS:
        F88.verify_bases(os.path.basename(data), n_test(data))
    rows, floor, n_ctl = set_rows(seeds)
    pooled = pooled_rows(seeds)
    spec = spec_block(seeds)
    fam_check = check_family(rows, seeds)
    infeas = {}
    for name, key, _ in SETS:
        res, sfx = REG.res_dir(key), REG.summary_tag(key)
        for r in R8:
            infeas[(name, r)] = int(sum(infeasible(res, REG.tag(JOINT, None, s), r, sfx)
                                        for s in seeds))
    sm = shift_magnitudes()
    return dict(
        generator=GENERATOR, setting=REG.describe(), environment=REG.env(), seeds=seeds,
        dry=bool(dry), boot=F88.B, alpha=F88.ALPHA,
        procedure=dict(
            difference='d = 100 (policy_s - best) / best per training seed s and test instance, '
                       'best = per-instance minimum of FIFO, MOR, SPT, MWKR; negative = the policy '
                       'is shorter (p88.seed_pairs, p88.released, p88.pool_reader)',
            interval='crossed hierarchical bootstrap over training seeds and test instances, '
                     f'B = {F88.B}, fresh default_rng({F88.FRESH_SEED}) per combination (p88.fresh), '
                     '2.5th to 97.5th percentile',
            tests='one-sided seed-level t (k - 1 df) and one-sided bootstrap p, both directions, '
                  "Holm over the set's family, each kind and direction its own family "
                  '(p88.superiority_block)',
            verdict=f'ahead: interval below zero and both Holm-adjusted p for H1 policy shorter '
                    f'below {F88.ALPHA:g}; behind: the mirrored condition; not distinguishable '
                    'otherwise (p89_public_family.superiority_verdict)',
            families={name: FAMILY[name] for name in FAMILY},
            not_reached='the arrays equal those inside the ranges (asserted); the rows are those '
                        'inside the ranges, not counted',
            pooled='per shifted set, d of the combinations it changes pooled by '
                   'p88.pooled_block: crossed hierarchical bootstrap over training seeds and '
                   'test instances, a drawn instance bringing all its combinations, '
                   f'B = {F88.B}, fresh default_rng({F88.FRESH_SEED}), and the direction of '
                   'the interval',
            specialist=f'{", ".join(TEXT[n] for n in SPEC_SETS)}: d = 100 (policy_s - '
                       'specialist_s) / specialist_s, seed s against seed s, pooled over the '
                       'combinations the set changes by p88.pooled_block, direction and verdicts '
                       'at the margins 0.5% and 1%'),
        sets={name: dict(data=data, folder=REG.res_dir(key), n=n_test(data),
                         reaches=list(REACHED[name]), holm_family=list(FAMILY[name]),
                         text=TEXT[name]) for name, key, data in SETS},
        rows={f'{k[0]}|{k[1]}': v for k, v in rows.items()},
        pooled=pooled,
        specialist=spec,
        counts={name: verdict_lists(rows, name) for name in SHIFTED},
        control=dict(arrays=n_ctl, floor=floor,
                     combinations={name: [r for r in R8 if r not in REACHED[name]]
                                   for name in SHIFTED}),
        family_check=fam_check,
        infeasible={f'{k[0]}|{k[1]}': v for k, v in infeas.items()},
        infeasible_total=int(sum(infeas.values())),
        shift=dict(setup_x=float(sm['setup_x']), window_x=float(sm['window_x']),
                   asymmetric=bool(sm['asym'])),
        ranges=drawn_ranges())


# ----------------------------------------------------------------- outputs
def setting_words(res):
    return f'environment {res["environment"]}, training seeds {",".join(map(str, res["seeds"]))}'


def pct(x):
    return f'{round(100.0 * x, 6):g}'


def names(xs):
    return ', '.join(xs) or 'none'


def holm_family_words(x):
    return ', '.join(x['holm_family'])


def macro_lines(res):
    f2 = F88.f2
    L = [f'% Generated by {GENERATOR} ({setting_words(res)}) -- do not hand-edit',
         '% The policy against the per-instance best of FIFO, MOR, SPT and MWKR at 10x25: Ind inside '
         'the training ranges, Setup with larger asymmetric setups (10x25_oodsetup), Window with '
         'denser machine unavailability windows (10x25_oodwindow). d = 100 (policy - best) / best '
         'per training seed and test instance, each seed against the best rule on the same instance '
         f'(negative = the policy is shorter); mean, hierarchical 95% interval (B = {res["boot"]}, '
         'fresh draws per combination) and combined verdict by the procedure of '
         'scripts/p88_family_inference.py (FamRule<R>): ahead = interval below zero and both '
         'Holm-adjusted one-sided p-values (seed-level t, bootstrap) below '
         f'{res["alpha"]:g}; behind = the mirrored condition; not distinguishable otherwise.',
         '% Holm families: Ind the eight combinations (the family of FamRule; the Ind values equal '
         'FamRule<R>, checked); Setup S, LS, SW, LSW; Window W, LW, SW, LSW (the combinations each '
         'shift changes). On a combination a shift does not reach, the arrays equal those inside '
         'the ranges (asserted), and the values and the verdict are those of Ind.']
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    sh, rg = res['shift'], res['ranges']
    L.append(f'\\newcommand{{\\{PREFIX}SetupX}}{{{sh["setup_x"]:.2f}}}'
             '  % 10x25_oodsetup: mean entry of the setup matrices of its S descriptors over that '
             f'of 10x25 (the matrix of instance_000 {"asymmetric" if sh["asymmetric"] else "symmetric"})')
    L.append(f'\\newcommand{{\\{PREFIX}WindowX}}{{{sh["window_x"]:.2f}}}'
             '  % 10x25_oodwindow: mean outage density of its W descriptors (outage hours over the '
             '25 machines times the horizon) over that of 10x25')
    s, w = rg['setup'], rg['window']
    L.append(f'\\newcommand{{\\{PREFIX}SetupCrossLo}}{{{pct(s["drawn"][0])}}}'
             f'\\newcommand{{\\{PREFIX}SetupCrossHi}}{{{pct(s["drawn"][1])}}}'
             f'\\newcommand{{\\{PREFIX}SetupCrossTrainLo}}{{{pct(s["trained"][0])}}}'
             f'\\newcommand{{\\{PREFIX}SetupCrossTrainHi}}{{{pct(s["trained"][1])}}}'
             '  % cross-class setups drawn uniformly between Lo and Hi percent of the mean processing '
             'time of the instance: 10x25_oodsetup (its dataset_meta.json, the matrix asymmetric) and '
             'the trained range (the default of draw_setup_matrix)')
    L.append(f'\\newcommand{{\\{PREFIX}WindowDensityLo}}{{{pct(w["drawn"][0])}}}'
             f'\\newcommand{{\\{PREFIX}WindowDensityHi}}{{{pct(w["drawn"][1])}}}'
             f'\\newcommand{{\\{PREFIX}WindowDensityTrainLo}}{{{pct(w["trained"][0])}}}'
             f'\\newcommand{{\\{PREFIX}WindowDensityTrainHi}}{{{pct(w["trained"][1])}}}'
             '  % outage density of each machine drawn uniformly between Lo and Hi percent of the '
             'horizon: 10x25_oodwindow (its dataset_meta.json) and the trained range (the default of '
             'draw_windows)')
    L.append(f'\\newcommand{{\\{PREFIX}Infeas}}{{{res["infeasible_total"]}}}'
             '  % schedules the validator rejected, summed over the three sets (inside the ranges, '
             '10x25_oodsetup, 10x25_oodwindow), the eight combinations and the training seeds')
    L.append(f'\\newcommand{{\\{PREFIX}Floor}}{{{res["control"]["floor"]:.2f}}}'
             '  % largest change of the mean difference on a combination a shift does not reach '
             '(points); zero because the arrays of the policy and of the four rules there equal '
             'those inside the ranges (asserted)')
    n_out = sum(len(v) for v in res['control']['combinations'].values())
    n_in = sum(len(REACHED[name]) for name in SHIFTED)
    L.append(f'\\newcommand{{\\{PREFIX}Placebo}}{{{n_out}}}\\newcommand{{\\{PREFIX}Treated}}{{{n_in}}}'
             '  % combinations the shift does not reach / reaches, summed over the two shifted sets')
    for name, _, _ in SETS:
        w_ = WORD[name]
        for r in R8:
            x = res['rows'][f'{name}|{r}']
            what = (f'{TEXT[name]}, {r}: the policy against the best dispatching rule (%), '
                    f'hierarchical 95% CI and combined verdict, Holm over {holm_family_words(x)}; '
                    f'{F88.holm_text(x)}')
            if not x['reached']:
                what = (f'{TEXT[name]}, {r}: not reached by the shift; the arrays are those inside '
                        f'the ranges, so these are the values and the verdict of \\{PREFIX}Ind{r}')
            L.append(f'\\newcommand{{\\{PREFIX}{w_}{r}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\{PREFIX}{w_}{r}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\{PREFIX}{w_}{r}Hi}}{{{f2(x["hi"])}}}'
                     f'\\newcommand{{\\{PREFIX}{w_}{r}Verdict}}{{{x["verdict"]}}}  % {what}')
    for name in SHIFTED:
        w_, v = WORD[name], res['counts'][name]
        L.append(''.join(f'\\newcommand{{\\{PREFIX}{w_}{VERDICT_KEY[x]}N}}{{{len(v[x])}}}'
                         for x in VERDICT_KEY)
                 + f'  % {TEXT[name]}: of the {len(REACHED[name])} combinations the shift reaches '
                   f'({", ".join(REACHED[name])}), those with the combined verdict ahead, not '
                   'distinguishable, behind')
        L.append(''.join(f'\\newcommand{{\\{PREFIX}{w_}{VERDICT_KEY[x]}Combos}}{{{names(v[x])}}}'
                         for x in VERDICT_KEY)
                 + f'  % {TEXT[name]}: the reached combinations of each combined verdict (ahead, not '
                   'distinguishable, behind)')
    for name in SHIFTED:
        x = res['pooled'][name]
        L.append(F88.row_line(
            f'{PREFIX}{WORD[name]}Pool', x,
            f'{TEXT[name]}: the policy against the best dispatching rule (%), pooled over the '
            f'{x["c"]} combinations the shift reaches ({", ".join(x["combos"])}), the differences '
            f'of the per-combination verdicts; mean, hierarchical 95% interval over training seeds '
            f'and test instances with all combinations of a drawn instance kept together '
            f'(B = {res["boot"]}, fresh draws), direction of the interval (ahead, behind, neither)',
            margins=False))
    for name in SPEC_SETS:
        x = res['specialist'][name]
        L.append(F88.row_line(
            f'{PREFIX}{WORD[name]}SpecPool', x,
            f'{TEXT[name]}: the policy against the specialist of each combination (%), seed '
            f'against seed, pooled over the {x["c"]} combinations the shift reaches '
            f'({", ".join(x["combos"])}); mean, hierarchical 95% interval as for '
            f'\\{PREFIX}{WORD[name]}Pool, direction, verdicts at the margins 0.5% and 1%'))
    return L


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


def check_names(lines):
    names_ = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names_ if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names_) != len(set(names_)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    clash = sorted(x for x in names_ if x in have)
    if clash:
        raise Refused('macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    return names_


def _ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def markdown(res):
    rg = res['ranges']
    md = [f'# The policy against the best dispatching rule outside the training ranges '
          f'({setting_words(res)})', '',
          f'{res["procedure"]["difference"]}. {res["procedure"]["interval"]}. '
          f'{res["procedure"]["tests"]}. Verdict: {res["procedure"]["verdict"]}.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          f'Inside the training ranges the Holm family is the eight combinations, and every row '
          f'equals FamRule<R> of `{res["family_check"]["json"]}` and `{res["family_check"]["macros"]}` '
          f'({res["family_check"]["compared"]} comparisons: mean, interval ends, Holm-adjusted '
          f'p-values and verdict exactly, the macros as printed).', '',
          f'Control: {res["control"]["arrays"]} arrays of the policy and of the four rules on the '
          f'combinations a shift does not reach are bit-identical to those inside the ranges '
          f'(largest change of the mean there {res["control"]["floor"]:.2f}). Schedules rejected by '
          f'the validator, three sets, eight combinations, every seed: {res["infeasible_total"]}.']
    heads = {'setup': (f'Cross-class setups drawn from U[{pct(rg["setup"]["drawn"][0])}%, '
                       f'{pct(rg["setup"]["drawn"][1])}%] of the mean processing time (trained '
                       f'U[{pct(rg["setup"]["trained"][0])}%, {pct(rg["setup"]["trained"][1])}%]), '
                       f'the matrix asymmetric; mean setup entry {res["shift"]["setup_x"]:.2f} times '
                       'that of 10x25.'),
             'window': (f'Outage density per machine drawn from U[{pct(rg["window"]["drawn"][0])}%, '
                        f'{pct(rg["window"]["drawn"][1])}%] of the horizon (trained '
                        f'U[{pct(rg["window"]["trained"][0])}%, {pct(rg["window"]["trained"][1])}%]); '
                        f'mean outage density {res["shift"]["window_x"]:.2f} times that of 10x25.')}
    for name in SHIFTED:
        md += ['', f'## {TEXT[name][0].upper() + TEXT[name][1:]}', '', heads[name]
               + f' Holm family outside the ranges: {", ".join(FAMILY[name])}.', '',
               '| comb. | inside the ranges | verdict | outside the ranges | verdict | '
               'Holm p shorter (t / boot) | Holm p longer (t / boot) | infeasible |',
               '|---|---|---|---|---|---|---|---|']
        for r in REACHED[name]:
            a, x = res['rows'][f'ind|{r}'], res['rows'][f'{name}|{r}']
            md.append(f'| {r} | {_ci(a)} | {a["verdict"]} | {_ci(x)} | {x["verdict"]} | '
                      f'{x["p_t_holm"]:.2g} / {x["p_boot_holm"]:.2g} | '
                      f'{x["p_t_greater_holm"]:.2g} / {x["p_boot_greater_holm"]:.2g} | '
                      f'{res["infeasible"][f"{name}|{r}"]} |')
        v = res['counts'][name]
        md += ['', f'Outside the ranges the policy is ahead on {names(v[VERDICT_WORDS[0]])}, not '
               f'distinguishable on {names(v[VERDICT_WORDS[2]])} and behind on '
               f'{names(v[VERDICT_WORDS[1]])}. Not reached, values as inside the ranges: '
               f'{", ".join(res["control"]["combinations"][name])}.']
        x = res['pooled'][name]
        md += ['', f'Pooled over {", ".join(x["combos"])} (a drawn test instance brings all its '
               f'combinations): {_ci(x)}, direction {x["direction"]}.']
        if name in res['specialist']:
            y = res['specialist'][name]
            md += ['', f'The policy against the specialists, seed against seed, pooled over '
                   f'{", ".join(y["combos"])}: {_ci(y)}, direction {y["direction"]}, '
                   f'{y["v0.5"]} at 0.5%, {y["v1.0"]} at 1% ({y["control_arrays"]} arrays of '
                   f'the specialists on the combinations not reached equal to those inside the '
                   f'ranges).']
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(dry=False, boot=None):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md)."""
    check_setting()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(dry, boot)
        lines = macro_lines(res)
        names_ = check_names(lines)
        emit(outs['macros'], lines)
        os.makedirs(os.path.dirname(outs['json']) or '.', exist_ok=True)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names_, md


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    seeds = check_setting()
    bad = input_problems(seeds)
    print(f'p33: folders {", ".join(REG.res_dir(key) for _, key, _ in SETS)}')
    print(f'p33: arrays and summaries of 10x25, 10x25_oodsetup and 10x25_oodwindow, seeds '
          f'{",".join(map(str, seeds))}, and the record of step family '
          f'({", ".join(family_paths().values())}): '
          + ('every one present, every array with one row per test instance' if not bad else
             f'{len(bad)} do not count'))
    for x in bad[:6]:
        print(f'p33:   {x}')
    if len(bad) > 6:
        print(f'p33:   ... and {len(bad) - 6} more')
    return 0 if not bad else 1


# ----------------------------------------------------------------- dry mode
# true differences in % against the best rule: inside the ranges on every combination, and on
# the combinations each shift reaches
DRY_LEAD = {'ind': {r: -2.0 for r in R8},
            'setup': {'S': -1.0, 'LS': -1.0, 'SW': 0.0, 'LSW': -1.0},
            'window': {'W': -3.0, 'LW': -3.0, 'SW': -3.0, 'LSW': -3.0}}
DRY_NOISE = 0.2
# true differences in % of the policy from the specialist of each combination, per set
DRY_SPEC = {'ind': 0.3, 'setup': 0.6, 'window': 0.9}


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


def dry_paths(tree):
    return dict(PACT_ENV=ENV, PACT_SEEDS='301-305', PACT_RESULTS=os.path.join(tree, 'results'),
                PACT_OUT=os.path.join(tree, 'out'))


def write_summaries(key, seeds, n):
    """Validator summaries of the policy with every schedule feasible."""
    for s in seeds:
        t = REG.tag(JOINT, None, s)
        rows = [f'| {r} | {t} | 0.0 | 0.0 | {n}/{n} | n/a | n/a |' for r in R8]
        with open(summary_path(key, s), 'w') as f:
            f.write(f'# FAMILY eval -- {t}\n\n| regime | method | mean | std | feasible | '
                    f'gap(cov) | gap to best known (N only) |\n|---|---|---|---|---|---|---|\n'
                    + '\n'.join(rows) + '\n')


def build_synthetic(tree, names_=('ind', 'setup', 'window'), lead=None, seed=33):
    """Arrays of the policy, of the specialists and of the four rules and validator summaries of
    the named sets under <tree>/results; a shifted set copies the arrays inside the ranges on the
    combinations it
    does not reach (so the ind set is built first, or exists). Call with dry_paths(tree) in the
    environment."""
    seeds = check_setting()
    lead = lead or DRY_LEAD
    g = np.random.default_rng(seed)
    gs = np.random.default_rng(seed + 1)       # the specialists' noise, its own stream
    by = {name: (key, data) for name, key, data in SETS}
    n = n_test(by['ind'][1])
    for name in names_:
        key, data = by[name]
        if n_test(data) != n:
            raise Refused('the sets hold different numbers of test instances')
        res = REG.res_dir(key)
        os.makedirs(res, exist_ok=True)
        for r in R8:
            if name != 'ind' and r not in REACHED[name]:
                ind = by['ind'][0]
                for s in seeds:
                    shutil.copyfile(REG.result_path(JOINT, None, s, ind, r),
                                    REG.result_path(JOINT, None, s, key, r))
                    shutil.copyfile(REG.result_path('specialist', r, s, ind, r),
                                    REG.result_path('specialist', r, s, key, r))
                for q in RULES:
                    shutil.copyfile(f'{REG.res_dir(ind)}/Result_{q}_{r}.npy',
                                    f'{res}/Result_{q}_{r}.npy')
                continue
            best = g.uniform(150.0, 350.0, n) * (1.3 if name != 'ind' else 1.0)
            for q, extra in zip(RULES, (2.0, 1.0, 0.0, 3.0)):
                np.save(f'{res}/Result_{q}_{r}.npy',
                        np.stack([best * (1 + extra / 100.0), np.zeros(n)], axis=1))
            for s in seeds:
                pol = best * (1 + (lead[name][r] + g.normal(0, DRY_NOISE, n)) / 100.0)
                np.save(REG.result_path(JOINT, None, s, key, r), np.stack([pol, np.zeros(n)], axis=1))
                spec = pol / (1 + (DRY_SPEC[name] + gs.normal(0, DRY_NOISE, n)) / 100.0)
                np.save(REG.result_path('specialist', r, s, key, r),
                        np.stack([spec, np.zeros(n)], axis=1))
        write_summaries(key, seeds, n)


def write_family_record(seeds=None, boot=None):
    """A record of step family for the synthetic arrays (dry run and tests): rule_pool of p88
    computed by the same functions from the arrays of FAMILY_FOLDER, written where
    family_record() reads it, with p88's macro names FamRule<R>, Lo, Hi, Verdict."""
    seeds = list(seeds or check_setting())
    F88.configure(boot=boot if boot is not None else F88.SF.B_DEFAULT, alpha=ALPHA)
    rows = family_rows(differences(FAMILY_FOLDER, R8, seeds), R8)
    for x in rows.values():
        x.pop('holm_family')
        x.pop('reached')
    paths = family_paths()
    with open(paths['json'], 'w') as f:
        json.dump(F88.jsonable(dict(generator='synthetic record (scripts/p33_ood_table.py dry run)',
                                    setting=REG.describe(), seeds=seeds, boot=F88.B,
                                    alpha=F88.ALPHA, rule_pool=rows)), f, indent=1)
    L = []
    for r, x in rows.items():
        L.append(f'\\newcommand{{\\FamRule{r}}}{{{F88.f2(x["mean"])}}}'
                 f'\\newcommand{{\\FamRule{r}Lo}}{{{F88.f2(x["lo"])}}}'
                 f'\\newcommand{{\\FamRule{r}Hi}}{{{F88.f2(x["hi"])}}}'
                 f'\\newcommand{{\\FamRule{r}Verdict}}{{{x["verdict"]}}}')
    with open(paths['macros'], 'w') as f:
        f.write('% synthetic record of the dry run\n' + '\n'.join(L) + '\n')
    return rows


def dry_run(tree, boot=DRY_BOOT):
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with environment(**dry_paths(tree)):
        build_synthetic(tree)
        write_family_record(boot=boot)
        return run(dry=True, boot=boot)


def dry_ok(res):
    """The synthetic differences are recovered within 0.1 points, the reached combinations with a
    difference of at least one point are ahead and the one without is not; the pooled rows
    recover the mean of the reached combinations and lie below zero, and the pooled gap to the
    specialists is recovered and lies above zero."""
    ok = True
    for name, _, _ in SETS:
        for r in R8:
            want = DRY_LEAD[name][r] if r in REACHED[name] else DRY_LEAD['ind'][r]
            x = res['rows'][f'{name}|{r}']
            ok &= abs(x['mean'] - want) < 0.1
            ok &= x['verdict'] == (VERDICT_WORDS[0] if want <= -1.0 else VERDICT_WORDS[2])
    ok &= res['control']['floor'] == 0.0 and res['infeasible_total'] == 0
    ok &= res['family_check']['identical'] is True
    for name in SHIFTED:                 # the pooled row: the mean of the reached combinations
        x = res['pooled'][name]
        want = float(np.mean([DRY_LEAD[name][r] for r in REACHED[name]]))
        ok &= abs(x['mean'] - want) < 0.1 and x['direction'] == 'ahead'
        ok &= x['combos'] == REACHED[name] and x['c'] == len(REACHED[name])
    for name in SPEC_SETS:
        x = res['specialist'][name]
        ok &= abs(x['mean'] - DRY_SPEC[name]) < 0.1 and x['direction'] == 'behind'
    return bool(ok)


# ----------------------------------------------------------------- the earlier environment
def legacy_arms():
    """The folders of the earlier analysis: (name, results dir, data dir, summary-file suffix).
    eval_family.py appends --tag to the SUMMARY name but not to the .npy name, so the
    in-distribution rerun's validator tables are summary_<model>_cpuref.md while its arrays
    are Result_<model>_<regime>.npy."""
    return [(name, REG.res_dir(key), data, REG.summary_tag(key)) for name, key, data in SETS]


def legacy_main():
    """The analysis of the earlier submission, unchanged; PACT_ENV=legacy only.

    The in-distribution arm is read from 10x25_cpuref, NOT from the published
    10x25 results, and the reason is the whole methodology of this study.

    The three arms share the same 100 base instances. On a regime where the shifted
    constraint is inactive, their descriptors are byte-identical field for field, so
    the dispatching rules return bit-identical makespans, as they must. The published
    policy results do NOT: evaluated two days earlier on a different device and thread
    count, they differ from the shifted arms on 32 of 100 instances of regime N, by up
    to 8 hours, on inputs that are identical. Arg-max ties in the policy break the
    other way and the greedy rollout goes somewhere else.

    Comparing that column against the shifted arms would fold our own evaluation
    nondeterminism into the measured effect of the shift, and the "noise floor" it
    produced (0.02 points) was measuring the pipeline, not the benchmark. So the
    in-distribution arm is re-evaluated by the SAME command as the shifted arms
    (ind_cpuref.sh), and the placebo cells then come out bit-identical: their drift is
    exactly zero, by construction, and is asserted to be."""
    global rng, SEEDS
    rng = np.random.default_rng(7)          # the stream starts where the earlier script's did
    SEEDS = REG.seeds()
    ARMS = legacy_arms()

    sh = shift_magnitudes()
    print(f'SHIFT: setups {sh["setup_x"]:.2f}x and '
          f'{"ASYMMETRIC" if sh["asym"] else "symmetric"}; '
          f'outage density {sh["window_x"]:.2f}x\n')

    lead, infeas = {}, {}
    for name, res, _, sfx in ARMS:
        print(f'--- {name}: joint policy vs the per-instance best of {"/".join(RULES)} '
              f'(%, negative = policy ahead) ---')
        for r in REGIMES:
            p, ns = pooled(res, JOINT, r)
            b = best_rule(res, r)
            g = 100.0 * (p - b) / b                     # paired, per instance
            m, lo, hi = ci(g)
            lead[(name, r)] = (m, lo, hi)
            # From the validator, per seed, summed. Not from the makespan column.
            infeas[(name, r)] = sum(
                infeasible(res, REG.tag(JOINT, None, s), r, sfx)
                for s in SEEDS)
            print(f'  {r:4s}  {m:+6.2f}  [{lo:+6.2f},{hi:+6.2f}]  {verdict(lo, hi):13s}'
                  f'  seeds={ns}  infeas={infeas[(name, r)]}')
        print()

    # -----------------------------------------------------------------------
    # THE CONTROL, ASSERTED RATHER THAN HOPED FOR.
    #
    # On a regime where the shifted constraint is inactive, the three arms hold the same
    # instances and byte-identical descriptors. The policy is deterministic given its
    # input and the evaluation command, and all three arms were evaluated by the same
    # command. The makespans must therefore be BIT-IDENTICAL, and the placebo drift must
    # be exactly zero.
    #
    # If it is not, the arms were not evaluated the same way, the "effect" of the shift
    # then the difference between the arms is not the shift alone, and no number below
    # means what it says. So this is an assertion, not a diagnostic print: the table is
    # not written unless the control holds bit for bit.
    # -----------------------------------------------------------------------
    ACTIVE = {'setup': 'S', 'window': 'W'}     # the constraint bit each arm shifts

    for name, res, _, sfx in ARMS[1:]:
        for r in REGIMES:
            if ACTIVE[name] in r:
                continue                        # treated, not placebo
            a, _ = pooled(ARMS[0][1], JOINT, r)
            b, _ = pooled(res, JOINT, r)
            if not np.array_equal(a, b):
                n = int((a != b).sum())
                raise SystemExit(
                    f'CONTROL FAILED: on the {name} arm, regime {r} has no {ACTIVE[name]} '
                    f'active, so the shift cannot reach it and the policy must return the '
                    f'same schedule it returns in distribution. It does not: {n} of '
                    f'{len(a)} instances differ, by up to {np.abs(a - b).max():.1f} h.\n'
                    f'The two arms were not produced by the same evaluation. Any "effect" '
                    f'this script reports would be that difference, not the shift. '
                    f'Re-evaluate the in-distribution arm with ind_cpuref.sh and rerun.')
    print(f'CONTROL PASSED: on every regime the shift cannot reach, the policy returns '
          f'bit-identical schedules across arms.\n')

    # -----------------------------------------------------------------------
    # PLACEBO AND TREATED CELLS.
    #
    # The three arms hold the SAME instances. A regime in which the shifted constraint is
    # inactive cannot be reached by the shift at all: in the setup arm the setup bit is
    # off in N, L, W and LW, and in the window arm the window bit is off in N, L, S and
    # LS. Those are the placebo cells, their drift is zero by construction, and the
    # assertion above has already proved it bit for bit.
    #
    # The floor against which a treated cell is read is therefore not an estimate of
    # noise. It is zero. Every movement below is the shift and nothing else.
    # -----------------------------------------------------------------------
    placebo, treated = [], []
    for name in ('setup', 'window'):
        for r in REGIMES:
            drift = lead[(name, r)][0] - lead[('ind', r)][0]   # + = lead shrank
            bucket = treated if ACTIVE[name] in r else placebo
            bucket.append((name, r, lead[('ind', r)][0], lead[(name, r)][0], drift))

    floor = max(abs(d) for *_, d in placebo) if placebo else 0.0
    print(f'PLACEBO CELLS (the shift cannot reach them; drift must be zero)')
    for name, r, a, b_, d in placebo:
        print(f'  {name:7s} {r:4s}  {a:+.2f} -> {b_:+.2f}   drift {d:+.2f}')
    print(f'  largest placebo drift: {floor:.2f} points')
    if floor > 0.005:
        raise SystemExit(
            f'CONTROL FAILED: a placebo cell moved by {floor:.2f} points. The schedules '
            f'were asserted bit-identical, so a nonzero drift here means the RULE '
            f'baseline differs between arms on a regime the shift cannot reach. Stop.')

    print(f'\nTREATED CELLS (the shifted constraint is ACTIVE)')
    hurt, helped = [], []
    for name, r, a, b_, d in treated:
        tag = '  LEAD SHRANK' if d > 0 else ('  LEAD GREW' if d < 0 else '')
        (hurt if d > 0 else helped).append((name, r, a, b_, d))
        print(f'  {name:7s} {r:4s}  {a:+.2f} -> {b_:+.2f}   drift {d:+.2f}{tag}')

    # A cell counts as "the policy stopped being ahead" only if it WAS ahead in
    # distribution and is not ahead now. Testing only the shifted interval would also
    # catch a cell that was never ahead to begin with (S is one), and would report a
    # loss the shift did not cause.
    reversed_ = [t for t in treated if t[2] < 0 <= t[3]]
    tied = [t for t in treated
            if lead[('ind', t[1])][2] < 0 <= lead[(t[0], t[1])][2]]
    print(f'\nof the {len(treated)} treated cells, the lead shrank on {len(hurt)} and grew '
          f'on {len(helped)}')
    if tied:
        print('the policy is no longer statistically ahead of the dispatching rules on:')
        for name, r, a, b_, _ in tied:
            m, lo, hi = lead[(name, r)]
            print(f'  {name:7s} {r:4s}  {m:+.2f}%  CI [{lo:+.2f},{hi:+.2f}] '
                  f'(was {a:+.2f}% in distribution)')
    if reversed_:
        print('THE RULES OVERTOOK THE POLICY on:')
        for name, r, a, b_, _ in reversed_:
            print(f'  {name:7s} {r:4s}  {a:+.2f}% -> {b_:+.2f}%')
    else:
        print('the mean lead never reverses sign: the policy is not overtaken anywhere')

    n_infeas = sum(infeas.values())
    print(f'\ninfeasible schedules across every arm and regime: {n_infeas}')

    # -----------------------------------------------------------------------
    # One explanation for the descriptor's null result (E2, E3) is that this benchmark
    # makes the active regime easy to read off ordinary features, so the description tells
    # the network only what it can already see. That explanation is falsifiable: if it is
    # right, a shift which makes the regime harder to read should hurt the released
    # (token-blind) policy more than the conditioned variant. Both are evaluated on
    # every arm, so the explanation is tested rather than argued.
    #
    # The quantity is the conditioned variant's gap to the released policy,
    # 100*(conditioned - released)/released, per instance. Positive means the
    # released policy is ahead, which is what E2 found in distribution. If the gap
    # moves toward the conditioned variant as the parameters shift, the explanation
    # holds. (JOINT is the released policy and BLIND the conditioned variant here;
    # the variable names predate the arm swap and the macro names below keep their
    # documented meaning.)
    # -----------------------------------------------------------------------
    print('\n--- conditioned variant vs released policy, 100*(conditioned - released)/released (%) ---')
    print('    positive = the released policy is ahead (the E2 finding)')
    tok = {}
    have_blind = not REG.legacy_only('conditioned variant vs released policy')
    # both sides on the seeds the conditioned variant shares with the active set
    TOK_SEEDS = REG.common_seeds(BLIND) if have_blind else ()
    for name, res, _, sfx in (ARMS if have_blind else []):
        row = []
        for r in REGIMES:
            j, _ = pooled(res, JOINT, r, TOK_SEEDS)
            b, _ = pooled(res, BLIND, r, TOK_SEEDS)
            if j is None or b is None:
                have_blind = False
                continue
            g = 100.0 * (b - j) / j
            m, lo, hi = ci(g)
            tok[(name, r)] = (m, lo, hi)
            row.append(m)
        if row:
            print(f'  {name:7s} ' + ' '.join(f'{v:+6.2f}' for v in row)
                  + f'   mean {np.mean(row):+.2f}')
    if not have_blind:
        print('  token-blind results are not on disk for every arm yet; rerun after')
        print('  ood_blind.sh finishes. The macros below omit what is missing.')

    lines = [
        f'\\newcommand{{\\OodSetupX}}{{{sh["setup_x"]:.2f}}}'
        '  % mean setup magnitude, shifted arm / published arm',
        f'\\newcommand{{\\OodWindowX}}{{{sh["window_x"]:.2f}}}'
        '  % mean outage density, shifted arm / published arm',
        f'\\newcommand{{\\OodInfeas}}{{{n_infeas}}}'
        '  % infeasible schedules over both shifted arms and all 8 regimes',
        f'\\newcommand{{\\OodFloor}}{{{floor:.2f}}}'
        '  % largest drift on a PLACEBO cell: the resampling noise floor (pp)',
        f'\\newcommand{{\\OodPlacebo}}{{{len(placebo)}}}'
        f'\\newcommand{{\\OodTreated}}{{{len(treated)}}}'
        '  % cells where the shifted constraint is inactive / active',
        f'\\newcommand{{\\OodHurt}}{{{len(hurt)}}}'
        f'\\newcommand{{\\OodHelped}}{{{len(helped)}}}'
        '  % treated cells whose lead shrank / grew by more than the noise floor',
        f'\\newcommand{{\\OodTied}}{{{len(tied)}}}'
        '  % treated cells where the policy is no longer statistically ahead',
        f'\\newcommand{{\\OodReversed}}{{{len(reversed_)}}}'
        '  % treated cells where the rules actually overtook the policy',
    ]
    for name, _, _, _ in ARMS:
        for r in REGIMES:
            if (name, r) not in lead:
                continue
            m, lo, hi = lead[(name, r)]
            w = WORD[name]
            lines.append(
                f'\\newcommand{{\\Ood{w}{r}}}{{{m:+.2f}}}'
                f'\\newcommand{{\\Ood{w}{r}Lo}}{{{lo:+.2f}}}'
                f'\\newcommand{{\\Ood{w}{r}Hi}}{{{hi:+.2f}}}'
                f'  % {name} {r}: policy vs best rule (%), CI')

    if tok:
        for name, _, _, _ in ARMS:
            vals = [tok[(name, r)][0] for r in REGIMES if (name, r) in tok]
            if not vals:
                continue
            w = WORD[name]
            lines.append(
                f'\\newcommand{{\\Tok{w}Mean}}{{{np.mean(vals):+.2f}}}'
                f'  % {name}: mean over regimes of 100*(blind-joint)/joint; '
                f'positive = the released policy is ahead')
            for r in REGIMES:
                if (name, r) not in tok:
                    continue
                m, lo, hi = tok[(name, r)]
                lines.append(
                    f'\\newcommand{{\\Tok{w}{r}}}{{{m:+.2f}}}'
                    f'\\newcommand{{\\Tok{w}{r}Lo}}{{{lo:+.2f}}}'
                    f'\\newcommand{{\\Tok{w}{r}Hi}}{{{hi:+.2f}}}'
                    f'  % {name} {r}: conditioned minus released (%), CI')

    emit(REG.out('paper/macros_ood.tex'), lines,
         header=('% auto-generated by scripts/p33_ood_table.py -- do not hand-edit\n'
                 '% Policy vs the per-instance best of FIFO/MOR/SPT/MWKR, the same\n'
                 '% baseline in distribution and out of it.'))
    print('\nwrote paper/macros_ood.tex')


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description='the policy against the best dispatching rule '
                                             'outside the training ranges')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    a = ap.parse_args()
    os.chdir(ROOT)
    if REG.env() == 'legacy' and not a.check and a.dry is None:
        REG.banner()
        legacy_main()
        return
    try:
        if a.check:
            sys.exit(check())
        if a.dry is None:
            REG.banner()
            res, names_, md = run()
            print('\n'.join(md))
            print(f'\n{len(names_)} macros; wrote ' + ', '.join(outputs().values()))
            return
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p33_dry_')
        try:
            res, names_, md = dry_run(tree)
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    ok = dry_ok(res)
    if keep:
        print('\n'.join(md))
    print(f'p33 dry run on synthetic arrays {"passed" if ok else "FAILED"}: {len(names_)} macros, '
          f'{res["control"]["arrays"]} control arrays bit-identical; setup S '
          f'{res["rows"]["setup|S"]["mean"]:+.3f} (true {DRY_LEAD["setup"]["S"]:+.2f}), verdict '
          f'{res["rows"]["setup|S"]["verdict"]}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
