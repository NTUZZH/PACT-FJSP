r"""P141: the choice of the mask repeated on the validation instances, by the dispatching rules.

Evaluation only, CPU only, one thread. No network is loaded, no solver runs and nothing is
trained. Every schedule is a rollout of a classical dispatching rule in the environment of the
paper.

QUESTION
  Each benchmark uses the mask under which the best of four dispatching rules builds the
  shorter schedules on its test instances: the factory benchmark the non-delay mask, the
  public benchmark the active mask (scripts/p107_mask_rule_variants.py, record
  results/mask_rule_variants.json, macro prefix Mrul). This script repeats the same comparison
  on the validation instances of each benchmark, the instances that select the checkpoint
  during training and that no reported result uses, and states whether they lead to the same
  mask.

THE VALIDATION INSTANCES (rebuilt in memory as train_family.py builds them, no file written)
  factory  the configuration of the policy run xmask-v1-s304 (arm_registry: train_log and
           config_path of the released arm, exact environment, seed 304): min(vali_size, 20)
           base instances ppvc_instance_generator(n_modules=n_j, class_mix=ppvc_mix,
           station_counts=<ppvc_factory>, seed=seed_train_vali_datagen + i) and per
           combination R the descriptor make_descriptor(R, jl, pt, meta, n_m,
           seed=seed_train_vali_datagen * 1000 + i, shift=dr_shift or None).
  public   the configuration of the policy run pubx-act-s304 (arm_registry.pub_rule_paths,
           the released recipe under the active mask): the stored records of pub_vali_dir,
           the first min(vali_size, 20) names of sorted(desc/N); the base instance from the
           record of desc/N, the descriptor of combination R from the record of desc/R.
  The training run covers six combinations; the validation instances of SW and LSW are built
  by the same rule with the configuration of the specialist runs of SW and LSW of the same
  seed, whose validation settings must equal those of the policy run.

CHECK (before any rollout; a failure refuses)
  For every combination, the mean MWKR makespan over the validation instances under the default
  mask with the legacy options (FJSPEnvFamily(n_j, n_m, family_mode=True, first_mask='legacy',
  mask_ties='legacy'), np.random.seed(seed_test), common_utils.heuristic_select_action('MWKR')),
  printed with one decimal, equals the value of the line '[vali] MWKR norms: ...' of the
  training log of the run that covers the combination (the policy run for N, L, S, W, LS, LW,
  the specialist run for SW and LSW).

PER CASE (one validation instance, one combination) AND MASK X in nondelay, block, active
  The makespans of FIFO, MOR, SPT and MWKR restricted to the pairs X admits, from the rollouts
  of p107 (P107.rule_rollout, P107.make_env with the exact mask options, the validator call
  P107.check_schedule, one decision per operation), and their per-instance best (the shortest
  of the four). The built-in checks of p107: under nondelay the admissible set computed from
  the state equals the environment's mask at every decision, and without setups the block rule
  admits the nondelay pairs at every decision and gives the same four schedules.

OUTCOMES, per benchmark, per combination and pooled over the eight combinations
  diff       for X in active, block: 100 (best_X - best_nondelay) / best_nondelay per case,
             mean and 95% percentile bootstrap over instances (P107.boot, the function p107
             uses for the same quantity on the test instances: stats_family.instance_boot,
             20,000 resamples, a fresh default_rng(7) per quantity; pooled, a drawn instance
             brings its eight combinations). Negative means that X gives the shorter schedules.
  direction  'shorter' when the interval lies below zero, 'longer' when it lies above zero,
             'neither' otherwise, for the first-named mask against nondelay.
  counts     the number of combinations with each direction.
  choice     the mask with the shorter best-rule schedules on the validation instances, from the
             pooled direction of active against nondelay (shorter: active; longer: nondelay;
             neither: no mask), and 'same' when it is the mask the benchmark uses (the mask_mode
             of the policy run's configuration), 'different' otherwise.
  test       beside each value, the same quantity on the test instances read from the record of
             p107 (sets 10x25 and pub10x5, rows per combination and All, quantity best_rule),
             not recomputed; the record must be complete and must agree with
             manuscript/macros_maskrule.tex to the printed digits.

OUTPUTS (arm_registry.out under PACT_OUT)
  results/mask_choice_validation.json, results/mask_choice_validation.md
  manuscript/macros_mask_validation.tex   prefix Mvl, letters only, every line with a comment; no
                                         name defined in another macro file of paper/ or
                                         manuscript/ (the file itself and its copy in manuscript/
                                         are skipped)
  A missing input or a failed check refuses with the reason and removes the three outputs.

USAGE (repository root, CPU, one thread)
  PACT_ENV=exact PACT_OUT=out/exact python scripts/p141_mask_choice_validation.py
  PACT_ENV=exact PACT_OUT=out/exact python scripts/p141_mask_choice_validation.py --check
      (every input present and consistent; reads only, runs no rollout, writes nothing)
"""
import argparse
import glob
import json
import os
import platform
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
sys.path.insert(0, ROOT)
if SCRIPTS not in sys.path:
    sys.path.append(SCRIPTS)

import arm_registry as REG  # noqa: E402  (removes --pact-* options from sys.argv)

_ARGV = sys.argv[1:]          # p107 replaces sys.argv on import, so params reads a CPU device

import numpy as np  # noqa: E402

import p107_mask_rule_variants as P107  # noqa: E402

P101, P105 = P107.P101, P107.P105

GENERATOR = 'scripts/p141_mask_choice_validation.py'
OUT_FILES = {'macros': 'manuscript/macros_mask_validation.tex',
             'json': 'results/mask_choice_validation.json',
             'md': 'results/mask_choice_validation.md'}
PREFIX = 'Mvl'
R8 = tuple(P107.R8)
RULES = tuple(P107.RULES)
ADM = tuple(P107.ADM)                      # nondelay, block, active
ALT = ('active', 'block')                  # the first-named masks, against nondelay
ADM_WORD = {'active': 'Act', 'block': 'Blk'}
ADM_TEXT = {'nondelay': 'non-delay', 'active': 'active', 'block': 'block starts'}
DIRECTIONS = ('shorter', 'longer', 'neither')
DIR_WORD = {'shorter': 'Shorter', 'longer': 'Longer', 'neither': 'Neither'}
B = P107.B                                 # 20,000 resamples
CI_SEED = P107.CI_SEED                     # 7
VALI_CAP = 20                              # train_family.py: min(config.vali_size, 20)
RUN_SEED = 304
# the settings of a configuration that define its validation instances; every run whose log is
# checked must agree with the policy run on all of them
VALI_KEYS = ('family_base', 'n_j', 'n_m', 'vali_size', 'seed_train_vali_datagen', 'seed_test',
             'ppvc_mix', 'ppvc_factory', 'dr_shift', 'pub_vali_dir')
TEST_RECORD = P107.OUT_FILES['json']       # written in place by p107 infer
TEST_MACROS = P107.OUT_FILES['macros']
TEST_PREFIX = P107.PREFIX                  # Mrul
BENCH = {
    'factory': dict(word='Ten', test_set='10x25', family_base='ppvc', uses='nondelay',
                    text='factory benchmark, training size (10 modules, 25 stations)'),
    'public': dict(word='Pub', test_set='pub10x5', family_base='sd2', uses='active',
                   text='public benchmark, SD2 10x5'),
}
HELD = ('SW', 'LSW')                       # combinations outside the policy's training set
NORMS = re.compile(r'\[vali\] MWKR norms: ([^\n]*)')
PAIR = re.compile(r'\b([A-Z]+)=([\d.]+)')


class Refused(SystemExit):
    """A missing input or a failed check; the outputs are removed and nothing is written."""


def refuse(msg):
    raise Refused(f'REFUSING: {msg}')


def expect(cond, msg):
    if not cond:
        refuse(msg)


def rel(path):
    """A path relative to the repository when it lies inside it."""
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


# ---------------------------------------------------------------------------
# the runs whose configurations define the validation instances
# ---------------------------------------------------------------------------

def runs_of(bench):
    """[(run name, training log, configuration)] of a benchmark: the policy run first, then the
    specialist runs of SW and LSW of the same seed."""
    out = []
    if bench == 'factory':
        out.append((REG.model('released', None, RUN_SEED, 'exact'),
                    REG.train_log('released', None, RUN_SEED, 'exact'),
                    REG.config_path('released', None, RUN_SEED, 'exact')))
        for R in HELD:
            out.append((REG.model('specialist', R, RUN_SEED, 'exact'),
                        REG.train_log('specialist', R, RUN_SEED, 'exact'),
                        REG.config_path('specialist', R, RUN_SEED, 'exact')))
    else:
        _, lp, _, cp = REG.pub_rule_paths('released', None, RUN_SEED)
        out.append((REG.pub_rule_model('released', None, RUN_SEED), lp, cp))
        for R in HELD:
            _, lp, _, cp = REG.pub_rule_paths('specialist', R, RUN_SEED)
            out.append((REG.pub_rule_model('specialist', R, RUN_SEED), lp, cp))
    return out


def read_config(path, run):
    p = os.path.join(ROOT, path)
    expect(os.path.exists(p), f'missing input {path}: the configuration of the training run '
                              f'{run}, written by train_family.py when the run starts')
    try:
        with open(p) as f:
            return json.load(f)
    except ValueError as e:
        refuse(f'{path} is not valid JSON: {e}')


def logged_norms(path, run):
    """{combination: printed value} of the first '[vali] MWKR norms' line of a training log."""
    p = os.path.join(ROOT, path)
    expect(os.path.exists(p), f'missing input {path}: the training log of the run {run}')
    with open(p, errors='replace') as f:
        m = NORMS.search(f.read())
    expect(m is not None, f'{path} holds no line "[vali] MWKR norms: ..."')
    return dict(PAIR.findall(m.group(1)))


def regimes_of(cfg):
    return [r.strip().upper() for r in str(cfg['family_regimes']).split(',') if r.strip()]


def setting(bench):
    """The policy run's configuration, the run that covers each combination, the logged norms
    and the mask the benchmark uses; refuses on a missing file or a disagreeing setting."""
    spec = BENCH[bench]
    runs = runs_of(bench)
    cfgs = {name: read_config(cp, name) for name, _, cp in runs}
    logs = {name: logged_norms(lp, name) for name, lp, _ in runs}
    policy = runs[0][0]
    cfg = cfgs[policy]
    expect(cfg.get('family_base', 'ppvc') == spec['family_base'],
           f'{policy}: family_base {cfg.get("family_base")}, want {spec["family_base"]}')
    expect(cfg.get('mask_mode') == spec['uses'],
           f'{policy}: mask_mode {cfg.get("mask_mode")}; the {spec["text"]} uses the '
           f'{spec["uses"]} mask')
    for k, v in P101.MASK.items():
        expect(cfg.get(k) == v, f'{policy}: {k}={cfg.get(k)}, want {v} (exact environment)')
    cover = {}
    for name, lp, cp in runs:
        c = cfgs[name]
        diff = [k for k in VALI_KEYS if c.get(k) != cfg.get(k)]
        expect(not diff, f'{name}: the validation settings {diff} differ from {policy} '
                         f'({[(k, c.get(k), cfg.get(k)) for k in diff]})')
        regs = regimes_of(c)
        expect(sorted(logs[name]) == sorted(regs),
               f'{lp}: the MWKR norms name {sorted(logs[name])}, the configuration trains '
               f'{regs}')
        for r in regs:
            cover.setdefault(r, (name, lp))
    missing = [r for r in R8 if r not in cover]
    expect(not missing, f'no checked run covers the combinations {missing}')
    return dict(policy=policy, config=cfg, uses=cfg['mask_mode'], runs=runs,
                cover={r: cover[r] for r in R8},
                logged={r: logs[cover[r][0]][r] for r in R8},
                settings={k: cfg.get(k) for k in VALI_KEYS})


# ---------------------------------------------------------------------------
# the validation instances, as train_family.py builds them
# ---------------------------------------------------------------------------

def vali_size(cfg):
    return min(int(cfg['vali_size']), VALI_CAP)


def factory_validation(cfg, combos):
    """(names, base [(jl, pt, meta)], {combination: [descriptor]}) of the factory benchmark:
    the base instances with seeds seed_train_vali_datagen + i and the descriptors with seeds
    seed_train_vali_datagen * 1000 + i (train_family.FamilyTrainer.__init__)."""
    from constraint_family.descriptor import make_descriptor
    from ppvc_instance_generator import (DEFAULT_FACTORY, SMALL_FACTORY, TIGHT_FACTORY,
                                         build_factory, ppvc_instance_generator)
    factory = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY,
               'tight': TIGHT_FACTORY}[cfg['ppvc_factory']]
    mch_type, _ = build_factory(factory)
    n_m = len(mch_type)
    expect(n_m == int(cfg['n_m']), f'the factory {cfg["ppvc_factory"]} has {n_m} stations, the '
                                   f'configuration {cfg["n_m"]}')
    vseed = int(cfg['seed_train_vali_datagen'])
    shift = json.loads(cfg['dr_shift']) if cfg.get('dr_shift') else None
    n = vali_size(cfg)
    base = [ppvc_instance_generator(n_modules=int(cfg['n_j']), class_mix=cfg['ppvc_mix'],
                                    station_counts=factory, seed=vseed + i) for i in range(n)]
    descs = {R: [make_descriptor(R, jl, pt, meta, n_m, seed=vseed * 1000 + i, shift=shift)
                 for i, (jl, pt, meta) in enumerate(base)] for R in combos}
    names = [f'vali_seed{vseed + i}' for i in range(n)]
    return names, base, descs


def load_family_record(path):
    from constraint_family.public_rules import load_family_record as load
    return load(path, root=ROOT)


def public_validation(cfg, combos):
    """(names, base, {combination: [descriptor]}) of the public benchmark: the first
    min(vali_size, 20) names of sorted(<pub_vali_dir>/desc/N), the base instance from the
    record of N and the descriptor of each combination from its own record
    (train_family.FamilyTrainer.__init__)."""
    stored = cfg.get('pub_vali_dir') or ''
    expect(stored, 'the public policy run has no pub_vali_dir; its validation instances are '
                   'drawn, not stored')
    folder = os.path.join(ROOT, stored, 'desc')
    for R in ('N',) + tuple(combos):
        expect(os.path.isdir(os.path.join(folder, R)),
               f'missing input {stored}/desc/{R}: the stored validation records of the public '
               f'benchmark')
    names = sorted(os.listdir(os.path.join(folder, 'N')))[:vali_size(cfg)]
    expect(len(names) == vali_size(cfg), f'{stored}/desc/N holds {len(names)} records, want '
                                         f'{vali_size(cfg)}')
    base, bases = [], []
    for nm in names:
        jl, pt, _, rec = load_family_record(os.path.join(folder, 'N', nm))
        base.append((jl, pt, {'op_type': None, 'mch_type': None}))
        bases.append(rec['base_instance'])
    descs = {}
    for R in combos:
        descs[R] = []
        for nm, b in zip(names, bases):
            p = os.path.join(folder, R, nm)
            expect(os.path.exists(p), f'missing input {rel(p)}')
            _, _, d, rec = load_family_record(p)
            expect(rec.get('regime') == R and rec.get('base_instance') == b,
                   f'{rel(p)} names regime {rec.get("regime")} and base instance '
                   f'{rec.get("base_instance")}, want {R} and {b}')
            descs[R].append(d)
    return [nm.split('.')[0] for nm in names], base, descs


def validation_set(bench, cfg, combos=R8):
    if bench == 'factory':
        return factory_validation(cfg, combos)
    return public_validation(cfg, combos)


# ---------------------------------------------------------------------------
# the check against the training logs
# ---------------------------------------------------------------------------

def mwkr_makespan(jl, pt, desc, cfg):
    """The MWKR schedule of the normalization of train_family.py: the default mask with the
    legacy options, np.random.seed(seed_test)."""
    from common_utils import heuristic_select_action
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    henv = FJSPEnvFamily(int(cfg['n_j']), int(cfg['n_m']), family_mode=True,
                         first_mask='legacy', mask_ties='legacy')
    henv.set_initial_data([jl], [pt], descriptor_list=[desc])
    np.random.seed(int(cfg['seed_test']))
    while not henv.done().all():
        a = heuristic_select_action('MWKR', henv)
        henv.step(np.array([a]))
    return float(henv.current_makespan[0])


def norm_check(base, descs, cfg, logged, cover):
    """{combination: dict(rebuilt, printed, logged, run, log)}; refuses unless the rebuilt mean
    MWKR makespan of every combination prints as its logged value."""
    out, bad = {}, []
    for R in descs:
        v = float(np.mean([mwkr_makespan(jl, pt, d, cfg)
                           for (jl, pt, _), d in zip(base, descs[R])]))
        printed = f'{v:.1f}'
        out[R] = dict(rebuilt=v, printed=printed, logged=logged[R], run=cover[R][0],
                      log=cover[R][1], equal=printed == logged[R])
        if printed != logged[R]:
            bad.append(f'{R}: rebuilt {printed}, {cover[R][1]} prints {logged[R]}')
    expect(not bad, 'the rebuilt validation instances do not reproduce the MWKR norms of the '
                    'training logs (' + '; '.join(bad) + ')')
    return out


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

def best_of_four(ms):
    """Per-instance best of the four rule makespans: the minimum over the last axis."""
    return np.min(np.asarray(ms, dtype=float), axis=-1)


def evaluate_case(jl, pt, meta, desc, combo, where):
    """{mask: [makespans of FIFO, MOR, SPT, MWKR]} of one case from the rollouts of p107, with
    the validator, one decision per operation, and the built-in checks of p107."""
    setups = P107.has_setups(desc, combo)
    ms, sched, differ = {}, {}, {}
    for adm in ADM:
        ms[adm], sched[adm], differ[adm] = [], [], []
        for r in RULES:
            x = P107.rule_rollout(r, jl, pt, meta, desc, adm)
            P107.check_schedule(f'{where}: {r} under {adm}', jl, pt, desc, x['amch'],
                                x['op_ct'], x['makespan'], len(x['counts']))
            ms[adm].append(float(x['makespan']))
            sched[adm].append((x['amch'], x['op_ct']))
            differ[adm].append(int(x['differ']))
    expect(not any(differ['nondelay']),
           f'{where}: under nondelay the admissible set computed from the state differs from '
           f'the environment\'s mask ({dict(zip(RULES, differ["nondelay"]))})')
    if not setups:
        same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                   for a, b in zip(sched['block'], sched['nondelay']))
        expect(same and not any(differ['block']),
               f'{where}: without setups the block rule differs from nondelay (decisions with '
               f'another admissible set per rule {dict(zip(RULES, differ["block"]))})')
    return ms, setups


def rollouts(names, base, descs, bench, progress=None):
    """{mask: array [n, 8, 4]} of rule makespans in the order of R8 and RULES."""
    n = len(base)
    out = {adm: np.zeros((n, len(R8), len(RULES))) for adm in ADM}
    setups = {}
    for k, R in enumerate(R8):
        for i, ((jl, pt, meta), d) in enumerate(zip(base, descs[R])):
            ms, s = evaluate_case(jl, pt, meta, d, R, f'{bench}/{R}/{names[i]}')
            setups[R] = s
            for adm in ADM:
                out[adm][i, k] = ms[adm]
        if progress:
            progress(bench, R)
    return out, setups


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def direction(lo, hi):
    """The first-named mask against nondelay: 'shorter' when the interval lies below zero,
    'longer' when it lies above zero, 'neither' otherwise."""
    if hi < 0:
        return 'shorter'
    if lo > 0:
        return 'longer'
    return 'neither'


def chosen_mask(pooled_active_direction):
    """The mask with the shorter best-rule schedules, from the pooled direction of active
    against nondelay; None when neither is shorter."""
    return {'shorter': 'active', 'longer': 'nondelay', 'neither': None}[pooled_active_direction]


def choice_word(pooled_active_direction, uses):
    """'same' when the mask chosen by the pooled direction is the mask the benchmark uses,
    'different' otherwise (also when neither mask is shorter)."""
    return 'same' if chosen_mask(pooled_active_direction) == uses else 'different'


def counts_of(rows):
    """{direction: [combinations]} over the eight combinations."""
    return {w: [R for R in R8 if rows[R]['direction'] == w] for w in DIRECTIONS}


def comparison(best):
    """best {mask: [n, 8]}; per first-named mask, the rows per combination and pooled over the
    eight (an instance brings its eight combinations), each with its direction."""
    out = {}
    for x in ALT:
        d = P107.gap(best[x], best['nondelay'])                     # [n, 8]
        rows = {}
        for k, R in enumerate(R8):
            g = P107.boot(d[:, [k]])
            g['direction'] = direction(g['lo'], g['hi'])
            g['shorter_cases'] = int((best[x][:, k] < best['nondelay'][:, k] - P107.EPS).sum())
            g['longer_cases'] = int((best[x][:, k] > best['nondelay'][:, k] + P107.EPS).sum())
            rows[R] = g
        pooled = P107.boot(d)
        pooled['direction'] = direction(pooled['lo'], pooled['hi'])
        out[x] = dict(rows=rows, All=pooled, counts=counts_of(rows))
    return out


# ---------------------------------------------------------------------------
# the test-instance values, read from the record of p107
# ---------------------------------------------------------------------------

def test_macro_names(word):
    """The Mrul macros of the pooled best-rule difference of one set, per first-named mask."""
    return {x: {k: f'{TEST_PREFIX}Diff{ADM_WORD[x]}BestRule{"" if k == "mean" else k}{word}All'
                for k in ('mean', 'Lo', 'Hi')} for x in ALT}


def read_test_record():
    p = os.path.join(ROOT, TEST_RECORD)
    expect(os.path.exists(p), f'missing input {TEST_RECORD}: written by '
                              f'scripts/p107_mask_rule_variants.py infer (step mask_rules of '
                              f'scripts/run_analysis.sh)')
    try:
        with open(p) as f:
            rec = json.load(f)
    except ValueError as e:
        refuse(f'{TEST_RECORD} is not valid JSON: {e}')
    expect(rec.get('generator') == P107.GENERATOR, f'{TEST_RECORD} names the generator '
                                                   f'{rec.get("generator")}')
    expect(rec.get('partial') is False and rec.get('complete') is True,
           f'{TEST_RECORD} is partial or incomplete')
    expect(list(rec.get('combos', [])) == list(R8), f'{TEST_RECORD} covers the combinations '
                                                    f'{rec.get("combos")}')
    return rec


def read_test_macros():
    p = os.path.join(ROOT, TEST_MACROS)
    expect(os.path.exists(p), f'missing input {TEST_MACROS}: written by '
                              f'scripts/p107_mask_rule_variants.py infer')
    with open(p, errors='replace') as f:
        return dict(re.findall(r'\\newcommand\{\\(' + TEST_PREFIX + r'[A-Za-z]+)\}\{([^}]*)\}',
                               f.read()))


def test_values(rec, macros, bench, uses):
    """The test-instance rows of one benchmark from the record of p107, with directions, counts
    and the choice word; the pooled rows must print as the Mrul macros."""
    spec = BENCH[bench]
    S = rec['sets'].get(spec['test_set'])
    expect(S is not None, f'{TEST_RECORD} holds no set {spec["test_set"]}')
    out = {}
    names = test_macro_names(spec['word'])
    for x in ALT:
        rows = {}
        for R in R8:
            g = dict(S['rows'][R]['diff'][x]['best_rule'])
            g['direction'] = direction(g['lo'], g['hi'])
            rows[R] = g
        g = dict(S['rows']['All']['diff'][x]['best_rule'])
        g['direction'] = direction(g['lo'], g['hi'])
        for key, k in (('mean', 'mean'), ('lo', 'Lo'), ('hi', 'Hi')):
            nm = names[x][k]
            expect(nm in macros, f'{TEST_MACROS} defines no \\{nm}')
            expect(macros[nm] == P101.f2(g[key]),
                   f'{TEST_RECORD} gives {P101.f2(g[key])} for \\{nm}, {TEST_MACROS} {macros[nm]}')
        out[x] = dict(rows=rows, All=g, counts=counts_of(rows))
    n = int(S['rows']['All']['n'])
    word = choice_word(out['active']['All']['direction'], uses)
    return dict(set=spec['test_set'], n=n, comparison=out, choice=word,
                chosen=chosen_mask(out['active']['All']['direction']),
                macro_names=names)


# ---------------------------------------------------------------------------
# macros, Markdown
# ---------------------------------------------------------------------------

f2 = P101.f2
u1 = P101.u1


def macro_lines(res):
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         '% The best of four dispatching rules (the shortest of the FIFO, MOR, SPT and MWKR '
         'schedules) under the active mask and under block starts against the non-delay mask, '
         'on the validation instances of each benchmark, beside the same quantity on its test '
         f'instances (the {TEST_PREFIX} macros of {TEST_MACROS}).',
         '% Differences in %: 100 (x_X - x_nondelay) / x_nondelay per case, mean over the eight '
         'combinations, 95% bootstrap over instances (an instance brings its combinations); '
         'negative means mask X gives the shorter schedules.',
         f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}  % bootstrap resamples']
    for bench, R in res['benchmarks'].items():
        w, T = BENCH[bench]['word'], R['test']
        L.append(f'\\newcommand{{\\{PREFIX}N{w}}}{{{R["n"]}}}'
                 f'\\newcommand{{\\{PREFIX}TestN{w}}}{{{T["n"]}}}'
                 f'  % {BENCH[bench]["text"]}: validation instances, and test instances of the '
                 f'record of p107')
        for x in ALT:
            X = ADM_WORD[x]
            for tag, C in (('', R['comparison'][x]), ('Test', T['comparison'][x])):
                g = C['All']
                where = 'validation' if not tag else 'test'
                L.append(f'\\newcommand{{\\{PREFIX}{tag}Diff{X}{w}}}{{{f2(g["mean"])}}}'
                         f'\\newcommand{{\\{PREFIX}{tag}Diff{X}Lo{w}}}{{{f2(g["lo"])}}}'
                         f'\\newcommand{{\\{PREFIX}{tag}Diff{X}Hi{w}}}{{{f2(g["hi"])}}}'
                         f'\\newcommand{{\\{PREFIX}{tag}Dir{X}{w}}}{{{g["direction"]}}}'
                         f'  % {bench}, {where} instances: best rule under {x} against '
                         f'nondelay pooled over the eight combinations, %, mean and 95% interval '
                         f'([{g["lo"]:+.4f}, {g["hi"]:+.4f}]), direction of {x}')
                L.append(''.join(f'\\newcommand{{\\{PREFIX}{tag}{DIR_WORD[d]}{X}{w}}}'
                                 f'{{{len(C["counts"][d])}}}' for d in DIRECTIONS)
                         + f'  % {bench}, {where} instances: combinations in which {x} gives '
                           f'shorter, longer and neither best-rule schedules than nondelay ('
                         + '; '.join(f'{d}: {",".join(C["counts"][d]) or "none"}'
                                     for d in DIRECTIONS) + ')')
        L.append(f'\\newcommand{{\\{PREFIX}Choice{w}}}{{{R["choice"]}}}'
                 f'\\newcommand{{\\{PREFIX}TestChoice{w}}}{{{T["choice"]}}}'
                 f'  % {bench}: whether the mask with the shorter best-rule schedules (pooled '
                 f'direction of active against nondelay) is the mask the benchmark uses '
                 f'({R["uses"]}), on the validation and on the test instances')
    return L


def macro_names(lines):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))


def collision_check(names, out_mac):
    """Every name is the prefix plus letters and unique, and no other macro file of paper/ or
    manuscript/ defines it; the output file and its copy in manuscript/ are skipped."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    expect(not bad, f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    dup = sorted({x for x in names if names.count(x) > 1})
    expect(not dup, f'duplicate macro names: {dup[:5]}')
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    files = sorted(set(glob.glob(os.path.join(ROOT, 'manuscript', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT, 'paper', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT, 'manuscript', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT, 'manuscript', 'sections_new', '**', '*.tex'),
                                   recursive=True)
                       + glob.glob(os.path.join(ROOT, 'manuscript', 'supp_new', '**', '*.tex'),
                                   recursive=True)))
    own = {os.path.abspath(out_mac),
           os.path.abspath(os.path.join(ROOT, 'manuscript', os.path.basename(out_mac)))}
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        expect(not clash, f'macro name collision with {rel(f)}: {sorted(clash)[:5]}')
    return len(files)


def ci(g):
    return f'{f2(g["mean"])} [{f2(g["lo"])}, {f2(g["hi"])}]'


def markdown(res):
    L = [f'# The choice of the mask repeated on the validation instances ({GENERATOR})', '',
         'Best of four dispatching rules (per instance the shortest of the FIFO, MOR, SPT and MWKR '
         'schedules) under each mask, from the rollouts of scripts/p107_mask_rule_variants.py. '
         f'Difference: {res["diff"]}. Interval: {res["bootstrap"]}. Direction: {res["direction"]}. '
         f'Choice: {res["choice"]}. Test-instance values read from {TEST_RECORD}.', '']
    for bench, R in res['benchmarks'].items():
        T = R['test']
        L += [f'## {bench}: {BENCH[bench]["text"]}', '',
              f'Validation instances: {R["n"]} ({R["instances"]}); test instances: {T["n"]}. '
              f'The benchmark uses the {R["uses"]} mask (mask_mode of {R["policy"]}). '
              f'Validation: {R["choice"]} (chosen: {R["chosen"]}); test: {T["choice"]} (chosen: '
              f'{T["chosen"]}).', '']
        for x in ALT:
            V, W = R['comparison'][x], T['comparison'][x]
            L += [f'{x} against nondelay, best rule, difference (%) and direction:', '',
                  '| row | validation | direction | cases shorter / longer | test | direction |',
                  '|---|---|---|---|---|---|']
            for c in R8:
                v, t = V['rows'][c], W['rows'][c]
                L.append(f'| {c} | {ci(v)} | {v["direction"]} | {v["shorter_cases"]} / '
                         f'{v["longer_cases"]} | {ci(t)} | {t["direction"]} |')
            L.append(f'| pooled over the eight | {ci(V["All"])} | {V["All"]["direction"]} | | '
                     f'{ci(W["All"])} | {W["All"]["direction"]} |')
            L += ['', 'Combinations per direction: validation ' + '; '.join(
                f'{d} {len(V["counts"][d])}' for d in DIRECTIONS) + '; test ' + '; '.join(
                f'{d} {len(W["counts"][d])}' for d in DIRECTIONS) + '.', '']
        L += ['Mean best-rule makespan (h) on the validation instances, nondelay / block / active:',
              '', '| row | ' + ' | '.join(R8) + ' |', '|---|' + '---|' * len(R8),
              '| best rule | ' + ' | '.join(' / '.join(u1(R['mean_best'][a][c]) for a in ADM)
                                           for c in R8) + ' |', '']
        L += ['Check: mean MWKR makespan on the rebuilt validation instances (default mask, legacy '
              'options) against the line "[vali] MWKR norms" of the training log:', '',
              '| combination | rebuilt | log | run |', '|---|---|---|---|']
        for c, k in R['norm_check'].items():
            L.append(f'| {c} | {k["printed"]} | {k["logged"]} | {k["run"]} |')
        L.append('')
    return L


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def check_setting():
    expect(REG.env() == 'exact', f'PACT_ENV={REG.env()}; this comparison runs in the exact-mask '
                                 f'environment (PACT_ENV=exact)')
    expect(os.environ.get('PACT_OUT') is not None,
           'set PACT_OUT (the output root of the chain, e.g. out/exact)')
    P101.check_mask_options()


def benchmark(bench, rec, macros, progress=None):
    """Every outcome of one benchmark."""
    st = setting(bench)
    cfg = st['config']
    names, base, descs = validation_set(bench, cfg)
    norm = norm_check(base, descs, cfg, st['logged'], st['cover'])
    if progress:
        progress(bench, 'MWKR check passed')
    ms, setups = rollouts(names, base, descs, bench, progress)
    best = {adm: best_of_four(ms[adm]) for adm in ADM}          # [n, 8]
    comp = comparison(best)
    word = choice_word(comp['active']['All']['direction'], st['uses'])
    test = test_values(rec, macros, bench, st['uses'])
    return dict(
        text=BENCH[bench]['text'], policy=st['policy'], uses=st['uses'],
        runs=[dict(run=r, log=lp, config=cp) for r, lp, cp in st['runs']],
        settings=st['settings'], n=len(names), instances=f'{names[0]} to {names[-1]}',
        names=names, setups={c: bool(s) for c, s in setups.items()}, norm_check=norm,
        comparison=comp, choice=word, chosen=chosen_mask(comp['active']['All']['direction']),
        mean_best={a: {c: float(best[a][:, k].mean()) for k, c in enumerate(R8)} for a in ADM},
        rule_makespans={a: ms[a] for a in ADM}, best_rule={a: best[a] for a in ADM},
        test=test)


def analyse(progress=None):
    rec = read_test_record()
    macros = read_test_macros()
    t0, c0 = time.perf_counter(), time.process_time()
    benches = {b: benchmark(b, rec, macros, progress) for b in BENCH}
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    return dict(
        generator=GENERATOR, environment='exact', mask_options=P101.MASK,
        masks={a: P107.ADM_TEXT[a] for a in ADM}, rules=list(RULES), combos=list(R8),
        diff='100 (best_X - best_nondelay) / best_nondelay per case; negative means mask X gives '
             'the shorter schedules',
        bootstrap=f'P107.boot: stats_family.instance_boot, {B} resamples, fresh '
                  f'default_rng({CI_SEED}) per quantity; pooled, a drawn instance brings its '
                  f'eight combinations',
        direction='shorter when the interval lies below zero, longer when it lies above zero, '
                  'neither otherwise (the first-named mask against nondelay)',
        choice='the mask with the shorter best-rule schedules from the pooled direction of active '
               'against nondelay (shorter: active, longer: nondelay, neither: none); same when it '
               'is the mask the benchmark uses (mask_mode of the policy run), different otherwise',
        test_record=TEST_RECORD, test_macros=TEST_MACROS,
        test_macro_family=f'{TEST_PREFIX} ({TEST_MACROS}): {TEST_PREFIX}Diff<Act|Blk>BestRule'
                          f'[Lo|Hi]<Ten|Pub>All for the pooled rows, <combination> in place of '
                          f'All per combination',
        rollouts='P107.rule_rollout (common_utils.masked_rule_action after np.random.seed(50)) '
                 'with P107.make_env (exact mask options), P107.check_schedule (validator of '
                 'p101) after every rollout',
        benchmarks=benches,
        timing=dict(wall_s=wall, cpu_s=cpu, cpu_share=cpu / max(wall, 1e-9),
                    affinity=sorted(os.sched_getaffinity(0)),
                    python=platform.python_version(), omp=os.environ.get('OMP_NUM_THREADS'),
                    finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                    note='planning figures only; the rollouts are deterministic'))


def write(res, outs):
    lines = macro_lines(res)
    names = macro_names(lines)
    res['macro_files_checked'] = collision_check(names, outs['macros'])
    res['macro_count'] = len(names)
    from macro_io import emit
    emit(outs['macros'], lines)
    P101.write_json_atomic(outs['json'], P101.jsonable(res))
    with open(outs['md'], 'w') as f:
        f.write('\n'.join(markdown(res)) + '\n')
    return names


def run(progress=None):
    check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        P101._cpu_torch()
        res = analyse(progress)
        names = write(res, outs)
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, outs


def check_inputs():
    """Every input present and consistent; reads only. Returns the problems."""
    try:
        check_setting()
        rec = read_test_record()
        macros = read_test_macros()
        for b in BENCH:
            st = setting(b)
            if b == 'public':
                cfg = st['config']
                folder = os.path.join(ROOT, cfg.get('pub_vali_dir') or '', 'desc')
                for R in R8:
                    d = os.path.join(folder, R)
                    expect(os.path.isdir(d) and len(os.listdir(d)) >= vali_size(cfg),
                           f'missing input {rel(d)} with {vali_size(cfg)} records')
            test_values(rec, macros, b, st['uses'])
    except SystemExit as e:
        return [str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__]
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='report whether every input is present and consistent; reads only')
    a = ap.parse_args(_ARGV if argv is None else argv)
    os.chdir(ROOT)
    if a.check:
        problems = check_inputs()
        for p in problems:
            print(f'p141: {p}')
        if problems:
            sys.exit(1)
        print('p141: every input present (training logs and configurations of the policy and '
              'specialist runs of seed 304 of both benchmarks, the stored public validation '
              f'records, {TEST_RECORD} and {TEST_MACROS} in agreement)')
        return
    REG.banner()
    t0 = time.perf_counter()

    def progress(bench, what):
        print(f'[p141] {bench} {what}  elapsed {time.perf_counter() - t0:6.0f} s', flush=True)
    try:
        res, names, outs = run(progress)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    print('\n'.join(markdown(res)))
    print(f'{len(names)} macros; wrote ' + ', '.join(outs[k] for k in ('macros', 'json', 'md')))


if __name__ == '__main__':
    main()
