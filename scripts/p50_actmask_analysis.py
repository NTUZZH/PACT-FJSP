r"""The non-delay restriction: the policy trained with the active-schedule mask against
the policy -> macros, a table and (exact-mask environment) a record.

EARLIER ENVIRONMENT (PACT_ENV=legacy, the default setting)
  Two analyses in one generated artifact:

    audit   p46 (results/idle_quant_cells.json): how often warm-started CP-SAT
            schedules dispatch outside the non-delay space, and how much of
            that lies beyond even the active-schedule bound at the decision
            it was made.
    test    the jointact arm (active-mask joint policy, 3 seeds) against
            jointblind (non-delay, 3 seeds), symmetric seed pooling, paired
            per-instance bootstrap CIs -- the same inference protocol as E1.

  Writes paper/macros_actmask.tex and paper/tables/actmask_table.tex.

EXACT-MASK ENVIRONMENT (PACT_ENV=exact)
  The arm xmask-active (arm_registry.SECONDARY_ARMS: the released recipe with
  --mask_mode active, the Giffler-Thompson condition for active schedules, trained and
  evaluated with --first_mask exact --mask_ties exact) against the policy (the released
  recipe, xmask-v1-s<seed>) on the seeds both have: the active seeds that are also
  arm_registry.SECONDARY_SEEDS (301-303). The folders and combinations are those the
  evaluation plan gives the arm (eval_newseeds_plan.SECONDARY_EVAL); the training size
  10x25 must be one of them, and every further size the plan names is reported the same
  way (macro word SIZE_WORD).

  SIGN AND PAIRING. d = 100 (a - b) / b per training seed and test instance, a the
  active-mask arm, b the policy: the difference in percent of the policy's makespan,
  positive when the active-mask arm has the longer schedule. Seed s of the arm meets
  seed s of the policy, and instance i is base instance i in every combination (checked
  on the test descriptors).

  INFERENCE (scripts/stats_family.py, with the conventions scripts/p95_secondary_inference.py
  applies to the supporting arms against the policy). k >= 2 seed pairs: crossed
  hierarchical bootstrap over seeds and base instances (hier_boot, B = --boot, a fresh
  default_rng(11) per quantity), with the seed-level Student t interval (seed_t) in the
  record; k == 1: bootstrap over base instances (instance_boot, a fresh default_rng(7)).
  Per combination, and pooled over the combinations of a size, where a drawn base
  instance brings all its combinations. The counts give the combinations whose 95%
  interval lies above zero (the arm longer) and below zero (the arm shorter).

  The audit of the earlier environment (results/idle_quant_cells.json, records of the
  earlier mask) is not part of this branch and writes no macro;
  scripts/p105_reference_replay.py carries that question in the exact environment.

  CHECKS. Every array exists, has one row per test instance and finite positive
  makespans, and is not older than the checkpoint of its run. Every run has its
  checkpoint, a training log with the completion line, a configuration that records its
  seed and the exact mask options, and an evaluation summary that records the exact mask.
  The configuration of the arm records mask_mode active, that of the policy the
  non-delay mask, and the two record the same updates, environments per update and
  training combinations. When no array of the arm exists, nothing is computed and the
  outputs of an earlier run are removed (skipped; --require all refuses instead and
  names the files). Any other failed check refuses: the outputs are removed, every
  problem is named, and the exit status is 1.

  OUTPUTS, through arm_registry.out (under PACT_OUT); every path in them is relative:
    manuscript/macros_actmask.tex         one comment per macro; the names of the
                                         quantities the earlier file also holds are kept
    manuscript/tables/actmask_table.tex   the table source
    results/active_mask_arm.json         every number, unrounded, and the files read
    results/active_mask_arm.md
  The macro names start with Act and are checked against every other macro file and
  staged section of paper/ and manuscript/ (the files named macros_actmask.tex hold the
  same quantities of the earlier environment under the same names and are skipped).

Usage:
  python scripts/p50_actmask_analysis.py                    (earlier environment)
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=<folder> \
      python scripts/p50_actmask_analysis.py [--require all] [--boot 20000] [--root <repo>]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm_registry as REG  # noqa: E402

BASE = REG.fixed_dir('10x25')
REGS = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
# The active-mask arm exists on the stored legacy seeds only; the released policy
# is paired with it on the seeds they share (arm_registry).
if REG.env() == 'legacy':
    _CS = REG.common_seeds('jointact')
    ACT = [REG.other_model('jointact', s) for s in _CS]
    BLD = [REG.model('released', None, s) for s in _CS]


def pool(models, r):
    return np.mean([np.load(f'{BASE}/Result_greedy+10x25+family+{m}_{r}.npy')[:, 0]
                    for m in models], axis=0)


def legacy_main():
    REG.banner()
    if REG.legacy_only('the active-mask arm (jointact)'):
        return
    with open('results/idle_quant_cells.json') as f:
        cells = json.load(f)

    rng = np.random.default_rng(11)
    mac = ['% auto-generated by scripts/p50_actmask_analysis.py -- do not hand-edit']
    rows = []
    worse = 0
    gaps = {}
    for r in REGS:
        a, b = pool(ACT, r), pool(BLD, r)
        ga = np.load(f'{BASE}/Result_GA_{r}.npy')[:, 0]
        rel = 100 * (a - b) / b
        boots = [rng.choice(rel, len(rel)).mean() for _ in range(5000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        if lo > 0:
            worse += 1
        gaps[r] = (rel.mean(), lo, hi)
        R = cells[r]
        frac = np.mean([100.0 * (c['active_only'] + c['beyond_active'])
                        / c['n_dispatch'] for c in R])
        delayed = sum(c['active_only'] + c['beyond_active'] for c in R)
        beyond = 100.0 * sum(c['beyond_active'] for c in R) / max(delayed, 1)
        va, vb = 100 * ((a - ga) / ga).mean(), 100 * ((b - ga) / ga).mean()
        rows.append((r, frac, beyond, rel.mean(), lo, hi))
        if r == 'S':
            mac += [f'\\newcommand{{\\ActSGaAct}}{{{va:+.2f}}}',
                    f'\\newcommand{{\\ActSGaBlind}}{{{vb:+.2f}}}']

    idle_lo = min(x[1] for x in rows)
    idle_hi = max(x[1] for x in rows)
    beyond_lo = min(x[2] for x in rows)
    beyond_hi = max(x[2] for x in rows)
    gap_lo = min(x[3] for x in rows)
    gap_hi = max(x[3] for x in rows)
    idle_n = next(x[1] for x in rows if x[0] == 'N')
    mac += [
        f'\\newcommand{{\\ActWorseN}}{{{worse}}}',
        f'\\newcommand{{\\IdleLo}}{{{idle_lo:.0f}}}',
        f'\\newcommand{{\\IdleHi}}{{{idle_hi:.0f}}}',
        f'\\newcommand{{\\IdleN}}{{{idle_n:.0f}}}',
        f'\\newcommand{{\\BeyondLo}}{{{beyond_lo:.0f}}}',
        f'\\newcommand{{\\BeyondHi}}{{{beyond_hi:.0f}}}',
        f'\\newcommand{{\\ActGapLo}}{{{gap_lo:.2f}}}',
        f'\\newcommand{{\\ActGapHi}}{{{gap_hi:.2f}}}',
        f'\\newcommand{{\\ActSDelta}}{{{gaps["S"][0]:+.2f}}}',
        f'\\newcommand{{\\ActSLo}}{{{gaps["S"][1]:+.2f}}}',
        f'\\newcommand{{\\ActSHi}}{{{gaps["S"][2]:+.2f}}}',
        f'\\newcommand{{\\ActNDelta}}{{{gaps["N"][0]:+.2f}}}',
    ]
    with open(REG.out('paper/macros_actmask.tex'), 'w') as f:
        f.write('\n'.join(mac) + '\n')

    lines = [
        '% auto-generated by scripts/p50_actmask_analysis.py -- do not hand-edit',
        '\\begin{tabular}{lrrrc}',
        '\\toprule',
        '& \\multicolumn{2}{c}{reference-schedule audit}'
        ' & \\multicolumn{2}{c}{active-mask policy vs.\\ non-delay} \\\\',
        '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}',
        'regime & \\%~dispatches delayed & \\%~of those beyond active'
        ' & gap (\\%) & 95\\%~CI \\\\',
        '\\midrule',
    ]
    for r, frac, beyond, g, lo, hi in rows:
        lines.append(f'${r}$ & {frac:.0f} & {beyond:.0f} & '
                     f'${g:+.2f}$ & $[{lo:+.2f},\\,{hi:+.2f}]$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    os.makedirs('paper/tables', exist_ok=True)
    with open(REG.out('paper/tables/actmask_table.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n'.join(mac[1:]))
    print(f'-> paper/macros_actmask.tex, paper/tables/actmask_table.tex '
          f'({worse}/8 regimes worse with CI clear of zero)')


# ================================================================== exact-mask environment
ARM = 'xmask-active'
FOLDER = '10x25'                 # the training size: the plan must evaluate the arm there
HIER_SEED, INST_SEED, ALPHA = 11, 7, 0.05
GENERATOR = 'scripts/p50_actmask_analysis.py'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FILES = ('manuscript/macros_actmask.tex', 'manuscript/tables/actmask_table.tex',
             'results/active_mask_arm.json', 'results/active_mask_arm.md')
MACRO_PREFIX = 'Act'
# the files of the earlier environment and their copy in manuscript/ define the same
# quantities under the same names; the collision check skips files with this name
SAME_QUANTITIES = 'macros_actmask.tex'
# macro word and table label of every further folder the plan may evaluate the arm on
SIZE_WORD = {'20x25': 'Twenty', '30x25': 'Thirty', '50x15': 'FiftyFifteen',
             '50x25': 'Fifty', '80x15': 'EightyFifteen', '80x25': 'Eighty',
             '10x25_oodsetup': 'OodSetup', '10x25_oodwindow': 'OodWindow'}
SIZE_TEXT = {'10x25': '10 modules', '20x25': '20 modules', '30x25': '30 modules',
             '50x15': '50 modules, 15 stations', '50x25': '50 modules',
             '80x15': '80 modules, 15 stations', '80x25': '80 modules',
             '10x25_oodsetup': '10 modules, setups outside the training range',
             '10x25_oodwindow': '10 modules, windows outside the training range'}
BUDGET_KEYS = ('max_updates', 'num_envs', 'family_regimes')
SUMMARY_MASK = 'mask: first_mask=exact mask_ties=exact'


class Refusal(SystemExit):
    """A check failed; the outputs are removed and nothing is certified."""


def refuse(msg):
    raise Refusal(f'REFUSING: {msg}')


def _template(path, name):
    if path.count(name) != 1:
        refuse(f'cannot derive a path layout from {path!r} and {name!r}')
    return path.replace(name, '{m}')


class Run:
    """One trained run: name (the model is arm_registry.PREFIX + name), training seed and
    the campaign folder its arrays are read from. Checkpoint, log and configuration
    follow the registry's layout of the principal arms."""

    def __init__(self, name, seed, folder):
        anchor = REG.model('released', None, REG.SECONDARY_SEEDS[0], 'exact')
        self.name, self.seed, self.folder = name, int(seed), folder
        s0 = REG.SECONDARY_SEEDS[0]
        self.ckpt = _template(REG.checkpoint('released', None, s0, 'exact'), anchor).format(m=name)
        self.log = _template(REG.train_log('released', None, s0, 'exact'), anchor).format(m=name)
        self.config = _template(REG.config_path('released', None, s0, 'exact'), anchor).format(m=name)
        self.dir = REG.res_dir(folder)

    def array(self, combo):
        return f'{self.dir}/Result_greedy+{REG.PREFIX}{self.name}_{combo}.npy'

    def summary(self):
        return f'{self.dir}/summary_{REG.PREFIX}{self.name}{REG.summary_tag(self.folder)}.md'


def plan_sets():
    """[(folder, combinations)] of the arm in the evaluation plan, the training size first."""
    import eval_newseeds_plan as PLAN
    sets = [(f, list(regs)) for f, regs in PLAN.SECONDARY_EVAL[ARM]]
    folders = [f for f, _ in sets]
    if FOLDER not in folders or len(set(folders)) != len(folders):
        refuse(f'the plan must evaluate {ARM} once at {FOLDER}; it names {folders}')
    for f, regs in sets:
        if f != FOLDER and f not in SIZE_WORD:
            refuse(f'the plan evaluates {ARM} at {f}, which has no macro word (SIZE_WORD)')
        if not regs or len(set(regs)) != len(regs) or not set(regs) <= set(REG.COMBOS):
            refuse(f'the plan names the combinations {regs} for {ARM} at {f}')
    return sorted(sets, key=lambda x: x[0] != FOLDER)


def check_layout():
    """The arm is a registered secondary arm whose checkpoint and log follow the layout
    this script derives, and the policy's arrays follow the registry."""
    if ARM not in REG.SECONDARY_ARMS:
        refuse(f'{ARM} is not in arm_registry.SECONDARY_ARMS')
    for s in REG.SECONDARY_SEEDS:
        r = Run(REG.secondary_model(ARM, s), s, FOLDER)
        if (r.ckpt, r.log) != REG.secondary_paths(ARM, s):
            refuse(f'the checkpoint and log layout of {r.name} differs from the registry')
    for s in REG.PLANNED_SEEDS['exact']:
        p = Run(REG.model('released', None, s, 'exact'), s, FOLDER)
        if (p.ckpt, p.log, p.config) != (REG.checkpoint('released', None, s, 'exact'),
                                         REG.train_log('released', None, s, 'exact'),
                                         REG.config_path('released', None, s, 'exact')):
            refuse(f'the path layout of {p.name} differs from the registry')
        if p.array('N') != REG.result_path('released', None, s, FOLDER, 'N'):
            refuse(f'the array layout of {p.name} differs from the registry')


def compared_seeds():
    """The active seeds the arm also has, in increasing order."""
    s = tuple(x for x in REG.seeds() if x in REG.SECONDARY_SEEDS)
    if not s:
        refuse(f'none of the active seeds {REG.seeds()} is a seed of {ARM} '
               f'{REG.SECONDARY_SEEDS}')
    return s


def runs_of(seeds, folder):
    arm = {s: Run(REG.secondary_model(ARM, s), s, folder) for s in seeds}
    pol = {s: Run(REG.model('released', None, s, 'exact'), s, folder) for s in seeds}
    return arm, pol


def test_instances(root, folder):
    """Number of test instances of a folder's data set; every combination must have the
    same file names and base instances, which are distinct."""
    import eval_newseeds_plan as PLAN
    data = PLAN.FACTORY[folder][0]
    names, bases = {}, {}
    for r in REG.COMBOS:
        ps = sorted(glob.glob(os.path.join(root, data, 'desc', r, '*.desc.json')))
        names[r] = [os.path.basename(p) for p in ps]
        bases[r] = []
        for p in ps:
            with open(p) as f:
                bases[r].append(json.load(f)['base_instance'])
    for r in REG.COMBOS:
        if names[r] != names['N'] or bases[r] != bases['N']:
            refuse(f'{data}/desc/{r} does not share file names and base instances with N; '
                   f'the base-instance clusters are not defined')
    if not names['N'] or len(set(bases['N'])) != len(bases['N']):
        refuse(f'{data}: no test instances, or base instances that repeat')
    return len(names['N'])


def _get(cfg, key):
    return cfg.get(key, '<absent>')


def run_problems(run, combos, root, n, cache, kind):
    """(missing files, other problems) of one run for the combinations read; arrays that
    pass go to cache[(name, folder, combo)] as float makespans. kind 'arm' or 'policy'."""
    miss, bad = [], []
    ck = os.path.join(root, run.ckpt)
    if not os.path.exists(ck):
        miss.append(run.ckpt)
    lg = os.path.join(root, run.log)
    if not os.path.exists(lg):
        miss.append(run.log)
    else:
        with open(lg, errors='replace') as f:
            if 'done in' not in f.read():
                bad.append(f'{run.log} has no completion line')
    cfg = None
    cp = os.path.join(root, run.config)
    if not os.path.exists(cp):
        miss.append(run.config)
    else:
        with open(cp) as f:
            cfg = json.load(f)
        want = {'first_mask': 'exact', 'mask_ties': 'exact'}
        if kind == 'arm':
            want['mask_mode'] = 'active'
        for k, v in want.items():
            if _get(cfg, k) != v:
                bad.append(f'{run.config} records {k}={_get(cfg, k)!r}, want {v!r}')
        if kind == 'policy' and _get(cfg, 'mask_mode') not in ('<absent>', 'nondelay'):
            bad.append(f'{run.config} records mask_mode={_get(cfg, "mask_mode")!r}, want the '
                       f'non-delay mask')
        seed = _get(cfg, 'seed_train')
        if seed == '<absent>' or int(seed) != run.seed:
            bad.append(f'{run.config} records seed {seed}, want {run.seed}')
        for k in BUDGET_KEYS:
            if _get(cfg, k) == '<absent>':
                bad.append(f'{run.config} records no {k}')
    sp = os.path.join(root, run.summary())
    if not os.path.exists(sp):
        miss.append(run.summary())
    else:
        with open(sp, errors='replace') as f:
            if SUMMARY_MASK not in f.read():
                bad.append(f'{run.summary()} does not record "{SUMMARY_MASK}"')
    ck_time = os.path.getmtime(ck) if os.path.exists(ck) else None
    for c in combos:
        rel = run.array(c)
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            miss.append(rel)
            continue
        a = np.load(p)
        if a.ndim not in (1, 2) or a.shape[0] != n:
            bad.append(f'{rel} has shape {a.shape}, want {n} rows')
            continue
        ms = (a[:, 0] if a.ndim == 2 else a).astype(float)
        if not np.all(np.isfinite(ms)) or np.any(ms <= 0):
            bad.append(f'{rel} holds makespans that are not finite and positive')
            continue
        if ck_time is not None and os.path.getmtime(p) < ck_time:
            bad.append(f'{rel} is older than its checkpoint {run.ckpt}')
            continue
        cache[(run.name, run.folder, c)] = ms
    return miss, bad, cfg


def assess(root, seeds=None):
    """Every input of the exact branch, read only. Returns dict(sets, seeds, n, files,
    arm_arrays, missing, problems, cache): the plan's (folder, combinations), the compared
    seeds, test instances per folder, the number of files checked, the number of arm
    arrays present, the missing files and the other problems (relative paths)."""
    check_layout()
    sets = plan_sets()
    seeds = tuple(seeds) if seeds else compared_seeds()
    out = dict(sets=sets, seeds=list(seeds), n={}, files=0, arm_arrays=0, missing=[],
               problems=[], cache={})
    for folder, combos in sets:
        n = test_instances(root, folder)
        out['n'][folder] = n
        arm, pol = runs_of(seeds, folder)
        for s in seeds:
            cfgs = {}
            for kind, run in (('arm', arm[s]), ('policy', pol[s])):
                miss, bad, cfg = run_problems(run, combos, root, n, out['cache'], kind)
                out['missing'] += [m for m in miss if m not in out['missing']]
                out['problems'] += [b for b in bad if b not in out['problems']]
                out['files'] += 4 + len(combos)    # checkpoint, log, configuration, summary
                cfgs[kind] = cfg
            out['arm_arrays'] += sum(os.path.exists(os.path.join(root, arm[s].array(c)))
                                     for c in combos)
            if cfgs['arm'] is not None and cfgs['policy'] is not None:
                for k in BUDGET_KEYS:
                    a, b = _get(cfgs['arm'], k), _get(cfgs['policy'], k)
                    if a != b:
                        out['problems'].append(
                            f'{arm[s].config} records {k}={a!r}, {pol[s].config} {b!r}: '
                            f'not the same training budget')
    out['problems'] = list(dict.fromkeys(out['problems']))
    return out


# ------------------------------------------------------------------ statistics
def component(d, B):
    """Interval package of one paired comparison, d [k, n] (one combination) or
    [k, n, c] (pooled; a drawn base instance brings its c combinations): k >= 2, crossed
    hierarchical bootstrap with a fresh default_rng(HIER_SEED) and the seed-level t
    interval; k == 1, bootstrap over base instances with a fresh default_rng(INST_SEED)."""
    import stats_family as SF
    x = np.asarray(d, dtype=float)
    k, n = x.shape[0], x.shape[1]
    c = x.shape[2] if x.ndim == 3 else 1
    if k >= 2:
        hb = SF.hier_boot(x, B=B, seed=HIER_SEED, alpha=ALPHA)
        t = SF.seed_t(x, ALPHA)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                   seed_means=t['seed_means'], scheme='hierarchical',
                   stream=f'default_rng({HIER_SEED})')
    else:
        ib = SF.instance_boot(x[0], B=B, seed=INST_SEED, alpha=ALPHA)
        out = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], t_lo=float('nan'),
                   t_hi=float('nan'), seed_means=[float(x[0].mean())], scheme='instance',
                   stream=f'default_rng({INST_SEED})')
    out.update(k=int(k), n=int(n), c=int(c), direction=SF.direction(out['lo'], out['hi']))
    return out


def differences(cache, seeds, folder, combos):
    """{combo: d [k seeds, n instances]}: d = 100 (arm - policy) / policy, seed s
    against seed s."""
    import stats_family as SF
    arm, pol = runs_of(seeds, folder)
    return {r: np.stack([SF.rel_gap(cache[(arm[s].name, folder, r)],
                                    cache[(pol[s].name, folder, r)]) for s in seeds])
            for r in combos}


def counts_of(comps):
    lo_above = [r for r, c in comps.items() if c['lo'] > 0]
    hi_below = [r for r, c in comps.items() if c['hi'] < 0]
    return dict(longer=len(lo_above), shorter=len(hi_below),
                neither=len(comps) - len(lo_above) - len(hi_below),
                longer_combos=lo_above, shorter_combos=hi_below, combos=len(comps))


def analyse(assessed, B):
    res = {}
    seeds = tuple(assessed['seeds'])
    for folder, combos in assessed['sets']:
        D = differences(assessed['cache'], seeds, folder, combos)
        comps = {r: component(D[r], B) for r in combos}
        pooled = component(np.stack([D[r] for r in combos], axis=2), B)
        pooled['combos'] = list(combos)
        arm, pol = runs_of(seeds, folder)
        for r in combos:
            comps[r]['arm_mean_makespan'] = float(np.mean(
                [assessed['cache'][(arm[s].name, folder, r)] for s in seeds]))
            comps[r]['policy_mean_makespan'] = float(np.mean(
                [assessed['cache'][(pol[s].name, folder, r)] for s in seeds]))
        res[folder] = dict(n=assessed['n'][folder], combos=list(combos), components=comps,
                           pooled=pooled, counts=counts_of(comps),
                           runs=dict(arm=[arm[s].name for s in seeds],
                                     policy=[pol[s].name for s in seeds]))
    return res


# ------------------------------------------------------------------ output
def f2(x):
    """Signed two decimals of the unrounded value, so a printed bound keeps its sign; an
    exact zero prints +0.00 (scripts/p89_public_family.py)."""
    x = float(x)
    return f'{0.0 if x == 0 else x:+.2f}'


def u2(x):
    """Two decimals without a forced sign (the format of ActGapLo and ActGapHi)."""
    x = float(x)
    return f'{0.0 if x == 0 else x:.2f}'


def macro_lines(res, seeds, B):
    """(name, value, comment) of every macro."""
    tr = res[FOLDER]
    comps, pooled, cnt = tr['components'], tr['pooled'], tr['counts']
    means = [c['mean'] for c in comps.values()]
    what = 'difference of the active-mask arm from the policy (% of the policy\'s makespan)'
    L = [
        ('ActWorseN', str(cnt['longer']),
         f'combinations (of {cnt["combos"]}, 10x25) whose 95% interval of the {what} lies '
         f'above zero (the arm has the longer schedule)'),
        ('ActBetterN', str(cnt['shorter']),
         f'combinations (of {cnt["combos"]}, 10x25) whose 95% interval of the {what} lies '
         f'below zero (the arm has the shorter schedule)'),
        ('ActCombN', str(cnt['combos']), 'combinations compared at 10x25'),
        ('ActGapLo', u2(min(means)), f'smallest mean {what} over the combinations, 10x25'),
        ('ActGapHi', u2(max(means)), f'largest mean {what} over the combinations, 10x25'),
    ]
    if 'S' in comps:
        L += [('ActSDelta', f2(comps['S']['mean']), f'mean {what}, combination S, 10x25'),
              ('ActSLo', f2(comps['S']['lo']), 'lower bound of its 95% interval'),
              ('ActSHi', f2(comps['S']['hi']), 'upper bound of its 95% interval')]
    if 'N' in comps:
        L += [('ActNDelta', f2(comps['N']['mean']), f'mean {what}, combination N, 10x25')]
    L += [
        ('ActPoolDelta', f2(pooled['mean']),
         f'mean {what}, pooled over the {len(pooled["combos"])} combinations, 10x25'),
        ('ActPoolLo', f2(pooled['lo']), 'lower bound of its 95% interval'),
        ('ActPoolHi', f2(pooled['hi']), 'upper bound of its 95% interval'),
        ('ActSeedN', str(len(seeds)),
         'training seeds compared, seed s of the arm against seed s of the policy'),
        ('ActInstN', str(tr['n']), 'test instances per combination at 10x25'),
        ('ActBoot', str(B), 'bootstrap replicates of every interval'),
    ]
    for folder, x in res.items():
        if folder == FOLDER:
            continue
        w, cs, pl = SIZE_WORD[folder], x['counts'], x['pooled']
        ms = [c['mean'] for c in x['components'].values()]
        L += [
            (f'Act{w}WorseN', str(cs['longer']),
             f'combinations (of {cs["combos"]}, {folder}) whose 95% interval of the {what} lies above zero'),
            (f'Act{w}BetterN', str(cs['shorter']),
             f'combinations (of {cs["combos"]}, {folder}) whose 95% interval of the {what} lies below zero'),
            (f'Act{w}GapLo', u2(min(ms)), f'smallest mean {what}, {folder}'),
            (f'Act{w}GapHi', u2(max(ms)), f'largest mean {what}, {folder}'),
            (f'Act{w}PoolDelta', f2(pl['mean']),
             f'mean {what}, pooled over the {len(pl["combos"])} combinations, {folder}'),
            (f'Act{w}PoolLo', f2(pl['lo']), 'lower bound of its 95% interval'),
            (f'Act{w}PoolHi', f2(pl['hi']), 'upper bound of its 95% interval'),
            (f'Act{w}InstN', str(x['n']), f'test instances per combination at {folder}'),
        ]
    return L


def defined_elsewhere(root, own):
    """{macro name: file} of every macro defined in the macro files and staged sections of
    paper/ and manuscript/, outside this script's outputs and the files that hold the same
    quantities of the earlier environment."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    files = set()
    for pat in ('manuscript/macros*.tex', 'manuscript/*/macros*.tex',
                'manuscript/sections_new/**/*.tex', 'manuscript/supp_new/**/*.tex',
                'paper/macros*.tex'):
        files |= set(glob.glob(os.path.join(root, pat), recursive=True))
    out = {}
    for f in sorted(files):
        if os.path.abspath(f) in own or os.path.basename(f) == SAME_QUANTITIES:
            continue
        with open(f, errors='replace') as fh:
            for name in rx.findall(fh.read()):
                out.setdefault(name, os.path.relpath(f, root))
    return out, len(files)


def check_names(names, root, own):
    bad = [x for x in names if not re.fullmatch(MACRO_PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {MACRO_PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse('duplicate macro names in the output')
    other, n_files = defined_elsewhere(root, own)
    clash = sorted(set(names) & set(other))
    if clash:
        refuse('macro name collision: ' + ', '.join(f'\\{c} ({other[c]})' for c in clash[:5]))
    return n_files


def header(seeds):
    return (f'Generated by {GENERATOR} -- do not hand-edit; environment exact, training seeds '
            f'{",".join(map(str, seeds))} (the seeds both arms have)')


def table_lines(res, seeds):
    L = [f'% {header(seeds)}', '\\begin{tabular}{lrc}', '\\toprule',
         'combination & difference (\\%) & 95\\%~interval \\\\', '\\midrule']
    many = len(res) > 1
    for i, (folder, x) in enumerate(res.items()):
        if i:
            L.append('\\midrule')
        if many:
            L.append(f'\\multicolumn{{3}}{{l}}{{{SIZE_TEXT[folder]}}} \\\\')
        for r in x['combos']:
            c = x['components'][r]
            L.append(f'${r}$ & ${f2(c["mean"])}$ & $[{f2(c["lo"])},\\,{f2(c["hi"])}]$ \\\\')
        p = x['pooled']
        label = 'all eight' if len(p['combos']) == 8 else f'all {len(p["combos"])}'
        L.append(f'{label} & ${f2(p["mean"])}$ & $[{f2(p["lo"])},\\,{f2(p["hi"])}]$ \\\\')
    L += ['\\bottomrule', '\\end{tabular}']
    return L


def ci(c, lo='lo', hi='hi'):
    a, b = c[lo], c[hi]
    if a is None or b is None or not (math.isfinite(a) and math.isfinite(b)):
        return 'n/a'
    return f'[{f2(a)}, {f2(b)}]'


def markdown(res, seeds, B, n_inputs):
    L = ['# The policy trained with the active-schedule mask against the policy', '',
         f'{header(seeds)}.', '',
         f'Arm {ARM} (the released recipe with --mask_mode active) against the policy '
         f'(the released recipe), seed s against seed s on seeds {", ".join(map(str, seeds))}. '
         f'd = 100 (arm - policy) / policy per seed and test instance; positive means the '
         f'arm has the longer schedule. 95% intervals: crossed hierarchical bootstrap over '
         f'seeds and base instances, B = {B}, a fresh default_rng({HIER_SEED}) per quantity '
         f'(one seed: bootstrap over base instances, default_rng({INST_SEED})); the '
         f'seed-level t interval alongside. Pooled rows keep the combinations of a drawn '
         f'base instance together. {n_inputs} files checked.', '']
    for folder, x in res.items():
        L += [f'## {folder} ({SIZE_TEXT.get(folder, folder)}, {x["n"]} test instances)', '',
              '| combination | arm, mean makespan | policy, mean makespan | difference (%) '
              '| 95% interval | seed-level t interval | direction |',
              '|---|---|---|---|---|---|---|']
        for r in x['combos']:
            c = x['components'][r]
            L.append(f'| {r} | {c["arm_mean_makespan"]:.1f} | {c["policy_mean_makespan"]:.1f} '
                     f'| {f2(c["mean"])} | {ci(c)} | {ci(c, "t_lo", "t_hi")} '
                     f'| {c["direction"]} |')
        p = x['pooled']
        L.append(f'| pooled ({len(p["combos"])}) | | | {f2(p["mean"])} | {ci(p)} '
                 f'| {ci(p, "t_lo", "t_hi")} | {p["direction"]} |')
        cs = x['counts']
        L += ['', f'Interval above zero (the arm longer) in {cs["longer"]} of {cs["combos"]} '
                  f'combinations ({", ".join(cs["longer_combos"]) or "none"}); below zero (the '
                  f'arm shorter) in {cs["shorter"]} ({", ".join(cs["shorter_combos"]) or "none"}).',
              '']
    return L


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def _remove(paths):
    for rel, p in paths:
        if os.path.exists(p):
            os.remove(p)
            print(f'removed {rel} (refusing to certify)')


def exact_main(argv=None):
    from macro_io import emit
    ap = argparse.ArgumentParser(description='the active-mask arm against the policy, '
                                             'exact-mask environment')
    ap.add_argument('--require', default='none', choices=('none', 'all'),
                    help="'all': refuse unless every input exists (default 'none': skip "
                         "when no array of the arm exists)")
    ap.add_argument('--boot', type=int, default=None, help='bootstrap replicates (20000)')
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    a = ap.parse_args(argv)
    import stats_family as SF
    B = a.boot or SF.B_DEFAULT
    root = os.path.abspath(a.root)
    REG.banner()
    outs = [(rel, REG.out(rel)) for rel in OUT_FILES]   # refuses without PACT_OUT
    _remove(outs)                                      # a refusal below leaves no stale output
    try:
        x = assess(root)
        if x['arm_arrays'] == 0:
            if a.require == 'all':
                refuse(f'--require all: no result array of {ARM} exists; missing files:\n  '
                       + '\n  '.join(x['missing']))
            print(f'skipped: no result array of {ARM} exists ({len(x["missing"])} of '
                  f'{x["files"]} files missing); nothing is written')
            return dict(status='skipped', missing=x['missing'])
        if x['missing'] or x['problems']:
            refuse(f'{len(x["missing"])} missing files and {len(x["problems"])} other '
                   f'problems:\n  ' + '\n  '.join(
                       [f'missing {m}' for m in x['missing']] + x['problems']))
        seeds = tuple(x['seeds'])
        res = analyse(x, B)
        L = macro_lines(res, seeds, B)
        names = [n for n, _, _ in L]
        n_files = check_names(names, root, {os.path.abspath(p) for _, p in outs})
        inputs = sorted({k for k in _inputs(x)})
        out = dict(generator=GENERATOR, environment='exact', setting=REG.describe(),
                   arm=ARM, recipe=REG.SECONDARY_ARMS[ARM][0], policy='released recipe',
                   seeds=list(seeds), boot=B, alpha=ALPHA,
                   sign='d = 100 (arm - policy) / policy; positive = the arm has the longer '
                        'schedule',
                   pairing='seed s against seed s, instance i against instance i',
                   streams=dict(hierarchical=f'fresh default_rng({HIER_SEED}) per quantity',
                                instance=f'fresh default_rng({INST_SEED}) per quantity'),
                   folders=res, macros={n: v for n, v, _ in L}, n_macros=len(L),
                   macro_files_checked=n_files, inputs=inputs)
        emit(outs[0][1], L, header=header(seeds))
        with open(outs[1][1], 'w') as f:
            f.write('\n'.join(table_lines(res, seeds)) + '\n')
        with open(outs[2][1], 'w') as f:
            json.dump(jsonable(out), f, indent=1)
        md = markdown(res, seeds, B, len(inputs))
        with open(outs[3][1], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        _remove(outs)
        raise
    print('\n'.join(md))
    print(f'{len(L)} macros; wrote ' + ', '.join(rel for rel, _ in outs))
    return dict(status='computed', result=out)


def _inputs(x):
    """Relative paths of every file the exact branch read."""
    seeds = tuple(x['seeds'])
    for folder, combos in x['sets']:
        arm, pol = runs_of(seeds, folder)
        for s in seeds:
            for run in (arm[s], pol[s]):
                yield from (run.ckpt, run.log, run.config, run.summary())
                yield from (run.array(c) for c in combos)


def main(argv=None):
    if REG.env() == 'legacy':
        return legacy_main()
    return exact_main(argv)


if __name__ == '__main__':
    main()
