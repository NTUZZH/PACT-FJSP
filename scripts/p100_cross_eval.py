r"""P100: every specialist under every combination, against the policy, in the
exact-mask environment.

QUESTION
  An inactive rule is a rule whose parameters are zero, and the policy reads no flag
  that says which rules are active. The eight combinations are therefore corners of one
  parameter space, and the specialist trained with all three rules active (LSW) has met
  every mechanism of the other seven. The 8 x 8 matrix tests whether a specialist run
  under a combination it was not trained for matches the policy on that combination.

INPUTS (scripts/arm_registry.py, PACT_ENV=exact, the active seeds PACT_SEEDS; the
evaluation plan scripts/eval_newseeds_plan.py writes seeds 301-305)
  policy        xmask-v1-s<seed> under each combination, folder res_dir('10x25')
                (group principal)
  diagonal      xmask-solo-<R>-s<seed> under R, the same folder (group principal)
  off-diagonal  xmask-solo-<R>-s<seed> under every C != R, folder res_dir(CROSS_FOLDER)
                = 10x25_cross_xmask (group cross)
  Arrays Result_greedy+10x25+family+<run>_<C>.npy, the makespan in column 0, row i the
  test instance i of the 10x25 test set (checked on its descriptors).

STATISTIC
  d = 100 (specialist - policy) / policy per training seed and test instance, the
  specialist of seed s against the policy of seed s (seed numbers, never positions),
  positive = the specialist's schedule is longer. Cell (R, C): row R is the combination
  the specialist was trained for, column C the combination it is run under. Interval:
  p95_secondary_inference.component, the crossed hierarchical bootstrap over seeds and
  base instances of scripts/stats_family.py (B = --boot, a fresh default_rng(11) per
  quantity; with one seed the bootstrap over base instances, default_rng(7)), with the
  seed-level Student t interval in the JSON and the Markdown. Verdicts at the margins
  0.5% and 1% (stats_family.verdict) and the direction (stats_family.direction) come
  from the unrounded interval.
  Pooled rows: each specialist over its seven off-combination columns, the seven
  combinations of a drawn base instance kept together; the LSW row is the LSW
  specialist pooled over the other seven combinations.
  Counts over the 56 off-combination cells and over the seven of the LSW row: cells
  whose interval lies above zero (direction behind, the specialist longer), cells
  equivalent at the 1% margin, cells whose interval lies below zero (direction ahead,
  the specialist shorter).
  Single-rule changes: the 24 off-combination cells whose two combinations differ by
  one rule, grouped by that rule (lag, setup, window) and by the direction. Rule added:
  the specialist trained for A run under B = A plus the rule. Rule removed: the
  specialist trained for B run under A. These groups use the difference of the family
  analysis, g = 100 (policy - specialist) / specialist, negative = the policy's
  schedule is shorter. Each group pools its four cells, and the rows AddAll and DropAll
  pool the twelve cells of a direction, the cells of a drawn base instance kept
  together. The twelve cells of AddAll are the rule-change pairs of
  p88_family_inference.py (section F, the policy against the specialist of the earlier
  combination): when the family record of the same seeds exists under PACT_OUT, its
  pooled mean must equal the mean of AddAll, and its interval is compared. The
  grouping is a description of the matrix. It adds no planned comparison.

CHECKS (before any number; a failure removes the outputs and exits with a message
that starts with "refused")
  * the environment is exact and the outputs go through arm_registry.out (PACT_OUT);
  * every combination of the 10x25 test set has the same file names and base instances
    (p95_secondary_inference.check_bases);
  * for every run read (policy, and each specialist in both folders): checkpoint,
    training log with its completion line and configuration exist, the configuration
    records the run's seed, the exact mask and its training combinations (six for the
    policy, R for specialist R); every array exists for every active seed, has one row
    per test instance, holds finite positive makespans and is not older than its
    checkpoint (p95_secondary_inference.run_problems);
  * an off-combination array that the principal 10x25 campaign also holds (the
    combinations of SPEC_EVAL_10x25 and the LSW specialist on all eight) must hold the
    same makespans: a greedy rollout on one thread does not depend on the job it ran in.

OUTPUTS (arm_registry.out): manuscript/macros_crosseval.tex (prefix Xev, letters only,
no name defined in another macro file or staged section), results/cross_eval.json,
results/cross_eval.md.
  \Xev<R>On<C>  mean of cell (R, C); suffixes Lo, Hi (95% interval), VHalf, VOne
                (verdicts at 0.5% and 1%), Dir (direction)
  \Xev<R>Off    specialist R pooled over its seven off-combination columns, same suffixes
  \XevOffBehindN, \XevOffEqOneN, \XevOffAheadN            counts over the 56 cells
  \XevLSWOffBehindN, \XevLSWOffEqOneN, \XevLSWOffAheadN   counts over the LSW row
  \XevChgAdd<Rule>, \XevChgDrop<Rule>   Rule in Lag, Setup, Window: the policy minus the
                specialist in percent of the specialist's makespan, pooled over the four
                cells of the rule added or removed, with the suffixes of a cell
  \XevChgAddAll, \XevChgDropAll  the twelve cells of a direction, the same suffixes
  \XevChgRulePairN               cells of one rule and one direction
  \XevBoot, \XevSeedN, \XevInstN, \XevOffN                replicates, seeds, test
                                                          instances, off-combination cells

Usage (repository root, CPU):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=. python scripts/p100_cross_eval.py [--boot 20000]
  [--root <repository root the inputs are read from>]
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import eval_newseeds_plan as PLAN  # noqa: E402
import p95_secondary_inference as P95  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import f2, jsonable  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = 'scripts/p100_cross_eval.py'
R8 = list(REG.COMBOS)
SIX = ','.join(REG.TRAIN_COMBOS)
FOLDER = '10x25'                       # the policy and the diagonal
CROSS = REG.CROSS_FOLDER               # the off-combination arrays
MARGINS = P95.MARGINS
PREFIX = 'Xev'
OUT_FILES = ('manuscript/macros_crosseval.tex', 'results/cross_eval.json',
             'results/cross_eval.md')
MAX_LISTED = 12                        # problems printed in a refusal
RULES = (('L', 'Lag', 'lag'), ('S', 'Setup', 'setup'), ('W', 'Window', 'window'))
FAMILY_RECORD = 'results/family_inference.json'   # section F holds the twelve cells of AddAll
MEAN_TOL = 1e-9                        # the two analyses average the same differences


class Refused(SystemExit):
    """A precondition failed; the outputs are removed and no number is written."""


def refuse(msg):
    raise Refused(f'refused: {msg}')


def off(r):
    """The combinations specialist r was not trained for, in the order of COMBOS."""
    return [c for c in R8 if c != r]


def rules_of(c):
    """The timing rules of a combination (N has none)."""
    return frozenset() if c == 'N' else frozenset(c)


def single_rule_changes():
    """{rule letter: the pairs (A, B) with B = A plus that rule}, A in the order of COMBOS."""
    return {x: [(a, b) for a in R8 for b in R8
                if x not in rules_of(a) and rules_of(b) == rules_of(a) | {x}]
            for x, _, _ in RULES}


def change_groups():
    """{group: (cells (trained for, run under), what the group holds)} of the 24 cells whose
    two combinations differ by one rule. Add<Rule>: the specialist of A under B = A plus the
    rule. Drop<Rule>: the specialist of B under A. AddAll, DropAll: the twelve of a direction."""
    ch = single_rule_changes()
    G = {}
    for x, name, word in RULES:
        G[f'Add{name}'] = (list(ch[x]), f'the {word} rule added: the specialist trained for A '
                                        f'run under B = A plus the {word} rule')
        G[f'Drop{name}'] = ([(b, a) for a, b in ch[x]],
                            f'the {word} rule removed: the specialist trained for B run under '
                            f'A = B without the {word} rule')
    up = [p for x, _, _ in RULES for p in ch[x]]
    G['AddAll'] = (up, 'one rule added: the specialist trained for A run under B = A plus one rule')
    G['DropAll'] = ([(b, a) for a, b in up],
                    'one rule removed: the specialist trained for B run under A = B without one rule')
    return G


# ================================================================== runs
def policy_run(seed):
    return P95.Run(REG.model('released', None, seed, 'exact'), 'family', seed, FOLDER)


def specialist_run(r, seed, folder):
    return P95.Run(REG.model('specialist', r, seed, 'exact'), 'family', seed, folder)


def principal_combos():
    """{specialist combination: combinations the principal 10x25 campaign evaluates}."""
    return {combo: set(regs) for arm, combo, regs in PLAN.principal_runs(FOLDER)
            if arm == 'specialist'}


def reads(seeds):
    """(Run, combinations, configuration values it must record) of every run read."""
    out = [(policy_run(s), R8, {'family_regimes': SIX}) for s in seeds]
    for r in R8:
        for s in seeds:
            out.append((specialist_run(r, s, FOLDER), [r], {'family_regimes': r}))
            out.append((specialist_run(r, s, CROSS), off(r), {'family_regimes': r}))
    return out


def check_layout():
    """The registry, the plan and this script name the same folders and files."""
    if REG.res_dir(CROSS) != f'{REG.results_root()}/{CROSS}{REG.FOLDER_SUFFIX["exact"]}':
        refuse(f'the cross folder {REG.res_dir(CROSS)} does not follow the registry layout')
    if PLAN.CROSS_TAG != REG.summary_tag(CROSS) or \
            f'{PLAN.CROSS_DATA}{PLAN.CROSS_TAG}' != f'{CROSS}{REG.FOLDER_SUFFIX["exact"]}':
        refuse('the plan writes the off-combination arrays into another folder than the '
               'registry names')
    for r in R8:
        if PLAN.cross_combos(r) != off(r):
            refuse(f'the plan evaluates specialist {r} on {PLAN.cross_combos(r)}, this '
                   f'script reads {off(r)}')
    s = REG.seeds()[0]
    for run, c in ((policy_run(s), 'N'), (specialist_run('LS', s, FOLDER), 'LS'),
                   (specialist_run('LS', s, CROSS), 'N')):
        arm, combo = ('released', None) if 'xmask-v1' in run.name else ('specialist', 'LS')
        want = REG.result_path(arm, combo, s, run.folder, c)
        if run.array(c) != want or run.ckpt != REG.checkpoint(arm, combo, s) or \
                run.log != REG.train_log(arm, combo, s) or run.config != REG.config_path(arm, combo, s):
            refuse(f'the paths of {run.name} differ from the registry ({run.array(c)} vs {want})')


def load_all(root, seeds):
    """Number of test instances, the makespan cache {(run, folder, combo): array} and
    the number of off-combination arrays compared with the principal campaign."""
    try:
        n = P95.check_bases(root, PLAN.FACTORY[FOLDER][0])
    except P95.Refusal as e:
        refuse(str(e).replace('REFUSING: ', ''))
    cache, problems = {}, []
    for run, combos, want in reads(seeds):
        problems += P95.run_problems(run, combos, root, n, True, cache, want)
    if problems:
        listed = [P95._relmsg(p, root) for p in problems[:MAX_LISTED]]
        more = len(problems) - len(listed)
        refuse(f'{len(problems)} problems in the inputs (seeds {",".join(map(str, seeds))}):\n  '
               + '\n  '.join(listed) + (f'\n  ... and {more} more' if more else ''))
    main = principal_combos()
    compared, differ = 0, []
    for r in R8:
        for s in seeds:
            for c in off(r):
                if c not in main[r]:
                    continue
                q = os.path.join(root, specialist_run(r, s, FOLDER).array(c))
                if not os.path.exists(q):
                    continue
                a = np.load(q)
                b = cache[(specialist_run(r, s, CROSS).name, CROSS, c)]
                compared += 1
                if a.shape[0] != n or not np.array_equal((a[:, 0] if a.ndim == 2 else a).astype(float), b):
                    differ.append(P95._relmsg(q, root))
    if differ:
        refuse(f'{len(differ)} off-combination arrays of the principal campaign differ from the '
               f'arrays in {REG.res_dir(CROSS)}: ' + '; '.join(differ[:MAX_LISTED]))
    return n, cache, compared


# ================================================================== statistics
def matrix(cache, seeds, B):
    """Cells {(R, C): package}, pooled off-combination rows {R: package}, counts, and the
    single-rule changes pooled by rule and direction {group: package}."""
    def ms(run, c):
        return cache[(run.name, run.folder, c)]

    D = {}
    for r in R8:
        for c in R8:
            folder = FOLDER if c == r else CROSS
            D[(r, c)] = np.stack([SF.rel_gap(ms(specialist_run(r, s, folder), c),
                                             ms(policy_run(s), c)) for s in seeds])
    cells = {k: P95.component(d, B)[0] for k, d in D.items()}
    rows = {}
    for r in R8:
        rows[r] = P95.component(np.stack([D[(r, c)] for c in off(r)], axis=2), B)[0]
        rows[r]['combos'] = off(r)

    def count(keys):
        cs = [cells[k] for k in keys]
        return dict(behind=sum(c['direction'] == 'behind' for c in cs),
                    equivalent_1=sum(c['v1.0'] == 'equivalent' for c in cs),
                    ahead=sum(c['direction'] == 'ahead' for c in cs), cells=len(cs))
    offk = [(r, c) for r in R8 for c in off(r)]
    counts = dict(off=count(offk), lsw=count([('LSW', c) for c in off('LSW')]),
                  diagonal=count([(r, r) for r in R8]))
    # the single-rule changes in the difference of the family analysis: the policy minus
    # the specialist, in percent of the specialist's makespan
    G = change_groups()
    P = {(r, c): np.stack([SF.rel_gap(ms(policy_run(s), c), ms(specialist_run(r, s, CROSS), c))
                           for s in seeds])
         for r, c in G['AddAll'][0] + G['DropAll'][0]}
    groups = {}
    for g, (keys, what) in G.items():
        groups[g] = P95.component(np.stack([P[k] for k in keys], axis=2), B)[0]
        groups[g].update(pairs=[list(k) for k in keys], what=what)
    return cells, rows, counts, groups


def family_check(groups, seeds, B):
    """AddAll against the pooled rule-change row of the family record (section F of
    p88_family_inference.py, the policy against the specialist of the earlier combination),
    when that record exists under PACT_OUT for the same seeds. The means must agree; the
    intervals are reported side by side."""
    path = REG.out(FAMILY_RECORD)
    if not os.path.exists(path):
        return dict(compared=False, reason=f'{FAMILY_RECORD} does not exist under PACT_OUT')
    with open(path) as f:
        fam = json.load(f)
    if list(fam.get('seeds', [])) != list(seeds):
        return dict(compared=False, reason=f'{FAMILY_RECORD} holds the seeds {fam.get("seeds")}')
    blk = fam['rule_change']['Pol']
    want = sorted(tuple(v) for v in blk['pairs'].values())
    if want != sorted(tuple(k) for k in groups['AddAll']['pairs']):
        refuse(f'the rule-change pairs of {FAMILY_RECORD} are not the twelve cells of AddAll')
    f, g = blk['pooled'], groups['AddAll']
    if abs(f['mean'] - g['mean']) > MEAN_TOL:
        refuse(f'the twelve rule-change pairs pooled: {g["mean"]:+.6f} here against '
               f'{f["mean"]:+.6f} in {FAMILY_RECORD}')
    return dict(compared=True, mean=g['mean'], family_mean=f['mean'],
                interval=[g['lo'], g['hi']], family_interval=[f['lo'], f['hi']],
                same_replicates=fam.get('boot') == B)


# ================================================================== output
def cell_line(name, c, what, unit='combinations'):
    """Macros of one comparison (mean, interval, verdicts at both margins, direction) with
    a comment that says what they hold."""
    s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
         f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
         f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
         + ''.join(f'\\newcommand{{\\{name}V{P95.MW[mg]}}}{{{SF.TEX[c[f"v{mg}"]]}}}' for mg in MARGINS)
         + f'\\newcommand{{\\{name}Dir}}{{{SF.TEX[c["direction"]]}}}')
    over = (f'{c["k"]} seed pairs and {c["n"]} base instances' if c['scheme'] == 'hierarchical'
            else f'{c["n"]} base instances of one training run')
    pooled = f', with the {c["c"]} {unit} of a base instance kept together' if c['c'] > 1 else ''
    return (s + f'  % {what}, with the mean, the 95% interval over {over}{pooled}, the verdicts at '
            f'the 0.5% and 1% margins and the direction of the interval [{c["lo"]:+.4f}, {c["hi"]:+.4f}].')


def macro_lines(res):
    B, seeds, n = res['boot'], res['seeds'], res['n']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Each specialist under each combination against the policy, under the exact mask, on '
         f'the 10x25 test set with {n} base instances and the training seeds '
         f'{",".join(map(str, seeds))}. The intervals come from {B} replicates of the crossed '
         f'hierarchical bootstrap, each quantity with a fresh default_rng({P95.HIER_SEED}).',
         '% A difference is the makespan of the specialist minus that of the policy, in percent of '
         'the policy\'s, with specialist seed s against policy seed s. A positive difference means '
         'a longer schedule for the specialist. Rows name the combination the specialist was '
         'trained for, and columns name the combination it is run under.',
         f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}\\newcommand{{\\{PREFIX}SeedN}}{{{len(seeds)}}}'
         f'\\newcommand{{\\{PREFIX}InstN}}{{{n}}}'
         f'\\newcommand{{\\{PREFIX}OffN}}{{{res["counts"]["off"]["cells"]}}}'
         f'  % Bootstrap replicates, training seeds, test instances and off-combination cells.']
    for r in R8:
        for c in R8:
            where = 'its own combination' if c == r else 'a combination it was not trained for'
            L.append(cell_line(f'{PREFIX}{r}On{c}', res['cells'][(r, c)],
                               f'Specialist {r} run under {c}, {where}, against the policy'))
    for r in R8:
        L.append(cell_line(f'{PREFIX}{r}Off', res['rows'][r],
                           f'Specialist {r} against the policy, pooled over {",".join(off(r))}'))
    for key, name, what in (('off', 'Off', 'the 56 off-combination cells'),
                            ('lsw', 'LSWOff', 'the seven off-combination cells of the LSW row')):
        x = res['counts'][key]
        L.append(f'\\newcommand{{\\{PREFIX}{name}BehindN}}{{{x["behind"]}}}'
                 f'\\newcommand{{\\{PREFIX}{name}EqOneN}}{{{x["equivalent_1"]}}}'
                 f'\\newcommand{{\\{PREFIX}{name}AheadN}}{{{x["ahead"]}}}'
                 f'  % Of {what}, the number whose interval lies above zero (specialist longer), '
                 f'the number equivalent at the 1% margin and the number whose interval lies below '
                 f'zero (specialist shorter).')
    per_rule = len(res['groups'][f'Add{RULES[0][1]}']['pairs'])
    L.append(f'\\newcommand{{\\{PREFIX}ChgRulePairN}}{{{per_rule}}}'
             f'  % Cells of one rule and one direction among the single-rule changes.')
    L.append('% Single-rule changes. The difference is the makespan of the policy minus that of '
             'the specialist, in percent of the specialist\'s, as in the comparison of the policy '
             'with the specialists. A negative difference means a shorter schedule for the policy.')
    for g, x in res['groups'].items():
        cells = ', '.join(f'{r} under {c}' for r, c in x['pairs'])
        L.append(cell_line(f'{PREFIX}Chg{g}', x, f'The policy against a specialist after '
                           f'{x["what"]}, pooled over the cells {cells}', unit='cells'))
    return L


def macro_names(lines):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))


def defined_elsewhere(root, own):
    """{macro name: file} of every macro defined in the paper's macro files and staged
    sections, outside this script's own output."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    files = set()
    for pat in ('manuscript/macros*.tex', 'manuscript/sections_new/**/*.tex',
                'manuscript/supp_new/**/*.tex', 'paper/macros*.tex'):
        files |= set(glob.glob(os.path.join(root, pat), recursive=True))
    out = {}
    for f in sorted(files):
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            for name in rx.findall(fh.read()):
                out.setdefault(name, os.path.relpath(f, root))
    return out, len(files)


def check_names(names, root, outs):
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse('duplicate macro names in the output')
    own = {os.path.abspath(p) for p in outs} | {os.path.abspath(os.path.join(root, OUT_FILES[0]))}
    other, n_files = defined_elsewhere(root, own)
    clash = sorted(x for x in other if x.startswith(PREFIX))
    if clash:
        refuse(f'the prefix {PREFIX} is in use: {clash[0]} in {other[clash[0]]}')
    return n_files


def ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def markdown(res):
    B, seeds = res['boot'], res['seeds']
    md = ['# Every specialist under every combination against the policy', '',
          f'Exact mask, 10x25 test set ({res["n"]} base instances per combination), seeds '
          f'{", ".join(map(str, seeds))}. d = 100 (specialist - policy) / policy, specialist seed '
          f's against policy seed s; positive = the specialist schedule is longer. 95% intervals: '
          f'crossed hierarchical bootstrap over seeds and base instances (B = {B}, fresh '
          f'default_rng({P95.HIER_SEED}) per quantity). Rows: the combination the specialist was '
          f'trained for; columns: the combination it is run under; the diagonal is the specialist '
          f'on its own combination.', '',
          '## Mean [95% interval]', '',
          '| trained for | ' + ' | '.join(R8) + ' |', '|---' * 9 + '|']
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(ci(res['cells'][(r, c)]) for c in R8) + ' |')
    md += ['', '## Verdicts at 0.5% / 1% and direction', '',
           '| trained for | ' + ' | '.join(R8) + ' |', '|---' * 9 + '|']
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(
            f'{x["v0.5"]} / {x["v1.0"]} / {x["direction"]}'
            for x in (res['cells'][(r, c)] for c in R8)) + ' |')
    md += ['', '## Each specialist pooled over its seven off-combination columns', '',
           '| specialist | mean [95% CI] | seed-t CI | 0.5% | 1% | direction |',
           '|---|---|---|---|---|---|']
    for r in R8:
        x = res['rows'][r]
        md.append(f'| {r} | {ci(x)} | {P95.tci(x)} | {x["v0.5"]} | {x["v1.0"]} | {x["direction"]} |')
    md += ['', '## Single-rule changes grouped by the rule and the direction', '',
           'The 24 off-combination cells whose two combinations differ by one rule. Rule added: '
           'the specialist trained for A run under B = A plus the rule. Rule removed: the '
           'specialist trained for B run under A. In this table g = 100 (policy - specialist) / '
           'specialist, negative = the policy schedule is shorter (the difference of the family '
           'analysis). Each row pools its cells, the cells of a drawn base instance kept together. '
           'The grouping describes the matrix and adds no planned comparison.', '',
           '| group | cells (trained for > run under) | mean [95% CI] | seed-t CI | 0.5% | 1% | '
           'direction |', '|---|---|---|---|---|---|---|']
    for g, x in res['groups'].items():
        md.append(f'| {g} | ' + ', '.join(f'{r}>{c}' for r, c in x['pairs'])
                  + f' | {ci(x)} | {P95.tci(x)} | {x["v0.5"]} | {x["v1.0"]} | {x["direction"]} |')
    fc = res['family_check']
    if fc['compared']:
        md += ['', f'AddAll holds the twelve rule-change pairs of the family record (section F): '
               f'mean {fc["mean"]:+.4f} here and {fc["family_mean"]:+.4f} there, interval '
               f'[{fc["interval"][0]:+.4f}, {fc["interval"][1]:+.4f}] here and '
               f'[{fc["family_interval"][0]:+.4f}, {fc["family_interval"][1]:+.4f}] there.']
    else:
        md += ['', f'AddAll was not compared with the family record: {fc["reason"]}.']
    md += ['', '## Counts', '']
    for key, what in (('off', 'off-combination cells'), ('lsw', 'off-combination cells of the LSW row'),
                      ('diagonal', 'diagonal cells')):
        x = res['counts'][key]
        md.append(f'- {x["cells"]} {what}: interval above zero {x["behind"]}, equivalent at 1% '
                  f'{x["equivalent_1"]}, interval below zero {x["ahead"]}.')
    md += ['', f'{res["compared"]} off-combination arrays of the principal 10x25 campaign were '
           f'compared with the arrays of {res["cross_folder"]}; all hold the same makespans.', '']
    return md


# ================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if REG.env() != 'exact':
        refuse(f'the off-combination arrays exist in the exact environment only (PACT_ENV='
               f'{REG.env()}); run with PACT_ENV=exact')
    try:
        outs = [REG.out(p) for p in OUT_FILES]
    except SystemExit as e:
        refuse(str(e).replace('REFUSING: ', ''))
    for p in outs:                     # a refusal below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    try:
        check_layout()
        seeds = REG.seeds()
        n, cache, compared = load_all(root, seeds)
        cells, rows, counts, groups = matrix(cache, seeds, a.boot)
        res = dict(generator=GENERATOR, environment='exact', boot=a.boot, alpha=P95.ALPHA,
                   margins=MARGINS, seeds=list(seeds), n=n, cross_folder=REG.res_dir(CROSS),
                   folder=REG.res_dir(FOLDER), compared=compared, cells=cells, rows=rows,
                   counts=counts, groups=groups,
                   family_check=family_check(groups, seeds, a.boot))
        L = macro_lines(res)
        names = macro_names(L)
        n_files = check_names(names, root, outs)
        out = dict(res, cells={f'{r}|{c}': v for (r, c), v in cells.items()},
                   sign='d = 100 (specialist - policy) / policy; positive = specialist longer',
                   pairing='specialist seed s against policy seed s, instance i against instance i',
                   rows_are='combination the specialist was trained for',
                   columns_are='combination the specialist is run under',
                   stream=f'fresh default_rng({P95.HIER_SEED}) per quantity',
                   lsw_pooled_off=rows['LSW'], files_read=len(cache), n_macros=len(names),
                   macro_files_checked=n_files)
        emit(outs[0], L)
        with open(outs[1], 'w') as f:
            json.dump(jsonable(out), f, indent=1)
        md = markdown(res)
        with open(outs[2], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException as e:
        for p in outs:
            if os.path.exists(p):
                os.remove(p)
                print(f'removed {p} (refused)')
        if isinstance(e, REG.MissingResult):
            raise Refused(f'refused: {e}')
        raise
    print('\n'.join(md))
    print(f'{len(names)} macros in {outs[0]}; {len(cache)} arrays read; {compared} arrays '
          f'compared with the principal campaign')
    return out


if __name__ == '__main__':
    main()
