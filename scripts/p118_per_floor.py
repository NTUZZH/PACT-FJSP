r"""P118: the per-floor batch, 27 modules on the default factory of 25 stations.

WHAT IS COMPUTED (exact environment, training seeds of arm_registry, PACT_SEEDS=301-305)
  For the test set 27x25 (data/FAMILY/27x25, 50 instances; arrays in arm_registry.res_dir('27x25'),
  written by scripts/run_further_sets_eval.sh SET=27x25) the quantities that section H of
  scripts/p88_family_inference.py reports for the shops of 20 and 30 modules, by section H
  itself (p88.section_H with its size lists set to 27x25 alone, p88_sizes):
    policy against the specialist of each combination: d = 100 (policy - specialist) /
      specialist, seed s against seed s; hierarchical 95% interval (fresh default_rng(11) per
      quantity), Bonferroni interval over eight, seed-level t interval, verdicts at the margins
      0.5% and 1% (stats_family.verdict), direction; intersection-union tests and pooled rows
      over the families All, Train and Held of section A (p88.iut_block, p88.pooled_block);
    policy against the best dispatching rule (per instance the best of FIFO, MOR, SPT and MWKR):
      the same d, the combined verdict ahead / behind / not distinguishable with Holm over the
      eight combinations (p88.superiority_block, p89_public_family.superiority_verdict), the
      pooled row and the verdict counts.
  B and alpha are those of the stored record (below).

SIDE BY SIDE
  20x25 and 30x25 are read from the stored record of p88, arm_registry.out(
  'results/family_inference.json') (key larger_shops). The record is refused unless p88 wrote
  it for the exact environment with the active seeds and section H was computed. Before any
  output is written, section H is recomputed from the stored arrays of 20x25 and 30x25 by the
  same call, and every mean, interval end, p-value, verdict and count of the record must be
  reproduced (REPRO_TOL), so the three sizes come from one procedure.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_perfloor.tex  macro_io.emit; names Pfl<quantity>, letters only, checked
                                 against every macro defined in a .tex file under manuscript/
                                 (the output itself left out); p88's row formats (row_line,
                                 iut_lines, count_line): signed values with two decimals
  results/per_floor.json         the three sizes and the reproduction summary
  results/per_floor.md           tables of the three sizes side by side
  A refusal removes all three.

CHECK (--check, read only)
  exit 0 when every array of 27x25, 20x25 and 30x25 that the run reads exists with one row per
  test instance and the stored record is p88's record of this setting with section H computed.

DRY MODE (--dry [DIR])
  The whole path on synthetic arrays in DIR (default a temporary folder, removed at the end):
  arrays of the policy, the specialists and the four dispatching rules at 20, 27 and 30 modules
  with known differences (DRY_GAP), and a record of p88 built from the synthetic 20x25 and 30x25
  arrays by section H; B = DRY_BOOT. The descriptors of data/FAMILY give the test sets.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p118_per_floor.py
       ... python scripts/p118_per_floor.py --check
       python scripts/p118_per_floor.py --dry [DIR]
"""
import argparse
import contextlib
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS  # noqa: E402

GENERATOR = 'scripts/p118_per_floor.py'
PREFIX = 'Pfl'
OUT_FILES = {'macros': 'manuscript/macros_perfloor.tex', 'json': 'results/per_floor.json',
             'md': 'results/per_floor.md'}
ENV = 'exact'
SIZE = '27x25'
SIDE = ('20x25', '30x25')                 # read from the stored record
SIDE_WORD = {'20x25': 'Twenty', '30x25': 'Thirty'}
RECORD = 'results/family_inference.json'
RECORD_GENERATOR = 'scripts/p88_family_inference.py'
REPRO_TOL = 1e-9
DRY_BOOT = 400
R8 = list(REG.COMBOS)


class Refused(Exception):
    """A setting, an input or a record that this script does not accept."""


# ----------------------------------------------------------------- setting
def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the analysis reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')
    return list(REG.seeds())


@contextlib.contextmanager
def p88_sizes(spec):
    """p88's size lists of section H set to the given specialist sizes, restored after."""
    old = F88.SPEC_SIZES, F88.POOL_SIZES
    F88.SPEC_SIZES, F88.POOL_SIZES = tuple(spec), ()
    try:
        yield
    finally:
        F88.SPEC_SIZES, F88.POOL_SIZES = old


def input_state(sizes, seeds):
    """p88.section_state of section H for the given sizes: every array present with one row
    per test instance."""
    with p88_sizes(sizes):
        return F88.section_state('H', seeds)


def section_h(sizes, seeds):
    """{size: section H block} of p88 for the given sizes."""
    with p88_sizes(sizes):
        return F88.section_H(None, seeds)


# ----------------------------------------------------------------- record
def record_path():
    try:
        return REG.out(RECORD)
    except SystemExit as e:
        raise Refused(str(e))


def read_record(seeds, path=None):
    """The stored record of p88 for this setting, with section H computed."""
    path = path or record_path()
    if not os.path.exists(path):
        raise Refused(f'the record {path} does not exist (run scripts/p88_family_inference.py '
                      f'under PACT_OUT first)')
    try:
        rec = json.load(open(path))
    except (OSError, ValueError) as e:
        raise Refused(f'{path} is not readable JSON: {e}')
    if rec.get('generator') != RECORD_GENERATOR:
        raise Refused(f'{path} was written by {rec.get("generator")!r}, not {RECORD_GENERATOR}')
    if not str(rec.get('setting', '')).startswith(f'environment {ENV},'):
        raise Refused(f'{path} was written for {rec.get("setting")!r}, not the {ENV} environment')
    if [int(x) for x in rec.get('seeds') or []] != list(seeds):
        raise Refused(f'{path} was written with seeds {rec.get("seeds")}, the active seeds are '
                      f'{list(seeds)}')
    shops = rec.get('larger_shops') or {}
    miss = [s for s in SIDE if s not in shops]
    if miss or not ((rec.get('sections') or {}).get('H') or {}).get('computed'):
        raise Refused(f'{path} holds no computed section H for {", ".join(miss) or SIDE}')
    return rec, path


def _cmp(a, b, where, out):
    """Compare a recomputed block a with the record b; append differences to out, return the
    number of compared leaves."""
    if isinstance(b, dict):
        n = 0
        for k, v in b.items():
            if k not in a:
                out.append(f'{where}.{k}: absent in the recomputation')
                continue
            n += _cmp(a[k], v, f'{where}.{k}', out)
        return n
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            out.append(f'{where}: list of another length')
            return 1
        return sum(_cmp(x, y, f'{where}[{i}]', out) for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(b, bool) or isinstance(b, str) or b is None:
        if a != b:
            out.append(f'{where}: {a!r} against {b!r}')
        return 1
    if isinstance(b, (int, float)):
        if not isinstance(a, (int, float)) or isinstance(a, bool) or \
                (math.isfinite(b) and abs(float(a) - float(b)) > REPRO_TOL) or \
                (not math.isfinite(b) and not (math.isnan(b) and math.isnan(a))):
            out.append(f'{where}: {a!r} against {b!r}')
        return 1
    out.append(f'{where}: unexpected type {type(b).__name__}')
    return 1


def reproduce(rec, seeds):
    """Section H of 20x25 and 30x25 recomputed from the arrays against the record."""
    mine = json.loads(json.dumps(F88.jsonable(section_h(SIDE, seeds))))
    bad, n = [], 0
    for size in SIDE:
        n += _cmp(mine[size], rec['larger_shops'][size], size, bad)
    if bad:
        raise Refused(f'section H recomputed from the arrays does not reproduce the record in '
                      f'{len(bad)} of {n} numbers (first: {bad[0]})')
    return dict(sizes=list(SIDE), compared=n, identical=True, tolerance=REPRO_TOL)


# ----------------------------------------------------------------- analysis
def analyse(dry=False):
    seeds = check_setting()
    state = input_state((SIZE,) + SIDE, seeds)
    if not state['ready']:
        raise Refused(f'inputs incomplete: {state["reason"]}')
    rec, rpath = read_record(seeds)
    F88.configure(boot=rec['boot'], alpha=rec['alpha'])
    repro = reproduce(rec, seeds)
    blk = section_h((SIZE,), seeds)[SIZE]
    sizes = {SIDE[0]: rec['larger_shops'][SIDE[0]], SIZE: json.loads(json.dumps(F88.jsonable(blk))),
             SIDE[1]: rec['larger_shops'][SIDE[1]]}
    between = {}
    for key, get in (('policy_vs_specialist_pooled', lambda x: x['parity_pooled']['All']['mean']),
                     ('policy_vs_best_rule_pooled', lambda x: x['rule_pool_pooled']['mean'])):
        a, m, b = (get(sizes[s]) for s in (SIDE[0], SIZE, SIDE[1]))
        between[key] = dict(at_20=a, at_27=m, at_30=b, between=bool(min(a, b) <= m <= max(a, b)))
    return dict(generator=GENERATOR, setting=REG.describe(), seeds=seeds, dry=bool(dry),
                boot=F88.B, alpha=F88.ALPHA, record=rpath, reproduction=repro,
                folder=REG.res_dir(SIZE), sizes=sizes, between=between,
                definitions=dict(
                    d='100 (policy - other) / other per training seed and test instance, seed s '
                      'against seed s; negative = the policy is shorter',
                    specialist='the specialist trained on the combination',
                    best_dispatching_rule='per instance the shortest of FIFO, MOR, SPT and MWKR '
                                          'under the non-delay mask',
                    procedure='scripts/p88_family_inference.py section_H (component, '
                              'iut_block, pooled_block, superiority_block) on the size 27x25; '
                              '20x25 and 30x25 read from the record and reproduced'))


# ----------------------------------------------------------------- outputs
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


def setting_words(res):
    """The setting for a printed header: environment and training seeds."""
    return f'environment {ENV}, training seeds {",".join(map(str, res["seeds"]))}'


def macro_lines(res):
    x = res['sizes'][SIZE]
    L = [f'% Generated by {GENERATOR} ({setting_words(res)}) -- do not hand-edit',
         f'% Per-floor batch, {SIZE}: the quantities of section H of '
         f'scripts/p88_family_inference.py, seed s against seed s, d = 100 (policy - other) / '
         f'other, B = {res["boot"]}, alpha = {res["alpha"]}. Par: the policy against the '
         f'specialist of the combination (hierarchical 95% CI, direction, verdicts at 0.5 and 1); '
         f'ParIut and ParPool over All, Train and Held; Rule: the policy against the best '
         f'dispatching rule (combined verdict, Holm over the eight combinations); RulePool: the '
         f'eight pooled. Twenty and Thirty: the same pooled rows at 20 and 30 modules, read from '
         f'the record of p88 and reproduced from the arrays.']
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    L.append(f'\\newcommand{{\\{PREFIX}N}}{{{x["n"]}}}'
             f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(res["seeds"])}}}'
             f'\\newcommand{{\\{PREFIX}Modules}}{{{SIZE.split("x")[0]}}}'
             f'  % {SIZE}: test instances per combination, training seeds, modules')
    for r in R8:
        L.append(F88.row_line(f'{PREFIX}Par{r}', x['parity'][r],
                              f'{SIZE}, policy vs specialist, {r}'))
    L += F88.iut_lines(f'{PREFIX}ParIut', x['parity_iut'], F88.PAR_FAM)
    for fam, y in x['parity_pooled'].items():
        L.append(F88.row_line(f'{PREFIX}ParPool{fam}', y, f'{SIZE}, policy vs specialist, '
                              f'pooled over {y["c"]} combinations'))
    for r in R8:
        y = x['rule_pool'][r]
        L.append(F88.row_line(f'{PREFIX}Rule{r}', y, f'{SIZE}, policy vs best dispatching rule, '
                              f'{r}; {F88.holm_text(y)}', margins=False, verdict=True))
    L.append(F88.row_line(f'{PREFIX}RulePool', x['rule_pool_pooled'],
                          f'{SIZE}, policy vs best dispatching rule, pooled over eight',
                          margins=False))
    L.append(F88.count_line(f'{PREFIX}Rule', x['rule_pool_counts'],
                            f'{SIZE}, policy vs best dispatching rule, of eight'))
    for size in SIDE:
        y = res['sizes'][size]
        w = PREFIX + SIDE_WORD[size]
        L.append(F88.row_line(f'{w}ParPoolAll', y['parity_pooled']['All'],
                              f'{size} (record of p88), policy vs specialist, pooled over eight'))
        L += F88.iut_lines(f'{w}ParIut', y['parity_iut'], {'All': R8})
        L.append(F88.row_line(f'{w}RulePool', y['rule_pool_pooled'],
                              f'{size} (record of p88), policy vs best dispatching rule, pooled '
                              f'over eight', margins=False))
        L.append(F88.count_line(f'{w}Rule', y['rule_pool_counts'],
                                f'{size} (record of p88), policy vs best dispatching rule, of '
                                f'eight'))
    return L


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    clash = sorted(x for x in names if x in have)
    if clash:
        raise Refused('macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def _ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def markdown(res):
    S = res['sizes']
    order = (SIDE[0], SIZE, SIDE[1])
    md = [f'# Per-floor batch, {SIZE} ({setting_words(res)})', '',
          f'Arrays of {SIZE}: `{res["folder"]}`; {S[SIZE]["n"]} test instances per combination. '
          f'{SIDE[0]} and {SIDE[1]}: the record `{res["record"]}` of p88, section H, reproduced '
          f'from the arrays in {res["reproduction"]["compared"]} numbers. B = {res["boot"]}.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          f'd = {res["definitions"]["d"]}. Verdicts at the margins 0.5% and 1% from the '
          'hierarchical 95% interval; combined verdict against the best dispatching rule with '
          'Holm over the eight combinations of a size.', '',
          '## Policy against the specialist of the combination', '',
          '| comb. | ' + ' | '.join(f'{s} | 0.5% | 1%' for s in order) + ' |',
          '|---|' + '---|---|---|' * len(order)]
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(
            f'{_ci(S[s]["parity"][r])} | {S[s]["parity"][r]["v0.5"]} | {S[s]["parity"][r]["v1.0"]}'
            for s in order) + ' |')
    for fam in F88.PAR_FAM:
        md.append(f'| pooled {fam} | ' + ' | '.join(
            f'{_ci(S[s]["parity_pooled"][fam])} | {S[s]["parity_pooled"][fam]["v0.5"]} | '
            f'{S[s]["parity_pooled"][fam]["v1.0"]}' for s in order) + ' |')
    md += ['', '| joint claim (IUT) | ' + ' | '.join(order) + ' |',
           '|---|' + '---|' * len(order)]
    for fam in F88.PAR_FAM:
        for mg in F88.MARGINS:
            for claim in ('equivalent', 'non-inferior'):
                k = f'{fam}|{mg}|{claim}'
                md.append(f'| {claim} on {fam} at +-{mg:g}% | ' + ' | '.join(
                    ('holds' if S[s]['parity_iut'][k]['holds'] else
                     'fails on ' + ','.join(S[s]['parity_iut'][k]['failing'])) for s in order)
                    + ' |')
    md += ['', '## Policy against the best dispatching rule', '',
           '| comb. | ' + ' | '.join(f'{s} | verdict' for s in order) + ' |',
           '|---|' + '---|---|' * len(order)]
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(
            f'{_ci(S[s]["rule_pool"][r])} | {S[s]["rule_pool"][r]["verdict"]}' for s in order)
            + ' |')
    md.append('| pooled | ' + ' | '.join(
        f'{_ci(S[s]["rule_pool_pooled"])} | {S[s]["rule_pool_pooled"]["direction"]}'
        for s in order) + ' |')
    cells = []
    for s in order:
        cells += [', '.join(str(S[s]['rule_pool_counts'][w]) for w in VERDICT_WORDS), '']
    md.append('| verdict counts (ahead, behind, not distinguishable) | ' + ' | '.join(cells)
              + ' |')
    md += ['', '## The per-floor batch between the two neighbouring sizes', '']
    for key, b in res['between'].items():
        md.append(f'- {key.replace("_", " ")}: {b["at_20"]:+.2f} at 20 modules, '
                  f'{b["at_27"]:+.2f} at 27, {b["at_30"]:+.2f} at 30; between the two: '
                  f'{"yes" if b["between"] else "no"}.')
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(dry=False):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md)."""
    check_setting()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(dry)
        lines = macro_lines(res)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    seeds = check_setting()
    st = input_state((SIZE,) + SIDE, seeds)
    print(f'p118: arrays of {SIZE}, {", ".join(SIDE)} (policy on eight combinations, each '
          f'specialist on its own, four dispatching rules), seeds {",".join(map(str, seeds))}: '
          + ('every one present with one row per test instance' if st['ready'] else st['reason']))
    for p in st['missing'][:6]:
        print(f'p118:   missing {p}')
    if len(st['missing']) > 6:
        print(f'p118:   ... and {len(st["missing"]) - 6} more missing')
    try:
        _, path = read_record(seeds)
        print(f'p118: record {path}: p88, {ENV} environment, these seeds, section H computed')
        rec_ok = True
    except Refused as e:
        print(f'p118: record refused: {e}')
        rec_ok = False
    return 0 if st['ready'] and rec_ok else 1


# ----------------------------------------------------------------- dry mode
# true differences in % (policy against specialist; policy against the best rule) per size
DRY_GAP = {'20x25': (0.4, -2.5), '27x25': (0.6, -2.0), '30x25': (0.8, -1.5)}
DRY_NOISE = 0.3


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


def build_synthetic(tree, boot=DRY_BOOT):
    """Arrays of the three sizes under <tree>/results and a record of p88 for 20x25 and 30x25
    under <tree>/out. Call with dry_paths(tree) in the environment."""
    seeds = check_setting()
    rng = np.random.default_rng(118)
    for size, (g_spec, g_rule) in DRY_GAP.items():
        n = F88.descriptor_count(size)
        if n == 0:
            raise Refused(f'data/FAMILY/{size} holds no descriptors')
        folder = REG.res_dir(size)
        os.makedirs(folder, exist_ok=True)
        for r in R8:
            spec = rng.uniform(300.0, 600.0, n)
            for k, s in enumerate(seeds):
                sp = np.round(spec * (1 + rng.normal(0, 0.005, n)), 1)
                pol = sp * (1 + (g_spec + rng.normal(0, DRY_NOISE, n)) / 100.0)
                for p, x in ((REG.result_path('specialist', r, s, size, r), sp),
                             (REG.result_path('released', None, s, size, r), pol)):
                    np.save(p, np.stack([x, np.zeros(n)], axis=1))
            # the rules: the best of the four sits g_rule % above the seed-mean policy
            pm = np.mean([np.load(REG.result_path('released', None, s, size, r))[:, 0]
                          for s in seeds], axis=0)
            best = pm / (1 + g_rule / 100.0)
            for q, extra in zip(F88.RULES, (0.0, 1.0, 2.0, 3.0)):
                np.save(f'{folder}/Result_{q}_{r}.npy',
                        np.stack([best * (1 + extra / 100.0), np.zeros(n)], axis=1))
    F88.configure(boot=boot, alpha=0.05)
    blk = F88.jsonable(section_h(SIDE, seeds))
    rec = dict(generator=RECORD_GENERATOR, setting=REG.describe(), seeds=seeds, boot=boot,
               alpha=0.05, sections={'H': dict(computed=True)}, larger_shops=blk)
    p = REG.out(RECORD)
    with open(p, 'w') as f:
        json.dump(rec, f, indent=1)
    return p


def dry_run(tree, boot=DRY_BOOT):
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with environment(**dry_paths(tree)):
        build_synthetic(tree, boot)
        return run(dry=True)


def dry_ok(res):
    """The synthetic differences are recovered: every pooled mean within 0.15 points."""
    x = res['sizes'][SIZE]
    return (abs(x['parity_pooled']['All']['mean'] - DRY_GAP[SIZE][0]) < 0.15 and
            abs(x['rule_pool_pooled']['mean'] - DRY_GAP[SIZE][1]) < 0.15 and
            all(res['between'][k]['between'] for k in res['between']))


def main():
    ap = argparse.ArgumentParser(description='the per-floor batch against 20 and 30 modules')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    a = ap.parse_args()
    os.chdir(ROOT)
    try:
        if a.check:
            sys.exit(check())
        if a.dry is None:
            REG.banner()
            res, names, md = run()
            print('\n'.join(md))
            print(f'\n{len(names)} macros; wrote ' + ', '.join(outputs().values()))
            return
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p118_dry_')
        try:
            res, names, md = dry_run(tree)
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    ok = dry_ok(res)
    if keep:
        print('\n'.join(md))
    x = res['sizes'][SIZE]
    print(f'p118 dry run on synthetic arrays {"passed" if ok else "FAILED"}: {len(names)} macros, '
          f'{res["reproduction"]["compared"]} record numbers reproduced; pooled policy vs '
          f'specialist {x["parity_pooled"]["All"]["mean"]:+.3f} (true {DRY_GAP[SIZE][0]:+.2f}), '
          f'vs best dispatching rule {x["rule_pool_pooled"]["mean"]:+.3f} (true '
          f'{DRY_GAP[SIZE][1]:+.2f})')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
