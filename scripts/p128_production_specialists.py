r"""P128: the policy against the specialists at the production batches, 50 and 80 modules.

WHAT IS COMPUTED (exact environment, training seeds of arm_registry, PACT_SEEDS=301-305)
  For the 30-instance test sets 50x25n30 and 80x25n30 (data/FAMILY/<set>) the quantities that
  scripts/p118_per_floor.py reports for 27 modules, by section H of
  scripts/p88_family_inference.py itself (p88.section_H with its size lists set to the two sets,
  p88_setting below; nothing re-implemented):
    policy against the specialist of each combination: d = 100 (policy - specialist) /
      specialist, seed s against seed s; hierarchical 95% interval (fresh default_rng(11) per
      quantity), Bonferroni interval over eight, seed-level t interval, verdicts at the margins
      0.5% and 1%, direction; intersection-union tests and pooled rows over the families All,
      Train and Held (p88.iut_block, p88.pooled_block);
    policy against the best dispatching rule (per instance the best of FIFO, MOR, SPT and MWKR):
      the same d, the combined verdict ahead / behind / not distinguishable with Holm over the
      eight combinations (p88.superiority_block), the pooled row and the verdict counts.
  B and alpha are those of the stored record of p88.

INPUTS
  policy     arm_registry.res_dir(<set>) (test_results/FAMILY/<set>_xmask), the n30 campaign
  rules      arm_registry.rule_dir(<set>) (the same folder)
  specialist <results>/<set>_xmask_spec (scripts/run_production_specialists_eval.sh; folder
             scripts/production_specialists_plan.py TAG), each specialist on its own
             combination; section H reads it through p88's module reader `specialist`, which
             p88_setting replaces by spec_reader for the two sets only.

CONTROLS (each one refuses the run)
  (i)   every input array exists with one row per test instance of its set;
  (ii)  the arrays of the policy, the specialists and the four classical rules on the
        combinations that scripts/p102_scale_sensitivity.py (W, LW, SW, LSW) and
        scripts/p103_five_classes.py (S, LS, SW, LSW) hold at their level default equal those
        studies' arrays exactly (key makespan of their npz files), instance by instance;
  (iii) policy against the best classical rule: the family of Holm's correction differs (eight
        combinations here, four in p102 and p103), so the mean and the two interval ends of
        each combination are compared with the default level of their records
        (results/scale_sensitivity.json, results/five_classes.json; key pol_rule), within
        REPRO_TOL; the p-values and verdicts are not compared;
  (iv)  policy against the specialist on the same combinations: mean, interval ends, seed-level
        t interval ends and both margin verdicts equal the records (key pol_spec), within
        REPRO_TOL (the same estimator and the same fresh stream; no Holm family involved).

SIDE BY SIDE (the last table)
  The pooled rows at 10, 20, 27, 30, 50 and 80 modules: the policy against the specialist pooled
  over the eight combinations and the policy against the best classical rule pooled over the
  eight, with the verdict counts.
    10 modules    p88's record: parity.pooled.All and rule_pool (per combination); recomputed
                  from the stored 10x25 arrays (p88.parity_components, pooled_block,
                  superiority_block) and compared; the pooled rule row is not in the record and
                  is computed here by p88.pooled_block.
    20, 30        p88's record, key larger_shops, reproduced from the arrays by section H in
                  every number (p118.reproduce).
    27            p118's record arm_registry.out('results/per_floor.json'), reproduced from the
                  arrays of 27x25 by section H in every number.
    50, 80        this script.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_prodspec.tex        macro_io.emit; names Pbs<size word><quantity>, letters
                                       only, checked against every macro defined in a .tex file
                                       under manuscript/ (the output itself left out); p88's row
                                       formats; every line's comment names the set, the sign
                                       and denominator and the seeds
  results/production_specialists.json  the two sets, the controls and the side-by-side rows
  results/production_specialists.md    one table per set and the side-by-side table
  A refusal removes all three.

CHECK (--check, read only)
  exit 0 when every array of the two sets exists with one row per test instance, the record of
  p88 is that of this setting with section H computed, p118's record is that of this setting,
  and the records and the default-level arrays of p102 and p103 exist.

DRY MODE (--dry [DIR])
  The whole path on synthetic arrays in DIR (default a temporary folder, removed at the end):
  arrays of the policy, the specialists and the four dispatching rules at 10, 20, 27, 30, 50 and
  80 modules with known differences (DRY_GAP), the record of p88 (section H of 20x25 and 30x25 by
  scripts/p118_per_floor.py build_synthetic, section A rows of 10x25 by p88's functions), the
  record of p118 (p118 run on the synthetic arrays), and the default level of p102 and p103 (the
  synthetic arrays copied into their result layout, their records written by p102's component
  and superiority). B = DRY_BOOT.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p128_production_specialists.py
       ... python scripts/p128_production_specialists.py --check
       python scripts/p128_production_specialists.py --dry [DIR]
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
import p118_per_floor as P118  # noqa: E402
import production_specialists_plan as PS  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS  # noqa: E402

GENERATOR = 'scripts/p128_production_specialists.py'
PREFIX = 'Pbs'
OUT_FILES = {'macros': 'manuscript/macros_prodspec.tex',
             'json': 'results/production_specialists.json',
             'md': 'results/production_specialists.md'}
ENV = 'exact'
SIZES = tuple(PS.SETS)                       # ('50x25n30', '80x25n30')
SIZE_WORD = {'50x25n30': 'Fifty', '80x25n30': 'Eighty'}
SIDE_SIZES = ('10x25', '20x25', '27x25', '30x25')
MODULES = {'10x25': 10, '20x25': 20, '27x25': 27, '30x25': 30, '50x25n30': 50, '80x25n30': 80}
RECORD = P118.RECORD                         # results/family_inference.json (p88)
PER_FLOOR = P118.OUT_FILES['json']           # results/per_floor.json (p118)
PER_FLOOR_GENERATOR = P118.GENERATOR
REPRO_TOL = 1e-9
DRY_BOOT = 400
R8 = list(REG.COMBOS)
LEVEL = 'default'
DRY_TREE = None                              # set by dry_run: the synthetic studies' folder


class Refused(Exception):
    """A setting, an input or a record that this script does not accept."""


# ----------------------------------------------------------------- setting
def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the analysis reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')
    return list(REG.seeds())


def spec_path(size, combo, seed):
    """Array of the specialist of combo, seed, on its own combination of the set."""
    return (f'{REG.results_root()}/{size}{PS.TAG}/'
            f'Result_{REG.tag("specialist", combo, seed)}_{combo}.npy')


def spec_reader(s, folder, r, n=None):
    """p88's reader `specialist` for the two sets: the specialist of r on r, from the folder of
    scripts/run_production_specialists_eval.sh."""
    if folder not in SIZES:
        raise Refused(f'spec_reader serves {", ".join(SIZES)}, not {folder}')
    return REG.read_path(spec_path(folder, r, s), n)


@contextlib.contextmanager
def p88_setting(sizes):
    """p88's size lists of section H set to the given sets and its specialist reader set to
    spec_reader; everything restored after."""
    old = F88.SPEC_SIZES, F88.POOL_SIZES, F88.specialist
    F88.SPEC_SIZES, F88.POOL_SIZES, F88.specialist = tuple(sizes), (), spec_reader
    try:
        yield
    finally:
        F88.SPEC_SIZES, F88.POOL_SIZES, F88.specialist = old


def section_h(sizes, seeds):
    """{set: section H block} of p88 for the production sets."""
    with p88_setting(sizes):
        return F88.section_H(None, seeds)


def inputs(size, seeds):
    """Every array the run reads for one set: policy and specialist per combination and seed,
    the four classical rules per combination."""
    out = []
    for r in R8:
        for s in seeds:
            out.append(REG.result_path('released', None, s, size, r))
            out.append(spec_path(size, r, s))
        for q in F88.RULES:
            out.append(f'{REG.rule_dir(size)}/Result_{q}_{r}.npy')
    return out


def input_state(seeds):
    """Control (i): every input array present with one row per test instance."""
    miss, bad, n_arr = [], [], 0
    for size in SIZES:
        n = F88.descriptor_count(size)
        if n == 0:
            bad.append(f'data/FAMILY/{size}/desc/N holds no descriptor')
        for p in inputs(size, seeds):
            n_arr += 1
            if not os.path.exists(p):
                miss.append(p)
                continue
            rows = len(REG.read_path(p))
            if rows != n:
                bad.append(f'{p}: {rows} rows, the test set has {n}')
    why = []
    if miss:
        why.append(f'{len(miss)} of {n_arr} input arrays missing (first {miss[0]})')
    if bad:
        why.append(f'{len(bad)} input arrays with another length than the test set '
                   f'(first {bad[0]})')
    return dict(ready=not why, reason='; '.join(why) or None, arrays=n_arr, missing=miss,
                wrong_length=bad)


# ----------------------------------------------------------------- records
def read_record(seeds):
    try:
        return P118.read_record(seeds, P118.record_path())
    except P118.Refused as e:
        raise Refused(str(e))


def read_per_floor(seeds, dry=False):
    try:
        path = REG.out(PER_FLOOR)
    except SystemExit as e:
        raise Refused(str(e))
    if not os.path.exists(path):
        raise Refused(f'the record {path} does not exist (run scripts/p118_per_floor.py under '
                      f'PACT_OUT first)')
    try:
        rec = json.load(open(path))
    except (OSError, ValueError) as e:
        raise Refused(f'{path} is not readable JSON: {e}')
    if rec.get('generator') != PER_FLOOR_GENERATOR:
        raise Refused(f'{path} was written by {rec.get("generator")!r}, not {PER_FLOOR_GENERATOR}')
    if not str(rec.get('setting', '')).startswith(f'environment {ENV},'):
        raise Refused(f'{path} was written for {rec.get("setting")!r}, not the {ENV} environment')
    if [int(x) for x in rec.get('seeds') or []] != list(seeds):
        raise Refused(f'{path} was written with seeds {rec.get("seeds")}, the active seeds are '
                      f'{list(seeds)}')
    if bool(rec.get('dry')) != bool(dry):
        raise Refused(f'{path} is {"a dry run" if rec.get("dry") else "not a dry run"}')
    if P118.SIZE not in (rec.get('sizes') or {}):
        raise Refused(f'{path} holds no block of {P118.SIZE}')
    return rec, path


def studies():
    """The two studies whose default level is compared: their combinations, the paths of their
    default-level arrays and their records (under DRY_TREE in a dry run)."""
    import p102_scale_sensitivity as P102
    import p103_five_classes as P103
    base = DRY_TREE or ROOT
    r102 = os.path.join(DRY_TREE, 'study_p102') if DRY_TREE else P102.DEFAULT_ROOT
    r103 = os.path.join(DRY_TREE, 'study_p103') if DRY_TREE else P103.DEFAULT_ROOT
    return {
        'p102': dict(combos=tuple(P102.WIN_COMBOS), generator=P102.GENERATOR,
                     record=os.path.join(base, P102.OUT_FILES['json']), root=r102,
                     path=lambda size, combo, label: P102.result_path(r102, LEVEL, size, combo,
                                                                      label)),
        'p103': dict(combos=tuple(P103.COMBOS), generator=P103.GENERATOR,
                     record=os.path.join(base, P103.OUT_FILES['json']), root=r103,
                     path=lambda size, combo, label: P103.result_path(r103, ENV, LEVEL, size,
                                                                      combo, label)),
    }


def read_study(name, st, seeds, boot):
    p = st['record']
    if not os.path.exists(p):
        raise Refused(f'the record {p} of {name} does not exist')
    try:
        rec = json.load(open(p))
    except (OSError, ValueError) as e:
        raise Refused(f'{p} is not readable JSON: {e}')
    if rec.get('generator') != st['generator']:
        raise Refused(f'{p} was written by {rec.get("generator")!r}, not {st["generator"]}')
    if rec.get('partial'):
        raise Refused(f'{p} is a partial record')
    if [int(x) for x in rec.get('seeds') or []] != list(seeds):
        raise Refused(f'{p} was written with seeds {rec.get("seeds")}, the active seeds are '
                      f'{list(seeds)}')
    if not {'pact', 'spec', 'rules'} <= set(rec.get('arms') or []):
        raise Refused(f'{p} holds the arms {rec.get("arms")}, the comparison needs pact, spec '
                      f'and rules')
    if int(rec.get('B') or 0) != int(boot):
        raise Refused(f'{p} was computed with B = {rec.get("B")}, the record of p88 with {boot}')
    if rec.get('default_policy', 'run') != 'run':
        raise Refused(f'{p}: the default-level policy was read from the campaign arrays '
                      f'(--default_policy {rec.get("default_policy")}), so its arrays are not '
                      f'an independent evaluation')
    for size in SIZES:
        rows = (((rec.get('sizes') or {}).get(size) or {}).get('policy') or {}).get(LEVEL)
        if not rows or not all(set(st['combos']) <= set(rows['rows'].get(k) or {})
                               for k in ('pol_spec', 'pol_rule')):
            raise Refused(f'{p} holds no default-level rows pol_spec and pol_rule of {size}')
    return rec


# ----------------------------------------------------------------- controls
def labels(combo, seed):
    return {'policy': 'greedy+' + REG.PREFIX + REG.model('released', None, seed, ENV),
            'specialist': 'greedy+' + REG.PREFIX + REG.model('specialist', combo, seed, ENV)}


def study_array(path):
    if not os.path.exists(path):
        raise Refused(f'the default-level array {path} does not exist')
    with np.load(path) as z:
        return np.asarray(z['makespan'], dtype=float)


def control_arrays(seeds, sts):
    """Control (ii): {study: {arm: [arrays, makespans]}}; refuses on any difference."""
    out = {}
    for name, st in sts.items():
        cnt = {'policy': [0, 0], 'specialist': [0, 0], 'rules': [0, 0]}
        for size in SIZES:
            n = F88.descriptor_count(size)
            for c in st['combos']:
                pairs = []
                for s in seeds:
                    lb = labels(c, s)
                    pairs.append(('policy', REG.result_path('released', None, s, size, c),
                                  st['path'](size, c, lb['policy'])))
                    pairs.append(('specialist', spec_path(size, c, s),
                                  st['path'](size, c, lb['specialist'])))
                for q in F88.RULES:
                    pairs.append(('rules', f'{REG.rule_dir(size)}/Result_{q}_{c}.npy',
                                  st['path'](size, c, q)))
                for arm, mine, theirs in pairs:
                    a, b = REG.read_path(mine), study_array(theirs)
                    if len(a) != n or len(b) != n or not np.array_equal(a, b):
                        k = int(np.argmax(a != b)) if len(a) == len(b) else -1
                        raise Refused(f'control (ii): {mine} differs from {theirs} of {name} '
                                      f'({len(a)} and {len(b)} rows; first differing instance '
                                      f'{k})')
                    cnt[arm][0] += 1
                    cnt[arm][1] += n
        out[name] = {k: dict(arrays=v[0], makespans=v[1]) for k, v in cnt.items()}
    return out


def _close(a, b):
    if isinstance(b, str) or isinstance(a, str):
        return a == b
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= REPRO_TOL


RULE_KEYS = ('mean', 'lo', 'hi')
SPEC_KEYS = ('mean', 'lo', 'hi', 't_lo', 't_hi', 'v0.5', 'v1.0')


def control_records(blocks, sts, recs):
    """Controls (iii) and (iv): {study: {...}}; refuses on any difference."""
    out = {}
    for name, st in sts.items():
        rec = recs[name]
        n_rule = n_spec = 0
        dmax = 0.0
        for size in SIZES:
            rows = rec['sizes'][size]['policy'][LEVEL]['rows']
            for c in st['combos']:
                for key, mine, keys in (('pol_rule', blocks[size]['rule_pool'][c], RULE_KEYS),
                                        ('pol_spec', blocks[size]['parity'][c], SPEC_KEYS)):
                    theirs = rows[key][c]
                    for k in keys:
                        if not _close(mine[k], theirs[k]):
                            raise Refused(f'control ({"iii" if key == "pol_rule" else "iv"}): '
                                          f'{size} {c} {key} {k} is {mine[k]!r} here and '
                                          f'{theirs[k]!r} in {st["record"]}')
                        if not isinstance(theirs[k], str):
                            dmax = max(dmax, abs(float(mine[k]) - float(theirs[k])))
                        if key == 'pol_rule':
                            n_rule += 1
                        else:
                            n_spec += 1
        out[name] = dict(record=st['record'], combinations=list(st['combos']),
                         rule_values=n_rule, specialist_values=n_spec, max_abs_diff=dmax,
                         holm_family='eight combinations here, four in ' + name +
                                     ': p-values and combined verdicts not compared')
    return out


# ----------------------------------------------------------------- side by side
def pooled_side(rec, rec27, seeds, rpath=RECORD, ppath=PER_FLOOR):
    """The pooled rows of 10, 20, 27 and 30 modules from the records, each reproduced from the
    arrays; returns ({size: row}, reproduction summary)."""
    repro = {}
    try:
        repro['20x25,30x25'] = P118.reproduce(rec, seeds)
    except P118.Refused as e:
        raise Refused(str(e))
    mine27 = json.loads(json.dumps(F88.jsonable(P118.section_h((P118.SIZE,), seeds))))
    bad = []
    n27 = P118._cmp(mine27[P118.SIZE], rec27['sizes'][P118.SIZE], P118.SIZE, bad)
    if bad:
        raise Refused(f'section H of {P118.SIZE} recomputed from the arrays does not reproduce '
                      f'{PER_FLOOR} in {len(bad)} of {n27} numbers (first: {bad[0]})')
    repro['27x25'] = dict(compared=n27, identical=True, tolerance=REPRO_TOL)
    # 10 modules: section A of p88
    if not ((rec.get('parity') or {}).get('pooled') or {}).get('All') or \
            set(rec.get('rule_pool') or {}) != set(R8):
        raise Refused('the record of p88 holds no parity.pooled.All and rule_pool of 10x25')
    D_par, _ = F88.parity_components('10x25', seeds)
    pooled = json.loads(json.dumps(F88.jsonable(F88.pooled_block(D_par, F88.PAR_FAM))))
    D_rule = {r: F88.seed_pairs(F88.released, F88.pool_reader, '10x25', r, seeds=seeds)
              for r in R8}
    rule = json.loads(json.dumps(F88.jsonable(F88.superiority_block(D_rule))))
    bad = []
    n10 = P118._cmp(pooled, rec['parity']['pooled'], 'parity.pooled', bad)
    n10 += P118._cmp(rule, rec['rule_pool'], 'rule_pool', bad)
    if bad:
        raise Refused(f'section A rows of 10x25 recomputed from the arrays do not reproduce the '
                      f'record of p88 in {len(bad)} of {n10} numbers (first: {bad[0]})')
    repro['10x25'] = dict(compared=n10, identical=True, tolerance=REPRO_TOL)
    rule10 = json.loads(json.dumps(F88.jsonable(F88.pooled_block(D_rule, {'All': R8})['All'])))
    r_rec, r_pf = os.path.relpath(rpath, ROOT), os.path.relpath(ppath, ROOT)
    rows = {'10x25': dict(n=F88.descriptor_count('10x25'), par=rec['parity']['pooled']['All'],
                          rule=rule10,
                          counts={w: sum(rec['rule_pool'][r]['verdict'] == w for r in R8)
                                  for w in VERDICT_WORDS},
                          source=f'{r_rec} (p88 section A); the pooled rule row computed here '
                                 f'by p88.pooled_block from the 10x25 arrays')}
    for size in ('20x25', '30x25'):
        x = rec['larger_shops'][size]
        rows[size] = dict(n=x['n'], par=x['parity_pooled']['All'], rule=x['rule_pool_pooled'],
                          counts=x['rule_pool_counts'], source=f'{r_rec} (p88 section H)')
    x = rec27['sizes'][P118.SIZE]
    rows[P118.SIZE] = dict(n=x['n'], par=x['parity_pooled']['All'], rule=x['rule_pool_pooled'],
                           counts=x['rule_pool_counts'], source=f'{r_pf} (p118)')
    return rows, repro


# ----------------------------------------------------------------- analysis
def analyse(dry=False):
    seeds = check_setting()
    state = input_state(seeds)
    if not state['ready']:
        raise Refused(f'control (i), inputs incomplete: {state["reason"]}')
    rec, rpath = read_record(seeds)
    rec27, ppath = read_per_floor(seeds, dry)
    F88.configure(boot=rec['boot'], alpha=rec['alpha'])
    sts = studies()
    recs = {k: read_study(k, st, seeds, F88.B) for k, st in sts.items()}
    arrays = control_arrays(seeds, sts)
    blocks = json.loads(json.dumps(F88.jsonable(section_h(SIZES, seeds))))
    rec_ctl = control_records(blocks, sts, recs)
    side, repro = pooled_side(rec, rec27, seeds, rpath, ppath)
    for size in SIZES:
        x = blocks[size]
        side[size] = dict(n=x['n'], par=x['parity_pooled']['All'], rule=x['rule_pool_pooled'],
                          counts=x['rule_pool_counts'], source='this script')
    n_ctl2 = sum(v['makespans'] for x in arrays.values() for v in x.values())
    return dict(generator=GENERATOR, setting=REG.describe(), seeds=seeds, dry=bool(dry),
                boot=F88.B, alpha=F88.ALPHA, record=rpath, per_floor_record=ppath,
                folders={s: dict(policy=REG.res_dir(s), rules=REG.rule_dir(s),
                                 specialists=os.path.dirname(spec_path(s, 'N', seeds[0])))
                         for s in SIZES},
                sizes=blocks,
                controls=dict(
                    rows=dict(arrays=state['arrays'], one_row_per_instance=True),
                    arrays=dict(per_study=arrays, makespans=n_ctl2, identical=True),
                    records=rec_ctl),
                side_by_side=dict(rows={s: side[s] for s in SIDE_SIZES + SIZES},
                                  reproduction=repro),
                definitions=dict(
                    d='100 (policy - other) / other per training seed and test instance, seed s '
                      'against seed s; negative = the policy is shorter',
                    specialist='the specialist trained on the combination, evaluated on it',
                    best_dispatching_rule='per instance the shortest of FIFO, MOR, SPT and MWKR '
                                          'under the non-delay mask',
                    procedure='scripts/p88_family_inference.py section_H (component, '
                              'iut_block, pooled_block, superiority_block) on 50x25n30 and '
                              '80x25n30, the specialists read from <set>_xmask_spec'))


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
    return f'environment {ENV}, training seeds {",".join(map(str, res["seeds"]))}'


def macro_lines(res):
    seeds = ','.join(map(str, res['seeds']))
    L = [f'% Generated by {GENERATOR} ({setting_words(res)}) -- do not hand-edit',
         f'% Production batches, 50x25n30 (Fifty) and 80x25n30 (Eighty), 30 test instances each: '
         f'the quantities of section H of scripts/p88_family_inference.py, d = 100 (policy - '
         f'other) / other in percent, the denominator the specialist or the per-instance best '
         f'of FIFO, MOR, SPT and MWKR, negative = the policy is shorter, training seed s against '
         f'seed s over seeds {seeds}, B = {res["boot"]}, alpha = {res["alpha"]}. Par: the policy '
         f'against the specialist of the combination (hierarchical 95% CI, direction, verdicts '
         f'at 0.5 and 1); ParIut and ParPool over All, Train and Held; Rule: the policy against '
         f'the best dispatching rule (combined verdict, Holm over the eight combinations); '
         f'RulePool: the eight pooled. TenRulePool: the pooled rule row at 10 modules (10x25, '
         f'computed from the arrays of p88 section A).']
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    L.append(f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(res["seeds"])}}}  % training seeds per arm')
    for size in SIZES:
        x = res['sizes'][size]
        w = PREFIX + SIZE_WORD[size]
        where = f'{size} ({x["n"]} instances, seeds {seeds})'
        sp = 'd = 100 (policy - specialist) / specialist, negative = policy shorter'
        ru = 'd = 100 (policy - best rule) / best rule, negative = policy shorter'
        L.append(f'\\newcommand{{\\{w}N}}{{{x["n"]}}}'
                 f'\\newcommand{{\\{w}Modules}}{{{MODULES[size]}}}'
                 f'  % {size}: test instances per combination, modules')
        for r in R8:
            L.append(F88.row_line(f'{w}Par{r}', x['parity'][r],
                                  f'{where}, policy vs specialist, {r}; {sp}'))
        L += F88.iut_lines(f'{w}ParIut', x['parity_iut'], F88.PAR_FAM)
        for fam, y in x['parity_pooled'].items():
            L.append(F88.row_line(f'{w}ParPool{fam}', y, f'{where}, policy vs specialist, '
                                  f'pooled over {y["c"]} combinations ({fam}); {sp}'))
        for r in R8:
            y = x['rule_pool'][r]
            L.append(F88.row_line(f'{w}Rule{r}', y, f'{where}, policy vs best dispatching rule, '
                                  f'{r}; {ru}; {F88.holm_text(y)}', margins=False, verdict=True))
        L.append(F88.row_line(f'{w}RulePool', x['rule_pool_pooled'],
                              f'{where}, policy vs best dispatching rule, pooled over eight; {ru}',
                              margins=False))
        L.append(F88.count_line(f'{w}Rule', x['rule_pool_counts'],
                                f'{where}, policy vs best dispatching rule, of eight'))
    y = res['side_by_side']['rows']['10x25']
    L.append(F88.row_line(f'{PREFIX}TenRulePool', y['rule'],
                          f'10x25 ({y["n"]} instances, seeds {seeds}), policy vs best '
                          f'dispatching rule, pooled over eight; d = 100 (policy - best rule) / '
                          f'best rule, negative = policy shorter', margins=False))
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
    c = res['controls']
    md = [f'# Production batches, policy against the specialists ({setting_words(res)})', '',
          f'd = {res["definitions"]["d"]}. B = {res["boot"]}. Verdicts at the margins 0.5% and 1% '
          'from the hierarchical 95% interval; combined verdict against the best dispatching rule '
          'with Holm over the eight combinations of a set.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          f'Controls: {c["rows"]["arrays"]} input arrays with one row per test instance; '
          f'{c["arrays"]["makespans"]} makespans equal to the default level of p102 and p103 ('
          + '; '.join(f'{k}: ' + ', '.join(f'{a} {v["arrays"]} arrays' for a, v in x.items())
                      for k, x in c['arrays']['per_study'].items()) + '); '
          + '; '.join(f'{k}: {v["rule_values"]} values of the rule rows (mean and interval ends; '
                      f'Holm family of four there, of eight here) and {v["specialist_values"]} '
                      f'values of the specialist rows equal within {REPRO_TOL:g} (largest '
                      f'difference {v["max_abs_diff"]:.1e})' for k, v in c['records'].items())
          + '.', '']
    for size in SIZES:
        x = S[size]
        md += [f'## {size} ({x["n"]} test instances; specialists in `{res["folders"][size]["specialists"]}`)',
               '', '| comb. | policy vs specialist | 0.5% | 1% | policy vs best rule | verdict |',
               '|---|---|---|---|---|---|']
        for r in R8:
            p, q = x['parity'][r], x['rule_pool'][r]
            md.append(f'| {r} | {_ci(p)} | {p["v0.5"]} | {p["v1.0"]} | {_ci(q)} | {q["verdict"]} |')
        for fam in F88.PAR_FAM:
            p = x['parity_pooled'][fam]
            rule = (f'{_ci(x["rule_pool_pooled"])} | {x["rule_pool_pooled"]["direction"]}'
                    if fam == 'All' else ' | ')
            md.append(f'| pooled {fam} | {_ci(p)} | {p["v0.5"]} | {p["v1.0"]} | {rule} |')
        md.append('| verdict counts (ahead, behind, not distinguishable) | | | | '
                  + ', '.join(str(x['rule_pool_counts'][w]) for w in VERDICT_WORDS) + ' | |')
        md += ['', '| joint claim (IUT) | holds |', '|---|---|']
        for fam in F88.PAR_FAM:
            for mg in F88.MARGINS:
                for claim in ('equivalent', 'non-inferior'):
                    k = f'{fam}|{mg}|{claim}'
                    y = x['parity_iut'][k]
                    md.append(f'| {claim} on {fam} at +-{mg:g}% | '
                              + ('holds' if y['holds'] else 'fails on ' + ','.join(y['failing']))
                              + ' |')
        md.append('')
    sb = res['side_by_side']
    md += ['## Pooled rows by shop size', '',
           '| modules | instances | policy vs specialist, pooled over eight | 0.5% | 1% | '
           'policy vs best rule, pooled over eight | verdict counts (ahead, behind, nd) | source |',
           '|---|---|---|---|---|---|---|---|']
    for size, y in sb['rows'].items():
        md.append(f'| {MODULES[size]} | {y["n"]} | {_ci(y["par"])} | {y["par"]["v0.5"]} | '
                  f'{y["par"]["v1.0"]} | {_ci(y["rule"])} | '
                  + ', '.join(str(y['counts'][w]) for w in VERDICT_WORDS) + f' | {y["source"]} |')
    md += ['', 'Reproduction of the stored rows from the arrays: '
           + '; '.join(f'{k} {v["compared"]} numbers' for k, v in sb['reproduction'].items())
           + f', all within {REPRO_TOL:g}.']
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
    st = input_state(seeds)
    print(f'p128: arrays of {", ".join(SIZES)} (policy on eight combinations, each specialist on '
          f'its own, four dispatching rules), seeds {",".join(map(str, seeds))}: '
          + (f'{st["arrays"]} present with one row per test instance' if st['ready']
             else st['reason']))
    for p in st['missing'][:6]:
        print(f'p128:   missing {p}')
    if len(st['missing']) > 6:
        print(f'p128:   ... and {len(st["missing"]) - 6} more missing')
    ok = st['ready']
    for what, fn in (('record of p88', lambda: read_record(seeds)[1]),
                     ('record of p118', lambda: read_per_floor(seeds)[1])):
        try:
            print(f'p128: {what} {fn()}: this setting')
        except Refused as e:
            print(f'p128: {what} refused: {e}')
            ok = False
    try:
        rec = read_record(seeds)[0]
        boot = rec['boot']
    except Refused:
        boot = F88.B
    for name, s in studies().items():
        try:
            read_study(name, s, seeds, boot)
            miss = [s['path'](size, c, lb) for size in SIZES for c in s['combos']
                    for x in seeds for lb in labels(c, x).values()
                    if not os.path.exists(s['path'](size, c, lb))]
            miss += [s['path'](size, c, q) for size in SIZES for c in s['combos']
                     for q in F88.RULES if not os.path.exists(s['path'](size, c, q))]
            print(f'p128: {name}: record {s["record"]} usable; default-level arrays '
                  + (f'{len(miss)} missing (first {miss[0]})' if miss else 'present'))
            ok = ok and not miss
        except Refused as e:
            print(f'p128: {name} refused: {e}')
            ok = False
    return 0 if ok else 1


# ----------------------------------------------------------------- dry mode
# true differences in % (policy against specialist; policy against the best rule) per size
DRY_GAP = {'10x25': (0.1, -1.8), '50x25n30': (0.5, -3.0), '80x25n30': (0.7, -3.5)}
DRY_NOISE = 0.3


@contextlib.contextmanager
def dry_setting(tree):
    global DRY_TREE
    old = DRY_TREE
    DRY_TREE = tree
    try:
        with P118.environment(**P118.dry_paths(tree)):
            yield
    finally:
        DRY_TREE = old


def _synthetic_size(size, rng, seeds, gaps, spec_of):
    n = F88.descriptor_count(size)
    if n == 0:
        raise Refused(f'data/FAMILY/{size} holds no descriptors')
    g_spec, g_rule = gaps
    folder = REG.res_dir(size)
    os.makedirs(folder, exist_ok=True)
    for r in R8:
        spec = rng.uniform(300.0, 600.0, n)
        for s in seeds:
            sp = np.round(spec * (1 + rng.normal(0, 0.005, n)), 1)
            pol = np.round(sp * (1 + (g_spec + rng.normal(0, DRY_NOISE, n)) / 100.0), 1)
            for p, x in ((spec_of(size, r, s), sp),
                         (REG.result_path('released', None, s, size, r), pol)):
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, np.stack([x, np.zeros(n)], axis=1))
        pm = np.mean([REG.read_path(REG.result_path('released', None, s, size, r))
                      for s in seeds], axis=0)
        best = np.round(pm / (1 + g_rule / 100.0), 1)
        for q, extra in zip(F88.RULES, (0.0, 1.0, 2.0, 3.0)):
            np.save(f'{folder}/Result_{q}_{r}.npy',
                    np.stack([np.round(best * (1 + extra / 100.0), 1), np.zeros(n)], axis=1))


def _synthetic_studies(seeds, boot):
    """The default level of p102 and p103: the synthetic arrays in their layout, and records
    computed by p102's component and superiority."""
    import p102_scale_sensitivity as P102
    st = P102._stats()
    SF = st[0]
    for name, s in studies().items():
        sizes = {}
        for size in SIZES:
            n = F88.descriptor_count(size)
            rows = {'pol_spec': {}, 'pol_rule': {}}
            D_rule = {}
            for c in s['combos']:
                arrs = {}
                for x in seeds:
                    lb = labels(c, x)
                    arrs[lb['policy']] = REG.read_path(REG.result_path('released', None, x, size, c))
                    arrs[lb['specialist']] = REG.read_path(spec_path(size, c, x))
                for q in F88.RULES:
                    arrs[q] = REG.read_path(f'{REG.rule_dir(size)}/Result_{q}_{c}.npy')
                for lb, a in arrs.items():
                    p = s['path'](size, c, lb)
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    np.savez(p, makespan=a)
                best = np.min(np.stack([arrs[q] for q in F88.RULES]), axis=0)
                pol = np.stack([arrs[labels(c, x)['policy']] for x in seeds])
                spc = np.stack([arrs[labels(c, x)['specialist']] for x in seeds])
                rows['pol_spec'][c] = P102.component(SF, 100.0 * (pol - spc) / spc, boot)[0]
                D_rule[c] = 100.0 * (pol - best[None, :]) / best[None, :]
                assert pol.shape == (len(seeds), n)
            rows['pol_rule'] = P102.superiority(st, D_rule, boot)
            sizes[size] = {'policy': {LEVEL: {'rows': rows}}}
        rec = dict(generator=s['generator'], partial=False, arms=['pact', 'spec', 'rules'],
                   seeds=list(seeds), B=boot, sizes=sizes)
        if name == 'p103':
            rec['default_policy'] = 'run'
        os.makedirs(os.path.dirname(s['record']), exist_ok=True)
        with open(s['record'], 'w') as f:
            json.dump(F88.jsonable(rec), f, indent=1)


def build_synthetic(tree, boot=DRY_BOOT):
    """Every input of the run under <tree>. Call inside dry_setting(tree)."""
    seeds = check_setting()
    rec_path = P118.build_synthetic(tree, boot)          # 20, 27, 30 and the record of p88
    rng = np.random.default_rng(128)
    _synthetic_size('10x25', rng, seeds, DRY_GAP['10x25'],
                    lambda size, r, s: REG.result_path('specialist', r, s, size, r))
    for size in SIZES:
        _synthetic_size(size, rng, seeds, DRY_GAP[size], spec_path)
    F88.configure(boot=boot, alpha=0.05)
    D_par, _ = F88.parity_components('10x25', seeds)
    D_rule = {r: F88.seed_pairs(F88.released, F88.pool_reader, '10x25', r, seeds=seeds)
              for r in R8}
    rec = json.load(open(rec_path))
    rec['parity'] = dict(pooled=F88.jsonable(F88.pooled_block(D_par, F88.PAR_FAM)))
    rec['rule_pool'] = F88.jsonable(F88.superiority_block(D_rule))
    with open(rec_path, 'w') as f:
        json.dump(rec, f, indent=1)
    P118.run(dry=True)                                   # the record of p118
    _synthetic_studies(seeds, boot)
    return rec_path


def dry_run(tree, boot=DRY_BOOT):
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with dry_setting(tree):
        build_synthetic(tree, boot)
        return run(dry=True)


def dry_ok(res):
    """The synthetic differences are recovered: every pooled mean within 0.15 points, and every
    control compared a positive number of values."""
    ok = True
    for size in SIZES:
        x = res['sizes'][size]
        ok &= abs(x['parity_pooled']['All']['mean'] - DRY_GAP[size][0]) < 0.15
        ok &= abs(x['rule_pool_pooled']['mean'] - DRY_GAP[size][1]) < 0.15
    c = res['controls']
    ok &= c['arrays']['makespans'] > 0
    ok &= all(v['rule_values'] > 0 and v['specialist_values'] > 0
              for v in c['records'].values())
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description='the policy against the specialists at 50 and 80 '
                                             'modules')
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
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p128_dry_')
        try:
            res, names, md = dry_run(tree)
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except (Refused, P118.Refused) as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    ok = dry_ok(res)
    if keep:
        print('\n'.join(md))
    c = res['controls']
    print(f'p128 dry run on synthetic arrays {"passed" if ok else "FAILED"}: {len(names)} macros; '
          + '; '.join(f'{s}: pooled policy vs specialist '
                      f'{res["sizes"][s]["parity_pooled"]["All"]["mean"]:+.3f} (true '
                      f'{DRY_GAP[s][0]:+.2f}), vs best dispatching rule '
                      f'{res["sizes"][s]["rule_pool_pooled"]["mean"]:+.3f} (true '
                      f'{DRY_GAP[s][1]:+.2f})' for s in SIZES)
          + f'; controls: {c["rows"]["arrays"]} arrays of the right length, '
          f'{c["arrays"]["makespans"]} makespans equal to p102 and p103, '
          + ', '.join(f'{k} {v["rule_values"]} rule and {v["specialist_values"]} specialist '
                      f'values' for k, v in c['records'].items())
          + '; reproduced: ' + ', '.join(f'{k} {v["compared"]}' for k, v in
                                          res['side_by_side']['reproduction'].items()))
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
