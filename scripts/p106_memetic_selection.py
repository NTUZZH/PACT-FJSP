r"""Selection of the configuration of the memetic algorithm: the numbers of the supplement
table tab:sn-memetic-sel and of the paragraph before it (manuscript/supp_new/sn05_baselines.tex,
Section sec:sn-memetic).

Candidates. Three configurations, numbered 1, 2 and 3 in the order of their record folders
A, B and C, each ran once on each of 24 cells: the first three instances (000, 001, 002) of
the production sets at 50 and 80 modules (50x25n30, 80x25n30) under the combinations N, S,
SW and LSW. The records were written by eval_memetic_family.py:
  <records>/<A|B|C>/<size>/<combination>/instance_<i>_seed1.json
The folder <records>/H holds the selected configuration started from one more schedule (a
schedule of a learned policy), which is another arm of the comparison. It is not a candidate,
and this script never opens it: it names the three candidate folders and walks only those.

Checks, each one a refusal (exit 1):
  - every candidate folder holds exactly the 24 records of the design;
  - every record: validated true and no violation; size, combination and instance equal to
    its path; random seed 1; budget_clock 'cpu' with budget_s 300 and no evaluation cap;
    cpu_s at least budget_s - 0.0005 s, the rounding of cpu_s to three decimals in the
    runner; exactly one starting schedule, the stored schedule of the best dispatching rule
    (one entry of seeds, from the record's warm_cache folder, whose makespan is the shortest
    of the four stored rule makespans, and an empty seed_from); a positive integer makespan
    equal to the objective;
  - in every cell the three candidates start from the same folder, rule and makespan;
  - one configuration per candidate: the stored configuration without its four derived values
    (DERIVED) is the same on the candidate's 24 records, and the derived values follow the
    rule of Memetic.__init__ in baselines/memetic/search.py from the number N of operations of
    the record's schedule: ts_no_improve = 100 for N <= 300 and int(ts_scale N) otherwise,
    ts_iter = 10 ts_no_improve, its_no_improve = max(1, ts_no_improve // 2), its_iter =
    10 its_no_improve. Every record has N > 300, where the table's I = floor(ts_scale N)
    holds;
  - the configuration keys whose values differ between the three candidates are exactly the
    five parameters the table prints (PRINTED), with values the table can print;
  - one candidate has the lowest printed score;
  - macro names are Msel plus letters, unique and unused by every other macro file of the
    paper folders, and every value is finite.
A refusal writes nothing and removes the outputs it would have written. With --from-json it
removes the markdown and the macros and keeps the JSON, which is its input.

Numbers. In a cell, best is the shortest makespan of the three candidates. The score of a
candidate is the mean over cells of 100 (makespan - best) / best: over the 24 cells, by size
and by combination. A cell in which several candidates reach best counts as a best cell for
each of them and is listed as a tie. Descriptive arithmetic on stored records, with no random
number and no timing.

Outputs, under --out (default the repository root):
  results/memetic_selection.json   the design; per cell and candidate the makespan, processor
                                   seconds, wall seconds, processor share, generations, seed,
                                   starting makespan, record path and sha256 of the record; per
                                   candidate the configuration, the values that differ between
                                   the three, the scores and the best cells
  results/memetic_selection.md     the makespans per cell and the scores, also by size and by
                                   combination
  manuscript/macros_memsel.tex      one \newcommand per number, names Msel<letters>
Macros (K = One, Two, Three, the configuration number of the table):
  Msel<K>Score            score in percent, two decimals
  Msel<K>ScoreBold        \bfseries when the printed score is the lowest printed score (every
                          one of a tie), empty otherwise
  Msel<K>Best             cells in which configuration K has the shortest makespan, a tie
                          counting for each candidate in it
  Msel<K>BestBold         \bfseries when Msel<K>Best is the largest count (every one of a tie)
  Msel<K>Children         children per generation (n_offspring)
  Msel<K>InitShare        budget share of the initial phase (init_frac), 'no limit' for None
  Msel<K>MachineMoves     machine moves of smallest estimate evaluated per tabu-search
                          iteration (ts_cap)
  Msel<K>TabuScale        factor s of I = floor(s N), the iterations without improvement that
                          stop a tabu search (ts_scale)
  Msel<K>ItsShare         share of the schedule evaluations spent on the iterated tabu search
                          (its_share)
  Msel<K>NoGenerationRuns runs in which the budget ended before the first generation of the
                          genetic phase (n_generations 0)
  MselRuns, MselInstances, MselBudget, MselSeed, MselTieRuns, MselSelected

  python scripts/p106_memetic_selection.py [--records <folder>] [--out <folder>]
  python scripts/p106_memetic_selection.py --from-json [--out <folder>]
Relative paths on the command line are read from the working directory.
"""
import argparse
import contextlib
import glob
import hashlib
import io
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p106_memetic_selection.py'
PREFIX = 'Msel'
MACRO_FILE = 'macros_memsel.tex'
OUTPUTS = (('macros', os.path.join('manuscript', MACRO_FILE)),
           ('json', os.path.join('results', 'memetic_selection.json')),
           ('md', os.path.join('results', 'memetic_selection.md')))
DEFAULT_RECORDS = os.path.join(ROOT, 'baselines', 'memetic', 'pilot_out', 'r2_sel')

# (record folder, configuration number of the table, macro word); the folder H is not here
CANDIDATES = (('A', 1, 'One'), ('B', 2, 'Two'), ('C', 3, 'Three'))
SIZES = ('50x25n30', '80x25n30')
SIZE_TEXT = {'50x25n30': '50 modules', '80x25n30': '80 modules'}
COMBINATIONS = ('N', 'S', 'SW', 'LSW')
INSTANCES = (0, 1, 2)
SEED = 1
BUDGET_S = 300.0
BUDGET_CLOCK = 'cpu'
BUDGET_TOL_S = 0.0005          # cpu_s is stored rounded to three decimals
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
DERIVED = ('ts_no_improve', 'ts_iter', 'its_no_improve', 'its_iter')
SMALL_N = 300                  # at or below this many operations ts_no_improve is 100
# the parameters whose values differ between the candidates, in the column order of the table:
# (configuration key, macro word, name in Table tab:sn-memetic)
PRINTED = (('n_offspring', 'Children', 'children per generation'),
           ('init_frac', 'InitShare', 'budget share of the initial phase'),
           ('ts_cap', 'MachineMoves', 'machine moves evaluated per iteration'),
           ('ts_scale', 'TabuScale', 'stop of a tabu search, I = floor(s N) with this s'),
           ('its_share', 'ItsShare', 'schedule evaluations in the iterated tabu search'))
BOLD = '\\bfseries'
CELLS = tuple((s, c, i) for s in SIZES for c in COMBINATIONS for i in INSTANCES)


class Refusal(Exception):
    """The records do not allow the analysis; args[0] is the list of reasons."""


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def record_rel(folder, cell):
    size, combo, i = cell
    return os.path.join(folder, size, combo, f'instance_{i:03d}_seed1.json')


def repo_rel(path):
    """A path relative to the repository root, or None when it lies outside the repository."""
    rel = os.path.relpath(os.path.abspath(path), ROOT)
    return None if rel == os.pardir or rel.startswith(os.pardir + os.sep) else rel


def output_paths(out_root):
    return {k: os.path.join(out_root, p) for k, p in OUTPUTS}


def shown(path, out_root):
    """An output path as printed: relative to --out."""
    return os.path.relpath(path, out_root)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

def is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def is_num(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def derived_values(cfg, n_ops):
    """The four derived values of Memetic.__init__ for a configuration and N operations."""
    ni = 100 if n_ops <= SMALL_N else int(cfg['ts_scale'] * n_ops)
    its_ni = max(1, ni // 2)
    return {'ts_no_improve': ni, 'ts_iter': 10 * ni, 'its_no_improve': its_ni,
            'its_iter': 10 * its_ni}


def check_record(rec, where, cell):
    """Problems of one record against the design, as a list of strings."""
    size, combo, i = cell
    bad = []
    if rec.get('validated') is not True:
        bad.append(f'{where}: validated is {rec.get("validated")!r}')
    if rec.get('violations') != []:
        bad.append(f'{where}: violations {str(rec.get("violations"))[:80]}')
    want = {'size': size, 'regime': combo, 'instance': f'instance_{i:03d}'}
    for k, v in want.items():
        if rec.get(k) != v:
            bad.append(f'{where}: {k} is {rec.get(k)!r}, the path says {v!r}')
    if rec.get('seed') != SEED or not is_int(rec.get('seed')):
        bad.append(f'{where}: seed {rec.get("seed")!r}, the design has {SEED}')
    if rec.get('budget_clock') != BUDGET_CLOCK:
        bad.append(f'{where}: budget_clock {rec.get("budget_clock")!r}, the design has '
                   f'{BUDGET_CLOCK!r}')
    if not is_num(rec.get('budget_s')) or float(rec['budget_s']) != BUDGET_S:
        bad.append(f'{where}: budget_s {rec.get("budget_s")!r}, the design has {BUDGET_S:g}')
    if rec.get('max_decodes') is not None:
        bad.append(f'{where}: evaluation cap {rec.get("max_decodes")!r}, the design has none')
    if not is_num(rec.get('cpu_s')) or float(rec['cpu_s']) < BUDGET_S - BUDGET_TOL_S:
        bad.append(f'{where}: cpu_s {rec.get("cpu_s")!r} does not reach the budget of '
                   f'{BUDGET_S:g} s')
    seeds = rec.get('seeds')
    if not isinstance(seeds, list) or len(seeds) != 1:
        bad.append(f'{where}: {len(seeds) if isinstance(seeds, list) else seeds!r} starting '
                   f'schedules, the design has one')
    else:
        s = seeds[0]
        rm = s.get('rule_makespans') or {}
        if s.get('source') != 'warm_cache' or s.get('folder') != rec.get('warm_cache') \
                or not rec.get('warm_cache'):
            bad.append(f'{where}: the starting schedule is not read from the warm_cache folder '
                       f'({s.get("source")!r}, {s.get("folder")!r})')
        if s.get('rule') not in RULES or sorted(rm) != sorted(RULES) \
                or not all(is_num(v) for v in rm.values()) \
                or not is_num(s.get('makespan')) \
                or float(s['makespan']) != min(float(v) for v in rm.values()):
            bad.append(f'{where}: the starting schedule is not the schedule of the best '
                       f'dispatching rule (rule {s.get("rule")!r}, makespan '
                       f'{s.get("makespan")!r}, rules {rm})')
    if rec.get('seed_from') not in ([], None):
        bad.append(f'{where}: further starting schedules from {rec.get("seed_from")!r}')
    ms = rec.get('makespan')
    if not is_int(ms) or ms <= 0:
        bad.append(f'{where}: makespan {ms!r} is not a positive integer')
    elif 'objective' in rec and rec['objective'] != ms:
        bad.append(f'{where}: objective {rec["objective"]!r} differs from makespan {ms}')
    cfg = rec.get('config')
    sched = rec.get('schedule') or {}
    n_ops = len(sched.get('op_mch') or [])
    if not isinstance(cfg, dict) or not all(k in cfg for k in DERIVED + ('ts_scale',)):
        bad.append(f'{where}: no configuration with {", ".join(DERIVED)} and ts_scale')
    elif n_ops <= SMALL_N:
        bad.append(f'{where}: the schedule has {n_ops} operations; the table needs more than '
                   f'{SMALL_N}')
    elif not is_num(cfg['ts_scale']):
        bad.append(f'{where}: ts_scale {cfg["ts_scale"]!r} is not a number')
    else:
        got, rule = {k: cfg[k] for k in DERIVED}, derived_values(cfg, n_ops)
        if got != rule:
            bad.append(f'{where}: derived values {got} do not follow the rule for '
                       f'N = {n_ops} ({rule})')
    return bad, n_ops


def read_records(rec_root):
    """(runs, candidate configurations) from the 72 records of the three candidate folders."""
    bad = []
    want = {f: {record_rel(f, c) for c in CELLS} for f, _, _ in CANDIDATES}
    for folder, _, _ in CANDIDATES:
        base = os.path.join(rec_root, folder)
        if not os.path.isdir(base):
            bad.append(f'no candidate folder {folder} under the records folder')
            continue
        found = {os.path.relpath(os.path.join(dp, f), rec_root)
                 for dp, _, fs in os.walk(base) for f in fs}
        for p in sorted(want[folder] - found):
            bad.append(f'missing record {p}')
        for p in sorted(found - want[folder]):
            bad.append(f'record or file outside the design: {p}')
    if bad:
        raise Refusal(bad)

    runs, base_cfg, derived = [], {}, {}
    for folder, num, _ in CANDIDATES:
        for cell in CELLS:
            rel = record_rel(folder, cell)
            path = os.path.join(rec_root, rel)
            try:
                with open(path) as f:
                    rec = json.load(f)
            except (OSError, ValueError) as e:
                bad.append(f'{rel}: cannot be read ({e})')
                continue
            problems, n_ops = check_record(rec, rel, cell)
            bad += problems
            if problems:
                continue
            cfg = rec['config']
            b = json.loads(json.dumps({k: v for k, v in cfg.items() if k not in DERIVED}))
            if folder in base_cfg and base_cfg[folder][0] != b:
                first, first_rel = base_cfg[folder]
                keys = sorted(k for k in set(b) | set(first) if b.get(k) != first.get(k))
                bad.append(f'{rel}: configuration differs from {first_rel} in {keys}')
            base_cfg.setdefault(folder, (b, rel))
            d = {k: cfg[k] for k in DERIVED}
            dz = derived.setdefault(folder, {}).setdefault(cell[0], dict(d, operations=n_ops))
            if dict(d, operations=n_ops) != dz:
                bad.append(f'{rel}: derived values or operations {dict(d, operations=n_ops)} '
                           f'differ from the other records of {cell[0]} ({dz})')
            s = rec['seeds'][0]
            runs.append(dict(candidate=num, folder=folder, size=cell[0], combination=cell[1],
                             instance=cell[2], makespan=rec['makespan'],
                             cpu_s=float(rec['cpu_s']), wall_s=rec.get('wall_s'),
                             cpu_share=rec.get('cpu_share'),
                             generations=rec.get('n_generations'), seed=rec['seed'],
                             start_folder=s['folder'], start_rule=s['rule'],
                             start_makespan=s['makespan'],
                             code=rec.get('code_sha256_16'), record=rel,
                             sha256=sha256_file(path)))
    if bad:
        raise Refusal(bad)
    configs = {folder: base_cfg[folder][0] for folder, _, _ in CANDIDATES}
    return runs, configs, derived


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def mean(xs):
    xs = list(xs)
    return math.fsum(xs) / len(xs)


def score_text(x):
    return f'{x:.2f}'


def pct_text(x):
    return f'{round(100.0 * x, 6):g}\\%'


PARAM_TEXT = {
    'n_offspring': lambda v: str(v),
    'init_frac': lambda v: 'no limit' if v is None else pct_text(v),
    'ts_cap': lambda v: str(v),
    'ts_scale': lambda v: f'{v:g}',
    'its_share': lambda v: pct_text(v),
}
PARAM_OK = {
    'n_offspring': lambda v: is_int(v) and v > 0,
    'init_frac': lambda v: v is None or (is_num(v) and 0 < v <= 1),
    'ts_cap': lambda v: is_int(v) and v > 0,
    'ts_scale': lambda v: is_num(v) and v > 0,
    'its_share': lambda v: is_num(v) and 0 <= v < 1,
}


def differing_keys(configs):
    keys = sorted(set().union(*(set(c) for c in configs.values())))
    return [k for k in keys
            if len({json.dumps(c.get(k, '<absent>'), sort_keys=True)
                    for c in configs.values()}) > 1]


def analyse(runs, configs, derived, records_root):
    """Every number of the outputs from the runs and the candidate configurations."""
    bad = []
    index = {}
    for r in runs:
        k = (r['candidate'], r['size'], r['combination'], r['instance'])
        if k in index:
            bad.append(f'two runs of configuration {k[0]} on {k[1]}/{k[2]}/{k[3]:03d}')
        index[k] = r
        where = f'configuration {r["candidate"]} on {r["size"]}/{r["combination"]}/' \
                f'{r["instance"]:03d}'
        if r['seed'] != SEED:
            bad.append(f'{where}: seed {r["seed"]!r}, the design has {SEED}')
        if not is_num(r['cpu_s']) or r['cpu_s'] < BUDGET_S - BUDGET_TOL_S:
            bad.append(f'{where}: cpu_s {r["cpu_s"]!r} does not reach the budget of '
                       f'{BUDGET_S:g} s')
        if not is_int(r['makespan']) or r['makespan'] <= 0:
            bad.append(f'{where}: makespan {r["makespan"]!r} is not a positive integer')
    nums = [n for _, n, _ in CANDIDATES]
    for cell in CELLS:
        for n in nums:
            if (n, *cell) not in index:
                bad.append(f'missing run of configuration {n} on {cell[0]}/{cell[1]}/'
                           f'{cell[2]:03d}')
    extra = [k for k in index if k[1:] not in CELLS or k[0] not in nums]
    if extra:
        bad.append(f'runs outside the design: {extra[:5]}')
    if bad:
        raise Refusal(bad)
    for cell in CELLS:
        start = {(index[(n, *cell)]['start_folder'], index[(n, *cell)]['start_rule'],
                  index[(n, *cell)]['start_makespan']) for n in nums}
        if len(start) != 1:
            bad.append(f'{cell[0]}/{cell[1]}/{cell[2]:03d}: the candidates start from '
                       f'different schedules {sorted(map(str, start))}')
    if sorted(configs) != sorted(f for f, _, _ in CANDIDATES):
        bad.append(f'configurations of {sorted(configs)}, the candidates are '
                   f'{[f for f, _, _ in CANDIDATES]}')
        raise Refusal(bad)
    diff = differing_keys(configs)
    printed = [k for k, _, _ in PRINTED]
    if sorted(diff) != sorted(printed):
        bad.append(f'the candidates differ in {diff}, the table prints {printed}')
    for k in printed:
        for f, n, _ in CANDIDATES:
            v = configs[f].get(k, '<absent>')
            if not PARAM_OK[k](v):
                bad.append(f'configuration {n}: {k} = {v!r} cannot be printed')
    if bad:
        raise Refusal(bad)

    cells, ties = [], []
    for cell in CELLS:
        ms = {n: index[(n, *cell)]['makespan'] for n in nums}
        best = min(ms.values())
        shortest = [n for n in nums if ms[n] == best]
        c = dict(size=cell[0], combination=cell[1], instance=cell[2], best=best,
                 makespan={str(n): ms[n] for n in nums}, shortest=shortest,
                 tie=len(shortest) > 1)
        cells.append(c)
        if c['tie']:
            ties.append(dict(size=cell[0], combination=cell[1], instance=cell[2],
                             candidates=shortest))
    parts = ([('all', 'all cells', lambda c: True)]
             + [(s, SIZE_TEXT[s], lambda c, s=s: c['size'] == s) for s in SIZES]
             + [(r, f'combination {r}', lambda c, r=r: c['combination'] == r)
                for r in COMBINATIONS])
    cand = []
    for f, n, w in CANDIDATES:
        excess = {(c['size'], c['combination'], c['instance']):
                  100.0 * (c['makespan'][str(n)] - c['best']) / c['best'] for c in cells}
        score = {key: mean(excess[(c['size'], c['combination'], c['instance'])]
                           for c in cells if keep(c))
                 for key, _, keep in parts}
        mine = [r for r in runs if r['candidate'] == n]
        cand.append(dict(
            number=n, word=w, folder=f, config=configs[f],
            derived=derived.get(f, {}),
            differing={k: configs[f][k] for k in printed},
            score=score, best=sum(n in c['shortest'] for c in cells),
            no_generation_runs=sum(r['generations'] == 0 for r in mine),
            generations_known=sum(is_int(r['generations']) for r in mine)))
    for x in cand:
        x['score_bold'] = score_text(x['score']['all']) == min(
            score_text(y['score']['all']) for y in cand)
        x['best_bold'] = x['best'] == max(y['best'] for y in cand)
    low = [x for x in cand if x['score_bold']]
    if len(low) != 1:
        raise Refusal([f'configurations {[x["number"] for x in low]} share the lowest '
                       f'printed score {score_text(low[0]["score"]["all"])}; no single '
                       f'configuration is selected'])

    def span(key):
        xs = [r[key] for r in runs if is_num(r.get(key))]
        return dict(min=min(xs), max=max(xs), n=len(xs)) if xs else None

    common = {k: v for k, v in configs[CANDIDATES[0][0]].items() if k not in diff}
    return dict(
        generator=GENERATOR,
        design=dict(sizes=list(SIZES), combinations=list(COMBINATIONS),
                    instances=list(INSTANCES), seed=SEED, budget_s=BUDGET_S,
                    budget_clock=BUDGET_CLOCK, budget_tolerance_s=BUDGET_TOL_S,
                    starting_schedule='the stored schedule of the best dispatching rule '
                                      '(warm_cache folder of the size), the only one',
                    candidates={str(n): f for f, n, _ in CANDIDATES},
                    not_a_candidate='H (the selected configuration with one more starting '
                                    'schedule, another arm; never read)'),
        records_root=repo_rel(records_root) if records_root else None,
        records=len(runs),
        score='mean over cells of 100 (makespan - best) / best, best = the shortest '
              'makespan of the three candidates in the cell; a tie counts for each',
        parts=[dict(key=k, text=t, cells=sum(1 for c in cells if keep(c)))
               for k, t, keep in parts],
        printed_parameters=[dict(key=k, word=w, name=t) for k, w, t in PRINTED],
        differing_keys=diff,
        common_config=common,
        candidates=cand,
        selected=low[0]['number'],
        cells=cells,
        ties=ties,
        runs=sorted(runs, key=lambda r: (SIZES.index(r['size']),
                                         COMBINATIONS.index(r['combination']),
                                         r['instance'], r['candidate'])),
        spans=dict(cpu_s=span('cpu_s'), wall_s=span('wall_s'), cpu_share=span('cpu_share')),
        code_fingerprints=sorted({str(r.get('code')) for r in runs}),
        checks=[
            'every candidate folder holds exactly the 24 records of the design',
            'validated true and no violation in every record',
            'size, combination and instance of every record equal to its path',
            f'random seed {SEED}, budget clock {BUDGET_CLOCK}, budget {BUDGET_S:g} s, no '
            f'evaluation cap, cpu_s at least {BUDGET_S:g} - {BUDGET_TOL_S:g} s',
            'one starting schedule, the stored schedule of the best dispatching rule, the same '
            'for the three candidates of a cell',
            'one configuration per candidate; derived values follow the rule of '
            'baselines/memetic/search.py from the number of operations, above 300 operations',
            'the differing configuration keys are the five parameters of the table'])


def recheck(R0):
    """Recompute the numbers from the JSON alone and refuse when they differ from it."""
    bad = []
    d = R0.get('design') or {}
    want = dict(sizes=list(SIZES), combinations=list(COMBINATIONS), instances=list(INSTANCES),
                seed=SEED, budget_s=BUDGET_S, budget_clock=BUDGET_CLOCK)
    for k, v in want.items():
        if d.get(k) != v:
            bad.append(f'the JSON design has {k} = {d.get(k)!r}, this script {v!r}')
    try:
        configs = {x['folder']: x['config'] for x in R0['candidates']}
        derived = {x['folder']: x.get('derived', {}) for x in R0['candidates']}
        runs = R0['runs']
    except (KeyError, TypeError) as e:
        raise Refusal([f'the JSON lacks {e}'])
    if bad:
        raise Refusal(bad)
    R = analyse(runs, configs, derived, None)
    R['records_root'] = R0.get('records_root')
    for k in ('candidates', 'cells', 'ties', 'selected', 'differing_keys', 'common_config'):
        if json.loads(json.dumps(R[k])) != R0.get(k):
            bad.append(f'the numbers recomputed from the runs differ from the JSON in {k!r}')
    if bad:
        raise Refusal(bad)
    return R


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def macro_lines(R):
    """[(name, value, comment)] of every macro."""
    L = []
    n_cells = len(R['cells'])
    for x in R['candidates']:
        w, who = PREFIX + x['word'], f'configuration {x["number"]} (record folder {x["folder"]})'
        L.append((f'{w}Score', score_text(x['score']['all']),
                  f'{who}: score, mean over the {n_cells} cells of the percentage by which its '
                  f'makespan exceeds the shortest makespan of the three configurations'))
        L.append((f'{w}ScoreBold', BOLD if x['score_bold'] else '',
                  f'{who}: bold marker of \\{w}Score, \\bfseries when its printed value is the '
                  f'lowest of the three'))
        L.append((f'{w}Best', str(x['best']),
                  f'{who}: cells of {n_cells} in which its makespan is the shortest of the '
                  f'three (a tie counts for each)'))
        L.append((f'{w}BestBold', BOLD if x['best_bold'] else '',
                  f'{who}: bold marker of \\{w}Best, \\bfseries when it is the largest count '
                  f'of the three'))
        for k, mw, name in PRINTED:
            L.append((f'{w}{mw}', PARAM_TEXT[k](x['differing'][k]),
                      f'{who}: {name} ({k} = {x["differing"][k]!r})'))
        L.append((f'{w}NoGenerationRuns', str(x['no_generation_runs']),
                  f'{who}: runs of {n_cells} in which the budget ended before the first '
                  f'generation of the genetic phase (n_generations 0)'))
    L += [
        (f'{PREFIX}Runs', str(n_cells),
         'selection runs per configuration, one per cell (instance and combination)'),
        (f'{PREFIX}Instances', str(len(INSTANCES)),
         'instances per production set and combination in the selection runs'),
        (f'{PREFIX}Budget', f'{BUDGET_S:g}',
         'budget of a selection run in seconds of processor time on one core'),
        (f'{PREFIX}Seed', str(SEED), 'random seed of every selection run'),
        (f'{PREFIX}TieRuns', str(len(R['ties'])),
         'cells in which two or more configurations share the shortest makespan'),
        (f'{PREFIX}Selected', str(R['selected']),
         'number of the configuration with the lowest score'),
    ]
    return L


def check_macro_names(lines, out_mac):
    """Every name Msel plus letters, unique, every value finite, and no other macro file of the
    repository's paper folders defines one of the names."""
    names = [x[0] for x in lines]
    bad = [n for n in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', n)]
    if bad:
        raise Refusal([f'macro names that are not {PREFIX} plus letters: {bad[:5]}'])
    if len(names) != len(set(names)):
        raise Refusal([f'duplicate macro names: '
                       f'{sorted({n for n in names if names.count(n) > 1})[:5]}'])
    nonfinite = [n for n, v, _ in lines
                 if re.search(r'(?<![A-Za-z])(nan|inf)(?![A-Za-z])', str(v), re.IGNORECASE)]
    if nonfinite:
        raise Refusal([f'non-finite macro values: {nonfinite[:5]}'])
    own = {os.path.abspath(out_mac), os.path.abspath(os.path.join(ROOT, OUTPUTS[0][1]))}
    files = sorted(set(glob.glob(os.path.join(ROOT, 'paper*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT, 'paper*', '*', 'macros*.tex'))
                       + glob.glob(os.path.join(ROOT, 'manuscript', 'sections_new', '*.tex'))))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    for f in files:
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            used = set(rx.findall(fh.read()))
        clash = set(names) & used
        pre = sorted(u for u in used if u.startswith(PREFIX))
        if clash or pre:
            raise Refusal([f'macro names of prefix {PREFIX} defined in '
                           f'{os.path.relpath(f, ROOT)}: {sorted(clash | set(pre))[:5]}'])
    return len(files)


def markdown(R):
    nums = [x['number'] for x in R['candidates']]
    d = R['design']
    md = ['# Selection of the configuration of the memetic algorithm', '',
          f'- generator: {R["generator"]}; records: '
          f'{R["records_root"] or "(a folder outside the repository)"}/<A|B|C>/<size>/'
          f'<combination>/instance_<i>_seed1.json, {R["records"]} records; the folder H is not '
          f'a candidate and is not read',
          f'- design: sizes {", ".join(d["sizes"])}; combinations '
          f'{" ".join(d["combinations"])}; instances '
          f'{", ".join(f"{i:03d}" for i in d["instances"])}; random seed {d["seed"]}; budget '
          f'{d["budget_s"]:g} s of processor time on one core (clock {d["budget_clock"]}); '
          f'{d["starting_schedule"]}',
          f'- score: {R["score"]}',
          f'- selected: configuration {R["selected"]}; ties for the shortest makespan: '
          f'{len(R["ties"])}', '',
          '## Candidates (values that differ; the others are common)', '',
          '| configuration | folder | ' + ' | '.join(k for k, _, _ in PRINTED)
          + ' | score (%) | best cells | runs without a generation |',
          '|---|---|' + '---:|' * (len(PRINTED) + 3)]
    for x in R['candidates']:
        md.append(f'| {x["number"]} | {x["folder"]} | '
                  + ' | '.join(json.dumps(x['differing'][k]) for k, _, _ in PRINTED)
                  + f' | {score_text(x["score"]["all"])} | {x["best"]} of {len(R["cells"])} | '
                    f'{x["no_generation_runs"]} of {x["generations_known"]} |')
    md += ['', 'Derived values by size (number of operations N, ts_no_improve = I):', '']
    for x in R['candidates']:
        for s, v in sorted(x['derived'].items()):
            md.append(f'- configuration {x["number"]}, {s}: N = {v["operations"]}, '
                      + ', '.join(f'{k} {v[k]}' for k in DERIVED))
    md += ['', f'Common configuration: `{json.dumps(R["common_config"], sort_keys=True)}`', '',
           '## Scores by part (%)', '',
           '| configuration | ' + ' | '.join(f'{p["text"]} ({p["cells"]})' for p in R['parts'])
           + ' |', '|---|' + '---:|' * len(R['parts'])]
    for x in R['candidates']:
        md.append(f'| {x["number"]} | '
                  + ' | '.join(f'{x["score"][p["key"]]:.3f}' for p in R['parts']) + ' |')
    md += ['', '## Makespan per cell', '',
           '| size | combination | instance | ' + ' | '.join(f'configuration {n}' for n in nums)
           + ' | shortest | best |', '|---|---|---|' + '---:|' * (len(nums) + 1) + '---|']
    for c in R['cells']:
        md.append(f'| {c["size"]} | {c["combination"]} | {c["instance"]:03d} | '
                  + ' | '.join(str(c['makespan'][str(n)]) for n in nums)
                  + f' | {c["best"]} | {", ".join(map(str, c["shortest"]))}'
                  + (' (tie)' if c['tie'] else '') + ' |')
    if R['ties']:
        md += ['', 'Ties for the shortest makespan:']
        md += [f'- {t["size"]}/{t["combination"]}/{t["instance"]:03d}: configurations '
               f'{", ".join(map(str, t["candidates"]))}' for t in R['ties']]
    sp = R['spans']
    md += ['', '## Runs', '']
    for k, text in (('cpu_s', 'processor seconds'), ('wall_s', 'wall seconds'),
                    ('cpu_share', 'processor share (processor over wall seconds)')):
        if sp.get(k):
            md.append(f'- {text}: {sp[k]["min"]:g} to {sp[k]["max"]:g} over {sp[k]["n"]} runs')
    md.append(f'- code fingerprints of the runs: {", ".join(R["code_fingerprints"])}')
    md += ['', 'Checks passed on the records:'] + [f'- {c}' for c in R['checks']]
    return '\n'.join(md) + '\n'


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def write_atomic(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write(text)
    os.replace(tmp, path)


def write_macros(path, lines):
    tmp = path + '.tmp'
    with contextlib.redirect_stdout(io.StringIO()):
        emit(tmp, lines, header='Selection of the configuration of the memetic algorithm on '
                                'the selection runs, written by '
                                f'{GENERATOR} from the records of the selection runs.')
    os.replace(tmp, path)


def remove(paths, keys):
    for k in keys:
        if os.path.exists(paths[k]):
            os.remove(paths[k])


def refuse(paths, keys, reasons):
    remove(paths, keys)
    print(f'REFUSING: {len(reasons)} problem(s)', file=sys.stderr)
    for r in reasons[:60]:
        print(f'  {r}', file=sys.stderr)
    if len(reasons) > 60:
        print(f'  ... and {len(reasons) - 60} more', file=sys.stderr)
    return 1


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--records', default=DEFAULT_RECORDS,
                    help='folder of the candidate folders A, B and C (read only)')
    ap.add_argument('--out', default=ROOT,
                    help='folder that receives results/ and manuscript/ (default the repository)')
    ap.add_argument('--from-json', action='store_true',
                    help='rebuild the markdown and the macros from results/memetic_selection.json '
                         'under --out, without the records')
    return ap


def main(argv=None):
    a = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    out_root = os.path.abspath(a.out)
    paths = output_paths(out_root)
    if a.from_json:
        keys = ('md', 'macros')
        if not os.path.exists(paths['json']):
            return refuse(paths, keys, [f'no {shown(paths["json"], out_root)} under --out'])
        try:
            with open(paths['json']) as f:
                R0 = json.load(f)
            R = recheck(R0)
            lines = macro_lines(R)
            check_macro_names(lines, paths['macros'])
        except ValueError as e:
            return refuse(paths, keys, [f'the JSON cannot be read ({e})'])
        except Refusal as e:
            return refuse(paths, keys, e.args[0])
    else:
        keys = ('json', 'md', 'macros')
        rec_root = os.path.abspath(a.records)
        try:
            runs, configs, derived = read_records(rec_root)
            R = analyse(runs, configs, derived, rec_root)
            lines = macro_lines(R)
            check_macro_names(lines, paths['macros'])
        except Refusal as e:
            return refuse(paths, keys, e.args[0])
        write_atomic(paths['json'], json.dumps(jsonable(R), indent=1) + '\n')
    write_atomic(paths['md'], markdown(R))
    write_macros(paths['macros'], lines)
    for k in keys:
        print(f'wrote {shown(paths[k], out_root)}')
    s = ', '.join(f'configuration {x["number"]} {score_text(x["score"]["all"])} '
                  f'(best in {x["best"]})' for x in R['candidates'])
    print(f'scores over {len(R["cells"])} cells: {s}; ties {len(R["ties"])}; selected '
          f'configuration {R["selected"]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
