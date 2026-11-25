r"""Selection of the solver's configuration on the selection runs: the numbers of the
supplement tables tab:sn-cpsat-sel and tab:sn-cp-model.

The solver CP-SAT receives one of eight candidate configurations, the rows 1 to 8 of
Table tab:sn-cpsat in the order of timed_campaign_plan.PILOT_ARMS. The selection runs are
the first instances (000-002) of S, SW, LSW and N at 50 and 80 modules, one 600-s run with
8 workers per candidate and instance, recorded in
  <records-root>/<size>_cparms/pilot/<candidate>/<R>/instance_<i>.json.
The script calls timed_campaign_plan.select_pilot (imported, not copied) on these records
with the frozen CPU-share bars and refuses, with the function's reasons, when the function
refuses. From its result and from the same records it computes, per candidate:
  (i)   the score of the selection rule over all runs and by part (50 modules, 80 modules,
        runs with setups, runs without setups), as select_pilot returns them;
  (ii)  the number of runs on which the candidate's reported makespan is the shortest of
        the eight; a tie counts for every candidate that reaches the minimum;
  (iii) the mean improvement over the warm start, 100 (warm - reported) / warm in percent,
        over all runs, by part, and by size and setup group (four groups);
  (iv)  the median, minimum and maximum of cpu_share and the median and maximum of
        peak_rss_gb, by size and setup group.
For the full model (row 2, ref_fullhint) it adds, by size and setup group, the median
number of variables of the initial model (p94_timed_inference.initial_variables; a record
without a count is reported as missing and never guessed) and the median build time; its
peak memory and improvement are those of (iii) and (iv). The reported makespan of a run is
the record's objective, the value select_pilot scores (the shorter of the warm start and
the best schedule found). Descriptive statistics only, no random number.

Outputs, under --out-root (default the repository root):
  manuscript/macros_solver_selection.tex   one \newcommand per number, names Csel<letters>
  results/solver_selection.json
  results/solver_selection.md
Macros (R = RowOne ... RowEight, the row of Table tab:sn-cpsat; P = All, Fifty, Eighty,
Setup, NoSetup; G = FiftyNoSetup, FiftySetup, EightyNoSetup, EightySetup):
  Csel<R>Score<P>            score of the selection rule, two decimals
  Csel<R>Score<P>Bold        \bfseries when the printed score is the lowest printed score of
                             its column (every one of a tie), empty otherwise
  Csel<R>Shortest            runs with the shortest reported makespan of the eight
  Csel<R>Improve<P>, Csel<R>Improve<G>
                             mean improvement over the warm start in percent, two decimals
  Csel<R>Share{Median|Min|Max}<G>   cpu_share, two decimals
  Csel<R>Memory{Median|Max}<G>      peak_rss_gb in GB, one decimal
  CselModelVars<G>, CselModelVarsMissing<G>, CselModelBuild<G>
                             full model: median variables (digits grouped as 562{,}796),
                             records without a count, median build seconds (two decimals)
  CselSelectedRow, CselSelectedConfiguration, CselRuns, CselInstances, CselBudgetMin,
  CselTieMargin, CselScoreLowest, CselScoreOthersMin, CselScoreOthersMax
A statistic without any value prints \textendash.

Refusal (exit 1): select_pilot refuses, the bar file is missing, a size has no macro
word, a record lacks a warm start or a CPU share, a record's objective differs from the
value select_pilot scored, or a macro name is malformed, repeated or uses the prefix of
another macro file. A refusal writes nothing and removes the three outputs if they exist.

  python scripts/p98_solver_selection.py [--out-root <folder>]
Relative paths given on the command line are read from the working directory; the default
records and bar file are those of the repository.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

# p94 imports p97, which imports p75_deploy_cost; p75 empties sys.argv and moves to the
# repository root at import. This script keeps its own command line and working directory
# from before the import, and resolves its paths from them.
_ARGV_KEPT, _CWD_KEPT = list(sys.argv), os.getcwd()
import numpy as np  # noqa: E402

import timed_campaign_plan as P  # noqa: E402
import p94_timed_inference as P94  # noqa: E402
from macro_io import emit  # noqa: E402
os.chdir(_CWD_KEPT)

PREFIX = 'Csel'
MACRO_FILE = 'macros_solver_selection.tex'
OUTPUTS = (('macros', os.path.join('manuscript', MACRO_FILE)),
           ('json', os.path.join('results', 'solver_selection.json')),
           ('md', os.path.join('results', 'solver_selection.md')))
BOLD = '\\bfseries'
NO_VALUE = '\\textendash'
ROW_WORDS = ('One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight')
# the plain words of each candidate in the table, from the text of Table tab:sn-cpsat
CANDIDATE_TEXT = {
    'ref': 'partial warm start',
    'ref_fullhint': 'full model',
    'ref_presolve_light_fullhint': 'lighter presolve',
    'ref_lnsonly_fullhint': "solver's neighborhood search",
    'compact': 'compact formulation',
    'fixassign': 'fixed machine assignment',
    'radius': 'restricted arcs',
    'lns': 'neighborhood search over sub-models',
}
FULL_MODEL = 'ref_fullhint'
SIZE_WORD = {s: P94.SIZE_WORD[s] for s in P.SIZES}
SIZE_TEXT = {'50x25n30': '50 modules', '80x25n30': '80 modules'}
GROUP_WORD = {'nosetup': 'NoSetup', 'setup': 'Setup'}
GROUP_TEXT = {'nosetup': 'without setups', 'setup': 'with setups'}


class Refusal(Exception):
    """The records do not allow the analysis; args[0] is the list of reasons."""


# ---------------------------------------------------------------------------
# numbers
# ---------------------------------------------------------------------------

def setup_group(regime):
    return 'setup' if regime in P.SETUP else 'nosetup'


def parts(cells, sizes):
    """[(score key of select_pilot, macro word, text, cells)] of the parts with runs."""
    out = [('all', 'All', 'all runs', list(cells))]
    out += [(s, SIZE_WORD[s], SIZE_TEXT[s], [c for c in cells if c[0] == s]) for s in sizes]
    out += [(g, GROUP_WORD[g], f'runs {GROUP_TEXT[g]}',
             [c for c in cells if setup_group(c[1]) == g]) for g in ('setup', 'nosetup')]
    return [p for p in out if p[3]]


def groups(cells, sizes):
    """[(key 'size|group', macro word, text, cells)] of the size and setup groups with runs."""
    out = []
    for s in sizes:
        for g in ('nosetup', 'setup'):
            cs = [c for c in cells if c[0] == s and setup_group(c[1]) == g]
            if cs:
                out.append((f'{s}|{g}', SIZE_WORD[s] + GROUP_WORD[g],
                            f'{SIZE_TEXT[s]}, {GROUP_TEXT[g]}', cs))
    return out


def stats(values):
    """Median, minimum and maximum (None without a value) and the number of values."""
    if not values:
        return dict(median=None, min=None, max=None, n=0)
    v = np.asarray(values, dtype=float)
    return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                n=int(v.size))


def improvement(warm, reported):
    return 100.0 * (warm - reported) / warm


def finite(x):
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def record_path(rec_root, folder, arm, cell):
    size, regime, i = cell
    return os.path.join(rec_root, f'{size}_cparms', folder, arm, regime, f'instance_{i:03d}.json')


def analyse(rec_root, bars, a):
    """All numbers of the outputs, or Refusal."""
    pre = []
    if not os.path.exists(bars):
        pre.append(f'no frozen CPU-share bar file {bars}')
    unknown = [s for s in a.sizes if s not in SIZE_WORD]
    if unknown:
        pre.append(f'sizes {unknown} have no macro word (known: {sorted(SIZE_WORD)})')
    if tuple(P.PILOT_ARMS) != tuple(CANDIDATE_TEXT) or len(P.PILOT_ARMS) != len(ROW_WORDS):
        pre.append(f'candidates {P.PILOT_ARMS} are not the eight rows of Table tab:sn-cpsat '
                   f'({tuple(CANDIDATE_TEXT)})')
    if pre:
        raise Refusal(pre)
    slots = [P.parse_cores(s) for s in a.slots]
    res = P.select_pilot(rec_root, list(a.sizes), slots, bars, budget=a.budget,
                         workers=a.workers, out=a.folder, regimes=tuple(a.regimes),
                         instances=tuple(a.instances))
    if res['refused']:
        raise Refusal(list(res['refused']))

    arms = list(P.PILOT_ARMS)
    cells = [(s, r, i) for s in a.sizes for r in a.regimes for i in a.instances]
    paths = {(m, c): record_path(rec_root, a.folder, m, c) for m in arms for c in cells}
    recs = {k: json.load(open(p)) for k, p in paths.items()}
    v = res['values']
    bad = []
    for k, r in recs.items():
        p = paths[k]
        if not finite(r.get('warm_makespan')) or float(r['warm_makespan']) <= 0:
            bad.append(f'no warm start makespan in {p}')
        if not finite(r.get('cpu_share')):
            bad.append(f'no cpu_share in {p}')
        if not finite(r.get('objective')) or float(r['objective']) != v[k]:
            bad.append(f'objective of {p} is not the value scored by the selection '
                       f'({r.get("objective")} against {v[k]})')
    if bad:
        raise Refusal(bad)

    warm = {k: float(r['warm_makespan']) for k, r in recs.items()}
    imp = {k: improvement(warm[k], v[k]) for k in recs}
    low = {c: min(v[m, c] for m in arms) for c in cells}
    shortest = {(m, c): v[m, c] <= low[c] + 1e-9 for m in arms for c in cells}
    P_ = parts(cells, a.sizes)
    G_ = groups(cells, a.sizes)

    rows = []
    for j, m in enumerate(arms):
        row = dict(row=j + 1, candidate=m, text=CANDIDATE_TEXT[m], word='Row' + ROW_WORDS[j],
                   scores={key: res['scores'][m][key] for key, *_ in P_},
                   shortest=sum(shortest[m, c] for c in cells),
                   improve={key: float(np.mean([imp[m, c] for c in cs])) for key, _, _, cs in P_},
                   improve_group={}, share={}, memory={}, memory_missing={})
        for key, _, _, cs in G_:
            row['improve_group'][key] = float(np.mean([imp[m, c] for c in cs]))
            row['share'][key] = stats([float(recs[m, c]['cpu_share']) for c in cs])
            mem = [(c, recs[m, c].get('peak_rss_gb')) for c in cs]
            row['memory'][key] = stats([float(x) for _, x in mem if finite(x)])
            row['memory_missing'][key] = [P.rel(paths[m, c]) for c, x in mem if not finite(x)]
        rows.append(row)

    # bold: the lowest printed score of each column, every one of a tie
    for r in rows:
        r['bold'] = []
    for key, *_ in P_:
        printed = {r['candidate']: float(score_text(r['scores'][key])) for r in rows}
        lowest = min(printed.values())
        for r in rows:
            if printed[r['candidate']] == lowest:
                r['bold'].append(key)

    model = {}
    for key, word, text, cs in G_:
        vars_, missing, build = [], [], []
        for c in cs:
            r = recs[FULL_MODEL, c]
            n, why = P94.initial_variables(r)
            if n is None:
                missing.append(dict(record=P.rel(paths[FULL_MODEL, c]), reason=why))
            else:
                vars_.append(n)
            if finite(r.get('build_s')):
                build.append(float(r['build_s']))
        model[key] = dict(runs=len(cs), variables=stats(vars_), variables_missing=missing,
                          build_s=stats(build),
                          build_missing=len(cs) - len(build))

    chosen = res['chosen']
    others = [res['scores'][m]['all'] for m in arms if m != chosen]
    budgets = {float(r['budget_s']) for r in recs.values()}
    return dict(
        inputs=dict(records_root=P.rel(rec_root), folder=a.folder, bars=P.rel(bars),
                    slots=[list(s) for s in slots], sizes=list(a.sizes),
                    regimes=list(a.regimes), instances=list(a.instances),
                    budget_s=a.budget, workers=a.workers, tie_margin_points=P.TIE_PP,
                    records=len(recs), candidates=arms),
        selection=dict(chosen=chosen, chosen_row=arms.index(chosen) + 1,
                       chosen_text=CANDIDATE_TEXT[chosen], within_tie=list(res['within_tie']),
                       best_score=res['best_score'], others_min=min(others),
                       others_max=max(others), runs=res['cells'],
                       instances_per_combination=len(a.instances),
                       budget_min=sorted(budgets)[0] / 60.0,
                       scores=res['scores'],
                       scores_at={str(t): x for t, x in res['at'].items()}),
        parts=[dict(key=k, word=w, text=t, runs=len(cs)) for k, w, t, cs in P_],
        groups=[dict(key=k, word=w, text=t, runs=len(cs)) for k, w, t, cs in G_],
        rows=rows,
        full_model=dict(candidate=FULL_MODEL, row=arms.index(FULL_MODEL) + 1, groups=model),
        runs=[dict(size=c[0], combination=c[1], instance=c[2], candidate=m,
                   reported=v[m, c], warm=warm[m, c], improvement=imp[m, c],
                   shortest=bool(shortest[m, c]), record=P.rel(paths[m, c]))
              for c in cells for m in arms],
        checks=dict(same_warm_start_per_run=all(
            len({warm[m, c] for m in arms}) == 1 for c in cells)))


# ---------------------------------------------------------------------------
# text of the numbers
# ---------------------------------------------------------------------------

def score_text(x):
    return f'{x:.2f}'


def pct_text(x):
    s = f'{x:.2f}'
    return '0.00' if s == '-0.00' else s


def share_text(x):
    return f'{x:.2f}'


def memory_text(x):
    return f'{x:.1f}'


def build_text(x):
    return f'{x:.2f}'


def or_dash(x, text):
    return NO_VALUE if x is None else text(x)


def macro_lines(R):
    """[(name, value, comment)] of every macro."""
    L = []
    sel = R['selection']
    n = sel['runs']
    for r in R['rows']:
        w, who = PREFIX + r['word'], f'row {r["row"]} ({r["candidate"]}, {r["text"]})'
        for p in R['parts']:
            L.append((f'{w}Score{p["word"]}', score_text(r['scores'][p['key']]),
                      f'{who}: score of the selection rule over {p["text"]} ({p["runs"]} runs), '
                      f'mean percent above the shortest reported makespan of the eight'))
        for p in R['parts']:
            L.append((f'{w}Score{p["word"]}Bold', BOLD if p['key'] in r['bold'] else '',
                      f'{who}: bold marker of \\{w}Score{p["word"]}, \\bfseries when its printed '
                      f'value is the lowest of the column'))
        L.append((f'{w}Shortest', str(r['shortest']),
                  f'{who}: runs of {n} with the shortest reported makespan of the eight '
                  f'(ties count for each)'))
        for p in R['parts']:
            L.append((f'{w}Improve{p["word"]}', pct_text(r['improve'][p['key']]),
                      f'{who}: mean improvement over the warm start in percent, {p["text"]} '
                      f'({p["runs"]} runs)'))
        for g in R['groups']:
            k, gw, gt = g['key'], g['word'], g['text']
            L.append((f'{w}Improve{gw}', pct_text(r['improve_group'][k]),
                      f'{who}: mean improvement over the warm start in percent, {gt} '
                      f'({g["runs"]} runs)'))
        for g in R['groups']:
            k, gw, gt = g['key'], g['word'], g['text']
            st = r['share'][k]
            for s, sw, st_text in (('median', 'Median', 'median'), ('min', 'Min', 'minimum'),
                                   ('max', 'Max', 'maximum')):
                L.append((f'{w}Share{sw}{gw}', or_dash(st[s], share_text),
                          f'{who}: {st_text} CPU share over {st["n"]} runs, {gt}'))
            mt = r['memory'][k]
            for s, sw, st_text in (('median', 'Median', 'median'), ('max', 'Max', 'maximum')):
                L.append((f'{w}Memory{sw}{gw}', or_dash(mt[s], memory_text),
                          f'{who}: {st_text} peak memory in GB over {mt["n"]} of {g["runs"]} '
                          f'runs, {gt}'))
    fm = R['full_model']
    for g in R['groups']:
        x, gw, gt = fm['groups'][g['key']], g['word'], g['text']
        L.append((f'{PREFIX}ModelVars{gw}', or_dash(x['variables']['median'], P94.count_text),
                  f'full model (row {fm["row"]}), {gt}: median variables of the initial model '
                  f'over {x["variables"]["n"]} of {x["runs"]} runs'))
        L.append((f'{PREFIX}ModelVarsMissing{gw}', str(len(x['variables_missing'])),
                  f'full model (row {fm["row"]}), {gt}: runs whose record states no variable '
                  f'count of the initial model'))
        L.append((f'{PREFIX}ModelBuild{gw}', or_dash(x['build_s']['median'], build_text),
                  f'full model (row {fm["row"]}), {gt}: median seconds to build the model over '
                  f'{x["build_s"]["n"]} of {x["runs"]} runs'))
    L += [
        (f'{PREFIX}SelectedRow', str(sel['chosen_row']),
         f'row of Table tab:sn-cpsat selected by the rule ({sel["chosen"]})'),
        (f'{PREFIX}SelectedConfiguration', sel['chosen_text'],
         'plain words of the selected configuration'),
        (f'{PREFIX}Runs', str(n), 'selection runs per candidate configuration'),
        (f'{PREFIX}Instances', str(sel['instances_per_combination']),
         'instances per size and combination in the selection runs'),
        (f'{PREFIX}BudgetMin', f'{sel["budget_min"]:g}', 'budget of a selection run, minutes'),
        (f'{PREFIX}TieMargin', f'{R["inputs"]["tie_margin_points"]:g}',
         'margin in percentage points within which the earlier row wins'),
        (f'{PREFIX}ScoreLowest', score_text(sel['best_score']),
         'lowest score of the selection rule over all runs'),
        (f'{PREFIX}ScoreOthersMin', score_text(sel['others_min']),
         'lowest score over all runs among the seven other candidates'),
        (f'{PREFIX}ScoreOthersMax', score_text(sel['others_max']),
         'highest score over all runs among the seven other candidates'),
    ]
    return L


def check_macro_names(lines, out_mac):
    """Every name Csel<letters>, unique, every value finite, and no other macro file of
    manuscript/ or paper/ uses the prefix."""
    names = [x[0] for x in lines]
    bad = [n for n in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', n)]
    if bad:
        raise Refusal([f'macro names not of the form {PREFIX}<letters>: {bad[:5]}'])
    if len(names) != len(set(names)):
        seen, dup = set(), set()
        for n in names:
            (dup if n in seen else seen).add(n)
        raise Refusal([f'duplicate macro names: {sorted(dup)[:5]}'])
    nonfinite = [n for n, val, _ in lines if re.search(r'(?<![A-Za-z])(nan|inf)(?![A-Za-z])',
                                                        str(val), re.IGNORECASE)]
    if nonfinite:
        raise Refusal([f'non-finite macro values: {nonfinite[:5]}'])
    files = (glob.glob(os.path.join(ROOT, 'manuscript', 'macros*.tex'))
             + glob.glob(os.path.join(ROOT, 'manuscript', 'sections_new', 'macros*.tex'))
             + glob.glob(os.path.join(ROOT, 'paper', 'macros*.tex')))
    for f in files:
        if os.path.basename(f) == MACRO_FILE or os.path.abspath(f) == os.path.abspath(out_mac):
            continue
        used = re.findall(r'\\(?:newcommand|renewcommand|providecommand)\*?\{?\\(' + PREFIX
                          + r'[A-Za-z]*)', open(f).read())
        if used:
            raise Refusal([f'the prefix {PREFIX} is used by {P.rel(f)}: {sorted(set(used))[:5]}'])


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def markdown(R):
    sel, inp = R['selection'], R['inputs']
    pk = [p['key'] for p in R['parts']]
    md = ['# Selection of the solver configuration on the selection runs', '',
          f'- records: {inp["records_root"]}/<size>_cparms/{inp["folder"]}/<candidate>/<R>/'
          f'instance_<i>.json, {inp["records"]} records; sizes {", ".join(inp["sizes"])}; '
          f'combinations {" ".join(inp["regimes"])}; instances '
          f'{", ".join(f"{i:03d}" for i in inp["instances"])}; budget {inp["budget_s"]:g} s, '
          f'{inp["workers"]} workers; CPU-share bars {inp["bars"]}',
          f'- score: mean over the runs of a part of the percent by which the reported makespan '
          f'exceeds the shortest reported makespan of the eight candidates; the earliest row '
          f'within {inp["tie_margin_points"]:g} points of the lowest score over all runs is '
          f'selected',
          f'- selected: row {sel["chosen_row"]} ({sel["chosen"]}, {sel["chosen_text"]}); lowest '
          f'score {sel["best_score"]:.3f}; within the margin: {", ".join(sel["within_tie"])}; '
          f'the seven others {sel["others_min"]:.3f} to {sel["others_max"]:.3f}',
          f'- the warm start is the same for the eight candidates on every run: '
          f'{R["checks"]["same_warm_start_per_run"]}', '',
          '## Scores (three decimals, the format of `timed_campaign_plan.py select`)', '',
          '| row | candidate | ' + ' | '.join(pk) + ' | '
          + ' | '.join(f'all@{t}s' for t in sel['scores_at']) + ' |',
          '|---|---|' + '---:|' * (len(pk) + len(sel['scores_at']))]
    for r in R['rows']:
        md.append(f'| {r["row"]} | {r["candidate"]} | '
                  + ' | '.join(f'{r["scores"][k]:.3f}' for k in pk) + ' | '
                  + ' | '.join(f'{sel["scores_at"][t][r["candidate"]]["all"]:.3f}'
                               for t in sel['scores_at']) + ' |')
    md += ['', '## Printed scores (bold = lowest printed value of the column), shortest '
           'makespans and improvement over the warm start (%)', '',
           '| row | configuration | ' + ' | '.join(pk) + ' | shortest | '
           + ' | '.join(f'improvement {p["text"]}' for p in R['parts']) + ' |',
           '|---|---|' + '---:|' * (2 * len(pk) + 1)]
    for r in R['rows']:
        sc = [(f'**{score_text(r["scores"][k])}**' if k in r['bold']
               else score_text(r['scores'][k])) for k in pk]
        md.append(f'| {r["row"]} | {r["text"]} | ' + ' | '.join(sc)
                  + f' | {r["shortest"]} of {sel["runs"]} | '
                  + ' | '.join(pct_text(r['improve'][k]) for k in pk) + ' |')
    md += ['', '## By size and setup group', '',
           '| row | group | runs | improvement (%) | CPU share median [min, max] | '
           'peak memory GB median / max |', '|---|---|---:|---:|---|---|']
    for r in R['rows']:
        for g in R['groups']:
            k = g['key']
            s, m = r['share'][k], r['memory'][k]
            md.append(f'| {r["row"]} | {g["text"]} | {g["runs"]} | '
                      f'{pct_text(r["improve_group"][k])} | '
                      f'{or_dash(s["median"], share_text)} [{or_dash(s["min"], share_text)}, '
                      f'{or_dash(s["max"], share_text)}] | {or_dash(m["median"], memory_text)} / '
                      f'{or_dash(m["max"], memory_text)} |'.replace('\\textendash', '-'))
    fm = R['full_model']
    md += ['', f'## Full model (row {fm["row"]}, {fm["candidate"]})', '',
           '| group | runs | variables median [min, max] | missing counts | build s median | '
           'peak memory GB median / max | improvement (%) |',
           '|---|---:|---|---:|---:|---|---:|']
    row2 = R['rows'][fm['row'] - 1]
    for g in R['groups']:
        x, k = fm['groups'][g['key']], g['key']
        vv, m = x['variables'], row2['memory'][k]
        md.append((f'| {g["text"]} | {x["runs"]} | {or_dash(vv["median"], P94.count_text)} '
                   f'[{or_dash(vv["min"], P94.count_text)}, {or_dash(vv["max"], P94.count_text)}] '
                   f'| {len(x["variables_missing"])} | {or_dash(x["build_s"]["median"], build_text)} '
                   f'| {or_dash(m["median"], memory_text)} / {or_dash(m["max"], memory_text)} | '
                   f'{pct_text(row2["improve_group"][k])} |')
                  .replace('{,}', ',').replace('\\textendash', '-'))
    for g in R['groups']:
        for x in fm['groups'][g['key']]['variables_missing']:
            md.append(f'- no variable count: {x["record"]} ({x["reason"]})')
    return '\n'.join(md) + '\n'


def output_paths(out_root):
    return {k: os.path.join(out_root, p) for k, p in OUTPUTS}


def remove_outputs(paths):
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)
            print(f'removed {P.rel(p)} (refusing to certify)')


def refuse(paths, reasons):
    remove_outputs(paths)
    print(f'REFUSING: {len(reasons)} problem(s)', file=sys.stderr)
    for r in reasons[:60]:
        print(f'  {r}', file=sys.stderr)
    if len(reasons) > 60:
        print(f'  ... and {len(reasons) - 60} more', file=sys.stderr)
    return 1


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def write_outputs(paths, R, lines):
    os.makedirs(os.path.dirname(paths['json']), exist_ok=True)
    tmp = paths['json'] + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(jsonable(R), f, indent=1)
    os.replace(tmp, paths['json'])
    print(f'wrote {P.rel(paths["json"])}')
    with open(paths['md'], 'w') as f:
        f.write(markdown(R))
    print(f'wrote {P.rel(paths["md"])}')
    emit(paths['macros'], lines,
         header='Selection of the solver configuration on the selection runs, written by '
                'scripts/p98_solver_selection.py from the records of the selection runs.')


def build_parser():
    d = P.build_parser().parse_args(['select'])
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out-root', default=ROOT,
                    help='folder that receives manuscript/ and results/ (default the repository)')
    ap.add_argument('--records-root', default=os.path.join(ROOT, 'test_results', 'FAMILY'),
                    help='folder of the <size>_cparms record trees')
    ap.add_argument('--bars', default=os.path.join(ROOT, d.bars),
                    help='frozen CPU-share bars of the cell types (read only)')
    ap.add_argument('--slots', nargs='+', default=list(d.slots),
                    help='affinity masks reserved for the runs')
    ap.add_argument('--sizes', nargs='+', default=list(d.pilot_sizes))
    ap.add_argument('--regimes', nargs='+', default=list(P.PILOT_REGIMES))
    ap.add_argument('--instances', nargs='+', type=int, default=list(P.PILOT_INSTANCES))
    ap.add_argument('--budget', type=float, default=d.pilot_budget)
    ap.add_argument('--workers', type=int, default=d.pilot_workers)
    ap.add_argument('--folder', default=d.pilot_out, help='record folder of the selection runs')
    return ap


def main(argv=None):
    a = build_parser().parse_args(_ARGV_KEPT[1:] if argv is None else argv)
    cwd = os.getcwd()
    paths = output_paths(os.path.abspath(a.out_root))
    rec_root, bars = os.path.abspath(a.records_root), os.path.abspath(a.bars)
    os.chdir(ROOT)
    try:
        try:
            R = analyse(P.rel(rec_root), P.rel(bars), a)
            lines = macro_lines(R)
            check_macro_names(lines, paths['macros'])
        except Refusal as e:
            return refuse(paths, e.args[0])
        write_outputs(paths, R, lines)
        sel = R['selection']
        print(f'selected row {sel["chosen_row"]} ({sel["chosen"]}) over {sel["runs"]} runs; '
              f'lowest score {sel["best_score"]:.3f}, the seven others {sel["others_min"]:.3f} '
              f'to {sel["others_max"]:.3f}')
        return 0
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    sys.exit(main())
