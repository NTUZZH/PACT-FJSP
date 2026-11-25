"""The time-budget figure fq3 of scripts/make_figures.py with --memetic-parallel: the
memetic algorithm of row Sel drawn as the best of eight runs on eight cores, from the curves
file of p124_memetic_parallel.py.

  * without the option the figure draws what it drew before: the memetic curves are the
    single run of the timed CSV, and passing no record gives the same artists;
  * with a full record, variant A draws the two best-of-eight curves and their bands in the
    colour and line styles of the memetic algorithm and no single-run memetic curve, the
    supplement companion its policy-started curve, variant B its single run (row All); every
    other curve, band and mark is the one drawn without the option;
  * every gate of the option refuses a planted defect: generator, run, dry run, clock,
    selected training run, curve grid, a time off the grid, the per-size times, n_runs, the
    method list, a method missing at one size, a rising curve, the start time, the curves file
    beside the record, and a value that disagrees with the decision table on eight cores; the
    run prints the count of the decision-table checks;
  * a preview of p124 is refused without --allow-partial; with it, only variant A is drawn,
    with the curve the preview holds and without the policy-started memetic curve, into files
    ending in _preview;
  * the command line passes the option through to the figure.

With --memetic-minute (the record of p137_memetic_one_minute.py, beside --memetic-parallel):
  * the reader accepts a full record and refuses a preview and a dry run (with or without
    --allow-partial), a record of another training run, instance count or bootstrap count than
    the record of p124, another generator or clock, a missing entry, a mean outside its
    interval and an entry over other combinations; without --memetic-parallel it is refused;
  * variant A holds, in each panel, two circles at x = 60 s with the record's means (filled:
    started from the best rule; open: from the policy's schedule), each over its interval bar
    in the memetic colour; a circle outside the value axis extends it; variant B and the
    companion are drawn as without the option, and the run prints the plotted values;
  * the legend of variant A holds nine entries in three columns of three rows, filled column
    by column, and the start keys carry the circle of their start;
  * without the option (or with None) the text content and the legend of every variant are
    unchanged;
  * the command line passes the option through, and refuses it without --memetic-parallel.

With --memetic-five (the record of p137 --budget 300, beside --memetic-minute):
  * the reader accepts a full five-minute record and refuses a preview and a dry run (with or
    without --allow-partial), the one-minute record given in its place, a record of another
    training run, instance count or bootstrap count, another generator, clock or budget, a
    missing entry, a mean outside its interval and an entry over other combinations; without
    --memetic-minute it is refused;
  * variant A holds, in each panel, two more circles at x = 300 s with the record's means, each
    over its interval bar, in the colour, face, size, bar width and z-order of the circles at
    60 s; every other artist is the one drawn without the option; a circle outside the value
    axis extends it; variant B and the companion do not change; the run prints both values with
    their intervals per size, read back from the record;
  * the third legend row names both budgets, with the same filled and open circle, in the
    column of the start keys; nothing else in the legend changes;
  * without the option (or with None) the figure is the figure with the one-minute circles alone;
  * a preview is drawn only through the keyword-only test switch of the drawing call, into
    files ending in _five_preview and never into manuscript/; the command line passes the record
    through, refuses it without --memetic-minute and refuses a preview.

Synthetic records only, written under the pytest temporary folder: a timed record of p94 on the
request clock, a record of p124 with its curves file, rows Pool only, and records of p137 read
at 60 s and at 300 s.
Nothing is read from the repository's records and no display is needed. Run from the
repository root on one core:
  taskset -c 3 python -B -m pytest -p no:cacheprovider --basetemp=<scratch>
      tests/test_fq3_time_budget.py
"""
import csv
import inspect
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import make_figures as M  # noqa: E402

SIZES = M.FQ3_SIZES
GRID = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 60.0, 120.0, 300.0, 600.0]
BUDGETS = (60, 300, 600)
SEL = 304
POLICY_TEXT = f'selected training run s{SEL}'
FIRST = {'50x25n30': (8.0, 1.5), '80x25n30': (16.0, 3.0)}   # policy_first_s, rule_first_s
SAMPLE_BUDGET = 120.0
HALF = 0.8                                                   # half width of every interval
# mean(t) = a - b ln(1 + t) from the first time of the arm; row All lies 0.3 higher
CURVES = {'memetic_rule': (2.0, 3.0, 1.2), 'cpsat_rule': (2.0, 3.5, 1.1),
          'memetic_policy': (10.0, -0.2, 0.7), 'cpsat_policy': (10.0, 0.0, 0.6)}
BEST_SHIFT = 0.5                                             # the best of eight lies lower
SINGLE_COLUMNS = ['run', 'row', 'origin', 'policy', 'size', 'group', 'combination', 'method',
                  'solver', 'start', 't_s', 'mean_pct', 'lo_pct', 'hi_pct', 'n_instances',
                  'n_combinations', 'n_runs', 'held', 'policy_budget_s', 'policy_first_s',
                  'rule_first_s']
ROSE = to_hex(M.FQ3_METHOD['memetic'][1])
GRID_GREY = to_hex(M.C_GRID)


# ================================================================== the synthetic records
def r6(x):
    return round(float(x), 6)


def f2(x):
    return f'{0.0 if x == 0 else x:+.2f}'


def arm_curve(arm, row='Sel', shift=0.0):
    t0, a, b = CURVES[arm]
    a += 0.3 if row == 'All' else 0.0
    t = [g for g in GRID if g >= t0]
    m = [r6(a - b * math.log1p(x) - shift) for x in t]
    return t, m


def sample_curve():
    t = [g for g in GRID if g >= 10.0]
    m, held = [], []
    for x in t:
        y = min(x, SAMPLE_BUDGET)
        m.append(r6(-0.8 - 0.3 * math.log(y / 10.0 + 1.0)))
        held.append('yes' if x > SAMPLE_BUDGET else 'no')
    return t, m, held


def row_of(size, method, solver, start, t, m, row='Sel', held='no', n_runs=1,
           run='full', policy=POLICY_TEXT):
    pf, rf = FIRST[size]
    return dict(run=run, row=row, origin='request', policy=policy if row == 'Sel' else 'all',
                size=size, group='all', combination='Pool', method=method, solver=solver,
                start=start, t_s=f'{t:g}', mean_pct=f'{m:.6f}', lo_pct=f'{m - HALF:.6f}',
                hi_pct=f'{m + HALF:.6f}', n_instances=9, n_combinations=8, n_runs=n_runs,
                held=held, policy_budget_s=f'{SAMPLE_BUDGET:g}', policy_first_s=f'{pf:.6f}',
                rule_first_s=f'{rf:.6f}')


def entry(t, m, b):
    v = m[t.index(float(b))]
    return dict(mean=v, lo=r6(v - HALF), hi=r6(v + HALF))


def timed_pair():
    """The timed CSV rows and JSON of p94, request clock, rows Pool of group all."""
    rows, results = [], {}
    for size in SIZES:
        S = dict(ix=dict(policy=dict(t=dict(median_s=FIRST[size][0])),
                         rules=dict(t_all=dict(median_s=FIRST[size][1]))),
                 policy_budget_s=SAMPLE_BUDGET, ii={}, vi={}, vii={})
        for arm, (solver, start) in M.FQ3_ARMS.items():
            for row in (('Sel', 'All') if start == 'rule' else ('Sel',)):
                t, m = arm_curve(arm, row)
                rows += [row_of(size, arm, solver, start, x, y, row) for x, y in zip(t, m)]
                for b in BUDGETS:
                    if start == 'policy':
                        S['ii'].setdefault(solver, {}).setdefault(str(b), {})['Pool'] = \
                            entry(t, m, b)
                    else:
                        key = 'selected' if row == 'Sel' else 'all_seeds'
                        S['vi'].setdefault(solver, {}).setdefault(str(b), {}).setdefault(
                            key, {})['Pool'] = entry(t, m, b)
        t, m, held = sample_curve()
        rows += [row_of(size, M.FQ3_SAMPLE, 'policy', 'policy', x, y, held=h)
                 for x, y, h in zip(t, m, held)]
        for b in BUDGETS:
            if b <= SAMPLE_BUDGET:
                S['vii'][str(b)] = dict(Pool=entry(t, m, b))
        results[size] = S
    J = dict(generator='scripts/p94_timed_inference.py', run='full', origin='request',
             sign=M.FQ3_SIGN + ' (synthetic)', selection=dict(seed=SEL),
             grid=dict(curve=GRID), budgets_per_size={s: list(BUDGETS) for s in SIZES},
             results=results)
    return rows, J


def mpar_pair(preview=False):
    """The record of p124 and its curves rows: the best of eight from each start (the first
    set alone in a preview)."""
    methods = ['memetic_rule_best8'] + ([] if preview else ['memetic_policy_best8'])
    rows, start_s, group_start_s, sizes = [], {}, {}, {}
    for size in SIZES:
        values = {b: {} for b in BUDGETS}
        for arm, method in M.FQ3_MPAR_METHODS.items():
            if method not in methods:
                continue
            t, m = arm_curve(arm, shift=BEST_SHIFT)
            rows += [row_of(size, method, 'memetic', M.FQ3_ARMS[arm][1], x, y, n_runs=8,
                            run='preview of the first set alone, not for the paper'
                            if preview else 'full') for x, y in zip(t, m)]
            start_s.setdefault(size, {})[method] = t[0]
            group_start_s.setdefault(size, {})[method] = dict(NoSetup=t[0], Setup=t[0])
            for b in BUDGETS:
                values[b][arm] = f2(m[t.index(float(b))])
        sizes[size] = dict(rows={str(b): dict(Pool=dict(values=dict(
            dict(rules='+3.00', policy_sample='-1.00', cpsat_rule='+0.00',
                 cpsat_policy='-2.00', memetic_policy=None), **values[b])))
            for b in BUDGETS})
    curves = dict(file='results/memetic_parallel_curves.csv', row='Sel', methods=methods,
                  n_runs=8, grid=GRID, start_s=start_s, group_start_s=group_start_s)
    J = dict(generator=M.FQ3_MPAR_GENERATOR, run='preview' if preview else 'full', dry=False,
             origin='request', selected_run=SEL,
             decision_table=dict(clock='request', budgets=list(BUDGETS), sizes=sizes,
                                 curves=curves))
    return rows, J


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SINGLE_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


@pytest.fixture
def files(tmp_path):
    """Paths of the timed pair and of a full record of p124 with its curves file."""
    rows, J = timed_pair()
    tc, tj = str(tmp_path / 'p94' / 'timed_curves.csv'), str(tmp_path / 'p94' / 'timed.json')
    write_csv(tc, rows)
    write_json(tj, J)
    return dict(tc=tc, tj=tj, tmp=tmp_path)


def put_mpar(tmp_path, name='full', preview=False, plant=None):
    rows, J = mpar_pair(preview)
    if plant is not None:
        plant(J, rows)
    mj = str(tmp_path / name / 'results' / 'memetic_parallel.json')
    write_json(mj, J)
    write_csv(os.path.join(os.path.dirname(mj), 'memetic_parallel_curves.csv'), rows)
    return mj


# ================================================================== drawing and its artists
def draw(files, monkeypatch, **kw):
    """Run fq3_time_budget with the figures kept open and nothing saved. Returns
    ({variant: [panel artists]}, written file names, the figures)."""
    figs, saved, close = [], [], M.plt.close
    monkeypatch.setattr(M.plt, 'close', lambda fig=None: figs.append(fig))
    monkeypatch.setattr(Figure, 'savefig', lambda self, path, **k: saved.append(path))
    try:
        written = M.fq3_time_budget(files['tc'], files['tj'], [], str(files['tmp'] / 'out'),
                                    **kw)
        stems = [os.path.basename(w)[:-len('.pdf')] for w in written if w.endswith('.pdf')]
        assert len(stems) == len(figs) and saved == written
        out = {}
        for stem, fig in zip(stems, figs):
            variant = stem[len('fq3_time_budget_'):].replace('_preview', '').replace('_dev', '')
            out[variant] = [panel_artists(ax) for ax in fig.axes]
    finally:
        for fig in figs:
            close(fig)
        monkeypatch.undo()
    return out, [os.path.basename(w) for w in written]


def panel_artists(ax):
    lines = [dict(color=to_hex(ln.get_color()), ls=str(ln.get_linestyle()),
                  marker=str(ln.get_marker()), xy=np.asarray(ln.get_xydata(), float))
             for ln in ax.get_lines()]
    bands = [dict(color=to_hex(c.get_facecolor()[0]), xy=np.asarray(c.get_paths()[0].vertices))
             for c in ax.collections if isinstance(c, PolyCollection)]
    return dict(lines=lines, bands=bands, ylim=ax.get_ylim(), xlim=ax.get_xlim())


def same(a, b):
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-12)


def rose_lines(panel):
    return [ln for ln in panel['lines'] if ln['color'] == ROSE]


def other_lines(panel):
    """The curves and marks that are not the memetic algorithm's (grid lines excluded, since
    they follow the value axis)."""
    return [ln for ln in panel['lines'] if ln['color'] not in (ROSE, GRID_GREY)]


def solid(ln):
    return ln['ls'] == '-'


def curve_xy(arm, row='Sel', shift=0.0):
    t, m = arm_curve(arm, row, shift)
    return np.column_stack([t, m])


# ================================================================== the default figure
def test_without_the_option_the_figure_is_todays(files, monkeypatch):
    """No record, or the record None: the same artists, and the memetic curves are the single
    run of the timed CSV."""
    plain, names = draw(files, monkeypatch)
    none, names2 = draw(files, monkeypatch, memetic_parallel=None)
    assert names == names2 == [f'fq3_time_budget_{v}.{e}' for v in ('A', 'B', 'B_supplement')
                               for e in ('pdf', 'png')]
    for v in plain:
        for p, q in zip(plain[v], none[v]):
            assert len(p['lines']) == len(q['lines']) and len(p['bands']) == len(q['bands'])
            assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['ls'] == y['ls']
                       for x, y in zip(p['lines'], q['lines']))
            assert all(same(x['xy'], y['xy']) for x, y in zip(p['bands'], q['bands']))
            assert p['ylim'] == q['ylim'] and p['xlim'] == q['xlim']
    # panel (a) of variant A: the single run of each start
    rose = rose_lines(plain['A'][0])
    assert len(rose) == 2
    assert same([ln for ln in rose if solid(ln)][0]['xy'], curve_xy('memetic_rule'))
    assert same([ln for ln in rose if not solid(ln)][0]['xy'], curve_xy('memetic_policy'))


# ================================================================== the option
def test_the_option_draws_the_two_best_of_eight_curves(files, monkeypatch, capsys):
    plain, _ = draw(files, monkeypatch)
    mj = put_mpar(files['tmp'])
    best, names = draw(files, monkeypatch, memetic_parallel=mj)
    assert names == [f'fq3_time_budget_{v}.{e}' for v in ('A', 'B', 'B_supplement')
                     for e in ('pdf', 'png')]
    single = [curve_xy(a) for a in M.FQ3_MPAR_METHODS] + \
        [curve_xy('memetic_rule', 'All')]
    want = {'memetic_rule': curve_xy('memetic_rule', shift=BEST_SHIFT),
            'memetic_policy': curve_xy('memetic_policy', shift=BEST_SHIFT)}
    for k, panel in enumerate(best['A']):
        rose = rose_lines(panel)
        assert len(rose) == 2, 'one solid and one dashed memetic curve per panel'
        assert same([ln for ln in rose if solid(ln)][0]['xy'], want['memetic_rule'])
        assert same([ln for ln in rose if not solid(ln)][0]['xy'], want['memetic_policy'])
        assert not any(same(ln['xy'], s) for ln in rose for s in single)
        # the bands of the memetic algorithm hold the best of eight's interval ends
        rose_bands = [b for b in panel['bands'] if b['color'] == ROSE]
        assert len(rose_bands) == 2
        for arm, b in zip(('memetic_rule', 'memetic_policy'), rose_bands):
            xy = want[arm]
            for t, m in xy:
                for y in (m - HALF, m + HALF):
                    assert np.isclose(b['xy'], [t, y], atol=1e-9).all(axis=1).any()
        # every other curve, band and mark is the one drawn without the option
        ref = plain['A'][k]
        assert len(other_lines(panel)) == len(other_lines(ref))
        assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['ls'] == y['ls']
                   for x, y in zip(other_lines(panel), other_lines(ref)))
        others = [b for b in panel['bands'] if b['color'] != ROSE]
        assert all(same(x['xy'], y['xy']) for x, y in
                   zip(others, [b for b in ref['bands'] if b['color'] != ROSE]))
        assert panel['xlim'] == ref['xlim']
    # the companion: its policy-started memetic curve is the best of eight
    for panel in best['B_supplement']:
        rose = rose_lines(panel)
        assert len(rose) == 1 and same(rose[0]['xy'], want['memetic_policy'])
    # variant B keeps the single run (row All): the same artists as without the option
    for p, q in zip(best['B'], plain['B']):
        assert all(same(x['xy'], y['xy']) for x, y in zip(p['lines'], q['lines']))
        assert p['ylim'] == q['ylim']
    out = capsys.readouterr().out
    assert 'decision-table gate: 12 of 12 values of the best of eight match' in out
    assert 'JSON gate: 44 of 44 checks match' in out


# ================================================================== the gates
def set_rows(rows, col, value, **where):
    for r in rows:
        if all(r[k] == v for k, v in where.items()):
            r[col] = value


def first_row(rows, **where):
    return next(r for r in rows if all(r[k] == v for k, v in where.items()))


def raise_mean(rows):
    r = first_row(rows, size='50x25n30', method='memetic_rule_best8', t_s='20')
    for c in ('mean_pct', 'lo_pct', 'hi_pct'):
        r[c] = f'{float(r[c]) + 3.0:.6f}'


def drop(rows, **where):
    rows[:] = [r for r in rows if not all(r[k] == v for k, v in where.items())]


DC = 'decision_table'
DEFECTS = [
    ('generator', lambda J, R: J.update(generator='scripts/p94_timed_inference.py'),
     'names the generator'),
    ('partial run', lambda J, R: J.update(run='partial'), 'is not a full run of p124'),
    ('dry run', lambda J, R: J.update(dry=True), 'is a dry run of p124'),
    ('run column', lambda J, R: set_rows(R, 'run', 'partial', size='80x25n30'),
     'the column run holds'),
    ('clock of the JSON', lambda J, R: J.update(origin='start'), 'not the request clock'),
    ('clock of the table', lambda J, R: J[DC].update(clock='start'), 'not the request clock'),
    ('clock of the CSV', lambda J, R: set_rows(R, 'origin', 'start'), 'the column origin holds'),
    ('selected run', lambda J, R: J.update(selected_run=SEL + 1),
     f'reads the selected training run {SEL + 1}'),
    ('policy column', lambda J, R: set_rows(R, 'policy', f'selected training run s{SEL + 1}'),
     'the column policy holds'),
    ('grid', lambda J, R: J[DC]['curves'].update(grid=GRID[:-1] + [610.0]),
     'is not the curve grid of the timed record'),
    ('time off the grid', lambda J, R: first_row(
        R, size='50x25n30', method='memetic_rule_best8', t_s='20').update(t_s='21'),
     'are not on the curve grid'),
    ('policy_first_s', lambda J, R: set_rows(R, 'policy_first_s', '16.500000',
                                              size='80x25n30'),
     'the column policy_first_s holds'),
    ('rule_first_s', lambda J, R: set_rows(R, 'rule_first_s', '1.400000', size='50x25n30'),
     'the column rule_first_s holds'),
    ('n_runs of a row', lambda J, R: first_row(R, method='memetic_policy_best8').update(
        n_runs=7), 'the column n_runs holds'),
    ('n_runs of the JSON', lambda J, R: J[DC]['curves'].update(n_runs=7), 'and 8 runs'),
    ('methods of the JSON', lambda J, R: J[DC]['curves'].update(methods=['memetic_rule_best8']),
     'decision_table.curves.methods'),
    ('a method at one size', lambda J, R: drop(R, method='memetic_policy_best8',
                                              size='80x25n30'),
     r'no pooled row \(Pool\) of memetic_policy_best8 at 80x25n30'),
    ('a rising curve', lambda J, R: raise_mean(R), 'the pooled mean rises'),
    ('start time', lambda J, R: J[DC]['curves']['start_s']['80x25n30'].update(
        memetic_rule_best8=5.0), 'starts at 2 s, the JSON at 5.0 s'),
    ('curves file', lambda J, R: J[DC]['curves'].update(file='results/other.csv'),
     'is not beside it'),
    ('decision table at 300 s', lambda J, R: J[DC]['sizes']['80x25n30']['rows']['300']['Pool'][
        'values'].update(memetic_policy='-9.99'),
     r"disagrees with the decision table on eight cores:\n  \('Sel', '80x25n30', "
     r"'memetic_policy'\) at 300 s"),
    ('decision table without a number', lambda J, R: J[DC]['sizes']['50x25n30']['rows']['600'][
        'Pool']['values'].update(memetic_rule=None), 'holds no number'),
]


@pytest.mark.parametrize('name,plant,match', DEFECTS, ids=[d[0] for d in DEFECTS])
def test_each_gate_refuses_a_planted_defect(name, plant, match, files, monkeypatch):
    mj = put_mpar(files['tmp'], plant=plant)
    with pytest.raises(SystemExit, match=match) as e:
        draw(files, monkeypatch, memetic_parallel=mj)
    assert str(e.value).startswith('fq3 REFUSING')


def test_the_unplanted_record_passes_every_gate(files, monkeypatch):
    """The control of the planted defects: the same record without a defect is drawn."""
    _, names = draw(files, monkeypatch, memetic_parallel=put_mpar(files['tmp']))
    assert len(names) == 6


def test_a_dry_run_is_drawn_only_with_allow_partial_and_named_dev(files, monkeypatch):
    mj = put_mpar(files['tmp'], plant=lambda J, R: J.update(dry=True))
    with pytest.raises(SystemExit, match='is a dry run of p124'):
        draw(files, monkeypatch, memetic_parallel=mj)
    _, names = draw(files, monkeypatch, memetic_parallel=mj, allow_partial=True)
    assert names == [f'fq3_time_budget_{v}_dev.{e}' for v in ('A', 'B', 'B_supplement')
                     for e in ('pdf', 'png')]


# ================================================================== the preview
def test_a_preview_is_refused_without_allow_partial_and_suffixed_with_it(files, monkeypatch,
                                                                       capsys):
    mj = put_mpar(files['tmp'], name='preview', preview=True)
    with pytest.raises(SystemExit, match='is a preview of p124'):
        draw(files, monkeypatch, memetic_parallel=mj)
    art, names = draw(files, monkeypatch, memetic_parallel=mj, allow_partial=True)
    assert names == ['fq3_time_budget_A_preview.pdf', 'fq3_time_budget_A_preview.png']
    for panel in art['A']:
        rose = rose_lines(panel)
        assert len(rose) == 1 and solid(rose[0])
        assert same(rose[0]['xy'], curve_xy('memetic_rule', shift=BEST_SHIFT))
        assert len([b for b in panel['bands'] if b['color'] == ROSE]) == 1
        # the dashed curves left are CP-SAT's, started from the policy's schedule
        assert [ln['color'] for ln in other_lines(panel) if not solid(ln)
                and ln['marker'] == 'None'] == [to_hex(M.C_EQUIV)]
    out = capsys.readouterr().out
    assert 'variant B: holds no curve of the preview, not drawn' in out
    assert 'variant B_supplement: holds no curve of the preview, not drawn' in out
    assert 'decision-table gate: 6 of 6 values' in out


def test_a_full_record_with_one_method_is_refused(files, monkeypatch):
    """A full record whose JSON and curves file agree on the first set alone is refused: a
    full record must hold both methods (a preview may hold one, see the test above)."""
    def one(J, R):
        J[DC]['curves'].update(methods=['memetic_rule_best8'])
        drop(R, method='memetic_policy_best8')
    with pytest.raises(SystemExit, match='decision_table.curves.methods'):
        draw(files, monkeypatch, memetic_parallel=put_mpar(files['tmp'], plant=one))


# ================================================================== the command line
def test_the_command_line_passes_the_record_to_the_figure(files, tmp_path):
    mj = put_mpar(files['tmp'], name='preview', preview=True)
    out = tmp_path / 'figs'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    for k in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        env.pop(k, None)
    base = [sys.executable, '-B', os.path.join(ROOT, 'scripts', 'make_figures.py'),
            '--only', 'fq3', '--timed-curves', files['tc'], '--timed-json', files['tj'],
            '--memetic-parallel', mj, '--out', str(out)]
    r = subprocess.run(base, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0 and 'is a preview of p124' in r.stdout + r.stderr
    r = subprocess.run(base + ['--allow-partial'], capture_output=True, text=True, env=env,
                       cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert sorted(os.listdir(out)) == ['fq3_time_budget_A_preview.pdf',
                                       'fq3_time_budget_A_preview.png']
    assert 'decision-table gate: 6 of 6 values' in r.stdout



# ================================================================== the one-minute budget
# The record of p137: per size and start, (mean, lo, hi) of the best of eight one-minute runs
# minus the policy's first schedule; the record of p124 it is read beside carries the same
# training run, instances and bootstrap count.
MINUTE = {'50x25n30': {'rule': (-1.0, -1.6, -0.4), 'policy': (-2.5, -3.0, -2.0)},
          '80x25n30': {'rule': (0.5, -0.2, 1.2), 'policy': (-1.8, -2.4, -1.3)}}
BOOT, INSTANCES = 20000, list(range(3, 12))
GREY = to_hex(M.FQ3_STYLE_KEY)
LEGEND_TODAY = ['memetic algorithm', 'CP-SAT solver', 'started from the best rule',
                'started from the policy’s schedule', 'policy, first schedule',
                'policy, sampled schedules']
LEGEND_MINUTE = ['memetic algorithm', 'CP-SAT solver', '',
                 'started from the best rule', 'started from the policy’s schedule',
                 'memetic algorithm, one-minute budget',
                 'policy, first schedule', 'policy, sampled schedules', '']


def minute_record():
    results = {}
    for size in SIZES:
        comp = {}
        for start, q in M.FQ3_MMIN_ENTRIES:
            m, lo, hi = MINUTE[size][start]
            comp[q] = dict(Pool=dict(mean=m, lo=lo, hi=hi, n=len(INSTANCES), c=8,
                                     combinations=list(M.FQ3_GROUPS['all'][1])))
        results[size] = dict(comparisons=comp)
    return dict(generator=M.FQ3_MMIN_GENERATOR, run='full', dry=False, origin='request',
                read_at_s=60, budget_s=60, selected_run=SEL, instances=list(INSTANCES),
                boot=BOOT, results=results)


def put_minute(tmp_path, name='minute', plant=None):
    J = minute_record()
    if plant is not None:
        plant(J)
    path = str(tmp_path / name / 'results' / 'memetic_one_minute.json')
    write_json(path, J)
    return path


def put_setting(tmp_path):
    """A full record of p124 that also states its instances and bootstrap count."""
    return put_mpar(tmp_path, name='setting',
                    plant=lambda J, R: J.update(boot=BOOT, instances=list(INSTANCES)))


def panel_full(ax):
    """panel_artists with the face colour of every line's marker."""
    p = panel_artists(ax)
    for d, ln in zip(p['lines'], ax.get_lines()):
        d['face'] = to_hex(ln.get_markerfacecolor())
    return p


def draw_full(files, monkeypatch, **kw):
    """As draw, with every figure's texts and legend: {variant: dict(panels, texts, legend,
    handles, legend_lines, columns)}."""
    figs, saved, close = [], [], M.plt.close
    monkeypatch.setattr(M.plt, 'close', lambda fig=None: figs.append(fig))
    monkeypatch.setattr(Figure, 'savefig', lambda self, path, **k: saved.append(path))
    try:
        written = M.fq3_time_budget(files['tc'], files['tj'], [], str(files['tmp'] / 'out'),
                                    **kw)
        stems = [os.path.basename(w)[:-len('.pdf')] for w in written if w.endswith('.pdf')]
        out = {}
        for stem, fig in zip(stems, figs):
            variant = stem[len('fq3_time_budget_'):]
            leg = fig.legends[0]
            rend = fig.canvas.get_renderer()
            out[variant] = dict(
                panels=[panel_full(ax) for ax in fig.axes],
                texts=[t.get_text() for t in fig.findobj(Text) if t.get_text().strip()],
                legend=[t.get_text() for t in leg.get_texts()],
                handles=[dict(marker=str(h.get_marker()), ls=str(h.get_linestyle()),
                              color=to_hex(h.get_color()), face=to_hex(h.get_markerfacecolor()))
                         if isinstance(h, Line2D) else type(h).__name__
                         for h in leg.legend_handles],
                legend_lines=[dict(marker=str(ln.get_marker()), color=to_hex(ln.get_color()),
                                   face=to_hex(ln.get_markerfacecolor()))
                              for ln in leg.findobj(Line2D)],
                columns=[round(t.get_window_extent(rend).x0, 3) for t in leg.get_texts()])
    finally:
        for fig in figs:
            close(fig)
        monkeypatch.undo()
    return out


def loaded(files, mj):
    T = M.load_timed(files['tc'], files['tj'], False)
    return T, M.load_memetic_parallel(mj, T, files['tc'], False)


def test_the_reader_accepts_a_full_record(files):
    T, mpar = loaded(files, put_setting(files['tmp']))
    got = M.load_memetic_minute(put_minute(files['tmp']), T, mpar)
    assert got['values'] == {s: {k: tuple(v) for k, v in MINUTE[s].items()} for s in SIZES}
    assert got['pool'] == 'Pool'


def comp(J, size, q):
    return J['results'][size]['comparisons'][q]['Pool']


MINUTE_DEFECTS = [
    ('preview', lambda J: J.update(run='preview'), 'is a preview of p137'),
    ('dry run', lambda J: J.update(dry=True), 'is a dry run of p137'),
    ('another training run', lambda J: J.update(selected_run=SEL + 1),
     f'has the selected training run {SEL + 1}, the record of p124'),
    ('another instance count', lambda J: J.update(instances=INSTANCES[:-1]),
     f'has the number of instances {len(INSTANCES) - 1}, the record of p124'),
    ('another bootstrap count', lambda J: J.update(boot=BOOT // 2),
     f'has the number of bootstrap draws {BOOT // 2}, the record of p124'),
    ('generator', lambda J: J.update(generator=M.FQ3_MPAR_GENERATOR), 'names the generator'),
    ('clock', lambda J: J.update(read_at_s=300), 'not the request clock read at 60 s'),
    ('missing entry', lambda J: J['results']['80x25n30']['comparisons'].pop('mp60_vs_pol'),
     r'80x25n30\.comparisons\.mp60_vs_pol\.Pool is not in the record'),
    ('mean outside its interval', lambda J: comp(J, '50x25n30', 'm60_vs_pol').update(mean=0.0),
     'not a finite mean inside its interval'),
    ('other combinations', lambda J: comp(J, '50x25n30', 'm60_vs_pol').update(c=4),
     'pools 4 combinations'),
]


@pytest.mark.parametrize('name,plant,match', MINUTE_DEFECTS, ids=[d[0] for d in MINUTE_DEFECTS])
def test_the_reader_refuses_a_planted_defect(name, plant, match, files, monkeypatch):
    mj = put_setting(files['tmp'])
    with pytest.raises(SystemExit, match=match) as e:
        draw_full(files, monkeypatch, memetic_parallel=mj,
                  memetic_minute=put_minute(files['tmp'], plant=plant))
    assert str(e.value).startswith('fq3 REFUSING')


@pytest.mark.parametrize('plant,match', [(lambda J: J.update(run='preview'), 'is a preview'),
                                         (lambda J: J.update(dry=True), 'is a dry run')],
                         ids=['preview', 'dry run'])
def test_a_preview_or_dry_run_is_refused_with_allow_partial_too(plant, match, files,
                                                                monkeypatch):
    mj = put_setting(files['tmp'])
    with pytest.raises(SystemExit, match=match):
        draw_full(files, monkeypatch, memetic_parallel=mj, allow_partial=True,
                  memetic_minute=put_minute(files['tmp'], plant=plant))


def test_the_option_needs_memetic_parallel(files, monkeypatch):
    with pytest.raises(SystemExit, match='needs --memetic-parallel'):
        draw_full(files, monkeypatch, memetic_minute=put_minute(files['tmp']))


def circles(panel):
    return [ln for ln in rose_lines(panel) if ln['marker'] == 'o']


def bars(panel):
    return [ln for ln in rose_lines(panel) if ln['marker'] == 'None' and len(ln['xy']) == 2
            and np.allclose(ln['xy'][:, 0], M.FQ3_MMIN_AT)]


def test_variant_a_holds_two_circles_per_panel_at_one_minute(files, monkeypatch, capsys):
    mj = put_setting(files['tmp'])
    plain = draw_full(files, monkeypatch, memetic_parallel=mj)
    capsys.readouterr()
    got = draw_full(files, monkeypatch, memetic_parallel=mj,
                    memetic_minute=put_minute(files['tmp']))
    assert sorted(got) == ['A', 'B', 'B_supplement']
    for size, panel, ref in zip(SIZES, got['A']['panels'], plain['A']['panels']):
        cs = circles(panel)
        assert len(cs) == 2
        by_face = {c['face']: c for c in cs}
        assert sorted(by_face) == sorted([ROSE, '#ffffff'])
        for start, face in (('rule', ROSE), ('policy', '#ffffff')):
            m, lo, hi = MINUTE[size][start]
            assert same(by_face[face]['xy'], np.array([[M.FQ3_MMIN_AT, m]]))
            assert any(same(b['xy'], np.array([[M.FQ3_MMIN_AT, lo], [M.FQ3_MMIN_AT, hi]]))
                       for b in bars(panel))
        assert len(bars(panel)) == 2
        # every curve, band and mark of the panel is the one drawn without the option
        extra = {id(ln) for ln in cs + bars(panel)}
        rest = [ln for ln in panel['lines'] if id(ln) not in extra]
        assert len(rest) == len(ref['lines'])
        assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['ls'] == y['ls']
                   for x, y in zip(rest, ref['lines']))
        assert all(same(x['xy'], y['xy']) for x, y in zip(panel['bands'], ref['bands']))
        assert panel['ylim'] == ref['ylim'] and panel['xlim'] == ref['xlim']
    # variant B and the companion: as without the option
    for v in ('B', 'B_supplement'):
        assert got[v]['texts'] == plain[v]['texts'] and got[v]['legend'] == plain[v]['legend']
        for p, q in zip(got[v]['panels'], plain[v]['panels']):
            assert len(p['lines']) == len(q['lines'])
            assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['marker'] ==
                       y['marker'] for x, y in zip(p['lines'], q['lines']))
            assert p['ylim'] == q['ylim']
    out = capsys.readouterr().out
    assert ('one-minute budget, 50x25n30, at 60 s: started from the best rule -1.00 '
            '[-1.60, -0.40] (filled circle), started from the policy’s schedule -2.50 '
            '[-3.00, -2.00] (open circle)') in out
    assert ('one-minute budget, 80x25n30, at 60 s: started from the best rule +0.50 '
            '[-0.20, +1.20] (filled circle), started from the policy’s schedule -1.80 '
            '[-2.40, -1.30] (open circle)') in out


def test_the_legend_of_variant_a_holds_nine_entries_in_order(files, monkeypatch):
    got = draw_full(files, monkeypatch, memetic_parallel=put_setting(files['tmp']),
                    memetic_minute=put_minute(files['tmp']))['A']
    assert got['legend'] == LEGEND_MINUTE
    # three columns of three rows, filled column by column
    x = got['columns']
    for col in ((0, 1), (3, 4, 5), (6, 7)):
        assert len({x[i] for i in col}) == 1
    assert x[0] < x[3] < x[6]
    # the start keys carry the circle of their start, in the grey of the keys
    h = got['handles']
    assert h[3] == dict(marker='o', ls='-', color=GREY, face=GREY)
    assert h[4]['marker'] == 'o' and h[4]['ls'] == '--' and h[4]['face'] == '#ffffff'
    # the one-minute entry: a filled and an open circle in the memetic colour
    rose_circles = [ln for ln in got['legend_lines'] if ln['marker'] == 'o'
                    and ln['color'] == ROSE]
    assert sorted(ln['face'] for ln in rose_circles) == sorted([ROSE, '#ffffff'])


def test_without_the_option_the_text_and_legend_are_unchanged(files, monkeypatch):
    mj = put_setting(files['tmp'])
    plain = draw_full(files, monkeypatch, memetic_parallel=mj)
    none = draw_full(files, monkeypatch, memetic_parallel=mj, memetic_minute=None)
    assert sorted(plain) == sorted(none) == ['A', 'B', 'B_supplement']
    for v in plain:
        assert plain[v]['texts'] == none[v]['texts']
        assert plain[v]['legend'] == none[v]['legend']
        assert plain[v]['handles'] == none[v]['handles']
    assert plain['A']['legend'] == LEGEND_TODAY
    assert all(h['marker'] == 'None' for h in plain['A']['handles'][2:4])
    assert not any(circles(p) for p in plain['A']['panels'])


def test_a_circle_outside_the_value_axis_extends_it(files, monkeypatch):
    mj = put_setting(files['tmp'])
    plain = draw_full(files, monkeypatch, memetic_parallel=mj)
    low = put_minute(files['tmp'], name='low', plant=lambda J: comp(
        J, '80x25n30', 'mp60_vs_pol').update(mean=-11.0, lo=-12.0, hi=-10.0))
    got = draw_full(files, monkeypatch, memetic_parallel=mj, memetic_minute=low)
    assert plain['A']['panels'][1]['ylim'][0] > -12.0
    assert got['A']['panels'][1]['ylim'][0] <= -12.0 + 1e-9
    assert got['B']['panels'][1]['ylim'] == plain['B']['panels'][1]['ylim']


def test_the_command_line_passes_the_one_minute_record(files, tmp_path):
    mj, mm = put_setting(files['tmp']), put_minute(files['tmp'])
    out = tmp_path / 'figs_minute'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    for k in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        env.pop(k, None)
    base = [sys.executable, '-B', os.path.join(ROOT, 'scripts', 'make_figures.py'),
            '--only', 'fq3', '--timed-curves', files['tc'], '--timed-json', files['tj'],
            '--memetic-minute', mm, '--out', str(out)]
    r = subprocess.run(base, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode != 0 and 'needs --memetic-parallel' in r.stdout + r.stderr
    r = subprocess.run(base + ['--memetic-parallel', mj], capture_output=True, text=True,
                       env=env, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert sorted(os.listdir(out)) == sorted(f'fq3_time_budget_{v}.{e}'
                                             for v in ('A', 'B', 'B_supplement')
                                             for e in ('pdf', 'png'))
    assert 'one-minute budget, 50x25n30, at 60 s: started from the best rule -1.00' in r.stdout


# ================================================================== the five-minute budget
# The record of p137 --budget 300: per size and start, (mean, lo, hi) of the best of eight
# five-minute runs minus the policy's first schedule, inside the value axis of the figure drawn
# without it; the same training run, instances and bootstrap count as the record of p124.
FIVE = {'50x25n30': {'rule': (-3.0, -3.7, -2.3), 'policy': (-4.6, -5.2, -4.0)},
        '80x25n30': {'rule': (-1.2, -1.9, -0.5), 'policy': (-3.4, -4.1, -2.8)}}
LEGEND_FIVE = [M.FQ3_MFIV_TEXT if t == M.FQ3_MMIN_TEXT else t for t in LEGEND_MINUTE]
AT5 = M.FQ3_MFIV_AT


def five_record():
    results = {}
    for size in SIZES:
        comp = {}
        for start, q in M.FQ3_MFIV_ENTRIES:
            m, lo, hi = FIVE[size][start]
            comp[q] = dict(Pool=dict(mean=m, lo=lo, hi=hi, n=len(INSTANCES), c=8,
                                     combinations=list(M.FQ3_GROUPS['all'][1])))
        results[size] = dict(comparisons=comp)
    return dict(generator=M.FQ3_MMIN_GENERATOR, run='full', dry=False, origin='request',
                read_at_s=300, budget_s=300, selected_run=SEL, instances=list(INSTANCES),
                boot=BOOT, results=results)


def put_five(tmp_path, name='five', plant=None):
    J = five_record()
    if plant is not None:
        plant(J)
    path = str(tmp_path / name / 'results' / 'memetic_five_minute.json')
    write_json(path, J)
    return path


def comp5(J, size, q):
    return J['results'][size]['comparisons'][q]['Pool']


def with_minute(files, **kw):
    """The records of p124 and of the one-minute budget, as the five-minute option needs them."""
    return dict(memetic_parallel=put_setting(files['tmp']),
                memetic_minute=put_minute(files['tmp']), **kw)


def marks_of(files, monkeypatch, **kw):
    """Variant A drawn with the figures kept open: per panel, the circles (marker o) and the
    vertical bars at the fixed budgets, with the properties that make them look alike."""
    figs, close = [], M.plt.close
    monkeypatch.setattr(M.plt, 'close', lambda fig=None: figs.append(fig))
    monkeypatch.setattr(Figure, 'savefig', lambda self, path, **k: None)
    try:
        written = M.fq3_time_budget(files['tc'], files['tj'], [], str(files['tmp'] / 'out'),
                                    **kw)
        stems = [os.path.basename(w) for w in written if w.endswith('.pdf')]
        fig = figs[[i for i, s in enumerate(stems) if s.startswith('fq3_time_budget_A')][0]]
        out = []
        for ax in fig.axes:
            dots, bars_ = [], []
            for ln in ax.get_lines():
                x = np.asarray(ln.get_xdata(), float)
                if ln.get_marker() == 'o':
                    dots.append(dict(x=float(x[0]), y=float(ln.get_ydata()[0]),
                                     color=to_hex(ln.get_color()), edge=to_hex(ln.get_mec()),
                                     face=to_hex(ln.get_mfc()), ms=ln.get_markersize(),
                                     mew=ln.get_mew(), z=ln.get_zorder()))
                elif len(x) == 2 and x[0] == x[1] and to_hex(ln.get_color()) == ROSE:
                    bars_.append(dict(x=float(x[0]), y=tuple(map(float, ln.get_ydata())),
                                      lw=ln.get_linewidth(), z=ln.get_zorder()))
            out.append(dict(dots=dots, bars=bars_))
    finally:
        for fig in figs:
            close(fig)
        monkeypatch.undo()
    return out


def test_the_five_minute_reader_accepts_a_full_record(files):
    T, mpar = loaded(files, put_setting(files['tmp']))
    got = M.load_memetic_five(put_five(files['tmp']), T, mpar)
    assert got['values'] == {s: {k: tuple(v) for k, v in FIVE[s].items()} for s in SIZES}
    assert got['pool'] == 'Pool' and got['preview'] is False
    assert got['keys'] == [f'results.{s}.comparisons.{q}.Pool' for s in SIZES
                           for _, q in M.FQ3_MFIV_ENTRIES]


FIVE_DEFECTS = [
    ('preview', lambda J: J.update(run='preview'),
     'is a preview of p137 .*the five-minute points are drawn from a full run only'),
    ('dry run', lambda J: J.update(dry=True), 'is a dry run of p137'),
    ('another training run', lambda J: J.update(selected_run=SEL + 1),
     f'has the selected training run {SEL + 1}, the record of p124'),
    ('another instance count', lambda J: J.update(instances=INSTANCES[:-1]),
     f'has the number of instances {len(INSTANCES) - 1}, the record of p124'),
    ('another bootstrap count', lambda J: J.update(boot=BOOT // 2),
     f'has the number of bootstrap draws {BOOT // 2}, the record of p124'),
    ('generator', lambda J: J.update(generator=M.FQ3_MPAR_GENERATOR), 'names the generator'),
    ('reading time', lambda J: J.update(read_at_s=60),
     'not the request clock read at 300 s with a budget of 300 s'),
    ('budget', lambda J: J.update(budget_s=600),
     'not the request clock read at 300 s with a budget of 300 s'),
    ('missing entry', lambda J: J['results']['80x25n30']['comparisons'].pop('mp300_vs_pol'),
     r'80x25n30\.comparisons\.mp300_vs_pol\.Pool is not in the record'),
    ('mean outside its interval', lambda J: comp5(J, '50x25n30', 'm300_vs_pol').update(mean=0.0),
     'not a finite mean inside its interval'),
    ('other combinations', lambda J: comp5(J, '50x25n30', 'm300_vs_pol').update(c=4),
     'pools 4 combinations'),
]


@pytest.mark.parametrize('name,plant,match', FIVE_DEFECTS, ids=[d[0] for d in FIVE_DEFECTS])
def test_the_five_minute_reader_refuses_a_planted_defect(name, plant, match, files,
                                                         monkeypatch):
    with pytest.raises(SystemExit, match=match) as e:
        draw_full(files, monkeypatch, **with_minute(
            files, memetic_five=put_five(files['tmp'], plant=plant)))
    assert str(e.value).startswith('fq3 REFUSING')


def test_the_one_minute_record_is_refused_in_the_place_of_the_five_minute_record(files,
                                                                                monkeypatch):
    with pytest.raises(SystemExit, match=r'read_at_s 60 and budget_s 60, not the request clock '
                                         r'read at 300 s with a budget of 300 s'):
        draw_full(files, monkeypatch, **with_minute(files,
                                                    memetic_five=put_minute(files['tmp'],
                                                                            name='in_place')))


@pytest.mark.parametrize('plant,match', [(lambda J: J.update(run='preview'), 'is a preview'),
                                         (lambda J: J.update(dry=True), 'is a dry run')],
                         ids=['preview', 'dry run'])
def test_a_five_minute_preview_or_dry_run_is_refused_with_allow_partial_too(plant, match, files,
                                                                           monkeypatch):
    with pytest.raises(SystemExit, match=match):
        draw_full(files, monkeypatch, allow_partial=True,
                  **with_minute(files, memetic_five=put_five(files['tmp'], plant=plant)))


def test_the_five_minute_option_needs_memetic_minute(files, monkeypatch):
    with pytest.raises(SystemExit, match='needs --memetic-minute'):
        draw_full(files, monkeypatch, memetic_parallel=put_setting(files['tmp']),
                  memetic_five=put_five(files['tmp']))


def test_variant_a_holds_two_circles_per_panel_at_five_minutes(files, monkeypatch, capsys):
    kw = with_minute(files)
    plain = draw_full(files, monkeypatch, **kw)
    capsys.readouterr()
    got = draw_full(files, monkeypatch, memetic_five=put_five(files['tmp']), **kw)
    out = capsys.readouterr().out
    assert sorted(got) == ['A', 'B', 'B_supplement']
    for size, panel, ref in zip(SIZES, got['A']['panels'], plain['A']['panels']):
        at5 = [c for c in circles(panel) if np.isclose(c['xy'][0, 0], AT5)]
        assert len(circles(panel)) == 4 and len(at5) == 2
        by_face = {c['face']: c for c in at5}
        assert sorted(by_face) == sorted([ROSE, '#ffffff'])
        bars5 = [ln for ln in rose_lines(panel) if ln['marker'] == 'None' and len(ln['xy']) == 2
                 and np.allclose(ln['xy'][:, 0], AT5)]
        assert len(bars5) == 2
        for start, face in (('rule', ROSE), ('policy', '#ffffff')):
            m, lo, hi = FIVE[size][start]
            assert same(by_face[face]['xy'], np.array([[AT5, m]]))
            assert any(same(b['xy'], np.array([[AT5, lo], [AT5, hi]])) for b in bars5)
        # the circles at 60 s and every curve, band and mark: as drawn without the option
        extra = {id(ln) for ln in at5 + bars5}
        rest = [ln for ln in panel['lines'] if id(ln) not in extra]
        assert len(rest) == len(ref['lines'])
        assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['ls'] == y['ls']
                   and x['marker'] == y['marker'] and x['face'] == y['face']
                   for x, y in zip(rest, ref['lines']))
        assert all(same(x['xy'], y['xy']) for x, y in zip(panel['bands'], ref['bands']))
        assert panel['ylim'] == ref['ylim'] and panel['xlim'] == ref['xlim']
    # variant B and the companion: as without the option
    for v in ('B', 'B_supplement'):
        assert got[v]['texts'] == plain[v]['texts'] and got[v]['legend'] == plain[v]['legend']
        for p, q in zip(got[v]['panels'], plain[v]['panels']):
            assert len(p['lines']) == len(q['lines'])
            assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['marker'] ==
                       y['marker'] for x, y in zip(p['lines'], q['lines']))
            assert p['ylim'] == q['ylim']
    assert ('five-minute budget, 50x25n30, at 300 s: started from the best rule -3.00 '
            '[-3.70, -2.30] (filled circle), started from the policy’s schedule -4.60 '
            '[-5.20, -4.00] (open circle); results.50x25n30.comparisons.m300_vs_pol.Pool and '
            'mp300_vs_pol.Pool, read back from the record') in out
    assert ('five-minute budget, 80x25n30, at 300 s: started from the best rule -1.20 '
            '[-1.90, -0.50] (filled circle), started from the policy’s schedule -3.40 '
            '[-4.10, -2.80] (open circle)') in out
    assert 'PREVIEW' not in out


def test_the_five_minute_circles_look_like_the_one_minute_circles(files, monkeypatch):
    """Same colour, edge, face per start, size, edge width and z-order; bars of the same width
    and z-order, behind their circles."""
    panels = marks_of(files, monkeypatch, memetic_five=put_five(files['tmp']),
                      **with_minute(files))
    for p in panels:
        assert sorted(d['x'] for d in p['dots']) == [60.0, 60.0, AT5, AT5]
        assert sorted(b['x'] for b in p['bars']) == [60.0, 60.0, AT5, AT5]
        for face in (ROSE, '#ffffff'):
            one, five = (next(dict(d, x=None, y=None) for d in p['dots']
                              if d['x'] == x and d['face'] == face) for x in (60.0, AT5))
            assert one == five
        look = {(b['lw'], b['z']) for b in p['bars']}
        assert look == {(M.FQ3_MMIN_BAR_LW, M.FQ3_MMIN_Z)}
        assert all(d['z'] > M.FQ3_MMIN_Z for d in p['dots'])


def test_the_legend_names_both_budgets(files, monkeypatch):
    kw = with_minute(files)
    plain = draw_full(files, monkeypatch, **kw)['A']
    got = draw_full(files, monkeypatch, memetic_five=put_five(files['tmp']), **kw)['A']
    assert got['legend'] == LEGEND_FIVE
    assert M.FQ3_MFIV_TEXT == 'memetic algorithm, budget equal to the time'
    # nothing else in the legend changes: the handles, the order and the columns
    assert got['handles'] == plain['handles']
    assert got['legend_lines'] == plain['legend_lines']
    x = got['columns']
    for col in ((0, 1), (3, 4, 5), (6, 7)):
        assert len({x[i] for i in col}) == 1
    assert x[0] < x[3] < x[6]
    rose_circles = [ln for ln in got['legend_lines'] if ln['marker'] == 'o'
                    and ln['color'] == ROSE]
    assert sorted(ln['face'] for ln in rose_circles) == sorted([ROSE, '#ffffff'])


def test_without_the_five_minute_option_the_figure_is_unchanged(files, monkeypatch, capsys):
    kw = with_minute(files)
    plain = draw_full(files, monkeypatch, **kw)
    none = draw_full(files, monkeypatch, memetic_five=None, **kw)
    assert 'five-minute' not in capsys.readouterr().out
    assert sorted(plain) == sorted(none) == ['A', 'B', 'B_supplement']
    for v in plain:
        assert plain[v]['texts'] == none[v]['texts']
        assert plain[v]['legend'] == none[v]['legend']
        assert plain[v]['handles'] == none[v]['handles']
        for p, q in zip(plain[v]['panels'], none[v]['panels']):
            assert len(p['lines']) == len(q['lines']) and len(p['bands']) == len(q['bands'])
            assert all(same(x['xy'], y['xy']) and x['color'] == y['color'] and x['ls'] == y['ls']
                       and x['marker'] == y['marker'] and x['face'] == y['face']
                       for x, y in zip(p['lines'], q['lines']))
            assert p['ylim'] == q['ylim'] and p['xlim'] == q['xlim']
    assert plain['A']['legend'] == LEGEND_MINUTE
    assert not any(np.isclose(c['xy'][0, 0], AT5) for p in plain['A']['panels']
                   for c in circles(p))


def test_a_five_minute_circle_outside_the_value_axis_extends_it(files, monkeypatch):
    kw = with_minute(files)
    plain = draw_full(files, monkeypatch, **kw)
    low = put_five(files['tmp'], name='low', plant=lambda J: comp5(
        J, '50x25n30', 'mp300_vs_pol').update(mean=-11.0, lo=-12.0, hi=-10.0))
    got = draw_full(files, monkeypatch, memetic_five=low, **kw)
    assert plain['A']['panels'][0]['ylim'][0] > -12.0
    assert got['A']['panels'][0]['ylim'][0] <= -12.0 + 1e-9
    assert got['B']['panels'][0]['ylim'] == plain['B']['panels'][0]['ylim']


def test_a_preview_is_drawn_only_through_the_test_switch(files, monkeypatch, capsys):
    """The keyword-only test switch reads a preview of p137 into files ending in
    _five_preview, and refuses the manuscript's folder; a dry run stays refused."""
    sig = inspect.signature(M.fq3_time_budget).parameters['test_only_five_preview']
    assert sig.kind is inspect.Parameter.KEYWORD_ONLY and sig.default is False
    prev = put_five(files['tmp'], name='prev', plant=lambda J: J.update(run='preview'))
    kw = with_minute(files, memetic_five=prev)
    with pytest.raises(SystemExit, match='is a preview of p137'):
        draw(files, monkeypatch, **kw)
    art, names = draw(files, monkeypatch, test_only_five_preview=True, **kw)
    assert names == [f'fq3_time_budget_{v}_five_preview.{e}' for v in ('A', 'B', 'B_supplement')
                     for e in ('pdf', 'png')]
    assert 'PREVIEW of p137' in capsys.readouterr().out
    for size, panel in zip(SIZES, art['A_five']):        # draw strips _preview from the name
        at5 = [ln for ln in rose_lines(panel) if ln['marker'] == 'o'
               and np.isclose(ln['xy'][0, 0], AT5)]
        assert sorted(float(ln['xy'][0, 1]) for ln in at5) == sorted(
            v[0] for v in FIVE[size].values())
    paper = os.path.join(ROOT, 'manuscript', 'sections_new', 'figures')
    with pytest.raises(SystemExit, match='is written outside'):
        M.fq3_time_budget(files['tc'], files['tj'], [], paper, test_only_five_preview=True,
                          **kw)
    dry = put_five(files['tmp'], name='dry', plant=lambda J: J.update(dry=True))
    with pytest.raises(SystemExit, match='is a dry run of p137'):
        draw(files, monkeypatch, test_only_five_preview=True,
             **with_minute(files, memetic_five=dry))


def test_the_command_line_passes_the_five_minute_record(files, tmp_path):
    mj, mm, m5 = put_setting(files['tmp']), put_minute(files['tmp']), put_five(files['tmp'])
    prev = put_five(files['tmp'], name='prev', plant=lambda J: J.update(run='preview'))
    out = tmp_path / 'figs_five'
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    for k in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        env.pop(k, None)
    base = [sys.executable, '-B', os.path.join(ROOT, 'scripts', 'make_figures.py'),
            '--only', 'fq3', '--timed-curves', files['tc'], '--timed-json', files['tj'],
            '--memetic-parallel', mj, '--out', str(out)]

    def run(*extra):
        return subprocess.run(base + list(extra), capture_output=True, text=True, env=env,
                              cwd=ROOT)
    r = run('--memetic-five', m5)
    assert r.returncode != 0 and 'needs --memetic-minute' in r.stdout + r.stderr
    for extra in ([], ['--allow-partial']):
        r = run('--memetic-minute', mm, '--memetic-five', prev, *extra)
        assert r.returncode != 0 and 'is a preview of p137' in r.stdout + r.stderr
    assert not out.exists() or not os.listdir(out)
    r = run('--memetic-minute', mm, '--memetic-five', m5)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert sorted(os.listdir(out)) == sorted(f'fq3_time_budget_{v}.{e}'
                                             for v in ('A', 'B', 'B_supplement')
                                             for e in ('pdf', 'png'))
    assert 'five-minute budget, 50x25n30, at 300 s: started from the best rule -3.00' in r.stdout
