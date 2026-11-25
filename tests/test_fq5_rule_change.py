"""The figure fq5_rule_change of scripts/make_figures.py: (a) training before the first use
against the number of rule combinations served, one specialist per combination against one
policy; (b) the time of the steps at a rule change, by batch size, on a logarithmic axis.

  * the synthetic records are written by the real writers: the record and the macro file of
    p99_rule_change_cost.py infer, run on synthetic training-cost, checks and count records (the
    builders of tests/test_rule_change_cost.py), and the macro file of
    p75_deploy_cost.write_artifacts from a record with the keys that the main function of p75
    writes (a test holds those keys against the source of p75), so a renamed key breaks these
    tests;
  * the full figure draws exactly the records: the mean line k times the mean of the eight
    specialists, the band from the sum of the k smallest to the sum of the k largest hours, the
    flat line of the policy, and in panel (b) one marker per step and batch size at its time,
    one shape per batch size in one neutral colour; the band is left out when it is not thicker
    than the line; a marker that would touch another of its row moves off the row's line;
  * there is no partial mode: the full mode refuses and names every missing input (record or
    macro file absent, the policy or a specialist or the checks at one size missing);
  * a record of p75 that is not of the paper's environment (an earlier environment, four
    threads, busy cores, an invalid size, another model) is refused with its reasons, and so
    is the copy of the macro file of an earlier environment;
  * the gate holds every plotted value against the macro that prints it and refuses one number
    that disagrees;
  * the layout: one height and one baseline for the two panels, every text on the canvas, no
    two texts overlapping, the legends inside their panels and clear of every mark, the PDF
    with the Times face only;
  * main draws fq5_rule_change only with --only fq5_rule_change, from the registry paths.

Synthetic records only, written under the pytest temporary folder; nothing is trained or timed
and no display is needed. Run from the repository root on one core:
  taskset -c 3 python -B -m pytest -p no:cacheprovider --basetemp=<scratch>
      tests/test_fq5_rule_change.py
"""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import arm_registry as REG  # noqa: E402
import make_figures as M  # noqa: E402
import test_rule_change_cost as TRC  # noqa: E402  (the synthetic inputs of p99 infer)

P99, P75 = M._fq5_scripts()
COMBOS = list(REG.COMBOS)
# synthetic magnitudes: a specialist 1.0 to 1.4 GPU hours, the policy about 2.5
SPEC_HOURS = {'N': 1.04, 'L': 1.12, 'S': 1.27, 'W': 1.08, 'LS': 1.31, 'LW': 1.19, 'SW': 1.36,
              'LSW': 1.40}
POLICY_HOURS = 2.52
# ms, median over the test instances: the descriptor checks and the schedule check, by p99 size
CHECKS = {'10x25': (1.20, 5.2), '50x25n30': (3.60, 31.0), '80x25n30': (5.80, 58.0)}
# s, re-planning of one batch in the combination with the largest median, by p75 shop size
REPLAN = {'10x25': 0.41, '50x25': 3.05, '80x25': 6.12}
SEED = 302
SHAPES = {10: 'o', 50: '^', 80: 'D'}


# ================================================================== the synthetic records
def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=1)


def load(path):
    with open(path) as f:
        return json.load(f)


def p75_record(replan=None, seed=SEED):
    """A record with the keys the main function of p75_deploy_cost.py writes (the parent's
    record, and per size the child's record with wall_s and valid), measured in the exact
    environment with one thread; the largest median of each size is that of LSW."""
    replan = REPLAN if replan is None else replan
    model = REG.PREFIX + REG.model('released', None, seed, environment='exact')
    sizes = {}
    for size, worst in replan.items():
        m = int(size.split('x')[0])
        per = {}
        for i, R in enumerate(COMBOS):
            med = worst if R == 'LSW' else worst * (0.90 + 0.01 * i)
            per[R] = {'median_s': med, 'p05_s': 0.95 * med, 'p95_s': 1.05 * med,
                      'min_s': 0.94 * med, 'max_s': 1.06 * med, 'n': 10 if m >= 50 else 30,
                      'descriptors_available': 20 if m >= 50 else 100,
                      'decisions_median': 25 * m, 'decisions_min': 25 * m,
                      'decisions_max': 25 * m, 'ms_per_decision_median': 40.0 * med / m,
                      'cpu_share': 0.99}
        sizes[size] = {'size': size, 'modules': m, 'per_regime': per, 'peak_rss_mb': 300 + m / 10,
                       'rss_after_import_mb': 250.0, 'rss_after_model_load_mb': 290.0,
                       'rollout_rss_increment_mb': 10 + m / 10, 'state_dict_numel': 32786,
                       'parameter_numel': 32786, 'torch_num_threads': 1,
                       'torch_num_interop_threads': 1, 'OMP_NUM_THREADS': '1',
                       'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                       'cuda_visible_devices': '', 'affinity': [3], 'loadavg_start': [0.5] * 3,
                       'loadavg_end': [0.5] * 3, 'model': model,
                       'checkpoint': f'trained_network/FAMILY/{model}.pth',
                       'environment': 'exact',
                       'mask': {'first_mask': 'exact', 'mask_ties': 'exact'},
                       'cpu_seconds': 99.0, 'wall_seconds': 100.0, 'cpu_share': 0.99,
                       'smoke_rows': [], 'wall_s': 120.0, 'valid': True}
    return {'device': 'cpu', 'model': model, 'checkpoint': f'trained_network/FAMILY/{model}.pth',
            'config': f'train_log/FAMILY/config_{model}.json', 'environment': 'exact',
            'seed_option': 'select', 'checkpoint_bytes': 156257, 'torch_threads_option': 1,
            'numpy_threads_option': 1, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'affinity': [3], 'cpu_count': 24, 'cpu_model': 'synthetic', 'core_max_mhz': {'3': 1},
            'hostname': 'synthetic', 'started': '2026-01-01 00:00:00',
            'loadavg_start': [0.5] * 3, 'allow_busy': False, 'sizes': sizes,
            'contention': {s: {'before': {'idle': True}, 'after': {'idle': True}} for s in sizes},
            'elapsed_s': 1000.0, 'loadavg_end': [0.5] * 3, 'finished': '2026-01-01 00:20:00',
            'valid': True, 'torch_num_threads': 1, 'torch_num_interop_threads': 1,
            'state_dict_numel': 32786, 'parameter_numel': 32786}


def write_deploy_macros(rec, record_path, out):
    """The macro file of p75 for this record (p75_deploy_cost.write_artifacts), under out/paper
    as arm_registry.out puts it; its header names the record as the source."""
    keep = {v: os.environ.get(v) for v in ('PACT_ENV', 'PACT_OUT', 'PACT_SEEDS')}
    os.environ.update(PACT_ENV='exact', PACT_OUT=str(out))
    os.environ.pop('PACT_SEEDS', None)
    try:
        P75.write_artifacts(rec, os.path.relpath(str(record_path), ROOT), True)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return os.path.join(str(out), 'paper', 'macros_deploy.tex')


def build_records(base, spec=None, combos=None, checks=None, replan=None):
    """Every input of the figure under base: p99 infer on synthetic training-cost, checks and
    count records, and the record of p75 with its macro file. Returns the four paths."""
    base = pathlib.Path(base)
    spec = SPEC_HOURS if spec is None else spec
    combos = COMBOS if combos is None else combos
    checks = CHECKS if checks is None else checks
    root, out = base / 'root', base / 'out'
    (root / TRC.TCE.RECORD_DIR).mkdir(parents=True)
    TRC.make_record(root, 'released', hours=POLICY_HOURS)
    for R in combos:
        TRC.make_record(root, 'specialist', R, hours=spec[R])
    (out / 'results').mkdir(parents=True)
    C = TRC.checks_record(sizes=tuple(checks))
    for s, (d, v) in checks.items():
        C['sizes'][s].update(descriptor_ms_median=d, descriptor_ms_max=1.5 * d,
                             descriptor_spread_ms=0.02, schedule_ms_median=v,
                             schedule_ms_max=1.5 * v, schedule_spread_ms=0.3)
    dump(out / 'results' / 'rule_change_checks.json', C)
    dump(out / 'results' / 'rule_change_count.json', TRC.count_record())
    r = TRC.infer(root, out)
    assert r.returncode == 0, r.stderr
    rec_path = out / 'results' / M.FQ5_DEPLOY_JSON.split('/')[-1]
    rec = p75_record(replan)
    dump(rec_path, rec)
    return dict(cost=str(out / 'results' / 'rule_change_cost.json'),
                cost_macros=str(out / 'manuscript' / 'macros_rulechange_cost.tex'),
                deploy=str(rec_path), deploy_macros=write_deploy_macros(rec, rec_path, out))


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    return build_records(tmp_path_factory.mktemp('records'))


@pytest.fixture
def world(records, tmp_path):
    """A private copy of the records (a test may change them); the macro file of p75 is written
    again for the copied record, since its header names the record's path."""
    out = {}
    for k, p in records.items():
        q = tmp_path / 'w' / k / os.path.basename(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, q)
        out[k] = str(q)
    out['deploy_macros'] = write_deploy_macros(load(out['deploy']), out['deploy'],
                                               tmp_path / 'w' / 'p75')
    return out


def args(w, **kw):
    a = dict(cost_json=w['cost'], cost_macros=w['cost_macros'], deploy_json=w['deploy'],
             deploy_macros=w['deploy_macros'])
    a.update(kw)
    return a


def draw(w, out, monkeypatch, **kw):
    kept = []
    monkeypatch.setattr(M.plt, 'close', kept.append)
    written = M.fq5_rule_change(out_dir=str(out), **args(w, **kw))
    return written, kept[0]


def edit_macro(path, name, new):
    text = open(path).read()
    pat = re.compile(r'(\\newcommand\{\\' + name + r'\})\{[^}]*\}')
    assert len(pat.findall(text)) == 1, name
    open(path, 'w').write(pat.sub(lambda m: m.group(1) + '{' + new + '}', text))


def points(ax):
    """Lines with more than one point [(x, y, colour)] and markers {marker: [(x, y)]}."""
    lines, marks = [], {}
    for ln in ax.lines:
        x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
        if ln.get_linestyle() not in ('None', 'none', ''):
            lines.append((x, y, ln.get_color()))
        else:
            marks.setdefault(ln.get_marker(), []).extend(zip(x, y))
    return lines, marks


def expected_a(w):
    """Mean line, band and policy hours from the record of p99."""
    J = load(w['cost'])
    h = np.sort([J['records'][f'specialist {R}']['hours'] for R in COMBOS])
    k = np.arange(1, 9)
    return (k * h.sum() / 8, np.cumsum(h), np.cumsum(h[::-1]),
            J['records']['released']['hours'])


# ================================================================== the figure
def test_the_full_figure_draws_the_records(world, tmp_path, monkeypatch):
    out = tmp_path / 'fig'
    written, fig = draw(world, out, monkeypatch)
    assert sorted(os.path.basename(p) for p in written) == ['fq5_rule_change.pdf',
                                                            'fq5_rule_change.png']
    assert sorted(os.listdir(out)) == sorted(os.path.basename(p) for p in written)
    w_in, h_in = fig.get_size_inches()
    assert round(w_in, 2) == M.FIG_W and h_in <= 2.6
    ax, bx = fig.axes
    # (a) the mean line with its squares, the band, the flat policy line with its circles
    line_s, lo, hi, pol = expected_a(world)
    lines, marks = points(ax)
    k = np.arange(1, 9)
    spec = [ln for ln in lines if ln[2] == M.FQ5_SPEC['color']]
    pols = [ln for ln in lines if ln[2] == M.FQ5_POL['color']]
    assert len(spec) == len(pols) == 1
    assert np.allclose(spec[0][0], k) and np.allclose(spec[0][1], line_s)
    assert np.allclose(pols[0][0], [1, 8]) and np.allclose(pols[0][1], [pol, pol])
    assert np.allclose(sorted(marks['s']), list(zip(k, line_s)))
    assert np.allclose(sorted(marks['o']), list(zip(k, [pol] * 8)))
    assert abs(line_s[-1] - sum(SPEC_HOURS.values())) < 1e-9 and abs(pol - POLICY_HOURS) < 1e-9
    (band,) = ax.collections
    v = band.get_paths()[0].vertices
    for kk, a, b in zip(k, lo, hi):
        ys = v[np.isclose(v[:, 0], kk), 1]
        assert np.isclose(ys.min(), a) and np.isclose(ys.max(), b)
    assert ax.get_ylim()[0] == 0 and ax.get_ylim()[1] > hi.max()
    assert [int(t) for t in ax.get_xticks()] == list(k)
    # (b) one marker per step and batch size at its time, on the line of its row
    rows = {t.get_text(): y for t, y in zip(bx.get_yticklabels(), bx.get_yticks())}
    assert list(rows) == list(M.FQ5_ROWS)
    _, mb = points(bx)
    assert {m: len(v) for m, v in mb.items()} == {'o': 3, '^': 3, 'D': 3}
    want = {}
    for (s, (d, sc)), shop in zip(CHECKS.items(), REPLAN):
        m = int(s.split('x')[0])
        want[(SHAPES[m], 'descriptor checks')] = d / 1000
        want[(SHAPES[m], 'schedule validation')] = sc / 1000
        want[(SHAPES[m], 're-planning')] = REPLAN[shop]
    for mk, pts in mb.items():
        for x, y in pts:
            row = [r for r, yy in rows.items() if abs(yy - y) < 1e-9]
            assert len(row) == 1, (mk, x, y)        # no marker moved off its row here
            assert np.isclose(x, want[(mk, row[0])]), (mk, row[0], x)
    assert bx.get_xscale() == 'log'
    assert [t.get_text() for t in bx.get_xticklabels()] == ['1 ms', '10 ms', '0.1 s', '1 s',
                                                            '10 s']
    faces = {tuple(np.round(M.matplotlib.colors.to_rgba(ln.get_markerfacecolor()), 4))
             for ln in bx.lines if ln.get_linestyle() in ('None', 'none', '')}
    assert faces == {tuple(np.round(M.matplotlib.colors.to_rgba(M.FQH_FILL_COLOR), 4))}
    # words, legends, titles
    assert ax.get_title() == '(a) Training before the first use'
    assert bx.get_title() == '(b) Steps at a rule change'
    assert ax.get_xlabel() == 'number of rule combinations served'
    assert ax.get_ylabel() == 'training (GPU hours)' and bx.get_xlabel() == 'time per batch'
    assert [t.get_text() for t in ax.get_legend().get_texts()] == list(M.FQ5_LEGEND_A)
    assert [t.get_text() for t in bx.get_legend().get_texts()] == ['10 modules', '50 modules',
                                                                   '80 modules']
    assert not ax.texts and not bx.texts and not fig.texts


def test_layout_one_grid_texts_on_the_canvas_legends_clear_of_the_data(world, tmp_path,
                                                                        monkeypatch):
    _, fig = draw(world, tmp_path / 'fig', monkeypatch)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax, bx = fig.axes
    ea, eb = ax.get_window_extent(rend), bx.get_window_extent(rend)
    assert abs(ea.y0 - eb.y0) < 0.5 and abs(ea.height - eb.height) < 0.5
    assert abs(ax.xaxis.label.get_window_extent(rend).y0
               - bx.xaxis.label.get_window_extent(rend).y0) < 0.5
    assert abs(ax.title.get_window_extent(rend).y0 - bx.title.get_window_extent(rend).y0) < 0.5
    assert M._check_layout(fig, (), ax.get_legend(), tag='test') > 20
    # every legend inside its panel; no marker centre and no point of a line in its box
    for a in (ax, bx):
        lb, e = a.get_legend().get_window_extent(rend), a.get_window_extent(rend)
        assert e.x0 <= lb.x0 and lb.x1 <= e.x1 and e.y0 <= lb.y0 and lb.y1 <= e.y1
        for ln in a.lines:
            if ln.get_color() == M.C_GRID:            # grid lines may run under a legend
                continue
            x, y = np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)
            if len(x) == 2 and ln.get_linestyle() not in ('None', 'none', ''):
                x, y = np.linspace(x[0], x[1], 200), np.linspace(y[0], y[1], 200)
            elif ln.get_linestyle() not in ('None', 'none', ''):
                xs = np.linspace(x[0], x[-1], 400)
                x, y = xs, np.interp(xs, x, y)
            d = a.transData.transform(np.column_stack([x, y]))
            assert not np.any((d[:, 0] >= lb.x0) & (d[:, 0] <= lb.x1) & (d[:, 1] >= lb.y0)
                              & (d[:, 1] <= lb.y1))
    # the row labels of (b) stay clear of panel (a)
    assert min(t.get_window_extent(rend).x0 for t in bx.get_yticklabels()) > ea.x1 + 20


def test_the_pdf_embeds_the_times_face_only(world, tmp_path, monkeypatch):
    written, _ = draw(world, tmp_path / 'fig', monkeypatch)
    M.assert_times_only(written)
    if shutil.which('pdffonts'):
        pdf = [p for p in written if p.endswith('.pdf')][0]
        lines = subprocess.run(['pdffonts', pdf], capture_output=True, text=True,
                               check=True).stdout.splitlines()[2:]
        names = {ln.split()[0].split('+')[-1] for ln in lines if ln.split()}
        assert names and all(n.startswith('TeXGyreTermes') for n in names), names


def test_where_the_lines_cross_both_markers_stay_visible(world, tmp_path, monkeypatch, capsys):
    # at two combinations the specialists' sum (2.44 h) lies next to the policy (2.52 h)
    draw(world, tmp_path / 'fig', monkeypatch)
    text = capsys.readouterr().out
    m = re.search(r'at least (\d+)% of each policy marker stays visible', text)
    assert m and int(m.group(1)) >= 100 * M.FQ5_VISIBLE
    assert 'band of the choice of combinations: drawn' in text
    assert 'markers moved off the line of their row: none' in text


def test_the_band_is_left_out_when_not_thicker_than_the_line(tmp_path, monkeypatch, capsys):
    flat = {R: 1.200 + 0.001 * i for i, R in enumerate(COMBOS)}
    w = build_records(tmp_path / 'flat', spec=flat)
    _, fig = draw(w, tmp_path / 'fig', monkeypatch)
    ax = fig.axes[0]
    assert not ax.collections
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [M.FQ5_LEGEND_A[0],
                                                                   M.FQ5_LEGEND_A[2]]
    assert 'band of the choice of combinations: left out' in capsys.readouterr().out


def test_a_marker_that_would_touch_another_moves_off_the_row(world, tmp_path, monkeypatch,
                                                             capsys):
    # the descriptor checks take the same time at 50 and 80 modules: record and macro file
    J = load(world['cost'])
    J['macros']['RccDescChecksMsEighty'] = J['macros']['RccDescChecksMsFifty']
    dump(world['cost'], J)
    edit_macro(world['cost_macros'], 'RccDescChecksMsEighty', J['macros']['RccDescChecksMsFifty'])
    _, fig = draw(world, tmp_path / 'fig', monkeypatch)
    bx = fig.axes[1]
    y_row = dict(zip([t.get_text() for t in bx.get_yticklabels()], bx.get_yticks()))
    _, mb = points(bx)
    x50 = [p for p in mb['^'] if np.isclose(p[0], 3.6e-3)]
    x80 = [p for p in mb['D'] if np.isclose(p[0], 3.6e-3)]
    assert len(x50) == len(x80) == 1
    assert x50[0][1] == y_row['descriptor checks'] and x80[0][1] != y_row['descriptor checks']
    y80, y50 = (bx.transData.transform(p)[1] for p in (x80[0], x50[0]))
    gap_pt = abs(y80 - y50) * 72 / fig.dpi
    assert np.isclose(gap_pt, M.FQ5_NUDGE)
    assert re.search(r'markers moved off the line of their row: descriptor checks, 80 modules, '
                     r'[+-]8 pt', capsys.readouterr().out)


def test_the_decade_ticks_are_written_in_units_a_reader_knows():
    assert [M._fq5_decade(e) for e in range(-4, 3)] == ['0.1 ms', '1 ms', '10 ms', '0.1 s',
                                                        '1 s', '10 s', '100 s']


# ================================================================== no partial mode
def test_a_missing_specialist_and_missing_checks_are_named(tmp_path):
    w = build_records(tmp_path / 'part', combos=[R for R in COMBOS if R != 'SW'],
                      checks={s: v for s, v in CHECKS.items() if s != '80x25n30'})
    with pytest.raises(SystemExit) as e:
        M.load_rule_change(w['cost'])
    msg = str(e.value)
    assert 'the training record of the specialist of SW' in msg
    for name in ('RccSpecHoursSW', 'RccCatalogueHours', 'RccDescChecksMsEighty',
                 'RccScheduleCheckMsEighty'):
        assert name in msg
    assert 'the record lists as missing: training records of the specialists SW' in msg


def test_a_missing_policy_is_named(world):
    J = load(world['cost'])
    del J['records']['released']
    del J['macros']['RccPolicyHours']
    dump(world['cost'], J)
    with pytest.raises(SystemExit, match=r'(?s)the training record of the policy.*RccPolicyHours'):
        M.load_rule_change(world['cost'])


@pytest.mark.parametrize('key,message', [
    ('cost', r'rule_change_cost\.json not found'),
    ('cost_macros', r'macro file .*macros_rulechange_cost\.tex not found'),
    ('deploy', r'deploy_cost_cpu_1thread\.json not found'),
    ('deploy_macros', r'macro file .*macros_deploy\.tex not found')])
def test_an_absent_input_is_refused(world, tmp_path, monkeypatch, key, message):
    os.remove(world[key])
    with pytest.raises(SystemExit, match=message):
        draw(world, tmp_path / 'fig', monkeypatch)
    assert not (tmp_path / 'fig').exists()


def test_a_size_without_re_planning_is_named(world):
    rec = load(world['deploy'])
    del rec['sizes']['80x25']
    dump(world['deploy'], rec)
    with pytest.raises(SystemExit,
                       match=r"no entry sizes\['80x25'\] \(re-planning at 80 modules\)"):
        M.load_deploy(world['deploy'])


# ================================================================== the paper's environment
def _stale(r):
    for k in ('environment', 'valid', 'allow_busy'):
        r.pop(k)
    r.update(torch_num_threads=4, torch_num_interop_threads=4, OMP_NUM_THREADS='4',
             MKL_NUM_THREADS='4')


@pytest.mark.parametrize('edit,messages', [
    (_stale, ['earlier environment', 'torch_num_threads = 4', 'OMP_NUM_THREADS',
              'allow_busy = None']),
    (lambda r: r.update(environment='legacy'), ["environment 'legacy'"]),
    (lambda r: r.update(allow_busy=True), ['allow_busy = True']),
    (lambda r: r.update(valid=False), ['valid = False']),
    (lambda r: r.update(model='10x25+family+jointblind-v1'), ['is not the policy of the exact']),
    (lambda r: r['sizes']['50x25'].update(cpu_share=0.5), ["sizes['50x25']: cpu_share = 0.5"]),
    (lambda r: r['sizes']['10x25'].update(valid=False, environment='legacy'),
     ["sizes['10x25']: valid = False; environment 'legacy'"]),
    (lambda r: r['sizes']['80x25'].update(torch_num_threads=4), ['torch_num_threads = 4']),
    (lambda r: r['sizes']['10x25']['per_regime'].pop('W'), ['combinations timed'])])
def test_a_record_of_p75_outside_the_paper_environment_is_refused(world, edit, messages):
    rec = load(world['deploy'])
    edit(rec)
    dump(world['deploy'], rec)
    with pytest.raises(SystemExit) as e:
        M.load_deploy(world['deploy'])
    assert "is not a record of the paper's environment" in str(e.value)
    for m in messages:
        assert m in str(e.value), m


def test_the_records_of_an_earlier_environment_on_disk_are_refused(world):
    stale = os.path.join(ROOT, 'results', 'deploy_cost_cpu_4core_pcore0to3.json')
    if not os.path.exists(stale):
        pytest.skip('no record of the earlier environment on disk')
    with pytest.raises(SystemExit, match=r'(?s)earlier environment.*torch_num_threads = 4'):
        M.load_deploy(stale)
    copy = os.path.join(ROOT, 'manuscript', 'macros_deploy.tex')   # read only
    D = M.load_deploy(world['deploy'])
    with pytest.raises(SystemExit, match=r'(?s)a macro file of another record.*one torch thread'):
        M.gate_deploy(D, copy)


@pytest.mark.parametrize('field,value,message', [
    ('generator', 'scripts/p99_rule_change_cost.py count', 'was written by'),
    ('environment', 'legacy', "holds the environment 'legacy'")])
def test_a_record_of_p99_of_another_origin_is_refused(world, field, value, message):
    J = load(world['cost'])
    J[field] = value
    dump(world['cost'], J)
    with pytest.raises(SystemExit, match=message):
        M.load_rule_change(world['cost'])


# ================================================================== the gate
def test_the_gate_matches_every_plotted_value(world):
    C = M.load_rule_change(world['cost'])
    rows = M.gate_rule_change(C, world['cost_macros'])
    D = M.load_deploy(world['deploy'])
    rows += M.gate_deploy(D, world['deploy_macros'])
    names = [r[2] for r in rows]
    assert all(r[-1] == 'match' for r in rows)
    assert set(names) >= ({'RccPolicyHours', 'RccCatalogueHours'}
                          | {f'RccSpecHours{R}' for R in COMBOS}
                          | {f'Rcc{s}Ms{w}' for s in ('DescChecks', 'ScheduleCheck')
                             for w in ('Ten', 'Fifty', 'Eighty')}
                          | {f'DeployLatency{w}' for w in ('Ten', 'Fifty', 'Eighty')})
    assert {m: round(t['seconds'], 6) for m, t in D['times'].items()} == {10: 0.41, 50: 3.05,
                                                                         80: 6.12}
    assert {t['regime'] for t in D['times'].values()} == {'LSW'}


@pytest.mark.parametrize('key,name,new', [
    ('cost_macros', 'RccSpecHoursLS', '1.33'),
    ('cost_macros', 'RccPolicyHours', '2.50'),
    ('cost_macros', 'RccCatalogueHours', '9.79'),
    ('cost_macros', 'RccDescChecksMsTen', '1.21'),
    ('cost_macros', 'RccScheduleCheckMsEighty', '58.1'),
    ('cost_macros', 'RccSpecMeasuredN', '7'),
    ('deploy_macros', 'DeployLatencyFifty', '3.07')])
def test_the_gate_refuses_a_macro_that_disagrees_in_one_number(world, tmp_path, monkeypatch, key,
                                                               name, new):
    edit_macro(world[key], name, new)
    with pytest.raises(SystemExit, match=name):
        draw(world, tmp_path / 'fig', monkeypatch)


def test_the_gate_refuses_a_deploy_macro_file_of_another_record_or_thread_count(world, tmp_path):
    D = M.load_deploy(world['deploy'])
    other = tmp_path / 'other.json'
    shutil.copy(world['deploy'], other)
    path = write_deploy_macros(load(other), other, tmp_path / 'o')
    with pytest.raises(SystemExit, match='a macro file of another record'):
        M.gate_deploy(D, path)
    rec = load(world['deploy'])
    rec['torch_num_threads'] = 4
    path = write_deploy_macros(rec, world['deploy'], tmp_path / 't')
    with pytest.raises(SystemExit, match='with one torch thread'):
        M.gate_deploy(D, path)


# ================================================================== names and main
def test_the_names_are_those_of_p99_and_p75():
    assert (M.FQ5_JSON, M.FQ5_MACROS) == (P99.OUT_JSON, P99.OUT_MACROS)
    src = open(os.path.join(ROOT, 'scripts', 'p99_rule_change_cost.py')).read()
    assert f"'generator': '{M.FQ5_GENERATOR}'" in src
    assert P75.out_paths()[1].endswith(M.FQ5_DEPLOY_MACROS)
    assert M.FQ5_DEPLOY_JSON == 'results/deploy_cost_cpu_1thread.json'   # --torch_threads 1
    assert "f'deploy_cost_cpu_{args.torch_threads}thread{args.out_tag}.json'" in \
        open(os.path.join(ROOT, 'scripts', 'p75_deploy_cost.py')).read()
    sizes = P99.DEFAULT_SIZES.split(',')
    assert [P99.modules(s) for s in sizes] == sorted(M.FQ5_SHAPE)
    assert all(M._fq5_shop(s) in P75.DEFAULT_SIZES.split(',') for s in sizes)
    assert list(P75.REGIMES) == COMBOS


def test_the_p75_keys_the_figure_reads_are_those_p75_writes():
    src = open(os.path.join(ROOT, 'scripts', 'p75_deploy_cost.py')).read()
    for snippet in ("'environment': REG.env()", "'allow_busy': args.allow_busy",
                    "rec['valid'] = valid", "r['valid'] = ok", "'device': 'cpu'",
                    "'OMP_NUM_THREADS': str(args.numpy_threads)",
                    "'MKL_NUM_THREADS': str(args.numpy_threads)",
                    "for k in ('torch_num_threads', 'torch_num_interop_threads',",
                    "'cpu_share': cpu_all / (wall_all * torch.get_num_threads())",
                    "'per_regime': per_regime", "'model': name",
                    "f'% source: {src}'", "f'% model {rec[\"model\"]}, CPU, torch threads '"):
        assert snippet in src, snippet
    rec = p75_record()
    for k in ('environment', 'allow_busy', 'valid', 'device', 'OMP_NUM_THREADS', 'model',
              'torch_num_threads', 'torch_num_interop_threads'):
        assert k in rec
    assert P75.worst_regime(rec['sizes']['80x25'])[0] == 'LSW'


NAMES = ('fq1_specialists', 'fq3_time_budget', 'fq4_boundary', 'fs_convergence', 'fq2_reuse',
         'fq2_heldout', 'fq5_rule_change')


def _record_calls(monkeypatch):
    calls = []
    for name in NAMES:
        monkeypatch.setattr(M, name, lambda *a, _n=name: calls.append((_n, a)) or [])
    return calls


def test_the_run_without_only_does_not_draw_fq5(monkeypatch):
    for v in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--factory-seeds', 'five'])
    monkeypatch.chdir(ROOT)
    M.main()
    called = [n for n, _ in calls]
    assert called[0] == 'fq1_specialists' and 'fq5_rule_change' not in called


def test_only_fq5_draws_it_alone_from_the_registry_paths(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    monkeypatch.setenv('PACT_OUT', 'out/exact')
    monkeypatch.delenv('PACT_RESULTS', raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--only', 'fq5_rule_change'])
    monkeypatch.chdir(ROOT)
    M.main()
    ((name, a),) = calls
    assert name == 'fq5_rule_change'
    ex = os.path.join(ROOT, 'out', 'exact')
    assert a == (os.path.join(ex, 'results', 'rule_change_cost.json'),
                 os.path.join(ex, 'manuscript', 'macros_rulechange_cost.tex'),
                 os.path.join(ex, 'results', 'deploy_cost_cpu_1thread.json'),
                 os.path.join(ex, 'paper', 'macros_deploy.tex'),
                 os.path.join(ROOT, M.OUT_DEFAULT))


def test_paths_given_on_the_command_line_are_used(monkeypatch, tmp_path):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_OUT', 'out/exact')
    calls = _record_calls(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--only', 'fq5_rule_change',
                                      '--rule-change', 'a.json', '--rule-change-macros', 'a.tex',
                                      '--deploy', 'b.json', '--deploy-macros', 'b.tex',
                                      '--out', 'figs'])
    M.main()
    ((_, a),) = calls
    assert a == tuple(str(tmp_path / p) for p in ('a.json', 'a.tex', 'b.json', 'b.tex', 'figs'))
