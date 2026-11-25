"""The figure fq2_heldout of scripts/make_figures.py: each trained model against the specialist
of the combination on the four combinations of two or three rules, filled where the model trained
on the combination and open where it was held out of its training.

  * the synthetic records are written by the real writers: the five- and ten-seed records of the
    policy with tests/test_fq1_inputs.make_world (p110's own macro_lines), the record of policies
    b and c by p122_heldout_splits.dry_run on synthetic arrays, the record of the policy trained
    on all eight by p95_secondary_inference.main on the synthetic tree of
    tests/test_p95_secondary_inference.build, so a renamed key breaks these tests;
  * the full figure draws exactly the records: per column five intervals and markers in the
    fixed order policy, policy b, policy c, policy trained on all eight, single-rule control,
    evenly spaced; the shape names the model, and the fill follows each model's list of training
    combinations (a changed list in the record changes the fill); no text inside the axes; the
    tick labels and the legend set the combination names in math italic ($LS$), and the printed
    lines keep the plain names;
  * the policy and the control are the rows fq1_inputs reads, in mode five and in mode ten;
  * the full mode refuses and names every absent or incomplete model; the partial mode draws the
    others and writes only fq2_heldout_partial.*; a record of a dry run of p122 is drawn only
    with allow_partial (file names ending in _dev);
  * the gate holds every plotted value with a macro against it and refuses one number that
    disagrees or a seed count of another run; a plotted value without a macro is compared with
    the record only;
  * records of another generator, sign, environment or training list, and cells whose verdict
    the interval does not give, are refused;
  * main draws fq2_heldout only with --only fq2_heldout, from the registry paths.

Synthetic records only, written under the pytest temporary folder; nothing is trained or
evaluated, and no display is needed. The record of p122 needs the descriptors of
data/FAMILY/10x25 (its loader checks the base instances).
Run from the repository root:  python -B -m pytest tests/test_fq2_heldout.py -q
"""
import json
import os
import pathlib
import re
import shutil
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
import p95_secondary_inference as P95  # noqa: E402
import p122_heldout_splits as P122  # noqa: E402
import xsplit_runs as XS  # noqa: E402
import test_fq1_inputs as T1  # noqa: E402  (the five- and ten-seed records of the policy)
import test_p95_secondary_inference as T95  # noqa: E402  (the synthetic tree of p95)

COLS = list(M.FQH_COLUMNS)
ORDER = ['policy', 'b', 'c', 'all8', 'control']
SHAPE = {'policy': 'o', 'b': '^', 'c': 'v', 'all8': 'D', 'control': 's'}
SPLIT_SEEDS = (301, 302, 303)
BOOT = 200

needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25/desc')),
                                reason='needs the descriptors of data/FAMILY/10x25')


# ================================================================== the synthetic records
def build_records(base):
    """Every record the figure reads, written by the real writers under base. Returns the
    paths and the records."""
    base = str(base)
    w = T1.make_world(pathlib.Path(base) / 'policy')
    # policies b and c: p122 on synthetic arrays, three seeds (its dry run)
    old_b, old_cwd = P122.F88.B, os.getcwd()
    try:
        os.chdir(ROOT)
        P122.dry_run(os.path.join(base, 'p122'), boot=BOOT, seeds=SPLIT_SEEDS)
    finally:
        P122.F88.configure(boot=old_b)
        os.chdir(old_cwd)
    # the policy trained on all eight: p95 on the synthetic tree of its own test
    tree = os.path.join(base, 'p95')
    os.makedirs(tree, exist_ok=True)
    saved = {v: os.environ.pop(v, None) for v in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT',
                                                  'PACT_RESULTS')}
    try:
        T95.build(tree, arms=('all8',), hgnn=False, factors={'all8': 1.004})
        P95.main(['--root', tree, '--out', os.path.join(tree, 'out'), '--boot', '400'])
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return dict(five=w['five'], five_macros=w['five_macros'], ten=w['ten'],
                ten_macros=w['ten_macros'], public=w['public'], public_macros=w['public_macros'],
                splits=os.path.join(base, 'p122', 'out', P122.OUT_FILES['json']),
                splits_macros=os.path.join(base, 'p122', 'out', P122.OUT_FILES['macros']),
                sec=os.path.join(tree, 'out', P95.OUT_FILES[1]),
                sec_macros=os.path.join(tree, 'out', P95.OUT_FILES[0]))


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    if not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25/desc')):
        pytest.skip('needs the descriptors of data/FAMILY/10x25')
    return build_records(tmp_path_factory.mktemp('records'))


@pytest.fixture
def world(records, tmp_path):
    """A private copy of the records (a test may change them), the p122 record marked as a
    full run unless a test says otherwise."""
    out = {}
    for k, p in records.items():
        q = tmp_path / 'w' / k / os.path.basename(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, q)
        out[k] = str(q)
    rec = load(out['splits'])
    assert rec['dry'] is True                      # what the dry run of p122 writes
    rec['dry'] = False
    dump(out['splits'], rec)
    return out


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def args(w, **kw):
    a = dict(factory_json=w['five'], factory_macros=w['five_macros'], factory_seeds='ten',
             seedext_json=w['ten'], seedext_macros=w['ten_macros'], splits_json=w['splits'],
             splits_macros=w['splits_macros'], sec_json=w['sec'], sec_macros=w['sec_macros'])
    a.update(kw)
    return a


def draw(w, out, monkeypatch, **kw):
    kept = []
    monkeypatch.setattr(M.plt, 'close', kept.append)
    written = M.fq2_heldout(out_dir=str(out), **args(w, **kw))
    return written, kept[0]


def _edit_macro(path, name, new):
    text = open(path).read()
    pat = re.compile(r'(\\newcommand\{\\' + name + r'\})\{[^}]*\}')
    assert len(pat.findall(text)) == 1, name
    open(path, 'w').write(pat.sub(lambda m: m.group(1) + '{' + new + '}', text))


def want_rows(w):
    """{model: {column: (mean, lo, hi)}} and {model: training list} from the records."""
    ten, five = load(w['ten']), load(w['five'])
    spl, sec = load(w['splits']), load(w['sec'])
    a8 = sec['blocks']['all8']['result']

    def t(c):
        return (c['mean'], c['lo'], c['hi'])
    rows = {'policy': {c: t(ten['ten'][c]) for c in COLS},
            'b': {c: t(spl['split']['b']['against_specialist'][c]) for c in COLS},
            'c': {c: t(spl['split']['c']['against_specialist'][c]) for c in COLS},
            'all8': {c: t(a8['vs_specialist']['components'][c]) for c in COLS},
            'control': {c: t(five['control']['components']['Spec'][c]) for c in COLS}}
    train = {'policy': REG.TRAIN_COMBOS, 'b': XS.SPLITS['b'], 'c': XS.SPLITS['c'],
             'all8': tuple(a8['effort']['combos']), 'control': REG.CONTROL_COMBOS}
    return rows, train


def marks(ax):
    """Intervals {x: (lo, hi)} and markers [(x, y, marker, filled)] of the axes."""
    iv, mk = {}, []
    for ln in ax.lines:
        x, y = list(map(float, ln.get_xdata())), list(map(float, ln.get_ydata()))
        if len(x) == 2 and x[0] == x[1]:
            iv[x[0]] = (y[0], y[1])
        elif len(x) == 1:
            mk.append((x[0], y[0], ln.get_marker(), str(ln.get_markerfacecolor()) != 'white'))
    return iv, mk


# ================================================================== the figure
@needs_data
def test_the_full_figure_draws_the_records_with_the_fill_of_each_training_list(world, tmp_path,
                                                                               monkeypatch):
    written, fig = draw(world, tmp_path / 'fig', monkeypatch)
    assert sorted(os.path.basename(p) for p in written) == ['fq2_heldout.pdf', 'fq2_heldout.png']
    assert all(os.path.isfile(p) for p in written)
    (ax,) = fig.axes
    iv, mk = marks(ax)
    rows, train = want_rows(world)
    assert len(mk) == len(iv) == len(COLS) * len(ORDER)
    for j, c in enumerate(COLS):
        col = sorted((m for m in mk if abs(m[0] - j) < 0.5), key=lambda m: m[0])
        assert [m[2] for m in col] == [SHAPE[k] for k in ORDER]
        gaps = np.diff([m[0] for m in col])
        assert np.allclose(gaps, gaps[0]) and abs(np.mean([m[0] for m in col]) - j) < 1e-9
        for (x, y, shape, filled), key in zip(col, ORDER):
            assert y == rows[key][c][0]
            assert iv[x] == rows[key][c][1:]
            assert filled == (c in train[key]), (key, c)
    # the fill as the training lists give it: the policy holds out SW and LSW, b LS and LSW,
    # c LW and LSW; the policy on all eight trains on all four; the control on none
    fills = {(COLS[int(round(m[0]))], ORDER[[SHAPE[k] for k in ORDER].index(m[2])]): m[3]
             for m in mk}
    assert [c for c in COLS if not fills[(c, 'policy')]] == ['SW', 'LSW']
    assert [c for c in COLS if not fills[(c, 'b')]] == ['LS', 'LSW']
    assert [c for c in COLS if not fills[(c, 'c')]] == ['LW', 'LSW']
    assert all(fills[(c, 'all8')] for c in COLS) and not any(fills[(c, 'control')] for c in COLS)
    # band, zero line, no text inside the axes, size, ticks and legend
    assert [(p.get_y(), p.get_y() + p.get_height()) for p in ax.patches] == [(-1.0, 1.0)]
    assert (0, 0) in [tuple(ln.get_ydata()) for ln in ax.lines
                      if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] != ln.get_xdata()[1]]
    assert not ax.texts
    assert tuple(round(v, 2) for v in fig.get_size_inches()) == (M.FIG_W, 2.80)
    # the combination names in math italic, in the ticks and inside the legend's phrases
    assert [t.get_text() for t in ax.get_xticklabels()] == ['$LS$', '$LW$', '$SW$', '$LSW$']
    assert ax.get_ylabel() == 'difference from the specialist (%)'
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        'policy', 'policy, $LS$ and $LSW$ held out', 'policy, $LW$ and $LSW$ held out',
        'policy trained on all eight', 'single-rule control', 'trained on the combination',
        'held out of training', '\u00b11% margin']


@needs_data
def test_the_fill_is_read_from_the_training_list_of_the_record(world, tmp_path, monkeypatch):
    rec = load(world['sec'])
    rec['blocks']['all8']['result']['effort']['combos'] = list(REG.TRAIN_COMBOS)
    dump(world['sec'], rec)
    _, fig = draw(world, tmp_path / 'fig', monkeypatch)
    _, mk = marks(fig.axes[0])
    diamonds = sorted((m[0], m[3]) for m in mk if m[2] == 'D')
    assert [f for _, f in diamonds] == [True, True, False, False]   # LS, LW trained; SW, LSW not


@needs_data
@pytest.mark.parametrize('mode', ['five', 'ten'])
def test_the_policy_and_the_control_are_the_rows_of_fq1(world, mode):
    X = M.fq2_heldout_inputs(**args(world, factory_seeds=mode))
    F = T1.inputs(world, factory_seeds=mode)['F']        # what fq1 draws in panel (a)
    got = {m['key']: m for m in X['models']}
    for c in COLS:
        assert got['policy']['rows'][c] == F['rows'][c]
        assert got['control']['rows'][c] == F['ctl'][c]
    assert got['policy']['seeds'] == F['seeds'] == (T1.TEN_SEEDS if mode == 'ten'
                                                    else T1.FIVE_SEEDS)
    assert got['control']['seeds'] == T1.FIVE_SEEDS


# ================================================================== full and partial modes
@needs_data
def test_the_full_mode_refuses_and_names_every_missing_model(world, tmp_path, monkeypatch):
    absent = str(tmp_path / 'absent.json')
    with pytest.raises(SystemExit, match=r'(?s)missing or incomplete:.*policies b and c: .*'
                                         r'not found.*policy trained on all eight: .*not found'):
        M.fq2_heldout_inputs(**args(world, splits_json=absent, sec_json=absent))
    out = tmp_path / 'fig'
    written, fig = draw(world, out, monkeypatch, splits_json=absent, sec_json=absent,
                        partial=True)
    assert sorted(os.listdir(out)) == ['fq2_heldout_partial.pdf', 'fq2_heldout_partial.png']
    assert sorted(os.path.basename(p) for p in written) == sorted(os.listdir(out))
    _, mk = marks(fig.axes[0])
    assert sorted({m[2] for m in mk}) == ['o', 's'] and len(mk) == 2 * len(COLS)
    assert [t.get_text() for t in fig.legends[0].get_texts()][:2] == ['policy',
                                                                       'single-rule control']


@needs_data
def test_the_names_are_italic_in_the_legend_and_plain_in_the_printed_lines(world):
    X = M.fq2_heldout_inputs(**args(world))
    got = {m['key']: m for m in X['models']}
    assert (got['b']['text'], got['b']['legend']) == ('policy, LS and LSW held out',
                                                      'policy, $LS$ and $LSW$ held out')
    assert (got['c']['text'], got['c']['legend']) == ('policy, LW and LSW held out',
                                                      'policy, $LW$ and $LSW$ held out')
    assert all(got[k]['legend'] == got[k]['text'] for k in ('policy', 'all8', 'control'))
    assert any(ln.startswith('    policy, LS and LSW held out: ') for ln in X['lines'])
    assert not any('$' in ln for ln in X['lines'])


@needs_data
def test_a_record_with_fewer_seeds_is_drawn_in_the_partial_mode_only(world, tmp_path,
                                                                      monkeypatch):
    rec = load(world['splits'])
    rec['seeds'] = [301, 302]
    dump(world['splits'], rec)
    _edit_macro(world['splits_macros'], 'HspSeeds', '2')
    with pytest.raises(SystemExit, match='holds the training seeds 301,302, the runs are '
                                         'trained for 301,302,303'):
        M.fq2_heldout_inputs(**args(world))
    X = M.fq2_heldout_inputs(**args(world, partial=True))
    assert [m['key'] for m in X['models']] == ORDER
    assert any('drawn, but' in ln and '301,302,' in ln for ln in X['lines'])
    rec = load(world['sec'])
    rec['blocks']['all8']['status'] = 'refused'
    rec['blocks']['all8']['problems'] = ['missing seed']
    dump(world['sec'], rec)
    X = M.fq2_heldout_inputs(**args(world, partial=True))
    assert 'all8' not in [m['key'] for m in X['models']]
    assert any('left out' in ln and 'block all8' in ln and 'refused' in ln for ln in X['lines'])


@needs_data
def test_a_dry_run_of_p122_is_drawn_only_with_allow_partial(world, tmp_path, monkeypatch):
    rec = load(world['splits'])
    rec['dry'] = True
    dump(world['splits'], rec)
    with pytest.raises(SystemExit, match='dry run of p122'):
        M.fq2_heldout_inputs(**args(world))
    written, _ = draw(world, tmp_path / 'fig', monkeypatch, allow_partial=True)
    assert sorted(os.path.basename(p) for p in written) == ['fq2_heldout_dev.pdf',
                                                             'fq2_heldout_dev.png']


# ================================================================== the gate
@needs_data
def test_the_gate_matches_every_plotted_value_with_a_macro(world):
    X = M.fq2_heldout_inputs(**args(world))
    status = {(k, c): s for k, c, _, _, s in X['table']}
    assert all(status[(k, c)] == 'match' for k in ('policy', 'all8', 'control') for c in COLS)
    for x in ('b', 'c'):
        for c in COLS:
            assert status[(x, c)] == 'match', (x, c)


@needs_data
@pytest.mark.parametrize('key,name,shift', [('splits_macros', 'HspBLSHi', 0.02),
                                            ('splits_macros', 'HspCLSW', -0.02),
                                            ('sec_macros', 'SecAllEightParSWLo', 0.02),
                                            ('ten_macros', 'SeedExtLSW', 0.02),
                                            ('five_macros', 'FamCtlSpecLWHi', 0.02)])
def test_the_gate_refuses_a_macro_that_disagrees_in_one_number(world, key, name, shift):
    text = open(world[key]).read()
    old = re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', text).group(1)
    _edit_macro(world[key], name, P122.f2(float(old) + shift))
    with pytest.raises(SystemExit, match=name):
        M.fq2_heldout_inputs(**args(world))


@needs_data
def test_the_gate_refuses_a_seed_count_of_another_run_or_a_missing_macro_file(world, tmp_path):
    _edit_macro(world['sec_macros'], 'SecSeedN', '5')
    with pytest.raises(SystemExit, match=r'SecSeedN = 5, record 3'):
        M.fq2_heldout_inputs(**args(world))
    with pytest.raises(SystemExit, match='macro file .* not found'):
        M.fq2_heldout_inputs(**args(world, splits_macros=str(tmp_path / 'absent.tex')))


# ================================================================== the records
@needs_data
@pytest.mark.parametrize('which,edit,message', [
    ('splits', lambda r: r.update(generator='scripts/p88_family_inference.py'), 'was written by'),
    ('splits', lambda r: r['definitions'].update(against_specialist='100 (specialist - policy)'),
     'defines the difference'),
    ('splits', lambda r: r['split']['b'].update(train=['N', 'L', 'S', 'W', 'LS', 'LW']),
     'xsplit_runs trains it on'),
    ('splits', lambda r: r['split']['c']['against_specialist']['LSW'].update(direction='ahead'),
     'direction'),
    ('sec', lambda r: r.update(check=True), 'earlier-check'),
    ('sec', lambda r: r.update(sign='d = 100 (b - a) / a'), 'states the sign'),
    ('sec', lambda r: r['blocks']['all8']['result'].update(stem='xmask-pair'), 'reads xmask-pair'),
    ('sec', lambda r: r['blocks']['all8']['result']['vs_specialist']['components']['LW'].update(
        mean=99.0), 'outside'),
])
def test_records_of_another_origin_or_with_a_wrong_cell_are_refused(world, which, edit, message):
    rec = load(world[which])
    edit(rec)
    dump(world[which], rec)
    with pytest.raises(SystemExit, match=message):
        M.fq2_heldout_inputs(**args(world, partial=True))


@needs_data
def test_a_verdict_that_the_interval_does_not_give_is_refused(world):
    rec = load(world['splits'])
    c = rec['split']['b']['against_specialist']['SW']
    c['v1.0'] = 'inferior' if c['v1.0'] != 'inferior' else 'equivalent'
    dump(world['splits'], rec)
    with pytest.raises(SystemExit, match=r'split\.b\.against_specialist\.SW: verdict'):
        M.fq2_heldout_inputs(**args(world))


def test_the_names_are_those_of_p122_and_p95():
    assert M.FQH_SPLITS_JSON == P122.OUT_FILES['json']
    assert M.FQH_SPLITS_MACROS == P122.OUT_FILES['macros']
    assert M.FQH_SPLITS_GENERATOR == P122.GENERATOR and M.FQH_SPLITS_PREFIX == P122.PREFIX
    assert M.FQH_SPLIT_WORD == P122.WORD and tuple(M.FQH_SPLIT_WORD) == tuple(XS.SPLITS)
    assert (M.FQH_SEC_MACROS, M.FQH_SEC_JSON) == P95.OUT_FILES[:2]
    assert M.FQH_SEC_GENERATOR == P95.GENERATOR and M.FQH_SEC_FOLDER == P95.FOLDER
    block, stem, word = M.FQH_ALL8
    assert P95.block_name(stem) == block and P95.word(stem) == word
    assert stem in REG.SECONDARY_ARMS
    assert set(M.FQH_COLUMNS) == {r for r in REG.COMBOS if len(r) >= 2}


# ================================================================== selection in main
NAMES = ('fq1_specialists', 'fq3_time_budget', 'fq4_boundary', 'fs_convergence', 'fq2_reuse',
         'fq2_heldout')


def _record_calls(monkeypatch):
    calls = []
    for name in NAMES:
        monkeypatch.setattr(M, name, lambda *a, _n=name: calls.append((_n, a)) or [])
    return calls


def test_the_run_without_only_does_not_draw_fq2_heldout(monkeypatch):
    for v in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--factory-seeds', 'five'])
    monkeypatch.chdir(ROOT)
    M.main()
    called = [n for n, _ in calls]
    assert called[0] == 'fq1_specialists' and 'fq2_heldout' not in called


@pytest.mark.parametrize('extra,partial', [([], False), (['--heldout-partial'], True)])
def test_only_fq2_heldout_draws_it_alone_from_the_registry_paths(monkeypatch, extra, partial):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    monkeypatch.setenv('PACT_OUT', 'out/exact')
    monkeypatch.delenv('PACT_RESULTS', raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--only', 'fq2_heldout',
                                      '--factory-seeds', 'five'] + extra)
    monkeypatch.chdir(ROOT)
    M.main()
    ((name, a),) = calls
    assert name == 'fq2_heldout'
    (fj, fm, mode, sx, sxm, hj, hm, sj, sm, out, part, allow) = a
    ex = os.path.join(ROOT, 'out', 'exact')
    assert fj == os.path.join(ex, 'results', 'family_inference.json')
    assert fm == os.path.join(ex, 'paper', 'macros_family.tex')
    assert (mode, sxm) == ('five', None)
    assert hj == os.path.join(ex, 'results', 'heldout_splits.json')
    assert hm == os.path.join(ex, 'manuscript', 'macros_heldout.tex')
    assert sj == os.path.join(ROOT, 'results', 'secondary_inference.json')
    assert sm == os.path.join(ROOT, 'manuscript', 'macros_secondary.tex')
    assert out == os.path.join(ROOT, M.OUT_DEFAULT) and (part, allow) == (partial, False)
