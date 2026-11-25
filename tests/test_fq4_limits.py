"""The figure fq4_limits of scripts/make_figures.py: where the policy reaches its limits,
(a) at the shop sizes 10 to 80 modules and (b) at the 22 changed levels of the sensitivity
study, the policy against the specialists and against the best dispatching rule.

  * the synthetic records are written by the real writers: the dry run of
    p128_production_specialists.py (the records of p88, p118 and p128 and the macro files of
    p118 and p128), the macro lines of p88 (section_H_macros, row_line) for its record, and the
    infer phase of p87_sensitivity.py on the synthetic result tree of tests/test_p87_infer.py
    with the five training seeds and the exact mask; the default level of p87 then holds the
    10-module rows of p88 and p128, as one evaluation does;
  * the full figure, in both variants, draws exactly the stored rows: six sizes and 22 levels,
    two markers each with their intervals, the specialists' series as filled circles and the
    rule's as open squares, in the order of the supplementary table;
  * the full mode refuses and names every missing size; the partial mode leaves out the sizes
    without a record, lists them and writes the files ending in _partial; the record of p87 is
    needed in every mode;
  * a record of another generator, other seeds or another environment is refused, and a record
    of a dry or partial run is drawn only with allow_partial (then the files end in _dev);
  * the 10-module rows must be one evaluation in p88, p87 and p128, and the 20 and 30 module
    rows of p88 and p118 must be equal; a level that sets another parameter or changes other
    combinations is refused;
  * the gate holds every plotted mean and interval end against its macro and refuses one number
    that disagrees, a missing macro or macro file, and a manuscript copy that prints otherwise;
  * the layout: 6.30 in wide, at most 3.1 in (A) or 4.6 in (B) high, black text of 8 pt or
    more, no two texts overlapping, the legend clear of the panels, the PDF with the Times face
    only; a layout that breaks a rule is refused; the group labels of variant A, of at most
    three lines, name their parameters in words, every line of them on the baseline of its
    row and centred under its group, and a third line makes the figure taller by one line
    below the axes while the panels keep their size;
  * the levels, their words, parameters and combinations are those of p87 and of the table, and
    the record and macro names those of p118 and p128;
  * main draws fq4_limits only with --only fq4_limits, from the registry paths.

Synthetic records only, written under the pytest temporary folder; nothing is trained or
evaluated and no display is needed. Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -B -m pytest -p no:cacheprovider
      --basetemp=<scratch> tests/test_fq4_limits.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=None,
                                        PACT_RESULTS=None)

import make_figures as M  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import p118_per_floor as P118  # noqa: E402
import p128_production_specialists as P128  # noqa: E402
import test_p87_infer as T87  # noqa: E402  (the synthetic result tree of p87, and p87 itself)

P87 = T87.P87
SEEDS = [301, 302, 303, 304, 305]
SIZES = (10, 20, 27, 30, 50, 80)
TABLE = os.path.join(ROOT, 'manuscript', 'sections_new', 'tables', 'ts_sensitivity.tex')
needs_table = pytest.mark.skipif(
    not os.path.isfile(TABLE),
    reason='needs the table of the manuscript manuscript/sections_new/tables/ts_sensitivity.tex, '
           'which the release does not hold')
VALUE_LABELS = ['0.1', '0.2', '0.3', '0.75', '1.25', '0.5', '0.75', '1.25', '1.5', '0.5', '0.75',
                '1.25', '1.5', 'zero', '7 of 25', '21 of 25', '0.5', '1.5', '2', '15', 'concrete',
                'steel']
LEGEND = ['against the specialists', 'against the best dispatching rule', '\u00b11% margin']


# ================================================================== the synthetic records
def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def p128_env():
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS'):
        e.pop(k, None)
    return e


def edit_macro(path, name, new):
    text = open(path).read()
    pat = re.compile(r'(\\newcommand\{\\' + name + r'\})\{[^}]*\}')
    assert len(pat.findall(text)) == 1, name
    open(path, 'w').write(pat.sub(lambda m: m.group(1) + '{' + new + '}', text))


def drop_macro(path, name):
    text = open(path).read()
    pat = re.compile(r'\\newcommand\{\\' + name + r'\}\{[^}]*\}')
    assert len(pat.findall(text)) == 1, name
    open(path, 'w').write(pat.sub('', text))


def build_records(base):
    """Every record and macro file the figure reads, written by the real writers under base."""
    base = str(base)
    tree = os.path.join(base, 'p128')
    r = subprocess.run([sys.executable, '-B', os.path.join(ROOT, 'scripts',
                                                           'p128_production_specialists.py'),
                        '--dry', tree], capture_output=True, text=True, cwd=ROOT, env=p128_env())
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = os.path.join(tree, 'out')
    w = dict(fam=os.path.join(out, 'results', 'family_inference.json'),
             per=os.path.join(out, P118.OUT_FILES['json']),
             per_m=os.path.join(out, P118.OUT_FILES['macros']),
             prod=os.path.join(out, P128.OUT_FILES['json']),
             prod_m=os.path.join(out, P128.OUT_FILES['macros']),
             fam_m=os.path.join(base, 'paper', 'macros_family.tex'))
    F = load(w['fam'])
    os.makedirs(os.path.dirname(w['fam_m']), exist_ok=True)
    with open(w['fam_m'], 'w') as f:                    # the macro lines p88 writes
        f.write('\n'.join(F88.section_H_macros(F['larger_shops'])
                          + [F88.row_line('FamParPoolAll', F['parity']['pooled']['All'],
                                          '10x25, policy vs specialist, pooled over eight')])
                + '\n')
    # p87 infer on the synthetic tree, five seeds, the exact mask
    root = os.path.join(base, 'p87')
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(P87, 'level_names', T87.names_of)
        mp.setitem(P87.ARM_NAMES, 'scheme', 'xmask')
        T87.build_tree(os.path.join(root, 'tree'), seeds=SEEDS, mask='exact', rule_set='masked')
        P87.infer(T87.args_for(os.path.join(root, 'tree'), os.path.join(root, 'out'),
                               mask='exact', seeds=list(SEEDS)))
    finally:
        mp.undo()
    files = P87.infer_outputs(os.path.join(root, 'out'))
    w.update(sens=files['json'], sens_m=files['macros'])
    # one evaluation of the 10-module instances: the default level holds the rows of p88 and p128
    S, P = load(w['sens']), load(w['prod'])
    pool = S['levels']['default']['pools']['pool']
    for kind, row, word in (('pol_spec', F['parity']['pooled']['All'], 'Spec'),
                            ('pol_rule', P['side_by_side']['rows']['10x25']['rule'], 'Rule')):
        for fld, suffix in (('mean', ''), ('lo', 'Lo'), ('hi', 'Hi')):
            pool[kind][fld] = row[fld]
            edit_macro(w['sens_m'], f'SensDefaultPol{word}Pool{suffix}', F88.f2(row[fld]))
    dump(w['sens'], S)
    return w


@pytest.fixture(scope='module')
def records(tmp_path_factory):
    return build_records(tmp_path_factory.mktemp('records'))


@pytest.fixture
def world(records, tmp_path):
    """A private copy of the records (a test may change them), the records of p118 and p128
    marked as full runs unless a test says otherwise."""
    out = {}
    for k, p in records.items():
        q = tmp_path / 'w' / k / os.path.basename(p)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, q)
        out[k] = str(q)
    for k in ('per', 'prod'):
        rec = load(out[k])
        assert rec['dry'] is True                      # what the dry run writes
        rec['dry'] = False
        dump(out[k], rec)
    return out


def args(w, **kw):
    a = dict(family_json=w['fam'], family_macros=w['fam_m'], perfloor_json=w['per'],
             perfloor_macros=w['per_m'], prod_json=w['prod'], prod_macros=w['prod_m'],
             sens_json=w['sens'], sens_macros=w['sens_m'], paper_dir=None)
    a.update(kw)
    return a


def draw(w, out, monkeypatch, **kw):
    kept = []
    monkeypatch.setattr(M.plt, 'close', kept.append)
    written = M.fq4_limits(out_dir=str(out), **args(w, **kw))
    return written, kept[0]


def table_rows():
    """(level word, Pool or Reached) of the supplementary table, in its order, default excepted."""
    rows = re.findall(r'\\sndSensRow\{(\w+)\}\{(\w+)\}\{[^}]*\}', open(TABLE).read())
    return [r for r in rows if r[0] != 'Default']


def want(w):
    """{size: (spec, rule)} and [(spec, rule)] for the levels in the table's order, each
    (mean, lo, hi), read from the records."""
    F, P1, P2, S = load(w['fam']), load(w['per']), load(w['prod']), load(w['sens'])

    def t(x):
        return (x['mean'], x['lo'], x['hi'])

    def pair(x):
        return (t(x['parity_pooled']['All']), t(x['rule_pool_pooled']))
    sizes = {10: (t(F['parity']['pooled']['All']),
                  t(S['levels']['default']['pools']['pool']['pol_rule'])),
             20: pair(F['larger_shops']['20x25']), 30: pair(F['larger_shops']['30x25']),
             27: pair(P1['sizes']['27x25']), 50: pair(P2['sizes']['50x25n30']),
             80: pair(P2['sizes']['80x25n30'])}
    key_of = {v: k for k, v in P87.LEVEL_WORDS.items()}
    levels = []
    for word, row in table_rows():
        pool = S['levels'][key_of[word]]['pools']['pool' if row == 'Pool' else 'reached']
        levels.append((t(pool['pol_spec']), t(pool['pol_rule'])))
    return sizes, levels


def marks(ax, vertical):
    """Markers [(position, value, shape, filled, edge colour)] and intervals [(position, lo, hi)]
    of one panel, in the order drawn."""
    mk, iv = [], []
    for ln in ax.lines:
        x, y = list(map(float, ln.get_xdata())), list(map(float, ln.get_ydata()))
        pos, val = (x, y) if vertical else (y, x)
        if len(x) == 1:
            mk.append((pos[0], val[0], ln.get_marker(),
                       to_rgba(ln.get_markerfacecolor()) != to_rgba('white'),
                       to_rgba(ln.get_markeredgecolor())))
        elif len(x) == 2 and pos[0] == pos[1] and ln.get_transform() == ax.transData:
            iv.append((pos[0], val[0], val[1]))
    return mk, iv


def check_panel(ax, vertical, expected):
    """expected: [(spec, rule)], each (mean, lo, hi); two marks per position, spec left (above)."""
    mk, iv = marks(ax, vertical)
    assert len(mk) == len(iv) == 2 * len(expected)
    for j, (spec, rule) in enumerate(expected):
        for k, (kind, it) in enumerate((('spec', spec), ('rule', rule))):
            p, v, shape, filled, edge = mk[2 * j + k]
            q, lo, hi = iv[2 * j + k]
            assert (p, v) == (q, it[0]) and (lo, hi) == (it[1], it[2])
            assert shape == M.FQL_STYLE[kind]['marker']
            assert filled == (kind == 'spec')
            if kind == 'rule':
                assert edge == to_rgba(M.FQL_RULE_COLOR)
        assert mk[2 * j][0] < mk[2 * j + 1][0]           # the specialists' mark first
        if j:
            assert mk[2 * j][0] > mk[2 * j - 1][0]       # positions in order


def tick_texts(labels):
    return [t.get_text() for t in labels]


# ================================================================== the figure
@needs_table
@pytest.mark.parametrize('variant', ['A', 'B'])
def test_the_full_figure_draws_every_stored_row(world, tmp_path, monkeypatch, capsys, variant):
    written, fig = draw(world, tmp_path / 'fig', monkeypatch, variant=variant)
    assert sorted(os.path.basename(p) for p in written) == [f'fq4_limits_{variant}.pdf',
                                                          f'fq4_limits_{variant}.png']
    assert all(os.path.isfile(p) for p in written)
    ax, bx = fig.axes
    sizes, levels = want(world)
    check_panel(ax, True, [sizes[m] for m in SIZES])
    check_panel(bx, variant == 'A', levels)
    assert tick_texts(ax.get_xticklabels()) == [str(m) for m in SIZES]
    if variant == 'A':
        assert tick_texts(bx.get_xticklabels()) == VALUE_LABELS
        assert ax.get_shared_y_axes().joined(ax, bx)
    else:
        rows = [t.get_text() for t in bx.texts if t.get_ha() == 'right']
        assert rows == VALUE_LABELS
    out = capsys.readouterr().out
    assert 'macro gate: 183 of 183 plotted means and interval ends match their macros' in out
    assert 'missing' not in out


def test_the_band_lies_behind_the_marks_and_is_lighter_than_every_one(world, tmp_path,
                                                                      monkeypatch):
    _, fig = draw(world, tmp_path / 'fig', monkeypatch)
    for ax in fig.axes:
        (band,) = [p for p in ax.patches if to_rgba(p.get_facecolor()) == to_rgba(M.BAND_FILL)]
        y = band.get_path().transformed(band.get_patch_transform()).vertices[:, 1]
        assert (min(y), max(y)) == (-M.MARGIN, M.MARGIN) and M.MARGIN == 1.0
        assert band.get_zorder() < min(ln.get_zorder() for ln in ax.lines)

    def lum(c):
        r, g, b, _ = to_rgba(c)
        return 0.299 * r + 0.587 * g + 0.114 * b
    colours = [st['color'] for st in M.FQL_STYLE.values()]
    assert all(lum(M.BAND_FILL) > lum(c) + 0.3 for c in colours)
    # the two series read in greyscale: a filled dark circle against an open square
    assert abs(lum(colours[0]) - lum(colours[1])) > 0.15


def test_the_full_mode_refuses_and_names_every_missing_size(world, tmp_path, monkeypatch):
    os.remove(world['prod'])
    with pytest.raises(SystemExit, match=r'all six sizes; missing:\n  50 modules: .*\n  80 modules'):
        draw(world, tmp_path / 'fig', monkeypatch)
    os.remove(world['per'])
    with pytest.raises(SystemExit, match=r'27 modules: the record of p118'):
        draw(world, tmp_path / 'fig', monkeypatch)
    assert not (tmp_path / 'fig').exists()


@needs_table
@pytest.mark.parametrize('variant', ['A', 'B'])
def test_the_partial_mode_leaves_out_the_sizes_without_record(world, tmp_path, monkeypatch,
                                                              capsys, variant):
    sizes, levels = want(world)
    os.remove(world['prod'])
    written, fig = draw(world, tmp_path / 'fig', monkeypatch, variant=variant, partial=True)
    assert sorted(os.path.basename(p) for p in written) == [
        f'fq4_limits_{variant}_partial.pdf', f'fq4_limits_{variant}_partial.png']
    ax, bx = fig.axes
    check_panel(ax, True, [sizes[m] for m in (10, 20, 27, 30)])
    check_panel(bx, variant == 'A', levels)
    assert tick_texts(ax.get_xticklabels()) == ['10', '20', '27', '30']
    out = capsys.readouterr().out
    assert 'sizes missing from panel (a): 50 modules, 80 modules' in out
    # the 10-module point against the rule needs no p128: it comes from the default level of p87
    assert 'levels.default.pools.pool.pol_rule' in out


def test_the_partial_mode_without_p88_and_p118_draws_the_production_sizes(world, tmp_path,
                                                                         monkeypatch, capsys):
    for k in ('fam', 'per'):
        os.remove(world[k])
    _, fig = draw(world, tmp_path / 'fig', monkeypatch, partial=True)
    assert tick_texts(fig.axes[0].get_xticklabels()) == ['50', '80']
    assert ('sizes missing from panel (a): 10 modules, 20 modules, 27 modules, 30 modules'
            in capsys.readouterr().out)


@pytest.mark.parametrize('partial', [False, True])
def test_the_record_of_p87_is_needed_in_every_mode(world, tmp_path, monkeypatch, partial):
    os.remove(world['sens'])
    with pytest.raises(SystemExit, match='the record of p87 infer .* not found'):
        draw(world, tmp_path / 'fig', monkeypatch, partial=partial)


# ================================================================== the records it accepts
def _set(path, keys, value):
    rec = load(path)
    x = rec
    for k in keys[:-1]:
        x = x[k]
    x[keys[-1]] = value
    dump(path, rec)


@pytest.mark.parametrize('which,keys,value,message', [
    ('per', ['seeds'], [301, 302, 303], 'holds the training seeds \\[301, 302, 303\\]'),
    ('prod', ['seeds'], [301, 302, 303, 304, 306], 'holds the training seeds'),
    ('fam', ['seeds'], [301, 302, 303, 304, 305, 306, 307, 308, 309, 310],
     'holds the training seeds'),
    ('sens', ['seeds'], [301, 302, 303], 'holds the training seeds'),
    ('fam', ['setting'], 'environment legacy, training seeds 301,302,303,304,305',
     'not of the exact environment'),
    ('per', ['setting'], 'environment legacy, training seeds 301,302,303,304,305',
     'not of the exact environment'),
    ('prod', ['setting'], 'environment legacy, training seeds 301,302,303,304,305',
     'not of the exact environment'),
    ('sens', ['mask'], 'legacy', 'not of the exact environment'),
    ('fam', ['generator'], 'scripts/p95_secondary_inference.py', 'was written by'),
    ('per', ['generator'], 'scripts/p88_family_inference.py', 'was written by'),
    ('prod', ['generator'], 'scripts/p118_per_floor.py', 'was written by'),
    ('sens', ['generator'], 'scripts/p87_sensitivity.py', 'was written by'),
])
def test_records_of_another_run_are_refused(world, tmp_path, monkeypatch, which, keys, value,
                                            message):
    _set(world[which], keys, value)
    for partial in (False, True):
        with pytest.raises(SystemExit, match=message):
            draw(world, tmp_path / 'fig', monkeypatch, partial=partial, allow_partial=True)


@pytest.mark.parametrize('which,keys', [('prod', ['dry']), ('per', ['dry']),
                                        ('sens', ['partial'])])
def test_a_dry_or_partial_run_is_drawn_only_with_allow_partial(world, tmp_path, monkeypatch,
                                                               which, keys):
    _set(world[which], keys, True)
    with pytest.raises(SystemExit, match='drawn only with --allow-partial'):
        draw(world, tmp_path / 'fig', monkeypatch)
    written, _ = draw(world, tmp_path / 'fig', monkeypatch, allow_partial=True)
    assert sorted(os.path.basename(p) for p in written) == ['fq4_limits_A_dev.pdf',
                                                          'fq4_limits_A_dev.png']


def test_the_10_module_rows_must_be_one_evaluation(world, tmp_path, monkeypatch):
    S = load(world['sens'])
    row = S['levels']['default']['pools']['pool']['pol_spec']
    row['mean'] = (row['mean'] + row['lo']) / 2                # still inside its interval
    dump(world['sens'], S)
    with pytest.raises(SystemExit, match='do not hold one evaluation of the 10-module instances'):
        draw(world, tmp_path / 'fig', monkeypatch)


def test_the_10_module_row_against_the_rule_of_p128_must_equal_that_of_p87(world, tmp_path,
                                                                           monkeypatch):
    _set(world['prod'], ['side_by_side', 'rows', '10x25', 'rule', 'lo'],
         load(world['prod'])['side_by_side']['rows']['10x25']['rule']['lo'] - 1e-6)
    with pytest.raises(SystemExit, match='10 modules against the best dispatching rule: .* so the '
                                         'records are not of one run'):
        draw(world, tmp_path / 'fig', monkeypatch)


@pytest.mark.parametrize('key,field', [('20x25', 'rule_pool_pooled'), ('30x25', 'parity_pooled')])
def test_the_rows_of_p118_at_20_and_30_modules_must_equal_those_of_p88(world, tmp_path,
                                                                       monkeypatch, key, field):
    P = load(world['per'])
    row = P['sizes'][key][field] if field == 'rule_pool_pooled' else \
        P['sizes'][key][field]['All']
    row['hi'] += 1e-6
    dump(world['per'], P)
    with pytest.raises(SystemExit, match=f'{key[:2]} modules: .* not of one run'):
        draw(world, tmp_path / 'fig', monkeypatch)


@pytest.mark.parametrize('level,field,value,message', [
    ('lag_x050', 'value', 0.6, 'sets .* the figure expects'),
    ('setup_sub_strict', 'parameter', 'setup scale', 'sets .* the figure expects'),
    ('win_x050', 'reached', ['W', 'LW', 'SW'], 'changes .* the figure expects'),
])
def test_a_level_of_another_parameter_or_reach_is_refused(world, tmp_path, monkeypatch, level,
                                                          field, value, message):
    _set(world['sens'], ['levels', level, field], value)
    with pytest.raises(SystemExit, match=message):
        draw(world, tmp_path / 'fig', monkeypatch)


def test_a_row_with_its_mean_outside_its_interval_or_another_count_is_refused(world, tmp_path,
                                                                             monkeypatch):
    _set(world['prod'], ['sizes', '80x25n30', 'rule_pool_pooled', 'c'], 4)
    with pytest.raises(SystemExit, match='pools 4 combinations, not 8'):
        draw(world, tmp_path / 'fig', monkeypatch)
    _set(world['prod'], ['sizes', '80x25n30', 'rule_pool_pooled', 'c'], 8)
    _set(world['prod'], ['sizes', '50x25n30', 'parity_pooled', 'All', 'mean'], 9.0)
    with pytest.raises(SystemExit, match='the mean \\+9.0000 lies outside'):
        draw(world, tmp_path / 'fig', monkeypatch)


# ================================================================== the gate
def test_the_gate_holds_every_plotted_value_against_its_macro(world):
    X = M.fql_inputs(*[args(world)[k] for k in ('family_json', 'family_macros', 'perfloor_json',
                                                 'perfloor_macros', 'prod_json', 'prod_macros',
                                                 'sens_json', 'sens_macros')])
    names = {r[5] for r in X['table']}
    # three numbers per mark: 6 sizes and 22 levels, two series, and the macros printed twice
    # (20 and 30 modules in p88 and p118, 10 modules against the rule in p87 and p128)
    assert len(X['table']) == 3 * (2 * 6 + 2 * 22) + 3 * 4 + 3
    for stem in ('FamParPoolAll', 'SensDefaultPolRulePool', 'PbsTenRulePool', 'PflParPoolAll',
                 'PflRulePool', 'FamShopTwentyParPoolAll', 'PflThirtyRulePool',
                 'PbsFiftyParPoolAll', 'PbsEightyRulePool', 'SensLagHalfPolSpecReached',
                 'SensMixSteelPolRulePool'):
        assert {stem, stem + 'Lo', stem + 'Hi'} <= names
    assert all(re.fullmatch(r'[+-]\d+\.\d\d', r[6]) for r in X['table'])


@pytest.mark.parametrize('which,name', [
    ('fam_m', 'FamParPoolAllHi'), ('fam_m', 'FamShopTwentyRulePoolLo'),
    ('per_m', 'PflParPoolAll'), ('per_m', 'PflThirtyRulePoolHi'),
    ('prod_m', 'PbsEightyRulePoolHi'), ('prod_m', 'PbsTenRulePool'),
    ('sens_m', 'SensDefaultPolRulePool'), ('sens_m', 'SensLagHalfPolSpecReached'),
    ('sens_m', 'SensMixSteelPolRulePoolLo'),
])
def test_the_gate_refuses_a_macro_that_disagrees_in_one_number(world, tmp_path, monkeypatch,
                                                              which, name):
    v = float(M._macros(world[which])[name])
    edit_macro(world[which], name, f'{v + 0.01:+.2f}')
    with pytest.raises(SystemExit, match=r'the macros disagree .*\n.*\\' + name + ' = '):
        draw(world, tmp_path / 'fig', monkeypatch)


def test_the_gate_refuses_a_missing_macro_or_macro_file(world, tmp_path, monkeypatch):
    drop_macro(world['prod_m'], 'PbsFiftyParPoolAllHi')
    with pytest.raises(SystemExit, match=r'\\PbsFiftyParPoolAllHi: no macro in'):
        draw(world, tmp_path / 'fig', monkeypatch)
    os.remove(world['sens_m'])
    with pytest.raises(SystemExit, match=r'the macro file .* not found'):
        draw(world, tmp_path / 'fig', monkeypatch)


def test_the_manuscript_copies_must_print_what_the_run_prints(world, tmp_path, monkeypatch,
                                                              capsys):
    paper = tmp_path / 'manuscript'
    paper.mkdir()
    shutil.copy(world['prod_m'], paper)
    shutil.copy(world['per_m'], paper)
    draw(world, tmp_path / 'fig', monkeypatch, paper_dir=str(paper))
    out = capsys.readouterr().out
    assert re.search(r'manuscript copy: .*macros_prodspec.tex prints the same 15 macros', out)
    assert re.search(r'manuscript copy: .*macros_perfloor.tex prints the same 18 macros', out)
    assert re.search(r'manuscript copy: .*macros_family.tex not found', out)
    edit_macro(str(paper / 'macros_prodspec.tex'), 'PbsFiftyRulePool', '+9.99')
    with pytest.raises(SystemExit, match=r'copies the manuscript inputs:\n.*\\PbsFiftyRulePool = '
                                         r'\+9.99 in'):
        draw(world, tmp_path / 'fig', monkeypatch, paper_dir=str(paper))


# ================================================================== the layout
@pytest.mark.parametrize('variant,height', [('A', 3.1), ('B', 4.6)])
def test_the_layout(world, tmp_path, monkeypatch, variant, height):
    written, fig = draw(world, tmp_path / 'fig', monkeypatch, variant=variant)
    w, h = fig.get_size_inches()
    assert abs(w - 6.30) < 1e-9 and h <= height
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    assert len(texts) > 40
    for t in texts:
        assert t.get_fontsize() >= 8, t.get_text()
        assert to_rgba(t.get_color())[:3] == (0.0, 0.0, 0.0), t.get_text()
        assert '\u00b7' not in t.get_text()
    boxes = [(t.get_window_extent(rend), t.get_text()) for t in texts]
    canvas = fig.bbox
    for i, (a, ta) in enumerate(boxes):
        assert canvas.x0 <= a.x0 and a.x1 <= canvas.x1 and canvas.y0 <= a.y0 and a.y1 <= canvas.y1
        for b, tb in boxes[i + 1:]:
            assert not (a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1), (ta, tb)
    (leg,) = fig.legends
    assert [t.get_text() for t in leg.get_texts()] == LEGEND
    assert leg.get_frame_on() and leg.get_frame().get_linewidth() == 0.5
    lb = leg.get_window_extent(rend)
    for ax in fig.axes:
        assert not lb.overlaps(ax.get_tightbbox(rend))
    a_box, b_box = (ax.get_position() for ax in fig.axes)
    assert abs(a_box.y1 - b_box.y1) < 1e-9 and abs(a_box.y0 - b_box.y0) < 1e-9   # one row
    assert a_box.width < b_box.width / 2
    M.assert_times_only(written)
    pdf = next(p for p in written if p.endswith('.pdf'))
    if shutil.which('pdffonts'):
        r = subprocess.run(['pdffonts', pdf], capture_output=True, text=True)
        faces = [ln.split()[0] for ln in r.stdout.splitlines()[2:] if ln.strip()]
        assert faces and all('TeXGyreTermes' in f for f in faces)


def test_the_group_labels_of_variant_a_stand_on_one_baseline(world, tmp_path, monkeypatch):
    _, fig = draw(world, tmp_path / 'fig', monkeypatch)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    words = {line for G in M.FQL_GROUPS for line in G['a'].split('\n')}
    texts = [t for t in fig.texts if t.get_text() in words]
    assert [t.get_text() for t in texts] == \
        [line for G in M.FQL_GROUPS for line in G['a'].split('\n')]
    assert all(t.get_va() == 'baseline' and t.get_ha() == 'center' for t in texts)
    size = next(t for t in fig.texts if t.get_text() == M.FQL_SIZE_LABEL)
    assert size.get_va() == 'baseline'
    base = fig.transFigure.transform(size.get_position())[1]   # the baseline of the size label
    pitch = M._fql_size(fig, rend, 'lp')[1] * fig.dpi           # one line, in pixels
    bx = fig.axes[1]
    it = iter(texts)
    pos = M._fql_positions([dict(items=G['levels']) for G in M.FQL_GROUPS], M.FQL_A_GAP)
    for G, ps in zip(M.FQL_GROUPS, pos):
        centre = bx.transData.transform(((ps[0] + ps[-1]) / 2, 0))[0]
        for k, line in enumerate(G['a'].split('\n')):
            t = next(it)
            x, y = fig.transFigure.transform(t.get_position())
            # line k of every label on the baseline of row k, one pitch below row k - 1, and
            # centred under its group
            assert abs(y - (base - k * pitch)) < 0.05, line        # pixels at 300 dpi
            assert abs(x - centre) < 0.05, line
    # the row of group labels lies below every value label
    ticks = [b.get_window_extent(rend) for b in bx.get_xticklabels() if b.get_text()]
    assert max(t.get_window_extent(rend).y1 for t in texts) < min(b.y0 for b in ticks)


def test_three_line_labels_make_the_figure_taller_and_keep_the_panels(world, tmp_path,
                                                                      monkeypatch):
    _, fig3 = draw(world, tmp_path / 'fig3', monkeypatch)
    two = tuple(dict(G, a='\n'.join(G['a'].split('\n')[:2])) for G in M.FQL_GROUPS)
    monkeypatch.setattr(M, 'FQL_GROUPS', two)
    _, fig2 = draw(world, tmp_path / 'fig2', monkeypatch)
    rend = fig2.canvas.get_renderer()
    pitch = M._fql_size(fig2, rend, 'lp')[1]
    h2, h3 = fig2.get_size_inches()[1], fig3.get_size_inches()[1]
    assert h2 == pytest.approx(M.FQL_A_H) and h3 == pytest.approx(M.FQL_A_H + pitch)
    for a2, a3 in zip(fig2.axes, fig3.axes):
        p2, p3 = a2.get_position(), a3.get_position()
        # the same panel in inches, measured from the top of the figure
        assert p3.width * M.FIG_W == pytest.approx(p2.width * M.FIG_W)
        assert p3.height * h3 == pytest.approx(p2.height * h2)
        assert (1 - p3.y1) * h3 == pytest.approx((1 - p2.y1) * h2)
        assert p3.x0 == pytest.approx(p2.x0)


def test_the_group_labels_of_variant_a_read_without_the_table():
    read = [G['a'].replace('-\n', '-').replace('\n', ' ') for G in M.FQL_GROUPS]
    assert read == ['processing-time noise', 'finishing, assembly times', 'lag scale',
                    'setup scale', 'same-class setups', 'stations with setups', 'window density',
                    'stations in the factory', 'one shell type']
    assert all(len(G['a'].split('\n')) <= 3 for G in M.FQL_GROUPS)


@pytest.mark.parametrize('name,value,message', [
    ('FQL_TEXT_CLEAR', {'x': 40.0, 'y': 40.0}, 'pt apart, less than 40.0 pt'),
    ('FQL_DX', 0.3, 'the (intervals|markers|marker) .* (touch|overlap)'),
    ('FQL_DX_SIZES', 2.0, 'the offset 2.0 pt of panel \\(a\\) lets the two markers'),
])
def test_a_layout_that_breaks_a_rule_is_refused(world, tmp_path, monkeypatch, name, value,
                                                message):
    monkeypatch.setattr(M, name, value)
    with pytest.raises(SystemExit, match=message):
        draw(world, tmp_path / 'fig', monkeypatch)
    assert not (tmp_path / 'fig').exists()


def test_an_unknown_variant_is_refused(world, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="variant 'C' is not one of"):
        draw(world, tmp_path / 'fig', monkeypatch, variant='C')


# ================================================================== the names
@needs_table
def test_the_levels_are_those_of_p87_and_of_the_table():
    words, rows = [], []
    for G in M.FQL_GROUPS:
        assert len(G['b']) <= max(2, len(G['levels'])) and len(G['a'].split('\n')) <= 3
        for key, word, label, par, val in G['levels']:
            assert P87.LEVEL_WORDS[key] == word
            p, v = P87.level_parameter(key)
            assert p == par
            assert v == val if isinstance(val, str) else v == pytest.approx(val)
            assert tuple(P87.reached(key)) == tuple(G['reach'])
            words.append(word)
            rows.append('Pool' if len(G['reach']) == len(P87.R8) else 'Reached')
    assert list(zip(words, rows)) == table_rows()
    assert len(words) == M.FQL_N_LEVELS == 22
    assert [x[2] for G in M.FQL_GROUPS for x in G['levels']] == VALUE_LABELS
    assert 'default' not in {x[0] for G in M.FQL_GROUPS for x in G['levels']}


def test_the_record_and_macro_names_are_those_of_the_writers():
    assert (M.FQL_PERFLOOR_GENERATOR, M.FQL_PERFLOOR_JSON, M.FQL_PERFLOOR_MACROS) == (
        P118.GENERATOR, P118.OUT_FILES['json'], P118.OUT_FILES['macros'])
    assert (M.FQL_PROD_GENERATOR, M.FQL_PROD_JSON, M.FQL_PROD_MACROS) == (
        P128.GENERATOR, P128.OUT_FILES['json'], P128.OUT_FILES['macros'])
    assert P118.PREFIX == 'Pfl' and P128.PREFIX == 'Pbs' and P118.SIZE == '27x25'
    assert P128.SIZES == ('50x25n30', '80x25n30')
    assert {P128.MODULES[s] for s in P128.SIZES + P128.SIDE_SIZES} == set(SIZES) == set(M.FQL_SIZES)
    assert M.FQL_SENS_GENERATOR == P87.INFER_GENERATOR
    assert M.fql_sens_json(M.FQL_SENS_CSV) == P87.INFER_FILES['json']
    assert M.FQL_SENS_CSV == P87.INFER_FILES['csv']
    assert M.FQL_SENS_MACROS == P87.INFER_FILES['macros']
    assert M.FQL_FAMILY_GENERATOR == 'scripts/p88_family_inference.py'
    assert tuple(M.FQL_SEEDS) == tuple(SEEDS) and M.FQL_ENV == 'exact'


# ================================================================== selection in main
NAMES = ('fq1_specialists', 'fq3_time_budget', 'fq4_boundary', 'fs_convergence', 'fq2_reuse',
         'fq2_heldout', 'fq5_rule_change', 'fq4_limits')


def _record_calls(monkeypatch):
    calls = []
    for name in NAMES:
        monkeypatch.setattr(M, name, lambda *a, _n=name: calls.append((_n, a)) or [])
    return calls


def test_the_run_without_only_does_not_draw_fq4_limits(monkeypatch):
    for v in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--factory-seeds', 'five'])
    monkeypatch.chdir(ROOT)
    M.main()
    called = [n for n, _ in calls]
    assert called[0] == 'fq1_specialists' and 'fq4_limits' not in called


@pytest.mark.parametrize('extra,variant,partial', [([], 'A', False),
                                                   (['--fq4-limits-variant', 'B',
                                                     '--fq4-limits-partial'], 'B', True)])
def test_only_fq4_limits_draws_it_alone_from_the_registry_paths(monkeypatch, extra, variant,
                                                                partial):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    monkeypatch.setenv('PACT_OUT', 'out/exact')
    monkeypatch.delenv('PACT_RESULTS', raising=False)
    calls = _record_calls(monkeypatch)
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--only', 'fq4_limits'] + extra)
    monkeypatch.chdir(ROOT)
    M.main()
    ((name, a),) = calls
    assert name == 'fq4_limits'
    (fj, fm, pj, pm, prj, prm, sj, sm, out, var, part, allow) = a
    ex = os.path.join(ROOT, 'out', 'exact')
    assert fj == os.path.join(ex, 'results', 'family_inference.json')
    assert fm == os.path.join(ex, 'paper', 'macros_family.tex')
    assert pj == os.path.join(ex, 'results', 'per_floor.json')
    assert pm == os.path.join(ex, 'manuscript', 'macros_perfloor.tex')
    assert prj == os.path.join(ex, 'results', 'production_specialists.json')
    assert prm == os.path.join(ex, 'manuscript', 'macros_prodspec.tex')
    assert sj == os.path.join(ROOT, 'results', 'sensitivity_inference.json')
    assert sm == os.path.join(ROOT, 'manuscript', 'macros_sensitivity.tex')
    assert out == os.path.join(ROOT, M.OUT_DEFAULT)
    assert (var, part, allow) == (variant, partial, False)
    assert M.FQL_PAPER_DIR == os.path.join(ROOT, 'manuscript')
