"""Inputs of the first result figure, fq1_specialists of scripts/make_figures.py: the
policy against the specialists on the factory benchmark and on the public instances.

  * with ten training seeds, the policy rows of panel (a) are the ten-seed record's ten[R]
    and pool_ten.All, the control rows of variant B are the five-seed record's, and the
    figure draws exactly these intervals and means (variants A and B, written under the
    temporary folder), its row labels the combination names in math italic ($LS$) above the
    upright pooled row;
  * the gate refuses a macro file of the ten-seed record, written by p110's own
    macro_lines, that disagrees in one number, that lacks a macro of a policy row, or that
    does not exist;
  * a ten-seed record that does not extend the five-seed record is refused, and the record
    of a dry run is drawn only with allow_partial, under file names ending in _dev;
  * --factory-seeds auto takes five when the ten-seed record is absent and ten when it is
    present; ten refuses without the record;
  * the public panel reads the record of the family under the active mask by default, its
    gate reads the macros named PubFamActSdten..., and a record of the non-delay family is
    refused unless that family is asked for.

Synthetic records only, written under the pytest temporary folder; nothing is trained or
evaluated, and no display is needed.
Run from the repository root:  python -m pytest tests/test_fq1_inputs.py -q
"""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import make_figures as M  # noqa: E402
import p110_seed_extension as P110  # noqa: E402
import stats_family as SF  # noqa: E402

COMBOS = list(M.COMBOS)
FAMS = {'All': COMBOS, 'Train': COMBOS[:6], 'Held': COMBOS[6:]}
SET = 'pub_SD2_test_10x5+mix'
FIVE_SEEDS = list(range(301, 306))
TEN_SEEDS = list(range(301, 311))

# Synthetic differences in % (mean, half width of the 95% interval).
FIVE = {'N': (0.24, 0.62), 'L': (0.18, 0.31), 'S': (-0.35, 0.43), 'W': (0.40, 0.91),
        'LS': (-0.07, 0.31), 'LW': (0.46, 0.41), 'SW': (-0.47, 0.41), 'LSW': (-0.15, 0.37)}
TEN = {'N': (0.11, 0.42), 'L': (0.21, 0.22), 'S': (-0.27, 0.30), 'W': (0.33, 0.58),
       'LS': (-0.12, 0.21), 'LW': (0.38, 0.29), 'SW': (-0.41, 0.28), 'LSW': (-0.09, 0.26)}
CONTROL = {'N': (-0.08, 0.54), 'L': (0.32, 0.31), 'S': (-0.22, 0.33), 'W': (-0.22, 1.01),
           'LS': (0.13, 0.27), 'LW': (0.56, 0.35), 'SW': (-0.40, 0.47), 'LSW': (0.01, 0.43)}
PUBLIC = {'N': (1.12, 1.40), 'L': (0.58, 0.63), 'S': (0.51, 1.07), 'W': (0.30, 1.36),
          'LS': (-0.20, 0.86), 'LW': (-0.09, 0.89), 'SW': (-0.83, 1.39), 'LSW': (-0.18, 0.92)}


def comp(mean, half, n_seeds=5):
    """One comparison as p88 and p89 store it: mean, interval, verdicts, direction."""
    lo, hi = mean - half, mean + half
    return {'mean': mean, 'lo': lo, 'hi': hi, 'n': 100,
            'v0.5': SF.verdict(lo, hi, 0.5), 'v1.0': SF.verdict(lo, hi, 1.0),
            'direction': SF.direction(lo, hi),
            'seed_means': [mean + 0.01 * (k - (n_seeds - 1) / 2) for k in range(n_seeds)]}


def pooled(table, n_seeds):
    """Pooled rows over All, Train and Held: the mean of the members, a narrower interval."""
    return {f: comp(sum(table[r][0] for r in rs) / len(rs), 0.21, n_seeds)
            for f, rs in FAMS.items()}


def iut(components):
    out = {}
    for fam, rs in FAMS.items():
        for mg in (0.5, 1.0):
            for claim in ('equivalent', 'non-inferior'):
                ok = {'equivalent'} if claim == 'equivalent' else {'equivalent',
                                                                    'non-inferior only'}
                failing = [r for r in rs if components[r][f'v{mg}'] not in ok]
                out[f'{fam}|{mg}|{claim}'] = {'holds': not failing, 'components': rs,
                                              'failing': failing}
    return out


def mac(name, x):
    return f'\\newcommand{{\\{name}}}{{{P110.f2(x)}}}'


def mac_rows(stem, it):
    return [mac(stem, it['mean']), mac(stem + 'Lo', it['lo']), mac(stem + 'Hi', it['hi']),
            f'\\newcommand{{\\{stem}VOne}}{{{SF.TEX[it["v1.0"]]}}}']


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=1))
    else:
        path.write_text(obj)
    return str(path)


def make_world(tmp_path):
    """The five-seed record and its macros, the ten-seed record and the macros p110 writes
    for it, and the public record of the family under the active mask with its macros."""
    par = {'components': {r: comp(*FIVE[r]) for r in COMBOS}, 'pooled': pooled(FIVE, 5)}
    par['iut'] = iut(par['components'])
    ctl = {'components': {'Spec': {r: comp(*CONTROL[r]) for r in COMBOS}},
           'pooled': {'Spec': {'All': comp(0.01, 0.24)}}}
    five = {'setting': 'environment exact, training seeds 301,302,303,304,305 (5 runs per '
                       'arm)', 'seeds': FIVE_SEEDS, 'parity': par, 'control': ctl}
    fam_lines = []
    for r in COMBOS:
        fam_lines += mac_rows(f'FamPar{r}', par['components'][r])
        fam_lines += mac_rows(f'FamCtlSpec{r}', ctl['components']['Spec'][r])
    fam_lines += mac_rows('FamParPoolAll', par['pooled']['All'])
    fam_lines += mac_rows('FamCtlSpecPoolAll', ctl['pooled']['Spec']['All'])

    ten_c = {r: comp(*TEN[r], n_seeds=10) for r in COMBOS}
    pool_ten = pooled(TEN, 10)

    def inst(pool):
        return {f: {k: x[k] for k in ('mean', 'lo', 'hi', 'direction')}
                for f, x in pool.items()}
    res = dict(generator='scripts/p110_seed_extension.py',
               setting='environment exact, training seeds 301 to 310 (10 runs per arm), '
                       'principal comparison only',
               seeds=TEN_SEEDS, first_seeds=FIVE_SEEDS, second_seeds=TEN_SEEDS[5:],
               boot=2000, alpha=0.05, triggered=['W'], dry=False,
               ten=ten_c, five=par['components'],
               first_five_mean={r: FIVE[r][0] for r in COMBOS},
               second_five_mean={r: 2 * TEN[r][0] - FIVE[r][0] for r in COMBOS},
               iut_ten=iut(ten_c), iut_five=par['iut'], pool_ten=pool_ten,
               pool_five=par['pooled'], pool_inst_ten=inst(pool_ten),
               pool_inst_five=inst(par['pooled']))

    pub_par = {'components': {r: comp(*PUBLIC[r]) for r in COMBOS},
               'pooled': {'All': comp(0.15, 0.42)}}
    pub_ctl = {'control_vs_specialist': {r: comp(PUBLIC[r][0] + 0.2, 1.0) for r in COMBOS},
               'pooled_control_vs_specialist': {'Unseen': comp(0.29, 0.81)}}
    public = {'family': {'key': 'active', 'prefix': 'PubFamAct'}, 'seeds': FIVE_SEEDS,
              'sets': {SET: {'n': 100, 'parity': pub_par, 'control': pub_ctl}}}
    pub_lines = []
    for r in COMBOS:
        pub_lines += mac_rows(f'PubFamActSdtenPar{r}', pub_par['components'][r])
    for r in ('LS', 'LW', 'SW', 'LSW'):        # the control's macros exist on these only
        pub_lines += mac_rows(f'PubFamActSdtenCtlSpec{r}', pub_ctl['control_vs_specialist'][r])
    pub_lines += mac_rows('PubFamActSdtenParPoolAll', pub_par['pooled']['All'])

    d = tmp_path / 'world'
    return dict(
        five=write(d / 'out' / 'results' / 'family_inference.json', five),
        five_macros=write(d / 'out' / 'paper' / 'macros_family.tex',
                          '\n'.join(fam_lines) + '\n'),
        ten=write(d / 'out' / 'results' / 'seed_extension.json', P110.jsonable(res)),
        ten_macros=write(d / 'out' / 'manuscript' / 'macros_seedext.tex',
                         '\n'.join(P110.macro_lines(res)) + '\n'),
        public=write(d / 'results' / 'public_family_act.json', public),
        public_macros=write(d / 'manuscript' / 'macros_pubfam_act.tex',
                            '\n'.join(pub_lines) + '\n'),
        res=res, five_rec=five, public_rec=public, root=d)


def inputs(w, **kw):
    args = dict(factory_macros=w['five_macros'], public_macros=w['public_macros'],
                factory_seeds='ten', seedext_json=w['ten'], seedext_macros=w['ten_macros'],
                public_family='active', allow_partial=False)
    args.update(kw)
    return M.fq1_inputs(w['five'], w['public'], SET, **args)


def triple(it):
    return (it['mean'], it['lo'], it['hi'], it['v1.0'] if 'v1.0' in it else it['verdict'])


@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path)


# ================================================================== ten seeds
def test_ten_seed_rows_are_the_record(world, capsys):
    X = inputs(world)
    res, five = world['res'], world['five_rec']
    assert X['mode'] == 'ten' and not X['dev']
    for r in COMBOS:
        assert triple(X['F']['rows'][r]) == triple(res['ten'][r])
        assert triple(X['F']['ctl'][r]) == triple(five['control']['components']['Spec'][r])
        assert triple(X['P']['rows'][r]) == triple(
            world['public_rec']['sets'][SET]['parity']['components'][r])
    assert triple(X['F']['pool']) == triple(res['pool_ten']['All'])
    assert triple(X['F']['ctl_pool']) == triple(five['control']['pooled']['Spec']['All'])
    assert X['F']['seeds'] == TEN_SEEDS
    # every policy row was held against the ten-seed macros and matched
    gated = {lab: st for fam, lab, _, _, st in X['table'] if fam == 'factory'}
    assert all(gated[lab] == 'match' for lab in M.POLICY_LABELS)
    assert all(gated[f'{r} control'] == 'match' for r in COMBOS)
    # one line names each record and its number of training seeds
    line = X['line']
    assert line.count('\n') == 0
    assert f'{world["ten"]} (10 training seeds)' in line
    assert f'the five-seed record {world["five"]} (5 training seeds) in every mode' in line
    assert f'{world["public"]} (5 training seeds)' in line


def _lines(ax):
    """(lo, hi) of every interval and x of every mean marker the axes draws, by role."""
    pol_iv, ctl_iv, pol_mean, ctl_mean = [], [], [], []
    for ln in ax.lines:
        x, y = list(ln.get_xdata()), list(ln.get_ydata())
        if len(x) == 2 and y[0] == y[1]:
            colour = str(ln.get_color()).lower()
            if colour in (M.C_EQUIV.lower(), M.C_OPEN.lower()):
                pol_iv.append((float(x[0]), float(x[1])))
            elif colour == M.C_CONTROL.lower():
                ctl_iv.append((float(x[0]), float(x[1])))
        elif len(x) == 1:
            (pol_mean if ln.get_marker() in ('o', 's') else ctl_mean).append(float(x[0]))
    return sorted(pol_iv), sorted(ctl_iv), sorted(pol_mean), sorted(ctl_mean)


def test_the_figure_draws_the_ten_seed_record(world, tmp_path, monkeypatch):
    kept = []
    monkeypatch.setattr(M.plt, 'close', kept.append)
    out = tmp_path / 'fig'
    written = M.fq1_specialists(world['five'], world['public'], SET, str(out),
                                world['five_macros'], world['public_macros'], 'ten',
                                world['ten'], world['ten_macros'], 'active')
    assert sorted(os.path.basename(p) for p in written) == [
        'fq1_specialists_A.pdf', 'fq1_specialists_A.png', 'fq1_specialists_B.pdf',
        'fq1_specialists_B.png']
    assert all(os.path.isfile(p) and p.startswith(str(out)) for p in written)
    res, five = world['res'], world['five_rec']
    ten_rows = [res['ten'][r] for r in COMBOS] + [res['pool_ten']['All']]
    ctl_rows = [five['control']['components']['Spec'][r] for r in COMBOS]
    ctl_rows.append(five['control']['pooled']['Spec']['All'])
    want_pol = (sorted((c['lo'], c['hi']) for c in ten_rows),
                sorted(c['mean'] for c in ten_rows))
    want_ctl = (sorted((c['lo'], c['hi']) for c in ctl_rows),
                sorted(c['mean'] for c in ctl_rows))
    assert len(kept) == 2
    for fig, with_ctl in zip(kept, (False, True)):
        pol_iv, ctl_iv, pol_mean, ctl_mean = _lines(fig.axes[0])
        assert (pol_iv, pol_mean) == want_pol
        assert (ctl_iv, ctl_mean) == (want_ctl if with_ctl else ([], []))
        pub_iv = _lines(fig.axes[1])[0]
        pub = world['public_rec']['sets'][SET]['parity']
        assert pub_iv == sorted((c['lo'], c['hi']) for c in
                                list(pub['components'].values()) + [pub['pooled']['All']])


def test_the_row_labels_set_the_combination_names_in_math_italic(world, tmp_path, monkeypatch):
    kept = []
    monkeypatch.setattr(M.plt, 'close', kept.append)
    M.fq1_specialists(world['five'], world['public'], SET, str(tmp_path / 'fig'),
                      world['five_macros'], world['public_macros'], 'ten', world['ten'],
                      world['ten_macros'], 'active')
    assert len(kept) == 2
    for fig in kept:
        ax, bx = fig.axes
        assert [t.get_text() for t in ax.get_yticklabels()] == [
            '$N$', '$L$', '$S$', '$W$', '$LS$', '$LW$', '$SW$', '$LSW$', 'All eight']
        assert not any(t.get_visible() for t in bx.get_yticklabels())
        # the words of the figure stay upright: no other text holds mathtext
        words = [t.get_text() for t in fig.texts + bx.texts + [ax.title, bx.title]]
        assert 'held out' in words and not any('$' in w for w in words)


def _edit_macro(path, name, new):
    text = open(path).read()
    pat = re.compile(r'(\\newcommand\{\\' + name + r'\})\{[^}]*\}')
    assert len(pat.findall(text)) == 1
    open(path, 'w').write(pat.sub(lambda m: m.group(1) + '{' + new + '}', text))


def test_the_gate_refuses_a_macro_file_that_disagrees_in_one_number(world):
    inputs(world)                                       # the file as written agrees
    hi = world['res']['ten']['LW']['hi']
    _edit_macro(world['ten_macros'], 'SeedExtLWHi', P110.f2(hi + 0.01))
    with pytest.raises(SystemExit, match=r'disagrees.*\n.*SeedExtLWHi'):
        inputs(world)


def test_the_gate_refuses_a_macro_file_without_a_policy_row(world):
    text = open(world['ten_macros']).read()
    text = re.sub(r'\\newcommand\{\\SeedExtPoolAllVOne\}\{[^}]*\}', '', text)
    open(world['ten_macros'], 'w').write(text)
    with pytest.raises(SystemExit, match=r'no macro \\SeedExtPoolAllVOne'):
        inputs(world)


def test_ten_seeds_refuse_without_the_macro_file(world, tmp_path):
    with pytest.raises(SystemExit, match='macro file of the ten-seed record'):
        inputs(world, seedext_macros=str(tmp_path / 'absent.tex'))


def test_the_ten_seed_record_must_extend_the_five_seed_record(world):
    rec = json.load(open(world['ten']))
    rec['five']['N']['mean'] += 0.01
    write_path = world['ten']
    open(write_path, 'w').write(json.dumps(rec))
    with pytest.raises(SystemExit, match='does not extend'):
        inputs(world)
    rec['five']['N']['mean'] -= 0.01
    rec['first_seeds'] = [301, 302, 303, 304, 306]
    open(write_path, 'w').write(json.dumps(rec))
    with pytest.raises(SystemExit, match='extends the seeds'):
        inputs(world)


def test_a_dry_run_is_drawn_only_with_allow_partial(world, tmp_path, monkeypatch):
    rec = json.load(open(world['ten']))
    rec['dry'] = True
    open(world['ten'], 'w').write(json.dumps(rec))
    with pytest.raises(SystemExit, match='dry run'):
        inputs(world)
    assert inputs(world, allow_partial=True)['dev']
    monkeypatch.setattr(M.plt, 'close', lambda fig: None)
    written = M.fq1_specialists(world['five'], world['public'], SET, str(tmp_path / 'fig'),
                                world['five_macros'], world['public_macros'], 'ten',
                                world['ten'], world['ten_macros'], 'active', True)
    assert sorted(os.path.basename(p) for p in written) == [
        'fq1_specialists_A_dev.pdf', 'fq1_specialists_A_dev.png',
        'fq1_specialists_B_dev.pdf', 'fq1_specialists_B_dev.png']


# ================================================================== the seed mode
def test_auto_takes_five_without_the_ten_seed_record_and_ten_with_it(world, tmp_path):
    absent = str(tmp_path / 'absent' / 'seed_extension.json')
    assert M.fq1_seed_mode('auto', absent) == 'five'
    assert M.fq1_seed_mode('auto', world['ten']) == 'ten'
    assert M.fq1_seed_mode('five', world['ten']) == 'five'
    X = inputs(world, factory_seeds='auto', seedext_json=absent, seedext_macros=None)
    assert X['mode'] == 'five'
    par = world['five_rec']['parity']
    for r in COMBOS:
        assert triple(X['F']['rows'][r]) == triple(par['components'][r])
    assert triple(X['F']['pool']) == triple(par['pooled']['All'])
    assert X['F']['seeds'] == FIVE_SEEDS
    assert '(--factory-seeds auto: five)' in X['line']
    X = inputs(world, factory_seeds='auto')
    assert X['mode'] == 'ten'
    for r in COMBOS:
        assert triple(X['F']['rows'][r]) == triple(world['res']['ten'][r])
    assert '(--factory-seeds auto: ten)' in X['line']


def test_ten_refuses_without_the_ten_seed_record(world, tmp_path):
    absent = str(tmp_path / 'absent.json')
    with pytest.raises(SystemExit, match='--factory-seeds ten'):
        M.fq1_seed_mode('ten', absent)
    with pytest.raises(SystemExit, match='--factory-seeds ten'):
        inputs(world, seedext_json=absent)


# ================================================================== the public panel
def test_the_public_loader_reads_the_active_mask_record(world):
    P = M.load_public(world['public'], SET)
    assert (P['family'], P['prefix']) == ('active', 'PubFamAct')
    assert M.FQ1_PUBLIC['active'] == ('results/public_family_act.json',
                                      'manuscript/macros_pubfam_act.tex', 'PubFamAct')
    X = inputs(world)
    gated = {lab: st for fam, lab, _, _, st in X['table'] if fam == 'public'}
    assert all(gated[lab] == 'match' for lab in M.POLICY_LABELS)
    _edit_macro(world['public_macros'], 'PubFamActSdtenParSW',
                P110.f2(world['public_rec']['sets'][SET]['parity']['components']['SW']['mean']
                        + 0.02))
    with pytest.raises(SystemExit, match='PubFamActSdtenParSW'):
        inputs(world)


def test_a_record_of_the_other_family_is_refused(world):
    rec = json.load(open(world['public']))
    del rec['family']                                   # the form of the non-delay record
    open(world['public'], 'w').write(json.dumps(rec))
    with pytest.raises(SystemExit, match='nondelay family'):
        inputs(world)
    text = open(world['public_macros']).read().replace('PubFamActSdten', 'PubSdten')
    open(world['public_macros'], 'w').write(text)
    X = inputs(world, public_family='nondelay')
    assert X['P']['prefix'] == 'Pub' and 'nondelay family' in X['line']


def test_main_reads_the_active_mask_record_by_default(monkeypatch):
    for v in ('PACT_ENV', 'PACT_SEEDS', 'PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    got = []
    monkeypatch.setattr(M, 'fq1_specialists', lambda *a: got.append(a) or [])
    monkeypatch.setattr(sys, 'argv', ['make_figures.py', '--only', 'fq1',
                                      '--factory-seeds', 'five'])
    monkeypatch.chdir(ROOT)
    M.main()
    (factory, public, public_set, out, fmac, pmac, mode, sx, sxm, family, partial), = got
    assert public == os.path.join(ROOT, 'results', 'public_family_act.json')
    assert pmac == os.path.join(ROOT, 'manuscript', 'macros_pubfam_act.tex')
    assert (public_set, mode, family, partial) == (SET, 'five', 'active', False)
    assert sx == os.path.join(ROOT, 'results', 'seed_extension.json') and sxm is None
