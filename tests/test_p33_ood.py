"""scripts/p33_ood_table.py: the policy against the best dispatching rule outside the training
ranges, by the procedure of FamRule<R> (scripts/p88_family_inference.py).

  * --dry builds the three sets with known differences, a FamRule record from the same arrays,
    and passes: the differences are recovered, the reached combinations carry the verdicts the
    synthetic differences imply, the combinations a shift does not reach carry the rows inside
    the ranges, and the record and the Markdown have one table per shifted set;
  * the families: the eight combinations inside the ranges, S, LS, SW, LSW for the setups and
    W, LW, SW, LSW for the windows; an interval does not depend on its family, a verdict does
    (the same differences are ahead with Holm over four and not distinguishable over eight);
  * the rows inside the ranges are checked against the FamRule record: a difference in a number
    of the record, in a printed macro or in its setting is refused, and so is a missing record;
  * an array of the policy or of a rule that differs on a combination a shift does not reach is
    refused, and no output is left behind;
  * the macro names (letters only, prefix Ood, the names of the brief, the count) and comments
    (families, the three sets of Infeas, the zero of Floor, the ranges of the generator);
  * the pooled row of each shifted set: the crossed hierarchical bootstrap of stats_family over
    the differences of the combinations it reaches, a drawn instance bringing all of them, with
    the seed and B of the pooled rows of p102 and p103; the synthetic means are recovered and
    printed as Ood<Set>Pool, Lo, Hi and Dir;
  * the policy against the specialists on the denser windows, pooled by p88.pooled_block with
    the verdicts at the margins; a specialist's array that differs where the shift does not reach
    and a missing specialist's array are refused;
  * --check exits 0 on the complete synthetic tree and 1 without a summary or the record;
  * on the stored arrays and the stored record of step family (when present), the procedure
    gives FamRule exactly.

Synthetic arrays under the pytest temporary folder; the descriptors of data/FAMILY and the stored
arrays and records are read only; nothing in the repository is written. Run from the repository
root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q tests/test_p33_ood.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

import arm_registry as REG  # noqa: E402
import p33_ood_table as P  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import stats_family as SF  # noqa: E402

PY = sys.executable
SCRIPT = os.path.join(ROOT, 'scripts', 'p33_ood_table.py')
R8 = list(REG.COMBOS)
SETUP4, WINDOW4 = ['S', 'LS', 'SW', 'LSW'], ['W', 'LW', 'SW', 'LSW']

pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(ROOT, d, 'desc', 'N')) for _, _, d in P.SETS),
    reason='needs the descriptors of data/FAMILY/10x25, 10x25_oodsetup and 10x25_oodwindow')


def env(**kv):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    e.update({k: str(v) for k, v in kv.items()})
    return e


@pytest.fixture(autouse=True)
def keep_p88_setting(monkeypatch):
    """p88's B and ALPHA are module settings that P.run changes; restore them after each test."""
    monkeypatch.setattr(F88, 'B', F88.B)
    monkeypatch.setattr(F88, 'ALPHA', F88.ALPHA)


@pytest.fixture(scope='module')
def dry_tree(tmp_path_factory):
    tree = tmp_path_factory.mktemp('p33') / 'tree'
    r = subprocess.run([PY, '-B', SCRIPT, '--dry', str(tree)], capture_output=True, text=True,
                       cwd=ROOT, env=env())
    return tree, r


@pytest.fixture
def copy_tree(dry_tree, tmp_path, monkeypatch):
    tree = tmp_path / 'tree'
    shutil.copytree(dry_tree[0], tree)
    for k, v in P.dry_paths(str(tree)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(ROOT)
    return tree


def record(tree):
    return json.load(open(tree / 'out' / P.OUT_FILES['json']))


def no_outputs():
    return not any(os.path.exists(x) for x in P.outputs().values())


# ----------------------------------------------------------------- the dry run
def test_dry_run_passes(dry_tree):
    tree, r = dry_tree
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'p33 dry run on synthetic arrays passed' in r.stdout
    res = record(tree)
    assert res['dry'] and res['boot'] == P.DRY_BOOT and res['environment'] == 'exact'
    assert res['control']['arrays'] == 2 * 4 * (5 + 4) and res['control']['floor'] == 0.0
    assert res['family_check']['identical'] and res['family_check']['compared'] == 8 * 12
    for name in ('ind', 'setup', 'window'):
        for rc in R8:
            x = res['rows'][f'{name}|{rc}']
            reached = rc in P.REACHED[name]
            want = P.DRY_LEAD[name][rc] if reached else P.DRY_LEAD['ind'][rc]
            assert abs(x['mean'] - want) < 0.1 and x['reached'] == reached, (name, rc)
            if not reached:                      # the row inside the ranges, unchanged
                assert {k: v for k, v in x.items() if k != 'reached'} == \
                    {k: v for k, v in res['rows'][f'ind|{rc}'].items() if k != 'reached'}
    assert res['counts']['setup'] == {'ahead': ['S', 'LS', 'LSW'], 'behind': [],
                                      'not distinguishable': ['SW']}
    assert res['counts']['window'] == {'ahead': WINDOW4, 'behind': [], 'not distinguishable': []}
    assert res['infeasible_total'] == 0
    md = (tree / 'out' / P.OUT_FILES['md']).read_text()
    assert md.count('| comb. | inside the ranges | verdict | outside the ranges | verdict |') == 2
    assert '## Larger asymmetric setups (10x25_oodsetup)' in md
    assert '## Denser machine unavailability windows (10x25_oodwindow)' in md


# ----------------------------------------------------------------- families
def test_the_families_are_those_of_the_protocol(dry_tree):
    res = record(dry_tree[0])
    assert res['sets']['ind']['holm_family'] == R8              # the family of FamRule
    assert res['sets']['setup']['holm_family'] == SETUP4
    assert res['sets']['window']['holm_family'] == WINDOW4
    for name, fam in (('ind', R8), ('setup', SETUP4), ('window', WINDOW4)):
        for rc in fam:
            assert res['rows'][f'{name}|{rc}']['holm_family'] == fam
    # a combination a shift does not reach keeps the row (and the family) inside the ranges
    assert res['rows']['setup|W']['holm_family'] == R8
    assert res['rows']['window|S']['holm_family'] == R8


def seed_level_differences(k=5, n=100):
    """d [k, n] whose per-seed means are fixed (all below zero), with instance noise of zero
    mean per seed: the hierarchical interval lies below zero, the bootstrap p is 1 / (B + 1),
    and the one-sided seed-level t p lies between 0.05/8 and 0.05/4."""
    means = np.array([-0.3, -0.55, -1.0, -1.45, -1.7])[:k]
    noise = np.random.default_rng(5).normal(0, 0.3, (k, n))
    return means[:, None] + noise - noise.mean(axis=1, keepdims=True)


def test_a_verdict_changes_with_the_family():
    F88.configure(boot=2000, alpha=0.05)
    d = seed_level_differences()
    p = SF.seed_t_p(d, 'less')
    assert 0.05 / 8 <= p < 0.05 / 4
    D = {r: d.copy() for r in R8}
    four, eight = P.family_rows(D, SETUP4), P.family_rows(D, R8)
    for k in ('mean', 'lo', 'hi', 'p_t', 'p_boot'):          # the interval and the raw tests
        assert four['S'][k] == eight['S'][k], k
    assert four['S']['hi'] < 0
    assert four['S']['p_t_holm'] == pytest.approx(4 * p) and \
        eight['S']['p_t_holm'] == pytest.approx(8 * p)
    assert four['S']['verdict'] == 'ahead'
    assert eight['S']['verdict'] == 'not distinguishable'
    assert four['S']['holm_family'] == SETUP4 and eight['S']['holm_family'] == R8


def test_the_procedure_is_the_one_of_famrule():
    """family_rows is p88.superiority_block over the family, with the family recorded."""
    F88.configure(boot=500, alpha=0.05)
    g = np.random.default_rng(3)
    D = {r: g.normal(-0.5, 1.0, (5, 100)) for r in R8}
    want = F88.superiority_block(D)
    got = P.family_rows(D, R8)
    for r in R8:
        assert {k: v for k, v in got[r].items() if k not in ('holm_family', 'reached')} == want[r]


# ----------------------------------------------------------------- the check against FamRule
def test_the_rows_inside_the_ranges_equal_the_famrule_record(copy_tree):
    res, names, _ = P.run(dry=True, boot=P.DRY_BOOT)
    pool = json.load(open(P.family_paths()['json']))['rule_pool']
    for r in R8:
        x = res['rows'][f'ind|{r}']
        for k in P.FAMILY_KEYS:
            assert x[k] == pool[r][k]
    assert res['family_check'] == dict(json=P.family_paths()['json'],
                                       macros=P.family_paths()['macros'], compared=96,
                                       identical=True, keys=list(P.FAMILY_KEYS),
                                       macro_suffixes=['', 'Lo', 'Hi', 'Verdict'])


@pytest.mark.parametrize('what', ['number', 'verdict', 'macro', 'seeds'])
def test_a_difference_from_the_famrule_record_is_refused(copy_tree, what):
    jp, mp = P.family_paths()['json'], P.family_paths()['macros']
    rec = json.load(open(jp))
    if what == 'number':
        rec['rule_pool']['S']['hi'] += 1e-12
    elif what == 'verdict':
        rec['rule_pool']['LW']['verdict'] = 'not distinguishable'
    elif what == 'seeds':
        rec['seeds'] = [301, 302, 303]
    json.dump(rec, open(jp, 'w'))
    if what == 'macro':
        text = open(mp).read()
        lo = re.search(r'\\newcommand\{\\FamRuleSWLo\}\{([^}]*)\}', text).group(1)
        open(mp, 'w').write(text.replace(f'{{\\FamRuleSWLo}}{{{lo}}}', '{\\FamRuleSWLo}{-9.99}'))
    match = {'number': r'S hi:', 'verdict': r'LW verdict:', 'macro': r'\\FamRuleSWLo:',
             'seeds': r'records'}[what]
    with pytest.raises(P.Refused, match=match):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert no_outputs()


def test_a_missing_famrule_record_is_refused(copy_tree):
    os.remove(P.family_paths()['json'])
    with pytest.raises(P.Refused, match='inputs do not count; first: missing .*family_inference'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert no_outputs()


# ----------------------------------------------------------------- the control
@pytest.mark.parametrize('name,combo,arm', [('window', 'N', 'policy'), ('setup', 'W', 'MOR'),
                                            ('setup', 'LW', 'policy')])
def test_an_unreached_combination_that_differs_is_refused(copy_tree, name, combo, arm):
    key = dict((n, k) for n, k, _ in P.SETS)[name]
    p = (REG.result_path('released', None, 302, key, combo) if arm == 'policy'
         else f'{REG.res_dir(key)}/Result_{arm}_{combo}.npy')
    a = np.load(p)
    a[3, 0] += 1.0
    np.save(p, a)
    with pytest.raises(P.Refused, match=f'CONTROL FAILED: {name}, combination {combo} is not '
                                        f'reached'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert no_outputs()


def test_a_reached_combination_may_differ(copy_tree):
    """The control looks only at the combinations a shift does not reach."""
    p = REG.result_path('released', None, 302, '10x25_oodwindow', 'SW')
    a = np.load(p)
    a[3, 0] += 1.0
    np.save(p, a)
    res, _, _ = P.run(dry=True, boot=P.DRY_BOOT)
    assert res['control']['floor'] == 0.0


# ----------------------------------------------------------------- macros
def macros(tree):
    text = (tree / 'out' / P.OUT_FILES['macros']).read_text()
    return text, dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))


def line_of(text, name):
    return next(x for x in text.splitlines() if f'\\newcommand{{\\{name}}}' in x)


def test_macro_names(dry_tree):
    text, mac = macros(dry_tree[0])
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names)) == 136
    assert all(re.fullmatch(r'Ood[A-Za-z]+', x) for x in names)
    want = ['OodSetupX', 'OodWindowX', 'OodInfeas', 'OodFloor', 'OodPlacebo', 'OodTreated',
            'OodSetupCrossLo', 'OodSetupCrossHi', 'OodSetupCrossTrainLo', 'OodSetupCrossTrainHi',
            'OodWindowDensityLo', 'OodWindowDensityHi', 'OodWindowDensityTrainLo',
            'OodWindowDensityTrainHi']
    for w in ('Ind', 'Setup', 'Window'):
        for r in R8:
            want += [f'Ood{w}{r}{s}' for s in ('', 'Lo', 'Hi', 'Verdict')]
    for w in ('Setup', 'Window'):
        want += [f'Ood{w}{v}{s}' for v in ('Ahead', 'Nd', 'Behind') for s in ('N', 'Combos')]
        want += [f'Ood{w}Pool{s}' for s in ('', 'Lo', 'Hi', 'Dir')]
    want += [f'OodWindowSpecPool{s}' for s in ('', 'Lo', 'Hi', 'Dir', 'VHalf', 'VOne')]
    assert sorted(want) == sorted(names)
    # the pooled rows follow the macros of the per-combination rows, the existing order kept
    assert names[-14:] == [x for x in want if 'Pool' in x]
    for gone in ('OodHurt', 'OodHelped', 'OodTied', 'OodReversed'):
        assert gone not in names
    assert mac['OodSetupAheadN'] == '3' and mac['OodSetupNdN'] == '1' and mac['OodSetupBehindN'] == '0'
    assert mac['OodSetupAheadCombos'] == 'S, LS, LSW' and mac['OodSetupNdCombos'] == 'SW'
    assert mac['OodSetupBehindCombos'] == 'none' and mac['OodWindowAheadCombos'] == 'W, LW, SW, LSW'
    assert mac['OodSetupSWVerdict'] == 'not distinguishable' and mac['OodWindowWVerdict'] == 'ahead'
    assert (mac['OodSetupCrossLo'], mac['OodSetupCrossHi'], mac['OodSetupCrossTrainLo'],
            mac['OodSetupCrossTrainHi']) == ('100', '200', '50', '100')
    assert (mac['OodWindowDensityLo'], mac['OodWindowDensityHi'], mac['OodWindowDensityTrainLo'],
            mac['OodWindowDensityTrainHi']) == ('20', '35', '5', '15')
    assert mac['OodFloor'] == '0.00' and mac['OodInfeas'] == '0'
    assert mac['OodPlacebo'] == mac['OodTreated'] == '8'
    # an unreached combination prints the values and the verdict inside the ranges
    for s in ('', 'Lo', 'Hi', 'Verdict'):
        assert mac[f'OodSetupN{s}'] == mac[f'OodIndN{s}'] and mac[f'OodWindowLS{s}'] == mac[f'OodIndLS{s}']


def test_macro_comments(dry_tree):
    text, _ = macros(dry_tree[0])
    first = text.splitlines()[0]
    assert 'scripts/p33_ood_table.py' in first
    assert 'environment exact, training seeds 301,302,303,304,305' in first
    assert 'summed over the three sets' in line_of(text, 'OodInfeas')
    assert 'zero because the arrays' in line_of(text, 'OodFloor')
    assert 'Holm over N, L, S, W, LS, LW, SW, LSW' in line_of(text, 'OodIndS')
    assert 'Holm over S, LS, SW, LSW' in line_of(text, 'OodSetupSW')
    assert 'Holm over W, LW, SW, LSW' in line_of(text, 'OodWindowSW')
    assert 'not reached by the shift' in line_of(text, 'OodSetupN') and \
        '\\OodIndN' in line_of(text, 'OodSetupN')
    assert 'dataset_meta.json' in line_of(text, 'OodSetupCrossLo') and \
        'draw_setup_matrix' in line_of(text, 'OodSetupCrossLo')
    assert 'draw_windows' in line_of(text, 'OodWindowDensityLo')
    assert 'of the 4 combinations the shift reaches (S, LS, SW, LSW)' in \
        line_of(text, 'OodSetupAheadN')
    assert not re.search(r'\b(regimes?|arms?|blind|placebo)\b|released policy|noise floor',
                         text.lower())


def test_the_drawn_ranges_are_those_of_the_generator(monkeypatch):
    import inspect
    from constraint_family.descriptor import draw_setup_matrix, draw_windows
    monkeypatch.chdir(ROOT)
    rg = P.drawn_ranges()
    meta = {n: json.load(open(os.path.join(ROOT, d, 'dataset_meta.json'))) for n, _, d in P.SETS}
    assert rg['setup']['drawn'] == meta['setup']['ood_shift']['cross'] == [1.0, 2.0]
    assert rg['window']['drawn'] == meta['window']['ood_shift']['density_range'] == [0.2, 0.35]
    assert tuple(rg['setup']['trained']) == \
        inspect.signature(draw_setup_matrix).parameters['cross'].default
    assert tuple(rg['window']['trained']) == \
        inspect.signature(draw_windows).parameters['density_range'].default
    assert 'ood_shift' not in meta['ind']


# ----------------------------------------------------------------- the pooled rows
def test_the_pooled_row_is_the_routine_of_the_complete_rows(monkeypatch):
    """pooled_rows pools the differences of the combinations a set reaches by the crossed
    hierarchical bootstrap of stats_family (a drawn instance brings all its combinations), with
    the seed and B of the pooled rows of p102 and p103, on the differences of the verdicts."""
    F88.configure(boot=300, alpha=0.05)
    g = np.random.default_rng(8)
    k, n = 5, 30
    D = {r: g.normal(-0.4 * (i + 1), 1.0, (k, n)) for i, r in enumerate(R8)}
    seen = []

    def fake(key, combos, seeds):
        seen.append((key, list(combos), list(seeds)))
        return {r: D[r] for r in combos}

    monkeypatch.setattr(P, 'differences', fake)
    got = P.pooled_rows([301, 302, 303, 304, 305])
    assert sorted(got) == ['setup', 'window']
    assert seen == [('10x25_oodsetup', SETUP4, [301, 302, 303, 304, 305]),
                    ('10x25_oodwindow', WINDOW4, [301, 302, 303, 304, 305])]
    for name, fam in (('setup', SETUP4), ('window', WINDOW4)):
        x3 = np.stack([D[r] for r in fam], axis=2)
        hb = SF.hier_boot(x3, B=300, seed=11, alpha=0.05)
        x = got[name]
        assert (x['mean'], x['lo'], x['hi']) == (hb['mean'], hb['lo'], hb['hi'])
        assert x['mean'] == pytest.approx(np.mean([D[r].mean() for r in fam]))
        assert x['direction'] == SF.direction(hb['lo'], hb['hi'])
        assert x['combos'] == fam and x['c'] == 4
        # one draw of an instance brings its four combinations: the per-instance mean is pooled
        assert (x['lo'], x['hi']) == SF.percentile_ci(
            SF.hier_draws_stream(x3.mean(axis=2), 300, np.random.default_rng(11)))
    p102 = open(os.path.join(ROOT, 'scripts', 'p102_scale_sensitivity.py')).read()
    assert 'HIER_SEED, INST_SEED, ALPHA = 11, 7, 0.05' in p102 and F88.FRESH_SEED == 11


def test_the_pooled_rows_of_the_dry_run(dry_tree):
    res = record(dry_tree[0])
    for name in ('setup', 'window'):
        x = res['pooled'][name]
        want = np.mean([P.DRY_LEAD[name][r] for r in P.REACHED[name]])
        assert abs(x['mean'] - want) < 0.05 and x['hi'] < 0 and x['direction'] == 'ahead'
        assert x['combos'] == P.REACHED[name] and x['c'] == 4
        # the mean of the pooled row is the mean of the per-combination means
        assert x['mean'] == pytest.approx(np.mean([res['rows'][f'{name}|{r}']['mean']
                                                   for r in P.REACHED[name]]))
    text, mac = macros(dry_tree[0])
    x = res['pooled']['window']
    assert (mac['OodWindowPool'], mac['OodWindowPoolLo'], mac['OodWindowPoolHi'],
            mac['OodWindowPoolDir']) == (F88.f2(x['mean']), F88.f2(x['lo']), F88.f2(x['hi']),
                                         'ahead')
    assert 'pooled over the 4 combinations the shift reaches (W, LW, SW, LSW)' in \
        line_of(text, 'OodWindowPool')
    assert 'pooled over the 4 combinations the shift reaches (S, LS, SW, LSW)' in \
        line_of(text, 'OodSetupPool')
    assert 'all combinations of a drawn instance kept together' in line_of(text, 'OodSetupPool')
    md = (dry_tree[0] / 'out' / P.OUT_FILES['md']).read_text()
    assert md.count('Pooled over ') == 2


def test_the_specialists_on_the_denser_windows(dry_tree):
    res = record(dry_tree[0])
    assert sorted(res['specialist']) == ['window']
    x = res['specialist']['window']
    assert abs(x['mean'] - P.DRY_SPEC['window']) < 0.05 and x['direction'] == 'behind'
    assert x['v0.5'] == 'inferior' and x['v1.0'] == 'equivalent'
    assert x['combos'] == WINDOW4 and x['c'] == 4 and x['control_arrays'] == 4 * 5
    text, mac = macros(dry_tree[0])
    assert mac['OodWindowSpecPoolVOne'] == 'equivalent' and mac['OodWindowSpecPoolDir'] == 'behind'
    assert mac['OodWindowSpecPool'] == F88.f2(x['mean'])
    assert 'the policy against the specialist of each combination' in \
        line_of(text, 'OodWindowSpecPool')


def test_the_specialist_comparison_is_p88s_pooled_block(copy_tree):
    F88.configure(boot=P.DRY_BOOT, alpha=0.05)
    seeds = P.check_setting()
    got = P.spec_block(seeds)['window']
    D = {r: F88.seed_pairs(F88.released, F88.specialist, '10x25_oodwindow', r, seeds=seeds)
         for r in WINDOW4}
    want = F88.pooled_block(D, {'Window': WINDOW4})['Window']
    for k in ('mean', 'lo', 'hi', 't_lo', 't_hi', 'direction', 'v0.5', 'v1.0', 'c'):
        assert got[k] == want[k], k


def test_a_specialist_that_differs_where_the_shift_does_not_reach_is_refused(copy_tree):
    p = REG.result_path('specialist', 'LS', 303, '10x25_oodwindow', 'LS')
    a = np.load(p)
    a[5, 0] += 1.0
    np.save(p, a)
    with pytest.raises(P.Refused, match='CONTROL FAILED: window, combination LS is not reached '
                                        'by the shift, but the specialist'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert no_outputs()


def test_a_missing_specialist_array_is_refused(copy_tree):
    p = REG.result_path('specialist', 'SW', 304, '10x25_oodwindow', 'SW')
    os.remove(p)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=env(**P.dry_paths(str(copy_tree))))
    assert r.returncode == 1 and os.path.basename(p) in r.stdout
    with pytest.raises(P.Refused, match='inputs do not count; first: missing .*solo-SW-s304'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert no_outputs()


# ----------------------------------------------------------------- --check and the setting
def test_check_exit_codes(copy_tree):
    e = env(**P.dry_paths(str(copy_tree)))
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 0, r.stdout + r.stderr
    gone = P.summary_path('10x25_oodwindow', 304)
    os.remove(gone)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 1 and os.path.basename(gone) in r.stdout
    shutil.copyfile(P.summary_path('10x25_oodwindow', 303), gone)   # a summary of another seed
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 1 and 'no feasible count for N, L, S, W, LS, LW, SW, LSW' in r.stdout


def test_check_names_the_missing_record(copy_tree):
    os.remove(copy_tree / 'out' / P.FAMILY_FILES['json'])
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=env(**P.dry_paths(str(copy_tree))))
    assert r.returncode == 1 and 'family_inference.json (the record of step family' in r.stdout


def test_the_earlier_environment_is_not_this_procedure(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(P.Refused, match='the procedure reads the exact environment'):
        P.check_setting()
    assert callable(P.legacy_main)


# ----------------------------------------------------------------- the stored arrays
FAMILY_RECORD = os.path.join(ROOT, 'out', 'exact', 'results', 'family_inference.json')
STORED = [REG.result_path('released', None, s, '10x25_cpuref', r)
          for s in (301, 302, 303, 304, 305) for r in R8] + \
    [f'{REG.res_dir("10x25_cpuref")}/Result_{q}_{r}.npy' for q in P.RULES for r in R8]


@pytest.mark.skipif(not os.path.exists(FAMILY_RECORD) or
                    not all(os.path.exists(os.path.join(ROOT, p)) for p in STORED),
                    reason='needs the stored exact-mask arrays of 10x25 and the record of step '
                           'family in out/exact')
def test_the_stored_arrays_give_famrule_exactly(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    for v in ('PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    rec = json.load(open(FAMILY_RECORD))
    if rec.get('seeds') != [301, 302, 303, 304, 305] or rec.get('boot') != 20000:
        pytest.skip('the stored record of step family is not the five-seed record with B = 20000')
    F88.configure(boot=20000, alpha=0.05)
    rows = P.family_rows(P.differences('10x25_cpuref', R8, [301, 302, 303, 304, 305]), R8)
    for r in R8:
        for k in P.FAMILY_KEYS:
            assert rows[r][k] == rec['rule_pool'][r][k], (r, k)
