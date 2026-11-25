"""scripts/p119_asym_setups.py: the procedure of scripts/p33_ood_table.py (the procedure of
FamRule<R>) for 10x25_asymsetup.

  * p119 imports p33 and uses its functions (set_rows, shift_magnitudes, infeasible,
    summary_path); importing p33 runs nothing of its analysis, prints and writes nothing;
    shift_magnitudes with a data folder reads that folder in place of 10x25_oodsetup (10x25
    itself gives the ratio 1 and a symmetric matrix);
  * on the stored arrays of 10x25 and 10x25_oodsetup (exact environment), the rule rows of p119
    equal the rows p33 computes for its own sets, bit for bit, and its infeasible counts;
  * --dry builds the three sets with known leads, runs p33's dry path in the same tree, and
    passes; the macro file holds letters-only names with the prefix Asy, no name twice, a verdict
    for every lead;
  * the pooled lead over S, LS, SW and LSW is p33's routine (p88.pooled_block on the differences
    of the per-combination verdicts), recovers the synthetic lead, and on 10x25_oodsetup equals
    the pooled row of p33's record (a difference or a missing row is refused);
  * --check exits 0 on the complete synthetic tree and 1 when a validator summary or p33's record
    is missing;
  * an array of 10x25_asymsetup that differs from the in-distribution array on a combination
    without SETUP fails the control, a row that differs from p33's record is refused, and no
    output is left behind.

Synthetic arrays under the pytest temporary folder, and the stored arrays read only; nothing in
the repository is written. Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      tests/test_p119_asym_setups.py
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
import p119_asym_setups as P  # noqa: E402
import p33_ood_table as P33  # noqa: E402
import p88_family_inference as F88  # noqa: E402

PY = sys.executable
SCRIPT = os.path.join(ROOT, 'scripts', 'p119_asym_setups.py')

pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(ROOT, d, 'desc', 'N')) for _, _, d in P.SETS + P33.SETS),
    reason='needs the descriptors of data/FAMILY/10x25, 10x25_oodsetup, 10x25_oodwindow and '
           '10x25_asymsetup')


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
    tree = tmp_path_factory.mktemp('p119') / 'tree'
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


# ----------------------------------------------------------------- p33's functions
def test_p33_is_imported_and_runs_nothing(tmp_path):
    code = ('import os, sys; sys.path.insert(0, "scripts"); import p33_ood_table as P; '
            'print(sorted(k for k in ("lead", "infeas", "sh", "lines", "placebo", "tok") '
            'if hasattr(P, k)))')
    r = subprocess.run([PY, '-B', '-c', code], capture_output=True, text=True, cwd=ROOT,
                       env=env(PACT_OUT=str(tmp_path / 'out')))
    assert r.returncode == 0, r.stderr
    assert r.stdout == '[]\n' and r.stderr == ''                 # nothing printed, no banner
    assert not (tmp_path / 'out').exists()
    assert P.P33 is P33 and P.summary_path is P33.summary_path
    assert not hasattr(P, 'load_p33')
    src = open(SCRIPT).read()
    assert 'P33.set_rows(seeds, SETS, REACHED)' in src and 'P33.shift_magnitudes(oods=data)' in src
    assert P.REACHED == {'ind': list(REG.COMBOS), 'setup': ['S', 'LS', 'SW', 'LSW'],
                         'asym': ['S', 'LS', 'SW', 'LSW']}


def test_shift_magnitudes_of_a_named_folder(monkeypatch):
    monkeypatch.chdir(ROOT)
    ref = P33.shift_magnitudes()
    assert ref['setup_x'] > 1.5 and ref['asym']                  # 10x25_oodsetup
    same = P33.shift_magnitudes(oods='data/FAMILY/10x25')
    assert same['setup_x'] == 1.0 and not same['asym']
    asym = P33.shift_magnitudes(oods='data/FAMILY/10x25_asymsetup')
    assert abs(asym['setup_x'] - 1.0) < 0.05 and asym['asym']
    assert ref['window_x'] == same['window_x'] == asym['window_x']


STORED = [REG.result_path('released', None, s, k, r)
          for k in ('10x25_cpuref', '10x25_oodsetup', '10x25_oodwindow')
          for s in (301, 302, 303, 304, 305) for r in REG.COMBOS]


@pytest.mark.skipif(not all(os.path.exists(os.path.join(ROOT, p)) for p in STORED),
                    reason='needs the stored exact-mask arrays of 10x25, 10x25_oodsetup and '
                           '10x25_oodwindow')
def test_rule_rows_equal_p33s_rows_on_the_stored_arrays(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    for v in ('PACT_OUT', 'PACT_RESULTS'):
        monkeypatch.delenv(v, raising=False)
    F88.configure(boot=500, alpha=0.05)
    seeds = P.check_setting()
    want, _, _ = P33.set_rows(seeds)                             # p33's own three sets
    monkeypatch.setattr(P, 'SETS', P.SETS[:2])
    lead, floor, infeas = P.rule_block(seeds)
    assert floor == 0.0
    n = 0
    for (name, r), x in lead.items():
        assert x == want[(name, r)], (name, r)
        res, sfx = REG.res_dir(dict((a, b) for a, b, _ in P.SETS)[name]), '_xmask'
        assert infeas[(name, r)] == sum(P33.infeasible(res, REG.tag('released', None, s), r, sfx)
                                        for s in seeds)
        n += 1
    assert n == 16


# ----------------------------------------------------------------- the dry run
def test_dry_run_passes(dry_tree):
    tree, r = dry_tree
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'p119 dry run on synthetic arrays passed' in r.stdout
    res = json.load(open(tree / 'out' / P.OUT_FILES['json']))
    assert res['dry'] and res['control_arrays'] == 2 * 4 * (5 * 2 + 4)
    for name in ('ind', 'setup', 'asym'):
        for rc in REG.COMBOS:
            want = P.DRY_LEAD[name] if 'S' in rc else P.DRY_OTHER_LEAD
            x = res['lead'][f'{name}|{rc}']
            assert abs(x['mean'] - want) < 0.1 and x['verdict'] == 'ahead'
            fam = list(REG.COMBOS) if (name == 'ind' or 'S' not in rc) else list(P.SETUP_COMBOS)
            assert x['holm_family'] == fam and x['reached'] == (name == 'ind' or 'S' in rc)
    c = res['classification']['asym']
    assert c['hurt'] == list(P.SETUP_COMBOS) and c['floor'] == 0.0 and not c['reversed']
    assert not c['tied'] and c['verdicts']['ahead'] == list(P.SETUP_COMBOS)
    assert res['rule']['p33_record'] == dict(file=str(tree / 'out' / P.P33_RECORD), compared=160,
                                             identical=True, keys=list(P.P33_KEYS))
    assert (tree / 'out' / 'paper' / 'macros_ood.tex').exists()   # p33's dry path in the tree
    md = (tree / 'out' / P.OUT_FILES['md']).read_text()
    assert '| 10x25_asymsetup | 10x25_oodsetup |' in md
    assert '(the procedure of FamRule<R>)' in md and 'p33 statistics' not in md


def test_macro_names(dry_tree):
    tree, _ = dry_tree
    text = (tree / 'out' / P.OUT_FILES['macros']).read_text()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names)) == 212
    assert all(re.fullmatch(r'Asy[A-Za-z]+', x) for x in names)
    for x in ('AsyN', 'AsySetupX', 'AsyRefSetupX', 'AsyInfeas', 'AsyFloor', 'AsyHurt', 'AsyTied',
              'AsyReversed', 'AsyLeadLSW', 'AsyLeadLSWLo', 'AsyRefLeadS', 'AsySpecS',
              'AsySpecPoolSetup', 'AsyRefSpecPoolAll', 'AsyLeadSVerdict', 'AsyRefLeadSWVerdict'):
        assert x in names, x
    for w in ('Asy', 'AsyRef'):                  # the pooled lead: mean, interval, direction
        assert [x for x in names if x.startswith(f'{w}LeadPool')] == \
            [f'{w}LeadPool{s}' for s in ('', 'Lo', 'Hi', 'Dir')]
    for w in ('Asy', 'AsyRef'):
        assert all(f'{w}Lead{r}Verdict' in names for r in REG.COMBOS)
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    assert mac['AsyLeadSVerdict'] == 'ahead'
    assert not re.search(r'\b(regimes?|arms?|blind)\b|released policy', text.lower())
    assert 'scripts/p33_ood_table.py' in text and 'Holm over S, LS, SW, LSW' in text
    assert 'default_rng(7)' not in text and 'functions of scripts/p33' not in text


def test_the_pooled_lead_of_the_dry_run(dry_tree):
    tree, _ = dry_tree
    res = json.load(open(tree / 'out' / P.OUT_FILES['json']))
    for name in ('asym', 'setup'):
        x = res['lead_pooled'][name]
        assert abs(x['mean'] - P.DRY_LEAD[name]) < 0.05 and x['hi'] < 0
        assert x['direction'] == 'ahead' and x['combos'] == list(P.SETUP_COMBOS) and x['c'] == 4
        assert x['mean'] == pytest.approx(np.mean([res['lead'][f'{name}|{r}']['mean']
                                                   for r in P.SETUP_COMBOS]))
    assert res['rule']['p33_pooled'] == dict(file=str(tree / 'out' / P.P33_RECORD),
                                             compared=len(P.P33_POOLED_KEYS), identical=True,
                                             keys=list(P.P33_POOLED_KEYS))
    text = (tree / 'out' / P.OUT_FILES['macros']).read_text()
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    x = res['lead_pooled']['asym']
    assert (mac['AsyLeadPool'], mac['AsyLeadPoolLo'], mac['AsyLeadPoolHi'],
            mac['AsyLeadPoolDir']) == (F88.f2(x['mean']), F88.f2(x['lo']), F88.f2(x['hi']), 'ahead')
    ood = (tree / 'out' / 'paper' / 'macros_ood.tex').read_text()
    for s in ('', 'Lo', 'Hi', 'Dir'):            # AsyRef equals the record of p33
        v = re.search(r'\\newcommand\{\\OodSetupPool' + s + r'\}\{([^}]*)\}', ood).group(1)
        assert mac[f'AsyRefLeadPool{s}'] == v
    line = next(x for x in text.splitlines() if '\\newcommand{\\AsyLeadPool}' in x)
    assert 'pooled over the 4 combinations with SETUP (S, LS, SW, LSW)' in line


def test_the_pooled_lead_is_p33s_routine(copy_tree):
    F88.configure(boot=P.DRY_BOOT, alpha=0.05)
    seeds = P.check_setting()
    got = P.rule_pooled(seeds)
    for name, key in (('setup', '10x25_oodsetup'), ('asym', '10x25_asymsetup')):
        want = F88.pooled_block(P33.differences(key, P.SETUP_COMBOS, seeds),
                                {'x': list(P.SETUP_COMBOS)})['x']
        for k in ('mean', 'lo', 'hi', 't_lo', 't_hi', 'direction', 'c'):
            assert got[name][k] == want[k], (name, k)


def test_a_pooled_row_that_differs_from_p33s_record_is_refused(copy_tree):
    p = copy_tree / 'out' / P.P33_RECORD
    rec = json.load(open(p))
    rec['pooled']['setup']['lo'] -= 1e-12
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P.Refused, match="pooled difference .* differs from p33's record:\n  "
                                        "setup pooled lo:"):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert not any(os.path.exists(x) for x in P.outputs().values())
    del rec['pooled']
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P.Refused, match='holds no pooled row of setup'):
        P.run(dry=True, boot=P.DRY_BOOT)


def test_check_exit_codes(copy_tree):
    e = env(**P.dry_paths(str(copy_tree)))
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 0, r.stdout + r.stderr
    gone = P.summary_path('10x25_asymsetup', 304)
    os.remove(gone)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 1 and os.path.basename(gone) in r.stdout
    shutil.copyfile(P.summary_path('10x25_asymsetup', 305), gone)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 0, r.stdout
    os.remove(copy_tree / 'out' / P.P33_RECORD)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 1 and 'ood.json (the record of step ood' in r.stdout


def test_control_failure_is_refused(copy_tree):
    p = REG.result_path('specialist', 'LW', 302, '10x25_asymsetup', 'LW')
    a = np.load(p)
    a[7, 0] += 1.0
    np.save(p, a)
    with pytest.raises(P.Refused, match='CONTROL FAILED: asym, combination LW'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert not any(os.path.exists(x) for x in P.outputs().values())


@pytest.mark.parametrize('key,what', [('setup|S', 'lo'), ('ind|LW', 'verdict'),
                                      ('setup|N', 'holm_family')])
def test_a_difference_from_p33s_record_is_refused(copy_tree, key, what):
    p = copy_tree / 'out' / P.P33_RECORD
    rec = json.load(open(p))
    x = rec['rows'][key]
    x[what] = {'lo': x['lo'] - 1e-12, 'verdict': 'behind', 'holm_family': ['N']}[what]
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P.Refused, match=f"differs from p33's record.*\n  {key.replace('|', ' ')} "
                                        f"{what}:"):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert not any(os.path.exists(x) for x in P.outputs().values())


def test_a_record_of_another_setting_is_refused(copy_tree):
    p = copy_tree / 'out' / P.P33_RECORD
    rec = json.load(open(p))
    rec['boot'] = 20000
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P.Refused, match='records .* this run uses'):
        P.run(dry=True, boot=P.DRY_BOOT)


def test_a_short_array_is_refused(copy_tree):
    p = REG.result_path('released', None, 305, '10x25_asymsetup', 'S')
    np.save(p, np.load(p)[:-1])
    with pytest.raises(P.Refused, match='99 rows, the test set has 100'):
        P.run(dry=True, boot=P.DRY_BOOT)
    assert not any(os.path.exists(x) for x in P.outputs().values())
