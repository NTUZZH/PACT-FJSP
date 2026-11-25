"""P128, the policy against the specialists at the production batches (50x25n30, 80x25n30).

  * --dry on synthetic arrays passes: the pooled differences are recovered, every control
    compares a positive number of values (6720 makespans of p102 and p103 in 224 arrays, 24 rule
    and 56 specialist values per study), the 10, 20, 27 and 30 module rows are reproduced, and
    every macro name is letters only with the prefix Pbs;
  * the run refuses, and leaves no output, when one default-level array of p103 differs in one
    makespan (control ii), when one interval end of a pol_rule row of p102 differs (control iii)
    and when one verdict of a pol_spec row of p103 differs (control iv);
  * the specialist arrays it reads are the files of the evaluation runner's jobs, and section H
    reads the two sets through spec_reader only inside p88_setting;
  * --check says no, naming the missing arrays, on an empty result tree.

Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      tests/test_p128_production_specialists.py
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None, RESULTS=None,
                                        SEEDS=None, PACT_FIRST_MASK=None, PACT_MASK_TOL=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

import p88_family_inference as F88  # noqa: E402
import p128_production_specialists as P  # noqa: E402
import production_specialists_plan as PS  # noqa: E402

SCRIPT = os.path.join(ROOT, 'scripts', 'p128_production_specialists.py')
PY = sys.executable


def env(**kv):
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kv.items()})
    return e


def test_dry_run_passes_and_names_its_macros(tmp_path):
    tree = tmp_path / 'dry'
    r = subprocess.run([PY, '-B', SCRIPT, '--dry', str(tree)], capture_output=True, text=True,
                       cwd=ROOT, env=env())
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert 'p128 dry run on synthetic arrays passed' in r.stdout
    assert '6720 makespans equal to p102 and p103' in r.stdout
    assert 'p102 24 rule and 56 specialist values, p103 24 rule and 56 specialist values' in r.stdout
    res = json.load(open(tree / 'out' / P.OUT_FILES['json']))
    assert res['dry'] is True and list(res['sizes']) == list(P.SIZES)
    assert list(res['side_by_side']['rows']) == ['10x25', '20x25', '27x25', '30x25', '50x25n30',
                                                 '80x25n30']
    assert set(res['side_by_side']['reproduction']) == {'20x25,30x25', '27x25', '10x25'}
    assert res['controls']['arrays']['makespans'] == 6720
    st = res['controls']['arrays']['per_study']
    assert {k: {a: v['arrays'] for a, v in x.items()} for k, x in st.items()} == {
        s: {'policy': 40, 'specialist': 40, 'rules': 32} for s in ('p102', 'p103')}
    tex = open(tree / 'out' / P.OUT_FILES['macros']).read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', tex)
    assert names and all(re.fullmatch(r'Pbs[A-Za-z]+', x) for x in names)
    assert len(names) == len(set(names)) == 259
    for w in ('Fifty', 'Eighty'):
        assert f'Pbs{w}ParPoolAll' in names and f'Pbs{w}RulePool' in names
        assert f'Pbs{w}RuleVerdictAheadN' in names and f'Pbs{w}ParIutAllOneNi' in names
    assert 'PbsTenRulePool' in names
    assert 'DRY RUN' in tex and 'negative = policy shorter' in tex


@pytest.fixture
def synthetic(tmp_path):
    tree = str(tmp_path / 'syn')
    with P.dry_setting(tree):
        P.build_synthetic(tree)
        yield tree


def outputs_absent():
    return not any(os.path.exists(p) for p in P.outputs().values())


def test_refuses_when_a_study_array_differs(synthetic):
    st = P.studies()['p103']
    p = st['path']('80x25n30', 'LS', P.labels('LS', 303)['specialist'])
    with np.load(p) as z:
        a = np.asarray(z['makespan'], dtype=float)
    a[17] += 1.0
    np.savez(p, makespan=a)
    with pytest.raises(P.Refused, match=r'control \(ii\).*first differing instance 17'):
        P.run(dry=True)
    assert outputs_absent()


@pytest.mark.parametrize('study,key,combo,field,control', [
    ('p102', 'pol_rule', 'LW', 'hi', 'iii'), ('p103', 'pol_spec', 'SW', 'v1.0', 'iv')])
def test_refuses_when_a_record_value_differs(synthetic, study, key, combo, field, control):
    path = P.studies()[study]['record']
    rec = json.load(open(path))
    row = rec['sizes']['50x25n30']['policy']['default']['rows'][key][combo]
    row[field] = row[field] + 1e-6 if isinstance(row[field], float) else 'inferior'
    json.dump(rec, open(path, 'w'))
    with pytest.raises(P.Refused, match=rf'control \({control}\): 50x25n30 {combo} {key}'):
        P.run(dry=True)
    assert outputs_absent()


def test_reads_the_files_of_the_runner_and_patches_p88_only_inside():
    for j in PS.jobs():
        name = j['campaign'][:-len('_spec')]
        seed = int(j['model'].rsplit('-s', 1)[1])
        assert P.spec_path(name, j['regimes'], seed) == j['files'][0]
    orig = F88.specialist
    with P.p88_setting(P.SIZES):
        assert F88.specialist is P.spec_reader and F88.SPEC_SIZES == P.SIZES
        assert F88.POOL_SIZES == ()
    assert F88.specialist is orig and F88.SPEC_SIZES == ('20x25', '30x25')
    with pytest.raises(P.Refused):
        P.spec_reader(301, '20x25', 'N')


def test_check_says_no_on_an_empty_result_tree(tmp_path):
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=env(PACT_RESULTS=tmp_path / 'results', PACT_OUT=tmp_path / 'out'))
    assert r.returncode == 1
    assert '224 of 224 input arrays missing' in r.stdout
    assert 'p128: record of p88 refused' in r.stdout
    assert not (tmp_path / 'results').exists()                  # nothing written there
