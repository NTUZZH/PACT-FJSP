"""scripts/p118_per_floor.py on synthetic arrays.

  * --dry builds arrays of 20, 27 and 30 modules with known differences and a record of p88 for
    20x25 and 30x25, and passes: the pooled differences of 27x25 are recovered, every number of
    the record is reproduced from the arrays, the 27-module values lie between the other two;
  * the macro file holds letters-only names with the prefix Pfl, no name twice, and one line per
    quantity group;
  * --check exits 0 on the complete synthetic tree and 1 when one array of 27x25 is missing;
  * a missing array and a record that the arrays do not reproduce are refused, and no output is
    left behind.

Synthetic arrays only, under the pytest temporary folder; nothing in the repository is written.
Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q tests/test_p118_per_floor.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

import arm_registry as REG  # noqa: E402
import p118_per_floor as P  # noqa: E402

PY = sys.executable
SCRIPT = os.path.join(ROOT, 'scripts', 'p118_per_floor.py')

pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(ROOT, 'data', 'FAMILY', s, 'desc', 'N'))
            for s in (P.SIZE,) + P.SIDE),
    reason='needs the descriptors of data/FAMILY/20x25, 27x25 and 30x25')


def env(**kv):
    e = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    e.update({k: str(v) for k, v in kv.items()})
    return e


@pytest.fixture(scope='module')
def dry_tree(tmp_path_factory):
    tree = tmp_path_factory.mktemp('p118') / 'tree'
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


def test_dry_run_passes(dry_tree):
    tree, r = dry_tree
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'p118 dry run on synthetic arrays passed' in r.stdout
    res = json.load(open(tree / 'out' / P.OUT_FILES['json']))
    assert res['dry'] and res['reproduction']['identical'] and res['reproduction']['compared'] > 900
    x = res['sizes'][P.SIZE]
    assert abs(x['parity_pooled']['All']['mean'] - P.DRY_GAP[P.SIZE][0]) < 0.15
    assert abs(x['rule_pool_pooled']['mean'] - P.DRY_GAP[P.SIZE][1]) < 0.15
    assert all(b['between'] for b in res['between'].values())
    assert (tree / 'out' / P.OUT_FILES['md']).read_text().startswith('# Per-floor batch, 27x25')


def test_macro_names(dry_tree):
    tree, _ = dry_tree
    text = (tree / 'out' / P.OUT_FILES['macros']).read_text()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names)) == 162
    assert all(re.fullmatch(r'Pfl[A-Za-z]+', x) for x in names)
    for x in ('PflN', 'PflSeedN', 'PflModules', 'PflParPoolAll', 'PflParPoolAllLo',
              'PflRulePool', 'PflRuleVerdictAheadN', 'PflTwentyParPoolAll', 'PflThirtyRulePool'):
        assert x in names
    assert re.search(r'\\newcommand\{\\PflN\}\{50\}', text)
    assert '% DRY RUN on synthetic arrays' in text
    assert not re.search(r'\b(regimes?|arms?|blind)\b|released policy', text.lower())


def test_check_exit_codes(copy_tree):
    e = env(**P.dry_paths(str(copy_tree)))
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 0, r.stdout + r.stderr
    gone = REG.result_path('released', None, 303, P.SIZE, 'LS')
    os.remove(gone)
    r = subprocess.run([PY, '-B', SCRIPT, '--check'], capture_output=True, text=True, cwd=ROOT,
                       env=e)
    assert r.returncode == 1
    assert os.path.basename(gone) in r.stdout


def test_refuses_a_missing_array(copy_tree):
    os.remove(REG.result_path('specialist', 'SW', 302, P.SIZE, 'SW'))
    with pytest.raises(P.Refused, match='inputs incomplete'):
        P.run(dry=True)
    assert not any(os.path.exists(p) for p in P.outputs().values())


def test_refuses_a_record_the_arrays_do_not_reproduce(copy_tree):
    path = copy_tree / 'out' / P.RECORD
    rec = json.load(open(path))
    rec['larger_shops']['30x25']['parity']['LW']['mean'] += 0.01
    json.dump(rec, open(path, 'w'), indent=1)
    with pytest.raises(P.Refused, match='does not reproduce the record in 1 of'):
        P.run(dry=True)
    assert not any(os.path.exists(p) for p in P.outputs().values())


def test_refuses_a_record_of_another_setting(copy_tree):
    path = copy_tree / 'out' / P.RECORD
    rec = json.load(open(path))
    rec['seeds'] = [301, 302, 303]
    json.dump(rec, open(path, 'w'), indent=1)
    with pytest.raises(P.Refused, match='written with seeds'):
        P.run(dry=True)
