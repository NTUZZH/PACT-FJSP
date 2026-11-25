"""scripts/p121_wrong_descriptor.py: schedules of the policy under a wrong constraint descriptor.

  * p40's pairs, cases and rule buckets are taken from its source, and nothing else of p40 runs
    (sys.argv unchanged, neither torch nor params.py imported);
  * --dry builds synthetic records and passes; its macro file holds letters-only names with the
    prefix Wrd, none twice; case A has no rejection, case C none against the configured
    descriptor, and every LSW schedule built under N is rejected against the true descriptor;
  * the check accepts the complete synthetic records under their design and refuses a missing
    record, and the analysis then refuses and leaves no output behind;
  * a stored record is never overwritten and a record of another design is refused;
  * each malformed descriptor of the fault table fails or warns at the expected check, by the
    repository's own check functions, and the two wrong descriptors pass every check;
  * the greedy rollout of the script reproduces the stored arrays of 10x25_xmask bit for bit,
    and the policy's output does not change with the descriptor vector of the other
    combination (two instances, every decision; needs the checkpoint).

Synthetic records under the pytest temporary folder; the stored files are read only and nothing
in the repository is written. Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 3 python -B -m pytest -q \
      tests/test_p121_wrong_descriptor.py
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
                                        PACT_FIRST_MASK=None, PACT_MASK_TOL=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

import arm_registry as REG  # noqa: E402
import p121_wrong_descriptor as P  # noqa: E402

PY = sys.executable
SCRIPT = os.path.join(ROOT, 'scripts', 'p121_wrong_descriptor.py')

pytestmark = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(ROOT, P.DATA, 'desc', c)) for c in ('LSW', 'N', 'SW', 'LS')),
    reason='needs the descriptors of data/FAMILY/10x25')


def env(**kv):
    e = {k: v for k, v in os.environ.items() if k not in ('PACT_FIRST_MASK', 'PACT_MASK_TOL')}
    e.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    e.update({k: str(v) for k, v in kv.items()})
    return e


@pytest.fixture(scope='module')
def dry_tree(tmp_path_factory):
    tree = tmp_path_factory.mktemp('p121') / 'tree'
    r = subprocess.run([PY, '-B', SCRIPT, '--dry', str(tree)], capture_output=True, text=True,
                       cwd=ROOT, env=env())
    return tree, r


@pytest.fixture
def copy_tree(dry_tree, tmp_path, monkeypatch):
    tree = tmp_path / 'tree'
    shutil.copytree(dry_tree[0], tree)
    shutil.rmtree(tree / 'out', ignore_errors=True)
    for k, v in P.dry_paths(str(tree)).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(ROOT)
    return tree


def dry_names():
    return P.instance_names('LSW', P.DRY_N)


# ----------------------------------------------------------------- p40's definitions
def test_p40_definitions_come_from_its_source_only():
    argv = list(sys.argv)
    before = set(sys.modules)
    ns = P.load_p40()
    new = set(sys.modules) - before
    assert sys.argv == argv
    assert not new & {'torch', 'params', 'scipy.stats', 'constraint_family.fjsp_env_family'}
    assert [tuple(p) for p in ns['PAIRS']] == [('LSW', 'N', 'Lsw'), ('SW', 'LS', 'Sw')]
    assert list(ns['CASES']) == ['A', 'B', 'C', 'D']
    c = ns['classify'](['job 0: op 5 starts before pred ct+lag',
                        'job 1: setup of op 7 starts before pred ct+lag',
                        'mch 3: op 9 setup+proc block overlaps predecessor',
                        'mch 4: op 2 block overlaps outage (30,36)', 'op 1 duration mismatch'])
    assert c == dict(lag=2, setup=1, window=1, other=1)
    assert P.partner_of(P.pairs_of(ns)) == {'LSW': 'N', 'N': 'LSW', 'SW': 'LS', 'LS': 'SW'}


def test_setups_before_their_job_is_present_are_counted_apart_from_lag():
    classify = P.load_p40()['classify']
    msgs = ['job 0: op 5 starts before pred ct+lag',
            'job 1: setup of op 7 starts before pred ct+lag',
            'job 2: setup of first op 10 starts before release 4',
            'mch 3: op 9 setup+proc block overlaps predecessor',
            'mch 4: op 2 block overlaps outage (30,36)']
    assert P.buckets(msgs, classify) == dict(lag=1, setup=1, window=1, other=0, presence=2)
    assert classify(msgs)['lag'] == 2


# ----------------------------------------------------------------- dry run
def test_dry_run_passes_with_the_known_answers(dry_tree):
    tree, r = dry_tree
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'known answers reproduced' in r.stdout
    text = open(tree / 'out' / 'manuscript' / 'macros_wrongdesc.tex').read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert names and len(names) == len(set(names))
    assert all(re.fullmatch(r'Wrd[A-Za-z]+', n) for n in names)
    assert not re.search(r'(?<![A-Za-z-])(nan|inf)(?![A-Za-z-])', text, re.I)
    res = json.load(open(tree / 'out' / 'results' / 'wrong_descriptor.json'))
    n = 5 * P.DRY_N
    for p in res['part1']:
        A, C = p['cases']['A']['pooled'], p['cases']['C']['pooled']
        assert A['true']['rejected'] == 0 and A['configured']['rejected'] == 0
        assert C['configured']['rejected'] == 0 and C['true']['schedules'] == n
        assert A['change'] == 0.0
    lsw = next(p for p in res['part1'] if p['true'] == 'LSW')
    assert lsw['cases']['C']['pooled']['true']['rejected'] == n
    sw = next(p for p in res['part1'] if p['true'] == 'SW')
    assert sw['cases']['C']['pooled']['true']['lag'] == 0
    assert sw['cases']['C']['pooled']['true']['setup'] == 0
    md = open(tree / 'out' / 'results' / 'wrong_descriptor.md').read()
    for word in ('regime', 'token', ' arm ', 'blind', 'released policy', 'reviewer'):
        assert word not in md.lower(), word


# ----------------------------------------------------------------- check and refusals
def test_check_accepts_complete_records_and_refuses_a_missing_one(copy_tree):
    assert P.check(log=lambda *_: None, n=P.DRY_N, design_of=P.dry_design) == 0
    os.remove(P.record_path(302, 'N'))
    lines = []
    assert P.check(log=lines.append, n=P.DRY_N, design_of=P.dry_design) == 1
    assert any('missing' in x and 'xmask-v1-s302_N' in x for x in lines), lines
    with pytest.raises(P.Incomplete):
        P.run(dry_names(), P.dry_design)
    outs = {k: P.REG.out(v) for k, v in P.OUT_FILES.items()}
    assert not any(os.path.exists(p) for p in outs.values())


def test_the_real_check_refuses_the_synthetic_records(copy_tree):
    assert P.check(log=lambda *_: None) == 1


def test_a_stored_record_is_never_overwritten_and_a_foreign_one_is_refused(copy_tree):
    names = dry_names()
    rec = json.load(open(P.record_path(301, 'LSW')))
    with pytest.raises(P.Refused, match='never overwritten'):
        P.write_record(301, 'LSW', rec)
    rec['design']['checkpoint_sha256'] = 'other'
    with open(P.record_path(301, 'LSW'), 'w') as f:
        json.dump(rec, f)
    with pytest.raises(P.Refused, match='another design'):
        P.read_record(301, 'LSW', P.dry_design(301), names, 'N')
    st = REG.result_path('released', None, 301, P.STORED, 'N')
    a = np.load(st)
    a[0] += 1.0
    np.save(st, a)
    with pytest.raises(P.Refused, match='stored array'):
        P.read_record(301, 'N', P.dry_design(301), names, 'LSW')


# ----------------------------------------------------------------- checks and faults
def test_each_fault_is_caught_by_the_expected_check():
    d = P.read_desc('LSW', 'instance_000')
    jl, pt, _ = P.read_instance('instance_000')
    rows = {r['fault']: r for r in P.fault_table(d, jl, pt)}
    want = {'neglag': ['C1'], 'negsetup': ['C1'], 'overlap': ['C1'], 'reversed': ['C1'],
            'twice': ['C4'], 'asym': ['C4']}
    assert {k: r['caught_by'] for k, r in rows.items()} == want
    for k in ('neglag', 'negsetup', 'overlap', 'reversed'):
        assert rows[k]['statuses']['C1'] == 'FAIL'
    for k in ('twice', 'asym'):
        assert rows[k]['statuses']['C1'] == 'PASS' and rows[k]['statuses']['C4'] == 'WARN'
    assert len(rows['asym']['messages']['C4']) == 1
    assert 'asymmetry' in rows['asym']['messages']['C4'][0]
    assert all(r['statuses']['C7'] == 'MANUAL' for r in rows.values())


def test_the_wrong_descriptors_pass_every_check():
    for nm in P.instance_names('LSW', 5):
        jl, pt, _ = P.read_instance(nm)
        for combo in ('N', 'LS', 'LSW', 'SW'):
            rep = P.descriptor_checks(P.read_desc(combo, nm), jl, pt)
            assert [rep[c][0] for c in P.CHECK_IDS] == ['PASS'] * 5 + ['MANUAL', 'MANUAL'], \
                (combo, nm, rep)
    loc = P.check_locations()
    assert all(re.fullmatch(r'constraint_family/checks\.py:\d+', loc[c]['location'])
               for c in P.CHECK_IDS)


# ----------------------------------------------------------------- the rollout path
PROBE = r'''
import json, sys
sys.path.insert(0, "scripts")
import numpy as np
import p121_wrong_descriptor as P
R = P.Roller(301)
names = P.instance_names("LSW", 2)
out = []
for combo, other in (("LSW", "N"), ("N", "LSW")):
    ref = P.stored_array(301, combo, P.N_TEST)
    for i, nm in enumerate(names):
        jl, pt, meta = P.read_instance(nm)
        row, k, eq = R.roll(P.read_desc(combo, nm), P.read_desc(other, nm), jl, pt, meta, True)
        out.append([combo, nm, row["makespan"], float(ref[i]), k, eq, int(pt.shape[0])])
import torch
print(json.dumps(dict(rows=out, threads=torch.get_num_threads())))
'''


@pytest.mark.skipif(not os.path.exists(os.path.join(
    ROOT, 'trained_network/FAMILY/10x25+family+xmask-v1-s301.pth')), reason='needs the checkpoint')
def test_rollout_reproduces_the_stored_arrays_and_ignores_the_descriptor_vector():
    r = subprocess.run([PY, '-B', '-c', PROBE], capture_output=True, text=True, cwd=ROOT,
                       env=env(PACT_ENV='exact', PACT_SEEDS='301-305'), timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    res = json.loads(r.stdout.strip().splitlines()[-1])
    assert res['threads'] == 1
    assert len(res['rows']) == 4
    for combo, nm, ms, ref, k, eq, n_ops in res['rows']:
        assert ms == ref, (combo, nm, ms, ref)
        assert k == n_ops and eq, (combo, nm, k, n_ops, eq)
