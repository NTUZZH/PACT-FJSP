"""Tests for scripts/p72_aware_rules.py in the exact environment.

  * the environment options: none in the legacy environment (and a refusal
    when a mask variable asks for an exact correction there), the registry's
    exact options in the exact environment (and a refusal of a contradicting
    variable);
  * result folders: <size> in the legacy environment, <size>_xmask in the
    exact one;
  * the per-decision checks of the exact smoke test hold on one instance of S
    and of W for every rule, and the same checks report the legacy first
    decision (every compatible pair legal, pair-start table at zero), so they
    are not vacuous;
  * under the exact options SPT through p72's code path equals the stored
    10x25_xmask array, ECTND equals it as well, and ECT equals the stored
    legacy ECT array;
  * an array is written through a temporary name with the bytes of np.save;
  * the exact summary is refused, writing nothing, while a PACT seed file is
    missing.

Run from the repository root:  python3 -m pytest tests/test_aware_rules_exact.py -q
CPU only, one thread, about half a minute.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None, PACT_FIRST_MASK=None,
                                        PACT_MASK_TOL=None)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402  (parses sys.argv once, at import)
finally:
    sys.argv = _argv
configs.device = 'cpu'


def _load_p72():
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        spec = importlib.util.spec_from_file_location(
            'p72_under_test', os.path.join(ROOT, 'scripts', 'p72_aware_rules.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


P72 = _load_p72()
REG = P72.REG


@pytest.fixture
def exact(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setattr(P72, 'ENV', 'exact')
    yield


def _task(regime, rule, k, masks):
    dp, stem = P72.descriptors('10x25', regime, 0)[k]
    return ('10x25', regime, rule, k, dp, stem, masks)


def test_options_legacy(monkeypatch):
    assert P72.mask_options() == {}
    assert P72.result_path('10x25', 'ECT', 'S') == 'test_results/FAMILY/10x25/Result_ECT_S.npy'
    monkeypatch.setenv('PACT_FIRST_MASK', '1')
    with pytest.raises(SystemExit, match='legacy environment is active'):
        P72.mask_options()


def test_options_exact(exact, monkeypatch):
    assert P72.mask_options() == {'first_mask': 'exact', 'mask_ties': 'exact'}
    assert P72.result_path('80x25', 'LLF', 'LSW') == \
        'test_results/FAMILY/80x25_xmask/Result_LLF_LSW.npy'
    assert P72._policy_tag() == 'greedy+10x25+family+xmask-v1'
    monkeypatch.setenv('PACT_MASK_TOL', '0')
    with pytest.raises(SystemExit, match='contradictory mask settings'):
        P72.mask_options()


def test_policy_tag_legacy():
    assert P72.POLICY == 'greedy+10x25+family+jointblind-v1'


@pytest.mark.parametrize('regime', ['S', 'W'])
def test_decision_checks_exact(exact, regime):
    masks = P72.mask_options()
    P72._init_worker()
    for rule in ('SPT', 'ECTND', 'ECT', 'SSPT', 'LLF'):
        chk = {'decisions': 0, 'fail': []}
        out = P72.rollout(_task(regime, rule, 0, masks), check=chk)
        assert out[6] == [], out[6]
        assert chk['decisions'] > 0 and chk['fail'] == [], chk['fail'][:3]


def test_decision_checks_detect_legacy_first_decision():
    """Instance 0 of S: the legacy first decision admits pairs that start after
    the earliest cold-start setup, and its pair-start table is still zero."""
    P72._init_worker()
    chk = {'decisions': 0, 'fail': []}
    P72.rollout(_task('S', 'SPT', 0, {}), check=chk)
    first = [x for x in chk['fail'] if ' step 1: ' in x]
    assert any('starting at the decision instant' in x for x in first), chk['fail'][:3]
    assert any('pair_free_time differs' in x for x in first), chk['fail'][:3]


def _legacy_ect(regime):
    p = P72.legacy_result_path('10x25', 'ECT', regime)
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


@pytest.mark.parametrize('regime', [
    pytest.param(r, marks=pytest.mark.skipif(
        not os.path.isfile(_legacy_ect(r)),
        reason=f'needs the stored ECT array of the prior implementation '
               f'{os.path.relpath(_legacy_ect(r), ROOT)}, which the release does not hold'))
    for r in ('N', 'SW')])
def test_reproduces_stored_exact_arrays(exact, regime):
    masks = P72.mask_options()
    P72._init_worker()
    spt = P72.stored('10x25', 'SPT', regime)
    ect_legacy = REG.read_path(P72.legacy_result_path('10x25', 'ECT', regime))
    for k in range(2):
        assert P72.rollout(_task(regime, 'SPT', k, masks))[4] == spt[k]
        assert P72.rollout(_task(regime, 'ECTND', k, masks))[4] == spt[k]
        assert P72.rollout(_task(regime, 'ECT', k, masks))[4] == ect_legacy[k]


def test_save_atomic(tmp_path):
    arr = np.stack([np.arange(5.0), np.linspace(0, 1, 5)], 1)
    a, b = tmp_path / 'Result_X_N.npy', tmp_path / 'ref.npy'
    P72.save_atomic(str(a), arr)
    np.save(str(b), arr)
    assert a.read_bytes() == b.read_bytes()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['Result_X_N.npy', 'ref.npy']


def test_parse_cores():
    assert P72.parse_cores('16-19,22') == [16, 17, 18, 19, 22]
    with pytest.raises(SystemExit):
        P72.parse_cores('3,3')


def test_exact_summary_refused_without_policy_files(exact, monkeypatch, tmp_path):
    res, out = tmp_path / 'res', tmp_path / 'out'
    d = res / '10x25_xmask'
    d.mkdir(parents=True)
    for rule in P72.NEW_RULES:
        for g in P72.REGIMES:
            np.save(str(d / f'Result_{rule}_{g}.npy'), np.full((4, 2), 100.0))
    monkeypatch.setenv('PACT_RESULTS', str(res))
    monkeypatch.setenv('PACT_OUT', str(out))
    assert P72.summarize(list(P72.NEW_RULES)) is False
    assert not any(p.is_file() for p in out.rglob('*'))
