"""The makespan estimate at the empty schedule against the shortest schedules found
(scripts/p113_instance_bound.py).

  * synthetic records of both searches and a synthetic estimate: the per-combination and
    pooled means, smallest and largest distances are those computed by hand, a makespan that
    equals the estimate counts at distance zero, and a makespan below the estimate stops the
    script (the CP-SAT record, the memetic record and the policy schedule alike);
  * the record layouts of the two searches, and the macro names (letters only, one prefix,
    neither of the two solver-bound names, no clash with another macro file under manuscript/);
  * the environment's estimate on a stored instance of the combination N equals the longest
    job chain of shortest processing times (no lag, no setup);
  * on the stored records (when present): the check exits 0.

Run from the repository root:  python -m pytest tests/test_instance_bound.py -q
CPU only; every file is written under a temporary folder.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import p113_instance_bound as P  # noqa: E402

R8 = P.R8
PY = sys.executable
NAMES = ['instance_000', 'instance_001']

needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25/desc')),
                                reason='needs the descriptors of data/FAMILY/10x25')
needs_records = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'test_results/FAMILY/10x25_cparms/evidence_rule600s')),
    reason='needs the 600 s records at 10x25')


def cell(arm, combo, idx, makespan, state='valid'):
    rec = (f'or_solution/FAMILY/10x25_memetic600s/{combo}/instance_{idx:03d}_seed1.json'
           if arm == 'memetic_rule' else
           f'test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/{combo}/instance_{idx:03d}.json')
    return dict(state=state, final_makespan=float(makespan), record=rec)


def synthetic(monkeypatch, mem, cps, pol, est=100.0):
    """Both instances of every combination: estimate est, memetic mem[idx], CP-SAT cps[idx],
    policy pol[idx]."""
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    recs = {(r, i): {'memetic_rule': cell('memetic_rule', r, i, mem[i]),
                     'cpsat_rule': cell('cpsat_rule', r, i, cps[i])}
            for r in R8 for i in range(len(NAMES))}
    monkeypatch.setattr(P, 'instance_names', lambda: list(NAMES))
    monkeypatch.setattr(P, 'records', lambda: (recs, 'lns', 304, {}))
    monkeypatch.setattr(P, 'run_config', lambda seed: {'first_mask': 'exact',
                                                       'mask_ties': 'exact'})
    monkeypatch.setattr(P, 'policy_arrays', lambda seed, n: {r: np.asarray(pol, float) for r in R8})
    monkeypatch.setattr(P, 'estimate', lambda name, combo, cfg: est)


def test_distances_computed_by_hand(monkeypatch):
    # instance 0: shortest found min(130, 120) = 120 -> 20 %; policy 150 -> 50 %
    # instance 1: shortest found min(100, 110) = 100 -> 0 % (at the estimate); policy 104 -> 4 %
    synthetic(monkeypatch, mem=[130, 100], cps=[120, 110], pol=[150, 104])
    res = P.analyse()
    for r in R8 + ['All']:
        x = res['per'][r]
        assert x['found']['mean'] == pytest.approx(10.0)
        assert (x['found']['min'], x['found']['max']) == (0.0, pytest.approx(20.0))
        assert x['policy']['mean'] == pytest.approx(27.0)
        assert x['policy']['min'] == pytest.approx(4.0) and x['policy']['max'] == pytest.approx(50.0)
    assert res['per']['All']['found']['n'] == 16 and res['per']['N']['found']['n'] == 2
    assert res['per']['All']['at_estimate'] == 8 and res['per']['W']['at_estimate'] == 1
    assert res['delta_mode'] == 'min' and res['cpsat_arm'] == 'lns' and res['policy_seed'] == 304


@pytest.mark.parametrize('which', ['memetic', 'cpsat', 'policy'])
def test_a_makespan_below_the_estimate_stops_the_script(monkeypatch, which):
    mem, cps, pol = [130, 100], [120, 110], [150, 104]
    {'memetic': mem, 'cpsat': cps, 'policy': pol}[which][1] = 99.0
    synthetic(monkeypatch, mem=mem, cps=cps, pol=pol)
    with pytest.raises(P.Refused, match='below the estimate'):
        P.analyse()


def test_rounding_of_the_estimate_is_not_a_violation(monkeypatch):
    synthetic(monkeypatch, mem=[100, 100], cps=[100, 100], pol=[100, 100], est=100.0000001)
    res = P.analyse()
    assert res['per']['All']['found']['max'] == 0.0 and res['per']['All']['at_estimate'] == 16


def test_record_layouts():
    import re
    assert re.match(P.LAYOUT['memetic_rule'], cell('memetic_rule', 'LSW', 7, 1)['record'])
    assert re.match(P.LAYOUT['cpsat_rule'], cell('cpsat_rule', 'LSW', 7, 1)['record'])
    assert not re.match(P.LAYOUT['cpsat_rule'],
                        'test_results/FAMILY/10x25_cparms/evidence_rule60s/lns/N/instance_000.json')
    assert not re.match(P.LAYOUT['memetic_rule'],
                        'or_solution/FAMILY/10x25_memetic60s/N/instance_000_seed1.json')


def test_macro_names(monkeypatch):
    synthetic(monkeypatch, mem=[130, 100], cps=[120, 110], pol=[150, 104])
    res = P.analyse()
    res['setting'] = 'synthetic'
    lines = P.macro_lines(res)
    names = P.check_names(lines)
    assert len(names) == 9 * 8 and all(n.startswith('Ibound') for n in names)
    assert not set(names) & set(P.NOT_WRITTEN)
    assert '\\newcommand{\\IboundFoundAll}{10.0}' in '\n'.join(lines)
    monkeypatch.setattr(P, 'defined_macros', lambda: {'IboundFoundAll': 'manuscript/x.tex'})
    with pytest.raises(P.Refused, match='already defined'):
        P.check_names(lines)


@needs_data
def test_estimate_on_a_stored_instance_of_n():
    import json
    from ppvc_instance_generator import load_instance
    rec = json.load(open(os.path.join(ROOT, 'data/FAMILY/10x25/desc/N/instance_000.desc.json')))
    jl, pt, _ = load_instance(os.path.join(ROOT, rec['base_instance']))
    pt = np.asarray(pt, dtype=float)
    shortest = np.where(pt > 0, pt, np.inf).min(axis=1)
    first, hand = 0, 0.0
    for n in np.asarray(jl, dtype=int):
        hand = max(hand, float(shortest[first:first + n].sum()))
        first += n
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        got = P.estimate('instance_000', 'N', {'first_mask': 'exact', 'mask_ties': 'exact'})
    finally:
        os.chdir(cwd)
    assert got == pytest.approx(hand, abs=1e-5)


@needs_records
def test_check_on_the_stored_records(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=str(tmp_path / 'out'))
    r = subprocess.run([PY, '-B', 'scripts/p113_instance_bound.py', '--check'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and 'inputs present' in r.stdout, r.stdout + r.stderr
