"""Tests for constraint_family/checks.py (configuration checks C1-C7).

Run from the repository root:  python3 -m pytest tests/test_checks.py -q
CPU only; reads the published 10x25 test set (data/FAMILY/10x25) and the
shifted-setup arm of the generalization study (data/FAMILY/10x25_oodsetup,
written by scripts/p29_ood.py with cross=(1.00, 2.00), symmetric=False).
"""
import copy
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # descriptor records store base_instance relative to the root

from constraint_family import checks  # noqa: E402
from constraint_family.checks import (FAIL, MANUAL, PASS, WARN,  # noqa: E402
                                      run_all)
from constraint_family.descriptor import (REGIMES_ALL,  # noqa: E402
                                          ConstraintDescriptor)
from constraint_family.descriptor import IDLE_CLASS  # noqa: E402
from constraint_family.reference_sim import simulate  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

# load_instance imports data_utils, which imports params, which parses
# sys.argv at import time; hide pytest's arguments from it (as p29_ood.py does).
_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    import data_utils  # noqa: E402,F401
finally:
    sys.argv = _argv

PUBLISHED = 'data/FAMILY/10x25'
SHIFTED_SETUP = 'data/FAMILY/10x25_oodsetup'
SETUP_REGIMES = tuple(r for r in REGIMES_ALL if 'S' in r)

_BASE_CACHE = {}


def load_record(root, regime, idx):
    path = os.path.join(root, 'desc', regime, f'instance_{idx:03d}.desc.json')
    with open(path) as f:
        rec = json.load(f)
    return rec, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def base(rec):
    stem = rec['base_instance']
    if stem not in _BASE_CACHE:
        _BASE_CACHE[stem] = load_instance(stem)
    return _BASE_CACHE[stem]


def records(root, regime):
    paths = sorted(glob.glob(os.path.join(root, 'desc', regime, '*.desc.json')))
    for p in paths:
        with open(p) as f:
            rec = json.load(f)
        yield rec, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def statuses(report):
    return {r.id: r.status for r in report}


@pytest.fixture(scope='module')
def lsw0():
    rec, desc = load_record(PUBLISHED, 'LSW', 0)
    jl, pt, meta = base(rec)
    return rec, desc, jl, pt


# ---------------------------------------------------------------- published

def test_published_lsw0_passes_c1_to_c5(lsw0):
    _, desc, _, pt = lsw0
    rep = run_all(desc, pt)
    assert [r.id for r in rep] == ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    st = statuses(rep)
    for cid in ('C1', 'C2', 'C3', 'C4', 'C5'):
        assert st[cid] == PASS, checks.format_report(rep)
    assert st['C6'] == MANUAL and st['C7'] == MANUAL
    assert rep[0].details['setup_all_positive'] is True


def test_run_all_is_deterministic(lsw0):
    _, desc, _, pt = lsw0
    a = [r.as_row() for r in run_all(desc, pt)]
    b = [r.as_row() for r in run_all(desc, pt)]
    assert a == b


def test_training_ranges_derived_from_generator():
    rng = checks.training_ranges()
    assert rng['setup_cross_frac_pbar'] == (0.50, 1.00)
    assert rng['setup_same_frac_pbar'] == (0.25, 0.50)
    assert rng['setup_cold_frac_pbar'] == (0.25, 0.50)
    assert rng['setup_symmetric'] is True
    assert rng['window_density_frac_H'] == (0.05, 0.15)
    assert rng['window_duration_h'] == (2, 8)
    assert rng['lag_nonzero_h'] == ((12, 24), (24, 48))


# ---------------------------------------------------------------- C1

def test_c1_malformed_windows_fail(lsw0):
    _, desc, _, pt = lsw0
    cases = {
        'overlap': lambda w: w[0].append((w[0][-1][0], w[0][-1][1] + 1)),
        'unsorted': lambda w: w[1].reverse() if len(w[1]) > 1 else w[1].insert(0, (500, 501)),
        'empty interval': lambda w: w[2].append((900, 900)),
        'non-integer': lambda w: w[3].append((900.5, 902)),
        'wrong machine count': lambda w: w.pop(),
    }
    for label, mutate in cases.items():
        d = copy.deepcopy(desc)
        d.windows = [list(s) for s in d.windows]
        mutate(d.windows)
        r = checks.check_schema_and_values(d, pt)
        assert r.status == FAIL, label


def test_c1_bit_parameter_consistency(lsw0):
    _, desc, _, pt = lsw0
    d = copy.deepcopy(desc)
    d.setup = None                                   # SETUP on, no matrix
    assert checks.check_schema_and_values(d, pt).status == FAIL
    d = copy.deepcopy(desc)
    d.active = (False, True, True)                   # LAG off, lags nonzero
    assert checks.check_schema_and_values(d, pt).status == FAIL
    d = copy.deepcopy(desc)
    d.setup = np.zeros((6, 4))                       # wrong shape
    assert checks.check_schema_and_values(d, pt).status == FAIL
    d = copy.deepcopy(desc)
    d.lag = d.lag.copy(); d.lag[0] = -1              # negative lag
    assert checks.check_schema_and_values(d, pt).status == FAIL
    d = copy.deepcopy(desc)
    d.op_class = d.op_class.copy(); d.op_class[0] = 4   # class out of range
    assert checks.check_schema_and_values(d, pt).status == FAIL
    # off bits with None / all-zero parameters are fine (regime N)
    rec, n = load_record(PUBLISHED, 'N', 0)
    assert checks.check_schema_and_values(n, base(rec)[1]).status == PASS


def test_c1_reports_zero_setup_entries(lsw0):
    _, desc, _, pt = lsw0
    d = copy.deepcopy(desc)
    d.setup = d.setup.copy(); d.setup[0, 0] = 0
    r = checks.check_schema_and_values(d, pt)
    assert r.status == PASS and r.details['setup_all_positive'] is False


# ---------------------------------------------------------------- C2

def test_c2_detached_and_hybrid_fail(lsw0):
    _, desc, _, pt = lsw0
    for conv in ('detached', 'hybrid'):
        d = copy.deepcopy(desc)
        d.setup_convention = conv
        r = checks.check_setup_convention(d)
        assert r.status == FAIL and 'Remark 1' in r.messages[0]
        assert statuses(run_all(d, pt))['C2'] == FAIL


CORE_KEYS = ('active', 'lag', 'op_class', 'setup', 'windows', 'horizon')


def test_setup_convention_stated_by_the_benchmark_loader(lsw0):
    rec, desc, _, _ = lsw0
    assert 'setup_convention' not in rec           # the stored file is unchanged
    assert desc.setup_convention == 'attached' and desc.setup_convention_declared
    out = desc.to_json()
    assert out['setup_convention'] == 'attached'   # written whenever SETUP is active
    for k in CORE_KEYS:
        assert json.dumps(out[k]) == json.dumps(rec[k])
    back = ConstraintDescriptor.from_json(json.loads(json.dumps(out)))
    assert back.setup_convention_declared
    assert checks.check_setup_convention(back).status == PASS
    d = copy.deepcopy(desc)
    d.setup_convention = 'detached'
    back = ConstraintDescriptor.from_json(json.loads(json.dumps(d.to_json())))
    assert back.setup_convention == 'detached'


def test_c2_record_with_setups_must_state_its_convention(lsw0):
    rec, _, _, pt = lsw0
    plant = ConstraintDescriptor.from_json(rec)    # read as a plant record
    assert plant.setup_convention == 'attached'    # what the environment applies
    assert not plant.setup_convention_declared
    r = checks.check_setup_convention(plant)
    assert r.status == FAIL and 'does not state its setup convention' in r.messages[0]
    assert r.details['declared'] is False
    assert statuses(run_all(plant, pt))['C2'] == FAIL
    # an undeclared record writes back unchanged, so it stays undeclared
    out = plant.to_json()
    assert 'setup_convention' not in out
    assert not ConstraintDescriptor.from_json(out).setup_convention_declared
    # stating the convention in the record itself passes
    stated = dict(rec, setup_convention='attached')
    assert checks.check_setup_convention(
        ConstraintDescriptor.from_json(stated)).status == PASS
    # a loader statement that contradicts the record is an error
    with pytest.raises(ValueError):
        ConstraintDescriptor.from_json(dict(rec, setup_convention='detached'),
                                       setup_convention='attached')


def test_c2_record_without_setups_needs_no_convention():
    for regime in ('N', 'L', 'W', 'LW'):
        rec, _ = load_record(PUBLISHED, regime, 0)
        d = ConstraintDescriptor.from_json(rec)
        assert d.setup_convention_declared
        assert checks.check_setup_convention(d).status == PASS
        assert 'setup_convention' not in d.to_json()


def test_c2_generated_descriptor_is_declared(lsw0):
    from constraint_family.descriptor import make_descriptor
    rec, _, jl, pt = lsw0
    meta = base(rec)[2]
    for regime in REGIMES_ALL:
        d = make_descriptor(regime, jl, pt, meta, pt.shape[1], seed=rec['desc_seed'])
        assert checks.check_setup_convention(d).status == PASS
        out = d.to_json()
        assert ('setup_convention' in out) == ('S' in regime)
        back = ConstraintDescriptor.from_json(json.loads(json.dumps(out)))
        assert checks.check_setup_convention(back).status == PASS


@pytest.mark.parametrize('root', [PUBLISHED] + [
    pytest.param(r, marks=needs_public_instances())
    for r in sorted(glob.glob('data/PUBLIC/family/pub_*'))])
def test_c2_stored_benchmark_records_pass_through_their_loaders(root):
    from constraint_family.public_rules import load_family_record
    n = 0
    for regime in REGIMES_ALL:
        for p in sorted(glob.glob(os.path.join(root, 'desc', regime, '*.desc.json'))):
            if root == PUBLISHED:
                with open(p) as f:
                    d = ConstraintDescriptor.from_json(json.load(f),
                                                       setup_convention='attached')
            else:
                d = load_family_record(p)[2]
            assert checks.check_setup_convention(d).status == PASS, p
            n += 1
    assert n >= 8 * 10, (root, n)


def test_convention_declaration_does_not_reach_the_dynamics(lsw0, monkeypatch):
    """The environment, the validator and the CP-SAT model never read the
    convention: an undeclared record and the same record stated attached give
    the same rollout, the same validation and the same model."""
    import torch
    import fjsp_env_same_op_nums
    monkeypatch.setattr(fjsp_env_same_op_nums.EnvState, 'device',
                        torch.device('cpu'))
    from constraint_family import cpsat_arms
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from common_utils import heuristic_select_action
    from constraint_family.reference_sim import validate_schedule
    rec, stated, jl, pt = lsw0
    meta = base(rec)[2]
    plant = ConstraintDescriptor.from_json(rec)
    runs = []
    for d in (stated, plant):
        env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
        env.set_initial_data([jl], [pt], descriptor_list=[d],
                             op_type_list=[meta['op_type']],
                             mch_type_list=[meta['mch_type']])
        amch = np.full(pt.shape[0], -1, dtype=int)
        feats = []
        np.random.seed(0)
        while not env.done().all():
            a = heuristic_select_action('SPT', env)
            amch[env.candidate[0, a // pt.shape[1]]] = a % pt.shape[1]
            env.step(np.array([a]))
            feats.append(env.fea_m.copy())
        ct = env.true_op_ct[0].copy()
        st = ct - pt[np.arange(pt.shape[0]), amch]
        runs.append((ct, np.stack(feats), validate_schedule(jl, pt, d, st, ct, amch)))
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1])
    assert runs[0][2] == runs[1][2] == []
    protos = [cpsat_arms.build_reference_model(
        cpsat_arms.Case(jl, pt, d))['model'].Proto().SerializeToString()
        for d in (stated, plant)]
    assert protos[0] == protos[1]


# ---------------------------------------------------------------- C3

def test_c3_unregistered_rule_fails(lsw0):
    _, desc, _, pt = lsw0
    r = checks.check_family_membership(desc, declared_rules=['BLOCKING'])
    assert r.status == FAIL and 'BLOCKING' in r.messages[0]
    assert statuses(run_all(desc, pt, declared_rules=['BLOCKING']))['C3'] == FAIL
    # an active bit whose rule type is missing from the registry also fails
    reg = {k: v for k, v in checks.RULE_REGISTRY.items() if k != 'WINDOW'}
    assert checks.check_family_membership(desc, registry=reg).status == FAIL


def test_c3_register_rule():
    reg = dict(checks.RULE_REGISTRY)
    checks.register_rule('CURING_OVEN', 'machine', 'trivial',
                         'Def. 1(ii) proof in note X', registry=reg)
    assert reg['CURING_OVEN'].hook == 'machine'
    assert 'CURING_OVEN' not in checks.RULE_REGISTRY   # global untouched
    with pytest.raises(ValueError):
        checks.register_rule('LAG', 'job', 'trivial', 'dup', registry=reg)
    with pytest.raises(ValueError):
        checks.register_rule('X', 'both', 'trivial', 'p', registry=reg)
    with pytest.raises(ValueError):
        checks.register_rule('Y', 'job', 'exact', 'p', registry=reg)
    with pytest.raises(ValueError):
        checks.register_rule('Z', 'job', 'trivial', '', registry=reg)
    rec, desc = load_record(PUBLISHED, 'L', 0)
    assert checks.check_family_membership(
        desc, declared_rules=['CURING_OVEN'], registry=reg).status == PASS


# ---------------------------------------------------------------- C4

def _c4_counts(root, regimes):
    out = {}
    for regime in regimes:
        n = flagged = 0
        for rec, desc in records(root, regime):
            _, pt, _ = base(rec)
            n += 1
            flagged += checks.check_magnitude_range(desc, pt).status == WARN
        out[regime] = (flagged, n)
    return out


def test_c4_flags_every_shifted_setup_descriptor():
    shifted = _c4_counts(SHIFTED_SETUP, SETUP_REGIMES)
    print('\nC4 shifted-setup (flagged, total):', shifted)
    for regime, (flagged, n) in shifted.items():
        assert n == 100 and flagged == 100, (regime, flagged, n)
    # the non-setup regimes of the shifted arm are unchanged and stay PASS
    other = _c4_counts(SHIFTED_SETUP, [r for r in REGIMES_ALL if 'S' not in r])
    assert all(f == 0 and n == 100 for f, n in other.values()), other


def test_c4_passes_every_published_descriptor():
    published = _c4_counts(PUBLISHED, REGIMES_ALL)
    print('\nC4 published (flagged, total):', published)
    assert all(f == 0 and n == 100 for f, n in published.values()), published


def test_c4_reports_setups_below_the_range_as_it_reports_them_above(lsw0):
    """An entry below the lower edge of its training range is reported in the
    same form as one above the upper edge. With p_bar of about 4.4 h the
    cross-class range is [2.2, 4.4] h, so a 1 h changeover is below it."""
    _, desc, _, pt = lsw0
    pbar = float(pt[pt > 0].mean())
    lo, hi = checks.SETUP_CROSS_RANGE
    assert lo * pbar - checks.ROUND_TOL_H > 1.0       # the case the clamp hid
    a, b = 0, 1                                       # one cross-class pair
    out = {}
    for label, v in (('below', 1.0), ('above', np.ceil(hi * pbar + 1))):
        d = copy.deepcopy(desc)
        d.setup = d.setup.copy()
        d.setup[a, b] = d.setup[b, a] = v
        r = checks.check_magnitude_range(d, pt)
        assert r.status == WARN, (label, r.messages)
        out[label] = [m for m in r.messages if 'cross-class entries' in m]
        assert len(out[label]) == 1 and out[label][0].startswith(
            'SETUP: 2 of 12 cross-class entries outside'), (label, r.messages)
    # the unchanged record stays inside the range
    assert checks.check_magnitude_range(desc, pt).status == PASS


def test_c4_flags_setups_drawn_at_half_scale():
    """Setups drawn at half the training scale (the sensitivity study's
    setup-scale level 0.5) on the 100 published S instances: the per-entry
    lower edges flag all but the few records whose entries all stay inside
    the training ranges after rounding to integer hours."""
    from constraint_family.descriptor import make_descriptor
    flagged = n = 0
    for rec, _ in records(PUBLISHED, 'S'):
        jl, pt, meta = base(rec)
        d = make_descriptor('S', jl, pt, meta, pt.shape[1], seed=rec['desc_seed'],
                            shift={'same': (0.125, 0.25), 'cross': (0.25, 0.50)})
        flagged += checks.check_magnitude_range(d, pt).status == WARN
        n += 1
    assert n == 100 and flagged >= 90, (flagged, n)


def test_c4_flags_scaled_down_lags(lsw0):
    """Lag edges carry no clamp, so a lag scaled down below its range is
    reported as one scaled up above it."""
    _, desc, _, pt = lsw0
    for s in (0.5, 2.0):
        d = copy.deepcopy(desc)
        d.lag = np.floor(d.lag * s + 0.5)
        r = checks.check_magnitude_range(d, pt)
        assert r.status == WARN and any(m.startswith('LAG') for m in r.messages), s


def test_c4_warns_but_never_fails(lsw0):
    _, desc, _, pt = lsw0
    d = copy.deepcopy(desc)
    d.lag = d.lag.copy(); d.lag[d.lag.argmax()] = 200
    r = checks.check_magnitude_range(d, pt)
    assert r.status == WARN and any(m.startswith('LAG') for m in r.messages)
    d = copy.deepcopy(desc)
    d.windows = [list(w) + [(1000, 1100)] for w in d.windows]
    r = checks.check_magnitude_range(d, pt)
    assert r.status == WARN and any(m.startswith('WINDOW') for m in r.messages)


# ---------------------------------------------------------------- C5

def test_c5_blocking_shop_fails(lsw0):
    _, desc, _, pt = lsw0
    assert statuses(run_all(desc, pt, finite_downstream_buffers=True))['C5'] == FAIL
    assert checks.check_applicability(None).status == FAIL
    assert checks.check_applicability(False).status == PASS


# ---------------------------------------------------------------- C6 / C7

def _serial_schedule(jl, pt, desc):
    """Round-robin over jobs, fastest compatible machine: a feasible schedule
    produced by the reference simulator."""
    jl = np.asarray(jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    nxt = [0] * len(jl)
    actions = []
    while len(actions) < pt.shape[0]:
        for j in range(len(jl)):
            if nxt[j] < jl[j]:
                row = pt[first[j] + nxt[j]]
                m = int(np.argmin(np.where(row > 0, row, np.inf)))
                actions.append((j, m))
                nxt[j] += 1
    return simulate(jl, pt, desc, actions)


def test_c6_valid_and_corrupted_schedule(lsw0):
    _, desc, jl, pt = lsw0
    res = _serial_schedule(jl, pt, desc)
    sched = {'op_mch': res['op_mch'], 'op_ct': res['op_ct'], 'op_st': res['op_st']}
    r = checks.check_schedule_validation(desc, jl, pt, **sched)
    assert r.status == PASS, r.messages
    assert statuses(run_all(desc, pt, schedule=sched, job_length=jl))['C6'] == PASS
    # pull one successor op into its predecessor's lag: must fail
    j1 = int(np.flatnonzero(desc.lag > 0)[0]) + 1
    bad = {k: np.array(v, dtype=float) for k, v in sched.items()}
    bad['op_mch'] = sched['op_mch']
    shift = desc.lag[j1 - 1]
    bad['op_st'][j1] -= shift
    bad['op_ct'][j1] -= shift
    r = checks.check_schedule_validation(desc, jl, pt, **bad)
    assert r.status == FAIL and r.details['violations']


def test_c6_rejects_setup_before_job_is_present(lsw0):
    """Attached convention: the changeover may start only once the job is
    ready. A schedule whose processing start is legal but whose setup block
    begins before the predecessor's completion plus lag must fail C6."""
    _, desc, jl, pt = lsw0
    res = _serial_schedule(jl, pt, desc)
    st, ct, mch = (np.array(res[k], dtype=float) for k in ('op_st', 'op_ct', 'op_mch'))
    mch = mch.astype(int)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    is_first = np.zeros(len(st), dtype=bool)
    is_first[first] = True
    found = 0
    for m in range(pt.shape[1]):
        ops = sorted(np.flatnonzero(mch == m), key=lambda o: st[o])
        last, prev_ct = IDLE_CLASS, 0.0
        for o in ops:
            s_dur = float(desc.setup[last, desc.op_class[o]])
            block = st[o] - s_dur
            if not is_first[o]:
                need = ct[o - 1] + desc.lag[o - 1]
                if s_dur >= 1 and block == need and prev_ct <= block - 1:
                    bad_st, bad_ct = st.copy(), ct.copy()
                    bad_st[o] -= 1
                    bad_ct[o] -= 1
                    assert bad_st[o] >= need       # processing start still legal
                    r = checks.check_schedule_validation(
                        desc, jl, pt, op_mch=mch, op_ct=bad_ct, op_st=bad_st)
                    assert r.status == FAIL
                    assert any('setup of op' in v for v in r.details['violations'])
                    found += 1
            last, prev_ct = desc.op_class[o], ct[o]
    assert found > 0, 'no job-bound operation with a setup in the fixture schedule'


def test_c7_records_and_is_manual(lsw0):
    _, desc, _, pt = lsw0
    r = checks.check_source_reconciliation(desc, 'v1.0', ['MES-42', 'CAL-7'])
    assert r.status == MANUAL
    assert r.details['descriptor_version'] == 'v1.0'
    assert r.details['source_records'] == ['CAL-7', 'MES-42']
    assert r.details['descriptor_sha256'] == checks.descriptor_fingerprint(desc)


# ---------------------------------------------------------------- release dates

def _with_release(desc, rel):
    d = copy.deepcopy(desc)
    d.release = np.asarray(rel, dtype=float)
    return d


def test_c1_release_dates(lsw0):
    _, desc, jl, pt = lsw0
    J = len(jl)
    good = _with_release(desc, np.arange(J))
    r = checks.check_schema_and_values(good, pt, jl)
    assert r.status == PASS and any('release' in m for m in r.messages)
    assert statuses(run_all(good, pt, job_length=jl))['C1'] == PASS
    cases = {
        'negative': np.r_[-1, np.zeros(J - 1)],
        'non-integer': np.r_[0.5, np.zeros(J - 1)],
        'wrong length': np.zeros(J + 1),
        'not finite': np.r_[np.inf, np.zeros(J - 1)],
        '2-D': np.zeros((J, 1)),
    }
    for label, rel in cases.items():
        r = checks.check_schema_and_values(_with_release(desc, rel), pt, jl)
        assert r.status == FAIL, label
    # without release the published record is unchanged by the new argument
    assert checks.check_schema_and_values(desc, pt, jl).status == PASS


def test_c3_release_is_a_registered_rule(lsw0):
    _, desc, jl, pt = lsw0
    assert 'RELEASE' not in checks.active_rule_types(desc)
    d = _with_release(desc, np.zeros(len(jl)))
    assert checks.active_rule_types(d)[-1] == 'RELEASE'
    r = checks.check_family_membership(d)
    assert r.status == PASS and checks.RULE_REGISTRY['RELEASE'].envelope == 'trivial'
    # a release date delays the job (its first operation), not a machine
    assert checks.RULE_REGISTRY['RELEASE'].hook == 'job'
    assert 'RELEASE: hook=job' in ' '.join(r.messages)
    reg = {k: v for k, v in checks.RULE_REGISTRY.items() if k != 'RELEASE'}
    assert checks.check_family_membership(d, registry=reg).status == FAIL


def test_c6_rejects_start_before_release(lsw0):
    _, desc, jl, pt = lsw0
    rel = np.zeros(len(jl))
    res = _serial_schedule(jl, pt, _with_release(desc, rel))
    sched = {'op_mch': res['op_mch'], 'op_ct': res['op_ct'], 'op_st': res['op_st']}
    assert checks.check_schedule_validation(
        _with_release(desc, rel), jl, pt, **sched).status == PASS
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    late = rel.copy()
    late[0] = res['op_st'][first[0]] + 1          # job 0 released after it starts
    r = checks.check_schedule_validation(_with_release(desc, late), jl, pt, **sched)
    assert r.status == FAIL
    assert any('[release]' in v for v in r.details['violations'])
    assert any('before release' in v for v in r.details['violations'])
    # the simulator honours release dates: re-simulated, the schedule passes
    res2 = _serial_schedule(jl, pt, _with_release(desc, late))
    assert res2['op_st'][first[0]] >= late[0]
    sched2 = {'op_mch': res2['op_mch'], 'op_ct': res2['op_ct'], 'op_st': res2['op_st']}
    assert checks.check_schedule_validation(
        _with_release(desc, late), jl, pt, **sched2).status == PASS
