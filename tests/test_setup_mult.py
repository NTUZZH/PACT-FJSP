"""Tests for the per-machine setup multiplier (descriptor field `setup_mult`).

  * without the field the environment is bit-identical to the frozen copy
    taken before the field existed (tests/fixtures/fjsp_env_family_pre_setupmult.py):
    same observations, masks, rewards, bound and schedules at every step, on
    stored test instances of five regimes, under SPT, MWKR and random actions,
    with the legacy and the exact masks;
  * an all-ones multiplier equals the absent field at every step;
  * the reference simulator and the validator are unchanged without the field
    (frozen copy tests/fixtures/reference_sim_pre_setupmult.py);
  * with a 0/1 multiplier (fuzz over generated instances, descriptors, subsets
    and random action sequences): the environment's completion times equal the
    reference simulator's; both validators accept the schedule; the makespan
    estimate never exceeds the realized makespan, never decreases, and equals
    it at the end; the machine channel reporting the cheapest setup into the
    next operation equals a brute-force recomputation;
  * the validator rejects a schedule that omits a charged setup;
  * JSON round trip, and records without the field are byte-identical;
  * check C1 rejects malformed multipliers, C4 warns on switched-off machines.

Run from the repository root:  python3 -m pytest tests/test_setup_mult.py -q
CPU only.
"""
import copy
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402
    import data_utils  # noqa: E402,F401
finally:
    sys.argv = _argv
configs.device = 'cpu'

import torch  # noqa: E402

import fjsp_env_same_op_nums  # noqa: E402
fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')

import schedule_validator  # noqa: E402
from common_utils import heuristic_select_action  # noqa: E402
from constraint_family import checks, reference_sim  # noqa: E402
from constraint_family.descriptor import (ConstraintDescriptor,  # noqa: E402
                                          make_descriptor)
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from ppvc_instance_generator import (DEFAULT_FACTORY, load_instance,  # noqa: E402
                                     ppvc_instance_generator)


def _load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FrozenEnv = _load_module('fjsp_env_family_pre_setupmult',
                         'tests/fixtures/fjsp_env_family_pre_setupmult.py').FJSPEnvFamily
frozen_ref = _load_module('reference_sim_pre_setupmult',
                          'tests/fixtures/reference_sim_pre_setupmult.py')

DATA = 'data/FAMILY/10x25'


def load(regime, idx):
    with open(os.path.join(DATA, 'desc', regime,
                           f'instance_{idx:03d}.desc.json')) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(rec['base_instance'])
    return jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def make(cls, jl, pt, meta, desc, **kw):
    env = cls(jl.shape[0], pt.shape[1], family_mode=True, **kw)
    env.delta_mode = 'min'
    env.reward_delta_env = True
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    return env


def snapshot(env):
    s = env.state
    return [s.fea_j_tensor.numpy(), s.fea_m_tensor.numpy(),
            s.fea_pairs_tensor.numpy(), s.dynamic_pair_mask_tensor.numpy(),
            s.comp_idx_tensor.numpy(), s.mch_mask_tensor.numpy(),
            s.candidate_tensor.numpy(), s.token_tensor.numpy(),
            env.op_ct, env.true_op_ct, env.op_ct_lb, env.pair_free_time,
            env.candidate_free_time, env.true_candidate_free_time,
            np.asarray(env.next_schedule_time)]


def assert_identical(a, b, where):
    for k, (x, y) in enumerate(zip(snapshot(a), snapshot(b))):
        assert x.shape == y.shape and np.array_equal(x, y), (where, k)


def random_legal(env, rng):
    legal = np.flatnonzero(~env.dynamic_pair_mask[0].reshape(-1))
    return int(rng.choice(legal))


def lockstep(env_a, env_b, chooser, where):
    assert_identical(env_a, env_b, (where, 'init'))
    t = 0
    while not env_a.done().all():
        a = chooser(env_a)
        _, ra, _ = env_a.step(np.array([a]))
        _, rb, _ = env_b.step(np.array([a]))
        assert np.array_equal(ra, rb), (where, t)
        assert_identical(env_a, env_b, (where, t))
        t += 1
    assert env_b.done().all()
    assert np.array_equal(env_a.current_makespan, env_b.current_makespan)


def choosers(idx):
    rng = np.random.default_rng(idx)
    yield 'random', (lambda e: random_legal(e, rng))
    for name in ('SPT', 'MWKR'):
        np.random.seed(50)
        yield name, (lambda e, n=name: heuristic_select_action(n, e))


def with_mult(desc, u):
    d = copy.deepcopy(desc)
    d.setup_mult = None if u is None else np.asarray(u, dtype=float)
    return d


# --------------------------------------------------------------------------
# the default path is unchanged
# --------------------------------------------------------------------------

@pytest.mark.parametrize('regime', ('N', 'S', 'LS', 'SW', 'LSW'))
def test_absent_field_bit_identical_to_frozen_env(regime):
    for idx in (0, 1):
        jl, pt, meta, desc = load(regime, idx)
        assert getattr(desc, 'setup_mult', None) is None
        for name, chooser in choosers(idx):
            lockstep(make(FJSPEnvFamily, jl, pt, meta, desc),
                     make(FrozenEnv, jl, pt, meta, desc),
                     chooser, (regime, idx, name))


@pytest.mark.parametrize('regime', ('LS', 'LSW'))
def test_absent_field_bit_identical_exact_masks(regime):
    jl, pt, meta, desc = load(regime, 2)
    kw = dict(first_mask='exact', mask_ties='exact')
    rng = np.random.default_rng(2)
    lockstep(make(FJSPEnvFamily, jl, pt, meta, desc, **kw),
             make(FrozenEnv, jl, pt, meta, desc, **kw),
             lambda e: random_legal(e, rng), (regime, 'exact'))


@pytest.mark.parametrize('regime', ('S', 'LSW'))
def test_all_ones_equals_absent(regime):
    jl, pt, meta, desc = load(regime, 3)
    ones = with_mult(desc, np.ones(pt.shape[1]))
    for name, chooser in choosers(3):
        lockstep(make(FJSPEnvFamily, jl, pt, meta, ones),
                 make(FJSPEnvFamily, jl, pt, meta, desc),
                 chooser, (regime, 'ones', name))


def test_reference_sim_unchanged_without_field():
    for regime in ('S', 'LSW', 'W'):
        jl, pt, meta, desc = load(regime, 4)
        env = make(FJSPEnvFamily, jl, pt, meta, desc)
        rng = np.random.default_rng(9)
        actions, amch = [], np.full(pt.shape[0], -1, dtype=int)
        M = pt.shape[1]
        while not env.done().all():
            a = random_legal(env, rng)
            j, m = a // M, a % M
            amch[env.candidate[0, j]] = m
            actions.append((j, m))
            env.step(np.array([a]))
        new = reference_sim.simulate(jl, pt, desc, actions)
        old = frozen_ref.simulate(jl, pt, desc, actions)
        for k in ('op_st', 'op_ct', 'op_mch'):
            assert np.array_equal(new[k], old[k])
        ct = env.true_op_ct[0]
        st = ct - pt[np.arange(pt.shape[0]), amch]
        assert (reference_sim.validate_schedule(jl, pt, desc, st, ct, amch)
                == frozen_ref.validate_schedule(jl, pt, desc, st, ct, amch) == [])
        # a corrupted schedule gets the same messages from both
        st2, ct2 = st.copy(), ct.copy()
        st2[5] -= 1
        ct2[5] -= 1
        assert (reference_sim.validate_schedule(jl, pt, desc, st2, ct2, amch)
                == frozen_ref.validate_schedule(jl, pt, desc, st2, ct2, amch))


# --------------------------------------------------------------------------
# with the field: environment = reference, bound admissible and exact
# --------------------------------------------------------------------------

def _fuzz_case(k):
    rng = np.random.default_rng(1000 + k)
    n_mod = int(rng.integers(2, 5))
    jl, pt, meta = ppvc_instance_generator(n_modules=n_mod, class_mix='mixed',
                                           station_counts=DEFAULT_FACTORY,
                                           seed=7000 + k)
    regime = ('S', 'LS', 'SW', 'LSW')[k % 4]
    desc = make_descriptor(regime, jl, pt, meta, pt.shape[1], seed=31000 + k)
    M = pt.shape[1]
    kind = k % 3
    if kind == 0:
        u = (rng.random(M) < 0.5).astype(float)       # random subset
    elif kind == 1:
        u = np.zeros(M)                               # setups nowhere
    else:
        u = np.ones(M)
        u[rng.choice(M, size=int(rng.integers(1, 4)), replace=False)] = 0
    return jl, pt, meta, with_mult(desc, u), u


def _min_next_setup_bruteforce(env, desc, u):
    """[M] cheapest setup from each machine's last class into a compatible
    candidate, times u_k, on the normalized axis (0 when none)."""
    M = env.number_of_machines
    out = np.zeros(M)
    for m in range(M):
        best = np.inf
        for j in range(env.number_of_jobs):
            if env.candidate_process_relation[0, j, m]:
                continue
            op = env.candidate[0, j]
            s = desc.setup[env.mch_last_class[0, m], desc.op_class[op]]
            best = min(best, u[m] * s * env._slope)
        out[m] = 0.0 if not np.isfinite(best) else best
    return out


@pytest.mark.parametrize('k', range(24))
def test_fuzz_env_matches_reference_and_bound_is_exact(k):
    jl, pt, meta, d, u = _fuzz_case(k)
    env = make(FJSPEnvFamily, jl, pt, meta, d)
    M = pt.shape[1]
    rng = np.random.default_rng(k)
    actions, amch = [], np.full(pt.shape[0], -1, dtype=int)
    bounds = [float(np.max(env.op_ct_lb[0]))]
    rewards = []
    raw_ch = None
    while not env.done().all():
        # machine channel before normalization equals the brute force
        raw_ch = env._mch_min_next_setup()[0]
        assert np.allclose(raw_ch, _min_next_setup_bruteforce(env, d, u),
                           atol=1e-12), (k, len(actions))
        a = random_legal(env, rng)
        j, m = a // M, a % M
        amch[env.candidate[0, j]] = m
        actions.append((j, m))
        _, r, _ = env.step(np.array([a]))
        rewards.append(float(r[0]))
        bounds.append(float(np.max(env.op_ct_lb[0])))
    ref = reference_sim.simulate(jl, pt, d, actions)
    assert np.allclose(env.true_op_ct[0], ref['op_ct']), k
    assert abs(env.current_makespan[0] - ref['makespan']) < 1e-9
    ct = env.true_op_ct[0]
    st = ct - pt[np.arange(pt.shape[0]), amch]
    assert reference_sim.validate_schedule(jl, pt, d, st, ct, amch) == []
    c6 = checks.check_schedule_validation(d, jl, pt, amch, ct, op_st=st)
    assert c6.status == checks.PASS, c6.messages
    realized = float(np.max(env.op_ct[0]))
    bounds = np.array(bounds)
    assert np.all(bounds <= realized + 1e-9), k
    assert np.all(np.diff(bounds) >= -1e-9), k
    assert abs(bounds[-1] - realized) < 1e-9, k
    assert np.all(np.array(rewards) <= 1e-12), k
    assert abs(realized / env._slope - env.current_makespan[0]) < 1e-6


@pytest.mark.parametrize('k', range(6))
def test_fuzz_rules_match_reference(k):
    """Dispatching rules read the setup-aware tables; their schedules must be
    reproduced by the reference simulator as well."""
    jl, pt, meta, d, u = _fuzz_case(100 + k)
    for name in ('SPT', 'MWKR', 'FIFO', 'MOR'):
        env = make(FJSPEnvFamily, jl, pt, meta, d)
        np.random.seed(50)
        M = pt.shape[1]
        actions = []
        while not env.done().all():
            a = heuristic_select_action(name, env)
            actions.append((a // M, a % M))
            env.step(np.array([a]))
        ref = reference_sim.simulate(jl, pt, d, actions)
        assert np.allclose(env.true_op_ct[0], ref['op_ct']), (k, name)


def test_bound_term_inside_minimum():
    """With setups switched off on every machine the setup term vanishes; with
    a subset it is the machine-wise minimum, never above op_min_pt + delta."""
    jl, pt, meta, desc = load('LS', 5)
    M = pt.shape[1]
    e_full = make(FJSPEnvFamily, jl, pt, meta, desc)
    e_zero = make(FJSPEnvFamily, jl, pt, meta, with_mult(desc, np.zeros(M)))
    e_plain = make(FJSPEnvFamily, jl, pt, meta, ConstraintDescriptor(
        active=(True, False, False), lag=desc.lag, op_class=desc.op_class,
        horizon=desc.horizon))
    assert np.array_equal(e_zero._bound_min_term(), e_zero.op_min_pt)
    assert np.allclose(e_zero.op_ct_lb, e_plain.op_ct_lb)
    u = np.ones(M)
    u[:7] = 0
    e_sub = make(FJSPEnvFamily, jl, pt, meta, with_mult(desc, u))
    t_sub, t_full = e_sub._bound_min_term(), e_full._bound_min_term()
    assert np.all(t_sub <= t_full + 1e-12)
    assert np.all(t_sub >= e_sub.op_min_pt - 1e-12)
    # brute force of the inner minimum on the normalized axis
    smin = desc.setup.min(axis=0) * e_sub._slope                # [C]
    for op in range(pt.shape[0]):
        ks = np.flatnonzero(pt[op] > 0)
        want = min(e_sub.op_pt.data[0, op, k] + u[k] * smin[desc.op_class[op]]
                   for k in ks)
        assert abs(t_sub[0, op] - want) < 1e-12


# --------------------------------------------------------------------------
# validator, schema, serialization
# --------------------------------------------------------------------------

def _schedule(jl, pt, meta, d, seed):
    env = make(FJSPEnvFamily, jl, pt, meta, d)
    rng = np.random.default_rng(seed)
    M = pt.shape[1]
    amch = np.full(pt.shape[0], -1, dtype=int)
    while not env.done().all():
        a = random_legal(env, rng)
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    ct = env.true_op_ct[0].copy()
    return ct - pt[np.arange(pt.shape[0]), amch], ct, amch


def test_validator_rejects_omitted_setup():
    jl, pt, meta, desc = load('S', 6)
    M = pt.shape[1]
    # setups nowhere: some operation starts at 0, where every machine owes a
    # cold-start setup under the full model
    d0 = with_mult(desc, np.zeros(M))
    st, ct, amch = _schedule(jl, pt, meta, d0, 1)
    assert reference_sim.validate_schedule(jl, pt, d0, st, ct, amch) == []
    v = reference_sim.validate_schedule(jl, pt, desc, st, ct, amch)
    assert v and any('setup' in s for s in v)
    assert checks.check_schedule_validation(desc, jl, pt, amch, ct,
                                            op_st=st).status == checks.FAIL
    # a subset: every violation found under the full model sits on a machine
    # whose setups the schedule did not pay
    u = np.ones(M)
    u[[0, 1, 2, 7, 8]] = 0
    du = with_mult(desc, u)
    st, ct, amch = _schedule(jl, pt, meta, du, 2)
    assert reference_sim.validate_schedule(jl, pt, du, st, ct, amch) == []
    v = reference_sim.validate_schedule(jl, pt, desc, st, ct, amch)
    assert v
    mch = {int(s.split()[1].rstrip(':')) for s in v if s.startswith('mch ')}
    assert mch and mch <= {0, 1, 2, 7, 8}


def test_json_round_trip_and_unchanged_records():
    for regime in ('N', 'LSW'):
        path = os.path.join(DATA, 'desc', regime, 'instance_000.desc.json')
        rec = json.load(open(path))
        desc = ConstraintDescriptor.from_json(rec)
        out = desc.to_json()
        assert 'setup_mult' not in out
        core = {k: rec[k] for k in out}
        assert json.dumps(out, sort_keys=True) == json.dumps(core, sort_keys=True)
    u = [1, 0] * 12 + [1]
    d = with_mult(desc, u)
    back = ConstraintDescriptor.from_json(json.loads(json.dumps(d.to_json())))
    assert back.setup_mult.tolist() == u
    assert checks.descriptor_fingerprint(back) == checks.descriptor_fingerprint(d)
    assert checks.descriptor_fingerprint(d) != checks.descriptor_fingerprint(desc)


def test_c1_and_c4_on_setup_mult():
    jl, pt, meta, desc = load('LS', 0)
    M = pt.shape[1]
    ok = with_mult(desc, np.r_[np.zeros(4), np.ones(M - 4)])
    c1 = checks.check_schema_and_values(ok, pt, jl)
    assert c1.status == checks.PASS and c1.details['setup_machines_charged'] == M - 4
    assert any('charged on' in s for s in
               checks.check_magnitude_range(ok, pt).messages)
    for bad in (np.full(M, 0.5), np.ones(M - 1), np.ones((M, 1))):
        assert checks.check_schema_and_values(with_mult(desc, bad), pt,
                                              jl).status == checks.FAIL
    _, _, _, dn = load('L', 0)
    assert checks.check_schema_and_values(with_mult(dn, np.ones(M)), pt,
                                          jl).status == checks.FAIL
    # absent field: C4 output unchanged by the new code
    assert not any('charged on' in s for s in
                   checks.check_magnitude_range(desc, pt).messages)
