"""Tests for the release-date rule (candidate fourth member of the family).

  * release=None is bit-identical to the environment as published
    (tests/fixtures/fjsp_env_family_published.py is a frozen copy of it):
    same observations, masks, rewards and schedules at every step, on
    instances of two regimes, under two action streams;
  * an all-zero release equals release=None (a job-side delay of zero);
  * with release dates the environment's schedule equals the reference
    simulator's, every first operation starts at or after its release date,
    both validators accept it, the potential stays a lower bound of the
    realized makespan at every step and is exact at completion;
  * the validators reject a first operation pulled before its release date;
  * the GA's batched decoder applies release dates as the reference
    simulator does, and is unchanged without them;
  * descriptor records without the field are byte-identical.

Run from the repository root:  python3 -m pytest tests/test_release.py -q
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

# params parses sys.argv at import time; hide pytest's arguments and force CPU
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
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from constraint_family import reference_sim  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'fjsp_env_family_published',
    os.path.join(ROOT, 'tests', 'fixtures', 'fjsp_env_family_published.py'))
_pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pub)
PublishedEnv = _pub.FJSPEnvFamily

DATA = 'data/FAMILY/10x25'
REGRESSION_REGIMES = ('LS', 'SW')
INSTANCES = (0, 1, 2)


def load(regime, idx):
    with open(os.path.join(DATA, 'desc', regime,
                           f'instance_{idx:03d}.desc.json')) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(rec['base_instance'])
    return jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def make(cls, jl, pt, meta, desc):
    env = cls(jl.shape[0], pt.shape[1], family_mode=True)
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
    """Roll env_a and env_b with the same action stream; assert every step's
    observation, mask, reward and schedule are bit-identical."""
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


@pytest.mark.parametrize('regime', REGRESSION_REGIMES)
def test_release_none_bit_identical_to_published_env(regime):
    for idx in INSTANCES:
        jl, pt, meta, desc = load(regime, idx)
        assert getattr(desc, 'release', None) is None
        for name in ('SPT', 'MWKR', 'random'):
            new = make(FJSPEnvFamily, jl, pt, meta, desc)
            old = make(PublishedEnv, jl, pt, meta, desc)
            if name == 'random':
                rng = np.random.default_rng(idx)
                chooser = lambda e: random_legal(e, rng)  # noqa: E731
            else:
                np.random.seed(50)
                chooser = lambda e, n=name: heuristic_select_action(n, e)  # noqa: E731
            lockstep(new, old, chooser, (regime, idx, name))


def with_release(desc, rel):
    d = copy.deepcopy(desc)
    d.release = np.asarray(rel, dtype=float)
    return d


@pytest.mark.parametrize('regime', ('N', 'LSW'))
def test_zero_release_equals_none(regime):
    jl, pt, meta, desc = load(regime, 0)
    zero = with_release(desc, np.zeros(jl.shape[0]))
    rng = np.random.default_rng(3)
    lockstep(make(FJSPEnvFamily, jl, pt, meta, zero),
             make(FJSPEnvFamily, jl, pt, meta, desc),
             lambda e: random_legal(e, rng), (regime, 'zero'))


def rollout_with_audit(env, chooser):
    """Roll out; return actions, per-op machine and the bound trajectory."""
    J, M = env.number_of_jobs, env.number_of_machines
    amch = np.full(env.number_of_ops, -1, dtype=int)
    actions, bounds, rewards = [], [float(np.max(env.op_ct_lb[0]))], []
    while not env.done().all():
        a = chooser(env)
        j, m = a // M, a % M
        amch[env.candidate[0, j]] = m
        actions.append((j, m))
        _, r, _ = env.step(np.array([a]))
        rewards.append(float(r[0]))
        bounds.append(float(np.max(env.op_ct_lb[0])))
    return actions, amch, np.array(bounds), np.array(rewards)


@pytest.mark.parametrize('regime', ('N', 'L', 'SW', 'LSW'))
def test_release_schedule_matches_reference_and_validates(regime):
    jl, pt, meta, desc = load(regime, 1)
    rng = np.random.default_rng(11)
    rel = rng.integers(0, desc.horizon // 4 + 1, jl.shape[0])
    d = with_release(desc, rel)
    env = make(FJSPEnvFamily, jl, pt, meta, d)
    # first decision: only jobs released at min(rho) are admissible
    ok_jobs = np.flatnonzero((~env.dynamic_pair_mask[0]).any(axis=1))
    assert set(ok_jobs) == set(np.flatnonzero(rel == rel.min()))
    arng = np.random.default_rng(5)
    actions, amch, bounds, rewards = rollout_with_audit(
        env, lambda e: random_legal(e, arng))
    ref = reference_sim.simulate(jl, pt, d, actions)
    assert np.allclose(env.true_op_ct[0], ref['op_ct'])
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    assert np.all(ref['op_st'][first] >= rel - 1e-9)
    op_st = env.true_op_ct[0] - pt[np.arange(pt.shape[0]), amch]
    assert reference_sim.validate_schedule(jl, pt, d, op_st, env.true_op_ct[0],
                                           amch) == []
    r = schedule_validator.validate_schedule(
        jl, pt, d.lag if d.has_lag else np.zeros(pt.shape[0]), amch,
        env.true_op_ct[0], release=rel)
    assert r['feasible'], r['violations']
    # bound audit on the normalized axis
    realized = float(np.max(env.op_ct[0]))
    assert np.all(bounds <= realized + 1e-9)
    assert abs(bounds[-1] - realized) < 1e-9
    assert np.all(rewards <= 1e-12)
    # normalized and true axes agree
    assert abs(realized / env._slope - env.current_makespan[0]) < 1e-6


def test_validators_reject_start_before_release():
    jl, pt, meta, desc = load('LSW', 2)
    rel = np.zeros(jl.shape[0])
    d0 = with_release(desc, rel)
    env = make(FJSPEnvFamily, jl, pt, meta, d0)
    arng = np.random.default_rng(1)
    _, amch, _, _ = rollout_with_audit(env, lambda e: random_legal(e, arng))
    ct = env.true_op_ct[0]
    op_st = ct - pt[np.arange(pt.shape[0]), amch]
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    j = int(np.argmax(op_st[first]))            # a job whose first op starts late
    assert op_st[first[j]] > 0
    bad = rel.copy()
    bad[j] = op_st[first[j]] + 1                 # release after its start
    d1 = with_release(desc, bad)
    v = reference_sim.validate_schedule(jl, pt, d1, op_st, ct, amch)
    assert any('release' in s for s in v)
    r = schedule_validator.validate_schedule(
        jl, pt, desc.lag, amch, ct, release=bad)
    assert not r['feasible'] and any(s.startswith('[release]')
                                     for s in r['violations'])
    # and the same schedule passes with the dates it respects
    assert reference_sim.validate_schedule(jl, pt, d0, op_st, ct, amch) == []


def _ga_module():
    # eval_ga_family hides the command line from params at import time
    _saved = sys.argv
    try:
        import eval_ga_family
    finally:
        sys.argv = _saved
    return eval_ga_family


@pytest.mark.parametrize('regime', ('N', 'L', 'SW', 'LSW'))
def test_ga_decoder_applies_release_dates(regime):
    ga = _ga_module()
    jl, pt, meta, desc = load(regime, 1)
    rng = np.random.default_rng(13)
    rel = rng.integers(0, desc.horizon // 4 + 1, jl.shape[0])
    d = with_release(desc, rel)
    js = ga._job_starts(jl)
    base = np.concatenate([np.full(int(L), j) for j, L in enumerate(jl)])
    compat = ga._compat(pt)
    os_pop = np.array([rng.permutation(base) for _ in range(8)])
    ma_pop = np.array([[c[rng.integers(len(c))] for c in compat] for _ in range(8)])
    ms, amch, op_ct = ga.decode_population(jl, pt, d, meta['op_type'],
                                           meta['mch_type'], os_pop, ma_pop)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    for p in range(len(os_pop)):
        nxt, acts = js.copy(), []
        for j in os_pop[p]:
            acts.append((int(j), int(ma_pop[p, nxt[j]])))
            nxt[j] += 1
        ref = reference_sim.simulate(jl, pt, d, acts)
        assert abs(ms[p] - ref['makespan']) < 1e-6
        assert np.allclose(op_ct[p], ref['op_ct'], atol=1e-6)
        assert np.all(ref['op_st'][first] >= rel - 1e-9)
        op_st = op_ct[p] - pt[np.arange(pt.shape[0]), amch[p]]
        assert reference_sim.validate_schedule(jl, pt, d, op_st, op_ct[p],
                                               amch[p]) == []
    # without release dates the decoder is unchanged: the same chromosomes
    # decode to the reference simulator's schedule under the stored record
    ms0, _, _ = ga.decode_population(jl, pt, desc, meta['op_type'],
                                     meta['mch_type'], os_pop, ma_pop)
    for p in range(len(os_pop)):
        nxt, acts = js.copy(), []
        for j in os_pop[p]:
            acts.append((int(j), int(ma_pop[p, nxt[j]])))
            nxt[j] += 1
        assert abs(ms0[p] - reference_sim.simulate(jl, pt, desc, acts)['makespan']) < 1e-6


def test_descriptor_json_unchanged_without_release():
    for regime in ('N', 'LSW'):
        path = os.path.join(DATA, 'desc', regime, 'instance_000.desc.json')
        rec = json.load(open(path))
        desc = ConstraintDescriptor.from_json(rec)
        out = desc.to_json()
        assert 'release' not in out
        core = {k: rec[k] for k in out}
        assert json.dumps(out, sort_keys=True) == json.dumps(core, sort_keys=True)
    d = with_release(desc, [0, 5, 7, 0, 1, 2, 3, 4, 5, 6])
    back = ConstraintDescriptor.from_json(json.loads(json.dumps(d.to_json())))
    assert back.release.tolist() == [0, 5, 7, 0, 1, 2, 3, 4, 5, 6]
