"""Tests for PACT_FIRST_MASK (first decision restricted to the defined mask).

  * flag unset: the environment is bit-identical to the frozen published copy
    (tests/fixtures/fjsp_env_family_published.py) at every step, on a regime
    with setups, one with setups and outages, and one without either, under
    two action streams;
  * flag set, regimes with setups: every pair the step-0 mask leaves legal
    starts processing at the minimum processing start over compatible pairs,
    computed independently on the true axis from the descriptor (setup from
    the idle class, push past outages by the reference simulator's scalar
    rule); the first pair dispatched by the policy-free rules and by a random
    legal choice therefore starts at that minimum; the step-0 decision
    instant equals it; and on these regimes the flag removes pairs;
  * flag set: reset() restores the same step-0 tables and mask.

Run from the repository root:  python3 -m pytest tests/test_first_mask.py -q
CPU only.
"""
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

from common_utils import heuristic_select_action  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from constraint_family.reference_sim import push_past_outages_scalar  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'fjsp_env_family_published',
    os.path.join(ROOT, 'tests', 'fixtures', 'fjsp_env_family_published.py'))
_pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pub)
PublishedEnv = _pub.FJSPEnvFamily

DATA = 'data/FAMILY/10x25'
INSTANCES = (0, 1, 2, 3, 4)


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
            env.dynamic_pair_mask, env.op_ct, env.true_op_ct, env.op_ct_lb,
            env.pair_free_time, env.candidate_free_time,
            env.true_candidate_free_time, np.asarray(env.next_schedule_time)]


def assert_identical(a, b, where):
    for k, (x, y) in enumerate(zip(snapshot(a), snapshot(b))):
        assert x.shape == y.shape and np.array_equal(x, y), (where, k)


def random_legal(env, rng):
    legal = np.flatnonzero(~env.dynamic_pair_mask[0].reshape(-1))
    return int(rng.choice(legal))


def true_min_start(jl, pt, desc):
    """Minimum processing start over compatible first-operation pairs at t=0,
    on the true axis, from the descriptor alone."""
    firsts = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    best = np.inf
    for op in firsts:
        s = float(desc.setup[IDLE_CLASS, desc.op_class[op]]) if desc.has_setup else 0.0
        for k in np.nonzero(pt[op] > 0)[0]:
            spans = desc.windows[k] if (desc.has_window and desc.windows is not None) else []
            best = min(best, push_past_outages_scalar(0.0, s + float(pt[op, k]), spans) + s)
    return best


def first_start(env, a, pt):
    """Step `a` and return the dispatched pair's processing start (true hours)."""
    job, m = a // pt.shape[1], a % pt.shape[1]
    op = int(env.candidate[0, job])
    env.step(np.array([a]))
    return float(env.true_op_ct[0, op] - env.true_op_pt[0, op, m])


@pytest.mark.parametrize('regime', ('S', 'SW', 'N'))
def test_flag_unset_bit_identical_to_published_env(regime, monkeypatch):
    monkeypatch.delenv('PACT_FIRST_MASK', raising=False)
    monkeypatch.delenv('PACT_MASK_TOL', raising=False)
    for idx in INSTANCES[:3]:
        jl, pt, meta, desc = load(regime, idx)
        for name in ('SPT', 'random'):
            new = make(FJSPEnvFamily, jl, pt, meta, desc)
            old = make(PublishedEnv, jl, pt, meta, desc)
            if name == 'random':
                rng = np.random.default_rng(idx)
                chooser = lambda e: random_legal(e, rng)  # noqa: E731
            else:
                np.random.seed(50)
                chooser = lambda e: heuristic_select_action('SPT', e)  # noqa: E731
            assert_identical(new, old, (regime, idx, name, 'init'))
            t = 0
            while not new.done().all():
                a = chooser(new)
                _, ra, _ = new.step(np.array([a]))
                _, rb, _ = old.step(np.array([a]))
                assert np.array_equal(ra, rb), (regime, idx, name, t)
                assert_identical(new, old, (regime, idx, name, t))
                t += 1
            assert np.array_equal(new.current_makespan, old.current_makespan)


@pytest.mark.parametrize('regime', ('S', 'LS', 'SW', 'LSW'))
def test_flag_on_first_dispatch_starts_at_minimum(regime, monkeypatch):
    monkeypatch.setenv('PACT_FIRST_MASK', '1')
    monkeypatch.delenv('PACT_MASK_TOL', raising=False)
    removed = 0
    for idx in INSTANCES:
        jl, pt, meta, desc = load(regime, idx)
        best = true_min_start(jl, pt, desc)
        env = make(FJSPEnvFamily, jl, pt, meta, desc)
        legal = np.flatnonzero(~env.dynamic_pair_mask[0].reshape(-1))
        compat = np.flatnonzero(~env.candidate_process_relation[0].reshape(-1))
        assert legal.size >= 1 and set(legal) <= set(compat)
        removed += compat.size - legal.size
        # the state the policy reads carries the same mask
        assert np.array_equal(env.state.dynamic_pair_mask_tensor.numpy(),
                              env.dynamic_pair_mask)
        # the decision instant is the minimum (normalized axis, slope scale)
        assert abs(float(env.next_schedule_time[0]) / env._slope - best) < 1e-6
        # every legal pair starts processing at the minimum
        for a in legal:
            e = make(FJSPEnvFamily, jl, pt, meta, desc)
            assert abs(first_start(e, int(a), pt) - best) < 1e-6, (regime, idx, a)
        # and so does the first pair SPT dispatches (SPT reads the mask; FIFO,
        # MOR and MWKR pick a job first and fall back to static compatibility
        # when that job has no legal machine, at this and every later step)
        e = make(FJSPEnvFamily, jl, pt, meta, desc)
        np.random.seed(50)
        a = heuristic_select_action('SPT', e)
        assert abs(first_start(e, a, pt) - best) < 1e-6, (regime, idx, 'SPT')
    assert removed > 0, f'the flag removed no pair on {regime}'


ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode')


def test_flag_on_released_policy_first_dispatch_at_minimum(monkeypatch):
    """The released policy (seed 301), greedy, on the setup regime S: with the
    flag its first dispatched pair starts processing at the minimum."""
    from common_utils import greedy_select_action
    from constraint_family.descriptor import compute_token
    monkeypatch.setenv('PACT_FIRST_MASK', '1')
    monkeypatch.delenv('PACT_MASK_TOL', raising=False)
    tag = '10x25+family+jointblind-v1'
    with open(f'train_log/FAMILY/config_{tag}.json') as f:
        snap = json.load(f)
    saved = {k: getattr(configs, k) for k in ARCH_KEYS if hasattr(configs, k)}
    try:
        for k in ARCH_KEYS:
            if k in snap:
                v = snap[k]
                if isinstance(v, str) and v in ('True', 'False'):
                    v = v == 'True'
                setattr(configs, k, v)
        from model.PPO import PPO_initialize
        ppo = PPO_initialize()
        ppo.policy.load_state_dict(torch.load(
            f'trained_network/FAMILY/{tag}.pth', map_location='cpu'))
        ppo.policy.eval()
        for idx in INSTANCES:
            jl, pt, meta, desc = load('S', idx)
            env = make(FJSPEnvFamily, jl, pt, meta, desc)
            env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()),
                                                    pt.shape[1])])
            env._attach_token()
            s = env.state
            with torch.no_grad():
                pi, _ = ppo.policy(
                    fea_j=s.fea_j_tensor, op_mask=s.op_mask_tensor,
                    candidate=s.candidate_tensor, fea_m=s.fea_m_tensor,
                    mch_mask=s.mch_mask_tensor, comp_idx=s.comp_idx_tensor,
                    dynamic_pair_mask=s.dynamic_pair_mask_tensor,
                    fea_pairs=s.fea_pairs_tensor, op_type=s.op_type_tensor,
                    mch_type=s.mch_type_tensor, token=s.token_tensor)
            a = int(greedy_select_action(pi).numpy()[0])
            assert abs(first_start(env, a, pt) - true_min_start(jl, pt, desc)) < 1e-6, idx
    finally:
        for k, v in saved.items():
            setattr(configs, k, v)


def test_flag_on_reset_restores_step0(monkeypatch):
    monkeypatch.setenv('PACT_FIRST_MASK', '1')
    jl, pt, meta, desc = load('SW', 0)
    env = make(FJSPEnvFamily, jl, pt, meta, desc)
    ref = [np.copy(env.dynamic_pair_mask), np.copy(env.pair_free_time),
           np.copy(env.next_schedule_time), np.copy(env.comp_idx)]
    np.random.seed(50)
    while not env.done().all():
        env.step(np.array([heuristic_select_action('SPT', env)]))
    env.reset()
    for x, y in zip(ref, [env.dynamic_pair_mask, env.pair_free_time,
                          env.next_schedule_time, env.comp_idx]):
        assert np.array_equal(x, y)
    assert np.array_equal(env.state.dynamic_pair_mask_tensor.numpy(), ref[0])
