"""Tests for the exact mask (first_mask='exact', mask_ties='exact').

  * at every decision of random rollouts, the first included, in all eight
    regimes on five test instances each, the mask equals the legal set of
    the non-delay rule computed independently on the true (hour) axis: a
    pure-Python state that follows reference_sim.simulate's transition, with
    processing start = push past outages of max(job ready, machine free) for
    setup + processing, plus the attached setup, decision instant = the
    minimum start over compatible pairs, legal = compatible and start equal
    to it. The legacy mask differs from the same reference on these rollouts
    (first decisions under setups or windows, and dropped ties);
  * a pair that ties the earliest processing start in hours, but sits a few
    ulps above it on the normalized axis, is dropped by the legacy comparison
    and admitted by the exact one, and when dispatched it starts at the
    decision instant;
  * every greedy rollout of the released policy (seed 301) on the same 40
    instances passes both independent validators, every dispatched pair
    starts processing at the decision instant, and the makespan equals the
    reference simulator's replay of the same actions;
  * the option route (constructor arguments) and the environment-variable
    route (PACT_FIRST_MASK=1, PACT_MASK_TOL=1) give identical states, masks
    and rewards at every step, for each correction alone and for both;
    explicit legacy options are bit-identical to the frozen published copy;
  * resolve_mask_option refuses a contradiction between an option and its
    environment variable, and eval_family.py refuses an exact-mask run
    without --tag and a contradictory command line before writing anything.

Run from the repository root:  python3 -m pytest tests/test_exact_mask.py -q
CPU only.
"""
import importlib.util
import json
import os
import subprocess
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

from common_utils import greedy_select_action, heuristic_select_action  # noqa: E402
from constraint_family import checks, reference_sim  # noqa: E402
from constraint_family.descriptor import (ConstraintDescriptor, IDLE_CLASS,  # noqa: E402
                                          compute_token)
from constraint_family.fjsp_env_family import (FJSPEnvFamily,  # noqa: E402
                                                mask_option_from_env,
                                                resolve_mask_option)
from ppvc_instance_generator import load_instance  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'fjsp_env_family_published',
    os.path.join(ROOT, 'tests', 'fixtures', 'fjsp_env_family_published.py'))
_pub = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pub)
PublishedEnv = _pub.FJSPEnvFamily

DATA = 'data/FAMILY/10x25'
REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
INSTANCES = (0, 1, 2, 3, 4)
MASK_VARS = ('PACT_FIRST_MASK', 'PACT_MASK_TOL')


@pytest.fixture(autouse=True)
def _no_mask_vars(monkeypatch):
    for v in MASK_VARS:
        monkeypatch.delenv(v, raising=False)


def load(regime, idx):
    with open(os.path.join(DATA, 'desc', regime,
                           f'instance_{idx:03d}.desc.json')) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(rec['base_instance'])
    return jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def make(jl, pt, meta, desc, cls=FJSPEnvFamily, **kw):
    env = cls(jl.shape[0], pt.shape[1], family_mode=True, **kw)
    env.delta_mode = 'min'
    env.reward_delta_env = True
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    return env


EXACT = dict(first_mask='exact', mask_ties='exact')


class NonDelayReference:
    """The non-delay rule on the true axis, with its own state: job ready
    times, machine free times and last classes, updated by the transition of
    reference_sim.simulate (attached setup, push past outages, lag after the
    operation). Shares nothing with the environment."""

    def __init__(self, jl, pt, desc):
        self.jl = np.asarray(jl, dtype=int)
        self.pt = np.asarray(pt, dtype=float)
        self.desc = desc
        self.J, self.M = len(self.jl), self.pt.shape[1]
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.next_op = self.first.copy()
        rel = getattr(desc, 'release', None)
        self.job_ready = (np.zeros(self.J) if rel is None
                          else np.asarray(rel, dtype=float).copy())
        self.mch_free = np.zeros(self.M)
        self.mch_last = np.full(self.M, IDLE_CLASS, dtype=int)

    def start(self, j, k):
        op = self.next_op[j]
        if op >= self.first[j] + self.jl[j] or self.pt[op, k] <= 0:
            return np.inf
        d = self.desc
        s = float(d.setup[self.mch_last[k], d.op_class[op]]) if d.has_setup else 0.0
        b = max(self.job_ready[j], self.mch_free[k])
        if d.has_window and d.windows is not None:
            b = reference_sim.push_past_outages_scalar(b, s + self.pt[op, k], d.windows[k])
        return b + s

    def table(self):
        st = np.array([[self.start(j, k) for k in range(self.M)]
                       for j in range(self.J)])
        t = st.min()
        return st, t, np.isfinite(st) & (st == t)

    def step(self, j, k):
        op = self.next_op[j]
        st = self.start(j, k)
        ct = st + self.pt[op, k]
        self.job_ready[j] = ct + (self.desc.lag[op] if self.desc.has_lag else 0.0)
        self.mch_free[k] = ct
        self.mch_last[k] = self.desc.op_class[op]
        self.next_op[j] += 1
        return st


def legal_of(env):
    return ~env.dynamic_pair_mask[0]


def test_exact_mask_equals_reference_nondelay_rule_at_every_decision():
    rng = np.random.default_rng(304)
    legacy_first, legacy_tie, n_dec = 0, 0, 0
    for regime in REGIMES:
        for idx in INSTANCES:
            jl, pt, meta, desc = load(regime, idx)
            M = pt.shape[1]
            ex = make(jl, pt, meta, desc, **EXACT)
            lg = make(jl, pt, meta, desc)
            ref = NonDelayReference(jl, pt, desc)
            step = 0
            while not ex.done().all():
                st, t, want = ref.table()
                got = legal_of(ex)
                assert np.array_equal(got, want), (regime, idx, step)
                assert np.array_equal(ex.state.dynamic_pair_mask_tensor.numpy(),
                                      ex.dynamic_pair_mask), (regime, idx, step)
                # the decision instant is the reference minimum (slope scale)
                assert abs(float(ex.next_schedule_time[0]) / ex._slope - t) < 1e-6
                # the legacy mask on the same state (dynamics do not read it)
                if not np.array_equal(legal_of(lg), want):
                    if step == 0:
                        legacy_first += 1
                    else:
                        legacy_tie += 1
                        # after the first decision legacy only loses ties
                        assert not (legal_of(lg) & ~want).any()
                n_dec += 1
                a = int(rng.choice(np.flatnonzero(want.reshape(-1))))
                j, k = divmod(a, M)
                op = int(ex.candidate[0, j])
                s_ref = ref.step(j, k)
                ex.step(np.array([a]))
                lg.step(np.array([a]))
                # the dispatched pair starts processing at the decision instant
                assert abs(s_ref - t) < 1e-9
                assert abs(float(ex.true_op_ct[0, op] - pt[op, k]) - t) < 1e-9
                step += 1
            assert np.array_equal(ex.current_makespan, lg.current_makespan)
    # the reference is sharp enough to see both legacy deviations
    assert legacy_first > 0 and legacy_tie > 0, (legacy_first, legacy_tie, n_dec)
    print(f'decisions {n_dec}: legacy mask differs from the non-delay rule at '
          f'{legacy_first} first decisions and {legacy_tie} later decisions')


def test_tied_pair_dropped_by_legacy_admitted_by_exact():
    rng = np.random.default_rng(7)
    found = 0
    for regime in REGIMES:
        for idx in INSTANCES[:2]:
            jl, pt, meta, desc = load(regime, idx)
            M = pt.shape[1]
            ex = make(jl, pt, meta, desc, mask_ties='exact')
            lg = make(jl, pt, meta, desc)
            ref = NonDelayReference(jl, pt, desc)
            prefix, dispatched = [], False
            while not ex.done().all():
                _, t, want = ref.table()
                extra = legal_of(ex) & ~legal_of(lg)
                for j, k in (map(int, p) for p in np.argwhere(extra)):
                    assert len(prefix) > 0      # the first mask is legacy in both
                    # a tie in hours, a few ulps above the instant when normalized
                    assert want[j, k] and abs(ref.start(j, k) - t) < 1e-9
                    gap = float(ex.pair_free_time[0, j, k] - ex.next_schedule_time[0])
                    assert 0.0 < gap < 1e-6 * ex._slope
                    found += 1
                    if dispatched:
                        continue
                    # dispatched from the same state, it starts at the instant
                    e2 = make(jl, pt, meta, desc, mask_ties='exact')
                    for a in prefix:
                        e2.step(np.array([a]))
                    op = int(e2.candidate[0, j])
                    e2.step(np.array([j * M + k]))
                    assert abs(float(e2.true_op_ct[0, op] - pt[op, k]) - t) < 1e-9
                    dispatched = True
                # step with a pair legal in both masks
                both = (legal_of(ex) & legal_of(lg)).reshape(-1)
                a = int(rng.choice(np.flatnonzero(both)))
                ref.step(*divmod(a, M))
                ex.step(np.array([a]))
                lg.step(np.array([a]))
                prefix.append(a)
    assert found > 0, 'no tied pair found on the sampled rollouts'
    print(f'tied pairs admitted by the exact comparison only: {found}')


ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode')


def test_released_policy_greedy_rollouts_validate_under_exact_mask():
    tag = '10x25+family+jointblind-v1'
    with open(f'train_log/FAMILY/config_{tag}.json') as f:
        snap = json.load(f)
    saved = {k: getattr(configs, k) for k in ARCH_KEYS if hasattr(configs, k)}
    n = 0
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
        for regime in REGIMES:
            for idx in INSTANCES:
                jl, pt, meta, desc = load(regime, idx)
                M = pt.shape[1]
                env = make(jl, pt, meta, desc, **EXACT)
                env.token_g = np.asarray([compute_token(
                    desc, float(pt[pt > 0].mean()), M)])
                env._attach_token()
                s = env.state
                ref = NonDelayReference(jl, pt, desc)
                amch = np.full(pt.shape[0], -1, dtype=int)
                actions = []
                while not env.done().all():
                    with torch.no_grad():
                        pi, _ = ppo.policy(
                            fea_j=s.fea_j_tensor, op_mask=s.op_mask_tensor,
                            candidate=s.candidate_tensor, fea_m=s.fea_m_tensor,
                            mch_mask=s.mch_mask_tensor, comp_idx=s.comp_idx_tensor,
                            dynamic_pair_mask=s.dynamic_pair_mask_tensor,
                            fea_pairs=s.fea_pairs_tensor, op_type=s.op_type_tensor,
                            mch_type=s.mch_type_tensor, token=s.token_tensor)
                    a = int(greedy_select_action(pi).numpy()[0])
                    j, k = divmod(a, M)
                    _, t, want = ref.table()
                    assert want[j, k], (regime, idx, len(actions))
                    assert abs(ref.step(j, k) - t) < 1e-9
                    amch[env.candidate[0, j]] = k
                    actions.append((j, k))
                    s, _, _ = env.step(np.array([a]))
                op_ct = env.true_op_ct[0]
                op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
                assert reference_sim.validate_schedule(jl, pt, desc, op_st, op_ct,
                                                       amch) == [], (regime, idx)
                c6 = checks.check_schedule_validation(desc, jl, pt, amch, op_ct)
                assert c6.status == checks.PASS, (regime, idx, c6.messages)
                sim = reference_sim.simulate(jl, pt, desc, actions)
                assert np.allclose(sim['op_ct'], op_ct, atol=1e-9)
                assert abs(sim['makespan'] - float(env.current_makespan[0])) < 1e-9
                n += 1
    finally:
        for k, v in saved.items():
            setattr(configs, k, v)
    assert n == len(REGIMES) * len(INSTANCES)


def snapshot(env):
    s = env.state
    return [s.fea_j_tensor.numpy(), s.fea_m_tensor.numpy(),
            s.fea_pairs_tensor.numpy(), s.dynamic_pair_mask_tensor.numpy(),
            s.comp_idx_tensor.numpy(), s.mch_mask_tensor.numpy(),
            s.candidate_tensor.numpy(), s.token_tensor.numpy(),
            env.dynamic_pair_mask, env.comp_idx, env.op_ct, env.true_op_ct,
            env.op_ct_lb, env.pair_free_time, env.candidate_free_time,
            env.true_candidate_free_time, np.asarray(env.next_schedule_time),
            env.mch_current_available_jc_nums]


def rollout(make_env, actions_from, monkeypatch, env_vars):
    """Per-step snapshots and rewards of one rollout, with `env_vars` set for
    its whole duration (the tie variable is read at every step)."""
    for v in MASK_VARS:
        monkeypatch.delenv(v, raising=False)
    for v in env_vars:
        monkeypatch.setenv(v, '1')
    env = make_env()
    out = [[np.copy(x) for x in snapshot(env)]]
    rewards, acts = [], []
    np.random.seed(50)
    rng = np.random.default_rng(11)
    t = 0
    while not env.done().all():
        if actions_from is None:
            a = (heuristic_select_action('SPT', env) if t % 2 else
                 int(rng.choice(np.flatnonzero(legal_of(env).reshape(-1)))))
        else:
            a = actions_from[t]
        acts.append(a)
        _, r, _ = env.step(np.array([a]))
        rewards.append(np.copy(r))
        out.append([np.copy(x) for x in snapshot(env)])
        t += 1
    # reset() restores the same step-0 state
    env.reset()
    out.append([np.copy(x) for x in snapshot(env)])
    for v in MASK_VARS:
        monkeypatch.delenv(v, raising=False)
    return out, rewards, acts, np.copy(env.current_makespan)


@pytest.mark.parametrize('opts,env_vars', [
    (dict(first_mask='exact', mask_ties='exact'), MASK_VARS),
    (dict(first_mask='exact'), ('PACT_FIRST_MASK',)),
    (dict(mask_ties='exact'), ('PACT_MASK_TOL',)),
])
def test_option_route_equals_environment_route(opts, env_vars, monkeypatch):
    for regime in ('S', 'W', 'SW', 'LSW', 'N'):
        for idx in INSTANCES[:2]:
            jl, pt, meta, desc = load(regime, idx)
            a_snap, a_rew, acts, _ = rollout(
                lambda: make(jl, pt, meta, desc, **opts), None, monkeypatch, ())
            b_snap, b_rew, _, _ = rollout(
                lambda: make(jl, pt, meta, desc), acts, monkeypatch, env_vars)
            assert len(a_snap) == len(b_snap)
            for t, (x, y) in enumerate(zip(a_snap, b_snap)):
                for c, (u, v) in enumerate(zip(x, y)):
                    assert u.shape == v.shape and np.array_equal(u, v), \
                        (regime, idx, opts, t, c)
            for u, v in zip(a_rew, b_rew):
                assert np.array_equal(u, v)


@pytest.mark.parametrize('regime', ('S', 'SW', 'N'))
def test_explicit_legacy_bit_identical_to_published_env(regime, monkeypatch):
    for idx in INSTANCES[:2]:
        jl, pt, meta, desc = load(regime, idx)
        a, ra, acts, ma = rollout(
            lambda: make(jl, pt, meta, desc, first_mask='legacy',
                         mask_ties='legacy'), None, monkeypatch, ())
        b, rb, _, mb = rollout(
            lambda: make(jl, pt, meta, desc, cls=PublishedEnv), acts,
            monkeypatch, ())
        for t, (x, y) in enumerate(zip(a, b)):
            for c, (u, v) in enumerate(zip(x, y)):
                assert u.shape == v.shape and np.array_equal(u, v), (regime, idx, t, c)
        for u, v in zip(ra, rb):
            assert np.array_equal(u, v)
        assert np.array_equal(ma, mb)


def test_resolve_mask_option(monkeypatch):
    for name, var in (('first_mask', 'PACT_FIRST_MASK'), ('mask_ties', 'PACT_MASK_TOL')):
        monkeypatch.delenv(var, raising=False)
        assert resolve_mask_option(name) == 'legacy'
        assert resolve_mask_option(name, 'exact') == 'exact'
        assert resolve_mask_option(name, 'legacy') == 'legacy'
        monkeypatch.setenv(var, '1')
        assert mask_option_from_env(name) == 'exact'
        assert resolve_mask_option(name) == 'exact'
        assert resolve_mask_option(name, 'exact') == 'exact'
        with pytest.raises(SystemExit):
            resolve_mask_option(name, 'legacy')
        monkeypatch.setenv(var, '0')
        assert resolve_mask_option(name) == 'legacy'
        with pytest.raises(SystemExit):
            resolve_mask_option(name, 'exact')
        monkeypatch.setenv(var, 'yes')
        with pytest.raises(ValueError):
            resolve_mask_option(name)
        monkeypatch.delenv(var, raising=False)
        with pytest.raises(ValueError):
            resolve_mask_option(name, 'tolerant')


def _eval(args, extra_env=None):
    env = {k: v for k, v in os.environ.items() if k not in MASK_VARS}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.update(extra_env or {})
    return subprocess.run([sys.executable, 'eval_family.py', '--regimes', 'S',
                           '--methods', 'SPT', '--limit', '1', '--device', 'cpu']
                          + args, cwd=ROOT, env=env, capture_output=True, text=True)


def test_eval_family_refuses_exact_mask_without_tag_or_with_contradiction():
    before = set(os.listdir('test_results/FAMILY')) if os.path.isdir('test_results/FAMILY') else set()
    r = _eval(['--first_mask', 'exact'])
    assert r.returncode != 0 and 'needs --tag' in (r.stdout + r.stderr), r.stdout + r.stderr
    r = _eval([], {'PACT_MASK_TOL': '1'})
    assert r.returncode != 0 and 'needs --tag' in (r.stdout + r.stderr), r.stdout + r.stderr
    r = _eval(['--first_mask', 'legacy', '--tag', '_unused'], {'PACT_FIRST_MASK': '1'})
    assert r.returncode != 0 and 'contradictory' in (r.stdout + r.stderr), r.stdout + r.stderr
    after = set(os.listdir('test_results/FAMILY')) if os.path.isdir('test_results/FAMILY') else set()
    assert after == before, 'a refused run wrote a result folder'
