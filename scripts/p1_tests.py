"""Unit + regression tests for the constraint family (this paper).

  T1  handcrafted SETUP case (hand-computed schedule)
  T2  handcrafted WINDOW case (hand-computed schedule)
  T3  fuzz: FJSPEnvFamily true dynamics == reference simulator, all regimes
  T4  reward soundness: telescoped bound admissible + tight at termination
  T5  E0 anchor: legacy family env, LAG-only == the released lag-only
      policy's eval (bit-exact)

Run:  python scripts/p1_tests.py [--skip-e0]
"""
import os
import sys

# params.py parses sys.argv at import time (project-wide config singleton),
# so our private flag must be consumed BEFORE any project import.
SKIP_E0 = '--skip-e0' in sys.argv
sys.argv = [a for a in sys.argv if a != '--skip-e0']

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from constraint_family.descriptor import (ConstraintDescriptor, make_descriptor,
                                          REGIMES_ALL, IDLE_CLASS)
from constraint_family.fjsp_env_family import FJSPEnvFamily
from constraint_family.reference_sim import simulate, validate_schedule


def _tiny_instance():
    """2 jobs x 2 ops, 2 machines; op3 incompatible with m1 (keeps the
    pt_lower_bound == 0 slope-normalization invariant of PPVC instances)."""
    job_length = np.array([2, 2])
    op_pt = np.array([[4, 6],
                      [3, 5],
                      [5, 4],
                      [2, 0]], dtype=float)
    op_class = np.array([0, 0, 1, 1])
    return job_length, op_pt, op_class


def t1_setup_handcrafted():
    jl, pt, op_class = _tiny_instance()
    setup = np.zeros((5, 4))
    setup[:, :] = 2.0          # every changeover (incl. idle row) = 2h
    setup[0, 0] = 1.0          # class0->class0 = 1h
    desc = ConstraintDescriptor(active=(False, True, False),
                                lag=np.zeros(4), op_class=op_class,
                                setup=setup, horizon=100)
    # actions: (job0 op0 -> m0), (job1 op0 -> m0), (job0 op1 -> m1), (job1 op1 -> m0)
    acts = [(0, 0), (1, 0), (0, 1), (1, 0)]
    r = simulate(jl, pt, desc, acts)
    # hand computation (ATTACHED setups: st = max(ready, free) + s):
    # op0 j0 m0: st = max(0,0)+S[idle,0]=2, ct = 6
    # op2 j1 m0: st = max(0,6)+S[0,1]=2 = 8, ct = 8+5 = 13
    # op1 j0 m1: st = max(6,0)+S[idle,0]=2 = 8, ct = 8+5 = 13
    # op3 j1 m0: st = max(13,13)+S[1,1]=2 = 15, ct = 17
    exp_ct = np.array([6, 13, 13, 17], dtype=float)
    assert np.allclose(r['op_ct'], exp_ct), r['op_ct']

    env = FJSPEnvFamily(2, 2, family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    for (j, m) in acts:
        env.step(np.array([j * 2 + m]))
    assert np.allclose(env.true_op_ct[0], exp_ct), env.true_op_ct[0]
    assert env.current_makespan[0] == 17.0
    print('T1 setup handcrafted        OK')


def t2_window_handcrafted():
    jl, pt, op_class = _tiny_instance()
    windows = [[(3, 6), (10, 12)],   # machine 0
               [(0, 5)]]             # machine 1
    desc = ConstraintDescriptor(active=(False, False, True),
                                lag=np.zeros(4), op_class=op_class,
                                windows=windows, horizon=100)
    acts = [(0, 0), (1, 1), (0, 0), (1, 0)]
    r = simulate(jl, pt, desc, acts)
    # op0 j0 m0: st=0, dur4 overlaps (3,6) -> st=6, ct=10
    # op2 j1 m1: st=0, dur4 overlaps (0,5) -> st=5, ct=9
    # op1 j0 m0: st=max(10,10)=10 dur3 overlaps (10,12) -> st=12, ct=15
    # op3 j1 m0: st=max(9,15)=15, dur2, no overlap -> ct=17
    exp_ct = np.array([10, 15, 9, 17], dtype=float)
    assert np.allclose(r['op_ct'], exp_ct), r['op_ct']

    env = FJSPEnvFamily(2, 2, family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    for (j, m) in acts:
        env.step(np.array([j * 2 + m]))
    assert np.allclose(env.true_op_ct[0], exp_ct), env.true_op_ct[0]
    print('T2 window handcrafted       OK')


def _random_instance(rng, J=3, M=3, ops_lo=2, ops_hi=4):
    jl = rng.integers(ops_lo, ops_hi + 1, size=J)
    N = int(jl.sum())
    pt = rng.integers(1, 20, size=(N, M)).astype(float)
    # knock out some compatibilities but keep >=1 per op
    mask = rng.random((N, M)) < 0.35
    for i in range(N):
        if mask[i].all():
            mask[i, rng.integers(0, M)] = False
    pt[mask] = 0.0
    meta = {'time_lag': rng.integers(0, 15, size=N),
            'routing_class': rng.integers(0, 4, size=J)}
    return jl, pt, meta


def t3_fuzz_vs_reference(n_cases=200, seed=7):
    rng = np.random.default_rng(seed)
    for case in range(n_cases):
        regime = REGIMES_ALL[case % len(REGIMES_ALL)]
        jl, pt, meta = _random_instance(rng)
        M = pt.shape[1]
        desc = make_descriptor(regime, jl, pt, meta, M, seed=1000 + case)

        env = FJSPEnvFamily(len(jl), M, family_mode=True)
        env.set_initial_data([jl], [pt], descriptor_list=[desc])

        # random valid action sequence
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        nxt = first.copy()
        acts = []
        jobs_left = list(range(len(jl)))
        while jobs_left:
            j = jobs_left[int(rng.integers(0, len(jobs_left)))]
            op = nxt[j]
            mchs = np.where(pt[op] > 0)[0]
            m = int(mchs[int(rng.integers(0, len(mchs)))])
            acts.append((j, m))
            nxt[j] += 1
            if nxt[j] >= first[j] + jl[j]:
                jobs_left.remove(j)

        ref = simulate(jl, pt, desc, acts)
        for (j, m) in acts:
            env.step(np.array([j * M + m]))

        assert np.allclose(env.true_op_ct[0], ref['op_ct'], atol=1e-6), \
            f'case {case} regime {regime}: env vs reference mismatch\n' \
            f'{env.true_op_ct[0]}\n{ref["op_ct"]}'
        assert abs(env.current_makespan[0] - ref['makespan']) < 1e-6

        # independent feasibility validation of the env's schedule
        viol = validate_schedule(jl, pt, desc, ref['op_st'], ref['op_ct'],
                                 ref['op_mch'])
        assert not viol, f'case {case} regime {regime}: {viol}'
    print(f'T3 fuzz vs reference        OK ({n_cases} cases, all regimes)')


def t4_reward_soundness(n_cases=160, seed=11):
    """Admissibility along rollouts: at every step the family bound must not
    exceed the FINAL realized makespan (any feasible completion), and must be
    exactly tight at termination. Also: telescoped rewards sum to
    init_bound - final_makespan (normalized axis)."""
    rng = np.random.default_rng(seed)
    for case in range(n_cases):
        regime = REGIMES_ALL[case % len(REGIMES_ALL)]
        jl, pt, meta = _random_instance(rng)
        M = pt.shape[1]
        desc = make_descriptor(regime, jl, pt, meta, M, seed=5000 + case)
        env = FJSPEnvFamily(len(jl), M, family_mode=True)
        env.set_initial_data([jl], [pt], descriptor_list=[desc])

        bounds = [float(np.max(env.op_ct_lb, axis=1)[0])]
        rewards = []
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        nxt = first.copy()
        jobs_left = list(range(len(jl)))
        while jobs_left:
            j = jobs_left[int(rng.integers(0, len(jobs_left)))]
            op = nxt[j]
            mchs = np.where(pt[op] > 0)[0]
            m = int(mchs[int(rng.integers(0, len(mchs)))])
            _, r, _ = env.step(np.array([j * M + m]))
            rewards.append(float(r[0]))
            bounds.append(float(np.max(env.op_ct_lb, axis=1)[0]))
            nxt[j] += 1
            if nxt[j] >= first[j] + jl[j]:
                jobs_left.remove(j)

        slope = env._slope
        final_ms_norm = env.current_makespan[0] * slope
        # admissibility at every prefix
        for t, b in enumerate(bounds):
            assert b <= final_ms_norm + 1e-6, \
                f'case {case} regime {regime}: bound {b} > final {final_ms_norm} at step {t}'
        # tightness at termination
        assert abs(bounds[-1] - final_ms_norm) < 1e-6, \
            f'case {case} regime {regime}: bound not tight at termination'
        # telescoping identity
        assert abs(sum(rewards) - (bounds[0] - final_ms_norm)) < 1e-5
    print(f'T4 reward soundness         OK ({n_cases} cases, all regimes)')


def t5_e0_anchor():
    """Legacy family env + LAG-only descriptor reproduces the released
    lag-only policy's evaluation bit-exactly (mean 210.9 on
    10x25+ppvc-mixed)."""
    import json
    import glob
    import torch
    from params import configs
    snap = json.load(open('./train_log/PPVC/config_10x25+ppvc-mixed+full.json'))
    for k in ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
              'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
              'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
              'layer_fea_output_dim', 'num_mlp_layers_actor',
              'hidden_dim_actor', 'num_mlp_layers_critic',
              'hidden_dim_critic', 'dropout_prob'):
        if k in snap:
            setattr(configs, k, snap[k])
    from model.PPO import PPO_initialize
    from common_utils import greedy_select_action
    from ppvc_instance_generator import load_instance

    device = torch.device(configs.device)
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        './trained_network/PPVC/10x25+ppvc-mixed+full.pth',
        map_location=device))
    ppo.policy.eval()

    ref = np.load('./test_results/PPVC/10x25+ppvc-mixed/'
                  'Result_greedy+10x25+ppvc-mixed+full_10x25+ppvc-mixed.npy')

    stems = sorted(glob.glob('./data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))
    stems = [s[:-len('.fjs')] for s in stems]
    ms = []
    for stem in stems:
        jl, pt, meta = load_instance(stem)
        N = pt.shape[0]
        desc = ConstraintDescriptor(
            active=(True, False, False), lag=meta['time_lag'],
            op_class=np.zeros(N, dtype=int), horizon=1)
        env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=False,
                            use_lag_features=True)
        state = env.set_initial_data(
            [jl], [pt], descriptor_list=[desc],
            op_type_list=[meta['op_type']], mch_type_list=[meta['mch_type']])
        while not env.done().all():
            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                                   op_mask=state.op_mask_tensor,
                                   candidate=state.candidate_tensor,
                                   fea_m=state.fea_m_tensor,
                                   mch_mask=state.mch_mask_tensor,
                                   comp_idx=state.comp_idx_tensor,
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                   fea_pairs=state.fea_pairs_tensor,
                                   op_type=state.op_type_tensor,
                                   mch_type=state.mch_type_tensor)
            action = greedy_select_action(pi)
            state, _, _ = env.step(action.cpu().numpy())
        ms.append(float(env.current_makespan[0]))
    ms = np.array(ms)
    exact = np.array_equal(ms, ref[:, 0])
    print(f'T5 E0 anchor                mean={ms.mean():.1f} '
          f'bit-exact vs published: {exact}')
    assert exact, 'E0 REGRESSION FAILED'


if __name__ == '__main__':
    t1_setup_handcrafted()
    t2_window_handcrafted()
    t3_fuzz_vs_reference()
    t4_reward_soundness()
    if not SKIP_E0:
        t5_e0_anchor()
    print('ALL P1 TESTS PASSED')
