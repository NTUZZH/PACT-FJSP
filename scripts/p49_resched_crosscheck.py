"""Cross-implementation check of the ReSched family port.

Two independent implementations of the family dynamics must agree exactly:
the ported FamilyReSchedEnv (built on ReSched's subproblem environment) and
this repo's reference simulator. For random dispatch sequences on random
instances of every regime:

  R1  identical start and end times, operation by operation, both axes
      integer hours (ReSched normalizes nothing, so equality is exact);
  R2  the ported estimate is admissible along the rollout (never exceeds the
      realized makespan of the completed schedule it leads to) and exact at
      termination -- the two properties of Theorem 1, now on the second
      backbone's environment;
  R3  the completed schedule passes the independent validator.

Run:  python scripts/p49_resched_crosscheck.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from constraint_family.descriptor import make_descriptor
from constraint_family.reference_sim import simulate, validate_schedule
from constraint_family.resched_port.family_env import FamilyReSchedEnv
from ppvc_instance_generator import ppvc_instance_generator, build_factory, \
    DEFAULT_FACTORY

REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')


def one_case(seed, regime, rng):
    mch_type, _ = build_factory(DEFAULT_FACTORY)
    n_m = len(mch_type)
    jl, pt, meta = ppvc_instance_generator(n_modules=4, class_mix='mixed',
                                           station_counts=DEFAULT_FACTORY,
                                           seed=seed)
    desc = make_descriptor(regime, jl, pt, meta, n_m, seed=seed * 3 + 1)

    env = FamilyReSchedEnv()
    env.load_family_batch([jl], [pt], [desc])

    acts = []
    est_seq = [float(env.state.est_last_makespan[0])]
    done = False
    while not done:
        mask = env.state.action_mask[0]                     # [R, M]
        pairs = np.argwhere(mask)
        r, m = pairs[rng.integers(len(pairs))]
        job = int(env.state.job_idx[0, r])
        acts.append((job, int(m)))
        _, _, done = env.step((np.array([r]), np.array([m])))
        est_seq.append(float(env.state.est_last_makespan[0]))
    # the terminal auto-dispatch appended one more decision: recover it from
    # the solution record so the reference replay sees the full sequence
    sol_jobs = env.solution.job_idx[0]
    sol_mchs = env.solution.machine_idx[0]
    acts = [(int(j), int(m)) for j, m in zip(sol_jobs, sol_mchs)]

    ref = simulate(jl, pt, desc, acts)

    st_env = np.full(pt.shape[0], -1.0)
    et_env = np.full(pt.shape[0], -1.0)
    mch_env = np.full(pt.shape[0], -1, dtype=int)
    stm, ftm = env.state.start_time[0], env.state.finish_time[0]
    for o in range(pt.shape[0]):
        m = np.flatnonzero(ftm[o] >= 0)
        assert len(m) == 1, f'op {o} scheduled on {len(m)} machines'
        mch_env[o] = m[0]
        st_env[o], et_env[o] = stm[o, m[0]], ftm[o, m[0]]

    # R1: exact agreement with the reference simulator
    assert np.array_equal(mch_env, ref['op_mch']), 'machine assignment drift'
    assert np.allclose(st_env, ref['op_st'], atol=1e-9), \
        f'start-time drift, max {np.abs(st_env - ref["op_st"]).max()}'
    assert np.allclose(et_env, ref['op_ct'], atol=1e-9), 'end-time drift'

    # R2: admissible along the rollout, exact at termination
    final_mk = float(ref['makespan'])
    assert all(e <= final_mk + 1e-9 for e in est_seq), \
        f'estimate exceeded the realized makespan: max {max(est_seq)} vs {final_mk}'
    assert abs(est_seq[-1] - final_mk) < 1e-9, \
        f'estimate not exact at termination: {est_seq[-1]} vs {final_mk}'

    # R3: independent validator
    viol = validate_schedule(jl, pt, desc, st_env, et_env, mch_env)
    assert not viol, viol[:3]
    return final_mk


def main():
    rng = np.random.default_rng(7)
    n = 0
    for regime in REGIMES:
        for seed in (201, 202, 203, 204, 205):
            one_case(seed, regime, rng)
            n += 1
    print(f'R1 exact start/end agreement with reference_sim   OK ({n} cases)')
    print(f'R2 estimate admissible along rollout, exact at T  OK ({n} cases)')
    print(f'R3 independent validator                          OK ({n} cases)')
    print('RESCHED PORT CROSS-CHECK PASSED (all regimes)')


if __name__ == '__main__':
    main()
