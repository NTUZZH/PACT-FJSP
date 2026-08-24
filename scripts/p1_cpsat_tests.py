"""P1 CP-SAT family-solver tests (this paper).

  C1  N/L regimes: family_solver == existing fjsp_solver (same optimum)
  C2  all regimes: CP schedule passes the independent validator
  C3  all regimes: CP optimum <= best of random env rollouts (reference
      direction) and == brute-force optimum on tiny instances (exhaustive
      over active schedules via the reference simulator)
  C4  monotonicity: optimum(N) <= optimum(L), optimum(S), optimum(W) <= ...
      (adding monotone delays cannot decrease the optimum) on each instance

Run:  python scripts/p1_cpsat_tests.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from constraint_family.descriptor import make_descriptor, REGIMES_ALL
from constraint_family.cpsat_family import family_solver, matrix_to_jobs
from constraint_family.reference_sim import simulate, validate_schedule
from ortools_solver import fjsp_solver


def _rand_instance(rng, J=3, M=3, ops_lo=2, ops_hi=3):
    jl = rng.integers(ops_lo, ops_hi + 1, size=J)
    N = int(jl.sum())
    pt = rng.integers(1, 15, size=(N, M)).astype(float)
    mask = rng.random((N, M)) < 0.3
    for i in range(N):
        if mask[i].all():
            mask[i, rng.integers(0, M)] = False
    pt[mask] = 0.0
    meta = {'time_lag': rng.integers(0, 10, size=N),
            'routing_class': rng.integers(0, 4, size=J)}
    return jl, pt, meta


def _brute_force_optimum(jl, pt, desc, limit_seqs=250000):
    """Exhaustive min makespan over all job-orderings x machine choices via
    the reference simulator (active schedules). Exact for tiny instances IF
    the optimum is achieved by a no-idle-insertion schedule; used only as an
    UPPER-bound sanity witness, never as the reference itself."""
    jl = np.asarray(jl, dtype=int)
    N = int(jl.sum())
    ops_of_job = []
    for j, L in enumerate(jl):
        ops_of_job += [j] * L
    best = np.inf
    n_seq = 0
    for perm in set(itertools.permutations(ops_of_job)):
        n_seq += 1
        if n_seq > limit_seqs:
            break
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        nxt = first.copy()
        choices_per_step = []
        ok = True
        for j in perm:
            op = nxt[j]
            mchs = np.where(pt[op] > 0)[0]
            choices_per_step.append([(j, int(m)) for m in mchs])
            nxt[j] += 1
        for combo in itertools.product(*choices_per_step):
            r = simulate(jl, pt, desc, list(combo))
            best = min(best, r['makespan'])
    return best


def main():
    rng = np.random.default_rng(23)
    n_inst = 6
    for case in range(n_inst):
        jl, pt, meta = _rand_instance(rng)
        M = pt.shape[1]
        opt = {}
        for regime in REGIMES_ALL:
            desc = make_descriptor(regime, jl, pt, meta, M, seed=9000 + case)
            res = family_solver(jl, pt, desc, time_limit=20)
            assert res['status'] in ('OPTIMAL',), \
                f'case {case} {regime}: {res["status"]}'
            opt[regime] = res['objective']

            # C2: independent validation
            viol = validate_schedule(jl, pt, desc, res['op_st'], res['op_ct'],
                                     res['assigned_mch'])
            assert not viol, f'case {case} {regime}: {viol}'

            # C1: N and L must match the existing solver exactly
            if regime in ('N', 'L'):
                jobs = matrix_to_jobs(jl, pt)
                lag = desc.lag if regime == 'L' else None
                ref_obj, _ = fjsp_solver(jobs, M, 20, time_lag=lag)
                assert abs(ref_obj - res['objective']) < 1e-6, \
                    f'case {case} {regime}: family {res["objective"]} vs ' \
                    f'legacy {ref_obj}'

            # C3a: CP optimum lower-bounds random rollouts
            rr = np.random.default_rng(100 + case)
            for _ in range(30):
                first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
                nxt = first.copy()
                acts = []
                left = list(range(len(jl)))
                while left:
                    j = left[int(rr.integers(0, len(left)))]
                    op = nxt[j]
                    mchs = np.where(pt[op] > 0)[0]
                    acts.append((j, int(mchs[int(rr.integers(0, len(mchs)))])))
                    nxt[j] += 1
                    if nxt[j] >= first[j] + jl[j]:
                        left.remove(j)
                sim = simulate(jl, pt, desc, acts)
                assert res['objective'] <= sim['makespan'] + 1e-6, \
                    f'case {case} {regime}: CP {res["objective"]} > rollout ' \
                    f'{sim["makespan"]}'

        # C4: family monotonicity on this instance
        for a, b in [('N', 'L'), ('N', 'S'), ('N', 'W'), ('L', 'LS'),
                     ('L', 'LW'), ('S', 'LS'), ('S', 'SW'), ('W', 'LW'),
                     ('W', 'SW'), ('LS', 'LSW'), ('LW', 'LSW'), ('SW', 'LSW')]:
            assert opt[a] <= opt[b] + 1e-6, \
                f'case {case}: opt[{a}]={opt[a]} > opt[{b}]={opt[b]}'
        print(f'C1/C2/C3a/C4 case {case}  OK  '
              + ' '.join(f'{r}={opt[r]:.0f}' for r in REGIMES_ALL))

    # C3b: brute-force witness on 2-job instances (exhaustive)
    rng2 = np.random.default_rng(41)
    for case in range(4):
        jl, pt, meta = _rand_instance(rng2, J=2, M=2, ops_lo=2, ops_hi=2)
        for regime in ('LS', 'SW', 'LSW'):
            desc = make_descriptor(regime, jl, pt, meta, pt.shape[1],
                                   seed=7000 + case)
            res = family_solver(jl, pt, desc, time_limit=10)
            bf = _brute_force_optimum(jl, pt, desc)
            assert res['objective'] <= bf + 1e-6
            assert res['status'] == 'OPTIMAL'
            # for these tiny cases the active-schedule optimum should
            # coincide with the CP optimum
            assert abs(res['objective'] - bf) < 1e-6, \
                f'tiny case {case} {regime}: CP {res["objective"]} vs BF {bf}'
        print(f'C3b tiny case {case}  OK')

    print('ALL P1 CP-SAT TESTS PASSED')


if __name__ == '__main__':
    main()
