"""E4 blocking environment tests (this paper, P5).

  B1  handcrafted case (hand-computed blocking schedule, with lag)
  B2  fuzz: FJSPEnvBlocking true dynamics == independent reference simulator
      on random instances with random legal action sequences (+ deadlock
      agreement between env masking and reference legality)

Run:  python scripts/p5_blocking_tests.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor
from constraint_family.fjsp_env_blocking import (FJSPEnvBlocking, BIG,
                                                 rollout_blocking_reference)


def make_env(jl, pt, lag):
    N = pt.shape[0]
    desc = ConstraintDescriptor(
        active=(lag is not None, False, False),
        lag=np.zeros(N) if lag is None else lag,
        op_class=np.zeros(N, dtype=int), horizon=100)
    env = FJSPEnvBlocking(len(jl), pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    return env


def b1_handcrafted():
    # 2 jobs x 2 ops, 2 machines; op3 incompatible with m1
    jl = np.array([2, 2])
    pt = np.array([[4., 6.], [3., 5.], [5., 4.], [2., 0.]])
    lag = np.array([2, 0, 0, 0])
    env = make_env(jl, pt, lag)
    acts = [(0, 0), (1, 1), (0, 1), (1, 0)]
    # j0 op0 m0: st 0, ct 4; m0 HELD by j0 (lag: ready 6, module cures on m0)
    # j1 op2 m1: st 0, ct 4; m1 held by j1
    # j0 op1 m1: m1 held by j1 -> would be masked... use m1? ILLEGAL.
    # -> reference returns None; env masks the pair. Choose op1 on m1 only
    #    after j1 releases: instead schedule j1 op3 first: m0 held by j0!
    #    op3 compatible only with m0 -> masked -> DEADLOCK scenario check:
    #    remaining pairs: j0 op1 on m0(held by j0: SELF -> allowed)/m1(held
    #    by j1 -> masked). So j0 op1 on m0 is legal:
    acts = [(0, 0), (1, 1), (0, 0), (1, 0)]
    # j0 op1 m0: self-held ok. st = max(ready=6, free=4) = 6, ct = 9;
    #            m0 held by j0? op1 is j0's LAST op -> released at ct.
    #            j1's op2 done on m1 (held by j1, ready 4).
    # j1 op3 m0: released at 9 -> st = max(4, 9) = 9, ct = 11;
    #            release m1 at st=9 (transfer).
    ref = rollout_blocking_reference(jl, pt, lag, acts)
    assert ref is not None
    assert np.allclose(ref['op_ct'], [4, 9, 4, 11]), ref['op_ct']
    env = make_env(jl, pt, lag)
    for (j, m) in acts:
        env.step(np.array([j * 2 + m]))
    assert np.allclose(env.true_op_ct[0], [4, 9, 4, 11]), env.true_op_ct[0]
    assert not env.deadlock[0]
    print('B1 blocking handcrafted     OK')


def b2_fuzz(n_cases=250, seed=17):
    rng = np.random.default_rng(seed)
    n_deadlocks = 0
    for case in range(n_cases):
        J, M = 3, 3
        jl = rng.integers(2, 4, size=J)
        N = int(jl.sum())
        pt = rng.integers(1, 15, size=(N, M)).astype(float)
        mask = rng.random((N, M)) < 0.35
        for i in range(N):
            if mask[i].all():
                mask[i, rng.integers(0, M)] = False
        pt[mask] = 0.0
        lag = rng.integers(0, 8, size=N).astype(float) \
            if case % 2 else None

        env = make_env(jl, pt, lag)
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        nxt = first.copy()
        held = np.full(M, -1, dtype=int)   # mirror of legality for sampling
        acts = []
        left = list(range(J))
        dead = False
        while left:
            # legal (job, mch) pairs under blocking
            legal = []
            for j in left:
                op = nxt[j]
                for m in np.where(pt[op] > 0)[0]:
                    if held[m] < 0 or held[m] == j:
                        legal.append((j, int(m)))
            if not legal:
                dead = True
                break
            j, m = legal[int(rng.integers(0, len(legal)))]
            acts.append((j, m))
            env.step(np.array([j * M + m]))
            # mirror bookkeeping
            op = nxt[j]
            # release previous holder of j
            for mm in range(M):
                if held[mm] == j and mm != m:
                    held[mm] = -1
            held[m] = j if op != first[j] + jl[j] - 1 else -1
            nxt[j] += 1
            if nxt[j] >= first[j] + jl[j]:
                left.remove(j)

        if dead:
            n_deadlocks += 1
            assert bool(env.deadlock[0]), \
                f'case {case}: reference deadlocked but env did not flag'
            continue

        ref = rollout_blocking_reference(jl, pt, lag, acts)
        assert ref is not None, f'case {case}: reference rejected env-legal seq'
        sched = env.true_op_ct[0] > 0
        assert np.allclose(env.true_op_ct[0][sched], ref['op_ct'][sched],
                           atol=1e-6), \
            f'case {case}:\n{env.true_op_ct[0]}\n{ref["op_ct"]}'
        assert abs(env.current_makespan[0] - ref['makespan']) < 1e-6
    print(f'B2 blocking fuzz            OK ({n_cases} cases, '
          f'{n_deadlocks} deadlocks — env flags agree)')


def b3_cpsat_blocking(n_cases=5, seed=29):
    """CP-SAT blocking model: optimum <= brute-force-over-legal-rollouts
    (CP may insert delays) and >= the non-blocking optimum (holds only
    tighten); no-wait optimum >= base optimum likewise."""
    import itertools
    from constraint_family.cpsat_family import family_solver
    rng = np.random.default_rng(seed)
    for case in range(n_cases):
        J, M = 2, 2
        jl = rng.integers(2, 3, size=J)
        N = int(jl.sum())
        pt = rng.integers(1, 12, size=(N, M)).astype(float)
        pt[N - 1, 1] = 0.0  # keep an incompatibility (slope invariant)
        lag = rng.integers(0, 6, size=N).astype(float)
        desc = ConstraintDescriptor(active=(True, False, False), lag=lag,
                                    op_class=np.zeros(N, dtype=int),
                                    horizon=200)
        base = family_solver(jl, pt, desc, time_limit=10)
        blk = family_solver(jl, pt, desc, time_limit=10, blocking=True)
        nw = family_solver(jl, pt, desc, time_limit=10, nowait=True)
        assert blk['status'] == 'OPTIMAL' and nw['status'] == 'OPTIMAL'
        assert blk['objective'] >= base['objective'] - 1e-6
        assert nw['objective'] >= base['objective'] - 1e-6

        # brute force over legal blocking rollouts (upper bound witness)
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        ops_of_job = []
        for j, L in enumerate(jl):
            ops_of_job += [j] * L
        best = np.inf
        for perm in set(itertools.permutations(ops_of_job)):
            nxt = first.copy()
            step_choices = []
            for j in perm:
                op = nxt[j]
                step_choices.append([(j, int(m))
                                     for m in np.where(pt[op] > 0)[0]])
                nxt[j] += 1
            for combo in itertools.product(*step_choices):
                r = rollout_blocking_reference(jl, pt, lag, list(combo))
                if r is not None:
                    best = min(best, r['makespan'])
        assert blk['objective'] <= best + 1e-6, \
            f'case {case}: CP blocking {blk["objective"]} > BF {best}'
        print(f'B3 case {case}  OK  base={base["objective"]:.0f} '
              f'blocking={blk["objective"]:.0f} (bf {best:.0f}) '
              f'nowait={nw["objective"]:.0f}')


if __name__ == '__main__':
    b1_handcrafted()
    b2_fuzz()
    b3_cpsat_blocking()
    print('ALL BLOCKING TESTS PASSED')
