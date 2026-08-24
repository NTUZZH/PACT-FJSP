"""Fuzz test: GA batched decoder == scalar reference simulator (this paper, E5).

The GA's speed depends on a lightweight batched decoder that re-implements the
family dynamics; this test guards it against divergence from the validated
reference simulator, across all regimes and random (OS, MA) chromosomes.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from params import configs
configs.device = 'cpu'

import numpy as np

from constraint_family.descriptor import make_descriptor, REGIMES_ALL
from constraint_family.reference_sim import simulate
from eval_ga_family import decode_population, _job_starts


def rand_instance(rng, J=3, M=3):
    jl = rng.integers(2, 4, size=J)
    N = int(jl.sum())
    pt = rng.integers(1, 15, size=(N, M)).astype(float)
    mask = rng.random((N, M)) < 0.35
    for i in range(N):
        if mask[i].all():
            mask[i, rng.integers(0, M)] = False
    pt[mask] = 0.0
    meta = {'time_lag': rng.integers(0, 12, size=N),
            'routing_class': rng.integers(0, 4, size=J),
            'op_type': np.zeros(N, dtype=int),
            'mch_type': np.zeros(M, dtype=int)}
    return jl, pt, meta


def rand_chromosomes(rng, jl, pt, P):
    J = len(jl)
    base = np.concatenate([np.full(int(L), j) for j, L in enumerate(jl)])
    N = pt.shape[0]
    compat = [np.where(pt[i] > 0)[0] for i in range(N)]
    os_pop, ma_pop = [], []
    for _ in range(P):
        o = base.copy(); rng.shuffle(o)
        m = np.array([c[rng.integers(len(c))] for c in compat])
        os_pop.append(o); ma_pop.append(m)
    return np.array(os_pop), np.array(ma_pop)


def main(n_cases=120, P=8, seed=31):
    rng = np.random.default_rng(seed)
    for case in range(n_cases):
        regime = REGIMES_ALL[case % len(REGIMES_ALL)]
        jl, pt, meta = rand_instance(rng)
        desc = make_descriptor(regime, jl, pt, meta, pt.shape[1],
                               seed=6000 + case)
        os_pop, ma_pop = rand_chromosomes(rng, jl, pt, P)
        ms_batch, amch, oct_ = decode_population(
            jl, pt, desc, meta['op_type'], meta['mch_type'], os_pop, ma_pop)
        js = _job_starts(jl)
        for p in range(P):
            nxt = js.copy()
            acts = []
            for j in os_pop[p]:
                op = nxt[j]
                acts.append((int(j), int(ma_pop[p, op])))
                nxt[j] += 1
            ref = simulate(jl, pt, desc, acts)
            assert abs(ms_batch[p] - ref['makespan']) < 1e-6, \
                f'case {case} {regime} ind {p}: batch {ms_batch[p]} != ' \
                f'ref {ref["makespan"]}'
            assert np.allclose(oct_[p], ref['op_ct'], atol=1e-6)
    print(f'GA batched decoder == reference simulator  OK '
          f'({n_cases} cases x {P} chromosomes, all regimes)')


if __name__ == '__main__':
    main()
