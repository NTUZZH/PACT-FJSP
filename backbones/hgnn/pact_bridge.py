"""
Bridge between the HGNN scheduler (Song et al., IEEE TII 2023) and PACT's
constraint-family code in the repository root.

Everything that defines the family is read from PACT, not re-derived here:
  * pact()                  imports PACT's environment, descriptor, generator,
                            reference simulator and dispatching rules. PACT's
                            params.py parses sys.argv at import, so the import
                            runs under a neutral argv ('--device cpu') and the
                            caller's argv is restored afterwards. PACT's
                            environment then lives on the CPU (numpy) whatever
                            device HGNN uses.
  * sample_family_batch     line-for-line copy of FamilyTrainer.sample_batch
                            (train_family.py): same generator, same per-update
                            regime balancing, same consumption of the global
                            numpy and random streams, so a run seeded like a
                            PACT run draws the same instances and descriptors.
                            tests/test_family_hgnn.py asserts the equality
                            against train_family.py itself.
  * bound_min_term          the per-operation term m_ij of the certified
                            estimate, computed by FJSPEnvFamily._bound_min_term
                            (delta_mode 'min') on the integer-hour axis.
  * make_shadow             a PACT FJSPEnvFamily over the same batch; the HGNN
                            family environment reads its admissible-pair mask.
  * family_vali_sets        PACT's validation protocol: 20 fixed base instances
                            per size (generator seed seed_vali + i), descriptor
                            seed seed_vali * 1000 + i per regime, normalised by
                            the mean MWKR makespan of that size and regime.
"""
import os
import random
import sys
import types

HGNN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HGNN_DIR))

import numpy as np  # noqa: E402

_PACT = None


def pact():
    '''PACT modules, imported once under a neutral argv.'''
    global _PACT
    if _PACT is not None:
        return _PACT
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    saved = sys.argv
    sys.argv = [saved[0] if saved else 'hgnn', '--device', 'cpu']
    try:
        import params  # noqa: F401  (parses the neutral argv once)
        from constraint_family import descriptor as D
        from constraint_family import fjsp_env_family as F
        from constraint_family import reference_sim as R
        import ppvc_instance_generator as G
        import common_utils as U
    finally:
        sys.argv = saved
    _PACT = types.SimpleNamespace(D=D, F=F, R=R, G=G, U=U)
    return _PACT


FACTORY_NAMES = ('default', 'small', 'tight')


def factory(name):
    G = pact().G
    return {'default': G.DEFAULT_FACTORY, 'small': G.SMALL_FACTORY, 'tight': G.TIGHT_FACTORY}[name]


def n_machines(factory_name):
    mch_type, _ = pact().G.build_factory(factory(factory_name))
    return len(mch_type)


def scale_lags(desc, lag_scale):
    '''Training-time lag scaling (rounded), as the lag-only port's --lag_scale. Applied after the descriptor is
    drawn, so the random streams and the setup and window draws are untouched.'''
    if lag_scale == 1.0 or not desc.has_lag:
        return desc
    desc.lag = np.rint(np.asarray(desc.lag, dtype=float) * float(lag_scale))
    return desc


def sample_family_batch(num_envs, regimes, n_j, n_m, ppvc_mix, factory_name, desc_shift=None):
    '''FamilyTrainer.sample_batch of train_family.py, statement for statement.

    num_envs fresh instances, regimes balanced; the remainder of num_envs over the regimes goes to a random
    subset of the regimes, so each regime's expected share is exactly num_envs / len(regimes).'''
    P = pact()
    station_counts = factory(factory_name)
    k, rem = divmod(num_envs, len(regimes))
    regs = list(regimes) * k
    if rem:
        regs += random.sample(list(regimes), rem)
    random.shuffle(regs)
    jls, pts, descs, opts, mchts = [], [], [], [], []
    for r in regs:
        inst_seed = int(np.random.randint(0, 2 ** 31 - 1))
        jl, pt, meta = P.G.ppvc_instance_generator(
            n_modules=n_j, class_mix=ppvc_mix,
            station_counts=station_counts, seed=inst_seed)
        desc_seed = int(np.random.randint(0, 2 ** 31 - 1))
        descs.append(P.D.make_descriptor(r, jl, pt, meta, n_m,
                                         seed=desc_seed,
                                         shift=desc_shift))
        jls.append(jl)
        pts.append(pt)
        opts.append(meta['op_type'])
        mchts.append(meta['mch_type'])
    return jls, pts, descs, opts, mchts


def bound_min_term(pts, descs):
    '''[B, N] per-operation term m_ij = min_k p_ijk + a_S * min over source classes of S[., class(ij)], in hours,
    evaluated by PACT's FJSPEnvFamily._bound_min_term with the admissible envelope (delta_mode 'min').'''
    P = pact()
    pt = np.asarray(pts, dtype=float)
    min_pt = np.where(pt > 0, pt, np.inf).min(axis=2)
    min_pt = np.where(np.isfinite(min_pt), min_pt, 0.0)
    C = P.D.N_CLASSES
    setup = np.zeros((len(descs), C + 1, C))
    for e, d in enumerate(descs):
        if d.has_setup and d.setup is not None:
            setup[e] = d.setup
    proxy = types.SimpleNamespace(
        delta_mode='min', reward_delta_env=True, op_min_pt=min_pt, setup_norm=setup,
        op_class=np.stack([np.asarray(d.op_class, dtype=int) for d in descs]),
        c_bits=np.array([[d.has_lag, d.has_setup, d.has_window] for d in descs], dtype=float))
    return P.F.FJSPEnvFamily._bound_min_term(proxy)


def make_shadow(jls, pts, descs):
    '''PACT's environment over the same batch, published non-delay mask; its admissible pairs are
    ~shadow.dynamic_pair_mask ([E, J, M], action job * M + machine).'''
    P = pact()
    env = P.F.FJSPEnvFamily(len(jls[0]), pts[0].shape[1], family_mode=True)
    env.set_initial_data(list(jls), list(pts), descriptor_list=list(descs))
    return env


def mwkr_makespan(jl, pt, desc, seed_test=50):
    '''Deterministic MWKR makespan in PACT's environment, as train_family.py computes its validation norms.'''
    P = pact()
    env = P.F.FJSPEnvFamily(len(jl), pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc])
    np.random.seed(seed_test)
    while not env.done().all():
        env.step(np.array([P.U.heuristic_select_action('MWKR', env)]))
    return float(env.current_makespan[0])


def family_vali_sets(regimes, sizes, ppvc_mix, factory_name, seed_vali=100, vali_size=20, lag_scale=1.0):
    '''{(size, regime): (jls, pts, descs, mwkr_norm)}: PACT's per-regime validation sets of train_family.py,
    optionally at several sizes (same generator seeds at every size). The MWKR norm is computed in PACT's
    environment on the descriptors actually used (after lag scaling).'''
    P = pact()
    station_counts = factory(factory_name)
    n_m = n_machines(factory_name)
    n = min(vali_size, 20)
    out = {}
    for size in sizes:
        base = [P.G.ppvc_instance_generator(n_modules=size, class_mix=ppvc_mix, station_counts=station_counts,
                                            seed=seed_vali + i) for i in range(n)]
        for regime in regimes:
            descs = [scale_lags(P.D.make_descriptor(regime, jl, pt, meta, n_m, seed=seed_vali * 1000 + i),
                                lag_scale)
                     for i, (jl, pt, meta) in enumerate(base)]
            norm = float(np.mean([mwkr_makespan(b[0], b[1], d) for b, d in zip(base, descs)]))
            out[(size, regime)] = ([b[0] for b in base], [b[1] for b in base], descs, norm)
    return out
