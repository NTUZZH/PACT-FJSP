"""Tests for the alternative CP-SAT arms (constraint_family/cpsat_arms.py).

  * the rebuilt current model equals cpsat_family.family_solver's model,
    proto for proto (variables, constraints and hints), in all eight
    regimes at 10x25;
  * on small generated instances solved to optimality, every exact arm
    (the current model under each parameter variant, compact, compact_red)
    reaches the optimum of family_solver, in all eight regimes; with
    release dates the exact arms agree with each other, family_solver
    models them (its optimum passes both validators and equals the compact
    arm's, its model equals the rebuilt one) and its horizon covers a
    release date beyond the release-free horizon;
  * every arm returns a schedule that passes both validators in all eight
    regimes, and a restriction never reports a value below the optimum;
  * the strict validator rejects a setup that starts before the job is
    present, a block over an outage and an early first operation;
  * the large-neighbourhood search never returns a schedule worse than its
    hint, its sub-models accept the incumbent as a feasible complete hint
    for every neighbourhood kind, and a 60 s run at 50x25 on a setup regime
    with 2 workers completes and validates;
  * the runner writes a validated record and schedule, skips existing
    records and refuses more workers than cores without --dev.

Run from the repository root:  python3 -m pytest tests/test_cpsat_arms.py -q
CPU only; CP-SAT runs with at most 2 workers.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from ortools.sat.python import cp_model  # noqa: E402

from constraint_family import cpsat_arms as A  # noqa: E402
from constraint_family import cpsat_family, reference_sim  # noqa: E402
from constraint_family.descriptor import REGIMES_ALL, make_descriptor  # noqa: E402
from ppvc_instance_generator import TIGHT_FACTORY, ppvc_instance_generator  # noqa: E402

WORKERS = 2
EXACT_ARMS = [a for a, s in A.ARMS.items() if s['exactness'] == 'exact']
RESTRICTIONS = [a for a, s in A.ARMS.items() if s['exactness'] == 'restriction']


def small_case(J, L, seed, regime, release=False):
    """J jobs, the first L operations of each PPVC route, on the
    capacity-tight factory; descriptor drawn as for the benchmark."""
    jl, pt, meta = ppvc_instance_generator(n_modules=J, class_mix='mixed',
                                           station_counts=TIGHT_FACTORY,
                                           seed=seed)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    keep = np.concatenate([np.arange(f, f + L) for f in first])
    pt = pt[keep]
    pt = pt[:, (pt > 0).any(0)]
    meta = dict(meta)
    meta['time_lag'] = np.asarray(meta['time_lag'])[keep]
    jl2 = np.full(J, L)
    desc = make_descriptor(regime, jl2, pt, meta, pt.shape[1], seed=seed)
    if release:
        rng = np.random.default_rng(seed + 5)
        desc.release = rng.integers(0, desc.horizon // 4 + 1,
                                    size=J).astype(float)
    return A.Case(jl2, pt, desc, f'small{seed}', regime)


def solve_full(case, budget=30):
    h = A.greedy_schedule(case)
    b = A.build_model(case, base=h, horizon=int(h[2].max()))
    b.add_hint(*h)
    return A.solve_built(b.model, b.extract, budget, WORKERS)


# ---------------------------------------------------------------------------
# the current model, rebuilt
# ---------------------------------------------------------------------------

class _Captured(Exception):
    pass


def _capture_family_solver_proto(case, ws):
    cap = {}
    orig = cp_model.CpSolver.Solve

    def patched(self, model, *a, **k):
        cap['proto'] = model.Proto().SerializeToString()
        raise _Captured

    cp_model.CpSolver.Solve = patched
    try:
        with pytest.raises(_Captured):
            cpsat_family.family_solver(case.jl, case.pt, case.desc,
                                       time_limit=1, warmstart=ws,
                                       num_workers=1)
    finally:
        cp_model.CpSolver.Solve = orig
    return cap['proto']


def needs_cache(path):
    """Skip when a warm-start schedule of the earlier campaign is absent."""
    return pytest.mark.skipif(
        not os.path.isfile(os.path.join(ROOT, path)),
        reason=f'needs the warm-start schedule {path} of the earlier campaign, which the '
               f'release does not hold')


@pytest.mark.parametrize('regime', [
    pytest.param(r, marks=needs_cache(f'or_solution/FAMILY/warmstart_cache/{r}/instance_000.npz'))
    for r in REGIMES_ALL])
def test_reference_model_proto_equals_family_solver(regime):
    case = A.load_case('10x25', regime, 'instance_000')
    z = np.load(f'or_solution/FAMILY/warmstart_cache/{regime}/instance_000.npz',
                allow_pickle=True)
    ws = (z['amch'], z['ct'])
    mine = A.build_reference_model(case, warmstart=ws)['model']
    assert mine.Proto().SerializeToString() == \
        _capture_family_solver_proto(case, ws)


def test_read_fjs_matches_repository_loader():
    saved = sys.argv
    sys.argv = saved[:1] + ['--device', 'cpu']
    try:
        from ppvc_instance_generator import load_instance
        for stem in ('data/PPVC/10x25+ppvc-mixed/instance_003',
                     'data/PPVC/50x25+ppvc-mixed/instance_000'):
            jl, pt, _ = load_instance(stem)
            jl2, pt2 = A.read_fjs(stem + '.fjs')
            assert np.array_equal(jl, jl2) and np.array_equal(pt, pt2)
    finally:
        sys.argv = saved


# ---------------------------------------------------------------------------
# exactness on small instances
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('regime', REGIMES_ALL)
@pytest.mark.parametrize('seed', [1, 2])
def test_exact_arms_reach_the_optimum_of_the_current_model(regime, seed):
    case = small_case(3, 6, seed, regime)
    h = A.greedy_schedule(case)
    ref = cpsat_family.family_solver(case.jl, case.pt, case.desc,
                                     time_limit=60, num_workers=WORKERS,
                                     warmstart=(h[0], h[2]))
    assert ref['status'] == 'OPTIMAL'
    for arm in EXACT_ARMS:
        r = A.run_arm(arm, case, h, 60, WORKERS, seed=1)
        assert r['status'] == 'OPTIMAL', arm
        assert r['objective'] == ref['objective'], arm
        assert not A.validate_all(case, *r['schedule']), arm
    for arm in RESTRICTIONS:
        r = A.run_arm(arm, case, h, 60, WORKERS, seed=1)
        assert r['objective'] >= ref['objective'], arm
        assert r['bound'] is None, arm          # not a bound of the problem
        assert not A.validate_all(case, *r['schedule']), arm


@pytest.mark.parametrize('regime', ['S', 'LS', 'SW', 'LSW'])
def test_exact_arms_agree_on_a_larger_setup_instance(regime):
    case = small_case(4, 7, 11, regime)
    h = A.greedy_schedule(case)
    ref = cpsat_family.family_solver(case.jl, case.pt, case.desc,
                                     time_limit=120, num_workers=WORKERS,
                                     warmstart=(h[0], h[2]))
    if ref['status'] != 'OPTIMAL':
        pytest.skip('current model not proven optimal within the limit')
    for arm in ('compact', 'compact_red'):
        r = A.run_arm(arm, case, h, 120, WORKERS, seed=1)
        assert r['status'] == 'OPTIMAL' and r['objective'] == ref['objective']


@pytest.mark.parametrize('regime', REGIMES_ALL)
def test_release_dates_exact_arms_agree_and_validate(regime):
    case = small_case(3, 6, 4, regime, release=True)
    h = A.greedy_schedule(case)
    assert not A.validate_all(case, *h)
    objs = {}
    for arm in ('ref', 'compact', 'compact_red'):
        r = A.run_arm(arm, case, h, 60, WORKERS, seed=1)
        assert r['status'] == 'OPTIMAL', arm
        assert not A.validate_all(case, *r['schedule']), arm
        objs[arm] = r['objective']
    assert len(set(objs.values())) == 1, objs


def _solver_schedule(res):
    return (np.asarray(res['assigned_mch'], dtype=int),
            np.rint(res['op_st']).astype(int), np.rint(res['op_ct']).astype(int))


@pytest.mark.parametrize('regime', ['N', 'LS', 'SW', 'LSW'])
def test_family_solver_models_release_dates(regime):
    """family_solver applies release dates (block of a job's first operation
    at or after its release date): its optimum passes both validators, equals
    the optimum of the compact arm, and its model equals the rebuilt one."""
    case = small_case(3, 6, 4, regime, release=True)
    assert case.release.max() > 0
    h = A.greedy_schedule(case)
    ws = (h[0], h[2])
    assert A.build_reference_model(case, warmstart=ws)['model'].Proto() \
        .SerializeToString() == _capture_family_solver_proto(case, ws)
    ref = cpsat_family.family_solver(case.jl, case.pt, case.desc,
                                     time_limit=60, num_workers=WORKERS)
    assert ref['status'] == 'OPTIMAL'
    amch, st, ct = _solver_schedule(ref)
    assert not A.validate_all(case, amch, st, ct)
    assert reference_sim.validate_schedule(case.jl, case.pt, case.desc,
                                           st, ct, amch) == []
    r = A.run_arm('compact', case, h, 60, WORKERS, seed=1)
    assert r['status'] == 'OPTIMAL' and r['objective'] == ref['objective']


def test_family_solver_release_beyond_the_release_free_horizon():
    """A release date larger than the horizon of the release-free model:
    the horizon grows by the latest release date, so the model stays
    feasible and the first operation of that job starts at or after it."""
    case = small_case(3, 6, 4, 'LS', release=True)
    rho = 10 * int(case.pt.max(axis=1).sum() + case.lag.sum()
                   + case.N * case.S.max())
    case.desc.release = np.r_[rho, np.zeros(case.J - 1)]
    case = A.Case(case.jl, case.pt, case.desc, case.name, case.regime)
    ref = cpsat_family.family_solver(case.jl, case.pt, case.desc,
                                     time_limit=60, num_workers=WORKERS)
    assert ref['status'] == 'OPTIMAL'
    amch, st, ct = _solver_schedule(ref)
    d = A.derive(case, amch, st, ct)
    assert d['block'][case.first[0]] >= rho
    assert not A.validate_all(case, amch, st, ct)


def test_pairwise_separation_relies_on_the_triangle_inequality():
    # every setup matrix of the benchmark satisfies it; the check exists
    for regime in ('S', 'LSW'):
        assert A.load_case('50x25', regime, 'instance_000').triangle_ok()


# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------

def _shift(sched, o, delta):
    amch, st, ct = (np.array(x) for x in sched)
    st[o] += delta
    ct[o] += delta
    return amch, st, ct


def test_strict_validator_rejects_setup_before_job_is_present():
    case = small_case(3, 6, 1, 'LS')
    h = A.greedy_schedule(case)
    d = A.derive(case, *h)
    # an operation whose block starts exactly at its job-ready time: pulling
    # it earlier by less than its setup keeps the processing start legal for
    # the reference check but starts the setup before the job arrives
    for o in range(case.N):
        if o == case.first[case.job[o]]:
            continue
        ready = h[2][o - 1] + case.lag[o - 1]
        if d['block'][o] == ready and d['setup'][o] >= 2:
            prev = d['prev'][o]
            if prev >= 0 and h[2][prev] > d['block'][o] - 1:
                continue
            bad = _shift(h, o, -1)
            if A.derive(case, *bad)['setup'][o] != d['setup'][o]:
                continue
            assert any('before ready' in v for v in A.validate_strict(case, *bad))
            # the repository's reference check applies the same rule to the
            # start of the setup block, so it rejects this schedule as well
            amch, st, ct = bad
            assert any('setup of op' in v for v in
                       reference_sim.validate_schedule(case.jl, case.pt,
                                                       case.desc, st, ct, amch))
            return
    pytest.skip('no suitable operation in this instance')


def test_strict_validator_rejects_outage_overlap_and_early_release():
    case = small_case(3, 6, 2, 'SW', release=True)
    h = A.greedy_schedule(case)
    assert not A.validate_strict(case, *h)
    j0 = case.first[0]
    rho = int(case.release[0])
    if rho > 0:
        bad = _shift(h, j0, -int(h[1][j0]))        # first op at time 0
        assert A.validate_strict(case, *bad)
    m = int(h[0][j0])
    if case.windows[m]:
        s, e = case.windows[m][0]
        amch, st, ct = (np.array(x) for x in h)
        p = case.pt[j0, m]
        st[j0], ct[j0] = s, s + p
        assert any('outage' in v for v in A.validate_strict(case, amch, st, ct))


# ---------------------------------------------------------------------------
# every arm in every regime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('regime', REGIMES_ALL)
def test_every_arm_returns_a_valid_schedule(regime):
    case = small_case(3, 6, 3, regime)
    h = A.greedy_schedule(case)
    hint_ms = int(h[2].max())
    for arm in A.ARMS:
        budget = 3 if arm == 'lns' else 20
        r = A.run_arm(arm, case, h, budget, WORKERS, seed=2)
        assert not A.validate_all(case, *r['schedule']), arm
        assert r['objective'] <= hint_ms, arm
        assert r['objective'] == float(np.max(r['schedule'][2]))


@pytest.mark.parametrize('regime', REGIMES_ALL)
def test_lns_never_worse_than_its_hint(regime):
    case = small_case(4, 8, 7, regime, release=(regime in ('LS', 'W')))
    h = A.greedy_schedule(case)
    r = A.lns(case, h, 4, WORKERS, seed=5)
    amch, st, ct = r['schedule']
    assert not A.validate_all(case, amch, st, ct)
    assert ct.max() <= h[2].max()
    ms = [p[1] for p in r['anytime']]
    assert ms == sorted(ms, reverse=True) and ms[0] == h[2].max()


@needs_cache('or_solution/FAMILY/warmstart_cache_50x25/LSW/instance_002.npz')
@pytest.mark.parametrize('kind', A.NEIGHBOURHOODS)
def test_lns_submodel_accepts_the_incumbent(kind):
    """The incumbent, fixed variable by variable, is a feasible complete
    solution of every sub-model (no rule is dropped or tightened)."""
    case = A.load_case('50x25', 'LSW', 'instance_002')
    hint, _ = A.load_hint_npz(
        'or_solution/FAMILY/warmstart_cache_50x25/LSW/instance_002.npz', case)
    rng = np.random.default_rng(1)
    d = A.derive(case, *hint)
    F = A._neighbourhood(kind, case, hint, d, 60, rng, case.machine_groups())
    t_low = int(min(d['block'][o] for o in F))
    frozen = [o for o in range(case.N) if hint[2][o] <= t_low]
    b = A.build_model(case, base=hint, free=F,
                      reseq=kind in ('reseq', 'station_reseq'), frozen=frozen,
                      horizon=int(hint[2].max()), secondary=True,
                      insert_radius=48)
    b.add_hint(*hint)
    s = cp_model.CpSolver()
    s.parameters.num_workers = 1
    s.parameters.max_time_in_seconds = 60
    s.parameters.fix_variables_to_their_hinted_value = True
    assert s.StatusName(s.Solve(b.model)) == 'OPTIMAL'
    got = b.extract(s)
    assert np.array_equal(got[1], hint[1]) and np.array_equal(got[2], hint[2])


@needs_cache('or_solution/FAMILY/warmstart_cache_50x25/SW/instance_000.npz')
def test_lns_functional_run_50x25_setup_regime():
    case = A.load_case('50x25', 'SW', 'instance_000')
    hint, meta = A.load_hint_npz(
        'or_solution/FAMILY/warmstart_cache_50x25/SW/instance_000.npz', case)
    r = A.run_arm('lns', case, hint, 60, WORKERS, seed=1)
    assert not A.validate_all(case, *r['schedule'])
    assert r['objective'] <= meta['makespan']
    assert r['wall_s'] < 75
    assert r['lns_stats'] and sum(s['n'] for s in r['lns_stats'].values()) > 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

@needs_cache('or_solution/FAMILY/warmstart_cache/S/instance_000.npz')   # --hint rule at 10x25
def test_runner_writes_skips_and_refuses(tmp_path):
    py = sys.executable
    base = [py, 'scripts/p85_cpsat_arms.py', '--arm', 'lns', '--size', '10x25',
            '--regimes', 'S', '--instances', '0', '--budget', '4',
            '--workers', '2', '--out', 't', '--root', str(tmp_path),
            '--seed', '1']
    cores = len(os.sched_getaffinity(0))
    if cores < 2:
        r = subprocess.run(base, capture_output=True, text=True)
        assert r.returncode != 0 and 'REFUSING' in (r.stdout + r.stderr)
    r = subprocess.run(base + ['--dev'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rec_p = tmp_path / '10x25_cparms' / 't' / 'lns' / 'S' / 'instance_000.json'
    rec = json.loads(rec_p.read_text())
    for k in ('instance', 'regime', 'budget_s', 'workers', 'warm_makespan',
              'status', 'objective', 'bound', 'solve_time', 'cpu_time',
              'cpu_share', 'affinity', 'loadavg', 'anytime', 'validated',
              'schedule_file'):
        assert k in rec, k
    assert rec['validated'] and rec['objective'] <= rec['warm_makespan']
    z = np.load(rec_p.parent / rec['schedule_file'])
    case = A.load_case('10x25', 'S', 'instance_000')
    assert not A.validate_all(case, z['amch'], z['st'], z['ct'])
    mtime = rec_p.stat().st_mtime
    r = subprocess.run(base + ['--dev'], capture_output=True, text=True)
    assert r.returncode == 0 and '0 solves' in r.stdout
    assert rec_p.stat().st_mtime == mtime
