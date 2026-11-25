"""Tests for the CP-SAT reference runner of the public family
(scripts/p91_public_cpsat_refs.py).

  * on small public-style instances (unequal operation counts, partial
    eligibility, a machine no operation can use) with descriptors drawn by
    the public rules, the compact model and the model of
    cpsat_family.family_solver are proved optimal at the exhaustive-search
    optimum in all eight combinations; the exhaustive search enumerates every
    (job, machine) dispatch sequence with the reference simulator's
    semantics; every returned schedule passes both validators and its replay
    through the reference simulator is no longer; the optima are monotone
    along the combinations;
  * both validators accept public instances and reject a setup that starts
    before the job is present, an ineligible machine and an outage overlap;
  * the masked-rule warm start reproduces the stored makespans of a stored
    record; regime N of a stored instance is proved at its best-known optimum;
  * record validity: a proved optimum is valid whatever its CPU share, a
    solve stopped by the cap needs a share of 0.9, development runs and
    foreign affinities are never valid; scripts/timing_validity.py agrees;
  * the consistency checks catch a stored makespan below the bound, a
    combination below its sub-combination's bound and an N optimum that
    differs from a best-known optimum;
  * the runner writes atomic records, skips existing ones, refuses more
    workers than cores without --dev, writes no record when a validator
    fails, and obeys start, stop and hold conditions on decoy state files;
  * load_refs returns arrays in sorted-name order.

Run from the repository root:  python3 -m pytest tests/test_public_cpsat_refs.py -q
CPU only; CP-SAT runs with at most 2 workers.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

import p91_public_cpsat_refs as P  # noqa: E402
from constraint_family import cpsat_arms as A  # noqa: E402
from constraint_family import reference_sim  # noqa: E402
from constraint_family.cpsat_family import family_solver  # noqa: E402
from constraint_family.descriptor import IDLE_CLASS, REGIMES_ALL  # noqa: E402
from constraint_family.public_adapter import to_fjs_text  # noqa: E402
from constraint_family.public_rules import draw_rules  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

PY = sys.executable
CORE = sorted(os.sched_getaffinity(0))[0]
SET10 = 'data/PUBLIC/family/pub_SD2_test_10x5+mix'

# job_length, op_pt: unequal operation counts, partial eligibility, machine 3
# of TINY_A is eligible for no operation
TINY_A = (np.array([2, 1, 3]),
          np.array([[3, 5, 0, 0], [0, 4, 2, 0], [6, 0, 4, 0],
                    [2, 0, 0, 0], [0, 3, 5, 0], [4, 4, 0, 0]]))
TINY_B = (np.array([1, 2, 2, 1]),
          np.array([[4, 0, 3], [2, 5, 0], [0, 3, 3],
                    [5, 0, 2], [0, 4, 0], [3, 2, 0]]))


def tiny_rules(jl, pt):
    """Public rules of a tiny instance, the first seed whose lags, outages
    and setups are all non-trivial."""
    for seed in range(1, 500):
        r = draw_rules(jl, pt, seed)
        if r.lag.any() and sum(len(w) for w in r.windows) >= 2:
            return r
    raise AssertionError('no seed with lags and outages')


def brute_force(jl, pt, desc):
    """Optimum over every (job, machine) dispatch sequence. Each operation is
    appended to its machine and starts as early as the rules allow, as in
    reference_sim.simulate; every machine sequence of every feasible schedule
    is reached by some order, and left-shifting never lengthens a schedule,
    so the minimum is the optimum. The best sequence is replayed through the
    reference simulator."""
    jl = np.asarray(jl)
    N, M = pt.shape
    J = len(jl)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]])
    last = first + jl - 1
    lag = desc.lag if desc.has_lag else np.zeros(N)
    S = desc.setup if desc.has_setup else None
    W = desc.windows if desc.has_window else None
    cls = desc.op_class
    best = [np.inf, None]

    def go(nxt, ready, free, lastc, ms, acts):
        if len(acts) == N:
            if ms < best[0]:
                best[0], best[1] = ms, list(acts)
            return
        if ms >= best[0]:
            return
        for j in range(J):
            o = nxt[j]
            if o > last[j]:
                continue
            for m in np.flatnonzero(pt[o]):
                s = 0.0 if S is None else S[lastc[m], cls[o]]
                p = pt[o, m]
                b = max(ready[j], free[m])
                if W is not None:
                    b = reference_sim.push_past_outages_scalar(b, s + p, W[m])
                ct = b + s + p
                nx, rd, fr, lc = nxt.copy(), ready.copy(), free.copy(), lastc.copy()
                nx[j] += 1
                rd[j] = ct + lag[o]
                fr[m] = ct
                lc[m] = cls[o]
                acts.append((j, int(m)))
                go(nx, rd, fr, lc, max(ms, ct), acts)
                acts.pop()

    go(first.copy(), np.zeros(J), np.zeros(M), np.full(M, IDLE_CLASS), 0.0, [])
    sim = reference_sim.simulate(jl, pt, desc, best[1])
    assert sim['makespan'] == best[0]
    return best[0]


@pytest.fixture(scope='module')
def tiny_optima():
    out = {}
    for tag, (jl, pt) in (('A', TINY_A), ('B', TINY_B)):
        rules = tiny_rules(jl, pt)
        for r in REGIMES_ALL:
            desc = rules.descriptor(r)
            out[(tag, r)] = (jl, pt, desc, brute_force(jl, pt, desc))
    return out


@pytest.mark.parametrize('tag', ['A', 'B'])
def test_models_reach_the_exhaustive_optimum(tiny_optima, tag):
    for r in REGIMES_ALL:
        jl, pt, desc, opt = tiny_optima[(tag, r)]
        case = A.Case(jl, pt, desc, tag, r)
        hint = A.greedy_schedule(case)
        assert not A.validate_all(case, *hint)
        b = A.build_model(case, base=hint, horizon=int(hint[2].max()))
        b.add_hint(*hint)
        res = A.solve_built(b.model, b.extract, 30, 2, seed=1)
        assert res['status'] == 'OPTIMAL' and res['objective'] == opt, (tag, r)
        assert not A.validate_all(case, *res['schedule'])
        assert P.replay_makespan(case, *res['schedule']) <= opt
        fs = family_solver(jl, pt, desc, time_limit=30, num_workers=2)
        assert fs['status'] == 'OPTIMAL' and fs['objective'] == opt, (tag, r, fs)
        sched = (fs['assigned_mch'], fs['op_st'].astype(int), fs['op_ct'].astype(int))
        assert not A.validate_all(case, *sched)
        assert not reference_sim.validate_schedule(jl, pt, desc, fs['op_st'],
                                                   fs['op_ct'], fs['assigned_mch'])


def test_optima_are_monotone_in_the_rules(tiny_optima):
    for tag in ('A', 'B'):
        for r in REGIMES_ALL:
            for s in P.subsets(r):
                assert tiny_optima[(tag, s)][3] <= tiny_optima[(tag, r)][3], (tag, s, r)
        # the rules bind on these instances, so the check has teeth
        assert tiny_optima[(tag, 'LSW')][3] > tiny_optima[(tag, 'N')][3]


def test_validators_accept_public_schedules_and_reject_defects():
    jl, pt = TINY_A
    rules = tiny_rules(jl, pt)
    rng = np.random.default_rng(3)
    for r in REGIMES_ALL:
        desc = rules.descriptor(r)
        case = A.Case(jl, pt, desc, 'A', r)
        for _ in range(20):
            nxt, acts = case.first.copy(), []
            while len(acts) < case.N:
                j = int(rng.choice([j for j in range(case.J) if nxt[j] <= case.last[j]]))
                acts.append((j, int(rng.choice(case.elig[nxt[j]]))))
                nxt[j] += 1
            sim = reference_sim.simulate(jl, pt, desc, acts)
            sched = (sim['op_mch'], sim['op_st'].astype(int), sim['op_ct'].astype(int))
            assert not A.validate_all(case, *sched), r
    desc = rules.descriptor('LSW')
    case = A.Case(jl, pt, desc, 'A', 'LSW')
    amch, st, ct = A.greedy_schedule(case)
    bad = amch.copy()
    bad[3] = 1                                   # operation 3 runs on machine 0 only
    assert A.validate_all(case, bad, st, ct)
    # setup starting before the job is present: pull a non-first operation
    # back to its ready time while its setup is positive
    d = A.derive(case, amch, st, ct)
    o = next(o for o in range(case.N) if o != case.first[case.job[o]] and d['setup'][o] > 0)
    ready = ct[o - 1] + case.lag[o - 1]
    st2, ct2 = st.copy(), ct.copy()
    shift = st2[o] - ready
    st2[o] -= shift
    ct2[o] -= shift
    assert A.validate_strict(case, amch, st2, ct2)
    # outage overlap: an operation placed inside the first outage of its machine
    m = int(amch[0])
    w = [(s, e) for (s, e) in desc.windows[m]]
    if w:
        s0 = w[0][0]
        st3, ct3 = st.copy(), ct.copy()
        p = pt[0, m]
        st3[0], ct3[0] = s0, s0 + p
        assert A.validate_all(case, amch, st3, ct3)


@needs_public_instances()
def test_warm_start_reproduces_stored_rules_and_N_optimum():
    names = P.set_names(SET10)
    i = names.index('10x5+mix_052')
    stored = P.load_stored('test_results/PUBFAMILY/pub_SD2_test_10x5+mix_xmask', ['N', 'S'],
                           len(names))
    bk = P.load_best_known(P.BEST_KNOWN['pub_SD2_test_10x5+mix'], names)
    for r in ('N', 'S'):
        case, _ = P.load_case(SET10, r, names[i])
        hint, meta = P.make_hint(case, stored[r], i)
        assert meta['stored_rules_match']
        assert meta['warm_makespan'] == min(float(stored[r][k][i]) for k in P.RULES)
        assert not A.validate_all(case, *hint)
    case, _ = P.load_case(SET10, 'N', names[i])
    hint, _ = P.make_hint(case)
    b = A.build_model(case, base=hint, horizon=int(hint[2].max()))
    b.add_hint(*hint)
    res = A.solve_built(b.model, b.extract, 40, 2, seed=1)
    assert bk[names[i]][2] and res['status'] == 'OPTIMAL'
    assert res['objective'] == bk[names[i]][0] == 415
    # a tampered stored makespan is refused
    bad = {k: v.copy() for k, v in stored['N'].items()}
    bad['SPT'][i] += 1
    with pytest.raises(ValueError, match='reproduce'):
        P.make_hint(case, bad, i)
    with pytest.raises(ValueError, match='no stored array'):
        P.make_hint(case, {}, i)


def _rec(**kw):
    d = {'validated': True, 'development_run': False, 'affinity': [0, 1],
         'cores': [0, 1], 'status': 'FEASIBLE', 'cpu_share': 0.97}
    d.update(kw)
    return d


def test_record_validity_and_timing_validity_agree(tmp_path):
    assert P.record_validity(_rec())[:1] == (True,)
    v = P.record_validity(_rec(status='OPTIMAL', cpu_share=0.41))
    assert v[0] and 'proved' in v[1] and v[2] is False
    assert not P.record_validity(_rec(cpu_share=0.85))[0]
    assert not P.record_validity(_rec(development_run=True, status='OPTIMAL'))[0]
    assert not P.record_validity(_rec(affinity=[0, 1, 2]))[0]
    assert not P.record_validity(_rec(validated=False))[0]
    import timing_validity as TV
    d = tmp_path / 'f' / 'N'
    d.mkdir(parents=True)
    for nm, rec in (('a', _rec(status='OPTIMAL', cpu_share=0.41, hint='complete')),
                    ('b', _rec(cpu_share=0.97, hint='complete'))):
        (d / f'{nm}.json').write_text(json.dumps(rec))
    got = TV.classify([str(d / 'a.json'), str(d / 'b.json')], slots=[[0, 1]],
                      thresh=str(tmp_path / 'none.json'))
    assert got[str(d / 'a.json')] == (True, 'optimal')
    assert got[str(d / 'b.json')] == (True, 'share')


def test_consistency_checks_catch_disagreements():
    rec = {'N': {'objective': 300.0, 'bound': 300.0, 'proved_optimal': True},
           'S': {'objective': 500.0, 'bound': 310.0, 'proved_optimal': False}}
    bk = {'x': (300.0, 300.0, True)}
    stored = {'N': {'SPT': np.array([0.0, 320.0])}, 'S': {'MWKR': np.array([0.0, 520.0])}}
    assert P.instance_violations('x', 1, rec, stored, bk) == []
    assert P.instance_violations('x', 1, rec, stored, {'x': (301.0, 290.0, True)})
    low = {'S': {'MWKR': np.array([0.0, 305.0])}}       # below the S bound 310
    assert any('below the CP-SAT bound' in v for v in P.instance_violations('x', 1, rec, low, bk))
    rec2 = dict(rec, S={'objective': 290.0, 'bound': 200.0, 'proved_optimal': False})
    assert any('only adds rules' in v for v in P.instance_violations('x', 1, rec2, {}, {}))
    # a stored S makespan below the N optimum is caught through the N bound
    low2 = {'S': {'MWKR': np.array([0.0, 299.0])}}
    rec3 = dict(rec, S={'objective': 500.0, 'bound': 200.0, 'proved_optimal': False})
    assert P.instance_violations('x', 1, rec3, low2, bk)


def test_state_file_gates(tmp_path):
    f = tmp_path / 'state.txt'
    f.write_text('IDLE pass 0 | written 2026-11-12 00:10:00\n')
    start = (str(f), 'LOADING pass 1 training start')
    stop = (str(f), 'IDLE pass 3 done')
    assert not P.state_is(start) and not P.state_is(stop)

    def later():
        time.sleep(0.6)
        with open(f, 'a') as g:
            g.write('LOADING pass 1 training start (model x) | written 2026-11-12 00:31:00\n')
            g.write('NOTE the lines below this one are remarks\n\n')

    th = threading.Thread(target=later)
    t0 = time.time()
    th.start()
    assert P.wait_for_start(start, stop, poll=0.05, log=lambda m: None)
    th.join()
    assert time.time() - t0 >= 0.5
    assert P.last_state(str(f)).startswith('LOADING pass 1')
    # an old start line followed by another state does not start the runner; a
    # start line appended while waiting does, even when the state moves on at once
    f.write_text('LOADING pass 1 training start (old) | written ...\nIDLE pass 1 done | x\n')

    def passing():
        time.sleep(0.6)
        with open(f, 'a') as g:
            g.write('LOADING pass 1 training start (new) | written ...\n')
            g.write('TRAINING pass 1 | written ...\n')

    th = threading.Thread(target=passing)
    t0 = time.time()
    th.start()
    assert P.wait_for_start(start, stop, poll=0.05, log=lambda m: None)
    th.join()
    assert time.time() - t0 >= 0.5
    f.write_text('LOADING pass 3 x | written ...\nIDLE pass 3 done | written ...\nNOTE ok\n')
    assert P.state_is(stop)
    assert not P.wait_for_start(start, stop, poll=0.05, log=lambda m: None)
    assert P.last_state(str(tmp_path / 'missing.txt')) is None
    hold = tmp_path / 'HOLD'
    hold.write_text('')
    f.write_text('TRAINING pass 2 | written ...\n')
    threading.Timer(0.5, hold.unlink).start()
    t0 = time.time()
    assert P.wait_while_hold(str(hold), stop, poll=0.05, log=lambda m: None)
    assert time.time() - t0 >= 0.4
    hold.write_text('')
    f.write_text('IDLE pass 3 done | written ...\n')
    assert not P.wait_while_hold(str(hold), stop, poll=0.05, log=lambda m: None)


def test_code_fingerprint_follows_the_source_files(tmp_path):
    f = tmp_path / 'module.py'
    f.write_text('x = 1\n')
    a = P.code_fingerprint([str(f)])
    assert P.code_fingerprint([str(f)]) == a
    f.write_text('x = 2\n')
    assert P.code_fingerprint([str(f)]) != a
    assert all(os.path.exists(P.in_root(x)) for x in P.CODE_FILES)


def test_atomic_write_keeps_the_old_record(tmp_path, monkeypatch):
    p = tmp_path / 'r.json'
    P.write_json_atomic(str(p), {'a': 1})

    class Boom(Exception):
        pass

    def bad_dump(obj, f):
        f.write('{"a": 2,')
        raise Boom

    monkeypatch.setattr(P.json, 'dump', bad_dump)
    with pytest.raises(Boom):
        P.write_json_atomic(str(p), {'a': 2})
    assert json.loads(p.read_text()) == {'a': 1}
    assert [x.name for x in tmp_path.iterdir()] == ['r.json']


def make_tiny_set(root):
    """Family set of two tiny public-style instances in a temporary folder;
    base instances are named by absolute stems."""
    sd = root / 'fam' / 'tinyset'
    for k, (jl, pt) in enumerate((TINY_A, TINY_B), 1):
        nm = f'tiny_{k:03d}'
        (root / f'{nm}.fjs').write_text(to_fjs_text(jl, pt))
        rules = tiny_rules(jl, pt)
        for r in REGIMES_ALL:
            d = rules.descriptor(r).to_json()
            d.update(base_instance=str(root / nm), base_format='fjs', regime=r,
                     desc_seed=rules.seed, rules_version='public-rules-1')
            (sd / 'desc' / r).mkdir(parents=True, exist_ok=True)
            (sd / 'desc' / r / f'{nm}.desc.json').write_text(json.dumps(d))
    return sd


def _run(args):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([PY, 'scripts/p91_public_cpsat_refs.py'] + args,
                          capture_output=True, text=True, env=env, timeout=240)


def test_runner_writes_skips_refuses_and_stops(tmp_path):
    make_tiny_set(tmp_path)
    out = tmp_path / 'out'
    base = ['--set', 'tinyset', '--data_root', str(tmp_path / 'fam'), '--stored', 'none',
            '--best_known', 'none', '--budget', '10', '--workers', '1',
            '--cores', str(CORE), '--out', str(out), '--hold_file', 'none',
            '--poll', '0.05', '--regimes', 'S', 'N']
    r = _run(base + ['--workers', '2'])
    assert r.returncode != 0 and 'REFUSING' in r.stdout + r.stderr
    state = tmp_path / 'state.txt'
    state.write_text('IDLE pass 3 done | written ...\nNOTE remark\n')
    r = _run(base + ['--stop_when', str(state), 'IDLE pass 3 done'])
    assert r.returncode == 0 and not (out / 'N').exists(), r.stdout + r.stderr
    r = _run(base + ['--phase', 'plan'])
    assert r.returncode == 0 and '4 solves to run' in r.stdout and not out.exists()
    assert 'in run order: N 2, S 2' in r.stdout
    r = _run(base + ['--phase', 'plan', '--order', 'S', 'L', 'W', 'LW', 'N', 'LS', 'SW', 'LSW'])
    assert r.returncode == 0 and 'in run order: S 2, N 2' in r.stdout
    r = _run(base + ['--phase', 'plan', '--order', 'S', 'S', 'W', 'LW', 'N', 'LS', 'SW', 'LSW'])
    assert r.returncode != 0 and 'REFUSING' in r.stdout + r.stderr
    state.write_text('LOADING pass 1 training start | written ...\n')
    r = _run(base + ['--start_when', str(state), 'LOADING pass 1 training start',
                     '--stop_when', str(state), 'IDLE pass 3 done'])
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'done: 4 records written' in r.stdout
    order = [ln.split()[3] for ln in r.stdout.splitlines() if 'solving (child' in ln]
    assert order == ['N', 'N', 'S', 'S']             # N before S whatever the option order
    rec_p = out / 'N' / 'tiny_001.json'
    rec = json.loads(rec_p.read_text())
    for k in ('set', 'instance', 'regime', 'objective', 'bound', 'status', 'proved_optimal',
              'time_to_last_improvement_s', 'solve_time_s', 'anytime', 'warm_makespan',
              'warm_source', 'budget_s', 'workers', 'seed', 'ortools', 'model_size',
              'cpu_share', 'cpu_build_s', 'affinity', 'loadavg', 'loadavg_end',
              'peak_rss_gb', 'started', 'ended', 'valid', 'valid_reason', 'timing_valid'):
        assert k in rec, k
    assert rec['proved_optimal'] and rec['valid'] and rec['affinity'] == [CORE]
    assert rec['code_sha'] == P.code_fingerprint()
    assert 'complete and is feasible' in rec['hint_log'][0]
    assert not any(p.name.endswith('.tmp') or '.tmp' in p.name for p in out.rglob('*'))
    mtime = rec_p.stat().st_mtime
    r = _run(base)
    assert r.returncode == 0 and '0 solves' in r.stdout and rec_p.stat().st_mtime == mtime
    r = _run(['--phase', 'check'] + base)
    assert r.returncode == 0 and 'violations: 0' in r.stdout, r.stdout
    refs = P.load_refs(str(out), data_root=str(tmp_path / 'fam'))
    assert refs['N']['names'] == ['tiny_001', 'tiny_002']
    assert refs['N']['proved'].all() and refs['N']['valid'].all()
    assert np.isnan(refs['L']['objective']).all() and not refs['L']['valid'].any()
    assert (refs['S']['bound_mono'] >= refs['N']['bound']).all()


def test_failed_validator_writes_no_record(tmp_path, monkeypatch):
    sd = make_tiny_set(tmp_path)
    case, _ = P.load_case(str(sd), 'LSW', 'tiny_001')
    hint = A.greedy_schedule(case)
    calls = {'n': 0}
    real = A.validate_all

    def second_call_fails(*a):
        calls['n'] += 1
        return real(*a) if calls['n'] == 1 else ['[strict] planted violation']

    monkeypatch.setattr(A, 'validate_all', second_call_fails)
    out = tmp_path / 'out'
    t = {'set': 'tinyset', 'set_dir': str(sd), 'name': 'tiny_001', 'index': 0,
         'regime': 'LSW', 'budget': 5.0, 'workers': 1, 'seed': 1, 'model': 'compact',
         'cores': [CORE], 'dev': False, 'out_dir': str(out),
         'out': str(out / 'LSW' / 'tiny_001.json'),
         'hint': [x.tolist() for x in hint],
         'meta': {'warm_makespan': float(hint[2].max()), 'warm_rule': 'greedy',
                  'warm_source': 'test', 'warm_all_rules': {}}}
    (out / 'LSW').mkdir(parents=True)
    P.solve_task(t)
    assert calls['n'] == 2
    assert not (out / 'LSW' / 'tiny_001.json').exists()
    assert 'planted violation' in (out / P.FAIL_LOG).read_text()


def test_load_refs_order(tmp_path):
    names = ['b_002', 'a_001', 'c_003']
    for nm, obj in (('c_003', 30.0), ('a_001', 10.0)):
        d = tmp_path / 'N'
        d.mkdir(exist_ok=True)
        P.write_json_atomic(str(d / f'{nm}.json'), {
            'set': 's', 'instance': nm, 'objective': obj, 'bound': obj - 1,
            'status': 'FEASIBLE', **_rec()})
    refs = P.load_refs(str(tmp_path), names=sorted(names))
    assert refs['N']['names'] == ['a_001', 'b_002', 'c_003']
    assert refs['N']['objective'][0] == 10 and np.isnan(refs['N']['objective'][1])
    assert refs['N']['objective'][2] == 30
    assert list(refs['N']['valid']) == [True, False, True]
    assert refs['LSW']['bound_mono'][2] == 29
