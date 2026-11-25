"""Tests for scripts/p107_mask_rule_variants.py.

  * on hand-built cases with setups, the three rules admit the sets derived by hand at the
    first two decisions: the example of the supplement (a machine idle at 0, operation u
    ready at 0 with a setup of 5 h, operation v ready at 3 h with a setup of 1 h; nondelay
    admits only v, block only u, active both), and a case in which an outage pushes two
    blocks to the same start (block admits both, nondelay and active one);
  * without setups the block rule admits the nondelay pairs at every decision and gives the
    same schedules (a hand case with an outage, and stored cases of N and LW), while under
    setups it gives other schedules;
  * under nondelay the rollouts of this script return what P101.rule_rollout,
    P101.random_rollouts and P105.replay return, and the rule rollouts reproduce the stored
    exact-mask arrays on a few stored cases;
  * the difference arithmetic, the intervals, the pair counts and the case counts on
    synthetic case records (per combination and pooled with an instance's combinations
    kept together);
  * the run refuses a case whose stored array is not reproduced, and a block rule that
    differs from nondelay without setups; infer refuses a stored case that disagrees with
    the stored arrays;
  * the command line writes under --root and --out only, no written file holds an
    absolute path, and no wildcard path pattern of another file matches the new folder.

Run from the repository root:
  python -B -m pytest tests/test_p107_mask_rule_variants.py -q -p no:cacheprovider
CPU only.
"""
import copy
import fnmatch
import glob
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401  (parses argv at import)
    import data_utils  # noqa: E402,F401
    if 'p107_mask_rule_variants' in sys.modules:
        M = sys.modules['p107_mask_rule_variants']
    else:
        _spec = importlib.util.spec_from_file_location(
            'p107_mask_rule_variants', os.path.join(ROOT, 'scripts', 'p107_mask_rule_variants.py'))
        M = importlib.util.module_from_spec(_spec)
        sys.modules['p107_mask_rule_variants'] = M
        _spec.loader.exec_module(M)
finally:
    sys.argv = _argv
configs.device = 'cpu'
P101, P105 = M.P101, M.P105

import stats_family as SF  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

META = {'op_type': None, 'mch_type': None}
ARM = 'memetic_rule600'


@pytest.fixture(scope='module', autouse=True)
def _cpu():
    P101._cpu_torch()


def plain(n):
    return ConstraintDescriptor(active=(False, False, False), lag=np.zeros(n),
                                op_class=np.zeros(n), horizon=1)


def as_ref(amch, st, ct):
    return dict(amch=np.asarray(amch, int), st=np.asarray(st, int), ct=np.asarray(ct, int))


def legal_set(env):
    js, ms = np.nonzero(M.admitted(env)[0])
    return set(zip(js.tolist(), ms.tolist()))


# ---------------------------------------------------------------------------
# hand-built cases with setups
# ---------------------------------------------------------------------------

def setup_matrix():
    """Rows: the class a machine processed last (0..3, idle = 4); columns: the next class.
    From idle: 5 h into class 0, 1 h into classes 1 to 3. From class 0 into class 1: 1 h;
    from class 1 into class 0: 2 h; same class 0 h; every other change 1 h."""
    S = np.ones((5, 4))
    for c in range(4):
        S[c, c] = 0
    S[IDLE_CLASS] = [5, 1, 1, 1]
    S[0, 1], S[1, 0] = 1, 2
    return S


def supplement_case(window=False):
    """Three jobs of one operation each on two machines:
       u (job 0, class 0): machine 0, 2 h; released at 0
       v (job 1, class 1): machine 0, 2 h; released at 3 h
       w (job 2, class 2): machine 1, 1 h; released at 8 h
    Machine 0 is idle at 0, so u needs a setup of 5 h and v one of 1 h (the example of the
    supplement); w keeps machine 1 out of the first decision. With window, machine 0 is
    closed on [1, 6)."""
    jl = np.array([1, 1, 1])
    pt = np.array([[2, 0], [2, 0], [0, 1]])
    act = (False, True, bool(window))
    desc = ConstraintDescriptor(active=act, lag=np.zeros(3), op_class=np.array([0, 1, 2]),
                                setup=setup_matrix(),
                                windows=[[(1, 6)], []] if window else None,
                                horizon=20, release=np.array([0, 3, 8]))
    return jl, pt, desc


def test_supplement_example_first_two_decisions():
    """Decision 0. u on m0: block 0, processing start 5, completion 7. v on m0: block 3,
    start 4, completion 6. w on m1: block 8, start 9, completion 10. nondelay: t = 4, only
    v. block: smallest block start 0, only u. active: C* = 6, u (5) and v (4).
    Decision 1 after nondelay dispatches v (m0 free at 6, last class 1): u block 6, start
    8 (setup 2), completion 10; w start 9, completion 10. nondelay {u}, block {u} (block 6
    < 8), active {u, w} (C* = 10).
    Decision 1 after block dispatches u (setup [0, 5), processing [5, 7]; m0 last class 0):
    v block 7, start 8 (setup 1), completion 10; w block 8, start 9. nondelay {v}, block {v},
    active {v, w}."""
    jl, pt, desc = supplement_case()
    M_ = pt.shape[1]
    want0 = {'nondelay': {(1, 0)}, 'block': {(0, 0)}, 'active': {(0, 0), (1, 0)}}
    envs = {a: M.make_env(jl, pt, META, [desc], a) for a in M.ADM}
    for a in M.ADM:
        assert legal_set(envs[a]) == want0[a], a
    # decision 1 along nondelay's choice and along block's choice
    for first, want1 in (((1, 0), {'nondelay': {(0, 0)}, 'block': {(0, 0)},
                                   'active': {(0, 0), (2, 1)}}),
                         ((0, 0), {'nondelay': {(1, 0)}, 'block': {(1, 0)},
                                   'active': {(1, 0), (2, 1)}})):
        for a in M.ADM:
            env = M.make_env(jl, pt, META, [desc], a)
            env.step(np.array([first[0] * M_ + first[1]]))
            assert legal_set(env) == want1[a], (first, a)
    # block dispatches u: setup on [0, 5), processing [5, 7]
    env = envs['block']
    env.step(np.array([0]))
    assert env.true_op_ct[0, 0] == 7.0
    # the nondelay set computed from the state agrees with each rule's own mask where it
    # should: equal under nondelay, different at decision 0 under block and active
    for a in M.ADM:
        env = M.make_env(jl, pt, META, [desc], a)
        nd = M.nondelay_admitted(env)[0]
        js, ms = np.nonzero(nd)
        assert set(zip(js.tolist(), ms.tolist())) == {(1, 0)}
        assert np.array_equal(nd, M.admitted(env)[0]) == (a == 'nondelay')


def test_full_rollouts_of_the_supplement_example():
    """SPT under each rule, every schedule validated: nondelay v [4, 6], u [8, 10],
    w [9, 10], makespan 10; block u [5, 7], v [8, 10], w [9, 10], makespan 10."""
    jl, pt, desc = supplement_case()
    r = {a: M.rule_rollout('SPT', jl, pt, META, desc, a) for a in M.ADM}
    assert r['nondelay']['op_ct'].tolist() == [10, 6, 10]
    assert r['block']['op_ct'].tolist() == [7, 10, 10]
    assert r['nondelay']['makespan'] == r['block']['makespan'] == 10.0
    assert r['nondelay']['differ'] == 0 and r['block']['differ'] >= 1
    for a in M.ADM:
        assert P101.validate(jl, pt, desc, r[a]['amch'], r[a]['op_ct'], r[a]['makespan']) == []
        assert len(r[a]['counts']) == 3


def test_block_start_includes_the_push_past_an_outage():
    """Machine 0 closed on [1, 6). u: block [0, 7) overlaps it and moves to 6, processing
    start 11, completion 13. v: block [3, 6) overlaps it and moves to 6, start 7, completion
    9. w: start 9. nondelay: t = 7, only v. block: both blocks start at 6, u and v. active:
    C* = 9, only v (start 7; u at 11 and w at 9 are not earlier than 9)."""
    jl, pt, desc = supplement_case(window=True)
    want = {'nondelay': {(1, 0)}, 'block': {(0, 0), (1, 0)}, 'active': {(1, 0)}}
    for a in M.ADM:
        env = M.make_env(jl, pt, META, [desc], a)
        assert legal_set(env) == want[a], a
        assert np.rint(env.pair_free_time[0, 0, 0] / env._slope) == 11
        assert np.rint(env.pair_free_time[0, 1, 0] / env._slope) == 7
    env = M.make_env(jl, pt, META, [desc], 'block')
    env.step(np.array([0]))                       # u: block 6, processing [11, 13]
    assert env.true_op_ct[0, 0] == 13.0


# ---------------------------------------------------------------------------
# without setups block is nondelay
# ---------------------------------------------------------------------------

def tiny(window=False):
    """The hand case of tests/test_p101_mask_controls.py: two jobs of two operations on two
    machines; op0 (job 0) m0 2 h, m1 3 h; op1 (job 0) m1 1 h; op2 (job 1) m0 1 h; op3 (job 1)
    m0 2 h (3 h with the outage), m1 2 h. The outage case closes machine 1 on [0, 2)."""
    jl = np.array([2, 2])
    pt = np.array([[2, 3], [0, 1], [1, 0], [3 if window else 2, 2]])
    if window:
        desc = ConstraintDescriptor(active=(False, False, True), lag=np.zeros(4),
                                    op_class=np.zeros(4), windows=[[], [(0, 2)]], horizon=10)
    else:
        desc = plain(4)
    return jl, pt, desc


@pytest.mark.parametrize('window', [False, True])
def test_block_equals_nondelay_on_hand_cases_without_setups(window):
    jl, pt, desc = tiny(window)
    combo = 'W' if window else 'N'
    stored = {r: M.rule_rollout(r, jl, pt, META, desc, 'nondelay')['makespan'] for r in M.RULES}
    ref = rule_reference(jl, pt, META, desc, 'active')
    seeds = M.draw_seeds('10x25', 'instance_000')
    rec, arrays = M.evaluate_case(jl, pt, META, desc, combo, seeds, {'r': ref}, stored)
    assert rec['same_as_nondelay']['block']['any'] is False
    assert all(rec['adm']['block']['rules'][r]['differ_nondelay'] == 0 for r in M.RULES)
    assert np.array_equal(arrays['block_rule_ct'], arrays['nondelay_rule_ct'])
    assert np.array_equal(arrays['block_spt_pairs'], arrays['nondelay_spt_pairs'])


@pytest.mark.parametrize('combo', ['N', 'LW'])
def test_block_equals_nondelay_on_stored_cases_without_setups(combo):
    jl, pt, meta, desc = P105.load_instance_case('10x25', combo, 'instance_000')
    for r in M.RULES:
        a = M.rule_rollout(r, jl, pt, meta, desc, 'nondelay')
        b = M.rule_rollout(r, jl, pt, meta, desc, 'block')
        assert b['differ'] == 0
        assert np.array_equal(a['amch'], b['amch']) and np.array_equal(a['op_ct'], b['op_ct'])
        assert np.array_equal(a['counts'], b['counts'])
    seeds = M.draw_seeds('10x25', 'instance_000')
    x = M.random_rollouts(jl, pt, meta, desc, seeds[:3], 'nondelay')
    y = M.random_rollouts(jl, pt, meta, desc, seeds[:3], 'block')
    assert all(np.array_equal(u, v) for u, v in zip(x, y))


def test_block_differs_under_setups():
    jl, pt, meta, desc = P105.load_instance_case('10x25', 'S', 'instance_000')
    a = M.rule_rollout('SPT', jl, pt, meta, desc, 'nondelay')
    b = M.rule_rollout('SPT', jl, pt, meta, desc, 'block')
    assert b['differ'] > 0
    assert not np.array_equal(a['op_ct'], b['op_ct'])
    assert P101.validate(jl, pt, desc, b['amch'], b['op_ct'], b['makespan']) == []


# ---------------------------------------------------------------------------
# under nondelay the rollouts are those of p101 and p105
# ---------------------------------------------------------------------------

def rule_reference(jl, pt, meta, desc, adm='nondelay', rule='SPT'):
    r = M.rule_rollout(rule, jl, pt, meta, desc, adm)
    ct = np.rint(r['op_ct']).astype(int)
    return as_ref(r['amch'], ct - pt[np.arange(pt.shape[0]), r['amch']], ct)


@pytest.mark.parametrize('case', [('10x25', 'S', 'instance_000'),
                                  ('10x25', 'LSW', 'instance_001'),
                                  pytest.param(('pub10x5', 'LSW', None),
                                               marks=needs_public_instances())])
def test_nondelay_variants_return_what_p101_and_p105_return(case):
    s, combo, name = case
    name = name or P105.public_names()[2]
    jl, pt, meta, desc = P105.load_instance_case(s, combo, name)
    for r in M.RULES:
        a, b = M.rule_rollout(r, jl, pt, meta, desc, 'nondelay'), P101.rule_rollout(r, jl, pt,
                                                                                    meta, desc)
        assert a['makespan'] == b['makespan']
        assert np.array_equal(a['amch'], b['amch']) and np.array_equal(a['op_ct'], b['op_ct'])
        assert np.array_equal(a['counts'], b['counts']) and a['differ'] == 0
    seeds = M.draw_seeds(s, name)[:4]
    x = M.random_rollouts(jl, pt, meta, desc, seeds, 'nondelay')
    y = P101.random_rollouts(jl, pt, meta, desc, seeds)
    assert all(np.array_equal(u, v) for u, v in zip(x, y))
    # a reference outside the nondelay class (an active-rule schedule): every order
    ref = rule_reference(jl, pt, meta, desc, 'active', 'MWKR')
    for od in M.ORDERS:
        a, b = M.replay(jl, pt, meta, desc, ref, od, 'nondelay'), P105.replay(jl, pt, meta,
                                                                            desc, ref, od)
        for k in ('makespan', 'off', 'exact', 'decisions'):
            assert a[k] == b[k], (od, k)
        assert np.array_equal(a['amch'], b['amch']) and np.array_equal(a['op_ct'], b['op_ct'])


def test_random_streams_are_those_of_p101():
    for size in ('10x25', '50x25n30', '80x25n30'):
        for nm in ('instance_003', 'instance_029'):
            assert M.draw_seeds(size, nm) == P101.draw_seeds(size, nm)
    assert M.draw_seeds('pub10x5', '10x5+mix_007')[3] == [P101.RANDOM_SEED, 105, 7, 3]


@pytest.mark.parametrize('case', [('10x25', 'S', 'instance_000'),
                                  ('10x25', 'LSW', 'instance_002'),
                                  pytest.param(('pub10x5', 'SW', None),
                                               marks=needs_public_instances())])
def test_nondelay_rollouts_reproduce_the_stored_arrays(case):
    s, combo, name = case
    name = name or P105.public_names()[5]
    jl, pt, meta, desc = P105.load_instance_case(s, combo, name)
    stored = M.stored_rules(s, combo, name)
    assert sorted(stored) == sorted(M.RULES)
    for r in M.RULES:
        assert M.rule_rollout(r, jl, pt, meta, desc, 'nondelay')['makespan'] == stored[r]


# ---------------------------------------------------------------------------
# outcome arithmetic on synthetic case records
# ---------------------------------------------------------------------------

def synthetic(ms, rand, rep, ref, spt, differs=False, exact=(False, False, False)):
    """A case record with the fields the outcomes read. ms {adm: {rule: makespan}}, rand
    {adm: (mean, best)}, rep {adm: best replay}, spt {adm: (decisions, pairs, one,
    differ)}."""
    adm = {}
    for i, a in enumerate(M.ADM):
        d, p, o, f = spt[a]
        rules = {r: dict(makespan=float(ms[a][r]), decisions=d, pairs_sum=p, one_pair=o,
                         differ_nondelay=f) for r in M.RULES}
        adm[a] = dict(rules=rules, best_rule=float(min(ms[a].values())),
                      random=dict(mean=float(rand[a][0]), best=float(rand[a][1])),
                      replay={ARM: dict(best=float(rep[a]),
                                        A=dict(exact=bool(exact[i])))})
    return dict(adm=adm, references={ARM: dict(state='valid', makespan=float(ref),
                                               proved_optimal=False)},
                same_as_nondelay={'block': dict(any=differs, rules=['SPT'] if differs else [],
                                                random=2 if differs else 0, replay=[])})


def test_difference_arithmetic_on_synthetic_cells():
    """Two instances, two combinations. Instance a: nondelay best rule 100, block 90, active
    110 in the first combination; 200 / 200 / 190 in the second. Instance b: 50 / 55 / 50
    and 80 / 72 / 80. Block differences -10, 0, +10, -10 (mean -2.5), active +10, -5, 0, 0
    (mean +1.25); per combination the mean over the two instances; pooled the instance
    bootstrap of stats_family with the combinations of an instance kept together."""
    def rec(nd, bl, ac, rep, ref, differs, spt):
        ms = {a: {r: v + 3 * i for i, r in enumerate(M.RULES)} for a, v in
              (('nondelay', nd), ('block', bl), ('active', ac))}
        ms = {a: dict(v, FIFO=v['FIFO']) for a, v in ms.items()}
        rand = {a: (v + 20, v + 5) for a, v in (('nondelay', nd), ('block', bl), ('active', ac))}
        return synthetic(ms, rand, rep, ref, spt, differs,
                         exact=(False, False, rep['active'] == ref))
    spt = {'nondelay': (10, 20, 4, 0), 'block': (10, 30, 2, 5), 'active': (10, 60, 0, 10)}
    A = [[rec(100, 90, 110, {'nondelay': 120, 'block': 110, 'active': 100}, 100, True, spt),
          rec(200, 200, 190, {'nondelay': 210, 'block': 210, 'active': 205}, 200, False, spt)],
         [rec(50, 55, 50, {'nondelay': 60, 'block': 54, 'active': 50}, 50, True, spt),
          rec(80, 72, 80, {'nondelay': 88, 'block': 80, 'active': 80}, 80, True, spt)]]
    R = M.block_stats(A, [ARM])
    blk = np.array([[-10.0, 0.0], [10.0, -10.0]])
    act = np.array([[10.0, -5.0], [0.0, 0.0]])
    assert R['n'] == 2 and R['combos'] == 2 and R['cases'] == 4
    for x, d in (('block', blk), ('active', act)):
        g = R['diff'][x]['best_rule']
        ib = SF.instance_boot(d, B=M.B, seed=M.CI_SEED)
        assert g['mean'] == pytest.approx(d.mean())
        assert (g['lo'], g['hi']) == (pytest.approx(ib['lo']), pytest.approx(ib['hi']))
        assert g['c'] == 2
    assert R['diff']['block']['shorter_cases']['best_rule'] == 2
    assert R['diff']['block']['longer_cases']['best_rule'] == 1
    # the rules are offset by the same 0, 3, 6, 9 in every rule, so FIFO is the best rule
    assert R['diff']['block']['FIFO']['mean'] == pytest.approx(blk.mean())
    # the replay: differences from nondelay and gaps to the reference
    rep = {a: np.array(v, float) for a, v in (('nondelay', [[120, 210], [60, 88]]),
                                              ('block', [[110, 210], [54, 80]]),
                                              ('active', [[100, 205], [50, 80]]))}
    ref = np.array([[100, 200], [50, 80]], float)
    P = R['replay'][ARM]
    for a in M.ADM:
        assert P['gap'][a]['mean'] == pytest.approx((100 * (rep[a] - ref) / ref).mean())
    assert P['diff']['block']['mean'] == pytest.approx(
        (100 * (rep['block'] - rep['nondelay']) / rep['nondelay']).mean())
    assert P['exact']['active'] == 75.0 and P['exact']['nondelay'] == 0.0
    # pairs along SPT: four cases of 10 decisions each
    assert R['pairs']['nondelay'] == dict(decisions=40, mean=2.0, one_share=40.0,
                                          differ_share=0.0)
    assert R['pairs']['block']['mean'] == 3.0 and R['pairs']['block']['differ_share'] == 50.0
    assert R['pairs']['active']['one_share'] == 0.0
    # cases in which block differs from nondelay
    assert R['block_differs']['any'] == 3 and R['block_differs']['cases'] == 4
    assert R['block_differs']['random_schedules'] == 6
    # per combination (c = 1)
    R1 = M.block_stats([[A[0][1]], [A[1][1]]], [ARM])
    assert R1['diff']['block']['best_rule']['mean'] == pytest.approx(-5.0)
    assert R1['block_differs']['any'] == 1
    # the macro lines carry the numbers with one comment per line
    S = dict(word='Ten', rows={'S': dict(R1, label='S'),
                               'All': dict(R, label='All', combo_list=['N', 'S'])})
    lines = M.macro_lines(dict(partial=True, complete=True, sets={'10x25': S}))
    text = '\n'.join(lines)
    assert f'\\newcommand{{\\{M.PREFIX}DiffBlkBestRuleTenAll}}{{-2.50}}' in text
    assert f'\\newcommand{{\\{M.PREFIX}DiffBlkBestRuleTenS}}{{-5.00}}' in text
    assert f'\\newcommand{{\\{M.PREFIX}BlkDiffCasesTenAll}}{{3}}' in text
    for ln in lines:
        if ln.startswith('\\newcommand'):
            assert '  % ' in ln, ln
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names)) and all(re.fullmatch('Mrul[A-Za-z]+', x)
                                                 for x in names)


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refuses_a_case_whose_stored_array_is_not_reproduced():
    jl, pt, desc = tiny()
    stored = {r: M.rule_rollout(r, jl, pt, META, desc, 'nondelay')['makespan'] for r in M.RULES}
    seeds = M.draw_seeds('10x25', 'instance_000')[:2]
    rec, _ = M.evaluate_case(jl, pt, META, desc, 'N', seeds, {}, stored)
    assert rec['checks']['stored_rules'].startswith('the four nondelay')
    bad = dict(stored, SPT=stored['SPT'] + 1)
    with pytest.raises(SystemExit, match='stored array'):
        M.evaluate_case(jl, pt, META, desc, 'N', seeds, {}, bad)
    with pytest.raises(SystemExit, match='no stored'):
        M.evaluate_case(jl, pt, META, desc, 'N', seeds, {}, {'SPT': stored['SPT']})


def test_refuses_a_block_rule_that_differs_without_setups(monkeypatch):
    """A broken block rule that admits every compatible pair fails the built-in check."""
    jl, pt, desc = tiny()
    stored = {r: M.rule_rollout(r, jl, pt, META, desc, 'nondelay')['makespan'] for r in M.RULES}
    monkeypatch.setattr(M.BlockStartEnv, '_compute_unavailable_pairs',
                        lambda self: np.zeros_like(self.candidate_process_relation))
    with pytest.raises(SystemExit, match='without setups'):
        M.evaluate_case(jl, pt, META, desc, 'N', M.draw_seeds('10x25', 'instance_000')[:2],
                        {}, stored)


def test_infer_refuses_a_case_that_disagrees_with_the_stored_arrays(monkeypatch):
    stored = {'FIFO': 10.0, 'MOR': 11.0, 'SPT': 9.0, 'MWKR': 12.0}
    rec = dict(adm={'nondelay': dict(rules={r: dict(makespan=v) for r, v in stored.items()})},
               same_as_nondelay={'block': dict(any=False)}, setups=False)
    monkeypatch.setattr(M, 'stored_rules', lambda s, c, n: dict(stored))
    M.recheck(rec, '10x25', 'N', 'instance_000')
    monkeypatch.setattr(M, 'stored_rules', lambda s, c, n: dict(stored, MOR=12.0))
    with pytest.raises(M.Refused, match='stored array'):
        M.recheck(rec, '10x25', 'N', 'instance_000')
    monkeypatch.setattr(M, 'stored_rules', lambda s, c, n: dict(stored))
    bad = copy.deepcopy(rec)
    bad['same_as_nondelay']['block']['any'] = True
    with pytest.raises(M.Refused, match='without setups'):
        M.recheck(bad, '10x25', 'N', 'instance_000')


def test_infer_without_partial_needs_every_case(tmp_path):
    a = dict(root=str(tmp_path / 'cases'), out=str(tmp_path / 'out'), sets=['pub10x5'],
             refs=list(M.DEFAULT_REFS), combos=list(M.R8), n=None, partial=False,
             cpsat_arm=None)
    with pytest.raises(M.Refused, match='--partial'):
        M.infer(type('Args', (), a)())
    a['partial'] = True
    a['out'] = None
    with pytest.raises(M.Refused, match='--out'):
        M.infer(type('Args', (), a)())


# ---------------------------------------------------------------------------
# the command line: where it writes, what it stores
# ---------------------------------------------------------------------------

def _snapshot(patterns):
    out = {}
    for pat in patterns:
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if os.path.isfile(p):
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


@needs_public_instances()
def test_command_line_writes_only_under_root_and_out(tmp_path):
    names = P105.public_names()[:1]
    combos = ('N', 'S')
    frozen = ([os.path.join(P105.SETS['pub10x5']['data'], 'desc', c, f'{n}.desc.json')
               for c in combos for n in names]
              + [os.path.join('or_solution', 'PUBFAMILY', f'{P105.PUB_SET}_cpsat150s_w2', c,
                              f'{n}.json') for c in combos for n in names]
              + [os.path.join(P105.SETS['pub10x5']['stored'], f'Result_{r}_{c}.npy')
                 for r in M.RULES for c in combos]
              + [os.path.join(P105.DEFAULT_ROOT, 'pub10x5', 'cpsat150', c, f'{n}.json')
                 for c in combos for n in names])
    outs = [os.path.join(ROOT, p) for p in M.OUT_FILES.values()]
    default_root = os.path.join(ROOT, M.DEFAULT_ROOT)
    top = sorted(os.listdir(ROOT))
    before = _snapshot(frozen)
    had = [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs + [default_root]]
    assert len(before) == len(frozen)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PACT_ENV', None)
    root, out = str(tmp_path / 'cases'), str(tmp_path / 'out')
    script = os.path.join(ROOT, 'scripts', 'p107_mask_rule_variants.py')
    sel = ['--sets', 'pub10x5', '--n', '1', '--combos', ','.join(combos)]
    for argv in (['run', '--root', root] + sel,
                 ['infer', '--partial', '--root', root, '--out', out] + sel):
        p = subprocess.run([sys.executable, '-B', script] + argv, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=110)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert _snapshot(frozen) == before
    assert [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs + [default_root]] == had
    assert sorted(os.listdir(ROOT)) == top
    written = [os.path.join(dp, f) for dp, _, fs in os.walk(str(tmp_path)) for f in fs]
    cases = sorted(os.path.relpath(f, root) for f in written if f.startswith(root + os.sep))
    assert cases == sorted(os.path.join('pub10x5', c, f'{names[0]}.{e}')
                           for c in combos for e in ('json', 'npz'))
    for p in M.OUT_FILES.values():
        assert os.path.exists(os.path.join(out, p))
    assert all(f.startswith((root + os.sep, out + os.sep)) for f in written)
    for f in written:
        if f.endswith(('.json', '.md', '.tex')):
            txt = open(f).read()
            for bad in (ROOT, str(tmp_path), '/home/', '/tmp/'):
                assert bad not in txt, f'{f} holds {bad}'
    rec = json.load(open(os.path.join(root, 'pub10x5', 'S', f'{names[0]}.json')))
    assert rec['checks']['p105_replay'] == {'cpsat150': 'equal to the case of p105 (orders A, '
                                                        'B, C)'}
    assert all(not os.path.isabs(k) for k in rec['references']['cpsat150']['files'])
    recn = json.load(open(os.path.join(root, 'pub10x5', 'N', f'{names[0]}.json')))
    assert recn['same_as_nondelay']['block']['any'] is False
    with np.load(os.path.join(root, 'pub10x5', 'S', f'{names[0]}.npz')) as z:
        assert z['nondelay_rule_amch'].shape == (4, P105.load_instance_case(
            'pub10x5', 'S', names[0])[1].shape[0])
    tex = open(os.path.join(out, M.OUT_FILES['macros'])).read()
    for ln in tex.splitlines():
        if ln.startswith('\\newcommand'):
            assert '  % ' in ln, ln
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', tex)
    assert names_m and all(x.startswith(M.PREFIX) for x in names_m)
    assert len(names_m) == len(set(names_m))


def _scanned_files():
    out = []
    for top in ('.', 'scripts', 'tests', 'constraint_family', 'baselines', 'backbones', 'model'):
        d = os.path.join(ROOT, top)
        if not os.path.isdir(d):
            continue
        if top == '.':
            out += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.py', '.sh'))]
            continue
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if not x.startswith(('.', '__'))]
            out += [os.path.join(dp, f) for f in fns if f.endswith(('.py', '.sh'))]
    return sorted(set(os.path.normpath(p) for p in out))


def test_new_folder_is_read_by_no_other_file():
    own = {os.path.normpath(os.path.join(ROOT, p)) for p in (
        'scripts/p107_mask_rule_variants.py', 'tests/test_p107_mask_rule_variants.py')}
    folder = os.path.basename(M.DEFAULT_ROOT)
    examples = [f'{M.DEFAULT_ROOT}/pub10x5/LSW/10x5+mix_001.json',
                f'{M.DEFAULT_ROOT}/80x25n30/N/instance_003.npz',
                f'{M.DEFAULT_ROOT}/10x25/S/instance_099.json']
    rx = re.compile('test_results' + r'/FAMILY/[^\'"\s)]*')
    patterns = {}
    for f in _scanned_files():
        if f in own:
            continue
        txt = open(f, errors='replace').read()
        assert folder not in txt, f'{f} names {folder}'
        for m in rx.findall(txt):
            first = m.split('/')[2] if m.count('/') >= 2 else ''
            if any(ch in first for ch in '*?['):
                patterns.setdefault(m, f)
    assert patterns, 'the scan found no wildcard pattern; the scan itself is broken'
    for pat, f in patterns.items():
        for ex in examples:
            assert not fnmatch.fnmatch(ex, pat), f'{f}: {pat} matches {ex}'
