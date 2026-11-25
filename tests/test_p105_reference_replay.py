"""Tests for scripts/p105_reference_replay.py.

  * the three priority orders pick the pair their keys name on a hand-built set of legal
    pairs (each order a different pair, and the tie rules in order), and on a hand-built
    case with two machines and three jobs give the schedules computed by hand (order B
    shorter than A and C);
  * a reference built under the mask (a rule schedule, also on an instance with an outage
    and on a public instance) is reproduced exactly by orders A and B, passes the class
    test, and no decision leaves it;
  * a reference with a deliberate idle period is not reproduced, fails the class test at
    the operation held idle, and every replayed schedule stays feasible;
  * the gap arithmetic, the shares and the intervals on synthetic arrays and on synthetic
    case files (per combination, pooled with an instance's combinations kept together, and
    the cases that do not count);
  * a reference whose stored objective disagrees with its schedule, whose durations
    disagree with the processing times, or whose record states two makespans, is refused;
  * the command line writes under --root and --out only, no written file holds an
    absolute path, and no wildcard path pattern of another file matches the new folder.

Run from the repository root:
  python -B -m pytest tests/test_p105_reference_replay.py -q -p no:cacheprovider
CPU only.
"""
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
    if 'p105_reference_replay' in sys.modules:
        P105 = sys.modules['p105_reference_replay']
    else:
        _spec = importlib.util.spec_from_file_location(
            'p105_reference_replay', os.path.join(ROOT, 'scripts', 'p105_reference_replay.py'))
        P105 = importlib.util.module_from_spec(_spec)
        sys.modules['p105_reference_replay'] = P105
        _spec.loader.exec_module(P105)
finally:
    sys.argv = _argv
configs.device = 'cpu'
P101 = P105.P101

import stats_family as SF  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

META = {'op_type': None, 'mch_type': None}


@pytest.fixture(scope='module', autouse=True)
def _cpu():
    P101._cpu_torch()


def plain(n):
    return ConstraintDescriptor(active=(False, False, False), lag=np.zeros(n),
                                op_class=np.zeros(n), horizon=1)


def as_ref(amch, st, ct):
    return dict(amch=np.asarray(amch, int), st=np.asarray(st, int), ct=np.asarray(ct, int))


# ---------------------------------------------------------------------------
# the priority orders
# ---------------------------------------------------------------------------

def test_orders_pick_the_pair_their_keys_name():
    """Three legal pairs at one decision, operations 0, 1, 2 of jobs 0, 1, 2:
       pair 0: (job 0, machine 0), s_R 2, c_R 9, reference machine 1 (off), completion 5
       pair 1: (job 1, machine 0), s_R 3, c_R 8, reference machine 0,       completion 6
       pair 2: (job 2, machine 1), s_R 5, c_R 6, reference machine 1,       completion 4
    A takes the smallest reference start (pair 0), B the reference machine first and then
    the smallest start (pair 1), C the smallest reference completion (pair 2)."""
    ref = as_ref([1, 0, 1], [2, 3, 5], [9, 8, 6])
    js, ms, ops = np.array([0, 1, 2]), np.array([0, 0, 1]), np.array([0, 1, 2])
    compl = np.array([5.0, 6.0, 4.0])
    assert [P105.choose(o, js, ms, ops, compl, ref) for o in 'ABC'] == [0, 1, 2]


def test_tie_rules_in_order():
    """One operation legal on three machines, reference machine 2: the reference machine
    first in A and C. Among pairs off the reference machine: the earlier completion, then
    the lower machine index, then the lower job index."""
    ref = as_ref([2, 0, 0], [4, 4, 4], [7, 7, 7])
    js, ms, ops = np.array([0, 0, 0]), np.array([0, 1, 2]), np.array([0, 0, 0])
    compl = np.array([6.0, 6.0, 9.0])
    assert P105.choose('A', js, ms, ops, compl, ref) == 2
    assert P105.choose('C', js, ms, ops, compl, ref) == 2
    js, ms, ops = np.array([0, 0]), np.array([0, 1]), np.array([0, 0])
    assert P105.choose('A', js, ms, ops, np.array([6.0, 5.0]), ref) == 1    # completion
    assert P105.choose('A', js, ms, ops, np.array([6.0, 6.0]), ref) == 0    # machine index
    js, ms, ops = np.array([2, 1]), np.array([1, 1]), np.array([2, 1])     # same start
    assert P105.choose('B', js, ms, ops, np.array([6.0, 6.0]), ref) == 1    # job index


def three_jobs():
    """Three jobs of one operation each on two machines (0 = incompatible):
       op 0 (job 0): machine 1 only, 4 h
       op 1 (job 1): machine 0 5 h, machine 1 2 h
       op 2 (job 2): machine 0 only, 4 h
    Reference R: op 0 on m1 [0, 4], op 1 on m1 [4, 6], op 2 on m0 [5, 9] (m0 idle on
    [0, 5]); makespan 9."""
    jl = np.array([1, 1, 1])
    pt = np.array([[0, 4], [5, 2], [4, 0]])
    return jl, pt, plain(3), as_ref([1, 1, 0], [0, 4, 5], [4, 6, 9])


def test_two_machine_case_by_hand():
    """At t = 0 every pair is legal; all three orders take op 0 on m1 (reference start 0,
    completion 4, the smallest key of each). Then m1 is free at 4 and m0 at 0, so the
    legal pairs are op 1 on m0 and op 2 on m0.
      A: op 1 (reference start 4 < 5) on m0 [0, 5], off its reference machine; op 2 on m0
         [5, 9]. Makespan 9, one decision off the reference.
      B: op 2 (on its reference machine) on m0 [0, 4]; op 1 starts at 4 on both machines,
         takes m1 [4, 6]. Makespan 6, no decision off the reference.
      C: as A (reference completion 6 < 9). Makespan 9.
    The class test: op 1 could start at 0 on m0 (an earlier pair) and op 2 could start at
    0 on m0 (later than its earliest start), so R is not in the class."""
    jl, pt, desc, ref = three_jobs()
    rec, sched = P105.replay_case(jl, pt, META, desc, ref)
    assert {o: rec['replay'][o]['makespan'] for o in 'ABC'} == {'A': 9.0, 'B': 6.0, 'C': 9.0}
    assert {o: rec['replay'][o]['off'] for o in 'ABC'} == {'A': 1, 'B': 0, 'C': 1}
    assert sched['A'][0].tolist() == [1, 0, 0] and sched['A'][1].tolist() == [4, 5, 9]
    assert sched['B'][0].tolist() == [1, 1, 0] and sched['B'][1].tolist() == [4, 6, 4]
    assert rec['replay_makespan'] == 6.0 and rec['best_order'] == 'B'
    assert not rec['in_class'] and not rec['exact_A']
    assert rec['class_failures'] == {'later': 1, 'earlier_pair': 1}
    assert rec['class_first_failure']['op'] == 1
    assert rec['class_first_failure']['kind'] == 'an earlier pair'
    assert rec['off_share_A'] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# references inside and outside the class
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


def rule_reference(jl, pt, meta, desc, rule='SPT'):
    r = P101.rule_rollout(rule, jl, pt, meta, desc)
    ct = np.rint(r['op_ct']).astype(int)
    return as_ref(r['amch'], ct - pt[np.arange(pt.shape[0]), r['amch']], ct), r['makespan']


@pytest.mark.parametrize('window', [False, True])
def test_a_reference_built_under_the_mask_is_reproduced(window):
    jl, pt, desc = tiny(window)
    ref, ms = rule_reference(jl, pt, META, desc)
    rec, sched = P105.replay_case(jl, pt, META, desc, ref)
    assert rec['in_class'] and rec['exact_A'] and rec['replay']['B']['exact']
    assert rec['replay']['A']['off'] == 0 and rec['replay']['B']['off'] == 0
    assert rec['replay_makespan'] == ms
    assert np.array_equal(sched['A'][0], ref['amch'])
    assert np.array_equal(sched['A'][1], ref['ct'])


@needs_public_instances()
def test_a_public_rule_schedule_is_reproduced():
    jl, pt, meta, desc = P105.load_instance_case('pub10x5', 'LSW', P105.public_names()[2])
    for rule in ('SPT', 'MWKR'):
        ref, ms = rule_reference(jl, pt, meta, desc, rule)
        rec, _ = P105.replay_case(jl, pt, meta, desc, ref)
        assert rec['in_class'] and rec['exact_A'] and rec['replay']['A']['off'] == 0
        assert rec['replay']['A']['makespan'] == ms


def test_a_reference_with_an_idle_period_is_not_reproduced():
    """R: op2 on m0 [0, 1], op0 on m1 [1, 4] (m1 idle on [0, 1]), op3 on m0 [1, 3], op1 on
    m1 [4, 5]; makespan 5. Every order places op2 on m0 [0, 1], then op0 on m1 [0, 3] (the
    only pair that starts at 0), op3 on m0 [1, 3] and op1 on m1 [3, 4]: makespan 4, the
    machine sequences of R, but op0 one hour earlier. The class test fails at op0, later
    than its earliest start; no other operation fails."""
    jl, pt, desc = tiny()
    ref = as_ref([1, 1, 0, 0], [1, 4, 0, 1], [4, 5, 1, 3])
    assert P101.validate(jl, pt, desc, ref['amch'], ref['ct'].astype(float), 5.0) == []
    rec, sched = P105.replay_case(jl, pt, META, desc, ref)
    assert not rec['exact_A'] and not rec['in_class']
    assert rec['class_failures'] == {'later': 1, 'earlier_pair': 0}
    assert rec['class_first_failure']['op'] == 0
    assert rec['class_first_failure']['kind'] == 'later than its earliest start'
    for o in 'ABC':
        assert rec['replay'][o]['makespan'] == 4.0
        assert rec['replay'][o]['off'] == 0
        amch, ct = sched[o]
        assert P101.validate(jl, pt, desc, amch, ct, float(ct.max())) == []
        assert amch.tolist() == [1, 1, 0, 0] and ct.tolist() == [3, 4, 1, 3]


# ---------------------------------------------------------------------------
# gap arithmetic and shares
# ---------------------------------------------------------------------------

def test_gap_arithmetic_and_shares_on_synthetic_arrays():
    """Gaps 0, +1, +20, -5 %: mean 4; at or below 2 of 4; within 1% 3 of 4 (0, 1 and -5);
    above 5% 1 of 4. Rule gaps 10, 10, 10, 0: mean 7.5. Shorter of replay and rule: 100,
    202, 55, 76, gaps 0, 1, 10, -5: mean 1.5. Intervals: stats_family.instance_boot."""
    ref = np.array([100.0, 200.0, 50.0, 80.0])
    rep = np.array([100.0, 202.0, 60.0, 76.0])
    rule = np.array([110.0, 220.0, 55.0, 80.0])
    C = P105.cell_stats(ref, rep, rule, np.array([True, False, False, True]),
                        np.array([0.0, 0.5, 0.25, 0.0]), {'A': rep, 'B': rep + 1, 'C': rep})
    g = np.array([0.0, 1.0, 20.0, -5.0])
    assert C['gap']['mean'] == pytest.approx(4.0)
    ib = SF.instance_boot(g, B=P105.B, seed=7)
    assert (C['gap']['lo'], C['gap']['hi']) == (pytest.approx(ib['lo']), pytest.approx(ib['hi']))
    assert C['shares'] == dict(at_or_below=50.0, within_one=75.0, above_five=25.0, n=4)
    assert C['buildable'] == 50.0 and C['left'] == pytest.approx(18.75)
    assert C['rule_gap']['mean'] == pytest.approx(7.5)
    assert C['bound_gap']['mean'] == pytest.approx(1.5)
    assert C['bound_shares']['within_one'] == 75.0
    assert C['orders']['A']['gap'] == pytest.approx(4.0)
    assert C['ref_ms'] == pytest.approx(107.5) and C['replay_ms'] == pytest.approx(109.5)
    # pooled: an instance brings its two combinations
    ref2, rep2 = np.stack([ref, ref], axis=1), np.stack([rep, ref], axis=1)
    Q = P105.cell_stats(ref2, rep2)
    d = 100 * (rep2 - ref2) / ref2
    ib2 = SF.instance_boot(d, B=P105.B, seed=7)
    assert Q['gap']['mean'] == pytest.approx(2.0) and Q['gap']['c'] == 2
    assert (Q['gap']['lo'], Q['gap']['hi']) == (pytest.approx(ib2['lo']), pytest.approx(ib2['hi']))
    assert Q['shares']['at_or_below'] == 75.0 and Q['cases'] == 8


def test_pair_stats_on_synthetic_case_files(tmp_path, monkeypatch):
    """Three planned instances, two combinations. Combination N: cases for a, b, c; L:
    cases for a and b, and c has no valid reference. Per combination over the cases that
    count; pooled over a and b, the instances with a case in both."""
    names = ['a', 'b', 'c']
    root = str(tmp_path)
    vals = {('N', 'a'): (100, 101, 110), ('N', 'b'): (200, 200, 210), ('N', 'c'): (50, 60, 55),
            ('L', 'a'): (100, 104, 104), ('L', 'b'): (100, 100, 120)}
    for (c, nm), (r, x, rule) in vals.items():
        js, npz = P105.case_paths(root, 'pub10x5', 'cpsat150', c, nm)
        os.makedirs(os.path.dirname(js), exist_ok=True)
        rec = dict(design=P101.jsonable(P105.design('pub10x5', 'cpsat150', c)), combo=c,
                   instance=nm, ref_makespan=float(r), replay_makespan=float(x),
                   replay={o: dict(makespan=float(x)) for o in P105.ORDERS},
                   in_class=(x == r), off_share_A=0.1, best_rule=float(rule), best_order='A',
                   class_failures=dict(later=int(c == 'L'), earlier_pair=2 * int(x != r)),
                   n_ops=10,
                   reference=dict(files={'rec': 'sha'}, proved_optimal=True))
        json.dump(rec, open(js, 'w'))
        np.savez(npz, amch=np.zeros((3, 1)))

    def state(states, s, r, c, nm):
        if (c, nm) == ('L', 'c'):
            return ('pending', 'not yet', None, None)
        return ('valid', '', 'rec', True)
    monkeypatch.setattr(P105, 'ref_state', state)
    monkeypatch.setattr(P105, 'read_reference', lambda s, r, p: (None,) * 4 + ({'rec': 'sha'}, {}))
    monkeypatch.setattr(P105, 'planned_names', lambda s, n=None: names)
    args = type('A', (), dict(root=root, n=None, combos=['N', 'L']))()
    P = P105.pair_stats(args, {}, 'pub10x5', 'cpsat150')
    N, L = P['combos']['N'], P['combos']['L']
    assert N['n'] == 3 and L['n'] == 2
    assert N['gap']['mean'] == pytest.approx((1 + 0 + 20) / 3)
    assert L['gap']['mean'] == pytest.approx(2.0)
    assert N['shares']['at_or_below'] == pytest.approx(100 / 3)
    assert L['rule_gap']['mean'] == pytest.approx(12.0)
    assert P['pooled']['combos'] == ['N', 'L'] and P['pooled']['n'] == 2
    assert P['pooled']['gap']['mean'] == pytest.approx((1 + 0 + 4 + 0) / 4)
    assert P['pooled']['proven']['combos'] == ['N', 'L']
    assert len(P['missing']) == 1 and 'L/c' in P['missing'][0] and 'pending' in P['missing'][0]
    assert (P['gap_min'], P['gap_max']) == (pytest.approx(2.0), pytest.approx(7.0))
    # operations of a reference outside the rule, % of its 10 operations: N has no late
    # operation and two earlier pairs in a and c; L has one late operation per case and two
    # earlier pairs in a
    assert N['outside'] == dict(later=pytest.approx(0.0), earlier_pair=pytest.approx(40 / 3),
                                total=pytest.approx(40 / 3))
    assert L['outside'] == dict(later=pytest.approx(10.0), earlier_pair=pytest.approx(10.0),
                                total=pytest.approx(20.0))
    assert P['pooled']['outside']['total'] == pytest.approx((20 + 0 + 30 + 10) / 4)
    lines = P105.macro_lines(dict(partial=False, complete=True, pairs={'k': P}))
    text = '\n'.join(lines)
    w = P['word']
    assert f'\\newcommand{{\\{P105.PREFIX}Outside{w}L}}{{20.0}}' in text
    assert f'\\newcommand{{\\{P105.PREFIX}OutsideLater{w}L}}{{10.0}}' in text


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def test_refuses_a_reference_whose_objective_disagrees_with_its_schedule():
    jl, pt, desc = tiny()
    ref, ms = rule_reference(jl, pt, META, desc)
    assert P105.check_reference(jl, pt, desc, ref['amch'], ref['st'], ref['ct'], ms, 'x') == ms
    with pytest.raises(SystemExit, match='objective'):
        P105.check_reference(jl, pt, desc, ref['amch'], ref['st'], ref['ct'], ms + 1, 'x')
    st = ref['st'].copy()
    st[0] -= 1
    with pytest.raises(SystemExit, match='duration'):
        P105.check_reference(jl, pt, desc, ref['amch'], st, ref['ct'], ms, 'x')


def test_refuses_a_record_with_two_makespans(tmp_path):
    rec = dict(makespan=10, objective=11.0,
               schedule=dict(op_mch=[0], op_st=[0], op_ct=[10]))
    p = tmp_path / 'instance_000_seed1.json'
    p.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match='objective'):
        P105.read_reference('10x25', 'memetic_rule600', str(p))


# ---------------------------------------------------------------------------
# the command line: where it writes, what it stores
# ---------------------------------------------------------------------------

def infer_args(root, out, **kw):
    a = dict(root=root, out=out, sets=['pub10x5'], refs=['cpsat150'], combos=list(P105.R8),
             n=None, require='all', partial=False, cpsat_arm=None)
    a.update(kw)
    return type('Args', (), a)()


def test_infer_require_all_refuses_a_missing_case(tmp_path):
    """No case was run: --require all refuses, names what is missing and leaves no output;
    --require present and a subset of instances need --partial."""
    root, out = str(tmp_path / 'cases'), str(tmp_path / 'out')
    with pytest.raises(P105.Refused, match='do not count'):
        P105.infer(infer_args(root, out))
    assert not os.path.exists(out) or not any(
        os.path.exists(p) for p in P105.infer_outputs(out).values())
    with pytest.raises(P105.Refused, match='--partial'):
        P105.infer(infer_args(root, out, require='present'))
    with pytest.raises(P105.Refused, match='--partial'):
        P105.infer(infer_args(root, out, n=3))


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
                 for r in P105.RULES for c in combos])
    outs = [os.path.join(ROOT, p) for p in P105.OUT_FILES.values()]
    default_root = os.path.join(ROOT, P105.DEFAULT_ROOT)
    top = sorted(os.listdir(ROOT))
    before = _snapshot(frozen)
    had = [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs + [default_root]]
    assert len(before) == len(frozen)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PACT_ENV', None)
    root, out = str(tmp_path / 'cases'), str(tmp_path / 'out')
    script = os.path.join(ROOT, 'scripts', 'p105_reference_replay.py')
    sel = ['--sets', 'pub10x5', '--refs', 'cpsat150', '--n', '1', '--combos', ','.join(combos)]
    for argv in (['run', '--root', root] + sel,
                 ['infer', '--partial', '--require', 'present', '--root', root, '--out', out]
                 + sel):
        p = subprocess.run([sys.executable, '-B', script] + argv, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=110)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert _snapshot(frozen) == before
    assert [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs + [default_root]] == had
    assert sorted(os.listdir(ROOT)) == top
    written = [os.path.join(dp, f) for dp, _, fs in os.walk(str(tmp_path)) for f in fs]
    cases = sorted(os.path.relpath(f, root) for f in written if f.startswith(root + os.sep))
    assert cases == sorted(os.path.join('pub10x5', 'cpsat150', c, f'{names[0]}.{e}')
                           for c in combos for e in ('json', 'npz'))
    for p in P105.OUT_FILES.values():
        assert os.path.exists(os.path.join(out, p))
    assert all(f.startswith((root + os.sep, out + os.sep)) for f in written)
    for f in written:
        if f.endswith(('.json', '.md', '.tex')):
            txt = open(f).read()
            for bad in (ROOT, str(tmp_path), '/home/', '/tmp/'):
                assert bad not in txt, f'{f} holds {bad}'
    rec = json.load(open(os.path.join(root, 'pub10x5', 'cpsat150', 'N', f'{names[0]}.json')))
    assert rec['spt_check'] == 'equal to the stored array'
    assert rec['in_class'] == rec['exact_A']
    assert all(not os.path.isabs(k) for k in rec['reference']['files'])
    tex = open(os.path.join(out, P105.OUT_FILES['macros'])).read()
    for ln in tex.splitlines():
        if ln.startswith('\\newcommand'):
            assert '  % ' in ln, ln
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', tex)
    assert names_m and all(x.startswith(P105.PREFIX) for x in names_m)
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
        'scripts/p105_reference_replay.py', 'tests/test_p105_reference_replay.py')}
    folder = os.path.basename(P105.DEFAULT_ROOT)
    examples = [f'{P105.DEFAULT_ROOT}/pub10x5/cpsat150/LSW/10x5+mix_001.json',
                f'{P105.DEFAULT_ROOT}/80x25n30/memetic_rule600/N/instance_003.npz',
                f'{P105.DEFAULT_ROOT}/50x25n30/cpsat_rule600/S/instance_029.json']
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
