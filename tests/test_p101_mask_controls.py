"""Tests for scripts/p101_mask_controls.py.

  * the random control is deterministic: the same seeds give the same rollouts, a
    run written twice gives identical arrays, and different draws differ;
  * the legal-pair counts, the coverage and the load equal hand-computed values on
    two small instances (one without constraints, one with an outage);
  * the coverage and load formulas on hand-made schedules (a window after the
    makespan counts nothing, a window that crosses it counts up to it);
  * the SPT rollout of the script reproduces the stored exact-mask array, and every
    decision has at least one legal pair;
  * infer refuses with a message that starts with "refused" when a cell is missing,
    and writes prefixed, unique macros on a complete partial run;
  * no existing script reads the new folders under test_results/FAMILY;
  * the frozen instance and descriptor folders and the stored result folders are
    unchanged by a run of p101 and of p102 (sha256 of every file before and after).

Run from the repository root:
  python -m pytest tests/test_p101_mask_controls.py -q -p no:cacheprovider
CPU only.
"""
import fnmatch
import hashlib
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
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401  (parses argv at import)
    import data_utils  # noqa: E402,F401
    _spec = importlib.util.spec_from_file_location(
        'p101_mask_controls', os.path.join(ROOT, 'scripts', 'p101_mask_controls.py'))
    P101 = importlib.util.module_from_spec(_spec)
    sys.modules['p101_mask_controls'] = P101
    _spec.loader.exec_module(P101)
finally:
    sys.argv = _argv
configs.device = 'cpu'

from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
ENV.pop('PACT_ENV', None)
FROZEN = ('data/FAMILY/50x25', 'data/FAMILY/80x25', 'data/FAMILY/50x25n30',
          'data/FAMILY/80x25n30', 'data/FAMILY/10x25',
          'data/PPVC/50x25+ppvc-mixed-n30', 'data/PPVC/80x25+ppvc-mixed-n30',
          'test_results/FAMILY/10x25_xmask', 'test_results/FAMILY/50x25n30_xmask',
          'test_results/FAMILY/80x25n30_xmask')
NEW_FOLDERS = ('maskctl_p101', 'sens_p102')


# ---------------------------------------------------------------------------
# hand-checked small cases
# ---------------------------------------------------------------------------

def tiny(window=False):
    """Two jobs of two operations on two machines (0 = incompatible).
       op0 (job 0): m0 2 h, m1 3 h      op1 (job 0): m1 1 h only
       op2 (job 1): m0 1 h only         op3 (job 1): m0 3 h (2 h without the outage
                                                     case), m1 2 h
    The outage case closes machine 1 on [0, 2)."""
    jl = np.array([2, 2])
    pt = np.array([[2, 3], [0, 1], [1, 0], [3 if window else 2, 2]])
    meta = {'op_type': None, 'mch_type': None}
    if window:
        desc = ConstraintDescriptor(active=(False, False, True), lag=np.zeros(4),
                                    op_class=np.zeros(4), windows=[[], [(0, 2)]],
                                    horizon=10)
    else:
        desc = ConstraintDescriptor(active=(False, False, False), lag=np.zeros(4),
                                    op_class=np.zeros(4), horizon=1)
    return jl, pt, meta, desc


def test_legal_pairs_by_hand_without_constraints():
    """Decision 1 (t = 0): (j0,m0), (j0,m1), (j1,m0) legal, SPT takes op2 on m0 [0,1].
    Decision 2: j0 on m1 starts at 0, every other pair at 1, so one legal pair; op0 on
    m1 [0,3]. Decision 3: j1 on m0 at 1 is the only start at t = 1; op3 on m0 [1,3].
    Decision 4: op1 on m1 [3,4]. Makespan 4, counts 3, 1, 1, 1."""
    jl, pt, meta, desc = tiny()
    r = P101.rule_rollout('SPT', jl, pt, meta, desc)
    assert r['counts'].tolist() == [3, 1, 1, 1]
    assert r['makespan'] == 4.0
    assert r['amch'].tolist() == [1, 1, 0, 0]
    assert r['lb0'] == pytest.approx(3.0, abs=1e-6)
    util, ms_lb, _ = P101.shop_load(pt, r['amch'], r['makespan'], r['lb0'])
    assert util == pytest.approx(7 / 8)
    assert ms_lb == pytest.approx(4 / 3, abs=1e-6)
    assert all(np.isnan(P101.window_coverage(desc, r['makespan'], 2)))


def test_legal_pairs_by_hand_with_an_outage():
    """Machine 1 is closed on [0, 2). Decision 1 (t = 0): op0 on m1 is pushed to 2, so
    (j0,m0) and (j1,m0) are legal; SPT takes op2 on m0 [0,1]. Decision 2 (t = 1):
    (j0,m0) and (j1,m0) start at 1, the m1 pairs at 2; SPT takes op0 on m0 [1,3].
    Decision 3: op3 on m1 starts at 2, op3 on m0 and op1 on m1 at 3; one legal pair,
    op3 on m1 [2,4]. Decision 4: op1 on m1 [4,5]. Makespan 5, counts 2, 2, 1, 1;
    window end 2 over 5, window time 2 over 2 x 5, processing 6 over 2 x 5."""
    jl, pt, meta, desc = tiny(window=True)
    r = P101.rule_rollout('SPT', jl, pt, meta, desc)
    assert r['counts'].tolist() == [2, 2, 1, 1]
    assert r['makespan'] == 5.0
    assert r['amch'].tolist() == [0, 1, 0, 1]
    assert r['op_ct'].tolist() == [3.0, 5.0, 1.0, 4.0]
    ce, ct, le = P101.window_coverage(desc, r['makespan'], 2)
    assert (ce, ct, le) == (pytest.approx(0.4), pytest.approx(0.2), 2.0)
    util, ms_lb, _ = P101.shop_load(pt, r['amch'], r['makespan'], r['lb0'])
    assert util == pytest.approx(0.6)
    assert ms_lb == pytest.approx(5 / 3, abs=1e-6)
    assert P101.validate(jl, pt, desc, r['amch'], r['op_ct'], r['makespan']) == []


def test_random_control_chooses_only_legal_pairs_on_the_hand_case():
    """With the outage, op0 can never start on m1 before 2, so no random rollout puts
    op0 on m1 at time 0; every schedule is feasible and non-delay."""
    jl, pt, meta, desc = tiny(window=True)
    seeds = [[P101.RANDOM_SEED, 1, 0, k] for k in range(20)]
    ms, amch, op_ct = P101.random_rollouts(jl, pt, meta, desc, seeds)
    for k in range(len(seeds)):
        assert P101.validate(jl, pt, desc, amch[k], op_ct[k], ms[k]) == []
        st0 = op_ct[k, 0] - pt[0, amch[k, 0]]
        assert not (amch[k, 0] == 1 and st0 < 2)


# ---------------------------------------------------------------------------
# formulas
# ---------------------------------------------------------------------------

def test_coverage_formula():
    d = ConstraintDescriptor(active=(False, False, True), lag=np.zeros(2),
                             op_class=np.zeros(2),
                             windows=[[(2, 5), (90, 95)], [(8, 14)], []], horizon=100)
    ce, ct, le = P101.window_coverage(d, 10.0, 3)
    assert le == 95.0
    assert ce == pytest.approx(9.5)
    # inside [0, 10): 3 h on machine 0 (the window at 90 counts nothing), 2 h of
    # [8, 14) on machine 1
    assert ct == pytest.approx(5.0 / 30.0)
    ce, ct, _ = P101.window_coverage(d, 200.0, 3)
    assert ce == pytest.approx(95 / 200) and ct == pytest.approx(14.0 / 600.0)


def test_load_formula():
    pt = np.array([[4, 0, 2], [0, 3, 3], [1, 1, 0], [5, 5, 5]])
    amch = np.array([0, 2, 1, 1])
    util, ms_lb, worst = P101.shop_load(pt, amch, 10.0, 8.0, mch_type=np.array([0, 1, 1]))
    assert util == pytest.approx((4 + 3 + 1 + 5) / 30.0)
    assert ms_lb == pytest.approx(1.25)
    # type 0 = machine 0 (4 h), type 1 = machines 1 and 2 (1 + 5 + 3 h)
    assert worst == pytest.approx(max(4 / 10, 9 / 20))


# ---------------------------------------------------------------------------
# determinism and reproduction of the stored arrays
# ---------------------------------------------------------------------------

def test_random_control_is_deterministic(tmp_path):
    cell = P101.load_cell('10x25', 'LSW', 1)
    name, jl, pt, meta, desc = cell[0]
    seeds = P101.draw_seeds('10x25', name)[:4]
    a = P101.random_rollouts(jl, pt, meta, desc, seeds)
    b = P101.random_rollouts(jl, pt, meta, desc, seeds)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
    assert len(set(a[0].tolist())) > 1, 'four draws gave one makespan'
    # the batch does not couple the copies: each draw alone gives the same rollout
    for k in range(len(seeds)):
        m1, amch1, _ = P101.random_rollouts(jl, pt, meta, desc, [seeds[k]])
        assert m1[0] == a[0][k] and np.array_equal(amch1[0], a[1][k])
    roots = [str(tmp_path / 'a'), str(tmp_path / 'b')]
    for r in roots:
        assert P101.run_random(r, '10x25', 'N', 2, False)[0] == 'done'
    za, zb = (np.load(P101.cell_paths(r, '10x25', 'N', 'random')[0]) for r in roots)
    assert np.array_equal(za['makespan'], zb['makespan'])
    assert P101.run_random(roots[0], '10x25', 'N', 2, False)[0] == 'skip'


def test_spt_trace_reproduces_the_stored_arrays(tmp_path):
    root = str(tmp_path)
    for combo in ('S', 'LSW'):
        assert P101.run_trace(root, '10x25', combo, 3, False)[0] == 'done'
        z = np.load(P101.cell_paths(root, '10x25', combo, 'trace_SPT')[0])
        stored = P101.stored_rule('10x25', 'SPT', combo, 3)
        assert np.array_equal(z['makespan'], stored)
        n_ops = z['amch'].shape[1]
        assert z['n_decisions'].tolist() == [n_ops] * 3
        assert z['legal_counts'].min() >= 1 and z['legal_counts'].size == 3 * n_ops
        assert np.all(z['util'] > 0) and np.all(z['util'] < 1)
        assert np.all(z['ms_over_lb0'] >= 1)
    assert np.all(np.isfinite(z['cov_end'])) and np.all(z['cov_time'] > 0)


def test_infer_refuses_a_missing_cell(tmp_path):
    args = _infer_args(str(tmp_path / 'empty'), str(tmp_path / 'out'))
    with pytest.raises(SystemExit) as e:
        P101.infer(args)
    assert str(e.value).startswith('refused'), str(e.value)
    assert not os.path.exists(tmp_path / 'out' / 'results' / 'mask_controls.json')


def test_infer_on_a_complete_partial_run(tmp_path, monkeypatch):
    # the released policy's exact-mask arrays appear while the campaign runs; this test
    # covers the rule comparators only
    monkeypatch.setattr(P101, 'stored_policy', lambda *a: (None, []))
    root = str(tmp_path / 'cells')
    for combo in ('N', 'W'):
        P101.run_random(root, '10x25', combo, 2, False)
        P101.run_trace(root, '10x25', combo, 2, False)
    res = P101.infer(_infer_args(root, str(tmp_path / 'out'), combos=['N', 'W']))
    C = res['sizes']['10x25']['combos']
    assert set(C) == {'N', 'W'}
    assert 'cov_end' in C['W']['trace']['SPT'] and 'cov_end' not in C['N']['trace']['SPT']
    txt = open(tmp_path / 'out' / 'manuscript' / 'macros_maskctl.tex').read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert names and len(names) == len(set(names))
    assert all(re.fullmatch(r'Mctl[A-Za-z]+', x) for x in names)
    assert 'PARTIAL' in txt


def test_policy_trace_path(tmp_path, monkeypatch):
    """The greedy-trace code path with a stored checkpoint of the earlier environment
    (the exact-mask runs are not trained yet): every operation is one decision, the
    parameter count is asserted, and infer reports the policy trace beside SPT."""
    monkeypatch.setattr(P101, 'stored_policy', lambda *a: (None, []))
    model = '10x25+family+jointblind-v1'
    if not os.path.exists(os.path.join(ROOT, 'trained_network', 'FAMILY', f'{model}.pth')):
        pytest.skip(f'{model} is not on disk')
    root = str(tmp_path / 'cells')
    runner = P101.PolicyRunner()
    for combo in ('N', 'W'):
        P101.run_random(root, '10x25', combo, 2, False)
        P101.run_trace(root, '10x25', combo, 2, False)
        assert P101.run_trace(root, '10x25', combo, 2, False, runner, model)[0] == 'done'
    assert runner.load(model)[1]['params'] == P101.EXPECT_PARAMS
    z = np.load(P101.cell_paths(root, '10x25', 'W', f'trace_{model}')[0])
    assert z['n_decisions'].tolist() == [z['amch'].shape[1]] * 2
    a = _infer_args(root, str(tmp_path / 'out'), combos=['N', 'W'])
    a.trace_arms, a.policy_models = ['SPT', 'policy'], [model]
    res = P101.infer(a)
    T = res['sizes']['10x25']['combos']['W']['trace']
    assert set(T) == {'SPT', 'policy'} and T['policy']['runs'] == [f'trace_{model}']


def _infer_args(root, out, combos=('N',)):
    class A:
        pass
    a = A()
    a.root, a.out, a.partial, a.n = root, out, True, 2
    a.sizes, a.combos = ['10x25'], list(combos)
    a.trace_arms, a.policy_seeds, a.policy_models = ['SPT'], [301, 302], None
    a.trace_seeds = [301, 302]
    return a


# ---------------------------------------------------------------------------
# folder isolation and frozen folders
# ---------------------------------------------------------------------------

def _scanned_files():
    out = []
    for top in ('.', 'scripts', 'tests', 'constraint_family', 'baselines', 'backbones',
                'model'):
        d = os.path.join(ROOT, top)
        if not os.path.isdir(d):
            continue
        if top == '.':
            out += [os.path.join(d, f) for f in os.listdir(d)
                    if f.endswith(('.py', '.sh'))]
            continue
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if not x.startswith(('.', '__'))]
            out += [os.path.join(dp, f) for f in fns if f.endswith(('.py', '.sh'))]
    return sorted(set(os.path.normpath(p) for p in out))


def test_new_folders_are_read_by_no_existing_script():
    """The new folders hold random.npz, trace_*.npz, <rule>.npz and descriptor JSON files
    at depth 3 to 7 below test_results/FAMILY. (1) No other script names them. (2) Every
    literal path pattern below test_results/FAMILY with a wildcard fails to match any
    of their file paths (fnmatch, whose * also crosses '/'). (3) The remaining
    references build the first folder from a size key or a registry folder plus a
    suffix; the new names start with no size and end in no registry suffix."""
    own = {os.path.normpath(os.path.join(ROOT, p)) for p in (
        'scripts/p101_mask_controls.py', 'scripts/p102_scale_sensitivity.py',
        'tests/test_p101_mask_controls.py', 'tests/test_p102_scale_sensitivity.py')}
    examples = [
        'test_results/FAMILY/maskctl_p101/10x25/LSW/random.npz',
        'test_results/FAMILY/maskctl_p101/80x25n30/W/random.json',
        'test_results/FAMILY/maskctl_p101/50x25n30/N/trace_SPT.npz',
        'test_results/FAMILY/maskctl_p101/10x25/LS/trace_10x25+family+xmask-v1-s301.npz',
        'test_results/FAMILY/sens_p102/results/exact/default/50x25n30/LSW/SPT.npz',
        'test_results/FAMILY/sens_p102/results/exact/win_full/80x25n30/W/'
        'greedy+10x25+family+xmask-v1-s301.npz',
        'test_results/FAMILY/sens_p102/data/win_full/50x25n30/desc/LW/instance_000.desc.json',
        'test_results/FAMILY/sens_p102/data/win_full/80x25n30/manifest.json']
    rx = re.compile(r'test_results/FAMILY/[^\'"\s)]*')
    patterns = {}
    scanned = [f for f in _scanned_files() if f not in own]
    for f in scanned:
        txt = open(f, errors='replace').read()
        for nm in NEW_FOLDERS:
            assert nm not in txt, f'{f} names {nm}'
        for m in rx.findall(txt):
            first = m.split('/')[2] if m.count('/') >= 2 else ''
            if any(ch in first for ch in '*?['):
                patterns.setdefault(m, f)
    # the scan counts what is present: the scripts of the release may name no such pattern,
    # so the finder is checked on a known line and the scan on the files it read
    assert scanned, 'the scan read no file; the scan itself is broken'
    known = rx.findall("glob('test_results/FAMILY/*_xmask/Result_*.npy')")
    assert known == ["test_results/FAMILY/*_xmask/Result_*.npy"], known
    for pat, f in patterns.items():
        for ex in examples:
            assert not fnmatch.fnmatch(ex, pat), f'{f}: {pat} matches {ex}'
    import arm_registry as REG
    suffixes = set(REG.FOLDER_SUFFIX.values()) | set(REG.LEGACY_TAG.values()) - {''}
    for nm in NEW_FOLDERS:
        assert not re.match(r'\d+x\d+', nm)
        assert not any(s and nm.endswith(s) for s in suffixes)
        assert nm not in REG.FOLDER_ALIAS['exact'] and nm not in REG.LEGACY_TAG


def _tree_digest(rel):
    h = hashlib.sha256()
    base = os.path.join(ROOT, rel)
    n = 0
    for dp, dns, fns in os.walk(base):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, base).encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
            n += 1
    return h.hexdigest(), n


def _run(args):
    r = subprocess.run([PY] + args, cwd=ROOT, env=ENV, capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def test_frozen_folders_unchanged_by_a_run(tmp_path):
    before = {d: _tree_digest(d) for d in FROZEN}
    assert all(n > 0 for _, n in before.values())
    r1, r2 = str(tmp_path / 'p101'), str(tmp_path / 'p102')
    _run(['scripts/p101_mask_controls.py', 'run', '--root', r1, '--sizes', '50x25n30',
          '--combos', 'LW', '--n', '1'])
    _run(['scripts/p102_scale_sensitivity.py', 'build', '--root', r2, '--sizes', '50x25n30',
          '--n', '1', '--verify_n', '1'])
    _run(['scripts/p102_scale_sensitivity.py', 'evaluate', '--root', r2, '--sizes',
          '50x25n30', '--combos', 'LW', '--n', '1', '--arms', 'rules'])
    after = {d: _tree_digest(d) for d in FROZEN}
    assert after == before
