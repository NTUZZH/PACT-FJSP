"""Tests for scripts/p104_divisor_probe.py.

  * the largest processing time of every test instance and the companion rule against an
    independent reading of the .fjs files, and the distribution over the three sizes;
  * the batch mechanics with the stored checkpoint 10x25+family+jointblind-v1 (skipped
    when it is absent): the loop of the script on a batch of one equals
    PolicyRoller.rollout_batch, the divisor of a batch is the largest processing time of
    the instances in it, a companion at the maximum changes the network input of an
    instance below it, and a second copy of the instance changes no input;
  * run with a development run writes a cell, skips it on a rerun, and the partial infer
    reads it; infer refuses --model without --partial;
  * infer on synthetic cells: every count of the design by construction, the signed
    differences against the instance alone (means, mean absolute differences and shares
    by hand, intervals against stats_family.instance_boot with an instance and its eight
    combinations as the unit of the total), the premises of the design (divisors) and a
    development cell refused, macros prefixed, unique and commented on every line;
  * infer without the selected run's validation logs is refused;
  * the command line writes under --root and --out only, and no wildcard path pattern of
    another file matches the new cells.

Run from the repository root:
  python -B -m pytest tests/test_p104_divisor_probe.py -q -p no:cacheprovider
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
import types

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
    if 'p104_divisor_probe' in sys.modules:
        P104 = sys.modules['p104_divisor_probe']
    else:
        _spec = importlib.util.spec_from_file_location(
            'p104_divisor_probe', os.path.join(ROOT, 'scripts', 'p104_divisor_probe.py'))
        P104 = importlib.util.module_from_spec(_spec)
        sys.modules['p104_divisor_probe'] = P104
        _spec.loader.exec_module(P104)
finally:
    sys.argv = _argv
configs.device = 'cpu'
P101 = P104.P101

import stats_family as SF  # noqa: E402

LEGACY = '10x25+family+jointblind-v1'
HAVE_LEGACY = os.path.exists(os.path.join(ROOT, 'trained_network', 'FAMILY', f'{LEGACY}.pth'))
needs_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason=f'{LEGACY} is not on disk')


def _fjs_max(path):
    """Largest processing time of a .fjs file (jobs, machines; per job the operation
    count, then per operation the machine count and machine, time pairs)."""
    with open(path) as f:
        lines = [ln.split() for ln in f if ln.strip()]
    n_jobs, best = int(lines[0][0]), 0.0
    for ln in lines[1:1 + n_jobs]:
        v, i = [float(x) for x in ln], 1
        for _ in range(int(v[0])):
            k = int(v[i])
            best = max(best, max(v[i + 2:i + 2 + 2 * k:2]))
            i += 1 + 2 * k
    return best


@pytest.fixture(scope='module')
def pts():
    return P104.own_max(P104.SIZE)


@pytest.fixture
def exact(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')


def test_largest_processing_time_and_companion(pts):
    base = os.path.join(ROOT, P101.SIZES['10x25']['base'])
    assert sorted(pts) == P101.names_of('10x25') and len(pts) == 100
    for nm, v in pts.items():
        assert v == _fjs_max(os.path.join(base, nm + '.fjs')), nm
    vmax = max(pts.values())
    at = sorted(nm for nm, v in pts.items() if v == vmax)
    assert len(at) >= 2
    for nm in pts:
        c = P104.companion_of(nm, pts)
        assert pts[c] == vmax and c != nm
        assert c == (at[1] if nm == at[0] else at[0])
    with pytest.raises(SystemExit):
        P104.companion_of('instance_000', {'instance_000': 3.0, 'instance_001': 2.0})


def test_distribution(pts):
    D = P104.distribution()
    assert list(D) == list(P101.SIZES)
    for size, d in D.items():
        base = os.path.join(ROOT, P101.SIZES[size]['base'])
        v = np.array([_fjs_max(os.path.join(base, nm + '.fjs')) for nm in P101.names_of(size)])
        assert d['n'] == v.size == P101.SIZES[size]['n']
        assert (d['min'], d['max']) == (v.min(), v.max())
        assert d['at_max'] + d['below_max'] == d['n'] == sum(d['counts'].values())
        assert d['at_max'] == int((v == v.max()).sum())
    assert D['10x25']['max'] == max(pts.values())


# ---------------------------------------------------------------------------
# batch mechanics with a stored checkpoint
# ---------------------------------------------------------------------------

@needs_legacy
def test_batch_mechanics(exact, pts):
    run = P101._p97().resolve_run(LEGACY, development=True)
    assert run['mask'] == P101.MASK
    vmax = max(pts.values())
    low = next(nm for nm in sorted(pts) if pts[nm] < vmax)
    top = sorted(nm for nm in pts if pts[nm] == vmax)
    r = P104.roller_of(run, 'LSW', low)
    c = P104.roller_of(run, 'LSW', top[0])
    assert P104.check_env(r._env) == pts[low]
    env3 = P104.pair_env(r, c)
    assert P104.check_env(env3) == vmax
    assert P104.input_diff(r._env.state, env3.state) > 0
    assert P104.input_diff(r._env.state, r.new_env(2).state) == 0
    c2 = P104.roller_of(run, 'LSW', top[1])
    assert P104.input_diff(c._env.state, P104.pair_env(c, c2).state) == 0
    m1, a1, t1 = r.rollout_batch(True, None, 1)
    mb, ab, tb, ties, gap = P104.greedy_batch(r, r.new_env(1))
    assert mb[0] == m1[0] and np.array_equal(ab, a1) and np.array_equal(tb, t1)
    assert ties.shape == (1,) and 0 <= gap[0] <= 1
    x = P104.probe_instance(P104.roller_of(run, 'LSW', low), c)     # a roller not yet stepped
    assert x['ms_alone'] == float(m1[0]) and x['own_max'] == pts[low]
    assert (x['div_alone'], x['div_twin'], x['div_pair']) == (pts[low], pts[low], vmax)
    assert x['input_diff'] > 0 and x['ms_twin_copy'] > 0 and x['ms_pair_companion'] > 0


def _args(root, **kw):
    a = dict(root=root, combos=['N'], seeds=list(P104.SEEDS), model=None, development=False,
             n=None, part=None, parts=1, force=False, partial=False, out=None)
    a.update(kw)
    return types.SimpleNamespace(**a)


@needs_legacy
def test_run_skip_and_partial_infer(tmp_path, exact, capsys):
    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    P104.run(_args(root, model=LEGACY, development=True, n=2))
    assert 'N    done' in capsys.readouterr().out
    npz, js = P101.cell_paths(root, P104.SIZE, 'N', P104.label_of(LEGACY))
    rec = json.load(open(js))
    assert rec['design']['development'] is True and rec['design']['model'] == LEGACY
    assert rec['instances'] == P101.names_of('10x25', 2)
    assert rec['greedy_check'] == 'no stored greedy array of this run'
    with np.load(npz) as z:
        assert z['ms_alone'].shape == (2,) and z['companion'].dtype == np.int64
    P104.run(_args(root, model=LEGACY, development=True, n=2))
    assert 'N    skip' in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        P104.run(_args(root, model=LEGACY, n=2))
    assert '--development' in str(e.value)
    with pytest.raises(SystemExit) as e:
        P104.infer(_args(root, model=LEGACY, n=None, out=out))
    assert str(e.value).startswith('refused') and '--partial' in str(e.value)
    res = P104.infer(_args(root, model=LEGACY, n=2, partial=True, out=out))
    assert res['partial'] and res['combos']['N']['n'] == 2 and res['total']['n'] == 2
    txt = open(os.path.join(out, 'manuscript', 'macros_divisor.tex')).read()
    assert '\\DivpCaseN}{2}' in txt and '% PARTIAL' in txt


# ---------------------------------------------------------------------------
# infer on synthetic cells
# ---------------------------------------------------------------------------

MODEL = '10x25+family+fake-s301'


def _write(root, combo, pts, development=False, edit=None):
    names = P101.names_of('10x25')
    own = np.array([pts[nm] for nm in names])
    vmax = own.max()
    alone = 200.0 + np.arange(100)
    z = dict(ms_alone=alone, ms_twin=alone.copy(), ms_twin_copy=alone.copy(),
             ms_pair=alone.copy(), ms_pair_companion=np.full(100, 250.0),
             same_twin=np.ones(100, bool), same_pair=np.ones(100, bool), own_max=own,
             div_alone=own.copy(), div_twin=own.copy(), div_pair=np.full(100, vmax),
             input_diff=np.where(own < vmax, 0.25, 0.0), ties_alone=np.zeros(100, np.int64),
             min_gap_alone=np.full(100, 0.1), companion=np.zeros(100, np.int64))
    if edit:
        edit(z)
    npz, js = P101.cell_paths(root, '10x25', combo, P104.label_of(MODEL))
    P101.save_npz_atomic(npz, **z)
    P101.write_json_atomic(js, {'design': {'kind': 'divisor', 'model': MODEL,
                                           'development': development},
                                'instances': names})


def test_counts_on_synthetic_cells(tmp_path, monkeypatch, exact, pts):
    """Combination N: instance b0 (below the maximum) longer by 2 h with the companion,
    instance t0 (at the maximum) shorter by 3 h twice in one batch, instance b1 with an
    equal makespan but another schedule twice in one batch, near ties on b0 and b1. The
    other seven combinations: all equal."""
    monkeypatch.setattr(P104, 'selected', lambda seeds: (301, MODEL, {}))
    names = P101.names_of('10x25')
    vmax = max(pts.values())
    below = [i for i, nm in enumerate(names) if pts[nm] < vmax]
    at = [i for i, nm in enumerate(names) if pts[nm] == vmax]
    b0, b1, t0 = below[0], below[1], at[0]

    def edit(z):
        z['ms_pair'][b0] += 2.0
        z['same_pair'][b0] = False
        z['ms_twin'][t0] -= 3.0
        z['ms_twin_copy'][t0] -= 3.0
        z['same_twin'][[t0, b1]] = False
        z['ties_alone'][[b0, b1]] = [2, 1]

    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    for combo in P101.R8:
        _write(root, combo, pts, edit=edit if combo == 'N' else None)
    res = P104.infer(_args(root, combos=list(P101.R8), out=out))
    nb, na = len(below), len(at)
    N = res['combos']['N']
    assert (N['n'], N['below'], N['same'], N['same_below']) == (100, nb, 98, nb - 1)
    assert N['same_schedule'] == 97 and N['same_twin'] == 99
    assert (N['same_pair_below'], N['same_pair_at_max']) == (nb - 1, na)
    assert (N['differ'], N['differ_below']) == (2, 1)
    assert N['max_rel_diff'] == pytest.approx(300.0 / (200.0 + t0))
    assert (N['input_changed_below'], N['max_input_diff']) == (nb, 0.25)
    assert (N['twin_copy_equal'], N['tie_cases'], N['differ_with_tie']) == (100, 2, 1)
    W = res['combos']['W']
    assert (W['same'], W['differ'], W['max_rel_diff'], W['tie_cases']) == (100, 0, 0.0, 0)
    T = res['total']
    assert (T['n'], T['same'], T['differ'], T['below']) == (800, 798, 2, 8 * nb)
    assert T['instances_below'] == nb and T['max_rel_diff'] == N['max_rel_diff']
    assert res['test_set_max'] == vmax and not res['partial']
    txt = open(os.path.join(out, 'manuscript', 'macros_divisor.tex')).read()
    lines = [ln for ln in txt.splitlines() if ln.startswith('\\newcommand')]
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(names_m) == len(set(names_m)) and all(re.fullmatch(r'Divp[A-Za-z]+', x)
                                                    for x in names_m)
    assert all('  % ' in ln for ln in lines) and 'PARTIAL' not in txt
    for want in ('\\DivpCaseN}{800}', '\\DivpDiffN}{2}', '\\DivpSameInN}{98}',
                 '\\DivpDiffInW}{0}', f'\\DivpBelowN}}{{{nb}}}', '\\DivpTieCaseN}{2}',
                 '\\DivpDiffTieN}{1}', f'\\DivpMaxPt}}{{{vmax:g}}}', '\\DivpPtNEighty}{30}'):
        assert want in txt, want
    md = open(os.path.join(out, 'results', 'divisor_probe.md')).read()
    assert '| total | 800 |' in md and '| 80x25n30 | 30 |' in md

    # signed differences, 100 (x - alone) / alone: +200/(200+b0) with the companion on b0,
    # -300/(200+t0) twice in one batch on t0, zero elsewhere
    up, down = 200.0 / (200 + b0), -300.0 / (200 + t0)
    SB = N['signed']
    want = {('pair', 'below'): (up / nb, abs(up) / nb, 0, nb - 1, 1, nb),
            ('pair', 'at_max'): (0.0, 0.0, 0, na, 0, na),
            ('twin', 'below'): (0.0, 0.0, 0, nb, 0, nb),
            ('twin', 'at_max'): (down / na, abs(down) / na, 1, na - 1, 0, na)}
    for (case, g), (mean, mabs, sh, eq, lo_, n) in want.items():
        X = SB[case][g]
        assert X['mean'] == pytest.approx(mean) and X['mean_abs'] == pytest.approx(mabs)
        assert (X['shorter'], X['equal'], X['longer'], X['cases']) == (sh, eq, lo_, n)
        assert X['shorter_pct'] == pytest.approx(100.0 * sh / n)
    d = np.zeros(nb)
    d[0] = up
    ib = SF.instance_boot(d, B=P101.B, seed=7)
    assert (SB['pair']['below']['lo'], SB['pair']['below']['hi']) == (ib['lo'], ib['hi'])
    assert SB['pair']['at_max']['lo'] == SB['pair']['at_max']['hi'] == 0.0
    WS = res['combos']['W']['signed']
    assert all(WS[c][g]['equal'] == WS[c][g]['cases'] and WS[c][g]['mean'] == 0
               for c in ('pair', 'twin') for g in ('below', 'at_max'))
    TS = T['signed']
    assert TS['pair']['below']['mean'] == pytest.approx(up / (8 * nb))
    assert TS['twin']['at_max']['mean'] == pytest.approx(down / (8 * na))
    assert (TS['pair']['below']['cases'], TS['pair']['below']['instances'],
            TS['pair']['below']['combinations']) == (8 * nb, nb, 8)
    assert (TS['twin']['at_max']['shorter'], TS['twin']['at_max']['equal']) == (1, 8 * na - 1)
    D = np.zeros((nb, 8))
    D[0, list(P101.R8).index('N')] = up
    ib = SF.instance_boot(D, B=P101.B, seed=7)            # an instance with its eight
    assert (TS['pair']['below']['lo'], TS['pair']['below']['hi']) == (ib['lo'], ib['hi'])
    for want_m in (f'\\DivpPairBelowMean}}{{{P101.f2(up / (8 * nb))}}}',
                   f'\\DivpPairBelowMeanInN}}{{{P101.f2(up / nb)}}}',
                   f'\\DivpPairBelowLoInN}}{{{P101.f2(SB["pair"]["below"]["lo"])}}}',
                   f'\\DivpTwinAtMaxAbsInN}}{{{abs(down) / na:.2f}}}',
                   f'\\DivpTwinAtMaxShorterInN}}{{{P101.u1(100.0 / na)}}}',
                   f'\\DivpPairBelowLongerInN}}{{{P101.u1(100.0 / nb)}}}',
                   '\\DivpTwinBelowEqualInW}{100.0}', '\\DivpPairAtMaxHi}{+0.00}'):
        assert want_m in txt, want_m
    assert 'with the companion, mean [interval]' in md
    assert f'| N | below the test-set maximum | {nb} | {P101.f2(up / nb)} [' in md

    def refused(match, **kw):
        with pytest.raises(SystemExit) as e:
            P104.infer(_args(root, combos=list(P101.R8), out=out))
        assert str(e.value).startswith('refused') and match in str(e.value), str(e.value)
        assert not os.path.exists(os.path.join(out, 'results', 'divisor_probe.json'))

    _write(root, 'LS', pts, edit=lambda z: z['div_pair'].__setitem__(b0, pts[names[b0]]))
    refused('case (iii)')
    _write(root, 'LS', pts, edit=lambda z: z['div_twin'].__setitem__(b0, vmax))
    refused('case twin')
    _write(root, 'LS', pts, development=True)
    refused('development run')
    res = P104.infer(_args(root, combos=['LS'], partial=True, out=out))
    assert res['partial'] and res['combos']['LS']['same'] == 100


def test_infer_without_the_selected_run_is_refused(tmp_path, exact):
    with pytest.raises(SystemExit) as e:
        P104.infer(_args(str(tmp_path), combos=list(P101.R8), out=str(tmp_path / 'out')))
    assert str(e.value).startswith('refused'), str(e.value)


def test_refused_outside_the_exact_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(SystemExit) as e:
        P104.infer(_args(str(tmp_path), combos=list(P101.R8), out=str(tmp_path / 'out')))
    assert 'PACT_ENV' in str(e.value)


# ---------------------------------------------------------------------------
# the command line and folder isolation
# ---------------------------------------------------------------------------

def _snapshot(patterns):
    out = {}
    for pat in patterns:
        for p in glob.glob(os.path.join(ROOT, pat), recursive=True):
            if os.path.isfile(p):
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime_ns)
    return out


@needs_legacy
def test_command_line_writes_only_under_root_and_out(tmp_path):
    """The instances, descriptors and stored rule arrays the probe reads stay unchanged
    (the folders of the checkpoints and of the policy arrays are left out: training and
    evaluation of other runs may write there meanwhile)."""
    xm = os.path.join(P101.RESULTS_ROOT, '10x25_xmask')
    frozen = [os.path.join(P101.SIZES['10x25']['data'], '**', '*'),
              os.path.join(P101.SIZES['10x25']['base'], '**', '*')] + [
        os.path.join(xm, f'Result_{r}_*.npy') for r in P101.RULES + P101.AWARE]
    outs = [os.path.join(ROOT, p) for p in P104.OUT_FILES.values()]
    before, had = _snapshot(frozen), [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs]
    assert len(before) > 200 + 7 * 8
    env = dict(os.environ, PACT_ENV='exact', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    script = os.path.join(ROOT, 'scripts', 'p104_divisor_probe.py')
    for argv in (['run', '--model', LEGACY, '--development', '--root', root, '--n', '1',
                  '--combos', 'W'],
                 ['infer', '--partial', '--model', LEGACY, '--root', root, '--out', out,
                  '--n', '1', '--combos', 'W']):
        p = subprocess.run([sys.executable, '-B', script] + argv, cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=110)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    assert _snapshot(frozen) == before
    assert [os.path.exists(p) and os.stat(p).st_mtime_ns for p in outs] == had
    assert sorted(os.listdir(os.path.join(root, '10x25', 'W'))) == [
        f'{P104.label_of(LEGACY)}.json', f'{P104.label_of(LEGACY)}.npz']
    for p in P104.OUT_FILES.values():
        assert os.path.exists(os.path.join(out, p))
    rec = json.load(open(os.path.join(root, '10x25', 'W', f'{P104.label_of(LEGACY)}.json')))
    assert ROOT not in json.dumps(rec)


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
        'scripts/p104_divisor_probe.py', 'tests/test_p104_divisor_probe.py')}
    folder = os.path.basename(P104.DEFAULT_ROOT)
    examples = [f'{P104.DEFAULT_ROOT}/10x25/LSW/{P104.label_of("10x25+family+xmask-v1-s301")}.npz',
                f'{P104.DEFAULT_ROOT}/10x25/N/{P104.label_of("10x25+family+xmask-v1-s305")}.json']
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
