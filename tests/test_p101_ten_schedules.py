"""Tests for the arm with ten schedules of the policy in scripts/p101_mask_controls.py
(QUANTITIES (e)).

  * the seed stream: a pure function of the size, the instance, the combination, the
    trained run and the schedule index, distinct from the stream of the random control,
    with one value pinned;
  * the cells of the other arms keep their design (the stored cells of the repository
    are 'done' under the design function);
  * schedule 0 of the arm, drawn through p97_policy_anytime.PolicyRoller, equals the
    greedy rollout of PolicyRunner (the policy trace) on the same instances; the sampled
    schedules are reproducible, feasible and each equals a rollout of its own seed;
  * a cell is written, skipped on a rerun, refused under another design, and refused
    when its greedy schedule differs from the greedy trace cell of the same run;
  * infer on synthetic cells: per-run rows with the bootstrap of the script, the pooled
    row with stats_family.hier_boot, the shares, the best of seven rules, prefixed and
    unique macros with a comment on every line, and the Markdown table;
  * a size that holds the selected run only: its rows, the statement of the form in the
    JSON, the Markdown and a macro, and no pooled row or pooled macro; --ten_runs at
    infer requires the form it names;
  * infer refuses cells for some runs or some combinations only, another number of
    schedules, a greedy schedule that differs from the trace of its run, and --ten_arm
    require without cells;
  * without cells of the arm, infer writes byte for byte what the script at commit
    3b2eb14, which has no such arm, writes, and what --ten_arm off writes;
  * --ten_runs: parsing, the cells of plan and run per size, parts that do not depend on
    which run the selection rule picks, and run with the selected run only;
  * plan lists the cells, the core-hours and the commands of the arm;
  * no wildcard path pattern of another script matches the new cells.

Run from the repository root:
  python -B -m pytest tests/test_p101_ten_schedules.py -q -p no:cacheprovider
CPU only; the policy tests use the stored checkpoint 10x25+family+jointblind-v1 (skipped
when it is absent).
"""
import fnmatch
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import types
from contextlib import redirect_stdout

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
    if 'p101_mask_controls' in sys.modules:
        P101 = sys.modules['p101_mask_controls']
    else:
        _spec = importlib.util.spec_from_file_location(
            'p101_mask_controls', os.path.join(ROOT, 'scripts', 'p101_mask_controls.py'))
        P101 = importlib.util.module_from_spec(_spec)
        sys.modules['p101_mask_controls'] = P101
        _spec.loader.exec_module(P101)
finally:
    sys.argv = _argv
configs.device = 'cpu'

import stats_family as SF  # noqa: E402

LEGACY = '10x25+family+jointblind-v1'
REFERENCE_COMMIT = '3b2eb14'          # p101 without the arm with ten schedules
HAVE_LEGACY = os.path.exists(os.path.join(ROOT, 'trained_network', 'FAMILY', f'{LEGACY}.pth'))
needs_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason=f'{LEGACY} is not on disk')


def has_commit(rev):
    """Whether the folder of the repository is the top of a git work tree that holds the
    commit rev (False without git, outside a work tree, or without that commit)."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode or os.path.realpath(top.stdout.strip()) != os.path.realpath(ROOT):
            return False
        return subprocess.run(['git', 'cat-file', '-e', rev + '^{commit}'], cwd=ROOT,
                              capture_output=True).returncode == 0
    except OSError:
        return False


@pytest.fixture(scope='module')
def runner():
    return P101.PolicyRunner()


# ---------------------------------------------------------------------------
# the seed stream
# ---------------------------------------------------------------------------

def test_seed_stream():
    e = P101.ten_entropy('10x25', 'instance_000', 'LSW', 301, 1)
    assert e == [P101.RANDOM_SEED, P101.TEN_TAG, 10, 0, 7, 301, 1]
    want = int(np.random.SeedSequence(e).generate_state(1, dtype=np.uint64)[0]) // 2
    assert P101.ten_torch_seed(e) == want == 2622056855213604940
    rnd = [list(x) for x in P101.draw_seeds('10x25', 'instance_000')]
    ten = [P101.ten_entropy('10x25', 'instance_000', 'LSW', 301, k) for k in range(1, 10)]
    assert not any(t[:4] == r[:4] or t == r for t in ten for r in rnd)
    assert P101.TEN_TAG not in [s['code'] for s in P101.SIZES.values()]
    base = [P101.ten_torch_seed(t) for t in ten]
    assert len(set(base)) == 9 and all(0 <= s < 2 ** 63 for s in base)
    for args in (('50x25n30', 'instance_000', 'LSW', 301), ('10x25', 'instance_001', 'LSW', 301),
                 ('10x25', 'instance_000', 'LS', 301), ('10x25', 'instance_000', 'LSW', 302)):
        other = [P101.ten_torch_seed(P101.ten_entropy(*args, k)) for k in range(1, 10)]
        assert not set(other) & set(base)
    assert P101.model_seed('10x25+family+xmask-v1-s304') == 304
    assert P101.model_seed(LEGACY) == 301


def test_designs_of_the_other_arms_unchanged():
    """The stored cells of the random control and of the SPT trace are 'done': the design
    function gives for them the design recorded in their files."""
    root = os.path.join(ROOT, P101.DEFAULT_ROOT)
    seen = 0
    for size in P101.SIZES:
        names = P101.names_of(size)
        for combo in P101.R8:
            for kind, label in (('random', 'random'), ('trace', 'trace_SPT')):
                if not os.path.exists(P101.cell_paths(root, size, combo, label)[1]):
                    continue
                seen += 1
                want = P101.design(kind, size, combo)
                assert P101.cell_state(root, size, combo, label, want, names) == 'done'
    if not seen:
        pytest.skip('no stored cell of the random control or the SPT trace')


# ---------------------------------------------------------------------------
# rollouts through p97
# ---------------------------------------------------------------------------

@needs_legacy
def test_greedy_schedule_equals_the_policy_trace(runner):
    """Schedule 0 of the arm equals PolicyRunner.rollout (the greedy trace) on the same
    instance; the arm is reproducible; each sampled schedule equals a fresh rollout of
    p97 under its own seed; every schedule is feasible."""
    sampled_differs = False
    for combo in ('N', 'LSW'):
        for name, jl, pt, meta, desc in P101.load_cell('10x25', combo, 2):
            g = runner.rollout(LEGACY, jl, pt, meta, desc)
            r = P101.ten_roller(runner, LEGACY, '10x25', combo, name)
            assert np.array_equal(r.jl, jl) and np.array_equal(r.pt, pt)
            ms, amch, ct, seeds = P101.ten_rollouts(r, '10x25', name, combo, 301, draws=4)
            assert ms[0] == g['makespan']
            assert np.array_equal(amch[0], g['amch']) and np.array_equal(ct[0], g['op_ct'])
            assert seeds[0] == 0 and len(set(seeds[1:].tolist())) == 3
            for k in range(4):
                assert P101.validate(jl, pt, desc, amch[k], ct[k], ms[k]) == []
            sampled_differs |= bool(np.any(ms[1:] != ms[0]))
            if name == 'instance_000':
                r2 = P101.ten_roller(runner, LEGACY, '10x25', combo, name)
                again = P101.ten_rollouts(r2, '10x25', name, combo, 301, draws=4)
                for x, y in zip((ms, amch, ct, seeds), again):
                    assert np.array_equal(x, y)
                r3 = P101.ten_roller(runner, LEGACY, '10x25', combo, name)
                one = r3.rollout_batch(False, int(seeds[2]), 1)
                assert float(one[0][0]) == ms[2] and np.array_equal(one[1][0], amch[2])
    assert sampled_differs, 'no sampled schedule differs from the greedy one'


@needs_legacy
def test_cell_written_skipped_and_refused(tmp_path, runner):
    root = str(tmp_path)
    assert P101.run_polten(root, '10x25', 'W', 2, False, runner, LEGACY, draws=3)[0] == 'done'
    npz, js = P101.cell_paths(root, '10x25', 'W', P101.ten_label(LEGACY))
    z, rec = np.load(npz), json.load(open(js))
    assert z['makespan'].shape == (2, 3) and z['torch_seed'].shape == (2, 3)
    assert z['torch_seed'][:, 0].tolist() == [0, 0]
    assert rec['design']['kind'] == 'polten' and rec['design']['draws'] == 3
    assert rec['design']['trained_seed'] == 301 and rec['policy']['params'] == 32786
    assert rec['instances'] == P101.names_of('10x25', 2)
    assert rec['seeds']['instance_001'][0] == P101.ten_entropy('10x25', 'instance_001', 'W', 301, 1)
    assert int(z['torch_seed'][1, 1]) == P101.ten_torch_seed(rec['seeds']['instance_001'][0])
    assert rec['greedy_check'] == 'no stored greedy array or trace cell of this run'
    assert P101.run_polten(root, '10x25', 'W', 2, False, runner, LEGACY, draws=3)[0] == 'skip'
    with pytest.raises(SystemExit) as e:                      # another design
        P101.run_polten(root, '10x25', 'W', 2, False, runner, LEGACY, draws=4)
    assert 'REFUSING' in str(e.value)
    # a greedy trace cell of the same run that disagrees with schedule 0
    assert P101.run_trace(root, '10x25', 'W', 2, False, runner, LEGACY)[0] == 'done'
    tn, _ = P101.cell_paths(root, '10x25', 'W', P101.trace_label(LEGACY))
    t = dict(np.load(tn))
    assert np.array_equal(t['makespan'], z['makespan'][:, 0])
    t['makespan'] = t['makespan'] + 1.0
    P101.save_npz_atomic(tn, **t)
    with pytest.raises(SystemExit) as e:
        P101.run_polten(root, '10x25', 'W', 2, True, runner, LEGACY, draws=3)
    assert 'greedy schedule' in str(e.value)


# ---------------------------------------------------------------------------
# infer on synthetic cells
# ---------------------------------------------------------------------------

MODELS = ['10x25+family+fake-s301', '10x25+family+fake-s302']


def _infer_args(root, out, combos, models=MODELS, ten_arm='auto', ten_runs=None):
    return types.SimpleNamespace(root=root, out=out, partial=True, n=3, sizes=['10x25'],
                                 combos=list(combos), trace_arms=['SPT'],
                                 policy_seeds=[301, 302], trace_seeds=[301, 302],
                                 policy_models=list(models), ten_arm=ten_arm,
                                 ten_runs=None if ten_runs is None else
                                 P101.parse_ten_runs(ten_runs))


def _base_cells(root, combos, n=3):
    for combo in combos:
        P101.run_random(root, '10x25', combo, n, False)
        P101.run_trace(root, '10x25', combo, n, False)


def _write_ten(root, combo, model, ms, names):
    npz, js = P101.cell_paths(root, '10x25', combo, P101.ten_label(model))
    P101.save_npz_atomic(npz, makespan=np.asarray(ms, dtype=float),
                         torch_seed=np.zeros(np.shape(ms), dtype=np.uint64))
    P101.write_json_atomic(js, {'design': {'kind': 'polten', 'model': model,
                                           'draws': int(np.shape(ms)[1])},
                                'instances': names})


def _random_cell(root, combo):
    with np.load(P101.cell_paths(root, '10x25', combo, 'random')[0]) as z:
        return np.asarray(z['makespan'], dtype=float)


def _synthetic(root, combo, names):
    """Ten schedules per instance for two runs, built from the random cell: run 1 has a
    best of ten shorter than, equal to and longer than the random best on instances 0, 1
    and 2; run 2 is two hours shorter everywhere."""
    R = _random_cell(root, combo)
    rb = R.min(axis=1)
    X1 = np.repeat(rb[:, None], 10, axis=1) + np.arange(10)[None, :] * 0.0
    X1 = X1 + 3.0
    X1[0, 4], X1[1, 7], X1[2, :] = rb[0] - 2.0, rb[1], rb[2] + 1.0
    X2 = X1 - 2.0
    _write_ten(root, combo, MODELS[0], X1, names)
    _write_ten(root, combo, MODELS[1], X2, names)
    return R, X1, X2


def test_infer_with_ten_cells(tmp_path, monkeypatch):
    monkeypatch.setattr(P101, 'stored_policy', lambda *a: (None, []))
    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    _base_cells(root, ('N', 'LSW'))
    names = P101.names_of('10x25', 3)
    syn = {c: _synthetic(root, c, names) for c in ('N', 'LSW')}
    res = P101.infer(_infer_args(root, out, ('N', 'LSW')))
    assert res['ten_arm']['selected']['model'] == MODELS[0]
    assert res['ten_arm']['runs'] == MODELS
    assert res['ten_arm']['runs_per_size'] == {'10x25': 'all'}
    assert res['sizes']['10x25']['ten_runs'] == 'all'
    for combo, (R, X1, X2) in syn.items():
        T = res['sizes']['10x25']['combos'][combo]['ten']
        rb, rm = R.min(axis=1), R.mean(axis=1)
        rules = {r: P101.stored_rule('10x25', r, combo, 3) for r in P101.RULES + P101.AWARE}
        b4 = np.min(np.stack([rules[r] for r in P101.RULES]), axis=0)
        b7 = np.min(np.stack(list(rules.values())), axis=0)
        D = np.stack([P101.gap(X.min(axis=1), rb) for X in (X1, X2)])
        for k, m in enumerate(MODELS):
            row = T['rows'][m]
            assert row['randbest'] == {**P101.ci(D[k]), 'direction': row['randbest']['direction']}
            X = (X1, X2)[k]
            assert row['rule']['mean'] == pytest.approx(P101.gap(X.min(axis=1), b4).mean())
            assert row['rule7']['mean'] == pytest.approx(P101.gap(X.min(axis=1), b7).mean())
            assert row['greedy_rand']['mean'] == pytest.approx(P101.gap(X[:, 0], rm).mean())
            assert row['samp_rand']['mean'] == pytest.approx(
                P101.gap(X[:, 1:].mean(axis=1), rm).mean())
        assert T['rows'][MODELS[0]]['shares'] == dict(n=3, shorter=1, equal=1, longer=1,
                                                       shorter_pct=pytest.approx(100 / 3),
                                                       equal_pct=pytest.approx(100 / 3),
                                                       longer_pct=pytest.approx(100 / 3))
        assert T['pooled_shares']['n'] == 6
        hb = SF.hier_boot(D, B=P101.B, seed=11)
        P = T['pooled']['randbest']
        assert (P['mean'], P['lo'], P['hi']) == (hb['mean'], hb['lo'], hb['hi'])
        assert P['direction'] == SF.direction(hb['lo'], hb['hi'])
        assert T['seed_range']['randbest']['min'] == pytest.approx(min(D.mean(axis=1)))
    txt = open(os.path.join(out, 'manuscript', 'macros_maskctl.tex')).read()
    names_m = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(names_m) == len(set(names_m))
    assert all(re.fullmatch(r'Mctl[A-Za-z]+', x) for x in names_m)
    pten = [ln for ln in txt.splitlines() if 'Pten' in ln]
    assert pten and all('  % PARTIAL' in ln for ln in pten)
    for q in ('RandBest', 'Rule', 'RuleSeven', 'GreedyRand', 'SampRand'):
        assert f'\\MctlPten{q}PoolTenLSW' in txt and f'\\MctlPten{q}SelLoTenN' in txt
    assert '\\MctlPtenSelSeed}{301}' in txt and '\\MctlPtenRunNTen}{2}' in txt
    Y = res['sizes']['10x25']['ten_summary']['randbest']
    sel_means = [res['sizes']['10x25']['combos'][c]['ten']['rows'][MODELS[0]]['randbest']['mean']
                 for c in ('N', 'LSW')]
    assert (Y['selected_min'], Y['selected_max']) == (min(sel_means), max(sel_means))
    assert f'\\MctlPtenRandBestSelMinTen}}{{{P101.f2(min(sel_means))}}}' in txt
    assert '\\MctlPtenRandBestPoolAheadNTen' in txt
    md = open(os.path.join(out, 'results', 'mask_controls.md')).read()
    assert 'This size holds every run of the arm' in md and md.count('| LSW |') >= 2
    with pytest.raises(SystemExit) as e:                      # the form named, not held
        P101.infer(_infer_args(root, out, ('N', 'LSW'), ten_runs='selected'))
    assert 'asks for the selected run only' in str(e.value)
    P101.infer(_infer_args(root, out, ('N', 'LSW'), ten_runs='all'))


def test_infer_refusals(tmp_path, monkeypatch):
    monkeypatch.setattr(P101, 'stored_policy', lambda *a: (None, []))
    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    _base_cells(root, ('N', 'W'))
    names = P101.names_of('10x25', 3)

    def refused(*match, **kw):
        with pytest.raises(SystemExit) as e:
            P101.infer(_infer_args(root, out, ('N', 'W'), **kw))
        assert str(e.value).startswith('refused'), str(e.value)
        assert all(m in str(e.value) for m in match), str(e.value)
        assert not os.path.exists(os.path.join(out, 'results', 'mask_controls.json'))

    refused('no cell of the ten schedules', ten_arm='require')
    R = _random_cell(root, 'N')
    _write_ten(root, 'N', MODELS[0], np.repeat(R.min(axis=1)[:, None], 10, axis=1), names)
    refused('some runs or combinations only', '(runs per combination: N 1, W 0)')
    _write_ten(root, 'N', MODELS[1], np.repeat(R.min(axis=1)[:, None], 10, axis=1), names)
    refused('some runs or combinations only', '(runs per combination: N 2, W 0)')
    for m in MODELS:
        _write_ten(root, 'W', m, np.repeat(_random_cell(root, 'W').min(axis=1)[:, None], 9,
                                           axis=1), names)
    refused('schedules per instance')
    for m in MODELS:
        _write_ten(root, 'W', m, np.repeat(_random_cell(root, 'W').min(axis=1)[:, None], 10,
                                           axis=1), names)
    P101.infer(_infer_args(root, out, ('N', 'W')))            # complete: no refusal
    # a greedy trace cell of the first run that disagrees with its schedule 0
    tn, tj = P101.cell_paths(root, '10x25', 'N', P101.trace_label(MODELS[0]))
    P101.save_npz_atomic(tn, makespan=R.min(axis=1) + 1.0)
    P101.write_json_atomic(tj, {'instances': names})
    refused('greedy schedule')
    res = P101.infer(_infer_args(root, out, ('N', 'W'), ten_arm='off'))
    assert 'ten_arm' not in res


def test_infer_with_the_selected_run_only(tmp_path, monkeypatch):
    """A size that holds the cells of the selected run only (the first run of
    --policy_models here): its rows, the statement of the form, no pooled row and no
    pooled macro. A second run in one combination, or a run other than the selected one,
    is refused."""
    monkeypatch.setattr(P101, 'stored_policy', lambda *a: (None, []))
    root, out = str(tmp_path / 'cells'), str(tmp_path / 'out')
    _base_cells(root, ('N', 'LSW'))
    names = P101.names_of('10x25', 3)
    X = {}
    for c in ('N', 'LSW'):
        R, X1, X2 = _synthetic(root, c, names)
        os.remove(P101.cell_paths(root, '10x25', c, P101.ten_label(MODELS[1]))[0])
        X[c] = (R, X1)
    res = P101.infer(_infer_args(root, out, ('N', 'LSW')))
    S = res['sizes']['10x25']
    assert S['ten_runs'] == 'selected' and res['ten_arm']['runs_per_size'] == {'10x25': 'selected'}
    for c, (R, X1) in X.items():
        T = S['combos'][c]['ten']
        assert T['form'] == 'selected' and list(T['rows']) == [MODELS[0]]
        assert T['pooled'] == {} and T['pooled_shares'] is None and T['seed_range'] == {}
        d = P101.gap(X1.min(axis=1), R.min(axis=1))
        assert T['rows'][MODELS[0]]['randbest'] == {**P101.ci(d), 'direction':
                                                    T['rows'][MODELS[0]]['randbest']['direction']}
        assert T['rows'][MODELS[0]]['shares']['shorter'] == 1
    Y = S['ten_summary']['randbest']
    assert 'pooled_ahead' not in Y and Y['n'] == 2
    txt = open(os.path.join(out, 'manuscript', 'macros_maskctl.tex')).read()
    pten = re.findall(r'\\newcommand\{\\(MctlPten[^}]*)\}', txt)
    assert pten and not [x for x in pten
                         if re.search(r'Pool|SeedMin|SeedMax|SeedAheadN|SeedBehindN', x)]
    assert '\\MctlPtenRunNTen}{1}' in txt and 'the selected run only' in txt
    for q in ('RandBest', 'Rule', 'RuleSeven', 'GreedyRand', 'SampRand'):
        for m in (f'{q}SelTenN', f'{q}SelLoTenLSW', f'{q}SelHiTenN', f'{q}SelMinTen',
                  f'{q}SelMaxTen', f'{q}SelAheadNTen', f'{q}CombNTen'):
            assert f'\\MctlPten{m}}}' in txt, m
    assert '\\MctlPtenShorterSelTenN}{33.3}' in txt
    md = open(os.path.join(out, 'results', 'mask_controls.md')).read()
    assert 'This size holds the selected run only' in md and 'no pooled row' in md

    def refused(*match, **kw):
        with pytest.raises(SystemExit) as e:
            P101.infer(_infer_args(root, out, ('N', 'LSW'), **kw))
        assert str(e.value).startswith('refused'), str(e.value)
        assert all(m in str(e.value) for m in match), str(e.value)

    refused('asks for every run of the arm', ten_runs='all')
    P101.infer(_infer_args(root, out, ('N', 'LSW'), ten_runs='10x25=selected,'
                           '50x25n30=all,80x25n30=all'))
    _write_ten(root, 'N', MODELS[1], X['N'][1] - 2.0, names)
    refused('some runs or combinations only', '(runs per combination: N 2, LSW 1)')
    os.remove(P101.cell_paths(root, '10x25', 'N', P101.ten_label(MODELS[1]))[0])
    for c in ('N', 'LSW'):                  # the cells of a run other than the selected one
        os.rename(P101.cell_paths(root, '10x25', c, P101.ten_label(MODELS[0]))[0],
                  P101.cell_paths(root, '10x25', c, P101.ten_label(MODELS[1]))[0])
        os.rename(P101.cell_paths(root, '10x25', c, P101.ten_label(MODELS[0]))[1],
                  P101.cell_paths(root, '10x25', c, P101.ten_label(MODELS[1]))[1])
    refused('some runs or combinations only', '(runs per combination: N 1, LSW 1)')


def test_ten_runs_option():
    every = {s: 'all' for s in P101.SIZES}
    assert P101.parse_ten_runs('all') == every
    assert P101.parse_ten_runs('selected') == {s: 'selected' for s in P101.SIZES}
    design = P101.parse_ten_runs(P101.DEFAULT_TEN_RUNS)
    assert design == {'10x25': 'all', '50x25n30': 'selected', '80x25n30': 'selected'}
    assert P101.ten_runs_text(design) == P101.DEFAULT_TEN_RUNS
    assert P101.ten_runs_text(every) == 'all'
    assert P101.parse_ten_runs(' 80x25n30=selected, 10x25=all,50x25n30=selected') == design
    for bad in ('some', '10x25=all,50x25n30=selected', '10x25=all,50x25n30=x,80x25n30=all',
                '10x25=all,10x25=all,50x25n30=all,80x25n30=all', '20x25=all'):
        with pytest.raises(SystemExit):
            P101.parse_ten_runs(bad)
    cells = P101.cell_list(list(P101.SIZES), ['N'], 'polten', [301, 302, 303],
                           ten_runs=design, selected='sel')
    assert [c[3] for c in cells if c[0] == '10x25'] == [P101.policy_model(s) for s in
                                                       (301, 302, 303)]
    assert [c[3] for c in cells if c[0] != '10x25'] == ['sel', 'sel']
    other = P101.cell_list(list(P101.SIZES), ['N'], 'polten', [301, 302, 303],
                           ten_runs=design, selected='another')
    assert P101.assign_parts(cells, 3, None) == P101.assign_parts(other, 3, None)
    assert P101.cell_list(['10x25'], ['N'], 'policy', [301], ten_runs={'10x25': 'selected'},
                          selected='sel') == [('10x25', 'N', 'policy', P101.policy_model(301))]


@needs_legacy
def test_run_with_the_selected_run_only(tmp_path, capsys):
    """run --ten_runs selected computes the cells of the selected run (the first run of
    --policy_models) and touches no other run, whose checkpoint does not exist."""
    a = types.SimpleNamespace(sizes=['10x25'], combos=['W'], trace_arm='polten',
                              policy_seeds=[301, 302], policy_models=[LEGACY, MODELS[1]],
                              ten_runs=P101.parse_ten_runs('selected'), part=None, parts=1,
                              n=1, root=str(tmp_path), force=False, draws=2)
    P101.run(a)
    assert f'the selected run {LEGACY} at 10x25' in capsys.readouterr().out
    assert sorted(os.listdir(os.path.join(str(tmp_path), '10x25', 'W'))) == [
        f'{P101.ten_label(LEGACY)}.json', f'{P101.ten_label(LEGACY)}.npz']


# ---------------------------------------------------------------------------
# without the arm, today's outputs
# ---------------------------------------------------------------------------

def _reference_module():
    src = subprocess.run(['git', 'show', f'{REFERENCE_COMMIT}:scripts/p101_mask_controls.py'],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType('p101_reference')
    mod.__file__ = os.path.join(ROOT, 'scripts', 'p101_mask_controls.py')
    argv = sys.argv
    try:
        exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
    finally:
        sys.argv = argv
    return mod


@pytest.mark.skipif(not has_commit(REFERENCE_COMMIT),
                    reason=f'needs the git history of the repository (commit {REFERENCE_COMMIT}), '
                           f'which the release does not hold')
def test_without_the_arm_outputs_equal_the_reference_commit(tmp_path, monkeypatch):
    ref = _reference_module()
    assert not hasattr(ref, 'run_polten')
    for m in (P101, ref):
        monkeypatch.setattr(m, 'stored_policy', lambda *a: (None, []))
    root = str(tmp_path / 'cells')
    _base_cells(root, ('N', 'W'), n=2)
    outs = {}
    for key, mod, ten in (('reference', ref, None), ('auto', P101, 'auto'), ('off', P101, 'off')):
        a = _infer_args(root, str(tmp_path / key), ('N', 'W'), models=[], ten_arm=ten)
        a.n, a.policy_models = 2, None
        if ten is None:
            del a.ten_arm
        with redirect_stdout(io.StringIO()):
            mod.infer(a)
        outs[key] = {f: open(os.path.join(str(tmp_path / key), p), 'rb').read()
                     for f, p in P101.OUT_FILES.items()}
    assert outs['auto'] == outs['reference'] == outs['off']
    assert b'Pten' not in outs['auto']['macros'] and b'ten_arm' not in outs['auto']['json']


# ---------------------------------------------------------------------------
# plan and folder isolation
# ---------------------------------------------------------------------------

def test_plan_of_the_arm(capsys, monkeypatch, tmp_path):
    """The design of the paper (every run at 10x25, the selected run at 50 and 80 modules)
    and every run at every size; the selected run cannot be determined before the
    exact-mask runs exist, and the parts do not depend on it. The plan runs in an empty
    folder, so no validation record of the repository is found."""
    monkeypatch.chdir(tmp_path)
    cost = {s: P101.COST_S[s]['polten'] * P101.SIZES[s]['n'] * 8 / 3600 for s in P101.SIZES}
    for text, runs, cells in ((P101.DEFAULT_TEN_RUNS, {'10x25': 5, '50x25n30': 1,
                                                       '80x25n30': 1}, 56),
                              ('all', {s: 5 for s in P101.SIZES}, 120)):
        a = types.SimpleNamespace(sizes=list(P101.SIZES), combos=list(P101.R8),
                                  trace_arm='polten', policy_seeds=[301, 302, 303, 304, 305],
                                  policy_models=None, parts=4, n=None,
                                  ten_runs=P101.parse_ten_runs(text))
        r = P101.plan(a)
        txt = capsys.readouterr().out
        assert r['cells'] == cells and r['size_runs'] == runs
        assert r['core_h'] == pytest.approx(sum(cost[s] * runs[s] for s in P101.SIZES))
        assert r['size_h'] == pytest.approx({s: cost[s] * runs[s] for s in P101.SIZES})
        cmds = [x for x in txt.splitlines() if 'p101_mask_controls.py run' in x]
        assert len(cmds) == 4 and all('--trace_arm polten' in x and ' -B ' in x
                                      and f'--ten_runs {text} ' in x for x in cmds)
        assert ('cannot be determined yet' in txt) == (text != 'all')
    assert round(sum(cost[s] * n for s, n in (('10x25', 5), ('50x25n30', 1),
                                               ('80x25n30', 1))), 1) == 17.6
    assert round(sum(cost.values()) * 5, 1) == 62.7


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


def test_new_cells_are_matched_by_no_wildcard_pattern():
    """The cells of the arm sit beside the other cells of the script; no literal path
    pattern with a wildcard below test_results/FAMILY in another file matches them."""
    base = P101.DEFAULT_ROOT
    examples = [f'{base}/10x25/LS/polten_10x25+family+xmask-v1-s301.npz',
                f'{base}/80x25n30/LSW/polten_10x25+family+xmask-v1-s305.json']
    rx = re.compile('test_results' + r'/FAMILY/[^\'"\s)]*')
    patterns = {}
    for f in _scanned_files():
        for m in rx.findall(open(f, errors='replace').read()):
            first = m.split('/')[2] if m.count('/') >= 2 else ''
            if any(ch in first for ch in '*?['):
                patterns.setdefault(m, f)
    assert patterns, 'the scan found no wildcard pattern; the scan itself is broken'
    for pat, f in patterns.items():
        for ex in examples:
            assert not fnmatch.fnmatch(ex, pat), f'{f}: {pat} matches {ex}'
