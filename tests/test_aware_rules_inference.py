"""Tests for the inference step of scripts/p72_aware_rules.py, on synthetic arrays.

  * the policy first: d = 100 (policy - X) / X per training run and instance, X one
    of ECT, SSPT, LLF, the per-instance best of the three and the best of all seven
    rules; intervals from stats_family.hier_boot with a fresh default_rng(11), the
    pooled row on [k, n, 8];
  * Holm over the eight combinations of a size and the combined verdict; the pooled
    row carries the interval and its direction only;
  * the best of the three further rules against the best of the four classical
    rules: stats_family.instance_boot with a fresh default_rng(7), Holm per
    direction, and its verdict (pooled row: interval and direction);
  * the sizes of the two environments: the exact one runs the 30-instance
    production sets 50x25n30 and 80x25n30 (every descriptor, the -n30 instance
    folders), the earlier one keeps 50x25 and 80x25 with the first --n_prod;
  * the earlier environment runs the step only with --inference_out outside the
    repository's paper and results folders; the exact environment writes in place
    or under that folder;
  * a missing rule or policy array and base instances that differ between
    combinations are refused, and no output of the step is left;
  * the summary calls the step only with all three further rules, and its own
    outputs are those it writes without the step;
  * the compact table: plain heads, bold on the lowest mean makespan of each row, the
    production table at 50 and 80 modules with n = 30, no mark on the pooled row.

Run from the repository root:  python3 -m pytest tests/test_aware_rules_inference.py -q
Synthetic arrays only (the descriptors of data/FAMILY are read, never written).
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None, PACT_FIRST_MASK=None,
                                        PACT_MASK_TOL=None)


def _load_p72():
    argv = sys.argv
    sys.argv = argv[:1]
    try:
        spec = importlib.util.spec_from_file_location(
            'p72_inference_under_test', os.path.join(ROOT, 'scripts', 'p72_aware_rules.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


P72 = _load_p72()
REG = P72.REG
import stats_family as SF  # noqa: E402
from p89_public_family import superiority_verdict  # noqa: E402

B = 200
N = 6
SEEDS = (301, 302, 303, 304, 305)
ALL7 = P72.NEW_RULES + P72.CLASSICAL


@pytest.fixture
def exact(monkeypatch, tmp_path):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    monkeypatch.setenv('PACT_RESULTS', str(tmp_path / 'res'))
    monkeypatch.setenv('PACT_OUT', str(tmp_path / 'out'))
    monkeypatch.setattr(P72, 'ENV', 'exact')
    yield tmp_path


def n_of(size):
    """Instances per combination of the synthetic arrays: all 30 on a 30-instance set."""
    return 30 if size in P72.N30 else N


def base_of(size, g):
    i = (P72.SIZES + P72.N30).index(size)
    return 100.0 + 7.0 * np.arange(n_of(size)) + 2.0 * P72.REGIMES.index(g) + 50 * i


def rules_of(size, g):
    """Classical rules: SPT the best, 2% over the base. Further rules: ECT at the base
    on even instances and 3% over on odd ones; SSPT 1% over with SETUP and LLF 1.5%
    (even) or 0.5% (odd) over with LAG, otherwise both equal to SPT, as their
    definitions force (the summary checks these identities)."""
    b = base_of(size, g)
    even = (np.arange(n_of(size)) % 2 == 0)
    spt = b * 1.02
    return {'FIFO': b * 1.10, 'MOR': b * 1.12, 'SPT': spt, 'MWKR': b * 1.05,
            'ECT': np.where(even, b, b * 1.03),
            'SSPT': b * 1.01 if 'S' in g else spt,
            'LLF': np.where(even, b * 1.015, b * 1.005) if 'L' in g else spt}


def policy_of(size, g, s):
    return base_of(size, g) * (0.98 - 0.002 * (s - 301))


def build(sizes=('10x25',), two_col=False, skip=()):
    for size in sizes:
        d = P72.rule_dir(size)
        os.makedirs(d, exist_ok=True)
        for g in P72.REGIMES:
            for r, v in rules_of(size, g).items():
                if (size, g, r) in skip:
                    continue
                arr = np.stack([v, np.ones(len(v))], 1) if two_col else v
                np.save(P72.result_path(size, r, g), arr)
            for s in REG.seeds():
                p = REG.result_path('released', None, s, size, g)
                if (size, g, s) in skip:
                    continue
                np.save(p, np.concatenate([policy_of(size, g, s), [999.0, 999.0]]))


def test_earlier_environment_needs_an_output_folder(tmp_path):
    assert P72.ENV == 'legacy'
    assert P72.inference(['10x25'], None, B=B) is None
    for bad in (ROOT, os.path.join(ROOT, 'results', 'x'), os.path.join(ROOT, 'manuscript'),
                os.path.join(ROOT, 'paper', 'tables'), os.path.join(ROOT, 'test_results')):
        with pytest.raises(P72.InferenceRefusal, match='paper, results'):
            P72.inference(['10x25'], bad, B=B)
    assert P72.inf_out_root(str(tmp_path)) == str(tmp_path)


def test_exact_environment_writes_in_place_by_default(exact):
    assert P72.inf_out_root(None) == P72.ROOT
    assert P72.inf_out_root(P72.ROOT) == P72.ROOT
    with pytest.raises(P72.InferenceRefusal):
        P72.inf_out_root(os.path.join(ROOT, 'results'))


def test_numbers_holm_and_verdicts(exact):
    build()
    out = exact / 'inf'
    res = P72.inference(['10x25'], str(out), B=B)
    S = res['sizes']['10x25']
    assert S['n'] == N and S['k'] == 5
    D = {}
    for g in P72.REGIMES:
        R = rules_of('10x25', g)
        P = np.stack([policy_of('10x25', g, s) for s in SEEDS])
        best3 = np.min([R[r] for r in P72.NEW_RULES], axis=0)
        best4 = np.min([R[r] for r in P72.CLASSICAL], axis=0)
        best7 = np.min([R[r] for r in ALL7], axis=0)
        assert np.array_equal(best7, best3)
        for q, v in (('ECT', R['ECT']), ('SSPT', R['SSPT']), ('LLF', R['LLF']),
                     ('best3', best3), ('best7', best7)):
            d = 100 * (P - v) / v
            D.setdefault(q, {})[g] = d
            hb = SF.hier_boot(d, B=B, seed=11)
            c = S['policy_vs'][q][g]
            assert c['mean'] == pytest.approx(d.mean(), abs=1e-12)
            assert (c['lo'], c['hi']) == (hb['lo'], hb['hi'])
            assert c['mean'] < 0 and c['verdict'] == 'ahead'
        if 'L' in g:                        # the per-instance minimum mixes rules
            assert not np.array_equal(best3, R['ECT']) and not np.array_equal(best3, R['LLF'])
        e = 100 * (best3 - best4) / best4
        ib = SF.instance_boot(e, B=B, seed=7)
        c = S['best3_vs_best4'][g]
        assert (c['mean'], c['lo'], c['hi']) == (pytest.approx(ib['mean']), ib['lo'], ib['hi'])
        assert c['mean'] < 0
    for q in P72.INF_QUANTITIES:
        rows = S['policy_vs'][q]
        for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
            assert [rows[g][col + '_holm'] for g in P72.REGIMES] == \
                SF.holm([rows[g][col] for g in P72.REGIMES])
        for g in P72.REGIMES:
            c = rows[g]
            assert c['verdict'] == superiority_verdict(
                c['lo'], c['hi'], c['p_t_holm'], c['p_boot_holm'], c['p_t_greater_holm'],
                c['p_boot_greater_holm'], 0.05)
        d3 = np.stack([D[q][g] for g in P72.REGIMES], axis=2)
        hb = SF.hier_boot(d3, B=B, seed=11)
        pool = rows['pool']
        assert (pool['lo'], pool['hi'], pool['c']) == (hb['lo'], hb['hi'], 8)
        assert pool['direction'] == SF.direction(pool['lo'], pool['hi'])
        assert not {'verdict', 'p_t', 'p_t_holm', 'holm_family'} & set(pool)
        assert all(rows[g]['holm_family'] == list(P72.REGIMES) for g in P72.REGIMES)
    tb = S['best3_vs_best4']
    for col in ('p_boot', 'p_boot_greater'):
        assert [tb[g][col + '_holm'] for g in P72.REGIMES] == \
            SF.holm([tb[g][col] for g in P72.REGIMES])
    e2 = np.stack([100 * (np.min([rules_of('10x25', g)[r] for r in P72.NEW_RULES], axis=0)
                          - np.min([rules_of('10x25', g)[r] for r in P72.CLASSICAL], axis=0))
                   / np.min([rules_of('10x25', g)[r] for r in P72.CLASSICAL], axis=0)
                   for g in P72.REGIMES], axis=1)
    ib = SF.instance_boot(e2, B=B, seed=7)
    tp = tb['pool']
    assert (tp['lo'], tp['hi'], tp['c']) == (ib['lo'], ib['hi'], 8)
    assert tp['direction'] == SF.direction(tp['lo'], tp['hi'])
    assert not {'verdict', 'p_boot', 'p_boot_holm'} & set(tp) and 'ties' in tp
    for g in P72.REGIMES:
        c = tb[g]
        want = ('ahead' if c['hi'] < 0 and c['p_boot_holm'] < 0.05 else
                'behind' if c['lo'] > 0 and c['p_boot_greater_holm'] < 0.05 else
                'not distinguishable')
        assert c['verdict'] == want
    assert tb['LSW']['verdict'] == 'ahead'
    assert S['counts']['best3']['ahead'] == 8


def test_outputs_and_macro_names(exact):
    build()
    out = exact / 'inf'
    P72.inference(['10x25'], str(out), B=B)
    mac = (out / 'manuscript' / 'macros_aware_rules.tex').read_text()
    assert mac.startswith('% Generated by scripts/p72_aware_rules.py (inference step) -- '
                          'do not hand-edit')
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', mac)
    assert names and all(re.fullmatch(r'Awr[A-Za-z]+', x) for x in names)
    assert len(names) == len(set(names))
    for x in ('AwrTenPolBestSevenLSWVerdict', 'AwrTenPolEctPoolLo', 'AwrTenThreeFourSHi',
              'AwrTenN', 'AwrSeedN', 'AwrBoot', 'AwrTenPolLlfAheadCount',
              'AwrTenPolEctPoolDir', 'AwrTenThreeFourPoolDir', 'AwrTenThreeFourLSVerdict'):
        assert x in names
    assert not [x for x in names if re.search(r'PoolVerdict$', x)]
    assert 'one test' not in mac
    js = json.loads((out / 'results' / 'aware_rules_inference.json').read_text())
    assert js['environment'] == 'exact' and js['seeds'] == list(SEEDS)
    assert js['tables_skipped'] == ['table', 'table_prod']
    assert js['policy'] == 'greedy+10x25+family+xmask-v1'
    assert (out / 'results' / 'aware_rules_inference.md').exists()
    assert not (out / 'manuscript' / 'tables').exists()


def test_refusals_leave_no_output(exact):
    build(skip={('10x25', 'LS', 'MWKR')})
    out = exact / 'inf'
    stale = out / 'results' / 'aware_rules_inference.json'
    stale.parent.mkdir(parents=True)
    stale.write_text('{}')
    with pytest.raises(P72.InferenceRefusal, match='rule arrays missing'):
        P72.inference(['10x25'], str(out), B=B)
    assert not any(p.is_file() for p in out.rglob('*'))
    build()
    os.remove(REG.result_path('released', None, 304, '10x25', 'W'))
    with pytest.raises(P72.InferenceRefusal, match='missing result file'):
        P72.inference(['10x25'], str(out), B=B)


def test_policy_shorter_than_the_rules_is_refused(exact):
    build()
    p = REG.result_path('released', None, 302, '10x25', 'SW')
    np.save(p, policy_of('10x25', 'SW', 302)[:N - 1])
    with pytest.raises(P72.InferenceRefusal, match='fewer than'):
        P72.inference(['10x25'], str(exact / 'inf'), B=B)


def test_base_instances_must_agree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for g in P72.REGIMES:
        d = tmp_path / 'data' / 'FAMILY' / '10x25' / 'desc' / g
        d.mkdir(parents=True)
        for i in range(3):
            base = f'b{i}' if not (g == 'LSW' and i == 2) else 'other'
            (d / f'instance_{i:03d}.desc.json').write_text(json.dumps({'base_instance': base}))
    assert P72.inf_bases('10x25', 2) == ['instance_000', 'instance_001']
    with pytest.raises(P72.InferenceRefusal, match='base instances'):
        P72.inf_bases('10x25', 3)
    with pytest.raises(P72.InferenceRefusal, match='descriptors'):
        P72.inf_bases('10x25', 4)


def test_summary_calls_the_step_and_keeps_its_own_outputs(exact, capsys, monkeypatch):
    """The summary's stream is one default_rng(7) per process, so each call here
    starts from a fresh one, as a new run would."""
    build(two_col=True)
    out, inf = exact / 'out', exact / 'inf'
    monkeypatch.setattr(P72, 'rng', np.random.default_rng(7))
    assert P72.summarize(list(P72.NEW_RULES), str(inf)) is None
    assert (out / 'results' / 'aware_rules.json').exists()
    assert (inf / 'manuscript' / 'macros_aware_rules.tex').exists()
    own = (out / 'results' / 'aware_rules.json').read_bytes()
    for p in inf.rglob('*'):
        if p.is_file():
            p.unlink()
    monkeypatch.setattr(P72, 'rng', np.random.default_rng(7))
    assert P72.summarize(['ECT'], str(inf)) is None
    assert 'inference step skipped' in capsys.readouterr().err
    assert not any(p.is_file() for p in inf.rglob('*'))
    assert (out / 'results' / 'aware_rules.json').read_bytes() != own   # one rule only
    monkeypatch.setattr(P72, 'rng', np.random.default_rng(7))
    P72.summarize(list(P72.NEW_RULES), str(inf))
    assert (out / 'results' / 'aware_rules.json').read_bytes() == own


def test_compact_table(exact):
    """Production table of the exact environment: the 30-instance sets, all 30
    instances per combination (the base instances are read from the descriptors of
    data/FAMILY/50x25n30 and 80x25n30)."""
    build(sizes=P72.N30)
    out = exact / 'inf'
    res = P72.inference(list(P72.N30), str(out), B=B)
    assert res['tables_skipped'] == ['table']
    assert res['tables']['table_prod'] == list(P72.N30)
    assert res['sizes_missing'] == ['10x25', '20x25', '30x25']
    assert all(res['sizes'][s]['n'] == 30 for s in P72.N30)
    t = (out / 'manuscript' / 'tables' / 'aware_rules_table_prod.tex').read_text()
    assert 'combination & ECT & SSPT & LLF & policy & ECT & SSPT & LLF' in t
    assert 'Regime' not in t and 'PACT' not in t and 'regime' not in t
    for m in (50, 80):
        assert ('\\multicolumn{8}{@{}l}{\\textit{' + f'{m} modules, $n = 30$ instances '
                'per combination}} \\\\') in t
    rows = [ln for ln in t.splitlines() if ln.startswith(('$', 'pooled'))]
    assert len(rows) == 18
    for ln in rows:                         # the policy is the shortest in every row
        cells = ln.split(' & ')
        assert cells[4].startswith('\\textbf{') and '\\textbf' not in ''.join(cells[1:4])
        assert ('$^{*}$' in ln) == (not ln.startswith('pooled'))
    mac = (out / 'manuscript' / 'macros_aware_rules.tex').read_text()
    assert '\\newcommand{\\AwrFiftyN}{30}' in mac and '\\newcommand{\\AwrEightyN}{30}' in mac


def test_sizes_of_the_two_environments(monkeypatch):
    assert P72.ENV == 'legacy' and P72.active_sizes() == P72.SIZES
    assert P72.SIZES == ('10x25', '20x25', '30x25', '50x25', '80x25')
    assert P72.inf_tables()['table_prod'] == ('50x25', '80x25')
    assert P72.size_words() == {'10x25': 'Ten', '80x25': 'Eighty'}
    assert P72.inst_base('50x25') == 'data/PPVC/50x25+ppvc-mixed'
    assert len(P72.descriptors('50x25', 'N', 10)) == 10
    monkeypatch.setattr(P72, 'ENV', 'exact')
    assert P72.active_sizes() == ('10x25', '20x25', '30x25', '50x25n30', '80x25n30')
    assert P72.inf_tables()['table_prod'] == P72.N30
    assert P72.size_words() == {'10x25': 'Ten', '80x25n30': 'Eighty'}
    for size in P72.N30:
        assert P72.inst_base(size) == f'data/PPVC/{size[:5]}+ppvc-mixed-n30'
        for g in P72.REGIMES:
            ds = P72.descriptors(size, g, 10)          # --n_prod has no effect here
            assert len(ds) == 30
            assert all(stem.startswith(P72.inst_base(size) + os.sep) for _, stem in ds)
        assert P72.inf_bases(size, 30) == [os.path.basename(d)[:-len('.desc.json')]
                                           for d, _ in P72.descriptors(size, 'N', 0)]
    monkeypatch.setattr(P72, '_ARGV', ['--sizes', '50x25'])
    with pytest.raises(SystemExit, match='unknown sizes'):
        P72.main()
