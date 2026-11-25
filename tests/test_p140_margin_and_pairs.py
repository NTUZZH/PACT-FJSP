"""scripts/p140_margin_and_pairs.py on small synthetic arrays.

  * the best rule is the per-instance minimum of the rule schedules;
  * the options B1 to B3 serve each combination with the specialist they name, and B4 with
    the specialist of the combination;
  * B4 equals a direct computation by section A of p88 (parity_components and pooled_block
    with a loader of the synthetic arrays);
  * the verdict words at the 0.5% and 1% margins and their counts;
  * a missing array or record refuses with a message that names it;
  * the macro names are letters only, carry the two prefixes, are unique and are defined in no
    macro file of the repository;
  * the oracle pairs pick the better specialist of each pair per combination.

Run from the repository root:  python -m pytest tests/test_p140_margin_and_pairs.py -q
CPU only; nothing is written outside the temporary folders of pytest.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import p88_family_inference as F88  # noqa: E402
import p110_seed_extension as P110  # noqa: E402
import p140_margin_and_pairs as P  # noqa: E402
import stats_family as SF  # noqa: E402

R8 = P.R8
B_SMALL = 300
SEEDS = [301, 302, 303]
N = 30


def synthetic(seed=0, k=SEEDS, n=N):
    """pol[s][C] and spec[X][s][C]: the specialist X run under C has makespans 500 + 10 * (index
    of X) + noise, the policy 505 + noise, so a difference identifies the specialist read."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(400, 600, n)
    pol = {s: {c: base + 5 + rng.normal(0, 1, n) for c in R8} for s in k}
    spec = {x: {s: {c: base + 10 * i + rng.normal(0, 1, n) for c in R8} for s in k}
            for i, x in enumerate(R8)}
    return pol, spec


# ----------------------------------------------------------------- best rule
def test_best_rule_is_the_per_instance_minimum():
    a = np.array([5.0, 9.0, 3.0, 7.0])
    b = np.array([6.0, 2.0, 4.0, 7.0])
    c = np.array([4.0, 8.0, 5.0, 1.0])
    d = np.array([9.0, 9.0, 9.0, 9.0])
    assert P.best_rule([a, b, c, d]).tolist() == [4.0, 2.0, 3.0, 1.0]
    with pytest.raises(P.Refused, match='differ in length'):
        P.best_rule([a, b[:3]])


# ----------------------------------------------------------------- options
def test_the_options_serve_each_combination_with_the_named_specialist():
    assert P.OPTIONS['One']['serve'] == {'N': 'N', 'S': 'N', 'W': 'N', 'SW': 'N',
                                         'L': 'LSW', 'LS': 'LSW', 'LW': 'LSW', 'LSW': 'LSW'}
    assert P.OPTIONS['Two']['serve'] == {'N': 'N', 'S': 'N', 'W': 'N', 'SW': 'N',
                                         'L': 'L', 'LS': 'L', 'LW': 'L', 'LSW': 'L'}
    assert P.OPTIONS['Three']['serve'] == {c: 'N' for c in R8}
    assert P.OPTIONS['Four']['serve'] == {c: c for c in R8}
    assert [P.OPTIONS[k]['name'] for k in ('One', 'Two', 'Three', 'Four')] == ['B1', 'B2', 'B3', 'B4']


@pytest.mark.parametrize('key', ['One', 'Two', 'Three', 'Four'])
def test_option_d_reads_the_arrays_of_the_serving_specialist(key):
    pol, spec = synthetic()
    serve = P.OPTIONS[key]['serve']
    D = P.option_d(pol, spec, serve, SEEDS)
    assert list(D) == R8
    for c in R8:
        want = np.stack([100 * (pol[s][c] - spec[serve[c]][s][c]) / spec[serve[c]][s][c]
                         for s in SEEDS])
        assert D[c].shape == (len(SEEDS), N)
        assert np.array_equal(D[c], want)
        # every other specialist gives another difference on this combination
        for x in R8:
            if x != serve[c]:
                other = np.stack([100 * (pol[s][c] - spec[x][s][c]) / spec[x][s][c] for s in SEEDS])
                assert not np.allclose(D[c], other)


def test_b4_equals_section_a_of_p88(monkeypatch):
    pol, spec = synthetic(1)

    def loader(arm, combo, s, folder, r, n=None):
        x = pol[s][r] if arm == 'released' else spec[combo][s][r]
        return x if n is None else x[:n]
    monkeypatch.setattr(F88, 'B', B_SMALL)
    D4 = P.option_d(pol, spec, P.OPTIONS['Four']['serve'], SEEDS)
    comps, pooled = P.option_rows(D4, B_SMALL)
    with P110.p88_setting(loader=loader):
        Dd, Cd = F88.parity_components('10x25', seeds=SEEDS)
        pd = F88.pooled_block(Dd, {'All': R8})['All']
    for r in R8:
        assert np.array_equal(D4[r], Dd[r])
        for k in ('mean', 'lo', 'hi', 'sim_lo', 'sim_hi', 't_lo', 't_hi', 'v0.5', 'v1.0',
                  'direction'):
            assert comps[r][k] == Cd[r][k], (r, k)
    for k in ('mean', 'lo', 'hi', 't_lo', 't_hi', 'v0.5', 'v1.0', 'direction'):
        assert pooled[k] == pd[k], k
    # the principal check accepts its own numbers and refuses a changed one
    fam = dict(parity=dict(components=Cd, pooled={'All': pd}))
    assert P.check_principal(dict(components=comps, pooled=pooled), fam)['compared'] == 8 * 10 + 8
    bad = dict(parity=dict(components=dict(Cd, N=dict(Cd['N'], lo=Cd['N']['lo'] + 0.02)),
                           pooled={'All': pd}))
    with pytest.raises(P.Refused, match='does not reproduce'):
        P.check_principal(dict(components=comps, pooled=pooled), bad)


def test_every_option_meets_the_same_resamples():
    """The per-combination draws of every option come from the same stream, so two options
    with the same differences on a combination get the same interval there."""
    pol, spec = synthetic(2)
    D1 = P.option_d(pol, spec, P.OPTIONS['One']['serve'], SEEDS)
    D3 = P.option_d(pol, spec, P.OPTIONS['Three']['serve'], SEEDS)
    c1, _ = P.option_rows(D1, B_SMALL)
    c3, _ = P.option_rows(D3, B_SMALL)
    for c in ('N', 'S', 'W', 'SW'):                     # served by N in both options
        assert (c1[c]['lo'], c1[c]['hi']) == (c3[c]['lo'], c3[c]['hi'])


# ----------------------------------------------------------------- verdicts
@pytest.mark.parametrize('lo,hi,half,one,direction', [
    (-0.30, 0.40, 'equivalent', 'equivalent', 'neither'),
    (-0.70, 0.40, 'non-inferior only', 'equivalent', 'neither'),
    (0.18, 0.89, 'inferior', 'equivalent', 'behind'),
    (-0.20, 1.30, 'inconclusive', 'inconclusive', 'neither'),
    (0.60, 1.40, 'inferior', 'inferior', 'behind'),
    (-1.40, -0.60, 'non-inferior only', 'non-inferior only', 'ahead'),
    (-0.68, 0.44, 'non-inferior only', 'equivalent', 'neither'),
])
def test_verdict_words_at_the_two_margins(lo, hi, half, one, direction):
    v = P.verdict_words(lo, hi)
    assert v == {'v0.5': half, 'v1.0': one, 'direction': direction}


def test_verdict_counts_name_the_combinations():
    rows = {c: P.verdict_words(-0.3, 0.4) for c in R8}
    rows['S'] = P.verdict_words(0.6, 1.4)
    rows['SW'] = P.verdict_words(-0.2, 1.3)
    cnt = P.verdict_counts(rows, R8)
    assert cnt['v1.0']['equivalent'] == ['N', 'L', 'W', 'LS', 'LW', 'LSW']
    assert cnt['v1.0']['inferior'] == ['S'] and cnt['v1.0']['inconclusive'] == ['SW']
    assert cnt['v0.5']['non-inferior only'] == []
    assert sum(len(v) for v in cnt['v0.5'].values()) == 8


# ----------------------------------------------------------------- refusals
def test_a_missing_rule_array_refuses_with_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_RESULTS', str(tmp_path / 'results'))
    with pytest.raises(P.Refused, match=r'missing input array: .*Result_FIFO_N\.npy'):
        P.rule_arrays(100)


def test_a_missing_specialist_array_refuses_with_its_name(tmp_path, monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    folder = tmp_path / 'results' / '10x25_xmask'
    folder.mkdir(parents=True)
    monkeypatch.setenv('PACT_RESULTS', str(tmp_path / 'results'))
    with pytest.raises(P.Refused, match=r'missing input array: .*xmask-v1-s301_N\.npy'):
        P.principal_arrays((301, 306), 100)


def test_a_missing_record_refuses_with_the_script_that_writes_it(tmp_path, monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    monkeypatch.setenv('PACT_OUT', str(tmp_path / 'out'))
    with pytest.raises(P.Refused, match=r'missing input .*seed_extension\.json: written by '
                                        r'scripts/p110_seed_extension\.py \(step seed_extension'):
        P.read_record('seedext')


def test_the_setting_is_checked(monkeypatch):
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-303')
    monkeypatch.setenv('PACT_OUT', 'out/exact')
    with pytest.raises(P.Refused, match='PACT_SEEDS=301-305'):
        P.check_setting()
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(P.Refused, match='exact'):
        P.check_setting()


# ----------------------------------------------------------------- macros
def fake_results():
    """Result structures of both analyses from synthetic differences (small B)."""
    rng = np.random.default_rng(3)
    D = {r: rng.normal(0.2, 1.0, (3, N)) for r in R8}
    comps = P.stream_components(D, B_SMALL, 11)
    pooled = P.P95.component(np.stack([D[r] for r in R8], axis=2), B_SMALL)[0]
    counts = P.verdict_counts(comps, R8)
    cells = {f'{x}|{c}': comps[c] for x in R8 for c in R8 if c != x}
    block = dict(seeds=SEEDS, rows=comps, pooled=pooled, counts=counts)
    A = dict(n=N,
             rule_minus_specialist=dict(ten=dict(components=comps, pooled=pooled, counts=counts)),
             read=dict(policy_minus_specialist=dict(components=comps, pooled=pooled, seeds=SEEDS),
                       policy_minus_rule=dict(components={r: dict(comps[r], verdict='ahead')
                                                          for r in R8},
                                              pooled=pooled, seeds=SEEDS)),
             others=dict(sparse=block, budget=block,
                         cross=dict(seeds=SEEDS, cells=cells, pooled=pooled,
                                    counts=P.verdict_counts(cells, list(cells)),
                                    counts_by_specialist=counts, counts_by_combination=counts)))
    Bres = {k: dict(components=comps, pooled=pooled, counts=counts, ahead=['S'], behind=[])
            for k in P.OPTIONS}
    return A, Bres


def test_macro_names_are_unique_and_new(tmp_path):
    A, Bres = fake_results()
    lines = P.macro_lines(A, Bres)
    names = P.macro_names(lines)
    assert len(names) == len(set(names))
    assert all(n.startswith(('Msn', 'Tws')) and n.isalpha() for n in names)
    # every macro line carries a comment
    assert all('  % ' in ln for ln in lines if ln.startswith('\\newcommand'))
    # no name or prefix is defined in a macro file of the repository
    assert P.check_names(names, str(tmp_path / 'manuscript' / 'macros_margin_pairs.tex')) == len(names)
    for k in ('One', 'Two', 'Three', 'Four'):
        for r in R8:
            assert f'Tws{k}{r}' in names and f'Tws{k}{r}VOne' in names
    for r in R8:
        assert {f'MsnRule{r}', f'MsnPol{r}', f'MsnPolRule{r}Verdict', f'TwsOne{r}Spec'} <= set(names)
    assert {'MsnRuleEqOneN', 'MsnRuleInfOneN', 'MsnRuleEqHalfList', 'MsnCrossPoolVOne',
            'MsnSparseInfOneN', 'MsnBudgetEqOneN', 'MsnCrossRowEqOneN', 'TwsOnePool'} <= set(names)


def test_a_duplicate_or_foreign_name_refuses(tmp_path):
    with pytest.raises(P.Refused, match='duplicate'):
        P.check_names(['MsnA', 'MsnA'], str(tmp_path / 'm.tex'))
    with pytest.raises(P.Refused, match='Msn or Tws'):
        P.check_names(['FamRuleN'], str(tmp_path / 'm.tex'))


# ----------------------------------------------------------------- oracle pairs
def test_oracle_pairs_pick_the_better_specialist_per_combination():
    pol, spec = synthetic(4)
    rows = P.oracle_pairs(pol, spec, SEEDS)
    assert len(rows) == 28 and [x['rank'] for x in rows] == list(range(1, 29))
    # specialist N has the shortest schedules everywhere, then L: the best pair is (N, L), and
    # N serves every combination
    assert rows[0]['pair'] == ['N', 'L'] and set(rows[0]['serve'].values()) == {'N'}
    assert rows[-1]['pair'] == ['SW', 'LSW']
    vals = [x['policy_minus_option'] for x in rows]
    assert vals == sorted(vals, reverse=True)
    x = next(r for r in rows if r['pair'] == ['W', 'LS'])
    assert set(x['serve'].values()) == {'W'}
