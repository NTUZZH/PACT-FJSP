"""Tests for scripts/stats_family.py on synthetic data with known answers.

Run from the repository root:  python3 -m pytest tests/test_stats_family.py -q
CPU only; no data files are read.
"""
import itertools
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import stats_family as SF  # noqa: E402

B = 4000


# ------------------------------------------------------------ reference copies
def p27_seed_level(mat, B, rng):
    """scripts/p27_seed_inference.seed_level, verbatim logic, on a shared stream."""
    k, n = mat.shape
    draws = np.empty(B)
    for b in range(B):
        si = rng.integers(0, k, k)
        ii = rng.integers(0, n, n)
        draws[b] = mat[si][:, ii].mean()
    return float(mat.mean()), *np.percentile(draws, [2.5, 97.5])


def p84_cluster_ci(cells, B):
    """scripts/p84_cluster_bootstrap.cluster_ci, verbatim logic."""
    per_base = cells.mean(axis=1)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(per_base), (B, len(per_base)))
    lo, hi = np.percentile(per_base[idx].mean(axis=1), [2.5, 97.5])
    return float(cells.mean()), float(lo), float(hi)


def synthetic(k=5, n=40, c=None, seed=0):
    g = np.random.default_rng(seed)
    shape = (k, n) if c is None else (k, n, c)
    return g.normal(0.3, 1.0, size=shape) + g.normal(0, 0.2, size=(k,) + (1,) * (len(shape) - 1))


# ------------------------------------------------------------ hierarchical bootstrap
def test_hier_boot_reproduces_p27_stream_exactly():
    mats = [synthetic(3, 100, seed=s) for s in range(3)]
    ref_rng, new_rng = np.random.default_rng(11), np.random.default_rng(11)
    for m in mats:                                   # one shared stream, regimes in order
        ref = p27_seed_level(m, B, ref_rng)
        new = SF.hier_boot_stream(m, B, new_rng)
        assert new == tuple(float(x) for x in ref)


def test_hier_boot_single_call_equals_stream_from_fresh_rng():
    m = synthetic(5, 30)
    a = SF.hier_boot(m, B=B, seed=11)
    b = SF.hier_boot_stream(m, B, np.random.default_rng(11))
    assert (a['mean'], a['lo'], a['hi']) == b
    assert a['B'] == B and a['seed'] == 11 and a['k'] == 5 and a['n'] == 30


def test_zero_seed_variance_is_the_instance_bootstrap_on_the_same_stream():
    g = np.random.default_rng(3)
    row = g.normal(0.2, 1.0, 50)
    mat = np.tile(row, (4, 1))                       # every seed identical
    out = SF.hier_boot(mat, B=B, seed=5, return_draws=True)
    rng = np.random.default_rng(5)
    want = np.empty(B)
    for b in range(B):
        rng.integers(0, 4, 4)                        # the seed draw, without effect
        want[b] = row[rng.integers(0, 50, 50)].mean()
    # same stream; only the summation order of the mean differs
    assert np.allclose(out['draws'], want, rtol=0, atol=1e-12)


def test_zero_instance_variance_is_the_seed_bootstrap():
    consts = np.array([-0.4, 0.1, 0.5, 0.9, 1.3])
    mat = np.repeat(consts[:, None], 60, axis=1)     # each seed constant over instances
    out = SF.hier_boot(mat, B=B, seed=9, return_draws=True)
    rng = np.random.default_rng(9)
    want = np.empty(B)
    for b in range(B):
        si = rng.integers(0, 5, 5)
        rng.integers(0, 60, 60)
        want[b] = consts[si].mean()
    assert np.allclose(out['draws'], want, rtol=0, atol=1e-12)
    # the interval is the percentile interval of seed-resample means of the constants
    assert out['lo'] >= consts.min() and out['hi'] <= consts.max()


def test_combination_copies_equal_the_two_dimensional_case():
    m = synthetic(5, 30)
    for c in (2, 8):                                 # powers of two: the means are exact
        m3 = np.repeat(m[:, :, None], c, axis=2)
        a, b = SF.hier_boot(m, B=B, seed=11), SF.hier_boot(m3, B=B, seed=11)
        assert (a['lo'], a['hi']) == (b['lo'], b['hi'])
        assert b['c'] == c


def test_cluster_keeps_the_combinations_of_a_base_instance_together():
    # two combinations perfectly anti-correlated within a base instance: the
    # cluster interval of their mean is degenerate, a cell bootstrap would not be
    g = np.random.default_rng(4)
    x = g.normal(0, 1, (3, 40))
    m3 = np.stack([x, -x], axis=2)
    out = SF.hier_boot(m3, B=B, seed=1)
    assert abs(out['lo']) < 1e-12 and abs(out['hi']) < 1e-12


def test_instance_boot_equals_p84_cluster_ci():
    g = np.random.default_rng(8)
    cells = g.normal(0.5, 1.0, (10, 8))
    out = SF.instance_boot(cells, B=B, seed=7)
    assert (out['mean'], out['lo'], out['hi']) == p84_cluster_ci(cells, B)


# ------------------------------------------------------------ seed-level t
def test_seed_t_known_values():
    mat = np.repeat(np.array([1.0, 2.0, 3.0])[:, None], 10, axis=1)
    out = SF.seed_t(mat)
    half = 4.302652729749462 * 1.0 / math.sqrt(3)    # t(0.975, 2) * sd / sqrt(k)
    assert out['mean'] == pytest.approx(2.0)
    assert out['sd'] == pytest.approx(1.0)
    assert out['lo'] == pytest.approx(2.0 - half, abs=1e-9)
    assert out['hi'] == pytest.approx(2.0 + half, abs=1e-9)
    assert out['df'] == 2


def test_seed_t_p_one_sided():
    mat = np.repeat(np.array([-1.0, -2.0, -3.0, -1.5, -2.5])[:, None], 4, axis=1)
    p = SF.seed_t_p(mat, 'less')
    # t = mean / (sd / sqrt(5)) = -2 / (0.790569 / 2.236068) = -5.6569, df 4
    assert p == pytest.approx(0.0024063392, abs=1e-9)
    assert math.isnan(SF.seed_t_p(np.zeros((3, 4))))


# ------------------------------------------------------------ verdicts
@pytest.mark.parametrize('shift, v05, v10, direction', [
    (0.3, 'equivalent', 'equivalent', 'behind'),
    (0.7, 'inferior', 'equivalent', 'behind'),
    (-0.7, 'non-inferior only', 'equivalent', 'ahead'),
    (1.5, 'inferior', 'inferior', 'behind'),
])
def test_constant_shift_verdicts(shift, v05, v10, direction):
    mat = np.full((5, 30), shift)
    out = SF.hier_boot(mat, B=500, seed=11)
    assert out['lo'] == pytest.approx(shift) and out['hi'] == pytest.approx(shift)
    assert SF.verdict(out['lo'], out['hi'], 0.5) == v05
    assert SF.verdict(out['lo'], out['hi'], 1.0) == v10
    assert SF.direction(out['lo'], out['hi']) == direction


def test_w_interval_of_the_reviews():
    lo, hi = 0.18, 0.89
    assert SF.verdict(lo, hi, 0.5) == 'inferior'
    assert SF.verdict(lo, hi, 1.0) == 'equivalent'
    assert SF.direction(lo, hi) == 'behind'


def test_inconclusive_and_boundaries():
    assert SF.verdict(-0.2, 0.6, 0.5) == 'inconclusive'      # contains 0, leaves the band
    assert SF.verdict(-0.6, 0.6, 0.5) == 'inconclusive'
    assert SF.verdict(-0.5, 0.2, 0.5) == 'non-inferior only'  # lo on the edge: not inside
    assert SF.verdict(0.1, 0.5, 0.5) == 'inferior'            # hi on the edge: not inside
    with pytest.raises(ValueError):
        SF.verdict(0.0, 0.1, 0.0)


def test_verdict_grid_exclusive_exhaustive_and_consistent():
    grid = np.round(np.arange(-2.0, 2.01, 0.1), 10)
    for margin in (0.5, 1.0):
        for lo, hi in itertools.combinations_with_replacement(grid, 2):
            v = SF.verdict(lo, hi, margin)
            assert v in SF.VERDICTS
            inside = -margin < lo and hi < margin
            assert (v == 'equivalent') == inside
            if v == 'non-inferior only':
                assert hi < margin and lo <= -margin
            if v == 'inferior':
                assert lo > 0 and hi >= margin
            if v == 'inconclusive':
                assert lo <= 0 < margin <= hi


def test_widening_the_margin_never_weakens_the_verdict():
    grid = np.round(np.arange(-2.0, 2.01, 0.1), 10)
    for lo, hi in itertools.combinations_with_replacement(grid, 2):
        a, b = SF.verdict(lo, hi, 0.5), SF.verdict(lo, hi, 1.0)
        if a == 'equivalent':
            assert b == 'equivalent'
        if a == 'non-inferior only':
            assert b in ('non-inferior only', 'equivalent')
        if b == 'inferior':
            assert a == 'inferior'


# ------------------------------------------------------------ families
def test_iut():
    allq = {r: 'equivalent' for r in ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')}
    assert SF.iut(allq) == (True, [])
    one = dict(allq, W='non-inferior only')
    assert SF.iut(one) == (False, ['W'])
    assert SF.iut(one, claim='non-inferior') == (True, [])
    bad = dict(allq, W='inconclusive', SW='inferior')
    assert SF.iut(bad, claim='non-inferior') == (False, ['W', 'SW'])


def test_holm_example_and_order():
    assert SF.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert SF.holm([0.5, 0.001]) == pytest.approx([0.5, 0.002])
    assert SF.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_bootstrap_p_add_one():
    assert SF.boot_p(np.array([1.0, 2.0, 3.0])) == pytest.approx(2 * 1 / 4)
    assert SF.boot_p_one_sided(np.array([-1.0, -2.0, 0.5]), 'less') == pytest.approx(2 / 4)
    assert SF.boot_p_one_sided(np.array([-1.0, -2.0, -3.0]), 'less') == pytest.approx(1 / 4)


def test_percentile_levels():
    assert SF._q(0.05) == [2.5, 97.5]
    assert SF._q(0.05 / 8) == pytest.approx([0.3125, 99.6875])
