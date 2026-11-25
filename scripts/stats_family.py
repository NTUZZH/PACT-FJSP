"""Shared statistics for the parity, control and held-out analyses.

All functions take relative makespan differences in percent,

    d = 100 * (a - b) / b,

computed per training seed and per base instance (and per combination when a
statistic is pooled over combinations). The caller fixes the sign. The verdict
functions read POSITIVE as "the policy under test is worse", the convention of
p11_equivalence.py and p27_seed_inference.py (policy minus comparator, over the
comparator).

Estimators
----------
hier_boot(d)       crossed hierarchical bootstrap over training seeds and base
                   instances. d has shape [k, n] (seeds x base instances) or
                   [k, n, c] (seeds x base instances x combinations). Each replicate
                   draws k seed indices with replacement, then n base-instance
                   indices with replacement, and averages d over the drawn seeds,
                   the drawn instances and all c combinations of each drawn
                   instance. The one instance draw is applied to every drawn seed,
                   because every seed was evaluated on every instance (crossed
                   design), and a drawn base instance brings all its combinations,
                   because the combinations of one base instance share its jobs
                   (cluster by base instance). Random stream: numpy
                   default_rng(seed); per replicate first integers(0, k, k), then
                   integers(0, n, n). This is the stream of p27_seed_inference.py,
                   so the three-seed seed-level table is reproduced exactly.
instance_boot(d)   cluster bootstrap over base instances only, for a FIXED set of
                   policies (the instance-level question). d has shape [n] or
                   [n, c]; a drawn instance brings its c combinations. Stream:
                   default_rng(seed).integers(0, n, (B, n)), the stream of
                   p84_cluster_bootstrap.py.
seed_t(d)          the training run as the unit: Student t interval over the k
                   per-seed means (k - 1 degrees of freedom); with d of shape
                   [k, n, c] the per-seed mean is over instances and combinations.

All intervals are two-sided percentile (bootstrap) or t intervals at level
1 - alpha, alpha = 0.05 by default. `simultaneous_alpha(alpha, m)` gives the
Bonferroni level for m simultaneous intervals.
"""
import math

import numpy as np

B_DEFAULT = 20000


def _q(alpha):
    """Percentile levels of a two-sided interval; the literal [2.5, 97.5] at
    alpha = 0.05, the call of p11, p27 and p84, so their numbers reproduce."""
    if alpha == 0.05:
        return [2.5, 97.5]
    return [100 * alpha / 2, 100 * (1 - alpha / 2)]


def _as3(d):
    d = np.asarray(d, dtype=float)
    if d.ndim == 2:
        return d[:, :, None]
    if d.ndim == 3:
        return d
    raise ValueError(f'want [seeds, instances] or [seeds, instances, combinations], got {d.shape}')


def hier_boot(d, B=B_DEFAULT, seed=11, alpha=0.05, return_draws=False):
    """Crossed hierarchical bootstrap (seeds x base instances[, x combinations]).

    Returns dict(mean, lo, hi, B, seed, alpha, k, n, c, scheme)."""
    x = _as3(d)
    k, n, c = x.shape
    rng = np.random.default_rng(seed)
    per = x.mean(axis=2)                      # [k, n]: mean over the combinations
    draws = np.empty(B)
    for b in range(B):
        si = rng.integers(0, k, k)
        ii = rng.integers(0, n, n)
        draws[b] = per[si][:, ii].mean()
    lo, hi = np.percentile(draws, _q(alpha))
    out = dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), B=int(B),
               seed=int(seed), alpha=float(alpha), k=k, n=n, c=c,
               scheme='crossed seeds x base instances, combinations kept together')
    if return_draws:
        out['draws'] = draws
    return out


def hier_draws_stream(d, B, rng):
    """The B replicate means of the crossed hierarchical bootstrap, drawn from an
    existing Generator in the order of hier_boot (per replicate the k seed indices,
    then the n instance indices)."""
    x = _as3(d)
    k, n, _ = x.shape
    per = x.mean(axis=2)
    draws = np.empty(B)
    for b in range(B):
        si = rng.integers(0, k, k)
        ii = rng.integers(0, n, n)
        draws[b] = per[si][:, ii].mean()
    return draws


def hier_boot_stream(d, B, rng, alpha=0.05):
    """hier_boot on an existing Generator (for scripts that share one stream over
    several quantities, as p27 does over the regimes). Returns (mean, lo, hi)."""
    x = _as3(d)
    draws = hier_draws_stream(x, B, rng)
    lo, hi = np.percentile(draws, _q(alpha))
    return float(x.mean()), float(lo), float(hi)


def percentile_ci(draws, alpha=0.05):
    """Two-sided percentile interval of replicate draws at level 1 - alpha."""
    lo, hi = np.percentile(np.asarray(draws), _q(alpha))
    return float(lo), float(hi)


def instance_boot(d, B=B_DEFAULT, seed=7, alpha=0.05, return_draws=False):
    """Cluster bootstrap over base instances for a fixed set of policies."""
    x = np.asarray(d, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f'want [instances] or [instances, combinations], got {x.shape}')
    n = x.shape[0]
    per = x.mean(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (B, n))
    draws = per[idx].mean(axis=1)
    lo, hi = np.percentile(draws, _q(alpha))
    out = dict(mean=float(x.mean()), lo=float(lo), hi=float(hi), B=int(B),
               seed=int(seed), alpha=float(alpha), n=n, c=x.shape[1],
               scheme='base instances, combinations kept together')
    if return_draws:
        out['draws'] = draws
    return out


def seed_t(d, alpha=0.05):
    """Student t interval over the k per-seed means (training run as the unit)."""
    from scipy.stats import t as _t
    x = _as3(d)
    k = x.shape[0]
    means = x.reshape(k, -1).mean(axis=1)
    m = float(means.mean())
    if k < 2:
        return dict(mean=m, lo=float('nan'), hi=float('nan'), k=k, sd=float('nan'),
                    seed_means=means.tolist(), alpha=alpha)
    sd = float(means.std(ddof=1))
    h = float(_t.ppf(1 - alpha / 2, k - 1)) * sd / math.sqrt(k)
    return dict(mean=m, lo=m - h, hi=m + h, k=k, sd=sd, seed_means=means.tolist(),
                alpha=float(alpha), df=k - 1)


def seed_t_p(d, alternative='less'):
    """One-sample t test over the k per-seed means of d against 0 (training run as
    the unit, k - 1 degrees of freedom). alternative='less' tests H1: mean < 0,
    i.e. the policy under test is FASTER than the comparator (d = 100 (a - b) / b).
    Returns nan when k < 2 or the per-seed means do not vary."""
    from scipy.stats import ttest_1samp
    x = _as3(d)
    k = x.shape[0]
    means = x.reshape(k, -1).mean(axis=1)
    if k < 2 or np.all(means == means[0]):
        return float('nan')
    return float(ttest_1samp(means, 0.0, alternative=alternative).pvalue)


def boot_p_one_sided(draws, alternative='less'):
    """One-sided bootstrap p from replicate draws, add-one: for 'less' (H1: mean
    < 0) p = (1 + #{draw >= 0}) / (B + 1)."""
    draws = np.asarray(draws)
    B = len(draws)
    bad = int((draws >= 0).sum()) if alternative == 'less' else int((draws <= 0).sum())
    return min(1.0, (1 + bad) / (B + 1))


def simultaneous_alpha(alpha, m):
    """Bonferroni: per-interval alpha for m simultaneous intervals at level 1 - alpha."""
    return alpha / m


# ----------------------------------------------------------------- verdicts
VERDICTS = ('equivalent', 'non-inferior only', 'inferior', 'inconclusive')


def verdict(lo, hi, margin):
    """Margin verdict of one two-sided interval [lo, hi] of d (positive = worse).

    THE RULE (margin delta > 0, applied to the same interval at every margin):
      equivalent         -delta < lo  and  hi < delta
                         the interval lies inside the margin band;
      non-inferior only  hi < delta  and  lo <= -delta
                         no worse than the margin, but equivalence is not shown;
      inferior           lo > 0      and  hi >= delta
                         the interval excludes zero on the wrong side and leaves
                         the band (W at +-0.5%: [+0.18, +0.89]);
      inconclusive       otherwise   (lo <= 0 < delta <= hi).
    The four labels are exclusive and exhaustive. Whether the interval excludes
    zero is reported separately by direction(): an interval inside the band that
    also lies above zero (W at +-1%) is 'equivalent' with direction 'behind'.
    A 95% interval inside the band is two one-sided tests at level 0.025 each."""
    if margin <= 0:
        raise ValueError('margin must be positive')
    if -margin < lo and hi < margin:
        return 'equivalent'
    if hi < margin:
        return 'non-inferior only'
    if lo > 0:
        return 'inferior'
    return 'inconclusive'


def direction(lo, hi):
    """'ahead' if the interval lies below zero (policy faster), 'behind' if above,
    'neither' if it contains zero."""
    if hi < 0:
        return 'ahead'
    if lo > 0:
        return 'behind'
    return 'neither'


# Macro values of the labels: plain words. macro_io.emit rejects a bare 'inf'
# token, so the small-capital codes of p27 are not used; the manuscript styles the words.
TEX = {'equivalent': 'equivalent', 'non-inferior only': 'non-inferior',
       'inferior': 'inferior', 'inconclusive': 'inconclusive',
       'ahead': 'ahead', 'behind': 'behind', 'neither': 'neither'}


# ----------------------------------------------------------------- families
def iut(component_verdicts, claim='equivalent'):
    """Intersection-union test for a joint claim over a family of components.

    The joint claim ("equivalent on all eight combinations") is the intersection
    of the component claims, and its null hypothesis is the union of the
    component nulls. Rejecting the union null at level alpha requires rejecting
    every component null at level alpha (Berger 1982), so no multiplicity
    correction is needed and none may be granted: the joint claim holds iff every
    component verdict is the claimed one. For claim='non-inferior', a component
    counts when it is 'equivalent' or 'non-inferior only'.
    Returns (holds, failing_components)."""
    ok = {'equivalent': ('equivalent',),
          'non-inferior': ('equivalent', 'non-inferior only')}[claim]
    failing = [k for k, v in component_verdicts.items() if v not in ok]
    return (not failing), failing


def holm(ps):
    """Holm-Bonferroni adjusted p-values, order preserved (as p11 and p17).
    For a family of superiority claims (discoveries), e.g. the policy ahead of
    the best dispatching rule on each combination."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def boot_p(draws):
    """Two-sided bootstrap p for 'the mean differs from zero', from the replicate
    draws, add-one (Davison-Hinkley), as p11."""
    draws = np.asarray(draws)
    B = len(draws)
    p = 2.0 * (1 + min(int((draws <= 0).sum()), int((draws >= 0).sum()))) / (B + 1)
    return min(1.0, p)


def rel_gap(a, b):
    """100 * (a - b) / b, elementwise."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 100.0 * (a - b) / b
