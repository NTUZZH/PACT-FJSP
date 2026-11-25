"""Controlled timing rules for public FJSP instances (the public family).

A base instance (job_length [J], op_pt [N, M]) from a public file or from the
SD2 distribution carries no lags, job classes, setups or outages. This module
draws them from a seed with the factory benchmark as the template, so that
the eight regimes N, L, S, W, LS, LW, SW, LSW can be built on any base
instance, and writes descriptors in the project's format
(constraint_family.descriptor.ConstraintDescriptor.to_json()).

Template (factory benchmark, ppvc_instance_generator and descriptor.py)
  lags      nonzero post-operation lags lie in two bands, 12 to 24 h (two
            paint coats on every module) and 24 to 48 h (concrete curing on
            RC modules, ponding test on wet modules). Under the 'mixed' class
            mix this puts a lag on 2/22 = 9.09 % of operations in the short
            band and (0.5 + 0.75)/22 = 5.68 % in the long band. The lags never
            sit on the last operation of a job.
  setups    descriptor.draw_setup_matrix: same-class resets and cold starts
            U[0.25, 0.50] p_bar, cross-class changeovers U[0.50, 1.00] p_bar,
            symmetric, integer, at least 1. Already relative to p_bar.
  windows   descriptor.draw_windows: per machine a target outage density
            U[0.05, 0.15] of the horizon H, outages of U[2, 8] h at uniform
            starts in [0, H), no overlap.
  horizon   descriptor.job_chain_horizon with the base instance's true lags,
            for every regime.

p_bar is the mean processing time over eligible (operation, machine) pairs
of an instance. Every hour-valued factory range is divided by the factory's
p_bar, FACTORY_MEAN_PT = 4.3788 h (the mean over the 100 published 10x25
test instances, data/PPVC/10x25+ppvc-mixed), and multiplied by the p_bar of
the target instance. On a factory instance the rules therefore reproduce the
factory's ranges: window durations 2 to 8 h, lag bands 12 to 24 h and 24 to
48 h.

Transfer rules (what changes on a public instance)
  lags      each non-final operation independently gets a long-band lag with
            probability f_long * N / (N - J) and a short-band lag with
            probability f_short * N / (N - J), so that the expected share of
            all operations with a lag equals the factory's 14.77 %. The
            magnitude is drawn uniformly in the band and rounded to an
            integer, at least 1.
  classes   K = 4 job classes, assigned by a seeded permutation of
            (0, 1, 2, 3, 0, 1, ...) over the jobs, so the class sizes differ
            by at most one. A public instance has no product routes, so the
            classes carry no meaning beyond the setup matrix.
  horizon   H = max(job-chain bound with the true lags, ceil(sum of minimum
            processing times / M)). The second term is the machine-load
            bound; on every factory instance the job-chain bound is larger
            (about 200 h against 34 h on 10x25), so H equals the factory's H
            there, while on machine-bound public instances it keeps the
            outages spread over the schedule.

Member streams and pairing
  Lags, classes, setup matrix and windows of a base instance are drawn from
  four independent streams of numpy.random.SeedSequence([seed, tag]) with
  tags 7, 5, 11 and 13 (11 and 13 are the factory's setup and window tags).
  A base instance therefore carries the same lags, classes, setup matrix,
  windows and horizon in every regime that activates them, and the regime
  only switches members on or off. Regimes without LAG store an all-zero
  lag vector; the class vector is stored in every regime.

The module depends on numpy and on the descriptor and generator modules
only; it does not import params.py.
"""
import argparse
import inspect
import json
import math
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constraint_family.descriptor import (N_CLASSES, REGIMES_ALL,  # noqa: E402
                                          REGIMES_HELDOUT, REGIMES_TRAIN,
                                          ConstraintDescriptor,
                                          draw_setup_matrix, draw_windows,
                                          job_chain_horizon)
from ppvc_instance_generator import (OPS_PER_MODULE, PRESET_MIXES,  # noqa: E402
                                     ROUTES, ROUTING_CLASSES)

RULES_VERSION = 'public-rules-1'

# Repository root: stored records name their base instance relative to it.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Mean processing time over eligible pairs, averaged over the 100 instances of
# data/PPVC/10x25+ppvc-mixed (4.378766687 h). tests/test_public_adapter.py
# re-measures it from the files.
FACTORY_MEAN_PT = 4.378766687268565

# Stream tags of numpy.random.SeedSequence([seed, tag]).
TAG_LAG, TAG_CLASS, TAG_SETUP, TAG_WINDOW = 7, 5, 11, 13

# Descriptor seeds of the stored test sets: desc_seed = SET_SEED0[set] + index.
SET_SEED0 = {
    'Brandimarte': 20261101000,
    'Hurink_edata': 20261102000,
    'Hurink_rdata': 20261103000,
    'Hurink_vdata': 20261104000,
    'SD2_test_10x5+mix': 20261105000,
    'SD2_test_20x10+mix': 20261106000,
    'SD2_vali_10x5+mix': 20261107000,
    'SD2_vali_20x10+mix': 20261108000,
}


def _kw_default(fn, name):
    return inspect.signature(fn).parameters[name].default


def factory_lag_bands():
    """Nonzero lag ranges (h) of the factory routes, sorted: [(12, 24), (24, 48)]."""
    return sorted({lag for route in ROUTES.values()
                   for (*_, lag) in route if lag[1] > 0})


def factory_lag_fractions(mix='mixed'):
    """Expected share of operations whose lag lies in each band, under the
    class mix of the factory benchmark.

    :return: dict band -> fraction of all operations, e.g.
             {(12, 24): 0.0909, (24, 48): 0.0568}
    """
    probs = PRESET_MIXES[mix] if isinstance(mix, str) else mix
    out = {b: 0.0 for b in factory_lag_bands()}
    for p, cls in zip(probs, ROUTING_CLASSES):
        for (*_, lag) in ROUTES[cls]:
            if lag[1] > 0:
                out[tuple(lag)] += p / OPS_PER_MODULE
    return out


def _round_half_up(x):
    return int(math.floor(x + 0.5))


@dataclass(frozen=True)
class PublicRuleParams:
    """Parameters of the rule generator. The defaults are the factory
    template; every time-valued range is a multiple of the instance's p_bar,
    except the window density, which is a fraction of the horizon H."""
    lag_short_frac: float = 0.0       # filled in by default_params()
    lag_long_frac: float = 0.0
    lag_short_pbar: tuple = (0.0, 0.0)
    lag_long_pbar: tuple = (0.0, 0.0)
    lag_scale: float = 1.0
    n_classes: int = N_CLASSES
    setup_same: tuple = tuple(_kw_default(draw_setup_matrix, 'same'))
    setup_cross: tuple = tuple(_kw_default(draw_setup_matrix, 'cross'))
    window_density: tuple = tuple(_kw_default(draw_windows, 'density_range'))
    window_dur_pbar: tuple = (0.0, 0.0)
    horizon_rule: str = 'max_chain_load'


def default_params():
    """The factory template expressed relative to p_bar."""
    (s_lo, s_hi), (l_lo, l_hi) = factory_lag_bands()
    frac = factory_lag_fractions('mixed')
    d_lo, d_hi = _kw_default(draw_windows, 'dur_range')
    pf = FACTORY_MEAN_PT
    return PublicRuleParams(
        lag_short_frac=frac[(s_lo, s_hi)], lag_long_frac=frac[(l_lo, l_hi)],
        lag_short_pbar=(s_lo / pf, s_hi / pf),
        lag_long_pbar=(l_lo / pf, l_hi / pf),
        window_dur_pbar=(d_lo / pf, d_hi / pf))


DEFAULT_PARAMS = default_params()


def mean_eligible_pt(op_pt):
    op_pt = np.asarray(op_pt)
    return float(op_pt[op_pt > 0].mean())


def machine_load_bound(op_pt):
    """ceil(sum over operations of the minimum eligible processing time / M):
    a lower bound on the makespan of the setup-free, outage-free instance."""
    op_pt = np.asarray(op_pt, dtype=float)
    mn = np.where(op_pt > 0, op_pt, np.inf).min(axis=1)
    return int(math.ceil(mn.sum() / op_pt.shape[1]))


def public_horizon(job_length, op_pt, lag, rule='max_chain_load'):
    """Horizon H on which outages are laid out (see the module docstring)."""
    chain = job_chain_horizon(job_length, op_pt, lag)
    if rule == 'chain':
        return chain
    if rule == 'max_chain_load':
        return max(chain, machine_load_bound(op_pt))
    raise ValueError(f'unknown horizon rule {rule!r}')


def window_duration_range(mean_pt, params=DEFAULT_PARAMS):
    """Integer outage-duration range for an instance with mean pt `mean_pt`."""
    lo = max(1, _round_half_up(params.window_dur_pbar[0] * mean_pt))
    hi = max(lo, _round_half_up(params.window_dur_pbar[1] * mean_pt))
    return lo, hi


def draw_lags(job_length, mean_pt, rng, params=DEFAULT_PARAMS):
    """[N] integer post-operation lags; zero on the last operation of a job.

    Draws one uniform number per operation for the band decision and one
    magnitude per operation in each band, so the stream is consumed the same
    way whatever the outcome."""
    job_length = np.asarray(job_length, dtype=int)
    N, J = int(job_length.sum()), len(job_length)
    u = rng.random(N)
    m_short = rng.uniform(*params.lag_short_pbar, size=N)
    m_long = rng.uniform(*params.lag_long_pbar, size=N)
    last = np.cumsum(job_length) - 1
    nonfinal = np.ones(N, dtype=bool)
    nonfinal[last] = False
    if N == J:
        return np.zeros(N, dtype=int)
    scale = N / (N - J)
    p_long = min(1.0, params.lag_long_frac * scale)
    p_short = min(1.0 - p_long, params.lag_short_frac * scale)
    lag = np.zeros(N, dtype=int)
    for i in range(N):
        if not nonfinal[i]:
            continue
        if u[i] < p_long:
            mag = m_long[i]
        elif u[i] < p_long + p_short:
            mag = m_short[i]
        else:
            continue
        lag[i] = max(1, _round_half_up(mag * mean_pt * params.lag_scale))
    return lag


def draw_job_classes(n_jobs, rng, n_classes=N_CLASSES):
    """[J] balanced class labels: a seeded permutation of (0, 1, ..., K-1, 0, ...)."""
    return rng.permutation(np.arange(n_jobs) % n_classes).astype(int)


@dataclass
class PublicRules:
    """Member parameters of one base instance, shared by every regime."""
    seed: int
    mean_pt: float
    lag: np.ndarray            # [N] int, true lags (used when LAG is on)
    job_class: np.ndarray      # [J] int
    op_class: np.ndarray       # [N] int
    setup: np.ndarray          # [C+1, C] int
    windows: list              # [M] lists of (start, end)
    horizon: int
    window_dur: tuple
    params: PublicRuleParams

    def descriptor(self, regime):
        """ConstraintDescriptor of `regime` on this base instance."""
        regime = regime.upper()
        if regime not in REGIMES_ALL:
            raise ValueError(f'unknown regime {regime!r}')
        has_lag, has_setup, has_window = ('L' in regime, 'S' in regime,
                                          'W' in regime)
        lag = self.lag if has_lag else np.zeros_like(self.lag)
        return ConstraintDescriptor(
            active=(has_lag, has_setup, has_window), lag=lag,
            op_class=self.op_class,
            setup=self.setup.copy() if has_setup else None,
            windows=([list(w) for w in self.windows] if has_window else None),
            horizon=self.horizon)

    def to_json(self):
        return {'rules_version': RULES_VERSION, 'seed': int(self.seed),
                'mean_pt': self.mean_pt, 'lag': self.lag.astype(int).tolist(),
                'job_class': self.job_class.astype(int).tolist(),
                'setup': self.setup.astype(int).tolist(),
                'windows': [[[int(s), int(e)] for (s, e) in w]
                            for w in self.windows],
                'horizon': int(self.horizon),
                'window_dur': list(self.window_dur),
                'params': asdict(self.params)}


def draw_rules(job_length, op_pt, seed, params=DEFAULT_PARAMS):
    """Draw the lags, classes, setup matrix and windows of one base instance.

    Deterministic in (job_length, op_pt, seed, params)."""
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt)
    N, M = op_pt.shape
    assert int(job_length.sum()) == N, 'job_length does not match op_pt'
    if params.n_classes != N_CLASSES:
        raise ValueError(f'the descriptor holds exactly {N_CLASSES} classes; '
                         f'n_classes={params.n_classes} needs the variable-K '
                         f'change described in the report')
    mean_pt = mean_eligible_pt(op_pt)

    def rng(tag):
        return np.random.default_rng(np.random.SeedSequence([int(seed), tag]))

    lag = draw_lags(job_length, mean_pt, rng(TAG_LAG), params)
    job_class = draw_job_classes(len(job_length), rng(TAG_CLASS),
                                 params.n_classes)
    op_class = np.repeat(job_class, job_length)
    setup = draw_setup_matrix(mean_pt, rng(TAG_SETUP), same=params.setup_same,
                              cross=params.setup_cross).astype(int)
    horizon = public_horizon(job_length, op_pt, lag, params.horizon_rule)
    dur = window_duration_range(mean_pt, params)
    windows = draw_windows(M, horizon, rng(TAG_WINDOW),
                           density_range=params.window_density,
                           dur_range=dur)
    return PublicRules(seed=int(seed), mean_pt=mean_pt, lag=lag,
                       job_class=job_class, op_class=op_class, setup=setup,
                       windows=windows, horizon=int(horizon),
                       window_dur=dur, params=params)


def make_public_descriptor(regime, job_length, op_pt, seed,
                           params=DEFAULT_PARAMS):
    """Descriptor of `regime` on a base instance; the counterpart of
    descriptor.make_descriptor for instances without factory metadata."""
    return draw_rules(job_length, op_pt, seed, params).descriptor(regime)


def magnitude_violations(rules, job_length, op_pt):
    """Entries outside the ranges the rules declare (empty list = all inside).

    The counterpart of configuration check C4 for the public family: C4
    compares lags and outage durations with the factory's ranges in hours,
    which do not apply to instances on another time scale."""
    p = rules.params
    pb = rules.mean_pt
    out = []
    job_length = np.asarray(job_length, dtype=int)
    last = np.cumsum(job_length) - 1
    if np.any(rules.lag[last] != 0):
        out.append('lag on the last operation of a job')
    for v in rules.lag[rules.lag > 0]:
        ok = any(lo * pb * p.lag_scale - 0.5 <= v <= hi * pb * p.lag_scale + 0.5
                 for lo, hi in (p.lag_short_pbar, p.lag_long_pbar))
        if not ok and v != 1:
            out.append(f'lag {v} outside both bands')
    S = rules.setup
    C = p.n_classes
    for a in range(C + 1):
        for b in range(C):
            lo, hi = (p.setup_cross if (a < C and a != b) else p.setup_same)
            v = S[a, b]
            # both edges floored at 1 h, the smallest setup drawn, as in C4
            if not (max(1, lo * pb - 0.5) <= v <= max(1, hi * pb + 0.5)):
                out.append(f'setup[{a}][{b}] = {v} outside [{lo}, {hi}] p_bar')
    if not np.array_equal(S[:C], S[:C].T):
        out.append('setup block not symmetric')
    dlo, dhi = rules.window_dur
    H = rules.horizon
    for m, w in enumerate(rules.windows):
        spans = sorted(w)
        for k, (s, e) in enumerate(spans):
            if not dlo <= e - s <= dhi:
                out.append(f'machine {m}: outage length {e - s} outside [{dlo}, {dhi}]')
            if s < 0 or e > H:
                out.append(f'machine {m}: outage ({s}, {e}) outside [0, {H}]')
            if k and s < spans[k - 1][1]:
                out.append(f'machine {m}: overlapping outages')
        dens = sum(e - s for s, e in w) / H
        if dens > p.window_density[1] + (dhi + 0.5) / H:
            out.append(f'machine {m}: outage density {dens:.3f} above range')
    if rules.horizon < job_chain_horizon(job_length, op_pt, rules.lag):
        out.append('horizon below the job-chain bound')
    return out


# ----------------------------------------------------------------------
# writing a family test set
# ----------------------------------------------------------------------

def _relstem(path, root):
    stem = os.path.splitext(path)[0]
    return os.path.relpath(stem, root).replace(os.sep, '/')


def build_family(set_dir, out_dir, seed0, regimes=REGIMES_ALL,
                 params=DEFAULT_PARAMS, root=REPO_ROOT):
    """Write descriptors of every representable instance in `set_dir`.

    Layout (mirrors data/FAMILY/<size>):
      out_dir/desc/<REGIME>/<name>.desc.json   one record per regime
      out_dir/rules/<name>.rules.json          the shared member parameters
      out_dir/dataset_meta.json

    Each record is ConstraintDescriptor.to_json() plus the metadata keys
    base_instance (path stem of the .fjs file relative to `root`, the
    repository root by default),
    base_format ('fjs', no .meta.json side-car), regime, desc_seed and
    rules_version. desc_seed = seed0 + index in natural file order.
    """
    from constraint_family.public_adapter import (UnrepresentableInstance,
                                                  list_instances, read_fjs)
    paths = list_instances(set_dir)
    os.makedirs(os.path.join(out_dir, 'rules'), exist_ok=True)
    written, excluded = [], []
    for idx, path in enumerate(paths):
        try:
            inst = read_fjs(path)
        except UnrepresentableInstance as e:
            excluded.append({'instance': os.path.basename(path),
                             'reason': str(e)})
            continue
        if inst.setup is not None:
            excluded.append({'instance': inst.name,
                             'reason': 'setup-time variant; not a base instance'})
            continue
        jl, pt = inst.env_arrays()
        seed = int(seed0) + idx
        rules = draw_rules(jl, pt, seed, params)
        with open(os.path.join(out_dir, 'rules', f'{inst.name}.rules.json'),
                  'w') as f:
            rec = rules.to_json()
            rec['base_instance'] = _relstem(path, root)
            json.dump(rec, f)
        for regime in regimes:
            d = rules.descriptor(regime).to_json()
            d['base_instance'] = _relstem(path, root)
            d['base_format'] = 'fjs'
            d['regime'] = regime
            d['desc_seed'] = seed
            d['rules_version'] = RULES_VERSION
            rdir = os.path.join(out_dir, 'desc', regime)
            os.makedirs(rdir, exist_ok=True)
            with open(os.path.join(rdir, f'{inst.name}.desc.json'), 'w') as f:
                json.dump(d, f)
        written.append(inst.name)
    meta = {'base_dir': os.path.relpath(set_dir, root).replace(os.sep, '/'),
            'n_instances': len(written), 'instances': written,
            'excluded': excluded, 'regimes': list(regimes),
            'regimes_train': list(REGIMES_TRAIN),
            'regimes_heldout': list(REGIMES_HELDOUT),
            'desc_seed0': int(seed0), 'rules_version': RULES_VERSION,
            'params': asdict(params),
            'factory_mean_pt': FACTORY_MEAN_PT,
            'convention': 'attached setups; contiguous setup+proc block '
                          'avoids outages; idle-row cold-start setup; '
                          'integer time units of the base instance'}
    with open(os.path.join(out_dir, 'dataset_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    return meta


def load_family_record(desc_path, root=REPO_ROOT):
    """(job_length, op_pt, descriptor, record) of one stored public record.

    The base instance path in the record is resolved against `root`, the
    repository root by default, so the call works from any directory. The
    records are written by build_family, which draws attached setups, so the
    loader states that convention for records that do not carry the field."""
    from constraint_family.public_adapter import read_fjs
    with open(desc_path) as f:
        rec = json.load(f)
    inst = read_fjs(os.path.join(root, rec['base_instance'] + '.fjs'))
    jl, pt = inst.env_arrays()
    return jl, pt, ConstraintDescriptor.from_json(
        rec, setup_convention='attached'), rec


# ----------------------------------------------------------------------
# training draws
# ----------------------------------------------------------------------

def draw_training_instance(regime, n_j, n_m, rng, params=DEFAULT_PARAMS):
    """One SD2 base instance and its descriptor of `regime`, both from `rng`.

    The counterpart of the per-environment draw in
    train_family.FamilyTrainer.sample_batch for the public family."""
    from constraint_family.public_adapter import sd2_instance
    inst_seed = int(rng.integers(0, 2 ** 31 - 1))
    jl, pt = sd2_instance(n_j, n_m, np.random.default_rng(inst_seed))
    desc_seed = int(rng.integers(0, 2 ** 31 - 1))
    return jl, pt, make_public_descriptor(regime, jl, pt, desc_seed, params)


def _main(argv=None):
    ap = argparse.ArgumentParser(description='Write public-family descriptors '
                                 'for one set of .fjs base instances.')
    ap.add_argument('--set', required=True, help='directory of .fjs files')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--name', default=None,
                    help='key of SET_SEED0 (default: the set directory name)')
    ap.add_argument('--seed0', type=int, default=None)
    a = ap.parse_args(argv)
    name = a.name or os.path.basename(os.path.normpath(a.set))
    seed0 = a.seed0 if a.seed0 is not None else SET_SEED0[name]
    meta = build_family(a.set, a.out, seed0)
    print(f'{name}: wrote {meta["n_instances"]} instances x '
          f'{len(meta["regimes"])} regimes under {a.out}; excluded '
          f'{[e["instance"] for e in meta["excluded"]]}')


if __name__ == '__main__':
    _main()
