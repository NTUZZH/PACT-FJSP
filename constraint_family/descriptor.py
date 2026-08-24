"""Constraint descriptors for the FJSP temporal-constraint family (this paper).

A descriptor c activates a subset of the monotone-delay family:
  LAG    — job-side, machine-free post-op time-lags (FJSP-TL);
  SETUP  — machine-side sequence-dependent setups keyed by the ROUTING CLASS
           of consecutive modules on a machine (product-family changeover);
  WINDOW — machine-side availability calendars (planned outages; no
           preemption: an op must fit entirely between outages).

Conventions
  * all durations / times are integer HOURS (the base environment's unit);
  * setup matrix S has shape [C+1, C]: rows 0..C-1 are the real routing
    classes (symmetric block), row C is the IDLE row = cold-start setup
    charged to the first op on each machine. Every entry > 0, so the
    per-op minimum-setup lower envelope (column minimum) is strictly
    positive and the family bound is strictly tighter than the lag-only
    bound;
  * windows[m] is a sorted list of non-overlapping (start, end) outages.

Regime names: 'N' (none), 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW'.
Train split (Appendix B): {N, L, S, W, LS, LW}; held out: {SW, LSW}.
"""
import numpy as np

REGIMES_ALL = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
REGIMES_TRAIN = ('N', 'L', 'S', 'W', 'LS', 'LW')
REGIMES_HELDOUT = ('SW', 'LSW')

N_CLASSES = 4          # PPVC routing classes (RC-wet, RC-dry, Steel-wet, Steel-dry)
IDLE_CLASS = N_CLASSES  # index of the setup matrix's cold-start row

TOKEN_DIM = 12


class ConstraintDescriptor:
    """Per-instance constraint descriptor (true-hour units).

    Attributes
      active   : (has_lag, has_setup, has_window) bools
      lag      : [N] int hours (all-zero when LAG inactive)
      op_class : [N] int 0..C-1 routing class of each op's job
      setup    : [C+1, C] int hours (None when SETUP inactive)
      windows  : list of M sorted non-overlapping (start, end) int pairs
                 (None when WINDOW inactive)
      horizon  : int hours — the deterministic horizon estimate H used to
                 lay out outage calendars (job-chain lower bound)
    """

    def __init__(self, active, lag, op_class, setup=None, windows=None,
                 horizon=None):
        self.active = tuple(bool(b) for b in active)
        self.lag = np.asarray(lag, dtype=float)
        self.op_class = np.asarray(op_class, dtype=int)
        self.setup = None if setup is None else np.asarray(setup, dtype=float)
        self.windows = windows
        self.horizon = horizon

    @property
    def has_lag(self):
        return self.active[0]

    @property
    def has_setup(self):
        return self.active[1]

    @property
    def has_window(self):
        return self.active[2]

    def to_json(self):
        return {
            'active': list(self.active),
            'lag': self.lag.astype(int).tolist(),
            'op_class': self.op_class.tolist(),
            'setup': None if self.setup is None
                     else self.setup.astype(int).tolist(),
            'windows': None if self.windows is None
                       else [[[int(s), int(e)] for (s, e) in w]
                             for w in self.windows],
            'horizon': None if self.horizon is None else int(self.horizon),
        }

    @classmethod
    def from_json(cls, d):
        return cls(active=d['active'], lag=d['lag'], op_class=d['op_class'],
                   setup=d['setup'],
                   windows=None if d['windows'] is None
                   else [[(s, e) for (s, e) in w] for w in d['windows']],
                   horizon=d['horizon'])


def job_chain_horizon(job_length, op_pt, time_lag):
    """Deterministic horizon estimate H: the job-chain lower bound.

    max over jobs of sum(min compatible pt + post-op lag) — computable from
    instance data alone, independent of any schedule.
    """
    op_pt = np.asarray(op_pt, dtype=float)
    masked = np.where(op_pt > 0, op_pt, np.inf)
    min_pt = masked.min(axis=1)
    lag = np.zeros(op_pt.shape[0]) if time_lag is None else np.asarray(time_lag, dtype=float)
    h, first = 0.0, 0
    for L in np.asarray(job_length, dtype=int):
        seg = slice(first, first + L)
        h = max(h, float(min_pt[seg].sum() + lag[seg].sum()))
        first += L
    return int(np.ceil(h))


def draw_setup_matrix(mean_pt, rng, same=(0.25, 0.50), cross=(0.50, 1.00),
                      symmetric=True):
    """Symmetric [C+1, C] class-pair setup matrix, integer hours, min 1 h.
    Row C (idle) is the cold-start setup of a machine's first op.

    Same-class resets and cold starts draw U[25%,50%] of the instance's mean
    compatible pt; cross-class (family) changeovers draw U[50%,100%]. On
    BCA-grounded PPVC processing times (mean ~4.4 h; the long durations live
    in the LAGS) these ranges give 1-4 h setups (jig/fixture changeover
    scale) with cross-family swaps costlier than same-family resets, so
    class-batching is a real scheduling trade-off.

    The three keyword arguments exist only for the out-of-distribution arms of
    the generalization study, which need setups drawn from ranges the policy
    never trained on and, in one arm, a matrix that is not symmetric. THEIR
    DEFAULTS REPRODUCE THE PUBLISHED BENCHMARK EXACTLY, including the order in
    which the rng is drawn from -- a reordered draw would silently rewrite every
    descriptor in the paper. scripts/p29_ood.py asserts that byte-for-byte
    against the descriptors on disk before it generates anything new.
    """
    lo_same, hi_same = same[0] * mean_pt, same[1] * mean_pt
    lo_x, hi_x = cross[0] * mean_pt, cross[1] * mean_pt
    S = np.zeros((N_CLASSES + 1, N_CLASSES))
    for a in range(N_CLASSES):
        for b in range(a, N_CLASSES):
            lo, hi = (lo_same, hi_same) if a == b else (lo_x, hi_x)
            v = max(1, int(round(rng.uniform(lo, hi))))
            S[a, b] = v
            # A second, independent draw for the reverse direction when the
            # matrix is allowed to be asymmetric. This changes the number of
            # rng calls, which is exactly why it must stay off by default.
            if symmetric or a == b:
                S[b, a] = v
            else:
                S[b, a] = max(1, int(round(rng.uniform(lo, hi))))
    for b in range(N_CLASSES):
        S[IDLE_CLASS, b] = max(1, int(round(rng.uniform(lo_same, hi_same))))
    return S


def draw_windows(n_machines, horizon, rng, density_range=(0.05, 0.15),
                 dur_range=(2, 8), max_tries=200):
    """Per-machine non-overlapping outage calendars on [0, H].

    Each machine draws a target outage density U[density_range] of H and
    accumulates outages of duration U[dur_range] hours at uniform starts,
    rejecting overlaps, until the target is reached (Appendix B)."""
    windows = []
    for _ in range(n_machines):
        target = rng.uniform(*density_range) * horizon
        acc, spans, tries = 0.0, [], 0
        while acc < target and tries < max_tries:
            tries += 1
            dur = int(rng.integers(dur_range[0], dur_range[1] + 1))
            if horizon - dur <= 0:
                break
            s = int(rng.integers(0, horizon - dur))
            e = s + dur
            if any((s < e2) and (e > s2) for (s2, e2) in spans):
                continue
            spans.append((s, e))
            acc += dur
        windows.append(sorted(spans))
    return windows


def make_descriptor(regime, job_length, op_pt, meta, n_machines, seed,
                    shift=None):
    """Build the descriptor of `regime` on top of a base PPVC instance.

    Deterministic in (instance, regime, seed). The base instance's lags/ops
    are reused; SETUP and WINDOW parameters are drawn from a dedicated rng so
    the SAME base instance carries every regime (paired comparisons).

    `shift` is for the out-of-distribution generalization study only. It is a
    dict of keyword arguments forwarded to draw_setup_matrix and draw_windows,
    and `shift=None` reproduces the published benchmark exactly.
    """
    shift = shift or {}
    # A shift key that nothing consumes is silently dropped by the filters below, and
    # the arm it was supposed to move comes out identical to the published benchmark
    # while still being named, evaluated and written up as out-of-distribution. That
    # failure is invisible in every artefact it produces. Refuse instead.
    _KNOWN = {'same', 'cross', 'symmetric',        # -> draw_setup_matrix
              'density_range', 'dur_range'}        # -> draw_windows
    unknown = set(shift) - _KNOWN
    if unknown:
        raise ValueError(
            f'make_descriptor: unknown shift key(s) {sorted(unknown)}. Known keys are '
            f'{sorted(_KNOWN)}. A misspelled key would be discarded without comment '
            f'and produce an in-distribution arm labelled out-of-distribution.')
    regime = regime.upper()
    assert regime in REGIMES_ALL, f'unknown regime {regime}'
    has_lag = 'L' in regime
    has_setup = 'S' in regime
    has_window = 'W' in regime

    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt)
    time_lag = np.asarray(meta['time_lag'], dtype=int)
    op_class = np.repeat(np.asarray(meta['routing_class'], dtype=int),
                         job_length)

    lag = time_lag if has_lag else np.zeros_like(time_lag)
    # H is computed from the base instance WITH its true lags for EVERY
    # regime, so the outage calendar of a given (instance, seed) is byte-
    # identical across all regimes containing W. Likewise the setup matrix is
    # drawn from a member-specific stream independent of the regime string.
    # This makes cross-regime comparisons paired (same physics, members
    # toggled) and gives exact family monotonicity opt(c) <= opt(c ∪ c').
    horizon = job_chain_horizon(job_length, op_pt, time_lag)

    setup = None
    if has_setup:
        rng_s = np.random.default_rng(np.random.SeedSequence([seed, 11]))
        mean_pt = float(op_pt[op_pt > 0].mean())
        setup = draw_setup_matrix(
            mean_pt, rng_s,
            **{k: v for k, v in shift.items()
               if k in ('same', 'cross', 'symmetric')})
    windows = None
    if has_window:
        rng_w = np.random.default_rng(np.random.SeedSequence([seed, 13]))
        windows = draw_windows(
            n_machines, horizon, rng_w,
            **{k: v for k, v in shift.items()
               if k in ('density_range', 'dur_range')})

    return ConstraintDescriptor(active=(has_lag, has_setup, has_window),
                                lag=lag, op_class=op_class, setup=setup,
                                windows=windows, horizon=horizon)


def compute_token(desc, mean_pt, n_machines):
    """Global constraint token g in R^12 (the paper's constraint descriptor).

    g = [b_L, b_S, b_W,
         mean_lag/mean_pt, max_lag/mean_pt, lag_density,
         mean_setup/mean_pt, max_setup/mean_pt, setup_cv,
         outage_density, mean_outage_dur/mean_pt, outage_count/(10*M)]
    All summary stats are 0 when the corresponding member is inactive.
    """
    g = np.zeros(TOKEN_DIM, dtype=float)
    g[0], g[1], g[2] = desc.has_lag, desc.has_setup, desc.has_window
    mp = max(float(mean_pt), 1e-8)
    if desc.has_lag and desc.lag.size:
        g[3] = desc.lag.mean() / mp
        g[4] = desc.lag.max() / mp
        g[5] = float((desc.lag > 0).mean())
    if desc.has_setup and desc.setup is not None:
        g[6] = desc.setup.mean() / mp
        g[7] = desc.setup.max() / mp
        g[8] = float(desc.setup.std() / max(desc.setup.mean(), 1e-8))
    if desc.has_window and desc.windows is not None:
        H = max(float(desc.horizon or 1), 1e-8)
        durs = [e - s for w in desc.windows for (s, e) in w]
        if durs:
            g[9] = float(sum(durs)) / (n_machines * H)
            g[10] = float(np.mean(durs)) / mp
            g[11] = len(durs) / (10.0 * n_machines)
    return g
