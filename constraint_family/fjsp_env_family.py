"""Constraint-family FJSP environment.

FJSPEnvFamily extends FJSPEnvForSameOpNums through the δ-hooks only:

  _earliest_start / _compute_pair_free_time
      block(b, k) = push_windows( max(job_ready(b), mch_free(k)),
                                  dur = s(class_last(k), class(b)) + p_bk )
      processing start = block + s;  completion = block + s + p.
      LAG stays inside job_ready (the base environment's release update);
      SETUP is the ATTACHED convention: the changeover starts only when
      both the job and the machine are available and occupies the machine
      contiguously with processing. (Detached setups make the
      minimum-inbound-setup lower envelope inadmissible, so the family
      fixes the attached semantics.)
      WINDOW pushes the block so [block, block+s+p] avoids every outage
      (no preemption, machine idle during outages).

  _bound_min_term
      per-op min term of the admissible bound gains the setup lower
      envelope δ̲_setup(op) = min_source S[·, class(op)] (> 0 by
      construction, incl. the idle row), keeping the telescoped reward
      objective-faithful for every regime (Uniform Admissibility).

Normalized/true consistency: PPVC instances always contain incompatible
pairs, so pt_lower_bound == 0 and the pt normalization is a PURE SLOPE
scaling; setups and outage calendars scale by the same slope and both time
axes stay order-isomorphic (asserted below).

Legacy mode (`family_mode=False`) keeps the parent's exact feature set —
the E0 bit-exact regression anchor.

RELEASE (candidate fourth rule, evaluation only): a descriptor may carry
release dates rho_i (integer hours per job). A release date is a job-side
delay: job i is not present before rho_i. It is applied by initialising the
job-ready time of job i to rho_i on both time axes, so the block of its first
operation (attached setup, then processing) starts no earlier, as every block
starts at max(job_ready, mch_free). The makespan estimate op_ct_lb is
unchanged: it starts every job chain at time zero, so it ignores the release
dates and stays a lower bound. The observation has no new channel, and the
mask is the parent's non-delay rule. The parent
builds its first-decision mask with every job ready at 0 and machine-side
delays ignored; with release dates the first decision instant is
min_i rho_i and only jobs released at that instant are admissible, which is
the same rule restricted to readiness. The branch runs only when some
descriptor carries a release field; with release=None every code path is the
unchanged published one (tests/test_release.py asserts it).

PACT_FIRST_MASK=1 (evaluation only, environment variable, off by default): the
parent builds the first-decision table with every pair start at 0, so at the
first decision every compatible pair is legal even when its processing start
(after the attached setup from the idle class and any push past an outage)
is later than the earliest one. With the flag, the step-0 pair-start table,
decision instant and mask are computed by the same code step() runs after
every dispatch (_first_mask_tables). Unset, every code path is the published
one (tests/test_first_mask.py asserts it against the frozen copy).

Mask options. The two mask corrections are also constructor arguments,
first_mask and mask_ties, each 'legacy' (the published behaviour) or 'exact':
  first_mask='exact'  the PACT_FIRST_MASK=1 behaviour above;
  mask_ties='exact'   the PACT_MASK_TOL=1 behaviour (_compute_unavailable_pairs):
                      a pair whose processing start equals the decision instant
                      in hours stays legal when float error on the normalized
                      axis puts it a few ulps above the minimum.
An explicit argument is authoritative. None (the default) defers to the
environment variable, read at the same point as before, so every existing
command behaves as it did. resolve_mask_option is the one place a script turns
an option and the variable into a setting, and it refuses a contradiction.

PER-MACHINE SETUP MULTIPLIER (sensitivity study, evaluation only): a
descriptor may carry setup_mult, one u_k in {0, 1} per machine. Machine k then
charges u_k * S[last class, next class]. The multiplier enters the three places
that read a setup: the machine-side delay (_setup_into_candidates and
_earliest_start, so the pair-start table, the mask and the transition), the
machine feature channel that reports the cheapest setup into the next operation
(it reads _setup_into_candidates), and the makespan estimate, whose per-op term
becomes min over eligible machines k of [p_ijk + u_k * min_src S[src, class]]
(_bound_min_term). That term is still a lower bound: machine k charges at
least u_k times the cheapest setup into the class, whatever its last class.
The branches run only when some descriptor carries the field; without it every
code path is the unchanged one (tests/test_setup_mult.py asserts it against the
frozen copy tests/fixtures/fjsp_env_family_pre_setupmult.py).
"""
import copy
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from constraint_family.descriptor import (ConstraintDescriptor, N_CLASSES,
                                          IDLE_CLASS, TOKEN_DIM, compute_token)
import torch

DELTA_MODES = ('min', 'zero', 'max')


def resolve_delta_mode(src):
    """The one place the shaping envelope is decided. `src` is a config or an env.

    Two flags can name the envelope: `delta_mode` (min|zero|max) and the older
    boolean `reward_delta_env`. `delta_mode` defaults to None, not to a mode,
    so that "unset" is representable and the boolean keeps its meaning on
    configs and checkpoints that carry only it; a genuine contradiction
    (reward_delta_env=False with delta_mode other than 'zero') is a hard
    error, never a guess.
    """
    mode = getattr(src, 'delta_mode', None)
    keep = bool(getattr(src, 'reward_delta_env', True))
    if mode is None:
        return 'min' if keep else 'zero'
    if mode not in DELTA_MODES:
        raise ValueError(f'delta_mode must be one of {DELTA_MODES}, got {mode!r}')
    if not keep and mode != 'zero':
        raise SystemExit(
            f'contradictory reward flags: reward_delta_env=False asks for no '
            f'envelope, delta_mode={mode!r} asks for one. Refusing to guess -- '
            f'say --delta_mode zero.')
    return mode


MASK_OPTIONS = ('legacy', 'exact')
MASK_ENV_VARS = {'first_mask': 'PACT_FIRST_MASK', 'mask_ties': 'PACT_MASK_TOL'}


def mask_option_from_env(name):
    """The setting the environment variable of mask correction `name` asks for:
    'exact' for 1, 'legacy' for 0, None when unset or empty. Any other value
    is refused, so a typo cannot silently select the legacy mask."""
    var = MASK_ENV_VARS[name]
    raw = os.environ.get(var, '')
    if raw == '':
        return None
    if raw == '1':
        return 'exact'
    if raw == '0':
        return 'legacy'
    raise ValueError(f'{var}={raw!r}: use 1 (exact), or 0 or unset (legacy)')


def resolve_mask_option(name, value=None):
    """The one place a script decides a mask correction ('legacy' or 'exact').

    `name` is 'first_mask' or 'mask_ties'; `value` is the explicit option, None
    when it was not given. Without an option the environment variable decides
    (legacy when unset), so the existing evaluation commands keep working. An
    option and a set variable that disagree are a hard error, never a guess.
    """
    env = mask_option_from_env(name)
    if value is None:
        return env or 'legacy'
    if value not in MASK_OPTIONS:
        raise ValueError(f'{name} must be one of {MASK_OPTIONS}, got {value!r}')
    if env is not None and env != value:
        raise SystemExit(
            f'contradictory mask settings: --{name} {value}, but '
            f'{MASK_ENV_VARS[name]}={os.environ[MASK_ENV_VARS[name]]} asks for '
            f'{env}. Refusing to guess -- unset the variable or change the option.')
    return value


class FJSPEnvFamily(FJSPEnvForSameOpNums):

    def __init__(self, n_j, n_m, family_mode=True, use_lag_features=None,
                 blind_channels=(), mask_mode='nondelay', pair_extra='none',
                 first_mask=None, mask_ties=None):
        # family mode always carries the two lag channels (zeros when LAG is
        # inactive) -> op feature width 12; plus 2 machine channels -> 10.
        # Legacy mode (family_mode=False) keeps the parent's exact feature
        # set (use_lag_features passes through) — the E0 regression anchor.
        # blind_channels (E3 channel-ablation): subset of
        # {'lag','setup','window'} whose PER-NODE MAGNITUDE channels are
        # zeroed at observation time (dynamics untouched) — the counterpart
        # of the token interventions, isolating what the magnitude channels
        # carry vs what the global token carries.
        # mask_mode (option of the environment): 'nondelay' = the published
        # mask; 'active' = the Giffler–Thompson relaxation, see
        # _compute_unavailable_pairs.
        # pair_extra (optional pair channels): 'none' = the published 8 pair channels;
        # 'sw' = append the exact setup this assignment would incur now and
        # the window-induced push delay, see construct_pair_features.
        # first_mask / mask_ties: the two mask corrections (module docstring);
        # None defers to PACT_FIRST_MASK / PACT_MASK_TOL.
        use_lag = True if family_mode else bool(use_lag_features)
        super().__init__(n_j, n_m, use_lag_features=use_lag)
        self.family_mode = family_mode
        self.blind_channels = frozenset(blind_channels)
        assert self.blind_channels <= {'lag', 'setup', 'window'}
        assert mask_mode in ('nondelay', 'active'), mask_mode
        assert pair_extra in ('none', 'sw'), pair_extra
        assert first_mask is None or first_mask in MASK_OPTIONS, first_mask
        assert mask_ties is None or mask_ties in MASK_OPTIONS, mask_ties
        self.mask_mode = mask_mode
        self.pair_extra = pair_extra
        self._first_mask_opt = first_mask
        self._mask_ties_opt = mask_ties
        if family_mode:
            self.mch_fea_dim = 10

    # ------------------------------------------------------------------
    # data loading
    # ------------------------------------------------------------------

    def set_initial_data(self, job_length_list, op_pt_list, descriptor_list=None,
                         op_type_list=None, mch_type_list=None, **kw):
        """descriptor_list: one ConstraintDescriptor per env instance.
        None -> all-inactive descriptors (base FJSP)."""
        E = len(job_length_list)
        M = op_pt_list[0].shape[1]
        N = op_pt_list[0].shape[0]

        if descriptor_list is None:
            descriptor_list = [
                ConstraintDescriptor(active=(False, False, False),
                                     lag=np.zeros(N), op_class=np.zeros(N),
                                     horizon=1)
                for _ in range(E)]
        assert len(descriptor_list) == E
        self.descriptors = descriptor_list

        # ---- static constraint state (needed by hooks during super().init) --
        self.c_bits = np.array([[d.has_lag, d.has_setup, d.has_window]
                                for d in descriptor_list], dtype=float)  # [E,3]
        self.op_class = np.stack([d.op_class for d in descriptor_list])  # [E,N]

        # normalization slope must match the parent's pt scaling; asserted
        # after super().set_initial_data below.
        raw_pt = np.array(op_pt_list)
        pt_lb, pt_ub = raw_pt.min(), raw_pt.max()
        slope = 1.0 / (pt_ub - pt_lb + 1e-8)
        self._slope = slope

        # setups: [E, C+1, C]; zero matrix when inactive
        self.true_setup = np.zeros((E, N_CLASSES + 1, N_CLASSES))
        for e, d in enumerate(descriptor_list):
            if d.has_setup and d.setup is not None:
                self.true_setup[e] = d.setup
        self.setup_norm = self.true_setup * slope

        # per-machine setup multiplier (sensitivity study). Guarded: without
        # any setup_mult field nothing below, in the hooks or in the bound
        # changes.
        self._has_setup_mult = any(getattr(d, 'setup_mult', None) is not None
                                   for d in descriptor_list)
        if self._has_setup_mult:
            self.setup_mult = np.ones((E, M))
            for e, d in enumerate(descriptor_list):
                u = getattr(d, 'setup_mult', None)
                if u is not None:
                    u = np.asarray(u, dtype=float)
                    assert u.shape == (M,), \
                        f'setup_mult must have length M={M}, got {u.shape}'
                    assert np.all((u == 0) | (u == 1)), \
                        'setup_mult entries must be 0 or 1'
                    self.setup_mult[e] = u

        # windows: padded [E, M, K]; +inf padding never matches an overlap
        K = max((max((len(w) for w in d.windows), default=0)
                 if d.has_window and d.windows is not None else 0)
                for d in descriptor_list)
        self.n_win = max(K, 0)
        self.true_win_s = np.full((E, M, max(K, 1)), np.inf)
        self.true_win_e = np.full((E, M, max(K, 1)), np.inf)
        for e, d in enumerate(descriptor_list):
            if d.has_window and d.windows is not None:
                for m, spans in enumerate(d.windows):
                    for k, (s, en) in enumerate(spans):
                        self.true_win_s[e, m, k] = s
                        self.true_win_e[e, m, k] = en
        self.win_s_norm = self.true_win_s * slope
        self.win_e_norm = self.true_win_e * slope

        # horizon estimates (for the time-to-next-outage sentinel)
        self.true_horizon = np.array([max(float(d.horizon or 1), 1.0)
                                      for d in descriptor_list])  # [E]
        self.horizon_norm = self.true_horizon * slope

        # release dates (candidate fourth rule). Guarded: without any release
        # field nothing below or in initial_vars changes.
        J = job_length_list[0].shape[0]
        self._has_release = any(getattr(d, 'release', None) is not None
                                for d in descriptor_list)
        if self._has_release:
            self.true_release = np.zeros((E, J))
            for e, d in enumerate(descriptor_list):
                if getattr(d, 'release', None) is not None:
                    r = np.asarray(d.release, dtype=float)
                    assert r.shape == (J,), f'release must have length J={J}'
                    self.true_release[e] = r
            self.release_norm = self.true_release * slope

        time_lag_list = [d.lag for d in descriptor_list]

        state = super().set_initial_data(job_length_list, op_pt_list,
                                         time_lag_list=time_lag_list,
                                         op_type_list=op_type_list,
                                         mch_type_list=mch_type_list)

        # the slope consistency this whole file relies on
        assert abs(self.pt_lower_bound) < 1e-12, \
            'family env requires pt_lower_bound == 0 (pure slope normalization)'

        if self._has_release:
            self._apply_release_first_mask()

        # first_mask='exact' (or PACT_FIRST_MASK=1 when the argument is None):
        # the first decision uses the same pair-start table, decision instant
        # and mask as every later step. Legacy, nothing below changes.
        self._first_mask = self.mask_setting('first_mask') == 'exact'
        if self._first_mask:
            self._apply_defined_first_mask()

        # global constraint token g [E, 12]
        mean_pt_true = np.array([float(p[p > 0].mean()) for p in raw_pt])
        self.token_g = np.stack([
            compute_token(d, mean_pt_true[e], M)
            for e, d in enumerate(descriptor_list)])
        self._attach_token()
        self.old_state.token_tensor = self.state.token_tensor
        return self.state

    def mask_setting(self, name):
        """'legacy' or 'exact' for mask correction `name`: the constructor
        argument when given, else the environment variable at call time."""
        opt = getattr(self, f'_{name}_opt', None)
        if opt is not None:
            return opt
        return mask_option_from_env(name) or 'legacy'

    def _apply_release_first_mask(self):
        """First-decision non-delay mask under release dates: only pairs
        whose job is released at the first decision instant min_i rho_i are
        admissible (initial_vars sets pair_free_time and next_schedule_time).
        Rebuilds the step-0 state the parent built with every job ready at 0."""
        unavailable = (self.pair_free_time
                       > self.next_schedule_time[:, None, None])
        self.dynamic_pair_mask = np.logical_or(self.candidate_process_relation,
                                               unavailable)
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask,
                                                    axis=1)
        self.construct_mch_features()
        self.construct_pair_features()
        self.old_state.update(self.fea_j, self.op_mask, self.fea_m,
                              self.mch_mask, self.dynamic_pair_mask,
                              self.comp_idx, self.candidate, self.fea_pairs,
                              op_type=self.op_type, mch_type=self.mch_type)
        self.old_mch_current_available_jc_nums = np.copy(
            self.mch_current_available_jc_nums)
        self.state = copy.deepcopy(self.old_state)

    def _first_mask_tables(self):
        """Step-0 pair-start table, decision instant and legality mask,
        computed by the same three lines step() runs after every dispatch:
        pair start = _compute_pair_free_time() (setup from the idle class and
        any push past an outage included), decision instant = its minimum over
        compatible pairs, mask = incompatible OR _compute_unavailable_pairs()."""
        self.pair_free_time = self._compute_pair_free_time()
        schedule_matrix = np.ma.array(self.pair_free_time,
                                      mask=self.candidate_process_relation)
        self.next_schedule_time = np.min(
            schedule_matrix.reshape(self.number_of_envs, -1), axis=1).data
        self.dynamic_pair_mask = np.logical_or(
            self.candidate_process_relation, self._compute_unavailable_pairs())
        self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)

    def _apply_defined_first_mask(self):
        """PACT_FIRST_MASK=1: restrict the first decision to the pairs whose
        earliest processing start equals the first decision instant, as at
        every later decision. The parent builds the step-0 table with every
        pair start at 0, so without the flag every compatible pair is legal at
        the first decision. Rebuilds the step-0 state the way the release
        branch does: mask, comp_idx, the per-machine count of legal pairs, and
        the machine and pair features (the time-to-next-outage channel reads
        the decision instant). The waiting-time channels keep their reset
        value 0, as in the release branch."""
        self._first_mask_tables()
        self.mch_current_available_jc_nums = np.sum(~self.dynamic_pair_mask,
                                                    axis=1)
        self.construct_mch_features()
        self.construct_pair_features()
        self.old_state.update(self.fea_j, self.op_mask, self.fea_m,
                              self.mch_mask, self.dynamic_pair_mask,
                              self.comp_idx, self.candidate, self.fea_pairs,
                              op_type=self.op_type, mch_type=self.mch_type)
        self.old_mch_current_available_jc_nums = np.copy(
            self.mch_current_available_jc_nums)
        self.state = copy.deepcopy(self.old_state)

    def reset(self):
        state = super().reset()
        if getattr(self, '_has_release', False):
            # the parent's reset restores the step-0 state but not the
            # env-side mask arrays the dispatching rules read; restore the
            # release-restricted step-0 ones
            self.dynamic_pair_mask = np.copy(
                self.old_state.dynamic_pair_mask_tensor.cpu().numpy())
            self.comp_idx = self.logic_operator(x=~self.dynamic_pair_mask)
        if getattr(self, '_first_mask', False):
            # initial_vars zeroed the table and the instant; recompute the
            # step-0 ones (mch_last_class is idle again, so they equal the
            # tables built in set_initial_data)
            self._first_mask_tables()
        self._attach_token()
        return state

    def _attach_token(self):
        self.state.token_tensor = torch.from_numpy(
            np.copy(self.token_g)).float().to(self.state.device)

    # ------------------------------------------------------------------
    # dynamic state
    # ------------------------------------------------------------------

    def initial_vars(self):
        super().initial_vars()
        # machine's last processed routing class; IDLE_CLASS = cold start
        if hasattr(self, 'c_bits'):
            self.mch_last_class = np.full(
                (self.number_of_envs, self.number_of_machines), IDLE_CLASS,
                dtype=int)
        if getattr(self, '_has_release', False):
            # job i is not present before rho_i: its ready time starts there
            # on both axes, so the setup and the processing of its first
            # operation cannot start earlier
            self.candidate_free_time = np.copy(self.release_norm)
            self.true_candidate_free_time = np.copy(self.true_release)
            self.next_schedule_time = self.release_norm.min(axis=1)
            self.pair_free_time = np.repeat(
                self.release_norm[:, :, None], self.number_of_machines, axis=2)

    # ------------------------------------------------------------------
    # δ-hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _push_past_outages(st, dur, win_s, win_e, eps):
        """Earliest start >= st such that [start, start+dur] avoids every
        outage. win_s/win_e sorted ascending along the last axis; one
        ascending pass is exact because outages are non-overlapping.

        eps guards boundary-touching comparisons: all times are integer
        multiples of the axis quantum (1 h true / slope normalized) with at
        most ~1e-13 float error, so half a quantum separates true overlaps
        from touching intervals. Without it the normalized axis can push
        past an outage the (exact-integer) true axis does not, and the two
        time axes diverge."""
        K = win_s.shape[-1]
        for k in range(K):
            ws = win_s[..., k]
            we = win_e[..., k]
            overlap = (st < we - eps) & (st + dur > ws + eps)
            st = np.where(overlap, we, st)
        return st

    def _setup_into_candidates(self, true_side=False):
        """Setup duration from each machine's last class into each candidate
        [E, J, M] (0 where SETUP inactive)."""
        S = self.true_setup if true_side else self.setup_norm
        cand_class = np.take_along_axis(self.op_class, self.candidate, axis=1)  # [E,J]
        s = S[self.env_idxs[:, None, None],
              self.mch_last_class[:, None, :],
              cand_class[:, :, None]]                                   # [E,J,M]
        s = s * self.c_bits[:, 1][:, None, None]
        if getattr(self, '_has_setup_mult', False):
            s = s * self.setup_mult[:, None, :]
        return s

    def _compute_pair_free_time(self):
        """ATTACHED setup convention: the changeover starts when both the job
        and the machine are available and occupies the machine contiguously
        with processing; [block, block+s+p] must avoid outages. Returns the
        PROCESSING start (block start + setup) per pair. Detached setups are
        deliberately NOT supported: under them the minimum-inbound-setup
        lower envelope of the reward bound is inadmissible."""
        candFT = np.expand_dims(self.candidate_free_time, axis=2)   # [E,J,1]
        mchFT = np.expand_dims(self.mch_free_time, axis=1)          # [E,1,M]
        s = self._setup_into_candidates()                            # [E,J,M]
        block = np.maximum(candFT, mchFT)
        if self.n_win:
            dur = s + self.candidate_pt                              # [E,J,M]
            block = self._push_past_outages(
                block, dur, self.win_s_norm[:, None, :, :],
                self.win_e_norm[:, None, :, :], eps=0.5 * self._slope)
        return block + s

    def _earliest_start(self, chosen_job, chosen_mch, chosen_op, true_side):
        e = self.env_idxs
        if true_side:
            candFT = self.true_candidate_free_time[e, chosen_job]
            mchFT = self.true_mch_free_time[e, chosen_mch]
            S, ws, we = self.true_setup, self.true_win_s, self.true_win_e
            dur = self.true_op_pt[e, chosen_op, chosen_mch]
        else:
            candFT = self.candidate_free_time[e, chosen_job]
            mchFT = self.mch_free_time[e, chosen_mch]
            S, ws, we = self.setup_norm, self.win_s_norm, self.win_e_norm
            dur = self.op_pt[e, chosen_op, chosen_mch]
        setup = S[e, self.mch_last_class[e, chosen_mch],
                  self.op_class[e, chosen_op]] * self.c_bits[:, 1]
        if getattr(self, '_has_setup_mult', False):
            setup = setup * self.setup_mult[e, chosen_mch]
        block = np.maximum(candFT, mchFT)
        if self.n_win:
            eps = 0.25 if true_side else 0.5 * self._slope
            block = self._push_past_outages(block, setup + dur,
                                            ws[e, chosen_mch],
                                            we[e, chosen_mch], eps=eps)
        return block + setup

    def _post_step_constraint_update(self, chosen_job, chosen_mch, chosen_op):
        self.mch_last_class[self.env_idxs, chosen_mch] = \
            self.op_class[self.env_idxs, chosen_op]

    def _compute_unavailable_pairs(self):
        """mask_mode='nondelay' (default): the parent's rule — only pairs that
        start at the decision instant are admissible.

        mask_mode='active': the Giffler–Thompson
        active-schedule condition. A pair stays admissible iff its processing
        start is strictly earlier than C*, the earliest completion achievable
        among compatible pairs. Choosing a pair that starts later than the
        decision instant IS the intentional-idling decision the non-delay mask
        forbids: the machine waits for that job while other machines schedule
        on. The non-delay set is always contained in the active set (C* is at
        least one processing time above the minimum start), so the action set
        can never be empty, each step still schedules exactly one operation,
        and episode length, reward and latency are unchanged.

        Strictness on the normalized axis: all times are integer multiples of
        the quantum (1 h true = self._slope normalized) up to ~1e-13 float
        error, so half a quantum separates "strictly earlier" from a tie —
        the same guard _push_past_outages uses.
        """
        if self.mask_mode != 'active':
            if self.mask_setting('mask_ties') == 'exact':
                # Tie-tolerant non-delay rule (mask_ties='exact', or
                # PACT_MASK_TOL=1 when the argument is None): a pair whose
                # start equals the decision instant in hours stays admissible
                # even when float error on the normalized axis puts it a few
                # ulps above the minimum. Half a quantum separates a tie from
                # a later start. Legacy by default, so every published number
                # is unchanged.
                return (self.pair_free_time
                        > self.next_schedule_time[:, None, None]
                        + 0.5 * self._slope)
            return super()._compute_unavailable_pairs()
        completion = np.where(self.candidate_process_relation, np.inf,
                              self.pair_free_time + self.candidate_pt)
        cstar = completion.reshape(self.number_of_envs, -1).min(axis=1)  # [E]
        eps = 0.5 * self._slope
        return self.pair_free_time >= cstar[:, None, None] - eps

    def _bound_min_term(self):
        """The machine-side term of the shaping potential: op_min_pt + delta.

        Three envelopes, selected by `delta_mode`. All three are TERMINALLY EXACT
        and action-independent at s_0, so by Proposition 1 all three give a reward
        that is exactly aligned with the makespan. They differ ONLY in
        admissibility, which is what makes the comparison a clean test of the
        theorem rather than of the reward's scale.

          'min'  (default, the paper's)  delta = min over source classes (incl.
                 idle) of the setup into this op's class. ADMISSIBLE: no feasible
                 completion can undercut it. Tight.

          'zero' delta = 0. ADMISSIBLE, and looser. Isolates TIGHTNESS with
                 admissibility held fixed.

          'max'  delta = MAX over source classes. INADMISSIBLE: it charges the
                 most expensive changeover any predecessor could impose, which the
                 schedule may never pay, so the estimate can exceed a realizable
                 completion. Isolates ADMISSIBILITY with tightness held roughly
                 fixed (it is the same construction with the inequality reversed).

        The third arm is the one that tests whether Theorem 1 buys anything.
        Potential-based shaping is policy-invariant for ANY terminally exact
        potential, so alignment alone does not need admissibility; if 'max' trains
        as well as 'min', the theorem is a correctness certificate for a property
        nothing depends on, and the paper should say so.
        """
        mode = resolve_delta_mode(self)
        if mode == 'zero':
            return self.op_min_pt
        col = (self.setup_norm.max(axis=1) if mode == 'max'          # [E,C]
               else self.setup_norm.min(axis=1))
        d = np.take_along_axis(col, self.op_class, axis=1)           # [E,N]
        if getattr(self, '_has_setup_mult', False):
            # Per-machine multiplier: the setup term moves inside the minimum
            # over eligible machines, min_k [p_ijk + u_k * delta(op)]. Machine
            # k charges at least u_k * delta(op) into this op, so the term is
            # still a lower bound of the op's machine occupancy. With every
            # u_k = 1 it equals op_min_pt + delta (rounding is monotone).
            # self.op_pt is the masked [E,N,M] array here (incompatible
            # machines masked), as for op_min_pt in the parent.
            dd = (d * self.c_bits[:, 1][:, None])[:, :, None]          # [E,N,1]
            cand = self.op_pt + self.setup_mult[:, None, :] * dd     # [E,N,M]
            return np.min(cand, axis=-1).data
        return self.op_min_pt + d * self.c_bits[:, 1][:, None]

    # ------------------------------------------------------------------
    # features (family mode adds 2 machine channels)
    # ------------------------------------------------------------------

    def _mch_min_next_setup(self):
        """[E, M] min setup from the machine's last class into any candidate
        it can process (0 when SETUP inactive or no compatible candidate)."""
        s = self._setup_into_candidates()                            # [E,J,M]
        blocked = self.candidate_process_relation                    # [E,J,M]
        s = np.where(blocked, np.inf, s)
        out = s.min(axis=1)
        return np.where(np.isfinite(out), out, 0.0)

    def _mch_time_to_next_outage(self):
        """[E, M] time from the current decision instant to the machine's
        next unfinished outage, clipped to [0, H]; H when none remain or
        WINDOW inactive (constant channels z-normalize to ~0)."""
        H = self.horizon_norm[:, None]                               # [E,1]
        if not self.n_win:
            return np.tile(H, (1, self.number_of_machines))
        t = self.next_schedule_time[:, None, None]                   # [E,1,1]
        pending = self.win_e_norm > t                                # [E,M,K]
        starts = np.where(pending, self.win_s_norm, np.inf)
        nxt = starts.min(axis=2)                                     # [E,M]
        ttno = np.clip(nxt - self.next_schedule_time[:, None], 0.0, H)
        return np.where(np.isfinite(nxt), ttno, np.tile(H, (1, self.number_of_machines)))

    def construct_mch_features(self):
        if not self.family_mode:
            return super().construct_mch_features()
        setup_ch = self._mch_min_next_setup()
        window_ch = self._mch_time_to_next_outage()
        if 'setup' in self.blind_channels:
            setup_ch = np.zeros_like(setup_ch)
        if 'window' in self.blind_channels:
            window_ch = np.zeros_like(window_ch)
        self.fea_m = np.stack((self.mch_current_available_jc_nums,
                               self.mch_current_available_op_nums,
                               self.mch_max_pt,
                               self.mch_mean_pt,
                               self.mch_waiting_time,
                               self.mch_remain_work,
                               self.mch_free_time,
                               self.mch_working_flag,
                               setup_ch,
                               window_ch), axis=2)
        if self.step_count != self.number_of_ops:
            self.norm_machine_features()

    def construct_pair_features(self):
        """pair_extra='sw' (optional pair channels) appends two channels to the parent's
        eight: the EXACT setup this (candidate, machine) assignment would
        incur from the machine's current last class, and the window-induced
        push delay of its block, both on the normalized time axis (the scale
        of the candidate_pt channel). These are the two quantities Sec. V-C
        names as unobserved; zero wherever the member is inactive. Respects
        the E3 observation blinding the machine channels respect."""
        super().construct_pair_features()
        if not (self.family_mode and self.pair_extra == 'sw'):
            return
        s = self._setup_into_candidates()                        # [E,J,M]
        start = self._compute_pair_free_time()                   # block + s
        block0 = np.maximum(self.candidate_free_time[:, :, None],
                            self.mch_free_time[:, None, :])
        push = np.maximum(start - s - block0, 0.0)               # [E,J,M]
        if 'setup' in self.blind_channels:
            s = np.zeros_like(s)
        if 'window' in self.blind_channels:
            push = np.zeros_like(push)
        self.fea_pairs = np.concatenate(
            [self.fea_pairs, s[..., None], push[..., None]], axis=-1)

    def construct_op_features(self):
        if self.family_mode and 'lag' in self.blind_channels:
            # zero the two anticipatory lag channels at observation time
            # (dynamics keep the true lags): op_lag / op_remain_lag feed
            # positions 10-11 of fea_j via the parent's stacking
            saved_lag, saved_rem = self.op_lag, self.op_remain_lag
            self.op_lag = np.zeros_like(self.op_lag)
            self.op_remain_lag = np.zeros_like(self.op_remain_lag)
            super().construct_op_features()
            self.op_lag, self.op_remain_lag = saved_lag, saved_rem
            return
        super().construct_op_features()
