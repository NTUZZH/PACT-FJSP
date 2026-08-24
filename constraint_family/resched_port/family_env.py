"""Family constraints on the ReSched backbone (the portability check of the supplement).

Ports the SHARED CONSTRUCTION, not the mask philosophy: the two-hook family
transition (attached setups, window push, job-side lags) and the certified
reward estimate of Theorem 1 are injected into ReSched's FJSPEnv
(external/ReSched, ICLR 2026), whose transformer policy, subproblem state and
fully order-permissive action space stay exactly as released.

Where the hooks land:

  _makespan   ReSched computes st = max(m_AT, o_AT); et = st + dur, and
              propagates a lower-bound estimate down the job chain by delta
              push. The family version computes
                block = push_windows(max(m_AT, o_AT), s + dur)
                st    = block + s,   et = st + dur,   s = S[last_class, class]
              and lets the successor become available at et + lag.

  est bound   ReSched initializes est_finish_time as the chain cumsum of
              per-op min durations: exactly the s0 value of our recursion
              with a zero envelope and zero lags. The family init uses
              m_ij = min over compatible k of (dur + a_S * min-inbound setup)
              and adds the lag between chain steps, which IS the estimate of
              Theorem 1; ReSched's own delta-push then maintains it, because
              the recursion is additive along job chains. The reward
              (negative change of the estimated makespan) is therefore the
              paper's certified reward, unchanged in form.

  final step  ReSched auto-schedules the last remaining operation on the
              machine with minimal m_AT + dur; the family override picks the
              machine by the family-aware finishing time instead.

Everything is numpy on the true integer-hour axis (ReSched normalizes
nothing), so cross-checking against constraint_family.reference_sim is exact:
scripts/p49_resched_crosscheck.py asserts identical start/end times for
identical dispatch sequences, terminal exactness of the estimate, and
admissibility on random rollouts.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESCHED_DIR = os.path.join(_ROOT, 'external', 'ReSched', 'PPO')
for p in (_ROOT, RESCHED_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from SchedulingEnvironment import FJSPEnv, SchedulingProblemEnv  # noqa: E402
from constraint_family.descriptor import IDLE_CLASS, N_CLASSES    # noqa: E402


def build_problem_arrays(jl_list, pt_list):
    """Our (job_length [J], op_pt [N, M]) instances -> ReSched Problem arrays.

    All instances in a batch must share N and M (same-op-nums protocol).
    Returns (job_idx, operation_idx, duration, dependency, connection).
    dependency[b, i, j] = 1 iff i is the immediate predecessor of j;
    connection = job-chain adjacency with self-loops (their o2o mask).
    """
    B = len(jl_list)
    N, M = pt_list[0].shape
    job_idx = np.zeros((B, N), dtype=int)
    op_idx = np.zeros((B, N), dtype=int)
    dur = np.zeros((B, N, M))
    dep = np.zeros((B, N, N), dtype=bool)
    for b, (jl, pt) in enumerate(zip(jl_list, pt_list)):
        assert pt.shape == (N, M), 'batch must share N and M'
        first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
        for j, L in enumerate(jl):
            for r in range(int(L)):
                o = first[j] + r
                job_idx[b, o] = j
                op_idx[b, o] = r
                if r > 0:
                    dep[b, o - 1, o] = True
        dur[b] = pt
    same_job = job_idx[:, :, None] == job_idx[:, None, :]
    eye = np.eye(N, dtype=bool)[None]
    connection = same_job | eye
    return job_idx, op_idx, dur, dep, connection


def descriptor_arrays(desc_list, n_ops, n_m):
    """Descriptor batch -> dense arrays on the true axis."""
    B = len(desc_list)
    bits = np.array([[d.has_lag, d.has_setup, d.has_window]
                     for d in desc_list], dtype=float)         # [B, 3]
    lag = np.stack([np.asarray(d.lag, dtype=float) for d in desc_list])
    cls = np.stack([np.asarray(d.op_class, dtype=int) for d in desc_list])
    # the family convention: rows 0..C-1 = source classes, row C = the idle
    # source (IDLE_CLASS); always allocated at the global class count so the
    # idle-row index stays valid in regimes with SETUP inactive
    setup = np.zeros((B, N_CLASSES + 1, N_CLASSES))
    for b, d in enumerate(desc_list):
        if d.has_setup and d.setup is not None:
            setup[b] = d.setup
    K = max((max((len(w) for w in d.windows), default=0)
             for d in desc_list if d.has_window and d.windows is not None),
            default=0)
    win_s = np.full((B, n_m, max(K, 1)), np.inf)
    win_e = np.full((B, n_m, max(K, 1)), np.inf)
    for b, d in enumerate(desc_list):
        if d.has_window and d.windows is not None:
            for m, spans in enumerate(d.windows):
                for k, (s, e) in enumerate(sorted(spans)):
                    win_s[b, m, k] = s
                    win_e[b, m, k] = e
    return bits, lag, cls, setup, win_s, win_e, K


class FamilyReSchedEnv(FJSPEnv):
    """ReSched's FJSPEnv with the family transition and certified bound."""

    EPS = 0.25          # true integer-hour axis; quarter-hour separates ties

    def __init__(self):
        SchedulingProblemEnv.__init__(self)
        self.generator = None            # data is always loaded explicitly
        self._fam = None

    # ---------------- data ----------------

    def load_family_batch(self, jl_list, pt_list, desc_list):
        self.batch_size = len(jl_list)
        (self.problem.job_idx, self.problem.operation_idx,
         self.problem.duration, self.problem.dependency,
         connection) = build_problem_arrays(jl_list, pt_list)
        self.problem.connection = self._with_self_connection(connection)
        N, M = pt_list[0].shape
        self._fam = descriptor_arrays(desc_list, N, M)
        return self.reset_state()

    # ---------------- family state ----------------

    def reset_state(self):
        state = super().reset_state()
        if self._fam is None:
            return state
        bits, lag, cls, setup, win_s, win_e, K = self._fam
        B, M = self.batch_size, self.problem.duration.shape[-1]
        self.mch_last_class = np.full((B, M), IDLE_CLASS, dtype=int)

        # certified estimate at s0: m_ij = min over compatible machines of
        # (dur + a_S * min-inbound setup into the op's class); chain cumsum
        # with the lag added between consecutive operations of a job.
        dur = self.problem.duration
        env_min = setup.min(axis=1)                             # [B, C]
        delta = np.take_along_axis(env_min, cls, axis=1)        # [B, N]
        delta = delta * bits[:, 1][:, None]
        m_term = np.where(dur == 0, np.inf, dur
                          + delta[:, :, None]).min(axis=-1)     # [B, N]
        est = np.zeros_like(m_term)
        job_idx = self.problem.job_idx
        lag_eff = lag * bits[:, 0][:, None]
        for b in range(B):
            for j in np.unique(job_idx[b]):
                mask = np.flatnonzero(job_idx[b] == j)
                acc = 0.0
                for i, o in enumerate(mask):
                    if i > 0:
                        acc += lag_eff[b, mask[i - 1]]
                    acc += m_term[b, o]
                    est[b, o] = acc
        self.state.est_finish_time = est
        self.state.est_last_makespan = est.max(axis=1)
        return self.state

    # ---------------- family transition ----------------

    @staticmethod
    def _push_windows_vec(st, dur, ws, we, eps):
        """Earliest start >= st avoiding every outage; vectorized over the
        batch. ws/we sorted ascending along the last axis."""
        K = ws.shape[-1]
        for k in range(K):
            overlap = (st < we[..., k] - eps) & (st + dur > ws[..., k] + eps)
            st = np.where(overlap, we[..., k], st)
        return st

    def _makespan(self, batch_idx, jo_idx, machine_idx):
        if self._fam is None:
            return super()._makespan(batch_idx, jo_idx, machine_idx)
        bits, lag, cls, setup, win_s, win_e, K = self._fam
        b = np.arange(len(batch_idx))

        dep = self.problem.dependency
        succ_mask = dep[b, jo_idx, :]

        dur = self.problem.duration[b, jo_idx, machine_idx]
        mat = self.state.m_AT[b, machine_idx]
        oat = self.state.available_time_o[b, jo_idx]

        s = (setup[b, self.mch_last_class[b, machine_idx], cls[b, jo_idx]]
             * bits[b, 1])
        block = np.maximum(mat, oat)
        if K:
            block = self._push_windows_vec(
                block, s + dur, win_s[b, machine_idx], win_e[b, machine_idx],
                self.EPS)
        st = block + s
        et = st + dur

        self.state.m_AT[b, machine_idx] = et
        self.mch_last_class[b, machine_idx] = cls[b, jo_idx]
        self.state.start_time[b, jo_idx, machine_idx] = st
        self.state.finish_time[b, jo_idx, machine_idx] = et

        # successor availability carries the job-side lag (hook 1)
        succ_exist = succ_mask.sum(axis=1) == 1
        first_succ = np.argmax(succ_mask, axis=1)
        release = et + lag[b, jo_idx] * bits[b, 0]
        self.state.available_time_o[b[succ_exist],
                                    first_succ[succ_exist]] = \
            release[succ_exist]

        # ReSched's delta push maintains the estimate down the job chain
        job_mat = self.problem.job_idx[b]
        op_mat = self.problem.operation_idx[b]
        job_id = self.problem.job_idx[b, jo_idx]
        op_id = self.problem.operation_idx[b, jo_idx]
        delta = et - self.state.est_finish_time[b, jo_idx]
        mask = (job_mat == job_id[:, None]) & (op_mat >= op_id[:, None])
        self.state.est_finish_time[b] += mask * delta[:, None]

        real_mk = self.state.finish_time.max(axis=1).max(axis=1)
        est_mk = self.state.est_finish_time.max(axis=1)
        return real_mk, est_mk

    # ---------------- family-aware terminal dispatch ----------------

    def step(self, action):
        assert self._fam is not None, 'family env used without a family batch'
        step_reward = self._step(action)
        if self.state.action_mask.shape[1] == 1:
            bits, lag, cls, setup, win_s, win_e, K = self._fam
            duration = self.state.duration.squeeze(axis=1)      # [B, M]
            jo = np.array([np.flatnonzero(
                (self.problem.job_idx[b2] == self.state.job_idx[b2, 0])
                & (self.problem.operation_idx[b2]
                   == self.state.operation_idx[b2, 0]))[0]
                for b2 in range(self.batch_size)])
            b = np.arange(self.batch_size)
            oat = self.state.available_time_o[b, jo][:, None]
            s = (setup[b[:, None], self.mch_last_class,
                       cls[b, jo][:, None]] * bits[:, 1][:, None])
            block = np.maximum(self.state.m_AT, oat)
            dmask = np.where(duration == 0, np.inf, duration)
            if K:
                block = self._push_windows_vec(block, s + dmask,
                                               win_s, win_e, self.EPS)
            finish = block + s + dmask
            machine_idx = np.argmin(finish, axis=1)
            action = (np.zeros(self.batch_size, dtype=int), machine_idx)
            _ = self._step(action)
            self.solution.job_idx = np.array(self.solution.job_idx).T
            self.solution.operation_idx = np.array(
                self.solution.operation_idx).T
            self.solution.machine_idx = np.array(self.solution.machine_idx).T
            self.solution.start_time = np.array(self.solution.start_time).T
            self.solution.end_time = np.array(self.solution.end_time).T
            done = True
        else:
            done = False
        return self.state, step_reward, done
