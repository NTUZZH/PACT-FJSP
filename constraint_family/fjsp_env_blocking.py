"""Out-of-family stress environment: BLOCKING (this paper, E4).

Blocking violates the monotone-delay premise: a machine is released not at
its own completion but when the job's NEXT operation starts downstream
(no intermediate buffer — in PPVC terms, a module holds its station until
the next station can take it; with LAG active it also cures in place).
This file exists to map where the family's guarantee ENDS, not to extend it.

Semantics (list scheduling):
  * scheduling (i,j) on k: start st = max(job_ready(i), release_base(k)),
    where release_base(k) is the completion of k's previous op;
  * if (i,j-1) ran on k' != k, machine k' is released at st (it held the
    job until transfer); if j-1 was on the SAME k, release is implicit;
  * while k' holds a job, every other candidate pair on k' is masked
    (earliest start = BIG). If ALL remaining pairs of an env are masked,
    the env is DEADLOCKED (classic blocking-shop deadlock): the rollout
    aborts and the instance is reported DNF.
  * the job-side LAG composes: with lag active the successor is ready at
    ct + lag, and the station stays blocked throughout (curing in place).

The policy's features see release_base as the machine-free time (the
family policy's plausible-but-wrong belief); true availability is enforced
through masking and the start-time hook.

The family bound remains SOUND under blocking, and this is provable rather than
lucky. Blocking only ADDS delay, so on the lag-only slice this experiment uses
(a_S = 0) a blocked completion is max(ready, .) + p >= ready + min_k p, which is
exactly the family bound. The zero over-estimations reported for E4 are therefore
a theorem, not a measurement. What blocking destroys is LIVENESS, not the
estimate: the policy deadlocks.

No-wait is the genuinely unsound case. It forces a changeover to run before its
job arrives, i.e. the detached convention that Remark 1 shows breaks the setup
envelope, which is why the no-wait probe is also confined to the lag-only slice.
"""
import numpy as np

from constraint_family.fjsp_env_family import FJSPEnvFamily

BIG = 1e9


class FJSPEnvBlocking(FJSPEnvFamily):
    """Family env (LAG only; SETUP/WINDOW off in v1) + blocking dynamics."""

    def initial_vars(self):
        super().initial_vars()
        E, J, M = (self.number_of_envs, self.number_of_jobs,
                   self.number_of_machines)
        self.job_prev_mch = np.full((E, J), -1, dtype=int)
        self.mch_blocked_by = np.full((E, M), -1, dtype=int)
        self.deadlock = np.zeros(E, dtype=bool)

    # ---- helpers ------------------------------------------------------

    def _blocked_pair_mask(self):
        """[E, J, M] True where the pair's machine is held by ANOTHER job."""
        blocked = self.mch_blocked_by[:, None, :]                  # [E,1,M]
        jobs = np.arange(self.number_of_jobs)[None, :, None]       # [1,J,1]
        return (blocked >= 0) & (blocked != jobs)

    # ---- δ-hooks ------------------------------------------------------

    def _compute_pair_free_time(self):
        est = super()._compute_pair_free_time()
        return np.where(self._blocked_pair_mask(), BIG, est)

    def _post_step_constraint_update(self, chosen_job, chosen_mch, chosen_op):
        """Blocking bookkeeping. The base step() calls this hook AFTER the
        chosen op's times are written but BEFORE the next decision's
        pair_free_time / masks are computed — exactly where holds and
        releases must land."""
        super()._post_step_constraint_update(chosen_job, chosen_mch,
                                             chosen_op)
        e_all = self.env_idxs
        # the executed action must never be on a machine held by another job
        holder = self.mch_blocked_by[e_all, chosen_mch]
        assert ((holder < 0) | (holder == chosen_job)).all(), \
            'blocking violation: scheduled on a held machine (masking bug)'

        # release the machine that held this job's PREDECESSOR at the chosen
        # op's start (transfer instant); starts recovered from written times
        st_true = self.true_op_ct[e_all, chosen_op] - \
            self.true_op_pt[e_all, chosen_op, chosen_mch]
        st_norm = self.op_ct[e_all, chosen_op] - np.asarray(
            self.op_pt[e_all, chosen_op, chosen_mch])
        prev = self.job_prev_mch[e_all, chosen_job]
        held = (prev >= 0) & (self.mch_blocked_by[e_all, prev]
                              == chosen_job) & (prev != chosen_mch)
        if held.any():
            e = e_all[held]
            self.true_mch_free_time[e, prev[held]] = st_true[held]
            self.mch_free_time[e, prev[held]] = st_norm[held]
            self.mch_blocked_by[e, prev[held]] = -1

        # hold the CHOSEN machine iff the job has a successor; else release
        has_succ = (chosen_op != self.job_last_op_id[e_all, chosen_job])
        self.mch_blocked_by[e_all, chosen_mch] = np.where(
            has_succ, chosen_job, -1)
        self.job_prev_mch[e_all, chosen_job] = chosen_mch

    def step(self, actions):
        state, reward, done = super().step(actions)
        # deadlock: every remaining pair masked (schedule matrix all BIG)
        live = ~self.mask.all(axis=1) & (self.step_count < self.number_of_ops)
        if live.any():
            self.deadlock = self.deadlock | (
                live & (self.next_schedule_time >= BIG / 2))
        return state, reward, done


def rollout_blocking_reference(job_length, op_pt, lag, actions):
    """Slow independent simulator of the blocking(+lag) dynamics for tests.

    Returns op completion times and makespan, or None if the action
    sequence deadlocks (a chosen machine is held by another job — such an
    action is illegal under blocking and the caller should only feed
    sequences whose pairs were unmasked)."""
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    J = len(job_length)
    M = op_pt.shape[1]
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)
    nxt = first.copy()
    ready = np.zeros(J)
    free = np.zeros(M)
    held_by = np.full(M, -1, dtype=int)
    prev_mch = np.full(J, -1, dtype=int)
    N = op_pt.shape[0]
    op_ct = np.full(N, -1.0)
    lag = np.zeros(N) if lag is None else np.asarray(lag, dtype=float)

    for (j, m) in actions:
        op = nxt[j]
        if held_by[m] >= 0 and held_by[m] != j:
            return None                       # illegal under blocking
        st = max(ready[j], free[m])
        p = prev_mch[j]
        if p >= 0 and held_by[p] == j and p != m:
            free[p] = st                      # release the holder at transfer
            held_by[p] = -1
        ct = st + op_pt[op, m]
        op_ct[op] = ct
        ready[j] = ct + lag[op]
        free[m] = ct
        if op != first[j] + job_length[j] - 1:
            held_by[m] = j                    # station held until transfer
        elif held_by[m] == j:
            held_by[m] = -1
        prev_mch[j] = m
        nxt[j] += 1
    return {'op_ct': op_ct, 'makespan': float(np.nanmax(op_ct))}
