"""Slow, obviously-correct reference simulator for the constraint family.

Independent code path used ONLY to cross-validate FJSPEnvFamily's vectorized
transition dynamics in unit tests (and to re-validate schedules): pure-Python
loops, true-hour units, no normalization, no feature machinery.
"""
import numpy as np

from constraint_family.descriptor import IDLE_CLASS


def push_past_outages_scalar(st, dur, spans):
    """Earliest start >= st with [start, start+dur] avoiding sorted,
    non-overlapping outage spans."""
    for (s, e) in spans:
        if st < e and st + dur > s:
            st = e
    return st


def simulate(job_length, op_pt, desc, actions):
    """Execute (job, machine) decisions under descriptor `desc`.

    :param job_length: [J] ops per job
    :param op_pt: [N, M] true processing times (0 = incompatible)
    :param desc: ConstraintDescriptor (true hours)
    :param actions: sequence of (job, machine) pairs, one per operation
    :return: dict with op start/completion arrays and the makespan
    """
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    J = len(job_length)
    M = op_pt.shape[1]
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)

    next_op = first.copy()
    job_ready = np.zeros(J)
    mch_free = np.zeros(M)
    mch_last = np.full(M, IDLE_CLASS, dtype=int)

    N = op_pt.shape[0]
    op_st = np.full(N, -1.0)
    op_ct = np.full(N, -1.0)
    op_mch = np.full(N, -1, dtype=int)

    for (j, m) in actions:
        op = next_op[j]
        assert op < first[j] + job_length[j], f'job {j} already finished'
        pt = op_pt[op, m]
        assert pt > 0, f'op {op} incompatible with machine {m}'

        # ATTACHED setup convention: the changeover starts only when both the
        # job and the machine are available, and occupies the machine
        # contiguously with processing ([block_st, block_st + s + p] must
        # avoid outages). Under the detached convention the minimum-inbound-
        # setup lower envelope is NOT admissible (setup can be absorbed into
        # the job's waiting time) — found by fuzz test.
        s = (desc.setup[mch_last[m], desc.op_class[op]]
             if desc.has_setup else 0.0)
        block_st = max(job_ready[j], mch_free[m])
        if desc.has_window and desc.windows is not None:
            block_st = push_past_outages_scalar(block_st, s + pt,
                                                desc.windows[m])
        st = block_st + s
        ct = st + pt

        op_st[op], op_ct[op], op_mch[op] = st, ct, m
        job_ready[j] = ct + (desc.lag[op] if desc.has_lag else 0.0)
        mch_free[m] = ct
        mch_last[m] = desc.op_class[op]
        next_op[j] += 1

    return {'op_st': op_st, 'op_ct': op_ct, 'op_mch': op_mch,
            'makespan': op_ct.max()}


def validate_schedule(job_length, op_pt, desc, op_st, op_ct, op_mch,
                      tol=1e-6):
    """Independent feasibility check of a completed schedule under `desc`.

    Returns a list of violation strings (empty = feasible).
    """
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    M = op_pt.shape[1]
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)
    v = []

    # processing times & compatibility
    for op in range(op_pt.shape[0]):
        m = int(op_mch[op])
        if op_pt[op, m] <= 0:
            v.append(f'op {op} on incompatible machine {m}')
        elif abs(op_ct[op] - op_st[op] - op_pt[op, m]) > tol:
            v.append(f'op {op} duration mismatch')
        if op_st[op] < -tol:
            v.append(f'op {op} negative start')

    # job precedence + lags
    for j, L in enumerate(job_length):
        for r in range(1, L):
            a, b = first[j] + r - 1, first[j] + r
            need = op_ct[a] + (desc.lag[a] if desc.has_lag else 0.0)
            if op_st[b] < need - tol:
                v.append(f'job {j}: op {b} starts before pred ct+lag')

    # machine capacity + setups + windows (attached-setup convention:
    # the machine is occupied over [op_st - setup, op_ct])
    for m in range(M):
        ops = [op for op in range(op_pt.shape[0]) if op_mch[op] == m]
        ops.sort(key=lambda o: op_st[o])
        last_class = IDLE_CLASS
        prev_ct = 0.0
        for o in ops:
            s_dur = (desc.setup[last_class, desc.op_class[o]]
                     if desc.has_setup else 0.0)
            block_st = op_st[o] - s_dur
            if block_st < prev_ct - tol:
                v.append(f'mch {m}: op {o} setup+proc block overlaps predecessor')
            if desc.has_window and desc.windows is not None:
                for (s, e) in desc.windows[m]:
                    if block_st < e - tol and op_ct[o] > s + tol:
                        v.append(f'mch {m}: op {o} block overlaps outage ({s},{e})')
            prev_ct = op_ct[o]
            last_class = desc.op_class[o]
    return v
