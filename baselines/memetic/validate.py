"""Validation of returned schedules.

Every schedule is checked twice:

  1. constraint_family.checks.check_schedule_validation (configuration check
     C6), the check the CP-SAT and policy schedules go through: it runs
     schedule_validator.validate_schedule and
     reference_sim.validate_schedule;
  2. attached_setup_violations below: the setup block of an operation starts
     no earlier than the time its job is present (end of the job
     predecessor plus its lag, or the release date of a first operation).
     reference_sim.validate_schedule tests the same condition; this second,
     independent implementation is kept so that a defect in either one is
     caught by the other.

A schedule is accepted only when both lists of violations are empty.
"""
import numpy as np

from constraint_family import checks


def setup_blocks(inst, op_mch, op_st):
    """Block start (processing start minus the attached setup from the
    machine predecessor in start order) of every operation."""
    op_mch = np.asarray(op_mch, dtype=np.int64)
    op_st = np.asarray(op_st, dtype=np.int64)
    blk = op_st.copy()
    if not inst.has_setup:
        return blk
    for m in range(inst.M):
        ops = np.nonzero(op_mch == m)[0]
        ops = ops[np.argsort(op_st[ops], kind='stable')]
        last = inst.idle
        for o in ops:
            blk[o] = op_st[o] - inst.S[last, inst.cls[o]]
            last = inst.cls[o]
    return blk


def attached_setup_violations(inst, op_mch, op_st, op_ct):
    """Violations of 'the setup starts only when the job is present'."""
    op_ct = np.asarray(op_ct, dtype=np.int64)
    blk = setup_blocks(inst, op_mch, op_st)
    v = []
    for o in range(inst.N):
        jp = inst.jprev[o]
        r = inst.rel_op[o] if jp < 0 else op_ct[jp] + inst.lag[jp]
        if blk[o] < r:
            v.append(f'op {o}: setup block starts at {blk[o]} before its job '
                     f'is present at {r}')
    return v


def validate(inst, op_mch, op_st, op_ct):
    """Return (ok, violations) for a complete schedule."""
    res = checks.check_schedule_validation(
        inst.desc, inst.job_length, inst.pt, np.asarray(op_mch, dtype=int),
        np.asarray(op_ct, dtype=float), op_st=np.asarray(op_st, dtype=float))
    viol = list(res.details.get('violations', []))
    viol += [f'[attached_setup] {s}' for s in
             attached_setup_violations(inst, op_mch, op_st, op_ct)]
    ms = float(np.max(op_ct))
    return (len(viol) == 0), viol, ms
