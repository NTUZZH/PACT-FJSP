"""
schedule_validator.py
---------------------
Standalone feasibility checker for FJSP schedules with post-operation time-lags.
No imports from the repository — only numpy.

Violation prefixes for grep:
  [compat]    machine incompatibility
  [start]     negative start time
  [precedence] job precedence or lag violation
  [overlap]   machine no-overlap violation
  [bound]     makespan below theoretical lower bound
  [release]   first operation of a job starts before the job's release date
"""

import numpy as np
import sys


def reconstruct_starts(op_pt, assigned_mch, op_ct):
    """Return start[i] = op_ct[i] - op_pt[i, assigned_mch[i]] for each op i."""
    op_pt = np.asarray(op_pt, dtype=float)
    assigned_mch = np.asarray(assigned_mch, dtype=int)
    op_ct = np.asarray(op_ct, dtype=float)
    n = len(op_ct)
    starts = np.empty(n, dtype=float)
    for i in range(n):
        starts[i] = op_ct[i] - op_pt[i, assigned_mch[i]]
    return starts


def validate_schedule(job_length, op_pt, time_lag, assigned_mch, op_ct, tol=1e-6,
                      release=None):
    """
    Check feasibility of an FJSP schedule with post-operation time-lags.

    Parameters
    ----------
    job_length   : array-like, shape (J,)  — number of ops per job
    op_pt        : array-like, shape (N, M) — processing times; 0 = incompatible
    time_lag     : array-like, shape (N,)  — mandatory wait after op i completes
    assigned_mch : array-like, shape (N,)  — machine index chosen for each op
    op_ct        : array-like, shape (N,)  — completion time of each op
    tol          : float — numerical tolerance for inequality checks
    release      : array-like, shape (J,), optional — release date of each job;
                   the job's first operation may not start before it.
                   None (default) = every job released at 0 (checks 1-5 only).

    Returns
    -------
    dict with keys:
        "feasible"    : bool
        "violations"  : list of human-readable strings (empty if feasible)
        "makespan"    : float  (max of op_ct)
        "lower_bound" : float  (theoretical lower bound, see check 5)
    """
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    assigned_mch = np.asarray(assigned_mch, dtype=int)
    op_ct = np.asarray(op_ct, dtype=float)

    n_ops = op_pt.shape[0]
    violations = []

    # Build op -> job mapping and per-job op lists
    op_to_job = np.empty(n_ops, dtype=int)
    job_op_ranges = []  # list of (start_idx, end_idx) for each job
    idx = 0
    for j, length in enumerate(job_length):
        job_op_ranges.append((idx, idx + length))
        op_to_job[idx: idx + length] = j
        idx += length

    # Derive start times
    starts = reconstruct_starts(op_pt, assigned_mch, op_ct)

    # ------------------------------------------------------------------
    # Check 1: Compatibility
    # ------------------------------------------------------------------
    for i in range(n_ops):
        m = assigned_mch[i]
        if op_pt[i, m] <= 0:
            violations.append(
                f"[compat] op {i} assigned to machine {m} but op_pt[{i},{m}]="
                f"{op_pt[i, m]:.4g} (must be > 0)"
            )

    # ------------------------------------------------------------------
    # Check 2: Non-negative start times
    # ------------------------------------------------------------------
    for i in range(n_ops):
        if starts[i] < -tol:
            violations.append(
                f"[start] op {i} has start={starts[i]:.6g} < 0"
            )

    # ------------------------------------------------------------------
    # Check 3: Job precedence with time-lag
    # ------------------------------------------------------------------
    for j, (s, e) in enumerate(job_op_ranges):
        for k in range(s + 1, e):  # k is the successor op
            pred = k - 1
            min_start = op_ct[pred] + time_lag[pred]
            if starts[k] < min_start - tol:
                violations.append(
                    f"[precedence] job {j}: op {k} starts at {starts[k]:.6g} "
                    f"but must be >= op_ct[{pred}]({op_ct[pred]:.6g}) "
                    f"+ lag[{pred}]({time_lag[pred]:.6g}) = {min_start:.6g}"
                )

    # ------------------------------------------------------------------
    # Check 4: Machine no-overlap
    # ------------------------------------------------------------------
    n_mch = op_pt.shape[1]
    for m in range(n_mch):
        ops_on_m = [i for i in range(n_ops) if assigned_mch[i] == m]
        if len(ops_on_m) < 2:
            continue
        ops_on_m.sort(key=lambda i: starts[i])
        for idx_a in range(len(ops_on_m) - 1):
            a = ops_on_m[idx_a]
            b = ops_on_m[idx_a + 1]
            if starts[b] < op_ct[a] - tol:
                violations.append(
                    f"[overlap] machine {m}: op {b} starts at {starts[b]:.6g} "
                    f"before op {a} completes at {op_ct[a]:.6g}"
                )

    # ------------------------------------------------------------------
    # Check 4b: Release dates (only when given)
    # ------------------------------------------------------------------
    if release is not None:
        release = np.asarray(release, dtype=float)
        if release.shape != (len(job_op_ranges),):
            violations.append(
                f"[release] release has shape {release.shape}, expected "
                f"({len(job_op_ranges)},)"
            )
        else:
            for j, (s, e) in enumerate(job_op_ranges):
                if e > s and starts[s] < release[j] - tol:
                    violations.append(
                        f"[release] job {j}: first op {s} starts at "
                        f"{starts[s]:.6g} before release {release[j]:.6g}"
                    )

    # ------------------------------------------------------------------
    # Check 5: Lower bound
    # ------------------------------------------------------------------
    lower_bound = 0.0
    for j, (s, e) in enumerate(job_op_ranges):
        job_min_pt_sum = 0.0
        job_lag_sum = 0.0
        for k in range(s, e):
            pos_pts = op_pt[k][op_pt[k] > 0]
            if len(pos_pts) == 0:
                # No compatible machine — incompatibility already flagged
                job_min_pt_sum += 0.0
            else:
                job_min_pt_sum += pos_pts.min()
            if k < e - 1:  # exclude last op's lag
                job_lag_sum += time_lag[k]
        lower_bound = max(lower_bound, job_min_pt_sum + job_lag_sum)

    makespan = float(op_ct.max())
    if makespan < lower_bound - tol:
        violations.append(
            f"[bound] makespan={makespan:.6g} < lower_bound={lower_bound:.6g} "
            f"(schedule is corrupt)"
        )

    return {
        "feasible": len(violations) == 0,
        "violations": violations,
        "makespan": makespan,
        "lower_bound": lower_bound,
    }


# ==========================================================================
# Self-test
# ==========================================================================

def _build_base_instance():
    """
    2 jobs: job0 has 3 ops, job1 has 2 ops. 3 machines.
    Global op indices: job0 -> ops 0,1,2  |  job1 -> ops 3,4

    op_pt (5 x 3):
         m0  m1  m2
    op0:  2   3   0    (m2 incompatible)
    op1:  0   4   2    (m0 incompatible)
    op2:  3   0   5    (m1 incompatible)
    op3:  1   3   0    (m2 incompatible)
    op4:  0   2   4    (m0 incompatible)

    time_lag: [1, 2, 0, 1, 0]
      op0 lag=1, op1 lag=2, op2 lag=0 (last in job0)
      op3 lag=1, op4 lag=0 (last in job1)
    """
    job_length = [3, 2]
    op_pt = np.array([
        [2, 3, 0],
        [0, 4, 2],
        [3, 0, 5],
        [1, 3, 0],
        [0, 2, 4],
    ], dtype=float)
    time_lag = np.array([1.0, 2.0, 0.0, 1.0, 0.0])
    return job_length, op_pt, time_lag


def _self_test():
    job_length, op_pt, time_lag = _build_base_instance()
    results = {}
    passed = True

    # ------------------------------------------------------------------
    # Case A: hand-verified FEASIBLE schedule
    # ------------------------------------------------------------------
    # Assignments: op0->m0, op1->m2, op2->m0, op3->m0, op4->m1
    # Timeline:
    #   op0: start=0, pt=2, ct=2; lag=1 -> successor op1 must start >= 3
    #   op3: start=2, pt=1, ct=3; lag=1 -> successor op4 must start >= 4
    #   op1: start=3, pt=2, ct=5; lag=2 -> successor op2 must start >= 7
    #   op4: start=4, pt=2, ct=6
    #   op2: start=7, pt=3, ct=10
    # Machine usage:
    #   m0: op0[0,2], op3[2,3], op2[7,10] — no overlap
    #   m2: op1[3,5]
    #   m1: op4[4,6]
    assigned_mch_A = np.array([0, 2, 0, 0, 1])
    op_ct_A = np.array([2.0, 5.0, 10.0, 3.0, 6.0])

    res_A = validate_schedule(job_length, op_pt, time_lag, assigned_mch_A, op_ct_A)
    results["A"] = res_A
    ok_A = res_A["feasible"] and len(res_A["violations"]) == 0
    print(f"Case A (feasible): {'PASS' if ok_A else 'FAIL'}")
    if not ok_A:
        passed = False
        for v in res_A["violations"]:
            print(f"  {v}")

    # ------------------------------------------------------------------
    # Case B: violate lag — op1 starts DURING op0's lag period
    # op0 ct=2, lag=1 -> op1 must start >= 3; corrupt: op1 starts at 2
    # ------------------------------------------------------------------
    assigned_mch_B = assigned_mch_A.copy()
    op_ct_B = op_ct_A.copy()
    op_ct_B[1] = 4.0   # op1: start=4-2=2, violates >=3
    # Need to also shift op2 to avoid cascading overlap on m0 (keep op2 start >=7 still fine)
    # Actually op2 start = op_ct_B[2] - pt[2,0] = 10-3=7 >= op_ct_B[1]+lag[1]=4+2=6 — still ok
    # But we want ONLY the lag violation. Let's just adjust op_ct[1] to introduce the lag violation.
    res_B = validate_schedule(job_length, op_pt, time_lag, assigned_mch_B, op_ct_B)
    results["B"] = res_B
    has_prec = any("[precedence]" in v for v in res_B["violations"])
    ok_B = (not res_B["feasible"]) and has_prec
    print(f"Case B (lag violation): {'PASS' if ok_B else 'FAIL'}")
    if not ok_B:
        passed = False
        print(f"  feasible={res_B['feasible']}, violations={res_B['violations']}")

    # ------------------------------------------------------------------
    # Case C: overlap two ops on the same machine
    # Move op3 to m0 with start=1 (overlaps op0 on m0 which runs [0,2])
    # op3: start=1, pt[3,0]=1, ct=2  -> overlaps op0[0,2]
    # ------------------------------------------------------------------
    assigned_mch_C = assigned_mch_A.copy()
    op_ct_C = op_ct_A.copy()
    assigned_mch_C[3] = 0   # op3 now on m0
    op_ct_C[3] = 2.0        # op3: start=2-1=1, overlaps op0[0,2]
    # op4 still on m1, start=4 >=op_ct[3]+lag[3]=2+1=3 — ok precedence
    res_C = validate_schedule(job_length, op_pt, time_lag, assigned_mch_C, op_ct_C)
    results["C"] = res_C
    has_overlap = any("[overlap]" in v for v in res_C["violations"])
    ok_C = (not res_C["feasible"]) and has_overlap
    print(f"Case C (machine overlap): {'PASS' if ok_C else 'FAIL'}")
    if not ok_C:
        passed = False
        print(f"  feasible={res_C['feasible']}, violations={res_C['violations']}")

    # ------------------------------------------------------------------
    # Case D: assign op0 to incompatible machine m2 (pt=0)
    # ------------------------------------------------------------------
    assigned_mch_D = assigned_mch_A.copy()
    op_ct_D = op_ct_A.copy()
    assigned_mch_D[0] = 2   # op0 on m2, pt[0,2]=0 => incompatible
    res_D = validate_schedule(job_length, op_pt, time_lag, assigned_mch_D, op_ct_D)
    results["D"] = res_D
    has_compat = any("[compat]" in v for v in res_D["violations"])
    ok_D = (not res_D["feasible"]) and has_compat
    print(f"Case D (incompatible machine): {'PASS' if ok_D else 'FAIL'}")
    if not ok_D:
        passed = False
        print(f"  feasible={res_D['feasible']}, violations={res_D['violations']}")

    print()
    if passed:
        print("ALL CASES PASSED")
    else:
        print("SOME CASES FAILED")
        sys.exit(1)


if __name__ == "__main__":
    _self_test()
