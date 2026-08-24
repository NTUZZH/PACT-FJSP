"""CP-SAT reference solver for the FJSP temporal-constraint family (this paper).

Extends the lab's lag-aware CP-SAT model (ortools_solver.fjsp_solver) to the
full family under the ATTACHED-setup convention used by FJSPEnvFamily:

  LAG     block(next) >= end(prev) + lag(prev)                (job chain)
  SETUP   per-machine AddCircuit over present alternatives; the arc a->b
          fixes setup_dur(b) = S[class(a), class(b)] (idle row for the
          machine's first op); processing start = block + setup_dur; the
          machine-side interval is the contiguous block [block, end].
  WINDOW  fixed dummy intervals inside each machine's NoOverlap: the block
          (setup + processing) can never overlap an outage, which also
          enforces no-straddle / no-preemption.

The model minimizes makespan (max of processing ends). It searches the FULL
schedule space (delays allowed), so its optimum lower-bounds every
environment rollout — the correct reference direction.
"""
import collections
import time

import numpy as np
from ortools.sat.python import cp_model

from constraint_family.descriptor import IDLE_CLASS


def matrix_to_jobs(job_length, op_pt):
    """[N,M] op_pt matrix -> OR-Tools jobs nested list (0 = incompatible)."""
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=int)
    jobs, first = [], 0
    for L in job_length:
        job = []
        for i in range(first, first + L):
            alts = [(int(op_pt[i, m]), m) for m in range(op_pt.shape[1])
                    if op_pt[i, m] > 0]
            job.append(alts)
        jobs.append(job)
        first += L
    return jobs


class _Trajectory(cp_model.CpSolverSolutionCallback):
    """Record every improving incumbent as [wall_s, objective, bound].

    One long solve then answers the whole budget ladder at once: the makespan
    the solver would have returned at any budget below its own is the last
    incumbent recorded before that instant. Running a separate solve per budget
    would cost the sum of the budgets and would additionally restart the search
    each time, so the ladder and the single trajectory are not even the same
    experiment. Attaching a callback changes nothing else about the solve.
    """

    def __init__(self, t0):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._t0 = t0
        self.points = []

    def on_solution_callback(self):
        self.points.append([round(time.time() - self._t0, 3),
                            float(self.ObjectiveValue()),
                            float(self.BestObjectiveBound())])


def family_solver(job_length, op_pt, desc, time_limit, warmstart=None,
                  num_workers=8, blocking=False, nowait=False, anytime=False):
    """Solve one family instance to (near-)optimality.

    :param desc: ConstraintDescriptor (true integer hours)
    :param warmstart: optional (assigned_mch [N], op_ct [N]) feasible schedule
        used as solution hints (same optimization, better first incumbent)
    :param blocking: (E4, out-of-family) each machine-side interval extends
        until the job's NEXT op starts (no buffer; hold during lag included);
        v1 supports blocking only with SETUP/WINDOW inactive
    :param nowait: (E4, out-of-family) job successor starts EXACTLY at
        predecessor end + lag (equality precedence)
    :return: dict(objective, bound, status, solve_time, assigned_mch, op_st,
                  op_ct) — op arrays flat in job-by-job precedence order
    """
    if blocking:
        assert not desc.has_setup and not desc.has_window, \
            'blocking reference v1 supports LAG only'
    if nowait:
        # attached setups force start > ready and window pushes do too, so
        # the equality precedence is only meaningful for the LAG-only case
        assert not desc.has_setup and not desc.has_window, \
            'no-wait reference v1 supports LAG only'
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=int)
    N, M = op_pt.shape
    jobs = matrix_to_jobs(job_length, op_pt)
    lag = desc.lag.astype(int) if desc.has_lag else np.zeros(N, dtype=int)
    S = None if not desc.has_setup else desc.setup.astype(int)
    windows = desc.windows if desc.has_window else None

    max_setup = 0 if S is None else int(S.max())
    total_outage = 0 if windows is None else int(
        sum(e - s for w in windows for (s, e) in w))
    max_op_pt = max((a[0] for job in jobs for task in job for a in task),
                    default=0)
    # Horizon must be a VALID upper bound. Under WINDOW an operation that does
    # not fit in the free interval before an outage is pushed to the outage
    # end, wasting up to (op duration) of idle PER OUTAGE on top of the outage
    # duration itself. The prior formula charged only total_outage and could
    # under-bound (fail-loud INFEASIBLE) on dense-outage / short-horizon /
    # long-op instances. Adding N*max_op_pt of window slack covers the
    # per-outage wasted-gap idle (there are at most N scheduled ops, hence at
    # most N distinct wasted gaps). Latent on the shipped PPVC benchmark
    # (lag-stretched horizons dwarf ~4 h ops) but required for correctness.
    window_slack = N * max_op_pt if windows is not None else 0
    horizon = int(sum(max(a[0] for a in task) for job in jobs for task in job)
                  + lag.sum() + N * max_setup + total_outage + window_slack)

    model = cp_model.CpModel()

    starts, blocks, ends = {}, {}, {}         # master vars per (job, task)
    presences = {}                            # (job, task, alt) -> literal
    setup_durs = {}                           # (job, task) -> IntVar
    alt_machine = {}                          # (job, task, alt) -> machine id
    # per machine: list of (job, task, alt, presence, l_block, l_end)
    per_mch = collections.defaultdict(list)

    op_idx = 0
    op_of_task = {}
    job_ends = []
    for j, job in enumerate(jobs):
        # pass 1: master vars + job chain (so pass 2 can reference the NEXT
        # task's start for blocking holds)
        prev_end, prev_lag = None, 0
        for t, task in enumerate(job):
            durs = [a[0] for a in task]
            sfx = f'_j{j}_t{t}'
            block = model.NewIntVar(0, horizon, 'block' + sfx)
            start = model.NewIntVar(0, horizon, 'start' + sfx)
            dur = model.NewIntVar(min(durs), max(durs), 'dur' + sfx)
            end = model.NewIntVar(0, horizon, 'end' + sfx)
            sd = model.NewIntVar(0, max_setup, 'setup' + sfx)
            model.Add(start == block + sd)
            model.Add(end == start + dur)
            starts[(j, t)], blocks[(j, t)], ends[(j, t)] = start, block, end
            setup_durs[(j, t)] = sd
            op_of_task[(j, t)] = op_idx

            # job chain with lags: the BLOCK (attached setup needs the job
            # present) waits for the predecessor's end + its lag; NOWAIT
            # (out-of-family) tightens the precedence to an equality
            if prev_end is not None:
                if nowait:
                    model.Add(block == prev_end + prev_lag)
                else:
                    model.Add(block >= prev_end + prev_lag)
            prev_end, prev_lag = end, int(lag[op_idx])
            op_idx += 1
        job_ends.append(prev_end)

        # pass 2: machine-side (optional) intervals; under BLOCKING the hold
        # extends until the job's next op STARTS (transfer instant)
        for t, task in enumerate(job):
            sfx = f'_j{j}_t{t}'
            block, dur, end = blocks[(j, t)], None, ends[(j, t)]
            hold_end = ends[(j, t)] if (not blocking or t == len(job) - 1) \
                else starts[(j, t + 1)]
            lits = []
            for a, (d, m) in enumerate(task):
                asfx = f'{sfx}_a{a}'
                p = model.NewBoolVar('p' + asfx)
                l_block = model.NewIntVar(0, horizon, 'lb' + asfx)
                l_end = model.NewIntVar(0, horizon, 'le' + asfx)
                l_bdur = model.NewIntVar(0, horizon, 'lbd' + asfx)
                l_int = model.NewOptionalIntervalVar(
                    l_block, l_bdur, l_end, p, 'li' + asfx)
                model.Add(l_block == block).OnlyEnforceIf(p)
                model.Add(l_end == hold_end).OnlyEnforceIf(p)
                model.Add(starts[(j, t)] + d == end).OnlyEnforceIf(p)
                lits.append(p)
                presences[(j, t, a)] = p
                alt_machine[(j, t, a)] = m
                per_mch[m].append((j, t, a, p, l_int))
            model.AddExactlyOne(lits)

    # machine capacity + outage calendars
    for m in range(M):
        intervals = [x[4] for x in per_mch[m]]
        if windows is not None:
            for k, (s, e) in enumerate(windows[m]):
                intervals.append(model.NewIntervalVar(
                    int(s), int(e - s), int(e), f'outage_m{m}_{k}'))
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # sequence-dependent ATTACHED setups: circuit per machine
    # arc-literal registries so a warm start can hint a COMPLETE solution
    # (hint completion of a bare start/presence hint fails on ~10-16% of
    # setup instances -> UNKNOWN with no incumbent)
    node_index = {}                 # m -> {(j, t): node position i}
    arc_lits = {}                   # m -> {(from_i, to_i): literal}
    if S is not None:
        op_class = desc.op_class.astype(int)
        for m in range(M):
            nodes = per_mch[m]
            if not nodes:
                continue
            node_index[m] = {}
            arc_lits[m] = {}
            arcs = []
            # empty-machine case: the circuit reduces to the 0 -> 0 self-loop
            empty = model.NewBoolVar(f'empty_m{m}')
            arcs.append((0, 0, empty))
            arc_lits[m][(0, 0)] = empty
            for i, (j, t, a, p, _) in enumerate(nodes):
                node_index[m][(j, t)] = i + 1
                # skip node iff not present on this machine
                arcs.append((i + 1, i + 1, p.Not()))
                cls = int(op_class[op_of_task[(j, t)]])
                # dummy start node 0 -> first op: idle (cold-start) setup
                first_lit = model.NewBoolVar(f'first_m{m}_{i}')
                arcs.append((0, i + 1, first_lit))
                arc_lits[m][(0, i + 1)] = first_lit
                model.Add(setup_durs[(j, t)] == int(S[IDLE_CLASS, cls])) \
                    .OnlyEnforceIf(first_lit)
                # last op -> dummy end (same node 0 closes the circuit)
                last_lit = model.NewBoolVar(f'last_m{m}_{i}')
                arcs.append((i + 1, 0, last_lit))
                arc_lits[m][(i + 1, 0)] = last_lit
                for i2, (j2, t2, a2, p2, _) in enumerate(nodes):
                    if i2 == i or (j2, t2) == (j, t):
                        continue
                    cls2 = int(op_class[op_of_task[(j2, t2)]])
                    lit = model.NewBoolVar(f'arc_m{m}_{i}_{i2}')
                    arcs.append((i + 1, i2 + 1, lit))
                    arc_lits[m][(i + 1, i2 + 1)] = lit
                    # consecutive on m: b's block starts after a ends, and
                    # b's setup is the class transition a -> b
                    model.Add(blocks[(j2, t2)] >= ends[(j, t)]) \
                        .OnlyEnforceIf(lit)
                    model.Add(setup_durs[(j2, t2)] == int(S[cls, cls2])) \
                        .OnlyEnforceIf(lit)
            model.AddCircuit(arcs)
    else:
        for (j, t) in setup_durs:
            model.Add(setup_durs[(j, t)] == 0)

    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    if warmstart is not None:
        # COMPLETE solution hint: presences, processing starts, and (when
        # setups are active) per-op setup durations, block starts, and the
        # full circuit arc selection reconstructed from the rollout's
        # per-machine completion order. Leaves nothing for hint completion
        # to search, so a feasible incumbent is available immediately.
        ws_mch, ws_ct = warmstart
        task_of_op, oi = {}, 0
        for j, job in enumerate(jobs):
            for t, task in enumerate(job):
                task_of_op[oi] = (j, t)
                oi += 1
        n_ops_total = oi

        # per-op durations on the assigned machine
        dur_of = {}
        for oi in range(n_ops_total):
            (j, t) = task_of_op[oi]
            for a, (d, mm) in enumerate(jobs[j][t]):
                model.AddHint(presences[(j, t, a)],
                              int(mm == int(ws_mch[oi])))
                if mm == int(ws_mch[oi]):
                    dur_of[oi] = d

        if S is not None:
            op_class = desc.op_class.astype(int)
            # machine sequences ordered by completion time
            seq = {m: [] for m in range(M)}
            for oi in range(n_ops_total):
                seq[int(ws_mch[oi])].append(oi)
            for m in range(M):
                seq[m].sort(key=lambda o: ws_ct[o])
                last_cls, prev_node = IDLE_CLASS, 0
                for o in seq[m]:
                    (j, t) = task_of_op[o]
                    sd = int(S[last_cls, op_class[o]])
                    st = int(ws_ct[o]) - dur_of[o]
                    model.AddHint(setup_durs[(j, t)], sd)
                    model.AddHint(starts[(j, t)], st)
                    model.AddHint(blocks[(j, t)], st - sd)
                    node = node_index[m][(j, t)]
                    if (prev_node, node) in arc_lits[m]:
                        model.AddHint(arc_lits[m][(prev_node, node)], 1)
                    last_cls, prev_node = int(op_class[o]), node
                # close the circuit (or mark the machine empty)
                if (prev_node, 0) in arc_lits.get(m, {}):
                    model.AddHint(arc_lits[m][(prev_node, 0)], 1)
        else:
            for oi in range(n_ops_total):
                (j, t) = task_of_op[oi]
                if oi in dur_of:
                    model.AddHint(starts[(j, t)],
                                  max(0, int(ws_ct[oi]) - dur_of[oi]))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = num_workers
    t0 = time.time()
    cb = _Trajectory(t0) if anytime else None
    status = solver.Solve(model, cb) if cb is not None else solver.Solve(model)
    dt = time.time() - t0

    out = {'status': solver.StatusName(status), 'solve_time': dt,
           'objective': None, 'bound': None,
           'assigned_mch': None, 'op_st': None, 'op_ct': None}
    if cb is not None:
        out['anytime'] = cb.points
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        amch = np.full(N, -1, dtype=int)
        ost = np.full(N, -1.0)
        oct_ = np.full(N, -1.0)
        for j, job in enumerate(jobs):
            for t, task in enumerate(job):
                oi = op_of_task[(j, t)]
                ost[oi] = solver.Value(starts[(j, t)])
                oct_[oi] = solver.Value(ends[(j, t)])
                for a in range(len(task)):
                    if solver.Value(presences[(j, t, a)]):
                        amch[oi] = alt_machine[(j, t, a)]
        search = (float(solver.ObjectiveValue()), amch, ost, oct_)
    else:
        search = None

    # AddHint is a search *suggestion*, not a solution: CP-SAT does not enter it
    # into its solution pool, so at a short budget it can return UNKNOWN, or an
    # incumbent worse than the complete feasible schedule it was handed. In both
    # cases that schedule is still in hand. The anytime value of a warm-started
    # solver is therefore min(hint, search) -- what a practitioner would actually
    # hold when the budget expires.
    hint = None
    if warmstart is not None:
        ws_amch, ws_ct = warmstart
        ws_amch = np.asarray(ws_amch, dtype=int)
        ws_ct = np.asarray(ws_ct, dtype=float)
        ws_st = np.array([ws_ct[oi] - op_pt[oi, ws_amch[oi]] if ws_amch[oi] >= 0
                          else -1.0 for oi in range(N)])
        hint = (float(np.max(ws_ct)), ws_amch, ws_st, ws_ct)

    best = min([c for c in (search, hint) if c is not None],
               key=lambda c: c[0], default=None)
    if best is not None:
        out['objective'], out['assigned_mch'], out['op_st'], out['op_ct'] = best
        out['bound'] = float(solver.BestObjectiveBound())
        out['from_warmstart'] = hint is not None and best is hint
    return out
