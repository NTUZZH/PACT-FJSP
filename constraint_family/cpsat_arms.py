"""Alternative CP-SAT arms for the FJSP timing-rule family.

Each arm takes one instance (a Case), a hint schedule, a wall budget and a
worker count, and returns the best schedule it holds when the budget ends,
together with an anytime list of [elapsed_s, makespan, bound or None].

Semantics shared by every arm (attached setups, as in FJSPEnvFamily and
cpsat_family.family_solver):
  * an operation o on machine m occupies m over the contiguous block
    [block_o, end_o] = [start_o - setup_o, start_o + p_om];
  * setup_o = S[class(pred), class(o)], where pred is the operation processed
    immediately before o on m, or the idle (cold-start) row for the first
    operation on m;
  * the block starts only when the job is present: block_o >= end of the
    job predecessor + its lag, and block_o >= release_j for a job's first
    operation when the descriptor carries release dates;
  * the block may not overlap an outage of m (no preemption).

Arms (ARMS below), with their exactness status:
  ref, ref_*     exact. The current model of cpsat_family.family_solver,
                 rebuilt verbatim by build_reference_model (the test suite
                 asserts that both produce the same model proto), solved with
                 the default parameters (ref) or a named parameter variant.
  compact        exact. Same circuit formulation with shared machine-side
                 intervals, one aggregated setup equality per (operation,
                 machine) instead of one per arc, arcs that contradict the
                 job order removed, and the horizon set to the hint makespan.
  compact_red    exact. compact plus a cumulative constraint per station
                 type (capacity = number of machines, outages as unit
                 demands).
  pairwise       restriction. Pairwise order literals with class-based
                 separation start_j >= end_i + S[c_i, c_j] for every pair on
                 a machine; job presence and outages are checked on a block
                 that reserves the largest setup of the operation's column.
  radius         restriction. compact with circuit arcs only between
                 operations whose hint block starts lie within R hours.
  fixassign      restriction. compact with every machine assignment fixed
                 to the hint's; sequences and times are free.
  lns            heuristic. Large-neighbourhood search around CP-SAT; every
                 sub-model keeps the exact rules, so every accepted schedule
                 is feasible for the full problem.

Times are integer hours. Operation arrays are flat in job-by-job order.
"""
import collections
import json
import os
import resource
import time

import numpy as np
from ortools.sat.python import cp_model

from constraint_family.cpsat_family import matrix_to_jobs, release_dates
from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS

# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------


def read_fjs(path):
    """Read a standard .fjs file into (job_length [J], op_pt [N, M]).

    Same parsing as data_utils.text_to_matrix, without importing the
    training configuration that module loads at import time."""
    import re
    with open(path) as f:
        text = f.readlines()
    nums = re.findall(r'\d+\.?\d*', text[0])
    n_j, n_m = int(nums[0]), int(nums[1])
    job_length = np.zeros(n_j, dtype=int)
    rows = []
    for i in range(n_j):
        c = [int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])]
        job_length[i] = c[0]
        idx = 1
        for _ in range(c[0]):
            k = c[idx]
            row = np.zeros(n_m, dtype=int)
            for a in range(k):
                row[c[idx + 2 * a + 1] - 1] = c[idx + 2 * a + 2]
            rows.append(row)
            idx += 2 * k + 1
    return job_length, np.array(rows, dtype=int)


class Case:
    """One instance under one descriptor, with the derived arrays the
    models and the validator need."""

    def __init__(self, job_length, op_pt, desc, name='', regime=''):
        self.jl = np.asarray(job_length, dtype=int)
        self.pt = np.asarray(op_pt, dtype=int)
        self.desc = desc
        self.name, self.regime = name, regime
        self.N, self.M = self.pt.shape
        self.J = len(self.jl)
        self.first = np.concatenate([[0], np.cumsum(self.jl)[:-1]]).astype(int)
        self.last = (self.first + self.jl - 1).astype(int)
        self.job = np.repeat(np.arange(self.J), self.jl)
        self.elig = [np.nonzero(self.pt[o])[0].tolist() for o in range(self.N)]
        self.lag = (np.asarray(desc.lag, dtype=int) if desc.has_lag
                    else np.zeros(self.N, dtype=int))
        self.cls = np.asarray(desc.op_class, dtype=int)
        self.S = (np.asarray(desc.setup, dtype=int) if desc.has_setup
                  else None)
        self.windows = ([[(int(s), int(e)) for (s, e) in w] for w in desc.windows]
                        if desc.has_window and desc.windows is not None else None)
        rel = getattr(desc, 'release', None)
        self.release = None if rel is None else np.asarray(rel, dtype=int)
        self.has_setup = self.S is not None

    def machine_groups(self):
        """Machine groups such that every operation's eligible set is exactly
        one group (station types on the PPVC benchmark), or None when the
        eligible sets do not partition the machines."""
        sets = {}
        for o in range(self.N):
            s = frozenset(self.elig[o])
            sets.setdefault(s, []).append(o)
        seen = set()
        for s in sets:
            if seen & s:
                return None
            seen |= s
        return [(sorted(s), ops) for s, ops in sets.items()]

    def setup_values(self, o):
        """Possible setup durations of operation o (its column of S)."""
        if self.S is None:
            return [0]
        return sorted(set(int(v) for v in self.S[:, self.cls[o]]))

    def triangle_ok(self):
        """S[a, c] <= S[a, b] + S[b, c] over the real classes and
        S[I, c] <= S[I, b] + S[b, c] for the idle row I."""
        if self.S is None:
            return True
        K = self.S.shape[1]
        rows = list(range(K)) + [IDLE_CLASS]
        return all(self.S[a, c] <= self.S[a, b] + self.S[b, c]
                   for a in rows for b in range(K) for c in range(K))


def load_case(size, regime, name, root='.'):
    """Load data/FAMILY/<size>/desc/<regime>/<name>.desc.json and its base
    instance. The benchmark generator draws attached setups, so the loader
    states that convention for records that do not carry the field."""
    dp = os.path.join(root, 'data', 'FAMILY', size, 'desc', regime,
                      f'{name}.desc.json')
    with open(dp) as f:
        d = json.load(f)
    stem = os.path.join(root, d['base_instance'])
    jl, pt = read_fjs(stem + '.fjs')
    return Case(jl, pt, ConstraintDescriptor.from_json(
        d, setup_convention='attached'), name, regime)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------


def schedule_from_hint(case, amch, ct):
    """(assigned machine, completion) -> integer (amch, st, ct), where st is
    the processing start."""
    amch = np.asarray(amch, dtype=int)
    ct = np.rint(np.asarray(ct, dtype=float)).astype(int)
    st = ct - case.pt[np.arange(case.N), amch]
    return amch, st, ct


def load_hint_npz(path, case):
    z = np.load(path, allow_pickle=True)
    amch, st, ct = schedule_from_hint(case, z['amch'], z['ct'])
    meta = {'path': path, 'makespan': float(z['ms']) if 'ms' in z.files
            else float(ct.max())}
    if 'rule' in z.files:
        meta['rule'] = str(z['rule'])
    if 'rule_names' in z.files and 'rule_ms' in z.files:
        meta['all_rules'] = {str(k): float(v)
                             for k, v in zip(z['rule_names'], z['rule_ms'])}
    return (amch, st, ct), meta


def sequences(case, amch, st):
    """Machine sequences ordered by processing start."""
    seq = [[] for _ in range(case.M)]
    for o in np.argsort(st, kind='stable'):
        seq[int(amch[o])].append(int(o))
    return seq


def derive(case, amch, st, ct):
    """Machine sequences, setup durations, block starts, predecessor and
    successor on the machine."""
    seq = sequences(case, amch, st)
    setup = np.zeros(case.N, dtype=int)
    prev = np.full(case.N, -1, dtype=int)
    nxt = np.full(case.N, -1, dtype=int)
    for m in range(case.M):
        last = IDLE_CLASS
        for i, o in enumerate(seq[m]):
            if case.S is not None:
                setup[o] = case.S[last, case.cls[o]]
            last = case.cls[o]
            if i:
                prev[o] = seq[m][i - 1]
                nxt[seq[m][i - 1]] = o
    return {'seq': seq, 'setup': setup, 'block': np.asarray(st) - setup,
            'prev': prev, 'next': nxt}


def validate_strict(case, amch, st, ct):
    """Feasibility of a schedule under the attached-setup semantics, coded
    independently of reference_sim.validate_schedule.

    The job must be present when the setup BEGINS (block >= job ready time
    and block >= release), as in FJSPEnvFamily and the CP-SAT models; the
    reference check applies the same condition to the block. This check also
    requires integer times. Returns a list of violation strings (empty =
    feasible)."""
    amch = np.asarray(amch)
    st = np.asarray(st)
    ct = np.asarray(ct)
    v = []
    if amch.shape != (case.N,) or st.shape != (case.N,) or ct.shape != (case.N,):
        return [f'shape mismatch {amch.shape} {st.shape} {ct.shape}']
    for arr, nm in ((st, 'start'), (ct, 'end')):
        if not np.all(np.asarray(arr, dtype=float) == np.rint(arr)):
            v.append(f'non-integer {nm} times')
    for o in range(case.N):
        m = int(amch[o])
        if not (0 <= m < case.M) or case.pt[o, m] <= 0:
            v.append(f'op {o}: machine {m} not eligible')
            continue
        if ct[o] - st[o] != case.pt[o, m]:
            v.append(f'op {o}: duration {ct[o] - st[o]} != {case.pt[o, m]}')
    if v:
        return v
    d = derive(case, amch, st, ct)
    block = d['block']
    for m in range(case.M):
        prev_end = 0
        for o in d['seq'][m]:
            if block[o] < prev_end:
                v.append(f'mch {m}: op {o} block {block[o]} before {prev_end}')
            if case.windows is not None:
                for (s, e) in case.windows[m]:
                    if block[o] < e and ct[o] > s:
                        v.append(f'mch {m}: op {o} block [{block[o]},{ct[o]}] '
                                 f'overlaps outage ({s},{e})')
            prev_end = ct[o]
    for j in range(case.J):
        f, L = case.first[j], case.jl[j]
        rho = 0 if case.release is None else int(case.release[j])
        if block[f] < rho:
            v.append(f'job {j}: block {block[f]} of op {f} before release {rho}')
        for o in range(f + 1, f + L):
            need = ct[o - 1] + case.lag[o - 1]
            if block[o] < need:
                v.append(f'job {j}: op {o} block {block[o]} before ready {need}')
    return v


def validate_all(case, amch, st, ct):
    """validate_strict plus the repository's C6 check (both independent
    validators)."""
    from constraint_family.checks import check_schedule_validation
    v = validate_strict(case, amch, st, ct)
    if v:
        return ['[strict] ' + x for x in v]
    r = check_schedule_validation(case.desc, case.jl, case.pt,
                                  np.asarray(amch), np.asarray(ct, dtype=float),
                                  op_st=np.asarray(st, dtype=float))
    return list(r.details.get('violations', []))


def greedy_schedule(case):
    """Earliest-completion list schedule under the exact rules (a simple
    feasible hint for small test instances)."""
    nxt = case.first.copy()
    ready = (np.zeros(case.J, dtype=int) if case.release is None
             else case.release.astype(int).copy())
    free = np.zeros(case.M, dtype=int)
    last = np.full(case.M, IDLE_CLASS, dtype=int)
    amch = np.full(case.N, -1, dtype=int)
    st = np.zeros(case.N, dtype=int)
    ct = np.zeros(case.N, dtype=int)
    for _ in range(case.N):
        best = None
        for j in range(case.J):
            o = nxt[j]
            if o > case.last[j]:
                continue
            for m in case.elig[o]:
                s = 0 if case.S is None else int(case.S[last[m], case.cls[o]])
                p = int(case.pt[o, m])
                b = max(int(ready[j]), int(free[m]))
                if case.windows is not None:
                    for (ws, we) in case.windows[m]:
                        if b < we and b + s + p > ws:
                            b = we
                key = (b + s + p, j, m)
                if best is None or key < best[0]:
                    best = (key, j, m, b, s, p)
        _, j, m, b, s, p = best
        o = nxt[j]
        amch[o], st[o], ct[o] = m, b + s, b + s + p
        ready[j] = ct[o] + case.lag[o]
        free[m] = ct[o]
        last[m] = case.cls[o]
        nxt[j] += 1
    return amch, st, ct


def total_completion(case, ct):
    return int(np.asarray(ct)[case.last].sum())


# ---------------------------------------------------------------------------
# Anytime callback
# ---------------------------------------------------------------------------


class Anytime(cp_model.CpSolverSolutionCallback):
    """Every improving incumbent as [elapsed_s, objective, bound]."""

    def __init__(self, t0, scale=1):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._t0 = t0
        self._scale = scale
        self.points = []

    def on_solution_callback(self):
        self.points.append([round(time.time() - self._t0, 3),
                            float(self.ObjectiveValue()) / self._scale,
                            float(self.BestObjectiveBound()) / self._scale])


def _apply_params(solver, budget, workers, params, seed=None, log=None):
    solver.parameters.max_time_in_seconds = float(budget)
    solver.parameters.num_search_workers = int(workers)
    if seed is not None:
        solver.parameters.random_seed = int(seed)
    for k, val in (params or {}).items():
        field = getattr(solver.parameters, k)
        if isinstance(val, (list, tuple)):
            del field[:]
            field.extend(val)
        else:
            setattr(solver.parameters, k, val)
    if log is not None:
        solver.parameters.log_search_progress = True
        solver.parameters.log_to_stdout = False
        solver.log_callback = log.append


# ---------------------------------------------------------------------------
# The current model, rebuilt verbatim
# ---------------------------------------------------------------------------


def build_reference_model(case, warmstart=None, complete_hint=False):
    """The model of cpsat_family.family_solver, statement for statement
    (default blocking=False, nowait=False), returned unsolved with the
    handles needed to read a solution. The proto equals family_solver's
    (asserted in tests/test_cpsat_arms.py), release dates included: when the
    descriptor carries them, the horizon grows by the latest release date and
    the constraint block(first op of job j) >= release_j is appended after
    the hints, as family_solver does.

    With complete_hint=True on a setup regime, the hint additionally covers
    every remaining variable (ends, durations, the alternatives' interval
    variables, every arc literal and the makespan), as family_solver already
    does without setups. CP-SAT then reports the hint complete and feasible
    instead of completing it by search; the model is unchanged."""
    desc = case.desc
    job_length, op_pt = case.jl, case.pt
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
    window_slack = N * max_op_pt if windows is not None else 0
    horizon = int(sum(max(a[0] for a in task) for job in jobs for task in job)
                  + lag.sum() + N * max_setup + total_outage + window_slack)
    release = release_dates(desc, len(jobs))
    if release is not None:
        horizon += int(release.max())

    model = cp_model.CpModel()
    starts, blocks, ends = {}, {}, {}
    dur_vars = {}
    alt_vars = {}
    presences = {}
    setup_durs = {}
    alt_machine = {}
    per_mch = collections.defaultdict(list)

    op_idx = 0
    op_of_task = {}
    job_ends = []
    for j, job in enumerate(jobs):
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
            dur_vars[(j, t)] = dur
            setup_durs[(j, t)] = sd
            op_of_task[(j, t)] = op_idx
            if prev_end is not None:
                model.Add(block >= prev_end + prev_lag)
            prev_end, prev_lag = end, int(lag[op_idx])
            op_idx += 1
        job_ends.append(prev_end)

        for t, task in enumerate(job):
            sfx = f'_j{j}_t{t}'
            block, end = blocks[(j, t)], ends[(j, t)]
            hold_end = ends[(j, t)]
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
                alt_vars[(j, t, a)] = (l_block, l_end, l_bdur)
                alt_machine[(j, t, a)] = m
                per_mch[m].append((j, t, a, p, l_int))
            model.AddExactlyOne(lits)

    for m in range(M):
        intervals = [x[4] for x in per_mch[m]]
        if windows is not None:
            for k, (s, e) in enumerate(windows[m]):
                intervals.append(model.NewIntervalVar(
                    int(s), int(e - s), int(e), f'outage_m{m}_{k}'))
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    node_index = {}
    arc_lits = {}
    if S is not None:
        op_class = desc.op_class.astype(int)
        for m in range(M):
            nodes = per_mch[m]
            if not nodes:
                continue
            node_index[m] = {}
            arc_lits[m] = {}
            arcs = []
            empty = model.NewBoolVar(f'empty_m{m}')
            arcs.append((0, 0, empty))
            arc_lits[m][(0, 0)] = empty
            for i, (j, t, a, p, _) in enumerate(nodes):
                node_index[m][(j, t)] = i + 1
                arcs.append((i + 1, i + 1, p.Not()))
                cls = int(op_class[op_of_task[(j, t)]])
                first_lit = model.NewBoolVar(f'first_m{m}_{i}')
                arcs.append((0, i + 1, first_lit))
                arc_lits[m][(0, i + 1)] = first_lit
                model.Add(setup_durs[(j, t)] == int(S[IDLE_CLASS, cls])) \
                    .OnlyEnforceIf(first_lit)
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
        ws_mch, ws_ct = warmstart
        task_of_op, oi = {}, 0
        for j, job in enumerate(jobs):
            for t, task in enumerate(job):
                task_of_op[oi] = (j, t)
                oi += 1
        n_ops_total = oi
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
            on_arcs, blk_of, st_of2 = set(), {}, {}
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
                        on_arcs.add((m, prev_node, node))
                    blk_of[(j, t)], st_of2[(j, t)] = st - sd, st
                    last_cls, prev_node = int(op_class[o]), node
                if (prev_node, 0) in arc_lits.get(m, {}):
                    model.AddHint(arc_lits[m][(prev_node, 0)], 1)
                    on_arcs.add((m, prev_node, 0))
            if complete_hint:
                end_of2 = {}
                for oi in range(n_ops_total):
                    (j, t) = task_of_op[oi]
                    end_of2[(j, t)] = st_of2[(j, t)] + dur_of[oi]
                    model.AddHint(ends[(j, t)], end_of2[(j, t)])
                    model.AddHint(dur_vars[(j, t)], dur_of[oi])
                for (j, t, a), (l_b, l_e, l_d) in alt_vars.items():
                    oi = op_of_task[(j, t)]
                    if jobs[j][t][a][1] == int(ws_mch[oi]):
                        vals = (blk_of[(j, t)], end_of2[(j, t)],
                                end_of2[(j, t)] - blk_of[(j, t)])
                    else:
                        vals = (0, 0, 0)
                    for var, v in zip((l_b, l_e, l_d), vals):
                        model.AddHint(var, v)
                for m, lits in arc_lits.items():
                    for (u, v), lit in lits.items():
                        if (m, u, v) in on_arcs:
                            continue
                        if (u, v) == (0, 0):
                            model.AddHint(lit, int(not seq[m]))
                        else:
                            model.AddHint(lit, 0)
                model.AddHint(makespan, max(end_of2.values()))
        else:
            st_of, end_of = {}, {}
            for oi in range(n_ops_total):
                (j, t) = task_of_op[oi]
                st = max(0, int(ws_ct[oi]) - dur_of[oi])
                st_of[(j, t)], end_of[(j, t)] = st, st + dur_of[oi]
            for (j, t), st in st_of.items():
                model.AddHint(starts[(j, t)], st)
                model.AddHint(blocks[(j, t)], st)
                model.AddHint(setup_durs[(j, t)], 0)
                model.AddHint(dur_vars[(j, t)], end_of[(j, t)] - st)
                model.AddHint(ends[(j, t)], end_of[(j, t)])
            for (j, t, a), (l_b, l_e, l_d) in alt_vars.items():
                oi = op_of_task[(j, t)]
                if jobs[j][t][a][1] == int(ws_mch[oi]):
                    hold = end_of[(j, t)]
                    vals = (st_of[(j, t)], hold, hold - st_of[(j, t)])
                else:
                    vals = (0, 0, 0)
                for var, v in zip((l_b, l_e, l_d), vals):
                    model.AddHint(var, v)
            model.AddHint(makespan, max(end_of.values()))

    if release is not None:
        for j, job in enumerate(jobs):
            if job:
                model.Add(blocks[(j, 0)] >= int(release[j]))

    def extract(solver):
        amch = np.full(N, -1, dtype=int)
        st = np.zeros(N, dtype=int)
        ct = np.zeros(N, dtype=int)
        for j, job in enumerate(jobs):
            for t, task in enumerate(job):
                oi = op_of_task[(j, t)]
                st[oi] = solver.Value(starts[(j, t)])
                ct[oi] = solver.Value(ends[(j, t)])
                for a in range(len(task)):
                    if solver.Value(presences[(j, t, a)]):
                        amch[oi] = alt_machine[(j, t, a)]
        return amch, st, ct

    return {'model': model, 'extract': extract, 'horizon': horizon}


# ---------------------------------------------------------------------------
# General model builder (full model, restrictions and LNS sub-models)
# ---------------------------------------------------------------------------


class Built:
    """A built CP-SAT model and the handles to hint it and read it back.

    Time handles (blk, st, en, su, du, bl) are IntVars, or Python ints for
    frozen operations and constant setups/durations."""

    def __init__(self, case):
        self.case = case
        self.model = cp_model.CpModel()
        self.blk, self.st, self.en = {}, {}, {}
        self.su, self.du, self.bl = {}, {}, {}
        self.allowed = {}
        self.frozen = set()
        self.pres = {}                  # (o, m) -> literal (multi-machine ops)
        self.arcs = {}                  # m -> {(u, v): literal}, op ids, -1 = depot
        self.empty = {}                 # m -> literal of the empty-machine loop
        self.pairs = {}                 # m -> {(i, j): literal i before j}
        self.mk = None
        self.weight = 1                 # objective = weight * makespan (+ secondary)
        self.n_arcs = 0
        self._hinted = set()

    def _hint(self, var, val):
        # constants and affine expressions follow from the hinted variables;
        # an expression such as blk + 0 is the variable itself, hinted once
        if not isinstance(var, cp_model.IntVar) or var.Index() in self._hinted:
            return
        self._hinted.add(var.Index())
        self.model.AddHint(var, int(val))

    def add_hint(self, amch, st, ct):
        """Hint every variable with the values of a complete schedule."""
        case = self.case
        d = derive(case, amch, st, ct)
        setup = d['setup']
        for o in range(case.N):
            if o in self.frozen:
                continue
            m = int(amch[o])
            # a constant setup (none, or the reserved maximum of the pairwise
            # form) fixes the block start; otherwise it follows the sequence
            su = self.su[o] if isinstance(self.su[o], int) else int(setup[o])
            self._hint(self.blk[o], st[o] - su)
            self._hint(self.st[o], st[o])
            self._hint(self.en[o], ct[o])
            self._hint(self.su[o], su)
            self._hint(self.du[o], ct[o] - st[o])
            self._hint(self.bl[o], ct[o] - st[o] + su)
            for mm in self.allowed[o]:
                if (o, mm) in self.pres:
                    self._hint(self.pres[(o, mm)], int(mm == m))
        for m, arcs in self.arcs.items():
            seq = [o for o in d['seq'][m] if o not in self.frozen]
            on = set()
            if seq:
                on.add((-1, seq[0]))
                on.add((seq[-1], -1))
                for a, c in zip(seq[:-1], seq[1:]):
                    on.add((a, c))
            for key, lit in arcs.items():
                self._hint(lit, int(key in on))
            if m in self.empty:
                self._hint(self.empty[m], int(not seq))
        for m, pairs in self.pairs.items():
            pos = {o: i for i, o in enumerate(d['seq'][m])}
            for (i, j), lit in pairs.items():
                self._hint(lit, int(i in pos and j in pos and pos[i] < pos[j]))
        if self.mk is not None:
            self._hint(self.mk, int(np.max(ct)))

    def extract(self, solver):
        case = self.case
        amch = np.full(case.N, -1, dtype=int)
        st = np.zeros(case.N, dtype=int)
        ct = np.zeros(case.N, dtype=int)
        for o in range(case.N):
            st[o] = solver.Value(self.st[o])
            ct[o] = solver.Value(self.en[o])
            al = self.allowed[o]
            if len(al) == 1:
                amch[o] = al[0]
            else:
                for m in al:
                    if solver.Value(self.pres[(o, m)]):
                        amch[o] = m
        return amch, st, ct


def build_model(case, *, base=None, free=None, reseq=False, frozen=(),
                horizon=None, setup_form='circuit', arc_ok=None,
                insert_radius=None, redundant=False, secondary=False):
    """Build an exact model of the family, or a restriction of it.

    :param base: (amch, st, ct) current schedule; required when free is not
        None, reseq is True or frozen is not empty.
    :param free: operations that may change machine and position; None = all.
        Every other operation keeps its base machine and its base order
        among the non-free operations of that machine; its times stay free.
    :param reseq: free operations keep their base machine (re-sequencing).
    :param frozen: non-free operations kept exactly as in the base schedule.
        They must form a prefix of every machine's base sequence; they leave
        the model, and each machine's depot then carries the class and end
        time of its last frozen operation.
    :param horizon: upper bound on every time variable (and the makespan).
    :param setup_form: 'circuit' (exact) or 'pairwise' (restriction, full
        model only).
    :param arc_ok: optional predicate (m, u, v) -> bool on circuit arcs
        between two free operations; the arcs of the base schedule must pass.
    :param insert_radius: when set, a free operation may be placed next to a
        non-free operation only if their base processing starts differ by at
        most this many hours (base neighbours are always allowed).
    :param redundant: add a cumulative constraint per machine group.
    :param secondary: minimise weight * makespan + sum of job completions,
        so that equal-makespan schedules are ranked by total completion.
    """
    N, M = case.N, case.M
    assert horizon is not None, 'a horizon is required'
    H = int(horizon)
    allfree = free is None
    free = set(range(N)) if allfree else set(int(o) for o in free)
    frozen = set(int(o) for o in frozen) - free
    if not allfree or reseq or frozen:
        assert base is not None, 'a base schedule is required'
    if setup_form == 'pairwise':
        assert allfree and not reseq, 'pairwise form is a full-model option'
    b = Built(case)
    b.frozen = frozen
    model = b.model
    S = case.S
    d = None
    if base is not None:
        bm, bst, bct = (np.asarray(x, dtype=int) for x in base)
        d = derive(case, bm, bst, bct)
    z = {}                                      # m -> last frozen op on m
    if frozen:
        for m in range(M):
            later = False
            for o in d['seq'][m]:
                if o in frozen:
                    assert not later, 'frozen operations must form a prefix'
                    z[m] = o
                else:
                    later = True

    # setups that the fixed order determines: non-free operations on a
    # machine that no free operation can reach
    const_su = {}
    if S is not None and setup_form == 'circuit' and base is not None \
            and not allfree:
        reach = set()
        for o in free:
            reach.update([int(bm[o])] if reseq else case.elig[o])
        for m in range(M):
            if m in reach:
                continue
            prev_c = IDLE_CLASS
            for o in d['seq'][m]:
                if o not in frozen:
                    const_su[o] = int(S[prev_c, case.cls[o]])
                prev_c = int(case.cls[o])

    # operation variables ---------------------------------------------------
    for o in range(N):
        if o in free and not reseq:
            al = list(case.elig[o])
        else:
            al = [int(bm[o])]
        b.allowed[o] = al
        if o in frozen:
            s = int(d['setup'][o])
            b.st[o], b.en[o], b.su[o] = int(bst[o]), int(bct[o]), s
            b.blk[o], b.du[o] = int(bst[o]) - s, int(bct[o] - bst[o])
            b.bl[o] = int(bct[o]) - b.blk[o]
            continue
        ps = [int(case.pt[o, m]) for m in al]
        lo = 0
        if case.release is not None and o == case.first[case.job[o]]:
            lo = int(case.release[case.job[o]])
        blk = model.NewIntVar(lo, H, f'b{o}')
        if (S is None or o in const_su) and len(set(ps)) == 1:
            # setup and duration are constants: start and end are affine
            # expressions of the block start
            su = 0 if S is None else const_su[o]
            b.blk[o], b.su[o], b.du[o], b.bl[o] = blk, su, ps[0], su + ps[0]
            b.st[o], b.en[o] = blk + su, blk + (su + ps[0])
            continue
        st = model.NewIntVar(lo, H, f's{o}')
        en = model.NewIntVar(lo, H, f'e{o}')
        if S is None:
            su = 0
        elif o in const_su:
            su = const_su[o]
        elif setup_form == 'pairwise':
            su = int(S[:, case.cls[o]].max())
        else:
            su = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(case.setup_values(o)), f'u{o}')
        if len(set(ps)) == 1:
            du = ps[0]
        else:
            du = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(sorted(set(ps))), f'd{o}')
        if isinstance(su, int) and isinstance(du, int):
            bl = su + du
        else:
            su_vals = [su] if isinstance(su, int) else case.setup_values(o)
            bl = model.NewIntVar(min(su_vals) + min(ps), max(su_vals) + max(ps),
                                 f'l{o}')
            model.Add(bl == su + du)
        model.Add(st == blk + su)
        model.Add(en == st + du)
        b.blk[o], b.st[o], b.en[o], b.su[o], b.du[o], b.bl[o] = \
            blk, st, en, su, du, bl

    # job chains --------------------------------------------------------------
    for j in range(case.J):
        for o in range(case.first[j] + 1, case.last[j] + 1):
            if o in frozen and o - 1 in frozen:
                continue
            model.Add(b.blk[o] >= b.en[o - 1] + int(case.lag[o - 1]))

    # machine-side intervals --------------------------------------------------
    per_m = collections.defaultdict(list)      # m -> [(o, presence or None, interval)]
    for o in range(N):
        if o in frozen:
            continue
        al = b.allowed[o]
        if len(al) == 1:
            itv = model.NewIntervalVar(b.blk[o], b.bl[o], b.en[o], f'i{o}')
            per_m[al[0]].append((o, None, itv))
        else:
            lits = []
            for m in al:
                p = model.NewBoolVar(f'p{o}_{m}')
                itv = model.NewOptionalIntervalVar(b.blk[o], b.bl[o], b.en[o],
                                                   p, f'i{o}_{m}')
                if not isinstance(b.du[o], int):
                    model.Add(b.du[o] == int(case.pt[o, m])).OnlyEnforceIf(p)
                b.pres[(o, m)] = p
                lits.append(p)
                per_m[m].append((o, p, itv))
            model.AddExactlyOne(lits)

    # every operation placed on m after its frozen prefix starts after it
    for m, zo in z.items():
        for (o, p, _) in per_m[m]:
            c = model.Add(b.blk[o] >= int(bct[zo]))
            if p is not None:
                c.OnlyEnforceIf(p)

    outage_itv = collections.defaultdict(list)
    if case.windows is not None:
        for m in range(M):
            t_free = int(bct[z[m]]) if m in z else 0
            for k, (s, e) in enumerate(case.windows[m]):
                if s >= H or e <= t_free:
                    continue
                outage_itv[m].append(model.NewIntervalVar(
                    int(s), int(e - s), int(e), f'w{m}_{k}'))

    for m in range(M):
        members = per_m[m]
        if setup_form == 'pairwise' and S is not None:
            # processing intervals are disjoint (the pairwise separation is
            # added below); each reserved block [start - max setup, end] must
            # avoid the outages, but reserved blocks may overlap each other
            proc = []
            for (o, p, itv) in members:
                du = b.du[o] if p is None else int(case.pt[o, m])
                if p is None:
                    proc.append(model.NewIntervalVar(b.st[o], du, b.en[o], f'q{o}'))
                else:
                    proc.append(model.NewOptionalIntervalVar(
                        b.st[o], du, b.en[o], p, f'q{o}_{m}'))
                if outage_itv[m]:
                    model.AddNoOverlap([itv] + outage_itv[m])
            if len(proc) > 1:
                model.AddNoOverlap(proc)
        else:
            itvs = [x[2] for x in members] + outage_itv[m]
            if len(itvs) > 1:
                model.AddNoOverlap(itvs)

    # order of the non-free operations ---------------------------------------
    fixed_seq = {m: [] for m in range(M)}
    if base is not None and (not allfree or reseq):
        for m in range(M):
            fixed_seq[m] = [o for o in d['seq'][m]
                            if o not in free and o not in frozen]
            for a, c in zip(fixed_seq[m][:-1], fixed_seq[m][1:]):
                model.Add(b.blk[c] >= b.en[a])

    near = None
    if insert_radius is not None and base is not None:
        base_pairs = set()
        for m in range(M):
            for a, c in zip(d['seq'][m][:-1], d['seq'][m][1:]):
                base_pairs.add((a, c))
        R = int(insert_radius)

        def near(u, v):
            return abs(int(bst[u]) - int(bst[v])) <= R or (u, v) in base_pairs

    # setups ------------------------------------------------------------------
    if S is not None and setup_form == 'circuit':
        for m in range(M):
            fx = fixed_seq[m]
            fr = [o for (o, p, _) in per_m[m] if o in free]
            c0 = IDLE_CLASS if m not in z else int(case.cls[z[m]])
            if not fr:
                # nothing can be inserted: the setups follow the fixed order
                prev_c = c0
                for a in fx:
                    if not isinstance(b.su[a], int):
                        model.Add(b.su[a] == int(S[prev_c, case.cls[a]]))
                    prev_c = case.cls[a]
                continue
            _circuit(b, m, fx, fr, per_m[m], arc_ok, c0, near)
    elif S is not None and setup_form == 'pairwise':
        for m in range(M):
            mem = per_m[m]
            b.pairs[m] = {}
            for x in range(len(mem)):
                i, pi, _ = mem[x]
                for y in range(x + 1, len(mem)):
                    j, pj, _ = mem[y]
                    enf = [q for q in (pi, pj) if q is not None]
                    if case.job[i] == case.job[j]:
                        # the job order fixes the machine order
                        a, c = (i, j) if i < j else (j, i)
                        model.Add(b.st[c] >= b.en[a] + int(S[case.cls[a], case.cls[c]])) \
                            .OnlyEnforceIf(enf)
                        continue
                    lit = model.NewBoolVar(f'o{m}_{i}_{j}')
                    b.pairs[m][(i, j)] = lit
                    model.Add(b.st[j] >= b.en[i] + int(S[case.cls[i], case.cls[j]])) \
                        .OnlyEnforceIf([lit] + enf)
                    model.Add(b.st[i] >= b.en[j] + int(S[case.cls[j], case.cls[i]])) \
                        .OnlyEnforceIf([lit.Not()] + enf)

    # redundant capacity per machine group -------------------------------------
    if redundant:
        groups = case.machine_groups()
        if groups is not None:
            for ms, ops in groups:
                ops = [o for o in ops if o not in frozen]
                if len(ms) < 2 or not ops:
                    continue
                itv = [model.NewIntervalVar(b.blk[o], b.bl[o], b.en[o], f'c{o}')
                       for o in ops]
                dem = [1] * len(itv)
                for m in ms:
                    itv += outage_itv[m]
                    dem += [1] * len(outage_itv[m])
                model.AddCumulative(itv, dem, len(ms))

    # objective -------------------------------------------------------------
    ends = [b.en[int(o)] for o in case.last]
    mk = model.NewIntVar(0, H, 'makespan')
    model.AddMaxEquality(mk, ends)
    b.mk = mk
    if secondary:
        b.weight = case.J * H + 1
        model.Minimize(b.weight * mk + sum(ends))
    else:
        model.Minimize(mk)
    return b


def _circuit(b, m, fx, fr, members, arc_ok, c0, near):
    """AddCircuit on machine m. fx: non-free operations in their fixed order
    (mandatory nodes); fr: free operations that may sit on m (optional nodes
    unless m is their only machine). Arc (u, v) means v is processed
    immediately after u on m; -1 is the depot, whose class is c0 (the idle
    row, or the class of the machine's last frozen operation)."""
    case, model = b.case, b.model
    S = case.S
    node = {}
    for o in fx + fr:
        node[o] = len(node) + 1
    pres = {o: p for (o, p, _) in members}
    arcs = []
    lits = {}
    into = collections.defaultdict(list)          # v -> [(u, literal)]

    def arc(u, v):
        lit = model.NewBoolVar('')
        lits[(u, v)] = lit
        arcs.append((0 if u < 0 else node[u], 0 if v < 0 else node[v], lit))
        if v >= 0:
            into[v].append((u, lit))
            if u >= 0:
                model.Add(b.blk[v] >= b.en[u]).OnlyEnforceIf(lit)

    if not fx:
        b.empty[m] = model.NewBoolVar('')
        arcs.append((0, 0, b.empty[m]))
    for o in fr:
        if pres[o] is not None:
            arcs.append((node[o], node[o], pres[o].Not()))

    def ok(u, v):
        # v may follow u only if v does not precede u in their common job
        if u >= 0 and v >= 0 and case.job[u] == case.job[v] and v < u:
            return False
        return True

    if fx:
        arc(-1, fx[0])
        for a, c in zip(fx[:-1], fx[1:]):
            arc(a, c)
        arc(fx[-1], -1)
        for f in fr:
            for a in fx:
                if near is not None and not (near(a, f) or near(f, a)):
                    continue
                if ok(a, f):
                    arc(a, f)
                if ok(f, a):
                    arc(f, a)
    for f in fr:
        arc(-1, f)
        arc(f, -1)
        for g in fr:
            if g != f and ok(f, g) and (arc_ok is None or arc_ok(m, f, g)):
                arc(f, g)
    b.n_arcs += len(lits)
    # one aggregated setup equality per (operation, machine): exactly one
    # incoming arc is active when the operation is on m, so its setup is the
    # class transition carried by that arc
    for v, ins in into.items():
        cv = case.cls[v]
        expr = cp_model.LinearExpr.WeightedSum(
            [lit for (_, lit) in ins],
            [int(S[c0 if u < 0 else case.cls[u], cv]) for (u, _) in ins])
        if pres[v] is None:
            model.Add(b.su[v] == expr)
        else:
            model.Add(b.su[v] == expr).OnlyEnforceIf(pres[v])
    model.AddCircuit(arcs)
    b.arcs[m] = lits


# ---------------------------------------------------------------------------
# Solving a built model
# ---------------------------------------------------------------------------


def _status_name(solver, status):
    return solver.StatusName(status)


def solve_built(model, extract, budget, workers, params=None, seed=None,
                log=None, scale=1, t0=None, anytime=True):
    solver = cp_model.CpSolver()
    _apply_params(solver, budget, workers, params, seed, log)
    t_solve = time.time()
    cb = Anytime(t_solve if t0 is None else t0, scale) if anytime else None
    status = solver.Solve(model, cb) if cb is not None else solver.Solve(model)
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError(f'model invalid: {model.Validate()[:300]}')
    out = {'status': solver.StatusName(status),
           'solve_s': round(time.time() - t_solve, 3),
           'anytime': cb.points if cb is not None else [],
           'objective': None, 'bound': None, 'schedule': None}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out['schedule'] = extract(solver)
        out['objective'] = float(solver.ObjectiveValue()) / scale
        out['bound'] = float(solver.BestObjectiveBound()) / scale
    else:
        try:
            out['bound'] = float(solver.BestObjectiveBound()) / scale
        except Exception:
            out['bound'] = None
    return out


def _credit(case, hint, res):
    """min(hint, search), as in family_solver: the hint is a schedule in
    hand even when the search returns nothing better."""
    hint_ms = int(np.max(hint[2]))
    sched = res['schedule']
    if sched is not None and int(np.max(sched[2])) <= hint_ms:
        return sched, False
    return hint, True


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

# Parameter variants of the current model. Every name was checked against
# the installed sat_parameters.proto (OR-Tools 9.14).
REF_VARIANTS = {
    'ref': {},
    # presolve is single-threaded and takes 35 s (50x25) to 150 s (80x25)
    # on the setup models; cut probing, symmetry detection and iterations
    'ref_presolve_light': {'cp_model_probing_level': 0, 'symmetry_level': 0,
                           'max_presolve_iterations': 1,
                           'presolve_probing_deterministic_time_limit': 1.0},
    'ref_nopresolve': {'cp_model_presolve': False},
    # every worker on neighbourhood search around the incumbent
    'ref_lnsonly': {'use_lns_only': True},
    # fewer full-search workers, more interleaved LNS workers
    'ref_lnsfocus': {'num_full_subsolvers': 2},
    'ref_lin2': {'linearization_level': 2},
    'ref_repair': {'repair_hint': True, 'hint_conflict_limit': 1000},
}

ARMS = {name: {'kind': 'ref', 'exactness': 'exact', 'params': p}
        for name, p in REF_VARIANTS.items()}
# each variant again with every variable hinted (build_reference_model with
# complete_hint=True): same model, same parameters, complete hint
for _name, _p in REF_VARIANTS.items():
    ARMS[f'{_name}_fullhint'] = {'kind': 'ref', 'exactness': 'exact',
                                 'params': _p, 'complete_hint': True}
ARMS.update({
    'compact': {'kind': 'full', 'exactness': 'exact'},
    'compact_red': {'kind': 'full', 'exactness': 'exact', 'redundant': True},
    'pairwise': {'kind': 'full', 'exactness': 'restriction',
                 'setup_form': 'pairwise'},
    'radius': {'kind': 'full', 'exactness': 'restriction', 'radius': 24},
    'fixassign': {'kind': 'full', 'exactness': 'restriction', 'reseq': True},
    'lns': {'kind': 'lns', 'exactness': 'heuristic'},
})


def _cpu_now():
    return time.process_time()


def run_arm(name, case, hint, budget, workers, *, seed=None, log=None,
            lns_options=None, params=None):
    """Run arm `name` on `case` from the complete feasible schedule `hint`
    (amch, st, ct) for `budget` seconds with `workers` CP-SAT workers.

    Returns a dict: objective (makespan of the returned schedule), bound
    (CP-SAT bound of the arm's own model, None when it has none), status,
    schedule (amch, st, ct), anytime, from_warmstart, build_s, solve_s,
    exactness and arm-specific statistics. For the ref and full-model arms
    the budget is CP-SAT's time limit and model building is reported
    separately (as in family_solver); for lns the budget covers everything."""
    spec = ARMS[name]
    hint = tuple(np.asarray(x, dtype=int) for x in hint)
    hint_ms = int(np.max(hint[2]))
    t0 = time.time()
    c0 = time.process_time()
    c_build = None
    if spec['kind'] == 'ref':
        p = dict(spec['params'])
        p.update(params or {})
        built = build_reference_model(case, warmstart=(hint[0], hint[2]),
                                      complete_hint=spec.get('complete_hint',
                                                             False))
        build_s = time.time() - t0
        c_build = time.process_time() - c0
        res = solve_built(built['model'], built['extract'], budget, workers,
                          params=p, seed=seed, log=log)
        sched, from_ws = _credit(case, hint, res)
        out = {'status': res['status'], 'bound': res['bound'],
               'anytime': res['anytime'], 'solve_s': res['solve_s'],
               'params': p}
    elif spec['kind'] == 'full':
        arc_ok = None
        if spec.get('radius'):
            arc_ok = _radius_filter(case, hint, spec['radius'])
        built = build_model(case, base=hint, horizon=hint_ms,
                            setup_form=spec.get('setup_form', 'circuit'),
                            reseq=spec.get('reseq', False), arc_ok=arc_ok,
                            redundant=spec.get('redundant', False))
        built.add_hint(*hint)
        build_s = time.time() - t0
        c_build = time.process_time() - c0
        res = solve_built(built.model, built.extract, budget, workers,
                          params=params, seed=seed, log=log)
        sched, from_ws = _credit(case, hint, res)
        out = {'status': res['status'], 'bound': res['bound'],
               'anytime': res['anytime'], 'solve_s': res['solve_s'],
               'params': dict(params or {}), 'n_arcs': built.n_arcs}
    elif spec['kind'] == 'lns':
        r = lns(case, hint, budget, workers, seed=seed,
                **(lns_options or {}))
        build_s = r['build_s']
        c_build = r['build_s']          # single-threaded: CPU = wall
        sched = r['schedule']
        from_ws = int(np.max(sched[2])) == hint_ms
        out = {'status': 'FEASIBLE', 'bound': None, 'anytime': r['anytime'],
               'solve_s': r['wall_s'], 'params': r['options'],
               'lns_stats': r['stats']}
    else:
        raise ValueError(name)
    # a restriction's bound bounds the restricted model only
    out['model_bound'] = out['bound']
    if spec['exactness'] != 'exact':
        out['bound'] = None
    cpu = time.process_time() - c0
    out.update({'arm': name, 'exactness': spec['exactness'],
                'schedule': sched, 'objective': float(np.max(sched[2])),
                'from_warmstart': bool(from_ws),
                'hint_makespan': float(hint_ms),
                'build_s': round(build_s, 3),
                'cpu_s': round(cpu, 2),
                'cpu_build_s': None if c_build is None else round(c_build, 2),
                'wall_s': round(time.time() - t0, 3)})
    return out


def _radius_filter(case, hint, R):
    """Arcs between two operations whose hint block starts differ by at most
    R hours, every depot arc, and every arc of the hint schedule."""
    d = derive(case, *hint)
    block = d['block']
    on = set()
    for m, seq in enumerate(d['seq']):
        for a, c in zip(seq[:-1], seq[1:]):
            on.add((m, a, c))

    def ok(m, u, v):
        if u < 0 or v < 0:
            return True
        return abs(int(block[u]) - int(block[v])) <= R or (m, u, v) in on
    return ok


# ---------------------------------------------------------------------------
# Large-neighbourhood search
# ---------------------------------------------------------------------------


def critical_ops(case, amch, st, ct, d=None):
    """Operations on a chain of tight constraints that ends at the makespan.

    An edge u -> o is tight when o's block starts exactly when u's
    constraint lets it: the job predecessor's end plus lag, or the machine
    predecessor's end. Every operation reached backwards from an operation
    that ends at the makespan is critical."""
    if d is None:
        d = derive(case, amch, st, ct)
    block, prev = d['block'], d['prev']
    ms = int(np.max(ct))
    crit = set()
    stack = [int(o) for o in np.nonzero(np.asarray(ct) == ms)[0]]
    while stack:
        o = stack.pop()
        if o in crit:
            continue
        crit.add(o)
        if o != case.first[case.job[o]]:
            p = o - 1
            if ct[p] + case.lag[p] == block[o]:
                stack.append(p)
        q = prev[o]
        if q >= 0 and ct[q] == block[o]:
            stack.append(int(q))
    return crit


NEIGHBOURHOODS = ('window', 'critical', 'station', 'jobs', 'reseq',
                  'station_reseq')


def _slice(ops, st, k, rng):
    """k operations consecutive in processing-start order, from a random
    position."""
    ops = sorted(ops, key=lambda o: st[o])
    if len(ops) <= k:
        return [int(o) for o in ops]
    i0 = int(rng.integers(0, len(ops) - k + 1))
    return [int(o) for o in ops[i0:i0 + k]]


def _neighbourhood(kind, case, cur, d, k, rng, groups):
    """Free operations of one iteration.

    window         k operations consecutive in start order (all machines)
    critical       critical operations and their machine neighbours, a slice
                   of k of them
    station        a slice of k operations of one or two machine groups
    jobs           every operation of about k / 22 random jobs
    reseq          3k consecutive operations, re-sequenced on their machines
    station_reseq  3k consecutive operations of one machine group,
                   re-sequenced on their machines (class batching)
    """
    amch, st, ct = cur
    N = case.N
    if kind == 'window':
        return _slice(range(N), st, k, rng)
    if kind == 'reseq':
        return _slice(range(N), st, 3 * k, rng)
    if kind == 'critical':
        crit = critical_ops(case, amch, st, ct, d)
        cand = set(crit)
        for o in crit:
            for q in (d['prev'][o], d['next'][o]):
                if q >= 0:
                    cand.add(int(q))
        return _slice(cand, st, k, rng)
    if kind in ('station', 'station_reseq'):
        if not groups:
            return _slice(range(N), st, k, rng)
        if kind == 'station_reseq':
            g = int(rng.integers(0, len(groups)))
            return _slice(groups[g][1], st, 3 * k, rng)
        ng = 2 if len(groups) > 1 and rng.random() < 0.5 else 1
        pick = rng.choice(len(groups), size=ng, replace=False)
        return _slice([o for g in pick for o in groups[g][1]], st, k, rng)
    if kind == 'jobs':
        nj = max(1, int(round(k / max(1, case.jl.mean()))))
        js = rng.choice(case.J, size=min(nj, case.J), replace=False)
        return [o for j in js for o in range(case.first[j], case.last[j] + 1)]
    raise ValueError(kind)


DEFAULT_LNS = {'k0': None, 'kmin': 10, 'kmax': 600, 'tl0': 2.0, 'tl_max': 16.0,
               'stall_double': 30, 'freeze_before': True, 'insert_radius': 48,
               'kinds': list(NEIGHBOURHOODS),
               # presolve probing on a 1,100-operation sub-model takes about
               # 0.75 s per round and delays the first solution past short
               # sub-solve limits; one light presolve pass reports the hint
               # within a fraction of a second
               'sub_params': {'cp_model_probing_level': 0,
                              'max_presolve_iterations': 1}}


class _SubBest(cp_model.CpSolverSolutionCallback):
    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.best = None

    def on_solution_callback(self):
        self.best = self.ObjectiveValue()


def _sub_solve(built, tl, workers, params, seed):
    """Solve one LNS sub-model. The search stops as soon as the makespan of
    its best solution is proven minimal for the neighbourhood (the bound
    reaches weight * makespan), so that no time goes into the secondary term
    once the makespan is settled."""
    W = built.weight
    solver = cp_model.CpSolver()
    _apply_params(solver, tl, workers, params, seed)
    cb = _SubBest()
    state = {'proven': False}

    def on_bound(bnd):
        if cb.best is not None and bnd >= W * (int(round(cb.best)) // W):
            state['proven'] = True
            solver.StopSearch()

    solver.best_bound_callback = on_bound
    t = time.time()
    status = solver.Solve(built.model, cb)
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError(f'sub-model invalid: {built.model.Validate()[:300]}')
    out = {'status': solver.StatusName(status), 'solve_s': time.time() - t,
           'schedule': None}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out['schedule'] = built.extract(solver)
        ms = int(np.max(out['schedule'][2]))
        state['proven'] = state['proven'] or status == cp_model.OPTIMAL or \
            solver.BestObjectiveBound() >= W * ms
    out['proven'] = state['proven']
    return out


def lns(case, hint, budget, workers, seed=None, **options):
    """Large-neighbourhood search around CP-SAT from a complete feasible
    schedule.

    Each iteration frees a neighbourhood F (one of NEIGHBOURHOODS), keeps
    every other operation on its machine and in its order among the
    non-free operations of that machine, freezes the operations that end
    before F begins (freeze_before), lets a free operation sit only next to
    non-free operations whose current start lies within insert_radius hours
    of its own, and re-solves with CP-SAT under the exact rules, the
    makespan bounded by the incumbent's, and the objective
    weight * makespan + sum of job completions. A result is accepted when it
    is lexicographically no worse (equal-makespan moves with a smaller or
    equal total completion diversify the search). The neighbourhood size
    grows by 20% after a sub-solve that proves its makespan minimal and
    shrinks by 15% after one stopped by its time limit; the sub-solve limit
    doubles after `stall_double` iterations without a makespan improvement
    and is reset by an improvement. Neighbourhood kinds are drawn with
    adaptive weights. Every accepted schedule passes validate_strict."""
    opt = dict(DEFAULT_LNS)
    opt.update(options)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    deadline = t0 + float(budget)
    cur = tuple(np.asarray(x, dtype=int) for x in hint)
    v = validate_strict(case, *cur)
    if v:
        raise ValueError(f'hint infeasible: {v[:3]}')
    cur_ms, cur_sec = int(cur[2].max()), total_completion(case, cur[2])
    anytime = [[round(time.time() - t0, 3), float(cur_ms), None]]
    groups = case.machine_groups() or []
    kinds = [kd for kd in opt['kinds']
             if case.S is not None or kd != 'station_reseq']
    k = opt['k0'] or int(np.clip(case.N // 20, 20, 120))
    tl = opt['tl0']
    w = {kd: 1.0 for kd in kinds}
    stats = {kd: {'n': 0, 'improved': 0, 'accepted': 0, 'proven': 0,
                  'solve_s': 0.0} for kd in kinds}
    build_s, stall, it = 0.0, 0, 0
    while True:
        left = deadline - time.time()
        if left < 0.5:
            break
        it += 1
        pw = np.array([w[x] for x in kinds])
        kd = kinds[int(rng.choice(len(kinds), p=pw / pw.sum()))]
        tb = time.time()
        d = derive(case, *cur)
        F = _neighbourhood(kd, case, cur, d, k, rng, groups)
        frozen = ()
        if opt['freeze_before'] and F:
            t_low = int(min(d['block'][o] for o in F))
            frozen = [o for o in range(case.N) if cur[2][o] <= t_low]
        built = build_model(case, base=cur, free=F,
                            reseq=kd in ('reseq', 'station_reseq'),
                            frozen=frozen, horizon=cur_ms, secondary=True,
                            insert_radius=opt['insert_radius'])
        built.add_hint(*cur)
        build_s += time.time() - tb
        left = deadline - time.time()
        if left < 0.2:
            break
        res = _sub_solve(built, min(tl, left), workers, opt['sub_params'],
                         int(rng.integers(1, 2**31 - 1)))
        st_k = stats[kd]
        st_k['n'] += 1
        st_k['solve_s'] += res['solve_s']
        st_k['proven'] += int(res['proven'])
        improved = False
        if res['schedule'] is not None:
            new = res['schedule']
            new_ms = int(new[2].max())
            new_sec = total_completion(case, new[2])
            if (new_ms, new_sec) <= (cur_ms, cur_sec):
                bad = validate_strict(case, *new)
                if bad:
                    raise RuntimeError(f'sub-model returned an infeasible '
                                       f'schedule ({kd}): {bad[:3]}')
                st_k['accepted'] += 1
                if new_ms < cur_ms:
                    improved = True
                    st_k['improved'] += 1
                    anytime.append([round(time.time() - t0, 3), float(new_ms),
                                    None])
                cur, cur_ms, cur_sec = new, new_ms, new_sec
        w[kd] = max(0.05, 0.8 * w[kd] + 0.2 * (1.0 if improved else 0.0))
        if res['proven']:
            k = min(opt['kmax'], case.N, int(k * 1.2) + 1)
        else:
            k = max(opt['kmin'], int(k * 0.85))
        if improved:
            stall, tl = 0, opt['tl0']
        else:
            stall += 1
            if stall >= opt['stall_double']:
                stall, tl = 0, min(opt['tl_max'], tl * 2)
    for s in stats.values():
        s['solve_s'] = round(s['solve_s'], 2)
    return {'schedule': cur, 'anytime': anytime,
            'build_s': round(build_s, 3), 'wall_s': round(time.time() - t0, 3),
            'options': {**opt, 'kinds': kinds, 'k_final': k, 'tl_final': tl,
                        'iterations': it},
            'stats': stats}


# ---------------------------------------------------------------------------
# Model size (for the report)
# ---------------------------------------------------------------------------


def proto_size(model):
    p = model.Proto()
    kinds = collections.Counter(c.WhichOneof('constraint') for c in p.constraints)
    n_arcs = sum(len(c.circuit.tails) for c in p.constraints
                 if c.WhichOneof('constraint') == 'circuit')
    return {'variables': len(p.variables), 'constraints': len(p.constraints),
            'by_kind': dict(kinds), 'circuit_arcs': n_arcs,
            'bytes': p.ByteSize()}


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)


def size_report(case, hint, arms=('ref', 'compact'), presolve=False,
                presolve_limit=280.0):
    """Model size of each arm's full model on one instance, its build time,
    and optionally the duration of CP-SAT's presolve on one worker
    (stop_after_presolve), with the hint status line of the log."""
    hint = tuple(np.asarray(x, dtype=int) for x in hint)
    hint_ms = int(np.max(hint[2]))
    out = {}
    for a in arms:
        spec = ARMS[a]
        t = time.time()
        if spec['kind'] == 'ref':
            model = build_reference_model(
                case, warmstart=(hint[0], hint[2]),
                complete_hint=spec.get('complete_hint', False))['model']
        else:
            arc_ok = (_radius_filter(case, hint, spec['radius'])
                      if spec.get('radius') else None)
            b = build_model(case, base=hint, horizon=hint_ms,
                            setup_form=spec.get('setup_form', 'circuit'),
                            reseq=spec.get('reseq', False), arc_ok=arc_ok,
                            redundant=spec.get('redundant', False))
            b.add_hint(*hint)
            model = b.model
        rec = proto_size(model)
        rec['build_s'] = round(time.time() - t, 2)
        if presolve:
            solver = cp_model.CpSolver()
            log = []
            _apply_params(solver, presolve_limit, 1,
                          {'stop_after_presolve': True}, log=log)
            t = time.time()
            solver.Solve(model)
            rec['presolve_s_1worker'] = round(time.time() - t, 2)
            rec['hint_log'] = [x for x in log if 'hint' in x.lower()][:3]
            rec['presolved'] = [x for x in log if x.startswith('Presolved')
                                or x.startswith('PresolvedNum')][:4]
        rec['peak_rss_gb'] = peak_rss_gb()
        out[a] = rec
        del model
    return out
