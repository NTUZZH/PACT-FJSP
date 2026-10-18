"""How often does a good reference schedule hold a machine idle on purpose?

The policy's mask is NON-DELAY: it admits only (candidate, machine) pairs
that can start at the decision instant, so the policy can never wait for a
cheaper or better-timed later job. CP-SAT and the GA search the full space.
The review asks for the missing quantification: in the solver's best-found
schedules, how often is a dispatch made that the non-delay mask would have
forbidden? If rarely, the restriction is a quantified non-issue; if often
(and where), that is the decision-space cost the jointact arm targets.

Method, per (instance, regime) cell:
  1. Warm-started CP-SAT solve at --budget seconds (same best-of-four-rules
     hint protocol as the published fairwarm tiers, same cache).
  2. LEFT-SHIFT canonicalization: keep the solver's per-machine sequences and
     job orders, replay them through the reference simulator (attached
     setups, window push, lags). This removes non-critical slack that CP-SAT
     leaves behind, which would otherwise be counted as intent.
  3. REPLAY the canonicalized schedule dispatch by dispatch. Before each
     dispatch, compute over all (job-front, compatible machine) pairs:
       t*  = the earliest processing start   (the non-delay decision instant)
       C*  = the earliest completion         (the active-schedule bound)
     Classify the dispatch of op o with processing start st_o:
       non-delay        st_o <= t*            (the mask would have allowed it)
       active-only      t* < st_o < C*        (needs the G&T relaxation)
       beyond-active    st_o >= C*            (outside even the active space)
  4. Report per regime: mean % of dispatches per class, % instances with at
     least one delayed dispatch, and the class split.

Outputs results/idle_quant.md + paper/macros_idle.tex.

Usage:
  taskset -c 8-23 python scripts/p46_idle_quant.py --budget 30 --workers 8 \
      --jobs 2 [--regimes S LS SW LSW N L W LSW] [--limit 100] [--merge]

--merge re-audits the given regimes and keeps the stored cells of the others.
Each cell records the solve's CPU share and affinity.
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS
from constraint_family.cpsat_family import family_solver
from constraint_family.reference_sim import simulate, validate_schedule
from ppvc_instance_generator import load_instance

DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'   # instance bodies (eval_family protocol)
WARM_CACHE = 'or_solution/FAMILY/warmstart_cache'
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
EPS = 0.25          # true axis is integer hours; quarter-hour separates ties


def _dispatch_order(jl, op_st, op_mch):
    """The (job, machine) dispatch list that regenerates the schedule:
    ops sorted by start time; ties broken by machine id then op id, which
    preserves every per-machine sequence and every job chain."""
    jl = np.asarray(jl, dtype=int)
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    job_of = np.concatenate([[j] * L for j, L in enumerate(jl)])
    order = sorted(range(len(op_st)),
                   key=lambda o: (op_st[o], op_mch[o], o))
    return [(int(job_of[o]), int(op_mch[o])) for o in order]


def _push_windows(st, dur, spans):
    """Earliest start >= st with [start, start+dur] avoiding every outage."""
    for (ws, we) in spans:
        if st < we - EPS and st + dur > ws + EPS:
            st = we
    return st


def _classify(jl, pt, desc, acts):
    """Replay `acts` (already left-shifted semantics via simulate) and count
    each dispatch's class against the non-delay / active bounds."""
    jl = np.asarray(jl, dtype=int)
    pt = np.asarray(pt, dtype=float)
    J, M = len(jl), pt.shape[1]
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    last = first + jl - 1
    a_L, a_S, a_W = desc.has_lag, desc.has_setup, desc.has_window
    lag = np.asarray(desc.lag, dtype=float)
    S = (np.asarray(desc.setup, dtype=float) if a_S and desc.setup is not None
         else np.zeros((1, 1)))
    cls = np.asarray(desc.op_class, dtype=int)
    wins = ([sorted(w) for w in desc.windows]
            if a_W and desc.windows is not None else [[] for _ in range(M)])

    next_op = first.copy()
    job_ready = np.zeros(J)
    mch_free = np.zeros(M)
    mch_last = np.full(M, IDLE_CLASS, dtype=int)
    counts = {'nondelay': 0, 'active_only': 0, 'beyond_active': 0}

    def earliest(op, m):
        s = float(S[mch_last[m], cls[op]]) if a_S else 0.0
        p = pt[op, m]
        block = max(job_ready[job_of[op]], mch_free[m])
        block = _push_windows(block, s + p, wins[m])
        return block + s, block + s + p          # processing start, completion

    job_of = np.concatenate([[j] * L for j, L in enumerate(jl)])

    for (j, m) in acts:
        op = next_op[j]
        # bounds over ALL admissible pairs at this state
        tstar, cstar = np.inf, np.inf
        for jj in range(J):
            oo = next_op[jj]
            if oo > last[jj]:
                continue
            for mm in np.flatnonzero(pt[oo] > 0):
                st_c, ct_c = earliest(oo, int(mm))
                tstar, cstar = min(tstar, st_c), min(cstar, ct_c)
        st, ct = earliest(op, m)
        if st <= tstar + EPS:
            counts['nondelay'] += 1
        elif st < cstar - EPS:
            counts['active_only'] += 1
        else:
            counts['beyond_active'] += 1
        # commit
        job_ready[j] = ct + (lag[op] if a_L else 0.0)
        mch_free[m] = ct
        if a_S:
            mch_last[m] = cls[op]
        next_op[j] += 1
    return counts


def one_cell(args):
    stem, regime, budget, workers = args
    name = os.path.basename(stem)
    jl, pt, meta = load_instance(stem)
    with open(os.path.join(DATA, 'desc', regime, f'{name}.desc.json')) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    z = np.load(f'{WARM_CACHE}/{regime}/{name}.npz', allow_pickle=True)
    warm = (z['amch'], z['ct'])
    import time as _t
    _c0, _w0 = _t.process_time(), _t.time()
    res = family_solver(jl, pt, desc, time_limit=budget,
                        num_workers=workers, warmstart=warm)
    share = (_t.process_time() - _c0) / max((_t.time() - _w0) * workers, 1e-9)
    if res.get('op_st') is None:
        return regime, name, None
    # credited convention: if the search returned nothing better than the
    # hint, the hint IS the schedule under audit
    if res.get('objective') is None or res['objective'] >= float(z['ms']):
        amch, ct = warm
        st = np.array([ct[i] - pt[i, amch[i]] for i in range(pt.shape[0])])
        src = 'warm'
    else:
        amch, st = np.asarray(res['assigned_mch']), np.asarray(res['op_st'])
        src = 'search'
    # left-shift: regenerate from the dispatch order under attached semantics
    acts = _dispatch_order(jl, st, amch)
    sim = simulate(jl, pt, desc, acts)
    viol = validate_schedule(jl, pt, desc, sim['op_st'], sim['op_ct'],
                             sim['op_mch'])
    if viol:
        return regime, name, {'error': f'left-shift infeasible: {viol[:2]}'}
    counts = _classify(jl, pt, desc, acts)
    n = sum(counts.values())
    return regime, name, {
        'src': src, 'makespan': float(sim['makespan']),
        'orig_makespan': float(np.max(st + np.array(
            [pt[i, amch[i]] for i in range(pt.shape[0])]))),
        'n_dispatch': n, 'cpu_share': round(share, 3),
        'affinity': sorted(os.sched_getaffinity(0)), **counts}


def main():
    a = list(_ARGV)
    budget = int(a[a.index('--budget') + 1]) if '--budget' in a else 30
    workers = int(a[a.index('--workers') + 1]) if '--workers' in a else 8
    jobs = int(a[a.index('--jobs') + 1]) if '--jobs' in a else 2
    limit = int(a[a.index('--limit') + 1]) if '--limit' in a else 0
    regimes = REGIMES
    if '--regimes' in a:
        i = a.index('--regimes') + 1
        regimes = []
        while i < len(a) and not a[i].startswith('--'):
            regimes.append(a[i]); i += 1

    tasks = []
    for r in regimes:
        descs = sorted(glob.glob(os.path.join(DATA, 'desc', r,
                                              'instance_*.desc.json')))
        names = [os.path.basename(d).replace('.desc.json', '') for d in descs]
        if limit:
            names = names[:limit]
        tasks += [(os.path.join(BASE, n), r, budget, workers) for n in names]

    print(f'[p46] {len(tasks)} cells, budget {budget}s x {workers} workers, '
          f'{jobs} concurrent', flush=True)
    per = {r: [] for r in regimes}
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for regime, name, rec in ex.map(one_cell, tasks):
            if rec is None or 'error' in rec:
                print(f'  !! {regime}/{name}: {rec}', flush=True)
                continue
            per[regime].append(rec)
            if len(per[regime]) % 25 == 0:
                print(f'  {regime}: {len(per[regime])} done', flush=True)

    if '--merge' in a:      # re-audit some regimes, keep the others' cells
        old = json.load(open('results/idle_quant_cells.json'))
        old.update({r: v for r, v in per.items() if v})
        per = old
        regimes = [r for r in REGIMES if r in per]
    lines = ['# How often the reference schedule leaves the non-delay space',
             '',
             f'Warm-started CP-SAT at {budget}s, credited vs its hint, '
             'left-shifted before audit.',
             '',
             '| regime | cells | % delayed dispatches (mean) | '
             '% cells with >=1 delayed | % of delayed beyond active | '
             'left-shift gain (mean h) |',
             '|---|---|---|---|---|---|']
    mac = ['% auto-generated by scripts/p46_idle_quant.py -- do not hand-edit']
    for r in regimes:
        R = per[r]
        if not R:
            continue
        frac = [100.0 * (c['active_only'] + c['beyond_active']) /
                c['n_dispatch'] for c in R]
        any1 = 100.0 * np.mean([(c['active_only'] + c['beyond_active']) > 0
                                for c in R])
        delayed = sum(c['active_only'] + c['beyond_active'] for c in R)
        beyond = sum(c['beyond_active'] for c in R)
        bfrac = 100.0 * beyond / delayed if delayed else 0.0
        gain = np.mean([c['orig_makespan'] - c['makespan'] for c in R])
        lines.append(f'| {r} | {len(R)} | {np.mean(frac):.2f} | {any1:.0f} | '
                     f'{bfrac:.0f} | {gain:.2f} |')
        mac.append(f'\\newcommand{{\\Idle{r}Pct}}{{{np.mean(frac):.1f}}}')
        mac.append(f'\\newcommand{{\\Idle{r}Any}}{{{any1:.0f}}}')
    with open('results/idle_quant.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    with open('results/idle_quant_cells.json', 'w') as f:
        json.dump(per, f)
    print('\n'.join(lines))
    print('-> results/idle_quant.md, results/idle_quant_cells.json')


if __name__ == '__main__':
    main()
