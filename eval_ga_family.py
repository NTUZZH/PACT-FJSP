"""GA baseline on the FAMILY benchmark (this paper, E5).

A time-budgeted genetic algorithm whose DECODER is the constraint-family
environment itself (batched), so the GA optimizes exactly the dynamics the
DRL policy and CP-SAT references target — its makespans are directly
comparable. Chromosome = (OS, MA):
  OS : operation sequence, job-repetition encoding (job j appears
       job_length[j] times); walking it yields a precedence-feasible order
       because the env always exposes a job's NEXT unscheduled op.
  MA : per-operation machine assignment (kept compatible by construction).
The whole population is decoded in ONE batched env rollout of N steps.

Operators mirror the lab GA (POX on OS, uniform on MA, low-rate mutation,
elitism, random-immigrant restarts on stagnation). PDR seeds (deterministic
family-env rollouts of each priority rule) put the starting front at PDR
level so the budget is spent BEATING the PDRs.

Usage:
  python eval_ga_family.py --regimes N L S W LS LW SW LSW \
      [--budget 30] [--pop 100] [--limit 100] [--jobs 6]
Writes Result_GA_<regime>.npy alongside the policy results and a summary.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# GA decodes through the numpy env; the only torch op is the token attach.
# Force CPU BEFORE the env import chain fixes EnvState.device, so the GA never
# builds a CUDA context (keeps GPU memory free for concurrent training).
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from params import configs
configs.device = 'cpu'

import numpy as np

from constraint_family.descriptor import ConstraintDescriptor
from constraint_family.fjsp_env_family import FJSPEnvFamily
from constraint_family.reference_sim import simulate, validate_schedule
from ppvc_instance_generator import load_instance
from common_utils import heuristic_select_action

# Shop size. The published E5 GA columns are 10x25 and that stays the default,
# so no existing command changes meaning; --size 20x25 points the same run at
# the scale campaigns E6 evaluates on.
SIZE = '10x25'
for _i, _a in enumerate(_ARGV):
    if _a == '--size':
        SIZE = _ARGV[_i + 1]
RES = f'test_results/FAMILY/{SIZE}'
BASE = f'data/PPVC/{SIZE}+ppvc-mixed'
DATA = f'data/FAMILY/{SIZE}'


def _job_starts(job_length):
    return np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(int)


def _compat(op_pt):
    return [np.where(op_pt[i] > 0)[0] for i in range(op_pt.shape[0])]


def _descriptor_arrays(desc, M):
    """Pack descriptor into dense true-hour arrays for the batched decoder."""
    from constraint_family.descriptor import N_CLASSES, IDLE_CLASS
    S = (desc.setup.astype(float) if desc.has_setup
         else np.zeros((N_CLASSES + 1, N_CLASSES)))
    if desc.has_window and desc.windows is not None:
        K = max((len(w) for w in desc.windows), default=0)
        ws = np.full((M, max(K, 1)), np.inf)
        we = np.full((M, max(K, 1)), np.inf)
        for m, spans in enumerate(desc.windows):
            for k, (s, e) in enumerate(spans):
                ws[m, k] = s; we[m, k] = e
        n_win = K
    else:
        ws = we = np.full((M, 1), np.inf)
        n_win = 0
    return S, ws, we, n_win, IDLE_CLASS


def decode_population(jl, pt, desc, opts, mchts, os_pop, ma_pop):
    """Lightweight BATCHED decode of a whole population (true-hour times only,
    no features). Vectorized over P individuals; mirrors the family env's
    attached-setup + window-push + lag dynamics. Fuzz-tested against the
    scalar reference simulator (scripts/p5_ga_tests.py). Returns
    (makespans [P], amch [P, N], op_ct [P, N])."""
    P, N, M = len(os_pop), pt.shape[0], pt.shape[1]
    js = _job_starts(jl)
    op_class = np.repeat(desc.op_class if desc.has_setup else
                         np.zeros(len(jl), dtype=int), 1) \
        if False else np.asarray(desc.op_class, dtype=int)
    lag = desc.lag.astype(float) if desc.has_lag else np.zeros(N)
    S, ws, we, n_win, IDLE = _descriptor_arrays(desc, M)
    eps = 0.25

    # precompute the global op id and machine scheduled at each OS position
    op_at = np.empty((P, N), dtype=int)
    occ = np.zeros((P, len(jl)), dtype=int)
    pr = np.arange(P)
    for t in range(N):
        j = os_pop[:, t]
        op_at[:, t] = js[j] + occ[pr, j]
        occ[pr, j] += 1
    m_at = np.take_along_axis(ma_pop, op_at, axis=1)      # [P, N]

    job_ready = np.zeros((P, len(jl)))
    mch_free = np.zeros((P, M))
    mch_last = np.full((P, M), IDLE, dtype=int)
    op_ct = np.zeros((P, N))
    makespan = np.zeros(P)
    amch = np.full((P, N), -1, dtype=int)

    for t in range(N):
        j = os_pop[:, t]
        op = op_at[:, t]
        m = m_at[:, t]
        amch[pr, op] = m
        setup = (S[mch_last[pr, m], op_class[op]] if desc.has_setup
                 else np.zeros(P))
        dur = pt[op, m].astype(float)
        block = np.maximum(job_ready[pr, j], mch_free[pr, m])
        if n_win:
            wsm, wem = ws[m], we[m]                       # [P, K]
            need = setup + dur
            for k in range(wsm.shape[1]):
                overlap = (block < wem[:, k] - eps) & \
                          (block + need > wsm[:, k] + eps)
                block = np.where(overlap, wem[:, k], block)
        st = block + setup
        ct = st + dur
        op_ct[pr, op] = ct
        job_ready[pr, j] = ct + (lag[op] if desc.has_lag else 0.0)
        mch_free[pr, m] = ct
        if desc.has_setup:
            mch_last[pr, m] = op_class[op]
        makespan = np.maximum(makespan, ct)
    return makespan, amch, op_ct


def pdr_seed_chromosomes(jl, pt, desc, rules=('FIFO', 'MOR', 'SPT', 'MWKR'),
                         seed=50):
    """Deterministic family-env PDR rollouts -> (OS, MA) chromosomes."""
    M = pt.shape[1]
    seeds = []
    for rule in rules:
        env = FJSPEnvFamily(jl.shape[0], M, family_mode=True)
        env.set_initial_data([jl], [pt], descriptor_list=[desc])
        np.random.seed(seed)
        os_vec, ma_vec = [], np.full(pt.shape[0], -1, dtype=int)
        while not env.done().all():
            a = heuristic_select_action(rule, env)
            job = a // M
            op = env.candidate[0, job]
            ma_vec[op] = a % M
            os_vec.append(int(job))
            env.step(np.array([a]))
        seeds.append((np.array(os_vec, dtype=int), ma_vec))
    return seeds


def random_chromosome(rng, base_os, compat):
    os_vec = base_os.copy()
    rng.shuffle(os_vec)
    ma = np.array([c[rng.integers(len(c))] for c in compat], dtype=int)
    return os_vec, ma


def pox(rng, p1, p2, n_jobs):
    perm = rng.permutation(n_jobs)
    cut = rng.integers(1, n_jobs) if n_jobs > 1 else 1
    set1 = set(perm[:cut].tolist())
    child = np.full_like(p1, -1)
    mask1 = np.array([j in set1 for j in p1])
    child[mask1] = p1[mask1]
    child[~mask1] = [j for j in p2 if j not in set1]
    return child


def run_ga(jl, pt, desc, opts, mchts, budget_s, rng, pop_size=100, elite=2,
           tour_k=3):
    n_jobs = len(jl)
    N = pt.shape[0]
    compat = _compat(pt)
    base_os = np.concatenate([np.full(int(L), j) for j, L in enumerate(jl)])

    seeds = pdr_seed_chromosomes(jl, pt, desc)
    pop_os = [s[0] for s in seeds]
    pop_ma = [s[1] for s in seeds]
    while len(pop_os) < pop_size:
        o, m = random_chromosome(rng, base_os, compat)
        pop_os.append(o); pop_ma.append(m)
    pop_os = np.array(pop_os); pop_ma = np.array(pop_ma)

    fit, _, _ = decode_population(jl, pt, desc, opts, mchts, pop_os, pop_ma)
    best_i = int(np.argmin(fit))
    best = (pop_os[best_i].copy(), pop_ma[best_i].copy())
    best_fit = float(fit[best_i])

    stagnate, stag_limit, imm_frac = 0, 25, 0.4
    t0, gen = time.time(), 0
    while time.time() - t0 < budget_s:
        gen += 1
        order = np.argsort(fit)
        new_os = [pop_os[i].copy() for i in order[:elite]]
        new_ma = [pop_ma[i].copy() for i in order[:elite]]
        while len(new_os) < pop_size:
            a = _tourn(rng, fit, tour_k)
            b = _tourn(rng, fit, tour_k)
            co = pox(rng, pop_os[a], pop_os[b], n_jobs)
            cm = np.where(rng.random(N) < 0.5, pop_ma[a], pop_ma[b])
            # mutation
            if rng.random() < 0.1 and N > 1:
                x, y = rng.integers(0, N, 2)
                co[x], co[y] = co[y], co[x]
            if rng.random() < 0.1:
                i = rng.integers(0, N)
                cm[i] = compat[i][rng.integers(len(compat[i]))]
            new_os.append(co); new_ma.append(cm)
        pop_os = np.array(new_os); pop_ma = np.array(new_ma)
        fit, _, _ = decode_population(jl, pt, desc, opts, mchts, pop_os, pop_ma)
        gi = int(np.argmin(fit))
        if fit[gi] < best_fit - 1e-9:
            best_fit = float(fit[gi])
            best = (pop_os[gi].copy(), pop_ma[gi].copy())
            stagnate = 0
        else:
            stagnate += 1
            if stagnate >= stag_limit:
                k = int(imm_frac * pop_size)
                worst = np.argsort(fit)[-k:]
                for w in worst:
                    o, m = random_chromosome(rng, base_os, compat)
                    pop_os[w] = o; pop_ma[w] = m
                stagnate = 0
    return best_fit, best, gen


def _tourn(rng, fit, k):
    cand = rng.integers(0, len(fit), size=k)
    return cand[np.argmin(fit[cand])]


def solve_one(task):
    stem, desc_path, budget, pop = task
    jl, pt, meta = load_instance(stem)
    with open(desc_path) as f:
        desc = ConstraintDescriptor.from_json(json.load(f))
    rng = np.random.default_rng(12345)
    t0 = time.time()
    best_fit, (os_v, ma_v), gen = run_ga(
        jl, pt, desc, meta['op_type'], meta['mch_type'], budget, rng,
        pop_size=pop)
    dt = time.time() - t0
    # independent validation of the winning chromosome via the scalar sim
    acts = []
    js = _job_starts(jl)
    nxt = js.copy()
    for j in os_v:
        op = nxt[j]
        acts.append((int(j), int(ma_v[op])))
        nxt[j] += 1
    ref = simulate(jl, pt, desc, acts)
    viol = validate_schedule(jl, pt, desc, ref['op_st'], ref['op_ct'],
                             ref['op_mch'])
    ok = (not viol) and abs(ref['makespan'] - best_fit) < 1e-6
    return (os.path.basename(stem), best_fit, dt, gen, ok)


def main():
    args = list(_ARGV)
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
    budget, pop, limit, jobs = 30, 100, 100, 6
    tag = ''          # '--tag 1s' -> Result_GA1s_<regime>.npy; '' keeps the
    i = 0             # canonical Result_GA_ name (the published 30 s run)
    while i < len(args):
        if args[i] == '--tag':
            tag = args[i + 1]; i += 2
        elif args[i] == '--budget':
            budget = int(args[i + 1]); i += 2
        elif args[i] == '--pop':
            pop = int(args[i + 1]); i += 2
        elif args[i] == '--limit':
            limit = int(args[i + 1]); i += 2
        elif args[i] == '--jobs':
            jobs = int(args[i + 1]); i += 2
        elif args[i] == '--size':
            i += 2        # consumed above, at module level
        elif args[i] == '--regimes':
            regimes = []
            i += 1
            while i < len(args) and not args[i].startswith('--'):
                regimes.append(args[i]); i += 1
        else:
            sys.exit(f'unknown arg {args[i]}')

    os.makedirs(RES, exist_ok=True)
    # The guard exists so a truncated run cannot overwrite the canonical 10x25
    # results. The scale campaigns are smaller by construction (50 instances at
    # 20x25, 20 at 30x25), so the full-coverage count is per size.
    full = {'10x25': 100, '20x25': 50, '30x25': 20}.get(SIZE)
    if full is None:
        sys.exit(f'unknown size {SIZE}: add its instance count to the guard')
    if limit and limit < full:
        sys.exit(f'partial GA runs must not overwrite the canonical results: '
                 f'use --limit {full} at {SIZE} (or edit RES for a scratch run)')
    for regime in regimes:
        descs = sorted(glob.glob(os.path.join(DATA, 'desc', regime,
                                              '*.desc.json')))[:limit]
        tasks = [(os.path.join(BASE, os.path.basename(d).replace(
            '.desc.json', '')), d, budget, pop) for d in descs]
        ms = np.full((len(tasks), 2), np.nan)
        n_ok = 0
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            for k, r in enumerate(ex.map(solve_one, tasks)):
                idx = int(r[0].split('_')[1])
                ms[idx] = [r[1], r[2]]
                n_ok += r[4]
                if (k + 1) % 20 == 0:
                    print(f'  {regime} {k+1}/{len(tasks)} '
                          f'({(time.time()-t0)/60:.1f}m)', flush=True)
        np.save(os.path.join(RES, f'Result_GA{tag}_{regime}.npy'), ms)
        print(f'[{regime}] GA mean={np.nanmean(ms[:,0]):.1f} '
              f'valid={n_ok}/{len(tasks)} budget={budget}s', flush=True)


if __name__ == '__main__':
    main()
