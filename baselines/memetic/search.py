"""Memetic algorithm: genetic search over (operation sequence, machine
assignment) chromosomes, each offspring improved by a tabu search over the
critical-path neighbourhood of its decoded schedule.

Structure (see README.md for the references and what follows which paper)
  chromosome     OS = job-repetition sequence, MA = machine of each operation
  decoding       decode_insert (gap insertion) or decode_append (decoder.py)
  initial MA     global selection, local selection and random selection
                 (Zhang, Gao and Shi, 2011), OS a random permutation; with
                 seed_schedules the dispatching-rule schedules join the
                 initial population
  crossover      POX on OS, uniform crossover on MA
  mutation       swap or insertion on OS, reassignment of random operations
                 (to a random or to the fastest compatible machine) on MA
  local search   tabu search on machine sequences; moves from the critical
                 path of the schedule under the active rules: swaps at the
                 ends of critical blocks, N7 insertions inside blocks, and
                 reassignment of critical operations to other machines;
                 every neighbour is evaluated exactly (eval_seq)
  replacement    (mu + lambda) truncation without duplicates (elitist)
  restart        after stall_gens generations without a new best, all but
                 the restart_keep best individuals are re-initialised
  phase bound    with init_frac, the initial population (seed schedules
                 first) is built only until that fraction of the budget is
                 used, so the genetic phase starts within it
  iterated TS    with its_share > 0, every generation also spends that share
                 of its evaluations on an iterated tabu search of the elite
                 (Lourenco, Martin and Stuetzle): a perturbation by its_kick
                 random critical-path moves (as in Kreipl's large-step
                 random walk), a short tabu search, and acceptance of the
                 result when it is not worse than the current state

After the tabu search an individual is written back as OS = its operations
in ascending block start, MA = its assignment; decode_append of that
chromosome reproduces the tabu-search schedule exactly, so the stored
makespan of every individual is the makespan of a schedule in hand.

Stopping: a wall-clock budget (seconds from the start of the search) or a
maximum number of schedule evaluations (chromosome decodes plus neighbour
evaluations); the second rule makes a run a deterministic function of the
seed.
"""
import time

import numpy as np

from baselines.memetic import decoder as D

# Frozen after the selection on the development cells (50x25 and 80x25,
# combinations N, S, SW, LSW, instances 000-002, 300 CPU-s): the first-round
# configuration is this one with n_offspring 20, ts_scale 5.0, init_frac None
# and its_share 0.0 (ROUND_ONE_CONFIG).
DEFAULT_CONFIG = {
    'decoder': 'insert',          # GA decoder: 'insert' or 'append'
    'pop_size': 20,
    'n_offspring': 5,             # offspring per generation
    'tournament': 2,
    'p_cross': 0.9,
    'p_mut_os': 0.2,              # per child: one swap or insertion on OS
    'p_mut_ma': 0.2,              # per child: reassign mut_ma_ops operations
    'mut_ma_ops': 2,
    'init_split': (0.6, 0.3, 0.1),  # global / local / random machine selection
    'ts_no_improve': None,        # stop a tabu search after this many non-improving
                                  # iterations; None = 100 when N <= 300, else ts_scale x N
    'ts_scale': 0.5,
    'ts_iter': None,              # maximum iterations per tabu search; None = 10 x ts_no_improve
    'ts_cap': 10,                 # insertion moves evaluated per iteration (best estimates)
    'ts_n7': True,                # N7 insertions inside critical blocks
    'ts_cap_n7': 5,               # N7 moves evaluated per iteration (random subset)
    'tie_break': 'ncrit',         # equal makespan: fewer operations on longest chains, or 'random'
    'tenure_min_base': 10,        # tenure in [Lmin, 1.4 Lmin], Lmin = base + J / M
    'stall_gens': 15,
    'restart_keep': 2,
    # diversification at scale
    'init_frac': 0.1,             # the initial population is built until this fraction
                                  # of the budget is used (seeds always enter); None = no bound
    'its_share': 0.9,             # share of each generation's evaluations spent on the
                                  # iterated tabu search of the elite; 0 = off
    'its_kick': 4,                # random critical-path moves per perturbation
    'its_no_improve': None,       # tabu search after a perturbation; None = ts_no_improve // 2
    'its_iter': None,             # None = 10 x its_no_improve
    'its_accept': 'not_worse',    # 'not_worse' (equal makespan accepted) or 'better'
}


ROUND_ONE_CONFIG = {'n_offspring': 20, 'ts_scale': 5.0, 'init_frac': None,
                    'its_share': 0.0}


class StopRule:
    """Time budget and/or evaluation cap, counted from start().

    clock='wall' (the rule for real runs) measures wall seconds; clock='cpu'
    measures the CPU seconds of this process, which keeps development runs
    on a shared core comparable with each other (it is not a deployment
    budget)."""

    def __init__(self, budget_s=None, max_evals=None, clock='wall'):
        if clock not in ('wall', 'cpu'):
            raise ValueError(f'clock must be wall or cpu, got {clock!r}')
        self.budget_s = budget_s
        self.max_evals = max_evals
        self.clock = time.perf_counter if clock == 'wall' else time.process_time
        self.evals = 0
        self.decodes = 0
        self.t0 = None

    def start(self):
        self.t0 = self.clock()

    def elapsed(self):
        return self.clock() - self.t0

    def done(self):
        if self.max_evals is not None and self.evals >= self.max_evals:
            return True
        if self.budget_s is not None and self.elapsed() >= self.budget_s:
            return True
        return False

    def fraction(self):
        """Used share of the budget (the larger of the time share and the
        evaluation share); 0 when neither limit is set."""
        f = 0.0
        if self.max_evals is not None:
            f = max(f, self.evals / self.max_evals)
        if self.budget_s is not None:
            f = max(f, self.elapsed() / self.budget_s)
        return f


class Memetic:
    """origin: (perf_counter, process_time) values taken at the start of the
    process; the anytime record then carries wall and CPU seconds from that
    point next to the budget clock. None = the start of the search."""

    def __init__(self, inst, seed, config=None, budget_s=None, max_evals=None,
                 clock='wall', origin=None):
        self.inst = inst
        self.cfg = dict(DEFAULT_CONFIG)
        if config:
            unknown = set(config) - set(DEFAULT_CONFIG)
            if unknown:
                raise ValueError(f'unknown config keys {sorted(unknown)}')
            self.cfg.update(config)
        if self.cfg['ts_no_improve'] is None:
            # pilots: at 10x25 (220 operations) 100 iterations suffice; at
            # 50x25 and 80x25 a single tabu search walks plateaus of several
            # thousand iterations between improvements
            self.cfg['ts_no_improve'] = (100 if inst.N <= 300
                                         else int(self.cfg['ts_scale'] * inst.N))
        if self.cfg['ts_iter'] is None:
            self.cfg['ts_iter'] = 10 * self.cfg['ts_no_improve']
        if self.cfg['its_no_improve'] is None:
            self.cfg['its_no_improve'] = max(1, self.cfg['ts_no_improve'] // 2)
        if self.cfg['its_iter'] is None:
            self.cfg['its_iter'] = 10 * self.cfg['its_no_improve']
        if not 0.0 <= self.cfg['its_share'] < 1.0:
            raise ValueError('its_share must lie in [0, 1)')
        if self.cfg['its_accept'] not in ('not_worse', 'better'):
            raise ValueError("its_accept must be 'not_worse' or 'better'")
        if self.cfg['init_frac'] is not None and not 0.0 < self.cfg['init_frac'] <= 1.0:
            raise ValueError('init_frac must lie in (0, 1]')
        self.origin = origin
        self.rng = np.random.default_rng(seed)
        self.stop = StopRule(budget_s, max_evals, clock)
        N, M = inst.N, inst.M
        self.N, self.M = N, M
        z = lambda n: np.zeros(n, dtype=np.int64)  # noqa: E731
        # working schedule (the tabu-search state)
        self.ma, self.mpred, self.msucc = z(N), z(N), z(N)
        self.mhead, self.mtail = z(M), z(M)
        self.blk, self.st, self.ct, self.rdy = z(N), z(N), z(N), z(N)
        self.indeg, self.order = z(N), z(N)
        self.path, self.via = z(N + 1), z(N + 1)
        self.buf = np.zeros((N * (2 * M + 4) + 8, 3), dtype=np.int64)
        self.out_ms, self.out_nc = z(len(self.buf)), z(len(self.buf))
        self.est, self.tail = z(len(self.buf)), z(N)
        self.kargs = inst.kernel_args()
        self.tabu_ma = np.zeros((N, M), dtype=np.int64)
        self.ts_clock = 0            # global tabu-search iteration counter
        self.n_nodes = N + 2 * M     # arc endpoints: ops, machine heads, tails
        # incumbent
        self.best_ms = None
        self.best_sched = None
        self.anytime = []
        self.n_restarts = 0
        self.n_generations = 0
        self.n_ts_iter = 0
        # iterated tabu search state (current solution of the walk)
        self.its_snap, self.its_ms = None, None
        self.n_kicks = self.n_kicks_accepted = self.n_kicks_improved = 0
        self.ga_start = None          # (budget fraction, budget-clock s) at the first generation
        self.search_start = None      # (wall s, CPU s) from the origin

    # ------------------------------------------------------------------
    # schedule evaluation and incumbent
    # ------------------------------------------------------------------

    def _eval_state(self):
        ms, nc = D.eval_seq(self.ma, self.mpred, self.msucc, *self.kargs,
                            self.blk, self.st, self.ct, self.indeg, self.order,
                            self.tail)
        self.stop.evals += 1
        return int(ms), int(nc)

    def _decode(self, os_, ma):
        self.ma[:] = ma
        fn = D.decode_insert if self.cfg['decoder'] == 'insert' else D.decode_append
        fn(os_, self.ma, self.inst.first, *self.kargs, self.blk, self.st,
           self.ct, self.rdy, self.mpred, self.msucc, self.mhead, self.mtail)
        self.stop.decodes += 1
        self.stop.evals += 1
        # the semi-active schedule of the decoded machine sequences is never
        # later than the decoded one; continue from it
        return self._eval_state()

    def _offer(self, ms):
        """Record the working schedule as incumbent when it improves. Each
        anytime entry is [budget-clock s since the search start, wall s from
        the origin, CPU s from the origin, makespan]."""
        if self.best_ms is None or ms < self.best_ms:
            self.best_ms = ms
            self.best_sched = (self.ma.copy(), self.st.copy(), self.ct.copy())
            if self.origin is None:
                self.origin = (time.perf_counter(), time.process_time())
            w0, c0 = self.origin
            self.anytime.append([round(self.stop.elapsed(), 3),
                                 round(time.perf_counter() - w0, 3),
                                 round(time.process_time() - c0, 3), int(ms)])

    def _snapshot(self):
        return (self.ma.copy(), self.mpred.copy(), self.msucc.copy(),
                self.mhead.copy(), self.mtail.copy())

    def _restore(self, snap):
        for dst, src in zip((self.ma, self.mpred, self.msucc, self.mhead,
                             self.mtail), snap):
            dst[:] = src

    def _chromosome(self):
        """Write the working schedule back as (OS, MA)."""
        order = np.lexsort((self.st, self.blk))
        return self.inst.job_of[order].copy(), self.ma.copy()

    # ------------------------------------------------------------------
    # tabu search
    # ------------------------------------------------------------------

    def _tenure(self):
        lmin = self.cfg['tenure_min_base'] + self.inst.J / self.inst.M
        return int(self.rng.integers(int(lmin), int(1.4 * lmin) + 1))

    def tabu_search(self, max_iter, no_improve=None, phase_end=None):
        """Improve the working schedule in place; returns its makespan.
        Stops after max_iter iterations, after no_improve (default
        ts_no_improve) iterations without a new best, when the stopping rule
        fires, or once the used budget share reaches phase_end."""
        cfg, N, M = self.cfg, self.N, self.M
        if no_improve is None:
            no_improve = cfg['ts_no_improve']
        lex = cfg['tie_break'] == 'ncrit'
        cur, cur_nc = self._eval_state()
        self._offer(cur)
        # the search state is ranked by (makespan, operations on longest
        # chains) when lex, by makespan alone otherwise
        best, best_snap = cur, self._snapshot()
        best_key = cur * (1 << 32) + (cur_nc if lex else 0)
        tabu = {}
        no_imp = 0
        cap = cfg['ts_cap']
        nn = self.n_nodes
        for _ in range(max_iter):
            if self.stop.done():
                break
            if phase_end is not None and self.stop.fraction() >= phase_end:
                break
            L = D.critical_path(self.ma, self.mpred, self.blk, self.ct,
                                self.inst.jprev, self.inst.lag, self.inst.rel_op,
                                self.path, self.via, int(self.rng.integers(1 << 30)))
            n = D.gen_moves(self.ma, self.mpred, self.msucc, self.mhead,
                            self.blk, self.ct, self.inst.pt, self.inst.jprev,
                            self.inst.jnext, self.inst.lag, self.inst.rel_op,
                            self.inst.cls, self.inst.S, np.int64(self.inst.idle),
                            self.inst.nw, self.inst.ws, self.inst.we,
                            self.inst.cptr, self.inst.cmach, self.tail,
                            self.path, self.via, L, cfg['ts_n7'], self.buf,
                            self.est)
            if n == 0:
                break
            est = self.est[:n]
            seq = np.nonzero(est == -1)[0]
            n7 = np.nonzero(est == -2)[0]
            ins = np.nonzero(est >= 0)[0]
            if len(n7) > cfg['ts_cap_n7']:
                n7 = self.rng.choice(n7, cfg['ts_cap_n7'], replace=False)
            if len(ins) > cap:
                # the cap insertion moves with the smallest estimate (random
                # order among equal estimates)
                pri = est[ins] * 4096 + self.rng.integers(0, 4096, len(ins))
                ins = ins[np.argpartition(pri, cap - 1)[:cap]]
            seq = np.concatenate([seq, n7])
            idx = np.sort(np.concatenate([seq, ins]))
            moves = np.ascontiguousarray(self.buf[idx])
            n = len(idx)
            D.eval_moves(moves, n, self.ma, self.mpred, self.msucc, self.mhead,
                         self.mtail, *self.kargs, self.blk, self.st, self.ct,
                         self.indeg, self.order, self.tail, self.out_ms,
                         self.out_nc)
            self.stop.evals += n
            ms_arr = self.out_ms[:n].copy()
            nc_arr = self.out_nc[:n].copy()
            valid = ms_arr >= 0
            if not valid.any():
                self._eval_state()
                break
            x, y, m2 = moves[:, 0], moves[:, 1], moves[:, 2]
            yy = np.where(y >= 0, y, 0)
            Y = np.where(y >= 0, y, N + m2)
            Z = np.where(y >= 0, self.msucc[yy], self.mhead[m2])
            Z = np.where(Z >= 0, Z, N + M + m2)
            active = np.array([c for c, e in tabu.items() if e > self.ts_clock],
                              dtype=np.int64)
            hit = (np.isin(Y * nn + x, active) | np.isin(x * nn + Z, active)
                   | ((m2 != self.ma[x]) & (self.tabu_ma[x, m2] > self.ts_clock)))
            key = ms_arr * (1 << 32) + (nc_arr if lex else 0)
            allowed = valid & (~hit | (key < best_key))
            if not allowed.any():
                allowed = valid
            key = np.where(allowed, key, np.iinfo(np.int64).max)
            kmin = key.min()
            cand = np.nonzero(key == kmin)[0]
            j = int(cand[self.rng.integers(len(cand))]) if len(cand) > 1 else int(cand[0])
            xj, yj, mj = int(x[j]), int(y[j]), int(m2[j])
            # removed arcs become tabu (forbid re-creating them)
            om = int(self.ma[xj])
            P = int(self.mpred[xj]) if self.mpred[xj] >= 0 else N + om
            Q = int(self.msucc[xj]) if self.msucc[xj] >= 0 else N + M + om
            ten = self._tenure()
            tabu[P * nn + xj] = self.ts_clock + ten
            tabu[xj * nn + Q] = self.ts_clock + ten
            if mj != om:
                self.tabu_ma[xj, om] = self.ts_clock + ten
            D.apply_move(xj, yj, mj, self.ma, self.mpred, self.msucc,
                         self.mhead, self.mtail)
            cur, cur_nc = self._eval_state()
            if cur != ms_arr[j]:
                raise RuntimeError('move evaluation and re-evaluation disagree')
            cur_key = cur * (1 << 32) + (cur_nc if lex else 0)
            self.ts_clock += 1
            self.n_ts_iter += 1
            if len(tabu) > 64:
                tabu = {c: e for c, e in tabu.items() if e > self.ts_clock}
            if cur_key < best_key:
                best, best_key, best_snap = cur, cur_key, self._snapshot()
                no_imp = 0
                self._offer(cur)
            else:
                no_imp += 1
                if no_imp >= no_improve:
                    break
        self._restore(best_snap)
        ms, _ = self._eval_state()
        if ms != best:
            raise RuntimeError('restored tabu-search state does not reproduce its makespan')
        return best

    # ------------------------------------------------------------------
    # iterated tabu search of the elite
    # ------------------------------------------------------------------

    def _kick(self, k):
        """Perturb the working schedule by k moves drawn uniformly from the
        critical-path neighbourhood of the current schedule (swaps, N7
        insertions, insertions on other machines), each applied whatever its
        makespan; a move that would create a cycle is redrawn (at most eight
        draws per move). Returns the makespan of the perturbed schedule."""
        inst = self.inst
        for _ in range(k):
            L = D.critical_path(self.ma, self.mpred, self.blk, self.ct,
                                inst.jprev, inst.lag, inst.rel_op, self.path,
                                self.via, int(self.rng.integers(1 << 30)))
            n = D.gen_moves(self.ma, self.mpred, self.msucc, self.mhead,
                            self.blk, self.ct, inst.pt, inst.jprev, inst.jnext,
                            inst.lag, inst.rel_op, inst.cls, inst.S,
                            np.int64(inst.idle), inst.nw, inst.ws, inst.we,
                            inst.cptr, inst.cmach, self.tail, self.path,
                            self.via, L, self.cfg['ts_n7'], self.buf, self.est)
            if n == 0:
                break
            for _try in range(8):
                x, y, m = (int(v) for v in self.buf[int(self.rng.integers(n))])
                om, op = D.apply_move(x, y, m, self.ma, self.mpred, self.msucc,
                                      self.mhead, self.mtail)
                ms, _ = self._eval_state()
                if ms >= 0:
                    break
                D.undo_move(x, om, op, self.ma, self.mpred, self.msucc,
                            self.mhead, self.mtail)
                self._eval_state()
        ms, _ = self._eval_state()
        return ms

    def _load_chromosome(self, os_, ma):
        """Working schedule = the semi-active schedule of a written-back
        chromosome (decode_append reproduces it exactly)."""
        self.ma[:] = ma
        D.decode_append(os_, self.ma, self.inst.first, *self.kargs, self.blk,
                        self.st, self.ct, self.rdy, self.mpred, self.msucc,
                        self.mhead, self.mtail)
        self.stop.decodes += 1
        self.stop.evals += 1
        return self._eval_state()[0]

    def its_step(self, quota):
        """Iterated tabu search from the current walk state until quota
        evaluations are used: perturb, run a short tabu search, accept the
        result as the new walk state when it is not worse (or, with
        its_accept='better', strictly better). Returns the walk state as a
        chromosome (OS, MA, makespan)."""
        cfg = self.cfg
        e0 = self.stop.evals
        while self.stop.evals - e0 < quota and not self.stop.done():
            self._restore(self.its_snap)
            self._eval_state()
            self._kick(cfg['its_kick'])
            ms = self.tabu_search(cfg['its_iter'], no_improve=cfg['its_no_improve'])
            self.n_kicks += 1
            ok = (ms < self.its_ms) if cfg['its_accept'] == 'better' else (ms <= self.its_ms)
            if ok:
                self.n_kicks_accepted += 1
                if ms < self.its_ms:
                    self.n_kicks_improved += 1
                self.its_snap, self.its_ms = self._snapshot(), ms
        self._restore(self.its_snap)
        ms, _ = self._eval_state()
        o2, m2 = self._chromosome()
        return o2, m2, ms

    # ------------------------------------------------------------------
    # genetic operators
    # ------------------------------------------------------------------

    def _ma_select(self, kind):
        inst, rng = self.inst, self.rng
        if kind == 'random':
            return np.array([c[rng.integers(len(c))] for c in inst.compat],
                            dtype=np.int64)
        ma = np.zeros(self.N, dtype=np.int64)
        load = np.zeros(self.M, dtype=np.int64)
        for j in rng.permutation(inst.J):
            if kind == 'local':
                load[:] = 0
            for o in range(inst.first[j], inst.first[j] + inst.job_length[j]):
                c = inst.compat[o]
                tot = load[c] + inst.pt[o, c]
                k = c[np.argmin(tot)]
                ma[o] = k
                load[k] += inst.pt[o, k]
        return ma

    def _random_individual(self, kind):
        os_ = self.rng.permutation(self.inst.base_os)
        return os_, self._ma_select(kind)

    def _pox(self, p1, p2):
        J = self.inst.J
        in1 = np.zeros(J, dtype=bool)
        k = int(self.rng.integers(1, J)) if J > 1 else 1
        in1[self.rng.permutation(J)[:k]] = True
        child = p1.copy()
        keep = in1[p1]
        child[~keep] = p2[~in1[p2]]
        return child

    def _mutate(self, os_, ma):
        rng, inst = self.rng, self.inst
        if rng.random() < self.cfg['p_mut_os']:
            a, b = rng.integers(0, self.N, 2)
            if rng.random() < 0.5:
                os_[a], os_[b] = os_[b], os_[a]
            else:
                g = os_[a]
                os_ = np.insert(np.delete(os_, a), b, g)
        if rng.random() < self.cfg['p_mut_ma']:
            for o in rng.integers(0, self.N, self.cfg['mut_ma_ops']):
                c = inst.compat[o]
                ma[o] = (inst.min_pt_mach[o] if rng.random() < 0.5
                         else c[rng.integers(len(c))])
        return os_, ma

    def _tournament(self, fit):
        k = self.cfg['tournament']
        cand = self.rng.integers(0, len(fit), size=k)
        return int(cand[np.argmin(fit[cand])])

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def _improve(self, os_, ma, iters, phase_end=None):
        self._decode(os_, ma)
        ms = self.tabu_search(iters, phase_end=phase_end)
        o2, m2 = self._chromosome()
        return o2, m2, ms

    def _seed_individual(self, amch, ct):
        """Chromosome of a complete schedule (e.g. a dispatching-rule
        schedule): machine sequences by processing start, then the
        semi-active schedule of those sequences."""
        inst = self.inst
        amch = np.asarray(amch, dtype=np.int64)
        ct = np.asarray(ct, dtype=np.int64)
        st = ct - inst.pt[np.arange(self.N), amch]
        self.ma[:] = amch
        order = np.lexsort((np.arange(self.N), st))
        D.seq_from_schedule(self.ma, st, order, self.mpred, self.msucc,
                            self.mhead, self.mtail)
        ms, _ = self._eval_state()
        return ms

    def _phase_over(self, phase_end):
        return phase_end is not None and self.stop.fraction() >= phase_end

    def _init_population(self, seeds):
        P = self.cfg['pop_size']
        phase_end = self.cfg['init_frac']
        pop = []
        # every seed schedule enters the incumbent record before any search;
        # the seeds are then improved best first; with init_frac, a seed met
        # after the phase has ended enters the population without search
        seed_ms = []
        for amch, ct in seeds:
            seed_ms.append(self._seed_individual(amch, ct))
            self._offer(seed_ms[-1])
        self.seed_makespans = [int(v) for v in seed_ms]
        for k in np.argsort(seed_ms, kind='stable'):
            amch, ct = seeds[k]
            if self.stop.done():
                break
            ms = self._seed_individual(amch, ct)
            if not self._phase_over(phase_end):
                ms = self.tabu_search(self.cfg['ts_iter'], phase_end=phase_end)
            o2, m2 = self._chromosome()
            pop.append((o2, m2, ms))
        g, l, _ = self.cfg['init_split']
        n_rand = P - len(pop)
        kinds = (['global'] * int(round(g * n_rand)) + ['local'] * int(round(l * n_rand)))
        kinds += ['random'] * (n_rand - len(kinds))
        for kind in kinds:
            if self.stop.done() or (pop and self._phase_over(phase_end)):
                break
            os_, ma = self._random_individual(kind)
            pop.append(self._improve(os_, ma, self.cfg['ts_iter'], phase_end=phase_end))
        return pop

    def run(self, seeds=()):
        """Run until the stopping rule fires. seeds: iterable of complete
        schedules (amch, ct) inserted into the initial population."""
        cfg = self.cfg
        if self.origin is None:
            self.origin = (time.perf_counter(), time.process_time())
        self.stop.start()
        self.search_start = (time.perf_counter() - self.origin[0],
                             time.process_time() - self.origin[1])
        pop = self._init_population(list(seeds))
        if not pop:
            raise RuntimeError('stopping rule fired before any individual was built')
        stall = 0
        best_seen = min(p[2] for p in pop)
        its_on = cfg['its_share'] > 0
        while not self.stop.done():
            if self.ga_start is None:
                self.ga_start = (round(self.stop.fraction(), 4),
                                 round(self.stop.elapsed(), 3))
            self.n_generations += 1
            fit = np.array([p[2] for p in pop], dtype=np.int64)
            children = []
            e0 = self.stop.evals
            for _ in range(cfg['n_offspring']):
                if self.stop.done():
                    break
                a, b = self._tournament(fit), self._tournament(fit)
                if self.rng.random() < cfg['p_cross']:
                    os_ = self._pox(pop[a][0], pop[b][0])
                    ma = np.where(self.rng.random(self.N) < 0.5, pop[a][1], pop[b][1])
                else:
                    os_, ma = pop[a][0].copy(), pop[a][1].copy()
                os_, ma = self._mutate(os_, ma)
                children.append(self._improve(os_, ma, cfg['ts_iter']))
            if its_on and not self.stop.done():
                # the walk restarts from the population's best when that is
                # better than the walk's current state
                elite = min(pop + children, key=lambda p: p[2])
                if self.its_snap is None or elite[2] < self.its_ms:
                    self.its_ms = self._load_chromosome(elite[0], elite[1])
                    self.its_snap = self._snapshot()
                share = cfg['its_share']
                quota = share / (1.0 - share) * max(1, self.stop.evals - e0)
                children.append(self.its_step(quota))
            pop = self._select(pop + children)
            if self.best_ms < best_seen:
                best_seen = self.best_ms
                stall = 0
            else:
                stall += 1
            if stall >= cfg['stall_gens'] and not self.stop.done():
                stall = 0
                self.n_restarts += 1
                keep = pop[:cfg['restart_keep']]
                fresh = []
                g, l, _ = cfg['init_split']
                # with init_frac, re-initialisation uses at most that share
                # of the remaining budget
                f0 = self.stop.fraction()
                phase_end = (None if cfg['init_frac'] is None
                             else f0 + cfg['init_frac'] * (1.0 - f0))
                for k in range(cfg['pop_size'] - len(keep)):
                    if self.stop.done() or (fresh and self._phase_over(phase_end)):
                        break
                    u = self.rng.random()
                    kind = 'global' if u < g else ('local' if u < g + l else 'random')
                    os_, ma = self._random_individual(kind)
                    fresh.append(self._improve(os_, ma, cfg['ts_iter'], phase_end=phase_end))
                pop = self._select(keep + fresh)
        return self.result()

    def _select(self, cand):
        cand = sorted(cand, key=lambda p: p[2])
        out, seen = [], set()
        for p in cand:
            h = (p[2], p[0].tobytes(), p[1].tobytes())
            if h in seen:
                continue
            seen.add(h)
            out.append(p)
            if len(out) == self.cfg['pop_size']:
                break
        return out

    def result(self):
        ma, st, ct = self.best_sched
        return {'makespan': int(self.best_ms), 'op_mch': ma, 'op_st': st,
                'op_ct': ct, 'anytime': self.anytime,
                'n_decodes': int(self.stop.decodes),
                'n_evaluations': int(self.stop.evals),
                'n_generations': self.n_generations,
                'n_restarts': self.n_restarts,
                'n_ts_iterations': self.n_ts_iter,
                'n_its_kicks': self.n_kicks,
                'n_its_accepted': self.n_kicks_accepted,
                'n_its_improved': self.n_kicks_improved,
                'ga_start_fraction': None if self.ga_start is None else self.ga_start[0],
                'ga_start_s': None if self.ga_start is None else self.ga_start[1],
                'seed_makespans': list(getattr(self, 'seed_makespans', [])),
                'search_start_wall_s': round(self.search_start[0], 3),
                'search_start_cpu_s': round(self.search_start[1], 3),
                'search_s': round(self.stop.elapsed(), 3),
                'search_wall_s': round(self.stop.elapsed(), 3)}
