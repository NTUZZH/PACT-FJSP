# Memetic algorithm for the FJSP timing-rule family

This package implements a memetic algorithm for the flexible job shop with
makespan objective under the three composable timing rules of the benchmark
(minimum time lags between consecutive operations of a job, attached
sequence-dependent setups keyed by the job classes of consecutive operations
on a machine, machine unavailability windows without preemption) and optional
per-job release dates. A genetic algorithm searches over chromosomes, and a
tabu search improves every offspring on the critical path of its decoded
schedule.

## Templates

1. X. Li and L. Gao, "An effective hybrid genetic algorithm and tabu search
   for flexible job shop scheduling problem", *International Journal of
   Production Economics* 174 (2016) 93-110, doi:10.1016/j.ijpe.2016.01.016.
   The overall design follows this paper: a genetic algorithm with a
   two-part chromosome (operation sequence and machine assignment) explores,
   and a tabu search on each offspring exploits.
2. L. Shen, S. Dauzère-Pérès and J. S. Neufeld, "Solving the flexible job
   shop scheduling problem with sequence-dependent setup times", *European
   Journal of Operational Research* 265(2) (2018) 503-516,
   doi:10.1016/j.ejor.2017.08.021 (corrigendum: EJOR 296(2) (2022)). The
   tabu search works on the disjunctive-graph representation in which the
   setup of an operation is fixed by its machine predecessor, and its moves
   act on critical operations within a machine and between machines, as in
   this paper.

Component sources used inside the two templates:

- initial machine assignment by global, local and random selection:
  G. Zhang, L. Gao and Y. Shi, *Expert Systems with Applications* 38(4) (2011)
  3563-3573, doi:10.1016/j.eswa.2010.08.145;
- block moves inside critical blocks: E. Nowicki and C. Smutnicki,
  *Management Science* 42(6) (1996) 797-813 (swaps at the block ends), and
  C. Y. Zhang, P. G. Li, Z. L. Guan and Y. Q. Rao, *Computers & Operations
  Research* 34(11) (2007) 3229-3242, doi:10.1016/j.cor.2005.12.002 (the N7
  insertions and the tenure range [Lmin, 1.4 Lmin], Lmin = 10 + J/M);
- insertion of a critical operation at the best estimated position of any
  compatible machine, with positions restricted so that the operation never
  goes after one of its successors or before one of its predecessors: the
  rule of M. Mastrolilli and L. M. Gambardella, *Journal of Scheduling* 3(1)
  (2000) 3-20;
- iterated local search with a short tabu search as its local search, and
  its acceptance criteria: H. R. Lourenço, O. C. Martin and T. Stützle,
  "Iterated local search: framework and applications", in M. Gendreau and
  J.-Y. Potvin (eds.), *Handbook of Metaheuristics*, 3rd ed., Springer (2019)
  129-168, doi:10.1007/978-3-319-91086-4_5; for the job shop with short tabu
  searches inside: H. R. Lourenço, *European Journal of Operational
  Research* 83(2) (1995) 347-364, doi:10.1016/0377-2217(95)00012-F;
- perturbation by a few random moves on critical arcs: S. Kreipl, "A large
  step random walk for minimizing total weighted tardiness in a job shop",
  *Journal of Scheduling* 3(3) (2000) 125-138.

## Method

**Chromosome.** OS is a job-repetition sequence (job j appears once per
operation of j; its k-th occurrence is the k-th operation of j). MA gives a
compatible machine to every operation.

**Decoding (one builder for all rules).** `decoder.py` implements the timing
semantics of `constraint_family/reference_sim.py` in integer hours. An
operation with job-ready time r (release date, or end of the job predecessor
plus its lag) and machine predecessor a occupies its machine over the block
[b, b + s + p], where s is the setup from the class of a (cold-start row when
a does not exist), b >= max(r, end(a)), and the block overlaps no outage. Two
builders exist:

- `decode_append` appends every operation to its machine at the earliest
  feasible time; it equals `reference_sim.simulate` exactly (tested);
- `decode_insert` (the default in the search) places every operation into
  the earliest idle interval of its machine in which (i) the job is present,
  (ii) the setup from the operation before the interval and the operation fit
  without overlapping an outage, and (iii) the operation after the interval,
  whose setup now comes from the inserted operation's class, keeps its
  processing start, its new setup block starts no earlier than its own job
  is present, and the enlarged block overlaps no outage. Every schedule it
  returns passes the project's validator in all eight regimes (tested).

Neither builder is restricted to the dispatching policy's non-delay rule: a
machine may stay idle while a job waits, as in the CP-SAT model and the
2008-style GA.

**Schedule evaluation for the local search.** `eval_seq` computes the
earliest schedule for fixed machine sequences (doubly linked lists) and a
fixed assignment by Kahn's algorithm over the precedence graph; it detects
cycles, and a backward pass gives every operation's tail (longest chain of
successor blocks). The earliest schedule for given sequences is the
component-wise minimum, because the outage push is monotone, so the
optimum is reachable through machine sequences.

**Tabu search.** From the current schedule a critical path is traced from an
operation that ends at the makespan through tight arcs (machine arc with the
setup, job arc with the lag); where an outage pushed a block past both arcs,
the path continues through the later arc, because an earlier end there can let
the block fit before the outage. The neighbourhood holds (i) swaps of the
first two and of the last two operations of each critical block, (ii) a random
subset of the N7 insertions inside blocks, and (iii) for every critical
operation and every compatible machine, the insertion at the position with the
smallest estimate (block end at that position, with the setups of the new
neighbours and the outage push of that machine, plus the longest tail through
the operation); the best estimates are kept. Every kept neighbour is
evaluated exactly with `eval_seq`, so no approximation enters the accepted
move. Moves are ranked by makespan, ties by the number of operations on
longest chains. Tabu attributes are the two machine arcs a move removes
(re-creating either is forbidden) and, after a reassignment, the old
(operation, machine) pair, with a random tenure in [Lmin, 1.4 Lmin]; a tabu
move is accepted when it beats the best state of the run. A run stops after
`ts_no_improve` iterations without improvement (100 up to 300 operations,
`ts_scale` x N above, i.e. 550 at 50x25 and 880 at 80x25 in the frozen
configuration) or after `ts_iter` iterations.

**Genetic search.** The initial population holds the seed schedules (with
`--seed_schedules --warm_cache F` the dispatching-rule schedule that CP-SAT
receives as its hint, read from F; with `--seed_schedules` alone the four
rules FIFO, MOR, SPT and MWKR rolled out; with `--seed_from F1 ...` in
addition the schedule stored in each folder, e.g. the policy's schedule),
each written back as a chromosome that reproduces it exactly, and
otherwise individuals with machine assignments from global selection (60%),
local selection (30%) and random selection (10%) and random sequences. Every
individual is improved by the tabu search and written back (OS = operations in
ascending block start, MA = assignment), so that `decode_append` of the stored
chromosome reproduces its schedule. Parents are chosen by binary tournament;
POX crossover acts on OS and uniform crossover on MA; mutation swaps or moves
one OS gene and reassigns operations to a random or to the fastest compatible
machine. Offspring are decoded by gap insertion and improved by the tabu
search; the next population is the best `pop_size` distinct individuals of
parents and offspring. After `stall_gens` generations without a new best, all
but the `restart_keep` best individuals are replaced by new ones.

**Phase bound.** With `init_frac` set, the initial population is built only
until that share of the budget (time share, or evaluation share under
`--max_decodes`) is used: the seeds are improved best first, every tabu
search stops at the bound, seeds met after it enter without search, and no
further random individual is built. The genetic phase therefore starts
within `init_frac` of the budget, plus at most one tabu-search iteration
(`ga_start_fraction` in the record). A restart uses at most `init_frac` of the
remaining budget in the same way.

**Iterated tabu search of the elite.** With `its_share > 0`, every
generation spends that share of its evaluations (the rest goes to the
offspring) on an iterated local search in the sense of Lourenço, Martin and
Stützle, whose local search is a short tabu search as in Lourenço (1995):
the walk starts from the population's best schedule (and restarts from it
whenever the population holds a better one); each step perturbs the walk's
current schedule by `its_kick` moves drawn uniformly from its critical-path
neighbourhood (swaps, N7 insertions, insertions on other machines), each
applied whatever its makespan and redrawn when it would create a cycle, as
the large-step random walk of Kreipl (2000) perturbs by a few random moves on
critical arcs; then the tabu search runs for at most `its_iter` iterations
and stops after `its_no_improve` iterations without a new best; the result
replaces the current schedule when its makespan is not worse (`its_accept =
'not_worse'`; `'better'` is the strict Better criterion of Lourenço et al.).
The walk's current schedule joins the offspring of the generation. Accepting
equal makespans lets the walk move along the plateaus that the diagnosis at
50 and 80 modules found (no single critical-path move lowers the makespan of
the final schedules there).

**Adaptations to the three timing rules.** Lags enter as weights of job arcs;
setups are attached to the machine arc that determines them, so a move
re-evaluates every setup it changes, before and after the moved operation;
outages are applied by the monotone push inside every evaluation and inside
the insertion estimate; the critical path is continued through outage pushes;
release dates are heads of first operations. Two operations of the same job
that follow each other on one machine are joined by a job arc for the block
structure, because they cannot be swapped.

## Configuration

`DEFAULT_CONFIG` in `search.py` was frozen by a selection fixed in advance:
sizes 50x25 and 80x25, combinations N, S, SW and LSW, instances 000-002 (24
development cells), 300 CPU seconds, seed 1, the stored rule warm start as the
only seed. Candidates were the first-round configuration (A), A with
`ts_cap` 50 (B), and one new configuration (C: `init_frac` 0.1, `its_share`
0.9, `ts_scale` 0.5, `its_kick` 4, `n_offspring` 5, chosen in three pilots on
the 80x25 LSW instance 000). The score was the mean over the cells of the
relative excess over the best candidate of the cell; C scored 0.11%, A 2.63%,
B 4.08%, and C was best on 20 of the 24 cells. The frozen default is C;
`ROUND_ONE_CONFIG` restores A (`--config '{"n_offspring": 20, "ts_scale": 5.0,
"init_frac": null, "its_share": 0.0}'`). At 10x25 (220 operations) the
first-round tests and figures used A; the selection did not cover that size.

## Files

- `instance.py`: integer arrays of an instance and its descriptor;
- `decoder.py`: numba kernels (decoders, sequence evaluation, critical path,
  moves);
- `search.py`: the memetic algorithm and its configuration (`DEFAULT_CONFIG`);
- `validate.py`: configuration check C6 plus the attached-setup presence
  check;
- `records.py`: `incumbent_at(record, t, clock, origin)`, the incumbent at
  time t on the wall or CPU clock, from the start of the solving process or
  of the search;
- `revalidate.py`: re-validates stored records from their schedules and,
  with `--timing`, applies the rule cpu_share >= 0.9 on one exclusive core
  (`python -m baselines.memetic.revalidate <dir> --timing`);
- `../../eval_memetic_family.py`: the runner (one process and one thread per
  instance, resumable, refuses to write a schedule that fails validation);
- `../../tests/test_memetic.py`: tests.

## Stopping rules and records

`--budget` is a wall-clock budget counted from the start of the search; the
instance loading, the seed schedules (read and validated, or rolled out) and
the numba warm-up run before it, as the CP-SAT warm start is built before its
budget. `--clock cpu` counts CPU seconds instead, for development runs on a
shared core only. `--max_decodes` caps the number of schedule evaluations
(chromosome decodes plus neighbour evaluations) and makes a run a
deterministic function of `--seed`.

The runner records every improvement of the incumbent as
[budget-clock s since the search start, wall s since the start of the
solving process, CPU s since the start of that process, makespan]
(`anytime_columns`); `search_start_wall_s` and `search_start_cpu_s` give the
span before the search (it includes the warm-up, also stored alone as
`jit_time_s`), so the incumbent at 1, 10, 60, 300, 600 or 1800 s can be read
from one run with `records.incumbent_at` on either clock and from either
origin. The record also holds the counts (decodes, evaluations, generations,
restarts, tabu-search iterations, perturbations made, accepted and
improving), `ga_start_fraction`, wall and CPU seconds of the timed window,
`cpu_share`, the affinity mask, the load average at the start and at the end,
the seeds with their source folder and makespan (`warm_cache`, `seed_from`,
`seeds`), and the final schedule. Sizes are folders under `data/FAMILY`
(`50x25n30`, `80x25n30` included); a descriptor carrying `setup_mult` is
refused.
