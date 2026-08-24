# One Policy Across Time Lags, Setups and Availability Windows

Artifacts for **"One Policy Across Time Lags, Setups and Availability Windows: A
Shared Environment and a Certified Reward for Flexible Job-Shop
Scheduling"**.

---

## The problem

A flexible job shop is normally scheduled by a policy trained against one fixed set of
timing rules. A real floor's rules do not stay fixed. A new curing recipe imposes a
mandatory wait after an operation. A change in product mix means consecutive jobs on a
machine need a changeover whose length depends on what ran before. A maintenance
program takes a station offline on a calendar. Each change today means training,
revalidating and certifying another policy, one per situation.

This work makes the rules a **run-time input**. One policy covers three constraint types
and all eight of their combinations, and is told at run time which are active.

| | |
|---|---|
| **L** | time lag: a mandatory wait after an operation before the next may start |
| **S** | sequence-dependent setup: a changeover whose length depends on what ran last |
| **W** | availability window: an interval in which a machine cannot run |

The eight *regimes* are the subsets of {L, S, W}, including the empty one.

## The obstacle, and the contribution

The obstacle is certifying a shared **training signal**, not conditioning a
network. A constructive
scheduling policy is trained from a dense reward that pays it, at each step, the
improvement in an estimate of the earliest the shop could still finish. That estimate
must be *exact at completion*, so that the summed payments equal the makespan, and it
must *never rule out a finish the shop can still reach*, so that each intermediate
payment measures real progress. Once the rules become a run-time input, both properties
must hold for every rule and every combination of rules at once.

The paper constructs one estimate that provably does, for the whole family, and
identifies the modeling convention (a *detached* setup) under which it silently stops
being a valid floor while its totals still telescope.

## Results

Numbers are from the macro files in `paper/`, which the scripts below regenerate.
`\EqMargin` is a ±1% equivalence margin, fixed before the experiments were run.

| | |
|---|---|
| Parity with per-regime specialists | equivalent on **5 of 6** training regimes and better on the sixth, on roughly a third of each specialist's experience |
| Zero-shot on the two held-out compositions | equivalent to the specialist trained on that composition alone (both intervals inside the ±1% margin) |
| Against the best dispatching rule | **+2.0%** at the policy's own one-second budget, pooled over all eight regimes |
| Latency | **0.49 s** on **4** CPU cores, no solver on site |
| Training cost | **6.0** GPU-h against **24** GPU-h for the 7-specialist catalogue it replaces (**4.0x**) |
| Scaled to 2--8x the job count, nothing retrained | overtakes the specialists past the training size, and ahead of credited warm-started CP-SAT by **+4.5%** at 50 and **+3.0%** at 80 modules with the solver given **30 min** per instance |
| Trained on the single-rule regimes only, so all **4** compositions are unseen | still outperforms the best dispatching rule on **every one** |
| Out of distribution (setups **1.76x** larger and asymmetric; windows **2.53x** denser) | **0** infeasible schedules; the lead grows under denser windows and dissolves to a statistical tie wherever a changeover competes with another rule. Retraining with the magnitudes randomized across the shifted ranges restores all but one of the collapsed leads, at a window-only price in distribution |

The paper reports its negative results as prominently as its positive ones. The
constraint description adds no accuracy, so the recommended deployment artifact is the
**token-blind** policy (32,786 parameters, the size of a single specialist, against
71,838 for the conditioned one). The description is still required: it configures the
*environment*, selecting the mask and the delay that are applied, so a wrong one builds
the schedule against the wrong rules even though no network reads it.

## Layout

```
constraint_family/          the family environment, the constraint descriptor, the CP-SAT model
model/                      the policy: the dual-attention backbone and the FiLM pathway
train_family.py             training
eval_family.py              evaluation (greedy rollouts, dispatching rules, interventions)
ppvc_instance_generator.py  the benchmark generator

data/PPVC/                  base instances: 10x25 train/test; 20x25 through 80x25 (and 50x15, 80x15) for scale transfer
data/FAMILY/                the constraint descriptors, one per (instance, regime)

trained_network/FAMILY/     the joint policies, the token-blind control, the single-rule
                            policy, the all-eight-regime arms, the domain-randomized
                            arm, and the per-regime specialists (3 seeds each)
trained_network/PPVC/       the lag-only checkpoint the E0 regression anchor is checked against

test_results/FAMILY/        per-instance makespans, every method and regime (.npy)
or_solution/FAMILY/         CP-SAT reference solutions at every anytime budget
results/                    the aggregated tables
paper/                      the generated macro files: every number the paper prints
scripts/                    the generators
```

**Every number in the paper is emitted from these result files by a script in
`scripts/`. None is typed by hand.** `paper/macros_*.tex` holds the output, so rerunning
a generator and diffing that directory is how you check us.

## Environment

Python 3.10+, PyTorch, NumPy, SciPy, OR-Tools (CP-SAT 9.14), Matplotlib. Training uses
one GPU; every analysis script is CPU-only.

## Instance separation

The three pools are disjoint by construction. Training instances are generated online
from fresh 31-bit random seeds and are never written to disk. The 20 validation
instances used for model selection come from seeds 100-119. The 100 test instances per
regime are fixed files from seeds 50000-50099, with descriptor seed 20260706.

## Reproducing the results

```bash
# E0  regression anchor. With conditioning off and a lag-only descriptor, this
#     environment reproduces the released lag-only policy's makespans exactly, so
#     every effect reported afterwards is an effect of the family extensions alone.
python scripts/p1_tests.py

# E1/E2  parity against per-regime specialists, and zero-shot transfer to the two
#        compositions the policy never trained on, as paired equivalence tests
python scripts/p11_equivalence.py

#        the same claims with the TRAINING RUN, not the test instance, as the unit
#        of inference, plus the verdicts at +-0.5%, +-1% and +-2%
python scripts/p27_seed_inference.py

# E5  against dispatching rules, a genetic algorithm, and CP-SAT
python scripts/p12_warm_cache.py --procs 16          # best-of-four warm starts
python scripts/p12_cpsat_fairwarm.py --budget 300    # the solver campaign
python scripts/p13_e5_table_fair.py                  # the table
python scripts/p30_anytime.py                        # where the solver overtakes the policy

# Why the joint policy trails its specialist on W: an experience-matched specialist
# separates "it saw less of W" from "joint training costs something on W".
python scripts/p31_w_causal.py

# Composition when the policy has only ever seen the rules ALONE: train on the
# single-rule regimes, read every combination zero-shot.
python scripts/p34_single_member.py

# The enumeration comparator: one policy trained on ALL eight regimes, at matched
# compute and at matched per-regime exposure, against the six-regime joint.
python scripts/p37_all8_joint.py

# CP-SAT model size, and the 5 s solver status on the setup regimes.
python scripts/p36_cpsat_modelsize.py

# Out of distribution: setups drawn larger and asymmetric, windows drawn denser.
python scripts/p29_ood.py          # build the shifted benchmarks (refuses to run if the
                                   # published benchmark does not regenerate byte-for-byte)
python scripts/p33_ood_table.py    # the table

# The domain-randomized arm: trained with member magnitudes drawn across the
# shifted ranges, evaluated into the same three campaigns as p33.
python scripts/p43_dr_arm.py

# The deployment artifact per regime: the token-blind policy against the
# conditioned joint and against each specialist, from the main campaign.
python scripts/p45_blind_perregime.py

# Residual optimality gaps at the 30 s and 300 s tiers, from the solver's own bounds.
python scripts/p38_optgap.py

# The solver ladders past the training size (the solves ship under or_solution/;
# p44_cpsat_scale.py re-runs a rung). p39 aggregates the 20x25 and 30x25
# crossover, p60 the 30-minute-per-instance campaign at 50x25 and 80x25, and
# p61 re-solves the one-hour boundary improvements and reports the flip count.
python scripts/p39_anytime_scale.py
python scripts/p60_scale1800.py
python scripts/p61_boundary_flip.py --report

# E3 probes: the conflicting-token cell across all three seeds, environment
# misconfiguration checked by the independent validator, and the first-dispatch
# mask exception, from one fresh rollout campaign.
python scripts/p40_misconfig.py

python scripts/make_family_figures.py
```

## Two things that will bite you

**The CP-SAT baseline is only as strong as the incumbent it starts from.** The seeding
heuristic is therefore part of the baseline's configuration, not an implementation
detail. This campaign warm-starts CP-SAT from the per-instance best of four dispatching
rules (FIFO, MOR, SPT, MWKR) rather than from a single fixed rule, and gives it 8 search
workers on dedicated cores: the budget is wall-clock, and a solver competing for CPU
silently returns worse solutions. Seeding from MWKR alone, which is far behind SPT on the
setup-bearing regimes, understates the solver by a wide margin. The solver is credited
with the better of its warm start and its search, since a hint is a suggestion the search
may fail even to match.

**The greedy rollout is not bit-reproducible across evaluation environments.** It is
deterministic in the instance, the descriptor, and the environment it is evaluated in.
It is not deterministic *across* environments: an arg-max over near-equal action scores
can break the other way and send the rollout down a different path. On one regime this
was worth up to eight hours of makespan on 32 of 100 instances, on identical inputs. Two
policies must therefore be evaluated by the same command on the same machine before
their makespans are compared. `test_results/FAMILY/10x25_cpuref/` is one such campaign,
and it is the one the out-of-distribution and single-rule comparisons are read from. The
dispatching rules have no such sensitivity: their makespans are bit-identical across
every campaign here.

## Relation to prior work

The lag-aware environment, its feature channels, the backbone configuration, and the
guidebook-grounded instances come from a prior study of scheduling under time lags alone
([arXiv:2607.11725](https://arxiv.org/abs/2607.11725)). Nothing above that base layer is
taken from it: the constraint descriptor, the two-hook family environment, the
uniform admissibility theorem, joint training, and every experiment are introduced
here. The paper's supplementary material sets out the division component by component,
and shows that its central theorem *contains* that study's bound as the special case
c = (1, 0, 0), which is what the E0 anchor checks.
