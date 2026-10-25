# PACT: Knowledge-driven reinforcement learning with a certified reward for flexible job-shop scheduling under changing timing rules

Artifacts for the manuscript **"PACT: Knowledge-driven reinforcement learning
with a certified reward for flexible job-shop scheduling under changing timing
rules"**
by Ziheng Zhang, Wei Zhang, Xin Lou and Hao Zhang, submitted to *Advanced
Engineering Informatics*. PACT stands for Policy Across Composable Timing Rules.

## The problem

A factory for prefabricated prefinished volumetric construction (PPVC) builds
room-sized building modules on a shared set of stations, and its schedule must
obey timing rules. This work studies three rule types. A minimum time lag is a
mandatory wait after an operation, such as concrete curing, before the module's
next operation may start. A sequence-dependent setup is a changeover on a station
whose length depends on the product that ran before it. A machine unavailability
window is an interval, such as planned maintenance, in which a station cannot
run. Which of these rules apply depends on the process, the product mix and the
equipment, so one plant meets different combinations of them over its life. The
eight<!-- \NRegimes, paper/macros.tex --> combinations, including the shop with
no rule active, are called regimes and are named by their active rules.

| regime | active rules |
|---|---|
| `N` | none |
| `L`, `S`, `W` | time lags, setups, or availability windows alone |
| `LS`, `LW`, `SW`, `LSW` | the rules named by the letters, together |

A constructive scheduler trained by deep reinforcement learning places one
operation per decision, and it is normally validated for one regime. A rule
change alters when jobs and stations become ready, which assignments are legal,
and the makespan estimate from which the policy receives its reward after every
dispatch. A policy validated under one regime is therefore not justified under
another, and the specialist approach keeps one validated policy per regime.

## The contribution

PACT records the active timing rules and their parameters in an explicit,
machine-readable constraint descriptor. The descriptor configures one shared
scheduling environment, which applies every rule through one of two hooks. One
hook sets when a job's next operation becomes ready, and the other sets when a
station's next block of work may start. From the same descriptor, the
environment computes a makespan estimate that the paper proves to be a lower
bound at every partial schedule and exact at completion, for the three rules and
all their combinations. The stepwise decrease of this estimate is therefore one
certified dense reward for every regime. With this reward, one policy trained
on six<!-- \NTrainRegimes, paper/macros.tex --> regimes schedules all eight,
including `SW` and `LSW`, which it never met in training. The released policy
does not read the descriptor, and a conditioned variant that reads it is
evaluated as a control. A membership test states when a further rule can join
the family, and release dates are added through this test without retraining.
A descriptor error would change the transition, the legal actions and the reward
at once, so configuration checks C1 to C5 test every revised descriptor before
use. Check C6, an independent validator, then tests every schedule against the
descriptor.

## What is in the release

| path | contents |
|---|---|
| `constraint_family/` | the descriptor (`descriptor.py`), the two-hook environment and certified estimate (`fjsp_env_family.py`), checks C1 to C6 (`checks.py`, with C7 as a written procedure), a reference simulator (`reference_sim.py`), the CP-SAT model (`cpsat_family.py`), the blocking shop (`fjsp_env_blocking.py`) and the port to ReSched (`resched_port/`) |
| `model/` | the dual-attention policy network and the feature-wise linear modulation (FiLM) layer of the conditioned variant |
| top level | `train_family.py`, `eval_family.py` (policies, dispatching rules, descriptor-token interventions), `eval_blocking.py`, `eval_ga_family.py` (genetic algorithm), `ppvc_instance_generator.py` (benchmark generator), `schedule_validator.py` (stand-alone validator), and the modules they import |
| `scripts/` | analysis scripts, each naming its outputs in its docstring |
| `backbones/hgnn/` | the heterogeneous graph neural network (HGNN) scheduler of Song et al. (2023), ported to the family, with its Apache-2.0 `LICENSE` and a `NOTICE` |
| `data/PPVC/` | base test instances by size, `<modules>x<stations>`, `10x25` to `80x25` and two 15-station sets |
| `data/FAMILY/` | one descriptor per instance and regime (`<size>/desc/<regime>/`), and the shifted-magnitude sets |
| `trained_network/` | every evaluated checkpoint (table below), and in `PPVC/` the prior study's lag-only policy |
| `test_results/` | per-instance results `Result_<method>_<regime>.npy` (makespan in hours, rollout seconds), one folder per campaign, and in `PPVC/` the prior study's evaluation |
| `or_solution/FAMILY/` | CP-SAT solutions per instance by size and budget (`<size>_fairwarm<budget>s`, with `R` marking re-solves from the repaired warm start), and the warm-start caches |
| `results/` | summaries the scripts write and read |
| `paper/` | every `macros_*.tex` file and every generated table in `paper/tables/` |
| `train_log/` | each checkpoint's resolved configuration (`FAMILY/config_<name>.json`) and training log |

Each size has its main campaign in the folder of that name. In
`10x25_cpuref` and `10x25_blindref`, every artifact of one equivalence
comparison was evaluated by one command on one machine. `10x25_cpuref` serves
the shifted-magnitude study and the single-rule control of the conditioned
architecture, and `10x25_blindref` serves the single-rule control of the
released architecture. The other folders hold the shifted-magnitude and
misconfiguration runs and a CP-SAT no-wait probe.

Checkpoints are named `10x25+family+<name>.pth`, and `-s30x` stands for the
three training seeds 301, 302 and 303.<!-- \SeedList, paper/macros.tex --> A name without a seed suffix, such as
`jointblind-v1`, is the seed-301 model.

| `<name>` | what it is |
|---|---|
| `jointblind-v1`, `-s302`, `-s303` | the released PACT policy, which does not read the descriptor |
| `joint-v1`, `-s302`, `-s303` | the conditioned variant, which reads the descriptor through FiLM |
| `spec-<regime>`, `-s302`, `-s303` | the per-regime specialists |
| `jointblind-eq-s30x` | the released architecture at the equal-experience budget |
| `jointblind-nlsw-s30x`, `joint-nlsw` | single-rule controls trained on `N`, `L`, `S`, `W` only, released and conditioned architecture |
| `spec-W-third` | a `W` specialist with the released policy's experience on `W` |
| `joint-all8-*`, `joint-all8e-*` | trained on all eight regimes, at matched updates or matched experience |
| `joint-dr-*` | the conditioned variant trained with randomized rule magnitudes |
| `joint-deltazero-*`, `joint-deltamax-*` | the trivial and the maximum reward envelopes |
| `joint-concat-*` | the descriptor read by direct concatenation |
| `jointpair-*`, `jointpairfilm-*` | pair-level setup and window information in the observation |
| `jointact-*` | the active-schedule action mask in place of the non-delay mask |
| `jointblind-mb-s301` | the released architecture at the catalogue's total update budget |
| `hgnn-certified-joint-s30x`, `hgnn-published-joint-s30x` | one HGNN policy over the training regimes, with the certified or the published reward |
| `hgnn-certified-<regime>-s301` | the HGNN per-regime specialists |

**Which PACT checkpoint to deploy.** `jointblind-v1`, `jointblind-v1-s302` and
`jointblind-v1-s303` are three training runs of one configuration. They share
the architecture, hyperparameters, training regimes, budget and validation
set, and differ only in the training seed, which sets the initial weights, the
instances drawn during training and the action sampling. The paper reports
their per-instance mean. At the training size the three runs are close, and
the spread grows with the shop: at 80 modules seed 301, the checkpoint without
a suffix, is the weakest of the three. For deployment, run all three on the
instance in parallel, which takes the latency of one rollout, and keep the
shortest schedule. Every schedule is validated, so the choice uses only its
makespan. `scripts/p82_best3.py` evaluates this option and builds the
best-of-three warm starts for CP-SAT.

## Environment

The results were produced with Python 3.11, PyTorch 2.11 (CUDA 12.8), NumPy
1.26, SciPy, OR-Tools 9.14.6206 (CP-SAT) and Matplotlib. Training used one
NVIDIA RTX PRO 5000 GPU. Evaluation, the solvers and the analyses run on the
CPU. The unmodified upstream HGNN scripts also import pandas, and
`backbones/hgnn/compat.py` supplies the `gym` and `pynvml` modules they expect.

## Instance separation and seeds

Validation and test instances are disjoint by construction. Training instances
are generated online from fresh 31-bit random seeds and are never written to
disk. The 20 validation instances used for model selection come from generator
seeds 100 to 119. The 100<!-- \nTest, paper/macros.tex --> test instances per
regime at the training size are fixed files from generator seeds 50000 to 50099,
with descriptor seed 20260706.<!-- seed0 and desc_seed0 in the 10x25 dataset_meta.json files -->
Each size's `dataset_meta.json` records its seeds. The 20- and
30-module test sets hold 50<!-- \nTestTwenty, paper/macros.tex --> and
20<!-- \nTestThirty, paper/macros.tex --> instances per regime. At 50 and 80
modules, the paper reads the first 10<!-- \ScPrefixN, paper/macros_scale1800.tex -->
instances per regime in sorted order.

The released policy, every specialist and every three-seed control are trained
with seeds 301, 302 and 303.<!-- \SeedList, paper/macros.tex --> Every
comparison pools the same seeds on both sides.

## Reproducing the results

Every number in the manuscript comes from a `paper/macros_*.tex` file or a
table in `paper/tables/`, which the scripts below write. The release ships the
authors' copies, so a rerun is checked by diffing `paper/`. Section, table,
figure and experiment (E1 to E7) labels follow the submitted manuscript, and
"S" marks a supplementary section. A new training run can be diffed against
the checkpoint's configuration in `train_log/FAMILY/`.

```bash
# released policy, seed 301 (the conditioned variant sets --use_film True)
python train_family.py --data_source FAMILY --n_j 10 --use_type_embedding True \
    --validate_timestep 25 --use_film False --max_updates 2000 \
    --seed_train 301 --model_suffix jointblind-v1
# a specialist on one regime
python train_family.py --data_source FAMILY --n_j 10 --use_type_embedding True \
    --validate_timestep 25 --family_regimes L --max_updates 1000 --model_suffix spec-L
# greedy evaluation and the four textbook dispatching rules on all regimes
python eval_family.py --family_model 10x25+family+jointblind-v1 \
    --regimes N L S W LS LW SW LSW --methods greedy FIFO MOR SPT MWKR
```
<!-- 2000 and 1000 updates: \ExpRelUpdates and \ExpSpecUpdates, paper/macros_eqexp.tex -->

### Reproduction check (Section 7.3)

```bash
python scripts/p1_tests.py     # unit and reward tests, and the lag-only reproduction (T5)
```

### One policy against the specialists (Section 8.1)

```bash
python scripts/p11_equivalence.py              # E1 and E2 verdicts (Table 4, S18)
python scripts/p55_equal_experience.py         # equal-experience budget (Table 4)
python scripts/p34_single_member.py --arm jointblind-nlsw   # S21
python scripts/p34_single_member.py            # conditioned architecture (S21)
python scripts/p54_coverage_lag.py             # writes results/coverage_lag.json
python scripts/p63_rulechange_table.py         # rule changes (Table 5)
python scripts/p69_release_dates.py --analyze-only   # release dates (S22)
```

`p11` tests paired equivalence within a ±1%<!-- \EqMargin, paper/macros.tex -->
margin fixed before the runs. `p69 --analyze-only` reads
`results/release_dates_raw.json`, and without the flag it reruns every rollout.
`p27_seed_inference.py` gives the seed-level and margin-sweep verdicts (S10,
S30.1), and `p31_w_causal.py` the shortfall on `W` (S7).

### Time budgets and shop size (Section 8.2)

```bash
python scripts/p13_e5_table_fair.py   # fixed-budget comparison, E5 macros
python scripts/p62_e5_table_aei.py    # Table 6, with the one-second rivals and HGNN
python scripts/p30_anytime.py         # CP-SAT budget ladder (Fig. 5(a), S4)
python scripts/p52_ga_ladder.py       # GA at short budgets
python scripts/p39_anytime_scale.py   # CP-SAT crossover at 20 and 30 modules
python scripts/p56_ga_scale.py        # GA at 10 to 30 modules
python scripts/p18_e6_table.py        # E6 scale transfer (S14)
python scripts/p53_load_axis.py       # against the specialists (Fig. 6, S20)
python scripts/p60_scale1800.py       # 30-minute CP-SAT at 50 and 80 modules
python scripts/p70_prodbatch_table.py # Table 7
python scripts/p82_best3.py --analyse # best of the three checkpoints (Section 8.2)
python scripts/p81_hybrid.py          # CP-SAT seeded with the policy's schedule (S15)
python scripts/p72_aware_rules.py     # constraint-aware rules (S5)
python scripts/p76_aware_table.py     # their tables
python scripts/p75_deploy_cost.py     # deployment cost (S16)
python scripts/make_family_figures.py # Figs. 5 and 6
```
<!-- 30 minutes: \ScBudgetMin, paper/macros_scale1800.tex -->

`p75` is a timing measurement, so run it on four pinned cores with
`OMP_NUM_THREADS=4`, as its docstring shows.

To rerun the shipped solver campaign, build the warm-start cache, repair it,
and solve on dedicated cores.

```bash
python scripts/p12_warm_cache.py --procs 16         # best of four rules
python scripts/p41_repaired_warm_cache.py           # repaired warm starts
python scripts/p12_cpsat_fairwarm.py --budget 300   # one 10x25 tier
python scripts/p44_cpsat_scale.py --size 30x25 --phase cache --procs 16
python scripts/p44_cpsat_scale.py --size 30x25 --phase solve --budget 30 --workers 8 --jobs 2
python scripts/p61_boundary_flip.py --report        # 30-minute boundary cases
python scripts/p38_optgap.py                        # gaps from solver bounds
python scripts/p36_cpsat_modelsize.py               # model size
```

The production-batch hybrid starts from the policies' own schedules. Export
them with `eval_family.py --save_schedules <dir>` for each of the three
checkpoints, build the best-of-three cache, and run the hybrid cells. The runner
solves each cell in its own process, lets only one 80-module model with setups
run at a time, and records the timing evidence that the audit checks.

```bash
python scripts/p82_best3.py --cache                                # best-of-three warm starts
python scripts/p80_cpsat_rerun.py --phase run --slots 0-15 0-15    # hybrid cells
python scripts/p80_cpsat_rerun.py --phase audit                    # timing validity
python scripts/p46_idle_quant.py --budget 30 --workers 8 --jobs 2  # non-delay audit (S8)
python scripts/p50_actmask_analysis.py                             # its macros and table
```

`p41` reads the 10x25 cache `or_solution/FAMILY/warmstart_cache`, which ships
and which `p12_warm_cache.py` rebuilds. The 5 s and 30 s re-solves from the
repaired warm start are run by `scripts/run_fairwarm_repaired.sh`, which calls
`p12_cpsat_fairwarm.py` with the repaired cache.

### The published HGNN scheduler as a learned rival (Section 8.2, S6)

```bash
CUDA_VISIBLE_DEVICES= python backbones/hgnn/tests/test_family_hgnn.py
python backbones/hgnn/train_family_hgnn.py --hgnn_reward certified --seed 301
python backbones/hgnn/train_family_hgnn.py --hgnn_reward published --seed 301
python backbones/hgnn/train_family_hgnn.py --family_regimes L --seed 301   # a specialist
python backbones/hgnn/eval_family_hgnn.py --family_model 10x25+family+hgnn-certified-joint-s301
python scripts/p78_hgnn_rival.py
```

The acceptance tests check the ported environment against the reference
simulator in every regime and against the upstream environment in regime `N`.
They also assert that its legal action set equals the PACT policy's mask. The
trainer keeps the upstream network of 204,250<!-- \HgnnParams, paper/macros_hgnn.tex -->
parameters and its training recipe. `backbones/hgnn/tools/run_upstream_test.py`
runs the unmodified upstream `test.py` on the CPU.

### Use of the descriptor and the reward (Section 8.3)

```bash
python scripts/p17_e3_multiplicity.py   # E3 token interventions (S12)
python scripts/p40_misconfig.py         # misconfigured stacks (S12)
python scripts/p51_info_arms.py         # pair-level timing information (S9)
python scripts/p22_admissibility.py     # E7 reward envelopes (S13)
python scripts/p24_envelope_fuzz.py     # envelope audit
python scripts/p28_exposure_and_certification.py
```

### Limits of the method (Section 8.4)

```bash
python eval_blocking.py --regime L --limit 100 --family_model 10x25+family+jointblind-v1
python scripts/p29_ood.py --build     # shifted-magnitude descriptors
python scripts/p33_ood_table.py       # shifted magnitudes (S23)
python scripts/p43_dr_arm.py          # randomized-magnitude arm (S23)
```

`eval_blocking.py` writes `paper/macros_e4.tex`, and the shipped result was
produced with the model named in the command. `p29_ood.py --build` refuses to
run unless the published benchmark regenerates exactly. `--build --eval` also
runs the evaluations.

### Supporting analyses

```bash
python scripts/p73_effect_sizes.py      # effect sizes (S11)
python scripts/p74_gantt.py             # schedule example (S2)
python scripts/p77_gantt_macros.py      # its makespans
python scripts/p64_instance_bound.py    # instance lower bound (Section 5.3)
python scripts/p65_check_counts.py      # configuration checks (Table 2)
python scripts/p49_resched_crosscheck.py   # port to ReSched (Section 6.2)
python scripts/make_route_figure.py     # Fig. 4
```

`p49` needs the ReSched release of Xiao et al. at `external/ReSched/`, which is
not part of this repository.

## Two things that will bite you

**The CP-SAT baseline is only as strong as the incumbent it starts from.** The
warm start is therefore part of the baseline's configuration. Each solve starts
from the per-instance best of four dispatching rules (FIFO, MOR, SPT, MWKR),
repaired so that every rule selects inside the policy's action mask. Seeding from
MWKR alone would understate the solver by a wide margin, because MWKR is far
behind SPT on the setup-bearing regimes. The solver is credited with the better
of its warm start and its search, because a hint is a suggestion the search may
fail to match. The hint assigns every model variable, so the solver accepts it
as a complete feasible solution and its first incumbent has the warm start's
makespan. The budget is wall-clock, so two solves run at a time on 16 reserved
cores, each with 8<!-- \CpWorkers, paper/macros.tex --> search workers. A
solver that competes for CPU returns worse solutions and reports no error, so
every re-solved cell records its CPU share (CPU seconds over wall-clock seconds
times workers) and its core affinity. `scripts/timing_validity.py` states the
rule the audit applies: a cell is valid at a share of at least 0.9, after an
optimality proof, or within 0.05 of the 5th percentile of its cell type, whose
bars are frozen in `or_solution/FAMILY/share_thresholds.json`.
Short budgets and the largest setup models spend a fixed single-threaded
presolve inside the budget, which lowers their share without contention.

**The greedy rollout is not bit-reproducible across evaluation environments.**
It is deterministic within one evaluation environment. On another device or
thread count, an arg-max over near-equal action scores can break the other way
and send the rollout down a different path, and S23.1 measures the size of this
effect on regime `N`. Two policies must
therefore be evaluated by the same command on the same machine before their
makespans are compared. The dispatching rules have no such sensitivity.

## Relation to prior work

The lag-aware environment, the backbone configuration and the
guidebook-grounded instances come from a prior study of scheduling under time
lags alone, which is a public preprint
([arXiv:2607.11725](https://arxiv.org/abs/2607.11725)). That backbone is the
dual-attention network of Wang et al. (2024). The descriptor, the two-hook
environment, the family-wide proof, the configuration checks and joint training
are introduced in this work. The central theorem contains the prior study's
bound as the lag-only case, and the reproduction check confirms this reduction. The supplementary material
restates the base process, the solver model and the generator, so the paper can
be reproduced without the prior study.

The code in `backbones/hgnn/` is adapted from
[songwenas12/fjsp-drl](https://github.com/songwenas12/fjsp-drl) (commit
`d7637b70029be6cb8fe78e450dcb07767876daea`) of Song et al., *IEEE Transactions
on Industrial Informatics* 19(2):1600-1610, 2023, doi:10.1109/TII.2022.3189725.
That code is licensed under the Apache License, Version 2.0, and
`backbones/hgnn/NOTICE` lists every changed and added file.

## License and citation

The code of this repository is released under the MIT License (see `LICENSE`).
The files in `backbones/hgnn/` remain under the Apache License, Version 2.0 of
their upstream repository.

The manuscript is under review and has no DOI yet. Until it is published,
please cite it as follows.

```bibtex
@unpublished{zhang2026pact,
  author = {Zhang, Ziheng and Zhang, Wei and Lou, Xin and Zhang, Hao},
  title  = {{PACT}: Knowledge-driven reinforcement learning with a certified
            reward for flexible job-shop scheduling under changing timing rules},
  note   = {Manuscript submitted to Advanced Engineering Informatics},
  year   = {2026}
}
```
