# PACT: One Deep Reinforcement Learning Policy for Flexible Job-Shop Scheduling Under Time Lags, Setups, and Machine Unavailability

This repository is the code and data release of the manuscript **"One Deep Reinforcement Learning Policy for Flexible Job-Shop Scheduling Under Time Lags, Setups, and Machine Unavailability"** by Ziheng Zhang, Wei Zhang, Xin Lou and Hao Zhang. The
manuscript is under review. PACT stands for Policy Across Composable Timing
Rules. The release holds the scheduling environment, the training and
evaluation code, the trained models and their evaluations, and the benchmark
generator with the constraint descriptor of every test instance. It also holds
the analysis chain that computes the statistics and figures of the results
sections from the stored evaluations.

## The problem

A plant changes the timing rules of its flexible job shop when its processes,
products or equipment change. The manuscript studies three rule types in a
factory that builds prefabricated prefinished volumetric construction (PPVC)
modules. A minimum time lag is a wait after an operation, such as concrete
curing, before the next operation of the same job may start. Sequence-dependent
setups are changeovers on a machine, and the length of each depends on the job
class processed before it. During a machine unavailability window, such as planned
maintenance, a machine cannot process. The three rule types give eight rule
combinations, each named by its active rules.

| combination | active rules |
|---|---|
| `N` | none |
| `L`, `S`, `W` | time lags, setups or unavailability windows alone |
| `LS`, `LW`, `SW`, `LSW` | the rules named by the letters, together |

Most reinforcement learning schedulers are trained as specialists for one rule
combination. A rule change then needs another specialist, with its own
training and validation.

In the code, a rule combination is called a regime, for example in the
training option `--family_regimes` and in the descriptor folders
`desc/<regime>/`.

## The contribution

The manuscript proves that one makespan estimate is a lower bound on the
makespan at every partial schedule and equals the makespan at completion,
under each rule combination. Because the estimate at the empty schedule does
not depend on the dispatches, its differences between consecutive dispatches
are rewards that sum to a constant minus the makespan. One environment computes this
reward, and a constraint descriptor sets its transition, action mask and
reward from the active rules and their parameters. Configuration checks test
every changed descriptor before use, and the validator checks every schedule
against the descriptor. One policy trains in this environment on six
combinations, with `SW` and `LSW` held out. With ten training seeds on the
factory benchmark, the policy is equivalent to the specialists at the 1%
margin on seven of the eight combinations. Equivalence means that the 95%
interval of the mean makespan difference lies between minus and plus 1%. The
seven include both held-out combinations, and the eighth, `N`, has no rule
active.

## What is in the release

| path | contents |
|---|---|
| `README.md`, `LICENSE` | this file and the MIT License |
| `train_family.py`, `params.py` | training of the policy and the specialists, and the options of training and evaluation |
| `eval_family.py` | greedy evaluation of a trained model and of the dispatching rules FIFO, MOR, SPT and MWKR |
| `eval_memetic_family.py`, `baselines/` | the memetic algorithm, a genetic algorithm whose offspring a tabu search improves (`baselines/memetic/`, with its `README.md`), and in `baselines/memetic/pilot_out/` the records of its configuration selection |
| `eval_ga_family.py` | a genetic algorithm whose decoder is the environment |
| `ppvc_instance_generator.py` | the generator of the factory benchmark |
| `schedule_validator.py` | a stand-alone feasibility checker of schedules that imports only NumPy |
| `fjsp_env_same_op_nums.py`, `common_utils.py`, `data_utils.py` | the base environment, the dispatching rules, and the reading and writing of instance files |
| `constraint_family/` | the constraint descriptor (`descriptor.py`), the environment with the rule family and the makespan estimate (`fjsp_env_family.py`), the configuration checks C1 to C7 (`checks.py`, C7 is a written procedure), a slow reference simulator for the tests (`reference_sim.py`), the CP-SAT model and the solver configurations (`cpsat_family.py`, `cpsat_arms.py`), and the benchmark from public instances (`public_adapter.py`, `public_rules.py`, `public_types.py`) |
| `model/` | the dual-attention policy network, its PPO training, and the layer of the variant with descriptor input (`film.py`) |
| `backbones/hgnn/` | the heterogeneous graph neural network scheduler of Song et al., ported to the rule family, with its tests in `tests/`, its Apache-2.0 `LICENSE` and a `NOTICE` |
| `scripts/` | the analysis chain `run_analysis.sh` and the scripts of its steps, the figure script `make_figures.py`, the model registry `arm_registry.py`, the evaluation plan `eval_newseeds_plan.py`, the training and evaluation queues of the campaign, `fetch_public_instances.py`, and the figure fonts in `fonts/` |
| `tests/` | pytest files, `conftest.py`, and frozen copies of the environment and the reference simulator in `fixtures/` |
| `data/PPVC/` | base instances of the factory benchmark, `<set>/instance_<i>.fjs` with a `.meta.json` record each and a `dataset_meta.json` per set |
| `data/FAMILY/` | the constraint descriptor of each base instance under each combination, `<set>/desc/<regime>/instance_<i>.desc.json`, and a `dataset_meta.json` per set that names its base folder |
| `data/PUBLIC/` | the sources, checksums, best-known values and descriptors of the benchmark from public instances, without the instance files |
| `trained_network/FAMILY/` | the checkpoints of the models in the next table |
| `train_log/` | for each model, the resolved configuration `FAMILY/config_<model>.json`, the validation record `FAMILY/valiquality_<model>.txt` and the training log `FAMILY_<name>.log` or `PUBFAMILY_<name>.log`, and in `training_cost/` the records of the training-time measurement |
| `test_results/FAMILY/`, `test_results/PUBFAMILY/` | evaluation arrays `Result_<method>_<regime>.npy`, one folder per test set and setting, and the solver records in `test_results/FAMILY/<set>_cparms/` |
| `or_solution/FAMILY/`, `or_solution/PUBFAMILY/` | records of the memetic algorithm, the policy's sampled schedules within a time budget, the schedules of the dispatching rules that start the searches, the warm-start caches, and CP-SAT reference schedules of the benchmark from public instances |
| `results/`, `out/exact/` | the records that the steps of the analysis chain write and read |
| `manuscript/` | the macro files with the numbers of the results, the supplementary tables in `tables/` and the figures in `sections_new/figures/` |

An evaluation array has one row per test instance, which holds the makespan in
hours and the rollout time in seconds. Folders with the suffix `_xmask` hold
evaluations in the environment of the manuscript. Folders without it hold
evaluations in the implementation of the prior lag-only study. Some analyses
read them beside the environment of the manuscript, for example the
comparison of the two environments (Supplementary Section
S16.1).

A model of the factory benchmark is named `10x25+family+<name>`, and a model of
the benchmark from public instances is named `10x5+pubfamily+<name>`. The
suffix `-s<S>` gives the training seed and `<R>` a combination. Section
numbers with the prefix S refer to the supplementary material.

| `<name>` | what the model is | training seeds | manuscript section |
|---|---|---|---|
| `xmask-v1-s<S>` | the policy, trained on `N`, `L`, `S`, `W`, `LS` and `LW` for 2,000 updates | 301 to 310 | VI-A, S10.1, S9.3 |
| `xmask-solo-<R>-s<S>` | a specialist, trained on combination `<R>` alone for 1,000 updates | 301 to 310 | VI-A, S10.1 |
| `xmask-ctl-s<S>` | the single-rule control, trained on `N`, `L`, `S` and `W` for 2,000 updates | 301 to 305 | VI-A, S10.7 |
| `xshare-solo-<R>-s<S>` | a specialist with 250 updates, so that the eight specialists share the policy's training budget | 301 to 310 | VI-A, S10.2 |
| `xlong-solo-<R>-s<S>` | the specialist with 2,000 updates | 301 to 305 | VI-A, S10.3 |
| `xsplit-b-s<S>`, `xsplit-c-s<S>` | the policy with `LS` and `LSW` held out (`b`) or with `LW` and `LSW` held out (`c`) | 301 to 303 | VI-B, S10.8 |
| `xmask-sparse-s<S>`, `xmask-dzero-s<S>`, `xmask-dmax-s<S>` | the policy with the whole return paid at the last step (`--reward_mode sparse`), or with the setup term of the estimate set to zero (`--delta_mode zero`) or to the largest setup (`--delta_mode max`, no lower bound) | 301 to 303 | S11.1 |
| `xmask-eq-s<S>`, `xmask-wthird-s<S>` | the policy with 6,000 updates, and a `W` specialist with 333 updates | 301 to 303 | S11.2 |
| `xmask-film-s<S>`, `xmask-pair-s<S>` | the variant with descriptor input (`--use_film True`), and the policy with two more channels per operation-machine pair (`--pair_extra sw`) | 301 to 303 | S8.5, S11.3 |
| `xmask-all8-s<S>`, `xmask-dr-s<S>` | the policy trained on the eight combinations, and the policy trained with randomized rule ranges (`--dr_shift`) | 301 to 303 | VI-B, S11.4 |
| `xmask-active-s<S>` | the policy trained under the active-schedule mask (`--mask_mode active`) | 301 to 303 | S15.4 |
| `xmask-rc-s<S>` | the policy trained on instances of reinforced-concrete modules alone (`--ppvc_mix rc_project`) | 301 to 303 | VI-D, S16.4 |
| `hgnn-certified-joint-s<S>-xmask`, `hgnn-published-joint-s<S>-xmask` | the heterogeneous graph neural network scheduler of Song et al., trained on the six training combinations with the PACT reward or with its published reward | 301 to 305, and 301 to 303 | V-C, S7.4, S12 |
| `hgnn-certified-<R>-s301-xmask` | the same scheduler trained with the PACT reward on combination `<R>` alone | 301 | S7.4 |
| `jointblind-v1`, `jointblind-nlsw-s<S>`, `spec-<R>` | the policy, the single-rule control and the specialists in the implementation of the prior lag-only study, where the seed-301 runs of the policy and the specialists carry no seed suffix | 301 to 303 | S16.1 |
| `pubx-act-s<S>`, `pubx-soloact-<R>-s<S>` | the policy and the specialists of the benchmark from public instances, under the active mask | 301 to 310 | VI-A, S17.1, S17.2 |
| `pubx-ctlact-s<S>`, `pubx-eqact-s<S>` | the single-rule control and the policy with 6,000 updates, under the active mask | 301 to 305 | S17.1, S17.5 |
| `pubx-v1-s<S>`, `pubx-ctl-s<S>`, `pubx-solo-<R>-s<S>`, `pubx-eq-s<S>` | the same four recipes under the non-delay mask | 301 to 305 | S17.8 |

In the names of the published scheduler and in its option `--hgnn_reward`,
`certified` stands for the PACT reward and `published` for the reward of the
published scheduler. The models of the factory benchmark and the published
scheduler trained on the GPU, and the models of the benchmark from public
instances trained on the CPU. The timed comparisons and the example schedules
use the training run of the policy with the lowest validation score among
seeds 301 to 305. The registry function `select_seed` in
`scripts/arm_registry.py` picks this run, which is `xmask-v1-s304`.

## Environment

The results were produced with Python 3.11.15, PyTorch 2.11.0 with CUDA 12.8,
NumPy 1.26.4, SciPy 1.17.1, OR-Tools 9.14.6206 (CP-SAT), Numba 0.68.0 and
Matplotlib 3.11.0. The tests use pytest 9.1.1. The file `requirements.txt`
pins these versions and names the wheel index of PyTorch with CUDA 12.8. The
memetic algorithm compiles
its decoder with Numba. When the package `gym` is absent,
`backbones/hgnn/compat.py` supplies the parts of it that the upstream files of
the published scheduler import. Supplementary Section
S8.3 describes the workstation that produced every result.

Run every command from the repository root. The analysis chain,
`scripts/run_rule_comparison.sh`, `scripts/run_timed_campaign.sh` and
`scripts/run_training_cost.sh` take the Python interpreter from the variable
`PY`, for example `PY=$(command -v python) DRY=1 scripts/run_analysis.sh`.
In the other shell scripts, the line that sets `PY` names the interpreter.

## Obtaining the public instance files

The instance files of the benchmark from public instances are not part of this
repository. This benchmark uses the Brandimarte set, the Hurink `edata`,
`rdata` and `vdata` sets, and the SD2 test and validation sets at 10x5 and
20x10. `data/PUBLIC/sources.json` names the source of each set, and the
`SHA256SUMS` file of each set folder holds the checksum of every instance
file. Download and verify the files once, before a script or a test reads
them.

```bash
python scripts/fetch_public_instances.py                 # every set of data/PUBLIC/sources.json
python scripts/fetch_public_instances.py --sets Brandimarte SD2/test/10x5+mix
python scripts/fetch_public_instances.py --verify-only   # no network: local files against SHA256SUMS
python scripts/fetch_public_instances.py --archive <local copy of a source archive>
python scripts/fetch_public_instances.py --sd2 generate  # SD2 from the repository's generator, no network
```

The script writes each file into its set folder under `data/PUBLIC/` and checks
it against its checksum. Tests that read these files are skipped with a
message that names this command until the files are present.
`data/PUBLIC/README.md` describes the sets, their best-known values and the
rule combinations built on them.

## Instances, seeds and their separation

Training instances are drawn while a model trains, and they are never written
to disk. For each instance, the trainer draws a generator seed and a
descriptor seed from the random stream that the training seed starts. On the
factory benchmark, the validation set that selects the kept checkpoint has 20
base instances from generator seeds 100 to 119, with descriptor seeds 100000
to 100019, under each training combination. The test instances are fixed files. Their generator
seeds start at `seed0` of `data/PPVC/<set>/dataset_meta.json`, and their
descriptor seeds start at 20260706 (`desc_seed0` of
`data/FAMILY/<set>/dataset_meta.json`). Instance `<i>` of a set uses the
start seed plus `<i>`.

| test set | base instances | generator seeds |
|---|---|---|
| `10x25`, the training size of 10 modules on 25 machines | 100 | 50000 to 50099 |
| `20x25`, `27x25` (the per-floor batch), `30x25` | 50, 50, 20 | 80000, 140000 and 90000 onward |
| `50x25n30`, `80x25n30`, the production batches | 30 each | 100000 to 100029, 110000 to 110029 |
| `50x15`, `80x15`, a factory with 15 machines | 20 each | 120000 and 130000 onward |
| `10x25_asymsetup`, `10x25_oodsetup`, `10x25_oodwindow` | 100 each | the base instances of `10x25`, with other setup matrices or windows |
| `data/PPVC/10x25+ppvc-rc_project`, `data/PPVC/10x25+ppvc-steel_project` | 50 each | 90000 and 95000 onward |

The production batches are frozen files, and the analyses read them as they
ship. Their instances 000 to 019 are byte copies of the 20-instance sets
`50x25` and `80x25`, and instances 020 to 029 continue the seed sequence. At
50 and 80 modules, instances 000 to 002 selected the configurations of the
solver and the memetic algorithm, and instances 003 to 029 carry the
evidence. The timed comparisons run on instances 003 to 011.
`scripts/p86_extend_scale_sets.py` wrote these folders, and its `build`
command refuses to replace a file whose bytes differ unless `--force` is
given. The read-only command
below checks the shipped files against the frozen sets and against a
regeneration in memory.

```bash
python scripts/p86_extend_scale_sets.py verify
```

The training seeds are 301 to 305 for the policy, the single-rule control and
the specialists. Seeds 306 to 310 extend the comparison of the policy with the
specialists to ten seeds (Supplementary Section S9.3).
Runs with the same seed share their initial weights and training instances,
so a comparison of two trained models pairs the runs of each seed.

## Reproducing the results

### The analysis chain

`scripts/run_analysis.sh` runs the analysis steps of the manuscript in a
fixed order. Each step reads stored arrays and records, and the chain starts
no training, no evaluation, no solver and no timed measurement. The one
exception is the step `example_schedule`, which rolls the selected run out
greedily on one test instance on the CPU.

```bash
DRY=1 scripts/run_analysis.sh                              # steps, commands, files and preconditions, nothing written
CORE=<n> scripts/run_analysis.sh                           # every step whose precondition holds, pinned to core <n>
CORE=<n> ONLY="family seed_extension" scripts/run_analysis.sh   # the named steps only
```

Every command of the chain runs with `PACT_ENV=exact PACT_SEEDS=301-305
PACT_OUT=out/exact`, one thread and no GPU. A step writes its records under
`out/exact/` or `results/`, and it writes or copies its macro file into
`manuscript/`. The dry run prints, for each step, the files it writes and
whether its precondition holds. A real run logs each step to
`train_log/analysis_chain/<step>.log`. The timed measurements are never run by
the chain, because their values depend on the processor time they receive.
The chain prints their commands at the end of every run.

A run ends with two lists. `STEP STATUS` gives each selected step with its
state, such as `skipped` when its precondition does not hold. The list of
macro files gives each macro file of `manuscript/` with its number of macros
and the generator named in its header. One file, `manuscript/macros_design.tex`,
is typed by hand. It holds the constants of the training recipe, the test
shops and the timing rule, and step `model_facts` compares each constant with
the configuration or code that its comment names. The release holds no source
of the manuscript, so this list ends with the lines
`build script of the manuscript not found, its copy check is left out:
manuscript/build.sh` and `no source of the manuscript is present (the public
release holds none), so no input and no duplicate macro name is checked`, and
the run exits with 0.

The subsections below list, for each results section of the manuscript, the
steps, their scripts, the macro files they write and the exhibits those files
fill. Scripts are in `scripts/`, and macro files are in `manuscript/`.

### One policy against the specialists (Section VI-A)

| step | script | macro file | exhibits |
|---|---|---|---|
| `example_schedule` | `p123_example_schedule.py` | `macros_example.tex` | Supplementary Fig. S5 |
| `family` | `p11_equivalence.py`, `p27_seed_inference.py`, `p88_family_inference.py --n30 --require-sections all` | `macros_family.tex` | Supplementary Table S4 |
| `seed_extension` | `p110_seed_extension.py` | `macros_seedext.tex` | Supplementary Table S4 |
| `public_act` | `p89_public_family.py --family active` | `macros_pubfam_act.tex` | Supplementary Table S4 |
| `public_seed_extension` | `p115_public_seed_extension.py` | `macros_pubseedext.tex` | text |
| `instance_counts` | `p125_instance_counts.py` | `macros_counts.tex` | text |
| `budget_specialists` | `p127_budget_specialists.py --seeds 301-310` | `macros_budgetspec.tex` | text |
| `long_specialists` | `p117_long_specialists.py --pact-seeds 301-305` | `macros_longspec.tex` | text |
| `model_facts` | `p111_model_facts.py` | `macros_modelfacts.tex` | text |
| `figures` | `make_figures.py --only fq1 --factory-seeds auto` | figure `fq1_specialists` | Fig. 2 |

The step `long_specialists` compares the specialists with 2,000 updates with
the specialists with 1,000 updates and with the policy, on each combination at
the training size. The step `budget_specialists` compares the policy with
eight specialists of 250 updates each, which together use the policy's
training budget.

### Combinations held out of training (Section VI-B)

| step | script | macro file | exhibits |
|---|---|---|---|
| `cross_eval` | `p100_cross_eval.py` | `macros_crosseval.tex` | Supplementary Table S5, Supplementary Fig. S7 |
| `heldout_splits` | `p122_heldout_splits.py --pact-seeds 301-303` | `macros_heldout.tex` | Supplementary Fig. S6 |
| `secondary` | `p95_secondary_inference.py --require all` | `macros_secondary.tex` | text, Supplementary Fig. S6 |
| `figures` | `make_figures.py --only fq2_reuse` and `--only fq2_heldout` | figures `fq2_reuse`, `fq2_heldout` | Supplementary Figs. S7 and S6 |

The step `cross_eval` compares each specialist under each of the eight
combinations with the policy. The step `secondary` compares the supporting
training runs and the published scheduler with the policy. Its precondition holds once
the evaluations of the published scheduler under the exact mask exist, which
`scripts/run_hgnn_exact_queue.sh` trains and `scripts/run_eval_newseeds.sh`
evaluates (group `hgnn`).

### Search methods, dispatching rules and the time budget (Section VI-C)

| step | script | macro file | exhibits |
|---|---|---|---|
| `timed` | `p94_timed_inference.py` | `macros_timed.tex` | Table II, Supplementary Fig. S8 |
| `memetic_parallel` | `p124_memetic_parallel.py --slots 16-23 8-15` | `macros_memetic_parallel.tex` | Table II |
| `memetic_one_minute` | `p137_memetic_one_minute.py --ref-slots 16-23 8-15 --pol-slots 16-23 8-15` | `macros_memetic_minute.tex` | the one-minute rows of Table II |
| `memetic_five_minute` | `p137_memetic_one_minute.py --budget 300 --slots 16-23 --pol-slots 8-15 --ref-slots 16-23 8-15` | `macros_memetic_five.tex` | the five-minute rows of Table II |
| `decision_rows` | `p138_decision_rows.py --ref-slots 16-23 8-15 --pol-slots 16-23 8-15 --five-slots 16-23 --five-pol-slots 8-15` | `macros_decision_rows.tex` | the tie marks of Table II |
| `randomized_rules` | `p135_randomized_rules.py infer` | `macros_randomized_rules.tex` | text of Supplementary Section S2 |
| `memetic_public` | `p136_memetic_public_check.py infer` | `macros_memetic_public.tex` | text of Supplementary Section S2 |
| `family` | as above | `macros_family.tex` | text |
| `figure_timed` | `make_figures.py --only fq3 --policy-time out/exact/results/deploy_cost_cpu_1thread.json --memetic-parallel out/exact/results/memetic_parallel.json --memetic-minute out/exact/results/memetic_one_minute.json --memetic-five out/exact/results/memetic_five_minute.json` | figure `fq3_time_budget` | Supplementary Fig. S8 |

The solver is a large-neighborhood search that solves sub-models with CP-SAT
(`constraint_family/cpsat_arms.py`, configuration `lns`). The steps
`solver_selection` (`p98_solver_selection.py`) and `memetic_selection`
(`p106_memetic_selection.py`) select the configurations of the solver and the
memetic algorithm on instances 000 to 002 of the production batches. Both
searches start from the schedule of the best dispatching rule or from the
policy's schedule. The policy's sampled schedules within a time budget come
from `scripts/p97_policy_anytime.py`, and `scripts/run_timed_campaign.sh` runs
the timed campaign.

The memetic algorithm ran eight times per cell, with seeds 1 to 8, on
instances 003 to 011 of both production batches and the eight combinations.
The runs from the policy's schedule start from the selected training run
`xmask-v1-s304`. The record folders below are in `or_solution/FAMILY/`, and
`<size>` is `50x25n30` or `80x25n30`.

| record folder | records | runs |
|---|---|---|
| `<size>_memetic600s/` | 1,440 | 600 s from the rule schedule; seed 1 also on instances 012 to 029 |
| `<size>_memetic600s_pactwarm_s304/` | 1,152 | 600 s from the rule schedule and the policy's schedule |
| `<size>_memetic300s/` | 1,152 | 300 s from the rule schedule |
| `<size>_memetic300s_pactwarm_s304/` | 1,152 | 300 s from the rule schedule and the policy's schedule |
| `<size>_memetic60s/` | 1,152 | 60 s from the rule schedule |
| `<size>_memetic60s_pactwarm_s304/` | 1,152 | 60 s from the rule schedule and the policy's schedule |

The step `memetic_one_minute` reads the 60-s runs beside the 600-s runs, and
the step `memetic_five_minute` reads the 300-s runs beside them. The step
`decision_rows` recomputes the rows of the decision table and gives, for each
row, the paired difference between its two lowest entries. The step
`randomized_rules` compares randomized dispatching rules on eight cores with
the policy and the searches on the same cells. The step `memetic_public` reads
30 runs of 600 s of the memetic algorithm on the ten Brandimarte instances
without rules, which are in `or_solution/PUBFAMILY/memetic_p136/`, and gives
their gap to the best-known makespans.

### Limits of the policy (Section VI-D)

| step | script | macro file | exhibits |
|---|---|---|---|
| `production_specialists` | `p128_production_specialists.py` | `macros_prodspec.tex` | Supplementary Table S6 |
| `per_floor` | `p118_per_floor.py` | `macros_perfloor.tex` | text |
| `best_of_seven` | `p126_best_of_seven.py` | `macros_bestseven.tex` | text |
| `ood` | `p33_ood_table.py` | `macros_ood.tex` | Supplementary Table S6 |
| `asym_setups` | `p119_asym_setups.py` | `macros_asym.tex` | Supplementary Table S6 |
| `release_dates` | `p69_release_dates.py analyze` | `macros_release.tex` | Supplementary Table S6 |
| `scale_sensitivity` | `p102_scale_sensitivity.py infer --seeds 301-305` | `macros_scalesens.tex` | Supplementary Table S6 |
| `five_classes` | `p103_five_classes.py infer --seeds 301-305 --default_policy run` | `macros_fiveclass.tex` | Supplementary Table S6 |
| `sensitivity` | `p87_sensitivity.py infer --mask exact --arm_names xmask --seeds 301,302,303,304,305` | `macros_sensitivity.tex` | text |
| `project_mix` | `p129_project_mix.py infer --seeds 301-305` | `macros_projectmix.tex` | text |
| `figure_limits` | `make_figures.py --only fq4_limits --fq4-limits-variant A` | figure `fq4_limits_A` | Fig. 3 |
| `rc_trained` | `p131_rc_trained_policy.py infer` | `macros_rctrained.tex` | text |

### Cost of a rule change (Section VI-E)

| step | script | macro file | exhibits |
|---|---|---|---|
| `rule_change_cost` | `p99_rule_change_cost.py count`, then `p99_rule_change_cost.py infer --require all` | `macros_rulechange_cost.tex` | Supplementary Table S7 |
| `deploy` | `p75_deploy_cost.py --from_json out/exact/results/deploy_cost_cpu_1thread.json --force` | `macros_deploy.tex` | Supplementary Table S7 |
| `figures` | `make_figures.py --only fq5_rule_change` | figure `fq5_rule_change` | Fig. 4 |

The step `rule_change_cost` compares one policy with one specialist per
combination. Its command `count` counts what is kept under change control,
which is the models, their parameters, checkpoint bytes, configuration files
and validation rollouts. Its command `infer` adds the training hours and
updates and the time of the computed checks at a rule change. The training
hours come from `scripts/run_training_cost.sh`, which repeats one finished
training run alone on the GPU and its processor cores. That script writes one
record per run under `train_log/training_cost/`, and
`scripts/training_cost_evidence.py` decides whether the record is valid. The
time of the checks comes from `p99_rule_change_cost.py checks`. The step
`deploy` reads the record of the deployment-time measurement of
`scripts/p75_deploy_cost.py`.

```bash
ARM=released DRY=1 scripts/run_training_cost.sh   # prints the command, the names, the checks and the files
```

### Supplementary analyses

| supplementary section | step | script |
|---|---|---|
| S5.4 | `lower_bound_reference` | `p120_lower_bound_reference.py` |
| S7.2 | `solver_selection` | `p98_solver_selection.py` |
| S7.3 | `memetic_selection`, `memetic_public` | `p106_memetic_selection.py`, `p136_memetic_public_check.py infer` |
| S7.4, S11, S12 | `secondary` | `p95_secondary_inference.py` |
| S8.2 | `convergence` | `p116_convergence.py`, `make_figures.py --only fs_convergence` (also Supplementary Fig. S4) |
| S9.3, S10.1 | `seed_extension`, `instance_counts`, `seed_spread` | `p110_seed_extension.py`, `p125_instance_counts.py`, `p112_seed_spread.py` |
| S10.2 | `budget_specialists` | `p127_budget_specialists.py` |
| S10.3 | `long_specialists` | `p117_long_specialists.py` |
| S10.5, S10.7 | `cross_eval`, `family` | `p100_cross_eval.py`, `p88_family_inference.py` |
| S10.8 | `heldout_splits` | `p122_heldout_splits.py` |
| S13.1, S13.2, S13.3 | `per_floor`, `production_specialists`, `family` | `p118_per_floor.py`, `p128_production_specialists.py`, `p88_family_inference.py` |
| S13.4 | `aware_rules` | `p72_aware_rules.py --summary_only` |
| S13.5 | `rulecmp_factory` | `p93_rule_comparison.py --family factory` |
| S14.1 to S14.4 | `timed` | `p94_timed_inference.py` |
| S14.5 | `timed_instance_sets` | `p94_timed_inference.py --tag startnine` and `--tag startall`, `p130_timed_instance_sets.py` |
| S14.6 | `memetic_parallel` | `p124_memetic_parallel.py` |
| S14.7 | `memetic_one_minute`, `memetic_five_minute` | `p137_memetic_one_minute.py`, `p137_memetic_one_minute.py --budget 300` |
| S14.8 | `decision_rows` | `p138_decision_rows.py` |
| S14.9 | `randomized_rules` | `p135_randomized_rules.py infer` |
| S15.1 | `reference_replay` | `p105_reference_replay.py infer` |
| S15.2 | `mask_rules` | `p107_mask_rule_variants.py infer` |
| S15.3 | `mask_controls` | `p101_mask_controls.py infer` |
| S15.4 | `actmask` | `p50_actmask_analysis.py --require all` |
| S16.1 | `environments` | `p96_two_environments.py` |
| S16.2 | `release_dates` | `p69_release_dates.py analyze` |
| S16.3 | `sensitivity` | `p87_sensitivity.py infer` |
| S16.4 | `project_mix`, `rc_trained` | `p129_project_mix.py infer`, `p131_rc_trained_policy.py infer` |
| S16.5 | `scale_sensitivity` | `p102_scale_sensitivity.py infer` |
| S16.6 | `five_classes` | `p103_five_classes.py infer` |
| S16.7 | `ood`, `asym_setups` | `p33_ood_table.py`, `p119_asym_setups.py` |
| S16.8 | `wrong_descriptor` | `p121_wrong_descriptor.py` |
| S17.1, S17.4, S17.5, S17.7 | `public_act` | `p89_public_family.py --family active` |
| S17.2 | `public_seed_extension` | `p115_public_seed_extension.py` |
| S17.3, S17.8 | `public`, `public_act` | `p89_public_family.py` |
| S17.6 | `rulecmp_public_act` | `p93_rule_comparison.py --family public --mask active`, once per test set with `--set` |

### Training and evaluation commands

These commands repeat single runs of the campaign. Each was checked against
the script's `--help` and against the launch line of the campaign's queue
script. The trainer replaces a checkpoint and its logs of the same name without
asking, and the evaluator replaces arrays of the same name. The commands
below use model names and output folders that no shipped file uses, so they
replace no shipped file.

The policy and a specialist train on the GPU (`scripts/run_exactmask_queue.sh`).
The trainer prints the parameter count of the network, which is 32,786 for
both. It writes `trained_network/FAMILY/10x25+family+<suffix>.pth` and the
files of `train_log/FAMILY/`.

```bash
# the policy, training seed 301
python train_family.py --data_source FAMILY --n_j 10 \
    --use_type_embedding True --validate_timestep 25 --use_film False \
    --family_regimes N,L,S,W,LS,LW --max_updates 2000 --seed_train 301 \
    --first_mask exact --mask_ties exact --model_suffix rerun-v1-s301
# the specialist of combination W, training seed 301
python train_family.py --data_source FAMILY --n_j 10 \
    --use_type_embedding True --validate_timestep 25 --use_film False \
    --family_regimes W --max_updates 1000 --seed_train 301 \
    --first_mask exact --mask_ties exact --model_suffix rerun-solo-W-s301
```

The further runs of the factory benchmark trained with the same command and
one changed option each. The specialists `xlong-solo-<R>-s<S>` and
`xshare-solo-<R>-s<S>` take `--max_updates 2000` and `--max_updates 250`. The
policies `xsplit-b-s<S>` and `xsplit-c-s<S>` take
`--family_regimes N,L,S,W,LW,SW` and `--family_regimes N,L,S,W,LS,SW`. The
policy `xmask-rc-s<S>` takes `--ppvc_mix rc_project`. The configuration file
`train_log/FAMILY/config_10x25+family+<name>.json` of each shipped run records
every option of its training.

Evaluation runs on the CPU with one thread (`scripts/run_eval_newseeds.sh`).
The arrays go to `<out_root>/<set><tag>/`, here `rerun_results/10x25_xmask/`.
A specialist is evaluated with the same command, its own model name and its
combination in `--regimes`.

```bash
# the policy, greedy, on the 100 test instances of each combination at the training size
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python eval_family.py \
    --family_model 10x25+family+xmask-v1-s301 --data data/FAMILY/10x25 \
    --base data/PPVC/10x25+ppvc-mixed --regimes N L S W LS LW SW LSW --methods greedy \
    --out_root rerun_results --device cpu --device_id "" --torch_threads 1 \
    --tag _xmask --first_mask exact --mask_ties exact
# the four dispatching rules, restricted to the pairs the mask admits
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python eval_family.py \
    --family_model rules --data data/FAMILY/10x25 \
    --base data/PPVC/10x25+ppvc-mixed --regimes N L S W LS LW SW LSW \
    --methods FIFO MOR SPT MWKR --out_root rerun_results --device cpu --device_id "" \
    --torch_threads 1 --tag _xmask --first_mask exact --mask_ties exact --rule_set masked
```

The memetic algorithm and the solver run with a wall-clock budget, one cell at
a time (`scripts/timed_campaign_plan.py`, `scripts/run_timed_campaign.sh`). A
cell is one test set, one combination and one instance. The memetic algorithm
runs on one core, and the solver runs eight workers on eight cores. Both start
from the warm-start cache of the best dispatching rule, which the release holds
for instances 003 to 011 of the production batches. The memetic algorithm
skips a record that exists unless `--force` is given.

```bash
# the memetic algorithm, 600 s, instance 003 of the 50-module production batch, combination N
taskset -c <core> env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    python eval_memetic_family.py --size 50x25n30 --regimes N --instances 3 --budget 600 \
    --seed 1 --seed_schedules --warm_cache or_solution/FAMILY/warmstart_cache_xmask_50x25n30 \
    --out rerun_results/50x25n30_memetic600s
# the solver, the same cell, eight workers on cores 8 to 15
taskset -c 8-15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    python scripts/p85_cpsat_arms.py --arm lns --size 50x25n30 --regimes N --instances 3 \
    --budget 600 --workers 8 --jobs 1 --seed 1 --slots 8-15 \
    --hint or_solution/FAMILY/warmstart_cache_xmask_50x25n30 --out evidence_rule600s \
    --solver-log --root rerun_results
```

The runs with 60 s and 300 s use the same command with `--budget 60` or
`--budget 300`. A run from the policy's schedule adds the stored schedules of
the selected training run with `--seed_from`. The planner prints the command
of every cell of a campaign arm, and its option `--policy-seed` names the
training run whose schedules start the search. With `--result-root`, the
printed commands write their records under another folder.

```bash
# the memetic algorithm, 60 s, the same cell, from the rule schedule
taskset -c <core> env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    python eval_memetic_family.py --size 50x25n30 --regimes N --instances 3 --budget 60 \
    --seed 1 --seed_schedules --warm_cache or_solution/FAMILY/warmstart_cache_xmask_50x25n30 \
    --out rerun_results/50x25n30_memetic60s
# the same, also from the policy's schedule of the training run xmask-v1-s304
taskset -c <core> env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
    python eval_memetic_family.py --size 50x25n30 --regimes N --instances 3 --budget 60 \
    --seed 1 --seed_schedules --warm_cache or_solution/FAMILY/warmstart_cache_xmask_50x25n30 \
    --seed_from or_solution/FAMILY/pactwarm_cache_xmask_s304_50x25n30 \
    --out rerun_results/50x25n30_memetic60s_pactwarm_s304
# prints the command of every cell of the 300-s runs from the policy's schedule, seed 1, and starts nothing
python scripts/timed_campaign_plan.py plan --arms 'memetic_policy@300:3-11' \
    --sizes 50x25n30 80x25n30 --policy-seed 304 --memetic-seed 1 \
    --result-root rerun_results --all-commands
```

The published scheduler trains on the GPU and is evaluated on the CPU
(`scripts/run_hgnn_exact_queue.sh`, `scripts/run_eval_newseeds.sh`). Its
trainer refuses to replace an existing checkpoint unless `--force` is given,
and `--out_root` writes the run under another folder. The learning rate
2e-4 was selected from 1e-4, 2e-4 and 4e-4 on the validation score
(`scripts/hgnn_select_lr.py`).

```bash
# trained with the PACT reward on the six training combinations, training seed 301
python backbones/hgnn/train_family_hgnn.py --family_regimes N,L,S,W,LS,LW \
    --hgnn_reward certified --hgnn_mask exact --max_iterations 2000 --seed 301 \
    --lr 2e-4 --model_suffix xmask --device cuda --threads 1 --out_root rerun_hgnn
# greedy evaluation of the shipped run on the eight combinations at the training size
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python backbones/hgnn/eval_family_hgnn.py \
    --family_model 10x25+family+hgnn-certified-joint-s301-xmask --data_root data/FAMILY \
    --sizes 10x25 --regimes N,L,S,W,LS,LW,SW,LSW --device cpu --threads 1 \
    --out_root rerun_results --tag _xmask
```

## Tests

```bash
python -m pytest tests -q -p no:cacheprovider
python -m pytest backbones/hgnn/tests -q -p no:cacheprovider
```

The tests run on the CPU. `tests/conftest.py` hides the GPU and runs every test
file as it runs alone, in any order. The tests check the following parts of
the release.

| part | test files |
|---|---|
| the environment, its masks and the makespan estimate against an independent reference simulator and the validator | `test_exact_mask.py`, `test_first_mask.py`, `test_setup_mult.py`, `test_release.py`, `test_masked_rules.py`, `test_instance_bound.py` |
| the configuration checks C1 to C7 | `test_checks.py`, `test_p121_wrong_descriptor.py` |
| the solver, the memetic records, the timed campaign and the validity rule of timed records | `test_cpsat_arms.py`, `test_p85_memory_gate.py`, `test_p92_warmstart_training_size.py`, `test_p97_policy_anytime.py`, `test_p98_solver_selection.py`, `test_p106_memetic_selection.py`, `test_timed_campaign_plan.py`, `test_p94_timed_inference.py`, `test_p130_timed_instance_sets.py`, `test_p124_memetic_parallel.py`, `test_p137_memetic_one_minute.py`, `test_p138_decision_rows.py`, `test_p135_randomized_rules.py`, `test_p136_memetic_public_check.py`, `test_timing_validity.py` |
| the statistics, on synthetic data with known answers | `test_stats_family.py`, `test_sign_and_verdict.py` |
| the steps of the analysis chain, on synthetic arrays and on the shipped records | the other `test_p*.py` files, `test_cross_eval.py`, `test_seed_spread.py`, `test_model_facts.py`, `test_rule_change_cost.py`, `test_aware_rules_*.py`, `test_release_dates_exact.py` |
| the figures | `test_fq*.py` |
| the evaluation plan and the queues | `test_eval_newseeds_plan.py`, `test_gpu_queue_lanes.py`, `test_further_sets_eval.py`, `test_production_specialists_eval.py`, `test_pubactfam_eval_plan.py`, `test_latency_threads.py` |
| the benchmark from public instances | `test_fetch_public_instances.py`, `test_public_*.py`, `test_pubeq_arm.py` |
| the published scheduler: its action sets under the exact mask against the environment, its port and its options | `backbones/hgnn/tests/` |

Some tests skip in this release, because they need files that it does not
hold. These are the sources and tables of the manuscript, the git history of
the working repository, and some stored arrays, configurations and summaries
of the prior implementation. They also include the warm-start schedules of an
earlier solver campaign, the checkpoint of the upstream scheduler and the
checksum list of one schedule cache. The tests of the benchmark from public
instances skip as well until `python scripts/fetch_public_instances.py` has
fetched the instance files. Each skipped test names the file it needs, and
`python -m pytest tests -q -rs` prints these reasons.

## Notes on use

**The environment of the manuscript is set by two options.** Training and
evaluation take `--first_mask exact --mask_ties exact`. The first applies the
non-delay mask at the first decision of an episode as at every later one, and
the second admits every pair that ties the earliest start in hours. Without
the options, the trainer uses the mask of the implementation of
the prior lag-only study. The evaluator then takes the setting from the
training configuration of the loaded model, and a run of the dispatching rules
alone uses the mask of the prior implementation. The variables
`PACT_FIRST_MASK` and `PACT_MASK_TOL` set the same two corrections, and the
trainer of the published scheduler refuses to start while either is set. A
step script run outside the chain reads the registry default, which is the
prior implementation with seeds 301 to 303. Set
`PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact`, as the chain does, to
read the setting of the manuscript.

**Makespans repeat only in the same evaluation setting.** Every array of the
`_xmask` folders comes from the CPU with one torch thread and one OpenMP and
MKL thread. A different thread count or device can change some makespans of a
greedy rollout. Compare two models only on arrays from the same setting.
Training on the GPU does not repeat exactly from the same seed, so a retrained
model differs from the shipped checkpoint.

**Results under a time budget are timing measurements.** The makespan that
the solver or the memetic algorithm returns depends on the processor time it
receives. Run each such job on cores that no other process uses for its whole
duration. Every timed record holds its CPU share, which is CPU seconds over
wall-clock seconds times workers, and its core affinity.
`scripts/timing_validity.py` states the rule that decides whether a record is
valid. A record of the memetic algorithm needs a share of at least 0.9. A
record of the solver needs a share of at least 0.9, a proof of optimality, or
a share no more than 0.05 below the 5th percentile of its cell type. The
timed campaign reads the frozen percentiles from
`test_results/FAMILY/timed_campaign_share_bars.json`.

## Relation to prior work

This work uses the publicly released lag-aware environment, backbone
configuration and instances on guidebook routes of a prior study of scheduling
under time lags alone ([arXiv:2607.11725](https://arxiv.org/abs/2607.11725)).
That backbone is the dual-attention network of Wang et al. (2024). This work adds setups and
windows to the transition and the mask, the minimum-setup term of the makespan
estimate, training across combinations, and the configuration checks. Its
environment applies the mask at every decision as defined, which differs from
the prior released code in two details (Supplementary Section
S4.3).

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
  title  = {One Deep Reinforcement Learning Policy for Flexible Job-Shop
            Scheduling Under Time Lags, Setups, and Machine Unavailability},
  note   = {Manuscript under review},
  year   = {2026}
}
```
