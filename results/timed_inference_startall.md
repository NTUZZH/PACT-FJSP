# Timed campaign against the policy (full, tag startall)

- setting: environment exact, training seeds 301,302,303,304,305 (5 runs per arm); policy: selected training run s304; CP-SAT configuration: lns
- clock origin: start (wall clock from the start of the run: memetic algorithm from the start of the process that solves the instance (instance loading, start schedules and numba warm-up included); CP-SAT model arms from the start of the arm, model building included (instance load and hint validation before it not timed by the runner); the neighbourhood-search arm from its start)
- budgets 60, 300, 600 s; verdict words at the stated budget of 600 s only; B = 20000; differences are first-named minus second in percent of the second; the interval of a selected run is the instance-level cluster bootstrap, of every run the crossed hierarchical bootstrap
- campaign: 50x25n30 80x25n30 = memetic_rule@600:3-29 cpsat_rule@600:3-29
- instances per size and arm (the instance range of a token is the design of its arm; a token without one plans the size's evidence instances): 50x25n30 memetic_rule 3-29 (27), cpsat_rule 3-29 (27); 80x25n30 memetic_rule 3-29 (27), cpsat_rule 3-29 (27)
- cells 864, valid 864
- pooled rows beside Pool: the combinations without setups (N, L, W, LW); the combinations with setups (S, LS, SW, LSW), each over the instances present in every combination of its group, with the estimator of Pool

## 50x25n30 (instances 3-29; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +2.49 [+0.47, +4.68] behind | +9.02 [+7.30, +10.91] behind | +10.34 [+8.63, +12.24] behind |
| L | +0.47 [-0.41, +1.33] neither | +6.56 [+5.54, +7.56] behind | +8.34 [+7.46, +9.20] behind |
| S | -1.49 [-2.45, -0.52] ahead | +3.26 [+2.47, +4.16] behind | +4.19 [+3.41, +5.11] behind |
| W | -0.72 [-2.51, +1.22] neither | +2.21 [+0.58, +3.92] behind | +3.44 [+1.92, +5.05] behind |
| LS | -1.19 [-1.95, -0.40] ahead | +4.38 [+3.59, +5.24] behind | +5.25 [+4.46, +6.11] behind |
| LW | -2.64 [-3.48, -1.85] ahead | +1.25 [+0.20, +2.29] behind | +3.06 [+2.01, +4.05] behind |
| SW | -1.26 [-2.25, -0.22] ahead | +1.77 [+0.86, +2.84] behind | +2.72 [+1.81, +3.77] behind |
| LSW | -2.37 [-3.19, -1.54] ahead | +1.91 [+1.17, +2.70] behind | +2.90 [+2.19, +3.66] behind |
| Pool | -0.84 [-1.39, -0.24] ahead | +3.79 [+3.27, +4.34] behind | +5.03 [+4.54, +5.55] behind |
| combinations without setups | -0.10 [-1.01, +0.90] neither | +4.76 [+3.94, +5.61] behind | +6.30 [+5.52, +7.12] behind |
| combinations with setups | -1.58 [-2.13, -1.03] ahead | +2.83 [+2.26, +3.47] behind | +3.77 [+3.17, +4.44] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +3.83 [+1.89, +5.95] behind | +9.68 [+7.85, +11.72] behind | +10.64 [+8.86, +12.59] behind |
| L | +0.68 [-0.12, +1.51] neither | +8.04 [+7.23, +8.89] behind | +9.48 [+8.76, +10.23] behind |
| S | -3.77 [-4.86, -2.72] ahead | -1.34 [-2.34, -0.30] ahead | -0.48 [-1.41, +0.53] neither |
| W | +2.87 [+1.20, +4.74] behind | +5.87 [+4.36, +7.59] behind | +6.86 [+5.34, +8.56] behind |
| LS | -3.40 [-4.15, -2.57] ahead | -1.88 [-2.61, -1.09] ahead | -0.83 [-1.49, -0.08] ahead |
| LW | +0.22 [-0.51, +0.90] neither | +5.08 [+4.29, +5.77] behind | +6.47 [+5.81, +7.09] behind |
| SW | -2.45 [-3.56, -1.31] ahead | -0.29 [-1.37, +0.89] neither | +0.28 [-0.78, +1.41] neither |
| LSW | -3.52 [-4.51, -2.50] ahead | -2.07 [-2.90, -1.19] ahead | -1.47 [-2.28, -0.63] ahead |
| Pool | -0.69 [-1.25, -0.10] ahead | +2.89 [+2.41, +3.41] behind | +3.87 [+3.38, +4.40] behind |
| combinations without setups | +1.90 [+1.02, +2.89] behind | +7.17 [+6.41, +8.03] behind | +8.36 [+7.63, +9.19] behind |
| combinations with setups | -3.28 [-3.93, -2.66] ahead | -1.40 [-2.02, -0.74] ahead | -0.62 [-1.24, +0.02] neither |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -0.18 [-1.89, +1.72] neither | +6.21 [+4.44, +8.07] behind | +7.50 [+5.74, +9.37] behind |
| L | -0.43 [-1.50, +0.67] neither | +5.60 [+4.38, +6.87] behind | +7.37 [+6.21, +8.56] behind |
| S | -0.42 [-1.51, +0.63] neither | +4.38 [+3.44, +5.30] behind | +5.32 [+4.39, +6.23] behind |
| W | -1.99 [-3.18, -0.66] ahead | +0.93 [-0.35, +2.25] neither | +2.15 [+0.91, +3.42] behind |
| LS | -1.49 [-2.55, -0.44] ahead | +4.05 [+3.02, +5.05] behind | +4.91 [+3.92, +5.87] behind |
| LW | -3.32 [-4.29, -2.38] ahead | +0.54 [-0.61, +1.71] neither | +2.34 [+1.21, +3.46] behind |
| SW | -0.75 [-1.66, +0.17] neither | +2.29 [+1.44, +3.15] behind | +3.25 [+2.41, +4.11] behind |
| LSW | -2.47 [-3.51, -1.47] ahead | +1.81 [+0.86, +2.74] behind | +2.79 [+1.88, +3.72] behind |
| Pool | -1.38 [-2.07, -0.74] ahead | +3.22 [+2.51, +3.93] behind | +4.46 [+3.76, +5.14] behind |
| combinations without setups | -1.48 [-2.36, -0.51] ahead | +3.32 [+2.33, +4.34] behind | +4.84 [+3.84, +5.85] behind |
| combinations with setups | -1.28 [-2.05, -0.52] ahead | +3.13 [+2.41, +3.88] behind | +4.07 [+3.35, +4.82] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +1.12 [-0.57, +3.00] neither | +6.85 [+5.09, +8.73] behind | +7.79 [+6.01, +9.67] behind |
| L | -0.23 [-1.33, +0.88] neither | +7.07 [+5.94, +8.24] behind | +8.50 [+7.37, +9.66] behind |
| S | -2.72 [-3.91, -1.59] ahead | -0.27 [-1.34, +0.74] neither | +0.60 [-0.43, +1.56] neither |
| W | +1.55 [+0.45, +2.79] behind | +4.53 [+3.50, +5.67] behind | +5.51 [+4.45, +6.67] behind |
| LS | -3.70 [-4.76, -2.61] ahead | -2.19 [-3.22, -1.15] ahead | -1.14 [-2.15, -0.12] ahead |
| LW | -0.48 [-1.40, +0.42] neither | +4.34 [+3.41, +5.28] behind | +5.73 [+4.81, +6.64] behind |
| SW | -1.94 [-2.97, -0.93] ahead | +0.22 [-0.76, +1.16] neither | +0.80 [-0.17, +1.73] neither |
| LSW | -3.62 [-4.77, -2.53] ahead | -2.17 [-3.22, -1.16] ahead | -1.57 [-2.63, -0.56] ahead |
| Pool | -1.25 [-1.96, -0.59] ahead | +2.30 [+1.60, +2.95] behind | +3.28 [+2.57, +3.93] behind |
| combinations without setups | +0.49 [-0.39, +1.49] neither | +5.70 [+4.79, +6.66] behind | +6.88 [+5.96, +7.85] behind |
| combinations with setups | -3.00 [-3.82, -2.20] ahead | -1.10 [-1.90, -0.33] ahead | -0.33 [-1.13, +0.45] neither |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +10.34 | [+8.63, +12.24] | 1 | 0.0004 | behind |
| L | +8.34 | [+7.46, +9.20] | 1 | 0.0004 | behind |
| S | +4.19 | [+3.41, +5.11] | 1 | 0.0004 | behind |
| W | +3.44 | [+1.92, +5.05] | 1 | 0.0004 | behind |
| LS | +5.25 | [+4.46, +6.11] | 1 | 0.0004 | behind |
| LW | +3.06 | [+2.01, +4.05] | 1 | 0.0004 | behind |
| SW | +2.72 | [+1.81, +3.77] | 1 | 0.0004 | behind |
| LSW | +2.90 | [+2.19, +3.66] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 8, not distinguishable 0

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.50 | [+5.74, +9.37] | 1 / 1 | 0.0025 / 0.0004 | behind |
| L | +7.37 | [+6.21, +8.56] | 1 / 1 | 0.00063 / 0.0004 | behind |
| S | +5.32 | [+4.39, +6.23] | 1 / 1 | 0.00059 / 0.0004 | behind |
| W | +2.15 | [+0.91, +3.42] | 1 / 1 | 0.0066 / 0.0004 | behind |
| LS | +4.91 | [+3.92, +5.87] | 1 / 1 | 0.00063 / 0.0004 | behind |
| LW | +2.34 | [+1.21, +3.46] | 1 / 1 | 0.0053 / 0.0004 | behind |
| SW | +3.25 | [+2.41, +4.11] | 1 / 1 | 0.00084 / 0.0004 | behind |
| LSW | +2.79 | [+1.88, +3.72] | 1 / 1 | 0.0025 / 0.0004 | behind |

Row All, combinations of 8: ahead 0, behind 8, not distinguishable 0

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +10.64 | [+8.86, +12.59] | 1 | 0.0004 | behind |
| L | +9.48 | [+8.76, +10.23] | 1 | 0.0004 | behind |
| S | -0.48 | [-1.41, +0.53] | 1 | 1 | not distinguishable |
| W | +6.86 | [+5.34, +8.56] | 1 | 0.0004 | behind |
| LS | -0.83 | [-1.49, -0.08] | 0.11 | 1 | not distinguishable |
| LW | +6.47 | [+5.81, +7.09] | 1 | 0.0004 | behind |
| SW | +0.28 | [-0.78, +1.41] | 1 | 1 | not distinguishable |
| LSW | -1.47 | [-2.28, -0.63] | 0.0024 | 1 | ahead |

Row Sel, combinations of 8: ahead 1, behind 4, not distinguishable 3

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.79 | [+6.01, +9.67] | 1 / 1 | 0.0037 / 0.0004 | behind |
| L | +8.50 | [+7.37, +9.66] | 1 / 1 | 0.00042 / 0.0004 | behind |
| S | +0.60 | [-0.43, +1.56] | 1 / 1 | 0.25 / 0.37 | not distinguishable |
| W | +5.51 | [+4.45, +6.67] | 1 / 1 | 0.0014 / 0.0004 | behind |
| LS | -1.14 | [-2.15, -0.12] | 0.11 / 0.11 | 1 / 1 | not distinguishable |
| LW | +5.73 | [+4.81, +6.64] | 1 / 1 | 0.00069 / 0.0004 | behind |
| SW | +0.80 | [-0.17, +1.73] | 1 / 1 | 0.091 / 0.21 | not distinguishable |
| LSW | -1.57 | [-2.63, -0.56] | 0.036 / 0.0076 | 1 / 1 | ahead |

Row All, combinations of 8: ahead 1, behind 4, not distinguishable 3

### (iv) best rule minus policy (selected)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +0.88 [-1.13, +2.75] neither | 33.3 | 0.0 | 66.7 | 27 |
| L | +4.20 [+3.47, +4.89] behind | 0.0 | 0.0 | 100.0 | 27 |
| S | +4.64 [+3.53, +5.87] behind | 0.0 | 0.0 | 100.0 | 27 |
| W | +2.27 [+0.35, +4.02] behind | 25.9 | 0.0 | 74.1 | 27 |
| LS | +4.08 [+3.19, +4.93] behind | 3.7 | 0.0 | 96.3 | 27 |
| LW | +3.99 [+3.19, +4.84] behind | 0.0 | 0.0 | 100.0 | 27 |
| SW | +3.23 [+2.00, +4.50] behind | 7.4 | 0.0 | 92.6 | 27 |
| LSW | +4.21 [+3.18, +5.24] behind | 3.7 | 3.7 | 92.6 | 27 |
| Pool | +3.44 [+2.80, +4.05] behind | 9.3 | 0.5 | 90.3 | 27 |
| combinations without setups | +2.83 [+1.84, +3.76] behind | 14.8 | 0.0 | 85.2 | 27 |
| combinations with setups | +4.04 [+3.36, +4.77] behind | 3.7 | 0.9 | 95.4 | 27 |

### (iv) best rule minus policy (all_seeds)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +3.42 [+1.58, +5.14] behind | 10.4 | 0.0 | 89.6 | 27 |
| L | +5.15 [+4.08, +6.23] behind | 0.0 | 0.0 | 100.0 | 27 |
| S | +3.54 [+2.23, +4.91] behind | 8.9 | 0.0 | 91.1 | 27 |
| W | +3.45 [+2.15, +4.65] behind | 8.1 | 0.7 | 91.1 | 27 |
| LS | +4.44 [+3.25, +5.60] behind | 5.9 | 1.5 | 92.6 | 27 |
| LW | +4.72 [+3.75, +5.71] behind | 0.0 | 0.0 | 100.0 | 27 |
| SW | +2.68 [+1.58, +3.87] behind | 15.6 | 0.0 | 84.4 | 27 |
| LSW | +4.33 [+3.17, +5.57] behind | 5.2 | 0.7 | 94.1 | 27 |
| Pool | +3.97 [+3.25, +4.74] behind | 6.8 | 0.4 | 92.9 | 27 |
| combinations without setups | +4.18 [+3.18, +5.11] behind | 4.6 | 0.2 | 95.2 | 27 |
| combinations with setups | +3.75 [+2.86, +4.67] behind | 8.9 | 0.6 | 90.6 | 27 |

### Size of the solver's model: 216 valid records of CP-SAT started from the rule schedule; median [minimum, maximum] over the cells

| entry | cells | variables of the initial model | seconds to build the model | peak memory of the solve in GB | seconds until the first incumbent, on the record's clock (for a model configuration from the solver call, after the model build) |
|---|---|---|---|---|---|
| N | 27 | - | 15.40 [9.94, 18.60] | 1.0 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| L | 27 | - | 19.30 [15.60, 21.56] | 1.0 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| S | 27 | - | 89.55 [59.73, 123.09] | 1.8 [1.5, 2.9] | 0.0 [0.0, 0.0] |
| W | 27 | - | 15.44 [9.43, 18.88] | 1.0 [0.9, 1.2] | 0.0 [0.0, 0.0] |
| LS | 27 | - | 95.91 [53.33, 126.66] | 1.9 [1.4, 2.5] | 0.0 [0.0, 0.0] |
| LW | 27 | - | 18.73 [13.33, 22.00] | 1.0 [0.9, 1.1] | 0.0 [0.0, 0.0] |
| SW | 27 | - | 94.79 [68.77, 129.10] | 2.1 [1.6, 3.9] | 0.0 [0.0, 0.0] |
| LSW | 27 | - | 100.26 [65.39, 131.90] | 2.1 [1.7, 2.6] | 0.0 [0.0, 0.0] |
| Pool | 216 | - | 37.67 [9.43, 131.90] | 1.3 [0.9, 3.9] | 0.0 [0.0, 0.0] |
| combinations without setups | 108 | - | 17.22 [9.43, 22.00] | 1.0 [0.9, 1.2] | 0.0 [0.0, 0.0] |
| combinations with setups | 108 | - | 94.95 [53.33, 131.90] | 2.0 [1.4, 3.9] | 0.0 [0.0, 0.0] |

- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/N/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/L/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/S/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/W/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LS/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LW/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/SW/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/50x25n30_cparms/evidence_rule600s/lns/LSW/instance_029.json (no solver_log_head)

### (v) rule-started memetic algorithm reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 2 | 60 | 0.0 | 360.48 | 324.56 | 358.19 |
| L | 57 | 58 | 0.0 | 447.85 | 396.85 | 429.89 |
| S | 79 | 78 | 0.0 | 492.89 | 452.30 | 471.19 |
| W | 94 | 179 | 22.2 | 388.52 | 368.04 | 380.63 |
| LS | 74 | 75 | 0.0 | 581.19 | 530.63 | 558.48 |
| LW | 195 | 163 | 11.1 | 469.44 | 438.33 | 451.52 |
| SW | 92 | 114 | 3.7 | 518.56 | 489.26 | 502.44 |
| LSW | 111 | 101 | 0.0 | 603.85 | 563.44 | 579.67 |
| Pool | 76 | 76 | 4.6 | 482.85 | 445.43 | 466.50 |
| combinations without setups | 64 | 68.5 | 8.3 | 416.57 | 381.94 | 405.06 |
| combinations with setups | 82 | 79.5 | 0.9 | 549.12 | 508.91 | 527.94 |

### (v) rule-started CP-SAT reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 11 | 37 | 0.0 | 360.48 | 323.70 | 358.19 |
| L | 52 | 55 | 0.0 | 447.85 | 392.70 | 429.89 |
| S | not within the budget | not within the budget | 51.9 | 492.89 | 473.59 | 471.19 |
| W | 25 | 37 | 0.0 | 388.52 | 356.15 | 380.63 |
| LS | not within the budget | not within the budget | 70.4 | 581.19 | 563.33 | 558.48 |
| LW | 56 | 53 | 0.0 | 469.44 | 424.11 | 451.52 |
| SW | 392 | 381 | 48.1 | 518.56 | 501.30 | 502.44 |
| LSW | not within the budget | not within the budget | 74.1 | 603.85 | 588.48 | 579.67 |
| Pool | 107 | 84 | 30.6 | 482.85 | 452.92 | 466.50 |
| combinations without setups | 37 | 49.5 | 0.0 | 416.57 | 374.17 | 405.06 |
| combinations with setups | not within the budget | not within the budget | 61.1 | 549.12 | 531.68 | 527.94 |

## 80x25n30 (instances 3-29; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -0.68 [-2.31, +0.99] neither | +6.34 [+4.85, +7.90] behind | +7.89 [+6.38, +9.44] behind |
| L | -0.99 [-2.68, +1.05] neither | +5.03 [+3.30, +6.97] behind | +7.10 [+5.34, +9.05] behind |
| S | -1.97 [-3.05, -0.73] ahead | +2.30 [+1.26, +3.49] behind | +2.97 [+1.90, +4.19] behind |
| W | -1.76 [-3.12, -0.33] ahead | -0.96 [-2.36, +0.47] neither | +0.16 [-1.16, +1.53] neither |
| LS | -0.64 [-1.65, +0.60] neither | +2.90 [+1.93, +4.07] behind | +3.80 [+2.84, +4.94] behind |
| LW | -1.26 [-3.01, +0.80] neither | -0.06 [-1.88, +2.02] neither | +1.39 [-0.42, +3.43] neither |
| SW | -4.32 [-5.52, -3.17] ahead | -1.72 [-2.88, -0.58] ahead | -0.90 [-2.00, +0.20] neither |
| LSW | -3.75 [-4.80, -2.63] ahead | -0.85 [-2.02, +0.38] neither | +0.20 [-0.80, +1.28] neither |
| Pool | -1.92 [-2.61, -1.20] ahead | +1.62 [+1.03, +2.25] behind | +2.83 [+2.27, +3.43] behind |
| combinations without setups | -1.17 [-2.22, -0.06] ahead | +2.59 [+1.76, +3.51] behind | +4.13 [+3.35, +4.99] behind |
| combinations with setups | -2.67 [-3.36, -1.96] ahead | +0.66 [+0.03, +1.30] behind | +1.52 [+0.91, +2.14] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.71 [-0.85, +2.33] neither | +6.66 [+5.27, +8.11] behind | +8.55 [+7.12, +10.03] behind |
| L | -0.09 [-1.77, +1.93] neither | +5.73 [+3.91, +7.82] behind | +8.60 [+6.78, +10.64] behind |
| S | -4.44 [-5.65, -3.15] ahead | -3.71 [-4.87, -2.48] ahead | -2.92 [-4.07, -1.71] ahead |
| W | +0.39 [-0.95, +1.80] neither | +4.68 [+3.44, +5.99] behind | +6.12 [+4.74, +7.57] behind |
| LS | -3.16 [-4.35, -1.81] ahead | -2.48 [-3.66, -1.13] ahead | -1.74 [-2.84, -0.45] ahead |
| LW | +0.31 [-1.47, +2.40] neither | +4.85 [+3.09, +6.86] behind | +6.91 [+5.15, +8.94] behind |
| SW | -4.50 [-5.72, -3.34] ahead | -3.86 [-5.02, -2.73] ahead | -3.10 [-4.20, -2.04] ahead |
| LSW | -4.07 [-5.07, -2.97] ahead | -3.60 [-4.64, -2.47] ahead | -3.11 [-4.15, -1.99] ahead |
| Pool | -1.85 [-2.58, -1.12] ahead | +1.03 [+0.40, +1.68] behind | +2.42 [+1.83, +3.02] behind |
| combinations without setups | +0.33 [-0.65, +1.36] neither | +5.48 [+4.70, +6.33] behind | +7.55 [+6.83, +8.35] behind |
| combinations with setups | -4.04 [-4.82, -3.29] ahead | -3.41 [-4.17, -2.68] ahead | -2.72 [-3.46, -2.01] ahead |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.82 [-4.20, -1.23] ahead | +4.07 [+2.60, +5.66] behind | +5.59 [+4.06, +7.18] behind |
| L | -3.73 [-5.18, -1.98] ahead | +2.14 [+0.46, +4.00] behind | +4.16 [+2.43, +6.04] behind |
| S | -1.33 [-2.44, -0.19] ahead | +2.96 [+1.96, +4.00] behind | +3.63 [+2.63, +4.68] behind |
| W | -3.79 [-5.02, -2.36] ahead | -3.01 [-4.31, -1.55] ahead | -1.91 [-3.15, -0.52] ahead |
| LS | -0.79 [-1.88, +0.28] neither | +2.75 [+1.67, +3.82] behind | +3.64 [+2.57, +4.70] behind |
| LW | -3.92 [-5.26, -2.22] ahead | -2.75 [-4.19, -0.99] ahead | -1.33 [-2.85, +0.41] neither |
| SW | -3.85 [-5.00, -2.77] ahead | -1.24 [-2.38, -0.20] ahead | -0.42 [-1.49, +0.57] neither |
| LSW | -3.64 [-4.59, -2.70] ahead | -0.74 [-1.82, +0.29] neither | +0.32 [-0.63, +1.23] neither |
| Pool | -2.98 [-3.77, -2.15] ahead | +0.52 [-0.30, +1.35] neither | +1.71 [+0.87, +2.55] behind |
| combinations without setups | -3.56 [-4.68, -2.14] ahead | +0.11 [-1.06, +1.51] neither | +1.63 [+0.42, +3.02] behind |
| combinations with setups | -2.40 [-3.22, -1.57] ahead | +0.93 [+0.15, +1.75] behind | +1.79 [+1.02, +2.60] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.45 [-2.78, +0.09] neither | +4.39 [+3.00, +5.90] behind | +6.24 [+4.81, +7.79] behind |
| L | -2.85 [-4.31, -1.10] ahead | +2.81 [+1.17, +4.69] behind | +5.62 [+3.89, +7.53] behind |
| S | -3.82 [-5.01, -2.63] ahead | -3.08 [-4.23, -1.92] ahead | -2.28 [-3.44, -1.11] ahead |
| W | -1.69 [-2.90, -0.28] ahead | +2.52 [+1.36, +3.87] behind | +3.92 [+2.72, +5.35] behind |
| LS | -3.31 [-4.49, -2.17] ahead | -2.63 [-3.81, -1.48] ahead | -1.90 [-3.03, -0.80] ahead |
| LW | -2.39 [-3.75, -0.67] ahead | +2.04 [+0.58, +3.78] behind | +4.05 [+2.56, +5.81] behind |
| SW | -4.04 [-5.19, -2.95] ahead | -3.39 [-4.51, -2.32] ahead | -2.62 [-3.71, -1.58] ahead |
| LSW | -3.95 [-4.89, -3.03] ahead | -3.49 [-4.42, -2.58] ahead | -3.00 [-3.93, -2.09] ahead |
| Pool | -2.94 [-3.73, -2.09] ahead | -0.10 [-0.90, +0.73] neither | +1.25 [+0.46, +2.09] behind |
| combinations without setups | -2.10 [-3.19, -0.69] ahead | +2.94 [+1.76, +4.35] behind | +4.96 [+3.73, +6.38] behind |
| combinations with setups | -3.78 [-4.63, -2.95] ahead | -3.15 [-3.97, -2.32] ahead | -2.45 [-3.28, -1.62] ahead |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.89 | [+6.38, +9.44] | 1 | 0.0004 | behind |
| L | +7.10 | [+5.34, +9.05] | 1 | 0.0004 | behind |
| S | +2.97 | [+1.90, +4.19] | 1 | 0.0004 | behind |
| W | +0.16 | [-1.16, +1.53] | 1 | 1 | not distinguishable |
| LS | +3.80 | [+2.84, +4.94] | 1 | 0.0004 | behind |
| LW | +1.39 | [-0.42, +3.43] | 1 | 0.29 | not distinguishable |
| SW | -0.90 | [-2.00, +0.20] | 0.43 | 1 | not distinguishable |
| LSW | +0.20 | [-0.80, +1.28] | 1 | 1 | not distinguishable |

Row Sel, combinations of 8: ahead 0, behind 4, not distinguishable 4

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +5.59 | [+4.06, +7.18] | 1 / 1 | 0.0057 / 0.0004 | behind |
| L | +4.16 | [+2.43, +6.04] | 1 / 1 | 0.025 / 0.0004 | behind |
| S | +3.63 | [+2.63, +4.68] | 1 / 1 | 0.0034 / 0.0004 | behind |
| W | -1.91 | [-3.15, -0.52] | 0.16 / 0.038 | 1 / 1 | not distinguishable |
| LS | +3.64 | [+2.57, +4.70] | 1 / 1 | 0.0034 / 0.0004 | behind |
| LW | -1.33 | [-2.85, +0.41] | 0.59 / 0.42 | 1 / 1 | not distinguishable |
| SW | -0.42 | [-1.49, +0.57] | 0.81 / 1 | 1 / 1 | not distinguishable |
| LSW | +0.32 | [-0.63, +1.23] | 1 / 1 | 0.87 / 1 | not distinguishable |

Row All, combinations of 8: ahead 0, behind 4, not distinguishable 4

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +8.55 | [+7.12, +10.03] | 1 | 0.0004 | behind |
| L | +8.60 | [+6.78, +10.64] | 1 | 0.0004 | behind |
| S | -2.92 | [-4.07, -1.71] | 0.0004 | 1 | ahead |
| W | +6.12 | [+4.74, +7.57] | 1 | 0.0004 | behind |
| LS | -1.74 | [-2.84, -0.45] | 0.028 | 1 | ahead |
| LW | +6.91 | [+5.15, +8.94] | 1 | 0.0004 | behind |
| SW | -3.10 | [-4.20, -2.04] | 0.0004 | 1 | ahead |
| LSW | -3.11 | [-4.15, -1.99] | 0.0004 | 1 | ahead |

Row Sel, combinations of 8: ahead 4, behind 4, not distinguishable 0

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.24 | [+4.81, +7.79] | 1 / 1 | 0.0052 / 0.0004 | behind |
| L | +5.62 | [+3.89, +7.53] | 1 / 1 | 0.013 / 0.0004 | behind |
| S | -2.28 | [-3.44, -1.11] | 0.013 / 0.0018 | 1 / 1 | ahead |
| W | +3.92 | [+2.72, +5.35] | 1 / 1 | 0.013 / 0.0004 | behind |
| LS | -1.90 | [-3.03, -0.80] | 0.019 / 0.0022 | 1 / 1 | ahead |
| LW | +4.05 | [+2.56, +5.81] | 1 / 1 | 0.02 / 0.0004 | behind |
| SW | -2.62 | [-3.71, -1.58] | 0.0044 / 0.0004 | 1 / 1 | ahead |
| LSW | -3.00 | [-3.93, -2.09] | 0.0044 / 0.0004 | 1 / 1 | ahead |

Row All, combinations of 8: ahead 4, behind 4, not distinguishable 0

### (iv) best rule minus policy (selected)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +1.44 [-0.29, +3.15] neither | 44.4 | 0.0 | 55.6 | 27 |
| L | +1.84 [-0.09, +3.47] neither | 18.5 | 0.0 | 81.5 | 27 |
| S | +4.88 [+3.49, +6.23] behind | 7.4 | 0.0 | 92.6 | 27 |
| W | +2.00 [+0.52, +3.41] behind | 29.6 | 3.7 | 66.7 | 27 |
| LS | +3.51 [+2.14, +4.77] behind | 11.1 | 0.0 | 88.9 | 27 |
| LW | +1.57 [-0.37, +3.24] neither | 22.2 | 0.0 | 77.8 | 27 |
| SW | +4.91 [+3.62, +6.28] behind | 3.7 | 0.0 | 96.3 | 27 |
| LSW | +4.49 [+3.31, +5.59] behind | 7.4 | 0.0 | 92.6 | 27 |
| Pool | +3.08 [+2.28, +3.87] behind | 18.1 | 0.5 | 81.5 | 27 |
| combinations without setups | +1.71 [+0.58, +2.78] behind | 28.7 | 0.9 | 70.4 | 27 |
| combinations with setups | +4.45 [+3.63, +5.31] behind | 7.4 | 0.0 | 92.6 | 27 |

### (iv) best rule minus policy (all_seeds)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +3.56 [+1.93, +4.98] behind | 11.1 | 0.7 | 88.1 | 27 |
| L | +4.59 [+2.83, +6.12] behind | 4.4 | 0.0 | 95.6 | 27 |
| S | +4.19 [+2.90, +5.50] behind | 9.6 | 0.0 | 90.4 | 27 |
| W | +4.09 [+2.59, +5.42] behind | 5.9 | 1.5 | 92.6 | 27 |
| LS | +3.65 [+2.43, +4.92] behind | 13.3 | 0.7 | 85.9 | 27 |
| LW | +4.23 [+2.52, +5.65] behind | 4.4 | 0.0 | 95.6 | 27 |
| SW | +4.40 [+3.19, +5.68] behind | 8.9 | 0.0 | 91.1 | 27 |
| LSW | +4.35 [+3.34, +5.38] behind | 5.2 | 0.0 | 94.8 | 27 |
| Pool | +4.13 [+3.24, +4.98] behind | 7.9 | 0.4 | 91.8 | 27 |
| combinations without setups | +4.12 [+2.66, +5.28] behind | 6.5 | 0.6 | 93.0 | 27 |
| combinations with setups | +4.14 [+3.24, +5.07] behind | 9.3 | 0.2 | 90.6 | 27 |

### Size of the solver's model: 216 valid records of CP-SAT started from the rule schedule; median [minimum, maximum] over the cells

| entry | cells | variables of the initial model | seconds to build the model | peak memory of the solve in GB | seconds until the first incumbent, on the record's clock (for a model configuration from the solver call, after the model build) |
|---|---|---|---|---|---|
| N | 27 | - | 25.42 [21.66, 28.57] | 1.1 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| L | 27 | - | 27.96 [24.11, 31.60] | 1.1 [1.0, 1.2] | 0.0 [0.0, 0.0] |
| S | 27 | - | 108.91 [64.41, 145.40] | 2.0 [1.7, 3.5] | 0.0 [0.0, 0.0] |
| W | 27 | - | 24.63 [19.27, 29.29] | 1.1 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| LS | 27 | - | 121.65 [78.29, 150.55] | 2.2 [1.4, 3.0] | 0.0 [0.0, 0.0] |
| LW | 27 | - | 28.48 [23.96, 32.62] | 1.1 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| SW | 27 | - | 113.92 [64.79, 145.20] | 2.0 [1.7, 3.0] | 0.0 [0.0, 0.0] |
| LSW | 27 | - | 122.89 [82.34, 146.26] | 2.1 [1.7, 3.6] | 0.0 [0.0, 0.0] |
| Pool | 216 | - | 48.52 [19.27, 150.55] | 1.3 [1.0, 3.6] | 0.0 [0.0, 0.0] |
| combinations without setups | 108 | - | 26.89 [19.27, 32.62] | 1.1 [1.0, 1.2] | 0.0 [0.0, 0.0] |
| combinations with setups | 108 | - | 117.64 [64.41, 150.55] | 2.1 [1.4, 3.6] | 0.0 [0.0, 0.0] |

- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_010.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_011.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_012.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_013.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_014.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_015.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_016.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_017.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_018.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_019.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_020.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_021.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_022.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_023.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_024.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_025.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_026.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_027.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_028.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/N/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/L/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/S/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/W/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LS/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LW/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/SW/instance_029.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/80x25n30_cparms/evidence_rule600s/lns/LSW/instance_029.json (no solver_log_head)

### (v) rule-started memetic algorithm reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 90 | 86 | 0.0 | 526.74 | 482.04 | 520.00 |
| L | 98 | 116 | 0.0 | 615.85 | 565.89 | 605.89 |
| S | 123 | 132 | 7.4 | 717.07 | 664.44 | 683.89 |
| W | 575 | not within the budget | 51.9 | 558.26 | 547.11 | 547.85 |
| LS | 102 | 117 | 3.7 | 798.00 | 743.15 | 771.37 |
| LW | 316 | 458 | 44.4 | 638.22 | 621.26 | 629.63 |
| SW | not within the budget | not within the budget | 51.9 | 750.37 | 722.19 | 715.59 |
| LSW | 527 | 355 | 44.4 | 829.15 | 792.44 | 793.89 |
| Pool | 150 | 171 | 25.5 | 679.21 | 642.31 | 658.51 |
| combinations without setups | 110 | 152 | 24.1 | 584.77 | 554.07 | 575.84 |
| combinations with setups | 210 | 193 | 26.9 | 773.65 | 730.56 | 741.19 |

### (v) rule-started CP-SAT reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 44 | 62 | 0.0 | 526.74 | 479.04 | 520.00 |
| L | 64 | 102 | 0.0 | 615.85 | 558.04 | 605.89 |
| S | not within the budget | not within the budget | 88.9 | 717.07 | 705.00 | 683.89 |
| W | 53 | 80 | 0.0 | 558.26 | 516.30 | 547.85 |
| LS | not within the budget | not within the budget | 81.5 | 798.00 | 785.44 | 771.37 |
| LW | 54 | 106 | 0.0 | 638.22 | 589.00 | 629.63 |
| SW | not within the budget | not within the budget | 85.2 | 750.37 | 738.81 | 715.59 |
| LSW | not within the budget | not within the budget | 92.6 | 829.15 | 819.56 | 793.89 |
| Pool | 266 | 226.5 | 43.5 | 679.21 | 648.90 | 658.51 |
| combinations without setups | 53 | 95.5 | 0.0 | 584.77 | 535.59 | 575.84 |
| combinations with setups | not within the budget | not within the budget | 87.0 | 773.65 | 762.20 | 741.19 |

## Record counts per cell

| arm, size, combination | states |
|---|---|
| cpsat_rule|50x25n30|L | valid 27 |
| cpsat_rule|50x25n30|LS | valid 27 |
| cpsat_rule|50x25n30|LSW | valid 27 |
| cpsat_rule|50x25n30|LW | valid 27 |
| cpsat_rule|50x25n30|N | valid 27 |
| cpsat_rule|50x25n30|S | valid 27 |
| cpsat_rule|50x25n30|SW | valid 27 |
| cpsat_rule|50x25n30|W | valid 27 |
| cpsat_rule|80x25n30|L | valid 27 |
| cpsat_rule|80x25n30|LS | valid 27 |
| cpsat_rule|80x25n30|LSW | valid 27 |
| cpsat_rule|80x25n30|LW | valid 27 |
| cpsat_rule|80x25n30|N | valid 27 |
| cpsat_rule|80x25n30|S | valid 27 |
| cpsat_rule|80x25n30|SW | valid 27 |
| cpsat_rule|80x25n30|W | valid 27 |
| memetic_rule|50x25n30|L | valid 27 |
| memetic_rule|50x25n30|LS | valid 27 |
| memetic_rule|50x25n30|LSW | valid 27 |
| memetic_rule|50x25n30|LW | valid 27 |
| memetic_rule|50x25n30|N | valid 27 |
| memetic_rule|50x25n30|S | valid 27 |
| memetic_rule|50x25n30|SW | valid 27 |
| memetic_rule|50x25n30|W | valid 27 |
| memetic_rule|80x25n30|L | valid 27 |
| memetic_rule|80x25n30|LS | valid 27 |
| memetic_rule|80x25n30|LSW | valid 27 |
| memetic_rule|80x25n30|LW | valid 27 |
| memetic_rule|80x25n30|N | valid 27 |
| memetic_rule|80x25n30|S | valid 27 |
| memetic_rule|80x25n30|SW | valid 27 |
| memetic_rule|80x25n30|W | valid 27 |
