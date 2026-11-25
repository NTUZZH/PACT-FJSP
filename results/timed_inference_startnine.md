# Timed campaign against the policy (full, tag startnine)

- setting: environment exact, training seeds 301,302,303,304,305 (5 runs per arm); policy: selected training run s304; CP-SAT configuration: lns
- clock origin: start (wall clock from the start of the run: memetic algorithm from the start of the process that solves the instance (instance loading, start schedules and numba warm-up included); CP-SAT model arms from the start of the arm, model building included (instance load and hint validation before it not timed by the runner); the neighbourhood-search arm from its start)
- budgets 60, 300, 600 s; verdict words at the stated budget of 600 s only; B = 20000; differences are first-named minus second in percent of the second; the interval of a selected run is the instance-level cluster bootstrap, of every run the crossed hierarchical bootstrap
- campaign: 50x25n30 80x25n30 = memetic_rule@600:3-11 cpsat_rule@600:3-11
- instances per size and arm (the instance range of a token is the design of its arm; a token without one plans the size's evidence instances): 50x25n30 memetic_rule 3-11 (9), cpsat_rule 3-11 (9); 80x25n30 memetic_rule 3-11 (9), cpsat_rule 3-11 (9)
- cells 288, valid 288
- pooled rows beside Pool: the combinations without setups (N, L, W, LW); the combinations with setups (S, LS, SW, LSW), each over the instances present in every combination of its group, with the estimator of Pool

## 50x25n30 (instances 3-11; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +2.34 [-1.00, +5.74] neither | +9.48 [+6.86, +11.94] behind | +10.93 [+8.32, +13.53] behind |
| L | -0.35 [-1.84, +1.08] neither | +6.56 [+5.12, +7.86] behind | +8.17 [+6.93, +9.40] behind |
| S | -2.19 [-3.88, -0.58] ahead | +3.10 [+1.86, +4.41] behind | +4.16 [+2.94, +5.40] behind |
| W | -1.19 [-3.68, +1.43] neither | +1.76 [-0.87, +4.29] neither | +2.96 [+0.67, +5.17] behind |
| LS | -0.29 [-1.47, +0.82] neither | +5.15 [+3.62, +6.78] behind | +5.91 [+4.20, +7.70] behind |
| LW | -3.66 [-5.43, -1.99] ahead | +0.48 [-1.74, +2.59] neither | +1.93 [-0.37, +4.20] neither |
| SW | -1.69 [-3.10, -0.21] ahead | +2.07 [+0.54, +3.77] behind | +2.96 [+1.51, +4.60] behind |
| LSW | -3.00 [-4.42, -1.67] ahead | +2.38 [+1.42, +3.39] behind | +3.65 [+2.75, +4.64] behind |
| Pool | -1.26 [-2.16, -0.30] ahead | +3.87 [+2.96, +4.75] behind | +5.08 [+4.23, +5.92] behind |
| combinations without setups | -0.71 [-2.33, +0.88] neither | +4.57 [+2.96, +5.96] behind | +6.00 [+4.49, +7.31] behind |
| combinations with setups | -1.80 [-2.97, -0.64] ahead | +3.18 [+2.17, +4.18] behind | +4.17 [+3.12, +5.23] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +4.27 [+1.37, +7.26] behind | +10.41 [+7.72, +13.10] behind | +11.57 [+9.15, +13.98] behind |
| L | +0.18 [-1.24, +1.70] neither | +7.42 [+6.13, +8.88] behind | +9.00 [+7.78, +10.26] behind |
| S | -5.39 [-7.41, -3.52] ahead | -2.05 [-4.10, -0.12] ahead | -1.10 [-2.98, +0.75] neither |
| W | +2.85 [+0.76, +5.44] behind | +5.59 [+3.70, +8.08] behind | +6.76 [+4.99, +9.18] behind |
| LS | -3.98 [-5.25, -2.72] ahead | -1.87 [-3.37, -0.35] ahead | -0.44 [-1.56, +0.70] neither |
| LW | -0.48 [-2.01, +0.89] neither | +4.31 [+2.53, +5.81] behind | +6.24 [+4.80, +7.62] behind |
| SW | -2.73 [-4.75, -0.70] ahead | -0.58 [-2.14, +1.13] neither | +0.13 [-1.30, +1.65] neither |
| LSW | -4.09 [-5.67, -2.42] ahead | -2.66 [-4.02, -1.28] ahead | -2.10 [-3.48, -0.72] ahead |
| Pool | -1.17 [-2.04, -0.23] ahead | +2.57 [+1.84, +3.35] behind | +3.76 [+3.09, +4.42] behind |
| combinations without setups | +1.71 [+0.33, +3.07] behind | +6.93 [+5.74, +8.12] behind | +8.39 [+7.34, +9.41] behind |
| combinations with setups | -4.05 [-5.41, -2.69] ahead | -1.79 [-3.06, -0.48] ahead | -0.88 [-2.10, +0.34] neither |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.20 [-3.18, +1.41] neither | +5.75 [+3.51, +8.14] behind | +7.15 [+4.94, +9.53] behind |
| L | -1.05 [-2.44, +0.18] neither | +5.82 [+4.19, +7.41] behind | +7.42 [+5.89, +8.96] behind |
| S | -0.68 [-2.59, +1.14] neither | +4.70 [+3.29, +6.13] behind | +5.77 [+4.43, +7.16] behind |
| W | -2.10 [-3.67, -0.43] ahead | +0.84 [-1.20, +2.98] neither | +2.05 [+0.14, +4.10] behind |
| LS | -0.56 [-2.11, +1.09] neither | +4.85 [+3.40, +6.22] behind | +5.60 [+4.05, +7.07] behind |
| LW | -4.18 [-5.66, -2.68] ahead | -0.06 [-2.06, +1.97] neither | +1.38 [-0.73, +3.54] neither |
| SW | -1.36 [-2.66, -0.08] ahead | +2.43 [+0.95, +3.93] behind | +3.32 [+1.92, +4.76] behind |
| LSW | -2.94 [-4.51, -1.45] ahead | +2.45 [+1.24, +3.73] behind | +3.72 [+2.59, +5.00] behind |
| Pool | -1.76 [-2.66, -0.85] ahead | +3.35 [+2.45, +4.38] behind | +4.55 [+3.64, +5.59] behind |
| combinations without setups | -2.13 [-3.29, -0.88] ahead | +3.09 [+1.59, +4.65] behind | +4.50 [+2.95, +6.07] behind |
| combinations with setups | -1.38 [-2.68, -0.13] ahead | +3.61 [+2.63, +4.62] behind | +4.60 [+3.60, +5.64] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.69 [-1.35, +3.13] neither | +6.64 [+4.48, +9.00] behind | +7.78 [+5.51, +10.11] behind |
| L | -0.53 [-2.03, +1.02] neither | +6.67 [+5.14, +8.22] behind | +8.24 [+6.70, +9.75] behind |
| S | -3.91 [-6.20, -1.70] ahead | -0.53 [-2.70, +1.41] neither | +0.42 [-1.52, +2.19] neither |
| W | +1.90 [+0.76, +3.23] behind | +4.62 [+3.53, +5.90] behind | +5.79 [+4.75, +7.03] behind |
| LS | -4.24 [-5.85, -2.39] ahead | -2.14 [-3.81, -0.30] ahead | -0.71 [-2.17, +0.85] neither |
| LW | -1.01 [-2.27, +0.15] neither | +3.74 [+2.34, +5.01] behind | +5.67 [+4.42, +6.89] behind |
| SW | -2.39 [-4.24, -0.59] ahead | -0.23 [-1.76, +1.28] neither | +0.47 [-0.93, +1.84] neither |
| LSW | -4.01 [-6.07, -2.13] ahead | -2.59 [-4.35, -0.95] ahead | -2.03 [-3.75, -0.42] ahead |
| Pool | -1.69 [-2.67, -0.72] ahead | +2.02 [+1.03, +2.94] behind | +3.21 [+2.22, +4.08] behind |
| combinations without setups | +0.27 [-0.82, +1.46] neither | +5.42 [+4.29, +6.59] behind | +6.87 [+5.67, +8.03] behind |
| combinations with setups | -3.64 [-5.17, -2.18] ahead | -1.37 [-2.80, +0.01] neither | -0.46 [-1.84, +0.86] neither |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +10.93 | [+8.32, +13.53] | 1 | 0.0004 | behind |
| L | +8.17 | [+6.93, +9.40] | 1 | 0.0004 | behind |
| S | +4.16 | [+2.94, +5.40] | 1 | 0.0004 | behind |
| W | +2.96 | [+0.67, +5.17] | 1 | 0.01 | behind |
| LS | +5.91 | [+4.20, +7.70] | 1 | 0.0004 | behind |
| LW | +1.93 | [-0.37, +4.20] | 1 | 0.05 | not distinguishable |
| SW | +2.96 | [+1.51, +4.60] | 1 | 0.0004 | behind |
| LSW | +3.65 | [+2.75, +4.64] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 7, not distinguishable 1

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.15 | [+4.94, +9.53] | 1 / 1 | 0.0059 / 0.0004 | behind |
| L | +7.42 | [+5.89, +8.96] | 1 / 1 | 0.00046 / 0.0004 | behind |
| S | +5.77 | [+4.43, +7.16] | 1 / 1 | 0.0005 / 0.0004 | behind |
| W | +2.05 | [+0.14, +4.10] | 1 / 1 | 0.011 / 0.036 | behind |
| LS | +5.60 | [+4.05, +7.07] | 1 / 1 | 0.00021 / 0.0004 | behind |
| LW | +1.38 | [-0.73, +3.54] | 1 / 1 | 0.011 / 0.1 | not distinguishable |
| SW | +3.32 | [+1.92, +4.76] | 1 / 1 | 0.0013 / 0.0004 | behind |
| LSW | +3.72 | [+2.59, +5.00] | 1 / 1 | 0.00059 / 0.0004 | behind |

Row All, combinations of 8: ahead 0, behind 7, not distinguishable 1

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +11.57 | [+9.15, +13.98] | 1 | 0.0004 | behind |
| L | +9.00 | [+7.78, +10.26] | 1 | 0.0004 | behind |
| S | -1.10 | [-2.98, +0.75] | 0.9 | 1 | not distinguishable |
| W | +6.76 | [+4.99, +9.18] | 1 | 0.0004 | behind |
| LS | -0.44 | [-1.56, +0.70] | 1 | 1 | not distinguishable |
| LW | +6.24 | [+4.80, +7.62] | 1 | 0.0004 | behind |
| SW | +0.13 | [-1.30, +1.65] | 1 | 1 | not distinguishable |
| LSW | -2.10 | [-3.48, -0.72] | 0.0088 | 1 | ahead |

Row Sel, combinations of 8: ahead 1, behind 4, not distinguishable 3

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.78 | [+5.51, +10.11] | 1 / 1 | 0.0073 / 0.0004 | behind |
| L | +8.24 | [+6.70, +9.75] | 1 / 1 | 0.00036 / 0.0004 | behind |
| S | +0.42 | [-1.52, +2.19] | 1 / 1 | 0.52 / 1 | not distinguishable |
| W | +5.79 | [+4.75, +7.03] | 1 / 1 | 0.00074 / 0.0004 | behind |
| LS | -0.71 | [-2.17, +0.85] | 0.25 / 1 | 1 / 1 | not distinguishable |
| LW | +5.67 | [+4.42, +6.89] | 1 / 1 | 0.00042 / 0.0004 | behind |
| SW | +0.47 | [-0.93, +1.84] | 1 / 1 | 0.47 / 1 | not distinguishable |
| LSW | -2.03 | [-3.75, -0.42] | 0.0079 / 0.049 | 1 / 1 | ahead |

Row All, combinations of 8: ahead 1, behind 4, not distinguishable 3

### (iv) best rule minus policy (selected)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +0.58 [-2.63, +3.77] neither | 44.4 | 0.0 | 55.6 | 9 |
| L | +4.63 [+3.27, +5.91] behind | 0.0 | 0.0 | 100.0 | 9 |
| S | +6.29 [+4.00, +8.76] behind | 0.0 | 0.0 | 100.0 | 9 |
| W | +2.71 [-0.10, +5.21] neither | 33.3 | 0.0 | 66.7 | 9 |
| LS | +4.77 [+3.43, +6.09] behind | 0.0 | 0.0 | 100.0 | 9 |
| LW | +4.98 [+3.45, +6.63] behind | 0.0 | 0.0 | 100.0 | 9 |
| SW | +3.59 [+1.23, +6.13] behind | 11.1 | 0.0 | 88.9 | 9 |
| LSW | +4.68 [+2.89, +6.40] behind | 0.0 | 11.1 | 88.9 | 9 |
| Pool | +4.03 [+2.88, +5.07] behind | 11.1 | 1.4 | 87.5 | 9 |
| combinations without setups | +3.23 [+1.58, +4.81] behind | 19.4 | 0.0 | 80.6 | 9 |
| combinations with setups | +4.83 [+3.26, +6.41] behind | 2.8 | 2.8 | 94.4 | 9 |

### (iv) best rule minus policy (all_seeds)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +4.04 [+1.48, +6.10] behind | 8.9 | 0.0 | 91.1 | 9 |
| L | +5.37 [+4.03, +6.75] behind | 0.0 | 0.0 | 100.0 | 9 |
| S | +4.71 [+2.12, +7.38] behind | 8.9 | 0.0 | 91.1 | 9 |
| W | +3.58 [+2.15, +4.86] behind | 6.7 | 0.0 | 93.3 | 9 |
| LS | +5.09 [+3.10, +6.86] behind | 6.7 | 0.0 | 93.3 | 9 |
| LW | +5.53 [+4.21, +6.87] behind | 0.0 | 0.0 | 100.0 | 9 |
| SW | +3.22 [+1.11, +5.51] behind | 17.8 | 0.0 | 82.2 | 9 |
| LSW | +4.64 [+2.57, +6.92] behind | 6.7 | 2.2 | 91.1 | 9 |
| Pool | +4.52 [+3.39, +5.66] behind | 6.9 | 0.3 | 92.8 | 9 |
| combinations without setups | +4.63 [+3.37, +5.75] behind | 3.9 | 0.0 | 96.1 | 9 |
| combinations with setups | +4.41 [+2.73, +6.19] behind | 10.0 | 0.6 | 89.4 | 9 |

### Size of the solver's model: 72 valid records of CP-SAT started from the rule schedule; median [minimum, maximum] over the cells

| entry | cells | variables of the initial model | seconds to build the model | peak memory of the solve in GB | seconds until the first incumbent, on the record's clock (for a model configuration from the solver call, after the model build) |
|---|---|---|---|---|---|
| N | 9 | - | 15.94 [9.94, 18.60] | 1.0 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| L | 9 | - | 19.61 [16.09, 21.56] | 1.0 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| S | 9 | - | 92.99 [68.91, 118.86] | 1.9 [1.6, 2.5] | 0.0 [0.0, 0.0] |
| W | 9 | - | 15.95 [9.43, 18.88] | 1.0 [0.9, 1.0] | 0.0 [0.0, 0.0] |
| LS | 9 | - | 100.36 [78.64, 115.54] | 1.8 [1.5, 2.5] | 0.0 [0.0, 0.0] |
| LW | 9 | - | 19.71 [18.71, 20.93] | 1.0 [1.0, 1.1] | 0.0 [0.0, 0.0] |
| SW | 9 | - | 95.11 [80.56, 124.46] | 2.3 [1.8, 3.1] | 0.0 [0.0, 0.0] |
| LSW | 9 | - | 106.58 [82.41, 126.81] | 2.1 [1.7, 2.6] | 0.0 [0.0, 0.0] |
| Pool | 72 | - | 45.23 [9.43, 126.81] | 1.3 [0.9, 3.1] | 0.0 [0.0, 0.0] |
| combinations without setups | 36 | - | 17.65 [9.43, 21.56] | 1.0 [0.9, 1.1] | 0.0 [0.0, 0.0] |
| combinations with setups | 36 | - | 95.51 [68.91, 126.81] | 2.1 [1.5, 3.1] | 0.0 [0.0, 0.0] |

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

### (v) rule-started memetic algorithm reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 2 | 60 | 0.0 | 362.56 | 325.56 | 361.22 |
| L | 63 | 59 | 0.0 | 447.67 | 395.67 | 427.89 |
| S | 85 | 86 | 0.0 | 486.11 | 439.11 | 457.11 |
| W | 108 | 64 | 33.3 | 387.56 | 367.11 | 377.78 |
| LS | 72 | 74 | 0.0 | 572.00 | 515.33 | 545.89 |
| LW | 254 | 210 | 33.3 | 471.11 | 440.67 | 448.78 |
| SW | 88 | 73 | 0.0 | 509.56 | 478.00 | 491.78 |
| LSW | 109 | 94 | 0.0 | 592.11 | 545.56 | 565.33 |
| Pool | 81 | 73.5 | 8.3 | 478.58 | 438.38 | 459.47 |
| combinations without setups | 72 | 63 | 16.7 | 417.22 | 382.25 | 403.92 |
| combinations with setups | 83 | 76.5 | 0.0 | 539.94 | 494.50 | 515.03 |

### (v) rule-started CP-SAT reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 5 | 34 | 0.0 | 362.56 | 323.67 | 361.22 |
| L | 59 | 65 | 0.0 | 447.67 | 392.67 | 427.89 |
| S | not within the budget | not within the budget | 55.6 | 486.11 | 462.78 | 457.11 |
| W | 26 | 32 | 0.0 | 387.56 | 353.78 | 377.78 |
| LS | not within the budget | not within the budget | 55.6 | 572.00 | 548.44 | 545.89 |
| LW | 69 | 58 | 0.0 | 471.11 | 422.56 | 448.78 |
| SW | 575 | 358 | 44.4 | 509.56 | 491.56 | 491.78 |
| LSW | not within the budget | not within the budget | 77.8 | 592.11 | 578.00 | 565.33 |
| Pool | 136 | 107.5 | 29.2 | 478.58 | 446.68 | 459.47 |
| combinations without setups | 42 | 50 | 0.0 | 417.22 | 373.17 | 403.92 |
| combinations with setups | not within the budget | not within the budget | 58.3 | 539.94 | 520.19 | 515.03 |

## 80x25n30 (instances 3-11; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.49 [-4.67, -0.06] ahead | +4.78 [+2.82, +7.11] behind | +6.26 [+4.21, +8.71] behind |
| L | +0.67 [-3.03, +5.40] neither | +7.30 [+3.78, +11.37] behind | +9.44 [+5.87, +13.50] behind |
| S | -2.34 [-4.08, -0.68] ahead | +1.61 [+0.15, +2.98] behind | +2.27 [+0.81, +3.63] behind |
| W | -4.52 [-5.81, -3.23] ahead | -3.90 [-5.37, -2.45] ahead | -2.59 [-3.69, -1.41] ahead |
| LS | +0.46 [-1.92, +3.53] neither | +4.12 [+1.75, +6.98] behind | +5.27 [+3.09, +7.89] behind |
| LW | -1.78 [-3.42, +0.63] neither | -0.66 [-2.41, +1.76] neither | +0.95 [-0.89, +3.16] neither |
| SW | -4.27 [-6.81, -1.94] ahead | -2.16 [-4.91, +0.45] neither | -1.34 [-3.91, +1.20] neither |
| LSW | -3.05 [-5.09, -0.48] ahead | +0.22 [-1.63, +2.69] neither | +1.15 [-0.59, +3.46] neither |
| Pool | -2.16 [-3.41, -0.90] ahead | +1.42 [+0.59, +2.27] behind | +2.68 [+1.95, +3.42] behind |
| combinations without setups | -2.03 [-3.74, +0.04] neither | +1.88 [+0.83, +3.26] behind | +3.52 [+2.49, +4.76] behind |
| combinations with setups | -2.30 [-3.66, -0.95] ahead | +0.95 [-0.09, +2.00] neither | +1.84 [+0.89, +2.79] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.33 [-3.54, +1.20] neither | +5.23 [+3.37, +7.42] behind | +7.01 [+5.20, +9.15] behind |
| L | +1.72 [-2.06, +6.35] neither | +8.04 [+4.26, +12.49] behind | +11.02 [+7.62, +15.07] behind |
| S | -5.56 [-7.57, -3.71] ahead | -4.80 [-6.75, -3.03] ahead | -4.11 [-6.07, -2.37] ahead |
| W | -2.36 [-3.49, -1.17] ahead | +2.02 [+1.19, +2.91] behind | +3.14 [+2.25, +4.02] behind |
| LS | -2.38 [-4.71, +0.67] neither | -1.78 [-4.18, +1.35] neither | -1.01 [-3.50, +2.16] neither |
| LW | -0.49 [-2.17, +1.95] neither | +4.61 [+2.93, +6.92] behind | +6.67 [+5.09, +8.71] behind |
| SW | -4.45 [-6.95, -2.16] ahead | -3.63 [-6.06, -1.36] ahead | -2.85 [-5.21, -0.78] ahead |
| LSW | -3.18 [-5.17, -0.65] ahead | -2.85 [-4.93, -0.21] ahead | -2.49 [-4.56, +0.06] neither |
| Pool | -2.25 [-3.61, -0.92] ahead | +0.85 [-0.29, +1.97] neither | +2.17 [+1.15, +3.20] behind |
| combinations without setups | -0.61 [-2.22, +1.36] neither | +4.97 [+3.71, +6.56] behind | +6.96 [+5.87, +8.35] behind |
| combinations with setups | -3.89 [-5.46, -2.57] ahead | -3.27 [-4.86, -1.85] ahead | -2.62 [-4.22, -1.25] ahead |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.04 [-4.52, -1.62] ahead | +4.20 [+2.75, +5.71] behind | +5.68 [+4.14, +7.32] behind |
| L | -3.20 [-4.98, -0.51] ahead | +3.25 [+0.93, +5.87] behind | +5.31 [+2.94, +7.96] behind |
| S | -1.25 [-3.08, +0.53] neither | +2.74 [+1.29, +4.20] behind | +3.40 [+1.98, +4.88] behind |
| W | -4.58 [-5.88, -3.41] ahead | -3.95 [-5.47, -2.55] ahead | -2.65 [-3.73, -1.44] ahead |
| LS | -0.58 [-2.58, +1.49] neither | +3.05 [+1.14, +5.11] behind | +4.19 [+2.49, +6.06] behind |
| LW | -3.69 [-4.89, -2.31] ahead | -2.59 [-3.96, -1.15] ahead | -0.99 [-2.70, +0.73] neither |
| SW | -3.96 [-6.21, -2.07] ahead | -1.86 [-4.11, +0.04] neither | -1.04 [-3.08, +0.79] neither |
| LSW | -3.73 [-4.95, -2.43] ahead | -0.48 [-1.69, +0.71] neither | +0.45 [-0.76, +1.57] neither |
| Pool | -3.00 [-3.91, -2.10] ahead | +0.55 [-0.24, +1.25] neither | +1.79 [+0.95, +2.55] behind |
| combinations without setups | -3.63 [-4.69, -2.36] ahead | +0.23 [-0.76, +1.16] neither | +1.84 [+0.72, +2.82] behind |
| combinations with setups | -2.38 [-3.56, -1.26] ahead | +0.86 [-0.09, +1.83] neither | +1.75 [+0.90, +2.65] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.90 [-3.14, -0.57] ahead | +4.64 [+3.59, +5.82] behind | +6.42 [+5.39, +7.58] behind |
| L | -2.18 [-4.08, +0.54] neither | +3.93 [+1.74, +6.64] behind | +6.83 [+4.48, +9.36] behind |
| S | -4.51 [-6.44, -2.70] ahead | -3.75 [-5.61, -2.07] ahead | -3.05 [-4.94, -1.32] ahead |
| W | -2.42 [-3.58, -1.31] ahead | +1.96 [+1.22, +2.75] behind | +3.08 [+2.27, +3.97] behind |
| LS | -3.40 [-5.27, -1.36] ahead | -2.80 [-4.76, -0.67] ahead | -2.04 [-4.04, +0.10] neither |
| LW | -2.42 [-3.68, -0.98] ahead | +2.58 [+1.24, +3.98] behind | +4.62 [+3.11, +6.06] behind |
| SW | -4.14 [-6.33, -2.35] ahead | -3.32 [-5.43, -1.50] ahead | -2.54 [-4.61, -0.79] ahead |
| LSW | -3.86 [-5.06, -2.58] ahead | -3.54 [-4.71, -2.26] ahead | -3.18 [-4.36, -1.93] ahead |
| Pool | -3.10 [-4.07, -2.16] ahead | -0.04 [-0.87, +0.78] neither | +1.27 [+0.47, +2.04] behind |
| combinations without setups | -2.23 [-3.18, -1.03] ahead | +3.28 [+2.29, +4.32] behind | +5.24 [+4.13, +6.26] behind |
| combinations with setups | -3.98 [-5.24, -2.87] ahead | -3.35 [-4.65, -2.20] ahead | -2.70 [-4.04, -1.53] ahead |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.26 | [+4.21, +8.71] | 1 | 0.0004 | behind |
| L | +9.44 | [+5.87, +13.50] | 1 | 0.0004 | behind |
| S | +2.27 | [+0.81, +3.63] | 1 | 0.0042 | behind |
| W | -2.59 | [-3.69, -1.41] | 0.0004 | 1 | ahead |
| LS | +5.27 | [+3.09, +7.89] | 1 | 0.0004 | behind |
| LW | +0.95 | [-0.89, +3.16] | 1 | 0.54 | not distinguishable |
| SW | -1.34 | [-3.91, +1.20] | 1 | 1 | not distinguishable |
| LSW | +1.15 | [-0.59, +3.46] | 1 | 0.52 | not distinguishable |

Row Sel, combinations of 8: ahead 1, behind 4, not distinguishable 3

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +5.68 | [+4.14, +7.32] | 1 / 1 | 0.00049 / 0.0004 | behind |
| L | +5.31 | [+2.94, +7.96] | 1 / 1 | 0.025 / 0.0004 | behind |
| S | +3.40 | [+1.98, +4.88] | 1 / 1 | 0.0039 / 0.0004 | behind |
| W | -2.65 | [-3.73, -1.44] | 0.0069 / 0.0004 | 1 / 1 | ahead |
| LS | +4.19 | [+2.49, +6.06] | 1 / 1 | 0.0023 / 0.0004 | behind |
| LW | -0.99 | [-2.70, +0.73] | 0.53 / 0.92 | 1 / 1 | not distinguishable |
| SW | -1.04 | [-3.08, +0.79] | 0.034 / 0.92 | 1 / 1 | not distinguishable |
| LSW | +0.45 | [-0.76, +1.57] | 1 / 1 | 0.77 / 0.84 | not distinguishable |

Row All, combinations of 8: ahead 1, behind 4, not distinguishable 3

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.01 | [+5.20, +9.15] | 1 | 0.0004 | behind |
| L | +11.02 | [+7.62, +15.07] | 1 | 0.0004 | behind |
| S | -4.11 | [-6.07, -2.37] | 0.0004 | 1 | ahead |
| W | +3.14 | [+2.25, +4.02] | 1 | 0.0004 | behind |
| LS | -1.01 | [-3.50, +2.16] | 1 | 1 | not distinguishable |
| LW | +6.67 | [+5.09, +8.71] | 1 | 0.0004 | behind |
| SW | -2.85 | [-5.21, -0.78] | 0.016 | 1 | ahead |
| LSW | -2.49 | [-4.56, +0.06] | 0.16 | 1 | not distinguishable |

Row Sel, combinations of 8: ahead 2, behind 4, not distinguishable 2

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.42 | [+5.39, +7.58] | 1 / 1 | 0.0003 / 0.0004 | behind |
| L | +6.83 | [+4.48, +9.36] | 1 / 1 | 0.011 / 0.0004 | behind |
| S | -3.05 | [-4.94, -1.32] | 0.0055 / 0.002 | 1 / 1 | ahead |
| W | +3.08 | [+2.27, +3.97] | 1 / 1 | 0.0042 / 0.0004 | behind |
| LS | -2.04 | [-4.04, +0.10] | 0.019 / 0.15 | 1 / 1 | not distinguishable |
| LW | +4.62 | [+3.11, +6.06] | 1 / 1 | 0.0059 / 0.0004 | behind |
| SW | -2.54 | [-4.61, -0.79] | 0.0013 / 0.0075 | 1 / 1 | ahead |
| LSW | -3.18 | [-4.36, -1.93] | 0.0061 / 0.0021 | 1 / 1 | ahead |

Row All, combinations of 8: ahead 3, behind 4, not distinguishable 1

### (iv) best rule minus policy (selected)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +3.48 [+0.96, +5.71] behind | 22.2 | 0.0 | 77.8 | 9 |
| L | +0.41 [-4.11, +4.07] neither | 22.2 | 0.0 | 77.8 | 9 |
| S | +6.14 [+3.97, +8.50] behind | 0.0 | 0.0 | 100.0 | 9 |
| W | +4.90 [+3.58, +6.27] behind | 0.0 | 0.0 | 100.0 | 9 |
| LS | +2.74 [-0.26, +5.10] neither | 11.1 | 0.0 | 88.9 | 9 |
| LW | +1.95 [-0.40, +3.58] neither | 11.1 | 0.0 | 88.9 | 9 |
| SW | +4.83 [+2.31, +7.68] behind | 11.1 | 0.0 | 88.9 | 9 |
| LSW | +3.52 [+0.94, +5.62] behind | 22.2 | 0.0 | 77.8 | 9 |
| Pool | +3.50 [+2.10, +4.97] behind | 12.5 | 0.0 | 87.5 | 9 |
| combinations without setups | +2.69 [+0.66, +4.42] behind | 13.9 | 0.0 | 86.1 | 9 |
| combinations with setups | +4.31 [+2.94, +6.03] behind | 11.1 | 0.0 | 88.9 | 9 |

### (iv) best rule minus policy (all_seeds)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +4.00 [+2.61, +5.39] behind | 6.7 | 0.0 | 93.3 | 9 |
| L | +4.14 [+1.43, +6.01] behind | 6.7 | 0.0 | 93.3 | 9 |
| S | +4.98 [+2.93, +7.18] behind | 4.4 | 0.0 | 95.6 | 9 |
| W | +4.96 [+3.74, +6.37] behind | 0.0 | 0.0 | 100.0 | 9 |
| LS | +3.74 [+1.67, +5.73] behind | 11.1 | 0.0 | 88.9 | 9 |
| LW | +3.92 [+2.51, +5.20] behind | 2.2 | 0.0 | 97.8 | 9 |
| SW | +4.48 [+2.49, +6.97] behind | 13.3 | 0.0 | 86.7 | 9 |
| LSW | +4.19 [+2.82, +5.52] behind | 6.7 | 0.0 | 93.3 | 9 |
| Pool | +4.30 [+3.28, +5.37] behind | 6.4 | 0.0 | 93.6 | 9 |
| combinations without setups | +4.25 [+2.95, +5.37] behind | 3.9 | 0.0 | 96.1 | 9 |
| combinations with setups | +4.35 [+3.13, +5.75] behind | 8.9 | 0.0 | 91.1 | 9 |

### Size of the solver's model: 72 valid records of CP-SAT started from the rule schedule; median [minimum, maximum] over the cells

| entry | cells | variables of the initial model | seconds to build the model | peak memory of the solve in GB | seconds until the first incumbent, on the record's clock (for a model configuration from the solver call, after the model build) |
|---|---|---|---|---|---|
| N | 9 | - | 25.66 [21.66, 27.34] | 1.1 [1.1, 1.1] | 0.0 [0.0, 0.0] |
| L | 9 | - | 27.96 [24.11, 30.83] | 1.1 [1.0, 1.2] | 0.0 [0.0, 0.0] |
| S | 9 | - | 112.82 [89.68, 143.42] | 1.9 [1.7, 3.5] | 0.0 [0.0, 0.0] |
| W | 9 | - | 24.63 [19.96, 27.03] | 1.1 [1.1, 1.1] | 0.0 [0.0, 0.0] |
| LS | 9 | - | 121.65 [104.51, 144.86] | 2.2 [1.8, 3.0] | 0.0 [0.0, 0.0] |
| LW | 9 | - | 29.16 [26.81, 30.48] | 1.1 [1.1, 1.1] | 0.0 [0.0, 0.0] |
| SW | 9 | - | 124.17 [91.13, 131.86] | 2.0 [1.9, 2.8] | 0.0 [0.0, 0.0] |
| LSW | 9 | - | 130.06 [94.03, 140.50] | 2.8 [1.7, 3.6] | 0.0 [0.0, 0.0] |
| Pool | 72 | - | 60.25 [19.96, 144.86] | 1.4 [1.0, 3.6] | 0.0 [0.0, 0.0] |
| combinations without setups | 36 | - | 27.04 [19.96, 30.83] | 1.1 [1.0, 1.2] | 0.0 [0.0, 0.0] |
| combinations with setups | 36 | - | 122.57 [89.68, 144.86] | 2.1 [1.7, 3.6] | 0.0 [0.0, 0.0] |

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

### (v) rule-started memetic algorithm reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 101 | 111 | 0.0 | 527.56 | 480.22 | 510.33 |
| L | 2 | 103 | 0.0 | 616.78 | 563.22 | 616.22 |
| S | 153 | 152 | 11.1 | 700.33 | 645.44 | 659.44 |
| W | not within the budget | not within the budget | 88.9 | 560.44 | 548.56 | 534.33 |
| LS | 8 | 117 | 0.0 | 782.22 | 723.89 | 762.11 |
| LW | 424 | 458 | 33.3 | 640.00 | 622.44 | 628.22 |
| SW | not within the budget | not within the budget | 55.6 | 731.22 | 708.00 | 698.22 |
| LSW | 264 | 485 | 44.4 | 810.44 | 774.78 | 783.78 |
| Pool | 159 | 205 | 29.2 | 671.12 | 633.32 | 649.08 |
| combinations without setups | 136 | 235 | 30.6 | 586.19 | 553.61 | 572.28 |
| combinations with setups | 179 | 193 | 27.8 | 756.06 | 713.03 | 725.89 |

### (v) rule-started CP-SAT reaches the policy (selected run)

| comb. | mean reached at s | median cell s | never % | arm mean at 0 | arm mean at end | policy mean |
|---|---|---|---|---|---|---|
| N | 88 | 96 | 0.0 | 527.56 | 476.78 | 510.33 |
| L | 5 | 56 | 0.0 | 616.78 | 555.11 | 616.22 |
| S | not within the budget | not within the budget | 100.0 | 700.33 | 688.78 | 659.44 |
| W | 137 | 154 | 0.0 | 560.44 | 518.00 | 534.33 |
| LS | not within the budget | not within the budget | 66.7 | 782.22 | 770.67 | 762.11 |
| LW | 84 | 103 | 0.0 | 640.00 | 588.89 | 628.22 |
| SW | not within the budget | not within the budget | 77.8 | 731.22 | 719.11 | 698.22 |
| LSW | not within the budget | not within the budget | 77.8 | 810.44 | 803.89 | 783.78 |
| Pool | 267 | 208 | 40.3 | 671.12 | 640.15 | 649.08 |
| combinations without setups | 75 | 106.5 | 0.0 | 586.19 | 534.69 | 572.28 |
| combinations with setups | not within the budget | not within the budget | 80.6 | 756.06 | 745.61 | 725.89 |

## Record counts per cell

| arm, size, combination | states |
|---|---|
| cpsat_rule|50x25n30|L | valid 9 |
| cpsat_rule|50x25n30|LS | valid 9 |
| cpsat_rule|50x25n30|LSW | valid 9 |
| cpsat_rule|50x25n30|LW | valid 9 |
| cpsat_rule|50x25n30|N | valid 9 |
| cpsat_rule|50x25n30|S | valid 9 |
| cpsat_rule|50x25n30|SW | valid 9 |
| cpsat_rule|50x25n30|W | valid 9 |
| cpsat_rule|80x25n30|L | valid 9 |
| cpsat_rule|80x25n30|LS | valid 9 |
| cpsat_rule|80x25n30|LSW | valid 9 |
| cpsat_rule|80x25n30|LW | valid 9 |
| cpsat_rule|80x25n30|N | valid 9 |
| cpsat_rule|80x25n30|S | valid 9 |
| cpsat_rule|80x25n30|SW | valid 9 |
| cpsat_rule|80x25n30|W | valid 9 |
| memetic_rule|50x25n30|L | valid 9 |
| memetic_rule|50x25n30|LS | valid 9 |
| memetic_rule|50x25n30|LSW | valid 9 |
| memetic_rule|50x25n30|LW | valid 9 |
| memetic_rule|50x25n30|N | valid 9 |
| memetic_rule|50x25n30|S | valid 9 |
| memetic_rule|50x25n30|SW | valid 9 |
| memetic_rule|50x25n30|W | valid 9 |
| memetic_rule|80x25n30|L | valid 9 |
| memetic_rule|80x25n30|LS | valid 9 |
| memetic_rule|80x25n30|LSW | valid 9 |
| memetic_rule|80x25n30|LW | valid 9 |
| memetic_rule|80x25n30|N | valid 9 |
| memetic_rule|80x25n30|S | valid 9 |
| memetic_rule|80x25n30|SW | valid 9 |
| memetic_rule|80x25n30|W | valid 9 |
