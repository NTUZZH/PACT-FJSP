# Timed campaign against the policy (full)

- setting: environment exact, training seeds 301,302,303,304,305 (5 runs per arm); policy: selected training run s304; CP-SAT configuration: lns
- clock origin: request (time since the request for a schedule, a cold start for every method: the dispatching rules from t_all of the rule record (all four rule schedules exist), the policy's greedy schedule from its t in the selected run's policy record (process start, checkpoint load included), the policy, sampled schedules, at its own t (best so far; after the record's budget the last value is held, which favours the other method); a search started from the rules at run time t - t_all of the start clock, a search started from the policy at run time t - its greedy t; no schedule before the offset; every training run takes the offsets of the selected run's records)
- budgets 10, 60, 300, 600 s; verdict words at the stated budget of 600 s only; B = 20000; differences are first-named minus second in percent of the second; the interval of a selected run is the instance-level cluster bootstrap, of every run the crossed hierarchical bootstrap
- request clock: budgets per size 50x25n30 60, 300, 600 s; 80x25n30 60, 300, 600 s; 10x25 10, 60, 300, 600 s; a number needs a schedule of both compared methods on every cell, else "no schedule yet" with the count of such cells; (held) marks a value of the policy, sampled schedules held after the budget of its records; valid records of the rule arm 384, of the policy arm 384
- campaign: 50x25n30 80x25n30 = memetic_rule@600:3-11 memetic_policy@600:3-11 cpsat_rule@600:3-11 cpsat_policy@600:3-11 policy_sample@120:3-11; 10x25 = memetic_rule@600 cpsat_rule@600:0-9 policy_sample@60
- instances per size and arm (the instance range of a token is the design of its arm; a token without one plans the size's evidence instances): 50x25n30 memetic_rule 3-11 (9), memetic_policy 3-11 (9), cpsat_rule 3-11 (9), cpsat_policy 3-11 (9), policy_sample 3-11 (9); 80x25n30 memetic_rule 3-11 (9), memetic_policy 3-11 (9), cpsat_rule 3-11 (9), cpsat_policy 3-11 (9), policy_sample 3-11 (9); 10x25 memetic_rule 0-29 (30), cpsat_rule 0-9 (10), policy_sample 0-29 (30)
- cells 896, valid 896
- pooled rows beside Pool: the combinations without setups (N, L, W, LW); the combinations with setups (S, LS, SW, LSW), each over the instances present in every combination of its group, with the estimator of Pool

## 50x25n30 (instances 3-11; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +2.09 [-1.33, +5.53] neither | +9.48 [+6.86, +11.94] behind | +10.93 [+8.32, +13.53] behind |
| L | -0.73 [-2.30, +0.80] neither | +6.53 [+5.09, +7.83] behind | +8.14 [+6.93, +9.31] behind |
| S | -2.24 [-3.92, -0.62] ahead | +3.10 [+1.86, +4.41] behind | +4.16 [+2.94, +5.40] behind |
| W | -1.25 [-3.73, +1.39] neither | +1.76 [-0.87, +4.29] neither | +2.96 [+0.67, +5.17] behind |
| LS | -0.29 [-1.47, +0.82] neither | +5.15 [+3.62, +6.78] behind | +5.91 [+4.20, +7.70] behind |
| LW | -3.85 [-5.56, -2.28] ahead | +0.46 [-1.76, +2.56] neither | +1.88 [-0.42, +4.14] neither |
| SW | -1.81 [-3.20, -0.32] ahead | +2.07 [+0.54, +3.77] behind | +2.96 [+1.51, +4.60] behind |
| LSW | -3.08 [-4.48, -1.73] ahead | +2.36 [+1.40, +3.37] behind | +3.63 [+2.75, +4.58] behind |
| Pool | -1.39 [-2.35, -0.39] ahead | +3.86 [+2.95, +4.75] behind | +5.07 [+4.22, +5.90] behind |
| combinations without setups | -0.93 [-2.57, +0.71] neither | +4.56 [+2.94, +5.95] behind | +5.98 [+4.48, +7.28] behind |
| combinations with setups | -1.85 [-3.02, -0.70] ahead | +3.17 [+2.17, +4.17] behind | +4.16 [+3.12, +5.21] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +4.07 [+1.19, +7.01] behind | +10.41 [+7.72, +13.10] behind | +11.57 [+9.15, +13.98] behind |
| L | -0.27 [-1.58, +1.13] neither | +7.33 [+6.03, +8.82] behind | +9.00 [+7.78, +10.26] behind |
| S | -5.39 [-7.41, -3.52] ahead | -2.05 [-4.10, -0.12] ahead | -1.13 [-3.00, +0.71] neither |
| W | +2.67 [+0.60, +5.22] behind | +5.55 [+3.68, +8.06] behind | +6.76 [+4.99, +9.18] behind |
| LS | -4.07 [-5.34, -2.78] ahead | -1.91 [-3.42, -0.37] ahead | -0.46 [-1.59, +0.70] neither |
| LW | -0.70 [-2.16, +0.61] neither | +4.28 [+2.52, +5.77] behind | +6.18 [+4.78, +7.54] behind |
| SW | -2.77 [-4.78, -0.78] ahead | -0.60 [-2.18, +1.12] neither | +0.13 [-1.30, +1.65] neither |
| LSW | -4.09 [-5.67, -2.42] ahead | -2.66 [-4.02, -1.28] ahead | -2.14 [-3.50, -0.75] ahead |
| Pool | -1.32 [-2.17, -0.39] ahead | +2.54 [+1.82, +3.33] behind | +3.74 [+3.08, +4.40] behind |
| combinations without setups | +1.44 [+0.03, +2.85] behind | +6.89 [+5.72, +8.08] behind | +8.38 [+7.33, +9.40] behind |
| combinations with setups | -4.08 [-5.46, -2.70] ahead | -1.80 [-3.09, -0.49] ahead | -0.90 [-2.12, +0.32] neither |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.45 [-3.40, +1.18] neither | +5.75 [+3.51, +8.14] behind | +7.15 [+4.94, +9.53] behind |
| L | -1.43 [-2.85, -0.14] ahead | +5.79 [+4.18, +7.36] behind | +7.39 [+5.87, +8.91] behind |
| S | -0.72 [-2.61, +1.08] neither | +4.70 [+3.29, +6.13] behind | +5.77 [+4.43, +7.16] behind |
| W | -2.16 [-3.72, -0.50] ahead | +0.84 [-1.20, +2.98] neither | +2.05 [+0.14, +4.10] behind |
| LS | -0.56 [-2.11, +1.09] neither | +4.85 [+3.40, +6.22] behind | +5.60 [+4.05, +7.07] behind |
| LW | -4.36 [-5.79, -2.94] ahead | -0.08 [-2.08, +1.95] neither | +1.33 [-0.79, +3.49] neither |
| SW | -1.47 [-2.76, -0.18] ahead | +2.43 [+0.95, +3.93] behind | +3.32 [+1.92, +4.76] behind |
| LSW | -3.01 [-4.59, -1.50] ahead | +2.43 [+1.22, +3.71] behind | +3.70 [+2.58, +4.94] behind |
| Pool | -1.90 [-2.82, -0.96] ahead | +3.34 [+2.44, +4.38] behind | +4.54 [+3.64, +5.57] behind |
| combinations without setups | -2.35 [-3.50, -1.09] ahead | +3.07 [+1.57, +4.64] behind | +4.48 [+2.94, +6.05] behind |
| combinations with setups | -1.44 [-2.74, -0.18] ahead | +3.60 [+2.63, +4.61] behind | +4.60 [+3.60, +5.63] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.50 [-1.53, +2.91] neither | +6.64 [+4.48, +9.00] behind | +7.78 [+5.51, +10.11] behind |
| L | -0.97 [-2.37, +0.48] neither | +6.58 [+5.06, +8.15] behind | +8.24 [+6.70, +9.75] behind |
| S | -3.91 [-6.20, -1.70] ahead | -0.53 [-2.70, +1.41] neither | +0.40 [-1.54, +2.15] neither |
| W | +1.72 [+0.54, +3.07] behind | +4.59 [+3.51, +5.89] behind | +5.79 [+4.75, +7.03] behind |
| LS | -4.33 [-5.96, -2.45] ahead | -2.18 [-3.86, -0.33] ahead | -0.73 [-2.21, +0.84] neither |
| LW | -1.23 [-2.43, -0.11] ahead | +3.72 [+2.33, +4.96] behind | +5.61 [+4.40, +6.80] behind |
| SW | -2.44 [-4.26, -0.66] ahead | -0.25 [-1.81, +1.27] neither | +0.47 [-0.93, +1.84] neither |
| LSW | -4.01 [-6.07, -2.13] ahead | -2.59 [-4.35, -0.95] ahead | -2.07 [-3.78, -0.45] ahead |
| Pool | -1.83 [-2.79, -0.90] ahead | +2.00 [+1.00, +2.92] behind | +3.19 [+2.21, +4.05] behind |
| combinations without setups | +0.00 [-1.07, +1.20] neither | +5.38 [+4.25, +6.56] behind | +6.86 [+5.66, +8.01] behind |
| combinations with setups | -3.67 [-5.22, -2.19] ahead | -1.39 [-2.82, +0.00] neither | -0.48 [-1.85, +0.84] neither |

### (ii) policy-started arm minus policy: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -5.04 [-8.28, -1.77] ahead | -8.85 [-11.65, -6.05] ahead | -9.72 [-12.48, -6.93] ahead |
| L | -1.64 [-2.27, -1.04] ahead | -6.05 [-6.72, -5.34] ahead | -7.03 [-7.82, -6.24] ahead |
| S | -1.27 [-1.75, -0.85] ahead | -3.90 [-4.41, -3.42] ahead | -4.36 [-4.80, -3.90] ahead |
| W | -2.84 [-5.51, -0.86] ahead | -3.49 [-6.08, -1.58] ahead | -3.85 [-6.37, -1.92] ahead |
| LS | -2.77 [-3.77, -1.74] ahead | -5.41 [-6.26, -4.56] ahead | -5.93 [-6.91, -5.01] ahead |
| LW | -0.64 [-0.91, -0.40] ahead | -2.24 [-3.26, -1.34] ahead | -3.23 [-4.38, -2.16] ahead |
| SW | -1.09 [-1.58, -0.64] ahead | -3.12 [-4.22, -2.22] ahead | -3.66 [-4.85, -2.74] ahead |
| LSW | -0.14 [-0.24, -0.04] ahead | -3.32 [-3.82, -2.81] ahead | -3.83 [-4.37, -3.29] ahead |
| Pool | -1.93 [-2.57, -1.37] ahead | -4.55 [-5.18, -3.94] ahead | -5.20 [-5.86, -4.56] ahead |
| combinations without setups | -2.54 [-3.88, -1.34] ahead | -5.16 [-6.37, -4.04] ahead | -5.96 [-7.11, -4.83] ahead |
| combinations with setups | -1.32 [-1.67, -1.01] ahead | -3.94 [-4.53, -3.45] ahead | -4.45 [-5.01, -3.96] ahead |

### (ii) policy-started arm minus policy: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -5.18 [-7.70, -2.71] ahead | -9.30 [-12.30, -6.32] ahead | -9.96 [-12.81, -7.12] ahead |
| L | -1.84 [-2.51, -1.27] ahead | -6.41 [-7.25, -5.61] ahead | -7.40 [-8.10, -6.81] ahead |
| S | -0.10 [-0.17, -0.02] ahead | -0.66 [-0.89, -0.44] ahead | -1.22 [-1.52, -0.91] ahead |
| W | -3.68 [-5.51, -2.24] ahead | -5.93 [-8.00, -4.34] ahead | -6.35 [-8.44, -4.68] ahead |
| LS | -0.36 [-0.64, -0.11] ahead | -1.88 [-2.57, -1.19] ahead | -2.49 [-3.32, -1.61] ahead |
| LW | -1.63 [-2.16, -1.11] ahead | -4.55 [-5.70, -3.56] ahead | -5.54 [-6.68, -4.49] ahead |
| SW | -0.38 [-0.61, -0.16] ahead | -1.02 [-1.27, -0.79] ahead | -1.47 [-1.79, -1.21] ahead |
| LSW | -0.22 [-0.34, -0.10] ahead | -0.67 [-0.98, -0.42] ahead | -1.15 [-1.59, -0.79] ahead |
| Pool | -1.67 [-2.21, -1.18] ahead | -3.80 [-4.35, -3.29] ahead | -4.44 [-4.97, -3.99] ahead |
| combinations without setups | -3.08 [-4.18, -2.10] ahead | -6.55 [-7.66, -5.49] ahead | -7.31 [-8.45, -6.27] ahead |
| combinations with setups | -0.26 [-0.36, -0.17] ahead | -1.06 [-1.25, -0.88] ahead | -1.58 [-1.86, -1.29] ahead |

### (iii) policy-started minus rule-started arm: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.31 [-4.45, -2.25] ahead | -0.38 [-1.52, +0.83] neither | -0.02 [-0.94, +0.90] neither |
| L | -2.34 [-4.34, -0.55] ahead | +0.08 [-0.81, +1.07] neither | +0.52 [-0.02, +1.11] neither |
| S | -3.49 [-5.04, -1.95] ahead | -0.93 [-2.07, +0.27] neither | -0.39 [-1.50, +0.63] neither |
| W | -4.18 [-5.44, -2.79] ahead | -1.90 [-3.70, -0.03] ahead | -1.09 [-2.81, +0.60] neither |
| LS | -3.05 [-4.60, -1.30] ahead | -0.56 [-1.75, +0.53] neither | -0.40 [-1.48, +0.52] neither |
| LW | -4.47 [-6.06, -3.00] ahead | -1.83 [-3.50, -0.18] ahead | -1.46 [-3.01, +0.10] neither |
| SW | -2.89 [-4.10, -1.69] ahead | -1.13 [-2.72, +0.28] neither | -0.82 [-2.28, +0.53] neither |
| LSW | -3.21 [-4.61, -1.85] ahead | -1.04 [-1.98, -0.19] ahead | -0.35 [-1.22, +0.42] neither |
| Pool | -3.37 [-4.17, -2.46] ahead | -0.96 [-1.63, -0.28] ahead | -0.50 [-1.09, +0.14] neither |
| combinations without setups | -3.57 [-4.57, -2.58] ahead | -1.01 [-1.95, +0.05] neither | -0.51 [-1.32, +0.40] neither |
| combinations with setups | -3.16 [-4.39, -1.91] ahead | -0.91 [-1.99, -0.02] ahead | -0.49 [-1.52, +0.34] neither |

### (iii) policy-started minus rule-started arm: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.49 [-2.27, -0.73] ahead | -0.06 [-1.04, +0.89] neither | +0.30 [-0.90, +1.50] neither |
| L | -2.12 [-3.05, -1.16] ahead | +0.43 [-0.31, +1.28] neither | +0.92 [+0.16, +1.69] behind |
| S | -5.48 [-7.50, -3.61] ahead | -2.69 [-4.76, -0.75] ahead | -2.34 [-4.14, -0.58] ahead |
| W | -1.20 [-2.02, -0.42] ahead | -0.80 [-1.45, -0.18] ahead | -0.09 [-1.13, +0.78] neither |
| LS | -4.41 [-5.73, -3.06] ahead | -3.75 [-5.46, -1.94] ahead | -2.93 [-4.44, -1.28] ahead |
| LW | -2.32 [-3.73, -0.98] ahead | -0.49 [-1.86, +0.87] neither | +0.28 [-0.85, +1.42] neither |
| SW | -3.15 [-5.06, -1.25] ahead | -1.62 [-3.08, +0.04] neither | -1.34 [-2.85, +0.24] neither |
| LSW | -4.30 [-5.81, -2.73] ahead | -3.32 [-4.65, -1.95] ahead | -3.26 [-4.61, -1.83] ahead |
| Pool | -3.06 [-3.72, -2.36] ahead | -1.53 [-2.22, -0.81] ahead | -1.06 [-1.73, -0.38] ahead |
| combinations without setups | -1.78 [-2.46, -1.12] ahead | -0.23 [-0.76, +0.32] neither | +0.35 [-0.28, +0.96] neither |
| combinations with setups | -4.33 [-5.67, -2.99] ahead | -2.84 [-4.18, -1.48] ahead | -2.47 [-3.78, -1.15] ahead |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +10.93 | [+8.32, +13.53] | 1 | 0.0004 | behind |
| L | +8.14 | [+6.93, +9.31] | 1 | 0.0004 | behind |
| S | +4.16 | [+2.94, +5.40] | 1 | 0.0004 | behind |
| W | +2.96 | [+0.67, +5.17] | 1 | 0.01 | behind |
| LS | +5.91 | [+4.20, +7.70] | 1 | 0.0004 | behind |
| LW | +1.88 | [-0.42, +4.14] | 1 | 0.055 | not distinguishable |
| SW | +2.96 | [+1.51, +4.60] | 1 | 0.0004 | behind |
| LSW | +3.63 | [+2.75, +4.58] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 7, not distinguishable 1

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.15 | [+4.94, +9.53] | 1 / 1 | 0.0059 / 0.0004 | behind |
| L | +7.39 | [+5.87, +8.91] | 1 / 1 | 0.00046 / 0.0004 | behind |
| S | +5.77 | [+4.43, +7.16] | 1 / 1 | 0.0005 / 0.0004 | behind |
| W | +2.05 | [+0.14, +4.10] | 1 / 1 | 0.011 / 0.036 | behind |
| LS | +5.60 | [+4.05, +7.07] | 1 / 1 | 0.00021 / 0.0004 | behind |
| LW | +1.33 | [-0.79, +3.49] | 1 / 1 | 0.011 / 0.11 | not distinguishable |
| SW | +3.32 | [+1.92, +4.76] | 1 / 1 | 0.0013 / 0.0004 | behind |
| LSW | +3.70 | [+2.58, +4.94] | 1 / 1 | 0.00061 / 0.0004 | behind |

Row All, combinations of 8: ahead 0, behind 7, not distinguishable 1

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +11.57 | [+9.15, +13.98] | 1 | 0.0004 | behind |
| L | +9.00 | [+7.78, +10.26] | 1 | 0.0004 | behind |
| S | -1.13 | [-3.00, +0.71] | 0.85 | 1 | not distinguishable |
| W | +6.76 | [+4.99, +9.18] | 1 | 0.0004 | behind |
| LS | -0.46 | [-1.59, +0.70] | 1 | 1 | not distinguishable |
| LW | +6.18 | [+4.78, +7.54] | 1 | 0.0004 | behind |
| SW | +0.13 | [-1.30, +1.65] | 1 | 1 | not distinguishable |
| LSW | -2.14 | [-3.50, -0.75] | 0.0088 | 1 | ahead |

Row Sel, combinations of 8: ahead 1, behind 4, not distinguishable 3

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.78 | [+5.51, +10.11] | 1 / 1 | 0.0073 / 0.0004 | behind |
| L | +8.24 | [+6.70, +9.75] | 1 / 1 | 0.00036 / 0.0004 | behind |
| S | +0.40 | [-1.54, +2.15] | 1 / 1 | 0.56 / 1 | not distinguishable |
| W | +5.79 | [+4.75, +7.03] | 1 / 1 | 0.00074 / 0.0004 | behind |
| LS | -0.73 | [-2.21, +0.84] | 0.23 / 1 | 1 / 1 | not distinguishable |
| LW | +5.61 | [+4.40, +6.80] | 1 / 1 | 0.00044 / 0.0004 | behind |
| SW | +0.47 | [-0.93, +1.84] | 1 / 1 | 0.47 / 1 | not distinguishable |
| LSW | -2.07 | [-3.78, -0.45] | 0.0074 / 0.045 | 1 / 1 | ahead |

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

### The policy, sampled schedules: records of 72 cells, budget 120 s (values after it are held)

### (vii) the policy, sampled schedules minus the greedy schedule

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -5.17 [-7.58, -2.85] ahead | -5.63 [-8.10, -3.25] ahead (held) | -5.63 [-8.10, -3.25] ahead (held) |
| L | -0.98 [-1.40, -0.62] ahead | -1.43 [-1.88, -1.04] ahead (held) | -1.43 [-1.88, -1.04] ahead (held) |
| S | -1.59 [-2.50, -0.79] ahead | -1.78 [-2.61, -1.08] ahead (held) | -1.78 [-2.61, -1.08] ahead (held) |
| W | -3.15 [-5.39, -1.50] ahead | -3.30 [-5.52, -1.64] ahead (held) | -3.30 [-5.52, -1.64] ahead (held) |
| LS | -2.20 [-3.29, -1.12] ahead | -2.45 [-3.46, -1.42] ahead (held) | -2.45 [-3.46, -1.42] ahead (held) |
| LW | -1.58 [-2.10, -1.04] ahead | -1.65 [-2.16, -1.10] ahead (held) | -1.65 [-2.16, -1.10] ahead (held) |
| SW | -2.57 [-3.52, -1.91] ahead | -2.78 [-3.63, -2.23] ahead (held) | -2.78 [-3.63, -2.23] ahead (held) |
| LSW | -1.66 [-2.11, -1.25] ahead | -2.01 [-2.37, -1.64] ahead (held) | -2.01 [-2.37, -1.64] ahead (held) |
| Pool | -2.36 [-2.85, -1.90] ahead | -2.63 [-3.10, -2.18] ahead (held) | -2.63 [-3.10, -2.18] ahead (held) |
| combinations without setups | -2.72 [-3.65, -1.86] ahead | -3.00 [-3.98, -2.09] ahead (held) | -3.00 [-3.98, -2.09] ahead (held) |
| combinations with setups | -2.00 [-2.51, -1.53] ahead | -2.26 [-2.69, -1.87] ahead (held) | -2.26 [-2.69, -1.87] ahead (held) |

### (viii) the policy, sampled schedules minus the rule-started memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.38 [-4.45, -2.26] ahead | +3.18 [+1.97, +4.39] behind (held) | +4.55 [+3.40, +5.73] behind (held) |
| L | -1.71 [-2.99, -0.54] ahead | +5.01 [+3.70, +6.28] behind (held) | +6.59 [+5.45, +7.80] behind (held) |
| S | -3.81 [-5.14, -2.66] ahead | +1.25 [+0.45, +2.22] behind (held) | +2.28 [+1.47, +3.21] behind (held) |
| W | -4.47 [-5.78, -3.10] ahead | -1.68 [-3.58, +0.26] neither (held) | -0.51 [-2.30, +1.43] neither (held) |
| LS | -2.49 [-3.95, -0.92] ahead | +2.55 [+1.41, +3.57] behind (held) | +3.28 [+2.08, +4.32] behind (held) |
| LW | -5.38 [-6.67, -4.16] ahead | -1.23 [-2.94, +0.42] neither (held) | +0.17 [-1.65, +2.02] neither (held) |
| SW | -4.35 [-5.41, -3.41] ahead | -0.78 [-1.94, +0.28] neither (held) | +0.08 [-0.96, +1.07] neither (held) |
| LSW | -4.69 [-5.88, -3.47] ahead | +0.30 [-0.50, +1.10] neither (held) | +1.55 [+0.74, +2.36] behind (held) |
| Pool | -3.79 [-4.43, -3.05] ahead | +1.07 [+0.44, +1.80] behind (held) | +2.25 [+1.64, +2.98] behind (held) |
| combinations without setups | -3.74 [-4.66, -2.88] ahead | +1.32 [+0.15, +2.51] behind (held) | +2.70 [+1.54, +3.93] behind (held) |
| combinations with setups | -3.84 [-4.90, -2.76] ahead | +0.83 [+0.06, +1.54] behind (held) | +1.80 [+0.99, +2.52] behind (held) |

### (viii) the policy, sampled schedules minus the rule-started CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.47 [-2.19, -0.72] ahead | +4.03 [+3.45, +4.72] behind (held) | +5.15 [+4.23, +6.07] behind (held) |
| L | -1.26 [-2.43, +0.11] neither | +5.80 [+4.63, +7.21] behind (held) | +7.44 [+6.32, +8.62] behind (held) |
| S | -6.91 [-8.51, -5.39] ahead | -3.82 [-5.32, -2.48] ahead (held) | -2.92 [-4.23, -1.74] ahead (held) |
| W | -0.68 [-1.44, +0.04] neither | +1.96 [+1.42, +2.56] behind (held) | +3.14 [+2.43, +3.83] behind (held) |
| LS | -6.18 [-7.71, -4.43] ahead | -4.32 [-5.85, -2.59] ahead (held) | -2.90 [-4.25, -1.41] ahead (held) |
| LW | -2.28 [-3.44, -1.21] ahead | +2.54 [+1.20, +3.68] behind (held) | +4.41 [+3.45, +5.40] behind (held) |
| SW | -5.29 [-6.97, -3.74] ahead | -3.38 [-4.55, -2.32] ahead (held) | -2.67 [-3.71, -1.70] ahead (held) |
| LSW | -5.68 [-7.27, -4.06] ahead | -4.62 [-5.90, -3.36] ahead (held) | -4.11 [-5.39, -2.84] ahead (held) |
| Pool | -3.72 [-4.37, -2.94] ahead | -0.23 [-0.87, +0.47] neither (held) | +0.94 [+0.36, +1.55] behind (held) |
| combinations without setups | -1.42 [-2.23, -0.66] ahead | +3.58 [+2.94, +4.28] behind (held) | +5.04 [+4.49, +5.69] behind (held) |
| combinations with setups | -6.02 [-7.33, -4.68] ahead | -4.04 [-5.23, -2.83] ahead (held) | -3.15 [-4.27, -2.02] ahead (held) |

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started memetic algorithm (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +4.55 | [+3.40, +5.73] | 1 | 0.0004 | behind |
| L | +6.59 | [+5.45, +7.80] | 1 | 0.0004 | behind |
| S | +2.28 | [+1.47, +3.21] | 1 | 0.0004 | behind |
| W | -0.51 | [-2.30, +1.43] | 1 | 1 | not distinguishable |
| LS | +3.28 | [+2.08, +4.32] | 1 | 0.0004 | behind |
| LW | +0.17 | [-1.65, +2.02] | 1 | 1 | not distinguishable |
| SW | +0.08 | [-0.96, +1.07] | 1 | 1 | not distinguishable |
| LSW | +1.55 | [+0.74, +2.36] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 5, not distinguishable 3

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started CP-SAT (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +5.15 | [+4.23, +6.07] | 1 | 0.0004 | behind |
| L | +7.44 | [+6.32, +8.62] | 1 | 0.0004 | behind |
| S | -2.92 | [-4.23, -1.74] | 0.0004 | 1 | ahead |
| W | +3.14 | [+2.43, +3.83] | 1 | 0.0004 | behind |
| LS | -2.90 | [-4.25, -1.41] | 0.0005 | 1 | ahead |
| LW | +4.41 | [+3.45, +5.40] | 1 | 0.0004 | behind |
| SW | -2.67 | [-3.71, -1.70] | 0.0004 | 1 | ahead |
| LSW | -4.11 | [-5.39, -2.84] | 0.0004 | 1 | ahead |

Row Sel, combinations of 8: ahead 4, behind 4, not distinguishable 0

### (v) rule-started memetic algorithm reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 5 | 63 | 0.0 | 4 | 362.56 | 325.56 | 361.22 |
| L | 66 | 63 | 0.0 | 4 | 447.67 | 395.78 | 427.89 |
| S | 88 | 89 | 0.0 | 4 | 485.67 | 439.11 | 457.11 |
| W | 111 | 68 | 33.3 | 5 | 387.44 | 367.11 | 377.78 |
| LS | 75 | 77 | 0.0 | 4 | 569.89 | 515.33 | 545.89 |
| LW | 258 | 214 | 33.3 | 5 | 471.11 | 440.89 | 448.78 |
| SW | 91 | 78 | 0.0 | 5 | 505.78 | 478.00 | 491.78 |
| LSW | 113 | 99 | 0.0 | 5 | 591.89 | 545.67 | 565.33 |
| Pool | 84 | 77.5 | 8.3 | 5 | 474.61 | 438.43 | 459.47 |
| combinations without setups | 75 | 66.5 | 16.7 | 5 | 416.03 | 382.33 | 403.92 |
| combinations with setups | 87 | 79.5 | 0.0 | 5 | 533.19 | 494.53 | 515.03 |

### (v) rule-started memetic algorithm reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 101 | 96 | 0.0 (held) | 4 | 362.56 | 325.56 | 340.33 |
| L | 78 | 75 | 0.0 (held) | 4 | 447.67 | 395.78 | 421.78 |
| S | 137 (held) | 160 (held) | 0.0 (held) | 4 | 485.67 | 439.11 | 449.00 |
| W | not within the budget (held) | 106 | 33.3 (held) | 5 | 387.44 | 367.11 | 364.89 |
| LS | 95 | 86 | 11.1 (held) | 4 | 569.89 | 515.33 | 532.22 |
| LW | 565 (held) | 478 (held) | 44.4 (held) | 5 | 471.11 | 440.89 | 441.33 |
| SW | 535 (held) | 349 (held) | 44.4 (held) | 5 | 505.78 | 478.00 | 478.11 |
| LSW | 237 (held) | 224 (held) | 11.1 (held) | 5 | 591.89 | 545.67 | 554.00 |
| Pool | 181 (held) | 141.5 (held) | 18.1 (held) | 5 | 474.61 | 438.43 | 447.71 |
| combinations without setups | 198 (held) | 102.5 | 19.4 (held) | 5 | 416.03 | 382.33 | 392.08 |
| combinations with setups | 168 (held) | 192.5 (held) | 16.7 (held) | 5 | 533.19 | 494.53 | 503.33 |

### (v) rule-started CP-SAT reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 8 | 37 | 0.0 | 4 | 362.33 | 323.67 | 361.22 |
| L | 62 | 68 | 0.0 | 4 | 447.56 | 392.67 | 427.89 |
| S | not within the budget | not within the budget | 55.6 | 4 | 486.11 | 462.89 | 457.11 |
| W | 30 | 36 | 0.0 | 5 | 386.89 | 353.78 | 377.78 |
| LS | not within the budget | not within the budget | 55.6 | 4 | 571.89 | 548.56 | 545.89 |
| LW | 73 | 62 | 0.0 | 5 | 470.33 | 422.78 | 448.78 |
| SW | 579 | 362 | 44.4 | 5 | 509.44 | 491.56 | 491.78 |
| LSW | not within the budget | not within the budget | 77.8 | 5 | 592.11 | 578.22 | 565.33 |
| Pool | 140 | 110.5 | 29.2 | 5 | 478.28 | 446.76 | 459.47 |
| combinations without setups | 45 | 53.5 | 0.0 | 5 | 416.69 | 373.22 | 403.92 |
| combinations with setups | not within the budget | not within the budget | 58.3 | 5 | 539.86 | 520.31 | 515.03 |

### (v) rule-started CP-SAT reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 88 | 82 | 0.0 (held) | 4 | 362.33 | 323.67 | 340.33 |
| L | 83 | 81 | 0.0 (held) | 4 | 447.56 | 392.67 | 421.78 |
| S | not within the budget (held) | not within the budget (held) | 100.0 (held) | 4 | 486.11 | 462.89 | 449.00 |
| W | 78 | 46 | 0.0 (held) | 5 | 386.89 | 353.78 | 364.89 |
| LS | not within the budget (held) | not within the budget (held) | 88.9 (held) | 4 | 571.89 | 548.56 | 532.22 |
| LW | 137 (held) | 131 (held) | 0.0 (held) | 5 | 470.33 | 422.78 | 441.33 |
| SW | not within the budget (held) | not within the budget (held) | 100.0 (held) | 5 | 509.44 | 491.56 | 478.11 |
| LSW | not within the budget (held) | not within the budget (held) | 88.9 (held) | 5 | 592.11 | 578.22 | 554.00 |
| Pool | 524 (held) | 257 (held) | 47.2 (held) | 5 | 478.28 | 446.76 | 447.71 |
| combinations without setups | 92 | 82 | 0.0 (held) | 5 | 416.69 | 373.22 | 392.08 |
| combinations with setups | not within the budget (held) | not within the budget (held) | 94.4 (held) | 5 | 539.86 | 520.31 | 503.33 |

### (ix) time of the first schedule (seconds since the request)

| method | time | median | minimum | maximum | cells |
|---|---|---|---|---|---|
| dispatching rules (best of four) | t_all | 3.46 | 2.59 | 4.33 | 72 |
| policy, greedy schedule | t | 10.11 | 8.95 | 10.88 | 72 |
| policy, greedy schedule, from loaded | t_resident | 8.69 | 7.65 | 9.44 | 72 |
| dispatching rules (best of four), combinations without setups | t_all | 3.44 | 2.69 | 4.33 | 36 |
| policy, greedy schedule, combinations without setups | t | 10.06 | 9.05 | 10.88 | 36 |
| policy, greedy schedule, from loaded, combinations without setups | t_resident | 8.63 | 7.73 | 9.44 | 36 |
| dispatching rules (best of four), combinations with setups | t_all | 3.48 | 2.59 | 4.32 | 36 |
| policy, greedy schedule, combinations with setups | t | 10.16 | 8.95 | 10.80 | 36 |
| policy, greedy schedule, from loaded, combinations with setups | t_resident | 8.73 | 7.65 | 9.32 | 36 |

### Curves: first grid time at which every cell of the method has a schedule

| method | from s |
|---|---|
| memetic_rule | 4.64 |
| memetic_policy | 12.1 |
| cpsat_rule | 4.64 |
| cpsat_policy | 12.1 |
| policy_sample | 12.1 |
| memetic_rule, combinations without setups | 4.64 |
| memetic_policy, combinations without setups | 12.1 |
| cpsat_rule, combinations without setups | 4.64 |
| cpsat_policy, combinations without setups | 12.1 |
| policy_sample, combinations without setups | 12.1 |
| memetic_rule, combinations with setups | 4.64 |
| memetic_policy, combinations with setups | 12.1 |
| cpsat_rule, combinations with setups | 4.64 |
| cpsat_policy, combinations with setups | 12.1 |
| policy_sample, combinations with setups | 12.1 |

### Decision table: difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Pool | +4.03 (9) | **-2.36** (9) | +1.54 (9) | +1.53 (9) | -1.93 (9) | -1.67 (9) | policy_sample | yes |
| 60 | combinations without setups | +3.23 (9) | -2.72 (9) | +1.13 (9) | -1.29 (9) | -2.54 (9) | **-3.08** (9) | cpsat_policy | yes |
| 60 | combinations with setups | +4.83 (9) | **-2.00** (9) | +1.95 (9) | +4.34 (9) | -1.32 (9) | -0.26 (9) | policy_sample | yes |
| 300 | Pool | +4.03 (9) | -2.63 (9) | -3.58 (9) | -2.21 (9) | **-4.55** (9) | -3.80 (9) | memetic_policy | yes |
| 300 | combinations without setups | +3.23 (9) | -3.00 (9) | -4.14 (9) | -6.33 (9) | -5.16 (9) | **-6.55** (9) | cpsat_policy | yes |
| 300 | combinations with setups | +4.83 (9) | -2.26 (9) | -3.02 (9) | +1.91 (9) | **-3.94** (9) | -1.06 (9) | memetic_policy | yes |
| 600 | Pool | +4.03 (9) | -2.63 (9) | -4.69 (9) | -3.33 (9) | **-5.20** (9) | -4.44 (9) | memetic_policy | yes |
| 600 | combinations without setups | +3.23 (9) | -3.00 (9) | -5.43 (9) | **-7.63** (9) | -5.96 (9) | -7.31 (9) | cpsat_rule | yes |
| 600 | combinations with setups | +4.83 (9) | -2.26 (9) | -3.95 (9) | +0.97 (9) | **-4.45** (9) | -1.58 (9) | memetic_policy | yes |

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

## 80x25n30 (instances 3-11; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.49 [-4.67, -0.06] ahead | +4.76 [+2.79, +7.09] behind | +6.24 [+4.21, +8.63] behind |
| L | +0.67 [-3.03, +5.40] neither | +7.19 [+3.60, +11.28] behind | +9.38 [+5.79, +13.48] behind |
| S | -2.34 [-4.08, -0.68] ahead | +1.54 [+0.09, +2.90] behind | +2.27 [+0.81, +3.63] behind |
| W | -4.52 [-5.81, -3.23] ahead | -3.94 [-5.38, -2.54] ahead | -2.72 [-3.80, -1.52] ahead |
| LS | +0.46 [-1.92, +3.53] neither | +4.07 [+1.72, +6.91] behind | +5.27 [+3.09, +7.89] behind |
| LW | -1.78 [-3.42, +0.63] neither | -0.73 [-2.45, +1.65] neither | +0.93 [-0.90, +3.15] neither |
| SW | -4.27 [-6.81, -1.94] ahead | -2.28 [-5.16, +0.41] neither | -1.34 [-3.91, +1.20] neither |
| LSW | -3.05 [-5.09, -0.48] ahead | +0.19 [-1.66, +2.68] neither | +1.15 [-0.59, +3.46] neither |
| Pool | -2.16 [-3.41, -0.90] ahead | +1.35 [+0.48, +2.23] behind | +2.65 [+1.91, +3.40] behind |
| combinations without setups | -2.03 [-3.74, +0.04] neither | +1.82 [+0.74, +3.21] behind | +3.46 [+2.43, +4.71] behind |
| combinations with setups | -2.30 [-3.66, -0.95] ahead | +0.88 [-0.20, +1.96] neither | +1.84 [+0.89, +2.79] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.52 [-3.70, +0.98] neither | +5.20 [+3.35, +7.39] behind | +6.96 [+5.17, +9.10] behind |
| L | +1.57 [-2.23, +6.24] neither | +7.89 [+4.13, +12.34] behind | +10.97 [+7.58, +15.03] behind |
| S | -5.57 [-7.59, -3.71] ahead | -4.80 [-6.75, -3.03] ahead | -4.14 [-6.08, -2.41] ahead |
| W | -2.58 [-3.74, -1.37] ahead | +2.00 [+1.17, +2.90] behind | +3.10 [+2.22, +3.98] behind |
| LS | -2.39 [-4.73, +0.66] neither | -1.78 [-4.18, +1.35] neither | -1.03 [-3.50, +2.11] neither |
| LW | -0.78 [-2.47, +1.66] neither | +4.53 [+2.88, +6.80] behind | +6.63 [+5.04, +8.68] behind |
| SW | -4.45 [-6.95, -2.16] ahead | -3.66 [-6.08, -1.39] ahead | -2.85 [-5.21, -0.78] ahead |
| LSW | -3.18 [-5.17, -0.65] ahead | -2.87 [-4.95, -0.21] ahead | -2.52 [-4.58, +0.05] neither |
| Pool | -2.36 [-3.71, -1.02] ahead | +0.81 [-0.32, +1.93] neither | +2.14 [+1.12, +3.17] behind |
| combinations without setups | -0.83 [-2.43, +1.14] neither | +4.91 [+3.66, +6.48] behind | +6.92 [+5.82, +8.30] behind |
| combinations with setups | -3.90 [-5.47, -2.59] ahead | -3.28 [-4.87, -1.86] ahead | -2.63 [-4.23, -1.27] ahead |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.04 [-4.52, -1.62] ahead | +4.18 [+2.73, +5.69] behind | +5.65 [+4.13, +7.27] behind |
| L | -3.20 [-4.98, -0.51] ahead | +3.13 [+0.82, +5.79] behind | +5.25 [+2.88, +7.92] behind |
| S | -1.25 [-3.08, +0.53] neither | +2.67 [+1.26, +4.08] behind | +3.40 [+1.98, +4.88] behind |
| W | -4.58 [-5.88, -3.41] ahead | -3.99 [-5.49, -2.63] ahead | -2.78 [-3.86, -1.59] ahead |
| LS | -0.58 [-2.58, +1.49] neither | +3.00 [+1.10, +5.05] behind | +4.19 [+2.49, +6.06] behind |
| LW | -3.69 [-4.89, -2.31] ahead | -2.66 [-4.01, -1.23] ahead | -1.01 [-2.70, +0.69] neither |
| SW | -3.96 [-6.21, -2.07] ahead | -1.98 [-4.39, -0.01] ahead | -1.04 [-3.08, +0.79] neither |
| LSW | -3.73 [-4.95, -2.43] ahead | -0.50 [-1.69, +0.67] neither | +0.45 [-0.76, +1.57] neither |
| Pool | -3.00 [-3.91, -2.10] ahead | +0.48 [-0.31, +1.21] neither | +1.76 [+0.92, +2.51] behind |
| combinations without setups | -3.63 [-4.69, -2.36] ahead | +0.16 [-0.81, +1.11] neither | +1.78 [+0.67, +2.75] behind |
| combinations with setups | -2.38 [-3.56, -1.26] ahead | +0.80 [-0.17, +1.77] neither | +1.75 [+0.90, +2.65] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.08 [-3.33, -0.76] ahead | +4.62 [+3.58, +5.81] behind | +6.37 [+5.32, +7.53] behind |
| L | -2.32 [-4.21, +0.41] neither | +3.79 [+1.60, +6.48] behind | +6.79 [+4.43, +9.32] behind |
| S | -4.53 [-6.46, -2.70] ahead | -3.75 [-5.61, -2.07] ahead | -3.08 [-4.97, -1.36] ahead |
| W | -2.63 [-3.84, -1.50] ahead | +1.94 [+1.21, +2.73] behind | +3.04 [+2.24, +3.92] behind |
| LS | -3.41 [-5.28, -1.37] ahead | -2.80 [-4.76, -0.67] ahead | -2.06 [-4.05, +0.06] neither |
| LW | -2.71 [-3.98, -1.27] ahead | +2.51 [+1.17, +3.88] behind | +4.58 [+3.07, +6.03] behind |
| SW | -4.14 [-6.33, -2.35] ahead | -3.35 [-5.45, -1.54] ahead | -2.54 [-4.61, -0.79] ahead |
| LSW | -3.86 [-5.06, -2.58] ahead | -3.56 [-4.73, -2.27] ahead | -3.21 [-4.39, -1.96] ahead |
| Pool | -3.21 [-4.17, -2.27] ahead | -0.08 [-0.91, +0.75] neither | +1.24 [+0.44, +2.01] behind |
| combinations without setups | -2.44 [-3.41, -1.22] ahead | +3.21 [+2.22, +4.24] behind | +5.19 [+4.08, +6.21] behind |
| combinations with setups | -3.98 [-5.25, -2.88] ahead | -3.37 [-4.66, -2.21] ahead | -2.72 [-4.05, -1.55] ahead |

### (ii) policy-started arm minus policy: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.83 [-4.20, -0.09] ahead | -5.02 [-6.89, -3.32] ahead | -6.12 [-8.14, -4.28] ahead |
| L | -3.67 [-7.81, -0.57] ahead | -6.85 [-10.95, -3.65] ahead | -8.13 [-11.81, -5.18] ahead |
| S | -0.72 [-1.02, -0.43] ahead | -2.30 [-2.87, -1.70] ahead | -2.87 [-3.43, -2.28] ahead |
| W | -0.38 [-0.76, -0.10] ahead | -1.26 [-1.65, -0.81] ahead | -1.47 [-1.91, -0.97] ahead |
| LS | -3.32 [-5.89, -1.58] ahead | -5.10 [-7.74, -3.34] ahead | -5.66 [-8.25, -3.88] ahead |
| LW | -1.36 [-3.62, -0.14] ahead | -2.94 [-4.95, -1.65] ahead | -3.51 [-5.51, -2.26] ahead |
| SW | -0.92 [-2.05, -0.18] ahead | -1.81 [-3.50, -0.74] ahead | -2.16 [-3.80, -1.08] ahead |
| LSW | -1.72 [-4.34, -0.17] ahead | -2.74 [-5.47, -0.92] ahead | -3.18 [-5.85, -1.40] ahead |
| Pool | -1.74 [-2.87, -0.85] ahead | -3.50 [-4.49, -2.71] ahead | -4.14 [-5.06, -3.40] ahead |
| combinations without setups | -1.81 [-3.81, -0.42] ahead | -4.02 [-5.71, -2.78] ahead | -4.81 [-6.40, -3.63] ahead |
| combinations with setups | -1.67 [-2.87, -0.88] ahead | -2.99 [-4.17, -2.17] ahead | -3.47 [-4.62, -2.63] ahead |

### (ii) policy-started arm minus policy: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -0.69 [-1.37, -0.21] ahead | -4.75 [-6.90, -2.89] ahead | -5.94 [-8.05, -4.09] ahead |
| L | -1.66 [-2.96, -0.60] ahead | -7.67 [-11.15, -5.10] ahead | -9.17 [-12.73, -6.46] ahead |
| S | -0.05 [-0.16, +0.00] neither | -0.31 [-0.47, -0.17] ahead | -0.44 [-0.61, -0.27] ahead |
| W | -0.45 [-0.69, -0.21] ahead | -3.52 [-4.00, -2.99] ahead | -4.49 [-5.15, -3.78] ahead |
| LS | -0.07 [-0.12, -0.03] ahead | -0.73 [-1.41, -0.26] ahead | -1.31 [-2.53, -0.54] ahead |
| LW | -1.18 [-1.99, -0.51] ahead | -5.60 [-7.63, -4.08] ahead | -6.48 [-8.45, -5.04] ahead |
| SW | -0.08 [-0.15, -0.02] ahead | -0.87 [-1.84, -0.25] ahead | -1.21 [-2.33, -0.50] ahead |
| LSW | -0.03 [-0.08, +0.00] neither | -0.61 [-1.31, -0.17] ahead | -1.01 [-2.02, -0.37] ahead |
| Pool | -0.53 [-0.77, -0.34] ahead | -3.01 [-3.80, -2.40] ahead | -3.76 [-4.61, -3.14] ahead |
| combinations without setups | -1.00 [-1.48, -0.64] ahead | -5.38 [-6.79, -4.29] ahead | -6.52 [-7.92, -5.46] ahead |
| combinations with setups | -0.06 [-0.10, -0.02] ahead | -0.63 [-0.92, -0.37] ahead | -0.99 [-1.43, -0.60] ahead |

### (iii) policy-started minus rule-started arm: memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -4.37 [-5.64, -3.02] ahead | -0.58 [-1.93, +0.54] neither | -0.36 [-1.59, +0.54] neither |
| L | -3.40 [-4.18, -2.68] ahead | -0.46 [-2.32, +1.53] neither | +0.21 [-1.39, +1.92] neither |
| S | -3.04 [-4.58, -1.62] ahead | -0.80 [-1.85, +0.25] neither | -0.67 [-1.75, +0.32] neither |
| W | -4.88 [-6.12, -3.58] ahead | -5.15 [-6.55, -3.91] ahead | -4.16 [-5.01, -3.16] ahead |
| LS | -3.01 [-3.86, -2.10] ahead | -1.35 [-2.66, +0.34] neither | -0.79 [-2.18, +0.93] neither |
| LW | -3.22 [-3.87, -2.56] ahead | -3.73 [-4.54, -3.13] ahead | -2.68 [-3.76, -1.66] ahead |
| SW | -5.18 [-7.43, -3.26] ahead | -4.11 [-6.51, -2.06] ahead | -3.53 [-5.52, -1.62] ahead |
| LSW | -4.83 [-5.93, -3.56] ahead | -2.66 [-3.76, -1.59] ahead | -2.17 [-3.26, -1.09] ahead |
| Pool | -3.99 [-4.78, -3.28] ahead | -2.36 [-3.05, -1.70] ahead | -1.77 [-2.43, -1.10] ahead |
| combinations without setups | -3.97 [-4.72, -3.14] ahead | -2.48 [-3.21, -1.79] ahead | -1.75 [-2.47, -1.09] ahead |
| combinations with setups | -4.02 [-5.01, -3.16] ahead | -2.23 [-3.15, -1.36] ahead | -1.79 [-2.67, -0.92] ahead |

### (iii) policy-started minus rule-started arm: CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.23 [-3.97, -0.30] ahead | +0.12 [-0.81, +1.04] neither | +0.52 [-0.31, +1.14] neither |
| L | -0.23 [-3.01, +3.03] neither | -0.65 [-2.06, +0.76] neither | +0.54 [-0.54, +1.71] neither |
| S | -5.62 [-7.60, -3.78] ahead | -5.10 [-6.92, -3.46] ahead | -4.57 [-6.38, -2.94] ahead |
| W | -3.02 [-4.09, -1.90] ahead | -1.60 [-2.08, -1.05] ahead | -1.54 [-1.98, -1.08] ahead |
| LS | -2.47 [-4.77, +0.56] neither | -2.52 [-4.61, +0.29] neither | -2.35 [-4.47, +0.51] neither |
| LW | -1.98 [-3.42, -0.21] ahead | -1.40 [-2.02, -0.73] ahead | -0.35 [-1.03, +0.36] neither |
| SW | -4.52 [-7.03, -2.22] ahead | -4.53 [-6.64, -2.71] ahead | -4.05 [-6.09, -2.38] ahead |
| LSW | -3.20 [-5.20, -0.68] ahead | -3.49 [-5.23, -1.48] ahead | -3.54 [-5.19, -1.80] ahead |
| Pool | -2.91 [-4.15, -1.81] ahead | -2.40 [-3.21, -1.72] ahead | -1.92 [-2.69, -1.26] ahead |
| combinations without setups | -1.87 [-3.14, -0.42] ahead | -0.88 [-1.24, -0.49] ahead | -0.21 [-0.62, +0.16] neither |
| combinations with setups | -3.95 [-5.51, -2.64] ahead | -3.91 [-5.41, -2.64] ahead | -3.63 [-5.11, -2.43] ahead |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.24 | [+4.21, +8.63] | 1 | 0.0004 | behind |
| L | +9.38 | [+5.79, +13.48] | 1 | 0.0004 | behind |
| S | +2.27 | [+0.81, +3.63] | 1 | 0.0042 | behind |
| W | -2.72 | [-3.80, -1.52] | 0.0004 | 1 | ahead |
| LS | +5.27 | [+3.09, +7.89] | 1 | 0.0004 | behind |
| LW | +0.93 | [-0.90, +3.15] | 1 | 0.56 | not distinguishable |
| SW | -1.34 | [-3.91, +1.20] | 1 | 1 | not distinguishable |
| LSW | +1.15 | [-0.59, +3.46] | 1 | 0.52 | not distinguishable |

Row Sel, combinations of 8: ahead 1, behind 4, not distinguishable 3

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +5.65 | [+4.13, +7.27] | 1 / 1 | 0.0005 / 0.0004 | behind |
| L | +5.25 | [+2.88, +7.92] | 1 / 1 | 0.026 / 0.0004 | behind |
| S | +3.40 | [+1.98, +4.88] | 1 / 1 | 0.0039 / 0.0004 | behind |
| W | -2.78 | [-3.86, -1.59] | 0.0057 / 0.0004 | 1 / 1 | ahead |
| LS | +4.19 | [+2.49, +6.06] | 1 / 1 | 0.0023 / 0.0004 | behind |
| LW | -1.01 | [-2.70, +0.69] | 0.52 / 0.87 | 1 / 1 | not distinguishable |
| SW | -1.04 | [-3.08, +0.79] | 0.034 / 0.87 | 1 / 1 | not distinguishable |
| LSW | +0.45 | [-0.76, +1.57] | 1 / 1 | 0.77 / 0.84 | not distinguishable |

Row All, combinations of 8: ahead 1, behind 4, not distinguishable 3

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.96 | [+5.17, +9.10] | 1 | 0.0004 | behind |
| L | +10.97 | [+7.58, +15.03] | 1 | 0.0004 | behind |
| S | -4.14 | [-6.08, -2.41] | 0.0004 | 1 | ahead |
| W | +3.10 | [+2.22, +3.98] | 1 | 0.0004 | behind |
| LS | -1.03 | [-3.50, +2.11] | 1 | 1 | not distinguishable |
| LW | +6.63 | [+5.04, +8.68] | 1 | 0.0004 | behind |
| SW | -2.85 | [-5.21, -0.78] | 0.016 | 1 | ahead |
| LSW | -2.52 | [-4.58, +0.05] | 0.16 | 1 | not distinguishable |

Row Sel, combinations of 8: ahead 2, behind 4, not distinguishable 2

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +6.37 | [+5.32, +7.53] | 1 / 1 | 0.00031 / 0.0004 | behind |
| L | +6.79 | [+4.43, +9.32] | 1 / 1 | 0.011 / 0.0004 | behind |
| S | -3.08 | [-4.97, -1.36] | 0.0052 / 0.0012 | 1 / 1 | ahead |
| W | +3.04 | [+2.24, +3.92] | 1 / 1 | 0.0044 / 0.0004 | behind |
| LS | -2.06 | [-4.05, +0.06] | 0.019 / 0.14 | 1 / 1 | not distinguishable |
| LW | +4.58 | [+3.07, +6.03] | 1 / 1 | 0.006 / 0.0004 | behind |
| SW | -2.54 | [-4.61, -0.79] | 0.0013 / 0.0075 | 1 / 1 | ahead |
| LSW | -3.21 | [-4.39, -1.96] | 0.0059 / 0.0021 | 1 / 1 | ahead |

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

### The policy, sampled schedules: records of 72 cells, budget 120 s (values after it are held)

### (vii) the policy, sampled schedules minus the greedy schedule

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.41 [-4.37, -0.79] ahead | -2.78 [-4.77, -1.12] ahead (held) | -2.78 [-4.77, -1.12] ahead (held) |
| L | -4.51 [-8.94, -1.14] ahead | -4.67 [-8.99, -1.29] ahead (held) | -4.67 [-8.99, -1.29] ahead (held) |
| S | -1.24 [-2.03, -0.50] ahead | -1.73 [-2.60, -0.90] ahead (held) | -1.73 [-2.60, -0.90] ahead (held) |
| W | -1.14 [-2.02, -0.45] ahead | -1.67 [-2.45, -0.98] ahead (held) | -1.67 [-2.45, -0.98] ahead (held) |
| LS | -3.17 [-5.76, -1.24] ahead | -3.47 [-6.03, -1.61] ahead (held) | -3.47 [-6.03, -1.61] ahead (held) |
| LW | -2.14 [-4.30, -0.68] ahead | -2.67 [-4.90, -1.11] ahead (held) | -2.67 [-4.90, -1.11] ahead (held) |
| SW | -2.12 [-3.89, -0.91] ahead | -2.33 [-4.14, -1.12] ahead (held) | -2.33 [-4.14, -1.12] ahead (held) |
| LSW | -2.53 [-4.91, -0.93] ahead | -2.95 [-5.73, -1.05] ahead (held) | -2.95 [-5.73, -1.05] ahead (held) |
| Pool | -2.41 [-3.40, -1.58] ahead | -2.78 [-3.73, -1.95] ahead (held) | -2.78 [-3.73, -1.95] ahead (held) |
| combinations without setups | -2.55 [-4.30, -1.21] ahead | -2.95 [-4.65, -1.60] ahead (held) | -2.95 [-4.65, -1.60] ahead (held) |
| combinations with setups | -2.26 [-3.37, -1.35] ahead | -2.62 [-3.75, -1.72] ahead (held) | -2.62 [-3.75, -1.72] ahead (held) |

### (viii) the policy, sampled schedules minus the rule-started memetic algorithm

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -4.92 [-6.31, -3.58] ahead | +1.77 [+0.36, +3.02] behind (held) | +3.21 [+1.73, +4.52] behind (held) |
| L | -4.26 [-4.99, -3.63] ahead | +1.84 [+0.45, +3.31] behind (held) | +3.93 [+2.50, +5.38] behind (held) |
| S | -3.57 [-4.96, -2.31] ahead | -0.23 [-1.14, +0.59] neither (held) | +0.48 [-0.42, +1.30] neither (held) |
| W | -5.61 [-6.90, -4.31] ahead | -5.55 [-6.80, -4.32] ahead (held) | -4.35 [-5.34, -3.15] ahead (held) |
| LS | -2.85 [-4.15, -1.56] ahead | +0.34 [-0.85, +1.77] neither (held) | +1.50 [+0.37, +2.80] behind (held) |
| LW | -3.98 [-4.64, -3.32] ahead | -3.47 [-4.08, -2.86] ahead (held) | -1.84 [-2.95, -0.61] ahead (held) |
| SW | -6.35 [-8.32, -4.67] ahead | -4.63 [-6.90, -2.76] ahead (held) | -3.71 [-5.60, -1.99] ahead (held) |
| LSW | -5.61 [-6.36, -4.67] ahead | -2.89 [-3.68, -2.13] ahead (held) | -1.95 [-2.83, -1.28] ahead (held) |
| Pool | -4.64 [-5.30, -4.00] ahead | -1.60 [-2.05, -1.11] ahead (held) | -0.34 [-0.87, +0.28] neither (held) |
| combinations without setups | -4.69 [-5.40, -3.92] ahead | -1.35 [-2.00, -0.75] ahead (held) | +0.24 [-0.66, +1.13] neither (held) |
| combinations with setups | -4.60 [-5.45, -3.90] ahead | -1.85 [-2.54, -1.22] ahead (held) | -0.92 [-1.51, -0.34] ahead (held) |

### (viii) the policy, sampled schedules minus the rule-started CP-SAT

| comb. | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.98 [-5.09, -2.86] ahead | +2.20 [+1.32, +2.98] behind (held) | +3.91 [+3.01, +4.70] behind (held) |
| L | -3.40 [-4.28, -2.51] ahead | +2.49 [+1.23, +3.75] behind (held) | +5.46 [+3.98, +6.81] behind (held) |
| S | -6.77 [-8.24, -5.44] ahead | -6.46 [-8.18, -5.05] ahead (held) | -5.81 [-7.55, -4.42] ahead (held) |
| W | -3.69 [-4.87, -2.46] ahead | +0.29 [-0.24, +0.82] neither (held) | +1.37 [+0.86, +1.97] behind (held) |
| LS | -5.62 [-6.77, -4.54] ahead | -5.33 [-6.44, -4.28] ahead (held) | -4.60 [-5.79, -3.46] ahead (held) |
| LW | -3.00 [-3.76, -2.23] ahead | +1.65 [+0.80, +2.47] behind (held) | +3.71 [+2.62, +4.78] behind (held) |
| SW | -6.53 [-8.44, -4.97] ahead | -5.96 [-7.86, -4.36] ahead (held) | -5.17 [-7.08, -3.63] ahead (held) |
| LSW | -5.73 [-6.51, -4.74] ahead | -5.86 [-6.73, -4.84] ahead (held) | -5.52 [-6.44, -4.49] ahead (held) |
| Pool | -4.84 [-5.60, -4.13] ahead | -2.12 [-2.67, -1.59] ahead (held) | -0.83 [-1.29, -0.37] ahead (held) |
| combinations without setups | -3.52 [-4.13, -2.77] ahead | +1.66 [+0.89, +2.37] behind (held) | +3.61 [+2.72, +4.41] behind (held) |
| combinations with setups | -6.16 [-7.22, -5.32] ahead | -5.90 [-7.00, -5.01] ahead (held) | -5.27 [-6.42, -4.32] ahead (held) |

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started memetic algorithm (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +3.21 | [+1.73, +4.52] | 1 | 0.0004 | behind |
| L | +3.93 | [+2.50, +5.38] | 1 | 0.0004 | behind |
| S | +0.48 | [-0.42, +1.30] | 1 | 0.71 | not distinguishable |
| W | -4.35 | [-5.34, -3.15] | 0.0004 | 1 | ahead |
| LS | +1.50 | [+0.37, +2.80] | 1 | 0.02 | behind |
| LW | -1.84 | [-2.95, -0.61] | 0.0087 | 1 | ahead |
| SW | -3.71 | [-5.60, -1.99] | 0.0004 | 1 | ahead |
| LSW | -1.95 | [-2.83, -1.28] | 0.0004 | 1 | ahead |

Row Sel, combinations of 8: ahead 4, behind 3, not distinguishable 1

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started CP-SAT (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +3.91 | [+3.01, +4.70] | 1 | 0.0004 | behind |
| L | +5.46 | [+3.98, +6.81] | 1 | 0.0004 | behind |
| S | -5.81 | [-7.55, -4.42] | 0.0004 | 1 | ahead |
| W | +1.37 | [+0.86, +1.97] | 1 | 0.0004 | behind |
| LS | -4.60 | [-5.79, -3.46] | 0.0004 | 1 | ahead |
| LW | +3.71 | [+2.62, +4.78] | 1 | 0.0004 | behind |
| SW | -5.17 | [-7.08, -3.63] | 0.0004 | 1 | ahead |
| LSW | -5.52 | [-6.44, -4.49] | 0.0004 | 1 | ahead |

Row Sel, combinations of 8: ahead 4, behind 4, not distinguishable 0

### (v) rule-started memetic algorithm reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 109 | 118 | 0.0 | 8 | 527.11 | 480.33 | 510.33 |
| L | 8 | 110 | 0.0 | 8 | 616.11 | 563.56 | 616.22 |
| S | 160 | 159 | 11.1 | 8 | 699.22 | 645.44 | 659.44 |
| W | not within the budget | not within the budget | 88.9 | 9 | 560.33 | 549.33 | 534.33 |
| LS | 15 | 125 | 0.0 | 8 | 780.78 | 723.89 | 762.11 |
| LW | 431 | 466 | 33.3 | 9 | 640.00 | 622.56 | 628.22 |
| SW | not within the budget | not within the budget | 55.6 | 9 | 731.22 | 708.00 | 698.22 |
| LSW | 272 | 493 | 44.4 | 9 | 810.22 | 774.78 | 783.78 |
| Pool | 167 | 212.5 | 29.2 | 9 | 668.99 | 633.49 | 649.08 |
| combinations without setups | 143 | 242.5 | 30.6 | 9 | 585.56 | 553.94 | 572.28 |
| combinations with setups | 187 | 200.5 | 27.8 | 9 | 752.42 | 713.03 | 725.89 |

### (v) rule-started memetic algorithm reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 168 (held) | 138 (held) | 0.0 (held) | 8 | 527.11 | 480.33 | 495.67 |
| L | 192 (held) | 152 (held) | 0.0 (held) | 8 | 616.11 | 563.56 | 585.56 |
| S | 351 (held) | 233 (held) | 22.2 (held) | 8 | 699.22 | 645.44 | 648.11 |
| W | not within the budget (held) | not within the budget (held) | 100.0 (held) | 9 | 560.33 | 549.33 | 525.33 |
| LS | 246 (held) | 140 (held) | 11.1 (held) | 8 | 780.78 | 723.89 | 734.56 |
| LW | not within the budget (held) | not within the budget (held) | 66.7 (held) | 9 | 640.00 | 622.56 | 611.00 |
| SW | not within the budget (held) | not within the budget (held) | 77.8 (held) | 9 | 731.22 | 708.00 | 681.00 |
| LSW | not within the budget (held) | not within the budget (held) | 77.8 (held) | 9 | 810.22 | 774.78 | 759.67 |
| Pool | not within the budget (held) | 438.5 (held) | 44.4 (held) | 9 | 668.99 | 633.49 | 630.11 |
| combinations without setups | 574 (held) | 245 (held) | 41.7 (held) | 9 | 585.56 | 553.94 | 554.39 |
| combinations with setups | not within the budget (held) | 542.5 (held) | 47.2 (held) | 9 | 752.42 | 713.03 | 705.83 |

### (v) rule-started CP-SAT reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 95 | 103 | 0.0 | 8 | 527.44 | 477.00 | 510.33 |
| L | 12 | 64 | 0.0 | 8 | 616.78 | 555.33 | 616.22 |
| S | not within the budget | not within the budget | 100.0 | 8 | 700.33 | 689.00 | 659.44 |
| W | 145 | 162 | 0.0 | 9 | 560.44 | 518.22 | 534.33 |
| LS | not within the budget | not within the budget | 66.7 | 8 | 782.22 | 770.78 | 762.11 |
| LW | 91 | 111 | 0.0 | 9 | 640.00 | 589.11 | 628.22 |
| SW | not within the budget | not within the budget | 77.8 | 9 | 731.22 | 719.11 | 698.22 |
| LSW | not within the budget | not within the budget | 77.8 | 9 | 810.44 | 804.11 | 783.78 |
| Pool | 275 | 215.5 | 40.3 | 9 | 671.07 | 640.33 | 649.08 |
| combinations without setups | 83 | 114 | 0.0 | 9 | 586.08 | 534.92 | 572.28 |
| combinations with setups | not within the budget | not within the budget | 80.6 | 9 | 756.06 | 745.75 | 725.89 |

### (v) rule-started CP-SAT reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 169 (held) | 146 (held) | 0.0 (held) | 8 | 527.44 | 477.00 | 495.67 |
| L | 169 (held) | 98 | 0.0 (held) | 8 | 616.78 | 555.33 | 585.56 |
| S | not within the budget (held) | not within the budget (held) | 100.0 (held) | 8 | 700.33 | 689.00 | 648.11 |
| W | 254 (held) | 263 (held) | 0.0 (held) | 9 | 560.44 | 518.22 | 525.33 |
| LS | not within the budget (held) | not within the budget (held) | 88.9 (held) | 8 | 782.22 | 770.78 | 734.56 |
| LW | 189 (held) | 187 (held) | 0.0 (held) | 9 | 640.00 | 589.11 | 611.00 |
| SW | not within the budget (held) | not within the budget (held) | 88.9 (held) | 9 | 731.22 | 719.11 | 681.00 |
| LSW | not within the budget (held) | not within the budget (held) | 77.8 (held) | 9 | 810.44 | 804.11 | 759.67 |
| Pool | not within the budget (held) | 353.5 (held) | 44.4 (held) | 9 | 671.07 | 640.33 | 630.11 |
| combinations without setups | 189 (held) | 181.5 (held) | 0.0 (held) | 9 | 586.08 | 534.92 | 554.39 |
| combinations with setups | not within the budget (held) | not within the budget (held) | 88.9 (held) | 9 | 756.06 | 745.75 | 705.83 |

### (ix) time of the first schedule (seconds since the request)

| method | time | median | minimum | maximum | cells |
|---|---|---|---|---|---|
| dispatching rules (best of four) | t_all | 7.26 | 5.36 | 8.61 | 72 |
| policy, greedy schedule | t | 24.06 | 19.44 | 28.06 | 72 |
| policy, greedy schedule, from loaded | t_resident | 22.53 | 18.08 | 26.44 | 72 |
| dispatching rules (best of four), combinations without setups | t_all | 7.31 | 5.36 | 8.61 | 36 |
| policy, greedy schedule, combinations without setups | t | 24.06 | 19.44 | 28.06 | 36 |
| policy, greedy schedule, from loaded, combinations without setups | t_resident | 22.53 | 18.08 | 26.44 | 36 |
| dispatching rules (best of four), combinations with setups | t_all | 7.25 | 5.36 | 8.27 | 36 |
| policy, greedy schedule, combinations with setups | t | 24.07 | 19.49 | 27.00 | 36 |
| policy, greedy schedule, from loaded, combinations with setups | t_resident | 22.54 | 18.11 | 25.41 | 36 |

### Curves: first grid time at which every cell of the method has a schedule

| method | from s |
|---|---|
| memetic_rule | 10 |
| memetic_policy | 31.6 |
| cpsat_rule | 10 |
| cpsat_policy | 31.6 |
| policy_sample | 31.6 |
| memetic_rule, combinations without setups | 10 |
| memetic_policy, combinations without setups | 31.6 |
| cpsat_rule, combinations without setups | 10 |
| cpsat_policy, combinations without setups | 31.6 |
| policy_sample, combinations without setups | 31.6 |
| memetic_rule, combinations with setups | 10 |
| memetic_policy, combinations with setups | 31.6 |
| cpsat_rule, combinations with setups | 10 |
| cpsat_policy, combinations with setups | 31.6 |
| policy_sample, combinations with setups | 31.6 |

### Decision table: difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Pool | +3.50 (9) | **-2.41** (9) | +2.40 (9) | +2.62 (9) | -1.74 (9) | -0.53 (9) | policy_sample | yes |
| 60 | combinations without setups | +2.69 (9) | **-2.55** (9) | +2.28 (9) | +1.02 (9) | -1.81 (9) | -1.00 (9) | policy_sample | yes |
| 60 | combinations with setups | +4.31 (9) | **-2.26** (9) | +2.52 (9) | +4.22 (9) | -1.67 (9) | -0.06 (9) | policy_sample | yes |
| 300 | Pool | +3.50 (9) | -2.78 (9) | -1.09 (9) | -0.49 (9) | **-3.50** (9) | -3.01 (9) | memetic_policy | yes |
| 300 | combinations without setups | +2.69 (9) | -2.95 (9) | -1.47 (9) | -4.52 (9) | -4.02 (9) | **-5.38** (9) | cpsat_policy | yes |
| 300 | combinations with setups | +4.31 (9) | -2.62 (9) | -0.70 (9) | +3.54 (9) | **-2.99** (9) | -0.63 (9) | memetic_policy | yes |
| 600 | Pool | +3.50 (9) | -2.78 (9) | -2.33 (9) | -1.73 (9) | **-4.14** (9) | -3.76 (9) | memetic_policy | yes |
| 600 | combinations without setups | +2.69 (9) | -2.95 (9) | -3.02 (9) | -6.31 (9) | -4.81 (9) | **-6.52** (9) | cpsat_policy | yes |
| 600 | combinations with setups | +4.31 (9) | -2.62 (9) | -1.65 (9) | +2.85 (9) | **-3.47** (9) | -0.99 (9) | memetic_policy | yes |

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

## 10x25 (instances 0-29; policy: selected training run s304)

### (i) policy minus rule-started arm, selected run: memetic algorithm

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +5.31 [+4.16, +6.45] behind | +6.63 [+5.41, +7.81] behind | +7.43 [+6.31, +8.53] behind | +7.57 [+6.52, +8.62] behind |
| L | +3.93 [+3.20, +4.71] behind | +4.17 [+3.49, +4.92] behind | +4.33 [+3.63, +5.09] behind | +4.35 [+3.65, +5.10] behind |
| S | +5.74 [+4.92, +6.59] behind | +7.11 [+6.41, +7.89] behind | +7.56 [+6.86, +8.33] behind | +7.66 [+6.94, +8.47] behind |
| W | +3.55 [+2.40, +4.75] behind | +7.32 [+5.84, +8.94] behind | +9.31 [+7.68, +11.02] behind | +9.59 [+7.97, +11.29] behind |
| LS | +4.82 [+3.99, +5.72] behind | +5.83 [+4.99, +6.72] behind | +6.21 [+5.35, +7.12] behind | +6.32 [+5.45, +7.24] behind |
| LW | +3.78 [+3.01, +4.58] behind | +4.82 [+4.06, +5.63] behind | +5.54 [+4.76, +6.36] behind | +5.71 [+4.89, +6.58] behind |
| SW | +4.52 [+3.45, +5.54] behind | +7.86 [+6.61, +9.11] behind | +9.46 [+8.08, +10.86] behind | +9.70 [+8.33, +11.05] behind |
| LSW | +4.90 [+3.72, +6.25] behind | +6.93 [+5.87, +8.13] behind | +7.90 [+6.73, +9.21] behind | +8.21 [+7.02, +9.56] behind |
| Pool | +4.57 [+4.10, +5.06] behind | +6.33 [+5.88, +6.82] behind | +7.22 [+6.73, +7.74] behind | +7.39 [+6.91, +7.90] behind |
| combinations without setups | +4.14 [+3.44, +4.83] behind | +5.74 [+5.03, +6.46] behind | +6.65 [+5.95, +7.38] behind | +6.81 [+6.09, +7.53] behind |
| combinations with setups | +4.99 [+4.44, +5.56] behind | +6.93 [+6.35, +7.52] behind | +7.78 [+7.17, +8.41] behind | +7.97 [+7.35, +8.61] behind |

### (i) policy minus rule-started arm, selected run: CP-SAT

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +5.94 [+4.11, +7.72] behind | +7.60 [+5.66, +9.60] behind | +7.70 [+5.68, +9.84] behind | +7.89 [+5.74, +10.19] behind |
| L | +3.38 [+2.39, +4.51] behind | +3.54 [+2.46, +4.83] behind | +3.60 [+2.50, +4.88] behind | +3.60 [+2.50, +4.88] behind |
| S | +1.31 [-0.15, +2.60] neither | +4.81 [+3.67, +6.05] behind | +5.73 [+4.70, +6.89] behind | +5.79 [+4.73, +6.96] behind |
| W | +5.24 [+3.56, +6.88] behind | +7.54 [+5.58, +9.36] behind | +8.49 [+6.97, +9.99] behind | +8.49 [+6.97, +9.99] behind |
| LS | +1.54 [+0.11, +3.00] behind | +4.14 [+2.62, +5.49] behind | +5.03 [+3.70, +6.26] behind | +5.29 [+3.93, +6.50] behind |
| LW | +4.57 [+3.38, +5.94] behind | +5.81 [+4.28, +7.55] behind | +5.92 [+4.37, +7.64] behind | +5.92 [+4.37, +7.64] behind |
| SW | +2.68 [+0.77, +4.41] behind | +4.75 [+2.74, +6.41] behind | +6.57 [+4.78, +8.32] behind | +7.51 [+5.34, +9.58] behind |
| LSW | +3.17 [+1.20, +5.33] behind | +5.05 [+3.28, +6.90] behind | +6.36 [+4.37, +8.85] behind | +6.72 [+4.69, +9.22] behind |
| Pool | +3.48 [+2.85, +4.16] behind | +5.40 [+4.60, +6.19] behind | +6.18 [+5.47, +6.93] behind | +6.40 [+5.69, +7.13] behind |
| combinations without setups | +4.78 [+3.71, +5.89] behind | +6.12 [+4.91, +7.35] behind | +6.43 [+5.30, +7.57] behind | +6.48 [+5.35, +7.60] behind |
| combinations with setups | +2.18 [+1.36, +2.99] behind | +4.69 [+3.76, +5.59] behind | +5.92 [+5.06, +6.85] behind | +6.33 [+5.36, +7.31] behind |

### (i) policy minus rule-started arm, every run: memetic algorithm

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +5.31 [+4.35, +6.25] behind | +6.63 [+5.61, +7.62] behind | +7.44 [+6.48, +8.37] behind | +7.58 [+6.67, +8.47] behind |
| L | +3.49 [+2.80, +4.21] behind | +3.73 [+3.07, +4.44] behind | +3.89 [+3.19, +4.63] behind | +3.91 [+3.21, +4.64] behind |
| S | +6.34 [+5.45, +7.27] behind | +7.72 [+6.89, +8.63] behind | +8.17 [+7.35, +9.06] behind | +8.27 [+7.45, +9.18] behind |
| W | +3.98 [+2.54, +5.52] behind | +7.74 [+6.20, +9.41] behind | +9.73 [+8.17, +11.37] behind | +10.01 [+8.46, +11.63] behind |
| LS | +4.63 [+3.82, +5.49] behind | +5.65 [+4.82, +6.49] behind | +6.02 [+5.19, +6.87] behind | +6.13 [+5.30, +6.99] behind |
| LW | +3.20 [+2.43, +4.02] behind | +4.24 [+3.47, +5.04] behind | +4.95 [+4.16, +5.77] behind | +5.13 [+4.31, +5.98] behind |
| SW | +4.72 [+3.85, +5.62] behind | +8.05 [+7.05, +9.04] behind | +9.65 [+8.56, +10.72] behind | +9.89 [+8.83, +10.93] behind |
| LSW | +4.53 [+3.46, +5.73] behind | +6.55 [+5.66, +7.55] behind | +7.53 [+6.48, +8.70] behind | +7.84 [+6.77, +9.03] behind |
| Pool | +4.53 [+4.05, +5.01] behind | +6.29 [+5.84, +6.76] behind | +7.17 [+6.69, +7.67] behind | +7.35 [+6.86, +7.85] behind |
| combinations without setups | +4.00 [+3.35, +4.65] behind | +5.59 [+4.91, +6.26] behind | +6.50 [+5.85, +7.17] behind | +6.66 [+5.98, +7.33] behind |
| combinations with setups | +5.05 [+4.53, +5.58] behind | +6.99 [+6.46, +7.52] behind | +7.84 [+7.27, +8.41] behind | +8.03 [+7.46, +8.61] behind |

### (i) policy minus rule-started arm, every run: CP-SAT

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +5.84 [+4.35, +7.51] behind | +7.49 [+5.90, +9.30] behind | +7.59 [+5.91, +9.58] behind | +7.78 [+5.97, +9.85] behind |
| L | +3.46 [+2.44, +4.63] behind | +3.62 [+2.50, +4.97] behind | +3.68 [+2.55, +5.02] behind | +3.68 [+2.55, +5.02] behind |
| S | +2.12 [+0.99, +3.26] behind | +5.66 [+4.49, +6.93] behind | +6.59 [+5.53, +7.76] behind | +6.65 [+5.56, +7.87] behind |
| W | +6.73 [+4.67, +9.04] behind | +9.06 [+6.70, +11.63] behind | +10.03 [+8.24, +12.17] behind | +10.03 [+8.24, +12.17] behind |
| LS | +1.33 [+0.33, +2.37] behind | +3.93 [+2.74, +5.09] behind | +4.82 [+3.70, +5.91] behind | +5.08 [+3.93, +6.20] behind |
| LW | +4.44 [+3.20, +5.91] behind | +5.67 [+4.23, +7.43] behind | +5.78 [+4.34, +7.50] behind | +5.78 [+4.34, +7.50] behind |
| SW | +2.35 [+0.68, +4.21] behind | +4.40 [+2.83, +6.12] behind | +6.22 [+4.78, +7.95] behind | +7.15 [+5.38, +9.09] behind |
| LSW | +1.74 [+0.43, +3.31] behind | +3.61 [+2.31, +4.97] behind | +4.89 [+3.38, +6.76] behind | +5.25 [+3.75, +7.12] behind |
| Pool | +3.50 [+2.90, +4.14] behind | +5.43 [+4.67, +6.15] behind | +6.20 [+5.54, +6.88] behind | +6.42 [+5.73, +7.13] behind |
| combinations without setups | +5.12 [+4.01, +6.28] behind | +6.46 [+5.24, +7.73] behind | +6.77 [+5.65, +7.93] behind | +6.82 [+5.68, +7.99] behind |
| combinations with setups | +1.89 [+1.26, +2.53] behind | +4.40 [+3.66, +5.11] behind | +5.63 [+4.94, +6.35] behind | +6.03 [+5.26, +6.83] behind |

### (i) verdict at the stated budget of 600 s: policy against the rule-started memetic algorithm

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.57 | [+6.52, +8.62] | 1 | 0.0004 | behind |
| L | +4.35 | [+3.65, +5.10] | 1 | 0.0004 | behind |
| S | +7.66 | [+6.94, +8.47] | 1 | 0.0004 | behind |
| W | +9.59 | [+7.97, +11.29] | 1 | 0.0004 | behind |
| LS | +6.32 | [+5.45, +7.24] | 1 | 0.0004 | behind |
| LW | +5.71 | [+4.89, +6.58] | 1 | 0.0004 | behind |
| SW | +9.70 | [+8.33, +11.05] | 1 | 0.0004 | behind |
| LSW | +8.21 | [+7.02, +9.56] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 8, not distinguishable 0

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.58 | [+6.67, +8.47] | 1 / 1 | 3.2e-07 / 0.0004 | behind |
| L | +3.91 | [+3.21, +4.64] | 1 / 1 | 8.3e-06 / 0.0004 | behind |
| S | +8.27 | [+7.45, +9.18] | 1 / 1 | 2.9e-06 / 0.0004 | behind |
| W | +10.01 | [+8.46, +11.63] | 1 / 1 | 2.9e-06 / 0.0004 | behind |
| LS | +6.13 | [+5.30, +6.99] | 1 / 1 | 8.3e-06 / 0.0004 | behind |
| LW | +5.13 | [+4.31, +5.98] | 1 / 1 | 8.3e-06 / 0.0004 | behind |
| SW | +9.89 | [+8.83, +10.93] | 1 / 1 | 6.8e-08 / 0.0004 | behind |
| LSW | +7.84 | [+6.77, +9.03] | 1 / 1 | 1.3e-06 / 0.0004 | behind |

Row All, combinations of 8: ahead 0, behind 8, not distinguishable 0

### (i) verdict at the stated budget of 600 s: policy against the rule-started CP-SAT

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.89 | [+5.74, +10.19] | 1 | 0.0004 | behind |
| L | +3.60 | [+2.50, +4.88] | 1 | 0.0004 | behind |
| S | +5.79 | [+4.73, +6.96] | 1 | 0.0004 | behind |
| W | +8.49 | [+6.97, +9.99] | 1 | 0.0004 | behind |
| LS | +5.29 | [+3.93, +6.50] | 1 | 0.0004 | behind |
| LW | +5.92 | [+4.37, +7.64] | 1 | 0.0004 | behind |
| SW | +7.51 | [+5.34, +9.58] | 1 | 0.0004 | behind |
| LSW | +6.72 | [+4.69, +9.22] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 8, not distinguishable 0

Row All (crossed hierarchical bootstrap over training runs and base instances, fresh default_rng(11)). ahead: the hierarchical 95% interval lies below zero and both Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t over the per-run means, hierarchical bootstrap) are below alpha; behind: the interval lies above zero and both Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family) are below alpha; not distinguishable otherwise (p89_public_family.superiority_verdict).

| comb. | difference | 95% interval | Holm p shorter (t / bootstrap) | Holm p longer (t / bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +7.78 | [+5.97, +9.85] | 1 / 1 | 7.3e-06 / 0.0004 | behind |
| L | +3.68 | [+2.55, +5.02] | 1 / 1 | 7.3e-06 / 0.0004 | behind |
| S | +6.65 | [+5.56, +7.87] | 1 / 1 | 3.4e-05 / 0.0004 | behind |
| W | +10.03 | [+8.24, +12.17] | 1 / 1 | 3.4e-05 / 0.0004 | behind |
| LS | +5.08 | [+3.93, +6.20] | 1 / 1 | 2.2e-05 / 0.0004 | behind |
| LW | +5.78 | [+4.34, +7.50] | 1 / 1 | 4.8e-06 / 0.0004 | behind |
| SW | +7.15 | [+5.38, +9.09] | 1 / 1 | 7.3e-06 / 0.0004 | behind |
| LSW | +5.25 | [+3.75, +7.12] | 1 / 1 | 0.00013 / 0.0004 | behind |

Row All, combinations of 8: ahead 0, behind 8, not distinguishable 0

### (iv) best rule minus policy (selected)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +1.47 [+0.35, +2.63] behind | 30.0 | 6.7 | 63.3 | 30 |
| L | +0.86 [-0.19, +1.83] neither | 33.3 | 3.3 | 63.3 | 30 |
| S | +1.44 [+0.65, +2.29] behind | 20.0 | 13.3 | 66.7 | 30 |
| W | +2.89 [+1.58, +4.30] behind | 20.0 | 3.3 | 76.7 | 30 |
| LS | +0.77 [+0.07, +1.50] behind | 30.0 | 10.0 | 60.0 | 30 |
| LW | +1.59 [+0.66, +2.51] behind | 26.7 | 3.3 | 70.0 | 30 |
| SW | +0.55 [-0.46, +1.53] neither | 33.3 | 6.7 | 60.0 | 30 |
| LSW | +0.75 [-0.27, +1.66] neither | 36.7 | 3.3 | 60.0 | 30 |
| Pool | +1.29 [+0.79, +1.77] behind | 28.7 | 6.2 | 65.0 | 30 |
| combinations without setups | +1.70 [+0.97, +2.42] behind | 27.5 | 4.2 | 68.3 | 30 |
| combinations with setups | +0.88 [+0.42, +1.36] behind | 30.0 | 8.3 | 61.7 | 30 |

### (iv) best rule minus policy (all_seeds)

| comb. | difference | shorter % | equal % | longer % | n |
|---|---|---|---|---|---|
| N | +1.46 [+0.49, +2.46] behind | 25.3 | 12.0 | 62.7 | 30 |
| L | +1.29 [+0.29, +2.25] behind | 26.0 | 6.0 | 68.0 | 30 |
| S | +0.89 [-0.01, +1.75] neither | 29.3 | 9.3 | 61.3 | 30 |
| W | +2.53 [+1.19, +3.86] behind | 20.0 | 8.0 | 72.0 | 30 |
| LS | +0.95 [+0.29, +1.71] behind | 26.7 | 6.7 | 66.7 | 30 |
| LW | +2.18 [+1.18, +3.22] behind | 20.7 | 4.7 | 74.7 | 30 |
| SW | +0.37 [-0.49, +1.19] neither | 36.7 | 8.0 | 55.3 | 30 |
| LSW | +1.09 [+0.27, +1.92] behind | 31.3 | 4.0 | 64.7 | 30 |
| Pool | +1.34 [+0.85, +1.83] behind | 27.0 | 7.3 | 65.7 | 30 |
| combinations without setups | +1.86 [+1.11, +2.58] behind | 23.0 | 7.7 | 69.3 | 30 |
| combinations with setups | +0.82 [+0.37, +1.29] behind | 31.0 | 7.0 | 62.0 | 30 |

### The policy, sampled schedules: records of 240 cells, budget 60 s (values after it are held)

### (vii) the policy, sampled schedules minus the greedy schedule

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | -2.90 [-3.44, -2.39] ahead | -3.63 [-4.12, -3.16] ahead | -3.63 [-4.12, -3.16] ahead (held) | -3.63 [-4.12, -3.16] ahead (held) |
| L | -2.35 [-2.77, -1.95] ahead | -2.81 [-3.23, -2.42] ahead | -2.81 [-3.23, -2.42] ahead (held) | -2.81 [-3.23, -2.42] ahead (held) |
| S | -2.70 [-3.14, -2.27] ahead | -3.26 [-3.71, -2.82] ahead | -3.26 [-3.71, -2.82] ahead (held) | -3.26 [-3.71, -2.82] ahead (held) |
| W | -4.43 [-5.45, -3.50] ahead | -5.65 [-6.73, -4.66] ahead | -5.65 [-6.73, -4.66] ahead (held) | -5.65 [-6.73, -4.66] ahead (held) |
| LS | -2.40 [-2.92, -1.94] ahead | -2.77 [-3.26, -2.31] ahead | -2.77 [-3.26, -2.31] ahead (held) | -2.77 [-3.26, -2.31] ahead (held) |
| LW | -2.67 [-3.30, -2.09] ahead | -3.45 [-4.16, -2.79] ahead | -3.45 [-4.16, -2.79] ahead (held) | -3.45 [-4.16, -2.79] ahead (held) |
| SW | -4.19 [-5.02, -3.42] ahead | -5.13 [-6.05, -4.25] ahead | -5.13 [-6.05, -4.25] ahead (held) | -5.13 [-6.05, -4.25] ahead (held) |
| LSW | -3.36 [-4.14, -2.67] ahead | -3.97 [-4.81, -3.23] ahead | -3.97 [-4.81, -3.23] ahead (held) | -3.97 [-4.81, -3.23] ahead (held) |
| Pool | -3.13 [-3.38, -2.87] ahead | -3.83 [-4.11, -3.56] ahead | -3.83 [-4.11, -3.56] ahead (held) | -3.83 [-4.11, -3.56] ahead (held) |
| combinations without setups | -3.09 [-3.44, -2.75] ahead | -3.89 [-4.26, -3.52] ahead | -3.89 [-4.26, -3.52] ahead (held) | -3.89 [-4.26, -3.52] ahead (held) |
| combinations with setups | -3.16 [-3.52, -2.82] ahead | -3.78 [-4.18, -3.39] ahead | -3.78 [-4.18, -3.39] ahead (held) | -3.78 [-4.18, -3.39] ahead (held) |

### (viii) the policy, sampled schedules minus the rule-started memetic algorithm

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +2.22 [+1.40, +3.01] behind | +2.72 [+1.82, +3.59] behind | +3.50 [+2.70, +4.29] behind (held) | +3.64 [+2.88, +4.38] behind (held) |
| L | +1.46 [+0.94, +2.02] behind | +1.23 [+0.76, +1.72] behind | +1.38 [+0.93, +1.85] behind (held) | +1.39 [+0.95, +1.87] behind (held) |
| S | +2.87 [+2.13, +3.60] behind | +3.61 [+2.95, +4.31] behind | +4.04 [+3.41, +4.70] behind (held) | +4.14 [+3.49, +4.82] behind (held) |
| W | -1.07 [-2.30, +0.09] neither | +1.19 [-0.08, +2.53] neither | +3.05 [+1.84, +4.37] behind (held) | +3.32 [+2.09, +4.63] behind (held) |
| LS | +2.29 [+1.55, +3.09] behind | +2.88 [+2.22, +3.59] behind | +3.25 [+2.56, +3.98] behind (held) | +3.36 [+2.67, +4.09] behind (held) |
| LW | +0.98 [+0.49, +1.46] behind | +1.17 [+0.75, +1.58] behind | +1.86 [+1.50, +2.20] behind (held) | +2.03 [+1.63, +2.40] behind (held) |
| SW | +0.10 [-0.71, +0.89] neither | +2.26 [+1.44, +3.14] behind | +3.77 [+2.90, +4.67] behind (held) | +4.00 [+3.16, +4.85] behind (held) |
| LSW | +1.32 [+0.48, +2.26] behind | +2.63 [+2.03, +3.26] behind | +3.56 [+2.83, +4.36] behind (held) | +3.86 [+3.11, +4.68] behind (held) |
| Pool | +1.27 [+0.88, +1.68] behind | +2.21 [+1.85, +2.59] behind | +3.05 [+2.70, +3.43] behind (held) | +3.22 [+2.87, +3.58] behind (held) |
| combinations without setups | +0.90 [+0.36, +1.42] behind | +1.58 [+1.00, +2.16] behind | +2.45 [+1.91, +3.00] behind (held) | +2.59 [+2.04, +3.16] behind (held) |
| combinations with setups | +1.64 [+1.15, +2.15] behind | +2.85 [+2.41, +3.30] behind | +3.65 [+3.19, +4.14] behind (held) | +3.84 [+3.37, +4.32] behind (held) |

### (viii) the policy, sampled schedules minus the rule-started CP-SAT

| comb. | 10 s | 60 s | 300 s | 600 s |
|---|---|---|---|---|
| N | +2.61 [+1.31, +3.95] behind | +3.46 [+2.16, +4.81] behind | +3.56 [+2.17, +5.05] behind (held) | +3.74 [+2.22, +5.38] behind (held) |
| L | +0.84 [+0.44, +1.30] behind | +0.65 [+0.24, +1.10] behind | +0.70 [+0.24, +1.21] behind (held) | +0.70 [+0.24, +1.21] behind (held) |
| S | -1.19 [-2.09, -0.25] ahead | +1.42 [+0.53, +2.43] behind | +2.32 [+1.56, +3.25] behind (held) | +2.37 [+1.59, +3.31] behind (held) |
| W | +2.43 [+1.02, +3.94] behind | +2.84 [+1.61, +4.15] behind | +3.77 [+2.61, +5.00] behind (held) | +3.77 [+2.61, +5.00] behind (held) |
| LS | -1.24 [-1.98, -0.56] ahead | +1.01 [+0.10, +1.89] behind | +1.89 [+1.09, +2.83] behind (held) | +2.14 [+1.35, +3.02] behind (held) |
| LW | +1.95 [+1.24, +2.61] behind | +2.42 [+1.59, +3.23] behind | +2.52 [+1.71, +3.30] behind (held) | +2.52 [+1.71, +3.30] behind (held) |
| SW | -1.99 [-3.62, -0.34] ahead | -1.03 [-2.77, +0.67] neither | +0.70 [-0.99, +2.45] neither (held) | +1.59 [-0.51, +3.66] neither (held) |
| LSW | -1.70 [-2.72, -0.54] ahead | -0.67 [-1.77, +0.34] neither | +0.55 [-0.77, +1.87] neither (held) | +0.89 [-0.44, +2.29] neither (held) |
| Pool | +0.22 [-0.11, +0.52] neither | +1.26 [+0.90, +1.58] behind | +2.00 [+1.73, +2.28] behind (held) | +2.22 [+1.89, +2.54] behind (held) |
| combinations without setups | +1.96 [+1.40, +2.56] behind | +2.34 [+1.67, +3.02] behind | +2.64 [+1.94, +3.32] behind (held) | +2.68 [+1.97, +3.39] behind (held) |
| combinations with setups | -1.53 [-2.22, -0.84] ahead | +0.18 [-0.60, +1.00] neither | +1.36 [+0.66, +2.13] behind (held) | +1.75 [+1.00, +2.51] behind (held) |

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started memetic algorithm (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +3.64 | [+2.88, +4.38] | 1 | 0.0004 | behind |
| L | +1.39 | [+0.95, +1.87] | 1 | 0.0004 | behind |
| S | +4.14 | [+3.49, +4.82] | 1 | 0.0004 | behind |
| W | +3.32 | [+2.09, +4.63] | 1 | 0.0004 | behind |
| LS | +3.36 | [+2.67, +4.09] | 1 | 0.0004 | behind |
| LW | +2.03 | [+1.63, +2.40] | 1 | 0.0004 | behind |
| SW | +4.00 | [+3.16, +4.85] | 1 | 0.0004 | behind |
| LSW | +3.86 | [+3.11, +4.68] | 1 | 0.0004 | behind |

Row Sel, combinations of 8: ahead 0, behind 8, not distinguishable 0

### (viii) verdict at the stated budget of 600 s: the policy, sampled schedules against the rule-started CP-SAT (held)

Row Sel (instance-level cluster bootstrap, fresh default_rng(7)). ahead: the instance-level 95% interval lies below zero and the Holm-adjusted one-sided bootstrap p for "the policy is shorter" (draws of that interval) is below alpha; behind: the interval lies above zero and the Holm-adjusted one-sided bootstrap p for "the policy is longer" (its own Holm family) is below alpha; not distinguishable otherwise. No seed-level test exists for one run.

| comb. | difference | 95% interval | Holm p shorter (bootstrap) | Holm p longer (bootstrap) | verdict |
|---|---|---|---|---|---|
| N | +3.74 | [+2.22, +5.38] | 1 | 0.0004 | behind |
| L | +0.70 | [+0.24, +1.21] | 1 | 0.00045 | behind |
| S | +2.37 | [+1.59, +3.31] | 1 | 0.0004 | behind |
| W | +3.77 | [+2.61, +5.00] | 1 | 0.0004 | behind |
| LS | +2.14 | [+1.35, +3.02] | 1 | 0.0004 | behind |
| LW | +2.52 | [+1.71, +3.30] | 1 | 0.0004 | behind |
| SW | +1.59 | [-0.51, +3.66] | 1 | 0.14 | not distinguishable |
| LSW | +0.89 | [-0.44, +2.29] | 1 | 0.14 | not distinguishable |

Row Sel, combinations of 8: ahead 0, behind 6, not distinguishable 2

### (v) rule-started memetic algorithm reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 4 | 3 | 0.0 | 2 | 130.87 | 120.00 | 129.00 |
| L | 3 | 3 | 0.0 | 2 | 213.03 | 202.73 | 211.57 |
| S | 3 | 3 | 0.0 | 2 | 194.27 | 178.07 | 191.67 |
| W | 5 | 4.5 | 0.0 | 2 | 149.97 | 133.20 | 145.83 |
| LS | 3 | 3 | 0.0 | 2 | 270.07 | 252.33 | 268.27 |
| LW | 4 | 3 | 0.0 | 2 | 224.10 | 208.80 | 220.70 |
| SW | 3 | 4.5 | 0.0 | 2 | 219.67 | 199.30 | 218.53 |
| LSW | 3 | 3 | 0.0 | 2 | 289.60 | 265.87 | 287.63 |
| Pool | 3 | 3 | 0.0 | 2 | 211.45 | 195.04 | 209.15 |
| combinations without setups | 4 | 3 | 0.0 | 2 | 179.49 | 166.18 | 176.78 |
| combinations with setups | 3 | 3 | 0.0 | 2 | 243.40 | 223.89 | 241.53 |

### (v) rule-started memetic algorithm reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 5 | 5 | 3.3 (held) | 2 | 130.87 | 120.00 | 124.30 |
| L | 4 | 3 | 0.0 (held) | 2 | 213.03 | 202.73 | 205.57 |
| S | 3 | 3 | 0.0 (held) | 2 | 194.27 | 178.07 | 185.40 |
| W | 25 | 12 | 10.0 (held) | 2 | 149.97 | 133.20 | 137.47 |
| LS | 3 | 3.5 | 0.0 (held) | 2 | 270.07 | 252.33 | 260.80 |
| LW | 5 | 5 | 3.3 (held) | 2 | 224.10 | 208.80 | 213.00 |
| SW | 10 | 7.5 | 0.0 (held) | 2 | 219.67 | 199.30 | 207.27 |
| LSW | 6 | 6 | 3.3 (held) | 2 | 289.60 | 265.87 | 276.10 |
| Pool | 6 | 5 | 2.5 (held) | 2 | 211.45 | 195.04 | 201.24 |
| combinations without setups | 5 | 5 | 4.2 (held) | 2 | 179.49 | 166.18 | 170.08 |
| combinations with setups | 6 | 6 | 0.8 (held) | 2 | 243.40 | 223.89 | 232.39 |

### (v) rule-started CP-SAT reaches the greedy makespan of the selected run

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 2 | 2 | 0.0 | 2 | 127.90 | 120.70 | 130.10 |
| L | 2 | 2 | 0.0 | 2 | 205.60 | 202.90 | 210.20 |
| S | 5 | 4 | 0.0 | 2 | 196.00 | 183.00 | 193.70 |
| W | 3 | 3 | 0.0 | 2 | 146.30 | 133.20 | 144.50 |
| LS | 2 | 3 | 0.0 | 2 | 265.60 | 253.20 | 266.60 |
| LW | 3 | 3.5 | 0.0 | 2 | 219.40 | 206.70 | 218.90 |
| SW | 2 | 2 | 0.0 | 2 | 220.00 | 206.40 | 221.70 |
| LSW | 2 | 2 | 0.0 | 2 | 286.00 | 272.90 | 291.10 |
| Pool | 2 | 2 | 0.0 | 2 | 208.35 | 197.38 | 209.60 |
| combinations without setups | 2 | 2 | 0.0 | 2 | 174.80 | 165.88 | 175.93 |
| combinations with setups | 2 | 2 | 0.0 | 2 | 241.90 | 228.88 | 243.28 |

### (v) rule-started CP-SAT reaches the policy, sampled schedules at the same time

| comb. | mean reached at s | median cell s | never % | all cells from s | arm mean then | arm mean at end | policy mean at end |
|---|---|---|---|---|---|---|---|
| N | 3 | 3.5 | 0.0 (held) | 2 | 127.90 | 120.70 | 125.10 |
| L | 3 | 3 | 0.0 (held) | 2 | 205.60 | 202.90 | 204.30 |
| S | 27 | 24 | 0.0 (held) | 2 | 196.00 | 183.00 | 187.40 |
| W | 4 | 3.5 | 0.0 (held) | 2 | 146.30 | 133.20 | 138.10 |
| LS | 28 | 13 | 0.0 (held) | 2 | 265.60 | 253.20 | 258.60 |
| LW | 6 | 5 | 0.0 (held) | 2 | 219.40 | 206.70 | 211.90 |
| SW | 161 (held) | 65.5 (held) | 30.0 (held) | 2 | 220.00 | 206.40 | 209.60 |
| LSW | 127 (held) | 28 | 30.0 (held) | 2 | 286.00 | 272.90 | 275.20 |
| Pool | 13 | 5 | 7.5 (held) | 2 | 208.35 | 197.38 | 201.28 |
| combinations without setups | 3 | 3 | 0.0 (held) | 2 | 174.80 | 165.88 | 169.85 |
| combinations with setups | 52 | 26.5 | 15.0 (held) | 2 | 241.90 | 228.88 | 232.70 |

### (ix) time of the first schedule (seconds since the request)

| method | time | median | minimum | maximum | cells |
|---|---|---|---|---|---|
| dispatching rules (best of four) | t_all | 1.13 | 0.94 | 1.29 | 240 |
| policy, greedy schedule | t | 2.31 | 1.96 | 2.56 | 240 |
| policy, greedy schedule, from loaded | t_resident | 0.88 | 0.69 | 1.04 | 240 |
| dispatching rules (best of four), combinations without setups | t_all | 1.13 | 0.94 | 1.29 | 120 |
| policy, greedy schedule, combinations without setups | t | 2.32 | 2.01 | 2.56 | 120 |
| policy, greedy schedule, from loaded, combinations without setups | t_resident | 0.88 | 0.72 | 1.01 | 120 |
| dispatching rules (best of four), combinations with setups | t_all | 1.13 | 0.95 | 1.29 | 120 |
| policy, greedy schedule, combinations with setups | t | 2.31 | 1.96 | 2.54 | 120 |
| policy, greedy schedule, from loaded, combinations with setups | t_resident | 0.88 | 0.69 | 1.04 | 120 |

### Curves: first grid time at which every cell of the method has a schedule

| method | from s |
|---|---|
| memetic_rule | 1.47 |
| cpsat_rule | 1.47 |
| policy_sample | 2.61 |
| memetic_rule, combinations without setups | 1.47 |
| cpsat_rule, combinations without setups | 1.47 |
| policy_sample, combinations without setups | 2.61 |
| memetic_rule, combinations with setups | 1.47 |
| cpsat_rule, combinations with setups | 1.47 |
| policy_sample, combinations with setups | 2.61 |

### Decision table: difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Pool | +1.29 (30) | -3.13 (30) | **-4.29** (30) | -3.28 (10) | - | - | memetic_rule | no, by design |
| 10 | combinations without setups | +1.70 (30) | -3.09 (30) | -3.91 (30) | **-4.51** (10) | - | - | cpsat_rule | no, by design |
| 10 | combinations with setups | +0.88 (30) | -3.16 (30) | **-4.68** (30) | -2.05 (10) | - | - | memetic_rule | no, by design |
| 60 | Pool | +1.29 (30) | -3.83 (30) | **-5.87** (30) | -5.05 (10) | - | - | memetic_rule | no, by design |
| 60 | combinations without setups | +1.70 (30) | -3.89 (30) | -5.33 (30) | **-5.68** (10) | - | - | cpsat_rule | no, by design |
| 60 | combinations with setups | +0.88 (30) | -3.78 (30) | **-6.41** (30) | -4.42 (10) | - | - | memetic_rule | no, by design |
| 300 | Pool | +1.29 (30) | -3.83 (30) | **-6.63** (30) | -5.74 (10) | - | - | memetic_rule | no, by design |
| 300 | combinations without setups | +1.70 (30) | -3.89 (30) | **-6.13** (30) | -5.95 (10) | - | - | memetic_rule | no, by design |
| 300 | combinations with setups | +0.88 (30) | -3.78 (30) | **-7.14** (30) | -5.53 (10) | - | - | memetic_rule | no, by design |
| 600 | Pool | +1.29 (30) | -3.83 (30) | **-6.78** (30) | -5.93 (10) | - | - | memetic_rule | no, by design |
| 600 | combinations without setups | +1.70 (30) | -3.89 (30) | **-6.26** (30) | -5.99 (10) | - | - | memetic_rule | no, by design |
| 600 | combinations with setups | +0.88 (30) | -3.78 (30) | **-7.30** (30) | -5.88 (10) | - | - | memetic_rule | no, by design |

### Decision table on the instances common to every method of the size (0-9, 10 instances): difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 10 | Pool | +1.52 (10) | -3.13 (10) | **-4.10** (10) | -3.28 (10) | - | - | memetic_rule | yes |
| 10 | combinations without setups | +2.25 (10) | -2.67 (10) | -3.14 (10) | **-4.51** (10) | - | - | cpsat_rule | yes |
| 10 | combinations with setups | +0.79 (10) | -3.58 (10) | **-5.06** (10) | -2.05 (10) | - | - | memetic_rule | yes |
| 60 | Pool | +1.52 (10) | -3.89 (10) | **-5.63** (10) | -5.05 (10) | - | - | memetic_rule | yes |
| 60 | combinations without setups | +2.25 (10) | -3.52 (10) | -4.20 (10) | **-5.68** (10) | - | - | cpsat_rule | yes |
| 60 | combinations with setups | +0.79 (10) | -4.26 (10) | **-7.05** (10) | -4.42 (10) | - | - | memetic_rule | yes |
| 300 | Pool | +1.52 (10) | -3.89 (10) | **-6.31** (10) | -5.74 (10) | - | - | memetic_rule | yes |
| 300 | combinations without setups | +2.25 (10) | -3.52 (10) | -5.00 (10) | **-5.95** (10) | - | - | cpsat_rule | yes |
| 300 | combinations with setups | +0.79 (10) | -4.26 (10) | **-7.62** (10) | -5.53 (10) | - | - | memetic_rule | yes |
| 600 | Pool | +1.52 (10) | -3.89 (10) | **-6.43** (10) | -5.93 (10) | - | - | memetic_rule | yes |
| 600 | combinations without setups | +2.25 (10) | -3.52 (10) | -5.01 (10) | **-5.99** (10) | - | - | cpsat_rule | yes |
| 600 | combinations with setups | +0.79 (10) | -4.26 (10) | **-7.85** (10) | -5.88 (10) | - | - | memetic_rule | yes |

### Size of the solver's model: 80 valid records of CP-SAT started from the rule schedule; median [minimum, maximum] over the cells

| entry | cells | variables of the initial model | seconds to build the model | peak memory of the solve in GB | seconds until the first incumbent, on the record's clock (for a model configuration from the solver call, after the model build) |
|---|---|---|---|---|---|
| N | 10 | - | 3.07 [2.50, 3.82] | 0.2 [0.2, 0.2] | 0.0 [0.0, 0.0] |
| L | 10 | - | 242.80 [3.69, 265.22] | 0.2 [0.2, 0.2] | 0.0 [0.0, 0.0] |
| S | 10 | - | 16.32 [13.56, 26.93] | 0.7 [0.5, 0.9] | 0.0 [0.0, 0.0] |
| W | 10 | - | 3.80 [2.94, 4.54] | 0.2 [0.2, 0.2] | 0.0 [0.0, 0.0] |
| LS | 10 | - | 27.37 [15.95, 47.06] | 0.8 [0.5, 0.9] | 0.0 [0.0, 0.0] |
| LW | 10 | - | 112.19 [3.00, 250.45] | 0.2 [0.2, 0.2] | 0.0 [0.0, 0.0] |
| SW | 10 | - | 36.26 [23.65, 48.30] | 1.0 [0.8, 1.0] | 0.0 [0.0, 0.0] |
| LSW | 10 | - | 33.88 [20.78, 46.67] | 0.9 [0.8, 0.9] | 0.0 [0.0, 0.0] |
| Pool | 80 | - | 24.46 [2.50, 265.22] | 0.4 [0.2, 1.0] | 0.0 [0.0, 0.0] |
| combinations without setups | 40 | - | 3.83 [2.50, 265.22] | 0.2 [0.2, 0.2] | 0.0 [0.0, 0.0] |
| combinations with setups | 40 | - | 28.44 [13.56, 48.30] | 0.8 [0.5, 1.0] | 0.0 [0.0, 0.0] |

- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_000.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_001.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_002.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_003.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_004.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_005.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_006.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_007.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_008.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/N/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/L/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/S/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/W/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LS/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/SW/instance_009.json (no solver_log_head)
- no value of the variables of the initial model: test_results/FAMILY/10x25_cparms/evidence_rule600s/lns/LSW/instance_009.json (no solver_log_head)

## Record counts per cell

| arm, size, combination | states |
|---|---|
| cpsat_policy|50x25n30|L | valid 9 |
| cpsat_policy|50x25n30|LS | valid 9 |
| cpsat_policy|50x25n30|LSW | valid 9 |
| cpsat_policy|50x25n30|LW | valid 9 |
| cpsat_policy|50x25n30|N | valid 9 |
| cpsat_policy|50x25n30|S | valid 9 |
| cpsat_policy|50x25n30|SW | valid 9 |
| cpsat_policy|50x25n30|W | valid 9 |
| cpsat_policy|80x25n30|L | valid 9 |
| cpsat_policy|80x25n30|LS | valid 9 |
| cpsat_policy|80x25n30|LSW | valid 9 |
| cpsat_policy|80x25n30|LW | valid 9 |
| cpsat_policy|80x25n30|N | valid 9 |
| cpsat_policy|80x25n30|S | valid 9 |
| cpsat_policy|80x25n30|SW | valid 9 |
| cpsat_policy|80x25n30|W | valid 9 |
| cpsat_rule|10x25|L | valid 10 |
| cpsat_rule|10x25|LS | valid 10 |
| cpsat_rule|10x25|LSW | valid 10 |
| cpsat_rule|10x25|LW | valid 10 |
| cpsat_rule|10x25|N | valid 10 |
| cpsat_rule|10x25|S | valid 10 |
| cpsat_rule|10x25|SW | valid 10 |
| cpsat_rule|10x25|W | valid 10 |
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
| memetic_policy|50x25n30|L | valid 9 |
| memetic_policy|50x25n30|LS | valid 9 |
| memetic_policy|50x25n30|LSW | valid 9 |
| memetic_policy|50x25n30|LW | valid 9 |
| memetic_policy|50x25n30|N | valid 9 |
| memetic_policy|50x25n30|S | valid 9 |
| memetic_policy|50x25n30|SW | valid 9 |
| memetic_policy|50x25n30|W | valid 9 |
| memetic_policy|80x25n30|L | valid 9 |
| memetic_policy|80x25n30|LS | valid 9 |
| memetic_policy|80x25n30|LSW | valid 9 |
| memetic_policy|80x25n30|LW | valid 9 |
| memetic_policy|80x25n30|N | valid 9 |
| memetic_policy|80x25n30|S | valid 9 |
| memetic_policy|80x25n30|SW | valid 9 |
| memetic_policy|80x25n30|W | valid 9 |
| memetic_rule|10x25|L | valid 30 |
| memetic_rule|10x25|LS | valid 30 |
| memetic_rule|10x25|LSW | valid 30 |
| memetic_rule|10x25|LW | valid 30 |
| memetic_rule|10x25|N | valid 30 |
| memetic_rule|10x25|S | valid 30 |
| memetic_rule|10x25|SW | valid 30 |
| memetic_rule|10x25|W | valid 30 |
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
| policy_sample|10x25|L | valid 30 |
| policy_sample|10x25|LS | valid 30 |
| policy_sample|10x25|LSW | valid 30 |
| policy_sample|10x25|LW | valid 30 |
| policy_sample|10x25|N | valid 30 |
| policy_sample|10x25|S | valid 30 |
| policy_sample|10x25|SW | valid 30 |
| policy_sample|10x25|W | valid 30 |
| policy_sample|50x25n30|L | valid 9 |
| policy_sample|50x25n30|LS | valid 9 |
| policy_sample|50x25n30|LSW | valid 9 |
| policy_sample|50x25n30|LW | valid 9 |
| policy_sample|50x25n30|N | valid 9 |
| policy_sample|50x25n30|S | valid 9 |
| policy_sample|50x25n30|SW | valid 9 |
| policy_sample|50x25n30|W | valid 9 |
| policy_sample|80x25n30|L | valid 9 |
| policy_sample|80x25n30|LS | valid 9 |
| policy_sample|80x25n30|LSW | valid 9 |
| policy_sample|80x25n30|LW | valid 9 |
| policy_sample|80x25n30|N | valid 9 |
| policy_sample|80x25n30|S | valid 9 |
| policy_sample|80x25n30|SW | valid 9 |
| policy_sample|80x25n30|W | valid 9 |
| rule_start|10x25|L | valid 30 |
| rule_start|10x25|LS | valid 30 |
| rule_start|10x25|LSW | valid 30 |
| rule_start|10x25|LW | valid 30 |
| rule_start|10x25|N | valid 30 |
| rule_start|10x25|S | valid 30 |
| rule_start|10x25|SW | valid 30 |
| rule_start|10x25|W | valid 30 |
| rule_start|50x25n30|L | valid 9 |
| rule_start|50x25n30|LS | valid 9 |
| rule_start|50x25n30|LSW | valid 9 |
| rule_start|50x25n30|LW | valid 9 |
| rule_start|50x25n30|N | valid 9 |
| rule_start|50x25n30|S | valid 9 |
| rule_start|50x25n30|SW | valid 9 |
| rule_start|50x25n30|W | valid 9 |
| rule_start|80x25n30|L | valid 9 |
| rule_start|80x25n30|LS | valid 9 |
| rule_start|80x25n30|LSW | valid 9 |
| rule_start|80x25n30|LW | valid 9 |
| rule_start|80x25n30|N | valid 9 |
| rule_start|80x25n30|S | valid 9 |
| rule_start|80x25n30|SW | valid 9 |
| rule_start|80x25n30|W | valid 9 |
