# Sensitivity of the factory benchmark: the protocol of the manuscript (scripts/p87_sensitivity.py infer)

Mask exact, training seeds 301,302,303,304,305, root test_results/FAMILY/sens_p87. Gaps in %: first-named method minus second over second, seed s against seed s; negative = first-named method shorter. Intervals: crossed hierarchical bootstrap over training runs and base instances, B = 20000, fresh default_rng(11) per quantity (one training run: bootstrap over base instances, default_rng(7)). Verdicts against the specialist at the margins 0.5% / 1%; against the best dispatching rule (per-instance minimum of FIFO, MOR, SPT, MWKR; rule set masked, common_utils.masked_rule_action) the combined verdict with Holm over the combinations the level reaches; pooled rows and combinations a level does not reach give the interval and the direction. A combination a level does not reach takes the default level's results (marked "not reached").

Measured variables: per level and combination, the share of decisions at which the mask admits exactly one (job, machine) pair, over all decisions of the instances, and the setup share of machine capacity, per instance the total setup time over the machines used times the makespan, averaged over the instances (the definitions of scripts/p93_rule_comparison.py trace_stats); the policy as the mean over its training runs, the best dispatching rule as the schedule of the per-instance best rule (a tie goes to the first of FIFO, MOR, SPT, MWKR); from the counts of evaluate --trace_counts

| level | parameter | mark | combination | n | policy - specialist | control - specialist | policy - best rule |
|---|---|---|---|---|---|---|---|
| default |  | inside | N | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] ahead |
| default |  | inside | L | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] ahead |
| default |  | inside | S | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] not distinguishable |
| default |  | inside | W | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] ahead |
| default |  | inside | LS | 100 | -0.07 [-0.37, +0.25] equivalent / equivalent | +0.13 [-0.14, +0.41] equivalent / equivalent | -1.10 [-1.53, -0.65] ahead |
| default |  | inside | LW | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] ahead |
| default |  | inside | SW | 100 | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.40 [-0.86, +0.09] non-inferior only / equivalent | -0.98 [-1.47, -0.50] ahead |
| default |  | inside | LSW | 100 | -0.15 [-0.53, +0.23] non-inferior only / equivalent | +0.01 [-0.44, +0.43] equivalent / equivalent | -1.21 [-1.65, -0.77] ahead |
| default |  | inside | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.03 [-0.18, +0.24] equivalent / equivalent | +0.01 [-0.23, +0.25] equivalent / equivalent | -1.39 [-1.66, -1.13] direction ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | N | 100 | +0.41 [-0.26, +1.04] inconclusive / inconclusive | +0.08 [-0.56, +0.76] inconclusive / equivalent | -1.81 [-2.38, -1.24] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | L | 100 | +0.21 [-0.05, +0.47] equivalent / equivalent | +0.20 [-0.16, +0.63] inconclusive / equivalent | -1.31 [-1.80, -0.82] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | S | 100 | -0.22 [-0.69, +0.22] non-inferior only / equivalent | -0.37 [-0.70, -0.04] non-inferior only / equivalent | -0.45 [-1.00, +0.14] not distinguishable |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | W | 100 | +0.11 [-0.59, +0.89] inconclusive / equivalent | -0.03 [-0.81, +0.78] inconclusive / equivalent | -2.26 [-2.99, -1.58] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | LS | 100 | -0.04 [-0.35, +0.30] equivalent / equivalent | +0.13 [-0.17, +0.43] equivalent / equivalent | -1.15 [-1.63, -0.65] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | LW | 100 | +0.38 [-0.05, +0.79] inconclusive / equivalent | +0.40 [+0.04, +0.73] inferior / equivalent | -1.81 [-2.29, -1.33] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | SW | 100 | -0.54 [-1.17, +0.06] non-inferior only / non-inferior only | -0.63 [-1.07, -0.19] non-inferior only / non-inferior only | -0.93 [-1.49, -0.40] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | LSW | 100 | -0.21 [-0.58, +0.20] non-inferior only / equivalent | -0.08 [-0.51, +0.38] non-inferior only / equivalent | -0.97 [-1.50, -0.42] ahead |
| dur_cv10 | duration coefficient of variation = 0.1 | inside (0.02 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.01 [-0.17, +0.18] equivalent / equivalent | -0.04 [-0.24, +0.16] equivalent / equivalent | -1.34 [-1.60, -1.08] direction ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | N | 100 | +0.42 [-0.33, +1.15] inconclusive / inconclusive | +0.06 [-0.53, +0.64] inconclusive / equivalent | -2.05 [-2.77, -1.28] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | L | 100 | +0.26 [-0.08, +0.58] inconclusive / equivalent | +0.09 [-0.20, +0.41] equivalent / equivalent | -0.94 [-1.48, -0.41] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | S | 100 | -0.33 [-0.91, +0.17] non-inferior only / equivalent | -0.41 [-0.85, +0.01] non-inferior only / equivalent | -0.62 [-1.23, -0.02] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | W | 100 | +0.08 [-0.74, +0.88] inconclusive / equivalent | -0.28 [-1.08, +0.54] inconclusive / non-inferior only | -3.00 [-3.77, -2.24] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | LS | 100 | +0.02 [-0.29, +0.36] equivalent / equivalent | +0.09 [-0.22, +0.40] equivalent / equivalent | -0.98 [-1.50, -0.43] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | LW | 100 | +0.24 [-0.13, +0.59] inconclusive / equivalent | +0.32 [-0.03, +0.70] inconclusive / equivalent | -1.49 [-2.04, -0.95] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | SW | 100 | -0.69 [-1.17, -0.21] non-inferior only / non-inferior only | -0.39 [-0.98, +0.24] non-inferior only / equivalent | -1.36 [-2.03, -0.71] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | LSW | 100 | -0.07 [-0.44, +0.32] equivalent / equivalent | +0.07 [-0.30, +0.47] equivalent / equivalent | -1.45 [-1.94, -0.96] ahead |
| dur_cv20 | duration coefficient of variation = 0.2 | outside in spread (0.08 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.01 [-0.23, +0.19] equivalent / equivalent | -0.06 [-0.28, +0.17] equivalent / equivalent | -1.49 [-1.80, -1.18] direction ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | N | 100 | +0.63 [-0.35, +1.50] inconclusive / inconclusive | +0.43 [-0.23, +1.08] inconclusive / inconclusive | -2.26 [-3.04, -1.49] ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | L | 100 | +0.23 [-0.12, +0.58] inconclusive / equivalent | +0.08 [-0.28, +0.47] equivalent / equivalent | -0.53 [-1.03, -0.04] not distinguishable |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | S | 100 | -0.28 [-0.81, +0.24] non-inferior only / equivalent | -0.45 [-0.86, -0.04] non-inferior only / equivalent | -0.46 [-1.12, +0.24] not distinguishable |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | W | 100 | -0.27 [-1.09, +0.60] inconclusive / non-inferior only | -0.40 [-1.18, +0.44] non-inferior only / non-inferior only | -3.24 [-4.15, -2.33] ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | LS | 100 | -0.11 [-0.45, +0.25] equivalent / equivalent | +0.13 [-0.26, +0.51] inconclusive / equivalent | -0.78 [-1.26, -0.28] ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | LW | 100 | +0.15 [-0.31, +0.61] inconclusive / equivalent | +0.10 [-0.28, +0.48] equivalent / equivalent | -1.08 [-1.62, -0.56] ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | SW | 100 | -0.34 [-0.85, +0.18] non-inferior only / equivalent | -0.54 [-1.13, +0.04] non-inferior only / non-inferior only | -0.47 [-1.20, +0.29] not distinguishable |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | LSW | 100 | -0.15 [-0.52, +0.22] non-inferior only / equivalent | -0.07 [-0.42, +0.29] equivalent / equivalent | -0.89 [-1.42, -0.37] ahead |
| dur_cv30 | duration coefficient of variation = 0.3 | outside in spread (0.16 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.02 [-0.28, +0.22] equivalent / equivalent | -0.09 [-0.30, +0.13] equivalent / equivalent | -1.21 [-1.53, -0.91] direction ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | N | 100 | -0.09 [-0.89, +0.59] inconclusive / equivalent | -0.21 [-1.11, +0.61] inconclusive / non-inferior only | -2.00 [-2.50, -1.49] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | L | 100 | +0.22 [-0.17, +0.60] inconclusive / equivalent | +0.21 [-0.19, +0.60] inconclusive / equivalent | -1.07 [-1.62, -0.51] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | S | 100 | -0.40 [-0.95, +0.15] non-inferior only / equivalent | -0.38 [-0.87, +0.07] non-inferior only / equivalent | -0.38 [-1.00, +0.27] not distinguishable |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | W | 100 | -0.11 [-0.69, +0.48] non-inferior only / equivalent | -0.34 [-0.89, +0.19] non-inferior only / equivalent | -2.37 [-3.09, -1.67] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | LS | 100 | -0.13 [-0.41, +0.15] equivalent / equivalent | +0.10 [-0.23, +0.45] equivalent / equivalent | -1.16 [-1.59, -0.72] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | LW | 100 | +0.72 [+0.26, +1.19] inferior / inferior | +0.88 [+0.47, +1.28] inferior / inferior | -1.55 [-2.21, -0.91] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | SW | 100 | -0.01 [-0.46, +0.44] equivalent / equivalent | -0.09 [-0.52, +0.33] non-inferior only / equivalent | -0.66 [-1.33, +0.01] not distinguishable |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | LSW | 100 | -0.09 [-0.60, +0.39] non-inferior only / equivalent | -0.09 [-0.63, +0.39] non-inferior only / equivalent | -1.03 [-1.55, -0.50] ahead |
| dur_long_m25 | duration scale of the finishing and assembly stations = 0.75 | inside (0.02 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.01 [-0.17, +0.19] equivalent / equivalent | +0.01 [-0.23, +0.21] equivalent / equivalent | -1.28 [-1.58, -0.98] direction ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | N | 100 | +0.94 [+0.42, +1.49] inferior / inferior | +0.41 [-0.31, +1.10] inconclusive / inconclusive | -1.60 [-2.20, -0.98] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | L | 100 | +0.15 [-0.08, +0.39] equivalent / equivalent | +0.19 [-0.10, +0.49] equivalent / equivalent | -1.57 [-2.06, -1.10] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | S | 100 | -0.33 [-0.72, +0.04] non-inferior only / equivalent | -0.35 [-0.64, -0.04] non-inferior only / equivalent | -0.94 [-1.49, -0.39] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | W | 100 | +0.34 [-0.73, +1.28] inconclusive / inconclusive | +0.18 [-1.14, +1.30] inconclusive / inconclusive | -2.74 [-3.48, -2.01] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | LS | 100 | -0.15 [-0.51, +0.22] non-inferior only / equivalent | +0.09 [-0.19, +0.39] equivalent / equivalent | -1.09 [-1.55, -0.62] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | LW | 100 | +0.16 [-0.28, +0.57] inconclusive / equivalent | +0.10 [-0.27, +0.44] equivalent / equivalent | -2.18 [-2.65, -1.72] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | SW | 100 | -0.38 [-0.77, +0.02] non-inferior only / equivalent | -0.40 [-0.81, -0.01] non-inferior only / equivalent | -0.43 [-0.96, +0.09] not distinguishable |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | LSW | 100 | +0.09 [-0.26, +0.50] equivalent / equivalent | -0.06 [-0.47, +0.37] equivalent / equivalent | -0.99 [-1.46, -0.51] ahead |
| dur_long_p25 | duration scale of the finishing and assembly stations = 1.25 | inside (0.08 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.10 [-0.09, +0.30] equivalent / equivalent | +0.02 [-0.29, +0.28] equivalent / equivalent | -1.44 [-1.70, -1.19] direction ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | L | 100 | -0.93 [-1.76, -0.11] non-inferior only / non-inferior only | -1.08 [-1.59, -0.58] non-inferior only / non-inferior only | -2.57 [-3.11, -2.00] ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | LS | 100 | -0.01 [-0.33, +0.31] equivalent / equivalent | -0.14 [-0.40, +0.12] equivalent / equivalent | -1.08 [-1.53, -0.63] ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | LW | 100 | -1.69 [-3.19, -0.59] non-inferior only / non-inferior only | -2.26 [-3.59, -1.25] non-inferior only / non-inferior only | -2.18 [-2.76, -1.59] ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | SW (not reached) | 100 | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.40 [-0.86, +0.09] non-inferior only / equivalent | -0.98 [-1.47, -0.50] direction ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | LSW | 100 | -0.42 [-0.94, +0.10] non-inferior only / equivalent | -0.54 [-1.02, -0.06] non-inferior only / non-inferior only | -1.59 [-2.15, -1.04] ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.40 [-0.57, -0.24] non-inferior only / equivalent | -0.62 [-0.81, -0.41] non-inferior only / equivalent | -1.60 [-1.89, -1.31] direction ahead |
| lag_x050 | lag scale = 0.5 | outside (0.87 outside) | pooled L,LS,LW,LSW | 100 | -0.76 [-1.25, -0.31] non-inferior only / non-inferior only | -1.00 [-1.29, -0.74] non-inferior only / non-inferior only | -1.86 [-2.19, -1.52] direction ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | L | 100 | +0.19 [-0.14, +0.53] inconclusive / equivalent | +0.29 [-0.07, +0.61] inconclusive / equivalent | -2.39 [-2.86, -1.90] ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | LS | 100 | -0.25 [-0.56, +0.05] non-inferior only / equivalent | -0.20 [-0.45, +0.06] equivalent / equivalent | -1.11 [-1.55, -0.68] ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | LW | 100 | -0.15 [-0.87, +0.49] non-inferior only / equivalent | -0.21 [-0.69, +0.25] non-inferior only / equivalent | -2.82 [-3.41, -2.25] ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | SW (not reached) | 100 | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.40 [-0.86, +0.09] non-inferior only / equivalent | -0.98 [-1.47, -0.50] direction ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | LSW | 100 | -0.34 [-0.82, +0.14] non-inferior only / equivalent | -0.26 [-0.83, +0.27] non-inferior only / equivalent | -1.69 [-2.20, -1.17] ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.09 [-0.27, +0.08] equivalent / equivalent | -0.16 [-0.40, +0.06] equivalent / equivalent | -1.67 [-1.95, -1.40] direction ahead |
| lag_x075 | lag scale = 0.75 | inside (0.31 outside) | pooled L,LS,LW,LSW | 100 | -0.14 [-0.34, +0.04] equivalent / equivalent | -0.10 [-0.30, +0.11] equivalent / equivalent | -2.00 [-2.34, -1.67] direction ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | L | 100 | +0.26 [+0.03, +0.51] inferior / equivalent | +0.20 [-0.07, +0.48] equivalent / equivalent | -0.48 [-0.96, -0.00] ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | LS | 100 | +0.00 [-0.22, +0.23] equivalent / equivalent | +0.20 [-0.09, +0.51] inconclusive / equivalent | -0.62 [-1.02, -0.21] ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | LW | 100 | +0.50 [+0.10, +0.89] inferior / equivalent | +0.51 [+0.12, +0.89] inferior / equivalent | -1.11 [-1.67, -0.57] ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | SW (not reached) | 100 | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.40 [-0.86, +0.09] non-inferior only / equivalent | -0.98 [-1.47, -0.50] direction ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | LSW | 100 | -0.12 [-0.55, +0.27] non-inferior only / equivalent | -0.14 [-0.58, +0.29] non-inferior only / equivalent | -0.45 [-0.94, +0.06] not distinguishable |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.06 [-0.14, +0.26] equivalent / equivalent | -0.02 [-0.26, +0.21] equivalent / equivalent | -1.00 [-1.28, -0.72] direction ahead |
| lag_x125 | lag scale = 1.25 | inside (0.40 outside) | pooled L,LS,LW,LSW | 100 | +0.16 [+0.00, +0.32] equivalent / equivalent | +0.19 [-0.02, +0.41] equivalent / equivalent | -0.66 [-1.02, -0.30] direction ahead |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | L | 100 | +0.16 [-0.04, +0.38] equivalent / equivalent | +0.09 [-0.15, +0.32] equivalent / equivalent | -0.14 [-0.55, +0.26] not distinguishable |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | LS | 100 | -0.06 [-0.27, +0.15] equivalent / equivalent | +0.18 [-0.06, +0.44] equivalent / equivalent | -0.18 [-0.55, +0.20] not distinguishable |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | LW | 100 | +0.22 [-0.05, +0.48] equivalent / equivalent | +0.24 [-0.04, +0.52] inconclusive / equivalent | -0.43 [-0.87, +0.01] not distinguishable |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | SW (not reached) | 100 | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.40 [-0.86, +0.09] non-inferior only / equivalent | -0.98 [-1.47, -0.50] direction ahead |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | LSW | 100 | -0.03 [-0.38, +0.33] equivalent / equivalent | +0.01 [-0.42, +0.45] equivalent / equivalent | -0.26 [-0.79, +0.28] not distinguishable |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.01 [-0.20, +0.22] equivalent / equivalent | -0.05 [-0.30, +0.18] equivalent / equivalent | -0.79 [-1.07, -0.52] direction ahead |
| lag_x150 | lag scale = 1.5 | outside (0.63 outside) | pooled L,LS,LW,LSW | 100 | +0.07 [-0.08, +0.23] equivalent / equivalent | +0.13 [-0.06, +0.31] equivalent / equivalent | -0.25 [-0.58, +0.09] direction neither |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | S | 100 | -0.54 [-1.36, +0.19] non-inferior only / non-inferior only | -0.84 [-1.46, -0.27] non-inferior only / non-inferior only | -0.89 [-1.43, -0.38] ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | LS | 100 | -0.12 [-0.45, +0.18] equivalent / equivalent | -0.05 [-0.39, +0.28] equivalent / equivalent | -1.48 [-1.95, -0.99] ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | SW | 100 | -0.72 [-1.19, -0.28] non-inferior only / non-inferior only | -0.81 [-1.33, -0.31] non-inferior only / non-inferior only | -1.42 [-2.03, -0.80] ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | LSW | 100 | +0.29 [-0.18, +0.74] inconclusive / equivalent | +0.42 [-0.04, +0.92] inconclusive / equivalent | -2.04 [-2.60, -1.52] ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.02 [-0.17, +0.22] equivalent / equivalent | -0.09 [-0.30, +0.13] equivalent / equivalent | -1.65 [-1.96, -1.34] direction ahead |
| setup_x050 | setup scale = 0.5 | outside (0.23 outside) | pooled S,LS,SW,LSW | 100 | -0.27 [-0.54, -0.02] non-inferior only / equivalent | -0.32 [-0.56, -0.08] non-inferior only / equivalent | -1.46 [-1.81, -1.12] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | S | 100 | -0.35 [-0.83, +0.10] non-inferior only / equivalent | -0.48 [-0.98, -0.04] non-inferior only / equivalent | -1.01 [-1.47, -0.54] ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | LS | 100 | +0.05 [-0.24, +0.35] equivalent / equivalent | +0.20 [-0.15, +0.60] inconclusive / equivalent | -0.81 [-1.21, -0.39] ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | SW | 100 | -0.40 [-0.92, +0.12] non-inferior only / equivalent | -0.42 [-1.04, +0.16] non-inferior only / non-inferior only | -1.11 [-1.75, -0.45] ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | LSW | 100 | +0.05 [-0.38, +0.47] equivalent / equivalent | -0.02 [-0.43, +0.40] equivalent / equivalent | -1.24 [-1.76, -0.72] ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.08 [-0.11, +0.26] equivalent / equivalent | -0.02 [-0.21, +0.18] equivalent / equivalent | -1.44 [-1.72, -1.15] direction ahead |
| setup_x075 | setup scale = 0.75 | inside (0.00 outside) | pooled S,LS,SW,LSW | 100 | -0.16 [-0.43, +0.10] equivalent / equivalent | -0.18 [-0.42, +0.05] equivalent / equivalent | -1.04 [-1.38, -0.69] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | S | 100 | -0.17 [-0.59, +0.24] non-inferior only / equivalent | -0.06 [-0.45, +0.29] equivalent / equivalent | -0.80 [-1.29, -0.31] ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | LS | 100 | -0.05 [-0.32, +0.22] equivalent / equivalent | +0.06 [-0.26, +0.39] equivalent / equivalent | -0.66 [-1.03, -0.28] ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | SW | 100 | -0.31 [-0.77, +0.16] non-inferior only / equivalent | -0.42 [-0.85, +0.01] non-inferior only / equivalent | -1.25 [-1.82, -0.69] ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | LSW | 100 | +0.04 [-0.36, +0.43] equivalent / equivalent | +0.06 [-0.45, +0.52] inconclusive / equivalent | -1.05 [-1.49, -0.60] ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.10 [-0.09, +0.29] equivalent / equivalent | +0.03 [-0.19, +0.24] equivalent / equivalent | -1.38 [-1.65, -1.12] direction ahead |
| setup_x125 | setup scale = 1.25 | inside (0.27 outside) | pooled S,LS,SW,LSW | 100 | -0.12 [-0.32, +0.07] equivalent / equivalent | -0.09 [-0.33, +0.15] equivalent / equivalent | -0.94 [-1.24, -0.64] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | S | 100 | -0.44 [-0.93, -0.01] non-inferior only / equivalent | -0.30 [-0.68, +0.08] non-inferior only / equivalent | -0.50 [-1.00, +0.01] not distinguishable |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | LS | 100 | -0.13 [-0.41, +0.16] equivalent / equivalent | +0.04 [-0.26, +0.35] equivalent / equivalent | -1.30 [-1.69, -0.89] ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | SW | 100 | +0.03 [-0.47, +0.57] inconclusive / equivalent | -0.06 [-0.51, +0.46] non-inferior only / equivalent | -0.54 [-1.05, -0.03] ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | LSW | 100 | -0.22 [-0.64, +0.19] non-inferior only / equivalent | -0.08 [-0.52, +0.36] non-inferior only / equivalent | -1.32 [-1.78, -0.85] ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.07 [-0.13, +0.27] equivalent / equivalent | +0.02 [-0.20, +0.26] equivalent / equivalent | -1.37 [-1.61, -1.13] direction ahead |
| setup_x150 | setup scale = 1.5 | outside (0.55 outside) | pooled S,LS,SW,LSW | 100 | -0.19 [-0.41, +0.01] equivalent / equivalent | -0.10 [-0.34, +0.16] equivalent / equivalent | -0.91 [-1.19, -0.65] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | S | 100 | +0.19 [-0.67, +0.89] inconclusive / equivalent | +0.09 [-0.58, +0.70] inconclusive / equivalent | +0.19 [-0.50, +0.91] not distinguishable |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | LS | 100 | +0.16 [-0.16, +0.51] inconclusive / equivalent | +0.26 [-0.07, +0.61] inconclusive / equivalent | -0.74 [-1.24, -0.23] ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | SW | 100 | +0.14 [-0.45, +0.79] inconclusive / equivalent | -0.10 [-0.70, +0.50] inconclusive / equivalent | +0.36 [-0.27, +1.04] not distinguishable |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | LSW | 100 | -0.06 [-0.65, +0.55] inconclusive / equivalent | -0.06 [-0.60, +0.45] non-inferior only / equivalent | -0.80 [-1.35, -0.23] ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.21 [-0.01, +0.44] equivalent / equivalent | +0.10 [-0.15, +0.34] equivalent / equivalent | -1.04 [-1.31, -0.77] direction ahead |
| setup_same0 | same-class setup scale = 0.0 | outside (0.20 outside) | pooled S,LS,SW,LSW | 100 | +0.11 [-0.23, +0.45] equivalent / equivalent | +0.05 [-0.22, +0.34] equivalent / equivalent | -0.25 [-0.60, +0.12] direction neither |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | S | 100 | -0.57 [-1.18, -0.02] non-inferior only / non-inferior only | -0.99 [-1.54, -0.50] non-inferior only / non-inferior only | -1.39 [-1.93, -0.87] ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | LS | 100 | -0.30 [-0.67, +0.08] non-inferior only / equivalent | -0.09 [-0.43, +0.25] equivalent / equivalent | -0.43 [-0.94, +0.07] not distinguishable |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | SW | 100 | -0.70 [-1.30, -0.06] non-inferior only / non-inferior only | -1.03 [-1.85, -0.09] non-inferior only / non-inferior only | -1.14 [-1.80, -0.48] ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | LSW | 100 | +0.01 [-0.51, +0.49] non-inferior only / equivalent | +0.22 [-0.21, +0.65] inconclusive / equivalent | -0.97 [-1.49, -0.46] ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.04 [-0.21, +0.13] equivalent / equivalent | -0.16 [-0.42, +0.08] equivalent / equivalent | -1.41 [-1.69, -1.13] direction ahead |
| setup_sub_strict | stations charged with setups = 7 | outside (0.00 outside) | pooled S,LS,SW,LSW | 100 | -0.39 [-0.65, -0.12] non-inferior only / equivalent | -0.47 [-0.80, -0.14] non-inferior only / equivalent | -0.99 [-1.33, -0.65] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | S | 100 | -0.16 [-0.64, +0.30] non-inferior only / equivalent | -0.42 [-0.90, +0.00] non-inferior only / equivalent | -1.08 [-1.62, -0.53] ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | W (not reached) | 100 | +0.40 [-0.51, +1.31] inconclusive / inconclusive | -0.22 [-1.29, +0.74] inconclusive / non-inferior only | -2.00 [-2.66, -1.33] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | LS | 100 | -0.03 [-0.39, +0.36] equivalent / equivalent | +0.10 [-0.21, +0.42] equivalent / equivalent | -1.03 [-1.48, -0.57] ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | LW (not reached) | 100 | +0.46 [+0.04, +0.88] inferior / equivalent | +0.56 [+0.21, +0.92] inferior / equivalent | -2.10 [-2.63, -1.57] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | SW | 100 | -0.51 [-1.07, +0.04] non-inferior only / non-inferior only | -0.42 [-1.02, +0.18] non-inferior only / non-inferior only | -1.00 [-1.65, -0.35] ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | LSW | 100 | -0.01 [-0.41, +0.39] equivalent / equivalent | +0.17 [-0.36, +0.67] inconclusive / equivalent | -1.08 [-1.56, -0.60] ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.07 [-0.15, +0.29] equivalent / equivalent | +0.00 [-0.24, +0.24] equivalent / equivalent | -1.44 [-1.72, -1.16] direction ahead |
| setup_sub_wide | stations charged with setups = 21 | outside (0.00 outside) | pooled S,LS,SW,LSW | 100 | -0.18 [-0.44, +0.09] equivalent / equivalent | -0.14 [-0.43, +0.15] equivalent / equivalent | -1.04 [-1.38, -0.71] direction ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | W | 100 | +0.05 [-0.52, +0.62] inconclusive / equivalent | -0.34 [-1.10, +0.41] non-inferior only / non-inferior only | -1.99 [-2.68, -1.31] ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | LS (not reached) | 100 | -0.07 [-0.37, +0.25] equivalent / equivalent | +0.13 [-0.14, +0.41] equivalent / equivalent | -1.10 [-1.53, -0.65] direction ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | LW | 100 | +0.41 [+0.07, +0.76] inferior / equivalent | +0.47 [+0.11, +0.83] inferior / equivalent | -1.72 [-2.22, -1.24] ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | SW | 100 | -0.27 [-0.70, +0.19] non-inferior only / equivalent | -0.36 [-0.75, +0.09] non-inferior only / equivalent | -0.74 [-1.29, -0.17] ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | LSW | 100 | -0.06 [-0.42, +0.27] equivalent / equivalent | +0.12 [-0.29, +0.57] inconclusive / equivalent | -1.19 [-1.69, -0.67] ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.02 [-0.14, +0.18] equivalent / equivalent | +0.00 [-0.19, +0.22] equivalent / equivalent | -1.31 [-1.60, -1.02] direction ahead |
| win_x050 | outage density scale = 0.5 | boundary (0.21 outside) | pooled W,LW,SW,LSW | 100 | +0.03 [-0.19, +0.26] equivalent / equivalent | -0.03 [-0.30, +0.28] equivalent / equivalent | -1.41 [-1.75, -1.08] direction ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | W | 100 | +0.53 [-0.57, +1.56] inconclusive / inconclusive | +0.10 [-1.04, +1.10] inconclusive / inconclusive | -1.80 [-2.61, -0.94] ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | LS (not reached) | 100 | -0.07 [-0.37, +0.25] equivalent / equivalent | +0.13 [-0.14, +0.41] equivalent / equivalent | -1.10 [-1.53, -0.65] direction ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | LW | 100 | +0.43 [-0.03, +0.89] inconclusive / equivalent | +0.47 [+0.02, +0.95] inferior / equivalent | -2.31 [-2.87, -1.77] ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | SW | 100 | -0.31 [-0.68, +0.07] non-inferior only / equivalent | -0.08 [-0.54, +0.37] non-inferior only / equivalent | -0.82 [-1.32, -0.33] ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | LSW | 100 | +0.10 [-0.34, +0.56] inconclusive / equivalent | +0.24 [-0.23, +0.71] inconclusive / equivalent | -1.22 [-1.75, -0.68] ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.09 [-0.11, +0.30] equivalent / equivalent | +0.11 [-0.10, +0.31] equivalent / equivalent | -1.37 [-1.64, -1.11] direction ahead |
| win_x150 | outage density scale = 1.5 | boundary (0.32 outside) | pooled W,LW,SW,LSW | 100 | +0.19 [-0.10, +0.47] equivalent / equivalent | +0.18 [-0.16, +0.49] equivalent / equivalent | -1.54 [-1.86, -1.21] direction ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | N (not reached) | 100 | +0.24 [-0.40, +0.86] inconclusive / equivalent | -0.08 [-0.63, +0.45] non-inferior only / equivalent | -1.86 [-2.36, -1.36] direction ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | L (not reached) | 100 | +0.18 [-0.11, +0.48] equivalent / equivalent | +0.32 [-0.00, +0.63] inconclusive / equivalent | -1.36 [-1.91, -0.82] direction ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | S (not reached) | 100 | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.22 [-0.55, +0.10] non-inferior only / equivalent | -0.52 [-1.08, +0.08] direction neither |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | W | 100 | +0.80 [-0.07, +1.60] inconclusive / inconclusive | +0.13 [-0.72, +0.96] inconclusive / equivalent | -2.44 [-3.25, -1.64] ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | LS (not reached) | 100 | -0.07 [-0.37, +0.25] equivalent / equivalent | +0.13 [-0.14, +0.41] equivalent / equivalent | -1.10 [-1.53, -0.65] direction ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | LW | 100 | +0.43 [-0.06, +1.00] inconclusive / inconclusive | +0.58 [+0.13, +1.04] inferior / inferior | -2.78 [-3.39, -2.18] ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | SW | 100 | -0.28 [-0.71, +0.16] non-inferior only / equivalent | -0.43 [-0.92, +0.09] non-inferior only / equivalent | -1.11 [-1.64, -0.55] ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | LSW | 100 | +0.17 [-0.27, +0.61] inconclusive / equivalent | +0.27 [-0.25, +0.79] inconclusive / equivalent | -1.42 [-1.94, -0.90] ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | pooled N,L,S,W,LS,LW,SW,LSW | 100 | +0.14 [-0.05, +0.32] equivalent / equivalent | +0.09 [-0.12, +0.28] equivalent / equivalent | -1.57 [-1.85, -1.30] direction ahead |
| win_x200 | outage density scale = 2.0 | outside (0.59 outside) | pooled W,LW,SW,LSW | 100 | +0.28 [-0.03, +0.59] inconclusive / equivalent | +0.14 [-0.18, +0.45] equivalent / equivalent | -1.94 [-2.28, -1.59] direction ahead |
| tight15 | stations in the factory = 15 | outside | N | 100 | -0.61 [-1.65, +0.34] non-inferior only / non-inferior only | -0.96 [-2.08, +0.09] non-inferior only / non-inferior only | -1.74 [-2.32, -1.15] ahead |
| tight15 | stations in the factory = 15 | outside | L | 100 | -1.10 [-2.43, +0.06] non-inferior only / non-inferior only | -0.38 [-0.93, +0.27] non-inferior only / equivalent | -3.39 [-4.11, -2.67] ahead |
| tight15 | stations in the factory = 15 | outside | S | 100 | -0.21 [-0.88, +0.48] non-inferior only / equivalent | -0.17 [-0.80, +0.45] non-inferior only / equivalent | -1.23 [-1.86, -0.60] ahead |
| tight15 | stations in the factory = 15 | outside | W | 100 | +0.19 [-0.61, +0.92] inconclusive / equivalent | -0.04 [-1.00, +0.91] inconclusive / equivalent | -2.75 [-3.31, -2.20] ahead |
| tight15 | stations in the factory = 15 | outside | LS | 100 | +0.12 [-0.36, +0.55] inconclusive / equivalent | +0.41 [-0.16, +0.95] inconclusive / equivalent | -1.61 [-2.17, -1.04] ahead |
| tight15 | stations in the factory = 15 | outside | LW | 100 | -1.92 [-4.62, +0.13] non-inferior only / non-inferior only | -1.48 [-4.31, +0.68] inconclusive / non-inferior only | -3.88 [-4.59, -3.18] ahead |
| tight15 | stations in the factory = 15 | outside | SW | 100 | -0.45 [-1.09, +0.15] non-inferior only / non-inferior only | -0.63 [-1.33, +0.08] non-inferior only / non-inferior only | -1.97 [-2.56, -1.39] ahead |
| tight15 | stations in the factory = 15 | outside | LSW | 100 | +0.43 [-0.30, +1.14] inconclusive / inconclusive | +0.55 [-0.32, +1.40] inconclusive / inconclusive | -1.61 [-2.23, -1.00] ahead |
| tight15 | stations in the factory = 15 | outside | pooled N,L,S,W,LS,LW,SW,LSW | 100 | -0.44 [-0.76, -0.14] non-inferior only / equivalent | -0.34 [-0.69, -0.02] non-inferior only / equivalent | -2.27 [-2.58, -1.97] direction ahead |
| mix_rc | project mix = reinforced concrete | outside | N | 50 | +2.90 [+2.01, +3.84] inferior / inferior | +0.28 [-0.92, +1.60] inconclusive / inconclusive | +0.98 [-0.45, +2.31] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | L | 50 | +0.17 [-0.43, +0.92] inconclusive / equivalent | -0.09 [-0.75, +0.56] inconclusive / equivalent | -1.54 [-2.34, -0.76] ahead |
| mix_rc | project mix = reinforced concrete | outside | S | 50 | +0.12 [-1.11, +1.19] inconclusive / inconclusive | -0.67 [-1.55, +0.16] non-inferior only / non-inferior only | +0.85 [-0.50, +2.41] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | W | 50 | +2.24 [-0.59, +4.45] inconclusive / inconclusive | -0.56 [-1.45, +0.28] non-inferior only / non-inferior only | +0.65 [-1.16, +2.44] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | LS | 50 | +0.09 [-0.31, +0.49] equivalent / equivalent | -0.22 [-0.69, +0.28] non-inferior only / equivalent | -0.34 [-0.90, +0.21] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | LW | 50 | -0.08 [-0.94, +0.68] inconclusive / equivalent | -0.37 [-1.39, +0.57] inconclusive / non-inferior only | -1.86 [-2.67, -1.01] ahead |
| mix_rc | project mix = reinforced concrete | outside | SW | 50 | +0.64 [-0.76, +2.10] inconclusive / inconclusive | +0.27 [-0.74, +1.36] inconclusive / inconclusive | -0.40 [-1.59, +0.87] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | LSW | 50 | -0.59 [-1.05, -0.11] non-inferior only / non-inferior only | -0.84 [-1.31, -0.35] non-inferior only / non-inferior only | -0.25 [-0.87, +0.38] not distinguishable |
| mix_rc | project mix = reinforced concrete | outside | pooled N,L,S,W,LS,LW,SW,LSW | 50 | +0.69 [+0.09, +1.28] inferior / inferior | -0.27 [-0.63, +0.11] non-inferior only / equivalent | -0.24 [-0.94, +0.45] direction neither |
| mix_steel | project mix = steel | outside | N | 50 | -1.42 [-3.14, +0.21] non-inferior only / non-inferior only | -1.05 [-2.79, +0.52] inconclusive / non-inferior only | -3.51 [-4.23, -2.74] ahead |
| mix_steel | project mix = steel | outside | L | 50 | -0.24 [-1.16, +0.62] inconclusive / non-inferior only | +0.43 [-0.76, +1.56] inconclusive / inconclusive | -3.78 [-4.85, -2.67] ahead |
| mix_steel | project mix = steel | outside | S | 50 | -0.30 [-0.80, +0.23] non-inferior only / equivalent | -0.03 [-0.51, +0.49] non-inferior only / equivalent | -1.62 [-2.51, -0.68] ahead |
| mix_steel | project mix = steel | outside | W | 50 | -1.76 [-2.97, -0.52] non-inferior only / non-inferior only | -1.51 [-2.58, -0.41] non-inferior only / non-inferior only | -4.58 [-5.31, -3.88] ahead |
| mix_steel | project mix = steel | outside | LS | 50 | -0.06 [-0.75, +0.65] inconclusive / equivalent | +0.06 [-0.51, +0.62] inconclusive / equivalent | -2.66 [-3.52, -1.83] ahead |
| mix_steel | project mix = steel | outside | LW | 50 | +1.08 [-0.24, +2.46] inconclusive / inconclusive | +1.59 [+0.57, +2.63] inferior / inferior | -4.19 [-5.40, -2.96] ahead |
| mix_steel | project mix = steel | outside | SW | 50 | -0.58 [-1.16, +0.02] non-inferior only / non-inferior only | -0.40 [-1.02, +0.20] non-inferior only / non-inferior only | -2.45 [-3.32, -1.55] ahead |
| mix_steel | project mix = steel | outside | LSW | 50 | +0.26 [-0.62, +1.08] inconclusive / inconclusive | +0.45 [-0.34, +1.19] inconclusive / inconclusive | -2.87 [-3.79, -1.98] ahead |
| mix_steel | project mix = steel | outside | pooled N,L,S,W,LS,LW,SW,LSW | 50 | -0.38 [-0.94, +0.18] non-inferior only / equivalent | -0.06 [-0.50, +0.40] non-inferior only / equivalent | -3.21 [-3.70, -2.73] direction ahead |

## Verdict counts over the reached combinations

| level | reached | policy - specialist at 0.5 (eq/ni/inf/inc) | at 1 | control - specialist at 0.5 | at 1 | policy - best rule (ahead/behind/nd) |
|---|---|---|---|---|---|---|
| default | 8 | 2/3/1/2 | 7/0/0/1 | 2/3/1/2 | 7/1/0/0 | 7/0/1 |
| dur_cv10 | 8 | 2/3/0/3 | 6/1/0/1 | 1/3/1/3 | 7/1/0/0 | 7/0/1 |
| dur_cv20 | 8 | 2/2/0/4 | 6/1/0/1 | 3/2/0/3 | 7/1/0/0 | 8/0/0 |
| dur_cv30 | 8 | 1/3/0/4 | 6/1/0/1 | 3/3/0/2 | 5/2/0/1 | 5/0/3 |
| dur_long_m25 | 8 | 2/3/1/2 | 7/0/1/0 | 1/4/1/2 | 6/1/1/0 | 6/0/2 |
| dur_long_p25 | 8 | 2/3/1/2 | 6/0/1/1 | 4/2/0/2 | 6/0/0/2 | 7/0/1 |
| lag_x050 | 4 | 1/3/0/0 | 2/2/0/0 | 1/3/0/0 | 1/3/0/0 | 4/0/0 |
| lag_x075 | 4 | 0/3/0/1 | 4/0/0/0 | 1/2/0/1 | 4/0/0/0 | 4/0/0 |
| lag_x125 | 4 | 1/1/2/0 | 4/0/0/0 | 1/1/1/1 | 4/0/0/0 | 3/0/1 |
| lag_x150 | 4 | 4/0/0/0 | 4/0/0/0 | 3/0/0/1 | 4/0/0/0 | 0/0/4 |
| setup_x050 | 4 | 1/2/0/1 | 2/2/0/0 | 1/2/0/1 | 2/2/0/0 | 4/0/0 |
| setup_x075 | 4 | 2/2/0/0 | 4/0/0/0 | 1/2/0/1 | 3/1/0/0 | 4/0/0 |
| setup_x125 | 4 | 2/2/0/0 | 4/0/0/0 | 2/1/0/1 | 4/0/0/0 | 4/0/0 |
| setup_x150 | 4 | 1/2/0/1 | 4/0/0/0 | 1/3/0/0 | 4/0/0/0 | 3/0/1 |
| setup_same0 | 4 | 0/0/0/4 | 4/0/0/0 | 0/1/0/3 | 4/0/0/0 | 2/0/2 |
| setup_sub_strict | 4 | 0/4/0/0 | 2/2/0/0 | 1/2/0/1 | 2/2/0/0 | 3/0/1 |
| setup_sub_wide | 4 | 2/2/0/0 | 3/1/0/0 | 1/2/0/1 | 3/1/0/0 | 4/0/0 |
| win_x050 | 4 | 1/1/1/1 | 4/0/0/0 | 0/2/1/1 | 3/1/0/0 | 4/0/0 |
| win_x150 | 4 | 0/1/0/3 | 3/0/0/1 | 0/1/1/2 | 3/0/0/1 | 4/0/0 |
| win_x200 | 4 | 0/1/0/3 | 2/0/0/2 | 0/1/1/2 | 3/0/1/0 | 4/0/0 |
| tight15 | 8 | 0/5/0/3 | 3/4/0/1 | 0/4/0/4 | 4/3/0/1 | 8/0/0 |
| mix_rc | 8 | 1/1/1/5 | 3/1/1/3 | 0/4/0/4 | 2/4/0/2 | 2/0/6 |
| mix_steel | 8 | 0/4/0/4 | 2/4/0/2 | 0/3/1/4 | 2/3/1/2 | 8/0/0 |

## Verdicts that differ from the default level (reached combinations)

- dur_cv10: pol_spec|v0.5: LW; pol_spec|v1.0: N,W,SW; ctl_spec|v0.5: N,LSW; ctl_spec|v1.0: W,SW
- dur_cv20: pol_spec|v0.5: L,LW,LSW; pol_spec|v1.0: N,W,SW; ctl_spec|v0.5: N,L,LW; pol_rule|verdict: S
- dur_cv30: pol_spec|v0.5: L,LW; pol_spec|v1.0: N,W; ctl_spec|v0.5: N,L,W,LS,LW; ctl_spec|v1.0: N,SW; pol_rule|verdict: L,SW
- dur_long_m25: pol_spec|v0.5: L,W,SW; pol_spec|v1.0: W,LW; ctl_spec|v0.5: N,W,LSW; ctl_spec|v1.0: N,W,LW; pol_rule|verdict: SW
- dur_long_p25: pol_spec|v0.5: N,LS,LW,LSW; pol_spec|v1.0: N; ctl_spec|v0.5: N,L,LW; ctl_spec|v1.0: N,W; pol_rule|verdict: S,SW
- lag_x050: pol_spec|v0.5: L,LW; pol_spec|v1.0: L,LW; ctl_spec|v0.5: L,LW,LSW; ctl_spec|v1.0: L,LW,LSW
- lag_x075: pol_spec|v0.5: L,LS,LW; ctl_spec|v0.5: LW,LSW
- lag_x125: pol_spec|v0.5: L; ctl_spec|v0.5: L,LS,LSW; pol_rule|verdict: LSW
- lag_x150: pol_spec|v0.5: LW,LSW; ctl_spec|v0.5: L,LW; pol_rule|verdict: L,LS,LW,LSW
- setup_x050: pol_spec|v0.5: LSW; pol_spec|v1.0: S,SW; ctl_spec|v0.5: LSW; ctl_spec|v1.0: S,SW; pol_rule|verdict: S
- setup_x075: pol_spec|v0.5: LSW; ctl_spec|v0.5: LS; ctl_spec|v1.0: SW; pol_rule|verdict: S
- setup_x125: pol_spec|v0.5: LSW; ctl_spec|v0.5: S,LSW; pol_rule|verdict: S
- setup_x150: pol_spec|v0.5: SW; ctl_spec|v0.5: LSW
- setup_same0: pol_spec|v0.5: S,LS,SW,LSW; ctl_spec|v0.5: S,LS,SW,LSW; pol_rule|verdict: SW
- setup_sub_strict: pol_spec|v0.5: LS; pol_spec|v1.0: S,SW; ctl_spec|v0.5: LSW; ctl_spec|v1.0: S,SW; pol_rule|verdict: S,LS
- setup_sub_wide: pol_spec|v0.5: LSW; pol_spec|v1.0: SW; ctl_spec|v0.5: LSW; ctl_spec|v1.0: SW; pol_rule|verdict: S
- win_x050: pol_spec|v0.5: LSW; pol_spec|v1.0: W; ctl_spec|v0.5: W,LSW
- win_x150: pol_spec|v0.5: LW,LSW; ctl_spec|v0.5: LSW; ctl_spec|v1.0: W
- win_x200: pol_spec|v0.5: LW,LSW; pol_spec|v1.0: LW; ctl_spec|v0.5: LSW; ctl_spec|v1.0: W,LW
- tight15: pol_spec|v0.5: N,L,LS,LW,LSW; pol_spec|v1.0: N,L,W,LW,SW,LSW; ctl_spec|v0.5: L,LS,LW,LSW; ctl_spec|v1.0: N,W,LW,SW,LSW; pol_rule|verdict: S
- mix_rc: not computed: the level uses other base instances than the default level, so its verdicts are not on the same instances
- mix_steel: not computed: the level uses other base instances than the default level, so its verdicts are not on the same instances

## Measured variables (%, reached combinations)

| level | combination | policy: single-pair decisions [range over runs] | policy: setup share [range] | best rule: single-pair decisions | best rule: setup share |
|---|---|---|---|---|---|
| default | N | 13.9 [13.4, 14.6] | 0.0 [0.0, 0.0] | 13.1 | 0.0 |
| default | L | 13.7 [13.2, 14.3] | 0.0 [0.0, 0.0] | 12.6 | 0.0 |
| default | S | 34.3 [32.9, 34.9] | 8.7 [8.7, 8.8] | 31.7 | 8.7 |
| default | W | 17.0 [16.4, 17.2] | 0.0 [0.0, 0.0] | 15.9 | 0.0 |
| default | LS | 38.2 [37.5, 38.6] | 6.3 [6.3, 6.4] | 36.3 | 6.3 |
| default | LW | 19.7 [19.1, 20.2] | 0.0 [0.0, 0.0] | 18.4 | 0.0 |
| default | SW | 37.4 [36.7, 37.8] | 7.9 [7.9, 7.9] | 33.9 | 7.9 |
| default | LSW | 41.3 [40.7, 41.7] | 6.0 [6.0, 6.1] | 39.3 | 6.0 |
| dur_cv10 | N | 13.8 [13.5, 14.1] | 0.0 [0.0, 0.0] | 12.9 | 0.0 |
| dur_cv10 | L | 13.2 [12.6, 13.8] | 0.0 [0.0, 0.0] | 11.8 | 0.0 |
| dur_cv10 | S | 34.3 [32.9, 34.9] | 8.8 [8.7, 8.8] | 30.8 | 8.8 |
| dur_cv10 | W | 16.9 [16.5, 17.4] | 0.0 [0.0, 0.0] | 16.2 | 0.0 |
| dur_cv10 | LS | 38.1 [37.7, 38.6] | 6.3 [6.3, 6.4] | 36.3 | 6.3 |
| dur_cv10 | LW | 19.5 [19.2, 19.8] | 0.0 [0.0, 0.0] | 17.5 | 0.0 |
| dur_cv10 | SW | 37.4 [36.0, 38.0] | 8.0 [7.9, 8.0] | 34.7 | 7.9 |
| dur_cv10 | LSW | 41.2 [40.7, 41.7] | 6.1 [6.0, 6.1] | 38.9 | 6.0 |
| dur_cv20 | N | 13.1 [12.5, 13.5] | 0.0 [0.0, 0.0] | 12.6 | 0.0 |
| dur_cv20 | L | 12.6 [12.3, 12.9] | 0.0 [0.0, 0.0] | 11.1 | 0.0 |
| dur_cv20 | S | 34.1 [32.7, 35.1] | 8.9 [8.9, 8.9] | 30.9 | 8.9 |
| dur_cv20 | W | 16.6 [16.2, 16.7] | 0.0 [0.0, 0.0] | 15.9 | 0.0 |
| dur_cv20 | LS | 37.8 [37.1, 38.1] | 6.4 [6.4, 6.4] | 35.2 | 6.4 |
| dur_cv20 | LW | 19.1 [18.8, 19.5] | 0.0 [0.0, 0.0] | 17.7 | 0.0 |
| dur_cv20 | SW | 37.3 [36.8, 38.0] | 8.1 [8.1, 8.1] | 34.5 | 8.0 |
| dur_cv20 | LSW | 41.4 [41.2, 41.6] | 6.1 [6.1, 6.2] | 39.4 | 6.0 |
| dur_cv30 | N | 12.3 [12.2, 12.5] | 0.0 [0.0, 0.0] | 12.1 | 0.0 |
| dur_cv30 | L | 11.9 [11.5, 12.0] | 0.0 [0.0, 0.0] | 10.5 | 0.0 |
| dur_cv30 | S | 33.6 [32.5, 34.4] | 9.0 [9.0, 9.1] | 31.0 | 9.1 |
| dur_cv30 | W | 16.0 [15.7, 16.3] | 0.0 [0.0, 0.0] | 15.1 | 0.0 |
| dur_cv30 | LS | 37.6 [37.0, 38.0] | 6.5 [6.5, 6.5] | 35.7 | 6.5 |
| dur_cv30 | LW | 18.6 [18.4, 18.9] | 0.0 [0.0, 0.0] | 17.0 | 0.0 |
| dur_cv30 | SW | 37.1 [36.5, 37.7] | 8.2 [8.2, 8.3] | 34.9 | 8.2 |
| dur_cv30 | LSW | 41.2 [40.9, 41.5] | 6.2 [6.2, 6.2] | 39.1 | 6.2 |
| dur_long_m25 | N | 14.0 [13.4, 14.4] | 0.0 [0.0, 0.0] | 12.4 | 0.0 |
| dur_long_m25 | L | 12.0 [11.5, 12.5] | 0.0 [0.0, 0.0] | 11.0 | 0.0 |
| dur_long_m25 | S | 33.4 [32.0, 34.2] | 9.3 [9.3, 9.4] | 30.9 | 9.4 |
| dur_long_m25 | W | 16.6 [16.4, 16.8] | 0.0 [0.0, 0.0] | 15.4 | 0.0 |
| dur_long_m25 | LS | 37.2 [36.3, 37.6] | 6.5 [6.5, 6.6] | 35.0 | 6.5 |
| dur_long_m25 | LW | 18.2 [17.7, 18.5] | 0.0 [0.0, 0.0] | 16.5 | 0.0 |
| dur_long_m25 | SW | 36.5 [35.8, 37.1] | 8.4 [8.4, 8.5] | 33.6 | 8.5 |
| dur_long_m25 | LSW | 40.8 [40.3, 41.3] | 6.3 [6.3, 6.3] | 38.4 | 6.2 |
| dur_long_p25 | N | 13.6 [13.4, 13.8] | 0.0 [0.0, 0.0] | 12.8 | 0.0 |
| dur_long_p25 | L | 15.1 [14.4, 15.7] | 0.0 [0.0, 0.0] | 13.9 | 0.0 |
| dur_long_p25 | S | 35.1 [33.6, 35.6] | 8.0 [8.0, 8.1] | 32.1 | 8.0 |
| dur_long_p25 | W | 17.3 [16.8, 17.5] | 0.0 [0.0, 0.0] | 16.5 | 0.0 |
| dur_long_p25 | LS | 38.8 [38.4, 39.3] | 6.0 [6.0, 6.0] | 36.3 | 6.0 |
| dur_long_p25 | LW | 20.0 [19.7, 20.4] | 0.0 [0.0, 0.0] | 19.3 | 0.0 |
| dur_long_p25 | SW | 38.4 [37.7, 38.8] | 7.2 [7.2, 7.3] | 35.2 | 7.3 |
| dur_long_p25 | LSW | 41.8 [41.5, 42.2] | 5.7 [5.7, 5.7] | 39.7 | 5.7 |
| lag_x050 | L | 14.9 [14.4, 15.4] | 0.0 [0.0, 0.0] | 14.5 | 0.0 |
| lag_x050 | LS | 36.7 [36.1, 37.2] | 7.3 [7.3, 7.4] | 34.5 | 7.2 |
| lag_x050 | LW | 18.5 [18.3, 18.7] | 0.0 [0.0, 0.0] | 18.2 | 0.0 |
| lag_x050 | LSW | 39.7 [39.3, 40.2] | 6.9 [6.9, 6.9] | 37.4 | 6.7 |
| lag_x075 | L | 15.1 [14.7, 15.4] | 0.0 [0.0, 0.0] | 14.1 | 0.0 |
| lag_x075 | LS | 37.6 [37.1, 37.9] | 6.8 [6.8, 6.9] | 35.3 | 6.8 |
| lag_x075 | LW | 19.5 [19.0, 20.1] | 0.0 [0.0, 0.0] | 18.7 | 0.0 |
| lag_x075 | LSW | 40.4 [39.9, 40.9] | 6.5 [6.4, 6.5] | 38.4 | 6.4 |
| lag_x125 | L | 11.9 [11.6, 12.2] | 0.0 [0.0, 0.0] | 10.8 | 0.0 |
| lag_x125 | LS | 38.3 [37.5, 38.7] | 5.8 [5.7, 5.8] | 36.2 | 5.8 |
| lag_x125 | LW | 18.3 [18.1, 18.6] | 0.0 [0.0, 0.0] | 17.7 | 0.0 |
| lag_x125 | LSW | 41.6 [41.4, 41.9] | 5.6 [5.5, 5.6] | 39.1 | 5.6 |
| lag_x150 | L | 10.6 [10.2, 10.8] | 0.0 [0.0, 0.0] | 9.6 | 0.0 |
| lag_x150 | LS | 38.4 [37.8, 38.9] | 5.3 [5.2, 5.3] | 36.7 | 5.3 |
| lag_x150 | LW | 17.1 [16.6, 17.3] | 0.0 [0.0, 0.0] | 16.0 | 0.0 |
| lag_x150 | LSW | 42.0 [41.3, 42.2] | 5.2 [5.2, 5.2] | 39.9 | 5.2 |
| setup_x050 | S | 27.1 [25.8, 27.6] | 5.9 [5.8, 5.9] | 24.4 | 5.8 |
| setup_x050 | LS | 29.6 [28.9, 30.2] | 4.0 [4.0, 4.1] | 27.8 | 4.0 |
| setup_x050 | SW | 29.9 [29.1, 30.3] | 5.3 [5.3, 5.3] | 27.1 | 5.2 |
| setup_x050 | LSW | 33.2 [33.0, 33.7] | 3.9 [3.9, 3.9] | 31.4 | 3.8 |
| setup_x075 | S | 31.7 [30.2, 32.4] | 6.9 [6.8, 6.9] | 28.7 | 6.8 |
| setup_x075 | LS | 35.5 [34.9, 36.3] | 4.8 [4.8, 4.9] | 33.7 | 4.8 |
| setup_x075 | SW | 35.0 [33.9, 35.9] | 6.2 [6.2, 6.2] | 32.0 | 6.2 |
| setup_x075 | LSW | 39.0 [38.5, 39.4] | 4.6 [4.6, 4.7] | 36.8 | 4.6 |
| setup_x125 | S | 36.2 [34.8, 37.0] | 10.1 [10.1, 10.1] | 32.5 | 10.1 |
| setup_x125 | LS | 39.4 [38.6, 39.7] | 7.4 [7.4, 7.5] | 36.9 | 7.5 |
| setup_x125 | SW | 38.7 [37.4, 39.5] | 9.2 [9.1, 9.2] | 35.7 | 9.1 |
| setup_x125 | LSW | 42.8 [42.5, 43.4] | 7.1 [7.1, 7.1] | 40.9 | 7.0 |
| setup_x150 | S | 37.2 [35.9, 38.0] | 11.2 [11.1, 11.3] | 34.1 | 11.3 |
| setup_x150 | LS | 40.5 [39.8, 40.8] | 8.5 [8.4, 8.5] | 38.5 | 8.4 |
| setup_x150 | SW | 40.4 [39.1, 41.2] | 10.2 [10.1, 10.3] | 37.4 | 10.2 |
| setup_x150 | LSW | 43.9 [43.4, 44.4] | 8.1 [8.0, 8.1] | 41.9 | 8.0 |
| setup_same0 | S | 30.8 [29.9, 31.9] | 3.7 [3.5, 3.8] | 28.4 | 3.7 |
| setup_same0 | LS | 38.8 [37.8, 39.6] | 2.4 [2.3, 2.5] | 37.1 | 2.4 |
| setup_same0 | SW | 33.7 [33.0, 34.7] | 3.5 [3.5, 3.5] | 31.3 | 3.5 |
| setup_same0 | LSW | 41.3 [40.8, 42.0] | 2.5 [2.5, 2.5] | 39.9 | 2.5 |
| setup_sub_strict | S | 16.6 [16.3, 17.2] | 2.2 [2.2, 2.2] | 15.0 | 2.1 |
| setup_sub_strict | LS | 16.5 [16.2, 16.9] | 1.4 [1.4, 1.4] | 14.4 | 1.4 |
| setup_sub_strict | SW | 20.2 [19.9, 20.7] | 2.0 [1.9, 2.0] | 18.6 | 1.9 |
| setup_sub_strict | LSW | 21.5 [21.0, 22.0] | 1.4 [1.3, 1.4] | 19.5 | 1.3 |
| setup_sub_wide | S | 29.2 [28.3, 30.0] | 7.0 [7.0, 7.0] | 26.9 | 7.0 |
| setup_sub_wide | LS | 31.8 [31.3, 32.3] | 5.0 [5.0, 5.0] | 29.8 | 5.0 |
| setup_sub_wide | SW | 32.5 [32.0, 32.7] | 6.3 [6.3, 6.3] | 29.9 | 6.3 |
| setup_sub_wide | LSW | 35.1 [34.7, 35.7] | 4.8 [4.7, 4.8] | 33.2 | 4.7 |
| win_x050 | W | 15.4 [15.0, 15.7] | 0.0 [0.0, 0.0] | 14.3 | 0.0 |
| win_x050 | LW | 16.8 [16.4, 17.2] | 0.0 [0.0, 0.0] | 15.3 | 0.0 |
| win_x050 | SW | 35.7 [34.6, 36.3] | 8.3 [8.2, 8.3] | 33.0 | 8.2 |
| win_x050 | LSW | 39.6 [39.0, 39.9] | 6.2 [6.2, 6.2] | 37.7 | 6.1 |
| win_x150 | W | 18.8 [18.3, 19.2] | 0.0 [0.0, 0.0] | 17.3 | 0.0 |
| win_x150 | LW | 21.3 [20.9, 21.6] | 0.0 [0.0, 0.0] | 19.9 | 0.0 |
| win_x150 | SW | 39.1 [38.5, 39.7] | 7.5 [7.5, 7.5] | 36.5 | 7.5 |
| win_x150 | LSW | 42.6 [42.1, 42.9] | 5.8 [5.8, 5.9] | 40.1 | 5.8 |
| win_x200 | W | 20.6 [20.1, 21.0] | 0.0 [0.0, 0.0] | 19.6 | 0.0 |
| win_x200 | LW | 23.6 [23.2, 23.9] | 0.0 [0.0, 0.0] | 21.7 | 0.0 |
| win_x200 | SW | 40.5 [39.9, 41.3] | 7.1 [7.1, 7.1] | 38.4 | 7.1 |
| win_x200 | LSW | 44.4 [44.0, 45.2] | 5.6 [5.6, 5.7] | 41.6 | 5.6 |
| tight15 | N | 22.7 [21.3, 24.0] | 0.0 [0.0, 0.0] | 21.0 | 0.0 |
| tight15 | L | 23.0 [22.8, 23.3] | 0.0 [0.0, 0.0] | 22.6 | 0.0 |
| tight15 | S | 43.6 [42.0, 44.7] | 10.7 [10.7, 10.8] | 39.5 | 10.5 |
| tight15 | W | 26.9 [26.1, 27.9] | 0.0 [0.0, 0.0] | 24.9 | 0.0 |
| tight15 | LS | 47.5 [46.9, 48.2] | 8.6 [8.5, 8.6] | 45.0 | 8.3 |
| tight15 | LW | 27.7 [27.3, 28.4] | 0.0 [0.0, 0.0] | 25.5 | 0.0 |
| tight15 | SW | 46.2 [45.1, 47.3] | 9.7 [9.6, 9.8] | 42.8 | 9.5 |
| tight15 | LSW | 50.2 [49.6, 50.6] | 8.0 [8.0, 8.0] | 47.2 | 7.8 |
| mix_rc | N | 13.0 [12.3, 13.6] | 0.0 [0.0, 0.0] | 13.4 | 0.0 |
| mix_rc | L | 14.3 [13.4, 14.8] | 0.0 [0.0, 0.0] | 12.3 | 0.0 |
| mix_rc | S | 25.2 [24.6, 25.8] | 8.6 [8.5, 8.7] | 25.3 | 8.6 |
| mix_rc | W | 16.5 [15.2, 17.0] | 0.0 [0.0, 0.0] | 16.5 | 0.0 |
| mix_rc | LS | 29.9 [29.3, 30.3] | 6.0 [5.9, 6.0] | 29.4 | 5.9 |
| mix_rc | LW | 19.7 [19.2, 20.2] | 0.0 [0.0, 0.0] | 17.9 | 0.0 |
| mix_rc | SW | 29.3 [28.2, 30.2] | 7.8 [7.7, 7.9] | 28.9 | 7.7 |
| mix_rc | LSW | 34.0 [33.5, 34.4] | 5.7 [5.7, 5.7] | 32.8 | 5.7 |
| mix_steel | N | 10.1 [8.3, 12.0] | 0.0 [0.0, 0.0] | 10.4 | 0.0 |
| mix_steel | L | 10.5 [10.1, 10.8] | 0.0 [0.0, 0.0] | 9.4 | 0.0 |
| mix_steel | S | 21.8 [21.0, 23.1] | 8.2 [8.2, 8.2] | 22.0 | 8.1 |
| mix_steel | W | 16.2 [15.6, 16.6] | 0.0 [0.0, 0.0] | 15.2 | 0.0 |
| mix_steel | LS | 24.6 [24.2, 25.1] | 6.6 [6.5, 6.6] | 23.5 | 6.5 |
| mix_steel | LW | 15.5 [15.0, 16.3] | 0.0 [0.0, 0.0] | 14.0 | 0.0 |
| mix_steel | SW | 26.2 [25.5, 26.8] | 7.4 [7.4, 7.5] | 23.9 | 7.3 |
| mix_steel | LSW | 27.9 [27.4, 28.1] | 6.1 [6.1, 6.2] | 27.1 | 6.0 |
