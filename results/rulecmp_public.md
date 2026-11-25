# Comparison with dispatching rules, public family

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 100; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -14.18 [-15.76, -12.54] | 1.6e-05, 0.0004 | ahead | 93/0/7 | -15.08 [-16.75, -13.38] | 9.3e-06, 0.0004 | ahead | 94/0/6 | -15.64 [-17.18, -14.05] | 4.6e-06, 0.0004 | ahead | 95/0/5 |
| L | -0.33 [-1.32, +0.65] | 0.23, 1 | not distinguishable | 45/9/46 | +0.02 [-0.97, +0.97] | 1, 1 | not distinguishable | 43/8/49 | -0.32 [-1.31, +0.64] | 0.23, 1 | not distinguishable | 47/12/41 |
| S | +5.56 [+3.91, +7.27] | 1, 1 | behind | 21/10/69 | +5.07 [+3.46, +6.72] | 1, 1 | behind | 24/9/67 | +5.86 [+4.37, +7.33] | 1, 1 | behind | 22/9/70 |
| W | -9.49 [-11.29, -7.74] | 6.6e-07, 0.0004 | ahead | 80/8/13 | -9.92 [-11.63, -8.16] | 4e-05, 0.0004 | ahead | 83/8/9 | -9.80 [-11.46, -8.16] | 7.7e-06, 0.0004 | ahead | 83/7/11 |
| LS | +4.04 [+2.77, +5.43] | 1, 1 | behind | 19/16/64 | +4.73 [+3.35, +6.25] | 1, 1 | behind | 18/17/65 | +4.90 [+3.47, +6.46] | 1, 1 | behind | 19/15/65 |
| LW | -1.56 [-2.79, -0.32] | 0.00024, 0.048 | ahead | 52/15/34 | -1.19 [-2.39, +0.03] | 0.0013, 0.17 | not distinguishable | 48/14/38 | -1.21 [-2.47, +0.05] | 0.00055, 0.18 | not distinguishable | 47/18/35 |
| SW | +4.15 [+2.74, +5.62] | 1, 1 | behind | 22/10/68 | +4.37 [+2.99, +5.74] | 1, 1 | behind | 22/9/69 | +4.47 [+3.16, +5.85] | 1, 1 | behind | 22/9/68 |
| LSW | +4.00 [+2.29, +5.79] | 1, 1 | behind | 24/13/63 | +4.01 [+2.36, +5.73] | 1, 1 | behind | 25/11/64 | +4.08 [+2.36, +5.92] | 1, 1 | behind | 27/8/66 |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -18.32 [-20.30, -16.31] | 4.4e-06, 0.0004 | ahead | 95/0/5 | -19.24 [-21.16, -17.29] | 2.7e-06, 0.0004 | ahead | 96/0/4 | -19.73 [-21.61, -17.79] | 1.3e-06, 0.0004 | ahead | 96/0/4 |
| L | -3.59 [-5.10, -2.13] | 4.5e-05, 0.0004 | ahead | 63/6/31 | -3.26 [-4.74, -1.82] | 2.5e-05, 0.0004 | ahead | 59/7/34 | -3.56 [-5.11, -2.12] | 3.6e-05, 0.0004 | ahead | 65/8/27 |
| S | -1.04 [-2.93, +0.86] | 0.016, 0.14 | not distinguishable | 49/9/42 | -1.50 [-3.33, +0.34] | 0.0023, 0.056 | not distinguishable | 52/8/40 | -0.72 [-2.55, +1.13] | 0.012, 0.22 | not distinguishable | 53/6/40 |
| W | -16.82 [-18.93, -14.75] | 5.8e-08, 0.0004 | ahead | 92/4/4 | -17.19 [-19.28, -15.09] | 2.7e-06, 0.0004 | ahead | 93/4/2 | -17.07 [-19.15, -15.00] | 7.2e-07, 0.0004 | ahead | 93/3/4 |
| LS | -2.50 [-4.16, -0.88] | 0.00085, 0.0042 | ahead | 56/16/28 | -1.91 [-3.53, -0.30] | 0.0017, 0.025 | ahead | 53/19/28 | -1.79 [-3.36, -0.30] | 0.0023, 0.026 | ahead | 50/21/29 |
| LW | -6.76 [-8.54, -5.02] | 1.1e-06, 0.0004 | ahead | 72/9/19 | -6.42 [-8.17, -4.72] | 2.1e-06, 0.0004 | ahead | 71/7/22 | -6.44 [-8.21, -4.72] | 1e-06, 0.0004 | ahead | 70/11/19 |
| SW | -2.60 [-4.12, -1.08] | 0.00087, 0.003 | ahead | 57/10/33 | -2.38 [-3.93, -0.84] | 2.4e-05, 0.004 | ahead | 56/10/34 | -2.23 [-3.83, -0.53] | 0.0022, 0.023 | ahead | 57/8/34 |
| LSW | -2.17 [-3.98, -0.33] | 0.0051, 0.021 | ahead | 50/16/34 | -2.18 [-3.95, -0.40] | 0.00024, 0.025 | ahead | 52/16/33 | -2.13 [-3.93, -0.24] | 0.0026, 0.027 | ahead | 51/13/36 |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -18.44 [-19.84, -17.01] | 8.9e-07, 0.0004 | ahead | 98/0/1 | -19.03 [-20.44, -17.58] | 4.4e-07, 0.0004 | ahead | 98/0/2 | -19.73 [-21.09, -18.35] | 1.7e-07, 0.0004 | ahead | 99/0/1 |
| L | -2.20 [-3.13, -1.29] | 2.8e-05, 0.0004 | ahead | 60/12/29 | -1.99 [-2.93, -1.09] | 0.00021, 0.0004 | ahead | 58/12/31 | -2.25 [-3.23, -1.30] | 0.0018, 0.0004 | ahead | 62/13/25 |
| S | +0.79 [-0.59, +2.17] | 1, 1 | not distinguishable | 44/8/48 | +0.74 [-0.64, +2.21] | 1, 1 | not distinguishable | 45/7/48 | +0.17 [-1.14, +1.51] | 0.66, 0.85 | not distinguishable | 49/6/44 |
| W | -13.94 [-15.59, -12.34] | 8.9e-07, 0.0004 | ahead | 92/5/3 | -14.26 [-15.93, -12.61] | 2.5e-06, 0.0004 | ahead | 92/5/3 | -14.33 [-15.84, -12.79] | 1e-07, 0.0004 | ahead | 93/5/2 |
| LS | +1.18 [-0.01, +2.45] | 1, 1 | not distinguishable | 40/13/47 | +1.68 [+0.56, +2.89] | 1, 1 | behind | 37/13/50 | -0.11 [-1.30, +1.12] | 0.66, 0.85 | not distinguishable | 49/8/43 |
| LW | -3.38 [-4.51, -2.24] | 7.7e-06, 0.0004 | ahead | 63/16/21 | -3.12 [-4.25, -1.98] | 2.4e-05, 0.0004 | ahead | 62/16/22 | -3.37 [-4.46, -2.28] | 2.5e-06, 0.0004 | ahead | 63/17/19 |
| SW | +0.21 [-1.00, +1.44] | 1, 1 | not distinguishable | 41/12/47 | +0.30 [-0.94, +1.55] | 1, 1 | not distinguishable | 41/12/47 | -0.60 [-1.68, +0.50] | 0.098, 0.41 | not distinguishable | 46/11/43 |
| LSW | -0.07 [-1.57, +1.46] | 1, 1 | not distinguishable | 43/13/44 | +0.28 [-1.19, +1.77] | 1, 1 | not distinguishable | 42/11/46 | -1.48 [-2.78, -0.18] | 0.00033, 0.054 | not distinguishable | 56/6/38 |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -19.95 [-21.28, -18.58] | 7.4e-07, 0.0004 | ahead | 100/0/0 | -20.31 [-21.61, -19.01] | 1.8e-07, 0.0004 | ahead | 99/0/1 | -21.20 [-22.47, -19.89] | 3e-07, 0.0004 | ahead | 100/0/0 |
| L | -2.50 [-3.32, -1.70] | 3e-05, 0.0004 | ahead | 62/17/21 | -2.30 [-3.13, -1.48] | 0.00015, 0.0004 | ahead | 60/17/23 | -2.77 [-3.60, -1.98] | 8.5e-05, 0.0004 | ahead | 65/18/16 |
| S | -0.24 [-1.58, +1.10] | 0.97, 1 | not distinguishable | 52/7/41 | -0.22 [-1.76, +1.37] | 1, 1 | not distinguishable | 52/6/42 | -1.64 [-2.99, -0.39] | 0.019, 0.0069 | ahead | 61/7/32 |
| W | -14.35 [-15.93, -12.75] | 4.4e-08, 0.0004 | ahead | 91/7/2 | -14.47 [-16.10, -12.82] | 1.4e-06, 0.0004 | ahead | 90/8/2 | -14.74 [-16.25, -13.18] | 2.1e-07, 0.0004 | ahead | 92/7/1 |
| LS | +0.38 [-0.90, +1.66] | 0.99, 1 | not distinguishable | 44/10/46 | +1.04 [-0.20, +2.30] | 1, 1 | not distinguishable | 40/10/50 | -2.02 [-3.44, -0.58] | 0.016, 0.0069 | ahead | 60/7/33 |
| LW | -3.24 [-4.31, -2.18] | 2e-05, 0.0004 | ahead | 62/20/18 | -2.83 [-3.93, -1.71] | 6.5e-05, 0.0004 | ahead | 61/20/20 | -3.37 [-4.44, -2.29] | 1.5e-05, 0.0004 | ahead | 63/22/15 |
| SW | -0.48 [-1.78, +0.84] | 0.7, 0.92 | not distinguishable | 47/14/39 | -0.43 [-1.70, +0.81] | 0.72, 0.99 | not distinguishable | 47/14/39 | -2.23 [-3.48, -1.06] | 0.013, 0.00045 | ahead | 60/12/29 |
| LSW | -0.01 [-1.44, +1.47] | 0.99, 1 | not distinguishable | 43/14/43 | +0.45 [-0.91, +1.79] | 1, 1 | not distinguishable | 40/14/46 | -2.50 [-3.68, -1.30] | 0.00055, 0.0004 | ahead | 63/10/27 |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -18.47 [-19.89, -17.02] | 1.5e-06, 0.0004 | ahead | 98/0/1 | -19.10 [-20.53, -17.65] | 4.1e-07, 0.0004 | ahead | 98/0/2 | -19.76 [-21.12, -18.37] | 1.9e-07, 0.0004 | ahead | 99/0/1 |
| L | -2.16 [-3.09, -1.27] | 1.5e-05, 0.0004 | ahead | 59/11/29 | -1.88 [-2.82, -0.97] | 0.00027, 0.0004 | ahead | 57/12/32 | -2.30 [-3.23, -1.41] | 0.00028, 0.0004 | ahead | 63/14/23 |
| S | +1.21 [-0.23, +2.67] | 1, 1 | not distinguishable | 41/8/51 | +0.98 [-0.49, +2.55] | 1, 1 | not distinguishable | 44/7/49 | +0.51 [-0.90, +1.91] | 1, 1 | not distinguishable | 46/7/46 |
| W | -13.83 [-15.50, -12.20] | 2.2e-07, 0.0004 | ahead | 92/5/3 | -14.11 [-15.76, -12.46] | 1.6e-06, 0.0004 | ahead | 92/5/3 | -14.27 [-15.82, -12.71] | 1.8e-07, 0.0004 | ahead | 93/5/2 |
| LS | +1.38 [+0.17, +2.66] | 1, 1 | not distinguishable | 38/13/49 | +1.96 [+0.82, +3.20] | 1, 1 | behind | 35/14/51 | +0.11 [-1.09, +1.40] | 1, 1 | not distinguishable | 47/10/42 |
| LW | -3.31 [-4.47, -2.15] | 1.6e-05, 0.0004 | ahead | 63/16/21 | -2.97 [-4.10, -1.84] | 1.8e-05, 0.0004 | ahead | 61/16/23 | -3.35 [-4.45, -2.24] | 1.1e-06, 0.0004 | ahead | 63/18/19 |
| SW | +0.57 [-0.66, +1.80] | 1, 1 | not distinguishable | 39/12/49 | +0.62 [-0.66, +1.90] | 1, 1 | not distinguishable | 39/12/49 | -0.43 [-1.54, +0.73] | 0.38, 0.68 | not distinguishable | 45/11/44 |
| LSW | +0.34 [-1.19, +1.89] | 1, 1 | not distinguishable | 41/14/46 | +0.63 [-0.84, +2.14] | 1, 1 | not distinguishable | 40/12/48 | -1.34 [-2.64, -0.02] | 0.00078, 0.094 | not distinguishable | 53/7/40 |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -19.91 [-21.24, -18.56] | 7.9e-07, 0.0004 | ahead | 100/0/0 | -20.28 [-21.59, -18.97] | 2e-07, 0.0004 | ahead | 99/0/1 | -21.15 [-22.42, -19.85] | 2.6e-07, 0.0004 | ahead | 99/0/1 |
| L | -2.49 [-3.30, -1.71] | 2.3e-05, 0.0004 | ahead | 61/17/22 | -2.27 [-3.10, -1.45] | 0.00014, 0.0004 | ahead | 59/17/24 | -2.76 [-3.58, -1.98] | 4.4e-05, 0.0004 | ahead | 65/19/16 |
| S | -0.12 [-1.46, +1.22] | 1, 1 | not distinguishable | 52/7/41 | -0.14 [-1.71, +1.46] | 1, 1 | not distinguishable | 51/6/42 | -1.54 [-2.93, -0.26] | 0.027, 0.01 | ahead | 60/7/33 |
| W | -14.28 [-15.86, -12.69] | 2.9e-08, 0.0004 | ahead | 91/7/2 | -14.41 [-16.04, -12.75] | 1.4e-06, 0.0004 | ahead | 90/8/2 | -14.68 [-16.19, -13.13] | 1.5e-07, 0.0004 | ahead | 91/8/1 |
| LS | +0.47 [-0.81, +1.77] | 1, 1 | not distinguishable | 43/10/47 | +1.13 [-0.11, +2.39] | 1, 1 | not distinguishable | 40/10/50 | -1.93 [-3.35, -0.48] | 0.02, 0.01 | ahead | 59/7/33 |
| LW | -3.21 [-4.29, -2.15] | 2.4e-05, 0.0004 | ahead | 62/20/18 | -2.79 [-3.88, -1.67] | 7.2e-05, 0.0004 | ahead | 60/20/20 | -3.34 [-4.41, -2.26] | 1.4e-05, 0.0004 | ahead | 63/22/15 |
| SW | -0.39 [-1.69, +0.92] | 0.87, 1 | not distinguishable | 46/14/39 | -0.35 [-1.63, +0.88] | 0.88, 1 | not distinguishable | 47/14/40 | -2.17 [-3.42, -0.98] | 0.016, 0.0006 | ahead | 59/12/29 |
| LSW | +0.09 [-1.35, +1.56] | 1, 1 | not distinguishable | 42/14/43 | +0.53 [-0.83, +1.87] | 1, 1 | not distinguishable | 40/14/46 | -2.46 [-3.65, -1.25] | 0.00094, 0.0004 | ahead | 62/10/27 |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | SPT | -4.75 | -4.75 | 21.0 | -1.99 | 10.2 |
| L | SPT (SPT) | SPT | -3.32 | -3.32 | 13.3 | -0.93 | 8.8 |
| S | MWKR (MWKR) | MOR | -5.86 | -6.13 | 15.7 | -2.02 | 7.1 |
| W | SPT (MWKR) | MWKR | -5.83 | -7.83 | 21.3 | -3.03 | 7.8 |
| LS | SPT (FIFO) | MOR | -4.17 | -6.15 | 13.1 | -1.59 | 6.2 |
| LW | SPT (SPT) | SPT | -5.24 | -5.24 | 15.0 | -1.47 | 7.7 |
| SW | FIFO (FIFO) | MWKR | -5.64 | -6.35 | 14.5 | -1.70 | 6.5 |
| LSW | SPT (SPT) | MWKR | -5.08 | -5.77 | 12.9 | -2.54 | 6.2 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 12.3 (11.4-13.6, n=5) | 15.5 (12.8-20.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 29.9 (28.7-30.6, n=5) | 28.7 (28.1-28.9, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 65.3 (64.5-65.9, n=5) | 66.2 (65.7-66.6, n=4) | 25.4 (25.0-25.6, n=5) | 25.1 (24.9-25.2, n=4) |
| W | 18.4 (17.6-19.5, n=5) | 21.7 (20.1-25.4, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 70.4 (70.1-70.7, n=5) | 70.8 (70.2-71.4, n=4) | 20.9 (20.8-21.0, n=5) | 20.5 (20.3-20.8, n=4) |
| LW | 33.4 (33.0-34.0, n=5) | 33.5 (32.3-34.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 67.5 (67.2-67.8, n=5) | 68.0 (67.4-68.4, n=4) | 22.2 (21.9-22.4, n=5) | 21.6 (21.4-21.7, n=4) |
| LSW | 71.7 (71.4-72.2, n=5) | 72.1 (71.7-72.4, n=4) | 19.0 (18.8-19.2, n=5) | 18.6 (18.5-18.7, n=4) |
