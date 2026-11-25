# Comparison with dispatching rules, public family, test set pub_Hurink_rdata

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.79 [-0.18, +1.80] | 1, 1 | not distinguishable | 51/2/48 | - | - | - | - | - | - | - | - |
| L | -0.20 [-1.26, +0.81] | 1, 1 | not distinguishable | 45/5/50 | - | - | - | - | - | - | - | - |
| S | +5.11 [+3.83, +6.40] | 1, 1 | behind | 22/2/77 | - | - | - | - | - | - | - | - |
| W | +2.08 [+0.93, +3.28] | 1, 1 | behind | 30/8/62 | - | - | - | - | - | - | - | - |
| LS | +4.50 [+2.95, +6.16] | 1, 1 | behind | 27/3/70 | - | - | - | - | - | - | - | - |
| LW | +0.04 [-1.41, +1.44] | 1, 1 | not distinguishable | 40/18/42 | - | - | - | - | - | - | - | - |
| SW | +4.12 [+2.79, +5.57] | 1, 1 | behind | 26/2/73 | - | - | - | - | - | - | - | - |
| LSW | +4.50 [+2.86, +6.15] | 1, 1 | behind | 20/3/77 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -10.15 [-12.28, -8.00] | 3.4e-06, 0.0004 | ahead | 89/0/11 | - | - | - | - | - | - | - | - |
| L | -5.13 [-7.13, -3.24] | 6.2e-05, 0.0004 | ahead | 70/5/25 | - | - | - | - | - | - | - | - |
| S | -0.13 [-1.76, +1.47] | 0.3, 0.6 | not distinguishable | 47/2/52 | - | - | - | - | - | - | - | - |
| W | -11.05 [-13.48, -8.58] | 3.1e-06, 0.0004 | ahead | 83/4/14 | - | - | - | - | - | - | - | - |
| LS | -0.67 [-2.77, +1.44] | 0.14, 0.6 | not distinguishable | 54/2/44 | - | - | - | - | - | - | - | - |
| LW | -7.53 [-9.88, -5.31] | 2.4e-06, 0.0004 | ahead | 78/9/13 | - | - | - | - | - | - | - | - |
| SW | -1.95 [-3.62, -0.22] | 0.0081, 0.053 | not distinguishable | 58/3/39 | - | - | - | - | - | - | - | - |
| LSW | -0.86 [-2.87, +1.12] | 0.031, 0.6 | not distinguishable | 56/2/42 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -2.19 [-2.96, -1.43] | 0.0023, 0.0004 | ahead | 75/2/23 | - | - | - | - | - | - | - | - |
| L | -2.55 [-3.52, -1.67] | 5.1e-06, 0.0004 | ahead | 69/7/24 | - | - | - | - | - | - | - | - |
| S | -0.32 [-1.23, +0.60] | 0.031, 0.75 | not distinguishable | 54/3/43 | - | - | - | - | - | - | - | - |
| W | -1.05 [-2.00, -0.11] | 0.024, 0.069 | not distinguishable | 56/8/35 | - | - | - | - | - | - | - | - |
| LS | -0.31 [-1.35, +0.72] | 0.024, 0.75 | not distinguishable | 53/2/45 | - | - | - | - | - | - | - | - |
| LW | -2.18 [-3.50, -1.01] | 9.5e-05, 0.0009 | ahead | 56/21/24 | - | - | - | - | - | - | - | - |
| SW | -0.31 [-1.20, +0.59] | 0.024, 0.75 | not distinguishable | 51/3/46 | - | - | - | - | - | - | - | - |
| LSW | -0.61 [-2.03, +0.73] | 0.0021, 0.75 | not distinguishable | 47/4/49 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.71 [-2.37, -1.09] | 0.0033, 0.0004 | ahead | 75/2/24 | - | - | - | - | - | - | - | - |
| L | -1.67 [-2.51, -0.89] | 3.9e-07, 0.0004 | ahead | 59/20/21 | - | - | - | - | - | - | - | - |
| S | -0.42 [-1.09, +0.35] | 0.22, 0.5 | not distinguishable | 58/5/37 | - | - | - | - | - | - | - | - |
| W | -0.89 [-1.84, +0.19] | 0.0051, 0.24 | not distinguishable | 64/7/29 | - | - | - | - | - | - | - | - |
| LS | +0.04 [-0.83, +0.82] | 1, 1 | not distinguishable | 48/7/44 | - | - | - | - | - | - | - | - |
| LW | -1.40 [-2.35, -0.48] | 6.9e-05, 0.009 | ahead | 46/29/25 | - | - | - | - | - | - | - | - |
| SW | +0.66 [-0.20, +1.63] | 1, 1 | not distinguishable | 48/7/46 | - | - | - | - | - | - | - | - |
| LSW | +0.29 [-0.66, +1.16] | 1, 1 | not distinguishable | 44/8/48 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -2.25 [-2.99, -1.51] | 0.0015, 0.0004 | ahead | 77/2/22 | - | - | - | - | - | - | - | - |
| L | -2.43 [-3.40, -1.54] | 1.2e-05, 0.0004 | ahead | 67/7/26 | - | - | - | - | - | - | - | - |
| S | -0.15 [-1.08, +0.80] | 0.23, 1 | not distinguishable | 53/3/44 | - | - | - | - | - | - | - | - |
| W | -1.02 [-2.00, -0.09] | 0.025, 0.079 | not distinguishable | 55/9/36 | - | - | - | - | - | - | - | - |
| LS | -0.19 [-1.27, +0.87] | 0.23, 1 | not distinguishable | 53/3/44 | - | - | - | - | - | - | - | - |
| LW | -2.10 [-3.43, -0.90] | 8.5e-05, 0.0015 | ahead | 55/21/24 | - | - | - | - | - | - | - | - |
| SW | -0.21 [-1.13, +0.73] | 0.21, 1 | not distinguishable | 51/3/46 | - | - | - | - | - | - | - | - |
| LSW | -0.44 [-1.85, +0.93] | 0.014, 1 | not distinguishable | 45/5/50 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.71 [-2.36, -1.08] | 0.0032, 0.0004 | ahead | 75/2/23 | - | - | - | - | - | - | - | - |
| L | -1.64 [-2.47, -0.86] | 7.2e-07, 0.0004 | ahead | 59/20/21 | - | - | - | - | - | - | - | - |
| S | -0.36 [-1.03, +0.41] | 0.29, 0.63 | not distinguishable | 57/5/37 | - | - | - | - | - | - | - | - |
| W | -0.85 [-1.81, +0.24] | 0.0065, 0.29 | not distinguishable | 63/7/30 | - | - | - | - | - | - | - | - |
| LS | +0.07 [-0.79, +0.85] | 1, 1 | not distinguishable | 49/7/44 | - | - | - | - | - | - | - | - |
| LW | -1.36 [-2.31, -0.45] | 7.9e-05, 0.01 | ahead | 46/29/25 | - | - | - | - | - | - | - | - |
| SW | +0.70 [-0.18, +1.67] | 1, 1 | not distinguishable | 48/7/46 | - | - | - | - | - | - | - | - |
| LSW | +0.32 [-0.61, +1.18] | 1, 1 | not distinguishable | 43/9/48 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | MWKR | -1.92 | -10.84 | 15.6 | -2.49 | 14.9 |
| L | SPT (SPT) | MOR | -2.63 | -4.94 | 9.3 | -2.11 | 13.3 |
| S | MWKR (MWKR) | MWKR | -4.91 | -4.91 | 12.8 | -2.97 | 14.4 |
| W | SPT (MWKR) | MWKR | -2.12 | -12.92 | 17.5 | -2.38 | 11.5 |
| LS | SPT (FIFO) | MWKR | -3.90 | -4.87 | 12.1 | -3.12 | 13.7 |
| LW | SPT (SPT) | MWKR | -2.92 | -7.51 | 12.1 | -2.22 | 9.8 |
| SW | FIFO (FIFO) | MWKR | -4.07 | -5.79 | 11.4 | -3.31 | 13.3 |
| LSW | SPT (SPT) | MOR | -4.21 | -5.04 | 11.9 | -3.75 | 13.0 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 30.8 (29.7-32.2, n=5) | 33.2 (31.9-34.6, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 39.3 (38.9-39.7, n=5) | 40.2 (39.3-40.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 70.1 (69.5-70.9, n=5) | 70.3 (69.8-70.8, n=4) | 24.1 (24.0-24.3, n=5) | 24.0 (23.6-24.2, n=4) |
| W | 32.8 (31.5-33.5, n=5) | 35.8 (34.5-36.9, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 74.8 (74.4-75.2, n=5) | 74.6 (74.0-75.0, n=4) | 19.3 (19.1-19.4, n=5) | 19.1 (18.8-19.3, n=4) |
| LW | 42.5 (41.8-43.4, n=5) | 44.1 (43.6-44.7, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 71.8 (71.3-72.2, n=5) | 72.1 (71.7-72.4, n=4) | 20.7 (20.5-20.8, n=5) | 20.4 (20.2-20.5, n=4) |
| LSW | 75.6 (75.4-75.9, n=5) | 75.4 (75.1-75.7, n=4) | 17.3 (17.2-17.5, n=5) | 17.1 (17.0-17.2, n=4) |
