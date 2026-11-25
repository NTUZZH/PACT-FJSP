# Comparison with dispatching rules, public family, test set pub_SD2_test_20x10+mix

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 100; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -23.41 [-24.86, -21.91] | 2.7e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | +0.23 [-0.70, +1.13] | 1, 1 | not distinguishable | 50/1/49 | - | - | - | - | - | - | - | - |
| S | +3.11 [+2.13, +4.14] | 1, 1 | behind | 32/0/68 | - | - | - | - | - | - | - | - |
| W | -22.17 [-23.57, -20.72] | 5e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | +5.25 [+3.93, +6.69] | 1, 1 | behind | 24/0/75 | - | - | - | - | - | - | - | - |
| LW | -0.51 [-1.43, +0.43] | 0.064, 0.85 | not distinguishable | 53/3/44 | - | - | - | - | - | - | - | - |
| SW | +2.85 [+1.75, +4.00] | 1, 1 | behind | 35/1/63 | - | - | - | - | - | - | - | - |
| LSW | +6.10 [+4.70, +7.49] | 1, 1 | behind | 26/0/74 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -23.56 [-25.04, -22.04] | 2.7e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -1.03 [-2.04, -0.05] | 0.026, 0.057 | not distinguishable | 60/0/40 | - | - | - | - | - | - | - | - |
| S | -3.60 [-4.79, -2.39] | 2.8e-05, 0.0004 | ahead | 70/0/30 | - | - | - | - | - | - | - | - |
| W | -23.27 [-24.82, -21.67] | 4e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -0.80 [-2.44, +0.93] | 0.039, 0.35 | not distinguishable | 55/0/45 | - | - | - | - | - | - | - | - |
| LW | -2.78 [-3.99, -1.60] | 7.9e-05, 0.0004 | ahead | 65/2/33 | - | - | - | - | - | - | - | - |
| SW | -2.38 [-3.76, -0.98] | 0.0022, 0.0022 | ahead | 63/0/37 | - | - | - | - | - | - | - | - |
| LSW | -0.70 [-2.57, +1.19] | 0.049, 0.35 | not distinguishable | 55/0/45 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -25.65 [-26.96, -24.29] | 8.5e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.16 [-2.90, -1.44] | 0.00051, 0.0004 | ahead | 71/1/28 | - | - | - | - | - | - | - | - |
| S | -2.28 [-2.99, -1.56] | 6.5e-05, 0.0004 | ahead | 69/0/31 | - | - | - | - | - | - | - | - |
| W | -24.61 [-25.88, -23.26] | 2.7e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -0.74 [-1.56, +0.08] | 0.00051, 0.082 | not distinguishable | 55/0/45 | - | - | - | - | - | - | - | - |
| LW | -3.24 [-4.13, -2.33] | 0.00017, 0.0004 | ahead | 75/4/21 | - | - | - | - | - | - | - | - |
| SW | -2.44 [-3.06, -1.82] | 2.7e-05, 0.0004 | ahead | 71/1/28 | - | - | - | - | - | - | - | - |
| LSW | -0.70 [-1.51, +0.11] | 0.0014, 0.082 | not distinguishable | 57/0/43 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -24.63 [-25.86, -23.36] | 1e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.71 [-3.35, -2.10] | 8.7e-05, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| S | -3.09 [-3.73, -2.46] | 5.7e-06, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| W | -25.07 [-26.45, -23.61] | 4.4e-07, 0.0004 | ahead | 99/1/-0 | - | - | - | - | - | - | - | - |
| LS | -1.67 [-2.36, -0.98] | 8.7e-05, 0.0004 | ahead | 68/0/32 | - | - | - | - | - | - | - | - |
| LW | -3.54 [-4.37, -2.72] | 3.5e-05, 0.0004 | ahead | 78/8/14 | - | - | - | - | - | - | - | - |
| SW | -3.15 [-3.71, -2.59] | 7.2e-07, 0.0004 | ahead | 81/1/18 | - | - | - | - | - | - | - | - |
| LSW | -2.47 [-3.21, -1.72] | 8.7e-05, 0.0004 | ahead | 72/0/28 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -25.97 [-27.28, -24.61] | 9.5e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.30 [-3.07, -1.56] | 0.00066, 0.0004 | ahead | 72/1/27 | - | - | - | - | - | - | - | - |
| S | -2.27 [-2.98, -1.54] | 4.8e-05, 0.0004 | ahead | 69/0/30 | - | - | - | - | - | - | - | - |
| W | -24.82 [-26.16, -23.41] | 6.6e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -0.64 [-1.50, +0.22] | 0.0052, 0.15 | not distinguishable | 53/0/46 | - | - | - | - | - | - | - | - |
| LW | -3.31 [-4.19, -2.40] | 0.00012, 0.0004 | ahead | 75/4/20 | - | - | - | - | - | - | - | - |
| SW | -2.44 [-3.09, -1.79] | 4.4e-05, 0.0004 | ahead | 71/1/28 | - | - | - | - | - | - | - | - |
| LSW | -0.52 [-1.41, +0.36] | 0.035, 0.15 | not distinguishable | 56/0/44 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -24.72 [-25.93, -23.47] | 1.1e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.74 [-3.38, -2.12] | 0.00011, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| S | -3.05 [-3.70, -2.42] | 9.3e-06, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| W | -25.15 [-26.55, -23.66] | 7.1e-07, 0.0004 | ahead | 99/1/-0 | - | - | - | - | - | - | - | - |
| LS | -1.62 [-2.31, -0.92] | 0.00011, 0.0004 | ahead | 67/0/33 | - | - | - | - | - | - | - | - |
| LW | -3.54 [-4.37, -2.72] | 3.8e-05, 0.0004 | ahead | 78/8/14 | - | - | - | - | - | - | - | - |
| SW | -3.13 [-3.70, -2.58] | 1.8e-06, 0.0004 | ahead | 80/1/18 | - | - | - | - | - | - | - | - |
| LSW | -2.42 [-3.18, -1.65] | 0.00011, 0.0004 | ahead | 71/0/29 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | SPT | -0.22 | -0.22 | 28.2 | -4.14 | 11.5 |
| L | SPT (SPT) | SPT | -1.24 | -1.24 | 9.7 | -1.28 | 11.1 |
| S | MWKR (MWKR) | SPT | -4.47 | -6.42 | 12.9 | -2.78 | 11.1 |
| W | SPT (MWKR) | SPT | -1.46 | -1.46 | 17.9 | -2.47 | 9.7 |
| LS | SPT (FIFO) | MWKR | -5.77 | -5.70 | 14.6 | -2.67 | 10.8 |
| LW | SPT (SPT) | SPT | -2.31 | -2.31 | 9.9 | -1.64 | 10.0 |
| SW | FIFO (FIFO) | SPT | -3.58 | -5.05 | 10.8 | -2.53 | 10.9 |
| LSW | SPT (SPT) | MWKR | -5.22 | -6.43 | 14.9 | -2.23 | 10.8 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 4.0 (3.2-4.8, n=5) | 6.1 (2.7-14.7, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 24.3 (23.9-25.1, n=5) | 23.4 (22.2-24.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 56.9 (56.7-57.1, n=5) | 57.0 (56.4-57.5, n=4) | 26.1 (26.0-26.2, n=5) | 25.4 (25.1-25.8, n=4) |
| W | 8.8 (7.7-9.4, n=5) | 12.7 (10.0-19.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 63.2 (63.0-63.4, n=5) | 63.2 (63.1-63.3, n=4) | 21.7 (21.5-21.9, n=5) | 21.3 (21.2-21.5, n=4) |
| LW | 28.9 (28.2-29.4, n=5) | 28.2 (26.7-30.4, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 57.4 (57.1-57.8, n=5) | 57.6 (57.3-57.8, n=4) | 23.5 (23.3-23.7, n=5) | 22.9 (22.7-23.2, n=4) |
| LSW | 63.4 (63.1-63.9, n=5) | 63.2 (63.0-63.7, n=4) | 20.0 (19.7-20.2, n=5) | 19.8 (19.7-19.9, n=4) |
