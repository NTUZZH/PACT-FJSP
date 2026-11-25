# Comparison with dispatching rules, public family, test set pub_Hurink_edata

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +1.28 [-0.04, +2.62] | 1, 1 | not distinguishable | 41/2/56 | - | - | - | - | - | - | - | - |
| L | -0.31 [-1.55, +0.94] | 0.19, 1 | not distinguishable | 48/8/44 | - | - | - | - | - | - | - | - |
| S | +3.76 [+2.89, +4.62] | 1, 1 | behind | 16/6/78 | - | - | - | - | - | - | - | - |
| W | +1.74 [+0.46, +3.04] | 1, 1 | behind | 35/10/55 | - | - | - | - | - | - | - | - |
| LS | +2.47 [+1.38, +3.58] | 1, 1 | behind | 30/7/63 | - | - | - | - | - | - | - | - |
| LW | -0.32 [-1.79, +1.16] | 1, 1 | not distinguishable | 41/24/36 | - | - | - | - | - | - | - | - |
| SW | +2.81 [+1.70, +3.98] | 1, 1 | behind | 22/11/67 | - | - | - | - | - | - | - | - |
| LSW | +3.80 [+2.51, +5.15] | 1, 1 | behind | 24/7/70 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -6.12 [-8.33, -3.91] | 0.00012, 0.0004 | ahead | 73/0/27 | - | - | - | - | - | - | - | - |
| L | -6.12 [-8.51, -3.83] | 2.2e-06, 0.0004 | ahead | 71/6/22 | - | - | - | - | - | - | - | - |
| S | -0.76 [-2.05, +0.50] | 0.0079, 0.24 | not distinguishable | 48/5/47 | - | - | - | - | - | - | - | - |
| W | -9.75 [-11.96, -7.52] | 6e-06, 0.0004 | ahead | 83/2/15 | - | - | - | - | - | - | - | - |
| LS | -0.98 [-2.09, +0.14] | 0.0013, 0.12 | not distinguishable | 53/12/35 | - | - | - | - | - | - | - | - |
| LW | -6.75 [-9.10, -4.51] | 7.4e-05, 0.0004 | ahead | 67/16/17 | - | - | - | - | - | - | - | - |
| SW | -0.61 [-1.75, +0.51] | 0.049, 0.24 | not distinguishable | 50/10/39 | - | - | - | - | - | - | - | - |
| LSW | -1.32 [-2.74, +0.05] | 0.0042, 0.12 | not distinguishable | 53/12/35 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.65 [-2.84, -0.46] | 0.041, 0.016 | ahead | 64/3/33 | - | - | - | - | - | - | - | - |
| L | -2.57 [-3.79, -1.41] | 9.5e-05, 0.0004 | ahead | 66/7/27 | - | - | - | - | - | - | - | - |
| S | +0.91 [+0.16, +1.62] | 1, 1 | not distinguishable | 33/9/58 | - | - | - | - | - | - | - | - |
| W | -0.82 [-1.98, +0.36] | 0.21, 0.42 | not distinguishable | 56/11/33 | - | - | - | - | - | - | - | - |
| LS | -0.12 [-1.04, +0.84] | 0.92, 1 | not distinguishable | 48/11/40 | - | - | - | - | - | - | - | - |
| LW | -2.12 [-3.43, -0.83] | 0.0016, 0.0056 | ahead | 51/26/24 | - | - | - | - | - | - | - | - |
| SW | +0.49 [-0.34, +1.34] | 1, 1 | not distinguishable | 40/12/47 | - | - | - | - | - | - | - | - |
| LSW | +0.76 [-0.31, +1.96] | 1, 1 | not distinguishable | 43/9/48 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.95 [-2.99, -0.96] | 0.02, 0.0009 | ahead | 71/4/25 | - | - | - | - | - | - | - | - |
| L | -2.38 [-3.54, -1.28] | 0.00036, 0.0004 | ahead | 63/9/28 | - | - | - | - | - | - | - | - |
| S | +1.22 [+0.47, +2.03] | 1, 1 | behind | 29/12/58 | - | - | - | - | - | - | - | - |
| W | -1.09 [-2.18, +0.02] | 0.076, 0.14 | not distinguishable | 57/12/31 | - | - | - | - | - | - | - | - |
| LS | +0.22 [-0.54, +1.05] | 1, 1 | not distinguishable | 42/13/44 | - | - | - | - | - | - | - | - |
| LW | -2.16 [-3.42, -0.96] | 0.0018, 0.0004 | ahead | 51/30/19 | - | - | - | - | - | - | - | - |
| SW | +0.46 [-0.26, +1.24] | 1, 1 | not distinguishable | 37/16/47 | - | - | - | - | - | - | - | - |
| LSW | +1.20 [+0.16, +2.34] | 1, 1 | not distinguishable | 34/15/51 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.60 [-2.78, -0.43] | 0.034, 0.019 | ahead | 62/3/34 | - | - | - | - | - | - | - | - |
| L | -2.46 [-3.69, -1.29] | 8.9e-05, 0.0004 | ahead | 66/7/27 | - | - | - | - | - | - | - | - |
| S | +1.07 [+0.34, +1.76] | 1, 1 | behind | 32/9/59 | - | - | - | - | - | - | - | - |
| W | -0.80 [-1.92, +0.33] | 0.1, 0.4 | not distinguishable | 55/11/34 | - | - | - | - | - | - | - | - |
| LS | +0.07 [-0.88, +1.05] | 1, 1 | not distinguishable | 47/11/42 | - | - | - | - | - | - | - | - |
| LW | -2.06 [-3.39, -0.76] | 0.0031, 0.0077 | ahead | 50/26/24 | - | - | - | - | - | - | - | - |
| SW | +0.66 [-0.20, +1.54] | 1, 1 | not distinguishable | 39/12/49 | - | - | - | - | - | - | - | - |
| LSW | +1.04 [-0.05, +2.24] | 1, 1 | not distinguishable | 41/9/50 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.93 [-2.96, -0.94] | 0.02, 0.0012 | ahead | 71/4/26 | - | - | - | - | - | - | - | - |
| L | -2.34 [-3.50, -1.23] | 0.00039, 0.0004 | ahead | 63/9/28 | - | - | - | - | - | - | - | - |
| S | +1.27 [+0.52, +2.09] | 1, 1 | behind | 29/12/59 | - | - | - | - | - | - | - | - |
| W | -1.06 [-2.14, +0.05] | 0.072, 0.15 | not distinguishable | 57/11/32 | - | - | - | - | - | - | - | - |
| LS | +0.27 [-0.49, +1.10] | 1, 1 | not distinguishable | 42/13/45 | - | - | - | - | - | - | - | - |
| LW | -2.12 [-3.39, -0.92] | 0.0021, 0.0007 | ahead | 51/30/19 | - | - | - | - | - | - | - | - |
| SW | +0.51 [-0.21, +1.29] | 1, 1 | not distinguishable | 36/16/48 | - | - | - | - | - | - | - | - |
| LSW | +1.25 [+0.22, +2.39] | 1, 1 | behind | 33/15/52 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | MWKR | -3.26 | -7.32 | 14.1 | -1.70 | 11.8 |
| L | SPT (SPT) | MOR | -3.83 | -5.87 | 12.3 | -1.72 | 11.1 |
| S | MWKR (MWKR) | SPT | -3.99 | -4.32 | 9.8 | -2.14 | 10.2 |
| W | SPT (MWKR) | MWKR | -3.36 | -11.22 | 18.2 | -1.86 | 8.7 |
| LS | SPT (FIFO) | MOR | -4.13 | -3.31 | 9.0 | -1.85 | 9.8 |
| LW | SPT (SPT) | MWKR | -3.76 | -6.46 | 13.3 | -1.41 | 7.9 |
| SW | FIFO (FIFO) | MOR | -3.32 | -3.27 | 8.1 | -1.49 | 8.4 |
| LSW | SPT (SPT) | FIFO | -3.04 | -4.81 | 8.6 | -1.97 | 8.2 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 40.1 (37.2-42.1, n=5) | 43.0 (41.7-45.5, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 51.3 (49.7-52.9, n=5) | 53.7 (52.8-55.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 74.2 (73.7-74.6, n=5) | 74.1 (73.8-74.4, n=4) | 22.4 (22.3-22.5, n=5) | 22.3 (22.0-22.4, n=4) |
| W | 41.8 (40.8-43.5, n=5) | 43.7 (42.5-45.6, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 78.9 (78.6-79.2, n=5) | 78.4 (78.2-78.5, n=4) | 19.1 (18.9-19.2, n=5) | 18.8 (18.6-18.9, n=4) |
| LW | 51.5 (49.9-52.9, n=5) | 52.8 (51.6-54.3, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 75.5 (75.1-76.1, n=5) | 75.7 (75.4-76.0, n=4) | 18.9 (18.7-19.0, n=5) | 18.9 (18.6-19.0, n=4) |
| LSW | 79.0 (78.8-79.3, n=5) | 78.8 (78.5-79.1, n=4) | 16.5 (16.4-16.6, n=5) | 16.5 (16.4-16.6, n=4) |
