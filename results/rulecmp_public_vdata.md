# Comparison with dispatching rules, public family, test set pub_Hurink_vdata

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +1.93 [+1.19, +2.84] | 1, 1 | behind | 24/2/74 | - | - | - | - | - | - | - | - |
| L | -0.94 [-1.91, -0.08] | 0.022, 0.13 | not distinguishable | 39/28/33 | - | - | - | - | - | - | - | - |
| S | +5.08 [+3.39, +6.99] | 1, 1 | behind | 26/0/74 | - | - | - | - | - | - | - | - |
| W | +2.61 [+1.31, +4.39] | 1, 1 | behind | 17/6/76 | - | - | - | - | - | - | - | - |
| LS | +3.58 [+2.02, +5.21] | 1, 1 | behind | 26/1/73 | - | - | - | - | - | - | - | - |
| LW | -0.99 [-2.11, +0.08] | 0.048, 0.24 | not distinguishable | 39/21/40 | - | - | - | - | - | - | - | - |
| SW | +5.46 [+3.66, +7.33] | 1, 1 | behind | 24/1/75 | - | - | - | - | - | - | - | - |
| LSW | +3.88 [+2.50, +5.37] | 1, 1 | behind | 29/1/70 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -10.11 [-12.28, -7.87] | 9.7e-06, 0.0004 | ahead | 83/1/16 | - | - | - | - | - | - | - | - |
| L | -4.42 [-6.36, -2.65] | 3.2e-05, 0.0004 | ahead | 63/19/18 | - | - | - | - | - | - | - | - |
| S | -0.13 [-2.16, +1.90] | 0.39, 0.56 | not distinguishable | 50/1/49 | - | - | - | - | - | - | - | - |
| W | -10.76 [-12.99, -8.29] | 0.00013, 0.0004 | ahead | 90/2/7 | - | - | - | - | - | - | - | - |
| LS | -0.76 [-2.40, +0.89] | 0.24, 0.54 | not distinguishable | 56/0/44 | - | - | - | - | - | - | - | - |
| LW | -6.95 [-9.34, -4.67] | 1.8e-05, 0.0004 | ahead | 73/10/16 | - | - | - | - | - | - | - | - |
| SW | -0.54 [-2.43, +1.34] | 0.38, 0.56 | not distinguishable | 51/2/47 | - | - | - | - | - | - | - | - |
| LSW | -1.63 [-3.42, +0.20] | 0.0068, 0.16 | not distinguishable | 61/2/36 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.01 [-0.51, +0.51] | 1, 1 | not distinguishable | 44/4/52 | - | - | - | - | - | - | - | - |
| L | -2.15 [-3.12, -1.29] | 7.4e-05, 0.0004 | ahead | 50/38/11 | - | - | - | - | - | - | - | - |
| S | -0.37 [-1.36, +0.58] | 0.11, 0.92 | not distinguishable | 52/1/47 | - | - | - | - | - | - | - | - |
| W | +0.18 [-0.72, +1.11] | 1, 1 | not distinguishable | 35/7/57 | - | - | - | - | - | - | - | - |
| LS | -0.51 [-1.50, +0.47] | 0.00037, 0.76 | not distinguishable | 53/4/43 | - | - | - | - | - | - | - | - |
| LW | -2.51 [-3.58, -1.52] | 0.0001, 0.0004 | ahead | 51/32/17 | - | - | - | - | - | - | - | - |
| SW | +0.33 [-0.57, +1.27] | 1, 1 | not distinguishable | 50/2/48 | - | - | - | - | - | - | - | - |
| LSW | -1.04 [-1.93, -0.16] | 0.00029, 0.057 | not distinguishable | 62/3/35 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.15 [-0.59, +0.28] | 0.75, 1 | not distinguishable | 39/14/47 | - | - | - | - | - | - | - | - |
| L | -1.60 [-2.35, -0.94] | 0.00012, 0.0004 | ahead | 46/48/7 | - | - | - | - | - | - | - | - |
| S | +0.21 [-0.32, +0.73] | 1, 1 | not distinguishable | 48/2/49 | - | - | - | - | - | - | - | - |
| W | +1.17 [+0.47, +1.92] | 1, 1 | behind | 25/5/69 | - | - | - | - | - | - | - | - |
| LS | +0.04 [-0.62, +0.70] | 1, 1 | not distinguishable | 46/11/43 | - | - | - | - | - | - | - | - |
| LW | -2.24 [-3.26, -1.31] | 6.9e-05, 0.0004 | ahead | 48/42/9 | - | - | - | - | - | - | - | - |
| SW | +0.05 [-0.71, +0.84] | 1, 1 | not distinguishable | 50/4/46 | - | - | - | - | - | - | - | - |
| LSW | -0.69 [-1.38, +0.00] | 0.014, 0.15 | not distinguishable | 56/6/38 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.04 [-0.56, +0.51] | 1, 1 | not distinguishable | 45/4/51 | - | - | - | - | - | - | - | - |
| L | -2.13 [-3.12, -1.25] | 0.00012, 0.0004 | ahead | 50/38/12 | - | - | - | - | - | - | - | - |
| S | -0.47 [-1.50, +0.51] | 0.081, 0.88 | not distinguishable | 54/1/45 | - | - | - | - | - | - | - | - |
| W | +0.04 [-0.91, +1.03] | 1, 1 | not distinguishable | 36/7/58 | - | - | - | - | - | - | - | - |
| LS | -0.47 [-1.54, +0.57] | 0.034, 0.88 | not distinguishable | 53/4/44 | - | - | - | - | - | - | - | - |
| LW | -2.47 [-3.55, -1.49] | 0.00016, 0.0004 | ahead | 51/32/17 | - | - | - | - | - | - | - | - |
| SW | +0.21 [-0.74, +1.20] | 1, 1 | not distinguishable | 50/2/47 | - | - | - | - | - | - | - | - |
| LSW | -0.89 [-1.80, +0.01] | 0.00098, 0.16 | not distinguishable | 62/2/36 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.15 [-0.59, +0.29] | 0.8, 1 | not distinguishable | 39/14/47 | - | - | - | - | - | - | - | - |
| L | -1.60 [-2.35, -0.94] | 0.00012, 0.0004 | ahead | 46/47/6 | - | - | - | - | - | - | - | - |
| S | +0.18 [-0.35, +0.72] | 1, 1 | not distinguishable | 49/2/49 | - | - | - | - | - | - | - | - |
| W | +1.15 [+0.44, +1.91] | 1, 1 | behind | 25/6/69 | - | - | - | - | - | - | - | - |
| LS | +0.05 [-0.62, +0.72] | 1, 1 | not distinguishable | 46/11/43 | - | - | - | - | - | - | - | - |
| LW | -2.23 [-3.25, -1.30] | 9e-05, 0.0004 | ahead | 48/42/10 | - | - | - | - | - | - | - | - |
| SW | +0.05 [-0.69, +0.84] | 1, 1 | not distinguishable | 51/4/45 | - | - | - | - | - | - | - | - |
| LSW | -0.64 [-1.34, +0.05] | 0.021, 0.21 | not distinguishable | 56/6/38 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | MWKR | -1.09 | -11.84 | 15.5 | -1.02 | 14.7 |
| L | SPT (SPT) | MOR | -1.65 | -3.57 | 7.3 | -1.09 | 10.4 |
| S | MWKR (MWKR) | MWKR | -4.89 | -4.89 | 14.9 | -3.67 | 15.2 |
| W | SPT (MWKR) | MWKR | -1.08 | -13.01 | 17.0 | -2.59 | 12.8 |
| LS | SPT (FIFO) | MWKR | -3.44 | -4.08 | 9.2 | -2.83 | 14.2 |
| LW | SPT (SPT) | MOR | -2.23 | -6.15 | 9.8 | -1.05 | 11.1 |
| SW | FIFO (FIFO) | MWKR | -4.32 | -5.60 | 13.5 | -2.82 | 14.8 |
| LSW | SPT (SPT) | MOR | -3.49 | -5.21 | 10.3 | -3.02 | 14.2 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 18.9 (18.4-19.4, n=5) | 21.1 (19.9-23.0, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 17.2 (16.8-17.7, n=5) | 17.7 (17.2-18.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 58.4 (57.8-59.0, n=5) | 58.2 (57.3-58.9, n=4) | 24.9 (24.5-25.1, n=5) | 24.5 (24.1-24.8, n=4) |
| W | 19.3 (18.3-20.6, n=5) | 22.2 (21.3-23.5, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 60.9 (60.3-61.2, n=5) | 60.6 (60.0-61.2, n=4) | 18.7 (18.4-18.9, n=5) | 18.5 (18.3-18.7, n=4) |
| LW | 23.5 (22.3-24.4, n=5) | 24.8 (23.8-26.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 59.3 (58.5-60.3, n=5) | 59.7 (59.1-60.6, n=4) | 21.3 (21.0-21.7, n=5) | 21.2 (20.8-21.5, n=4) |
| LSW | 63.9 (63.4-64.9, n=5) | 64.0 (63.7-64.3, n=4) | 17.3 (17.2-17.4, n=5) | 17.2 (17.0-17.4, n=4) |
