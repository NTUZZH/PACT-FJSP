# Comparison with dispatching rules, public family, test set pub_Brandimarte

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 10; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -5.64 [-11.24, -0.73] | 9.1e-05, 0.074 | not distinguishable | 56/26/18 | - | - | - | - | - | - | - | - |
| L | -0.13 [-4.56, +4.08] | 1, 1 | not distinguishable | 46/6/48 | - | - | - | - | - | - | - | - |
| S | +5.79 [+1.90, +10.10] | 1, 1 | behind | 22/2/76 | - | - | - | - | - | - | - | - |
| W | -4.50 [-8.45, -1.01] | 3.1e-05, 0.039 | ahead | 56/18/26 | - | - | - | - | - | - | - | - |
| LS | +4.02 [+0.88, +7.38] | 1, 1 | behind | 32/2/66 | - | - | - | - | - | - | - | - |
| LW | -1.70 [-6.67, +3.13] | 0.25, 1 | not distinguishable | 60/4/36 | - | - | - | - | - | - | - | - |
| SW | +5.10 [+2.21, +8.56] | 1, 1 | behind | 22/0/78 | - | - | - | - | - | - | - | - |
| LSW | +4.61 [+1.45, +8.24] | 1, 1 | behind | 26/0/74 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -21.84 [-28.24, -15.51] | 3.8e-07, 0.0004 | ahead | 98/0/2 | - | - | - | - | - | - | - | - |
| L | -11.28 [-16.60, -6.26] | 0.0001, 0.0004 | ahead | 92/4/4 | - | - | - | - | - | - | - | - |
| S | -0.20 [-4.26, +4.64] | 0.87, 0.87 | not distinguishable | 58/0/42 | - | - | - | - | - | - | - | - |
| W | -20.75 [-25.97, -15.56] | 6e-08, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -1.51 [-5.33, +2.50] | 0.098, 0.62 | not distinguishable | 60/6/34 | - | - | - | - | - | - | - | - |
| LW | -13.62 [-19.54, -8.34] | 8.3e-05, 0.0004 | ahead | 96/4/0 | - | - | - | - | - | - | - | - |
| SW | +1.76 [-1.21, +5.53] | 0.99, 0.87 | not distinguishable | 40/2/58 | - | - | - | - | - | - | - | - |
| LSW | -3.94 [-7.80, +1.34] | 7.2e-05, 0.27 | not distinguishable | 84/4/12 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -7.38 [-12.74, -2.65] | 1.6e-06, 0.0016 | ahead | 65/29/6 | - | - | - | - | - | - | - | - |
| L | -2.23 [-6.21, +1.46] | 0.023, 0.62 | not distinguishable | 59/7/35 | - | - | - | - | - | - | - | - |
| S | -0.45 [-2.94, +1.64] | 0.052, 0.99 | not distinguishable | 48/7/46 | - | - | - | - | - | - | - | - |
| W | -5.03 [-8.98, -1.42] | 1.4e-05, 0.014 | ahead | 62/20/18 | - | - | - | - | - | - | - | - |
| LS | -0.74 [-2.79, +1.67] | 0.033, 0.99 | not distinguishable | 61/3/36 | - | - | - | - | - | - | - | - |
| LW | -4.49 [-8.94, -0.31] | 0.0026, 0.1 | not distinguishable | 70/2/27 | - | - | - | - | - | - | - | - |
| SW | -0.60 [-2.68, +1.78] | 0.014, 0.99 | not distinguishable | 59/3/38 | - | - | - | - | - | - | - | - |
| LSW | -0.02 [-2.76, +2.82] | 0.43, 0.99 | not distinguishable | 46/3/51 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -7.11 [-12.61, -2.39] | 4.8e-07, 0.0004 | ahead | 66/31/2 | - | - | - | - | - | - | - | - |
| L | -3.26 [-7.28, +0.22] | 0.0021, 0.17 | not distinguishable | 52/23/25 | - | - | - | - | - | - | - | - |
| S | +0.31 [-1.24, +1.93] | 1, 1 | not distinguishable | 47/3/50 | - | - | - | - | - | - | - | - |
| W | -4.39 [-7.83, -1.31] | 4.7e-05, 0.0038 | ahead | 65/25/10 | - | - | - | - | - | - | - | - |
| LS | +1.18 [-1.26, +3.86] | 1, 1 | not distinguishable | 45/6/49 | - | - | - | - | - | - | - | - |
| LW | -3.97 [-8.17, -0.44] | 0.0018, 0.072 | not distinguishable | 63/9/28 | - | - | - | - | - | - | - | - |
| SW | -1.90 [-4.13, +0.46] | 0.00037, 0.21 | not distinguishable | 68/4/28 | - | - | - | - | - | - | - | - |
| LSW | +1.32 [-0.34, +3.19] | 1, 1 | not distinguishable | 32/3/65 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -7.43 [-12.87, -2.62] | 3.6e-07, 0.0016 | ahead | 64/30/6 | - | - | - | - | - | - | - | - |
| L | -2.23 [-6.32, +1.53] | 0.036, 0.64 | not distinguishable | 58/8/34 | - | - | - | - | - | - | - | - |
| S | -0.30 [-3.00, +1.99] | 0.43, 1 | not distinguishable | 47/7/46 | - | - | - | - | - | - | - | - |
| W | -5.32 [-9.31, -1.68] | 1.1e-05, 0.008 | ahead | 61/20/19 | - | - | - | - | - | - | - | - |
| LS | -0.61 [-2.72, +1.82] | 0.11, 1 | not distinguishable | 61/2/37 | - | - | - | - | - | - | - | - |
| LW | -4.31 [-8.79, -0.08] | 0.0034, 0.14 | not distinguishable | 72/1/27 | - | - | - | - | - | - | - | - |
| SW | -0.26 [-2.31, +2.14] | 0.26, 1 | not distinguishable | 58/2/40 | - | - | - | - | - | - | - | - |
| LSW | +0.33 [-2.50, +3.32] | 0.97, 1 | not distinguishable | 45/3/52 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -7.11 [-12.63, -2.38] | 3.3e-07, 0.0004 | ahead | 66/32/2 | - | - | - | - | - | - | - | - |
| L | -3.22 [-7.24, +0.28] | 0.0023, 0.19 | not distinguishable | 52/22/25 | - | - | - | - | - | - | - | - |
| S | +0.36 [-1.19, +2.00] | 1, 1 | not distinguishable | 47/3/51 | - | - | - | - | - | - | - | - |
| W | -4.39 [-7.82, -1.31] | 4.3e-05, 0.0035 | ahead | 65/24/11 | - | - | - | - | - | - | - | - |
| LS | +1.25 [-1.17, +3.92] | 1, 1 | not distinguishable | 44/6/50 | - | - | - | - | - | - | - | - |
| LW | -3.92 [-8.13, -0.39] | 0.0019, 0.079 | not distinguishable | 62/9/29 | - | - | - | - | - | - | - | - |
| SW | -1.81 [-4.03, +0.54] | 0.00054, 0.25 | not distinguishable | 68/4/29 | - | - | - | - | - | - | - | - |
| LSW | +1.42 [-0.27, +3.28] | 1, 1 | not distinguishable | 31/3/65 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | SPT (SPT) | MWKR | -3.07 | -17.13 | 22.7 | -1.62 | 10.7 |
| L | SPT (SPT) | MWKR | -2.34 | -10.94 | 15.8 | -0.75 | 11.1 |
| S | MWKR (MWKR) | MWKR | -5.65 | -5.65 | 18.8 | -3.94 | 11.6 |
| W | SPT (MWKR) | MOR | -1.61 | -16.88 | 21.7 | -1.79 | 9.2 |
| LS | SPT (FIFO) | FIFO | -3.74 | -5.21 | 11.3 | -4.38 | 11.8 |
| LW | SPT (SPT) | MWKR | -1.64 | -11.93 | 16.8 | -2.21 | 10.4 |
| SW | FIFO (FIFO) | MWKR | -1.57 | -3.15 | 11.8 | -1.53 | 11.3 |
| LSW | SPT (SPT) | MOR | -5.88 | -8.24 | 14.3 | -4.14 | 12.0 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 20.7 (19.2-21.7, n=5) | 21.7 (20.6-22.6, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 26.0 (25.1-26.4, n=5) | 27.0 (25.2-29.0, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 50.6 (49.6-52.1, n=5) | 51.3 (50.0-52.3, n=4) | 23.3 (22.5-24.1, n=5) | 22.9 (21.1-23.6, n=4) |
| W | 22.1 (20.8-24.3, n=5) | 22.8 (22.3-23.3, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 56.6 (56.4-57.1, n=5) | 55.2 (53.1-56.4, n=4) | 20.1 (19.7-20.7, n=5) | 19.8 (19.6-20.1, n=4) |
| LW | 28.1 (26.9-29.1, n=5) | 29.2 (28.4-30.2, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 53.6 (52.9-54.5, n=5) | 53.0 (50.5-54.2, n=4) | 20.4 (20.2-20.7, n=5) | 20.7 (19.9-21.3, n=4) |
| LSW | 57.9 (56.9-59.1, n=5) | 56.9 (55.4-58.9, n=4) | 18.1 (18.1-18.2, n=5) | 17.8 (17.6-18.1, n=4) |
