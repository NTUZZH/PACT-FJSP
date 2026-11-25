# Comparison with dispatching rules, public family under the active mask, test set pub_Hurink_rdata

every run trained and every schedule built under the active mask: the policy's sampled and greedy schedules under its run's own mask, the four rules rolled out under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask active); a trace counts the pairs the active mask admits

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +1.28 [-0.25, +2.95] | 1, 1 | not distinguishable | 40/1/59 | - | - | - | - | - | - | - | - |
| L | -1.15 [-2.40, +0.09] | 0.024, 0.24 | not distinguishable | 58/7/35 | - | - | - | - | - | - | - | - |
| S | +0.75 [-0.46, +2.04] | 1, 1 | not distinguishable | 48/0/51 | - | - | - | - | - | - | - | - |
| W | +2.90 [+0.96, +5.11] | 1, 1 | not distinguishable | 33/8/59 | - | - | - | - | - | - | - | - |
| LS | -1.68 [-2.84, -0.53] | 0.011, 0.019 | ahead | 64/0/36 | - | - | - | - | - | - | - | - |
| LW | -0.55 [-1.96, +0.84] | 0.37, 1 | not distinguishable | 51/11/37 | - | - | - | - | - | - | - | - |
| SW | +1.25 [-0.03, +2.59] | 1, 1 | not distinguishable | 43/1/56 | - | - | - | - | - | - | - | - |
| LSW | -1.18 [-2.54, +0.29] | 0.086, 0.33 | not distinguishable | 62/2/36 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.12 [-2.71, +0.61] | 0.21, 0.19 | not distinguishable | 57/1/42 | - | - | - | - | - | - | - | - |
| L | -4.28 [-5.81, -2.75] | 0.00014, 0.0004 | ahead | 78/5/17 | - | - | - | - | - | - | - | - |
| S | -3.29 [-4.78, -1.79] | 0.0042, 0.00045 | ahead | 74/0/26 | - | - | - | - | - | - | - | - |
| W | +0.23 [-1.66, +2.33] | 0.59, 0.57 | not distinguishable | 46/10/44 | - | - | - | - | - | - | - | - |
| LS | -5.00 [-6.53, -3.50] | 0.00014, 0.0004 | ahead | 78/0/22 | - | - | - | - | - | - | - | - |
| LW | -4.83 [-6.59, -3.16] | 0.00021, 0.0004 | ahead | 67/15/17 | - | - | - | - | - | - | - | - |
| SW | -2.72 [-3.94, -1.43] | 0.0042, 0.0004 | ahead | 71/2/27 | - | - | - | - | - | - | - | - |
| LSW | -3.92 [-5.68, -2.11] | 0.00082, 0.0004 | ahead | 72/1/27 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.86 [-3.16, -0.47] | 0.046, 0.012 | ahead | 68/1/30 | - | - | - | - | - | - | - | - |
| L | -3.20 [-4.48, -1.99] | 0.00012, 0.0004 | ahead | 71/10/19 | - | - | - | - | - | - | - | - |
| S | -1.64 [-2.70, -0.49] | 0.035, 0.012 | ahead | 67/1/32 | - | - | - | - | - | - | - | - |
| W | -0.74 [-1.99, +0.60] | 0.12, 0.13 | not distinguishable | 56/7/37 | - | - | - | - | - | - | - | - |
| LS | -4.61 [-5.60, -3.65] | 5.1e-05, 0.0004 | ahead | 87/0/13 | - | - | - | - | - | - | - | - |
| LW | -2.87 [-4.12, -1.71] | 0.0003, 0.0004 | ahead | 68/14/18 | - | - | - | - | - | - | - | - |
| SW | -1.68 [-2.80, -0.47] | 0.029, 0.012 | ahead | 66/2/32 | - | - | - | - | - | - | - | - |
| LSW | -4.52 [-5.55, -3.52] | 2.1e-05, 0.0004 | ahead | 86/1/13 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.92 [-2.04, +0.32] | 0.2, 0.14 | not distinguishable | 63/2/34 | - | - | - | - | - | - | - | - |
| L | -2.46 [-3.59, -1.41] | 0.0001, 0.0004 | ahead | 64/22/14 | - | - | - | - | - | - | - | - |
| S | -1.51 [-2.44, -0.45] | 0.061, 0.012 | not distinguishable | 69/1/30 | - | - | - | - | - | - | - | - |
| W | -0.74 [-1.84, +0.46] | 0.2, 0.14 | not distinguishable | 55/10/36 | - | - | - | - | - | - | - | - |
| LS | -3.14 [-3.90, -2.38] | 0.00013, 0.0004 | ahead | 84/1/15 | - | - | - | - | - | - | - | - |
| LW | -2.45 [-3.49, -1.50] | 0.00013, 0.0004 | ahead | 55/33/12 | - | - | - | - | - | - | - | - |
| SW | -0.80 [-1.63, +0.16] | 0.15, 0.14 | not distinguishable | 60/2/38 | - | - | - | - | - | - | - | - |
| LSW | -3.68 [-4.62, -2.74] | 9.6e-05, 0.0004 | ahead | 82/1/17 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -1.89 [-3.26, -0.41] | 0.054, 0.013 | not distinguishable | 68/1/31 | - | - | - | - | - | - | - | - |
| L | -3.22 [-4.50, -2.02] | 0.00012, 0.0004 | ahead | 71/10/18 | - | - | - | - | - | - | - | - |
| S | -1.92 [-2.95, -0.77] | 0.02, 0.0032 | ahead | 71/0/29 | - | - | - | - | - | - | - | - |
| W | -0.70 [-2.02, +0.76] | 0.16, 0.16 | not distinguishable | 56/7/37 | - | - | - | - | - | - | - | - |
| LS | -4.68 [-5.69, -3.70] | 4.5e-05, 0.0004 | ahead | 87/0/13 | - | - | - | - | - | - | - | - |
| LW | -2.93 [-4.19, -1.74] | 0.00037, 0.0004 | ahead | 68/14/18 | - | - | - | - | - | - | - | - |
| SW | -1.83 [-2.95, -0.63] | 0.02, 0.0045 | ahead | 68/2/31 | - | - | - | - | - | - | - | - |
| LSW | -4.55 [-5.61, -3.53] | 4.4e-05, 0.0004 | ahead | 85/1/13 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.94 [-2.08, +0.33] | 0.2, 0.14 | not distinguishable | 64/2/34 | - | - | - | - | - | - | - | - |
| L | -2.46 [-3.59, -1.41] | 9.9e-05, 0.0004 | ahead | 64/22/14 | - | - | - | - | - | - | - | - |
| S | -1.58 [-2.50, -0.52] | 0.054, 0.008 | not distinguishable | 70/1/29 | - | - | - | - | - | - | - | - |
| W | -0.70 [-1.82, +0.52] | 0.2, 0.14 | not distinguishable | 55/9/36 | - | - | - | - | - | - | - | - |
| LS | -3.16 [-3.93, -2.40] | 9.9e-05, 0.0004 | ahead | 84/1/15 | - | - | - | - | - | - | - | - |
| LW | -2.45 [-3.50, -1.50] | 0.00015, 0.0004 | ahead | 55/33/12 | - | - | - | - | - | - | - | - |
| SW | -0.82 [-1.66, +0.13] | 0.14, 0.13 | not distinguishable | 60/2/38 | - | - | - | - | - | - | - | - |
| LSW | -3.66 [-4.60, -2.72] | 9.9e-05, 0.0004 | ahead | 82/1/17 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | MWKR (MWKR) | FIFO | -3.82 | -2.34 | 38.3 | -3.00 | 15.4 |
| L | MOR (MOR) | MOR | -3.15 | -3.15 | 18.5 | -1.94 | 14.9 |
| S | MOR (MWKR) | MWKR | -2.94 | -3.95 | 26.8 | -2.36 | 15.6 |
| W | MWKR (MWKR) | FIFO | -3.74 | -2.49 | 35.4 | -2.39 | 13.1 |
| LS | MOR (MOR) | MWKR | -3.04 | -3.38 | 21.1 | -3.35 | 15.6 |
| LW | MOR (MWKR) | MWKR | -3.20 | -4.27 | 23.3 | -1.80 | 12.4 |
| SW | MOR (MOR) | MWKR | -1.52 | -3.81 | 25.8 | -3.03 | 15.2 |
| LSW | MOR (MWKR) | MOR | -2.81 | -2.81 | 23.3 | -3.15 | 15.5 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 1.9 (1.5-2.2, n=5) | 3.6 (2.7-5.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 3.8 (3.1-4.2, n=5) | 5.8 (4.3-7.7, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 4.0 (3.0-4.7, n=5) | 6.3 (4.2-8.5, n=4) | 29.7 (29.3-30.1, n=5) | 26.9 (23.3-28.3, n=4) |
| W | 3.4 (2.8-3.8, n=5) | 5.5 (3.8-7.3, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 5.7 (4.7-6.4, n=5) | 8.1 (5.6-10.4, n=4) | 23.6 (23.4-23.9, n=5) | 21.0 (19.0-21.8, n=4) |
| LW | 5.1 (4.3-5.8, n=5) | 7.3 (4.6-9.7, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 5.6 (4.8-6.4, n=5) | 7.7 (5.1-9.9, n=4) | 25.0 (24.8-25.2, n=5) | 22.9 (20.0-24.0, n=4) |
| LSW | 7.2 (6.6-8.0, n=5) | 9.3 (5.9-12.3, n=4) | 21.4 (21.2-21.7, n=5) | 19.0 (16.9-20.0, n=4) |
