# Comparison with dispatching rules, public family under the active mask, test set pub_Brandimarte

every run trained and every schedule built under the active mask: the policy's sampled and greedy schedules under its run's own mask, the four rules rolled out under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask active); a trace counts the pairs the active mask admits

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 10; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -4.07 [-8.48, -0.69] | 0.00023, 0.036 | ahead | 70/14/16 | - | - | - | - | - | - | - | - |
| L | -4.29 [-8.21, -0.98] | 0.0004, 0.027 | ahead | 70/4/26 | - | - | - | - | - | - | - | - |
| S | -0.64 [-3.37, +1.97] | 0.17, 0.38 | not distinguishable | 50/2/48 | - | - | - | - | - | - | - | - |
| W | -2.71 [-5.65, -0.16] | 0.0024, 0.089 | not distinguishable | 54/30/16 | - | - | - | - | - | - | - | - |
| LS | -3.48 [-7.12, +0.03] | 0.00033, 0.1 | not distinguishable | 64/4/32 | - | - | - | - | - | - | - | - |
| LW | -4.79 [-8.48, -1.01] | 0.00035, 0.046 | ahead | 78/8/14 | - | - | - | - | - | - | - | - |
| SW | -1.87 [-4.10, +0.34] | 0.038, 0.14 | not distinguishable | 60/2/38 | - | - | - | - | - | - | - | - |
| LSW | -1.84 [-6.05, +2.30] | 0.038, 0.38 | not distinguishable | 58/0/42 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -6.38 [-11.68, -2.56] | 3.3e-05, 0.0004 | ahead | 82/10/8 | - | - | - | - | - | - | - | - |
| L | -7.94 [-13.30, -3.78] | 3.5e-05, 0.0004 | ahead | 90/2/8 | - | - | - | - | - | - | - | - |
| S | -6.46 [-8.84, -3.78] | 0.00035, 0.0004 | ahead | 86/8/6 | - | - | - | - | - | - | - | - |
| W | -4.29 [-8.16, -0.92] | 0.00035, 0.01 | ahead | 64/24/12 | - | - | - | - | - | - | - | - |
| LS | -6.56 [-12.34, -1.68] | 2.3e-05, 0.0085 | ahead | 78/0/22 | - | - | - | - | - | - | - | - |
| LW | -8.04 [-12.97, -3.01] | 3.5e-05, 0.0052 | ahead | 82/4/14 | - | - | - | - | - | - | - | - |
| SW | -6.75 [-10.21, -3.65] | 0.00035, 0.0004 | ahead | 90/0/10 | - | - | - | - | - | - | - | - |
| LSW | -6.77 [-12.56, -1.37] | 0.00035, 0.01 | ahead | 72/4/24 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -5.95 [-10.22, -2.47] | 5.2e-06, 0.0004 | ahead | 78/20/2 | - | - | - | - | - | - | - | - |
| L | -5.96 [-9.88, -2.50] | 2.3e-06, 0.0004 | ahead | 82/7/11 | - | - | - | - | - | - | - | - |
| S | -3.46 [-5.79, -1.43] | 4.7e-05, 0.0004 | ahead | 81/5/14 | - | - | - | - | - | - | - | - |
| W | -3.79 [-6.67, -1.17] | 1.8e-05, 0.0026 | ahead | 68/22/10 | - | - | - | - | - | - | - | - |
| LS | -5.97 [-9.36, -2.87] | 3.2e-06, 0.0004 | ahead | 84/3/13 | - | - | - | - | - | - | - | - |
| LW | -6.57 [-10.02, -3.21] | 3.2e-06, 0.0004 | ahead | 85/3/12 | - | - | - | - | - | - | - | - |
| SW | -4.21 [-6.18, -2.46] | 3.2e-06, 0.0004 | ahead | 87/1/12 | - | - | - | - | - | - | - | - |
| LSW | -5.11 [-9.06, -1.42] | 2.5e-06, 0.0026 | ahead | 71/3/25 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -5.90 [-9.95, -2.43] | 1.2e-05, 0.0004 | ahead | 70/28/2 | - | - | - | - | - | - | - | - |
| L | -6.07 [-9.80, -2.69] | 6.3e-06, 0.0004 | ahead | 75/18/7 | - | - | - | - | - | - | - | - |
| S | -4.08 [-5.85, -2.56] | 2.2e-05, 0.0004 | ahead | 96/2/2 | - | - | - | - | - | - | - | - |
| W | -4.10 [-7.36, -1.29] | 1.2e-05, 0.0005 | ahead | 67/22/11 | - | - | - | - | - | - | - | - |
| LS | -4.97 [-8.30, -2.24] | 6.6e-06, 0.0004 | ahead | 88/6/7 | - | - | - | - | - | - | - | - |
| LW | -5.40 [-9.38, -1.40] | 6.7e-06, 0.0041 | ahead | 75/6/18 | - | - | - | - | - | - | - | - |
| SW | -3.98 [-5.33, -2.65] | 6e-06, 0.0004 | ahead | 91/2/7 | - | - | - | - | - | - | - | - |
| LSW | -6.09 [-9.61, -3.00] | 4.5e-08, 0.0004 | ahead | 88/3/9 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -5.88 [-10.22, -2.40] | 1.3e-05, 0.0004 | ahead | 78/19/2 | - | - | - | - | - | - | - | - |
| L | -5.96 [-9.81, -2.58] | 2.5e-06, 0.0004 | ahead | 83/6/11 | - | - | - | - | - | - | - | - |
| S | -3.51 [-5.76, -1.54] | 3.2e-05, 0.0004 | ahead | 80/4/16 | - | - | - | - | - | - | - | - |
| W | -3.90 [-6.82, -1.33] | 3.2e-05, 0.0018 | ahead | 69/23/8 | - | - | - | - | - | - | - | - |
| LS | -5.92 [-9.33, -2.77] | 5.9e-06, 0.0004 | ahead | 83/4/14 | - | - | - | - | - | - | - | - |
| LW | -6.75 [-10.14, -3.48] | 6.3e-06, 0.0004 | ahead | 91/1/8 | - | - | - | - | - | - | - | - |
| SW | -4.39 [-6.36, -2.60] | 3.2e-05, 0.0004 | ahead | 86/1/12 | - | - | - | - | - | - | - | - |
| LSW | -5.09 [-9.03, -1.38] | 1.3e-05, 0.0028 | ahead | 72/2/26 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -5.87 [-9.96, -2.39] | 1.8e-05, 0.0004 | ahead | 69/28/2 | - | - | - | - | - | - | - | - |
| L | -6.04 [-9.76, -2.67] | 6.5e-06, 0.0004 | ahead | 75/18/7 | - | - | - | - | - | - | - | - |
| S | -4.13 [-5.84, -2.62] | 1.8e-05, 0.0004 | ahead | 96/2/2 | - | - | - | - | - | - | - | - |
| W | -4.13 [-7.40, -1.31] | 9.5e-06, 0.0005 | ahead | 67/24/9 | - | - | - | - | - | - | - | - |
| LS | -4.93 [-8.30, -2.20] | 6.5e-06, 0.0004 | ahead | 87/7/6 | - | - | - | - | - | - | - | - |
| LW | -5.47 [-9.40, -1.47] | 1.5e-05, 0.0038 | ahead | 76/7/17 | - | - | - | - | - | - | - | - |
| SW | -4.07 [-5.43, -2.71] | 1.8e-05, 0.0004 | ahead | 90/2/8 | - | - | - | - | - | - | - | - |
| LSW | -6.09 [-9.61, -2.98] | 2.5e-07, 0.0004 | ahead | 88/3/9 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | MWKR (MWKR) | MWKR | -2.47 | -2.47 | 53.5 | -1.36 | 11.3 |
| L | MOR (MOR) | MWKR | -2.67 | -3.92 | 33.6 | -1.25 | 11.2 |
| S | MOR (MWKR) | MWKR | -1.37 | -5.79 | 49.1 | -1.25 | 12.9 |
| W | MWKR (MWKR) | MWKR | -1.68 | -1.68 | 43.9 | -0.70 | 9.8 |
| LS | MOR (MOR) | MWKR | -1.76 | -3.35 | 33.1 | -3.17 | 13.4 |
| LW | MOR (MWKR) | MWKR | -1.75 | -3.52 | 34.9 | -2.59 | 11.4 |
| SW | MOR (MOR) | MWKR | -2.18 | -4.96 | 37.8 | -1.96 | 12.5 |
| LSW | MOR (MWKR) | MWKR | -1.64 | -5.13 | 33.7 | -1.17 | 13.2 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 2.3 (2.1-2.5, n=5) | 3.6 (2.3-4.3, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 3.3 (2.5-3.9, n=5) | 4.7 (3.3-6.0, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 4.0 (3.0-4.6, n=5) | 5.8 (4.0-6.9, n=4) | 33.0 (32.4-33.4, n=5) | 28.8 (22.7-31.4, n=4) |
| W | 3.3 (3.1-3.4, n=5) | 5.2 (3.3-7.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 5.6 (4.7-6.3, n=5) | 6.9 (5.1-8.7, n=4) | 26.7 (26.5-26.9, n=5) | 23.3 (19.4-24.7, n=4) |
| LW | 4.6 (4.4-4.7, n=5) | 6.3 (4.2-8.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 5.7 (5.2-6.4, n=5) | 8.0 (5.3-9.6, n=4) | 28.8 (28.3-29.1, n=5) | 24.6 (20.7-26.4, n=4) |
| LSW | 6.7 (6.2-7.8, n=5) | 8.6 (5.7-11.3, n=4) | 24.4 (24.1-24.8, n=5) | 21.2 (17.7-22.8, n=4) |
