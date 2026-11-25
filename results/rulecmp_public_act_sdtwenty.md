# Comparison with dispatching rules, public family under the active mask, test set pub_SD2_test_20x10+mix

every run trained and every schedule built under the active mask: the policy's sampled and greedy schedules under its run's own mask, the four rules rolled out under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask active); a trace counts the pairs the active mask admits

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 100; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -30.53 [-31.47, -29.60] | 1.3e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.32 [-2.98, -1.67] | 3.8e-05, 0.0004 | ahead | 71/2/28 | - | - | - | - | - | - | - | - |
| S | -15.78 [-16.68, -14.88] | 1.4e-06, 0.0004 | ahead | 99/0/1 | - | - | - | - | - | - | - | - |
| W | -26.14 [-27.56, -24.66] | 1.1e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -4.06 [-5.02, -3.08] | 1.1e-05, 0.0004 | ahead | 78/1/21 | - | - | - | - | - | - | - | - |
| LW | -3.40 [-4.26, -2.57] | 1.6e-05, 0.0004 | ahead | 69/8/23 | - | - | - | - | - | - | - | - |
| SW | -12.20 [-13.17, -11.17] | 3.7e-06, 0.0004 | ahead | 97/1/2 | - | - | - | - | - | - | - | - |
| LSW | -4.66 [-5.69, -3.61] | 2.8e-05, 0.0004 | ahead | 80/1/20 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -34.18 [-35.07, -33.30] | 7.2e-08, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -5.70 [-6.62, -4.80] | 1.9e-06, 0.0004 | ahead | 91/0/9 | - | - | - | - | - | - | - | - |
| S | -19.25 [-20.28, -18.26] | 6e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| W | -27.90 [-29.39, -26.33] | 7e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -6.99 [-8.15, -5.85] | 1.3e-06, 0.0004 | ahead | 87/0/13 | - | - | - | - | - | - | - | - |
| LW | -6.03 [-7.06, -5.04] | 1.4e-06, 0.0004 | ahead | 86/2/12 | - | - | - | - | - | - | - | - |
| SW | -15.37 [-16.39, -14.31] | 1.3e-06, 0.0004 | ahead | 99/0/1 | - | - | - | - | - | - | - | - |
| LSW | -8.35 [-9.60, -7.11] | 1.9e-06, 0.0004 | ahead | 90/0/10 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -32.20 [-33.13, -31.16] | 4.2e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.98 [-3.63, -2.34] | 1.3e-05, 0.0004 | ahead | 79/2/20 | - | - | - | - | - | - | - | - |
| S | -18.73 [-19.36, -18.10] | 9.6e-08, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| W | -28.37 [-29.71, -26.89] | 1.5e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -6.79 [-7.67, -5.90] | 2.1e-06, 0.0004 | ahead | 92/0/7 | - | - | - | - | - | - | - | - |
| LW | -4.54 [-5.43, -3.67] | 1e-05, 0.0004 | ahead | 77/8/15 | - | - | - | - | - | - | - | - |
| SW | -15.39 [-16.12, -14.66] | 2.2e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LSW | -8.46 [-9.31, -7.63] | 1.9e-06, 0.0004 | ahead | 97/0/3 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -31.62 [-32.47, -30.67] | 3.1e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.49 [-3.04, -1.95] | 6.8e-06, 0.0004 | ahead | 75/6/19 | - | - | - | - | - | - | - | - |
| S | -19.04 [-19.53, -18.55] | 7.6e-08, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| W | -28.47 [-29.76, -27.08] | 1.1e-06, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -6.90 [-7.62, -6.17] | 1.4e-06, 0.0004 | ahead | 95/0/5 | - | - | - | - | - | - | - | - |
| LW | -3.92 [-4.71, -3.14] | 1.4e-06, 0.0004 | ahead | 73/14/13 | - | - | - | - | - | - | - | - |
| SW | -15.92 [-16.51, -15.34] | 2.1e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LSW | -8.44 [-9.22, -7.65] | 1.4e-06, 0.0004 | ahead | 97/0/3 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -32.52 [-33.39, -31.61] | 1.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -3.29 [-3.92, -2.67] | 2.3e-06, 0.0004 | ahead | 82/2/16 | - | - | - | - | - | - | - | - |
| S | -18.83 [-19.49, -18.18] | 1.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| W | -28.77 [-30.05, -27.43] | 5.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -6.97 [-7.85, -6.08] | 1.6e-06, 0.0004 | ahead | 93/0/7 | - | - | - | - | - | - | - | - |
| LW | -4.80 [-5.67, -3.93] | 2.3e-06, 0.0004 | ahead | 78/9/13 | - | - | - | - | - | - | - | - |
| SW | -15.45 [-16.18, -14.72] | 1.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LSW | -8.57 [-9.44, -7.72] | 1.8e-06, 0.0004 | ahead | 97/0/3 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -31.71 [-32.53, -30.82] | 2.1e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| L | -2.54 [-3.10, -2.02] | 3.3e-06, 0.0004 | ahead | 76/7/17 | - | - | - | - | - | - | - | - |
| S | -19.06 [-19.55, -18.56] | 9.9e-08, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| W | -28.61 [-29.84, -27.30] | 5.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LS | -6.93 [-7.65, -6.21] | 9.2e-07, 0.0004 | ahead | 95/0/5 | - | - | - | - | - | - | - | - |
| LW | -3.96 [-4.75, -3.19] | 1.2e-06, 0.0004 | ahead | 74/14/13 | - | - | - | - | - | - | - | - |
| SW | -15.93 [-16.51, -15.37] | 1.8e-07, 0.0004 | ahead | 100/0/0 | - | - | - | - | - | - | - | - |
| LSW | -8.48 [-9.27, -7.69] | 1.2e-06, 0.0004 | ahead | 97/0/3 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | MWKR (MWKR) | MWKR | -5.09 | -5.09 | 14.9 | -3.22 | 13.4 |
| L | MOR (MOR) | MOR | -3.44 | -3.44 | 12.0 | -1.47 | 12.1 |
| S | MOR (MWKR) | MWKR | -3.70 | -4.09 | 19.9 | -1.98 | 12.9 |
| W | MWKR (MWKR) | MWKR | -2.39 | -2.39 | 19.7 | -2.81 | 10.3 |
| LS | MOR (MOR) | MOR | -3.05 | -3.05 | 18.8 | -2.14 | 12.8 |
| LW | MOR (MWKR) | MOR | -2.70 | -2.70 | 13.3 | -1.83 | 10.4 |
| SW | MOR (MOR) | MWKR | -3.25 | -3.56 | 20.0 | -1.83 | 12.6 |
| LSW | MOR (MWKR) | MWKR | -3.26 | -3.84 | 22.6 | -2.65 | 12.7 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 0.1 (0.1-0.2, n=5) | 0.7 (0.4-1.3, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 1.1 (1.0-1.3, n=5) | 2.1 (1.6-2.4, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 0.9 (0.7-1.1, n=5) | 3.4 (2.2-4.1, n=4) | 44.1 (43.6-44.6, n=5) | 33.6 (31.9-35.1, n=4) |
| W | 1.2 (0.9-1.4, n=5) | 3.9 (1.8-4.9, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 1.9 (1.8-2.1, n=5) | 4.5 (2.8-5.5, n=4) | 30.4 (30.2-30.6, n=5) | 26.8 (25.5-28.0, n=4) |
| LW | 1.7 (1.5-1.8, n=5) | 3.7 (2.6-4.5, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 1.9 (1.7-2.1, n=5) | 5.0 (2.4-6.1, n=4) | 37.0 (36.3-37.4, n=5) | 29.5 (27.8-30.7, n=4) |
| LSW | 2.8 (2.4-3.0, n=5) | 5.7 (3.2-6.8, n=4) | 28.6 (28.3-28.8, n=5) | 24.6 (22.9-25.5, n=4) |
