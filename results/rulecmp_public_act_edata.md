# Comparison with dispatching rules, public family under the active mask, test set pub_Hurink_edata

every run trained and every schedule built under the active mask: the policy's sampled and greedy schedules under its run's own mask, the four rules rolled out under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask active); a trace counts the pairs the active mask admits

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.63 [-2.20, +1.03] | 1, 1 | not distinguishable | 58/1/41 | - | - | - | - | - | - | - | - |
| L | -1.73 [-3.24, -0.23] | 5.5e-05, 0.086 | not distinguishable | 62/5/33 | - | - | - | - | - | - | - | - |
| S | +1.68 [+0.60, +2.77] | 1, 1 | behind | 44/0/56 | - | - | - | - | - | - | - | - |
| W | +0.65 [-0.80, +2.08] | 1, 1 | not distinguishable | 43/5/52 | - | - | - | - | - | - | - | - |
| LS | +0.17 [-1.17, +1.50] | 1, 1 | not distinguishable | 50/0/50 | - | - | - | - | - | - | - | - |
| LW | -1.78 [-3.21, -0.38] | 0.0026, 0.051 | not distinguishable | 52/19/29 | - | - | - | - | - | - | - | - |
| SW | +2.31 [+1.22, +3.56] | 1, 1 | behind | 30/1/69 | - | - | - | - | - | - | - | - |
| LSW | -0.36 [-1.57, +0.83] | 0.17, 1 | not distinguishable | 50/2/48 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -3.59 [-5.18, -1.94] | 0.0079, 0.0004 | ahead | 76/0/24 | - | - | - | - | - | - | - | - |
| L | -5.67 [-7.47, -3.96] | 6.6e-07, 0.0004 | ahead | 75/5/20 | - | - | - | - | - | - | - | - |
| S | -2.56 [-3.87, -1.31] | 0.00083, 0.0004 | ahead | 69/1/31 | - | - | - | - | - | - | - | - |
| W | -1.81 [-3.38, -0.10] | 0.014, 0.025 | ahead | 61/7/32 | - | - | - | - | - | - | - | - |
| LS | -3.33 [-4.97, -1.74] | 0.00022, 0.0004 | ahead | 68/0/32 | - | - | - | - | - | - | - | - |
| LW | -5.54 [-7.32, -3.81] | 2.6e-05, 0.0004 | ahead | 68/17/16 | - | - | - | - | - | - | - | - |
| SW | -1.65 [-3.01, -0.22] | 0.01, 0.025 | ahead | 61/1/38 | - | - | - | - | - | - | - | - |
| LSW | -4.40 [-5.88, -2.95] | 1.6e-05, 0.0004 | ahead | 76/2/23 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -3.61 [-4.74, -2.51] | 0.00083, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| L | -3.91 [-5.28, -2.56] | 1.6e-07, 0.0004 | ahead | 74/7/19 | - | - | - | - | - | - | - | - |
| S | -1.85 [-2.82, -0.95] | 0.0024, 0.0004 | ahead | 70/1/29 | - | - | - | - | - | - | - | - |
| W | -2.76 [-3.83, -1.77] | 0.0014, 0.0004 | ahead | 70/6/24 | - | - | - | - | - | - | - | - |
| LS | -3.18 [-4.33, -2.04] | 1.4e-05, 0.0004 | ahead | 74/0/26 | - | - | - | - | - | - | - | - |
| LW | -3.85 [-5.20, -2.56] | 5e-06, 0.0004 | ahead | 63/21/16 | - | - | - | - | - | - | - | - |
| SW | -0.90 [-1.80, +0.01] | 0.014, 0.026 | not distinguishable | 60/1/39 | - | - | - | - | - | - | - | - |
| LSW | -3.36 [-4.45, -2.34] | 5.3e-05, 0.0004 | ahead | 75/2/22 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -2.91 [-3.99, -1.91] | 0.0016, 0.0004 | ahead | 78/3/19 | - | - | - | - | - | - | - | - |
| L | -3.16 [-4.38, -2.01] | 1.5e-08, 0.0004 | ahead | 67/13/19 | - | - | - | - | - | - | - | - |
| S | -1.37 [-2.23, -0.52] | 0.0071, 0.0025 | ahead | 65/2/33 | - | - | - | - | - | - | - | - |
| W | -1.94 [-3.02, -0.90] | 0.0053, 0.0009 | ahead | 64/11/25 | - | - | - | - | - | - | - | - |
| LS | -2.83 [-3.73, -1.94] | 3.1e-05, 0.0004 | ahead | 80/0/19 | - | - | - | - | - | - | - | - |
| LW | -3.28 [-4.48, -2.13] | 1e-05, 0.0004 | ahead | 61/27/12 | - | - | - | - | - | - | - | - |
| SW | -1.01 [-1.74, -0.28] | 0.0079, 0.003 | ahead | 63/2/35 | - | - | - | - | - | - | - | - |
| LSW | -2.64 [-3.41, -1.89] | 9.2e-05, 0.0004 | ahead | 79/2/19 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -3.60 [-4.81, -2.42] | 0.0016, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |
| L | -3.91 [-5.31, -2.53] | 9.9e-08, 0.0004 | ahead | 75/7/18 | - | - | - | - | - | - | - | - |
| S | -1.79 [-2.71, -0.93] | 0.0021, 0.0004 | ahead | 70/1/30 | - | - | - | - | - | - | - | - |
| W | -2.72 [-3.83, -1.70] | 0.0021, 0.0004 | ahead | 70/6/24 | - | - | - | - | - | - | - | - |
| LS | -3.10 [-4.25, -1.96] | 2.1e-05, 0.0004 | ahead | 73/0/26 | - | - | - | - | - | - | - | - |
| LW | -3.87 [-5.23, -2.55] | 1.7e-05, 0.0004 | ahead | 63/21/16 | - | - | - | - | - | - | - | - |
| SW | -0.86 [-1.75, +0.05] | 0.016, 0.034 | not distinguishable | 59/2/40 | - | - | - | - | - | - | - | - |
| LSW | -3.30 [-4.36, -2.30] | 1.6e-05, 0.0004 | ahead | 75/2/22 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -2.89 [-3.98, -1.89] | 0.0016, 0.0004 | ahead | 78/3/19 | - | - | - | - | - | - | - | - |
| L | -3.13 [-4.36, -1.98] | 1.4e-08, 0.0004 | ahead | 68/13/19 | - | - | - | - | - | - | - | - |
| S | -1.35 [-2.20, -0.52] | 0.0065, 0.002 | ahead | 65/2/33 | - | - | - | - | - | - | - | - |
| W | -1.94 [-3.03, -0.89] | 0.0065, 0.001 | ahead | 64/10/26 | - | - | - | - | - | - | - | - |
| LS | -2.77 [-3.68, -1.88] | 3.8e-05, 0.0004 | ahead | 80/0/19 | - | - | - | - | - | - | - | - |
| LW | -3.27 [-4.47, -2.11] | 1.6e-05, 0.0004 | ahead | 61/27/13 | - | - | - | - | - | - | - | - |
| SW | -0.98 [-1.71, -0.25] | 0.0084, 0.0041 | ahead | 63/2/36 | - | - | - | - | - | - | - | - |
| LSW | -2.61 [-3.37, -1.86] | 7.4e-05, 0.0004 | ahead | 78/2/20 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | MWKR (MWKR) | MWKR | -2.95 | -2.95 | 26.3 | -2.95 | 14.3 |
| L | MOR (MOR) | FIFO | -4.91 | -3.98 | 22.3 | -2.20 | 13.4 |
| S | MOR (MWKR) | FIFO | -5.13 | -4.12 | 21.5 | -2.69 | 14.8 |
| W | MWKR (MWKR) | MWKR | -2.40 | -2.40 | 24.1 | -2.96 | 11.1 |
| LS | MOR (MOR) | MOR | -3.46 | -3.46 | 20.1 | -2.45 | 14.9 |
| LW | MOR (MWKR) | MWKR | -3.63 | -3.81 | 19.9 | -2.18 | 10.7 |
| SW | MOR (MOR) | MWKR | -2.25 | -3.84 | 20.5 | -1.97 | 14.1 |
| LSW | MOR (MWKR) | MWKR | -3.69 | -4.05 | 19.4 | -2.75 | 14.0 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 4.4 (3.5-4.8, n=5) | 6.8 (5.3-8.9, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 7.5 (6.4-8.1, n=5) | 10.7 (8.7-13.5, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 8.1 (6.7-8.6, n=5) | 10.1 (7.3-12.5, n=4) | 25.6 (25.5-25.7, n=5) | 23.9 (21.8-24.8, n=4) |
| W | 6.5 (5.2-7.1, n=5) | 9.2 (6.7-11.2, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 10.3 (9.4-10.9, n=5) | 13.2 (10.2-16.2, n=4) | 21.7 (21.5-22.0, n=5) | 19.9 (18.4-20.6, n=4) |
| LW | 9.2 (8.2-9.8, n=5) | 12.3 (9.1-15.1, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 10.2 (8.6-10.8, n=5) | 12.3 (9.0-14.5, n=4) | 21.7 (21.6-21.8, n=5) | 20.5 (18.9-21.4, n=4) |
| LSW | 12.3 (11.0-13.2, n=5) | 14.6 (11.1-17.1, n=4) | 19.4 (19.3-19.5, n=5) | 17.8 (16.7-18.3, n=4) |
