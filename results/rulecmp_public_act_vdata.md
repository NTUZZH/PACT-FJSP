# Comparison with dispatching rules, public family under the active mask, test set pub_Hurink_vdata

every run trained and every schedule built under the active mask: the policy's sampled and greedy schedules under its run's own mask, the four rules rolled out under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask active); a trace counts the pairs the active mask admits

d = 100 (a - b) / b per training seed and test instance; crossed hierarchical bootstrap over seeds and instances (stats_family.hier_boot, B = 20000, fresh numpy default_rng(11) per quantity), 95% percentile interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) and one-sided bootstrap p, Holm over the combinations of each arm and form (the stats_family calls of p89_public_family.component, stats_family.holm)

Test instances: 65; seeds [301, 302, 303, 304, 305]; K = 32; tie-break seeds [50, 51, 52, 53].

## GreedyBestFour: one greedy schedule vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +5.70 [+3.55, +8.05] | 1, 1 | behind | 20/0/80 | - | - | - | - | - | - | - | - |
| L | -1.80 [-3.08, -0.63] | 0.0005, 0.0077 | ahead | 47/15/38 | - | - | - | - | - | - | - | - |
| S | +2.63 [+1.18, +4.21] | 1, 1 | not distinguishable | 26/1/73 | - | - | - | - | - | - | - | - |
| W | +5.45 [+3.53, +7.62] | 1, 1 | behind | 15/5/79 | - | - | - | - | - | - | - | - |
| LS | -1.56 [-2.62, -0.51] | 0.0065, 0.012 | ahead | 60/1/39 | - | - | - | - | - | - | - | - |
| LW | -2.37 [-3.89, -0.95] | 0.0011, 0.0024 | ahead | 56/6/39 | - | - | - | - | - | - | - | - |
| SW | +3.77 [+2.30, +5.39] | 1, 1 | behind | 15/1/84 | - | - | - | - | - | - | - | - |
| LSW | -1.23 [-2.42, -0.06] | 0.002, 0.099 | not distinguishable | 58/0/42 | - | - | - | - | - | - | - | - |

## GreedyValiRule: one greedy schedule vs the rule with the lowest validation mean

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +5.36 [+3.21, +7.71] | 1, 1 | behind | 22/0/78 | - | - | - | - | - | - | - | - |
| L | -3.59 [-4.85, -2.39] | 3e-05, 0.0004 | ahead | 69/11/20 | - | - | - | - | - | - | - | - |
| S | -1.59 [-3.18, +0.09] | 0.17, 0.12 | not distinguishable | 61/1/38 | - | - | - | - | - | - | - | - |
| W | +4.42 [+2.51, +6.59] | 1, 1 | behind | 19/10/71 | - | - | - | - | - | - | - | - |
| LS | -3.54 [-4.93, -2.24] | 0.00024, 0.0004 | ahead | 72/1/27 | - | - | - | - | - | - | - | - |
| LW | -4.31 [-5.95, -2.80] | 8.3e-05, 0.0004 | ahead | 72/6/22 | - | - | - | - | - | - | - | - |
| SW | +0.59 [-1.06, +2.32] | 1, 1 | not distinguishable | 43/1/56 | - | - | - | - | - | - | - | - |
| LSW | -3.83 [-5.09, -2.59] | 3.3e-05, 0.0004 | ahead | 73/0/27 | - | - | - | - | - | - | - | - |

## SampleFour: best of 4 sampled vs the per-instance best of the four rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.42 [+0.79, +4.20] | 1, 1 | not distinguishable | 33/1/66 | - | - | - | - | - | - | - | - |
| L | -2.96 [-4.22, -1.82] | 5.9e-06, 0.0004 | ahead | 62/22/17 | - | - | - | - | - | - | - | - |
| S | +0.62 [-0.60, +2.04] | 1, 1 | not distinguishable | 41/1/58 | - | - | - | - | - | - | - | - |
| W | +2.60 [+0.95, +4.51] | 1, 1 | not distinguishable | 28/6/66 | - | - | - | - | - | - | - | - |
| LS | -3.68 [-4.64, -2.76] | 5.9e-06, 0.0004 | ahead | 80/1/18 | - | - | - | - | - | - | - | - |
| LW | -4.03 [-5.46, -2.75] | 1.1e-05, 0.0004 | ahead | 76/6/18 | - | - | - | - | - | - | - | - |
| SW | +1.30 [+0.08, +2.71] | 1, 1 | not distinguishable | 35/1/64 | - | - | - | - | - | - | - | - |
| LSW | -3.47 [-4.52, -2.44] | 3.6e-05, 0.0004 | ahead | 79/1/20 | - | - | - | - | - | - | - | - |

## SampleSixteen: best of 16 sampled vs the per-instance best of 16 rule schedules (four rules x tie-break seeds 50-53)

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.51 [+1.16, +4.06] | 1, 1 | not distinguishable | 28/3/69 | - | - | - | - | - | - | - | - |
| L | -2.31 [-3.29, -1.42] | 6e-06, 0.0004 | ahead | 47/46/7 | - | - | - | - | - | - | - | - |
| S | +1.25 [+0.18, +2.51] | 1, 1 | not distinguishable | 31/2/67 | - | - | - | - | - | - | - | - |
| W | +2.99 [+1.41, +4.85] | 1, 1 | not distinguishable | 21/4/75 | - | - | - | - | - | - | - | - |
| LS | -2.80 [-3.59, -2.04] | 2.4e-05, 0.0004 | ahead | 81/2/17 | - | - | - | - | - | - | - | - |
| LW | -3.17 [-4.27, -2.14] | 1.5e-05, 0.0004 | ahead | 69/21/10 | - | - | - | - | - | - | - | - |
| SW | +1.55 [+0.42, +2.85] | 1, 1 | not distinguishable | 28/1/70 | - | - | - | - | - | - | - | - |
| LSW | -3.10 [-4.03, -2.22] | 6.2e-05, 0.0004 | ahead | 79/2/19 | - | - | - | - | - | - | - | - |

## GreedyPlusThree: greedy plus 3 sampled vs the per-instance best of the four rules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.46 [+0.78, +4.30] | 1, 1 | not distinguishable | 33/1/66 | - | - | - | - | - | - | - | - |
| L | -2.98 [-4.25, -1.84] | 2.2e-06, 0.0004 | ahead | 62/22/16 | - | - | - | - | - | - | - | - |
| S | +0.37 [-0.91, +1.81] | 1, 1 | not distinguishable | 44/1/55 | - | - | - | - | - | - | - | - |
| W | +2.58 [+0.95, +4.50] | 1, 1 | not distinguishable | 30/6/64 | - | - | - | - | - | - | - | - |
| LS | -3.76 [-4.73, -2.84] | 7.1e-06, 0.0004 | ahead | 80/1/18 | - | - | - | - | - | - | - | - |
| LW | -4.10 [-5.53, -2.82] | 8.2e-06, 0.0004 | ahead | 76/7/17 | - | - | - | - | - | - | - | - |
| SW | +1.28 [+0.02, +2.73] | 1, 1 | not distinguishable | 34/1/64 | - | - | - | - | - | - | - | - |
| LSW | -3.68 [-4.71, -2.68] | 1.3e-05, 0.0004 | ahead | 80/1/19 | - | - | - | - | - | - | - | - |

## GreedyPlusFifteen: greedy plus 15 sampled vs the per-instance best of 16 rule schedules

| comb. | policy d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | single-rule control d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % | specialist of the combination d [95% CI] | Holm p shorter (t, boot) | verdict | short/eq/long % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.53 [+1.17, +4.08] | 1, 1 | not distinguishable | 28/3/69 | - | - | - | - | - | - | - | - |
| L | -2.31 [-3.28, -1.42] | 5.5e-06, 0.0004 | ahead | 47/46/7 | - | - | - | - | - | - | - | - |
| S | +1.18 [+0.08, +2.46] | 1, 1 | not distinguishable | 32/2/66 | - | - | - | - | - | - | - | - |
| W | +3.01 [+1.43, +4.87] | 1, 1 | not distinguishable | 21/5/75 | - | - | - | - | - | - | - | - |
| LS | -2.82 [-3.62, -2.06] | 2.3e-05, 0.0004 | ahead | 81/2/17 | - | - | - | - | - | - | - | - |
| LW | -3.17 [-4.28, -2.15] | 1.5e-05, 0.0004 | ahead | 69/21/10 | - | - | - | - | - | - | - | - |
| SW | +1.57 [+0.43, +2.89] | 1, 1 | not distinguishable | 28/1/71 | - | - | - | - | - | - | - | - |
| LSW | -3.17 [-4.10, -2.30] | 4.2e-05, 0.0004 | ahead | 79/2/19 | - | - | - | - | - | - | - | - |

## Rules

| comb. | validation rule (all records) | lowest test mean | best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | distinct of 16 |
|---|---|---|---|---|---|---|---|
| N | MWKR (MWKR) | MWKR | -0.32 | -0.32 | 42.8 | -1.88 | 15.4 |
| L | MOR (MOR) | MOR | -1.79 | -1.79 | 16.4 | -1.30 | 14.2 |
| S | MOR (MWKR) | MWKR | -1.36 | -4.05 | 35.5 | -2.33 | 15.6 |
| W | MWKR (MWKR) | MWKR | -0.93 | -0.93 | 43.1 | -2.31 | 14.1 |
| LS | MOR (MOR) | MOR | -2.03 | -2.03 | 19.9 | -2.35 | 15.5 |
| LW | MOR (MWKR) | MOR | -1.98 | -1.98 | 18.8 | -2.00 | 13.7 |
| SW | MOR (MOR) | MWKR | -1.89 | -3.05 | 32.3 | -2.02 | 15.3 |
| LSW | MOR (MWKR) | MOR | -2.58 | -2.58 | 20.7 | -2.21 | 15.5 |

## Traces (percent)

| comb. | one pair, policy (seeds) | one pair, rules | setup share, policy | setup share, rules |
|---|---|---|---|---|
| N | 0.6 (0.5-0.7, n=5) | 1.6 (0.8-2.6, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| L | 0.9 (0.7-1.0, n=5) | 1.8 (1.0-2.9, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| S | 1.7 (0.9-2.4, n=5) | 3.2 (1.6-4.9, n=4) | 31.5 (30.6-32.5, n=5) | 28.2 (23.1-30.1, n=4) |
| W | 2.2 (1.7-2.6, n=5) | 3.4 (1.1-5.0, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| LS | 2.1 (1.3-2.5, n=5) | 3.7 (1.7-5.3, n=4) | 23.8 (23.2-24.1, n=5) | 21.0 (19.0-22.0, n=4) |
| LW | 2.1 (1.6-2.7, n=5) | 3.3 (1.5-4.8, n=4) | 0.0 (0.0-0.0, n=5) | 0.0 (0.0-0.0, n=4) |
| SW | 2.9 (1.9-3.6, n=5) | 4.3 (2.0-6.2, n=4) | 26.8 (26.4-27.2, n=5) | 24.5 (20.5-25.9, n=4) |
| LSW | 3.3 (2.5-3.7, n=5) | 5.0 (2.3-6.9, n=4) | 22.0 (21.6-22.2, n=5) | 19.5 (17.6-20.4, n=4) |
