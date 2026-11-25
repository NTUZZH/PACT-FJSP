# Principal comparison with ten training seeds (environment exact, training seeds 301 to 310 (10 runs per arm), principal comparison only)

The extension rule: If, with five seeds, the interval of any single combination of the principal comparison (policy against specialist at 10x25) is inconclusive at the 1% margin, seeds 306 to 310 of the policy and the eight specialists are trained and every verdict is recomputed with ten seeds. The verdict rule does not change.

The rule applied in the five-seed record `out/exact/results/family_inference.json`: inconclusive at the 1% margin with seeds 301 to 305: W.

The ten-seed intervals are not adjusted for the look at the five-seed results that decided the extension.

Estimator and verdicts: section A of p88 (hierarchical bootstrap over training seeds and base instances, B = 20000, verdicts at +-0.5% and +-1% from the 95% interval); ten-seed draws from default_rng(110). Reproduction of the five-seed section A from the arrays: 212 numbers and words identical at the printed precision. Validation norms of the ten policy runs equal.

| comb. | ten seeds: mean | 95% CI | 0.5% | 1% | direction | five seeds (stored): mean | 95% CI | 0.5% | 1% | direction | first five | second five |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.56 | [+0.04, +1.07] | inferior | inferior | behind | +0.24 | [-0.40, +0.86] | inconclusive | equivalent | neither | +0.24 | +0.88 |
| L | +0.15 | [-0.09, +0.37] | equivalent | equivalent | neither | +0.18 | [-0.11, +0.49] | equivalent | equivalent | neither | +0.18 | +0.11 |
| S | -0.40 | [-0.76, -0.05] | non-inferior only | equivalent | ahead | -0.35 | [-0.80, +0.06] | non-inferior only | equivalent | neither | -0.35 | -0.44 |
| W | +0.34 | [-0.26, +0.97] | inconclusive | equivalent | neither | +0.40 | [-0.51, +1.31] | inconclusive | inconclusive | neither | +0.40 | +0.29 |
| LS | -0.23 | [-0.49, +0.02] | equivalent | equivalent | neither | -0.07 | [-0.38, +0.25] | equivalent | equivalent | neither | -0.07 | -0.39 |
| LW | +0.20 | [-0.14, +0.55] | inconclusive | equivalent | neither | +0.46 | [+0.06, +0.88] | inferior | equivalent | behind | +0.46 | -0.06 |
| SW | -0.31 | [-0.65, +0.06] | non-inferior only | equivalent | neither | -0.47 | [-0.89, -0.08] | non-inferior only | equivalent | ahead | -0.47 | -0.14 |
| LSW | -0.26 | [-0.55, +0.02] | non-inferior only | equivalent | neither | -0.15 | [-0.53, +0.22] | non-inferior only | equivalent | neither | -0.15 | -0.36 |

Per-seed means, ten seeds (policy minus specialist over specialist, %):

| comb. | s301 | s302 | s303 | s304 | s305 | s306 | s307 | s308 | s309 | s310 |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.41 | -0.49 | +0.32 | -0.19 | +1.16 | +0.72 | +1.22 | +0.16 | +0.71 | +1.59 |
| L | -0.00 | +0.18 | +0.09 | +0.51 | +0.14 | +0.40 | -0.35 | +0.02 | +0.15 | +0.31 |
| S | -0.25 | +0.02 | -0.10 | -1.10 | -0.33 | -0.35 | -0.37 | -0.14 | -1.13 | -0.23 |
| W | -0.96 | +0.99 | +1.06 | -0.32 | +1.21 | +0.36 | +0.64 | -0.75 | +0.16 | +1.07 |
| LS | -0.36 | -0.24 | +0.32 | +0.01 | -0.09 | -0.44 | -0.55 | -0.33 | -0.28 | -0.38 |
| LW | +0.22 | +0.44 | +0.06 | +1.01 | +0.57 | +0.05 | +0.12 | -0.51 | -0.40 | +0.46 |
| SW | -0.62 | -0.61 | -0.15 | -0.55 | -0.42 | -0.84 | -0.02 | -0.16 | -0.18 | +0.49 |
| LSW | +0.02 | -0.18 | +0.22 | -0.41 | -0.39 | -0.33 | +0.04 | -0.62 | -0.33 | -0.58 |

Joint verdicts (intersection-union test):

| family | margin | equivalent on all (ten / five) | non-inferior on all (ten / five) | failing, equivalent (ten / five) |
|---|---|---|---|---|
| All | 0.5 | False / False | False / False | N,S,W,LW,SW,LSW / N,S,W,LW,SW,LSW |
| All | 1.0 | False / False | False / False | N / W |
| Train | 0.5 | False / False | False / False | N,S,W,LW / N,S,W,LW |
| Train | 1.0 | False / False | False / False | N / W |
| Held | 0.5 | False / False | True / True | SW,LSW / SW,LSW |
| Held | 1.0 | True / True | True / True | - / - |

Pooled rows (hierarchical; seed-pooled instance-level):

| family | ten: mean | CI | 0.5% | 1% | instance-level CI | five: mean | CI | 0.5% | 1% | instance-level CI |
|---|---|---|---|---|---|---|---|---|---|---|
| All | +0.01 | [-0.17, +0.19] | equivalent | equivalent | [-0.14, +0.08] | +0.03 | [-0.18, +0.24] | equivalent | equivalent | [-0.12, +0.12] |
| Train | +0.10 | [-0.09, +0.31] | equivalent | equivalent | [-0.06, +0.19] | +0.14 | [-0.10, +0.39] | equivalent | equivalent | [-0.01, +0.24] |
| Held | -0.28 | [-0.50, -0.05] | non-inferior only | equivalent | [-0.47, -0.17] | -0.31 | [-0.60, -0.01] | non-inferior only | equivalent | [-0.54, -0.15] |

Ten seeds at the level of the two readings (97.5%, the same draws as the 95% intervals):

| comb. | mean | 95% CI | 1% | 97.5% CI | 1% | 0.5% |
|---|---|---|---|---|---|---|
| N | +0.56 | [+0.04, +1.07] | inferior | [-0.04, +1.15] | inconclusive | inconclusive |
| L | +0.15 | [-0.09, +0.37] | equivalent | [-0.13, +0.40] | equivalent | equivalent |
| S | -0.40 | [-0.76, -0.05] | equivalent | [-0.81, +0.01] | equivalent | non-inferior only |
| W | +0.34 | [-0.26, +0.97] | equivalent | [-0.35, +1.05] | inconclusive | inconclusive |
| LS | -0.23 | [-0.49, +0.02] | equivalent | [-0.53, +0.05] | equivalent | non-inferior only |
| LW | +0.20 | [-0.14, +0.55] | equivalent | [-0.19, +0.61] | equivalent | inconclusive |
| SW | -0.31 | [-0.65, +0.06] | equivalent | [-0.70, +0.12] | equivalent | non-inferior only |
| LSW | -0.26 | [-0.55, +0.02] | equivalent | [-0.60, +0.06] | equivalent | non-inferior only |
| pooled All | +0.01 | [-0.17, +0.19] | equivalent | [-0.19, +0.21] | equivalent | equivalent |
| pooled Train | +0.10 | [-0.09, +0.31] | equivalent | [-0.13, +0.34] | equivalent | equivalent |
| pooled Held | -0.28 | [-0.50, -0.05] | equivalent | [-0.54, -0.02] | equivalent | non-inferior only |

The comparison has two readings (five seeds, then ten seeds), so intervals at level 1 - alpha/2 = 97.5% from the same ten-seed draws bound the one-sided error of a verdict over both readings by alpha/2 = 0.025 (union bound), which is the one-sided level of a single 95% reading.

Combinations whose verdict differs between 95% and 97.5%: N, W at the 1% margin and N, LS at the 0.5% margin.
