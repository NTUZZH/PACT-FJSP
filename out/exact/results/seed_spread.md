# Spread of the mean makespan across training seeds (environment exact, training seeds 301 to 305, 306 to 310, 301 to 310)

Arrays: `test_results/FAMILY/10x25_xmask`, 100 test instances per combination at 10x25.

Definition: v_s = 100 (m_s / mean_s m_s - 1), m_s the mean makespan of seed s over the test instances; spread = standard deviation of v_s over the seeds, ddof 1; difference = per-seed mean of 100 (policy - specialist) / specialist (p88 section A); paired test of equal variances of Pitman and Morgan, t with n - 2 df, Holm over the eight combinations; RMS = root mean square over the eight combinations.

## seeds 301 to 305

| comb. | policy spread % [min, max] | specialist spread % [min, max] | difference spread % | r | p | Holm p |
|---|---|---|---|---|---|---|
| N | 0.10 [-0.16, +0.11] | 0.69 [-0.89, +0.79] | 0.63 | -0.964 | 0.0081 | 0.065 |
| L | 0.18 [-0.25, +0.20] | 0.21 [-0.25, +0.22] | 0.20 | -0.140 | 0.82 | 1 |
| S | 0.23 [-0.34, +0.25] | 0.25 [-0.24, +0.41] | 0.44 | -0.078 | 0.9 | 1 |
| W | 0.22 [-0.30, +0.30] | 0.88 [-0.77, +1.34] | 0.98 | -0.893 | 0.041 | 0.29 |
| LS | 0.26 [-0.29, +0.41] | 0.08 [-0.10, +0.11] | 0.26 | +0.846 | 0.071 | 0.43 |
| LW | 0.28 [-0.28, +0.44] | 0.15 [-0.11, +0.27] | 0.37 | +0.560 | 0.33 | 1 |
| SW | 0.11 [-0.14, +0.14] | 0.22 [-0.34, +0.24] | 0.20 | -0.620 | 0.26 | 1 |
| LSW | 0.15 [-0.22, +0.11] | 0.28 [-0.26, +0.39] | 0.27 | -0.555 | 0.33 | 1 |

Root mean square over the eight combinations: policy 0.20%, specialist 0.43%, difference 0.49%; Holm p below 0.05 on 0 of 8 combinations.

## seeds 306 to 310

| comb. | policy spread % [min, max] | specialist spread % [min, max] | difference spread % | r | p | Holm p |
|---|---|---|---|---|---|---|
| N | 0.28 [-0.26, +0.42] | 0.27 [-0.29, +0.42] | 0.55 | +0.084 | 0.89 | 1 |
| L | 0.23 [-0.40, +0.15] | 0.33 [-0.31, +0.55] | 0.29 | -0.397 | 0.51 | 1 |
| S | 0.18 [-0.28, +0.15] | 0.23 [-0.20, +0.38] | 0.39 | -0.409 | 0.49 | 1 |
| W | 0.33 [-0.48, +0.27] | 0.51 [-0.65, +0.60] | 0.68 | -0.441 | 0.46 | 1 |
| LS | 0.19 [-0.22, +0.26] | 0.26 [-0.20, +0.43] | 0.11 | -0.662 | 0.22 | 1 |
| LW | 0.26 [-0.46, +0.22] | 0.29 [-0.40, +0.40] | 0.40 | -0.086 | 0.89 | 1 |
| SW | 0.25 [-0.30, +0.30] | 0.28 [-0.31, +0.39] | 0.47 | -0.130 | 0.83 | 1 |
| LSW | 0.10 [-0.08, +0.17] | 0.18 [-0.25, +0.21] | 0.26 | -0.796 | 0.11 | 0.85 |

Root mean square over the eight combinations: policy 0.24%, specialist 0.31%, difference 0.43%; Holm p below 0.05 on 0 of 8 combinations.

## seeds 301 to 310

| comb. | policy spread % [min, max] | specialist spread % [min, max] | difference spread % | r | p | Holm p |
|---|---|---|---|---|---|---|
| N | 0.22 [-0.25, +0.51] | 0.55 [-0.66, +1.02] | 0.65 | -0.740 | 0.014 | 0.12 |
| L | 0.19 [-0.39, +0.20] | 0.26 [-0.29, +0.59] | 0.24 | -0.327 | 0.36 | 1 |
| S | 0.21 [-0.41, +0.21] | 0.26 [-0.36, +0.50] | 0.40 | -0.242 | 0.5 | 1 |
| W | 0.34 [-0.68, +0.51] | 0.70 [-0.81, +1.50] | 0.79 | -0.620 | 0.056 | 0.39 |
| LS | 0.21 [-0.27, +0.43] | 0.23 [-0.23, +0.57] | 0.25 | -0.091 | 0.8 | 1 |
| LW | 0.27 [-0.54, +0.52] | 0.28 [-0.28, +0.57] | 0.45 | -0.051 | 0.89 | 1 |
| SW | 0.19 [-0.35, +0.25] | 0.33 [-0.53, +0.46] | 0.38 | -0.513 | 0.13 | 0.65 |
| LSW | 0.12 [-0.21, +0.16] | 0.24 [-0.36, +0.31] | 0.28 | -0.601 | 0.066 | 0.4 |

Root mean square over the eight combinations: policy 0.23%, specialist 0.39%, difference 0.47%; Holm p below 0.05 on 0 of 8 combinations.

