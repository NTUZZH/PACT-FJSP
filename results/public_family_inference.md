# Public-instance family: policy, control, specialists and rules

Training seeds 301, 302, 303, 304, 305 per arm, exact mask, greedy decoding. d = 100 (a - b) / b per training seed and base instance, seed s against seed s; positive = first arm slower. Hierarchical 95% intervals (seeds x base instances, B = 20000, default_rng(11)), seed-level t intervals (4 df), verdicts at 0.5% / 1% on the hierarchical interval, intersection-union test for joint claims, Holm over the eight combinations for superiority over the rules.

| set | role | base instances |
|---|---|---|
| pub_SD2_test_10x5+mix | in distribution | 100 |
| pub_SD2_test_20x10+mix | size transfer | 100 |
| pub_Brandimarte | distribution transfer | 10 |
| pub_Hurink_edata | distribution transfer | 65 |
| pub_Hurink_rdata | distribution transfer | 65 |
| pub_Hurink_vdata | distribution transfer | 65 |

## Checks

- pub_SD2_test_10x5+mix: 100 records in one sorted order in all eight regimes (excluded: none; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 96 instances; duplicate rule files: 0.
- pub_SD2_test_20x10+mix: 100 records in one sorted order in all eight regimes (excluded: none; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 1 instances; duplicate rule files: 0.
- pub_Brandimarte: 10 records in one sorted order in all eight regimes (excluded: none; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 10 instances; duplicate rule files: 0.
- pub_Hurink_edata: 65 records in one sorted order in all eight regimes (excluded: orb7; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 65 instances; duplicate rule files: 0.
- pub_Hurink_rdata: 65 records in one sorted order in all eight regimes (excluded: orb7; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 65 instances; duplicate rule files: 0.
- pub_Hurink_vdata: 65 records in one sorted order in all eight regimes (excluded: orb7; lag scale 1.0); 51 summary files with n/n feasible, matching means and matching regime-N gaps; 19 regime-N arrays at or above the bound of 65 instances; duplicate rule files: 0.

## 1. Policy vs specialists

### pub_SD2_test_10x5+mix (in distribution)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.12 | [+0.37, +3.97] | [-0.19, +4.43] | inferior | inferior | inconclusive | inconclusive | behind | 1.0 | +1.20, +2.32, -0.05, +2.15, +4.99 |
| L | +0.10 | [-0.63, +0.82] | [-0.54, +0.75] | inconclusive | equivalent | inconclusive | equivalent | neither | 16.0 | +0.13, -0.80, +0.37, +0.47, +0.35 |
| S | +0.15 | [-1.24, +1.60] | [-0.98, +1.28] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 19.4 | -0.71, +1.56, +0.43, -0.57, +0.05 |
| W | +0.73 | [-0.62, +2.13] | [-0.33, +1.79] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 15.8 | +2.01, -0.39, +0.62, +0.78, +0.63 |
| LS | -0.50 | [-1.63, +0.77] | [-1.54, +0.53] | inconclusive | non-inferior only | inconclusive | non-inferior only | neither | 25.6 | -1.16, +0.49, -0.02, -0.29, -1.54 |
| LW | -0.18 | [-0.86, +0.49] | [-0.46, +0.09] | non-inferior only | equivalent | equivalent | equivalent | neither | 26.6 | -0.08, -0.26, -0.36, +0.15, -0.37 |
| SW | +0.12 | [-1.34, +1.56] | [-1.10, +1.34] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 21.2 | +1.09, +0.03, -0.26, +1.02, -1.27 |
| LSW | +0.41 | [-1.39, +2.21] | [-1.51, +2.34] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 22.4 | +0.24, +2.21, -1.97, +0.37, +1.21 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | +0.37 | [-0.11, +0.83] | inconclusive | equivalent | False / False | False / False | N,S,W,LS,SW,LSW |
| Train | +0.40 | [-0.04, +0.84] | inconclusive | equivalent | False / False | False / False | N,S,W,LS |
| Held | +0.27 | [-0.87, +1.39] | inconclusive | inconclusive | False / False | False / False | SW,LSW |

### pub_SD2_test_20x10+mix (size transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +3.73 | [+2.23, +5.49] | [+1.41, +6.04] | inferior | inferior | inferior | inferior | behind | 0.6 | +2.24, +3.06, +4.02, +2.49, +6.83 |
| L | -0.12 | [-0.81, +0.55] | [-0.71, +0.47] | inconclusive | equivalent | non-inferior only | equivalent | neither | 1.0 | +0.42, -0.67, -0.53, -0.06, +0.25 |
| S | -0.21 | [-1.20, +0.84] | [-0.92, +0.50] | inconclusive | non-inferior only | non-inferior only | equivalent | neither | 0.2 | -0.47, -0.49, -0.28, +0.79, -0.61 |
| W | -0.25 | [-2.72, +1.75] | [-3.51, +3.02] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 14.6 | +0.15, -4.71, +1.70, -0.05, +1.68 |
| LS | -0.15 | [-1.41, +1.16] | [-1.02, +0.73] | inconclusive | inconclusive | inconclusive | non-inferior only | neither | 0.2 | -1.00, +0.35, +0.72, -0.14, -0.66 |
| LW | +0.25 | [-0.44, +0.93] | [-0.30, +0.81] | inconclusive | equivalent | inconclusive | equivalent | neither | 5.8 | +0.56, +0.24, +0.82, -0.11, -0.25 |
| SW | +0.02 | [-1.11, +1.23] | [-0.95, +1.00] | inconclusive | inconclusive | inconclusive | equivalent | neither | 0.4 | -0.10, -0.08, -0.53, -0.54, +1.37 |
| LSW | -0.08 | [-1.57, +1.42] | [-1.36, +1.20] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.4 | +0.56, +0.87, -1.00, +0.54, -1.38 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | +0.40 | [-0.06, +0.88] | inconclusive | equivalent | False / False | False / False | N,S,W,LS,SW,LSW |
| Train | +0.54 | [-0.07, +1.15] | inconclusive | inconclusive | False / False | False / False | N,S,W,LS |
| Held | -0.03 | [-0.90, +0.84] | inconclusive | equivalent | False / False | False / False | SW,LSW |

### pub_Brandimarte (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.28 | [-1.48, +2.30] | [-1.46, +2.03] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 26.0 | +0.16, -2.03, +1.66, +1.00, +0.63 |
| L | -3.73 | [-9.35, +0.49] | [-9.06, +1.60] | non-inferior only | non-inferior only | inconclusive | inconclusive | neither | 4.0 | -3.96, -2.45, -10.51, -3.04, +1.32 |
| S | -0.58 | [-5.25, +4.77] | [-6.49, +5.34] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 6.0 | +3.94, -0.68, +4.48, -5.52, -5.10 |
| W | -1.42 | [-2.97, -0.12] | [-2.48, -0.36] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 38.0 | -2.06, -2.27, -1.75, -0.62, -0.40 |
| LS | -0.36 | [-4.19, +3.32] | [-2.10, +1.38] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.0 | +0.74, -2.73, -0.21, +0.66, -0.27 |
| LW | +0.18 | [-3.33, +3.79] | [-2.34, +2.70] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 10.0 | +2.31, +2.00, -2.16, -1.59, +0.32 |
| SW | +1.81 | [-1.56, +5.03] | [+0.71, +2.92] | inconclusive | inconclusive | inferior | inferior | neither | 4.0 | +1.74, +1.80, +1.99, +3.03, +0.52 |
| LSW | -1.19 | [-4.40, +1.83] | [-3.55, +1.18] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.0 | -3.49, -1.42, +1.20, +0.20, -2.42 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.62 | [-2.08, +0.92] | inconclusive | non-inferior only | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.94 | [-2.67, +0.82] | inconclusive | non-inferior only | False / False | False / False | N,L,S,W,LS,LW |
| Held | +0.31 | [-2.26, +2.71] | inconclusive | inconclusive | False / False | False / False | SW,LSW |

### pub_Hurink_edata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.29 | [-0.68, +1.23] | [-0.70, +1.27] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 4.9 | +1.18, -0.91, +0.02, +0.74, +0.40 |
| L | -0.52 | [-1.64, +0.48] | [-1.66, +0.62] | non-inferior only | non-inferior only | inconclusive | non-inferior only | neither | 7.4 | +0.23, -0.29, -1.63, -1.34, +0.41 |
| S | -0.67 | [-1.47, +0.12] | [-1.20, -0.13] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 18.8 | +0.02, -0.73, -0.66, -1.17, -0.78 |
| W | -1.27 | [-2.42, -0.19] | [-2.07, -0.47] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 14.2 | -1.89, -2.02, -0.64, -1.05, -0.74 |
| LS | -0.76 | [-1.69, +0.15] | [-1.27, -0.25] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 18.8 | -1.31, -0.24, -0.67, -1.00, -0.56 |
| LW | -0.53 | [-1.52, +0.49] | [-1.58, +0.51] | non-inferior only | non-inferior only | inconclusive | non-inferior only | neither | 29.2 | +0.90, -0.76, -0.56, -1.29, -0.94 |
| SW | -0.08 | [-1.03, +0.96] | [-1.17, +1.00] | inconclusive | non-inferior only | inconclusive | inconclusive | neither | 18.2 | -0.49, +1.32, +0.14, -0.96, -0.43 |
| LSW | +0.78 | [-0.27, +1.95] | [-0.22, +1.78] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 19.1 | +0.11, +0.57, +1.83, +1.39, -0.00 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.35 | [-0.67, -0.00] | non-inferior only | equivalent | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.58 | [-0.98, -0.15] | non-inferior only | equivalent | False / False | False / False | N,L,S,W,LS,LW |
| Held | +0.35 | [-0.37, +1.10] | inconclusive | inconclusive | False / False | False / False | SW,LSW |

### pub_Hurink_rdata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.51 | [-0.37, +1.43] | [-0.21, +1.22] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.8 | +0.66, -0.01, +1.42, +0.37, +0.10 |
| L | -1.11 | [-2.40, +0.06] | [-2.48, +0.26] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 6.5 | -1.42, -0.11, -1.98, -2.24, +0.21 |
| S | +0.07 | [-1.27, +1.33] | [-0.73, +0.86] | inconclusive | inconclusive | inconclusive | equivalent | neither | 6.8 | +0.63, -0.07, -0.99, +0.41, +0.35 |
| W | +0.11 | [-1.09, +1.25] | [-1.17, +1.39] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 10.5 | +0.68, -1.39, +1.34, -0.28, +0.20 |
| LS | +0.21 | [-1.10, +1.56] | [-0.58, +1.01] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 5.2 | -0.32, +0.42, +0.59, -0.57, +0.96 |
| LW | -0.52 | [-1.71, +0.62] | [-1.66, +0.62] | inconclusive | non-inferior only | inconclusive | non-inferior only | neither | 23.1 | -0.89, +0.94, -0.20, -1.20, -1.24 |
| SW | -0.46 | [-1.91, +0.97] | [-1.81, +0.89] | inconclusive | non-inferior only | inconclusive | non-inferior only | neither | 7.4 | +1.15, -0.29, -1.88, -0.55, -0.73 |
| LSW | +0.23 | [-1.25, +1.67] | [-0.69, +1.14] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 8.6 | +1.06, +0.42, +0.72, -0.66, -0.41 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.12 | [-0.59, +0.36] | non-inferior only | equivalent | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.12 | [-0.61, +0.35] | non-inferior only | equivalent | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.12 | [-1.21, +1.04] | inconclusive | inconclusive | False / False | False / False | SW,LSW |

### pub_Hurink_vdata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.32 | [-0.49, +1.30] | [-0.83, +1.48] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.2 | -0.19, -0.47, +1.89, +0.42, -0.03 |
| L | -0.61 | [-1.63, +0.23] | [-1.16, -0.06] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 25.8 | -1.26, -0.53, -0.56, -0.68, -0.01 |
| S | -0.60 | [-1.96, +0.72] | [-1.31, +0.12] | inconclusive | non-inferior only | non-inferior only | non-inferior only | neither | 1.5 | +0.15, -1.18, -0.23, -1.13, -0.60 |
| W | +0.63 | [-1.01, +2.10] | [-1.41, +2.67] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 8.6 | +1.15, -1.71, +2.76, -0.00, +0.95 |
| LS | -0.02 | [-1.53, +1.66] | [-1.86, +1.83] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 1.5 | +2.55, -0.91, -0.18, -1.13, -0.43 |
| LW | +0.09 | [-0.87, +1.02] | [-0.96, +1.13] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 21.8 | +0.64, -0.23, +1.16, -1.03, -0.10 |
| SW | -1.24 | [-2.86, +0.41] | [-2.95, +0.48] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 3.1 | -1.45, -2.49, -0.31, +0.61, -2.55 |
| LSW | +0.40 | [-0.86, +1.68] | [-0.59, +1.40] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 5.5 | +0.55, +1.09, +1.17, -0.73, -0.07 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.13 | [-0.70, +0.48] | non-inferior only | equivalent | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.03 | [-0.69, +0.64] | inconclusive | equivalent | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.42 | [-1.36, +0.51] | inconclusive | non-inferior only | False / False | False / False | SW,LSW |

## 2. Control on the combinations it never saw

LS and LW are seen by the policy and not by the control; SW and LSW by neither.

### pub_SD2_test_10x5+mix

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.50 [-1.63, +0.77] | non-inferior only | +0.14 [-1.13, +1.54] | inconclusive | inconclusive | -0.46 [-1.33, +0.43] | non-inferior only | non-inferior only |
| LW | -0.18 [-0.86, +0.49] | equivalent | +0.22 [-0.53, +0.96] | inconclusive | equivalent | -0.28 [-0.90, +0.31] | non-inferior only | equivalent |
| SW | +0.12 [-1.34, +1.56] | inconclusive | +0.29 [-1.01, +1.64] | inconclusive | inconclusive | +0.02 [-0.96, +1.03] | inconclusive | inconclusive |
| LSW | +0.41 [-1.39, +2.21] | inconclusive | +0.35 [-1.04, +1.72] | inconclusive | inconclusive | +0.32 [-0.88, +1.51] | inconclusive | inconclusive |
| pooled Pair | | | +0.18 [-0.62, +1.01] | inconclusive | inconclusive | -0.37 [-0.89, +0.13] | non-inferior only | equivalent |
| pooled Held | | | +0.32 [-0.63, +1.29] | inconclusive | inconclusive | +0.17 [-0.60, +0.91] | inconclusive | equivalent |
| pooled Unseen | | | +0.25 [-0.42, +0.89] | inconclusive | equivalent | -0.10 [-0.61, +0.34] | non-inferior only | equivalent |

Control on its own combinations (control vs specialist): N +1.00 [-0.49, +2.41], L +0.48 [-0.31, +1.29], S -0.36 [-1.52, +0.91], W +0.36 [-1.03, +1.90]

### pub_SD2_test_20x10+mix

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.15 [-1.41, +1.16] | inconclusive | +0.36 [-1.09, +1.87] | inconclusive | inconclusive | -0.02 [-1.27, +1.27] | inconclusive | inconclusive |
| LW | +0.25 [-0.44, +0.93] | equivalent | +0.34 [-0.83, +1.63] | inconclusive | inconclusive | +0.05 [-0.97, +1.00] | inconclusive | equivalent |
| SW | +0.02 [-1.11, +1.23] | inconclusive | +0.00 [-1.36, +1.40] | inconclusive | inconclusive | +0.34 [-0.87, +1.63] | inconclusive | inconclusive |
| LSW | -0.08 [-1.57, +1.42] | inconclusive | +0.23 [-1.12, +1.66] | inconclusive | inconclusive | +0.12 [-1.42, +1.71] | inconclusive | inconclusive |
| pooled Pair | | | +0.35 [-0.66, +1.33] | inconclusive | inconclusive | +0.02 [-0.70, +0.74] | inconclusive | equivalent |
| pooled Held | | | +0.12 [-0.79, +1.04] | inconclusive | inconclusive | +0.23 [-0.73, +1.24] | inconclusive | inconclusive |
| pooled Unseen | | | +0.23 [-0.46, +0.91] | inconclusive | equivalent | +0.12 [-0.48, +0.76] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +2.40 [+1.24, +3.64], L -0.19 [-1.01, +0.65], S -0.17 [-1.26, +0.97], W -1.00 [-2.55, +0.63]

### pub_Brandimarte

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.36 [-4.19, +3.32] | inconclusive | -0.51 [-4.77, +3.41] | inconclusive | inconclusive | +0.56 [-2.63, +4.43] | inconclusive | inconclusive |
| LW | +0.18 [-3.33, +3.79] | inconclusive | +1.00 [-2.49, +4.37] | inconclusive | inconclusive | -0.70 [-3.71, +2.32] | inconclusive | inconclusive |
| SW | +1.81 [-1.56, +5.03] | inconclusive | +1.17 [-2.21, +4.54] | inconclusive | inconclusive | +0.83 [-1.98, +4.00] | inconclusive | inconclusive |
| LSW | -1.19 [-4.40, +1.83] | inconclusive | -0.50 [-4.52, +4.04] | inconclusive | inconclusive | -0.28 [-3.57, +2.68] | inconclusive | inconclusive |
| pooled Pair | | | +0.25 [-3.10, +3.21] | inconclusive | inconclusive | -0.07 [-2.34, +2.68] | inconclusive | inconclusive |
| pooled Held | | | +0.33 [-2.51, +3.01] | inconclusive | inconclusive | +0.28 [-1.93, +2.62] | inconclusive | inconclusive |
| pooled Unseen | | | +0.29 [-2.13, +2.38] | inconclusive | inconclusive | +0.10 [-1.62, +1.85] | inconclusive | inconclusive |

Control on its own combinations (control vs specialist): N -0.21 [-1.61, +1.08], L -2.84 [-9.25, +1.41], S -1.76 [-6.14, +2.81], W +0.10 [-1.54, +1.70]

### pub_Hurink_edata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.76 [-1.69, +0.15] | non-inferior only | -0.56 [-1.50, +0.41] | non-inferior only | non-inferior only | -0.11 [-0.78, +0.57] | inconclusive | equivalent |
| LW | -0.53 [-1.52, +0.49] | non-inferior only | -0.24 [-1.49, +0.81] | inconclusive | non-inferior only | -0.19 [-1.30, +0.90] | inconclusive | non-inferior only |
| SW | -0.08 [-1.03, +0.96] | non-inferior only | +0.04 [-0.87, +1.04] | inconclusive | inconclusive | -0.03 [-0.86, +0.81] | inconclusive | equivalent |
| LSW | +0.78 [-0.27, +1.95] | inconclusive | +1.11 [-0.12, +2.55] | inconclusive | inconclusive | -0.20 [-1.15, +0.77] | inconclusive | non-inferior only |
| pooled Pair | | | -0.40 [-1.27, +0.43] | non-inferior only | non-inferior only | -0.15 [-0.86, +0.53] | inconclusive | equivalent |
| pooled Held | | | +0.57 [-0.23, +1.56] | inconclusive | inconclusive | -0.11 [-0.67, +0.44] | non-inferior only | equivalent |
| pooled Unseen | | | +0.09 [-0.61, +0.82] | inconclusive | equivalent | -0.13 [-0.57, +0.30] | non-inferior only | equivalent |

Control on its own combinations (control vs specialist): N -0.37 [-1.19, +0.49], L -0.78 [-1.79, +0.19], S -0.34 [-1.43, +0.75], W -0.77 [-2.27, +0.68]

### pub_Hurink_rdata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | +0.21 [-1.10, +1.56] | inconclusive | +0.05 [-1.36, +1.42] | inconclusive | inconclusive | +0.40 [-1.27, +2.15] | inconclusive | inconclusive |
| LW | -0.52 [-1.71, +0.62] | non-inferior only | -0.11 [-1.29, +1.17] | inconclusive | inconclusive | -0.27 [-1.24, +0.58] | inconclusive | non-inferior only |
| SW | -0.46 [-1.91, +0.97] | non-inferior only | -0.07 [-1.17, +1.10] | inconclusive | inconclusive | -0.28 [-1.32, +0.76] | inconclusive | non-inferior only |
| LSW | +0.23 [-1.25, +1.67] | inconclusive | +0.37 [-1.20, +1.92] | inconclusive | inconclusive | +0.12 [-1.04, +1.33] | inconclusive | inconclusive |
| pooled Pair | | | -0.03 [-0.97, +0.89] | inconclusive | equivalent | +0.07 [-0.88, +1.00] | inconclusive | inconclusive |
| pooled Held | | | +0.15 [-0.93, +1.25] | inconclusive | inconclusive | -0.08 [-0.87, +0.77] | inconclusive | equivalent |
| pooled Unseen | | | +0.06 [-0.72, +0.79] | inconclusive | equivalent | -0.01 [-0.59, +0.63] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +0.25 [-0.64, +1.21], L -1.22 [-2.53, -0.09], S +0.20 [-1.07, +1.43], W +0.24 [-0.85, +1.41]

### pub_Hurink_vdata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.02 [-1.53, +1.66] | inconclusive | +0.17 [-0.99, +1.42] | inconclusive | inconclusive | +0.03 [-1.25, +1.26] | inconclusive | inconclusive |
| LW | +0.09 [-0.87, +1.02] | inconclusive | +0.01 [-1.15, +1.15] | inconclusive | inconclusive | +0.17 [-0.85, +1.15] | inconclusive | inconclusive |
| SW | -1.24 [-2.86, +0.41] | non-inferior only | +0.07 [-1.90, +1.86] | inconclusive | inconclusive | -0.91 [-2.75, +0.85] | inconclusive | non-inferior only |
| LSW | +0.40 [-0.86, +1.68] | inconclusive | +0.52 [-0.80, +1.87] | inconclusive | inconclusive | +0.20 [-0.83, +1.27] | inconclusive | inconclusive |
| pooled Pair | | | +0.09 [-0.79, +0.90] | inconclusive | equivalent | +0.10 [-0.89, +1.06] | inconclusive | inconclusive |
| pooled Held | | | +0.30 [-0.87, +1.44] | inconclusive | inconclusive | -0.35 [-1.38, +0.62] | inconclusive | non-inferior only |
| pooled Unseen | | | +0.19 [-0.61, +1.00] | inconclusive | equivalent | -0.13 [-0.99, +0.61] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +0.55 [-0.36, +1.53], L -0.62 [-1.86, +0.48], S +0.20 [-1.71, +1.90], W +0.59 [-0.74, +1.81]

## 3. Policy vs dispatching rules

Best rule = per-instance minimum of FIFO, MOR, SPT and MWKR under the same mask. Shares are percent of base instances, averaged over the five seeds (seed mean) or for the selected seed (s303).

### pub_SD2_test_10x5+mix

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -14.18 | [-15.76, -12.54] | 93 / 0 / 7 | 1.6e-05 | 0.0004 | -15.17 | [-16.80, -13.45] | 95 / 0 / 5 | FIFO 11, MOR 13, SPT 56, MWKR 23 |
| L | -0.33 | [-1.32, +0.65] | 45 / 9 / 46 | 0.23 | 1 | +0.12 | [-1.23, +1.58] | 42 / 12 / 46 | FIFO 20, MOR 22, SPT 49, MWKR 25 |
| S | +5.56 | [+3.91, +7.27] | 21 / 10 / 69 | 1 | 1 | +5.20 | [+3.35, +7.09] | 23 / 10 / 67 | FIFO 22, MOR 29, SPT 32, MWKR 32 |
| W | -9.49 | [-11.29, -7.74] | 80 / 8 / 13 | 6.6e-07 | 0.0004 | -9.36 | [-11.33, -7.37] | 78 / 8 / 14 | FIFO 23, MOR 16, SPT 44, MWKR 41 |
| LS | +4.04 | [+2.77, +5.43] | 19 / 16 / 64 | 1 | 1 | +4.32 | [+2.90, +5.87] | 18 / 17 / 65 | FIFO 32, MOR 43, SPT 32, MWKR 33 |
| LW | -1.56 | [-2.79, -0.32] | 52 / 15 / 34 | 0.00024 | 0.048 | -1.59 | [-2.97, -0.17] | 50 / 17 / 33 | FIFO 28, MOR 34, SPT 40, MWKR 33 |
| SW | +4.15 | [+2.74, +5.62] | 22 / 10 / 68 | 1 | 1 | +3.78 | [+2.12, +5.53] | 24 / 12 / 64 | FIFO 26, MOR 24, SPT 35, MWKR 34 |
| LSW | +4.00 | [+2.29, +5.79] | 24 / 13 / 63 | 1 | 1 | +2.61 | [+0.91, +4.30] | 28 / 12 / 60 | FIFO 24, MOR 32, SPT 38, MWKR 38 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -23.80 (100) | -23.71 (99) | -18.32 (95) | -22.21 (99) |
| L | -7.03 (84) | -6.85 (79) | -3.59 (63) | -7.32 (79) |
| S | -1.69 (56) | -0.73 (49) | -1.60 (52) | -1.04 (49) |
| W | -17.84 (93) | -18.84 (93) | -16.82 (92) | -14.70 (86) |
| LS | -1.07 (49) | -0.35 (45) | -2.50 (56) | -1.92 (57) |
| LW | -8.03 (76) | -7.18 (74) | -6.76 (72) | -7.65 (76) |
| SW | -2.60 (57) | -2.21 (56) | -2.24 (54) | -1.78 (54) |
| LSW | -2.21 (58) | -1.45 (51) | -2.17 (50) | -1.40 (50) |

### pub_SD2_test_20x10+mix

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -23.41 | [-24.86, -21.91] | 100 / 0 / 0 | 2.7e-06 | 0.0004 | -22.99 | [-24.30, -21.64] | 100 / 0 / 0 | FIFO 1, MOR 1, SPT 95, MWKR 3 |
| L | +0.23 | [-0.70, +1.13] | 50 / 1 / 49 | 1 | 1 | +0.40 | [-0.55, +1.39] | 48 / 2 / 50 | FIFO 5, MOR 11, SPT 65, MWKR 19 |
| S | +3.11 | [+2.13, +4.14] | 32 / 0 / 68 | 1 | 1 | +3.12 | [+1.74, +4.56] | 39 / 0 / 61 | FIFO 12, MOR 25, SPT 41, MWKR 22 |
| W | -22.17 | [-23.57, -20.72] | 100 / 0 / 0 | 5e-07 | 0.0004 | -22.02 | [-23.50, -20.54] | 100 / 0 / 0 | FIFO 5, MOR 9, SPT 76, MWKR 12 |
| LS | +5.25 | [+3.93, +6.69] | 24 / 0 / 75 | 1 | 1 | +6.32 | [+4.71, +8.12] | 21 / 0 / 79 | FIFO 16, MOR 23, SPT 35, MWKR 26 |
| LW | -0.51 | [-1.43, +0.43] | 53 / 3 / 44 | 0.064 | 0.85 | -0.12 | [-1.05, +0.84] | 53 / 2 / 45 | FIFO 12, MOR 18, SPT 52, MWKR 25 |
| SW | +2.85 | [+1.75, +4.00] | 35 / 1 / 63 | 1 | 1 | +2.53 | [+1.17, +3.93] | 40 / 0 / 60 | FIFO 24, MOR 17, SPT 36, MWKR 24 |
| LSW | +6.10 | [+4.70, +7.49] | 26 / 0 / 74 | 1 | 1 | +5.23 | [+3.56, +6.94] | 25 / 0 / 75 | FIFO 22, MOR 22, SPT 31, MWKR 25 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -38.33 (100) | -38.47 (100) | -23.56 (100) | -36.97 (100) |
| L | -6.34 (91) | -5.24 (86) | -1.03 (60) | -4.61 (82) |
| S | -3.67 (70) | -2.74 (63) | -1.54 (59) | -3.60 (70) |
| W | -31.79 (100) | -31.73 (100) | -23.27 (100) | -29.23 (100) |
| LS | -1.84 (59) | -0.91 (56) | -0.80 (55) | -0.87 (55) |
| LW | -6.93 (87) | -5.87 (85) | -2.78 (65) | -5.09 (80) |
| SW | -2.38 (63) | -2.56 (64) | -0.85 (56) | -1.92 (63) |
| LSW | -0.61 (57) | -1.09 (57) | -0.70 (55) | +0.48 (50) |

### pub_Brandimarte

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -5.64 | [-11.24, -0.73] | 56 / 26 / 18 | 9.1e-05 | 0.074 | -4.77 | [-10.96, +0.51] | 40 / 30 / 30 | FIFO 20, MOR 30, SPT 0, MWKR 50 |
| L | -0.13 | [-4.56, +4.08] | 46 / 6 / 48 | 1 | 1 | -0.59 | [-5.89, +4.72] | 60 / 0 / 40 | FIFO 10, MOR 40, SPT 0, MWKR 50 |
| S | +5.79 | [+1.90, +10.10] | 22 / 2 / 76 | 1 | 1 | +7.25 | [+0.29, +14.29] | 20 / 10 / 70 | FIFO 20, MOR 50, SPT 10, MWKR 20 |
| W | -4.50 | [-8.45, -1.01] | 56 / 18 / 26 | 3.1e-05 | 0.039 | -4.47 | [-8.38, -1.01] | 50 / 30 / 20 | FIFO 40, MOR 50, SPT 0, MWKR 50 |
| LS | +4.02 | [+0.88, +7.38] | 32 / 2 / 66 | 1 | 1 | +5.26 | [+1.00, +9.85] | 30 / 0 / 70 | FIFO 20, MOR 10, SPT 50, MWKR 30 |
| LW | -1.70 | [-6.67, +3.13] | 60 / 4 / 36 | 0.25 | 1 | -3.29 | [-8.16, +2.00] | 70 / 0 / 30 | FIFO 20, MOR 40, SPT 10, MWKR 40 |
| SW | +5.10 | [+2.21, +8.56] | 22 / 0 / 78 | 1 | 1 | +6.01 | [+1.19, +12.72] | 20 / 0 / 80 | FIFO 20, MOR 30, SPT 10, MWKR 40 |
| LSW | +4.61 | [+1.45, +8.24] | 26 / 0 / 74 | 1 | 1 | +5.33 | [+1.83, +8.78] | 30 / 0 / 70 | FIFO 20, MOR 50, SPT 10, MWKR 30 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -8.72 (92) | -9.25 (88) | -21.84 (98) | -8.50 (70) |
| L | -8.36 (90) | -3.82 (70) | -11.28 (92) | -2.50 (58) |
| S | -0.73 (52) | +1.26 (42) | -9.01 (86) | -0.20 (58) |
| W | -7.04 (80) | -6.00 (72) | -20.75 (100) | -7.18 (66) |
| LS | +0.09 (60) | -3.17 (82) | -1.51 (60) | -1.08 (56) |
| LW | -8.59 (82) | -5.47 (80) | -13.62 (96) | -3.35 (68) |
| SW | +1.76 (40) | +1.07 (46) | -4.08 (60) | +3.46 (28) |
| LSW | -2.17 (70) | -1.86 (62) | -3.94 (84) | -0.83 (58) |

### pub_Hurink_edata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | +1.28 | [-0.04, +2.62] | 41 / 2 / 56 | 1 | 1 | +0.52 | [-0.91, +1.96] | 46 / 0 / 54 | FIFO 12, MOR 25, SPT 28, MWKR 37 |
| L | -0.31 | [-1.55, +0.94] | 48 / 8 / 44 | 0.19 | 1 | -0.44 | [-1.71, +0.82] | 48 / 9 / 43 | FIFO 20, MOR 26, SPT 37, MWKR 31 |
| S | +3.76 | [+2.89, +4.62] | 16 / 6 / 78 | 1 | 1 | +3.36 | [+2.34, +4.42] | 18 / 9 / 72 | FIFO 23, MOR 29, SPT 26, MWKR 34 |
| W | +1.74 | [+0.46, +3.04] | 35 / 10 / 55 | 1 | 1 | +1.50 | [+0.11, +2.95] | 38 / 8 / 54 | FIFO 12, MOR 43, SPT 12, MWKR 42 |
| LS | +2.47 | [+1.38, +3.58] | 30 / 7 / 63 | 1 | 1 | +2.21 | [+0.86, +3.56] | 31 / 8 / 62 | FIFO 28, MOR 23, SPT 32, MWKR 34 |
| LW | -0.32 | [-1.79, +1.16] | 41 / 24 / 36 | 1 | 1 | -0.67 | [-2.04, +0.70] | 40 / 20 / 40 | FIFO 23, MOR 40, SPT 32, MWKR 42 |
| SW | +2.81 | [+1.70, +3.98] | 22 / 11 / 67 | 1 | 1 | +2.55 | [+1.44, +3.69] | 22 / 11 / 68 | FIFO 20, MOR 34, SPT 23, MWKR 40 |
| LSW | +3.80 | [+2.51, +5.15] | 24 / 7 / 70 | 1 | 1 | +4.53 | [+2.85, +6.40] | 26 / 9 / 65 | FIFO 35, MOR 31, SPT 23, MWKR 31 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -6.06 (84) | -3.93 (74) | -6.12 (73) | -2.08 (66) |
| L | -5.41 (79) | -4.18 (78) | -6.12 (71) | -4.90 (70) |
| S | -1.23 (54) | -0.32 (47) | -0.42 (52) | -0.76 (48) |
| W | -7.18 (83) | -2.22 (59) | -9.75 (83) | -1.73 (62) |
| LS | -1.59 (57) | -1.81 (61) | -0.98 (53) | -1.13 (53) |
| LW | -7.15 (76) | -4.44 (65) | -6.75 (67) | -4.12 (62) |
| SW | -0.61 (50) | -0.65 (53) | -2.23 (61) | +0.01 (49) |
| LSW | +0.60 (46) | +0.66 (46) | -1.32 (53) | -0.27 (50) |

### pub_Hurink_rdata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.79 | [-0.18, +1.80] | 51 / 2 / 48 | 1 | 1 | +1.53 | [+0.44, +2.66] | 45 / 3 / 52 | FIFO 18, MOR 28, SPT 9, MWKR 46 |
| L | -0.20 | [-1.26, +0.81] | 45 / 5 / 50 | 1 | 1 | +0.27 | [-0.88, +1.42] | 42 / 5 / 54 | FIFO 22, MOR 32, SPT 34, MWKR 34 |
| S | +5.11 | [+3.83, +6.40] | 22 / 2 / 77 | 1 | 1 | +4.98 | [+3.12, +6.84] | 25 / 3 / 72 | FIFO 34, MOR 23, SPT 17, MWKR 32 |
| W | +2.08 | [+0.93, +3.28] | 30 / 8 / 62 | 1 | 1 | +3.07 | [+1.74, +4.51] | 20 / 5 / 75 | FIFO 23, MOR 35, SPT 8, MWKR 46 |
| LS | +4.50 | [+2.95, +6.16] | 27 / 3 / 70 | 1 | 1 | +4.43 | [+2.44, +6.52] | 31 / 6 / 63 | FIFO 23, MOR 29, SPT 29, MWKR 34 |
| LW | +0.04 | [-1.41, +1.44] | 40 / 18 / 42 | 1 | 1 | +0.11 | [-1.54, +1.75] | 45 / 17 / 38 | FIFO 42, MOR 37, SPT 20, MWKR 54 |
| SW | +4.12 | [+2.79, +5.57] | 26 / 2 / 73 | 1 | 1 | +3.26 | [+1.68, +5.04] | 34 / 2 / 65 | FIFO 17, MOR 20, SPT 25, MWKR 43 |
| LSW | +4.50 | [+2.86, +6.15] | 20 / 3 / 77 | 1 | 1 | +4.72 | [+3.18, +6.41] | 15 / 6 / 78 | FIFO 18, MOR 25, SPT 29, MWKR 31 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -4.14 (82) | -3.46 (77) | -10.15 (89) | -1.15 (62) |
| L | -4.10 (70) | -2.83 (68) | -5.13 (70) | -3.41 (67) |
| S | +0.07 (48) | -0.62 (57) | -2.04 (59) | -0.13 (47) |
| W | -2.33 (64) | -1.30 (54) | -11.05 (83) | -0.13 (45) |
| LS | -2.21 (60) | -0.56 (53) | -0.67 (54) | +0.39 (48) |
| LW | -4.12 (62) | -3.45 (63) | -7.53 (78) | -2.90 (59) |
| SW | -1.95 (58) | -1.01 (53) | -2.07 (62) | -0.20 (50) |
| LSW | -1.05 (54) | +0.03 (46) | -0.86 (56) | -1.23 (51) |

### pub_Hurink_vdata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | +1.93 | [+1.19, +2.84] | 24 / 2 / 74 | 1 | 1 | +3.08 | [+2.05, +4.36] | 12 / 0 / 88 | FIFO 17, MOR 20, SPT 8, MWKR 62 |
| L | -0.94 | [-1.91, -0.08] | 39 / 28 / 33 | 0.022 | 0.13 | -0.50 | [-1.28, +0.25] | 37 / 28 / 35 | FIFO 35, MOR 48, SPT 42, MWKR 42 |
| S | +5.08 | [+3.39, +6.99] | 26 / 0 / 74 | 1 | 1 | +5.55 | [+3.45, +7.89] | 26 / 0 / 74 | FIFO 18, MOR 31, SPT 14, MWKR 37 |
| W | +2.61 | [+1.31, +4.39] | 17 / 6 / 76 | 1 | 1 | +5.17 | [+3.53, +7.18] | 8 / 6 / 86 | FIFO 17, MOR 23, SPT 6, MWKR 74 |
| LS | +3.58 | [+2.02, +5.21] | 26 / 1 / 73 | 1 | 1 | +3.84 | [+2.20, +5.61] | 23 / 0 / 77 | FIFO 17, MOR 28, SPT 26, MWKR 40 |
| LW | -0.99 | [-2.11, +0.08] | 39 / 21 / 40 | 0.048 | 0.24 | -0.54 | [-1.73, +0.62] | 35 / 18 / 46 | FIFO 25, MOR 45, SPT 29, MWKR 40 |
| SW | +5.46 | [+3.66, +7.33] | 24 / 1 / 75 | 1 | 1 | +6.00 | [+4.14, +7.90] | 20 / 0 / 80 | FIFO 22, MOR 22, SPT 15, MWKR 45 |
| LSW | +3.88 | [+2.50, +5.37] | 29 / 1 / 70 | 1 | 1 | +4.60 | [+3.08, +6.25] | 26 / 2 / 72 | FIFO 26, MOR 28, SPT 28, MWKR 29 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -2.13 (72) | -1.18 (63) | -10.11 (83) | +0.82 (36) |
| L | -3.37 (66) | -2.56 (58) | -4.42 (63) | -4.50 (61) |
| S | -0.78 (54) | -0.87 (57) | -2.49 (61) | -0.13 (50) |
| W | -2.53 (65) | -0.62 (51) | -10.76 (90) | +1.43 (25) |
| LS | -1.30 (61) | -1.49 (61) | -0.76 (56) | -0.11 (49) |
| LW | -4.39 (75) | -3.22 (66) | -6.95 (73) | -4.55 (62) |
| SW | -0.54 (51) | +0.01 (53) | -1.90 (60) | +0.74 (44) |
| LSW | -0.57 (51) | +0.19 (53) | -1.63 (61) | -0.25 (58) |

## 4. Regime N: gap to the best-known value (%)

Best-known values: Behnke and Geiger (2012) for Brandimarte and Hurink (the smaller of the literature value and the table's CP Optimizer value); per-instance OR-Tools results with an 1800 s limit from the SD2 release for SD2. On SD2 20x10 only one reference is proved optimal, so the gap there is a gap to a strong solver.

| set | proved optimal | policy (seed mean, sd) | selected | control | N specialist | FIFO | MOR | SPT | MWKR | best rule | policy on proved | policy to lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix | 96 of 100 | 27.81 (1.43) | 26.23 | 26.37 | 25.60 (1.13) | 68.85 | 68.97 | 58.30 | 65.77 | 49.83 | 27.90 | - |
| pub_SD2_test_20x10+mix | 1 of 100 | 37.11 (1.79) | 37.87 | 35.35 | 32.45 (0.95) | 123.07 | 123.46 | 80.49 | 118.31 | 80.05 | 30.04 | - |
| pub_Brandimarte | 3 of 10 | 11.66 (0.80) | 12.86 | 11.06 | 11.38 (1.06) | 23.32 | 24.61 | 44.70 | 23.66 | 19.42 | 6.44 | 27.60 |
| pub_Hurink_edata | 26 of 65 | 15.78 (0.90) | 14.90 | 15.00 | 15.58 (0.50) | 23.79 | 20.92 | 24.06 | 18.38 | 14.43 | 13.95 | 32.53 |
| pub_Hurink_rdata | 17 of 65 | 13.81 (0.62) | 14.66 | 13.54 | 13.41 (0.49) | 18.88 | 18.10 | 27.56 | 15.33 | 12.94 | 9.67 | 18.32 |
| pub_Hurink_vdata | 45 of 65 | 5.48 (0.69) | 6.67 | 5.71 | 5.19 (0.43) | 7.95 | 6.87 | 18.39 | 4.69 | 3.50 | 5.23 | 5.60 |

## 5. Selected seed

Rule: best validation score of the kept checkpoint (arm_registry.select_seed), validation on the first 20 records of pub_SD2_vali_10x5+mix. Selected: s303. Scores: s301 0.86621 (update index 30), s302 0.86125 (update index 77), s303 0.86067 (update index 68), s304 0.87057 (update index 78), s305 0.86747 (update index 60).

## 6. Per-seed gap to the per-instance best rule (%)

### pub_SD2_test_10x5+mix

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -14.47 | -14.55 | -15.17 | -14.02 | -12.68 | -14.18 | 0.93 |
| N | control | -15.29 | -15.07 | -15.46 | -13.67 | -15.90 | -15.08 | 0.84 |
| N | specialist | -15.10 | -16.17 | -14.79 | -15.59 | -16.55 | -15.64 | 0.73 |
| L | policy | -0.18 | -0.81 | +0.12 | -0.34 | -0.47 | -0.33 | 0.34 |
| L | control | -0.03 | -0.31 | +0.26 | +0.34 | -0.16 | +0.02 | 0.28 |
| L | specialist | -0.17 | +0.07 | -0.18 | -0.66 | -0.65 | -0.32 | 0.32 |
| S | policy | +5.39 | +6.98 | +5.20 | +5.12 | +5.09 | +5.56 | 0.80 |
| S | control | +5.09 | +4.95 | +4.20 | +5.99 | +5.10 | +5.07 | 0.64 |
| S | specialist | +6.50 | +5.75 | +5.17 | +6.27 | +5.63 | +5.86 | 0.53 |
| W | policy | -9.19 | -9.80 | -9.36 | -9.76 | -9.34 | -9.49 | 0.27 |
| W | control | -9.87 | -8.57 | -10.55 | -9.98 | -10.63 | -9.92 | 0.83 |
| W | specialist | -10.58 | -9.13 | -9.58 | -10.02 | -9.71 | -9.80 | 0.54 |
| LS | policy | +3.21 | +4.64 | +4.32 | +4.13 | +3.93 | +4.04 | 0.53 |
| LS | control | +4.38 | +4.36 | +5.17 | +5.60 | +4.14 | +4.73 | 0.62 |
| LS | specialist | +4.56 | +4.55 | +4.65 | +4.81 | +5.91 | +4.90 | 0.58 |
| LW | policy | -1.66 | -1.82 | -1.59 | -1.26 | -1.46 | -1.56 | 0.21 |
| LW | control | -1.10 | -0.93 | -1.16 | -1.14 | -1.60 | -1.19 | 0.25 |
| LW | specialist | -1.42 | -1.37 | -1.12 | -1.23 | -0.91 | -1.21 | 0.20 |
| SW | policy | +5.09 | +3.71 | +3.78 | +4.57 | +3.60 | +4.15 | 0.65 |
| SW | control | +4.12 | +4.52 | +4.44 | +4.59 | +4.16 | +4.37 | 0.21 |
| SW | specialist | +4.36 | +4.15 | +4.22 | +4.09 | +5.55 | +4.47 | 0.61 |
| LSW | policy | +3.71 | +5.12 | +2.61 | +3.98 | +4.60 | +4.00 | 0.95 |
| LSW | control | +3.53 | +3.98 | +3.94 | +4.67 | +3.91 | +4.01 | 0.41 |
| LSW | specialist | +3.99 | +3.29 | +5.33 | +4.01 | +3.77 | +4.08 | 0.76 |

### pub_SD2_test_20x10+mix

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -24.27 | -24.27 | -22.99 | -23.60 | -21.90 | -23.41 | 1.00 |
| N | control | -24.05 | -23.42 | -23.98 | -24.73 | -25.72 | -24.38 | 0.88 |
| N | specialist | -25.72 | -26.31 | -25.84 | -25.38 | -26.76 | -26.00 | 0.54 |
| L | policy | +0.73 | -0.77 | +0.40 | +0.24 | +0.57 | +0.23 | 0.59 |
| L | control | -0.47 | +0.85 | +0.24 | +0.46 | -0.29 | +0.16 | 0.54 |
| L | specialist | +0.47 | -0.04 | +1.12 | +0.40 | +0.41 | +0.47 | 0.42 |
| S | policy | +3.61 | +3.14 | +3.12 | +2.89 | +2.77 | +3.11 | 0.33 |
| S | control | +4.01 | +2.61 | +3.75 | +3.00 | +2.41 | +3.16 | 0.70 |
| S | specialist | +4.47 | +3.93 | +3.67 | +2.43 | +3.71 | +3.64 | 0.75 |
| W | policy | -22.20 | -22.79 | -22.02 | -22.57 | -21.25 | -22.17 | 0.59 |
| W | control | -22.94 | -21.29 | -22.37 | -23.62 | -23.49 | -22.74 | 0.95 |
| W | specialist | -22.10 | -18.52 | -23.06 | -22.37 | -22.35 | -21.68 | 1.80 |
| LS | policy | +4.99 | +4.88 | +6.32 | +4.94 | +5.11 | +5.25 | 0.60 |
| LS | control | +5.75 | +5.16 | +5.45 | +6.84 | +5.45 | +5.73 | 0.65 |
| LS | specialist | +6.50 | +4.97 | +6.06 | +5.70 | +6.29 | +5.90 | 0.60 |
| LW | policy | -0.56 | -0.77 | -0.12 | -0.83 | -0.27 | -0.51 | 0.31 |
| LW | control | -1.22 | -0.33 | +1.40 | -0.55 | -1.44 | -0.43 | 1.12 |
| LW | specialist | -0.99 | -0.93 | -0.84 | -0.59 | +0.10 | -0.65 | 0.45 |
| SW | policy | +3.29 | +1.83 | +2.53 | +3.38 | +3.23 | +2.85 | 0.66 |
| SW | control | +1.70 | +2.91 | +2.45 | +4.04 | +3.10 | +2.84 | 0.86 |
| SW | specialist | +3.66 | +2.32 | +3.49 | +4.27 | +2.10 | +3.17 | 0.93 |
| LSW | policy | +6.18 | +6.88 | +5.23 | +6.88 | +5.35 | +6.10 | 0.79 |
| LSW | control | +5.55 | +6.38 | +7.68 | +5.97 | +6.58 | +6.43 | 0.80 |
| LSW | specialist | +6.07 | +6.59 | +6.74 | +6.81 | +7.47 | +6.74 | 0.50 |

### pub_Brandimarte

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -5.76 | -6.34 | -4.77 | -5.86 | -5.48 | -5.64 | 0.58 |
| N | control | -5.46 | -5.62 | -6.53 | -6.69 | -6.37 | -6.13 | 0.56 |
| N | specialist | -6.09 | -4.34 | -6.26 | -6.56 | -5.96 | -5.84 | 0.87 |
| L | policy | +2.35 | -1.42 | -0.59 | -1.49 | +0.51 | -0.13 | 1.60 |
| L | control | -0.65 | +1.74 | +1.30 | -0.42 | +1.03 | +0.60 | 1.07 |
| L | specialist | +8.08 | +0.93 | +13.98 | +1.99 | -0.67 | +4.86 | 6.08 |
| S | policy | +7.33 | +8.62 | +7.25 | +2.90 | +2.83 | +5.79 | 2.72 |
| S | control | +2.01 | +4.90 | +6.01 | +4.25 | +5.05 | +4.44 | 1.50 |
| S | specialist | +3.67 | +9.82 | +2.69 | +9.48 | +9.02 | +6.94 | 3.46 |
| W | policy | -4.07 | -4.35 | -4.47 | -4.97 | -4.66 | -4.50 | 0.34 |
| W | control | -2.24 | -2.66 | -3.72 | -2.56 | -4.07 | -3.05 | 0.80 |
| W | specialist | -2.10 | -2.07 | -2.69 | -4.38 | -4.31 | -3.11 | 1.15 |
| LS | policy | +5.61 | +2.43 | +5.26 | +4.05 | +2.74 | +4.02 | 1.43 |
| LS | control | +1.01 | +3.29 | +6.23 | +3.25 | +4.69 | +3.69 | 1.93 |
| LS | specialist | +5.32 | +6.32 | +5.72 | +3.70 | +3.61 | +4.93 | 1.22 |
| LW | policy | +0.51 | -2.56 | -3.29 | -2.81 | -0.37 | -1.70 | 1.67 |
| LW | control | -2.99 | -0.56 | +1.66 | -1.27 | -0.93 | -0.82 | 1.67 |
| LW | specialist | -1.53 | -4.54 | -0.86 | -0.92 | +0.65 | -1.44 | 1.91 |
| SW | policy | +4.56 | +5.57 | +6.01 | +6.02 | +3.36 | +5.10 | 1.14 |
| SW | control | +3.01 | +5.88 | +7.12 | +3.20 | +3.28 | +4.50 | 1.88 |
| SW | specialist | +3.11 | +4.31 | +4.04 | +3.02 | +3.55 | +3.61 | 0.56 |
| LSW | policy | +4.55 | +4.30 | +5.33 | +4.69 | +4.20 | +4.61 | 0.44 |
| LSW | control | +6.91 | +2.86 | +9.84 | +1.65 | +5.39 | +5.33 | 3.26 |
| LSW | specialist | +9.01 | +6.27 | +4.20 | +4.68 | +7.08 | +6.25 | 1.93 |

### pub_Hurink_edata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +2.22 | +0.49 | +0.52 | +1.34 | +1.81 | +1.28 | 0.77 |
| N | control | +0.24 | +0.90 | +0.69 | +0.94 | +0.24 | +0.60 | 0.34 |
| N | specialist | +1.12 | +1.51 | +0.61 | +0.71 | +1.55 | +1.10 | 0.44 |
| L | policy | -0.05 | -0.21 | -0.44 | -0.66 | -0.17 | -0.31 | 0.24 |
| L | control | -1.11 | +0.36 | -0.30 | -0.75 | -1.00 | -0.56 | 0.60 |
| L | specialist | -0.11 | +0.18 | +1.59 | +0.83 | -0.48 | +0.40 | 0.82 |
| S | policy | +3.85 | +4.23 | +3.36 | +3.91 | +3.44 | +3.76 | 0.36 |
| S | control | +3.96 | +3.43 | +4.53 | +3.88 | +4.69 | +4.10 | 0.51 |
| S | specialist | +4.06 | +5.19 | +4.19 | +5.37 | +4.37 | +4.64 | 0.60 |
| W | policy | +2.44 | +0.93 | +1.50 | +1.72 | +2.13 | +1.74 | 0.58 |
| W | control | +1.36 | +2.70 | +3.04 | +1.75 | +2.42 | +2.25 | 0.69 |
| W | specialist | +4.85 | +3.32 | +2.37 | +3.02 | +3.02 | +3.32 | 0.93 |
| LS | policy | +2.28 | +2.53 | +2.21 | +2.82 | +2.53 | +2.47 | 0.24 |
| LS | control | +2.52 | +3.32 | +2.16 | +3.24 | +2.25 | +2.70 | 0.55 |
| LS | specialist | +3.84 | +2.96 | +2.98 | +4.02 | +3.36 | +3.43 | 0.48 |
| LW | policy | +0.72 | -0.87 | -0.67 | -1.03 | +0.22 | -0.32 | 0.76 |
| LW | control | -0.18 | +0.85 | -0.09 | -0.23 | -0.57 | -0.04 | 0.53 |
| LW | specialist | -0.09 | -0.06 | +0.04 | +0.37 | +1.41 | +0.34 | 0.63 |
| SW | policy | +2.47 | +4.00 | +2.55 | +2.60 | +2.42 | +2.81 | 0.67 |
| SW | control | +2.57 | +3.02 | +3.41 | +2.58 | +3.04 | +2.92 | 0.36 |
| SW | specialist | +3.07 | +2.72 | +2.53 | +3.84 | +2.97 | +3.03 | 0.50 |
| LSW | policy | +3.99 | +3.13 | +4.53 | +3.71 | +3.63 | +3.80 | 0.51 |
| LSW | control | +3.97 | +4.76 | +5.08 | +3.41 | +3.61 | +4.17 | 0.73 |
| LSW | specialist | +4.20 | +2.74 | +2.72 | +2.53 | +3.84 | +3.21 | 0.75 |

### pub_Hurink_rdata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +0.60 | +0.95 | +1.53 | +0.82 | +0.05 | +0.79 | 0.54 |
| N | control | +0.18 | +1.48 | +1.13 | +0.68 | -0.64 | +0.57 | 0.83 |
| N | specialist | +0.13 | +1.14 | +0.30 | +0.56 | +0.08 | +0.44 | 0.43 |
| L | policy | -0.10 | +0.13 | +0.27 | -1.08 | -0.20 | -0.20 | 0.53 |
| L | control | -0.65 | -0.26 | -0.34 | +0.12 | -0.56 | -0.34 | 0.30 |
| L | specialist | +1.73 | +0.37 | +2.63 | +1.28 | -0.32 | +1.14 | 1.15 |
| S | policy | +5.69 | +4.99 | +4.98 | +5.51 | +4.36 | +5.11 | 0.52 |
| S | control | +5.83 | +5.16 | +5.45 | +5.39 | +4.44 | +5.25 | 0.51 |
| S | specialist | +5.36 | +5.54 | +6.34 | +5.48 | +4.26 | +5.40 | 0.74 |
| W | policy | +2.07 | +1.61 | +3.07 | +1.76 | +1.89 | +2.08 | 0.58 |
| W | control | +2.21 | +2.01 | +2.87 | +2.53 | +1.65 | +2.25 | 0.47 |
| W | specialist | +1.48 | +3.36 | +1.90 | +2.34 | +1.89 | +2.19 | 0.72 |
| LS | policy | +4.07 | +5.06 | +4.43 | +3.36 | +5.58 | +4.50 | 0.86 |
| LS | control | +5.86 | +4.19 | +3.75 | +5.01 | +3.03 | +4.37 | 1.10 |
| LS | specialist | +4.78 | +4.88 | +4.02 | +4.27 | +5.25 | +4.64 | 0.49 |
| LW | policy | +0.02 | +0.46 | +0.11 | -0.39 | +0.01 | +0.04 | 0.30 |
| LW | control | -0.49 | +0.13 | +1.16 | +0.66 | +0.67 | +0.43 | 0.63 |
| LW | specialist | +1.08 | -0.33 | +0.59 | +0.97 | +1.53 | +0.77 | 0.70 |
| SW | policy | +5.35 | +4.15 | +3.26 | +4.06 | +3.78 | +4.12 | 0.77 |
| SW | control | +5.29 | +4.99 | +4.74 | +4.19 | +3.86 | +4.62 | 0.58 |
| SW | specialist | +4.42 | +4.93 | +5.71 | +4.92 | +4.88 | +4.97 | 0.46 |
| LSW | policy | +5.07 | +3.57 | +4.72 | +4.69 | +4.47 | +4.50 | 0.57 |
| LSW | control | +5.51 | +4.13 | +5.03 | +3.89 | +4.83 | +4.68 | 0.66 |
| LSW | specialist | +4.51 | +3.35 | +4.58 | +5.54 | +5.63 | +4.72 | 0.93 |

### pub_Hurink_vdata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +1.44 | +1.66 | +3.08 | +1.66 | +1.84 | +1.93 | 0.66 |
| N | control | +1.55 | +1.88 | +2.87 | +2.83 | +1.70 | +2.17 | 0.64 |
| N | specialist | +1.70 | +2.18 | +1.19 | +1.29 | +1.96 | +1.66 | 0.43 |
| L | policy | -0.60 | -1.31 | -0.50 | -1.33 | -0.99 | -0.94 | 0.39 |
| L | control | -1.29 | -0.40 | -0.80 | -1.15 | -1.29 | -0.99 | 0.39 |
| L | specialist | +1.03 | -0.70 | +0.18 | -0.57 | -0.92 | -0.20 | 0.80 |
| S | policy | +4.91 | +6.65 | +5.55 | +4.52 | +3.78 | +5.08 | 1.08 |
| S | control | +5.66 | +4.90 | +6.74 | +6.42 | +5.43 | +5.83 | 0.75 |
| S | specialist | +5.11 | +8.29 | +6.14 | +6.13 | +4.92 | +6.12 | 1.34 |
| W | policy | +1.47 | +2.25 | +5.17 | +2.19 | +1.98 | +2.61 | 1.46 |
| W | control | +1.77 | +2.92 | +3.09 | +3.59 | +1.30 | +2.53 | 0.96 |
| W | specialist | +0.72 | +4.46 | +2.43 | +2.31 | +1.15 | +2.21 | 1.45 |
| LS | policy | +5.12 | +2.83 | +3.84 | +2.46 | +3.65 | +3.58 | 1.03 |
| LS | control | +3.84 | +4.37 | +3.59 | +3.36 | +3.73 | +3.78 | 0.38 |
| LS | specialist | +2.75 | +4.30 | +4.20 | +3.95 | +4.41 | +3.92 | 0.68 |
| LW | policy | -0.42 | -1.72 | -0.54 | -1.07 | -1.21 | -0.99 | 0.53 |
| LW | control | -1.68 | -0.27 | -1.04 | -1.33 | -1.10 | -1.08 | 0.52 |
| LW | specialist | -0.92 | -1.40 | -1.60 | +0.15 | -1.04 | -0.96 | 0.68 |
| SW | policy | +5.73 | +3.30 | +6.00 | +6.71 | +5.57 | +5.46 | 1.28 |
| SW | control | +7.69 | +7.82 | +6.52 | +6.93 | +5.05 | +6.80 | 1.12 |
| SW | specialist | +7.55 | +6.28 | +6.57 | +6.40 | +8.78 | +7.12 | 1.06 |
| LSW | policy | +3.67 | +3.88 | +4.60 | +3.05 | +4.19 | +3.88 | 0.58 |
| LSW | control | +3.43 | +3.89 | +4.25 | +3.25 | +4.78 | +3.92 | 0.62 |
| LSW | specialist | +3.41 | +3.07 | +3.78 | +4.03 | +4.52 | +3.76 | 0.56 |

## 7. Sensitivity: pub_SD2_test_10x5+mix_lag050 ({'lag_scale': 0.5}) beside pub_SD2_test_10x5+mix

Records: 800 compared with pub_SD2_test_10x5+mix; fields changed: L|horizon (100), L|lag (100), LS|horizon (100), LS|lag (100), LS|setup_convention (100), LSW|horizon (100), LSW|lag (100), LSW|setup_convention (100), LSW|windows (100), LW|horizon (100), LW|lag (100), LW|windows (100), N|horizon (100), S|horizon (100), S|setup_convention (100), SW|horizon (100), SW|setup_convention (100), SW|windows (100), W|horizon (100), W|windows (100). Free check: 38 N and S arrays (3800 makespans) of every run and rule equal the scale-1.0 arrays bit for bit. Combinations with W also move their outage calendars (H shrinks with the lags); only the combinations with L are compared.

| comb. | best-rule R/N (1.0 / 0.5) | policy R/N (1.0 / 0.5) |
|---|---|---|
| L | 1.81 / 1.23 | 2.11 / 1.41 |
| LS | 2.40 / 1.98 | 2.91 / 2.45 |
| LW | 1.89 / 1.37 | 2.18 / 1.52 |
| LSW | 2.65 / 2.17 | 3.20 / 2.67 |

Policy vs specialist:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | +0.10 [-0.63, +0.82] | inconclusive | equivalent | -0.99 [-2.52, +0.36] | [-2.59, +0.62] | non-inferior only | non-inferior only | -1.16, -0.46, -3.15, +0.07, -0.24 | -1.09 [-2.79, +0.40] |
| LS | -0.50 [-1.63, +0.77] | inconclusive | non-inferior only | -0.96 [-2.55, +0.64] | [-2.40, +0.49] | inconclusive | non-inferior only | -0.71, -0.45, -0.28, -0.33, -3.02 | -0.45 [-2.03, +1.15] |
| LW | -0.18 [-0.86, +0.49] | non-inferior only | equivalent | -1.59 [-3.68, +0.56] | [-4.61, +1.42] | inconclusive | non-inferior only | -3.24, +0.95, +1.15, -3.01, -3.81 | -1.41 [-3.66, +0.92] |
| LSW | +0.41 [-1.39, +2.21] | inconclusive | inconclusive | -1.13 [-2.31, +0.10] | [-1.87, -0.39] | non-inferior only | non-inferior only | -1.84, -1.01, -1.43, -0.23, -1.13 | -1.54 [-3.53, +0.48] |
| pooled | -0.04 [-0.61, +0.52] | inconclusive | equivalent | -1.17 [-1.95, -0.37] | [-2.07, -0.27] | non-inferior only | non-inferior only | | |

Policy vs control:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | -0.24 [-0.85, +0.34] | non-inferior only | equivalent | +0.11 [-1.05, +1.23] | [-0.98, +1.21] | inconclusive | inconclusive | +1.01, -0.64, -0.95, +0.84, +0.32 | +0.36 [-0.89, +1.56] |
| LS | -0.46 [-1.33, +0.43] | non-inferior only | non-inferior only | -0.00 [-0.98, +0.96] | [-0.45, +0.44] | inconclusive | equivalent | -0.59, -0.04, +0.30, +0.25, +0.06 | +0.45 [-0.78, +1.74] |
| LW | -0.28 [-0.90, +0.31] | non-inferior only | equivalent | -0.15 [-1.22, +0.94] | [-1.20, +0.91] | inconclusive | non-inferior only | -0.63, -0.51, +0.28, -0.99, +1.13 | +0.13 [-0.95, +1.23] |
| LSW | +0.32 [-0.88, +1.51] | inconclusive | inconclusive | +0.14 [-0.65, +0.96] | [-0.05, +0.32] | inconclusive | equivalent | +0.01, +0.13, +0.30, +0.28, -0.03 | -0.18 [-1.61, +1.24] |
| pooled | -0.17 [-0.66, +0.28] | non-inferior only | equivalent | +0.03 [-0.47, +0.52] | [-0.26, +0.31] | inconclusive | equivalent | | |

Control vs specialist:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | +0.48 [-0.31, +1.29] | inconclusive | inconclusive | -0.78 [-2.16, +0.55] | [-2.19, +0.63] | inconclusive | non-inferior only | -1.98, +0.59, -1.93, -0.41, -0.17 | - |
| LS | +0.14 [-1.13, +1.54] | inconclusive | inconclusive | -0.63 [-2.30, +1.03] | [-2.22, +0.95] | inconclusive | inconclusive | +0.28, -0.02, -0.26, -0.28, -2.88 | - |
| LW | +0.22 [-0.53, +0.96] | inconclusive | equivalent | -1.16 [-3.53, +1.13] | [-4.51, +2.20] | inconclusive | inconclusive | -2.29, +1.79, +1.29, -1.83, -4.74 | - |
| LSW | +0.35 [-1.04, +1.72] | inconclusive | inconclusive | -1.03 [-2.18, +0.23] | [-1.69, -0.36] | non-inferior only | non-inferior only | -1.65, -0.79, -1.45, -0.30, -0.94 | - |
| pooled | +0.30 [-0.26, +0.86] | inconclusive | equivalent | -0.90 [-1.85, +0.05] | [-2.09, +0.30] | non-inferior only | non-inferior only | | |

Policy vs best rule:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | -0.33 [-1.32, +0.65] | inconclusive | non-inferior only | -2.34 [-3.90, -0.77] | [-2.90, -1.79] | non-inferior only | non-inferior only | -1.79, -2.89, -2.48, -2.00, -2.57 | -2.01 [-3.67, -0.42] |
| LS | +4.04 [+2.77, +5.43] | inferior | inferior | +6.30 [+4.63, +8.06] | [+5.53, +7.06] | inferior | inferior | +5.55, +6.25, +7.05, +6.75, +5.86 | +2.25 [+0.47, +4.08] |
| LW | -1.56 [-2.79, -0.32] | non-inferior only | non-inferior only | -4.82 [-6.06, -3.54] | [-5.57, -4.06] | non-inferior only | non-inferior only | -5.17, -4.55, -3.99, -5.59, -4.78 | -3.26 [-4.94, -1.57] |
| LSW | +4.00 [+2.29, +5.79] | inferior | inferior | +5.65 [+4.13, +7.25] | [+5.02, +6.28] | inferior | inferior | +5.13, +5.13, +5.70, +6.24, +6.05 | +1.65 [-0.46, +3.86] |
| pooled | +1.54 [+0.79, +2.29] | inferior | inferior | +1.20 [+0.37, +2.06] | [+0.87, +1.53] | inferior | inferior | | |

Equivalent on all four combinations with L at lag scale 0.5 (0.5% / 1%): False / False (failing at 1%: L,LS,LW,LSW); scale 1.0: False / False.

Policy vs each rule, mean d (shorter %), scale 1.0 / 0.5:

| comb. | FIFO | MOR | SPT | MWKR |
|---|---|---|---|---|
| L | -7.03 (84) / -12.83 (91) | -6.85 (79) / -10.54 (86) | -3.59 (63) / -6.86 (70) | -7.32 (79) / -11.63 (86) |
| LS | -1.07 (49) / -0.01 (48) | -0.35 (45) / -2.19 (54) | -2.50 (56) / -2.04 (57) | -1.92 (57) / +0.21 (48) |
| LW | -8.03 (76) / -12.17 (90) | -7.18 (74) / -10.85 (87) | -6.76 (72) / -11.62 (84) | -7.65 (76) / -10.34 (87) |
| LSW | -2.21 (58) / -0.85 (52) | -1.45 (51) / -0.62 (52) | -2.17 (50) / -1.13 (45) | -1.40 (50) / -1.82 (55) |

## Diagnostics

### Validation at the kept checkpoints (20 SD2 10x5 validation records)

| comb. | specialist (mean of seeds) | policy minus specialist % per seed | control minus specialist % per seed | specialist kept update | specialist gain in the last quarter % |
|---|---|---|---|---|---|
| N | 393.0 | +1.90, +2.53, +1.84, +2.59, +4.74 | +3.37, +3.20, +2.68, +1.23, +0.05 | 900, 775, 700, 700, 675 | 0.68, 1.15, -0.11, -0.39, -2.04 |
| L | 890.1 | +0.41, +0.28, +0.82, -0.07, +0.66 | -0.04, +1.04, +0.71, -0.11, +0.47 | 825, 875, 175, 550, 950 | 0.02, 0.02, -0.30, -0.35, 0.03 |
| S | 885.4 | +0.32, +1.75, +2.21, +1.11, +1.18 | -0.04, +1.15, +0.19, +0.96, +1.50 | 875, 225, 925, 825, 225 | 0.22, -0.34, 0.73, 0.39, -1.41 |
| W | 467.2 | +4.44, -0.55, +0.65, +3.65, +1.41 | +2.44, +1.21, -0.35, +2.89, +0.11 | 950, 875, 475, 850, 700 | 0.63, 0.03, -3.16, 3.06, -0.25 |
| LS | 1215.0 | +0.07, +1.12, -0.48, +4.45, +0.96 | - | 475, 575, 900, 575, 75 | -0.15, -0.04, 0.41, -1.83, -0.82 |
| LW | 911.3 | +1.86, +0.90, +0.63, +1.33, +1.35 | - | 675, 650, 900, 975, 850 | -1.19, -0.88, 0.55, 0.05, 0.24 |
| SW | 1042.5 | - | - | 875, 925, 300, 975, 400 | 0.97, 0.66, -0.50, 0.43, -0.45 |
| LSW | 1284.1 | - | - | 150, 600, 975, 450, 775 | -1.81, -2.06, 0.05, -0.78, 1.00 |
| (released) | | | | 775, 1950, 1725, 1975, 1525 | -0.11, 0.78, 0.12, 0.09, 0.44 |
| (control) | | | | 1425, 1000, 1900, 1650, 1925 | -0.01, -0.58, 0.71, 0.52, 0.43 |

### Rule effects (mean over instances of the R / N makespan ratio) and exact ties

| set | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix best rule | 1.81 | 1.82 | 1.20 | 2.40 | 1.89 | 2.12 | 2.65 |
| pub_SD2_test_10x5+mix policy | 2.11 | 2.23 | 1.26 | 2.91 | 2.18 | 2.58 | 3.20 |
| pub_SD2_test_20x10+mix best rule | 1.80 | 2.01 | 1.17 | 2.52 | 1.88 | 2.26 | 2.72 |
| pub_SD2_test_20x10+mix policy | 2.35 | 2.70 | 1.19 | 3.46 | 2.44 | 3.04 | 3.77 |
| pub_Brandimarte best rule | 1.56 | 1.74 | 1.19 | 2.23 | 1.68 | 1.99 | 2.40 |
| pub_Brandimarte policy | 1.67 | 1.96 | 1.20 | 2.47 | 1.78 | 2.24 | 2.68 |
| pub_Hurink_edata best rule | 1.55 | 1.60 | 1.24 | 2.03 | 1.69 | 1.92 | 2.30 |
| pub_Hurink_edata policy | 1.52 | 1.64 | 1.24 | 2.06 | 1.67 | 1.95 | 2.36 |
| pub_Hurink_rdata best rule | 1.72 | 1.65 | 1.20 | 2.22 | 1.82 | 1.96 | 2.46 |
| pub_Hurink_rdata policy | 1.71 | 1.72 | 1.21 | 2.29 | 1.81 | 2.02 | 2.55 |
| pub_Hurink_vdata best rule | 1.90 | 1.56 | 1.16 | 2.28 | 1.95 | 1.83 | 2.46 |
| pub_Hurink_vdata policy | 1.85 | 1.60 | 1.17 | 2.31 | 1.90 | 1.89 | 2.50 |

Instances on which every policy seed, every specialist seed and the best rule give the same makespan, and the correlation over instances of the seed-mean policy and specialist makespans:

| set | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix | 0 (0.94) | 0 (1.00) | 1 (0.86) | 1 (0.96) | 7 (0.97) | 3 (1.00) | 0 (0.88) | 0 (0.95) |
| pub_SD2_test_20x10+mix | 0 (0.90) | 0 (0.99) | 0 (0.72) | 0 (0.91) | 0 (0.92) | 0 (0.99) | 0 (0.81) | 0 (0.87) |
| pub_Brandimarte | 1 (1.00) | 0 (0.99) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_edata | 0 (1.00) | 1 (1.00) | 1 (1.00) | 0 (1.00) | 0 (1.00) | 8 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_rdata | 0 (1.00) | 0 (1.00) | 0 (1.00) | 1 (1.00) | 0 (1.00) | 3 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_vdata | 0 (1.00) | 8 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 2 (1.00) | 0 (1.00) | 0 (1.00) |

## Combined verdict against the per-instance best rule

d = policy minus per-instance best rule, over the rule (five seeds). ahead: the hierarchical 95% CI lies below zero and the Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t and hierarchical bootstrap) are both below 0.05. behind: the CI lies above zero and the Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family over the eight combinations) are both below 0.05. not distinguishable: otherwise.

### pub_SD2_test_10x5+mix

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -14.18 | [-15.76, -12.54] | 1.6e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| L | -0.33 | [-1.32, +0.65] | 0.23 / 1 | 0.95 / 0.75 | 1 / 1 | not distinguishable |
| S | +5.56 | [+3.91, +7.27] | 1 / 1 | 5.1e-05 / 5e-05 | 0.00036 / 0.0004 | behind |
| W | -9.49 | [-11.29, -7.74] | 6.6e-07 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LS | +4.04 | [+2.77, +5.43] | 1 / 1 | 3.6e-05 / 5e-05 | 0.00029 / 0.0004 | behind |
| LW | -1.56 | [-2.79, -0.32] | 0.00024 / 0.048 | 1 / 0.99 | 1 / 1 | ahead |
| SW | +4.15 | [+2.74, +5.62] | 1 / 1 | 7.1e-05 / 5e-05 | 0.00042 / 0.0004 | behind |
| LSW | +4.00 | [+2.29, +5.79] | 1 / 1 | 0.00036 / 5e-05 | 0.0018 / 0.0004 | behind |

Ahead on 3, behind on 4, not distinguishable on 1 of 8 combinations.

### pub_SD2_test_20x10+mix

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -23.41 | [-24.86, -21.91] | 2.7e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| L | +0.23 | [-0.70, +1.13] | 1 / 1 | 0.21 / 0.3 | 0.85 / 1 | not distinguishable |
| S | +3.11 | [+2.13, +4.14] | 1 / 1 | 1.4e-05 / 5e-05 | 0.00011 / 0.0004 | behind |
| W | -22.17 | [-23.57, -20.72] | 5e-07 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LS | +5.25 | [+3.93, +6.69] | 1 / 1 | 2.1e-05 / 5e-05 | 0.00014 / 0.0004 | behind |
| LW | -0.51 | [-1.43, +0.43] | 0.064 / 0.85 | 0.99 / 0.86 | 1 / 1 | not distinguishable |
| SW | +2.85 | [+1.75, +4.00] | 1 / 1 | 0.00032 / 5e-05 | 0.0016 / 0.0004 | behind |
| LSW | +6.10 | [+4.70, +7.49] | 1 / 1 | 3.4e-05 / 5e-05 | 0.0002 / 0.0004 | behind |

Ahead on 2, behind on 4, not distinguishable on 2 of 8 combinations.

### pub_Brandimarte

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -5.64 | [-11.24, -0.73] | 9.1e-05 / 0.074 | 1 / 0.99 | 1 / 1 | not distinguishable |
| L | -0.13 | [-4.56, +4.08] | 1 / 1 | 0.57 / 0.52 | 1 / 1 | not distinguishable |
| S | +5.79 | [+1.90, +10.10] | 1 / 1 | 0.0045 / 0.0013 | 0.022 / 0.0094 | behind |
| W | -4.50 | [-8.45, -1.01] | 3.1e-05 / 0.039 | 1 / 1 | 1 / 1 | ahead |
| LS | +4.02 | [+0.88, +7.38] | 1 / 1 | 0.0017 / 0.007 | 0.0099 / 0.035 | behind |
| LW | -1.70 | [-6.67, +3.13] | 0.25 / 1 | 0.96 / 0.75 | 1 / 1 | not distinguishable |
| SW | +5.10 | [+2.21, +8.56] | 1 / 1 | 0.00028 / 0.0002 | 0.002 / 0.0016 | behind |
| LSW | +4.61 | [+1.45, +8.24] | 1 / 1 | 1e-05 / 0.0014 | 8e-05 / 0.0094 | behind |

Ahead on 1, behind on 4, not distinguishable on 3 of 8 combinations.

### pub_Hurink_edata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | +1.28 | [-0.04, +2.62] | 1 / 1 | 0.01 / 0.028 | 0.031 / 0.083 | not distinguishable |
| L | -0.31 | [-1.55, +0.94] | 0.19 / 1 | 0.98 / 0.68 | 1 / 1 | not distinguishable |
| S | +3.76 | [+2.89, +4.62] | 1 / 1 | 1e-05 / 5e-05 | 8.1e-05 / 0.0004 | behind |
| W | +1.74 | [+0.46, +3.04] | 1 / 1 | 0.0013 / 0.0041 | 0.0052 / 0.017 | behind |
| LS | +2.47 | [+1.38, +3.58] | 1 / 1 | 1.1e-05 / 5e-05 | 8.1e-05 / 0.0004 | behind |
| LW | -0.32 | [-1.79, +1.16] | 1 / 1 | 0.8 / 0.67 | 1 / 1 | not distinguishable |
| SW | +2.81 | [+1.70, +3.98] | 1 / 1 | 0.00036 / 5e-05 | 0.0018 / 0.0004 | behind |
| LSW | +3.80 | [+2.51, +5.15] | 1 / 1 | 3.9e-05 / 5e-05 | 0.00023 / 0.0004 | behind |

Ahead on 0, behind on 5, not distinguishable on 3 of 8 combinations.

### pub_Hurink_rdata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | +0.79 | [-0.18, +1.80] | 1 / 1 | 0.015 / 0.057 | 0.046 / 0.17 | not distinguishable |
| L | -0.20 | [-1.26, +0.81] | 1 / 1 | 0.77 / 0.64 | 0.77 / 0.95 | not distinguishable |
| S | +5.11 | [+3.83, +6.40] | 1 / 1 | 1.3e-05 / 5e-05 | 0.0001 / 0.0004 | behind |
| W | +2.08 | [+0.93, +3.28] | 1 / 1 | 0.00065 / 0.00055 | 0.0026 / 0.0022 | behind |
| LS | +4.50 | [+2.95, +6.16] | 1 / 1 | 0.00015 / 5e-05 | 0.00084 / 0.0004 | behind |
| LW | +0.04 | [-1.41, +1.44] | 1 / 1 | 0.38 / 0.47 | 0.77 / 0.95 | not distinguishable |
| SW | +4.12 | [+2.79, +5.57] | 1 / 1 | 0.00014 / 5e-05 | 0.00084 / 0.0004 | behind |
| LSW | +4.50 | [+2.86, +6.15] | 1 / 1 | 2.9e-05 / 5e-05 | 0.00021 / 0.0004 | behind |

Ahead on 0, behind on 5, not distinguishable on 3 of 8 combinations.

### pub_Hurink_vdata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | +1.93 | [+1.19, +2.84] | 1 / 1 | 0.0014 / 5e-05 | 0.0055 / 0.0004 | behind |
| L | -0.94 | [-1.91, -0.08] | 0.022 / 0.13 | 1 / 0.98 | 1 / 1 | not distinguishable |
| S | +5.08 | [+3.39, +6.99] | 1 / 1 | 0.00023 / 5e-05 | 0.0016 / 0.0004 | behind |
| W | +2.61 | [+1.31, +4.39] | 1 / 1 | 0.0081 / 5e-05 | 0.024 / 0.0004 | behind |
| LS | +3.58 | [+2.02, +5.21] | 1 / 1 | 0.00075 / 5e-05 | 0.0037 / 0.0004 | behind |
| LW | -0.99 | [-2.11, +0.08] | 0.048 / 0.24 | 0.99 / 0.97 | 1 / 1 | not distinguishable |
| SW | +5.46 | [+3.66, +7.33] | 1 / 1 | 0.00034 / 5e-05 | 0.002 / 0.0004 | behind |
| LSW | +3.88 | [+2.50, +5.37] | 1 / 1 | 5.9e-05 / 5e-05 | 0.00047 / 0.0004 | behind |

Ahead on 0, behind on 6, not distinguishable on 2 of 8 combinations.


## 8. Equal-experience policy (pubx-eq, 6000 updates on N,L,S,W,LS,LW)

The released recipe with 6000 updates instead of 2000, so that each of its six combinations gets as many episodes as a specialist (1000 updates) gets of its one. Seeds 301, 302, 303, 304, 305, seed s against seed s; same estimators, margins and streams as sections 1 to 7. Configurations equal those of pubx-v1 of the same seed except for max_updates, model_suffix.

- Checks: pub_SD2_test_10x5+mix: 5 summaries, 5 regime-N arrays at or above the bound; pub_SD2_test_20x10+mix: 5 summaries, 5 regime-N arrays at or above the bound; pub_Brandimarte: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_edata: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_rdata: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_vdata: 5 summaries, 5 regime-N arrays at or above the bound; pub_SD2_test_10x5+mix_lag050: 5 summaries, 5 regime-N arrays at or above the bound, free check 10 N and S arrays bit for bit.

### pub_SD2_test_10x5+mix (in distribution)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.12 [+0.37, +3.97] | inferior | +1.21 [-0.33, +2.64] | [-0.44, +2.87] | inconclusive | inconclusive | neither | +1.69, +0.77, -0.88, +1.92, +2.57 | -0.55 [-1.80, +0.77] | inconclusive | non-inferior only | neither |
| L | +0.10 [-0.63, +0.82] | equivalent | +0.36 [-0.41, +1.19] | [-0.26, +0.99] | inconclusive | inconclusive | neither | +0.25, -0.22, +0.18, +1.15, +0.47 | +0.35 [-0.29, +0.99] | inconclusive | equivalent | neither |
| S | +0.15 [-1.24, +1.60] | inconclusive | -0.41 [-1.67, +0.84] | [-0.96, +0.15] | inconclusive | non-inferior only | neither | -0.24, -0.56, -0.08, -1.12, -0.04 | -0.30 [-1.40, +0.83] | inconclusive | non-inferior only | neither |
| W | +0.73 [-0.62, +2.13] | inconclusive | -0.06 [-1.67, +1.56] | [-1.75, +1.62] | inconclusive | inconclusive | neither | +1.93, -1.52, -0.19, +0.50, -1.03 | -0.34 [-1.69, +1.07] | inconclusive | inconclusive | neither |
| LS | -0.50 [-1.63, +0.77] | non-inferior only | -0.30 [-1.46, +0.87] | [-1.50, +0.90] | inconclusive | non-inferior only | neither | +0.45, +0.37, -0.60, +0.15, -1.86 | +0.39 [-0.53, +1.36] | inconclusive | inconclusive | neither |
| LW | -0.18 [-0.86, +0.49] | equivalent | -0.14 [-0.74, +0.48] | [-0.45, +0.17] | non-inferior only | equivalent | neither | -0.38, -0.20, -0.35, +0.15, +0.11 | +0.21 [-0.42, +0.84] | inconclusive | equivalent | neither |
| SW | +0.12 [-1.34, +1.56] | inconclusive | -0.35 [-1.59, +0.88] | [-1.22, +0.51] | inconclusive | non-inferior only | neither | -0.01, -0.91, +0.24, +0.20, -1.28 | -0.17 [-1.17, +0.87] | inconclusive | non-inferior only | neither |
| LSW | +0.41 [-1.39, +2.21] | inconclusive | -0.01 [-1.50, +1.49] | [-1.17, +1.14] | inconclusive | inconclusive | neither | +0.03, +0.74, -1.42, +0.88, -0.30 | -0.15 [-1.32, +0.99] | inconclusive | non-inferior only | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | +0.37 [-0.11, +0.83] | +0.04 [-0.45, +0.53] | inconclusive | equivalent | False / False | N,L,S,W,LS,SW,LSW | -0.07 [-0.52, +0.39] | non-inferior only | equivalent |
| Train | +0.40 [-0.04, +0.84] | +0.11 [-0.39, +0.61] | inconclusive | equivalent | False / False | N,L,S,W,LS | -0.04 [-0.55, +0.50] | inconclusive | equivalent |
| Held | +0.27 [-0.87, +1.39] | -0.18 [-1.16, +0.79] | inconclusive | non-inferior only | False / False | SW,LSW | -0.16 [-0.92, +0.64] | inconclusive | equivalent |

Regime N, gap to best known: equal-experience 26.66 (sd 1.33), released 27.81, N specialist 25.60.

### pub_SD2_test_20x10+mix (size transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +3.73 [+2.23, +5.49] | inferior | +1.86 [+0.80, +2.96] | [+0.69, +3.03] | inferior | inferior | behind | +1.67, +1.90, +1.47, +0.86, +3.39 | -1.60 [-2.72, -0.50] | non-inferior only | non-inferior only | ahead |
| L | -0.12 [-0.81, +0.55] | equivalent | -0.77 [-1.50, -0.17] | [-1.32, -0.22] | non-inferior only | non-inferior only | ahead | -0.90, -0.43, -1.50, -0.53, -0.51 | -0.56 [-1.24, +0.11] | non-inferior only | non-inferior only | neither |
| S | -0.21 [-1.20, +0.84] | non-inferior only | -0.07 [-1.34, +1.25] | [-1.44, +1.29] | inconclusive | inconclusive | neither | -1.25, -1.18, +0.47, +1.26, +0.33 | +0.42 [-0.74, +1.56] | inconclusive | inconclusive | neither |
| W | -0.25 [-2.72, +1.75] | inconclusive | -1.32 [-3.55, +0.61] | [-4.27, +1.63] | inconclusive | non-inferior only | neither | -0.38, -5.19, +1.12, -0.51, -1.63 | -0.79 [-2.28, +0.57] | inconclusive | non-inferior only | neither |
| LS | -0.15 [-1.41, +1.16] | inconclusive | +0.22 [-1.23, +1.65] | [-0.98, +1.42] | inconclusive | inconclusive | neither | -0.06, +1.52, -0.97, +0.82, -0.22 | +0.72 [-0.67, +2.06] | inconclusive | inconclusive | neither |
| LW | +0.25 [-0.44, +0.93] | equivalent | -0.11 [-0.94, +0.70] | [-1.02, +0.80] | inconclusive | equivalent | neither | +0.59, +0.46, -0.28, -0.08, -1.25 | -0.22 [-1.06, +0.61] | inconclusive | non-inferior only | neither |
| SW | +0.02 [-1.11, +1.23] | inconclusive | -0.44 [-1.48, +0.61] | [-1.10, +0.22] | inconclusive | non-inferior only | neither | -0.54, +0.39, -0.63, -1.06, -0.36 | -0.13 [-1.28, +1.08] | inconclusive | inconclusive | neither |
| LSW | -0.08 [-1.57, +1.42] | inconclusive | +0.02 [-1.19, +1.21] | [-0.76, +0.80] | inconclusive | inconclusive | neither | -0.11, +0.54, -0.93, -0.04, +0.63 | +0.56 [-0.87, +2.00] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | +0.40 [-0.06, +0.88] | -0.08 [-0.42, +0.27] | equivalent | equivalent | False / False | N,L,S,W,LS,SW,LSW | -0.20 [-0.63, +0.20] | non-inferior only | equivalent |
| Train | +0.54 [-0.07, +1.15] | -0.03 [-0.44, +0.38] | equivalent | equivalent | False / False | N,L,S,W,LS | -0.34 [-0.87, +0.17] | non-inferior only | equivalent |
| Held | -0.03 [-0.90, +0.84] | -0.21 [-1.05, +0.61] | inconclusive | non-inferior only | False / False | SW,LSW | +0.21 [-0.58, +0.99] | inconclusive | equivalent |

Regime N, gap to best known: equal-experience 34.62 (sd 0.46), released 37.11, N specialist 32.45.

### pub_Brandimarte (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.28 [-1.48, +2.30] | inconclusive | -0.89 [-2.87, +0.90] | [-2.67, +0.88] | inconclusive | non-inferior only | neither | +0.23, -3.21, -1.12, +0.30, -0.66 | -1.11 [-2.94, +0.48] | non-inferior only | non-inferior only | neither |
| L | -3.73 [-9.35, +0.49] | non-inferior only | -3.61 [-9.52, +0.16] | [-9.04, +1.81] | non-inferior only | non-inferior only | neither | -5.00, -0.98, -10.71, -0.63, -0.74 | +0.20 [-1.92, +2.50] | inconclusive | inconclusive | neither |
| S | -0.58 [-5.25, +4.77] | inconclusive | -1.89 [-6.32, +2.10] | [-5.29, +1.50] | inconclusive | inconclusive | neither | +1.37, -3.78, +0.71, -4.60, -3.16 | -1.03 [-5.13, +2.67] | inconclusive | inconclusive | neither |
| W | -1.42 [-2.97, -0.12] | non-inferior only | -0.98 [-3.24, +0.90] | [-3.00, +1.03] | inconclusive | non-inferior only | neither | -0.67, -3.27, -1.91, +0.72, +0.21 | +0.49 [-0.94, +1.89] | inconclusive | inconclusive | neither |
| LS | -0.36 [-4.19, +3.32] | inconclusive | -0.88 [-4.59, +2.78] | [-3.59, +1.82] | inconclusive | inconclusive | neither | -2.58, -3.17, -1.21, +0.39, +2.15 | -0.17 [-3.89, +3.16] | inconclusive | inconclusive | neither |
| LW | +0.18 [-3.33, +3.79] | inconclusive | +0.17 [-3.15, +3.50] | [-2.42, +2.77] | inconclusive | inconclusive | neither | +0.64, +3.31, -2.42, -0.07, -0.60 | +0.14 [-2.05, +2.53] | inconclusive | inconclusive | neither |
| SW | +1.81 [-1.56, +5.03] | inconclusive | -0.55 [-3.71, +2.59] | [-2.88, +1.78] | inconclusive | inconclusive | neither | -2.63, -1.80, -1.16, +1.26, +1.59 | -2.13 [-4.59, +0.75] | inconclusive | non-inferior only | neither |
| LSW | -1.19 [-4.40, +1.83] | inconclusive | -0.44 [-4.42, +4.22] | [-4.44, +3.56] | inconclusive | inconclusive | neither | -4.85, -2.54, +0.24, +2.77, +2.19 | +0.79 [-2.20, +4.83] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.62 [-2.08, +0.92] | -1.13 [-2.78, +0.34] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW,SW,LSW | -0.35 [-1.33, +0.95] | inconclusive | non-inferior only |
| Train | -0.94 [-2.67, +0.82] | -1.35 [-3.10, +0.31] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW | -0.25 [-1.21, +0.86] | inconclusive | non-inferior only |
| Held | +0.31 [-2.26, +2.71] | -0.49 [-3.20, +2.25] | inconclusive | inconclusive | False / False | SW,LSW | -0.67 [-2.75, +2.02] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 10.22 (sd 0.96), released 11.66, N specialist 11.38.

### pub_Hurink_edata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.29 [-0.68, +1.23] | inconclusive | -0.29 [-1.17, +0.78] | [-1.40, +0.83] | inconclusive | non-inferior only | neither | +1.09, -1.03, +0.14, -0.72, -0.92 | -0.45 [-1.35, +0.46] | non-inferior only | non-inferior only | neither |
| L | -0.52 [-1.64, +0.48] | non-inferior only | -0.55 [-1.71, +0.48] | [-1.79, +0.70] | non-inferior only | non-inferior only | neither | +0.24, +0.04, -2.15, -0.92, +0.06 | +0.06 [-0.56, +0.72] | inconclusive | equivalent | neither |
| S | -0.67 [-1.47, +0.12] | non-inferior only | -0.45 [-1.55, +0.55] | [-1.47, +0.57] | inconclusive | non-inferior only | neither | +0.43, -1.40, -0.21, -1.24, +0.16 | +0.31 [-0.57, +1.21] | inconclusive | inconclusive | neither |
| W | -1.27 [-2.42, -0.19] | non-inferior only | -1.17 [-2.43, -0.02] | [-2.24, -0.09] | non-inferior only | non-inferior only | ahead | -2.46, -1.49, -0.31, -1.08, -0.49 | +0.25 [-0.65, +1.11] | inconclusive | inconclusive | neither |
| LS | -0.76 [-1.69, +0.15] | non-inferior only | -0.57 [-1.59, +0.41] | [-1.48, +0.34] | non-inferior only | non-inferior only | neither | -0.99, +0.03, -0.04, -1.66, -0.20 | +0.30 [-0.42, +0.96] | inconclusive | equivalent | neither |
| LW | -0.53 [-1.52, +0.49] | non-inferior only | -0.18 [-1.39, +1.06] | [-1.59, +1.23] | inconclusive | inconclusive | neither | +1.20, +0.78, -0.61, -0.77, -1.52 | +0.43 [-0.47, +1.41] | inconclusive | inconclusive | neither |
| SW | -0.08 [-1.03, +0.96] | non-inferior only | -0.14 [-1.00, +0.77] | [-1.02, +0.74] | inconclusive | equivalent | neither | -0.25, +0.47, +0.69, -0.64, -0.96 | +0.04 [-0.71, +0.81] | inconclusive | equivalent | neither |
| LSW | +0.78 [-0.27, +1.95] | inconclusive | +0.44 [-0.63, +1.64] | [-0.56, +1.43] | inconclusive | inconclusive | neither | +0.30, +1.28, +0.42, +1.00, -0.81 | -0.16 [-1.16, +1.01] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.35 [-0.67, -0.00] | -0.36 [-0.74, +0.02] | non-inferior only | equivalent | False / False | N,L,S,W,LS,LW,LSW | +0.10 [-0.20, +0.39] | equivalent | equivalent |
| Train | -0.58 [-0.98, -0.15] | -0.53 [-0.97, -0.10] | non-inferior only | equivalent | False / False | N,L,S,W,LS,LW | +0.15 [-0.21, +0.48] | equivalent | equivalent |
| Held | +0.35 [-0.37, +1.10] | +0.15 [-0.61, +0.93] | inconclusive | equivalent | False / False | LSW | -0.06 [-0.61, +0.59] | inconclusive | equivalent |

Regime N, gap to best known: equal-experience 15.12 (sd 0.96), released 15.78, N specialist 15.58.

### pub_Hurink_rdata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.51 [-0.37, +1.43] | inconclusive | +0.69 [-0.38, +1.89] | [-0.58, +1.96] | inconclusive | inconclusive | neither | +2.29, -0.49, +0.80, +0.61, +0.24 | +0.33 [-0.69, +1.48] | inconclusive | inconclusive | neither |
| L | -1.11 [-2.40, +0.06] | non-inferior only | -0.94 [-2.52, +0.37] | [-2.61, +0.72] | non-inferior only | non-inferior only | neither | -0.23, -0.61, -2.89, -1.59, +0.60 | +0.21 [-0.69, +1.18] | inconclusive | inconclusive | neither |
| S | +0.07 [-1.27, +1.33] | inconclusive | +0.31 [-0.87, +1.50] | [-0.18, +0.81] | inconclusive | inconclusive | neither | -0.12, +0.37, +0.14, +0.94, +0.24 | +0.52 [-0.84, +1.85] | inconclusive | inconclusive | neither |
| W | +0.11 [-1.09, +1.25] | inconclusive | +0.09 [-1.11, +1.28] | [-1.11, +1.30] | inconclusive | inconclusive | neither | +1.31, -1.04, +0.32, -0.73, +0.62 | +0.13 [-0.89, +1.17] | inconclusive | inconclusive | neither |
| LS | +0.21 [-1.10, +1.56] | inconclusive | -0.04 [-1.41, +1.38] | [-1.43, +1.35] | inconclusive | inconclusive | neither | -0.08, -0.05, +0.15, +1.47, -1.68 | +0.07 [-1.59, +1.82] | inconclusive | inconclusive | neither |
| LW | -0.52 [-1.71, +0.62] | non-inferior only | -0.35 [-1.48, +0.80] | [-1.46, +0.75] | inconclusive | non-inferior only | neither | +0.87, +0.31, -0.88, -1.00, -1.09 | +0.31 [-0.67, +1.48] | inconclusive | inconclusive | neither |
| SW | -0.46 [-1.91, +0.97] | non-inferior only | -0.11 [-1.47, +1.21] | [-1.29, +1.08] | inconclusive | inconclusive | neither | +0.91, -0.56, -1.38, -0.25, +0.75 | +0.58 [-0.48, +1.70] | inconclusive | inconclusive | neither |
| LSW | +0.23 [-1.25, +1.67] | inconclusive | +0.53 [-0.82, +1.88] | [-0.63, +1.69] | inconclusive | inconclusive | neither | +1.05, +0.91, -1.00, +1.37, +0.32 | +0.72 [-0.76, +2.32] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.12 [-0.59, +0.36] | +0.02 [-0.48, +0.57] | inconclusive | equivalent | False / False | N,L,S,W,LS,LW,SW,LSW | +0.36 [-0.15, +0.92] | inconclusive | equivalent |
| Train | -0.12 [-0.61, +0.35] | -0.04 [-0.54, +0.50] | non-inferior only | equivalent | False / False | N,L,S,W,LS,LW | +0.26 [-0.29, +0.91] | inconclusive | equivalent |
| Held | -0.12 [-1.21, +1.04] | +0.21 [-0.85, +1.25] | inconclusive | inconclusive | False / False | SW,LSW | +0.65 [-0.24, +1.61] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 14.04 (sd 0.88), released 13.81, N specialist 13.41.

### pub_Hurink_vdata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.32 [-0.49, +1.30] | inconclusive | +0.50 [-0.49, +1.52] | [-0.85, +1.85] | inconclusive | inconclusive | neither | +2.09, -0.14, +0.47, +0.86, -0.79 | +0.25 [-0.90, +1.46] | inconclusive | inconclusive | neither |
| L | -0.61 [-1.63, +0.23] | non-inferior only | -0.88 [-1.76, -0.12] | [-1.47, -0.28] | non-inferior only | non-inferior only | ahead | -1.29, -0.50, -1.49, -0.64, -0.46 | -0.22 [-0.80, +0.33] | non-inferior only | equivalent | neither |
| S | -0.60 [-1.96, +0.72] | non-inferior only | -0.78 [-2.66, +1.00] | [-3.08, +1.52] | inconclusive | non-inferior only | neither | +0.54, -3.27, +0.46, -2.28, +0.64 | +0.18 [-1.54, +1.86] | inconclusive | inconclusive | neither |
| W | +0.63 [-1.01, +2.10] | inconclusive | +0.80 [-0.82, +2.77] | [-1.66, +3.26] | inconclusive | inconclusive | neither | +4.09, -1.15, -0.11, +0.29, +0.88 | +0.33 [-1.52, +2.08] | inconclusive | inconclusive | neither |
| LS | -0.02 [-1.53, +1.66] | inconclusive | -0.67 [-1.99, +0.66] | [-2.00, +0.66] | inconclusive | non-inferior only | neither | +0.81, -0.86, -1.35, -0.05, -1.89 | -0.43 [-1.65, +1.00] | inconclusive | non-inferior only | neither |
| LW | +0.09 [-0.87, +1.02] | inconclusive | +0.19 [-0.79, +1.09] | [-0.79, +1.17] | inconclusive | inconclusive | neither | +1.30, +0.30, +0.05, -0.92, +0.22 | +0.17 [-0.58, +0.91] | inconclusive | equivalent | neither |
| SW | -1.24 [-2.86, +0.41] | non-inferior only | -0.53 [-2.04, +0.92] | [-1.82, +0.76] | inconclusive | non-inferior only | neither | +0.31, +0.18, +0.10, -1.24, -2.02 | +1.04 [-0.77, +3.01] | inconclusive | inconclusive | neither |
| LSW | +0.40 [-0.86, +1.68] | inconclusive | +0.34 [-0.87, +1.58] | [-0.27, +0.94] | inconclusive | inconclusive | neither | +0.49, +1.06, -0.20, +0.34, -0.01 | +0.13 [-1.05, +1.33] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.13 [-0.70, +0.48] | -0.13 [-0.71, +0.60] | inconclusive | equivalent | False / False | N,L,S,W,LS,LW,SW,LSW | +0.18 [-0.47, +0.79] | inconclusive | equivalent |
| Train | -0.03 [-0.69, +0.64] | -0.14 [-0.83, +0.68] | inconclusive | equivalent | False / False | N,L,S,W,LS,LW | +0.05 [-0.61, +0.66] | inconclusive | equivalent |
| Held | -0.42 [-1.36, +0.51] | -0.10 [-1.08, +0.90] | inconclusive | non-inferior only | False / False | SW,LSW | +0.58 [-0.41, +1.70] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 5.65 (sd 1.04), released 5.48, N specialist 5.19.

### Validation at the kept checkpoints (20 SD2 10x5 validation records)

| comb. | equal-exp. minus specialist % per seed | mean | released minus specialist (mean) |
|---|---|---|---|
| N | +1.50, +5.03, +0.45, -0.53, +3.10 | +1.91 | +2.72 |
| L | -0.48, +0.56, -0.35, +1.02, +0.21 | +0.19 | +0.42 |
| S | -1.33, +0.88, +0.40, +1.17, +0.89 | +0.40 | +1.32 |
| W | +0.93, +0.06, +1.83, +2.14, -1.68 | +0.66 | +1.92 |
| LS | +2.61, +0.16, +0.92, +3.33, -0.41 | +1.32 | +1.23 |
| LW | +0.38, -0.19, +0.27, -0.04, +0.68 | +0.22 | +1.21 |

Kept update per seed: 3725, 5975, 4700, 3375, 3250 (of 6000); gain in the last quarter %: -0.00, 0.05, 0.40, -0.05, -1.11.

## 9. Gap to the references (pub_SD2_test_10x5+mix)

Reference: the best schedule CP-SAT finds within 150 s on 2 workers (compact model, warm-started from the best of the four rules; scripts/p91_public_cpsat_refs.py, records it marks valid only; folder or_solution/PUBFAMILY/pub_SD2_test_10x5+mix_cpsat150s_w2). d = 100 (method - reference) / reference per instance; trained arms (five runs): crossed hierarchical 95% CI; rules: instance-level 95% CI; interval and direction only.

| comb. | n | proved | at limit (mean gap %) | policy | = / > / < % | control | specialist | best rule | FIFO | MOR | SPT | MWKR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 100 | 100 | 0 | +27.81 [+26.07, +29.72] behind | 0.0 / 100.0 / 0.0 | +26.37 [+24.70, +28.16] behind | +25.60 [+24.03, +27.29] behind | +49.83 [+47.15, +52.50] behind | +68.85 [+65.61, +72.10] behind | +68.97 [+65.38, +72.57] behind | +58.30 [+54.79, +61.89] behind | +65.77 [+62.20, +69.33] behind |
| L | 100 | 100 | 0 | +6.95 [+5.65, +8.34] behind | 9.6 / 90.4 / 0.0 | +7.35 [+5.98, +8.84] behind | +6.99 [+5.61, +8.50] behind | +7.52 [+5.97, +9.20] behind | +15.75 [+13.37, +18.23] behind | +15.67 [+13.14, +18.37] behind | +11.59 [+9.46, +13.87] behind | +16.19 [+13.77, +18.62] behind |
| S | 100 | 0 | 100 (44.31) | +50.01 [+47.15, +53.02] behind | 0.0 / 100.0 / 0.0 | +49.34 [+46.48, +52.35] behind | +50.40 [+47.86, +53.14] behind | +42.52 [+39.93, +45.29] behind | +53.09 [+50.12, +56.11] behind | +52.01 [+48.80, +55.32] behind | +53.43 [+50.10, +57.02] behind | +52.60 [+49.15, +56.19] behind |
| W | 100 | 100 | 0 | +28.36 [+25.46, +31.39] behind | 2.6 / 97.4 / 0.0 | +27.85 [+24.85, +30.93] behind | +27.96 [+25.12, +31.00] behind | +42.81 [+39.18, +46.43] behind | +58.25 [+53.59, +62.89] behind | +60.14 [+55.59, +64.76] behind | +56.20 [+51.93, +60.62] behind | +52.66 [+48.10, +57.17] behind |
| LS | 100 | 57 | 43 (4.66) | +31.83 [+28.21, +35.48] behind | 0.0 / 100.0 / 0.0 | +32.69 [+29.03, +36.42] behind | +32.80 [+29.24, +36.44] behind | +26.88 [+23.55, +30.47] behind | +34.01 [+29.93, +38.30] behind | +32.95 [+29.07, +37.07] behind | +36.13 [+31.86, +40.60] behind | +34.78 [+31.14, +38.69] behind |
| LW | 100 | 100 | 0 | +6.80 [+5.52, +8.13] behind | 15.8 / 84.2 / 0.0 | +7.27 [+5.84, +8.80] behind | +7.23 [+5.77, +8.80] behind | +8.85 [+7.17, +10.63] behind | +17.38 [+14.64, +20.27] behind | +16.12 [+13.50, +18.85] behind | +15.72 [+12.96, +18.65] behind | +16.74 [+14.08, +19.52] behind |
| SW | 100 | 1 | 99 (39.53) | +41.01 [+38.41, +43.69] behind | 0.0 / 100.0 / 0.0 | +41.36 [+38.68, +44.12] behind | +41.47 [+38.92, +44.15] behind | +35.61 [+33.32, +37.98] behind | +45.22 [+42.61, +47.90] behind | +44.75 [+41.98, +47.67] behind | +45.00 [+41.90, +48.24] behind | +44.43 [+41.30, +47.66] behind |
| LSW | 100 | 46 | 54 (7.89) | +36.46 [+33.07, +39.77] behind | 0.0 / 100.0 / 0.0 | +36.45 [+33.15, +39.68] behind | +36.57 [+33.15, +39.95] behind | +31.69 [+28.42, +34.96] behind | +40.29 [+36.75, +43.91] behind | +39.24 [+35.55, +43.02] behind | +40.46 [+36.50, +44.42] behind | +39.41 [+35.57, +43.32] behind |
| pooled | 800 | 504 | 296 | +28.65 [+27.43, +29.84] behind | 3.5 / 96.5 / 0.0 | +28.58 [+27.36, +29.81] behind | +28.63 [+27.46, +29.79] behind | +30.71 [+29.45, +31.98] behind | +41.61 [+39.98, +43.26] behind | +41.23 [+39.61, +42.87] behind | +39.60 [+38.08, +41.15] behind | +40.32 [+38.52, +42.08] behind |

Regime N against the best-known values (100 references): below 0, equal 100, above 0 (0 of them where the table marks the value proved optimal); proved optimal here and in the table: 96.
Policy shorter than the reference: 0 (run, instance) cases, listed in the JSON (possible where the reference is not proved optimal).
Records worth a look: worse than warm start 0; best rule shorter 0; ended before limit unproved 0; other status 0; records not counted 0.
