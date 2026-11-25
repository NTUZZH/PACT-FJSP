# Public-instance family under the active mask: policy, control, specialists and rules

Training seeds 301, 302, 303, 304, 305 per arm, the active mask (mask_mode active; exact first-decision and tie options; rules rolled out under it by scripts/p109_public_active_rules.py), greedy decoding. d = 100 (a - b) / b per training seed and base instance, seed s against seed s; positive = first arm slower. Hierarchical 95% intervals (seeds x base instances, B = 20000, default_rng(11)), seed-level t intervals (4 df), verdicts at 0.5% / 1% on the hierarchical interval, intersection-union test for joint claims, Holm over the eight combinations for superiority over the rules.

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
| N | +1.12 | [-0.27, +2.52] | [-0.53, +2.78] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.2 | +2.77, +1.83, -0.54, +0.12, +1.45 |
| L | +0.58 | [-0.05, +1.21] | [+0.01, +1.14] | inconclusive | inconclusive | inferior | inferior | neither | 27.8 | +1.01, +0.66, +0.92, +0.43, -0.13 |
| S | +0.51 | [-0.50, +1.64] | [-0.38, +1.39] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.6 | +0.34, -0.03, +0.89, -0.20, +1.54 |
| W | +0.30 | [-1.06, +1.67] | [-0.49, +1.09] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 16.2 | -0.03, +0.56, -0.21, -0.11, +1.29 |
| LS | -0.20 | [-1.04, +0.68] | [-0.83, +0.43] | inconclusive | non-inferior only | non-inferior only | equivalent | neither | 4.4 | -0.29, +0.46, -0.78, +0.17, -0.55 |
| LW | -0.09 | [-1.04, +0.75] | [-0.94, +0.76] | inconclusive | non-inferior only | inconclusive | equivalent | neither | 36.6 | +0.82, -0.98, -0.49, +0.07, +0.12 |
| SW | -0.83 | [-2.23, +0.56] | [-2.60, +0.94] | inconclusive | non-inferior only | inconclusive | non-inferior only | neither | 2.8 | +0.67, -1.93, +0.16, -0.35, -2.69 |
| LSW | -0.18 | [-1.11, +0.74] | [-0.94, +0.58] | inconclusive | non-inferior only | inconclusive | equivalent | neither | 4.8 | +0.42, -1.18, +0.14, -0.28, +0.00 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | +0.15 | [-0.25, +0.58] | inconclusive | equivalent | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | +0.37 | [-0.08, +0.84] | inconclusive | equivalent | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.50 | [-1.42, +0.42] | non-inferior only | non-inferior only | False / False | False / True | SW,LSW |

### pub_SD2_test_20x10+mix (size transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.31 | [-1.05, +1.90] | [-1.76, +2.37] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.8 | +3.02, -1.03, +0.10, -1.05, +0.50 |
| L | -0.10 | [-0.75, +0.55] | [-0.91, +0.70] | inconclusive | equivalent | inconclusive | equivalent | neither | 3.0 | +0.85, -0.15, +0.07, -0.93, -0.35 |
| S | -1.32 | [-3.04, +0.47] | [-3.87, +1.22] | non-inferior only | non-inferior only | inconclusive | inconclusive | neither | 0.4 | +0.49, -3.20, -2.17, +1.23, -2.97 |
| W | -1.26 | [-2.98, +0.85] | [-3.77, +1.25] | inconclusive | non-inferior only | inconclusive | inconclusive | neither | 13.2 | -2.13, -2.74, +2.30, -1.81, -1.93 |
| LS | -0.80 | [-1.55, -0.04] | [-1.05, -0.55] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 0.2 | -0.75, -0.70, -1.04, -0.54, -0.96 |
| LW | +0.06 | [-0.47, +0.63] | [-0.18, +0.29] | inconclusive | equivalent | equivalent | equivalent | neither | 14.2 | +0.03, -0.21, +0.33, +0.07, +0.06 |
| SW | -3.07 | [-4.41, -1.84] | [-4.74, -1.41] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 1.0 | -3.67, -2.51, -2.53, -1.59, -5.07 |
| LSW | -0.74 | [-1.87, +0.44] | [-2.02, +0.54] | non-inferior only | non-inferior only | inconclusive | non-inferior only | neither | 0.4 | -1.36, -0.41, -1.28, +0.92, -1.58 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.87 | [-1.38, -0.36] | non-inferior only | non-inferior only | False / False | False / False | N,S,W,LS,SW,LSW |
| Train | -0.52 | [-1.11, +0.08] | non-inferior only | non-inferior only | False / False | False / False | N,S,W,LS |
| Held | -1.91 | [-2.93, -0.84] | non-inferior only | non-inferior only | False / False | True / True | SW,LSW |

### pub_Brandimarte (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.56 | [-0.79, +2.15] | [-1.09, +2.21] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 34.0 | +2.17, +0.96, -0.08, -1.35, +1.08 |
| L | -3.19 | [-6.83, -0.24] | [-4.97, -1.42] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 2.0 | -3.88, -3.57, -4.58, -3.13, -0.82 |
| S | -0.37 | [-2.15, +1.78] | [-2.83, +2.08] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 8.0 | +3.01, -0.89, -0.51, -1.99, -1.49 |
| W | +0.33 | [-1.56, +2.76] | [-1.20, +1.86] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 20.0 | -0.16, -0.53, +0.95, -0.80, +2.18 |
| LS | -1.84 | [-4.54, +0.35] | [-4.22, +0.54] | non-inferior only | non-inferior only | inconclusive | non-inferior only | neither | 0.0 | +0.13, -2.99, -4.59, -1.08, -0.67 |
| LW | -0.70 | [-2.54, +1.18] | [-2.17, +0.76] | inconclusive | inconclusive | inconclusive | non-inferior only | neither | 10.0 | -0.49, +1.18, -1.96, -1.27, -0.96 |
| SW | -1.19 | [-3.22, +0.67] | [-2.36, -0.02] | inconclusive | non-inferior only | non-inferior only | non-inferior only | neither | 6.0 | -1.14, -1.09, -0.03, -1.02, -2.66 |
| LSW | +0.06 | [-1.96, +2.07] | [-1.48, +1.59] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 2.0 | -0.68, +2.23, -0.57, -0.08, -0.61 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.79 | [-1.50, -0.08] | non-inferior only | non-inferior only | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.87 | [-1.89, +0.08] | non-inferior only | non-inferior only | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.57 | [-2.00, +0.70] | inconclusive | non-inferior only | False / False | False / False | SW,LSW |

### pub_Hurink_edata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.44 | [-1.90, +2.93] | [-3.19, +4.08] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.9 | +3.85, -0.72, -1.60, -2.59, +3.27 |
| L | -1.57 | [-2.71, -0.62] | [-2.49, -0.65] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 8.6 | -2.01, -1.63, -2.49, -1.16, -0.58 |
| S | +0.45 | [-0.58, +1.48] | [-0.48, +1.37] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.3 | +1.17, +0.80, +0.27, -0.75, +0.76 |
| W | +0.11 | [-1.29, +1.71] | [-1.71, +1.94] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 9.2 | +0.39, -0.06, -1.34, -0.88, +2.45 |
| LS | -0.23 | [-1.45, +0.94] | [-1.58, +1.11] | inconclusive | non-inferior only | inconclusive | inconclusive | neither | 0.9 | +0.78, -0.78, -1.71, +0.84, -0.30 |
| LW | -1.10 | [-2.31, -0.03] | [-2.09, -0.11] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 23.4 | -0.62, -2.01, -1.93, -0.43, -0.51 |
| SW | -0.40 | [-1.43, +0.60] | [-1.22, +0.42] | inconclusive | non-inferior only | non-inferior only | non-inferior only | neither | 0.3 | -0.10, -0.99, +0.20, -1.23, +0.10 |
| LSW | -0.46 | [-1.22, +0.34] | [-0.78, -0.14] | non-inferior only | non-inferior only | non-inferior only | equivalent | neither | 3.1 | -0.45, -0.56, -0.02, -0.69, -0.56 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.35 | [-1.02, +0.36] | non-inferior only | non-inferior only | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | -0.32 | [-1.18, +0.55] | inconclusive | non-inferior only | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.43 | [-1.12, +0.27] | non-inferior only | non-inferior only | False / False | False / True | SW,LSW |

### pub_Hurink_rdata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.45 | [+0.42, +4.78] | [-0.85, +5.75] | inferior | inferior | inconclusive | inconclusive | behind | 1.5 | +3.40, +0.36, +2.11, -0.12, +6.49 |
| L | -1.10 | [-2.49, +0.26] | [-2.84, +0.63] | non-inferior only | non-inferior only | inconclusive | non-inferior only | neither | 8.3 | -2.66, -0.90, -1.74, -1.32, +1.11 |
| S | -0.45 | [-1.57, +0.86] | [-1.96, +1.06] | inconclusive | non-inferior only | inconclusive | inconclusive | neither | 0.0 | -0.67, -1.26, -1.09, -0.93, +1.69 |
| W | +0.81 | [-1.14, +3.14] | [-2.35, +3.96] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 8.6 | +1.70, -0.04, -1.11, -1.33, +4.82 |
| LS | -0.21 | [-1.15, +0.66] | [-1.11, +0.69] | inconclusive | non-inferior only | inconclusive | non-inferior only | neither | 0.9 | -0.02, +0.08, -1.35, +0.59, -0.35 |
| LW | -1.05 | [-2.27, +0.15] | [-2.51, +0.41] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 20.0 | +0.43, -2.86, -1.05, -0.95, -0.83 |
| SW | -0.63 | [-2.57, +1.02] | [-3.19, +1.94] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.6 | +1.09, -3.93, +0.28, -1.32, +0.74 |
| LSW | -0.02 | [-1.21, +1.25] | [-1.25, +1.21] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 1.2 | -1.27, +1.25, -0.06, -0.61, +0.59 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | -0.03 | [-0.85, +0.97] | inconclusive | equivalent | False / False | False / False | N,L,S,W,LS,LW,SW,LSW |
| Train | +0.07 | [-0.85, +1.21] | inconclusive | inconclusive | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.32 | [-1.23, +0.55] | inconclusive | non-inferior only | False / False | False / False | SW,LSW |

### pub_Hurink_vdata (distribution transfer)

| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | direction | ties % | per seed |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +3.09 | [+1.85, +4.32] | [+1.65, +4.52] | inferior | inferior | inferior | inferior | behind | 0.6 | +2.39, +1.58, +3.73, +3.16, +4.56 |
| L | -0.91 | [-1.87, -0.17] | [-1.69, -0.14] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | ahead | 13.8 | -2.00, -0.58, -0.79, -0.79, -0.42 |
| S | +0.06 | [-1.86, +2.06] | [-2.88, +3.01] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.9 | +2.35, -0.83, -2.41, -1.61, +2.81 |
| W | +1.63 | [-0.23, +3.73] | [-1.26, +4.52] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 7.4 | +2.69, +0.13, +1.33, -0.99, +4.99 |
| LS | -0.00 | [-1.25, +1.15] | [-1.50, +1.50] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 1.2 | -0.32, +0.07, -1.85, +0.79, +1.31 |
| LW | -0.79 | [-1.79, +0.13] | [-1.93, +0.35] | non-inferior only | non-inferior only | non-inferior only | non-inferior only | neither | 12.0 | +0.36, -2.21, -0.80, -0.64, -0.65 |
| SW | -1.54 | [-5.87, +1.76] | [-7.64, +4.55] | inconclusive | inconclusive | inconclusive | inconclusive | neither | 0.9 | +0.79, -9.61, +2.46, -2.73, +1.37 |
| LSW | -0.09 | [-0.90, +0.80] | [-0.79, +0.62] | inconclusive | equivalent | inconclusive | equivalent | neither | 1.2 | -0.15, +0.29, +0.30, +0.18, -1.05 |

| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | non-inferior on all (0.5 / 1) | failing at 1% (eq.) |
|---|---|---|---|---|---|---|---|
| All | +0.18 | [-0.76, +1.10] | inconclusive | inconclusive | False / False | False / False | N,L,S,W,LS,LW,SW |
| Train | +0.51 | [-0.29, +1.43] | inconclusive | inconclusive | False / False | False / False | N,L,S,W,LS,LW |
| Held | -0.82 | [-2.87, +0.84] | inconclusive | non-inferior only | False / False | False / False | SW |

## 2. Control on the combinations it never saw

LS and LW are seen by the policy and not by the control; SW and LSW by neither.

### pub_SD2_test_10x5+mix

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.20 [-1.04, +0.68] | non-inferior only | +0.70 [-0.19, +1.55] | inconclusive | inconclusive | -0.70 [-1.58, +0.21] | non-inferior only | non-inferior only |
| LW | -0.09 [-1.04, +0.75] | non-inferior only | +0.61 [-0.43, +1.66] | inconclusive | inconclusive | -0.58 [-1.15, -0.07] | non-inferior only | non-inferior only |
| SW | -0.83 [-2.23, +0.56] | non-inferior only | -0.41 [-2.34, +1.39] | inconclusive | inconclusive | -0.07 [-1.46, +1.41] | inconclusive | inconclusive |
| LSW | -0.18 [-1.11, +0.74] | non-inferior only | +0.25 [-1.08, +1.66] | inconclusive | inconclusive | -0.14 [-1.30, +1.00] | inconclusive | non-inferior only |
| pooled Pair | | | +0.65 [-0.07, +1.41] | inconclusive | inconclusive | -0.64 [-1.24, -0.09] | non-inferior only | non-inferior only |
| pooled Held | | | -0.08 [-1.46, +1.18] | inconclusive | inconclusive | -0.11 [-1.04, +0.87] | inconclusive | non-inferior only |
| pooled Unseen | | | +0.29 [-0.52, +1.10] | inconclusive | inconclusive | -0.37 [-0.93, +0.18] | non-inferior only | equivalent |

Control on its own combinations (control vs specialist): N +0.87 [-0.10, +1.94], L +0.71 [-0.19, +1.72], S +0.14 [-0.97, +1.24], W -0.36 [-2.26, +1.42]

### pub_SD2_test_20x10+mix

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.80 [-1.55, -0.04] | non-inferior only | -0.17 [-1.03, +0.73] | inconclusive | non-inferior only | -0.45 [-1.27, +0.36] | non-inferior only | non-inferior only |
| LW | +0.06 [-0.47, +0.63] | equivalent | +0.22 [-0.39, +0.85] | inconclusive | equivalent | -0.07 [-0.57, +0.42] | non-inferior only | equivalent |
| SW | -3.07 [-4.41, -1.84] | non-inferior only | -2.89 [-4.50, -1.40] | non-inferior only | non-inferior only | +0.06 [-1.17, +1.44] | inconclusive | inconclusive |
| LSW | -0.74 [-1.87, +0.44] | non-inferior only | -0.66 [-1.94, +0.57] | inconclusive | non-inferior only | +0.19 [-0.91, +1.29] | inconclusive | inconclusive |
| pooled Pair | | | +0.02 [-0.56, +0.63] | inconclusive | equivalent | -0.26 [-0.75, +0.23] | non-inferior only | equivalent |
| pooled Held | | | -1.77 [-2.96, -0.67] | non-inferior only | non-inferior only | +0.12 [-0.82, +1.10] | inconclusive | inconclusive |
| pooled Unseen | | | -0.88 [-1.65, -0.13] | non-inferior only | non-inferior only | -0.07 [-0.69, +0.54] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +0.27 [-0.82, +1.34], L +0.11 [-0.57, +0.74], S -1.73 [-3.01, -0.51], W -0.69 [-3.46, +1.98]

### pub_Brandimarte

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -1.84 [-4.54, +0.35] | non-inferior only | -1.46 [-4.61, +1.26] | inconclusive | inconclusive | -0.17 [-2.28, +2.20] | inconclusive | inconclusive |
| LW | -0.70 [-2.54, +1.18] | inconclusive | -0.43 [-2.53, +1.52] | inconclusive | inconclusive | -0.08 [-1.87, +2.21] | inconclusive | inconclusive |
| SW | -1.19 [-3.22, +0.67] | non-inferior only | -1.01 [-3.70, +0.99] | inconclusive | non-inferior only | -0.08 [-2.08, +2.22] | inconclusive | inconclusive |
| LSW | +0.06 [-1.96, +2.07] | inconclusive | +0.55 [-2.38, +4.14] | inconclusive | inconclusive | -0.20 [-3.08, +2.75] | inconclusive | inconclusive |
| pooled Pair | | | -0.94 [-2.96, +0.69] | inconclusive | non-inferior only | -0.13 [-1.70, +1.24] | inconclusive | inconclusive |
| pooled Held | | | -0.23 [-2.62, +2.32] | inconclusive | inconclusive | -0.14 [-2.23, +2.13] | inconclusive | inconclusive |
| pooled Unseen | | | -0.58 [-1.89, +0.70] | inconclusive | non-inferior only | -0.13 [-1.27, +1.18] | inconclusive | inconclusive |

Control on its own combinations (control vs specialist): N +1.08 [-0.53, +2.87], L -3.18 [-7.06, +0.03], S -0.40 [-2.46, +1.34], W +0.78 [-1.14, +3.61]

### pub_Hurink_edata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.23 [-1.45, +0.94] | non-inferior only | -0.65 [-2.15, +0.79] | inconclusive | non-inferior only | +0.58 [-0.49, +1.62] | inconclusive | inconclusive |
| LW | -1.10 [-2.31, -0.03] | non-inferior only | -0.42 [-1.84, +1.09] | inconclusive | inconclusive | -0.52 [-1.60, +0.62] | inconclusive | non-inferior only |
| SW | -0.40 [-1.43, +0.60] | non-inferior only | -0.70 [-1.96, +0.52] | inconclusive | non-inferior only | +0.42 [-0.85, +1.85] | inconclusive | inconclusive |
| LSW | -0.46 [-1.22, +0.34] | non-inferior only | -0.35 [-1.25, +0.65] | inconclusive | non-inferior only | +0.05 [-0.93, +1.00] | inconclusive | equivalent |
| pooled Pair | | | -0.54 [-1.68, +0.69] | inconclusive | non-inferior only | +0.03 [-0.80, +0.91] | inconclusive | equivalent |
| pooled Held | | | -0.53 [-1.39, +0.29] | non-inferior only | non-inferior only | +0.23 [-0.62, +1.17] | inconclusive | inconclusive |
| pooled Unseen | | | -0.53 [-1.35, +0.33] | non-inferior only | non-inferior only | +0.13 [-0.61, +0.96] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +0.17 [-1.48, +1.66], L -1.10 [-2.06, -0.22], S -0.03 [-1.20, +1.26], W -0.51 [-1.72, +0.71]

### pub_Hurink_rdata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.21 [-1.15, +0.66] | non-inferior only | +0.30 [-1.08, +1.63] | inconclusive | inconclusive | -0.33 [-1.33, +0.67] | inconclusive | non-inferior only |
| LW | -1.05 [-2.27, +0.15] | non-inferior only | -0.98 [-2.03, +0.15] | non-inferior only | non-inferior only | +0.03 [-0.87, +0.92] | inconclusive | equivalent |
| SW | -0.63 [-2.57, +1.02] | inconclusive | -0.43 [-1.61, +0.63] | inconclusive | non-inferior only | -0.06 [-1.55, +1.39] | inconclusive | inconclusive |
| LSW | -0.02 [-1.21, +1.25] | inconclusive | -0.14 [-1.22, +0.97] | inconclusive | non-inferior only | +0.33 [-0.97, +1.71] | inconclusive | inconclusive |
| pooled Pair | | | -0.34 [-1.35, +0.58] | inconclusive | non-inferior only | -0.15 [-0.95, +0.64] | inconclusive | equivalent |
| pooled Held | | | -0.28 [-1.06, +0.54] | inconclusive | non-inferior only | +0.13 [-0.85, +1.06] | inconclusive | inconclusive |
| pooled Unseen | | | -0.31 [-0.95, +0.35] | non-inferior only | equivalent | -0.01 [-0.70, +0.65] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +2.35 [+0.90, +4.00], L -1.03 [-2.31, +0.19], S +0.48 [-0.93, +1.73], W +0.56 [-1.32, +2.45]

### pub_Hurink_vdata

| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | policy vs control | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| LS | -0.00 [-1.25, +1.15] | inconclusive | -0.16 [-1.20, +0.72] | inconclusive | non-inferior only | +0.22 [-0.60, +1.16] | inconclusive | inconclusive |
| LW | -0.79 [-1.79, +0.13] | non-inferior only | -0.65 [-1.83, +0.50] | inconclusive | non-inferior only | -0.05 [-0.76, +0.58] | inconclusive | equivalent |
| SW | -1.54 [-5.87, +1.76] | inconclusive | -2.07 [-5.98, +0.60] | inconclusive | non-inferior only | +0.64 [-0.47, +1.78] | inconclusive | inconclusive |
| LSW | -0.09 [-0.90, +0.80] | equivalent | +0.22 [-0.86, +1.37] | inconclusive | inconclusive | -0.17 [-0.99, +0.64] | inconclusive | equivalent |
| pooled Pair | | | -0.40 [-1.05, +0.31] | non-inferior only | non-inferior only | +0.08 [-0.50, +0.69] | inconclusive | equivalent |
| pooled Held | | | -0.92 [-2.76, +0.60] | inconclusive | non-inferior only | +0.24 [-0.39, +0.91] | inconclusive | equivalent |
| pooled Unseen | | | -0.66 [-1.73, +0.28] | non-inferior only | non-inferior only | +0.16 [-0.30, +0.65] | inconclusive | equivalent |

Control on its own combinations (control vs specialist): N +1.72 [-0.67, +3.92], L -0.70 [-1.66, +0.07], S -0.28 [-2.24, +1.28], W +0.57 [-1.15, +2.82]

## 3. Policy vs dispatching rules

Best rule = per-instance minimum of FIFO, MOR, SPT and MWKR under the same mask. Shares are percent of base instances, averaged over the five seeds (seed mean) or for the selected seed (s303).

### pub_SD2_test_10x5+mix

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -13.72 | [-15.16, -12.25] | 94 / 0 / 6 | 1.3e-06 | 0.0004 | -14.32 | [-15.70, -12.92] | 97 / 1 / 2 | FIFO 30, MOR 24, SPT 8, MWKR 41 |
| L | -2.15 | [-3.16, -1.15] | 54 / 18 / 28 | 7.6e-05 | 0.0004 | -1.90 | [-2.92, -0.90] | 53 / 22 / 25 | FIFO 26, MOR 42, SPT 29, MWKR 26 |
| S | -6.23 | [-7.52, -4.89] | 84 / 0 / 15 | 7.6e-05 | 0.0004 | -6.24 | [-7.80, -4.62] | 82 / 1 / 17 | FIFO 24, MOR 35, SPT 4, MWKR 39 |
| W | -8.34 | [-10.03, -6.65] | 76 / 12 / 12 | 5.3e-05 | 0.0004 | -8.89 | [-10.77, -7.06] | 78 / 11 / 11 | FIFO 24, MOR 33, SPT 9, MWKR 49 |
| LS | -3.62 | [-4.86, -2.28] | 72 / 0 / 28 | 5.3e-05 | 0.0004 | -3.94 | [-5.27, -2.55] | 74 / 1 / 25 | FIFO 23, MOR 29, SPT 16, MWKR 33 |
| LW | -3.27 | [-4.51, -2.03] | 55 / 22 / 23 | 5.3e-05 | 0.0004 | -3.39 | [-4.74, -2.08] | 54 / 24 / 22 | FIFO 29, MOR 39, SPT 27, MWKR 36 |
| SW | -4.93 | [-6.27, -3.62] | 75 / 2 / 24 | 7.6e-05 | 0.0004 | -5.09 | [-6.37, -3.79] | 78 / 2 / 20 | FIFO 30, MOR 37, SPT 0, MWKR 39 |
| LSW | -4.38 | [-5.91, -2.78] | 70 / 1 / 29 | 4.9e-06 | 0.0004 | -4.67 | [-6.41, -2.81] | 69 / 1 / 30 | FIFO 15, MOR 33, SPT 10, MWKR 43 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -19.79 (98) | -19.63 (99) | -29.21 (100) | -18.08 (96) |
| L | -6.81 (87) | -5.65 (75) | -9.63 (85) | -7.35 (78) |
| S | -11.63 (93) | -11.64 (91) | -24.00 (100) | -10.38 (93) |
| W | -15.24 (90) | -14.76 (88) | -27.38 (98) | -12.00 (86) |
| LS | -8.47 (86) | -8.67 (86) | -15.17 (90) | -9.10 (87) |
| LW | -9.71 (85) | -7.46 (77) | -12.62 (84) | -7.86 (77) |
| SW | -9.76 (90) | -9.27 (88) | -22.13 (99) | -8.49 (85) |
| LSW | -12.11 (93) | -9.70 (85) | -18.16 (96) | -8.88 (78) |

### pub_SD2_test_20x10+mix

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -30.53 | [-31.47, -29.60] | 100 / 0 / 0 | 1.3e-07 | 0.0004 | -30.05 | [-30.85, -29.23] | 100 / 0 / 0 | FIFO 24, MOR 20, SPT 29, MWKR 29 |
| L | -2.32 | [-2.98, -1.67] | 71 / 2 / 28 | 3.8e-05 | 0.0004 | -2.28 | [-2.92, -1.64] | 70 / 2 / 28 | FIFO 22, MOR 25, SPT 32, MWKR 22 |
| S | -15.78 | [-16.68, -14.88] | 99 / 0 / 1 | 1.4e-06 | 0.0004 | -16.00 | [-16.91, -15.07] | 99 / 0 / 1 | FIFO 36, MOR 26, SPT 1, MWKR 38 |
| W | -26.14 | [-27.56, -24.66] | 100 / 0 / 0 | 1.1e-06 | 0.0004 | -25.19 | [-26.60, -23.75] | 100 / 0 / 0 | FIFO 29, MOR 28, SPT 11, MWKR 51 |
| LS | -4.06 | [-5.02, -3.08] | 78 / 1 / 21 | 1.1e-05 | 0.0004 | -4.56 | [-5.57, -3.56] | 85 / 0 / 15 | FIFO 24, MOR 36, SPT 10, MWKR 30 |
| LW | -3.40 | [-4.26, -2.57] | 69 / 8 / 23 | 1.6e-05 | 0.0004 | -3.18 | [-4.11, -2.24] | 68 / 6 / 26 | FIFO 23, MOR 43, SPT 26, MWKR 25 |
| SW | -12.20 | [-13.17, -11.17] | 97 / 1 / 2 | 3.7e-06 | 0.0004 | -12.71 | [-13.63, -11.77] | 99 / 0 / 1 | FIFO 31, MOR 26, SPT 0, MWKR 44 |
| LSW | -4.66 | [-5.69, -3.61] | 80 / 1 / 20 | 2.8e-05 | 0.0004 | -5.28 | [-6.55, -3.97] | 81 / 0 / 19 | FIFO 20, MOR 36, SPT 2, MWKR 44 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -34.71 (100) | -35.04 (100) | -35.65 (100) | -34.18 (100) |
| L | -6.11 (93) | -5.70 (91) | -8.79 (84) | -6.07 (89) |
| S | -18.96 (100) | -19.25 (100) | -28.85 (100) | -18.96 (100) |
| W | -30.25 (100) | -29.77 (100) | -36.63 (100) | -27.90 (100) |
| LS | -8.38 (92) | -6.99 (87) | -16.75 (94) | -7.75 (92) |
| LW | -7.27 (92) | -6.03 (86) | -11.18 (86) | -6.87 (86) |
| SW | -15.17 (99) | -15.37 (99) | -26.35 (100) | -15.07 (99) |
| LSW | -9.47 (92) | -8.35 (90) | -20.43 (98) | -7.76 (84) |

### pub_Brandimarte

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -4.07 | [-8.48, -0.69] | 70 / 14 / 16 | 0.00023 | 0.036 | -4.09 | [-8.26, -1.04] | 70 / 20 / 10 | FIFO 10, MOR 30, SPT 0, MWKR 60 |
| L | -4.29 | [-8.21, -0.98] | 70 / 4 / 26 | 0.0004 | 0.027 | -3.22 | [-6.92, +0.13] | 70 / 0 / 30 | FIFO 10, MOR 30, SPT 10, MWKR 50 |
| S | -0.64 | [-3.37, +1.97] | 50 / 2 / 48 | 0.17 | 0.38 | -1.56 | [-4.78, +1.47] | 50 / 0 / 50 | FIFO 40, MOR 0, SPT 0, MWKR 70 |
| W | -2.71 | [-5.65, -0.16] | 54 / 30 / 16 | 0.0024 | 0.089 | -2.91 | [-6.20, -0.40] | 40 / 50 / 10 | FIFO 40, MOR 30, SPT 0, MWKR 50 |
| LS | -3.48 | [-7.12, +0.03] | 64 / 4 / 32 | 0.00033 | 0.1 | -4.34 | [-8.44, -0.63] | 60 / 10 / 30 | FIFO 20, MOR 40, SPT 10, MWKR 50 |
| LW | -4.79 | [-8.48, -1.01] | 78 / 8 / 14 | 0.00035 | 0.046 | -3.80 | [-8.36, +1.36] | 80 / 0 / 20 | FIFO 20, MOR 20, SPT 10, MWKR 70 |
| SW | -1.87 | [-4.10, +0.34] | 60 / 2 / 38 | 0.038 | 0.14 | -2.61 | [-5.87, +0.20] | 50 / 10 / 40 | FIFO 10, MOR 20, SPT 0, MWKR 70 |
| LSW | -1.84 | [-6.05, +2.30] | 58 / 0 / 42 | 0.038 | 0.38 | -2.05 | [-7.24, +3.32] | 60 / 0 / 40 | FIFO 30, MOR 20, SPT 0, MWKR 50 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -8.05 (96) | -6.67 (88) | -35.72 (100) | -6.38 (82) |
| L | -9.30 (96) | -7.94 (90) | -26.36 (90) | -6.90 (84) |
| S | -4.16 (58) | -6.46 (86) | -30.76 (100) | -2.06 (70) |
| W | -5.43 (90) | -5.89 (74) | -30.65 (100) | -4.29 (64) |
| LS | -8.89 (98) | -6.56 (78) | -25.87 (96) | -5.22 (76) |
| LW | -10.11 (96) | -8.04 (82) | -27.23 (98) | -6.48 (84) |
| SW | -7.77 (92) | -6.75 (90) | -27.04 (100) | -4.04 (72) |
| LSW | -8.87 (86) | -6.77 (72) | -25.82 (98) | -3.55 (76) |

### pub_Hurink_edata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | -0.63 | [-2.20, +1.03] | 58 / 1 / 41 | 1 | 1 | -1.09 | [-2.30, +0.19] | 62 / 0 / 38 | FIFO 35, MOR 26, SPT 5, MWKR 34 |
| L | -1.73 | [-3.24, -0.23] | 62 / 5 / 33 | 5.5e-05 | 0.086 | -1.92 | [-3.53, -0.29] | 62 / 6 / 32 | FIFO 25, MOR 42, SPT 15, MWKR 32 |
| S | +1.68 | [+0.60, +2.77] | 44 / 0 / 56 | 1 | 1 | +1.69 | [+0.41, +3.08] | 45 / 0 / 55 | FIFO 26, MOR 32, SPT 2, MWKR 40 |
| W | +0.65 | [-0.80, +2.08] | 43 / 5 / 52 | 1 | 1 | -0.61 | [-2.04, +0.83] | 48 / 8 / 45 | FIFO 29, MOR 25, SPT 2, MWKR 54 |
| LS | +0.17 | [-1.17, +1.50] | 50 / 0 / 50 | 1 | 1 | +0.32 | [-1.18, +1.80] | 46 / 0 / 54 | FIFO 20, MOR 38, SPT 5, MWKR 38 |
| LW | -1.78 | [-3.21, -0.38] | 52 / 19 / 29 | 0.0026 | 0.051 | -2.31 | [-4.02, -0.62] | 55 / 20 / 25 | FIFO 25, MOR 46, SPT 11, MWKR 43 |
| SW | +2.31 | [+1.22, +3.56] | 30 / 1 / 69 | 1 | 1 | +1.87 | [+0.71, +3.08] | 32 / 0 / 68 | FIFO 15, MOR 32, SPT 3, MWKR 51 |
| LSW | -0.36 | [-1.57, +0.83] | 50 / 2 / 48 | 0.17 | 1 | -0.26 | [-1.66, +1.14] | 49 / 3 / 48 | FIFO 20, MOR 35, SPT 14, MWKR 31 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -4.43 (75) | -5.18 (82) | -19.75 (96) | -3.59 (76) |
| L | -6.65 (80) | -5.67 (75) | -15.82 (94) | -6.93 (78) |
| S | -3.53 (72) | -2.56 (69) | -14.77 (94) | -1.69 (66) |
| W | -4.54 (69) | -4.50 (69) | -16.83 (95) | -1.81 (61) |
| LS | -5.60 (77) | -3.33 (68) | -14.13 (94) | -4.27 (72) |
| LW | -7.62 (82) | -5.54 (68) | -14.94 (89) | -5.36 (70) |
| SW | -3.32 (70) | -1.65 (61) | -13.84 (93) | -0.05 (50) |
| LSW | -6.67 (82) | -4.40 (76) | -13.58 (93) | -4.04 (75) |

### pub_Hurink_rdata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | +1.28 | [-0.25, +2.95] | 40 / 1 / 59 | 1 | 1 | +1.24 | [+0.01, +2.50] | 40 / 0 / 60 | FIFO 26, MOR 32, SPT 0, MWKR 43 |
| L | -1.15 | [-2.40, +0.09] | 58 / 7 / 35 | 0.024 | 0.24 | -1.40 | [-2.79, -0.04] | 62 / 8 / 31 | FIFO 20, MOR 37, SPT 5, MWKR 42 |
| S | +0.75 | [-0.46, +2.04] | 48 / 0 / 51 | 1 | 1 | +0.98 | [-0.34, +2.37] | 51 / 2 / 48 | FIFO 31, MOR 34, SPT 3, MWKR 34 |
| W | +2.90 | [+0.96, +5.11] | 33 / 8 / 59 | 1 | 1 | +1.81 | [+0.46, +3.24] | 43 / 5 / 52 | FIFO 28, MOR 22, SPT 0, MWKR 54 |
| LS | -1.68 | [-2.84, -0.53] | 64 / 0 / 36 | 0.011 | 0.019 | -2.06 | [-3.39, -0.72] | 71 / 0 / 29 | FIFO 11, MOR 32, SPT 6, MWKR 51 |
| LW | -0.55 | [-1.96, +0.84] | 51 / 11 / 37 | 0.37 | 1 | +0.07 | [-1.30, +1.48] | 49 / 11 / 40 | FIFO 26, MOR 35, SPT 6, MWKR 43 |
| SW | +1.25 | [-0.03, +2.59] | 43 / 1 / 56 | 1 | 1 | +1.46 | [+0.01, +3.06] | 49 / 0 / 51 | FIFO 23, MOR 28, SPT 2, MWKR 51 |
| LSW | -1.18 | [-2.54, +0.29] | 62 / 2 / 36 | 0.086 | 0.33 | -1.45 | [-2.84, -0.04] | 58 / 2 / 40 | FIFO 20, MOR 45, SPT 2, MWKR 34 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -2.60 (69) | -3.26 (73) | -25.70 (100) | -1.12 (57) |
| L | -5.67 (83) | -4.28 (78) | -14.04 (93) | -4.30 (75) |
| S | -4.29 (77) | -3.29 (74) | -19.33 (99) | -2.22 (66) |
| W | -1.01 (59) | -1.86 (66) | -23.16 (99) | +0.23 (46) |
| LS | -7.63 (87) | -5.00 (78) | -15.90 (93) | -4.65 (77) |
| LW | -6.12 (77) | -4.83 (67) | -17.02 (94) | -3.69 (69) |
| SW | -3.85 (76) | -2.72 (71) | -18.67 (99) | -0.30 (57) |
| LSW | -7.15 (86) | -3.92 (72) | -18.16 (97) | -5.10 (77) |

### pub_Hurink_vdata

| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | rule that is best (% of instances) |
|---|---|---|---|---|---|---|---|---|---|
| N | +5.70 | [+3.55, +8.05] | 20 / 0 / 80 | 1 | 1 | +5.75 | [+4.15, +7.35] | 22 / 0 / 78 | FIFO 8, MOR 9, SPT 0, MWKR 85 |
| L | -1.80 | [-3.08, -0.63] | 47 / 15 / 38 | 0.0005 | 0.0077 | -1.32 | [-2.71, -0.06] | 37 / 8 / 55 | FIFO 20, MOR 46, SPT 3, MWKR 45 |
| S | +2.63 | [+1.18, +4.21] | 26 / 1 / 73 | 1 | 1 | +2.58 | [+1.52, +3.65] | 22 / 0 / 78 | FIFO 18, MOR 22, SPT 0, MWKR 60 |
| W | +5.45 | [+3.53, +7.62] | 15 / 5 / 79 | 1 | 1 | +4.83 | [+3.26, +6.51] | 17 / 6 / 77 | FIFO 12, MOR 14, SPT 0, MWKR 82 |
| LS | -1.56 | [-2.62, -0.51] | 60 / 1 / 39 | 0.0065 | 0.012 | -1.34 | [-2.53, -0.20] | 58 / 0 / 42 | FIFO 8, MOR 49, SPT 3, MWKR 42 |
| LW | -2.37 | [-3.89, -0.95] | 56 / 6 / 39 | 0.0011 | 0.0024 | -2.27 | [-3.85, -0.82] | 54 / 9 / 37 | FIFO 23, MOR 45, SPT 2, MWKR 37 |
| SW | +3.77 | [+2.30, +5.39] | 15 / 1 / 84 | 1 | 1 | +3.38 | [+2.24, +4.63] | 18 / 2 / 80 | FIFO 3, MOR 34, SPT 0, MWKR 65 |
| LSW | -1.23 | [-2.42, -0.06] | 58 / 0 / 42 | 0.002 | 0.099 | -1.25 | [-2.65, +0.12] | 58 / 0 / 42 | FIFO 15, MOR 43, SPT 3, MWKR 42 |

| comb. | vs FIFO (shorter %) | vs MOR (shorter %) | vs SPT (shorter %) | vs MWKR (shorter %) |
|---|---|---|---|---|
| N | -1.69 (66) | -0.01 (54) | -25.08 (100) | +5.36 (22) |
| L | -5.45 (82) | -3.59 (69) | -13.61 (95) | -5.18 (67) |
| S | -3.43 (70) | -1.59 (61) | -23.58 (100) | +1.26 (38) |
| W | -1.19 (60) | +0.29 (50) | -25.58 (100) | +4.42 (19) |
| LS | -7.26 (90) | -3.54 (72) | -15.46 (93) | -5.15 (75) |
| LW | -6.30 (82) | -4.31 (72) | -16.04 (93) | -5.90 (74) |
| SW | -2.72 (68) | +0.59 (43) | -20.87 (100) | +1.75 (27) |
| LSW | -6.98 (86) | -3.83 (73) | -16.50 (96) | -4.83 (74) |

## 4. Regime N: gap to the best-known value (%)

Best-known values: Behnke and Geiger (2012) for Brandimarte and Hurink (the smaller of the literature value and the table's CP Optimizer value); per-instance OR-Tools results with an 1800 s limit from the SD2 release for SD2. On SD2 20x10 only one reference is proved optimal, so the gap there is a gap to a strong solver.

| set | proved optimal | policy (seed mean, sd) | selected | control | N specialist | FIFO | MOR | SPT | MWKR | best rule | policy on proved | policy to lower bound |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix | 96 of 100 | 22.01 (0.64) | 21.18 | 21.76 | 20.97 (1.05) | 53.57 | 52.95 | 74.82 | 50.30 | 42.16 | 22.01 | - |
| pub_SD2_test_20x10+mix | 1 of 100 | 24.83 (1.07) | 25.79 | 24.81 | 24.64 (1.45) | 91.93 | 93.00 | 95.80 | 90.26 | 80.19 | 22.15 | - |
| pub_Brandimarte | 3 of 10 | 11.99 (0.70) | 12.02 | 12.62 | 11.38 (0.97) | 22.74 | 20.64 | 78.68 | 20.66 | 17.38 | 8.44 | 27.16 |
| pub_Hurink_edata | 26 of 65 | 18.47 (1.76) | 17.94 | 18.20 | 18.13 (1.91) | 24.36 | 25.36 | 49.07 | 23.12 | 19.38 | 16.20 | 35.57 |
| pub_Hurink_rdata | 17 of 65 | 18.72 (2.05) | 18.66 | 18.59 | 15.96 (1.07) | 22.19 | 23.21 | 61.02 | 20.27 | 17.33 | 12.95 | 23.47 |
| pub_Hurink_vdata | 45 of 65 | 12.47 (2.39) | 12.53 | 10.91 | 9.15 (1.87) | 14.62 | 12.66 | 51.56 | 6.81 | 6.45 | 12.04 | 12.60 |

## 5. Selected seed

Rule: best validation score of the kept checkpoint (arm_registry.select_seed), validation on the first 20 records of pub_SD2_vali_10x5+mix. Selected: s303. Scores: s301 0.78852 (update index 41), s302 0.78971 (update index 53), s303 0.78770 (update index 59), s304 0.79108 (update index 65), s305 0.78908 (update index 70).

## 6. Per-seed gap to the per-instance best rule (%)

### pub_SD2_test_10x5+mix

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -13.22 | -13.29 | -14.32 | -13.97 | -13.80 | -13.72 | 0.46 |
| N | control | -15.22 | -13.08 | -13.50 | -13.77 | -14.05 | -13.92 | 0.81 |
| N | specialist | -15.39 | -14.75 | -13.66 | -13.94 | -14.81 | -14.51 | 0.70 |
| L | policy | -2.01 | -2.04 | -1.90 | -2.25 | -2.56 | -2.15 | 0.26 |
| L | control | -1.24 | -2.47 | -1.33 | -3.11 | -1.95 | -2.02 | 0.79 |
| L | specialist | -2.90 | -2.56 | -2.68 | -2.55 | -2.37 | -2.61 | 0.20 |
| S | policy | -5.83 | -7.30 | -6.24 | -6.65 | -5.11 | -6.23 | 0.82 |
| S | control | -5.95 | -6.57 | -6.38 | -6.68 | -7.38 | -6.59 | 0.52 |
| S | specialist | -5.88 | -7.05 | -6.90 | -6.23 | -6.27 | -6.47 | 0.49 |
| W | policy | -7.70 | -8.40 | -8.89 | -9.33 | -7.39 | -8.34 | 0.81 |
| W | control | -10.38 | -8.15 | -9.33 | -7.60 | -9.21 | -8.94 | 1.08 |
| W | specialist | -7.21 | -8.55 | -8.31 | -8.93 | -8.20 | -8.24 | 0.64 |
| LS | policy | -3.31 | -3.60 | -3.94 | -3.27 | -3.99 | -3.62 | 0.34 |
| LS | control | -2.44 | -3.31 | -2.27 | -3.32 | -2.54 | -2.77 | 0.50 |
| LS | specialist | -2.90 | -3.86 | -2.86 | -3.22 | -3.28 | -3.22 | 0.40 |
| LW | policy | -2.84 | -3.65 | -3.39 | -3.33 | -3.12 | -3.27 | 0.31 |
| LW | control | -1.90 | -3.12 | -2.34 | -2.86 | -2.82 | -2.61 | 0.49 |
| LW | specialist | -3.50 | -2.48 | -2.74 | -3.27 | -3.13 | -3.03 | 0.41 |
| SW | policy | -3.88 | -5.54 | -5.09 | -5.38 | -4.78 | -4.93 | 0.65 |
| SW | control | -5.07 | -4.31 | -4.11 | -3.67 | -5.74 | -4.58 | 0.82 |
| SW | specialist | -4.29 | -3.37 | -4.92 | -4.77 | -1.92 | -3.85 | 1.24 |
| LSW | policy | -4.12 | -4.35 | -4.67 | -4.50 | -4.25 | -4.38 | 0.22 |
| LSW | control | -2.81 | -3.80 | -3.44 | -4.49 | -5.40 | -3.99 | 1.00 |
| LSW | specialist | -4.31 | -2.97 | -4.63 | -3.93 | -4.04 | -3.98 | 0.62 |

### pub_SD2_test_20x10+mix

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -29.99 | -31.40 | -30.05 | -30.81 | -30.41 | -30.53 | 0.59 |
| N | control | -32.31 | -29.64 | -29.87 | -30.95 | -30.02 | -30.56 | 1.10 |
| N | specialist | -31.95 | -30.61 | -29.97 | -29.98 | -30.65 | -30.63 | 0.81 |
| L | policy | -1.80 | -2.46 | -2.28 | -2.55 | -2.53 | -2.32 | 0.31 |
| L | control | -1.97 | -2.31 | -1.67 | -2.51 | -2.08 | -2.11 | 0.32 |
| L | specialist | -2.58 | -2.25 | -2.29 | -1.57 | -2.10 | -2.16 | 0.38 |
| S | policy | -15.11 | -16.27 | -16.00 | -15.19 | -16.35 | -15.78 | 0.59 |
| S | control | -15.99 | -15.59 | -15.38 | -16.74 | -16.92 | -16.13 | 0.68 |
| S | specialist | -15.38 | -13.32 | -13.93 | -16.10 | -13.61 | -14.47 | 1.21 |
| W | policy | -25.19 | -27.08 | -25.19 | -26.65 | -26.59 | -26.14 | 0.89 |
| W | control | -27.70 | -25.03 | -24.25 | -26.01 | -25.68 | -25.73 | 1.29 |
| W | specialist | -23.39 | -24.89 | -26.61 | -25.11 | -24.98 | -25.00 | 1.14 |
| LS | policy | -3.93 | -3.88 | -4.56 | -3.95 | -3.99 | -4.06 | 0.28 |
| LS | control | -3.40 | -3.28 | -3.11 | -3.33 | -4.11 | -3.44 | 0.39 |
| LS | specialist | -3.03 | -3.05 | -3.26 | -3.24 | -2.81 | -3.08 | 0.18 |
| LW | policy | -3.12 | -3.70 | -3.18 | -3.69 | -3.30 | -3.40 | 0.28 |
| LW | control | -3.24 | -3.22 | -2.83 | -3.68 | -3.30 | -3.25 | 0.30 |
| LW | specialist | -3.09 | -3.45 | -3.41 | -3.64 | -3.25 | -3.37 | 0.21 |
| SW | policy | -12.37 | -12.14 | -12.71 | -11.18 | -12.59 | -12.20 | 0.61 |
| SW | control | -11.67 | -11.30 | -11.41 | -12.87 | -13.01 | -12.05 | 0.82 |
| SW | specialist | -8.84 | -9.61 | -10.25 | -9.59 | -7.71 | -9.20 | 0.97 |
| LSW | policy | -3.96 | -4.63 | -5.28 | -4.52 | -4.89 | -4.66 | 0.49 |
| LSW | control | -4.76 | -4.70 | -3.82 | -4.83 | -4.92 | -4.60 | 0.45 |
| LSW | specialist | -2.34 | -4.08 | -3.84 | -5.20 | -3.12 | -3.72 | 1.07 |

### pub_Brandimarte

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | -3.41 | -4.02 | -4.09 | -4.84 | -3.97 | -4.07 | 0.51 |
| N | control | -3.97 | -2.60 | -3.91 | -3.69 | -3.75 | -3.58 | 0.56 |
| N | specialist | -5.33 | -4.98 | -3.92 | -3.56 | -5.04 | -4.57 | 0.77 |
| L | policy | -4.08 | -4.43 | -3.22 | -5.05 | -4.67 | -4.29 | 0.69 |
| L | control | -4.65 | -4.71 | -2.72 | -4.83 | -4.46 | -4.27 | 0.88 |
| L | specialist | +0.33 | -0.05 | +2.42 | -1.52 | -3.85 | -0.53 | 2.33 |
| S | policy | +1.70 | -1.03 | -1.56 | -1.46 | -0.86 | -0.64 | 1.34 |
| S | control | -0.22 | -0.20 | -1.11 | -1.25 | -0.88 | -0.73 | 0.50 |
| S | specialist | -1.29 | -0.12 | -1.02 | +0.61 | +0.84 | -0.20 | 0.95 |
| W | policy | -2.82 | -3.46 | -2.91 | -2.92 | -1.47 | -2.71 | 0.74 |
| W | control | -2.31 | -1.62 | -2.77 | -1.93 | -2.67 | -2.26 | 0.49 |
| W | specialist | -2.41 | -2.92 | -3.71 | -2.10 | -3.43 | -2.91 | 0.68 |
| LS | policy | -3.36 | -3.38 | -4.34 | -3.14 | -3.17 | -3.48 | 0.49 |
| LS | control | -3.66 | -3.58 | -1.06 | -3.76 | -4.03 | -3.22 | 1.22 |
| LS | specialist | -3.36 | +0.07 | +0.64 | -1.81 | -2.40 | -1.37 | 1.68 |
| LW | policy | -5.43 | -4.26 | -3.80 | -5.24 | -5.19 | -4.79 | 0.71 |
| LW | control | -4.42 | -5.55 | -4.18 | -4.49 | -4.19 | -4.56 | 0.57 |
| LW | specialist | -4.84 | -5.35 | -1.59 | -3.69 | -4.33 | -3.96 | 1.46 |
| SW | policy | -1.84 | -3.38 | -2.61 | -0.27 | -1.26 | -1.87 | 1.20 |
| SW | control | -0.90 | -0.77 | -2.24 | -1.89 | -2.76 | -1.71 | 0.86 |
| SW | specialist | -0.59 | -2.29 | -2.52 | +0.95 | +1.54 | -0.58 | 1.84 |
| LSW | policy | -1.33 | -2.19 | -2.05 | -3.50 | -0.15 | -1.84 | 1.23 |
| LSW | control | -0.05 | -0.16 | +0.70 | -3.80 | -3.24 | -1.31 | 2.06 |
| LSW | specialist | -0.27 | -4.35 | -1.59 | -3.20 | +0.56 | -1.77 | 2.02 |

### pub_Hurink_edata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +1.51 | -1.08 | -1.09 | -2.50 | +0.03 | -0.63 | 1.49 |
| N | control | -1.31 | +1.38 | -0.89 | -1.83 | -1.63 | -0.85 | 1.30 |
| N | specialist | -2.13 | -0.27 | +0.62 | +0.13 | -3.02 | -0.93 | 1.56 |
| L | policy | -1.83 | -1.66 | -1.92 | -1.54 | -1.70 | -1.73 | 0.15 |
| L | control | -1.38 | -1.34 | -1.00 | -0.96 | -1.80 | -1.29 | 0.34 |
| L | specialist | +0.39 | +0.13 | +0.76 | -0.30 | -1.02 | -0.01 | 0.69 |
| S | policy | +1.90 | +1.76 | +1.69 | +0.79 | +2.28 | +1.68 | 0.55 |
| S | control | +1.47 | +2.72 | +1.08 | +0.25 | +0.67 | +1.24 | 0.95 |
| S | specialist | +0.81 | +1.12 | +1.55 | +1.64 | +1.63 | +1.35 | 0.37 |
| W | policy | +1.73 | +0.82 | -0.61 | -0.53 | +1.83 | +0.65 | 1.18 |
| W | control | -0.22 | +0.90 | +0.92 | -1.08 | -0.50 | +0.00 | 0.88 |
| W | specialist | +1.58 | +1.11 | +0.89 | +0.63 | -0.44 | +0.75 | 0.75 |
| LS | policy | +0.10 | -0.65 | +0.32 | +0.48 | +0.60 | +0.17 | 0.49 |
| LS | control | +0.46 | -0.07 | -0.82 | -0.40 | -0.53 | -0.27 | 0.49 |
| LS | specialist | -0.57 | +0.34 | +2.25 | -0.33 | +1.04 | +0.55 | 1.14 |
| LW | policy | -1.86 | -1.58 | -2.31 | -1.95 | -1.17 | -1.78 | 0.43 |
| LW | control | -0.23 | -1.24 | -0.97 | -0.92 | -2.37 | -1.14 | 0.78 |
| LW | specialist | -1.15 | +0.66 | -0.16 | -1.35 | -0.55 | -0.51 | 0.81 |
| SW | policy | +2.24 | +2.16 | +1.87 | +1.56 | +3.73 | +2.31 | 0.84 |
| SW | control | +2.16 | +3.43 | +1.90 | +1.69 | +0.97 | +2.03 | 0.90 |
| SW | specialist | +2.55 | +3.46 | +1.80 | +3.02 | +3.78 | +2.92 | 0.78 |
| LSW | policy | -0.71 | -0.23 | -0.26 | -0.62 | +0.03 | -0.36 | 0.30 |
| LSW | control | +0.36 | -0.04 | -1.20 | -0.05 | -0.36 | -0.26 | 0.58 |
| LSW | specialist | -0.10 | +0.46 | -0.13 | +0.26 | +0.77 | +0.25 | 0.38 |

### pub_Hurink_rdata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +1.55 | -0.02 | +1.24 | -0.36 | +4.00 | +1.28 | 1.72 |
| N | control | -1.16 | +2.84 | +0.84 | +0.96 | +2.44 | +1.18 | 1.58 |
| N | specialist | -1.72 | -0.31 | -0.76 | -0.13 | -2.19 | -1.02 | 0.90 |
| L | policy | -1.05 | -1.20 | -1.40 | -1.72 | -0.37 | -1.15 | 0.50 |
| L | control | -0.73 | -1.55 | -0.96 | -1.27 | -0.87 | -1.07 | 0.33 |
| L | specialist | +1.86 | -0.09 | +0.58 | -0.27 | -1.39 | +0.14 | 1.20 |
| S | policy | +0.03 | -0.07 | +0.98 | +0.23 | +2.55 | +0.75 | 1.09 |
| S | control | +1.80 | +3.13 | +0.37 | +1.44 | +1.60 | +1.67 | 0.99 |
| S | specialist | +0.89 | +1.31 | +2.18 | +1.32 | +0.96 | +1.33 | 0.52 |
| W | policy | +4.05 | +1.74 | +1.81 | +0.71 | +6.21 | +2.90 | 2.21 |
| W | control | +1.18 | +4.60 | +1.08 | +2.43 | +4.18 | +2.69 | 1.64 |
| W | specialist | +2.49 | +1.94 | +3.13 | +2.22 | +1.50 | +2.26 | 0.61 |
| LS | policy | -1.81 | -2.33 | -2.06 | -1.28 | -0.93 | -1.68 | 0.57 |
| LS | control | -0.82 | -0.66 | -2.52 | -1.27 | -0.78 | -1.21 | 0.77 |
| LS | specialist | -1.74 | -2.26 | -0.57 | -1.77 | -0.34 | -1.34 | 0.84 |
| LW | policy | -1.04 | -0.83 | +0.07 | -1.23 | +0.29 | -0.55 | 0.68 |
| LW | control | -1.02 | +0.14 | -0.72 | -0.78 | -0.04 | -0.48 | 0.50 |
| LW | specialist | -1.37 | +2.16 | +1.35 | -0.15 | +1.28 | +0.65 | 1.40 |
| SW | policy | +1.04 | +0.86 | +1.46 | +0.19 | +2.69 | +1.25 | 0.93 |
| SW | control | +0.46 | +2.98 | +1.03 | +1.90 | +0.95 | +1.46 | 0.99 |
| SW | specialist | +0.07 | +5.23 | +1.31 | +1.69 | +2.09 | +2.08 | 1.92 |
| LSW | policy | -1.71 | -0.20 | -1.45 | -2.03 | -0.51 | -1.18 | 0.79 |
| LSW | control | -1.43 | -2.01 | -2.07 | -0.54 | -0.50 | -1.31 | 0.76 |
| LSW | specialist | -0.26 | -1.17 | -1.18 | -1.25 | -0.95 | -0.96 | 0.41 |

### pub_Hurink_vdata

| comb. | arm | s301 | s302 | s303 | s304 | s305 | mean | sd |
|---|---|---|---|---|---|---|---|---|
| N | policy | +6.74 | +4.12 | +5.75 | +3.09 | +8.82 | +5.70 | 2.24 |
| N | control | +1.62 | +4.91 | +3.03 | +5.25 | +6.46 | +4.25 | 1.92 |
| N | specialist | +4.34 | +2.54 | +1.99 | -0.00 | +4.10 | +2.59 | 1.76 |
| L | policy | -1.98 | -1.84 | -1.32 | -1.89 | -1.95 | -1.80 | 0.27 |
| L | control | -1.41 | -1.71 | -1.49 | -1.98 | -1.53 | -1.63 | 0.23 |
| L | specialist | +0.22 | -1.26 | -0.43 | -1.04 | -1.52 | -0.81 | 0.70 |
| S | policy | +3.51 | +1.03 | +2.58 | +1.13 | +4.87 | +2.63 | 1.63 |
| S | control | +1.77 | +1.87 | +1.12 | +2.72 | +3.90 | +2.28 | 1.07 |
| S | specialist | +1.21 | +1.98 | +5.22 | +2.90 | +2.05 | +2.67 | 1.55 |
| W | policy | +6.19 | +4.81 | +4.83 | +3.03 | +8.40 | +5.45 | 1.99 |
| W | control | +1.73 | +4.09 | +3.50 | +4.35 | +8.11 | +4.35 | 2.33 |
| W | specialist | +3.61 | +4.72 | +3.52 | +4.29 | +3.32 | +3.89 | 0.59 |
| LS | policy | -1.95 | -2.15 | -1.34 | -1.54 | -0.82 | -1.56 | 0.52 |
| LS | control | -1.40 | -1.80 | -1.25 | -1.70 | -2.28 | -1.69 | 0.40 |
| LS | specialist | -1.58 | -2.14 | +0.66 | -2.20 | -2.02 | -1.46 | 1.21 |
| LW | policy | -1.71 | -2.97 | -2.27 | -2.56 | -2.32 | -2.37 | 0.46 |
| LW | control | -1.50 | -3.02 | -1.44 | -3.07 | -2.29 | -2.26 | 0.79 |
| LW | specialist | -2.00 | -0.65 | -1.39 | -1.88 | -1.60 | -1.50 | 0.53 |
| SW | policy | +4.53 | +2.65 | +3.38 | +2.07 | +6.21 | +3.77 | 1.65 |
| SW | control | +3.02 | +2.82 | +2.16 | +3.19 | +4.89 | +3.22 | 1.02 |
| SW | specialist | +3.85 | +14.23 | +0.98 | +5.29 | +4.84 | +5.84 | 4.98 |
| LSW | policy | -1.27 | -1.47 | -1.25 | -0.75 | -1.42 | -1.23 | 0.29 |
| LSW | control | -0.63 | -1.03 | -0.63 | -0.93 | -1.63 | -0.97 | 0.41 |
| LSW | specialist | -1.05 | -1.70 | -1.41 | -0.74 | -0.22 | -1.02 | 0.58 |

## 7. Sensitivity: pub_SD2_test_10x5+mix_lag050 ({'lag_scale': 0.5}) beside pub_SD2_test_10x5+mix

Records: 800 compared with pub_SD2_test_10x5+mix; fields changed: L|horizon (100), L|lag (100), LS|horizon (100), LS|lag (100), LS|setup_convention (100), LSW|horizon (100), LSW|lag (100), LSW|setup_convention (100), LSW|windows (100), LW|horizon (100), LW|lag (100), LW|windows (100), N|horizon (100), S|horizon (100), S|setup_convention (100), SW|horizon (100), SW|setup_convention (100), SW|windows (100), W|horizon (100), W|windows (100). Free check: 38 N and S arrays (3800 makespans) of every run and rule equal the scale-1.0 arrays bit for bit. Combinations with W also move their outage calendars (H shrinks with the lags); only the combinations with L are compared.

| comb. | best-rule R/N (1.0 / 0.5) | policy R/N (1.0 / 0.5) |
|---|---|---|
| L | 1.88 / 1.25 | 2.15 / 1.36 |
| LS | 2.35 / 1.84 | 2.64 / 2.02 |
| LW | 1.96 / 1.39 | 2.22 / 1.52 |
| LSW | 2.54 / 2.06 | 2.83 / 2.27 |

Policy vs specialist:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | +0.58 [-0.05, +1.21] | inconclusive | inconclusive | -1.47 [-3.01, -0.16] | [-3.16, +0.22] | non-inferior only | non-inferior only | -1.33, +0.03, -3.50, -1.97, -0.59 | -2.05 [-3.81, -0.60] |
| LS | -0.20 [-1.04, +0.68] | inconclusive | non-inferior only | -1.55 [-2.70, -0.44] | [-2.68, -0.42] | non-inferior only | non-inferior only | -1.06, -1.82, -2.32, -0.22, -2.33 | -1.35 [-2.56, -0.16] |
| LW | -0.09 [-1.04, +0.75] | inconclusive | non-inferior only | -3.29 [-4.81, -1.73] | [-5.04, -1.53] | non-inferior only | non-inferior only | -1.26, -5.22, -3.01, -3.57, -3.37 | -3.19 [-4.55, -1.84] |
| LSW | -0.18 [-1.11, +0.74] | inconclusive | non-inferior only | -1.63 [-2.81, -0.51] | [-2.64, -0.61] | non-inferior only | non-inferior only | -3.00, -1.72, -1.25, -1.27, -0.90 | -1.45 [-2.99, +0.04] |
| pooled | +0.03 [-0.45, +0.50] | equivalent | equivalent | -1.98 [-2.64, -1.36] | [-2.43, -1.54] | non-inferior only | non-inferior only | | |

Policy vs control:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | -0.05 [-0.79, +0.70] | inconclusive | equivalent | -0.29 [-1.13, +0.58] | [-1.12, +0.54] | inconclusive | non-inferior only | -0.36, +0.84, -0.88, -0.67, -0.35 | -0.23 [-1.24, +0.70] |
| LS | -0.70 [-1.58, +0.21] | non-inferior only | non-inferior only | +0.05 [-1.01, +1.09] | [-0.78, +0.88] | inconclusive | inconclusive | -0.06, -0.42, -0.71, +0.50, +0.92 | +0.75 [-0.54, +2.01] |
| LW | -0.58 [-1.15, -0.07] | non-inferior only | non-inferior only | +0.41 [-0.49, +1.34] | [-0.43, +1.26] | inconclusive | inconclusive | +0.64, +0.74, -0.72, +0.36, +1.05 | +1.00 [+0.13, +1.95] |
| LSW | -0.14 [-1.30, +1.00] | inconclusive | non-inferior only | -0.31 [-1.31, +0.78] | [-1.32, +0.69] | inconclusive | non-inferior only | -0.16, -0.83, -0.71, -0.92, +1.04 | -0.17 [-1.47, +1.13] |
| pooled | -0.37 [-0.97, +0.16] | non-inferior only | equivalent | -0.04 [-0.62, +0.52] | [-0.67, +0.60] | inconclusive | equivalent | | |

Control vs specialist:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | +0.71 [-0.19, +1.72] | inconclusive | inconclusive | -1.02 [-2.35, +0.10] | [-2.11, +0.07] | non-inferior only | non-inferior only | -0.73, -0.72, -2.43, -1.14, -0.07 | - |
| LS | +0.70 [-0.19, +1.55] | inconclusive | inconclusive | -1.24 [-2.54, +0.02] | [-2.46, -0.02] | non-inferior only | non-inferior only | -0.72, -1.10, -1.10, -0.37, -2.92 | - |
| LW | +0.61 [-0.43, +1.66] | inconclusive | inconclusive | -3.51 [-5.17, -1.87] | [-5.54, -1.48] | non-inferior only | non-inferior only | -1.63, -5.72, -2.20, -3.75, -4.23 | - |
| LSW | +0.25 [-1.08, +1.66] | inconclusive | inconclusive | -1.03 [-2.34, +0.20] | [-2.25, +0.20] | non-inferior only | non-inferior only | -2.47, -0.77, -0.31, -0.03, -1.55 | - |
| pooled | +0.57 [-0.15, +1.40] | inconclusive | inconclusive | -1.70 [-2.37, -0.99] | [-2.20, -1.19] | non-inferior only | non-inferior only | | |

Policy vs best rule:

| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | change 0.5 - 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| L | -2.15 [-3.16, -1.15] | non-inferior only | non-inferior only | -6.23 [-7.63, -4.82] | [-6.69, -5.77] | non-inferior only | non-inferior only | -6.16, -5.64, -6.52, -6.54, -6.29 | -4.08 [-5.35, -2.80] |
| LS | -3.62 [-4.86, -2.28] | non-inferior only | non-inferior only | -5.24 [-6.46, -3.98] | [-5.82, -4.65] | non-inferior only | non-inferior only | -5.59, -5.73, -5.38, -4.78, -4.70 | -1.62 [-3.07, -0.21] |
| LW | -3.27 [-4.51, -2.03] | non-inferior only | non-inferior only | -6.10 [-7.43, -4.84] | [-6.75, -5.44] | non-inferior only | non-inferior only | -5.61, -6.19, -6.79, -6.35, -5.53 | -2.83 [-4.34, -1.31] |
| LSW | -4.38 [-5.91, -2.78] | non-inferior only | non-inferior only | -5.16 [-6.38, -3.90] | [-5.70, -4.63] | non-inferior only | non-inferior only | -5.63, -5.38, -5.00, -5.31, -4.51 | -0.79 [-2.54, +0.93] |
| pooled | -3.35 [-4.34, -2.29] | non-inferior only | non-inferior only | -5.68 [-6.52, -4.82] | [-5.99, -5.37] | non-inferior only | non-inferior only | | |

Equivalent on all four combinations with L at lag scale 0.5 (0.5% / 1%): False / False (failing at 1%: L,LS,LW,LSW); scale 1.0: False / False.

Policy vs each rule, mean d (shorter %), scale 1.0 / 0.5:

| comb. | FIFO | MOR | SPT | MWKR |
|---|---|---|---|---|
| L | -6.81 (87) / -12.52 (96) | -5.65 (75) / -10.95 (93) | -9.63 (85) / -22.35 (97) | -7.35 (78) / -12.14 (87) |
| LS | -8.47 (86) / -11.35 (92) | -8.67 (86) / -11.87 (91) | -15.17 (90) / -24.82 (98) | -9.10 (87) / -10.30 (88) |
| LW | -9.71 (85) / -11.71 (90) | -7.46 (77) / -10.47 (88) | -12.62 (84) / -24.14 (96) | -7.86 (77) / -10.13 (84) |
| LSW | -12.11 (93) / -10.89 (91) | -9.70 (85) / -10.62 (90) | -18.16 (96) / -24.08 (99) | -8.88 (78) / -9.64 (89) |

## Diagnostics

### Validation at the kept checkpoints (20 SD2 10x5 validation records)

| comb. | specialist (mean of seeds) | policy minus specialist % per seed | control minus specialist % per seed | specialist kept update | specialist gain in the last quarter % |
|---|---|---|---|---|---|
| N | 383.8 | +3.29, +2.86, +0.47, -1.03, +3.02 | +1.51, +3.60, +0.57, -1.36, +1.31 | 1000, 700, 675, 925, 725 | 1.92, -0.74, -0.12, 0.82, -2.31 |
| L | 866.0 | +0.38, -0.35, +0.62, +0.73, +0.08 | +1.63, -0.14, +1.19, -0.28, +0.08 | 1000, 875, 675, 750, 600 | 0.43, 0.03, -0.05, -0.26, -0.18 |
| S | 704.0 | +0.20, -0.99, +1.03, +0.70, -1.46 | +0.80, -0.08, +1.29, -0.92, -0.86 | 775, 950, 475, 975, 825 | 0.13, 1.50, -1.23, 0.23, 0.28 |
| W | 458.3 | -0.02, +1.83, -1.60, +2.23, +2.70 | -1.78, -1.68, -2.42, +1.63, +0.99 | 900, 600, 775, 700, 625 | 0.15, -0.78, 0.52, -1.25, -1.02 |
| LS | 1056.7 | -0.79, +0.19, +0.21, -0.61, +0.35 | - | 625, 925, 800, 950, 900 | -0.58, 0.10, 0.16, 0.39, 1.49 |
| LW | 884.9 | +0.11, +0.05, +0.75, +2.59, +0.14 | - | 450, 375, 425, 800, 900 | -0.62, -0.46, -0.60, 0.16, 0.08 |
| SW | 854.8 | - | - | 425, 950, 675, 775, 300 | -0.12, 0.36, -0.61, 1.34, -0.15 |
| LSW | 1104.9 | - | - | 575, 350, 950, 425, 700 | -0.77, -1.01, 0.61, -0.85, -0.53 |
| (released) | | | | 1050, 1350, 1500, 1650, 1775 | -0.55, -0.25, -0.21, 0.03, 0.14 |
| (control) | | | | 1700, 1975, 1850, 1725, 1850 | 0.71, 0.04, 0.34, 0.86, 0.62 |

### Rule effects (mean over instances of the R / N makespan ratio) and exact ties

| set | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix best rule | 1.88 | 1.69 | 1.20 | 2.35 | 1.96 | 2.02 | 2.54 |
| pub_SD2_test_10x5+mix policy | 2.15 | 1.84 | 1.27 | 2.64 | 2.22 | 2.23 | 2.83 |
| pub_SD2_test_20x10+mix best rule | 1.73 | 1.75 | 1.15 | 2.30 | 1.82 | 2.00 | 2.47 |
| pub_SD2_test_20x10+mix policy | 2.44 | 2.12 | 1.22 | 3.18 | 2.54 | 2.53 | 3.39 |
| pub_Brandimarte best rule | 1.61 | 1.60 | 1.19 | 2.13 | 1.72 | 1.87 | 2.27 |
| pub_Brandimarte policy | 1.63 | 1.66 | 1.21 | 2.15 | 1.71 | 1.91 | 2.33 |
| pub_Hurink_edata best rule | 1.48 | 1.53 | 1.23 | 1.91 | 1.64 | 1.81 | 2.15 |
| pub_Hurink_edata policy | 1.47 | 1.57 | 1.24 | 1.93 | 1.63 | 1.86 | 2.15 |
| pub_Hurink_rdata best rule | 1.66 | 1.55 | 1.18 | 2.09 | 1.75 | 1.84 | 2.29 |
| pub_Hurink_rdata policy | 1.62 | 1.55 | 1.19 | 2.04 | 1.72 | 1.84 | 2.23 |
| pub_Hurink_vdata best rule | 1.86 | 1.52 | 1.16 | 2.26 | 1.93 | 1.76 | 2.39 |
| pub_Hurink_vdata policy | 1.72 | 1.48 | 1.16 | 2.10 | 1.78 | 1.73 | 2.23 |

Instances on which every policy seed, every specialist seed and the best rule give the same makespan, and the correlation over instances of the seed-mean policy and specialist makespans:

| set | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| pub_SD2_test_10x5+mix | 0 (0.96) | 3 (1.00) | 0 (0.94) | 1 (0.95) | 0 (0.99) | 10 (0.99) | 0 (0.95) | 0 (0.99) |
| pub_SD2_test_20x10+mix | 0 (0.91) | 0 (1.00) | 0 (0.86) | 0 (0.87) | 0 (0.97) | 2 (0.99) | 0 (0.90) | 0 (0.96) |
| pub_Brandimarte | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_edata | 0 (1.00) | 1 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 3 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_rdata | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 2 (1.00) | 0 (1.00) | 0 (1.00) |
| pub_Hurink_vdata | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |

## Combined verdict against the per-instance best rule

d = policy minus per-instance best rule, over the rule (five seeds). ahead: the hierarchical 95% CI lies below zero and the Holm-adjusted one-sided p-values for "the policy is shorter" (seed-level t and hierarchical bootstrap) are both below 0.05. behind: the CI lies above zero and the Holm-adjusted one-sided p-values for "the policy is longer" (each kind with its own Holm family over the eight combinations) are both below 0.05. not distinguishable: otherwise.

### pub_SD2_test_10x5+mix

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -13.72 | [-15.16, -12.25] | 1.3e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| L | -2.15 | [-3.16, -1.15] | 7.6e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| S | -6.23 | [-7.52, -4.89] | 7.6e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| W | -8.34 | [-10.03, -6.65] | 5.3e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LS | -3.62 | [-4.86, -2.28] | 5.3e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LW | -3.27 | [-4.51, -2.03] | 5.3e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| SW | -4.93 | [-6.27, -3.62] | 7.6e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LSW | -4.38 | [-5.91, -2.78] | 4.9e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |

Ahead on 8, behind on 0, not distinguishable on 0 of 8 combinations.

### pub_SD2_test_20x10+mix

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -30.53 | [-31.47, -29.60] | 1.3e-07 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| L | -2.32 | [-2.98, -1.67] | 3.8e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| S | -15.78 | [-16.68, -14.88] | 1.4e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| W | -26.14 | [-27.56, -24.66] | 1.1e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LS | -4.06 | [-5.02, -3.08] | 1.1e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LW | -3.40 | [-4.26, -2.57] | 1.6e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| SW | -12.20 | [-13.17, -11.17] | 3.7e-06 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| LSW | -4.66 | [-5.69, -3.61] | 2.8e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |

Ahead on 8, behind on 0, not distinguishable on 0 of 8 combinations.

### pub_Brandimarte

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -4.07 | [-8.48, -0.69] | 0.00023 / 0.036 | 1 / 0.99 | 1 / 1 | ahead |
| L | -4.29 | [-8.21, -0.98] | 0.0004 / 0.027 | 1 / 1 | 1 / 1 | ahead |
| S | -0.64 | [-3.37, +1.97] | 0.17 / 0.38 | 0.83 / 0.68 | 1 / 1 | not distinguishable |
| W | -2.71 | [-5.65, -0.16] | 0.0024 / 0.089 | 1 / 0.98 | 1 / 1 | not distinguishable |
| LS | -3.48 | [-7.12, +0.03] | 0.00033 / 0.1 | 1 / 0.97 | 1 / 1 | not distinguishable |
| LW | -4.79 | [-8.48, -1.01] | 0.00035 / 0.046 | 1 / 0.99 | 1 / 1 | ahead |
| SW | -1.87 | [-4.10, +0.34] | 0.038 / 0.14 | 0.99 / 0.95 | 1 / 1 | not distinguishable |
| LSW | -1.84 | [-6.05, +2.30] | 0.038 / 0.38 | 0.99 / 0.81 | 1 / 1 | not distinguishable |

Ahead on 3, behind on 0, not distinguishable on 5 of 8 combinations.

### pub_Hurink_edata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | -0.63 | [-2.20, +1.03] | 1 / 1 | 0.8 / 0.78 | 1 / 1 | not distinguishable |
| L | -1.73 | [-3.24, -0.23] | 5.5e-05 / 0.086 | 1 / 0.99 | 1 / 1 | not distinguishable |
| S | +1.68 | [+0.60, +2.77] | 1 / 1 | 0.0012 / 0.0013 | 0.0093 / 0.0091 | behind |
| W | +0.65 | [-0.80, +2.08] | 1 / 1 | 0.14 / 0.18 | 0.86 / 1 | not distinguishable |
| LS | +0.17 | [-1.17, +1.50] | 1 / 1 | 0.24 / 0.4 | 1 / 1 | not distinguishable |
| LW | -1.78 | [-3.21, -0.38] | 0.0026 / 0.051 | 1 / 0.99 | 1 / 1 | not distinguishable |
| SW | +2.31 | [+1.22, +3.56] | 1 / 1 | 0.0017 / 5e-05 | 0.012 / 0.0004 | behind |
| LSW | -0.36 | [-1.57, +0.83] | 0.17 / 1 | 0.97 / 0.72 | 1 / 1 | not distinguishable |

Ahead on 0, behind on 2, not distinguishable on 6 of 8 combinations.

### pub_Hurink_rdata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | +1.28 | [-0.25, +2.95] | 1 / 1 | 0.086 / 0.054 | 0.51 / 0.32 | not distinguishable |
| L | -1.15 | [-2.40, +0.09] | 0.024 / 0.24 | 1 / 0.97 | 1 / 1 | not distinguishable |
| S | +0.75 | [-0.46, +2.04] | 1 / 1 | 0.1 / 0.12 | 0.51 / 0.61 | not distinguishable |
| W | +2.90 | [+0.96, +5.11] | 1 / 1 | 0.021 / 0.00085 | 0.16 / 0.0068 | not distinguishable |
| LS | -1.68 | [-2.84, -0.53] | 0.011 / 0.019 | 1 / 1 | 1 / 1 | ahead |
| LW | -0.55 | [-1.96, +0.84] | 0.37 / 1 | 0.93 / 0.78 | 1 / 1 | not distinguishable |
| SW | +1.25 | [-0.03, +2.59] | 1 / 1 | 0.02 / 0.027 | 0.16 / 0.19 | not distinguishable |
| LSW | -1.18 | [-2.54, +0.29] | 0.086 / 0.33 | 0.99 / 0.95 | 1 / 1 | not distinguishable |

Ahead on 1, behind on 0, not distinguishable on 7 of 8 combinations.

### pub_Hurink_vdata

| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|---|
| N | +5.70 | [+3.55, +8.05] | 1 / 1 | 0.0024 / 5e-05 | 0.017 / 0.0004 | behind |
| L | -1.80 | [-3.08, -0.63] | 0.0005 / 0.0077 | 1 / 1 | 1 / 1 | ahead |
| S | +2.63 | [+1.18, +4.21] | 1 / 1 | 0.011 / 0.00015 | 0.057 / 0.00075 | not distinguishable |
| W | +5.45 | [+3.53, +7.62] | 1 / 1 | 0.0018 / 5e-05 | 0.014 / 0.0004 | behind |
| LS | -1.56 | [-2.62, -0.51] | 0.0065 / 0.012 | 1 / 1 | 1 / 1 | ahead |
| LW | -2.37 | [-3.89, -0.95] | 0.0011 / 0.0024 | 1 / 1 | 1 / 1 | ahead |
| SW | +3.77 | [+2.30, +5.39] | 1 / 1 | 0.0034 / 5e-05 | 0.021 / 0.0004 | behind |
| LSW | -1.23 | [-2.42, -0.06] | 0.002 / 0.099 | 1 / 0.98 | 1 / 1 | not distinguishable |

Ahead on 3, behind on 3, not distinguishable on 2 of 8 combinations.


## 8. Equal-experience policy (pubx-eqact, 6000 updates on N,L,S,W,LS,LW)

The released recipe with 6000 updates instead of 2000, so that each of its six combinations gets as many episodes as a specialist (1000 updates) gets of its one. Seeds 301, 302, 303, 304, 305, seed s against seed s; same estimators, margins and streams as sections 1 to 7. Configurations equal those of pubx-eq of the same seed except for mask_mode, model_suffix.

- Checks: pub_SD2_test_10x5+mix: 5 summaries, 5 regime-N arrays at or above the bound; pub_SD2_test_20x10+mix: 5 summaries, 5 regime-N arrays at or above the bound; pub_Brandimarte: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_edata: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_rdata: 5 summaries, 5 regime-N arrays at or above the bound; pub_Hurink_vdata: 5 summaries, 5 regime-N arrays at or above the bound; pub_SD2_test_10x5+mix_lag050: 5 summaries, 5 regime-N arrays at or above the bound, free check 10 N and S arrays bit for bit.

### pub_SD2_test_10x5+mix (in distribution)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +1.12 [-0.27, +2.52] | inconclusive | +0.76 [-0.60, +2.09] | [-0.74, +2.26] | inconclusive | inconclusive | neither | +2.37, +0.91, -0.21, -0.63, +1.35 | -0.12 [-1.09, +0.85] | inconclusive | non-inferior only | neither |
| L | +0.58 [-0.05, +1.21] | inconclusive | -0.13 [-0.69, +0.48] | [-0.67, +0.41] | non-inferior only | equivalent | neither | +0.62, -0.35, -0.12, -0.45, -0.34 | -0.63 [-1.16, -0.09] | non-inferior only | non-inferior only | ahead |
| S | +0.51 [-0.50, +1.64] | inconclusive | +0.14 [-0.87, +1.21] | [-0.73, +1.01] | inconclusive | inconclusive | neither | -0.24, +1.33, -0.00, +0.09, -0.46 | -0.10 [-1.34, +1.23] | inconclusive | inconclusive | neither |
| W | +0.30 [-1.06, +1.67] | inconclusive | -0.46 [-1.77, +0.84] | [-1.19, +0.27] | inconclusive | non-inferior only | neither | -0.41, -0.18, -0.12, -1.49, -0.09 | -0.35 [-1.54, +0.82] | inconclusive | non-inferior only | neither |
| LS | -0.20 [-1.04, +0.68] | non-inferior only | +0.15 [-0.83, +1.19] | [-0.95, +1.24] | inconclusive | inconclusive | neither | +0.33, +1.61, -0.51, -0.42, -0.27 | +0.51 [-0.35, +1.40] | inconclusive | inconclusive | neither |
| LW | -0.09 [-1.04, +0.75] | non-inferior only | -0.14 [-0.88, +0.51] | [-0.58, +0.30] | inconclusive | equivalent | neither | +0.25, -0.52, -0.52, +0.05, +0.02 | +0.04 [-0.56, +0.66] | inconclusive | equivalent | neither |
| SW | -0.83 [-2.23, +0.56] | non-inferior only | -0.91 [-2.41, +0.52] | [-2.75, +0.93] | inconclusive | non-inferior only | neither | -0.21, -1.48, +0.03, +0.36, -3.25 | +0.15 [-0.87, +1.19] | inconclusive | inconclusive | neither |
| LSW | -0.18 [-1.11, +0.74] | non-inferior only | -0.14 [-1.15, +0.89] | [-1.23, +0.94] | inconclusive | non-inferior only | neither | +0.79, -1.26, +0.69, -0.37, -0.57 | +0.25 [-0.56, +1.08] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | +0.15 [-0.25, +0.58] | -0.09 [-0.51, +0.33] | non-inferior only | equivalent | False / False | N,S,W,LS,SW,LSW | -0.03 [-0.39, +0.33] | equivalent | equivalent |
| Train | +0.37 [-0.08, +0.84] | +0.05 [-0.45, +0.54] | inconclusive | equivalent | False / False | N,S,W,LS | -0.11 [-0.50, +0.28] | equivalent | equivalent |
| Held | -0.50 [-1.42, +0.42] | -0.53 [-1.53, +0.47] | non-inferior only | non-inferior only | False / False | SW,LSW | +0.20 [-0.48, +0.87] | inconclusive | equivalent |

Regime N, gap to best known: equal-experience 21.57 (sd 0.52), released 22.01, N specialist 20.97.

### pub_SD2_test_20x10+mix (size transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.31 [-1.05, +1.90] | inconclusive | +0.38 [-1.50, +2.12] | [-2.26, +3.01] | inconclusive | inconclusive | neither | +1.60, -0.53, +0.90, -2.79, +2.70 | +0.25 [-1.19, +1.74] | inconclusive | inconclusive | neither |
| L | -0.10 [-0.75, +0.55] | equivalent | -0.07 [-0.81, +0.55] | [-0.96, +0.81] | inconclusive | equivalent | neither | +0.15, +0.01, +0.52, -1.31, +0.25 | +0.07 [-0.50, +0.62] | inconclusive | equivalent | neither |
| S | -1.32 [-3.04, +0.47] | non-inferior only | -0.71 [-1.85, +0.42] | [-1.95, +0.53] | non-inferior only | non-inferior only | neither | -0.32, +0.13, -1.09, +0.00, -2.28 | +0.80 [-0.74, +2.47] | inconclusive | inconclusive | neither |
| W | -1.26 [-2.98, +0.85] | non-inferior only | -1.33 [-2.75, +0.18] | [-2.97, +0.30] | non-inferior only | non-inferior only | neither | -2.46, -1.30, +0.53, -2.69, -0.74 | +0.36 [-1.14, +1.80] | inconclusive | inconclusive | neither |
| LS | -0.80 [-1.55, -0.04] | non-inferior only | -0.61 [-1.50, +0.25] | [-1.31, +0.10] | non-inferior only | non-inferior only | neither | -0.33, +0.15, -0.79, -0.68, -1.39 | +0.34 [-0.45, +1.15] | inconclusive | inconclusive | neither |
| LW | +0.06 [-0.47, +0.63] | equivalent | -0.05 [-0.59, +0.47] | [-0.33, +0.23] | non-inferior only | equivalent | neither | -0.23, +0.09, +0.18, -0.35, +0.04 | -0.01 [-0.56, +0.57] | inconclusive | equivalent | neither |
| SW | -3.07 [-4.41, -1.84] | non-inferior only | -2.92 [-4.45, -1.42] | [-4.88, -0.96] | non-inferior only | non-inferior only | ahead | -2.78, -0.80, -2.42, -3.46, -5.14 | +0.34 [-1.06, +1.69] | inconclusive | inconclusive | neither |
| LSW | -0.74 [-1.87, +0.44] | non-inferior only | -0.60 [-1.93, +0.76] | [-2.25, +1.05] | inconclusive | non-inferior only | neither | -1.72, +0.75, -0.77, +0.77, -2.04 | +0.35 [-0.65, +1.38] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.87 [-1.38, -0.36] | -0.74 [-1.20, -0.27] | non-inferior only | non-inferior only | False / False | N,S,W,LS,SW,LSW | +0.31 [-0.33, +0.96] | inconclusive | equivalent |
| Train | -0.52 [-1.11, +0.08] | -0.40 [-0.93, +0.09] | non-inferior only | equivalent | False / False | N,S,W,LS | +0.30 [-0.36, +0.98] | inconclusive | equivalent |
| Held | -1.91 [-2.93, -0.84] | -1.76 [-2.97, -0.60] | non-inferior only | non-inferior only | False / False | SW,LSW | +0.35 [-0.59, +1.27] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 24.92 (sd 2.29), released 24.83, N specialist 24.64.

### pub_Brandimarte (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.56 [-0.79, +2.15] | inconclusive | -0.27 [-2.23, +1.49] | [-1.85, +1.32] | inconclusive | inconclusive | neither | +0.04, +1.73, -1.56, -1.12, -0.44 | -0.74 [-2.76, +1.44] | inconclusive | inconclusive | neither |
| L | -3.19 [-6.83, -0.24] | non-inferior only | -3.54 [-7.67, -0.54] | [-6.30, -0.78] | non-inferior only | non-inferior only | ahead | -2.15, -4.31, -6.21, -4.50, -0.52 | -0.37 [-1.97, +1.49] | inconclusive | inconclusive | neither |
| S | -0.37 [-2.15, +1.78] | inconclusive | -0.15 [-1.89, +1.72] | [-1.24, +0.94] | inconclusive | inconclusive | neither | +0.31, -0.89, +1.14, -0.37, -0.93 | +0.35 [-1.86, +2.94] | inconclusive | inconclusive | neither |
| W | +0.33 [-1.56, +2.76] | inconclusive | +0.86 [-0.85, +3.28] | [-0.42, +2.14] | inconclusive | inconclusive | neither | +1.32, -0.42, +1.30, +0.02, +2.09 | +0.58 [-0.34, +1.55] | inconclusive | inconclusive | neither |
| LS | -1.84 [-4.54, +0.35] | non-inferior only | -1.06 [-4.93, +3.19] | [-5.89, +3.77] | inconclusive | inconclusive | neither | +5.24, -3.28, -5.04, -0.94, -1.27 | +0.82 [-1.59, +3.72] | inconclusive | inconclusive | neither |
| LW | -0.70 [-2.54, +1.18] | inconclusive | -0.92 [-3.42, +1.22] | [-3.27, +1.43] | inconclusive | inconclusive | neither | +1.27, -0.58, -3.89, -1.23, -0.19 | -0.15 [-2.40, +2.11] | inconclusive | inconclusive | neither |
| SW | -1.19 [-3.22, +0.67] | non-inferior only | +0.18 [-1.99, +2.46] | [-1.05, +1.40] | inconclusive | inconclusive | neither | -0.43, +1.34, +1.16, -0.63, -0.56 | +1.49 [-0.45, +3.82] | inconclusive | inconclusive | neither |
| LSW | +0.06 [-1.96, +2.07] | inconclusive | -0.08 [-3.14, +2.86] | [-3.77, +3.61] | inconclusive | inconclusive | neither | +1.65, +1.21, -2.05, +3.01, -4.23 | +0.01 [-2.94, +3.03] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.79 [-1.50, -0.08] | -0.62 [-1.67, +0.45] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW,SW,LSW | +0.25 [-0.47, +1.16] | inconclusive | inconclusive |
| Train | -0.87 [-1.89, +0.08] | -0.85 [-2.24, +0.48] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW | +0.08 [-0.74, +1.05] | inconclusive | inconclusive |
| Held | -0.57 [-2.00, +0.70] | +0.05 [-1.85, +1.89] | inconclusive | inconclusive | False / False | SW,LSW | +0.75 [-0.74, +2.33] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 10.97 (sd 1.06), released 11.99, N specialist 11.38.

### pub_Hurink_edata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.44 [-1.90, +2.93] | inconclusive | -0.07 [-2.17, +2.11] | [-3.21, +3.07] | inconclusive | inconclusive | neither | +1.64, -0.26, -2.41, -2.57, +3.24 | -0.39 [-1.50, +0.60] | inconclusive | non-inferior only | neither |
| L | -1.57 [-2.71, -0.62] | non-inferior only | -1.38 [-2.45, -0.39] | [-2.39, -0.37] | non-inferior only | non-inferior only | ahead | -1.36, -1.93, -1.47, -2.11, -0.04 | +0.30 [-0.65, +1.23] | inconclusive | inconclusive | neither |
| S | +0.45 [-0.58, +1.48] | inconclusive | -0.53 [-1.31, +0.28] | [-1.27, +0.20] | non-inferior only | non-inferior only | neither | -0.22, +0.34, -1.09, -0.97, -0.73 | -0.80 [-1.66, +0.09] | non-inferior only | non-inferior only | neither |
| W | +0.11 [-1.29, +1.71] | inconclusive | -0.45 [-1.85, +1.03] | [-2.02, +1.11] | inconclusive | inconclusive | neither | -1.03, -0.06, -1.36, -1.40, +1.58 | -0.41 [-1.36, +0.48] | non-inferior only | non-inferior only | neither |
| LS | -0.23 [-1.45, +0.94] | non-inferior only | -0.64 [-2.21, +0.73] | [-2.63, +1.35] | inconclusive | non-inferior only | neither | +0.98, -0.01, -3.27, -0.76, -0.14 | -0.25 [-1.49, +0.94] | inconclusive | non-inferior only | neither |
| LW | -1.10 [-2.31, -0.03] | non-inferior only | -0.68 [-1.77, +0.38] | [-1.54, +0.18] | non-inferior only | non-inferior only | neither | +0.31, -1.62, -0.88, -0.64, -0.55 | +0.57 [-0.34, +1.51] | inconclusive | inconclusive | neither |
| SW | -0.40 [-1.43, +0.60] | non-inferior only | -0.65 [-1.68, +0.49] | [-1.90, +0.61] | non-inferior only | non-inferior only | neither | +0.91, -1.28, -0.16, -1.49, -1.21 | -0.08 [-1.05, +1.02] | inconclusive | inconclusive | neither |
| LSW | -0.46 [-1.22, +0.34] | non-inferior only | -0.24 [-1.39, +1.00] | [-1.64, +1.17] | inconclusive | inconclusive | neither | +0.92, -0.58, +0.91, -0.78, -1.67 | +0.34 [-0.76, +1.44] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.35 [-1.02, +0.36] | -0.58 [-1.25, +0.09] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW,SW,LSW | -0.09 [-0.46, +0.31] | equivalent | equivalent |
| Train | -0.32 [-1.18, +0.55] | -0.63 [-1.48, +0.23] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW | -0.16 [-0.58, +0.26] | non-inferior only | equivalent |
| Held | -0.43 [-1.12, +0.27] | -0.44 [-1.39, +0.58] | inconclusive | non-inferior only | False / False | SW,LSW | +0.13 [-0.77, +1.05] | inconclusive | inconclusive |

Regime N, gap to best known: equal-experience 17.86 (sd 1.25), released 18.47, N specialist 18.13.

### pub_Hurink_rdata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.45 [+0.42, +4.78] | inferior | +0.76 [-1.68, +3.38] | [-3.05, +4.57] | inconclusive | inconclusive | neither | +4.52, -1.21, -1.50, -1.69, +3.68 | -1.55 [-2.99, +0.05] | non-inferior only | non-inferior only | neither |
| L | -1.10 [-2.49, +0.26] | non-inferior only | -1.40 [-2.74, -0.00] | [-3.15, +0.35] | non-inferior only | non-inferior only | ahead | -2.18, -1.53, -2.59, -1.70, +1.01 | -0.21 [-0.97, +0.56] | inconclusive | equivalent | neither |
| S | -0.45 [-1.57, +0.86] | non-inferior only | -0.77 [-1.84, +0.30] | [-1.79, +0.24] | non-inferior only | non-inferior only | neither | +0.39, -0.95, -1.39, -1.61, -0.31 | -0.20 [-1.42, +1.04] | inconclusive | inconclusive | neither |
| W | +0.81 [-1.14, +3.14] | inconclusive | -0.55 [-2.40, +1.51] | [-3.32, +2.22] | inconclusive | inconclusive | neither | +1.27, -1.34, -2.71, -2.28, +2.33 | -1.18 [-2.24, -0.19] | non-inferior only | non-inferior only | ahead |
| LS | -0.21 [-1.15, +0.66] | non-inferior only | -0.07 [-1.33, +1.26] | [-1.60, +1.46] | inconclusive | inconclusive | neither | -0.12, +1.34, -1.33, +1.00, -1.23 | +0.25 [-0.77, +1.35] | inconclusive | inconclusive | neither |
| LW | -1.05 [-2.27, +0.15] | non-inferior only | -1.41 [-3.02, +0.34] | [-3.76, +0.95] | non-inferior only | non-inferior only | neither | +1.68, -3.39, -2.39, -1.57, -1.37 | -0.25 [-1.24, +0.84] | inconclusive | non-inferior only | neither |
| SW | -0.63 [-2.57, +1.02] | inconclusive | -1.15 [-3.50, +0.86] | [-4.42, +2.13] | inconclusive | non-inferior only | neither | +1.84, -5.39, -0.53, -1.20, -0.46 | -0.38 [-1.45, +0.85] | inconclusive | non-inferior only | neither |
| LSW | -0.02 [-1.21, +1.25] | inconclusive | -0.37 [-1.39, +0.74] | [-1.24, +0.50] | inconclusive | non-inferior only | neither | -1.09, -0.36, +0.79, -0.51, -0.68 | -0.19 [-1.37, +1.01] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | -0.03 [-0.85, +0.97] | -0.62 [-1.52, +0.36] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW,SW,LSW | -0.46 [-1.07, +0.21] | non-inferior only | non-inferior only |
| Train | +0.07 [-0.85, +1.21] | -0.57 [-1.63, +0.55] | inconclusive | non-inferior only | False / False | N,L,S,W,LS,LW | -0.52 [-1.19, +0.21] | non-inferior only | non-inferior only |
| Held | -0.32 [-1.23, +0.55] | -0.76 [-2.00, +0.32] | non-inferior only | non-inferior only | False / False | SW,LSW | -0.28 [-1.22, +0.64] | inconclusive | non-inferior only |

Regime N, gap to best known: equal-experience 16.72 (sd 2.63), released 18.72, N specialist 15.96.

### pub_Hurink_vdata (distribution transfer)

| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | 1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +3.09 [+1.85, +4.32] | inferior | +1.22 [-0.34, +2.80] | [-1.03, +3.47] | inconclusive | inconclusive | neither | +3.56, +0.33, -1.28, +1.80, +1.71 | -1.66 [-3.51, +0.14] | non-inferior only | non-inferior only | neither |
| L | -0.91 [-1.87, -0.17] | non-inferior only | -0.90 [-1.80, -0.12] | [-1.61, -0.18] | non-inferior only | non-inferior only | ahead | -1.69, -0.71, -0.99, -1.00, -0.10 | +0.06 [-0.44, +0.52] | inconclusive | equivalent | neither |
| S | +0.06 [-1.86, +2.06] | inconclusive | -1.04 [-2.85, +0.64] | [-3.64, +1.56] | inconclusive | non-inferior only | neither | +1.57, -1.75, -4.04, -1.08, +0.09 | -1.01 [-2.10, +0.10] | non-inferior only | non-inferior only | neither |
| W | +1.63 [-0.23, +3.73] | inconclusive | +0.23 [-2.06, +2.78] | [-3.31, +3.78] | inconclusive | inconclusive | neither | +3.91, -1.46, -1.55, -2.42, +2.68 | -1.29 [-2.62, +0.22] | non-inferior only | non-inferior only | neither |
| LS | -0.00 [-1.25, +1.15] | inconclusive | -0.32 [-1.67, +0.79] | [-1.98, +1.34] | inconclusive | non-inferior only | neither | -0.31, +0.12, -2.60, +0.48, +0.72 | -0.25 [-0.81, +0.32] | non-inferior only | equivalent | neither |
| LW | -0.79 [-1.79, +0.13] | non-inferior only | -0.83 [-1.84, +0.20] | [-2.03, +0.36] | non-inferior only | non-inferior only | neither | +0.73, -1.67, -1.20, -1.44, -0.57 | +0.01 [-0.66, +0.68] | inconclusive | equivalent | neither |
| SW | -1.54 [-5.87, +1.76] | inconclusive | -2.70 [-6.94, +0.28] | [-8.54, +3.13] | non-inferior only | non-inferior only | neither | -0.54, -10.66, +1.21, -3.06, -0.48 | -1.07 [-1.77, -0.38] | non-inferior only | non-inferior only | ahead |
| LSW | -0.09 [-0.90, +0.80] | equivalent | +0.06 [-0.86, +1.05] | [-0.53, +0.65] | inconclusive | inconclusive | neither | +0.15, +0.04, +0.80, -0.29, -0.42 | +0.26 [-0.65, +1.18] | inconclusive | inconclusive | neither |

| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| All | +0.18 [-0.76, +1.10] | -0.53 [-1.49, +0.44] | non-inferior only | non-inferior only | False / False | N,L,S,W,LS,LW,SW,LSW | -0.62 [-1.14, -0.07] | non-inferior only | non-inferior only |
| Train | +0.51 [-0.29, +1.43] | -0.27 [-1.35, +0.78] | inconclusive | non-inferior only | False / False | N,L,S,W,LS,LW | -0.69 [-1.37, +0.01] | non-inferior only | non-inferior only |
| Held | -0.82 [-2.87, +0.84] | -1.32 [-3.50, +0.40] | non-inferior only | non-inferior only | False / False | SW,LSW | -0.41 [-0.94, +0.15] | non-inferior only | equivalent |

Regime N, gap to best known: equal-experience 10.42 (sd 3.23), released 12.47, N specialist 9.15.

### Validation at the kept checkpoints (20 SD2 10x5 validation records)

| comb. | equal-exp. minus specialist % per seed | mean | released minus specialist (mean) |
|---|---|---|---|
| N | +1.31, +1.63, -1.95, -2.34, +3.39 | +0.41 | +1.72 |
| L | +0.39, -0.32, +0.86, +0.01, +0.71 | +0.33 | +0.29 |
| S | -0.74, +0.16, +0.03, -0.27, -1.19 | -0.40 | -0.10 |
| W | -0.80, -1.33, +0.97, +2.23, -0.02 | +0.21 | +1.03 |
| LS | -0.67, +0.55, -0.51, -0.93, +0.33 | -0.25 | -0.13 |
| LW | +0.52, +0.37, +1.20, +1.10, +0.41 | +0.72 | +0.73 |

Kept update per seed: 2975, 4175, 5400, 5425, 3650 (of 6000); gain in the last quarter %: -0.24, -0.19, 0.31, 0.11, -0.17.

## 9. Gap to the references (pub_SD2_test_10x5+mix)

Reference: the best schedule CP-SAT finds within 150 s on 2 workers (compact model, warm-started from the best of the four rules; scripts/p91_public_cpsat_refs.py, records it marks valid only; folder or_solution/PUBFAMILY/pub_SD2_test_10x5+mix_cpsat150s_w2). d = 100 (method - reference) / reference per instance; trained arms (five runs): crossed hierarchical 95% CI; rules: instance-level 95% CI; interval and direction only.

| comb. | n | proved | at limit (mean gap %) | policy | = / > / < % | control | specialist | best rule | FIFO | MOR | SPT | MWKR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 100 | 100 | 0 | +22.01 [+20.78, +23.30] behind | 0.0 / 100.0 / 0.0 | +21.76 [+20.29, +23.25] behind | +20.97 [+19.65, +22.36] behind | +42.16 [+39.97, +44.33] behind | +53.57 [+50.50, +56.72] behind | +52.95 [+50.08, +55.91] behind | +74.82 [+70.83, +78.83] behind | +50.30 [+47.31, +53.36] behind |
| L | 100 | 100 | 0 | +3.61 [+2.69, +4.64] behind | 30.6 / 69.4 / 0.0 | +3.74 [+2.64, +4.98] behind | +3.10 [+2.19, +4.07] behind | +6.19 [+4.70, +7.81] behind | +11.76 [+9.81, +13.80] behind | +10.42 [+8.43, +12.54] behind | +16.28 [+13.20, +19.54] behind | +12.81 [+10.35, +15.36] behind |
| S | 100 | 0 | 100 (44.31) | +17.79 [+16.45, +19.26] behind | 0.0 / 100.0 / 0.0 | +17.35 [+16.14, +18.60] behind | +17.47 [+16.44, +18.59] behind | +25.86 [+24.62, +27.10] behind | +33.86 [+32.05, +35.74] behind | +34.17 [+32.01, +36.39] behind | +56.78 [+53.40, +60.23] behind | +31.90 [+30.20, +33.61] behind |
| W | 100 | 100 | 0 | +23.85 [+21.07, +26.73] behind | 4.0 / 96.0 / 0.0 | +23.01 [+20.27, +25.85] behind | +23.96 [+21.36, +26.57] behind | +35.76 [+32.72, +38.91] behind | +47.91 [+43.75, +52.21] behind | +46.74 [+42.83, +50.84] behind | +73.33 [+68.08, +78.65] behind | +41.85 [+38.20, +45.58] behind |
| LS | 100 | 57 | 43 (4.66) | +13.07 [+11.58, +14.70] behind | 0.0 / 100.0 / 0.0 | +14.11 [+12.37, +15.99] behind | +13.53 [+11.93, +15.33] behind | +17.69 [+15.81, +19.64] behind | +24.26 [+21.84, +26.75] behind | +24.60 [+22.09, +27.21] behind | +35.89 [+31.76, +40.13] behind | +25.68 [+22.43, +29.11] behind |
| LW | 100 | 100 | 0 | +3.60 [+2.60, +4.71] behind | 34.4 / 65.6 / 0.0 | +4.36 [+3.15, +5.68] behind | +3.92 [+2.67, +5.37] behind | +7.61 [+5.76, +9.60] behind | +15.86 [+13.27, +18.56] behind | +12.82 [+10.43, +15.38] behind | +20.42 [+17.23, +23.71] behind | +13.40 [+10.90, +15.98] behind |
| SW | 100 | 1 | 99 (39.53) | +16.40 [+15.11, +17.76] behind | 0.0 / 99.6 / 0.4 | +16.93 [+15.36, +18.54] behind | +17.76 [+16.02, +19.66] behind | +22.89 [+21.13, +24.78] behind | +29.59 [+27.53, +31.67] behind | +28.90 [+26.98, +30.91] behind | +51.04 [+47.76, +54.32] behind | +27.86 [+25.77, +30.06] behind |
| LSW | 100 | 46 | 54 (7.89) | +14.39 [+12.76, +16.12] behind | 0.6 / 99.4 / 0.0 | +14.90 [+13.05, +16.94] behind | +14.94 [+13.10, +16.91] behind | +20.20 [+18.10, +22.31] behind | +31.16 [+28.31, +34.07] behind | +27.76 [+24.97, +30.53] behind | +42.25 [+38.30, +46.15] behind | +26.72 [+23.82, +29.65] behind |
| pooled | 800 | 504 | 296 | +14.34 [+13.53, +15.17] behind | 8.7 / 91.2 / 0.1 | +14.52 [+13.69, +15.37] behind | +14.46 [+13.68, +15.27] behind | +22.30 [+21.26, +23.33] behind | +31.00 [+29.68, +32.30] behind | +29.80 [+28.53, +31.05] behind | +46.35 [+44.20, +48.57] behind | +28.82 [+27.28, +30.37] behind |

Regime N against the best-known values (100 references): below 0, equal 100, above 0 (0 of them where the table marks the value proved optimal); proved optimal here and in the table: 96.
Policy shorter than the reference: 2 (run, instance) cases, listed in the JSON (possible where the reference is not proved optimal).
Records worth a look: worse than warm start 0; best rule shorter 0; ended before limit unproved 0; other status 0; records not counted 0.

## 10. The active mask against the non-delay mask

d = 100 (a - b) / b per training seed and base instance, a the run under the active mask, b its non-delay counterpart of the same seed; rules: per base instance; negative = the active mask gives the shorter schedule. Runs: crossed hierarchical 95% CI over seeds and base instances; rules: instance-level 95% CI. Pooled rows keep the combinations of a base instance together.

### pub_SD2_test_10x5+mix (in distribution)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | -4.07 [-5.52, -2.65] | ahead | -3.21 [-4.63, -1.80] | ahead | -3.27 [-4.57, -1.99] | ahead | -4.70 [-6.22, -3.09] | ahead | MWKR / SPT |
| L | -2.91 [-3.63, -2.22] | ahead | -3.12 [-4.14, -2.16] | ahead | -3.36 [-4.26, -2.51] | ahead | -1.10 [-2.04, -0.13] | ahead | MOR / SPT |
| S | -20.67 [-22.48, -18.89] | ahead | -20.54 [-22.21, -18.86] | ahead | -21.15 [-22.55, -19.77] | ahead | -11.01 [-12.66, -9.40] | ahead | MWKR / MOR |
| W | -3.00 [-4.32, -1.67] | ahead | -3.10 [-4.68, -1.55] | ahead | -2.59 [-4.11, -1.01] | ahead | -4.43 [-5.97, -2.88] | ahead | MWKR / MWKR |
| LS | -12.97 [-14.95, -10.90] | ahead | -12.77 [-14.78, -10.67] | ahead | -13.29 [-15.24, -11.23] | ahead | -6.33 [-8.05, -4.67] | ahead | FIFO / MOR |
| LW | -2.77 [-3.59, -1.97] | ahead | -2.45 [-3.44, -1.44] | ahead | -2.83 [-3.75, -1.88] | ahead | -1.06 [-2.10, -0.02] | ahead | MOR / SPT |
| SW | -16.72 [-18.21, -15.20] | ahead | -16.51 [-18.15, -14.88] | ahead | -16.00 [-17.58, -14.42] | ahead | -8.98 [-10.43, -7.54] | ahead | MWKR / MWKR |
| LSW | -15.05 [-17.04, -12.95] | ahead | -14.71 [-16.71, -12.58] | ahead | -14.71 [-16.75, -12.56] | ahead | -7.84 [-9.72, -6.01] | ahead | MWKR / MWKR |
| pooled All | -9.77 [-10.46, -9.06] | ahead | -9.55 [-10.26, -8.81] | ahead | -9.65 [-10.31, -8.96] | ahead | -5.68 [-6.33, -5.02] | ahead |  |
| pooled Train | -7.73 [-8.35, -7.08] | ahead | -7.53 [-8.22, -6.83] | ahead | - | - | -4.77 [-5.41, -4.13] | ahead |  |
| pooled Held | -15.88 [-17.16, -14.56] | ahead | -15.61 [-16.96, -14.24] | ahead | - | - | -8.41 [-9.63, -7.19] | ahead |  |
| pooled Setup | -16.35 [-17.54, -15.11] | ahead | -16.13 [-17.33, -14.88] | ahead | -16.29 [-17.42, -15.10] | ahead | -8.54 [-9.62, -7.45] | ahead |  |
| pooled NoSetup | -3.19 [-3.82, -2.55] | ahead | -2.97 [-3.58, -2.33] | ahead | -3.01 [-3.52, -2.49] | ahead | -2.82 [-3.55, -2.10] | ahead |  |

### pub_SD2_test_20x10+mix (size transfer)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | -8.74 [-9.88, -7.54] | ahead | -7.54 [-9.23, -5.97] | ahead | -5.66 [-6.94, -4.48] | ahead | +0.63 [-1.14, +2.36] | neither | MWKR / SPT |
| L | -5.51 [-6.28, -4.71] | ahead | -5.24 [-6.10, -4.40] | ahead | -5.57 [-6.40, -4.83] | ahead | -3.11 [-3.87, -2.36] | ahead | MOR / SPT |
| S | -28.37 [-29.23, -27.49] | ahead | -28.65 [-29.58, -27.72] | ahead | -27.62 [-28.67, -26.60] | ahead | -12.49 [-13.37, -11.60] | ahead | MWKR / SPT |
| W | -6.51 [-7.85, -5.18] | ahead | -5.13 [-6.78, -3.54] | ahead | -5.46 [-7.50, -3.61] | ahead | -1.44 [-3.00, +0.17] | neither | MWKR / SPT |
| LS | -15.84 [-17.29, -14.47] | ahead | -15.61 [-16.99, -14.30] | ahead | -15.45 [-16.81, -14.06] | ahead | -8.03 [-9.21, -6.85] | ahead | MOR / MWKR |
| LW | -5.41 [-6.13, -4.69] | ahead | -5.33 [-6.38, -4.36] | ahead | -5.25 [-6.09, -4.44] | ahead | -2.64 [-3.52, -1.74] | ahead | MOR / SPT |
| SW | -23.97 [-25.04, -22.92] | ahead | -23.87 [-25.21, -22.55] | ahead | -21.62 [-22.94, -20.15] | ahead | -11.19 [-12.04, -10.37] | ahead | MWKR / SPT |
| LSW | -17.46 [-18.90, -16.02] | ahead | -17.70 [-19.05, -16.36] | ahead | -17.14 [-18.69, -15.58] | ahead | -8.58 [-9.74, -7.42] | ahead | MWKR / MWKR |
| pooled All | -13.98 [-14.52, -13.43] | ahead | -13.63 [-14.11, -13.15] | ahead | -12.97 [-13.47, -12.45] | ahead | -5.86 [-6.36, -5.36] | ahead |  |
| pooled Train | -11.73 [-12.29, -11.16] | ahead | -11.25 [-11.78, -10.69] | ahead | - | - | -4.51 [-5.06, -3.97] | ahead |  |
| pooled Held | -20.72 [-21.61, -19.82] | ahead | -20.78 [-21.75, -19.79] | ahead | - | - | -9.89 [-10.61, -9.17] | ahead |  |
| pooled Setup | -21.41 [-22.19, -20.65] | ahead | -21.46 [-22.25, -20.67] | ahead | -20.46 [-21.30, -19.64] | ahead | -10.07 [-10.69, -9.45] | ahead |  |
| pooled NoSetup | -6.54 [-7.20, -5.88] | ahead | -5.81 [-6.51, -5.06] | ahead | -5.49 [-6.01, -4.99] | ahead | -1.64 [-2.33, -0.97] | ahead |  |

### pub_Brandimarte (distribution transfer)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.53 [-2.08, +3.10] | neither | +1.49 [-0.64, +3.82] | neither | +0.21 [-2.14, +2.58] | neither | -1.38 [-4.01, +0.68] | neither | MWKR / MWKR |
| L | -2.22 [-4.98, +0.20] | neither | -2.98 [-5.74, -0.58] | ahead | -2.87 [-6.11, +0.53] | neither | +1.88 [-0.69, +4.51] | neither | MWKR / MWKR |
| S | -13.72 [-19.17, -7.83] | ahead | -12.62 [-18.74, -5.97] | ahead | -14.29 [-19.83, -8.84] | ahead | -8.58 [-13.42, -3.71] | ahead | MWKR / MWKR |
| W | +1.10 [-0.42, +2.83] | neither | +0.13 [-2.34, +2.95] | neither | -0.58 [-2.74, +1.35] | neither | -0.85 [-3.06, +1.45] | neither | MWKR / MOR |
| LS | -12.98 [-17.81, -8.19] | ahead | -12.48 [-17.63, -7.57] | ahead | -11.87 [-16.36, -7.67] | ahead | -6.54 [-10.60, -3.03] | ahead | MWKR / FIFO |
| LW | -2.86 [-6.25, +0.42] | neither | -3.45 [-6.41, -0.22] | ahead | -2.13 [-4.86, +0.44] | neither | -0.01 [-1.69, +1.70] | neither | MWKR / MWKR |
| SW | -13.31 [-18.06, -8.45] | ahead | -12.62 [-17.56, -7.46] | ahead | -10.95 [-15.33, -6.36] | ahead | -7.58 [-10.28, -4.51] | ahead | MWKR / MWKR |
| LSW | -12.80 [-17.47, -8.22] | ahead | -12.83 [-17.43, -7.90] | ahead | -13.98 [-17.93, -9.75] | ahead | -7.06 [-12.04, -2.55] | ahead | MWKR / MOR |
| pooled All | -7.03 [-9.09, -4.81] | ahead | -6.92 [-9.24, -4.30] | ahead | -7.06 [-8.62, -5.29] | ahead | -3.76 [-5.23, -2.23] | ahead |  |
| pooled Train | -5.03 [-6.71, -3.22] | ahead | -4.98 [-7.02, -2.73] | ahead | - | - | -2.58 [-3.99, -1.02] | ahead |  |
| pooled Held | -13.06 [-16.95, -9.08] | ahead | -12.73 [-16.41, -8.56] | ahead | - | - | -7.32 [-10.25, -4.60] | ahead |  |
| pooled Setup | -13.20 [-17.12, -9.12] | ahead | -12.64 [-16.49, -8.35] | ahead | -12.77 [-15.77, -9.39] | ahead | -7.44 [-10.09, -4.89] | ahead |  |
| pooled NoSetup | -0.86 [-2.45, +0.63] | neither | -1.20 [-2.91, +0.51] | neither | -1.34 [-2.57, -0.08] | ahead | -0.09 [-1.46, +1.24] | neither |  |

### pub_Hurink_edata (distribution transfer)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | +2.46 [+1.14, +3.74] | behind | +2.88 [+1.53, +4.32] | behind | +2.34 [+0.55, +4.14] | behind | +4.38 [+3.20, +5.54] | behind | MWKR / MWKR |
| L | -1.18 [-2.10, -0.24] | ahead | -0.43 [-1.45, +0.60] | neither | -0.06 [-1.24, +1.19] | neither | +0.26 [-0.62, +1.19] | neither | FIFO / MOR |
| S | -1.39 [-3.26, +0.54] | neither | -2.17 [-4.15, -0.03] | ahead | -2.51 [-4.30, -0.71] | ahead | +0.53 [-1.21, +2.26] | neither | FIFO / SPT |
| W | +2.53 [+1.23, +3.80] | behind | +1.36 [+0.30, +2.44] | behind | +1.11 [-0.02, +2.27] | neither | +3.51 [+2.53, +4.53] | behind | MWKR / MWKR |
| LS | -3.61 [-5.56, -1.69] | ahead | -4.26 [-6.11, -2.45] | ahead | -4.12 [-6.30, -2.00] | ahead | -1.52 [-3.16, +0.02] | neither | MOR / MOR |
| LW | -0.08 [-1.05, +0.89] | neither | +0.29 [-0.78, +1.41] | neither | +0.62 [-0.69, +2.11] | neither | +1.39 [+0.36, +2.53] | behind | MWKR / MWKR |
| SW | -2.03 [-3.85, -0.14] | ahead | -2.41 [-4.19, -0.56] | ahead | -1.71 [-3.26, -0.12] | ahead | -1.65 [-3.18, -0.11] | ahead | MWKR / MOR |
| LSW | -6.12 [-7.86, -4.40] | ahead | -6.33 [-8.31, -4.44] | ahead | -5.02 [-6.78, -3.32] | ahead | -2.34 [-4.02, -0.71] | ahead | MWKR / FIFO |
| pooled All | -1.18 [-1.97, -0.37] | ahead | -1.39 [-2.18, -0.54] | ahead | -1.17 [-2.02, -0.32] | ahead | +0.57 [-0.10, +1.28] | neither |  |
| pooled Train | -0.21 [-0.92, +0.49] | neither | -0.39 [-1.13, +0.37] | neither | - | - | +1.43 [+0.79, +2.09] | behind |  |
| pooled Held | -4.08 [-5.46, -2.64] | ahead | -4.37 [-5.84, -2.87] | ahead | - | - | -2.00 [-3.27, -0.71] | ahead |  |
| pooled Setup | -3.29 [-4.59, -1.93] | ahead | -3.79 [-5.09, -2.45] | ahead | -3.34 [-4.56, -2.10] | ahead | -1.25 [-2.31, -0.16] | ahead |  |
| pooled NoSetup | +0.93 [+0.33, +1.50] | behind | +1.02 [+0.41, +1.65] | behind | +1.00 [+0.10, +1.88] | behind | +2.39 [+1.80, +2.99] | behind |  |

### pub_Hurink_rdata (distribution transfer)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | +4.43 [+2.64, +6.54] | behind | +4.58 [+2.91, +6.37] | behind | +2.40 [+1.37, +3.45] | behind | +3.85 [+2.84, +4.87] | behind | FIFO / MWKR |
| L | -0.81 [-1.65, +0.04] | neither | -0.59 [-1.57, +0.41] | neither | -0.73 [-1.92, +0.62] | neither | +0.17 [-0.73, +1.12] | neither | MOR / MOR |
| S | -5.82 [-7.64, -3.85] | ahead | -5.05 [-6.80, -3.21] | ahead | -5.44 [-7.06, -3.76] | ahead | -1.98 [-3.35, -0.62] | ahead | MWKR / MWKR |
| W | +3.17 [+1.28, +5.31] | behind | +2.89 [+0.97, +4.93] | behind | +2.53 [+1.24, +3.85] | behind | +2.30 [+1.24, +3.38] | behind | FIFO / MWKR |
| LS | -6.91 [-9.01, -4.82] | ahead | -6.34 [-8.55, -4.16] | ahead | -6.68 [-8.81, -4.56] | ahead | -1.46 [-3.12, +0.07] | neither | MWKR / MWKR |
| LW | -0.39 [-1.42, +0.68] | neither | -0.72 [-1.76, +0.28] | neither | +0.14 [-1.58, +1.92] | neither | +0.12 [-0.94, +1.12] | neither | MWKR / MWKR |
| SW | -4.44 [-6.31, -2.56] | ahead | -4.71 [-6.55, -2.90] | ahead | -4.36 [-6.39, -2.04] | ahead | -1.95 [-3.52, -0.43] | ahead | MWKR / MWKR |
| LSW | -7.82 [-9.97, -5.75] | ahead | -8.13 [-10.15, -6.12] | ahead | -7.80 [-9.88, -5.77] | ahead | -2.94 [-4.64, -1.35] | ahead | MOR / MOR |
| pooled All | -2.32 [-3.18, -1.30] | ahead | -2.26 [-3.19, -1.28] | ahead | -2.49 [-3.21, -1.79] | ahead | -0.24 [-0.84, +0.38] | neither |  |
| pooled Train | -1.05 [-1.95, +0.03] | neither | -0.87 [-1.84, +0.17] | neither | - | - | +0.50 [-0.08, +1.08] | neither |  |
| pooled Held | -6.13 [-7.55, -4.68] | ahead | -6.42 [-7.84, -4.97] | ahead | - | - | -2.45 [-3.60, -1.36] | ahead |  |
| pooled Setup | -6.25 [-7.42, -5.02] | ahead | -6.06 [-7.32, -4.76] | ahead | -6.07 [-7.27, -4.89] | ahead | -2.08 [-3.03, -1.19] | ahead |  |
| pooled NoSetup | +1.60 [+0.60, +2.84] | behind | +1.54 [+0.61, +2.55] | behind | +1.08 [+0.49, +1.70] | behind | +1.61 [+1.07, +2.17] | behind |  |

### pub_Hurink_vdata (distribution transfer)

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | +6.67 [+4.55, +8.98] | behind | +4.95 [+3.12, +6.89] | behind | +3.80 [+2.27, +5.41] | behind | +2.86 [+2.05, +3.74] | behind | MWKR / MWKR |
| L | +0.03 [-0.73, +0.77] | neither | +0.28 [-0.51, +1.06] | neither | +0.36 [-0.37, +1.08] | neither | +0.95 [+0.32, +1.69] | behind | MOR / MOR |
| S | -1.37 [-3.67, +0.86] | neither | -2.37 [-4.14, -0.61] | ahead | -2.20 [-4.32, -0.03] | ahead | +0.65 [-0.90, +2.13] | neither | MWKR / MWKR |
| W | +5.63 [+3.18, +8.34] | behind | +4.65 [+2.22, +7.77] | behind | +4.51 [+3.04, +6.06] | behind | +2.66 [+1.51, +3.82] | behind | MWKR / MWKR |
| LS | -2.98 [-4.99, -1.07] | ahead | -3.33 [-5.16, -1.60] | ahead | -3.15 [-5.18, -1.23] | ahead | +1.70 [+0.32, +2.98] | behind | MOR / MWKR |
| LW | +0.70 [-0.04, +1.42] | neither | +0.91 [-0.24, +1.99] | neither | +1.61 [+0.51, +2.78] | behind | +2.21 [+1.34, +3.22] | behind | MOR / MOR |
| SW | -1.87 [-3.89, +0.17] | neither | -3.59 [-5.53, -1.56] | ahead | -1.46 [-5.04, +3.28] | neither | -0.65 [-1.94, +0.62] | neither | MWKR / MWKR |
| LSW | -4.57 [-6.67, -2.56] | ahead | -4.37 [-6.46, -2.34] | ahead | -4.32 [-6.14, -2.59] | ahead | -0.07 [-1.41, +1.17] | neither | MOR / MOR |
| pooled All | +0.28 [-0.82, +1.46] | neither | -0.36 [-1.31, +0.67] | neither | -0.11 [-0.91, +0.70] | neither | +1.29 [+0.77, +1.81] | behind |  |
| pooled Train | +1.45 [+0.34, +2.67] | behind | +0.85 [-0.08, +1.91] | neither | - | - | +1.84 [+1.33, +2.36] | behind |  |
| pooled Held | -3.22 [-4.60, -1.82] | ahead | -3.98 [-5.32, -2.68] | ahead | - | - | -0.36 [-1.38, +0.62] | neither |  |
| pooled Setup | -2.70 [-4.01, -1.41] | ahead | -3.42 [-4.56, -2.30] | ahead | -2.78 [-4.07, -1.42] | ahead | +0.40 [-0.51, +1.28] | neither |  |
| pooled NoSetup | +3.26 [+2.11, +4.51] | behind | +2.70 [+1.66, +3.94] | behind | +2.57 [+1.87, +3.31] | behind | +2.17 [+1.73, +2.62] | behind |  |

### pub_SD2_test_10x5+mix_lag050 (sensitivity of pub_SD2_test_10x5+mix ({'lag_scale': 0.5}))

| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | lowest-mean rule (active / non-delay) |
|---|---|---|---|---|---|---|---|---|---|
| N | -4.07 [-5.52, -2.65] | ahead | -3.21 [-4.63, -1.80] | ahead | -3.27 [-4.57, -1.99] | ahead | -4.70 [-6.22, -3.09] | ahead | MWKR / SPT |
| L | -7.17 [-8.27, -6.07] | ahead | -6.90 [-8.00, -5.81] | ahead | -6.62 [-7.96, -5.32] | ahead | -3.37 [-4.84, -1.89] | ahead | MOR / SPT |
| S | -20.67 [-22.48, -18.89] | ahead | -20.54 [-22.21, -18.86] | ahead | -21.15 [-22.55, -19.77] | ahead | -11.01 [-12.66, -9.40] | ahead | MWKR / MOR |
| W | -1.95 [-3.15, -0.73] | ahead | -1.74 [-3.45, +0.12] | neither | -2.46 [-3.57, -1.41] | ahead | -3.76 [-5.13, -2.23] | ahead | MWKR / MWKR |
| LS | -20.52 [-22.22, -18.78] | ahead | -20.51 [-22.26, -18.66] | ahead | -20.07 [-21.92, -18.14] | ahead | -11.19 [-13.00, -9.41] | ahead | MWKR / MWKR |
| LW | -4.40 [-5.74, -3.14] | ahead | -4.99 [-6.20, -3.82] | ahead | -2.60 [-4.87, -0.05] | ahead | -3.18 [-4.38, -1.99] | ahead | MWKR / MWKR |
| SW | -17.77 [-19.31, -16.19] | ahead | -18.18 [-19.60, -16.76] | ahead | -17.17 [-18.75, -15.65] | ahead | -9.82 [-11.26, -8.39] | ahead | MWKR / FIFO |
| LSW | -18.37 [-20.06, -16.58] | ahead | -18.00 [-19.71, -16.23] | ahead | -18.02 [-19.66, -16.30] | ahead | -9.42 [-10.98, -7.87] | ahead | MWKR / MOR |
| pooled All | -11.87 [-12.59, -11.14] | ahead | -11.76 [-12.51, -11.00] | ahead | -11.42 [-12.18, -10.67] | ahead | -7.05 [-7.77, -6.36] | ahead |  |
| pooled Train | -9.80 [-10.48, -9.11] | ahead | -9.65 [-10.38, -8.89] | ahead | - | - | -6.20 [-6.92, -5.49] | ahead |  |
| pooled Held | -18.07 [-19.36, -16.74] | ahead | -18.09 [-19.34, -16.82] | ahead | - | - | -9.62 [-10.76, -8.49] | ahead |  |
| pooled Setup | -19.33 [-20.52, -18.10] | ahead | -19.31 [-20.44, -18.14] | ahead | -19.10 [-20.23, -18.00] | ahead | -10.36 [-11.47, -9.28] | ahead |  |
| pooled NoSetup | -4.40 [-5.06, -3.71] | ahead | -4.21 [-5.01, -3.36] | ahead | -3.74 [-4.66, -2.86] | ahead | -3.75 [-4.51, -2.97] | ahead |  |

