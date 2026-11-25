# Eight independent runs of the memetic algorithm read as one method (full)

- environment exact, training seeds 301,302,303,304,305; selected training run s304; CP-SAT configuration lns; B = 20000
- clock origin request: time since the request for a schedule, a cold start for every method: the dispatching rules from t_all of the rule record (all four rule schedules exist), the policy's greedy schedule from its t in the selected run's policy record (process start, checkpoint load included), the policy, sampled schedules, at its own t (best so far; after the record's budget the last value is held, which favours the other method); a search started from the rules at run time t - t_all of the start clock, a search started from the policy at run time t - its greedy t; no schedule before the offset; every training run takes the offsets of the selected run's records
- best of eight: the shortest incumbent of seeds 1, 2, 3, 4, 5, 6, 7, 8 at the same time, cells with at least 8 valid runs; instances 3-11
- differences in %: first-named minus second over second, negative = first-named shorter; instance-level 95% interval (p94.compare_at); (held): a value of the sampled schedules held after the policy record's budget

## 50x25n30

Cells used (at least 8 of 8 runs valid): 72 of 72; cells by number of valid runs: 8: 72

Core sets of the valid records per seed: seed 1: 16-23 72; seed 2: 16-23 72; seed 3: 16-23 72; seed 4: 16-23 72; seed 5: 16-23 72; seed 6: 8-15 72; seed 7: 8-15 72; seed 8: 8-15 72

### the policy's greedy schedule minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +4.17 [+1.20, +7.05] behind | +10.60 [+7.88, +13.20] behind | +11.81 [+9.12, +14.46] behind |
| L | +1.69 [+0.43, +2.95] behind | +7.74 [+6.52, +9.01] behind | +9.24 [+8.12, +10.48] behind |
| S | -1.00 [-2.62, +0.54] neither | +4.41 [+3.36, +5.56] behind | +5.10 [+4.06, +6.27] behind |
| W | -0.33 [-2.66, +2.15] neither | +2.82 [+0.13, +5.81] behind | +4.07 [+1.49, +7.08] behind |
| LS | +0.46 [-0.81, +1.81] neither | +5.98 [+4.20, +7.81] behind | +6.63 [+4.82, +8.46] behind |
| LW | -3.16 [-5.07, -1.38] ahead | +1.50 [-1.07, +4.00] neither | +3.35 [+0.94, +5.70] behind |
| SW | -0.76 [-2.10, +0.62] neither | +2.81 [+1.50, +4.43] behind | +4.05 [+2.89, +5.56] behind |
| LSW | -2.51 [-3.91, -1.23] ahead | +3.10 [+2.02, +4.31] behind | +4.21 [+3.21, +5.44] behind |
| Pool | -0.18 [-0.98, +0.65] neither | +4.87 [+3.95, +5.76] behind | +6.06 [+5.22, +6.88] behind |
| combinations without setups | +0.59 [-0.84, +1.98] neither | +5.67 [+3.98, +7.21] behind | +7.12 [+5.58, +8.52] behind |
| combinations with setups | -0.95 [-1.99, +0.07] neither | +4.08 [+3.13, +5.09] behind | +5.00 [+4.06, +6.02] behind |

### the policy, sampled schedules, minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.37 [-2.45, -0.26] ahead | +4.24 [+2.81, +5.78] behind (held) | +5.37 [+4.22, +6.64] behind (held) |
| L | +0.68 [-0.42, +1.82] neither | +6.20 [+5.06, +7.43] behind (held) | +7.68 [+6.62, +8.87] behind (held) |
| S | -2.59 [-3.79, -1.42] ahead | +2.53 [+1.75, +3.31] behind (held) | +3.21 [+2.43, +3.97] behind (held) |
| W | -3.57 [-4.87, -2.24] ahead | -0.68 [-2.23, +0.86] neither (held) | +0.52 [-1.00, +2.09] neither (held) |
| LS | -1.76 [-3.12, -0.44] ahead | +3.35 [+2.02, +4.57] behind (held) | +3.98 [+2.70, +5.10] behind (held) |
| LW | -4.71 [-6.19, -3.30] ahead | -0.20 [-2.25, +1.87] neither (held) | +1.61 [-0.28, +3.55] neither (held) |
| SW | -3.33 [-4.37, -2.37] ahead | -0.07 [-0.96, +0.77] neither (held) | +1.14 [+0.38, +1.86] behind (held) |
| LSW | -4.13 [-5.32, -2.96] ahead | +1.03 [+0.12, +2.02] behind (held) | +2.12 [+1.29, +3.10] behind (held) |
| Pool | -2.60 [-3.19, -1.89] ahead | +2.05 [+1.37, +2.82] behind (held) | +3.21 [+2.59, +3.89] behind (held) |
| combinations without setups | -2.24 [-3.19, -1.31] ahead | +2.39 [+1.07, +3.74] behind (held) | +3.80 [+2.60, +5.04] behind (held) |
| combinations with setups | -2.95 [-3.89, -2.02] ahead | +1.71 [+1.01, +2.40] behind (held) | +2.61 [+1.94, +3.28] behind (held) |

### CP-SAT (8 workers) started from the rule schedule minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.10 [-0.77, +0.93] neither | +0.19 [-0.70, +1.11] neither | +0.21 [-0.28, +0.72] neither |
| L | +1.98 [+0.88, +2.92] behind | +0.39 [-0.25, +1.09] neither | +0.23 [-0.28, +0.76] neither |
| S | +4.67 [+3.65, +5.66] behind | +6.66 [+5.15, +8.39] behind | +6.35 [+5.09, +7.82] behind |
| W | -2.92 [-3.73, -1.90] ahead | -2.61 [-3.73, -1.53] ahead | -2.54 [-3.99, -1.20] ahead |
| LS | +4.74 [+3.67, +5.95] behind | +8.10 [+5.84, +10.52] behind | +7.14 [+5.18, +9.27] behind |
| LW | -2.49 [-3.47, -1.39] ahead | -2.68 [-4.10, -1.24] ahead | -2.69 [-4.00, -1.40] ahead |
| SW | +2.12 [+0.95, +3.73] behind | +3.44 [+2.93, +3.91] behind | +3.94 [+3.29, +4.68] behind |
| LSW | +1.68 [+0.65, +3.12] behind | +5.95 [+4.66, +7.31] behind | +6.53 [+5.07, +8.06] behind |
| Pool | +1.24 [+0.92, +1.54] behind | +2.43 [+1.63, +3.18] behind | +2.40 [+1.62, +3.15] behind |
| combinations without setups | -0.83 [-1.12, -0.52] ahead | -1.18 [-1.89, -0.49] ahead | -1.20 [-1.87, -0.56] ahead |
| combinations with setups | +3.30 [+2.75, +3.84] behind | +6.04 [+4.88, +7.21] behind | +5.99 [+4.82, +7.18] behind |

### the best of the eight memetic runs minus the run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.03 [-2.94, -1.19] ahead | -1.01 [-1.45, -0.58] ahead | -0.78 [-1.01, -0.55] ahead |
| L | -2.36 [-3.68, -1.14] ahead | -1.12 [-1.95, -0.42] ahead | -1.01 [-1.53, -0.51] ahead |
| S | -1.25 [-1.81, -0.71] ahead | -1.25 [-1.81, -0.78] ahead | -0.90 [-1.38, -0.46] ahead |
| W | -0.93 [-1.53, -0.39] ahead | -1.02 [-1.89, -0.32] ahead | -1.03 [-2.12, -0.27] ahead |
| LS | -0.74 [-1.36, -0.21] ahead | -0.77 [-1.24, -0.34] ahead | -0.67 [-0.94, -0.43] ahead |
| LW | -0.70 [-0.97, -0.43] ahead | -1.01 [-1.79, -0.37] ahead | -1.42 [-2.07, -0.79] ahead |
| SW | -1.06 [-1.49, -0.65] ahead | -0.72 [-1.18, -0.30] ahead | -1.05 [-1.54, -0.62] ahead |
| LSW | -0.58 [-0.89, -0.32] ahead | -0.71 [-1.05, -0.41] ahead | -0.56 [-0.93, -0.22] ahead |
| Pool | -1.21 [-1.56, -0.85] ahead | -0.95 [-1.14, -0.78] ahead | -0.93 [-1.12, -0.74] ahead |
| combinations without setups | -1.50 [-2.12, -0.89] ahead | -1.04 [-1.36, -0.73] ahead | -1.06 [-1.30, -0.81] ahead |
| combinations with setups | -0.91 [-1.09, -0.69] ahead | -0.87 [-1.03, -0.71] ahead | -0.80 [-1.01, -0.62] ahead |

### the policy's greedy schedule minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +2.09 [-1.33, +5.53] neither | +9.48 [+6.86, +11.94] behind | +10.93 [+8.32, +13.53] behind |
| L | -0.73 [-2.30, +0.80] neither | +6.53 [+5.09, +7.83] behind | +8.14 [+6.93, +9.31] behind |
| S | -2.24 [-3.92, -0.62] ahead | +3.10 [+1.86, +4.41] behind | +4.16 [+2.94, +5.40] behind |
| W | -1.25 [-3.73, +1.39] neither | +1.76 [-0.87, +4.29] neither | +2.96 [+0.67, +5.17] behind |
| LS | -0.29 [-1.47, +0.82] neither | +5.15 [+3.62, +6.78] behind | +5.91 [+4.20, +7.70] behind |
| LW | -3.85 [-5.56, -2.28] ahead | +0.46 [-1.76, +2.56] neither | +1.88 [-0.42, +4.14] neither |
| SW | -1.81 [-3.20, -0.32] ahead | +2.07 [+0.54, +3.77] behind | +2.96 [+1.51, +4.60] behind |
| LSW | -3.08 [-4.48, -1.73] ahead | +2.36 [+1.40, +3.37] behind | +3.63 [+2.75, +4.58] behind |
| Pool | -1.39 [-2.35, -0.39] ahead | +3.86 [+2.95, +4.75] behind | +5.07 [+4.22, +5.90] behind |
| combinations without setups | -0.93 [-2.57, +0.71] neither | +4.56 [+2.94, +5.95] behind | +5.98 [+4.48, +7.28] behind |
| combinations with setups | -1.85 [-3.02, -0.70] ahead | +3.17 [+2.17, +4.17] behind | +4.16 [+3.12, +5.21] behind |

### the policy, sampled schedules, minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.38 [-4.45, -2.26] ahead | +3.18 [+1.97, +4.39] behind (held) | +4.55 [+3.40, +5.73] behind (held) |
| L | -1.71 [-2.99, -0.54] ahead | +5.01 [+3.70, +6.28] behind (held) | +6.59 [+5.45, +7.80] behind (held) |
| S | -3.81 [-5.14, -2.66] ahead | +1.25 [+0.45, +2.22] behind (held) | +2.28 [+1.47, +3.21] behind (held) |
| W | -4.47 [-5.78, -3.10] ahead | -1.68 [-3.58, +0.26] neither (held) | -0.51 [-2.30, +1.43] neither (held) |
| LS | -2.49 [-3.95, -0.92] ahead | +2.55 [+1.41, +3.57] behind (held) | +3.28 [+2.08, +4.32] behind (held) |
| LW | -5.38 [-6.67, -4.16] ahead | -1.23 [-2.94, +0.42] neither (held) | +0.17 [-1.65, +2.02] neither (held) |
| SW | -4.35 [-5.41, -3.41] ahead | -0.78 [-1.94, +0.28] neither (held) | +0.08 [-0.96, +1.07] neither (held) |
| LSW | -4.69 [-5.88, -3.47] ahead | +0.30 [-0.50, +1.10] neither (held) | +1.55 [+0.74, +2.36] behind (held) |
| Pool | -3.79 [-4.43, -3.05] ahead | +1.07 [+0.44, +1.80] behind (held) | +2.25 [+1.64, +2.98] behind (held) |
| combinations without setups | -3.74 [-4.66, -2.88] ahead | +1.32 [+0.15, +2.51] behind (held) | +2.70 [+1.54, +3.93] behind (held) |
| combinations with setups | -3.84 [-4.90, -2.76] ahead | +0.83 [+0.06, +1.54] behind (held) | +1.80 [+0.99, +2.52] behind (held) |

### CP-SAT (8 workers) started from the rule schedule minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.94 [-2.83, -1.12] ahead | -0.83 [-1.49, -0.19] ahead | -0.58 [-1.11, -0.04] ahead |
| L | -0.45 [-1.66, +0.74] neither | -0.74 [-1.73, +0.30] neither | -0.78 [-1.41, -0.03] ahead |
| S | +3.36 [+2.35, +4.35] behind | +5.31 [+4.00, +6.68] behind | +5.39 [+4.35, +6.46] behind |
| W | -3.82 [-4.78, -2.84] ahead | -3.59 [-5.07, -2.09] ahead | -3.54 [-5.11, -1.96] ahead |
| LS | +3.95 [+3.09, +4.94] behind | +7.24 [+5.31, +9.31] behind | +6.41 [+4.65, +8.36] behind |
| LW | -3.17 [-4.09, -2.18] ahead | -3.67 [-4.91, -2.37] ahead | -4.07 [-5.46, -2.62] ahead |
| SW | +1.03 [-0.00, +2.37] neither | +2.69 [+2.04, +3.31] behind | +2.83 [+2.40, +3.25] behind |
| LSW | +1.08 [+0.22, +2.26] behind | +5.19 [+3.94, +6.51] behind | +5.93 [+4.50, +7.42] behind |
| Pool | +0.01 [-0.27, +0.27] neither | +1.45 [+0.74, +2.14] behind | +1.45 [+0.78, +2.11] behind |
| combinations without setups | -2.34 [-2.73, -1.93] ahead | -2.21 [-2.84, -1.61] ahead | -2.24 [-2.90, -1.61] ahead |
| combinations with setups | +2.35 [+1.98, +2.75] behind | +5.11 [+4.07, +6.19] behind | +5.14 [+4.11, +6.21] behind |

### Verdict at 600 s: the policy's greedy schedule minus the best of the eight memetic runs

| combination | difference | 95% interval | Holm p shorter | Holm p longer | verdict |
|---|---|---|---|---|---|
| N | +11.81 | [+9.12, +14.46] | 1 | 0.0004 | behind |
| L | +9.24 | [+8.12, +10.48] | 1 | 0.0004 | behind |
| S | +5.10 | [+4.06, +6.27] | 1 | 0.0004 | behind |
| W | +4.07 | [+1.49, +7.08] | 1 | 0.0006 | behind |
| LS | +6.63 | [+4.82, +8.46] | 1 | 0.0004 | behind |
| LW | +3.35 | [+0.94, +5.70] | 1 | 0.0034 | behind |
| SW | +4.05 | [+2.89, +5.56] | 1 | 0.0004 | behind |
| LSW | +4.21 | [+3.21, +5.44] | 1 | 0.0004 | behind |

ahead 0, behind 8, not distinguishable 0

### Verdict at 600 s: the policy, sampled schedules, minus the best of the eight memetic runs

| combination | difference | 95% interval | Holm p shorter | Holm p longer | verdict |
|---|---|---|---|---|---|
| N | +5.37 | [+4.22, +6.64] | 1 | 0.0004 | behind |
| L | +7.68 | [+6.62, +8.87] | 1 | 0.0004 | behind |
| S | +3.21 | [+2.43, +3.97] | 1 | 0.0004 | behind |
| W | +0.52 | [-1.00, +2.09] | 1 | 0.26 | not distinguishable |
| LS | +3.98 | [+2.70, +5.10] | 1 | 0.0004 | behind |
| LW | +1.61 | [-0.28, +3.55] | 1 | 0.094 | not distinguishable |
| SW | +1.14 | [+0.38, +1.86] | 1 | 0.0054 | behind |
| LSW | +2.12 | [+1.29, +3.10] | 1 | 0.0004 | behind |

ahead 0, behind 6, not distinguishable 2 (uses held values)

### Spread over the eight runs at 600 s

| entry | cells | mean of (largest - smallest) / smallest, % | largest, % |
|---|---|---|---|
| N | 9 | 1.48 | 2.42 |
| L | 9 | 2.63 | 4.00 |
| S | 9 | 1.21 | 2.33 |
| W | 9 | 2.35 | 5.74 |
| LS | 9 | 1.73 | 4.41 |
| LW | 9 | 2.74 | 6.09 |
| SW | 9 | 1.80 | 3.93 |
| LSW | 9 | 2.24 | 3.83 |
| Pool | 72 | 2.02 | 6.09 |
| combinations without setups | 36 | 2.30 | 6.09 |
| combinations with setups | 36 | 1.75 | 4.41 |

### Mean makespans (cells used)

| budget | combination | cells | best of eight | seed 1 | policy greedy | policy sampled | CP-SAT |
|---|---|---|---|---|---|---|---|
| 60 | N | 9 | 346.78 | 354.00 | 361.22 | 342.00 | 347.11 |
| 60 | L | 9 | 420.89 | 431.22 | 427.89 | 423.67 | 429.22 |
| 60 | S | 9 | 462.22 | 468.11 | 457.11 | 449.89 | 483.89 |
| 60 | W | 9 | 379.22 | 382.78 | 377.78 | 365.44 | 368.00 |
| 60 | LS | 9 | 543.44 | 547.67 | 545.89 | 533.56 | 569.33 |
| 60 | LW | 9 | 463.78 | 467.00 | 448.78 | 441.67 | 452.11 |
| 60 | SW | 9 | 495.89 | 501.22 | 491.78 | 479.11 | 506.44 |
| 60 | LSW | 9 | 580.33 | 583.78 | 565.33 | 556.00 | 590.11 |
| 300 | N | 9 | 326.56 | 329.89 | 361.22 | 340.33 | 327.11 |
| 300 | L | 9 | 397.22 | 401.78 | 427.89 | 421.78 | 398.78 |
| 300 | S | 9 | 438.00 | 443.67 | 457.11 | 449.00 | 467.33 |
| 300 | W | 9 | 367.67 | 371.56 | 377.78 | 364.89 | 357.89 |
| 300 | LS | 9 | 515.00 | 519.00 | 545.89 | 532.22 | 556.78 |
| 300 | LW | 9 | 442.67 | 447.11 | 448.78 | 441.33 | 430.56 |
| 300 | SW | 9 | 478.67 | 482.22 | 491.78 | 478.11 | 495.22 |
| 300 | LSW | 9 | 548.56 | 552.44 | 565.33 | 554.00 | 581.33 |
| 600 | N | 9 | 323.00 | 325.56 | 361.22 | 340.33 | 323.67 |
| 600 | L | 9 | 391.78 | 395.78 | 427.89 | 421.78 | 392.67 |
| 600 | S | 9 | 435.11 | 439.11 | 457.11 | 449.00 | 462.89 |
| 600 | W | 9 | 363.22 | 367.11 | 377.78 | 364.89 | 353.78 |
| 600 | LS | 9 | 511.89 | 515.33 | 545.89 | 532.22 | 548.56 |
| 600 | LW | 9 | 434.67 | 440.89 | 448.78 | 441.33 | 422.78 |
| 600 | SW | 9 | 472.89 | 478.00 | 491.78 | 478.11 | 491.56 |
| 600 | LSW | 9 | 542.67 | 545.67 | 565.33 | 554.00 | 578.22 |

## 80x25n30

Cells used (at least 8 of 8 runs valid): 72 of 72; cells by number of valid runs: 8: 72

Core sets of the valid records per seed: seed 1: 16-23 72; seed 2: 16-23 72; seed 3: 16-23 72; seed 4: 16-23 72; seed 5: 16-23 72; seed 6: 8-15 72; seed 7: 8-15 72; seed 8: 8-15 72

### the policy's greedy schedule minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.08 [-3.17, +1.37] neither | +5.74 [+3.88, +7.92] behind | +7.20 [+5.27, +9.44] behind |
| L | +1.19 [-2.44, +5.73] neither | +9.97 [+7.39, +13.07] behind | +11.51 [+8.78, +14.71] behind |
| S | -1.72 [-3.37, -0.17] ahead | +1.91 [+0.56, +3.19] behind | +2.78 [+1.48, +4.02] behind |
| W | -4.37 [-5.71, -3.04] ahead | -2.81 [-4.29, -1.51] ahead | -1.28 [-2.37, -0.19] ahead |
| LS | +1.10 [-1.17, +4.01] neither | +4.66 [+2.33, +7.47] behind | +5.87 [+3.65, +8.49] behind |
| LW | -1.70 [-3.34, +0.73] neither | +0.72 [-0.99, +3.06] neither | +2.18 [+0.63, +4.33] behind |
| SW | -3.89 [-6.58, -1.46] ahead | -1.20 [-3.59, +1.08] neither | -0.05 [-1.85, +1.88] neither |
| LSW | -2.75 [-4.73, -0.25] ahead | +0.81 [-1.29, +3.74] neither | +2.07 [+0.11, +4.84] behind |
| Pool | -1.65 [-2.89, -0.43] ahead | +2.47 [+1.77, +3.23] behind | +3.78 [+3.17, +4.43] behind |
| combinations without setups | -1.49 [-3.07, +0.45] neither | +3.40 [+2.54, +4.58] behind | +4.90 [+4.13, +5.96] behind |
| combinations with setups | -1.82 [-3.18, -0.56] ahead | +1.54 [+0.50, +2.69] behind | +2.67 [+1.80, +3.69] behind |

### the policy, sampled schedules, minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.56 [-4.60, -2.46] ahead | +2.72 [+1.51, +3.77] behind (held) | +4.14 [+2.87, +5.25] behind (held) |
| L | -3.76 [-4.49, -3.14] ahead | +4.58 [+2.49, +6.34] behind (held) | +6.03 [+3.99, +7.76] behind (held) |
| S | -2.96 [-4.17, -1.91] ahead | +0.12 [-0.67, +0.87] neither (held) | +0.99 [+0.25, +1.71] behind (held) |
| W | -5.46 [-6.76, -4.19] ahead | -4.44 [-5.82, -3.15] ahead (held) | -2.93 [-4.03, -1.71] ahead (held) |
| LS | -2.23 [-3.57, -1.01] ahead | +0.92 [-0.40, +2.38] neither (held) | +2.09 [+0.90, +3.42] behind (held) |
| LW | -3.90 [-4.55, -3.25] ahead | -2.06 [-2.85, -1.14] ahead (held) | -0.62 [-1.60, +0.36] neither (held) |
| SW | -5.99 [-8.08, -4.21] ahead | -3.56 [-5.37, -1.95] ahead (held) | -2.44 [-3.56, -1.27] ahead (held) |
| LSW | -5.31 [-6.01, -4.46] ahead | -2.31 [-2.92, -1.75] ahead (held) | -1.08 [-1.61, -0.62] ahead (held) |
| Pool | -4.15 [-4.79, -3.52] ahead | -0.50 [-0.98, +0.02] neither (held) | +0.77 [+0.17, +1.35] behind (held) |
| combinations without setups | -4.17 [-4.83, -3.45] ahead | +0.20 [-0.77, +1.07] neither (held) | +1.65 [+0.51, +2.63] behind (held) |
| combinations with setups | -4.13 [-4.98, -3.41] ahead | -1.21 [-1.78, -0.67] ahead (held) | -0.11 [-0.54, +0.32] neither (held) |

### CP-SAT (8 workers) started from the rule schedule minus the best of the eight memetic runs

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.45 [+0.03, +0.86] behind | +0.51 [+0.05, +0.99] behind | +0.21 [-0.27, +0.77] neither |
| L | -0.36 [-0.89, +0.13] neither | +2.05 [+0.57, +3.56] behind | +0.54 [-0.35, +1.36] neither |
| S | +4.11 [+3.00, +5.18] behind | +7.09 [+5.96, +8.49] behind | +7.27 [+6.01, +8.82] behind |
| W | -1.84 [-2.29, -1.40] ahead | -4.71 [-6.07, -3.61] ahead | -4.24 [-5.05, -3.37] ahead |
| LS | +3.59 [+2.87, +4.41] behind | +6.60 [+5.40, +7.97] behind | +7.04 [+5.42, +8.79] behind |
| LW | -0.92 [-1.21, -0.65] ahead | -3.64 [-4.54, -2.75] ahead | -4.17 [-5.07, -3.29] ahead |
| SW | +0.57 [+0.31, +0.85] behind | +2.56 [+2.09, +3.07] behind | +2.92 [+1.91, +4.17] behind |
| LSW | +0.44 [+0.18, +0.75] behind | +3.79 [+2.87, +4.61] behind | +4.72 [+3.76, +5.56] behind |
| Pool | +0.75 [+0.50, +1.01] behind | +1.78 [+1.31, +2.33] behind | +1.79 [+1.24, +2.41] behind |
| combinations without setups | -0.67 [-0.93, -0.40] ahead | -1.45 [-2.01, -0.96] ahead | -1.91 [-2.28, -1.55] ahead |
| combinations with setups | +2.18 [+1.86, +2.52] behind | +5.01 [+4.36, +5.83] behind | +5.49 [+4.57, +6.63] behind |

### the best of the eight memetic runs minus the run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -1.42 [-2.10, -0.78] ahead | -0.93 [-1.29, -0.60] ahead | -0.90 [-1.28, -0.56] ahead |
| L | -0.53 [-0.98, -0.22] ahead | -2.59 [-3.85, -1.41] ahead | -1.96 [-3.05, -0.92] ahead |
| S | -0.62 [-1.16, -0.19] ahead | -0.36 [-0.64, -0.12] ahead | -0.50 [-0.71, -0.30] ahead |
| W | -0.16 [-0.34, +0.00] neither | -1.16 [-1.63, -0.73] ahead | -1.46 [-2.23, -0.76] ahead |
| LS | -0.63 [-1.00, -0.30] ahead | -0.57 [-0.81, -0.34] ahead | -0.57 [-0.84, -0.32] ahead |
| LW | -0.09 [-0.15, -0.02] ahead | -1.44 [-2.03, -0.87] ahead | -1.22 [-1.87, -0.64] ahead |
| SW | -0.38 [-0.60, -0.17] ahead | -1.12 [-1.77, -0.56] ahead | -1.32 [-2.30, -0.50] ahead |
| LSW | -0.31 [-0.47, -0.16] ahead | -0.59 [-1.05, -0.21] ahead | -0.88 [-1.32, -0.59] ahead |
| Pool | -0.52 [-0.64, -0.39] ahead | -1.09 [-1.36, -0.83] ahead | -1.10 [-1.39, -0.82] ahead |
| combinations without setups | -0.55 [-0.74, -0.35] ahead | -1.53 [-1.92, -1.11] ahead | -1.38 [-1.81, -0.97] ahead |
| combinations with setups | -0.49 [-0.66, -0.32] ahead | -0.66 [-0.84, -0.50] ahead | -0.82 [-1.11, -0.58] ahead |

### the policy's greedy schedule minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.49 [-4.67, -0.06] ahead | +4.76 [+2.79, +7.09] behind | +6.24 [+4.21, +8.63] behind |
| L | +0.67 [-3.03, +5.40] neither | +7.19 [+3.60, +11.28] behind | +9.38 [+5.79, +13.48] behind |
| S | -2.34 [-4.08, -0.68] ahead | +1.54 [+0.09, +2.90] behind | +2.27 [+0.81, +3.63] behind |
| W | -4.52 [-5.81, -3.23] ahead | -3.94 [-5.38, -2.54] ahead | -2.72 [-3.80, -1.52] ahead |
| LS | +0.46 [-1.92, +3.53] neither | +4.07 [+1.72, +6.91] behind | +5.27 [+3.09, +7.89] behind |
| LW | -1.78 [-3.42, +0.63] neither | -0.73 [-2.45, +1.65] neither | +0.93 [-0.90, +3.15] neither |
| SW | -4.27 [-6.81, -1.94] ahead | -2.28 [-5.16, +0.41] neither | -1.34 [-3.91, +1.20] neither |
| LSW | -3.05 [-5.09, -0.48] ahead | +0.19 [-1.66, +2.68] neither | +1.15 [-0.59, +3.46] neither |
| Pool | -2.16 [-3.41, -0.90] ahead | +1.35 [+0.48, +2.23] behind | +2.65 [+1.91, +3.40] behind |
| combinations without setups | -2.03 [-3.74, +0.04] neither | +1.82 [+0.74, +3.21] behind | +3.46 [+2.43, +4.71] behind |
| combinations with setups | -2.30 [-3.66, -0.95] ahead | +0.88 [-0.20, +1.96] neither | +1.84 [+0.89, +2.79] behind |

### the policy, sampled schedules, minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -4.92 [-6.31, -3.58] ahead | +1.77 [+0.36, +3.02] behind (held) | +3.21 [+1.73, +4.52] behind (held) |
| L | -4.26 [-4.99, -3.63] ahead | +1.84 [+0.45, +3.31] behind (held) | +3.93 [+2.50, +5.38] behind (held) |
| S | -3.57 [-4.96, -2.31] ahead | -0.23 [-1.14, +0.59] neither (held) | +0.48 [-0.42, +1.30] neither (held) |
| W | -5.61 [-6.90, -4.31] ahead | -5.55 [-6.80, -4.32] ahead (held) | -4.35 [-5.34, -3.15] ahead (held) |
| LS | -2.85 [-4.15, -1.56] ahead | +0.34 [-0.85, +1.77] neither (held) | +1.50 [+0.37, +2.80] behind (held) |
| LW | -3.98 [-4.64, -3.32] ahead | -3.47 [-4.08, -2.86] ahead (held) | -1.84 [-2.95, -0.61] ahead (held) |
| SW | -6.35 [-8.32, -4.67] ahead | -4.63 [-6.90, -2.76] ahead (held) | -3.71 [-5.60, -1.99] ahead (held) |
| LSW | -5.61 [-6.36, -4.67] ahead | -2.89 [-3.68, -2.13] ahead (held) | -1.95 [-2.83, -1.28] ahead (held) |
| Pool | -4.64 [-5.30, -4.00] ahead | -1.60 [-2.05, -1.11] ahead (held) | -0.34 [-0.87, +0.28] neither (held) |
| combinations without setups | -4.69 [-5.40, -3.92] ahead | -1.35 [-2.00, -0.75] ahead (held) | +0.24 [-0.66, +1.13] neither (held) |
| combinations with setups | -4.60 [-5.45, -3.90] ahead | -1.85 [-2.54, -1.22] ahead (held) | -0.92 [-1.51, -0.34] ahead (held) |

### CP-SAT (8 workers) started from the rule schedule minus the memetic run of seed 1

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -0.98 [-1.50, -0.42] ahead | -0.43 [-1.17, +0.32] neither | -0.69 [-1.40, +0.05] neither |
| L | -0.89 [-1.39, -0.42] ahead | -0.63 [-1.37, +0.18] neither | -1.44 [-2.25, -0.61] ahead |
| S | +3.46 [+2.10, +4.82] behind | +6.71 [+5.50, +8.15] behind | +6.73 [+5.47, +8.16] behind |
| W | -1.99 [-2.44, -1.55] ahead | -5.82 [-7.11, -4.70] ahead | -5.64 [-6.43, -4.71] ahead |
| LS | +2.94 [+2.26, +3.74] behind | +6.00 [+4.80, +7.33] behind | +6.43 [+4.79, +8.16] behind |
| LW | -1.01 [-1.27, -0.75] ahead | -5.04 [-5.74, -4.43] ahead | -5.34 [-6.46, -4.30] ahead |
| SW | +0.19 [+0.01, +0.38] behind | +1.41 [+0.72, +2.15] behind | +1.55 [+0.65, +2.39] behind |
| LSW | +0.13 [-0.04, +0.32] neither | +3.17 [+2.37, +4.01] behind | +3.79 [+2.85, +4.76] behind |
| Pool | +0.23 [-0.05, +0.49] neither | +0.67 [+0.37, +1.03] behind | +0.67 [+0.23, +1.14] behind |
| combinations without setups | -1.22 [-1.56, -0.93] ahead | -2.98 [-3.29, -2.67] ahead | -3.28 [-3.63, -2.88] ahead |
| combinations with setups | +1.68 [+1.30, +2.10] behind | +4.32 [+3.69, +5.06] behind | +4.62 [+3.82, +5.54] behind |

### Verdict at 600 s: the policy's greedy schedule minus the best of the eight memetic runs

| combination | difference | 95% interval | Holm p shorter | Holm p longer | verdict |
|---|---|---|---|---|---|
| N | +7.20 | [+5.27, +9.44] | 1 | 0.0004 | behind |
| L | +11.51 | [+8.78, +14.71] | 1 | 0.0004 | behind |
| S | +2.78 | [+1.48, +4.02] | 1 | 0.0005 | behind |
| W | -1.28 | [-2.37, -0.19] | 0.087 | 1 | not distinguishable |
| LS | +5.87 | [+3.65, +8.49] | 1 | 0.0004 | behind |
| LW | +2.18 | [+0.63, +4.33] | 1 | 0.0016 | behind |
| SW | -0.05 | [-1.85, +1.88] | 1 | 1 | not distinguishable |
| LSW | +2.07 | [+0.11, +4.84] | 1 | 0.044 | behind |

ahead 0, behind 6, not distinguishable 2

### Verdict at 600 s: the policy, sampled schedules, minus the best of the eight memetic runs

| combination | difference | 95% interval | Holm p shorter | Holm p longer | verdict |
|---|---|---|---|---|---|
| N | +4.14 | [+2.87, +5.25] | 1 | 0.0004 | behind |
| L | +6.03 | [+3.99, +7.76] | 1 | 0.0004 | behind |
| S | +0.99 | [+0.25, +1.71] | 1 | 0.023 | behind |
| W | -2.93 | [-4.03, -1.71] | 0.0004 | 1 | ahead |
| LS | +2.09 | [+0.90, +3.42] | 1 | 0.0004 | behind |
| LW | -0.62 | [-1.60, +0.36] | 0.52 | 1 | not distinguishable |
| SW | -2.44 | [-3.56, -1.27] | 0.0004 | 1 | ahead |
| LSW | -1.08 | [-1.61, -0.62] | 0.0004 | 1 | ahead |

ahead 3, behind 4, not distinguishable 1 (uses held values)

### Spread over the eight runs at 600 s

| entry | cells | mean of (largest - smallest) / smallest, % | largest, % |
|---|---|---|---|
| N | 9 | 2.01 | 3.40 |
| L | 9 | 3.40 | 5.62 |
| S | 9 | 1.18 | 1.92 |
| W | 9 | 2.42 | 4.81 |
| LS | 9 | 1.53 | 4.03 |
| LW | 9 | 2.53 | 4.49 |
| SW | 9 | 1.88 | 4.40 |
| LSW | 9 | 2.10 | 3.27 |
| Pool | 72 | 2.13 | 5.62 |
| combinations without setups | 36 | 2.59 | 5.62 |
| combinations with setups | 36 | 1.67 | 4.40 |

### Mean makespans (cells used)

| budget | combination | cells | best of eight | seed 1 | policy greedy | policy sampled | CP-SAT |
|---|---|---|---|---|---|---|---|
| 60 | N | 9 | 516.00 | 523.56 | 510.33 | 497.56 | 518.33 |
| 60 | L | 9 | 609.44 | 612.67 | 616.22 | 586.56 | 607.22 |
| 60 | S | 9 | 671.89 | 676.33 | 659.44 | 651.33 | 699.44 |
| 60 | W | 9 | 558.89 | 559.78 | 534.33 | 528.11 | 548.56 |
| 60 | LS | 9 | 754.33 | 759.11 | 762.11 | 736.78 | 781.44 |
| 60 | LW | 9 | 639.22 | 639.78 | 628.22 | 614.33 | 633.33 |
| 60 | SW | 9 | 727.00 | 729.78 | 698.22 | 682.56 | 731.11 |
| 60 | LSW | 9 | 806.00 | 808.56 | 783.78 | 763.11 | 809.56 |
| 300 | N | 9 | 482.56 | 487.11 | 510.33 | 495.67 | 485.00 |
| 300 | L | 9 | 560.11 | 575.11 | 616.22 | 585.56 | 571.44 |
| 300 | S | 9 | 647.67 | 650.11 | 659.44 | 648.11 | 693.78 |
| 300 | W | 9 | 550.00 | 556.44 | 534.33 | 525.33 | 523.78 |
| 300 | LS | 9 | 728.44 | 732.56 | 762.11 | 734.56 | 776.67 |
| 300 | LW | 9 | 623.89 | 633.00 | 628.22 | 611.00 | 601.00 |
| 300 | SW | 9 | 706.89 | 715.00 | 698.22 | 681.00 | 725.11 |
| 300 | LSW | 9 | 777.56 | 782.22 | 783.78 | 759.67 | 807.00 |
| 600 | N | 9 | 476.00 | 480.33 | 510.33 | 495.67 | 477.00 |
| 600 | L | 9 | 552.44 | 563.56 | 616.22 | 585.56 | 555.33 |
| 600 | S | 9 | 642.11 | 645.44 | 659.44 | 648.11 | 689.00 |
| 600 | W | 9 | 541.33 | 549.33 | 534.33 | 525.33 | 518.22 |
| 600 | LS | 9 | 719.78 | 723.89 | 762.11 | 734.56 | 770.78 |
| 600 | LW | 9 | 614.89 | 622.56 | 628.22 | 611.00 | 589.11 |
| 600 | SW | 9 | 698.44 | 708.00 | 698.22 | 681.00 | 719.11 |
| 600 | LSW | 9 | 767.89 | 774.78 | 783.78 | 759.67 | 804.11 |

## Second set: eight runs started from the policy's schedule

- seeds 1, 2, 3, 4, 5, 6, 7, 8 of memetic_policy@600, one thread each, started from the rule schedule and the greedy schedule of run s304; best of eight as above, cells with all eight runs valid; CP-SAT: cpsat_policy@600 (lns, 8 workers)
- request clock: a memetic run started from the policy's schedule at run time t - max(o_pol, o_rule), CP-SAT started from it at t - o_pol, the greedy schedule from o_pol on, the runs started from the rule schedule as above
- the last two tables are the form of the decision table: difference from the policy's greedy schedule, in percent of it

### 50x25n30, second set

Cells used (all 8 runs valid): 72 of 72; valid runs per seed: seed 1: 72 (8-15 72); seed 2: 72 (16-23 72); seed 3: 72 (16-23 72); seed 4: 72 (16-23 72); seed 5: 72 (16-23 72); seed 6: 72 (16-23 72); seed 7: 72 (8-15 72); seed 8: 72 (8-15 72)

#### the policy, sampled schedules, minus the best of the eight memetic runs started from the policy's schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.88 [-0.55, +2.84] neither | +4.44 [+3.50, +5.89] behind (held) | +5.40 [+4.44, +6.82] behind (held) |
| L | +1.93 [+1.41, +2.51] behind | +5.63 [+4.80, +6.39] behind (held) | +6.88 [+6.10, +7.59] behind (held) |
| S | +0.08 [-0.81, +0.88] neither | +2.85 [+2.09, +3.63] behind (held) | +3.32 [+2.52, +4.14] behind (held) |
| W | +0.07 [-0.63, +0.76] neither | +0.69 [-0.07, +1.35] neither (held) | +1.50 [+0.63, +2.26] behind (held) |
| LS | +1.05 [+0.26, +1.82] behind | +3.81 [+3.25, +4.28] behind (held) | +4.60 [+4.08, +5.07] behind (held) |
| LW | -0.30 [-0.70, +0.07] neither | +1.88 [+1.14, +2.68] behind (held) | +2.70 [+2.00, +3.50] behind (held) |
| SW | -1.29 [-2.01, -0.68] ahead | +1.16 [+0.68, +1.67] behind (held) | +1.85 [+1.27, +2.50] behind (held) |
| LSW | -0.86 [-1.26, -0.44] ahead | +1.96 [+1.45, +2.41] behind (held) | +2.80 [+2.15, +3.38] behind (held) |
| Pool | +0.19 [-0.14, +0.51] neither | +2.80 [+2.33, +3.18] behind (held) | +3.63 [+3.12, +4.08] behind (held) |
| combinations without setups | +0.64 [+0.15, +1.24] behind | +3.16 [+2.53, +3.79] behind (held) | +4.12 [+3.45, +4.79] behind (held) |
| combinations with setups | -0.26 [-0.71, +0.15] neither | +2.44 [+2.02, +2.83] behind (held) | +3.14 [+2.64, +3.62] behind (held) |

#### CP-SAT (8 workers) started from the policy's schedule minus the best of the eight memetic runs started from the policy's schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +0.84 [-0.47, +1.93] neither | +0.31 [-0.27, +0.85] neither | +0.52 [-0.06, +1.15] neither |
| L | +1.05 [+0.35, +1.53] behind | +0.28 [-0.02, +0.58] neither | +0.40 [-0.11, +0.85] neither |
| S | +1.59 [+1.22, +1.99] behind | +4.03 [+3.70, +4.42] behind | +3.92 [+3.35, +4.42] behind |
| W | -0.45 [-1.22, +0.55] neither | -2.05 [-2.56, -1.43] ahead | -1.69 [-2.27, -1.00] ahead |
| LS | +2.97 [+2.25, +3.68] behind | +4.42 [+4.01, +4.85] behind | +4.57 [+3.92, +5.20] behind |
| LW | -0.35 [-0.95, +0.08] neither | -1.13 [-1.93, -0.35] ahead | -1.37 [-2.10, -0.72] ahead |
| SW | +0.93 [+0.57, +1.29] behind | +3.00 [+2.11, +4.24] behind | +3.24 [+2.22, +4.59] behind |
| LSW | +0.60 [+0.22, +1.02] behind | +3.35 [+2.91, +3.81] behind | +3.71 [+3.00, +4.41] behind |
| Pool | +0.90 [+0.57, +1.21] behind | +1.53 [+1.21, +1.86] behind | +1.66 [+1.25, +2.08] behind |
| combinations without setups | +0.27 [-0.29, +0.76] neither | -0.65 [-1.02, -0.26] ahead | -0.54 [-0.99, -0.12] ahead |
| combinations with setups | +1.52 [+1.23, +1.85] behind | +3.70 [+3.35, +4.12] behind | +3.86 [+3.37, +4.34] behind |

#### the best of the eight memetic runs started from the policy's schedule minus the best of the eight started from the rule schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.19 [-3.49, -0.75] ahead | -0.18 [-1.13, +0.80] neither | -0.02 [-0.64, +0.59] neither |
| L | -1.22 [-2.33, +0.03] neither | +0.54 [-0.03, +1.20] neither | +0.75 [+0.18, +1.36] behind |
| S | -2.65 [-4.05, -1.34] ahead | -0.30 [-1.18, +0.61] neither | -0.09 [-0.91, +0.77] neither |
| W | -3.62 [-5.05, -2.08] ahead | -1.36 [-2.68, +0.01] neither | -0.97 [-2.14, +0.19] neither |
| LS | -2.77 [-4.10, -1.40] ahead | -0.44 [-1.64, +0.62] neither | -0.60 [-1.67, +0.23] neither |
| LW | -4.42 [-6.07, -2.90] ahead | -2.06 [-3.62, -0.46] ahead | -1.07 [-2.56, +0.42] neither |
| SW | -2.05 [-3.26, -0.95] ahead | -1.20 [-2.34, -0.18] ahead | -0.68 [-1.93, +0.40] neither |
| LSW | -3.29 [-4.64, -1.99] ahead | -0.91 [-1.84, +0.05] neither | -0.66 [-1.57, +0.28] neither |
| Pool | -2.78 [-3.49, -1.93] ahead | -0.74 [-1.25, -0.13] ahead | -0.42 [-0.85, +0.09] neither |
| combinations without setups | -2.86 [-3.89, -1.64] ahead | -0.77 [-1.66, +0.24] neither | -0.33 [-1.06, +0.49] neither |
| combinations with setups | -2.69 [-3.78, -1.64] ahead | -0.71 [-1.55, +0.03] neither | -0.51 [-1.35, +0.20] neither |

#### the best of the eight memetic runs started from the policy's schedule minus the policy's greedy schedule, in percent of the greedy schedule (the form of the decision table)

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -5.91 [-9.15, -2.59] ahead | -9.60 [-12.23, -6.98] ahead | -10.44 [-12.93, -7.91] ahead |
| L | -2.85 [-3.59, -2.27] ahead | -6.67 [-7.47, -5.81] ahead | -7.76 [-8.55, -6.95] ahead |
| S | -1.66 [-2.04, -1.32] ahead | -4.50 [-4.94, -4.07] ahead | -4.94 [-5.45, -4.41] ahead |
| W | -3.19 [-5.90, -1.21] ahead | -3.94 [-6.53, -2.00] ahead | -4.71 [-7.29, -2.73] ahead |
| LS | -3.22 [-4.06, -2.41] ahead | -6.03 [-6.91, -5.25] ahead | -6.74 [-7.75, -5.80] ahead |
| LW | -1.28 [-1.79, -0.84] ahead | -3.46 [-4.53, -2.39] ahead | -4.22 [-5.25, -3.25] ahead |
| SW | -1.30 [-1.75, -0.86] ahead | -3.88 [-5.04, -3.00] ahead | -4.53 [-5.71, -3.60] ahead |
| LSW | -0.80 [-1.18, -0.47] ahead | -3.89 [-4.30, -3.42] ahead | -4.67 [-5.23, -4.11] ahead |
| Pool | -2.53 [-3.19, -1.94] ahead | -5.25 [-5.91, -4.60] ahead | -6.00 [-6.70, -5.34] ahead |
| combinations without setups | -3.31 [-4.70, -2.07] ahead | -5.92 [-7.18, -4.71] ahead | -6.78 [-8.01, -5.61] ahead |
| combinations with setups | -1.74 [-2.11, -1.44] ahead | -4.58 [-5.12, -4.10] ahead | -5.22 [-5.84, -4.65] ahead |

#### the best of the eight memetic runs started from the rule schedule minus the policy's greedy schedule, in percent of the greedy schedule (the form of the decision table)

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.82 [-6.46, -1.07] ahead | -9.46 [-11.59, -7.20] ahead | -10.44 [-12.55, -8.27] ahead |
| L | -1.62 [-2.85, -0.39] ahead | -7.16 [-8.24, -6.10] ahead | -8.43 [-9.46, -7.50] ahead |
| S | +1.07 [-0.51, +2.75] neither | -4.20 [-5.25, -3.24] ahead | -4.83 [-5.87, -3.89] ahead |
| W | +0.47 [-1.98, +2.81] neither | -2.57 [-5.29, -0.03] ahead | -3.75 [-6.41, -1.41] ahead |
| LS | -0.42 [-1.74, +0.84] neither | -5.58 [-7.19, -4.00] ahead | -6.15 [-7.75, -4.57] ahead |
| LW | +3.36 [+1.43, +5.44] behind | -1.34 [-3.77, +1.22] neither | -3.12 [-5.33, -0.81] ahead |
| SW | +0.81 [-0.58, +2.18] neither | -2.69 [-4.17, -1.46] ahead | -3.86 [-5.21, -2.80] ahead |
| LSW | +2.62 [+1.26, +4.13] behind | -2.98 [-4.10, -1.96] ahead | -4.02 [-5.12, -3.11] ahead |
| Pool | +0.31 [-0.53, +1.11] neither | -4.50 [-5.31, -3.66] ahead | -5.58 [-6.31, -4.83] ahead |
| combinations without setups | -0.40 [-1.78, +1.02] neither | -5.13 [-6.54, -3.57] ahead | -6.44 [-7.67, -5.06] ahead |
| combinations with setups | +1.02 [-0.03, +2.11] neither | -3.86 [-4.79, -3.00] ahead | -4.71 [-5.63, -3.87] ahead |

#### Mean makespans of the second set (cells used)

| budget | combination | cells | best of eight started from the policy's schedule | CP-SAT started from the policy's schedule |
|---|---|---|---|---|
| 60 | N | 9 | 339.11 | 341.89 |
| 60 | L | 9 | 415.67 | 420.00 |
| 60 | S | 9 | 449.56 | 456.67 |
| 60 | W | 9 | 365.22 | 363.56 |
| 60 | LS | 9 | 528.11 | 543.89 |
| 60 | LW | 9 | 443.00 | 441.44 |
| 60 | SW | 9 | 485.44 | 489.89 |
| 60 | LSW | 9 | 560.78 | 564.11 |
| 300 | N | 9 | 325.89 | 326.89 |
| 300 | L | 9 | 399.33 | 400.44 |
| 300 | S | 9 | 436.56 | 454.11 |
| 300 | W | 9 | 362.44 | 355.00 |
| 300 | LS | 9 | 512.67 | 535.44 |
| 300 | LW | 9 | 433.22 | 428.33 |
| 300 | SW | 9 | 472.67 | 486.78 |
| 300 | LSW | 9 | 543.33 | 561.56 |
| 600 | N | 9 | 322.89 | 324.56 |
| 600 | L | 9 | 394.67 | 396.22 |
| 600 | S | 9 | 434.56 | 451.56 |
| 600 | W | 9 | 359.56 | 353.44 |
| 600 | LS | 9 | 508.78 | 532.11 |
| 600 | LW | 9 | 429.78 | 423.89 |
| 600 | SW | 9 | 469.44 | 484.56 |
| 600 | LSW | 9 | 538.89 | 558.89 |

### 80x25n30, second set

Cells used (all 8 runs valid): 72 of 72; valid runs per seed: seed 1: 72 (8-15 72); seed 2: 72 (16-23 72); seed 3: 72 (16-23 72); seed 4: 72 (16-23 72); seed 5: 72 (16-23 72); seed 6: 72 (16-23 72); seed 7: 72 (8-15 72); seed 8: 72 (8-15 72)

#### the policy, sampled schedules, minus the best of the eight memetic runs started from the policy's schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -0.24 [-0.96, +0.44] neither | +3.27 [+2.59, +3.94] behind (held) | +4.35 [+3.66, +5.08] behind (held) |
| L | -0.55 [-1.26, +0.23] neither | +5.57 [+4.16, +6.76] behind (held) | +6.69 [+5.11, +8.05] behind (held) |
| S | +0.13 [-0.50, +0.82] neither | +1.30 [+0.59, +1.89] behind (held) | +1.77 [+1.06, +2.32] behind (held) |
| W | -0.62 [-1.53, +0.07] neither | -0.03 [-0.87, +0.80] neither (held) | +0.54 [-0.21, +1.30] neither (held) |
| LS | +0.70 [+0.14, +1.18] behind | +2.63 [+1.84, +3.38] behind (held) | +3.30 [+2.49, +4.11] behind (held) |
| LW | -0.67 [-1.44, +0.04] neither | +0.99 [+0.23, +1.76] behind (held) | +2.16 [+1.42, +3.04] behind (held) |
| SW | -0.96 [-1.67, -0.16] ahead | -0.11 [-0.56, +0.43] neither (held) | +0.15 [-0.34, +0.70] neither (held) |
| LSW | -0.47 [-1.17, +0.13] neither | +0.43 [-0.49, +1.31] neither (held) | +0.99 [+0.24, +1.75] behind (held) |
| Pool | -0.34 [-0.66, -0.00] ahead | +1.76 [+1.41, +2.12] behind (held) | +2.49 [+2.10, +2.89] behind (held) |
| combinations without setups | -0.52 [-0.98, -0.07] ahead | +2.45 [+1.81, +3.03] behind (held) | +3.44 [+2.72, +4.11] behind (held) |
| combinations with setups | -0.15 [-0.61, +0.29] neither | +1.06 [+0.68, +1.42] behind (held) | +1.55 [+1.21, +1.88] behind (held) |

#### CP-SAT (8 workers) started from the policy's schedule minus the best of the eight memetic runs started from the policy's schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +1.59 [-0.00, +3.70] neither | +1.16 [+0.50, +1.85] behind | +0.94 [+0.52, +1.39] behind |
| L | +2.71 [+0.03, +6.08] behind | +2.32 [+1.50, +3.06] behind | +1.71 [+0.70, +2.72] behind |
| S | +1.34 [+0.83, +1.88] behind | +2.77 [+2.16, +3.39] behind | +3.10 [+2.47, +3.70] behind |
| W | +0.07 [-0.23, +0.41] neither | -1.91 [-2.40, -1.42] ahead | -2.35 [-2.94, -1.73] ahead |
| LS | +4.06 [+2.11, +6.95] behind | +5.66 [+3.67, +8.54] behind | +5.74 [+3.47, +8.75] behind |
| LW | +0.37 [-0.96, +2.31] neither | -2.04 [-3.17, -1.02] ahead | -1.83 [-2.70, -0.97] ahead |
| SW | +1.14 [+0.21, +2.60] behind | +1.41 [+0.90, +2.12] behind | +1.32 [+0.87, +1.88] behind |
| LSW | +2.21 [+0.34, +5.24] behind | +2.97 [+1.41, +5.34] behind | +3.11 [+1.72, +5.17] behind |
| Pool | +1.69 [+0.79, +2.80] behind | +1.54 [+1.08, +2.19] behind | +1.47 [+1.02, +2.07] behind |
| combinations without setups | +1.18 [-0.06, +2.92] neither | -0.12 [-0.42, +0.17] neither | -0.38 [-0.64, -0.15] ahead |
| combinations with setups | +2.19 [+1.26, +3.56] behind | +3.20 [+2.51, +4.32] behind | +3.32 [+2.60, +4.39] behind |

#### the best of the eight memetic runs started from the policy's schedule minus the best of the eight started from the rule schedule

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -3.31 [-4.51, -2.17] ahead | -0.52 [-1.62, +0.36] neither | -0.20 [-1.18, +0.52] neither |
| L | -3.21 [-4.25, -2.37] ahead | -0.95 [-1.76, -0.15] ahead | -0.63 [-1.40, +0.07] neither |
| S | -3.08 [-4.58, -1.62] ahead | -1.16 [-2.06, -0.24] ahead | -0.76 [-1.55, +0.04] neither |
| W | -4.87 [-6.14, -3.55] ahead | -4.40 [-5.90, -3.15] ahead | -3.45 [-4.54, -2.39] ahead |
| LS | -2.92 [-3.82, -1.96] ahead | -1.67 [-2.64, -0.64] ahead | -1.17 [-2.24, -0.16] ahead |
| LW | -3.24 [-3.86, -2.60] ahead | -3.01 [-3.87, -2.11] ahead | -2.72 [-3.48, -1.89] ahead |
| SW | -5.06 [-7.38, -3.08] ahead | -3.45 [-5.28, -1.80] ahead | -2.58 [-3.75, -1.35] ahead |
| LSW | -4.85 [-5.94, -3.67] ahead | -2.71 [-3.78, -1.77] ahead | -2.04 [-2.85, -1.31] ahead |
| Pool | -3.82 [-4.61, -3.13] ahead | -2.23 [-2.69, -1.77] ahead | -1.69 [-2.09, -1.29] ahead |
| combinations without setups | -3.66 [-4.41, -2.88] ahead | -2.22 [-2.85, -1.64] ahead | -1.75 [-2.38, -1.19] ahead |
| combinations with setups | -3.98 [-4.97, -3.12] ahead | -2.25 [-2.98, -1.60] ahead | -1.64 [-2.21, -1.09] ahead |

#### the best of the eight memetic runs started from the policy's schedule minus the policy's greedy schedule, in percent of the greedy schedule (the form of the decision table)

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | -2.15 [-4.64, -0.35] ahead | -5.84 [-8.04, -3.91] ahead | -6.82 [-8.87, -4.99] ahead |
| L | -4.00 [-8.11, -0.88] ahead | -9.77 [-12.73, -7.34] ahead | -10.71 [-13.61, -8.29] ahead |
| S | -1.37 [-1.89, -0.86] ahead | -2.99 [-3.58, -2.37] ahead | -3.43 [-4.02, -2.81] ahead |
| W | -0.52 [-0.91, -0.25] ahead | -1.63 [-2.11, -1.08] ahead | -2.19 [-2.71, -1.61] ahead |
| LS | -3.85 [-6.36, -2.11] ahead | -5.93 [-8.55, -3.99] ahead | -6.54 [-9.13, -4.60] ahead |
| LW | -1.47 [-3.74, -0.22] ahead | -3.62 [-5.71, -2.18] ahead | -4.73 [-6.63, -3.36] ahead |
| SW | -1.17 [-2.52, -0.29] ahead | -2.23 [-3.80, -1.21] ahead | -2.48 [-4.06, -1.46] ahead |
| LSW | -2.05 [-4.72, -0.37] ahead | -3.36 [-6.08, -1.57] ahead | -3.90 [-6.62, -2.12] ahead |
| Pool | -2.07 [-3.25, -1.17] ahead | -4.42 [-5.27, -3.73] ahead | -5.10 [-5.87, -4.46] ahead |
| combinations without setups | -2.04 [-4.06, -0.61] ahead | -5.22 [-6.66, -4.17] ahead | -6.11 [-7.46, -5.13] ahead |
| combinations with setups | -2.11 [-3.29, -1.30] ahead | -3.63 [-4.74, -2.87] ahead | -4.09 [-5.21, -3.32] ahead |

#### the best of the eight memetic runs started from the rule schedule minus the policy's greedy schedule, in percent of the greedy schedule (the form of the decision table)

| entry | 60 s | 300 s | 600 s |
|---|---|---|---|
| N | +1.22 [-1.19, +3.29] neither | -5.34 [-7.24, -3.70] ahead | -6.63 [-8.52, -4.98] ahead |
| L | -0.80 [-4.92, +2.53] neither | -8.93 [-11.39, -6.84] ahead | -10.17 [-12.63, -8.04] ahead |
| S | +1.82 [+0.21, +3.54] behind | -1.83 [-3.08, -0.53] ahead | -2.67 [-3.85, -1.42] ahead |
| W | +4.61 [+3.17, +6.09] behind | +2.94 [+1.55, +4.55] behind | +1.33 [+0.21, +2.45] behind |
| LS | -0.93 [-3.64, +1.22] neither | -4.32 [-6.79, -2.25] ahead | -5.43 [-7.68, -3.49] ahead |
| LW | +1.83 [-0.53, +3.47] neither | -0.62 [-2.81, +1.01] neither | -2.06 [-4.02, -0.61] ahead |
| SW | +4.23 [+1.56, +7.26] behind | +1.35 [-0.97, +3.85] neither | +0.13 [-1.75, +1.92] neither |
| LSW | +2.95 [+0.42, +5.01] behind | -0.66 [-3.37, +1.34] neither | -1.91 [-4.41, -0.08] ahead |
| Pool | +1.87 [+0.68, +3.14] behind | -2.18 [-2.84, -1.54] ahead | -3.43 [-3.96, -2.91] ahead |
| combinations without setups | +1.72 [-0.10, +3.24] neither | -2.99 [-3.98, -2.24] ahead | -4.38 [-5.22, -3.76] ahead |
| combinations with setups | +2.02 [+0.78, +3.46] behind | -1.37 [-2.39, -0.38] ahead | -2.47 [-3.34, -1.70] ahead |

#### Mean makespans of the second set (cells used)

| budget | combination | cells | best of eight started from the policy's schedule | CP-SAT started from the policy's schedule |
|---|---|---|---|---|
| 60 | N | 9 | 498.89 | 506.67 |
| 60 | L | 9 | 589.78 | 605.44 |
| 60 | S | 9 | 650.44 | 659.11 |
| 60 | W | 9 | 531.56 | 531.89 |
| 60 | LS | 9 | 731.89 | 761.56 |
| 60 | LW | 9 | 618.56 | 620.56 |
| 60 | SW | 9 | 689.22 | 697.67 |
| 60 | LSW | 9 | 766.89 | 783.56 |
| 300 | N | 9 | 480.00 | 485.56 |
| 300 | L | 9 | 554.67 | 567.56 |
| 300 | S | 9 | 639.78 | 657.44 |
| 300 | W | 9 | 525.56 | 515.44 |
| 300 | LS | 9 | 716.00 | 756.33 |
| 300 | LW | 9 | 605.11 | 592.56 |
| 300 | SW | 9 | 681.78 | 691.56 |
| 300 | LSW | 9 | 756.56 | 778.78 |
| 600 | N | 9 | 475.00 | 479.44 |
| 600 | L | 9 | 548.89 | 558.22 |
| 600 | S | 9 | 636.89 | 656.56 |
| 600 | W | 9 | 522.56 | 510.22 |
| 600 | LS | 9 | 711.33 | 751.89 |
| 600 | LW | 9 | 598.11 | 587.00 |
| 600 | SW | 9 | 680.00 | 689.11 |
| 600 | LSW | 9 | 752.33 | 775.56 |

## Decision table on eight cores

- p94's decision table with the memetic algorithm as the best of the eight runs from each start, on eight cores; request clock; difference from the greedy schedule of run s304 in % of it
- the columns of the memetic algorithm (started from the rule schedule, started from the policy's schedule) hold the best of the eight runs from that start; the columns rules, policy_sample, cpsat_rule, cpsat_policy equal the values of results/timed_inference.json
- counts over the rows the table prints (combinations without setups, combinations with setups at 60, 300, 600 s)
- curves: results/memetic_parallel_curves.csv, methods memetic_rule_best8, memetic_policy_best8, 1111 rows

### 50x25n30: difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Pool | +4.03 (9) | -2.36 (9) | +0.31 (9) | +1.53 (9) | **-2.53** (9) | -1.67 (9) | memetic_policy | yes |
| 60 | combinations without setups | +3.23 (9) | -2.72 (9) | -0.40 (9) | -1.29 (9) | **-3.31** (9) | -3.08 (9) | memetic_policy | yes |
| 60 | combinations with setups | +4.83 (9) | **-2.00** (9) | +1.02 (9) | +4.34 (9) | -1.74 (9) | -0.26 (9) | policy_sample | yes |
| 300 | Pool | +4.03 (9) | -2.63 (9) | -4.50 (9) | -2.21 (9) | **-5.25** (9) | -3.80 (9) | memetic_policy | yes |
| 300 | combinations without setups | +3.23 (9) | -3.00 (9) | -5.13 (9) | -6.33 (9) | -5.92 (9) | **-6.55** (9) | cpsat_policy | yes |
| 300 | combinations with setups | +4.83 (9) | -2.26 (9) | -3.86 (9) | +1.91 (9) | **-4.58** (9) | -1.06 (9) | memetic_policy | yes |
| 600 | Pool | +4.03 (9) | -2.63 (9) | -5.58 (9) | -3.33 (9) | **-6.00** (9) | -4.44 (9) | memetic_policy | yes |
| 600 | combinations without setups | +3.23 (9) | -3.00 (9) | -6.44 (9) | **-7.63** (9) | -6.78 (9) | -7.31 (9) | cpsat_rule | yes |
| 600 | combinations with setups | +4.83 (9) | -2.26 (9) | -4.71 (9) | +0.97 (9) | **-5.22** (9) | -1.58 (9) | memetic_policy | yes |

### 80x25n30: difference from the greedy schedule in % (row Sel); bold = the lowest printed value of the row; (n) = instances of the value

| s since the request | entry | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | lowest | same instances |
|---|---|---|---|---|---|---|---|---|---|
| 60 | Pool | +3.50 (9) | **-2.41** (9) | +1.87 (9) | +2.62 (9) | -2.07 (9) | -0.53 (9) | policy_sample | yes |
| 60 | combinations without setups | +2.69 (9) | **-2.55** (9) | +1.72 (9) | +1.02 (9) | -2.04 (9) | -1.00 (9) | policy_sample | yes |
| 60 | combinations with setups | +4.31 (9) | **-2.26** (9) | +2.02 (9) | +4.22 (9) | -2.11 (9) | -0.06 (9) | policy_sample | yes |
| 300 | Pool | +3.50 (9) | -2.78 (9) | -2.18 (9) | -0.49 (9) | **-4.42** (9) | -3.01 (9) | memetic_policy | yes |
| 300 | combinations without setups | +2.69 (9) | -2.95 (9) | -2.99 (9) | -4.52 (9) | -5.22 (9) | **-5.38** (9) | cpsat_policy | yes |
| 300 | combinations with setups | +4.31 (9) | -2.62 (9) | -1.37 (9) | +3.54 (9) | **-3.63** (9) | -0.63 (9) | memetic_policy | yes |
| 600 | Pool | +3.50 (9) | -2.78 (9) | -3.43 (9) | -1.73 (9) | **-5.10** (9) | -3.76 (9) | memetic_policy | yes |
| 600 | combinations without setups | +2.69 (9) | -2.95 (9) | -4.38 (9) | -6.31 (9) | -6.11 (9) | **-6.52** (9) | cpsat_policy | yes |
| 600 | combinations with setups | +4.31 (9) | -2.62 (9) | -2.47 (9) | +2.85 (9) | **-4.09** (9) | -0.99 (9) | memetic_policy | yes |

### Rows by lowest column

| sizes | rows | best of the four dispatching rules | the policy, sampled schedules | memetic algorithm started from the rule schedule | CP-SAT started from the rule schedule | memetic algorithm started from the policy's schedule | CP-SAT started from the policy's schedule | ties | rows at 300 and 600 s | of these, lowest a search from the policy's schedule | lowest uses the policy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 50x25n30 | 6 | 0 | 1 | 0 | 1 | 3 | 1 | 0 | 4 | 3 | 5 |
| 80x25n30 | 6 | 0 | 2 | 0 | 0 | 2 | 2 | 0 | 4 | 4 | 6 |
| both | 12 | 0 | 3 | 0 | 1 | 5 | 3 | 0 | 8 | 7 | 11 |

