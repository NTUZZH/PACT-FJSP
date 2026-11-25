# Policy trained on reinforced-concrete instances alone (scripts/p131_rc_trained_policy.py infer)

Arm: xmask-rc-s<seed>: the released recipe with --ppvc_mix rc_project. Exact mask; training seeds 301,302,303 for every arm; B = 20000. Levels: mix_rc (reinforced-concrete modules alone), default (the mixed test sets). d = 100 (first - second) / second, seed s against seed s, per base instance; negative means the first-named method is shorter. Best rule: per-instance minimum of FIFO, MOR, SPT, MWKR; best of seven: per-instance minimum of FIFO, MOR, SPT, MWKR, ECT, SSPT, LLF. Holm over the eight combinations of a level and size. Verdicts: (i) and (iv) the combined superiority verdict; (ii) and (iii) the verdicts at the margins 0.5% / 1%.

Contamination: 6060 training and validation base instances compared with 150 stored 10-module test instances, 0 seed-block hits and 0 equal instances; the replayed seed stream equals the trainer's sampler on 120 instances.

Evaluation path (the policy's seed 301, this script against the campaign): 10x25 N 3 of 3 equal; 10x25 LSW 3 of 3 equal; 50x25n30 N 3 of 3 equal; 50x25n30 LSW 3 of 3 equal; 80x25n30 N 3 of 3 equal; 80x25n30 LSW 3 of 3 equal.

Rule means reproduced: 432 values of results/project_mix.json.

## 10x25, mix_rc (10 modules, 50 instances, reinforced-concrete modules alone)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -1.58 [-2.37, -0.76] ahead | -1.38 [-2.17, -0.56] ahead | -2.79 [-4.41, -0.67] non-inferior only / non-inferior only, ahead | +0.25 [-0.90, +1.44] inconclusive / inconclusive, neither | +1.32 [-0.83, +2.97] not distinguishable | +1.53 [-0.61, +3.20] not distinguishable |
| L | -2.15 [-2.96, -1.32] ahead | -1.80 [-2.60, -1.01] ahead | -0.56 [-1.21, -0.05] non-inferior only / non-inferior only, ahead | -0.60 [-1.28, -0.03] non-inferior only / non-inferior only, ahead | -1.58 [-2.38, -0.73] ahead | -1.23 [-2.01, -0.40] not distinguishable |
| S | -0.38 [-1.19, +0.49] not distinguishable | +0.89 [+0.01, +1.83] not distinguishable | -1.62 [-3.60, +0.03] non-inferior only / non-inferior only, neither | -0.84 [-2.30, +0.34] non-inferior only / non-inferior only, neither | +1.34 [-0.56, +3.37] not distinguishable | +2.64 [+0.69, +4.75] not distinguishable |
| W | -2.29 [-3.19, -1.43] ahead | -2.16 [-3.05, -1.31] ahead | -3.17 [-5.54, -0.03] non-inferior only / non-inferior only, ahead | -1.36 [-2.92, +0.03] non-inferior only / non-inferior only, neither | +1.06 [-2.01, +3.44] not distinguishable | +1.19 [-1.86, +3.57] not distinguishable |
| LS | -0.72 [-1.31, -0.13] ahead | +0.11 [-0.49, +0.72] not distinguishable | -0.37 [-0.87, +0.19] non-inferior only / equivalent, neither | -0.35 [-0.98, +0.25] non-inferior only / equivalent, neither | -0.35 [-0.99, +0.25] not distinguishable | +0.49 [-0.16, +1.10] not distinguishable |
| LW | -2.67 [-3.47, -1.87] ahead | -2.04 [-2.84, -1.23] ahead | -0.58 [-1.02, -0.17] non-inferior only / non-inferior only, ahead | -1.06 [-2.06, -0.22] non-inferior only / non-inferior only, ahead | -2.08 [-2.86, -1.28] ahead | -1.44 [-2.24, -0.61] ahead |
| SW | -0.71 [-1.58, +0.13] not distinguishable | +0.63 [-0.19, +1.42] not distinguishable | -0.80 [-2.32, +0.55] inconclusive / non-inferior only, neither | +0.53 [-0.23, +1.31] inconclusive / inconclusive, neither | +0.17 [-1.23, +1.64] not distinguishable | +1.53 [+0.12, +3.01] not distinguishable |
| LSW | -0.14 [-0.83, +0.58] not distinguishable | +0.81 [+0.14, +1.48] not distinguishable | +0.09 [-0.67, +0.79] inconclusive / equivalent, neither | -0.46 [-1.01, +0.15] non-inferior only / non-inferior only, neither | -0.21 [-0.82, +0.44] not distinguishable | +0.74 [+0.18, +1.33] not distinguishable |
| pooled | -1.33 [-1.70, -0.96] direction ahead | -0.62 [-1.00, -0.24] direction ahead | -1.23 [-2.09, -0.06] non-inferior only / non-inferior only, ahead | -0.49 [-0.76, -0.24] non-inferior only / equivalent, ahead | -0.04 [-1.15, +0.81] direction neither | +0.68 [-0.44, +1.56] direction neither |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 131.2 | 229.5 | 196.4 | 146.9 | 288.3 | 239.0 | 221.8 | 306.4 |
| policy | 135.1 | 230.8 | 199.9 | 152.0 | 289.4 | 240.4 | 223.8 | 306.1 |
| specialist | 131.0 | 230.9 | 198.2 | 149.1 | 289.3 | 241.6 | 220.7 | 307.9 |
| best rule | 133.4 | 234.6 | 197.3 | 150.5 | 290.5 | 245.7 | 223.6 | 306.9 |
| best of seven | 133.1 | 233.7 | 194.8 | 150.3 | 288.1 | 244.1 | 220.5 | 304.0 |

## 10x25, default (10 modules, 100 instances, mixed test sets)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -1.09 [-1.84, -0.21] not distinguishable | -0.85 [-1.58, +0.01] not distinguishable | +0.87 [+0.17, +1.69] inferior / inferior, behind | +0.92 [-0.18, +1.97] inconclusive / inconclusive, neither | -1.91 [-2.41, -1.40] ahead | -1.68 [-2.15, -1.19] ahead |
| L | -1.57 [-2.14, -1.02] ahead | -1.22 [-1.74, -0.71] ahead | -0.14 [-0.60, +0.31] non-inferior only / equivalent, neither | -0.09 [-0.54, +0.40] non-inferior only / equivalent, neither | -1.40 [-1.97, -0.86] ahead | -1.06 [-1.57, -0.56] ahead |
| S | -0.13 [-0.70, +0.50] not distinguishable | +0.82 [+0.24, +1.47] not distinguishable | +0.27 [-0.13, +0.71] inconclusive / equivalent, neither | +0.13 [-0.33, +0.59] inconclusive / equivalent, neither | -0.36 [-0.94, +0.25] not distinguishable | +0.58 [+0.02, +1.18] not distinguishable |
| W | -1.31 [-2.20, -0.39] not distinguishable | -0.94 [-1.77, -0.05] not distinguishable | +0.76 [+0.03, +1.60] inferior / inferior, behind | +1.07 [-0.35, +2.53] inconclusive / inconclusive, neither | -2.00 [-2.68, -1.31] ahead | -1.63 [-2.26, -1.00] ahead |
| LS | -1.01 [-1.41, -0.61] ahead | -0.31 [-0.71, +0.08] not distinguishable | +0.10 [-0.36, +0.52] inconclusive / equivalent, neither | -0.02 [-0.31, +0.27] equivalent / equivalent, neither | -1.08 [-1.58, -0.55] ahead | -0.39 [-0.88, +0.13] not distinguishable |
| LW | -2.33 [-2.85, -1.84] ahead | -1.82 [-2.33, -1.33] ahead | -0.03 [-0.40, +0.33] equivalent / equivalent, neither | +0.18 [-0.27, +0.58] inconclusive / equivalent, neither | -2.27 [-2.79, -1.75] ahead | -1.75 [-2.30, -1.22] ahead |
| SW | -0.86 [-1.35, -0.37] ahead | +0.19 [-0.31, +0.68] not distinguishable | +0.16 [-0.24, +0.57] inconclusive / equivalent, neither | -0.35 [-0.92, +0.18] non-inferior only / equivalent, neither | -0.97 [-1.50, -0.44] ahead | +0.07 [-0.44, +0.59] not distinguishable |
| LSW | -1.08 [-1.53, -0.60] ahead | -0.00 [-0.47, +0.47] not distinguishable | +0.17 [-0.23, +0.58] inconclusive / equivalent, neither | +0.17 [-0.22, +0.57] inconclusive / equivalent, neither | -1.22 [-1.72, -0.70] ahead | -0.15 [-0.63, +0.35] not distinguishable |
| pooled | -1.17 [-1.48, -0.85] direction ahead | -0.52 [-0.82, -0.20] direction ahead | +0.27 [+0.10, +0.45] equivalent / equivalent, behind | +0.25 [-0.05, +0.57] inconclusive / equivalent, neither | -1.40 [-1.68, -1.13] direction ahead | -0.75 [-1.02, -0.49] direction ahead |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 130.7 | 210.8 | 194.2 | 148.2 | 269.5 | 220.0 | 218.1 | 286.8 |
| policy | 129.7 | 211.1 | 193.7 | 147.2 | 269.3 | 220.2 | 217.8 | 286.4 |
| specialist | 129.6 | 211.0 | 194.0 | 146.8 | 269.7 | 219.7 | 219.0 | 286.4 |
| best rule | 132.3 | 214.2 | 194.5 | 150.4 | 272.3 | 225.3 | 220.1 | 290.0 |
| best of seven | 131.9 | 213.4 | 192.7 | 149.8 | 270.4 | 224.2 | 217.8 | 286.9 |

## 50x25n30, mix_rc (50 modules, 30 instances, reinforced-concrete modules alone)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -6.02 [-6.65, -5.38] ahead | -6.02 [-6.65, -5.38] ahead | -7.68 [-12.33, -0.92] non-inferior only / non-inferior only, ahead | -5.24 [-8.95, -2.02] non-inferior only / non-inferior only, ahead | +2.13 [-5.20, +7.30] not distinguishable | +2.13 [-5.20, +7.30] not distinguishable |
| L | -3.15 [-3.82, -2.38] ahead | -2.88 [-3.54, -2.14] ahead | -5.82 [-9.28, -0.20] non-inferior only / non-inferior only, ahead | -3.19 [-7.48, +0.01] non-inferior only / non-inferior only, neither | +3.05 [-3.03, +7.15] not distinguishable | +3.34 [-2.77, +7.46] not distinguishable |
| S | -2.96 [-3.83, -1.93] ahead | -2.19 [-3.08, -1.19] ahead | -4.09 [-8.19, -0.26] non-inferior only / non-inferior only, ahead | -6.47 [-18.60, +0.48] non-inferior only / non-inferior only, neither | +1.36 [-3.05, +6.13] not distinguishable | +2.17 [-2.28, +6.93] not distinguishable |
| W | -5.38 [-5.97, -4.76] ahead | -5.38 [-5.97, -4.76] ahead | -5.80 [-10.41, -0.50] non-inferior only / non-inferior only, ahead | -7.19 [-12.22, -2.51] non-inferior only / non-inferior only, ahead | +0.67 [-4.88, +5.63] not distinguishable | +0.67 [-4.88, +5.63] not distinguishable |
| LS | -1.04 [-2.22, +0.22] not distinguishable | -0.02 [-1.24, +1.23] not distinguishable | -2.61 [-5.25, +0.87] inconclusive / non-inferior only, neither | -1.47 [-2.73, -0.28] non-inferior only / non-inferior only, ahead | +1.73 [-1.69, +4.28] not distinguishable | +2.77 [-0.74, +5.28] not distinguishable |
| LW | -2.91 [-3.60, -2.05] ahead | -2.62 [-3.33, -1.78] ahead | -4.38 [-6.85, -0.78] non-inferior only / non-inferior only, ahead | -12.79 [-22.44, -1.17] non-inferior only / non-inferior only, ahead | +1.66 [-2.52, +4.70] not distinguishable | +1.96 [-2.22, +5.01] not distinguishable |
| SW | -2.67 [-3.65, -1.69] ahead | -1.89 [-2.69, -1.06] ahead | -3.55 [-6.51, -0.38] non-inferior only / non-inferior only, ahead | -2.54 [-7.46, +0.67] inconclusive / non-inferior only, neither | +1.06 [-2.41, +4.63] not distinguishable | +1.88 [-1.61, +5.45] not distinguishable |
| LSW | -1.91 [-3.07, -0.73] ahead | +0.11 [-1.07, +1.48] not distinguishable | -2.20 [-3.99, -0.09] non-inferior only / non-inferior only, ahead | -1.75 [-3.48, -0.29] non-inferior only / non-inferior only, ahead | +0.37 [-2.11, +2.53] not distinguishable | +2.42 [-0.12, +4.57] not distinguishable |
| pooled | -3.26 [-3.75, -2.66] direction ahead | -2.61 [-3.13, -1.99] direction ahead | -4.52 [-7.59, -0.20] non-inferior only / non-inferior only, ahead | -5.08 [-7.40, -2.94] non-inferior only / non-inferior only, ahead | +1.51 [-3.20, +5.17] direction neither | +2.17 [-2.54, +5.85] direction neither |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 345.1 | 442.1 | 494.2 | 376.7 | 598.1 | 467.0 | 526.8 | 625.3 |
| policy | 375.1 | 470.4 | 515.9 | 400.8 | 614.7 | 488.9 | 547.0 | 639.9 |
| specialist | 364.7 | 457.8 | 534.5 | 406.8 | 607.4 | 542.3 | 541.9 | 637.0 |
| best rule | 367.3 | 456.5 | 509.3 | 398.2 | 604.4 | 481.0 | 541.7 | 638.0 |
| best of seven | 367.3 | 455.2 | 505.3 | 398.2 | 598.1 | 479.6 | 537.1 | 625.0 |

## 50x25n30, default (50 modules, 30 instances, mixed test sets)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -1.03 [-6.33, +5.90] not distinguishable | -1.03 [-6.33, +5.90] not distinguishable | +3.13 [-1.45, +11.08] inconclusive / inconclusive, neither | +1.43 [-8.33, +11.03] inconclusive / inconclusive, neither | -3.99 [-5.58, -2.02] ahead | -3.99 [-5.58, -2.02] not distinguishable |
| L | -0.94 [-5.81, +6.51] not distinguishable | -0.59 [-5.49, +6.88] not distinguishable | +3.90 [-1.11, +10.28] inconclusive / inconclusive, neither | +1.25 [-4.14, +8.30] inconclusive / inconclusive, neither | -4.67 [-5.86, -3.43] ahead | -4.33 [-5.55, -3.05] not distinguishable |
| S | -4.11 [-5.69, -2.43] not distinguishable | -3.34 [-4.87, -1.75] not distinguishable | -0.81 [-1.62, +0.14] non-inferior only / non-inferior only, neither | -1.08 [-2.77, +0.32] non-inferior only / non-inferior only, neither | -3.30 [-4.63, -1.98] ahead | -2.52 [-3.79, -1.30] not distinguishable |
| W | -1.93 [-6.04, +3.40] not distinguishable | -1.93 [-6.04, +3.40] not distinguishable | +2.22 [-1.92, +8.75] inconclusive / inconclusive, neither | +2.26 [-6.12, +8.95] inconclusive / inconclusive, neither | -4.03 [-5.11, -2.87] ahead | -4.03 [-5.11, -2.87] not distinguishable |
| LS | -3.65 [-5.15, -1.96] not distinguishable | -2.74 [-4.26, -0.99] not distinguishable | +0.50 [-0.62, +1.62] inconclusive / inconclusive, neither | -0.09 [-0.71, +0.57] inconclusive / equivalent, neither | -4.10 [-5.38, -2.84] ahead | -3.20 [-4.51, -1.91] not distinguishable |
| LW | -1.03 [-5.11, +5.33] not distinguishable | -0.65 [-4.75, +5.75] not distinguishable | +3.49 [-0.86, +9.22] inconclusive / inconclusive, neither | -7.14 [-13.89, -1.70] non-inferior only / non-inferior only, ahead | -4.38 [-5.39, -3.46] ahead | -4.01 [-5.04, -3.07] not distinguishable |
| SW | -2.83 [-4.11, -1.59] not distinguishable | -2.20 [-3.35, -1.01] not distinguishable | -0.08 [-0.71, +0.56] inconclusive / equivalent, neither | -0.69 [-1.94, +0.28] non-inferior only / non-inferior only, neither | -2.74 [-3.94, -1.57] ahead | -2.11 [-3.19, -1.05] not distinguishable |
| LSW | -3.91 [-5.04, -2.72] ahead | -2.76 [-3.78, -1.66] not distinguishable | +0.15 [-0.71, +1.05] inconclusive / inconclusive, neither | -0.07 [-0.72, +0.60] inconclusive / equivalent, neither | -4.04 [-5.18, -2.94] ahead | -2.89 [-3.87, -1.95] not distinguishable |
| pooled | -2.43 [-4.85, +1.33] direction neither | -1.90 [-4.32, +1.87] direction neither | +1.56 [-0.68, +5.15] inconclusive / inconclusive, neither | -0.52 [-2.64, +1.36] inconclusive / inconclusive, neither | -3.91 [-4.82, -3.11] direction ahead | -3.39 [-4.30, -2.59] direction ahead |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 356.6 | 443.0 | 473.7 | 381.9 | 560.8 | 464.7 | 505.4 | 582.0 |
| policy | 345.9 | 426.3 | 477.7 | 373.8 | 558.1 | 449.0 | 505.9 | 581.3 |
| specialist | 352.7 | 437.6 | 479.2 | 374.4 | 561.4 | 504.0 | 509.3 | 582.6 |
| best rule | 360.3 | 447.3 | 494.5 | 389.6 | 582.4 | 469.7 | 520.7 | 606.1 |
| best of seven | 360.3 | 445.7 | 490.5 | 389.6 | 577.0 | 467.9 | 517.2 | 598.8 |

## 80x25n30, mix_rc (80 modules, 30 instances, reinforced-concrete modules alone)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -6.16 [-6.70, -5.61] ahead | -6.16 [-6.70, -5.61] ahead | -6.35 [-11.68, -0.57] non-inferior only / non-inferior only, ahead | -7.37 [-9.08, -5.39] non-inferior only / non-inferior only, ahead | +0.46 [-5.24, +6.25] not distinguishable | +0.46 [-5.24, +6.25] not distinguishable |
| L | -4.74 [-5.30, -4.14] ahead | -4.51 [-5.07, -3.91] ahead | -5.56 [-8.48, -0.67] non-inferior only / non-inferior only, ahead | -3.67 [-7.28, -0.15] non-inferior only / non-inferior only, ahead | +1.02 [-4.35, +4.36] not distinguishable | +1.27 [-4.10, +4.59] not distinguishable |
| S | -4.50 [-5.40, -3.34] ahead | -4.08 [-5.00, -2.91] ahead | -4.87 [-7.80, -0.52] non-inferior only / non-inferior only, ahead | -10.16 [-27.21, -0.79] non-inferior only / non-inferior only, ahead | +0.56 [-4.48, +4.51] not distinguishable | +1.01 [-4.05, +4.98] not distinguishable |
| W | -5.80 [-6.47, -5.12] ahead | -5.80 [-6.47, -5.12] ahead | -5.44 [-10.02, -0.30] non-inferior only / non-inferior only, ahead | -9.09 [-13.29, -2.04] non-inferior only / non-inferior only, ahead | -0.18 [-5.07, +4.79] not distinguishable | -0.18 [-5.07, +4.79] not distinguishable |
| LS | -2.41 [-3.66, -1.22] ahead | -1.27 [-2.25, -0.30] not distinguishable | -2.67 [-5.50, +1.00] inconclusive / inconclusive, neither | -1.29 [-2.10, -0.49] non-inferior only / non-inferior only, ahead | +0.39 [-3.31, +3.15] not distinguishable | +1.57 [-2.22, +4.31] not distinguishable |
| LW | -4.46 [-5.08, -3.73] ahead | -4.24 [-4.87, -3.52] ahead | -4.55 [-6.76, -0.92] non-inferior only / non-inferior only, ahead | -13.00 [-20.56, -2.15] non-inferior only / non-inferior only, ahead | +0.19 [-3.95, +2.88] not distinguishable | +0.42 [-3.73, +3.11] not distinguishable |
| SW | -3.76 [-4.87, -2.65] ahead | -3.27 [-4.39, -2.17] ahead | -4.39 [-7.38, -0.45] non-inferior only / non-inferior only, ahead | -5.63 [-15.98, +0.86] inconclusive / non-inferior only, neither | +0.82 [-3.16, +4.58] not distinguishable | +1.33 [-2.67, +5.10] not distinguishable |
| LSW | -2.31 [-3.35, -1.22] ahead | -1.32 [-2.19, -0.29] not distinguishable | -2.95 [-5.56, -0.02] non-inferior only / non-inferior only, ahead | -3.05 [-6.02, -0.35] non-inferior only / non-inferior only, ahead | +0.75 [-2.29, +3.08] not distinguishable | +1.78 [-1.30, +4.16] not distinguishable |
| pooled | -4.27 [-4.83, -3.68] direction ahead | -3.83 [-4.39, -3.24] direction ahead | -4.60 [-7.36, -0.23] non-inferior only / non-inferior only, ahead | -6.66 [-10.50, -3.19] non-inferior only / non-inferior only, ahead | +0.50 [-4.05, +3.88] direction neither | +0.96 [-3.61, +4.34] direction neither |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 502.6 | 595.4 | 706.3 | 533.9 | 819.6 | 619.8 | 741.3 | 845.6 |
| policy | 538.1 | 631.5 | 743.9 | 565.8 | 843.6 | 650.0 | 776.5 | 872.3 |
| specialist | 543.1 | 619.6 | 806.0 | 589.5 | 830.5 | 719.3 | 792.1 | 873.8 |
| best rule | 535.7 | 625.2 | 739.8 | 567.0 | 840.3 | 648.8 | 770.5 | 866.2 |
| best of seven | 535.7 | 623.7 | 736.5 | 567.0 | 830.3 | 647.3 | 766.6 | 857.3 |

## 80x25n30, default (80 modules, 30 instances, mixed test sets)

| combination | (i) arm - best rule | (i) arm - best of seven | (ii) arm - policy | (iii) arm - specialist | (iv) policy - best rule | (iv) policy - best of seven |
|---|---|---|---|---|---|---|
| N | -0.96 [-5.65, +6.48] not distinguishable | -0.96 [-5.65, +6.48] not distinguishable | +3.09 [-1.96, +11.82] inconclusive / inconclusive, neither | +0.48 [-6.11, +8.54] inconclusive / inconclusive, neither | -3.87 [-5.16, -2.19] ahead | -3.87 [-5.16, -2.19] not distinguishable |
| L | -0.26 [-5.58, +9.17] not distinguishable | -0.14 [-5.46, +9.30] not distinguishable | +4.43 [-1.68, +13.76] inconclusive / inconclusive, neither | +3.31 [-2.74, +12.57] inconclusive / inconclusive, neither | -4.50 [-5.68, -3.46] ahead | -4.38 [-5.57, -3.32] not distinguishable |
| S | -4.12 [-5.32, -2.99] ahead | -3.46 [-4.65, -2.29] ahead | -0.44 [-1.35, +0.44] non-inferior only / non-inferior only, neither | -2.31 [-7.04, +0.98] inconclusive / non-inferior only, neither | -3.67 [-5.12, -2.30] ahead | -3.01 [-4.44, -1.61] not distinguishable |
| W | -2.76 [-5.78, +1.39] not distinguishable | -2.76 [-5.78, +1.39] not distinguishable | +1.89 [-1.30, +7.16] inconclusive / inconclusive, neither | +0.18 [-6.41, +4.42] inconclusive / inconclusive, neither | -4.52 [-5.63, -3.27] ahead | -4.52 [-5.63, -3.27] not distinguishable |
| LS | -3.16 [-4.29, -1.85] ahead | -2.37 [-3.47, -1.10] not distinguishable | +0.29 [-0.65, +1.22] inconclusive / inconclusive, neither | -0.14 [-1.87, +1.78] inconclusive / inconclusive, neither | -3.40 [-4.70, -2.21] ahead | -2.62 [-3.86, -1.47] not distinguishable |
| LW | -1.00 [-5.20, +6.17] not distinguishable | -0.67 [-4.90, +6.49] not distinguishable | +3.45 [-1.15, +10.66] inconclusive / inconclusive, neither | -5.87 [-9.12, -1.67] non-inferior only / non-inferior only, ahead | -4.31 [-5.09, -3.54] ahead | -3.98 [-4.79, -3.22] ahead |
| SW | -4.65 [-5.83, -3.57] ahead | -3.91 [-4.92, -2.98] ahead | -0.69 [-1.31, -0.09] non-inferior only / non-inferior only, ahead | -2.67 [-4.32, -1.26] non-inferior only / non-inferior only, ahead | -3.96 [-5.26, -2.72] ahead | -3.22 [-4.34, -2.10] not distinguishable |
| LSW | -3.55 [-4.54, -2.43] ahead | -2.55 [-3.51, -1.47] not distinguishable | +0.36 [-0.65, +1.40] inconclusive / inconclusive, neither | -0.67 [-1.87, +0.28] non-inferior only / non-inferior only, neither | -3.88 [-4.89, -2.94] ahead | -2.89 [-3.88, -1.96] not distinguishable |
| pooled | -2.56 [-4.97, +1.24] direction neither | -2.10 [-4.52, +1.71] direction neither | +1.55 [-1.01, +5.54] inconclusive / inconclusive, neither | -0.96 [-3.05, +0.64] inconclusive / non-inferior only, neither | -4.01 [-4.91, -3.12] direction ahead | -3.56 [-4.45, -2.66] direction ahead |

| mean makespan (h) | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| arm | 520.5 | 612.9 | 688.8 | 542.7 | 775.8 | 632.1 | 717.3 | 800.8 |
| policy | 505.1 | 586.9 | 692.1 | 532.8 | 773.8 | 611.1 | 722.3 | 797.9 |
| specialist | 518.7 | 593.5 | 706.9 | 542.7 | 777.7 | 673.8 | 737.7 | 807.0 |
| best rule | 525.7 | 614.6 | 719.2 | 558.2 | 801.6 | 638.6 | 753.2 | 830.6 |
| best of seven | 525.7 | 613.9 | 714.3 | 558.2 | 795.0 | 636.5 | 747.1 | 822.0 |

