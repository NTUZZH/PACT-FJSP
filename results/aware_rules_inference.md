# Three further dispatching rules: the protocol of the manuscript (scripts/p72_aware_rules.py, inference step)

Environment exact, training seeds 301,302,303,304,305, B = 20000. Gaps in %: first-named method minus second over second; negative = first-named method shorter. Policy against a rule quantity: crossed hierarchical bootstrap over the policy's runs and the base instances, fresh default_rng(11); combined verdict with Holm over the eight combinations of the size (pooled row: the interval and its direction). Best of the three further rules against the best of the four classical rules: bootstrap over base instances, fresh default_rng(7), Holm per direction over the eight combinations (pooled row: the interval and its direction).

## 10x25 (100 instances per combination, 5 runs of the policy)

| combination | ECT | SSPT | LLF | FIFO | MOR | SPT | MWKR | policy |
|---|---|---|---|---|---|---|---|---|
| N | 135.7 | 132.3 | 132.3 | 148.7 | 165.1 | 132.3 | 157.9 | 129.7 |
| L | 217.1 | 216.1 | 215.9 | 224.9 | 228.6 | 216.1 | 218.3 | 211.2 |
| S | 198.0 | 195.0 | 194.8 | 208.6 | 213.4 | 194.8 | 210.0 | 193.4 |
| W | 154.7 | 150.4 | 150.4 | 169.7 | 186.3 | 150.4 | 179.0 | 147.2 |
| LS | 276.7 | 274.9 | 274.4 | 284.3 | 286.2 | 274.1 | 281.1 | 269.3 |
| LW | 229.4 | 227.2 | 226.8 | 238.8 | 243.1 | 227.2 | 231.9 | 220.6 |
| SW | 225.7 | 220.3 | 220.9 | 234.0 | 238.5 | 220.9 | 235.4 | 217.8 |
| LSW | 295.3 | 291.8 | 292.1 | 302.6 | 306.0 | 292.1 | 300.1 | 286.4 |
| pool | 216.6 | 213.5 | 213.4 | 226.5 | 233.4 | 213.5 | 226.7 | 209.5 |

| combination | policy vs ECT | policy vs SSPT | policy vs LLF | policy vs the best of ECT, SSPT and LLF | policy vs the best of the seven rules | best3 vs best4 |
|---|---|---|---|---|---|---|
| N | -4.35 [-4.89, -3.82] ahead | -1.86 [-2.37, -1.36] ahead | -1.86 [-2.37, -1.36] ahead | -1.62 [-2.08, -1.16] ahead | -1.62 [-2.08, -1.16] ahead | -0.24 [-0.42, -0.10] ahead |
| L | -2.69 [-3.25, -2.12] ahead | -2.22 [-2.88, -1.58] ahead | -2.12 [-2.76, -1.51] ahead | -1.45 [-1.98, -0.91] ahead | -1.02 [-1.51, -0.52] ahead | +0.10 [-0.18, +0.39] not distinguishable |
| S | -2.24 [-2.92, -1.55] ahead | -0.79 [-1.33, -0.22] ahead | -0.63 [-1.21, -0.01] ahead | +0.40 [-0.14, +0.99] not distinguishable | +0.42 [-0.13, +1.01] not distinguishable | -0.91 [-1.19, -0.65] ahead |
| W | -4.72 [-5.48, -3.97] ahead | -2.04 [-2.70, -1.37] ahead | -2.04 [-2.70, -1.37] ahead | -1.67 [-2.28, -1.05] ahead | -1.63 [-2.24, -1.01] ahead | -0.34 [-0.59, -0.13] ahead |
| LS | -2.63 [-3.28, -2.00] ahead | -2.00 [-2.52, -1.48] ahead | -1.85 [-2.34, -1.38] ahead | -0.82 [-1.27, -0.36] ahead | -0.41 [-0.83, +0.04] not distinguishable | -0.27 [-0.57, +0.04] not distinguishable |
| LW | -3.80 [-4.49, -3.11] ahead | -2.89 [-3.53, -2.25] ahead | -2.69 [-3.35, -2.03] ahead | -2.02 [-2.62, -1.40] ahead | -1.59 [-2.13, -1.03] ahead | -0.06 [-0.42, +0.37] not distinguishable |
| SW | -3.37 [-3.97, -2.78] ahead | -1.07 [-1.63, -0.52] ahead | -1.32 [-1.83, -0.81] ahead | -0.07 [-0.56, +0.41] not distinguishable | +0.07 [-0.40, +0.53] not distinguishable | -0.89 [-1.21, -0.56] ahead |
| LSW | -2.94 [-3.63, -2.25] ahead | -1.79 [-2.38, -1.20] ahead | -1.87 [-2.47, -1.30] ahead | -0.59 [-1.13, -0.07] ahead | -0.14 [-0.58, +0.30] not distinguishable | -0.59 [-0.99, -0.16] ahead |
| pool | -3.34 [-3.71, -3.00] direction ahead | -1.83 [-2.13, -1.54] direction ahead | -1.80 [-2.08, -1.52] direction ahead | -0.98 [-1.25, -0.72] direction ahead | -0.74 [-1.01, -0.48] direction ahead | -0.40 [-0.52, -0.27] direction ahead |

## 20x25 (50 instances per combination, 5 runs of the policy)

| combination | ECT | SSPT | LLF | FIFO | MOR | SPT | MWKR | policy |
|---|---|---|---|---|---|---|---|---|
| N | 197.1 | 188.5 | 188.5 | 248.2 | 301.1 | 188.5 | 280.9 | 184.2 |
| L | 267.7 | 264.0 | 264.4 | 305.5 | 334.5 | 264.0 | 319.5 | 256.9 |
| S | 280.0 | 273.1 | 272.6 | 315.5 | 341.2 | 272.6 | 325.4 | 266.1 |
| W | 225.7 | 216.9 | 216.9 | 272.3 | 325.5 | 216.9 | 299.2 | 210.4 |
| LS | 359.6 | 349.8 | 349.4 | 385.5 | 401.2 | 349.6 | 393.0 | 344.0 |
| LW | 292.2 | 285.0 | 285.6 | 327.1 | 351.8 | 285.0 | 343.8 | 275.7 |
| SW | 313.4 | 302.6 | 301.9 | 344.2 | 365.1 | 301.9 | 352.0 | 296.0 |
| LSW | 384.8 | 376.1 | 376.5 | 409.0 | 422.5 | 376.1 | 413.0 | 366.9 |
| pool | 290.0 | 282.0 | 282.0 | 325.9 | 355.4 | 281.8 | 340.9 | 275.0 |

| combination | policy vs ECT | policy vs SSPT | policy vs LLF | policy vs the best of ECT, SSPT and LLF | policy vs the best of the seven rules | best3 vs best4 |
|---|---|---|---|---|---|---|
| N | -6.48 [-7.42, -5.59] ahead | -2.24 [-3.13, -1.40] ahead | -2.24 [-3.13, -1.40] ahead | -2.22 [-3.11, -1.38] ahead | -2.22 [-3.11, -1.38] ahead | -0.02 [-0.05, +0.00] not distinguishable |
| L | -3.99 [-4.59, -3.40] ahead | -2.66 [-3.32, -1.99] ahead | -2.82 [-3.48, -2.18] ahead | -2.02 [-2.63, -1.43] ahead | -2.02 [-2.63, -1.43] ahead | -0.65 [-0.95, -0.40] ahead |
| S | -4.90 [-5.93, -3.90] ahead | -2.45 [-3.38, -1.48] ahead | -2.29 [-3.19, -1.36] ahead | -1.42 [-2.30, -0.52] ahead | -1.42 [-2.30, -0.52] ahead | -0.88 [-1.29, -0.52] ahead |
| W | -6.70 [-7.71, -5.67] ahead | -2.95 [-3.90, -2.03] ahead | -2.95 [-3.90, -2.03] ahead | -2.95 [-3.89, -2.02] ahead | -2.95 [-3.89, -2.02] ahead | -0.01 [-0.03, +0.00] not distinguishable |
| LS | -4.30 [-5.12, -3.45] ahead | -1.64 [-2.34, -0.89] ahead | -1.52 [-2.24, -0.78] ahead | -0.96 [-1.65, -0.24] ahead | -0.73 [-1.43, +0.00] not distinguishable | -0.62 [-0.98, -0.26] ahead |
| LW | -5.62 [-6.26, -5.00] ahead | -3.24 [-3.85, -2.66] ahead | -3.42 [-4.06, -2.81] ahead | -2.63 [-3.21, -2.08] ahead | -2.63 [-3.21, -2.08] ahead | -0.61 [-0.85, -0.40] ahead |
| SW | -5.44 [-6.46, -4.35] ahead | -2.11 [-2.98, -1.18] ahead | -1.88 [-2.72, -0.95] ahead | -1.15 [-1.97, -0.24] ahead | -1.15 [-1.97, -0.24] ahead | -0.73 [-1.03, -0.46] ahead |
| LSW | -4.63 [-5.39, -3.88] ahead | -2.43 [-3.14, -1.70] ahead | -2.55 [-3.22, -1.87] ahead | -1.70 [-2.36, -1.01] ahead | -1.38 [-2.08, -0.68] ahead | -0.72 [-1.20, -0.27] ahead |
| pool | -5.26 [-5.70, -4.82] direction ahead | -2.47 [-2.85, -2.06] direction ahead | -2.46 [-2.85, -2.05] direction ahead | -1.88 [-2.27, -1.48] direction ahead | -1.81 [-2.20, -1.42] direction ahead | -0.53 [-0.66, -0.41] direction ahead |

## 30x25 (20 instances per combination, 5 runs of the policy)

| combination | ECT | SSPT | LLF | FIFO | MOR | SPT | MWKR | policy |
|---|---|---|---|---|---|---|---|---|
| N | 257.8 | 245.9 | 245.9 | 341.0 | 438.9 | 245.9 | 378.8 | 238.5 |
| L | 334.3 | 324.9 | 324.9 | 391.8 | 473.2 | 324.9 | 423.0 | 312.2 |
| S | 356.6 | 348.6 | 349.0 | 436.1 | 492.8 | 349.0 | 456.1 | 338.0 |
| W | 286.1 | 275.2 | 275.2 | 368.6 | 460.4 | 275.2 | 400.9 | 266.8 |
| LS | 445.0 | 425.8 | 424.6 | 483.3 | 529.3 | 427.7 | 498.2 | 415.0 |
| LW | 359.6 | 348.5 | 348.5 | 413.2 | 489.1 | 348.5 | 445.9 | 336.4 |
| SW | 388.1 | 380.2 | 380.6 | 446.7 | 507.1 | 380.6 | 476.2 | 367.9 |
| LSW | 466.9 | 450.4 | 450.9 | 503.7 | 554.5 | 450.2 | 511.1 | 437.5 |
| pool | 361.8 | 349.9 | 349.9 | 423.1 | 493.2 | 350.2 | 448.8 | 339.0 |

| combination | policy vs ECT | policy vs SSPT | policy vs LLF | policy vs the best of ECT, SSPT and LLF | policy vs the best of the seven rules | best3 vs best4 |
|---|---|---|---|---|---|---|
| N | -7.43 [-8.77, -6.12] ahead | -2.97 [-4.30, -1.69] ahead | -2.97 [-4.30, -1.69] ahead | -2.95 [-4.28, -1.66] ahead | -2.95 [-4.28, -1.66] ahead | -0.02 [-0.06, +0.00] not distinguishable |
| L | -6.58 [-7.67, -5.53] ahead | -3.86 [-4.96, -2.81] ahead | -3.86 [-4.98, -2.83] ahead | -3.58 [-4.69, -2.54] ahead | -3.58 [-4.69, -2.54] ahead | -0.29 [-0.51, -0.11] ahead |
| S | -5.20 [-6.47, -3.94] ahead | -2.95 [-4.05, -1.80] ahead | -3.10 [-4.16, -2.04] ahead | -2.04 [-3.11, -0.99] ahead | -2.04 [-3.11, -0.99] ahead | -1.06 [-2.09, -0.33] ahead |
| W | -6.71 [-7.92, -5.48] ahead | -3.03 [-4.05, -1.90] ahead | -3.03 [-4.05, -1.90] ahead | -2.93 [-3.97, -1.81] ahead | -2.93 [-3.97, -1.81] ahead | -0.10 [-0.31, +0.00] not distinguishable |
| LS | -6.68 [-8.15, -5.02] ahead | -2.47 [-3.78, -0.90] ahead | -2.21 [-3.58, -0.59] ahead | -1.61 [-2.94, -0.07] ahead | -1.57 [-2.88, -0.01] ahead | -1.30 [-1.98, -0.69] ahead |
| LW | -6.42 [-7.60, -5.21] ahead | -3.46 [-4.55, -2.32] ahead | -3.47 [-4.51, -2.40] ahead | -3.08 [-4.10, -2.01] ahead | -3.08 [-4.10, -2.01] ahead | -0.40 [-0.66, -0.17] ahead |
| SW | -5.12 [-6.39, -3.79] ahead | -3.16 [-4.34, -1.92] ahead | -3.25 [-4.56, -1.95] ahead | -2.28 [-3.36, -1.13] ahead | -2.28 [-3.36, -1.13] ahead | -1.00 [-1.68, -0.46] ahead |
| LSW | -6.25 [-7.51, -4.93] ahead | -2.80 [-3.79, -1.81] ahead | -2.90 [-4.11, -1.72] ahead | -2.25 [-3.27, -1.22] ahead | -1.66 [-2.59, -0.72] ahead | -0.50 [-1.39, +0.39] not distinguishable |
| pool | -6.30 [-7.15, -5.47] direction ahead | -3.09 [-3.82, -2.34] direction ahead | -3.10 [-3.85, -2.36] direction ahead | -2.59 [-3.35, -1.84] direction ahead | -2.51 [-3.26, -1.78] direction ahead | -0.58 [-0.79, -0.39] direction ahead |

## 50x25n30 (30 instances per combination, 5 runs of the policy)

| combination | ECT | SSPT | LLF | FIFO | MOR | SPT | MWKR | policy |
|---|---|---|---|---|---|---|---|---|
| N | 382.2 | 360.3 | 360.3 | 552.0 | 759.3 | 360.3 | 650.2 | 348.6 |
| L | 464.7 | 447.3 | 447.9 | 582.7 | 794.2 | 447.3 | 686.3 | 425.7 |
| S | 523.7 | 492.3 | 494.5 | 646.5 | 730.6 | 494.5 | 678.7 | 475.8 |
| W | 410.4 | 389.6 | 389.6 | 572.1 | 772.9 | 389.6 | 664.5 | 375.7 |
| LS | 619.4 | 581.8 | 581.9 | 701.3 | 781.3 | 582.4 | 720.9 | 557.1 |
| LW | 486.0 | 469.7 | 470.5 | 600.7 | 805.8 | 469.7 | 707.5 | 448.6 |
| SW | 553.6 | 521.4 | 520.7 | 669.9 | 756.8 | 520.7 | 702.3 | 505.4 |
| LSW | 637.7 | 603.9 | 606.9 | 721.9 | 797.0 | 606.1 | 738.7 | 580.3 |
| pool | 509.7 | 483.3 | 484.0 | 630.9 | 774.7 | 483.8 | 693.6 | 464.7 |

| combination | policy vs ECT | policy vs SSPT | policy vs LLF | policy vs the best of ECT, SSPT and LLF | policy vs the best of the seven rules | best3 vs best4 |
|---|---|---|---|---|---|---|
| N | -8.74 [-10.30, -6.94] ahead | -3.23 [-4.81, -1.43] ahead | -3.23 [-4.81, -1.43] ahead | -3.23 [-4.81, -1.43] ahead | -3.23 [-4.81, -1.43] ahead | +0.00 [+0.00, +0.00] not distinguishable |
| L | -8.36 [-9.43, -7.28] ahead | -4.80 [-5.83, -3.74] ahead | -4.91 [-6.00, -3.79] ahead | -4.46 [-5.51, -3.36] ahead | -4.46 [-5.51, -3.36] ahead | -0.35 [-0.60, -0.15] ahead |
| S | -9.08 [-10.06, -8.11] ahead | -3.25 [-4.47, -2.08] ahead | -3.69 [-4.92, -2.43] ahead | -2.91 [-4.13, -1.74] ahead | -2.91 [-4.13, -1.74] ahead | -0.79 [-1.34, -0.35] ahead |
| W | -8.44 [-9.39, -7.44] ahead | -3.54 [-4.63, -2.30] ahead | -3.54 [-4.63, -2.30] ahead | -3.54 [-4.63, -2.30] ahead | -3.54 [-4.63, -2.30] ahead | +0.00 [+0.00, +0.00] not distinguishable |
| LS | -10.04 [-10.96, -9.10] ahead | -4.14 [-5.35, -2.92] ahead | -4.17 [-5.26, -3.06] ahead | -3.68 [-4.80, -2.54] ahead | -3.37 [-4.46, -2.26] ahead | -0.61 [-1.10, -0.11] ahead |
| LW | -7.68 [-8.55, -6.82] ahead | -4.47 [-5.38, -3.58] ahead | -4.62 [-5.57, -3.70] ahead | -4.10 [-5.02, -3.20] ahead | -4.10 [-5.02, -3.20] ahead | -0.39 [-0.61, -0.19] ahead |
| SW | -8.65 [-9.51, -7.80] ahead | -3.03 [-3.93, -2.13] ahead | -2.84 [-3.92, -1.79] ahead | -2.21 [-3.12, -1.29] ahead | -2.21 [-3.12, -1.29] ahead | -0.65 [-1.15, -0.22] ahead |
| LSW | -8.98 [-9.78, -8.19] ahead | -3.88 [-4.76, -3.06] ahead | -4.34 [-5.30, -3.42] ahead | -3.32 [-4.18, -2.49] ahead | -3.06 [-3.96, -2.21] ahead | -0.92 [-1.50, -0.34] ahead |
| pool | -8.75 [-9.40, -8.13] direction ahead | -3.79 [-4.51, -3.12] direction ahead | -3.92 [-4.61, -3.26] direction ahead | -3.43 [-4.14, -2.76] direction ahead | -3.36 [-4.07, -2.69] direction ahead | -0.46 [-0.59, -0.34] direction ahead |

## 80x25n30 (30 instances per combination, 5 runs of the policy)

| combination | ECT | SSPT | LLF | FIFO | MOR | SPT | MWKR | policy |
|---|---|---|---|---|---|---|---|---|
| N | 558.1 | 525.7 | 525.7 | 833.3 | 1179.2 | 525.7 | 1002.9 | 508.8 |
| L | 641.0 | 614.6 | 617.4 | 860.3 | 1200.8 | 614.6 | 1050.6 | 588.8 |
| S | 760.3 | 720.3 | 719.2 | 975.1 | 1121.2 | 719.2 | 1056.4 | 690.5 |
| W | 588.2 | 558.2 | 558.2 | 837.2 | 1186.5 | 558.2 | 1019.4 | 536.8 |
| LS | 853.7 | 804.5 | 803.1 | 1010.1 | 1151.9 | 801.6 | 1068.5 | 771.2 |
| LW | 664.1 | 638.6 | 639.8 | 876.5 | 1217.3 | 638.6 | 1076.0 | 613.6 |
| SW | 795.6 | 754.7 | 753.2 | 1005.3 | 1146.4 | 753.2 | 1071.4 | 720.3 |
| LSW | 880.9 | 826.0 | 832.2 | 1031.3 | 1172.5 | 830.6 | 1090.9 | 796.2 |
| pool | 717.7 | 680.3 | 681.1 | 928.6 | 1172.0 | 680.2 | 1054.5 | 653.3 |

| combination | policy vs ECT | policy vs SSPT | policy vs LLF | policy vs the best of ECT, SSPT and LLF | policy vs the best of the seven rules | best3 vs best4 |
|---|---|---|---|---|---|---|
| N | -8.82 [-10.07, -7.38] ahead | -3.18 [-4.57, -1.57] ahead | -3.18 [-4.57, -1.57] ahead | -3.18 [-4.57, -1.57] ahead | -3.18 [-4.57, -1.57] ahead | +0.00 [+0.00, +0.00] not distinguishable |
| L | -8.14 [-9.63, -6.42] ahead | -4.19 [-5.70, -2.36] ahead | -4.61 [-6.13, -2.80] ahead | -4.07 [-5.59, -2.25] ahead | -4.07 [-5.59, -2.25] ahead | -0.12 [-0.23, -0.03] ahead |
| S | -9.15 [-10.17, -8.07] ahead | -4.03 [-5.17, -2.83] ahead | -3.90 [-5.01, -2.76] ahead | -3.23 [-4.34, -2.09] ahead | -3.23 [-4.34, -2.09] ahead | -0.68 [-1.21, -0.26] ahead |
| W | -8.72 [-9.89, -7.37] ahead | -3.81 [-5.08, -2.36] ahead | -3.81 [-5.08, -2.36] ahead | -3.81 [-5.08, -2.36] ahead | -3.81 [-5.08, -2.36] ahead | +0.00 [+0.00, +0.00] not distinguishable |
| LS | -9.63 [-10.62, -8.66] ahead | -4.05 [-5.31, -2.82] ahead | -3.89 [-5.16, -2.68] ahead | -3.34 [-4.53, -2.18] ahead | -2.94 [-4.07, -1.83] ahead | -0.38 [-0.93, +0.18] not distinguishable |
| LW | -7.59 [-9.03, -5.75] ahead | -3.90 [-5.38, -1.96] ahead | -4.06 [-5.55, -2.04] ahead | -3.58 [-5.06, -1.57] ahead | -3.58 [-5.06, -1.57] ahead | -0.33 [-0.56, -0.15] ahead |
| SW | -9.41 [-10.32, -8.49] ahead | -4.44 [-5.53, -3.39] ahead | -4.23 [-5.42, -3.11] ahead | -3.49 [-4.52, -2.48] ahead | -3.49 [-4.52, -2.48] ahead | -0.77 [-1.28, -0.34] ahead |
| LSW | -9.57 [-10.46, -8.65] ahead | -3.57 [-4.53, -2.63] ahead | -4.28 [-5.29, -3.29] ahead | -3.34 [-4.30, -2.42] ahead | -3.11 [-4.02, -2.24] ahead | -0.78 [-1.27, -0.27] ahead |
| pool | -8.88 [-9.69, -7.98] direction ahead | -3.90 [-4.76, -2.97] direction ahead | -4.00 [-4.86, -3.06] direction ahead | -3.51 [-4.37, -2.57] direction ahead | -3.43 [-4.29, -2.50] direction ahead | -0.38 [-0.50, -0.26] direction ahead |

