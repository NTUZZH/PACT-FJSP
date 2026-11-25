# The policy against the best of seven dispatching rules (environment exact, training seeds 301,302,303,304,305)

- d = 100 (policy - rule) / rule per instance, negative = the policy shorter; the policy is the per-instance mean over the training seeds 301, 302, 303, 304, 305 (p72.pact)
- intervals: p72.ci (paired bootstrap, B = 20000, fresh default_rng(7) per quantity); per run: p72.inf_policy_rows (hierarchical, Holm, combined verdict)
- reproduction: 55 numbers of p72's summary for 10x25 identical, including "LLF vs PACT, pooled, 10x25" = +1.92 [+1.71, +2.13] in out/exact/results/aware_rules.md; 175 per-run numbers and verdicts of results/aware_rules_inference.json identical

## 10x25 (100 instances per combination, rules from test_results/FAMILY/10x25_xmask)

| combination | policy, mean | best of seven | best of four | policy vs best of seven, % | policy vs best of four, % | per run vs seven | per run vs four |
|---|---|---|---|---|---|---|---|
| N | 129.7 | 131.9 | 132.3 | -1.62 [-2.06, -1.19] ahead | -1.86 [-2.33, -1.38] ahead | -1.62 [-2.08, -1.16] ahead (ahead) | -1.86 [-2.36, -1.36] ahead (ahead) |
| L | 211.2 | 213.4 | 214.2 | -1.02 [-1.48, -0.56] ahead | -1.36 [-1.88, -0.86] ahead | -1.02 [-1.51, -0.52] ahead (ahead) | -1.36 [-1.91, -0.82] ahead (ahead) |
| S | 193.4 | 192.7 | 194.5 | +0.42 [-0.09, +0.95] neither | -0.52 [-1.04, +0.03] neither | +0.42 [-0.13, +1.01] neither (not distinguishable) | -0.52 [-1.08, +0.08] neither (not distinguishable) |
| W | 147.2 | 149.8 | 150.4 | -1.63 [-2.19, -1.07] ahead | -2.00 [-2.62, -1.39] ahead | -1.63 [-2.24, -1.01] ahead (ahead) | -2.00 [-2.66, -1.33] ahead (ahead) |
| LS | 269.3 | 270.4 | 272.3 | -0.41 [-0.78, -0.04] ahead | -1.10 [-1.48, -0.73] ahead | -0.41 [-0.83, +0.04] neither (not distinguishable) | -1.10 [-1.53, -0.65] ahead (ahead) |
| LW | 220.6 | 224.2 | 225.3 | -1.59 [-2.06, -1.11] ahead | -2.10 [-2.58, -1.64] ahead | -1.59 [-2.13, -1.03] ahead (ahead) | -2.10 [-2.63, -1.57] ahead (ahead) |
| SW | 217.8 | 217.8 | 220.1 | +0.07 [-0.38, +0.50] neither | -0.98 [-1.44, -0.53] ahead | +0.07 [-0.40, +0.53] neither (not distinguishable) | -0.98 [-1.47, -0.50] ahead (ahead) |
| LSW | 286.4 | 286.9 | 290.0 | -0.14 [-0.55, +0.25] neither | -1.21 [-1.63, -0.81] ahead | -0.14 [-0.58, +0.30] neither (not distinguishable) | -1.21 [-1.65, -0.77] ahead (ahead) |
| pooled | 209.5 | 210.9 | 212.4 | -0.74 [-0.91, -0.57] ahead | -1.39 [-1.57, -1.22] ahead | -0.74 [-1.01, -0.48] ahead | -1.39 [-1.66, -1.13] ahead |

Percent of instances on which a rule is the shortest of the seven (ties counted for each rule):

| combination | FIFO | MOR | SPT | MWKR | ECT | SSPT | LLF | ties |
|---|---|---|---|---|---|---|---|---|
| N | 0.0 | 0.0 | 87.0 | 0.0 | 23.0 | 87.0 | 87.0 | 87.0 |
| L | 1.0 | 0.0 | 41.0 | 33.0 | 25.0 | 41.0 | 41.0 | 43.0 |
| S | 1.0 | 2.0 | 46.0 | 1.0 | 23.0 | 42.0 | 46.0 | 48.0 |
| W | 1.0 | 0.0 | 84.0 | 1.0 | 23.0 | 84.0 | 84.0 | 84.0 |
| LS | 4.0 | 2.0 | 25.0 | 14.0 | 30.0 | 20.0 | 27.0 | 19.0 |
| LW | 2.0 | 1.0 | 46.0 | 23.0 | 17.0 | 46.0 | 47.0 | 49.0 |
| SW | 3.0 | 3.0 | 39.0 | 3.0 | 17.0 | 50.0 | 39.0 | 41.0 |
| LSW | 4.0 | 2.0 | 15.0 | 14.0 | 22.0 | 34.0 | 23.0 | 12.0 |
| pooled | 2.0 | 1.2 | 47.9 | 11.1 | 22.5 | 50.5 | 49.2 | 47.9 |

## 20x25 (50 instances per combination, rules from test_results/FAMILY/20x25_xmask)

| combination | policy, mean | best of seven | best of four | policy vs best of seven, % | policy vs best of four, % | per run vs seven | per run vs four |
|---|---|---|---|---|---|---|---|
| N | 184.2 | 188.4 | 188.5 | -2.22 [-2.78, -1.66] ahead | -2.24 [-2.80, -1.68] ahead | -2.22 [-3.11, -1.38] ahead (ahead) | -2.24 [-3.13, -1.40] ahead (ahead) |
| L | 256.9 | 262.3 | 264.0 | -2.02 [-2.53, -1.52] ahead | -2.66 [-3.23, -2.08] ahead | -2.02 [-2.63, -1.43] ahead (ahead) | -2.66 [-3.32, -1.99] ahead (ahead) |
| S | 266.1 | 270.1 | 272.6 | -1.42 [-2.09, -0.75] ahead | -2.29 [-2.99, -1.58] ahead | -1.42 [-2.30, -0.52] ahead (ahead) | -2.29 [-3.19, -1.36] ahead (ahead) |
| W | 210.4 | 216.9 | 216.9 | -2.95 [-3.58, -2.30] ahead | -2.95 [-3.59, -2.31] ahead | -2.95 [-3.89, -2.02] ahead (ahead) | -2.95 [-3.90, -2.03] ahead (ahead) |
| LS | 344.0 | 346.6 | 349.6 | -0.73 [-1.23, -0.26] ahead | -1.57 [-2.08, -1.07] ahead | -0.73 [-1.43, +0.00] neither (not distinguishable) | -1.57 [-2.28, -0.83] ahead (ahead) |
| LW | 275.7 | 283.2 | 285.0 | -2.63 [-3.14, -2.14] ahead | -3.23 [-3.76, -2.71] ahead | -2.63 [-3.21, -2.08] ahead (ahead) | -3.23 [-3.83, -2.65] ahead (ahead) |
| SW | 296.0 | 299.6 | 301.9 | -1.15 [-1.82, -0.45] ahead | -1.88 [-2.57, -1.15] ahead | -1.15 [-1.97, -0.24] ahead (ahead) | -1.88 [-2.72, -0.95] ahead (ahead) |
| LSW | 366.9 | 372.1 | 376.1 | -1.38 [-1.86, -0.92] ahead | -2.41 [-3.02, -1.81] ahead | -1.38 [-2.08, -0.68] ahead (ahead) | -2.41 [-3.19, -1.63] ahead (ahead) |
| pooled | 275.0 | 279.9 | 281.8 | -1.81 [-2.03, -1.60] ahead | -2.41 [-2.63, -2.18] ahead | -1.81 [-2.20, -1.42] ahead | -2.41 [-2.80, -1.98] ahead |

Percent of instances on which a rule is the shortest of the seven (ties counted for each rule):

| combination | FIFO | MOR | SPT | MWKR | ECT | SSPT | LLF | ties |
|---|---|---|---|---|---|---|---|---|
| N | 0.0 | 0.0 | 96.0 | 0.0 | 10.0 | 96.0 | 96.0 | 96.0 |
| L | 0.0 | 0.0 | 46.0 | 0.0 | 14.0 | 46.0 | 48.0 | 48.0 |
| S | 0.0 | 0.0 | 50.0 | 0.0 | 22.0 | 42.0 | 50.0 | 52.0 |
| W | 0.0 | 0.0 | 98.0 | 0.0 | 4.0 | 98.0 | 98.0 | 98.0 |
| LS | 0.0 | 0.0 | 36.0 | 0.0 | 6.0 | 38.0 | 40.0 | 20.0 |
| LW | 0.0 | 0.0 | 50.0 | 0.0 | 6.0 | 50.0 | 54.0 | 54.0 |
| SW | 0.0 | 0.0 | 50.0 | 0.0 | 6.0 | 56.0 | 50.0 | 50.0 |
| LSW | 0.0 | 0.0 | 36.0 | 0.0 | 6.0 | 40.0 | 26.0 | 8.0 |
| pooled | 0.0 | 0.0 | 57.8 | 0.0 | 9.2 | 58.2 | 57.8 | 53.2 |

## 30x25 (20 instances per combination, rules from test_results/FAMILY/30x25_xmask)

| combination | policy, mean | best of seven | best of four | policy vs best of seven, % | policy vs best of four, % | per run vs seven | per run vs four |
|---|---|---|---|---|---|---|---|
| N | 238.5 | 245.8 | 245.9 | -2.95 [-3.83, -2.10] ahead | -2.97 [-3.84, -2.12] ahead | -2.95 [-4.28, -1.66] ahead (ahead) | -2.97 [-4.30, -1.69] ahead (ahead) |
| L | 312.3 | 323.9 | 324.9 | -3.58 [-4.37, -2.83] ahead | -3.86 [-4.65, -3.09] ahead | -3.58 [-4.69, -2.54] ahead (ahead) | -3.86 [-4.96, -2.81] ahead (ahead) |
| S | 338.0 | 345.2 | 349.0 | -2.04 [-2.90, -1.15] ahead | -3.10 [-4.01, -2.18] ahead | -2.04 [-3.11, -0.99] ahead (ahead) | -3.10 [-4.16, -2.04] ahead (ahead) |
| W | 266.8 | 274.9 | 275.2 | -2.93 [-3.58, -2.23] ahead | -3.03 [-3.67, -2.33] ahead | -2.93 [-3.97, -1.81] ahead (ahead) | -3.03 [-4.05, -1.90] ahead (ahead) |
| LS | 415.0 | 421.9 | 427.7 | -1.57 [-2.62, -0.35] ahead | -2.89 [-4.10, -1.54] ahead | -1.57 [-2.88, -0.01] ahead (ahead) | -2.89 [-4.32, -1.21] ahead (ahead) |
| LW | 336.4 | 347.1 | 348.5 | -3.08 [-3.87, -2.25] ahead | -3.46 [-4.35, -2.57] ahead | -3.08 [-4.10, -2.01] ahead (ahead) | -3.46 [-4.55, -2.32] ahead (ahead) |
| SW | 367.9 | 376.8 | 380.6 | -2.28 [-3.20, -1.37] ahead | -3.25 [-4.44, -2.16] ahead | -2.28 [-3.36, -1.13] ahead (ahead) | -3.25 [-4.56, -1.95] ahead (ahead) |
| LSW | 437.5 | 445.1 | 450.2 | -1.66 [-2.42, -0.90] ahead | -2.76 [-3.68, -1.83] ahead | -1.66 [-2.59, -0.72] ahead (ahead) | -2.76 [-3.80, -1.70] ahead (ahead) |
| pooled | 339.0 | 347.6 | 350.2 | -2.51 [-2.84, -2.18] ahead | -3.16 [-3.51, -2.82] ahead | -2.51 [-3.26, -1.78] ahead | -3.16 [-3.90, -2.43] ahead |

Percent of instances on which a rule is the shortest of the seven (ties counted for each rule):

| combination | FIFO | MOR | SPT | MWKR | ECT | SSPT | LLF | ties |
|---|---|---|---|---|---|---|---|---|
| N | 0.0 | 0.0 | 95.0 | 0.0 | 5.0 | 95.0 | 95.0 | 95.0 |
| L | 0.0 | 0.0 | 60.0 | 0.0 | 0.0 | 60.0 | 55.0 | 60.0 |
| S | 0.0 | 0.0 | 45.0 | 0.0 | 15.0 | 50.0 | 45.0 | 45.0 |
| W | 0.0 | 0.0 | 95.0 | 0.0 | 5.0 | 95.0 | 95.0 | 95.0 |
| LS | 0.0 | 0.0 | 25.0 | 0.0 | 5.0 | 50.0 | 40.0 | 20.0 |
| LW | 0.0 | 0.0 | 50.0 | 0.0 | 0.0 | 50.0 | 60.0 | 50.0 |
| SW | 0.0 | 0.0 | 40.0 | 0.0 | 30.0 | 40.0 | 40.0 | 40.0 |
| LSW | 0.0 | 0.0 | 40.0 | 0.0 | 0.0 | 35.0 | 35.0 | 10.0 |
| pooled | 0.0 | 0.0 | 56.2 | 0.0 | 7.5 | 59.4 | 58.1 | 51.9 |

## 50x25n30 (30 instances per combination, rules from test_results/FAMILY/50x25n30_xmask)

| combination | policy, mean | best of seven | best of four | policy vs best of seven, % | policy vs best of four, % | per run vs seven | per run vs four |
|---|---|---|---|---|---|---|---|
| N | 348.6 | 360.3 | 360.3 | -3.23 [-3.84, -2.60] ahead | -3.23 [-3.84, -2.60] ahead | -3.23 [-4.81, -1.43] ahead (ahead) | -3.23 [-4.81, -1.43] ahead (ahead) |
| L | 425.7 | 445.7 | 447.3 | -4.46 [-4.95, -3.95] ahead | -4.80 [-5.22, -4.38] ahead | -4.46 [-5.51, -3.36] ahead (ahead) | -4.80 [-5.83, -3.74] ahead (ahead) |
| S | 475.8 | 490.5 | 494.5 | -2.91 [-3.97, -1.95] ahead | -3.69 [-4.78, -2.65] ahead | -2.91 [-4.13, -1.74] ahead (ahead) | -3.69 [-4.92, -2.43] ahead (ahead) |
| W | 375.7 | 389.6 | 389.6 | -3.54 [-4.32, -2.73] ahead | -3.54 [-4.32, -2.73] ahead | -3.54 [-4.63, -2.30] ahead (ahead) | -3.54 [-4.63, -2.30] ahead (ahead) |
| LS | 557.1 | 577.0 | 582.4 | -3.37 [-4.23, -2.50] ahead | -4.28 [-5.09, -3.44] ahead | -3.37 [-4.46, -2.26] ahead (ahead) | -4.28 [-5.32, -3.21] ahead (ahead) |
| LW | 448.6 | 467.9 | 469.7 | -4.10 [-4.64, -3.57] ahead | -4.47 [-4.99, -3.97] ahead | -4.10 [-5.02, -3.20] ahead (ahead) | -4.47 [-5.38, -3.58] ahead (ahead) |
| SW | 505.4 | 517.2 | 520.7 | -2.21 [-2.94, -1.50] ahead | -2.84 [-3.76, -1.97] ahead | -2.21 [-3.12, -1.29] ahead (ahead) | -2.84 [-3.92, -1.79] ahead (ahead) |
| LSW | 580.3 | 598.8 | 606.1 | -3.06 [-3.73, -2.39] ahead | -4.20 [-5.07, -3.37] ahead | -3.06 [-3.96, -2.21] ahead (ahead) | -4.20 [-5.25, -3.21] ahead (ahead) |
| pooled | 464.7 | 480.9 | 483.8 | -3.36 [-3.63, -3.09] ahead | -3.88 [-4.16, -3.59] ahead | -3.36 [-4.07, -2.69] ahead | -3.88 [-4.60, -3.20] ahead |

Percent of instances on which a rule is the shortest of the seven (ties counted for each rule):

| combination | FIFO | MOR | SPT | MWKR | ECT | SSPT | LLF | ties |
|---|---|---|---|---|---|---|---|---|
| N | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| L | 0.0 | 0.0 | 60.0 | 0.0 | 0.0 | 60.0 | 43.3 | 60.0 |
| S | 0.0 | 0.0 | 46.7 | 0.0 | 0.0 | 60.0 | 46.7 | 46.7 |
| W | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| LS | 0.0 | 0.0 | 36.7 | 0.0 | 0.0 | 46.7 | 20.0 | 3.3 |
| LW | 0.0 | 0.0 | 60.0 | 0.0 | 0.0 | 60.0 | 46.7 | 60.0 |
| SW | 0.0 | 0.0 | 63.3 | 0.0 | 0.0 | 36.7 | 63.3 | 63.3 |
| LSW | 0.0 | 0.0 | 33.3 | 0.0 | 3.3 | 36.7 | 30.0 | 3.3 |
| pooled | 0.0 | 0.0 | 62.5 | 0.0 | 0.4 | 62.5 | 56.2 | 54.6 |

## 80x25n30 (30 instances per combination, rules from test_results/FAMILY/80x25n30_xmask)

| combination | policy, mean | best of seven | best of four | policy vs best of seven, % | policy vs best of four, % | per run vs seven | per run vs four |
|---|---|---|---|---|---|---|---|
| N | 508.8 | 525.7 | 525.7 | -3.18 [-3.88, -2.50] ahead | -3.18 [-3.88, -2.50] ahead | -3.18 [-4.57, -1.57] ahead (ahead) | -3.18 [-4.57, -1.57] ahead (ahead) |
| L | 588.8 | 613.9 | 614.6 | -4.07 [-4.57, -3.57] ahead | -4.19 [-4.68, -3.69] ahead | -4.07 [-5.59, -2.25] ahead (ahead) | -4.19 [-5.70, -2.36] ahead (ahead) |
| S | 690.5 | 714.3 | 719.2 | -3.23 [-4.12, -2.40] ahead | -3.90 [-4.77, -3.04] ahead | -3.23 [-4.34, -2.09] ahead (ahead) | -3.90 [-5.01, -2.76] ahead (ahead) |
| W | 536.8 | 558.2 | 558.2 | -3.81 [-4.48, -3.14] ahead | -3.81 [-4.48, -3.14] ahead | -3.81 [-5.08, -2.36] ahead (ahead) | -3.81 [-5.08, -2.36] ahead (ahead) |
| LS | 771.2 | 795.0 | 801.6 | -2.94 [-3.76, -2.11] ahead | -3.72 [-4.62, -2.82] ahead | -2.94 [-4.07, -1.83] ahead (ahead) | -3.72 [-4.91, -2.57] ahead (ahead) |
| LW | 613.6 | 636.5 | 638.6 | -3.58 [-4.21, -2.94] ahead | -3.90 [-4.50, -3.33] ahead | -3.58 [-5.06, -1.57] ahead (ahead) | -3.90 [-5.38, -1.96] ahead (ahead) |
| SW | 720.3 | 747.1 | 753.2 | -3.49 [-4.34, -2.66] ahead | -4.23 [-5.26, -3.26] ahead | -3.49 [-4.52, -2.48] ahead (ahead) | -4.23 [-5.42, -3.11] ahead (ahead) |
| LSW | 796.2 | 822.0 | 830.6 | -3.11 [-3.70, -2.51] ahead | -4.10 [-4.70, -3.49] ahead | -3.11 [-4.02, -2.24] ahead (ahead) | -4.10 [-5.02, -3.22] ahead (ahead) |
| pooled | 653.3 | 676.6 | 680.2 | -3.43 [-3.68, -3.17] ahead | -3.88 [-4.15, -3.61] ahead | -3.43 [-4.29, -2.50] ahead | -3.88 [-4.74, -2.95] ahead |

Percent of instances on which a rule is the shortest of the seven (ties counted for each rule):

| combination | FIFO | MOR | SPT | MWKR | ECT | SSPT | LLF | ties |
|---|---|---|---|---|---|---|---|---|
| N | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| L | 0.0 | 0.0 | 80.0 | 0.0 | 0.0 | 80.0 | 26.7 | 80.0 |
| S | 0.0 | 0.0 | 53.3 | 0.0 | 3.3 | 43.3 | 53.3 | 53.3 |
| W | 0.0 | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 |
| LS | 0.0 | 0.0 | 43.3 | 0.0 | 6.7 | 30.0 | 26.7 | 6.7 |
| LW | 0.0 | 0.0 | 56.7 | 0.0 | 0.0 | 56.7 | 46.7 | 56.7 |
| SW | 0.0 | 0.0 | 56.7 | 0.0 | 6.7 | 36.7 | 56.7 | 56.7 |
| LSW | 0.0 | 0.0 | 30.0 | 0.0 | 0.0 | 63.3 | 13.3 | 6.7 |
| pooled | 0.0 | 0.0 | 65.0 | 0.0 | 2.1 | 63.8 | 52.9 | 57.5 |

