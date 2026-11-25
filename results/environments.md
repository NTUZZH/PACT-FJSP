# The environment of the paper against the prior implementation (10x25)

Training seeds 301, 302, 303 in both environments, seed s against seed s; 100 test instances per combination; B = 20000. d = 100 (paper - prior) / prior, negative = the environment of the paper gives the shorter schedule.

## Policy, specialists and single-rule control (hierarchical 95% CI)

| comb. | released policy | specialist (own comb.) | single-rule control |
|---|---|---|---|
| N | -0.32 [-0.78, +0.15] neither | -0.76 [-1.26, -0.24] ahead | -0.63 [-1.03, -0.21] ahead |
| L | -0.32 [-0.74, +0.05] neither | -0.16 [-0.58, +0.27] neither | +0.04 [-0.26, +0.36] neither |
| S | -0.18 [-0.72, +0.36] neither | -0.18 [-0.56, +0.22] neither | -0.23 [-0.61, +0.15] neither |
| W | +0.16 [-0.58, +0.79] neither | +0.39 [-0.43, +1.31] neither | -0.76 [-1.58, -0.01] ahead |
| LS | -0.16 [-0.57, +0.30] neither | -0.16 [-0.49, +0.18] neither | -0.10 [-0.57, +0.35] neither |
| LW | -0.48 [-0.89, -0.08] ahead | -0.55 [-1.11, -0.02] ahead | -0.03 [-0.36, +0.31] neither |
| SW | -0.28 [-0.78, +0.23] neither | -0.13 [-0.84, +0.60] neither | -0.22 [-0.80, +0.39] neither |
| LSW | -0.08 [-0.53, +0.35] neither | -0.30 [-0.69, +0.08] neither | -0.32 [-0.75, +0.14] neither |

released policy, pooled All: -0.21 [-0.51, +0.08] (neither)
released policy, pooled Train: -0.22 [-0.55, +0.12] (neither)
released policy, pooled Held: -0.18 [-0.57, +0.21] (neither)

specialist of the combination, on its own combination, pooled All: -0.23 [-0.40, -0.05] (ahead)
specialist of the combination, on its own combination, pooled Train: -0.23 [-0.50, +0.01] (neither)
specialist of the combination, on its own combination, pooled Held: -0.22 [-0.66, +0.23] (neither)

single-rule control, pooled All: -0.28 [-0.57, -0.01] (ahead)
single-rule control, pooled Seen: -0.40 [-0.72, -0.07] (ahead)
single-rule control, pooled Unseen: -0.17 [-0.48, +0.14] (neither)

## Dispatching rules (instance-level cluster 95% CI)

| comb. | FIFO | MOR | SPT | MWKR | per-instance best |
|---|---|---|---|---|---|
| N | -3.66 [-4.25, -3.07] | -1.19 [-2.09, -0.25] | -0.26 [-0.63, +0.11] | -2.18 [-3.15, -1.22] | -0.27 [-0.64, +0.10] |
| L | -2.69 [-3.06, -2.32] | -2.20 [-2.63, -1.74] | -0.07 [-0.34, +0.22] | -2.89 [-3.26, -2.53] | -0.71 [-1.02, -0.41] |
| S | -9.24 [-10.28, -8.22] | -17.43 [-19.41, -15.47] | -0.42 [-0.76, -0.07] | -13.81 [-15.26, -12.36] | -0.53 [-0.87, -0.21] |
| W | -5.66 [-6.65, -4.66] | -2.74 [-4.07, -1.36] | -0.33 [-0.79, +0.14] | -2.82 [-4.19, -1.47] | -0.36 [-0.83, +0.11] |
| LS | -5.79 [-6.68, -4.93] | -8.72 [-9.71, -7.71] | -0.45 [-0.76, -0.16] | -8.14 [-9.16, -7.14] | -1.03 [-1.42, -0.67] |
| LW | -3.53 [-4.21, -2.89] | -2.51 [-3.29, -1.71] | -0.12 [-0.40, +0.15] | -3.06 [-3.74, -2.37] | -0.60 [-0.93, -0.27] |
| SW | -9.81 [-10.80, -8.81] | -15.43 [-17.14, -13.72] | -0.57 [-1.12, -0.05] | -12.69 [-13.95, -11.45] | -0.92 [-1.49, -0.38] |
| LSW | -7.62 [-8.52, -6.74] | -9.27 [-10.27, -8.26] | -0.14 [-0.58, +0.27] | -9.40 [-10.34, -8.45] | -0.82 [-1.34, -0.35] |
| pooled | -6.00 [-6.37, -5.63] | -7.44 [-7.99, -6.89] | -0.29 [-0.44, -0.15] | -6.87 [-7.35, -6.40] | -0.66 [-0.83, -0.49] |

## Verdict tables side by side: released policy vs specialist

| comb. | prior implementation | 0.5% | 1% | dir. | environment of the paper | 0.5% | 1% | dir. |
|---|---|---|---|---|---|---|---|---|
| N | -0.39 [-0.78, -0.00] | non-inferior only | equivalent | ahead | +0.08 [-0.61, +0.66] | inconclusive | equivalent | neither |
| L | +0.25 [-0.03, +0.55] | inconclusive | equivalent | neither | +0.09 [-0.24, +0.42] | equivalent | equivalent | neither |
| S | -0.10 [-0.65, +0.43] | non-inferior only | equivalent | neither | -0.11 [-0.47, +0.26] | equivalent | equivalent | neither |
| W | +0.57 [-0.16, +1.29] | inconclusive | inconclusive | neither | +0.36 [-0.83, +1.47] | inconclusive | inconclusive | neither |
| LS | -0.09 [-0.46, +0.27] | equivalent | equivalent | neither | -0.09 [-0.53, +0.36] | non-inferior only | equivalent | neither |
| LW | +0.18 [-0.36, +0.68] | inconclusive | equivalent | neither | +0.24 [-0.15, +0.63] | inconclusive | equivalent | neither |
| SW | -0.33 [-1.13, +0.46] | non-inferior only | non-inferior only | neither | -0.46 [-1.06, +0.10] | non-inferior only | non-inferior only | neither |
| LSW | -0.20 [-0.67, +0.25] | non-inferior only | equivalent | neither | +0.02 [-0.35, +0.42] | equivalent | equivalent | neither |
| pooled All | -0.01 [-0.22, +0.21] | equivalent | equivalent | neither | +0.02 [-0.23, +0.27] | equivalent | equivalent | neither |
| pooled Train | +0.07 [-0.11, +0.26] | equivalent | equivalent | neither | +0.09 [-0.18, +0.37] | equivalent | equivalent | neither |
| pooled Held | -0.27 [-0.81, +0.26] | non-inferior only | equivalent | neither | -0.22 [-0.60, +0.16] | non-inferior only | equivalent | neither |

Verdict differs at 0.5%: N, L, S, LS, LSW; at 1%: none; direction differs: N.
