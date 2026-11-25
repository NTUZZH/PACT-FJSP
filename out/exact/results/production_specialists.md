# Production batches, policy against the specialists (environment exact, training seeds 301,302,303,304,305)

d = 100 (policy - other) / other per training seed and test instance, seed s against seed s; negative = the policy is shorter. B = 20000. Verdicts at the margins 0.5% and 1% from the hierarchical 95% interval; combined verdict against the best dispatching rule with Holm over the eight combinations of a set.

Controls: 224 input arrays with one row per test instance; 6720 makespans equal to the default level of p102 and p103 (p102: policy 40 arrays, specialist 40 arrays, rules 32 arrays; p103: policy 40 arrays, specialist 40 arrays, rules 32 arrays); p102: 24 values of the rule rows (mean and interval ends; Holm family of four there, of eight here) and 56 values of the specialist rows equal within 1e-09 (largest difference 0.0e+00); p103: 24 values of the rule rows (mean and interval ends; Holm family of four there, of eight here) and 56 values of the specialist rows equal within 1e-09 (largest difference 0.0e+00).

## 50x25n30 (30 test instances; specialists in `test_results/FAMILY/50x25n30_xmask_spec`)

| comb. | policy vs specialist | 0.5% | 1% | policy vs best rule | verdict |
|---|---|---|---|---|---|
| N | -0.14 [-4.02, +2.68] | inconclusive | inconclusive | -3.23 [-4.81, -1.43] | ahead |
| L | -2.59 [-4.40, -0.80] | non-inferior only | non-inferior only | -4.80 [-5.83, -3.74] | ahead |
| S | -0.56 [-1.92, +0.67] | inconclusive | non-inferior only | -3.69 [-4.92, -2.43] | ahead |
| W | +1.23 [-1.88, +3.79] | inconclusive | inconclusive | -3.54 [-4.63, -2.30] | ahead |
| LS | -0.51 [-1.53, +0.57] | inconclusive | non-inferior only | -4.28 [-5.32, -3.21] | ahead |
| LW | -8.83 [-15.80, -3.03] | non-inferior only | non-inferior only | -4.47 [-5.38, -3.58] | ahead |
| SW | -1.33 [-3.02, -0.12] | non-inferior only | non-inferior only | -2.84 [-3.92, -1.79] | ahead |
| LSW | -0.60 [-1.91, +0.45] | non-inferior only | non-inferior only | -4.20 [-5.25, -3.21] | ahead |
| pooled All | -1.66 [-3.21, -0.06] | non-inferior only | non-inferior only | -3.88 [-4.60, -3.20] | ahead |
| pooled Train | -1.90 [-3.81, +0.05] | non-inferior only | non-inferior only |  |  |
| pooled Held | -0.96 [-2.41, +0.08] | non-inferior only | non-inferior only |  |  |
| verdict counts (ahead, behind, not distinguishable) | | | | 8, 0, 0 | |

| joint claim (IUT) | holds |
|---|---|
| equivalent on All at +-0.5% | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-0.5% | fails on N,S,W,LS |
| equivalent on All at +-1% | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-1% | fails on N,W |
| equivalent on Train at +-0.5% | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-0.5% | fails on N,S,W,LS |
| equivalent on Train at +-1% | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-1% | fails on N,W |
| equivalent on Held at +-0.5% | fails on SW,LSW |
| non-inferior on Held at +-0.5% | holds |
| equivalent on Held at +-1% | fails on SW,LSW |
| non-inferior on Held at +-1% | holds |

## 80x25n30 (30 test instances; specialists in `test_results/FAMILY/80x25n30_xmask_spec`)

| comb. | policy vs specialist | 0.5% | 1% | policy vs best rule | verdict |
|---|---|---|---|---|---|
| N | -0.75 [-3.53, +1.86] | inconclusive | inconclusive | -3.18 [-4.57, -1.57] | ahead |
| L | -1.36 [-3.84, +0.97] | inconclusive | non-inferior only | -4.19 [-5.70, -2.36] | ahead |
| S | -1.94 [-4.46, +0.28] | non-inferior only | non-inferior only | -3.90 [-5.01, -2.76] | ahead |
| W | +0.44 [-3.53, +4.27] | inconclusive | inconclusive | -3.81 [-5.08, -2.36] | ahead |
| LS | -0.78 [-2.19, +0.91] | inconclusive | non-inferior only | -3.72 [-4.91, -2.57] | ahead |
| LW | -7.47 [-13.11, -2.01] | non-inferior only | non-inferior only | -3.90 [-5.38, -1.96] | ahead |
| SW | -2.17 [-4.00, -0.53] | non-inferior only | non-inferior only | -4.23 [-5.42, -3.11] | ahead |
| LSW | -1.30 [-2.85, -0.00] | non-inferior only | non-inferior only | -4.10 [-5.02, -3.22] | ahead |
| pooled All | -1.92 [-3.93, +0.16] | non-inferior only | non-inferior only | -3.88 [-4.74, -2.95] | ahead |
| pooled Train | -1.98 [-4.38, +0.50] | non-inferior only | non-inferior only |  |  |
| pooled Held | -1.73 [-3.33, -0.34] | non-inferior only | non-inferior only |  |  |
| verdict counts (ahead, behind, not distinguishable) | | | | 8, 0, 0 | |

| joint claim (IUT) | holds |
|---|---|
| equivalent on All at +-0.5% | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-0.5% | fails on N,L,W,LS |
| equivalent on All at +-1% | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-1% | fails on N,W |
| equivalent on Train at +-0.5% | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-0.5% | fails on N,L,W,LS |
| equivalent on Train at +-1% | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-1% | fails on N,W |
| equivalent on Held at +-0.5% | fails on SW,LSW |
| non-inferior on Held at +-0.5% | holds |
| equivalent on Held at +-1% | fails on SW,LSW |
| non-inferior on Held at +-1% | holds |

## Pooled rows by shop size

| modules | instances | policy vs specialist, pooled over eight | 0.5% | 1% | policy vs best rule, pooled over eight | verdict counts (ahead, behind, nd) | source |
|---|---|---|---|---|---|---|---|
| 10 | 100 | +0.03 [-0.18, +0.24] | equivalent | equivalent | -1.39 [-1.66, -1.13] | 7, 0, 1 | out/exact/results/family_inference.json (p88 section A); the pooled rule row computed here by p88.pooled_block from the 10x25 arrays |
| 20 | 50 | -0.61 [-1.12, -0.12] | non-inferior only | non-inferior only | -2.41 [-2.80, -1.98] | 8, 0, 0 | out/exact/results/family_inference.json (p88 section H) |
| 27 | 50 | -1.09 [-1.91, -0.26] | non-inferior only | non-inferior only | -2.79 [-3.29, -2.28] | 8, 0, 0 | out/exact/results/per_floor.json (p118) |
| 30 | 20 | -1.09 [-2.16, -0.05] | non-inferior only | non-inferior only | -3.16 [-3.90, -2.43] | 8, 0, 0 | out/exact/results/family_inference.json (p88 section H) |
| 50 | 30 | -1.66 [-3.21, -0.06] | non-inferior only | non-inferior only | -3.88 [-4.60, -3.20] | 8, 0, 0 | this script |
| 80 | 30 | -1.92 [-3.93, +0.16] | non-inferior only | non-inferior only | -3.88 [-4.74, -2.95] | 8, 0, 0 | this script |

Reproduction of the stored rows from the arrays: 20x25,30x25 1078 numbers; 27x25 534 numbers; 10x25 202 numbers, all within 1e-09.
