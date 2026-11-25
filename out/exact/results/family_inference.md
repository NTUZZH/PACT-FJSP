# Family inference (environment exact, training seeds 301,302,303,304,305 (5 runs per arm))

B = 20000; verdicts at +-0.5% / +-1% from the hierarchical 95% interval; IUT for joint claims; Holm for superiority. Reproduction of p27: 56 macros identical.

## A. Released policy vs specialist (10x25)

| comb. | mean | hier. 95% CI | Bonf. CI | seed-t CI | 0.5% | 1% | direction | per seed |
|---|---|---|---|---|---|---|---|---|
| N | +0.24 | [-0.40, +0.86] | [-0.68, +1.11] | [-0.54, +1.03] | inconclusive | equivalent | neither | +0.41, -0.49, +0.32, -0.19, +1.16 |
| L | +0.18 | [-0.11, +0.49] | [-0.23, +0.63] | [-0.06, +0.43] | equivalent | equivalent | neither | -0.00, +0.18, +0.09, +0.51, +0.14 |
| S | -0.35 | [-0.80, +0.06] | [-1.01, +0.22] | [-0.90, +0.19] | non-inferior only | equivalent | neither | -0.25, +0.02, -0.10, -1.10, -0.33 |
| W | +0.40 | [-0.51, +1.31] | [-0.89, +1.65] | [-0.82, +1.61] | inconclusive | inconclusive | neither | -0.96, +0.99, +1.06, -0.32, +1.21 |
| LS | -0.07 | [-0.38, +0.25] | [-0.52, +0.41] | [-0.39, +0.25] | equivalent | equivalent | neither | -0.36, -0.24, +0.32, +0.01, -0.09 |
| LW | +0.46 | [+0.06, +0.88] | [-0.11, +1.07] | [+0.00, +0.91] | inferior | equivalent | behind | +0.22, +0.44, +0.06, +1.01, +0.57 |
| SW | -0.47 | [-0.89, -0.08] | [-1.09, +0.09] | [-0.71, -0.23] | non-inferior only | equivalent | ahead | -0.62, -0.61, -0.15, -0.55, -0.42 |
| LSW | -0.15 | [-0.53, +0.22] | [-0.73, +0.40] | [-0.48, +0.19] | non-inferior only | equivalent | neither | +0.02, -0.18, +0.22, -0.41, -0.39 |

| family | margin | equivalent on all | non-inferior on all | failing (eq.) |
|---|---|---|---|---|
| All | 0.5 | False | False | N,S,W,LW,SW,LSW |
| All | 1.0 | False | False | W |
| Train | 0.5 | False | False | N,S,W,LW |
| Train | 1.0 | False | False | W |
| Held | 0.5 | False | True | SW,LSW |
| Held | 1.0 | True | True | - |

Published instance-level intervals (p11, macros_eq.tex) under the same rule:

| comb. | published CI | 0.5% | 1% | direction |
|---|---|---|---|---|
| N | [-0.14, +0.55] | inconclusive | equivalent | neither |
| L | [-0.02, +0.35] | equivalent | equivalent | neither |
| S | [-0.59, -0.17] | non-inferior only | equivalent | ahead |
| W | [-0.07, +0.73] | inconclusive | equivalent | neither |
| LS | [-0.26, +0.08] | equivalent | equivalent | neither |
| LW | [+0.22, +0.67] | inferior | equivalent | behind |
| SW | [-0.79, -0.25] | non-inferior only | equivalent | ahead |
| LSW | [-0.42, +0.08] | equivalent | equivalent | neither |

| family | margin | equivalent on all | non-inferior on all | failing (eq.) |
|---|---|---|---|---|
| All | 0.5 | False | False | N,S,W,LW,SW |
| All | 1.0 | True | True | - |
| Train | 0.5 | False | False | N,S,W,LW |
| Train | 1.0 | True | True | - |
| Held | 0.5 | False | True | SW |
| Held | 1.0 | True | True | - |

## B. Released policy vs rule pool (10x25), Holm over 8

| comb. | mean | 95% CI | seed-t p | Holm | boot p | Holm |
|---|---|---|---|---|---|---|
| N | -1.86 | [-2.36, -1.36] | 1.4e-06 | 1.1e-05 | 5e-05 | 0.0004 |
| L | -1.36 | [-1.91, -0.82] | 3.4e-05 | 0.00013 | 5e-05 | 0.0004 |
| S | -0.52 | [-1.08, +0.08] | 0.0039 | 0.0039 | 0.044 | 0.044 |
| W | -2.00 | [-2.66, -1.33] | 1.9e-05 | 0.00011 | 5e-05 | 0.0004 |
| LS | -1.10 | [-1.53, -0.65] | 0.00036 | 0.00072 | 5e-05 | 0.0004 |
| LW | -2.10 | [-2.63, -1.57] | 3.2e-05 | 0.00013 | 5e-05 | 0.0004 |
| SW | -0.98 | [-1.47, -0.50] | 1.4e-05 | 0.0001 | 0.0002 | 0.0004 |
| LSW | -1.21 | [-1.65, -0.77] | 2.5e-05 | 0.00013 | 5e-05 | 0.0004 |

## C. Single-rule control (10x25_blindref)

| comb. | vs released | 0.5% | 1% | vs specialist | 0.5% | 1% | vs pool | Holm p |
|---|---|---|---|---|---|---|---|---|
| N | -0.30 [-0.66, +0.11] | non-inferior only | equivalent | -0.08 [-0.63, +0.45] | non-inferior only | equivalent | -2.17 [-2.73, -1.57] | 0.00019 |
| L | +0.16 [-0.13, +0.44] | equivalent | equivalent | +0.32 [-0.00, +0.63] | inconclusive | equivalent | -1.22 [-1.77, -0.68] | 0.0001 |
| S | +0.16 [-0.17, +0.49] | equivalent | equivalent | -0.22 [-0.55, +0.10] | non-inferior only | equivalent | -0.39 [-0.92, +0.17] | 0.0012 |
| W | -0.55 [-1.03, -0.09] | non-inferior only | non-inferior only | -0.22 [-1.29, +0.74] | inconclusive | non-inferior only | -2.59 [-3.31, -1.90] | 0.00012 |
| LS | +0.22 [-0.08, +0.52] | inconclusive | equivalent | +0.13 [-0.14, +0.41] | equivalent | equivalent | -0.90 [-1.31, -0.49] | 0.00013 |
| LW | +0.13 [-0.23, +0.48] | equivalent | equivalent | +0.56 [+0.21, +0.92] | inferior | equivalent | -2.01 [-2.54, -1.47] | 5.7e-05 |
| SW | +0.12 [-0.26, +0.52] | inconclusive | equivalent | -0.40 [-0.86, +0.09] | non-inferior only | equivalent | -0.90 [-1.43, -0.38] | 0.0012 |
| LSW | +0.18 [-0.15, +0.53] | inconclusive | equivalent | +0.01 [-0.44, +0.43] | equivalent | equivalent | -1.06 [-1.52, -0.60] | 1.4e-05 |

## E. Production scale, single checkpoints (selected: s304)

The 1800 s solver rows are skipped in the exact environment: the solver comparison at the production batches comes from scripts/p94_timed_inference.py (the timed campaign); or_solution/FAMILY/{50x25,80x25}_fairwarm1800s are not read and no macro of them is written.

| set | rival | s301 | s302 | s303 | s304 | s305 | selected | seed mean |
|---|---|---|---|---|---|---|---|---|
| 50x25 | pool | +5.51 [+4.79, +6.19] | +3.77 [+3.11, +4.43] | +3.95 [+3.24, +4.68] | +4.03 [+3.28, +4.72] | +4.98 [+4.31, +5.62] | +4.03 | +4.41 |
| 50x25n30 | pool | +5.14 [+4.61, +5.68] | +3.51 [+3.00, +4.05] | +3.78 [+3.27, +4.33] | +3.58 [+2.96, +4.18] | +4.62 [+4.11, +5.13] | +3.58 | +4.09 |
| 80x25 | pool | +5.37 [+4.78, +5.99] | +3.53 [+2.97, +4.09] | +4.14 [+3.66, +4.64] | +3.07 [+2.17, +3.94] | +5.07 [+4.62, +5.48] | +3.07 | +4.18 |
| 80x25n30 | pool | +5.22 [+4.72, +5.73] | +3.31 [+2.79, +3.84] | +4.23 [+3.86, +4.61] | +2.86 [+2.08, +3.63] | +5.08 [+4.65, +5.50] | +2.86 | +4.09 |

## B'. Combined verdict against the rule pool (10x25)

ahead: the hierarchical 95% CI lies below zero and the Holm-adjusted one-sided p-values for "the first arm is faster" (seed-level t and hierarchical bootstrap) are both below 0.05. behind: the CI lies above zero and the Holm-adjusted one-sided p-values for "the first arm is slower" (each kind with its own Holm family over the eight combinations) are both below 0.05. not distinguishable: otherwise.

| arm | comb. | mean | 95% CI | Holm p faster (t / boot) | p slower (t / boot) | Holm p slower (t / boot) | verdict |
|---|---|---|---|---|---|---|---|
| released | N | -1.86 | [-2.36, -1.36] | 1.1e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | L | -1.36 | [-1.91, -0.82] | 0.00013 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | S | -0.52 | [-1.08, +0.08] | 0.0039 / 0.044 | 1 / 0.96 | 1 / 1 | not distinguishable |
| released | W | -2.00 | [-2.66, -1.33] | 0.00011 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | LS | -1.10 | [-1.53, -0.65] | 0.00072 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | LW | -2.10 | [-2.63, -1.57] | 0.00013 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | SW | -0.98 | [-1.47, -0.50] | 0.0001 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| released | LSW | -1.21 | [-1.65, -0.77] | 0.00013 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | N | -2.17 | [-2.73, -1.57] | 0.00019 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | L | -1.22 | [-1.77, -0.68] | 0.0001 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | S | -0.39 | [-0.92, +0.17] | 0.0012 / 0.083 | 1 / 0.92 | 1 / 1 | not distinguishable |
| control | W | -2.59 | [-3.31, -1.90] | 0.00012 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | LS | -0.90 | [-1.31, -0.49] | 0.00013 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | LW | -2.01 | [-2.54, -1.47] | 5.7e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |
| control | SW | -0.90 | [-1.43, -0.38] | 0.0012 / 0.001 | 1 / 1 | 1 / 1 | ahead |
| control | LSW | -1.06 | [-1.52, -0.60] | 1.4e-05 / 0.0004 | 1 / 1 | 1 / 1 | ahead |

## E'. Production scale, policy minus rival over rival (selected: s304)

d = 100 (policy - rival) / rival per instance, negative = policy shorter; recomputed from the arrays with the cluster bootstrap of E.

The 1800 s solver rows are skipped in the exact environment: the solver comparison at the production batches comes from scripts/p94_timed_inference.py (the timed campaign); or_solution/FAMILY/{50x25,80x25}_fairwarm1800s are not read and no macro of them is written.

| set | rival | s301 | s302 | s303 | s304 | s305 | selected | seed mean |
|---|---|---|---|---|---|---|---|---|
| 50x25 | pool | -5.15 [-5.77, -4.50] | -3.56 [-4.17, -2.94] | -3.72 [-4.39, -3.07] | -3.75 [-4.41, -3.03] | -4.67 [-5.25, -4.06] | -3.75 | -4.17 |
| 50x25n30 | pool | -4.83 [-5.31, -4.34] | -3.32 [-3.82, -2.84] | -3.57 [-4.08, -3.10] | -3.34 [-3.91, -2.72] | -4.34 [-4.81, -3.87] | -3.34 | -3.88 |
| 80x25 | pool | -5.04 [-5.58, -4.51] | -3.35 [-3.86, -2.83] | -3.91 [-4.36, -3.47] | -2.77 [-3.63, -1.87] | -4.76 [-5.13, -4.36] | -2.77 | -3.97 |
| 80x25n30 | pool | -4.90 [-5.36, -4.45] | -3.15 [-3.64, -2.65] | -4.00 [-4.34, -3.66] | -2.58 [-3.35, -1.81] | -4.77 [-5.15, -4.39] | -2.58 | -3.88 |

## Sections F, G, H

- F, rule-change pairs, 10x25: computed
- G, specialist of the full combination LSW, 10x25: computed
- H, larger shops: 20x25, 30x25, 50x15, 80x15: computed

## F. Rule-change pairs (10x25), on the later combination B

d = 100 (first - second) / second, seed s against seed s; hierarchical 95% CI; combined verdict with Holm over the twelve pairs (each direction its own family).

| A to B | policy vs specialist of A | 0.5% | 1% | verdict | specialist of B vs specialist of A | 0.5% | 1% | verdict |
|---|---|---|---|---|---|---|---|---|
| N to L | -0.96 [-1.78, -0.31] | non-inferior only | non-inferior only | not distinguishable | -1.11 [-1.98, -0.36] | non-inferior only | non-inferior only | not distinguishable |
| N to S | -0.21 [-0.60, +0.14] | non-inferior only | equivalent | not distinguishable | +0.17 [-0.28, +0.61] | inconclusive | equivalent | not distinguishable |
| N to W | +0.56 [-0.16, +1.27] | inconclusive | inconclusive | not distinguishable | +0.23 [-0.46, +1.01] | inconclusive | inconclusive | not distinguishable |
| L to LS | -0.27 [-0.57, +0.03] | non-inferior only | equivalent | not distinguishable | -0.18 [-0.51, +0.12] | non-inferior only | equivalent | not distinguishable |
| L to LW | +0.07 [-0.40, +0.62] | inconclusive | equivalent | not distinguishable | -0.36 [-0.69, -0.02] | non-inferior only | equivalent | not distinguishable |
| S to LS | -0.60 [-1.00, -0.25] | non-inferior only | non-inferior only | not distinguishable | -0.51 [-0.91, -0.14] | non-inferior only | equivalent | not distinguishable |
| S to SW | -0.57 [-1.00, -0.15] | non-inferior only | equivalent | ahead | -0.04 [-0.51, +0.47] | non-inferior only | equivalent | not distinguishable |
| W to LW | -1.62 [-2.17, -1.09] | non-inferior only | non-inferior only | ahead | -2.04 [-2.66, -1.46] | non-inferior only | non-inferior only | ahead |
| W to SW | +0.10 [-0.44, +0.63] | inconclusive | equivalent | not distinguishable | +0.63 [+0.15, +1.11] | inferior | inferior | not distinguishable |
| LS to LSW | -0.36 [-0.80, +0.03] | non-inferior only | equivalent | not distinguishable | -0.17 [-0.69, +0.35] | non-inferior only | equivalent | not distinguishable |
| LW to LSW | -0.32 [-0.65, +0.01] | non-inferior only | equivalent | not distinguishable | -0.14 [-0.57, +0.33] | non-inferior only | equivalent | not distinguishable |
| SW to LSW | -0.78 [-1.38, -0.17] | non-inferior only | non-inferior only | not distinguishable | -0.60 [-1.25, +0.09] | non-inferior only | non-inferior only | not distinguishable |
| pooled | -0.42 [-0.63, -0.22] | non-inferior only | equivalent | ahead | -0.34 [-0.54, -0.15] | non-inferior only | equivalent | ahead |

policy vs specialist of A: ahead on 2, behind on 0, not distinguishable on 10 of 12 pairs.

specialist of B vs specialist of A: ahead on 1, behind on 0, not distinguishable on 11 of 12 pairs.

## G. The LSW specialist on every combination (10x25_blindref)

d = 100 (LSW specialist - other) / other, seed s against seed s; hierarchical 95% CI.

| comb. | vs released policy | 0.5% | 1% | dir. | vs specialist of the combination | 0.5% | 1% | dir. |
|---|---|---|---|---|---|---|---|---|
| N | +6.58 [+4.27, +8.81] | inferior | inferior | behind | +6.85 [+4.14, +9.54] | inferior | inferior | behind |
| L | +0.07 [-0.21, +0.34] | equivalent | equivalent | neither | +0.23 [-0.07, +0.54] | inconclusive | equivalent | neither |
| S | +1.59 [+1.04, +2.21] | inferior | inferior | behind | +1.20 [+0.65, +1.78] | inferior | inferior | behind |
| W | +5.35 [+3.51, +7.46] | inferior | inferior | behind | +5.76 [+3.23, +8.57] | inferior | inferior | behind |
| LS | +0.12 [-0.28, +0.52] | inconclusive | equivalent | neither | +0.03 [-0.30, +0.38] | equivalent | equivalent | neither |
| LW | -0.01 [-0.37, +0.35] | equivalent | equivalent | neither | +0.41 [+0.08, +0.77] | inferior | equivalent | behind |
| SW | +1.29 [+0.77, +1.84] | inferior | inferior | behind | +0.78 [+0.16, +1.42] | inferior | inferior | behind |
| LSW | +0.21 [-0.17, +0.60] | inconclusive | equivalent | neither | the same run | - | - | - |
| pooled All | +1.90 [+1.33, +2.56] | inferior | inferior | behind | - | - | - | - |
| pooled Other | +2.14 [+1.48, +2.89] | inferior | inferior | behind | +2.18 [+1.36, +3.10] | inferior | inferior | behind |

## H. Larger shops

d = 100 (released policy - other) / other, seed s against seed s; rule pool = per-instance best of the four rules; combined verdict with Holm over the eight combinations of a size; the selected run with the instance-level cluster bootstrap.

### 20x25 (50 instances per combination)

| comb. | vs specialist | 0.5% | 1% | dir. | vs rule pool | verdict | selected run vs rule pool | verdict |
|---|---|---|---|---|---|---|---|---|
| N | +0.51 [-1.59, +2.01] | inconclusive | inconclusive | neither | -2.24 [-3.13, -1.40] | ahead | - | - |
| L | -2.15 [-3.47, -0.89] | non-inferior only | non-inferior only | ahead | -2.66 [-3.32, -1.99] | ahead | - | - |
| S | -0.59 [-1.52, +0.18] | non-inferior only | non-inferior only | neither | -2.29 [-3.19, -1.36] | ahead | - | - |
| W | +1.28 [-1.05, +3.09] | inconclusive | inconclusive | neither | -2.95 [-3.90, -2.03] | ahead | - | - |
| LS | -0.15 [-1.21, +0.91] | inconclusive | non-inferior only | neither | -1.57 [-2.28, -0.83] | ahead | - | - |
| LW | -3.19 [-6.06, -1.20] | non-inferior only | non-inferior only | ahead | -3.23 [-3.83, -2.65] | ahead | - | - |
| SW | -0.37 [-1.34, +0.52] | inconclusive | non-inferior only | neither | -1.88 [-2.72, -0.95] | ahead | - | - |
| LSW | -0.26 [-1.34, +0.62] | inconclusive | non-inferior only | neither | -2.41 [-3.19, -1.63] | ahead | - | - |
| pooled | -0.61 [-1.12, -0.12] | non-inferior only | non-inferior only | ahead | -2.41 [-2.80, -1.98] | ahead | - | - |

### 30x25 (20 instances per combination)

| comb. | vs specialist | 0.5% | 1% | dir. | vs rule pool | verdict | selected run vs rule pool | verdict |
|---|---|---|---|---|---|---|---|---|
| N | -0.22 [-2.93, +1.82] | inconclusive | inconclusive | neither | -2.97 [-4.30, -1.69] | ahead | - | - |
| L | -2.61 [-5.49, -0.33] | non-inferior only | non-inferior only | ahead | -3.86 [-4.96, -2.81] | ahead | - | - |
| S | -0.59 [-1.63, +0.36] | non-inferior only | non-inferior only | neither | -3.10 [-4.16, -2.04] | ahead | - | - |
| W | +1.01 [-1.07, +2.82] | inconclusive | inconclusive | neither | -3.03 [-4.05, -1.90] | ahead | - | - |
| LS | -0.27 [-1.46, +0.86] | inconclusive | non-inferior only | neither | -2.89 [-4.32, -1.21] | ahead | - | - |
| LW | -5.24 [-9.78, -1.48] | non-inferior only | non-inferior only | ahead | -3.46 [-4.55, -2.32] | ahead | - | - |
| SW | -0.89 [-2.52, +0.56] | inconclusive | non-inferior only | neither | -3.25 [-4.56, -1.95] | ahead | - | - |
| LSW | +0.08 [-1.39, +1.27] | inconclusive | inconclusive | neither | -2.76 [-3.80, -1.70] | ahead | - | - |
| pooled | -1.09 [-2.16, -0.05] | non-inferior only | non-inferior only | ahead | -3.16 [-3.90, -2.43] | ahead | - | - |

### 50x15 (20 instances per combination)

| comb. | vs specialist | 0.5% | 1% | dir. | vs rule pool | verdict | selected run vs rule pool | verdict |
|---|---|---|---|---|---|---|---|---|
| N | - | - | - | - | -2.16 [-2.85, -1.44] | ahead | -2.70 [-3.33, -2.06] | ahead |
| L | - | - | - | - | -3.47 [-4.29, -2.65] | ahead | -3.01 [-3.80, -2.18] | ahead |
| S | - | - | - | - | -1.78 [-2.81, -0.83] | ahead | -2.32 [-3.32, -1.41] | ahead |
| W | - | - | - | - | -2.79 [-3.29, -2.28] | ahead | -3.18 [-3.70, -2.68] | ahead |
| LS | - | - | - | - | -2.52 [-3.60, -1.51] | ahead | -1.84 [-2.93, -0.73] | ahead |
| LW | - | - | - | - | -3.24 [-3.95, -2.42] | ahead | -3.28 [-3.85, -2.75] | ahead |
| SW | - | - | - | - | -1.81 [-2.50, -1.08] | ahead | -2.36 [-2.94, -1.79] | ahead |
| LSW | - | - | - | - | -1.82 [-2.70, -0.91] | ahead | -1.74 [-2.71, -0.75] | ahead |
| pooled | - | - | - | - | -2.45 [-2.91, -1.97] | ahead | -2.55 [-2.99, -2.14] | ahead |

### 80x15 (20 instances per combination)

| comb. | vs specialist | 0.5% | 1% | dir. | vs rule pool | verdict | selected run vs rule pool | verdict |
|---|---|---|---|---|---|---|---|---|
| N | - | - | - | - | -2.42 [-3.08, -1.79] | ahead | -2.93 [-3.56, -2.32] | ahead |
| L | - | - | - | - | -3.05 [-3.76, -2.25] | ahead | -3.05 [-3.65, -2.36] | ahead |
| S | - | - | - | - | -1.81 [-2.54, -1.13] | ahead | -1.95 [-2.68, -1.28] | ahead |
| W | - | - | - | - | -2.65 [-3.20, -2.08] | ahead | -3.00 [-3.51, -2.46] | ahead |
| LS | - | - | - | - | -1.86 [-2.56, -1.17] | ahead | -1.52 [-2.27, -0.77] | ahead |
| LW | - | - | - | - | -3.02 [-3.63, -2.31] | ahead | -3.21 [-3.69, -2.70] | ahead |
| SW | - | - | - | - | -1.66 [-2.36, -1.02] | ahead | -1.76 [-2.34, -1.20] | ahead |
| LSW | - | - | - | - | -1.99 [-2.71, -1.25] | ahead | -2.43 [-3.07, -1.81] | ahead |
| pooled | - | - | - | - | -2.31 [-2.77, -1.82] | ahead | -2.48 [-2.86, -2.10] | ahead |
