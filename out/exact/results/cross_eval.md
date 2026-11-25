# Every specialist under every combination against the policy

Exact mask, 10x25 test set (100 base instances per combination), seeds 301, 302, 303, 304, 305. d = 100 (specialist - policy) / policy, specialist seed s against policy seed s; positive = the specialist schedule is longer. 95% intervals: crossed hierarchical bootstrap over seeds and base instances (B = 20000, fresh default_rng(11) per quantity). Rows: the combination the specialist was trained for; columns: the combination it is run under; the diagonal is the specialist on its own combination.

## Mean [95% interval]

| trained for | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| N | -0.16 [-0.79, +0.51] | +1.06 [+0.37, +1.93] | +0.28 [-0.08, +0.68] | -0.41 [-1.12, +0.32] | +0.85 [+0.46, +1.29] | +1.53 [+0.93, +2.31] | +0.21 [-0.26, +0.69] | +1.11 [+0.68, +1.62] |
| L | +4.47 [+2.40, +6.43] | -0.14 [-0.43, +0.17] | +1.54 [+0.82, +2.35] | +3.97 [+2.19, +5.74] | +0.32 [+0.02, +0.62] | -0.00 [-0.54, +0.46] | +1.31 [+0.82, +1.84] | +0.52 [+0.11, +0.93] |
| S | +1.88 [+0.92, +3.13] | +0.64 [+0.16, +1.22] | +0.41 [-0.01, +0.88] | +1.66 [+0.68, +2.71] | +0.66 [+0.30, +1.08] | +0.98 [+0.35, +1.69] | +0.68 [+0.24, +1.11] | +0.68 [+0.21, +1.26] |
| W | +0.33 [-0.23, +0.87] | +1.63 [+1.03, +2.27] | +0.03 [-0.40, +0.45] | -0.24 [-1.15, +0.66] | +0.88 [+0.38, +1.40] | +1.78 [+1.21, +2.37] | -0.00 [-0.54, +0.55] | +1.00 [+0.55, +1.54] |
| LS | +2.68 [+1.53, +3.80] | +0.36 [-0.14, +0.92] | +0.74 [+0.22, +1.25] | +2.54 [+1.85, +3.29] | +0.11 [-0.21, +0.42] | +0.73 [+0.14, +1.45] | +0.63 [+0.17, +1.09] | +0.42 [+0.02, +0.86] |
| LW | +7.51 [+4.92, +11.54] | -0.38 [-0.68, -0.09] | +1.51 [+0.84, +2.39] | +7.04 [+4.36, +10.93] | +0.09 [-0.26, +0.41] | -0.40 [-0.80, +0.01] | +1.57 [+0.79, +2.51] | +0.38 [+0.05, +0.72] |
| SW | +2.09 [+1.34, +2.97] | +1.19 [+0.29, +2.14] | +0.30 [-0.08, +0.65] | +1.72 [+1.20, +2.26] | +0.69 [+0.14, +1.27] | +1.65 [+0.67, +2.65] | +0.58 [+0.17, +1.01] | +0.88 [+0.25, +1.52] |
| LSW | +6.58 [+4.27, +8.81] | +0.07 [-0.21, +0.34] | +1.59 [+1.04, +2.21] | +5.35 [+3.51, +7.46] | +0.12 [-0.28, +0.52] | -0.01 [-0.37, +0.35] | +1.29 [+0.77, +1.84] | +0.21 [-0.17, +0.60] |

## Verdicts at 0.5% / 1% and direction

| trained for | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| N | inconclusive / equivalent / neither | inferior / inferior / behind | inconclusive / equivalent / neither | non-inferior only / non-inferior only / neither | inferior / inferior / behind | inferior / inferior / behind | inconclusive / equivalent / neither | inferior / inferior / behind |
| L | inferior / inferior / behind | equivalent / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | inferior / equivalent / behind | non-inferior only / equivalent / neither | inferior / inferior / behind | inferior / equivalent / behind |
| S | inferior / inferior / behind | inferior / inferior / behind | inconclusive / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind |
| W | inconclusive / equivalent / neither | inferior / inferior / behind | equivalent / equivalent / neither | inconclusive / non-inferior only / neither | inferior / inferior / behind | inferior / inferior / behind | inconclusive / equivalent / neither | inferior / inferior / behind |
| LS | inferior / inferior / behind | inconclusive / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | equivalent / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | inferior / equivalent / behind |
| LW | inferior / inferior / behind | non-inferior only / equivalent / ahead | inferior / inferior / behind | inferior / inferior / behind | equivalent / equivalent / neither | non-inferior only / equivalent / neither | inferior / inferior / behind | inferior / equivalent / behind |
| SW | inferior / inferior / behind | inferior / inferior / behind | inconclusive / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind | inferior / inferior / behind |
| LSW | inferior / inferior / behind | equivalent / equivalent / neither | inferior / inferior / behind | inferior / inferior / behind | inconclusive / equivalent / neither | equivalent / equivalent / neither | inferior / inferior / behind | inconclusive / equivalent / neither |

## Each specialist pooled over its seven off-combination columns

| specialist | mean [95% CI] | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|
| N | +0.66 [+0.30, +1.10] | [+0.12, +1.20] | inferior | inferior | behind |
| L | +1.73 [+1.01, +2.42] | [+0.69, +2.77] | inferior | inferior | behind |
| S | +1.03 [+0.75, +1.31] | [+0.66, +1.40] | inferior | inferior | behind |
| W | +0.81 [+0.53, +1.10] | [+0.52, +1.10] | inferior | inferior | behind |
| LS | +1.16 [+0.83, +1.48] | [+0.70, +1.61] | inferior | inferior | behind |
| LW | +2.53 [+1.60, +3.91] | [+0.74, +4.32] | inferior | inferior | behind |
| SW | +1.22 [+0.72, +1.71] | [+0.47, +1.97] | inferior | inferior | behind |
| LSW | +2.14 [+1.48, +2.89] | [+1.12, +3.16] | inferior | inferior | behind |

## Single-rule changes grouped by the rule and the direction

The 24 off-combination cells whose two combinations differ by one rule. Rule added: the specialist trained for A run under B = A plus the rule. Rule removed: the specialist trained for B run under A. In this table g = 100 (policy - specialist) / specialist, negative = the policy schedule is shorter (the difference of the family analysis). Each row pools its cells, the cells of a drawn base instance kept together. The grouping describes the matrix and adds no planned comparison.

| group | cells (trained for > run under) | mean [95% CI] | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| AddLag | N>L, S>LS, W>LW, SW>LSW | -0.99 [-1.36, -0.68] | [-1.37, -0.61] | non-inferior only | non-inferior only | ahead |
| DropLag | L>N, LS>S, LW>W, LSW>SW | -2.99 [-3.77, -2.24] | [-4.01, -1.97] | non-inferior only | non-inferior only | ahead |
| AddSetup | N>S, L>LS, W>SW, LW>LSW | -0.18 [-0.40, +0.03] | [-0.38, +0.03] | equivalent | equivalent | neither |
| DropSetup | S>N, LS>L, SW>W, LSW>LW | -0.88 [-1.17, -0.60] | [-1.24, -0.53] | non-inferior only | non-inferior only | ahead |
| AddWindow | N>W, L>LW, S>SW, LS>LSW | -0.08 [-0.33, +0.16] | [-0.30, +0.15] | equivalent | equivalent | neither |
| DropWindow | W>N, LW>L, SW>S, LSW>LS | -0.03 [-0.22, +0.17] | [-0.19, +0.13] | equivalent | equivalent | neither |
| AddAll | N>L, S>LS, W>LW, SW>LSW, N>S, L>LS, W>SW, LW>LSW, N>W, L>LW, S>SW, LS>LSW | -0.42 [-0.63, -0.22] | [-0.63, -0.20] | non-inferior only | equivalent | ahead |
| DropAll | L>N, LS>S, LW>W, LSW>SW, S>N, LS>L, SW>W, LSW>LW, W>N, LW>L, SW>S, LSW>LS | -1.30 [-1.55, -1.08] | [-1.55, -1.05] | non-inferior only | non-inferior only | ahead |

AddAll holds the twelve rule-change pairs of the family record (section F): mean -0.4152 here and -0.4152 there, interval [-0.6271, -0.2228] here and [-0.6271, -0.2228] there.

## Counts

- 56 off-combination cells: interval above zero 42, equivalent at 1% 17, interval below zero 1.
- 7 off-combination cells of the LSW row: interval above zero 4, equivalent at 1% 3, interval below zero 0.
- 8 diagonal cells: interval above zero 1, equivalent at 1% 6, interval below zero 0.

95 off-combination arrays of the principal 10x25 campaign were compared with the arrays of test_results/FAMILY/10x25_cross_xmask; all hold the same makespans.

