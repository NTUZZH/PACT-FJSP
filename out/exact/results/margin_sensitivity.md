# What the equivalence test returns for comparators known to differ from the specialist

Factory benchmark at the training size (100 test instances), exact mask. A difference is the makespan of the first-named method minus that of the second, in percent of the second (positive = the first-named method is longer). Verdicts at the 0.5% and 1% margins from the 95% interval (B = 20000).

Analysis A describes what the verdict rule of the principal comparison returns for comparators whose quality relative to the specialist is known from other evidence; it adds no planned comparison and changes no verdict of the principal comparison. The rule makespans do not depend on a training seed, so the seed level of the interval resamples the specialist only. The columns read from records are not recomputed.

## Best of four rules minus the specialist, beside the two read comparisons

Best rule minus specialist: ten seeds (one default_rng(110) stream in the order ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW'], a fresh default_rng(110) for the pooled row). Policy minus specialist: results/seed_extension.json (scripts/p110_seed_extension.py), keys ten and pool_ten. Policy minus best rule: results/family_inference.json (scripts/p88_family_inference.py), key rule_pool (the FamRule macros); pooled row from results/production_specialists.json (scripts/p128_production_specialists.py), side_by_side.rows.10x25.rule; the verdict is the combined verdict with Holm over the eight combinations, the pooled row gives the direction.

| comb. | rule - spec. | 0.5% | 1% | dir. | policy - spec. | 0.5% | 1% | policy - rule | verdict |
|---|---|---|---|---|---|---|---|---|---|
| N | +2.42 [+1.85, +2.99] | inferior | inferior | behind | +0.56 [+0.04, +1.07] | inferior | inferior | -1.86 [-2.36, -1.36] | ahead |
| L | +1.59 [+1.05, +2.15] | inferior | inferior | behind | +0.15 [-0.09, +0.37] | equivalent | equivalent | -1.36 [-1.91, -0.82] | ahead |
| S | +0.11 [-0.37, +0.59] | inconclusive | equivalent | neither | -0.40 [-0.76, -0.05] | non-inferior only | equivalent | -0.52 [-1.08, +0.08] | not distinguishable |
| W | +2.68 [+1.92, +3.41] | inferior | inferior | behind | +0.34 [-0.26, +0.97] | inconclusive | equivalent | -2.00 [-2.66, -1.33] | ahead |
| LS | +0.93 [+0.50, +1.37] | inferior | inferior | behind | -0.23 [-0.49, +0.02] | equivalent | equivalent | -1.10 [-1.53, -0.65] | ahead |
| LW | +2.50 [+1.98, +3.03] | inferior | inferior | behind | +0.20 [-0.14, +0.55] | inconclusive | equivalent | -2.10 [-2.63, -1.57] | ahead |
| SW | +0.77 [+0.26, +1.28] | inferior | inferior | behind | -0.31 [-0.65, +0.06] | non-inferior only | equivalent | -0.98 [-1.47, -0.50] | ahead |
| LSW | +1.02 [+0.57, +1.47] | inferior | inferior | behind | -0.26 [-0.55, +0.02] | non-inferior only | equivalent | -1.21 [-1.65, -0.77] | ahead |
| pooled | +1.50 [+1.25, +1.76] | inferior | inferior | behind | +0.01 [-0.17, +0.19] | equivalent | equivalent | -1.39 [-1.66, -1.13] | ahead |

Counts of the best rule minus the specialist (ten seeds):

- best rule against the specialist, 0.5% margin: equivalent 0; non-inferior only 0; inferior 7 (N, L, W, LS, LW, SW, LSW); inconclusive 1 (S)
- best rule against the specialist, 1% margin: equivalent 1 (S); non-inferior only 0; inferior 7 (N, L, W, LS, LW, SW, LSW); inconclusive 0

## The same with the five seeds 301 to 305

Streams of the five-seed record (one default_rng(11) stream in the order ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW'], a fresh default_rng(11) for the pooled row).

| comb. | rule - spec. | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|
| N | +2.18 [+1.48, +2.89] | [+1.34, +3.02] | inferior | inferior | behind |
| L | +1.63 [+1.07, +2.22] | [+1.38, +1.89] | inferior | inferior | behind |
| S | +0.23 [-0.32, +0.77] | [-0.07, +0.53] | inconclusive | equivalent | neither |
| W | +2.52 [+1.55, +3.44] | [+1.41, +3.63] | inferior | inferior | behind |
| LS | +1.07 [+0.65, +1.50] | [+0.98, +1.16] | inferior | inferior | behind |
| LW | +2.68 [+2.15, +3.22] | [+2.48, +2.87] | inferior | inferior | behind |
| SW | +0.56 [+0.02, +1.09] | [+0.29, +0.82] | inferior | inferior | behind |
| LSW | +1.12 [+0.62, +1.61] | [+0.80, +1.44] | inferior | inferior | behind |
| pooled | +1.50 [+1.22, +1.78] | [+1.29, +1.71] | inferior | inferior | behind |

- best rule against the specialist, five seeds, 0.5% margin: equivalent 0; non-inferior only 0; inferior 7 (N, L, W, LS, LW, SW, LSW); inconclusive 1 (S)
- best rule against the specialist, five seeds, 1% margin: equivalent 1 (S); non-inferior only 0; inferior 7 (N, L, W, LS, LW, SW, LSW); inconclusive 0

## Further comparators against the specialist of each combination

| comparator | seeds | comparisons | pooled | 0.5% | 1% | source |
|---|---|---|---|---|---|---|
| sparse | 3 | 8 | +4.99 [+3.64, +6.19] | inferior | inferior | results/secondary_inference.json (scripts/p95_secondary_inference.py), block sparse, vs_specialist |
| budget | 10 | 8 | +0.65 [+0.50, +0.80] | inferior | equivalent | results/budget_specialists.json (scripts/p127_budget_specialists.py), share_vs_specialist |
| cross | 5 | 56 | +1.41 [+1.05, +1.74] | inferior | inferior | computed from the arrays of the cross campaign with the readers of scripts/p100_cross_eval.py and p95.component (no record compares these runs with the specialist of the combination) |

- sparse, 0.5% margin: equivalent 0; non-inferior only 0; inferior 8 (N, L, S, W, LS, LW, SW, LSW); inconclusive 0
- sparse, 1% margin: equivalent 0; non-inferior only 0; inferior 8 (N, L, S, W, LS, LW, SW, LSW); inconclusive 0
- budget, 0.5% margin: equivalent 1 (L); non-inferior only 0; inferior 6 (N, W, LS, LW, SW, LSW); inconclusive 1 (S)
- budget, 1% margin: equivalent 5 (L, S, LS, LW, LSW); non-inferior only 0; inferior 3 (N, W, SW); inconclusive 0
- cross (56 cells), 0.5% margin: equivalent 3 (LW|L, LW|LS, LSW|LS); non-inferior only 5 (N|S, N|SW, W|S, W|SW, SW|S); inferior 35 (N|L, N|LS, N|LW, N|LSW, L|N, L|S, L|W, L|LW, L|SW, S|N, S|L, S|W, S|LS, S|LW, W|L, W|LS, W|LW, W|LSW, LS|N, LS|W, LS|LW, LW|N, LW|S, LW|W, LW|SW, SW|N, SW|L, SW|W, SW|LS, SW|LW, LSW|N, LSW|S, LSW|W, LSW|LW, LSW|SW); inconclusive 13 (N|W, L|LS, L|LSW, S|SW, S|LSW, W|N, LS|L, LS|S, LS|SW, LS|LSW, LW|LSW, SW|LSW, LSW|L)
- cross (56 cells), 1% margin: equivalent 19 (N|S, N|W, N|SW, L|LS, L|LW, L|LSW, S|LS, S|SW, W|S, LS|S, LS|SW, LS|LSW, LW|L, LW|LS, LW|LSW, SW|S, LSW|L, LSW|LS, LSW|LW); non-inferior only 1 (W|SW); inferior 32 (N|L, N|LS, N|LW, N|LSW, L|N, L|S, L|W, L|SW, S|N, S|L, S|W, S|LW, W|L, W|LS, W|LW, W|LSW, LS|N, LS|W, LS|LW, LW|N, LW|S, LW|W, LW|SW, SW|N, SW|L, SW|W, SW|LS, SW|LW, LSW|N, LSW|S, LSW|W, LSW|SW); inconclusive 4 (S|LSW, W|N, LS|L, SW|LSW)
- cross, each specialist pooled over its seven other combinations, 0.5% margin: equivalent 0; non-inferior only 0; inferior 8 (N, L, S, W, LS, LW, SW, LSW); inconclusive 0
- cross, each specialist pooled over its seven other combinations, 1% margin: equivalent 1 (W); non-inferior only 0; inferior 7 (N, L, S, LS, LW, SW, LSW); inconclusive 0
- cross, each combination pooled over the seven specialists not trained for it, 0.5% margin: equivalent 0; non-inferior only 0; inferior 7 (N, L, S, W, LS, LW, LSW); inconclusive 1 (SW)
- cross, each combination pooled over the seven specialists not trained for it, 1% margin: equivalent 4 (S, LS, SW, LSW); non-inferior only 0; inferior 4 (N, L, W, LW); inconclusive 0

Sparse reward and budget, per combination (read):

| comb. | sparse | 0.5% | 1% | budget | 0.5% | 1% |
|---|---|---|---|---|---|---|
| N | +6.61 [+3.94, +8.81] | inferior | inferior | +1.05 [+0.64, +1.44] | inferior | inferior |
| L | +4.74 [+2.78, +6.53] | inferior | inferior | +0.11 [-0.27, +0.46] | equivalent | equivalent |
| S | +3.05 [+1.92, +4.11] | inferior | inferior | +0.30 [-0.12, +0.75] | inconclusive | equivalent |
| W | +7.68 [+3.42, +10.85] | inferior | inferior | +1.52 [+0.81, +2.31] | inferior | inferior |
| LS | +3.60 [+2.84, +4.49] | inferior | inferior | +0.56 [+0.25, +0.86] | inferior | equivalent |
| LW | +6.15 [+3.08, +8.67] | inferior | inferior | +0.57 [+0.24, +0.93] | inferior | equivalent |
| SW | +3.74 [+2.78, +4.71] | inferior | inferior | +0.65 [+0.15, +1.16] | inferior | inferior |
| LSW | +4.38 [+3.44, +5.38] | inferior | inferior | +0.48 [+0.14, +0.84] | inferior | equivalent |

Cross: specialist (row) run under combination (column) against the specialist of the column, mean [95% interval], verdict at 1% (d = 100 (specialist X under C - specialist of C) / specialist of C):

| specialist | N | L | S | W | LS | LW | SW | LSW | pooled over its seven |
|---|---|---|---|---|---|---|---|---|---|
| N | - | +1.22 [+0.44, +2.13] inferior | -0.10 [-0.54, +0.35] equivalent | -0.10 [-0.86, +0.57] equivalent | +0.76 [+0.37, +1.19] inferior | +1.97 [+1.25, +2.79] inferior | -0.31 [-0.76, +0.14] equivalent | +0.94 [+0.41, +1.49] inferior | +0.63 [+0.28, +1.05] inferior |
| L | +4.72 [+2.42, +6.83] inferior | - | +1.15 [+0.30, +2.06] inferior | +4.34 [+2.28, +6.24] inferior | +0.22 [-0.08, +0.56] equivalent | +0.42 [+0.08, +0.75] equivalent | +0.79 [+0.30, +1.31] inferior | +0.34 [-0.10, +0.77] equivalent | +1.71 [+0.91, +2.41] inferior |
| S | +2.11 [+0.99, +3.32] inferior | +0.80 [+0.35, +1.36] inferior | - | +2.01 [+0.57, +3.35] inferior | +0.57 [+0.19, +0.98] equivalent | +1.42 [+0.79, +2.16] inferior | +0.16 [-0.34, +0.62] equivalent | +0.51 [-0.03, +1.09] inconclusive | +1.08 [+0.67, +1.47] inferior |
| W | +0.52 [-0.21, +1.16] inconclusive | +1.80 [+1.08, +2.55] inferior | -0.35 [-0.78, +0.07] equivalent | - | +0.79 [+0.37, +1.24] inferior | +2.22 [+1.59, +2.89] inferior | -0.52 [-1.01, -0.02] non-inferior only | +0.83 [+0.26, +1.42] inferior | +0.76 [+0.52, +0.99] equivalent |
| LS | +2.90 [+1.39, +4.42] inferior | +0.52 [-0.05, +1.11] inconclusive | +0.35 [-0.13, +0.88] equivalent | +2.89 [+1.87, +3.88] inferior | - | +1.15 [+0.51, +1.89] inferior | +0.11 [-0.33, +0.55] equivalent | +0.25 [-0.28, +0.77] equivalent | +1.17 [+0.79, +1.52] inferior |
| LW | +7.78 [+4.84, +11.89] inferior | -0.23 [-0.46, -0.03] equivalent | +1.13 [+0.27, +2.13] inferior | +7.44 [+4.55, +11.80] inferior | -0.01 [-0.36, +0.36] equivalent | - | +1.06 [+0.18, +2.14] inferior | +0.21 [-0.25, +0.64] equivalent | +2.48 [+1.41, +4.02] inferior |
| SW | +2.31 [+1.13, +3.67] inferior | +1.36 [+0.51, +2.26] inferior | -0.08 [-0.53, +0.36] equivalent | +2.07 [+0.97, +3.15] inferior | +0.60 [+0.17, +1.05] inferior | +2.08 [+1.12, +3.04] inferior | - | +0.71 [-0.00, +1.38] inconclusive | +1.29 [+0.76, +1.80] inferior |
| LSW | +6.85 [+4.14, +9.54] inferior | +0.23 [-0.07, +0.54] equivalent | +1.20 [+0.65, +1.78] inferior | +5.76 [+3.23, +8.57] inferior | +0.03 [-0.30, +0.38] equivalent | +0.41 [+0.08, +0.77] equivalent | +0.78 [+0.16, +1.42] inferior | - | +2.18 [+1.36, +3.10] inferior |
| pooled over the seven others | +3.88 [+2.57, +5.10] inferior | +0.81 [+0.43, +1.23] inferior | +0.47 [+0.05, +0.85] equivalent | +3.49 [+2.23, +4.83] inferior | +0.42 [+0.20, +0.64] equivalent | +1.38 [+0.95, +1.80] inferior | +0.30 [-0.06, +0.66] equivalent | +0.54 [+0.12, +0.91] equivalent | all 56: +1.41 [+1.05, +1.74] inferior |

## Checks

- The LSW row of the cross comparison reproduces section G of results/family_inference.json: 21 numbers identical at the printed digits (largest absolute difference 0.00e+00).
- The arrays read here give the means of the read records to 1e-09: the policy minus the specialist over ten seeds (results/seed_extension.json), the policy minus the best rule over five seeds (results/family_inference.json), and the pooled policy minus the best rule (results/production_specialists.json).

