# The policy against two specialists that serve all eight combinations

Factory benchmark at the training size, exact mask, seeds 301, 302, 303, 304, 305, policy seed s against specialist seed s. A difference is the makespan of the policy minus that of the serving specialist, in percent of the specialist (negative = the policy is shorter). Per combination the crossed hierarchical bootstrap with the draws of one default_rng(11) stream in the order of p88 section A (every option meets the same resamples); pooled over the eight with a fresh default_rng(11), a drawn base instance with all its combinations. B = 20000.

The options B1 and B2 were named after the test results were known (the matrix of every specialist under every combination, results/cross_eval.json), so their comparisons are post hoc. B3 is the single specialist of the combination without rules. B4 is the five-seed principal comparison and reproduces its record. The oracle pairs choose, for each combination, the better of the two specialists on the test instances themselves, so their means are an optimistic bound on a choice made before testing.

Inputs: 360 arrays read and checked by the readers of scripts/p100_cross_eval.py; 95 off-combination arrays equal to those of the principal campaign.

## B1: two specialists by the lag rule: N under N, S, W, SW and LSW under L, LS, LW, LSW

| comb. | served by | policy - option | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| N | N | +0.24 [-0.40, +0.86] | [-0.54, +1.03] | inconclusive | equivalent | neither |
| L | LSW | -0.03 [-0.31, +0.25] | [-0.18, +0.13] | equivalent | equivalent | neither |
| S | N | -0.21 [-0.60, +0.14] | [-0.51, +0.08] | non-inferior only | equivalent | neither |
| W | N | +0.56 [-0.18, +1.28] | [-0.29, +1.41] | inconclusive | inconclusive | neither |
| LS | LSW | -0.08 [-0.47, +0.32] | [-0.54, +0.39] | equivalent | equivalent | neither |
| LW | LSW | +0.08 [-0.28, +0.44] | [-0.21, +0.36] | equivalent | equivalent | neither |
| SW | N | -0.12 [-0.60, +0.33] | [-0.58, +0.34] | non-inferior only | equivalent | neither |
| LSW | LSW | -0.15 [-0.53, +0.22] | [-0.48, +0.19] | non-inferior only | equivalent | neither |
| pooled | - | +0.04 [-0.19, +0.25] | [-0.21, +0.28] | equivalent | equivalent | neither |

- B1, 0.5% margin: equivalent 3 (L, LS, LW); non-inferior only 3 (S, SW, LSW); inferior 0; inconclusive 2 (N, W)
- B1, 1% margin: equivalent 7 (N, L, S, LS, LW, SW, LSW); non-inferior only 0; inferior 0; inconclusive 1 (W)
- B1: interval below zero (policy shorter) on none; above zero on none.

## B2: two specialists by the lag rule with L in the place of LSW: N under N, S, W, SW and L under L, LS, LW, LSW

| comb. | served by | policy - option | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| N | N | +0.24 [-0.40, +0.86] | [-0.54, +1.03] | inconclusive | equivalent | neither |
| L | L | +0.18 [-0.11, +0.49] | [-0.06, +0.43] | equivalent | equivalent | neither |
| S | N | -0.21 [-0.60, +0.14] | [-0.51, +0.08] | non-inferior only | equivalent | neither |
| W | N | +0.56 [-0.18, +1.28] | [-0.29, +1.41] | inconclusive | inconclusive | neither |
| LS | L | -0.27 [-0.57, +0.03] | [-0.50, -0.04] | non-inferior only | equivalent | neither |
| LW | L | +0.07 [-0.40, +0.63] | [-0.56, +0.69] | inconclusive | equivalent | neither |
| SW | N | -0.12 [-0.60, +0.33] | [-0.58, +0.34] | non-inferior only | equivalent | neither |
| LSW | L | -0.46 [-0.87, -0.05] | [-0.89, -0.02] | non-inferior only | equivalent | ahead |
| pooled | - | -0.00 [-0.21, +0.20] | [-0.21, +0.21] | equivalent | equivalent | neither |

- B2, 0.5% margin: equivalent 1 (L); non-inferior only 4 (S, LS, SW, LSW); inferior 0; inconclusive 3 (N, W, LW)
- B2, 1% margin: equivalent 7 (N, L, S, LS, LW, SW, LSW); non-inferior only 0; inferior 0; inconclusive 1 (W)
- B2: interval below zero (policy shorter) on LSW; above zero on none.

## B3: one specialist: N under all eight combinations

| comb. | served by | policy - option | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| N | N | +0.24 [-0.40, +0.86] | [-0.54, +1.03] | inconclusive | equivalent | neither |
| L | N | -0.96 [-1.77, -0.30] | [-1.97, +0.05] | non-inferior only | non-inferior only | ahead |
| S | N | -0.21 [-0.60, +0.14] | [-0.51, +0.08] | non-inferior only | equivalent | neither |
| W | N | +0.56 [-0.18, +1.28] | [-0.29, +1.41] | inconclusive | inconclusive | neither |
| LS | N | -0.78 [-1.19, -0.41] | [-1.17, -0.39] | non-inferior only | non-inferior only | ahead |
| LW | N | -1.40 [-2.14, -0.83] | [-2.23, -0.57] | non-inferior only | non-inferior only | ahead |
| SW | N | -0.12 [-0.60, +0.33] | [-0.58, +0.34] | non-inferior only | equivalent | neither |
| LSW | N | -1.00 [-1.47, -0.60] | [-1.37, -0.63] | non-inferior only | non-inferior only | ahead |
| pooled | - | -0.46 [-0.88, -0.11] | [-0.98, +0.07] | non-inferior only | equivalent | ahead |

- B3, 0.5% margin: equivalent 0; non-inferior only 6 (L, S, LS, LW, SW, LSW); inferior 0; inconclusive 2 (N, W)
- B3, 1% margin: equivalent 3 (N, S, SW); non-inferior only 4 (L, LS, LW, LSW); inferior 0; inconclusive 1 (W)
- B3: interval below zero (policy shorter) on L, LS, LW, LSW; above zero on none.

## B4: the eight specialists, each under its own combination (the principal comparison with five seeds)

| comb. | served by | policy - option | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| N | N | +0.24 [-0.40, +0.86] | [-0.54, +1.03] | inconclusive | equivalent | neither |
| L | L | +0.18 [-0.11, +0.49] | [-0.06, +0.43] | equivalent | equivalent | neither |
| S | S | -0.35 [-0.80, +0.06] | [-0.90, +0.19] | non-inferior only | equivalent | neither |
| W | W | +0.40 [-0.51, +1.31] | [-0.82, +1.61] | inconclusive | inconclusive | neither |
| LS | LS | -0.07 [-0.38, +0.25] | [-0.39, +0.25] | equivalent | equivalent | neither |
| LW | LW | +0.46 [+0.06, +0.88] | [+0.00, +0.91] | inferior | equivalent | behind |
| SW | SW | -0.47 [-0.89, -0.08] | [-0.71, -0.23] | non-inferior only | equivalent | ahead |
| LSW | LSW | -0.15 [-0.53, +0.22] | [-0.48, +0.19] | non-inferior only | equivalent | neither |
| pooled | - | +0.03 [-0.18, +0.24] | [-0.21, +0.27] | equivalent | equivalent | neither |

- B4, 0.5% margin: equivalent 2 (L, LS); non-inferior only 3 (S, SW, LSW); inferior 1 (LW); inconclusive 2 (N, W)
- B4, 1% margin: equivalent 7 (N, L, S, LS, LW, SW, LSW); non-inferior only 0; inferior 0; inconclusive 1 (W)
- B4: interval below zero (policy shorter) on SW; above zero on LW.

B4 reproduces the five-seed principal comparison of results/family_inference.json: 88 numbers and words identical at the printed digits (largest absolute difference 0.00e+00).

## Oracle pairs

For each pair of specialists, each combination is served by the one of the two with the higher mean of the policy minus the specialist on the test instances (an oracle choice made on the test instances). Columns: the mean over the eight combinations of the policy minus the option (in percent of the option) and of the option minus the policy (in percent of the policy), and the serving specialist of N, L, S, W, LS, LW, SW, LSW. Sorted by the first column, highest first: the first pair gives the shortest schedules relative to the policy.

| rank | pair | policy - option | option - policy | served by |
|---|---|---|---|---|
| 1 | N, LW | +0.123 | -0.048 | N LW N N LW LW N LW |
| 2 | W, LW | +0.101 | -0.025 | W LW W W LW LW W LW |
| 3 | N, LSW | +0.036 | +0.038 | N LSW N N LSW LSW N LSW |
| 4 | W, LSW | +0.014 | +0.062 | W LSW W W LSW LSW W LSW |
| 5 | N, L | -0.001 | +0.078 | N L N N L L N L |
| 6 | L, W | -0.023 | +0.102 | W L W W L L W L |
| 7 | N, LS | -0.113 | +0.192 | N LS N N LS LS N LS |
| 8 | W, LS | -0.135 | +0.216 | W LS W W LS LS W LS |
| 9 | N, S | -0.278 | +0.362 | N S N N S S N S |
| 10 | S, W | -0.300 | +0.386 | W S W W S S W S |
| 11 | N, W | -0.389 | +0.487 | N N W N N N W W |
| 12 | N, SW | -0.413 | +0.510 | N N N N SW N N SW |
| 13 | S, LW | -0.452 | +0.539 | S LW S S LW LW S LW |
| 14 | LW, SW | -0.460 | +0.546 | SW LW SW SW LW LW SW LW |
| 15 | W, SW | -0.464 | +0.566 | W SW W W SW SW W SW |
| 16 | S, LSW | -0.539 | +0.626 | S LSW S S LSW LSW S LSW |
| 17 | SW, LSW | -0.547 | +0.633 | SW LSW SW SW LSW LSW SW LSW |
| 18 | L, S | -0.576 | +0.665 | S L S S L L S L |
| 19 | L, SW | -0.584 | +0.672 | SW L SW SW L L SW L |
| 20 | S, LS | -0.684 | +0.773 | S LS S S LS LS LS LS |
| 21 | LS, LW | -0.687 | +0.784 | LS LW LS LS LW LW LS LW |
| 22 | LS, SW | -0.696 | +0.787 | SW LS SW SW LS LS SW LS |
| 23 | LS, LSW | -0.773 | +0.871 | LS LSW LS LS LS LSW LS LSW |
| 24 | L, LS | -0.774 | +0.873 | LS L LS LS LS L LS LS |
| 25 | S, SW | -0.824 | +0.922 | S S SW S S S SW S |
| 26 | L, LW | -1.203 | +1.368 | L LW LW L LW LW L LW |
| 27 | L, LSW | -1.268 | +1.430 | L L L L LSW LSW LSW LSW |
| 28 | LW, LSW | -1.566 | +1.781 | LSW LW LW LSW LW LW LSW LSW |

Best pair: N, LW; by the option minus the policy the lowest mean is the pair N, LW. The pair of B1 (N, LSW) ranks 3 and the pair of B2 (N, L) ranks 5 of 28.

