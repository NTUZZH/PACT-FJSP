# The policy against the best dispatching rule outside the training ranges (environment exact, training seeds 301,302,303,304,305)

d = 100 (policy_s - best) / best per training seed s and test instance, best = per-instance minimum of FIFO, MOR, SPT, MWKR; negative = the policy is shorter (p88.seed_pairs, p88.released, p88.pool_reader). crossed hierarchical bootstrap over training seeds and test instances, B = 20000, fresh default_rng(11) per combination (p88.fresh), 2.5th to 97.5th percentile. one-sided seed-level t (k - 1 df) and one-sided bootstrap p, both directions, Holm over the set's family, each kind and direction its own family (p88.superiority_block). Verdict: ahead: interval below zero and both Holm-adjusted p for H1 policy shorter below 0.05; behind: the mirrored condition; not distinguishable otherwise (p89_public_family.superiority_verdict).

Inside the training ranges the Holm family is the eight combinations, and every row equals FamRule<R> of `out/exact/results/family_inference.json` and `out/exact/paper/macros_family.tex` (96 comparisons: mean, interval ends, Holm-adjusted p-values and verdict exactly, the macros as printed).

Control: 72 arrays of the policy and of the four rules on the combinations a shift does not reach are bit-identical to those inside the ranges (largest change of the mean there 0.00). Schedules rejected by the validator, three sets, eight combinations, every seed: 0.

## Larger asymmetric setups (10x25_oodsetup)

Cross-class setups drawn from U[100%, 200%] of the mean processing time (trained U[50%, 100%]), the matrix asymmetric; mean setup entry 1.76 times that of 10x25. Holm family outside the ranges: S, LS, SW, LSW.

| comb. | inside the ranges | verdict | outside the ranges | verdict | Holm p shorter (t / boot) | Holm p longer (t / boot) | infeasible |
|---|---|---|---|---|---|---|---|
| S | -0.52 [-1.08, +0.08] | not distinguishable | -0.75 [-1.60, +0.10] | not distinguishable | 0.043 / 0.086 | 1 / 1 | 0 |
| LS | -1.10 [-1.53, -0.65] | ahead | -1.15 [-1.67, -0.63] | ahead | 0.002 / 0.0002 | 1 / 1 | 0 |
| SW | -0.98 [-1.47, -0.50] | ahead | +0.15 [-0.64, +1.00] | not distinguishable | 0.96 / 0.63 | 0.15 / 1 | 0 |
| LSW | -1.21 [-1.65, -0.77] | ahead | -0.77 [-1.26, -0.25] | ahead | 0.0098 / 0.0096 | 1 / 1 | 0 |

Outside the ranges the policy is ahead on LS, LSW, not distinguishable on S, SW and behind on none. Not reached, values as inside the ranges: N, L, W, LW.

Pooled over S, LS, SW, LSW (a drawn test instance brings all its combinations): -0.63 [-1.04, -0.22], direction ahead.

## Denser machine unavailability windows (10x25_oodwindow)

Outage density per machine drawn from U[20%, 35%] of the horizon (trained U[5%, 15%]); mean outage density 2.53 times that of 10x25. Holm family outside the ranges: W, LW, SW, LSW.

| comb. | inside the ranges | verdict | outside the ranges | verdict | Holm p shorter (t / boot) | Holm p longer (t / boot) | infeasible |
|---|---|---|---|---|---|---|---|
| W | -2.00 [-2.66, -1.33] | ahead | -3.08 [-3.96, -2.24] | ahead | 0.00038 / 0.0002 | 1 / 1 | 0 |
| LW | -2.10 [-2.63, -1.57] | ahead | -3.32 [-4.05, -2.59] | ahead | 0.00038 / 0.0002 | 1 / 1 | 0 |
| SW | -0.98 [-1.47, -0.50] | ahead | -0.98 [-1.51, -0.43] | ahead | 0.00054 / 0.00045 | 1 / 1 | 0 |
| LSW | -1.21 [-1.65, -0.77] | ahead | -1.11 [-1.64, -0.55] | ahead | 0.00054 / 0.0002 | 1 / 1 | 0 |

Outside the ranges the policy is ahead on W, LW, SW, LSW, not distinguishable on none and behind on none. Not reached, values as inside the ranges: N, L, S, LS.

Pooled over W, LW, SW, LSW (a drawn test instance brings all its combinations): -2.12 [-2.55, -1.68], direction ahead.

The policy against the specialists, seed against seed, pooled over W, LW, SW, LSW: +0.21 [-0.18, +0.60], direction neither, inconclusive at 0.5%, equivalent at 1% (20 arrays of the specialists on the combinations not reached equal to those inside the ranges).
