# Specialists that share the policy's training budget (environment exact, training seeds 301,302,303,304,305,306,307,308,309,310 (10 run(s) per combination))

The comparison is not planned in the statistical protocol.

Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = 20000).

## Training effort (one training seed)

| | runs | updates per run | episodes per update | episodes per run | updates | episodes |
|---|---|---|---|---|---|---|
| policy | 1 | 2000 | 20 | 40000 | 2000 | 40000 |
| budget-sharing specialists (250 updates) | 8 | 250 | 20 | 5000 | 2000 | 40000 |
| specialists (1000 updates) | 8 | 1000 | 20 | 20000 | 8000 | 160000 |

Episodes over those of the policy: budget-sharing specialists 1, specialists 4.

## The policy against each specialist

Positive: the policy is slower. Against the specialist (1000 updates): p88.parity_components on the same seeds (equal to the record: out/exact/results/seed_extension.json). Combined verdict: p88.superiority_block over the eight combinations, its interval from fresh draws. Random streams: p110.TEN_SEED, the streams of the ten-seed record.

| combination | 250 updates: mean | 95% CI | 0.5% | 1% | direction | combined verdict | 1000 updates: mean | 95% CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.45 | [-1.01, +0.05] | non-inferior only | non-inferior only | neither | not distinguishable [-1.01, +0.05] | +0.56 | [+0.04, +1.07] | inferior | inferior | behind |
| L | +0.06 | [-0.27, +0.38] | equivalent | equivalent | neither | not distinguishable [-0.27, +0.39] | +0.15 | [-0.09, +0.37] | equivalent | equivalent | neither |
| S | -0.65 | [-1.11, -0.18] | non-inferior only | non-inferior only | ahead | ahead [-1.12, -0.18] | -0.40 | [-0.76, -0.05] | non-inferior only | equivalent | ahead |
| W | -1.07 | [-1.81, -0.39] | non-inferior only | non-inferior only | ahead | ahead [-1.81, -0.39] | +0.34 | [-0.26, +0.97] | inconclusive | equivalent | neither |
| LS | -0.75 | [-1.07, -0.45] | non-inferior only | non-inferior only | ahead | ahead [-1.06, -0.45] | -0.23 | [-0.49, +0.02] | equivalent | equivalent | neither |
| LW | -0.32 | [-0.65, -0.01] | non-inferior only | equivalent | ahead | not distinguishable [-0.64, -0.01] | +0.20 | [-0.14, +0.55] | inconclusive | equivalent | neither |
| SW | -0.87 | [-1.31, -0.43] | non-inferior only | non-inferior only | ahead | ahead [-1.31, -0.42] | -0.31 | [-0.65, +0.06] | non-inferior only | equivalent | neither |
| LSW | -0.70 | [-1.01, -0.39] | non-inferior only | non-inferior only | ahead | ahead [-1.01, -0.39] | -0.26 | [-0.55, +0.02] | non-inferior only | equivalent | neither |
| pooled, All | -0.59 | [-0.80, -0.40] | non-inferior only | equivalent | ahead | | +0.01 | [-0.17, +0.19] | equivalent | equivalent | neither |
| pooled, Train | -0.53 | [-0.76, -0.31] | non-inferior only | equivalent | ahead | | +0.10 | [-0.09, +0.31] | equivalent | equivalent | neither |
| pooled, Held | -0.78 | [-1.06, -0.51] | non-inferior only | non-inferior only | ahead | | -0.28 | [-0.50, -0.05] | non-inferior only | equivalent | ahead |

Per-seed means under N (%, seed s against seed s; positive: the policy is slower):

| | s301 | s302 | s303 | s304 | s305 | s306 | s307 | s308 | s309 | s310 |
|---|---|---|---|---|---|---|---|---|---|---|
| against the budget-sharing specialist (250 updates) | -0.66 | -2.00 | -0.37 | -1.06 | +0.30 | -0.10 | -0.05 | +0.41 | -1.09 | +0.08 |
| against the specialist (1000 updates) | +0.41 | -0.49 | +0.32 | -0.19 | +1.16 | +0.72 | +1.22 | +0.16 | +0.71 | +1.59 |

Combined verdict of the policy against the budget-sharing specialist: ahead 5 (S, W, LS, SW, LSW); behind 0 (none); not distinguishable 3 (N, L, LW).
Margin verdicts that differ between the two specialists: at 0.5% N, W, LS, LW; at 1% N, S, W, LS, SW, LSW.

| joint claim | margin | 250 updates | 1000 updates |
|---|---|---|---|
| equivalent on All (N, L, S, W, LS, LW, SW, LSW) | 0.5% | fails: N, S, W, LS, LW, SW, LSW | fails: N, S, W, LW, SW, LSW |
| non-inferior on All (N, L, S, W, LS, LW, SW, LSW) | 0.5% | holds | fails: N, W, LW |
| equivalent on All (N, L, S, W, LS, LW, SW, LSW) | 1.0% | fails: N, S, W, LS, SW, LSW | fails: N |
| non-inferior on All (N, L, S, W, LS, LW, SW, LSW) | 1.0% | holds | fails: N |
| equivalent on Train (N, L, S, W, LS, LW) | 0.5% | fails: N, S, W, LS, LW | fails: N, S, W, LW |
| non-inferior on Train (N, L, S, W, LS, LW) | 0.5% | holds | fails: N, W, LW |
| equivalent on Train (N, L, S, W, LS, LW) | 1.0% | fails: N, S, W, LS | fails: N |
| non-inferior on Train (N, L, S, W, LS, LW) | 1.0% | holds | fails: N |
| equivalent on Held (SW, LSW) | 0.5% | fails: SW, LSW | fails: SW, LSW |
| non-inferior on Held (SW, LSW) | 0.5% | holds | holds |
| equivalent on Held (SW, LSW) | 1.0% | fails: SW, LSW | holds |
| non-inferior on Held (SW, LSW) | 1.0% | holds | holds |

## The budget-sharing specialist against the specialist (1000 updates)

Positive: longer schedules after 250 updates.

| combination | mean | 95% CI | within 0.5% | direction |
|---|---|---|---|---|
| N | +1.05 | [+0.64, +1.44] | no | behind |
| L | +0.11 | [-0.27, +0.46] | yes | neither |
| S | +0.30 | [-0.12, +0.75] | no | neither |
| W | +1.52 | [+0.81, +2.31] | no | behind |
| LS | +0.56 | [+0.25, +0.86] | no | behind |
| LW | +0.57 | [+0.24, +0.93] | no | behind |
| SW | +0.65 | [+0.15, +1.16] | no | behind |
| LSW | +0.48 | [+0.14, +0.84] | no | behind |
| pooled, All | +0.65 | [+0.50, +0.80] | no | behind |
| pooled, Train | +0.68 | [+0.50, +0.87] | no | behind |
| pooled, Held | +0.56 | [+0.28, +0.84] | no | behind |

Interval above zero: N, W, LS, LW, SW, LSW; below zero: none.
