# Policies trained with other held-out pairs of combinations (environment exact, training seeds 301,302,303 (3 run(s) per policy))

Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = 20000). Policy b trains on N, L, S, W, LW, SW; policy c on N, L, S, W, LS, SW; the policy (xmask-v1) on N, L, S, W, LS, LW.

## Each policy against the specialist on its held-out combinations

Positive: the policy is slower than the specialist.

| policy | held-out combination | mean | 95% CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| policy b | LS | -0.03 | [-0.40, +0.37] | equivalent | equivalent | neither |
| policy b | LSW | +0.06 | [-0.37, +0.52] | inconclusive | equivalent | neither |
| policy c | LW | +0.63 | [+0.16, +1.06] | inferior | inferior | behind |
| policy c | LSW | +0.02 | [-0.38, +0.46] | equivalent | equivalent | neither |
| policy (xmask-v1) | SW | -0.46 | [-1.06, +0.10] | non-inferior only | non-inferior only | neither |
| policy (xmask-v1) | LSW | +0.02 | [-0.35, +0.42] | equivalent | equivalent | neither |

## Pooled against the specialists

Training and held-out combinations of each policy; base instances kept together.

| policy | combinations | mean | 95% CI | 0.5% | 1% | equivalent on every one at 0.5% | at 1% |
|---|---|---|---|---|---|---|---|
| policy b | N, L, S, W, LW, SW | -0.12 | [-0.48, +0.26] | equivalent | equivalent | no | no |
| policy b | LS, LSW | +0.01 | [-0.31, +0.38] | equivalent | equivalent | no | yes |
| policy b | N, L, S, W, LS, LW, SW, LSW | -0.09 | [-0.40, +0.27] | equivalent | equivalent | no | no |
| policy c | N, L, S, W, LS, SW | +0.22 | [-0.07, +0.49] | equivalent | equivalent | no | no |
| policy c | LW, LSW | +0.32 | [+0.06, +0.58] | inferior | equivalent | no | no |
| policy c | N, L, S, W, LS, LW, SW, LSW | +0.25 | [+0.02, +0.46] | equivalent | equivalent | no | no |
| policy (xmask-v1) | N, L, S, W, LS, LW | +0.09 | [-0.18, +0.37] | equivalent | equivalent | no | no |
| policy (xmask-v1) | SW, LSW | -0.22 | [-0.60, +0.16] | non-inferior only | equivalent | no | no |
| policy (xmask-v1) | N, L, S, W, LS, LW, SW, LSW | +0.02 | [-0.23, +0.27] | equivalent | equivalent | no | no |

## Policies b and c against the policy (xmask-v1) of the same seed

Positive: the split policy is slower than the policy.

| policy | combination | mean | 95% CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|---|
| policy b | LS | +0.08 | [-0.22, +0.40] | equivalent | equivalent | neither |
| policy b | LSW | +0.07 | [-0.32, +0.44] | equivalent | equivalent | neither |
| policy c | LW | +0.42 | [+0.04, +0.80] | inferior | equivalent | behind |
| policy c | LSW | +0.02 | [-0.34, +0.40] | equivalent | equivalent | neither |

## Every combination (supplement table)

Each policy against the specialist; held-out combinations marked with *.

| combination | policy b: mean | 95% CI | 0.5% | 1% | policy c: mean | 95% CI | 0.5% | 1% | policy (xmask-v1): mean | 95% CI | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | -0.20 | [-1.00, +0.51] | inconclusive | non-inferior only | +0.66 | [-0.30, +1.46] | inconclusive | inconclusive | +0.08 | [-0.61, +0.66] | inconclusive | equivalent |
| L | +0.33 | [-0.05, +0.67] | inconclusive | equivalent | +0.16 | [-0.14, +0.45] | equivalent | equivalent | +0.09 | [-0.24, +0.42] | equivalent | equivalent |
| S | -0.29 | [-0.65, +0.09] | non-inferior only | equivalent | +0.16 | [-0.25, +0.57] | inconclusive | equivalent | -0.11 | [-0.47, +0.26] | equivalent | equivalent |
| W | -0.43 | [-1.56, +0.69] | inconclusive | non-inferior only | +0.47 | [-0.61, +1.49] | inconclusive | inconclusive | +0.36 | [-0.83, +1.47] | inconclusive | inconclusive |
| LS | -0.03* | [-0.40, +0.37] | equivalent | equivalent | +0.13 | [-0.36, +0.57] | inconclusive | equivalent | -0.09 | [-0.53, +0.36] | non-inferior only | equivalent |
| LW | +0.50 | [+0.12, +0.90] | inferior | equivalent | +0.63* | [+0.16, +1.06] | inferior | inferior | +0.24 | [-0.15, +0.63] | inconclusive | equivalent |
| SW | -0.61 | [-1.23, -0.01] | non-inferior only | non-inferior only | -0.24 | [-0.84, +0.33] | non-inferior only | equivalent | -0.46* | [-1.06, +0.10] | non-inferior only | non-inferior only |
| LSW | +0.06* | [-0.37, +0.52] | inconclusive | equivalent | +0.02* | [-0.38, +0.46] | equivalent | equivalent | +0.02* | [-0.35, +0.42] | equivalent | equivalent |

Held-out combinations of policies b and c: b:LS, b:LSW, c:LW, c:LSW. Equivalent at 0.5%: b:LS, c:LSW; at 1%: b:LS, b:LSW, c:LSW. Interval above zero: c:LW.

The policy (xmask-v1) against the specialist: p88.parity_components on the same seeds (the record holds seeds [301, 302, 303, 304, 305]: out/exact/results/family_inference.json).
