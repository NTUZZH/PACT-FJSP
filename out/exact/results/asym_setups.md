# Asymmetric setups at the trained magnitude (environment exact, training seeds 301,302,303,304,305)

Sets: 10x25_asymsetup (trained magnitude, asymmetric), beside 10x25_oodsetup (twice the magnitude, asymmetric) and 10x25 (in distribution); 100 test instances per combination. Arrays: `test_results/FAMILY/10x25_xmask`, `test_results/FAMILY/10x25_oodsetup_xmask`, `test_results/FAMILY/10x25_asymsetup_xmask`.

Control: 112 arrays on N, L, W and LW (policy, specialists, the four dispatching rules) are bit-identical to the in-distribution arrays.

| | 10x25_asymsetup | 10x25_oodsetup |
|---|---|---|
| mean setup entry over that of 10x25 | 1.00 | 1.76 |
| matrix of instance_000 asymmetric | True | True |
| schedules rejected by the validator | 0 | 0 |

## Policy against the best dispatching rule (the procedure of FamRule<R>)

d = 100 (policy_s - best) / best per training seed s and test instance, best = the per-instance best of FIFO, MOR, SPT, MWKR; negative = the policy is shorter; mean, hierarchical 95% interval and combined verdict (ahead, behind, not distinguishable) with Holm over the family of the set (p33, the procedure of FamRule<R>). B = 20000, fresh default_rng(11) per combination; Holm families: in distribution the eight combinations, 10x25_asymsetup and 10x25_oodsetup S, LS, SW, LSW; on N, L, W, LW the values in distribution (identical arrays). The 10x25_oodsetup and in-distribution rows equal p33's record `out/exact/results/ood.json` (160 comparisons, exact).

| comb. | in distribution | verdict | 10x25_asymsetup | verdict | drift | 10x25_oodsetup | verdict | drift |
|---|---|---|---|---|---|---|---|---|
| N | -1.86 [-2.36, -1.36] | ahead | -1.86 [-2.36, -1.36] | ahead | +0.00 | -1.86 [-2.36, -1.36] | ahead | +0.00 |
| L | -1.36 [-1.91, -0.82] | ahead | -1.36 [-1.91, -0.82] | ahead | +0.00 | -1.36 [-1.91, -0.82] | ahead | +0.00 |
| S | -0.52 [-1.08, +0.08] | not distinguishable | -0.77 [-1.25, -0.30] | ahead | -0.25 | -0.75 [-1.60, +0.10] | not distinguishable | -0.23 |
| W | -2.00 [-2.66, -1.33] | ahead | -2.00 [-2.66, -1.33] | ahead | +0.00 | -2.00 [-2.66, -1.33] | ahead | +0.00 |
| LS | -1.10 [-1.53, -0.65] | ahead | -0.95 [-1.30, -0.62] | ahead | +0.15 | -1.15 [-1.67, -0.63] | ahead | -0.05 |
| LW | -2.10 [-2.63, -1.57] | ahead | -2.10 [-2.63, -1.57] | ahead | +0.00 | -2.10 [-2.63, -1.57] | ahead | +0.00 |
| SW | -0.98 [-1.47, -0.50] | ahead | -1.00 [-1.52, -0.45] | ahead | -0.02 | +0.15 [-0.64, +1.00] | not distinguishable | +1.13 |
| LSW | -1.21 [-1.65, -0.77] | ahead | -0.88 [-1.37, -0.41] | ahead | +0.33 | -0.77 [-1.26, -0.25] | ahead | +0.44 |

Pooled over S, LS, SW, LSW (a drawn test instance brings all its combinations): 10x25_asymsetup -0.90 [-1.17, -0.65], direction ahead; 10x25_oodsetup -0.63 [-1.04, -0.22], direction ahead (equal to the pooled row of `out/exact/results/ood.json`, exact).

| treated cells (S, LS, SW, LSW) | 10x25_asymsetup | 10x25_oodsetup |
|---|---|---|
| lead shrank | LS, LSW | SW, LSW |
| lead grew or unchanged | S, SW | S, LS |
| no longer ahead (verdict) | none | SW |
| mean changed sign | none | SW |
| verdict ahead | S, LS, SW, LSW | LS, LSW |
| verdict behind | none | none |
| verdict not distinguishable | none | S, SW |
| largest placebo drift | 0.00 | 0.00 |

## Policy against the specialist of the combination

100 (policy - specialist) / specialist, seed s against seed s (p88 component, pooled_block); B = 20000.

| comb. | in distribution | 0.5% | 1% | 10x25_asymsetup | 0.5% | 1% | 10x25_oodsetup | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.24 [-0.40, +0.86] | inconclusive | equivalent | +0.24 [-0.40, +0.86] | inconclusive | equivalent | +0.24 [-0.40, +0.86] | inconclusive | equivalent |
| L | +0.18 [-0.11, +0.48] | equivalent | equivalent | +0.18 [-0.11, +0.48] | equivalent | equivalent | +0.18 [-0.11, +0.48] | equivalent | equivalent |
| S | -0.35 [-0.81, +0.07] | non-inferior only | equivalent | -0.41 [-0.96, +0.11] | non-inferior only | equivalent | -0.04 [-0.96, +0.80] | inconclusive | equivalent |
| W | +0.40 [-0.51, +1.31] | inconclusive | inconclusive | +0.40 [-0.51, +1.31] | inconclusive | inconclusive | +0.40 [-0.51, +1.31] | inconclusive | inconclusive |
| LS | -0.07 [-0.37, +0.25] | equivalent | equivalent | -0.14 [-0.40, +0.12] | equivalent | equivalent | -0.01 [-0.58, +0.55] | inconclusive | equivalent |
| LW | +0.46 [+0.04, +0.88] | inferior | equivalent | +0.46 [+0.04, +0.88] | inferior | equivalent | +0.46 [+0.04, +0.88] | inferior | equivalent |
| SW | -0.47 [-0.88, -0.08] | non-inferior only | equivalent | -0.20 [-0.71, +0.37] | non-inferior only | equivalent | +0.21 [-0.53, +0.98] | inconclusive | equivalent |
| LSW | -0.15 [-0.53, +0.23] | non-inferior only | equivalent | +0.26 [-0.12, +0.65] | inconclusive | equivalent | -0.30 [-0.85, +0.27] | non-inferior only | equivalent |
| pooled Setup | -0.26 [-0.49, -0.02] | equivalent | equivalent | -0.12 [-0.36, +0.13] | equivalent | equivalent | -0.04 [-0.57, +0.45] | non-inferior only | equivalent |
| pooled All | +0.03 [-0.18, +0.24] | equivalent | equivalent | +0.10 [-0.08, +0.27] | equivalent | equivalent | +0.14 [-0.18, +0.44] | equivalent | equivalent |
