# Per-floor batch, 27x25 (environment exact, training seeds 301,302,303,304,305)

Arrays of 27x25: `test_results/FAMILY/27x25_xmask`; 50 test instances per combination. 20x25 and 30x25: the record `out/exact/results/family_inference.json` of p88, section H, reproduced from the arrays in 1078 numbers. B = 20000.

d = 100 (policy - other) / other per training seed and test instance, seed s against seed s; negative = the policy is shorter. Verdicts at the margins 0.5% and 1% from the hierarchical 95% interval; combined verdict against the best dispatching rule with Holm over the eight combinations of a size.

## Policy against the specialist of the combination

| comb. | 20x25 | 0.5% | 1% | 27x25 | 0.5% | 1% | 30x25 | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.51 [-1.59, +2.01] | inconclusive | inconclusive | -0.20 [-3.04, +1.66] | inconclusive | inconclusive | -0.22 [-2.93, +1.82] | inconclusive | inconclusive |
| L | -2.15 [-3.47, -0.89] | non-inferior only | non-inferior only | -2.54 [-5.42, -0.62] | non-inferior only | non-inferior only | -2.61 [-5.49, -0.33] | non-inferior only | non-inferior only |
| S | -0.59 [-1.52, +0.18] | non-inferior only | non-inferior only | -0.52 [-1.31, +0.32] | non-inferior only | non-inferior only | -0.59 [-1.63, +0.36] | non-inferior only | non-inferior only |
| W | +1.28 [-1.05, +3.09] | inconclusive | inconclusive | +0.64 [-1.53, +2.13] | inconclusive | inconclusive | +1.01 [-1.07, +2.82] | inconclusive | inconclusive |
| LS | -0.15 [-1.21, +0.91] | inconclusive | non-inferior only | -0.25 [-1.16, +0.66] | inconclusive | non-inferior only | -0.27 [-1.46, +0.86] | inconclusive | non-inferior only |
| LW | -3.19 [-6.06, -1.20] | non-inferior only | non-inferior only | -5.07 [-9.05, -2.04] | non-inferior only | non-inferior only | -5.24 [-9.78, -1.48] | non-inferior only | non-inferior only |
| SW | -0.37 [-1.34, +0.52] | inconclusive | non-inferior only | -0.70 [-1.71, +0.20] | non-inferior only | non-inferior only | -0.89 [-2.52, +0.56] | inconclusive | non-inferior only |
| LSW | -0.26 [-1.34, +0.62] | inconclusive | non-inferior only | -0.07 [-0.94, +0.72] | inconclusive | equivalent | +0.08 [-1.39, +1.27] | inconclusive | inconclusive |
| pooled All | -0.61 [-1.12, -0.12] | non-inferior only | non-inferior only | -1.09 [-1.91, -0.26] | non-inferior only | non-inferior only | -1.09 [-2.16, -0.05] | non-inferior only | non-inferior only |
| pooled Train | -0.72 [-1.30, -0.16] | non-inferior only | non-inferior only | -1.33 [-2.22, -0.42] | non-inferior only | non-inferior only | -1.32 [-2.52, -0.09] | non-inferior only | non-inferior only |
| pooled Held | -0.31 [-1.27, +0.47] | non-inferior only | non-inferior only | -0.39 [-1.27, +0.38] | non-inferior only | non-inferior only | -0.41 [-1.84, +0.69] | inconclusive | non-inferior only |

| joint claim (IUT) | 20x25 | 27x25 | 30x25 |
|---|---|---|---|
| equivalent on All at +-0.5% | fails on N,L,S,W,LS,LW,SW,LSW | fails on N,L,S,W,LS,LW,SW,LSW | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-0.5% | fails on N,W,LS,SW,LSW | fails on N,W,LS,LSW | fails on N,W,LS,SW,LSW |
| equivalent on All at +-1% | fails on N,L,S,W,LS,LW,SW,LSW | fails on N,L,S,W,LS,LW,SW | fails on N,L,S,W,LS,LW,SW,LSW |
| non-inferior on All at +-1% | fails on N,W | fails on N,W | fails on N,W,LSW |
| equivalent on Train at +-0.5% | fails on N,L,S,W,LS,LW | fails on N,L,S,W,LS,LW | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-0.5% | fails on N,W,LS | fails on N,W,LS | fails on N,W,LS |
| equivalent on Train at +-1% | fails on N,L,S,W,LS,LW | fails on N,L,S,W,LS,LW | fails on N,L,S,W,LS,LW |
| non-inferior on Train at +-1% | fails on N,W | fails on N,W | fails on N,W |
| equivalent on Held at +-0.5% | fails on SW,LSW | fails on SW,LSW | fails on SW,LSW |
| non-inferior on Held at +-0.5% | fails on SW,LSW | fails on LSW | fails on SW,LSW |
| equivalent on Held at +-1% | fails on SW,LSW | fails on SW | fails on SW,LSW |
| non-inferior on Held at +-1% | holds | holds | fails on LSW |

## Policy against the best dispatching rule

| comb. | 20x25 | verdict | 27x25 | verdict | 30x25 | verdict |
|---|---|---|---|---|---|---|
| N | -2.24 [-3.13, -1.40] | ahead | -3.03 [-4.01, -2.09] | ahead | -2.97 [-4.30, -1.69] | ahead |
| L | -2.66 [-3.32, -1.99] | ahead | -2.63 [-3.56, -1.76] | ahead | -3.86 [-4.96, -2.81] | ahead |
| S | -2.29 [-3.19, -1.36] | ahead | -2.70 [-3.81, -1.62] | ahead | -3.10 [-4.16, -2.04] | ahead |
| W | -2.95 [-3.90, -2.03] | ahead | -3.44 [-4.19, -2.69] | ahead | -3.03 [-4.05, -1.90] | ahead |
| LS | -1.57 [-2.28, -0.83] | ahead | -1.91 [-2.67, -1.10] | ahead | -2.89 [-4.32, -1.21] | ahead |
| LW | -3.23 [-3.83, -2.65] | ahead | -3.64 [-4.42, -2.84] | ahead | -3.46 [-4.55, -2.32] | ahead |
| SW | -1.88 [-2.72, -0.95] | ahead | -2.29 [-3.21, -1.39] | ahead | -3.25 [-4.56, -1.95] | ahead |
| LSW | -2.41 [-3.19, -1.63] | ahead | -2.65 [-3.48, -1.81] | ahead | -2.76 [-3.80, -1.70] | ahead |
| pooled | -2.41 [-2.80, -1.98] | ahead | -2.79 [-3.29, -2.28] | ahead | -3.16 [-3.90, -2.43] | ahead |
| verdict counts (ahead, behind, not distinguishable) | 8, 0, 0 |  | 8, 0, 0 |  | 8, 0, 0 |  |

## The per-floor batch between the two neighbouring sizes

- policy vs specialist pooled: -0.61 at 20 modules, -1.09 at 27, -1.09 at 30; between the two: yes.
- policy vs best rule pooled: -2.41 at 20 modules, -2.79 at 27, -3.16 at 30; between the two: yes.
