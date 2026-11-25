# supporting arms and HGNN, paired inference

exact environment, 10x25 test set (100 base instances per combination). d = 100 (a - b) / b, a the first-named method, seed s against seed s; negative = first-named method shorter. 95% intervals: crossed hierarchical bootstrap over seeds and base instances (B = 20000, default_rng(11)), one-run bootstrap over base instances (default_rng(7)); verdicts at 0.5% and 1%; joint claims by the intersection-union test; HGNN superiority by Holm over the eight combinations.

| block | status | seeds | reason |
|---|---|---|---|
| policy | computed | 301,302,303 | - |
| eq | computed | 301,302,303 | - |
| sparse | computed | 301,302,303 | - |
| dzero | computed | 301,302,303 | - |
| dmax | computed | 301,302,303 | - |
| film | computed | 301,302,303 | - |
| dr | computed | 301,302,303 | - |
| all8 | computed | 301,302,303 | - |
| pair | computed | 301,302,303 | - |
| wthird | computed | 301,302,303 | - |
| hgnn | computed | 301,302,303,304,305 | - |
| hgnnpub | computed | 301,302,303 | - |
| hgnnsolo (optional) | skipped | - | no result array of the arm exists (0 of 8); checkpoints 0 of 8 |

HGNN learning rate: 2e-4 (selection: 2e-4)

## policy: policy vs specialist of each combination (seeds of the supporting arms)

| comb. | mean [95% CI] | seed-t CI | 0.5% | 1% | direction |
|---|---|---|---|---|---|
| N | +0.08 [-0.61, +0.66] | [-1.16, +1.32] | inconclusive | equivalent | neither |
| L | +0.09 [-0.25, +0.41] | [-0.13, +0.31] | equivalent | equivalent | neither |
| S | -0.11 [-0.47, +0.26] | [-0.44, +0.22] | equivalent | equivalent | neither |
| W | +0.36 [-0.86, +1.48] | [-2.49, +3.21] | inconclusive | inconclusive | neither |
| LS | -0.09 [-0.53, +0.36] | [-0.99, +0.80] | non-inferior only | equivalent | neither |
| LW | +0.24 [-0.16, +0.62] | [-0.23, +0.71] | inconclusive | equivalent | neither |
| SW | -0.46 [-1.05, +0.09] | [-1.13, +0.20] | non-inferior only | non-inferior only | neither |
| LSW | +0.02 [-0.35, +0.43] | [-0.47, +0.51] | equivalent | equivalent | neither |
| pooled All | +0.02 [-0.23, +0.27] | [-0.51, +0.54] | equivalent | equivalent | neither |
| pooled Train | +0.09 [-0.18, +0.37] | [-0.48, +0.66] | equivalent | equivalent | neither |
| pooled Held | -0.22 [-0.60, +0.16] | [-0.78, +0.34] | non-inferior only | equivalent | neither |

Joint claims: All 0.5 equivalent fails (N,W,LS,LW,SW); All 0.5 non-inferior fails (N,W,LW); All 1.0 equivalent fails (W,SW); All 1.0 non-inferior fails (W); Train 0.5 equivalent fails (N,W,LS,LW); Train 0.5 non-inferior fails (N,W,LW); Train 1.0 equivalent fails (W); Train 1.0 non-inferior fails (W); Held 0.5 equivalent fails (SW); Held 0.5 non-inferior holds; Held 1.0 equivalent fails (SW); Held 1.0 non-inferior holds

Training effort, policy: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.
Training effort, specialist of N (each specialist trains on its combination): 1000 updates x 20 episodes on N = 20000 episodes per combination in expectation.

## eq: xmask-eq vs the policy and vs the specialist of each combination

Arm xmask-eq (released recipe, 6000 updates), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | -0.31 [-0.84, +0.30] | non-inferior only | equivalent | neither | -0.27 [-1.08, +0.61] | inconclusive | non-inferior only | neither | inconclusive / equivalent |
| L | -0.15 [-0.54, +0.26] | non-inferior only | equivalent | neither | -0.09 [-0.52, +0.31] | non-inferior only | equivalent | neither | equivalent / equivalent |
| S | -0.23 [-0.71, +0.28] | non-inferior only | equivalent | neither | -0.37 [-0.86, +0.16] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | -0.55 [-1.28, +0.20] | non-inferior only | non-inferior only | neither | -0.26 [-0.99, +0.44] | non-inferior only | equivalent | neither | inconclusive / inconclusive |
| LS | -0.04 [-0.33, +0.26] | equivalent | equivalent | neither | -0.15 [-0.53, +0.24] | non-inferior only | equivalent | neither | non-inferior only / equivalent |
| LW | -0.03 [-0.41, +0.36] | equivalent | equivalent | neither | +0.18 [-0.24, +0.58] | inconclusive | equivalent | neither | inconclusive / equivalent |
| SW | -0.29 [-0.82, +0.24] | non-inferior only | equivalent | neither | -0.80 [-1.29, -0.30] | non-inferior only | non-inferior only | ahead | non-inferior only / non-inferior only |
| LSW | -0.04 [-0.43, +0.33] | equivalent | equivalent | neither | -0.05 [-0.40, +0.30] | equivalent | equivalent | neither | equivalent / equivalent |
| pooled All | -0.20 [-0.48, +0.06] | equivalent | equivalent | neither | -0.23 [-0.39, -0.06] | equivalent | equivalent | ahead | - |
| pooled Train | -0.22 [-0.51, +0.07] | non-inferior only | equivalent | neither | -0.16 [-0.34, +0.03] | equivalent | equivalent | neither | - |
| pooled Held | -0.17 [-0.51, +0.17] | non-inferior only | equivalent | neither | -0.43 [-0.73, -0.13] | non-inferior only | equivalent | ahead | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,L,S,W,LS,LW,SW); All 0.5 non-inferior fails (N,LW); All 1.0 equivalent fails (N,SW); All 1.0 non-inferior holds; Train 0.5 equivalent fails (N,L,S,W,LS,LW); Train 0.5 non-inferior fails (N,LW); Train 1.0 equivalent fails (N); Train 1.0 non-inferior holds; Held 0.5 equivalent fails (SW); Held 0.5 non-inferior holds; Held 1.0 equivalent fails (SW); Held 1.0 non-inferior holds

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% L,S,W, at 1% N,W.

Training effort, xmask-eq: 6000 updates x 20 episodes on N,L,S,W,LS,LW = 20000 episodes per combination in expectation.

## sparse: xmask-sparse vs the policy and vs the specialist of each combination

Arm xmask-sparse (released recipe, --reward_mode sparse), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | +6.56 [+3.53, +8.72] | inferior | inferior | behind | +6.61 [+3.94, +8.81] | inferior | inferior | behind | inconclusive / equivalent |
| L | +4.68 [+2.80, +6.46] | inferior | inferior | behind | +4.74 [+2.78, +6.53] | inferior | inferior | behind | equivalent / equivalent |
| S | +3.20 [+2.02, +4.37] | inferior | inferior | behind | +3.05 [+1.92, +4.11] | inferior | inferior | behind | equivalent / equivalent |
| W | +7.36 [+4.52, +9.80] | inferior | inferior | behind | +7.68 [+3.42, +10.85] | inferior | inferior | behind | inconclusive / inconclusive |
| LS | +3.71 [+3.10, +4.39] | inferior | inferior | behind | +3.60 [+2.84, +4.49] | inferior | inferior | behind | non-inferior only / equivalent |
| LW | +5.93 [+2.87, +8.61] | inferior | inferior | behind | +6.15 [+3.08, +8.67] | inferior | inferior | behind | inconclusive / equivalent |
| SW | +4.28 [+3.33, +5.29] | inferior | inferior | behind | +3.74 [+2.78, +4.71] | inferior | inferior | behind | non-inferior only / non-inferior only |
| LSW | +4.38 [+3.56, +5.25] | inferior | inferior | behind | +4.38 [+3.44, +5.38] | inferior | inferior | behind | equivalent / equivalent |
| pooled All | +5.01 [+3.87, +6.03] | inferior | inferior | behind | +4.99 [+3.64, +6.19] | inferior | inferior | behind | - |
| pooled Train | +5.24 [+3.57, +6.55] | inferior | inferior | behind | +5.30 [+3.38, +6.78] | inferior | inferior | behind | - |
| pooled Held | +4.33 [+3.63, +5.05] | inferior | inferior | behind | +4.06 [+3.25, +4.85] | inferior | inferior | behind | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,L,S,W,LS,LW,SW,LSW); All 0.5 non-inferior fails (N,L,S,W,LS,LW,SW,LSW); All 1.0 equivalent fails (N,L,S,W,LS,LW,SW,LSW); All 1.0 non-inferior fails (N,L,S,W,LS,LW,SW,LSW); Train 0.5 equivalent fails (N,L,S,W,LS,LW); Train 0.5 non-inferior fails (N,L,S,W,LS,LW); Train 1.0 equivalent fails (N,L,S,W,LS,LW); Train 1.0 non-inferior fails (N,L,S,W,LS,LW); Held 0.5 equivalent fails (SW,LSW); Held 0.5 non-inferior fails (SW,LSW); Held 1.0 equivalent fails (SW,LSW); Held 1.0 non-inferior fails (SW,LSW)

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% N,L,S,W,LS,LW,SW,LSW, at 1% N,L,S,W,LS,LW,SW,LSW.

Training effort, xmask-sparse: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## dzero: xmask-dzero vs the policy and vs the specialist of each combination

Arm xmask-dzero (released recipe, --delta_mode zero (the legacy arm used the descriptor-input variant, use_film True)), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | -0.15 [-0.80, +0.43] | non-inferior only | equivalent | neither | -0.10 [-1.26, +0.82] | inconclusive | non-inferior only | neither | inconclusive / equivalent |
| L | -0.03 [-0.33, +0.26] | equivalent | equivalent | neither | +0.03 [-0.26, +0.32] | equivalent | equivalent | neither | equivalent / equivalent |
| S | -0.19 [-0.63, +0.23] | non-inferior only | equivalent | neither | -0.33 [-0.70, +0.05] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | -0.59 [-1.22, +0.06] | non-inferior only | non-inferior only | neither | -0.30 [-1.26, +0.65] | inconclusive | non-inferior only | neither | inconclusive / inconclusive |
| LS | +0.01 [-0.34, +0.34] | equivalent | equivalent | neither | -0.10 [-0.39, +0.20] | equivalent | equivalent | neither | non-inferior only / equivalent |
| LW | -0.02 [-0.44, +0.39] | equivalent | equivalent | neither | +0.18 [-0.15, +0.53] | inconclusive | equivalent | neither | inconclusive / equivalent |
| SW | +0.04 [-0.44, +0.53] | inconclusive | equivalent | neither | -0.46 [-1.13, +0.20] | non-inferior only | non-inferior only | neither | non-inferior only / non-inferior only |
| LSW | -0.07 [-0.63, +0.49] | non-inferior only | equivalent | neither | -0.07 [-0.62, +0.44] | non-inferior only | equivalent | neither | equivalent / equivalent |
| pooled All | -0.13 [-0.32, +0.06] | equivalent | equivalent | neither | -0.15 [-0.39, +0.11] | equivalent | equivalent | neither | - |
| pooled Train | -0.16 [-0.43, +0.10] | equivalent | equivalent | neither | -0.10 [-0.36, +0.13] | equivalent | equivalent | neither | - |
| pooled Held | -0.01 [-0.38, +0.33] | equivalent | equivalent | neither | -0.27 [-0.71, +0.18] | non-inferior only | equivalent | neither | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,S,W,LW,SW,LSW); All 0.5 non-inferior fails (N,W,LW); All 1.0 equivalent fails (N,W,SW); All 1.0 non-inferior holds; Train 0.5 equivalent fails (N,S,W,LW); Train 0.5 non-inferior fails (N,W,LW); Train 1.0 equivalent fails (N,W); Train 1.0 non-inferior holds; Held 0.5 equivalent fails (SW,LSW); Held 0.5 non-inferior holds; Held 1.0 equivalent fails (SW); Held 1.0 non-inferior holds

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% S,LS,LSW, at 1% N,W.

Training effort, xmask-dzero: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## dmax: xmask-dmax vs the policy and vs the specialist of each combination

Arm xmask-dmax (released recipe, --delta_mode max (the legacy arm used the descriptor-input variant, use_film True)), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | -0.14 [-0.63, +0.35] | non-inferior only | equivalent | neither | -0.09 [-0.60, +0.43] | non-inferior only | equivalent | neither | inconclusive / equivalent |
| L | +0.07 [-0.35, +0.52] | inconclusive | equivalent | neither | +0.13 [-0.31, +0.60] | inconclusive | equivalent | neither | equivalent / equivalent |
| S | -0.19 [-0.67, +0.26] | non-inferior only | equivalent | neither | -0.33 [-0.77, +0.14] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | -0.79 [-1.46, -0.09] | non-inferior only | non-inferior only | ahead | -0.49 [-1.28, +0.30] | non-inferior only | non-inferior only | neither | inconclusive / inconclusive |
| LS | -0.04 [-0.43, +0.39] | equivalent | equivalent | neither | -0.15 [-0.53, +0.23] | non-inferior only | equivalent | neither | non-inferior only / equivalent |
| LW | +0.10 [-0.26, +0.51] | inconclusive | equivalent | neither | +0.31 [-0.14, +0.79] | inconclusive | equivalent | neither | inconclusive / equivalent |
| SW | +0.04 [-0.42, +0.50] | inconclusive | equivalent | neither | -0.47 [-0.96, +0.01] | non-inferior only | equivalent | neither | non-inferior only / non-inferior only |
| LSW | +0.05 [-0.35, +0.45] | equivalent | equivalent | neither | +0.05 [-0.30, +0.41] | equivalent | equivalent | neither | equivalent / equivalent |
| pooled All | -0.11 [-0.38, +0.15] | equivalent | equivalent | neither | -0.13 [-0.32, +0.10] | equivalent | equivalent | neither | - |
| pooled Train | -0.16 [-0.45, +0.12] | equivalent | equivalent | neither | -0.11 [-0.35, +0.18] | equivalent | equivalent | neither | - |
| pooled Held | +0.05 [-0.31, +0.39] | equivalent | equivalent | neither | -0.21 [-0.50, +0.08] | equivalent | equivalent | neither | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,L,S,W,LS,LW,SW); All 0.5 non-inferior fails (L,LW); All 1.0 equivalent fails (W); All 1.0 non-inferior holds; Train 0.5 equivalent fails (N,L,S,W,LS,LW); Train 0.5 non-inferior fails (L,LW); Train 1.0 equivalent fails (W); Train 1.0 non-inferior holds; Held 0.5 equivalent fails (SW); Held 0.5 non-inferior holds; Held 1.0 equivalent holds; Held 1.0 non-inferior holds

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% N,L,S,W, at 1% W,SW.

Training effort, xmask-dmax: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## film: xmask-film vs the policy and vs the specialist of each combination

Arm xmask-film (descriptor-input variant, --use_film True), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.75 [-0.13, +1.54] | inconclusive | inconclusive | neither | +0.81 [+0.05, +1.60] | inferior | inferior | behind | inconclusive / equivalent |
| L | +0.55 [+0.01, +1.22] | inferior | inferior | behind | +0.61 [+0.02, +1.30] | inferior | inferior | behind | equivalent / equivalent |
| S | +0.37 [-0.11, +0.85] | inconclusive | equivalent | neither | +0.23 [-0.20, +0.66] | inconclusive | equivalent | neither | equivalent / equivalent |
| W | +0.48 [-0.09, +1.08] | inconclusive | inconclusive | neither | +0.79 [-0.74, +2.08] | inconclusive | inconclusive | neither | inconclusive / inconclusive |
| LS | +0.60 [+0.00, +1.21] | inferior | inferior | behind | +0.48 [+0.13, +0.84] | inferior | equivalent | behind | non-inferior only / equivalent |
| LW | +0.79 [+0.37, +1.23] | inferior | inferior | behind | +0.99 [+0.56, +1.41] | inferior | inferior | behind | inconclusive / equivalent |
| SW | +0.65 [-0.04, +1.36] | inconclusive | inconclusive | neither | +0.13 [-0.49, +0.80] | inconclusive | equivalent | neither | non-inferior only / non-inferior only |
| LSW | +0.88 [+0.30, +1.52] | inferior | inferior | behind | +0.87 [+0.37, +1.40] | inferior | inferior | behind | equivalent / equivalent |
| pooled All | +0.63 [+0.41, +0.88] | inferior | equivalent | behind | +0.61 [+0.34, +0.89] | inferior | equivalent | behind | - |
| pooled Train | +0.59 [+0.35, +0.87] | inferior | equivalent | behind | +0.65 [+0.27, +1.01] | inferior | inferior | behind | - |
| pooled Held | +0.76 [+0.26, +1.23] | inferior | inferior | behind | +0.50 [+0.13, +0.89] | inferior | equivalent | behind | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,L,S,W,LS,LW,SW,LSW); All 0.5 non-inferior fails (N,L,S,W,LS,LW,SW,LSW); All 1.0 equivalent fails (N,L,W,LW,LSW); All 1.0 non-inferior fails (N,L,W,LW,LSW); Train 0.5 equivalent fails (N,L,S,W,LS,LW); Train 0.5 non-inferior fails (N,L,S,W,LS,LW); Train 1.0 equivalent fails (N,L,W,LW); Train 1.0 non-inferior fails (N,L,W,LW); Held 0.5 equivalent fails (SW,LSW); Held 0.5 non-inferior fails (SW,LSW); Held 1.0 equivalent fails (LSW); Held 1.0 non-inferior fails (LSW)

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% N,L,S,LS,LW,SW,LSW, at 1% N,L,LW,SW,LSW.

Training effort, xmask-film: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## dr: xmask-dr vs the policy and vs the specialist of each combination

Arm xmask-dr (randomized ranges on the released architecture (the legacy arm used the descriptor-input variant with --dr_shift)), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.42 [-0.36, +1.35] | inconclusive | inconclusive | neither | +0.48 [-0.59, +1.61] | inconclusive | inconclusive | neither | inconclusive / equivalent |
| L | +0.11 [-0.36, +0.61] | inconclusive | equivalent | neither | +0.16 [-0.24, +0.56] | inconclusive | equivalent | neither | equivalent / equivalent |
| S | +0.09 [-0.36, +0.55] | inconclusive | equivalent | neither | -0.05 [-0.56, +0.43] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | +0.30 [-0.69, +1.35] | inconclusive | inconclusive | neither | +0.61 [-1.49, +2.42] | inconclusive | inconclusive | neither | inconclusive / inconclusive |
| LS | -0.06 [-0.42, +0.27] | equivalent | equivalent | neither | -0.17 [-0.51, +0.18] | non-inferior only | equivalent | neither | non-inferior only / equivalent |
| LW | +0.25 [-0.14, +0.65] | inconclusive | equivalent | neither | +0.45 [+0.11, +0.80] | inferior | equivalent | behind | inconclusive / equivalent |
| SW | +0.60 [-0.08, +1.29] | inconclusive | inconclusive | neither | +0.09 [-0.79, +1.01] | inconclusive | inconclusive | neither | non-inferior only / non-inferior only |
| LSW | +0.07 [-0.37, +0.54] | inconclusive | equivalent | neither | +0.06 [-0.38, +0.49] | equivalent | equivalent | neither | equivalent / equivalent |
| pooled All | +0.22 [-0.05, +0.52] | inconclusive | equivalent | neither | +0.20 [-0.24, +0.68] | inconclusive | equivalent | neither | - |
| pooled Train | +0.18 [-0.09, +0.51] | inconclusive | equivalent | neither | +0.25 [-0.19, +0.73] | inconclusive | equivalent | neither | - |
| pooled Held | +0.33 [-0.12, +0.76] | inconclusive | equivalent | neither | +0.08 [-0.47, +0.63] | inconclusive | equivalent | neither | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,L,S,W,LS,LW,SW); All 0.5 non-inferior fails (N,L,W,LW,SW); All 1.0 equivalent fails (N,W,SW); All 1.0 non-inferior fails (N,W,SW); Train 0.5 equivalent fails (N,L,S,W,LS,LW); Train 0.5 non-inferior fails (N,L,W,LW); Train 1.0 equivalent fails (N,W); Train 1.0 non-inferior fails (N,W); Held 0.5 equivalent fails (SW); Held 0.5 non-inferior fails (SW); Held 1.0 equivalent fails (SW); Held 1.0 non-inferior fails (SW)

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% L,S,LW,SW, at 1% N,SW.

Training effort, xmask-dr: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## all8: xmask-all8 vs the policy and vs the specialist of each combination

Arm xmask-all8 (released recipe on all eight combinations, --family_regimes N,L,S,W,LS,LW,SW,LSW, 2000 updates (the legacy arm used the descriptor-input variant, use_film True)), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.16 [-0.25, +0.59] | inconclusive | equivalent | neither | +0.21 [-0.31, +0.67] | inconclusive | equivalent | neither | inconclusive / equivalent |
| L | -0.00 [-0.32, +0.33] | equivalent | equivalent | neither | +0.06 [-0.26, +0.36] | equivalent | equivalent | neither | equivalent / equivalent |
| S | -0.15 [-0.59, +0.28] | non-inferior only | equivalent | neither | -0.29 [-0.79, +0.17] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | -0.07 [-0.73, +0.73] | inconclusive | equivalent | neither | +0.25 [-1.34, +1.70] | inconclusive | inconclusive | neither | inconclusive / inconclusive |
| LS | +0.02 [-0.29, +0.32] | equivalent | equivalent | neither | -0.08 [-0.51, +0.33] | non-inferior only | equivalent | neither | non-inferior only / equivalent |
| LW | -0.03 [-0.40, +0.36] | equivalent | equivalent | neither | +0.18 [-0.20, +0.55] | inconclusive | equivalent | neither | inconclusive / equivalent |
| SW | +0.00 [-0.54, +0.54] | inconclusive | equivalent | neither | -0.51 [-1.16, +0.08] | non-inferior only | non-inferior only | neither | non-inferior only / non-inferior only |
| LSW | -0.01 [-0.45, +0.45] | equivalent | equivalent | neither | -0.02 [-0.41, +0.37] | equivalent | equivalent | neither | equivalent / equivalent |
| pooled All | -0.01 [-0.24, +0.26] | equivalent | equivalent | neither | -0.02 [-0.39, +0.29] | equivalent | equivalent | neither | - |
| pooled Train | -0.01 [-0.24, +0.25] | equivalent | equivalent | neither | +0.05 [-0.34, +0.40] | equivalent | equivalent | neither | - |
| pooled Held | -0.00 [-0.41, +0.42] | equivalent | equivalent | neither | -0.26 [-0.67, +0.11] | non-inferior only | equivalent | neither | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,S,W,LS,LW,SW); All 0.5 non-inferior fails (N,W,LW); All 1.0 equivalent fails (W,SW); All 1.0 non-inferior fails (W); Train 0.5 equivalent fails (N,S,W,LS,LW); Train 0.5 non-inferior fails (N,W,LW); Train 1.0 equivalent fails (W); Train 1.0 non-inferior fails (W); Held 0.5 equivalent fails (SW); Held 0.5 non-inferior holds; Held 1.0 equivalent fails (SW); Held 1.0 non-inferior holds

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% S, at 1% none.

Training effort, xmask-all8: 2000 updates x 20 episodes on N,L,S,W,LS,LW,SW,LSW = 5000 episodes per combination in expectation.

## pair: xmask-pair vs the policy and vs the specialist of each combination

Arm xmask-pair (released recipe, --pair_extra sw), folder 10x25, against the policy and against the specialist of each combination.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| N | +0.12 [-0.22, +0.47] | equivalent | equivalent | neither | +0.17 [-0.53, +0.79] | inconclusive | equivalent | neither | inconclusive / equivalent |
| L | -0.13 [-0.44, +0.20] | equivalent | equivalent | neither | -0.06 [-0.44, +0.28] | equivalent | equivalent | neither | equivalent / equivalent |
| S | +0.03 [-0.47, +0.52] | inconclusive | equivalent | neither | -0.11 [-0.50, +0.30] | non-inferior only | equivalent | neither | equivalent / equivalent |
| W | -0.24 [-0.75, +0.29] | non-inferior only | equivalent | neither | +0.07 [-0.97, +1.02] | inconclusive | inconclusive | neither | inconclusive / inconclusive |
| LS | +0.03 [-0.48, +0.57] | inconclusive | equivalent | neither | -0.09 [-0.50, +0.29] | equivalent | equivalent | neither | non-inferior only / equivalent |
| LW | +0.00 [-0.36, +0.35] | equivalent | equivalent | neither | +0.21 [-0.13, +0.57] | inconclusive | equivalent | neither | inconclusive / equivalent |
| SW | +0.19 [-0.29, +0.69] | inconclusive | equivalent | neither | -0.33 [-0.93, +0.26] | non-inferior only | equivalent | neither | non-inferior only / non-inferior only |
| LSW | +0.09 [-0.28, +0.46] | equivalent | equivalent | neither | +0.08 [-0.31, +0.45] | equivalent | equivalent | neither | equivalent / equivalent |
| pooled All | +0.01 [-0.18, +0.20] | equivalent | equivalent | neither | -0.01 [-0.22, +0.20] | equivalent | equivalent | neither | - |
| pooled Train | -0.03 [-0.24, +0.18] | equivalent | equivalent | neither | +0.03 [-0.17, +0.23] | equivalent | equivalent | neither | - |
| pooled Held | +0.14 [-0.18, +0.47] | equivalent | equivalent | neither | -0.13 [-0.52, +0.26] | non-inferior only | equivalent | neither | - |

Joint claims against the specialists: All 0.5 equivalent fails (N,S,W,LW,SW); All 0.5 non-inferior fails (N,W,LW); All 1.0 equivalent fails (W); All 1.0 non-inferior fails (W); Train 0.5 equivalent fails (N,S,W,LW); Train 0.5 non-inferior fails (N,W,LW); Train 1.0 equivalent fails (W); Train 1.0 non-inferior fails (W); Held 0.5 equivalent fails (SW); Held 0.5 non-inferior holds; Held 1.0 equivalent holds; Held 1.0 non-inferior holds

Verdict against the specialist differs from the policy's on the same seeds: at 0.5% S,LS, at 1% SW.

Training effort, xmask-pair: 2000 updates x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation.

## wthird: xmask-wthird vs the W specialist and vs the policy

Arm xmask-wthird (W specialist, 333 updates), folder 10x25, against the policy and against the W specialist.

| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | policy vs specialist, same seeds (0.5% / 1%) |
|---|---|---|---|---|---|---|---|---|---|
| W | +0.32 [-0.30, +0.96] | inconclusive | equivalent | neither | +0.62 [-0.36, +1.54] | inconclusive | inconclusive | neither | - |

Training effort, xmask-wthird: 333 updates x 20 episodes on W = 6660 episodes per combination in expectation.

## hgnn: policy vs HGNN (PACT reward)

| comb. | policy minus HGNN [95% CI] | seed-t CI | Holm p shorter (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|
| N | -4.67 [-5.11, -4.25] | [-5.09, -4.25] | 2.6e-05 / 0.0004 | 1 / 1 | ahead |
| L | -1.03 [-1.54, -0.49] | [-1.63, -0.42] | 0.014 / 0.0004 | 1 / 1 | ahead |
| S | -1.97 [-2.48, -1.42] | [-2.41, -1.54] | 0.00069 / 0.0004 | 1 / 1 | ahead |
| W | -3.69 [-4.32, -3.08] | [-4.39, -2.99] | 0.00045 / 0.0004 | 1 / 1 | ahead |
| LS | -1.05 [-1.48, -0.56] | [-1.52, -0.57] | 0.0072 / 0.0004 | 1 / 1 | ahead |
| LW | -0.74 [-1.23, -0.21] | [-1.22, -0.27] | 0.014 / 0.0037 | 1 / 1 | ahead |
| SW | -1.60 [-2.10, -1.10] | [-1.98, -1.22] | 0.00076 / 0.0004 | 1 / 1 | ahead |
| LSW | -0.85 [-1.35, -0.38] | [-1.37, -0.33] | 0.014 / 0.0007 | 1 / 1 | ahead |
| pooled All | -1.95 [-2.17, -1.73] | [-2.15, -1.75] | - | - | direction ahead |
| pooled Train | -2.19 [-2.43, -1.95] | [-2.42, -1.96] | - | - | direction ahead |
| pooled Held | -1.23 [-1.53, -0.94] | [-1.37, -1.08] | - | - | direction ahead |

PACT reward; seeds 301, 302, 303, 304, 305 (HGNN seed s against policy seed s; the block reads seeds 301-305 or 301-303, the set whose runs exist).

Training effort, hgnn-certified-joint-xmask: 2000 iterations x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation. Learning rate 2e-4.

## hgnnpub: policy vs HGNN (published reward)

| comb. | policy minus HGNN [95% CI] | seed-t CI | Holm p shorter (t / boot) | Holm p longer (t / boot) | verdict |
|---|---|---|---|---|---|
| N | -5.67 [-6.17, -5.12] | [-6.30, -5.03] | 0.0024 / 0.0004 | 1 / 1 | ahead |
| L | -3.19 [-3.60, -2.78] | [-3.58, -2.80] | 0.0024 / 0.0004 | 1 / 1 | ahead |
| S | -1.86 [-2.28, -1.42] | [-2.32, -1.39] | 0.0068 / 0.0004 | 1 / 1 | ahead |
| W | -4.59 [-5.22, -3.96] | [-5.07, -4.11] | 0.0023 / 0.0004 | 1 / 1 | ahead |
| LS | -1.42 [-1.83, -0.99] | [-2.14, -0.70] | 0.011 / 0.0004 | 1 / 1 | ahead |
| LW | -3.19 [-3.78, -2.63] | [-3.86, -2.52] | 0.0059 / 0.0004 | 1 / 1 | ahead |
| SW | -1.49 [-1.98, -1.00] | [-1.96, -1.03] | 0.0077 / 0.0004 | 1 / 1 | ahead |
| LSW | -1.47 [-1.97, -0.98] | [-2.14, -0.80] | 0.011 / 0.0004 | 1 / 1 | ahead |
| pooled All | -2.86 [-3.05, -2.68] | [-2.93, -2.79] | - | - | direction ahead |
| pooled Train | -3.32 [-3.53, -3.11] | [-3.44, -3.20] | - | - | direction ahead |
| pooled Held | -1.48 [-1.81, -1.16] | [-1.60, -1.37] | - | - | direction ahead |

Published reward; seeds 301, 302, 303 (HGNN seed s against policy seed s; the block reads seeds 301-303, the set whose runs exist).

Training effort, hgnn-published-joint-xmask: 2000 iterations x 20 episodes on N,L,S,W,LS,LW = 6667 episodes per combination in expectation. Learning rate 2e-4.

