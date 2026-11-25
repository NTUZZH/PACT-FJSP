# The policy against the specialists with ten training seeds: the benchmark from public instances under the active mask

**ADDED AFTER THE FIVE-SEED RESULT WAS KNOWN. The second five training seeds 306 to 310 of the policy and of the specialists were trained after the five-seed result of this comparison was known. The five-seed record of scripts/p89_public_family.py --family active (results/public_family_act.json, manuscript/macros_pubfam_act.tex) stays the reading of record; the ten-seed reading is reported beside it.**

The ten-seed intervals are not adjusted for the look at the five-seed result that preceded them.

Test set pub_SD2_test_10x5+mix (100 instances), active mask (mask_mode active; exact first-decision and tie options); the policy against the specialist of each combination; d = 100 (policy - specialist) / specialist, seed s against seed s. Training seeds 301 to 310: the first five seeds 301 to 305 and the second five seeds 306 to 310. Estimator, margins and verdicts of p89 (crossed hierarchical bootstrap over training seeds and base instances, B = 20000, fresh default_rng(11) per quantity (p89.component and p89.pooled), five and ten seeds; verdicts at +-0.5% and +-1% from the 95% interval). The five-seed rows recomputed from the arrays equal the stored record (`results/public_family_act.json`, `manuscript/macros_pubfam_act.tex`): 274 numbers and words at the printed precision and 118 macros character for character. Validation norms of each recipe equal over the ten seeds.

| combination | ten seeds: mean | 95% CI | 0.5% | 1% | direction | first five seeds (record): mean | 95% CI | 0.5% | 1% | direction | mean of the second five seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | +0.92 | [-0.07, +1.95] | inconclusive | inconclusive | neither | +1.12 | [-0.27, +2.52] | inconclusive | inconclusive | neither | +0.73 |
| L | +0.51 | [+0.01, +1.03] | inferior | inferior | behind | +0.58 | [-0.05, +1.21] | inconclusive | inconclusive | neither | +0.44 |
| S | +0.34 | [-0.43, +1.13] | inconclusive | inconclusive | neither | +0.51 | [-0.50, +1.64] | inconclusive | inconclusive | neither | +0.17 |
| W | +0.32 | [-0.69, +1.36] | inconclusive | inconclusive | neither | +0.30 | [-1.06, +1.67] | inconclusive | inconclusive | neither | +0.33 |
| LS | +0.23 | [-0.60, +1.09] | inconclusive | inconclusive | neither | -0.20 | [-1.04, +0.68] | inconclusive | non-inferior only | neither | +0.65 |
| LW | -0.05 | [-0.70, +0.53] | inconclusive | equivalent | neither | -0.09 | [-1.04, +0.75] | inconclusive | non-inferior only | neither | -0.00 |
| SW | +0.01 | [-1.04, +1.01] | inconclusive | inconclusive | neither | -0.83 | [-2.23, +0.56] | inconclusive | non-inferior only | neither | +0.85 |
| LSW | -0.36 | [-1.16, +0.37] | non-inferior only | non-inferior only | neither | -0.18 | [-1.11, +0.74] | inconclusive | non-inferior only | neither | -0.54 |

Per-seed means (policy minus specialist over specialist, %):

| combination | s301 | s302 | s303 | s304 | s305 | s306 | s307 | s308 | s309 | s310 |
|---|---|---|---|---|---|---|---|---|---|---|
| N | +2.77 | +1.83 | -0.54 | +0.12 | +1.45 | +2.58 | +1.13 | +0.28 | +0.58 | -0.95 |
| L | +1.01 | +0.66 | +0.92 | +0.43 | -0.13 | +0.49 | +0.01 | +0.62 | +0.74 | +0.35 |
| S | +0.34 | -0.03 | +0.89 | -0.20 | +1.54 | +1.07 | -0.91 | +0.19 | -0.19 | +0.68 |
| W | -0.03 | +0.56 | -0.21 | -0.11 | +1.29 | +0.74 | +1.43 | -0.48 | +0.78 | -0.80 |
| LS | -0.29 | +0.46 | -0.78 | +0.17 | -0.55 | +1.26 | +1.44 | +1.15 | -0.85 | +0.25 |
| LW | +0.82 | -0.98 | -0.49 | +0.07 | +0.12 | -0.03 | +0.16 | -0.71 | +0.32 | +0.27 |
| SW | +0.67 | -1.93 | +0.16 | -0.35 | -2.69 | +1.04 | +1.05 | +1.39 | +1.13 | -0.33 |
| LSW | +0.42 | -1.18 | +0.14 | -0.28 | +0.00 | -0.76 | +0.23 | +0.15 | -1.75 | -0.57 |

Claims over several combinations (intersection-union test):

| combinations | margin | equivalent on all (ten / five) | non-inferior on all (ten / five) | failing, equivalent (ten / five) |
|---|---|---|---|---|
| All | 0.5 | False / False | False / False | N,L,S,W,LS,LW,SW,LSW / N,L,S,W,LS,LW,SW,LSW |
| All | 1.0 | False / False | False / False | N,L,S,W,LS,SW,LSW / N,L,S,W,LS,LW,SW,LSW |
| Train | 0.5 | False / False | False / False | N,L,S,W,LS,LW / N,L,S,W,LS,LW |
| Train | 1.0 | False / False | False / False | N,L,S,W,LS / N,L,S,W,LS,LW |
| Held | 0.5 | False / False | False / False | SW,LSW / SW,LSW |
| Held | 1.0 | False / False | False / True | SW,LSW / SW,LSW |

Pooled rows (combinations of a base instance together):

| combinations | ten seeds: mean | 95% CI | 0.5% | 1% | first five seeds (record): mean | 95% CI | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| All | +0.24 | [-0.09, +0.58] | inconclusive | equivalent | +0.15 | [-0.25, +0.58] | inconclusive | equivalent |
| Train | +0.38 | [+0.01, +0.76] | inferior | equivalent | +0.37 | [-0.08, +0.84] | inconclusive | equivalent |
| Held | -0.17 | [-0.83, +0.46] | non-inferior only | equivalent | -0.50 | [-1.42, +0.42] | non-inferior only | non-inferior only |
