# The policy under the active mask against the policy, public family (scripts/p108_public_active_arm.py)

Headline set pub_SD2_test_10x5+mix (100 base instances, eight combinations). Arm pubx-act: the released recipe trained and evaluated under the active mask (the condition of Giffler and Thompson for active schedules); comparator pubx-v1 (the non-delay mask, on processing starts); training seeds 301, 302, 303, 304, 305, seed s against seed s. d = 100 (active - policy) / policy per training seed and base instance, seed s against seed s; negative = the policy under the active mask shorter. Hierarchical 95% intervals over seeds and base instances (B = 20000, fresh default_rng(11) per quantity), seed-level t intervals (4 df), margin verdicts at 0.5% and 1% on the hierarchical interval; pooled rows keep the combinations of a base instance together.

## Checks

- Records: 100 in one sorted order in all eight combinations; best-known table complete.
- policy under the active mask (pubx-act): 5 summaries with n/n feasible, matching means and regime-N gaps and the exact mask options; 5 regime-N arrays at or above the bound of 96 instances; every array not older than its checkpoint and newer than the records.
- policy (pubx-v1): 5 summaries with n/n feasible, matching means and regime-N gaps and the exact mask options; 5 regime-N arrays at or above the bound of 96 instances; every array not older than its checkpoint and newer than the records.
- Configurations against the comparator of the same seed differ only in: s301 mask_mode, model_suffix; s302 mask_mode, model_suffix; s303 mask_mode, model_suffix; s304 mask_mode, model_suffix; s305 mask_mode, model_suffix. The arm's logs state the mask ([train_family] OPTIONS: mask_mode=active pair_extra=none); MWKR validation norms equal.
- References: 800 of 800 counted (504 proved optimal, 0 not counted; folder or_solution/PUBFAMILY/pub_SD2_test_10x5+mix_cpsat150s_w2); no makespan of the ten runs below a counted reference bound (800 instance-combination bounds).

## 1. Paired difference per combination

| comb. | mean | hier. 95% CI | seed-t CI | direction | 0.5% | 1% | per seed | shorter / equal / longer % |
|---|---|---|---|---|---|---|---|---|
| N | -4.07 | [-5.52, -2.65] | [-5.49, -2.66] | ahead | non-inferior only | non-inferior only | -3.12, -3.21, -3.64, -4.55, -5.85 | 70.0 / 1.0 / 29.0 |
| L | -2.91 | [-3.63, -2.22] | [-3.32, -2.50] | ahead | non-inferior only | non-inferior only | -2.95, -2.35, -2.97, -3.03, -3.23 | 67.4 / 14.4 / 18.2 |
| S | -20.67 | [-22.48, -18.89] | [-22.11, -19.23] | ahead | non-inferior only | non-inferior only | -20.24, -22.56, -20.36, -20.75, -19.43 | 98.0 / 0.0 / 2.0 |
| W | -3.00 | [-4.32, -1.67] | [-3.89, -2.10] | ahead | non-inferior only | non-inferior only | -2.67, -2.68, -3.74, -3.75, -2.13 | 54.6 / 18.2 / 27.2 |
| LS | -12.97 | [-14.95, -10.90] | [-13.74, -12.20] | ahead | non-inferior only | non-inferior only | -12.01, -13.48, -13.44, -12.68, -13.22 | 91.4 / 0.2 / 8.4 |
| LW | -2.77 | [-3.59, -1.97] | [-3.18, -2.36] | ahead | non-inferior only | non-inferior only | -2.26, -2.91, -2.85, -3.13, -2.69 | 63.4 / 20.4 / 16.2 |
| SW | -16.72 | [-18.21, -15.20] | [-17.42, -16.01] | ahead | non-inferior only | non-inferior only | -16.45, -16.95, -16.45, -17.59, -16.15 | 97.4 / 0.2 / 2.4 |
| LSW | -15.05 | [-17.04, -12.95] | [-15.84, -14.25] | ahead | non-inferior only | non-inferior only | -14.76, -15.84, -14.15, -15.08, -15.40 | 92.8 / 0.6 / 6.6 |

Interval below zero (the policy under the active mask shorter): 8 of 8 (N, L, S, W, LS, LW, SW, LSW); above zero: 0 of 8 (none).

## 2. Pooled

| family | combinations | mean | hier. 95% CI | seed-t CI | direction | 0.5% | 1% | per seed | shorter / equal / longer % |
|---|---|---|---|---|---|---|---|---|---|
| All | N,L,S,W,LS,LW,SW,LSW | -9.77 | [-10.46, -9.06] | [-10.14, -9.40] | ahead | non-inferior only | non-inferior only | -9.31, -10.00, -9.70, -10.07, -9.76 | 79.4 / 6.9 / 13.8 |
| Train | N,L,S,W,LS,LW | -7.73 | [-8.35, -7.08] | [-8.11, -7.36] | ahead | non-inferior only | non-inferior only | -7.21, -7.87, -7.84, -7.98, -7.76 | 74.1 / 9.0 / 16.8 |
| Held | SW,LSW | -15.88 | [-17.16, -14.56] | [-16.47, -15.29] | ahead | non-inferior only | non-inferior only | -15.61, -16.39, -15.30, -16.34, -15.78 | 95.1 / 0.4 / 4.5 |
| Setup | S,LS,SW,LSW | -16.35 | [-17.54, -15.11] | [-17.02, -15.69] | ahead | non-inferior only | non-inferior only | -15.87, -17.21, -16.10, -16.53, -16.05 | 94.9 / 0.2 / 4.8 |
| NoSetup | N,L,W,LW | -3.19 | [-3.82, -2.55] | [-3.68, -2.69] | ahead | non-inferior only | non-inferior only | -2.75, -2.79, -3.30, -3.62, -3.48 | 63.9 / 13.5 / 22.6 |

## 3. Gap to the references

g = 100 (makespan - reference) / reference per seed and instance; references: the schedule of every record of scripts/p91_public_cpsat_refs.py that p91 marks valid (the records section 9 of p89 reads); interval and direction only.

| comb. | policy under the active mask | = / > / < % | policy | = / > / < % |
|---|---|---|---|---|
| N | +22.01 [+20.78, +23.30] behind | 0.0 / 100.0 / 0.0 | +27.81 [+26.07, +29.72] behind | 0.0 / 100.0 / 0.0 |
| L | +3.61 [+2.69, +4.64] behind | 30.6 / 69.4 / 0.0 | +6.95 [+5.65, +8.34] behind | 9.6 / 90.4 / 0.0 |
| S | +17.79 [+16.45, +19.26] behind | 0.0 / 100.0 / 0.0 | +50.01 [+47.15, +53.02] behind | 0.0 / 100.0 / 0.0 |
| W | +23.85 [+21.07, +26.73] behind | 4.0 / 96.0 / 0.0 | +28.36 [+25.46, +31.39] behind | 2.6 / 97.4 / 0.0 |
| LS | +13.07 [+11.58, +14.70] behind | 0.0 / 100.0 / 0.0 | +31.83 [+28.21, +35.48] behind | 0.0 / 100.0 / 0.0 |
| LW | +3.60 [+2.60, +4.71] behind | 34.4 / 65.6 / 0.0 | +6.80 [+5.52, +8.13] behind | 15.8 / 84.2 / 0.0 |
| SW | +16.40 [+15.11, +17.76] behind | 0.0 / 99.6 / 0.4 | +41.01 [+38.41, +43.69] behind | 0.0 / 100.0 / 0.0 |
| LSW | +14.39 [+12.76, +16.12] behind | 0.6 / 99.4 / 0.0 | +36.46 [+33.07, +39.77] behind | 0.0 / 100.0 / 0.0 |
| pooled | +14.34 [+13.53, +15.17] behind | 8.7 / 91.2 / 0.1 | +28.65 [+27.43, +29.84] behind | 3.5 / 96.5 / 0.0 |

## 4. Validation at the kept checkpoints (20 SD2 10x5 validation records, each run under its own mask)

| comb. | policy under the active mask minus the policy, % per seed | mean |
|---|---|---|
| N | -1.52, -1.41, -4.02, -5.45, -4.06 | -3.29 |
| L | -3.23, -2.71, -3.07, -2.20, -2.94 | -2.83 |
| S | -21.38, -22.18, -21.49, -21.33, -21.63 | -21.60 |
| W | -4.81, +0.13, -2.36, -3.62, -2.98 | -2.73 |
| LS | -14.08, -14.29, -12.56, -15.57, -14.39 | -14.18 |
| LW | -4.46, -3.43, -2.93, -2.08, -3.91 | -3.36 |
| score | -8.97, -8.31, -8.48, -9.13, -9.04 | -8.78 |

Score of the kept checkpoint per seed: the policy under the active mask 0.7885, 0.7897, 0.7877, 0.7911, 0.7891; the policy 0.8662, 0.8613, 0.8607, 0.8706, 0.8675. Kept update per seed: the policy under the active mask 1050, 1350, 1500, 1650, 1775; the policy 775, 1950, 1725, 1975, 1525 (of 2000); gain in the last quarter %: the policy under the active mask -0.55, -0.25, -0.21, 0.03, 0.14; the policy -0.11, 0.78, 0.12, 0.09, 0.44.
