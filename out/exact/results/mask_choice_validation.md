# The choice of the mask repeated on the validation instances (scripts/p141_mask_choice_validation.py)

Best of four dispatching rules (per instance the shortest of the FIFO, MOR, SPT and MWKR schedules) under each mask, from the rollouts of scripts/p107_mask_rule_variants.py. Difference: 100 (best_X - best_nondelay) / best_nondelay per case; negative means mask X gives the shorter schedules. Interval: P107.boot: stats_family.instance_boot, 20000 resamples, fresh default_rng(7) per quantity; pooled, a drawn instance brings its eight combinations. Direction: shorter when the interval lies below zero, longer when it lies above zero, neither otherwise (the first-named mask against nondelay). Choice: the mask with the shorter best-rule schedules from the pooled direction of active against nondelay (shorter: active, longer: nondelay, neither: none); same when it is the mask the benchmark uses (mask_mode of the policy run), different otherwise. Test-instance values read from results/mask_rule_variants.json.

## factory: factory benchmark, training size (10 modules, 25 stations)

Validation instances: 20 (vali_seed100 to vali_seed119); test instances: 100. The benchmark uses the nondelay mask (mask_mode of xmask-v1-s304). Validation: same (chosen: nondelay); test: same (chosen: nondelay).

active against nondelay, best rule, difference (%) and direction:

| row | validation | direction | cases shorter / longer | test | direction |
|---|---|---|---|---|---|
| N | +7.08 [+5.58, +8.83] | longer | 0 / 20 | +8.27 [+7.56, +8.99] | longer |
| L | +2.28 [+1.41, +3.18] | longer | 2 / 16 | +2.28 [+1.85, +2.72] | longer |
| S | +7.30 [+5.89, +8.76] | longer | 0 / 20 | +9.36 [+8.56, +10.18] | longer |
| W | +9.44 [+7.64, +11.31] | longer | 0 / 20 | +8.00 [+7.24, +8.76] | longer |
| LS | +4.99 [+3.50, +6.57] | longer | 1 / 19 | +5.60 [+4.96, +6.26] | longer |
| LW | +2.90 [+1.89, +3.86] | longer | 2 / 18 | +2.80 [+2.33, +3.28] | longer |
| SW | +6.13 [+4.40, +7.74] | longer | 2 / 18 | +7.59 [+6.80, +8.39] | longer |
| LSW | +5.51 [+4.32, +6.86] | longer | 0 / 20 | +5.71 [+5.13, +6.30] | longer |
| pooled over the eight | +5.70 [+5.12, +6.27] | longer | | +6.20 [+5.92, +6.48] | longer |

Combinations per direction: validation shorter 0; longer 8; neither 0; test shorter 0; longer 8; neither 0.

block against nondelay, best rule, difference (%) and direction:

| row | validation | direction | cases shorter / longer | test | direction |
|---|---|---|---|---|---|
| N | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| L | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| S | +4.01 [+2.00, +6.08] | longer | 5 / 14 | +4.49 [+3.74, +5.28] | longer |
| W | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| LS | +3.57 [+2.27, +5.22] | longer | 0 / 20 | +3.80 [+3.14, +4.52] | longer |
| LW | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| SW | +2.27 [+0.72, +3.75] | longer | 4 / 16 | +4.54 [+3.81, +5.30] | longer |
| LSW | +3.30 [+1.47, +5.35] | longer | 4 / 16 | +3.73 [+2.97, +4.53] | longer |
| pooled over the eight | +1.64 [+1.12, +2.22] | longer | | +2.07 [+1.81, +2.34] | longer |

Combinations per direction: validation shorter 0; longer 4; neither 4; test shorter 0; longer 4; neither 4.

Mean best-rule makespan (h) on the validation instances, nondelay / block / active:

| row | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| best rule | 132.6 / 132.6 / 141.9 | 214.8 / 214.8 / 219.7 | 192.6 / 200.3 / 206.5 | 150.2 / 150.2 / 164.4 | 271.8 / 281.3 / 285.1 | 225.6 / 225.6 / 232.1 | 217.3 / 222.2 / 230.5 | 291.0 / 300.3 / 306.9 |

Check: mean MWKR makespan on the rebuilt validation instances (default mask, legacy options) against the line "[vali] MWKR norms" of the training log:

| combination | rebuilt | log | run |
|---|---|---|---|
| N | 156.6 | 156.6 | xmask-v1-s304 |
| L | 226.9 | 226.9 | xmask-v1-s304 |
| S | 235.3 | 235.3 | xmask-v1-s304 |
| W | 177.7 | 177.7 | xmask-v1-s304 |
| LS | 304.4 | 304.4 | xmask-v1-s304 |
| LW | 243.4 | 243.4 | xmask-v1-s304 |
| SW | 260.9 | 260.9 | xmask-solo-SW-s304 |
| LSW | 331.9 | 331.9 | xmask-solo-LSW-s304 |

## public: public benchmark, SD2 10x5

Validation instances: 20 (10x5+mix_001 to 10x5+mix_020); test instances: 100. The benchmark uses the active mask (mask_mode of pubx-act-s304). Validation: same (chosen: active); test: same (chosen: active).

active against nondelay, best rule, difference (%) and direction:

| row | validation | direction | cases shorter / longer | test | direction |
|---|---|---|---|---|---|
| N | -5.51 [-8.86, -2.20] | shorter | 14 / 6 | -4.70 [-6.22, -3.09] | shorter |
| L | -2.39 [-4.47, -0.48] | shorter | 14 / 2 | -1.10 [-2.04, -0.13] | shorter |
| S | -10.97 [-14.19, -7.75] | shorter | 18 / 2 | -11.01 [-12.66, -9.40] | shorter |
| W | -1.64 [-4.94, +2.66] | neither | 13 / 6 | -4.43 [-5.97, -2.88] | shorter |
| LS | -8.45 [-12.98, -4.31] | shorter | 17 / 3 | -6.33 [-8.05, -4.67] | shorter |
| LW | -1.80 [-3.72, +0.11] | neither | 11 / 6 | -1.06 [-2.10, -0.02] | shorter |
| SW | -13.06 [-16.08, -10.43] | shorter | 20 / 0 | -8.98 [-10.43, -7.54] | shorter |
| LSW | -9.85 [-14.41, -5.69] | shorter | 18 / 2 | -7.84 [-9.72, -6.01] | shorter |
| pooled over the eight | -6.71 [-8.23, -5.37] | shorter | | -5.68 [-6.33, -5.02] | shorter |

Combinations per direction: validation shorter 6; longer 0; neither 2; test shorter 8; longer 0; neither 0.

block against nondelay, best rule, difference (%) and direction:

| row | validation | direction | cases shorter / longer | test | direction |
|---|---|---|---|---|---|
| N | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| L | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| S | -7.60 [-12.06, -3.24] | shorter | 15 / 5 | -5.78 [-7.66, -3.95] | shorter |
| W | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| LS | -6.53 [-10.71, -2.70] | shorter | 15 / 5 | -5.23 [-7.00, -3.51] | shorter |
| LW | +0.00 [+0.00, +0.00] | neither | 0 / 0 | +0.00 [+0.00, +0.00] | neither |
| SW | -5.93 [-8.41, -3.62] | shorter | 18 / 2 | -5.28 [-6.66, -3.91] | shorter |
| LSW | -7.80 [-12.55, -3.35] | shorter | 15 / 5 | -6.70 [-8.68, -4.73] | shorter |
| pooled over the eight | -3.48 [-4.86, -2.24] | shorter | | -2.87 [-3.42, -2.33] | shorter |

Combinations per direction: validation shorter 4; longer 0; neither 4; test shorter 4; longer 0; neither 4.

Mean best-rule makespan (h) on the validation instances, nondelay / block / active:

| row | N | L | S | W | LS | LW | SW | LSW |
|---|---|---|---|---|---|---|---|---|
| best rule | 489.4 / 489.4 / 460.8 | 915.6 / 915.6 / 897.9 | 869.6 / 798.6 / 771.5 | 536.5 / 536.5 / 525.1 | 1191.8 / 1111.7 / 1089.2 | 950.0 / 950.0 / 931.0 | 1025.5 / 962.0 / 888.3 | 1278.7 / 1173.5 / 1148.8 |

Check: mean MWKR makespan on the rebuilt validation instances (default mask, legacy options) against the line "[vali] MWKR norms" of the training log:

| combination | rebuilt | log | run |
|---|---|---|---|
| N | 538.2 | 538.2 | pubx-act-s304 |
| L | 1019.8 | 1019.8 | pubx-act-s304 |
| S | 903.2 | 903.2 | pubx-act-s304 |
| W | 680.7 | 680.7 | pubx-act-s304 |
| LS | 1213.8 | 1213.8 | pubx-act-s304 |
| LW | 1073.7 | 1073.7 | pubx-act-s304 |
| SW | 1077.8 | 1077.8 | pubx-soloact-SW-s304 |
| LSW | 1348.0 | 1348.0 | pubx-soloact-LSW-s304 |

