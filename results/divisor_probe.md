# Batch composition and the greedy schedule (scripts/p104_divisor_probe.py infer)

Run 10x25+family+xmask-v1-s304 (training seed 304; arm_registry.select_seed over the exact-mask runs of the released recipe), exact mask, 10x25 test set, 100 instances. Each instance is rolled out greedily alone, twice in one batch, and with a companion whose largest processing time is the maximum of the test set (14 h), so that the divisor of the time features is that maximum. The divisor: fjsp_env_same_op_nums.py, set_initial_data: self.pt_upper_bound = np.max(self.op_pt) over the batch [E, N, M], then op_pt and op_lag divided by (pt_upper_bound - pt_lower_bound + 1e-8); constraint_family/fjsp_env_family.py, set_initial_data: raw_pt = np.array(op_pt_list); slope = 1.0 / (raw_pt.max() - raw_pt.min() + 1e-8), which scales setups, windows, horizon and release dates.

| combination | instances | own maximum below the test-set maximum | same makespan in all three | same schedule in all three | same alone and twice | same alone and with the companion, below / at the maximum | makespans differ | largest relative difference (%) | input changed (below the maximum) | largest input difference | near tie alone (gap <= 1e-06) | differ with a near tie |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 100 | 38 | 62 | 20 | 83 | 12 / 54 | 38 | 5.60 | 38 | 0.0833 | 100 | 38 |
| L | 100 | 38 | 71 | 20 | 92 | 15 / 56 | 29 | 2.53 | 38 | 0.0833 | 100 | 29 |
| S | 100 | 38 | 66 | 44 | 81 | 17 / 52 | 34 | 3.72 | 38 | 0.0833 | 100 | 34 |
| W | 100 | 38 | 63 | 29 | 91 | 8 / 55 | 37 | 3.60 | 38 | 0.0833 | 99 | 36 |
| LS | 100 | 38 | 69 | 28 | 85 | 21 / 51 | 31 | 2.86 | 38 | 0.0833 | 100 | 31 |
| LW | 100 | 38 | 68 | 32 | 91 | 14 / 55 | 32 | 8.02 | 38 | 0.0833 | 100 | 32 |
| SW | 100 | 38 | 66 | 49 | 80 | 16 / 53 | 34 | 8.04 | 38 | 0.0833 | 99 | 33 |
| LSW | 100 | 38 | 61 | 35 | 73 | 21 / 44 | 39 | 4.61 | 38 | 0.0833 | 100 | 39 |
| total | 800 | 304 | 526 | 257 | 676 | 124 / 420 | 274 | 8.04 | 304 | 0.0833 | 798 | 272 |

Signed difference of the makespan against the instance alone, 100 (x - x_alone) / x_alone (%), negative = shorter than alone: mean [95% bootstrap interval over instances; in the total an instance brings all its combinations], mean absolute difference, and the shares of the cases shorter / equal / longer than alone (%).

| combination | cases of | cases | with the companion, mean [interval] | mean absolute | shorter / equal / longer | twice in one batch, mean [interval] | mean absolute | shorter / equal / longer |
|---|---|---|---|---|---|---|---|---|
| N | below the test-set maximum | 38 | +0.07 [-0.50, +0.66] | 1.21 | 36.8 / 31.6 / 31.6 | +0.05 [-0.29, +0.44] | 0.46 | 13.2 / 76.3 / 10.5 |
| N | at the test-set maximum | 62 | -0.24 [-0.42, -0.09] | 0.24 | 12.9 / 87.1 / 0.0 | -0.24 [-0.42, -0.09] | 0.24 | 12.9 / 87.1 / 0.0 |
| L | below the test-set maximum | 38 | +0.22 [-0.03, +0.49] | 0.56 | 23.7 / 39.5 / 36.8 | +0.06 [+0.00, +0.16] | 0.06 | 0.0 / 94.7 / 5.3 |
| L | at the test-set maximum | 62 | -0.03 [-0.11, +0.05] | 0.09 | 4.8 / 90.3 / 4.8 | -0.03 [-0.11, +0.05] | 0.09 | 4.8 / 90.3 / 4.8 |
| S | below the test-set maximum | 38 | +0.33 [-0.11, +0.78] | 0.91 | 23.7 / 44.7 / 31.6 | +0.09 [-0.06, +0.25] | 0.22 | 10.5 / 76.3 / 13.2 |
| S | at the test-set maximum | 62 | -0.00 [-0.12, +0.12] | 0.17 | 8.1 / 83.9 / 8.1 | -0.00 [-0.12, +0.12] | 0.17 | 8.1 / 83.9 / 8.1 |
| W | below the test-set maximum | 38 | +0.22 [-0.25, +0.69] | 1.19 | 28.9 / 21.1 / 50.0 | +0.06 [-0.11, +0.28] | 0.13 | 2.6 / 94.7 / 2.6 |
| W | at the test-set maximum | 62 | +0.09 [-0.04, +0.24] | 0.17 | 3.2 / 88.7 / 8.1 | +0.09 [-0.04, +0.24] | 0.17 | 3.2 / 88.7 / 8.1 |
| LS | below the test-set maximum | 38 | +0.01 [-0.33, +0.34] | 0.59 | 15.8 / 55.3 / 28.9 | +0.04 [-0.03, +0.14] | 0.08 | 5.3 / 89.5 / 5.3 |
| LS | at the test-set maximum | 62 | +0.02 [-0.06, +0.10] | 0.12 | 8.1 / 82.3 / 9.7 | +0.02 [-0.06, +0.10] | 0.12 | 8.1 / 82.3 / 9.7 |
| LW | below the test-set maximum | 38 | -0.12 [-0.69, +0.33] | 0.85 | 34.2 / 36.8 / 28.9 | +0.06 [+0.00, +0.16] | 0.06 | 0.0 / 94.7 / 5.3 |
| LW | at the test-set maximum | 62 | -0.08 [-0.19, +0.01] | 0.12 | 8.1 / 88.7 / 3.2 | -0.08 [-0.19, +0.01] | 0.12 | 8.1 / 88.7 / 3.2 |
| SW | below the test-set maximum | 38 | +0.38 [-0.40, +1.24] | 1.57 | 28.9 / 42.1 / 28.9 | -0.36 [-0.80, +0.03] | 0.62 | 21.1 / 71.1 / 7.9 |
| SW | at the test-set maximum | 62 | +0.07 [-0.11, +0.27] | 0.26 | 6.5 / 85.5 / 8.1 | +0.07 [-0.11, +0.27] | 0.26 | 6.5 / 85.5 / 8.1 |
| LSW | below the test-set maximum | 38 | -0.12 [-0.49, +0.28] | 0.65 | 23.7 / 55.3 / 21.1 | -0.07 [-0.24, +0.09] | 0.23 | 13.2 / 76.3 / 10.5 |
| LSW | at the test-set maximum | 62 | -0.05 [-0.23, +0.13] | 0.29 | 14.5 / 71.0 / 14.5 | -0.05 [-0.23, +0.13] | 0.29 | 14.5 / 71.0 / 14.5 |
| total | below the test-set maximum | 304 | +0.12 [-0.06, +0.31] | 0.94 | 27.0 / 40.8 / 32.2 | -0.01 [-0.08, +0.07] | 0.23 | 8.2 / 84.2 / 7.6 |
| total | at the test-set maximum | 496 | -0.03 [-0.07, +0.02] | 0.18 | 8.3 / 84.7 / 7.1 | -0.03 [-0.07, +0.02] | 0.18 | 8.3 / 84.7 / 7.1 |

Largest processing time of an instance over the test instances (no policy):

| size | instances | minimum (h) | maximum (h) | at the maximum | below it | counts per value |
|---|---|---|---|---|---|---|
| 10x25 | 100 | 12 | 14 | 62 | 38 | 12 h: 7, 13 h: 31, 14 h: 62 |
| 50x25n30 | 30 | 14 | 14 | 30 | 0 | 14 h: 30 |
| 80x25n30 | 30 | 14 | 14 | 30 | 0 | 14 h: 30 |

