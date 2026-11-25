# Lower-bound reference for the makespans (environment exact, training seeds 301,302,303,304,305)

Reference bound LB per cell: the largest of the instance bound (job chains), the machine-load bound and the stored solver bound. Gap = 100 (makespan - reference bound) / reference bound per instance, %; mean over the instances with the 95% interval of stats_family.instance_boot (B = 20000, default_rng(7)). Policy run: seed 304 (lowest validation score); CP-SAT configuration lns; searches at 600 s.

Self-check: 150708 pairs (instance bound, feasible makespan) and 150708 pairs (machine-load bound, feasible makespan) over 1792 cells, no makespan below a bound; smallest slack instance bound (job chains) 0 h at 10x25/L/instance_027 (10x25_memetic600s:makespan); machine-load bound 1 h at 10x25/N/instance_006 (10x25_memetic600s:makespan).

Stored solver bounds: 13264 records with a bound read; 13112 verified (schedule 3566, warm start 9546), 9546 used, 0 excluded (above a validated feasible makespan), 0 not verifiable, 152 outside the analyzed instances, 0 naming another cell, 0 of other data.

Records written before 2026-07-08 (file modification time) give no bound, since the window horizon of the constraint model was enlarged on 2026-07-07: 3566 verified records left out (their validated schedules remain feasible makespans); 0 records left out because the descriptor carries release dates or a per-machine setup multiplier.


Timed comparison as launched: "50x25n30 80x25n30 = memetic_rule@600 memetic_policy@600 cpsat_rule@600 cpsat_policy@600 policy_sample@120"; "10x25 = memetic_rule@600 policy_sample@60"; "10x25 = cpsat_rule@600:0-9" (read past: memetic_policy@600, cpsat_policy@600, policy_sample@120, policy_sample@60).

- memetic algorithm started from the best dispatching rule, 600 s, 50x25n30, instances 3-29: 216 cells, 216 valid
- memetic algorithm started from the best dispatching rule, 600 s, 80x25n30, instances 3-29: 216 cells, 216 valid
- CP-SAT started from the best dispatching rule, 600 s, 50x25n30, instances 3-29: 216 cells, 216 valid
- CP-SAT started from the best dispatching rule, 600 s, 80x25n30, instances 3-29: 216 cells, 216 valid
- memetic algorithm started from the best dispatching rule, 600 s, 10x25, instances 0-29: 240 cells, 240 valid
- CP-SAT started from the best dispatching rule, 600 s, 10x25, instances 0-9: 80 cells, 80 valid

## 10x25 (100 instances per combination)

Station types: 22000 of 22000 operations have their eligible machines in one station type, 22000 the whole type; the eligible sets partition the operations on 100 of 100 instances.

| comb. | policy, run with the lowest validation score | policy, mean over the training seeds | specialist of the combination, mean over the training seeds | best dispatching rule | memetic algorithm, 600 s | CP-SAT, 600 s |
|---|---|---|---|---|---|---|
| N | 17.3 [16.0, 18.6] (n 100) | 17.1 [15.9, 18.3] (n 100) | 16.8 [15.9, 17.8] (n 100) | 19.3 [18.3, 20.4] (n 100) | 9.2 [7.7, 10.6] (n 30) | 5.6 [2.9, 8.3] (n 10) |
| L | 4.8 [4.3, 5.5] (n 100) | 4.6 [4.0, 5.3] (n 100) | 4.5 [3.8, 5.2] (n 100) | 6.2 [5.2, 7.2] (n 100) | 0.2 [0.1, 0.5] (n 30) | 0.3 [0.0, 0.7] (n 10) |
| S | 26.9 [25.3, 28.5] (n 100) | 27.3 [25.7, 28.9] (n 100) | 27.8 [26.3, 29.3] (n 100) | 28.0 [26.5, 29.5] (n 100) | 18.1 [16.0, 20.2] (n 30) | 18.7 [14.1, 23.1] (n 10) |
| W | 28.9 [27.5, 30.4] (n 100) | 29.3 [28.0, 30.7] (n 100) | 28.9 [27.7, 30.1] (n 100) | 32.0 [30.7, 33.4] (n 100) | 17.1 [15.1, 19.1] (n 30) | 14.4 [11.5, 17.4] (n 10) |
| LS | 14.0 [13.1, 14.8] (n 100) | 13.9 [13.1, 14.8] (n 100) | 14.0 [13.2, 14.9] (n 100) | 15.2 [14.3, 16.2] (n 100) | 6.7 [5.4, 8.2] (n 30) | 6.4 [3.9, 8.9] (n 10) |
| LW | 7.8 [7.0, 8.6] (n 100) | 7.3 [6.6, 8.1] (n 100) | 6.8 [6.2, 7.6] (n 100) | 9.7 [8.8, 10.7] (n 100) | 1.3 [0.8, 2.1] (n 30) | 1.2 [0.3, 2.4] (n 10) |
| SW | 37.6 [36.0, 39.2] (n 100) | 37.8 [36.2, 39.4] (n 100) | 38.5 [37.0, 40.1] (n 100) | 39.2 [37.7, 40.7] (n 100) | 24.8 [22.4, 27.3] (n 30) | 29.2 [24.4, 33.9] (n 10) |
| LSW | 20.2 [19.2, 21.3] (n 100) | 20.1 [19.2, 21.1] (n 100) | 20.4 [19.3, 21.4] (n 100) | 21.7 [20.6, 22.8] (n 100) | 10.8 [9.1, 12.5] (n 30) | 12.9 [9.4, 16.5] (n 10) |
| pooled | 19.7 [19.1, 20.3] (n 100) | 19.7 [19.1, 20.3] (n 100) | 19.7 [19.2, 20.3] (n 100) | 21.4 [20.8, 22.0] (n 100) | 11.0 [10.2, 11.9] (n 30) | 11.1 [9.6, 12.6] (n 10) |

Policy on the instances of each search (pooled; per combination in the JSON):

- instances of the memetic algorithm, 600 s (n 30): policy 19.4 [18.3, 20.4], memetic algorithm 11.0 [10.2, 11.9]
- instances of the CP-SAT, 600 s (n 10): policy 18.3 [16.2, 20.1], CP-SAT 11.1 [9.6, 12.6]

| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % | 100 LB / best known: chain / load / stored / reference |
|---|---|---|---|
| N | 0.0 / 99.0 / 4.0 | 0.0 / 96.0 / 1.0 | 74.8 / 91.1 / 84.1 / 91.1 |
| L | 64.0 / 0.0 / 100.0 | 0.0 / 0.0 / 36.0 | 98.8 / 80.2 / 99.4 / 99.4 |
| S | 0.0 / 54.0 / 50.0 | 0.0 / 50.0 / 46.0 | 70.2 / 81.7 / 79.7 / 83.0 |
| W | 0.0 / 54.0 / 56.0 | 0.0 / 44.0 / 46.0 | 66.9 / 81.6 / 81.5 / 83.8 |
| LS | 20.0 / 10.0 / 91.0 | 0.0 / 9.0 / 70.0 | 89.2 / 78.2 / 91.1 / 91.3 |
| LW | 8.0 / 0.0 / 100.0 | 0.0 / 0.0 / 92.0 | 96.1 / 78.1 / 98.5 / 98.5 |
| SW | 0.0 / 33.0 / 68.0 | 0.0 / 32.0 / 67.0 | 62.7 / 73.1 / 75.5 / 77.2 |
| LSW | 4.0 / 8.0 / 92.0 | 0.0 / 8.0 / 88.0 | 84.4 / 74.1 / 87.1 / 87.2 |
| pooled | 12.0 / 32.2 / 70.1 | 0.0 / 29.9 / 55.8 | 80.4 / 79.8 / 87.1 / 88.9 |

Stored solver bounds: 5840 used on 800 cells, 0 excluded, 3566 written before 2026-07-08 not used.

| folder of the records used | records | budget_s (records) | warm start | verified by | written, oldest to newest | cells where it gives the reference |
|---|---|---|---|---|---|---|
| 10x25_fairwarm10s | 800 | 10 s (800) | 800 of 800 | warm start 800 | 2026-07-14 04:39 to 2026-10-16 21:35 | 168 |
| 10x25_fairwarm1s | 800 | 1 s (800) | 800 of 800 | warm start 800 | 2026-07-14 03:31 to 2026-10-16 21:35 | 0 |
| 10x25_fairwarm2s | 800 | 2 s (800) | 800 of 800 | warm start 800 | 2026-07-14 03:51 to 2026-10-16 21:35 | 0 |
| 10x25_fairwarm300s | 240 | 300 s (240) | 240 of 240 | warm start 240 | 2026-07-12 01:08 to 2026-10-16 14:46 | 129 |
| 10x25_fairwarm30s | 800 | 30 s (800) | 800 of 800 | warm start 800 | 2026-07-11 21:08 to 2026-10-16 19:27 | 140 |
| 10x25_fairwarm5s | 800 | 5 s (800) | 800 of 800 | warm start 800 | 2026-07-11 20:09 to 2026-10-16 10:17 | 0 |
| 10x25_fairwarmR30s | 800 | 30 s (800) | 800 of 800 | warm start 800 | 2026-07-17 12:24 to 2026-10-16 16:54 | 124 |
| 10x25_fairwarmR5s | 800 | 5 s (800) | 800 of 800 | warm start 800 | 2026-07-17 11:52 to 2026-10-16 21:35 | 0 |

Cells proved optimal (best known makespan equal to the reference bound): 125 of 800 (L 83, LW 42); the reference is attained there by: instance bound (job chains) 60, machine-load bound 0, stored solver bound 125 (ties counted for each).

## 20x25 (50 instances per combination)

Station types: 22000 of 22000 operations have their eligible machines in one station type, 22000 the whole type; the eligible sets partition the operations on 50 of 50 instances.

| comb. | policy, run with the lowest validation score | policy, mean over the training seeds | specialist of the combination, mean over the training seeds | best dispatching rule | memetic algorithm, 600 s | CP-SAT, 600 s |
|---|---|---|---|---|---|---|
| N | 18.2 [17.3, 19.1] (n 50) | 17.7 [16.5, 18.8] (n 50) | 17.2 [16.1, 18.2] (n 50) | 20.5 [19.0, 21.9] (n 50) | - | - |
| L | 21.5 [20.3, 22.6] (n 50) | 21.1 [20.0, 22.2] (n 50) | 23.8 [22.6, 25.0] (n 50) | 24.4 [23.3, 25.5] (n 50) | - | - |
| S | 27.4 [25.4, 29.5] (n 50) | 28.3 [26.2, 30.3] (n 50) | 29.1 [26.9, 31.3] (n 50) | 31.4 [29.0, 33.8] (n 50) | - | - |
| W | 34.8 [33.2, 36.3] (n 50) | 34.5 [32.8, 36.1] (n 50) | 32.9 [31.2, 34.5] (n 50) | 38.7 [36.6, 40.7] (n 50) | - | - |
| LS | 38.2 [35.7, 40.7] (n 50) | 37.7 [35.3, 40.0] (n 50) | 37.9 [35.5, 40.3] (n 50) | 39.9 [37.5, 42.2] (n 50) | - | - |
| LW | 28.6 [27.4, 29.8] (n 50) | 28.0 [26.8, 29.2] (n 50) | 32.4 [31.0, 33.8] (n 50) | 32.3 [30.9, 33.7] (n 50) | - | - |
| SW | 41.7 [39.4, 43.9] (n 50) | 42.8 [40.4, 45.1] (n 50) | 43.4 [40.9, 45.8] (n 50) | 45.6 [42.8, 48.4] (n 50) | - | - |
| LSW | 46.1 [43.6, 48.6] (n 50) | 45.6 [43.2, 47.9] (n 50) | 46.0 [43.5, 48.4] (n 50) | 49.2 [46.7, 51.8] (n 50) | - | - |
| pooled | 32.1 [30.9, 33.2] (n 50) | 31.9 [30.6, 33.2] (n 50) | 32.9 [31.5, 34.1] (n 50) | 35.3 [33.8, 36.6] (n 50) | - | - |

| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % | 100 LB / best known: chain / load / stored / reference |
|---|---|---|---|
| N | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 51.9 / 88.2 / 69.6 / 88.2 |
| L | 42.0 / 4.0 / 96.0 | 0.0 / 4.0 / 54.0 | 83.6 / 76.0 / 84.4 / 84.6 |
| S | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 50.3 / 80.5 / 65.7 / 80.5 |
| W | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 45.8 / 77.9 / 65.8 / 77.9 |
| LS | 38.0 / 40.0 / 48.0 | 14.0 / 36.0 / 24.0 | 71.5 / 72.9 / 72.6 / 74.9 |
| LW | 6.0 / 4.0 / 96.0 | 0.0 / 4.0 / 90.0 | 78.0 / 70.9 / 80.0 / 80.2 |
| SW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 45.3 / 72.4 / 60.0 / 72.4 |
| LSW | 14.0 / 32.0 / 58.0 | 10.0 / 32.0 / 54.0 | 67.0 / 68.3 / 68.8 / 70.7 |
| pooled | 12.5 / 60.0 / 37.2 | 3.0 / 59.5 / 27.8 | 61.7 / 75.9 / 70.9 / 78.7 |

Stored solver bounds: 2400 used on 400 cells, 0 excluded, 0 written before 2026-07-08 not used.

| folder of the records used | records | budget_s (records) | warm start | verified by | written, oldest to newest | cells where it gives the reference |
|---|---|---|---|---|---|---|
| 20x25_fairwarm10s | 400 | 10 s (400) | 400 of 400 | warm start 400 | 2026-07-17 05:12 to 2026-10-16 11:05 | 47 |
| 20x25_fairwarm1s | 400 | 1 s (400) | 400 of 400 | warm start 400 | 2026-07-17 04:30 to 2026-10-16 08:25 | 0 |
| 20x25_fairwarm2s | 400 | 2 s (400) | 400 of 400 | warm start 400 | 2026-07-17 04:37 to 2026-10-16 08:42 | 0 |
| 20x25_fairwarm30s | 400 | 30 s (400) | 400 of 400 | warm start 400 | 2026-07-17 06:05 to 2026-10-16 14:21 | 24 |
| 20x25_fairwarm5s | 400 | 5 s (400) | 400 of 400 | warm start 400 | 2026-07-17 04:49 to 2026-10-16 09:19 | 0 |
| 20x25_fairwarm60s | 400 | 60 s (400) | 400 of 400 | warm start 400 | 2026-07-17 08:14 to 2026-10-16 18:06 | 78 |

Cells proved optimal (best known makespan equal to the reference bound): 0 of 400.

## 30x25 (20 instances per combination)

Station types: 13200 of 13200 operations have their eligible machines in one station type, 13200 the whole type; the eligible sets partition the operations on 20 of 20 instances.

| comb. | policy, run with the lowest validation score | policy, mean over the training seeds | specialist of the combination, mean over the training seeds | best dispatching rule | memetic algorithm, 600 s | CP-SAT, 600 s |
|---|---|---|---|---|---|---|
| N | 14.9 [12.6, 17.1] (n 20) | 14.3 [11.8, 16.6] (n 20) | 14.7 [12.4, 16.8] (n 20) | 17.8 [14.9, 20.7] (n 20) | - | - |
| L | 30.1 [26.4, 33.4] (n 20) | 29.1 [25.9, 32.1] (n 20) | 32.8 [28.8, 36.4] (n 20) | 34.3 [31.2, 37.0] (n 20) | - | - |
| S | 23.2 [20.0, 26.2] (n 20) | 23.8 [20.6, 26.7] (n 20) | 24.6 [21.7, 27.2] (n 20) | 27.9 [24.1, 31.5] (n 20) | - | - |
| W | 28.8 [25.7, 31.7] (n 20) | 27.8 [25.0, 30.3] (n 20) | 26.6 [24.0, 29.1] (n 20) | 31.8 [29.0, 34.5] (n 20) | - | - |
| LS | 34.4 [31.1, 37.4] (n 20) | 34.6 [31.6, 37.5] (n 20) | 35.0 [31.9, 37.9] (n 20) | 38.8 [35.2, 42.4] (n 20) | - | - |
| LW | 39.5 [36.3, 42.4] (n 20) | 39.0 [36.2, 41.5] (n 20) | 47.3 [43.0, 51.2] (n 20) | 44.0 [40.9, 46.9] (n 20) | - | - |
| SW | 35.9 [32.3, 39.3] (n 20) | 34.8 [31.1, 38.2] (n 20) | 36.1 [32.4, 39.6] (n 20) | 39.3 [36.2, 42.4] (n 20) | - | - |
| LSW | 41.9 [38.0, 45.6] (n 20) | 42.0 [38.5, 45.2] (n 20) | 41.9 [38.2, 45.4] (n 20) | 46.1 [41.9, 50.2] (n 20) | - | - |
| pooled | 31.1 [28.3, 33.6] (n 20) | 30.7 [27.9, 33.2] (n 20) | 32.4 [29.4, 35.1] (n 20) | 35.0 [32.0, 37.8] (n 20) | - | - |

| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % | 100 LB / best known: chain / load / stored / reference |
|---|---|---|---|
| N | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 40.4 / 90.9 / 63.8 / 90.9 |
| L | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 68.3 / 79.9 / 69.8 / 79.9 |
| S | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 39.7 / 83.4 / 58.6 / 83.4 |
| W | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 36.2 / 81.5 / 61.6 / 81.5 |
| LS | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 58.6 / 76.8 / 61.2 / 76.8 |
| LW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 63.4 / 74.2 / 66.9 / 74.2 |
| SW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 36.5 / 76.6 / 54.3 / 76.6 |
| LSW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 55.5 / 72.7 / 58.2 / 72.7 |
| pooled | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 49.8 / 79.5 / 61.8 / 79.5 |

Stored solver bounds: 960 used on 160 cells, 0 excluded, 0 written before 2026-07-08 not used.

| folder of the records used | records | budget_s (records) | warm start | verified by | written, oldest to newest | cells where it gives the reference |
|---|---|---|---|---|---|---|
| 30x25_fairwarm10s | 160 | 10 s (160) | 160 of 160 | warm start 160 | 2026-07-17 02:13 to 2026-10-16 10:31 | 0 |
| 30x25_fairwarm1s | 160 | 1 s (160) | 160 of 160 | warm start 160 | 2026-07-17 01:52 to 2026-10-16 08:22 | 0 |
| 30x25_fairwarm2s | 160 | 2 s (160) | 160 of 160 | warm start 160 | 2026-07-17 01:56 to 2026-10-16 08:35 | 0 |
| 30x25_fairwarm30s | 160 | 30 s (160) | 160 of 160 | warm start 160 | 2026-07-17 02:35 to 2026-10-16 12:40 | 0 |
| 30x25_fairwarm5s | 160 | 5 s (160) | 160 of 160 | warm start 160 | 2026-07-17 02:02 to 2026-10-16 09:02 | 0 |
| 30x25_fairwarm60s | 160 | 60 s (160) | 160 of 160 | warm start 160 | 2026-07-17 03:28 to 2026-10-16 19:27 | 0 |

Cells proved optimal (best known makespan equal to the reference bound): 0 of 160.

## 50x25n30 (27 instances per combination)

Station types: 29700 of 29700 operations have their eligible machines in one station type, 29700 the whole type; the eligible sets partition the operations on 27 of 27 instances.

| comb. | policy, run with the lowest validation score | policy, mean over the training seeds | specialist of the combination, mean over the training seeds | best dispatching rule | memetic algorithm, 600 s | CP-SAT, 600 s |
|---|---|---|---|---|---|---|
| N | 19.1 [16.9, 21.4] (n 27) | 16.0 [15.4, 16.6] (n 27) | - | 19.8 [19.0, 20.6] (n 27) | 7.9 [7.5, 8.3] (n 27) | 7.6 [7.2, 8.0] (n 27) |
| L | 29.3 [28.5, 30.0] (n 27) | 28.1 [27.5, 28.7] (n 27) | - | 34.7 [33.9, 35.3] (n 27) | 19.3 [18.7, 20.0] (n 27) | 18.1 [17.6, 18.6] (n 27) |
| S | 20.9 [19.1, 22.8] (n 27) | 22.2 [20.5, 24.1] (n 27) | - | 26.5 [24.3, 28.9] (n 27) | 16.0 [14.4, 17.9] (n 27) | 21.5 [19.6, 23.7] (n 27) |
| W | 26.5 [24.5, 28.8] (n 27) | 24.9 [24.1, 25.8] (n 27) | - | 29.1 [28.1, 30.1] (n 27) | 22.3 [21.1, 23.5] (n 27) | 18.4 [17.7, 19.0] (n 27) |
| LS | 32.0 [29.8, 34.4] (n 27) | 31.6 [29.5, 33.8] (n 27) | - | 37.4 [34.9, 39.9] (n 27) | 25.5 [23.6, 27.4] (n 27) | 33.2 [30.8, 35.6] (n 27) |
| LW | 35.8 [34.9, 36.7] (n 27) | 34.8 [34.0, 35.7] (n 27) | - | 41.1 [40.1, 42.2] (n 27) | 31.8 [30.4, 33.1] (n 27) | 27.5 [26.8, 28.2] (n 27) |
| SW | 28.9 [26.9, 31.0] (n 27) | 29.6 [27.7, 31.7] (n 27) | - | 33.1 [30.5, 36.0] (n 27) | 25.6 [23.5, 27.7] (n 27) | 28.7 [26.3, 31.2] (n 27) |
| LSW | 37.1 [34.9, 39.4] (n 27) | 36.9 [34.8, 39.0] (n 27) | - | 42.8 [40.3, 45.4] (n 27) | 33.2 [31.1, 35.4] (n 27) | 39.2 [36.6, 41.8] (n 27) |
| pooled | 28.7 [27.6, 29.8] (n 27) | 28.0 [27.0, 29.1] (n 27) | - | 33.1 [31.9, 34.3] (n 27) | 22.7 [21.7, 23.7] (n 27) | 24.3 [23.2, 25.4] (n 27) |

Policy on the instances of each search (pooled; per combination in the JSON):

- instances of the memetic algorithm, 600 s (n 27): policy 28.7 [27.6, 29.8], memetic algorithm 22.7 [21.7, 23.7]
- instances of the CP-SAT, 600 s (n 27): policy 28.7 [27.6, 29.8], CP-SAT 24.3 [23.2, 25.4]

| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % | 100 LB / best known: chain / load / stored / reference |
|---|---|---|---|
| N | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 28.9 / 93.4 / 61.6 / 93.4 |
| L | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 54.2 / 84.9 / 64.0 / 84.9 |
| S | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 29.4 / 86.8 / 66.0 / 86.8 |
| W | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 26.2 / 84.7 / 60.3 / 84.7 |
| LS | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 45.7 / 80.3 / 66.7 / 80.3 |
| LW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 50.2 / 78.6 / 62.3 / 78.6 |
| SW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 27.2 / 80.4 / 64.4 / 80.4 |
| LSW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 43.2 / 75.9 / 64.8 / 75.9 |
| pooled | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 38.1 / 83.1 / 63.8 / 83.1 |

Stored solver bounds: 178 used on 56 cells, 0 excluded, 0 written before 2026-07-08 not used.

| folder of the records used | records | budget_s (records) | warm start | verified by | written, oldest to newest | cells where it gives the reference |
|---|---|---|---|---|---|---|
| 50x25_fairwarm1800s | 56 | 1800 s (56) | 56 of 56 | warm start 56 | 2026-08-18 07:39 to 2026-10-15 03:38 | 0 |
| 50x25_fairwarm1800s_rep2 | 2 | 1800 s (2) | 2 of 2 | warm start 2 | 2026-08-20 02:28 to 2026-08-20 02:58 | 0 |
| 50x25_fairwarm1800s_repB | 8 | 1800 s (8) | 8 of 8 | warm start 8 | 2026-08-23 11:42 to 2026-10-16 22:19 | 0 |
| 50x25_fairwarm600s_pactbest3 | 56 | 600 s (56) | 56 of 56 | warm start 56 | 2026-10-16 23:34 to 2026-10-17 09:15 | 0 |
| 50x25_fairwarm600s_pactwarm | 56 | 600 s (56) | 56 of 56 | warm start 56 | 2026-10-15 13:58 to 2026-10-15 20:22 | 0 |

Cells proved optimal (best known makespan equal to the reference bound): 0 of 216.

## 80x25n30 (27 instances per combination)

Station types: 47520 of 47520 operations have their eligible machines in one station type, 47520 the whole type; the eligible sets partition the operations on 27 of 27 instances.

| comb. | policy, run with the lowest validation score | policy, mean over the training seeds | specialist of the combination, mean over the training seeds | best dispatching rule | memetic algorithm, 600 s | CP-SAT, 600 s |
|---|---|---|---|---|---|---|
| N | 16.7 [15.1, 18.4] (n 27) | 14.2 [13.4, 14.9] (n 27) | - | 18.2 [17.1, 19.3] (n 27) | 8.2 [7.7, 8.7] (n 27) | 7.5 [7.1, 7.9] (n 27) |
| L | 26.9 [25.1, 29.0] (n 27) | 23.5 [22.7, 24.2] (n 27) | - | 29.0 [28.0, 30.0] (n 27) | 18.6 [17.7, 19.4] (n 27) | 16.9 [16.1, 17.7] (n 27) |
| S | 17.4 [15.4, 19.8] (n 27) | 18.2 [16.2, 20.3] (n 27) | - | 23.1 [20.9, 25.4] (n 27) | 14.1 [12.2, 16.0] (n 27) | 21.0 [18.8, 23.4] (n 27) |
| W | 23.0 [21.3, 24.6] (n 27) | 20.4 [19.5, 21.2] (n 27) | - | 25.3 [24.1, 26.4] (n 27) | 22.8 [21.7, 23.9] (n 27) | 15.9 [15.3, 16.4] (n 27) |
| LS | 25.2 [23.0, 27.7] (n 27) | 25.1 [22.8, 27.4] (n 27) | - | 29.6 [27.1, 32.2] (n 27) | 20.7 [18.5, 23.0] (n 27) | 27.5 [25.0, 30.2] (n 27) |
| LW | 31.9 [29.8, 34.3] (n 27) | 28.4 [27.4, 29.2] (n 27) | - | 33.7 [32.7, 34.7] (n 27) | 30.2 [29.0, 31.3] (n 27) | 23.4 [22.7, 24.1] (n 27) |
| SW | 22.9 [20.3, 25.6] (n 27) | 23.4 [21.3, 25.7] (n 27) | - | 28.8 [26.4, 31.5] (n 27) | 24.0 [21.7, 26.5] (n 27) | 26.9 [24.3, 29.6] (n 27) |
| LSW | 28.9 [26.4, 31.6] (n 27) | 29.1 [26.8, 31.5] (n 27) | - | 34.6 [32.0, 37.2] (n 27) | 28.7 [26.3, 31.2] (n 27) | 33.1 [30.5, 35.7] (n 27) |
| pooled | 24.1 [22.9, 25.5] (n 27) | 22.8 [21.5, 24.1] (n 27) | - | 27.8 [26.5, 29.1] (n 27) | 20.9 [19.7, 22.1] (n 27) | 21.5 [20.3, 22.8] (n 27) |

Policy on the instances of each search (pooled; per combination in the JSON):

- instances of the memetic algorithm, 600 s (n 27): policy 24.1 [22.9, 25.5], memetic algorithm 20.9 [19.7, 22.1]
- instances of the CP-SAT, 600 s (n 27): policy 24.1 [22.9, 25.5], CP-SAT 21.5 [20.3, 22.8]

| comb. | largest: chain / load / stored, % | alone: chain / load / stored, % | 100 LB / best known: chain / load / stored / reference |
|---|---|---|---|
| N | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 19.7 / 93.4 / 56.8 / 93.4 |
| L | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 39.1 / 86.0 / 57.2 / 86.0 |
| S | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 20.1 / 88.3 / 60.2 / 88.3 |
| W | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 18.3 / 86.8 / 56.6 / 86.8 |
| LS | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 33.3 / 83.7 / 59.8 / 83.7 |
| LW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 36.9 / 81.3 / 56.1 / 81.3 |
| SW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 18.9 / 83.2 / 58.0 / 83.2 |
| LSW | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 31.7 / 79.7 / 58.7 / 79.7 |
| pooled | 0.0 / 100.0 / 0.0 | 0.0 / 100.0 / 0.0 | 27.3 / 85.3 / 57.9 / 85.3 |

Stored solver bounds: 168 used on 56 cells, 0 excluded, 0 written before 2026-07-08 not used.

| folder of the records used | records | budget_s (records) | warm start | verified by | written, oldest to newest | cells where it gives the reference |
|---|---|---|---|---|---|---|
| 80x25_fairwarm1800s | 56 | 1800 s (56) | 56 of 56 | warm start 56 | 2026-08-20 07:47 to 2026-10-15 13:38 | 0 |
| 80x25_fairwarm600s_pactbest3 | 56 | 600 s (56) | 56 of 56 | warm start 56 | 2026-10-16 23:35 to 2026-10-18 13:57 | 0 |
| 80x25_fairwarm600s_pactwarm | 56 | 600 s (56) | 56 of 56 | warm start 56 | 2026-10-15 20:42 to 2026-10-16 08:06 | 0 |

Cells proved optimal (best known makespan equal to the reference bound): 0 of 216.
