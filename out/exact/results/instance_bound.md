# Makespan estimate at the empty schedule against the shortest schedules found (environment exact, training seeds 301,302,303,304,305 (5 runs per arm))

Size 10x25; CP-SAT arm lns and the memetic algorithm, 600 s records that scripts/timing_validity.py accepts; policy seed 304 (greedy); estimate with delta_mode min, mask options {'first_mask': 'exact', 'mask_ties': 'exact'}. Values: 100 (makespan / C^(s_0;c) - 1), %.

| comb. | pairs | found at the estimate | found: mean [min, max] | policy: mean [min, max] |
|---|---|---|---|---|
| N | 10 | 0 | 34.7 [27.2, 47.3] | 45.2 [34.8, 57.1] |
| L | 10 | 6 | 0.5 [0.0, 2.1] | 4.1 [1.4, 10.3] |
| S | 10 | 0 | 44.6 [33.1, 53.1] | 55.6 [41.2, 66.7] |
| W | 10 | 0 | 48.6 [34.8, 62.6] | 61.2 [48.9, 80.2] |
| LS | 10 | 0 | 6.3 [1.7, 12.0] | 13.5 [8.4, 19.2] |
| LW | 10 | 0 | 2.4 [0.9, 7.7] | 8.5 [2.8, 19.5] |
| SW | 10 | 0 | 61.3 [45.6, 72.1] | 78.3 [55.1, 94.4] |
| LSW | 10 | 0 | 13.2 [6.5, 19.8] | 23.9 [14.0, 33.1] |
| pooled | 80 | 6 | 26.4 [0.0, 72.1] | 36.3 [1.4, 94.4] |

Record states (arm|size|combination):

- cpsat_rule|10x25|L: missing 20, valid 10
- cpsat_rule|10x25|LS: missing 20, valid 10
- cpsat_rule|10x25|LSW: missing 20, valid 10
- cpsat_rule|10x25|LW: missing 20, valid 10
- cpsat_rule|10x25|N: missing 20, valid 10
- cpsat_rule|10x25|S: missing 20, valid 10
- cpsat_rule|10x25|SW: missing 20, valid 10
- cpsat_rule|10x25|W: missing 20, valid 10
- memetic_rule|10x25|L: valid 30
- memetic_rule|10x25|LS: valid 30
- memetic_rule|10x25|LSW: valid 30
- memetic_rule|10x25|LW: valid 30
- memetic_rule|10x25|N: valid 30
- memetic_rule|10x25|S: valid 30
- memetic_rule|10x25|SW: valid 30
- memetic_rule|10x25|W: valid 30
