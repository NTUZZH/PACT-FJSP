# Selection of the configuration of the memetic algorithm

- generator: scripts/p106_memetic_selection.py; records: baselines/memetic/pilot_out/r2_sel/<A|B|C>/<size>/<combination>/instance_<i>_seed1.json, 72 records; the folder H is not a candidate and is not read
- design: sizes 50x25n30, 80x25n30; combinations N S SW LSW; instances 000, 001, 002; random seed 1; budget 300 s of processor time on one core (clock cpu); the stored schedule of the best dispatching rule (warm_cache folder of the size), the only one
- score: mean over cells of 100 (makespan - best) / best, best = the shortest makespan of the three candidates in the cell; a tie counts for each
- selected: configuration 3; ties for the shortest makespan: 0

## Candidates (values that differ; the others are common)

| configuration | folder | n_offspring | init_frac | ts_cap | ts_scale | its_share | score (%) | best cells | runs without a generation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | A | 20 | null | 10 | 5.0 | 0.0 | 2.63 | 1 of 24 | 24 of 24 |
| 2 | B | 20 | null | 50 | 5.0 | 0.0 | 4.08 | 3 of 24 | 19 of 24 |
| 3 | C | 5 | 0.1 | 10 | 0.5 | 0.9 | 0.11 | 20 of 24 | 0 of 24 |

Derived values by size (number of operations N, ts_no_improve = I):

- configuration 1, 50x25n30: N = 1100, ts_no_improve 5500, ts_iter 55000, its_no_improve 2750, its_iter 27500
- configuration 1, 80x25n30: N = 1760, ts_no_improve 8800, ts_iter 88000, its_no_improve 4400, its_iter 44000
- configuration 2, 50x25n30: N = 1100, ts_no_improve 5500, ts_iter 55000, its_no_improve 2750, its_iter 27500
- configuration 2, 80x25n30: N = 1760, ts_no_improve 8800, ts_iter 88000, its_no_improve 4400, its_iter 44000
- configuration 3, 50x25n30: N = 1100, ts_no_improve 550, ts_iter 5500, its_no_improve 275, its_iter 2750
- configuration 3, 80x25n30: N = 1760, ts_no_improve 880, ts_iter 8800, its_no_improve 440, its_iter 4400

Common configuration: `{"decoder": "insert", "init_split": [0.6, 0.3, 0.1], "its_accept": "not_worse", "its_kick": 4, "mut_ma_ops": 2, "p_cross": 0.9, "p_mut_ma": 0.2, "p_mut_os": 0.2, "pop_size": 20, "restart_keep": 2, "stall_gens": 15, "tenure_min_base": 10, "tie_break": "ncrit", "tournament": 2, "ts_cap_n7": 5, "ts_n7": true}`

## Scores by part (%)

| configuration | all cells (24) | 50 modules (12) | 80 modules (12) | combination N (6) | combination S (6) | combination SW (6) | combination LSW (6) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.626 | 3.274 | 1.977 | 3.669 | 0.936 | 2.865 | 3.032 |
| 2 | 4.082 | 4.796 | 3.368 | 6.413 | 5.767 | 1.188 | 2.959 |
| 3 | 0.110 | 0.032 | 0.187 | 0.000 | 0.034 | 0.250 | 0.155 |

## Makespan per cell

| size | combination | instance | configuration 1 | configuration 2 | configuration 3 | shortest | best |
|---|---|---|---:|---:|---:|---:|---|
| 50x25n30 | N | 000 | 339 | 359 | 326 | 326 | 3 |
| 50x25n30 | N | 001 | 338 | 345 | 323 | 323 | 3 |
| 50x25n30 | N | 002 | 342 | 353 | 334 | 334 | 3 |
| 50x25n30 | S | 000 | 481 | 500 | 472 | 472 | 3 |
| 50x25n30 | S | 001 | 446 | 465 | 441 | 441 | 3 |
| 50x25n30 | S | 002 | 493 | 511 | 494 | 493 | 1 |
| 50x25n30 | SW | 000 | 549 | 536 | 523 | 523 | 3 |
| 50x25n30 | SW | 001 | 502 | 489 | 474 | 474 | 3 |
| 50x25n30 | SW | 002 | 551 | 536 | 537 | 536 | 2 |
| 50x25n30 | LSW | 000 | 581 | 602 | 577 | 577 | 3 |
| 50x25n30 | LSW | 001 | 568 | 563 | 536 | 536 | 3 |
| 50x25n30 | LSW | 002 | 645 | 645 | 615 | 615 | 3 |
| 80x25n30 | N | 000 | 507 | 522 | 496 | 496 | 3 |
| 80x25n30 | N | 001 | 507 | 513 | 481 | 481 | 3 |
| 80x25n30 | N | 002 | 522 | 525 | 505 | 505 | 3 |
| 80x25n30 | S | 000 | 711 | 760 | 700 | 700 | 3 |
| 80x25n30 | S | 001 | 644 | 692 | 643 | 643 | 3 |
| 80x25n30 | S | 002 | 715 | 733 | 709 | 709 | 3 |
| 80x25n30 | SW | 000 | 771 | 763 | 773 | 763 | 2 |
| 80x25n30 | SW | 001 | 679 | 675 | 666 | 666 | 3 |
| 80x25n30 | SW | 002 | 781 | 778 | 777 | 777 | 3 |
| 80x25n30 | LSW | 000 | 874 | 865 | 841 | 841 | 3 |
| 80x25n30 | LSW | 001 | 780 | 773 | 768 | 768 | 3 |
| 80x25n30 | LSW | 002 | 870 | 860 | 868 | 860 | 2 |

## Runs

- processor seconds: 300 to 300.002 over 72 runs
- wall seconds: 300.018 to 787.238 over 72 runs
- processor share (processor over wall seconds): 0.3811 to 0.9999 over 72 runs
- code fingerprints of the runs: 8b0660e1788e16d8

Checks passed on the records:
- every candidate folder holds exactly the 24 records of the design
- validated true and no violation in every record
- size, combination and instance of every record equal to its path
- random seed 1, budget clock cpu, budget 300 s, no evaluation cap, cpu_s at least 300 - 0.0005 s
- one starting schedule, the stored schedule of the best dispatching rule, the same for the three candidates of a cell
- one configuration per candidate; derived values follow the rule of baselines/memetic/search.py from the number of operations, above 300 operations
- the differing configuration keys are the five parameters of the table
