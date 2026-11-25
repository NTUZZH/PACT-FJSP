# Selection of the solver configuration on the selection runs

- records: test_results/FAMILY/<size>_cparms/pilot/<candidate>/<R>/instance_<i>.json, 192 records; sizes 50x25n30, 80x25n30; combinations S SW LSW N; instances 000, 001, 002; budget 600 s, 8 workers; CPU-share bars test_results/FAMILY/timed_campaign_share_bars.json
- score: mean over the runs of a part of the percent by which the reported makespan exceeds the shortest reported makespan of the eight candidates; the earliest row within 0.1 points of the lowest score over all runs is selected
- selected: row 8 (lns, neighborhood search over sub-models); lowest score 0.000; within the margin: lns; the seven others 3.877 to 4.152
- the warm start is the same for the eight candidates on every run: True

## Scores (three decimals, the format of `timed_campaign_plan.py select`)

| row | candidate | all | 50x25n30 | 80x25n30 | setup | nosetup | all@60s | all@300s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | ref | 4.101 | 4.856 | 3.346 | 2.943 | 7.575 | 0.700 | 2.842 |
| 2 | ref_fullhint | 4.152 | 4.921 | 3.383 | 2.969 | 7.702 | 0.748 | 2.881 |
| 3 | ref_presolve_light_fullhint | 4.152 | 4.886 | 3.417 | 2.979 | 7.669 | 0.805 | 2.902 |
| 4 | ref_lnsonly_fullhint | 4.114 | 4.926 | 3.301 | 2.880 | 7.813 | 0.756 | 2.948 |
| 5 | compact | 3.935 | 4.625 | 3.244 | 2.935 | 6.933 | 0.690 | 2.852 |
| 6 | fixassign | 3.965 | 4.559 | 3.372 | 2.834 | 7.359 | 0.717 | 2.799 |
| 7 | radius | 3.877 | 4.638 | 3.116 | 2.812 | 7.073 | 0.667 | 2.743 |
| 8 | lns | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.025 | 0.000 |

## Printed scores (bold = lowest printed value of the column), shortest makespans and improvement over the warm start (%)

| row | configuration | all | 50x25n30 | 80x25n30 | setup | nosetup | shortest | improvement all runs | improvement 50 modules | improvement 80 modules | improvement runs with setups | improvement runs without setups |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | partial warm start | 4.10 | 4.86 | 3.35 | 2.94 | 7.58 | 0 of 24 | 0.41 | 0.68 | 0.15 | 0.13 | 1.27 |
| 2 | full model | 4.15 | 4.92 | 3.38 | 2.97 | 7.70 | 0 of 24 | 0.37 | 0.62 | 0.11 | 0.11 | 1.16 |
| 3 | lighter presolve | 4.15 | 4.89 | 3.42 | 2.98 | 7.67 | 0 of 24 | 0.37 | 0.66 | 0.08 | 0.10 | 1.18 |
| 4 | solver's neighborhood search | 4.11 | 4.93 | 3.30 | 2.88 | 7.81 | 0 of 24 | 0.41 | 0.62 | 0.19 | 0.19 | 1.05 |
| 5 | compact formulation | 3.93 | 4.62 | 3.24 | 2.94 | 6.93 | 0 of 24 | 0.57 | 0.89 | 0.24 | 0.14 | 1.86 |
| 6 | fixed machine assignment | 3.97 | 4.56 | 3.37 | 2.83 | 7.36 | 0 of 24 | 0.54 | 0.96 | 0.12 | 0.24 | 1.47 |
| 7 | restricted arcs | 3.88 | 4.64 | 3.12 | 2.81 | 7.07 | 0 of 24 | 0.63 | 0.89 | 0.37 | 0.26 | 1.73 |
| 8 | neighborhood search over sub-models | **0.00** | **0.00** | **0.00** | **0.00** | **0.00** | 24 of 24 | 4.24 | 5.21 | 3.28 | 2.93 | 8.20 |

## By size and setup group

| row | group | runs | improvement (%) | CPU share median [min, max] | peak memory GB median / max |
|---|---|---:|---:|---|---|
| 1 | 50 modules, without setups | 3 | 2.15 | 1.00 [1.00, 1.00] | 1.2 / 1.2 |
| 1 | 50 modules, with setups | 9 | 0.19 | 0.91 [0.91, 0.92] | 17.8 / 19.4 |
| 1 | 80 modules, without setups | 3 | 0.39 | 1.00 [1.00, 1.00] | 1.7 / 1.7 |
| 1 | 80 modules, with setups | 9 | 0.07 | 0.78 [0.78, 0.79] | 34.8 / 37.1 |
| 2 | 50 modules, without setups | 3 | 1.86 | 1.00 [1.00, 1.00] | 1.2 / 1.2 |
| 2 | 50 modules, with setups | 9 | 0.21 | 0.91 [0.91, 0.92] | 19.4 / 19.8 |
| 2 | 80 modules, without setups | 3 | 0.45 | 1.00 [1.00, 1.00] | 1.7 / 1.7 |
| 2 | 80 modules, with setups | 9 | 0.00 | 0.78 [0.76, 0.78] | 35.5 / 36.1 |
| 3 | 50 modules, without setups | 3 | 2.05 | 1.00 [1.00, 1.00] | 1.2 / 1.2 |
| 3 | 50 modules, with setups | 9 | 0.19 | 0.96 [0.95, 0.96] | 19.4 / 20.0 |
| 3 | 80 modules, without setups | 3 | 0.32 | 1.00 [1.00, 1.00] | 1.6 / 1.7 |
| 3 | 80 modules, with setups | 9 | 0.00 | 0.89 [0.89, 0.90] | 36.2 / 38.2 |
| 4 | 50 modules, without setups | 3 | 1.40 | 1.00 [1.00, 1.00] | 1.2 / 1.2 |
| 4 | 50 modules, with setups | 9 | 0.37 | 0.91 [0.91, 0.92] | 10.7 / 11.2 |
| 4 | 80 modules, without setups | 3 | 0.71 | 1.00 [1.00, 1.00] | 1.4 / 1.4 |
| 4 | 80 modules, with setups | 9 | 0.01 | 0.79 [0.77, 0.79] | 21.1 / 21.9 |
| 5 | 50 modules, without setups | 3 | 2.88 | 1.00 [1.00, 1.00] | 0.8 / 0.8 |
| 5 | 50 modules, with setups | 9 | 0.23 | 0.94 [0.94, 0.95] | 13.6 / 13.8 |
| 5 | 80 modules, without setups | 3 | 0.84 | 0.99 [0.99, 1.00] | 1.1 / 1.1 |
| 5 | 80 modules, with setups | 9 | 0.04 | 0.86 [0.85, 0.86] | 30.3 / 30.7 |
| 6 | 50 modules, without setups | 3 | 2.61 | 1.00 [1.00, 1.00] | 0.6 / 0.7 |
| 6 | 50 modules, with setups | 9 | 0.41 | 0.98 [0.98, 0.98] | 2.4 / 2.5 |
| 6 | 80 modules, without setups | 3 | 0.32 | 1.00 [1.00, 1.00] | 0.6 / 0.6 |
| 6 | 80 modules, with setups | 9 | 0.06 | 0.97 [0.97, 0.97] | 4.8 / 5.2 |
| 7 | 50 modules, without setups | 3 | 2.50 | 1.00 [1.00, 1.00] | 0.8 / 0.8 |
| 7 | 50 modules, with setups | 9 | 0.35 | 0.96 [0.94, 0.98] | 3.3 / 3.6 |
| 7 | 80 modules, without setups | 3 | 0.97 | 0.99 [0.99, 0.99] | 1.1 / 1.2 |
| 7 | 80 modules, with setups | 9 | 0.16 | 0.94 [0.91, 0.96] | 4.6 / 5.0 |
| 8 | 50 modules, without setups | 3 | 9.75 | 0.94 [0.93, 0.95] | 1.0 / 1.1 |
| 8 | 50 modules, with setups | 9 | 3.70 | 0.67 [0.60, 0.73] | 1.7 / 2.8 |
| 8 | 80 modules, without setups | 3 | 6.65 | 0.89 [0.87, 0.89] | 1.1 / 1.1 |
| 8 | 80 modules, with setups | 9 | 2.15 | 0.58 [0.52, 0.62] | 2.1 / 3.0 |

## Full model (row 2, ref_fullhint)

| group | runs | variables median [min, max] | missing counts | build s median | peak memory GB median / max | improvement (%) |
|---|---:|---|---:|---:|---|---:|
| 50 modules, without setups | 3 | 19,225 [19,213, 19,249] | 0 | 0.26 | 1.2 / 1.2 | 1.86 |
| 50 modules, with setups | 9 | 561,446 [561,026, 562,796] | 0 | 11.99 | 19.4 / 19.8 | 0.21 |
| 80 modules, without setups | 3 | 30,673 [30,625, 30,697] | 0 | 0.43 | 1.7 / 1.7 | 0.45 |
| 80 modules, with setups | 9 | 1,414,166 [1,413,656, 1,417,226] | 0 | 33.23 | 35.5 / 36.1 | 0.00 |
