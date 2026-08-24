# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 210.9 | 9.3 | 1.8702 | 100/100 | 3.82% (100) |
