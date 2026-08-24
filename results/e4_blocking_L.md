# E4 boundary map: BLOCKING (regime L, model 10x25+family+joint-v1)

| method | DNF (deadlock) | completed mean ms | n completed |
|---|---|---|---|
| greedy | 52/100 (52%) | 288.5 | 48 |
| MWKR | 66/100 (66%) | 291.1 | 34 |
| SPT | 30/100 (30%) | 288.3 | 70 |
| FIFO | 36/100 (36%) | 289.5 | 64 |

Admissibility audit: the family bound OVER-estimates the realized blocking makespan on 0/48 completed greedy schedules. Blocking only ADDS delay, so the optimistic monotone-delay bound remains a valid lower bound (expected 0 over-estimations); the out-of-family failure is therefore dynamics-inexpressibility and DEADLOCK, not reward unsoundness.
