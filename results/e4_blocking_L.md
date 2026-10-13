# E4 boundary map: BLOCKING (regime L, model 10x25+family+jointblind-v1)

| method | DNF (deadlock) | completed mean ms | n completed |
|---|---|---|---|
| greedy | 65/100 (65%) | 290.8 | 35 |
| MWKR | 66/100 (66%) | 291.1 | 34 |
| SPT | 30/100 (30%) | 288.3 | 70 |
| FIFO | 36/100 (36%) | 289.5 | 64 |

Admissibility audit: the family bound OVER-estimates the realized blocking makespan on 0/35 completed greedy schedules. Blocking only ADDS delay, so the optimistic monotone-delay bound remains a valid lower bound (expected 0 over-estimations); the out-of-family failure is therefore dynamics-inexpressibility and DEADLOCK, not reward unsoundness.
