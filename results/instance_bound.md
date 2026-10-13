# Instance lower bound from the certified estimate at the empty schedule

C^(s_0;c) is Eq. (3) evaluated before any operation is scheduled: per job, the sum over its operations of the predecessor lag (if LAG is active), the shortest compatible processing time and the cheapest inbound setup (if SETUP is active), maximized over jobs. It was computed by the environment at reset (normalized axis, divided by the slope) and independently in numpy on the hour axis; the two agree to 8.5e-14 h on all 800 instance-regime pairs. No policy or CP-SAT makespan falls below it.

Gap to C^ = mean of 100*(makespan/C^ - 1), an upper bound on the optimality gap. PACT pooled over three seeds on all 100 instances; CP-SAT credited min(repaired hint, search) as in Table III; 300 s on the 30-instance prefix (the PACT column restricted to the same prefix is given for comparison). Solver bound = CP-SAT BestObjectiveBound.

| regime | mean C^ (h) | PACT gap to C^ (%) | PACT, 300 s prefix (%) | CP 30 s gap to C^ (%) | CP 300 s gap to C^ (%) | CP 300 s gap to own bound (%) | PACT gap to max(bound, C^), prefix (%) | solver bound / C^ (300 s) | solver bound / C^ (30 s) | solver bound tighter (of n) |
|---|---|---|---|---|---|---|---|---|---|---|
| N | 91.0 | 43.0 | 44.1 | 36.9 | 35.6 | 19.6 | 27.1 | 1.134 | 1.119 | 30/30 |
| L | 200.8 | 5.6 | 5.0 | 1.4 | 0.5 | 0.2 | 4.6 | 1.004 | 1.006 | 10/30 |
| S | 128.9 | 51.1 | 52.7 | 48.8 | 48.8 | 28.7 | 32.0 | 1.158 | 1.127 | 30/30 |
| W | 91.0 | 61.7 | 61.9 | 52.9 | 50.9 | 23.3 | 32.3 | 1.225 | 1.204 | 30/30 |
| LS | 231.2 | 17.0 | 16.4 | 16.7 | 15.1 | 12.5 | 13.8 | 1.023 | 1.019 | 24/30 |
| LW | 200.8 | 10.4 | 9.1 | 4.7 | 3.4 | 1.1 | 6.7 | 1.022 | 1.023 | 25/30 |
| SW | 128.9 | 70.2 | 73.2 | 69.3 | 70.7 | 37.7 | 39.7 | 1.242 | 1.184 | 30/30 |
| LSW | 231.2 | 24.3 | 24.3 | 24.4 | 23.2 | 18.7 | 19.7 | 1.039 | 1.026 | 30/30 |
| *all 8* | | 35.4 | 35.8 | 31.9 | 31.0 | 17.7 | 22.0 | 1.106 | 1.088 | |
| *setup* | | 40.7 | 41.6 | 39.8 | 39.5 | 24.4 | 26.3 | 1.115 | 1.089 | |
| *no setup* | | 30.2 | 30.0 | 24.0 | 22.6 | 11.1 | 17.7 | 1.096 | 1.088 | |

## Plain reading

Pooled over the eight regimes, the PACT makespan is 35.4% above C^(s_0;c), and the 300 s CP-SAT schedule is 31.0% above it. These percentages are upper bounds on the optimality gap. They are tens of percent even for the 300 s solver, so the bound is too loose to report as an optimality certificate. Most of the distance measures the bound, not the schedule: Eq. (3) follows each job chain alone and ignores competition for machines, and it gives WINDOW no credit at all, so it is loosest on N, S, W and SW, where lags do not dominate the job chain. It is close only where long lags make the job chain the binding constraint (L, LW). The CP-SAT proven bound at 300 s averages 1.11 times C^(s_0;c) and exceeds it on 209 of 240 instance-regime pairs, so the solver bound is the tighter of the two. Measured against the better of the two bounds on the 30-instance prefix, the PACT gap bound falls to 22.0%. C^(s_0;c) earns its place as the admissible shaping potential of Theorem 1, and it is not the bound to quote for an optimality gap.
