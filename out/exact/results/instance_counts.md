# The principal comparison in counts and in hours (environment exact, training seeds 301,302,303,304,305,306,307,308,309,310)

- seeds 301, 302, 303, 304, 305, 306, 307, 308, 309, 310: the extension rule applies (W) and all 45 runs of the second five seeds are complete, so the ten seeds of p110
- policy minus specialist, seed s against seed s, 100 test instances per combination; makespans in integer hours
- pair shares: over (seed pair, instance), equal = the same makespan; instance shares: on the per-instance means over the seeds, equal = |difference| < 0.5 h
- hours: mean difference with the hierarchical 95% interval of p88 section A and the seed-pooled instance-level interval, B = 20000; parity stream default_rng(110) in the order ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']; pooled rows fresh default_rng(110); instance-level rows default_rng(110)
- reproduction: 212 numbers and words of section A in out/exact/results/family_inference.json identical at the printed precision; 42 numbers of out/exact/results/seed_extension.json identical

| entry | pairs: shorter / equal / longer, % | instances: shorter / equal / longer, % | mean, h | hierarchical 95% CI, h | instance-level 95% CI, h | mean, % | policy, h | specialist, h |
|---|---|---|---|---|---|---|---|---|
| N | 35.4 / 13.0 / 51.6 | 25.0 / 15.0 / 60.0 | +0.65 | [-0.04, +1.32] neither | [+0.25, +1.04] | +0.56 | 129.8 | 129.2 |
| L | 39.0 / 15.3 / 45.7 | 29.0 / 30.0 / 41.0 | +0.27 | [-0.23, +0.75] neither | [-0.01, +0.55] | +0.15 | 211.2 | 211.0 |
| S | 53.1 / 7.7 / 39.2 | 58.0 / 17.0 / 25.0 | -0.86 | [-1.57, -0.17] ahead | [-1.30, -0.42] | -0.40 | 193.5 | 194.4 |
| W | 38.7 / 11.8 / 49.5 | 26.0 / 25.0 / 49.0 | +0.36 | [-0.56, +1.28] neither | [-0.18, +0.86] | +0.34 | 146.9 | 146.6 |
| LS | 50.9 / 9.6 / 39.5 | 52.0 / 22.0 / 26.0 | -0.70 | [-1.41, -0.03] ahead | [-1.18, -0.24] | -0.23 | 269.2 | 269.9 |
| LW | 42.3 / 12.2 / 45.5 | 28.0 / 25.0 / 47.0 | +0.40 | [-0.35, +1.17] neither | [+0.03, +0.78] | +0.20 | 220.4 | 220.0 |
| SW | 50.4 / 7.3 / 42.3 | 57.0 / 13.0 / 30.0 | -0.80 | [-1.58, +0.00] neither | [-1.28, -0.32] | -0.31 | 217.7 | 218.5 |
| LSW | 50.5 / 7.0 / 42.5 | 51.0 / 22.0 / 27.0 | -0.83 | [-1.70, -0.01] ahead | [-1.38, -0.30] | -0.26 | 286.4 | 287.2 |
| all eight combinations | 45.0 / 10.5 / 44.5 | 40.8 / 21.1 / 38.1 | -0.19 | [-0.52, +0.14] neither | [-0.41, +0.02] | +0.01 | 209.4 | 209.6 |
| the six training combinations | 43.2 / 11.6 / 45.2 | 36.3 / 22.3 / 41.3 | +0.02 | [-0.33, +0.38] neither | [-0.21, +0.24] | +0.10 | 195.2 | 195.2 |
| the two held-out combinations | 50.4 / 7.1 / 42.4 | 54.0 / 17.5 / 28.5 | -0.81 | [-1.38, -0.25] ahead | [-1.19, -0.44] | -0.28 | 252.0 | 252.9 |

## Largest per-instance loss and gain of the policy (means over the seeds)

| entry | loss, h (instance) | gain, h (instance) | loss, % (instance) | gain, % (instance) |
|---|---|---|---|---|
| N | +5.90 (N 4) | -5.30 (N 62) | +4.75 (N 4) | -3.72 (N 62) |
| L | +4.30 (L 6) | -3.10 (L 97) | +2.18 (L 6) | -1.56 (L 97) |
| S | +5.40 (S 81) | -6.60 (S 62) | +2.76 (S 81) | -3.10 (S 62) |
| W | +6.10 (W 13) | -13.70 (W 7) | +4.03 (W 13) | -8.33 (W 7) |
| LS | +6.30 (LS 57) | -12.30 (LS 62) | +2.43 (LS 57) | -4.34 (LS 62) |
| LW | +6.40 (LW 95) | -4.00 (LW 46) | +2.82 (LW 95) | -1.90 (LW 46) |
| SW | +5.00 (SW 81) | -10.20 (SW 58) | +2.27 (SW 81) | -4.34 (SW 58) |
| LSW | +7.50 (LSW 38) | -11.10 (LSW 62) | +2.74 (LSW 38) | -3.65 (LSW 62) |
| all eight combinations | +7.50 (LSW 38) | -13.70 (W 7) | +4.75 (N 4) | -8.33 (W 7) |
| the six training combinations | +6.40 (LW 95) | -13.70 (W 7) | +4.75 (N 4) | -8.33 (W 7) |
| the two held-out combinations | +7.50 (LSW 38) | -11.10 (LSW 62) | +2.74 (LSW 38) | -4.34 (SW 58) |

## Counts

| entry | pairs | shorter | equal | longer | instances | shorter | equal | longer |
|---|---|---|---|---|---|---|---|---|
| N | 1000 | 354 | 130 | 516 | 100 | 25 | 15 | 60 |
| L | 1000 | 390 | 153 | 457 | 100 | 29 | 30 | 41 |
| S | 1000 | 531 | 77 | 392 | 100 | 58 | 17 | 25 |
| W | 1000 | 387 | 118 | 495 | 100 | 26 | 25 | 49 |
| LS | 1000 | 509 | 96 | 395 | 100 | 52 | 22 | 26 |
| LW | 1000 | 423 | 122 | 455 | 100 | 28 | 25 | 47 |
| SW | 1000 | 504 | 73 | 423 | 100 | 57 | 13 | 30 |
| LSW | 1000 | 505 | 70 | 425 | 100 | 51 | 22 | 27 |
| all eight combinations | 8000 | 3603 | 839 | 3558 | 800 | 326 | 169 | 305 |
| the six training combinations | 6000 | 2594 | 696 | 2710 | 600 | 218 | 134 | 248 |
| the two held-out combinations | 2000 | 1009 | 143 | 848 | 200 | 108 | 35 | 57 |
