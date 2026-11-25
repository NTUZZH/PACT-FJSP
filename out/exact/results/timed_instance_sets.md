# Timed comparison: the instances of the design against every evidence instance

- inputs: design results/timed_inference_startnine.json (tag startnine, sha256 62095abd548189ab48ed1d9f768bd76fbd86dc6ad758956faad27860a9a4f1c0); evidence results/timed_inference_startall.json (tag startall, sha256 e9a60b45bb6fdd2b50bb8927c6fd1b03d364d96c6d84c973516c5db0dc14714f)
- environment exact, training seeds 301,302,303,304,305 (5 runs per arm); selected training run s304; CP-SAT configuration lns; B = 20000, alpha 0.05
- difference: the makespan of the policy's greedy schedule (selected training run) minus that of the search started from the rule schedule, per instance 100 (policy - search) / search, mean over the instances and combinations of the entry (p94 comparison (i), row Sel; a positive difference means the search has the shorter schedule)
- clock origin start (wall clock from the start of the run: memetic algorithm from the start of the process that solves the instance (instance loading, start schedules and numba warm-up included); CP-SAT model arms from the start of the arm, model building included (instance load and hint validation before it not timed by the runner); the neighbourhood-search arm from its start)
- design: instances 3-11 (9 per combination); evidence: instances 3-29 (27 per combination); shift: the mean on the instances of the timed comparison (design) minus the mean on every evidence instance (evidence), in percentage points

## 50x25n30, memetic algorithm started from the rule schedule

| s | entry | instances 3-11 | instances 3-29 | shift | same direction |
|---|---|---|---|---|---|
| 60 | N | +2.34 [-1.00, +5.74] neither | +2.49 [+0.47, +4.68] behind | -0.15 | no |
| 60 | L | -0.35 [-1.84, +1.08] neither | +0.47 [-0.41, +1.33] neither | -0.82 | yes |
| 60 | S | -2.19 [-3.88, -0.58] ahead | -1.49 [-2.45, -0.52] ahead | -0.70 | yes |
| 60 | W | -1.19 [-3.68, +1.43] neither | -0.72 [-2.51, +1.22] neither | -0.47 | yes |
| 60 | LS | -0.29 [-1.47, +0.82] neither | -1.19 [-1.95, -0.40] ahead | +0.89 | no |
| 60 | LW | -3.66 [-5.43, -1.99] ahead | -2.64 [-3.48, -1.85] ahead | -1.02 | yes |
| 60 | SW | -1.69 [-3.10, -0.21] ahead | -1.26 [-2.25, -0.22] ahead | -0.43 | yes |
| 60 | LSW | -3.00 [-4.42, -1.67] ahead | -2.37 [-3.19, -1.54] ahead | -0.63 | yes |
| 60 | Pool | -1.26 [-2.16, -0.30] ahead | -0.84 [-1.39, -0.24] ahead | -0.42 | yes |
| 60 | combinations without setups | -0.71 [-2.33, +0.88] neither | -0.10 [-1.01, +0.90] neither | -0.62 | yes |
| 60 | combinations with setups | -1.80 [-2.97, -0.64] ahead | -1.58 [-2.13, -1.03] ahead | -0.22 | yes |
| 300 | N | +9.48 [+6.86, +11.94] behind | +9.02 [+7.30, +10.91] behind | +0.46 | yes |
| 300 | L | +6.56 [+5.12, +7.86] behind | +6.56 [+5.54, +7.56] behind | +0.00 | yes |
| 300 | S | +3.10 [+1.86, +4.41] behind | +3.26 [+2.47, +4.16] behind | -0.16 | yes |
| 300 | W | +1.76 [-0.87, +4.29] neither | +2.21 [+0.58, +3.92] behind | -0.45 | no |
| 300 | LS | +5.15 [+3.62, +6.78] behind | +4.38 [+3.59, +5.24] behind | +0.78 | yes |
| 300 | LW | +0.48 [-1.74, +2.59] neither | +1.25 [+0.20, +2.29] behind | -0.77 | no |
| 300 | SW | +2.07 [+0.54, +3.77] behind | +1.77 [+0.86, +2.84] behind | +0.31 | yes |
| 300 | LSW | +2.38 [+1.42, +3.39] behind | +1.91 [+1.17, +2.70] behind | +0.47 | yes |
| 300 | Pool | +3.87 [+2.96, +4.75] behind | +3.79 [+3.27, +4.34] behind | +0.08 | yes |
| 300 | combinations without setups | +4.57 [+2.96, +5.96] behind | +4.76 [+3.94, +5.61] behind | -0.19 | yes |
| 300 | combinations with setups | +3.18 [+2.17, +4.18] behind | +2.83 [+2.26, +3.47] behind | +0.35 | yes |
| 600 | N | +10.93 [+8.32, +13.53] behind | +10.34 [+8.63, +12.24] behind | +0.59 | yes |
| 600 | L | +8.17 [+6.93, +9.40] behind | +8.34 [+7.46, +9.20] behind | -0.18 | yes |
| 600 | S | +4.16 [+2.94, +5.40] behind | +4.19 [+3.41, +5.11] behind | -0.03 | yes |
| 600 | W | +2.96 [+0.67, +5.17] behind | +3.44 [+1.92, +5.05] behind | -0.48 | yes |
| 600 | LS | +5.91 [+4.20, +7.70] behind | +5.25 [+4.46, +6.11] behind | +0.66 | yes |
| 600 | LW | +1.93 [-0.37, +4.20] neither | +3.06 [+2.01, +4.05] behind | -1.13 | no |
| 600 | SW | +2.96 [+1.51, +4.60] behind | +2.72 [+1.81, +3.77] behind | +0.24 | yes |
| 600 | LSW | +3.65 [+2.75, +4.64] behind | +2.90 [+2.19, +3.66] behind | +0.75 | yes |
| 600 | Pool | +5.08 [+4.23, +5.92] behind | +5.03 [+4.54, +5.55] behind | +0.05 | yes |
| 600 | combinations without setups | +6.00 [+4.49, +7.31] behind | +6.30 [+5.52, +7.12] behind | -0.30 | yes |
| 600 | combinations with setups | +4.17 [+3.12, +5.23] behind | +3.77 [+3.17, +4.44] behind | +0.40 | yes |

## 50x25n30, CP-SAT started from the rule schedule

| s | entry | instances 3-11 | instances 3-29 | shift | same direction |
|---|---|---|---|---|---|
| 60 | N | +4.27 [+1.37, +7.26] behind | +3.83 [+1.89, +5.95] behind | +0.45 | yes |
| 60 | L | +0.18 [-1.24, +1.70] neither | +0.68 [-0.12, +1.51] neither | -0.50 | yes |
| 60 | S | -5.39 [-7.41, -3.52] ahead | -3.77 [-4.86, -2.72] ahead | -1.62 | yes |
| 60 | W | +2.85 [+0.76, +5.44] behind | +2.87 [+1.20, +4.74] behind | -0.02 | yes |
| 60 | LS | -3.98 [-5.25, -2.72] ahead | -3.40 [-4.15, -2.57] ahead | -0.58 | yes |
| 60 | LW | -0.48 [-2.01, +0.89] neither | +0.22 [-0.51, +0.90] neither | -0.70 | yes |
| 60 | SW | -2.73 [-4.75, -0.70] ahead | -2.45 [-3.56, -1.31] ahead | -0.28 | yes |
| 60 | LSW | -4.09 [-5.67, -2.42] ahead | -3.52 [-4.51, -2.50] ahead | -0.57 | yes |
| 60 | Pool | -1.17 [-2.04, -0.23] ahead | -0.69 [-1.25, -0.10] ahead | -0.48 | yes |
| 60 | combinations without setups | +1.71 [+0.33, +3.07] behind | +1.90 [+1.02, +2.89] behind | -0.19 | yes |
| 60 | combinations with setups | -4.05 [-5.41, -2.69] ahead | -3.28 [-3.93, -2.66] ahead | -0.76 | yes |
| 300 | N | +10.41 [+7.72, +13.10] behind | +9.68 [+7.85, +11.72] behind | +0.72 | yes |
| 300 | L | +7.42 [+6.13, +8.88] behind | +8.04 [+7.23, +8.89] behind | -0.62 | yes |
| 300 | S | -2.05 [-4.10, -0.12] ahead | -1.34 [-2.34, -0.30] ahead | -0.70 | yes |
| 300 | W | +5.59 [+3.70, +8.08] behind | +5.87 [+4.36, +7.59] behind | -0.28 | yes |
| 300 | LS | -1.87 [-3.37, -0.35] ahead | -1.88 [-2.61, -1.09] ahead | +0.01 | yes |
| 300 | LW | +4.31 [+2.53, +5.81] behind | +5.08 [+4.29, +5.77] behind | -0.77 | yes |
| 300 | SW | -0.58 [-2.14, +1.13] neither | -0.29 [-1.37, +0.89] neither | -0.29 | yes |
| 300 | LSW | -2.66 [-4.02, -1.28] ahead | -2.07 [-2.90, -1.19] ahead | -0.60 | yes |
| 300 | Pool | +2.57 [+1.84, +3.35] behind | +2.89 [+2.41, +3.41] behind | -0.31 | yes |
| 300 | combinations without setups | +6.93 [+5.74, +8.12] behind | +7.17 [+6.41, +8.03] behind | -0.24 | yes |
| 300 | combinations with setups | -1.79 [-3.06, -0.48] ahead | -1.40 [-2.02, -0.74] ahead | -0.39 | yes |
| 600 | N | +11.57 [+9.15, +13.98] behind | +10.64 [+8.86, +12.59] behind | +0.93 | yes |
| 600 | L | +9.00 [+7.78, +10.26] behind | +9.48 [+8.76, +10.23] behind | -0.48 | yes |
| 600 | S | -1.10 [-2.98, +0.75] neither | -0.48 [-1.41, +0.53] neither | -0.63 | yes |
| 600 | W | +6.76 [+4.99, +9.18] behind | +6.86 [+5.34, +8.56] behind | -0.09 | yes |
| 600 | LS | -0.44 [-1.56, +0.70] neither | -0.83 [-1.49, -0.08] ahead | +0.39 | no |
| 600 | LW | +6.24 [+4.80, +7.62] behind | +6.47 [+5.81, +7.09] behind | -0.24 | yes |
| 600 | SW | +0.13 [-1.30, +1.65] neither | +0.28 [-0.78, +1.41] neither | -0.15 | yes |
| 600 | LSW | -2.10 [-3.48, -0.72] ahead | -1.47 [-2.28, -0.63] ahead | -0.63 | yes |
| 600 | Pool | +3.76 [+3.09, +4.42] behind | +3.87 [+3.38, +4.40] behind | -0.11 | yes |
| 600 | combinations without setups | +8.39 [+7.34, +9.41] behind | +8.36 [+7.63, +9.19] behind | +0.03 | yes |
| 600 | combinations with setups | -0.88 [-2.10, +0.34] neither | -0.62 [-1.24, +0.02] neither | -0.26 | yes |

## 80x25n30, memetic algorithm started from the rule schedule

| s | entry | instances 3-11 | instances 3-29 | shift | same direction |
|---|---|---|---|---|---|
| 60 | N | -2.49 [-4.67, -0.06] ahead | -0.68 [-2.31, +0.99] neither | -1.81 | no |
| 60 | L | +0.67 [-3.03, +5.40] neither | -0.99 [-2.68, +1.05] neither | +1.65 | yes |
| 60 | S | -2.34 [-4.08, -0.68] ahead | -1.97 [-3.05, -0.73] ahead | -0.37 | yes |
| 60 | W | -4.52 [-5.81, -3.23] ahead | -1.76 [-3.12, -0.33] ahead | -2.76 | yes |
| 60 | LS | +0.46 [-1.92, +3.53] neither | -0.64 [-1.65, +0.60] neither | +1.10 | yes |
| 60 | LW | -1.78 [-3.42, +0.63] neither | -1.26 [-3.01, +0.80] neither | -0.53 | yes |
| 60 | SW | -4.27 [-6.81, -1.94] ahead | -4.32 [-5.52, -3.17] ahead | +0.05 | yes |
| 60 | LSW | -3.05 [-5.09, -0.48] ahead | -3.75 [-4.80, -2.63] ahead | +0.70 | yes |
| 60 | Pool | -2.16 [-3.41, -0.90] ahead | -1.92 [-2.61, -1.20] ahead | -0.24 | yes |
| 60 | combinations without setups | -2.03 [-3.74, +0.04] neither | -1.17 [-2.22, -0.06] ahead | -0.86 | no |
| 60 | combinations with setups | -2.30 [-3.66, -0.95] ahead | -2.67 [-3.36, -1.96] ahead | +0.37 | yes |
| 300 | N | +4.78 [+2.82, +7.11] behind | +6.34 [+4.85, +7.90] behind | -1.55 | yes |
| 300 | L | +7.30 [+3.78, +11.37] behind | +5.03 [+3.30, +6.97] behind | +2.28 | yes |
| 300 | S | +1.61 [+0.15, +2.98] behind | +2.30 [+1.26, +3.49] behind | -0.68 | yes |
| 300 | W | -3.90 [-5.37, -2.45] ahead | -0.96 [-2.36, +0.47] neither | -2.93 | no |
| 300 | LS | +4.12 [+1.75, +6.98] behind | +2.90 [+1.93, +4.07] behind | +1.21 | yes |
| 300 | LW | -0.66 [-2.41, +1.76] neither | -0.06 [-1.88, +2.02] neither | -0.60 | yes |
| 300 | SW | -2.16 [-4.91, +0.45] neither | -1.72 [-2.88, -0.58] ahead | -0.44 | no |
| 300 | LSW | +0.22 [-1.63, +2.69] neither | -0.85 [-2.02, +0.38] neither | +1.07 | yes |
| 300 | Pool | +1.42 [+0.59, +2.27] behind | +1.62 [+1.03, +2.25] behind | -0.21 | yes |
| 300 | combinations without setups | +1.88 [+0.83, +3.26] behind | +2.59 [+1.76, +3.51] behind | -0.70 | yes |
| 300 | combinations with setups | +0.95 [-0.09, +2.00] neither | +0.66 [+0.03, +1.30] behind | +0.29 | no |
| 600 | N | +6.26 [+4.21, +8.71] behind | +7.89 [+6.38, +9.44] behind | -1.62 | yes |
| 600 | L | +9.44 [+5.87, +13.50] behind | +7.10 [+5.34, +9.05] behind | +2.34 | yes |
| 600 | S | +2.27 [+0.81, +3.63] behind | +2.97 [+1.90, +4.19] behind | -0.70 | yes |
| 600 | W | -2.59 [-3.69, -1.41] ahead | +0.16 [-1.16, +1.53] neither | -2.75 | no |
| 600 | LS | +5.27 [+3.09, +7.89] behind | +3.80 [+2.84, +4.94] behind | +1.47 | yes |
| 600 | LW | +0.95 [-0.89, +3.16] neither | +1.39 [-0.42, +3.43] neither | -0.44 | yes |
| 600 | SW | -1.34 [-3.91, +1.20] neither | -0.90 [-2.00, +0.20] neither | -0.44 | yes |
| 600 | LSW | +1.15 [-0.59, +3.46] neither | +0.20 [-0.80, +1.28] neither | +0.95 | yes |
| 600 | Pool | +2.68 [+1.95, +3.42] behind | +2.83 [+2.27, +3.43] behind | -0.15 | yes |
| 600 | combinations without setups | +3.52 [+2.49, +4.76] behind | +4.13 [+3.35, +4.99] behind | -0.62 | yes |
| 600 | combinations with setups | +1.84 [+0.89, +2.79] behind | +1.52 [+0.91, +2.14] behind | +0.32 | yes |

## 80x25n30, CP-SAT started from the rule schedule

| s | entry | instances 3-11 | instances 3-29 | shift | same direction |
|---|---|---|---|---|---|
| 60 | N | -1.33 [-3.54, +1.20] neither | +0.71 [-0.85, +2.33] neither | -2.04 | yes |
| 60 | L | +1.72 [-2.06, +6.35] neither | -0.09 [-1.77, +1.93] neither | +1.80 | yes |
| 60 | S | -5.56 [-7.57, -3.71] ahead | -4.44 [-5.65, -3.15] ahead | -1.11 | yes |
| 60 | W | -2.36 [-3.49, -1.17] ahead | +0.39 [-0.95, +1.80] neither | -2.75 | no |
| 60 | LS | -2.38 [-4.71, +0.67] neither | -3.16 [-4.35, -1.81] ahead | +0.78 | no |
| 60 | LW | -0.49 [-2.17, +1.95] neither | +0.31 [-1.47, +2.40] neither | -0.80 | yes |
| 60 | SW | -4.45 [-6.95, -2.16] ahead | -4.50 [-5.72, -3.34] ahead | +0.05 | yes |
| 60 | LSW | -3.18 [-5.17, -0.65] ahead | -4.07 [-5.07, -2.97] ahead | +0.89 | yes |
| 60 | Pool | -2.25 [-3.61, -0.92] ahead | -1.85 [-2.58, -1.12] ahead | -0.40 | yes |
| 60 | combinations without setups | -0.61 [-2.22, +1.36] neither | +0.33 [-0.65, +1.36] neither | -0.95 | yes |
| 60 | combinations with setups | -3.89 [-5.46, -2.57] ahead | -4.04 [-4.82, -3.29] ahead | +0.15 | yes |
| 300 | N | +5.23 [+3.37, +7.42] behind | +6.66 [+5.27, +8.11] behind | -1.44 | yes |
| 300 | L | +8.04 [+4.26, +12.49] behind | +5.73 [+3.91, +7.82] behind | +2.31 | yes |
| 300 | S | -4.80 [-6.75, -3.03] ahead | -3.71 [-4.87, -2.48] ahead | -1.09 | yes |
| 300 | W | +2.02 [+1.19, +2.91] behind | +4.68 [+3.44, +5.99] behind | -2.66 | yes |
| 300 | LS | -1.78 [-4.18, +1.35] neither | -2.48 [-3.66, -1.13] ahead | +0.70 | no |
| 300 | LW | +4.61 [+2.93, +6.92] behind | +4.85 [+3.09, +6.86] behind | -0.25 | yes |
| 300 | SW | -3.63 [-6.06, -1.36] ahead | -3.86 [-5.02, -2.73] ahead | +0.23 | yes |
| 300 | LSW | -2.85 [-4.93, -0.21] ahead | -3.60 [-4.64, -2.47] ahead | +0.75 | yes |
| 300 | Pool | +0.85 [-0.29, +1.97] neither | +1.03 [+0.40, +1.68] behind | -0.18 | no |
| 300 | combinations without setups | +4.97 [+3.71, +6.56] behind | +5.48 [+4.70, +6.33] behind | -0.51 | yes |
| 300 | combinations with setups | -3.27 [-4.86, -1.85] ahead | -3.41 [-4.17, -2.68] ahead | +0.15 | yes |
| 600 | N | +7.01 [+5.20, +9.15] behind | +8.55 [+7.12, +10.03] behind | -1.54 | yes |
| 600 | L | +11.02 [+7.62, +15.07] behind | +8.60 [+6.78, +10.64] behind | +2.42 | yes |
| 600 | S | -4.11 [-6.07, -2.37] ahead | -2.92 [-4.07, -1.71] ahead | -1.19 | yes |
| 600 | W | +3.14 [+2.25, +4.02] behind | +6.12 [+4.74, +7.57] behind | -2.98 | yes |
| 600 | LS | -1.01 [-3.50, +2.16] neither | -1.74 [-2.84, -0.45] ahead | +0.73 | no |
| 600 | LW | +6.67 [+5.09, +8.71] behind | +6.91 [+5.15, +8.94] behind | -0.25 | yes |
| 600 | SW | -2.85 [-5.21, -0.78] ahead | -3.10 [-4.20, -2.04] ahead | +0.24 | yes |
| 600 | LSW | -2.49 [-4.56, +0.06] neither | -3.11 [-4.15, -1.99] ahead | +0.62 | no |
| 600 | Pool | +2.17 [+1.15, +3.20] behind | +2.42 [+1.83, +3.02] behind | -0.24 | yes |
| 600 | combinations without setups | +6.96 [+5.87, +8.35] behind | +7.55 [+6.83, +8.35] behind | -0.59 | yes |
| 600 | combinations with setups | -2.62 [-4.22, -1.25] ahead | -2.72 [-3.46, -2.01] ahead | +0.10 | yes |

## Summary

| level | entries | same direction | flips (ahead against behind) | largest absolute shift | where |
|---|---|---|---|---|---|
| pooled | 36 | 33 | 0 | 0.95 | 80x25n30 CP-SAT 60 s combinations without setups (-0.95) |
| pool | 12 | 11 | 0 | 0.48 | 50x25n30 CP-SAT 60 s Pool (-0.48) |
| single | 96 | 81 | 0 | 2.98 | 80x25n30 CP-SAT 600 s W (-2.98) |
| all | 132 | 114 | 0 | 2.98 | 80x25n30 CP-SAT 600 s W (-2.98) |

## Entries whose direction word differs (18)

- 50x25n30 memetic algorithm 60 s N: +2.34 [-1.00, +5.74] neither | +2.49 [+0.47, +4.68] behind
- 50x25n30 memetic algorithm 60 s LS: -0.29 [-1.47, +0.82] neither | -1.19 [-1.95, -0.40] ahead
- 50x25n30 memetic algorithm 300 s W: +1.76 [-0.87, +4.29] neither | +2.21 [+0.58, +3.92] behind
- 50x25n30 memetic algorithm 300 s LW: +0.48 [-1.74, +2.59] neither | +1.25 [+0.20, +2.29] behind
- 50x25n30 memetic algorithm 600 s LW: +1.93 [-0.37, +4.20] neither | +3.06 [+2.01, +4.05] behind
- 50x25n30 CP-SAT 600 s LS: -0.44 [-1.56, +0.70] neither | -0.83 [-1.49, -0.08] ahead
- 80x25n30 memetic algorithm 60 s N: -2.49 [-4.67, -0.06] ahead | -0.68 [-2.31, +0.99] neither
- 80x25n30 memetic algorithm 60 s combinations without setups: -2.03 [-3.74, +0.04] neither | -1.17 [-2.22, -0.06] ahead
- 80x25n30 memetic algorithm 300 s W: -3.90 [-5.37, -2.45] ahead | -0.96 [-2.36, +0.47] neither
- 80x25n30 memetic algorithm 300 s SW: -2.16 [-4.91, +0.45] neither | -1.72 [-2.88, -0.58] ahead
- 80x25n30 memetic algorithm 300 s combinations with setups: +0.95 [-0.09, +2.00] neither | +0.66 [+0.03, +1.30] behind
- 80x25n30 memetic algorithm 600 s W: -2.59 [-3.69, -1.41] ahead | +0.16 [-1.16, +1.53] neither
- 80x25n30 CP-SAT 60 s W: -2.36 [-3.49, -1.17] ahead | +0.39 [-0.95, +1.80] neither
- 80x25n30 CP-SAT 60 s LS: -2.38 [-4.71, +0.67] neither | -3.16 [-4.35, -1.81] ahead
- 80x25n30 CP-SAT 300 s LS: -1.78 [-4.18, +1.35] neither | -2.48 [-3.66, -1.13] ahead
- 80x25n30 CP-SAT 300 s Pool: +0.85 [-0.29, +1.97] neither | +1.03 [+0.40, +1.68] behind
- 80x25n30 CP-SAT 600 s LS: -1.01 [-3.50, +2.16] neither | -1.74 [-2.84, -0.45] ahead
- 80x25n30 CP-SAT 600 s LSW: -2.49 [-4.56, +0.06] neither | -3.11 [-4.15, -1.99] ahead
