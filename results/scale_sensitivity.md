# Sensitivity at the production sizes (scripts/p102_scale_sensitivity.py infer)

Exact mask. Arms pact, spec, single, rules; training seeds 301,302,303,304,305. Level default: the stored windows. Level win_full: windows drawn again by the generator's rule on the horizon of the stored SPT makespan without windows. Best rule: per-instance minimum of FIFO, MOR, SPT, MWKR. Shift: 100 (full - default) / default per instance, 95% bootstrap over instances.

## 50x25n30 (30 instances)

| combination | stored horizon (h) | new horizon (h) | new density (%) | windows per machine, stored | new | C4 warnings |
|---|---|---|---|---|---|---|
| W | 213 | 360 | 10.7 | 4.9 | 7.8 | 0.00 |
| LW | 213 | 447 | 10.6 | 4.9 | 9.6 | 0.00 |
| SW | 213 | 494 | 10.6 | 4.9 | 10.6 | 0.00 |
| LSW | 213 | 582 | 10.7 | 4.9 | 12.6 | 0.00 |

| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | SPT: last window end / makespan [min, max] | SPT: window time (%) | default level equals stored arrays |
|---|---|---|---|---|---|---|---|
| W | default | 389.6 | 389.6 | 389.6 | 0.54 [0.50, 0.57] | 6.3 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| W | win_full | 412.5 | 412.5 | 412.5 | 0.87 [0.81, 0.92] | 9.4 |  |
| LW | default | 469.7 | 469.7 | 467.9 | 0.45 [0.40, 0.47] | 5.2 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LW | win_full | 499.1 | 499.1 | 497.8 | 0.89 [0.85, 0.94] | 9.5 |  |
| SW | default | 520.7 | 520.7 | 517.2 | 0.41 [0.36, 0.45] | 4.7 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| SW | win_full | 571.3 | 571.3 | 569.1 | 0.86 [0.82, 0.91] | 9.2 |  |
| LSW | default | 606.1 | 606.1 | 598.8 | 0.35 [0.32, 0.38] | 4.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LSW | win_full | 656.2 | 656.2 | 648.4 | 0.88 [0.83, 0.92] | 9.5 |  |

| combination | SPT shift (%) | best classical rule shift (%) |
|---|---|---|
| W | +5.91 [+4.79, +7.01] | +5.91 [+4.79, +7.01] |
| LW | +6.27 [+5.34, +7.27] | +6.27 [+5.34, +7.27] |
| SW | +9.75 [+8.56, +10.97] | +9.75 [+8.56, +10.97] |
| LSW | +8.28 [+7.33, +9.26] | +8.28 [+7.33, +9.26] |

| level | combination | policy - specialist | control - specialist | policy - best classical rule | policy - best of seven |
|---|---|---|---|---|---|
| default | W | +1.23 [-1.88, +3.79] inconclusive / inconclusive | -0.14 [-2.75, +1.95] inconclusive / inconclusive | -3.54 [-4.63, -2.30] ahead | -3.54 [-4.63, -2.30] ahead |
| default | LW | -8.83 [-15.80, -3.03] non-inferior only / non-inferior only | -9.95 [-16.97, -4.58] non-inferior only / non-inferior only | -4.47 [-5.38, -3.58] ahead | -4.10 [-5.02, -3.20] ahead |
| default | SW | -1.33 [-3.02, -0.12] non-inferior only / non-inferior only | -1.46 [-3.33, -0.16] non-inferior only / non-inferior only | -2.84 [-3.92, -1.79] ahead | -2.21 [-3.12, -1.29] ahead |
| default | LSW | -0.60 [-1.91, +0.45] non-inferior only / non-inferior only | -1.28 [-2.23, -0.50] non-inferior only / non-inferior only | -4.20 [-5.25, -3.21] ahead | -3.06 [-3.96, -2.21] ahead |
| default | pooled | -2.38 [-4.64, -0.13] non-inferior only / non-inferior only | -3.21 [-5.02, -1.39] non-inferior only / non-inferior only | -3.76 [-4.40, -3.14] direction ahead | -3.23 [-3.83, -2.63] direction ahead |
| win_full | W | +1.33 [-0.96, +3.47] inconclusive / inconclusive | -0.06 [-2.08, +1.68] inconclusive / inconclusive | -4.00 [-4.96, -3.03] ahead | -4.00 [-4.96, -3.03] ahead |
| win_full | LW | -7.30 [-13.69, -2.11] non-inferior only / non-inferior only | -8.87 [-14.88, -4.34] non-inferior only / non-inferior only | -4.53 [-5.56, -3.53] ahead | -4.27 [-5.31, -3.29] ahead |
| win_full | SW | -1.06 [-2.53, +0.21] non-inferior only / non-inferior only | -1.18 [-2.71, -0.02] non-inferior only / non-inferior only | -3.42 [-4.18, -2.64] ahead | -3.04 [-3.81, -2.26] ahead |
| win_full | LSW | -0.68 [-1.78, +0.29] non-inferior only / non-inferior only | -1.43 [-2.26, -0.65] non-inferior only / non-inferior only | -3.64 [-4.53, -2.71] ahead | -2.48 [-3.32, -1.63] ahead |
| win_full | pooled | -1.93 [-4.03, +0.14] non-inferior only / non-inferior only | -2.88 [-4.53, -1.22] non-inferior only / non-inferior only | -3.90 [-4.50, -3.28] direction ahead | -3.45 [-4.02, -2.85] direction ahead |

Shift of each gap from the stored to the new windows (percentage points):

| combination | policy - specialist | single-rule control - specialist | policy - best classical rule | policy - best of the seven rules |
|---|---|---|---|---|
| W | +0.10 [-1.17, +1.30] | +0.08 [-0.72, +1.04] | -0.47 [-1.63, +0.36] | -0.47 [-1.63, +0.36] |
| LW | +1.53 [+0.70, +2.41] | +1.09 [+0.10, +2.21] | -0.06 [-0.79, +0.63] | -0.18 [-0.88, +0.47] |
| SW | +0.27 [-0.59, +1.20] | +0.28 [-0.73, +1.29] | -0.58 [-1.52, +0.51] | -0.82 [-1.56, -0.03] |
| LSW | -0.08 [-0.84, +0.71] | -0.15 [-0.96, +0.65] | +0.56 [-0.45, +1.57] | +0.58 [-0.14, +1.38] |
| pooled | +0.46 [+0.00, +0.89] | +0.33 [-0.11, +0.77] | -0.14 [-0.65, +0.38] | -0.22 [-0.64, +0.18] |

Verdicts that change: pol_spec|v0.5: none; pol_spec|v1.0: none; ctl_spec|v0.5: none; ctl_spec|v1.0: none; pol_rule|verdict: none; pol_rule7|verdict: none

## 80x25n30 (30 instances)

| combination | stored horizon (h) | new horizon (h) | new density (%) | windows per machine, stored | new | C4 warnings |
|---|---|---|---|---|---|---|
| W | 217 | 526 | 10.7 | 4.9 | 11.4 | 0.00 |
| LW | 217 | 615 | 10.5 | 4.9 | 13.1 | 0.00 |
| SW | 217 | 719 | 10.3 | 4.9 | 15.0 | 0.00 |
| LSW | 217 | 802 | 10.3 | 4.9 | 16.7 | 0.00 |

| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | SPT: last window end / makespan [min, max] | SPT: window time (%) | default level equals stored arrays |
|---|---|---|---|---|---|---|---|
| W | default | 558.2 | 558.2 | 558.2 | 0.38 [0.36, 0.41] | 4.4 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| W | win_full | 603.4 | 603.4 | 603.4 | 0.87 [0.84, 0.91] | 9.3 |  |
| LW | default | 638.6 | 638.6 | 636.5 | 0.34 [0.32, 0.35] | 3.8 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LW | win_full | 688.0 | 688.0 | 685.7 | 0.89 [0.85, 0.92] | 9.4 |  |
| SW | default | 753.2 | 753.2 | 747.1 | 0.29 [0.24, 0.32] | 3.3 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| SW | win_full | 826.8 | 826.8 | 823.0 | 0.87 [0.82, 0.94] | 9.0 |  |
| LSW | default | 830.6 | 830.6 | 822.0 | 0.26 [0.23, 0.28] | 3.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LSW | win_full | 915.0 | 915.0 | 907.5 | 0.87 [0.82, 0.92] | 9.0 |  |

| combination | SPT shift (%) | best classical rule shift (%) |
|---|---|---|
| W | +8.11 [+7.32, +8.91] | +8.11 [+7.32, +8.91] |
| LW | +7.73 [+6.96, +8.48] | +7.73 [+6.96, +8.48] |
| SW | +9.83 [+8.50, +11.20] | +9.83 [+8.50, +11.20] |
| LSW | +10.16 [+8.83, +11.53] | +10.16 [+8.83, +11.53] |

| level | combination | policy - specialist | control - specialist | policy - best classical rule | policy - best of seven |
|---|---|---|---|---|---|
| default | W | +0.44 [-3.53, +4.27] inconclusive / inconclusive | -0.83 [-3.81, +1.95] inconclusive / inconclusive | -3.81 [-5.08, -2.36] ahead | -3.81 [-5.08, -2.36] ahead |
| default | LW | -7.47 [-13.11, -2.01] non-inferior only / non-inferior only | -9.63 [-14.25, -5.39] non-inferior only / non-inferior only | -3.90 [-5.38, -1.96] ahead | -3.58 [-5.06, -1.57] ahead |
| default | SW | -2.17 [-4.00, -0.53] non-inferior only / non-inferior only | -1.39 [-3.06, +0.03] non-inferior only / non-inferior only | -4.23 [-5.42, -3.11] ahead | -3.49 [-4.52, -2.48] ahead |
| default | LSW | -1.30 [-2.85, -0.00] non-inferior only / non-inferior only | -1.57 [-2.61, -0.64] non-inferior only / non-inferior only | -4.10 [-5.02, -3.22] ahead | -3.11 [-4.02, -2.24] ahead |
| default | pooled | -2.62 [-5.24, +0.04] non-inferior only / non-inferior only | -3.35 [-5.05, -1.59] non-inferior only / non-inferior only | -4.01 [-4.86, -3.08] direction ahead | -3.50 [-4.33, -2.56] direction ahead |
| win_full | W | +0.55 [-1.73, +3.09] inconclusive / inconclusive | -0.48 [-2.31, +1.28] inconclusive / inconclusive | -4.00 [-4.99, -2.98] ahead | -4.00 [-4.99, -2.98] ahead |
| win_full | LW | -6.70 [-11.04, -2.69] non-inferior only / non-inferior only | -8.57 [-12.16, -5.26] non-inferior only / non-inferior only | -4.27 [-5.12, -3.48] ahead | -3.95 [-4.80, -3.15] ahead |
| win_full | SW | -1.48 [-2.49, -0.42] non-inferior only / non-inferior only | -0.87 [-2.13, +0.33] non-inferior only / non-inferior only | -3.98 [-4.95, -3.01] ahead | -3.53 [-4.49, -2.59] ahead |
| win_full | LSW | -0.58 [-1.89, +0.67] inconclusive / non-inferior only | -1.13 [-2.03, -0.15] non-inferior only / non-inferior only | -4.13 [-5.14, -3.14] ahead | -3.35 [-4.31, -2.44] ahead |
| win_full | pooled | -2.05 [-3.84, -0.12] non-inferior only / non-inferior only | -2.76 [-3.92, -1.44] non-inferior only / non-inferior only | -4.09 [-4.72, -3.42] direction ahead | -3.71 [-4.31, -3.05] direction ahead |

Shift of each gap from the stored to the new windows (percentage points):

| combination | policy - specialist | single-rule control - specialist | policy - best classical rule | policy - best of the seven rules |
|---|---|---|---|---|
| W | +0.12 [-2.36, +2.31] | +0.35 [-1.19, +1.87] | -0.19 [-1.84, +1.11] | -0.19 [-1.84, +1.11] |
| LW | +0.77 [-1.65, +2.79] | +1.06 [-0.33, +2.60] | -0.37 [-2.02, +0.59] | -0.37 [-2.08, +0.62] |
| SW | +0.68 [-0.52, +1.83] | +0.52 [-0.34, +1.43] | +0.25 [-0.72, +1.23] | -0.05 [-0.92, +0.77] |
| LSW | +0.72 [-0.34, +1.80] | +0.44 [-0.40, +1.39] | -0.02 [-0.90, +0.84] | -0.24 [-1.05, +0.52] |
| pooled | +0.57 [-0.98, +1.62] | +0.59 [-0.28, +1.37] | -0.08 [-1.06, +0.68] | -0.21 [-1.19, +0.51] |

Verdicts that change: pol_spec|v0.5: LSW; pol_spec|v1.0: none; ctl_spec|v0.5: none; ctl_spec|v1.0: none; pol_rule|verdict: none; pol_rule7|verdict: none

