# Five setup classes (scripts/p103_five_classes.py infer)

Exact mask, exact run names. Arms pact, spec, rules; training seeds 301,302,303,304,305. Level default: the stored descriptors, four classes. Level five: a quarter of the modules (rounded down) form a fifth class whose changeovers are drawn from the generator's ranges. Best rule: per-instance minimum of FIFO, MOR, SPT, MWKR. Shift: 100 (five - default) / default per instance, 95% bootstrap over instances.

## 10x25 (100 instances; 13600 schedules validated, 0 violations)

Fifth class: 2 of 10 modules; new changeovers on average cross 3.27 h, same 1.58 h, cold start 1.61 h; C4 warning share S 0.00, LS 0.00, SW 0.00, LSW 0.00.

| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | SPT changeover (h) | fifth-class share (%) | default level equals stored arrays |
|---|---|---|---|---|---|---|---|
| S | default | 194.8 | 194.5 | 192.7 | 423.5 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| S | five | 196.3 | 195.7 | 194.0 | 435.3 | 29.5 |  |
| LS | default | 274.1 | 272.3 | 270.4 | 423.5 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LS | five | 278.6 | 275.9 | 274.0 | 437.6 | 30.1 |  |
| SW | default | 220.9 | 220.1 | 217.8 | 430.6 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| SW | five | 221.9 | 220.8 | 219.4 | 445.4 | 30.6 |  |
| LSW | default | 292.1 | 290.0 | 286.9 | 431.3 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LSW | five | 296.9 | 293.2 | 291.3 | 448.2 | 31.0 |  |

| combination | SPT shift (%) | best classical rule shift (%) | best of seven shift (%) |
|---|---|---|---|
| S | +0.89 [+0.17, +1.60] | +0.65 [-0.00, +1.30] | +0.72 [+0.12, +1.31] |
| LS | +1.71 [+0.99, +2.45] | +1.35 [+0.81, +1.90] | +1.35 [+0.85, +1.86] |
| SW | +0.51 [-0.16, +1.18] | +0.37 [-0.24, +0.98] | +0.80 [+0.20, +1.41] |
| LSW | +1.75 [+0.98, +2.55] | +1.16 [+0.59, +1.75] | +1.56 [+1.04, +2.10] |

| level | combination | policy - specialist | policy - best classical rule | policy - best of seven |
|---|---|---|---|---|
| default | S | -0.35 [-0.81, +0.07] non-inferior only / equivalent | -0.52 [-1.08, +0.08] not distinguishable | +0.42 [-0.13, +1.01] not distinguishable |
| default | LS | -0.07 [-0.37, +0.25] equivalent / equivalent | -1.10 [-1.53, -0.65] ahead | -0.41 [-0.83, +0.04] not distinguishable |
| default | SW | -0.47 [-0.88, -0.08] non-inferior only / equivalent | -0.98 [-1.47, -0.50] ahead | +0.07 [-0.40, +0.53] not distinguishable |
| default | LSW | -0.15 [-0.53, +0.23] non-inferior only / equivalent | -1.21 [-1.65, -0.77] ahead | -0.14 [-0.58, +0.30] not distinguishable |
| default | pooled | -0.26 [-0.49, -0.02] equivalent / equivalent | -0.95 [-1.26, -0.65] direction ahead | -0.02 [-0.33, +0.29] direction neither |
| five | S | -0.21 [-0.58, +0.18] non-inferior only / equivalent | -0.22 [-0.80, +0.39] not distinguishable | +0.63 [+0.10, +1.19] behind |
| five | LS | +0.04 [-0.24, +0.32] equivalent / equivalent | -0.73 [-1.16, -0.30] ahead | -0.05 [-0.46, +0.37] not distinguishable |
| five | SW | -0.10 [-0.54, +0.32] non-inferior only / equivalent | -0.34 [-0.88, +0.21] not distinguishable | +0.29 [-0.27, +0.84] not distinguishable |
| five | LSW | +0.25 [-0.05, +0.57] inconclusive / equivalent | -0.60 [-1.09, -0.10] ahead | +0.07 [-0.40, +0.56] not distinguishable |
| five | pooled | -0.01 [-0.18, +0.16] equivalent / equivalent | -0.47 [-0.79, -0.16] direction ahead | +0.23 [-0.07, +0.53] direction neither |

Shift of each gap from four to five classes (percentage points):

| combination | policy - specialist | policy - best classical rule | policy - best of the seven rules |
|---|---|---|---|
| S | +0.14 [-0.37, +0.62] | +0.29 [-0.36, +0.95] | +0.21 [-0.39, +0.80] |
| LS | +0.11 [-0.28, +0.50] | +0.37 [-0.12, +0.85] | +0.36 [-0.12, +0.85] |
| SW | +0.37 [-0.15, +0.90] | +0.64 [+0.01, +1.26] | +0.22 [-0.41, +0.84] |
| LSW | +0.40 [-0.07, +0.89] | +0.61 [+0.06, +1.15] | +0.21 [-0.33, +0.74] |
| pooled | +0.25 [-0.01, +0.53] | +0.48 [+0.13, +0.81] | +0.25 [-0.09, +0.58] |

Verdicts that change: pol_spec|v0.5: LSW; pol_spec|v1.0: none; pol_rule|verdict: SW; pol_rule7|verdict: S

Policy arrays: S/default: run, equal to the stored array; S/five: run; LS/default: run, equal to the stored array; LS/five: run; SW/default: run, equal to the stored array; SW/five: run; LSW/default: run, equal to the stored array; LSW/five: run

## 50x25n30 (30 instances; 4080 schedules validated, 0 violations)

Fifth class: 12 of 50 modules; new changeovers on average cross 3.33 h, same 1.70 h, cold start 1.73 h; C4 warning share S 0.00, LS 0.00, SW 0.00, LSW 0.00.

| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | SPT changeover (h) | fifth-class share (%) | default level equals stored arrays |
|---|---|---|---|---|---|---|---|
| S | default | 494.5 | 494.5 | 490.5 | 1962.7 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| S | five | 492.1 | 492.1 | 488.9 | 2048.6 | 32.3 |  |
| LS | default | 582.4 | 582.4 | 577.0 | 1968.6 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LS | five | 581.7 | 581.7 | 573.9 | 2070.5 | 32.5 |  |
| SW | default | 520.7 | 520.7 | 517.2 | 1975.9 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| SW | five | 523.3 | 523.3 | 519.3 | 2054.0 | 32.1 |  |
| LSW | default | 606.1 | 606.1 | 598.8 | 1999.1 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LSW | five | 606.0 | 606.0 | 598.9 | 2090.8 | 32.8 |  |

| combination | SPT shift (%) | best classical rule shift (%) | best of seven shift (%) |
|---|---|---|---|
| S | -0.38 [-1.62, +0.80] | -0.38 [-1.62, +0.80] | -0.24 [-1.37, +0.86] |
| LS | +0.02 [-1.48, +1.64] | +0.02 [-1.48, +1.64] | -0.40 [-1.74, +1.00] |
| SW | +0.61 [-0.62, +1.88] | +0.61 [-0.62, +1.88] | +0.50 [-0.44, +1.50] |
| LSW | +0.07 [-0.98, +1.19] | +0.07 [-0.98, +1.19] | +0.10 [-0.87, +1.13] |

| level | combination | policy - specialist | policy - best classical rule | policy - best of seven |
|---|---|---|---|---|
| default | S | -0.56 [-1.92, +0.67] inconclusive / non-inferior only | -3.69 [-4.92, -2.43] ahead | -2.91 [-4.13, -1.74] ahead |
| default | LS | -0.51 [-1.53, +0.57] inconclusive / non-inferior only | -4.28 [-5.32, -3.21] ahead | -3.37 [-4.46, -2.26] ahead |
| default | SW | -1.33 [-3.02, -0.12] non-inferior only / non-inferior only | -2.84 [-3.92, -1.79] ahead | -2.21 [-3.12, -1.29] ahead |
| default | LSW | -0.60 [-1.91, +0.45] non-inferior only / non-inferior only | -4.20 [-5.25, -3.21] ahead | -3.06 [-3.96, -2.21] ahead |
| default | pooled | -0.75 [-1.79, +0.16] non-inferior only / non-inferior only | -3.75 [-4.57, -2.92] direction ahead | -2.89 [-3.70, -2.07] direction ahead |
| five | S | -0.42 [-1.32, +0.37] non-inferior only / non-inferior only | -3.13 [-4.01, -2.24] ahead | -2.49 [-3.33, -1.66] ahead |
| five | LS | -0.14 [-1.12, +0.71] inconclusive / non-inferior only | -2.92 [-3.87, -1.92] ahead | -1.61 [-2.55, -0.65] ahead |
| five | SW | -1.14 [-2.71, +0.14] non-inferior only / non-inferior only | -3.10 [-4.22, -1.97] ahead | -2.40 [-3.37, -1.39] ahead |
| five | LSW | -0.23 [-1.44, +0.92] inconclusive / non-inferior only | -3.40 [-4.38, -2.36] ahead | -2.27 [-3.14, -1.36] ahead |
| five | pooled | -0.48 [-1.45, +0.34] non-inferior only / non-inferior only | -3.14 [-3.90, -2.35] direction ahead | -2.19 [-2.96, -1.42] direction ahead |

Shift of each gap from four to five classes (percentage points):

| combination | policy - specialist | policy - best classical rule | policy - best of the seven rules |
|---|---|---|---|
| S | +0.13 [-0.98, +1.38] | +0.56 [-0.58, +1.77] | +0.43 [-0.61, +1.54] |
| LS | +0.36 [-0.43, +1.13] | +1.36 [+0.37, +2.31] | +1.77 [+0.93, +2.64] |
| SW | +0.19 [-0.72, +1.16] | -0.26 [-1.50, +1.07] | -0.18 [-1.13, +0.82] |
| LSW | +0.38 [-0.37, +1.16] | +0.80 [-0.19, +1.79] | +0.79 [-0.12, +1.75] |
| pooled | +0.26 [-0.22, +0.75] | +0.62 [-0.01, +1.25] | +0.70 [+0.15, +1.28] |

Verdicts that change: pol_spec|v0.5: S,LSW; pol_spec|v1.0: none; pol_rule|verdict: none; pol_rule7|verdict: none

Policy arrays: S/default: run, run, equal to the stored array; S/five: run; LS/default: run, run, equal to the stored array; LS/five: run; SW/default: run, run, equal to the stored array; SW/five: run; LSW/default: run, run, equal to the stored array; LSW/five: run

## 80x25n30 (30 instances; 4080 schedules validated, 0 violations)

Fifth class: 20 of 80 modules; new changeovers on average cross 3.21 h, same 1.77 h, cold start 1.53 h; C4 warning share S 0.00, LS 0.00, SW 0.00, LSW 0.00.

| combination | level | SPT (h) | best classical rule (h) | best of seven (h) | SPT changeover (h) | fifth-class share (%) | default level equals stored arrays |
|---|---|---|---|---|---|---|---|
| S | default | 719.2 | 719.2 | 714.3 | 3127.2 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| S | five | 720.4 | 720.4 | 712.5 | 3246.3 | 32.7 |  |
| LS | default | 801.6 | 801.6 | 795.0 | 3162.5 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LS | five | 802.3 | 802.3 | 794.3 | 3290.2 | 33.4 |  |
| SW | default | 753.2 | 753.2 | 747.1 | 3144.1 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| SW | five | 754.6 | 754.6 | 750.0 | 3270.8 | 33.3 |  |
| LSW | default | 830.6 | 830.6 | 822.0 | 3186.2 | 0.0 | FIFO: equal, MOR: equal, SPT: equal, MWKR: equal, ECT: equal, SSPT: equal, LLF: equal |
| LSW | five | 833.7 | 833.7 | 826.0 | 3320.2 | 34.1 |  |

| combination | SPT shift (%) | best classical rule shift (%) | best of seven shift (%) |
|---|---|---|---|
| S | +0.25 [-1.18, +1.63] | +0.25 [-1.18, +1.63] | -0.17 [-1.44, +1.10] |
| LS | +0.16 [-1.03, +1.38] | +0.16 [-1.03, +1.38] | -0.04 [-1.13, +1.12] |
| SW | +0.32 [-0.95, +1.60] | +0.32 [-0.95, +1.60] | +0.47 [-0.66, +1.60] |
| LSW | +0.43 [-0.66, +1.54] | +0.43 [-0.66, +1.54] | +0.53 [-0.28, +1.30] |

| level | combination | policy - specialist | policy - best classical rule | policy - best of seven |
|---|---|---|---|---|
| default | S | -1.94 [-4.46, +0.28] non-inferior only / non-inferior only | -3.90 [-5.01, -2.76] ahead | -3.23 [-4.34, -2.09] ahead |
| default | LS | -0.78 [-2.19, +0.91] inconclusive / non-inferior only | -3.72 [-4.91, -2.57] ahead | -2.94 [-4.07, -1.83] ahead |
| default | SW | -2.17 [-4.00, -0.53] non-inferior only / non-inferior only | -4.23 [-5.42, -3.11] ahead | -3.49 [-4.52, -2.48] ahead |
| default | LSW | -1.30 [-2.85, -0.00] non-inferior only / non-inferior only | -4.10 [-5.02, -3.22] ahead | -3.11 [-4.02, -2.24] ahead |
| default | pooled | -1.55 [-2.97, -0.16] non-inferior only / non-inferior only | -3.99 [-4.81, -3.16] direction ahead | -3.19 [-4.02, -2.36] direction ahead |
| five | S | -1.51 [-3.00, -0.22] non-inferior only / non-inferior only | -3.89 [-5.10, -2.62] ahead | -2.84 [-3.99, -1.66] ahead |
| five | LS | -0.76 [-1.94, +0.30] non-inferior only / non-inferior only | -3.31 [-4.20, -2.45] ahead | -2.33 [-3.14, -1.55] ahead |
| five | SW | -2.06 [-3.90, -0.43] non-inferior only / non-inferior only | -4.27 [-5.26, -3.26] ahead | -3.67 [-4.73, -2.57] ahead |
| five | LSW | -0.83 [-2.04, +0.23] non-inferior only / non-inferior only | -3.73 [-4.68, -2.72] ahead | -2.85 [-3.81, -1.80] ahead |
| five | pooled | -1.29 [-2.41, -0.26] non-inferior only / non-inferior only | -3.80 [-4.58, -2.99] direction ahead | -2.92 [-3.67, -2.13] direction ahead |

Shift of each gap from four to five classes (percentage points):

| combination | policy - specialist | policy - best classical rule | policy - best of the seven rules |
|---|---|---|---|
| S | +0.43 [-0.97, +1.81] | +0.01 [-1.33, +1.45] | +0.39 [-0.79, +1.67] |
| LS | +0.02 [-1.28, +1.25] | +0.41 [-0.78, +1.68] | +0.60 [-0.37, +1.67] |
| SW | +0.11 [-0.76, +1.02] | -0.05 [-1.19, +1.20] | -0.18 [-1.21, +0.90] |
| LSW | +0.47 [-0.47, +1.52] | +0.38 [-0.43, +1.18] | +0.27 [-0.51, +1.02] |
| pooled | +0.26 [-0.35, +0.88] | +0.19 [-0.55, +0.90] | +0.27 [-0.37, +0.91] |

Verdicts that change: pol_spec|v0.5: LS; pol_spec|v1.0: none; pol_rule|verdict: none; pol_rule7|verdict: none

Policy arrays: S/default: run, run, equal to the stored array; S/five: run; LS/default: run, run, equal to the stored array; LS/five: run; SW/default: run, run, equal to the stored array; SW/five: run; LSW/default: run, run, equal to the stored array; LSW/five: run

