# E5 with one-second columns (review copy of paper/tables/e5_table_aei.tex)

Mean makespan (h) per regime, 100 instances; CP 300 s on a 30-instance prefix (italic). **bold** = row minimum over all 100-instance columns; _underline_ = minimum over the columns that answer within 1 s (PACT, HGNN, SPT, pool, GA 1 s, CP 1 s); + = the CP-SAT search improved on its warm start on no instance.

| regime | PACT | HGNN | SPT rule | rule pool | GA 1 s | CP 1 s | GA 30 s | CP 5 s | CP 30 s | CP 300 s | opt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 130.1 | 136.6 | 132.6 | 132.6 | 132.4 | _129.8_ | 132.1 | 127.5 | **124.4** | *121.0* | *0* |
| L | 211.9 | 215.9 | 216.3 | 215.8 | 215.3 | _207.0_ | 214.2 | 204.2 | **203.5** | *202.5* | *26* |
| S | 194.2 | 196.7 | 195.6 | 195.6 | _194.1_ | 195.4+ | 192.8 | 195.4+ | **191.2** | *188.2* | *0* |
| W | 147.1 | 153.3 | 151.0 | 151.0 | 150.5 | _146.2_ | 150.3 | 142.5 | **139.4** | *133.5* | *0* |
| LS | _269.8_ | 272.1 | 275.3 | 275.2 | 273.4 | 273.1+ | 272.0 | 273.1+ | **269.0** | *265.5* | *0* |
| LW | 221.3 | 224.5 | 227.5 | 226.7 | 226.4 | _218.5_ | 226.7 | 212.8 | **209.9** | *208.0* | *16* |
| SW | _218.6_ | 221.4 | 222.2 | 222.2 | 221.3 | 221.2+ | 220.2 | 221.2+ | **217.5** | *215.9* | *0* |
| LSW | **_286.7_** | 290.0 | 292.6 | 292.6 | 291.5 | 290.0+ | 290.5 | 290.0+ | **286.7** | *284.2* | *0* |

PACT alone best within 1 s: 3 of 8 (LS, SW, LSW).
PACT alone best over all 100-instance columns: 0 of 8 (none).

## Per-regime gap of each one-second rival to PACT

Mean per-instance relative gap, positive = the rival is slower; paired bootstrap, 20k resamples, seed 7. CP 1 s column cross-checked against \AnyOne<regime>Lead in paper/macros_anytime.tex.

| regime | GA 1 s - PACT (%) [95% CI] | CP 1 s - PACT (%) [95% CI] |
|---|---|---|
| N | +1.75 [+1.26, +2.22] | -0.18 [-0.62, +0.27] |
| L | +1.69 [+1.16, +2.23] | -2.25 [-2.66, -1.82] |
| S | +0.00 [-0.60, +0.59] | +0.67 [+0.13, +1.19] |
| W | +2.38 [+1.67, +3.10] | -0.54 [-1.19, +0.11] |
| LS | +1.35 [+0.94, +1.78] | +1.24 [+0.87, +1.61] |
| LW | +2.37 [+1.80, +2.95] | -1.20 [-1.65, -0.75] |
| SW | +1.29 [+0.68, +1.91] | +1.23 [+0.66, +1.82] |
| LSW | +1.70 [+1.11, +2.31] | +1.17 [+0.72, +1.63] |
