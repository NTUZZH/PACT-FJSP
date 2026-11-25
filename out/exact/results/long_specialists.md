# Specialists trained for 2000 updates (environment exact, training seeds 301,302,303,304,305 (5 run(s) per combination))

Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = 20000).

## Specialist trained for 2000 updates against the specialist (1000 updates)

Positive: longer schedules after 2000 updates.

| combination | mean | 95% CI | within 0.5% | direction |
|---|---|---|---|---|
| N | -0.65 | [-1.04, -0.26] | no | ahead |
| L | -0.04 | [-0.30, +0.19] | yes | neither |
| S | -0.06 | [-0.51, +0.38] | no | neither |
| W | -1.26 | [-2.24, -0.41] | no | ahead |
| LS | +0.09 | [-0.20, +0.41] | yes | neither |
| LW | +0.19 | [-0.10, +0.50] | yes | neither |
| SW | -0.33 | [-0.83, +0.17] | no | neither |
| LSW | +0.29 | [-0.17, +0.75] | no | neither |
| pooled, All | -0.22 | [-0.46, +0.00] | yes | neither |
| pooled, Train | -0.29 | [-0.55, -0.05] | no | ahead |
| pooled, Held | -0.02 | [-0.38, +0.34] | yes | neither |

## The policy against each specialist

Positive: the policy is slower. Against the specialist (1000 updates): p88.parity_components on the same seeds (equal to the record: out/exact/results/family_inference.json).

| combination | 2000 updates: mean | 95% CI | 0.5% | 1% | 1000 updates: mean | 95% CI | 0.5% | 1% |
|---|---|---|---|---|---|---|---|---|
| N | +0.92 | [+0.36, +1.42] | inferior | inferior | +0.24 | [-0.40, +0.86] | inconclusive | equivalent |
| L | +0.24 | [-0.05, +0.51] | inconclusive | equivalent | +0.18 | [-0.11, +0.49] | equivalent | equivalent |
| S | -0.26 | [-0.59, +0.05] | non-inferior only | equivalent | -0.35 | [-0.80, +0.06] | non-inferior only | equivalent |
| W | +1.73 | [+1.18, +2.28] | inferior | inferior | +0.40 | [-0.51, +1.31] | inconclusive | inconclusive |
| LS | -0.14 | [-0.47, +0.17] | equivalent | equivalent | -0.07 | [-0.38, +0.25] | equivalent | equivalent |
| LW | +0.30 | [-0.16, +0.79] | inconclusive | equivalent | +0.46 | [+0.06, +0.88] | inferior | equivalent |
| SW | -0.09 | [-0.52, +0.33] | non-inferior only | equivalent | -0.47 | [-0.89, -0.08] | non-inferior only | equivalent |
| LSW | -0.41 | [-0.75, -0.05] | non-inferior only | equivalent | -0.15 | [-0.53, +0.22] | non-inferior only | equivalent |
| pooled, All | +0.29 | [+0.13, +0.43] | equivalent | equivalent | +0.03 | [-0.18, +0.24] | equivalent | equivalent |
| pooled, Train | +0.46 | [+0.30, +0.62] | inferior | equivalent | +0.14 | [-0.10, +0.39] | equivalent | equivalent |
| pooled, Held | -0.25 | [-0.54, +0.03] | non-inferior only | equivalent | -0.31 | [-0.60, -0.01] | non-inferior only | equivalent |

Verdicts that differ: at 0.5% N, L, W, LW; at 1% N, W.

## Validation score during training

Score: mean greedy validation makespan over the MWKR norm, lower is better (four decimals in the log). Curves compared at updates 25 to 1000.

| run | best (2000 updates) | at update | best up to 1000 | at update | best (1000 updates) | at update | change of best, % | largest difference of the curves | first update that differs |
|---|---|---|---|---|---|---|---|---|---|
| xlong-solo-N-s301 | 0.8113 | 1450 | 0.8234 | 950 | 0.8161 | 650 | -0.59 | 0.0252 | 25 |
| xlong-solo-L-s301 | 0.9172 | 725 | 0.9172 | 725 | 0.9209 | 275 | -0.40 | 0.0106 | 25 |
| xlong-solo-S-s301 | 0.8085 | 800 | 0.8085 | 800 | 0.8109 | 950 | -0.30 | 0.0212 | 25 |
| xlong-solo-W-s301 | 0.8061 | 1500 | 0.8137 | 950 | 0.8196 | 875 | -1.65 | 0.0312 | 25 |
| xlong-solo-LS-s301 | 0.8771 | 1950 | 0.8829 | 125 | 0.8835 | 900 | -0.72 | 0.0202 | 25 |
| xlong-solo-LW-s301 | 0.8975 | 1250 | 0.8979 | 425 | 0.8952 | 700 | +0.26 | 0.0142 | 25 |
| xlong-solo-SW-s301 | 0.8116 | 1325 | 0.8189 | 700 | 0.8177 | 825 | -0.75 | 0.0194 | 25 |
| xlong-solo-LSW-s301 | 0.8548 | 675 | 0.8548 | 675 | 0.8555 | 950 | -0.08 | 0.0105 | 25 |
| xlong-solo-N-s302 | 0.8167 | 425 | 0.8167 | 425 | 0.8285 | 975 | -1.42 | 0.0284 | 25 |
| xlong-solo-L-s302 | 0.9169 | 900 | 0.9169 | 900 | 0.9187 | 825 | -0.20 | 0.0313 | 25 |
| xlong-solo-S-s302 | 0.8094 | 1975 | 0.8126 | 625 | 0.8141 | 800 | -0.58 | 0.0312 | 25 |
| xlong-solo-W-s302 | 0.8080 | 1875 | 0.8196 | 725 | 0.8227 | 800 | -1.79 | 0.0211 | 50 |
| xlong-solo-LS-s302 | 0.8801 | 925 | 0.8801 | 925 | 0.8801 | 575 | +0.00 | 0.0305 | 25 |
| xlong-solo-LW-s302 | 0.8926 | 700 | 0.8926 | 700 | 0.8934 | 925 | -0.09 | 0.0187 | 25 |
| xlong-solo-SW-s302 | 0.8145 | 1475 | 0.8248 | 225 | 0.8194 | 325 | -0.60 | 0.0117 | 50 |
| xlong-solo-LSW-s302 | 0.8566 | 1850 | 0.8584 | 900 | 0.8552 | 675 | +0.16 | 0.0121 | 25 |
| xlong-solo-N-s303 | 0.8135 | 1675 | 0.8158 | 775 | 0.8139 | 550 | -0.05 | 0.0332 | 25 |
| xlong-solo-L-s303 | 0.9176 | 1200 | 0.9216 | 1000 | 0.9189 | 275 | -0.14 | 0.0139 | 25 |
| xlong-solo-S-s303 | 0.8096 | 875 | 0.8096 | 875 | 0.8092 | 750 | +0.05 | 0.0602 | 25 |
| xlong-solo-W-s303 | 0.8086 | 1400 | 0.8126 | 975 | 0.8159 | 925 | -0.89 | 0.0796 | 25 |
| xlong-solo-LS-s303 | 0.8763 | 750 | 0.8763 | 750 | 0.8778 | 650 | -0.17 | 0.0097 | 25 |
| xlong-solo-LW-s303 | 0.8973 | 975 | 0.8973 | 975 | 0.8979 | 1000 | -0.07 | 0.0175 | 25 |
| xlong-solo-SW-s303 | 0.8168 | 925 | 0.8168 | 925 | 0.8146 | 975 | +0.27 | 0.0352 | 25 |
| xlong-solo-LSW-s303 | 0.8557 | 1025 | 0.8579 | 1000 | 0.8584 | 475 | -0.31 | 0.0140 | 25 |
| xlong-solo-N-s304 | 0.8174 | 1900 | 0.8333 | 1000 | 0.8298 | 950 | -1.49 | 0.0163 | 25 |
| xlong-solo-L-s304 | 0.9183 | 1950 | 0.9222 | 550 | 0.9178 | 625 | +0.05 | 0.0181 | 25 |
| xlong-solo-S-s304 | 0.8109 | 1925 | 0.8115 | 1000 | 0.8128 | 925 | -0.23 | 0.0185 | 25 |
| xlong-solo-W-s304 | 0.8159 | 1875 | 0.8278 | 875 | 0.8249 | 575 | -1.09 | 0.0208 | 25 |
| xlong-solo-LS-s304 | 0.8771 | 1450 | 0.8807 | 350 | 0.8791 | 100 | -0.23 | 0.0106 | 25 |
| xlong-solo-LW-s304 | 0.8905 | 575 | 0.8905 | 575 | 0.8961 | 600 | -0.62 | 0.0115 | 25 |
| xlong-solo-SW-s304 | 0.8156 | 1025 | 0.8179 | 500 | 0.8162 | 750 | -0.07 | 0.0178 | 25 |
| xlong-solo-LSW-s304 | 0.8539 | 975 | 0.8539 | 975 | 0.8578 | 525 | -0.45 | 0.0372 | 25 |
| xlong-solo-N-s305 | 0.8110 | 1275 | 0.8170 | 750 | 0.8132 | 650 | -0.27 | 0.0150 | 25 |
| xlong-solo-L-s305 | 0.9205 | 1125 | 0.9220 | 900 | 0.9172 | 900 | +0.36 | 0.0128 | 25 |
| xlong-solo-S-s305 | 0.8113 | 1900 | 0.8143 | 700 | 0.8153 | 225 | -0.49 | 0.0301 | 50 |
| xlong-solo-W-s305 | 0.8114 | 1575 | 0.8117 | 725 | 0.8190 | 775 | -0.93 | 0.0163 | 50 |
| xlong-solo-LS-s305 | 0.8798 | 900 | 0.8798 | 900 | 0.8824 | 975 | -0.29 | 0.0145 | 25 |
| xlong-solo-LW-s305 | 0.8961 | 1800 | 0.8973 | 350 | 0.8932 | 650 | +0.32 | 0.0178 | 25 |
| xlong-solo-SW-s305 | 0.8146 | 1700 | 0.8231 | 450 | 0.8183 | 975 | -0.45 | 0.0184 | 25 |
| xlong-solo-LSW-s305 | 0.8548 | 1850 | 0.8584 | 625 | 0.8581 | 950 | -0.38 | 0.0168 | 25 |

Runs of 2000 updates whose best score came after update 1000: 26 of 40.
