# E5: mean makespan per regime (fair-warm, incumbent-retaining CP-SAT)

CP columns: best-of-four-PDR warm start, 8 workers on dedicated cores,
value = min(warm start, search). "+" = the search improved on the warm
start on ZERO instances, so the cell is the best dispatching rule.

| regime | greedy | best single PDR | PDR portfolio | GA-30s | CP 5s | CP 30s | CP 300s | opt | n(300s) |
|---|---|---|---|---|---|---|---|---|---|
| N | 130.1 | 132.6 (SPT) | 132.6 | 132.1 | 127.5 | 124.4 | 121.0 | 0 | 30 |
| L | 211.9 | 216.3 (SPT) | 215.8 | 214.2 | 204.2 | 203.5 | 202.5 | 26 | 30 |
| S | 194.2 | 195.6 (SPT) | 195.6 | 192.8 | 195.4+ | 191.2 | 188.2 | 0 | 30 |
| W | 147.1 | 151.0 (SPT) | 151.0 | 150.3 | 142.5 | 139.4 | 133.5 | 0 | 30 |
| LS | 269.8 | 275.3 (SPT) | 275.2 | 272.0 | 273.1+ | 269.0 | 265.5 | 0 | 30 |
| LW | 221.3 | 227.5 (SPT) | 226.7 | 226.7 | 212.8 | 209.9 | 208.0 | 16 | 30 |
| SW | 218.6 | 222.2 (SPT) | 222.2 | 220.2 | 221.2+ | 217.5 | 215.9 | 0 | 30 |
| LSW | 286.7 | 292.6 (SPT) | 292.6 | 290.5 | 290.0+ | 286.7 | 284.2 | 0 | 30 |

## LaTeX body

```
N & \textbf{130.1} & 132.6 & 132.6 & 132.1 & 127.5 & \textbf{124.4} & \textit{121.0} & \textit{0} \\
L & \textbf{211.9} & 216.3 & 215.8 & 214.2 & 204.2 & \textbf{203.5} & \textit{202.5} & \textit{26} \\
S & \textbf{194.2} & 195.6 & 195.6 & 192.8 & 195.4$^\dagger$ & \textbf{191.2} & \textit{188.2} & \textit{0} \\
W & \textbf{147.1} & 151.0 & 151.0 & 150.3 & 142.5 & \textbf{139.4} & \textit{133.5} & \textit{0} \\
LS & \textbf{269.8} & 275.3 & 275.2 & 272.0 & 273.1$^\dagger$ & \textbf{269.0} & \textit{265.5} & \textit{0} \\
LW & \textbf{221.3} & 227.5 & 226.7 & 226.7 & 212.8 & \textbf{209.9} & \textit{208.0} & \textit{16} \\
\midrule
SW & \textbf{218.6} & 222.2 & 222.2 & 220.2 & 221.2$^\dagger$ & \textbf{217.5} & \textit{215.9} & \textit{0} \\
LSW & \textbf{286.7} & 292.6 & 292.6 & 290.5 & 290.0$^\dagger$ & \textbf{286.7} & \textit{284.2} & \textit{0} \\
```
