r"""P81: the policy's schedule as CP-SAT's warm start, at the two production
batches (50 and 80 modules, first 10 instances per regime), in two variants:
the released checkpoint's schedule (folder suffix pactwarm, macro prefix Hyb,
tables/hybrid_released_table.tex) and the best of the three released
checkpoints' schedules per instance (pactbest3, prefix HybBest,
tables/hybrid_table.tex; p82_best3.py builds its warm-start cache).

The hybrid is the plant's natural escalation path: the policy returns a
validated schedule in under a second, and if the planner can wait, CP-SAT
continues from that schedule. Each hybrid solve ran 600 s with eight workers
and its incumbent trajectory recorded (scripts/p80_cpsat_rerun.py). The
rule-seeded comparator is the 1800 s campaign of scripts/p60_scale1800.py,
whose trajectories give its value at every smaller budget. Both are read with
p60's rule: credited(b) = min(warm start, best incumbent at or before b).

Quantities (all percent, mean of per-instance relative gaps, 95% percentile
paired bootstrap, 20,000 resamples, default_rng(7)):
  Imp      improvement of the hybrid over its own start, the released policy's
           schedule: 100*(start - credited(b))/start.
  VsRule   hybrid against the rule-seeded solver at the SAME budget b:
           100*(rule(b) - hybrid(b))/hybrid(b), positive = hybrid shorter.
  VsRuleLong  hybrid at 600 s against the rule-seeded solver at 1800 s.
Emits paper/macros_hybrid.tex and paper/tables/hybrid_table.tex. Refuses
unless every cell is present, re-solved with the complete hint, validated, and
valid under the timing rule of timing_validity.py.

Usage: python scripts/p81_hybrid.py [--variants pactwarm pactbest3]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from macro_io import emit  # noqa: E402
from timing_validity import classify_folders  # noqa: E402

R8 = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SIZES = {'50x25': 'Fifty', '80x25': 'Eighty'}
RUNGS = {1: 'One', 5: 'Five', 60: 'Sixty', 300: 'ThreeHundred', 600: 'SixHundred'}
OR = 'or_solution/FAMILY'
VALID = classify_folders(OR, [f'{s}_{k}' for s in ('50x25', '80x25')
                                for k in ('fairwarm600s_pactwarm', 'fairwarm600s_pactbest3',
                                          'fairwarm1800s')])
B = 20000


def _nz(x):
    v = round(float(x), 2)
    return 0.0 if v == 0 else v


def ci(x):
    rng = np.random.default_rng(7)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def credited_at(d, b):
    best = d['warm_makespan']
    for t, ub, _ in d.get('anytime', []):
        if t <= b and ub < best:
            best = ub
    return best


def load(folder, r, n=10, need_complete=False):
    fs = sorted(glob.glob(f'{OR}/{folder}/{r}/*.json'))[:n]
    if len(fs) < n:
        sys.exit(f'REFUSING: {folder}/{r} has {len(fs)} of {n} cells')
    out = []
    for f in fs:
        d = json.load(open(f))
        if not d.get('validated', False):
            sys.exit(f'REFUSING: {f} not validated')
        if 'anytime' not in d:
            sys.exit(f'REFUSING: {f} has no trajectory')
        if need_complete:
            if d.get('hint') != 'complete':
                sys.exit(f'REFUSING: {f} lacks the complete hint')
            ok, why = VALID.get(f, (False, 'not classified'))
            if not ok:
                sys.exit(f'REFUSING: {f} fails the timing rule ({why})')
        out.append(d)
    return out


def trio(name, m, lo, hi, cmt):
    return (f'\\newcommand{{\\{name}}}{{{_nz(m):+.2f}}}'
            f'\\newcommand{{\\{name}Abs}}{{{abs(_nz(m)):.2f}}}'
            f'\\newcommand{{\\{name}Lo}}{{{_nz(lo):+.2f}}}'
            f'\\newcommand{{\\{name}Hi}}{{{_nz(hi):+.2f}}}  % {cmt}')


VARIANTS = {
    # folder suffix: (macro prefix, table file, start label)
    'pactwarm': ('Hyb', 'paper/tables/hybrid_released_table.tex', 'default checkpoint'),
    'pactbest3': ('HybBest', 'paper/tables/hybrid_table.tex', 'best of three'),
}


def analyse(variant):
    P, table, start_label = VARIANTS[variant]
    lines, rows = [], {}
    for size, w in SIZES.items():
        hyb = {r: load(f'{size}_fairwarm600s_{variant}', r, need_complete=True) for r in R8}
        rule = {r: load(f'{size}_fairwarm1800s', r,
                        need_complete=r in ('N', 'L', 'W', 'LW')) for r in R8}
        for r in R8:
            for h, q in zip(hyb[r], rule[r]):
                if h['instance'] != q['instance']:
                    sys.exit('REFUSING: instance order differs')
        for b, bw in RUNGS.items():
            imp, vs, nimp = [], [], 0
            for r in R8:
                st = np.array([h['warm_makespan'] for h in hyb[r]])
                hb = np.array([credited_at(h, b) for h in hyb[r]])
                qb = np.array([credited_at(q, b) for q in rule[r]])
                imp.append(100 * (st - hb) / st)
                vs.append(100 * (qb - hb) / hb)
                nimp += int(np.sum(hb < st))
            m, lo, hi = ci(np.concatenate(imp))
            lines.append(trio(f'{P}{w}Imp{bw}', m, lo, hi,
                              f'{size}: hybrid improvement over its start ({start_label}) at {b} s (%)'))
            lines.append(f'\\newcommand{{\\{P}{w}ImprovedN{bw}}}{{{nimp}}}  % instances of 80 improved by {b} s')
            m, lo, hi = ci(np.concatenate(vs))
            lines.append(trio(f'{P}{w}VsRule{bw}', m, lo, hi,
                              f'{size}: rule-seeded minus hybrid at the same {b} s budget, relative to hybrid (%, + = hybrid shorter)'))
            print(f'{variant} {size} {b:4d}s  imp {np.concatenate(imp).mean():+.2f}  improved {nimp}/80  '
                  f'vs rule same budget {np.concatenate(vs).mean():+.2f}')
        # hybrid at 600 s against the rule-seeded solver at 1800 s, per regime and pooled
        allg = []
        rows[size] = []
        for r in R8:
            st = np.array([h['warm_makespan'] for h in hyb[r]])
            h6 = np.array([credited_at(h, 600) for h in hyb[r]])
            q18 = np.array([credited_at(q, 1800) for q in rule[r]])
            g = 100 * (q18 - h6) / h6
            allg.append(g)
            m, lo, hi = ci(g)
            lines.append(trio(f'{P}{w}Long{r}', m, lo, hi,
                              f'{size} {r}: rule-seeded at 1800 s minus hybrid at 600 s (%, + = hybrid shorter)'))
            rows[size].append((r, st.mean(), h6.mean(), q18.mean(), m, lo, hi))
        m, lo, hi = ci(np.concatenate(allg))
        lines.append(trio(f'{P}{w}VsRuleLong', m, lo, hi,
                          f'{size}: rule-seeded at 1800 s minus hybrid at 600 s, pooled (%, + = hybrid shorter)'))
        ahead = sum(1 for g in allg if ci(g)[1] > 0)
        behind = sum(1 for g in allg if ci(g)[2] < 0)
        lines.append(f'\\newcommand{{\\{P}{w}LongAheadN}}{{{ahead}}}\\newcommand{{\\{P}{w}LongBehindN}}{{{behind}}}'
                     f'  % regimes, interval clear of zero')
        print(f'{variant} {size} hybrid@600 vs rule@1800 pooled {m:+.2f} [{lo:+.2f},{hi:+.2f}] '
              f'ahead {ahead} behind {behind}')

    with open(table, 'w') as f:
        f.write('% auto-generated by scripts/p81_hybrid.py -- do not hand-edit\n')
        f.write('\\begin{tabular}{l' + 'rrrr@{\\,}l' * 2 + '}\n\\toprule\n')
        f.write(' & ' + ' & '.join(f'\\multicolumn{{5}}{{c}}{{{s[:2]} modules}}' for s in SIZES) + '\\\\\n')
        f.write('\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}\n')
        hdr = (f'{start_label} & hybrid 600\\,s & rule-seeded 1800\\,s & '
               '\\multicolumn{2}{c}{lead (\\%)}')
        f.write('regime & ' + hdr + ' & ' + hdr + '\\\\\n\\midrule\n')
        for i, r in enumerate(R8):
            cells = []
            for size in SIZES:
                _, st, h6, q18, m, lo, hi = rows[size][i]
                best = min(st, h6, q18)
                fmt = lambda v: f'\\textbf{{{v:.1f}}}' if abs(v - best) < 1e-9 else f'{v:.1f}'
                cells.append(f'{fmt(st)} & {fmt(h6)} & {fmt(q18)} & ${_nz(m):+.2f}$ & '
                             f'\\footnotesize{{$[{_nz(lo):+.2f},\\,{_nz(hi):+.2f}]$}}')
            f.write(f'${r}$ & ' + ' & '.join(cells) + '\\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')
    print(f'wrote {table}')
    return lines


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    a = ap.parse_args()
    out = []
    for v in a.variants:
        out += analyse(v)
    emit('paper/macros_hybrid.tex', out, header='Generated by scripts/p81_hybrid.py')
