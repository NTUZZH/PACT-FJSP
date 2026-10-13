r"""Per-regime view of the production-batch solver study.

For 50x25 and 80x25 (first 10 test instances per regime), reports the PACT
policy (arm jointblind-v1, per-instance mean over seeds 301/302/303), the
warm start, and warm-started CP-SAT at a fixed 1800 s budget credited with
min(warm start, best incumbent), plus the PACT lead with a paired-bootstrap
95% CI and the count of solves in which the search improved on its warm start.

Every convention is taken from scripts/p60_scale1800.py and not re-derived:
  lead  = mean over instances of 100*(credited - policy)/policy,
          positive = policy ahead (p39 anytime convention);
  CI    = percentile bootstrap, 20000 resamples, numpy default_rng(7),
          drawn in p60's exact order (per size: 8 regimes, then pooled), so
          the per-regime intervals are the ones p60 used for \ScXAhead and
          the pooled interval is p60's own;
  improved = objective < warm_makespan - 1e-9 (p60's "one ruler").

The pooled row per size is asserted, at printed precision, against
paper/macros_scale1800.tex; on any mismatch the script writes nothing and
exits with the discrepancy.

Outputs: paper/tables/prodbatch_table.tex, paper/macros_prodbatch.tex,
results/prodbatch_table.md.

Usage: python scripts/p70_prodbatch_table.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from macro_io import emit

# --- conventions copied verbatim from p60 ------------------------------------
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
POLICY = 'greedy+10x25+family+jointblind-v1'
SEED_SFX = ('', '-s302', '-s303')
SIZEW = {'50x25': 'Fifty', '80x25': 'Eighty'}
MODULES = {'50x25': 50, '80x25': 80}
N_BOOT = 20000
BUDGET = 1800
rng = np.random.default_rng(7)

REF_MACROS = 'paper/macros_scale1800.tex'
MACROS = 'paper/macros_prodbatch.tex'
TABLE = 'paper/tables/prodbatch_table.tex'
MD = 'results/prodbatch_table.md'

for size in SIZEW:
    n = len(glob.glob(f'or_solution/FAMILY/{size}_fairwarm1800s/*/*.json'))
    if n < 80:
        sys.exit(f'REFUSING: {size} at {n}/80 solves.')


def recs(size, r):
    return [json.load(open(f)) for f in
            sorted(glob.glob(f'or_solution/FAMILY/{size}_fairwarm1800s/{r}/*.json'))]


def credited_at(d, b):
    best = d['warm_makespan']
    for t, ub, _ in d.get('anytime', []):
        if t <= b and ub < best:
            best = ub
    return best


def policy_pooled(size, r):
    xs = []
    for s in SEED_SFX:
        p = f'test_results/FAMILY/{size}/Result_{POLICY}{s}_{r}.npy'
        if not os.path.exists(p):
            sys.exit(f'REFUSING: policy seeds missing at {size}/{r}')
        a = np.load(p)
        xs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return np.mean(xs, axis=0)


def ci(d):
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def improved(d):
    o, w = d['objective'], d['warm_makespan']
    return o is not None and o < w - 1e-9


# --- compute -----------------------------------------------------------------
rows = {}      # size -> list of row dicts (8 regimes + pooled)
for size in SIZEW:
    out, allg, allp, allw, allc, allimp, allt, allst = [], [], [], [], [], 0, [], []
    for r in REGIMES:
        data = recs(size, r)
        pol = policy_pooled(size, r)
        cred = np.array([credited_at(d, BUDGET) for d in data])
        warm = np.array([d['warm_makespan'] for d in data], dtype=float)
        n = min(len(cred), len(pol))
        g = 100 * (cred[:n] - pol[:n]) / pol[:n]
        m, lo, hi = ci(g)
        imp = sum(improved(d) for d in data[:n])
        out.append(dict(reg=r, n=n, pact=pol[:n].mean(), warm=warm[:n].mean(),
                        cred=cred[:n].mean(), lead=m, lo=lo, hi=hi, imp=imp))
        allg.append(g); allp.append(pol[:n]); allw.append(warm[:n]); allc.append(cred[:n])
        allimp += imp
        allt += [d['solve_time'] for d in data[:n]]
        allst += [d['status'] for d in data[:n]]
    m, lo, hi = ci(np.concatenate(allg))
    out.append(dict(reg='Pooled', n=sum(len(x) for x in allg),
                    pact=np.concatenate(allp).mean(), warm=np.concatenate(allw).mean(),
                    cred=np.concatenate(allc).mean(), lead=m, lo=lo, hi=hi, imp=allimp,
                    tmin=min(allt), tmax=max(allt), tmean=float(np.mean(allt)),
                    status=sorted(set(allst))))
    rows[size] = out

# --- reproduction gate against the published macros -------------------------
ref = dict(re.findall(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open(REF_MACROS).read()))
bad = []
for size in SIZEW:
    w, p = SIZEW[size], rows[size][-1]
    checks = {f'Sc{w}Pooled': f'{p["lead"]:+.2f}', f'Sc{w}PooledLo': f'{p["lo"]:+.2f}',
              f'Sc{w}PooledHi': f'{p["hi"]:+.2f}', f'Sc{w}EighteenHundred': f'{p["lead"]:+.2f}',
              f'Imp{w}N': f'{p["imp"]}',
              f'Sc{w}Ahead': f'{sum(x["lo"] > 0 for x in rows[size][:-1])}',
              f'Sc{w}Behind': f'{sum(x["hi"] < 0 for x in rows[size][:-1])}'}
    for k, v in checks.items():
        status = 'ok' if ref.get(k) == v else 'MISMATCH'
        print(f'  gate {k:24s} published {ref.get(k)!s:>7s}  recomputed {v:>7s}  {status}')
        if ref.get(k) != v:
            bad.append((k, ref.get(k), v))
if bad:
    sys.exit('REFUSING: pooled rows do not reproduce macros_scale1800.tex; nothing written.\n'
             + '\n'.join(f'  \\{k}: published {a}, recomputed {b}' for k, a, b in bad))
print('pooled rows reproduce paper/macros_scale1800.tex at printed precision')

# --- LaTeX table -------------------------------------------------------------
def f1(x):
    return f'{x:.1f}'


def sgn(x):
    return f'${x:+.2f}$'


def cellpair(x):
    """PACT and credited CP-SAT means; bold the lower at printed precision (ties: both)."""
    p, c = f1(x['pact']), f1(x['cred'])
    pb = f'\\textbf{{{p}}}' if float(p) <= float(c) else p
    cb = f'\\textbf{{{c}}}' if float(c) <= float(p) else c
    return pb, cb


def half(x):
    pb, cb = cellpair(x)
    return (f'{pb} & {cb} & ${x["lead"]:+.2f}$ $[{x["lo"]:+.2f},{x["hi"]:+.2f}]$ & '
            f'{x["imp"]}/{x["n"]}')


# Side-by-side layout (50 modules | 80 modules): half the height of a stacked
# table, so the float sits next to the paragraph that cites it. The warm-start
# mean is left to results/prodbatch_table.md; the "improved" column carries how
# often the search moved off it.
tex = ['% auto-generated by scripts/p70_prodbatch_table.py -- do not hand-edit',
       '% lead = mean per-instance 100*(credited CP-SAT - PACT)/PACT; positive = PACT ahead',
       '% (p60 convention). CP-SAT: warm-started, fixed 1800 s, credited with min(warm, incumbent).',
       '% tabular only: the caption and label live in the manuscript (paper_aei/sections/s7_results.tex)',
       '\\begin{tabular}{lcccccccc}',
       '\\toprule',
       f' & \\multicolumn{{4}}{{c}}{{{MODULES['50x25']} modules}} & \\multicolumn{{4}}{{c}}{{{MODULES['80x25']} modules}} \\\\',
       '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
       'regime & PACT (h) & CP-SAT (h) & lead (\\%) [95\\% CI] & improved '
       '& PACT (h) & CP-SAT (h) & lead (\\%) [95\\% CI] & improved \\\\',
       '\\midrule']
for a, b in zip(rows['50x25'][:-1], rows['80x25'][:-1]):
    tex.append(f'${a["reg"]}$ & {half(a)} & {half(b)} \\\\')
tex += ['\\midrule', f'pooled & {half(rows['50x25'][-1])} & {half(rows['80x25'][-1])} \\\\',
        '\\bottomrule', '\\end{tabular}']
os.makedirs(os.path.dirname(TABLE), exist_ok=True)
with open(TABLE, 'w') as fh:
    fh.write('\n'.join(tex) + '\n')
print(f'wrote {TABLE}')

# --- macros ------------------------------------------------------------------
mac = ['% auto-generated by scripts/p70_prodbatch_table.py -- do not hand-edit',
       '% Fixed 1800 s budget, first 10 test instances per regime; lead positive = PACT ahead.']
for size in SIZEW:
    w, reg = SIZEW[size], rows[size][:-1]
    lo_r = min(reg, key=lambda x: x['lead'])
    hi_r = max(reg, key=lambda x: x['lead'])
    mac += [(f'Pb{w}AheadN', sum(x['lo'] > 0 for x in reg), 'regimes of 8 with CI above zero'),
            (f'Pb{w}BehindN', sum(x['hi'] < 0 for x in reg), 'regimes of 8 with CI below zero'),
            (f'Pb{w}MinLead', f'{lo_r["lead"]:+.2f}'), (f'Pb{w}MinLeadRegime', lo_r['reg']),
            (f'Pb{w}MaxLead', f'{hi_r["lead"]:+.2f}'), (f'Pb{w}MaxLeadRegime', hi_r['reg'])]
    for x in rows[size]:
        k = f'Pb{w}{x["reg"]}'
        mac += [(f'{k}Pact', f1(x['pact'])), (f'{k}Warm', f1(x['warm'])),
                (f'{k}Cpsat', f1(x['cred'])), (f'{k}Lead', f'{x["lead"]:+.2f}'),
                (f'{k}Lo', f'{x["lo"]:+.2f}'), (f'{k}Hi', f'{x["hi"]:+.2f}'),
                (f'{k}Imp', x['imp']), (f'{k}Of', x['n'])]
emit(MACROS, mac)

# --- markdown ----------------------------------------------------------------
md = ['# Production-batch study, per regime',
      '',
      'Generated by `scripts/p70_prodbatch_table.py`. PACT (jointblind-v1, per-instance '
      'mean over seeds 301/302/303) against warm-started CP-SAT at a fixed 1800 s per '
      'instance, credited with min(warm start, best incumbent). First 10 test instances '
      'per regime. Lead = mean per-instance 100*(CP-SAT - PACT)/PACT, positive = PACT '
      'ahead; 95% percentile paired bootstrap (20000 resamples, p60 settings). Lower '
      'makespan is better; bold marks the lower of PACT and CP-SAT.', '']
for size in SIZEW:
    md += [f'## {MODULES[size]} modules, 25 stations', '',
           '| regime | PACT (h) | warm start (h) | CP-SAT (h) | lead (%) | 95% CI | improved |',
           '|---|---:|---:|---:|---:|---|---:|']
    for x in rows[size]:
        p, c = f1(x['pact']), f1(x['cred'])
        if float(p) <= float(c):
            p = f'**{p}**'
        if float(c) <= float(f1(x['pact'])):
            c = f'**{c}**'
        name = '**pooled**' if x['reg'] == 'Pooled' else x['reg']
        md.append(f'| {name} | {p} | {f1(x["warm"])} | {c} | {x["lead"]:+.2f} | '
                  f'[{x["lo"]:+.2f}, {x["hi"]:+.2f}] | {x["imp"]}/{x["n"]} |')
    md.append('')
md += ['## Notes', '']
for size in SIZEW:
    reg, p = rows[size][:-1], rows[size][-1]
    lo_r = min(reg, key=lambda x: x['lead'])
    ahead = sum(x['lo'] > 0 for x in reg)
    md.append(f'- {MODULES[size]} modules: weakest regime is {lo_r["reg"]} '
              f'(lead {lo_r["lead"]:+.2f}%, CI [{lo_r["lo"]:+.2f}, {lo_r["hi"]:+.2f}]); '
              f'the interval lies entirely above zero in {ahead} of 8 regimes. '
              f'All {p["n"]} solves ended with status {", ".join(p["status"])} after '
              f'{p["tmin"]:.0f} to {p["tmax"]:.0f} s wall-clock (mean {p["tmean"]:.0f} s), '
              f'so no status or time column is shown.')
os.makedirs(os.path.dirname(MD), exist_ok=True)
with open(MD, 'w') as fh:
    fh.write('\n'.join(md) + '\n')
print(f'wrote {MD}')
