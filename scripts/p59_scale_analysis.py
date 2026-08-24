r"""The 1800-second campaign, analysed under the rules agreed before it finished.

The rules, each argued out while the queue ran and recorded in
FINALISATION_CHECKLIST.md before any number existed to tempt them:

  ONE RULER.   Every cross-size statement uses "did the search improve on its
               own warm start", never mixed with "did the solver beat the
               policy" (different numerator and denominator).
  EQUAL n.     Group ceilings take max over ALL solves with unchanged = 0%,
               so setup-bearing and non-setup compare at 40 against 40.
  QUALIFIER.   Budget is constant while size varies, so every cross-size
               sentence carries "under a fixed 1800 s budget".
  DISCLOSURE.  Counts are reported as they are, then how many sit at the
               one-hour minimum resolution. No threshold: one chosen after
               seeing the distribution cannot be defended.
  NO PARTIAL.  This script refuses to run unless both cells are at 80.

Usage: python scripts/p59_scale_analysis.py
"""
import json
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SETUP = {'S', 'LS', 'SW', 'LSW'}
POLICY = 'greedy+10x25+family+jointblind-v1'
SEED_SFX = ('', '-s302', '-s303')
N_BOOT = 20000
rng = np.random.default_rng(7)

# ---- refuse partial cells ---------------------------------------------------
for size in ('50x25', '80x25'):
    n = len(glob.glob(f'or_solution/FAMILY/{size}_fairwarm1800s/*/*.json'))
    if n < 80:
        sys.exit(f'REFUSING: {size} at {n}/80. A maximum computed from a '
                 f'partial cell understates the true ceiling.')

def solves(size, budget_dir, regime):
    """[(improvement %, absolute hours, warm, credited)] with unchanged = 0."""
    out = []
    for f in sorted(glob.glob(f'or_solution/FAMILY/{budget_dir}/{regime}/*.json')):
        d = json.load(open(f))
        o, w = d['objective'], d['warm_makespan']
        cred = min(o, w) if o is not None else w
        if o is not None and o < w - 1e-9:
            out.append((100 * (w - o) / w, w - o, w, cred))
        else:
            out.append((0.0, 0.0, w, cred))
    return out

def policy_pooled(size, regime):
    xs = []
    for s in SEED_SFX:
        p = f'test_results/FAMILY/{size}/Result_{POLICY}{s}_{regime}.npy'
        if not os.path.exists(p):
            return None
        a = np.load(p)
        xs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return np.mean(xs, axis=0)

def ci(d):
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

print('=' * 74)
print('1. SEARCH vs ITS OWN WARM START, per regime  (the one ruler)')
print('=' * 74)
tab = {}
for size in ('50x25', '80x25'):
    print(f'\n{size} @ 1800 s:')
    print(f'  {"regime":7s}{"improved":>9s}{"at 1 h min":>11s}{"max %":>8s}')
    for r in REGIMES:
        ss = solves(size, f'{size}_fairwarm1800s', r)
        imp = [s for s in ss if s[0] > 0]
        min_res = sum(1 for s in imp if s[1] <= 1.0)
        tab[(size, r)] = (len(imp), len(ss), min_res, max(s[0] for s in ss))
        print(f'  {r:7s}{len(imp):>6d}/10{min_res:>11d}{max(s[0] for s in ss):>8.2f}')

print()
print('=' * 74)
print('2. GROUP CEILINGS, equal samples (max over ALL solves, unchanged = 0)')
print('=' * 74)
for size in ('50x25', '80x25'):
    for name, grp in (('setup-bearing (S/LS/SW/LSW)', SETUP),
                      ('non-setup    (N/L/W/LW)   ', set(REGIMES) - SETUP)):
        vals, nimp, nmin = [], 0, 0
        for r in grp:
            ss = solves(size, f'{size}_fairwarm1800s', r)
            vals += [s[0] for s in ss]
            nimp += sum(1 for s in ss if s[0] > 0)
            nmin += sum(1 for s in ss if 0 < s[0] and s[1] <= 1.0)
        print(f'  {size} {name}: n={len(vals)}  improved {nimp}  '
              f'at-1h-min {nmin}  MAX {max(vals):.2f}%')

print()
print('=' * 74)
print('3. THE SAME RULER ACROSS SIZES  (s/op = budget / operations)')
print('=' * 74)
OPS = {'10x25': 220, '50x25': 1100, '80x25': 1760}
rows = [('10x25', '10x25_fairwarm30s', 30), ('10x25', '10x25_fairwarm300s', 300),
        ('50x25', '50x25_fairwarm1800s', 1800), ('80x25', '80x25_fairwarm1800s', 1800)]
for size, d, b in rows:
    vals = []
    for r in REGIMES:
        vals += [s[0] for s in solves(size, d, r)]
    if not vals:
        continue
    n = sum(1 for v in vals if v > 0)
    print(f'  {size} @ {b:>4d}s  {b/OPS[size]:5.2f} s/op   improved '
          f'{n:>3d}/{len(vals):<3d} ({100*n/len(vals):3.0f}%)   max {max(vals):5.2f}%')

print()
print('=' * 74)
print('4. POLICY vs CREDITED SOLVER  (min(search, warm), paired, 3 seeds pooled)')
print('=' * 74)
for size in ('50x25', '80x25'):
    gaps = []
    per = {}
    for r in REGIMES:
        pol = policy_pooled(size, r)
        cred = np.array([s[3] for s in solves(size, f'{size}_fairwarm1800s', r)])
        if pol is None:
            sys.exit(f'REFUSING: policy seeds missing at {size}/{r}')
        n = min(len(pol), len(cred))
        g = 100 * (cred[:n] - pol[:n]) / pol[:n]     # + = policy ahead
        per[r] = ci(g)
        gaps.append(g)
    m, lo, hi = ci(np.concatenate(gaps))
    ahead = sum(1 for r in REGIMES if per[r][1] > 0)
    behind = sum(1 for r in REGIMES if per[r][2] < 0)
    print(f'\n  {size}: pooled {m:+.2f}%  CI [{lo:+.2f},{hi:+.2f}]   '
          f'per-regime ahead {ahead}/8, behind {behind}/8')
    for r in REGIMES:
        m2, lo2, hi2 = per[r]
        v = 'policy ahead' if lo2 > 0 else ('solver ahead' if hi2 < 0 else 'not separable')
        print(f'    {r:4s} {m2:+7.2f}%  [{lo2:+6.2f},{hi2:+6.2f}]  {v}')

print()
print('=' * 74)
print('5. cpu_per_wall audit (recorded on solves after the field was added)')
print('=' * 74)
vals = []
for size in ('50x25', '80x25'):
    for f in glob.glob(f'or_solution/FAMILY/{size}_fairwarm1800s/*/*.json'):
        d = json.load(open(f))
        if 'cpu_per_wall' in d:
            vals.append(d['cpu_per_wall'])
if vals:
    a = np.array(vals)
    print(f'  {len(a)} solves carry the field: min {a.min():.2f}, '
          f'median {np.median(a):.2f}, max {a.max():.2f}  (expected ~8.0)')
    low = (a < 7.0).sum()
    print(f'  below 7.0: {low}  ' + ('<- investigate' if low else '(none: every budget honoured)'))
else:
    print('  no solves carry the field')
