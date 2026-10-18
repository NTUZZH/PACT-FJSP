r"""P82: the best of the three released checkpoints, per instance.

The three training seeds of the released architecture (jointblind-v1, -s302,
-s303) are all part of the release. A plant can run all three greedy rollouts
on one instance and keep the shortest schedule; each rollout is a validated
schedule, so the selection uses only the observable makespan. This option was
identified after the scale results were in (post hoc), and the manuscript says so.

  --cache    builds or_solution/FAMILY/pactbest3_cache_<size>/<regime>/<inst>.npz
             at 50x25 and 80x25 (first 10 instances per regime) from the three
             per-seed schedule exports, after asserting that every exported
             makespan equals the published per-seed result.
  --analyse  writes paper/macros_best3.tex: the best of three against the
             fair-warm CP-SAT 1 s rung at 10x25 (p62's credit rule, pooled over
             the six training regimes and per regime), against the 1800 s
             rule-seeded CP-SAT at 50x25 and 80x25 (p60's credit rule, pooled
             over all eight regimes), each training run alone against
             the same 1800 s solver, and the gain of the best of three over the
             seed mean at every size. Gaps are 100*(rival - option)/option,
             positive = the option is ahead; paired bootstrap, 20,000
             resamples, default_rng(7); a regime is ahead or behind when its
             95% interval excludes zero. Each training run is also compared
             on its own (Run<One,Two,Three>, seeds 301-303).

Usage:
  python scripts/p82_best3.py --cache
  python scripts/p82_best3.py --analyse
"""
import argparse
import glob
import os
import sys

import numpy as np

OR = 'or_solution/FAMILY'
P = 'greedy+10x25+family+jointblind-v1'
SEEDS = ('', '-s302', '-s303')
SEED_CACHE = {'': 'pactwarm_cache_{size}', '-s302': 'pactwarm_cache_s302_{size}',
              '-s303': 'pactwarm_cache_s303_{size}'}
R8 = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
LIMIT = 10


def published(size, seed, r):
    a = np.load(f'test_results/FAMILY/{size}/Result_{P}{seed}_{r}.npy')
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def build_cache():
    for size in ('50x25', '80x25'):
        picks = {0: 0, 1: 0, 2: 0}
        for r in R8:
            names = sorted(os.path.basename(f)[:-4] for f in
                           glob.glob(f'{OR}/{SEED_CACHE[""].format(size=size)}/{r}/*.npz'))[:LIMIT]
            if len(names) < LIMIT:
                sys.exit(f'REFUSING: {size}/{r} has {len(names)} released-seed schedules')
            out_dir = f'{OR}/pactbest3_cache_{size}/{r}'
            os.makedirs(out_dir, exist_ok=True)
            for i, name in enumerate(names):
                zs = []
                for k, seed in enumerate(SEEDS):
                    p = f'{OR}/{SEED_CACHE[seed].format(size=size)}/{r}/{name}.npz'
                    if not os.path.exists(p):
                        sys.exit(f'REFUSING: missing {p}')
                    z = dict(np.load(p, allow_pickle=True))
                    pub = published(size, seed, r)[i]
                    if abs(float(z['ms']) - pub) > 1e-6:
                        sys.exit(f'REFUSING: {p} makespan {float(z["ms"])} != published {pub}')
                    zs.append(z)
                k = int(np.argmin([float(z['ms']) for z in zs]))   # ties go to the released seed
                picks[k] += 1
                best = dict(zs[k])
                best['seed'] = np.array(['s301', 's302', 's303'][k])
                np.savez(f'{out_dir}/{name}.npz', **best)
        print(f'{size}: best-of-3 cache written; picks s301/s302/s303 = '
              f'{picks[0]}/{picks[1]}/{picks[2]}')


TRAIN = ['N', 'L', 'S', 'W', 'LS', 'LW']
WORD = {'10x25': 'Ten', '20x25': 'Twenty', '30x25': 'Thirty', '50x25': 'Fifty', '80x25': 'Eighty'}
N_BOOT = 20000


def seeds(size, r, n=None):
    arr = np.array([published(size, s, r) for s in SEEDS])
    return arr if n is None else arr[:, :n]


def ci(d, rng):
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def cp_one_10x25(r):
    """p62.cp(r, 1): min(repaired hint, min(search, original hint)), instance order."""
    import json
    rep = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))
    vals = []
    for f in sorted(glob.glob(f'{OR}/10x25_fairwarm1s/{r}/*.json')):
        d = json.load(open(f))
        vals.append(min(rep[r][d['instance']]['repaired_ms'], min(d['objective'], d['warm_makespan'])))
    return np.array(vals)


def credited_at(d, b):
    """p60.credited_at: min(warm, best incumbent at or before b)."""
    best = d['warm_makespan']
    for t, ub, _ in d.get('anytime', []):
        if t <= b and ub < best:
            best = ub
    return best


def cp_1800(size, r):
    import json
    fs = sorted(glob.glob(f'{OR}/{size}_fairwarm1800s/{r}/*.json'))
    return np.array([credited_at(json.load(open(f)), 1800) for f in fs])


def trio(name, m, lo, hi):
    return (f'\\newcommand{{\\{name}}}{{{m:+.2f}}}\\newcommand{{\\{name}Abs}}{{{abs(m):.2f}}}'
            f'\\newcommand{{\\{name}Lo}}{{{lo:+.2f}}}\\newcommand{{\\{name}Hi}}{{{hi:+.2f}}}')


def analyse():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from macro_io import emit
    rng = np.random.default_rng(7)
    lines = ['% auto-generated by scripts/p82_best3.py --analyse -- do not hand-edit',
             '% Best of the three released checkpoints per instance (post hoc option).',
             '% Gaps: 100*(rival - option)/option, positive = the option is ahead.']
    # gain of the best of three over the seed mean, every size, all eight regimes
    for size, w in WORD.items():
        g = np.concatenate([100 * (seeds(size, r).mean(0) - seeds(size, r).min(0)) / seeds(size, r).min(0)
                            for r in R8])
        lines.append(f'\\newcommand{{\\BestThreeGain{w}}}{{{g.mean():.2f}}}')
    # 10x25: against the 1 s CP-SAT rung, training regimes pooled, and per regime
    gaps, ahead, behind, per = [], [], [], {}
    for r in R8:
        best = seeds('10x25', r).min(0)
        c = cp_one_10x25(r)
        if len(c) != len(best):
            sys.exit(f'REFUSING: 10x25/{r} CP-SAT 1 s has {len(c)} of {len(best)} cells')
        g = 100 * (c - best) / best
        m, lo, hi = ci(g, rng)
        per[r] = m
        (ahead if lo > 0 else behind if hi < 0 else []).append(r)
        if r in TRAIN:
            gaps.append(g)
    m, lo, hi = ci(np.concatenate(gaps), rng)
    lines.append(trio('BestThreeTenCpOne', m, lo, hi))
    lines.append(f'\\newcommand{{\\BestThreeTenCpOneAheadN}}{{{len(ahead)}}}'
                 f'\\newcommand{{\\BestThreeTenCpOneBehindN}}{{{len(behind)}}}'
                 f'  % of 8 regimes; behind: {",".join(behind) or "none"}')
    # 50x25 / 80x25: against the 1800 s rule-seeded CP-SAT, all eight regimes pooled
    for size in ('50x25', '80x25'):
        w = WORD[size]
        for tag, pick in (('BestThree', lambda a: a.min(0)),):
            gaps, ahead, behind = [], [], []
            for r in R8:
                c = cp_1800(size, r)
                x = pick(seeds(size, r, len(c)))
                g = 100 * (c - x) / x
                m, lo, hi = ci(g, rng)
                (ahead if lo > 0 else behind if hi < 0 else []).append(r)
                gaps.append(g)
            m, lo, hi = ci(np.concatenate(gaps), rng)
            lines.append(trio(f'{tag}{w}Cp', m, lo, hi))
            lines.append(f'\\newcommand{{\\{tag}{w}CpAheadN}}{{{len(ahead)}}}'
                         f'\\newcommand{{\\{tag}{w}CpBehindN}}{{{len(behind)}}}'
                         f'  % behind: {",".join(behind) or "none"}')
    # each training run on its own against the same 1800 s solver
    for size in ('50x25', '80x25'):
        w = WORD[size]
        means, clear, tie = [], 0, 0
        for k, word in enumerate(('One', 'Two', 'Three')):
            gaps, ahead, behind = [], [], []
            for r in R8:
                c = cp_1800(size, r)
                x = seeds(size, r, len(c))[k]
                g = 100 * (c - x) / x
                m, lo, hi = ci(g, rng)
                (ahead if lo > 0 else behind if hi < 0 else []).append(r)
                gaps.append(g)
            m, lo, hi = ci(np.concatenate(gaps), rng)
            lines.append(trio(f'Run{word}{w}Cp', m, lo, hi))
            lines.append(f'\\newcommand{{\\Run{word}{w}CpAheadN}}{{{len(ahead)}}}'
                         f'\\newcommand{{\\Run{word}{w}CpBehindN}}{{{len(behind)}}}'
                         f'  % seed {300 + k + 1}; behind: {",".join(behind) or "none"}')
            means.append(m)
            clear += lo > 0
            tie += lo <= 0 <= hi
        lines.append(f'\\newcommand{{\\Runs{w}CpMinAbs}}{{{min(means):.2f}}}'
                     f'\\newcommand{{\\Runs{w}CpMaxAbs}}{{{max(means):.2f}}}'
                     f'\\newcommand{{\\Runs{w}CpClearN}}{{{clear}}}'
                     f'\\newcommand{{\\Runs{w}CpTieN}}{{{tie}}}'
                     f'  % of the three runs: pooled interval above zero / including zero')
    emit('paper/macros_best3.tex', lines)
    print('\n'.join(lines))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='store_true')
    ap.add_argument('--analyse', action='store_true')
    a = ap.parse_args()
    if a.cache:
        build_cache()
    if a.analyse:
        analyse()
