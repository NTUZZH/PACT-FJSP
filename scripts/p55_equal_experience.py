r"""E1's budget question: is the comparison with the specialists fair, and what
happens when it is made exactly fair?

Three counts run through this paper and only the first governs experience: the
policy TRAINS on six regimes, everything is EVALUATED on eight, and the
catalogue it replaces holds seven specialists. A PPO update deals ENVS parallel
environments. A specialist puts all of them on its own regime; the joint policy
spreads them over its six, so one regime receives ENVS/6 per update. Experience
on a regime is therefore

    updates x (ENVS if specialist else ENVS/6)

and the two sides are equal when updates_joint = 6 x updates_specialist. Against
the published 1000-update specialists that is 6000 updates, which is the arm
this script reports next to the released 2000-update policy.

Note the direction of the remaining asymmetry at that point: the catalogue costs
7 x 1000 = 7000 updates against the joint policy's 6000, so the equal-experience
arm is also the cheaper one in total. Nothing here is tilted toward the policy.

Conventions are p11_equivalence.py's: mean per-instance relative makespan gap,
paired over instances, 20k-resample bootstrap on rng seed 7, and both sides
pooled over the SAME seeds or the cell is refused. The superiority verdicts on
the six training regimes are one family and are Holm-adjusted across it.

Usage: python scripts/p55_equal_experience.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from macro_io import emit, refuse

RES = 'test_results/FAMILY/10x25'
TRAIN = ['N', 'L', 'S', 'W', 'LS', 'LW']
HELD = ['SW', 'LSW']
SEED_SFX = ('-s301', '-s302', '-s303')
RELEASED = 'greedy+10x25+family+jointblind-v1'   # 2000 updates, seed suffixes differ
EQUAL = 'greedy+10x25+family+jointblind-eq'      # 6000 updates
ENVS = 20
NTRAIN = 6
SPEC_UPDATES = 1000
EQ_UPDATES = 6000
REL_UPDATES = 2000
NSPEC = 7
MARGIN = 1.0
ALPHA = 0.05
N_BOOT = 20000
MACROS = 'paper/macros_eqexp.tex'
DATA = 'results/equal_experience.json'

rng = np.random.default_rng(7)


def arr(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def pooled(base, r, released=False):
    """Per-instance mean over the seed replicates that exist.

    The released policy's three runs are named jointblind-v1, -v1-s302 and
    -v1-s303; the equal-experience arm names its seeds uniformly. Both resolve
    here so the caller never has to know."""
    if released:
        tags = [base, f'{base}-s302', f'{base}-s303']
    else:
        tags = [f'{base}{s}' for s in SEED_SFX]
    xs = [arr(t, r) for t in tags]
    xs = [x for x in xs if x is not None]
    return (np.mean(xs, axis=0), len(xs)) if xs else (None, 0)


def ci(d):
    n = len(d)
    bs = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2.0 * (1 + min(int((bs <= 0).sum()), int((bs >= 0).sum()))) / (N_BOOT + 1)
    return float(d.mean()), float(lo), float(hi), min(1.0, p)


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def compare(base, released, regimes):
    """One arm against the specialists, regime by regime, plus the pooled gap."""
    cells, gaps = [], []
    for r in regimes:
        a, na = pooled(base, r, released)
        b, nb = pooled('greedy+10x25+family+spec-' + r, r, released=True)
        if a is None or b is None:
            return None, f'missing data on {r} ({na} vs {nb} seeds)'
        if na != nb:
            return None, f'asymmetric seed pools on {r}: {na} vs {nb}'
        g = 100 * (a - b) / b
        m, lo, hi, p = ci(g)
        cells.append(dict(regime=r, mean=m, lo=lo, hi=hi, p=p, seeds=na))
        gaps.append(g)
    pm, plo, phi, _ = ci(np.concatenate(gaps))
    return dict(cells=cells, pooled=dict(mean=pm, lo=plo, hi=phi)), None


arms = {}
for name, base, released in (('released', RELEASED, True),
                             ('equal', EQUAL, False)):
    got, why = compare(base, released, TRAIN)
    if got is None:
        refuse(MACROS, f'{name} arm: {why}')
        sys.exit(1)
    arms[name] = got

# The equal-experience arm's six training-regime verdicts are the discovery
# family; the released arm's are E1's and already carry their own correction.
eq = arms['equal']['cells']
for c, q in zip(eq, holm([c['p'] for c in eq])):
    c['p_holm'] = q
    if c['hi'] < 0 and q < ALPHA:
        c['verdict'] = 'policy'
    elif c['lo'] > 0 and q < ALPHA:
        c['verdict'] = 'specialist'
    elif -MARGIN < c['lo'] and c['hi'] < MARGIN:
        c['verdict'] = 'equivalent'
    else:
        c['verdict'] = 'inconclusive'

print(f'{"arm":10s} {"regime":7s} {"gap":>7s}  {"CI":>16s}  verdict')
for name in ('released', 'equal'):
    for c in arms[name]['cells']:
        v = c.get('verdict', '(E1)')
        print(f'{name:10s} {c["regime"]:7s} {c["mean"]:+7.2f}  '
              f'[{c["lo"]:+6.2f},{c["hi"]:+6.2f}]  {v}')
    p = arms[name]['pooled']
    print(f'  -> {name:8s} pooled over {len(TRAIN)} trained regimes: '
          f'{p["mean"]:+.2f}% [{p["lo"]:+.2f},{p["hi"]:+.2f}]\n')

nwin = sum(c['verdict'] == 'policy' for c in eq)
nlose = sum(c['verdict'] == 'specialist' for c in eq)
spec_exp = SPEC_UPDATES * ENVS
eq_exp = int(round(EQ_UPDATES * ENVS / NTRAIN))
rel_exp = int(round(REL_UPDATES * ENVS / NTRAIN))

lines = [
    '% auto-generated by scripts/p55_equal_experience.py -- do not hand-edit',
    '% Negative = the joint policy finishes earlier than the specialist.',
    # the protocol arithmetic, emitted here so the paper and this script cannot
    # disagree about what "equal experience" means
    f'\\newcommand{{\\ExpEnvs}}{{{ENVS}}}',
    f'\\newcommand{{\\ExpTrainRegimes}}{{{NTRAIN}}}',
    # every count above a thousand carries its separator, so the table never
    # prints "7,000" next to "6000"
    f'\\newcommand{{\\ExpSpecUpdates}}{{{SPEC_UPDATES:,}}}',
    f'\\newcommand{{\\ExpEqUpdates}}{{{EQ_UPDATES:,}}}',
    f'\\newcommand{{\\ExpRelUpdates}}{{{REL_UPDATES:,}}}',
    f'\\newcommand{{\\ExpSpecPerRegime}}{{{spec_exp:,}}}',
    f'\\newcommand{{\\ExpEqPerRegime}}{{{eq_exp:,}}}',
    f'\\newcommand{{\\ExpRelPerRegime}}{{{rel_exp:,}}}',
    f'\\newcommand{{\\ExpRelRatio}}{{{spec_exp / rel_exp:.1f}}}',
    f'\\newcommand{{\\ExpCatalogueUpdates}}{{{NSPEC * SPEC_UPDATES:,}}}',
    f'\\newcommand{{\\ExpEqCheaperPct}}{{'
    f'{100 * (NSPEC * SPEC_UPDATES - EQ_UPDATES) / (NSPEC * SPEC_UPDATES):.0f}}}',
    # the results
    f'\\newcommand{{\\EqExpWin}}{{{nwin}}}\\newcommand{{\\EqExpLose}}{{{nlose}}}',
    # NOT \EqExpN: the per-regime macros are \EqExp<REGIME>, and regime N would
    # collide with a count named that way. A LaTeX \newcommand collision is a
    # build error here, but the same shape once silently changed which number
    # the paper printed depending on \input order (see macros.tex, E2 note).
    f'\\newcommand{{\\EqExpRegimes}}{{{len(TRAIN)}}}',
]
for name, key in (('released', 'Rel'), ('equal', 'Eq')):
    p = arms[name]['pooled']
    lines.append(f'\\newcommand{{\\{key}Pooled}}{{{p["mean"]:+.2f}}}'
                 f'\\newcommand{{\\{key}PooledAbs}}{{{abs(p["mean"]):.2f}}}'
                 f'\\newcommand{{\\{key}PooledLo}}{{{p["lo"]:+.2f}}}'
                 f'\\newcommand{{\\{key}PooledHi}}{{{p["hi"]:+.2f}}}')
for c in eq:
    lines.append(f'\\newcommand{{\\EqExp{c["regime"]}}}{{{c["mean"]:+.2f}}}'
                 f'\\newcommand{{\\EqExp{c["regime"]}Lo}}{{{c["lo"]:+.2f}}}'
                 f'\\newcommand{{\\EqExp{c["regime"]}Hi}}{{{c["hi"]:+.2f}}}')
emit(MACROS, lines)

print(f'equal-experience arm: policy ahead on {nwin} of {len(TRAIN)} trained '
      f'regimes, specialist ahead on {nlose}')
print(f'experience per regime: specialist {spec_exp:,}, equal arm {eq_exp:,}, '
      f'released {rel_exp:,}')
print(f'total updates: catalogue {NSPEC * SPEC_UPDATES:,}, equal arm '
      f'{EQ_UPDATES:,} ({100 * (NSPEC * SPEC_UPDATES - EQ_UPDATES) / (NSPEC * SPEC_UPDATES):.0f}% less)')

os.makedirs('results', exist_ok=True)
with open(DATA, 'w') as f:
    json.dump(dict(arms=arms, train=TRAIN, held=HELD, envs=ENVS,
                   spec_updates=SPEC_UPDATES, eq_updates=EQ_UPDATES,
                   rel_updates=REL_UPDATES, margin=MARGIN, alpha=ALPHA), f,
              indent=1)
print(f'wrote {DATA}')
