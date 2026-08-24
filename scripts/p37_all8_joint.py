"""P37: the all-eight-regime joint policy -- the comparator S-XXI conceded it lacked.

WHY. For a family of three members, a plant could enumerate all eight regimes and
train one joint policy on the lot: also one network, also no retraining for any
known combination. The paper defended zero-shot composition only by a scaling
argument. This script costs that missing comparator, in both of the forms a fair
reading demands:

  matched compute   joint-all8-s*   same 2000 updates as joint-v1, regime set 6->8,
                                    so each regime gets LESS experience (dilution arm)
  matched exposure  joint-all8e-s*  2667 updates (= 2000*8/6), so each of the 8
                                    regimes gets the SAME per-regime experience the
                                    6-regime joint gave its six (+33% updates)

Everything else is identical to joint-v1 and ASSERTED below, not assumed: the
regime set, the update budget, the FiLM/type-embedding flags, the seed, and the
checkpoint parameter count (71,838). Both arms and joint-v1 are evaluated by the
same eval_family.py command on the same machine into the same campaign directory,
which is the paper's requirement for comparing greedy rollouts at all.

Estimator: identical to p11_equivalence.py. Per-instance PAIRED relative gap (%),
both sides pooled over seeds 301/302/303, paired-bootstrap 95% CI (B=20000, rng 7).
Sign convention: negative = the all-8 arm is BETTER than joint-v1.

Compute cost is reported in UPDATES (2000 vs 2667, exact), never in hours: the
all-8 runs shared the GPU with another user's job by consent, so their wall-clock
is contended and is not a measurement.

Emits paper/macros_all8.tex and paper/tables/all8_table.tex.

Usage: python scripts/p37_all8_joint.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

RES = 'test_results/FAMILY/10x25'
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SHARED = ['N', 'L', 'S', 'W', 'LS', 'LW']      # joint-v1's training set
HELD = ['SW', 'LSW']                            # zero-shot for v1, trained for all8
SEEDS = [301, 302, 303]
MARGIN = 1.0                                    # the paper's pre-specified +-1%
PARAMS = 71838

V1 = ['10x25+family+joint-v1', '10x25+family+joint-v1-s302',
      '10x25+family+joint-v1-s303']
ARMS = {'c': ('joint-all8',  2000),             # matched compute
        'x': ('joint-all8e', 2667)}             # matched exposure (2000 * 8/6)

rng = np.random.default_rng(7)


def arr(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    assert os.path.exists(p), f'missing {p}'
    a = np.load(p)
    v = (a[:, 0] if a.ndim > 1 else a).astype(float)
    assert len(v) == 100 and np.isfinite(v).all() and (v > 0).all(), \
        f'{p}: {len(v)} entries, finite={np.isfinite(v).all()}'
    return v


def pooled(tags, r):
    arrs = [arr(f'greedy+{t}', r) for t in tags]
    assert len(arrs) == 3, f'need 3 seeds, have {len(arrs)} for {tags[0]} on {r}'
    return np.mean(arrs, axis=0)


def ci(d, B=20000):
    idx = rng.integers(0, len(d), (B, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def guard():
    """Refuse to report unless every run is what the comparison claims it is."""
    for key, (stem, upd) in ARMS.items():
        for s in SEEDS:
            name = f'10x25+family+{stem}-s{s}'
            cfg = json.load(open(f'train_log/FAMILY/config_{name}.json'))
            assert cfg['family_regimes'] == ','.join(REGIMES), \
                f'{name}: trained on {cfg["family_regimes"]}, not all 8'
            assert cfg['max_updates'] == upd, \
                f'{name}: {cfg["max_updates"]} updates, expected {upd}'
            assert cfg['use_film'] is True and cfg['use_type_embedding'] is True, \
                f'{name}: architecture flags differ from joint-v1'
            assert cfg['seed_train'] == s, f'{name}: seed {cfg["seed_train"]}'
            sd = torch.load(f'trained_network/FAMILY/{name}.pth',
                            map_location='cpu', weights_only=False)
            n = sum(v.numel() for v in sd.values())
            assert n == PARAMS, f'{name}: {n} params, expected {PARAMS}'
    print(f'guard: 6 runs verified (regimes, updates, flags, seeds, '
          f'{PARAMS} params each)')


def fmt(x):
    return f'{x:+.2f}'


if __name__ == '__main__':
    guard()

    v1 = {r: pooled(V1, r) for r in REGIMES}
    stats = {}          # stats[key][r] = (mean, lo, hi)
    for key, (stem, _) in ARMS.items():
        tags = [f'10x25+family+{stem}-s{s}' for s in SEEDS]
        for r in REGIMES:
            d = 100.0 * (pooled(tags, r) - v1[r]) / v1[r]
            stats[(key, r)] = ci(d)

    # pooled over the six shared regimes: mean of the per-regime paired gaps
    shared = {k: float(np.mean([stats[(k, r)][0] for r in SHARED]))
              for k in ARMS}

    # TOST at the paper's margin: equivalent iff the whole CI sits inside +-1%
    def equivalent(k, r):
        _, lo, hi = stats[(k, r)]
        return -MARGIN < lo and hi < MARGIN

    for k in ARMS:
        for r in HELD:
            m, lo, hi = stats[(k, r)]
            print(f'  {ARMS[k][0]:12s} {r:4s} {fmt(m)}% [{fmt(lo)},{fmt(hi)}] '
                  f'{"EQUIVALENT" if equivalent(k, r) else "not equivalent"} at +-{MARGIN}%')
    # The paragraph this feeds claims equivalence on both compositions for the
    # matched-compute arm. Assert it so the prose cannot outlive the data.
    assert all(equivalent('c', r) for r in HELD), \
        'matched-compute arm is not equivalent on both compositions; rewrite the prose'

    os.makedirs('paper/tables', exist_ok=True)
    with open('paper/macros_all8.tex', 'w') as f:
        f.write('% auto-generated by scripts/p37_all8_joint.py -- do not hand-edit\n')
        f.write('% negative = the all-8 arm beats the 6-regime joint-v1\n')
        for k, pre in (('c', 'AEc'), ('x', 'AEx')):
            f.write(f'\\newcommand{{\\{pre}SharedDelta}}{{{fmt(shared[k])}}}\n')
            for r, name in (('SW', 'Sw'), ('LSW', 'Lsw'), ('S', 'S')):
                m, lo, hi = stats[(k, r)]
                f.write(f'\\newcommand{{\\{pre}{name}Delta}}{{{fmt(m)}}}'
                        f'\\newcommand{{\\{pre}{name}Lo}}{{{fmt(lo)}}}'
                        f'\\newcommand{{\\{pre}{name}Hi}}{{{fmt(hi)}}}\n')
        f.write('\\newcommand{\\AEcUpdates}{2000}\n')
        f.write('\\newcommand{\\AExUpdates}{2667}\n')
        f.write('\\newcommand{\\AExExtraPct}{33}\n')

    with open('paper/tables/all8_table.tex', 'w') as f:
        f.write('% auto-generated by scripts/p37_all8_joint.py -- do not hand-edit\n')
        f.write('\\begin{tabular}{lcc}\n\\toprule\n')
        f.write('regime & all eight, matched compute & all eight, matched exposure \\\\\n')
        f.write(' & (\\AEcUpdates{} updates) & (\\AExUpdates{} updates) \\\\\n\\midrule\n')
        for r in REGIMES:
            cells = []
            for k in ('c', 'x'):
                m, lo, hi = stats[(k, r)]
                cells.append(f'${fmt(m)}\\%$ $[{fmt(lo)},{fmt(hi)}]$')
            star = '$^{\\dagger}$' if r in HELD else ''
            f.write(f'${r}${star} & {cells[0]} & {cells[1]} \\\\\n')
        f.write('\\bottomrule\n\\end{tabular}\n')

    print(f'pooled over the six shared regimes: '
          f'compute {fmt(shared["c"])}%, exposure {fmt(shared["x"])}%')
    print('wrote paper/macros_all8.tex and paper/tables/all8_table.tex')
