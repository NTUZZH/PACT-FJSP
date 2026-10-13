r"""P78: the published learned rival (HGNN, Song et al. 2023) against PACT.

WHAT IS COMPARED. Three HGNN arms, ported to this benchmark in backbones/hgnn
(204,250 parameters, the upstream recipe, PACT's action mask, seed 301):

  hgnn-certified-joint-s301   one HGNN policy trained on the six training regimes
                              with PACT's certified reward
  hgnn-published-joint-s301   the same, with HGNN's own published reward
                              (makespan-estimate decrease)
  hgnn-certified-<R>-s301     one HGNN policy per regime R, certified reward,
                              read on its own regime only

against the released PACT policy (per-instance mean over the three seed files
Result_greedy+10x25+family+jointblind-v1{,-s302,-s303}_<regime>.npy) and SPT, at
10, 20, 30, 50 and 80 modules on the published test instances. At 50 and 80
modules every arm is read on the first 10 descriptors per regime in sorted
order, the prefix the constraint-aware rules of p72 use.

SEEDS. PACT pools its three training seeds; the HGNN joint arms pool every seed
file on disk and the specialists have one. A pooled gap with unequal seed counts
on the two sides can hide a seed effect, so every pooled gap is also computed
against each PACT seed alone (the least favourable is emitted as
\HgnnJointGapWorst...) and for each HGNN seed alone against pooled PACT (the most
favourable is emitted as \HgnnJointGapBestSeed...). The text states a verdict
only where it holds on every seed alone.

ESTIMATOR. Per-instance relative gap 100*(a-b)/b, mean with a paired-bootstrap
95% CI (20,000 resamples, default_rng(7) drawn in a fixed order), p11's
estimator. "gap to PACT" is positive when the HGNN arm is worse; "gap to SPT" is
negative when the arm is ahead of SPT. Pooled rows concatenate the per-instance
gaps of the eight regimes.

Emits results/hgnn_rival.{json,md}, paper/macros_hgnn.tex,
paper/tables/hgnn_table.tex (10, 20, 30 modules) and
paper/tables/hgnn_table_prod.tex (50 and 80 modules).

Usage: python scripts/p78_hgnn_rival.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from macro_io import emit

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SIZES = ['10x25', '20x25', '30x25', '50x25', '80x25']
WORD = {'10x25': 'Ten', '20x25': 'Twenty', '30x25': 'Thirty',
        '50x25': 'Fifty', '80x25': 'Eighty'}
MODULES = {'10x25': 10, '20x25': 20, '30x25': 30, '50x25': 50, '80x25': 80}
N_PROD = 10                       # p72's prefix at 50x25 and 80x25
HGNN_SEED = 301                   # the specialists' seed; joint arms pool every seed on disk
HGNN_PARAMS = 204250
ARMS = {'joint': 'hgnn-certified-joint', 'pub': 'hgnn-published-joint'}
PACT_RE = re.compile(r'^Result_greedy\+10x25\+family\+jointblind-v1(-s30[23])?_(\w+)\.npy$')
N_BOOT = 20000
rng = np.random.default_rng(7)


def ci(d, B=N_BOOT):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def load(size, tag, regime):
    p = f'test_results/FAMILY/{size}/Result_{tag}_{regime}.npy'
    if not os.path.exists(p):
        sys.exit(f'missing {p}')
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def hgnn_seeds(size, base, regime):
    """Every HGNN seed array on disk for (size, base arm, regime), in seed order."""
    files = sorted(glob.glob(f'test_results/FAMILY/{size}/Result_{base}-s30[0-9]_{regime}.npy'))
    if not files:
        sys.exit(f'missing HGNN files for {base} at {size}/{regime}')
    out = []
    for f in files:
        a = np.load(f)
        out.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return out, [re.search(r'-s(30\d)_', f).group(1) for f in files]


def pact_seeds(size, regime):
    """The three PACT seed arrays, in seed order, for (size, regime)."""
    d = f'test_results/FAMILY/{size}'
    names = sorted(os.path.basename(p) for p in
                   glob.glob(f'{d}/Result_greedy+10x25+family+jointblind-v1*_{regime}.npy'))
    names = [n for n in names if PACT_RE.match(n) and PACT_RE.match(n).group(2) == regime]
    if len(names) != 3:
        sys.exit(f'expected 3 PACT seed files at {size}/{regime}, found {names}')
    return [load(size, n[len('Result_'):-len(f'_{regime}.npy')], regime) for n in names]


def n_at(size):
    return N_PROD if size in ('50x25', '80x25') else 0


def cut(a, size):
    n = n_at(size)
    return a[:n] if n else a


def hgnn_train_hours():
    out = {}
    for f in glob.glob('train_log/FAMILY_hgnn-*-s301.log'):
        name = os.path.basename(f)[len('FAMILY_'):-len('.log')]
        for line in open(f):
            m = re.search(r'done in ([0-9.]+) h', line)
            if m:
                out[name] = float(m.group(1))
    return out


def main():
    hseeds = {}
    out = {'hgnn_params': HGNN_PARAMS, 'hgnn_spec_seed': HGNN_SEED, 'n_prod': N_PROD,
           'estimator': f'per-instance relative gap, paired bootstrap {N_BOOT}, default_rng(7)',
           'sizes': {}}
    md = ['# HGNN (Song et al. 2023) ported to the timing-rule family, against PACT',
          '',
          f'HGNN: {HGNN_PARAMS} parameters, seed {HGNN_SEED}, PACT action mask. '
          'PACT: per-instance mean over three seeds. gap to PACT > 0 = HGNN worse; '
          'gap to SPT < 0 = ahead of SPT. 95% paired-bootstrap CIs.', '']
    macros = []
    rows_small, rows_prod = [], []

    for size in SIZES:
        S = {'per_regime': {}, 'pooled': {}}
        gaps = {k: [] for k in ('joint', 'pub', 'spec')}          # to PACT
        gaps_spt = {k: [] for k in ('joint', 'pub', 'spec', 'pact')}
        gaps_seed = {k: [[], [], []] for k in ('joint', 'pub', 'spec')}  # to each PACT seed
        gaps_hseed = {k: {} for k in ('joint', 'pub')}                   # each HGNN seed to pooled PACT
        gap_cert_pub, gap_joint_spec = [], []
        means = {k: [] for k in ('spt', 'pact', 'joint', 'pub', 'spec')}
        rows = rows_prod if size in ('50x25', '80x25') else rows_small
        n_inst = None
        rows.append(r'\midrule' if rows else '')
        for r in REGIMES:
            seeds = [cut(a, size) for a in pact_seeds(size, r)]
            pact = np.mean(seeds, axis=0)
            spt = cut(load(size, 'SPT', r), size)
            arms, arm_seeds = {}, {}
            for k, t in ARMS.items():
                xs, sd = hgnn_seeds(size, t, r)
                xs = [cut(x, size) for x in xs]
                arms[k], arm_seeds[k] = np.mean(xs, axis=0), xs
                hseeds.setdefault(k, sd)
                if hseeds[k] != sd:
                    sys.exit(f'HGNN seed set differs across cells for {t}: {hseeds[k]} vs {sd} at {size}/{r}')
            arms['spec'] = cut(load(size, f'hgnn-certified-{r}-s{HGNN_SEED}', r), size)
            arm_seeds['spec'] = [arms['spec']]
            n = len(pact)
            if n_inst is None:
                n_inst = n
            assert all(len(x) == n for x in [spt, *arms.values(), *seeds]), (size, r)
            R = {'n': n, 'mean_SPT': float(spt.mean()), 'mean_PACT': float(pact.mean())}
            means['spt'].append(spt); means['pact'].append(pact)
            for k in ('joint', 'pub', 'spec'):
                a = arms[k]
                means[k].append(a)
                g = 100.0 * (a - pact) / pact
                gaps[k].append(g)
                R[f'mean_{k}'] = float(a.mean())
                R[f'gap_PACT_{k}'] = list(ci(g))
                gs = 100.0 * (a - spt) / spt
                gaps_spt[k].append(gs)
                R[f'gap_SPT_{k}'] = list(ci(gs))
                for i, s in enumerate(seeds):
                    gaps_seed[k][i].append(100.0 * (a - s) / s)
                if k in gaps_hseed:
                    for i, x in enumerate(arm_seeds[k]):
                        gaps_hseed[k].setdefault(i, []).append(100.0 * (x - pact) / pact)
            gp = 100.0 * (pact - spt) / spt
            gaps_spt['pact'].append(gp)
            R['gap_SPT_pact'] = list(ci(gp))
            gcp = 100.0 * (arms['joint'] - arms['pub']) / arms['pub']
            gap_cert_pub.append(gcp)
            R['gap_cert_vs_pub'] = list(ci(gcp))
            gjs = 100.0 * (arms['joint'] - arms['spec']) / arms['spec']
            gap_joint_spec.append(gjs)
            R['gap_joint_vs_spec'] = list(ci(gjs))
            S['per_regime'][r] = R

            vals = [R['mean_SPT'], R['mean_PACT'], R['mean_joint'], R['mean_pub'], R['mean_spec']]
            best = min(vals)

            def cell(v):
                return (r'\textbf{%.1f}' % v) if abs(v - best) < 1e-9 else '%.1f' % v

            def gtxt(c):
                return _g(c)
            rows.append(f'${r}$ & {cell(vals[0])} & {cell(vals[1])} & '
                        f'{cell(vals[2])} & {gtxt(R["gap_PACT_joint"])} & '
                        f'{cell(vals[3])} & {gtxt(R["gap_PACT_pub"])} & '
                        f'{cell(vals[4])} & {gtxt(R["gap_PACT_spec"])} \\\\')
        # pooled over the eight regimes
        P = {}
        for k in ('joint', 'pub', 'spec'):
            P[f'gap_PACT_{k}'] = list(ci(np.concatenate(gaps[k])))
            P[f'gap_SPT_{k}'] = list(ci(np.concatenate(gaps_spt[k])))
            P[f'gap_PACT_{k}_per_seed'] = [list(ci(np.concatenate(gaps_seed[k][i]))) for i in range(3)]
            if k in gaps_hseed:
                P[f'gap_PACT_{k}_per_hgnn_seed'] = [list(ci(np.concatenate(gaps_hseed[k][i])))
                                                    for i in sorted(gaps_hseed[k])]
            P[f'beats_SPT_{k}'] = sum(1 for r in REGIMES if S['per_regime'][r][f'gap_SPT_{k}'][2] < 0)
            P[f'PACT_ahead_{k}'] = sum(1 for r in REGIMES if S['per_regime'][r][f'gap_PACT_{k}'][1] > 0)
            P[f'{k}_ahead_PACT'] = sum(1 for r in REGIMES if S['per_regime'][r][f'gap_PACT_{k}'][2] < 0)
        P['gap_SPT_pact'] = list(ci(np.concatenate(gaps_spt['pact'])))
        P['pact_beats_SPT'] = sum(1 for r in REGIMES if S['per_regime'][r]['gap_SPT_pact'][2] < 0)
        P['gap_cert_vs_pub'] = list(ci(np.concatenate(gap_cert_pub)))
        P['gap_joint_vs_spec'] = list(ci(np.concatenate(gap_joint_spec)))
        P['n_per_regime'] = n_inst
        for k in means:
            P[f'mean_{k}'] = float(np.concatenate(means[k]).mean())
        S['pooled'] = P
        out['sizes'][size] = S
        pv = [P['mean_spt'], P['mean_pact'], P['mean_joint'], P['mean_pub'], P['mean_spec']]
        pb = min(pv)
        pc = [(r'\textbf{%.1f}' % v) if abs(v - pb) < 1e-9 else '%.1f' % v for v in pv]
        rows.insert(len(rows) - 8, r'\multicolumn{9}{@{}l}{\textit{%d modules, $n = %d$ instances per regime}} \\'
                    % (MODULES[size], n_inst))
        rows.append(f'pooled & {pc[0]} & {pc[1]} & {pc[2]} & {_g(P["gap_PACT_joint"])} & '
                    f'{pc[3]} & {_g(P["gap_PACT_pub"])} & {pc[4]} & {_g(P["gap_PACT_spec"])} \\\\')

        # console + markdown
        w = WORD[size]
        md += [f'## {size} (n = {n_inst} per regime)', '',
               '| regime | SPT | PACT | HGNN joint (cert.) | gap to PACT | HGNN joint (publ.) | gap to PACT | HGNN specialist | gap to PACT | joint vs SPT | spec vs SPT |',
               '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in REGIMES:
            R = S['per_regime'][r]
            md.append(f'| {r} | {R["mean_SPT"]:.1f} | {R["mean_PACT"]:.1f} | {R["mean_joint"]:.1f} | '
                      f'{_m(R["gap_PACT_joint"])} | {R["mean_pub"]:.1f} | {_m(R["gap_PACT_pub"])} | '
                      f'{R["mean_spec"]:.1f} | {_m(R["gap_PACT_spec"])} | {_m(R["gap_SPT_joint"])} | {_m(R["gap_SPT_spec"])} |')
        md.append(f'| pooled | {P["mean_spt"]:.1f} | {P["mean_pact"]:.1f} | {P["mean_joint"]:.1f} | '
                  f'{_m(P["gap_PACT_joint"])} | {P["mean_pub"]:.1f} | {_m(P["gap_PACT_pub"])} | '
                  f'{P["mean_spec"]:.1f} | {_m(P["gap_PACT_spec"])} | {_m(P["gap_SPT_joint"])} | {_m(P["gap_SPT_spec"])} |')
        md += ['', f'- PACT vs SPT pooled: {_m(P["gap_SPT_pact"])}',
               f'- certified reward vs published reward on HGNN (joint, pooled): {_m(P["gap_cert_vs_pub"])} (negative = certified better)',
               f'- HGNN joint vs HGNN specialists (pooled): {_m(P["gap_joint_vs_spec"])} (negative = joint better)',
               f'- regimes where PACT is ahead (CI clear) of HGNN joint / published / specialist: '
               f'{P["PACT_ahead_joint"]} / {P["PACT_ahead_pub"]} / {P["PACT_ahead_spec"]} of 8; '
               f'HGNN ahead of PACT: {P["joint_ahead_PACT"]} / {P["pub_ahead_PACT"]} / {P["spec_ahead_PACT"]}',
               f'- regimes where the arm is ahead of SPT (CI clear): joint {P["beats_SPT_joint"]}, published {P["beats_SPT_pub"]}, specialist {P["beats_SPT_spec"]} of 8',
               '- gap to each PACT seed alone (pooled): ' + '; '.join(
                   f'{k}: ' + ', '.join(_m(c) for c in P[f'gap_PACT_{k}_per_seed']) for k in ('joint', 'pub', 'spec')),
               '- each HGNN seed alone vs pooled PACT: ' + '; '.join(
                   f'{k} (seeds {",".join(hseeds[k])}): ' + ', '.join(_m(c) for c in P[f'gap_PACT_{k}_per_hgnn_seed'])
                   for k in ('joint', 'pub')),
               '']
        print(f'== {size}: HGNN joint(cert) gap to PACT {_m(P["gap_PACT_joint"])}  '
              f'published {_m(P["gap_PACT_pub"])}  specialist {_m(P["gap_PACT_spec"])} | '
              f'vs SPT: joint {_m(P["gap_SPT_joint"])} spec {_m(P["gap_SPT_spec"])} PACT {_m(P["gap_SPT_pact"])}')
        for k, name in (('joint', 'Joint'), ('pub', 'Pub'), ('spec', 'Spec')):
            g = P[f'gap_PACT_{k}']
            worst = min(P[f'gap_PACT_{k}_per_seed'], key=lambda c: c[0])
            if k in ('joint', 'pub'):
                best_h = min(P[f'gap_PACT_{k}_per_hgnn_seed'], key=lambda c: c[0])
                macros += [(f'Hgnn{name}GapBestSeed{w}', f'{best_h[0]:.1f}',
                            f'{size}: gap of the most favourable single HGNN seed to pooled PACT'),
                           (f'Hgnn{name}GapBestSeed{w}Lo', f'{best_h[1]:.1f}', '95% CI low')]
            gs = P[f'gap_SPT_{k}']
            macros += [
                (f'Hgnn{name}Gap{w}', f'{g[0]:.1f}', f'{size}: pooled gap of HGNN {k} to PACT, % (positive = HGNN worse)'),
                (f'Hgnn{name}Gap{w}Lo', f'{g[1]:.1f}', '95% CI low'),
                (f'Hgnn{name}Gap{w}Hi', f'{g[2]:.1f}', '95% CI high'),
                (f'Hgnn{name}GapWorst{w}', f'{worst[0]:.1f}', f'{size}: the same gap against the least favourable single PACT seed'),
                (f'Hgnn{name}GapWorst{w}Lo', f'{worst[1]:.1f}', '95% CI low'),
                (f'Hgnn{name}SptGap{w}', f'{gs[0]:.1f}', f'{size}: pooled gap of HGNN {k} to SPT, % (negative = ahead of SPT)'),
                (f'Hgnn{name}SptGap{w}Lo', f'{gs[1]:.1f}', '95% CI low'),
                (f'Hgnn{name}SptGap{w}Hi', f'{gs[2]:.1f}', '95% CI high'),
                (f'Hgnn{name}BeatsSpt{w}', str(P[f'beats_SPT_{k}']), f'{size}: regimes (of 8) where HGNN {k} is ahead of SPT with the CI clear of zero'),
                (f'HgnnPactAhead{name}{w}', str(P[f'PACT_ahead_{k}']), f'{size}: regimes (of 8) where PACT is ahead of HGNN {k} with the CI clear of zero'),
                (f'Hgnn{name}AheadPact{w}', str(P[f'{k}_ahead_PACT']), f'{size}: regimes (of 8) where HGNN {k} is ahead of PACT with the CI clear of zero'),
            ]
        macros += [
            (f'HgnnCertVsPub{w}', f'{P["gap_cert_vs_pub"][0]:.1f}', f'{size}: HGNN joint, certified reward vs published reward, % (negative = certified better)'),
            (f'HgnnCertVsPub{w}Lo', f'{P["gap_cert_vs_pub"][1]:.1f}', '95% CI low'),
            (f'HgnnCertVsPub{w}Hi', f'{P["gap_cert_vs_pub"][2]:.1f}', '95% CI high'),
            (f'HgnnJointVsSpec{w}', f'{P["gap_joint_vs_spec"][0]:.1f}', f'{size}: HGNN joint vs HGNN specialists, % (negative = joint better)'),
            (f'HgnnJointVsSpec{w}Lo', f'{P["gap_joint_vs_spec"][1]:.1f}', '95% CI low'),
            (f'HgnnJointVsSpec{w}Hi', f'{P["gap_joint_vs_spec"][2]:.1f}', '95% CI high'),
            (f'HgnnN{w}', str(n_inst), f'{size}: instances per regime'),
            (f'HgnnPactSptGap{w}', f'{P["gap_SPT_pact"][0]:.1f}', f'{size}: pooled gap of PACT to SPT, % (negative = ahead)'),
            (f'HgnnPactSptGap{w}Abs', f'{abs(P["gap_SPT_pact"][0]):.1f}', 'absolute value'),
            (f'HgnnPactSptGap{w}Lo', f'{P["gap_SPT_pact"][1]:.1f}', '95% CI low'),
            (f'HgnnPactSptGap{w}Hi', f'{P["gap_SPT_pact"][2]:.1f}', '95% CI high'),
            (f'HgnnPactBeatsSpt{w}', str(P['pact_beats_SPT']), f'{size}: regimes (of 8) where PACT is ahead of SPT with the CI clear of zero'),
        ]

    # Mask tie rounding, measured by the port's action-set test (F3) on random
    # rollouts: decisions after the first at which PACT's mask is strictly
    # smaller than the exact non-delay set because a pair that ties the
    # decision instant exactly falls out by one rounding unit on the normalized
    # axis. results/hgnn_port_f3.txt is the saved pytest -k f3 -s output.
    f3 = open('results/hgnn_port_f3.txt').read()
    m = re.search(r"F3 total: \{'decisions': (\d+), 'pact_smaller': (\d+), 'first_pact_larger': (\d+)", f3)
    if not m:
        sys.exit('results/hgnn_port_f3.txt has no F3 total line; rerun pytest -k f3 -s')
    n_dec, n_tie, n_first = (int(x) for x in m.groups())
    macros += [('MaskTieDecisions', f'{n_dec:,}', 'random-rollout decisions sampled by the action-set test (8 regimes x 5 instances)'),
               ('MaskTiePct', f'{100.0 * n_tie / n_dec:.1f}', 'percent of those decisions at which the mask drops an exactly tied pair'),
               ('MaskFirstLaterPct', f'{100.0 * n_first / n_dec:.1f}', 'percent of decisions (all first decisions of an episode) at which the mask admits a later-starting pair')]
    out['mask_tie'] = dict(decisions=n_dec, tie_dropped=n_tie, first_later=n_first)

    # Training budget and convergence of the HGNN arms, for the supplement's
    # protocol paragraph: episodes per regime under the upstream recipe (1000
    # iterations x 20 instances), the iteration at which each one-policy run
    # reached its best validation score, and the upstream reproduction check.
    best_iter = {}
    for f in glob.glob('train_log/FAMILY_hgnn-*-joint-s30[0-9].log'):
        name = os.path.basename(f)[len('FAMILY_'):-len('.log')]
        pts = [(float(m.group(2)), int(m.group(1))) for m in
               re.finditer(r'\[vali @(\d+)\] rel=([0-9.]+)', open(f).read())]
        if pts:
            best_iter[name] = min(pts)[1]
    out['hgnn_best_validation_iteration'] = best_iter
    up = open('results/hgnn_upstream_test.txt').read()
    mu = re.search(r'mean makespan: ([0-9.]+)', up)
    macros += [('HgnnIters', '1,000', 'HGNN training iterations (upstream recipe)'),
               ('HgnnBatch', '20', 'instances per iteration (upstream recipe)'),
               ('HgnnJointEpisodesPerRegime', f'{1000 * 20 // 6:,}', 'episodes per training regime for the one-policy arms (20,000 over six regimes)'),
               ('HgnnSpecEpisodes', f'{1000 * 20:,}', 'episodes for each HGNN specialist (one regime)'),
               ('HgnnBestIterMin', str(min(best_iter.values())), 'earliest best-validation iteration among the one-policy runs'),
               ('HgnnBestIterMax', str(max(best_iter.values())), 'latest best-validation iteration among the one-policy runs'),
               ('HgnnUpstreamMs', f'{float(mu.group(1)):.2f}', 'mean makespan of the upstream checkpoint on the upstream 10x5 test set, reproduced by the port')]

    hours = hgnn_train_hours()
    out['train_hours'] = hours
    out['hgnn_joint_seeds'] = hseeds
    jh = hours.get(f"{ARMS['joint']}-s{HGNN_SEED}")
    sh = [hours[f'hgnn-certified-{r}-s301'] for r in REGIMES if f'hgnn-certified-{r}-s301' in hours]
    macros += [('HgnnParams', f'{HGNN_PARAMS:,}', 'HGNN parameter count'),
               ('HgnnJointSeeds', str(len(hseeds['joint'])), 'HGNN training seeds pooled in the joint arms'),
               ('HgnnSpecSeeds', '1', 'HGNN training seeds per specialist'),
               ('HgnnNProd', str(N_PROD), 'instances per regime at 50 and 80 modules')]
    if jh is not None:
        macros.append(('HgnnJointTrainH', f'{jh:.1f}', 'HGNN joint (certified) training wall-clock, h, GPU'))
    if sh:
        macros.append(('HgnnSpecTrainH', f'{np.mean(sh):.1f}', 'HGNN specialist training wall-clock, mean h, GPU'))
    md += ['## Training wall-clock (GPU, h)', ''] + [f'- {k}: {v:.2f}' for k, v in sorted(hours.items())]

    os.makedirs('results', exist_ok=True)
    with open('results/hgnn_rival.json', 'w') as f:
        json.dump(out, f, indent=1)
    with open('results/hgnn_rival.md', 'w') as f:
        f.write('\n'.join(md) + '\n')
    emit('paper/macros_hgnn.tex',
         [f'\\newcommand{{\\{n}}}{{{v}}}  % {c}' for n, v, c in macros],
         header='% auto-generated by scripts/p78_hgnn_rival.py -- do not hand-edit')
    for path, rows in (('paper/tables/hgnn_table.tex', rows_small),
                       ('paper/tables/hgnn_table_prod.tex', rows_prod)):
        head = [r'% auto-generated by scripts/p78_hgnn_rival.py -- do not hand-edit', '',
                r'\begin{tabular}{@{}l rr rl rl rl@{}}', r'\toprule',
                r' & & & \multicolumn{2}{c}{HGNN, one policy, certified reward} & \multicolumn{2}{c}{HGNN, one policy, published reward} & \multicolumn{2}{c}{HGNN, one policy per regime} \\',
                r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}',
                r'Regime & SPT & PACT & mean & gap to PACT [95\% CI] & mean & gap to PACT [95\% CI] & mean & gap to PACT [95\% CI] \\']
        body = [x for x in rows if x != '']
        with open(path, 'w') as f:
            f.write('\n'.join(head + body + [r'\bottomrule', r'\end{tabular}']) + '\n')
    print('wrote results/hgnn_rival.{json,md}, paper/macros_hgnn.tex, paper/tables/hgnn_table{,_prod}.tex')


def _m(c):
    return f'{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}]'


def _s(v):
    if abs(v) < 0.005:
        return '0.00'
    return ('$+$' if v > 0 else '$-$') + f'{abs(v):.2f}'


def _g(c):
    return f'{_s(c[0])} [{_s(c[1])}, {_s(c[2])}]'


if __name__ == '__main__':
    main()
