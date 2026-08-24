r"""Regenerate the E6 scale-transfer table and its macros.

The 10-module-trained joint policy, evaluated unchanged at 20 and 30 modules,
against the dispatching rules. The rules select under the same feasibility mask
the policy's actor receives, as everywhere else in the paper, so a machine that
is mid-outage or awaiting a setup is not offered to them as idle.

E6 pools every training seed present on disk (scripts/run_evals_e6seeds.sh
produces three). Per regime and size it reports the mean per-instance relative
lead over the strongest dispatching rule with a paired-bootstrap 95% CI, and a
two-sided paired Wilcoxon signed-rank test, Holm-corrected across the eight
regimes within each size. Per-seed means are printed alongside, because three
seeds are three runs, not a 150-instance sample.

The whole tabular is emitted and the manuscript \inputs it. Nothing is retyped,
so nothing can drift from the data behind it.

Usage: python scripts/p18_e6_table.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import wilcoxon

from macro_io import emit, refuse

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
SIZES = ['20x25', '30x25']
WORD = {'20x25': 'Twenty', '30x25': 'Thirty'}
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
# PROTAGONIST SWAP (2026-08-11): E6 measures the released blind joint
JOINT = '10x25+family+jointblind-v1'     # the 10-module-trained policy
SEED_SFX = ('', '-s302', '-s303')
SEED_NAME = {'': 's301', '-s302': 's302', '-s303': 's303'}
MACROS = 'paper/macros_e6.tex'
TABLE = 'paper/tables/e6_table.tex'
ALPHA = 0.05
N_BOOT = 20000

rng = np.random.default_rng(7)           # one stream, fixed consumption order


def refuse_all(why):
    """Refuse BOTH outputs, not just the macro file. The supplementary pulls
    the table through \\InputIfFileExists with a conspicuous ?? fallback, so a
    table left on disk would keep printing numbers this run declined to
    certify, while a removed one is visibly withheld."""
    if os.path.exists(TABLE):
        os.remove(TABLE)
        print(f'removed {TABLE} (refusing to certify)')
    refuse(MACROS, why)


def arr(size, tag, r):
    p = f'test_results/FAMILY/{size}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def fresh_seeds(size):
    """Seed suffixes usable at this size: all 8 regimes on disk AND every file
    newer than the checkpoint it was evaluated from. The transfer directories
    are not purged on retrain the way 10x25 is, so mtime against the checkpoint
    is what keeps a superseded policy's results out of this table."""
    out, notes = [], []
    for sfx in SEED_SFX:
        ck = f'trained_network/FAMILY/{JOINT}{sfx}.pth'
        if not os.path.exists(ck):
            notes.append(f'{SEED_NAME[sfx]}: no checkpoint, cannot certify')
            continue
        t0 = os.path.getmtime(ck)
        ps = [f'test_results/FAMILY/{size}/Result_greedy+{JOINT}{sfx}_{r}.npy'
              for r in REGIMES]
        n_fresh = sum(1 for p in ps
                      if os.path.exists(p) and os.path.getmtime(p) >= t0)
        if n_fresh == len(REGIMES):
            out.append(sfx)
        elif any(os.path.exists(p) for p in ps):
            notes.append(f'{SEED_NAME[sfx]}: {n_fresh}/8 regime files fresh '
                         f'(newer than its checkpoint)')
    return out, notes


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved (as in p17)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * pvals[k])
        adj[k] = min(1.0, running)
    return adj


def boot_ci(d):
    """Percentile 95% CI of the mean of d under instance resampling."""
    idx = rng.integers(0, len(d), (N_BOOT, len(d)))
    bs = d[idx].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    # ------------------------------------------------------------ load + vet
    problems, per_size = [], {}
    for size in SIZES:
        sfx_list, notes = fresh_seeds(size)
        for n in notes:
            print(f'note: {size}: seed {n} -> excluded')
        if not sfx_list:
            problems.append(f'{size}: no seed has all 8 regimes fresh on disk')
            continue
        regs = {}
        for r in REGIMES:
            rules = {k: arr(size, k, r) for k in RULES}
            miss = [k for k, v in rules.items() if v is None]
            if miss:
                problems.append(f'{size}/{r}: missing rule result(s): '
                                + ', '.join(miss))
                continue
            pol = [arr(size, f'greedy+{JOINT}{sfx}', r) for sfx in sfx_list]
            ns = {len(v) for v in list(rules.values()) + pol}
            if len(ns) != 1:
                problems.append(f'{size}/{r}: instance counts differ across '
                                f'files ({sorted(ns)}); per-instance pairing '
                                'is impossible')
                continue
            pol = np.stack(pol)                      # [seeds, n_instances]
            if not (np.isfinite(pol).all()
                    and all(np.isfinite(v).all() for v in rules.values())):
                problems.append(f'{size}/{r}: non-finite makespan in a result '
                                'file')
                continue
            regs[r] = (pol, rules)
        per_size[size] = (sfx_list, regs)

    if problems:
        refuse_all('E6 inputs incomplete or unpaired:\n  '
                   + '\n  '.join(problems)
                   + '\n(scripts/run_evals_e6seeds.sh produces the three-seed '
                   'transfer evaluations.)')

    # One \EsixSeeds goes into the paper, so it must be one number. Refusing
    # here beats certifying a table whose two halves rest on different amounts
    # of evidence without saying so.
    counts = {s: len(sfx) for s, (sfx, _) in per_size.items()}
    if len(set(counts.values())) != 1:
        refuse_all('seed coverage differs by size: '
                   + ', '.join(f'{s}: {c}' for s, c in counts.items())
                   + '\nFinish the eval campaign (scripts/run_evals_e6seeds.sh) '
                   'so both sizes rest on the same seeds.')
    n_seeds = counts[SIZES[0]]

    # ------------------------------------------------------------ statistics
    stats = {s: {} for s in SIZES}
    pooled = {}
    for size in SIZES:
        sfx_list, regs = per_size[size]
        for r in REGIMES:
            pol, rules = regs[r]
            best = min(rules, key=lambda k: float(rules[k].mean()))
            bv = rules[best]
            # The Protocol's estimator: per-instance relative gap, policy in
            # the denominator (as in p13). Positive = the policy is faster.
            lead = 100.0 * (bv[None, :] - pol) / pol       # [seeds, n]
            # The CI resamples INSTANCES. A seed rerun of the same instance is
            # not a new draw from the instance distribution, so leads are
            # averaged over seeds per instance before resampling; bootstrapping
            # the flat seed-by-instance pool would shrink the interval by about
            # sqrt(n_seeds).
            d_bar = lead.mean(axis=0)
            lo, hi = boot_ci(d_bar)
            # Wilcoxon pairs by instance for the same reason: the policy value
            # of an instance is its seed mean, so n stays at the instance count
            # instead of being inflated by replicate seeds.
            x = pol.mean(axis=0)
            if np.allclose(x, bv):
                p = 1.0            # no nonzero differences, nothing to rank
            else:
                p = float(wilcoxon(x, bv, alternative='two-sided').pvalue)
            stats[size][r] = dict(
                best=best, bestv=float(bv.mean()), ours=float(pol.mean()),
                spt=float(rules['SPT'].mean()), mwkr=float(rules['MWKR'].mean()),
                mean=float(lead.mean()), lo=lo, hi=hi, p=p,
                lead=lead, dbar=d_bar, n=pol.shape[1])
        adj = holm(np.array([stats[size][r]['p'] for r in REGIMES]))
        for r, a in zip(REGIMES, adj):
            stats[size][r]['holm'] = float(a)
        # Size-level headline: per-instance seed-mean leads pooled over the
        # eight regimes (instances are independent draws, so a flat resample is
        # paired the same way p13's pooled_gap is).
        d_all = np.concatenate([stats[size][r]['dbar'] for r in REGIMES])
        plo, phi = boot_ci(d_all)
        pooled[size] = (float(d_all.mean()), plo, phi)

    # --------------------------------------------------------------- stdout
    names = {s: [SEED_NAME[x] for x in per_size[s][0]] for s in SIZES}
    seed_means = {}
    print(f'E6 scale transfer: {n_seeds} seed(s) pooled '
          f'({", ".join(names[SIZES[0]])}); lead = 100*(rule - ours)/ours, '
          'positive = ours faster')
    for size in SIZES:
        print(f'\n{size} (n={stats[size][REGIMES[0]]["n"]} instances):')
        print(f'{"regime":>6} {"best":>5} {"ours":>7} {"rule":>7} {"lead%":>7} '
              f'{"95% CI":>15} {"raw p":>9} {"Holm p":>9}')
        for r in REGIMES:
            st = stats[size][r]
            ci = f'[{st["lo"]:+.1f},{st["hi"]:+.1f}]'
            mark = ' *' if st['holm'] < ALPHA else ''
            print(f'{r:>6} {st["best"]:>5} {st["ours"]:7.1f} {st["bestv"]:7.1f} '
                  f'{st["mean"]:+7.2f} {ci:>15} {st["p"]:9.2e} '
                  f'{st["holm"]:9.2e}{mark}')
        seed_means[size] = [
            float(np.concatenate(
                [stats[size][r]['lead'][j] for r in REGIMES]).mean())
            for j in range(n_seeds)]
        print('  per-seed mean lead: ' + '  '.join(
            f'{nm} {v:+.2f}%' for nm, v in zip(names[size], seed_means[size])))
        m, lo, hi = pooled[size]
        print(f'  pooled over regimes: {m:+.2f}%  CI [{lo:+.2f},{hi:+.2f}]')

    # ------------------------------------------------- macros (checks first)
    beat = {s: sum(1 for r in REGIMES
                   if stats[s][r]['ours'] < stats[s][r]['bestv'])
            for s in SIZES}
    worst = max(100.0 * (stats[s][r]['ours'] - stats[s][r]['bestv'])
                / stats[s][r]['bestv'] for s in SIZES for r in REGIMES)
    leads_mwkr = [100.0 * (stats[s][r]['mwkr'] - stats[s][r]['ours'])
                  / stats[s][r]['mwkr'] for s in SIZES for r in REGIMES]
    # Regimes ahead of the strongest rule AND still significant after Holm.
    # Direction matters: a significant loss must not inflate this count.
    sig = {s: sum(1 for r in REGIMES
                  if stats[s][r]['holm'] < ALPHA and stats[s][r]['mean'] > 0)
           for s in SIZES}

    # The manuscript reports the two sizes separately, so each count gets its
    # own macro; a single shared count would only be honest if they agreed.
    macros = [
        '% auto-generated by scripts/p18_e6_table.py -- do not hand-edit',
        f'\\newcommand{{\\EsixBeatPdrTwenty}}{{{beat["20x25"]}}}'
        f'\\newcommand{{\\EsixBeatPdrThirty}}{{{beat["30x25"]}}}'
        '  % regimes (of 8) beating every rule, per size',
        f'\\newcommand{{\\EsixWorstSptGap}}{{{worst:.1f}}}'
        '  % worst pooled-mean gap behind the best rule, %',
        f'\\newcommand{{\\EsixMwkrLo}}{{{min(leads_mwkr):.0f}}}'
        f'\\newcommand{{\\EsixMwkrHi}}{{{max(leads_mwkr):.0f}}}'
        f'  % lead over MWKR, % (span {min(leads_mwkr):.1f}-{max(leads_mwkr):.1f})',
        f'\\newcommand{{\\EsixSeeds}}{{{n_seeds}}}'
        '  % training seeds pooled into every E6 number',
    ]
    for s in SIZES:
        m, lo, hi = pooled[s]
        w = WORD[s]
        macros += [
            f'\\newcommand{{\\Esix{w}Delta}}{{{m:+.1f}}}'
            f'\\newcommand{{\\Esix{w}Lo}}{{{lo:+.2f}}}'
            f'\\newcommand{{\\Esix{w}Hi}}{{{hi:+.2f}}}'
            f'  % {s}: mean per-instance lead over the strongest rule, '
            '% + paired-bootstrap 95% CI',
            f'\\newcommand{{\\Esix{w}Sig}}{{{sig[s]}}}'
            f'  % {s}: regimes (of 8) ahead with Holm-adjusted Wilcoxon '
            f'p < {ALPHA}',
            '% per-seed mean lead at ' + s + ': ' + '  '.join(
                f'{nm} {v:+.2f}%' for nm, v in zip(names[s], seed_means[s])),
        ]

    # ------------------------------------------------------------ the table
    tex = []
    for r in REGIMES:
        out = [r]
        for size in SIZES:
            st = stats[size][r]
            j, spt = st['ours'], st['spt']
            # bold whichever of ours / SPT is better; MWKR never wins
            jb = f'\\textbf{{{j:.0f}}}' if j < spt else f'{j:.0f}'
            sb = f'\\textbf{{{spt:.0f}}}' if spt < j else f'{spt:.0f}'
            cell = (f'{st["mean"]:+.1f}\\,[{st["lo"]:+.1f},{st["hi"]:+.1f}]'
                    + ('$^{*}$' if st['holm'] < ALPHA else ''))
            out += [jb, sb, f'{st["mwkr"]:.0f}', cell]
        tex.append(' & '.join(out) + r' \\')

    # The WHOLE tabular is emitted, not just the rows, and the manuscript
    # inputs it from OUTSIDE any alignment: LaTeX's \input leaves a token after
    # the file contents, which opens a fresh cell and misplaces the \bottomrule
    # that follows. The legend rows live inside the tabular so the star is
    # defined in the same generated file that prints it.
    head = [r'\begin{tabular}{lcccl@{\hskip 8pt}cccl}', r'\toprule',
            r'& \multicolumn{4}{c}{20 modules} & \multicolumn{4}{c}{30 modules} \\',
            r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',
            r'regime & ours & SPT & MWKR & lead (\%) & '
            r'ours & SPT & MWKR & lead (\%) \\', r'\midrule']
    foot = [r'\midrule',
            r'\multicolumn{9}{l}{\scriptsize ours = mean makespan over '
            rf'{n_seeds} training seeds; lead = mean per-instance gap to the '
            r'strongest rule,} \\',
            r'\multicolumn{9}{l}{\scriptsize positive = ours faster, with '
            r'paired-bootstrap 95\% CI; $^{*}$ = two-sided paired Wilcoxon, '
            r'Holm $p<0.05$.} \\',
            r'\bottomrule', r'\end{tabular}']
    os.makedirs('paper/tables', exist_ok=True)
    with open(TABLE, 'w') as f:
        f.write('% auto-generated by scripts/p18_e6_table.py '
                '-- do not hand-edit\n')
        f.write('\n'.join(head + tex + foot) + '\n')
    print(f'\nLaTeX table (bold = best solver-free) -> {TABLE}')

    emit(MACROS, macros)

    print(f'\nbeats best PDR on {beat["20x25"]}/8 at 20 modules, '
          f'{beat["30x25"]}/8 at 30')
    print(f'worst gap behind best PDR: {worst:+.1f}%')
    print(f'lead over MWKR: {min(leads_mwkr):.1f}-{max(leads_mwkr):.1f}%')
    print('significant after Holm (ahead): '
          + ', '.join(f'{s}: {sig[s]}/8' for s in SIZES))


if __name__ == '__main__':
    main()
