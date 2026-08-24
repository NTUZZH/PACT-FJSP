"""P17: E3 token-causality tests with a family-wise multiplicity correction.

E3 runs three interventions (conflicting token, withheld token, full-blind) on
five regimes: fifteen one-sided paired Wilcoxon tests, reported raw. With fifteen
tests at alpha = 0.05 the chance of at least one spurious "significant" result is
about 54% under the null, and the paper leans on the pattern of which cells are
significant, not on any single cell. That pattern has to survive correction.

Holm-Bonferroni is used rather than Bonferroni: it controls the same family-wise
error rate but is uniformly more powerful, and it needs no independence
assumption (the tests share the true-token baseline, so they are correlated).

The correction is applied across the whole family of fifteen, not per intervention
row: splitting a family into subfamilies after seeing the results is exactly the
move that manufactures significance.

Usage: python scripts/p17_e3_multiplicity.py [--model joint-v1]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np
from scipy.stats import wilcoxon

RES = 'test_results/FAMILY/10x25'
REGIMES = ['L', 'S', 'W', 'LS', 'LW']
# Which CONFLICTING regime's token each regime is fed (scripts/p2_gate_g1.py's
# CF_MAP). Result files are named counterfactual-{fed token}_{actual regime}.
CF = {'L': 'S', 'S': 'L', 'W': 'LS', 'LS': 'W', 'LW': 'S'}
# full-blind removes the token AND the channels of the regime's ACTIVE members.
CH = {'L': 'lag', 'S': 'setup', 'W': 'window', 'LS': 'lag-setup',
      'LW': 'lag-window'}
INTERVENTIONS = [
    ('conflicting', lambda m, r: f'{m}+counterfactual-{CF[r]}'),
    ('withheld',    lambda m, r: f'{m}+withhold'),
    ('full-blind',  lambda m, r: f'{m}+withhold+chblind-{CH[r]}'),
]


def arr(tag, r):
    """Result vector, searching RES then any --extra-res fallbacks (in order)."""
    for d in [RES] + EXTRA_RES:
        p = f'{d}/Result_{tag}_{r}.npy'
        if os.path.exists(p):
            a = np.load(p)
            return (a[:, 0] if a.ndim > 1 else a).astype(float)
    return None


EXTRA_RES = []


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * pvals[k])
        adj[k] = min(1.0, running)
    return adj


def main():
    a = list(_ARGV)
    model = a[a.index('--model') + 1] if '--model' in a else 'joint-v1'
    if '--extra-res' in a:
        EXTRA_RES.append(a[a.index('--extra-res') + 1])
    base_tag = f'greedy+10x25+family+{model}'

    rows, raw, raw2 = [], [], []
    for name, namer in INTERVENTIONS:
        for r in REGIMES:
            base = arr(base_tag, r)
            interv = arr(f'greedy+10x25+family+{namer(model, r)}', r)
            if base is None or interv is None:
                print(f'MISSING {name} {r}: '
                      f'{namer(model, r)} -- regenerate the intervention evals')
                continue
            # one-sided: is the INTERVENED policy worse (larger makespan)?
            p = wilcoxon(interv, base, alternative='greater').pvalue
            # TWO-SIDED, over the same family. The one-sided test above answers
            # "does the intervention HURT?", which is the question E3 asks, but
            # it is structurally blind to the opposite answer: an intervention
            # that HELPS returns p -> 1 and prints as "no effect". It is not.
            # On S, withholding the token and asserting a WRONG one both make the
            # policy significantly FASTER after correction. The paper concluded
            # "the token buys no accuracy" from the one-sided nulls; the two-sided
            # family says something stronger and less comfortable, and the honest
            # move is to print it rather than to keep the tail that cannot see it.
            p2 = wilcoxon(interv, base).pvalue
            rows.append([name, r, float(np.mean(interv - base)), p, p2])
            raw.append(p)
            raw2.append(p2)

    if len(raw) != len(REGIMES) * len(INTERVENTIONS):
        print(f'\nINCOMPLETE: {len(raw)}/{len(REGIMES)*len(INTERVENTIONS)} tests. '
              'Not correcting a partial family.')
        return

    adj = holm(np.array(raw))
    adj2 = holm(np.array(raw2))
    # cells the two-sided family finds significant AND in which the intervention
    # made the policy FASTER: the effects the one-sided test cannot report
    helped = [(name, r, d) for (name, r, d, _, _), q2
              in zip(rows, adj2) if q2 < 0.05 and d < 0]
    print(f'E3 token causality, model={model}. {len(raw)} paired Wilcoxon tests, '
          f'Holm-Bonferroni across the whole family, one- and two-sided.\n')
    print(f'{"intervention":13s} {"reg":4s} {"delta (h)":>10s} {"Holm q (1s)":>12s} '
          f'{"Holm q (2s)":>12s}  verdict')
    print('-' * 70)
    survivors = 0
    for (name, r, d, p, p2), q, q2 in zip(rows, adj, adj2):
        sig = q < 0.05
        survivors += sig
        note = ''
        if q2 < 0.05:
            note = ('   <-- INTERVENTION MADE IT FASTER (invisible one-sided)'
                    if d < 0 else '   <-- slower')
        print(f'{name:13s} {r:4s} {d:+10.2f} {q:12.2e} {q2:12.2e}  '
              f'{"significant" if sig else "n.s.":12s}{note}')
    print(f'\n{survivors}/{len(raw)} survive the ONE-SIDED family-wise correction '
          f'at 0.05 (the paper\'s test: "did the intervention hurt?")')
    print(f'{int((adj2 < 0.05).sum())}/{len(raw)} survive the TWO-SIDED one; of '
          f'those, {len(helped)} are cells where the intervention HELPED:')
    for name, r, d in helped:
        print(f'    {name} on {r}: {d:+.2f} h -- the policy is FASTER without a '
              f'truthful token')

    macros = ['% auto-generated by scripts/p17_e3_multiplicity.py -- do not hand-edit',
              f'\\newcommand{{\\EthreeNTests}}{{{len(raw)}}}',
              f'\\newcommand{{\\EthreeNSig}}{{{survivors}}}'
              '  % E3 tests significant AFTER Holm correction (one-sided: did it hurt?)',
              f'\\newcommand{{\\EthreeNHelp}}{{{len(helped)}}}'
              '  % ... and cells where a TWO-SIDED family finds the intervention HELPED']
    for (name, r, d, p, p2), q, q2 in zip(rows, adj, adj2):
        key = {'conflicting': 'Cf', 'withheld': 'Wh', 'full-blind': 'Fb'}[name]
        macros.append(f'\\newcommand{{\\E{key}{r}Delta}}{{{d:+.2f}}}'
                      f'\\newcommand{{\\E{key}{r}Q}}{{{q:.1e}}}'
                      f'\\newcommand{{\\E{key}{r}QT}}{{{q2:.1e}}}')
    with open('paper/macros_e3.tex', 'w') as f:
        f.write('\n'.join(macros) + '\n')
    print('wrote paper/macros_e3.tex')


if __name__ == '__main__':
    main()
