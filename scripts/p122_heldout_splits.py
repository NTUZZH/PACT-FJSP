r"""P122: policies trained with other held-out pairs of combinations.

QUESTION
  The policy xmask-v1 trains on N, L, S, W, LS, LW and is tested on the held-out combinations SW
  and LSW, so both held-out combinations test the interaction of the setup and wait constraints.
  Two further policies hold out the two other pairwise interactions:
    policy b  xsplit-b-s<seed>  trains on N, L, S, W, LW, SW   held out: LS and LSW
    policy c  xsplit-c-s<seed>  trains on N, L, S, W, LS, SW   held out: LW and LSW
  Is each of them within the margins of the specialist on its held-out combinations, as the
  policy is on SW and LSW?

WHAT IS READ (exact environment; names and paths from scripts/xsplit_runs.py)
  policies b and c   their arrays on all eight combinations at 10x25 in <results>/10x25_xsplit
                     (scripts/run_heldout_splits_eval.sh), and their training records
  specialist         xmask-solo-<R>-s<seed>: its stored array on R in 10x25_xmask
  policy             xmask-v1-s<seed>: its stored array on R in 10x25_xmask
  for the same seeds. PACT_SEEDS selects the seeds (a subset of 301 to 303); unset, the seeds
  whose two split runs are trained and evaluated (xsplit_runs.complete_seeds). Every input must
  exist with one row per test instance, and every split run must be trained (its log finished,
  32786 parameters, exact mask, the configuration of xmask-v1 except model_suffix and
  family_regimes), or nothing is computed.

QUANTITIES (d = 100 (first - second) / second per training seed and test instance, seed s
against seed s, as in section A of scripts/p88_family_inference.py)
  (a) policy b or c against the specialist, on every combination: positive d = the split
      policy's schedules are longer. The functions of p88 for the policy against the specialist
      (p88.parity_components), with the split policy's arrays in the place of the policy's: one
      default_rng(11) per split policy in the order N, L, S, W, LS, LW, SW, LSW, p88.component
      on its draws (mean, hierarchical 95% interval, Bonferroni interval over the eight
      combinations, seed-level t interval, verdicts at 0.5% and 1%, direction). Every d of a
      combination has the same shape, so the three policies draw the same seed and instance
      indices for a combination. Reported: the two held-out combinations of each split, and
      all eight in the supplement table. Joint verdicts p88.iut_block over the six training and
      the two held-out combinations of the split; pooled over them p88.pooled_block, the
      combinations of a base instance kept together.
  (b) the policy xmask-v1 against the specialist, p88.parity_components on the same seeds, with
      p88.iut_block and p88.pooled_block over p88.PAR_FAM (training combinations N, L, S, W, LS,
      LW; held out SW, LSW). When the record of p88 (results/family_inference.json) holds the
      same seeds, B and alpha, its section A must be reproduced (p117_long_specialists.
      record_check), else the JSON says which seeds the record holds.
  (c) policy b or c against the policy xmask-v1 of the same seed on the split's two held-out
      combinations: positive d = the split policy's schedules are longer. p88.component on the
      draws of a fresh default_rng(11) (p88.fresh), as p88 does for comparisons outside
      section A.
  "Within the margin" is the verdict 'equivalent' of stats_family.verdict: the interval lies in
  the band -m < lo, hi < m, m = 0.5% or 1%.
  ONE SEED. With k = 1 the hierarchical bootstrap draws the one seed in every replicate, so
  every interval covers test instances only, and the seed-level t interval is undefined (null
  in the JSON). The JSON (seeds, k) and the first line of the macro file say how many seeds
  entered.

OUTPUTS (arm_registry.out under PACT_OUT, default out/exact)
  manuscript/macros_heldout.tex    macro_io.emit; names Hsp<quantity>, letters only, checked
                                  against every macro defined in a .tex file under manuscript/;
                                  signed values with two decimals (p88 f2, for math mode),
                                  verdict words of stats_family.TEX. For each policy P (B, C,
                                  Main) and each of the eight combinations R, (a) or (b) as
                                  Hsp<P><R> with Lo, Hi, VHalf, VOne and Dir; pooled rows and
                                  joint verdicts over the training (Train), held-out (Held)
                                  and all eight (All) combinations; (c) as Hsp<P>VsMain<R> on
                                  the held-out combinations of B and C
  results/heldout_splits.json     every component, block and record check
  results/heldout_splits.md       the tables
CHECK (--check, read only): lists every missing or unfinished input of the selected seeds and
  exits 0 only when none is missing.
DRY (--dry [DIR]): the whole path on synthetic arrays and training records in DIR (default a
  temporary folder outside the repository, removed afterwards), with B = --boot (default 1000).

Usage: PACT_ENV=exact PACT_OUT=out/exact python scripts/p122_heldout_splits.py [--root DIR]
       ... python scripts/p122_heldout_splits.py --check
       python scripts/p122_heldout_splits.py --dry [DIR] [--boot 1000]
       PACT_SEEDS=301-303 (or --pact-seeds 301-303) selects the seeds; --root resolves the
       training records against DIR (default the repository).
"""
import argparse
import contextlib
import glob
import json
import math
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import longspec_runs as LR  # noqa: E402
import p117_long_specialists as P117  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import stats_family as SF  # noqa: E402
import xsplit_runs as X  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p122_heldout_splits.py'
PREFIX = 'Hsp'
OUT_FILES = {'macros': 'manuscript/macros_heldout.tex', 'json': 'results/heldout_splits.json',
             'md': 'results/heldout_splits.md'}
R8 = list(REG.COMBOS)
SPLITS = tuple(X.SPLITS)
WORD = {'b': 'B', 'c': 'C'}                # macro words of the split policies
MAIN_WORD = 'Main'
NAME = {'b': 'policy b', 'c': 'policy c'}  # names of the split policies in the macro comments
MAIN_NAME = 'the policy'
MARGINS = F88.MARGINS
MW = F88.MW
FOLDER = X.FOLDER
DRY_BOOT = 1000
Refused = X.Refused


class Incomplete(Refused):
    """An input of the selected seeds that is missing or not finished."""


def families(split):
    """The combination families of a split policy (training, held out, all eight)."""
    return {'Train': list(X.SPLITS[split]), 'Held': list(X.HELD[split]), 'All': R8}


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    X.require_exact()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')


def selected_seeds(root='.'):
    """PACT_SEEDS when set (arm_registry.seeds, then a subset of 301 to 303), else the seeds
    whose two split runs are complete."""
    if os.environ.get('PACT_SEEDS'):
        try:
            s = tuple(REG.seeds())
        except SystemExit as e:
            raise Refused(str(e))
        bad = [x for x in s if x not in X.SEEDS]
        if bad:
            raise Refused(f'PACT_SEEDS names the seeds {bad}; the split policies are trained for '
                          f'the seeds {X.SEEDS[0]} to {X.SEEDS[-1]} (pass --pact-seeds 301-303)')
        return s
    s = X.complete_seeds(root)
    if not s:
        raise Incomplete('PACT_SEEDS is unset and no seed has both split runs trained and '
                         'evaluated (scripts/xsplit_runs.py state)')
    return s


def missing_inputs(seeds, root='.'):
    """Every missing or unfinished input of the seeds, one line each."""
    miss = []
    for s in seeds:
        for x in SPLITS:
            nm = X.name(x, s)
            tp = X.training_problems(x, s, root)
            if tp:
                miss.append(f'{nm}: not trained: {tp[0]}' + (f' (and {len(tp) - 1} more)' if len(tp) > 1 else ''))
            ep = X.evaluation_problems(x, s)
            if ep:
                miss.append(f'{nm}: not evaluated: {ep[0]}' + (f' (and {len(ep) - 1} more)' if len(ep) > 1 else ''))
        for r in R8:
            n = X.n_test(r)
            for what, p in (('specialist', LR.standard_result(r, s)), ('policy', LR.policy_result(r, s))):
                for m in LR.rows_problems(n, [p]):
                    miss.append(f'{what} of seed {s} on {r}: {m}')
    return miss


# ----------------------------------------------------------------- loaders
def split_loader(split):
    """Makespans of the split policy of seed s on r (the signature of p88's loaders)."""
    X.check_split(split)

    def load(s, folder, r, n=None):
        if folder != FOLDER:
            raise Refused(f'the split policies are evaluated at {FOLDER} only, not {folder}')
        return REG.read_path(X.result_path(split, s, r), n)
    return load


# ----------------------------------------------------------------- quantities
def split_vs_specialist(split, seeds):
    """(a): ({R: d}, {R: component}) on one default_rng(STREAM_SEED) in the order of
    p88.parity_components, the split policy in the place of the policy."""
    rng = np.random.default_rng(F88.STREAM_SEED)
    load = split_loader(split)
    D, C = {}, {}
    for r in F88.TRAIN + F88.HELD:                       # the stream order of p88
        d = F88.seed_pairs(load, F88.specialist, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, SF.hier_draws_stream(d, F88.B, rng))
    return D, C


def split_vs_policy(split, seeds):
    """(c): ({R: d}, {R: component}) on the split's held-out combinations, fresh draws."""
    load = split_loader(split)
    D, C = {}, {}
    for r in X.HELD[split]:
        d = F88.seed_pairs(load, F88.released, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, F88.fresh(d))
    return D, C


def analyse(root='.', dry=False):
    """Every input, then (a), (b) and (c); returns the result dict. Raises Refused."""
    check_setting()
    seeds = selected_seeds(root)
    miss = missing_inputs(seeds, root)
    if miss:
        raise Incomplete(f'{len(miss)} inputs of seeds {",".join(map(str, seeds))} are missing or '
                         'unfinished:\n  ' + '\n  '.join(miss[:12])
                         + (f'\n  ... and {len(miss) - 12} more' if len(miss) > 12 else ''))
    F88.verify_bases(FOLDER, max(X.n_test(r) for r in R8))
    split = {}
    for x in SPLITS:
        D, C = split_vs_specialist(x, seeds)
        DM, CM = split_vs_policy(x, seeds)
        fam = families(x)
        split[x] = dict(name=f'xsplit-{x}', train=list(X.SPLITS[x]), held=list(X.HELD[x]),
                        against_specialist=C, iut=F88.iut_block(C, fam),
                        pooled=F88.pooled_block(D, fam), against_policy=CM)
    DS, CS = F88.parity_components(FOLDER, seeds=list(seeds))
    main = dict(name='xmask-v1', train=list(F88.TRAIN), held=list(F88.HELD), against_specialist=CS,
                iut=F88.iut_block(CS, F88.PAR_FAM), pooled=F88.pooled_block(DS, F88.PAR_FAM),
                record=P117.record_check(seeds, CS))
    held = [(x, r) for x in SPLITS for r in X.HELD[x]]
    counts = {str(mg): dict(equivalent=[f'{x}:{r}' for x, r in held
                                        if split[x]['against_specialist'][r][f'v{mg}'] == 'equivalent'],
                            non_inferior=[f'{x}:{r}' for x, r in held
                                          if split[x]['against_specialist'][r][f'v{mg}']
                                          in ('equivalent', 'non-inferior only')])
              for mg in MARGINS}
    behind = [f'{x}:{r}' for x, r in held if split[x]['against_specialist'][r]['direction'] == 'behind']
    return dict(generator=GENERATOR, setting=f'environment {X.ENV}, training seeds '
                f'{",".join(map(str, seeds))} ({len(seeds)} run(s) per policy)',
                seeds=list(seeds), k=len(seeds), dry=bool(dry), boot=F88.B, alpha=F88.ALPHA,
                one_seed_note=('one seed: every interval covers test instances only; the '
                               'seed-level t interval is undefined') if len(seeds) == 1 else None,
                streams=dict(against_specialist=f'one default_rng({F88.STREAM_SEED}) per policy in '
                                                f'the order {F88.TRAIN + F88.HELD} '
                                                '(p88.parity_components)',
                             against_policy=f'fresh default_rng({F88.FRESH_SEED}) per combination '
                                            '(p88.fresh); pooled rows p88.pooled_block'),
                definitions=dict(
                    against_specialist='100 (policy - specialist) / specialist per seed and test '
                                       'instance; positive = the policy is slower',
                    against_policy='100 (split policy - policy xmask-v1) / policy xmask-v1 per '
                                   'seed and test instance; positive = the split policy is slower',
                    within='verdict equivalent: -m < lo and hi < m'),
                split=split, main=main,
                held_out=dict(components=[f'{x}:{r}' for x, r in held], counts=counts,
                              behind=behind))


# ----------------------------------------------------------------- outputs
f2 = F88.f2
shown = P117.shown


def tex(v):
    return SF.TEX[v]


def comp_line(name, c, what):
    return (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
            f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
            f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
            + ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}' for mg in MARGINS)
            + f'\\newcommand{{\\{name}Dir}}{{{tex(c["direction"])}}}'
            + f'  % {what}: mean, hierarchical 95% CI, verdicts at 0.5% and 1%, direction')


def pool_lines(prefix, block, fams, what):
    out = []
    for fam in fams:
        x = block[fam]
        out.append(f'\\newcommand{{\\{prefix}{fam}}}{{{f2(x["mean"])}}}'
                   f'\\newcommand{{\\{prefix}{fam}Lo}}{{{f2(x["lo"])}}}'
                   f'\\newcommand{{\\{prefix}{fam}Hi}}{{{f2(x["hi"])}}}'
                   + ''.join(f'\\newcommand{{\\{prefix}{fam}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                             for mg in MARGINS)
                   + f'  % {what}, pooled over {x["c"]} combinations, base instances kept '
                     'together: mean, hierarchical 95% CI, verdicts')
    return out


def iut_lines(prefix, block, fams, what):
    out = []
    for fam in fams:
        for mg in MARGINS:
            for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                x = block[f'{fam}|{mg}|{claim}']
                out.append(f'\\newcommand{{\\{prefix}{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                           f'  % {what}: {claim} on all of {",".join(x["components"])} at '
                           f'+-{mg}% (1 = holds); failing {",".join(x["failing"]) or "none"}')
    return out


def seed_words(seeds):
    return ('seed ' if len(seeds) == 1 else 'seeds ') + ', '.join(map(str, seeds))


def specialist_lines(w, P, name):
    """Lines of one policy P against the specialists: Hsp<P><R> with Lo, Hi, VHalf, VOne and Dir
    for each of the eight combinations in the order of arm_registry.COMBOS, then the pooled rows
    and the joint verdicts over Train and Held, each followed by its row over All. A held-out
    combination keeps the comment that names the policy by its training combinations; every
    other line names the policy, its training combinations and the training seeds."""
    who = f'the policy trained on {", ".join(P["train"])}'
    L = []
    for r in R8:
        what = (f'{who} against the specialist on the held-out {r}' if r in P['held'] else
                f'{name}, against the specialist on the training combination {r}')
        L.append(comp_line(f'{w}{r}', P['against_specialist'][r], what))
    L += pool_lines(w, P['pooled'], ('Train', 'Held'), f'{who} against the specialists')
    L += pool_lines(w, P['pooled'], ('All',),
                    f'{name}, against the specialists on its {len(P["train"])} training and '
                    f'{len(P["held"])} held-out combinations')
    L += iut_lines(w, P['iut'], ('Train', 'Held'), f'{who} against the specialists')
    L += iut_lines(w, P['iut'], ('All',), f'{name}, against the specialists')
    return L


def macro_lines(res):
    k = res['k']
    sw = seed_words(res['seeds'])
    L = [f'% Generated by {GENERATOR}, training seeds {", ".join(map(str, res["seeds"]))} '
         f'({k} seed{"s" if k > 1 else ""}'
         + ('; every interval covers test instances only)' if k == 1 else
            '; intervals over training seeds and test instances)'),
         f'% Gaps in %, seed s against seed s, hierarchical 95% intervals of p88, B = {res["boot"]}. '
         'HspB<R>, HspC<R>: the policy trained on N, L, S, W, LW, SW (B) or on N, L, S, W, LS, SW '
         '(C) minus the specialist, over the specialist (positive = the policy is slower); '
         'HspMain<R>: the same for the policy trained on N, L, S, W, LS, LW; HspBVsMain<R>, '
         'HspCVsMain<R>: B or C minus the policy trained on N, L, S, W, LS, LW, over it.',
         f'\\newcommand{{\\{PREFIX}Seeds}}{{{k}}}\\newcommand{{\\{PREFIX}Runs}}{{{len(SPLITS) * k}}}'
         '  % training seeds, runs of the two policies with other held-out pairs']
    for x in SPLITS:
        S, w = res['split'][x], PREFIX + WORD[x]
        who = f'the policy trained on {", ".join(S["train"])}'
        L += specialist_lines(w, S, f'{NAME[x]}, trained on {", ".join(S["train"])} ({sw})')
        for r in S['held']:
            L.append(comp_line(f'{w}VsMain{r}', S['against_policy'][r],
                               f'{who} against the policy trained on '
                               f'{", ".join(res["main"]["train"])}, on {r}'))
    M, w = res['main'], PREFIX + MAIN_WORD
    L += specialist_lines(w, M, f'{MAIN_NAME}, trained on {", ".join(M["train"])} ({sw})')
    H = res['held_out']
    n = len(H['components'])
    L.append(f'\\newcommand{{\\{PREFIX}HeldN}}{{{n}}}'
             + ''.join(f'\\newcommand{{\\{PREFIX}HeldEq{MW[mg]}N}}{{{len(H["counts"][str(mg)]["equivalent"])}}}'
                       f'\\newcommand{{\\{PREFIX}HeldNi{MW[mg]}N}}{{{len(H["counts"][str(mg)]["non_inferior"])}}}'
                       for mg in MARGINS)
             + f'\\newcommand{{\\{PREFIX}HeldBehindN}}{{{len(H["behind"])}}}'
             + f'  % held-out combinations of B and C ({", ".join(H["components"])}): their number; '
               'equivalent and non-inferior at 0.5% ('
             + (', '.join(H['counts']['0.5']['equivalent']) or 'none') + '; '
             + (', '.join(H['counts']['0.5']['non_inferior']) or 'none') + ') and at 1% ('
             + (', '.join(H['counts']['1.0']['equivalent']) or 'none') + '; '
             + (', '.join(H['counts']['1.0']['non_inferior']) or 'none') + '); interval above zero ('
             + (', '.join(H['behind']) or 'none') + ')')
    return L


def defined_macros():
    """{name: file} of every macro defined in a .tex file under manuscript/, the output left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '**', '*.tex'), recursive=True)):
        if os.path.basename(f) == own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, os.path.relpath(f, ROOT))
    return out


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise Refused(f'duplicate macro names in the output: {dup[:5]}')
    have = defined_macros()
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def ci(x):
    return f'[{f2(x["lo"])}, {f2(x["hi"])}]'


def markdown(res):
    S, M = res['split'], res['main']
    md = [f'# Policies trained with other held-out pairs of combinations ({res["setting"]})', '']
    if res['one_seed_note']:
        md += [f'With {res["one_seed_note"]}.', '']
    if res['dry']:
        md += ['DRY RUN on synthetic arrays and training records.', '']
    md += [f'Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = {res["boot"]}). '
           'Policy b trains on ' + ', '.join(S['b']['train']) + '; policy c on '
           + ', '.join(S['c']['train']) + '; the policy (xmask-v1) on ' + ', '.join(M['train']) + '.',
           '', '## Each policy against the specialist on its held-out combinations', '',
           'Positive: the policy is slower than the specialist.', '',
           '| policy | held-out combination | mean | 95% CI | 0.5% | 1% | direction |',
           '|---|---|---|---|---|---|---|']
    rows = [(f'policy {x}', r, S[x]['against_specialist'][r]) for x in SPLITS for r in S[x]['held']]
    rows += [('policy (xmask-v1)', r, M['against_specialist'][r]) for r in M['held']]
    for who, r, c in rows:
        md.append(f'| {who} | {r} | {f2(c["mean"])} | {ci(c)} | {c["v0.5"]} | {c["v1.0"]} | '
                  f'{c["direction"]} |')
    md += ['', '## Pooled against the specialists', '',
           'Training and held-out combinations of each policy; base instances kept together.', '',
           '| policy | combinations | mean | 95% CI | 0.5% | 1% | equivalent on every one at 0.5% '
           '| at 1% |', '|---|---|---|---|---|---|---|---|']
    for who, P in [(f'policy {x}', S[x]) for x in SPLITS] + [('policy (xmask-v1)', M)]:
        for fam in ('Train', 'Held', 'All'):
            c = P['pooled'][fam]
            regs = {'Train': P['train'], 'Held': P['held'], 'All': R8}[fam]
            eq = [P['iut'][f'{fam}|{mg}|equivalent']['holds'] for mg in MARGINS]
            md.append(f'| {who} | {", ".join(regs)} | {f2(c["mean"])} | {ci(c)} | {c["v0.5"]} | '
                      f'{c["v1.0"]} | {"yes" if eq[0] else "no"} | {"yes" if eq[1] else "no"} |')
    md += ['', '## Policies b and c against the policy (xmask-v1) of the same seed', '',
           'Positive: the split policy is slower than the policy.', '',
           '| policy | combination | mean | 95% CI | 0.5% | 1% | direction |',
           '|---|---|---|---|---|---|---|']
    for x in SPLITS:
        for r in S[x]['held']:
            c = S[x]['against_policy'][r]
            md.append(f'| policy {x} | {r} | {f2(c["mean"])} | {ci(c)} | {c["v0.5"]} | {c["v1.0"]} '
                      f'| {c["direction"]} |')
    md += ['', '## Every combination (supplement table)', '',
           'Each policy against the specialist; held-out combinations marked with *.', '',
           '| combination | ' + ' | '.join(f'policy {x}: mean | 95% CI | 0.5% | 1%' for x in SPLITS)
           + ' | policy (xmask-v1): mean | 95% CI | 0.5% | 1% |',
           '|---|' + '---|---|---|---|' * (len(SPLITS) + 1)]
    for r in R8:
        cells = []
        for P in [S[x] for x in SPLITS] + [M]:
            c = P['against_specialist'][r]
            star = '*' if r in P['held'] else ''
            cells.append(f'{f2(c["mean"])}{star} | {ci(c)} | {c["v0.5"]} | {c["v1.0"]}')
        md.append(f'| {r} | ' + ' | '.join(cells) + ' |')
    H, rec = res['held_out'], M['record']
    md += ['', f'Held-out combinations of policies b and c: {", ".join(H["components"])}. '
               f'Equivalent at 0.5%: {", ".join(H["counts"]["0.5"]["equivalent"]) or "none"}; at 1%: '
               f'{", ".join(H["counts"]["1.0"]["equivalent"]) or "none"}. Interval above zero: '
               f'{", ".join(H["behind"]) or "none"}.', '',
           f'The policy (xmask-v1) against the specialist: p88.parity_components on the same seeds '
           f'({rec["status"]}: {rec["path"]}).']
    return md


def clean(x):
    """JSON-ready: numpy scalars to Python, non-finite floats to null."""
    x = F88.jsonable(x)

    def f(v):
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if isinstance(v, dict):
            return {k: f(w) for k, w in v.items()}
        if isinstance(v, list):
            return [f(w) for w in v]
        return v
    return f(x)


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(root='.', dry=False):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md, outs)."""
    check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, dry)
        lines = macro_lines(res)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(clean(res), f, indent=1, allow_nan=False)
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md, outs


def check(root='.'):
    """Read-only precondition of the chain step; returns the exit code."""
    X.require_exact()
    try:
        seeds = selected_seeds(root)
    except Incomplete as e:
        print(f'p122: {e}')
        for s in X.SEEDS:
            st = X.state((s,), root)
            print(f'p122:   seed {s}: {sum(x["complete"] for x in st)} of {len(st)} runs trained '
                  'and evaluated')
        return 1
    miss = missing_inputs(seeds, root)
    print(f'p122: seeds {",".join(map(str, seeds))}: {len(SPLITS) * len(seeds)} split runs on the '
          'eight combinations at 10x25, the specialists and the policy: '
          + ('every input present (arrays with one row per test instance, finished training logs)'
             if not miss else f'{len(miss)} inputs missing or unfinished'))
    for m in miss[:8]:
        print(f'p122:   {m}')
    if len(miss) > 8:
        print(f'p122:   ... and {len(miss) - 8} more')
    return 0 if not miss else 1


# ----------------------------------------------------------------- dry mode
# true mean differences (%) against the specialist: the policy, policy b and policy c
TRUE = {'main': {'N': 0.1, 'L': 0.2, 'S': -0.1, 'W': 0.4, 'LS': 0.0, 'LW': 0.3, 'SW': 0.1, 'LSW': 0.2},
        'b': {'N': 0.1, 'L': 0.2, 'S': -0.1, 'W': 0.4, 'LS': 0.6, 'LW': 0.2, 'SW': 0.0, 'LSW': 0.3},
        'c': {'N': 0.0, 'L': 0.3, 'S': -0.2, 'W': 0.5, 'LS': 0.1, 'LW': 1.6, 'SW': 0.1, 'LSW': 0.4}}
NOISE = 0.3
DRY_SEEDS = (301, 302)
DRY_NORM = 200.0
environment = P117.environment


def _put(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([x, np.zeros(len(x))], axis=1))


def synthetic_log(model, combos, updates, scores):
    """A training log in the format of train_family.py."""
    regs = ', '.join(f"'{r}'" for r in combos)
    out = [LR.MASK_LINE, '[vali] MWKR norms: ' + '  '.join(f'{r}={DRY_NORM:.1f}' for r in combos),
           f'[model] params total={X.PARAMS}  film=0 (0.00%)',
           f'[train_family] model={REG.PREFIX}{model} regimes=[{regs}] updates={updates}']
    best = float('inf')
    for i, s in enumerate(scores):
        best = min(best, s)
        out.append(f'[vali @{25 * (i + 1)}] rel={s:.4f} (best {best:.4f})')
    out.append(f'[train_family] done in 1.00 h; best rel={best:.4f}')
    return '\n'.join(out) + '\n'


def synthetic_records(tree, seed):
    """Configuration of xmask-v1-s<seed>, and configuration, log, validation record and checkpoint
    of the two split runs of the seed."""
    main = X.main_name(seed)
    cfg = dict(model_suffix=main, seed_train=int(seed), max_updates=X.UPDATES,
               family_regimes=','.join(REG.TRAIN_COMBOS), first_mask='exact', mask_ties='exact',
               device='cuda', lr=3e-4, num_envs=20)
    _put(os.path.join(tree, REG.config_path('released', None, seed, X.ENV)), json.dumps(cfg, indent=1))
    scores = [0.95 - 0.001 * i for i in range(80)]
    for x in SPLITS:
        p = X.paths(x, seed)
        c = dict(cfg, model_suffix=X.name(x, seed), family_regimes=X.regimes_arg(x))
        _put(os.path.join(tree, p['checkpoint']), '')
        _put(os.path.join(tree, p['config']), json.dumps(c, indent=1))
        _put(os.path.join(tree, p['vali']), str(scores))
        _put(os.path.join(tree, p['log']), synthetic_log(X.name(x, seed), X.SPLITS[x], X.UPDATES, scores))


def build_synthetic(tree, seeds=DRY_SEEDS):
    """Arrays of the specialist, the policy and the two split policies per seed and combination,
    and the training records."""
    rng = np.random.default_rng(2026)
    for s in seeds:
        for r in R8:
            n = X.n_test(r)
            spec = np.round(rng.uniform(400.0, 900.0, n) * (1 + rng.normal(0, 0.01, n)), 1)
            _save(LR.standard_result(r, s), spec)
            _save(LR.policy_result(r, s), spec * (1 + (TRUE['main'][r] + rng.normal(0, NOISE, n)) / 100.0))
            for x in SPLITS:
                _save(X.result_path(x, s, r), spec * (1 + (TRUE[x][r] + rng.normal(0, NOISE, n)) / 100.0))
        synthetic_records(tree, s)


def dry_paths(tree):
    return dict(PACT_ENV=X.ENV, PACT_SEEDS=None, PACT_RESULTS=os.path.join(tree, 'results'),
                PACT_OUT=os.path.join(tree, 'out'))


def dry_run(tree, boot=DRY_BOOT, seeds=DRY_SEEDS):
    """The whole path on a synthetic tree; returns (res, names, md, outs)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    F88.configure(boot=boot)
    with environment(**dry_paths(tree)):
        build_synthetic(tree, seeds)
        return run(root=tree, dry=True)


def main():
    ap = argparse.ArgumentParser(description='policies trained with other held-out pairs')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes '
                         f'B = {SF.B_DEFAULT}, the B of p88')
    ap.add_argument('--root', default='.', help='folder the training records are resolved against')
    a = ap.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('PACT_ENV', X.ENV)
    root = os.path.abspath(a.root) if a.root not in ('.', '') else '.'
    if a.dry is None:
        if a.boot is not None:
            ap.error('--boot is for the dry run; a real run uses the B of p88')
        os.environ.setdefault('PACT_OUT', 'out/exact')
        try:
            if a.check:
                sys.exit(check(root))
            REG.banner()
            res, names, md, outs = run(root)
        except Refused as e:
            sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
        print('\n'.join(md))
        print(f'\n{len(names)} macros; wrote ' + ', '.join(outs.values()))
        return
    if a.check:
        ap.error('--check and --dry exclude each other')
    boot = DRY_BOOT if a.boot is None else a.boot
    keep = bool(a.dry)
    tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p122_dry_')
    try:
        res, names, md, outs = dry_run(tree, boot)
    except Refused as e:
        sys.exit(f'p122 self-test on synthetic arrays FAILED: {e}')
    finally:
        if not keep:
            shutil.rmtree(tree, ignore_errors=True)
    if keep:
        print('\n'.join(md))
        print(f'\n{len(names)} macros; outputs in {os.path.join(tree, "out")}')
    else:
        h = res['held_out']['counts']
        print(f'p122 self-test on synthetic arrays passed: seeds {",".join(map(str, res["seeds"]))}, '
              f'B = {boot}, {len(names)} macros, held-out combinations equivalent at 0.5%: '
              f'{", ".join(h["0.5"]["equivalent"]) or "none"}; at 1%: '
              f'{", ".join(h["1.0"]["equivalent"]) or "none"}')


if __name__ == '__main__':
    main()
