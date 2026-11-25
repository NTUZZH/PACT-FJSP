r"""P117: specialists trained for the policy's number of updates.

QUESTION
  The policy trains for 2000 updates on six combinations; each specialist trains for 1000 updates
  on one combination. Does a specialist that trains for 2000 updates produce shorter schedules on
  the test instances of its combination than the 1000-update specialist of the same seed, and
  does the comparison of the policy with the specialist change when the longer-trained
  specialist is used?

WHAT IS READ (exact environment; names and paths from scripts/longspec_runs.py)
  longer-trained specialist  xlong-solo-<R>-s<seed>, 2000 updates: its array on R at 10x25 in
                             <results>/10x25_xlong (scripts/run_longspec_eval.sh) and its
                             training log
  specialist                 xmask-solo-<R>-s<seed>, 1000 updates: its stored array on R in
                             10x25_xmask and its training log
  policy                     xmask-v1-s<seed>: its stored array on R in 10x25_xmask
  for the same seeds. PACT_SEEDS selects the seeds; unset, the seeds whose eight longer-trained
  runs are trained and evaluated (longspec_runs.complete_seeds). Every input must exist with
  one row per test instance, and every training log must be finished, or nothing is computed.

QUANTITIES (d = 100 (first - second) / second per training seed and test instance, seed s
against seed s, as in section A of scripts/p88_family_inference.py)
  (a) longer-trained specialist against specialist, on R: positive d = the longer-trained
      specialist's schedules are longer. Per combination p88.component on the draws of a fresh
      default_rng(11) (p88.fresh; mean, hierarchical 95% interval, Bonferroni interval over the
      eight combinations, seed-level t interval, verdicts at 0.5% and 1%, direction); pooled
      over the eight combinations (and over the six training and the two held-out ones)
      p88.pooled_block, the combinations of a base instance kept together. "Within 0.5%": the
      interval lies inside the band -0.5 < lo, hi < 0.5 (the 'equivalent' rule of
      stats_family.verdict at 0.5).
  (b) policy against the longer-trained specialist, on R: positive d = the policy's schedules
      are longer. p88.component on the parity stream of p88.parity_components (one
      default_rng(11) in the order N, L, S, W, LS, LW, SW, LSW), the joint verdicts of
      p88.iut_block; beside it the same quantities against the specialist, from
      p88.parity_components on the same seeds. When the record of p88
      (arm_registry.out('results/family_inference.json')) holds the same seeds, B and alpha,
      the components against the specialist must equal its section A (mean and interval within
      1e-9, verdicts and direction exactly), else the script refuses; otherwise the JSON says
      that the record covers other seeds or is absent. The two streams draw the same seed and
      instance indices for a combination, because every d has the same shape.
  (c) from the training logs, read by p116_convergence.read_curve (validation every 25 updates;
      the score is the mean greedy validation makespan over the MWKR norm, lower is better; the
      log prints four decimals): the best score of each run and the first update at which it
      was reached (p116_convergence.run_statistics), for the longer-trained run also its best
      score up to update 1000, and the largest difference between the two runs' scores at
      updates 25 to 1000 with the first update at which they differ. The trainer reads
      max_updates only as its loop bound (train_family.py), so the two runs share seed, recipe
      and schedule of every coefficient; training on the GPU is not bit-reproducible, so the
      two curves can part after the first updates.
  ONE SEED. With k = 1 the hierarchical bootstrap draws the one seed in every replicate, so
  every interval covers test instances only, and the seed-level t interval is undefined (null
  in the JSON). The JSON (seeds, k) and the first line of the macro file say how many seeds
  entered.

OUTPUTS (arm_registry.out under PACT_OUT, default out/exact)
  manuscript/macros_longspec.tex   macro_io.emit; names Lsp<quantity>, letters only, checked
                                  against every macro defined in a .tex file under manuscript/;
                                  signed values with two decimals and their sign (p88 f2, for
                                  math mode), verdict words of stats_family.TEX
  results/long_specialists.json   every component, block, record check and training row
  results/long_specialists.md     the tables
CHECK (--check, read only): lists every missing or unfinished input of the selected seeds and
  exits 0 only when none is missing.
DRY (--dry [DIR]): the whole path on synthetic arrays and training records in DIR (default a
  temporary folder outside the repository, removed afterwards), with B = --boot (default 1000).

Usage: PACT_ENV=exact PACT_OUT=out/exact python scripts/p117_long_specialists.py [--root DIR]
       ... python scripts/p117_long_specialists.py --check
       python scripts/p117_long_specialists.py --dry [DIR] [--boot 1000]
       PACT_SEEDS=301 (or --pact-seeds 301) selects the seeds; --root resolves the training
       records against DIR (default the repository).
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
import p116_convergence as P116  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p117_long_specialists.py'
PREFIX = 'Lsp'
OUT_FILES = {'macros': 'manuscript/macros_longspec.tex', 'json': 'results/long_specialists.json',
             'md': 'results/long_specialists.md'}
RECORD = 'results/family_inference.json'
RECORD_GENERATOR = 'scripts/p88_family_inference.py'
R8 = list(REG.COMBOS)
FAM = F88.PAR_FAM
MARGINS = F88.MARGINS
MW = F88.MW
WITHIN = 0.5
FOLDER = LR.FOLDER
HALF = LR.STANDARD_UPDATES                 # the specialist's number of updates
RECORD_TOL = 1e-9
DRY_BOOT = 1000
Refused = LR.Refused


class Incomplete(Refused):
    """An input of the selected seeds that is missing or not finished."""


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    LR.require_exact()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')


def selected_seeds(root='.'):
    """PACT_SEEDS when set (arm_registry.seeds), else the seeds whose eight runs are complete."""
    if os.environ.get('PACT_SEEDS'):
        try:
            return tuple(REG.seeds())
        except SystemExit as e:
            raise Refused(str(e))
    s = LR.complete_seeds(root)
    if not s:
        raise Incomplete('PACT_SEEDS is unset and no seed has all eight longer-trained runs '
                         'trained and evaluated (scripts/longspec_runs.py state --seeds 301-305)')
    return s


def _at(root, p):
    return p if root in ('.', '') else os.path.join(root, p)


@contextlib.contextmanager
def p116_updates(updates):
    """p116's reader with the specialist's number of updates exchanged, restored afterwards."""
    old = P116.UPDATES['specialist']
    P116.UPDATES['specialist'] = int(updates)
    try:
        yield
    finally:
        P116.UPDATES['specialist'] = old


def read_run(kind, combo, seed, root='.'):
    """(updates, curve, best) of the longer-trained run ('long') or the specialist ('standard')
    from its training log; P116.Incomplete or P116.Refused name the run."""
    if kind == 'long':
        nm, log, u = LR.name(combo, seed), LR.train_log(combo, seed), LR.UPDATES
    else:
        nm, log, u = LR.standard_name(combo, seed), REG.train_log('specialist', combo, seed, LR.ENV), HALF
    with p116_updates(u):
        return P116.read_curve(_at(root, log), 'specialist', combo, nm)


def missing_inputs(seeds, root='.'):
    """Every missing or unfinished input of the seeds, one line each; a defective log
    (P116.Refused that is not Incomplete) raises Refused."""
    miss = []
    for s in seeds:
        for r in R8:
            nm = LR.name(r, s)
            tp = LR.training_problems(r, s, root)
            if tp:
                miss.append(f'{nm}: not trained: {tp[0]}' + (f' (and {len(tp) - 1} more)' if len(tp) > 1 else ''))
            ep = LR.evaluation_problems(r, s)
            if ep:
                miss.append(f'{nm}: not evaluated: {ep[0]}')
            n = LR.n_test(r)
            for what, p in (('specialist', LR.standard_result(r, s)),
                            ('policy', LR.policy_result(r, s))):
                for x in LR.rows_problems(n, [p]):
                    miss.append(f'{what} of seed {s} on {r}: {x}')
            for kind in ('long', 'standard'):
                if kind == 'long' and tp:
                    continue
                try:
                    read_run(kind, r, s, root)
                except P116.Incomplete as e:
                    miss.append(str(e))
                except P116.Refused as e:
                    raise Refused(f'defective training log: {e}')
    return miss


# ----------------------------------------------------------------- loaders
def long_specialist(s, folder, r, n=None):
    """Makespans of the longer-trained specialist of seed s on its combination r (the signature
    of p88's loaders)."""
    if folder != FOLDER:
        raise Refused(f'the longer-trained specialists are evaluated at {FOLDER} only, not {folder}')
    return REG.read_path(LR.result_path(r, s), n)


# ----------------------------------------------------------------- quantities
def long_vs_standard(seeds):
    """(a): ({R: d}, {R: component}, pooled block)."""
    D, C = {}, {}
    for r in R8:
        d = F88.seed_pairs(long_specialist, F88.specialist, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, F88.fresh(d))
    return D, C, F88.pooled_block(D, FAM)


def policy_vs_long(seeds):
    """(b): ({R: d}, {R: component}) on the parity stream of p88 (one default_rng(STREAM_SEED)
    in the order of p88.parity_components)."""
    rng = np.random.default_rng(F88.STREAM_SEED)
    D, C = {}, {}
    for r in F88.TRAIN + F88.HELD:
        d = F88.seed_pairs(F88.released, long_specialist, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, SF.hier_draws_stream(d, F88.B, rng))
    return D, C


def record_check(seeds, C_std):
    """Compare the components against the specialist with section A of the p88 record when it
    holds the same seeds, B and alpha; refuses on a difference."""
    p = REG.out(RECORD)
    out = dict(path=shown(p))
    if not os.path.exists(p):
        return dict(out, status='absent', compared=False)
    rec = json.load(open(p))
    if rec.get('generator') != RECORD_GENERATOR or not str(rec.get('setting', '')).startswith(
            f'environment {LR.ENV},'):
        return dict(out, status='not a p88 record of the exact environment', compared=False)
    if [int(x) for x in rec.get('seeds', [])] != [int(x) for x in seeds]:
        return dict(out, status=f'the record holds seeds {rec.get("seeds")}', compared=False)
    if int(rec.get('boot', -1)) != F88.B or float(rec.get('alpha', -1)) != F88.ALPHA:
        return dict(out, status=f'the record was computed with B = {rec.get("boot")}, alpha = '
                                f'{rec.get("alpha")}', compared=False)
    st = rec['parity']['components']
    bad, dmax = [], 0.0
    for r in R8:
        for k in ('mean', 'lo', 'hi'):
            dmax = max(dmax, abs(float(C_std[r][k]) - float(st[r][k])))
            if abs(float(C_std[r][k]) - float(st[r][k])) > RECORD_TOL:
                bad.append(f'{r} {k}: {C_std[r][k]!r} here, {st[r][k]!r} in the record')
        for k in ('v0.5', 'v1.0', 'direction'):
            if C_std[r][k] != st[r][k]:
                bad.append(f'{r} {k}: {C_std[r][k]!r} here, {st[r][k]!r} in the record')
    if bad:
        raise Refused(f'the policy against the specialist does not reproduce {shown(p)}: '
                      + '; '.join(bad[:6]))
    return dict(out, status='equal to the record', compared=True, largest_difference=dmax)


def training_rows(seeds, root='.'):
    """(c) per run."""
    rows = []
    for s in seeds:
        for r in R8:
            ul, cl, bl = read_run('long', r, s, root)
            us, cs, bs = read_run('standard', r, s, root)
            L, S = P116.run_statistics(cl, ul), P116.run_statistics(cs, us)
            early = [(u, x) for u, x in cl if u <= HALF]
            e_best = min(x for _, x in early)
            e_at = min(u for u, x in early if x == e_best)
            sd = dict(cs)
            common = [(u, x, sd[u]) for u, x in early if u in sd]
            diff = [abs(x - y) for _, x, y in common]
            first = next((u for u, x, y in common if x != y), None)
            rows.append(dict(combo=r, seed=int(s), name=LR.name(r, s),
                             standard=LR.standard_name(r, s),
                             long_best=float(min(x for _, x in cl)), long_best_update=L['best_update'],
                             long_best_first_half=float(e_best), long_best_first_half_update=int(e_at),
                             standard_best=float(min(x for _, x in cs)),
                             standard_best_update=S['best_update'],
                             best_after_half=bool(L['best_update'] > HALF),
                             best_change=float(100.0 * (min(x for _, x in cl) / min(x for _, x in cs) - 1.0)),
                             curves_compared=len(common), curves_largest_difference=float(max(diff)),
                             curves_first_difference=first,
                             long_late_change=L['late_change'], standard_late_change=S['late_change']))
    return rows


def analyse(root='.', dry=False):
    """Every input, then (a), (b) and (c); returns the result dict. Raises Refused."""
    check_setting()
    seeds = selected_seeds(root)
    miss = missing_inputs(seeds, root)
    if miss:
        raise Incomplete(f'{len(miss)} inputs of seeds {",".join(map(str, seeds))} are missing or '
                         'unfinished:\n  ' + '\n  '.join(miss[:12])
                         + (f'\n  ... and {len(miss) - 12} more' if len(miss) > 12 else ''))
    F88.verify_bases(FOLDER, max(LR.n_test(r) for r in R8))
    DA, CA, PA = long_vs_standard(seeds)
    DB, CB = policy_vs_long(seeds)
    DS, CS = F88.parity_components(FOLDER, seeds=list(seeds))
    rec = record_check(seeds, CS)
    iut_long, iut_std = F88.iut_block(CB, FAM), F88.iut_block(CS, FAM)
    pool_long, pool_std = F88.pooled_block(DB, FAM), F88.pooled_block(DS, FAM)
    train = training_rows(seeds, root)
    means = {r: CA[r]['mean'] for r in R8}
    hi_c = max(R8, key=lambda r: (means[r], -R8.index(r)))
    lo_c = min(R8, key=lambda r: (means[r], R8.index(r)))
    within = [r for r in R8 if -WITHIN < CA[r]['lo'] and CA[r]['hi'] < WITHIN]
    ahead = [r for r in R8 if CA[r]['direction'] == 'ahead']
    changed = {mg: [r for r in R8 if CB[r][f'v{mg}'] != CS[r][f'v{mg}']] for mg in MARGINS}
    return dict(generator=GENERATOR, setting=f'environment {LR.ENV}, training seeds '
                f'{",".join(map(str, seeds))} ({len(seeds)} run(s) per combination)',
                seeds=list(seeds), k=len(seeds), dry=bool(dry), boot=F88.B, alpha=F88.ALPHA,
                one_seed_note=('one seed: every interval covers test instances only; the '
                               'seed-level t interval is undefined') if len(seeds) == 1 else None,
                streams=dict(long_vs_standard=f'fresh default_rng({F88.FRESH_SEED}) per combination '
                                              'and per pooled row (p88.fresh, p88.pooled_block)',
                             policy=f'one default_rng({F88.STREAM_SEED}) in the order '
                                    f'{F88.TRAIN + F88.HELD} for each of the two comparisons '
                                    '(p88.parity_components)'),
                updates=dict(longer_trained=LR.UPDATES, specialist=HALF),
                definitions=dict(
                    long_vs_standard='100 (longer-trained specialist - specialist) / specialist '
                                     'per seed and test instance; positive = longer schedules '
                                     'after 2000 updates',
                    policy='100 (policy - specialist) / specialist per seed and test instance; '
                           'positive = the policy is slower',
                    within=f'-{WITHIN} < lo and hi < {WITHIN}'),
                long_vs_standard=dict(components=CA, pooled=PA, largest=dict(combo=hi_c, mean=means[hi_c]),
                                      smallest=dict(combo=lo_c, mean=means[lo_c]),
                                      within_half=within, shorter=ahead),
                policy=dict(against_longer_trained=CB, against_specialist=CS,
                            iut_longer_trained=iut_long, iut_specialist=iut_std,
                            pooled_longer_trained=pool_long, pooled_specialist=pool_std,
                            verdict_changes={str(mg): v for mg, v in changed.items()},
                            record=rec),
                training=dict(runs=train,
                              best_after_half=[x['name'] for x in train if x['best_after_half']]))


# ----------------------------------------------------------------- outputs
def shown(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def f2(x):
    return F88.f2(x)


def tex(v):
    return SF.TEX[v]


def tex_list(rs):
    """Combinations for running text: '$N$', '$N$ and $W$', '$N$, $W$ and $SW$'; 'none' when empty."""
    items = [f'${r}$' for r in rs]
    if not items:
        return 'none'
    if len(items) == 1:
        return items[0]
    return ', '.join(items[:-1]) + ' and ' + items[-1]


def macro_lines(res):
    k = res['k']
    A, P = res['long_vs_standard'], res['policy']
    L = [f'% Generated by {GENERATOR}, training seeds {", ".join(map(str, res["seeds"]))} '
         f'({k} seed{"s" if k > 1 else ""}'
         + ('; every interval covers test instances only)' if k == 1 else
            '; intervals over training seeds and test instances)'),
         f'% Gaps in %, seed s against seed s, hierarchical 95% intervals of p88, B = {res["boot"]}. '
         'Lsp<R>: specialist trained for 2000 updates minus the specialist (1000 updates), over '
         'the specialist (positive = longer schedules after 2000 updates); LspPol<R>: the policy '
         'minus the specialist trained for 2000 updates, over it (positive = the policy is '
         'slower); LspStd<R>: the verdicts of the policy against the specialist (1000 updates).',
         f'\\newcommand{{\\{PREFIX}Seeds}}{{{k}}}\\newcommand{{\\{PREFIX}Runs}}{{{8 * k}}}'
         '  % training seeds, runs of the specialists trained for 2000 updates']
    pa = A['pooled']['All']
    L.append(f'\\newcommand{{\\{PREFIX}Pool}}{{{f2(pa["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}PoolLo}}{{{f2(pa["lo"])}}}'
             f'\\newcommand{{\\{PREFIX}PoolHi}}{{{f2(pa["hi"])}}}'
             '  % pooled over the eight combinations, base instances kept together')
    hi, lo = A['largest'], A['smallest']
    L.append(f'\\newcommand{{\\{PREFIX}Max}}{{{f2(hi["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}MaxCombo}}{{{hi["combo"]}}}'
             f'\\newcommand{{\\{PREFIX}Min}}{{{f2(lo["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}MinCombo}}{{{lo["combo"]}}}'
             '  % largest and smallest per-combination difference, and its combination')
    L.append(f'\\newcommand{{\\{PREFIX}WithinHalfN}}{{{len(A["within_half"])}}}'
             f'\\newcommand{{\\{PREFIX}ShorterN}}{{{len(A["shorter"])}}}'
             f'\\newcommand{{\\{PREFIX}ShorterCombos}}{{{", ".join(A["shorter"]) or "none"}}}'
             f'  % combinations whose interval lies within +-{WITHIN}% '
             f'({", ".join(A["within_half"]) or "none"}); whose interval lies below zero')
    ch = P['verdict_changes']
    L.append(f'\\newcommand{{\\{PREFIX}VerdictChangeHalfN}}{{{len(ch["0.5"])}}}'
             f'\\newcommand{{\\{PREFIX}VerdictChangeOneN}}{{{len(ch["1.0"])}}}'
             '  % combinations whose verdict of the policy differs between the two specialists, '
             f'at 0.5% ({", ".join(ch["0.5"]) or "none"}) and at 1% ({", ".join(ch["1.0"]) or "none"})')
    L.append(f'\\newcommand{{\\{PREFIX}PoolDir}}{{{tex(pa["direction"])}}}'
             '  % direction of the pooled interval: ahead = shorter schedules after 2000 updates')
    longer = [r for r in R8 if A['components'][r]['direction'] == 'behind']
    nd = [r for r in R8 if A['components'][r]['direction'] == 'neither']
    L.append(f'\\newcommand{{\\{PREFIX}LongerN}}{{{len(longer)}}}'
             f'\\newcommand{{\\{PREFIX}LongerCombos}}{{{", ".join(longer) or "none"}}}'
             f'\\newcommand{{\\{PREFIX}NdN}}{{{len(nd)}}}'
             f'\\newcommand{{\\{PREFIX}NdCombos}}{{{", ".join(nd) or "none"}}}'
             '  % combinations whose interval lies above zero (longer schedules after 2000 '
             'updates); whose interval contains zero')
    pp = P['pooled_longer_trained']['All']
    L.append(f'\\newcommand{{\\{PREFIX}PolPool}}{{{f2(pp["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}PolPoolLo}}{{{f2(pp["lo"])}}}'
             f'\\newcommand{{\\{PREFIX}PolPoolHi}}{{{f2(pp["hi"])}}}'
             + ''.join(f'\\newcommand{{\\{PREFIX}PolPoolV{MW[mg]}}}{{{tex(pp[f"v{mg}"])}}}'
                       for mg in MARGINS)
             + f'\\newcommand{{\\{PREFIX}PolPoolDir}}{{{tex(pp["direction"])}}}'
             + '  % the policy against the specialists trained for 2000 updates, pooled over the '
               'eight combinations: mean, interval, verdicts at 0.5% and 1%, direction')
    one = {r: P['against_longer_trained'][r]['v1.0'] for r in R8}
    eq = [r for r in R8 if one[r] == 'equivalent']
    inf = [r for r in R8 if one[r] == 'inferior']
    L.append(f'\\newcommand{{\\{PREFIX}PolEquivOneN}}{{{len(eq)}}}'
             f'\\newcommand{{\\{PREFIX}PolInferiorOneN}}{{{len(inf)}}}'
             f'\\newcommand{{\\{PREFIX}PolInferiorOneCombos}}{{{", ".join(inf) or "none"}}}'
             '  % combinations on which the policy is equivalent at 1% to the specialist trained '
             f'for 2000 updates ({", ".join(eq) or "none"}), and on which it is inferior')
    for fam in ('Train', 'Held'):
        x, y = A['pooled'][fam], P['pooled_longer_trained'][fam]
        L.append(f'\\newcommand{{\\{PREFIX}Pool{fam}}}{{{f2(x["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pool{fam}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pool{fam}Hi}}{{{f2(x["hi"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pool{fam}Dir}}{{{tex(x["direction"])}}}'
                 f'\\newcommand{{\\{PREFIX}PolPool{fam}}}{{{f2(y["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}PolPool{fam}Lo}}{{{f2(y["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}PolPool{fam}Hi}}{{{f2(y["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}PolPool{fam}V{MW[mg]}}}{{{tex(y[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}PolPool{fam}Dir}}{{{tex(y["direction"])}}}'
                 + f'  % pooled over the {"six training" if fam == "Train" else "two held-out"} '
                   'combinations: 2000 against 1000 updates (mean, interval, direction); the '
                   'policy against the specialists trained for 2000 updates (mean, interval, '
                   'verdicts at 0.5% and 1%, direction)')
    ps = P['pooled_specialist']['All']
    L.append(f'\\newcommand{{\\{PREFIX}StdPool}}{{{f2(ps["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}StdPoolLo}}{{{f2(ps["lo"])}}}'
             f'\\newcommand{{\\{PREFIX}StdPoolHi}}{{{f2(ps["hi"])}}}'
             + ''.join(f'\\newcommand{{\\{PREFIX}StdPoolV{MW[mg]}}}{{{tex(ps[f"v{mg}"])}}}'
                       for mg in MARGINS)
             + '  % the policy against the specialists (1000 updates) on the same seeds, pooled '
               'over the eight combinations: mean, interval, verdicts at 0.5% and 1%')
    L.append(f'\\newcommand{{\\{PREFIX}ShorterList}}{{{tex_list(A["shorter"])}}}'
             f'\\newcommand{{\\{PREFIX}LongerList}}{{{tex_list(longer)}}}'
             f'\\newcommand{{\\{PREFIX}NdList}}{{{tex_list(nd)}}}'
             f'\\newcommand{{\\{PREFIX}PolEquivOneList}}{{{tex_list(eq)}}}'
             f'\\newcommand{{\\{PREFIX}PolInferiorOneList}}{{{tex_list(inf)}}}'
             f'\\newcommand{{\\{PREFIX}VerdictChangeHalfList}}{{{tex_list(ch["0.5"])}}}'
             f'\\newcommand{{\\{PREFIX}VerdictChangeOneList}}{{{tex_list(ch["1.0"])}}}'
             '  % the lists above for running text: shorter, longer and indistinguishable after '
             '2000 updates; the policy equivalent and inferior at 1%; verdicts that differ at '
             '0.5% and at 1%')
    part = [x['curves_first_difference'] for x in res['training']['runs']
            if x['curves_first_difference'] is not None]
    L.append(f'\\newcommand{{\\{PREFIX}PartN}}{{{len(part)}}}'
             f'\\newcommand{{\\{PREFIX}PartFirstMin}}{{{min(part) if part else 0}}}'
             f'\\newcommand{{\\{PREFIX}PartFirstMax}}{{{max(part) if part else 0}}}'
             '  % runs whose validation curve differs from that of the specialist of the same '
             'seed at some update up to 1000; the first such update, smallest and largest over '
             'these runs (0 when no curve differs)')
    late = res['training']['best_after_half']
    L.append(f'\\newcommand{{\\{PREFIX}BestLateN}}{{{len(late)}}}'
             f'  % runs of 2000 updates whose best validation score came after update {HALF}')
    for r in R8:
        a, b, s = A['components'][r], P['against_longer_trained'][r], P['against_specialist'][r]
        L.append(f'\\newcommand{{\\{PREFIX}{r}}}{{{f2(a["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Lo}}{{{f2(a["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Hi}}{{{f2(a["hi"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pol{r}}}{{{f2(b["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pol{r}Lo}}{{{f2(b["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}Pol{r}Hi}}{{{f2(b["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}Pol{r}V{MW[mg]}}}{{{tex(b[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + ''.join(f'\\newcommand{{\\{PREFIX}Std{r}V{MW[mg]}}}{{{tex(s[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}Std{r}}}{{{f2(s["mean"])}}}'
                 + f'\\newcommand{{\\{PREFIX}Std{r}Lo}}{{{f2(s["lo"])}}}'
                 + f'\\newcommand{{\\{PREFIX}Std{r}Hi}}{{{f2(s["hi"])}}}'
                 + f'\\newcommand{{\\{PREFIX}{r}Dir}}{{{tex(a["direction"])}}}'
                 + f'\\newcommand{{\\{PREFIX}Pol{r}Dir}}{{{tex(b["direction"])}}}'
                 + f'  % {r}: 2000 against 1000 updates, mean and interval; the policy against '
                   'the specialist trained for 2000 updates, mean, interval, verdicts at 0.5% and '
                   '1%; the verdicts against the specialist (1000 updates) and its mean and '
                   'interval; the directions of the first two intervals')
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
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def ci(x):
    return f'[{f2(x["lo"])}, {f2(x["hi"])}]'


def markdown(res):
    A, P, T = res['long_vs_standard'], res['policy'], res['training']
    md = [f'# Specialists trained for 2000 updates ({res["setting"]})', '']
    if res['one_seed_note']:
        md += [f'With {res["one_seed_note"]}.', '']
    if res['dry']:
        md += ['DRY RUN on synthetic arrays and training records.', '']
    md += [f'Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = {res["boot"]}).',
           '', '## Specialist trained for 2000 updates against the specialist (1000 updates)', '',
           'Positive: longer schedules after 2000 updates.', '',
           '| combination | mean | 95% CI | within 0.5% | direction |', '|---|---|---|---|---|']
    for r in R8:
        c = A['components'][r]
        md.append(f'| {r} | {f2(c["mean"])} | {ci(c)} | {"yes" if r in A["within_half"] else "no"} '
                  f'| {c["direction"]} |')
    for fam in FAM:
        c = A['pooled'][fam]
        md.append(f'| pooled, {fam} | {f2(c["mean"])} | {ci(c)} | '
                  f'{"yes" if -WITHIN < c["lo"] and c["hi"] < WITHIN else "no"} | {c["direction"]} |')
    rec = P['record']
    md += ['', '## The policy against each specialist', '',
           'Positive: the policy is slower. Against the specialist (1000 updates): '
           f'p88.parity_components on the same seeds ({rec["status"]}: {rec["path"]}).', '',
           '| combination | 2000 updates: mean | 95% CI | 0.5% | 1% | 1000 updates: mean | 95% CI '
           '| 0.5% | 1% |', '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        b, s = P['against_longer_trained'][r], P['against_specialist'][r]
        md.append(f'| {r} | {f2(b["mean"])} | {ci(b)} | {b["v0.5"]} | {b["v1.0"]} | '
                  f'{f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} |')
    for fam in FAM:
        b, s = P['pooled_longer_trained'][fam], P['pooled_specialist'][fam]
        md.append(f'| pooled, {fam} | {f2(b["mean"])} | {ci(b)} | {b["v0.5"]} | {b["v1.0"]} | '
                  f'{f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} |')
    ch = P['verdict_changes']
    md += ['', f'Verdicts that differ: at 0.5% {", ".join(ch["0.5"]) or "none"}; at 1% '
               f'{", ".join(ch["1.0"]) or "none"}.', '',
           '## Validation score during training', '',
           'Score: mean greedy validation makespan over the MWKR norm, lower is better (four '
           'decimals in the log). Curves compared at updates 25 to 1000.', '',
           '| run | best (2000 updates) | at update | best up to 1000 | at update | best (1000 '
           'updates) | at update | change of best, % | largest difference of the curves | '
           'first update that differs |', '|---|---|---|---|---|---|---|---|---|---|']
    for x in T['runs']:
        md.append(f'| {x["name"]} | {x["long_best"]:.4f} | {x["long_best_update"]} | '
                  f'{x["long_best_first_half"]:.4f} | {x["long_best_first_half_update"]} | '
                  f'{x["standard_best"]:.4f} | {x["standard_best_update"]} | {f2(x["best_change"])} | '
                  f'{x["curves_largest_difference"]:.4f} | {x["curves_first_difference"] or "none"} |')
    md += ['', f'Runs of 2000 updates whose best score came after update {HALF}: '
               f'{len(T["best_after_half"])} of {len(T["runs"])}.']
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
    LR.require_exact()
    try:
        seeds = selected_seeds(root)
    except Incomplete as e:
        print(f'p117: {e}')
        for line in [f'seed {s}: {sum(x["complete"] for x in LR.state((s,), root))} of 8 runs '
                     'trained and evaluated' for s in LR.SEEDS]:
            print(f'p117:   {line}')
        return 1
    miss = missing_inputs(seeds, root)
    print(f'p117: seeds {",".join(map(str, seeds))}: {8 * len(seeds)} runs of 2000 updates, the '
          'specialists (1000 updates) and the policy on each combination at 10x25: '
          + ('every input present (arrays with one row per test instance, finished training logs)'
             if not miss else f'{len(miss)} inputs missing or unfinished'))
    for m in miss[:8]:
        print(f'p117:   {m}')
    if len(miss) > 8:
        print(f'p117:   ... and {len(miss) - 8} more')
    return 0 if not miss else 1


# ----------------------------------------------------------------- dry mode
TRUE_A = {'N': 0.0, 'L': -0.2, 'S': 0.1, 'W': -0.9, 'LS': 0.0, 'LW': 0.2, 'SW': -0.1, 'LSW': 0.1}
TRUE_B = {'N': 0.1, 'L': 0.3, 'S': -0.2, 'W': 0.6, 'LS': -0.1, 'LW': 0.4, 'SW': -0.3, 'LSW': 0.0}
NOISE = 0.3
DRY_SEEDS = (301, 302)
DRY_NORM = 200.0


@contextlib.contextmanager
def environment(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _put(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([x, np.zeros(len(x))], axis=1))


def synthetic_log(model, combo, updates, scores):
    """A training log in the format of train_family.py."""
    out = [LR.MASK_LINE, f'[vali] MWKR norms: {combo}={DRY_NORM:.1f}',
           f'[model] params total={LR.PARAMS}  film=0 (0.00%)',
           f"[train_family] model={REG.PREFIX}{model} regimes=['{combo}'] updates={updates}"]
    best = float('inf')
    for k, s in enumerate(scores):
        best = min(best, s)
        out.append(f'[vali @{25 * (k + 1)}] rel={s:.4f} (best {best:.4f})  {combo}={s * DRY_NORM:.1f}')
    out.append(f'[train_family] done in 1.00 h; best rel={best:.4f}')
    return '\n'.join(out) + '\n'


def synthetic_records(tree, combo, seed, rng):
    """Configuration, log, validation record and checkpoint of the specialist and of the run of
    2000 updates; their curves agree up to update 1000 for seed 301 and part after it."""
    std, lng = LR.standard_name(combo, seed), LR.name(combo, seed)
    sp, lp = LR.standard_paths(combo, seed), LR.paths(combo, seed)
    cfg = dict(model_suffix=std, seed_train=int(seed), max_updates=HALF, family_regimes=combo,
               first_mask='exact', mask_ties='exact', device='cuda', lr=3e-4, num_envs=20)
    s = np.round(0.95 - 0.06 * (1 - np.exp(-np.arange(1, 81) / 25.0))
                 + rng.normal(0, 0.002, 80), 4)
    s2 = s.copy()
    if seed != DRY_SEEDS[0]:
        s2[10:] = np.round(s2[10:] + rng.normal(0, 0.002, 70), 4)
    for p, c, sc, upd in ((sp, cfg, s[:40], HALF),
                          (lp, dict(cfg, model_suffix=lng, max_updates=LR.UPDATES), s2, LR.UPDATES)):
        _put(os.path.join(tree, p['checkpoint']), '')
        _put(os.path.join(tree, p['config']), json.dumps(c, indent=1))
        _put(os.path.join(tree, p['vali']), str([float(x) for x in sc]))
        _put(os.path.join(tree, p['log']), synthetic_log(c['model_suffix'], combo, upd, sc))


def build_synthetic(tree, seeds=DRY_SEEDS):
    """Arrays of the three runs per seed and combination, and the training records."""
    rng = np.random.default_rng(2026)
    for s in seeds:
        for r in R8:
            n = LR.n_test(r)
            spec = np.round(rng.uniform(400.0, 900.0, n) * (1 + rng.normal(0, 0.01, n)), 1)
            lng = spec * (1 + (TRUE_A[r] + rng.normal(0, NOISE, n)) / 100.0)
            pol = spec * (1 + (TRUE_B[r] + rng.normal(0, NOISE, n)) / 100.0)
            _save(LR.standard_result(r, s), spec)
            _save(LR.policy_result(r, s), pol)
            _save(LR.result_path(r, s), lng)
            synthetic_records(tree, r, s, rng)


def dry_paths(tree):
    return dict(PACT_ENV=LR.ENV, PACT_SEEDS=None, PACT_RESULTS=os.path.join(tree, 'results'),
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
    ap = argparse.ArgumentParser(description='specialists trained for 2000 updates')
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
    os.environ.setdefault('PACT_ENV', LR.ENV)
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
    tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p117_dry_')
    try:
        res, names, md, outs = dry_run(tree, boot)
    except Refused as e:
        sys.exit(f'p117 self-test on synthetic arrays FAILED: {e}')
    finally:
        if not keep:
            shutil.rmtree(tree, ignore_errors=True)
    if keep:
        print('\n'.join(md))
        print(f'\n{len(names)} macros; outputs in {os.path.join(tree, "out")}')
    else:
        print(f'p117 self-test on synthetic arrays passed: seeds {",".join(map(str, res["seeds"]))}, '
              f'B = {boot}, {len(names)} macros, pooled difference {f2(res["long_vs_standard"]["pooled"]["All"]["mean"])}% '
              f'{ci(res["long_vs_standard"]["pooled"]["All"])}')


if __name__ == '__main__':
    main()
