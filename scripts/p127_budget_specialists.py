r"""P127: specialists that share the policy's training budget.

QUESTION
  The policy trains once, for 2000 updates on six combinations. The principal comparison sets it
  against eight specialists of 1000 updates each, four times the policy's training in total.
  Eight specialists of 250 updates each train for as many updates and episodes as the policy.
  On the test instances of each combination, are the policy's schedules shorter than or equal to
  those of the specialist that trains for its share of the same budget, and how much do the
  other 750 updates shorten the schedules of a specialist?

WHAT IS READ (exact environment; names and paths from scripts/budgetspec_runs.py)
  budget-sharing specialist  xshare-solo-<R>-s<seed>, 250 updates: its array on R at 10x25 in
                             <results>/10x25_xshare (scripts/run_budgetspec_eval.sh), its
                             training log and its configuration
  specialist                 xmask-solo-<R>-s<seed>, 1000 updates: its stored array on R in
                             10x25_xmask, its training log and its configuration
  policy                     xmask-v1-s<seed>: its stored array on R in 10x25_xmask and its
                             configuration
  for the same seeds. The seeds are this script's own setting, --seeds (default 301 to 310, the
  ten seeds of the principal comparison; any subset of them is accepted); PACT_SEEDS is not
  read. For a seed 306 to 310 every name, path and array of the specialist and of the policy
  comes from scripts/seed_extension.py (the registry plans the seeds 301 to 305 only), as in
  scripts/p110_seed_extension.py. Every array must exist with one row per test instance, every
  run of 250 updates must be trained (budgetspec_runs.training_problems), every training log
  must be finished (p116_convergence.read_curve, as p117 reads them) and every configuration
  must exist, or nothing is computed.

QUANTITIES (d = 100 (first - second) / second per training seed and test instance, seed s
against seed s, the estimators of section A of scripts/p88_family_inference.py)
  p88's functions run as in scripts/p110_seed_extension.py: inside p110.p88_setting, with p88's
  loader exchanged for seed_extension.registry_load (arm_registry.load for a first seed) and,
  when a seed 306 to 310 is selected, p88's random seeds exchanged for p110.TEN_SEED
  (default_rng(110), the streams of the ten-seed record); otherwise p88's own (default_rng(11)).
  The hierarchical bootstrap resamples the seed pairs of all selected seeds and, within each,
  the test instances (stats_family.hier_draws_stream, p88.fresh, stats_family.hier_boot).
  (a) the policy against the budget-sharing specialist, on R: positive d = the policy's
      schedules are longer. p88.component on the parity stream (one generator in the order N,
      L, S, W, LS, LW, SW, LSW, the stream of p88.parity_components; every d of a combination
      has the same shape, so (a) and (c) draw the same seed and instance indices): mean,
      hierarchical 95% interval, verdicts at 0.5% and 1%, direction, the per-seed means (those
      of N also in a table of their own). Joint verdicts
      p88.iut_block and pooled rows p88.pooled_block (the combinations of a base instance kept
      together) over the eight, the six training and the two held-out combinations. The
      combined verdict of p88's comparison with the best dispatching rule:
      p88.superiority_block over the eight combinations (fresh default_rng(11) draws per
      combination, one-sided seed-level t and bootstrap p-values in each direction, Holm over
      the eight, each direction its own family) with the rule
      p89_public_family.superiority_verdict: ahead when the interval lies below zero and both
      Holm-adjusted p-values for H1: d < 0 are below alpha, behind in the mirror case, not
      distinguishable otherwise. Each verdict is checked against superiority_verdict on the
      row's own values; the counts and the names of the combinations of each verdict.
  (b) the budget-sharing specialist against the specialist (1000 updates), on R: positive d =
      longer schedules after 250 updates, so -d is what the other 750 updates give.
      p88.component on fresh draws (p88.fresh) per combination and p88.pooled_block, the
      quantities p117 reports for the specialist of 2000 updates.
  (c) the policy against the specialist (1000 updates) on the same seeds, beside (a):
      p88.parity_components, p88.iut_block and p88.pooled_block. It is held against the record
      of the same seed set (arm_registry.out): for the seeds 301 to 310 the ten-seed record of
      p110 (results/seed_extension.json: ten, iut_ten, pool_ten), for the seeds 301 to 305 the
      five-seed record of p88 (results/family_inference.json: section A). When that record
      holds the same seeds, B and alpha, every component and pooled row must equal it (mean
      and interval within 1e-9, verdicts and direction exactly) and every joint verdict too,
      else the script refuses; otherwise the JSON says why nothing was compared (no record of
      this seed set, the record absent, or other seeds, B or alpha in it).
  (d) the training effort as the configurations record it: max_updates and num_envs of the
      policy, of each specialist and of each budget-sharing specialist. train_family.py rolls
      out one episode in each of its num_envs environments per update, so the episodes of a run
      are max_updates x num_envs. Totals of one policy and of the eight specialists of each
      kind for one training seed. Every run of a kind must record the same two values, or the
      script refuses.
  The comparison is not planned in the statistical protocol; the JSON and the Markdown say so.
  ONE SEED. With k = 1 the hierarchical bootstrap draws the one seed in every replicate, so
  every interval covers test instances only, the seed-level t interval is undefined (null in
  the JSON) and the combined verdict is never ahead or behind (its seed-level p-value is 1).

OUTPUTS (arm_registry.out under PACT_OUT, default out/exact)
  manuscript/macros_budgetspec.tex   macro_io.emit; names Bsp<quantity>, letters only, checked
                                    against every macro defined in a .tex file under manuscript/;
                                    signed values with two decimals (p88 f2, for math mode),
                                    verdict words of stats_family.TEX and of
                                    p89_public_family.VERDICT_WORDS
  results/budget_specialists.json   every component, block, verdict, record check and effort row,
                                    and the per-seed means under N (per_seed_N)
  results/budget_specialists.md     the tables
CHECK (--check, read only): lists every missing or unfinished input of the selected seeds and
  exits 0 only when none is missing.
DRY (--dry [DIR]): the whole path on synthetic arrays and training records in DIR (default a
  temporary folder outside the repository, removed afterwards), with B = --boot (default 1000),
  for the seeds 301, 302 and 306 (a first and a second seed set's names, paths and streams).

Usage: PACT_ENV=exact PACT_OUT=out/exact python scripts/p127_budget_specialists.py [--seeds S]
                                                                     [--root DIR]
       ... python scripts/p127_budget_specialists.py --check [--seeds S]
       python scripts/p127_budget_specialists.py --dry [DIR] [--boot 1000]
       --seeds takes '301', '301,306' or '301-310' (default 301-310); --root resolves the
       training records and configurations against DIR (default the repository).
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import budgetspec_runs as BR  # noqa: E402
import p110_seed_extension as P110  # noqa: E402
import p116_convergence as P116  # noqa: E402
import p117_long_specialists as P117  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import seed_extension as SE  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS, superiority_verdict  # noqa: E402

GENERATOR = 'scripts/p127_budget_specialists.py'
PREFIX = 'Bsp'
OUT_FILES = {'macros': 'manuscript/macros_budgetspec.tex', 'json': 'results/budget_specialists.json',
             'md': 'results/budget_specialists.md'}
R8 = list(REG.COMBOS)
FAM = F88.PAR_FAM
FAMS = ('All', 'Train', 'Held')
MARGINS = F88.MARGINS
MW = F88.MW
WITHIN = 0.5
FOLDER = BR.FOLDER
SPEC_UPDATES = BR.STANDARD_UPDATES          # the specialist's number of updates
PROTOCOL = 'not planned in the statistical protocol'
DRY_BOOT = 1000
Refused = BR.Refused                        # the class of longspec_runs and p117 as well
REFUSALS = (Refused, SE.Refused)            # SE.Refused: seed_extension, p110
P110_GENERATOR = 'scripts/p110_seed_extension.py'


class Incomplete(Refused):
    """An input of the selected seeds that is missing or not finished."""


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    BR.require_exact()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')


def selected_seeds(text=None):
    """The seeds of --seeds (budgetspec_runs.parse_seeds: a subset of 301 to 310); None gives
    the ten seeds. PACT_SEEDS is not read."""
    return tuple(BR.DEFAULT_SEEDS) if text in (None, '') else BR.parse_seeds(text)


def stream_seed(seeds):
    """p110.TEN_SEED when a seed 306 to 310 is selected (the streams of the ten-seed record),
    None (p88's own streams) otherwise."""
    return P110.TEN_SEED if any(SE.is_second(s) for s in seeds) else None


def _at(root, p):
    return p if root in ('.', '') else os.path.join(root, p)


def config_paths(combo, seed):
    """{kind: configuration path} of the three runs of a combination and a seed."""
    return dict(policy=SE.config_path('released', None, seed),
                specialist=SE.config_path('specialist', combo, seed),
                share=BR.config_path(combo, seed))


def read_run(kind, combo, seed, root='.'):
    """(updates, curve, best) of the budget-sharing run ('share') or the specialist
    ('specialist') from its training log, read as p117 reads its runs; P116.Incomplete or
    P116.Refused name the run."""
    if kind == 'share':
        nm, log, u = BR.name(combo, seed), BR.train_log(combo, seed), BR.UPDATES
    else:
        nm, log, u = (BR.standard_name(combo, seed), SE.train_log('specialist', combo, seed),
                      SPEC_UPDATES)
    with P117.p116_updates(u):
        return P116.read_curve(_at(root, log), 'specialist', combo, nm)


def missing_inputs(seeds, root='.'):
    """Every missing or unfinished input of the seeds, one line each; a defective log
    (P116.Refused that is not Incomplete) raises Refused."""
    miss = []
    for s in seeds:
        pc = _at(root, config_paths('N', s)['policy'])
        if not os.path.exists(pc):
            miss.append(f'policy of seed {s}: no configuration {pc}')
        for r in R8:
            nm = BR.name(r, s)
            tp = BR.training_problems(r, s, root)
            if tp:
                miss.append(f'{nm}: not trained: {tp[0]}' + (f' (and {len(tp) - 1} more)' if len(tp) > 1 else ''))
            ep = BR.evaluation_problems(r, s)
            if ep:
                miss.append(f'{nm}: not evaluated: {ep[0]}')
            n = BR.n_test(r)
            for what, p in (('specialist', BR.standard_result(r, s)), ('policy', BR.policy_result(r, s))):
                for x in BR.rows_problems(n, [p]):
                    miss.append(f'{what} of seed {s} on {r}: {x}')
            for kind in ('share', 'specialist'):
                if kind == 'share' and tp:
                    continue
                try:
                    read_run(kind, r, s, root)
                except P116.Incomplete as e:
                    miss.append(str(e))
                except P116.Refused as e:
                    raise Refused(f'defective training log: {e}')
    return miss


# ----------------------------------------------------------------- loaders
def share_specialist(s, folder, r, n=None):
    """Makespans of the budget-sharing specialist of seed s on its combination r (the signature
    of p88's loaders)."""
    if folder != FOLDER:
        raise Refused(f'the budget-sharing specialists are evaluated at {FOLDER} only, not {folder}')
    return REG.read_path(BR.result_path(r, s), n)


# ----------------------------------------------------------------- quantities
def policy_vs_share(seeds):
    """(a): ({R: d}, {R: component}) on the parity stream of p88 (one default_rng(STREAM_SEED)
    in the order of p88.parity_components)."""
    rng = np.random.default_rng(F88.STREAM_SEED)
    D, C = {}, {}
    for r in F88.TRAIN + F88.HELD:
        d = F88.seed_pairs(F88.released, share_specialist, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, SF.hier_draws_stream(d, F88.B, rng))
    return {r: D[r] for r in R8}, {r: C[r] for r in R8}


def superiority(D):
    """(a), combined verdict: p88.superiority_block over the eight combinations, each verdict
    checked against p89's superiority_verdict; returns (rows, {verdict: [combinations]})."""
    sup = F88.superiority_block({r: D[r] for r in R8})
    for r in R8:
        x = sup[r]
        want = superiority_verdict(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                                   x['p_t_greater_holm'], x['p_boot_greater_holm'], F88.ALPHA)
        if x['verdict'] != want:
            raise Refused(f'{r}: p88.superiority_block gives {x["verdict"]!r}, '
                          f'p89.superiority_verdict {want!r}')
    return sup, {w: [r for r in R8 if sup[r]['verdict'] == w] for w in VERDICT_WORDS}


def share_vs_specialist(seeds):
    """(b): ({R: d}, {R: component}, pooled block), fresh draws per combination."""
    D, C = {}, {}
    for r in R8:
        d = F88.seed_pairs(share_specialist, F88.specialist, FOLDER, r, seeds=list(seeds))
        D[r] = d
        C[r] = F88.component(d, F88.fresh(d))
    return D, C, F88.pooled_block(D, FAM)


def effort(seeds, root='.'):
    """(d): per kind the updates and episodes per update its configurations record, and the
    totals of one policy and of eight specialists of each kind."""
    cfg = {'policy': [], 'specialist': [], 'share': []}
    for s in seeds:
        for r in R8:
            for kind, p in config_paths(r, s).items():
                if kind == 'policy' and r != R8[0]:
                    continue
                c = json.load(open(_at(root, p)))
                cfg[kind].append((p, int(c['max_updates']), int(c['num_envs'])))
    out = {}
    for kind, rows in cfg.items():
        vals = sorted({(u, e) for _, u, e in rows})
        if len(vals) != 1:
            raise Refused(f'the configurations of the {kind} runs record different (max_updates, '
                          f'num_envs): {vals[:4]}')
        u, e = vals[0]
        runs = 1 if kind == 'policy' else len(R8)
        out[kind] = dict(updates=u, episodes_per_update=e, episodes=u * e, runs=runs,
                         total_updates=runs * u, total_episodes=runs * u * e,
                         configurations=len(rows))
    pe = out['policy']['total_episodes']
    out['ratio'] = dict(share=out['share']['total_episodes'] / pe,
                        specialist=out['specialist']['total_episodes'] / pe)
    return out


def record_of(seeds):
    """(file, generator, reader) of the record of the policy against the specialist on exactly
    these seeds, or None: p110's ten-seed record for 301 to 310, p88's five-seed record for 301
    to 305. reader(record) gives (components, joint verdicts, pooled rows)."""
    seeds = tuple(int(s) for s in seeds)
    if seeds == tuple(SE.ALL_SEEDS):
        return (P110.OUT_FILES['json'], P110_GENERATOR,
                lambda r: (r['ten'], r['iut_ten'], r['pool_ten']))
    if seeds == tuple(SE.FIRST_SEEDS):
        return (P117.RECORD, P117.RECORD_GENERATOR,
                lambda r: (r['parity']['components'], r['parity']['iut'], r['parity']['pooled']))
    return None


def record_check(seeds, C, iut, pool):
    """(c) against the record of the same seed set (record_of) when it holds the same seeds, B
    and alpha: means and intervals within 1e-9 (p117.RECORD_TOL), verdicts, directions and
    joint verdicts exactly; refuses on a difference."""
    found = record_of(seeds)
    if found is None:
        return dict(path=None, generator=None, status='no record of this seed set', compared=False)
    rel, gen, read = found
    p = REG.out(rel)
    out = dict(path=shown(p), generator=gen)
    if not os.path.exists(p):
        return dict(out, status='absent', compared=False)
    rec = json.load(open(p))
    if rec.get('generator') != gen or not str(rec.get('setting', '')).startswith(
            f'environment {BR.ENV},'):
        return dict(out, status=f'not a record of {gen} for the {BR.ENV} environment', compared=False)
    if [int(x) for x in rec.get('seeds', [])] != [int(x) for x in seeds]:
        return dict(out, status=f'the record holds seeds {rec.get("seeds")}', compared=False)
    if int(rec.get('boot', -1)) != F88.B or float(rec.get('alpha', -1)) != F88.ALPHA:
        return dict(out, status=f'the record was computed with B = {rec.get("boot")}, alpha = '
                                f'{rec.get("alpha")}', compared=False)
    comps, riut, rpool = read(rec)
    bad, dmax, n = [], 0.0, 0
    rows = [(r, C[r], comps[r]) for r in R8] + [(f'pooled {f}', pool[f], rpool[f]) for f in FAMS]
    for where, x, y in rows:
        for k in ('mean', 'lo', 'hi'):
            n += 1
            dmax = max(dmax, abs(float(x[k]) - float(y[k])))
            if abs(float(x[k]) - float(y[k])) > P117.RECORD_TOL:
                bad.append(f'{where} {k}: {x[k]!r} here, {y[k]!r} in the record')
        for k in ('v0.5', 'v1.0', 'direction'):
            n += 1
            if x[k] != y[k]:
                bad.append(f'{where} {k}: {x[k]!r} here, {y[k]!r} in the record')
    for key, x in iut.items():
        n += 1
        y = riut[key]
        if bool(x['holds']) != bool(y['holds']) or list(x['failing']) != list(y['failing']):
            bad.append(f'joint verdict {key}: holds {x["holds"]} failing {x["failing"]} here, holds '
                       f'{y["holds"]} failing {y["failing"]} in the record')
    if bad:
        raise Refused(f'the policy against the specialist does not reproduce {shown(p)}: '
                      + '; '.join(bad[:6]))
    return dict(out, status='equal to the record', compared=True, largest_difference=dmax,
                values=n)


def analyse(root='.', dry=False, seeds=None):
    """Every input, then (a) to (d); returns the result dict. Raises Refused."""
    check_setting()
    seeds = selected_seeds() if seeds is None else tuple(seeds)
    miss = missing_inputs(seeds, root)
    if miss:
        raise Incomplete(f'{len(miss)} inputs of seeds {",".join(map(str, seeds))} are missing or '
                         'unfinished:\n  ' + '\n  '.join(miss[:12])
                         + (f'\n  ... and {len(miss) - 12} more' if len(miss) > 12 else ''))
    F88.verify_bases(FOLDER, max(BR.n_test(r) for r in R8))
    rs = stream_seed(seeds)
    with P110.p88_setting(SE.registry_load, rs, rs):
        DA, CA = policy_vs_share(seeds)
        SUP, verdicts = superiority(DA)
        IA, PA = F88.iut_block(CA, FAM), F88.pooled_block(DA, FAM)
        DB, CB, PB = share_vs_specialist(seeds)
        DS, CS = F88.parity_components(FOLDER, seeds=list(seeds))
        IS, PS = F88.iut_block(CS, FAM), F88.pooled_block(DS, FAM)
        stream, fresh = F88.STREAM_SEED, F88.FRESH_SEED
    rec = record_check(seeds, CS, IS, PS)
    eff = effort(seeds, root)
    means = {r: CB[r]['mean'] for r in R8}
    hi_c = max(R8, key=lambda r: (means[r], -R8.index(r)))
    lo_c = min(R8, key=lambda r: (means[r], R8.index(r)))
    changed = {str(mg): [r for r in R8 if CA[r][f'v{mg}'] != CS[r][f'v{mg}']] for mg in MARGINS}
    origin = ('p110.TEN_SEED, the streams of the ten-seed record' if rs is not None
              else "p88's own")
    return dict(generator=GENERATOR, setting=f'environment {BR.ENV}, training seeds '
                f'{",".join(map(str, seeds))} ({len(seeds)} run(s) per combination)',
                seeds=list(seeds), k=len(seeds), dry=bool(dry), boot=F88.B, alpha=F88.ALPHA,
                protocol=PROTOCOL,
                one_seed_note=('one seed: every interval covers test instances only; the '
                               'seed-level t interval is undefined') if len(seeds) == 1 else None,
                loader='seed_extension.registry_load in place of p88.load (p110.p88_setting)',
                streams=dict(origin=origin,
                             policy=f'one default_rng({stream}) in the order '
                                    f'{F88.TRAIN + F88.HELD} for each of the two comparisons '
                                    '(p88.parity_components)',
                             combined_verdict=f'fresh default_rng({fresh}) per combination '
                                              '(p88.superiority_block)',
                             share_vs_specialist=f'fresh default_rng({fresh}) per combination and '
                                                 'per pooled row (p88.fresh, p88.pooled_block)'),
                definitions=dict(
                    policy='100 (policy - specialist) / specialist per seed and test instance; '
                           'positive = the policy is slower',
                    share_vs_specialist='100 (budget-sharing specialist - specialist) / specialist '
                                        'per seed and test instance; positive = longer schedules '
                                        f'after {BR.UPDATES} updates',
                    within=f'-{WITHIN} < lo and hi < {WITHIN}',
                    combined_verdict='ahead: interval below zero and both Holm-adjusted one-sided '
                                     'p-values (seed-level t, bootstrap; H1 the policy faster) below '
                                     'alpha; behind: interval above zero and both Holm-adjusted '
                                     'p-values for H1 the policy slower below alpha; Holm over the '
                                     'eight combinations, each direction its own family'),
                updates=dict(policy=eff['policy']['updates'], share=BR.UPDATES,
                             specialist=SPEC_UPDATES),
                against_share=dict(components=CA, combined=SUP, verdicts=verdicts, iut=IA, pooled=PA),
                against_specialist=dict(components=CS, iut=IS, pooled=PS, record=rec),
                per_seed_N=dict(seeds=list(seeds), policy_vs_share=list(CA['N']['seed_means']),
                                policy_vs_specialist=list(CS['N']['seed_means'])),
                verdict_changes=changed,
                share_vs_specialist=dict(components=CB, pooled=PB,
                                         largest=dict(combo=hi_c, mean=means[hi_c]),
                                         smallest=dict(combo=lo_c, mean=means[lo_c]),
                                         within_half=[r for r in R8
                                                      if -WITHIN < CB[r]['lo'] and CB[r]['hi'] < WITHIN],
                                         longer=[r for r in R8 if CB[r]['direction'] == 'behind'],
                                         shorter=[r for r in R8 if CB[r]['direction'] == 'ahead']),
                effort=eff)


# ----------------------------------------------------------------- outputs
f2 = F88.f2
shown = P117.shown
row_line = F88.row_line


def num(x):
    """An integer with the LaTeX thousands separator, 40{,}000."""
    return f'{int(x):,}'.replace(',', '{,}')


def ratio(x):
    return f'{x:.2f}'.rstrip('0').rstrip('.')


def names(xs):
    return ', '.join(xs) or 'none'


def macro_lines(res):
    k, A, S, B, E = (res['k'], res['against_share'], res['against_specialist'],
                     res['share_vs_specialist'], res['effort'])
    L = [f'% Generated by {GENERATOR}, training seeds {", ".join(map(str, res["seeds"]))} '
         f'({k} seed{"s" if k > 1 else ""}'
         + ('; every interval covers test instances only)' if k == 1 else
            '; intervals over training seeds and test instances); the comparison is '
            + PROTOCOL),
         f'% Gaps in %, seed s against seed s, hierarchical 95% intervals of p88, B = {res["boot"]}. '
         f'BspPol<R>: the policy minus the budget-sharing specialist ({BR.UPDATES} updates), over '
         'it (positive = the policy is slower); BspStd<R>: the policy minus the specialist '
         f'({SPEC_UPDATES} updates), over it; BspShare<R>: the budget-sharing specialist minus the '
         f'specialist, over the specialist (positive = longer schedules after {BR.UPDATES} '
         'updates). Pool<F>: pooled over All (eight), Train (six) or Held (two) combinations.',
         f'% BspPol<R>Verdict: ahead = hierarchical {100 * (1 - res["alpha"]):g}% CI below zero and '
         'both Holm-adjusted one-sided p-values (seed-level t, bootstrap; H1 the policy faster) '
         f'below {res["alpha"]:g}; behind = CI above zero and both Holm-adjusted p-values for H1 '
         'the policy slower (its own Holm family over the eight combinations) below '
         f'{res["alpha"]:g}; not distinguishable otherwise (p88.superiority_block, fresh draws per '
         'combination).',
         f'\\newcommand{{\\{PREFIX}Seeds}}{{{k}}}\\newcommand{{\\{PREFIX}Runs}}{{{len(R8) * k}}}'
         '  % training seeds, runs of the budget-sharing specialists']
    for key, word in (('policy', 'Policy'), ('share', 'Share'), ('specialist', 'Spec')):
        x = E[key]
        L.append(f'\\newcommand{{\\{PREFIX}{word}Updates}}{{{num(x["updates"])}}}'
                 f'\\newcommand{{\\{PREFIX}{word}EpisodesPerUpdate}}{{{num(x["episodes_per_update"])}}}'
                 f'\\newcommand{{\\{PREFIX}{word}Episodes}}{{{num(x["episodes"])}}}'
                 + (f'\\newcommand{{\\{PREFIX}{word}TotalUpdates}}{{{num(x["total_updates"])}}}'
                    f'\\newcommand{{\\{PREFIX}{word}TotalEpisodes}}{{{num(x["total_episodes"])}}}'
                    if key != 'policy' else '')
                 + f'  % {key if key != "share" else "budget-sharing specialist"}: updates, episodes '
                   'per update and episodes of one run (max_updates, num_envs of its configuration)'
                 + (f'; updates and episodes of the {x["runs"]} runs of one seed' if key != 'policy' else ''))
    L.append(f'\\newcommand{{\\{PREFIX}ShareRatio}}{{{ratio(E["ratio"]["share"])}}}'
             f'\\newcommand{{\\{PREFIX}SpecRatio}}{{{ratio(E["ratio"]["specialist"])}}}'
             '  % episodes of the eight budget-sharing specialists and of the eight specialists, '
             'over the episodes of the policy')
    for r in R8:
        x, c = A['components'][r], A['combined'][r]
        L.append(row_line(f'{PREFIX}Pol{r}', dict(x, verdict=c['verdict']),
                          f'the policy against the budget-sharing specialist, {r}; combined verdict '
                          f'on the interval [{c["lo"]:+.2f}, {c["hi"]:+.2f}] of its fresh draws, '
                          f'{F88.holm_text(c)}', verdict=True))
    for fam in FAMS:
        x = A['pooled'][fam]
        L.append(row_line(f'{PREFIX}PolPool{fam}', x, f'the policy against the budget-sharing '
                          f'specialists, pooled over {x["c"]} combinations, base instances together'))
    L += F88.iut_lines(f'{PREFIX}PolIut', A['iut'], FAMS)
    v = A['verdicts']
    L.append(F88.count_line(f'{PREFIX}Pol', {w: len(v[w]) for w in VERDICT_WORDS},
                            'the policy against the budget-sharing specialist, of eight combinations'))
    L.append(f'\\newcommand{{\\{PREFIX}PolVerdictAheadCombos}}{{{names(v[VERDICT_WORDS[0]])}}}'
             f'\\newcommand{{\\{PREFIX}PolVerdictBehindCombos}}{{{names(v[VERDICT_WORDS[1]])}}}'
             f'\\newcommand{{\\{PREFIX}PolVerdictNdCombos}}{{{names(v[VERDICT_WORDS[2]])}}}'
             '  % the combinations of each combined verdict: ahead, behind, not distinguishable')
    for r in R8:
        L.append(row_line(f'{PREFIX}Std{r}', S['components'][r],
                          f'the policy against the specialist ({SPEC_UPDATES} updates), {r}'))
    for fam in FAMS:
        x = S['pooled'][fam]
        L.append(row_line(f'{PREFIX}StdPool{fam}', x, f'the policy against the specialists '
                          f'({SPEC_UPDATES} updates), pooled over {x["c"]} combinations'))
    ch = res['verdict_changes']
    L.append(f'\\newcommand{{\\{PREFIX}VerdictChangeHalfN}}{{{len(ch["0.5"])}}}'
             f'\\newcommand{{\\{PREFIX}VerdictChangeOneN}}{{{len(ch["1.0"])}}}'
             '  % combinations whose margin verdict of the policy differs between the two '
             f'specialists, at 0.5% ({names(ch["0.5"])}) and at 1% ({names(ch["1.0"])})')
    for r in R8:
        L.append(row_line(f'{PREFIX}Share{r}', B['components'][r],
                          f'the budget-sharing specialist against the specialist, {r}'))
    for fam in FAMS:
        x = B['pooled'][fam]
        L.append(row_line(f'{PREFIX}SharePool{fam}', x, f'the budget-sharing specialists against '
                          f'the specialists, pooled over {x["c"]} combinations'))
    hi, lo = B['largest'], B['smallest']
    L.append(f'\\newcommand{{\\{PREFIX}ShareMax}}{{{f2(hi["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareMaxCombo}}{{{hi["combo"]}}}'
             f'\\newcommand{{\\{PREFIX}ShareMin}}{{{f2(lo["mean"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareMinCombo}}{{{lo["combo"]}}}'
             '  % the budget-sharing specialist against the specialist: largest and smallest '
             'per-combination difference, and its combination')
    L.append(f'\\newcommand{{\\{PREFIX}ShareWithinHalfN}}{{{len(B["within_half"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareLongerN}}{{{len(B["longer"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareLongerCombos}}{{{names(B["longer"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareShorterN}}{{{len(B["shorter"])}}}'
             f'\\newcommand{{\\{PREFIX}ShareShorterCombos}}{{{names(B["shorter"])}}}'
             f'  % combinations whose interval lies within +-{WITHIN}% ({names(B["within_half"])}), '
             'above zero (longer schedules after '
             f'{BR.UPDATES} updates) and below zero')
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
    found = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in found if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(found) != len(set(found)):
        dup = sorted({x for x in found if found.count(x) > 1})
        raise Refused(f'duplicate macro names in the output: {dup[:5]}')
    have = defined_macros()
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return found


def ci(x):
    return f'[{f2(x["lo"])}, {f2(x["hi"])}]'


def markdown(res):
    A, S, B, E = res['against_share'], res['against_specialist'], res['share_vs_specialist'], res['effort']
    md = [f'# Specialists that share the policy\'s training budget ({res["setting"]})', '',
          f'The comparison is {PROTOCOL}.', '']
    if res['one_seed_note']:
        md += [f'With {res["one_seed_note"]}.', '']
    if res['dry']:
        md += ['DRY RUN on synthetic arrays and training records.', '']
    md += [f'Gaps in %, seed s against seed s; hierarchical 95% intervals of p88 (B = {res["boot"]}).',
           '', '## Training effort (one training seed)', '',
           '| | runs | updates per run | episodes per update | episodes per run | updates | episodes |',
           '|---|---|---|---|---|---|---|']
    for key, what in (('policy', 'policy'), ('share', f'budget-sharing specialists ({BR.UPDATES} updates)'),
                      ('specialist', f'specialists ({SPEC_UPDATES} updates)')):
        x = E[key]
        md.append(f'| {what} | {x["runs"]} | {x["updates"]} | {x["episodes_per_update"]} | '
                  f'{x["episodes"]} | {x["total_updates"]} | {x["total_episodes"]} |')
    md += ['', f'Episodes over those of the policy: budget-sharing specialists '
               f'{ratio(E["ratio"]["share"])}, specialists {ratio(E["ratio"]["specialist"])}.', '',
           '## The policy against each specialist', '',
           f'Positive: the policy is slower. Against the specialist ({SPEC_UPDATES} updates): '
           f'p88.parity_components on the same seeds ({S["record"]["status"]}'
           + (f': {S["record"]["path"]}' if S['record']['path'] else '') + '). '
           'Combined verdict: p88.superiority_block over the eight combinations, its interval from '
           f'fresh draws. Random streams: {res["streams"]["origin"]}.', '',
           f'| combination | {BR.UPDATES} updates: mean | 95% CI | 0.5% | 1% | direction | combined '
           f'verdict | {SPEC_UPDATES} updates: mean | 95% CI | 0.5% | 1% | direction |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        a, c, s = A['components'][r], A['combined'][r], S['components'][r]
        md.append(f'| {r} | {f2(a["mean"])} | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | {a["direction"]} '
                  f'| {c["verdict"]} {ci(c)} | {f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} '
                  f'| {s["direction"]} |')
    for fam in FAMS:
        a, s = A['pooled'][fam], S['pooled'][fam]
        md.append(f'| pooled, {fam} | {f2(a["mean"])} | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | '
                  f'{a["direction"]} | | {f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} | '
                  f'{s["direction"]} |')
    ps = res['per_seed_N']
    md += ['', 'Per-seed means under N (%, seed s against seed s; positive: the policy is slower):',
           '', '| | ' + ' | '.join(f's{x}' for x in ps['seeds']) + ' |',
           '|---|' + '---|' * len(ps['seeds']),
           f'| against the budget-sharing specialist ({BR.UPDATES} updates) | '
           + ' | '.join(f2(m) for m in ps['policy_vs_share']) + ' |',
           f'| against the specialist ({SPEC_UPDATES} updates) | '
           + ' | '.join(f2(m) for m in ps['policy_vs_specialist']) + ' |']
    v, ch = A['verdicts'], res['verdict_changes']
    md += ['', 'Combined verdict of the policy against the budget-sharing specialist: '
               + '; '.join(f'{w} {len(v[w])} ({names(v[w])})' for w in VERDICT_WORDS) + '.',
           f'Margin verdicts that differ between the two specialists: at 0.5% {names(ch["0.5"])}; '
           f'at 1% {names(ch["1.0"])}.', '', '| joint claim | margin | '
           f'{BR.UPDATES} updates | {SPEC_UPDATES} updates |', '|---|---|---|---|']
    for fam in FAMS:
        for mg in MARGINS:
            for claim in ('equivalent', 'non-inferior'):
                key = f'{fam}|{mg}|{claim}'
                a, s = A['iut'][key], S['iut'][key]
                md.append(f'| {claim} on {fam} ({", ".join(a["components"])}) | {mg}% | '
                          f'{"holds" if a["holds"] else "fails: " + ", ".join(a["failing"])} | '
                          f'{"holds" if s["holds"] else "fails: " + ", ".join(s["failing"])} |')
    md += ['', f'## The budget-sharing specialist against the specialist ({SPEC_UPDATES} updates)', '',
           f'Positive: longer schedules after {BR.UPDATES} updates.', '',
           '| combination | mean | 95% CI | within 0.5% | direction |', '|---|---|---|---|---|']
    for r in R8:
        c = B['components'][r]
        md.append(f'| {r} | {f2(c["mean"])} | {ci(c)} | {"yes" if r in B["within_half"] else "no"} '
                  f'| {c["direction"]} |')
    for fam in FAMS:
        c = B['pooled'][fam]
        md.append(f'| pooled, {fam} | {f2(c["mean"])} | {ci(c)} | '
                  f'{"yes" if -WITHIN < c["lo"] and c["hi"] < WITHIN else "no"} | {c["direction"]} |')
    md += ['', f'Interval above zero: {names(B["longer"])}; below zero: {names(B["shorter"])}.']
    return md


clean = P117.clean


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(root='.', dry=False, seeds=None):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md, outs)."""
    check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, dry, seeds)
        lines = macro_lines(res)
        found = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(clean(res), f, indent=1, allow_nan=False)
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, found, md, outs


def check(root='.', seeds=None):
    """Read-only precondition of the chain step; returns the exit code."""
    BR.require_exact()
    seeds = selected_seeds() if seeds is None else tuple(seeds)
    miss = missing_inputs(seeds, root)
    st = BR.state(seeds, root)
    print(f'p127: seeds {",".join(map(str, seeds))}: {len(st)} runs of {BR.UPDATES} updates '
          f'({sum(x["trained"] for x in st)} trained, {sum(x["evaluated"] for x in st)} evaluated), '
          'the specialists and the policy on each combination at 10x25: '
          + ('every input present (arrays with one row per test instance, finished training logs, '
             'configurations)' if not miss else f'{len(miss)} inputs missing or unfinished'))
    for m in miss[:8]:
        print(f'p127:   {m}')
    if len(miss) > 8:
        print(f'p127:   ... and {len(miss) - 8} more')
    return 0 if not miss else 1


# ----------------------------------------------------------------- dry mode
# true mean differences (%): the policy against the budget-sharing specialist, and the
# budget-sharing specialist against the specialist
TRUE_POL = {'N': 0.6, 'L': -1.0, 'S': -0.5, 'W': -1.2, 'LS': -0.8, 'LW': -1.1, 'SW': 0.0, 'LSW': -0.9}
TRUE_SHARE = {'N': 0.9, 'L': 1.2, 'S': 0.6, 'W': 1.5, 'LS': 1.0, 'LW': 1.4, 'SW': 0.1, 'LSW': 1.1}
NOISE = 0.2
DRY_SEEDS = (301, 302, 306)                # a second seed among them: its paths and p110's streams
DRY_NORM = 200.0
environment = P117.environment


def _put(path, text):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def _save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([x, np.zeros(len(x))], axis=1))


def synthetic_log(model, combo, updates, scores):
    """A training log in the format of train_family.py."""
    out = [BR.MASK_LINE, f'[vali] MWKR norms: {combo}={DRY_NORM:.1f}',
           f'[model] params total={BR.PARAMS}  film=0 (0.00%)',
           f"[train_family] model={REG.PREFIX}{model} regimes=['{combo}'] updates={updates}"]
    best = float('inf')
    for i, s in enumerate(scores):
        best = min(best, s)
        out.append(f'[vali @{25 * (i + 1)}] rel={s:.4f} (best {best:.4f})  {combo}={s * DRY_NORM:.1f}')
    out.append(f'[train_family] done in 1.00 h; best rel={best:.4f}')
    return '\n'.join(out) + '\n'


def synthetic_records(tree, seed):
    """Configuration of the policy, and configuration, log, validation record and checkpoint of
    the specialist and of the budget-sharing run of each combination."""
    base = dict(seed_train=int(seed), first_mask='exact', mask_ties='exact', device='cuda',
                lr=3e-4, num_envs=20)
    _put(os.path.join(tree, SE.config_path('released', None, seed)),
         json.dumps(dict(base, model_suffix=SE.name('released', None, seed),
                         max_updates=2000, family_regimes=','.join(REG.TRAIN_COMBOS)), indent=1))
    scores = np.round(0.95 - 0.06 * (1 - np.exp(-np.arange(1, SPEC_UPDATES // 25 + 1) / 25.0)), 4)
    for r in R8:
        std = dict(base, model_suffix=BR.standard_name(r, seed), max_updates=SPEC_UPDATES,
                   family_regimes=r)
        for p, c, sc in ((BR.standard_paths(r, seed), std, scores),
                         (BR.paths(r, seed), dict(std, model_suffix=BR.name(r, seed),
                                                  max_updates=BR.UPDATES),
                          scores[:BR.UPDATES // 25])):
            _put(os.path.join(tree, p['checkpoint']), '')
            _put(os.path.join(tree, p['config']), json.dumps(c, indent=1))
            _put(os.path.join(tree, p['vali']), str([float(x) for x in sc]))
            _put(os.path.join(tree, p['log']), synthetic_log(c['model_suffix'], r, c['max_updates'], sc))


def build_synthetic(tree, seeds=DRY_SEEDS):
    """Arrays of the three runs per seed and combination, and the training records."""
    rng = np.random.default_rng(2026)
    for s in seeds:
        for r in R8:
            n = BR.n_test(r)
            spec = np.round(rng.uniform(400.0, 900.0, n) * (1 + rng.normal(0, 0.01, n)), 1)
            share = spec * (1 + (TRUE_SHARE[r] + rng.normal(0, NOISE, n)) / 100.0)
            pol = share * (1 + (TRUE_POL[r] + rng.normal(0, NOISE, n)) / 100.0)
            _save(BR.standard_result(r, s), spec)
            _save(BR.policy_result(r, s), pol)
            _save(BR.result_path(r, s), share)
        synthetic_records(tree, s)


def dry_paths(tree):
    return dict(PACT_ENV=BR.ENV, PACT_RESULTS=os.path.join(tree, 'results'),
                PACT_OUT=os.path.join(tree, 'out'))


def dry_run(tree, boot=DRY_BOOT, seeds=DRY_SEEDS):
    """The whole path on a synthetic tree; returns (res, names, md, outs)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    F88.configure(boot=boot)
    with environment(**dry_paths(tree)):
        build_synthetic(tree, seeds)
        return run(root=tree, dry=True, seeds=seeds)


def main():
    ap = argparse.ArgumentParser(description="specialists that share the policy's training budget")
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes '
                         f'B = {SF.B_DEFAULT}, the B of p88')
    ap.add_argument('--root', default='.', help='folder the training records are resolved against')
    ap.add_argument('--seeds', default=None,
                    help=f'training seeds, e.g. 301-310 or 301,306 (default '
                         f'{BR.DEFAULT_SEEDS[0]}-{BR.DEFAULT_SEEDS[-1]}); PACT_SEEDS is not read')
    a = ap.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('PACT_ENV', BR.ENV)
    root = os.path.abspath(a.root) if a.root not in ('.', '') else '.'
    if a.dry is None:
        if a.boot is not None:
            ap.error('--boot is for the dry run; a real run uses the B of p88')
        os.environ.setdefault('PACT_OUT', 'out/exact')
        try:
            seeds = selected_seeds(a.seeds)
            if a.check:
                sys.exit(check(root, seeds))
            print(f'[p127] environment {REG.env()}, training seeds {",".join(map(str, seeds))}; '
                  f'results {REG.results_root()}; outputs {os.environ["PACT_OUT"]}', file=sys.stderr)
            res, found, md, outs = run(root, seeds=seeds)
        except REFUSALS as e:
            sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
        print('\n'.join(md))
        print(f'\n{len(found)} macros; wrote ' + ', '.join(outs.values()))
        return
    if a.check:
        ap.error('--check and --dry exclude each other')
    if a.seeds is not None:
        ap.error(f'the dry run takes the seeds {",".join(map(str, DRY_SEEDS))}')
    boot = DRY_BOOT if a.boot is None else a.boot
    keep = bool(a.dry)
    tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p127_dry_')
    try:
        res, found, md, outs = dry_run(tree, boot)
    except REFUSALS as e:
        sys.exit(f'p127 self-test on synthetic arrays FAILED: {e}')
    finally:
        if not keep:
            shutil.rmtree(tree, ignore_errors=True)
    if keep:
        print('\n'.join(md))
        print(f'\n{len(found)} macros; outputs in {os.path.join(tree, "out")}')
    else:
        v = res['against_share']['verdicts']
        pool = res['against_share']['pooled']['All']
        print(f'p127 self-test on synthetic arrays passed: seeds {",".join(map(str, res["seeds"]))}, '
              f'B = {boot}, {len(found)} macros, the policy against the budget-sharing specialists '
              f'pooled {f2(pool["mean"])}% {ci(pool)}; combined verdict '
              + ', '.join(f'{w} {len(v[w])}' for w in VERDICT_WORDS))


if __name__ == '__main__':
    main()
