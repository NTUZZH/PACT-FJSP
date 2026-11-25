r"""P140: what the equivalence test returns for comparators known to differ from the specialist
(analysis A), and the policy against two specialists that serve all eight combinations
(analysis B). Factory benchmark at the training size (10x25, 100 test instances), exact mask.
Both analyses read stored makespans and records; no model is run.

SIGN AND VERDICTS
  d = 100 (a - b) / b per training seed and test instance, a the first-named method, b the
  second; positive d means a longer schedule for the first-named method. Verdicts at the margins
  0.5% and 1% (stats_family.verdict on the unrounded 95% interval: equivalent, non-inferior
  only, inferior, inconclusive) and the direction of the interval (ahead, behind, neither), as
  in the principal comparison (scripts/p88_family_inference.py, section A).

A. MARGIN SENSITIVITY (results/margin_sensitivity.json and .md, macro prefix Msn)
  A1  The per-instance best of the four dispatching rules (the shortest of the FIFO, MOR, SPT
      and MWKR schedules on each instance, p88_family_inference.pool, the rule quantity of the
      FamRule macros) minus the specialist of each combination, in percent of the specialist.
      The rule schedules do not depend on a training seed, so each seed pairs the same rule
      makespans with its own specialist. Seeds: the ten seeds of the principal comparison, 301
      to 310 (the second five read by seed_extension.registry_load). Estimator and verdicts of
      the ten-seed principal comparison (scripts/p110_seed_extension.py): p88.component on the
      draws of one default_rng(110) stream in the order of p88 section A (N, L, S, W, LS, LW,
      SW, LSW), and the pooled row p88.pooled_block with a fresh default_rng(110), B = 20,000.
      The same with seeds 301 to 305 and the streams of the five-seed record (default_rng(11))
      is reported in the JSON and the Markdown.
  A2  Read from the records and not recomputed: the policy minus the specialist (the ten-seed
      record of p110, results/seed_extension.json, keys ten and pool_ten) and the policy minus
      the best rule (the record of p88, results/family_inference.json, key rule_pool, the
      FamRule macros, five seeds; its pooled row from the record of p128,
      results/production_specialists.json, side_by_side.rows.10x25.rule, which p128 computes
      with p88.pooled_block from the same arrays). The arrays read here must give the means of
      these records (to 1e-9).
  A3  Counts of the verdicts of A1 over the eight combinations at each margin, with the
      combinations.
  Three further comparators against the specialist of each combination, with verdict counts
  and the pooled verdict:
    sparse  the policy's recipe trained with the reward at the end of the episode only
            (xmask-sparse), read from results/secondary_inference.json (p95, block sparse,
            vs_specialist; seeds 301 to 303), the record p95 writes at the repository root;
    budget  the specialists of 250 updates (the policy's training budget shared by eight)
            against the specialists of 1000 updates, read from results/budget_specialists.json
            (p127, share_vs_specialist; seeds 301 to 310);
    cross   each specialist run under each of its seven other combinations against the
            specialist of that combination, computed here from the arrays of the cross campaign
            (test_results/FAMILY/10x25_cross_xmask) with the readers and the interval of p100
            (p100.load_all, p95.component: a fresh default_rng(11) per quantity), because no
            record compares these runs with the specialist of the combination (p100 compares
            them with the policy, p88 section G the LSW specialist only). Counts over the 56
            cells, over the eight specialists (each pooled over its seven other combinations)
            and over the eight combinations (each pooled over the seven other specialists), and
            the pooled row of the 56 cells; a drawn base instance brings all its cells. The LSW
            row must reproduce section G of the record of p88 to the printed digits.

B. TWO SPECIALISTS (results/two_specialists.json and .md, macro prefix Tws)
  The policy minus an option, in percent of the option, where the option serves each
  combination C with one specialist run under C:
    B1 (One)    the specialist of N under N, S, W, SW and the specialist of LSW under L, LS,
                LW, LSW (two specialists chosen by whether minimum time lags are active);
    B2 (Two)    the same with the specialist of L in the place of LSW;
    B3 (Three)  the specialist of N under all eight combinations;
    B4 (Four)   each specialist under its own combination, the five-seed principal comparison.
  Readers and checks of p100 (check_layout, load_all: checkpoints, logs, configurations, one
  row per test instance, the off-combination arrays that the principal campaign also holds
  equal), seeds 301 to 305, policy seed s against specialist seed s, instance i against
  instance i. Per combination the crossed hierarchical bootstrap over seed pairs and base
  instances with the draws of one default_rng(11) stream in the order of p88 section A, the
  stream of the five-seed record, so that every option meets the same resamples and B4
  reproduces that record; pooled over the eight, p95.component (a fresh default_rng(11), a
  drawn base instance with all its combinations), the interval of p100's pooled rows and of
  p88's pooled rows. B4 must reproduce the five-seed record (results/family_inference.json,
  parity) to the printed digits; otherwise the script refuses.
  Oracle pairs, plain numbers in the JSON and the Markdown: for each of the 28 pairs of
  specialists, each combination served by the one of the two whose mean difference to the
  policy is the more favourable on the test instances (an oracle choice made on the test
  instances), the mean over the eight combinations of the policy minus the option and of the
  option minus the policy (in percent of the policy); the pairs ranked by the first, so the
  first pair is the pair with the shortest schedules relative to the policy.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_margin_pairs.tex   prefixes Msn (A) and Tws (B), letters only, every line
                                      with a comment; no name defined in another macro file of
                                      paper/ or manuscript/ (the file itself and its copy in
                                      manuscript/ are skipped)
  results/margin_sensitivity.json, results/margin_sensitivity.md
  results/two_specialists.json, results/two_specialists.md
  A missing input or a failed check refuses with the reason and removes the five outputs.

Usage (repository root, CPU, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p140_margin_and_pairs.py
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p140_margin_and_pairs.py --check
      (every input present and in the expected setting; reads only, writes nothing)
"""
import argparse
import glob
import itertools
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import p95_secondary_inference as P95  # noqa: E402
import p100_cross_eval as P100  # noqa: E402
import p110_seed_extension as P110  # noqa: E402
import seed_extension as SE  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p140_margin_and_pairs.py'
OUT_FILES = {'macros': 'manuscript/macros_margin_pairs.tex',
             'margin_json': 'results/margin_sensitivity.json',
             'margin_md': 'results/margin_sensitivity.md',
             'pairs_json': 'results/two_specialists.json',
             'pairs_md': 'results/two_specialists.md'}
PREFIX_A, PREFIX_B = 'Msn', 'Tws'
R8 = list(REG.COMBOS)
ORDER = list(F88.TRAIN) + list(F88.HELD)        # stream order of p88 section A
MARGINS = F88.MARGINS                           # (0.5, 1.0)
MW = F88.MW                                     # {0.5: 'Half', 1.0: 'One'}
B_BOOT = SF.B_DEFAULT
ALPHA = 0.05
FOLDER = '10x25'
STREAM_FIVE = F88.STREAM_SEED                   # 11, the streams of the five-seed record
STREAM_TEN = P110.TEN_SEED                      # 110, the streams of the ten-seed record
SEEDS_FIVE = tuple(SE.FIRST_SEEDS)
SEEDS_TEN = tuple(SE.ALL_SEEDS)
MEAN_TOL = 1e-9
RECORDS = {   # key: (under PACT_OUT?, path, script that writes it, step of run_analysis.sh)
    'family': (True, 'results/family_inference.json', 'scripts/p88_family_inference.py', 'family'),
    'seedext': (True, 'results/seed_extension.json', 'scripts/p110_seed_extension.py',
                'seed_extension'),
    'prodspec': (True, 'results/production_specialists.json',
                 'scripts/p128_production_specialists.py', 'production_specialists'),
    'budget': (True, 'results/budget_specialists.json', 'scripts/p127_budget_specialists.py',
               'budget_specialists'),
    'secondary': (False, 'results/secondary_inference.json', 'scripts/p95_secondary_inference.py',
                  'secondary'),
}
VERDICT_KEYS = (('equivalent', 'Eq'), ('non-inferior only', 'Ni'), ('inferior', 'Inf'),
                ('inconclusive', 'Inc'))
COMPARATORS = (('sparse', 'Sparse'), ('budget', 'Budget'), ('cross', 'Cross'))


def lag_pair(with_lag):
    """{combination: specialist}: the specialist of N where the lag rule is inactive and the
    specialist with_lag where it is active."""
    return {c: (with_lag if 'L' in c else 'N') for c in R8}


OPTIONS = {
    'One': dict(name='B1', serve=lag_pair('LSW'),
                what='two specialists by the lag rule: N under N, S, W, SW and LSW under L, LS, '
                     'LW, LSW'),
    'Two': dict(name='B2', serve=lag_pair('L'),
                what='two specialists by the lag rule with L in the place of LSW: N under N, S, '
                     'W, SW and L under L, LS, LW, LSW'),
    'Three': dict(name='B3', serve={c: 'N' for c in R8},
                  what='one specialist: N under all eight combinations'),
    'Four': dict(name='B4', serve={c: c for c in R8},
                 what='the eight specialists, each under its own combination (the principal '
                      'comparison with five seeds)'),
}
SPEC_MACRO_OPTIONS = ('One', 'Two')             # options whose serving specialist gets a macro

NOTES_A = (
    'Analysis A describes what the verdict rule of the principal comparison returns for '
    'comparators whose quality relative to the specialist is known from other evidence; it adds '
    'no planned comparison and changes no verdict of the principal comparison. The rule makespans '
    'do not depend on a training seed, so the seed level of the interval resamples the '
    'specialist only. The columns read from records are not recomputed.')
NOTES_B = (
    'The options B1 and B2 were named after the test results were known (the matrix of every '
    'specialist under every combination, results/cross_eval.json), so their comparisons are post '
    'hoc. B3 is the single specialist of the combination without rules. B4 is the five-seed '
    'principal comparison and reproduces its record. The oracle pairs choose, for each '
    'combination, the better of the two specialists on the test instances themselves, so their '
    'means are an optimistic bound on a choice made before testing.')


class Refused(Exception):
    """A missing input or a failed check; the outputs are removed and nothing is written."""


# ----------------------------------------------------------------- small helpers
def f2(x):
    return F88.f2(x)


def tex(word):
    return SF.TEX[word]


def ci(x):
    return f'{f2(x["mean"])} [{f2(x["lo"])}, {f2(x["hi"])}]'


def rel(a, b):
    return SF.rel_gap(a, b)


def running_list(names):
    """Combinations as running text in math mode ('$S$ and $SW$'); 'none' when empty."""
    return P110.running_list(list(names))


def shown(path):
    """A path relative to the repository when it lies inside it."""
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def jsonable(x):
    return P100.jsonable(x)


def best_rule(rule_arrays):
    """Per-instance best of the rule schedules: the elementwise minimum of the arrays (the
    shortest rule schedule on each instance), as p88_family_inference.pool."""
    xs = [np.asarray(a, dtype=float) for a in rule_arrays]
    if len({len(x) for x in xs}) != 1:
        raise Refused('the rule arrays differ in length')
    return np.min(np.stack(xs), axis=0)


def verdict_words(lo, hi):
    """{'v0.5': word, 'v1.0': word, 'direction': word} of one interval."""
    out = {f'v{mg}': SF.verdict(lo, hi, mg) for mg in MARGINS}
    out['direction'] = SF.direction(lo, hi)
    return out


def verdict_counts(rows, keys):
    """Per margin, the keys with each verdict word: {'v0.5': {word: [keys]}, 'v1.0': ...}."""
    return {f'v{mg}': {w: [k for k in keys if rows[k][f'v{mg}'] == w] for w in SF.VERDICTS}
            for mg in MARGINS}


def stream_components(D, B, seed, order=None):
    """p88.component of each d in D, the draws of one default_rng(seed) stream in the given order
    (default the order of p88 section A), as p88.parity_components draws them."""
    rng = np.random.default_rng(seed)
    return {r: F88.component(D[r], SF.hier_draws_stream(D[r], B, rng)) for r in (order or ORDER)}


def option_d(pol, spec, serve, seeds):
    """{C: d [k, n]}: d = 100 (policy_s - spec_{serve[C], s}) / spec_{serve[C], s} on combination
    C, policy seed s against specialist seed s. pol[s][C] and spec[X][s][C] are the makespans
    of the policy and of specialist X run under C."""
    return {c: np.stack([rel(pol[s][c], spec[serve[c]][s][c]) for s in seeds]) for c in R8}


def option_rows(D, B):
    """Per combination (p88.component on one default_rng(11) stream in the order of p88
    section A) and pooled over the eight (p95.component, a fresh default_rng(11))."""
    comps = stream_components(D, B, STREAM_FIVE)
    pooled = P95.component(np.stack([D[c] for c in R8], axis=2), B)[0]
    return comps, pooled


def oracle_pairs(pol, spec, seeds):
    """The 28 pairs of specialists, each combination served by the one of the two with the
    higher mean of the policy minus the specialist (the shorter schedules relative to the
    policy; a tie goes to the first of the pair in the order of the combinations). Returns the
    pairs sorted by the mean over the eight combinations of the policy minus the option, highest
    first, each with that mean, the mean of the option minus the policy in percent of the policy,
    and the specialist serving each combination."""
    pm = {(x, c): float(np.mean([rel(pol[s][c], spec[x][s][c]) for s in seeds]))
          for x in R8 for c in R8}
    om = {(x, c): float(np.mean([rel(spec[x][s][c], pol[s][c]) for s in seeds]))
          for x in R8 for c in R8}
    rows = []
    for a, b in itertools.combinations(R8, 2):
        serve = {c: (a if pm[(a, c)] >= pm[(b, c)] else b) for c in R8}
        rows.append(dict(pair=[a, b], serve=serve,
                         policy_minus_option=float(np.mean([pm[(serve[c], c)] for c in R8])),
                         option_minus_policy=float(np.mean([om[(serve[c], c)] for c in R8]))))
    rows.sort(key=lambda x: -x['policy_minus_option'])
    for i, x in enumerate(rows):
        x['rank'] = i + 1
    return rows


# ----------------------------------------------------------------- setting and records
def check_setting():
    """Exact environment, active seeds 301 to 305, PACT_OUT set."""
    try:
        SE.require_exact()
    except SE.Refused as e:
        raise Refused(str(e))
    if REG.seeds() != SEEDS_FIVE:
        raise Refused(f'run with PACT_SEEDS={SEEDS_FIVE[0]}-{SEEDS_FIVE[-1]} (the setting of the '
                      f'five-seed records); the ten seeds of analysis A are passed explicitly')
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')


def record_path(key):
    under_out, path, _, _ = RECORDS[key]
    return REG.out(path) if under_out else path


def read_record(key):
    """One record as a dict; refuses with the script and the chain step that write it."""
    p = record_path(key)
    _, _, script, step = RECORDS[key]
    if not os.path.exists(p):
        raise Refused(f'missing input {shown(p)}: written by {script} (step {step} of '
                      f'scripts/run_analysis.sh)')
    try:
        with open(p) as f:
            return json.load(f)
    except ValueError as e:
        raise Refused(f'{shown(p)} is not valid JSON: {e}')


def expect(cond, msg):
    if not cond:
        raise Refused(msg)


def check_records(recs):
    """The settings of the records read: environment, seeds, replicates, level."""
    fam = recs['family']
    expect(tuple(fam.get('seeds', ())) == SEEDS_FIVE and 'environment exact' in fam.get('setting', ''),
           f'{RECORDS["family"][1]}: want the exact environment with seeds 301 to 305, got '
           f'{fam.get("setting")!r}')
    se = recs['seedext']
    expect(tuple(se.get('seeds', ())) == SEEDS_TEN and not se.get('dry'),
           f'{RECORDS["seedext"][1]}: want the ten seeds 301 to 310 of a real run, got '
           f'{se.get("seeds")}')
    ps = recs['prodspec']
    expect(tuple(ps.get('seeds', ())) == SEEDS_FIVE and not ps.get('dry'),
           f'{RECORDS["prodspec"][1]}: want seeds 301 to 305 of a real run, got {ps.get("seeds")}')
    bu = recs['budget']
    expect(tuple(bu.get('seeds', ())) == SEEDS_TEN and not bu.get('dry'),
           f'{RECORDS["budget"][1]}: want seeds 301 to 310 of a real run, got {bu.get("seeds")}')
    sec = recs['secondary']
    expect(sec.get('environment') == 'exact' and sec.get('folder') == FOLDER
           and not sec.get('check'),
           f'{RECORDS["secondary"][1]}: want the exact environment at {FOLDER}, got '
           f'{sec.get("environment")!r} at {sec.get("folder")!r}')
    sp = sec.get('blocks', {}).get('sparse', {})
    expect(sp.get('status') == 'computed',
           f'{RECORDS["secondary"][1]}: block sparse is {sp.get("status")!r}, not computed')
    for key, rec in recs.items():
        expect(int(rec.get('boot', -1)) == B_BOOT and float(rec.get('alpha', -1)) == ALPHA,
               f'{RECORDS[key][1]}: B = {rec.get("boot")}, alpha = {rec.get("alpha")}; want '
               f'{B_BOOT} and {ALPHA}')
    n = SE.n_test()
    expect(int(sec.get('n_instances', -1)) == n,
           f'{RECORDS["secondary"][1]}: {sec.get("n_instances")} test instances, want {n}')


def read_all_records():
    recs = {k: read_record(k) for k in RECORDS}
    check_records(recs)
    return recs


# ----------------------------------------------------------------- arrays
def load_vec(loader, *args):
    try:
        return loader(*args)
    except (REG.MissingResult, SE.Refused) as e:
        raise Refused(f'missing input array: {e}')


def rule_arrays(n):
    """{R: per-instance best of the four rules} at 10x25; refuses on a missing array or another
    length than the test set."""
    out = {}
    for r in R8:
        xs = []
        for q in F88.RULES:
            p = f'{REG.rule_dir(FOLDER)}/Result_{q}_{r}.npy'
            x = load_vec(REG.read_path, p)
            expect(len(x) == n, f'{p}: {len(x)} rows, the test set has {n}')
            xs.append(x)
        out[r] = best_rule(xs)
    return out


def principal_arrays(seeds, n):
    """The policy on every combination and the specialist of each combination on its own, at
    10x25, for the seeds given (second seeds through seed_extension.registry_load)."""
    pol, spec = {}, {}
    for s in seeds:
        for r in R8:
            for arm, combo, store in (('released', None, pol), ('specialist', r, spec)):
                x = load_vec(SE.registry_load, arm, combo, s, FOLDER, r)
                expect(len(x) == n and np.all(np.isfinite(x)) and np.all(x > 0),
                       f'{arm} {combo or ""} seed {s} on {r}: {len(x)} rows or makespans that are '
                       f'not finite and positive')
                store[(s, r)] = x
    return pol, spec


def cross_arrays(root, seeds):
    """The arrays of p100 through its readers and checks: pol[s][C] and spec[X][s][C] (the
    specialist X run under C, from the principal folder when C = X and from the cross folder
    otherwise)."""
    try:
        P100.check_layout()
        n, cache, compared = P100.load_all(root, seeds)
    except (P100.Refused, P95.Refusal) as e:
        raise Refused(str(e).replace('refused: ', '').replace('REFUSING: ', ''))
    except REG.MissingResult as e:
        raise Refused(f'missing input array: {e}')
    pol = {s: {c: cache[(P100.policy_run(s).name, P100.FOLDER, c)] for c in R8} for s in seeds}
    spec = {x: {s: {c: cache[(P100.specialist_run(x, s, P100.FOLDER if c == x else P100.CROSS).name,
                              P100.FOLDER if c == x else P100.CROSS, c)] for c in R8}
                for s in seeds} for x in R8}
    return n, pol, spec, dict(files=len(cache), compared_with_principal=compared)


# ----------------------------------------------------------------- analysis A
def analysis_a(recs, cross):
    n = SE.n_test()
    F88.verify_bases(FOLDER, n)
    best = rule_arrays(n)
    pol10, spec10 = principal_arrays(SEEDS_TEN, n)          # the first five are seeds 301 to 305
    out = {}
    for label, seeds, stream in (('ten', SEEDS_TEN, STREAM_TEN), ('five', SEEDS_FIVE, STREAM_FIVE)):
        D = {r: np.stack([rel(best[r], spec10[(s, r)]) for s in seeds]) for r in R8}
        comps = stream_components(D, B_BOOT, stream)
        with P110.p88_setting(fresh_seed=stream):
            pooled = F88.pooled_block(D, {'All': R8})['All']
        out[label] = dict(seeds=list(seeds), components=comps, pooled=pooled,
                          counts=verdict_counts(comps, R8),
                          stream=f'one default_rng({stream}) stream in the order {ORDER}, a fresh '
                                 f'default_rng({stream}) for the pooled row')
    # the arrays read here give the means of the records of the read columns
    se, fam, ps = recs['seedext'], recs['family'], recs['prodspec']
    checks = []
    for r in R8:
        m = float(np.mean([rel(pol10[(s, r)], spec10[(s, r)]) for s in SEEDS_TEN]))
        expect(abs(m - se['ten'][r]['mean']) <= MEAN_TOL,
               f'{r}: the policy minus the specialist over ten seeds is {m:+.6f} from the arrays '
               f'and {se["ten"][r]["mean"]:+.6f} in {RECORDS["seedext"][1]}')
        m2 = float(np.mean([rel(pol10[(s, r)], best[r]) for s in SEEDS_FIVE]))
        expect(abs(m2 - fam['rule_pool'][r]['mean']) <= MEAN_TOL,
               f'{r}: the policy minus the best rule over five seeds is {m2:+.6f} from the arrays '
               f'and {fam["rule_pool"][r]["mean"]:+.6f} in {RECORDS["family"][1]}')
        checks.append(dict(combination=r, policy_minus_specialist=m, policy_minus_rule=m2))
    pool_rule = ps['side_by_side']['rows'][FOLDER]['rule']
    m3 = float(np.mean([fam['rule_pool'][r]['mean'] for r in R8]))
    expect(abs(m3 - pool_rule['mean']) <= MEAN_TOL,
           f'the pooled policy minus the best rule in {RECORDS["prodspec"][1]} is '
           f'{pool_rule["mean"]:+.6f}, the mean of the eight rows of {RECORDS["family"][1]} '
           f'{m3:+.6f}')
    read = dict(
        policy_minus_specialist=dict(
            source=f'{RECORDS["seedext"][1]} (scripts/p110_seed_extension.py), keys ten and pool_ten',
            seeds=list(SEEDS_TEN),
            components={r: se['ten'][r] for r in R8}, pooled=se['pool_ten']['All']),
        policy_minus_rule=dict(
            source=f'{RECORDS["family"][1]} (scripts/p88_family_inference.py), key rule_pool (the '
                   f'FamRule macros); pooled row from {RECORDS["prodspec"][1]} '
                   f'(scripts/p128_production_specialists.py), side_by_side.rows.{FOLDER}.rule',
            seeds=list(SEEDS_FIVE),
            components={r: fam['rule_pool'][r] for r in R8}, pooled=pool_rule,
            verdict_rule=fam.get('superiority_verdict_rule'),
            pooled_word='direction of the pooled interval (the record has no combined verdict for '
                        'the pooled row)'))
    others = other_comparators(recs, cross)
    return dict(rule_minus_specialist=out, read=read, others=others, checks=checks, n=n)


def other_comparators(recs, cross):
    """The three further comparators against the specialist of each combination."""
    sp = recs['secondary']['blocks']['sparse']['result']
    vs = sp['vs_specialist']
    expect(list(sp['combos']) == R8, f'block sparse covers {sp["combos"]}, want {R8}')
    sparse = dict(source=f'{RECORDS["secondary"][1]} (scripts/p95_secondary_inference.py), block '
                         f'sparse, vs_specialist', arm=sp['arm'], recipe=sp['recipe'],
                  seeds=list(sp['seeds']), unit='combination',
                  rows={r: vs['components'][r] for r in R8}, pooled=vs['pooled']['All'])
    bu = recs['budget']['share_vs_specialist']
    budget = dict(source=f'{RECORDS["budget"][1]} (scripts/p127_budget_specialists.py), '
                         f'share_vs_specialist', seeds=list(recs['budget']['seeds']),
                  updates=recs['budget'].get('updates'), unit='combination',
                  rows={r: bu['components'][r] for r in R8}, pooled=bu['pooled']['All'])
    for x in (sparse, budget):
        x['counts'] = verdict_counts(x['rows'], R8)
    return dict(sparse=sparse, budget=budget, cross=cross)


def cross_comparator(pol, spec, seeds, fam):
    """Each specialist X run under each of its seven other combinations C against the specialist
    of C: d = 100 (spec_X on C - spec_C on C) / spec_C on C, seed s against seed s; p95.component
    per cell, per row (X over its seven C), per column (C over its seven X) and over all 56."""
    D = {(x, c): np.stack([rel(spec[x][s][c], spec[c][s][c]) for s in seeds])
         for x in R8 for c in R8 if c != x}
    cells = {f'{x}|{c}': P95.component(d, B_BOOT)[0] for (x, c), d in D.items()}
    rows = {x: P95.component(np.stack([D[(x, c)] for c in R8 if c != x], axis=2), B_BOOT)[0]
            for x in R8}
    cols = {c: P95.component(np.stack([D[(x, c)] for x in R8 if x != c], axis=2), B_BOOT)[0]
            for c in R8}
    pooled = P95.component(np.stack([D[k] for k in D], axis=2), B_BOOT)[0]
    # the LSW row reproduces section G of the record of p88 (the LSW specialist against the
    # specialist of each other combination)
    g = fam.get('full_combination_specialist') or {}
    grow = (g.get('Spec') or {}).get('rows')
    expect(grow is not None, f'{RECORDS["family"][1]} has no section G (full_combination_specialist)')
    dmax, nchk = 0.0, 0
    for c in R8:
        if c == 'LSW':
            continue
        a, b = cells[f'LSW|{c}'], grow[c]
        for k in ('mean', 'lo', 'hi'):
            nchk += 1
            dmax = max(dmax, abs(a[k] - b[k]))
            expect(f2(a[k]) == f2(b[k]),
                   f'LSW under {c} against the specialist of {c}: {k} {f2(a[k])} here, {f2(b[k])} '
                   f'in section G of {RECORDS["family"][1]}')
        expect(a['v1.0'] == b['v1.0'] and a['v0.5'] == b['v0.5'],
               f'LSW under {c}: verdicts differ from section G of {RECORDS["family"][1]}')
    keys = list(cells)
    return dict(source='computed from the arrays of the cross campaign with the readers of '
                       'scripts/p100_cross_eval.py and p95.component (no record compares these '
                       'runs with the specialist of the combination)',
                seeds=list(seeds), unit='cell (specialist X run under combination C)',
                sign='d = 100 (specialist X under C - specialist of C) / specialist of C',
                cells=cells, rows_by_specialist=rows, columns_by_combination=cols, pooled=pooled,
                counts=verdict_counts(cells, keys),
                counts_by_specialist=verdict_counts(rows, R8),
                counts_by_combination=verdict_counts(cols, R8),
                section_g_check=dict(compared=nchk, identical_at_printed_digits=True,
                                     largest_abs_difference=dmax))


# ----------------------------------------------------------------- analysis B
def analysis_b(pol, spec, seeds, fam):
    res = {}
    for key, opt in OPTIONS.items():
        D = option_d(pol, spec, opt['serve'], seeds)
        comps, pooled = option_rows(D, B_BOOT)
        res[key] = dict(name=opt['name'], what=opt['what'], serve=opt['serve'], components=comps,
                        pooled=pooled, counts=verdict_counts(comps, R8),
                        ahead=[c for c in R8 if comps[c]['direction'] == 'ahead'],
                        behind=[c for c in R8 if comps[c]['direction'] == 'behind'],
                        iut=F88.iut_block(comps, {'All': R8}))
    check = check_principal(res['Four'], fam)
    return res, check


def check_principal(b4, fam):
    """B4 against the five-seed record (results/family_inference.json, parity): every number
    to the printed digits (two decimals), verdict words and directions exactly."""
    st = fam['parity']
    bad, n, dmax = [], 0, 0.0

    def num(where, a, b):
        nonlocal n, dmax
        n += 1
        dmax = max(dmax, abs(float(a) - float(b)))
        if f2(a) != f2(b):
            bad.append(f'{where}: record {f2(b)}, here {f2(a)}')

    def word(where, a, b):
        nonlocal n
        n += 1
        if a != b:
            bad.append(f'{where}: record {b!r}, here {a!r}')
    for r in R8:
        a, b = b4['components'][r], st['components'][r]
        for k in ('mean', 'lo', 'hi', 'sim_lo', 'sim_hi', 't_lo', 't_hi'):
            num(f'{r} {k}', a[k], b[k])
        for k in ('v0.5', 'v1.0', 'direction'):
            word(f'{r} {k}', a[k], b[k])
    a, b = b4['pooled'], st['pooled']['All']
    for k in ('mean', 'lo', 'hi', 't_lo', 't_hi'):
        num(f'pooled {k}', a[k], b[k])
    for k in ('v0.5', 'v1.0', 'direction'):
        word(f'pooled {k}', a[k], b[k])
    if bad:
        raise Refused(f'B4 does not reproduce the five-seed principal comparison of '
                      f'{RECORDS["family"][1]} ({len(bad)} of {n}): ' + '; '.join(bad[:6]))
    return dict(record=RECORDS['family'][1], compared=n, identical_at_printed_digits=True,
                largest_abs_difference=dmax)


# ----------------------------------------------------------------- macros
def comp_macros(name, c, what, dir_word=True):
    s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
         f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
         f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
         + ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}' for mg in MARGINS))
    if dir_word:
        s += f'\\newcommand{{\\{name}Dir}}{{{tex(c["direction"])}}}'
    return (s + f'  % {what}: mean difference (%), 95% interval, verdicts at the 0.5% and 1% '
            f'margins' + (', direction of the interval' if dir_word else '')
            + f' [{c["lo"]:+.4f}, {c["hi"]:+.4f}]')


def count_macros(name, counts, what, with_lists=True):
    L = []
    for mg in MARGINS:
        cm = counts[f'v{mg}']
        s = ''.join(f'\\newcommand{{\\{name}{w}{MW[mg]}N}}{{{len(cm[word])}}}'
                    for word, w in VERDICT_KEYS)
        if with_lists:
            s += ''.join(f'\\newcommand{{\\{name}{w}{MW[mg]}List}}{{{running_list(cm[word])}}}'
                         for word, w in VERDICT_KEYS)
        L.append(s + f'  % {what} at the {mg:g}% margin: number of combinations equivalent, '
                 f'non-inferior only, inferior, inconclusive'
                 + (', and the combinations of each as running text' if with_lists else ''))
    return L


def macro_lines(A, Bres):
    rm = A['rule_minus_specialist']['ten']
    rd = A['read']
    L = [f'% Generated by {GENERATOR} (environment exact) -- do not hand-edit',
         '% A difference is the makespan of the first-named method minus that of the second, in '
         'percent of the second; positive means a longer schedule for the first-named method.',
         f'% Msn: analysis A, the verdict rule of the principal comparison applied to comparators '
         f'known to differ from the specialist (10x25). Tws: analysis B, the policy against '
         f'options that serve each combination with one specialist, seeds 301 to 305. B = '
         f'{B_BOOT}.',
         f'\\newcommand{{\\{PREFIX_A}Boot}}{{{B_BOOT}}}'
         f'\\newcommand{{\\{PREFIX_A}InstN}}{{{A["n"]}}}'
         f'\\newcommand{{\\{PREFIX_A}SeedN}}{{{len(SEEDS_TEN)}}}'
         f'\\newcommand{{\\{PREFIX_A}PolRuleSeedN}}{{{len(rd["policy_minus_rule"]["seeds"])}}}'
         f'  % bootstrap replicates, test instances, training seeds of the best rule minus the '
         f'specialist and of the policy minus the specialist, training seeds of the policy minus '
         f'the best rule']
    for r in R8:
        L.append(comp_macros(f'{PREFIX_A}Rule{r}', rm['components'][r],
                             f'best of four rules minus the specialist of {r}, ten seeds'))
    L.append(comp_macros(f'{PREFIX_A}RulePool', rm['pooled'],
                         'best of four rules minus the specialist, pooled over the eight '
                         'combinations, ten seeds'))
    L += count_macros(f'{PREFIX_A}Rule', rm['counts'], 'best of four rules minus the specialist')
    for r in R8:
        L.append(comp_macros(f'{PREFIX_A}Pol{r}', rd['policy_minus_specialist']['components'][r],
                             f'policy minus the specialist of {r}, ten seeds, read from the record '
                             f'of p110'))
    L.append(comp_macros(f'{PREFIX_A}PolPool', rd['policy_minus_specialist']['pooled'],
                         'policy minus the specialist, pooled over the eight combinations, ten '
                         'seeds, read from the record of p110'))
    for r in R8:
        x = rd['policy_minus_rule']['components'][r]
        L.append(f'\\newcommand{{\\{PREFIX_A}PolRule{r}}}{{{f2(x["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX_A}PolRule{r}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX_A}PolRule{r}Hi}}{{{f2(x["hi"])}}}'
                 f'\\newcommand{{\\{PREFIX_A}PolRule{r}Verdict}}{{{x["verdict"]}}}'
                 f'  % policy minus the best of four rules on {r}, five seeds, read from the record '
                 f'of p88 (FamRule{r}): mean difference (%), 95% interval, verdict with Holm\'s '
                 f'correction over the eight combinations')
    x = rd['policy_minus_rule']['pooled']
    L.append(f'\\newcommand{{\\{PREFIX_A}PolRulePool}}{{{f2(x["mean"])}}}'
             f'\\newcommand{{\\{PREFIX_A}PolRulePoolLo}}{{{f2(x["lo"])}}}'
             f'\\newcommand{{\\{PREFIX_A}PolRulePoolHi}}{{{f2(x["hi"])}}}'
             f'\\newcommand{{\\{PREFIX_A}PolRulePoolDir}}{{{tex(x["direction"])}}}'
             f'  % policy minus the best of four rules pooled over the eight combinations, five '
             f'seeds, read from the record of p128: mean difference (%), 95% interval, direction '
             f'of the interval [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')
    oth = A['others']
    for key, word in COMPARATORS:
        x = oth[key]
        units = len(x['cells']) if key == 'cross' else len(R8)
        L.append(f'\\newcommand{{\\{PREFIX_A}{word}SeedN}}{{{len(x["seeds"])}}}'
                 f'\\newcommand{{\\{PREFIX_A}{word}UnitN}}{{{units}}}'
                 f'  % {key}: training seeds, comparisons counted ('
                 + ('cells of a specialist under one of its seven other combinations'
                    if key == 'cross' else 'combinations') + ')')
        L += count_macros(f'{PREFIX_A}{word}', x['counts'],
                          f'{key} against the specialist of the combination', with_lists=False)
        L.append(comp_macros(f'{PREFIX_A}{word}Pool', x['pooled'],
                             f'{key} against the specialist of the combination, pooled over all '
                             f'{units} comparisons'))
    c = oth['cross']
    L += count_macros(f'{PREFIX_A}CrossRow', c['counts_by_specialist'],
                      'cross, each specialist pooled over its seven other combinations (eight '
                      'specialists counted)', with_lists=False)
    L += count_macros(f'{PREFIX_A}CrossCol', c['counts_by_combination'],
                      'cross, each combination pooled over the seven specialists not trained for '
                      'it (eight combinations counted)', with_lists=False)
    # analysis B
    L.append(f'\\newcommand{{\\{PREFIX_B}Boot}}{{{B_BOOT}}}'
             f'\\newcommand{{\\{PREFIX_B}SeedN}}{{{len(SEEDS_FIVE)}}}'
             f'\\newcommand{{\\{PREFIX_B}InstN}}{{{A["n"]}}}'
             f'  % analysis B: bootstrap replicates, training seeds, test instances')
    for key, opt in OPTIONS.items():
        x = Bres[key]
        for r in R8:
            L.append(comp_macros(f'{PREFIX_B}{key}{r}', x['components'][r],
                                 f'{opt["name"]}, policy minus the specialist of '
                                 f'{opt["serve"][r]} run under {r}'))
            if key in SPEC_MACRO_OPTIONS:
                L.append(f'\\newcommand{{\\{PREFIX_B}{key}{r}Spec}}{{{opt["serve"][r]}}}'
                         f'  % {opt["name"]}: the specialist that serves {r}')
        L.append(comp_macros(f'{PREFIX_B}{key}Pool', x['pooled'],
                             f'{opt["name"]}, policy minus the option, pooled over the eight '
                             f'combinations'))
        cm = x['counts']
        L.append(f'\\newcommand{{\\{PREFIX_B}{key}EqOneN}}{{{len(cm["v1.0"]["equivalent"])}}}'
                 f'\\newcommand{{\\{PREFIX_B}{key}EqHalfN}}{{{len(cm["v0.5"]["equivalent"])}}}'
                 f'\\newcommand{{\\{PREFIX_B}{key}AheadN}}{{{len(x["ahead"])}}}'
                 f'\\newcommand{{\\{PREFIX_B}{key}BehindN}}{{{len(x["behind"])}}}'
                 f'  % {opt["name"]}: combinations equivalent at the 1% and at the 0.5% margin, '
                 f'combinations whose interval lies below zero (policy shorter) and above zero')
    return L


def macro_names(lines):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))


def defined_elsewhere(out_mac):
    """{name: file} of every macro defined in the macro files of paper/ and manuscript/ and in the
    staged sections, the output file and its copy in manuscript/ left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\*?\s*\{?\\([A-Za-z]+)')
    files = set()
    for pat in ('manuscript/macros*.tex', 'paper/macros*.tex', 'manuscript/sections_new/**/*.tex',
                'manuscript/supp_new/**/*.tex'):
        files |= set(glob.glob(os.path.join(ROOT, pat), recursive=True))
    own = {os.path.abspath(out_mac),
           os.path.abspath(os.path.join(ROOT, 'manuscript', os.path.basename(out_mac)))}
    out = {}
    for f in sorted(files):
        if os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            for name in rx.findall(fh.read()):
                out.setdefault(name, os.path.relpath(f, ROOT))
    return out


def check_names(names, out_mac):
    bad = [x for x in names if not re.fullmatch(r'(Msn|Tws)[A-Za-z]+', x)]
    expect(not bad, f'macro names that are not Msn or Tws plus letters: {bad[:5]}')
    dup = sorted({x for x in names if names.count(x) > 1})
    expect(not dup, f'duplicate macro names in the output: {dup[:5]}')
    other = defined_elsewhere(out_mac)
    clash = sorted(x for x in other if x.startswith(PREFIX_A) or x.startswith(PREFIX_B))
    expect(not clash, f'the prefix of {clash[0]} is in use in {other[clash[0]]}' if clash else '')
    return len(names)


# ----------------------------------------------------------------- Markdown
def md_counts(counts, what):
    out = []
    for mg in MARGINS:
        cm = counts[f'v{mg}']
        out.append(f'- {what}, {mg:g}% margin: ' + '; '.join(
            f'{w} {len(cm[w])}' + (f' ({", ".join(cm[w])})' if cm[w] else '')
            for w in SF.VERDICTS))
    return out


def markdown_a(A):
    rm, rd, oth = A['rule_minus_specialist'], A['read'], A['others']
    md = ['# What the equivalence test returns for comparators known to differ from the specialist',
          '', f'Factory benchmark at the training size ({A["n"]} test instances), exact mask. A '
          'difference is the makespan of the first-named method minus that of the second, in '
          'percent of the second (positive = the first-named method is longer). Verdicts at the '
          f'0.5% and 1% margins from the 95% interval (B = {B_BOOT}).', '', NOTES_A, '',
          '## Best of four rules minus the specialist, beside the two read comparisons', '',
          f'Best rule minus specialist: ten seeds ({rm["ten"]["stream"]}). Policy minus specialist: '
          f'{rd["policy_minus_specialist"]["source"]}. Policy minus best rule: '
          f'{rd["policy_minus_rule"]["source"]}; the verdict is the combined verdict with Holm '
          'over the eight combinations, the pooled row gives the direction.', '',
          '| comb. | rule - spec. | 0.5% | 1% | dir. | policy - spec. | 0.5% | 1% | '
          'policy - rule | verdict |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        a, b = rm['ten']['components'][r], rd['policy_minus_specialist']['components'][r]
        c = rd['policy_minus_rule']['components'][r]
        md.append(f'| {r} | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | {a["direction"]} | {ci(b)} | '
                  f'{b["v0.5"]} | {b["v1.0"]} | {ci(c)} | {c["verdict"]} |')
    a, b = rm['ten']['pooled'], rd['policy_minus_specialist']['pooled']
    c = rd['policy_minus_rule']['pooled']
    md.append(f'| pooled | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | {a["direction"]} | {ci(b)} | '
              f'{b["v0.5"]} | {b["v1.0"]} | {ci(c)} | {c["direction"]} |')
    md += ['', 'Counts of the best rule minus the specialist (ten seeds):', '']
    md += md_counts(rm['ten']['counts'], 'best rule against the specialist')
    md += ['', '## The same with the five seeds 301 to 305', '',
           f'Streams of the five-seed record ({rm["five"]["stream"]}).', '',
           '| comb. | rule - spec. | seed-t CI | 0.5% | 1% | direction |', '|---|---|---|---|---|---|']
    for r in R8 + ['pooled']:
        a = rm['five']['pooled'] if r == 'pooled' else rm['five']['components'][r]
        md.append(f'| {r} | {ci(a)} | [{f2(a["t_lo"])}, {f2(a["t_hi"])}] | {a["v0.5"]} | '
                  f'{a["v1.0"]} | {a["direction"]} |')
    md += ['']
    md += md_counts(rm['five']['counts'], 'best rule against the specialist, five seeds')
    md += ['', '## Further comparators against the specialist of each combination', '',
           '| comparator | seeds | comparisons | pooled | 0.5% | 1% | source |', '|---|---|---|---|---|---|---|']
    for key, _ in COMPARATORS:
        x = oth[key]
        units = len(x['cells']) if key == 'cross' else len(R8)
        md.append(f'| {key} | {len(x["seeds"])} | {units} | {ci(x["pooled"])} | '
                  f'{x["pooled"]["v0.5"]} | {x["pooled"]["v1.0"]} | {x["source"]} |')
    md += ['']
    for key, _ in COMPARATORS:
        md += md_counts(oth[key]['counts'], key + (' (56 cells)' if key == 'cross' else ''))
    c = oth['cross']
    md += md_counts(c['counts_by_specialist'], 'cross, each specialist pooled over its seven other '
                    'combinations')
    md += md_counts(c['counts_by_combination'], 'cross, each combination pooled over the seven '
                    'specialists not trained for it')
    md += ['', 'Sparse reward and budget, per combination (read):', '',
           '| comb. | sparse | 0.5% | 1% | budget | 0.5% | 1% |', '|---|---|---|---|---|---|---|']
    for r in R8:
        s, b = oth['sparse']['rows'][r], oth['budget']['rows'][r]
        md.append(f'| {r} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} | {ci(b)} | {b["v0.5"]} | '
                  f'{b["v1.0"]} |')
    md += ['', f'Cross: specialist (row) run under combination (column) against the specialist of '
           f'the column, mean [95% interval], verdict at 1% ({c["sign"]}):', '',
           '| specialist | ' + ' | '.join(R8) + ' | pooled over its seven |', '|---' * 10 + '|']
    for x in R8:
        cells = [('-' if x == col else f'{ci(c["cells"][f"{x}|{col}"])} '
                  f'{c["cells"][f"{x}|{col}"]["v1.0"]}') for col in R8]
        rw = c['rows_by_specialist'][x]
        md.append(f'| {x} | ' + ' | '.join(cells) + f' | {ci(rw)} {rw["v1.0"]} |')
    md.append('| pooled over the seven others | ' + ' | '.join(
        f'{ci(c["columns_by_combination"][col])} {c["columns_by_combination"][col]["v1.0"]}'
        for col in R8) + f' | all 56: {ci(c["pooled"])} {c["pooled"]["v1.0"]} |')
    g = c['section_g_check']
    md += ['', '## Checks', '',
           f'- The LSW row of the cross comparison reproduces section G of '
           f'{RECORDS["family"][1]}: {g["compared"]} numbers identical at the printed digits '
           f'(largest absolute difference {g["largest_abs_difference"]:.2e}).',
           f'- The arrays read here give the means of the read records to {MEAN_TOL:g}: the policy '
           f'minus the specialist over ten seeds ({RECORDS["seedext"][1]}), the policy minus the '
           f'best rule over five seeds ({RECORDS["family"][1]}), and the pooled policy minus the '
           f'best rule ({RECORDS["prodspec"][1]}).', '']
    return md


def markdown_b(Bres, check, oracle, info):
    md = ['# The policy against two specialists that serve all eight combinations', '',
          f'Factory benchmark at the training size, exact mask, seeds '
          f'{", ".join(map(str, SEEDS_FIVE))}, policy seed s against specialist seed s. A '
          'difference is the makespan of the policy minus that of the serving specialist, in '
          'percent of the specialist (negative = the policy is shorter). Per combination the '
          f'crossed hierarchical bootstrap with the draws of one default_rng({STREAM_FIVE}) stream '
          'in the order of p88 section A (every option meets the same resamples); pooled over the '
          f'eight with a fresh default_rng({P95.HIER_SEED}), a drawn base instance with all its '
          f'combinations. B = {B_BOOT}.', '', NOTES_B, '',
          f'Inputs: {info["files"]} arrays read and checked by the readers of '
          f'scripts/p100_cross_eval.py; {info["compared_with_principal"]} off-combination arrays '
          'equal to those of the principal campaign.', '']
    for key, opt in OPTIONS.items():
        x = Bres[key]
        md += [f'## {opt["name"]}: {opt["what"]}', '',
               '| comb. | served by | policy - option | seed-t CI | 0.5% | 1% | direction |',
               '|---|---|---|---|---|---|---|']
        for r in R8:
            a = x['components'][r]
            md.append(f'| {r} | {opt["serve"][r]} | {ci(a)} | [{f2(a["t_lo"])}, {f2(a["t_hi"])}] | '
                      f'{a["v0.5"]} | {a["v1.0"]} | {a["direction"]} |')
        a = x['pooled']
        md.append(f'| pooled | - | {ci(a)} | [{f2(a["t_lo"])}, {f2(a["t_hi"])}] | {a["v0.5"]} | '
                  f'{a["v1.0"]} | {a["direction"]} |')
        md += ['']
        md += md_counts(x['counts'], opt['name'])
        md += [f'- {opt["name"]}: interval below zero (policy shorter) on '
               f'{", ".join(x["ahead"]) or "none"}; above zero on {", ".join(x["behind"]) or "none"}.',
               '']
    md += [f'B4 reproduces the five-seed principal comparison of {check["record"]}: '
           f'{check["compared"]} numbers and words identical at the printed digits (largest '
           f'absolute difference {check["largest_abs_difference"]:.2e}).', '',
           '## Oracle pairs', '',
           'For each pair of specialists, each combination is served by the one of the two with '
           'the higher mean of the policy minus the specialist on the test instances (an oracle '
           'choice made on the test instances). Columns: the mean over the eight combinations of '
           'the policy minus the option (in percent of the option) and of the option minus the '
           'policy (in percent of the policy), and the serving specialist of N, L, S, W, LS, LW, '
           'SW, LSW. Sorted by the first column, highest first: the first pair gives the shortest '
           'schedules relative to the policy.', '',
           '| rank | pair | policy - option | option - policy | served by |', '|---|---|---|---|---|']
    for x in oracle['pairs']:
        md.append(f'| {x["rank"]} | {x["pair"][0]}, {x["pair"][1]} | '
                  f'{x["policy_minus_option"]:+.3f} | {x["option_minus_policy"]:+.3f} | '
                  + ' '.join(x['serve'][c] for c in R8) + ' |')
    md += ['', f'Best pair: {", ".join(oracle["best"]["pair"])}; by the option minus the policy '
           f'the lowest mean is the pair {", ".join(oracle["best_by_option_minus_policy"]["pair"])}. '
           f'The pair of B1 (N, LSW) ranks {oracle["rank_of"]["N,LSW"]} and the pair of B2 (N, L) '
           f'ranks {oracle["rank_of"]["N,L"]} of {len(oracle["pairs"])}.', '']
    return md


# ----------------------------------------------------------------- run
def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def analyse(root=ROOT):
    """Every input and check, then both analyses; returns (A, B, check of B4, oracle, info)."""
    check_setting()
    F88.configure(boot=B_BOOT, alpha=ALPHA)
    recs = read_all_records()
    n, pol, spec, info = cross_arrays(root, list(SEEDS_FIVE))
    expect(n == SE.n_test(), f'{n} test instances in the cross campaign, want {SE.n_test()}')
    cross = cross_comparator(pol, spec, list(SEEDS_FIVE), recs['family'])
    A = analysis_a(recs, cross)
    Bres, check = analysis_b(pol, spec, list(SEEDS_FIVE), recs['family'])
    pairs = oracle_pairs(pol, spec, list(SEEDS_FIVE))
    rank_of = {','.join(x['pair']): x['rank'] for x in pairs}
    by_om = min(pairs, key=lambda x: x['option_minus_policy'])
    oracle = dict(pairs=pairs, best=pairs[0], best_by_option_minus_policy=by_om,
                  same_best_under_both_signs=by_om['pair'] == pairs[0]['pair'],
                  rank_of=rank_of,
                  rule='each combination served by the one of the two specialists with the higher '
                       'mean of the policy minus the specialist (in percent of the specialist) on '
                       'the test instances; a tie goes to the first of the pair in the order of '
                       'the combinations; pairs sorted by the mean over the eight combinations of '
                       'the policy minus the option, highest first',
                  note='an oracle choice made on the test instances, an optimistic bound on a '
                       'choice made before testing')
    return A, Bres, check, oracle, info


def write(A, Bres, check, oracle, info, outs):
    lines = macro_lines(A, Bres)
    names = macro_names(lines)
    check_names(names, outs['macros'])
    common = dict(generator=GENERATOR, environment='exact', folder=FOLDER, n_instances=A['n'],
                  boot=B_BOOT, alpha=ALPHA, margins=list(MARGINS),
                  verdict_rule='stats_family.verdict on the 95% interval: equivalent -m < lo and '
                               'hi < m; non-inferior only hi < m and lo <= -m; inferior lo > 0 and '
                               'hi >= m; inconclusive otherwise')
    ja = dict(common, notes=NOTES_A,
              sign='d = 100 (first-named - second) / second; the comparators are first-named, the '
                   'specialist of the combination second; positive = the comparator is longer',
              best_rule='per instance the shortest of the FIFO, MOR, SPT and MWKR schedules '
                        '(p88_family_inference.pool); the rule makespans do not depend on a '
                        'training seed',
              rule_minus_specialist=A['rule_minus_specialist'], read=A['read'],
              other_comparators=A['others'], checks=dict(arrays_match_records=A['checks']))
    jb = dict(common, notes=NOTES_B, seeds=list(SEEDS_FIVE),
              sign='d = 100 (policy - serving specialist) / serving specialist; negative = the '
                   'policy is shorter',
              pairing='policy seed s against specialist seed s, instance i against instance i',
              streams=dict(per_combination=f'one default_rng({STREAM_FIVE}) stream in the order '
                                           f'{ORDER} per option (the stream of the five-seed '
                                           f'record)',
                           pooled=f'fresh default_rng({P95.HIER_SEED}) (p95.component)'),
              inputs=info, options=Bres, principal_check=check, oracle_pairs=oracle)
    md_a, md_b = markdown_a(A), markdown_b(Bres, check, oracle, info)
    emit(outs['macros'], lines)
    for key, obj in (('margin_json', ja), ('pairs_json', jb)):
        with open(outs[key], 'w') as f:
            json.dump(jsonable(obj), f, indent=1)
    for key, md in (('margin_md', md_a), ('pairs_md', md_b)):
        with open(outs[key], 'w') as f:
            f.write('\n'.join(md) + '\n')
    return names, md_a, md_b


def run():
    check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        A, Bres, check, oracle, info = analyse()
        names, md_a, md_b = write(A, Bres, check, oracle, info, outs)
    except BaseException:
        remove_outputs(outs)
        raise
    return names, md_a, md_b, outs


def check_inputs(root=ROOT):
    """Every input present and in the expected setting; reads only. Returns the problems."""
    problems = []
    try:
        check_setting()
        read_all_records()
    except Refused as e:
        return [str(e)]
    n = SE.n_test()
    try:
        rule_arrays(n)
        principal_arrays(SEEDS_TEN, n)
    except Refused as e:
        problems.append(str(e))
    try:
        cross_arrays(root, list(SEEDS_FIVE))
    except Refused as e:
        problems.append(str(e).splitlines()[0])
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true',
                    help='report whether every input is present; reads only, writes nothing')
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.check:
        try:
            problems = check_inputs()
        except BaseException as e:            # a refusal of a reader is a 'no'
            problems = [f'{type(e).__name__}: {str(e).strip().splitlines()[0] if str(e).strip() else ""}']
        for p in problems:
            print(f'p140: {p}')
        if problems:
            sys.exit(1)
        print('p140: every record and array present; seeds 301 to 310 (analysis A) and 301 to 305 '
              '(analysis B)')
        return
    REG.banner()
    try:
        names, md_a, md_b, outs = run()
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    print('\n'.join(md_a + md_b))
    print(f'{len(names)} macros; wrote ' + ', '.join(outs.values()))


if __name__ == '__main__':
    main()
