r"""P88: parity, control, held-out and production-scale inference with the
training run as the unit.

The submitted analysis (p11, p34, p60) treats the test instance as the unit: it
pools the training seeds' makespans per instance and resamples instances. This
script adds the analyses two first-touch reviews asked for, for any number of
training seeds and either environment (scripts/arm_registry.py). It writes new
macros only (paper/macros_family.tex, every name starts with Fam) and changes
no existing number.

UNITS AND ESTIMATORS (scripts/stats_family.py)
  d = 100 (a - b) / b per training seed and test instance, where a and b are two
  runs paired by their training-seed index (seed s of one arm against seed s of
  the other, as p27 pairs them). For the parity, control and rule comparisons
  positive d means the first arm is SLOWER. For production scale the sign is the
  one of p60 and p82: 100 (rival - policy) / policy, positive = policy ahead.
  * Crossed hierarchical bootstrap: per replicate draw k seed indices, then n
    base-instance indices; one instance draw serves every drawn seed (crossed
    design); a statistic pooled over combinations keeps the combinations of a
    drawn base instance together (cluster by base instance).
  * Seed-level t: Student t over the k per-seed means (k - 1 df).
  * Instance-level cluster bootstrap (fixed policies, p84's procedure): for the
    seed-pooled estimator and for one checkpoint at production scale.
  B = --boot (default 20,000). Random streams: the parity components draw from
  ONE default_rng(11) stream in the order N, L, S, W, LS, LW, SW, LSW, exactly as
  p27 does, so with the three stored seeds the per-combination intervals equal
  the published seed-level intervals of macros_seed.tex (checked; the script
  refuses on a mismatch). Every other hierarchical quantity uses a fresh
  default_rng(11); every instance-level quantity a fresh default_rng(7).

VERDICTS (stats_family.verdict; one rule, one 95% interval, both margins)
  equivalent         -delta < lo and hi < delta
  non-inferior only  hi < delta and lo <= -delta
  inferior           lo > 0 and hi >= delta
  inconclusive       otherwise
  reported at delta = 0.5% and 1%, with the direction (ahead / behind / neither)
  of the same interval reported separately.

FAMILIES AND THE CLAIMS EACH PROCEDURE COVERS
  A. Parity, released policy vs the specialist of each combination, 10x25,
     all eight combinations. Joint claims by the intersection-union test (IUT):
     "equivalent (non-inferior) on all eight", "... on the six training
     combinations", "... on the two held-out combinations" hold at level alpha
     iff every component verdict does; no multiplicity correction is needed or
     granted. Each component's 95% interval inside the band is a pair of
     one-sided tests at 2.5% each. This covers the E1 and E2 parity statements
     (macros_eq, macros_seed). Bonferroni-simultaneous intervals (alpha / 8)
     from the same draws are reported beside them, for a reader who wants all
     eight intervals to hold jointly. The pooled parity over a family keeps the
     combinations of a base instance together. The published instance-level
     intervals (p11, macros_eq.tex) are read, not recomputed, and given the same
     verdict rule and IUT (FamParInst*), so the change of rule and the change of
     unit can be told apart.
  B. Superiority over the rule pool (per-instance best of FIFO, MOR, SPT,
     MWKR), released policy, 10x25, all eight combinations. Discoveries, so Holm
     over the eight one-sided seed-level t tests (H1: the policy is faster); the
     hierarchical one-sided bootstrap p is reported with its own Holm column.
     This covers the per-combination "ahead of the rule pool" statements.
  C. Single-rule control (trained on N, L, S, W), 10x25_blindref (one
     campaign for every arm): control vs released policy and control vs
     specialist on all eight, IUT over the four compositions the control never
     saw (LS, LW, SW, LSW) and over all eight; control vs rule pool with Holm
     over eight. Covers the single-rule-control statements (macros_singleblind).
  D. Held-out compositions: the two held-out rows of A (IUT family Held) and of
     B. Covers the zero-shot statements on SW and LSW.
  E. Single deployable checkpoint at production scale, 50x25 and 80x25. The
     selection rule is fixed before the new runs: the training seed whose kept
     checkpoint has the best validation score (arm_registry.select_seed; the
     trainer keeps the checkpoint at the minimum of valiquality_<model>.txt).
     Per seed and for the selected seed, separately from the seed mean: the
     lead over the 1800 s warm-started CP-SAT (p60/p82 credit rule, the first
     10 instances per regime, the instances the solver ran on) and over the rule
     pool (every stored instance; with --n30 also the 30-instance
     sets). Instance-level cluster bootstrap by base instance over the eight
     regimes. The seed-mean row is p60's estimator (makespans averaged over the
     seeds, then the gap).
     In the exact environment the solver rows are skipped: the paper's solver
     comparison at the production batches comes from scripts/p94_timed_inference.py
     (the timed campaign) only, so the folders or_solution/FAMILY/<size>_fairwarm1800s
     are not read, no macro of them is written, and the JSON (key
     production_solver_1800s) and the Markdown say so; the rule-pool rows stay,
     with the base instances of every stored instance checked. A comparison that
     would hold no instance (a regime without a solver record, a rule file without a
     row) refuses before any output is removed.

  E'. The same rows under the paper's convention, policy minus rival over rival
     (d = 100 (policy - rival) / rival per instance, negative = policy shorter),
     recomputed from the arrays; this is a different statistic from the lead,
     because the denominator is the rival's makespan. Same sources, the same
     instance-level cluster bootstrap (fresh default_rng(7) per quantity, B
     resamples), pooled and per regime; a regime or a seed counts as ahead when its
     interval lies below zero. Macros: every production name with the suffix Rel
     (FamProd<set><rival><row>Rel, ...LoRel, ...HiRel, ...AheadNRel, ...BehindNRel,
     ...SeedMinRel, ...SeedMaxRel, ...SeedClearNRel, ...SeedTieNRel,
     ...SeedBehindNRel), appended after every existing macro; no existing name ends
     in Rel, so no name can collide. SeedMinRel and SeedMaxRel are the smallest and
     largest single-seed difference, so the seed with the largest lead gives
     SeedMinRel. JSON key production_policy_minus_rival.

COMBINED VERDICT AGAINST THE RULE POOL (B and C)
  ahead                the hierarchical 95% interval lies below zero and both
                       Holm-adjusted one-sided p-values for H1: d < 0 (seed-level t,
                       hierarchical bootstrap) are below alpha;
  behind               the interval lies above zero and both Holm-adjusted one-sided
                       p-values for H1: d > 0 are below alpha, each kind with its own
                       Holm family over the same eight combinations;
  not distinguishable  otherwise.
  The rule is p89_public_family.superiority_verdict (one definition for both
  families). The H1: d > 0 p-values use stats_family.seed_t_p and boot_p_one_sided
  with alternative 'greater', on the same per-seed means and the same hierarchical
  draws, so no random number is added. Macros FamRule<R>Verdict and
  FamCtlRule<R>Verdict and the counts FamRuleVerdictAheadN, FamRuleVerdictBehindN,
  FamCtlRuleVerdictAheadN, FamCtlRuleVerdictBehindN, appended; JSON keys
  p_t_greater, p_boot_greater, p_t_greater_holm, p_boot_greater_holm and verdict in
  each row of rule_pool and control.rule_pool.

DIRECTION OF THE POOLED ROWS
  FamParPool<F>Dir, FamParPoolInst<F>Dir, FamCtlRelPool<F>Dir and FamCtlSpecPool<F>Dir:
  ahead, behind or neither, from stats_family.direction on the unrounded interval
  ends (the rule of the per-combination Dir macros), appended after every other
  macro; the comment of each shows the ends to four decimals, because a bound close
  to zero prints as -0.00 or +0.00. JSON: the pooled rows already
  hold 'direction'; parity.pooled_instance_level.<F> gets it as a new key.

SECTIONS F, G, H (appended after every macro above; JSON keys rule_change,
full_combination_specialist, larger_shops and sections)
  All three use d = 100 (first - second) / second with the first-named method first, seed
  s against seed s, hierarchical intervals from a fresh default_rng(11) per quantity.
  F. Rule-change pairs, 10x25: the twelve pairs (A, B) with B = A plus one rule, on B.
     FamChgPol<A>To<B>: the released policy against the specialist of A (the specialist
     still in service after the rule was added); FamChgSpec<A>To<B>: the specialist of B
     against the specialist of A (what the change costs a catalogue of specialists that is
     not retrained). Per pair the interval, the verdicts at both margins, the direction and
     the combined verdict with Holm over the twelve pairs, each direction its own family;
     the twelve pooled (a base instance with all its pairs); counts of pairs ahead, behind,
     not distinguishable.
  G. The specialist of the full combination LSW on every combination of the campaign that
     evaluates it on all eight (10x25_blindref; the 10x25 campaign in the exact
     environment): FamFullPol<R> against the released policy (eight combinations),
     FamFullSpec<R> against the specialist of R (the seven others); verdicts at both
     margins and direction; pooled rows.
  H. Larger shops. 20x25 and 30x25 (FamShopTwenty, FamShopThirty): the released policy
     against the specialist of each combination (verdicts, direction, IUT and pooled rows
     over the families of A) and against the per-instance best of the four rules (combined
     verdict, Holm over the eight combinations of the size). 50x15 and 80x15
     (FamShopFiftyFifteen, FamShopEightyFifteen): against the same rule quantity, every run
     (hierarchical, combined verdict) and the selected run of E (Sel: instance-level cluster
     bootstrap, fresh default_rng(7) per quantity, one-sided bootstrap p in both directions
     with their own Holm families, superiority_verdict with the bootstrap p in place of the
     seed-level p, which one run does not have).
  A section is computed when every input array of the active environment and seeds exists
  with as many rows as its folder has test instances (descriptors in data/FAMILY); otherwise
  the JSON (sections) and the Markdown list it as skipped with the reason and the macro file
  carries one comment line for it. With --require-sections all (or a comma list) a skipped
  section refuses before anything is written.

Post hoc options (best of the checkpoints, the policy-seeded hybrid) are not
computed here; they stay in p82 and p81 and are labelled exploratory there.

The functions can be imported (scripts/p96_two_environments.py does); the analysis runs
only when the file is run as a script, and configure() sets B and ALPHA for an importer.

Usage: python scripts/p88_family_inference.py [--boot 20000] [--n30]
       [--require-sections none|all|F,G,H]
       [--pact-seeds 301-305] [--pact-env exact]   (outputs need PACT_OUT outside
       the submitted setting)
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS, superiority_verdict  # noqa: E402  (one rule, both families)

ap = argparse.ArgumentParser()
ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
ap.add_argument('--n30', action='store_true',
                help='also compare with the rule pool on the 30-instance production sets')
ap.add_argument('--alpha', type=float, default=0.05)
ap.add_argument('--require-sections', default='none',
                help='refuse when one of these sections is skipped for want of its input '
                     'arrays: none (default), all, or a comma list of F, G, H')
B, ALPHA = SF.B_DEFAULT, 0.05   # set from the arguments when run as a script (configure)
if __name__ == '__main__':
    ARGS = ap.parse_args()
    REG.banner()
    B, ALPHA = ARGS.boot, ARGS.alpha
SEEDS = REG.seeds()
K = len(SEEDS)
WORDS = REG.seed_words()
R8 = list(REG.COMBOS)
TRAIN = list(REG.TRAIN_COMBOS)
HELD = list(REG.HELD_OUT)
CTL_SEEN = list(REG.CONTROL_COMBOS)
CTL_UNSEEN = [r for r in R8 if r not in CTL_SEEN]
MARGINS = (0.5, 1.0)
MW = {0.5: 'Half', 1.0: 'One'}
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
M_SIM = len(R8)                       # Bonferroni family size of the simultaneous intervals
PAR_FAM = {'All': R8, 'Train': TRAIN, 'Held': HELD}
STREAM_SEED, FRESH_SEED, INST_SEED = 11, 11, 7
PROD = (('50x25', 'Fifty'), ('80x25', 'Eighty'))
PROD_N30 = {'50x25': '50x25n30', '80x25': '80x25n30'}
SOLVER_ROWS = REG.env() != 'exact'           # the 1800 s solver rows of E and E'
SOLVER_DIR = 'or_solution/FAMILY/{size}_fairwarm1800s'
SOLVER_SKIP = ('skipped in the exact environment: the solver comparison at the production '
               'batches comes from scripts/p94_timed_inference.py (the timed campaign); '
               'or_solution/FAMILY/{50x25,80x25}_fairwarm1800s are not read and no macro of '
               'them is written')


def empty_comparisons(solver_rows, n30):
    """Production comparisons that would hold no instance: a regime without a 1800 s
    solver record (with the solver rows), a rule file of a production set without a row.
    A rule file that is absent is left to the readers below (MissingResult)."""
    bad = []
    for size, _ in PROD:
        if solver_rows:
            d = SOLVER_DIR.format(size=size)
            empty = [r for r in R8 if not glob.glob(f'{d}/{r}/*.json')]
            if empty:
                bad.append(f'{size}: no 1800 s solver record in {d}/<regime> for '
                           f'{", ".join(empty)}')
        for folder in (size, PROD_N30[size]) if n30 else (size,):
            for r in R8:
                for q in RULES:
                    p = f'{REG.rule_dir(folder)}/Result_{q}_{r}.npy'
                    if os.path.exists(p) and len(REG.read_path(p)) == 0:
                        bad.append(f'{p} holds no instance')
    return bad


if __name__ == '__main__':
    OUT_MAC = REG.out('paper/macros_family.tex')
    OUT_JSON = REG.out('results/family_inference.json')
    OUT_MD = REG.out('results/family_inference.md')
    _EMPTY = empty_comparisons(SOLVER_ROWS, ARGS.n30)
    if _EMPTY:                                 # refuse before any output is removed
        sys.exit('REFUSING: a production comparison would hold no instance; no output was '
                 'removed or written:\n  ' + '\n  '.join(_EMPTY))
    for _p in (OUT_MAC, OUT_JSON, OUT_MD):     # a refusal anywhere below leaves no stale output
        if os.path.exists(_p):
            os.remove(_p)
    REG.refuse_on_missing(OUT_MAC, OUT_JSON, OUT_MD)


# ------------------------------------------------------------------ data
def rel(a, b):
    return 100.0 * (a - b) / b


def load(arm, combo, s, folder, r, n=None):
    return REG.load(arm, combo, s, folder, r, n)


def pool(folder, r, n=None):
    """Per-instance best of the four rules, evaluated under the active mask."""
    xs = [REG.read_path(f'{REG.rule_dir(folder)}/Result_{q}_{r}.npy', n) for q in RULES]
    if len({len(x) for x in xs}) != 1:
        sys.exit(f'REFUSING: rule files of {folder}/{r} differ in length')
    return np.min(np.stack(xs), axis=0)


def seed_pairs(fa, fb, folder, r, n=None, seeds=None):
    """[k, n]: d = 100 (a_s - b_s) / b_s, seed s against seed s, over the active seeds
    or the seeds given."""
    return np.stack([rel(fa(s, folder, r, n), fb(s, folder, r, n))
                     for s in (seeds or SEEDS)])


def released(s, folder, r, n=None):
    return load('released', None, s, folder, r, n)


def control(s, folder, r, n=None):
    return load('control', None, s, folder, r, n)


def specialist(s, folder, r, n=None):
    return load('specialist', r, s, folder, r, n)


def verify_bases(folder_data, n_used):
    """Instance i of every regime is the same base instance (the cluster unit)."""
    names, bases = {}, {}
    for r in R8:
        ps = sorted(glob.glob(f'data/FAMILY/{folder_data}/desc/{r}/*.desc.json'))
        names[r] = [os.path.basename(p) for p in ps]
        bases[r] = [json.load(open(p))['base_instance'] for p in ps]
    for r in R8:
        if names[r] != names['N'] or bases[r] != bases['N']:
            sys.exit(f'REFUSING: data/FAMILY/{folder_data}/{r} does not share file names '
                     f'and base instances with N; the clusters are not defined')
    if len(set(bases['N'])) != len(bases['N']) or len(bases['N']) < n_used:
        sys.exit(f'REFUSING: {folder_data} base instances are not distinct or too few')
    return len(names['N'])


# ------------------------------------------------------------------ components
def component(d, draws, m_sim=None):
    """Interval package of one comparison d [k, n] from its hierarchical draws; the
    simultaneous intervals are Bonferroni over m_sim comparisons (default M_SIM, the eight
    combinations)."""
    m = M_SIM if m_sim is None else m_sim
    lo, hi = SF.percentile_ci(draws, ALPHA)
    slo, shi = SF.percentile_ci(draws, ALPHA / m)
    t, ts = SF.seed_t(d, ALPHA), SF.seed_t(d, ALPHA / m)
    out = dict(mean=float(d.mean()), lo=lo, hi=hi, sim_lo=slo, sim_hi=shi,
               t_lo=t['lo'], t_hi=t['hi'], t_sim_lo=ts['lo'], t_sim_hi=ts['hi'],
               seed_means=t['seed_means'], seed_sd=t['sd'], n=int(d.shape[1]),
               direction=SF.direction(lo, hi))
    for mg in MARGINS:
        out[f'v{mg}'] = SF.verdict(lo, hi, mg)
        out[f'v{mg}_sim'] = SF.verdict(slo, shi, mg)
        out[f'v{mg}_t'] = SF.verdict(t['lo'], t['hi'], mg)
    return out


def fresh(d):
    return SF.hier_draws_stream(d, B, np.random.default_rng(FRESH_SEED))


def iut_block(comp, families):
    res = {}
    for fam, regs in families.items():
        for mg in MARGINS:
            for claim in ('equivalent', 'non-inferior'):
                holds, failing = SF.iut({r: comp[r][f'v{mg}'] for r in regs}, claim)
                res[f'{fam}|{mg}|{claim}'] = dict(holds=holds, failing=failing,
                                                  components=list(regs), margin=mg)
    return res


def pooled_block(D, families):
    """Pooled over the combinations of each family: hierarchical (combinations of a
    base instance together) and seed-level t."""
    res = {}
    for fam, regs in families.items():
        d3 = np.stack([D[r] for r in regs], axis=2)            # [k, n, c]
        hb = SF.hier_boot(d3, B=B, seed=FRESH_SEED, alpha=ALPHA)
        t = SF.seed_t(d3, ALPHA)
        res[fam] = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'],
                        t_hi=t['hi'], seed_means=t['seed_means'], c=len(regs),
                        direction=SF.direction(hb['lo'], hb['hi']),
                        **{f'v{mg}': SF.verdict(hb['lo'], hb['hi'], mg) for mg in MARGINS})
    return res


def superiority_block(D):
    """One-sided (policy faster) tests per combination, Holm over the family; then
    the one-sided tests of the opposite direction (policy slower) on the same
    per-seed means and the same draws, with their own Holm family, and the combined
    verdict (superiority_verdict)."""
    rows, behind = {}, {}
    for r, d in D.items():
        draws = fresh(d)
        lo, hi = SF.percentile_ci(draws, ALPHA)
        p_t = SF.seed_t_p(d, 'less')
        rows[r] = dict(mean=float(d.mean()), lo=lo, hi=hi,
                       p_t=1.0 if np.isnan(p_t) else p_t,
                       p_boot=SF.boot_p_one_sided(draws, 'less'),
                       seed_means=SF.seed_t(d)['seed_means'], n=int(d.shape[1]))
        p_g = SF.seed_t_p(d, 'greater')
        behind[r] = dict(p_t=1.0 if np.isnan(p_g) else p_g,
                         p_boot=SF.boot_p_one_sided(draws, 'greater'))
    keys = list(rows)
    for col in ('p_t', 'p_boot'):
        for r, a in zip(keys, SF.holm([rows[r][col] for r in keys])):
            rows[r][col + '_holm'] = a
            rows[r][col + '_holm_ahead'] = bool(a < ALPHA)
    for col in ('p_t', 'p_boot'):
        for r, a in zip(keys, SF.holm([behind[r][col] for r in keys])):
            rows[r][col + '_greater'] = behind[r][col]
            rows[r][col + '_greater_holm'] = a
    for r in keys:
        x = rows[r]
        x['verdict'] = superiority_verdict(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                                           x['p_t_greater_holm'], x['p_boot_greater_holm'], ALPHA)
    return rows


def configure(boot=None, alpha=None):
    """Replicates B and level ALPHA of the functions above, for a caller that imports this
    module (scripts/p96_two_environments.py); the script sets them from its arguments."""
    global B, ALPHA
    if boot is not None:
        B = int(boot)
    if alpha is not None:
        ALPHA = float(alpha)


def parity_components(folder='10x25', seeds=None):
    """Components of section A: the released policy against the specialist of each
    combination on folder, d [k, n] seed s against seed s, the hierarchical draws of every
    combination from ONE default_rng(STREAM_SEED) stream in p27's order (N, L, S, W, LS, LW,
    SW, LSW). Returns ({R: d}, {R: component})."""
    rng = np.random.default_rng(STREAM_SEED)
    D, C = {}, {}
    for r in TRAIN + HELD:                                       # p27's stream order
        d = seed_pairs(released, specialist, folder, r, seeds=seeds)
        D[r] = d
        C[r] = component(d, SF.hier_draws_stream(d, B, rng))
    return D, C


# ================================================================== F, G, H: definitions
# Sections F, G and H write their macros after every existing macro. Each is computed when
# every input array of the active environment and seeds exists with as many rows as its
# folder has test instances; otherwise the JSON (key sections) and the Markdown list it as
# skipped with the reason, and --require-sections refuses.
SECTIONS = ('F', 'G', 'H')
SECTION_TITLES = {'F': 'rule-change pairs, 10x25',
                  'G': 'specialist of the full combination LSW, 10x25',
                  'H': 'larger shops: 20x25, 30x25, 50x15, 80x15'}
CHANGE_FOLDER = '10x25'           # every specialist on its own and each one-more-rule combination
FULL = 'LSW'                      # the full combination
FULL_FOLDER = '10x25_blindref'    # the LSW specialist on all eight (exact: the 10x25 campaign)
SPEC_SIZES = ('20x25', '30x25')   # the specialists are evaluated on their own combination
POOL_SIZES = ('50x15', '80x15')   # the released policy and the rules only
SHOP_WORD = {'20x25': 'Twenty', '30x25': 'Thirty', '50x15': 'FiftyFifteen',
             '80x15': 'EightyFifteen'}
SUP_KEYS = ('p_t', 'p_boot', 'p_t_holm', 'p_boot_holm', 'p_t_greater', 'p_boot_greater',
            'p_t_greater_holm', 'p_boot_greater_holm', 'verdict')


def rules_of(c):
    """The timing rules of a combination (N has none)."""
    return frozenset() if c == 'N' else frozenset(c)


def rule_change_pairs():
    """(A, B) with B = A plus one rule, A in the order of the combinations: twelve pairs."""
    return [(a, b) for a in R8 for b in R8
            if rules_of(a) < rules_of(b) and len(rules_of(b)) == len(rules_of(a)) + 1]


PAIRS = rule_change_pairs()
PAIR_NAME = {p: f'{p[0]}To{p[1]}' for p in PAIRS}


def descriptor_count(data_folder):
    """Test instances of a folder of data/FAMILY: the descriptors of combination N."""
    return len(glob.glob(f'data/FAMILY/{data_folder}/desc/N/*.desc.json'))


def section_inputs(sec, seeds=None):
    """{data folder: [array paths]} that section sec reads in the active environment."""
    seeds = seeds or SEEDS
    out = {}

    def add(data, path):
        if path not in out.setdefault(data, []):
            out[data].append(path)
    if sec == 'F':
        for a, b in PAIRS:
            for s in seeds:
                for arm, c in (('released', None), ('specialist', a), ('specialist', b)):
                    add('10x25', REG.result_path(arm, c, s, CHANGE_FOLDER, b))
    elif sec == 'G':
        for r in R8:
            for s in seeds:
                for arm, c in (('specialist', FULL), ('released', None), ('specialist', r)):
                    add('10x25', REG.result_path(arm, c, s, FULL_FOLDER, r))
    elif sec == 'H':
        for size in SPEC_SIZES + POOL_SIZES:
            for r in R8:
                for s in seeds:
                    add(size, REG.result_path('released', None, s, size, r))
                    if size in SPEC_SIZES:
                        add(size, REG.result_path('specialist', r, s, size, r))
                for q in RULES:
                    add(size, f'{REG.rule_dir(size)}/Result_{q}_{r}.npy')
    else:
        raise ValueError(f'unknown section {sec!r}')
    return out


def section_state(sec, seeds=None):
    """Whether section sec can be computed: every input array exists and has as many rows as
    its folder has test instances. dict(ready, reason, arrays, missing, wrong_length, ...)."""
    miss, bad, why, n_arr = [], [], [], 0
    for data, paths in section_inputs(sec, seeds).items():
        n = descriptor_count(data)
        if n == 0:
            why.append(f'data/FAMILY/{data}/desc/N holds no descriptor')
        n_arr += len(paths)
        for p in paths:
            if not os.path.exists(p):
                miss.append(p)
                continue
            rows = len(REG.read_path(p))
            if n and rows != n:
                bad.append(f'{p}: {rows} rows, the test set has {n}')
    if miss:
        why.append(f'{len(miss)} of {n_arr} input arrays missing (first {miss[0]})')
    if bad:
        why.append(f'{len(bad)} input arrays with another length than the test set '
                   f'(first {bad[0]})')
    return dict(section=sec, title=SECTION_TITLES[sec], ready=not why,
                reason='; '.join(why) if why else None, arrays=n_arr, missing=miss,
                wrong_length=bad, environment=REG.env(), seeds=list(seeds or SEEDS))


def required_sections(text):
    """The sections --require-sections names: 'none', 'all' or a comma list of F, G, H."""
    t = str(text).replace(' ', '')
    if t in ('', 'none'):
        return []
    if t == 'all':
        return list(SECTIONS)
    req = t.split(',')
    bad = [x for x in req if x not in SECTIONS]
    if bad:
        sys.exit(f'REFUSING: --require-sections {text!r}: unknown section(s) {bad} (none, '
                 f'all, or a comma list of {", ".join(SECTIONS)})')
    return [x for x in SECTIONS if x in req]


def specialist_of(a):
    """Reader of the specialist trained on combination a, evaluated on any combination r."""
    return lambda s, folder, r, n=None: load('specialist', a, s, folder, r, n)


def pool_reader(s, folder, r, n=None):
    """The per-instance best of the four rules as a reader of seed s (the same for every s)."""
    return pool(folder, r, n)


def pair_block(D, m_sim):
    """A family of comparisons {name: d [k, n]}: per comparison the component (hierarchical
    interval from fresh draws, margin verdicts, direction) and the combined verdict of
    superiority_block (Holm over the family, each direction its own family); the pooled row
    over the family, a drawn base instance bringing every comparison; verdict counts."""
    rows = {p: component(d, fresh(d), m_sim=m_sim) for p, d in D.items()}
    sup = superiority_block(D)
    for p in D:
        rows[p].update({k: sup[p][k] for k in SUP_KEYS})
    counts = {w: sum(rows[p]['verdict'] == w for p in D) for w in VERDICT_WORDS}
    return dict(rows=rows, pooled=pooled_block(D, {'All': list(D)})['All'], counts=counts)


def section_F(seeds=None):
    """Rule-change pairs (A, B = A plus one rule) on B, seed s against seed s, 10x25:
    Pol   the released policy against the specialist of A, the specialist still in service
          after the rule was added;
    Spec  the specialist of B against the specialist of A, what the change costs a catalogue
          of specialists that is not retrained.
    d = 100 (first - second) / second; per pair the hierarchical interval (fresh
    default_rng(11)), the verdicts at both margins and the direction, and the combined
    verdict with Holm over the twelve pairs in both directions; the pooled row over the
    twelve pairs keeps the pairs of a base instance together."""
    verify_bases('10x25', descriptor_count('10x25'))
    D = {'Pol': {}, 'Spec': {}}
    for a, b in PAIRS:
        D['Pol'][PAIR_NAME[(a, b)]] = seed_pairs(released, specialist_of(a), CHANGE_FOLDER, b,
                                                 seeds=seeds)
        D['Spec'][PAIR_NAME[(a, b)]] = seed_pairs(specialist_of(b), specialist_of(a),
                                                  CHANGE_FOLDER, b, seeds=seeds)
    return {key: dict(pair_block(Dk, len(PAIRS)),
                      pairs={PAIR_NAME[p]: list(p) for p in PAIRS})
            for key, Dk in D.items()}


def section_G(seeds=None):
    """The specialist of the full combination LSW on every combination it is evaluated on
    (all eight, folder FULL_FOLDER), seed s against seed s:
    Pol   the LSW specialist against the released policy, all eight combinations;
    Spec  the LSW specialist against the specialist of the combination, the seven others.
    d = 100 (LSW specialist - other) / other; hierarchical interval (fresh default_rng(11)),
    verdicts at both margins and direction; the pooled rows keep the combinations of a base
    instance together."""
    verify_bases('10x25', descriptor_count('10x25'))
    full = specialist_of(FULL)
    others = [r for r in R8 if r != FULL]
    D = {'Pol': {r: seed_pairs(full, released, FULL_FOLDER, r, seeds=seeds) for r in R8},
         'Spec': {r: seed_pairs(full, specialist, FULL_FOLDER, r, seeds=seeds) for r in others}}
    fams = {'Pol': {'All': R8, 'Other': others}, 'Spec': {'Other': others}}
    return {key: dict(rows={r: component(d, fresh(d), m_sim=len(Dk)) for r, d in Dk.items()},
                      pooled=pooled_block(Dk, fams[key]), combinations=list(Dk))
            for key, Dk in D.items()}


def selected_rows(cells, combos=None):
    """One training run against a rule quantity, cells [n, c] of d per instance and
    combination: per combination and pooled (a drawn base instance brings its combinations),
    the instance-level cluster bootstrap with a fresh default_rng(INST_SEED) per quantity; the
    one-sided bootstrap p for H1: d < 0 and for H1: d > 0, each kind with its own Holm family
    over the combinations; the verdict is superiority_verdict with the bootstrap p in place of
    both p-values, since one run has no seed-level test."""
    combos = list(combos or R8)
    rows = {}
    for j, r in enumerate(combos):
        ib = SF.instance_boot(cells[:, j], B=B, seed=INST_SEED, alpha=ALPHA, return_draws=True)
        rows[r] = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], n=ib['n'],
                       direction=SF.direction(ib['lo'], ib['hi']),
                       p_boot=SF.boot_p_one_sided(ib['draws'], 'less'),
                       p_boot_greater=SF.boot_p_one_sided(ib['draws'], 'greater'))
    for col in ('p_boot', 'p_boot_greater'):
        for r, a in zip(combos, SF.holm([rows[r][col] for r in combos])):
            rows[r][col + '_holm'] = a
    for r in combos:
        x = rows[r]
        x['verdict'] = superiority_verdict(x['lo'], x['hi'], x['p_boot_holm'], x['p_boot_holm'],
                                           x['p_boot_greater_holm'], x['p_boot_greater_holm'],
                                           ALPHA)
    ib = SF.instance_boot(cells, B=B, seed=INST_SEED, alpha=ALPHA)
    pooled = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], n=ib['n'], c=ib['c'],
                  direction=SF.direction(ib['lo'], ib['hi']))
    counts = {w: sum(rows[r]['verdict'] == w for r in combos) for w in VERDICT_WORDS}
    return dict(rows=rows, pooled=pooled, counts=counts)


def section_H(sel, seeds=None):
    """Larger shops, d = 100 (released policy - other) / other, seed s against seed s:
    20x25, 30x25  against the specialist of each combination (verdicts at both margins,
                  direction, the IUT and pooled rows of the families of A) and against the
                  per-instance best of the four rules (combined verdict, Holm over the eight
                  combinations of the size, pooled row);
    50x15, 80x15  against the per-instance best of the four rules, every run (as above) and
                  the selected run sel (selected_rows)."""
    out = {}
    for size in SPEC_SIZES + POOL_SIZES:
        n = descriptor_count(size)
        verify_bases(size, n)
        D_rule = {r: seed_pairs(released, pool_reader, size, r, seeds=seeds) for r in R8}
        rule = superiority_block(D_rule)
        x = dict(n=n, rule_pool=rule, rule_pool_pooled=pooled_block(D_rule, {'All': R8})['All'],
                 rule_pool_counts={w: sum(rule[r]['verdict'] == w for r in R8)
                                   for w in VERDICT_WORDS})
        if size in SPEC_SIZES:
            D_par = {r: seed_pairs(released, specialist, size, r, seeds=seeds) for r in R8}
            comps = {r: component(d, fresh(d)) for r, d in D_par.items()}
            x.update(parity=comps, parity_iut=iut_block(comps, PAR_FAM),
                     parity_pooled=pooled_block(D_par, PAR_FAM))
        else:
            cells = np.stack([rel(load('released', None, sel, size, r), pool(size, r))
                              for r in R8], axis=1)
            x['selected'] = dict(selected_rows(cells), seed=sel)
        out[size] = x
    return out


if __name__ == '__main__':
    # sections F, G, H: the inputs are checked before any computation
    REQUIRED = required_sections(ARGS.require_sections)
    STATE = {sec: section_state(sec) for sec in SECTIONS}
    _SKIPPED = [f'{sec} ({STATE[sec]["title"]}): {STATE[sec]["reason"]}'
                for sec in REQUIRED if not STATE[sec]['ready']]
    if _SKIPPED:
        sys.exit(f'REFUSING: --require-sections {ARGS.require_sections} and these sections would '
                 f'be skipped (no output written):\n  ' + '\n  '.join(_SKIPPED))


# ================================================================== A. parity
if __name__ == '__main__':
    n10 = verify_bases('10x25', 100)
    D_PAR, PAR = parity_components('10x25')
    PAR_IUT = iut_block(PAR, PAR_FAM)
    PAR_POOL = pooled_block(D_PAR, PAR_FAM)
    PAR_POOL_INST = {}
    for fam, regs in PAR_FAM.items():
        cells = np.stack([rel(REG.pooled('released', None, '10x25', r),
                              REG.pooled('specialist', r, '10x25', r)) for r in regs], axis=1)
        PAR_POOL_INST[fam] = SF.instance_boot(cells, B=B, seed=INST_SEED, alpha=ALPHA)


# ---- reproduction of the published seed-level table (p27, macros_seed.tex)
def reproduce_p27():
    path = REG.out('paper/macros_seed.tex')
    if not os.path.exists(path):
        sys.exit(f'REFUSING: {path} is missing; run scripts/p27_seed_inference.py first '
                 f'(this script reproduces its seed-level intervals)')
    mac = dict(re.findall(r'\\newcommand\{\\(Sd\w+?)\}\{([-+0-9.]+)\}', open(path).read()))
    bad, n = [], 0
    for r in TRAIN + HELD:
        want = {f'Sd{r}Lo': PAR[r]['lo'], f'Sd{r}Hi': PAR[r]['hi']}
        want.update({f'Sd{r}{w}': m for w, m in zip(WORDS, PAR[r]['seed_means'])})
        for name, v in want.items():
            n += 1
            if mac.get(name) != f'{v:+.2f}':
                bad.append(f'\\{name}: {path} {mac.get(name)!r}, here {v:+.2f}')
    if bad:
        sys.exit('REFUSING: the seed-level intervals do not reproduce p27:\n  ' + '\n  '.join(bad))
    return dict(file=path, macros_compared=n, identical=True)


if __name__ == '__main__':
    REPRO = reproduce_p27()


def published_instance_intervals():
    """p11's instance-level intervals as the paper prints them (macros_eq.tex), read
    and not recomputed, so that the verdict rule above is applied to the published
    numbers themselves (W: [+0.18, +0.89])."""
    path = REG.out('paper/macros_eq.tex')
    if not os.path.exists(path):
        sys.exit(f'REFUSING: {path} is missing; run scripts/p11_equivalence.py first')
    mac = dict(re.findall(r'\\newcommand\{\\(Eq\w+?)\}\{([-+0-9.]+)\}', open(path).read()))
    out = {}
    for r in R8:
        key = 'Eone' + r if r in TRAIN else ('Sw' if r == 'SW' else 'Lsw')
        if f'Eq{key}Lo' not in mac or f'Eq{key}Hi' not in mac:
            sys.exit(f'REFUSING: {path} has no interval for {r}')
        lo, hi = float(mac[f'Eq{key}Lo']), float(mac[f'Eq{key}Hi'])
        out[r] = dict(lo=lo, hi=hi, direction=SF.direction(lo, hi), source=path,
                      **{f'v{mg}': SF.verdict(lo, hi, mg) for mg in MARGINS})
    return out


if __name__ == '__main__':
    PAR_INST = published_instance_intervals()
    PAR_INST_IUT = iut_block(PAR_INST, PAR_FAM)

# ================================================================== B. rule pool
if __name__ == '__main__':
    D_RULE = {r: np.stack([rel(released(s, '10x25', r), pool('10x25', r)) for s in SEEDS])
              for r in R8}
    RULE = superiority_block(D_RULE)

    # ================================================================== C. control
    verify_bases('10x25', 100)
    CTL = {'Rel': {}, 'Spec': {}}
    D_CTL = {'Rel': {}, 'Spec': {}}
    for r in R8:
        for key, other in (('Rel', released), ('Spec', specialist)):
            d = seed_pairs(control, other, '10x25_blindref', r)
            D_CTL[key][r] = d
            CTL[key][r] = component(d, fresh(d))
    # the unseen combinations also split as in p89_public_family (Pair: unseen pairs of
    # trained rules; Held: the two combinations no run was trained on)
    CTL_FAM = {'Unseen': CTL_UNSEEN, 'Pair': [r for r in CTL_UNSEEN if r not in HELD],
               'Held': [r for r in CTL_UNSEEN if r in HELD], 'All': R8}
    CTL_IUT = {key: iut_block(CTL[key], CTL_FAM) for key in CTL}
    CTL_POOL = {key: pooled_block(D_CTL[key], CTL_FAM) for key in CTL}
    D_CTL_RULE = {r: np.stack([rel(control(s, '10x25_blindref', r), pool('10x25_blindref', r))
                               for s in SEEDS]) for r in R8}
    CTL_RULE = superiority_block(D_CTL_RULE)

# ================================================================== E. production scale
if __name__ == '__main__':
    SEL, SEL_REC = REG.select_seed('released')
    if SOLVER_ROWS:
        import p82_best3 as p82  # noqa: E402  (the 1800 s credit rule, shared with p60/p82/p84)


def lead_block(cells_of):
    """cells_of(x) -> [n, 8] leads (+ = policy ahead) of a policy makespan source."""
    out = {}
    for label, src in [(f's{s}', (lambda size, r, n, s=s: released(s, size, r, n)))
                       for s in SEEDS] + \
                      [('mean', lambda size, r, n: REG.pooled('released', None, size, r, n))]:
        cells = cells_of(src)
        ci = SF.instance_boot(cells, B=B, seed=INST_SEED, alpha=ALPHA)
        ahead = behind = 0
        for j in range(cells.shape[1]):
            c = SF.instance_boot(cells[:, j], B=B, seed=INST_SEED, alpha=ALPHA)
            ahead += c['lo'] > 0
            behind += c['hi'] < 0
        out[label] = dict(mean=ci['mean'], lo=ci['lo'], hi=ci['hi'], n=ci['n'],
                          regimes_ahead=int(ahead), regimes_behind=int(behind),
                          direction={'ahead': 'behind', 'behind': 'ahead',
                                     'neither': 'neither'}[SF.direction(ci['lo'], ci['hi'])])
    seeds_only = [out[f's{s}'] for s in SEEDS]
    out['seed_summary'] = dict(
        min=min(x['mean'] for x in seeds_only), max=max(x['mean'] for x in seeds_only),
        clear=sum(x['lo'] > 0 for x in seeds_only),
        tie=sum(x['lo'] <= 0 <= x['hi'] for x in seeds_only),
        behind=sum(x['hi'] < 0 for x in seeds_only))
    out['selected'] = dict(out[f's{SEL}'], seed=SEL)
    return out


def diff_block(cells_of):
    """The rows of lead_block under the paper's convention: cells_of(x) -> [n, 8]
    differences 100 (policy - rival) / rival of a policy makespan source (negative =
    policy shorter), recomputed per instance from the arrays. Same sources, the same
    instance-level cluster bootstrap (B, fresh default_rng(INST_SEED)), pooled and per
    regime; a regime or a seed is ahead when its interval lies below zero."""
    out = {}
    for label, src in [(f's{s}', (lambda size, r, n, s=s: released(s, size, r, n)))
                       for s in SEEDS] + \
                      [('mean', lambda size, r, n: REG.pooled('released', None, size, r, n))]:
        cells = cells_of(src)
        ci = SF.instance_boot(cells, B=B, seed=INST_SEED, alpha=ALPHA)
        ahead = behind = 0
        for j in range(cells.shape[1]):
            c = SF.instance_boot(cells[:, j], B=B, seed=INST_SEED, alpha=ALPHA)
            ahead += c['hi'] < 0
            behind += c['lo'] > 0
        out[label] = dict(mean=ci['mean'], lo=ci['lo'], hi=ci['hi'], n=ci['n'],
                          regimes_ahead=int(ahead), regimes_behind=int(behind),
                          direction=SF.direction(ci['lo'], ci['hi']))
    seeds_only = [out[f's{s}'] for s in SEEDS]
    out['seed_summary'] = dict(
        min=min(x['mean'] for x in seeds_only), max=max(x['mean'] for x in seeds_only),
        clear=sum(x['hi'] < 0 for x in seeds_only),
        tie=sum(x['lo'] <= 0 <= x['hi'] for x in seeds_only),
        behind=sum(x['lo'] > 0 for x in seeds_only))
    out['selected'] = dict(out[f's{SEL}'], seed=SEL)
    return out


def production(solver_rows, with_n30):
    """Rows of E (lead) and E' (policy minus rival): two dicts {(folder, rival): block}.
    The 1800 s solver rows only with solver_rows (outside the exact environment); the
    rule-pool rows always, and with with_n30 also on the 30-instance sets."""
    prodr, prodd = {}, {}
    for size, word in PROD:
        if solver_rows:
            cp = {r: p82.cp_1800(size, r) for r in R8}
            ncp = {len(v) for v in cp.values()}
            if len(ncp) != 1:
                sys.exit(f'REFUSING: {size} 1800 s solver cells differ in count across regimes: {ncp}')
            ncp = ncp.pop()
            verify_bases(size, ncp)
            for r in R8:                       # solver records in instance order (p84's check)
                fs = sorted(glob.glob(f'{SOLVER_DIR.format(size=size)}/{r}/*.json'))
                inst = [json.load(open(f))['instance'] for f in fs]
                want = sorted(os.path.basename(p)[:-len('.desc.json')] for p in
                              glob.glob(f'data/FAMILY/{size}/desc/{r}/*.desc.json'))[:len(inst)]
                if inst != want:
                    sys.exit(f'REFUSING: {size}/{r} solver records are not in instance order')
            prodr[(size, 'cpsat1800')] = lead_block(
                lambda src, size=size, n=ncp: np.stack(
                    [rel(cp[r], src(size, r, n)) for r in R8], axis=1))
            prodd[(size, 'cpsat1800')] = diff_block(
                lambda src, size=size, n=ncp: np.stack(
                    [rel(src(size, r, n), cp[r]) for r in R8], axis=1))
        nst = len(pool(size, 'N'))
        if not solver_rows:                # the clusters of every stored instance
            verify_bases(size, nst)
        prodr[(size, 'pool')] = lead_block(
            lambda src, size=size, n=nst: np.stack(
                [rel(pool(size, r, n), src(size, r, n)) for r in R8], axis=1))
        prodd[(size, 'pool')] = diff_block(
            lambda src, size=size, n=nst: np.stack(
                [rel(src(size, r, n), pool(size, r, n)) for r in R8], axis=1))
        if with_n30:
            f30 = PROD_N30[size]
            n30 = verify_bases(f30, 30)
            prodr[(f30, 'pool')] = lead_block(
                lambda src, f30=f30, n=n30: np.stack(
                    [rel(pool(f30, r, n), src(f30, r, n)) for r in R8], axis=1))
            prodd[(f30, 'pool')] = diff_block(
                lambda src, f30=f30, n=n30: np.stack(
                    [rel(src(f30, r, n), pool(f30, r, n)) for r in R8], axis=1))
    return prodr, prodd


if __name__ == '__main__':
    PRODR, PRODD = production(SOLVER_ROWS, ARGS.n30)
    # sections F, G, H, each when its inputs exist (STATE); H uses the selected run of E
    SEC = {'F': section_F() if STATE['F']['ready'] else None,
           'G': section_G() if STATE['G']['ready'] else None,
           'H': section_H(SEL) if STATE['H']['ready'] else None}


# ================================================================== output
def f2(x):
    """Signed two decimals of the unrounded value, so a printed bound keeps its sign: a
    negative value that rounds to zero prints -0.00, a positive one +0.00, an exact zero
    (also -0.0) +0.00. Verdicts and directions use the unrounded values, never this text."""
    x = float(x)
    return f'{0.0 if x == 0 else x:+.2f}'


def tex(v):
    return SF.TEX[v]


if __name__ == '__main__':
    L = [f'% Generated by scripts/p88_family_inference.py ({REG.describe()}) -- do not hand-edit',
         f'% B = {B}, alpha = {ALPHA}; parity components on one default_rng({STREAM_SEED}) stream '
         f'(p27 order), other hierarchical quantities fresh default_rng({FRESH_SEED}), '
         f'instance-level fresh default_rng({INST_SEED}).',
         '% Gaps in %: parity, control and rule rows first arm minus second over second '
         '(positive = first arm slower); production rows rival minus policy over policy '
         '(positive = policy ahead).',
         '% Appended after every other macro: names ending in Rel repeat the production rows '
         'under the paper\'s convention, policy minus rival over rival (negative = policy '
         'shorter), recomputed per instance with the same cluster bootstrap; names ending in '
         'Verdict or VerdictAheadN/VerdictBehindN give the combined verdict against the rule '
         'pool (ahead, behind, not distinguishable).',
         f'\\newcommand{{\\FamSeedN}}{{{K}}}  % training seeds per arm',
         f'\\newcommand{{\\FamBoot}}{{{B}}}  % bootstrap replicates',
         f'\\newcommand{{\\FamSelSeed}}{{{SEL}}}  % selected checkpoint: best validation score '
         f'({", ".join(f"s{s} {v[0]:.5f}" for s, v in SEL_REC.items())})',
         f'\\newcommand{{\\FamSelScore}}{{{SEL_REC[SEL][0]:.4f}}}  % its validation score']


def comp_macros(prefix, c, what):
    L.append(f'\\newcommand{{\\{prefix}}}{{{f2(c["mean"])}}}'
             f'\\newcommand{{\\{prefix}Lo}}{{{f2(c["lo"])}}}'
             f'\\newcommand{{\\{prefix}Hi}}{{{f2(c["hi"])}}}'
             f'\\newcommand{{\\{prefix}SimLo}}{{{f2(c["sim_lo"])}}}'
             f'\\newcommand{{\\{prefix}SimHi}}{{{f2(c["sim_hi"])}}}'
             f'\\newcommand{{\\{prefix}TLo}}{{{f2(c["t_lo"])}}}'
             f'\\newcommand{{\\{prefix}THi}}{{{f2(c["t_hi"])}}}'
             + ''.join(f'\\newcommand{{\\{prefix}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}'
                       for mg in MARGINS)
             + f'\\newcommand{{\\{prefix}Dir}}{{{tex(c["direction"])}}}'
             f'  % {what}: mean, hierarchical 95% CI, Bonferroni/{M_SIM} CI, seed-t CI, '
             f'verdicts at 0.5 and 1, direction')


def iut_macros(prefix, block, fams):
    for fam in fams:
        for mg in MARGINS:
            for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                x = block[f'{fam}|{mg}|{claim}']
                L.append(f'\\newcommand{{\\{prefix}{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                         f'  % IUT, {claim} on {",".join(x["components"])} at +-{mg}%: '
                         f'failing {",".join(x["failing"]) or "none"}')


def pool_macros(prefix, block):
    for fam, x in block.items():
        L.append(f'\\newcommand{{\\{prefix}{fam}}}{{{f2(x["mean"])}}}'
                 f'\\newcommand{{\\{prefix}{fam}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{prefix}{fam}Hi}}{{{f2(x["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{prefix}{fam}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'  % pooled over {x["c"]} combinations, base-instance clusters, '
                 f'hierarchical 95% CI and verdicts')


def rule_macros(prefix, block, what):
    for r, x in block.items():
        L.append(f'\\newcommand{{\\{prefix}{r}}}{{{f2(x["mean"])}}}'
                 f'\\newcommand{{\\{prefix}{r}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{prefix}{r}Hi}}{{{f2(x["hi"])}}}'
                 f'\\newcommand{{\\{prefix}{r}PHolm}}{{{x["p_t_holm"]:.2g}}}'
                 f'  % {what} on {r}: mean, hierarchical 95% CI, Holm-adjusted one-sided '
                 f'seed-t p (bootstrap Holm p {x["p_boot_holm"]:.2g})')
    L.append(f'\\newcommand{{\\{prefix}HolmN}}'
             f'{{{sum(x["p_t_holm_ahead"] for x in block.values())}}}'
             f'\\newcommand{{\\{prefix}HolmBootN}}'
             f'{{{sum(x["p_boot_holm_ahead"] for x in block.values())}}}'
             f'  % combinations (of {len(block)}) where the policy is ahead after Holm: '
             f'seed-t, hierarchical bootstrap')


if __name__ == '__main__':
    for r in R8:
        comp_macros(f'FamPar{r}', PAR[r], f'released policy vs specialist, {r}')
    iut_macros('FamParIut', PAR_IUT, PAR_FAM)
    for r in R8:
        x = PAR_INST[r]
        L.append(''.join(f'\\newcommand{{\\FamParInst{r}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                         for mg in MARGINS)
                 + f'\\newcommand{{\\FamParInst{r}Dir}}{{{tex(x["direction"])}}}'
                 f'  % {r}: verdicts at 0.5 and 1 and direction of the published instance-level '
                 f'interval [{x["lo"]:+.2f}, {x["hi"]:+.2f}] (macros_eq.tex, read)')
    iut_macros('FamParInstIut', PAR_INST_IUT, PAR_FAM)
    pool_macros('FamParPool', PAR_POOL)
    for fam, x in PAR_POOL_INST.items():
        L.append(f'\\newcommand{{\\FamParPoolInst{fam}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\FamParPoolInst{fam}Hi}}{{{f2(x["hi"])}}}'
                 f'  % seed-pooled gap (p11 estimator), base-instance cluster 95% CI, {fam}')
    rule_macros('FamRule', RULE, 'released policy vs rule pool')
    for key, what in (('Rel', 'released policy'), ('Spec', 'specialist')):
        for r in R8:
            comp_macros(f'FamCtl{key}{r}', CTL[key][r], f'single-rule control vs {what}, {r}')
        iut_macros(f'FamCtl{key}Iut', CTL_IUT[key], CTL_FAM)
        pool_macros(f'FamCtl{key}Pool', CTL_POOL[key])
    rule_macros('FamCtlRule', CTL_RULE, 'single-rule control vs rule pool')

    SETW = {'cpsat1800': 'Cp', 'pool': 'Pool'}
    if not SOLVER_ROWS:
        L.append(f'% Production scale, 1800 s solver rows: {SOLVER_SKIP}.')
    for (folder, rival), blk in PRODR.items():
        size = folder[:5]
        w = dict(PROD)[size] + ('Thirty' if folder.endswith('n30') else '') + SETW[rival]
        for s, word in zip(SEEDS, WORDS):
            x = blk[f's{s}']
            L.append(f'\\newcommand{{\\FamProd{w}{word}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\FamProd{w}{word}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\FamProd{w}{word}Hi}}{{{f2(x["hi"])}}}'
                     f'  % {folder}, seed {s} alone vs {rival}: lead %, cluster 95% CI '
                     f'(regimes ahead {x["regimes_ahead"]}, behind {x["regimes_behind"]})')
        for lab, key in (('Sel', 'selected'), ('Mean', 'mean')):
            x = blk[key]
            L.append(f'\\newcommand{{\\FamProd{w}{lab}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}Hi}}{{{f2(x["hi"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}AheadN}}{{{x["regimes_ahead"]}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}BehindN}}{{{x["regimes_behind"]}}}'
                     f'  % {folder}, {"selected checkpoint (s%d)" % SEL if key == "selected" else "seed-mean policy (p60 estimator)"}'
                     f' vs {rival}: lead %, cluster 95% CI, regimes ahead/behind of 8')
        ss = blk['seed_summary']
        L.append(f'\\newcommand{{\\FamProd{w}SeedMin}}{{{f2(ss["min"])}}}'
                 f'\\newcommand{{\\FamProd{w}SeedMax}}{{{f2(ss["max"])}}}'
                 f'\\newcommand{{\\FamProd{w}SeedClearN}}{{{ss["clear"]}}}'
                 f'\\newcommand{{\\FamProd{w}SeedTieN}}{{{ss["tie"]}}}'
                 f'\\newcommand{{\\FamProd{w}SeedBehindN}}{{{ss["behind"]}}}'
                 f'\\newcommand{{\\FamProd{w}N}}{{{blk["selected"]["n"]}}}'
                 f'  % {folder} vs {rival}: smallest and largest single-seed lead, seeds '
                 f'clear / tied / behind, instances per regime')

    # ---- appended: production rows under the paper's convention (suffix Rel) and the
    # combined verdict against the rule pool; every macro above is unchanged.
    for (folder, rival), blk in PRODD.items():
        size = folder[:5]
        w = dict(PROD)[size] + ('Thirty' if folder.endswith('n30') else '') + SETW[rival]
        for s, word in zip(SEEDS, WORDS):
            x = blk[f's{s}']
            L.append(f'\\newcommand{{\\FamProd{w}{word}Rel}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\FamProd{w}{word}LoRel}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\FamProd{w}{word}HiRel}}{{{f2(x["hi"])}}}'
                     f'  % {folder}, seed {s} alone vs {rival}: policy minus rival over rival %, '
                     f'cluster 95% CI (regimes ahead {x["regimes_ahead"]}, behind {x["regimes_behind"]})')
        for lab, key in (('Sel', 'selected'), ('Mean', 'mean')):
            x = blk[key]
            L.append(f'\\newcommand{{\\FamProd{w}{lab}Rel}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}LoRel}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}HiRel}}{{{f2(x["hi"])}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}AheadNRel}}{{{x["regimes_ahead"]}}}'
                     f'\\newcommand{{\\FamProd{w}{lab}BehindNRel}}{{{x["regimes_behind"]}}}'
                     f'  % {folder}, {"selected checkpoint (s%d)" % SEL if key == "selected" else "seed-mean policy (p60 estimator)"}'
                     f' vs {rival}: policy minus rival over rival %, cluster 95% CI, regimes (of 8) '
                     f'with the interval below / above zero')
        ss = blk['seed_summary']
        L.append(f'\\newcommand{{\\FamProd{w}SeedMinRel}}{{{f2(ss["min"])}}}'
                 f'\\newcommand{{\\FamProd{w}SeedMaxRel}}{{{f2(ss["max"])}}}'
                 f'\\newcommand{{\\FamProd{w}SeedClearNRel}}{{{ss["clear"]}}}'
                 f'\\newcommand{{\\FamProd{w}SeedTieNRel}}{{{ss["tie"]}}}'
                 f'\\newcommand{{\\FamProd{w}SeedBehindNRel}}{{{ss["behind"]}}}'
                 f'  % {folder} vs {rival}: smallest and largest single-seed difference (policy minus '
                 f'rival over rival), seeds with the interval below zero / containing zero / above zero')
    L.append(f'% Combined verdict against the rule pool: ahead = hierarchical {100 * (1 - ALPHA):g}% CI '
             f'below zero and both Holm-adjusted one-sided p-values (seed-level t, bootstrap; H1 first '
             f'arm faster) below {ALPHA:g}; behind = CI above zero and both Holm-adjusted one-sided '
             f'p-values for H1 first arm slower (own Holm family over the eight combinations) below '
             f'{ALPHA:g}; not distinguishable otherwise.')
    for prefix, blk, what in (('FamRule', RULE, 'released policy vs rule pool'),
                              ('FamCtlRule', CTL_RULE, 'single-rule control vs rule pool')):
        for r, x in blk.items():
            L.append(f'\\newcommand{{\\{prefix}{r}Verdict}}{{{x["verdict"]}}}'
                     f'  % {what} on {r}: CI [{x["lo"]:+.2f}, {x["hi"]:+.2f}]; Holm p faster '
                     f'{x["p_t_holm"]:.2g} (t), {x["p_boot_holm"]:.2g} (bootstrap); Holm p slower '
                     f'{x["p_t_greater_holm"]:.2g} (t), {x["p_boot_greater_holm"]:.2g} (bootstrap)')
        vs = [x['verdict'] for x in blk.values()]
        L.append(f'\\newcommand{{\\{prefix}VerdictAheadN}}{{{vs.count(VERDICT_WORDS[0])}}}'
                 f'\\newcommand{{\\{prefix}VerdictBehindN}}{{{vs.count(VERDICT_WORDS[1])}}}'
                 f'  % {what}: combinations (of {len(blk)}) with the combined verdict ahead, behind')


def dir_macros(prefix, block, what):
    """<prefix><family>Dir (ahead / behind / neither) of the pooled rows {family: row},
    from the unrounded interval ends (stats_family.direction, the rule of the
    per-combination Dir macros)."""
    return [f'\\newcommand{{\\{prefix}{fam}Dir}}{{{tex(SF.direction(x["lo"], x["hi"]))}}}'
            f'  % {what}, {fam}: direction of the interval [{x["lo"]:+.4f}, {x["hi"]:+.4f}]'
            for fam, x in block.items()]


# ---- macros of sections F, G, H (appended after every macro above)
def row_line(name, x, what, margins=True, verdict=False):
    """<name>, <name>Lo, <name>Hi and <name>Dir of one row (mean, interval, direction from the
    unrounded ends), with <name>VHalf and <name>VOne (margin verdicts) and <name>Verdict
    (combined verdict) when asked, on one line."""
    s = (f'\\newcommand{{\\{name}}}{{{f2(x["mean"])}}}'
         f'\\newcommand{{\\{name}Lo}}{{{f2(x["lo"])}}}'
         f'\\newcommand{{\\{name}Hi}}{{{f2(x["hi"])}}}'
         f'\\newcommand{{\\{name}Dir}}{{{tex(SF.direction(x["lo"], x["hi"]))}}}')
    if margins:
        s += ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}' for mg in MARGINS)
    if verdict:
        s += f'\\newcommand{{\\{name}Verdict}}{{{x["verdict"]}}}'
    return s + f'  % {what}'


def holm_text(x):
    """The Holm-adjusted one-sided p-values behind a combined verdict, for a comment."""
    if 'p_t_holm' in x:
        return (f'Holm p faster {x["p_t_holm"]:.2g} (t), {x["p_boot_holm"]:.2g} (bootstrap); '
                f'Holm p slower {x["p_t_greater_holm"]:.2g} (t), '
                f'{x["p_boot_greater_holm"]:.2g} (bootstrap)')
    return (f'Holm p faster {x["p_boot_holm"]:.2g}, Holm p slower {x["p_boot_greater_holm"]:.2g} '
            f'(instance-level bootstrap)')


def count_line(prefix, counts, what):
    return (f'\\newcommand{{\\{prefix}VerdictAheadN}}{{{counts[VERDICT_WORDS[0]]}}}'
            f'\\newcommand{{\\{prefix}VerdictBehindN}}{{{counts[VERDICT_WORDS[1]]}}}'
            f'\\newcommand{{\\{prefix}VerdictNdN}}{{{counts[VERDICT_WORDS[2]]}}}'
            f'  % {what}: combined verdict ahead, behind, not distinguishable')


def iut_lines(prefix, block, fams):
    out = []
    for fam in fams:
        for mg in MARGINS:
            for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                x = block[f'{fam}|{mg}|{claim}']
                out.append(f'\\newcommand{{\\{prefix}{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                           f'  % IUT, {claim} on {",".join(x["components"])} at +-{mg}%: '
                           f'failing {",".join(x["failing"]) or "none"}')
    return out


def section_F_macros(res):
    L2 = ['% F. Rule-change pairs (A, B = A plus one rule) on B, 10x25, seed s against seed s. '
          'FamChgPol: released policy vs the specialist of A; FamChgSpec: the specialist of B '
          'vs the specialist of A. d = 100 (first - second) / second, hierarchical 95% CI, '
          'verdicts at 0.5 and 1, direction, combined verdict with Holm over the twelve pairs '
          '(each direction its own family); Pool: the twelve pairs pooled, a base instance '
          'with all its pairs.',
          f'\\newcommand{{\\FamChgPairN}}{{{len(PAIRS)}}}  % rule-change pairs']
    for key, what in (('Pol', 'released policy vs the specialist of'),
                      ('Spec', 'specialist of the later combination vs the specialist of')):
        blk = res[key]
        for (a, b), name in PAIR_NAME.items():
            x = blk['rows'][name]
            L2.append(row_line(f'FamChg{key}{name}', x, f'{a} to {b}, on {b}: {what} {a}; '
                               f'{holm_text(x)}', verdict=True))
        L2.append(row_line(f'FamChg{key}Pool', blk['pooled'],
                           f'{what} the earlier combination, pooled over the twelve pairs'))
        L2.append(count_line(f'FamChg{key}', blk['counts'], f'{what} A, of twelve pairs'))
    return L2


def section_G_macros(res):
    L2 = ['% G. The specialist of the full combination LSW on every combination, 10x25 (the '
          'campaign that evaluates it on all eight), seed s against seed s. FamFullPol: LSW '
          'specialist vs released policy; FamFullSpec: LSW specialist vs the specialist of '
          'the combination (seven combinations). d = 100 (LSW specialist - other) / other, '
          'hierarchical 95% CI, verdicts at 0.5 and 1, direction; Pool<family>: pooled, a '
          'base instance with all its combinations.']
    for key, what in (('Pol', 'LSW specialist vs released policy'),
                      ('Spec', 'LSW specialist vs the specialist of the combination')):
        blk = res[key]
        for r in blk['combinations']:
            L2.append(row_line(f'FamFull{key}{r}', blk['rows'][r], f'{what}, {r}'))
        for fam, x in blk['pooled'].items():
            L2.append(row_line(f'FamFull{key}Pool{fam}', x,
                               f'{what}, pooled over {x["c"]} combinations'))
    return L2


def section_H_macros(res):
    L2 = ['% H. Larger shops, seed s against seed s, d = 100 (released policy - other) / other. '
          'Par: vs the specialist of the combination (hierarchical 95% CI, verdicts at 0.5 and '
          '1, direction; Iut and Pool over the families of A); Rule: vs the per-instance best '
          'of the four rules, every run (combined verdict, Holm over the eight combinations of '
          'the size; Pool: the eight pooled); Sel: the selected run vs the same rule quantity '
          '(instance-level cluster 95% CI, Holm over eight).']
    for size, x in res.items():
        w = 'FamShop' + SHOP_WORD[size]
        L2.append(f'\\newcommand{{\\{w}N}}{{{x["n"]}}}  % {size}: test instances per combination')
        if 'parity' in x:
            for r in R8:
                L2.append(row_line(f'{w}Par{r}', x['parity'][r],
                                   f'{size}, released policy vs specialist, {r}'))
            L2 += iut_lines(f'{w}ParIut', x['parity_iut'], PAR_FAM)
            for fam, y in x['parity_pooled'].items():
                L2.append(row_line(f'{w}ParPool{fam}', y, f'{size}, released policy vs '
                                   f'specialist, pooled over {y["c"]} combinations'))
        for r in R8:
            y = x['rule_pool'][r]
            L2.append(row_line(f'{w}Rule{r}', y, f'{size}, released policy vs rule pool, {r}; '
                               f'{holm_text(y)}', margins=False, verdict=True))
        L2.append(row_line(f'{w}RulePool', x['rule_pool_pooled'],
                           f'{size}, released policy vs rule pool, pooled over eight', margins=False))
        L2.append(count_line(f'{w}Rule', x['rule_pool_counts'],
                             f'{size}, released policy vs rule pool, of eight'))
        if 'selected' in x:
            sel = x['selected']
            for r in R8:
                y = sel['rows'][r]
                L2.append(row_line(f'{w}Sel{r}', y, f'{size}, selected run s{sel["seed"]} vs rule '
                                   f'pool, {r}; {holm_text(y)}', margins=False, verdict=True))
            L2.append(row_line(f'{w}SelPool', sel['pooled'], f'{size}, selected run '
                               f's{sel["seed"]} vs rule pool, pooled over eight', margins=False))
            L2.append(count_line(f'{w}Sel', sel['counts'],
                                 f'{size}, selected run vs rule pool, of eight'))
    return L2


SECTION_MACROS = {'F': section_F_macros, 'G': section_G_macros, 'H': section_H_macros}


def section_lines(state, results):
    """The macro lines of sections F, G and H, or one comment line per skipped section."""
    out = []
    for sec in SECTIONS:
        if results.get(sec) is None:
            out.append(f'% Section {sec} ({state[sec]["title"]}) skipped: {state[sec]["reason"]}')
        else:
            out += SECTION_MACROS[sec](results[sec])
    return out


if __name__ == '__main__':
    # ---- appended: direction of the pooled parity and control rows
    L.append('% Direction (ahead / behind / neither) of the pooled parity and control rows, from '
             'the unrounded interval ends: ahead = interval below zero, behind = above zero.')
    L += dir_macros('FamParPool', PAR_POOL, 'released policy vs specialist, pooled, hierarchical CI')
    for fam, x in PAR_POOL_INST.items():
        x['direction'] = SF.direction(x['lo'], x['hi'])
    L += dir_macros('FamParPoolInst', PAR_POOL_INST,
                    'released policy vs specialist, seed-pooled gap, base-instance cluster CI')
    for key, what in (('Rel', 'released policy'), ('Spec', 'specialist')):
        L += dir_macros(f'FamCtl{key}Pool', CTL_POOL[key],
                        f'single-rule control vs {what}, pooled, hierarchical CI')
    # ---- appended: sections F, G, H (after every macro above)
    L += section_lines(STATE, SEC)

    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(L))
    if len(names) != len(set(names)):
        sys.exit('REFUSING: duplicate macro names in the output')
    for f in glob.glob('paper/macros_*.tex') + ['paper/macros.tex']:
        if not os.path.exists(f) or os.path.basename(f) == os.path.basename(OUT_MAC):
            continue
        clash = set(names) & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(f).read()))
        if clash:
            sys.exit(f'REFUSING: macro name collision with {f}: {sorted(clash)[:5]}')
    emit(OUT_MAC, L)


def ci_text(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def section_markdown(state, results):
    """The Markdown of sections F, G and H, appended after every existing part."""
    md = ['', '## Sections F, G, H', '']
    for sec in SECTIONS:
        st = state[sec]
        md.append(f'- {sec}, {st["title"]}: ' + ('computed' if results.get(sec) is not None
                                                 else f'skipped: {st["reason"]}'))
    F, G, H = (results.get(sec) for sec in SECTIONS)
    if F is not None:
        md += ['', '## F. Rule-change pairs (10x25), on the later combination B', '',
               'd = 100 (first - second) / second, seed s against seed s; hierarchical 95% CI; '
               'combined verdict with Holm over the twelve pairs (each direction its own family).',
               '', '| A to B | policy vs specialist of A | 0.5% | 1% | verdict | specialist of B '
               'vs specialist of A | 0.5% | 1% | verdict |', '|---|---|---|---|---|---|---|---|---|']
        for (a, b), name in PAIR_NAME.items():
            p, q = F['Pol']['rows'][name], F['Spec']['rows'][name]
            md.append(f'| {a} to {b} | {ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | {p["verdict"]} | '
                      f'{ci_text(q)} | {q["v0.5"]} | {q["v1.0"]} | {q["verdict"]} |')
        p, q = F['Pol']['pooled'], F['Spec']['pooled']
        md.append(f'| pooled | {ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | {p["direction"]} | '
                  f'{ci_text(q)} | {q["v0.5"]} | {q["v1.0"]} | {q["direction"]} |')
        for key, what in (('Pol', 'policy vs specialist of A'),
                          ('Spec', 'specialist of B vs specialist of A')):
            c = F[key]['counts']
            md.append(f'\n{what}: ahead on {c[VERDICT_WORDS[0]]}, behind on {c[VERDICT_WORDS[1]]}, '
                      f'not distinguishable on {c[VERDICT_WORDS[2]]} of {len(PAIRS)} pairs.')
    if G is not None:
        md += ['', f'## G. The LSW specialist on every combination ({FULL_FOLDER})', '',
               'd = 100 (LSW specialist - other) / other, seed s against seed s; hierarchical 95% CI.',
               '', '| comb. | vs released policy | 0.5% | 1% | dir. | vs specialist of the '
               'combination | 0.5% | 1% | dir. |', '|---|---|---|---|---|---|---|---|---|']
        for r in R8:
            p = G['Pol']['rows'][r]
            q = G['Spec']['rows'].get(r)
            md.append(f'| {r} | {ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | {p["direction"]} | '
                      + (f'{ci_text(q)} | {q["v0.5"]} | {q["v1.0"]} | {q["direction"]} |' if q
                         else 'the same run | - | - | - |'))
        for fam in G['Pol']['pooled']:
            p = G['Pol']['pooled'][fam]
            q = G['Spec']['pooled'].get(fam)
            md.append(f'| pooled {fam} | {ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | '
                      f'{p["direction"]} | ' + (f'{ci_text(q)} | {q["v0.5"]} | {q["v1.0"]} | '
                                                f'{q["direction"]} |' if q else '- | - | - | - |'))
    if H is not None:
        md += ['', '## H. Larger shops', '',
               'd = 100 (released policy - other) / other, seed s against seed s; rule pool = '
               'per-instance best of the four rules; combined verdict with Holm over the eight '
               'combinations of a size; the selected run with the instance-level cluster bootstrap.']
        for size, x in H.items():
            md += ['', f'### {size} ({x["n"]} instances per combination)', '',
                   '| comb. | vs specialist | 0.5% | 1% | dir. | vs rule pool | verdict | selected run '
                   'vs rule pool | verdict |', '|---|---|---|---|---|---|---|---|---|']
            for r in R8:
                p = x.get('parity', {}).get(r)
                y = x['rule_pool'][r]
                z = x.get('selected', {}).get('rows', {}).get(r)
                md.append(f'| {r} | ' + (f'{ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | '
                                         f'{p["direction"]} | ' if p else '- | - | - | - | ')
                          + f'{ci_text(y)} | {y["verdict"]} | '
                          + (f'{ci_text(z)} | {z["verdict"]} |' if z else '- | - |'))
            p = x.get('parity_pooled', {}).get('All')
            z = x.get('selected', {}).get('pooled')
            md.append('| pooled | ' + (f'{ci_text(p)} | {p["v0.5"]} | {p["v1.0"]} | '
                                       f'{p["direction"]} | ' if p else '- | - | - | - | ')
                      + f'{ci_text(x["rule_pool_pooled"])} | {x["rule_pool_pooled"]["direction"]} | '
                      + (f'{ci_text(z)} | {z["direction"]} |' if z else '- | - |'))
    return md


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


if __name__ == '__main__':
    out = dict(generator='scripts/p88_family_inference.py', setting=REG.describe(),
               seeds=SEEDS, boot=B, alpha=ALPHA, simultaneous_family=M_SIM,
               streams=dict(parity=f'one default_rng({STREAM_SEED}), order {TRAIN + HELD}',
                            hierarchical=f'fresh default_rng({FRESH_SEED}) per quantity',
                            instance=f'fresh default_rng({INST_SEED}) per quantity'),
               reproduction_of_p27=REPRO,
               selection=dict(rule='best validation score of the kept checkpoint '
                                   '(arm_registry.select_seed)', seed=SEL,
                              records={f's{s}': dict(score=v[0], index=v[1])
                                       for s, v in SEL_REC.items()}),
               parity=dict(components=PAR, iut=PAR_IUT, pooled=PAR_POOL,
                           pooled_instance_level=PAR_POOL_INST,
                           published_instance_level=PAR_INST,
                           iut_published_instance_level=PAR_INST_IUT),
               rule_pool=RULE,
               control=dict(components=CTL, iut=CTL_IUT, pooled=CTL_POOL, rule_pool=CTL_RULE),
               production={f'{k[0]}|{k[1]}': v for k, v in PRODR.items()},
               production_policy_minus_rival={f'{k[0]}|{k[1]}': v for k, v in PRODD.items()},
               production_policy_minus_rival_rule=(
                   'd = 100 (policy - rival) / rival per instance (negative = policy shorter), '
                   'recomputed from the arrays; same sources, instance-level cluster bootstrap '
                   f'(B = {B}, fresh default_rng({INST_SEED})) as production; regimes_ahead and seed '
                   'counts use the interval below zero'),
               superiority_verdict_rule=(
                   f'ahead: hierarchical {100 * (1 - ALPHA):g}% CI below zero and both Holm-adjusted '
                   f'one-sided p-values (seed-level t, bootstrap; H1 first arm faster) below {ALPHA:g}; '
                   'behind: CI above zero and both Holm-adjusted one-sided p-values for H1 first arm '
                   f'slower (own Holm family over the eight combinations) below {ALPHA:g}; '
                   'not distinguishable otherwise'))
    if not SOLVER_ROWS:
        out['production_solver_1800s'] = dict(computed=False, reason=SOLVER_SKIP)
    # ---- appended: sections F, G, H and their state
    out['sections'] = {
        sec: dict({k: v for k, v in STATE[sec].items() if k not in ('missing', 'wrong_length')},
                  computed=SEC[sec] is not None, required=sec in REQUIRED,
                  missing_n=len(STATE[sec]['missing']), missing=STATE[sec]['missing'][:40],
                  wrong_length=STATE[sec]['wrong_length'][:40])
        for sec in SECTIONS}
    out['sections_rule'] = (
        'a section is computed when every input array of the active environment and seeds '
        'exists with as many rows as its folder has test instances, and listed as skipped with '
        'the reason otherwise; --require-sections refuses a skipped section. d = 100 (first - '
        'second) / second, seed s against seed s; hierarchical intervals from a fresh '
        f'default_rng({FRESH_SEED}) per quantity, instance-level intervals (selected run) from a '
        f'fresh default_rng({INST_SEED}); combined verdicts as superiority_verdict_rule, Holm over '
        'the twelve pairs (F) or the eight combinations of a size (H)')
    out['rule_change'] = SEC['F']
    out['full_combination_specialist'] = SEC['G']
    out['larger_shops'] = SEC['H']
    os.makedirs(os.path.dirname(OUT_JSON) or '.', exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(jsonable(out), f, indent=1)

    md = [f'# Family inference ({REG.describe()})', '',
          f'B = {B}; verdicts at +-0.5% / +-1% from the hierarchical 95% interval; '
          f'IUT for joint claims; Holm for superiority. Reproduction of p27: '
          f'{REPRO["macros_compared"]} macros identical.', '',
          '## A. Released policy vs specialist (10x25)', '',
          '| comb. | mean | hier. 95% CI | Bonf. CI | seed-t CI | 0.5% | 1% | direction | per seed |',
          '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        c = PAR[r]
        md.append(f'| {r} | {c["mean"]:+.2f} | [{c["lo"]:+.2f}, {c["hi"]:+.2f}] | '
                  f'[{c["sim_lo"]:+.2f}, {c["sim_hi"]:+.2f}] | [{c["t_lo"]:+.2f}, {c["t_hi"]:+.2f}] | '
                  f'{c["v0.5"]} | {c["v1.0"]} | {c["direction"]} | '
                  + ', '.join(f'{m:+.2f}' for m in c['seed_means']) + ' |')
    md += ['', '| family | margin | equivalent on all | non-inferior on all | failing (eq.) |',
           '|---|---|---|---|---|']
    for fam in PAR_FAM:
        for mg in MARGINS:
            e, ni = PAR_IUT[f'{fam}|{mg}|equivalent'], PAR_IUT[f'{fam}|{mg}|non-inferior']
            md.append(f'| {fam} | {mg} | {e["holds"]} | {ni["holds"]} | {",".join(e["failing"]) or "-"} |')
    md += ['', 'Published instance-level intervals (p11, macros_eq.tex) under the same rule:', '',
           '| comb. | published CI | 0.5% | 1% | direction |', '|---|---|---|---|---|']
    for r in R8:
        x = PAR_INST[r]
        md.append(f'| {r} | [{x["lo"]:+.2f}, {x["hi"]:+.2f}] | {x["v0.5"]} | {x["v1.0"]} | {x["direction"]} |')
    md += ['', '| family | margin | equivalent on all | non-inferior on all | failing (eq.) |',
           '|---|---|---|---|---|']
    for fam in PAR_FAM:
        for mg in MARGINS:
            e, ni = PAR_INST_IUT[f'{fam}|{mg}|equivalent'], PAR_INST_IUT[f'{fam}|{mg}|non-inferior']
            md.append(f'| {fam} | {mg} | {e["holds"]} | {ni["holds"]} | {",".join(e["failing"]) or "-"} |')
    md += ['', '## B. Released policy vs rule pool (10x25), Holm over 8', '',
           '| comb. | mean | 95% CI | seed-t p | Holm | boot p | Holm |', '|---|---|---|---|---|---|---|']
    for r, x in RULE.items():
        md.append(f'| {r} | {x["mean"]:+.2f} | [{x["lo"]:+.2f}, {x["hi"]:+.2f}] | {x["p_t"]:.2g} | '
                  f'{x["p_t_holm"]:.2g} | {x["p_boot"]:.2g} | {x["p_boot_holm"]:.2g} |')
    md += ['', '## C. Single-rule control (10x25_blindref)', '',
           '| comb. | vs released | 0.5% | 1% | vs specialist | 0.5% | 1% | vs pool | Holm p |',
           '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        a, b, q = CTL['Rel'][r], CTL['Spec'][r], CTL_RULE[r]
        md.append(f'| {r} | {a["mean"]:+.2f} [{a["lo"]:+.2f}, {a["hi"]:+.2f}] | {a["v0.5"]} | {a["v1.0"]} | '
                  f'{b["mean"]:+.2f} [{b["lo"]:+.2f}, {b["hi"]:+.2f}] | {b["v0.5"]} | {b["v1.0"]} | '
                  f'{q["mean"]:+.2f} [{q["lo"]:+.2f}, {q["hi"]:+.2f}] | {q["p_t_holm"]:.2g} |')
    md += ['', f'## E. Production scale, single checkpoints (selected: s{SEL})', '']
    if not SOLVER_ROWS:
        md += [f'The 1800 s solver rows are {SOLVER_SKIP}.', '']
    md += ['| set | rival | ' + ' | '.join(f's{s}' for s in SEEDS) + ' | selected | seed mean |',
           '|---|---|' + '---|' * (K + 2)]
    for (folder, rival), blk in PRODR.items():
        md.append(f'| {folder} | {rival} | ' + ' | '.join(
            f'{blk[f"s{s}"]["mean"]:+.2f} [{blk[f"s{s}"]["lo"]:+.2f}, {blk[f"s{s}"]["hi"]:+.2f}]'
            for s in SEEDS) + f' | {blk["selected"]["mean"]:+.2f} | {blk["mean"]["mean"]:+.2f} |')
    md += ['', "## B'. Combined verdict against the rule pool (10x25)", '',
           f'ahead: the hierarchical {100 * (1 - ALPHA):g}% CI lies below zero and the Holm-adjusted '
           f'one-sided p-values for "the first arm is faster" (seed-level t and hierarchical '
           f'bootstrap) are both below {ALPHA:g}. behind: the CI lies above zero and the '
           f'Holm-adjusted one-sided p-values for "the first arm is slower" (each kind with its own '
           f'Holm family over the eight combinations) are both below {ALPHA:g}. not '
           f'distinguishable: otherwise.', '',
           '| arm | comb. | mean | 95% CI | Holm p faster (t / boot) | p slower (t / boot) | '
           'Holm p slower (t / boot) | verdict |', '|---|---|---|---|---|---|---|---|']
    for arm, blk in (('released', RULE), ('control', CTL_RULE)):
        for r, x in blk.items():
            md.append(f'| {arm} | {r} | {x["mean"]:+.2f} | [{x["lo"]:+.2f}, {x["hi"]:+.2f}] | '
                      f'{x["p_t_holm"]:.2g} / {x["p_boot_holm"]:.2g} | {x["p_t_greater"]:.2g} / '
                      f'{x["p_boot_greater"]:.2g} | {x["p_t_greater_holm"]:.2g} / '
                      f'{x["p_boot_greater_holm"]:.2g} | {x["verdict"]} |')
    md += ['', f"## E'. Production scale, policy minus rival over rival (selected: s{SEL})", '',
           'd = 100 (policy - rival) / rival per instance, negative = policy shorter; recomputed from '
           'the arrays with the cluster bootstrap of E.', '']
    if not SOLVER_ROWS:
        md += [f'The 1800 s solver rows are {SOLVER_SKIP}.', '']
    md += ['| set | rival | ' + ' | '.join(f's{s}' for s in SEEDS) + ' | selected | seed mean |',
           '|---|---|' + '---|' * (K + 2)]
    for (folder, rival), blk in PRODD.items():
        md.append(f'| {folder} | {rival} | ' + ' | '.join(
            f'{blk[f"s{s}"]["mean"]:+.2f} [{blk[f"s{s}"]["lo"]:+.2f}, {blk[f"s{s}"]["hi"]:+.2f}]'
            for s in SEEDS) + f' | {blk["selected"]["mean"]:+.2f} | {blk["mean"]["mean"]:+.2f} |')
    md += section_markdown(STATE, SEC)
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\n{len(names)} macros in {OUT_MAC}; wrote {OUT_JSON}, {OUT_MD}')
