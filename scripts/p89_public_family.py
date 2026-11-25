r"""P89: the three principal arms on the public-instance family.

The public family (arm_registry.PUB_*) repeats the factory design on SD2
instances: one policy trained on N, L, S, W, LS and LW (pubx-v1), a single-rule
control trained on N, L, S and W (pubx-ctl) and one specialist per combination
(pubx-solo-<R>), training seeds 301-305, all at 10x5, all under the exact
non-delay mask. Every run and the four dispatching rules (restricted to the
pairs the mask admits) were evaluated greedily on six public test sets
(scripts/eval_newseeds_plan.py, group public):

  pub_SD2_test_10x5+mix    100 instances  in distribution (the headline set)
  pub_SD2_test_20x10+mix   100            size transfer
  pub_Brandimarte           10            distribution transfer
  pub_Hurink_edata          65            distribution transfer (orb7 excluded)
  pub_Hurink_rdata          65            distribution transfer (orb7 excluded)
  pub_Hurink_vdata          65            distribution transfer (orb7 excluded)

This script reads those arrays and writes new outputs only (every macro name
starts with Pub). It changes no existing number.

ARRAYS
  test_results/PUBFAMILY/<set>_xmask/Result_<method>_<R>.npy holds one row per
  base instance, (makespan, seconds). Row i is the i-th descriptor of
  data/PUBLIC/family/<set>/desc/<R>/ in sorted file-name order, because
  eval_family.py walks sorted(glob(...)). Method is greedy+10x5+pubfamily+<run>
  for a trained run and FIFO, MOR, SPT or MWKR for a rule.

CHECKS (before anything is computed; any failure refuses and removes the
outputs)
  * Records: in every set the eight regimes hold the same sorted file names,
    every record of a name names the same base instance, whose file stem is the
    name, with one descriptor seed; the names equal the set's dataset_meta.json
    list and the expected count (orb7 is the only exclusion, Hurink sets); the
    best-known table covers every name.
  * Arrays: every expected file exists, has shape [n, 2] for its set, holds
    finite positive makespans and no NaN.
  * Row order: the records are older than every result file, so every run read
    the same sorted list; each run's summary file reports n/n feasible
    schedules, per-regime means equal to the array means, and (regime N) a mean
    gap to best known equal to the one recomputed here from the arrays in the
    sorted-name order (a permutation of the rows changes this mean); in regime
    N no makespan of any run or rule is below the instance's lower bound or
    below a reference proved optimal.
  * Duplicates: rule arrays found in any other folder of the results root for
    the same set must equal the stored ones.
  * Training records: every run's log states the 29,602-parameter network and
    the exact mask, and its validation lines match valiquality_<model>.txt.

UNITS AND ESTIMATORS (scripts/stats_family.py)
  d = 100 (a - b) / b per training seed and base instance, seed s of one arm
  against seed s of the other; positive d means the first arm is slower.
  * Crossed hierarchical bootstrap over seeds and base instances (combinations
    of a base instance kept together when pooled), B = --boot, a fresh
    default_rng(11) per quantity.
  * Seed-level Student t interval over the five per-seed means (4 df).
  * One fixed checkpoint (the selected seed): cluster bootstrap over base
    instances, a fresh default_rng(7) per quantity.
  Verdicts (stats_family.verdict) at margins 0.5% and 1% on the hierarchical
  95% interval; the seed-t interval gets the same rule in a second column.
  Joint claims by the intersection-union test; superiority over the rules by
  Holm over the eight combinations of a set.

QUANTITIES
  1. policy vs the specialist of each combination, all eight, pooled over all
     eight, the six training and the two held-out combinations;
  2. on LS, LW (seen by the policy, not by the control) and SW, LSW (seen by
     neither): policy vs control and control vs specialist;
  3. policy (seed mean and selected seed) vs the per-instance best of the four
     rules and vs each rule: mean d, shares shorter / equal / longer;
  4. regime N: gap to the best-known value for policy, control, N specialist
     and the rules, with the status of the references;
  5. selected seed by arm_registry.select_seed (best validation score of the
     kept checkpoint) on the public records;
  6. per-seed gap of every arm to the per-instance best rule.
  7. sensitivity sets (arm_registry.PUB_SENS_SETS: lag scale 0.5 on SD2 10x5,
     the stored policies evaluated on re-drawn records): on the combinations
     that contain L, policy vs specialist, vs control and vs the best rule and
     each rule, beside the same comparison on the scale-1.0 records, and the
     paired change between the two scales.
  8. equal-experience policy (arm_registry.PUB_EQ_*, pubx-eq-s<seed>: the released
     recipe with 6000 updates, so that each of its six combinations gets as many
     episodes as a specialist gets of its one; scripts/run_pubfamily_queue.sh
     ARMS=eq, evaluated by the pubeq group of scripts/eval_newseeds_plan.py), seed
     s against seed s: vs the specialist of each combination and pooled over all
     eight, the six training and the two held-out combinations, with the joint
     claims; vs the released policy per combination and pooled; per test set;
     regime-N gap to best known; validation at the kept checkpoints. Macros
     PubEq*: per combination and the joint claims on the headline set
     pub_SD2_test_10x5+mix, pooled on every test set.
  Diagnostics: validation at the kept checkpoints (specialist against
  policy), position of each kept checkpoint, rule effects R/N, exact ties.

COMBINED VERDICT AGAINST THE BEST RULE (section 3, every test set)
  One word per combination (superiority_verdict), from the hierarchical 95%
  interval [lo, hi] of d (policy minus per-instance best rule, over the rule) and
  Holm-adjusted one-sided p-values over the eight combinations of the set:
    ahead                hi < 0, and the Holm-adjusted seed-level t p and bootstrap p
                         for H1: d < 0 are both below 0.05;
    behind               lo > 0, and the Holm-adjusted seed-level t p and bootstrap p
                         for H1: d > 0 are both below 0.05 (each kind with its own
                         Holm family over the same eight combinations);
    not distinguishable  otherwise.
  The H1: d > 0 p-values come from stats_family.seed_t_p and boot_p_one_sided with
  alternative 'greater', on the same per-seed means and the same hierarchical
  draws as the H1: d < 0 ones, so no random number is added. Outputs: JSON keys
  p_t_greater, p_boot_greater, p_t_greater_holm, p_boot_greater_holm and verdict in
  sets.<set>.rules.<R>.best; macros Pub<set>Rule<R>Verdict and the counts
  Pub<set>RuleVerdictAheadN and Pub<set>RuleVerdictBehindN, written after the
  macros of sections 1 to 7 and before those of section 8 (n_macros counts every
  macro except these, n_macros_verdict counts these); a Markdown section after
  the diagnostics. The sensitivity sets of section 7 run no Holm family and get
  no verdict.

DIRECTION OF THE POOLED ROWS
  Every pooled comparison with interval macros (<name>, <name>Lo, <name>Hi) gets
  <name>Dir: ahead, behind or neither, from stats_family.direction on the unrounded
  interval ends (the rule of the per-combination Dir macros, the 'direction' key the
  JSON already holds for the row). A bound close to zero prints as -0.00 or +0.00;
  the comment of each Dir macro shows the ends to four decimals. Sections 1, 2 and 7: written after the verdict macros and before
  section 8 (n_macros does not count them, n_macros_dir does). Section 8: written
  after the macros of section 8 and only with them (n_macros counts them, as every
  other macro of section 8).

EQUAL-EXPERIENCE ARM (section 8)
  Computed only when every file of the five pubx-eq runs exists: checkpoint,
  training log with its completion line, validation record, configuration, and
  the eight arrays and the summary in each test and sensitivity folder. When no
  file of the arm exists, the script runs and writes exactly what it writes
  without the arm. When some exist and not all (or an unexpected one, such as a
  sixth seed), it refuses. The arm's files pass the checks above (shape, summary,
  record order, bounds, free check on the sensitivity set, training record, now
  with 6000 updates); its configuration must equal that of pubx-v1 of the same
  seed except for the name and max_updates. Its arrays are kept apart from the
  principal arms', so every number of sections 1 to 7 and every count of their
  checks is the same with or without the arm; its quantities use the same
  estimators and a fresh stream per quantity like every other.

SENSITIVITY CHECKS (before anything is computed)
  The sensitivity records carry the names, base instances and descriptor seeds
  of their test set and differ from its records only in the horizon, the lags
  (combinations with L), the outage calendars (combinations with W) and an
  explicit attached-setup convention; their rule parameters differ only in the
  declared change. Because N and S read neither lags nor outages, every N and S
  array of every run and rule must equal the test set's array bit for bit.

REFERENCES OF THE PUBLIC BENCHMARK (section 9, REF_SET = pub_SD2_test_10x5+mix)
  The references are the records of scripts/p91_public_cpsat_refs.py (campaign REF_CAMPAIGN:
  compact model, 150 s, 2 workers, solver seed 1; folder p91.default_out), read with p91's
  own readers (read_records, load_refs, record_validity). A record counts only when p91
  marked it valid, record_validity still says valid, and its set, combination, instance,
  model, budget, workers, solver seed, descriptor seed and rules version are those of the
  campaign and of the test set's record; any other record is listed with the reason.
  Fields read: objective (the reference makespan), bound, status (OPTIMAL = proved),
  solve_time_s, warm_makespan, valid and the campaign fields.
  Computed only when every one of the 8 x 100 references is counted; otherwise one line
  with the counts (JSON key references, the Markdown, one comment line in the macro file)
  and every other section unchanged. Per combination and pooled over the eight (a base
  instance with all its combinations), d = 100 (method - reference) / reference: policy,
  control and specialist (five runs, crossed hierarchical bootstrap, fresh
  default_rng(11)); the per-instance best of the four rules and each rule (instance-level
  cluster bootstrap, fresh default_rng(7)). Interval and direction only, no margin verdict
  and no combined verdict: a gap to a reference is not a comparison of two methods under
  test. Counts: references proved optimal, stopped at the time limit with a gap between
  objective and bound (mean gap in % of the objective), the policy's schedules equal to,
  longer and shorter than the reference (per run, averaged; the shorter cases are listed in
  the JSON); regime N against the best-known table: references below, equal to, above.
  Records that are unexpected without being wrong are listed (a reference longer than its
  warm start, a rule shorter than the reference, an unproved solve that ended before the
  limit, a status other than OPTIMAL or FEASIBLE).
  Refusals (the whole run, as every other check, since the same records and arrays feed
  sections 1 to 4), checked on every counted reference whether the section is complete or
  not: a regime-N reference below a best-known value the table marks proved optimal, a
  regime-N bound above a best-known makespan, a stored makespan of a run or a rule below
  the largest counted bound of its combination and sub-combinations; each names the instance.
  Macros PubRef..., after every other macro. --partial-references DIR computes the section
  on the combinations that are complete and writes only REF_PARTIAL_FILES into DIR, which
  may not be the repository root or lie in its paper, manuscript or results folders.

THE FAMILY UNDER THE ACTIVE MASK (--family active)
  Every recipe above trained again under the active mask (mask_mode active, the condition of
  Giffler and Thompson; arm_registry.pub_rule_model with the rule word act: pubx-act-s<seed>,
  pubx-ctlact-s<seed>, pubx-soloact-<R>-s<seed>, pubx-eqact-s<seed>; queue
  scripts/run_pubfamily_queue.sh ARMS=actfam and ARMS=acteq; evaluation groups pubact,
  pubactfam, pubactsens and pubacteq of scripts/eval_newseeds_plan.py), and the four rules
  rolled out under the active mask (scripts/p109_public_active_rules.py, folder
  test_results/PUBFAMILY/rules_act/<set>_xmask, the layout of the stored rule arrays). The
  same code computes every section with the same definitions, estimators, streams and checks;
  only the run names, the folder of the rule arrays and the outputs change, and these checks
  are added: every training log states the active mask (the trainer's line 'OPTIONS:
  mask_mode=active pair_extra=none'), every configuration equals the configuration of its
  non-delay counterpart of the same seed (arm_registry.pub_rule_counterpart) except for
  model_suffix and mask_mode (active; keys the trainer added since count as equal at their
  inert default), the rule summaries record 'pairs: mask_mode=active'. Section 8 compares the
  equal-experience recipe's configuration with pubx-eq of the same seed in the same way.
  Section 10 (active only), the comparison across masks: the non-delay family is read and
  checked as in a default run, and per test set, and on the halved-lag sensitivity set
  (macro word LagHalf) when both families hold its arrays of every run and rule (mask_sets)
    * d = 100 (a - b) / b per training seed s and base instance, a the makespan of a run under
      the active mask, b that of its non-delay counterpart of seed s (negative = the active
      mask gives the shorter schedule): the policy and the single-rule control on each of the
      eight combinations, the specialist of each combination on its own; crossed hierarchical
      bootstrap, seed-level t interval, margin verdicts and direction (component, fresh
      default_rng(11) per quantity); pooled (a base instance with all its combinations) over
      All, Train, Held, Setup (S, LS, SW, LSW) and NoSetup (N, L, W, LW), the specialists over
      All, Setup and NoSetup (each combination by its own specialist);
    * the per-instance best of the four rules and each rule: d = 100 (a - b) / b per base
      instance, a under the active mask and b under the non-delay mask, instance-level cluster
      bootstrap (fresh default_rng(7)), per combination and pooled; the rule with the lowest
      mean makespan under each mask and the share of instances on which each rule is the best.
  Outputs results/public_family_act.json, results/public_family_act.md and
  manuscript/macros_pubfam_act.tex, every macro named PubFamAct...: the macro of the default
  run with Pub replaced by PubFamAct (same definition), and the macros of section 10,
  PubFamActMask<set>....

PARTIAL REPORT (--partial, either family)
  Without it a missing file refuses. With --partial --out <folder> (not the repository root
  nor its paper, manuscript or results folders) the sections are computed on the training
  seeds for which every run of the principal arms has every file (at least two seeds); section
  8 when the equal-experience arm is complete on those seeds, section 9 when its references
  are; the rule arrays and the records must be complete. Every output says PARTIAL REPORT and
  names the seeds read.

Usage: python scripts/p89_public_family.py [--boot 20000] [--out <root>]
       [--root <repository root>] [--family nondelay|active] [--partial]
       python scripts/p89_public_family.py --partial-references <folder> [--boot 20000]
"""
import argparse
import ast
import csv
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
import p91_public_cpsat_refs as P91  # noqa: E402  (the reader of the reference records, section 9)
import p109_public_active_rules as P109  # noqa: E402  (the folder of the rule arrays under the active mask)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R8 = list(REG.COMBOS)
TRAIN = list(REG.TRAIN_COMBOS)
HELD = list(REG.HELD_OUT)
CTL_SEEN = list(REG.CONTROL_COMBOS)
PAIR = [r for r in TRAIN if r not in CTL_SEEN]          # LS, LW
UNSEEN = PAIR + HELD                                    # combinations the control never saw
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
SEEDS = tuple(REG.PUB_SEEDS)
WORDS = [REG.SEED_WORDS[s] for s in SEEDS]
MARGINS = (0.5, 1.0)
MW = {0.5: 'Half', 1.0: 'One'}
HIER_SEED, INST_SEED = 11, 7
XS = REG.FOLDER_SUFFIX['exact']
N_PARAMS = 29602
RULES_VERSION = 'public-rules-1'

# set -> (macro word, role, expected instances, instances left out of the family)
SETS = {
    'pub_SD2_test_10x5+mix': ('Sdten', 'in distribution', 100, ()),
    'pub_SD2_test_20x10+mix': ('Sdtwenty', 'size transfer', 100, ()),
    'pub_Brandimarte': ('Brand', 'distribution transfer', 10, ()),
    'pub_Hurink_edata': ('Edata', 'distribution transfer', 65, ('orb7',)),
    'pub_Hurink_rdata': ('Rdata', 'distribution transfer', 65, ('orb7',)),
    'pub_Hurink_vdata': ('Vdata', 'distribution transfer', 65, ('orb7',)),
}
# sensitivity sets of the public family (arm_registry.PUB_SENS_SETS) -> macro word
SENS_WORD = {'pub_SD2_test_10x5+mix_lag050': 'LagHalf'}
L_COMBOS = [r for r in R8 if 'L' in r]
OUT_FILES = ('results/public_family_inference.json', 'results/public_family_inference.md',
             'manuscript/macros_public.tex')
# equal-experience arm (section 8): the set of its per-combination macros, and the
# configuration keys the trainer added after the stored pubx-v1 runs were written,
# with the inert defaults they take (as in scripts/run_pubfamily_queue.sh)
EQ_HEADLINE = 'pub_SD2_test_10x5+mix'
EQ_INERT_NEW = {'dr_shift': ('',), 'delta_mode': (None,), 'mask_mode': (None, 'nondelay'),
                'pair_extra': (None, 'none'), 'reward_mode': ('dense',),
                'pub_type_source': ('none',)}
# references of the public benchmark (section 9): the test set they cover, the settings of
# the campaign of scripts/p91_public_cpsat_refs.py that wrote them (its folder name follows
# from these), the macro words of the rules
REF_SET = 'pub_SD2_test_10x5+mix'
REF_CAMPAIGN = dict(budget=150.0, workers=2, model='compact', seed=1)
REF_RULE_WORD = {'FIFO': 'Fifo', 'MOR': 'Mor', 'SPT': 'Spt', 'MWKR': 'Mwkr'}
REF_EPS = 1e-6
REF_PARTIAL_FILES = ('public_references_partial.json', 'public_references_partial.md')


class Fam:
    """One family of the public benchmark: the names and paths of its runs, the folder of its
    rule arrays and its outputs. key 'nondelay' is the family of the default run (every name
    and path as before); 'active' the family under the active mask (module docstring)."""

    def __init__(self, key):
        self.key = key
        self.rule = key != 'nondelay'
        if not self.rule:
            self.mask_mode, self.word = 'nondelay', None
            self.out_files = OUT_FILES
            self.prefix = 'Pub'
            self.rule_root = REG.PUB_RESULTS
            self.eq_stem = REG.PUB_EQ_STEM
            self.banner = None
        else:
            self.mask_mode, self.word = key, REG.PUB_RULE_WORD[key]
            self.out_files = (f'results/public_family_{self.word}.json',
                              f'results/public_family_{self.word}.md',
                              f'manuscript/macros_pubfam_{self.word}.tex')
            self.prefix = 'PubFam' + self.word.capitalize()
            self.rule_root = P109.RULE_ROOT[key]
            self.eq_stem = REG.PUB_RULE_STEM['eq'].format(W=self.word)
            self.banner = f'[train_family] OPTIONS: mask_mode={key} pair_extra=none'

    def model(self, arm, combo=None, seed=None):
        if not self.rule:
            return REG.pub_model(arm, combo, seed)
        return REG.pub_rule_model(arm, combo, seed, word=self.word)

    def paths(self, arm, combo=None, seed=None):
        """(checkpoint, training log, validation record, configuration) of a principal run."""
        if not self.rule:
            ck, lp = REG.pub_paths(arm, combo, seed)
            return (ck, lp, REG.pub_vali_log(arm, combo, seed),
                    f'train_log/FAMILY/config_{REG.PUB_PREFIX}{REG.pub_model(arm, combo, seed)}.json')
        return REG.pub_rule_paths(arm, combo, seed, word=self.word)

    def eq_model(self, seed):
        return REG.pub_eq_model(seed) if not self.rule else self.model('eq', None, seed)

    def eq_paths(self, seed):
        return REG.pub_eq_paths(seed) if not self.rule else self.paths('eq', None, seed)


FAMILY_KEYS = ('nondelay', 'active')
RULE_CONFIG_FREE = ('mask_mode', 'model_suffix')     # what a run under the active mask may change
MASK_FAMILIES = {'All': R8, 'Train': TRAIN, 'Held': HELD,
                 'Setup': [r for r in R8 if 'S' in r], 'NoSetup': [r for r in R8 if 'S' not in r]}
SPEC_MASK_FAMILIES = ('All', 'Setup', 'NoSetup')


class Refusal(SystemExit):
    """A check failed; the outputs are removed and nothing is certified."""


def refuse(msg):
    raise Refusal(f'REFUSING: {msg}')


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


# ================================================================== inputs
class Inputs:
    """Every array, record, table and training record the analysis reads, checked."""

    def __init__(self, root, fam=None):
        self.root = root
        self.fam = fam or Fam('nondelay')
        self.config_evidence = {}  # run -> configuration keys that differ from its counterpart
        self.ms = {}           # (set, arm, combo, seed, regime) -> makespans [n]
        self.rule = {}         # (set, rule, regime) -> makespans [n]
        self.names = {}        # set -> sorted instance names (row order)
        self.bk = {}           # set -> dict(bk, lb, status) aligned with names
        self.evidence = {}     # set -> what the order checks found
        self.n_files = 0
        self.eq_ms = {}        # equal-experience arm: (set, seed, regime) -> makespans [n]
        self.eq_evidence = {}  # set -> what the checks of the arm's files found
        self.n_files_eq = 0
        self.desc_seeds = {}   # set -> descriptor seeds aligned with names (section 9)

    def p(self, *parts):
        return os.path.join(self.root, *parts)

    @staticmethod
    def spec(st):
        """(macro word, role, expected instances, exclusions) of a test or sensitivity set."""
        if st in SETS:
            return SETS[st]
        base, _, change = REG.PUB_SENS_SETS[st]
        _, _, n, excl = SETS[base]
        return SENS_WORD[st], f'sensitivity of {base} ({change})', n, excl

    @staticmethod
    def table(st):
        return REG.PUB_SETS[st] if st in REG.PUB_SETS else REG.PUB_SENS_SETS[st][1]

    # ---------------------------------------------------------- records
    def check_records(self, st):
        _, _, n_want, excl = self.spec(st)
        d = self.p('data', 'PUBLIC', 'family', st)
        names, bases, seeds_ = {}, {}, {}
        newest = 0.0
        for r in R8:
            ps = sorted(glob.glob(os.path.join(d, 'desc', r, '*.desc.json')))
            names[r] = [os.path.basename(x)[:-len('.desc.json')] for x in ps]
            recs = []
            for x in ps:
                newest = max(newest, os.stat(x).st_mtime)
                with open(x) as f:
                    recs.append(json.load(f))
            for nm, rec in zip(names[r], recs):
                if rec.get('regime') != r or rec.get('rules_version') != RULES_VERSION:
                    refuse(f'{st}/{r}/{nm}: record regime {rec.get("regime")!r} or rules '
                           f'version {rec.get("rules_version")!r} is not {r!r}, {RULES_VERSION!r}')
                if os.path.basename(rec['base_instance']) != nm:
                    refuse(f'{st}/{r}/{nm}: base instance {rec["base_instance"]} has another name')
            bases[r] = [rec['base_instance'] for rec in recs]
            seeds_[r] = [rec['desc_seed'] for rec in recs]
        for r in R8:
            if names[r] != names['N'] or bases[r] != bases['N'] or seeds_[r] != seeds_['N']:
                refuse(f'{st}: regime {r} does not share file names, base instances and '
                       f'descriptor seeds with N, so row i is not one base instance')
        nm = names['N']
        if len(nm) != n_want or len(set(bases['N'])) != len(nm):
            refuse(f'{st}: {len(nm)} records ({len(set(bases["N"]))} distinct base instances), '
                   f'want {n_want}')
        with open(os.path.join(d, 'dataset_meta.json')) as f:
            meta = json.load(f)
        excluded = sorted(os.path.splitext(e['instance'])[0] for e in meta.get('excluded', []))
        if sorted(meta['instances']) != nm or meta['n_instances'] != n_want or \
                excluded != sorted(excl) or set(excl) & set(nm):
            refuse(f'{st}: dataset_meta.json lists {meta["n_instances"]} instances, excluded '
                   f'{excluded}; want the {n_want} records and exclusions {list(excl)}')
        self.names[st] = nm
        self.desc_seeds[st] = seeds_['N']
        self.evidence[st] = dict(n=len(nm), excluded=excluded,
                                 lag_scale=meta.get('params', {}).get('lag_scale'),
                                 desc_seed0=meta.get('desc_seed0'), records_newest_mtime=newest)
        self.check_best_known(st)

    def check_best_known(self, st):
        path = self.p(self.table(st))
        if not os.path.exists(path):
            raise REG.MissingResult(f'missing best-known table {self.table(st)}')
        with open(path) as f:
            rows = {r['instance']: r for r in csv.DictReader(f)}
        miss = [x for x in self.names[st] if x not in rows or not rows[x]['best_known_makespan']]
        if miss:
            refuse(f'{st}: best-known table has no value for {miss[:5]}')
        extra = sorted(set(rows) - set(self.names[st]))
        if extra != sorted(self.spec(st)[3]):
            refuse(f'{st}: best-known table rows outside the family {extra}, want {self.spec(st)[3]}')
        bk = np.array([float(rows[x]['best_known_makespan']) for x in self.names[st]])
        lb = np.array([float(rows[x]['lower_bound_if_known']) if rows[x]['lower_bound_if_known']
                       else np.nan for x in self.names[st]])
        status = [rows[x]['optimal'] for x in self.names[st]]
        proved = np.array([s.startswith('yes') or (np.isfinite(b) and b == v)
                           for s, b, v in zip(status, lb, bk)])
        self.bk[st] = dict(bk=bk, lb=lb, status=status, proved=proved,
                           claimed=np.array(['claimed' in s for s in status]))

    # ---------------------------------------------------------- arrays
    def folder(self, st):
        return self.p(REG.PUB_RESULTS, st + XS)

    def rule_folder(self, st):
        """Folder of the rule arrays of the family (the run folder in the default family)."""
        return self.p(self.fam.rule_root, st + XS)

    def _load(self, path, n, eq=False):
        if not os.path.exists(path):
            raise REG.MissingResult(f'missing result file: {os.path.relpath(path, self.root)}')
        a = np.load(path)
        if a.ndim != 2 or a.shape != (n, 2):
            refuse(f'{os.path.relpath(path, self.root)} has shape {a.shape}, want ({n}, 2)')
        if not np.all(np.isfinite(a)) or np.any(a[:, 0] <= 0):
            refuse(f'{os.path.relpath(path, self.root)} holds NaN, infinite or non-positive values')
        if eq:
            self.n_files_eq += 1
        else:
            self.n_files += 1
        return a[:, 0].astype(float), os.stat(path).st_mtime

    def runs(self):
        """(arm, combo, regimes) of the evaluated runs."""
        return [('released', None, R8), ('control', None, R8)] + \
               [('specialist', r, [r]) for r in R8]

    def load_set(self, st):
        n = len(self.names[st])
        oldest = np.inf
        for arm, combo, regs in self.runs():
            for s in SEEDS:
                name = self.fam.model(arm, combo, s)
                for r in regs:
                    v, t = self._load(os.path.join(
                        self.folder(st), f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'), n)
                    self.ms[(st, arm, combo, s, r)] = v
                    oldest = min(oldest, t)
                self.check_summary(st, f'summary_{REG.PUB_PREFIX}{name}{XS}.md',
                                   {r: f'greedy+{REG.PUB_PREFIX}{name}' for r in regs},
                                   lambda r, m, a=arm, c=combo, s=s: self.ms[(st, a, c, s, r)])
        for q in RULES:
            for r in R8:
                v, t = self._load(os.path.join(self.rule_folder(st), f'Result_{q}_{r}.npy'), n)
                self.rule[(st, q, r)] = v
                oldest = min(oldest, t)
        self.check_summary(st, f'summary_rules{XS}.md', None,
                           lambda r, m: self.rule[(st, m, r)], folder=self.rule_folder(st))
        if self.fam.rule:
            sp = os.path.join(self.rule_folder(st), f'summary_rules{XS}.md')
            if f'pairs: mask_mode={self.fam.mask_mode}' not in open(sp).read().splitlines():
                refuse(f'{os.path.relpath(sp, self.root)} does not record the rules under the '
                       f'{self.fam.mask_mode} mask (pairs: mask_mode={self.fam.mask_mode})')
        if not self.evidence[st]['records_newest_mtime'] <= oldest:
            refuse(f'{st}: a descriptor record is newer than a result file, so the runs may '
                   f'not have read one sorted record list')
        self.evidence[st]['results_oldest_mtime'] = oldest
        self.check_lower_bounds(st)
        self.check_duplicates(st)

    def check_summary(self, st, fname, methods, get, evidence=None, folder=None):
        """n/n feasible, means equal to the arrays, regime-N gap to best known equal to
        the recomputed one (methods: regime -> method label; None = the four rules).
        The count goes to evidence (default: the set's evidence); folder defaults to the
        set's run folder."""
        path = os.path.join(folder or self.folder(st), fname)
        if not os.path.exists(path):
            raise REG.MissingResult(f'missing summary file: {os.path.relpath(path, self.root)}')
        n = len(self.names[st])
        rows = []
        for ln in open(path):
            c = [x.strip() for x in ln.strip().strip('|').split('|')]
            if len(c) == 7 and c[0] in R8:
                rows.append(c)
        want = ({(r, q) for r in R8 for q in RULES} if methods is None
                else {(r, m) for r, m in methods.items()})
        got = {(c[0], c[1]) for c in rows}
        if got != want or len(rows) != len(want):
            refuse(f'{fname}: rows {sorted(got - want)[:3]} / missing {sorted(want - got)[:3]}')
        bk = self.bk[st]['bk']
        for r, m, mean, _, feas, _, gap in rows:
            v = get(r, m)
            if feas != f'{n}/{n}':
                refuse(f'{st} {fname}: {m} in {r} has {feas} feasible schedules')
            if abs(float(mean) - v.mean()) > 0.05 + 1e-9:
                refuse(f'{st} {fname}: {m} in {r} summary mean {mean}, array mean {v.mean():.3f}')
            if r == 'N':
                g = re.match(r'(-?[\d.]+)% \((\d+)\)', gap)
                here = float(np.mean(rel(v, bk)))
                if not g or int(g.group(2)) != n or abs(float(g.group(1)) - here) > 0.005 + 1e-9:
                    refuse(f'{st} {fname}: {m} gap to best known {gap!r}, recomputed in the '
                           f'sorted-name order {here:.3f}% ({n}); the row order is not the record order')
        ev = self.evidence[st] if evidence is None else evidence
        ev.setdefault('summaries_checked', 0)
        ev['summaries_checked'] += 1

    def check_lower_bounds(self, st, vecs=None, evidence=None):
        """Regime-N makespans at or above the bound; vecs default to every run and rule
        of the set, the counts go to evidence (default: the set's evidence)."""
        b = self.bk[st]
        floor = np.where(b['proved'], b['bk'], b['lb'])          # nan where nothing is known
        if vecs is None:
            vecs = [v for (s_, a, c, _, r), v in self.ms.items() if s_ == st and r == 'N'] + \
                   [self.rule[(st, q, 'N')] for q in RULES]
        known = np.isfinite(floor)
        for v in vecs:
            if np.any(v[known] < floor[known] - 1e-9):
                i = int(np.argmax(v[known] < floor[known] - 1e-9))
                refuse(f'{st}: a regime-N makespan {v[known][i]} lies below the bound '
                       f'{floor[known][i]} of {np.array(self.names[st])[known][i]}')
        ev = self.evidence[st] if evidence is None else evidence
        ev['n_bound_checked'] = int(known.sum())
        ev['n_arrays_bound_checked'] = len(vecs)

    def check_duplicates(self, st):
        """Rule arrays of the same set in any other folder must equal the stored ones."""
        base = self.p(self.fam.rule_root)
        found = []
        registered = {x + XS for x in list(REG.PUB_SETS) + list(REG.PUB_SENS_SETS)}
        for d in sorted(glob.glob(os.path.join(base, st + '*'))):
            if os.path.abspath(d) == os.path.abspath(self.rule_folder(st)) or not os.path.isdir(d) \
                    or os.path.basename(d) in registered:     # another set's own folder
                continue
            for q in RULES:
                for r in R8:
                    f = os.path.join(d, f'Result_{q}_{r}.npy')
                    if not os.path.exists(f):
                        continue
                    a = np.load(f)
                    a = a[:, 0] if a.ndim > 1 else a
                    if len(a) == len(self.names[st]):
                        found.append(os.path.relpath(f, self.root))
                        if not np.array_equal(a, self.rule[(st, q, r)]):
                            refuse(f'{os.path.relpath(f, self.root)} differs from the stored rule array')
        self.evidence[st]['duplicate_rule_files'] = found

    # ---------------------------------------------------------- sensitivity sets
    def check_sensitivity(self, st):
        """Records against the test set's records; the free check on N and S."""
        base, _, change = REG.PUB_SENS_SETS[st]
        if self.names[st] != self.names[base]:
            refuse(f'{st}: instance names differ from {base}')
        fa = self.p('data', 'PUBLIC', 'family', base)
        fb = self.p('data', 'PUBLIC', 'family', st)
        changed = {}
        for r in R8:
            allowed = {'horizon', 'setup_convention'} | ({'lag'} if 'L' in r else set()) | \
                      ({'windows'} if 'W' in r else set())
            for nm in self.names[st]:
                with open(os.path.join(fa, 'desc', r, f'{nm}.desc.json')) as f:
                    a = json.load(f)
                with open(os.path.join(fb, 'desc', r, f'{nm}.desc.json')) as f:
                    b = json.load(f)
                diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
                conv = {a.get('setup_convention', 'attached'), b.get('setup_convention', 'attached')}
                if not diff <= allowed or conv != {'attached'}:
                    refuse(f'{st}/{r}/{nm}: the record differs from {base} in {sorted(diff)}, '
                           f'allowed {sorted(allowed)} (setup convention {sorted(conv)})')
                for k in diff:
                    changed[(r, k)] = changed.get((r, k), 0) + 1
        with open(os.path.join(fa, 'dataset_meta.json')) as f:
            pa = json.load(f)['params']
        with open(os.path.join(fb, 'dataset_meta.json')) as f:
            pb = json.load(f)['params']
        dp = {k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k)}
        if dp != set(change) or any(pb[k] != v for k, v in change.items()):
            refuse(f'{st}: rule parameters differ from {base} in {sorted(dp)}, want {change}')
        n_cmp, n_arr = 0, 0
        for (s_, a, c, s, r), v in list(self.ms.items()):
            if s_ == st and r in ('N', 'S'):
                w = self.ms[(base, a, c, s, r)]
                if not np.array_equal(v, w):
                    refuse(f'free check failed: {st} {a} {c or ""} s{s} {r}: '
                           f'{int((v != w).sum())} of {len(v)} makespans differ from {base}')
                n_cmp += len(v)
                n_arr += 1
        for q in RULES:
            for r in ('N', 'S'):
                if not np.array_equal(self.rule[(st, q, r)], self.rule[(base, q, r)]):
                    refuse(f'free check failed: {st} {q} {r} differs from {base}')
                n_cmp += len(self.rule[(st, q, r)])
                n_arr += 1
        self.evidence[st].update(
            base=base, change=change, records_compared=len(self.names[st]) * len(R8),
            fields_changed={f'{r}|{k}': v for (r, k), v in sorted(changed.items())},
            free_check=dict(arrays=n_arr, makespans=n_cmp, identical=True))

    # ---------------------------------------------------------- training records
    VALI = re.compile(r'^\[vali @(\d+)\] rel=([\d.]+) \(best ([\d.]+)\)\s+(.*)$')

    def training_record(self, arm, combo, seed, paths=None):
        """Validation history of one run: per validation the score and the greedy
        validation makespan of each trained combination; the kept checkpoint.
        paths=(training log, validation record) reads another run's files."""
        if paths:
            lp, vp = paths
        else:
            _, lp, vp, _ = self.fam.paths(arm, combo, seed)
        for x in (lp, vp):
            if not os.path.exists(self.p(x)):
                raise REG.MissingResult(f'missing training record: {x}')
        txt = open(self.p(lp), errors='replace').read()
        if f'params total={N_PARAMS}' not in txt:
            refuse(f'{lp} does not state the {N_PARAMS}-parameter network')
        if 'first_mask=exact mask_ties=exact' not in txt:
            refuse(f'{lp} was not trained under the exact mask')
        if self.fam.banner and self.fam.banner not in txt.splitlines():
            refuse(f'{lp} does not state the {self.fam.mask_mode} mask ({self.fam.banner!r})')
        vq = [float(x) for x in ast.literal_eval(open(self.p(vp)).read())]
        lines = [m.groups() for m in map(self.VALI.match, txt.splitlines()) if m]
        if len(lines) != len(vq) or any(abs(float(g[1]) - v) > 6e-5 for g, v in zip(lines, vq)):
            refuse(f'{lp}: {len(lines)} validation lines do not match the {len(vq)} scores of {vp}')
        per = [dict(re.findall(r'\b([A-Z]+)=([\d.]+)', g[3])) for g in lines]
        k = int(np.argmin(vq))
        steps = [int(g[0]) for g in lines]
        q3 = (3 * len(vq)) // 4
        return dict(score=vq[k], kept_update=steps[k], last_update=steps[-1],
                    kept_fraction=steps[k] / steps[-1],
                    kept={r: float(v) for r, v in per[k].items()},
                    final={r: float(v) for r, v in per[-1].items()},
                    last_quarter_gain=100.0 * (min(vq[:q3]) - min(vq[q3:])) / min(vq[:q3]))

    def validation_record(self, arm, combo=None, seed=None):
        """arm_registry.validation_record of a principal run of the family."""
        if not self.fam.rule:
            return REG.pub_validation_record(arm, combo, seed, root=self.root)
        _, lp, vp, _ = self.fam.paths(arm, combo, seed)
        return REG.validation_record(arm, combo, seed, paths=(self.p(vp), self.p(lp)))

    def rule_config(self, cp, ref_p, want=None):
        """The configuration cp of a run under the family's mask against ref_p of its
        non-delay counterpart: equal except for model_suffix and mask_mode, mask_mode equal to
        the family's in cp and nondelay (or absent) in ref_p, keys added to the trainer since
        equal at their inert default; want: further keys and their required values in cp.
        Returns the keys that differ."""
        for x in (ref_p, cp):
            if not os.path.exists(self.p(x)):
                raise REG.MissingResult(f'missing training record: {x}')
        with open(self.p(ref_p)) as f:
            ref = json.load(f)
        with open(self.p(cp)) as f:
            new = json.load(f)
        inert = {k: v for k, v in EQ_INERT_NEW.items() if k != 'mask_mode'}
        diff = sorted(k for k in set(ref) | set(new)
                      if ref.get(k, '<absent>') != new.get(k, '<absent>') and not
                      (k in inert and k not in ref and new.get(k) in inert[k]))
        if new.get('mask_mode') != self.fam.mask_mode or \
                ref.get('mask_mode', 'nondelay') != 'nondelay':
            refuse(f'mask_mode {new.get("mask_mode")!r} in {cp} and {ref.get("mask_mode")!r} in '
                   f'{ref_p}; want {self.fam.mask_mode!r} and nondelay')
        bad = {k: (v, new.get(k)) for k, v in (want or {}).items() if new.get(k) != v}
        if set(diff) - set(RULE_CONFIG_FREE) or bad:
            refuse(f'{cp} is not the recipe of {ref_p} under the {self.fam.mask_mode} mask: differs '
                   f'in ' + str({k: (ref.get(k), new.get(k)) for k in diff if k not in RULE_CONFIG_FREE})
                   + (f'; want {bad}' if bad else ''))
        return diff

    def check_rule_configs(self):
        """Family under a mask: the configuration of every principal run against its non-delay
        counterpart of the same seed (rule_config)."""
        for arm, combo, _ in self.runs():
            for s in SEEDS:
                regs = {'released': TRAIN, 'control': CTL_SEEN}.get(arm, [combo])
                ref_p = (f'train_log/FAMILY/config_{REG.PUB_PREFIX}'
                         f'{REG.pub_rule_counterpart(arm, combo, s)}.json')
                self.config_evidence[self.fam.model(arm, combo, s)] = self.rule_config(
                    self.fam.paths(arm, combo, s)[3], ref_p,
                    want=dict(family_regimes=','.join(regs), seed_train=s, first_mask='exact',
                              mask_ties='exact'))

    # ---------------------------------------------------------- equal-experience arm
    def eq_expected(self):
        """Every file of the five pubx-eq runs, relative to the root."""
        files = []
        for s in SEEDS:
            files += list(self.fam.eq_paths(s))
        for st in list(SETS) + list(SENS_WORD):
            for s in SEEDS:
                m = REG.PUB_PREFIX + self.fam.eq_model(s)
                d = os.path.join(REG.PUB_RESULTS, st + XS)
                files += [os.path.join(d, f'Result_greedy+{m}_{r}.npy') for r in R8]
                files.append(os.path.join(d, f'summary_{m}{XS}.md'))
        return files

    def eq_state(self):
        """False when no file of the arm exists (checkpoint, training log, validation
        record, configuration, or a result or summary file in a test or sensitivity
        folder); True when every expected file exists, every run has finished and no
        other file of the arm exists in those places; otherwise refuses."""
        want = self.eq_expected()
        m = REG.PUB_PREFIX + self.fam.eq_stem
        pats = [os.path.join('trained_network', 'FAMILY', f'{m}-*.pth'),
                os.path.join('train_log', f'PUBFAMILY_{self.fam.eq_stem}-*.log'),
                os.path.join('train_log', 'FAMILY', f'valiquality_{m}-*.txt'),
                os.path.join('train_log', 'FAMILY', f'config_{m}-*.json')]
        for st in list(SETS) + list(SENS_WORD):
            d = os.path.join(REG.PUB_RESULTS, st + XS)
            pats += [os.path.join(d, f'Result_greedy+{m}-*'), os.path.join(d, f'summary_{m}-*')]
        found = {x for x in want if os.path.exists(self.p(x))}
        for pat in pats:
            found |= {os.path.relpath(x, self.root) for x in glob.glob(self.p(pat))}
        if not found:
            return False
        missing = [x for x in want if x not in found]
        extra = sorted(found - set(want))
        open_runs = [self.fam.eq_model(s) for s in SEEDS
                     if os.path.exists(self.p(self.fam.eq_paths(s)[1])) and
                     'done in' not in open(self.p(self.fam.eq_paths(s)[1]), errors='replace').read()]
        if missing or extra or open_runs:
            refuse(f'the equal-experience arm {self.fam.eq_stem} is incomplete: {len(found)} of '
                   f'{len(want)} files exist; missing {missing[:4]}{" ..." if len(missing) > 4 else ""}; '
                   f'unexpected {extra[:4]}; not finished {open_runs}. Section 8 needs every file of '
                   f'the five runs, the other sections need none of them')
        return True

    def load_eq_set(self, st):
        """The arm's arrays of one set, with the summary, record-order and bound checks."""
        n = len(self.names[st])
        ev = self.eq_evidence.setdefault(st, {})
        oldest = np.inf
        for s in SEEDS:
            name = self.fam.eq_model(s)
            for r in R8:
                v, t = self._load(os.path.join(
                    self.folder(st), f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'), n, eq=True)
                self.eq_ms[(st, s, r)] = v
                oldest = min(oldest, t)
            self.check_summary(st, f'summary_{REG.PUB_PREFIX}{name}{XS}.md',
                               {r: f'greedy+{REG.PUB_PREFIX}{name}' for r in R8},
                               lambda r, m, s=s: self.eq_ms[(st, s, r)], evidence=ev)
        if not self.evidence[st]['records_newest_mtime'] <= oldest:
            refuse(f'{st}: a descriptor record is newer than a result file of {self.fam.eq_stem}')
        ev['results_oldest_mtime'] = oldest
        self.check_lower_bounds(st, vecs=[self.eq_ms[(st, s, 'N')] for s in SEEDS], evidence=ev)

    def check_eq_sensitivity(self, st):
        """The free check of a sensitivity set for the arm: N and S bit for bit."""
        base = REG.PUB_SENS_SETS[st][0]
        n_cmp, n_arr = 0, 0
        for s in SEEDS:
            for r in ('N', 'S'):
                v, w = self.eq_ms[(st, s, r)], self.eq_ms[(base, s, r)]
                if not np.array_equal(v, w):
                    refuse(f'free check failed: {st} {self.fam.eq_model(s)} {r}: '
                           f'{int((v != w).sum())} of {len(v)} makespans differ from {base}')
                n_cmp += len(v)
                n_arr += 1
        self.eq_evidence[st]['free_check'] = dict(arrays=n_arr, makespans=n_cmp, identical=True)

    def eq_training_record(self, seed):
        """training_record of a pubx-eq run, which must have validated up to update
        PUB_EQ_UPDATES; its configuration must equal that of pubx-v1 of the same seed
        except for the name and max_updates (keys added to the trainer since count as
        equal at their inert default)."""
        _, lp, vp, cp = self.fam.eq_paths(seed)
        rec = self.training_record(None, None, seed, paths=(lp, vp))
        if rec['last_update'] != REG.PUB_EQ_UPDATES:
            refuse(f'{vp}: last validation at update {rec["last_update"]}, want {REG.PUB_EQ_UPDATES}')
        if self.fam.rule:            # the recipe of pubx-eq of the same seed under the mask
            rec['config_differs_in'] = self.rule_config(
                cp, REG.pub_eq_paths(seed)[3],
                want=dict(max_updates=REG.PUB_EQ_UPDATES, family_regimes=','.join(TRAIN),
                          seed_train=seed))
            return rec
        ref_p = f'train_log/FAMILY/config_{REG.PUB_PREFIX}{REG.pub_model("released", None, seed)}.json'
        for x in (ref_p, cp):
            if not os.path.exists(self.p(x)):
                raise REG.MissingResult(f'missing training record: {x}')
        with open(self.p(ref_p)) as f:
            ref = json.load(f)
        with open(self.p(cp)) as f:
            new = json.load(f)
        diff = sorted(k for k in set(ref) | set(new)
                      if ref.get(k, '<absent>') != new.get(k, '<absent>') and not
                      (k in EQ_INERT_NEW and k not in ref and new.get(k) in EQ_INERT_NEW[k]))
        if set(diff) - {'model_suffix', 'max_updates'} or \
                int(new.get('max_updates', -1)) != REG.PUB_EQ_UPDATES or \
                new.get('family_regimes') != ','.join(TRAIN):
            refuse(f'{cp} is not the recipe of {ref_p} with {REG.PUB_EQ_UPDATES} updates: differs in '
                   + str({k: (ref.get(k), new.get(k)) for k in diff}))
        rec['config_differs_in'] = diff
        return rec

    # ---------------------------------------------------------- references (section 9)
    def ref_folder(self, st=REF_SET):
        """Folder of the reference records, relative to the root (p91's default_out)."""
        c = REF_CAMPAIGN
        return P91.default_out(st, c['budget'], c['workers'], c['model'], c['seed'])

    def load_refs(self, st=REF_SET):
        """The reference records of st, read with the readers of
        scripts/p91_public_cpsat_refs.py (read_records, load_refs, record_validity), in the
        sorted-name order of the arrays. A record counts as a reference only when p91 marked
        it valid (its field valid), record_validity still says valid, and it belongs to the
        campaign: its set, combination and instance name its place, and its model, budget,
        workers, solver seed, descriptor seed and rules version are those of the campaign and
        of the record of the test set; a development run never counts. Nothing is refused
        here: a record that does not count is listed with the reason."""
        c = REF_CAMPAIGN
        folder = self.ref_folder(st)
        d = self.p(folder)
        names = self.names[st]
        recs = P91.read_records(d) if os.path.isdir(d) else {}
        arr = P91.load_refs(d, names=names) if os.path.isdir(d) else None
        out = dict(set=st, folder=folder, names=list(names), campaign=dict(c), regimes={},
                   not_counted=[], outside_the_set=[])
        for r in R8:
            rr = recs.get(r, {})
            out['outside_the_set'] += [f'{r}/{k}' for k in sorted(set(rr) - set(names))]
            n = len(names)
            x = dict(present=np.zeros(n, bool), valid=np.zeros(n, bool),
                     objective=np.full(n, np.nan), bound=np.full(n, np.nan),
                     proved=np.zeros(n, bool), status=[None] * n, gap_to_bound=np.full(n, np.nan),
                     solve_s=np.full(n, np.nan), warm=np.full(n, np.nan))
            if arr is not None:
                x['objective'], x['bound'] = arr[r]['objective'].copy(), arr[r]['bound'].copy()
                x['proved'] = arr[r]['proved'].copy()
            for i, nm in enumerate(names):
                rec = rr.get(nm)
                if rec is None:
                    continue
                x['present'][i] = True
                x['status'][i] = rec.get('status')
                x['solve_s'][i] = float(rec.get('solve_time_s', np.nan))
                x['warm'][i] = float(rec.get('warm_makespan', np.nan))
                if rec.get('gap_to_bound') is not None:
                    x['gap_to_bound'][i] = float(rec['gap_to_bound'])
                why = []
                for key, want in (('set', st), ('regime', r), ('instance', nm), ('model', c['model']),
                                  ('workers', c['workers']), ('seed', c['seed']),
                                  ('desc_seed', self.desc_seeds[st][i]),
                                  ('rules_version', RULES_VERSION)):
                    if rec.get(key) != want:
                        why.append(f'{key} {rec.get(key)!r}, want {want!r}')
                if float(rec.get('budget_s', -1)) != c['budget']:
                    why.append(f'budget {rec.get("budget_s")!r} s, want {c["budget"]:g}')
                if rec.get('development_run'):
                    why.append('development run')
                again, reason, _ = P91.record_validity(rec)
                if rec.get('valid') is not True:
                    why.append(f'not marked valid ({rec.get("valid_reason")!r})')
                elif not again:
                    why.append(f'marked valid, record_validity says {reason!r}')
                if arr is not None and arr[r]['valid'][i] != again:
                    why.append('load_refs and record_validity disagree')
                if why:
                    out['not_counted'].append(f'{r}/{nm}: ' + '; '.join(why))
                else:
                    x['valid'][i] = True
            out['regimes'][r] = x
        for r in R8:                 # largest valid bound over the combination and its subsets
            bs = np.vstack([np.where(out['regimes'][q]['valid'], out['regimes'][q]['bound'], np.nan)
                            for q in P91.subsets(r)])
            with np.errstate(all='ignore'):
                out['regimes'][r]['bound_mono'] = np.where(
                    np.isnan(bs).all(0), np.nan, np.nanmax(np.where(np.isnan(bs), -np.inf, bs), axis=0))
        return out


# ================================================================== statistics
VERDICT_WORDS = ('ahead', 'behind', 'not distinguishable')


def superiority_verdict(lo, hi, p_t_ahead, p_boot_ahead, p_t_behind, p_boot_behind,
                        alpha=0.05):
    """Combined verdict of the first-named method against a rule quantity, for
    d = 100 (a - b) / b with a the method and b the rule quantity.

    lo, hi       the hierarchical (1 - alpha) interval of d;
    p_*_ahead    Holm-adjusted one-sided p-values for H1: d < 0 (seed-level t and
                 hierarchical bootstrap), Holm over the combinations of the family;
    p_*_behind   Holm-adjusted one-sided p-values for H1: d > 0, each kind with its
                 own Holm family over the same combinations.
    Returns 'ahead' when hi < 0 and both ahead p-values are below alpha, 'behind'
    when lo > 0 and both behind p-values are below alpha, and 'not distinguishable'
    otherwise. The first two exclude each other because lo <= hi; a NaN p-value never
    meets its condition."""
    if hi < 0 and p_t_ahead < alpha and p_boot_ahead < alpha:
        return VERDICT_WORDS[0]
    if lo > 0 and p_t_behind < alpha and p_boot_behind < alpha:
        return VERDICT_WORDS[1]
    return VERDICT_WORDS[2]


def component(d, B, behind=False):
    """Interval package of one paired comparison d [k, n]. behind=True adds the
    one-sided p-values for H1: d > 0 (first arm slower), p_t_greater from the same
    per-seed means and p_boot_greater from the same hierarchical draws."""
    hb = SF.hier_boot(d, B=B, seed=HIER_SEED, return_draws=True)
    t = SF.seed_t(d)
    out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
               seed_means=t['seed_means'], seed_sd=t['sd'], n=int(d.shape[1]),
               seed_min=float(min(t['seed_means'])), seed_max=float(max(t['seed_means'])),
               ties=float(np.mean(d == 0)) * 100.0, direction=SF.direction(hb['lo'], hb['hi']),
               p_t_less=SF.seed_t_p(d, 'less'), p_boot_less=SF.boot_p_one_sided(hb['draws'], 'less'))
    for mg in MARGINS:
        out[f'v{mg}'] = SF.verdict(hb['lo'], hb['hi'], mg)
        out[f'v{mg}_t'] = SF.verdict(t['lo'], t['hi'], mg)
    if behind:
        out['p_t_greater'] = SF.seed_t_p(d, 'greater')
        out['p_boot_greater'] = SF.boot_p_one_sided(hb['draws'], 'greater')
    return out


def pooled(ds, B):
    """Pooled over combinations [k, n, c], combinations of a base instance together."""
    d3 = np.stack(ds, axis=2)
    hb = SF.hier_boot(d3, B=B, seed=HIER_SEED)
    t = SF.seed_t(d3)
    return dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                seed_means=t['seed_means'], c=d3.shape[2], direction=SF.direction(hb['lo'], hb['hi']),
                **{f'v{mg}': SF.verdict(hb['lo'], hb['hi'], mg) for mg in MARGINS})


def iut_block(comp, families):
    res = {}
    for fam, regs in families.items():
        for mg in MARGINS:
            for claim in ('equivalent', 'non-inferior'):
                holds, failing = SF.iut({r: comp[r][f'v{mg}'] for r in regs}, claim)
                res[f'{fam}|{mg}|{claim}'] = dict(holds=holds, failing=failing,
                                                  components=list(regs), margin=mg)
    return res


def shares(d):
    """Percent of entries shorter / equal / longer (d < 0, = 0, > 0)."""
    d = np.asarray(d)
    return dict(shorter=float(np.mean(d < 0)) * 100, equal=float(np.mean(d == 0)) * 100,
                longer=float(np.mean(d > 0)) * 100)


def seed_shares(d):
    """shares per seed of d [k, n], averaged over the seeds, with min and max."""
    per = [shares(x) for x in d]
    return {key: float(np.mean([p[key] for p in per])) for key in per[0]} | \
        {f'{key}_min': float(min(p[key] for p in per)) for key in per[0]} | \
        {f'{key}_max': float(max(p[key] for p in per)) for key in per[0]}


# ================================================================== analysis
def analyse(I, B):
    pol = lambda st, s, r: I.ms[(st, 'released', None, s, r)]      # noqa: E731
    ctl = lambda st, s, r: I.ms[(st, 'control', None, s, r)]       # noqa: E731
    spec = lambda st, s, r: I.ms[(st, 'specialist', r, s, r)]      # noqa: E731

    def pairs(fa, fb, st, r):
        return np.stack([rel(fa(st, s, r), fb(st, s, r)) for s in SEEDS])

    def best_rule(st, r):
        return np.min(np.stack([I.rule[(st, q, r)] for q in RULES]), axis=0)

    # ---- 5. selection (first, the rule tables use it)
    sel = dict(rule='best validation score of the kept checkpoint (arm_registry.select_seed), '
                    'validation on the first 20 records of pub_SD2_vali_10x5+mix')
    try:
        s_best, recs = REG.select_seed('released', None, seed_list=SEEDS,
                                       record=I.validation_record)
        sel.update(applied=True, seed=int(s_best),
                   records={f's{s}': dict(score=v[0], index=v[1]) for s, v in recs.items()})
    except REG.MissingResult:
        raise
    except SystemExit as e:
        sel.update(applied=False, seed=None, reason=str(e))
    SEL = sel['seed']

    # ---- training records (diagnostics)
    TR = {}
    for arm, combo, _ in I.runs():
        for s in SEEDS:
            TR[(arm, combo, s)] = I.training_record(arm, combo, s)
    vali = {}
    for r in R8:
        sp = [TR[('specialist', r, s)]['kept'][r] for s in SEEDS]
        row = dict(spec_kept=sp, spec_kept_mean=float(np.mean(sp)),
                   spec_kept_update=[TR[('specialist', r, s)]['kept_update'] for s in SEEDS],
                   spec_last_quarter_gain=[TR[('specialist', r, s)]['last_quarter_gain'] for s in SEEDS],
                   spec_score=[TR[('specialist', r, s)]['score'] for s in SEEDS])
        for arm, regs in (('released', TRAIN), ('control', CTL_SEEN)):
            if r in regs:
                v = [TR[(arm, None, s)]['kept'][r] for s in SEEDS]
                g = [100.0 * (a - b) / b for a, b in zip(v, sp)]
                row[f'{arm}_kept'] = v
                row[f'{arm}_minus_spec'] = g
                row[f'{arm}_minus_spec_mean'] = float(np.mean(g))
        vali[r] = row
    for arm in ('released', 'control'):
        vali[arm] = dict(kept_update=[TR[(arm, None, s)]['kept_update'] for s in SEEDS],
                         score=[TR[(arm, None, s)]['score'] for s in SEEDS],
                         last_quarter_gain=[TR[(arm, None, s)]['last_quarter_gain'] for s in SEEDS])

    out = dict(selection=sel, validation=vali, sets={})
    for st in SETS:
        n = len(I.names[st])
        S = dict(role=SETS[st][1], n=n, checks=I.evidence[st])
        # ---- 1. policy vs specialists
        D = {r: pairs(pol, spec, st, r) for r in R8}
        PAR = {r: component(D[r], B) for r in R8}
        fams = {'All': R8, 'Train': TRAIN, 'Held': HELD}
        S['parity'] = dict(components=PAR, iut=iut_block(PAR, fams),
                           pooled={f: pooled([D[r] for r in regs], B) for f, regs in fams.items()})
        # ---- 2. control on the combinations it never saw
        DPC = {r: pairs(pol, ctl, st, r) for r in R8}
        DCS = {r: pairs(ctl, spec, st, r) for r in R8}
        PC = {r: component(DPC[r], B) for r in R8}
        CS = {r: component(DCS[r], B) for r in R8}
        cf = {'Unseen': UNSEEN, 'Pair': PAIR, 'Held': HELD}
        S['control'] = dict(
            policy_vs_control=PC, control_vs_specialist=CS,
            iut_control_vs_specialist=iut_block(CS, cf), iut_policy_vs_control=iut_block(PC, cf),
            pooled_policy_vs_control={f: pooled([DPC[r] for r in regs], B) for f, regs in cf.items()},
            pooled_control_vs_specialist={f: pooled([DCS[r] for r in regs], B)
                                          for f, regs in cf.items()})
        # ---- 3. rules
        RU, BEHIND = {}, {}
        for r in R8:
            best = best_rule(st, r)
            d = np.stack([rel(pol(st, s, r), best) for s in SEEDS])
            c = component(d, B, behind=True)
            row = dict(best=dict(mean=c['mean'], lo=c['lo'], hi=c['hi'], t_lo=c['t_lo'],
                                 t_hi=c['t_hi'], seed_means=c['seed_means'],
                                 p_t=1.0 if np.isnan(c['p_t_less']) else c['p_t_less'],
                                 p_boot=c['p_boot_less'], **seed_shares(d)),
                       rules={})
            BEHIND[r] = dict(p_t=1.0 if np.isnan(c['p_t_greater']) else c['p_t_greater'],
                             p_boot=c['p_boot_greater'])
            if SEL is not None:
                ds = rel(pol(st, SEL, r), best)
                ib = SF.instance_boot(ds, B=B, seed=INST_SEED)
                row['best_selected'] = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'],
                                            direction=SF.direction(ib['lo'], ib['hi']), **shares(ds))
            wins = np.stack([I.rule[(st, q, r)] for q in RULES])
            row['best_rule_share'] = {q: float(np.mean(wins[j] == best)) * 100
                                      for j, q in enumerate(RULES)}
            for q in RULES:
                dq = np.stack([rel(pol(st, s, r), I.rule[(st, q, r)]) for s in SEEDS])
                row['rules'][q] = dict(mean=float(dq.mean()), **seed_shares(dq))
                if SEL is not None:
                    dqs = rel(pol(st, SEL, r), I.rule[(st, q, r)])
                    row['rules'][q]['selected'] = dict(mean=float(dqs.mean()), **shares(dqs))
            RU[r] = row
        for col in ('p_t', 'p_boot'):
            for r, a in zip(R8, SF.holm([RU[r]['best'][col] for r in R8])):
                RU[r]['best'][col + '_holm'] = a
                RU[r]['best'][col + '_holm_ahead'] = bool(a < 0.05)
        for col in ('p_t', 'p_boot'):       # H1: the policy is longer, its own Holm family
            for r, a in zip(R8, SF.holm([BEHIND[r][col] for r in R8])):
                RU[r]['best'][col + '_greater'] = BEHIND[r][col]
                RU[r]['best'][col + '_greater_holm'] = a
        for r in R8:
            x = RU[r]['best']
            x['verdict'] = superiority_verdict(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                                               x['p_t_greater_holm'], x['p_boot_greater_holm'])
        S['rules'] = RU
        # ---- 4. regime-N anchor
        b = I.bk[st]
        anchor = dict(n=n, proved=int(b['proved'].sum()), claimed=int(b['claimed'].sum()),
                      lb_known=int(np.isfinite(b['lb']).sum()), arms={})

        def gaps(v):
            g = rel(v, b['bk'])
            x = dict(gap=float(g.mean()))
            if b['proved'].any():
                x['gap_proved'] = float(g[b['proved']].mean())
            if np.isfinite(b['lb']).all():
                x['gap_lb'] = float(rel(v, b['lb']).mean())
            return x

        for lab, f in (('policy', pol), ('control', ctl), ('specialist N', spec)):
            per = {s: gaps(f(st, s, 'N')) for s in SEEDS}
            gs = [per[s]['gap'] for s in SEEDS]
            anchor['arms'][lab] = dict(per_seed={f's{s}': per[s] for s in SEEDS},
                                       gap=float(np.mean(gs)), sd=float(np.std(gs, ddof=1)),
                                       min=float(min(gs)), max=float(max(gs)),
                                       **{k: float(np.mean([per[s][k] for s in SEEDS]))
                                          for k in per[SEEDS[0]] if k != 'gap'})
        if SEL is not None:
            anchor['arms']['policy selected'] = dict(seed=SEL, **gaps(pol(st, SEL, 'N')))
        for q in RULES:
            anchor['arms'][q] = gaps(I.rule[(st, q, 'N')])
        anchor['arms']['best rule'] = gaps(best_rule(st, 'N'))
        S['anchor'] = anchor
        # ---- 6. per-seed gaps to the per-instance best rule
        PS = {}
        for r in R8:
            best = best_rule(st, r)
            for lab, f in (('policy', pol), ('control', ctl), ('specialist', spec)):
                g = [float(rel(f(st, s, r), best).mean()) for s in SEEDS]
                PS[f'{r}|{lab}'] = dict(per_seed=g, mean=float(np.mean(g)),
                                        sd=float(np.std(g, ddof=1)), min=min(g), max=max(g))
        S['per_seed'] = PS
        # ---- diagnostics: rule effects and trivial instances
        bN = best_rule(st, 'N')
        pN = np.mean([pol(st, s, 'N') for s in SEEDS], axis=0)
        eff, triv = {}, {}
        for r in R8:
            pR = np.mean([pol(st, s, r) for s in SEEDS], axis=0)
            eff[r] = dict(best_rule=float(np.mean(best_rule(st, r) / bN)),
                          policy=float(np.mean(pR / pN)))
            allv = np.stack([pol(st, s, r) for s in SEEDS] + [spec(st, s, r) for s in SEEDS]
                            + [best_rule(st, r)])
            same = np.all(allv == allv[0], axis=0)
            triv[r] = dict(count=int(same.sum()),
                           names=[I.names[st][i] for i in np.flatnonzero(same)][:20])
            spm = np.mean([spec(st, s, r) for s in SEEDS], axis=0)
            pm = np.mean([pol(st, s, r) for s in SEEDS], axis=0)
            triv[r]['corr_policy_specialist'] = float(np.corrcoef(pm, spm)[0, 1]) \
                if np.std(pm) > 0 and np.std(spm) > 0 else 1.0
        S['rule_effect'] = eff
        S['trivial'] = triv
        out['sets'][st] = S

    # ---- 7. sensitivity sets: combinations with L, beside the test set
    out['sensitivity'] = {}
    for st in SENS_WORD:
        base, _, change = REG.PUB_SENS_SETS[st]
        X = dict(base=base, change=change, n=len(I.names[st]), checks=I.evidence[st], combos={})
        D = {}
        for r in L_COMBOS:
            row = {}
            for lab, ss in (('half', st), ('one', base)):
                best = best_rule(ss, r)
                d = dict(par=pairs(pol, spec, ss, r), polctl=pairs(pol, ctl, ss, r),
                         ctlspec=pairs(ctl, spec, ss, r),
                         rule=np.stack([rel(pol(ss, s, r), best) for s in SEEDS]))
                D[(lab, r)] = d
                row[lab] = dict(
                    policy_vs_specialist=component(d['par'], B),
                    policy_vs_control=component(d['polctl'], B),
                    control_vs_specialist=component(d['ctlspec'], B),
                    policy_vs_best_rule=dict(component(d['rule'], B), **seed_shares(d['rule'])),
                    rules={q: dict(mean=float(dq.mean()), **seed_shares(dq)) for q, dq in
                           ((q, np.stack([rel(pol(ss, s, r), I.rule[(ss, q, r)]) for s in SEEDS]))
                            for q in RULES)},
                    rule_effect=dict(best_rule=float(np.mean(best / best_rule(ss, 'N'))),
                                     policy=float(np.mean(
                                         np.mean([pol(ss, s, r) for s in SEEDS], axis=0) /
                                         np.mean([pol(ss, s, 'N') for s in SEEDS], axis=0)))))
            # paired change between the scales (same seeds, same base instances)
            row['change'] = {k: component(D[('half', r)][k] - D[('one', r)][k], B)
                             for k in ('par', 'polctl', 'rule')}
            X['combos'][r] = row
        X['pooled'] = {lab: {k: pooled([D[(lab, r)][k] for r in L_COMBOS], B)
                             for k in ('par', 'polctl', 'ctlspec', 'rule')}
                       for lab in ('half', 'one')}
        X['iut'] = {lab: iut_block({r: X['combos'][r][lab]['policy_vs_specialist']
                                    for r in L_COMBOS}, {'L': L_COMBOS})
                    for lab in ('half', 'one')}
        for r in L_COMBOS:              # the scale-1.0 rows are those of section 1
            a = X['combos'][r]['one']['policy_vs_specialist']
            b = out['sets'][base]['parity']['components'][r]
            if (a['mean'], a['lo'], a['hi']) != (b['mean'], b['lo'], b['hi']):
                refuse(f'{st}: the scale-1.0 parity of {r} does not reproduce section 1')
        out['sensitivity'][st] = X
    return out


def analyse_eq(I, B):
    """Section 8: the equal-experience policy, seed s against seed s, against the
    specialists and against the released policy, on every test set."""
    eqp = lambda st, s, r: I.eq_ms[(st, s, r)]                      # noqa: E731
    pol = lambda st, s, r: I.ms[(st, 'released', None, s, r)]      # noqa: E731
    spec = lambda st, s, r: I.ms[(st, 'specialist', r, s, r)]      # noqa: E731

    def pairs(fa, fb, st, r):
        return np.stack([rel(fa(st, s, r), fb(st, s, r)) for s in SEEDS])

    fams = {'All': R8, 'Train': TRAIN, 'Held': HELD}
    # ---- training records: validation at the kept checkpoints against the specialists
    TR = {s: I.eq_training_record(s) for s in SEEDS}
    vali = {}
    for r in TRAIN:
        sp = [I.training_record('specialist', r, s)['kept'][r] for s in SEEDS]
        e = [TR[s]['kept'][r] for s in SEEDS]
        p = [I.training_record('released', None, s)['kept'][r] for s in SEEDS]
        ge = [100.0 * (a - b) / b for a, b in zip(e, sp)]
        gp = [100.0 * (a - b) / b for a, b in zip(p, sp)]
        vali[r] = dict(eq_kept=e, spec_kept=sp, eq_minus_spec=ge, eq_minus_spec_mean=float(np.mean(ge)),
                       released_minus_spec_mean=float(np.mean(gp)))
    vali['eq'] = dict(kept_update=[TR[s]['kept_update'] for s in SEEDS],
                      kept_fraction=[TR[s]['kept_fraction'] for s in SEEDS],
                      score=[TR[s]['score'] for s in SEEDS],
                      last_quarter_gain=[TR[s]['last_quarter_gain'] for s in SEEDS])
    out = dict(arm=I.fam.eq_stem, updates=REG.PUB_EQ_UPDATES, seeds=list(SEEDS),
               regimes=list(TRAIN), headline=EQ_HEADLINE,
               config_differs_in={f's{s}': TR[s]['config_differs_in'] for s in SEEDS},
               checks=I.eq_evidence, files_read=I.n_files_eq, validation=vali, sets={})
    if I.fam.rule:
        out['config_reference'] = f'{REG.PUB_EQ_STEM} of the same seed'
    for st in SETS:
        DP = {r: pairs(eqp, spec, st, r) for r in R8}
        DR = {r: pairs(eqp, pol, st, r) for r in R8}
        PAR = {r: component(DP[r], B) for r in R8}
        S = dict(role=SETS[st][1], n=len(I.names[st]),
                 parity=dict(components=PAR, iut=iut_block(PAR, fams),
                             pooled={f: pooled([DP[r] for r in regs], B) for f, regs in fams.items()}),
                 vs_policy=dict(components={r: component(DR[r], B) for r in R8},
                                pooled={f: pooled([DR[r] for r in regs], B) for f, regs in fams.items()}))
        # regime N: gap to the best-known value, as section 4 does for the released policy
        bk = I.bk[st]['bk']
        gs = [float(rel(eqp(st, s, 'N'), bk).mean()) for s in SEEDS]
        S['anchor'] = dict(per_seed={f's{s}': g for s, g in zip(SEEDS, gs)}, gap=float(np.mean(gs)),
                           sd=float(np.std(gs, ddof=1)), min=float(min(gs)), max=float(max(gs)))
        out['sets'][st] = S
    return out


# ================================================================== section 9: references
def ref_state(R):
    """Counts of the reference records per combination and whether the section can be
    computed: every reference of the eight combinations present and counted."""
    n = len(R['names'])
    per = {r: dict(present=int(R['regimes'][r]['present'].sum()),
                   valid=int(R['regimes'][r]['valid'].sum()),
                   proved=int((R['regimes'][r]['valid'] & R['regimes'][r]['proved']).sum()))
           for r in R8}
    tot = {k: sum(v[k] for v in per.values()) for k in ('present', 'valid', 'proved')}
    complete = [r for r in R8 if per[r]['valid'] == n]
    reason = None if len(complete) == len(R8) else (
        f'{tot["valid"]} of {n * len(R8)} references of {R["set"]} are valid ({tot["present"]} '
        f'present, {len(R["not_counted"])} not counted); valid per combination '
        + ', '.join(f'{r} {per[r]["valid"]}' for r in R8)
        + f' of {n}; the section needs all {n * len(R8)}')
    return dict(n=n, expected=n * len(R8), per=per, total=tot, complete=complete,
                complete_all=reason is None, reason=reason)


def ref_problems(I, R, arrays=True):
    """Contradictions between the counted references, the best-known table (combination N)
    and, with arrays, every stored makespan of the set; each means that the references and
    the evaluations do not describe the same problem:
      * N: a reference below a best-known value that the table marks as proved optimal;
      * N: a reference bound above a best-known makespan;
      * every combination: a stored makespan of a run or a rule below the largest counted
        bound of the combination and its sub-combinations (a rule switched on only delays).
    Returns the list of problems, each naming the instance (empty = consistent)."""
    st, names = R['set'], R['names']
    b, x = I.bk[st], R['regimes']['N']
    out = []
    for i in np.flatnonzero(x['valid']):
        if b['proved'][i] and x['objective'][i] < b['bk'][i] - REF_EPS:
            out.append(f'{names[i]} N: reference {x["objective"][i]:g} below the best-known value '
                       f'{b["bk"][i]:g} that {I.table(st)} marks as proved optimal')
        if np.isfinite(x['bound'][i]) and x['bound'][i] > b['bk'][i] + REF_EPS:
            out.append(f'{names[i]} N: reference bound {x["bound"][i]:g} above the best-known '
                       f'makespan {b["bk"][i]:g}')
    if arrays:
        for r in R8:
            lb = R['regimes'][r]['bound_mono']
            known = np.isfinite(lb)
            vecs = [(I.fam.model(a, c, s), v) for (s_, a, c, s, rr), v in I.ms.items()
                    if s_ == st and rr == r] + [(q, I.rule[(st, q, r)]) for q in RULES]
            for lab, v in vecs:
                for i in np.flatnonzero(known & (v < lb - REF_EPS))[:3]:
                    out.append(f'{names[i]} {r}: stored makespan {v[i]:g} of {lab} below the '
                               f'reference bound {lb[i]:g}')
    return out


def ref_best_known(I, R):
    """Combination N: counted references below, equal to and above the best-known value."""
    st, names = R['set'], R['names']
    b, x = I.bk[st], R['regimes']['N']
    ok, o, v = x['valid'], x['objective'], b['bk']
    below, above = ok & (o < v - REF_EPS), ok & (o > v + REF_EPS)
    equal = ok & ~below & ~above

    def cases(mask):
        return [dict(instance=names[i], reference=float(o[i]), best_known=float(v[i]),
                     table=b['status'][i], reference_status=x['status'][i]) for i in np.flatnonzero(mask)]
    return dict(counted=int(ok.sum()), below=int(below.sum()), equal=int(equal.sum()),
                above=int(above.sum()), table_proved=int(b['proved'].sum()),
                proved_here_and_in_table=int((ok & x['proved'] & b['proved']).sum()),
                above_where_table_proved=int((above & b['proved']).sum()),
                below_cases=cases(below), above_cases=cases(above))


def ref_gap(d, B):
    """Gap to the reference of a trained arm, d [k, n] or [k, n, c]: crossed hierarchical
    bootstrap (fresh default_rng(HIER_SEED)), interval and direction."""
    hb = SF.hier_boot(d, B=B, seed=HIER_SEED)
    return dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], direction=SF.direction(hb['lo'], hb['hi']),
                seed_means=SF.seed_t(d)['seed_means'])


def ref_inst_gap(d, B):
    """Gap to the reference of a rule quantity, d [n] or [n, c]: instance-level cluster
    bootstrap (fresh default_rng(INST_SEED)), interval and direction."""
    ib = SF.instance_boot(d, B=B, seed=INST_SEED)
    return dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], direction=SF.direction(ib['lo'], ib['hi']))


def analyse_refs(I, R, B, combos):
    """Section 9 on the combinations combos (every reference counted): d = 100 (method -
    reference) / reference per instance, per combination and pooled over combos (a base
    instance with all its combinations), interval and direction only; reference counts; the
    policy's schedules equal to, longer and shorter than the reference; records that are
    unexpected without being wrong."""
    st, names, budget = R['set'], R['names'], R['campaign']['budget']
    arm = {'policy': lambda s, r: I.ms[(st, 'released', None, s, r)],
           'control': lambda s, r: I.ms[(st, 'control', None, s, r)],
           'specialist': lambda s, r: I.ms[(st, 'specialist', r, s, r)]}
    rows, D, Q, shorter = {}, {}, {}, []
    odd = dict(worse_than_warm_start=[], best_rule_shorter=[], ended_before_limit_unproved=[],
               other_status=[])
    for r in combos:
        x = R['regimes'][r]
        if not x['valid'].all():
            raise ValueError(f'section 9: combination {r} is not complete')
        ref, bnd = x['objective'], x['bound']
        lim = ~x['proved'] & (ref > bnd + REF_EPS)
        gap = 100.0 * (ref - bnd) / ref
        st_count = {}
        for v in x['status']:
            st_count[v] = st_count.get(v, 0) + 1
        row = dict(n=len(ref), proved=int(x['proved'].sum()), at_limit=int(lim.sum()),
                   at_limit_gap_mean=float(gap[lim].mean()) if lim.any() else None,
                   at_limit_gap_max=float(gap[lim].max()) if lim.any() else None,
                   unproved_without_gap=int((~x['proved'] & ~lim).sum()), status=st_count,
                   reference_mean=float(ref.mean()), arms={}, rules={})
        for lab, f in arm.items():
            d = np.stack([rel(f(s, r), ref) for s in SEEDS])
            D[(lab, r)] = d
            row['arms'][lab] = dict(ref_gap(d, B), **seed_shares(d))
        best = np.min(np.stack([I.rule[(st, q, r)] for q in RULES]), axis=0)
        for q in ('best',) + RULES:
            dq = rel(best if q == 'best' else I.rule[(st, q, r)], ref)
            Q[(q, r)] = dq
            row['rules'][q] = dict(ref_inst_gap(dq, B), **shares(dq))
        for s in SEEDS:
            p = arm['policy'](s, r)
            shorter += [dict(combination=r, instance=names[i], seed=s, policy=float(p[i]),
                             reference=float(ref[i]), bound=float(bnd[i]), status=x['status'][i])
                        for i in np.flatnonzero(p < ref - REF_EPS)]
        for key, mask in (('worse_than_warm_start', ref > x['warm'] + REF_EPS),
                          ('best_rule_shorter', best < ref - REF_EPS),
                          ('ended_before_limit_unproved', ~x['proved'] & (x['solve_s'] < budget - 1.0)),
                          ('other_status', np.array([v not in ('OPTIMAL', 'FEASIBLE')
                                                     for v in x['status']]))):
            odd[key] += [dict(combination=r, instance=names[i], reference=float(ref[i]),
                              warm_start=float(x['warm'][i]), best_rule=float(best[i]),
                              status=x['status'][i], solve_s=float(x['solve_s'][i]))
                         for i in np.flatnonzero(mask)]
        rows[r] = row
    pooled = {}
    if combos:
        for lab in arm:
            d3 = np.stack([D[(lab, r)] for r in combos], axis=2)
            pooled[lab] = dict(ref_gap(d3, B), c=len(combos),
                               **seed_shares(d3.reshape(d3.shape[0], -1)))
        for q in ('best',) + RULES:
            cells = np.stack([Q[(q, r)] for r in combos], axis=1)
            pooled[q] = dict(ref_inst_gap(cells, B), c=len(combos), **shares(cells))
    return dict(combinations=list(combos), rows=rows, pooled=pooled,
                proved=sum(rows[r]['proved'] for r in combos),
                at_limit=sum(rows[r]['at_limit'] for r in combos),
                policy_shorter=shorter, policy_shorter_n=len(shorter), unexpected=odd)


def section_refs(I, R, B, combos=None):
    """Section 9: the state of the references, the regime-N comparison with the best-known
    values, and the gaps on combos (default: all eight when every reference is counted,
    none otherwise; the partial reading passes the complete combinations)."""
    state = ref_state(R)
    if combos is None:
        combos = list(R8) if state['complete_all'] else []
    out = dict(set=R['set'], folder=R['folder'], campaign=R['campaign'], state=state,
               computed=bool(combos) and list(combos) == list(R8),
               reference=('the schedule of every record of scripts/p91_public_cpsat_refs.py that '
                          'p91 marks valid: CP-SAT, compact model, warm-started from the best of '
                          'the four rules'),
               difference='d = 100 (method - reference) / reference per instance; interval and '
                          'direction only (a gap to a reference is not a comparison of two '
                          'methods under test)',
               best_known_N=ref_best_known(I, R), not_counted_n=len(R['not_counted']),
               not_counted=R['not_counted'][:50], outside_the_set=R['outside_the_set'][:50])
    if combos:
        out.update(analyse_refs(I, R, B, combos))
    return out


def reference_summary(root=REPO, st=REF_SET):
    """The state of the references for the driver's dry run, without the arrays: counts per
    combination and the regime-N problems (the stored makespans are checked by p89 itself)."""
    I = Inputs(root)
    I.check_records(st)
    R = I.load_refs(st)
    return dict(folder=R['folder'], state=ref_state(R), problems=ref_problems(I, R, arrays=False),
                not_counted=R['not_counted'], best_known_N=ref_best_known(I, R))


# ================================================================== output
def f2(x):
    """Signed two decimals of the unrounded value, so a printed bound keeps its sign: a
    negative value that rounds to zero prints -0.00, a positive one +0.00, an exact zero
    (also -0.0) +0.00. Verdicts and directions use the unrounded values, never this text."""
    x = float(x)
    return f'{0.0 if x == 0 else x:+.2f}'


def u1(x):
    return f'{float(x):.1f}'


def tex(v):
    return SF.TEX[v]


def comp_lines(L, name, c, what):
    """Macros of one paired comparison: mean, hierarchical CI, seed-t CI, verdicts, direction."""
    L.append(f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
             f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
             f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}'
             f'\\newcommand{{\\{name}TLo}}{{{f2(c["t_lo"])}}}'
             f'\\newcommand{{\\{name}THi}}{{{f2(c["t_hi"])}}}'
             + ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}' for mg in MARGINS)
             + f'\\newcommand{{\\{name}Dir}}{{{tex(c["direction"])}}}'
             f'  % {what}: mean, hierarchical 95% CI, seed-t CI, verdicts at 0.5 and 1, direction')


def pool_lines(L, name, x, what):
    """Macros of one pooled comparison: mean, hierarchical CI, verdicts."""
    L.append(f'\\newcommand{{\\{name}}}{{{f2(x["mean"])}}}'
             f'\\newcommand{{\\{name}Lo}}{{{f2(x["lo"])}}}'
             f'\\newcommand{{\\{name}Hi}}{{{f2(x["hi"])}}}'
             + ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}' for mg in MARGINS)
             + f'  % {what}, pooled over {x["c"]} combinations (base-instance clusters)')


def iut_lines(L, name, block, fams):
    """Macros of the joint claims (1 = holds) per family, margin and claim."""
    for fam in fams:
        for mg in MARGINS:
            for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                x = block[f'{fam}|{mg}|{claim}']
                L.append(f'\\newcommand{{\\{name}{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                         f'  % IUT, {claim} on {",".join(x["components"])} at +-{mg}%: '
                         f'failing {",".join(x["failing"]) or "none"}')


def macros(res, B):
    L = ['% Generated by scripts/p89_public_family.py -- do not hand-edit',
         f'% Public-instance family, exact mask, training seeds {SEEDS[0]}-{SEEDS[-1]}; B = {B}; '
         f'hierarchical intervals fresh default_rng({HIER_SEED}), selected-seed intervals fresh '
         f'default_rng({INST_SEED}).',
         '% Gaps in %: first arm minus second over second (positive = first arm slower); '
         'gaps to best known: arm minus reference over reference.',
         f'\\newcommand{{\\PubSeedN}}{{{len(SEEDS)}}}  % training seeds per arm',
         f'\\newcommand{{\\PubBoot}}{{{B}}}  % bootstrap replicates']
    sel = res['selection']
    if sel['applied']:
        L.append(f'\\newcommand{{\\PubSelSeed}}{{{sel["seed"]}}}'
                 f'\\newcommand{{\\PubSelScore}}{{{sel["records"]["s%d" % sel["seed"]]["score"]:.4f}}}'
                 '  % selected policy seed and its validation score ('
                 + ', '.join(f'{k} {v["score"]:.5f}' for k, v in sel['records'].items()) + ')')
    for r in TRAIN:
        L.append(f'\\newcommand{{\\PubVali{r}}}{{{f2(res["validation"][r]["released_minus_spec_mean"])}}}'
                 f'  % validation, kept checkpoints: policy minus specialist on {r}, mean of five seeds')

    def comp(name, c, what):
        comp_lines(L, name, c, what)

    def pool(name, x, what):
        pool_lines(L, name, x, what)

    def iut(name, block, fams):
        iut_lines(L, name, block, fams)

    for st, S in res['sets'].items():
        w = SETS[st][0]
        L.append(f'\\newcommand{{\\Pub{w}N}}{{{S["n"]}}}  % {st}: base instances ({S["role"]})')
        P = S['parity']
        for r in R8:
            c = P['components'][r]
            comp(f'Pub{w}Par{r}', c, f'{st}, policy vs specialist, {r}')
            L.append(f'\\newcommand{{\\Pub{w}Par{r}SeedMin}}{{{f2(c["seed_min"])}}}'
                     f'\\newcommand{{\\Pub{w}Par{r}SeedMax}}{{{f2(c["seed_max"])}}}'
                     f'\\newcommand{{\\Pub{w}Par{r}Tie}}{{{u1(c["ties"])}}}'
                     f'  % {st}, {r}: smallest and largest per-seed mean, % of identical makespans')
        for fam in ('All', 'Train', 'Held'):
            pool(f'Pub{w}ParPool{fam}', P['pooled'][fam], f'{st}, policy vs specialist, {fam}')
        iut(f'Pub{w}ParIut', P['iut'], ('All', 'Train', 'Held'))
        C = S['control']
        for r in UNSEEN:
            comp(f'Pub{w}PolCtl{r}', C['policy_vs_control'][r], f'{st}, policy vs control, {r}')
            comp(f'Pub{w}CtlSpec{r}', C['control_vs_specialist'][r],
                 f'{st}, control vs specialist, {r}')
        for fam in ('Unseen', 'Pair', 'Held'):
            pool(f'Pub{w}PolCtlPool{fam}', C['pooled_policy_vs_control'][fam],
                 f'{st}, policy vs control, {fam}')
            pool(f'Pub{w}CtlSpecPool{fam}', C['pooled_control_vs_specialist'][fam],
                 f'{st}, control vs specialist, {fam}')
        iut(f'Pub{w}CtlSpecIut', C['iut_control_vs_specialist'], ('Unseen', 'Pair', 'Held'))
        for r in R8:
            x = S['rules'][r]['best']
            L.append(f'\\newcommand{{\\Pub{w}Rule{r}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\Pub{w}Rule{r}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\Pub{w}Rule{r}Hi}}{{{f2(x["hi"])}}}'
                     f'\\newcommand{{\\Pub{w}Rule{r}Short}}{{{u1(x["shorter"])}}}'
                     f'\\newcommand{{\\Pub{w}Rule{r}Long}}{{{u1(x["longer"])}}}'
                     f'\\newcommand{{\\Pub{w}Rule{r}PHolm}}{{{x["p_t_holm"]:.2g}}}'
                     f'  % {st}, {r}: policy (five seeds) vs per-instance best rule, mean, hierarchical '
                     f'95% CI, % of instances shorter and longer (seed average), Holm one-sided seed-t p')
            if 'best_selected' in S['rules'][r]:
                y = S['rules'][r]['best_selected']
                L.append(f'\\newcommand{{\\Pub{w}RuleSel{r}}}{{{f2(y["mean"])}}}'
                         f'\\newcommand{{\\Pub{w}RuleSel{r}Lo}}{{{f2(y["lo"])}}}'
                         f'\\newcommand{{\\Pub{w}RuleSel{r}Hi}}{{{f2(y["hi"])}}}'
                         f'\\newcommand{{\\Pub{w}RuleSel{r}Short}}{{{u1(y["shorter"])}}}'
                         f'  % {st}, {r}: selected seed vs per-instance best rule, mean, '
                         f'instance-level 95% CI, % of instances shorter')
        L.append(f'\\newcommand{{\\Pub{w}RuleHolmN}}'
                 f'{{{sum(S["rules"][r]["best"]["p_t_holm_ahead"] for r in R8)}}}'
                 f'\\newcommand{{\\Pub{w}RuleHolmBootN}}'
                 f'{{{sum(S["rules"][r]["best"]["p_boot_holm_ahead"] for r in R8)}}}'
                 f'  % {st}: combinations (of 8) with the policy ahead of the best rule after Holm, '
                 f'seed-t and hierarchical bootstrap')
        A = S['anchor']
        for lab, key in (('Pol', 'policy'), ('PolSel', 'policy selected'), ('Ctl', 'control'),
                         ('Spec', 'specialist N'), ('Fifo', 'FIFO'), ('Mor', 'MOR'),
                         ('Spt', 'SPT'), ('Mwkr', 'MWKR'), ('Pool', 'best rule')):
            if key in A['arms']:
                L.append(f'\\newcommand{{\\Pub{w}Bk{lab}}}{{{f2(A["arms"][key]["gap"])}}}'
                         f'  % {st}, regime N: {key}, mean gap to best known')
        L.append(f'\\newcommand{{\\Pub{w}BkProvedN}}{{{A["proved"]}}}'
                 f'  % {st}: best-known values proved optimal (of {A["n"]})')
    for st, X in res.get('sensitivity', {}).items():
        w, base = SENS_WORD[st], X['base']
        what = f'{st} ({X["change"]}) vs {base}'
        fc = X['checks']['free_check']
        L.append(f'\\newcommand{{\\Pub{w}N}}{{{X["n"]}}}'
                 f'\\newcommand{{\\Pub{w}FreeArrays}}{{{fc["arrays"]}}}'
                 f'\\newcommand{{\\Pub{w}FreeMakespans}}{{{fc["makespans"]}}}'
                 f'  % {what}: base instances; N and S arrays and makespans equal bit for bit')
        for r in L_COMBOS:
            C = X['combos'][r]
            for lab, pre in (('half', ''), ('one', 'Base')):
                for key, nm in (('policy_vs_specialist', 'Par'), ('policy_vs_control', 'PolCtl'),
                                ('control_vs_specialist', 'CtlSpec'), ('policy_vs_best_rule', 'Rule')):
                    comp(f'Pub{w}{pre}{nm}{r}', C[lab][key],
                         f'{what}, {"lag scale 0.5" if lab == "half" else "scale 1.0"}, {key}, {r}')
                x = C[lab]['policy_vs_best_rule']
                L.append(f'\\newcommand{{\\Pub{w}{pre}Rule{r}Short}}{{{u1(x["shorter"])}}}'
                         f'\\newcommand{{\\Pub{w}{pre}Rule{r}Long}}{{{u1(x["longer"])}}}'
                         + ''.join(f'\\newcommand{{\\Pub{w}{pre}Vs{q}{r}}}{{{f2(C[lab]["rules"][q]["mean"])}}}'
                                   for q in RULES)
                         + f'\\newcommand{{\\Pub{w}{pre}Effect{r}}}'
                         f'{{{C[lab]["rule_effect"]["best_rule"]:.2f}}}'
                         f'  % {what}, {lab}: % of instances shorter and longer than the best rule '
                         f'(seed average), mean d vs each rule, best-rule {r}/N makespan ratio')
            for key, nm in (('par', 'Par'), ('polctl', 'PolCtl'), ('rule', 'Rule')):
                comp(f'Pub{w}Diff{nm}{r}', C['change'][key],
                     f'{what}, {r}: paired change of d, lag scale 0.5 minus 1.0, {key}')
        for lab, pre in (('half', ''), ('one', 'Base')):
            for key, nm in (('par', 'Par'), ('polctl', 'PolCtl'), ('ctlspec', 'CtlSpec'),
                            ('rule', 'Rule')):
                pool(f'Pub{w}{pre}{nm}PoolL', X['pooled'][lab][key], f'{what}, {lab}, {key}')
            iut(f'Pub{w}{pre}ParIut', X['iut'][lab], ('L',))
    return L


def macros_eq(E):
    """Section 8 macros, all named PubEq...: per combination and the joint claims on
    the headline set, regime N alone on the other sets, pooled comparisons and the
    regime-N gap on every test set."""
    arm = f'equal-experience policy ({E["arm"]}, {E["updates"]} updates)'
    L = [f'% Section 8: the {arm}, seed s against seed s; same estimators and streams as above.',
         f'\\newcommand{{\\PubEqUpdates}}{{{E["updates"]}}}'
         f'  % updates of the {arm} (released policy 2000, specialists 1000)']
    for r in TRAIN:
        L.append(f'\\newcommand{{\\PubEqVali{r}}}{{{f2(E["validation"][r]["eq_minus_spec_mean"])}}}'
                 f'  % validation, kept checkpoints: {arm} minus specialist on {r}, mean of five seeds')
    fams = ('All', 'Train', 'Held')
    for st, S in E['sets'].items():
        w = SETS[st][0]
        if st == E['headline']:
            for r in R8:
                c = S['parity']['components'][r]
                comp_lines(L, f'PubEq{w}Par{r}', c, f'{st}, {arm} vs specialist, {r}')
                L.append(f'\\newcommand{{\\PubEq{w}Par{r}SeedMin}}{{{f2(c["seed_min"])}}}'
                         f'\\newcommand{{\\PubEq{w}Par{r}SeedMax}}{{{f2(c["seed_max"])}}}'
                         f'\\newcommand{{\\PubEq{w}Par{r}Tie}}{{{u1(c["ties"])}}}'
                         f'  % {st}, {r}: smallest and largest per-seed mean, % of identical makespans')
            iut_lines(L, f'PubEq{w}ParIut', S['parity']['iut'], fams)
            for r in R8:
                comp_lines(L, f'PubEq{w}VsPol{r}', S['vs_policy']['components'][r],
                           f'{st}, {arm} vs released policy, {r}')
        else:
            # Regime N alone on the other sets: the one combination where the policy is
            # behind its specialist, so the text can state what the longer training changes.
            comp_lines(L, f'PubEq{w}ParN', S['parity']['components']['N'],
                       f'{st}, {arm} vs specialist, N')
            comp_lines(L, f'PubEq{w}VsPolN', S['vs_policy']['components']['N'],
                       f'{st}, {arm} vs released policy, N')
        for fam in fams:
            pool_lines(L, f'PubEq{w}ParPool{fam}', S['parity']['pooled'][fam],
                       f'{st}, {arm} vs specialist, {fam}')
            pool_lines(L, f'PubEq{w}VsPolPool{fam}', S['vs_policy']['pooled'][fam],
                       f'{st}, {arm} vs released policy, {fam}')
        L.append(f'\\newcommand{{\\PubEq{w}BkPol}}{{{f2(S["anchor"]["gap"])}}}'
                 f'  % {st}, regime N: {arm}, mean gap to best known (mean of five seeds)')
    return L


def macros_verdict(res):
    """Combined verdict of the policy against the per-instance best rule
    (superiority_verdict) per test set and combination, all named Pub<set>Rule...Verdict...,
    with the number of combinations ahead and behind per set."""
    L = ['% Combined verdict of the policy (five seeds) against the per-instance best rule: '
         'ahead = hierarchical 95% CI below zero and both Holm-adjusted one-sided p-values '
         '(seed-level t, bootstrap; H1 policy shorter) below 0.05; behind = CI above zero and '
         'both Holm-adjusted one-sided p-values for H1 policy longer (own Holm family over the '
         'eight combinations) below 0.05; not distinguishable otherwise.']
    for st, S in res['sets'].items():
        w = SETS[st][0]
        for r in R8:
            x = S['rules'][r]['best']
            L.append(f'\\newcommand{{\\Pub{w}Rule{r}Verdict}}{{{x["verdict"]}}}'
                     f'  % {st}, {r}: policy vs per-instance best rule, CI [{x["lo"]:+.2f}, '
                     f'{x["hi"]:+.2f}]; Holm p shorter {x["p_t_holm"]:.2g} (t), '
                     f'{x["p_boot_holm"]:.2g} (bootstrap); Holm p longer '
                     f'{x["p_t_greater_holm"]:.2g} (t), {x["p_boot_greater_holm"]:.2g} (bootstrap)')
        vs = [S['rules'][r]['best']['verdict'] for r in R8]
        L.append(f'\\newcommand{{\\Pub{w}RuleVerdictAheadN}}{{{vs.count(VERDICT_WORDS[0])}}}'
                 f'\\newcommand{{\\Pub{w}RuleVerdictBehindN}}{{{vs.count(VERDICT_WORDS[1])}}}'
                 f'  % {st}: combinations (of 8) with the combined verdict ahead, behind')
    return L


def dir_line(L, name, x, what):
    """<name>Dir of one pooled row x (dict with unrounded lo and hi)."""
    L.append(f'\\newcommand{{\\{name}Dir}}{{{tex(SF.direction(x["lo"], x["hi"]))}}}'
             f'  % {what}: direction of the interval [{x["lo"]:+.4f}, {x["hi"]:+.4f}]')


def macros_pool_dir(res):
    """Direction macros of the pooled rows of sections 1, 2 and 7."""
    L = ['% Direction (ahead / behind / neither) of every pooled row above, from the '
         'unrounded interval ends: ahead = interval below zero, behind = above zero.']
    for st, S in res['sets'].items():
        w = SETS[st][0]
        for fam in ('All', 'Train', 'Held'):
            dir_line(L, f'Pub{w}ParPool{fam}', S['parity']['pooled'][fam],
                     f'{st}, policy vs specialist, {fam}')
        C = S['control']
        for fam in ('Unseen', 'Pair', 'Held'):
            dir_line(L, f'Pub{w}PolCtlPool{fam}', C['pooled_policy_vs_control'][fam],
                     f'{st}, policy vs control, {fam}')
            dir_line(L, f'Pub{w}CtlSpecPool{fam}', C['pooled_control_vs_specialist'][fam],
                     f'{st}, control vs specialist, {fam}')
    for st, X in res.get('sensitivity', {}).items():
        w = SENS_WORD[st]
        for lab, pre in (('half', ''), ('one', 'Base')):
            for key, nm in (('par', 'Par'), ('polctl', 'PolCtl'), ('ctlspec', 'CtlSpec'),
                            ('rule', 'Rule')):
                dir_line(L, f'Pub{w}{pre}{nm}PoolL', X['pooled'][lab][key],
                         f'{st} vs {X["base"]}, {lab}, {key}, pooled over the combinations with L')
    return L


def macros_eq_pool_dir(E):
    """Direction macros of the pooled rows of section 8, all named PubEq...Dir."""
    L = ['% Section 8: direction (ahead / behind / neither) of its pooled rows, from the '
         'unrounded interval ends.']
    for st, S in E['sets'].items():
        w = SETS[st][0]
        for fam in ('All', 'Train', 'Held'):
            dir_line(L, f'PubEq{w}ParPool{fam}', S['parity']['pooled'][fam],
                     f'{st}, equal-experience policy vs specialist, {fam}')
            dir_line(L, f'PubEq{w}VsPolPool{fam}', S['vs_policy']['pooled'][fam],
                     f'{st}, equal-experience policy vs released policy, {fam}')
    return L


def ref_line(L, name, x, what):
    """<name>, <name>Lo, <name>Hi, <name>Dir of one gap to the reference."""
    L.append(f'\\newcommand{{\\{name}}}{{{f2(x["mean"])}}}'
             f'\\newcommand{{\\{name}Lo}}{{{f2(x["lo"])}}}'
             f'\\newcommand{{\\{name}Hi}}{{{f2(x["hi"])}}}'
             f'\\newcommand{{\\{name}Dir}}{{{tex(x["direction"])}}}'
             f'  % {what}: mean gap to the reference, 95% CI, direction')


def macros_refs(X):
    """Section 9 macros, all named PubRef..., after every other macro; one comment line when
    the section is not computed."""
    st, c = X['set'], X['campaign']
    if not X['computed']:
        return [f'% Section 9 (gap to the references of {st}) skipped: {X["state"]["reason"]}.']
    L = [f'% Section 9: gap to the references of {st}, the best schedule CP-SAT finds within '
         f'{c["budget"]:g} s on {c["workers"]} workers (compact model, warm-started from the best '
         f'of the four rules; scripts/p91_public_cpsat_refs.py), records it marks valid only. '
         f'd = 100 (method - reference) / reference per instance (negative = shorter than the '
         f'reference); trained arms: crossed hierarchical bootstrap, fresh default_rng({HIER_SEED}); '
         f'rules: instance-level cluster bootstrap, fresh default_rng({INST_SEED}); interval and '
         f'direction only; Pool: the eight combinations, a base instance with all of them.',
         f'\\newcommand{{\\PubRefN}}{{{X["state"]["n"]}}}'
         f'\\newcommand{{\\PubRefBudget}}{{{c["budget"]:g}}}'
         f'\\newcommand{{\\PubRefWorkers}}{{{c["workers"]}}}'
         f'\\newcommand{{\\PubRefProvedN}}{{{X["proved"]}}}'
         f'\\newcommand{{\\PubRefLimitN}}{{{X["at_limit"]}}}'
         f'\\newcommand{{\\PubRefPolShortN}}{{{X["policy_shorter_n"]}}}'
         f'  % {st}: instances per combination, time limit (s), workers, references proved '
         f'optimal and stopped at the limit with a gap to the bound (of {X["state"]["expected"]}), '
         f'cases (run, instance) where the policy is shorter than the reference']
    for r in R8:
        row = X['rows'][r]
        g = row['at_limit_gap_mean']
        L.append(f'\\newcommand{{\\PubRef{r}Proved}}{{{row["proved"]}}}'
                 f'\\newcommand{{\\PubRef{r}Limit}}{{{row["at_limit"]}}}'
                 f'\\newcommand{{\\PubRef{r}LimitGap}}{{{"--" if g is None else f"{g:.2f}"}}}'
                 f'  % {st}, {r}: references proved optimal, stopped at the limit with a gap, mean '
                 f'gap between objective and bound of those in % of the objective'
                 + (' (none at the limit)' if g is None else ''))
        for lab, w in (('policy', 'Pol'), ('control', 'Ctl'), ('specialist', 'Spec')):
            ref_line(L, f'PubRef{w}{r}', row['arms'][lab], f'{st}, {r}: {lab}, five runs')
        x = row['arms']['policy']
        L.append(f'\\newcommand{{\\PubRefPol{r}Eq}}{{{u1(x["equal"])}}}'
                 f'\\newcommand{{\\PubRefPol{r}Long}}{{{u1(x["longer"])}}}'
                 f'\\newcommand{{\\PubRefPol{r}Short}}{{{u1(x["shorter"])}}}'
                 f'  % {st}, {r}: % of instances where the policy equals, is longer than, is '
                 f'shorter than the reference (average over the five runs)')
        for q in ('best',) + RULES:
            ref_line(L, f'PubRef{"Best" if q == "best" else REF_RULE_WORD[q]}{r}', row['rules'][q],
                     f'{st}, {r}: ' + ('per-instance best of the four rules' if q == 'best' else q))
    P = X['pooled']
    for lab, w in (('policy', 'Pol'), ('control', 'Ctl'), ('specialist', 'Spec')):
        ref_line(L, f'PubRef{w}Pool', P[lab], f'{st}: {lab}, pooled over the eight combinations')
    L.append(f'\\newcommand{{\\PubRefPolPoolEq}}{{{u1(P["policy"]["equal"])}}}'
             f'\\newcommand{{\\PubRefPolPoolLong}}{{{u1(P["policy"]["longer"])}}}'
             f'\\newcommand{{\\PubRefPolPoolShort}}{{{u1(P["policy"]["shorter"])}}}'
             f'  % {st}: % of instance-combination cells where the policy equals, is longer than, '
             f'is shorter than the reference (average over the five runs)')
    for q in ('best',) + RULES:
        ref_line(L, f'PubRef{"Best" if q == "best" else REF_RULE_WORD[q]}Pool', P[q],
                 f'{st}: ' + ('per-instance best of the four rules' if q == 'best' else q)
                 + ', pooled over the eight combinations')
    bk = X['best_known_N']
    L.append(f'\\newcommand{{\\PubRefBkBelowN}}{{{bk["below"]}}}'
             f'\\newcommand{{\\PubRefBkEqualN}}{{{bk["equal"]}}}'
             f'\\newcommand{{\\PubRefBkAboveN}}{{{bk["above"]}}}'
             f'  % {st}, regime N: references below, equal to, above the best-known value (of '
             f'{bk["counted"]}; {bk["table_proved"]} values marked proved optimal in the table)')
    return L


def collision_check(names, root, out_mac, skip_name=None):
    """No macro file of manuscript/ or paper/ of the input root defines one of the names; the
    output itself is skipped, and with skip_name every file of that name (an earlier output of
    the same family under another output root)."""
    files = sorted(glob.glob(os.path.join(root, 'manuscript', 'macros_*.tex')) +
                   glob.glob(os.path.join(root, 'paper', 'macros_*.tex')) +
                   [os.path.join(root, d, 'macros.tex') for d in ('manuscript', 'paper')])
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) == os.path.abspath(out_mac) or \
                (skip_name and os.path.basename(f) == skip_name):
            continue
        clash = set(names) & set(rx.findall(open(f, errors='replace').read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, root)}: {sorted(clash)[:5]}')
    return len(files)


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None                                     # no bare NaN token in the JSON
    return x


def ci(x, lo='lo', hi='hi'):
    return f'[{x[lo]:+.2f}, {x[hi]:+.2f}]'


def markdown(res, B):
    sel = res['selection']
    md = ['# Public-instance family: policy, control, specialists and rules', '',
          f'Training seeds {", ".join(map(str, SEEDS))} per arm, exact mask, greedy decoding. '
          f'd = 100 (a - b) / b per training seed and base instance, seed s against seed s; '
          f'positive = first arm slower. Hierarchical 95% intervals (seeds x base instances, '
          f'B = {B}, default_rng({HIER_SEED})), seed-level t intervals (4 df), verdicts at 0.5% / 1% '
          f'on the hierarchical interval, intersection-union test for joint claims, Holm over '
          f'the eight combinations for superiority over the rules.', '',
          '| set | role | base instances |', '|---|---|---|']
    for st, S in res['sets'].items():
        md.append(f'| {st} | {S["role"]} | {S["n"]} |')
    md += ['', '## Checks', '']
    for st, S in res['sets'].items():
        e = S['checks']
        md.append(f'- {st}: {e["n"]} records in one sorted order in all eight regimes (excluded: '
                  f'{", ".join(e["excluded"]) or "none"}; lag scale {e["lag_scale"]}); '
                  f'{e["summaries_checked"]} summary files with n/n feasible, matching means and '
                  f'matching regime-N gaps; {e["n_arrays_bound_checked"]} regime-N arrays at or '
                  f'above the bound of {e["n_bound_checked"]} instances; duplicate rule files: '
                  f'{len(e["duplicate_rule_files"])}.')
    md += ['', '## 1. Policy vs specialists', '']
    for st, S in res['sets'].items():
        P = S['parity']
        md += [f'### {st} ({S["role"]})', '',
               '| comb. | mean | hier. 95% CI | seed-t CI | 0.5% | 1% | 0.5% (t) | 1% (t) | '
               'direction | ties % | per seed |', '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in R8:
            c = P['components'][r]
            md.append(f'| {r} | {c["mean"]:+.2f} | {ci(c)} | {ci(c, "t_lo", "t_hi")} | {c["v0.5"]} | '
                      f'{c["v1.0"]} | {c["v0.5_t"]} | {c["v1.0_t"]} | {c["direction"]} | '
                      f'{c["ties"]:.1f} | ' + ', '.join(f'{m:+.2f}' for m in c['seed_means']) + ' |')
        md += ['', '| family | mean | hier. 95% CI | 0.5% | 1% | equivalent on all (0.5 / 1) | '
               'non-inferior on all (0.5 / 1) | failing at 1% (eq.) |',
               '|---|---|---|---|---|---|---|---|']
        for fam in ('All', 'Train', 'Held'):
            x, it = P['pooled'][fam], P['iut']
            md.append(f'| {fam} | {x["mean"]:+.2f} | {ci(x)} | {x["v0.5"]} | {x["v1.0"]} | '
                      f'{it[f"{fam}|0.5|equivalent"]["holds"]} / {it[f"{fam}|1.0|equivalent"]["holds"]} | '
                      f'{it[f"{fam}|0.5|non-inferior"]["holds"]} / {it[f"{fam}|1.0|non-inferior"]["holds"]} | '
                      f'{",".join(it[f"{fam}|1.0|equivalent"]["failing"]) or "-"} |')
        md.append('')
    md += ['## 2. Control on the combinations it never saw', '',
           'LS and LW are seen by the policy and not by the control; SW and LSW by neither.', '']
    for st, S in res['sets'].items():
        C, P = S['control'], S['parity']['components']
        md += [f'### {st}', '',
               '| comb. | policy vs specialist | 1% | control vs specialist | 0.5% | 1% | '
               'policy vs control | 0.5% | 1% |', '|---|---|---|---|---|---|---|---|---|']
        for r in UNSEEN:
            a, b, c = P[r], C['control_vs_specialist'][r], C['policy_vs_control'][r]
            md.append(f'| {r} | {a["mean"]:+.2f} {ci(a)} | {a["v1.0"]} | {b["mean"]:+.2f} {ci(b)} | '
                      f'{b["v0.5"]} | {b["v1.0"]} | {c["mean"]:+.2f} {ci(c)} | {c["v0.5"]} | {c["v1.0"]} |')
        for fam in ('Pair', 'Held', 'Unseen'):
            b, c = C['pooled_control_vs_specialist'][fam], C['pooled_policy_vs_control'][fam]
            md.append(f'| pooled {fam} | | | {b["mean"]:+.2f} {ci(b)} | {b["v0.5"]} | {b["v1.0"]} | '
                      f'{c["mean"]:+.2f} {ci(c)} | {c["v0.5"]} | {c["v1.0"]} |')
        md += ['', 'Control on its own combinations (control vs specialist): ' + ', '.join(
            f'{r} {C["control_vs_specialist"][r]["mean"]:+.2f} {ci(C["control_vs_specialist"][r])}'
            for r in CTL_SEEN), '']
    md += ['## 3. Policy vs dispatching rules', '',
           'Best rule = per-instance minimum of FIFO, MOR, SPT and MWKR under the same mask. '
           'Shares are percent of base instances, averaged over the five seeds (seed mean) or '
           f'for the selected seed (s{sel["seed"]}).' if sel['applied'] else
           'Best rule = per-instance minimum of FIFO, MOR, SPT and MWKR under the same mask.', '']
    for st, S in res['sets'].items():
        md += [f'### {st}', '',
               '| comb. | vs best rule | hier. 95% CI | shorter / equal / longer % | Holm p (t) | '
               'Holm p (boot) | selected | selected CI | selected shorter / equal / longer % | '
               'rule that is best (% of instances) |', '|---|---|---|---|---|---|---|---|---|---|']
        for r in R8:
            x, y = S['rules'][r]['best'], S['rules'][r].get('best_selected')
            ys = (f'{y["mean"]:+.2f} | {ci(y)} | {y["shorter"]:.0f} / {y["equal"]:.0f} / {y["longer"]:.0f}'
                  if y else '- | - | -')
            md.append(f'| {r} | {x["mean"]:+.2f} | {ci(x)} | {x["shorter"]:.0f} / {x["equal"]:.0f} / '
                      f'{x["longer"]:.0f} | {x["p_t_holm"]:.2g} | {x["p_boot_holm"]:.2g} | {ys} | '
                      + ', '.join(f'{q} {v:.0f}' for q, v in S['rules'][r]['best_rule_share'].items())
                      + ' |')
        md += ['', '| comb. | ' + ' | '.join(f'vs {q} (shorter %)' for q in RULES) + ' |',
               '|---|' + '---|' * len(RULES)]
        for r in R8:
            md.append(f'| {r} | ' + ' | '.join(
                f'{S["rules"][r]["rules"][q]["mean"]:+.2f} ({S["rules"][r]["rules"][q]["shorter"]:.0f})'
                for q in RULES) + ' |')
        md.append('')
    md += ['## 4. Regime N: gap to the best-known value (%)', '',
           'Best-known values: Behnke and Geiger (2012) for Brandimarte and Hurink (the smaller of '
           'the literature value and the table\'s CP Optimizer value); per-instance OR-Tools '
           'results with an 1800 s limit from the SD2 release for SD2. On SD2 20x10 only one '
           'reference is proved optimal, so the gap there is a gap to a strong solver.', '',
           '| set | proved optimal | policy (seed mean, sd) | selected | control | N specialist | '
           'FIFO | MOR | SPT | MWKR | best rule | policy on proved | policy to lower bound |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st, S in res['sets'].items():
        A = S['anchor']['arms']
        g = lambda k: f'{A[k]["gap"]:.2f}' if k in A else '-'          # noqa: E731
        md.append(f'| {st} | {S["anchor"]["proved"]} of {S["n"]} | {A["policy"]["gap"]:.2f} '
                  f'({A["policy"]["sd"]:.2f}) | {g("policy selected")} | {g("control")} | '
                  f'{A["specialist N"]["gap"]:.2f} ({A["specialist N"]["sd"]:.2f}) | {g("FIFO")} | '
                  f'{g("MOR")} | {g("SPT")} | {g("MWKR")} | {g("best rule")} | '
                  f'{A["policy"].get("gap_proved", float("nan")):.2f} | '
                  + (f'{A["policy"]["gap_lb"]:.2f}' if 'gap_lb' in A['policy'] else '-') + ' |')
    md += ['', '## 5. Selected seed', '']
    if sel['applied']:
        md.append(f'Rule: {sel["rule"]}. Selected: s{sel["seed"]}. Scores: ' + ', '.join(
            f'{k} {v["score"]:.5f} (update index {v["index"]})' for k, v in sel['records'].items()) + '.')
    else:
        md.append(f'The selection rule could not be applied: {sel["reason"]}. Seed means only.')
    md += ['', '## 6. Per-seed gap to the per-instance best rule (%)', '']
    for st, S in res['sets'].items():
        md += [f'### {st}', '', '| comb. | arm | ' + ' | '.join(f's{s}' for s in SEEDS) +
               ' | mean | sd |', '|---|---|' + '---|' * (len(SEEDS) + 2)]
        for r in R8:
            for lab in ('policy', 'control', 'specialist'):
                x = S['per_seed'][f'{r}|{lab}']
                md.append(f'| {r} | {lab} | ' + ' | '.join(f'{v:+.2f}' for v in x['per_seed'])
                          + f' | {x["mean"]:+.2f} | {x["sd"]:.2f} |')
        md.append('')
    for st, X in res.get('sensitivity', {}).items():
        e = X['checks']
        md += [f'## 7. Sensitivity: {st} ({X["change"]}) beside {X["base"]}', '',
               f'Records: {e["records_compared"]} compared with {X["base"]}; fields changed: '
               + ', '.join(f'{k} ({v})' for k, v in e['fields_changed'].items()) + '. '
               f'Free check: {e["free_check"]["arrays"]} N and S arrays '
               f'({e["free_check"]["makespans"]} makespans) of every run and rule equal the '
               f'scale-1.0 arrays bit for bit. Combinations with W also move their outage '
               f'calendars (H shrinks with the lags); only the combinations with L are compared.', '',
               '| comb. | best-rule R/N (1.0 / 0.5) | policy R/N (1.0 / 0.5) |', '|---|---|---|']
        for r in L_COMBOS:
            C = X['combos'][r]
            md.append(f'| {r} | {C["one"]["rule_effect"]["best_rule"]:.2f} / '
                      f'{C["half"]["rule_effect"]["best_rule"]:.2f} | '
                      f'{C["one"]["rule_effect"]["policy"]:.2f} / {C["half"]["rule_effect"]["policy"]:.2f} |')
        for key, title, ch in (('policy_vs_specialist', 'Policy vs specialist', 'par'),
                               ('policy_vs_control', 'Policy vs control', 'polctl'),
                               ('control_vs_specialist', 'Control vs specialist', None),
                               ('policy_vs_best_rule', 'Policy vs best rule', 'rule')):
            md += ['', f'{title}:', '',
                   '| comb. | scale 1.0 | 0.5% | 1% | scale 0.5 | seed-t CI | 0.5% | 1% | per seed (0.5) | '
                   'change 0.5 - 1.0 |', '|---|---|---|---|---|---|---|---|---|---|']
            for r in L_COMBOS:
                a, b = X['combos'][r]['one'][key], X['combos'][r]['half'][key]
                c = X['combos'][r]['change'][ch] if ch else None
                md.append(f'| {r} | {a["mean"]:+.2f} {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | '
                          f'{b["mean"]:+.2f} {ci(b)} | {ci(b, "t_lo", "t_hi")} | {b["v0.5"]} | {b["v1.0"]} | '
                          + ', '.join(f'{m:+.2f}' for m in b['seed_means'])
                          + (f' | {c["mean"]:+.2f} {ci(c)} |' if c else ' | - |'))
            pk = {'policy_vs_specialist': 'par', 'policy_vs_control': 'polctl',
                  'control_vs_specialist': 'ctlspec', 'policy_vs_best_rule': 'rule'}[key]
            a, b = X['pooled']['one'][pk], X['pooled']['half'][pk]
            md.append(f'| pooled | {a["mean"]:+.2f} {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | '
                      f'{b["mean"]:+.2f} {ci(b)} | {ci(b, "t_lo", "t_hi")} | {b["v0.5"]} | {b["v1.0"]} | | |')
        it = X['iut']['half']
        md += ['', 'Equivalent on all four combinations with L at lag scale 0.5 (0.5% / 1%): '
               f'{it["L|0.5|equivalent"]["holds"]} / {it["L|1.0|equivalent"]["holds"]} (failing at 1%: '
               f'{",".join(it["L|1.0|equivalent"]["failing"]) or "none"}); scale 1.0: '
               f'{X["iut"]["one"]["L|0.5|equivalent"]["holds"]} / {X["iut"]["one"]["L|1.0|equivalent"]["holds"]}.',
               '', 'Policy vs each rule, mean d (shorter %), scale 1.0 / 0.5:', '',
               '| comb. | ' + ' | '.join(RULES) + ' |', '|---|' + '---|' * len(RULES)]
        for r in L_COMBOS:
            C = X['combos'][r]
            md.append(f'| {r} | ' + ' | '.join(
                f'{C["one"]["rules"][q]["mean"]:+.2f} ({C["one"]["rules"][q]["shorter"]:.0f}) / '
                f'{C["half"]["rules"][q]["mean"]:+.2f} ({C["half"]["rules"][q]["shorter"]:.0f})'
                for q in RULES) + ' |')
        md.append('')
    md += ['## Diagnostics', '', '### Validation at the kept checkpoints (20 SD2 10x5 validation '
           'records)', '',
           '| comb. | specialist (mean of seeds) | policy minus specialist % per seed | control '
           'minus specialist % per seed | specialist kept update | specialist gain in the last '
           'quarter % |', '|---|---|---|---|---|---|']
    V = res['validation']
    for r in R8:
        x = V[r]
        pm = ', '.join(f'{v:+.2f}' for v in x['released_minus_spec']) if 'released_minus_spec' in x else '-'
        cm = ', '.join(f'{v:+.2f}' for v in x['control_minus_spec']) if 'control_minus_spec' in x else '-'
        md.append(f'| {r} | {x["spec_kept_mean"]:.1f} | {pm} | {cm} | '
                  + ', '.join(map(str, x['spec_kept_update'])) + ' | '
                  + ', '.join(f'{v:.2f}' for v in x['spec_last_quarter_gain']) + ' |')
    for arm in ('released', 'control'):
        md.append(f'| ({arm}) | | | | ' + ', '.join(map(str, V[arm]['kept_update'])) + ' | '
                  + ', '.join(f'{v:.2f}' for v in V[arm]['last_quarter_gain']) + ' |')
    md += ['', '### Rule effects (mean over instances of the R / N makespan ratio) and exact ties', '',
           '| set | ' + ' | '.join(R8[1:]) + ' |', '|---|' + '---|' * (len(R8) - 1)]
    for st, S in res['sets'].items():
        md.append(f'| {st} best rule | ' + ' | '.join(
            f'{S["rule_effect"][r]["best_rule"]:.2f}' for r in R8[1:]) + ' |')
        md.append(f'| {st} policy | ' + ' | '.join(
            f'{S["rule_effect"][r]["policy"]:.2f}' for r in R8[1:]) + ' |')
    md += ['', 'Instances on which every policy seed, every specialist seed and the best rule '
           'give the same makespan, and the correlation over instances of the seed-mean policy '
           'and specialist makespans:', '', '| set | ' + ' | '.join(R8) + ' |',
           '|---|' + '---|' * len(R8)]
    for st, S in res['sets'].items():
        md.append(f'| {st} | ' + ' | '.join(
            f'{S["trivial"][r]["count"]} ({S["trivial"][r]["corr_policy_specialist"]:.2f})'
            for r in R8) + ' |')
    return md


def markdown_eq(res):
    """Section 8 in the report, beside the released policy's numbers of sections 1 and 4."""
    E = res['equal_experience']
    md = ['', f'## 8. Equal-experience policy ({E["arm"]}, {E["updates"]} updates on '
          f'{",".join(E["regimes"])})', '',
          f'The released recipe with {E["updates"]} updates instead of 2000, so that each of its six '
          f'combinations gets as many episodes as a specialist (1000 updates) gets of its one. '
          f'Seeds {", ".join(map(str, E["seeds"]))}, seed s against seed s; same estimators, margins '
          f'and streams as sections 1 to 7. Configurations equal those of '
          f'{E.get("config_reference", "pubx-v1 of the same seed")} '
          f'except for ' + ', '.join(sorted({k for v in E['config_differs_in'].values() for k in v}))
          + '.', '',
          '- Checks: ' + '; '.join(
              f'{st}: {e["summaries_checked"]} summaries, {e["n_arrays_bound_checked"]} regime-N '
              f'arrays at or above the bound' + (f', free check {e["free_check"]["arrays"]} N and S '
                                                 f'arrays bit for bit' if 'free_check' in e else '')
              for st, e in E['checks'].items()) + '.', '']
    for st, S in E['sets'].items():
        P0 = res['sets'][st]['parity']
        md += [f'### {st} ({S["role"]})', '',
               '| comb. | released vs specialist | 1% | equal-exp. vs specialist | seed-t CI | 0.5% | '
               '1% | direction | per seed | equal-exp. vs released | 0.5% | 1% | direction |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for r in R8:
            a, c, d = P0['components'][r], S['parity']['components'][r], S['vs_policy']['components'][r]
            md.append(f'| {r} | {a["mean"]:+.2f} {ci(a)} | {a["v1.0"]} | {c["mean"]:+.2f} {ci(c)} | '
                      f'{ci(c, "t_lo", "t_hi")} | {c["v0.5"]} | {c["v1.0"]} | {c["direction"]} | '
                      + ', '.join(f'{m:+.2f}' for m in c['seed_means'])
                      + f' | {d["mean"]:+.2f} {ci(d)} | {d["v0.5"]} | {d["v1.0"]} | {d["direction"]} |')
        it = S['parity']['iut']
        md += ['', '| family | released vs specialist | equal-exp. vs specialist | 0.5% | 1% | '
               'equivalent on all (0.5 / 1) | failing at 1% (eq.) | equal-exp. vs released | 0.5% | 1% |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for fam in ('All', 'Train', 'Held'):
            a, x, y = P0['pooled'][fam], S['parity']['pooled'][fam], S['vs_policy']['pooled'][fam]
            md.append(f'| {fam} | {a["mean"]:+.2f} {ci(a)} | {x["mean"]:+.2f} {ci(x)} | {x["v0.5"]} | '
                      f'{x["v1.0"]} | {it[f"{fam}|0.5|equivalent"]["holds"]} / '
                      f'{it[f"{fam}|1.0|equivalent"]["holds"]} | '
                      f'{",".join(it[f"{fam}|1.0|equivalent"]["failing"]) or "-"} | '
                      f'{y["mean"]:+.2f} {ci(y)} | {y["v0.5"]} | {y["v1.0"]} |')
        A = S['anchor']
        md += ['', f'Regime N, gap to best known: equal-experience {A["gap"]:.2f} (sd {A["sd"]:.2f}), '
               f'released {res["sets"][st]["anchor"]["arms"]["policy"]["gap"]:.2f}, N specialist '
               f'{res["sets"][st]["anchor"]["arms"]["specialist N"]["gap"]:.2f}.', '']
    V = E['validation']
    md += ['### Validation at the kept checkpoints (20 SD2 10x5 validation records)', '',
           '| comb. | equal-exp. minus specialist % per seed | mean | released minus specialist (mean) |',
           '|---|---|---|---|']
    for r in TRAIN:
        md.append(f'| {r} | ' + ', '.join(f'{v:+.2f}' for v in V[r]['eq_minus_spec'])
                  + f' | {V[r]["eq_minus_spec_mean"]:+.2f} | {V[r]["released_minus_spec_mean"]:+.2f} |')
    md += ['', 'Kept update per seed: ' + ', '.join(map(str, V['eq']['kept_update']))
           + f' (of {E["updates"]}); gain in the last quarter %: '
           + ', '.join(f'{v:.2f}' for v in V['eq']['last_quarter_gain']) + '.']
    return md


def markdown_verdict(res):
    """The combined verdict against the per-instance best rule, after the diagnostics."""
    md = ['', '## Combined verdict against the per-instance best rule', '',
          'd = policy minus per-instance best rule, over the rule (five seeds). ahead: the '
          'hierarchical 95% CI lies below zero and the Holm-adjusted one-sided p-values for '
          '"the policy is shorter" (seed-level t and hierarchical bootstrap) are both below 0.05. '
          'behind: the CI lies above zero and the Holm-adjusted one-sided p-values for "the '
          'policy is longer" (each kind with its own Holm family over the eight combinations) '
          'are both below 0.05. not distinguishable: otherwise.', '']
    for st, S in res['sets'].items():
        md += [f'### {st}', '',
               '| comb. | mean | hier. 95% CI | Holm p shorter (t / boot) | p longer (t / boot) | '
               'Holm p longer (t / boot) | verdict |', '|---|---|---|---|---|---|---|']
        for r in R8:
            x = S['rules'][r]['best']
            md.append(f'| {r} | {x["mean"]:+.2f} | {ci(x)} | {x["p_t_holm"]:.2g} / '
                      f'{x["p_boot_holm"]:.2g} | {x["p_t_greater"]:.2g} / {x["p_boot_greater"]:.2g} | '
                      f'{x["p_t_greater_holm"]:.2g} / {x["p_boot_greater_holm"]:.2g} | {x["verdict"]} |')
        vs = [S['rules'][r]['best']['verdict'] for r in R8]
        md += ['', f'Ahead on {vs.count(VERDICT_WORDS[0])}, behind on {vs.count(VERDICT_WORDS[1])}, '
               f'not distinguishable on {vs.count(VERDICT_WORDS[2])} of 8 combinations.', '']
    return md


def markdown_refs(X, partial=False):
    """Section 9 in the report: one line when it is not computed; with partial=True the
    complete combinations of a partial reading."""
    st, c, S = X['set'], X['campaign'], X['state']
    md = ['', f'## 9. Gap to the references ({st})', '']
    if not X.get('combinations'):
        md.append(f'Skipped: {S["reason"] or "no complete combination"}.')
        return md
    combos = X['combinations']
    if partial:
        md += [f'PARTIAL READING: complete combinations {", ".join(combos)}; '
               + (f'incomplete {", ".join(r for r in R8 if r not in combos)}; ' if len(combos) < 8 else '')
               + f'{S["total"]["valid"]} of {S["expected"]} references valid. The pooled rows cover '
               'the complete combinations only. Not for the manuscript.', '']
    md += [f'Reference: the best schedule CP-SAT finds within {c["budget"]:g} s on {c["workers"]} '
           f'workers (compact model, warm-started from the best of the four rules; '
           f'scripts/p91_public_cpsat_refs.py, records it marks valid only; folder {X["folder"]}). '
           'd = 100 (method - reference) / reference per instance; trained arms (five runs): crossed '
           'hierarchical 95% CI; rules: instance-level 95% CI; interval and direction only.', '',
           '| comb. | n | proved | at limit (mean gap %) | policy | = / > / < % | control | specialist | '
           'best rule | FIFO | MOR | SPT | MWKR |', '|---' * 13 + '|']

    def cell(x):
        return f'{x["mean"]:+.2f} {ci(x)} {x["direction"]}'
    for r in combos:
        row = X['rows'][r]
        g = row['at_limit_gap_mean']
        p = row['arms']['policy']
        md.append(f'| {r} | {row["n"]} | {row["proved"]} | {row["at_limit"]}'
                  + (f' ({g:.2f})' if g is not None else '') + f' | {cell(p)} | '
                  f'{p["equal"]:.1f} / {p["longer"]:.1f} / {p["shorter"]:.1f} | '
                  f'{cell(row["arms"]["control"])} | {cell(row["arms"]["specialist"])} | '
                  + ' | '.join(cell(row['rules'][q]) for q in ('best',) + RULES) + ' |')
    P = X['pooled']
    p = P['policy']
    md.append(f'| pooled | {S["n"] * len(combos)} | {X["proved"]} | {X["at_limit"]} | {cell(p)} | '
              f'{p["equal"]:.1f} / {p["longer"]:.1f} / {p["shorter"]:.1f} | {cell(P["control"])} | '
              f'{cell(P["specialist"])} | ' + ' | '.join(cell(P[q]) for q in ('best',) + RULES) + ' |')
    bk = X['best_known_N']
    U = X['unexpected']
    md += ['', f'Regime N against the best-known values ({bk["counted"]} references): below '
           f'{bk["below"]}, equal {bk["equal"]}, above {bk["above"]} ({bk["above_where_table_proved"]} '
           f'of them where the table marks the value proved optimal); proved optimal here and in '
           f'the table: {bk["proved_here_and_in_table"]}.',
           f'Policy shorter than the reference: {X["policy_shorter_n"]} (run, instance) cases, listed '
           f'in the JSON (possible where the reference is not proved optimal).',
           'Records worth a look: ' + '; '.join(f'{k.replace("_", " ")} {len(v)}' for k, v in U.items())
           + f'; records not counted {X["not_counted_n"]}.']
    return md


# ================================================================== section 10: across masks
MASK_ARM_WORD = {'policy': 'Pol', 'control': 'Ctl', 'specialist': 'Spec'}


def analyse_mask(IA, IN, B):
    """Section 10 (family under a mask): each run of IA against its non-delay counterpart of
    the same seed in IN, and the rules under the two masks, per test set (module docstring).
    d = 100 (a - b) / b, a under the mask of IA, b under the non-delay mask."""
    if IN.fam.rule or not IA.fam.rule:
        raise ValueError('section 10 compares a family under a mask with the non-delay family')
    out = dict(mask=IA.fam.mask_mode, reference='nondelay', seeds=list(SEEDS),
               difference='d = 100 (a - b) / b per training seed and base instance, a the run '
                          f'under the {IA.fam.mask_mode} mask, b its non-delay counterpart of the '
                          'same seed; rules: per base instance; negative = the '
                          f'{IA.fam.mask_mode} mask gives the shorter schedule',
               families={f: list(v) for f, v in MASK_FAMILIES.items()}, sets={})
    for st in mask_sets(IA, IN):
        S = dict(role=Inputs.spec(st)[1], n=len(IA.names[st]), arms={}, rules={})
        if IA.names[st] != IN.names[st]:
            refuse(f'{st}: the two families read other records')
        for arm, key in (('policy', 'released'), ('control', 'control')):
            D = {r: np.stack([rel(IA.ms[(st, key, None, s, r)], IN.ms[(st, key, None, s, r)])
                              for s in SEEDS]) for r in R8}
            S['arms'][arm] = dict(
                components={r: dict(component(D[r], B), **seed_shares(D[r])) for r in R8},
                pooled={f: dict(pooled([D[r] for r in regs], B), combinations=list(regs))
                        for f, regs in MASK_FAMILIES.items()})
        D = {r: np.stack([rel(IA.ms[(st, 'specialist', r, s, r)], IN.ms[(st, 'specialist', r, s, r)])
                          for s in SEEDS]) for r in R8}
        S['arms']['specialist'] = dict(
            components={r: dict(component(D[r], B), **seed_shares(D[r])) for r in R8},
            pooled={f: dict(pooled([D[r] for r in MASK_FAMILIES[f]], B),
                            combinations=list(MASK_FAMILIES[f])) for f in SPEC_MASK_FAMILIES})
        for q in ('best',) + RULES:
            if q == 'best':
                a = {r: np.min(np.stack([IA.rule[(st, x, r)] for x in RULES]), axis=0) for r in R8}
                b = {r: np.min(np.stack([IN.rule[(st, x, r)] for x in RULES]), axis=0) for r in R8}
            else:
                a = {r: IA.rule[(st, q, r)] for r in R8}
                b = {r: IN.rule[(st, q, r)] for r in R8}
            dq = {r: rel(a[r], b[r]) for r in R8}
            S['rules'][q] = dict(
                components={r: dict(ref_inst_gap(dq[r], B), **shares(dq[r]), mean_active=float(a[r].mean()),
                                    mean_nondelay=float(b[r].mean())) for r in R8},
                pooled={f: dict(ref_inst_gap(np.stack([dq[r] for r in regs], axis=1), B),
                                combinations=list(regs),
                                **shares(np.stack([dq[r] for r in regs], axis=1)))
                        for f, regs in MASK_FAMILIES.items()})
        best = {}
        for lab, I in (('active', IA), ('nondelay', IN)):
            for r in R8:
                m = {q: float(I.rule[(st, q, r)].mean()) for q in RULES}
                stk = np.stack([I.rule[(st, q, r)] for q in RULES])
                low = stk.min(axis=0)
                best[f'{lab}|{r}'] = dict(
                    lowest_mean=min(RULES, key=lambda q: (m[q], RULES.index(q))), means=m,
                    best_share={q: float(np.mean(stk[j] == low)) * 100 for j, q in enumerate(RULES)})
        S['best_rule_by_mask'] = best
        out['sets'][st] = S
    return out


def mask_sets(IA, IN):
    """The sets of section 10: the six test sets, then each sensitivity set (the halved
    lags) whose arrays both families hold for every run and rule (load_set read them)."""
    def held(I, st):
        return all((st, arm, combo, s, r) in I.ms for arm, combo, regs in I.runs()
                   for s in SEEDS for r in regs) and \
            all((st, q, r) in I.rule for q in RULES for r in R8)
    return list(SETS) + [st for st in SENS_WORD if held(IA, st) and held(IN, st)]


def macros_mask(X, prefix):
    """Section 10 macros, all named <prefix>Mask<set>...: per combination on every test set
    and the sensitivity set, and pooled; the rule with the lowest mean under each mask on the
    headline set."""
    mk = X['mask']
    L = [f'% Section 10: the runs under the {mk} mask against their non-delay counterparts of the '
         f'same seed and the rules under the two masks; d = 100 ({mk} - non-delay) / non-delay '
         'per seed and base instance (rules: per base instance), negative = shorter under the '
         f'{mk} mask; runs: hierarchical intervals, fresh default_rng({HIER_SEED}); rules: '
         f'instance-level intervals, fresh default_rng({INST_SEED}).']
    for st, S in X['sets'].items():
        w = Inputs.spec(st)[0]
        for arm, aw in MASK_ARM_WORD.items():
            A = S['arms'][arm]
            for r in R8:
                comp_lines(L, f'{prefix}Mask{w}{aw}{r}', A['components'][r],
                           f'{st}, {r}: {arm} under the {mk} mask vs its non-delay counterpart')
            for f, x in A['pooled'].items():
                pool_lines(L, f'{prefix}Mask{w}{aw}Pool{f}', x,
                           f'{st}: {arm} under the {mk} mask vs non-delay, {f}')
                dir_line(L, f'{prefix}Mask{w}{aw}Pool{f}', x,
                         f'{st}: {arm} under the {mk} mask vs non-delay, pooled over {f}')
        for q in ('best',) + RULES:
            qw = 'BestRule' if q == 'best' else REF_RULE_WORD[q]
            what = 'per-instance best of the four rules' if q == 'best' else q
            R = S['rules'][q]
            for r, x in list(R['components'].items()) + [(f'Pool{f}', x) for f, x in R['pooled'].items()]:
                L.append(f'\\newcommand{{\\{prefix}Mask{w}{qw}{r}}}{{{f2(x["mean"])}}}'
                         f'\\newcommand{{\\{prefix}Mask{w}{qw}{r}Lo}}{{{f2(x["lo"])}}}'
                         f'\\newcommand{{\\{prefix}Mask{w}{qw}{r}Hi}}{{{f2(x["hi"])}}}'
                         f'\\newcommand{{\\{prefix}Mask{w}{qw}{r}Dir}}{{{tex(x["direction"])}}}'
                         f'  % {st}, {r}: {what} under the {mk} mask vs the non-delay mask, mean d, '
                         f'instance-level 95% CI, direction')
        if st == EQ_HEADLINE:
            for lab, lw in (('active', 'Act'), ('nondelay', 'Nd')):
                for r in R8:
                    b = S['best_rule_by_mask'][f'{lab}|{r}']
                    L.append(f'\\newcommand{{\\{prefix}Mask{w}LowRule{lw}{r}}}{{{b["lowest_mean"]}}}'
                             f'  % {st}, {r}: the rule with the lowest mean makespan under the '
                             f'{"non-delay" if lab == "nondelay" else lab} mask ('
                             + ', '.join(f'{q} {v:.1f}' for q, v in b['means'].items()) + ')')
    return L


def markdown_mask(X):
    mk = X['mask']
    md = ['', f'## 10. The {mk} mask against the non-delay mask', '',
          X['difference'] + '. Runs: crossed hierarchical 95% CI over seeds and base instances; '
          'rules: instance-level 95% CI. Pooled rows keep the combinations of a base instance '
          'together.', '']
    for st, S in X['sets'].items():
        md += [f'### {st} ({S["role"]})', '',
               '| comb. | policy | dir. | control | dir. | specialist | dir. | best rule | dir. | '
               f'lowest-mean rule ({mk} / non-delay) |', '|---|---|---|---|---|---|---|---|---|---|']
        for r in R8:
            cells = []
            for arm in MASK_ARM_WORD:
                x = S['arms'][arm]['components'][r]
                cells += [f'{x["mean"]:+.2f} {ci(x)}', x['direction']]
            x = S['rules']['best']['components'][r]
            cells += [f'{x["mean"]:+.2f} {ci(x)}', x['direction'],
                      f'{S["best_rule_by_mask"][f"active|{r}"]["lowest_mean"]} / '
                      f'{S["best_rule_by_mask"][f"nondelay|{r}"]["lowest_mean"]}']
            md.append(f'| {r} | ' + ' | '.join(cells) + ' |')
        for f in MASK_FAMILIES:
            cells = []
            for arm in MASK_ARM_WORD:
                x = S['arms'][arm]['pooled'].get(f)
                cells += ([f'{x["mean"]:+.2f} {ci(x)}', x['direction']] if x else ['-', '-'])
            x = S['rules']['best']['pooled'][f]
            cells += [f'{x["mean"]:+.2f} {ci(x)}', x['direction'], '']
            md.append(f'| pooled {f} | ' + ' | '.join(cells) + ' |')
        md.append('')
    return md


# ================================================================== partial reports
def run_files(I, arm, combo, s):
    """Every file of one principal run of the family: training records, and the arrays and
    the summary in every test and sensitivity folder."""
    regs = R8 if arm != 'specialist' else [combo]
    name = I.fam.model(arm, combo, s)
    out = list(I.fam.paths(arm, combo, s))
    for st in list(SETS) + list(SENS_WORD):
        d = os.path.join(REG.PUB_RESULTS, st + XS)
        out += [os.path.join(d, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy') for r in regs]
        out.append(os.path.join(d, f'summary_{REG.PUB_PREFIX}{name}{XS}.md'))
    return out


def family_state(I, seeds):
    """Per seed the files of the principal runs that are missing (a log without its completion
    line counts as missing), and the rule arrays and summaries that are missing."""
    per = {}
    for s in seeds:
        miss = []
        for arm, combo, _ in I.runs():
            for x in run_files(I, arm, combo, s):
                p = I.p(x)
                if not os.path.exists(p) or (x.endswith('.log') and 'done in' not in
                                             open(p, errors='replace').read()):
                    miss.append(x)
        per[s] = miss
    rules = [os.path.relpath(os.path.join(I.rule_folder(st), f), I.root)
             for st in list(SETS) + list(SENS_WORD)
             for f in [f'Result_{q}_{r}.npy' for q in RULES for r in R8] + [f'summary_rules{XS}.md']
             if not os.path.exists(os.path.join(I.rule_folder(st), f))]
    return dict(missing=per, rules_missing=rules,
                complete=[s for s in seeds if not per[s]])


def outside_outputs(where, root, what):
    """Refuses a folder in the repository root or its paper, manuscript or results folders."""
    d = os.path.abspath(where)
    for x in sorted({REPO, root}):
        if d == x or any(d == os.path.join(x, sub) or d.startswith(os.path.join(x, sub) + os.sep)
                         for sub in ('paper', 'manuscript', 'results')):
            refuse(f'{what} {where}: a partial reading is never written into the '
                   f'repository root or its paper, manuscript or results folders')
    return d


# ================================================================== main
def partial_references(root, where, B):
    """--partial-references: section 9 on the combinations whose references are all
    counted, JSON and Markdown only (REF_PARTIAL_FILES), in a folder outside the paper and
    results folders of the repository; no other output is written or removed. The checks of
    the test set and the consistency checks of section 9 apply as in a full run."""
    d = os.path.abspath(where)
    for x in sorted({REPO, root}):
        if d == x or any(d == os.path.join(x, sub) or d.startswith(os.path.join(x, sub) + os.sep)
                         for sub in ('paper', 'manuscript', 'results')):
            refuse(f'--partial-references {where}: a partial reading is never written into the '
                   f'repository root or its paper, manuscript or results folders')
    outs = [os.path.join(d, f) for f in REF_PARTIAL_FILES]
    for p in outs:
        if os.path.exists(p):
            os.remove(p)
    I = Inputs(root)
    I.check_records(REF_SET)
    I.load_set(REF_SET)
    R = I.load_refs()
    probs = ref_problems(I, R)
    if probs:
        refuse(f'section 9: the references and the evaluations of {REF_SET} contradict each '
               f'other ({len(probs)}): ' + '; '.join(probs[:5]))
    X = section_refs(I, R, B, combos=ref_state(R)['complete'])
    X.update(partial=True, generator='scripts/p89_public_family.py --partial-references',
             boot=B, seeds=SEEDS,
             note='partial reading: the combinations whose references are all counted; '
                  'not for the manuscript')
    os.makedirs(d, exist_ok=True)
    with open(outs[0], 'w') as f:
        json.dump(jsonable(X), f, indent=1)
    md = [f'# Gap to the references, partial reading ({REF_SET})'] + markdown_refs(X, partial=True)
    with open(outs[1], 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\nwrote {outs[0]}, {outs[1]}')
    return X


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--out', default=REPO, help='root of the outputs (default: the repository)')
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    ap.add_argument('--partial-references', default=None, metavar='DIR',
                    help='only section 9, on the combinations whose references are complete; '
                         'JSON and Markdown in DIR (outside the paper and results folders)')
    ap.add_argument('--family', default='nondelay', choices=FAMILY_KEYS,
                    help='the family read: nondelay (default, every run of the paper so far) or '
                         'active (every recipe trained under the active mask, the rules rolled out '
                         'under it; outputs results/public_family_act.* and '
                         'manuscript/macros_pubfam_act.tex)')
    ap.add_argument('--partial', action='store_true',
                    help='a partial report on the seeds with every file, only under --out '
                         'outside the paper and results folders; marked PARTIAL REPORT')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.partial_references:
        return partial_references(root, a.partial_references, a.boot)
    global SEEDS
    keep_seeds = SEEDS
    try:
        return run_family(a, root, Fam(a.family))
    finally:
        SEEDS = keep_seeds


def run_family(a, root, fam):
    global SEEDS
    if a.partial:
        outside_outputs(a.out, root, '--partial --out')
    outs = [os.path.join(os.path.abspath(a.out), p) for p in fam.out_files]
    out_json, out_md, out_mac = outs
    for p in outs:                    # a refusal anywhere below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    if set(SETS) != set(REG.PUB_SETS) or set(SENS_WORD) != set(REG.PUB_SENS_SETS):
        refuse(f'the registry lists the sets {sorted(REG.PUB_SETS)} and the sensitivity sets '
               f'{sorted(REG.PUB_SENS_SETS)}, this script {sorted(SETS)} and {sorted(SENS_WORD)}')
    partial = None
    try:
        I = Inputs(root, fam)
        if fam.rule or a.partial:
            fs = family_state(I, SEEDS)
            n_miss = sum(len(v) for v in fs['missing'].values())
            if fs['rules_missing']:
                refuse(f'the rule arrays of the {fam.key} family are incomplete: '
                       f'{len(fs["rules_missing"])} files missing, first {fs["rules_missing"][:3]} '
                       f'(scripts/p109_public_active_rules.py run)')
            if n_miss and not a.partial:
                first = [x for s in SEEDS for x in fs['missing'][s]][:4]
                refuse(f'the {fam.key} family is incomplete: {n_miss} files of the principal runs '
                       f'missing or unfinished (complete seeds: {fs["complete"] or "none"}), first '
                       f'{first}; a partial report needs --partial --out <folder>')
            if a.partial:
                if len(fs['complete']) < 2:
                    refuse(f'a partial report needs two seeds with every file of the principal '
                           f'runs; complete seeds: {fs["complete"] or "none"} ('
                           + ', '.join(f's{s}: {len(v)} missing' for s, v in fs['missing'].items()) + ')')
                partial = dict(seeds=list(fs['complete']),
                               left_out={f's{s}': len(v) for s, v in fs['missing'].items() if v},
                               note='PARTIAL REPORT: the seeds with every file of the principal runs; '
                                    'not for the manuscript')
                SEEDS = tuple(fs['complete'])
        eq = I.eq_state() if not partial else partial_eq_state(I)
        for st in list(SETS) + list(SENS_WORD):
            I.check_records(st)
        for st in list(SETS) + list(SENS_WORD):
            I.load_set(st)
        for st in SENS_WORD:
            I.check_sensitivity(st)
        if fam.rule:
            I.check_rule_configs()
        if eq:
            for st in list(SETS) + list(SENS_WORD):
                I.load_eq_set(st)
            for st in SENS_WORD:
                I.check_eq_sensitivity(st)
        R = I.load_refs()              # section 9: the reference records, checked before use
        probs = ref_problems(I, R)
        if probs:
            refuse(f'section 9: the references and the evaluations of {REF_SET} contradict each '
                   f'other ({len(probs)}): ' + '; '.join(probs[:5]))
        IN = None
        if fam.rule:                   # section 10: the non-delay family, checked as in a default run
            IN = Inputs(root, Fam('nondelay'))
            for st in list(SETS) + list(SENS_WORD):
                IN.check_records(st)
            for st in list(SETS) + list(SENS_WORD):
                IN.load_set(st)
            for st in SENS_WORD:
                IN.check_sensitivity(st)
        res = analyse(I, a.boot)
        if eq:
            res['equal_experience'] = analyse_eq(I, a.boot)
        REFX = section_refs(I, R, a.boot)
        L = macros(res, a.boot)
        LV = macros_verdict(res)       # after sections 1 to 7, before section 8
        LD = macros_pool_dir(res)      # after the verdict macros, before section 8
        L += LV + LD
        if eq:
            L += macros_eq(res['equal_experience'])
            L += macros_eq_pool_dir(res['equal_experience'])
        LR = macros_refs(REFX)         # section 9, after every other macro
        L += LR
        LM = []
        if fam.rule:
            L = [rename_line(x, fam) for x in L]
            L[0] = L[0] + f' (--family {fam.key})'
            L[1] = L[1].replace('Public-instance family, exact mask,',
                                f'Public-instance family under the {fam.key} mask (mask_mode '
                                f'{fam.mask_mode}, exact first-decision and tie options; rules '
                                f'rolled out under it by scripts/p109_public_active_rules.py),')
            res['mask_comparison'] = analyse_mask(I, IN, a.boot)
            LM = macros_mask(res['mask_comparison'], fam.prefix)   # section 10, last
            L += LM
        if partial:
            L.insert(0, f'% {partial["note"]}: training seeds {", ".join(map(str, SEEDS))} only.')
        names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(L))
        if len(names) != len(set(names)):
            refuse('duplicate macro names in the output')
        if fam.rule and any(not x.startswith(fam.prefix) for x in names):
            refuse(f'macro names without the prefix {fam.prefix}')
        n_verdict = len(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(LV)))
        n_dir = len(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(LD)))
        n_mask = len(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(LM)))
        n_mac_files = collision_check(names, root, out_mac,
                                      skip_name=os.path.basename(out_mac) if fam.rule else None)
        res.update(generator='scripts/p89_public_family.py' + (f' --family {fam.key}' if fam.rule else ''),
                   seeds=SEEDS, boot=a.boot,
                   alpha=0.05, margins=MARGINS,
                   streams=dict(hierarchical=f'fresh default_rng({HIER_SEED}) per quantity',
                                instance=f'fresh default_rng({INST_SEED}) per quantity'),
                   files_read=I.n_files, macro_files_checked=n_mac_files,
                   n_macros=len(names) - n_verdict - n_dir - n_mask,
                   arrays='row i = i-th record of desc/<R>/ in sorted file-name order',
                   n_macros_verdict=n_verdict,
                   n_macros_note='n_macros counts every macro of the file except the '
                                 'n_macros_verdict combined-verdict macros (Pub<set>Rule...Verdict...)',
                   superiority_verdict_rule=(
                       'ahead: hierarchical 95% CI below zero and both Holm-adjusted one-sided '
                       'p-values (seed-level t, bootstrap; H1 policy shorter) below 0.05; '
                       'behind: CI above zero and both Holm-adjusted one-sided p-values for H1 '
                       'policy longer (own Holm family over the eight combinations) below 0.05; '
                       'not distinguishable otherwise'),
                   n_macros_dir=n_dir,
                   n_macros_dir_note='n_macros does not count the n_macros_dir direction macros '
                                     'of the pooled rows of sections 1, 2 and 7 (names ending in '
                                     'Dir after the verdict macros); the direction macros of '
                                     'section 8 are counted in n_macros with that section')
        if fam.rule:
            res.update(family=dict(key=fam.key, mask_mode=fam.mask_mode, word=fam.word,
                                   prefix=fam.prefix, rule_arrays=fam.rule_root,
                                   runs={arm: fam.model(arm, 'N' if arm == 'specialist' else None, 301)
                                         for arm in ('released', 'control', 'specialist', 'eq')},
                                   config_differs_in=I.config_evidence),
                       n_macros_mask=n_mask, files_read_nondelay=IN.n_files,
                       n_macros_mask_note='n_macros does not count the n_macros_mask macros of '
                                          'section 10 (names <prefix>Mask...)')
        if partial:
            res['partial'] = partial
        res['references'] = REFX
        res['n_macros_references'] = len(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',
                                                    '\n'.join(LR)))
        emit(out_mac, L)
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(jsonable(res), f, indent=1)
        md = markdown(res, a.boot)
        md += markdown_verdict(res)    # after the diagnostics, before section 8
        if eq:
            md += markdown_eq(res)
        md += markdown_refs(REFX)      # section 9, after every other part
        if fam.rule:
            md[0] = md[0].replace('# Public-instance family:',
                                  f'# Public-instance family under the {fam.key} mask:')
            md[2] = md[2].replace('exact mask, greedy decoding',
                                  f'the {fam.key} mask (mask_mode {fam.mask_mode}; exact first-decision '
                                  f'and tie options; rules rolled out under it by '
                                  f'scripts/p109_public_active_rules.py), greedy decoding')
            md += markdown_mask(res['mask_comparison'])
        if partial:
            md = [f'# {partial["note"]}', '', f'Training seeds {", ".join(map(str, SEEDS))}; left '
                  f'out (files missing): ' + ', '.join(f'{k} {v}' for k, v in partial['left_out'].items())
                  + '.', ''] + md
        with open(out_md, 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException as e:
        for p in outs:
            if os.path.exists(p):
                os.remove(p)
                print(f'removed {os.path.relpath(p, a.out)} (refusing to certify)')
        if isinstance(e, REG.MissingResult):
            raise Refusal(f'*** REFUSING TO CERTIFY ***\n{e}')
        raise
    print('\n'.join(md))
    print(f'\n{len(names)} macros in {os.path.relpath(out_mac, a.out)}; '
          f'{I.n_files} result arrays read; wrote {os.path.relpath(out_json, a.out)}, '
          f'{os.path.relpath(out_md, a.out)}')
    return res


def rename_line(line, fam):
    """A macro line of the default family with every macro name Pub... renamed to the
    family's prefix (the same definition under another name)."""
    return line.replace('\\newcommand{\\Pub', '\\newcommand{\\' + fam.prefix)


def partial_eq_state(I):
    """Section 8 in a partial report: True when every file of the equal-experience arm of
    the seeds read exists and every one of its runs has finished, False otherwise (the
    section is left out, never refused)."""
    try:
        return I.eq_state()
    except Refusal:
        return False


if __name__ == '__main__':
    main()
