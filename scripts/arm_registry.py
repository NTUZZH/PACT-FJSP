"""The one place where the principal arms, their training seeds, checkpoint names
and result folders are defined.

Three principal arms are trained in two environments:

  arm          legacy environment            exact-mask environment
  released     jointblind-v1                 xmask-v1          six combinations, 2000 updates
  control      jointblind-nlsw               xmask-ctl         N, L, S, W, 2000 updates
  specialist   spec-<R>                      xmask-solo-<R>    one combination R, 1000 updates

A run is named <model>-s<seed>, with one exception: in the legacy environment the
seed-301 run of the released policy and of each specialist carries no suffix
(jointblind-v1, spec-N), while the control names every seed (jointblind-nlsw-s301).
The exact-mask environment names every seed.

Every evaluation of a run writes Result_greedy+10x25+family+<model>_<R>.npy into a
campaign folder test_results/FAMILY/<folder>/. The legacy campaigns are the folders
named by size (10x25, 20x25, 30x25, 50x25, 80x25) and the special campaigns
(10x25_cpuref, 10x25_blindref, ...). The exact-mask evaluation of the same campaign
goes to <folder>_xmask (eval_family.py --tag _xmask), because the dispatching rules
re-evaluated under the exact mask share the published rule file names.

Selection of the active setting, in this order:
  command line   --pact-env legacy|exact   --pact-seeds 301,302,303 (or 301-305)
                 (removed from sys.argv when this module is imported, so the
                 scripts' own argument parsers never see them)
  environment    PACT_ENV, PACT_SEEDS
  default        legacy, seeds 301, 302, 303 (the setting of the submitted analysis)

Output root. A script writes a generated file through out('paper/macros_x.tex').
In the default setting this is the path itself. In any other setting PACT_OUT must
be given (PACT_OUT=. writes in place); out() then maps paper/... and results/...
under that root, so a five-seed or exact-mask run cannot overwrite the submitted
macros by accident. Files a script reads that another script generated go through
out() as well, so one run of the pipeline reads its own outputs.

Results root. PACT_RESULTS (default test_results/FAMILY) points the readers at
another results tree, for example a scratch tree built to test five-seed code paths.

A requested result file that does not exist is an error naming the file
(MissingResult). Nothing is skipped silently.

Exact-mask environment: every array is produced on the CPU with one torch thread
by scripts/run_eval_newseeds.sh, so the device campaigns 10x25_blindref and
10x25_cpuref fold into 10x25_xmask (FOLDER_ALIAS), and summary_tag() gives the
tag of a folder's summary files. Further exact-mask arms (SECONDARY_ARMS, seeds
301-303), the folder of the specialists' off-combination evaluation (CROSS_FOLDER) and
the public-instance family (PUB_*) are named here too.
"""
import os
import re
import sys

import numpy as np

COMBOS = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
TRAIN_COMBOS = ('N', 'L', 'S', 'W', 'LS', 'LW')
HELD_OUT = ('SW', 'LSW')
CONTROL_COMBOS = ('N', 'L', 'S', 'W')
PREFIX = '10x25+family+'          # every family checkpoint is trained at 10x25

ENVIRONMENTS = ('legacy', 'exact')
DEFAULT_ENV = 'legacy'
DEFAULT_SEEDS = (301, 302, 303)
# Seeds each environment's queues train (scripts/run_seeds45_queue.sh,
# scripts/run_exactmask_queue.sh). A seed outside this set is refused.
PLANNED_SEEDS = {'legacy': (301, 302, 303, 304, 305),
                 'exact': (301, 302, 303, 304, 305)}

ARMS = ('released', 'control', 'specialist')
_STEM = {'legacy': {'released': 'jointblind-v1', 'control': 'jointblind-nlsw',
                    'specialist': 'spec-{R}'},
         'exact': {'released': 'xmask-v1', 'control': 'xmask-ctl',
                   'specialist': 'xmask-solo-{R}'}}
# arms whose seed-301 run carries no suffix
_BARE_301 = {'legacy': ('released', 'specialist'), 'exact': ()}
FOLDER_SUFFIX = {'legacy': '', 'exact': '_xmask'}
MASK_OPTIONS = {'legacy': (), 'exact': ('--first_mask', 'exact', '--mask_ties', 'exact')}
# The legacy campaigns 10x25_blindref and 10x25_cpuref exist because the stored
# arrays of one comparison had to come from one command on one device. In the
# exact-mask environment every array is produced on the CPU with one torch thread
# by one campaign (scripts/run_eval_newseeds.sh), so these folders are the 10x25
# campaign itself: res_dir('10x25_blindref') is 10x25_xmask there.
FOLDER_ALIAS = {'legacy': {}, 'exact': {'10x25_blindref': '10x25', '10x25_cpuref': '10x25'}}
# The --tag a legacy campaign was evaluated with (eval_family.py appends it to the
# summary file name, not to the arrays); every exact-mask campaign uses _xmask.
LEGACY_TAG = {'10x25_cpuref': '_cpuref', '10x25_blindref': '_blindref',
              '10x25_firstmask': '_firstmask', '10x25_firstmask_off': '_firstmask_off',
              '10x25_masktol': '_masktol', '80x25_masktol': '_masktol',
              '50x25_pactwarm': '_pactwarm', '80x25_pactwarm': '_pactwarm'}

# Further arms trained under the exact mask only, seeds 301-303, named
# <stem>-s<seed>; checkpoint 10x25+family+<name>, log train_log/FAMILY_<name>.log.
# (recipe, stored legacy counterpart or None). The counterpart's stored files
# define the evaluation set (scripts/eval_newseeds_plan.py, SECONDARY_EVAL).
SECONDARY_ARMS = {
    'xmask-eq': ('released recipe, 6000 updates', 'jointblind-eq'),
    'xmask-wthird': ('W specialist, 333 updates', 'spec-W-third'),
    'xmask-sparse': ('released recipe, --reward_mode sparse', None),
    'xmask-dzero': ('released recipe, --delta_mode zero (the legacy arm used the '
                    'descriptor-input variant, use_film True)', 'joint-deltazero'),
    'xmask-dmax': ('released recipe, --delta_mode max (the legacy arm used the '
                   'descriptor-input variant, use_film True)', 'joint-deltamax'),
    'xmask-film': ('descriptor-input variant, --use_film True', 'joint-v1'),
    'xmask-dr': ('randomized ranges on the released architecture (the legacy arm '
                 'used the descriptor-input variant with --dr_shift)', 'joint-dr'),
    # the reference without a held-out combination, trained on every combination
    'xmask-all8': ('released recipe on all eight combinations, --family_regimes '
                   'N,L,S,W,LS,LW,SW,LSW, 2000 updates (the legacy arm used the '
                   'descriptor-input variant, use_film True)', 'joint-all8'),
    # two more channels per operation-machine pair, the exact setup the assignment would
    # incur now and the delay a machine unavailability window adds to its start
    # (32914 parameters)
    'xmask-pair': ('released recipe, --pair_extra sw', 'jointpair'),
    # the Giffler-Thompson mask for active schedules: a pair stays legal while its
    # processing start is strictly earlier than the earliest completion among the
    # compatible pairs, so the policy may hold a machine idle for a later job; under
    # --first_mask exact the first decision uses the same condition (32786 parameters).
    # eval_family.py restores mask_mode from the run's configuration, so the arm is
    # evaluated under its own mask (scripts/p50_actmask_analysis.py reads it)
    'xmask-active': ('released recipe, --mask_mode active', 'jointact'),
}
SECONDARY_SEEDS = (301, 302, 303)

# Off-combination evaluation of the exact-mask specialists (scripts/eval_newseeds_plan.py,
# group cross): every specialist under each combination it was not trained for, at 10x25.
# The arrays go to res_dir(CROSS_FOLDER) = <results>/10x25_cross_xmask (eval_family.py
# --tag _cross_xmask), because 19 of the 56 off-combination arrays of a seed carry the
# names of arrays of the 10x25 campaign (SPEC_EVAL_10x25 and the LSW specialist on all
# eight combinations), and a job whose files partly exist is refused by the queue.
CROSS_FOLDER = '10x25_cross'
EXACT_TAG = {CROSS_FOLDER: '_cross_xmask'}    # summary tag of an exact folder, default _xmask

# The public-instance family (scripts/run_pubfamily_queue.sh, MASK=exact): models
# 10x5+pubfamily+<name>, checkpoints under trained_network/FAMILY, logs
# train_log/PUBFAMILY_<name>.log, seeds 301-305; evaluated with --base record on the
# stored test records, results under test_results/PUBFAMILY/<set>_xmask.
PUB_PREFIX = '10x5+pubfamily+'
PUB_STEM = {'released': 'pubx-v1', 'control': 'pubx-ctl', 'specialist': 'pubx-solo-{R}'}
PUB_SEEDS = (301, 302, 303, 304, 305)
# Equal-experience arm of the public family (scripts/run_pubfamily_queue.sh ARMS=eq):
# the released recipe (six combinations) with PUB_EQ_UPDATES updates instead of 2000,
# so that each combination gets as many episodes as a specialist (1000 updates) gets
# of its own. Names pubx-eq-s<seed>, seeds PUB_SEEDS. It is not a principal arm:
# PUB_STEM, pub_model and every loop over the principal arms leave it out.
PUB_EQ_STEM = 'pubx-eq'
PUB_EQ_UPDATES = 6000
# Active-rule arm of the public family (scripts/run_pubfamily_queue.sh ARMS=act): the
# released recipe (six combinations, 2000 updates) trained under the admissibility rule
# PUB_ACT_RULE (--mask_mode active, the Giffler-Thompson condition for active schedules)
# instead of the non-delay rule of every other public run. PUB_RULE_WORD gives the word
# of the run names for each rule, so the arm of a further rule is named
# pubx-<word>-s<seed>. Names pubx-act-s<seed>, seeds PUB_ACT_SEEDS. It is not a
# principal arm: PUB_STEM, pub_model and every loop over the principal arms leave it out.
PUB_RULE_WORD = {'active': 'act'}
PUB_ACT_RULE = 'active'
PUB_ACT_STEM = f'pubx-{PUB_RULE_WORD[PUB_ACT_RULE]}'
PUB_ACT_SEEDS = (301, 302, 303)
# The whole public family under the rule of a rule arm (scripts/run_pubfamily_queue.sh
# ARMS=<word>fam and ARMS=<word>eq): each recipe trained under --mask_mode <rule> and
# nothing else changed. PUB_RULE_STEM gives the stem of each recipe under the rule word W:
# W follows the recipe word of the non-delay stem without a separator (pubx-ctl ->
# pubx-ctlW, pubx-solo-<R> -> pubx-soloW-<R>, pubx-eq -> pubx-eqW), so the globs that read
# the files of another arm (pubx-act-* of p108, pubx-eq-* of p89) take none of these runs;
# the released recipe drops its word v1 and keeps the name of the rule arm (pubx-W), so
# pubx-act-s301..303 (PUB_ACT_*) are the policy runs of seeds 301 to 303 of the family.
# Under the active rule the family is trained on PUB_ACTFAM_SEEDS, and the
# equal-experience recipe on the seeds with a finished pubx-eq run (the queue reads them
# from the stored files). Not principal arms: PUB_STEM, pub_model and every loop over the
# principal arms leave them out.
PUB_RULE_STEM = {'released': 'pubx-{W}', 'control': 'pubx-ctl{W}',
                 'specialist': 'pubx-solo{W}-{R}', 'eq': 'pubx-eq{W}'}
PUB_ACTFAM_SEEDS = (301, 302, 303, 304, 305)
PUB_RESULTS = 'test_results/PUBFAMILY'
PUB_SETS = {  # test set -> best-known table (regime N gap column of the summary)
    'pub_SD2_test_10x5+mix': 'data/PUBLIC/SD2/best_known_test_10x5+mix.csv',
    'pub_SD2_test_20x10+mix': 'data/PUBLIC/SD2/best_known_test_20x10+mix.csv',
    'pub_Brandimarte': 'data/PUBLIC/Brandimarte/best_known.csv',
    'pub_Hurink_edata': 'data/PUBLIC/Hurink_edata/best_known.csv',
    'pub_Hurink_rdata': 'data/PUBLIC/Hurink_rdata/best_known.csv',
    'pub_Hurink_vdata': 'data/PUBLIC/Hurink_vdata/best_known.csv',
}
# Sensitivity sets of the public family, evaluation only (the stored policies,
# no retraining), kept apart from PUB_SETS so that nothing that loops over the six
# test sets changes. Each re-draws the records of a test set with one rule
# parameter changed (constraint_family.public_rules.PublicRuleParams), from the
# same descriptor seeds: set -> (test set it re-draws, best-known table, change).
# lag_scale also moves the horizon H (the lag-inclusive job-chain bound), so the
# outage calendars of W and SW change with it; N and S differ only in the
# horizon field, which no N or S result reads.
PUB_SENS_SETS = {
    'pub_SD2_test_10x5+mix_lag050': ('pub_SD2_test_10x5+mix',
                                     'data/PUBLIC/SD2/best_known_test_10x5+mix.csv',
                                     {'lag_scale': 0.5}),
}

# Combinations each specialist is evaluated on in the 10x25 campaign: its own and
# every combination with one more rule (the stale-coverage reading of p54).
SPEC_EVAL_10x25 = {'N': ('N', 'L', 'S', 'W'), 'L': ('L', 'LS', 'LW'),
                   'S': ('S', 'LS', 'SW'), 'W': ('W', 'LW', 'SW'),
                   'LS': ('LS', 'LSW'), 'LW': ('LW', 'LSW'), 'SW': ('SW', 'LSW'),
                   'LSW': ('LSW',)}

# Arms that are not principal: stored in the legacy environment only, never
# retrained. (stem, seeds on disk, seed-301 run without suffix)
OTHER_ARMS = {
    'joint-v1': ((301, 302, 303), True),
    'joint-nlsw': ((301, 302, 303), True),
    'spec-W-third': ((301, 302, 303), True),
    'joint-v2': ((301,), True),
    'jointact': ((301, 302, 303), False),
    'joint-all8': ((301, 302, 303), False),
    'joint-all8e': ((301, 302, 303), False),
    'jointblind-eq': ((301, 302, 303), False),
    'jointblind-mb': ((301,), False),
    'joint-concat': ((301, 302, 303), False),
    'joint-deltamax': ((301, 302, 303), False),
    'joint-deltazero': ((301, 302, 303), False),
    'joint-dr': ((301, 302, 303), False),
    'jointpair': ((301, 302, 303), False),
    'jointpairfilm': ((301, 302, 303), False),
}

SEED_WORDS = {301: 'One', 302: 'Two', 303: 'Three', 304: 'Four', 305: 'Five'}


class MissingResult(FileNotFoundError):
    """A result file of a requested seed is not on disk."""


class NotInEnvironment(KeyError):
    """An arm that does not exist in the active environment was requested."""


# ----------------------------------------------------------------- active setting
def _pop_cli():
    for flag, var in (('--pact-env', 'PACT_ENV'), ('--pact-seeds', 'PACT_SEEDS'),
                      ('--pact-out', 'PACT_OUT'), ('--pact-results', 'PACT_RESULTS')):
        while flag in sys.argv:
            i = sys.argv.index(flag)
            if i + 1 >= len(sys.argv):
                raise SystemExit(f'{flag} needs a value')
            os.environ[var] = sys.argv[i + 1]
            del sys.argv[i:i + 2]


_pop_cli()


def parse_seeds(text):
    """'301,302,303' or '301-305' or '301-303,305' -> (301, 302, 303, ...)."""
    out = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if not out or len(set(out)) != len(out) or out != sorted(out):
        raise SystemExit(f'REFUSING: seed list {text!r} must be non-empty, increasing '
                         f'and without repeats')
    return tuple(out)


def env():
    e = os.environ.get('PACT_ENV', DEFAULT_ENV)
    if e not in ENVIRONMENTS:
        raise SystemExit(f'REFUSING: PACT_ENV={e!r}, want one of {ENVIRONMENTS}')
    return e


def seeds():
    s = parse_seeds(os.environ['PACT_SEEDS']) if os.environ.get('PACT_SEEDS') \
        else DEFAULT_SEEDS
    bad = [x for x in s if x not in PLANNED_SEEDS[env()]]
    if bad:
        raise SystemExit(f'REFUSING: seeds {bad} are not trained in the {env()} environment '
                         f'(planned: {PLANNED_SEEDS[env()]})')
    return s


def is_default():
    return env() == DEFAULT_ENV and seeds() == DEFAULT_SEEDS


def describe():
    s = seeds()
    return (f'environment {env()}, training seeds {",".join(map(str, s))} '
            f'({len(s)} runs per arm)')


def seed_words():
    return [SEED_WORDS[s] for s in seeds()]


def results_root():
    return os.environ.get('PACT_RESULTS', 'test_results/FAMILY')


def out(path):
    """Where a generated file (paper/..., results/...) is written and read back."""
    root = os.environ.get('PACT_OUT')
    if root is None:
        if not is_default():
            raise SystemExit(
                f'REFUSING: {describe()} is not the setting of the submitted analysis, '
                f'and PACT_OUT is not set, so {path} would be overwritten. Set '
                f'PACT_OUT=<folder> (or PACT_OUT=. to write in place).')
        return path
    if root in ('.', ''):
        return path
    p = os.path.join(root, path)
    os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
    return p


# ----------------------------------------------------------------- names
def _check_arm(arm, combo):
    if arm not in ARMS:
        raise NotInEnvironment(f'unknown principal arm {arm!r}; want one of {ARMS}')
    if arm == 'specialist':
        if combo not in COMBOS:
            raise ValueError(f'a specialist needs a combination from {COMBOS}, got {combo!r}')
    elif combo not in (None,):
        raise ValueError(f'arm {arm} takes no combination (got {combo!r})')


def model(arm, combo=None, seed=None, environment=None):
    """Run name, e.g. ('released', None, 302) -> 'jointblind-v1-s302'."""
    e = environment or env()
    _check_arm(arm, combo)
    if seed is None:
        raise ValueError('a run needs a seed')
    seed = int(seed)
    if seed not in PLANNED_SEEDS[e]:
        raise ValueError(f'seed {seed} is not planned in the {e} environment')
    stem = _STEM[e][arm].format(R=combo)
    if seed == 301 and arm in _BARE_301[e]:
        return stem
    return f'{stem}-s{seed}'


def suffix(arm, seed, environment=None):
    """The part of the run name after the stem ('' or '-s302')."""
    e = environment or env()
    return '' if (int(seed) == 301 and arm in _BARE_301[e]) else f'-s{int(seed)}'


def suffixes(arm):
    """Suffixes of the active seeds in seed order, e.g. ['', '-s302', '-s303']."""
    return [suffix(arm, s) for s in seeds()]


def tag(arm, combo=None, seed=None, method='greedy', environment=None):
    """Method tag inside Result_<tag>_<R>.npy."""
    return f'{method}+{PREFIX}{model(arm, combo, seed, environment)}'


def tags(arm, combo=None, method='greedy'):
    return [tag(arm, combo, s, method) for s in seeds()]


def checkpoint(arm, combo=None, seed=None, environment=None):
    return f'trained_network/FAMILY/{PREFIX}{model(arm, combo, seed, environment)}.pth'


def config_path(arm, combo=None, seed=None, environment=None):
    return f'train_log/FAMILY/config_{PREFIX}{model(arm, combo, seed, environment)}.json'


def train_log(arm, combo=None, seed=None, environment=None):
    return f'train_log/FAMILY_{model(arm, combo, seed, environment)}.log'


def vali_log(arm, combo=None, seed=None, environment=None):
    return f'train_log/FAMILY/valiquality_{PREFIX}{model(arm, combo, seed, environment)}.txt'


def schedule_cache(seed, size, environment=None):
    """Folder of the released policy's exported schedules at one production size
    (eval_family.py --save_schedules), or_solution/FAMILY/<name>/<R>/<instance>.npz.
    Stored legacy names: pactwarm_cache_<size> (seed 301), pactwarm_cache_s302_<size>,
    pactwarm_cache_s303_<size>; the exact-mask environment adds _xmask after the
    prefix and names every seed (pactwarm_cache_xmask_s301_<size>)."""
    e = environment or env()
    model('released', None, seed, e)                 # validates the seed
    if e == 'legacy':
        return (f'or_solution/FAMILY/pactwarm_cache_{size}' if int(seed) == 301
                else f'or_solution/FAMILY/pactwarm_cache_s{int(seed)}_{size}')
    return f'or_solution/FAMILY/pactwarm_cache_xmask_s{int(seed)}_{size}'


# ----------------------------------------------------------------- other arms
def other_model(stem, seed):
    if stem not in OTHER_ARMS:
        raise NotInEnvironment(f'{stem!r} is not a registered arm')
    seeds_on_disk, bare = OTHER_ARMS[stem]
    if int(seed) not in seeds_on_disk:
        raise ValueError(f'{stem} has no seed {seed} (stored: {seeds_on_disk})')
    return stem if (bare and int(seed) == 301) else f'{stem}-s{int(seed)}'


def secondary_model(stem, seed):
    """Run name of a further exact-mask arm, e.g. ('xmask-eq', 301) -> 'xmask-eq-s301'."""
    if stem not in SECONDARY_ARMS:
        raise NotInEnvironment(f'{stem!r} is not a registered exact-mask arm')
    if int(seed) not in SECONDARY_SEEDS:
        raise ValueError(f'{stem} is trained on seeds {SECONDARY_SEEDS}, not {seed}')
    return f'{stem}-s{int(seed)}'


def secondary_paths(stem, seed):
    """(checkpoint, training log) of a further exact-mask arm."""
    m = secondary_model(stem, seed)
    return f'trained_network/FAMILY/{PREFIX}{m}.pth', f'train_log/FAMILY_{m}.log'


def pub_model(arm, combo=None, seed=None):
    """Run name of the public-instance family, e.g. ('specialist', 'LS', 302) ->
    'pubx-solo-LS-s302'; the model is PUB_PREFIX + name."""
    _check_arm(arm, combo)
    if int(seed) not in PUB_SEEDS:
        raise ValueError(f'the public family is trained on seeds {PUB_SEEDS}, not {seed}')
    return f'{PUB_STEM[arm].format(R=combo)}-s{int(seed)}'


def pub_paths(arm, combo=None, seed=None):
    """(checkpoint, training log) of a public-family run."""
    m = pub_model(arm, combo, seed)
    return f'trained_network/FAMILY/{PUB_PREFIX}{m}.pth', f'train_log/PUBFAMILY_{m}.log'


def pub_vali_log(arm, combo=None, seed=None):
    """Validation record of a public-family run (one score per validation)."""
    return f'train_log/FAMILY/valiquality_{PUB_PREFIX}{pub_model(arm, combo, seed)}.txt'


def pub_eq_model(seed):
    """Run name of the equal-experience arm of the public family, e.g. 302 ->
    'pubx-eq-s302'; the model is PUB_PREFIX + name."""
    if int(seed) not in PUB_SEEDS:
        raise ValueError(f'{PUB_EQ_STEM} is trained on seeds {PUB_SEEDS}, not {seed}')
    return f'{PUB_EQ_STEM}-s{int(seed)}'


def pub_eq_paths(seed):
    """(checkpoint, training log, validation record, configuration) of an
    equal-experience run of the public family."""
    m = pub_eq_model(seed)
    return (f'trained_network/FAMILY/{PUB_PREFIX}{m}.pth', f'train_log/PUBFAMILY_{m}.log',
            f'train_log/FAMILY/valiquality_{PUB_PREFIX}{m}.txt',
            f'train_log/FAMILY/config_{PUB_PREFIX}{m}.json')


def pub_act_model(seed):
    """Run name of the active-rule arm of the public family, e.g. 302 ->
    'pubx-act-s302'; the model is PUB_PREFIX + name."""
    if int(seed) not in PUB_ACT_SEEDS:
        raise ValueError(f'{PUB_ACT_STEM} is trained on seeds {PUB_ACT_SEEDS}, not {seed}')
    return f'{PUB_ACT_STEM}-s{int(seed)}'


def pub_act_paths(seed):
    """(checkpoint, training log, validation record, configuration) of an active-rule
    run of the public family."""
    m = pub_act_model(seed)
    return (f'trained_network/FAMILY/{PUB_PREFIX}{m}.pth', f'train_log/PUBFAMILY_{m}.log',
            f'train_log/FAMILY/valiquality_{PUB_PREFIX}{m}.txt',
            f'train_log/FAMILY/config_{PUB_PREFIX}{m}.json')


def pub_rule_model(arm, combo=None, seed=None, word=None):
    """Run name of a recipe of the public family under the rule of a rule word (default
    the word of PUB_ACT_RULE), e.g. ('control', None, 302) -> 'pubx-ctlact-s302',
    ('specialist', 'LS', 301) -> 'pubx-soloact-LS-s301', ('eq', None, 303) ->
    'pubx-eqact-s303', ('released', None, 304) -> 'pubx-act-s304'; the model is
    PUB_PREFIX + name. arm is a principal arm or 'eq' (the equal-experience recipe)."""
    word = PUB_RULE_WORD[PUB_ACT_RULE] if word is None else word
    if word not in PUB_RULE_WORD.values():
        raise ValueError(f'unknown rule word {word!r}; want one of {tuple(PUB_RULE_WORD.values())}')
    if arm not in PUB_RULE_STEM:
        raise NotInEnvironment(f'unknown recipe {arm!r}; want one of {tuple(PUB_RULE_STEM)}')
    if arm == 'specialist':
        if combo not in COMBOS:
            raise ValueError(f'a specialist needs a combination from {COMBOS}, got {combo!r}')
    elif combo is not None:
        raise ValueError(f'recipe {arm} takes no combination (got {combo!r})')
    if seed is None or int(seed) not in PUB_ACTFAM_SEEDS:
        raise ValueError(f'the public family under a rule is trained on seeds {PUB_ACTFAM_SEEDS}, '
                         f'not {seed}')
    return f'{PUB_RULE_STEM[arm].format(W=word, R=combo)}-s{int(seed)}'


def pub_rule_paths(arm, combo=None, seed=None, word=None):
    """(checkpoint, training log, validation record, configuration) of a run of the
    public family under a rule."""
    m = pub_rule_model(arm, combo, seed, word)
    return (f'trained_network/FAMILY/{PUB_PREFIX}{m}.pth', f'train_log/PUBFAMILY_{m}.log',
            f'train_log/FAMILY/valiquality_{PUB_PREFIX}{m}.txt',
            f'train_log/FAMILY/config_{PUB_PREFIX}{m}.json')


def pub_rule_counterpart(arm, combo=None, seed=None):
    """The non-delay run whose recipe a run of the family under a rule repeats: the
    principal run of the same arm, combination and seed, or pubx-eq-s<seed>."""
    if arm == 'eq':
        return pub_eq_model(seed)
    return pub_model(arm, combo, seed)


def pub_validation_record(arm, combo=None, seed=None, root='.'):
    """validation_record of a public-family run; paths resolved against root."""
    lp = pub_paths(arm, combo, seed)[1]
    return validation_record(arm, combo, seed,
                             paths=(os.path.join(root, pub_vali_log(arm, combo, seed)),
                                    os.path.join(root, lp)))


def other_tag(stem, seed, method='greedy'):
    return f'{method}+{PREFIX}{other_model(stem, seed)}'


def other_seeds(stem):
    """Active seeds on which a non-principal arm exists (legacy only)."""
    if stem not in OTHER_ARMS:
        raise NotInEnvironment(f'{stem!r} is not a registered arm')
    if env() != 'legacy':
        raise NotInEnvironment(f'{stem} exists only in the legacy environment')
    s = tuple(x for x in seeds() if x in OTHER_ARMS[stem][0])
    if not s:
        raise NotInEnvironment(f'{stem} has none of the active seeds {seeds()}')
    return s


def common_seeds(*stems):
    """Active seeds shared by the principal arms and the named other arms: the
    seed set a paired comparison against a non-principal arm uses."""
    s = set(seeds())
    for st in stems:
        s &= set(other_seeds(st))
    if not s:
        raise NotInEnvironment(f'no active seed is shared by {stems}')
    return tuple(sorted(s))


def legacy_only(what):
    """True when a legacy-only comparison must be skipped in this environment."""
    if env() == 'legacy':
        return False
    print(f'[arm_registry] skipped in the {env()} environment: {what} '
          f"(legacy-only arm)", file=sys.stderr)
    return True


# ----------------------------------------------------------------- folders
def res_dir(folder='10x25'):
    """Campaign folder of the principal arms' evaluations (FOLDER_ALIAS applies in
    the exact-mask environment)."""
    return f'{results_root()}/{FOLDER_ALIAS[env()].get(folder, folder)}{FOLDER_SUFFIX[env()]}'


def summary_tag(folder='10x25'):
    """The --tag eval_family.py appended to the summary files of a campaign folder
    (the arrays carry no tag): the legacy campaign tag, or _xmask (_cross_xmask for
    CROSS_FOLDER)."""
    if env() == 'legacy':
        return LEGACY_TAG.get(folder, '')
    return EXACT_TAG.get(folder, FOLDER_SUFFIX[env()])


def rule_dir(folder='10x25'):
    """Folder of the dispatching rules evaluated under the active mask."""
    return res_dir(folder)


def fixed_dir(folder='10x25'):
    """Folder of results that do not depend on the mask (solver, GA, legacy-only arms)."""
    return f'{results_root()}/{folder}'


def result_path(arm, combo, seed, folder, regime, method='greedy'):
    return f'{res_dir(folder)}/Result_{tag(arm, combo, seed, method)}_{regime}.npy'


def _read(p, n=None):
    if not os.path.exists(p):
        raise MissingResult(f'missing result file for a requested seed: {p} ({describe()})')
    a = np.load(p)
    a = (a[:, 0] if a.ndim > 1 else a).astype(float)
    return a if n is None else a[:n]


def read_path(p, n=None):
    """Makespan column of a result file; MissingResult if absent."""
    return _read(p, n)


def load(arm, combo, seed, folder, regime, n=None):
    """Makespans of one run on one combination (vector over instances)."""
    return _read(result_path(arm, combo, seed, folder, regime), n)


def load_other(stem, seed, folder, regime, n=None, directory=None):
    d = directory or fixed_dir(folder)
    return _read(f'{d}/Result_{other_tag(stem, seed)}_{regime}.npy', n)


def stack(arm, combo, folder, regime, n=None, seed_list=None):
    """[k seeds, n instances] in seed order."""
    return np.stack([load(arm, combo, s, folder, regime, n)
                     for s in (seed_list or seeds())])


def pooled(arm, combo, folder, regime, n=None, seed_list=None):
    """Per-instance mean over the active seeds, np.mean over the list in seed order
    (the estimator every published script uses)."""
    return np.mean([load(arm, combo, s, folder, regime, n)
                    for s in (seed_list or seeds())], axis=0)


def require_all(arm, regimes, folder, own_combo=False):
    """Raise MissingResult naming every missing file of the active seeds. For the
    specialist arm with own_combo=True, each regime R is read from spec-R."""
    miss = []
    for r in regimes:
        for s in seeds():
            p = result_path(arm, r if (arm == 'specialist' and own_combo) else None,
                            s, folder, r)
            if not os.path.exists(p):
                miss.append(p)
    if miss:
        raise MissingResult('missing result files for requested seeds:\n  ' +
                            '\n  '.join(miss))


# ----------------------------------------------------------------- model selection
_NORMS = re.compile(r'\[vali\] MWKR norms: ([^\n]*)')


def validation_record(arm, combo=None, seed=None, paths=None):
    """(best validation score, update index of the kept checkpoint, MWKR norms).

    The trainer validates every validate_timestep updates and saves the
    checkpoint whenever the score `rel` (mean over the trained combinations of the
    greedy validation makespan divided by that combination's MWKR norm; lower is
    better) improves, so the kept checkpoint is the one at the minimum of
    valiquality_<model>.txt. The minimum is cross-checked against the training
    log's closing 'best rel=' line. paths=(validation record, training log)
    reads the files of another family (pub_validation_record); by default the
    principal arm's files are read."""
    import ast
    vp, lp = paths if paths else (vali_log(arm, combo, seed), train_log(arm, combo, seed))
    for p in (vp, lp):
        if not os.path.exists(p):
            raise MissingResult(f'missing training record for a requested seed: {p}')
    vals = ast.literal_eval(open(vp).read())
    best = float(min(vals))
    txt = open(lp, errors='replace').read()
    m = re.findall(r'best rel=([\d.]+)', txt)
    if not m or abs(float(m[-1]) - best) > 5e-5:
        raise SystemExit(f'REFUSING: {lp} reports best rel={m[-1] if m else None}, '
                         f'{vp} has minimum {best:.6f}')
    nm = _NORMS.search(txt)
    norms = dict(re.findall(r'\b([A-Z]+)=([\d.]+)', nm.group(1))) if nm else {}
    return best, int(np.argmin(vals)), norms


def select_seed(arm='released', combo=None, seed_list=None, record=None):
    """The selection rule fixed before the new runs: among the active seeds, the
    training seed whose kept checkpoint has the best (lowest) validation score.
    Ties go to the lower seed. Refuses unless every seed's validation norms are
    identical, since otherwise the scores are not on one scale.
    seed_list and record (a function like validation_record) apply the same
    rule to another seed set or family, e.g. PUB_SEEDS with
    pub_validation_record; by default the active seeds of the principal arm.
    Returns (seed, {seed: (score, update_index)})."""
    seed_list = tuple(seed_list) if seed_list else seeds()
    record = record or validation_record
    recs = {s: record(arm, combo, s) for s in seed_list}
    norms = {s: r[2] for s, r in recs.items()}
    first = norms[seed_list[0]]
    for s, nm in norms.items():
        if nm != first:
            raise SystemExit(f'REFUSING: validation norms of seed {s} {nm} differ from '
                             f'seed {seed_list[0]} {first}; scores are not comparable')
    best = min(seed_list, key=lambda s: (recs[s][0], s))
    return best, {s: (recs[s][0], recs[s][1]) for s in seed_list}


def refuse_on_missing(*paths):
    """If a MissingResult escapes the script, remove these generated files (as
    macro_io.refuse does) so no stale number survives, and exit non-zero."""
    prev = sys.excepthook

    def hook(t, v, tb):
        if isinstance(v, MissingResult):
            for p in paths:
                if p and os.path.exists(p):
                    os.remove(p)
                    print(f'removed {p} (refusing to certify)')
            sys.stderr.write(f'*** REFUSING TO CERTIFY ***\n{v}\n')
            return
        prev(t, v, tb)
    sys.excepthook = hook


def banner():
    """One line on stderr (stdout of the scripts stays byte-identical)."""
    print(f'[arm_registry] {describe()}; results {results_root()}; '
          f'outputs {os.environ.get("PACT_OUT", "in place (default setting)")}',
          file=sys.stderr)
