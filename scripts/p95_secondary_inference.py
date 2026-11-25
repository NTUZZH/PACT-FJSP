r"""P95: paired inference for the supporting training arms and for the published
learned scheduler, in the exact-mask environment.

The supporting arms (arm_registry.SECONDARY_ARMS, training seeds SECONDARY_SEEDS)
are evaluated by scripts/run_eval_newseeds.sh (plan scripts/eval_newseeds_plan.py,
group secondary) and the HGNN port of Song et al. (backbones/hgnn, trained by
scripts/run_hgnn_exact_queue.sh) by backbones/hgnn/eval_family_hgnn.py --tag _xmask
(plan group hgnn), greedily, on the test instances of every combination at the
training size (10x25).
This script reads those arrays together with the arrays of the policy (the released
recipe) and of the specialists, and writes new outputs only (every macro name starts
with Sec):

  manuscript/macros_secondary.tex   results/secondary_inference.json
  results/secondary_inference.md

SIGN AND PAIRING
  d = 100 (a - b) / b per training seed and test instance, a the first-named method,
  b the second; negative d means that the first-named method has the shorter
  schedule. Seed s of a meets seed s of b, on the seeds both arms have (the seed
  number, not the position in a list), and instance i of every array is base
  instance i (checked on the descriptors of the test set).

ESTIMATORS (scripts/stats_family.py, the conventions of p88 and p89)
  * k >= 2 seed pairs: crossed hierarchical bootstrap over seeds and base instances,
    B = --boot (20,000), a fresh default_rng(11) per quantity; a pooled row averages
    the combinations named, and a drawn base instance brings all of them along.
    The seed-level Student t interval (k - 1 df) goes to the JSON and the Markdown.
  * k == 1 (one training run, the HGNN specialists): bootstrap over base instances
    alone, a fresh default_rng(7) per quantity, again with all combinations of a
    drawn base instance.
  Verdicts (stats_family.verdict) at the margins 0.5% and 1% and the direction
  (stats_family.direction) come from the unrounded interval; the printed numbers
  use p89_public_family.f2. Joint claims over a family of combinations use the
  intersection-union test (stats_family.iut). The comparison of the policy with
  HGNN uses the combined superiority verdict of p88 and p89
  (p89_public_family.superiority_verdict): the hierarchical interval below zero and
  the Holm-adjusted one-sided p-values (seed-level t and bootstrap, Holm over the
  eight combinations) below 0.05 for 'ahead', the mirrored condition with its own
  Holm family for 'behind', 'not distinguishable' otherwise.

BLOCKS
  policy    the policy against the specialist of each combination, on the seeds of
            the supporting arms, per combination and pooled over all eight, the six
            training and the two held-out combinations, with the joint claims. It is
            the reference for the verdict comparison below; the five-seed analysis is
            scripts/p88_family_inference.py.
  eq, sparse, dzero, dmax, film, dr, all8, pair
            the arms that follow the policy's training setting: against the policy
            and against the specialist of each combination, per combination of the
            arm's 10x25 evaluation set (eval_newseeds_plan.SECONDARY_EVAL) and pooled,
            with the joint claims against the specialists and the combinations whose
            verdict against the specialist differs from that of the policy block.
            all8 trains on all eight combinations, and its pooled rows Train and Held
            keep the policy's split into the six combinations the policy trains on and
            the pair SW and LSW. pair reads the exact setup of each operation-machine
            pair and the delay a window adds to its start. Each run of the two arms must
            record its own option (ARM_CONFIG), and their macro words are AllEight and
            Pair (MACRO_WORD).
  wthird    the W specialist with the policy's experience of one combination:
            against the W specialist and against the policy on every combination the
            plan evaluates it on (W, and any combination with one more rule), pooled
            over them when there is more than one.
  hgnn      the policy against HGNN trained with the PACT reward (configuration value
            hgnn_reward certified) at the learning rate chosen by
            scripts/hgnn_select_lr.py (or --hgnn-lr), per combination with the combined
            superiority verdict, and pooled. Seeds 301-305 when the runs of all five
            exist, 301-303 when only those exist (scripts/run_hgnn_exact_queue.sh,
            PHASE=tune, panel and panel2); runs of any other set of seeds refuse the
            block. HGNN seed s meets policy seed s.
  hgnnpub   the same against HGNN trained with the reward of the released code, seeds
            301-303.
  hgnnsolo  optional: HGNN (PACT reward) against the HGNN specialist of each
            combination (seed 301 only, so the interval is the one-run interval).
  Each block also reports the training effort its configurations record (updates
  or iterations, episodes per update, training combinations, episodes per
  combination in expectation); no wall-clock value is read.
  The arm with the active-schedule mask (xmask-active) is analysed by
  scripts/p50_actmask_analysis.py (ELSEWHERE); no block here reads it.

  HGNN runs are named hgnn-<reward>-<joint|R>-s<seed>-xmask[-lr<rate>] (prefix
  10x25+family+, scripts/run_hgnn_exact_queue.sh); their arrays are
  Result_<name>_<R>.npy in the exact 10x25 folder, the layout of
  backbones/hgnn/eval_family_hgnn.py.

CHECKS AND BLOCK STATUS
  Before anything is computed, the descriptors of the 10x25 test set must give every
  combination the same file names and base instances. A block is
    skipped   when no result array of its own arm exists (listed with the reason);
    refused   when some exist but the block is incomplete or inconsistent: a missing
              array or seed, an array whose row count is not the number of test
              instances or whose makespans are not finite and positive, an array
              older than its checkpoint, a missing checkpoint, configuration or
              training log, a log without its completion line, a configuration that
              records another seed or (exact environment) another mask, or seeds of
              one arm that disagree on the training effort;
    computed  otherwise.
  Refused blocks write no number and are listed; the script then exits with status
  1. --require all refuses (and leaves no output) unless every block except the
  optional hgnnsolo is computed; --require <block>[,<block>...] names the blocks.

CHECK ON THE EARLIER ENVIRONMENT (--earlier-check)
  Maps each supporting arm to its stored counterpart (the second entry of
  SECONDARY_ARMS; seeds from OTHER_ARMS), the policy and the specialists to their
  stored runs, and HGNN to its stored runs (earlier mask, 1000 iterations, released
  learning rate), and reads the earlier environment's arrays (the first of the 10x25
  campaign folders that holds every array of the arm, the policy and the specialists
  read from the same folder). Accepted only with --out outside the repository's
  paper and results folders; the outputs carry the suffix _check and every line is
  marked as a check. sparse has no counterpart and is not applicable.

Usage:
  python scripts/p95_secondary_inference.py [--boot 20000] [--require all]
         [--hgnn-lr 1e-4|2e-4|4e-4] [--out <root>] [--root <repository root>]
  python scripts/p95_secondary_inference.py --earlier-check --out <folder>
"""
import argparse
import contextlib
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import eval_newseeds_plan as PLAN  # noqa: E402
import hgnn_select_lr as HLR  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS, f2, jsonable, superiority_verdict  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = 'scripts/p95_secondary_inference.py'
R8 = list(REG.COMBOS)
TRAIN = list(REG.TRAIN_COMBOS)
HELD = list(REG.HELD_OUT)
FAMILIES = {'All': R8, 'Train': TRAIN, 'Held': HELD}
MARGINS = (0.5, 1.0)
MW = {0.5: 'Half', 1.0: 'One'}
ALPHA = 0.05
HIER_SEED, INST_SEED = 11, 7
FOLDER = '10x25'                 # the training size: the folder every block reads
RECIPE = ('xmask-eq', 'xmask-sparse', 'xmask-dzero', 'xmask-dmax', 'xmask-film', 'xmask-dr',
          'xmask-all8', 'xmask-pair')
WTHIRD = 'xmask-wthird'
WTHIRD_COMBO = 'W'               # the combination xmask-wthird trains on
# configuration values a run of the arm must record (checked like the mask options)
ARM_CONFIG = {WTHIRD: {'family_regimes': WTHIRD_COMBO},
              'xmask-all8': {'family_regimes': ','.join(R8)},
              'xmask-pair': {'pair_extra': 'sw'}}
# macro word of an arm whose run name is not letters after the prefix
MACRO_WORD = {'xmask-all8': 'AllEight'}
# arms of the registry that another script analyses; this script reads none of their files
ELSEWHERE = {'xmask-active': 'scripts/p50_actmask_analysis.py'}
# seed sets of the joint HGNN runs, the first whose runs are the runs that exist is read:
# PACT reward PHASE=tune (301), panel (302, 303) and panel2 (304, 305); published reward panel
HGNN_SEED_SETS = {'certified': ((301, 302, 303, 304, 305), (301, 302, 303)),
                  'published': ((301, 302, 303),)}
HGNN_SOLO_SEEDS = (301,)         # PHASE=solo
HGNN_REWARD = {'hgnn': 'certified', 'hgnnpub': 'published'}   # block -> configuration value
REWARD_WORD = {'certified': 'PACT', 'published': 'published'}   # configuration value -> text
OPTIONAL = ('hgnnsolo',)
OUT_FILES = ('manuscript/macros_secondary.tex', 'results/secondary_inference.json',
             'results/secondary_inference.md')
CHECK_SUFFIX = '_check'
GUARDED = ('results', 'test_results', 'or_solution', 'trained_network', 'train_log')


class Refusal(SystemExit):
    """A precondition failed; the outputs are removed and nothing is certified."""


def refuse(msg):
    raise Refusal(f'REFUSING: {msg}')


def word(stem):
    """'xmask-eq' -> 'Eq': the macro word of a supporting arm ('xmask-all8' -> 'AllEight')."""
    return MACRO_WORD.get(stem) or stem.split('-', 1)[1].capitalize()


def block_name(stem):
    """'xmask-eq' -> 'eq': the block name of a supporting arm."""
    return stem.split('-', 1)[1]


# ================================================================== layout
def _template(path, name):
    if path.count(name) != 1:
        refuse(f'cannot derive a path layout from {path!r} and {name!r}')
    return path.replace(name, '{m}')


_ANCHOR = (REG.SECONDARY_SEEDS[0], REG.model('released', None, REG.SECONDARY_SEEDS[0], 'exact'))
CKPT = _template(REG.checkpoint('released', None, _ANCHOR[0], 'exact'), _ANCHOR[1])
LOG = _template(REG.train_log('released', None, _ANCHOR[0], 'exact'), _ANCHOR[1])
CONFIG = _template(REG.config_path('released', None, _ANCHOR[0], 'exact'), _ANCHOR[1])


def check_layout():
    """The registry, the plan module and this script name the same arms, and the
    path layout derived from the registry gives the registry's own paths."""
    if set(ELSEWHERE) & (set(RECIPE) | {WTHIRD}):
        refuse(f'the arms {sorted(set(ELSEWHERE) & (set(RECIPE) | {WTHIRD}))} are analysed '
               f'here and by another script')
    if set(RECIPE) | {WTHIRD} | set(ELSEWHERE) != set(REG.SECONDARY_ARMS):
        refuse(f'the registry lists the arms {sorted(REG.SECONDARY_ARMS)}, this script '
               f'{sorted(set(RECIPE) | {WTHIRD})} and leaves {sorted(ELSEWHERE)} to other '
               f'scripts')
    if set(PLAN.SECONDARY_EVAL) != set(REG.SECONDARY_ARMS):
        refuse(f'the plan evaluates {sorted(PLAN.SECONDARY_EVAL)}, the registry lists '
               f'{sorted(REG.SECONDARY_ARMS)}')
    for stem in REG.SECONDARY_ARMS:
        if FOLDER not in dict(PLAN.SECONDARY_EVAL[stem]):
            refuse(f'the plan does not evaluate {stem} at {FOLDER}')
        m = REG.secondary_model(stem, REG.SECONDARY_SEEDS[0])
        if REG.secondary_paths(stem, REG.SECONDARY_SEEDS[0]) != (CKPT.format(m=m), LOG.format(m=m)):
            refuse(f'the checkpoint and log layout of {stem} differs from the registry')
    for e in REG.ENVIRONMENTS:
        for arm, combo in (('released', None), ('specialist', 'LS')):
            s = REG.PLANNED_SEEDS[e][-1]
            m = REG.model(arm, combo, s, e)
            if (REG.checkpoint(arm, combo, s, e), REG.train_log(arm, combo, s, e),
                    REG.config_path(arm, combo, s, e)) != (CKPT.format(m=m), LOG.format(m=m),
                                                           CONFIG.format(m=m)):
                refuse(f'the path layout of {m} differs from the registry')
    if hgnn_name('certified', 'joint', 301, HLR.run_name(HLR.DEFAULT_LR)[len(HLR.STEM):],
                 'exact') != HLR.run_name(HLR.DEFAULT_LR) or HLR.PREFIX != REG.PREFIX:
        refuse('the HGNN run names of this script differ from scripts/hgnn_select_lr.py')
    for reward, sets in HGNN_SEED_SETS.items():   # the runs the plan's group hgnn evaluates
        if tuple(sorted(set().union(*sets))) != tuple(PLAN.HGNN_JOINT_SEEDS[reward]):
            refuse(f'the HGNN seeds of this script ({sets}) differ from the plan '
                   f'({PLAN.HGNN_JOINT_SEEDS[reward]})')
        for rate in HLR.RATES:
            for part, s in (('joint', 305), ('LS', 301)):
                if hgnn_name(reward, part, s, lr_tag(rate), 'exact') != PLAN.hgnn_name(reward, part, s, rate):
                    refuse('the HGNN run names of this script differ from the plan')
    if tuple(HGNN_SOLO_SEEDS) != tuple(PLAN.HGNN_SOLO_SEEDS) or FOLDER != PLAN.HGNN_FOLDER:
        refuse('the HGNN specialists or folder of this script differ from the plan')


def hgnn_name(reward, part, seed, lr_tag, environment):
    """Run name of an HGNN run: exact environment hgnn-<reward>-<part>-s<seed>-xmask<lr_tag>
    (lr_tag '' for the released rate, '-lr<rate>' otherwise), earlier environment
    hgnn-<reward>-<part>-s<seed>."""
    base = f'hgnn-{reward}-{part}-s{int(seed)}'
    return base + '-xmask' + lr_tag if environment == 'exact' else base


def lr_tag(rate):
    return HLR.run_name(rate)[len(HLR.STEM):]


class Run:
    """One trained run: its name (the model is PREFIX + name), the trainer that wrote it
    ('family' for train_family.py, 'hgnn' for backbones/hgnn/train_family_hgnn.py), its
    training seed and the campaign folder its arrays are read from. The folder is
    resolved when the run is created (arm_registry.res_dir of the active environment)."""

    def __init__(self, name, kind, seed, folder):
        self.name, self.kind, self.seed, self.folder = name, kind, int(seed), folder
        self.dir = REG.res_dir(folder)

    @property
    def key(self):
        return (self.name, self.folder)

    @property
    def ckpt(self):
        return CKPT.format(m=self.name)

    @property
    def log(self):
        return LOG.format(m=self.name)

    @property
    def config(self):
        return CONFIG.format(m=self.name)

    def array(self, combo):
        if self.kind == 'family':
            return f'{self.dir}/Result_greedy+{REG.PREFIX}{self.name}_{combo}.npy'
        return f'{self.dir}/Result_{self.name}_{combo}.npy'


class Arm:
    """The runs of one arm, {seed: Run}."""

    def __init__(self, label, runs):
        self.label, self.runs = label, dict(runs)

    @property
    def seeds(self):
        return tuple(sorted(self.runs))


def principal(arm, combo, environment, folder):
    runs = {}
    for s in REG.PLANNED_SEEDS[environment]:
        r = Run(REG.model(arm, combo, s, environment), 'family', s, folder)
        if r.array('N') != REG.result_path(arm, combo, s, folder, 'N'):
            refuse(f'the array layout of {r.name} differs from the registry')
        runs[s] = r
    return Arm('policy' if arm == 'released' else f'specialist {combo}', runs)


def secondary(stem, environment, folder):
    if environment == 'exact':
        return Arm(stem, {s: Run(REG.secondary_model(stem, s), 'family', s, folder)
                          for s in REG.SECONDARY_SEEDS})
    cp = REG.SECONDARY_ARMS[stem][1]
    return Arm(cp, {s: Run(REG.other_model(cp, s), 'family', s, folder)
                    for s in REG.OTHER_ARMS[cp][0]})


def hgnn_arm(reward, part, seeds, tag, environment, folder):
    runs = {s: Run(hgnn_name(reward, part, s, tag, environment), 'hgnn', s, folder) for s in seeds}
    return Arm(runs[min(runs)].name.replace(f'-s{min(runs)}', ''), runs)


def common(*arms, within=None):
    """Seeds every arm has (and, with within, only those), in increasing order."""
    s = set(arms[0].runs)
    for a in arms[1:]:
        s &= set(a.runs)
    if within is not None:
        s &= set(within)
    return tuple(sorted(s))


def run_present(run, root):
    """True when any file of the run exists: checkpoint, training log, configuration or
    a result array of any combination."""
    paths = [run.ckpt, run.log, run.config] + [run.array(c) for c in R8]
    return any(os.path.exists(os.path.join(root, p)) for p in paths)


def hgnn_seeds(root, reward, tag, environment, folder):
    """(seeds, problem) of the joint HGNN runs of one reward. With one seed set, that set.
    Otherwise the set of HGNN_SEED_SETS[reward] whose runs are exactly the runs that
    exist (no file of another seed); the last set when no run exists (the block is then
    skipped for want of arrays); and when the existing runs match no set, the union of
    the sets with the problem that refuses the block."""
    sets = HGNN_SEED_SETS[reward]
    if len(sets) == 1:
        return tuple(sets[0]), None
    full = tuple(sorted(set().union(*sets)))
    have = tuple(s for s in full if run_present(
        Run(hgnn_name(reward, 'joint', s, tag, environment), 'hgnn', s, folder), root))
    if not have:
        return tuple(sets[-1]), None
    for st in sets:
        if have == tuple(st):
            return tuple(st), None
    return full, (f'HGNN ({REWARD_WORD[reward]} reward) has runs of seeds '
                  f'{",".join(map(str, have))}; the block reads seeds '
                  + ' or '.join(f'{st[0]}-{st[-1]}' for st in sets)
                  + ', all runs of one set and none of another seed')


# ================================================================== checks
def _relmsg(msg, root):
    return str(msg).replace(root.rstrip(os.sep) + os.sep, '')


def check_bases(root, data):
    """Every combination of the test set has the same file names and base instances,
    which are distinct; returns the number of instances (rows of every array)."""
    names, bases = {}, {}
    for r in R8:
        ps = sorted(glob.glob(os.path.join(root, data, 'desc', r, '*.desc.json')))
        names[r] = [os.path.basename(p) for p in ps]
        bases[r] = []
        for p in ps:
            with open(p) as f:
                bases[r].append(json.load(f)['base_instance'])
    for r in R8:
        if names[r] != names['N'] or bases[r] != bases['N']:
            refuse(f'{data}/desc/{r} does not share file names and base instances with N; '
                   f'the base-instance clusters are not defined')
    if not names['N'] or len(set(bases['N'])) != len(bases['N']):
        refuse(f'{data}: no test instances, or base instances that repeat')
    if [n[:-len('.desc.json')] for n in names['N']] != PLAN.instance_names(os.path.join(root, data)):
        refuse(f'{data}: the instance list differs from the plan module\'s')
    return len(names['N'])


def _get(cfg, dotted):
    x = cfg
    for k in dotted.split('.'):
        if not isinstance(x, dict) or k not in x:
            return '<absent>'
        x = x[k]
    return x


EFFORT_KEYS = {'family': ('max_updates', 'num_envs', 'family_regimes', 'seed_train'),
               'hgnn': ('args.max_iterations', 'args.num_envs', 'args.family_regimes', 'args.seed')}
RECIPE_KEYS = {'family': ('use_film', 'delta_mode', 'reward_mode', 'dr_shift', 'first_mask',
                          'mask_ties'),
               'hgnn': ('hgnn_reward', 'hgnn_mask', 'train_paras.lr')}


def run_problems(run, combos, root, n, exact, cache, want):
    """Problems of one run for the combinations a block reads (empty when it passes).
    Arrays that pass are kept in cache[(name, folder, combo)] as float makespans."""
    out = []
    ck = os.path.join(root, run.ckpt)
    if not os.path.exists(ck):
        out.append(f'missing checkpoint {run.ckpt}')
    lg = os.path.join(root, run.log)
    if not os.path.exists(lg):
        out.append(f'missing training log {run.log}')
    else:
        with open(lg, errors='replace') as f:
            if 'done in' not in f.read():
                out.append(f'{run.log} has no completion line')
    cp = os.path.join(root, run.config)
    if not os.path.exists(cp):
        out.append(f'missing configuration {run.config}')
    else:
        with open(cp) as f:
            cfg = json.load(f)
        for k in EFFORT_KEYS[run.kind]:
            if _get(cfg, k) == '<absent>':
                out.append(f'{run.config} records no {k}')
        seed = _get(cfg, EFFORT_KEYS[run.kind][3])
        if seed != '<absent>' and int(seed) != run.seed:
            out.append(f'{run.config} records seed {seed}, want {run.seed}')
        must = dict(want or {})
        if exact:
            must.update({'first_mask': 'exact', 'mask_ties': 'exact'} if run.kind == 'family'
                        else {'hgnn_mask': 'exact'})
        for k, v in must.items():
            got = _get(cfg, k)
            if got != v and not (isinstance(v, float) and isinstance(got, (int, float))
                                 and abs(got - v) < 1e-12):
                out.append(f'{run.config} records {k}={got!r}, want {v!r}')
    ck_time = os.path.getmtime(ck) if os.path.exists(ck) else None
    for c in combos:
        rel = run.array(c)
        p = os.path.join(root, rel)
        if not os.path.exists(p):
            out.append(f'missing {rel}')
            continue
        a = np.load(p)
        if a.ndim not in (1, 2) or a.shape[0] != n:
            out.append(f'{rel} has shape {a.shape}, want {n} rows')
            continue
        ms = (a[:, 0] if a.ndim == 2 else a).astype(float)
        if not np.all(np.isfinite(ms)) or np.any(ms <= 0):
            out.append(f'{rel} holds makespans that are not finite and positive')
            continue
        if ck_time is not None and os.path.getmtime(p) < ck_time:
            out.append(f'{rel} is older than its checkpoint {run.ckpt}')
            continue
        cache[(run.name, run.folder, c)] = ms
    return out


def effort_of(arm, root, unit):
    """Training effort of an arm from the configurations of its runs: the counts every
    seed records (refused by the block check when they differ)."""
    per = {}
    for s, run in sorted(arm.runs.items()):
        with open(os.path.join(root, run.config)) as f:
            cfg = json.load(f)
        upd, envs, regs, _ = (_get(cfg, k) for k in EFFORT_KEYS[run.kind])
        combos = [r.strip() for r in str(regs).split(',') if r.strip()]
        per[s] = dict(updates=int(upd), envs=int(envs), combos=combos,
                      recipe={k: _get(cfg, k) for k in RECIPE_KEYS[run.kind]})
    first = per[min(per)]
    same = all((v['updates'], v['envs'], v['combos']) ==
               (first['updates'], first['envs'], first['combos']) for v in per.values())
    return dict(unit=unit, updates=first['updates'], envs_per_update=first['envs'],
                combos=first['combos'], episodes=first['updates'] * first['envs'],
                episodes_per_combination=first['updates'] * first['envs'] / len(first['combos']),
                seeds=list(per), agree=same,
                per_seed={f's{s}': dict(updates=v['updates'], envs=v['envs'],
                                        combos=v['combos'], recipe=v['recipe'])
                          for s, v in per.items()})


class Block:
    """One analysis: the runs and combinations it reads (own arm or comparator), a
    status, and its result."""

    def __init__(self, name, title, optional=False):
        self.name, self.title, self.optional = name, title, optional
        self.reads = []                 # (Run, combos, own, wanted configuration values)
        self.effort_arms = []           # (key, Arm, unit) whose configurations must agree
        self.status, self.reason, self.problems, self.result = None, '', [], None
        self.pre_problems = []          # problems found while the block was built
        self.compute = None

    def need(self, arm, seeds, combos, own, want=None):
        for s in seeds:
            self.reads.append((arm.runs[s], list(combos), own, want))

    def assess(self, root, n, exact, cache):
        own = [(r, c) for r, cs, o, _ in self.reads if o for c in cs]
        if self.status == 'skipped':
            return
        if not any(os.path.exists(os.path.join(root, r.array(c))) for r, c in own):
            ck = {r.key for r, _ in own if os.path.exists(os.path.join(root, r.ckpt))}
            self.status = 'skipped'
            self.reason = (self.reason + '; ' if self.reason else '') + (
                f'no result array of the arm exists (0 of {len(own)}); checkpoints '
                f'{len(ck)} of {len({r.key for r, _ in own})}')
            return
        merged, want = {}, {}
        for r, cs, _, w in self.reads:
            m = merged.setdefault(r.key, (r, []))
            m[1].extend(c for c in cs if c not in m[1])
            want.setdefault(r.key, {}).update(w or {})
        problems = list(self.pre_problems)
        for k, (r, cs) in merged.items():
            problems += run_problems(r, cs, root, n, exact, cache, want[k])
        if not problems:
            for key, arm, unit in self.effort_arms:
                e = effort_of(arm, root, unit)
                if not e['agree']:
                    problems.append(f'the seeds of {arm.label} record different training '
                                    f'effort: {e["per_seed"]}')
        self.problems = [_relmsg(p, root) for p in problems]
        self.status = 'refused' if problems else 'computed'


# ================================================================== statistics
def pairs(a, b, seeds, combo, cache):
    """[k, n]: d = 100 (a_s - b_s) / b_s on one combination, seed s of arm a against seed
    s of arm b for every s of seeds."""
    return np.stack([SF.rel_gap(cache[(a.runs[s].name, a.runs[s].folder, combo)],
                                cache[(b.runs[s].name, b.runs[s].folder, combo)])
                     for s in seeds])


def component(d, B):
    """Interval package of one paired comparison, d [k, n] (one combination) or [k, n, c]
    (pooled; a drawn base instance brings its c combinations). k >= 2: crossed
    hierarchical bootstrap, fresh default_rng(HIER_SEED), and the seed-level t
    interval; k == 1: bootstrap over base instances, fresh default_rng(INST_SEED).
    Returns (package, bootstrap draws)."""
    x = np.asarray(d, dtype=float)
    k, n = x.shape[0], x.shape[1]
    c = x.shape[2] if x.ndim == 3 else 1
    if k >= 2:
        hb = SF.hier_boot(x, B=B, seed=HIER_SEED, alpha=ALPHA, return_draws=True)
        t = SF.seed_t(x, ALPHA)
        out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
                   seed_means=t['seed_means'], scheme='hierarchical', stream=f'default_rng({HIER_SEED})')
        draws = hb['draws']
    else:
        ib = SF.instance_boot(x[0], B=B, seed=INST_SEED, alpha=ALPHA, return_draws=True)
        out = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], t_lo=float('nan'),
                   t_hi=float('nan'), seed_means=[float(x[0].mean())], scheme='instance',
                   stream=f'default_rng({INST_SEED})')
        draws = ib['draws']
    out.update(k=int(k), n=int(n), c=int(c), direction=SF.direction(out['lo'], out['hi']),
               **{f'v{mg}': SF.verdict(out['lo'], out['hi'], mg) for mg in MARGINS})
    return out, draws


def comparisons(D, B, families):
    """Per-combination packages of {combo: d} and the pooled rows of the families."""
    comps = {r: component(d, B)[0] for r, d in D.items()}
    pooled = {f: component(np.stack([D[r] for r in regs], axis=2), B)[0]
              for f, regs in families.items()}
    for f, regs in families.items():
        pooled[f]['combos'] = list(regs)
    return comps, pooled


def iut_block(comps, families):
    res = {}
    for fam, regs in families.items():
        for mg in MARGINS:
            for claim in ('equivalent', 'non-inferior'):
                holds, failing = SF.iut({r: comps[r][f'v{mg}'] for r in regs}, claim)
                res[f'{fam}|{mg}|{claim}'] = dict(holds=holds, failing=failing,
                                                  components=list(regs), margin=mg)
    return res


def superiority(D, B):
    """Per combination: interval, one-sided p-values for H1: d < 0 and H1: d > 0
    (seed-level t, bootstrap from the same draws), Holm over the combinations of D for
    each kind, and the combined verdict (p89_public_family.superiority_verdict)."""
    rows = {}
    for r, d in D.items():
        c, draws = component(d, B)
        pl, pg = SF.seed_t_p(d, 'less'), SF.seed_t_p(d, 'greater')
        c.update(p_t=1.0 if np.isnan(pl) else pl, p_boot=SF.boot_p_one_sided(draws, 'less'),
                 p_t_greater=1.0 if np.isnan(pg) else pg,
                 p_boot_greater=SF.boot_p_one_sided(draws, 'greater'))
        rows[r] = c
    keys = list(rows)
    for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
        for r, a in zip(keys, SF.holm([rows[r][col] for r in keys])):
            rows[r][col + '_holm'] = a
    for r in keys:
        x = rows[r]
        x['verdict'] = superiority_verdict(x['lo'], x['hi'], x['p_t_holm'], x['p_boot_holm'],
                                           x['p_t_greater_holm'], x['p_boot_greater_holm'], ALPHA)
    return rows


def families_of(combos):
    fams = {f: regs for f, regs in FAMILIES.items() if set(regs) <= set(combos)}
    if not fams and len(combos) > 1:
        fams = {'Eval': list(combos)}
    return fams


# ================================================================== blocks
def build_blocks(root, environment, hgnn_rate, hgnn_note):
    """The blocks of one run of the script, with the runs each reads."""
    exact = environment == 'exact'
    ex_folders = [FOLDER] + sorted(k for k, v in REG.FOLDER_ALIAS['exact'].items() if v == FOLDER)

    def folder_for(make, combos):
        """Exact environment: FOLDER. Earlier environment: the first 10x25 campaign
        folder that holds every array of the arm made by make(folder)."""
        if exact:
            return FOLDER
        for f in ex_folders:
            a = make(f)
            if all(os.path.exists(os.path.join(root, r.array(c))) for r in a.runs.values()
                   for c in combos):
                return f
        return FOLDER

    blocks, notes = [], {}
    pol = principal('released', None, environment, FOLDER)
    spec = {r: principal('specialist', r, environment, FOLDER) for r in R8}

    # ---- policy against the specialists on the seeds of the supporting arms
    b = Block('policy', 'policy vs specialist of each combination (seeds of the supporting arms)')
    seeds = common(pol, *spec.values(), within=REG.SECONDARY_SEEDS)
    b.need(pol, seeds, R8, True)
    for r in R8:
        b.need(spec[r], seeds, [r], False)
    b.effort_arms = [('policy', Arm('policy', {s: pol.runs[s] for s in seeds}), 'update')] + \
        [(f'specialist {r}', Arm(f'specialist {r}', {s: spec[r].runs[s] for s in seeds}), 'update')
         for r in R8]

    def compute_policy(b=b, seeds=seeds):
        D = {r: pairs(pol, spec[r], seeds, r, CACHE) for r in R8}
        comps, pooled = comparisons(D, ARGS_B[0], FAMILIES)
        eff = {k: effort_of(a, root, u) for k, a, u in b.effort_arms}
        return dict(seeds=list(seeds), combos=R8, first='policy', second='specialist',
                    components=comps, pooled=pooled, iut=iut_block(comps, FAMILIES),
                    effort=dict(policy=eff['policy'],
                                specialists={r: eff[f'specialist {r}'] for r in R8}))
    b.compute = compute_policy
    blocks.append(b)

    # ---- arms that follow the policy's training setting, and the third-budget W specialist
    for stem in RECIPE + (WTHIRD,):
        name = block_name(stem)
        combos = list(dict(PLAN.SECONDARY_EVAL[stem])[FOLDER])
        if not exact and REG.SECONDARY_ARMS[stem][1] is None:
            notes[name] = 'no stored counterpart in the earlier environment'
            continue
        f = folder_for(lambda f, stem=stem: secondary(stem, environment, f), combos)
        arm = secondary(stem, environment, f)
        apol = principal('released', None, environment, f)
        if stem == WTHIRD:
            aspec = {r: principal('specialist', WTHIRD_COMBO, environment, f) for r in combos}
            title = f'{arm.label} vs the {WTHIRD_COMBO} specialist and vs the policy'
        else:
            aspec = {r: principal('specialist', r, environment, f) for r in combos}
            title = f'{arm.label} vs the policy and vs the specialist of each combination'
        b = Block(name, title)
        seeds = common(arm, apol, *aspec.values())
        want = ARM_CONFIG.get(stem)
        b.need(arm, seeds, combos, True, want)
        b.need(apol, seeds, combos, False)
        for r in combos:
            b.need(aspec[r], seeds, [r], False)
        b.effort_arms = [(stem, Arm(arm.label, {s: arm.runs[s] for s in seeds}), 'update')]

        def compute_arm(b=b, stem=stem, arm=arm, apol=apol, aspec=aspec, seeds=seeds,
                        combos=combos, f=f):
            fams = families_of(combos)
            Dp = {r: pairs(arm, apol, seeds, r, CACHE) for r in combos}
            Ds = {r: pairs(arm, aspec[r], seeds, r, CACHE) for r in combos}
            cp, pp = comparisons(Dp, ARGS_B[0], fams)
            cs, ps = comparisons(Ds, ARGS_B[0], fams)
            return dict(arm=arm.label, stem=stem, recipe=REG.SECONDARY_ARMS[stem][0],
                        folder=f, seeds=list(seeds), combos=combos,
                        specialist=(f'{WTHIRD_COMBO} specialist' if stem == WTHIRD
                                    else 'specialist of each combination'),
                        vs_policy=dict(components=cp, pooled=pp),
                        vs_specialist=dict(components=cs, pooled=ps, iut=iut_block(cs, fams)),
                        effort=effort_of(b.effort_arms[0][1], root, 'update'))
        b.compute = compute_arm
        blocks.append(b)

    # ---- HGNN
    if exact:
        tag = lr_tag(hgnn_rate) if hgnn_rate else None
    else:
        tag = ''
    hg = {}
    for name, reward in HGNN_REWARD.items():
        b = Block(name, f'policy vs HGNN ({REWARD_WORD[reward]} reward)')
        if tag is None:
            b.status, b.reason = 'skipped', hgnn_note
            blocks.append(b)
            continue
        sets = HGNN_SEED_SETS[reward]
        f = folder_for(lambda f, reward=reward, sets=sets: hgnn_arm(reward, 'joint', sets[-1], tag,
                                                                   environment, f), R8)
        hseeds, problem = hgnn_seeds(root, reward, tag, environment, f)
        if problem:
            b.pre_problems = [problem]
        arm = hgnn_arm(reward, 'joint', hseeds, tag, environment, f)
        hg[name] = arm
        apol = principal('released', None, environment, f)
        seeds = common(arm, apol)
        want = {'hgnn_reward': reward, 'args.family_regimes': ','.join(TRAIN)}
        if exact:
            want['train_paras.lr'] = float(hgnn_rate)
        b.need(arm, seeds, R8, True, want)
        b.need(apol, seeds, R8, False)
        b.effort_arms = [(name, Arm(arm.label, {s: arm.runs[s] for s in seeds}), 'iteration')]

        def compute_hgnn(b=b, arm=arm, apol=apol, seeds=seeds, f=f, reward=reward, sets=sets):
            D = {r: pairs(apol, arm, seeds, r, CACHE) for r in R8}
            rows = superiority(D, ARGS_B[0])
            pooled = {fm: component(np.stack([D[r] for r in regs], axis=2), ARGS_B[0])[0]
                      for fm, regs in FAMILIES.items()}
            return dict(arm=arm.label, reward=REWARD_WORD[reward], hgnn_reward=reward, folder=f,
                        seeds=list(seeds), seed_sets=[list(st) for st in sets], combos=R8,
                        pairing='HGNN seed s against policy seed s',
                        first='policy', second=arm.label, learning_rate=hgnn_rate,
                        components=rows, pooled=pooled,
                        effort=effort_of(b.effort_arms[0][1], root, 'iteration'))
        b.compute = compute_hgnn
        blocks.append(b)

    b = Block('hgnnsolo', 'HGNN (PACT reward) vs the HGNN specialist of each combination',
              optional='hgnnsolo' in OPTIONAL)
    if tag is None:
        b.status, b.reason = 'skipped', hgnn_note
    else:
        f = hg['hgnn'].runs[min(hg['hgnn'].runs)].folder
        solo = {r: hgnn_arm('certified', r, HGNN_SOLO_SEEDS, tag, environment, f) for r in R8}
        joint = hg['hgnn']
        seeds = common(joint, *solo.values())
        for r in R8:
            want = {'hgnn_reward': 'certified', 'args.family_regimes': r}
            if exact:
                want['train_paras.lr'] = float(hgnn_rate)
            b.need(solo[r], seeds, [r], True, want)
        b.need(joint, seeds, R8, False)
        b.effort_arms = [(f'solo {r}', Arm(solo[r].label, {s: solo[r].runs[s] for s in seeds}),
                          'iteration') for r in R8]

        def compute_solo(b=b, joint=joint, solo=solo, seeds=seeds, f=f):
            D = {r: pairs(joint, solo[r], seeds, r, CACHE) for r in R8}
            comps, pooled = comparisons(D, ARGS_B[0], FAMILIES)
            return dict(arm=joint.label, folder=f, seeds=list(seeds), combos=R8,
                        first=joint.label, second='HGNN specialist of each combination',
                        components=comps, pooled=pooled, iut=iut_block(comps, FAMILIES),
                        effort={r: effort_of(a, root, u) for (k, a, u), r in zip(b.effort_arms, R8)})
        b.compute = compute_solo
    blocks.append(b)
    return blocks, notes


CACHE = {}
ARGS_B = [SF.B_DEFAULT]


def hgnn_selection(root, override, exact):
    """(rate or None, note): the learning rate of the HGNN runs, from
    scripts/hgnn_select_lr.py or --hgnn-lr."""
    if not exact:
        return None, 'stored runs of the earlier environment (released learning rate)'
    try:
        chosen, scores = HLR.select(root)
        sel = dict(chosen=chosen, scores={lr: dict(score=v[0], iteration=v[1])
                                          for lr, v in scores.items()})
    except SystemExit as e:
        chosen, sel = None, dict(chosen=None, reason=_relmsg(e, root))
    if override:
        if override not in HLR.RATES:
            refuse(f'--hgnn-lr {override} is not one of {HLR.RATES}')
        return override, dict(sel, used=override, source='--hgnn-lr')
    if chosen is None:
        return None, dict(sel, used=None)
    return chosen, dict(sel, used=chosen, source='scripts/hgnn_select_lr.py')


# ================================================================== analysis
def analyse(root, environment, B, hgnn_override=None):
    exact = environment == 'exact'
    CACHE.clear()
    ARGS_B[0] = B
    n = check_bases(root, PLAN.FACTORY[FOLDER][0])
    rate, sel = hgnn_selection(root, hgnn_override, exact)
    if exact and rate is None:
        stray = sorted(glob.glob(os.path.join(root, REG.res_dir(FOLDER), 'Result_hgnn-*-xmask*_*.npy')))
        note = f'HGNN learning rate not selected: {sel.get("reason")}'
        if stray:
            note = f'{note}; but {len(stray)} HGNN result arrays exist'
    else:
        stray, note = [], ''
    blocks, notes = build_blocks(root, environment, rate, note)
    for b in blocks:
        b.assess(root, n, exact, CACHE)
        if b.name in HGNN_REWARD or b.name == 'hgnnsolo':
            if stray and b.status == 'skipped':
                b.status, b.problems = 'refused', [note]
    for b in blocks:
        if b.status == 'computed':
            b.result = b.compute()
    # the verdict of each arm against the specialists beside the policy's on the same
    # seeds and the same campaign folder (the policy block reads FOLDER)
    ref = next(b for b in blocks if b.name == 'policy')
    for b in blocks:
        if b.status != 'computed' or b.name in ('policy', 'hgnnsolo') or b.name in HGNN_REWARD:
            continue
        vs = b.result['vs_specialist']
        if ref.status != 'computed' or ref.result['seeds'] != b.result['seeds'] or \
                b.result['folder'] != FOLDER or b.name == block_name(WTHIRD):
            vs['changed'] = None
            continue
        vs['changed'] = {str(mg): [r for r in b.result['combos'] if r in ref.result['components']
                                   and vs['components'][r][f'v{mg}']
                                   != ref.result['components'][r][f'v{mg}']]
                         for mg in MARGINS}
        vs['reference'] = {r: {str(mg): ref.result['components'][r][f'v{mg}'] for mg in MARGINS}
                           for r in b.result['combos'] if r in ref.result['components']}
    return dict(blocks=blocks, notes=notes, n=n, hgnn_selection=sel, hgnn_rate=rate)


# ================================================================== output
def tex(v):
    return SF.TEX[v]


def _scheme(c):
    if c['scheme'] == 'hierarchical':
        return (f'hierarchical 95% CI over {c["k"]} seed pairs and {c["n"]} base instances'
                + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))
    return (f'one training run: 95% CI over {c["n"]} base instances'
            + (f', {c["c"]} combinations kept together' if c['c'] > 1 else ''))


def comp_line(name, c, what, mark, verdicts=True, extra='', extra_what=''):
    """Macros of one comparison: mean, interval, verdicts at both margins (optional),
    direction, then the extra macros, with one comment."""
    s = (f'\\newcommand{{\\{name}}}{{{f2(c["mean"])}}}'
         f'\\newcommand{{\\{name}Lo}}{{{f2(c["lo"])}}}'
         f'\\newcommand{{\\{name}Hi}}{{{f2(c["hi"])}}}')
    if verdicts:
        s += ''.join(f'\\newcommand{{\\{name}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}' for mg in MARGINS)
    s += f'\\newcommand{{\\{name}Dir}}{{{tex(c["direction"])}}}' + extra
    return (s + f'  % {mark}{what}: mean, {_scheme(c)}'
            + (', verdicts at 0.5 and 1' if verdicts else '')
            + f', direction of [{c["lo"]:+.4f}, {c["hi"]:+.4f}]' + extra_what)


def iut_lines(L, name, block, fams, mark):
    for fam in fams:
        for mg in MARGINS:
            for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                x = block[f'{fam}|{mg}|{claim}']
                L.append(f'\\newcommand{{\\{name}{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                         f'  % {mark}IUT, {claim} on {",".join(x["components"])} at +-{mg}%: '
                         f'failing {",".join(x["failing"]) or "none"}')


def effort_line(name, e, what, mark):
    unit = 'Updates' if e['unit'] == 'update' else 'Iterations'
    return (f'\\newcommand{{\\{name}{unit}}}{{{e["updates"]:,}}}'
            f'\\newcommand{{\\{name}Episodes}}{{{int(round(e["episodes_per_combination"])):,}}}'
            f'\\newcommand{{\\{name}Combos}}{{{len(e["combos"])}}}'
            f'  % {mark}{what}: {e["unit"]}s, episodes per training combination in expectation '
            f'({e["updates"]} x {e["envs_per_update"]} / {len(e["combos"])}), training '
            f'combinations ({",".join(e["combos"])}), from the stored configurations')


def macro_lines(res, B, environment, check):
    mark = 'CHECK (earlier environment, stored counterparts) ' if check else ''
    blocks = {b.name: b for b in res['blocks']}
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Supporting arms and HGNN, {environment} environment, 10x25 test set; B = {B}; '
         f'hierarchical intervals fresh default_rng({HIER_SEED}), one-run intervals fresh '
         f'default_rng({INST_SEED}).',
         '% Gaps in %: first-named method minus second over second (negative = first-named '
         'method shorter), seed s against seed s.']
    if check:
        L.insert(1, '% CHECK ONLY: earlier environment, stored counterpart arms; not for the '
                    'manuscript.')
    L.append(f'\\newcommand{{\\SecBoot}}{{{B}}}\\newcommand{{\\SecSeedN}}'
             f'{{{len(REG.SECONDARY_SEEDS)}}}  % {mark}bootstrap replicates; training seeds per '
             f'supporting arm')
    b = blocks['policy']
    if b.status == 'computed':
        x = b.result
        for r in R8:
            L.append(comp_line(f'SecPolPar{r}', x['components'][r],
                               f'policy vs specialist, {r}, seeds {x["seeds"][0]}-{x["seeds"][-1]}', mark))
        for fam in FAMILIES:
            L.append(comp_line(f'SecPolParPool{fam}', x['pooled'][fam],
                               f'policy vs specialist, pooled {fam}', mark))
        iut_lines(L, 'SecPolParIut', x['iut'], FAMILIES, mark)
        L.append(effort_line('SecPol', x['effort']['policy'], 'policy', mark))
        sp = x['effort']['specialists']
        if all((v['updates'], v['envs_per_update']) == (sp['N']['updates'], sp['N']['envs_per_update'])
               and len(v['combos']) == 1 for v in sp.values()):
            L.append(effort_line('SecSpec', sp['N'], 'each specialist', mark))
    for stem in RECIPE + (WTHIRD,):
        b = blocks.get(block_name(stem))
        if b is None or b.status != 'computed':
            continue
        x, w = b.result, word(stem)
        fams = list(x['vs_policy']['pooled'])
        what = f'{x["arm"]} ({x["recipe"]})'
        if stem == WTHIRD:
            for r in x['combos']:
                L.append(comp_line(f'Sec{w}VsSpec{r}', x['vs_specialist']['components'][r],
                                   f'{what} vs {WTHIRD_COMBO} specialist, {r}', mark))
                L.append(comp_line(f'Sec{w}VsPol{r}', x['vs_policy']['components'][r],
                                   f'{what} vs policy, {r}', mark))
            for fam in fams:
                L.append(comp_line(f'Sec{w}VsSpecPool{fam}', x['vs_specialist']['pooled'][fam],
                                   f'{what} vs {WTHIRD_COMBO} specialist, pooled {fam}', mark))
                L.append(comp_line(f'Sec{w}VsPolPool{fam}', x['vs_policy']['pooled'][fam],
                                   f'{what} vs policy, pooled {fam}', mark))
        else:
            for r in x['combos']:
                L.append(comp_line(f'Sec{w}VsPol{r}', x['vs_policy']['components'][r],
                                   f'{what} vs policy, {r}', mark))
            for fam in fams:
                L.append(comp_line(f'Sec{w}VsPolPool{fam}', x['vs_policy']['pooled'][fam],
                                   f'{what} vs policy, pooled {fam}', mark))
            for r in x['combos']:
                L.append(comp_line(f'Sec{w}Par{r}', x['vs_specialist']['components'][r],
                                   f'{what} vs specialist, {r}', mark))
            for fam in fams:
                L.append(comp_line(f'Sec{w}ParPool{fam}', x['vs_specialist']['pooled'][fam],
                                   f'{what} vs specialist, pooled {fam}', mark))
            iut_lines(L, f'Sec{w}ParIut', x['vs_specialist']['iut'], fams, mark)
            ch = x['vs_specialist'].get('changed')
            if ch is not None:
                L.append(''.join(f'\\newcommand{{\\Sec{w}ParChanged{MW[mg]}N}}{{{len(ch[str(mg)])}}}'
                                 for mg in MARGINS)
                         + f'  % {mark}{what}: combinations (of {len(x["combos"])}) whose verdict '
                         f'against the specialist differs from the policy\'s on the same seeds, at 0.5 '
                         f'({",".join(ch["0.5"]) or "none"}) and 1 ({",".join(ch["1.0"]) or "none"})')
        L.append(effort_line(f'Sec{w}', x['effort'], what, mark))
    for name, w in (('hgnn', 'Hgnn'), ('hgnnpub', 'HgnnPub')):
        b = blocks[name]
        if b.status != 'computed':
            continue
        x = b.result
        sd = x['seeds']
        what = f'policy vs {x["arm"]} ({x["reward"]} reward, seeds {sd[0]}-{sd[-1]})'
        L.append(f'\\newcommand{{\\SecPol{w}SeedN}}{{{len(sd)}}}  % {mark}{what}: training seeds, '
                 f'HGNN seed s against policy seed s ({",".join(map(str, sd))})')
        for r in R8:
            c = x['components'][r]
            L.append(comp_line(
                f'SecPol{w}{r}', c, f'{what}, {r}', mark, verdicts=False,
                extra=(f'\\newcommand{{\\SecPol{w}{r}Verdict}}{{{c["verdict"]}}}'
                       f'\\newcommand{{\\SecPol{w}{r}PHolm}}{{{c["p_t_holm"]:.2g}}}'),
                extra_what=(f'; combined verdict, Holm-adjusted seed-t p shorter; Holm p shorter '
                            f'{c["p_t_holm"]:.2g} (t), {c["p_boot_holm"]:.2g} (bootstrap), Holm p '
                            f'longer {c["p_t_greater_holm"]:.2g} (t), '
                            f'{c["p_boot_greater_holm"]:.2g} (bootstrap)')))
        vs = [x['components'][r]['verdict'] for r in R8]
        L.append(f'\\newcommand{{\\SecPol{w}VerdictAheadN}}{{{vs.count(VERDICT_WORDS[0])}}}'
                 f'\\newcommand{{\\SecPol{w}VerdictBehindN}}{{{vs.count(VERDICT_WORDS[1])}}}'
                 f'  % {mark}{what}: combinations (of 8) with the combined verdict ahead, behind')
        for fam in FAMILIES:
            L.append(comp_line(f'SecPol{w}Pool{fam}', x['pooled'][fam], f'{what}, pooled {fam}',
                               mark, verdicts=False))
        L.append(effort_line(f'Sec{w}', x['effort'], x['arm'], mark))
        if name == 'hgnn' and x['learning_rate']:
            m, e = x['learning_rate'].split('e')
            L.append(f'\\newcommand{{\\SecHgnnLr}}{{{m}\\times10^{{{int(e)}}}}}'
                     f'  % {mark}learning rate of the HGNN runs (math mode)')
    b = blocks['hgnnsolo']
    if b.status == 'computed':
        x = b.result
        for r in R8:
            L.append(comp_line(f'SecHgnnPar{r}', x['components'][r],
                               f'{x["arm"]} vs HGNN specialist, {r}', mark))
        for fam in FAMILIES:
            L.append(comp_line(f'SecHgnnParPool{fam}', x['pooled'][fam],
                               f'{x["arm"]} vs HGNN specialist, pooled {fam}', mark))
        iut_lines(L, 'SecHgnnParIut', x['iut'], FAMILIES, mark)
        e = x['effort']
        if all((v['updates'], v['envs_per_update']) == (e['N']['updates'], e['N']['envs_per_update'])
               for v in e.values()):
            L.append(effort_line('SecHgnnSpec', e['N'], 'each HGNN specialist', mark))
    return L


def macro_names(lines):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))


def check_names(names, root, out_paths):
    """Every name is Sec plus letters, unique, and absent from every other macro file."""
    bad = [x for x in names if not re.fullmatch(r'Sec[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not Sec plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse('duplicate macro names in the output')
    own = {os.path.abspath(p) for p in out_paths} | {
        os.path.abspath(os.path.join(root, p)) for p in (OUT_FILES[0], _check_name(OUT_FILES[0]))}
    files = sorted(glob.glob(os.path.join(root, 'manuscript', 'macros_*.tex')) +
                   glob.glob(os.path.join(root, 'paper', 'macros_*.tex')) +
                   [os.path.join(root, d, 'macros.tex') for d in ('manuscript', 'paper')])
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) in own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            refuse(f'macro name collision with {os.path.relpath(f, root)}: {sorted(clash)[:5]}')
    return len(files)


def ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def tci(x):
    """Seed-level t interval, '-' for one training run."""
    if not (np.isfinite(x['t_lo']) and np.isfinite(x['t_hi'])):
        return '-'
    return f'[{x["t_lo"]:+.2f}, {x["t_hi"]:+.2f}]'


def markdown(res, B, environment, check):
    blocks = res['blocks']
    md = [('# CHECK (earlier environment, stored counterpart arms): ' if check else '# ')
          + 'supporting arms and HGNN, paired inference', '',
          f'{environment} environment, 10x25 test set ({res["n"]} base instances per combination). '
          f'd = 100 (a - b) / b, a the first-named method, seed s against seed s; negative = '
          f'first-named method shorter. 95% intervals: crossed hierarchical bootstrap over seeds '
          f'and base instances (B = {B}, default_rng({HIER_SEED})), one-run bootstrap over base '
          f'instances (default_rng({INST_SEED})); verdicts at 0.5% and 1%; joint claims by the '
          f'intersection-union test; HGNN superiority by Holm over the eight combinations.', '',
          '| block | status | seeds | reason |', '|---|---|---|---|']
    for b in blocks:
        seeds = ','.join(map(str, b.result['seeds'])) if b.result else '-'
        why = b.reason if b.status == 'skipped' else '; '.join(b.problems[:6]) + \
            (f'; ... {len(b.problems) - 6} more' if len(b.problems) > 6 else '')
        md.append(f'| {b.name}{" (optional)" if b.optional else ""} | {b.status} | {seeds} | '
                  f'{why or "-"} |')
    for name, why in res['notes'].items():
        md.append(f'| {name} | not applicable | - | {why} |')
    sel = res['hgnn_selection']
    md += ['', f'HGNN learning rate: {sel if isinstance(sel, str) else sel.get("used")} '
           + ('' if isinstance(sel, str) else f'(selection: {sel.get("chosen")}'
              + (f', {sel["reason"]}' if sel.get('reason') else '') + ')'), '']

    def effort_md(e, what):
        return (f'Training effort, {what}: {e["updates"]} {e["unit"]}s x {e["envs_per_update"]} '
                f'episodes on {",".join(e["combos"])} = {e["episodes_per_combination"]:.0f} episodes '
                f'per combination in expectation.')

    for b in blocks:
        if b.status != 'computed':
            continue
        x = b.result
        md += [f'## {b.name}: {b.title}', '']
        if b.name in ('policy', 'hgnnsolo'):
            md += ['| comb. | mean [95% CI] | seed-t CI | 0.5% | 1% | direction |',
                   '|---|---|---|---|---|---|']
            for r in R8:
                c = x['components'][r]
                md.append(f'| {r} | {ci(c)} | {tci(c)} | {c["v0.5"]} | '
                          f'{c["v1.0"]} | {c["direction"]} |')
            for fam, c in x['pooled'].items():
                md.append(f'| pooled {fam} | {ci(c)} | {tci(c)} | '
                          f'{c["v0.5"]} | {c["v1.0"]} | {c["direction"]} |')
            md += ['', 'Joint claims: ' + '; '.join(
                f'{k.replace("|", " ")} {"holds" if v["holds"] else "fails (" + ",".join(v["failing"]) + ")"}'
                for k, v in x['iut'].items()), '']
            if b.name == 'policy':
                md.append(effort_md(x['effort']['policy'], 'policy'))
                md.append(effort_md(x['effort']['specialists']['N'], 'specialist of N (each specialist '
                                    'trains on its combination)'))
            else:
                md.append(effort_md(x['effort']['N'], 'HGNN specialist of N'))
            md.append('')
        elif b.name in HGNN_REWARD:
            md += ['| comb. | policy minus HGNN [95% CI] | seed-t CI | Holm p shorter (t / boot) | '
                   'Holm p longer (t / boot) | verdict |', '|---|---|---|---|---|---|']
            for r in R8:
                c = x['components'][r]
                md.append(f'| {r} | {ci(c)} | {tci(c)} | '
                          f'{c["p_t_holm"]:.2g} / {c["p_boot_holm"]:.2g} | {c["p_t_greater_holm"]:.2g} / '
                          f'{c["p_boot_greater_holm"]:.2g} | {c["verdict"]} |')
            for fam, c in x['pooled'].items():
                md.append(f'| pooled {fam} | {ci(c)} | {tci(c)} | - | - | '
                          f'direction {c["direction"]} |')
            md += ['', f'{x["reward"][0].upper()}{x["reward"][1:]} reward; seeds '
                   f'{", ".join(map(str, x["seeds"]))} (HGNN seed s against policy seed '
                   f's; the block reads seeds '
                   + ' or '.join(f'{st[0]}-{st[-1]}' for st in x['seed_sets'])
                   + ', the set whose runs exist).',
                   '', effort_md(x['effort'], x['arm']) + f' Learning rate {x["learning_rate"] or "released"}.', '']
        else:
            vsp, vss = x['vs_policy'], x['vs_specialist']
            ref = vss.get('reference') or {}
            md += [f'Arm {x["arm"]} ({x["recipe"]}), folder {x["folder"]}, against the policy and '
                   f'against the {x["specialist"]}.', '',
                   '| comb. | vs policy | 0.5% | 1% | dir. | vs specialist | 0.5% | 1% | dir. | '
                   'policy vs specialist, same seeds (0.5% / 1%) |', '|---|---|---|---|---|---|---|---|---|---|']
            rows = [(r, vsp['components'][r], vss['components'][r]) for r in x['combos']] + \
                [(f'pooled {f}', vsp['pooled'][f], vss['pooled'][f]) for f in vsp['pooled']]
            for r, p, s in rows:
                rv = ref.get(r)
                md.append(f'| {r} | {ci(p)} | {p["v0.5"]} | {p["v1.0"]} | {p["direction"]} | {ci(s)} | '
                          f'{s["v0.5"]} | {s["v1.0"]} | {s["direction"]} | '
                          f'{(rv["0.5"] + " / " + rv["1.0"]) if rv else "-"} |')
            if vss['iut']:
                md += ['', 'Joint claims against the specialists: ' + '; '.join(
                    f'{k.replace("|", " ")} {"holds" if v["holds"] else "fails (" + ",".join(v["failing"]) + ")"}'
                    for k, v in vss['iut'].items())]
            if vss.get('changed') is not None:
                md += ['', 'Verdict against the specialist differs from the policy\'s on the same '
                       f'seeds: at 0.5% {",".join(vss["changed"]["0.5"]) or "none"}, at 1% '
                       f'{",".join(vss["changed"]["1.0"]) or "none"}.']
            md += ['', effort_md(x['effort'], x['arm']), '']
    return md


def block_json(b):
    return dict(status=b.status, optional=b.optional, title=b.title, reason=b.reason,
                problems=b.problems, result=b.result)


def _check_name(path):
    stem, ext = os.path.splitext(path)
    return stem + CHECK_SUFFIX + ext


def out_paths(out_root, check):
    return [os.path.join(out_root, _check_name(p) if check else p) for p in OUT_FILES]


def guard_check_out(out, root):
    """--earlier-check writes only outside the repository's paper and results folders
    (and outside the folders of results, schedules, checkpoints and logs)."""
    o = os.path.realpath(out)
    for base in {os.path.realpath(REPO), os.path.realpath(root)}:
        rel = os.path.relpath(o, base)
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            continue
        first = rel.split(os.sep)[0]
        if rel == os.curdir or first in GUARDED or first.startswith('paper'):
            refuse(f'--earlier-check writes its outputs under <out>/manuscript and <out>/results; '
                   f'--out {out} would put them in the repository root or in one of its paper, '
                   f'results, checkpoint or log folders. Give a folder outside them.')


@contextlib.contextmanager
def environment_set(e):
    """Selects the environment of arm_registry for this run and restores it afterwards."""
    old = os.environ.get('PACT_ENV')
    if old not in (None, '', e):
        refuse(f'PACT_ENV={old}, but this run reads the {e} environment')
    os.environ['PACT_ENV'] = e
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('PACT_ENV', None)
        else:
            os.environ['PACT_ENV'] = old


def required_blocks(spec, blocks):
    names = [b.name for b in blocks]
    if spec in (None, '', 'none'):
        return []
    if spec == 'all':
        return [b.name for b in blocks if not b.optional]
    want = [x.strip() for x in spec.split(',') if x.strip()]
    unknown = [x for x in want if x not in names]
    if unknown:
        refuse(f'--require names unknown blocks {unknown}; the blocks are {names}')
    return want


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--require', default='none',
                    help="'all' (every block except the optional ones), 'none', or block names")
    ap.add_argument('--hgnn-lr', default=None, help='learning rate of the HGNN runs, instead of the '
                    'selection of scripts/hgnn_select_lr.py')
    ap.add_argument('--earlier-check', action='store_true',
                    help='read the stored counterparts in the earlier environment (a check)')
    ap.add_argument('--out', default=None, help='root of the outputs (default: the repository; '
                    'required with --earlier-check)')
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    check = a.earlier_check
    if check:
        if a.out is None:
            refuse('--earlier-check needs --out <folder> outside the repository\'s paper and '
                   'results folders')
        guard_check_out(a.out, root)
        if a.hgnn_lr:
            refuse('--hgnn-lr applies to the exact environment only')
    out_root = os.path.abspath(a.out or REPO)
    outs = out_paths(out_root, check)
    out_mac, out_json, out_md = outs
    environment = 'legacy' if check else 'exact'
    with environment_set(environment):
        check_layout()
        for p in outs:                 # a refusal anywhere below leaves no stale output
            if os.path.exists(p):
                os.remove(p)
        try:
            res = analyse(root, environment, a.boot, a.hgnn_lr)
            blocks = res['blocks']
            missing = [b.name for b in blocks if b.name in required_blocks(a.require, blocks)
                       and b.status != 'computed']
            if missing:
                refuse('required blocks not computed: ' + '; '.join(
                    f'{b.name} ({b.status}: {b.reason or "; ".join(b.problems[:3])})'
                    for b in blocks if b.name in missing))
            L = macro_lines(res, a.boot, environment, check)
            names = macro_names(L)
            n_files = check_names(names, root, outs)
            out = dict(generator=GENERATOR, check=check,
                       mode=('CHECK: earlier environment, stored counterpart arms' if check
                             else 'exact environment'),
                       environment=environment, folder=FOLDER, n_instances=res['n'], boot=a.boot,
                       alpha=ALPHA, margins=MARGINS,
                       sign='d = 100 (a - b) / b, a the first-named method; negative = a shorter',
                       pairing='seed s against seed s on the seeds both arms have, instance i '
                               'against instance i',
                       streams=dict(hierarchical=f'fresh default_rng({HIER_SEED}) per quantity',
                                    instance=f'fresh default_rng({INST_SEED}) per quantity'),
                       superiority_verdict_rule=(
                           'ahead: hierarchical 95% CI below zero and both Holm-adjusted one-sided '
                           'p-values (seed-level t, bootstrap; H1 policy shorter) below 0.05; '
                           'behind: CI above zero and both Holm-adjusted one-sided p-values for H1 '
                           'policy longer (own Holm family over the eight combinations) below 0.05; '
                           'not distinguishable otherwise'),
                       hgnn_selection=res['hgnn_selection'], hgnn_learning_rate=res['hgnn_rate'],
                       computed=[b.name for b in blocks if b.status == 'computed'],
                       skipped={b.name: b.reason for b in blocks if b.status == 'skipped'},
                       refused={b.name: b.problems for b in blocks if b.status == 'refused'},
                       not_applicable=res['notes'], require=a.require,
                       blocks={b.name: block_json(b) for b in blocks},
                       files_read=len(CACHE), n_macros=len(names), macro_files_checked=n_files)
            emit(out_mac, L)
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, 'w') as f:
                json.dump(jsonable(out), f, indent=1)
            md = markdown(res, a.boot, environment, check)
            with open(out_md, 'w') as f:
                f.write('\n'.join(md) + '\n')
        except BaseException as e:
            for p in outs:
                if os.path.exists(p):
                    os.remove(p)
                    print(f'removed {os.path.relpath(p, out_root)} (refusing to certify)')
            if isinstance(e, REG.MissingResult):
                raise Refusal(f'*** REFUSING TO CERTIFY ***\n{e}')
            raise
    print('\n'.join(md))
    print(f'\n{len(names)} macros in {os.path.relpath(out_mac, out_root)}; {len(CACHE)} arrays '
          f'read; computed {out["computed"]}; skipped {sorted(out["skipped"])}; refused '
          f'{sorted(out["refused"])}')
    if out['refused']:
        print('REFUSED BLOCKS (no number written for them): ' + '; '.join(
            f'{k}: {"; ".join(v[:3])}' for k, v in out['refused'].items()), file=sys.stderr)
    return out


if __name__ == '__main__':
    sys.exit(1 if main()['refused'] else 0)
