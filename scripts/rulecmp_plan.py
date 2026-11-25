"""Plan of the comparison with dispatching rules (scripts/run_rule_comparison.sh).

The comparison is reported in three forms in both benchmark families:
  (a) one greedy schedule of a policy against the per-instance best of the four
      masked rules (FIFO, MOR, SPT, MWKR, stored arrays);
  (b) the best of k sampled schedules of the policy against the best of the same
      number of rule schedules (k = 4: the four stored rule schedules; k = 16:
      the four rules, each rolled out with the tie-break seeds 50, 51, 52, 53),
      both estimated from K sampled schedules per instance (K_DEFAULT = 32,
      environment variable RULECMP_K; at least 16);
  (c) one greedy schedule against the single rule with the lowest mean makespan
      on the validation instances of the combination: the instances the training
      runs validated on for checkpoint selection (min(vali_size, 20) of them).
This module defines, for one family per call, the arrays these forms need beyond
the stored greedy and rule arrays, the jobs that produce them, the checks an
output must pass before it counts as done, and the scheduler that runs the jobs
one per core. scripts/p93_rule_comparison.py reads the same definitions.

FAMILIES (run names, seeds, checkpoints and folders from scripts/arm_registry.py)
  public   SD2 10x5 (arm_registry.PUB_*), seeds 301-305, test records
           data/PUBLIC/family/pub_SD2_test_10x5+mix, stored arrays
           test_results/PUBFAMILY/pub_SD2_test_10x5+mix_xmask, validation records
           data/PUBLIC/family/pub_SD2_vali_10x5+mix (all 100 records, evaluated
           by eval_family.py --rule_set masked like the stored test arrays; the
           trainer validates on the first min(vali_size, 20) of them in sorted
           order, which select the rule of form (c); all 100 give a second
           selection reported beside it).
           Other public test sets (--set, arm_registry.PUB_SETS): the same runs on
           data/PUBLIC/family/<set>, outputs in the rulecmp_* subfolders of
           test_results/PUBFAMILY/<set>_xmask; by default the released policy only
           (--arms adds the control and the specialists); no validation job: form
           (c) applies the rule selected on the headline set's validation
           instances, a property of the trained family, to the set's own stored
           rule arrays. The records must equal the set's dataset_meta.json list
           with its count and exclusions (orb7 in the Hurink sets), as
           scripts/p89_public_family.py checks them.
  factory  10x25, the exact-mask names (PACT_ENV=exact; seeds from PACT_SEEDS),
           test records data/FAMILY/10x25 on data/PPVC/10x25+ppvc-mixed, stored
           arrays arm_registry.res_dir('10x25'); the validation instances are the
           ones the trainer builds in memory (no stored records exist), evaluated
           by p90 rollout --mode valirules from the training configuration of the
           released seed of the first active seed, checked against the MWKR norm
           line of every training log of the comparison.

THE PUBLIC FAMILY UNDER THE ACTIVE MASK (--family public --mask active)
  The same jobs, checks and forms for the public family trained under the active mask
  (arm_registry.pub_rule_model with the rule word act: pubx-act-s<seed>,
  pubx-ctlact-s<seed>, pubx-soloact-<R>-s<seed>, seeds PUB_ACTFAM_SEEDS), with both sides
  under the active mask:
    * the policy's schedules (sampled, greedy traces) are built under the run's own mask,
      which p90 takes from the run's training configuration; every p90 command carries
      --expect_mask active, so a run trained under another mask refuses, and every output
      records its mask (key mask_mode), which the checks below compare;
    * the stored greedy arrays are the runs' own (eval_family.py restores mask_mode from the
      configuration), in the same result folder test_results/PUBFAMILY/<set>_xmask;
    * the rules: the stored arrays (tie-break seed 50) are those of
      scripts/p109_public_active_rules.py in test_results/PUBFAMILY/rules_act/<set>_xmask;
      the tie-break seeds 51-53 and the validation records are p109 run jobs; the rule
      traces are p90 --rule <rule> --rule_mask active;
    * every rulecmp_* subfolder lies in test_results/PUBFAMILY/rules_act/<set>_xmask (the
      campaign folder), so no file of the non-delay campaign is read or replaced;
    * label public_act (public_act_<tag> for another public test set), macro prefix RcAct
      (RcAct<tag>); every rule summary records 'pairs: mask_mode=active' and every rule
      folder's rules_meta.json names the active mask and its tie-break seed.
  A sampled or trace file without the key mask_mode counts as the non-delay mask: the files
  of the non-delay campaign carry no such key, every run they read is trained under the
  non-delay mask and every rule of that campaign is rolled out under it.

OUTPUTS (subfolders of the stored result folder F of the family)
  F/rulecmp_sampled/sample<K>+<model>_<R>.npz   p90 --mode sample, K draws, every
                     seed of the released policy and the control on the eight
                     combinations, every specialist on its own combination
  F/rulecmp_trace/trace+<model>_<R>.npz          p90 --mode trace, greedy, every
                     seed of the released policy, eight combinations
  F/rulecmp_trace/trace+<rule>_<R>.npz           p90 --mode trace --rule <rule>
  F/rulecmp_ties/seed<t>/<set>_xmask/Result_<rule>_<R>.npy   eval_family.py
                     --rule_set masked --seed_test t, t = 51, 52, 53 (t = 50 is the
                     stored rule array)
  F/rulecmp_vali/<vali folder>/Result_<rule>_<R>.npy   rules on the validation
                     instances
No other script reads these folders: the analyses glob only the top level of the
result folders (scripts/p89_public_family.py check_duplicates lists the folders
test_results/PUBFAMILY/<set>*, not their subfolders).

JOBS run with one thread, each pinned to one core of the given list; an output is
written under a temporary name in its subfolder and renamed after its check; a
finished output that passes its check is skipped; one that fails is moved to
<subfolder>/invalid/ and its job runs again.

--rules-only keeps the jobs that need no trained run (the rule traces, the tie-break seeds,
the validation records), so they can run while the runs train.

Usage (the launcher calls these; every command is run from the repository root):
  python scripts/rulecmp_plan.py plan --family public --cores 16-23 [--all-commands]
  python scripts/rulecmp_plan.py check --family public
  python scripts/rulecmp_plan.py preflight --family public --cores 16-23
  python scripts/rulecmp_plan.py run --family public --cores 16-23
  python scripts/rulecmp_plan.py plan --family public --mask active [--set <set>] [--rules-only]
"""
import argparse
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p109_public_active_rules as P109  # noqa: E402  (the rule arrays under the active mask)

RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
MASKS = ('nondelay', 'active')           # the masks of p90 and of params.py --mask_mode
# macro prefix of the public family under each mask (another public test set adds its tag)
PUB_MACRO = {'nondelay': 'Rcp', 'active': 'RcAct'}
R8 = tuple(REG.COMBOS)
K_DEFAULT = 32                           # sampled schedules per instance
K_ENV = 'RULECMP_K'                      # the environment variable that overrides it
K_MIN = 16                               # the largest k of form (b)
STORED_TIE_SEED = 50                     # params.py default seed_test: the stored rule arrays
TIE_SEEDS = (50, 51, 52, 53)
SUB = {'sample': 'rulecmp_sampled', 'trace': 'rulecmp_trace', 'ties': 'rulecmp_ties',
       'vali': 'rulecmp_vali'}
KINDS = tuple(SUB)
MASK = ['--first_mask', 'exact', '--mask_ties', 'exact']
XS = REG.FOLDER_SUFFIX['exact']
PREFLIGHT_COMBO = 'S'
PREFLIGHT_N = 2
ARMS = ('released', 'control', 'specialist')
HEADLINE = 'pub_SD2_test_10x5+mix'
# public test set -> (macro tag, expected records, records left out): the table of
# scripts/p89_public_family.py SETS (the tests compare the two). The headline set
# keeps the prefix Rcp alone; the others write Rcp<tag> into every macro name.
PUB_SET_INFO = {
    'pub_SD2_test_10x5+mix': ('Sdten', 100, ()),
    'pub_SD2_test_20x10+mix': ('Sdtwenty', 100, ()),
    'pub_Brandimarte': ('Brand', 10, ()),
    'pub_Hurink_edata': ('Edata', 65, ('orb7',)),
    'pub_Hurink_rdata': ('Rdata', 65, ('orb7',)),
    'pub_Hurink_vdata': ('Vdata', 65, ('orb7',)),
}
HOLD = 'train_log/RULECMP_HOLD'
NOTES = 'train_log/rulecmp_launch_notes.txt'

# Wall seconds per job on one core, public family (SD2 10x5, 100 test records), from
# the logs of the study of 2026-11-11 (one thread, one efficiency core): sampling
# K = 100 took 95.7 to 115.2 s per run and combination (mean 101.0 over 70 files);
# a greedy policy trace 11.6 s and a rule trace 3.6 s per combination; the four
# masked rules on 100 records and eight combinations 113 s. STARTUP is the
# interpreter, torch and model loading of one process.
PUBLIC_SECONDS = {'sample': 101.0, 'trace_policy': 11.6, 'trace_rule': 3.6,
                  'rules_100': 113.0}
SAMPLE_K_MEASURED = 100                  # the K of both sampling measurements
# Factory sampling: 26 CPU seconds per test instance at K = 100, measured on two
# instances of S with the earlier checkpoint 10x25+family+jointblind-v1 (one thread,
# a shared core: 53 CPU s and 107 wall s for the two instances).
FACTORY_SAMPLE_CPU_PER_INSTANCE = 26.0
STARTUP = 3.0
# The trainer keeps min(vali_size, 20) validation instances (train_family.py); every
# run of a family must agree on the settings that define them.
VALI_MAX = 20
VALI_KEYS = ('family_base', 'n_j', 'n_m', 'vali_size', 'seed_train_vali_datagen',
             'ppvc_mix', 'ppvc_factory', 'dr_shift', 'pub_vali_dir')


class Refusal(SystemExit):
    """A check failed; nothing starts."""


def refuse(msg):
    raise Refusal(f'REFUSING: {msg}')


def k_setting(value=None):
    """K, the sampled schedules per instance: `value`, else the environment
    variable RULECMP_K, else K_DEFAULT; at least K_MIN, the largest k of form (b)."""
    v = value if value not in (None, '') else (os.environ.get(K_ENV, '') or K_DEFAULT)
    try:
        k = int(v)
    except (TypeError, ValueError):
        refuse(f'K = {v!r} is not an integer')
    if str(k) != str(v).strip():
        refuse(f'K = {v!r} is not an integer')
    if k < K_MIN:
        refuse(f'K = {k}: the best of {K_MIN} needs at least {K_MIN} sampled schedules')
    return k


def mask_word(mask):
    """'' for the non-delay mask, else the word of the mask in run names (arm_registry.PUB_RULE_WORD)."""
    if mask in (None, 'nondelay'):
        return ''
    if mask not in REG.PUB_RULE_WORD:
        refuse(f'unknown mask {mask!r} (one of {", ".join(MASKS)})')
    return REG.PUB_RULE_WORD[mask]


def label_of(name, set_name=None, mask=None):
    """Label of the outputs of a family and public test set: the family name for
    the headline set (and the factory), public_<tag in lower case> otherwise; under
    the active mask public_act and public_act_<tag>."""
    w = mask_word(mask)
    if w and name != 'public':
        refuse(f'--mask {mask} applies to the public family only')
    base = f'{name}_{w}' if w else name
    if name != 'public' or not set_name or set_name == HEADLINE:
        return base
    if set_name not in PUB_SET_INFO:
        refuse(f'unknown public test set {set_name!r} (one of {", ".join(PUB_SET_INFO)})')
    return f'{base}_{PUB_SET_INFO[set_name][0].lower()}'


def select_rule(means):
    """Index of the rule with the lowest mean; a tie goes to the first in RULES order."""
    return int(np.argmin(np.asarray(means, float)))


def rel(p):
    """A path relative to the repository root when it lies inside it."""
    if not os.path.isabs(p):
        return p
    r = os.path.relpath(p, ROOT)
    return p if r.startswith('..') else r


# ----------------------------------------------------------------- families
class Run:
    """One trained run: arm, combination (specialists), seed, names and files."""

    def __init__(self, arm, combo, seed, model, checkpoint, config, log, regimes):
        self.arm, self.combo, self.seed = arm, combo, seed
        self.model, self.checkpoint, self.config, self.log = model, checkpoint, config, log
        self.regimes = tuple(regimes)


class Family:
    """Every name, folder and run of one family, from arm_registry."""

    def __init__(self, name, K=None, set_name=None, arms=None, mask=None):
        self.name = name
        self.K = k_setting(K)
        self.mask = mask or 'nondelay'
        self.word = mask_word(self.mask)
        self.headline, self.tag, self.label = True, '', label_of(name, None, self.mask)
        if name == 'public':
            self.prefix = REG.PUB_PREFIX
            self.seeds = tuple(REG.PUB_ACTFAM_SEEDS if self.word else REG.PUB_SEEDS)
            self.set = set_name or HEADLINE
            if self.set not in REG.PUB_SETS or self.set not in PUB_SET_INFO:
                refuse(f'unknown public test set {self.set!r} (one of {", ".join(REG.PUB_SETS)})')
            self.headline = self.set == HEADLINE
            self.p89_tag, self.n_expected, self.excluded = PUB_SET_INFO[self.set]
            if not self.headline:
                self.tag = self.p89_tag
                self.label = label_of(name, self.set, self.mask)
            self.data = f'data/PUBLIC/family/{self.set}'
            self.base = 'record'
            self.folder = f'{REG.PUB_RESULTS}/{self.set}{XS}'
            # the rule arrays of tie-break seed 50 and the rulecmp_* subfolders: the result
            # folder under the non-delay mask, the folder of p109 under another mask
            rules = P109.RULE_ROOT[self.mask] if self.word else REG.PUB_RESULTS
            self.rule_folder = self.campaign = f'{rules}/{self.set}{XS}'
            self.headline_folder = f'{rules}/{HEADLINE}{XS}'
            self.macro = PUB_MACRO[self.mask] + self.tag

            def mk(arm, combo, s):
                if self.word:
                    m = REG.pub_rule_model(arm, combo, s, word=self.word)
                    ck, lg, _, cf = REG.pub_rule_paths(arm, combo, s, word=self.word)
                    return self.prefix + m, ck, cf, lg
                m = REG.pub_model(arm, combo, s)
                ck, lg = REG.pub_paths(arm, combo, s)
                return self.prefix + m, ck, f'train_log/FAMILY/config_{self.prefix}{m}.json', lg
            self.vali_data = 'data/PUBLIC/family/pub_SD2_vali_10x5+mix'
            self.vali_dir = os.path.basename(self.vali_data) + XS
            self.vali_kind = 'records'
        elif name == 'factory':
            if set_name:
                refuse('--set applies to the public family only')
            if REG.env() != 'exact':
                refuse(f'the factory family is the exact-mask environment: set PACT_ENV=exact '
                       f'(active: {REG.env()})')
            self.prefix = REG.PREFIX
            self.seeds = tuple(REG.seeds())
            self.set = '10x25'
            self.data = 'data/FAMILY/10x25'
            self.base = 'data/PPVC/10x25+ppvc-mixed'
            self.folder = REG.res_dir('10x25')
            self.rule_folder = self.campaign = self.folder
            self.macro = 'Rcf'

            def mk(arm, combo, s):
                return (self.prefix + REG.model(arm, combo, s), REG.checkpoint(arm, combo, s),
                        REG.config_path(arm, combo, s), REG.train_log(arm, combo, s))
            self.vali_data = None
            self.vali_dir = '10x25_vali' + XS
            self.vali_kind = 'trainer'
        else:
            refuse(f'unknown family {name!r} (public or factory)')
        if arms is None or len(arms) == 0:
            arms = ARMS if self.headline else ('released',)
        bad = set(arms) - set(ARMS)
        arms = tuple(a for a in ARMS if a in set(arms))
        if bad or 'released' not in arms:
            refuse(f'arms {sorted(arms)}: choose from {", ".join(ARMS)}, the released policy '
                   f'included (its traces carry the measured variables)')
        self.arms = arms
        self.size_tag = os.path.basename(self.data.rstrip('/'))
        self.runs = []
        for arm, combos in (('released', [None]), ('control', [None]), ('specialist', list(R8))):
            if arm not in self.arms:
                continue
            for c in combos:
                for s in self.seeds:
                    m, ck, cf, lg = mk(arm, c, s)
                    self.runs.append(Run(arm, c, s, m, ck, cf, lg, R8 if c is None else (c,)))

    # ---- names and folders
    def released(self):
        return [r for r in self.runs if r.arm == 'released']

    def run_of(self, arm, combo, seed):
        c = combo if arm == 'specialist' else None
        return next(r for r in self.runs if (r.arm, r.combo, r.seed) == (arm, c, seed))

    def sub(self, kind, base=None):
        """Subfolder of one kind of output; `base` replaces the campaign folder
        <root>/<campaign> (p93 --input-root), the stored result folder under the
        non-delay mask."""
        return os.path.join(base or os.path.join(ROOT, self.campaign), SUB[kind])

    def names(self, R, data=None, root=None):
        """Instance names of a combination, in the sorted-file order every
        evaluation program walks."""
        d = os.path.join(root or ROOT, data or self.data, 'desc', R)
        return [os.path.basename(p)[:-len('.desc.json')]
                for p in sorted(glob.glob(os.path.join(d, '*.desc.json')))]

    def records_mtime(self, data=None, root=None):
        ps = glob.glob(os.path.join(root or ROOT, data or self.data, 'desc', '*', '*.desc.json'))
        return max(os.stat(p).st_mtime for p in ps) if ps else 0.0

    def greedy_path(self, run, R, root=None):
        return os.path.join(root or ROOT, self.folder, f'Result_greedy+{run.model}_{R}.npy')

    def rule_path(self, q, R, root=None):
        return os.path.join(root or ROOT, self.rule_folder, f'Result_{q}_{R}.npy')

    def sample_path(self, run, R, base=None):
        return os.path.join(self.sub('sample', base), f'sample{self.K}+{run.model}_{R}.npz')

    def trace_path(self, who, R, base=None):
        return os.path.join(self.sub('trace', base), f'trace+{who}_{R}.npz')

    def ties_dir(self, t, base=None, root=None):
        """Folder of the rule arrays with tie-break seed t (t = 50: the stored folder)."""
        if t == STORED_TIE_SEED:
            return os.path.join(root or ROOT, self.rule_folder)
        return os.path.join(self.sub('ties', base), f'seed{t}', self.size_tag + XS)

    def vali_folder(self, base=None, vali_base=None, root=None):
        """Folder of the rules on the validation instances. A public test set other
        than the headline has none of its own: it reads the headline set's folder
        (`vali_base` stands in for the headline result folder)."""
        if self.headline:
            return os.path.join(self.sub('vali', base), self.vali_dir)
        return os.path.join(vali_base or os.path.join(root or ROOT, self.headline_folder),
                            SUB['vali'], self.vali_dir)

    def check_records(self, root=None):
        """Reasons the public test records are not the set's dataset_meta.json list
        with its expected count and exclusions (the check of p89); [] if they are."""
        if self.name != 'public':
            return []
        p = os.path.join(root or ROOT, self.data, 'dataset_meta.json')
        if not os.path.exists(p):
            return [f'missing {self.data}/dataset_meta.json']
        with open(p) as f:
            meta = json.load(f)
        names = self.names('N', root=root)
        excluded = sorted(os.path.splitext(e['instance'])[0] for e in meta.get('excluded', []))
        if sorted(meta.get('instances', [])) != names or meta.get('n_instances') != \
                self.n_expected or len(names) != self.n_expected or \
                excluded != sorted(self.excluded) or set(self.excluded) & set(names):
            return [f'{self.set}: {len(names)} records, dataset_meta.json lists '
                    f'{meta.get("n_instances")} instances and excludes {excluded}; want '
                    f'{self.n_expected} records and the exclusions {list(self.excluded)}']
        return []

    def vali_source(self):
        """The run whose training configuration defines the factory validation
        instances, and the training logs whose MWKR norms must agree."""
        return self.run_of('released', None, self.seeds[0]), [r.log for r in self.runs]

    def vali_select(self, root=None):
        """(n, problems): the number of validation instances the training runs
        validated on, min(vali_size, 20), and the reasons the runs do not share one
        validation set (configurations that differ in VALI_KEYS; for the public
        records, a folder whose sorted listing is not the sorted record names). n is
        None while a configuration is missing."""
        root = root or ROOT
        seen, probs = {}, []
        for r in self.runs:
            p = os.path.join(root, r.config)
            if not os.path.exists(p):
                return None, []
            with open(p) as f:
                c = json.load(f)
            seen.setdefault(tuple(json.dumps(c.get(k), sort_keys=True) for k in VALI_KEYS),
                            []).append(r.model)
        if len(seen) > 1:
            groups = sorted(seen.items(), key=lambda kv: -len(kv[1]))
            ref = groups[0][0]
            for key, models in groups[1:]:
                diff = [f'{k} {b} (not {a})' for k, a, b in zip(VALI_KEYS, ref, key) if a != b]
                probs.append(f'{len(models)} run(s), e.g. {models[0]}, differ in the validation '
                             f'settings: ' + ', '.join(diff))
        size = json.loads(next(iter(seen))[VALI_KEYS.index('vali_size')])
        if size is None:
            return None, probs + ['the training configurations carry no vali_size']
        n = min(int(size), VALI_MAX)
        if self.vali_kind == 'records':
            d = os.path.join(root, self.vali_data, 'desc', 'N')
            listed = sorted(os.listdir(d))[:n] if os.path.isdir(d) else []
            want = [x + '.desc.json' for x in self.names('N', self.vali_data, root)[:n]]
            if listed != want:
                probs.append(f'the first {n} entries of {self.vali_data}/desc/N are not the '
                             f'first {n} sorted records')
        return n, probs

    def vali_n(self, root=None):
        if self.vali_kind == 'records':
            return len(self.names('N', self.vali_data, root))
        run, _ = self.vali_source()
        p = os.path.join(root or ROOT, run.config)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return min(int(json.load(f)['vali_size']), 20)


# ----------------------------------------------------------------- output checks
def mask_problem(z, mask):
    """None when the p90 output z (an open .npz) was built under `mask` (None: not
    checked), else the reason. A file without the key mask_mode counts as the
    non-delay mask (module docstring)."""
    if mask is None:
        return None
    got = str(z['mask_mode']) if 'mask_mode' in z.files else None
    if got is None and mask != 'nondelay':
        return f'records no mask (a file without the key counts as the non-delay mask), want the {mask} mask'
    if got is not None and got != mask:
        return f'built under the {got} mask, want the {mask} mask'
    return None


def check_sample(path, names, K, checkpoint_mtime=None, mask=None):
    """None if the sampled file holds K draws per instance and is complete and
    right (and built under `mask` when given), else the reason."""
    try:
        z = np.load(path, allow_pickle=False)
        keys = set(z.files)
        if not {'instances', 'makespan', 'feasible', 'K', 'seed_test'} <= keys:
            return f'keys {sorted(keys)}'
        why = mask_problem(z, mask)
        if why:
            return why
        if list(map(str, z['instances'])) != list(names):
            return 'instance names differ from the sorted records'
        ms = z['makespan']
        if int(z['K']) != K:
            return f'holds K = {int(z["K"])} sampled schedules per instance, not K = {K}'
        if int(z['seed_test']) != STORED_TIE_SEED:
            return f'seed_test {int(z["seed_test"])}'
        if ms.shape != (len(names), K) or not np.all(np.isfinite(ms)) or np.any(ms <= 0):
            return f'makespan shape {ms.shape} or values'
        if not z['feasible'].all():
            return 'an infeasible sampled schedule'
    except (OSError, ValueError, KeyError) as e:
        return f'unreadable ({e})'
    if checkpoint_mtime is not None and os.stat(path).st_mtime < checkpoint_mtime:
        return 'older than its checkpoint'
    return None


def load_ms(path):
    a = np.load(path)
    return (a[:, 0] if a.ndim == 2 else a).astype(float)


def check_trace(path, names, stored=None, checkpoint_mtime=None, mask=None):
    """None if the trace file is complete and its makespans equal the stored
    array `stored` (greedy or rule) when given (and built under `mask` when
    given), else the reason."""
    try:
        z = np.load(path, allow_pickle=True)
        if not {'instances', 'makespan', 'decisions', 'ops'} <= set(z.files):
            return f'keys {sorted(z.files)}'
        why = mask_problem(z, mask)
        if why:
            return why
        if list(map(str, z['instances'])) != list(names):
            return 'instance names differ from the sorted records'
        if len(z['decisions']) != len(names) or len(z['ops']) != len(names):
            return 'decision or operation records missing'
        ms = z['makespan']
        if ms.shape != (len(names),) or not np.all(np.isfinite(ms)) or np.any(ms <= 0):
            return f'makespan shape {ms.shape} or values'
        if stored is not None and not np.array_equal(ms, stored):
            return f'makespans differ from the stored array on {int((ms != stored).sum())} instances'
    except (OSError, ValueError, KeyError) as e:
        return f'unreadable ({e})'
    if checkpoint_mtime is not None and os.stat(path).st_mtime < checkpoint_mtime:
        return 'older than its checkpoint'
    return None


def summary_rows(folder):
    """{(regime, method): [(mean, feasible text), ...]} over the summary_*.md files of
    a folder (one file for an output of this pipeline; a copy may sit beside it)."""
    fs = sorted(glob.glob(os.path.join(folder, 'summary_*.md')))
    if not fs:
        return None
    out = {}
    for fn in fs:
        for ln in open(fn):
            c = [x.strip() for x in ln.strip().strip('|').split('|')]
            if len(c) == 7 and c[0] in R8:
                out.setdefault((c[0], c[1]), []).append((float(c[2]), c[4]))
    return out


def rule_mask_problem(folder, mask, seed=None):
    """None when the rule folder was rolled out under `mask` (and with the tie-break seed
    `seed` when given), else the reason. Under the non-delay mask no summary may name
    another mask (eval_family.py writes no such line); under another mask every summary
    records 'pairs: mask_mode=<mask>' and rules_meta.json (p109) names the mask and the
    seed."""
    fs = sorted(glob.glob(os.path.join(folder, 'summary_*.md')))
    lines = [ln.strip() for fn in fs for ln in open(fn) if ln.startswith('pairs: mask_mode=')]
    if mask == 'nondelay':
        other = sorted({ln for ln in lines if ln != 'pairs: mask_mode=nondelay'})
        return f'a summary records {other[0]!r}' if other else None
    want = f'pairs: mask_mode={mask}'
    for fn in fs:
        if want not in [ln.strip() for ln in open(fn)]:
            return f'{os.path.basename(fn)} does not record {want!r}'
    p = os.path.join(folder, P109.META)
    if not os.path.exists(p):
        return f'no {P109.META}'
    with open(p) as f:
        m = json.load(f)
    if m.get('mask_mode') != mask or (seed is not None and m.get('tie_break_seed') != seed):
        return (f'{P109.META} names the {m.get("mask_mode")} mask and the tie-break seed '
                f'{m.get("tie_break_seed")}, want the {mask} mask' +
                (f' and the seed {seed}' if seed is not None else ''))
    return None


def check_rule_folder(folder, n, regimes=R8, vali_json=False, mask=None, seed=None):
    """None if the folder holds Result_<rule>_<R>.npy [n, 2] for every rule and
    regime, and every summary row of it reports n/n feasible schedules and the
    array's mean (and, when `mask` is given, the folder was rolled out under it:
    rule_mask_problem), else the reason."""
    rows = summary_rows(folder)
    if rows is None:
        return 'no summary_*.md'
    if mask is not None:
        why = rule_mask_problem(folder, mask, seed)
        if why:
            return why
    for R in regimes:
        for q in RULES:
            p = os.path.join(folder, f'Result_{q}_{R}.npy')
            if not os.path.exists(p):
                return f'missing {os.path.basename(p)}'
            a = np.load(p)
            if a.shape != (n, 2) or not np.all(np.isfinite(a)) or np.any(a[:, 0] <= 0):
                return f'{os.path.basename(p)} shape {a.shape} or values'
            if (R, q) not in rows:
                return f'summary has no row {R} {q}'
            for mean, feas in rows[(R, q)]:
                if feas != f'{n}/{n}' or abs(mean - a[:, 0].mean()) > 0.05 + 1e-9:
                    return f'summary row {R} {q}: {feas} feasible, mean {mean}'
    if vali_json:
        p = os.path.join(folder, 'valirules.json')
        if not os.path.exists(p):
            return 'no valirules.json'
        with open(p) as f:
            v = json.load(f)
        if v.get('norm_check') != 'agree' or v.get('limit') or int(v.get('n', -1)) != n:
            return f'validation check {v.get("norm_check")!r}, n {v.get("n")}, limit {v.get("limit")}'
    return None


# ----------------------------------------------------------------- jobs
class Job:
    def __init__(self, kind, jid, est, needs, outputs):
        self.kind, self.id, self.est, self.needs, self.outputs = kind, jid, est, needs, outputs
        self.regimes = None          # trace jobs: combinations still to produce
        self.status, self.reason = 'open', ''


def py():
    return os.environ.get('PY', sys.executable)


def p90(fam, *args):
    """A p90 rollout command; under a mask other than the non-delay mask it carries
    --expect_mask, so that no schedule is built under another mask."""
    return [py(), 'scripts/p90_public_rule_comparison.py', 'rollout', *args,
            '--torch_threads', '1', '--device', 'cpu', '--device_id', '', *MASK] + \
        (['--expect_mask', fam.mask] if fam.word else [])


def p109(fam, set_name, tmp, seed=None):
    """A p109 run of the four rules under the family's mask on one set, written into tmp."""
    return [py(), 'scripts/p109_public_active_rules.py', 'run', '--mask', fam.mask, '--sets',
            set_name] + (['--seed', str(seed)] if seed is not None else []) + ['--out-root', tmp]


def ops_ratio(fam):
    """Mean operations per test instance of the family over that of the public
    family: the scale of the factory estimate."""
    if fam.name == 'public' and fam.headline:
        return 1.0
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from constraint_family.public_adapter import load_base_instance

    def mean_ops(data, base):
        out = []
        for p in sorted(glob.glob(os.path.join(ROOT, data, 'desc', 'N', '*.desc.json'))):
            with open(p) as f:
                rec = json.load(f)
            nm = os.path.basename(p)[:-len('.desc.json')]
            stem = rec['base_instance'] if base == 'record' else os.path.join(base, nm)
            out.append(load_base_instance(stem)[1].shape[0])
        return float(np.mean(out))
    # the factory reader imports params.py, which parses sys.argv on import
    cwd, argv = os.getcwd(), sys.argv
    os.chdir(ROOT)
    sys.argv = sys.argv[:1]
    try:
        return mean_ops(fam.data, fam.base) / mean_ops('data/PUBLIC/family/pub_SD2_test_10x5+mix',
                                                       'record')
    finally:
        os.chdir(cwd)
        sys.argv = argv


def sample_seconds(fam, n_test, ratio):
    """Estimated seconds of one sampling job: the measurement at K = 100 scaled
    linearly to the family's K (public: 101 wall s per job of the headline set, for
    another set also scaled by its operations per instance, `ratio`, and its number
    of instances; factory: 26 CPU s per instance)."""
    f = fam.K / SAMPLE_K_MEASURED
    if fam.name == 'public':
        scale = 1.0 if fam.headline else ratio * n_test / 100.0
        return STARTUP + PUBLIC_SECONDS['sample'] * f * scale
    return STARTUP + FACTORY_SAMPLE_CPU_PER_INSTANCE * n_test * f


def jobs(fam, ratio=None, kinds=KINDS, rules_only=False):
    """Every job of the family with its status: done, open, invalid (an output
    exists and fails its check; moved aside before the job runs) or blocked (an
    input is missing). rules_only: the jobs of the rules only (no sampling, no
    policy trace)."""
    ratio = ops_ratio(fam) if ratio is None else ratio
    n_test = len(fam.names('N'))
    out = []
    ck_mtime = {r.model: (os.stat(os.path.join(ROOT, r.checkpoint)).st_mtime
                          if os.path.exists(os.path.join(ROOT, r.checkpoint)) else None)
                for r in fam.runs}

    def run_needs(r):
        return [r.checkpoint, r.config]

    if 'sample' in kinds and not rules_only:
        per_job = sample_seconds(fam, n_test, ratio)
        for r in fam.runs:
            for R in r.regimes:
                j = Job('sample', f'sample/{r.model}_{R}', per_job, run_needs(r),
                        [fam.sample_path(r, R)])
                j.run, j.R = r, R
                p = j.outputs[0]
                if os.path.exists(p):
                    why = check_sample(p, fam.names(R), fam.K, ck_mtime[r.model], fam.mask)
                    j.status, j.reason = ('done', '') if why is None else ('invalid', why)
                out.append(j)
    if 'trace' in kinds:
        whos = ([] if rules_only else [(r.model, r) for r in fam.released()]) + \
            [(q, None) for q in RULES]
        for who, r in whos:
            per = PUBLIC_SECONDS['trace_policy' if r else 'trace_rule'] * ratio * n_test / 100.0
            needs = (run_needs(r) + [rel(fam.greedy_path(r, R)) for R in R8]) if r else \
                [rel(fam.rule_path(q, R)) for q in [who] for R in R8]
            j = Job('trace', f'trace/{who}', STARTUP + per * len(R8), needs,
                    [fam.trace_path(who, R) for R in R8])
            j.run, j.who = r, who
            queue, bad = [], []
            for R, p in zip(R8, j.outputs):
                if not os.path.exists(p):
                    queue.append(R)
                    continue
                sp = fam.greedy_path(r, R) if r else fam.rule_path(who, R)
                stored = load_ms(sp) if os.path.exists(sp) else None
                why = check_trace(p, fam.names(R), stored, ck_mtime[r.model] if r else None,
                                  fam.mask)
                if why is not None:
                    queue.append(R)
                    bad.append(f'{R}: {why}')
            j.regimes = queue
            j.est = STARTUP + per * len(queue) if queue else 0.0
            if not queue:
                j.status = 'done'
            elif bad:
                j.status, j.reason = 'invalid', '; '.join(bad)
            out.append(j)
    if 'ties' in kinds:
        for t in TIE_SEEDS[1:]:
            d = fam.ties_dir(t)
            j = Job('ties', f'ties/seed{t}', STARTUP + PUBLIC_SECONDS['rules_100'] * ratio *
                    n_test / 100.0, [rel(fam.rule_path(q, R)) for q in RULES for R in R8],
                    [os.path.dirname(d)])
            j.t = t
            if os.path.isdir(os.path.dirname(d)):
                why = check_rule_folder(d, n_test, mask=fam.mask, seed=t)
                j.status, j.reason = ('done', '') if why is None else ('invalid', why)
            out.append(j)
    if 'vali' in kinds and fam.headline:
        nv = fam.vali_n()
        d = fam.vali_folder()
        if fam.vali_kind == 'records':
            needs = []
            est = STARTUP + PUBLIC_SECONDS['rules_100'] * nv / 100.0
        else:
            src, logs = fam.vali_source()
            needs = [src.config] + logs
            nv_est = nv if nv is not None else 20
            # the rules plus the MWKR norm check (one classical rule per instance)
            est = STARTUP + PUBLIC_SECONDS['rules_100'] * ratio * nv_est / 100.0 * 1.25
        j = Job('vali', 'vali', est, needs, [d])
        if os.path.isdir(d) and nv is not None:
            why = check_rule_folder(d, nv, vali_json=fam.vali_kind == 'trainer', mask=fam.mask,
                                    seed=STORED_TIE_SEED)
            j.status, j.reason = ('done', '') if why is None else ('invalid', why)
        out.append(j)
    for j in out:
        miss = [p for p in j.needs if not os.path.exists(os.path.join(ROOT, p))]
        if miss and j.status != 'done':
            j.status, j.reason = 'blocked', f'missing {miss[0]}' + (
                f' and {len(miss) - 1} more' if len(miss) > 1 else '')
            j.missing = miss
        elif fam.word and j.status != 'done' and getattr(j, 'run', None) is not None \
                and not run_finished(j.run):
            # the family under a mask is planned while its runs train: a policy job waits
            # for the training log's completion line, so it never reads an interim checkpoint
            j.status, j.reason = 'blocked', f'the training of {j.run.model} has not finished'
            j.missing = [j.run.log]
    return out


def run_finished(run, root=None):
    """True when the run's training log holds the trainer's completion line."""
    p = os.path.join(root or ROOT, run.log)
    return os.path.exists(p) and 'done in' in open(p, errors='replace').read()


def command(fam, j, tmp):
    """The command of a job writing into the temporary folder tmp."""
    if j.kind == 'sample':
        return p90(fam, '--mode', 'sample', '--K', str(fam.K), '--family_model', j.run.model,
                   '--data', fam.data, '--base', fam.base, '--regimes', j.R, '--out', tmp)
    if j.kind == 'trace':
        who = ['--family_model', j.who] if j.run else \
            ['--rule', j.who] + (['--rule_mask', fam.mask] if fam.word else [])
        return p90(fam, '--mode', 'trace', *who, '--data', fam.data, '--base', fam.base,
                   '--regimes', *(j.regimes or R8), '--out', tmp)
    if j.kind == 'ties':
        if fam.word:
            return p109(fam, fam.set, tmp, j.t)
        return [py(), 'eval_family.py', '--family_model', 'rules', '--data', fam.data,
                '--base', fam.base, '--regimes', *R8, '--methods', *RULES, '--rule_set',
                'masked', '--seed_test', str(j.t), '--out_root', tmp, '--tag', XS, *MASK,
                '--device', 'cpu', '--device_id', '']
    if j.kind == 'vali':
        if fam.word:
            return p109(fam, os.path.basename(fam.vali_data), tmp)
        if fam.vali_kind == 'records':
            return [py(), 'eval_family.py', '--family_model', 'rules', '--data', fam.vali_data,
                    '--base', 'record', '--regimes', *R8, '--methods', *RULES, '--rule_set',
                    'masked', '--out_root', tmp, '--tag', XS, *MASK, '--device', 'cpu',
                    '--device_id', '']
        src, logs = fam.vali_source()
        return p90(fam, '--mode', 'valirules', '--family_model', src.model, '--regimes', *R8,
                   '--out', os.path.join(tmp, fam.vali_dir), '--check_logs', *logs)
    raise ValueError(j.kind)


def job_env():
    e = {k: v for k, v in os.environ.items() if k not in ('PACT_FIRST_MASK', 'PACT_MASK_TOL')}
    e.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
             CUDA_VISIBLE_DEVICES='')
    return e


def tmp_dir(fam, j):
    return os.path.join(fam.sub(j.kind), '.tmp_' + j.id.replace('/', '_'))


def finalize(fam, j, tmp):
    """Check the outputs a job wrote into tmp and move them into place; returns
    None or the reason of a failure (the outputs then stay where they were)."""
    if j.kind == 'sample':
        src = os.path.join(tmp, os.path.basename(j.outputs[0]))
        if not os.path.exists(src):
            return 'no output file'
        why = check_sample(src, fam.names(j.R), fam.K, mask=fam.mask)
        if why:
            return why
        os.replace(src, j.outputs[0])
    elif j.kind == 'trace':
        pairs = []
        for R in j.regimes:
            src = os.path.join(tmp, os.path.basename(fam.trace_path(j.who, R)))
            if not os.path.exists(src):
                return f'no output file for {R}'
            sp = fam.greedy_path(j.run, R) if j.run else fam.rule_path(j.who, R)
            why = check_trace(src, fam.names(R), load_ms(sp), mask=fam.mask)
            if why:
                return f'{R}: {why}'
            pairs.append((src, fam.trace_path(j.who, R)))
        for src, dst in pairs:
            os.replace(src, dst)
    elif j.kind == 'ties':
        why = check_rule_folder(os.path.join(tmp, fam.size_tag + XS), len(fam.names('N')),
                                mask=fam.mask, seed=j.t)
        if why:
            return why
        os.rename(tmp, j.outputs[0])
        return None
    elif j.kind == 'vali':
        src = os.path.join(tmp, fam.vali_dir)
        nv = fam.vali_n()
        why = check_rule_folder(src, nv, vali_json=fam.vali_kind == 'trainer', mask=fam.mask,
                                seed=STORED_TIE_SEED)
        if why:
            return why
        os.rename(src, j.outputs[0])
    shutil.rmtree(tmp, ignore_errors=True)
    return None


def move_invalid(j):
    """Move the outputs of an invalid job aside (never overwritten)."""
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for p in j.outputs:
        if j.kind == 'trace' and os.path.basename(p).split('_')[-1][:-4] not in (j.regimes or ()):
            continue
        if os.path.exists(p):
            d = os.path.join(os.path.dirname(p), 'invalid')
            os.makedirs(d, exist_ok=True)
            os.rename(p, os.path.join(d, f'{os.path.basename(p)}.{stamp}'))


# ----------------------------------------------------------------- plan text
def parse_cores(spec):
    out = []
    for part in str(spec).replace(' ', '').split(','):
        if not part:
            continue
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    if not out or len(set(out)) != len(out):
        refuse(f'core list {spec!r} is empty or repeats a core')
    return out


def simulate(ests, n_cores):
    """Wall clock of the jobs, longest first, one per core."""
    free = [0.0] * max(1, n_cores)
    for e in sorted(ests, reverse=True):
        i = int(np.argmin(free))
        free[i] += e
    return max(free) if ests else 0.0


def missing_inputs(fam):
    """Checkpoints, configurations, training logs and stored arrays that are absent."""
    miss = {'checkpoints': [], 'configurations': [], 'training logs': [], 'stored arrays': []}
    for r in fam.runs:
        for key, p in (('checkpoints', r.checkpoint), ('configurations', r.config),
                       ('training logs', r.log)):
            if not os.path.exists(os.path.join(ROOT, p)):
                miss[key].append(p)
        for R in r.regimes:
            p = fam.greedy_path(r, R)
            if not os.path.exists(p):
                miss['stored arrays'].append(rel(p))
    for q in RULES:
        for R in R8:
            p = fam.rule_path(q, R)
            if not os.path.exists(p):
                miss['stored arrays'].append(rel(p))
    return miss


def validation_rules(fam):
    """({R: rule} selected on the first min(vali_size, 20) validation instances,
    reason) from the validation folder the family's form (c) reads; ({}, reason)
    while that folder is absent or fails its check."""
    folder = fam.vali_folder()
    nv, (ns, probs) = fam.vali_n(), fam.vali_select()
    if probs or ns is None or nv is None:
        return {}, 'the validation instances are undefined (see above)'
    if not os.path.isdir(folder):
        return {}, f'not yet: {rel(folder)} does not exist (the vali job of the headline set)'
    why = check_rule_folder(folder, nv, vali_json=fam.vali_kind == 'trainer', mask=fam.mask,
                            seed=STORED_TIE_SEED)
    if why:
        return {}, f'{rel(folder)} fails its check: {why}'
    out = {}
    for R in R8:
        m = [load_ms(os.path.join(folder, f'Result_{q}_{R}.npy'))[:ns].mean() for q in RULES]
        out[R] = RULES[select_rule(m)]
    return out, f'from {rel(folder)}, first {ns} instances'


def describe(fam, js, ratio, cores, all_commands=False):
    L = []
    n_test = len(fam.names('N'))
    L.append(f'rule comparison, family {fam.name}: {n_test} test records in {fam.data} '
             f'(base {fam.base}), stored arrays in {fam.folder}')
    if fam.word:
        L.append(f'under the {fam.mask} mask: runs trained under it (arm_registry.pub_rule_model, '
                 f'e.g. {fam.runs[0].model}), every schedule built under it (p90 --expect_mask '
                 f'{fam.mask}; rule traces p90 --rule_mask {fam.mask}; tie-break seeds and '
                 f'validation records scripts/p109_public_active_rules.py); stored rule arrays in '
                 f'{fam.rule_folder}; outputs in the rulecmp_* subfolders of {fam.campaign}; '
                 f'label {fam.label}, macro prefix {fam.macro}')
    if fam.name == 'public' and not fam.headline:
        L.append(f'public test set {fam.set} (not the headline set {HEADLINE}): macro prefix '
                 f'{fam.macro}, outputs labelled {fam.label}; arms sampled: {", ".join(fam.arms)}')
    L += [f'  RECORDS: {x}' for x in fam.check_records()]
    L.append(f'runs: {len(fam.released())} released, '
             f'{sum(r.arm == "control" for r in fam.runs)} control, '
             f'{sum(r.arm == "specialist" for r in fam.runs)} specialist; seeds '
             f'{",".join(map(str, fam.seeds))}; first run {fam.runs[0].model}')
    L.append(f'sampled schedules per instance: K = {fam.K} ({K_ENV} or the launcher\'s K; '
             f'default {K_DEFAULT}); file names sample{fam.K}+<run>_<R>.npz')
    vn = fam.vali_n()
    ns, probs = fam.vali_select()
    nst = ns if ns is not None else '?'
    L.append(f'validation instances (form (c)): '
             + (f'the rule selected on the headline set\'s validation instances (the first '
                f'{nst} of the {vn} records of {fam.vali_data}, the instances the training runs '
                f'validated on), applied to this set\'s stored rule arrays; no validation job '
                f'for this set' if fam.vali_kind == 'records' and not fam.headline else
                f'the first {nst} of the {vn} records of {fam.vali_data}, the instances the '
                f'training runs validated on; all {vn} are evaluated for the second selection'
                if fam.vali_kind == 'records' else
                f'the {nst} instances the trainer builds from the configuration of '
                f'{fam.vali_source()[0].model}, MWKR norms checked against '
                f'{len(fam.vali_source()[1])} training logs'))
    L += [f'  VALIDATION SETTINGS DIFFER: {x}' for x in probs]
    vr, why = validation_rules(fam)
    L.append('validation rule per combination (form (c)): ' +
             (', '.join(f'{R} {q}' for R, q in vr.items()) + f' ({why})' if vr else why))
    L.append('outputs: ' + ', '.join(rel(fam.sub(k)) for k in KINDS
                                     if k != 'vali' or fam.headline))
    L.append('')
    L.append(f'{"kind":<7} {"jobs":>5} {"done":>5} {"to run":>6} {"invalid":>7} {"blocked":>7} '
             f'{"core-h to run":>13}')
    tot_h = 0.0
    for k in KINDS:
        sel = [j for j in js if j.kind == k]
        run = [j for j in sel if j.status in ('open', 'invalid')]
        blk = [j for j in sel if j.status == 'blocked']
        h = sum(j.est for j in run + blk) / 3600.0
        tot_h += h
        L.append(f'{k:<7} {len(sel):>5} {sum(j.status == "done" for j in sel):>5} '
                 f'{sum(j.status == "open" for j in sel):>6} '
                 f'{sum(j.status == "invalid" for j in sel):>7} {len(blk):>7} {h:>13.2f}')
    pending = [j.est for j in js if j.status != 'done']
    L.append(f'estimated core-hours of the jobs not done: {tot_h:.2f}')
    f = fam.K / SAMPLE_K_MEASURED
    if fam.name == 'public' and fam.headline:
        L.append(f'  sampling: {PUBLIC_SECONDS["sample"]:.0f} s per job measured at K = '
                 f'{SAMPLE_K_MEASURED} in the study of 2026-11-11, scaled linearly to K = {fam.K} '
                 f'(x{f:.2f}); the other jobs: the wall times of that study (one core, one thread)')
    elif fam.name == 'public':
        L.append(f'  an estimate: the headline-set wall times (sampling {PUBLIC_SECONDS["sample"]:.0f} '
                 f's per job at K = {SAMPLE_K_MEASURED}, 100 instances) scaled by the operations per '
                 f'instance of this set (x{ratio:.2f}) and its {n_test} instances (x{n_test / 100:.2f}), '
                 f'and the sampling linearly to K = {fam.K} (x{f:.2f}); nothing was measured on this set')
        smp = sum(j.est - STARTUP for j in js if j.kind == 'sample' and j.status != 'done')
        L.append(f'  the factory measurement grew about as the square of the operation ratio '
                 f'(x4.40 operations, x25.7 time per instance); scaled that way the sampling takes '
                 f'{smp * ratio / 3600.0:.2f} core-hours instead of {smp / 3600.0:.2f}')
    else:
        L.append(f'  sampling: {FACTORY_SAMPLE_CPU_PER_INSTANCE:.0f} CPU s per instance measured '
                 f'at K = {SAMPLE_K_MEASURED} on two instances with the earlier checkpoint '
                 f'10x25+family+jointblind-v1, scaled linearly to K = {fam.K} (x{f:.2f}); the other '
                 f'jobs: the public wall times scaled by the operations per instance, '
                 f'x{ratio:.2f} (an estimate)')
    L.append('  linear scaling in K gives a lower bound: the part of the cost of a decision that '
             'does not depend on K does not shrink with it')
    if fam.word:
        L.append(f'  the rule jobs under the {fam.mask} mask are estimated at the wall times under '
                 f'the non-delay mask; the {fam.mask} mask admits more pairs per decision')
    if cores:
        L.append(f'estimated wall clock on {len(cores)} cores ({",".join(map(str, cores))}): '
                 f'{simulate(pending, len(cores)) / 3600.0:.2f} h, plus the preflight '
                 f'(under 2 minutes)')
    miss = missing_inputs(fam)
    L.append('')
    for key, v in miss.items():
        L.append(f'missing {key}: {len(v)}' + (':' if v else ''))
        L += [f'  {p}' for p in v[:60]]
        if len(v) > 60:
            L.append(f'  ... and {len(v) - 60} more')
    bad = [j for j in js if j.status == 'invalid']
    if bad:
        L.append('')
        L.append('outputs that fail their check (moved to invalid/ before their job runs again):')
        L += [f'  {j.id}: {j.reason}' for j in bad]
    L.append('')
    L.append('jobs (status, estimated seconds on one core):')
    for j in js:
        extra = f' {j.reason}' if j.status == 'blocked' else ''
        L.append(f'  {j.id:<58} {j.status:<8} {j.est:7.0f}{extra}')
    L.append('')
    L.append('commands (one thread, pinned to one core of CORES by the launcher; <tmp> is the '
             'temporary folder of the job):')
    shown = set()
    for j in js:
        if j.status == 'done':
            continue
        if not all_commands and j.kind in shown:
            continue
        shown.add(j.kind)
        cmd = command(fam, j, '<tmp>')
        L.append(f'  [{j.id}] ' + ' '.join(x if x else "''" for x in cmd))
    if not all_commands:
        L.append('  (one command per kind shown; --all-commands prints every job)')
    return L


# ----------------------------------------------------------------- machine state
def proc_snapshot():
    snap = {}
    for tdir in glob.glob('/proc/[0-9]*/task/[0-9]*'):
        try:
            s = open(tdir + '/stat').read()
        except OSError:
            continue
        f = s[s.rfind(')') + 2:].split()
        snap[tdir] = (int(f[11]) + int(f[12]), int(f[36]))
    return snap


def cmdline(pid):
    try:
        return open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(
            errors='replace').strip()[:150]
    except OSError:
        return '?'


def ancestors():
    out, pid = set(), os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            pid = int(open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def busy_on(cores, window=2.0):
    """Processes on the given cores (threads above 5% of a core over the window,
    and processes whose lifetime share is above 5% and that last ran there), and
    every process above 5%, as the timed campaign records them."""
    cores, mine = set(cores), ancestors()
    a = proc_snapshot()
    time.sleep(window)
    b = proc_snapshot()
    hz = os.sysconf('SC_CLK_TCK')
    busy = {}
    for k, (tb, cpu) in b.items():
        pid = int(k.split('/')[2])
        if cpu not in cores or pid in mine:
            continue
        pct = 100.0 * (tb - a.get(k, (tb, cpu))[0]) / hz / window
        if pct > 5:
            busy[pid] = f'pid {pid} thread {k.split("/")[4]} core {cpu} {pct:.0f}% now: {cmdline(pid)}'
    # The listing process is a child of this one and may run on a requested core: it
    # is left out by its pid.
    lister = subprocess.Popen(['ps', '-eo', 'pid,psr,pcpu,etime,args', '--no-headers'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    ps = lister.communicate()[0].splitlines()
    every = []
    for line in ps:
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        pid, psr, pcpu = int(f[0]), int(f[1]), float(f[2])
        if pid == lister.pid:
            continue
        if pcpu > 5:
            every.append(line[:160])
            if psr in cores and pid not in mine and pid not in busy:
                busy[pid] = f'pid {pid} core {psr} {pcpu:.0f}% (lifetime): {f[4][:120]}'
    return list(busy.values()), every


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or '?'
    except OSError:
        return '?'


def launch_note(fam, js, cores, busy, every, kinds):
    run = [j for j in js if j.status in ('open', 'invalid')]
    per = {k: sum(j.kind == k for j in run) for k in KINDS}
    n_test = len(fam.names('N'))
    return [
        f'=== rule comparison {time.strftime("%Y-%m-%d %H:%M:%S")}, commit {git_commit()}, '
        f'family {fam.name}' + (f', set {fam.set}, arms {" ".join(fam.arms)}'
                                if fam.name == 'public' and not fam.headline else '') +
        f', K = {fam.K}, cores {",".join(map(str, cores))}, '
        f'kinds {" ".join(kinds)}, '
        f'{len(run)} job(s): ' + ', '.join(f'{k} {v}' for k, v in per.items()),
        '1. Purpose: the arrays of the comparison with dispatching rules in three forms '
        '(greedy against the best of four rules; best of 4 and 16 sampled against the best of '
        '4 and 16 rule schedules; greedy against the rule chosen on the validation instances) '
        'and the decision traces; read by scripts/p93_rule_comparison.py into '
        f'manuscript/macros_rulecmp_{fam.label}.tex.',
        '2. Expected: every job writes its arrays and they pass their checks (greedy traces '
        'equal the stored greedy arrays, rule traces the stored rule arrays, every sampled and '
        'rule schedule feasible); the preflight showed the trace, the sampling and the tie-break '
        'seed 50 reproduce the stored arrays. A failed check leaves the job undone.',
        '3. Contamination: new files only, in the rulecmp_* subfolders, written under a '
        'temporary name and renamed after the check; an existing output that fails its check '
        'is moved to invalid/; the evaluations are deterministic, so sharing a core changes '
        'their wall time and not their results; the cores are checked below so that no timed '
        'experiment shares them.',
        f'4. Data: {n_test} test records per combination in {fam.data} (base {fam.base}), stored '
        f'arrays in {fam.folder}, checkpoints {fam.runs[0].checkpoint} and the other '
        f'{len(fam.runs) - 1} runs of the registry, validation instances '
        + (fam.vali_data if fam.vali_kind == 'records'
           else f'of the training configuration of {fam.vali_source()[0].model}') + '.',
        f'load average {" ".join(open("/proc/loadavg").read().split()[:3])}',
        'processes above 5% CPU, with the core they last ran on:',
    ] + [f'  {x}' for x in every] + (
        ['processes on the requested cores:'] + [f'  {x}' for x in busy] if busy else
        ['no other process above 5% on the requested cores'])


# ----------------------------------------------------------------- commands
def cmd_plan(fam, a):
    ratio = ops_ratio(fam)
    js = jobs(fam, ratio, a.kinds, a.rules_only)
    print('\n'.join(describe(fam, js, ratio, parse_cores(a.cores) if a.cores else [],
                             a.all_commands)))
    return 0


def cmd_check(fam, a):
    """Refuse while a checkpoint, configuration, log or stored array is missing (with
    --rules-only: a stored rule array)."""
    ratio = ops_ratio(fam)
    js = jobs(fam, ratio, a.kinds, a.rules_only)
    print('\n'.join(describe(fam, js, ratio, parse_cores(a.cores) if a.cores else [])))
    miss = missing_inputs(fam)
    if a.rules_only:
        miss = {'stored arrays': [rel(fam.rule_path(q, R)) for q in RULES for R in R8
                                  if not os.path.exists(fam.rule_path(q, R))]}
    blocked = [j for j in js if j.status == 'blocked']
    _, probs = fam.vali_select()
    if probs:
        print('REFUSING to start: the runs do not share one validation set:\n  ' +
              '\n  '.join(probs))
        return 1
    recs = fam.check_records()
    if recs:
        print('REFUSING to start: ' + '; '.join(recs))
        return 1
    if any(miss.values()) or blocked:
        print(f'REFUSING to start: {sum(map(len, miss.values()))} input file(s) missing, '
              f'{len(blocked)} job(s) blocked (listed above)')
        return 1
    return 0


def run_logged(cmd, core, log):
    with open(log, 'w') as f:
        return subprocess.run(['taskset', '-c', str(core)] + cmd, cwd=ROOT, env=job_env(),
                              stdout=f, stderr=subprocess.STDOUT).returncode


def cmd_preflight(fam, a):
    """On PREFLIGHT_N instances of PREFLIGHT_COMBO: the greedy trace of the
    released seed reproduces the stored greedy makespans, the best of the K
    sampled schedules equals what eval_family.py --methods sampling keeps, and
    eval_family.py with tie-break seed 50 reproduces the stored rule arrays (under another
    mask p109 check-arrays reproduces the arrays of its folder). With --rules-only the rules
    alone are checked."""
    cores = parse_cores(a.cores)
    core = cores[0]
    busy, _ = busy_on(cores)
    if busy:
        print(('STARTING although' if a.allow_busy else 'REFUSING:') +
              ' processes run on the requested cores:\n  ' + '\n  '.join(busy))
        if not a.allow_busy:
            return 1
    R, n = PREFLIGHT_COMBO, PREFLIGHT_N
    r = fam.released()[0]
    logdir = a.workdir or os.path.join(ROOT, 'train_log', f'rulecmp_{fam.label}')
    os.makedirs(logdir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix='preflight_', dir=logdir)
    t0 = time.time()
    fails = []

    def step(name, cmd):
        rc = run_logged(cmd, core, os.path.join(tmp, f'{name}.log'))
        if rc != 0:
            fails.append(f'{name}: exit {rc} (log {rel(os.path.join(tmp, name + ".log"))})')
        return rc == 0

    lim = ['--limit', str(n)]
    kp = fam.K
    rules_only = getattr(a, 'rules_only', False)
    if not rules_only and step('trace', p90(fam, '--mode', 'trace', '--family_model', r.model, '--data', fam.data,
                         '--base', fam.base, '--regimes', R, '--out', os.path.join(tmp, 'tr'),
                         *lim)):
        got = np.load(os.path.join(tmp, 'tr', f'trace+{r.model}_{R}.npz'),
                      allow_pickle=True)['makespan']
        want = load_ms(fam.greedy_path(r, R))[:n]
        if not np.array_equal(got, want):
            fails.append(f'greedy trace {got.tolist()} differs from the stored {want.tolist()}')
    if not rules_only and step('sample', p90(fam, '--mode', 'sample', '--K', str(kp), '--family_model', r.model,
                          '--data', fam.data, '--base', fam.base, '--regimes', R, '--out',
                          os.path.join(tmp, 'sp'), *lim)) and \
            step('sampling', [py(), 'eval_family.py', '--family_model', r.model, '--data',
                              fam.data, '--base', fam.base, '--regimes', R, '--methods',
                              'sampling', '--sample_times', str(kp), '--out_root',
                              os.path.join(tmp, 'ef'), '--tag', XS, *MASK, '--device', 'cpu',
                              '--device_id', '', '--torch_threads', '1', *lim]):
        z = np.load(os.path.join(tmp, 'sp', f'sample{kp}+{r.model}_{R}.npz'))
        got = z['makespan'].min(1)
        want = load_ms(os.path.join(tmp, 'ef', fam.size_tag + XS,
                                    f'Result_sampling+{r.model}_{R}.npy'))
        if not np.array_equal(got, want):
            fails.append(f'best of {kp} sampled {got.tolist()} differs from eval_family.py '
                         f'sampling {want.tolist()}')
        if not z['feasible'].all():
            fails.append('an infeasible sampled schedule')
        if mask_problem(z, fam.mask):
            fails.append(f'sampled schedules: {mask_problem(z, fam.mask)}')
    if fam.word:
        # the rules under the family's mask: p109 reproduces the stored arrays of its folder
        step('rules', [py(), 'scripts/p109_public_active_rules.py', 'check-arrays', '--mask',
                       fam.mask, '--sets', fam.set, '--combos', R, '--n', str(n), '--seed',
                       str(STORED_TIE_SEED), '--ref-root', os.path.dirname(fam.rule_folder)])
    elif step('rules', [py(), 'eval_family.py', '--family_model', 'rules', '--data', fam.data,
                        '--base', fam.base, '--regimes', R, '--methods', *RULES, '--rule_set',
                        'masked', '--seed_test', str(STORED_TIE_SEED), '--out_root',
                        os.path.join(tmp, 'ru'), '--tag', XS, *MASK, '--device', 'cpu',
                        '--device_id', '', *lim]):
        for q in RULES:
            got = load_ms(os.path.join(tmp, 'ru', fam.size_tag + XS, f'Result_{q}_{R}.npy'))
            want = load_ms(fam.rule_path(q, R))[:n]
            if not np.array_equal(got, want):
                fails.append(f'{q} with tie-break seed {STORED_TIE_SEED}: {got.tolist()} differs '
                             f'from the stored {want.tolist()}')
    dt = time.time() - t0
    if fails:
        print(f'PREFLIGHT FAILED ({dt:.0f} s; files kept in {rel(tmp)}):\n  ' + '\n  '.join(fails))
        return 1
    shutil.rmtree(tmp, ignore_errors=True)
    if rules_only:
        print(f'preflight passed in {dt:.0f} s on core {core}: on {n} instances of {R}, the four '
              f'rules with tie-break seed {STORED_TIE_SEED} equal the stored rule arrays')
        return 0
    print(f'preflight passed in {dt:.0f} s on core {core}: on {n} instances of {R}, the greedy '
          f'trace of {r.model} equals the stored greedy makespans, the best of {kp} sampled equals '
          f'eval_family.py sampling, and the four rules with tie-break seed {STORED_TIE_SEED} '
          f'equal the stored rule arrays')
    return 0


def cmd_run(fam, a):
    cores = parse_cores(a.cores)
    hold = os.path.join(ROOT, a.hold)
    ratio = ops_ratio(fam)
    js = jobs(fam, ratio, a.kinds, a.rules_only)
    blocked = [j for j in js if j.status == 'blocked']
    if blocked:
        print(f'REFUSING: {len(blocked)} job(s) blocked, e.g. {blocked[0].id}: {blocked[0].reason}')
        return 1
    queue = sorted([j for j in js if j.status in ('open', 'invalid')], key=lambda j: -j.est)
    busy, every = busy_on(cores)
    note = launch_note(fam, js, cores, busy, every, a.kinds)
    if busy:
        note.append('STARTED although the cores are not idle (--allow-busy)' if a.allow_busy
                    else 'REFUSED: the requested cores are not idle')
    os.makedirs(os.path.dirname(os.path.join(ROOT, a.notes)), exist_ok=True)
    with open(os.path.join(ROOT, a.notes), 'a') as f:
        f.write('\n'.join(note) + '\n')
    if busy and not a.allow_busy:
        print('REFUSING: processes run on the requested cores (launch note in '
              f'{a.notes}):\n  ' + '\n  '.join(busy))
        return 1
    logdir = a.logdir or os.path.join(ROOT, 'train_log', f'rulecmp_{fam.label}')
    os.makedirs(logdir, exist_ok=True)
    running = {}                      # core -> (job, Popen, tmp, t0)
    stop = {'now': False}

    def on_signal(sig, frame):
        stop['now'] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    say = lambda m: print(f'[rulecmp {time.strftime("%m-%d %H:%M:%S")}] {m}', flush=True)  # noqa: E731
    say(f'family {fam.name}: {len(queue)} job(s) on cores {",".join(map(str, cores))}; '
        f'hold file {a.hold}')
    failed, held = [], False
    while queue or running:
        if stop['now']:
            for core, (j, p, tmp, _) in running.items():
                p.terminate()
            for core, (j, p, tmp, _) in running.items():
                p.wait()
                shutil.rmtree(tmp, ignore_errors=True)
            say('stopped by a signal: running jobs killed, their temporary folders removed')
            return 143
        if os.path.exists(hold) and not held:
            held = True
            say(f'hold file {a.hold}: no new job; waiting for {len(running)} running job(s)')
        for core in cores:
            if core in running or not queue or held:
                continue
            j = queue.pop(0)
            if j.status == 'invalid':
                move_invalid(j)
            tmp = tmp_dir(fam, j)
            shutil.rmtree(tmp, ignore_errors=True)
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            if j.kind == 'sample' or j.kind == 'trace':
                os.makedirs(tmp)
            log = os.path.join(logdir, j.id.replace('/', '__') + '.log')
            cmd = command(fam, j, rel(tmp))
            with open(log, 'w') as lf:
                lf.write(' '.join(x if x else "''" for x in cmd) + '\n')
                lf.flush()
                p = subprocess.Popen(['taskset', '-c', str(core)] + cmd, cwd=ROOT,
                                     env=job_env(), stdout=lf, stderr=subprocess.STDOUT)
            running[core] = (j, p, tmp, time.time())
            say(f'core {core}: start {j.id} (estimate {j.est:.0f} s)')
        if not running:
            break
        time.sleep(a.poll)
        for core in list(running):
            j, p, tmp, t0 = running[core]
            rc = p.poll()
            if rc is None:
                continue
            del running[core]
            why = finalize(fam, j, tmp) if rc == 0 else f'exit {rc}'
            if why:
                failed.append(f'{j.id}: {why}')
                say(f'core {core}: FAILED {j.id} after {time.time() - t0:.0f} s: {why} '
                    f'(temporary folder {rel(tmp)} kept)')
            else:
                say(f'core {core}: done {j.id} in {time.time() - t0:.0f} s')
    left = len(queue)
    if failed:
        say(f'{len(failed)} job(s) failed:\n  ' + '\n  '.join(failed))
    if held:
        say(f'held by {a.hold}: {left} job(s) not started; remove the file and start again')
        return 0 if not failed else 1
    say('all jobs done' if not failed else 'finished with failures')
    return 1 if failed else 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('plan', 'check', 'preflight', 'run', 'label'))
    ap.add_argument('--family', required=True, choices=('public', 'factory'))
    ap.add_argument('--mask', default='nondelay', choices=MASKS,
                    help='public family: the mask of the runs and of every schedule (default '
                         'nondelay; active: the family trained under the active mask)')
    ap.add_argument('--rules-only', action='store_true',
                    help='only the jobs of the rules (rule traces, tie-break seeds, validation '
                         'records), which need no trained run')
    ap.add_argument('--logdir', default='',
                    help='run: folder of the job logs (default train_log/rulecmp_<label>)')
    ap.add_argument('--set', default='', help=f'public test set (default {HEADLINE})')
    ap.add_argument('--arms', nargs='*', default=None, choices=ARMS,
                    help='arms to sample (default: all three on the headline set, the '
                         'released policy elsewhere)')
    ap.add_argument('--cores', default='')
    ap.add_argument('--kinds', nargs='+', default=list(KINDS), choices=KINDS)
    ap.add_argument('--all-commands', action='store_true')
    ap.add_argument('--hold', default=HOLD)
    ap.add_argument('--notes', default=NOTES)
    ap.add_argument('--poll', type=float, default=2.0)
    ap.add_argument('--allow-busy', action='store_true',
                    help='run although other processes use the requested cores')
    ap.add_argument('--pidfile', default='', help='run: removed when the scheduler ends')
    ap.add_argument('--workdir', default='',
                    help='preflight: folder of its temporary files (default train_log/rulecmp_<family>)')
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd in ('preflight', 'run') and not a.cores:
        refuse(f'{a.cmd} needs --cores')
    os.chdir(ROOT)
    try:
        fam = Family(a.family, set_name=a.set or None, arms=a.arms, mask=a.mask)
        if a.cmd == 'label':
            print(fam.label)
            return 0
        return {'plan': cmd_plan, 'check': cmd_check, 'preflight': cmd_preflight,
                'run': cmd_run}[a.cmd](fam, a)
    finally:
        if a.pidfile and os.path.exists(a.pidfile):
            os.remove(a.pidfile)


if __name__ == '__main__':
    sys.exit(main())
