r"""P108: the public-family policy trained under the active mask against the policy.

Subcommands plan and infer. Analysis of stored arrays and records only: no training, no
evaluation, no solver, nothing on the GPU.

THE ARM
  Every public-family run (arm_registry.PUB_*) builds its schedules under the non-delay
  mask of the paper: the mask admits a pair of an operation and a machine when its
  processing start is within half a time quantum of the earliest start over the compatible
  pairs (mask_mode nondelay). The environment also offers the active mask, the condition of
  Giffler and Thompson for active schedules (mask_mode active): it admits a pair while its
  start is earlier than the earliest completion over the compatible pairs, so a machine may
  wait for a later job. Without training (scripts/p107_mask_rule_variants.py) the four
  dispatching rules give shorter schedules under the active mask on the public SD2 set at
  10x5. The arm pubx-act (arm_registry.PUB_ACT_*; scripts/run_pubfamily_queue.sh ARMS=act)
  is the released recipe (six combinations N, L, S, W, LS, LW, 2000 updates, the exact
  first-decision and tie options, 29,602 parameters) trained under the active mask with
  nothing else changed, training seeds 301 to 303, joined by the policy runs pubx-act-s304 and -s305 of the
  public family under the active mask (see SEEDS). eval_family.py restores mask_mode from a
  run's stored configuration, so its greedy schedules are built under the active mask
  (scripts/eval_newseeds_plan.py, groups pubact and pubactfam). This script compares it
  with the released policy pubx-v1 of the same seeds on the headline set
  pub_SD2_test_10x5+mix (100 base instances, all eight combinations), seed s against seed s.

INPUTS (relative to --root)
  For each run (pubx-act-s<seed> and pubx-v1-s<seed>, the seeds of SEEDS): the
  checkpoint trained_network/FAMILY/10x5+pubfamily+<run>.pth, the training log
  train_log/PUBFAMILY_<run>.log with its completion line, the validation record
  train_log/FAMILY/valiquality_10x5+pubfamily+<run>.txt, the configuration
  train_log/FAMILY/config_10x5+pubfamily+<run>.json, and in
  test_results/PUBFAMILY/pub_SD2_test_10x5+mix_xmask the eight arrays
  Result_greedy+10x5+pubfamily+<run>_<R>.npy and the summary
  summary_10x5+pubfamily+<run>_xmask.md. The records and the best-known table of the set,
  and the reference records of scripts/p91_public_cpsat_refs.py that
  scripts/p89_public_family.py reads (section 9), all 800 counted.
  Row i of an array is the i-th record of data/PUBLIC/family/<set>/desc/<R>/ in sorted
  file-name order (eval_family.py walks the sorted list).

CHECKS (before anything is computed; any failure refuses: the outputs are removed, the
problem is named, the exit status is non-zero)
  * The arm has every file listed above and no other file of its stem (a fourth seed, an
    array of another name) in the training folders and the headline folder, and every
    run has finished; the comparator has every file. plan lists what is missing.
  * Records, arrays, summaries and the regime-N bounds with the readers of p89
    (Inputs.check_records, _load, check_summary, check_lower_bounds): one sorted record
    order, shape [100, 2], finite positive makespans, n/n feasible, summary means and
    regime-N gaps equal to the arrays (a permutation of the rows changes the gap), regime-N
    makespans at or above the instance's bound. Every summary records the exact mask
    options. Every array is not older than its run's checkpoint, and the records are not
    newer than any array.
  * Training records (p89 Inputs.training_record): the 29,602-parameter network, the exact
    mask, validation lines equal to the validation record. The arm's log names the active
    mask (the trainer's line 'OPTIONS: mask_mode=active pair_extra=none'), the
    comparator's log names no other mask. Both runs of a seed validated up to the same
    update and on the same MWKR norms (one scale), and each log's closing best score equals
    the minimum of its record (arm_registry.validation_record).
  * Configurations: the arm's configuration equals the comparator's of the same seed except
    for model_suffix and mask_mode (active in the arm, nondelay or absent in the
    comparator); keys the trainer added after the comparator was written count as equal at
    their inert default (p89 EQ_INERT_NEW). Both record the six training combinations,
    2000 updates, the seed and the exact mask options.
  * References: every one of the 800 is counted by p89's reader (load_refs, ref_state); no
    counted regime-N reference contradicts the best-known table (p89 ref_problems); no
    makespan of the runs lies below the largest counted reference bound of its
    combination and sub-combinations.

QUANTITIES (all on pub_SD2_test_10x5+mix; units %)
  d = 100 (a - p) / p per training seed s and base instance, a the makespan of
  pubx-act-s<s>, p that of pubx-v1-s<s>; negative d means the policy under the active mask
  gives the shorter schedule.
  * Per combination (eight): the mean of d over seeds and instances; the 95% interval of
    the crossed hierarchical bootstrap over seeds and base instances
    (stats_family.hier_boot through p89 component, B = --boot, a fresh default_rng(11) per
    quantity); the seed-level t interval (2 degrees of freedom); the direction (below
    zero: ahead, above: behind, otherwise neither); the margin verdicts at 0.5% and 1%;
    the per-seed means; the shares of instances shorter, equal and longer (seed average).
  * Pooled (p89 pooled: d [seeds, instances, combinations], a drawn base instance brings
    all its combinations): All (eight), Train (N, L, S, W, LS, LW), Held (SW, LSW), Setup
    (S, LS, SW, LSW), NoSetup (N, L, W, LW).
  * Counts: combinations whose 95% interval lies below zero, and above zero.
  * Per-seed pooled difference: for each seed the mean of d over the 100 instances and the
    eight combinations (and over each family's combinations).
  * Gap to the references: g = 100 (m - ref) / ref per seed and instance, m the makespan of
    either policy and ref the objective of the counted reference record; per combination
    and pooled over the eight, mean, hierarchical 95% interval (p89 ref_gap, fresh
    default_rng(11)), direction, and the shares of instances equal to, longer and shorter
    than the reference (seed average).
  * Validation at the kept checkpoints (the 20 SD2 10x5 validation records, each run under
    its own mask): per training combination the paired difference 100 (a - p) / p of the
    greedy validation makespans per seed and its mean; the validation score (mean over the
    six combinations of the makespan over the MWKR norm, lower is better) of each run and
    the paired difference of the scores; the kept update and the gain in the last quarter
    of training of each run.

OUTPUTS (under --out, default the repository root; every path in them is relative)
  results/public_active_arm.json   every number, unrounded, the checks and the files read
  results/public_active_arm.md
  manuscript/macros_pubact.tex      macro prefix PubAct, one comment per macro line;
                                   checked for collisions against every macros*.tex of
                                   manuscript/ and paper/ (and their subfolders) of the
                                   input root and the repository, own output skipped

SEEDS (--seeds)
  auto (default)  seeds 301 to 305 (arm_registry.PUB_ACTFAM_SEEDS: the policy runs
                  pubx-act-s304 and -s305 of the family under the active mask, trained by
                  scripts/run_pubfamily_queue.sh ARMS=actfam and evaluated on the headline set
                  by group pubactfam) once any file of seed 304 or 305 exists in the places
                  listed under INPUTS, seeds 301 to 303 otherwise; a five-seed reading needs
                  every file of the five runs of each arm, as the three-seed one does;
  301-303         the three seeds of the arm (arm_registry.PUB_ACT_SEEDS) whatever else exists:
                  the files of seeds 304 and 305 are then neither read nor unexpected, so the
                  stored three-seed result is reproduced byte for byte;
  301-305         the five seeds.
  The seed-level t interval has k - 1 degrees of freedom for k seeds; every other definition
  is the same for three and five seeds.

USAGE (repository root; CPU only)
  python scripts/p108_public_active_arm.py plan  [--root <root>] [--seeds auto|301-303|301-305]
  python scripts/p108_public_active_arm.py infer [--boot 20000] [--out <root>] [--root <root>]
         [--seeds auto|301-303|301-305]
"""
import argparse
import contextlib
import fnmatch
import glob
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
import p89_public_family as P89  # noqa: E402  (readers, estimators, macro lines)

REPO = P89.REPO
GENERATOR = 'scripts/p108_public_active_arm.py'
R8 = list(REG.COMBOS)
TRAIN = list(REG.TRAIN_COMBOS)
HELD = list(REG.HELD_OUT)
SETUP = [r for r in R8 if 'S' in r]                 # S, LS, SW, LSW
NO_SETUP = [r for r in R8 if 'S' not in r]          # N, L, W, LW
FAMILIES = {'All': R8, 'Train': TRAIN, 'Held': HELD, 'Setup': SETUP, 'NoSetup': NO_SETUP}
FAMILY_TEXT = {'All': 'all eight combinations', 'Train': 'the six training combinations',
               'Held': 'the two held-out combinations SW and LSW',
               'Setup': 'the four combinations with setups S, LS, SW and LSW',
               'NoSetup': 'the four combinations without setups N, L, W and LW'}
HEADLINE = P89.EQ_HEADLINE                          # pub_SD2_test_10x5+mix
assert HEADLINE == P89.REF_SET
SEEDS = tuple(REG.PUB_ACT_SEEDS)                   # the reading in use (choose_seeds)
THREE = tuple(REG.PUB_ACT_SEEDS)                   # 301-303, the seeds of the arm
FIVE = tuple(REG.PUB_ACTFAM_SEEDS)                 # 301-305, with the family's policy runs
SEED_CHOICES = {'301-303': THREE, '301-305': FIVE}
RUNS_WORD = {6: 'six', 10: 'ten'}                  # runs read (two arms) in the Markdown text
RULE = REG.PUB_ACT_RULE
STEM = REG.PUB_ACT_STEM
COMPARATOR = REG.PUB_STEM['released']
RECIPE_UPDATES = 2000
XS = P89.XS
PREFIX = 'PubAct'
ARMS = ('active', 'policy')
ARM_TEXT = {'active': f'policy under the active mask ({STEM})', 'policy': f'policy ({COMPARATOR})'}
ACT = 'the policy under the active mask'           # the arm in the texts of the outputs
OUT_FILES = ('results/public_active_arm.json', 'results/public_active_arm.md',
             'manuscript/macros_pubact.tex')
# the configuration keys that may differ between the arm and its comparator, and the keys
# the trainer added after the comparator was written, at their inert defaults (the mask is
# the deliberate difference, so it is not inert here)
CONFIG_FREE = ('mask_mode', 'model_suffix')
INERT_NEW = {k: v for k, v in P89.EQ_INERT_NEW.items() if k != 'mask_mode'}
BANNER = f'[train_family] OPTIONS: mask_mode={RULE} pair_extra=none'
MASK_LINE = 'mask: first_mask=exact mask_ties=exact'
EPS = P89.REF_EPS

refuse = P89.refuse
rel = P89.rel


# ================================================================== files
def run_name(arm, seed):
    """pubx-act-s<seed> (seeds 301-303 the arm, 304-305 the policy of the family under the
    active mask, arm_registry.pub_rule_model) or pubx-v1-s<seed>."""
    if arm == 'active':
        return REG.pub_act_model(seed) if int(seed) in THREE else REG.pub_rule_model('released', None, seed)
    return REG.pub_model('released', None, seed)


def run_paths(arm, seed):
    """(checkpoint, training log, validation record, configuration) of one run."""
    if arm == 'active':
        return REG.pub_act_paths(seed) if int(seed) in THREE else REG.pub_rule_paths('released', None, seed)
    m = REG.PUB_PREFIX + run_name(arm, seed)
    ck, lp = REG.pub_paths('released', None, seed)
    return ck, lp, REG.pub_vali_log('released', None, seed), f'train_log/FAMILY/config_{m}.json'


def folder():
    return os.path.join(REG.PUB_RESULTS, HEADLINE + XS)


def array_path(arm, seed, r):
    return os.path.join(folder(), f'Result_greedy+{REG.PUB_PREFIX}{run_name(arm, seed)}_{r}.npy')


def summary_name(arm, seed):
    return f'summary_{REG.PUB_PREFIX}{run_name(arm, seed)}{XS}.md'


def expected(arm, seeds=None):
    """Every file of one arm's runs (the seeds in use, or seeds), relative to the root."""
    out = []
    for s in (SEEDS if seeds is None else seeds):
        out += list(run_paths(arm, s))
        out += [array_path(arm, s, r) for r in R8]
        out.append(os.path.join(folder(), summary_name(arm, s)))
    return out


def second_seed_files():
    """The files of the policy runs pubx-act-s306 to -s310 of the second five seeds
    (scripts/pub_seed_extension.py): they carry the name of the arm, belong to the ten-seed
    comparison of p115 only, and are neither read nor unexpected here."""
    import pub_seed_extension as PSE
    out = set()
    for s in PSE.SECOND_SEEDS:
        n = PSE.name('released', None, s)
        out |= set(PSE.training_paths('released', None, s))
        out |= {os.path.join(folder(), f'Result_greedy+{REG.PUB_PREFIX}{n}_{r}.npy') for r in R8}
        out.add(os.path.join(folder(), f'summary_{REG.PUB_PREFIX}{n}{XS}.md'))
    return {os.path.normpath(x) for x in out}


def listing(root, pat):
    """The files that glob.glob(os.path.join(root, pat)) lists (pat relative to root, its
    wildcards in the file name only), relative to root, without the files of the second five
    seeds; read from the folder's entries, so no file of a second seed is opened or stat'ed."""
    d, base = os.path.split(pat)
    skip = second_seed_files()
    try:
        names = sorted(e.name for e in os.scandir(os.path.join(root, d)))
    except (FileNotFoundError, NotADirectoryError):
        return set()
    return {os.path.join(d, x) for x in names if fnmatch.fnmatchcase(x, base)
            and (not x.startswith('.') or base.startswith('.'))
            and os.path.normpath(os.path.join(d, x)) not in skip}


def arm_state(root):
    """State of the files of the arm under the active mask: 'none' when no file of its stem exists in the
    training folders or the headline folder, 'complete' when every expected file exists,
    every run has finished and no other file of the stem exists there, 'partial'
    otherwise; with the missing, unexpected and unfinished items. The files of the second five
    seeds (second_seed_files) are left out."""
    want = expected('active')
    m = REG.PUB_PREFIX + STEM
    pats = [os.path.join('trained_network', 'FAMILY', f'{m}-*.pth'),
            os.path.join('train_log', f'PUBFAMILY_{STEM}-*.log'),
            os.path.join('train_log', 'FAMILY', f'valiquality_{m}-*.txt'),
            os.path.join('train_log', 'FAMILY', f'config_{m}-*.json'),
            os.path.join(folder(), f'Result_greedy+{m}-*'),
            os.path.join(folder(), f'summary_{m}-*')]
    found = {x for x in want if os.path.exists(os.path.join(root, x))}
    for pat in pats:
        found |= listing(root, pat)
    # the files of the seeds of the other reading (304 and 305 when three are read) are its
    # own, neither read nor unexpected here
    other = set(expected('active', [s for s in FIVE if s not in SEEDS]))
    found -= other
    missing = [x for x in want if x not in found]
    extra = sorted(found - set(want))
    unfinished = []
    for s in SEEDS:
        lp = os.path.join(root, run_paths('active', s)[1])
        if os.path.exists(lp) and 'done in' not in open(lp, errors='replace').read():
            unfinished.append(run_name('active', s))
    if not found:
        state = 'none'
    elif missing or extra or unfinished:
        state = 'partial'
    else:
        state = 'complete'
    return dict(state=state, expected=len(want), found=len(found), missing=missing,
                unexpected=extra, unfinished=unfinished)


def comparator_missing(root):
    return [x for x in expected('policy') if not os.path.exists(os.path.join(root, x))]


def choose_seeds(root, mode='auto'):
    """The seeds of the reading (module docstring, SEEDS): sets the module's SEEDS and
    returns them."""
    global SEEDS
    if mode == 'auto':
        later = expected('active', [s for s in FIVE if s not in THREE])
        SEEDS = FIVE if any(os.path.exists(os.path.join(root, x)) for x in later) else THREE
    elif mode in SEED_CHOICES:
        SEEDS = SEED_CHOICES[mode]
    else:
        raise SystemExit(f'REFUSING: --seeds {mode!r}, want auto, 301-303 or 301-305')
    return SEEDS


def readiness(root, mode='auto'):
    """What infer needs, without reading an array: the seeds read, the arm's state, the
    comparator's missing files and the count of counted references (p89's reader)."""
    choose_seeds(root, mode)
    st = arm_state(root)
    cm = comparator_missing(root)
    I = P89.Inputs(root)
    I.check_records(HEADLINE)
    R = I.load_refs(HEADLINE)
    rs = P89.ref_state(R)
    return dict(arm=st, seeds=list(SEEDS), comparator_missing=cm, references=dict(
        folder=R['folder'], valid=rs['total']['valid'], expected=rs['expected'],
        complete=rs['complete_all'], reason=rs['reason']),
        ready=st['state'] == 'complete' and not cm and rs['complete_all'])


# ================================================================== loading and checks
def config_diff(ref, new):
    """Keys whose values differ, a key absent from ref and at its inert default in new
    counting as equal."""
    return sorted(k for k in set(ref) | set(new)
                  if ref.get(k, '<absent>') != new.get(k, '<absent>') and not
                  (k in INERT_NEW and k not in ref and new.get(k) in INERT_NEW[k]))


def load(root):
    """Every input, checked; returns (Inputs of p89, dict of the arrays and records, the
    reference records)."""
    st = arm_state(root)
    if st['state'] != 'complete':
        what = (f'no file of {STEM} exists' if st['state'] == 'none' else
                f'{st["found"]} of {st["expected"]} files of {STEM} exist')
        refuse(f'the arm {STEM} under the active mask is incomplete: {what}; missing '
               f'{st["missing"][:4]}{" ..." if len(st["missing"]) > 4 else ""}; unexpected '
               f'{st["unexpected"][:4]}; not finished {st["unfinished"]}. The analysis needs '
               f'every file of the {len(SEEDS)} runs of seeds {SEEDS[0]}-{SEEDS[-1]} (python '
               f'{GENERATOR} plan lists them; --seeds 301-303 reads the three seeds of the arm)')
    cm = comparator_missing(root)
    if cm:
        refuse(f'the comparator {COMPARATOR} is incomplete: missing {cm[:4]}'
               f'{" ..." if len(cm) > 4 else ""} ({len(cm)} files)')
    I = P89.Inputs(root)
    I.check_records(HEADLINE)
    names = I.names[HEADLINE]
    n = len(names)
    X = dict(ms={}, evidence={}, train={}, config={}, norms={}, files=[], config_differs_in={})
    for arm in ARMS:
        ev = {}
        oldest = np.inf
        for s in SEEDS:
            name = run_name(arm, s)
            ck = os.path.join(root, run_paths(arm, s)[0])
            t_ck = os.stat(ck).st_mtime
            for r in R8:
                p = array_path(arm, s, r)
                v, t = I._load(os.path.join(root, p), n)
                if t < t_ck:
                    refuse(f'{p} is older than the checkpoint of {name}')
                X['ms'][(arm, s, r)] = v
                X['files'].append(p)
                oldest = min(oldest, t)
            sp = os.path.join(I.folder(HEADLINE), summary_name(arm, s))
            I.check_summary(HEADLINE, summary_name(arm, s),
                            {r: f'greedy+{REG.PUB_PREFIX}{name}' for r in R8},
                            lambda r, m, a=arm, s=s: X['ms'][(a, s, r)], evidence=ev)
            if MASK_LINE not in open(sp, errors='replace').read():
                refuse(f'{os.path.relpath(sp, root)} does not record the exact mask options')
        if not I.evidence[HEADLINE]['records_newest_mtime'] <= oldest:
            refuse(f'{HEADLINE}: a descriptor record is newer than a result file of {ARM_TEXT[arm]}')
        I.check_lower_bounds(HEADLINE, vecs=[X['ms'][(arm, s, 'N')] for s in SEEDS], evidence=ev)
        ev['arrays_not_older_than_checkpoint'] = True
        X['evidence'][arm] = ev
    for s in SEEDS:
        for arm in ARMS:
            ck, lp, vp, cp = run_paths(arm, s)
            rec = I.training_record(None, None, s, paths=(lp, vp))
            txt = open(os.path.join(root, lp), errors='replace').read()
            if arm == 'active' and BANNER not in txt:
                refuse(f'{lp} does not state the {RULE} mask ({BANNER!r})')
            if arm == 'policy' and 'OPTIONS' in txt:
                refuse(f'{lp} states a mask other than the non-delay mask')
            _, _, norms = REG.validation_record(None, paths=(os.path.join(root, vp),
                                                             os.path.join(root, lp)))
            X['train'][(arm, s)] = rec
            X['norms'][(arm, s)] = norms
            with open(os.path.join(root, cp)) as f:
                X['config'][(arm, s)] = json.load(f)
        a, p = X['train'][('active', s)], X['train'][('policy', s)]
        if a['last_update'] != p['last_update']:
            refuse(f'{run_name("active", s)} validated up to update {a["last_update"]}, '
                   f'{run_name("policy", s)} up to {p["last_update"]}')
        if not X['norms'][('active', s)] or X['norms'][('active', s)] != X['norms'][('policy', s)]:
            refuse(f'seed {s}: MWKR validation norms {X["norms"][("active", s)]} of the arm and '
                   f'{X["norms"][("policy", s)]} of the comparator differ; scores are not on one scale')
        ca, cp_ = X['config'][('active', s)], X['config'][('policy', s)]
        for arm, c in (('active', ca), ('policy', cp_)):
            for k, v in (('family_regimes', ','.join(TRAIN)), ('max_updates', RECIPE_UPDATES),
                         ('seed_train', s), ('first_mask', 'exact'), ('mask_ties', 'exact')):
                if c.get(k) != v:
                    refuse(f'{run_paths(arm, s)[3]} records {k}={c.get(k)!r}, want {v!r}')
        if ca.get('mask_mode') != RULE or cp_.get('mask_mode', 'nondelay') != 'nondelay':
            refuse(f'seed {s}: mask_mode {ca.get("mask_mode")!r} in {run_paths("active", s)[3]} and '
                   f'{cp_.get("mask_mode")!r} in {run_paths("policy", s)[3]}; want {RULE!r} and nondelay')
        diff = config_diff(cp_, ca)
        if set(diff) - set(CONFIG_FREE):
            refuse(f'{run_paths("active", s)[3]} is not the recipe of {run_paths("policy", s)[3]} '
                   f'under the {RULE} mask: differs in '
                   + str({k: (cp_.get(k), ca.get(k)) for k in diff}))
        X['config_differs_in'][f's{s}'] = diff
    R = I.load_refs(HEADLINE)
    rs = P89.ref_state(R)
    if not rs['complete_all']:
        refuse(f'references: {rs["reason"]}')
    probs = P89.ref_problems(I, R, arrays=False)
    for arm in ARMS:
        for s in SEEDS:
            for r in R8:
                lb, v = R['regimes'][r]['bound_mono'], X['ms'][(arm, s, r)]
                known = np.isfinite(lb)
                for i in np.flatnonzero(known & (v < lb - EPS))[:3]:
                    probs.append(f'{names[i]} {r}: makespan {v[i]:g} of {run_name(arm, s)} below the '
                                 f'reference bound {lb[i]:g}')
    if probs:
        refuse(f'the references and the evaluations of {HEADLINE} contradict each other '
               f'({len(probs)}): ' + '; '.join(probs[:5]))
    X['ref_bound_checked'] = int(sum(np.isfinite(R['regimes'][r]['bound_mono']).sum() for r in R8))
    return I, X, R


# ================================================================== analysis
def analyse(I, X, R, B):
    ms = X['ms']

    def d_of(r):
        return np.stack([rel(ms[('active', s, r)], ms[('policy', s, r)]) for s in SEEDS])

    D = {r: d_of(r) for r in R8}
    comps = {}
    for r in R8:
        c = P89.component(D[r], B)
        c.update(P89.seed_shares(D[r]))
        comps[r] = c
    pooled = {}
    for fam, regs in FAMILIES.items():
        x = P89.pooled([D[r] for r in regs], B)
        x.update(combinations=list(regs),
                 **P89.seed_shares(np.stack([D[r] for r in regs], axis=2).reshape(len(SEEDS), -1)))
        pooled[fam] = x
    below = [r for r in R8 if comps[r]['hi'] < 0]
    above = [r for r in R8 if comps[r]['lo'] > 0]
    per_seed = {fam: {f's{s}': float(m) for s, m in zip(SEEDS, pooled[fam]['seed_means'])}
                for fam in FAMILIES}
    # gap to the references
    refs = {}
    for arm in ARMS:
        G = {r: np.stack([rel(ms[(arm, s, r)], R['regimes'][r]['objective']) for s in SEEDS])
             for r in R8}
        g3 = np.stack([G[r] for r in R8], axis=2)
        refs[arm] = dict(components={r: dict(P89.ref_gap(G[r], B), **P89.seed_shares(G[r]))
                                     for r in R8},
                         pooled=dict(P89.ref_gap(g3, B), c=len(R8),
                                     **P89.seed_shares(g3.reshape(len(SEEDS), -1))))
    rs = P89.ref_state(R)
    ref_out = dict(set=R['set'], folder=R['folder'], campaign=R['campaign'],
                   counted=rs['total']['valid'], proved=rs['total']['proved'],
                   expected=rs['expected'], not_counted=len(R['not_counted']), arms=refs,
                   reference='the schedule of every record of scripts/p91_public_cpsat_refs.py '
                             'that p91 marks valid (the records section 9 of p89 reads)')
    # validation at the kept checkpoints
    T = X['train']
    vali = {}
    for r in TRAIN:
        a = [T[('active', s)]['kept'][r] for s in SEEDS]
        p = [T[('policy', s)]['kept'][r] for s in SEEDS]
        g = [100.0 * (x - y) / y for x, y in zip(a, p)]
        vali[r] = dict(active_kept=a, policy_kept=p, active_minus_policy=g, mean=float(np.mean(g)))
    sa = [T[('active', s)]['score'] for s in SEEDS]
    sp = [T[('policy', s)]['score'] for s in SEEDS]
    gs = [100.0 * (x - y) / y for x, y in zip(sa, sp)]
    vali['score'] = dict(active=sa, policy=sp, active_mean=float(np.mean(sa)),
                         policy_mean=float(np.mean(sp)), active_minus_policy=gs, mean=float(np.mean(gs)))
    for key in ('kept_update', 'kept_fraction', 'last_update', 'last_quarter_gain'):
        vali[key] = {arm: [T[(arm, s)][key] for s in SEEDS] for arm in ARMS}
    vali['norms'] = {f's{s}': X['norms'][('active', s)] for s in SEEDS}
    return dict(set=HEADLINE, n=len(I.names[HEADLINE]), arm=STEM, mask=RULE, comparator=COMPARATOR,
                comparator_mask='nondelay', seeds=list(SEEDS),
                difference='d = 100 (active - policy) / policy per training seed and base instance, '
                           f'seed s against seed s; negative = {ACT} shorter',
                families={f: list(v) for f, v in FAMILIES.items()},
                components=comps, pooled=pooled, below=below, above=above,
                below_n=len(below), above_n=len(above), per_seed=per_seed,
                references=ref_out, validation=vali,
                config_differs_in=X['config_differs_in'],
                checks=dict(records=I.evidence[HEADLINE], arms=X['evidence'],
                            reference_bounds_checked=X['ref_bound_checked'],
                            norms_equal=True, banner=BANNER))


# ================================================================== macros
f2 = P89.f2
u1 = P89.u1


def macros(res, B):
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Public family, {res["set"]}: the policy trained under the {RULE} mask ({STEM}) '
         f'against the policy ({COMPARATOR}, the non-delay mask), training seeds '
         f'{SEEDS[0]}-{SEEDS[-1]}, seed s against seed s; B = {B}; hierarchical intervals over '
         f'seeds and base instances, fresh default_rng({P89.HIER_SEED}) per quantity.',
         '% d = 100 (active - policy) / policy per seed and base instance, in %; negative = '
         f'{ACT} shorter. Gaps to the references: makespan minus reference over reference.',
         f'\\newcommand{{\\{PREFIX}InstN}}{{{res["n"]}}}  % base instances of {res["set"]}',
         f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(SEEDS)}}}  % training seeds per arm',
         f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}  % bootstrap replicates']
    for r in R8:
        P89.comp_lines(L, f'{PREFIX}Diff{r}', res['components'][r],
                       f'{res["set"]}, {r}: {ACT} vs the policy, d in %')
    for fam in FAMILIES:
        x = res['pooled'][fam]
        P89.pool_lines(L, f'{PREFIX}DiffPool{fam}', x,
                       f'{res["set"]}: {ACT} vs the policy on {FAMILY_TEXT[fam]}, d in %')
        P89.dir_line(L, f'{PREFIX}DiffPool{fam}', x,
                     f'{res["set"]}, {ACT} vs the policy, pooled over {FAMILY_TEXT[fam]}')
    L.append(f'\\newcommand{{\\{PREFIX}BelowN}}{{{res["below_n"]}}}'
             f'\\newcommand{{\\{PREFIX}AboveN}}{{{res["above_n"]}}}'
             f'  % combinations (of 8) whose 95% interval of d lies below zero ({ACT} '
             f'shorter: {",".join(res["below"]) or "none"}) and above zero '
             f'({",".join(res["above"]) or "none"})')
    for s in SEEDS:
        L.append(f'\\newcommand{{\\{PREFIX}Seed{REG.SEED_WORDS[s]}}}'
                 f'{{{f2(res["per_seed"]["All"][f"s{s}"])}}}'
                 f'  % seed {s}: mean d over the {res["n"]} instances and the eight combinations')
    for arm, w in (('active', 'Act'), ('policy', 'Pol')):
        G = res['references']['arms'][arm]
        for r in R8:
            P89.ref_line(L, f'{PREFIX}Ref{w}{r}', G['components'][r],
                         f'{res["set"]}, {r}: {ARM_TEXT[arm]}, seeds {SEEDS[0]}-{SEEDS[-1]}')
        P89.ref_line(L, f'{PREFIX}Ref{w}Pool', G['pooled'],
                     f'{res["set"]}: {ARM_TEXT[arm]}, pooled over the eight combinations')
    V = res['validation']
    for r in TRAIN:
        L.append(f'\\newcommand{{\\{PREFIX}Vali{r}}}{{{f2(V[r]["mean"])}}}'
                 f'  % validation at the kept checkpoints, {r}: {ACT} minus the policy, % of '
                 f'the policy\'s greedy validation makespan, mean of {len(SEEDS)} seeds')
    L.append(f'\\newcommand{{\\{PREFIX}ValiScore}}{{{f2(V["score"]["mean"])}}}'
             f'  % validation score of the kept checkpoint: {ACT} minus the policy, % of the '
             f'policy\'s score, mean of {len(SEEDS)} seeds')
    L.append(f'\\newcommand{{\\{PREFIX}ValiScoreAct}}{{{V["score"]["active_mean"]:.4f}}}'
             f'\\newcommand{{\\{PREFIX}ValiScorePol}}{{{V["score"]["policy_mean"]:.4f}}}'
             f'  % validation score of the kept checkpoint (mean over the six training combinations '
             f'of makespan over MWKR norm), {ACT} and the policy, mean of {len(SEEDS)} seeds')
    return L


def macro_names(L):
    return re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(L))


def check_macro_names(names, roots, out_mac):
    """Every name is the prefix plus letters and unique, and no macros*.tex of manuscript/ or
    paper/ (or a subfolder) of the given roots defines one of them; the output file itself
    and files of the output's name in those folders are skipped."""
    bad = [x for x in names if not re.fullmatch(PREFIX + r'[A-Za-z]+', x)]
    if bad:
        refuse(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        refuse(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    own = os.path.basename(OUT_FILES[2])
    files = {}                        # absolute path -> path relative to its root
    for root in roots:
        for d in ('manuscript', 'paper'):
            for pat in (os.path.join(root, d, 'macros*.tex'), os.path.join(root, d, '*', 'macros*.tex')):
                for f in glob.glob(pat):
                    files.setdefault(os.path.abspath(f), os.path.relpath(f, root))
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    for f, shown in sorted(files.items()):
        if f == os.path.abspath(out_mac) or os.path.basename(f) == own:
            continue
        with open(f, errors='replace') as fh:
            clash = set(names) & set(rx.findall(fh.read()))
        if clash:
            refuse(f'macro name collision with {shown}: {sorted(clash)[:5]}')
    return len(files)


# ================================================================== markdown
def ci(x, lo='lo', hi='hi'):
    return f'[{f2(x[lo])}, {f2(x[hi])}]'


def markdown(res, B):
    E = res['checks']
    md = [f'# The policy under the active mask against the policy, public family ({GENERATOR})', '',
          f'Headline set {res["set"]} ({res["n"]} base instances, eight combinations). Arm {STEM}: the '
          f'released recipe trained and evaluated under the {RULE} mask (the condition of Giffler '
          f'and Thompson for active schedules); comparator {COMPARATOR} (the non-delay mask, on '
          f'processing starts); training seeds {", ".join(map(str, SEEDS))}, seed s against seed s. '
          f'{res["difference"]}. Hierarchical 95% intervals over seeds and base instances (B = {B}, '
          f'fresh default_rng({P89.HIER_SEED}) per quantity), seed-level t intervals '
          f'({len(SEEDS) - 1} df), margin verdicts at 0.5% and 1% on the hierarchical interval; '
          f'pooled rows keep the combinations of a base instance together.', '',
          '## Checks', '']
    ev = E['records']
    md.append(f'- Records: {ev["n"]} in one sorted order in all eight combinations; best-known '
              f'table complete.')
    for arm in ARMS:
        e = E['arms'][arm]
        md.append(f'- {ARM_TEXT[arm]}: {e["summaries_checked"]} summaries with n/n feasible, matching '
                  f'means and regime-N gaps and the exact mask options; '
                  f'{e["n_arrays_bound_checked"]} regime-N arrays at or above the bound of '
                  f'{e["n_bound_checked"]} instances; every array not older than its checkpoint and '
                  f'newer than the records.')
    md.append('- Configurations against the comparator of the same seed differ only in: '
              + '; '.join(f'{k} {", ".join(v)}' for k, v in res['config_differs_in'].items())
              + f'. The arm\'s logs state the mask ({E["banner"]}); MWKR validation norms equal.')
    rf = res['references']
    md += [f'- References: {rf["counted"]} of {rf["expected"]} counted ({rf["proved"]} proved optimal, '
           f'{rf["not_counted"]} not counted; folder {rf["folder"]}); no makespan of the '
           f'{RUNS_WORD.get(2 * len(SEEDS), 2 * len(SEEDS))} runs '
           f'below a counted reference bound ({E["reference_bounds_checked"]} instance-combination '
           f'bounds).', '',
           '## 1. Paired difference per combination', '',
           '| comb. | mean | hier. 95% CI | seed-t CI | direction | 0.5% | 1% | per seed | '
           'shorter / equal / longer % |', '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        c = res['components'][r]
        md.append(f'| {r} | {f2(c["mean"])} | {ci(c)} | {ci(c, "t_lo", "t_hi")} | {c["direction"]} | '
                  f'{c["v0.5"]} | {c["v1.0"]} | ' + ', '.join(f2(m) for m in c['seed_means'])
                  + f' | {c["shorter"]:.1f} / {c["equal"]:.1f} / {c["longer"]:.1f} |')
    md += ['', f'Interval below zero ({ACT} shorter): {res["below_n"]} of 8 '
           f'({", ".join(res["below"]) or "none"}); above zero: {res["above_n"]} of 8 '
           f'({", ".join(res["above"]) or "none"}).', '',
           '## 2. Pooled', '',
           '| family | combinations | mean | hier. 95% CI | seed-t CI | direction | 0.5% | 1% | '
           'per seed | shorter / equal / longer % |', '|---|---|---|---|---|---|---|---|---|---|']
    for fam in FAMILIES:
        x = res['pooled'][fam]
        md.append(f'| {fam} | {",".join(x["combinations"])} | {f2(x["mean"])} | {ci(x)} | '
                  f'{ci(x, "t_lo", "t_hi")} | {x["direction"]} | {x["v0.5"]} | {x["v1.0"]} | '
                  + ', '.join(f2(m) for m in x['seed_means'])
                  + f' | {x["shorter"]:.1f} / {x["equal"]:.1f} / {x["longer"]:.1f} |')
    md += ['', '## 3. Gap to the references', '',
           f'g = 100 (makespan - reference) / reference per seed and instance; references: '
           f'{rf["reference"]}; interval and direction only.', '',
           '| comb. | policy under the active mask | = / > / < % | policy | = / > / < % |',
           '|---|---|---|---|---|']

    def cell(x):
        return f'{f2(x["mean"])} {ci(x)} {x["direction"]}'

    def sh(x):
        return f'{x["equal"]:.1f} / {x["longer"]:.1f} / {x["shorter"]:.1f}'
    A, P = rf['arms']['active'], rf['arms']['policy']
    for r in R8:
        a, p = A['components'][r], P['components'][r]
        md.append(f'| {r} | {cell(a)} | {sh(a)} | {cell(p)} | {sh(p)} |')
    md.append(f'| pooled | {cell(A["pooled"])} | {sh(A["pooled"])} | {cell(P["pooled"])} | '
              f'{sh(P["pooled"])} |')
    V = res['validation']
    md += ['', '## 4. Validation at the kept checkpoints (20 SD2 10x5 validation records, each run '
           'under its own mask)', '',
           '| comb. | policy under the active mask minus the policy, % per seed | mean |', '|---|---|---|']
    for r in TRAIN:
        md.append(f'| {r} | ' + ', '.join(f2(v) for v in V[r]['active_minus_policy'])
                  + f' | {f2(V[r]["mean"])} |')
    sc = V['score']
    md.append(f'| score | ' + ', '.join(f2(v) for v in sc['active_minus_policy'])
              + f' | {f2(sc["mean"])} |')
    md += ['', f'Score of the kept checkpoint per seed: {ACT} '
           + ', '.join(f'{v:.4f}' for v in sc['active']) + '; the policy '
           + ', '.join(f'{v:.4f}' for v in sc['policy']) + f'. Kept update per seed: {ACT} '
           + ', '.join(map(str, V['kept_update']['active'])) + '; the policy '
           + ', '.join(map(str, V['kept_update']['policy'])) + f' (of {V["last_update"]["policy"][0]}); '
           f'gain in the last quarter %: {ACT} '
           + ', '.join(f'{v:.2f}' for v in V['last_quarter_gain']['active']) + '; the policy '
           + ', '.join(f'{v:.2f}' for v in V['last_quarter_gain']['policy']) + '.']
    return md


# ================================================================== phases
def plan(a):
    root = os.path.abspath(a.root)
    x = readiness(root, a.seeds)
    st = x['arm']
    print(f'[p108] {HEADLINE}: arm {STEM} (the {RULE} mask), comparator {COMPARATOR} (the non-delay '
          f'mask), seeds '
          f'{", ".join(map(str, SEEDS))}')
    print(f'arm: {st["state"]}, {st["found"]} of {st["expected"]} files')
    for f in st['missing']:
        print(f'  missing {f}')
    for f in st['unexpected']:
        print(f'  unexpected {f}')
    for f in st['unfinished']:
        print(f'  not finished {f}')
    n_cmp = len(expected('policy'))
    print(f'comparator: {n_cmp - len(x["comparator_missing"])} of {n_cmp} files')
    for f in x['comparator_missing']:
        print(f'  missing {f}')
    rf = x['references']
    print(f'references: {rf["folder"]}: {rf["valid"]} of {rf["expected"]} counted'
          + ('' if rf['complete'] else f' ({rf["reason"]})'))
    print('infer: ' + ('ready' if x['ready'] else 'not ready'))
    return x


def infer(a):
    root = os.path.abspath(a.root)
    choose_seeds(root, a.seeds)
    out_root = os.path.abspath(a.out)
    outs = [os.path.join(out_root, p) for p in OUT_FILES]
    out_json, out_md, out_mac = outs
    for p in outs:                    # a refusal anywhere below leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    try:
        I, X, R = load(root)
        res = analyse(I, X, R, a.boot)
        L = macros(res, a.boot)
        names = macro_names(L)
        n_mac_files = check_macro_names(names, sorted({root, REPO}), out_mac)
        res.update(generator=GENERATOR, boot=a.boot, alpha=0.05, margins=list(P89.MARGINS),
                   streams=dict(hierarchical=f'fresh default_rng({P89.HIER_SEED}) per quantity'),
                   files_read=len(X['files']), arrays=X['files'],
                   rows='row i = i-th record of desc/<R>/ in sorted file-name order',
                   macro_files_checked=n_mac_files, n_macros=len(names))
        md = markdown(res, a.boot)
        with contextlib.redirect_stdout(io.StringIO()):
            emit(out_mac, L)
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, 'w') as f:
            json.dump(P89.jsonable(res), f, indent=1)
        with open(out_md, 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException as e:
        for p in outs:
            if os.path.exists(p):
                os.remove(p)
                print(f'removed {os.path.relpath(p, out_root)} (refusing to certify)')
        if isinstance(e, REG.MissingResult):
            raise P89.Refusal(f'*** REFUSING TO CERTIFY ***\n{e}')
        raise
    print('\n'.join(md))
    print(f'\n{len(names)} macros in {os.path.relpath(out_mac, out_root)}; {len(X["files"])} result '
          f'arrays read; wrote {os.path.relpath(out_json, out_root)}, '
          f'{os.path.relpath(out_md, out_root)}')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('phase', choices=['plan', 'infer'])
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    ap.add_argument('--out', default=REPO, help='root of the outputs (default: the repository)')
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--seeds', default='auto', help='auto (default), 301-303 or 301-305 (see SEEDS)')
    a = ap.parse_args(argv)
    global SEEDS
    keep = SEEDS
    try:
        return {'plan': plan, 'infer': infer}[a.phase](a)
    finally:
        SEEDS = keep


if __name__ == '__main__':
    main()
