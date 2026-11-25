r"""P93: the comparison with dispatching rules in three forms, for one benchmark family.

FORMS (per combination R; d = 100 (a - b) / b per training seed and test instance,
a the policy quantity, b the rule quantity; positive d = the policy is longer)
  (a) GreedyBestFour     one greedy schedule against the per-instance best of the
                         four masked rules (FIFO, MOR, SPT, MWKR; the stored arrays,
                         tie-break seed 50)
  (b) SampleFour         the best of 4 sampled schedules against the per-instance best
                         of the same four rule schedules
      SampleSixteen      the best of 16 sampled schedules against the per-instance
                         best of 16 rule schedules: the four rules, each rolled out
                         with the tie-break seeds 50, 51, 52 and 53
      GreedyPlusThree    the greedy schedule plus 3 sampled ones (the shorter of the
                         greedy schedule and the best of 3) against the best of four
      GreedyPlusFifteen  the greedy schedule plus 15 sampled ones against the best of 16
  (c) GreedyValiRule     one greedy schedule against the single rule with the lowest
                         mean makespan on the validation instances of R (a tie goes to
                         the first of FIFO, MOR, SPT, MWKR): the instances the training
                         runs validated on for checkpoint selection, min(vali_size, 20)
                         of them (public: the first records of the validation folder in
                         sorted order; factory: the trainer's in-memory instances). The
                         public JSON also gives the selection on all validation records
                         (field all_records), which no macro uses.
  each for the released policy (Pol, eight combinations), the single-rule control (Ctl,
  eight) and the specialist of R (Spec). The best of k sampled schedules is the
  unbiased order-statistic estimate of the expected minimum of k draws from the K
  stored draws (scripts/p90_public_rule_comparison.py best_of_k); K is --K, by default
  the setting of scripts/rulecmp_plan.py (RULECMP_K, else 32), at least 16, and must
  equal the K the arrays hold.

RULE SPREAD per R: the per-instance best of four against the single rule with the
  lowest test mean and against the validation rule (mean d); the mean per-instance
  range (max - min) / min of the four; the gap between the largest and the smallest
  rule mean; the best of 16 rule schedules against the best of four; the mean number of
  distinct makespans among the 16.
TRACES per R (the definitions of the study of 2026-11-11), for each seed of the
  released policy (greedy rollout) and for each rule:
  * share of single-pair decisions: the fraction of all dispatching decisions, pooled
    over the test instances, at which the non-delay mask admits exactly one
    (job, machine) pair;
  * setup share of machine capacity: per instance, the total setup time of the
    realised schedule divided by (number of machines that process at least one
    operation) x makespan, averaged over the instances.

STATISTICS (manuscript/supp_new/sn07_statistics.tex, scripts/stats_family.py)
  Every interval is the crossed hierarchical bootstrap over training seeds and test
  instances (stats_family.hier_boot), B = --boot, a fresh numpy default_rng(11) per
  quantity, 95% percentile interval, with no widening for the number of combinations.
  "Shorter than the rule quantity": the one-sided Student t test on the per-seed mean
  differences (k - 1 degrees of freedom), a one-sided bootstrap p beside it, and Holm's
  correction over the eight combinations of each arm and form: the stats_family calls
  of p89_public_family.component and its Holm step, which p89 applies to form (a) (the
  public JSON reports the agreement with the form (a) macros of macros_public.tex).
  "Longer than the rule quantity": the same two tests in the other direction
  (alternative 'greater'), with their own Holm step over the eight combinations.
  VERDICT per arm, form and combination, in plain words: "ahead" when the 95% interval
  lies below zero and both Holm-adjusted p-values for "shorter" are below 0.05;
  "behind" when the interval lies above zero and both Holm-adjusted p-values for
  "longer" are below 0.05; "not distinguishable" otherwise.
  Shares shorter, equal and longer: forms (a) and (c) from the sign of d (seed
  average); the forms with sampled schedules give the expected shares under the same
  order-statistic estimator (p90 p_shorter_equal).

INPUTS (names and folders from scripts/rulecmp_plan.py, which takes them from
scripts/arm_registry.py)
  stored greedy and rule arrays in the family's result folder; the sampled arrays,
  traces, tie-break-seed rule arrays and validation rule arrays in its rulecmp_*
  subfolders (or under --input-root, which stands in for the result folder).

REFUSALS (the outputs are removed and nothing is certified): an expected array is
  missing, has the wrong shape, holds a non-positive or non-finite makespan, lists other
  instances than the sorted records, is older than the test records or than the
  checkpoint of its run (or that checkpoint is missing); a sampled schedule is
  infeasible or a sampled file holds another K than --K (K below 16 is refused before
  any read); the training runs differ in a setting that defines the validation
  instances (rulecmp_plan.VALI_KEYS) or validated on another record folder; a trace
  does not reproduce the stored greedy or rule makespans; a rule
  folder's summary reports an infeasible schedule or another mean; the factory
  validation instances did not pass the MWKR norm check; a macro name collides with
  another macro file. --partial (reproduction against another array tree only) skips a
  quantity whose arrays are absent instead of refusing and never writes the macro file.

PUBLIC TEST SETS (--set, arm_registry.PUB_SETS; default the headline set
  pub_SD2_test_10x5+mix): each set is analysed on its own records and stored arrays,
  with the arms given by --arms (default: all three on the headline set, the released
  policy elsewhere; an arm left out has no macro at all). Form (c) of every set uses
  the rule selected on the headline set's validation instances, a property of the
  trained family, applied to the set's own stored rule arrays. The records must equal
  the set's dataset_meta.json list, count and exclusions (orb7 in the Hurink sets).

THE PUBLIC FAMILY UNDER THE ACTIVE MASK (--family public --mask active)
  The family trained under the active mask (pubx-act, pubx-ctlact, pubx-soloact-<R>; seeds
  301-305) against the four rules rolled out under the active mask, both sides under that
  mask (scripts/rulecmp_plan.py, THE PUBLIC FAMILY UNDER THE ACTIVE MASK): the same forms,
  estimators, streams, verdicts and macro definitions; the stored rule arrays are those of
  scripts/p109_public_active_rules.py (test_results/PUBFAMILY/rules_act/<set>_xmask) and
  the rulecmp_* subfolders lie in that folder. Added checks: every sampled and trace file
  records the active mask; every rule folder (tie-break seeds 50 to 53, validation records)
  records 'pairs: mask_mode=active' and a rules_meta.json with the active mask and its seed;
  every run's configuration names the active mask and its training log holds the completion
  line. A trace counts the pairs the active mask admits. Labels public_act and
  public_act_<tag>, macro prefix RcAct and RcAct<tag>: every macro of the default run under
  the name with Rcp replaced by RcAct. Form (a) is cross-checked against the macros of
  scripts/p89_public_family.py --family active (macros_pubfam_act.tex, PubFamAct<tag>Rule...).
  A partial campaign refuses; --partial writes a partial report (JSON and Markdown, each
  marked PARTIAL REPORT, no macro file) and needs --out outside the repository root and its
  paper, manuscript and results folders.

OUTPUTS (label: the family; public_<tag> for a public test set other than the headline)
  <out>/manuscript/macros_rulecmp_<label>.tex   (prefix Rcp public, Rcp<tag> for another
                                               public test set with the tags of p89:
                                               Sdtwenty, Brand, Edata, Rdata, Vdata;
                                               Rcf factory; RcAct, RcAct<tag> under the
                                               active mask)
  <out>/results/rulecmp_<label>.json           every number and the settings
  <out>/results/rulecmp_<label>.md             tables for reading

Usage (from the repository root):
  python scripts/p93_rule_comparison.py --family public [--K 32] [--boot 20000] [--out <root>]
  python scripts/p93_rule_comparison.py --family public --set pub_Brandimarte [--arms released]
  PACT_ENV=exact PACT_SEEDS=301-305 python scripts/p93_rule_comparison.py --family factory
  python scripts/p93_rule_comparison.py --family public --K 100 --input-root <folder> \
      --partial --out <folder>   (arrays of another tree; JSON and Markdown only)
  python scripts/p93_rule_comparison.py --family public --mask active [--set <set>]
  python scripts/p93_rule_comparison.py --family public --mask active --partial --out <folder>
"""
import argparse
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import rulecmp_plan as PLAN  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p90_public_rule_comparison import best_of_k, p_shorter_equal  # noqa: E402
from p89_public_family import superiority_verdict  # noqa: E402  (one definition of the combined verdict)
import p89_public_family as P89  # noqa: E402  (the macro file of form (a) under each mask)

HIER_SEED = 11          # the hierarchical stream of p89_public_family (HIER_SEED)

RULES = PLAN.RULES
R8 = list(PLAN.R8)
SETUP_COMBOS = [r for r in R8 if 'S' in r]
PLAIN_COMBOS = [r for r in R8 if 'S' not in r]
ARMS = (('released', 'Pol', 'policy'), ('control', 'Ctl', 'single-rule control'),
        ('specialist', 'Spec', 'specialist of the combination'))
# form key -> (macro word, k of the sampled side or None, rule side, description)
FORMS = {
    'greedy_best4': ('GreedyBestFour', None, 'best4',
                     'one greedy schedule vs the per-instance best of the four rules'),
    'greedy_vali': ('GreedyValiRule', None, 'vali',
                    'one greedy schedule vs the rule with the lowest validation mean'),
    'sample4': ('SampleFour', 4, 'best4',
                'best of 4 sampled vs the per-instance best of the four rule schedules'),
    'sample16': ('SampleSixteen', 16, 'best16',
                 'best of 16 sampled vs the per-instance best of 16 rule schedules '
                 '(four rules x tie-break seeds 50-53)'),
    'greedy3': ('GreedyPlusThree', 3, 'best4',
                'greedy plus 3 sampled vs the per-instance best of the four rules'),
    'greedy15': ('GreedyPlusFifteen', 15, 'best16',
                 'greedy plus 15 sampled vs the per-instance best of 16 rule schedules'),
}
OUT_REL = ('manuscript/macros_rulecmp_{f}.tex', 'results/rulecmp_{f}.json',
           'results/rulecmp_{f}.md')
NAME_RX = re.compile(r'^[A-Za-z]+$')
ALPHA = 0.05
VERDICTS = ('ahead', 'behind', 'not distinguishable')
# the family under a mask other than the non-delay mask (--mask)
MASK_NOTE = ('every run trained and every schedule built under the {m} mask: the policy\'s '
             'sampled and greedy schedules under its run\'s own mask, the four rules rolled out '
             'under it (scripts/p109_public_active_rules.py; traces p90 --rule_mask {m}); a trace '
             'counts the pairs the {m} mask admits')
PARTIAL_NOTE = 'PARTIAL REPORT: the quantities whose arrays exist; not for the manuscript'


def outside_outputs(where, root):
    """A partial report under a mask is written only outside the repository root and its
    paper, manuscript and results folders (p89 outside_outputs)."""
    return P89.outside_outputs(where, root, '--partial --out')


class Refusal(SystemExit):
    """A check failed; the outputs are removed and nothing is certified."""


def refuse(msg):
    raise Refusal(f'REFUSING: {msg}')


def rel(a, b):
    return 100.0 * (np.asarray(a, float) - np.asarray(b, float)) / np.asarray(b, float)


def relpath(p, root):
    try:
        r = os.path.relpath(p, root)
    except ValueError:
        return p
    return p if r.startswith('..') else r


# ================================================================== inputs
class Inputs:
    """Every array the comparison reads, checked as it is read."""

    def __init__(self, fam, root, base, partial, vali_base=None):
        self.fam, self.root, self.base, self.partial = fam, root, base, partial
        self.vali_base = vali_base
        self.missing, self.n_files = [], 0
        recs = fam.check_records(root)
        if recs:
            refuse('; '.join(recs))
        self.names = {R: fam.names(R, root=root) for R in R8}
        if not self.names['N']:
            refuse(f'no test records in {fam.data}')
        for R in R8:
            if self.names[R] != self.names['N']:
                refuse(f'combination {R} does not share the instance names of N, so row i is '
                       f'not one base instance')
        self.n = len(self.names['N'])
        self.rec_mtime = fam.records_mtime(root=root)
        self.vali_sel, probs = fam.vali_select(root)
        if probs:
            refuse('the runs do not share one validation set: ' + '; '.join(probs))
        if self.vali_sel is None:
            if not partial:
                raise REG.MissingResult('missing training configuration: the validation '
                                        'instances of form (c) are undefined')
        if fam.vali_kind == 'records':
            self.vali_names = fam.names('N', fam.vali_data, root)
            self.vali_mtime = fam.records_mtime(fam.vali_data, root)
            # the validation records are the ones every run of the family validated on
            for r in fam.runs:
                cp = os.path.join(root, r.config)
                if not os.path.exists(cp):
                    if not partial:
                        raise REG.MissingResult(f'missing training configuration: {r.config}')
                    continue
                with open(cp) as f:
                    v = json.load(f).get('pub_vali_dir')
                if v != fam.vali_data:
                    refuse(f'{r.config} validated on {v!r}, not on {fam.vali_data}')
        else:
            self.vali_names, self.vali_mtime = None, 0.0
        ns = self.vali_sel if self.vali_sel is not None else '?'
        self.vali_text = (
            f'the first {ns} records of {fam.vali_data} in sorted order, the instances the '
            f'training runs validated on (all {len(self.vali_names)} records: field all_records)'
            if fam.vali_kind == 'records' else
            f'the {ns} instances the trainer builds from the training configuration of '
            f'{fam.vali_source()[0].model}, the instances the training runs validated on')
        # the mask of every run: its training configuration (absent = the non-delay mask)
        for r in fam.runs:
            cp = os.path.join(root, r.config)
            if os.path.exists(cp):
                with open(cp) as f:
                    m = json.load(f).get('mask_mode') or 'nondelay'
                if m != fam.mask:
                    refuse(f'{r.config} records the {m} mask; the comparison reads the family '
                           f'under the {fam.mask} mask')
        # under a mask the family is read while runs may still train: a run counts once its
        # training log holds the completion line; the stored rule arrays are those of p109
        self.unfinished = set()
        if fam.word:
            for r in fam.runs:
                if not PLAN.run_finished(r, root):
                    if not partial:
                        refuse(f'the training of {r.model} has not finished (no completion line '
                               f'in {r.log})')
                    self.unfinished.add(r.model)
                    self.missing.append(f'{r.log} (training not finished)')
            d = os.path.join(root, fam.rule_folder)
            if os.path.isdir(d):
                why = PLAN.rule_mask_problem(d, fam.mask, PLAN.STORED_TIE_SEED)
                if why:
                    refuse(f'{fam.rule_folder}: {why}')

    def rp(self, p):
        return relpath(p, self.root)

    def absent(self, path):
        if self.partial:
            self.missing.append(self.rp(path))
            return None
        raise REG.MissingResult(f'missing input: {self.rp(path)}')

    def ck_mtime(self, run):
        p = os.path.join(self.root, run.checkpoint)
        if os.path.exists(p):
            return os.stat(p).st_mtime
        if self.partial:
            return None
        refuse(f'the checkpoint {run.checkpoint} is missing, so the age of its arrays cannot '
               f'be checked')

    def fresh(self, path, ck=None, records=None):
        t = os.stat(path).st_mtime
        if t < (self.rec_mtime if records is None else records):
            refuse(f'{self.rp(path)} is older than the records it was evaluated on')
        if ck is not None and t < ck:
            refuse(f'{self.rp(path)} is older than the checkpoint it belongs to')

    def result(self, path, n, ck=None, records=None):
        """Makespan column of a Result_*.npy file [n, 2]."""
        if not os.path.exists(path):
            return self.absent(path)
        a = np.load(path)
        if a.shape != (n, 2) or not np.all(np.isfinite(a)) or np.any(a[:, 0] <= 0):
            refuse(f'{self.rp(path)} has shape {a.shape} or non-positive values, want ({n}, 2)')
        self.fresh(path, ck, records)
        self.n_files += 1
        return a[:, 0].astype(float)

    def greedy(self, run, R):
        if run.model in self.unfinished:          # partial reading only (see __init__)
            return None
        return self.result(self.fam.greedy_path(run, R, self.root), self.n, self.ck_mtime(run))

    def rule(self, q, R):
        return self.result(self.fam.rule_path(q, R, self.root), self.n)

    def sample(self, run, R):
        if run.model in self.unfinished:
            return None
        p = self.fam.sample_path(run, R, self.base)
        if not os.path.exists(p):
            return self.absent(p)
        why = PLAN.check_sample(p, self.names[R], self.fam.K, self.ck_mtime(run), self.fam.mask)
        if why:
            refuse(f'{self.rp(p)}: {why}')
        self.fresh(p)
        self.n_files += 1
        return np.load(p)['makespan'].astype(float)

    def trace(self, who, R, stored, run=None):
        if run is not None and run.model in self.unfinished:
            return None
        p = self.fam.trace_path(who, R, self.base)
        if not os.path.exists(p):
            return self.absent(p)
        why = PLAN.check_trace(p, self.names[R], stored, self.ck_mtime(run) if run else None,
                               self.fam.mask)
        if why:
            refuse(f'{self.rp(p)}: {why}')
        self.fresh(p)
        self.n_files += 1
        return np.load(p, allow_pickle=True)

    def rule_folder(self, folder, R, n, records, vali_json=False, seed=None):
        """[4, n] makespans of the four rules in one folder of rule arrays (tie-break
        seed or validation), with its summary and its mask checked; None if absent
        (partial)."""
        paths = [os.path.join(folder, f'Result_{q}_{R}.npy') for q in RULES]
        miss = [p for p in paths if not os.path.exists(p)]
        if miss:
            for p in miss[1:]:
                self.absent(p)
            return self.absent(miss[0])
        why = PLAN.check_rule_folder(folder, n, regimes=(R,), vali_json=vali_json,
                                     mask=self.fam.mask, seed=seed)
        if why:
            refuse(f'{self.rp(folder)}: {why}')
        return np.stack([self.result(p, n, records=records) for p in paths])

    def ties(self, t, R):
        return self.rule_folder(self.fam.ties_dir(t, self.base, self.root), R, self.n,
                                self.rec_mtime, seed=t)

    def vali(self, R):
        folder = self.fam.vali_folder(self.base, self.vali_base, self.root)
        if self.fam.vali_kind == 'records':
            n = len(self.vali_names)
            return self.rule_folder(folder, R, n, self.vali_mtime, seed=PLAN.STORED_TIE_SEED)
        jp = os.path.join(folder, 'valirules.json')
        if not os.path.exists(jp):
            return self.absent(jp)
        with open(jp) as f:
            v = json.load(f)
        n, src = int(v['n']), self.fam.vali_source()[0].model
        if v.get('family_model') != src:
            refuse(f'{self.rp(jp)} holds the validation instances of {v.get("family_model")}, '
                   f'want {src}')
        want = self.fam.vali_n(self.root)
        if want is not None and n != want:
            refuse(f'{self.rp(jp)} lists {n} validation instances, the configuration {want}')
        return self.rule_folder(folder, R, n, 0.0, vali_json=True, seed=PLAN.STORED_TIE_SEED)


# ================================================================== statistics
def best_k(S, k):
    """[seeds, n] unbiased estimates of the best of k draws from S [seeds, n, K]."""
    return np.array([[best_of_k(S[a, i], k) for i in range(S.shape[1])]
                     for a in range(S.shape[0])])


def p_se(S, k, ref):
    """[seeds, n, 2]: P(best of k < ref), P(best of k == ref) per seed and instance."""
    return np.array([[p_shorter_equal(S[a, i], k, ref[i]) for i in range(S.shape[1])]
                     for a in range(S.shape[0])])


def shares(d):
    """Percent of entries shorter / equal / longer (d < 0, = 0, > 0)."""
    d = np.asarray(d)
    return dict(shorter=float(np.mean(d < 0)) * 100, equal=float(np.mean(d == 0)) * 100,
                longer=float(np.mean(d > 0)) * 100)


def seed_shares(d):
    """shares per seed of d [seeds, n], averaged over the seeds (p89 seed_shares)."""
    per = [shares(x) for x in d]
    return {key: float(np.mean([p[key] for p in per])) for key in per[0]}


def package(d, B, sh=None):
    """Interval package of one comparison d [seeds, n], the calls of p89's
    component(): stats_family.hier_boot (B, seed 11, draws kept), seed_t,
    seed_t_p(less) and boot_p_one_sided(less) on the hierarchical draws; with the
    shares shorter, equal and longer in percent (sign of d, or the expected shares
    sh [seeds, n, 2] of a sampled quantity)."""
    hb = SF.hier_boot(d, B=B, seed=HIER_SEED, return_draws=True)
    t = SF.seed_t(d)
    p_t, p_tg = SF.seed_t_p(d, 'less'), SF.seed_t_p(d, 'greater')
    out = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], t_lo=t['lo'], t_hi=t['hi'],
               seed_means=t['seed_means'], direction=SF.direction(hb['lo'], hb['hi']),
               p_t=1.0 if np.isnan(p_t) else p_t,
               p_boot=SF.boot_p_one_sided(hb['draws'], 'less'),
               p_t_greater=1.0 if np.isnan(p_tg) else p_tg,
               p_boot_greater=SF.boot_p_one_sided(hb['draws'], 'greater'),
               n=int(d.shape[1]), seeds=int(d.shape[0]))
    if sh is None:
        out.update(seed_shares(d))
    else:
        a, e = float(sh[..., 0].mean()) * 100.0, float(sh[..., 1].mean()) * 100.0
        out.update(shorter=a, equal=e, longer=100.0 - a - e)
    return out


def verdict(c):
    """'ahead', 'behind' or 'not distinguishable' for one comparison after Holm:
    ahead needs the 95% interval below zero and both Holm-adjusted p-values for
    "shorter" (seed t, bootstrap) below 0.05; behind is the mirror image with the
    p-values for "longer"; anything else is not distinguishable."""
    return superiority_verdict(c['lo'], c['hi'], c['p_t_holm'], c['p_boot_holm'],
                               c['p_t_greater_holm'], c['p_boot_greater_holm'], ALPHA)


select_rule = PLAN.select_rule      # lowest mean; a tie goes to the first in RULES order


def trace_stats(z):
    """Share of single-pair decisions and setup share of machine capacity, in percent."""
    dec, ops, msp = z['decisions'], z['ops'], np.asarray(z['makespan'], float)
    pairs = np.concatenate([np.asarray(x, float)[:, 0] for x in dec])
    frac, s_tot, p_tot = [], 0.0, 0.0
    for o, cm in zip(ops, msp):
        o = np.asarray(o, float)
        m, pt, s = o[:, 0], o[:, 1], o[:, 2]
        used = len(np.unique(m.astype(int)))
        frac.append(s.sum() / (used * cm))
        s_tot += s.sum()
        p_tot += pt.sum()
    return dict(one_pair=100.0 * float((pairs == 1).mean()), setup_share=100.0 * float(np.mean(frac)),
                pairs_mean=float(pairs.mean()), decisions=int(len(pairs)),
                setup_over_processing=float(s_tot / p_tot) if p_tot else 0.0)


def analyse(I, B):
    fam = I.fam
    res = dict(combos={})
    for R in R8:
        E = dict(arms={})
        rules4 = [I.rule(q, R) for q in RULES]
        if any(x is None for x in rules4):
            res['combos'][R] = E
            continue
        rules4 = np.stack(rules4)
        best4 = rules4.min(0)
        ties = [I.ties(t, R) for t in PLAN.TIE_SEEDS[1:]]
        rules16 = None if any(x is None for x in ties) else np.concatenate([rules4] + ties)
        best16 = None if rules16 is None else rules16.min(0)
        vali = I.vali(R)
        qv = None
        if vali is not None and I.vali_sel is not None:
            ns = I.vali_sel
            if vali.shape[1] < ns:
                refuse(f'{R}: {vali.shape[1]} validation instances, the runs validated on {ns}')
            vm = vali[:, :ns].mean(1)
            qv = select_rule(vm)
            E['vali'] = dict(rule=RULES[qv], means={q: float(v) for q, v in zip(RULES, vm)},
                             n=int(ns), instances=I.vali_text)
            if fam.vali_kind == 'records':
                va = vali.mean(1)
                E['vali']['all_records'] = dict(
                    rule=RULES[select_rule(va)], means={q: float(v) for q, v in zip(RULES, va)},
                    n=int(vali.shape[1]),
                    note='selection on all validation records, for the supplement; no macro')
        means = rules4.mean(1)
        qs = select_rule(means)
        sp = dict(best_single=RULES[qs], means={q: float(v) for q, v in zip(RULES, means)},
                  best4_vs_best_single=float(rel(best4, rules4[qs]).mean()),
                  range=float(rel(rules4.max(0), rules4.min(0)).mean()),
                  mean_gap=float(100.0 * (means.max() - means.min()) / means.min()),
                  best_rule_share={q: float(np.mean(rules4[j] == best4)) * 100.0
                                   for j, q in enumerate(RULES)})
        if qv is not None:
            sp['best4_vs_vali_rule'] = float(rel(best4, rules4[qv]).mean())
        if rules16 is not None:
            sp['best16_vs_best4'] = float(rel(best16, best4).mean())
            sp['distinct16'] = float(np.mean([len(np.unique(rules16[:, i]))
                                              for i in range(I.n)]))
        E['rules'] = sp
        ref = {'best4': best4, 'best16': best16,
               'vali': None if qv is None else rules4[qv]}
        for arm, _, _ in ARMS:
            runs = [r for r in fam.runs if r.arm == arm and R in r.regimes]
            if not runs:                      # an arm left out by --arms
                continue
            g = [I.greedy(r, R) for r in runs]
            if any(x is None for x in g):
                continue
            g = np.stack(g)
            S = [I.sample(r, R) for r in runs]
            S = None if any(x is None for x in S) else np.stack(S)
            A = {}
            for key, (_, k, side, _) in FORMS.items():
                b = ref[side]
                if b is None or (k is not None and S is None):
                    continue
                if k is None:
                    A[key] = package(rel(g, b[None]), B)
                elif key.startswith('greedy'):
                    Sg = np.minimum(S, g[:, :, None])
                    A[key] = package(rel(best_k(Sg, k), b[None]), B, p_se(Sg, k, b))
                else:
                    A[key] = package(rel(best_k(S, k), b[None]), B, p_se(S, k, b))
            E['arms'][arm] = A
        # traces: each seed of the released policy and each rule
        tr = dict(policy={}, rules={})
        for r in fam.released():
            z = I.trace(r.model, R, I.greedy(r, R), r)
            if z is not None:
                tr['policy'][f's{r.seed}'] = trace_stats(z)
        for j, q in enumerate(RULES):
            z = I.trace(q, R, rules4[j])
            if z is not None:
                tr['rules'][q] = trace_stats(z)
        for who in ('policy', 'rules'):
            if tr[who]:
                for key in ('one_pair', 'setup_share'):
                    v = [x[key] for x in tr[who].values()]
                    tr[f'{who}_{key}'] = dict(mean=float(np.mean(v)), lo=float(min(v)),
                                              hi=float(max(v)), n=len(v))
        E['traces'] = tr
        res['combos'][R] = E
    # Holm over the combinations of each arm and form
    holm = {}
    for arm, _, _ in ARMS:
        for key in FORMS:
            have = [R for R in R8 if key in res['combos'][R].get('arms', {}).get(arm, {})]
            if not have:
                continue
            for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
                adj = SF.holm([res['combos'][R]['arms'][arm][key][col] for R in have])
                for R, x in zip(have, adj):
                    res['combos'][R]['arms'][arm][key][f'{col}_holm'] = x
            cells = [res['combos'][R]['arms'][arm][key] for R in have]
            for c in cells:
                c['verdict'] = verdict(c)
            holm[f'{arm}|{key}'] = dict(
                combinations=have, family_size=len(have),
                **{v.replace(' ', '_'): sum(c['verdict'] == v for c in cells) for v in VERDICTS})
    res['holm'] = holm
    # ranges of the measured variables over the combinations with and without setups
    rng = {}
    for lab, combos in (('setup', SETUP_COMBOS), ('plain', PLAIN_COMBOS)):
        for key in ('one_pair', 'setup_share'):
            v = [res['combos'][R]['traces'][f'policy_{key}']['mean'] for R in combos
                 if f'policy_{key}' in res['combos'][R].get('traces', {})]
            if v and len(v) == len(combos):
                rng[f'{lab}_{key}'] = dict(lo=float(min(v)), hi=float(max(v)), combos=combos)
    res['trace_ranges'] = rng
    return res


# ================================================================== cross-check
def cross_check_p89(res, root, tag='Sdten', mac='manuscript/macros_public.tex', prefix='Pub'):
    """Form (a) of the policy against the macros p89 wrote for the public test set
    (\\<prefix><tag>Rule<R>..., Sdten for the headline set; the family under a mask: the
    macro file and prefix of p89 --family <mask>); reported, not enforced."""
    p = os.path.join(root, mac)
    if not os.path.exists(p):
        return dict(status=f'{mac} absent')
    txt = open(p).read()
    got = dict(re.findall(r'\\newcommand\{\\(' + prefix + tag + r'Rule[A-Za-z]*)\}\{([^}]*)\}',
                          txt))
    out, n_eq, n_all = {}, 0, 0
    for R in R8:
        c = res['combos'][R].get('arms', {}).get('released', {}).get('greedy_best4')
        if c is None:
            continue
        mine = {'': f2(c['mean']), 'Lo': f2(c['lo']), 'Hi': f2(c['hi']),
                'Short': u1(c['shorter']), 'Long': u1(c['longer']),
                'PHolm': f'{c["p_t_holm"]:.2g}'}
        row = {}
        for suf, v in mine.items():
            name = f'{prefix}{tag}Rule{R}{suf}'
            if name in got:
                n_all += 1
                n_eq += got[name] == v
                row[suf or 'mean'] = dict(p89=got[name], p93=v, equal=got[name] == v)
        out[R] = row
    return dict(status=f'{n_eq} of {n_all} values equal', combos=out)


# ================================================================== output
def f2(x):
    """Signed two decimals of the unrounded value (a negative value that rounds to zero
    prints -0.00), the formatter of p88 and p89."""
    x = float(x)
    return f'{0.0 if x == 0 else x:+.2f}'


def u1(x):
    return f'{float(x):.1f}'


def pv(x):
    return f'{float(x):.2g}'


def macros(fam, res, I, B):
    M = fam.macro
    L = []

    def line(cmds, comment):
        for name, _ in cmds:
            if not NAME_RX.match(name):
                raise ValueError(f'macro name {name!r} is not letters only')
        L.append(''.join(f'\\newcommand{{\\{name}}}{{{val}}}' for name, val in cmds)
                 + f'  % {comment}')

    line([(f'{M}SeedN', len(fam.seeds)), (f'{M}Boot', B), (f'{M}SampleK', fam.K),
          (f'{M}InstN', I.n), (f'{M}TieSeedN', len(PLAN.TIE_SEEDS))],
         'training seeds per arm, bootstrap replicates, sampled schedules stored per instance, '
         'test instances, tie-break seeds per rule (16 rule schedules)')
    nv = next((res['combos'][R]['vali']['n'] for R in R8 if 'vali' in res['combos'][R]), None)
    line([(f'{M}ValiN', nv)], 'validation instances on which the rule of form (c) is chosen '
         '(the instances the training runs validated on)')
    for arm, aw, alab in ARMS:
        if arm not in fam.arms:               # left out by --arms: no macro at all
            continue
        for key, (fw, _, _, desc) in FORMS.items():
            for R in R8:
                c = res['combos'][R]['arms'][arm][key]
                nm = f'{M}{aw}{fw}{R}'
                line([(nm, f2(c['mean'])), (nm + 'Lo', f2(c['lo'])), (nm + 'Hi', f2(c['hi'])),
                      (nm + 'TLo', f2(c['t_lo'])), (nm + 'THi', f2(c['t_hi'])),
                      (nm + 'Verdict', c['verdict']),
                      (nm + 'PT', pv(c['p_t'])), (nm + 'PBoot', pv(c['p_boot'])),
                      (nm + 'PHolm', pv(c['p_t_holm'])), (nm + 'PBootHolm', pv(c['p_boot_holm'])),
                      (nm + 'PTBehind', pv(c['p_t_greater'])),
                      (nm + 'PBootBehind', pv(c['p_boot_greater'])),
                      (nm + 'PHolmBehind', pv(c['p_t_greater_holm'])),
                      (nm + 'PBootHolmBehind', pv(c['p_boot_greater_holm'])),
                      (nm + 'Dir', SF.TEX[c['direction']]), (nm + 'Short', u1(c['shorter'])),
                      (nm + 'Equal', u1(c['equal'])), (nm + 'Long', u1(c['longer']))],
                     f'{alab}, {R}: {desc}; mean d, hierarchical 95% CI, seed-t CI, verdict '
                     f'(ahead / behind / not distinguishable), one-sided p for shorter (seed t, '
                     f'bootstrap; raw, Holm), the same for longer, direction of the interval '
                     f'alone, % shorter / equal / longer')
            h = res['holm'][f'{arm}|{key}']
            line([(f'{M}{aw}{fw}AheadN', h['ahead']), (f'{M}{aw}{fw}BehindN', h['behind']),
                  (f'{M}{aw}{fw}NotDistN', h['not_distinguishable'])],
                 f'{alab}: combinations (of {h["family_size"]}) with the verdict ahead, behind '
                 f'and not distinguishable, {desc}')
    for R in R8:
        E = res['combos'][R]
        sp, v = E['rules'], E['vali']
        line([(f'{M}ValiRule{R}', v['rule']), (f'{M}BestSingleRule{R}', sp['best_single']),
              (f'{M}RuleSpread{R}', f2(sp['best4_vs_best_single'])),
              (f'{M}RuleSpreadVali{R}', f2(sp['best4_vs_vali_rule'])),
              (f'{M}RuleRange{R}', u1(sp['range'])), (f'{M}RuleMeanGap{R}', f'{sp["mean_gap"]:.2f}'),
              (f'{M}RuleSixteenVsFour{R}', f2(sp['best16_vs_best4'])),
              (f'{M}RuleDistinct{R}', u1(sp['distinct16']))],
             f'{R}: rule chosen on the validation instances, rule with the lowest test mean, '
             f'best of four vs that rule and vs the validation rule (mean %), mean per-instance '
             f'range of the four (%), gap between the rule means (%), best of 16 rule schedules '
             f'vs best of four (%), distinct makespans among the 16')
        tr = E['traces']
        line([(f'{M}OnePair{R}', u1(tr['policy_one_pair']['mean'])),
              (f'{M}OnePairRuleLo{R}', u1(tr['rules_one_pair']['lo'])),
              (f'{M}OnePairRuleHi{R}', u1(tr['rules_one_pair']['hi'])),
              (f'{M}SetupShare{R}', u1(tr['policy_setup_share']['mean'])),
              (f'{M}SetupShareRuleLo{R}', u1(tr['rules_setup_share']['lo'])),
              (f'{M}SetupShareRuleHi{R}', u1(tr['rules_setup_share']['hi']))],
             f'{R}: % of decisions with exactly one admitted pair (policy, mean of the seeds; '
             f'lowest and highest rule), % of machine capacity taken by setups (same)')
    rg = res['trace_ranges']
    line([(f'{M}OnePairSetupLo', u1(rg['setup_one_pair']['lo'])),
          (f'{M}OnePairSetupHi', u1(rg['setup_one_pair']['hi'])),
          (f'{M}OnePairPlainLo', u1(rg['plain_one_pair']['lo'])),
          (f'{M}OnePairPlainHi', u1(rg['plain_one_pair']['hi'])),
          (f'{M}SetupShareLo', u1(rg['setup_setup_share']['lo'])),
          (f'{M}SetupShareHi', u1(rg['setup_setup_share']['hi']))],
         'policy: lowest and highest % of single-pair decisions over the combinations with '
         'setups and over those without, lowest and highest setup share of machine capacity '
         'over the combinations with setups')
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(L))
    if len(names) != len(set(names)):
        refuse('duplicate macro names in the output')
    return L, names


def collision_check(names, root, out_mac):
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand|def)\s*\{?\\([A-Za-z]+)')
    import glob as _g
    files = sorted(_g.glob(os.path.join(root, 'manuscript', 'macros_*.tex')) +
                   _g.glob(os.path.join(root, 'paper', 'macros_*.tex')) +
                   [os.path.join(root, d, 'macros.tex') for d in ('manuscript', 'paper')])
    for f in files:
        if not os.path.exists(f) or os.path.abspath(f) == os.path.abspath(out_mac):
            continue
        clash = set(names) & set(rx.findall(open(f, errors='replace').read()))
        if clash:
            refuse(f'macro name collision with {relpath(f, root)}: {sorted(clash)[:5]}')


def ci(c):
    return f'{f2(c["mean"])} [{f2(c["lo"])}, {f2(c["hi"])}]'


def markdown(fam, res, settings):
    md = [f'# Comparison with dispatching rules, {fam.name} family'
          + (f' under the {fam.mask} mask' if fam.word else '')
          + ('' if fam.headline else f', test set {fam.set}'), '']
    if fam.word:
        md += [settings['mask_note'], '']
        if settings.get('partial_report'):
            md = [f'# {PARTIAL_NOTE}', ''] + md
    md += [f'{settings["statistics"]}', '',
           f'Test instances: {settings["n_test"]}; seeds {settings["seeds"]}; K = {fam.K}; '
           f'tie-break seeds {list(PLAN.TIE_SEEDS)}.', '']
    for key, (fw, _, _, desc) in FORMS.items():
        md += [f'## {fw}: {desc}', '',
               '| comb. | ' + ' | '.join(f'{lab} d [95% CI] | Holm p shorter (t, boot) | '
                                         f'verdict | short/eq/long %'
                                         for _, _, lab in ARMS) + ' |',
               '|---|' + '---|---|---|---|' * len(ARMS)]
        for R in R8:
            cells = []
            for arm, _, _ in ARMS:
                c = res['combos'][R].get('arms', {}).get(arm, {}).get(key)
                if c is None:
                    cells += ['-', '-', '-', '-']
                else:
                    cells += [ci(c), f'{pv(c.get("p_t_holm", float("nan")))}, '
                                     f'{pv(c.get("p_boot_holm", float("nan")))}',
                              c.get('verdict', '-'),
                              f'{c["shorter"]:.0f}/{c["equal"]:.0f}/{c["longer"]:.0f}']
            md.append(f'| {R} | ' + ' | '.join(cells) + ' |')
        md.append('')
    md += ['## Rules', '', '| comb. | validation rule (all records) | lowest test mean | '
           'best4 vs that rule | best4 vs validation rule | range | best16 vs best4 | '
           'distinct of 16 |', '|---|---|---|---|---|---|---|---|']
    for R in R8:
        E = res['combos'][R]
        sp, v = E.get('rules', {}), E.get('vali')
        alt = v.get('all_records', {}).get('rule', '-') if v else '-'
        md.append(f'| {R} | ' + (f'{v["rule"]} ({alt})' if v else '-') +
                  f' | {sp.get("best_single", "-")} | '
                  + ' | '.join(f2(sp[k]) if k in sp else '-' for k in
                               ('best4_vs_best_single', 'best4_vs_vali_rule'))
                  + f' | {u1(sp["range"]) if "range" in sp else "-"} | '
                  + (f2(sp['best16_vs_best4']) if 'best16_vs_best4' in sp else '-')
                  + f' | {u1(sp["distinct16"]) if "distinct16" in sp else "-"} |')
    md += ['', '## Traces (percent)', '', '| comb. | one pair, policy (seeds) | one pair, rules | '
           'setup share, policy | setup share, rules |', '|---|---|---|---|---|']
    for R in R8:
        tr = res['combos'][R].get('traces', {})

        def cell(k):
            x = tr.get(k)
            return '-' if x is None else f'{x["mean"]:.1f} ({x["lo"]:.1f}-{x["hi"]:.1f}, n={x["n"]})'
        md.append(f'| {R} | {cell("policy_one_pair")} | {cell("rules_one_pair")} | '
                  f'{cell("policy_setup_share")} | {cell("rules_setup_share")} |')
    return '\n'.join(md) + '\n'


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--family', required=True, choices=('public', 'factory'))
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--set', default='',
                    help=f'public test set (default {PLAN.HEADLINE}, the headline set)')
    ap.add_argument('--arms', nargs='*', default=None, choices=PLAN.ARMS,
                    help='arms to analyse (default: all three on the headline set, the '
                         'released policy elsewhere)')
    ap.add_argument('--vali-input-root', default='',
                    help='another public test set: folder standing in for the headline '
                         'result folder whose rulecmp_vali the rule of form (c) comes from')
    ap.add_argument('--K', default=None,
                    help='sampled schedules per instance the arrays must hold (default: '
                         'RULECMP_K, else 32; at least 16)')
    ap.add_argument('--root', default=REPO, help='repository root (records, checkpoints, '
                                                 'stored arrays)')
    ap.add_argument('--input-root', default='',
                    help='folder holding the rulecmp_* subfolders (default: the family\'s '
                         'result folder under --root)')
    ap.add_argument('--out', default=REPO, help='root of the outputs (manuscript/, results/)')
    ap.add_argument('--partial', action='store_true',
                    help='skip quantities whose arrays are absent; never write the macro file '
                         '(under the active mask: a partial report, only under --out outside '
                         'the paper and results folders)')
    ap.add_argument('--mask', default='nondelay', choices=PLAN.MASKS,
                    help='public family: nondelay (default) or active (the family trained '
                         'under the active mask against the rules under the active mask)')
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    if a.set and a.family != 'public':
        sys.exit('*** REFUSING TO CERTIFY ***\nREFUSING: --set applies to the public family')
    try:
        label = PLAN.label_of(a.family, a.set or None, a.mask)
        if a.partial and PLAN.mask_word(a.mask):
            outside_outputs(a.out, root)
    except SystemExit as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    outs = [os.path.join(a.out, p.format(f=label)) for p in OUT_REL]
    if a.partial:
        outs[0] = None

    def clean():
        for p in outs:
            if p and os.path.exists(p):
                os.remove(p)
                print(f'removed {p} (refusing to certify)')
    try:
        fam = PLAN.Family(a.family, K=a.K, set_name=a.set or None, arms=a.arms, mask=a.mask)
        base = os.path.abspath(a.input_root) if a.input_root else os.path.join(root, fam.campaign)
        I = Inputs(fam, root, base, a.partial,
                   os.path.abspath(a.vali_input_root) if a.vali_input_root else None)
        res = analyse(I, a.boot)
        L = names = None
        if not a.partial:
            L, names = macros(fam, res, I, a.boot)
            collision_check(names, root, outs[0])
    except (Refusal, REG.MissingResult, SystemExit) as e:
        clean()
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    settings = dict(
        family=a.family, label=fam.label, test_set=getattr(fam, 'set', None),
        arms=list(fam.arms), n_test=I.n, seeds=list(fam.seeds), runs=len(fam.runs),
        test_records=fam.data, stored_folder=fam.folder,
        input_folder=relpath(base, root), K=fam.K, tie_seeds=list(PLAN.TIE_SEEDS),
        rules=list(RULES), boot=a.boot, partial=a.partial, files_read=I.n_files,
        validation=I.vali_text,
        verdict_rule=('ahead: 95% interval below zero and both Holm-adjusted one-sided p for '
                      'shorter (seed t, bootstrap) below 0.05; behind: interval above zero and '
                      'both Holm-adjusted one-sided p for longer below 0.05 (own Holm step); '
                      'otherwise not distinguishable'),
        statistics=(f'd = 100 (a - b) / b per training seed and test instance; crossed '
                    f'hierarchical bootstrap over seeds and instances (stats_family.hier_boot, '
                    f'B = {a.boot}, fresh numpy default_rng(11) per quantity), 95% percentile '
                    f'interval without Bonferroni widening; one-sided seed-level t test (k - 1 df) '
                    f'and one-sided bootstrap p, Holm over the combinations of each arm and form '
                    f'(the stats_family calls of p89_public_family.component, stats_family.holm)'),
        protocol_note=('The study of 2026-11-11 widened its intervals by Bonferroni over the '
                       'eight combinations (alpha 0.05/8); the protocol of '
                       'manuscript/supp_new/sn07_statistics.tex uses 95% intervals and Holm on the '
                       'one-sided tests, which this script follows.'))
    if fam.word:
        settings.update(mask=fam.mask, rule_folder=fam.rule_folder, campaign_folder=fam.campaign,
                        mask_note=MASK_NOTE.format(m=fam.mask))
        if a.partial:
            settings['partial_report'] = PARTIAL_NOTE
    out = dict(settings=settings, forms={k: v[3] for k, v in FORMS.items()}, **res)
    if a.family == 'public':
        if fam.word:
            pf = P89.Fam(fam.mask)
            mac, pre = pf.out_files[2], pf.prefix
        else:
            mac, pre = 'manuscript/macros_public.tex', 'Pub'
        out['cross_check_p89'] = cross_check_p89(res, root, fam.p89_tag, mac, pre)
        print(f'cross-check with {os.path.basename(mac)} (form (a), policy, {pre}{fam.p89_tag}Rule): '
              f'{out["cross_check_p89"]["status"]}')
    if a.partial:
        out['missing'] = I.missing
    for p in outs[1:]:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(outs[1], 'w') as f:
        json.dump(jsonable(out), f, indent=1)
    with open(outs[2], 'w') as f:
        f.write(markdown(fam, res, settings))
    if not a.partial:
        header = (f'Generated by scripts/p93_rule_comparison.py -- do not hand-edit\n'
                  f'% {a.family} family' +
                  (f' under the {fam.mask} mask (--mask {fam.mask}: {MASK_NOTE.format(m=fam.mask)})'
                   if fam.word else '') +
                  ('' if fam.headline else f', test set {fam.set}, '
                                           f'arms {" ".join(fam.arms)}') +
                  f', seeds {",".join(map(str, fam.seeds))}, {I.n} test '
                  f'instances, K = {fam.K} sampled schedules per instance, rule schedules with '
                  f'tie-break seeds {",".join(map(str, PLAN.TIE_SEEDS))}; B = {a.boot}, hierarchical '
                  f'intervals fresh default_rng(11), 95%, Holm over the eight combinations.\n'
                  f'% Gaps in %: policy quantity minus rule quantity over the rule quantity '
                  f'(positive = the policy is longer).')
        emit(outs[0], L, header=header)
    print(f'wrote {", ".join(p for p in outs if p)}' +
          (f' ({len(I.missing)} inputs absent, skipped)' if a.partial else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
