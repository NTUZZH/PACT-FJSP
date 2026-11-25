r"""P119: asymmetric setup matrices at the trained magnitude, against the out-of-range setups.

THE SETS (exact environment, training seeds of arm_registry, PACT_SEEDS=301-305)
  ind     data/FAMILY/10x25, arrays in arm_registry.res_dir('10x25_cpuref') (10x25_xmask in the
          exact environment), the in-distribution set p33 reads
  setup   data/FAMILY/10x25_oodsetup: cross-class setups U[100%, 200%] of the mean processing
          time, the matrix asymmetric
  asym    data/FAMILY/10x25_asymsetup: cross-class setups U[50%, 100%], the trained range, the
          matrix asymmetric (scripts/p29_ood.py; arrays from scripts/run_further_sets_eval.sh)
  setup and asym come from one generator call with the cross range alone changed, the same base
  instances and descriptor seeds; on N, L, W and LW all three sets hold the same constraint
  fields.

THE POLICY AGAINST THE BEST DISPATCHING RULE: the procedure of scripts/p33_ood_table.py (the
procedure of FamRule<R> of scripts/p88_family_inference.py), by p33's functions
  lead       d = 100 (policy_s - best) / best per training seed s and test instance, best the
             per-instance minimum of FIFO, MOR, SPT and MWKR (p33.differences: p88.seed_pairs,
             p88.released, p88.pool_reader); per combination p88.superiority_block
             (p33.family_rows): mean, hierarchical 95% interval (B = 20000, a fresh
             default_rng(11) per combination), one-sided seed-level t and bootstrap p in both
             directions with Holm over the family, combined verdict ahead / behind / not
             distinguishable. Families (p33.set_rows): ind the eight combinations (the family of
             FamRule), setup and asym S, LS, SW and LSW, the combinations a change of the setups
             reaches; on N, L, W and LW the rows of setup and asym are those of ind.
  pooled     per shifted set, d of S, LS, SW and LSW (the arrays of the verdicts above) pooled by
             p33.pooled_rows (p88.pooled_block: the crossed hierarchical bootstrap over training
             seeds and test instances, a drawn instance bringing all its combinations, B = 20000
             from a fresh default_rng(11)), with the direction of the interval (ahead, behind,
             neither)
  CHECK      the rows of ind and setup equal p33's record of the same setting (results/ood.json of
             step ood: mean, interval ends, Holm-adjusted p-values, verdict, family, exact), and
             p33 checks its rows of ind against FamRule<R>; the pooled row of setup equals the
             pooled row of p33's record (mean, interval ends, seed-level t interval, direction,
             combinations, exact); refusal on any difference.
  shift      mean setup entry of the S descriptors of the set over that of 10x25, and whether
             the matrix of instance_000 is asymmetric (p33.shift_magnitudes on the set's folder)
  infeasible schedules the validator rejected, from the summaries (p33.infeasible), per set over
             the eight combinations and the seeds
  and the classification of the treated combinations: the control (on N, L, W and LW the arrays
  of the policy, the specialists and the rules equal those of ind bit for bit; asserted), the
  placebo drift (exactly zero), the treated drift lead(set) - lead(ind) of the means on S, LS, SW
  and LSW, the cells whose mean difference rose (lead shrank) or did not, tied (combined verdict
  ahead in distribution and not ahead on the set), reversed (mean below zero in distribution,
  not below now).

THE POLICY AGAINST THE SPECIALISTS (p88 functions, the procedure of its section H)
  d = 100 (policy - specialist) / specialist, seed s against seed s, per combination
  (p88.component on fresh draws: hierarchical 95% interval, verdicts at 0.5% and 1%, direction)
  and pooled over the setup-bearing combinations and over all eight (p88.pooled_block), for
  setup, asym and ind; B and alpha of p88's defaults (20000, 0.05).

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_asym.tex   macro_io.emit; names Asy<quantity> (the asym set) and AsyRef<quantity>
                              (the same quantity on 10x25_oodsetup), letters only, checked
                              against every macro defined in a .tex file under manuscript/
  results/asym_setups.json    every number of the three sets
  results/asym_setups.md      the asym set with the 10x25_oodsetup values beside it
  p33's outputs are read, not touched. A refusal removes all three.

CHECK (--check, read only)
  exit 0 when every array and summary the run reads exists, every array with one row per test
  instance: the policy on the eight combinations, each specialist on its own, the four rules, in
  the three folders; the policy's validator summaries; p33's record results/ood.json.

DRY MODE (--dry [DIR])
  The whole path on synthetic arrays in DIR (default a temporary folder, removed at the end): the
  three sets with identical arrays on N, L, W and LW and known leads on S, LS, SW and LSW
  (DRY_LEAD), validator summaries with every schedule feasible; then p33's dry path in the same
  tree (its window set, a FamRule record from the synthetic arrays, p33.run), so that the check
  against p33's record runs; B = DRY_BOOT for both procedures.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p119_asym_setups.py
       ... python scripts/p119_asym_setups.py --check
       python scripts/p119_asym_setups.py --dry [DIR]
"""
import argparse
import contextlib
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
import p33_ood_table as P33  # noqa: E402
import p88_family_inference as F88  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import VERDICT_WORDS  # noqa: E402

GENERATOR = 'scripts/p119_asym_setups.py'
PREFIX = 'Asy'
OUT_FILES = {'macros': 'manuscript/macros_asym.tex', 'json': 'results/asym_setups.json',
             'md': 'results/asym_setups.md'}
ENV = 'exact'
# (name, results folder key, data folder): p33's order of the first two
SETS = (('ind', '10x25_cpuref', 'data/FAMILY/10x25'),
        ('setup', '10x25_oodsetup', 'data/FAMILY/10x25_oodsetup'),
        ('asym', '10x25_asymsetup', 'data/FAMILY/10x25_asymsetup'))
SHIFTED = ('setup', 'asym')
WORD = {'asym': PREFIX, 'setup': PREFIX + 'Ref'}
TEXT = {'ind': '10x25 (in distribution)', 'setup': '10x25_oodsetup (twice the magnitude, '
        'asymmetric)', 'asym': '10x25_asymsetup (trained magnitude, asymmetric)'}
SETUP_COMBOS = ('S', 'LS', 'SW', 'LSW')
# the combinations each set changes, its Holm family against the best dispatching rule
REACHED = {'ind': list(REG.COMBOS), 'setup': list(SETUP_COMBOS), 'asym': list(SETUP_COMBOS)}
P33_RECORD = P33.OUT_FILES['json']               # results/ood.json of step ood
P33_KEYS = P33.FAMILY_KEYS + ('holm_family', 'reached')
P33_POOLED_KEYS = ('mean', 'lo', 'hi', 't_lo', 't_hi', 'direction', 'c', 'combos')
DRY_BOOT = 400
R8 = list(REG.COMBOS)


class Refused(Exception):
    """A setting or an input that this script does not accept."""


# ----------------------------------------------------------------- setting and inputs
def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the analysis reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')
    return list(REG.seeds())


def n_test(data):
    return len(glob.glob(os.path.join(data, 'desc', 'N', '*.desc.json')))


summary_path = P33.summary_path     # the validator summary p33.infeasible reads for seed s


def inputs(seeds):
    """[(path, rows or None)] of every file the run reads."""
    out = []
    for _, key, data in SETS:
        res, n = REG.res_dir(key), n_test(data)
        for s in seeds:
            for r in R8:
                out.append((REG.result_path('released', None, s, key, r), n))
                out.append((REG.result_path('specialist', r, s, key, r), n))
            out.append((summary_path(key, s), None))
        for q in ('FIFO', 'MOR', 'SPT', 'MWKR'):
            for r in R8:
                out.append((f'{res}/Result_{q}_{r}.npy', n))
    return out


def input_problems(seeds):
    bad = []
    for _, _, data in SETS:
        if n_test(data) == 0:
            bad.append(f'{data}/desc/N holds no descriptor')
    for p, n in inputs(seeds):
        if not os.path.exists(p):
            bad.append(f'missing {p}')
        elif n is not None and len(REG.read_path(p)) != n:
            bad.append(f'{p}: {len(REG.read_path(p))} rows, the test set has {n}')
    if not os.path.exists(REG.out(P33_RECORD)):
        bad.append(f'missing {REG.out(P33_RECORD)} (the record of step ood, '
                   f'scripts/p33_ood_table.py)')
    return bad


# ----------------------------------------------------------------- analysis
def rule_block(seeds):
    """The policy against the best dispatching rule on the three sets by p33's procedure
    (p33.set_rows: p88.superiority_block per family) and the infeasible counts."""
    if P33.RULES != ['FIFO', 'MOR', 'SPT', 'MWKR'] or P33.R8 != R8:
        raise Refused('p33 names other rules or combinations than this script reads')
    try:
        lead, floor, _ = P33.set_rows(seeds, SETS, REACHED)
    except P33.Refused as e:
        raise Refused(str(e)) from None
    infeas = {}
    for name, key, _ in SETS:
        res, sfx = REG.res_dir(key), REG.summary_tag(key)
        for r in R8:
            infeas[(name, r)] = int(sum(P33.infeasible(res, REG.tag(P33.JOINT, None, s), r, sfx)
                                        for s in seeds))
    return lead, floor, infeas


def check_p33(lead, seeds, dry):
    """The rows of ind and setup against p33's record of the same setting, exactly; refuses on
    any difference. Returns the comparison record."""
    p = REG.out(P33_RECORD)
    if not os.path.exists(p):
        raise Refused(f'{p} is missing; run step ood (scripts/p33_ood_table.py) for this setting '
                      f'first')
    rec = json.load(open(p))
    want = dict(environment=ENV, seeds=list(seeds), boot=F88.B, alpha=F88.ALPHA, dry=bool(dry))
    got = {k: rec.get(k) for k in want}
    if got != want:
        raise Refused(f'{p} records {got}, this run uses {want}')
    bad, n = [], 0
    for name in ('ind', 'setup'):
        for r in R8:
            x, y = lead[(name, r)], (rec.get('rows') or {}).get(f'{name}|{r}')
            if y is None:
                bad.append(f'{name} {r}: no row in {p}')
                continue
            for k in P33_KEYS:
                n += 1
                if x[k] != y.get(k):
                    bad.append(f'{name} {r} {k}: {p} {y.get(k)!r}, here {x[k]!r}')
    if bad:
        raise Refused(f'the policy against the best dispatching rule differs from p33\'s record '
                      f'in {len(bad)} of {n} comparisons:\n  ' + '\n  '.join(bad[:12]))
    return dict(file=p, compared=n, identical=True, keys=list(P33_KEYS))


def rule_pooled(seeds):
    """The policy against the best dispatching rule pooled over S, LS, SW and LSW of each shifted
    set, by p33's routine (p33.pooled_rows: p88.pooled_block on the differences of the
    per-combination verdicts)."""
    try:
        return P33.pooled_rows(seeds, SETS, REACHED)
    except P33.Refused as e:
        raise Refused(str(e)) from None


def check_p33_pooled(pooled):
    """The pooled row of setup against the pooled row of p33's record, exactly; refuses on any
    difference. Returns the comparison record (the setting of the record is checked by
    check_p33)."""
    p = REG.out(P33_RECORD)
    y = (json.load(open(p)).get('pooled') or {}).get('setup')
    if y is None:
        raise Refused(f'{p} holds no pooled row of setup')
    x = pooled['setup']
    bad = [f'setup pooled {k}: {p} {y.get(k)!r}, here {x[k]!r}' for k in P33_POOLED_KEYS
           if x[k] != y.get(k)]
    if bad:
        raise Refused('the pooled difference from the best dispatching rule differs from p33\'s '
                      'record:\n  ' + '\n  '.join(bad))
    return dict(file=p, compared=len(P33_POOLED_KEYS), identical=True,
                keys=list(P33_POOLED_KEYS))


def control(seeds):
    """On the combinations without SETUP every shifted set holds the arrays of ind, bit for bit:
    the policy, the specialists and the four rules. Returns the number of arrays compared."""
    ind_key = SETS[0][1]
    n = 0
    for name, key, _ in SETS[1:]:
        for r in R8:
            if 'S' in r:
                continue
            pairs = [(REG.result_path(arm, c, s, ind_key, r), REG.result_path(arm, c, s, key, r))
                     for s in seeds for arm, c in (('released', None), ('specialist', r))]
            pairs += [(f'{REG.res_dir(ind_key)}/Result_{q}_{r}.npy',
                       f'{REG.res_dir(key)}/Result_{q}_{r}.npy')
                      for q in ('FIFO', 'MOR', 'SPT', 'MWKR')]
            for a, b in pairs:
                x, y = REG.read_path(a), REG.read_path(b)
                if not np.array_equal(x, y):
                    raise Refused(
                        f'CONTROL FAILED: {name}, combination {r} has no SETUP, so the shift '
                        f'cannot reach it, but {b} differs from {a} on '
                        f'{int((x != y).sum())} of {len(x)} instances. The sets were not '
                        f'evaluated by the same command; no number of this script would be the '
                        f'shift alone.')
                n += 1
    return n


def classify(lead, name):
    """The placebo and treated cells of one shifted set: drift of the means, the placebo drift
    (exactly zero, the rows there are those of ind), and the treated cells whose mean difference
    rose (hurt) or did not (helped), that are no longer ahead (tied: combined verdict ahead in
    distribution, not ahead on the set) and whose mean changed sign (reversed)."""
    placebo, treated = [], []
    for r in R8:
        a, b = lead[('ind', r)], lead[(name, r)]
        cell = dict(combo=r, ind=a['mean'], shifted=b['mean'], drift=b['mean'] - a['mean'],
                    ind_verdict=a['verdict'], verdict=b['verdict'])
        (treated if r in REACHED[name] else placebo).append(cell)
    floor = max(abs(c['drift']) for c in placebo) if placebo else 0.0
    if floor != 0.0:
        raise Refused(f'CONTROL FAILED: a placebo cell of {name} moved by {floor} points')
    ahead = VERDICT_WORDS[0]
    hurt = [c['combo'] for c in treated if c['drift'] > 0]
    helped = [c['combo'] for c in treated if not c['drift'] > 0]
    tied = [c['combo'] for c in treated if c['ind_verdict'] == ahead and c['verdict'] != ahead]
    reversed_ = [c['combo'] for c in treated if c['ind'] < 0 <= c['shifted']]
    verdicts = {w: [c['combo'] for c in treated if c['verdict'] == w] for w in VERDICT_WORDS}
    return dict(placebo=placebo, treated=treated, floor=floor, hurt=hurt, helped=helped,
                tied=tied, reversed=reversed_, verdicts=verdicts)


def specialist_block(seeds):
    """Policy against the specialist of each combination, per set (p88 section H procedure)."""
    out = {}
    fams = {'Setup': list(SETUP_COMBOS), 'All': R8}
    for name, key, data in SETS:
        F88.verify_bases(os.path.basename(data), n_test(data))
        D = {r: F88.seed_pairs(F88.released, F88.specialist, key, r, seeds=seeds) for r in R8}
        out[name] = dict(rows={r: F88.component(d, F88.fresh(d)) for r, d in D.items()},
                         pooled=F88.pooled_block(D, fams))
    return out


def analyse(dry=False, boot=None):
    seeds = check_setting()
    bad = input_problems(seeds)
    if bad:
        raise Refused(f'{len(bad)} inputs do not count; first: {bad[0]}')
    F88.configure(boot=boot if boot is not None else F88.SF.B_DEFAULT, alpha=0.05)
    n_ctl = control(seeds)
    lead, _, infeas = rule_block(seeds)
    p33_check = check_p33(lead, seeds, dry)
    pooled = rule_pooled(seeds)
    p33_pooled = check_p33_pooled(pooled)
    shift = {}
    for name, _, data in SETS[1:]:
        sm = P33.shift_magnitudes(oods=data)
        shift[name] = dict(setup_x=float(sm['setup_x']), asymmetric=bool(sm['asym']))
    cls = {name: classify(lead, name) for name in SHIFTED}
    spec = specialist_block(seeds)
    return dict(generator=GENERATOR, setting=REG.describe(), seeds=seeds, dry=bool(dry),
                rule=dict(source='scripts/p33_ood_table.py (differences, family_rows, set_rows: '
                                 'p88.seed_pairs and p88.superiority_block, the procedure of '
                                 'FamRule<R>)',
                          B=F88.B, alpha=F88.ALPHA,
                          stream=f'fresh default_rng({F88.FRESH_SEED}) per combination',
                          families={name: list(REACHED[name]) for name, _, _ in SETS},
                          p33_record=p33_check, p33_pooled=p33_pooled),
                p88=dict(B=F88.B, alpha=F88.ALPHA, stream='fresh default_rng(11) per quantity'),
                n={name: n_test(data) for name, _, data in SETS},
                folders={name: REG.res_dir(key) for name, key, _ in SETS},
                control_arrays=n_ctl, shift=shift,
                lead={f'{k[0]}|{k[1]}': v for k, v in lead.items()},
                lead_pooled=pooled,
                infeasible={name: int(sum(infeas[(name, r)] for r in R8)) for name, _, _ in SETS},
                classification=cls, specialist=spec,
                definitions=dict(
                    lead='d = 100 (policy_s - best) / best per training seed s and test instance, '
                         'best = the per-instance best of FIFO, MOR, SPT, MWKR; negative = the '
                         'policy is shorter; mean, hierarchical 95% interval and combined verdict '
                         '(ahead, behind, not distinguishable) with Holm over the family of the '
                         'set (p33, the procedure of FamRule<R>)',
                    drift='mean difference on the shifted set minus that in distribution; '
                          'positive = the lead shrank',
                    tied='combined verdict ahead in distribution (Holm over the eight '
                         'combinations) and not ahead on the shifted set (Holm over S, LS, SW, '
                         'LSW)',
                    reversed='mean difference below zero in distribution and not below zero on '
                             'the shifted set',
                    specialist='100 (policy - specialist) / specialist, seed s against seed s '
                               '(p88 component, pooled_block)'))


# ----------------------------------------------------------------- outputs
def defined_macros():
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


def setting_words(res):
    """The setting for a printed header: environment and training seeds."""
    return f'environment {ENV}, training seeds {",".join(map(str, res["seeds"]))}'


def macro_lines(res):
    f2 = F88.f2
    L = [f'% Generated by {GENERATOR} ({setting_words(res)}) -- do not hand-edit',
         '% Asy: asymmetric setup matrices at the trained magnitude (10x25_asymsetup); AsyRef: the '
         'same quantity on 10x25_oodsetup (twice the magnitude, asymmetric). Lead: the policy '
         'against the per-instance best of FIFO, MOR, SPT and MWKR, d = 100 (policy - best) / best '
         'per training seed and test instance, mean, hierarchical 95% interval and combined '
         'verdict (ahead, behind, not distinguishable) by the procedure of FamRule<R> '
         '(scripts/p33_ood_table.py, p88.superiority_block), Holm over S, LS, SW, LSW; on N, L, W, '
         'LW the values of the set inside the ranges (the arrays are identical, asserted); the '
         'AsyRef values equal those of macros_ood.tex (checked). Spec: the policy against the '
         'specialist, seed s against seed s, p88 procedure (hierarchical 95% CI, direction, '
         'verdicts at 0.5 and 1); SpecPoolSetup over S, LS, SW, LSW.']
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    L.append(f'\\newcommand{{\\{PREFIX}N}}{{{res["n"]["asym"]}}}'
             f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(res["seeds"])}}}'
             '  % test instances per combination, training seeds')
    for name in ('asym', 'setup'):
        w, c, sh = WORD[name], res['classification'][name], res['shift'][name]
        L.append(f'\\newcommand{{\\{w}SetupX}}{{{sh["setup_x"]:.2f}}}'
                 f'  % {TEXT[name]}: mean setup entry over that of 10x25 (matrix of instance_000 '
                 f'{"asymmetric" if sh["asymmetric"] else "symmetric"})')
        L.append(f'\\newcommand{{\\{w}Infeas}}{{{res["infeasible"][name]}}}'
                 f'  % {TEXT[name]}: schedules the validator rejected, eight combinations, all seeds')
        L.append(f'\\newcommand{{\\{w}Floor}}{{{c["floor"]:.2f}}}'
                 f'\\newcommand{{\\{w}Placebo}}{{{len(c["placebo"])}}}'
                 f'\\newcommand{{\\{w}Treated}}{{{len(c["treated"])}}}'
                 f'  % {TEXT[name]}: largest change of the mean difference from the best '
                 f'dispatching rule on a combination without SETUP (points; zero, the arrays there '
                 f'are those in distribution, asserted), combinations without and with SETUP')
        L.append(f'\\newcommand{{\\{w}Hurt}}{{{len(c["hurt"])}}}'
                 f'\\newcommand{{\\{w}Helped}}{{{len(c["helped"])}}}'
                 f'\\newcommand{{\\{w}Tied}}{{{len(c["tied"])}}}'
                 f'\\newcommand{{\\{w}Reversed}}{{{len(c["reversed"])}}}'
                 f'  % {TEXT[name]}: combinations with SETUP whose mean difference from the best '
                 f'dispatching rule rose (lead shrank) / did not rise, whose combined verdict was '
                 f'ahead in distribution and is not ahead here '
                 f'({", ".join(c["tied"]) or "none"}), whose mean changed from below zero to not '
                 f'below zero ({", ".join(c["reversed"]) or "none"})')
        for r in R8:
            x, a = res['lead'][f'{name}|{r}'], res['lead'][f'ind|{r}']
            what = (f'{TEXT[name]}, {r}: policy vs best dispatching rule (%), hierarchical 95% CI '
                    f'and combined verdict, Holm over {", ".join(x["holm_family"])}; '
                    f'{F88.holm_text(x)} (in distribution {a["mean"]:+.2f}, {a["verdict"]})')
            if not x['reached']:
                what = (f'{TEXT[name]}, {r}: no SETUP, the arrays are those in distribution, so '
                        f'these are the values and the verdict in distribution')
            L.append(f'\\newcommand{{\\{w}Lead{r}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\{w}Lead{r}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\{w}Lead{r}Hi}}{{{f2(x["hi"])}}}'
                     f'\\newcommand{{\\{w}Lead{r}Verdict}}{{{x["verdict"]}}}  % {what}')
        y = res['lead_pooled'][name]
        L.append(F88.row_line(
            f'{w}LeadPool', y,
            f'{TEXT[name]}: policy vs best dispatching rule (%), pooled over the {y["c"]} '
            f'combinations with SETUP ({", ".join(y["combos"])}), the differences of the '
            f'per-combination verdicts; mean, hierarchical 95% interval over training seeds and '
            f'test instances with all combinations of a drawn instance kept together, direction of '
            f'the interval (ahead, behind, neither)'
            + (' (equal to \\OodSetupPool of macros_ood.tex, checked)' if name == 'setup' else ''),
            margins=False))
        sp = res['specialist'][name]
        for r in R8:
            L.append(F88.row_line(f'{w}Spec{r}', sp['rows'][r],
                                  f'{TEXT[name]}, policy vs specialist, {r}'))
        for fam, y in sp['pooled'].items():
            L.append(F88.row_line(f'{w}SpecPool{fam}', y, f'{TEXT[name]}, policy vs specialist, '
                                  f'pooled over {y["c"]} combinations'))
    return L


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    clash = sorted(x for x in names if x in have)
    if clash:
        raise Refused('macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def _ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def markdown(res):
    md = [f'# Asymmetric setups at the trained magnitude ({setting_words(res)})', '',
          f'Sets: {TEXT["asym"]}, beside {TEXT["setup"]} and {TEXT["ind"]}; {res["n"]["asym"]} '
          f'test instances per combination. Arrays: '
          + ', '.join(f'`{v}`' for v in res['folders'].values()) + '.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          f'Control: {res["control_arrays"]} arrays on N, L, W and LW (policy, specialists, the four '
          f'dispatching rules) are bit-identical to the in-distribution arrays.', '',
          '| | 10x25_asymsetup | 10x25_oodsetup |', '|---|---|---|']
    for k, txt in (('setup_x', 'mean setup entry over that of 10x25'),):
        md.append(f'| {txt} | {res["shift"]["asym"][k]:.2f} | {res["shift"]["setup"][k]:.2f} |')
    md.append(f'| matrix of instance_000 asymmetric | {res["shift"]["asym"]["asymmetric"]} | '
              f'{res["shift"]["setup"]["asymmetric"]} |')
    md.append(f'| schedules rejected by the validator | {res["infeasible"]["asym"]} | '
              f'{res["infeasible"]["setup"]} |')
    md += ['', '## Policy against the best dispatching rule (the procedure of FamRule<R>)', '',
           f'{res["definitions"]["lead"]}. B = {res["rule"]["B"]}, {res["rule"]["stream"]}; Holm '
           f'families: in distribution the eight combinations, 10x25_asymsetup and 10x25_oodsetup '
           f'S, LS, SW, LSW; on N, L, W, LW the values in distribution (identical arrays). The '
           f'10x25_oodsetup and in-distribution rows equal p33\'s record '
           f'`{res["rule"]["p33_record"]["file"]}` ({res["rule"]["p33_record"]["compared"]} '
           f'comparisons, exact).', '',
           '| comb. | in distribution | verdict | 10x25_asymsetup | verdict | drift | '
           '10x25_oodsetup | verdict | drift |', '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        a, x, y = (res['lead'][f'{n}|{r}'] for n in ('ind', 'asym', 'setup'))
        md.append(f'| {r} | {_ci(a)} | {a["verdict"]} | {_ci(x)} | {x["verdict"]} | '
                  f'{x["mean"] - a["mean"]:+.2f} | {_ci(y)} | {y["verdict"]} | '
                  f'{y["mean"] - a["mean"]:+.2f} |')
    pa, ps = res['lead_pooled']['asym'], res['lead_pooled']['setup']
    md += ['', f'Pooled over {", ".join(pa["combos"])} (a drawn test instance brings all its '
           f'combinations): 10x25_asymsetup {_ci(pa)}, direction {pa["direction"]}; '
           f'10x25_oodsetup {_ci(ps)}, direction {ps["direction"]} (equal to the pooled row of '
           f'`{res["rule"]["p33_pooled"]["file"]}`, exact).']
    md += ['', '| treated cells (S, LS, SW, LSW) | 10x25_asymsetup | 10x25_oodsetup |',
           '|---|---|---|']
    for k, txt in (('hurt', 'lead shrank'), ('helped', 'lead grew or unchanged'),
                   ('tied', 'no longer ahead (verdict)'), ('reversed', 'mean changed sign')):
        md.append(f'| {txt} | {", ".join(res["classification"]["asym"][k]) or "none"} | '
                  f'{", ".join(res["classification"]["setup"][k]) or "none"} |')
    for w in VERDICT_WORDS:
        md.append(f'| verdict {w} | '
                  f'{", ".join(res["classification"]["asym"]["verdicts"][w]) or "none"} | '
                  f'{", ".join(res["classification"]["setup"]["verdicts"][w]) or "none"} |')
    md.append(f'| largest placebo drift | {res["classification"]["asym"]["floor"]:.2f} | '
              f'{res["classification"]["setup"]["floor"]:.2f} |')
    md += ['', '## Policy against the specialist of the combination', '',
           f'{res["definitions"]["specialist"]}; B = {res["p88"]["B"]}.', '',
           '| comb. | in distribution | 0.5% | 1% | 10x25_asymsetup | 0.5% | 1% | 10x25_oodsetup | '
           '0.5% | 1% |', '|---|---|---|---|---|---|---|---|---|---|']
    sp = res['specialist']
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(
            f'{_ci(sp[n]["rows"][r])} | {sp[n]["rows"][r]["v0.5"]} | {sp[n]["rows"][r]["v1.0"]}'
            for n in ('ind', 'asym', 'setup')) + ' |')
    for fam in sp['asym']['pooled']:
        md.append(f'| pooled {fam} | ' + ' | '.join(
            f'{_ci(sp[n]["pooled"][fam])} | {sp[n]["pooled"][fam]["v0.5"]} | '
            f'{sp[n]["pooled"][fam]["v1.0"]}' for n in ('ind', 'asym', 'setup')) + ' |')
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(dry=False, boot=None):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md)."""
    check_setting()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(dry, boot)
        lines = macro_lines(res)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
        md = markdown(res)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md


def check():
    """Read-only precondition of the chain step; returns the exit code."""
    seeds = check_setting()
    bad = input_problems(seeds)
    print(f'p119: arrays and summaries of 10x25, 10x25_oodsetup and 10x25_asymsetup, seeds '
          f'{",".join(map(str, seeds))}: '
          + ('every one present, every array with one row per test instance' if not bad else
             f'{len(bad)} do not count'))
    for x in bad[:6]:
        print(f'p119:   {x}')
    if len(bad) > 6:
        print(f'p119:   ... and {len(bad) - 6} more')
    return 0 if not bad else 1


# ----------------------------------------------------------------- dry mode
# true leads in % against the best rule (ind, setup, asym) on the setup-bearing combinations,
# and the true gap of the policy against the specialist on them (ind, setup, asym)
DRY_LEAD = {'ind': -3.0, 'setup': -2.0, 'asym': -2.6}
DRY_SPEC = {'ind': 0.2, 'setup': 1.0, 'asym': 0.5}
DRY_OTHER_LEAD, DRY_OTHER_SPEC = -4.0, 0.1


@contextlib.contextmanager
def environment(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def dry_paths(tree):
    return dict(PACT_ENV=ENV, PACT_SEEDS='301-305', PACT_RESULTS=os.path.join(tree, 'results'),
                PACT_OUT=os.path.join(tree, 'out'))


def build_synthetic(tree):
    """Arrays and validator summaries of the three sets under <tree>/results. Call with
    dry_paths(tree) in the environment."""
    seeds = check_setting()
    rng = np.random.default_rng(119)
    n = n_test(SETS[0][2])
    if not all(n_test(d) == n for _, _, d in SETS):
        raise Refused('the three sets hold different numbers of test instances')
    base = {r: rng.uniform(150.0, 350.0, n) for r in R8}
    noise = {(r, s): rng.normal(0, 0.2, n) for r in R8 for s in seeds}
    for name, key, _ in SETS:
        res = REG.res_dir(key)
        os.makedirs(res, exist_ok=True)
        for r in R8:
            treated = 'S' in r
            g_spec = DRY_SPEC[name] if treated else DRY_OTHER_SPEC
            g_lead = DRY_LEAD[name] if treated else DRY_OTHER_LEAD
            scale = 1.0 + (0.3 if (treated and name == 'setup') else 0.0)
            spec = base[r] * scale
            pols = []
            for s in seeds:
                pol = spec * (1 + (g_spec + noise[(r, s)]) / 100.0)
                pols.append(pol)
                for p, x in ((REG.result_path('released', None, s, key, r), pol),
                             (REG.result_path('specialist', r, s, key, r), spec)):
                    np.save(p, np.stack([x, np.zeros(n)], axis=1))
            best = np.mean(pols, axis=0) / (1 + g_lead / 100.0)
            for q, extra in zip(('FIFO', 'MOR', 'SPT', 'MWKR'), (2.0, 1.0, 0.0, 3.0)):
                np.save(f'{res}/Result_{q}_{r}.npy',
                        np.stack([best * (1 + extra / 100.0), np.zeros(n)], axis=1))
        for s in seeds:
            t = REG.tag('released', None, s)
            rows = [f'| {r} | {t} | 0.0 | 0.0 | {n}/{n} | n/a | n/a |' for r in R8]
            with open(summary_path(key, s), 'w') as f:
                f.write(f'# FAMILY eval -- {t}\n\n| regime | method | mean | std | feasible | '
                        f'gap(cov) | gap to best known (N only) |\n|---|---|---|---|---|---|---|\n'
                        + '\n'.join(rows) + '\n')


def dry_run(tree, boot=DRY_BOOT):
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with environment(**dry_paths(tree)):
        build_synthetic(tree)
        try:                         # p33's dry path in the same tree: its record for the check
            P33.build_synthetic(tree, names_=('window',))
            P33.write_family_record(boot=boot)
            P33.run(dry=True, boot=boot)
        except P33.Refused as e:
            raise Refused(f'the dry path of p33: {e}') from None
        return run(dry=True, boot=boot)


def dry_ok(res):
    """The synthetic leads and gaps are recovered within 0.1 points, every combination is ahead
    of the best dispatching rule, the classification follows from them, and the pooled leads
    recover the lead of S, LS, SW and LSW with an interval below zero."""
    ok = all(abs(res['lead'][f'{n}|{r}']['mean'] - (DRY_LEAD[n] if 'S' in r else DRY_OTHER_LEAD))
             < 0.1 for n in DRY_LEAD for r in R8)
    ok &= all(res['lead'][f'{n}|{r}']['verdict'] == VERDICT_WORDS[0] for n in DRY_LEAD for r in R8)
    ok &= all(abs(res['specialist'][n]['pooled']['Setup']['mean'] - DRY_SPEC[n]) < 0.1
              for n in DRY_SPEC)
    ok &= res['classification']['asym']['hurt'] == list(SETUP_COMBOS)
    ok &= not res['classification']['asym']['tied']
    ok &= res['classification']['asym']['floor'] == 0.0 and res['infeasible']['asym'] == 0
    ok &= res['rule']['p33_record']['identical'] is True
    for n in ('asym', 'setup'):         # the pooled row: the lead of S, LS, SW, LSW
        x = res['lead_pooled'][n]
        ok &= abs(x['mean'] - DRY_LEAD[n]) < 0.1 and x['direction'] == 'ahead'
        ok &= x['combos'] == list(SETUP_COMBOS)
    ok &= res['rule']['p33_pooled']['identical'] is True
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description='asymmetric setups at the trained magnitude')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    a = ap.parse_args()
    os.chdir(ROOT)
    try:
        if a.check:
            sys.exit(check())
        if a.dry is None:
            REG.banner()
            res, names, md = run()
            print('\n'.join(md))
            print(f'\n{len(names)} macros; wrote ' + ', '.join(outputs().values()))
            return
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p119_dry_')
        try:
            res, names, md = dry_run(tree)
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    ok = dry_ok(res)
    if keep:
        print('\n'.join(md))
    print(f'p119 dry run on synthetic arrays {"passed" if ok else "FAILED"}: {len(names)} macros, '
          f'{res["control_arrays"]} control arrays bit-identical; lead on S '
          f'{res["lead"]["asym|S"]["mean"]:+.3f} (true {DRY_LEAD["asym"]:+.2f}); policy vs '
          f'specialist over S, LS, SW, LSW {res["specialist"]["asym"]["pooled"]["Setup"]["mean"]:+.3f} '
          f'(true {DRY_SPEC["asym"]:+.2f})')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
