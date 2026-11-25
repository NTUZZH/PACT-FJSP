r"""P110: the principal comparison with ten training seeds, 301 to 310.

THE EXTENSION RULE (scripts/seed_extension.py)
  If, with five seeds, the interval of any single combination of the principal comparison
  (policy against specialist at 10x25) is inconclusive at the 1% margin, seeds 306 to 310
  of the policy and the eight specialists are trained and every verdict is recomputed with
  ten seeds. The verdict rule does not change.

WHAT IS COMPUTED
  Section A of scripts/p88_family_inference.py with the ten seeds: per combination
  d = 100 (policy - specialist) / specialist, seed s against seed s, the crossed
  hierarchical bootstrap, the verdicts at the 0.5% and 1% margins and the direction
  (p88.parity_components, p88.component, stats_family.verdict); the joint verdicts by the
  intersection-union test over the families All, Train and Held (p88.iut_block); the pooled
  rows of each family, hierarchical (p88.pooled_block) and instance-level on the seed-pooled
  makespans (stats_family.instance_boot, the estimator of section A of p88). Nothing is
  re-implemented: p88's functions run with the loader of scripts/seed_extension.py, which
  returns arm_registry.load for a first seed and reads the array of a second seed with the
  same checks (p88's module-level loader name, load, is exchanged for the call and restored).
  Random streams: B and alpha of the five-seed record; the ten-seed draws come from
  default_rng(TEN_SEED) (the one parity stream in p88's order N, L, S, W, LS, LW, SW, LSW, a
  fresh generator for each pooled hierarchical row and for each instance-level row), a fixed
  seed of this script, so they do not reuse p88's streams.

THE READING AT THE LEVEL OF THE TWO READINGS
  The comparison has two readings: five seeds, then ten seeds when the rule applies. The
  ten-seed section A is computed a second time at alpha / 2 with the same random streams, so
  the draws and the means are the same and only the percentiles differ. Intervals at level
  1 - alpha/2 (97.5% at alpha = 0.05) bound the one-sided error of a verdict over both
  readings by alpha/2 (union bound), the one-sided level of a single reading. A mean that
  differs between the two computations, or an adjusted interval that does not contain the
  interval at alpha, refuses. Every number at alpha stays as it is.

PRECONDITIONS (checked before anything is computed; any failure refuses, no output stays)
  1. the extension rule applies in the five-seed record (seed_extension.triggered);
  2. all 45 runs of the second five seeds are complete (seed_extension.state: trained and
     evaluated);
  3. the MWKR validation norms of the ten policy runs are equal (one validation scale);
  4. the five-seed section A recomputed here from the arrays with p88's own estimator and
     random streams (default_rng(11) parity stream, fresh default_rng(11), default_rng(7))
     reproduces every number of section A stored in the record to the printed precision
     (two decimals; verdict words, directions and joint verdicts exactly).

OUTPUT (arm_registry.out under PACT_OUT, or under the folder of --out)
  manuscript/macros_seedext.tex  macro_io.emit; every name starts with SeedExt, holds letters
                                only, and is checked against every macro name defined in
                                manuscript/*.tex and manuscript/sections_new/*.tex; the names of
                                the adjusted reading start with SeedExtAdj and follow the others
  results/seed_extension.json   per combination the ten-seed component (mean, hierarchical
                                95% interval, Bonferroni and seed-t intervals, verdicts at
                                0.5% and 1%, direction, the ten per-seed means), the five-seed
                                component as stored (read, not recomputed), the mean of the
                                first five and of the second five seeds; joint verdicts and
                                pooled rows of both; under the key adjusted the reading at
                                1 - alpha/2 (level, number of readings, per combination and
                                pooled row mean, interval and verdicts at both margins, the
                                combinations whose verdict differs between the two levels)
  results/seed_extension.md     the rule, the combinations that triggered it, the tables, that
                                the ten-seed intervals are not adjusted for the look at five
                                seeds, and the table of the adjusted reading after the others

DRY MODE (--dry [DIR])
  The whole path on synthetic arrays: a tree in DIR (default a temporary folder, removed at
  the end) with ten seeds of arrays (PACT_RESULTS=<tree>/results), training records of the 45
  runs and of their seed-301 counterparts (configuration, log, validation record, an empty
  checkpoint), and a five-seed record written by p88's section A functions on the synthetic
  arrays of the first five seeds (PACT_OUT=<tree>/out). The synthetic gaps are TRUE_GAP per
  combination plus a seed effect that averages to zero over the ten seeds plus instance
  noise, with a wide seed effect on W, so that the rule applies. Nothing is read from the
  result or training folders of the repository; the descriptors of data/FAMILY/10x25 give
  the test set and its base instances. Without DIR one line reports the outcome.

Usage: python scripts/p110_seed_extension.py              (PACT_ENV=exact PACT_SEEDS=301-305
                                                           PACT_OUT=<root>, as the chain sets)
       python scripts/p110_seed_extension.py --out DIR    (the same, outputs under DIR, a
                                                           folder outside the repository)
       python scripts/p110_seed_extension.py --dry [DIR] [--boot 2000]
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
import p88_family_inference as F88  # noqa: E402
import seed_extension as SE  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

TEN_SEED = 110
PREFIX = 'SeedExt'
OUT_FILES = {'macros': 'manuscript/macros_seedext.tex', 'json': 'results/seed_extension.json',
             'md': 'results/seed_extension.md'}
DRY_BOOT = 2000
RULE_TEXT = ('If, with five seeds, the interval of any single combination of the principal '
             'comparison (policy against specialist at 10x25) is inconclusive at the 1% margin, '
             'seeds 306 to 310 of the policy and the eight specialists are trained and every '
             'verdict is recomputed with ten seeds. The verdict rule does not change.')
NOT_ADJUSTED = ('The ten-seed intervals are not adjusted for the look at the five-seed '
                'results that decided the extension.')
READINGS = 2                  # the comparison is read with five seeds, then with ten seeds
ADJ = PREFIX + 'Adj'          # macro prefix of the reading at level 1 - alpha / READINGS
R8 = list(REG.COMBOS)
FAM = F88.PAR_FAM
MARGINS = F88.MARGINS
MW = F88.MW
Refused = SE.Refused


# ----------------------------------------------------------------- p88 with another loader
@contextlib.contextmanager
def p88_setting(loader=None, stream_seed=None, fresh_seed=None, alpha=None):
    """p88's module-level loader, random seeds and level ALPHA exchanged for the duration of a
    call of its functions, and restored afterwards; None keeps p88's own."""
    new = {'load': loader, 'STREAM_SEED': stream_seed, 'FRESH_SEED': fresh_seed, 'ALPHA': alpha}
    old = {}
    try:
        for k, v in new.items():
            if v is None:
                continue
            if not hasattr(F88, k):
                raise Refused(f'p88 has no {k}; the ten-seed computation cannot use its functions')
            old[k] = getattr(F88, k)
            setattr(F88, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(F88, k, v)


def section_a(seeds, loader, rng_seed=None, alpha=None):
    """Section A of p88 on the given seeds: ({R: d}, {R: component}, IUT block, pooled block,
    instance-level pooled block). rng_seed None: p88's own streams (STREAM_SEED, FRESH_SEED,
    INST_SEED); otherwise rng_seed for all three. alpha None: p88's ALPHA; otherwise the level
    1 - alpha of every interval and verdict (the draws do not depend on it)."""
    seeds = list(seeds)
    with p88_setting(loader, rng_seed, rng_seed, alpha):
        D, C = F88.parity_components(SE.FOLDER, seeds=seeds)
        iut = F88.iut_block(C, FAM)
        pool = F88.pooled_block(D, FAM)
    inst_seed = F88.INST_SEED if rng_seed is None else rng_seed
    inst_alpha = F88.ALPHA if alpha is None else alpha
    read = loader or F88.load

    def pooled(arm, combo, r):          # arm_registry.pooled: np.mean over the seeds in order
        return np.mean([read(arm, combo, s, SE.FOLDER, r) for s in seeds], axis=0)
    pinst = {}
    for fam, regs in FAM.items():
        cells = np.stack([F88.rel(pooled('released', None, r), pooled('specialist', r, r))
                          for r in regs], axis=1)
        x = SF.instance_boot(cells, B=F88.B, seed=inst_seed, alpha=inst_alpha)
        x['direction'] = SF.direction(x['lo'], x['hi'])
        pinst[fam] = x
    return D, C, iut, pool, pinst


# ----------------------------------------------------------------- checks
def f2(x):
    return F88.f2(x)


def reproduction(rec, C, iut, pool, pinst):
    """Numbers of the recomputed five-seed section A that differ from the record at the
    printed precision (two decimals; words exactly). Returns (n compared, [differences],
    largest absolute difference of a number)."""
    st = rec['parity']
    bad, n, dmax = [], 0, 0.0

    def num(where, a, b):
        nonlocal n, dmax
        n += 1
        dmax = max(dmax, abs(float(a) - float(b)))
        if f2(a) != f2(b):
            bad.append(f'{where}: stored {f2(b)}, recomputed {f2(a)}')

    def word(where, a, b):
        nonlocal n
        n += 1
        if a != b:
            bad.append(f'{where}: stored {b!r}, recomputed {a!r}')
    for r in R8:
        s, c = st['components'][r], C[r]
        for k in ('mean', 'lo', 'hi', 'sim_lo', 'sim_hi', 't_lo', 't_hi'):
            num(f'{r} {k}', c[k], s[k])
        if len(s['seed_means']) != len(c['seed_means']):
            bad.append(f'{r}: {len(s["seed_means"])} stored seed means, {len(c["seed_means"])} here')
        for i, (a, b) in enumerate(zip(c['seed_means'], s['seed_means'])):
            num(f'{r} seed mean {i + 1}', a, b)
        for mg in MARGINS:
            for sfx in ('', '_sim', '_t'):
                word(f'{r} v{mg}{sfx}', c[f'v{mg}{sfx}'], s[f'v{mg}{sfx}'])
        word(f'{r} direction', c['direction'], s['direction'])
    for k, x in iut.items():
        y = st['iut'][k]
        word(f'IUT {k} holds', bool(x['holds']), bool(y['holds']))
        word(f'IUT {k} failing', list(x['failing']), list(y['failing']))
    for fam in FAM:
        x, y = pool[fam], st['pooled'][fam]
        for k in ('mean', 'lo', 'hi', 't_lo', 't_hi'):
            num(f'pooled {fam} {k}', x[k], y[k])
        for k in ('v0.5', 'v1.0', 'direction'):
            word(f'pooled {fam} {k}', x[k], y[k])
        x, y = pinst[fam], st['pooled_instance_level'][fam]
        for k in ('mean', 'lo', 'hi'):
            num(f'pooled instance-level {fam} {k}', x[k], y[k])
        word(f'pooled instance-level {fam} direction', x['direction'], y['direction'])
    return n, bad, dmax


def level(alpha):
    """The level 100 (1 - alpha) in percent as the outputs print it (95, 97.5)."""
    return f'{100 * (1 - alpha):g}'


def adjusted_rule(alpha):
    """The rule of the adjusted reading in one sentence, for a record level 1 - alpha."""
    a = alpha / READINGS
    return (f'The comparison has two readings (five seeds, then ten seeds), so intervals at '
            f'level 1 - alpha/{READINGS} = {level(a)}% from the same ten-seed draws bound the '
            f'one-sided error of a verdict over both readings by alpha/{READINGS} = {a:g} (union '
            f'bound), which is the one-sided level of a single {level(alpha)}% reading.')


def adjusted_reading(C, pool, pinst, Ca, poola, pinsta, alpha):
    """The reading of the ten seeds at level 1 - alpha / READINGS. C, pool and pinst are the
    ten-seed section A at alpha; Ca, poola and pinsta the same section at alpha / READINGS with
    the same random streams. Refuses when a mean differs between the two or an adjusted
    interval does not contain the interval at alpha (the draws are then not the same). Returns
    the level, the number of readings, per combination and pooled row the mean, the interval
    and the verdicts at both margins, the combinations whose verdict differs between the two
    levels, and the number of combinations equivalent at each margin."""
    bad = []
    rows = ([(r, C[r], Ca[r]) for r in R8] + [(f'pooled {f}', pool[f], poola[f]) for f in FAM]
            + [(f'pooled instance-level {f}', pinst[f], pinsta[f]) for f in FAM])
    for where, x, y in rows:
        if float(x['mean']) != float(y['mean']):
            bad.append(f'{where}: mean {x["mean"]!r} at alpha = {alpha:g}, {y["mean"]!r} at the '
                       f'adjusted level')
        if not (y['lo'] <= x['lo'] and x['hi'] <= y['hi']):
            bad.append(f'{where}: the adjusted interval [{y["lo"]!r}, {y["hi"]!r}] does not '
                       f'contain [{x["lo"]!r}, {x["hi"]!r}]')
    if bad:
        raise Refused(f'the ten-seed section A at the adjusted level does not use the same draws '
                      f'({len(bad)} differences): ' + '; '.join(bad[:6]))
    a = alpha / READINGS

    def row(y):
        return dict(mean=y['mean'], lo=y['lo'], hi=y['hi'],
                    **{f'v{mg}': y[f'v{mg}'] for mg in MARGINS})
    return dict(readings=READINGS, alpha=a, level=100 * (1 - a), record_alpha=alpha,
                record_level=100 * (1 - alpha), rule=adjusted_rule(alpha),
                components={r: row(Ca[r]) for r in R8}, pooled={f: row(poola[f]) for f in FAM},
                changed={f'v{mg}': [r for r in R8 if Ca[r][f'v{mg}'] != C[r][f'v{mg}']]
                         for mg in MARGINS},
                equivalent_n={f'v{mg}': sum(Ca[r][f'v{mg}'] == 'equivalent' for r in R8)
                              for mg in MARGINS})


def policy_norms(root):
    """MWKR validation norms of the ten policy runs; refuses unless all are equal."""
    norms = {}
    for s in SE.ALL_SEEDS:
        vp = os.path.join(root, SE.vali_log('released', None, s))
        lp = os.path.join(root, SE.train_log('released', None, s))
        try:
            norms[s] = REG.validation_record('released', None, s, paths=(vp, lp))[2]
        except (SystemExit, REG.MissingResult) as e:
            raise Refused(f'validation record of the policy run of seed {s}: {e}')
    first = norms[SE.ALL_SEEDS[0]]
    if not first:
        raise Refused(f'no MWKR norm line in the training log of seed {SE.ALL_SEEDS[0]}')
    diff = {s: v for s, v in norms.items() if v != first}
    if diff:
        raise Refused(f'validation norms of the policy runs differ from seed {SE.ALL_SEEDS[0]} '
                      f'{first}: {diff}')
    return first


def defined_macros():
    """{name: [files]} of every macro defined in manuscript/*.tex and
    manuscript/sections_new/*.tex, the output file itself left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for f in sorted(glob.glob(os.path.join(ROOT, 'manuscript', '*.tex'))
                    + glob.glob(os.path.join(ROOT, 'manuscript', 'sections_new', '*.tex'))):
        if os.path.basename(f) == own:
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for a, b in rx.findall(text):
            out.setdefault(a or b, []).append(os.path.relpath(f, ROOT))
    return out


# ----------------------------------------------------------------- macros, JSON, markdown
def tex(v):
    return SF.TEX[v]


def macro_lines(res):
    L = [f'% Generated by scripts/p110_seed_extension.py ({res["setting"]}) -- do not hand-edit',
         f'% Section A of p88 with ten training seeds, B = {res["boot"]}, alpha = {res["alpha"]}, '
         f'draws from default_rng({TEN_SEED}); the Five names repeat the five-seed record '
         '(read, not recomputed). Gaps in %: policy minus specialist over specialist.',
         f'% The ten-seed intervals are not adjusted for the look at five seeds.']
    L.append(f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(SE.ALL_SEEDS)}}}'
             f'\\newcommand{{\\{PREFIX}Boot}}{{{res["boot"]}}}'
             f'\\newcommand{{\\{PREFIX}RngSeed}}{{{TEN_SEED}}}'
             f'\\newcommand{{\\{PREFIX}TriggerN}}{{{len(res["triggered"])}}}'
             f'\\newcommand{{\\{PREFIX}Trigger}}{{{", ".join(res["triggered"])}}}'
             f'  % seeds, replicates, random seed, combinations that triggered the rule')
    for r in R8:
        c, s = res['ten'][r], res['five'][r]
        L.append(f'\\newcommand{{\\{PREFIX}{r}}}{{{f2(c["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Lo}}{{{f2(c["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Hi}}{{{f2(c["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}{r}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}{r}Dir}}{{{tex(c["direction"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}FirstMean}}{{{f2(res["first_five_mean"][r])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}SecondMean}}{{{f2(res["second_five_mean"][r])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}{r}Seed{SE.SEED_WORDS[sd]}}}{{{f2(m)}}}'
                           for sd, m in zip(SE.ALL_SEEDS, c['seed_means']))
                 + f'  % {r}, ten seeds: mean, hierarchical 95% CI, verdicts at 0.5 and 1, '
                 f'direction, mean of the first and of the second five seeds, per seed')
        L.append(f'\\newcommand{{\\{PREFIX}Five{r}}}{{{f2(s["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}Five{r}Lo}}{{{f2(s["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}Five{r}Hi}}{{{f2(s["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}Five{r}V{MW[mg]}}}{{{tex(s[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}Five{r}Dir}}{{{tex(s["direction"])}}}'
                 f'  % {r}, five seeds as stored')
    for key, blk in (('', res['iut_ten']), ('Five', res['iut_five'])):
        for fam in FAM:
            for mg in MARGINS:
                for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                    x = blk[f'{fam}|{mg}|{claim}']
                    L.append(f'\\newcommand{{\\{PREFIX}{key}Iut{fam}{MW[mg]}{cw}}}'
                             f'{{{int(x["holds"])}}}  % IUT, {"five" if key else "ten"} seeds, '
                             f'{claim} on {",".join(x["components"])} at +-{mg}%: failing '
                             f'{",".join(x["failing"]) or "none"}')
    for key, pool, pinst in (('', res['pool_ten'], res['pool_inst_ten']),
                             ('Five', res['pool_five'], res['pool_inst_five'])):
        for fam in FAM:
            x, y = pool[fam], pinst[fam]
            L.append(f'\\newcommand{{\\{PREFIX}{key}Pool{fam}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Hi}}{{{f2(x["hi"])}}}'
                     + ''.join(f'\\newcommand{{\\{PREFIX}{key}Pool{fam}V{MW[mg]}}}'
                               f'{{{tex(x[f"v{mg}"])}}}' for mg in MARGINS)
                     + f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Dir}}{{{tex(x["direction"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}PoolInst{fam}}}{{{f2(y["mean"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}PoolInst{fam}Lo}}{{{f2(y["lo"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}PoolInst{fam}Hi}}{{{f2(y["hi"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}PoolInst{fam}Dir}}{{{tex(y["direction"])}}}'
                     f'  % pooled over {fam} ({"five seeds, stored" if key else "ten seeds"}): '
                     f'hierarchical CI, verdicts, direction; seed-pooled instance-level CI')
    return L


def running_list(names):
    """Combinations as running text in math mode: '$N$', '$N$ and $W$', '$N$, $W$ and $SW$';
    'none' when there are none."""
    m = [f'${x}$' for x in names]
    if not m:
        return 'none'
    return m[0] if len(m) == 1 else ', '.join(m[:-1]) + ' and ' + m[-1]


def adjusted_macro_lines(res):
    """The macros of the adjusted reading (prefix SeedExtAdj), written after those of
    macro_lines; the means are the SeedExt<R> and SeedExtPool<F> macros of macro_lines."""
    A = res['adjusted']
    lv, lr = level(A['alpha']), level(A['record_alpha'])
    L = [f'% The same ten-seed draws read at level {lv}% = 1 - alpha/{A["readings"]} for the two '
         f'readings (five seeds, then ten seeds): the means are those of the SeedExt names above, '
         f'the intervals are wider.']
    L.append(f'\\newcommand{{\\{ADJ}Level}}{{{lv}}}'
             f'\\newcommand{{\\{ADJ}Readings}}{{{A["readings"]}}}'
             f'  % level of the adjusted reading in percent, number of readings')
    for r in R8:
        x = A['components'][r]
        L.append(f'\\newcommand{{\\{ADJ}{r}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{ADJ}{r}Hi}}{{{f2(x["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{ADJ}{r}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'  % {r}, ten seeds at {lv}%: hierarchical CI, verdicts at 0.5 and 1 '
                 f'(mean: {PREFIX}{r})')
    for fam in FAM:
        x = A['pooled'][fam]
        L.append(f'\\newcommand{{\\{ADJ}Pool{fam}Lo}}{{{f2(x["lo"])}}}'
                 f'\\newcommand{{\\{ADJ}Pool{fam}Hi}}{{{f2(x["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{ADJ}Pool{fam}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'  % pooled over {fam}, ten seeds at {lv}%: hierarchical CI, verdicts at 0.5 '
                 f'and 1 (mean: {PREFIX}Pool{fam})')
    L.append(f'\\newcommand{{\\{ADJ}EquivOneN}}{{{A["equivalent_n"]["v1.0"]}}}'
             f'  % combinations equivalent at the 1% margin at {lv}%')
    for mg in MARGINS:
        ch, w = A['changed'][f'v{mg}'], MW[mg]
        L.append(f'\\newcommand{{\\{ADJ}Change{w}N}}{{{len(ch)}}}'
                 f'\\newcommand{{\\{ADJ}Change{w}Combos}}{{{", ".join(ch) or "none"}}}'
                 f'\\newcommand{{\\{ADJ}Change{w}List}}{{{running_list(ch)}}}'
                 f'  % combinations whose verdict at the {mg:g}% margin differs between {lr}% '
                 f'and {lv}%: number, names, running text')
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
        raise Refused(f'macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x][0]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by {taken[0]} ({have[taken[0]][0]})')
    return names


def ci(x):
    return f'[{f2(x["lo"])}, {f2(x["hi"])}]'


def markdown(res):
    md = [f'# Principal comparison with ten training seeds ({res["setting"]})', '',
          f'The extension rule: {RULE_TEXT}', '',
          f'The rule applied in the five-seed record `{res["record"]}`: inconclusive at the 1% '
          f'margin with seeds 301 to 305: {", ".join(res["triggered"])}.', '',
          NOT_ADJUSTED, '',
          f'Estimator and verdicts: section A of p88 (hierarchical bootstrap over training '
          f'seeds and base instances, B = {res["boot"]}, verdicts at +-0.5% and +-1% from the '
          f'95% interval); ten-seed draws from default_rng({TEN_SEED}). Reproduction of the '
          f'five-seed section A from the arrays: {res["reproduction"]["compared"]} numbers and '
          f'words identical at the printed precision. Validation norms of the ten policy runs '
          f'equal.' + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          '| comb. | ten seeds: mean | 95% CI | 0.5% | 1% | direction | five seeds (stored): mean '
          '| 95% CI | 0.5% | 1% | direction | first five | second five |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        c, s = res['ten'][r], res['five'][r]
        md.append(f'| {r} | {f2(c["mean"])} | {ci(c)} | {c["v0.5"]} | {c["v1.0"]} | '
                  f'{c["direction"]} | {f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} | '
                  f'{s["direction"]} | {f2(res["first_five_mean"][r])} | '
                  f'{f2(res["second_five_mean"][r])} |')
    md += ['', 'Per-seed means, ten seeds (policy minus specialist over specialist, %):', '',
           '| comb. | ' + ' | '.join(f's{s}' for s in SE.ALL_SEEDS) + ' |',
           '|---|' + '---|' * len(SE.ALL_SEEDS)]
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(f2(m) for m in res['ten'][r]['seed_means']) + ' |')
    md += ['', 'Joint verdicts (intersection-union test):', '',
           '| family | margin | equivalent on all (ten / five) | non-inferior on all (ten / five) '
           '| failing, equivalent (ten / five) |', '|---|---|---|---|---|']
    for fam in FAM:
        for mg in MARGINS:
            t = {c: res['iut_ten'][f'{fam}|{mg}|{c}'] for c in ('equivalent', 'non-inferior')}
            v = {c: res['iut_five'][f'{fam}|{mg}|{c}'] for c in ('equivalent', 'non-inferior')}
            md.append(f'| {fam} | {mg} | {t["equivalent"]["holds"]} / {v["equivalent"]["holds"]} '
                      f'| {t["non-inferior"]["holds"]} / {v["non-inferior"]["holds"]} | '
                      f'{",".join(t["equivalent"]["failing"]) or "-"} / '
                      f'{",".join(v["equivalent"]["failing"]) or "-"} |')
    md += ['', 'Pooled rows (hierarchical; seed-pooled instance-level):', '',
           '| family | ten: mean | CI | 0.5% | 1% | instance-level CI | five: mean | CI | 0.5% '
           '| 1% | instance-level CI |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for fam in FAM:
        x, y = res['pool_ten'][fam], res['pool_inst_ten'][fam]
        u, w = res['pool_five'][fam], res['pool_inst_five'][fam]
        md.append(f'| {fam} | {f2(x["mean"])} | {ci(x)} | {x["v0.5"]} | {x["v1.0"]} | {ci(y)} | '
                  f'{f2(u["mean"])} | {ci(u)} | {u["v0.5"]} | {u["v1.0"]} | {ci(w)} |')
    return md


def adjusted_markdown(res):
    """The section of the adjusted reading, written after the lines of markdown()."""
    A = res['adjusted']
    lv, lr = level(A['alpha']), level(A['record_alpha'])
    md = ['', f'Ten seeds at the level of the two readings ({lv}%, the same draws as the {lr}% '
              f'intervals):', '',
          f'| comb. | mean | {lr}% CI | 1% | {lv}% CI | 1% | 0.5% |', '|---|---|---|---|---|---|---|']
    rows = ([(r, res['ten'][r], A['components'][r]) for r in R8]
            + [(f'pooled {f}', res['pool_ten'][f], A['pooled'][f]) for f in FAM])
    for name, x, y in rows:
        md.append(f'| {name} | {f2(x["mean"])} | {ci(x)} | {x["v1.0"]} | {ci(y)} | {y["v1.0"]} | '
                  f'{y["v0.5"]} |')
    md += ['', A['rule'], '',
           f'Combinations whose verdict differs between {lr}% and {lv}%: '
           f'{", ".join(A["changed"]["v1.0"]) or "none"} at the 1% margin and '
           f'{", ".join(A["changed"]["v0.5"]) or "none"} at the 0.5% margin.']
    return md


def jsonable(x):
    return F88.jsonable(x)


# ----------------------------------------------------------------- the analysis
def outputs(out_root=None):
    """The three output paths: arm_registry.out under PACT_OUT, or under out_root when given
    (the five-seed record is read under PACT_OUT either way)."""
    if out_root is None:
        return {k: REG.out(v) for k, v in OUT_FILES.items()}
    outs = {k: os.path.join(out_root, v) for k, v in OUT_FILES.items()}
    for p in outs.values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
    return outs


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def check_setting():
    """The exact environment with the active seeds 301 to 305 (the setting of the five-seed
    record; the ten seeds are passed to p88's functions explicitly), PACT_OUT given."""
    SE.require_exact()
    if REG.seeds() != SE.FIRST_SEEDS:
        raise Refused(f'run with PACT_SEEDS={SE.FIRST_SEEDS[0]}-{SE.FIRST_SEEDS[-1]} (the '
                      f'setting of the five-seed record); the ten seeds are passed explicitly')
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')


def shown(path):
    """A path as the outputs print it: relative to the repository when inside it."""
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def analyse(root='.', record=None, dry=False, boot=None):
    """Every precondition, then the ten-seed section A; returns the result dict. Raises
    Refused. root resolves the training records (default the repository)."""
    check_setting()
    rec_path = record or SE.record_path()
    rec = SE.read_record(rec_path)
    hit = SE.inconclusive(rec)
    if not hit:
        raise Refused(f'the extension rule does not apply in {rec_path}: no combination is '
                      f'inconclusive at the 1% margin with five seeds')
    st = SE.state(root)
    miss = [x for x in st if not x['complete']]
    if miss:
        x = miss[0]
        why = (x['training_problems'] + x['evaluation_problems'])[0]
        raise Refused(f'{len(miss)} of {len(st)} runs of the second five seeds are not complete '
                      f'(first: {x["name"]}: {why}); see scripts/seed_extension.py state')
    norms = policy_norms(root)
    F88.configure(boot=rec['boot'] if boot is None else boot, alpha=rec['alpha'])
    if int(rec['boot']) != F88.B:
        raise Refused(f'the record was computed with B = {rec["boot"]}, not {F88.B}')
    F88.verify_bases(SE.FOLDER, SE.n_test())
    # the five-seed section A, p88's own loader and streams
    _, C5, iut5, pool5, pinst5 = section_a(SE.FIRST_SEEDS, None, None)
    n_cmp, bad, dmax = reproduction(rec, C5, iut5, pool5, pinst5)
    if bad:
        raise Refused(f'the five-seed section A does not reproduce {rec_path} ({len(bad)} of '
                      f'{n_cmp}): ' + '; '.join(bad[:6]))
    # the ten seeds
    D10, C10, iut10, pool10, pinst10 = section_a(SE.ALL_SEEDS, SE.registry_load, TEN_SEED)
    k5 = len(SE.FIRST_SEEDS)
    first = {r: float(D10[r][:k5].mean()) for r in R8}
    second = {r: float(D10[r][k5:].mean()) for r in R8}
    for r in R8:
        if abs(first[r] - rec['parity']['components'][r]['mean']) > 1e-9:
            raise Refused(f'{r}: the first five seeds of the ten-seed arrays give {first[r]}, the '
                          f'record {rec["parity"]["components"][r]["mean"]}')
    # the same ten-seed draws read at the level of the two readings
    _, Ca, _, poola, pinsta = section_a(SE.ALL_SEEDS, SE.registry_load, TEN_SEED,
                                        alpha=F88.ALPHA / READINGS)
    adj = adjusted_reading(C10, pool10, pinst10, Ca, poola, pinsta, F88.ALPHA)
    st_rec = rec['parity']
    return dict(generator='scripts/p110_seed_extension.py',
                setting=f'environment {SE.ENV}, training seeds {SE.ALL_SEEDS[0]} to '
                        f'{SE.ALL_SEEDS[-1]} ({len(SE.ALL_SEEDS)} runs per arm), principal '
                        f'comparison only',
                seeds=list(SE.ALL_SEEDS), first_seeds=list(SE.FIRST_SEEDS),
                second_seeds=list(SE.SECOND_SEEDS), boot=F88.B, alpha=F88.ALPHA,
                streams=dict(ten=f'default_rng({TEN_SEED}): one parity stream in the order '
                                 f'{F88.TRAIN + F88.HELD}, fresh per pooled row',
                             five_reproduction=f'p88: default_rng({F88.STREAM_SEED}) stream, '
                                               f'fresh default_rng({F88.FRESH_SEED}), '
                                               f'default_rng({F88.INST_SEED})'),
                rule=RULE_TEXT, not_adjusted=NOT_ADJUSTED, record=shown(rec_path),
                record_setting=rec['setting'], triggered=hit, dry=bool(dry),
                trigger_rule=f'verdict {SE.VERDICT_KEY} of section A in {SE.TRIGGER_WORDS}',
                reproduction=dict(compared=n_cmp, identical=True, largest_abs_difference=dmax),
                validation_norms=norms,
                ten=C10, five={r: st_rec['components'][r] for r in R8},
                first_five_mean=first, second_five_mean=second,
                iut_ten=iut10, iut_five=st_rec['iut'], pool_ten=pool10,
                pool_five=st_rec['pooled'], pool_inst_ten=pinst10,
                pool_inst_five=st_rec['pooled_instance_level'],
                verdict_changes={r: dict(five=st_rec['components'][r]['v1.0'],
                                         ten=C10[r]['v1.0']) for r in R8},
                adjusted=adj)


def write(res, outs):
    lines = macro_lines(res) + adjusted_macro_lines(res)
    names = check_names(lines)
    emit(outs['macros'], lines)
    with open(outs['json'], 'w') as f:
        json.dump(jsonable(res), f, indent=1)
    md = markdown(res) + adjusted_markdown(res)
    with open(outs['md'], 'w') as f:
        f.write('\n'.join(md) + '\n')
    return names, md


def run(root='.', record=None, dry=False, boot=None, out_root=None):
    """analyse() and write(); a refusal removes the outputs. out_root: write the outputs under
    this folder in place of PACT_OUT. Returns (res, names, md)."""
    check_setting()
    outs = outputs(out_root)
    remove_outputs(outs)
    try:
        res = analyse(root, record, dry, boot)
        names, md = write(res, outs)
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md


# ----------------------------------------------------------------- dry mode
TRUE_GAP = {'N': 0.0, 'L': 0.1, 'S': -0.1, 'W': 0.4, 'LS': -0.1, 'LW': 0.2, 'SW': -0.4, 'LSW': 0.0}
# seed effects (in %, seeds 301 to 310), mean zero over the ten seeds; wide on W
SEED_EFFECT = {r: [0.10, -0.10, 0.05, -0.05, 0.02, -0.02, 0.08, -0.08, 0.04, -0.04]
               for r in R8}
SEED_EFFECT['W'] = [-1.2, 1.0, 1.1, -0.9, 0.6, 1.0, -1.1, -0.8, 1.2, -0.9]
NOISE = 0.3
NORMS = {'N': '156.6', 'L': '226.9', 'S': '235.3', 'W': '177.7', 'LS': '304.4', 'LW': '243.4',
         'SW': '210.0', 'LSW': '280.0'}


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


def _put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


def synthetic_records(tree, arm, combo, seed):
    """Checkpoint, training log, configuration and validation record of one run."""
    nm = SE.name(arm, combo, seed)
    regs = (F88.TRAIN if arm == 'released' else [combo])
    _put(os.path.join(tree, SE.checkpoint(arm, combo, seed)), '')
    _put(os.path.join(tree, SE.vali_log(arm, combo, seed)), '[0.9, 0.85, 0.86]')
    _put(os.path.join(tree, SE.train_log(arm, combo, seed)),
         '[train_family] MASK: first_mask=exact mask_ties=exact\n'
         '[vali] MWKR norms: ' + '  '.join(f'{r}={NORMS[r]}' for r in regs) + '\n'
         f'[model] params total={SE.PARAMS}  film=0 (0.00%)\n'
         '[train_family] done in 1.00 h; best rel=0.8500\n')
    _put(os.path.join(tree, SE.config_path(arm, combo, seed)),
         json.dumps(dict(model_suffix=nm, seed_train=int(seed), first_mask='exact',
                         mask_ties='exact', family_regimes=','.join(regs),
                         max_updates=2000 if arm == 'released' else 1000), indent=1))


def build_synthetic(tree, n=None, boot=DRY_BOOT):
    """A tree of synthetic arrays, records and a five-seed record (module docstring). Must
    be called with PACT_ENV=exact PACT_SEEDS=301-305 PACT_RESULTS=<tree>/results
    PACT_OUT=<tree>/out set."""
    SE.require_exact()
    n = SE.n_test() if n is None else n
    rng = np.random.default_rng(2026)
    base = {r: rng.uniform(400.0, 900.0, n) for r in R8}
    for k, s in enumerate(SE.ALL_SEEDS):
        for r in R8:
            spec = np.round(base[r] * (1 + rng.normal(0, 0.01, n)), 1)
            d = TRUE_GAP[r] + SEED_EFFECT[r][k] + rng.normal(0, NOISE, n)
            pol = spec * (1 + d / 100.0)
            for arm, combo, x in (('specialist', r, spec), ('released', None, pol)):
                p = SE.result_path(arm, combo, s, r)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                np.save(p, np.stack([x, np.zeros(n)], axis=1))
        for arm in SE.ARMS:
            for c in (R8 if arm == 'specialist' else [None]):
                synthetic_records(tree, arm, c, s)
    # the five-seed record of p88, section A, on the synthetic first five seeds
    F88.configure(boot=boot, alpha=0.05)
    _, C, iut, pool, pinst = section_a(SE.FIRST_SEEDS, None, None)
    rec = dict(generator=SE.GENERATOR, setting=REG.describe(), seeds=list(SE.FIRST_SEEDS),
               boot=F88.B, alpha=F88.ALPHA, simultaneous_family=F88.M_SIM,
               parity=dict(components=C, iut=iut, pooled=pool, pooled_instance_level=pinst))
    p = SE.record_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        json.dump(jsonable(rec), f, indent=1)
    return p


def dry_paths(tree):
    return dict(PACT_ENV=SE.ENV, PACT_SEEDS=f'{SE.FIRST_SEEDS[0]}-{SE.FIRST_SEEDS[-1]}',
                PACT_RESULTS=os.path.join(tree, 'results'), PACT_OUT=os.path.join(tree, 'out'))


def dry_run(tree, boot=DRY_BOOT):
    """The whole path on a synthetic tree; returns (res, names, md)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with environment(**dry_paths(tree)):
        build_synthetic(tree, boot=boot)
        return run(root=tree, dry=True, boot=boot)


def main():
    ap = argparse.ArgumentParser(description='the principal comparison with ten seeds')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='run the whole path on synthetic arrays in DIR (default: a temporary '
                         'folder, removed afterwards)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes B '
                         f'of the five-seed record')
    ap.add_argument('--out', default=None, metavar='DIR',
                    help='a real run that writes its three outputs under DIR, a folder outside '
                         'the repository, in place of PACT_OUT; the five-seed record is still '
                         'read under PACT_OUT')
    a = ap.parse_args()
    out_root = None
    if a.out is not None:
        if a.dry is not None:
            ap.error('--out is for a real run; a dry run writes in its own tree')
        out_root = os.path.abspath(a.out)
        if os.path.commonpath([out_root, ROOT]) == ROOT:
            ap.error(f'--out {out_root} lies inside the repository')
    os.chdir(ROOT)
    if a.dry is None:
        if a.boot is not None:
            ap.error('--boot is for the dry run; a real run uses B of the five-seed record')
        REG.banner()
        try:
            res, names, md = run(out_root=out_root)
        except Refused as e:
            sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
        print('\n'.join(md))
        print(f'\n{len(names)} macros; wrote ' + ', '.join(outputs(out_root).values()))
        return
    boot = DRY_BOOT if a.boot is None else a.boot
    keep = bool(a.dry)
    tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p110_dry_')
    try:
        res, names, md = dry_run(tree, boot)
    except Refused as e:
        sys.exit(f'p110 self-test on synthetic arrays FAILED: {e}')
    finally:
        if not keep:
            shutil.rmtree(tree, ignore_errors=True)
    if keep:
        print('\n'.join(md))
        print(f'\n{len(names)} macros; outputs in {os.path.join(tree, "out")}')
    else:
        print(f'p110 self-test on synthetic arrays passed: B = {boot}, {len(names)} macros, '
              f'{res["reproduction"]["compared"]} five-seed numbers reproduced, rule triggered by '
              f'{", ".join(res["triggered"])}, every interval and verdict computed for ten seeds')


if __name__ == '__main__':
    main()
