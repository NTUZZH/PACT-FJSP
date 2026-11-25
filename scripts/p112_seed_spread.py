r"""P112: the spread of the mean makespan across training seeds, policy and specialists.

WHAT IS COMPUTED (exact environment, factory benchmark, training size 10x25, greedy schedules)
  For a combination R and an arm (the policy; the specialist of R), m_s is the mean over the
  test instances of the makespan of training seed s under R, read from
  arm_registry.res_dir('10x25') (the first five seeds through the registry, the second five
  through scripts/seed_extension.py, registry_load). The per-seed value of the arm is
      v_s = 100 (m_s / mean_s(m_s) - 1),
  the deviation of the seed's mean makespan from the arm's own mean over the seeds, in percent.
  The spread of the arm on R is the sample standard deviation (ddof = 1) of v_s over the seeds;
  the smallest and the largest v_s are reported beside it. The spread of the comparison is the
  sample standard deviation of the per-seed mean differences of the policy against the
  specialist, d_s = mean over the instances of 100 (policy_s - specialist_s) / specialist_s,
  formed by section A of scripts/p88_family_inference.py (p88.seed_pairs, seed s against
  seed s).
  Paired test of equal variances (Pitman 1939, Morgan 1939): the two arms are paired by seed;
  with u_s = v_s(policy) + v_s(specialist) and w_s = v_s(policy) - v_s(specialist), the
  variances are equal exactly when u and w are uncorrelated, so the statistic is
      t = r sqrt((n - 2) / (1 - r^2)),  r the Pearson correlation of u and w over the n seeds,
  two-sided against Student t with n - 2 degrees of freedom; Holm's adjustment over the eight
  combinations of a seed set. Over the eight combinations: the root mean square of the spreads
  of each arm, sqrt(mean_R spread_R^2).
  The analysis is descriptive and was formed on the first five seeds; the second five seeds,
  trained under the extension rule of scripts/seed_extension.py, are a check on runs that
  played no part in forming it.

SEED SETS
  First   seeds 301 to 305, always;
  Second  seeds 306 to 310, and Ten, seeds 301 to 310: only when all 45 runs of the second five
          are complete (seed_extension.complete: trained and evaluated); otherwise both are left
          out, and the macro file, the JSON and the Markdown say so.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_seedspread.tex  macro_io.emit; names Sspr<set><quantity><R>, letters only,
                                   checked against every macro defined in a .tex file under
                                   manuscript/ (the output itself left out)
  results/seed_spread.json         every number per seed set and combination
  results/seed_spread.md           tables
  Spreads print with two decimals, per-seed extremes with two decimals and their sign
  (p88 f2), p-values with two significant digits.

CHECK (--check, read only)
  exit 0 when every array of the first five seeds exists with one row per test instance
  (the policy on the eight combinations, each specialist on its own). The files of the second
  five seeds are read by the run only, never by the check.

DRY MODE (--dry [DIR] [--dry-seeds 5|10])
  The whole path on synthetic arrays in DIR (default a temporary folder, removed at the end):
  arrays of the policy and the specialists with known per-seed deviations (DRY_EFFECT), and
  with --dry-seeds 10 the training records of the 45 runs of the second five seeds (those of
  scripts/p110_seed_extension.py's dry mode), so that the second five are complete. Nothing is
  read from the result or training folders of the repository; the descriptors of
  data/FAMILY/10x25 give the test set.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p112_seed_spread.py
       python scripts/p112_seed_spread.py --check
       python scripts/p112_seed_spread.py --dry [DIR] [--dry-seeds 5|10]
"""
import argparse
import contextlib
import glob
import json
import math
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

PREFIX = 'Sspr'
OUT_FILES = {'macros': 'manuscript/macros_seedspread.tex', 'json': 'results/seed_spread.json',
             'md': 'results/seed_spread.md'}
R8 = list(REG.COMBOS)
ALPHA = 0.05
SET_WORDS = (('First', 'seeds 301 to 305'), ('Second', 'seeds 306 to 310'),
             ('Ten', 'seeds 301 to 310'))
ARM_WORDS = (('Pol', 'policy'), ('Spec', 'specialist'))
Refused = SE.Refused


# ----------------------------------------------------------------- statistics
def spread(values):
    """(sample standard deviation with ddof 1, smallest, largest) of per-seed values."""
    x = np.asarray(values, dtype=float)
    return float(np.std(x, ddof=1)), float(x.min()), float(x.max())


def relative(means):
    """v_s = 100 (m_s / mean_s(m_s) - 1) of per-seed mean makespans m_s."""
    m = np.asarray(means, dtype=float)
    return 100.0 * (m / m.mean() - 1.0)


def pitman_morgan(x, y):
    """Paired test of equal variances of x and y (paired by index): r = corr(x + y, x - y),
    t = r sqrt((n - 2) / (1 - r^2)), two-sided p against Student t with n - 2 degrees of
    freedom. Returns dict(r, t, df, p); t is None and p 0 when |r| = 1."""
    from scipy.stats import t as student
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    n = len(x)
    if n < 3 or len(y) != n:
        raise Refused(f'the paired test of equal variances needs three or more pairs, got {n}')
    u, w = x + y, x - y
    if np.std(u) == 0 or np.std(w) == 0:
        raise Refused('the sums or the differences of the two arms do not vary')
    r = float(np.corrcoef(u, w)[0, 1])
    df = n - 2
    if 1.0 - r * r <= 0.0:
        return dict(r=r, t=None, df=df, p=0.0)
    t = r * math.sqrt(df / (1.0 - r * r))
    return dict(r=r, t=float(t), df=df, p=float(2.0 * student.sf(abs(t), df)))


def rms(values):
    return float(math.sqrt(np.mean(np.square(np.asarray(values, dtype=float)))))


# ----------------------------------------------------------------- data
@contextlib.contextmanager
def p88_loader(loader):
    """p88's module-level loader exchanged for the duration of a call of its functions."""
    old = F88.load
    F88.load = loader
    try:
        yield
    finally:
        F88.load = old


def seed_block(seeds, n):
    """Per combination: per-seed mean makespans of both arms, their per-seed values, spreads,
    the per-seed mean differences of section A of p88 and the paired test; Holm over R8."""
    seeds = list(seeds)
    rows = {}
    for r in R8:
        mp = [float(np.mean(SE.registry_load('released', None, s, SE.FOLDER, r, n))) for s in seeds]
        ms = [float(np.mean(SE.registry_load('specialist', r, s, SE.FOLDER, r, n))) for s in seeds]
        with p88_loader(SE.registry_load):
            d = F88.seed_pairs(F88.released, F88.specialist, SE.FOLDER, r, n, seeds=seeds)
        vp, vs = relative(mp), relative(ms)
        dm = d.mean(axis=1)
        sp, ss, sd = spread(vp), spread(vs), spread(dm)
        rows[r] = dict(policy_means=mp, specialist_means=ms, policy_values=vp.tolist(),
                       specialist_values=vs.tolist(), difference_seed_means=dm.tolist(),
                       policy=dict(spread=sp[0], min=sp[1], max=sp[2]),
                       specialist=dict(spread=ss[0], min=ss[1], max=ss[2]),
                       difference=dict(spread=sd[0], min=sd[1], max=sd[2],
                                       mean=float(d.mean())),
                       test=pitman_morgan(vp, vs), n=int(d.shape[1]))
    for r, a in zip(R8, SF.holm([rows[r]['test']['p'] for r in R8])):
        rows[r]['test']['p_holm'] = a
    return dict(seeds=seeds, rows=rows,
                rms=dict(policy=rms([rows[r]['policy']['spread'] for r in R8]),
                         specialist=rms([rows[r]['specialist']['spread'] for r in R8]),
                         difference=rms([rows[r]['difference']['spread'] for r in R8])),
                holm_below=sum(rows[r]['test']['p_holm'] < ALPHA for r in R8))


def first_five_missing(n):
    """Arrays of the first five seeds that are absent or do not hold n rows."""
    out = []
    for s in SE.FIRST_SEEDS:
        for arm, combo, r in [('released', None, r) for r in R8] + \
                             [('specialist', r, r) for r in R8]:
            p = SE.result_path(arm, combo, s, r)
            if not os.path.exists(p):
                out.append(f'missing {p}')
            elif np.load(p).shape[0] != n:
                out.append(f'{p}: {np.load(p).shape[0]} rows, the test set has {n}')
    return out


def check_setting():
    SE.require_exact()
    if REG.seeds() != SE.FIRST_SEEDS:
        raise Refused(f'run with PACT_SEEDS={SE.FIRST_SEEDS[0]}-{SE.FIRST_SEEDS[-1]}; the second '
                      f'five seeds are read through scripts/seed_extension.py')


def analyse(root='.', dry=False):
    """The seed sets the disk allows; returns the result dict. Raises Refused."""
    check_setting()
    n = SE.n_test()
    miss = first_five_missing(n)
    if miss:
        raise Refused(f'{len(miss)} arrays of the first five seeds do not count; first: {miss[0]}')
    F88.verify_bases(SE.FOLDER, n)
    second = SE.complete(root, n)
    sets = {'First': seed_block(SE.FIRST_SEEDS, n)}
    if second:
        sets['Second'] = seed_block(SE.SECOND_SEEDS, n)
        sets['Ten'] = seed_block(SE.ALL_SEEDS, n)
    left = ('' if second else
            'the second five seeds are not complete (scripts/seed_extension.py state: '
            f'{sum(not x["complete"] for x in SE.state(root, n))} of 45 runs not trained or not '
            'evaluated), so the seed sets Second and Ten are left out')
    names = {'First': SE.FIRST_SEEDS, 'Second': SE.SECOND_SEEDS, 'Ten': SE.ALL_SEEDS}
    setting = (f'environment {REG.env()}, training seeds '
               + ', '.join(f'{names[k][0]} to {names[k][-1]}' for k in sets))
    return dict(generator='scripts/p112_seed_spread.py', setting=setting,
                folder=REG.res_dir(SE.FOLDER), n=n, dry=bool(dry), sets=sets,
                left_out=left, alpha=ALPHA,
                definition=('v_s = 100 (m_s / mean_s m_s - 1), m_s the mean makespan of seed s '
                            'over the test instances; spread = standard deviation of v_s over the '
                            'seeds, ddof 1; difference = per-seed mean of 100 (policy - '
                            'specialist) / specialist (p88 section A); paired test of equal '
                            'variances of Pitman and Morgan, t with n - 2 df, Holm over the eight '
                            'combinations; RMS = root mean square over the eight combinations'))


# ----------------------------------------------------------------- outputs
def defined_macros():
    """{name: file} of every macro defined in a .tex file under manuscript/, the output file
    left out."""
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


def macro_lines(res):
    f2 = F88.f2
    L = [f'% Generated by scripts/p112_seed_spread.py ({res["setting"]}) -- do not hand-edit',
         '% Spread of the mean makespan across training seeds at 10x25: standard deviation '
         '(ddof 1) over the seeds of 100 (m_s / mean m - 1), %, per arm; Diff: standard deviation '
         'of the per-seed mean differences of the policy against the specialist (p88 section A); '
         'P: Pitman-Morgan paired test of equal variances, Holm over the eight combinations.']
    if res['dry']:
        L.append('% DRY RUN on synthetic arrays')
    if res['left_out']:
        L.append(f'% Left out: {res["left_out"]}.')
    L.append(f'\\newcommand{{\\{PREFIX}SetsN}}{{{len(res["sets"])}}}'
             f'\\newcommand{{\\{PREFIX}Instances}}{{{res["n"]}}}'
             '  % seed sets computed, test instances per combination')
    for word, text in SET_WORDS:
        blk = res['sets'].get(word)
        if blk is None:
            continue
        L.append(f'\\newcommand{{\\{PREFIX}{word}SeedN}}{{{len(blk["seeds"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}{word}{aw}Rms}}{{{blk["rms"][arm]:.2f}}}'
                           for aw, arm in ARM_WORDS)
                 + f'\\newcommand{{\\{PREFIX}{word}DiffRms}}{{{blk["rms"]["difference"]:.2f}}}'
                 f'\\newcommand{{\\{PREFIX}{word}HolmN}}{{{blk["holm_below"]}}}'
                 f'  % {text}: seeds; root mean square of the spreads over the eight '
                 f'combinations (policy, specialist, difference); combinations with Holm p '
                 f'below {ALPHA:g}')
        for r in R8:
            x = blk['rows'][r]
            s = ''
            for aw, arm in ARM_WORDS:
                s += (f'\\newcommand{{\\{PREFIX}{word}{aw}{r}}}{{{x[arm]["spread"]:.2f}}}'
                      f'\\newcommand{{\\{PREFIX}{word}{aw}Min{r}}}{{{f2(x[arm]["min"])}}}'
                      f'\\newcommand{{\\{PREFIX}{word}{aw}Max{r}}}{{{f2(x[arm]["max"])}}}')
            s += (f'\\newcommand{{\\{PREFIX}{word}Diff{r}}}{{{x["difference"]["spread"]:.2f}}}'
                  f'\\newcommand{{\\{PREFIX}{word}P{r}}}{{{x["test"]["p_holm"]:.2g}}}')
            L.append(s + f'  % {text}, {r}: spread, smallest and largest per-seed value of the '
                     f'policy and of the specialist, spread of the differences, Holm p (raw p '
                     f'{x["test"]["p"]:.3g}, r {x["test"]["r"]:+.3f})')
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
        raise Refused(f'the prefix {PREFIX} is used by {taken[0]} ({have[taken[0]]})')
    return names


def markdown(res):
    f2 = F88.f2
    md = [f'# Spread of the mean makespan across training seeds ({res["setting"]})', '',
          f'Arrays: `{res["folder"]}`, {res["n"]} test instances per combination at 10x25.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          f'Definition: {res["definition"]}.', '']
    if res['left_out']:
        md += [f'Left out: {res["left_out"]}.', '']
    for word, text in SET_WORDS:
        blk = res['sets'].get(word)
        if blk is None:
            continue
        md += [f'## {text}', '',
               '| comb. | policy spread % [min, max] | specialist spread % [min, max] | '
               'difference spread % | r | p | Holm p |', '|---|---|---|---|---|---|---|']
        for r in R8:
            x = blk['rows'][r]
            md.append(f'| {r} | {x["policy"]["spread"]:.2f} [{f2(x["policy"]["min"])}, '
                      f'{f2(x["policy"]["max"])}] | {x["specialist"]["spread"]:.2f} '
                      f'[{f2(x["specialist"]["min"])}, {f2(x["specialist"]["max"])}] | '
                      f'{x["difference"]["spread"]:.2f} | {x["test"]["r"]:+.3f} | '
                      f'{x["test"]["p"]:.2g} | {x["test"]["p_holm"]:.2g} |')
        md += ['', f'Root mean square over the eight combinations: policy '
               f'{blk["rms"]["policy"]:.2f}%, specialist {blk["rms"]["specialist"]:.2f}%, '
               f'difference {blk["rms"]["difference"]:.2f}%; Holm p below {ALPHA:g} on '
               f'{blk["holm_below"]} of 8 combinations.', '']
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(root='.', dry=False):
    """analyse() and the three outputs; a refusal removes them. Returns (res, names, md)."""
    check_setting()
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, dry)
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
    check_setting()
    n = SE.n_test()
    miss = first_five_missing(n)
    print(f'p112: arrays of the first five seeds: '
          + ('every one present with one row per test instance' if not miss else
             f'{len(miss)} do not count; first {miss[0]}'))
    print('p112: the seed sets Second and Ten are computed by the run when all 45 runs of the '
          'second five seeds are complete (scripts/seed_extension.py), and left out otherwise; '
          'this check does not read their files')
    return 0 if not miss else 1


# ----------------------------------------------------------------- dry mode
# per-seed deviations of the mean makespan in % (seeds 301 to 310); each half and the whole
# list sum to zero, so v_s equals the deviation exactly in every seed set
DRY_EFFECT = {'policy': [0.2, -0.1, 0.1, -0.3, 0.1, -0.2, 0.1, 0.2, 0.0, -0.1],
              'specialist': [1.0, -0.6, 0.8, -1.4, 0.2, 0.9, -1.1, 0.5, -0.7, 0.4]}


def dry_expected(seeds):
    """The spread of each arm that the synthetic arrays of the given seeds carry."""
    idx = [list(SE.ALL_SEEDS).index(s) for s in seeds]
    return {arm: float(np.std(np.asarray(v)[idx], ddof=1)) for arm, v in DRY_EFFECT.items()}


def build_synthetic(tree, n, seeds):
    """Arrays of the policy and the specialists of the given seeds under <tree>/results, and
    the training records of the second five (p110's synthetic records) when they are given."""
    import p110_seed_extension as P110
    rng = np.random.default_rng(112)
    base = {r: rng.uniform(100.0, 300.0, n) for r in R8}
    for s in seeds:
        k = list(SE.ALL_SEEDS).index(s)
        for r in R8:
            for arm, combo, e in (('released', None, DRY_EFFECT['policy'][k]),
                                  ('specialist', r, DRY_EFFECT['specialist'][k])):
                p = SE.result_path(arm, combo, s, r)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                x = base[r] * (1.0 + e / 100.0)
                np.save(p, np.stack([x, np.zeros(n)], axis=1))
        if SE.is_second(s) or len(seeds) > len(SE.FIRST_SEEDS):
            for arm in SE.ARMS:
                for c in (R8 if arm == 'specialist' else [None]):
                    P110.synthetic_records(tree, arm, c, s)


def dry_run(tree, n_seeds=10):
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    import p110_seed_extension as P110
    seeds = SE.ALL_SEEDS if n_seeds == 10 else SE.FIRST_SEEDS
    with P110.environment(**P110.dry_paths(tree)):
        build_synthetic(tree, SE.n_test(), seeds)
        return run(root=tree, dry=True)


def main():
    ap = argparse.ArgumentParser(description='spread of the mean makespan across training seeds')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder, '
                         'removed afterwards)')
    ap.add_argument('--dry-seeds', type=int, choices=(5, 10), default=10,
                    help='synthetic seeds of the dry run: 5 (the second five absent) or 10')
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
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p112_dry_')
        try:
            res, names, md = dry_run(tree, a.dry_seeds)
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    want = {w: dry_expected(blk['seeds']) for w, blk in res['sets'].items()}
    got = {w: {arm: blk['rows'][R8[0]][arm]['spread'] for arm in DRY_EFFECT}
           for w, blk in res['sets'].items()}
    ok = all(abs(res['sets'][w]['rows'][r][arm]['spread'] - want[w][arm]) < 1e-9
             for w in want for r in R8 for arm in DRY_EFFECT)
    if keep:
        print('\n'.join(md))
    print(f'p112 dry run on synthetic arrays {"passed" if ok else "FAILED"}: seed sets '
          f'{", ".join(res["sets"])}, {len(names)} macros; spreads '
          + '; '.join(f'{w} policy {got[w]["policy"]:.4f} (want {want[w]["policy"]:.4f}), '
                      f'specialist {got[w]["specialist"]:.4f} (want {want[w]["specialist"]:.4f})'
                      for w in want)
          + (f'; left out: {res["left_out"]}' if res['left_out'] else ''))
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
