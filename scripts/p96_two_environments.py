r"""P96: the environment of the paper against the implementation released with the prior
lag-only study, on the training seeds both environments hold (301 to 303), at 10x25.

The prior implementation is the registry environment of the stored runs (jointblind-v1,
jointblind-nlsw, spec-<R> and the dispatching rules in test_results/FAMILY/10x25); the
environment of the paper is the exact mask (xmask-v1, xmask-ctl, xmask-solo-<R> and the rules
re-evaluated under the exact mask in test_results/FAMILY/10x25_xmask). Each run is evaluated
in its own environment on the same test instances (data/FAMILY/10x25), and seed s of one
environment meets seed s of the other (the same seed number).

QUANTITIES  d = 100 (paper - prior) / prior per instance; negative = the environment of the
            paper gives the shorter schedule
  Pol<R>, Spec<R>, Ctl<R>
      the released policy (eight combinations), the specialist of each combination on its own
      combination, the single-rule control (eight combinations): d [k, n] seed s against seed
      s (p88_family_inference.seed_pairs), crossed hierarchical bootstrap over the seeds and
      the base instances with a fresh default_rng(11) per quantity (p88 component and fresh),
      direction from the unrounded interval; the seed-level t interval goes to the JSON.
      Pool<F>: pooled over the combinations of a family (policy and specialists: All, Train,
      Held; control: All, Seen = N, L, S, W, Unseen), a drawn base instance bringing all its
      combinations (p88 pooled_block).
  Rule<Q><R>, Best<R>
      each dispatching rule (FIFO, MOR, SPT, MWKR) and their per-instance best (p88 pool):
      the same difference per instance, instance-level cluster bootstrap with a fresh
      default_rng(7) per quantity, per combination and pooled (Pool), direction.
  Par<E><R>  (E = Prior, Exact)
      the two verdict tables side by side: the released policy against the specialist of each
      combination in each environment on these seeds, by p88's own procedure
      (p88_family_inference.parity_components: one default_rng(11) stream in p27's order),
      with p88's IUT (Iut<F><Half|One><Eq|Ni>) and pooled rows (Pool<F>); verdicts at 0.5% and
      1% and direction. ParDiffN<Half|One>: combinations whose verdict differs between the
      environments; ParDirDiffN: combinations whose direction differs.

The analysis refuses when an array of either environment is missing or does not have one row
per test instance, and then removes its earlier outputs.

OUTPUTS (under --out, default the repository root)
  manuscript/macros_environments.tex   names Env..., letters only
  results/environments.json, results/environments.md

Usage: python scripts/p96_two_environments.py [--boot 20000] [--seeds 301-303]
           [--root <repository root>] [--out <root of the outputs>] [--check]
       --check lists the missing arrays of both environments and writes nothing (exit 0
       when every array exists).
"""
import argparse
import contextlib
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import f2  # noqa: E402

PRIOR, PAPER = 'legacy', 'exact'            # registry keys of the two environments
ENVS = (PRIOR, PAPER)
ENV_WORD = {PRIOR: 'Prior', PAPER: 'Exact'}
ENV_TEXT = {PRIOR: 'prior implementation', PAPER: 'environment of the paper'}
FOLDER = '10x25'
DEFAULT_SEEDS = '301-303'
R8, TRAIN, HELD = list(F88.R8), list(F88.TRAIN), list(F88.HELD)
CTL_SEEN = list(REG.CONTROL_COMBOS)
RULES = F88.RULES
MARGINS, MW = F88.MARGINS, F88.MW
INST_SEED = 7
ARMS = (('Pol', 'released policy', F88.released, 'released'),
        ('Spec', 'specialist of the combination, on its own combination', F88.specialist,
         'specialist'),
        ('Ctl', 'single-rule control', F88.control, 'control'))
FAMS = {'Pol': dict(F88.PAR_FAM), 'Spec': dict(F88.PAR_FAM),
        'Ctl': {'All': R8, 'Seen': CTL_SEEN, 'Unseen': [r for r in R8 if r not in CTL_SEEN]}}
OUT_FILES = ('manuscript/macros_environments.tex', 'results/environments.json',
             'results/environments.md')


@contextlib.contextmanager
def environment(e):
    """The registry reads environment e inside the block (PACT_ENV), the previous one after."""
    previous = os.environ.get('PACT_ENV')
    os.environ['PACT_ENV'] = e
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('PACT_ENV', None)
        else:
            os.environ['PACT_ENV'] = previous


def in_env(e, reader):
    """A reader (s, folder, r, n) of p88 that reads environment e."""
    def read(s, folder, r, n=None):
        with environment(e):
            return reader(s, folder, r, n)
    return read


def n_instances():
    return F88.descriptor_count(FOLDER)


def input_paths(seeds):
    """{environment: [array paths]} of every array this analysis reads."""
    out = {}
    for e in ENVS:
        with environment(e):
            ps = [REG.result_path(arm, r if arm == 'specialist' else None, s, FOLDER, r)
                  for _, _, _, arm in ARMS for r in R8 for s in seeds]
            ps += [f'{REG.rule_dir(FOLDER)}/Result_{q}_{r}.npy' for q in RULES for r in R8]
        out[e] = ps
    return out


def missing_inputs(seeds=None):
    """{environment text: [problems]}: arrays that are absent or do not have one row per test
    instance of data/FAMILY/10x25; empty lists when the analysis can run."""
    seeds = tuple(seeds or REG.parse_seeds(DEFAULT_SEEDS))
    n = n_instances()
    out = {}
    for e, paths in input_paths(seeds).items():
        bad = [] if n else [f'data/FAMILY/{FOLDER}/desc/N holds no descriptor']
        for p in paths:
            if not os.path.exists(p):
                bad.append(f'missing {p}')
            elif len(REG.read_path(p)) != n:
                bad.append(f'{p}: {len(REG.read_path(p))} rows, the test set has {n}')
        out[ENV_TEXT[e]] = bad
    return out


# ------------------------------------------------------------------ statistics
def arm_block(key, reader, seeds):
    """Paper minus prior for one arm: per combination the p88 component of d [k, n] (fresh
    draws), pooled rows over the families of the arm."""
    combos = R8
    D = {r: F88.seed_pairs(in_env(PAPER, reader), in_env(PRIOR, reader), FOLDER, r, seeds=seeds)
         for r in combos}
    rows = {r: F88.component(d, F88.fresh(d)) for r, d in D.items()}
    return dict(rows=rows, pooled=F88.pooled_block(D, FAMS[key]), D=D)


def instance_row(d):
    ib = SF.instance_boot(d, B=F88.B, seed=INST_SEED, alpha=F88.ALPHA)
    return dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'], n=ib['n'], c=ib['c'],
                direction=SF.direction(ib['lo'], ib['hi']))


def rule_arrays(q, r, e):
    with environment(e):
        if q == 'Best':
            return F88.pool(FOLDER, r)
        return REG.read_path(f'{REG.rule_dir(FOLDER)}/Result_{q}_{r}.npy')


def rule_block():
    """Paper minus prior for each rule and their per-instance best, per instance: per
    combination and pooled over the eight (a drawn base instance with all its combinations),
    instance-level cluster bootstrap, fresh default_rng(7) per quantity."""
    out = {}
    for q in RULES + ('Best',):
        cells = np.stack([SF.rel_gap(rule_arrays(q, r, PAPER), rule_arrays(q, r, PRIOR))
                          for r in R8], axis=1)                       # [n, 8]
        out[q] = dict(rows={r: instance_row(cells[:, j]) for j, r in enumerate(R8)},
                      pooled=instance_row(cells))
    return out


def verdict_tables(seeds):
    """p88's section A in each environment on these seeds: components, IUT, pooled rows."""
    out = {}
    for e in ENVS:
        with environment(e):
            D, C = F88.parity_components(FOLDER, seeds)
        out[e] = dict(components=C, iut=F88.iut_block(C, F88.PAR_FAM),
                      pooled=F88.pooled_block(D, F88.PAR_FAM))
    a, b = out[PRIOR]['components'], out[PAPER]['components']
    out['differences'] = {
        **{f'v{mg}': [r for r in R8 if a[r][f'v{mg}'] != b[r][f'v{mg}']] for mg in MARGINS},
        'direction': [r for r in R8 if a[r]['direction'] != b[r]['direction']]}
    return out


def analyse(seeds):
    n = n_instances()
    F88.verify_bases(FOLDER, n)
    res = dict(arms={key: arm_block(key, reader, seeds) for key, _, reader, _ in ARMS},
               rules=rule_block(), verdict_tables=verdict_tables(seeds), n=n)
    for x in res['arms'].values():
        x.pop('D')
    return res


# ------------------------------------------------------------------ macros
def tex(v):
    return SF.TEX[v]


def interval(name, x, what):
    return (f'\\newcommand{{\\{name}}}{{{f2(x["mean"])}}}'
            f'\\newcommand{{\\{name}Lo}}{{{f2(x["lo"])}}}'
            f'\\newcommand{{\\{name}Hi}}{{{f2(x["hi"])}}}'
            f'\\newcommand{{\\{name}Dir}}{{{tex(SF.direction(x["lo"], x["hi"]))}}}'
            f'  % {what}')


def macro_lines(res, seeds, boot, alpha):
    sw = ','.join(map(str, seeds))
    L = ['% Generated by scripts/p96_two_environments.py -- do not hand-edit',
         f'% The environment of the paper (exact mask) against the implementation released with '
         f'the prior study, 10x25, training seeds {sw} (seed s against seed s), B = {boot}, '
         f'alpha = {alpha}.',
         '% d = 100 (paper - prior) / prior per instance, negative = the environment of the '
         'paper gives the shorter schedule. Policy, specialists, control: crossed hierarchical '
         'bootstrap over seeds and base instances, fresh default_rng(11) per quantity; rules: '
         'instance-level cluster bootstrap, fresh default_rng(7) per quantity; Dir from the '
         'unrounded interval (ahead = the environment of the paper gives the shorter schedule, '
         'behind = the longer one, neither = the interval contains zero), except in the verdict '
         'tables, where ahead = the released policy is shorter than the specialist.',
         f'\\newcommand{{\\EnvSeedN}}{{{len(seeds)}}}  % training seeds in both environments',
         f'\\newcommand{{\\EnvInstN}}{{{res["n"]}}}  % test instances per combination',
         f'\\newcommand{{\\EnvBoot}}{{{boot}}}  % bootstrap replicates']
    for key, what, _, _ in ARMS:
        blk = res['arms'][key]
        for r in R8:
            L.append(interval(f'Env{key}{r}', blk['rows'][r],
                              f'{what}, {r}: paper minus prior implementation, hierarchical CI'))
        for fam, x in blk['pooled'].items():
            L.append(interval(f'Env{key}Pool{fam}', x, f'{what}: pooled over {x["c"]} '
                              f'combinations ({fam}), base instances with all their combinations'))
    for q, blk in res['rules'].items():
        name = 'EnvBest' if q == 'Best' else f'EnvRule{q}'
        what = 'per-instance best of the four rules' if q == 'Best' else f'rule {q}'
        for r in R8:
            L.append(interval(f'{name}{r}', blk['rows'][r], f'{what}, {r}: paper minus prior '
                              f'implementation, instance-level cluster CI'))
        L.append(interval(f'{name}Pool', blk['pooled'], f'{what}: pooled over the eight '
                          f'combinations, a base instance with all its combinations'))
    vt = res['verdict_tables']
    L.append('% Verdict tables: released policy vs the specialist of each combination in each '
             'environment on these seeds (p88 section A procedure); Prior = prior '
             'implementation, Exact = environment of the paper.')
    for e in ENVS:
        w = f'EnvPar{ENV_WORD[e]}'
        for r in R8:
            c = vt[e]['components'][r]
            L.append(f'\\newcommand{{\\{w}{r}}}{{{f2(c["mean"])}}}'
                     f'\\newcommand{{\\{w}{r}Lo}}{{{f2(c["lo"])}}}'
                     f'\\newcommand{{\\{w}{r}Hi}}{{{f2(c["hi"])}}}'
                     + ''.join(f'\\newcommand{{\\{w}{r}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}'
                               for mg in MARGINS)
                     + f'\\newcommand{{\\{w}{r}Dir}}{{{tex(c["direction"])}}}'
                     f'  % {ENV_TEXT[e]}, {r}: released policy vs specialist, hierarchical CI, '
                     f'verdicts at 0.5 and 1, direction')
        for fam in F88.PAR_FAM:
            for mg in MARGINS:
                for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                    x = vt[e]['iut'][f'{fam}|{mg}|{claim}']
                    L.append(f'\\newcommand{{\\{w}Iut{fam}{MW[mg]}{cw}}}{{{int(x["holds"])}}}'
                             f'  % {ENV_TEXT[e]}: IUT, {claim} on {",".join(x["components"])} '
                             f'at +-{mg}%, failing {",".join(x["failing"]) or "none"}')
        for fam, x in vt[e]['pooled'].items():
            L.append(f'\\newcommand{{\\{w}Pool{fam}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\{w}Pool{fam}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\{w}Pool{fam}Hi}}{{{f2(x["hi"])}}}'
                     + ''.join(f'\\newcommand{{\\{w}Pool{fam}V{MW[mg]}}}{{{tex(x[f"v{mg}"])}}}'
                               for mg in MARGINS)
                     + f'\\newcommand{{\\{w}Pool{fam}Dir}}'
                       f'{{{tex(SF.direction(x["lo"], x["hi"]))}}}'
                     f'  % {ENV_TEXT[e]}: pooled over {x["c"]} combinations ({fam})')
    dif = vt['differences']
    L.append(''.join(f'\\newcommand{{\\EnvParDiffN{MW[mg]}}}{{{len(dif[f"v{mg}"])}}}'
                     for mg in MARGINS)
             + f'\\newcommand{{\\EnvParDirDiffN}}{{{len(dif["direction"])}}}'
             f'  % combinations whose verdict at 0.5 / at 1 / direction differs between the two '
             f'environments: {", ".join(dif["v0.5"]) or "none"} / '
             f'{", ".join(dif["v1.0"]) or "none"} / {", ".join(dif["direction"]) or "none"}')
    return L


DEF_RX = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')


def collisions(names, root, out_mac):
    """Names this file defines that another macro file of the manuscripts already defines
    (manuscript/macros*.tex, manuscript/*/macros*.tex, paper/macros*.tex under root); a file
    with the name of the output is skipped, so a run under another output root works."""
    hits = {}
    files = (glob.glob(os.path.join(root, 'manuscript', 'macros*.tex'))
             + glob.glob(os.path.join(root, 'manuscript', '*', 'macros*.tex'))
             + glob.glob(os.path.join(root, 'paper', 'macros*.tex')))
    for f in sorted(set(files)):
        if os.path.basename(f) == os.path.basename(out_mac):
            continue
        defined = {a or b for a, b in DEF_RX.findall(open(f, errors='replace').read())}
        clash = sorted(set(names) & defined)
        if clash:
            hits[os.path.relpath(f, root)] = clash
    return hits


# ------------------------------------------------------------------ report
def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def ci(x):
    return f'{x["mean"]:+.2f} [{x["lo"]:+.2f}, {x["hi"]:+.2f}]'


def markdown(res, seeds, boot):
    md = ['# The environment of the paper against the prior implementation (10x25)', '',
          f'Training seeds {", ".join(map(str, seeds))} in both environments, seed s against seed '
          f's; {res["n"]} test instances per combination; B = {boot}. d = 100 (paper - prior) / '
          'prior, negative = the environment of the paper gives the shorter schedule.', '',
          '## Policy, specialists and single-rule control (hierarchical 95% CI)', '',
          '| comb. | released policy | specialist (own comb.) | single-rule control |',
          '|---|---|---|---|']
    A = res['arms']
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(f'{ci(A[k]["rows"][r])} {A[k]["rows"][r]["direction"]}'
                                           for k in ('Pol', 'Spec', 'Ctl')) + ' |')
    for key, what, _, _ in ARMS:
        md.append('')
        md += [f'{what}, pooled {fam}: {ci(x)} ({x["direction"]})'
               for fam, x in A[key]['pooled'].items()]
    md += ['', '## Dispatching rules (instance-level cluster 95% CI)', '',
           '| comb. | ' + ' | '.join(RULES) + ' | per-instance best |', '|---' * 6 + '|']
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(ci(res['rules'][q]['rows'][r])
                                           for q in RULES + ('Best',)) + ' |')
    md.append('| pooled | ' + ' | '.join(ci(res['rules'][q]['pooled']) for q in RULES + ('Best',))
              + ' |')
    vt = res['verdict_tables']
    md += ['', '## Verdict tables side by side: released policy vs specialist', '',
           '| comb. | prior implementation | 0.5% | 1% | dir. | environment of the paper | 0.5% | '
           '1% | dir. |', '|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        a, b = vt[PRIOR]['components'][r], vt[PAPER]['components'][r]
        md.append(f'| {r} | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | {a["direction"]} | {ci(b)} | '
                  f'{b["v0.5"]} | {b["v1.0"]} | {b["direction"]} |')
    for fam in F88.PAR_FAM:
        a, b = vt[PRIOR]['pooled'][fam], vt[PAPER]['pooled'][fam]
        md.append(f'| pooled {fam} | {ci(a)} | {a["v0.5"]} | {a["v1.0"]} | {a["direction"]} | '
                  f'{ci(b)} | {b["v0.5"]} | {b["v1.0"]} | {b["direction"]} |')
    dif = vt['differences']
    md += ['', f'Verdict differs at 0.5%: {", ".join(dif["v0.5"]) or "none"}; at 1%: '
           f'{", ".join(dif["v1.0"]) or "none"}; direction differs: '
           f'{", ".join(dif["direction"]) or "none"}.']
    return md


# ------------------------------------------------------------------ main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--boot', type=int, default=SF.B_DEFAULT)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--seeds', default=DEFAULT_SEEDS,
                    help='training seeds held by both environments (default 301-303)')
    ap.add_argument('--root', default=REPO, help='repository root the inputs are read from')
    ap.add_argument('--out', default=None, help='root of manuscript/ and results/ (default --root)')
    ap.add_argument('--check', action='store_true',
                    help='list the missing arrays of both environments; write nothing')
    return ap


def remove_outputs(paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)
            print(f'removed {p} (refusing to certify)')


def main(argv=None):
    a = build_parser().parse_args(argv)
    root = os.path.abspath(a.root)
    out_root = os.path.abspath(a.out or a.root)
    seeds = REG.parse_seeds(a.seeds)
    for e in ENVS:
        bad = [s for s in seeds if s not in REG.PLANNED_SEEDS[e]]
        if bad:
            sys.exit(f'REFUSING: seeds {bad} are not planned in both environments')
    os.chdir(root)
    paths = [os.path.join(out_root, p) for p in OUT_FILES]
    miss = missing_inputs(seeds)
    n_miss = sum(len(v) for v in miss.values())
    if a.check:
        for e, v in miss.items():
            print(f'{e}: {len(v)} problem(s)' + ''.join(f'\n  {x}' for x in v[:20])
                  + (f'\n  ... and {len(v) - 20} more' if len(v) > 20 else ''))
        return 0 if n_miss == 0 else 1
    if n_miss:
        remove_outputs(paths)
        print('REFUSING: arrays of the two environments are missing or incomplete:',
              file=sys.stderr)
        for e, v in miss.items():
            print(f'  {e}: {len(v)}' + ''.join(f'\n    {x}' for x in v[:10]), file=sys.stderr)
        return 1
    F88.configure(boot=a.boot, alpha=a.alpha)
    res = analyse(seeds)
    lines = macro_lines(res, seeds, a.boot, a.alpha)
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(lines))
    problems = []
    if len(names) != len(set(names)):
        problems.append('duplicate macro names in the output')
    if not all(re.fullmatch(r'Env[A-Za-z]+', x) for x in names):
        problems.append('a macro name does not start with Env')
    clash = collisions(names, root, paths[0])
    if clash:
        problems.append('macro names already defined: ' + '; '.join(
            f'{f}: {", ".join(v[:5])}' for f, v in clash.items()))
    if problems:
        remove_outputs(paths)
        print('REFUSING: ' + '; '.join(problems), file=sys.stderr)
        return 1
    emit(paths[0], lines)
    out = dict(generator='scripts/p96_two_environments.py', seeds=list(seeds), boot=a.boot,
               alpha=a.alpha, folder=FOLDER, n_instances=res['n'],
               environments={'prior': 'implementation released with the prior study',
                             'paper': 'environment of the paper (exact mask)'},
               difference='d = 100 (paper - prior) / prior per instance, seed s against seed s; '
                          'negative = the environment of the paper gives the shorter schedule',
               streams=dict(arms='crossed hierarchical bootstrap, fresh default_rng(11) per '
                                 'quantity', rules='instance-level cluster bootstrap, fresh '
                                                   f'default_rng({INST_SEED}) per quantity',
                            verdict_tables='p88 section A: one default_rng(11) stream in '
                                           "p27's order per environment"),
               inputs={'prior': input_paths(seeds)[PRIOR], 'paper': input_paths(seeds)[PAPER]},
               arms=res['arms'], rules=res['rules'],
               verdict_tables={'prior': res['verdict_tables'][PRIOR],
                               'paper': res['verdict_tables'][PAPER],
                               'differences': res['verdict_tables']['differences']},
               n_macros=len(names))
    os.makedirs(os.path.dirname(paths[1]), exist_ok=True)
    with open(paths[1], 'w') as f:
        json.dump(jsonable(out), f, indent=1)
    md = markdown(res, seeds, a.boot)
    with open(paths[2], 'w') as f:
        f.write('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'\n{len(names)} macros in {paths[0]}; wrote {paths[1]}, {paths[2]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
