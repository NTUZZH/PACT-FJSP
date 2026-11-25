r"""P125: the principal comparison in counts and in hours.

QUESTION
  The principal comparison (the policy against the specialist of each combination on the 100
  test instances at 10x25; scripts/p88_family_inference.py section A for five training seeds,
  scripts/p110_seed_extension.py for ten) reports mean differences in percent. This script
  states the same comparison as counts of instances and in hours.

SEEDS (the pairing of p110, its completeness rule)
  Seed s of the policy against seed s of the specialist. Ten seeds (301 to 310) only when the
  extension rule applies in the five-seed record (seed_extension.triggered) and all 45 runs
  of the second five seeds are trained and evaluated (seed_extension.complete, the check of
  `scripts/seed_extension.py check` and p110's precondition 2); then the validation norms of
  the ten policy runs must be equal (p110.policy_norms) and the ten-seed record of p110
  (results/seed_extension.json) must exist. Otherwise the five seeds of p88 (301 to 305), and
  every output says so.

WHAT IS COMPUTED (per combination; pooled over All (eight), Train (six) and Held (two), p88's
families PAR_FAM)
  counts   over (seed pair, instance): policy shorter, equal makespan, policy longer, as
           shares in percent (p93_rule_comparison.shares: sign of the difference; makespans
           are integer hours); on the per-instance means over the seeds (100 instances per
           combination): the same shares with "equal" meaning |mean difference| < 0.5 h
           (EQUAL_H; the difference set to 0 below it, then p93_rule_comparison.shares)
  hours    the mean of policy minus specialist in hours with the interval p110 reports for
           the percent difference, from the same functions on the hour differences:
           per combination p88.component on the draws of stats_family.hier_draws_stream from
           one parity stream in p88's order (TRAIN + HELD), pooled rows p88.pooled_block
           (fresh generator), and the seed-pooled instance-level row stats_family.instance_boot;
           seeds of the streams: p88's (11, 11, 7) for five seeds, p110.TEN_SEED for ten (via
           p110.p88_setting). The draws consume a generator by shape only, so the hour
           intervals are taken over the same resamples as the percent intervals.
  extremes the largest per-instance loss and gain of the policy (per-instance means over the
           seeds) in hours and in percent (p88.rel of the seed-pooled makespans), with the
           instance.

CHECK BEFORE ANY NUMBER (a failure refuses and removes the outputs)
  Section A of p88 recomputed from the same loaders (p110.section_a) reproduces every number
  of the five-seed record results/family_inference.json at the printed precision
  (p110.reproduction); with ten seeds also the ten-seed record of p110 (mean and interval of
  every combination and pooled row). The percent differences built here equal section A's
  arrays exactly.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_counts.tex   macro_io.emit; names Cnt<letters>; the prefix Cnt must be used
                                by no other .tex file under manuscript/
  results/instance_counts.json  every number, the seeds and the reason for them
  results/instance_counts.md    tables

DRY (--dry [DIR]): p110.build_synthetic on a tree in DIR or a temporary folder: the five-seed
branch (one array of a second seed removed) and the ten-seed branch (p110's own output written
first), B = --boot (default 500).

Usage (repository root, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p125_instance_counts.py
  ... --check
  python scripts/p125_instance_counts.py --dry [DIR]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import p93_rule_comparison as P93  # noqa: E402
import p110_seed_extension as P110  # noqa: E402
import seed_extension as SE  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p125_instance_counts.py'
PREFIX = 'Cnt'
OUT_FILES = {'macros': 'manuscript/macros_counts.tex', 'json': 'results/instance_counts.json',
             'md': 'results/instance_counts.md'}
TEN_RECORD = P110.OUT_FILES['json']
EQUAL_H = 0.5
R8 = list(F88.R8)
FAM = dict(F88.PAR_FAM)
ORDER = list(F88.TRAIN) + list(F88.HELD)          # p88's parity stream order
DRY_BOOT = 500
Refused = SE.Refused
FAM_TEXT = {'All': 'all eight combinations', 'Train': 'the six training combinations',
            'Held': 'the two held-out combinations'}



def setting_text(seeds=None):
    """The environment and the training seeds of the analysis, in the words of the outputs."""
    import arm_registry as REG
    s = REG.seeds() if seeds is None else seeds
    return f'environment {REG.env()}, training seeds {",".join(map(str, s))}'

def shown(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def entry_text(r):
    return FAM_TEXT.get(r, r)


# ----------------------------------------------------------------- the seeds
def seed_setting(root=ROOT):
    """(branch, seeds, loader, stream seed or None for p88's, reason). Raises Refused when the
    five-seed record is missing or foreign."""
    hit = SE.triggered()
    if not hit:
        return ('five', list(SE.FIRST_SEEDS), None, None,
                'the extension rule does not apply in the five-seed record, so the five seeds '
                'of p88')
    st = SE.state(root)
    miss = [x for x in st if not x['complete']]
    if miss:
        return ('five', list(SE.FIRST_SEEDS), None, None,
                f'the extension rule applies ({", ".join(hit)}), but {len(miss)} of {len(st)} '
                f'runs of the second five seeds are not complete (first {miss[0]["name"]}), so '
                f'the five seeds of p88')
    return ('ten', list(SE.ALL_SEEDS), SE.registry_load, P110.TEN_SEED,
            f'the extension rule applies ({", ".join(hit)}) and all {len(st)} runs of the second '
            f'five seeds are complete, so the ten seeds of p110')


def check(root=ROOT):
    """--check: the record, the seeds and every input array; 1 while any is missing."""
    try:
        P110.check_setting()
        rec = SE.read_record()
        branch, seeds, loader, _, why = seed_setting(root)
    except Refused as e:
        print(f'p125: missing: {e}')
        print('p125: incomplete')
        return 1
    print(f'p125: {branch} seeds ({", ".join(map(str, seeds))}): {why}')
    print(f'p125: five-seed record {shown(SE.record_path())}: B = {rec["boot"]}')
    miss = []
    read = loader or F88.load
    n = SE.n_test()
    for s in seeds:
        for r in R8:
            for arm, combo in (('released', None), ('specialist', r)):
                try:
                    x = read(arm, combo, s, SE.FOLDER, r)
                    if len(x) != n:
                        miss.append(f'{arm} {combo or ""} seed {s} on {r}: {len(x)} rows, want {n}')
                except (REG.MissingResult, SystemExit, Refused) as e:
                    miss.append(f'{arm} {combo or ""} seed {s} on {r}: {e}')
    if branch == 'ten':
        try:
            P110.policy_norms(root)
        except Refused as e:
            miss.append(str(e))
        if not os.path.exists(REG.out(TEN_RECORD)):
            miss.append(f'the ten-seed record {shown(REG.out(TEN_RECORD))} of p110 does not exist')
    for x in miss:
        print(f'p125: missing: {x}')
    print(f'p125: {"incomplete" if miss else "every input present"}')
    return 1 if miss else 0


# ----------------------------------------------------------------- reproduction
def reproduce(branch, seeds, loader, stream, root):
    """Section A in percent through p110.section_a: the five-seed record reproduced
    (p110.reproduction), and with ten seeds the ten-seed record of p110. Returns (the
    reproduction summary, D of section A for the seeds used)."""
    rec = SE.read_record()
    F88.configure(boot=rec['boot'], alpha=rec['alpha'])
    if int(rec['boot']) != F88.B:
        raise Refused(f'the record was computed with B = {rec["boot"]}, not {F88.B}')
    F88.verify_bases(SE.FOLDER, SE.n_test())
    D5, C5, iut5, pool5, pinst5 = P110.section_a(SE.FIRST_SEEDS, None, None)
    n_cmp, bad, dmax = P110.reproduction(rec, C5, iut5, pool5, pinst5)
    if bad:
        raise Refused(f'section A does not reproduce {shown(SE.record_path())} ({len(bad)} of '
                      f'{n_cmp}): ' + '; '.join(bad[:6]))
    out = dict(five=dict(record=shown(SE.record_path()), compared=n_cmp, identical=True,
                         largest_abs_difference=dmax,
                         example={k: dict(mean=C5[k]['mean'], lo=C5[k]['lo'], hi=C5[k]['hi'],
                                          v1=C5[k]['v1.0']) for k in R8},
                         pooled={f: dict(mean=pool5[f]['mean'], lo=pool5[f]['lo'],
                                         hi=pool5[f]['hi']) for f in FAM}))
    D = D5
    if branch == 'ten':
        P110.policy_norms(root)
        path = REG.out(TEN_RECORD)
        if not os.path.exists(path):
            raise Refused(f'the ten-seed record {shown(path)} of p110 does not exist')
        ten = json.load(open(path))
        D, C, _, pool, pinst = P110.section_a(seeds, loader, stream)
        bad, n = [], 0
        for k in R8:
            for q in ('mean', 'lo', 'hi'):
                n += 1
                if F88.f2(C[k][q]) != F88.f2(ten['ten'][k][q]):
                    bad.append(f'{k} {q}: stored {F88.f2(ten["ten"][k][q])}, here {F88.f2(C[k][q])}')
        for f in FAM:
            for blk, mine in (('pool_ten', pool), ('pool_inst_ten', pinst)):
                for q in ('mean', 'lo', 'hi'):
                    n += 1
                    if F88.f2(mine[f][q]) != F88.f2(ten[blk][f][q]):
                        bad.append(f'{blk} {f} {q}: stored {F88.f2(ten[blk][f][q])}, here '
                                   f'{F88.f2(mine[f][q])}')
        if bad:
            raise Refused(f'the ten-seed section A does not reproduce {shown(path)} ({len(bad)} '
                          f'of {n}): ' + '; '.join(bad[:6]))
        out['ten'] = dict(record=shown(path), compared=n, identical=True)
    return out, D


# ----------------------------------------------------------------- the analysis
def threshold(d):
    """Mean differences below EQUAL_H in absolute value set to 0 (equal)."""
    d = np.asarray(d, dtype=float)
    return np.where(np.abs(d) < EQUAL_H, 0.0, d)


def counts(dh):
    """Shares and counts of policy shorter / equal / longer over the entries of dh."""
    sh = P93.shares(dh)
    x = np.asarray(dh).ravel()
    return dict(sh, n=int(x.size), n_shorter=int((x < 0).sum()), n_equal=int((x == 0).sum()),
                n_longer=int((x > 0).sum()))


def extremes(dm, pm, cols):
    """Largest loss and gain of the policy over the per-instance means: dm hours [n, c], pm
    percent [n, c]; cols the combinations of the columns."""
    def at(x, f):
        i, j = np.unravel_index(int(f(x)), x.shape)
        return dict(value=float(x[i, j]), instance=int(i), combination=cols[j],
                    hours=float(dm[i, j]), pct=float(pm[i, j]))
    return dict(loss_hours=at(dm, np.argmax), gain_hours=at(dm, np.argmin),
                loss_pct=at(pm, np.argmax), gain_pct=at(pm, np.argmin))


def analyse(root=ROOT, dry=False):
    P110.check_setting()
    branch, seeds, loader, stream, why = seed_setting(root)
    repro, D = reproduce(branch, seeds, loader, stream, root)
    inst_seed = F88.INST_SEED if stream is None else stream
    pol, spec, Dh, Dp, comp = {}, {}, {}, {}, {}
    with P110.p88_setting(loader, stream, stream):
        rng = np.random.default_rng(F88.STREAM_SEED)
        for r in ORDER:
            pol[r] = np.stack([F88.released(s, SE.FOLDER, r) for s in seeds])
            spec[r] = np.stack([F88.specialist(s, SE.FOLDER, r) for s in seeds])
            Dh[r] = pol[r] - spec[r]
            Dp[r] = F88.rel(pol[r], spec[r])
            if not np.array_equal(Dp[r], D[r]):
                raise Refused(f'{r}: the percent differences built here differ from section A')
            c = F88.component(Dh[r], SF.hier_draws_stream(Dh[r], F88.B, rng))
            comp[r] = {k: c[k] for k in ('mean', 'lo', 'hi', 'sim_lo', 'sim_hi', 't_lo', 't_hi',
                                         'seed_means', 'n', 'direction')}
        pooled = {f: {k: v for k, v in x.items() if not k.startswith('v')}
                  for f, x in F88.pooled_block(Dh, FAM).items()}
    integer = all(np.all(x[r] == np.rint(x[r])) for x in (pol, spec) for r in R8)
    rows = {}
    for key, regs in [(r, [r]) for r in R8] + list(FAM.items()):
        dh = np.stack([Dh[r] for r in regs], axis=2)                       # [k, n, c]
        P_ = np.stack([pol[r] for r in regs], axis=2)
        S_ = np.stack([spec[r] for r in regs], axis=2)
        dm = P_.mean(axis=0) - S_.mean(axis=0)                             # [n, c]
        pm = F88.rel(P_.mean(axis=0), S_.mean(axis=0))
        x = dict(combinations=list(regs), pairs=counts(dh), instances=counts(threshold(dm)),
                 hours=comp[key] if key in comp else pooled[key],
                 extremes=extremes(dm, pm, list(regs)),
                 policy_mean_h=float(P_.mean()), specialist_mean_h=float(S_.mean()),
                 pct_mean=float(np.stack([Dp[r] for r in regs], axis=2).mean()))
        ib = SF.instance_boot(dm, B=F88.B, seed=inst_seed, alpha=F88.ALPHA)
        x['hours_instance_level'] = dict(mean=ib['mean'], lo=ib['lo'], hi=ib['hi'],
                                         direction=SF.direction(ib['lo'], ib['hi']),
                                         scheme=ib['scheme'])
        rows[key] = x
    return dict(generator=GENERATOR, dry=bool(dry), setting=setting_text(seeds), branch=branch,
                seeds=seeds, seeds_reason=why, boot=F88.B, alpha=F88.ALPHA,
                equal_threshold_h=EQUAL_H, n_instances=int(SE.n_test()),
                integer_makespans=bool(integer),
                streams=(f'parity stream default_rng({F88.STREAM_SEED if stream is None else stream}) '
                         f'in the order {ORDER}; pooled rows fresh default_rng('
                         f'{F88.FRESH_SEED if stream is None else stream}); instance-level rows '
                         f'default_rng({inst_seed})'),
                statistics=('p88.component on stats_family.hier_draws_stream (parity stream), '
                            'p88.pooled_block (stats_family.hier_boot, seed_t), '
                            'stats_family.instance_boot, stats_family.direction, '
                            'p93_rule_comparison.shares; seeds and streams via p110.p88_setting; '
                            'reproduction via p110.section_a and p110.reproduction'),
                reproduction=repro, rows=rows)


# ----------------------------------------------------------------- outputs
def hours(v):
    """Hours with two decimals and a sign; a value that rounds to zero prints as 0.00."""
    s = f'{v:+.2f}'
    return '0.00' if s in ('+0.00', '-0.00') else s


def defined_macros(own):
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    out = {}
    for d in sorted({os.path.join(ROOT, 'manuscript'), os.path.dirname(os.path.abspath(own))}):
        for f in sorted(glob.glob(os.path.join(d, '**', '*.tex'), recursive=True)):
            if os.path.basename(f) == os.path.basename(own):
                continue
            text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                             for x in open(f, errors='replace').read().splitlines())
            for a, b in rx.findall(text):
                out.setdefault(a or b, shown(f))
    return out


def check_names(lines, own):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x)]
    if bad:
        raise Refused(f'macro names that are not {PREFIX} plus letters: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused(f'duplicate macro names: {sorted({x for x in names if names.count(x) > 1})[:5]}')
    have = defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


def macro_lines(res):
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {res["setting"]}; the principal comparison at 10x25 with {len(res["seeds"])} training '
         f'seeds ({res["seeds"][0]} to {res["seeds"][-1]}; '
         f'{"the seeds of p110" if res["branch"] == "ten" else "the seeds of p88"}).',
         f'% Policy minus specialist, seed s against seed s. Pair shares: over (seed pair, '
         f'instance), equal = the same makespan. Instance shares: on the per-instance means over '
         f'the seeds, equal = |difference| < {EQUAL_H:g} h. Hours: mean difference and the '
         f'hierarchical 95% interval of p88 section A (B = {res["boot"]}; {res["streams"]}).']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}SeedN', len(res['seeds']), 'training seeds per method')
    nc(f'{PREFIX}InstN', res['n_instances'], 'test instances per combination')
    nc(f'{PREFIX}EqualHours', f'{EQUAL_H:g}', 'hours below which two per-instance means count as '
       'equal')
    for r, x in res['rows'].items():
        what = f'{entry_text(r)}'
        for kind, word in (('pairs', 'Pair'), ('instances', 'Inst')):
            c = x[kind]
            for k, w in (('shorter', 'Shorter'), ('equal', 'Equal'), ('longer', 'Longer')):
                nc(f'{PREFIX}{word}{w}{r}', f'{c[k]:.1f}',
                   f'{what}: percent of the {c["n"]} {"seed pairs x instances" if kind == "pairs" else "instances"} '
                   f'on which the policy is {k}')
        h = x['hours']
        nc(f'{PREFIX}Hours{r}', hours(h['mean']), f'{what}: mean difference in hours')
        nc(f'{PREFIX}Hours{r}Lo', hours(h['lo']), f'{what}: lower end of the hierarchical 95% '
           f'interval, hours ({h["lo"]:+.4f})')
        nc(f'{PREFIX}Hours{r}Hi', hours(h['hi']), f'{what}: upper end of the hierarchical 95% '
           f'interval, hours ({h["hi"]:+.4f})')
        nc(f'{PREFIX}Hours{r}Dir', SF.TEX[h['direction']], f'{what}: direction of that interval')
        e = x['extremes']
        nc(f'{PREFIX}LossHours{r}', hours(e['loss_hours']['value']),
           f'{what}: largest per-instance loss of the policy in hours (instance '
           f'{e["loss_hours"]["instance"]}, {e["loss_hours"]["combination"]})')
        nc(f'{PREFIX}GainHours{r}', hours(e['gain_hours']['value']),
           f'{what}: largest per-instance gain of the policy in hours, negative (instance '
           f'{e["gain_hours"]["instance"]}, {e["gain_hours"]["combination"]})')
        nc(f'{PREFIX}LossPct{r}', F88.f2(e['loss_pct']['value']),
           f'{what}: largest per-instance loss in percent (instance {e["loss_pct"]["instance"]}, '
           f'{e["loss_pct"]["combination"]})')
        nc(f'{PREFIX}GainPct{r}', F88.f2(e['gain_pct']['value']),
           f'{what}: largest per-instance gain in percent, negative (instance '
           f'{e["gain_pct"]["instance"]}, {e["gain_pct"]["combination"]})')
    return L


def share_text(c):
    return f'{c["shorter"]:.1f} / {c["equal"]:.1f} / {c["longer"]:.1f}'


def markdown(res):
    md = [f'# The principal comparison in counts and in hours ({res["setting"]})', '']
    if res['dry']:
        md += ['DRY RUN on synthetic arrays.', '']
    rp = res['reproduction']
    md += [f'- seeds {", ".join(map(str, res["seeds"]))}: {res["seeds_reason"]}',
           f'- policy minus specialist, seed s against seed s, {res["n_instances"]} test instances '
           f'per combination; makespans '
           + ('in integer hours' if res['integer_makespans'] else 'not all integer hours'),
           f'- pair shares: over (seed pair, instance), equal = the same makespan; instance shares: '
           f'on the per-instance means over the seeds, equal = |difference| < {EQUAL_H:g} h',
           f'- hours: mean difference with the hierarchical 95% interval of p88 section A and the '
           f'seed-pooled instance-level interval, B = {res["boot"]}; {res["streams"]}',
           f'- reproduction: {rp["five"]["compared"]} numbers and words of section A in '
           f'{rp["five"]["record"]} identical at the printed precision'
           + (f'; {rp["ten"]["compared"]} numbers of {rp["ten"]["record"]} identical'
              if 'ten' in rp else ''), '',
           '| entry | pairs: shorter / equal / longer, % | instances: shorter / equal / longer, % '
           '| mean, h | hierarchical 95% CI, h | instance-level 95% CI, h | mean, % | policy, h '
           '| specialist, h |', '|---|---|---|---|---|---|---|---|---|']
    for r, x in res['rows'].items():
        h, hi = x['hours'], x['hours_instance_level']
        md.append(f'| {entry_text(r)} | {share_text(x["pairs"])} | {share_text(x["instances"])} | '
                  f'{h["mean"]:+.2f} | [{h["lo"]:+.2f}, {h["hi"]:+.2f}] {h["direction"]} | '
                  f'[{hi["lo"]:+.2f}, {hi["hi"]:+.2f}] | {F88.f2(x["pct_mean"])} | '
                  f'{x["policy_mean_h"]:.1f} | {x["specialist_mean_h"]:.1f} |')
    md += ['', '## Largest per-instance loss and gain of the policy (means over the seeds)', '',
           '| entry | loss, h (instance) | gain, h (instance) | loss, % (instance) | '
           'gain, % (instance) |', '|---|---|---|---|---|']
    for r, x in res['rows'].items():
        e = x['extremes']

        def cell(q, pct=False):
            v = e[q]
            return (f'{F88.f2(v["value"]) if pct else hours(v["value"])} '
                    f'({v["combination"]} {v["instance"]})')
        md.append(f'| {entry_text(r)} | {cell("loss_hours")} | {cell("gain_hours")} | '
                  f'{cell("loss_pct", True)} | {cell("gain_pct", True)} |')
    md += ['', '## Counts', '', '| entry | pairs | shorter | equal | longer | instances | shorter '
           '| equal | longer |', '|---|---|---|---|---|---|---|---|---|']
    for r, x in res['rows'].items():
        p, i = x['pairs'], x['instances']
        md.append(f'| {entry_text(r)} | {p["n"]} | {p["n_shorter"]} | {p["n_equal"]} | '
                  f'{p["n_longer"]} | {i["n"]} | {i["n_shorter"]} | {i["n_equal"]} | '
                  f'{i["n_longer"]} |')
    return '\n'.join(md) + '\n'


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(root=ROOT, dry=False):
    """analyse and write; a refusal removes the outputs. Returns (res, names, outs)."""
    P110.check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, dry)
        lines = macro_lines(res)
        names = check_names(lines, outs['macros'])
        emit(outs['macros'], lines)
        os.makedirs(os.path.dirname(outs['json']), exist_ok=True)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
        with open(outs['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, outs


# ----------------------------------------------------------------- dry mode
def dry_run(tree, boot=DRY_BOOT, ten=False):
    """p110's synthetic tree; ten: p110 is run first and the ten-seed branch is taken,
    otherwise one array of a second seed is removed and the five-seed branch is taken.
    Returns (res, names, outs)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    keep = (F88.B, F88.ALPHA)
    try:
        with P110.environment(**P110.dry_paths(tree)):
            P110.build_synthetic(tree, boot=boot)
            if ten:
                P110.run(root=tree, dry=True, boot=boot)
            else:
                os.remove(SE.result_path('specialist', 'W', SE.SECOND_SEEDS[0], 'W'))
            return run(root=tree, dry=True)
    finally:
        F88.configure(*keep)


def dry_expected(tree, seeds):
    """The pooled counts and hours of the synthetic arrays, read with numpy alone."""
    with P110.environment(**P110.dry_paths(tree)):
        dh = np.stack([np.stack([SE.load('released', None, s, r) - SE.load('specialist', r, s, r)
                                 for s in seeds]) for r in R8], axis=2)
    return dict(mean=float(dh.mean()), shorter=float((dh < 0).mean() * 100),
                longer=float((dh > 0).mean() * 100))


def dry_verify(res, tree):
    bad = []
    want = dry_expected(tree, res['seeds'])
    x = res['rows']['All']
    if abs(x['hours']['mean'] - want['mean']) > 1e-9:
        bad.append(f'All: mean {x["hours"]["mean"]}, built {want["mean"]}')
    for k in ('shorter', 'longer'):
        if abs(x['pairs'][k] - want[k]) > 1e-9:
            bad.append(f'All: {k} {x["pairs"][k]}, built {want[k]}')
    for r, y in res['rows'].items():
        if abs(sum(y['pairs'][k] for k in ('shorter', 'equal', 'longer')) - 100) > 1e-9:
            bad.append(f'{r}: pair shares do not sum to 100')
        if not y['hours']['lo'] <= y['hours']['mean'] <= y['hours']['hi']:
            bad.append(f'{r}: the mean lies outside its interval')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description='the principal comparison in counts and in hours')
    ap.add_argument('--check', action='store_true', help='list every missing input')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes B of '
                         f'the five-seed record')
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.dry is not None:
        if a.check:
            ap.error('--check and --dry exclude each other')
        boot = DRY_BOOT if a.boot is None else a.boot
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p125_dry_')
        msg = []
        try:
            for ten in (False, True):
                sub = os.path.join(tree, 'ten' if ten else 'five')
                res, names, _ = dry_run(sub, boot, ten)
                bad = dry_verify(res, sub)
                want = 'ten' if ten else 'five'
                if res['branch'] != want:
                    bad.append(f'branch {res["branch"]}, want {want}')
                if bad:
                    raise Refused('; '.join(bad[:6]))
                x = res['rows']['All']
                msg.append(f'{res["branch"]} seeds: {len(names)} macros, all eight '
                           f'{hours(x["hours"]["mean"])} h [{hours(x["hours"]["lo"])}, '
                           f'{hours(x["hours"]["hi"])}], pairs shorter {x["pairs"]["shorter"]:.1f}%')
        except Refused as e:
            sys.exit(f'p125 self-test on synthetic arrays FAILED: {e}')
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
        print(f'p125 self-test on synthetic arrays passed: B = {boot}; ' + '; '.join(msg)
              + (f'; outputs in {tree}' if keep else ''))
        return 0
    if a.boot is not None:
        ap.error('--boot is for the dry run; a real run uses B of the five-seed record')
    if a.check:
        return check()
    REG.banner()
    try:
        res, names, outs = run()
    except Refused as e:
        print(f'*** REFUSING TO CERTIFY *** {e}', file=sys.stderr)
        return 1
    x = res['rows']['All']
    print(f'p125: {res["branch"]} seeds; all eight: pairs shorter / equal / longer '
          f'{share_text(x["pairs"])} %, instances {share_text(x["instances"])} %, mean '
          f'{x["hours"]["mean"]:+.2f} h [{x["hours"]["lo"]:+.2f}, {x["hours"]["hi"]:+.2f}]; '
          f'{len(names)} macros; wrote ' + ', '.join(shown(p) for p in outs.values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
