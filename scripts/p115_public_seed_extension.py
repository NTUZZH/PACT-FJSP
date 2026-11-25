r"""P115: the policy against the specialists on the benchmark from public instances under the
active mask, with ten training seeds, 301 to 310.

STATUS: AN ADDITION AFTER THE FIVE-SEED RESULT WAS KNOWN
  The comparison was read with the first five training seeds by
  scripts/p89_public_family.py --family active (results/public_family_act.json,
  manuscript/macros_pubfam_act.tex). After that result was known, the second five seeds 306 to
  310 of the policy and the specialists were added (scripts/pub_seed_extension.py). The
  five-seed record stays the reading of record and is not changed; the ten-seed reading is
  reported beside it as an added analysis, and its intervals are not adjusted for the look at
  the five-seed result. Every output states this in its first lines.

WHAT IS COMPUTED
  Section 1 of p89 on pub_SD2_test_10x5+mix (the policy against the specialist of each
  combination) and nothing else: per combination d = 100 (policy - specialist) / specialist,
  seed s against seed s and instance i against instance i; p89.component (the crossed
  hierarchical bootstrap over training seeds and base instances, the seed-level t interval,
  the verdicts at the 0.5% and 1% margins of stats_family.verdict, the direction);
  p89.iut_block (the claims over all eight combinations, the six training combinations and the
  two held-out ones); p89.pooled over the same three families. The readers are p89's too
  (p89.Inputs: records, best-known table, arrays, summaries, bounds, training records), so
  nothing of the estimator or of the checks is written twice. Random draws: B of the five-seed
  record and p89's own stream, a fresh default_rng(11) per quantity, for the five seeds and
  for the ten.

PRECONDITIONS (checked before anything is written; any failure refuses and no output stays)
  1. the five-seed record under --root was written by p89 --family active with seeds 301 to
     305, its margins and its stream, and holds section 1 for pub_SD2_test_10x5+mix;
  2. all 45 runs of the second five seeds are complete (pub_seed_extension.state);
  3. p89's checks on the ten seeds: the records and the best-known table of the test set, every
     array [n, 2] finite and positive, every summary (feasibility, means, the gap on N), no
     makespan on N below a proven bound, the records older than the arrays, every training
     record (29602 parameters, exact mask, active mask, validation lines), the validation norms
     of each recipe equal over the ten seeds;
  4. the five-seed rows recomputed here from the arrays equal the stored record: every number
     of the JSON at the printed precision (two decimals; ties at one), every word, direction
     and claim exactly, and every macro of the comparison (PubFamActSdtenPar...) in
     manuscript/macros_pubfam_act.tex character for character.

OUTPUT (arm_registry.out under PACT_OUT)
  manuscript/macros_pubseedext.tex     every name starts with PubSeedExt and holds letters only,
                                      and is defined in no other .tex file under manuscript/ or
                                      in paper/macros_*.tex
  results/public_seed_extension.json  per combination and pooled the ten-seed rows, the
                                      five-seed rows as stored, the mean of the first and of the
                                      second five seeds, the per-seed means, the claims
  results/public_seed_extension.md    the same as tables, with the status in its first lines

DRY MODE (--dry [DIR])
  The whole path on synthetic arrays in DIR (default a temporary folder, removed afterwards):
  the records of the test set and its best-known table linked from the repository, synthetic
  arrays, summaries and training records of the ten seeds, and a five-seed record written by
  p89's functions on the synthetic first five seeds. Nothing is read from the result or
  training folders of the repository, and nothing is written outside DIR.

Usage: PACT_OUT=<root> python scripts/p115_public_seed_extension.py [--root D]
       python scripts/p115_public_seed_extension.py --dry [DIR] [--boot 2000]
"""
import argparse
import contextlib
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
import p89_public_family as P89  # noqa: E402
import pub_seed_extension as PSE  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p115_public_seed_extension.py'
PREFIX = 'PubSeedExt'
OUT_FILES = {'macros': 'manuscript/macros_pubseedext.tex',
             'json': 'results/public_seed_extension.json',
             'md': 'results/public_seed_extension.md'}
FAM = P89.Fam(PSE.RULE)
RECORD_JSON, RECORD_MD, RECORD_MACROS = FAM.out_files
RECORD_GENERATOR = f'scripts/p89_public_family.py --family {PSE.RULE}'
HEADLINE = PSE.HEADLINE
assert HEADLINE == P89.EQ_HEADLINE
SET_WORD = P89.SETS[HEADLINE][0]                 # Sdten
STORED_PREFIX = f'{FAM.prefix}{SET_WORD}Par'     # PubFamActSdtenPar
FAMILIES = {'All': list(REG.COMBOS), 'Train': list(P89.TRAIN), 'Held': list(P89.HELD)}
R8 = list(REG.COMBOS)
MARGINS = P89.MARGINS
MW = P89.MW
FIRST, SECOND, ALL = PSE.FIRST_SEEDS, PSE.SECOND_SEEDS, PSE.ALL_SEEDS
DRY_BOOT = 2000
STATUS = ('ADDED AFTER THE FIVE-SEED RESULT WAS KNOWN. The second five training seeds 306 to 310 '
          'of the policy and of the specialists were trained after the five-seed result of this '
          'comparison was known. The five-seed record of scripts/p89_public_family.py --family '
          'active (results/public_family_act.json, manuscript/macros_pubfam_act.tex) stays the '
          'reading of record; the ten-seed reading is reported beside it.')
NOT_ADJUSTED = ('The ten-seed intervals are not adjusted for the look at the five-seed result '
                'that preceded them.')
Refused = PSE.Refused


# ----------------------------------------------------------------- the stored record
def read_record(root):
    """The five-seed record of p89 --family active under root, checked: (json, {macro: value})."""
    jp, mp = os.path.join(root, RECORD_JSON), os.path.join(root, RECORD_MACROS)
    for p in (jp, mp):
        if not os.path.exists(p):
            raise Refused(f'the five-seed record {os.path.relpath(p, root)} does not exist')
    try:
        rec = json.load(open(jp))
    except (OSError, ValueError) as e:
        raise Refused(f'{RECORD_JSON} is not readable JSON: {e}')
    if rec.get('generator') != RECORD_GENERATOR:
        raise Refused(f'{RECORD_JSON} was written by {rec.get("generator")!r}, not {RECORD_GENERATOR}')
    if [int(s) for s in rec.get('seeds') or []] != list(FIRST):
        raise Refused(f'{RECORD_JSON} holds the seeds {rec.get("seeds")}, not {list(FIRST)}')
    if [float(x) for x in rec.get('margins') or []] != list(MARGINS):
        raise Refused(f'{RECORD_JSON} holds the margins {rec.get("margins")}, not {list(MARGINS)}')
    stream = f'fresh default_rng({P89.HIER_SEED}) per quantity'
    if (rec.get('streams') or {}).get('hierarchical') != stream:
        raise Refused(f'{RECORD_JSON} records the hierarchical stream '
                      f'{(rec.get("streams") or {}).get("hierarchical")!r}, not {stream!r}')
    if not isinstance(rec.get('boot'), int) or rec['boot'] < 1:
        raise Refused(f'{RECORD_JSON} records B = {rec.get("boot")!r}')
    par = ((rec.get('sets') or {}).get(HEADLINE) or {}).get('parity')
    if not par or set(par) != {'components', 'iut', 'pooled'} or \
            set(par['components']) != set(R8) or set(par['pooled']) != set(FAMILIES):
        raise Refused(f'{RECORD_JSON} holds no section 1 for {HEADLINE} with the eight combinations '
                      f'and the families {list(FAMILIES)}')
    text = open(mp, errors='replace').read()
    if not text.startswith(f'% Generated by scripts/p89_public_family.py -- do not hand-edit '
                           f'(--family {PSE.RULE})'):
        raise Refused(f'{RECORD_MACROS} was not written by {RECORD_GENERATOR}')
    return rec, macro_values(text)


def macro_values(text):
    """{name: value} of every \\newcommand of a macro file or of macro lines."""
    out = {}
    for n, v in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}', text):
        if n in out:
            raise Refused(f'the macro {n} is defined twice')
        out[n] = v
    return out


def stored_macro_lines(comp, iut, pool):
    """The macro lines of section 1 on the test set as p89 --family active writes them, from
    p89's own line writers (comp_lines, pool_lines, iut_lines, dir_line, rename_line)."""
    L, w = [], SET_WORD
    for r in R8:
        c = comp[r]
        P89.comp_lines(L, f'Pub{w}Par{r}', c, f'{HEADLINE}, policy vs specialist, {r}')
        L.append(f'\\newcommand{{\\Pub{w}Par{r}SeedMin}}{{{P89.f2(c["seed_min"])}}}'
                 f'\\newcommand{{\\Pub{w}Par{r}SeedMax}}{{{P89.f2(c["seed_max"])}}}'
                 f'\\newcommand{{\\Pub{w}Par{r}Tie}}{{{P89.u1(c["ties"])}}}')
    for fam in FAMILIES:
        P89.pool_lines(L, f'Pub{w}ParPool{fam}', pool[fam], f'{HEADLINE}, policy vs specialist, {fam}')
    P89.iut_lines(L, f'Pub{w}ParIut', iut, tuple(FAMILIES))
    for fam in FAMILIES:
        P89.dir_line(L, f'Pub{w}ParPool{fam}', pool[fam], f'{HEADLINE}, policy vs specialist, {fam}')
    return [P89.rename_line(x, FAM) for x in L]


# ----------------------------------------------------------------- the inputs
def check_setting():
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    if PSE.results_root() != REG.PUB_RESULTS:
        raise Refused(f'PUB_RESULTS={PSE.results_root()}: p89 reads the arrays under '
                      f'{REG.PUB_RESULTS} of --root, so the second five seeds are read there too')


def load_inputs(root):
    """Every array of the comparison for the ten seeds, checked with p89's readers:
    (I, {(recipe, combo, seed, R): makespans}, evidence)."""
    I = P89.Inputs(root, FAM)
    I.check_records(HEADLINE)
    n = len(I.names[HEADLINE])
    ms, oldest, ev = {}, np.inf, {}
    for arm in PSE.ARMS:
        for c in (R8 if arm == 'specialist' else [None]):
            for s in ALL:
                nm = PSE.name(arm, c, s)
                for r in PSE.regimes_read(arm, c):
                    p = os.path.join(I.folder(HEADLINE), f'Result_greedy+{REG.PUB_PREFIX}{nm}_{r}.npy')
                    if os.path.normpath(p) != os.path.normpath(os.path.join(root, PSE.result_path(arm, c, s, r))):
                        raise Refused(f'the array of {nm} on {r}: p89 reads {p}, '
                                      f'pub_seed_extension names {PSE.result_path(arm, c, s, r)}')
                    ms[(arm, c, s, r)], t = I._load(p, n)
                    oldest = min(oldest, t)
                regs = PSE.regimes_read(arm, c)
                I.check_summary(HEADLINE, f'summary_{REG.PUB_PREFIX}{nm}{P89.XS}.md',
                                {r: f'greedy+{REG.PUB_PREFIX}{nm}' for r in regs},
                                lambda r, m, a=arm, c_=c, s_=s: ms[(a, c_, s_, r)], evidence=ev)
    if not I.evidence[HEADLINE]['records_newest_mtime'] <= oldest:
        raise Refused(f'{HEADLINE}: a descriptor record is newer than a result file, so the runs '
                      f'may not have read one sorted record list')
    I.check_lower_bounds(HEADLINE, vecs=[v for k, v in ms.items() if k[3] == 'N'], evidence=ev)
    ev['files_read'] = I.n_files
    return I, ms, ev


def training_records(I, root):
    """p89's check of every training record of the ten seeds, and the validation norms of each
    recipe, equal over the ten seeds. Returns {recipe stem: norms}."""
    norms = {}
    for arm in PSE.ARMS:
        for c in (R8 if arm == 'specialist' else [None]):
            per = {}
            for s in ALL:
                _, lp, vp, _ = PSE.training_paths(arm, c, s)
                I.training_record(arm, c, s, paths=(lp, vp))
                try:
                    per[s] = REG.validation_record(arm, c, s, paths=(os.path.join(root, vp),
                                                                     os.path.join(root, lp)))[2]
                except (SystemExit, REG.MissingResult) as e:
                    raise Refused(f'validation record of {PSE.name(arm, c, s)}: {e}')
            first = per[ALL[0]]
            if not first:
                raise Refused(f'no MWKR norm line in the training log of {PSE.name(arm, c, ALL[0])}')
            diff = {s: v for s, v in per.items() if v != first}
            if diff:
                raise Refused(f'the validation norms of {PSE.stem(arm, c)} differ from seed '
                              f'{ALL[0]} {first}: {diff}')
            norms[PSE.stem(arm, c)] = first
    return norms


# ----------------------------------------------------------------- the comparison
def section_one(ms, seeds, B):
    """({R: d [k, n]}, {R: p89.component}, p89.iut_block, {family: p89.pooled}) on the seeds."""
    D = {r: np.stack([P89.rel(ms[('released', None, s, r)], ms[('specialist', r, s, r)])
                      for s in seeds]) for r in R8}
    C = {r: P89.component(D[r], B) for r in R8}
    return D, C, P89.iut_block(C, FAMILIES), {f: P89.pooled([D[r] for r in regs], B)
                                              for f, regs in FAMILIES.items()}


def reproduction(par, C, iut, pool, stored_macros):
    """Differences of the recomputed five-seed rows from the stored record at the printed
    precision; returns (numbers and words compared, macros compared, [differences], largest
    absolute difference of a number)."""
    bad, n, dmax = [], 0, 0.0

    def num(where, a, b, fmt=P89.f2):
        nonlocal n, dmax
        n += 1
        dmax = max(dmax, abs(float(a) - float(b)))
        if fmt(a) != fmt(b):
            bad.append(f'{where}: stored {fmt(b)}, recomputed {fmt(a)}')

    def word(where, a, b):
        nonlocal n
        n += 1
        if a != b:
            bad.append(f'{where}: stored {b!r}, recomputed {a!r}')
    for r in R8:
        s, c = par['components'][r], C[r]
        if set(s) != set(c):
            bad.append(f'{r}: stored keys {sorted(set(s) - set(c))} / recomputed keys '
                       f'{sorted(set(c) - set(s))} differ')
            continue
        for k in ('mean', 'lo', 'hi', 't_lo', 't_hi', 'seed_sd', 'seed_min', 'seed_max',
                  'p_t_less', 'p_boot_less'):
            num(f'{r} {k}', c[k], s[k])
        num(f'{r} ties', c['ties'], s['ties'], P89.u1)
        word(f'{r} n', int(c['n']), int(s['n']))
        word(f'{r} seeds', len(c['seed_means']), len(s['seed_means']))
        for i, (a, b) in enumerate(zip(c['seed_means'], s['seed_means'])):
            num(f'{r} seed mean {i + 1}', a, b)
        for k in [f'v{mg}{x}' for mg in MARGINS for x in ('', '_t')] + ['direction']:
            word(f'{r} {k}', c[k], s[k])
    if set(iut) != set(par['iut']):
        bad.append(f'claims: stored {sorted(par["iut"])}, recomputed {sorted(iut)}')
    for k in sorted(set(iut) & set(par['iut'])):
        x, y = iut[k], par['iut'][k]
        for f in ('holds', 'failing', 'components', 'margin'):
            word(f'claim {k} {f}', P89.jsonable(x[f]), y[f])
    for fam in FAMILIES:
        x, y = pool[fam], par['pooled'][fam]
        for k in ('mean', 'lo', 'hi', 't_lo', 't_hi'):
            num(f'pooled {fam} {k}', x[k], y[k])
        for i, (a, b) in enumerate(zip(x['seed_means'], y['seed_means'])):
            num(f'pooled {fam} seed mean {i + 1}', a, b)
        for k in ('c', 'v0.5', 'v1.0', 'direction'):
            word(f'pooled {fam} {k}', x[k], y[k])
    mine = macro_values('\n'.join(stored_macro_lines(C, iut, pool)))
    theirs = {k: v for k, v in stored_macros.items() if k.startswith(STORED_PREFIX)}
    if set(mine) != set(theirs):
        bad.append(f'macros of section 1: stored {sorted(set(theirs) - set(mine))[:4]} / '
                   f'recomputed {sorted(set(mine) - set(theirs))[:4]} have no counterpart')
    for k in sorted(set(mine) & set(theirs)):
        if mine[k] != theirs[k]:
            bad.append(f'\\{k}: stored {theirs[k]}, recomputed {mine[k]}')
    return n, len(set(mine) & set(theirs)), bad, dmax


def analyse(root, boot=None, dry=False):
    """Every precondition, then the ten-seed rows; returns the result dict. Raises Refused."""
    check_setting()
    root = os.path.abspath(root)
    rec, stored_macros = read_record(root)
    B = int(rec['boot']) if boot is None else int(boot)
    if boot is not None and not dry:
        raise Refused('B is the B of the five-seed record outside the dry run')
    st = PSE.state(root)
    miss = [x for x in st if not x['complete']]
    if miss:
        x = miss[0]
        why = (x['training_problems'] + x['evaluation_problems'])[0].replace(root + os.sep, '')
        raise Refused(f'{len(miss)} of {len(st)} runs of the second five seeds are not complete '
                      f'(first: {x["name"]}: {why}); see scripts/pub_seed_extension.py status')
    try:
        I, ms, ev = load_inputs(root)
        norms = training_records(I, root)
    except (P89.Refusal, REG.MissingResult) as e:
        raise Refused(str(e).replace('REFUSING: ', ''))
    par = rec['sets'][HEADLINE]['parity']
    D5, C5, iut5, pool5 = section_one(ms, FIRST, B)
    n_cmp, n_mac, bad, dmax = reproduction(par, C5, iut5, pool5, stored_macros)
    if bad:
        raise Refused(f'the five-seed rows do not reproduce the stored record {RECORD_JSON} and '
                      f'{RECORD_MACROS} ({len(bad)} differences): ' + '; '.join(bad[:6]))
    D10, C10, iut10, pool10 = section_one(ms, ALL, B)
    k5 = len(FIRST)
    for r in R8:
        if not np.array_equal(D10[r][:k5], D5[r]):
            raise Refused(f'{r}: the first five seeds of the ten-seed differences are not the '
                          f'five-seed differences')
    first = {r: float(D10[r][:k5].mean()) for r in R8}
    second = {r: float(D10[r][k5:].mean()) for r in R8}
    return dict(generator=GENERATOR, status=STATUS, not_adjusted=NOT_ADJUSTED,
                benchmark='the benchmark from public instances', test_set=HEADLINE,
                mask=f'active mask (mask_mode {PSE.RULE}; exact first-decision and tie options)',
                comparison='the policy against the specialist of each combination; d = 100 '
                           '(policy - specialist) / specialist, seed s against seed s',
                seeds=list(ALL), first_seeds=list(FIRST), second_seeds=list(SECOND), n=int(D10['N'].shape[1]),
                boot=B, alpha=0.05, margins=list(MARGINS),
                streams=dict(hierarchical=f'fresh default_rng({P89.HIER_SEED}) per quantity '
                                          f'(p89.component and p89.pooled), five and ten seeds'),
                record=dict(json=RECORD_JSON, macros=RECORD_MACROS, generator=RECORD_GENERATOR,
                            boot=int(rec['boot']),
                            reproduction=dict(compared=n_cmp, macros_compared=n_mac, identical=True,
                                              largest_abs_difference=dmax)),
                validation_norms=norms, files_read=ev['files_read'],
                summaries_checked=ev.get('summaries_checked', 0), dry=bool(dry),
                ten=dict(components=C10, iut=iut10, pooled=pool10),
                five=dict(components=par['components'], iut=par['iut'], pooled=par['pooled']),
                first_five_mean=first, second_five_mean=second,
                verdict_changes={r: {f'v{mg}': dict(five=par['components'][r][f'v{mg}'],
                                                    ten=C10[r][f'v{mg}']) for mg in MARGINS}
                                 for r in R8})


# ----------------------------------------------------------------- outputs
def tex(v):
    return SF.TEX[v]


def macro_lines(res):
    f2, B = P89.f2, res['boot']
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {STATUS}',
         f'% {NOT_ADJUSTED}',
         f'% {res["benchmark"].capitalize()}, {HEADLINE}, {res["mask"]}: {res["comparison"]}; '
         f'training seeds {ALL[0]} to {ALL[-1]}; estimator of p89 (component, pooled, iut_block), '
         f'B = {B}, {res["streams"]["hierarchical"]}. The Five names repeat the five-seed record, '
         f'recomputed here from the arrays and equal to it at the printed digits.',
         f'\\newcommand{{\\{PREFIX}SeedN}}{{{len(ALL)}}}'
         f'\\newcommand{{\\{PREFIX}FiveSeedN}}{{{len(FIRST)}}}'
         f'\\newcommand{{\\{PREFIX}Boot}}{{{B}}}'
         f'  % training seeds of the ten-seed rows and of the record, bootstrap replicates']
    ten, five = res['ten'], res['five']
    for r in R8:
        c, s = ten['components'][r], five['components'][r]
        L.append(f'\\newcommand{{\\{PREFIX}{r}}}{{{f2(c["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Lo}}{{{f2(c["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}Hi}}{{{f2(c["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}{r}V{MW[mg]}}}{{{tex(c[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}{r}Dir}}{{{tex(c["direction"])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}FirstMean}}{{{f2(res["first_five_mean"][r])}}}'
                 f'\\newcommand{{\\{PREFIX}{r}SecondMean}}{{{f2(res["second_five_mean"][r])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}{r}Seed{PSE.SEED_WORDS[sd]}}}{{{f2(m)}}}'
                           for sd, m in zip(ALL, c['seed_means']))
                 + f'  % {r}, ten seeds: mean, hierarchical 95% CI, verdicts at 0.5 and 1, '
                 f'direction, mean of the first and of the second five seeds, per seed')
        L.append(f'\\newcommand{{\\{PREFIX}Five{r}}}{{{f2(s["mean"])}}}'
                 f'\\newcommand{{\\{PREFIX}Five{r}Lo}}{{{f2(s["lo"])}}}'
                 f'\\newcommand{{\\{PREFIX}Five{r}Hi}}{{{f2(s["hi"])}}}'
                 + ''.join(f'\\newcommand{{\\{PREFIX}Five{r}V{MW[mg]}}}{{{tex(s[f"v{mg}"])}}}'
                           for mg in MARGINS)
                 + f'\\newcommand{{\\{PREFIX}Five{r}Dir}}{{{tex(s["direction"])}}}'
                 f'  % {r}, the first five seeds as stored')
    for key, blk in (('', ten['iut']), ('Five', five['iut'])):
        for fam in FAMILIES:
            for mg in MARGINS:
                for claim, cw in (('equivalent', 'Eq'), ('non-inferior', 'Ni')):
                    x = blk[f'{fam}|{mg}|{claim}']
                    L.append(f'\\newcommand{{\\{PREFIX}{key}Iut{fam}{MW[mg]}{cw}}}'
                             f'{{{int(x["holds"])}}}  % claim over several combinations, '
                             f'{"the first five seeds as stored" if key else "ten seeds"}, '
                             f'{claim} on {",".join(x["components"])} at +-{mg}%: failing '
                             f'{",".join(x["failing"]) or "none"}')
    for key, pool in (('', ten['pooled']), ('Five', five['pooled'])):
        for fam in FAMILIES:
            x = pool[fam]
            L.append(f'\\newcommand{{\\{PREFIX}{key}Pool{fam}}}{{{f2(x["mean"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Lo}}{{{f2(x["lo"])}}}'
                     f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Hi}}{{{f2(x["hi"])}}}'
                     + ''.join(f'\\newcommand{{\\{PREFIX}{key}Pool{fam}V{MW[mg]}}}'
                               f'{{{tex(x[f"v{mg}"])}}}' for mg in MARGINS)
                     + f'\\newcommand{{\\{PREFIX}{key}Pool{fam}Dir}}'
                     f'{{{tex(SF.direction(x["lo"], x["hi"]))}}}'
                     f'  % pooled over {fam} ({len(FAMILIES[fam])} combinations), '
                     f'{"the first five seeds as stored" if key else "ten seeds"}: mean, '
                     f'hierarchical 95% CI, verdicts, direction')
    return L


def defined_macros(roots):
    """{name: [files]} of every macro defined in a .tex file under manuscript/ (any depth) or in
    paper/macros_*.tex of each root, the output file itself left out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    own = os.path.basename(OUT_FILES['macros'])
    out = {}
    for root in roots:
        fs = []
        for d, _, names in os.walk(os.path.join(root, 'manuscript')):
            fs += [os.path.join(d, f) for f in names if f.endswith('.tex')]
        pd = os.path.join(root, 'paper')
        if os.path.isdir(pd):
            fs += [os.path.join(pd, f) for f in os.listdir(pd)
                   if f.startswith('macros') and f.endswith('.tex')]
        for f in sorted(fs):
            if os.path.basename(f) == own:
                continue
            text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                             for x in open(f, errors='replace').read().splitlines())
            for a, b in rx.findall(text):
                out.setdefault(a or b, []).append(os.path.relpath(f, root))
    return out


def check_names(lines, roots):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros(roots)
    clash = sorted(x for x in names if x in have)
    if clash:
        raise Refused('macro names already defined: '
                      + ', '.join(f'\\{x} ({have[x][0]})' for x in clash[:5]))
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by {taken[0]} ({have[taken[0]][0]})')
    return names


def ci(x):
    return f'[{P89.f2(x["lo"])}, {P89.f2(x["hi"])}]'


def markdown(res):
    f2 = P89.f2
    ten, five = res['ten'], res['five']
    rp = res['record']['reproduction']
    md = ['# The policy against the specialists with ten training seeds: the benchmark from '
          'public instances under the active mask', '',
          f'**{STATUS}**', '', NOT_ADJUSTED, '',
          f'Test set {HEADLINE} ({res["n"]} instances), {res["mask"]}; {res["comparison"]}. '
          f'Training seeds {ALL[0]} to {ALL[-1]}: the first five seeds {FIRST[0]} to {FIRST[-1]} '
          f'and the second five seeds {SECOND[0]} to {SECOND[-1]}. Estimator, margins and verdicts '
          f'of p89 (crossed hierarchical bootstrap over training seeds and base instances, '
          f'B = {res["boot"]}, {res["streams"]["hierarchical"]}; verdicts at +-0.5% and +-1% from '
          f'the 95% interval). The five-seed rows recomputed from the arrays equal the stored record '
          f'(`{RECORD_JSON}`, `{RECORD_MACROS}`): {rp["compared"]} numbers and words at the printed '
          f'precision and {rp["macros_compared"]} macros character for character. Validation norms '
          f'of each recipe equal over the ten seeds.'
          + (' DRY RUN on synthetic arrays.' if res['dry'] else ''), '',
          '| combination | ten seeds: mean | 95% CI | 0.5% | 1% | direction | first five seeds '
          '(record): mean | 95% CI | 0.5% | 1% | direction | mean of the second five seeds |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in R8:
        c, s = ten['components'][r], five['components'][r]
        md.append(f'| {r} | {f2(c["mean"])} | {ci(c)} | {c["v0.5"]} | {c["v1.0"]} | '
                  f'{c["direction"]} | {f2(s["mean"])} | {ci(s)} | {s["v0.5"]} | {s["v1.0"]} | '
                  f'{s["direction"]} | {f2(res["second_five_mean"][r])} |')
    md += ['', 'Per-seed means (policy minus specialist over specialist, %):', '',
           '| combination | ' + ' | '.join(f's{s}' for s in ALL) + ' |',
           '|---|' + '---|' * len(ALL)]
    for r in R8:
        md.append(f'| {r} | ' + ' | '.join(f2(m) for m in ten['components'][r]['seed_means']) + ' |')
    md += ['', 'Claims over several combinations (intersection-union test):', '',
           '| combinations | margin | equivalent on all (ten / five) | non-inferior on all '
           '(ten / five) | failing, equivalent (ten / five) |', '|---|---|---|---|---|']
    for fam in FAMILIES:
        for mg in MARGINS:
            t = {c: ten['iut'][f'{fam}|{mg}|{c}'] for c in ('equivalent', 'non-inferior')}
            v = {c: five['iut'][f'{fam}|{mg}|{c}'] for c in ('equivalent', 'non-inferior')}
            md.append(f'| {fam} | {mg} | {t["equivalent"]["holds"]} / {v["equivalent"]["holds"]} '
                      f'| {t["non-inferior"]["holds"]} / {v["non-inferior"]["holds"]} | '
                      f'{",".join(t["equivalent"]["failing"]) or "-"} / '
                      f'{",".join(v["equivalent"]["failing"]) or "-"} |')
    md += ['', 'Pooled rows (combinations of a base instance together):', '',
           '| combinations | ten seeds: mean | 95% CI | 0.5% | 1% | first five seeds (record): '
           'mean | 95% CI | 0.5% | 1% |', '|---|---|---|---|---|---|---|---|---|']
    for fam in FAMILIES:
        x, u = ten['pooled'][fam], five['pooled'][fam]
        md.append(f'| {fam} | {f2(x["mean"])} | {ci(x)} | {x["v0.5"]} | {x["v1.0"]} | '
                  f'{f2(u["mean"])} | {ci(u)} | {u["v0.5"]} | {u["v1.0"]} |')
    return md


def outputs():
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def write(res, outs, roots):
    lines = macro_lines(res)
    names = check_names(lines, roots)
    emit(outs['macros'], lines)
    with open(outs['json'], 'w') as f:
        json.dump(P89.jsonable(res), f, indent=1)
    md = markdown(res)
    with open(outs['md'], 'w') as f:
        f.write('\n'.join(md) + '\n')
    return names, md


def run(root=ROOT, dry=False, boot=None):
    """analyse() and write(); a refusal removes the outputs. Returns (res, names, md)."""
    check_setting()
    outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, boot, dry)
        names, md = write(res, outs, sorted({ROOT, os.path.abspath(root)}))
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, md


# ----------------------------------------------------------------- dry mode
TRUE_GAP = {'N': 0.6, 'L': 0.3, 'S': -0.2, 'W': 0.1, 'LS': 0.4, 'LW': -0.3, 'SW': -0.5, 'LSW': 0.2}
SEED_EFFECT = [0.4, -0.3, 0.2, -0.2, 0.1, -0.1, 0.3, -0.4, 0.2, -0.2]     # mean zero, seeds 301-310
NOISE = 0.5
VALI = [0.95, 0.9, 0.88, 0.9]


@contextlib.contextmanager
def environment(**kv):
    """The environment with kv set (a value None removes the key), restored afterwards."""
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
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


def write_summary(path, model, rows, n, bk):
    """The table of the summary eval_family.py writes, the rows p89 reads; rows =
    [(combination, makespans)]."""
    out = [f'# FAMILY eval — {model}{P89.XS}', '', 'mask: first_mask=exact mask_ties=exact', '',
           '| combination | method | mean | std | feasible | gap(cov) | gap to best known (N only) |',
           '|---|---|---|---|---|---|---|']
    for r, v in rows:
        g = f'{np.mean(100 * (v - bk) / bk):.2f}% ({n})' if r == 'N' else 'n/a'
        out.append(f'| {r} | greedy+{model} | {v.mean():.1f} | {v.std():.1f} | {n}/{n} | n/a | {g} |')
    _put(path, '\n'.join(out) + '\n')


def synthetic_training(tree, arm, combo, seed):
    """Checkpoint, training log, validation record and configuration of one run."""
    nm = PSE.name(arm, combo, seed)
    regs = list(P89.TRAIN) if arm == 'released' else [combo]
    ck, lp, vp, cp = PSE.training_paths(arm, combo, seed)
    _put(os.path.join(tree, ck), '')
    _put(os.path.join(tree, vp), repr(VALI))
    lines = [PSE.BANNER, PSE.MASK_LINE,
             '[vali] MWKR norms: ' + '  '.join(f'{r}={300.0 + 10 * R8.index(r):.1f}' for r in regs),
             f'[model] params total={PSE.PARAMS}  film=0 (0.00%)']
    best = 9.0
    for k, sc in enumerate(VALI):
        best = min(best, sc)
        lines.append(f'[vali @{25 * (k + 1)}] rel={sc:.4f} (best {best:.4f})  '
                     + '  '.join(f'{r}={300 * sc:.1f}' for r in regs))
    lines.append(f'[train_family] done in 1.00 h; best rel={min(VALI):.4f}')
    _put(os.path.join(tree, lp), '\n'.join(lines) + '\n')
    _put(os.path.join(tree, cp), json.dumps(dict(
        model_suffix=nm, seed_train=int(seed), device='cpu', first_mask='exact', mask_ties='exact',
        mask_mode=PSE.RULE, pair_extra='none', reward_mode='dense', pub_type_source='none',
        family_base='sd2', n_m=5, use_type_embedding=False, family_regimes=','.join(regs),
        max_updates=PSE.RECIPE[arm][1]), indent=1))


def build_synthetic(tree, boot=DRY_BOOT):
    """The tree of the dry run (module docstring); returns its root."""
    tree = os.path.abspath(tree)
    for sub in (os.path.join('data', 'PUBLIC', 'family', HEADLINE),
                os.path.dirname(REG.PUB_SETS[HEADLINE])):
        dst = os.path.join(tree, sub)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, sub), dst)
    I = P89.Inputs(tree, FAM)
    I.check_records(HEADLINE)
    n, bk = len(I.names[HEADLINE]), I.bk[HEADLINE]['bk']
    rng = np.random.default_rng(2026)
    ms = {}
    for k, s in enumerate(ALL):
        for r in R8:
            spec = np.round(bk * (1.15 + 0.05 * R8.index(r)) * (1 + np.abs(rng.normal(0, 0.02, n))))
            d = TRUE_GAP[r] + SEED_EFFECT[k] + rng.normal(0, NOISE, n)
            ms[('specialist', r, s, r)] = spec
            ms[('released', None, s, r)] = np.round(spec * (1 + d / 100.0))
    for (arm, c, s, r), v in ms.items():
        p = os.path.join(tree, PSE.result_path(arm, c, s, r))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p, np.stack([v, np.full(n, 0.1)], axis=1))
    for arm in PSE.ARMS:
        for c in (R8 if arm == 'specialist' else [None]):
            for s in ALL:
                synthetic_training(tree, arm, c, s)
                m = REG.PUB_PREFIX + PSE.name(arm, c, s)
                write_summary(os.path.join(tree, PSE.summary_path(arm, c, s)), m,
                              [(r, ms[(arm, c, s, r)]) for r in PSE.regimes_read(arm, c)], n, bk)
    # the five-seed record of p89 --family active on the synthetic first five seeds
    _, C, iut, pool = section_one(ms, FIRST, boot)
    rec = dict(generator=RECORD_GENERATOR, seeds=list(FIRST), boot=int(boot), alpha=0.05,
               margins=list(MARGINS),
               streams=dict(hierarchical=f'fresh default_rng({P89.HIER_SEED}) per quantity',
                            instance=f'fresh default_rng({P89.INST_SEED}) per quantity'),
               sets={HEADLINE: dict(n=n, parity=dict(components=C, iut=iut, pooled=pool))})
    _put(os.path.join(tree, RECORD_JSON), json.dumps(P89.jsonable(rec), indent=1))
    os.makedirs(os.path.join(tree, os.path.dirname(RECORD_MACROS)), exist_ok=True)
    emit(os.path.join(tree, RECORD_MACROS),
         [f'% Generated by scripts/p89_public_family.py -- do not hand-edit (--family {PSE.RULE})']
         + stored_macro_lines(C, iut, pool))
    return tree


def dry_run(tree, boot=DRY_BOOT):
    """The whole path on a synthetic tree; returns (res, names, md)."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with environment(PACT_OUT=os.path.join(tree, 'out'), PUB_RESULTS=None):
        build_synthetic(tree, boot)
        return run(root=tree, dry=True, boot=boot)


def main(argv=None):
    ap = argparse.ArgumentParser(description='the policy against the specialists with ten seeds, '
                                             'benchmark from public instances (added after the '
                                             'five-seed result was known)')
    ap.add_argument('--root', default=ROOT, help='root of the inputs (default the repository)')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='run the whole path on synthetic arrays in DIR (default: a temporary '
                         'folder, removed afterwards)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes B of '
                         f'the five-seed record')
    a = ap.parse_args(argv)
    if a.dry is None:
        if a.boot is not None:
            ap.error('--boot is for the dry run; a real run uses B of the five-seed record')
        try:
            res, names, md = run(a.root)
        except Refused as e:
            sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
        print('\n'.join(md))
        print(f'\n{len(names)} macros; wrote ' + ', '.join(outputs().values()))
        return 0
    boot = DRY_BOOT if a.boot is None else a.boot
    keep = bool(a.dry)
    tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p115_dry_')
    try:
        res, names, md = dry_run(tree, boot)
    except Refused as e:
        sys.exit(f'p115 self-test on synthetic arrays FAILED: {e}')
    finally:
        if not keep:
            shutil.rmtree(tree, ignore_errors=True)
    if keep:
        print('\n'.join(md))
        print(f'\n{len(names)} macros; outputs in {os.path.join(tree, "out")}')
    else:
        print(f'p115 self-test on synthetic arrays passed: B = {boot}, {len(names)} macros, '
              f'{res["record"]["reproduction"]["compared"]} five-seed numbers and words and '
              f'{res["record"]["reproduction"]["macros_compared"]} macros reproduced, ten-seed rows '
              f'computed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
