r"""P130: the timed comparison on its nine instances against all 27 evidence instances.

QUESTION
  At the production batches 50x25n30 and 80x25n30 the arms of the timed comparison that
  involve the policy ran on the instances 003-011 of every combination (nine per combination,
  the design of scripts/p94_timed_inference.py). The two searches started from the schedule
  of the best dispatching rule (memetic_rule@600, cpsat_rule@600) have valid records on all 27
  evidence instances (003-029). This script sets comparison (i) of p94, row Sel, on the nine
  instances beside the same comparison on the 27, at 60, 300 and 600 s of one clock: the
  makespan of the policy's greedy schedule (selected training run) minus that of the search,
  per instance 100 (policy - search) / search, so a positive difference means the search has
  the shorter schedule. It computes no new statistic: every mean, interval and direction is
  p94's, read from two of its records, and the script sets them side by side.

INPUTS (two tagged records of p94: --tag, full mode, start or search clock)
  under --records-root (default the repository root):
    design    results/timed_inference_<--design-tag>.json, default tag startnine, p94 run with
              --campaign "50x25n30 80x25n30 = memetic_rule@600:3-11 cpsat_rule@600:3-11"
    evidence  results/timed_inference_<--evidence-tag>.json, default tag startall, the same
              with the instance range 3-29 in both tokens
  Each record's entry results.<size>.i.<method>.<budget>.selected.<entry>: mean, lo, hi,
  direction, n (instances) and c (combinations).

GUARDS (any failure refuses: nothing is written, and the outputs of an earlier run are removed)
  * each record: generator p94, run full, not partial, not a stand-in, its field tag the
    requested tag, the eight combinations, a clock of start or search, the budgets 60, 300 and
    600 s among its budgets, and the setting of this run (PACT_ENV, PACT_SEEDS);
  * the two records: the same selected training run (seed and policy text), the same clock,
    the same budgets, setting, training seeds, CP-SAT configuration, bootstrap size and alpha,
    the same sizes, and at every size the same two arms memetic_rule and cpsat_rule, each with
    the same budget in both campaigns;
  * instances: in each record every arm has the instances of its size; the evidence record's
    instances are the evidence instances of the size (timed_campaign_plan.DEFAULT_INSTANCES),
    the design record's are a proper subset of them, and each set holds the same instances at
    every size;
  * every entry exists in both records with n the number of instances of its record, c the
    number of combinations of the entry and the direction of its interval
    (stats_family.direction).

QUANTITIES (per size, search, budget and entry; entries: the eight combinations, Pool, the
combinations without setups (NoSetup) and with setups (Setup), as in p94)
  design, evidence  mean, lo, hi and direction of the entry in the record of each set
  shift             the design mean minus the evidence mean, in percentage points
  same_direction    the two direction words are equal
  flip              one direction is ahead and the other behind
  Per level (pooled: Pool, NoSetup and Setup; pool: Pool alone; single: the eight
  combinations; all: every entry): the number of entries, of entries with equal direction
  words and of flips, and the largest absolute shift with the entry where it occurs (the
  first in the order of the outputs when two are equal).

OUTPUTS (arm_registry.out under PACT_OUT, or under --out-root)
  manuscript/macros_timedsets.tex     macro_io.emit; names Tis<letters>, each with a comment
                                     (what, which instances, which clock); the prefix Tis must
                                     be defined by no other .tex file under manuscript/
    Tis<Size><Method><Budget><Entry><Set>      mean difference (p89 f2); Set Design (the
                                               nine instances) or Evidence (the 27)
    ...<Set>Lo, ...<Set>Hi, ...<Set>Dir        ends of the 95% interval and its direction
    Tis<Size><Method><Budget><Entry>Shift      design mean minus evidence mean
    Tis<Level>EntryN, Tis<Level>SameDirN,      entries, entries with equal direction words,
    Tis<Level>FlipN, Tis<Level>ShiftMax        flips, largest absolute shift (two decimals,
                                               no sign); Level Pooled, Pool, Single or empty
                                               (every entry)
    TisDesignInstN, TisEvidenceInstN           instances per combination of each set
    TisBudgetShortMin, TisBudgetMidMin,        the three budgets in minutes
    TisBudgetLongMin
    Size Fifty or Eighty, Method Mem or Cp, Budget Sixty, ThreeHundred or SixHundred (p94's
    words); Entry N, L, S, W, LS, LW, SW, LSW, Pool, NoSetup or Setup.
  results/timed_instance_sets.json   every entry of both sets, the shifts, the summary, the
                                     inputs with their SHA-256
  results/timed_instance_sets.md     tables for reading

CHECK (--check): reads both records and applies every guard; writes nothing; exit 1 while a
record is missing or a guard fails.

Usage (repository root):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p130_timed_instance_sets.py
  ... --check
  ... --design-tag <letters> --evidence-tag <letters> [--records-root DIR] [--out-root DIR]
"""
import argparse
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
_ARGV = list(sys.argv)          # p94 imports p97, which empties sys.argv at import

import arm_registry as REG  # noqa: E402
import p94_timed_inference as T  # noqa: E402
import stats_family as SF  # noqa: E402
from macro_io import emit  # noqa: E402
from p89_public_family import f2  # noqa: E402

P = T.P
GENERATOR = 'scripts/p130_timed_instance_sets.py'
P94_GENERATOR = 'scripts/p94_timed_inference.py'
PREFIX = 'Tis'
OUT_FILES = {'macros': 'manuscript/macros_timedsets.tex',
             'json': 'results/timed_instance_sets.json',
             'md': 'results/timed_instance_sets.md'}
# the two instance sets: (key, macro word, default tag of the p94 record, text)
SETS = (('design', 'Design', 'startnine', 'the instances of the timed comparison'),
        ('evidence', 'Evidence', 'startall', 'every evidence instance'))
ARMS = ('memetic_rule', 'cpsat_rule')       # the two searches started from the rule schedule
METHODS = tuple(P.ARMS[a][0] for a in ARMS)   # memetic, cpsat
BUDGETS = (60, 300, 600)
BUDGET_WORD = dict(zip(BUDGETS, ('Short', 'Mid', 'Long')))
CLOCKS = ('start', 'search')
COMPARISON, ROW = 'i', 'selected'           # p94's comparison (i), row Sel
COMBOS = tuple(T.COMBOS)
POOLED = (T.POOL,) + T.group_keys()
ENTRIES = COMBOS + POOLED
GROUP_MEMBERS = {g[0]: g[2] for g in T.GROUPS}
# (key, macro word, the entries of the level)
LEVELS = (('pooled', 'Pooled', POOLED), ('pool', 'Pool', (T.POOL,)),
          ('single', 'Single', COMBOS), ('all', '', ENTRIES))
# the fields the two records must share
SHARED = ('origin', 'origin_text', 'budgets', 'setting', 'seeds', 'cpsat_arm', 'boot', 'alpha',
          'policy', 'combinations')
ENTRY_FIELDS = ('mean', 'lo', 'hi', 'direction', 'n', 'c')
COMPARISON_TEXT = ("the makespan of the policy's greedy schedule (selected training run) minus "
                   "that of the search started from the rule schedule, per instance 100 (policy "
                   "- search) / search, mean over the instances and combinations of the entry "
                   "(p94 comparison (i), row Sel; a positive difference means the search has the "
                   "shorter schedule)")
SHIFT_TEXT = ('the mean on the instances of the timed comparison (design) minus the mean on '
              'every evidence instance (evidence), in percentage points')
Refusal = T.Refusal


# ----------------------------------------------------------------- small helpers
def shown(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else os.path.basename(a)


def record_path(root, tag):
    return os.path.join(root, dict(T.TAG_OUTPUTS)['json'].format(tag=tag))


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def level_of(entry):
    return 'pooled' if entry in POOLED else 'single'


def n_combinations(entry):
    if entry == T.POOL:
        return len(COMBOS)
    return len(GROUP_MEMBERS[entry]) if entry in GROUP_MEMBERS else 1


def evidence_instances(size):
    return P.parse_instances(P.DEFAULT_INSTANCES.get(size, P.OTHER_DEFAULT_INSTANCES))


def campaign_arms(J):
    """{size: [(arm, budget text)]} of the campaigns of a record, sorted."""
    out = {}
    for spec in J.get('campaigns', []):
        sizes, tokens = T.parse_campaign(spec)
        for s in sizes:
            out.setdefault(s, [])
            for tok in tokens:
                arm, _, rest = tok.partition('@')
                out[s].append((arm, rest.partition(':')[0]))
    return {s: sorted(v) for s, v in out.items()}


# ----------------------------------------------------------------- reading and guards
def read_records(a):
    """{set key: (path, JSON or None)} and the problems of the files."""
    recs, problems = {}, []
    for key, _, _, _ in SETS:
        path = record_path(a.records_root, a.tags[key])
        if not os.path.exists(path):
            problems.append(f'{key}: {shown(path)} missing (p94 --tag {a.tags[key]})')
            recs[key] = (path, None)
            continue
        try:
            with open(path) as f:
                recs[key] = (path, json.load(f))
        except (OSError, ValueError) as e:
            problems.append(f'{key}: {shown(path)} unreadable ({e})')
            recs[key] = (path, None)
    return recs, problems


def record_problems(key, J, tag, setting):
    """The guards on one record."""
    bad = []
    want = dict(generator=P94_GENERATOR, run='full', partial=False, stand_in=False, tag=tag,
                combinations=list(COMBOS), setting=setting)
    for field, value in want.items():
        if J.get(field) != value:
            bad.append(f'{key}: {field} is {J.get(field)!r}, want {value!r}')
    if J.get('origin') not in CLOCKS:
        bad.append(f'{key}: clock {J.get("origin")!r}, want one of {CLOCKS}')
    missing = [b for b in BUDGETS if b not in (J.get('budgets') or [])]
    if missing:
        bad.append(f'{key}: budgets {J.get("budgets")} lack {missing}')
    if not J.get('results'):
        bad.append(f'{key}: no results')
    return bad


def pair_problems(D, E):
    """The guards on the two records together (design D, evidence E)."""
    bad = []
    for field in SHARED:
        if D.get(field) != E.get(field):
            bad.append(f'{field} differs: design {D.get(field)!r}, evidence {E.get(field)!r}')
    sd, se = (J.get('selection', {}).get('seed') for J in (D, E))
    if sd is None or sd != se:
        bad.append(f'selected training run differs or is missing: design {sd}, evidence {se}')
    sizes_d, sizes_e = list(D.get('results', {})), list(E.get('results', {}))
    if sizes_d != sizes_e:
        bad.append(f'sizes differ: design {sizes_d}, evidence {sizes_e}')
    unknown = [s for s in sizes_d if s not in T.SIZE_WORD]
    if unknown:
        bad.append(f'sizes {unknown} have no macro word')
    ca_d, ca_e = campaign_arms(D), campaign_arms(E)
    for s in sizes_d:
        for key, J, ca in (('design', D, ca_d), ('evidence', E, ca_e)):
            arms = J['results'].get(s, {}).get('arms')
            if sorted(arms or []) != sorted(ARMS):
                bad.append(f'{key}: {s} has the arms {arms}, want {list(ARMS)}')
            if sorted(a for a, _ in ca.get(s, [])) != sorted(ARMS):
                bad.append(f'{key}: the campaign of {s} has the tokens {ca.get(s)}, want one '
                           f'token of each of {list(ARMS)}')
        if ca_d.get(s) != ca_e.get(s):
            bad.append(f'{s}: the arm budgets differ: design {ca_d.get(s)}, evidence {ca_e.get(s)}')
    return bad


def instance_problems(D, E):
    """The instances of each record and their relation. Returns (problems, {set: {size:
    instances}})."""
    bad, inst = [], {'design': {}, 'evidence': {}}
    for key, J in (('design', D), ('evidence', E)):
        for s, S in J['results'].items():
            have = list(S.get('instances') or [])
            inst[key][s] = have
            for arm in ARMS:
                got = (S.get('design') or {}).get(arm, {}).get('instances')
                if got != have:
                    bad.append(f'{key}: {s} {arm} has the instances {got}, the size {have}')
    for s in inst['evidence']:
        evid, des = inst['evidence'][s], inst['design'].get(s, [])
        if evid != evidence_instances(s):
            bad.append(f'evidence: {s} has the instances {P.compress(evid)}, the evidence '
                       f'instances are {P.compress(evidence_instances(s))}')
        if not des or not set(des) < set(evid):
            bad.append(f'design: {s} instances {P.compress(des) or "none"} are not a proper '
                       f'subset of the evidence instances {P.compress(evid)}')
    for key in inst:
        if len({tuple(v) for v in inst[key].values()}) > 1:
            bad.append(f'{key}: the instances differ between the sizes: '
                       + ', '.join(f'{s} {P.compress(v)}' for s, v in inst[key].items()))
    return bad, inst


def entry_of(J, size, method, budget):
    return (J['results'].get(size, {}).get(COMPARISON, {}).get(method, {}).get(str(budget), {})
            .get(ROW, {}))


def entry_problems(key, J, inst):
    bad = []
    for s, S in J['results'].items():
        for m in METHODS:
            for b in BUDGETS:
                block = entry_of(J, s, m, b)
                for r in ENTRIES:
                    x = block.get(r)
                    where = f'{key}: {s} {m} {b} s {r}'
                    if x is None or any(x.get(f) is None for f in ENTRY_FIELDS):
                        bad.append(f'{where}: entry or a field of {ENTRY_FIELDS} missing')
                        continue
                    if x['n'] != len(inst[key][s]):
                        bad.append(f'{where}: n {x["n"]}, the record has {len(inst[key][s])} '
                                   f'instances')
                    if x['c'] != n_combinations(r):
                        bad.append(f'{where}: c {x["c"]}, want {n_combinations(r)}')
                    if x['direction'] != SF.direction(x['lo'], x['hi']):
                        bad.append(f'{where}: direction {x["direction"]} is not that of '
                                   f'[{x["lo"]}, {x["hi"]}]')
    return bad


def guarded(a):
    """Both records and every guard. Returns (records {set: (path, JSON)}, instances);
    raises Refusal with every problem found."""
    recs, problems = read_records(a)
    if problems:
        raise Refusal(problems)
    setting = REG.describe()
    D, E = recs['design'][1], recs['evidence'][1]
    for key, _, _, _ in SETS:
        problems += record_problems(key, recs[key][1], a.tags[key], setting)
    if problems:
        raise Refusal(problems)
    problems += pair_problems(D, E)
    if problems:
        raise Refusal(problems)
    bad, inst = instance_problems(D, E)
    if bad:
        raise Refusal(bad)
    problems = entry_problems('design', D, inst) + entry_problems('evidence', E, inst)
    if problems:
        raise Refusal(problems)
    return recs, inst


# ----------------------------------------------------------------- the comparison
def kept(x):
    return {f: x[f] for f in ENTRY_FIELDS}


def compare_sets(D, E):
    """{size: {method: {budget: {entry: pair}}}} and the summary per level."""
    res, rows = {}, []
    for s in D['results']:
        for m in METHODS:
            for b in BUDGETS:
                bd, be = entry_of(D, s, m, b), entry_of(E, s, m, b)
                for r in ENTRIES:
                    d, e = bd[r], be[r]
                    pair = dict(design=kept(d), evidence=kept(e), shift=d['mean'] - e['mean'],
                                same_direction=d['direction'] == e['direction'],
                                flip={d['direction'], e['direction']} == {'ahead', 'behind'})
                    res.setdefault(s, {}).setdefault(m, {}).setdefault(str(b), {})[r] = pair
                    rows.append(dict(size=s, method=m, budget_s=b, entry=r, **pair))
    summary = {}
    for key, _, members in LEVELS:
        sel = [x for x in rows if x['entry'] in members]
        top = max(sel, key=lambda x: abs(x['shift']))
        summary[key] = dict(
            entries=len(sel), same_direction=sum(x['same_direction'] for x in sel),
            differ=sum(not x['same_direction'] for x in sel), flips=sum(x['flip'] for x in sel),
            largest_abs_shift=abs(top['shift']),
            largest_at=dict(size=top['size'], method=top['method'], budget_s=top['budget_s'],
                            entry=top['entry'], shift=top['shift']))
    differing = [dict(size=x['size'], method=x['method'], budget_s=x['budget_s'],
                      entry=x['entry'], design=x['design'], evidence=x['evidence'],
                      flip=x['flip'])
                 for x in rows if not x['same_direction']]
    return res, summary, differing


# ----------------------------------------------------------------- outputs
def macro_lines(res):
    inst, origin = res['instances'], res['origin']
    sets = {key: (word, f'instances {P.compress(next(iter(inst[key].values())))}')
            for key, word, _, _ in SETS}
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% Inputs: ' + ' and '.join(f'{x["path"]} ({key})' for key, x in res['records'].items())
         + f', records of {P94_GENERATOR} --tag; {res["setting"]}; selected training run '
         f's{res["selected_run"]}; CP-SAT configuration {res["cpsat_configuration"]}.',
         f'% Each difference: {COMPARISON_TEXT}. 95% interval: instance-level cluster bootstrap '
         f'of p94 (B = {res["boot"]}, alpha {res["alpha"]}, fresh default_rng({T.INST_SEED}) per '
         f'quantity); direction from the unrounded interval (ahead: below zero, the policy '
         f'shorter; behind: above zero; neither otherwise).',
         f'% Clock origin {origin}: {T.ORIGIN_TEXT[origin]}.',
         f'% Design: {sets["design"][1]} ({res["n_instances"]["design"]} per combination); '
         f'Evidence: {sets["evidence"][1]} ({res["n_instances"]["evidence"]} per combination); '
         f'Shift: {SHIFT_TEXT}.']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    for key, word, _, text in SETS:
        nc(f'{PREFIX}{word}InstN', res['n_instances'][key],
           f'instances per combination, {text} ({sets[key][1]}), the same at every size')
    for b in BUDGETS:
        nc(f'{PREFIX}Budget{BUDGET_WORD[b]}Min', f'{b / 60:g}',
           f'reported budget of {b} s in minutes, {origin} clock')
    for s, by_m in res['results'].items():
        sw = T.SIZE_WORD[s]
        for m, by_b in by_m.items():
            mw = T.METHOD_WORD[m]
            for b, by_r in by_b.items():
                bw = T.number_word(int(b))
                for r, x in by_r.items():
                    base = f'{PREFIX}{sw}{mw}{bw}{r}'
                    what = (f'{s}, {T.METHOD_TEXT[m]} started from the rule schedule, {b} s on '
                            f'the {origin} clock, {T.entry_text(r)}')
                    for key, word, _, _ in SETS:
                        y = x[key]
                        cells = (f'{sets[key][1]}, {y["n"]} instances x {y["c"]} '
                                 f'combination{"s" if y["c"] > 1 else ""}')
                        nc(base + word, f2(y['mean']),
                           f'{what}, {cells}: mean difference, policy minus search')
                        nc(base + word + 'Lo', f2(y['lo']),
                           f'{what}, {cells}: lower end of the 95% interval')
                        nc(base + word + 'Hi', f2(y['hi']),
                           f'{what}, {cells}: upper end of the 95% interval')
                        nc(base + word + 'Dir', SF.TEX[y['direction']],
                           f'{what}, {cells}: direction of [{y["lo"]:+.4f}, {y["hi"]:+.4f}]')
                    nc(base + 'Shift', f2(x['shift']),
                       f'{what}: mean on {sets["design"][1]} minus mean on '
                       f'{sets["evidence"][1]}, percentage points')
    for key, word, members in LEVELS:
        sm = res['summary'][key]
        lv = {'pooled': 'pooled entries (Pool and the two setup groups)',
              'pool': 'the entry Pool (all eight combinations)',
              'single': 'single combinations', 'all': 'every entry'}[key]
        span = (f'{lv}, over the sizes, both searches and the budgets '
                f'{", ".join(map(str, BUDGETS))} s, {origin} clock')
        at = sm['largest_at']
        nc(f'{PREFIX}{word}EntryN', sm['entries'], f'{span}: entries')
        nc(f'{PREFIX}{word}SameDirN', sm['same_direction'],
           f'{span}: entries whose direction word is the same on both instance sets')
        nc(f'{PREFIX}{word}FlipN', sm['flips'],
           f'{span}: entries ahead on one instance set and behind on the other')
        nc(f'{PREFIX}{word}ShiftMax', f'{sm["largest_abs_shift"]:.2f}',
           f'{span}: largest absolute shift, percentage points ({at["size"]}, '
           f'{T.METHOD_TEXT[at["method"]]}, {at["budget_s"]} s, {T.entry_text(at["entry"])}: '
           f'{f2(at["shift"])})')
    return L


def defined_macros(own):
    """{name: file} of every macro defined in a .tex file under manuscript/ (the repository's and
    the output root's), the files of the output's name left out."""
    import glob
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)'
                    r'|\\def\s*\\([A-Za-z@]+)')
    roots = {os.path.join(ROOT, 'manuscript'), os.path.dirname(os.path.abspath(own))}
    out = {}
    for d in sorted(roots):
        for f in sorted(glob.glob(os.path.join(d, '**', '*.tex'), recursive=True)):
            if os.path.basename(f) == os.path.basename(own):
                continue
            text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                             for x in open(f, errors='replace').read().splitlines())
            for x, y in rx.findall(text):
                out.setdefault(x or y, shown(f))
    return out


def check_names(lines, own):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(rf'{PREFIX}[A-Za-z]+', x)]
    if bad:
        raise Refusal([f'macro names that are not {PREFIX} plus letters: {bad[:5]}'])
    if len(names) != len(set(names)):
        dup = sorted({x for x in names if names.count(x) > 1})
        raise Refusal([f'duplicate macro names: {dup[:5]}'])
    have = defined_macros(own)
    taken = sorted(x for x in have if x.startswith(PREFIX) and x[len(PREFIX):][:1].isupper())
    if taken:
        raise Refusal([f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})'])
    return names


def cell_text(y):
    return f'{f2(y["mean"])} [{f2(y["lo"])}, {f2(y["hi"])}] {y["direction"]}'


def markdown(res):
    inst = res['instances']
    head = {key: P.compress(next(iter(inst[key].values()))) for key, _, _, _ in SETS}
    md = ['# Timed comparison: the instances of the design against every evidence instance', '',
          f'- inputs: ' + '; '.join(f'{key} {x["path"]} (tag {x["tag"]}, sha256 {x["sha256"]})'
                                    for key, x in res['records'].items()),
          f'- {res["setting"]}; selected training run s{res["selected_run"]}; CP-SAT '
          f'configuration {res["cpsat_configuration"]}; B = {res["boot"]}, alpha {res["alpha"]}',
          f'- difference: {COMPARISON_TEXT}',
          f'- clock origin {res["origin"]} ({T.ORIGIN_TEXT[res["origin"]]})',
          f'- design: instances {head["design"]} ({res["n_instances"]["design"]} per '
          f'combination); evidence: instances {head["evidence"]} '
          f'({res["n_instances"]["evidence"]} per combination); shift: {SHIFT_TEXT}', '']
    for s, by_m in res['results'].items():
        for m, by_b in by_m.items():
            md += [f'## {s}, {T.METHOD_TEXT[m]} started from the rule schedule', '',
                   f'| s | entry | instances {head["design"]} | instances {head["evidence"]} | '
                   f'shift | same direction |', '|---|---|---|---|---|---|']
            for b, by_r in by_b.items():
                for r, x in by_r.items():
                    md.append(f'| {b} | {T.entry_text(r)} | {cell_text(x["design"])} | '
                              f'{cell_text(x["evidence"])} | {f2(x["shift"])} | '
                              f'{"yes" if x["same_direction"] else "no"} |')
            md.append('')
    md += ['## Summary', '', '| level | entries | same direction | flips (ahead against behind) | '
           'largest absolute shift | where |', '|---|---|---|---|---|---|']
    for key, _, _ in LEVELS:
        sm = res['summary'][key]
        at = sm['largest_at']
        md.append(f'| {key} | {sm["entries"]} | {sm["same_direction"]} | {sm["flips"]} | '
                  f'{sm["largest_abs_shift"]:.2f} | {at["size"]} {T.METHOD_TEXT[at["method"]]} '
                  f'{at["budget_s"]} s {T.entry_text(at["entry"])} ({f2(at["shift"])}) |')
    md += ['', f'## Entries whose direction word differs ({len(res["differing"])})', '']
    md += [f'- {x["size"]} {T.METHOD_TEXT[x["method"]]} {x["budget_s"]} s '
           f'{T.entry_text(x["entry"])}: {cell_text(x["design"])} | {cell_text(x["evidence"])}'
           + (' (flip)' if x['flip'] else '') for x in res['differing']]
    return '\n'.join(md) + '\n'


def output_paths(a):
    if a.out_root:
        return {k: os.path.join(a.out_root, v) for k, v in OUT_FILES.items()}
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(paths):
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)


# ----------------------------------------------------------------- the run
def run(a):
    """Guards, comparison and the three outputs; a refusal removes them. Returns (result,
    macro names, paths)."""
    paths = output_paths(a)
    remove_outputs(paths)
    try:
        recs, inst = guarded(a)
        D, E = recs['design'][1], recs['evidence'][1]
        results, summary, differing = compare_sets(D, E)
        res = dict(generator=GENERATOR, setting=D['setting'], seeds=D['seeds'],
                   selected_run=D['selection']['seed'], policy=D['policy'],
                   cpsat_configuration=D['cpsat_arm'], origin=D['origin'],
                   origin_text=D['origin_text'], budgets=list(BUDGETS), boot=D['boot'],
                   alpha=D['alpha'], comparison=COMPARISON_TEXT, shift=SHIFT_TEXT,
                   arms=list(ARMS), sizes=list(D['results']), entries=list(ENTRIES),
                   levels={key: list(members) for key, _, members in LEVELS},
                   records={key: dict(path=shown(recs[key][0]), tag=a.tags[key],
                                      sha256=sha256(recs[key][0]),
                                      campaigns=recs[key][1]['campaigns'])
                            for key, _, _, _ in SETS},
                   instances=inst,
                   n_instances={key: len(next(iter(inst[key].values()))) for key in inst},
                   results=results, summary=summary, differing=differing)
        lines = macro_lines(res)
        names = check_names(lines, paths['macros'])
        emit(paths['macros'], lines)
        for key in ('json', 'md'):
            os.makedirs(os.path.dirname(paths[key]) or '.', exist_ok=True)
        with open(paths['json'], 'w') as f:
            json.dump(T.jsonable(res), f, indent=1)
        with open(paths['md'], 'w') as f:
            f.write(markdown(res))
    except BaseException:
        remove_outputs(paths)
        raise
    return res, names, paths


def build_parser():
    ap = argparse.ArgumentParser(description='the timed comparison on its nine instances against '
                                             'all 27 evidence instances')
    for key, _, tag, text in SETS:
        ap.add_argument(f'--{key}-tag', default=tag,
                        help=f'tag of the p94 record of {text} (default {tag})')
    ap.add_argument('--records-root', default=ROOT,
                    help='root of results/timed_inference_<tag>.json (default the repository)')
    ap.add_argument('--out-root', default=None,
                    help='root of manuscript/ and results/ (default PACT_OUT through arm_registry.out)')
    ap.add_argument('--check', action='store_true',
                    help='read both records and apply every guard; write nothing')
    return ap


def parse(argv):
    """The options, with the tags per set in a.tags and absolute roots; exits on a bad tag."""
    a = build_parser().parse_args(argv)
    a.tags = {key: getattr(a, f'{key}_tag') for key, _, _, _ in SETS}
    bad = [f'--{k}-tag {t!r}: letters only' for k, t in a.tags.items()
           if not T.TAG_PATTERN.fullmatch(t)]
    if a.tags['design'] == a.tags['evidence']:
        bad.append('the two tags name the same record')
    if bad:
        sys.exit('REFUSING: ' + '; '.join(bad))
    a.records_root = os.path.abspath(a.records_root)
    if a.out_root:
        a.out_root = os.path.abspath(a.out_root)
    return a


def main(argv=None):
    a = parse(_ARGV[1:] if argv is None else argv)
    os.chdir(ROOT)
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    if REG.env() != 'exact':
        sys.exit(f'REFUSING: PACT_ENV={REG.env()}: the timed records are of the exact environment')
    if a.check:
        try:
            recs, inst = guarded(a)
        except Refusal as e:
            for x in e.args[0][:40]:
                print(f'p130: {x}')
            print('p130: not ready')
            return 1
        print('p130: both records present and consistent: ' + '; '.join(
            f'{key} {shown(recs[key][0])} ('
            + ', '.join(f'{s} {P.compress(v)}' for s, v in inst[key].items()) + ')'
            for key, _, _, _ in SETS))
        return 0
    if not a.out_root and os.environ.get('PACT_OUT') is None:
        sys.exit('REFUSING: set PACT_OUT (the output root of the chain, e.g. out/exact)')
    REG.banner()
    try:
        res, names, paths = run(a)
    except Refusal as e:
        print(f'*** REFUSING TO CERTIFY *** {len(e.args[0])} problem(s)', file=sys.stderr)
        for x in e.args[0][:40]:
            print(f'  {x}', file=sys.stderr)
        return 1
    sm = res['summary']
    print(f'p130: {len(names)} macros; direction word the same in {sm["all"]["same_direction"]} '
          f'of {sm["all"]["entries"]} entries (pooled {sm["pooled"]["same_direction"]} of '
          f'{sm["pooled"]["entries"]}), flips {sm["all"]["flips"]}, largest absolute shift '
          f'Pool {sm["pool"]["largest_abs_shift"]:.2f}, pooled entries '
          f'{sm["pooled"]["largest_abs_shift"]:.2f}, single '
          f'{sm["single"]["largest_abs_shift"]:.2f}; wrote '
          + ', '.join(shown(p) for p in paths.values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
