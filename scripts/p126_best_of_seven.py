r"""P126: the policy against the per-instance best of seven dispatching rules.

QUESTION
  scripts/p93_rule_comparison.py compares one greedy schedule of the policy with the
  per-instance best of the four classical rules (FIFO, MOR, SPT, MWKR); scripts/p72_aware_rules.py
  compares the three rules that read the timing rules (ECT, SSPT, LLF) with the policy one at a
  time. This script compares the policy with the per-instance best of all seven, beside the best
  of the four from the same arrays, and counts how often each rule is the shortest.

WHAT IS READ (p72's own readers; p72 is imported after the environment is fixed)
  the policy  p72.pact: the per-instance mean of the greedy makespans over the training seeds
              found (the active seeds, 301 to 305), in p72's file order
  the rules   p72.stored: the seven rule arrays of the size's rule folder (p72.rule_dir); n is
              the number of instances of the three further rules, the policy and the classical
              rules are cut to n as p72's summary does
  sizes       p72.SIZES_EXACT (10x25, 20x25, 30x25, 50x25n30, 80x25n30); combinations
              p72.REGIMES

QUANTITIES (per size, per combination and pooled over the eight; d = 100 (a - b) / b,
negative = the policy shorter)
  seven   a = the policy, b = the per-instance shortest of the seven rules
  four    a = the policy, b = the per-instance shortest of FIFO, MOR, SPT, MWKR
  interval  p72.ci, the paired bootstrap of p72's summary (B = p72.N_BOOT, percentile 95%),
            with a fresh default_rng(7) set as p72's stream before each quantity; the pooled
            row concatenates the eight combinations as p72's summary does
  shortest  per combination and pooled, the percent of instances on which each rule's makespan
            equals the shortest of the seven (ties counted for each rule)
  per run   the inference protocol of p72 for the same two quantities (p72.inf_policy_rows:
            d per training run, crossed hierarchical bootstrap, Holm over the combinations and
            the combined verdict); its seven-rule rows must equal p72's stored inference record
            (results/aware_rules_inference.json, field policy_vs.best7)

CHECK BEFORE ANY NEW NUMBER (a failure refuses and removes the outputs)
  p72's summary for 10x25 replayed from these readers with p72.ci on p72's stream in p72's order
  reproduces every number of results/aware_rules.json for 10x25 exactly, and the text of
  results/aware_rules.md for "LLF vs PACT", pooled, 10x25.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_bestseven.tex  macro_io.emit; names Bsv<letters>; the prefix Bsv must be used
                                  by no other .tex file under manuscript/
  results/best_of_seven.json      every number
  results/best_of_seven.md        tables

DRY (--dry [DIR]): synthetic rule and policy arrays of the five sizes, a summary record written by
the replay, in DIR or a temporary folder, B = --boot (default 500).

Usage (repository root, one thread):
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p126_best_of_seven.py
  ... --check
  python scripts/p126_best_of_seven.py --dry [DIR]
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
sys.path.insert(0, ROOT)
_ARGV = list(sys.argv)          # p72 empties sys.argv at import

import numpy as np  # noqa: E402

GENERATOR = 'scripts/p126_best_of_seven.py'
PREFIX = 'Bsv'
OUT_FILES = {'macros': 'manuscript/macros_bestseven.tex', 'json': 'results/best_of_seven.json',
             'md': 'results/best_of_seven.md'}
INFERENCE_RECORD = 'results/aware_rules_inference.json'
REPRO_TEXT = ('LLF vs PACT', 'pooled', '10x25')
DRY_BOOT = 500
RULE_WORD = {'FIFO': 'Fifo', 'MOR': 'Mor', 'SPT': 'Spt', 'MWKR': 'Mwkr', 'ECT': 'Ect',
             'SSPT': 'Sspt', 'LLF': 'Llf'}
QUANT = {'seven': ('the per-instance best of the seven rules', 'Seven'),
         'four': ('the per-instance best of the four classical rules', 'Four')}
POOL = 'pool'


class Refused(Exception):
    """A check failed; no output stays."""


def load_p72():
    """p72, imported once the environment is fixed (it reads PACT_ENV at import and changes
    the working folder to the repository root)."""
    import arm_registry as REG
    if REG.env() != 'exact':
        raise Refused(f'PACT_ENV={REG.env()}: this analysis reads the exact environment')
    import p72_aware_rules as P72
    if P72.ENV != 'exact':
        raise Refused(f'p72 was imported in the {P72.ENV} environment')
    return P72



def setting_text():
    """The environment and the training seeds, in the words of the outputs."""
    import arm_registry as REG
    s = REG.seeds()
    return f'environment {REG.env()}, training seeds {",".join(map(str, s))}'

def shown(path):
    a = os.path.abspath(path)
    return os.path.relpath(a, ROOT) if a.startswith(ROOT + os.sep) else path


def stats():
    import stats_family as SF
    return SF


# ----------------------------------------------------------------- reading
def inputs(P72, sizes):
    """{size: {combination: dict(pol [n], P [k, n] per run, rules {rule: [n]}, files)}} and the
    missing inputs."""
    import arm_registry as REG
    miss, out = [], {}
    for size in sizes:
        S = {}
        for g in P72.REGIMES:
            rules = {r: P72.stored(size, r, g) for r in P72.NEW_RULES + P72.CLASSICAL}
            gone = [shown(P72.result_path(size, r, g)) for r, v in rules.items() if v is None]
            pol_files = [REG.result_path('released', None, s, size, g) for s in REG.seeds()]
            gone += [shown(p) for p in pol_files if not os.path.exists(p)]
            if gone:
                miss += gone
                continue
            n = len(rules[P72.NEW_RULES[0]])
            ns = {len(rules[r]) for r in P72.NEW_RULES}
            short = [r for r in P72.CLASSICAL if len(rules[r]) < n]
            if len(ns) != 1 or short:
                miss.append(f'{size}/{g}: the three further rules cover {sorted(ns)} instances, '
                            f'classical rules shorter than {n}: {short}')
                continue
            pol, names = P72.pact(size, g)
            if len(pol) < n:
                miss.append(f'{size}/{g}: the policy files hold {len(pol)} instances, fewer than {n}')
                continue
            P = np.stack([REG.read_path(p)[:n] for p in pol_files])
            S[g] = dict(pol=pol[:n], P=P, rules={r: v[:n] for r, v in rules.items()}, n=n,
                        files=names)
        out[size] = S
    return out, miss


# ----------------------------------------------------------------- reproduction
def replay_summary(P72, A, size):
    """p72's summary of one size, the first size it processes, with p72.ci on p72's stream
    from default_rng(7), in p72's order (per combination ECT, SSPT, LLF against SPT and then
    against the policy; then pooled in the same order). Returns the summary's numbers."""
    P72.rng = np.random.default_rng(7)
    out = {'regimes': {}, 'pooled': {}}
    pooled = {r: {'spt': [], 'pact': []} for r in P72.NEW_RULES}
    for g in P72.REGIMES:
        x = A[size][g]
        spt, pol = x['rules']['SPT'], x['pol']
        R = {}
        for r in P72.NEW_RULES:
            v = x['rules'][r]
            gs = 100 * (v - spt) / spt
            gp = 100 * (v - pol) / pol
            R[f'gap_SPT_{r}'] = list(P72.ci(gs))
            R[f'gap_PACT_{r}'] = list(P72.ci(gp))
            pooled[r]['spt'].append(gs)
            pooled[r]['pact'].append(gp)
        out['regimes'][g] = R
    for r in P72.NEW_RULES:
        out['pooled'][r] = {'gap_SPT': list(P72.ci(np.concatenate(pooled[r]['spt']))),
                            'gap_PACT': list(P72.ci(np.concatenate(pooled[r]['pact'])))}
    return out


def reproduce(P72, A):
    """The 10x25 numbers of p72's summary record and the Markdown text, exactly."""
    import arm_registry as REG
    size = '10x25'
    jp, mp = REG.out(P72.JSON_PATH), REG.out(P72.MD_PATH)
    for p in (jp, mp):
        if not os.path.exists(p):
            raise Refused(f'the summary record {shown(p)} of p72 does not exist')
    rec = json.load(open(jp))
    if list(rec.get('sizes', {}))[:1] != [size] or rec.get('seeds') != list(REG.seeds()) \
            or rec.get('rules') != list(P72.NEW_RULES):
        raise Refused(f'{shown(jp)}: sizes {list(rec.get("sizes", {}))}, seeds {rec.get("seeds")}, '
                      f'rules {rec.get("rules")}; want {size} first, the active seeds and '
                      f'{list(P72.NEW_RULES)}')
    mine = replay_summary(P72, A, size)
    S = rec['sizes'][size]
    bad, n = [], 0
    for g in P72.REGIMES:
        for k, v in mine['regimes'][g].items():
            n += 1
            if S['regimes'][g][k] != v:
                bad.append(f'{g} {k}: stored {S["regimes"][g][k]}, replayed {v}')
        if S['regimes'][g]['pact_files'] != A[size][g]['files']:
            bad.append(f'{g}: policy files {S["regimes"][g]["pact_files"]}, here {A[size][g]["files"]}')
    for r in P72.NEW_RULES:
        for k in ('gap_SPT', 'gap_PACT'):
            n += 1
            if S['pooled'][r][k] != mine['pooled'][r][k]:
                bad.append(f'pooled {r} {k}: stored {S["pooled"][r][k]}, replayed {mine["pooled"][r][k]}')
    text = P72._g(mine['pooled']['LLF']['gap_PACT'])
    md = open(mp).read()
    block = md.split(f'## {size}\n', 1)[1].split('\n## ', 1)[0] if f'## {size}\n' in md else ''
    row = next((x for x in block.splitlines() if x.startswith('| pooled |')), '')
    cells = [c.strip() for c in row.strip('|').split('|')]
    head = next((x for x in block.splitlines() if 'vs PACT' in x), '')
    cols = [c.strip() for c in head.strip('|').split('|')]
    have = cells[cols.index('LLF vs PACT')] if 'LLF vs PACT' in cols and len(cells) == len(cols) \
        else None
    if have != text:
        bad.append(f'{shown(mp)}: "LLF vs PACT", pooled, {size}: {have!r}, replayed {text!r}')
    if bad:
        raise Refused(f'the summary of p72 does not reproduce ({len(bad)} of {n + 1}): '
                      + '; '.join(bad[:6]))
    return dict(record=shown(jp), markdown=shown(mp), size=size, compared=n + 1,
                identical=True, number=dict(what='LLF vs PACT, pooled, 10x25',
                                            text=text, values=mine['pooled']['LLF']['gap_PACT']))


# ----------------------------------------------------------------- the analysis
def fresh_ci(P72, d, B):
    """p72.ci on a fresh default_rng(7)."""
    P72.rng = np.random.default_rng(7)
    m, lo, hi = P72.ci(np.asarray(d, dtype=float), B)
    SF = stats()
    return dict(mean=m, lo=lo, hi=hi, n=int(np.asarray(d).size), direction=SF.direction(lo, hi))


def shortest_shares(rules, best7):
    """Percent of instances on which each rule equals the best of seven (ties for each)."""
    return {r: float(np.mean(v == best7)) * 100 for r, v in rules.items()}


def analyse(P72, A, B, hier=True):
    st = P72._inf_stats()
    res = {}
    for size, S in A.items():
        X = {q: {} for q in QUANT}
        D = {q: {} for q in QUANT}
        rows = {q: {} for q in QUANT}
        short, means, ties = {}, {}, {}
        for g in P72.REGIMES:
            x = S[g]
            R = x['rules']
            best4 = np.min([R[r] for r in P72.CLASSICAL], axis=0)
            best7 = np.minimum(np.min([R[r] for r in P72.NEW_RULES], axis=0), best4)
            for q, b in (('seven', best7), ('four', best4)):
                X[q][g] = 100.0 * (x['pol'] - b) / b
                D[q][g] = 100.0 * (x['P'] - b[None, :]) / b[None, :]
                rows[q][g] = fresh_ci(P72, X[q][g], B)
            short[g] = shortest_shares(R, best7)
            ties[g] = float(np.mean(np.sum([R[r] == best7 for r in R], axis=0) > 1)) * 100
            means[g] = dict(policy=float(x['pol'].mean()), best7=float(best7.mean()),
                            best4=float(best4.mean()), n=int(x['n']))
        for q in QUANT:
            rows[q][POOL] = fresh_ci(P72, np.concatenate([X[q][g] for g in P72.REGIMES]), B)
        n_all = sum(means[g]['n'] for g in P72.REGIMES)
        short[POOL] = {r: sum(short[g][r] * means[g]['n'] for g in P72.REGIMES) / n_all
                       for r in short[P72.REGIMES[0]]}
        ties[POOL] = sum(ties[g] * means[g]['n'] for g in P72.REGIMES) / n_all
        means[POOL] = dict({k: sum(means[g][k] * means[g]['n'] for g in P72.REGIMES) / n_all
                            for k in ('policy', 'best7', 'best4')}, n=n_all)
        out = dict(n=means[P72.REGIMES[0]]['n'], seeds=None, files=S[P72.REGIMES[0]]['files'],
                   rule_folder=P72.rule_dir(size), rows=rows, shortest=short, ties_pct=ties,
                   means=means)
        if hier:
            out['per_run'] = {q: P72.inf_policy_rows(st, D[q], B) for q in QUANT}
        res[size] = out
    return res


def check_inference(P72, res):
    """The per-run seven-rule rows against p72's stored inference record (same B)."""
    p = os.path.join(ROOT, INFERENCE_RECORD)
    if not os.path.exists(p):
        raise Refused(f'the inference record {INFERENCE_RECORD} of p72 does not exist')
    rec = json.load(open(p))
    bad, n = [], 0
    for size, x in res.items():
        have = rec['sizes'].get(size, {}).get('policy_vs', {}).get('best7')
        if have is None:
            bad.append(f'{size}: no best7 rows in {INFERENCE_RECORD}')
            continue
        for g, row in x['per_run']['seven'].items():
            for k in ('mean', 'lo', 'hi'):
                n += 1
                if abs(have[g][k] - row[k]) > 1e-9:
                    bad.append(f'{size} {g} {k}: stored {have[g][k]}, here {row[k]}')
            if g != POOL:
                n += 1
                if have[g]['verdict'] != row['verdict']:
                    bad.append(f'{size} {g} verdict: stored {have[g]["verdict"]}, here {row["verdict"]}')
    if bad:
        raise Refused(f'the per-run rows do not reproduce {INFERENCE_RECORD} ({len(bad)} of {n}): '
                      + '; '.join(bad[:6]))
    return dict(record=INFERENCE_RECORD, compared=n, identical=True)


# ----------------------------------------------------------------- outputs
def f2(x):
    return f'{x:+.2f}'


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


def macro_lines(P72, out):
    SF = stats()
    L = [f'% Generated by {GENERATOR} -- do not hand-edit',
         f'% {out["setting"]}; d = 100 (policy - rule) / rule per instance, negative = the policy '
         f'shorter; the policy is the per-instance mean of the greedy makespans over the training '
         f'seeds {", ".join(map(str, out["seeds"]))} (p72.pact); Seven: the per-instance best of '
         f'FIFO, MOR, SPT, MWKR, ECT, SSPT and LLF; Four: of FIFO, MOR, SPT and MWKR.',
         f'% Intervals: p72.ci (paired bootstrap over instances, B = {out["boot"]}, fresh '
         f'default_rng(7) per quantity); pooled = the eight combinations concatenated. Hier: the '
         f'per-run protocol of p72 (hierarchical over training runs and base instances). '
         f'Shortest: percent of instances on which the rule is the shortest of the seven, ties '
         f'counted for each.']

    def nc(name, value, comment):
        L.append(f'\\newcommand{{\\{name}}}{{{value}}}  % {comment}')
    nc(f'{PREFIX}Boot', out['boot'], 'bootstrap resamples')
    for size, x in out['sizes'].items():
        sw = P72.INF_SIZE_WORDS[size]
        nc(f'{PREFIX}{sw}InstN', x['n'], f'{size}: instances per combination')
        for q, (text, qw) in QUANT.items():
            for g, c in x['rows'][q].items():
                gw = 'Pool' if g == POOL else g
                base = f'{PREFIX}{sw}{qw}{gw}'
                what = f'{size}, {"pooled" if g == POOL else g}: policy against {text}'
                nc(base, f2(c['mean']), f'{what}: mean, %')
                nc(base + 'Lo', f2(c['lo']), f'{what}: lower end of the 95% interval')
                nc(base + 'Hi', f2(c['hi']), f'{what}: upper end of the 95% interval')
                nc(base + 'Dir', SF.TEX[c['direction']],
                   f'{what}: direction of [{c["lo"]:+.4f}, {c["hi"]:+.4f}]')
            if 'per_run' in x:
                c = x['per_run'][q][POOL]
                base = f'{PREFIX}{sw}Hier{qw}Pool'
                what = f'{size}, pooled: policy against {text}, per training run'
                nc(base, f2(c['mean']), f'{what}: mean, %')
                nc(base + 'Lo', f2(c['lo']), f'{what}: lower end of the hierarchical 95% interval')
                nc(base + 'Hi', f2(c['hi']), f'{what}: upper end of the hierarchical 95% interval')
                nc(base + 'Dir', SF.TEX[c['direction']], f'{what}: direction')
                counts = {w: sum(x['per_run'][q][g]['verdict'] == w for g in P72.REGIMES)
                          for w in ('ahead', 'behind')}
                nc(f'{PREFIX}{sw}Hier{qw}AheadCount', counts['ahead'],
                   f'{size}: combinations (of 8) on which the policy is ahead of {text}, per run')
                nc(f'{PREFIX}{sw}Hier{qw}BehindCount', counts['behind'],
                   f'{size}: combinations (of 8) on which the policy is behind {text}, per run')
        for r, v in x['shortest'][POOL].items():
            nc(f'{PREFIX}{sw}Shortest{RULE_WORD[r]}', f'{v:.1f}',
               f'{size}, pooled: percent of instances on which {r} is the shortest of the seven')
        nc(f'{PREFIX}{sw}ShortestTies', f'{x["ties_pct"][POOL]:.1f}',
           f'{size}, pooled: percent of instances on which two or more rules share the shortest')
    return L


def cell(c):
    return f'{f2(c["mean"])} [{f2(c["lo"])}, {f2(c["hi"])}] {c["direction"]}'


def markdown(P72, out):
    md = [f'# The policy against the best of seven dispatching rules ({out["setting"]})', '']
    if out['dry']:
        md += ['DRY RUN on synthetic arrays.', '']
    rp = out['reproduction']
    md += [f'- d = 100 (policy - rule) / rule per instance, negative = the policy shorter; the '
           f'policy is the per-instance mean over the training seeds '
           f'{", ".join(map(str, out["seeds"]))} (p72.pact)',
           f'- intervals: p72.ci (paired bootstrap, B = {out["boot"]}, fresh default_rng(7) per '
           f'quantity); per run: p72.inf_policy_rows (hierarchical, Holm, combined verdict)',
           f'- reproduction: {rp["summary"]["compared"]} numbers of p72\'s summary for 10x25 '
           f'identical, including "{rp["summary"]["number"]["what"]}" = '
           f'{rp["summary"]["number"]["text"]} in {rp["summary"]["markdown"]}'
           + (f'; {rp["inference"]["compared"]} per-run numbers and verdicts of '
              f'{rp["inference"]["record"]} identical' if 'inference' in rp else ''), '']
    for size, x in out['sizes'].items():
        md += [f'## {size} ({x["n"]} instances per combination, rules from {x["rule_folder"]})', '',
               '| combination | policy, mean | best of seven | best of four | policy vs best of '
               'seven, % | policy vs best of four, % | per run vs seven | per run vs four |',
               '|---|---|---|---|---|---|---|---|']
        for g in list(P72.REGIMES) + [POOL]:
            m = x['means'].get(g)
            pr = x.get('per_run')

            def per(q):
                if pr is None:
                    return '-'
                c = pr[q][g]
                return cell(c) + (f' ({c["verdict"]})' if 'verdict' in c else '')
            md.append(f'| {"pooled" if g == POOL else g} | '
                      + (f'{m["policy"]:.1f} | {m["best7"]:.1f} | {m["best4"]:.1f}' if m else '- | - | -')
                      + f' | {cell(x["rows"]["seven"][g])} | {cell(x["rows"]["four"][g])} | '
                      f'{per("seven")} | {per("four")} |')
        rules = list(P72.CLASSICAL) + list(P72.NEW_RULES)
        md += ['', 'Percent of instances on which a rule is the shortest of the seven (ties '
               'counted for each rule):', '',
               '| combination | ' + ' | '.join(rules) + ' | ties |', '|---|' + '---|' * (len(rules) + 1)]
        for g in list(P72.REGIMES) + [POOL]:
            md.append(f'| {"pooled" if g == POOL else g} | '
                      + ' | '.join(f'{x["shortest"][g][r]:.1f}' for r in rules)
                      + f' | {x["ties_pct"][g]:.1f} |')
        md.append('')
    return '\n'.join(md) + '\n'


def outputs():
    import arm_registry as REG
    return {k: REG.out(v) for k, v in OUT_FILES.items()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def run(B=None, dry=False, inference_check=True):
    """Inputs, reproduction, analysis and outputs; a refusal removes the outputs."""
    import arm_registry as REG
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    P72 = load_p72()
    B = P72.N_BOOT if B is None else int(B)
    outs = outputs()
    remove_outputs(outs)
    try:
        A, miss = inputs(P72, P72.SIZES_EXACT)
        if miss:
            raise Refused(f'{len(miss)} inputs missing: ' + '; '.join(miss[:6]))
        repro = dict(summary=reproduce(P72, A))
        sizes = analyse(P72, A, B)
        if inference_check:
            if B != P72.N_BOOT:
                raise Refused(f'the per-run rows are compared with p72\'s record at B = '
                              f'{P72.N_BOOT}, not {B}')
            repro['inference'] = check_inference(P72, sizes)
        out = dict(generator=GENERATOR, dry=bool(dry), setting=setting_text(),
                   seeds=list(REG.seeds()), boot=B, policy=P72.POLICY,
                   statistics=('p72.pact, p72.stored, p72.ci (paired bootstrap, fresh '
                               'default_rng(7) per quantity), p72.inf_policy_rows '
                               '(stats_family.hier_boot, seed_t, seed_t_p, boot_p_one_sided, '
                               'holm, p89_public_family.superiority_verdict), '
                               'stats_family.direction; reproduction by p72.ci on p72.rng'),
                   reproduction=repro, sizes=sizes)
        for x in sizes.values():
            x['seeds'] = out['seeds']
        lines = macro_lines(P72, out)
        names = check_names(lines, outs['macros'])
        from macro_io import emit
        emit(outs['macros'], lines)
        os.makedirs(os.path.dirname(outs['json']), exist_ok=True)
        st = P72._inf_stats()
        with open(outs['json'], 'w') as f:
            json.dump(st[3](out), f, indent=1)
        with open(outs['md'], 'w') as f:
            f.write(markdown(P72, out))
    except BaseException:
        remove_outputs(outs)
        raise
    return out, names, outs


def check():
    """--check: every input of p72's readers and the records; 1 while any is missing."""
    import arm_registry as REG
    try:
        P72 = load_p72()
    except Refused as e:
        print(f'p126: missing: {e}')
        return 1
    A, miss = inputs(P72, P72.SIZES_EXACT)
    for p in (REG.out(P72.JSON_PATH), REG.out(P72.MD_PATH), os.path.join(ROOT, INFERENCE_RECORD)):
        if not os.path.exists(p):
            miss.append(f'{shown(p)} does not exist')
    for x in miss:
        print(f'p126: missing: {x}')
    for size, S in A.items():
        if S:
            print(f'p126: {size}: {len(S)} combinations, {S[P72.REGIMES[0]]["n"]} instances, policy '
                  f'files {len(S[P72.REGIMES[0]]["files"])} per combination')
    print(f'p126: {"incomplete" if miss else "every input present"}')
    return 1 if miss else 0


# ----------------------------------------------------------------- dry mode
DRY_N = {'10x25': 20, '20x25': 12, '30x25': 12, '50x25n30': 10, '80x25n30': 10}
# the synthetic shortest rule: SPT on even instances, LLF on odd ones, MWKR tied with SPT on
# every fourth; the policy 2 h below the best of seven
DRY_GAP_H = 2.0


def dry_rules(i):
    base = 200.0 + 3 * i
    v = {r: base + 10 + k for k, r in enumerate(('FIFO', 'MOR', 'SPT', 'MWKR', 'ECT', 'SSPT', 'LLF'))}
    if i % 2 == 0:
        v['SPT'] = base
        if i % 4 == 0:
            v['MWKR'] = base
    else:
        v['LLF'] = base
    return v


@contextlib.contextmanager
def dry_setting(tree):
    keep = {k: os.environ.get(k) for k in ('PACT_ENV', 'PACT_SEEDS', 'PACT_RESULTS', 'PACT_OUT')}
    os.environ.update(PACT_ENV='exact', PACT_SEEDS='301-305',
                      PACT_RESULTS=os.path.join(tree, 'res'), PACT_OUT=os.path.join(tree, 'out'))
    try:
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build_world(tree):
    """Rule and policy arrays of every size and p72's summary record of 10x25 written by the
    replay (the Markdown row too)."""
    import arm_registry as REG
    P72 = load_p72()
    for size in P72.SIZES_EXACT:
        n = DRY_N[size]
        d = REG.rule_dir(size)
        os.makedirs(d, exist_ok=True)
        for g in P72.REGIMES:
            V = [dry_rules(i) for i in range(n)]
            for r in P72.NEW_RULES + P72.CLASSICAL:
                np.save(f'{d}/Result_{r}_{g}.npy',
                        np.stack([[v[r] for v in V], np.zeros(n)], axis=1))
            best = np.array([min(v.values()) for v in V])
            for k, s in enumerate(REG.seeds()):
                p = REG.result_path('released', None, s, size, g)
                np.save(p, np.stack([best - DRY_GAP_H + (k - 2), np.zeros(n)], axis=1))
    A, miss = inputs(P72, ['10x25'])
    assert not miss, miss
    mine = replay_summary(P72, A, '10x25')
    rec = dict(seeds=list(REG.seeds()), rules=list(P72.NEW_RULES), sizes={'10x25': dict(
        regimes={g: dict(mine['regimes'][g], pact_files=A['10x25'][g]['files'])
                 for g in P72.REGIMES}, pooled=mine['pooled'])})
    jp, mp = REG.out(P72.JSON_PATH), REG.out(P72.MD_PATH)
    os.makedirs(os.path.dirname(jp), exist_ok=True)
    json.dump(rec, open(jp, 'w'))
    rules = list(P72.NEW_RULES)
    with open(mp, 'w') as f:
        f.write('## 10x25\n\n| regime | ' + ' | '.join(f'{r} vs SPT' for r in rules) + ' | '
                + ' | '.join(f'{r} vs PACT' for r in rules) + ' |\n'
                + '| pooled | ' + ' | '.join(P72._g(mine['pooled'][r]['gap_SPT']) for r in rules)
                + ' | ' + ' | '.join(P72._g(mine['pooled'][r]['gap_PACT']) for r in rules) + ' |\n')
    return P72


def dry_run(tree, boot=DRY_BOOT, before=None):
    """The whole path on a synthetic tree; returns (out, names, outs). before: a function of
    the tree called after it is built."""
    tree = os.path.abspath(tree)
    if os.path.commonpath([tree, ROOT]) == ROOT:
        raise Refused(f'the dry-run folder {tree} lies inside the repository')
    with dry_setting(tree):
        build_world(tree)
        if before is not None:
            before(tree)
        return run(B=boot, dry=True, inference_check=False)


def dry_verify(out):
    """Means and shortest-rule shares against the values built into the tree."""
    bad = []
    for size, x in out['sizes'].items():
        n = DRY_N[size]
        best = np.array([min(dry_rules(i).values()) for i in range(n)])
        want = float(np.mean(100 * (-DRY_GAP_H) / best))
        if abs(x['rows']['seven'][POOL]['mean'] - want) > 1e-9:
            bad.append(f'{size}: policy vs seven {x["rows"]["seven"][POOL]["mean"]}, built {want}')
        b4 = np.array([min(v for r, v in dry_rules(i).items() if r in ('FIFO', 'MOR', 'SPT', 'MWKR'))
                       for i in range(n)])
        want = float(np.mean(100 * (best - DRY_GAP_H - b4) / b4))
        if abs(x['rows']['four'][POOL]['mean'] - want) > 1e-9:
            bad.append(f'{size}: policy vs four {x["rows"]["four"][POOL]["mean"]}, built {want}')
        sh = x['shortest'][POOL]
        want = {'SPT': 100 * len(range(0, n, 2)) / n, 'LLF': 100 * len(range(1, n, 2)) / n,
                'MWKR': 100 * len(range(0, n, 4)) / n}
        for r, v in want.items():
            if abs(sh[r] - v) > 1e-9:
                bad.append(f'{size}: {r} shortest {sh[r]}, built {v}')
        if any(sh[r] for r in ('FIFO', 'MOR', 'ECT', 'SSPT')):
            bad.append(f'{size}: a rule that is never the shortest has a share')
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description='the policy against the best of seven rules')
    ap.add_argument('--check', action='store_true', help='list every missing input')
    ap.add_argument('--dry', nargs='?', const='', default=None, metavar='DIR',
                    help='the whole path on synthetic arrays in DIR (default: a temporary folder)')
    ap.add_argument('--boot', type=int, default=None,
                    help=f'replicates of the dry run (default {DRY_BOOT}); a real run takes p72\'s B')
    a = ap.parse_args(_ARGV[1:] if argv is None else argv)
    os.chdir(ROOT)
    if a.dry is not None:
        if a.check:
            ap.error('--check and --dry exclude each other')
        keep = bool(a.dry)
        tree = os.path.abspath(a.dry) if keep else tempfile.mkdtemp(prefix='p126_dry_')
        try:
            out, names, _ = dry_run(tree, DRY_BOOT if a.boot is None else a.boot)
            bad = dry_verify(out)
            if bad:
                raise Refused('; '.join(bad[:6]))
        except Refused as e:
            sys.exit(f'p126 self-test on synthetic arrays FAILED: {e}')
        finally:
            if not keep:
                shutil.rmtree(tree, ignore_errors=True)
        x = out['sizes']['10x25']['rows']
        print(f'p126 self-test on synthetic arrays passed: B = {out["boot"]}, {len(names)} macros; '
              f'10x25 pooled: policy vs seven {cell(x["seven"][POOL])}, vs four '
              f'{cell(x["four"][POOL])}; every value equal to the built one'
              + (f'; outputs in {tree}' if keep else ''))
        return 0
    if a.boot is not None:
        ap.error('--boot is for the dry run; a real run uses p72\'s B')
    os.environ.setdefault('PACT_ENV', 'exact')
    os.environ.setdefault('PACT_SEEDS', '301-305')
    if a.check:
        return check()
    import arm_registry as REG
    REG.banner()
    try:
        out, names, outs = run()
    except Refused as e:
        print(f'*** REFUSING TO CERTIFY *** {e}', file=sys.stderr)
        return 1
    x = out['sizes']['10x25']
    print(f'p126: reproduced {out["reproduction"]["summary"]["number"]["what"]} = '
          f'{out["reproduction"]["summary"]["number"]["text"]}; 10x25 pooled: policy vs seven '
          f'{cell(x["rows"]["seven"][POOL])}, vs four {cell(x["rows"]["four"][POOL])}; '
          f'{len(names)} macros; wrote ' + ', '.join(shown(p) for p in outs.values()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
