r"""The E5 table with its two one-second rivals added (GA 1 s, CP-SAT 1 s).

Table III (p13_e5_table_fair.py) compares the policy with the rules inside one
budget class and with the GA and CP-SAT inside another. Its GA column is a 30 s
run and its shortest solver column is 5 s, so a reader cannot see in the table
who is best when the answer is needed within one second. This script rebuilds
the same table with the GA at 1 s and the fair-warm CP-SAT 1 s rung added.

Every value that also appears in Table III is computed with p13's own
conventions (policy pooled over three seeds; best single rule by per-regime
mean; per-instance best-of-four portfolio; GA 30 s; CP-SAT 5/30/300 s credited
min(hint, search) against the repaired best-of-four hint), and the script
asserts cell by cell that those columns print exactly what
paper/tables/e5_table.tex prints. The 1 s rung is credited as p30_anytime.py
credits it: the tier was not re-solved from the repaired hint, so its value is
min(repaired hint, min(search, original hint)), and a dagger marks a regime
where the search beat the repaired hint on zero instances.

BOLD RULE. In each row, bold the single lowest printed value across ALL
columns except the 300 s column (a 30-instance prefix, not comparable with the
100-instance columns) and the opt count, mechanically, whoever it lands on.
UNDERLINE the lowest printed value among the columns that answer within one
second (PACT, SPT rule, rule pool, GA 1 s, CP-SAT 1 s). Ties are marked on
every tied cell. Comparison is on the value as printed (one decimal), as in p13.

Writes paper/tables/e5_table_aei.tex, paper/macros_e5aei.tex,
results/e5_table_aei.md.
Usage: python3 scripts/p62_e5_table_aei.py
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from macro_io import emit, refuse

RES = 'test_results/FAMILY/10x25'
OR = 'or_solution/FAMILY'
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
TRAIN = ['N', 'L', 'S', 'W', 'LS', 'LW']
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
POLICY = 'greedy+10x25+family+jointblind-v1'
SEEDS = ('', '-s302', '-s303')
# The published learned rival (Song et al. 2023, HGNN) ported to this benchmark
# (backbones/hgnn), one policy over the six training regimes trained with the
# certified reward, read on all eight regimes; per-instance mean over every
# training seed on disk (scripts/p78_hgnn_rival.py reports the arm in full).
HGNN = 'hgnn-certified-joint'
MACROS = 'paper/macros_e5aei.tex'
TABLE = 'paper/tables/e5_table_aei.tex'
REF_TABLE = 'paper/tables/e5_table.tex'
MD = 'results/e5_table_aei.md'
N_BOOT = 20000

_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def npy(tag, r):
    p = f'{RES}/Result_{tag}_{r}.npy'
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def policy(r):
    """p13.policy: per-instance mean over the three released seeds."""
    arrs = [npy(f'{POLICY}{s}', r) for s in SEEDS]
    if any(a is None for a in arrs):
        return None
    return np.mean(arrs, axis=0)


def hgnn(r):
    """Per-instance mean over the HGNN seed files found for regime r."""
    files = sorted(glob.glob(f'test_results/FAMILY/10x25/Result_{HGNN}-s30[0-9]_{r}.npy'))
    if not files:
        return None, 0
    arrs = [np.load(f) for f in files]
    arrs = [(a[:, 0] if a.ndim > 1 else a).astype(float) for a in arrs]
    return np.mean(arrs, axis=0), len(files)


def cp(r, b):
    """Credited fair-warm CP-SAT, per instance, in instance order.

    5 s and 30 s: p13.cp (re-solved from the repaired hint, min(warm, search)).
    1 s and 300 s: not re-solved; min(repaired hint, min(search, original hint)),
    which is p13.cp for 300 s and p30_anytime.cells for 1 s (identical rule).
    Returns (per-instance credited array, n_opt, n, n_search_improved).
    """
    rerun = b in (5, 30)
    src = (f'{OR}/10x25_fairwarmR{b}s/{r}/*.json' if rerun
           else f'{OR}/10x25_fairwarm{b}s/{r}/*.json')
    vals, nopt, improved = [], 0, 0
    files = sorted(glob.glob(src))
    for f in files:
        d = json.load(open(f))
        d = d[0] if isinstance(d, list) else d
        obj = d.get('objective')
        if obj is None:
            raise SystemExit(f'{f}: objective is null; p13 would drop this cell '
                             f'and p30 would crash on it. Refusing to guess.')
        if d.get('warm_makespan') is None:
            raise SystemExit(f'{f}: no warm_makespan, not a fair-warm run')
        if rerun:
            hint = d['warm_makespan']
            vals.append(min(obj, hint))
        else:
            hint = _REPAIRED[r][d['instance']]['repaired_ms']
            vals.append(min(hint, min(obj, d['warm_makespan'])))
        if obj < hint - 1e-9:
            improved += 1
        if d.get('status') == 'OPTIMAL':
            nopt += 1
    return np.array(vals, dtype=float), nopt, len(files), improved


# ------------------------------------------------------------------ data
rows, missing = [], []
for r in REGIMES:
    pol = policy(r)
    per_rule = {m: npy(m, r) for m in RULES}
    if pol is None or any(v is None for v in per_rule.values()):
        missing.append(f'policy/rules on {r}')
        continue
    hg, n_hg = hgnn(r)
    if hg is None:
        missing.append(f'HGNN on {r}')
        continue
    best = min(per_rule, key=lambda m: per_rule[m].mean())
    port = np.min(np.stack(list(per_rule.values())), axis=0)
    ga30, ga1 = npy('GA', r), npy('GA1s', r)
    if ga30 is None or ga1 is None:
        missing.append(f'GA on {r}')
        continue
    c = {b: cp(r, b) for b in (1, 5, 30, 300)}
    for b in (1, 5, 30):
        if c[b][2] != 100:
            missing.append(f'CP {b}s on {r}: {c[b][2]} of 100')
    rows.append(dict(r=r, pol=pol, hgnn=hg, n_hgnn=n_hg, rule=per_rule[best],
                     rule_name=best, port=port, ga1=ga1, ga30=ga30, cp=c))

if missing:
    refuse(MACROS, 'missing inputs: ' + '; '.join(missing))

n300 = {row['cp'][300][2] for row in rows}
if len(n300) != 1:
    refuse(MACROS, f'ragged 300 s tier: {n300}')
n300 = n300.pop()

rule_names = {row['rule_name'] for row in rows}
if rule_names != {'SPT'}:
    refuse(MACROS, f'best single rule is not SPT in every regime ({rule_names}); '
                   f'the column header "SPT rule" would be false')


# ------------------------------------------------------------------ cells
# Column keys in table order. 'fast' = answers within one second.
COLS = [('pol', True), ('hgnn', True), ('rule', True), ('port', True), ('ga1', True),
        ('cp1', True), ('ga30', False), ('cp5', False), ('cp30', False)]


def printed(row):
    """{key: (printed value rounded to 0.1, dagger?)} for the comparable columns."""
    out = {}
    for k in ('pol', 'hgnn', 'rule', 'port', 'ga1', 'ga30'):
        out[k] = (round(float(row[k].mean()), 1), False)
    for b in (1, 5, 30):
        v, _, _, imp = row['cp'][b]
        out[f'cp{b}'] = (round(float(v.mean()), 1), imp == 0)
    return out


def best_keys(vals, keys):
    lo = min(vals[k][0] for k in keys)
    return {k for k in keys if vals[k][0] == lo}


tex, md, bold_at, under_at = [], [], {}, {}
for row in rows:
    r = row['r']
    v = printed(row)
    bold = best_keys(v, [k for k, _ in COLS])
    under = best_keys(v, [k for k, fast in COLS if fast])
    bold_at[r], under_at[r] = bold, under
    cells = []
    for k, _ in COLS:
        s = f'{v[k][0]:.1f}'
        # No second, class-restricted mark (author rule 2026-08-24): the
        # within-one-second comparison is stated in prose via
        # \EfiveOneSecBestN, never as an underline in the table.
        if k in bold:
            s = rf'\textbf{{{s}}}'
        if v[k][1]:
            s += r'$^\dagger$'
        cells.append(s)
    m300, nopt, _, imp300 = row['cp'][300]
    c300 = f'{m300.mean():.1f}' + (r'$^\dagger$' if imp300 == 0 else '')
    if r == 'SW':
        tex.append(r'\midrule')
    tex.append(f'{r} & ' + ' & '.join(cells) +
               rf' & \textit{{{c300}}} & \textit{{{nopt}}} \\')

    def mdc(k):
        s = f'{v[k][0]:.1f}' + ('+' if v[k][1] else '')
        if k in under:
            s = f'_{s}_'
        if k in bold:
            s = f'**{s}**'
        return s
    md.append(f'| {r} | ' + ' | '.join(mdc(k) for k, _ in COLS) +
              f' | *{m300.mean():.1f}* | *{nopt}* |')


# ------------------------------------------------------------------ cross-check
# The columns shared with Table III must print exactly what Table III prints.
ref = {}
for ln in open(REF_TABLE):
    ln = ln.strip()
    if not ln.endswith(r'\\') or '&' not in ln or ln.startswith('regime'):
        continue
    parts = [re.sub(r'\\textbf\{|\\textit\{|\}|\$\^\\dagger\$', '', p).strip()
             for p in ln[:-2].split('&')]
    ref[parts[0]] = parts[1:]
if set(ref) != set(REGIMES):
    refuse(MACROS, f'could not parse {REF_TABLE}: rows {sorted(ref)}')
mismatch = []
for row in rows:
    v = printed(row)
    m300, nopt, _, _ = row['cp'][300]
    mine = [f'{v[k][0]:.1f}' for k in ('pol', 'rule', 'port', 'ga30', 'cp5', 'cp30')]
    mine += [f'{m300.mean():.1f}', str(nopt)]
    if mine != ref[row['r']]:
        mismatch.append(f'{row["r"]}: mine {mine} vs table {ref[row["r"]]}')
if mismatch:
    refuse(MACROS, 'shared columns do not match tables/e5_table.tex:\n  '
           + '\n  '.join(mismatch))
dag_ref = {}
for ln in open(REF_TABLE):
    if '&' in ln and ln.strip().endswith(r'\\') and not ln.startswith('regime'):
        parts = ln.split('&')
        dag_ref[parts[0].strip()] = [r'\dagger' in p for p in parts[5:8]]
for row in rows:
    mine = [row['cp'][b][3] == 0 for b in (5, 30, 300)]
    if mine != dag_ref[row['r']]:
        refuse(MACROS, f'dagger mismatch on {row["r"]}: {mine} vs {dag_ref[row["r"]]}')
print(f'cross-check: all 8 shared columns and daggers match {REF_TABLE} '
      f'on {len(rows)} regimes')


# ------------------------------------------------------------------ macros
rng = np.random.default_rng(7)


def pooled(regs, key):
    """Mean per-instance relative gap of `key` over the policy, pooled over
    `regs`, 95% paired bootstrap. Positive = the rival is slower."""
    d = np.concatenate([100.0 * (get(row, key) - row['pol']) / row['pol']
                        for row in rows if row['r'] in regs])
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def get(row, key):
    return row['cp'][1][0] if key == 'cp1' else row[key]


lines = ['% auto-generated by scripts/p62_e5_table_aei.py -- do not hand-edit',
         '% E5 with the one-second rivals added (GA 1 s, fair-warm CP-SAT 1 s).',
         '% Makespans in hours (per-regime means over 100 instances); gaps are the',
         '% mean per-instance relative gap to the policy, positive = rival slower.']
for row in rows:
    r = row['r']
    lines.append(
        f'\\newcommand{{\\EfiveAeiGaOne{r}}}{{{row["ga1"].mean():.1f}}}'
        f'\\newcommand{{\\EfiveAeiCpOne{r}}}{{{row["cp"][1][0].mean():.1f}}}'
        f'  % {r}: GA 1 s and credited CP-SAT 1 s mean makespan (h)'
        f'{"; CP 1 s search improved on 0 instances" if row["cp"][1][3] == 0 else ""}')

one_best = [row['r'] for row in rows if under_at[row['r']] == {'pol'}]
all_best = [row['r'] for row in rows if bold_at[row['r']] == {'pol'}]
cp1_dag = [row['r'] for row in rows if row['cp'][1][3] == 0]
lines += [
    f'\\newcommand{{\\EfiveOneSecBestN}}{{{len(one_best)}}}'
    f'  % regimes (of \\EfiveAllN) where PACT alone is best among the <=1 s columns (HGNN included):'
    f' {",".join(one_best)}',
    f'\\newcommand{{\\EfiveAeiBoldPolicyN}}{{{len(all_best)}}}'
    f'  % regimes where PACT alone is best across all 100-instance columns:'
    f' {",".join(all_best) or "none"}',
    f'\\newcommand{{\\EfiveAeiCpOneDaggerN}}{{{len(cp1_dag)}}}'
    f'  % regimes where the 1 s search improved on the repaired hint on 0 instances:'
    f' {",".join(cp1_dag) or "none"}',
]
# Pooled over the SIX TRAINING regimes. The all-eight pools already exist as
# \GaOneGapAll and \CpOneGapAll (p52_ga_ladder.py, different bootstrap); they are
# not re-emitted here.
n_hgnn_seeds = {row['n_hgnn'] for row in rows}
if len(n_hgnn_seeds) != 1:
    refuse(MACROS, f'ragged HGNN seed count across regimes: {n_hgnn_seeds}')
n_hgnn_seeds = n_hgnn_seeds.pop()
lines.append(f'\\newcommand{{\\EfiveAeiHgnnSeeds}}{{{n_hgnn_seeds}}}'
             '  % HGNN training seeds pooled into the HGNN column')
for row in rows:
    lines.append(f'\\newcommand{{\\EfiveAeiHgnn{row["r"]}}}{{{row["hgnn"].mean():.1f}}}'
                 f'  % {row["r"]}: HGNN (certified reward, one policy) mean makespan (h)')
for key, name in (('ga1', 'GaOne'), ('cp1', 'CpOne'), ('hgnn', 'Hgnn')):
    m, lo, hi = pooled(TRAIN, key)
    lines.append(
        f'\\newcommand{{\\EfiveAei{name}Train}}{{{m:+.2f}}}'
        f'\\newcommand{{\\EfiveAei{name}TrainAbs}}{{{abs(m):.2f}}}'
        f'\\newcommand{{\\EfiveAei{name}TrainLo}}{{{lo:+.2f}}}'
        f'\\newcommand{{\\EfiveAei{name}TrainHi}}{{{hi:+.2f}}}'
        f'  % {name} vs PACT pooled over the six training regimes, 95% CI')
    print(f'{name} vs policy, training regimes pooled: {m:+.2f}% [{lo:+.2f},{hi:+.2f}]')

# Collision guard: no name emitted here may be defined in another macro file.
mine = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', '\n'.join(lines)))
for f in glob.glob('paper/macros_*.tex') + ['paper/macros.tex']:
    if os.path.abspath(f) == os.path.abspath(MACROS):
        continue
    clash = mine & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(f).read()))
    if clash:
        refuse(MACROS, f'macro name collision with {f}: {sorted(clash)}')

# The 300 s column is a fixed prefix of the test set, so its comparison with the
# policy is made on the SAME instances (the full-column means are not comparable).
_pr = np.random.default_rng(7)
cap_ahead, cap_tie = [], []
for row in rows:
    v = row['cp'][300][0]
    n = len(v)
    pol = row['pol'][:n]
    d = 100.0 * (v - pol) / pol
    bs = d[_pr.integers(0, n, (N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    if hi < 0:
        cap_ahead.append(row['r'])
    elif lo <= 0:
        cap_tie.append(row['r'])
lines.append(f'\\newcommand{{\\EfiveCapPrefixAheadN}}{{{len(cap_ahead)}}}'
             f'  % regimes where CP-SAT at 300 s is ahead of PACT on the same prefix: '
             f'{",".join(cap_ahead)}; not separable: {",".join(cap_tie) or "none"}')
emit(MACROS, lines)

head = [r'\begin{tabular}{lccccccccccc}', r'\toprule',
        r'regime & PACT & HGNN & SPT rule & rule pool & GA\,1\,s & CP\,1\,s & GA\,30\,s & '
        r'CP\,\CpFiveSec{}\,s & CP\,30\,s & CP\,\CpsatCap{}\,s & opt \\',
        r'\midrule']
foot = [r'\bottomrule', r'\end{tabular}']
os.makedirs('paper/tables', exist_ok=True)
with open(TABLE, 'w') as f:
    f.write('% auto-generated by scripts/p62_e5_table_aei.py -- do not hand-edit\n')
    f.write('% bold: lowest mean in the row over all 100-instance columns;\n')
    f.write(f'% italic: CP 300 s on a {n300}-instance prefix (excluded from both marks);\n')
    f.write('% dagger: the CP-SAT search improved on its warm start on no instance.\n')
    f.write('\n'.join(head + tex + foot) + '\n')
print(f'wrote {TABLE}')

os.makedirs('results', exist_ok=True)
with open(MD, 'w') as f:
    f.write('# E5 with one-second columns (review copy of '
            'paper/tables/e5_table_aei.tex)\n\n'
            'Mean makespan (h) per regime, 100 instances; CP 300 s on a '
            f'{n300}-instance prefix (italic). **bold** = row minimum over all '
            '100-instance columns; _underline_ = minimum over the columns that '
            'answer within 1 s (PACT, HGNN, SPT, pool, GA 1 s, CP 1 s); + = the CP-SAT '
            'search improved on its warm start on no instance.\n\n'
            '| regime | PACT | HGNN | SPT rule | rule pool | GA 1 s | CP 1 s | GA 30 s '
            '| CP 5 s | CP 30 s | CP 300 s | opt |\n' + '|---' * 12 + '|\n')
    f.write('\n'.join(md) + '\n\n')
    f.write(f'PACT alone best within 1 s: {len(one_best)} of {len(rows)} '
            f'({", ".join(one_best)}).\n'
            f'PACT alone best over all 100-instance columns: {len(all_best)} of '
            f'{len(rows)} ({", ".join(all_best) or "none"}).\n')
print(f'wrote {MD}')
print('\n'.join(md))

# ------------------------------------------------------------------ per-regime gaps
# Review aid only (no macros): is each one-second lead in the table separable
# from zero? Same estimator; a separate rng so the pooled macros above are fixed.
_rr = np.random.default_rng(7)


def regime_gap(row, key):
    d = 100.0 * (get(row, key) - row['pol']) / row['pol']
    bs = d[_rr.integers(0, len(d), (N_BOOT, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


# The CP 1 s means must agree with p30_anytime.py's \AnyOne<r>Lead, which uses
# the same credit rule and the same estimator.
anyt = dict(re.findall(r'\\AnyOne([A-Z]+)Lead\}\{([-+0-9.]+)\}',
                       open('paper/macros_anytime.tex').read()))
tab = ['| regime | GA 1 s - PACT (%) [95% CI] | CP 1 s - PACT (%) [95% CI] |',
       '|---|---|---|']
for row in rows:
    g = regime_gap(row, 'ga1')
    c = regime_gap(row, 'cp1')
    if row['r'] not in anyt or f'{c[0]:+.2f}' != anyt[row['r']]:
        refuse(MACROS, f'CP 1 s gap on {row["r"]} is {c[0]:+.2f}, p30 prints '
                       f'{anyt.get(row["r"])}')
    tab.append(f'| {row["r"]} | {g[0]:+.2f} [{g[1]:+.2f}, {g[2]:+.2f}] | '
               f'{c[0]:+.2f} [{c[1]:+.2f}, {c[2]:+.2f}] |')
with open(MD, 'a') as f:
    f.write('\n## Per-regime gap of each one-second rival to PACT\n\n'
            'Mean per-instance relative gap, positive = the rival is slower; '
            'paired bootstrap, 20k resamples, seed 7. CP 1 s column cross-checked '
            'against \\AnyOne<regime>Lead in paper/macros_anytime.tex.\n\n'
            + '\n'.join(tab) + '\n')
print('\n'.join(tab))
