r"""P116: whether the specialists, the policy and the single-rule control had stopped improving
when their training ended.

WHAT IS READ
  The training log of every run of four arms for the active seeds (PACT_ENV=exact, PACT_SEEDS):
  the three principal arms through arm_registry.train_log, that is the policy (arm released, six
  combinations, 2000 updates), the single-rule control (arm control, N, L, S and W, 2000 updates)
  and the specialist of each of the eight combinations (1000 updates); and the specialist of each
  combination trained for the policy's 2000 updates (arm long; run names and training logs
  through scripts/longspec_runs.py, as scripts/p117_long_specialists.py reads them).
  train_family.py validates every 25 updates and writes one line per validation,
      [vali @950] rel=0.9244 (best 0.9209)  L=209.8
  where rel is the validation score: for each training combination of the run, the mean greedy
  makespan over the 20 validation instances divided by the mean makespan of the dispatching rule
  MWKR over the same instances (MWKR under the environment's default mask), averaged over the
  combinations, so lower is better. The run keeps the first checkpoint with the lowest score, and
  a finished run ends with the line
      [train_family] done in 1.84 h; best rel=0.9209
  The validation curve of a run is the list of (update, score) of these lines. A last validation
  line that appears twice in identical form is read once.

REFUSALS (each names the run)
  incomplete  the log is absent, holds no completion line, or holds fewer validation lines than
              updates / 25. Without --partial the whole set is refused; with --partial the run is
              left out and named.
  defective   the header line of the log names another run, other training combinations or
              another number of updates than the arm trains; the validation updates are not
              25, 50, ..., updates; the last update appears twice with different lines; or the
              best score of the completion line differs from the minimum of the curve by more
              than 5e-5 (the log prints four decimals). Refused in both modes.

PER RUN (U updates, score s_u at update u, s* = min_u s_u, the score of the kept checkpoint)
  late_change       100 (mean{s_u : 0.8 U < u <= U} / mean{s_u : 0.6 U < u <= 0.8 U} - 1): the
                    mean score over the last fifth of training against the mean over the fifth
                    before it, in percent of the latter; negative means still improving.
  best_at           min{u : s_u = s*} / U: the share of training after which the kept
                    checkpoint's score was first reached.
  gain_second_half  100 (s* / min{s_u : u <= 0.5 U} - 1): the best score of the whole run against
                    the best score of its first half, in percent of the latter; zero or negative.
  curve             100 (s_u / s* - 1) against 100 u / U: the score in percent above the run's own
                    best score, against the share of training in percent.
  raw curve         the list of (update, score) itself (key curve of each run).
  change_after_1000 (arm long only) 100 (mean{s_u : 1600 < u <= 2000}
                    / mean{s_u : 800 < u <= 1000} - 1): the mean score over the last fifth of the
                    2000 updates against the mean score over the fifth before update 1000 (the
                    last fifth of a specialist's 1000 updates), in percent of the latter;
                    negative means that the validation schedules shortened after update 1000.
PER ARM, AND PER COMBINATION FOR THE SPECIALISTS
  the number of runs; median, minimum and maximum over the seeds of late_change and
  gain_second_half; the median of best_at; the same over all specialist runs pooled (arms
  released, control and specialist). The curves in percent above the run's best (key curves):
  at each validation, the median, minimum and maximum over the seeds of the curve.
  The raw curves (key raw_curves), for every arm: at each validation update, the median, minimum
  and maximum over the seeds of the raw score, per arm for the policy and the control and per
  combination for the specialists of both lengths; for the specialists of both lengths also the
  mean over the combinations of these per-combination medians (key mean, with the combinations
  it covers).
  Arm long (key arms.long): change_after_1000 per combination as the median over the seeds; over
  all runs the median, the minimum and the maximum, with the run and combination of the minimum
  and of the maximum (ties: the earlier combination in N, L, S, W, LS, LW, SW, LSW, then the
  lower seed); the combinations whose median is negative.

OUTPUTS (arm_registry.out under PACT_OUT)
  manuscript/macros_convergence.tex  macro_io.emit; names Cvg<quantity>, letters only, checked
                                    against every macro defined in a .tex file under manuscript/
                                    (the output itself left out); signed values with two decimals
                                    and their sign (p88 f2), shares of training in percent; for
                                    arm long the names CvgLong<quantity>: Runs and Seeds (runs and
                                    training seeds), Med, Min and Max (change_after_1000 over all
                                    runs), MinCombo and MaxCombo (their combinations), Med<R> (the
                                    median over the seeds of combination R), NegN, NegCombos and
                                    NegList (the combinations whose median is negative: number,
                                    names, running text)
  results/convergence.json          every per-run curve and value, the summaries, the curves in
                                    percent above each run's best and the raw curves, so that
                                    every macro and every plotted line can be recomputed from it
  --partial writes both files to the folder given by --out (required), never to the paths above,
  and the first line of the macro file says that it is partial.

CHECK (--check, read only)
  exit 0 when every log of the active seeds is complete and not defective; prints the number
  of complete logs, in all and per arm.

Usage: PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=out/exact python scripts/p116_convergence.py
       ... python scripts/p116_convergence.py --check
       ... python scripts/p116_convergence.py --partial --out DIR
       --root DIR resolves the training logs against DIR (default: the repository).
"""
import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np  # noqa: E402

import arm_registry as REG  # noqa: E402
import longspec_runs as LR  # noqa: E402
import p88_family_inference as F88  # noqa: E402
from macro_io import emit  # noqa: E402

GENERATOR = 'scripts/p116_convergence.py'
PREFIX = 'Cvg'
OUT_FILES = {'macros': 'manuscript/macros_convergence.tex', 'json': 'results/convergence.json'}
ENV = 'exact'
VALI_EVERY = 25
UPDATES = {'released': 2000, 'control': 2000, 'specialist': 1000, 'long': LR.UPDATES}
BEST_TOL = 5e-5
ARM_TEXT = {'released': 'policy', 'control': 'single-rule control', 'specialist': 'specialist',
            'long': 'specialist of 2000 updates'}
ARM_WORD = {'released': 'Pol', 'control': 'Ctl', 'specialist': 'Spec', 'long': 'Long'}
SPEC_ARMS = ('specialist', 'long')                # one run per combination and seed
R8 = tuple(REG.COMBOS)
# change_after_1000 of arm long: the windows (a, b] of updates whose mean scores it compares
AFTER_REF = (4 * LR.STANDARD_UPDATES // 5, LR.STANDARD_UPDATES)      # 800 < u <= 1000
AFTER_LATE = (4 * LR.UPDATES // 5, LR.UPDATES)                        # 1600 < u <= 2000

_VALI = re.compile(r'^\[vali @(\d+)\] rel=([0-9.]+) \(best ([0-9.]+)\)[^\n]*$', re.M)
_DONE = re.compile(r'^\[train_family\] done in [^\n;]*; best rel=([0-9.]+)\s*$', re.M)
_HEAD = re.compile(r'^\[train_family\] model=(\S+) regimes=\[([^\]]*)\] updates=(\d+)\s*$', re.M)


class Refused(Exception):
    """A setting, a log or a set of runs that this script does not accept."""


class Incomplete(Refused):
    """A log that is absent or whose run has not finished."""


# ----------------------------------------------------------------- runs and logs
def arm_combos():
    """(arm, combination) of every curve group: the policy, the control, the eight specialists
    and the eight specialists trained for 2000 updates."""
    return ([('released', None), ('control', None)] + [('specialist', r) for r in R8]
            + [('long', r) for r in R8])


def runs(seeds):
    return [(a, c, s) for a, c in arm_combos() for s in seeds]


def trained_combos(arm, combo):
    if arm in SPEC_ARMS:
        return (combo,)
    return tuple(REG.TRAIN_COMBOS) if arm == 'released' else tuple(REG.CONTROL_COMBOS)


def run_name(arm, combo, seed):
    """Name of the run: arm_registry.model, or longspec_runs.name for arm long."""
    if arm != 'long':
        return REG.model(arm, combo, seed)
    try:
        return LR.name(combo, seed)
    except LR.Refused as e:
        raise Refused(str(e))


def run_log(arm, combo, seed):
    """Training log of the run: arm_registry.train_log, or longspec_runs.train_log for arm long
    (the log of the 1000-update specialist with the run name exchanged)."""
    if arm != 'long':
        return REG.train_log(arm, combo, seed)
    try:
        return LR.train_log(combo, seed)
    except LR.Refused as e:
        raise Refused(str(e))


def read_curve(path, arm, combo, name):
    """(updates, [(update, score)], reported best) of one finished run; raises Incomplete or
    Refused, naming the run."""
    if not os.path.exists(path):
        raise Incomplete(f'{name}: no training log {path}')
    text = open(path, errors='replace').read()
    want_u = UPDATES[arm]
    head = _HEAD.findall(text)
    if len(head) != 1:
        raise Refused(f'{name}: {path} holds {len(head)} header lines of train_family.py, want one')
    model, regimes, upd = head[0]
    regimes = tuple(x.strip().strip("'\"") for x in regimes.split(',') if x.strip())
    if model != REG.PREFIX + name:
        raise Refused(f'{name}: {path} is the log of {model}')
    if regimes != trained_combos(arm, combo):
        raise Refused(f'{name}: {path} trains {regimes}, the arm trains {trained_combos(arm, combo)}')
    if int(upd) != want_u:
        raise Refused(f'{name}: {path} declares {upd} updates, the arm trains {want_u}')
    done = _DONE.findall(text)
    if not done:
        raise Incomplete(f'{name}: {path} holds no completion line')
    lines = [(m.group(0), int(m.group(1)), float(m.group(2))) for m in _VALI.finditer(text)]
    if len(lines) >= 2 and lines[-1][1] == lines[-2][1]:
        if lines[-1][0] != lines[-2][0]:
            raise Refused(f'{name}: {path} holds two different lines for update {lines[-1][1]}')
        lines = lines[:-1]
    want_n = want_u // VALI_EVERY
    if len(lines) < want_n:
        raise Incomplete(f'{name}: {path} holds {len(lines)} validation lines, want {want_n}')
    ups = [u for _, u, _ in lines]
    if ups != [VALI_EVERY * (k + 1) for k in range(want_n)]:
        raise Refused(f'{name}: {path} validates at updates {ups[:3]} ... {ups[-3:]}, want '
                      f'{VALI_EVERY}, {2 * VALI_EVERY}, ..., {want_u}')
    curve = [(u, s) for _, u, s in lines]
    best = float(done[-1])
    low = min(s for _, s in curve)
    if abs(best - low) > BEST_TOL:
        raise Refused(f'{name}: {path} reports best rel={best:.4f}, the minimum of its curve is '
                      f'{low:.4f}')
    return want_u, curve, best


# ----------------------------------------------------------------- statistics
def run_statistics(curve, updates):
    """late_change, best_at and gain_second_half of one curve [(update, score)] (module
    docstring), and the curve in percent above its own best."""
    u = np.array([x for x, _ in curve], dtype=int)
    s = np.array([y for _, y in curve], dtype=float)
    U = int(updates)
    last = s[(5 * u > 4 * U) & (u <= U)]
    prev = s[(5 * u > 3 * U) & (5 * u <= 4 * U)]
    first_half = s[2 * u <= U]
    if not len(last) or not len(prev) or not len(first_half):
        raise Refused(f'a curve of {len(s)} validations over {U} updates has an empty fifth or half')
    best = float(s.min())
    return dict(late_change=float(100.0 * (last.mean() / prev.mean() - 1.0)),
                best_at=float(u[int(np.argmin(s))] / U),
                best_update=int(u[int(np.argmin(s))]),
                gain_second_half=float(100.0 * (best / first_half.min() - 1.0)),
                share=(100.0 * u / U).tolist(),
                above_best=(100.0 * (s / best - 1.0)).tolist())


def summary(rows):
    """Number of runs, median, minimum and maximum of late_change and gain_second_half, median
    of best_at, over the given per-run records."""
    out = dict(n=len(rows), seeds=[r['seed'] for r in rows])
    for k in ('late_change', 'gain_second_half'):
        v = np.array([r[k] for r in rows], dtype=float)
        out[k] = dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    out['best_at'] = dict(median=float(np.median([r['best_at'] for r in rows])))
    return out


def band(rows):
    """The curve group of the figure: share of training, and the median, minimum and maximum
    over the seeds of the score above the run's best, at each validation."""
    shares = {tuple(r['share']) for r in rows}
    if len(shares) != 1:
        raise Refused('the runs of one group do not validate at the same shares of training')
    a = np.array([r['above_best'] for r in rows], dtype=float)
    return dict(share=list(shares.pop()), median=np.median(a, axis=0).tolist(),
                min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist(), n=len(rows))


def change_after_1000(curve):
    """100 (mean score over AFTER_LATE / mean score over AFTER_REF - 1) of one curve
    [(update, score)] of 2000 updates (module docstring)."""
    u = np.array([x for x, _ in curve], dtype=int)
    s = np.array([y for _, y in curve], dtype=float)
    late = s[(u > AFTER_LATE[0]) & (u <= AFTER_LATE[1])]
    ref = s[(u > AFTER_REF[0]) & (u <= AFTER_REF[1])]
    if not len(late) or not len(ref):
        raise Refused(f'a curve of {len(s)} validations up to update {u.max() if len(u) else 0} '
                      f'has no validation in {AFTER_REF} or in {AFTER_LATE}')
    return float(100.0 * (late.mean() / ref.mean() - 1.0))


def raw_band(rows):
    """A raw curve group: the validation updates, and the median, minimum and maximum over the
    seeds of the score at each of them."""
    grids = {tuple(u for u, _ in r['curve']) for r in rows}
    if len(grids) != 1:
        raise Refused('the runs of one group do not validate at the same updates')
    a = np.array([[s for _, s in r['curve']] for r in rows], dtype=float)
    return dict(update=list(grids.pop()), median=np.median(a, axis=0).tolist(),
                min=a.min(axis=0).tolist(), max=a.max(axis=0).tolist(), n=len(rows),
                seeds=[r['seed'] for r in rows])


def mean_of_medians(groups):
    """{update, mean, combos, n}: at each validation update the mean over the combinations of
    groups {R: raw_band} of their medians."""
    combos = [c for c in R8 if c in groups]
    grids = {tuple(groups[c]['update']) for c in combos}
    if len(grids) != 1:
        raise Refused('the combinations of one arm do not validate at the same updates')
    a = np.array([groups[c]['median'] for c in combos], dtype=float)
    return dict(update=list(grids.pop()), mean=a.mean(axis=0).tolist(), combos=combos,
                n=len(combos))


def tex_list(combos):
    """Combinations for running text: '$N$', '$N$ and $W$', '$N$, $W$ and $SW$'; 'none' when
    empty."""
    t = [f'${c}$' for c in combos]
    if not t:
        return 'none'
    return t[0] if len(t) == 1 else ', '.join(t[:-1]) + ' and ' + t[-1]


def long_summary(rows):
    """Arm long: change_after_1000 per combination (median over the seeds) and over all runs."""
    per = {}
    for c in R8:
        v = [r['change_after_1000'] for r in rows if r['combo'] == c]
        if v:
            per[c] = dict(n=len(v), seeds=[r['seed'] for r in rows if r['combo'] == c],
                          median=float(np.median(v)))
    v = np.array([r['change_after_1000'] for r in rows], dtype=float)
    order = sorted(rows, key=lambda r: (R8.index(r['combo']), r['seed']))
    lo = min(order, key=lambda r: r['change_after_1000'])
    hi = max(order, key=lambda r: r['change_after_1000'])
    neg = [c for c in R8 if c in per and per[c]['median'] < 0]
    return dict(n=len(rows), seeds=sorted({r['seed'] for r in rows}), combos=per,
                windows=dict(ref=list(AFTER_REF), late=list(AFTER_LATE)),
                pooled=dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max())),
                min_run=dict(combo=lo['combo'], seed=lo['seed'], name=lo['name'],
                             value=lo['change_after_1000']),
                max_run=dict(combo=hi['combo'], seed=hi['seed'], name=hi['name'],
                             value=hi['change_after_1000']),
                negative=dict(n=len(neg), combos=neg))


def check_setting():
    if REG.env() != ENV:
        raise Refused(f'the analysis reads the {ENV} environment; the active environment is '
                      f'{REG.env()} (set PACT_ENV={ENV})')
    return REG.seeds()


def collect(root='.'):
    """(records of the readable runs, [(name, log, reason)] of the incomplete ones); a defective
    log raises Refused."""
    seeds = check_setting()
    got, missing = [], []
    for arm, combo, seed in runs(seeds):
        name = run_name(arm, combo, seed)
        log = run_log(arm, combo, seed)
        path = log if root in ('.', '') else os.path.join(root, log)
        try:
            U, curve, best = read_curve(path, arm, combo, name)
        except Incomplete as e:
            missing.append(dict(arm=arm, combo=combo, seed=int(seed), name=name, log=log,
                                reason=str(e)))
            continue
        rec = dict(arm=arm, combo=combo, seed=int(seed), name=name, log=log, updates=U,
                   reported_best=best, curve=[[u, s] for u, s in curve])
        rec.update(run_statistics(curve, U))
        if arm == 'long':
            rec['change_after_1000'] = change_after_1000(curve)
        got.append(rec)
    return got, missing


def analyse(root='.', partial=False):
    """The result dict; refuses an incomplete set unless partial."""
    seeds = check_setting()
    got, missing = collect(root)
    if missing and not partial:
        raise Refused(f'{len(missing)} of {len(got) + len(missing)} runs are incomplete (use '
                      '--partial to report the others):\n  '
                      + '\n  '.join(m['reason'] for m in missing))
    if not got:
        raise Refused('no run of the active seeds is complete')
    arms, curves = {}, {}
    for arm in ('released', 'control'):
        rows = [r for r in got if r['arm'] == arm]
        if rows:
            arms[arm] = summary(rows)
            curves[arm] = band(rows)
    spec = [r for r in got if r['arm'] == 'specialist']
    if spec:
        per = {}
        curves['specialist'] = {}
        for c in R8:
            rows = [r for r in spec if r['combo'] == c]
            if rows:
                per[c] = summary(rows)
                curves['specialist'][c] = band(rows)
        lowest = min(spec, key=lambda r: (r['late_change'], R8.index(r['combo']), r['seed']))
        low_c = min(per, key=lambda c: (per[c]['late_change']['median'], R8.index(c)))
        arms['specialist'] = dict(pooled=summary(spec), combos=per,
                                  largest_late_improvement=dict(
                                      combo=lowest['combo'], seed=lowest['seed'],
                                      name=lowest['name'], late_change=lowest['late_change']),
                                  lowest_combo_median=dict(
                                      combo=low_c, late_change=per[low_c]['late_change']['median']))
    long_rows = [r for r in got if r['arm'] == 'long']
    if long_rows:
        arms['long'] = long_summary(long_rows)
    raw = {}
    for arm in ('released', 'control'):
        rows = [r for r in got if r['arm'] == arm]
        if rows:
            raw[arm] = raw_band(rows)
    for arm in SPEC_ARMS:
        groups = {c: raw_band([r for r in got if r['arm'] == arm and r['combo'] == c])
                  for c in R8 if any(r['arm'] == arm and r['combo'] == c for r in got)}
        if groups:
            raw[arm] = dict(combos=groups, mean=mean_of_medians(groups))
    return dict(generator=GENERATOR, setting=REG.describe(), seeds=list(seeds),
                partial=bool(missing), expected=len(got) + len(missing), present=len(got),
                missing=missing, vali_every=VALI_EVERY, updates=dict(UPDATES), runs=got, arms=arms,
                curves=curves, raw_curves=raw,
                definitions=dict(
                    score='rel of train_family.py: per training combination, the mean greedy '
                          'makespan over the validation instances divided by the mean MWKR '
                          'makespan over the same instances, averaged over the combinations; '
                          'lower is better',
                    late_change='100 (mean score over 0.8U < u <= U / mean score over '
                                '0.6U < u <= 0.8U - 1); negative = still improving',
                    best_at='first update at which the minimum score was reached, divided by U',
                    gain_second_half='100 (minimum score over the run / minimum score over '
                                     'u <= 0.5U - 1); zero or negative',
                    curve='100 (score / minimum score - 1) against 100 u / U',
                    raw_curve='the score against the update (key curve of each run); per group '
                              'the median, minimum and maximum over the seeds at each update '
                              '(raw_curves); for the specialists of both lengths also the mean '
                              'over the combinations of the per-combination medians',
                    change_after_1000='arm long: 100 (mean score over 1600 < u <= 2000 / mean '
                                      'score over 800 < u <= 1000 - 1); negative = the '
                                      'validation schedules shortened after update 1000'))


# ----------------------------------------------------------------- macros
def pct(x):
    """A share of training in percent, with the decimals it needs (72.5, 71.25, 70)."""
    s = f'{100.0 * float(x):.2f}'.rstrip('0').rstrip('.')
    return s if s not in ('', '-0') else '0'


def macro_lines(res, partial_line=None):
    f2 = F88.f2
    L = []
    if partial_line:
        L.append(f'% {partial_line}')
    L.append(f'% Generated by {GENERATOR}, training seeds '
             + ', '.join(str(s) for s in res['seeds']))
    A = res['arms']
    left = [ARM_TEXT[a] for a in ('specialist', 'released', 'control', 'long') if a not in A]
    if left:
        L.append(f'% Left out, no complete run: {", ".join(left)}')
    if 'specialist' in A:
        P = A['specialist']['pooled']
        L.append(f'\\newcommand{{\\{PREFIX}SpecRuns}}{{{P["n"]}}}'
                 f'\\newcommand{{\\{PREFIX}Seeds}}{{{len(res["seeds"])}}}'
                 '  % specialist runs, training seeds')
    else:
        L.append(f'\\newcommand{{\\{PREFIX}Seeds}}{{{len(res["seeds"])}}}  % training seeds')
    for arm in ('specialist', 'released', 'control'):
        if arm not in A:
            continue
        S = A[arm]['pooled'] if arm == 'specialist' else A[arm]
        w = PREFIX + ARM_WORD[arm]
        lc, g = S['late_change'], S['gain_second_half']
        who = 'all specialist runs' if arm == 'specialist' else f'the {ARM_TEXT[arm]} runs'
        L.append(f'\\newcommand{{\\{w}LateMed}}{{{f2(lc["median"])}}}'
                 f'\\newcommand{{\\{w}LateMin}}{{{f2(lc["min"])}}}'
                 f'\\newcommand{{\\{w}LateMax}}{{{f2(lc["max"])}}}'
                 f'  % {who}: mean score of the last fifth against the fifth before, %, '
                 'median, minimum, maximum')
        L.append(f'\\newcommand{{\\{w}GainMin}}{{{f2(g["min"])}}}'
                 f'\\newcommand{{\\{w}GainMax}}{{{f2(g["max"])}}}'
                 f'\\newcommand{{\\{w}BestAtMed}}{{{pct(S["best_at"]["median"])}}}'
                 f'  % {who}: best score against the best of the first half, %, minimum and '
                 'maximum; median share of training at which the best score was first reached, %')
    if 'specialist' in A:
        lo, lm = A['specialist']['largest_late_improvement'], A['specialist']['lowest_combo_median']
        L.append(f'\\newcommand{{\\{PREFIX}SpecLateMinCombo}}{{{lo["combo"]}}}'
                 f'\\newcommand{{\\{PREFIX}SpecLateMedLow}}{{{f2(lm["late_change"])}}}'
                 f'\\newcommand{{\\{PREFIX}SpecLateMedLowCombo}}{{{lm["combo"]}}}'
                 f'  % combination of the specialist run at {PREFIX}SpecLateMin (seed '
                 f'{lo["seed"]}); lowest median over the seeds among the eight combinations, '
                 'and its combination')
    if 'long' in A:
        L += long_macro_lines(A['long'])
    return L


def long_macro_lines(G):
    """The macros CvgLong of arm long (module docstring), each line with its comment."""
    f2, w = F88.f2, PREFIX + ARM_WORD['long']
    P = G['pooled']
    what = ('mean validation score over updates 1600 to 2000 against updates 800 to 1000, %; '
            'negative = shorter validation schedules after update 1000')
    L = [f'\\newcommand{{\\{w}Runs}}{{{G["n"]}}}\\newcommand{{\\{w}Seeds}}{{{len(G["seeds"])}}}'
         '  % specialists trained for 2000 updates: runs, training seeds',
         f'\\newcommand{{\\{w}Med}}{{{f2(P["median"])}}}\\newcommand{{\\{w}Min}}{{{f2(P["min"])}}}'
         f'\\newcommand{{\\{w}Max}}{{{f2(P["max"])}}}'
         f'  % all runs of 2000 updates: {what}; median, minimum, maximum',
         f'\\newcommand{{\\{w}MinCombo}}{{{G["min_run"]["combo"]}}}'
         f'\\newcommand{{\\{w}MaxCombo}}{{{G["max_run"]["combo"]}}}'
         f'  % combinations of the runs at {w}Min (seed {G["min_run"]["seed"]}) and {w}Max '
         f'(seed {G["max_run"]["seed"]})']
    have = [c for c in R8 if c in G['combos']]
    for part in (have[:4], have[4:]):
        if part:
            L.append(''.join(f'\\newcommand{{\\{w}Med{c}}}{{{f2(G["combos"][c]["median"])}}}'
                             for c in part)
                     + f'  % {", ".join(part)}: median over the seeds of the same change, %')
    neg = G['negative']['combos']
    L.append(f'\\newcommand{{\\{w}NegN}}{{{len(neg)}}}'
             f'\\newcommand{{\\{w}NegCombos}}{{{", ".join(neg) if neg else "none"}}}'
             f'\\newcommand{{\\{w}NegList}}{{{tex_list(neg)}}}'
             '  % combinations whose median over the seeds is negative (shorter validation '
             'schedules after update 1000): number, names, running text')
    return L


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


def check_names(lines):
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    bad = [x for x in names if not re.fullmatch(r'[A-Za-z]+', x) or not x.startswith(PREFIX)]
    if bad:
        raise Refused(f'macro names that are not letters only with the prefix {PREFIX}: {bad[:5]}')
    if len(names) != len(set(names)):
        raise Refused('duplicate macro names in the output')
    have = defined_macros()
    taken = sorted(x for x in have if x.startswith(PREFIX))
    if taken:
        raise Refused(f'the prefix {PREFIX} is used by \\{taken[0]} ({have[taken[0]]})')
    return names


# ----------------------------------------------------------------- outputs
def outputs(out_dir=None):
    """Paths of the two outputs: arm_registry.out of OUT_FILES, or the folder out_dir (partial)."""
    if out_dir is None:
        return {k: REG.out(v) for k, v in OUT_FILES.items()}
    return {k: os.path.join(out_dir, os.path.basename(v)) for k, v in OUT_FILES.items()}


def real_paths():
    """Absolute paths of the full run's outputs, in place and under PACT_OUT, computed without
    arm_registry.out (which creates the folders)."""
    roots = ['']
    if os.environ.get('PACT_OUT') not in (None, '', '.'):
        roots.append(os.environ['PACT_OUT'])
    return {os.path.abspath(os.path.join(ROOT, r, v)) for r in roots for v in OUT_FILES.values()}


def remove_outputs(outs):
    for p in outs.values():
        if os.path.exists(p):
            os.remove(p)


def table(res):
    """Plain-text table of the summaries, one row per arm and per combination."""
    A = res['arms']
    rows = []
    for key, S in ([('policy', A.get('released')), ('single-rule control', A.get('control'))]
                   + ([('specialists, pooled', A['specialist']['pooled'])]
                      + [(f'specialist {c}', A['specialist']['combos'][c])
                         for c in R8 if c in A['specialist']['combos']]
                      if 'specialist' in A else [])):
        if S is None:
            continue
        lc, g = S['late_change'], S['gain_second_half']
        rows.append(f'  {key:22s} {S["n"]:3d}  {lc["median"]:+7.2f} {lc["min"]:+7.2f} '
                    f'{lc["max"]:+7.2f}   {g["median"]:+7.2f} {g["min"]:+7.2f} {g["max"]:+7.2f}'
                    f'   {100 * S["best_at"]["median"]:6.2f}')
    head = ('  group                    n  late_change %: median    min     max   '
            'gain_second_half %: median min max   best_at median %')
    out = [head] + rows
    if 'long' in A:
        G = A['long']
        out.append('  specialists of 2000 updates, change_after_1000 % (updates 1600 to 2000 '
                   'against 800 to 1000):')
        out.append(f'  {"all runs":22s} {G["n"]:3d}  median {G["pooled"]["median"]:+7.2f}  '
                   f'min {G["min_run"]["value"]:+7.2f} ({G["min_run"]["name"]})  '
                   f'max {G["max_run"]["value"]:+7.2f} ({G["max_run"]["name"]})')
        for c in R8:
            if c in G['combos']:
                g = G['combos'][c]
                out.append(f'  {"long " + c:22s} {g["n"]:3d}  median {g["median"]:+7.2f}')
        out.append(f'  negative medians: {G["negative"]["n"]} '
                   f'({", ".join(G["negative"]["combos"]) or "none"})')
    return out


def run(root='.', partial=False, out_dir=None):
    """analyse() and the two outputs; a refusal removes them. Returns (res, names, outs)."""
    check_setting()
    if partial:
        if not out_dir:
            raise Refused('--partial writes to the folder given by --out; give --out DIR')
        out_dir = os.path.abspath(out_dir)
        if {os.path.abspath(p) for p in outputs(out_dir).values()} & real_paths():
            raise Refused(f'--out {out_dir} is a folder of the real outputs; a partial run '
                          'writes elsewhere')
        os.makedirs(out_dir, exist_ok=True)
        outs = outputs(out_dir)
    else:
        if out_dir:
            raise Refused('--out is for --partial only; the full run writes through '
                          'arm_registry.out under PACT_OUT')
        if os.environ.get('PACT_OUT') is None:
            raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
        outs = outputs()
    remove_outputs(outs)
    try:
        res = analyse(root, partial)
        note = None
        if partial:
            note = (f'PARTIAL: {res["present"]} of {res["expected"]} runs; not for the manuscript'
                    + ('' if res['missing'] else ' (no run is missing)'))
        lines = macro_lines(res, note)
        names = check_names(lines)
        emit(outs['macros'], lines)
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(res), f, indent=1)
    except BaseException:
        remove_outputs(outs)
        raise
    return res, names, outs


def check(root='.'):
    """Read-only precondition of the chain step; returns the exit code."""
    got, missing = collect(root)
    n = len(got) + len(missing)
    print(f'p116: training logs of seeds {",".join(map(str, REG.seeds()))}: {len(got)} of {n} '
          'complete (completion line, one validation line per 25 updates, best score equal to '
          'the minimum of the curve)')
    for arm in ('released', 'control', 'specialist', 'long'):
        k = sum(r['arm'] == arm for r in got)
        print(f'p116:   {ARM_TEXT[arm]}: {k} of {k + sum(m["arm"] == arm for m in missing)}')
    for m in missing[:6]:
        print(f'p116:   {m["reason"]}')
    if len(missing) > 6:
        print(f'p116:   ... and {len(missing) - 6} more')
    return 0 if not missing else 1


def main():
    ap = argparse.ArgumentParser(description='whether training had stopped improving')
    ap.add_argument('--check', action='store_true', help='read-only precondition of the chain')
    ap.add_argument('--partial', action='store_true',
                    help='report the complete runs and name the others; writes to --out only')
    ap.add_argument('--out', default=None, help='output folder of --partial')
    ap.add_argument('--root', default='.', help='folder the training logs are resolved against')
    a = ap.parse_args()
    out_dir = os.path.abspath(a.out) if a.out else None
    root = os.path.abspath(a.root) if a.root not in ('.', '') else '.'
    os.chdir(ROOT)
    try:
        if a.check:
            sys.exit(check(root))
        REG.banner()
        res, names, outs = run(root, a.partial, out_dir)
    except Refused as e:
        sys.exit(f'*** REFUSING TO CERTIFY ***\n{e}')
    print(f'p116: {res["present"]} of {res["expected"]} runs ({res["setting"]})')
    print('\n'.join(table(res)))
    if 'specialist' in res['arms']:
        lo = res['arms']['specialist']['largest_late_improvement']
        lm = res['arms']['specialist']['lowest_combo_median']
        print(f'  smallest late_change of a specialist run: {lo["late_change"]:+.2f}% '
              f'({lo["name"]}); lowest median over seeds: {lm["late_change"]:+.2f}% ({lm["combo"]})')
    for m in res['missing']:
        print(f'  missing: {m["reason"]}')
    print(f'\n{len(names)} macros; wrote ' + ', '.join(outs.values()))


if __name__ == '__main__':
    main()
