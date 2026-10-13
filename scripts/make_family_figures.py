"""this paper figures (F2 parity bars, F3 zero-shot heatmap, F4 causality).

Each figure is generated AT ITS PLACED PHYSICAL SIZE (IEEE single column =
3.5 in) with fonts >= 7 pt, then rendered to PNG for manual inspection and
PDF for inclusion. Colors: Okabe-Ito CVD-safe palette (Wong, Nature Methods
2011), FIXED entity->hue mapping (never cycled); the heatmap uses a
single-hue sequential ramp with direct cell labels (identity/value never
color-alone).

Usage: python scripts/make_family_figures.py [--only f2|f3|f4]
Missing inputs are skipped with a message (rerun as results land).
"""
import glob
import json
import os
import re
import sys

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator

RES = 'test_results/FAMILY/10x25'
OUT = 'paper/figures'

# fixed entity -> color (Okabe-Ito)
C_JOINT = '#0072B2'      # blue
C_SPEC = '#E69F00'       # orange
C_BLIND = '#999999'      # gray
C_PDR = '#009E73'        # green

# The figures use the SAME Times face as the manuscript body: TeX Gyre Termes,
# the face newtxtext puts on every page. Liberation Serif and Nimbus Roman are
# only metric-compatible; their glyphs (squarer digits, shallower bowls) are
# visibly not the body font when a figure sits in the page. matplotlib cannot
# embed the Termes OTF cleanly (CFF outlines under pdf.fonttype 42 produce a
# "Mismatch between font type and embedded font file" warning in pdffonts), so
# scripts/make_termes_ttf.py converts it to TrueType once; the converted files
# are committed in scripts/fonts/ and registered here.
from matplotlib import font_manager as _fm
_FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
for _f in glob.glob(os.path.join(_FONTDIR, 'TeXGyreTermes-*.ttf')):
    _fm.fontManager.addfont(_f)
_SERIF = 'TeX Gyre Termes'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [_SERIF, 'Liberation Serif', 'Times New Roman'],
    # Math is NOT exempt from the Times rule. matplotlib's default and the
    # 'stix' set both pull glyphs from STIXGeneral, so a single $\pm$ or
    # $\Delta$ silently embeds a second, visibly non-Times family in the
    # figure. Map every mathtext class onto the same serif face; the figures
    # additionally spell +-, the minus sign and Delta as Unicode literals in
    # the regular font, so no mathtext survives to be rendered at all.
    'mathtext.fontset': 'custom',
    'mathtext.rm': _SERIF, 'mathtext.it': f'{_SERIF}:italic',
    'mathtext.bf': f'{_SERIF}:bold', 'mathtext.sf': _SERIF,
    'mathtext.tt': _SERIF, 'mathtext.cal': f'{_SERIF}:italic',
    'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
    'ytick.major.width': 0.6, 'pdf.fonttype': 42,
})

# imshow() writes a RASTER image into the PDF, always. On a heat map this small it
# lands as a 264 ppi indexed bitmap: below IEEE's 300 dpi floor for images, and a
# bitmap in a figure that is otherwise pure vector. The cells are rectangles, so we
# draw rectangles. Same colours, same mapping, no raster object in the output.
def _cells(ax, shade, cmap='Blues', vmin=0.0, vmax=1.15):
    """Draw a heat map as vector rectangles, one per cell, in imshow's coordinates
    (cell (r,c) is centred on (c, r) and spans one unit in each direction)."""
    cm = plt.get_cmap(cmap)
    nr, nc = shade.shape
    for r in range(nr):
        for c in range(nc):
            v = (shade[r, c] - vmin) / (vmax - vmin)
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1,
                                   facecolor=cm(float(np.clip(v, 0, 1))),
                                   edgecolor='none', linewidth=0, zorder=0))
    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)          # imshow's origin is top-left
    ax.set_aspect('auto')




def _stale(name, why):
    """Overwrite a figure whose inputs are missing with a loud placeholder.

    The placeholder keeps the document compiling while making the figure
    impossible to mistake for a result: a stale .pdf left on disk would keep
    rendering silently, and a deleted one would break the build.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.10), dpi=300)
    ax.axis('off')
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                               fill=False, ec='#D55E00', lw=1.5))
    ax.text(0.5, 0.62, 'FIGURE NOT REGENERATED', ha='center', va='center',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            color='#D55E00')
    ax.text(0.5, 0.32, why, ha='center', va='center', wrap=True,
            transform=ax.transAxes, fontsize=7, color='#D55E00')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/{name}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print(f'  PLACEHOLDER written for {name}: {why}')

def _load(tag, regime):
    """Per-instance makespans for one method on one regime.

    Result files come in two shapes: [n, 2] (makespan, seconds) from the policy
    evaluations, and [n] from methods that record no timing, so the load guards
    on ndim before taking column 0.
    """
    p = os.path.join(RES, f'Result_{tag}_{regime}.npy')
    if not os.path.exists(p):
        return None
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def _pooled(base, regime):
    """Per-instance makespan of a trained model, meaned across its three seeds.

    EVERY learned policy in a figure goes through this. Nothing may reach a panel
    through the bare _load(), because _load() returns seed 301 alone, and a panel
    that pools three seeds on one side of a comparison and one on the other is
    measuring its estimator rather than its policies. That is not hypothetical:
    panel (b) of f23 did exactly this. The joint row was pooled and the
    specialist and token-blind rows were not, so the baseline was represented by
    a single draw -- which on BOTH held-out compositions happened to be that
    baseline's worst seed. The figure showed the joint beating the SW specialist
    while the paper's own E2 macro (\\EqSwDelta) reported it 0.2% behind.

    Falls back to whatever seeds exist, and says so, rather than silently
    comparing unequal pools.
    """
    seeds = [base, f'{base}-s302', f'{base}-s303']
    arrs = [_load(f'greedy+{m}', regime) for m in seeds]
    have = [s for s, a in zip(seeds, arrs) if a is not None]
    arrs = [a for a in arrs if a is not None]
    if not arrs:
        return None
    if len(arrs) < 3:
        print(f'  seed note: {base} on {regime} has {len(arrs)}/3 seeds '
              f'({", ".join(h.split("+")[-1] for h in have)})')
    return np.mean(arrs, axis=0)


def _load_joint(model, regime):
    return _pooled(model, regime)


def _load_spec(regime):
    return _pooled(f'10x25+family+spec-{regime}', regime)


def _paired_boot(d, n=20000, seed=7):
    """95% CI of the mean paired gap. Same estimator as p11_equivalence.py."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def f2_parity(joint='10x25+family+jointblind-v1', margin=1.0):
    """Forest plot of the PAIRED relative gap, joint minus specialist (E1).

    Shows exactly the statistic the parity claim rests on: E1 is a PAIRED
    equivalence test against a +-1% margin, so the figure plots one point per
    regime, its paired-bootstrap 95% CI, and the equivalence band. It reads
    the same per-seed arrays as p11_equivalence.py, so figure and macros
    cannot drift.
    """
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW']
    los, his, mids = [], [], []
    for r in regimes:
        j = _load_joint(joint, r)
        s = _load_spec(r)
        if j is None or s is None:
            print(f'F2: missing inputs for {r}')
            _stale('f2_parity', f'joint or specialist results missing for regime {r}')
            return
        d = 100.0 * (j - s) / s          # per-instance relative gap, %
        lo, hi = _paired_boot(d)
        los.append(lo); his.append(hi); mids.append(float(d.mean()))

    # Three verdicts, three colours. An interval wholly inside the band is
    # EQUIVALENT; one wholly beyond it is INFERIOR, a decided loss; one that
    # straddles the band edge is INCONCLUSIVE. Drawing the last two alike would
    # let a reader take a decided loss for an undecided one.
    def _verdict(lo, hi):
        if hi < 0:
            return 'superior'          # whole CI below zero; at these widths
                                       # it also sits inside the margin band
        if -margin < lo and hi < margin:
            return 'equivalent'
        if lo > margin:
            return 'inferior'
        return 'inconclusive'

    STYLE = {'superior':     ('#009E73', '^'),
             'equivalent':   (C_JOINT,   'o'),
             'inconclusive': ('#999999', 's'),
             'inferior':     ('#D55E00', 'D')}

    y = np.arange(len(regimes))[::-1]
    fig, ax = plt.subplots(figsize=(3.5, 2.10), dpi=300)
    ax.axvspan(-margin, margin, color=C_SPEC, alpha=0.16, lw=0,
               label=f'±{margin:g}% margin')
    ax.axvline(0, color='0.35', lw=0.6, zorder=1)
    seen = set()
    for yi, lo, hi, m in zip(y, los, his, mids):
        v = _verdict(lo, hi)
        c, mk = STYLE[v]
        ax.plot([lo, hi], [yi, yi], lw=1.4, color=c, solid_capstyle='round',
                zorder=3)
        ax.plot([m], [yi], mk, ms=3.4, color=c, zorder=4,
                label=v if v not in seen else None)
        seen.add(v)
    ax.set_yticks(y); ax.set_yticklabels(regimes)
    ax.set_ylim(-0.7, len(regimes) - 0.3)
    ax.set_xlabel('paired makespan gap, PACT minus specialist (%)')
    ax.set_ylabel('constraint regime')
    ax.grid(axis='x', lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.legend(frameon=False, ncol=2, loc='upper left', handlelength=1.2,
              columnspacing=1.0, bbox_to_anchor=(0, 1.30))
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f2_parity.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F2 written')


def f3_zeroshot(joint='10x25+family+jointblind-v1',
                cond='10x25+family+joint-v1'):
    """Heatmap: methods x held-out regimes, mean makespan (E2, money fig).

    PROTAGONIST SWAP RESIDUE, fixed 2026-08-23. After the released policy became
    the token-blind arm, the first two rows were BOTH jointblind-v1 -- one pooled
    over three seeds, one a bare seed-301 read -- while the caption still called
    row one the conditioned joint. Two rows of one model under two seed poolings
    is exactly the defect _pooled()'s docstring warns about, wearing a caption
    that misattributed the numbers. Every learned row now goes through _pooled(),
    and the second row is the conditioned variant it always claimed to be."""
    rows = [('PACT (zero-shot)', joint, None),
            ('conditioned variant', cond, None),
            ('composition specialist', '10x25+family+spec-{R}', None),
            ('best dispatching rule', None, ('FIFO', 'MOR', 'SPT', 'MWKR'))]
    regimes = ['SW', 'LSW']
    M = np.full((len(rows), len(regimes)), np.nan)
    for c, reg in enumerate(regimes):
        for r, (label, tag, pdrs) in enumerate(rows):
            if pdrs:
                vals = [_load(p, reg) for p in pdrs]
                vals = [v.mean() for v in vals if v is not None]
                M[r, c] = min(vals) if vals else np.nan
            else:
                v = _pooled(tag.replace('{R}', reg), reg)
                M[r, c] = v.mean() if v is not None else np.nan
    if np.isnan(M).any():
        print('F3: missing inputs')
        _stale('f3_zeroshot', 'one or more methods have no results on SW/LSW')
        return
    # per-COLUMN normalized shading: absolute makespans differ across
    # regimes, and the quantity being compared is the within-regime ranking of
    # methods, so each column carries its own scale (lighter = better). Shading
    # is therefore not comparable ACROSS columns; the cell labels are, and the
    # best cell per column is bolded.
    shade = np.zeros_like(M)
    for c in range(M.shape[1]):
        col = M[:, c]
        lo, hi = np.nanmin(col), np.nanmax(col)
        shade[:, c] = (col - lo) / (hi - lo + 1e-9)
    fig, ax = plt.subplots(figsize=(2.6, 1.9), dpi=300)
    _cells(ax, shade, 'Blues', 0, 1.15)
    for c in range(M.shape[1]):
        best = np.nanargmin(M[:, c])
        for r in range(M.shape[0]):
            if not np.isnan(M[r, c]):
                dark = shade[r, c] > 0.6
                ax.text(c, r, f'{M[r, c]:.1f}', ha='center', va='center',
                        fontsize=7.5,
                        fontweight='bold' if r == best else 'normal',
                        color='white' if dark else '#1a1a1a')
    ax.set_xticks(range(len(regimes)))
    ax.set_xticklabels([f'{r}\n(held out)' for r in regimes])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f3_zeroshot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F3 written')


def f4_causality(model='10x25+family+joint-v1'):
    """Token interventions: mean degradation vs true token per regime (E3)."""
    regimes = ['L', 'S', 'W', 'LS', 'LW']
    CF = {'L': 'S', 'S': 'L', 'W': 'LS', 'LS': 'W', 'LW': 'S'}
    CH = {'L': 'lag', 'S': 'setup', 'W': 'window', 'LS': 'lag-setup',
          'LW': 'lag-window'}
    # The third label wraps onto two lines so the frameless upper-left legend
    # ends left of the W group's error whisker instead of printing across it.
    modes = [('withhold', C_BLIND, 'zero token'),
             ('counterfactual-{CF}', C_SPEC, 'conflicting token'),
             ('withhold+chblind-{CH}', C_JOINT, 'token + channels\nremoved')]
    fig, ax = plt.subplots(figsize=(3.5, 1.52), dpi=300)
    x = np.arange(len(regimes))
    w = 0.26
    plotted = False
    for k, (mode, color, label) in enumerate(modes):
        deltas, errs = [], []
        for r in regimes:
            base = _load(f'greedy+{model}', r)
            tag = mode.replace('{CF}', CF[r]).replace('{CH}', CH[r])
            pert = _load(f'greedy+{model}+{tag}', r)
            if base is None or pert is None:
                deltas.append(np.nan); errs.append(0)
                continue
            d = pert - base
            deltas.append(d.mean())
            errs.append(1.96 * d.std() / np.sqrt(len(d)))
            plotted = True
        ax.bar(x + (k - 1) * w, deltas, w, yerr=errs, color=color,
               error_kw=dict(lw=0.7), label=label)
    if not plotted:
        print('F4: no inputs yet, skipping')
        plt.close(fig)
        return
    ax.axhline(0, color='0.35', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(regimes)
    ax.set_ylabel('Δ makespan vs\ntrue token (h)', fontsize=8.5)
    ax.set_xlabel('constraint regime', fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis='y', lw=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    # The only tall bar is W (centre); L and LW ends are short, so the upper-left
    # corner is clear. Keep the legend small and pinned there.
    leg = ax.legend(loc='upper left', fontsize=6.3, handlelength=0.9,
                    handletextpad=0.4, labelspacing=0.25, borderpad=0.3,
                    framealpha=0.92, edgecolor='0.75')
    leg.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f4_causality.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F4 written')


# --- E5 anytime frontier -----------------------------------------------------
# These three helpers replicate scripts/p30_anytime.py exactly so the figure and
# macros_anytime.tex cannot drift: same result files, same credited min(warm,
# search) rule, same per-instance relative-gap averaging. Do not "improve" the
# aggregation here; p30 is the generator and this only redraws its numbers.
OR = 'or_solution/FAMILY'
# PROTAGONIST SWAP (2026-08-11): the frontier is the released blind joint's
ANY_JOINT = 'greedy+10x25+family+jointblind-v1'
ANY_SEEDS = ['', '-s302', '-s303']         # p30's three seeds
ANY_REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
ANY_SETUP = ['S', 'LS', 'SW', 'LSW']       # the setup-bearing regimes
ANY_RUNGS = [1, 2, 5, 10, 30]              # 300 s is a partial prefix; excluded
ANY_WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty'}
ANY_RERUN = (5, 30)                        # tiers re-solved from the repaired hint
# --- panel (c): the same frontier at the two larger shops -------------------
# Mirrors scripts/p39_anytime_scale.py, which credits every size against the
# repaired hint exactly as p30 credits 10x25. size -> (result dir, instances
# per regime, macro word, modules, repaired-cache dir).
SCALE = [
    ('10x25', 'test_results/FAMILY/10x25', 100, 'Ten', 10,
     'warmstart_cache_repaired'),
    ('20x25', 'test_results/FAMILY/20x25', 50, 'Twenty', 20,
     'warmstart_cache_repaired_20x25'),
    ('30x25', 'test_results/FAMILY/30x25', 20, 'Thirty', 30,
     'warmstart_cache_repaired_30x25'),
]
SCALE_RUNGS = [1, 2, 5, 10, 30, 60]        # 10x25 stops at 30; the rest run to 60
# The 1800 s cells (50 and 80 modules) carry anytime trajectories, so one solve
# answers every smaller budget: credited(b) = min(warm, best incumbent <= b).
# Their pooled per-rung leads are emitted by p60_scale1800.py, and panel (b)
# asserts its plotted values against those macros.
TRAJ_SIZES = [('50x25', 'test_results/FAMILY/50x25', 50, 'Fifty'),
              ('80x25', 'test_results/FAMILY/80x25', 80, 'Eighty')]
TRAJ_RUNGS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800]
TRAJ_WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty',
             60: 'Sixty', 120: 'HundredTwenty', 300: 'ThreeHundred',
             600: 'SixHundred', 1800: 'EighteenHundred'}
SCALE_RESOLVED = {('10x25', 5), ('10x25', 30)}
# The repaired best-of-four warm-start cache (scripts/p41_repaired_warm_cache.py):
# the 5s and 30s rungs were re-solved from it, the 1s/2s/10s rungs have their
# credited value floored at it. Mirrors p30_anytime exactly.
_ANY_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def _any_rung_dir(budget):
    return (f'{OR}/10x25_fairwarmR{budget}s' if budget in ANY_RERUN
            else f'{OR}/10x25_fairwarm{budget}s')


def _any_credited(budget, regime):
    """Per-instance CREDITED solver makespans against the repaired hint, for one
    (budget, regime). Exactly p30_anytime.cells() without the improved-flag."""
    out = []
    rerun = budget in ANY_RERUN
    for p in sorted(glob.glob(f'{_any_rung_dir(budget)}/{regime}/*.json')):
        d = json.load(open(p))
        x = d[0] if isinstance(d, list) else d
        obj, warm = x['objective'], x.get('warm_makespan')
        if warm is None:
            return None                    # not a fair-warm run; treat as missing
        if rerun:
            out.append(min(obj, warm))     # warm_makespan is the repaired hint
        else:
            hint = _ANY_REPAIRED[regime][x['instance']]['repaired_ms']
            out.append(min(hint, min(obj, warm)))
    return np.array(out, dtype=float) if out else None


def _any_policy(regime):
    """Per-instance greedy-policy makespan, meaned across p30's three seeds."""
    arrs = []
    for s in ANY_SEEDS:
        p = f'{RES}/Result_{ANY_JOINT}{s}_{regime}.npy'
        if not os.path.exists(p):
            return None
        a = np.load(p)
        arrs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return np.mean(arrs, axis=0)


def _scale_credited(size, budget, regime, cache):
    """p39's credited makespan: the better of the search result and the hint,
    the hint being the repaired warm start wherever the tier was not re-solved
    from it. Kept identical to p39 on purpose; if one of them is changed the
    gate below stops the figure rather than letting the two drift."""
    rep = cache.get(regime)
    d = (f'{OR}/{size}_fairwarmR{budget}s'
         if (size, budget) in SCALE_RESOLVED else f'{OR}/{size}_fairwarm{budget}s')
    out = []
    for p in sorted(glob.glob(f'{d}/{regime}/*.json')):
        x = json.load(open(p))
        x = x[0] if isinstance(x, list) else x
        best = min(x['objective'], x['warm_makespan'])
        if (size, budget) not in SCALE_RESOLVED and rep is not None:
            best = min(best, rep[x['instance']]['repaired_ms'])
        out.append(best)
    return np.array(out, dtype=float)


def _scale_frontier():
    """(pooled lead per (size, rung), per-regime crossing budget per size).

    The pooled lead is p30's estimator carried across regimes: the mean of the
    per-instance relative gaps, all 8 regimes concatenated, positive = policy
    ahead. Returns None if any size is short a rung, so the figure is skipped
    rather than drawn from a partial ladder."""
    pooled, crossing = {}, {}
    for size, res, n_inst, word, _jobs, cdir in SCALE:
        cpath = f'{OR}/{cdir}/makespans.json'
        if not os.path.exists(cpath):
            return None, f'{cdir} missing: p41 has not built the repaired hint'
        cache = json.load(open(cpath))
        rungs = [b for b in SCALE_RUNGS
                 if all(len(glob.glob(
                     (f'{OR}/{size}_fairwarmR{b}s' if (size, b) in SCALE_RESOLVED
                      else f'{OR}/{size}_fairwarm{b}s') + f'/{r}/*.json')) == n_inst
                     for r in ANY_REGIMES)]
        if not rungs:
            return None, f'{size}: no complete rung'
        pol = {}
        for r in ANY_REGIMES:
            a = []
            for s in ANY_SEEDS:
                p = f'{res}/Result_{ANY_JOINT}{s}_{r}.npy'
                if not os.path.exists(p):
                    return None, f'{size}/{r}: policy results missing'
                x = np.load(p)
                a.append((x[:, 0] if x.ndim > 1 else x).astype(float))
            pol[r] = np.mean(a, axis=0)
        cross = {}
        for b in rungs:
            gaps = []
            for r in ANY_REGIMES:
                sol = _scale_credited(size, b, r, cache)
                if len(sol) != len(pol[r]):
                    return None, f'{size}/{r}@{b}s: {len(sol)} solves vs {len(pol[r])}'
                gaps.append(100.0 * (sol - pol[r]) / pol[r])
                if sol.mean() <= pol[r].mean() and r not in cross:
                    cross[r] = b
            pooled[(size, b)] = float(np.concatenate(gaps).mean())
        crossing[size] = (cross, rungs)
    return (pooled, crossing), None


def f5_anytime():
    """Line plot of the policy's lead over the credited fair-warm CP-SAT solver as
    the solver's time budget grows (E5). One line per regime; y is the mean
    per-instance relative gap in percent (positive = policy ahead), the same
    quantity macros_anytime.tex reports; x is the budget at the five measured
    rungs (1, 2, 5, 10, 30 s), log-scaled.

    The setup-bearing regimes (S, LS, SW, LSW) are drawn as saturated solid lines
    and the rest (N, L, W, LW) as muted dashed lines, because the story splits on
    exactly that line: where setups are active the search returns nothing better
    than its warm start until after 5 s, so those lines stay flat and positive,
    then dip toward zero by 30 s.
    """
    # A rung is only admissible if all 8 regimes x 100 instances are present, the
    # same 800-cell completeness gate p30 enforces before it will report a rung.
    for b in ANY_RUNGS:
        if len(glob.glob(f'{_any_rung_dir(b)}/*/*.json')) != 800:
            _stale('f5_anytime', f'fair-warm rung {b}s is incomplete '
                                 f'(need 800 cells: 8 regimes x 100 instances)')
            return
    if _any_policy('N') is None:
        _stale('f5_anytime', 'joint policy results (joint-v1, 3 seeds) missing')
        return

    # Compute the plotted grid: lead[(rung, regime)] = mean per-instance relative
    # gap (%), positive = policy ahead.
    lead = {}
    for r in ANY_REGIMES:
        pol = _any_policy(r)
        for b in ANY_RUNGS:
            sol = _any_credited(b, r)
            lead[(b, r)] = float((100.0 * (sol - pol) / pol).mean())

    # INTEGRITY GATE. The per-regime frontier is no longer a panel of this
    # figure (it lives in the supplement's frontier table), but the check stays:
    # macros_anytime.tex is p30's output and is ground truth, and every value
    # recomputed here must reproduce its \Any<Word><Regime>Lead macro to two
    # decimals. p9 compares timestamps, not values, so without this a drifted
    # result file would change the supplement's table with nothing to catch it.
    txt = open('paper/macros_anytime.tex').read()
    for b in ANY_RUNGS:
        for r in ANY_REGIMES:
            m = re.search(r'\\newcommand\{\\Any' + ANY_WORD[b] + r
                          + r'Lead\}\{([+-][0-9.]+)\}', txt)
            if m is None:
                raise SystemExit(f'f5: macro \\Any{ANY_WORD[b]}{r}Lead not found '
                                 f'in macros_anytime.tex')
            macro = float(m.group(1))
            if abs(round(lead[(b, r)], 2) - macro) > 0.005:
                raise SystemExit(
                    f'f5 REFUSING: {r} at {b}s computes {lead[(b, r)]:+.2f}% but '
                    f'macros_anytime.tex says {macro:+.2f}%. The figure and the '
                    f'paper disagree; fix the input, do not ship the figure.')

    # Fixed entity -> (colour, marker) mapping, Okabe-Ito, never cycled. Setup
    # regimes get saturated hues and solid lines; the rest get muted hues and a
    # dashed stroke, so the two families separate by dash even in greyscale, and
    # every regime is told apart by its marker without relying on colour.
    STYLE = {
        'S':   ('#D55E00', 'o', '-'),      # vermillion
        'LS':  ('#0072B2', 's', '-'),      # blue
        'SW':  ('#009E73', '^', '-'),      # bluish green
        'LSW': ('#CC79A7', 'D', '-'),      # reddish purple
        'N':   ('#000000', 'v', (0, (4, 2))),
        'L':   ('#E69F00', 'X', (0, (4, 2))),   # orange
        'W':   ('#56B4E9', 'P', (0, (4, 2))),   # sky blue
        'LW':  ('#999999', '*', (0, (4, 2))),   # grey
    }

    # ---------------- panel (b) inputs: every alternative's gap TO the policy
    # at matched budgets, pooled by setup-bearing vs not (user directive
    # 2026-08-11: the main exhibits must state WHEN this policy is the best
    # option). Positive = the alternative is worse. Asserted against
    # macros_gal.tex (p52) below, the same discipline as panel (a).
    def _ga(tag, r):
        return _load(f'GA{tag}', r)

    def _gap_pool(vals_by_reg, regs):
        gaps = []
        for r in regs:
            pol = _any_policy(r)
            v = vals_by_reg[r]
            n = min(len(v), len(pol))
            gaps.append(100.0 * (v[:n] - pol[:n]) / pol[:n])
        return float(np.concatenate(gaps).mean())

    METHODS = {                        # label -> (per-regime loader, budgets)
        'GA':     (lambda b, r: _ga({1: '1s', 5: '5s', 30: ''}[b], r),
                   [1, 5, 30]),
        'CP-SAT': (lambda b, r: _any_credited(b, r), ANY_RUNGS),
        'best rule': (lambda b, r: _load('SPT', r), None),   # budget-free
    }
    MAC_NAME = {('GA', 1): 'GaOne', ('GA', 5): 'GaFive', ('GA', 30): 'GaThirty',
                ('CP-SAT', 1): 'CpOne', ('CP-SAT', 2): 'CpTwo',
                ('CP-SAT', 5): 'CpFive', ('CP-SAT', 10): 'CpTen',
                ('CP-SAT', 30): 'CpThirty', ('best rule', None): 'Rule'}
    galp = 'results/ga_ladder_gaps.json'
    if not os.path.exists(galp):
        _stale('f5_anytime', 'ga_ladder_gaps.json missing -- run p52 first')
        return
    galgrid = json.load(open(galp))
    bvals = {}                          # (label, budget, fam) -> pooled gap
    for label, (loader, budgets) in METHODS.items():
        for b in (budgets or [None]):
            per = {r: loader(b, r) for r in ANY_REGIMES}
            if any(v is None for v in per.values()):
                _stale('f5_anytime', f'{label} at {b}s incomplete')
                return
            for fam, regs in (('Setup', ANY_SETUP),
                              ('Other', [r for r in ANY_REGIMES
                                         if r not in ANY_SETUP])):
                g = _gap_pool(per, regs)
                key = MAC_NAME[(label, b)]
                ref = galgrid.get(key, {}).get(f'Gap{fam}')
                if ref is None or abs(g - ref) > 0.005:
                    raise SystemExit(
                        f'f5(a) REFUSING: {key} Gap{fam} computes {g:+.2f} '
                        f'but p52 recorded {ref}. The figure and the paper '
                        f'disagree; fix the input, do not ship the figure.')
                bvals[(label, b, fam)] = g

    # ---------------- panel (b) data, and the gate that keeps it honest.
    (scale, why) = _scale_frontier()
    if scale is None:
        _stale('f5_anytime', f'panel (b): {why}')
        return
    spooled, scross = scale
    # CORRECTNESS GATE, the same discipline panels (a) and (b) carry: every
    # per-regime crossing this figure derives must reproduce the \ScaleCross
    # macro p39 emitted. A changed credited rule on either side dies here.
    stxt = open('paper/macros_anytime_scale.tex').read()
    for size, _res, _n, word, _jobs, _c in SCALE:
        cross, rungs = scross[size]
        for r in ANY_REGIMES:
            m = re.search(r'\\newcommand\{\\ScaleCross' + word + r
                          + r'\}\{([0-9]+)\}', stxt)
            if m is None:
                raise SystemExit(f'f5(b): macro \\ScaleCross{word}{r} not found')
            want = int(m.group(1))          # 0 = never crosses within the ladder
            got = cross.get(r, 0)
            if got != want:
                raise SystemExit(
                    f'f5(b) REFUSING: {size}/{r} crosses at {got or "never"} but '
                    f'macros_anytime_scale.tex says {want or "never"}. The figure '
                    f'and the paper disagree; fix the input, not the figure.')

    # ---------------- the 1800 s cells, from their anytime trajectories.
    traj = {}
    for size, res, jobs, word in TRAJ_SIZES:
        pol = {}
        for r in ANY_REGIMES:
            a = []
            for sd in ANY_SEEDS:
                pth = f'{res}/Result_{ANY_JOINT}{sd}_{r}.npy'
                if not os.path.exists(pth):
                    _stale('f5_anytime', f'{size}/{r}: policy results missing')
                    return
                x = np.load(pth)
                a.append((x[:, 0] if x.ndim > 1 else x).astype(float))
            pol[r] = np.mean(a, axis=0)
        for b in TRAJ_RUNGS:
            gaps = []
            for r in ANY_REGIMES:
                cred = []
                fs = sorted(glob.glob(
                    f'{OR}/{size}_fairwarm1800s/{r}/*.json'))
                if len(fs) != 10:
                    _stale('f5_anytime', f'{size}/{r}: {len(fs)}/10 solves')
                    return
                for fp in fs:
                    d = json.load(open(fp))
                    best = d['warm_makespan']
                    for t, ub, _ in d.get('anytime', []):
                        if t <= b and ub < best:
                            best = ub
                    cred.append(best)
                cred = np.array(cred)
                n = min(len(cred), len(pol[r]))
                gaps.append(100 * (cred[:n] - pol[r][:n]) / pol[r][:n])
            traj[(size, b)] = float(np.concatenate(gaps).mean())
    # GATE: every plotted value must reproduce p60's macro to two decimals.
    ttxt = open('paper/macros_scale1800.tex').read()
    for size, _res, _j, word in TRAJ_SIZES:
        for b in TRAJ_RUNGS:
            m = re.search(r'\\newcommand\{\\Sc' + word + TRAJ_WORD[b]
                          + r'\}\{([+-][0-9.]+)\}', ttxt)
            if m is None:
                raise SystemExit(f'f5(b): macro Sc{word}{TRAJ_WORD[b]} missing')
            if abs(round(traj[(size, b)], 2) - float(m.group(1))) > 0.005:
                raise SystemExit(
                    f'f5(b) REFUSING: {size}@{b}s computes '
                    f'{traj[(size, b)]:+.2f} but p60 emitted {m.group(1)}.')

    # Sized to print at one full \columnwidth (3.5 in): the canvas IS the print
    # size, so every font below lands on paper at its stated point size.
    # The panels do NOT share an axis: panel (a)'s ladder stops at 30 s and
    # panel (b) now runs to 1800 s, so a shared axis would squeeze (a)'s data
    # into the left fifth of its width and leave the rest empty. Each panel
    # labels its own ticks instead.
    fig, (bx, cx) = plt.subplots(
        2, 1, figsize=(3.5, 2.86), dpi=300,
        gridspec_kw={'height_ratios': [0.92, 0.88], 'hspace': 0.72})
    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f4_causality.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F4 written')


# --- E5 anytime frontier -----------------------------------------------------
# These three helpers replicate scripts/p30_anytime.py exactly so the figure and
# macros_anytime.tex cannot drift: same result files, same credited min(warm,
# search) rule, same per-instance relative-gap averaging. Do not "improve" the
# aggregation here; p30 is the generator and this only redraws its numbers.
OR = 'or_solution/FAMILY'
# PROTAGONIST SWAP (2026-08-11): the frontier is the released blind joint's
ANY_JOINT = 'greedy+10x25+family+jointblind-v1'
ANY_SEEDS = ['', '-s302', '-s303']         # p30's three seeds
ANY_REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
ANY_SETUP = ['S', 'LS', 'SW', 'LSW']       # the setup-bearing regimes
ANY_RUNGS = [1, 2, 5, 10, 30]              # 300 s is a partial prefix; excluded
ANY_WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty'}
ANY_RERUN = (5, 30)                        # tiers re-solved from the repaired hint
# --- panel (c): the same frontier at the two larger shops -------------------
# Mirrors scripts/p39_anytime_scale.py, which credits every size against the
# repaired hint exactly as p30 credits 10x25. size -> (result dir, instances
# per regime, macro word, modules, repaired-cache dir).
SCALE = [
    ('10x25', 'test_results/FAMILY/10x25', 100, 'Ten', 10,
     'warmstart_cache_repaired'),
    ('20x25', 'test_results/FAMILY/20x25', 50, 'Twenty', 20,
     'warmstart_cache_repaired_20x25'),
    ('30x25', 'test_results/FAMILY/30x25', 20, 'Thirty', 30,
     'warmstart_cache_repaired_30x25'),
]
SCALE_RUNGS = [1, 2, 5, 10, 30, 60]        # 10x25 stops at 30; the rest run to 60
# The 1800 s cells (50 and 80 modules) carry anytime trajectories, so one solve
# answers every smaller budget: credited(b) = min(warm, best incumbent <= b).
# Their pooled per-rung leads are emitted by p60_scale1800.py, and panel (b)
# asserts its plotted values against those macros.
TRAJ_SIZES = [('50x25', 'test_results/FAMILY/50x25', 50, 'Fifty'),
              ('80x25', 'test_results/FAMILY/80x25', 80, 'Eighty')]
TRAJ_RUNGS = [1, 2, 5, 10, 30, 60, 120, 300, 600, 1800]
TRAJ_WORD = {1: 'One', 2: 'Two', 5: 'Five', 10: 'Ten', 30: 'Thirty',
             60: 'Sixty', 120: 'HundredTwenty', 300: 'ThreeHundred',
             600: 'SixHundred', 1800: 'EighteenHundred'}
SCALE_RESOLVED = {('10x25', 5), ('10x25', 30)}
# The repaired best-of-four warm-start cache (scripts/p41_repaired_warm_cache.py):
# the 5s and 30s rungs were re-solved from it, the 1s/2s/10s rungs have their
# credited value floored at it. Mirrors p30_anytime exactly.
_ANY_REPAIRED = json.load(open(f'{OR}/warmstart_cache_repaired/makespans.json'))


def _any_rung_dir(budget):
    return (f'{OR}/10x25_fairwarmR{budget}s' if budget in ANY_RERUN
            else f'{OR}/10x25_fairwarm{budget}s')


def _any_credited(budget, regime):
    """Per-instance CREDITED solver makespans against the repaired hint, for one
    (budget, regime). Exactly p30_anytime.cells() without the improved-flag."""
    out = []
    rerun = budget in ANY_RERUN
    for p in sorted(glob.glob(f'{_any_rung_dir(budget)}/{regime}/*.json')):
        d = json.load(open(p))
        x = d[0] if isinstance(d, list) else d
        obj, warm = x['objective'], x.get('warm_makespan')
        if warm is None:
            return None                    # not a fair-warm run; treat as missing
        if rerun:
            out.append(min(obj, warm))     # warm_makespan is the repaired hint
        else:
            hint = _ANY_REPAIRED[regime][x['instance']]['repaired_ms']
            out.append(min(hint, min(obj, warm)))
    return np.array(out, dtype=float) if out else None


def _any_policy(regime):
    """Per-instance greedy-policy makespan, meaned across p30's three seeds."""
    arrs = []
    for s in ANY_SEEDS:
        p = f'{RES}/Result_{ANY_JOINT}{s}_{regime}.npy'
        if not os.path.exists(p):
            return None
        a = np.load(p)
        arrs.append((a[:, 0] if a.ndim > 1 else a).astype(float))
    return np.mean(arrs, axis=0)


def _scale_credited(size, budget, regime, cache):
    """p39's credited makespan: the better of the search result and the hint,
    the hint being the repaired warm start wherever the tier was not re-solved
    from it. Kept identical to p39 on purpose; if one of them is changed the
    gate below stops the figure rather than letting the two drift."""
    rep = cache.get(regime)
    d = (f'{OR}/{size}_fairwarmR{budget}s'
         if (size, budget) in SCALE_RESOLVED else f'{OR}/{size}_fairwarm{budget}s')
    out = []
    for p in sorted(glob.glob(f'{d}/{regime}/*.json')):
        x = json.load(open(p))
        x = x[0] if isinstance(x, list) else x
        best = min(x['objective'], x['warm_makespan'])
        if (size, budget) not in SCALE_RESOLVED and rep is not None:
            best = min(best, rep[x['instance']]['repaired_ms'])
        out.append(best)
    return np.array(out, dtype=float)


def _scale_frontier():
    """(pooled lead per (size, rung), per-regime crossing budget per size).

    The pooled lead is p30's estimator carried across regimes: the mean of the
    per-instance relative gaps, all 8 regimes concatenated, positive = policy
    ahead. Returns None if any size is short a rung, so the figure is skipped
    rather than drawn from a partial ladder."""
    pooled, crossing = {}, {}
    for size, res, n_inst, word, _jobs, cdir in SCALE:
        cpath = f'{OR}/{cdir}/makespans.json'
        if not os.path.exists(cpath):
            return None, f'{cdir} missing: p41 has not built the repaired hint'
        cache = json.load(open(cpath))
        rungs = [b for b in SCALE_RUNGS
                 if all(len(glob.glob(
                     (f'{OR}/{size}_fairwarmR{b}s' if (size, b) in SCALE_RESOLVED
                      else f'{OR}/{size}_fairwarm{b}s') + f'/{r}/*.json')) == n_inst
                     for r in ANY_REGIMES)]
        if not rungs:
            return None, f'{size}: no complete rung'
        pol = {}
        for r in ANY_REGIMES:
            a = []
            for s in ANY_SEEDS:
                p = f'{res}/Result_{ANY_JOINT}{s}_{r}.npy'
                if not os.path.exists(p):
                    return None, f'{size}/{r}: policy results missing'
                x = np.load(p)
                a.append((x[:, 0] if x.ndim > 1 else x).astype(float))
            pol[r] = np.mean(a, axis=0)
        cross = {}
        for b in rungs:
            gaps = []
            for r in ANY_REGIMES:
                sol = _scale_credited(size, b, r, cache)
                if len(sol) != len(pol[r]):
                    return None, f'{size}/{r}@{b}s: {len(sol)} solves vs {len(pol[r])}'
                gaps.append(100.0 * (sol - pol[r]) / pol[r])
                if sol.mean() <= pol[r].mean() and r not in cross:
                    cross[r] = b
            pooled[(size, b)] = float(np.concatenate(gaps).mean())
        crossing[size] = (cross, rungs)
    return (pooled, crossing), None


def f5_anytime():
    """Line plot of the policy's lead over the credited fair-warm CP-SAT solver as
    the solver's time budget grows (E5). One line per regime; y is the mean
    per-instance relative gap in percent (positive = policy ahead), the same
    quantity macros_anytime.tex reports; x is the budget at the five measured
    rungs (1, 2, 5, 10, 30 s), log-scaled.

    The setup-bearing regimes (S, LS, SW, LSW) are drawn as saturated solid lines
    and the rest (N, L, W, LW) as muted dashed lines, because the story splits on
    exactly that line: where setups are active the search returns nothing better
    than its warm start until after 5 s, so those lines stay flat and positive,
    then dip toward zero by 30 s.
    """
    # A rung is only admissible if all 8 regimes x 100 instances are present, the
    # same 800-cell completeness gate p30 enforces before it will report a rung.
    for b in ANY_RUNGS:
        if len(glob.glob(f'{_any_rung_dir(b)}/*/*.json')) != 800:
            _stale('f5_anytime', f'fair-warm rung {b}s is incomplete '
                                 f'(need 800 cells: 8 regimes x 100 instances)')
            return
    if _any_policy('N') is None:
        _stale('f5_anytime', 'joint policy results (joint-v1, 3 seeds) missing')
        return

    # Compute the plotted grid: lead[(rung, regime)] = mean per-instance relative
    # gap (%), positive = policy ahead.
    lead = {}
    for r in ANY_REGIMES:
        pol = _any_policy(r)
        for b in ANY_RUNGS:
            sol = _any_credited(b, r)
            lead[(b, r)] = float((100.0 * (sol - pol) / pol).mean())

    # INTEGRITY GATE. The per-regime frontier is no longer a panel of this
    # figure (it lives in the supplement's frontier table), but the check stays:
    # macros_anytime.tex is p30's output and is ground truth, and every value
    # recomputed here must reproduce its \Any<Word><Regime>Lead macro to two
    # decimals. p9 compares timestamps, not values, so without this a drifted
    # result file would change the supplement's table with nothing to catch it.
    txt = open('paper/macros_anytime.tex').read()
    for b in ANY_RUNGS:
        for r in ANY_REGIMES:
            m = re.search(r'\\newcommand\{\\Any' + ANY_WORD[b] + r
                          + r'Lead\}\{([+-][0-9.]+)\}', txt)
            if m is None:
                raise SystemExit(f'f5: macro \\Any{ANY_WORD[b]}{r}Lead not found '
                                 f'in macros_anytime.tex')
            macro = float(m.group(1))
            if abs(round(lead[(b, r)], 2) - macro) > 0.005:
                raise SystemExit(
                    f'f5 REFUSING: {r} at {b}s computes {lead[(b, r)]:+.2f}% but '
                    f'macros_anytime.tex says {macro:+.2f}%. The figure and the '
                    f'paper disagree; fix the input, do not ship the figure.')

    # Fixed entity -> (colour, marker) mapping, Okabe-Ito, never cycled. Setup
    # regimes get saturated hues and solid lines; the rest get muted hues and a
    # dashed stroke, so the two families separate by dash even in greyscale, and
    # every regime is told apart by its marker without relying on colour.
    STYLE = {
        'S':   ('#D55E00', 'o', '-'),      # vermillion
        'LS':  ('#0072B2', 's', '-'),      # blue
        'SW':  ('#009E73', '^', '-'),      # bluish green
        'LSW': ('#CC79A7', 'D', '-'),      # reddish purple
        'N':   ('#000000', 'v', (0, (4, 2))),
        'L':   ('#E69F00', 'X', (0, (4, 2))),   # orange
        'W':   ('#56B4E9', 'P', (0, (4, 2))),   # sky blue
        'LW':  ('#999999', '*', (0, (4, 2))),   # grey
    }

    # ---------------- panel (b) inputs: every alternative's gap TO the policy
    # at matched budgets, pooled by setup-bearing vs not (user directive
    # 2026-08-11: the main exhibits must state WHEN this policy is the best
    # option). Positive = the alternative is worse. Asserted against
    # macros_gal.tex (p52) below, the same discipline as panel (a).
    def _ga(tag, r):
        return _load(f'GA{tag}', r)

    def _gap_pool(vals_by_reg, regs):
        gaps = []
        for r in regs:
            pol = _any_policy(r)
            v = vals_by_reg[r]
            n = min(len(v), len(pol))
            gaps.append(100.0 * (v[:n] - pol[:n]) / pol[:n])
        return float(np.concatenate(gaps).mean())

    METHODS = {                        # label -> (per-regime loader, budgets)
        'GA':     (lambda b, r: _ga({1: '1s', 5: '5s', 30: ''}[b], r),
                   [1, 5, 30]),
        'CP-SAT': (lambda b, r: _any_credited(b, r), ANY_RUNGS),
        'best rule': (lambda b, r: _load('SPT', r), None),   # budget-free
    }
    MAC_NAME = {('GA', 1): 'GaOne', ('GA', 5): 'GaFive', ('GA', 30): 'GaThirty',
                ('CP-SAT', 1): 'CpOne', ('CP-SAT', 2): 'CpTwo',
                ('CP-SAT', 5): 'CpFive', ('CP-SAT', 10): 'CpTen',
                ('CP-SAT', 30): 'CpThirty', ('best rule', None): 'Rule'}
    galp = 'results/ga_ladder_gaps.json'
    if not os.path.exists(galp):
        _stale('f5_anytime', 'ga_ladder_gaps.json missing -- run p52 first')
        return
    galgrid = json.load(open(galp))
    bvals = {}                          # (label, budget, fam) -> pooled gap
    for label, (loader, budgets) in METHODS.items():
        for b in (budgets or [None]):
            per = {r: loader(b, r) for r in ANY_REGIMES}
            if any(v is None for v in per.values()):
                _stale('f5_anytime', f'{label} at {b}s incomplete')
                return
            for fam, regs in (('Setup', ANY_SETUP),
                              ('Other', [r for r in ANY_REGIMES
                                         if r not in ANY_SETUP])):
                g = _gap_pool(per, regs)
                key = MAC_NAME[(label, b)]
                ref = galgrid.get(key, {}).get(f'Gap{fam}')
                if ref is None or abs(g - ref) > 0.005:
                    raise SystemExit(
                        f'f5(a) REFUSING: {key} Gap{fam} computes {g:+.2f} '
                        f'but p52 recorded {ref}. The figure and the paper '
                        f'disagree; fix the input, do not ship the figure.')
                bvals[(label, b, fam)] = g

    # ---------------- panel (b) data, and the gate that keeps it honest.
    (scale, why) = _scale_frontier()
    if scale is None:
        _stale('f5_anytime', f'panel (b): {why}')
        return
    spooled, scross = scale
    # CORRECTNESS GATE, the same discipline panels (a) and (b) carry: every
    # per-regime crossing this figure derives must reproduce the \ScaleCross
    # macro p39 emitted. A changed credited rule on either side dies here.
    stxt = open('paper/macros_anytime_scale.tex').read()
    for size, _res, _n, word, _jobs, _c in SCALE:
        cross, rungs = scross[size]
        for r in ANY_REGIMES:
            m = re.search(r'\\newcommand\{\\ScaleCross' + word + r
                          + r'\}\{([0-9]+)\}', stxt)
            if m is None:
                raise SystemExit(f'f5(b): macro \\ScaleCross{word}{r} not found')
            want = int(m.group(1))          # 0 = never crosses within the ladder
            got = cross.get(r, 0)
            if got != want:
                raise SystemExit(
                    f'f5(b) REFUSING: {size}/{r} crosses at {got or "never"} but '
                    f'macros_anytime_scale.tex says {want or "never"}. The figure '
                    f'and the paper disagree; fix the input, not the figure.')

    # ---------------- the 1800 s cells, from their anytime trajectories.
    traj = {}
    for size, res, jobs, word in TRAJ_SIZES:
        pol = {}
        for r in ANY_REGIMES:
            a = []
            for sd in ANY_SEEDS:
                pth = f'{res}/Result_{ANY_JOINT}{sd}_{r}.npy'
                if not os.path.exists(pth):
                    _stale('f5_anytime', f'{size}/{r}: policy results missing')
                    return
                x = np.load(pth)
                a.append((x[:, 0] if x.ndim > 1 else x).astype(float))
            pol[r] = np.mean(a, axis=0)
        for b in TRAJ_RUNGS:
            gaps = []
            for r in ANY_REGIMES:
                cred = []
                fs = sorted(glob.glob(
                    f'{OR}/{size}_fairwarm1800s/{r}/*.json'))
                if len(fs) != 10:
                    _stale('f5_anytime', f'{size}/{r}: {len(fs)}/10 solves')
                    return
                for fp in fs:
                    d = json.load(open(fp))
                    best = d['warm_makespan']
                    for t, ub, _ in d.get('anytime', []):
                        if t <= b and ub < best:
                            best = ub
                    cred.append(best)
                cred = np.array(cred)
                n = min(len(cred), len(pol[r]))
                gaps.append(100 * (cred[:n] - pol[r][:n]) / pol[r][:n])
            traj[(size, b)] = float(np.concatenate(gaps).mean())
    # GATE: every plotted value must reproduce p60's macro to two decimals.
    ttxt = open('paper/macros_scale1800.tex').read()
    for size, _res, _j, word in TRAJ_SIZES:
        for b in TRAJ_RUNGS:
            m = re.search(r'\\newcommand\{\\Sc' + word + TRAJ_WORD[b]
                          + r'\}\{([+-][0-9.]+)\}', ttxt)
            if m is None:
                raise SystemExit(f'f5(b): macro Sc{word}{TRAJ_WORD[b]} missing')
            if abs(round(traj[(size, b)], 2) - float(m.group(1))) > 0.005:
                raise SystemExit(
                    f'f5(b) REFUSING: {size}@{b}s computes '
                    f'{traj[(size, b)]:+.2f} but p60 emitted {m.group(1)}.')

    # Sized to print at one full \columnwidth (3.5 in): the canvas IS the print
    # size, so every font below lands on paper at its stated point size.
    # The panels do NOT share an axis: panel (a)'s ladder stops at 30 s and
    # panel (b) now runs to 1800 s, so a shared axis would squeeze (a)'s data
    # into the left fifth of its width and leave the rest empty. Each panel
    # labels its own ticks instead.
    fig, (bx, cx) = plt.subplots(
        2, 1, figsize=(3.5, 3.62), dpi=300,
        gridspec_kw={'height_ratios': [0.92, 0.88], 'hspace': 0.76})
    from matplotlib.lines import Line2D
    # ---------------- panel (a): every alternative's gap TO the policy at
    # matched budgets, pooled setup-bearing (solid) vs the rest (dashed).
    # Above the zero line the released policy is the better option; the
    # crossings show where each alternative overtakes it.
    bx.axhline(0, color='0.35', lw=0.6, zorder=1)
    # the policy's own latency, from the SAME artifact p13 turns into
    # \PolicySec -- never hardcoded, so a re-measurement moves the marker
    with open('results/policy_latency_cpu_4core_blind.json') as _lf:
        _psec = json.load(_lf)['worst_regime_median_s']
    bx.plot([_psec], [0.0], marker='v', ms=5, color='black', zorder=6,
            clip_on=False)
    MSTYLE = {'GA': ('#009E73', '^'), 'CP-SAT': ('#0072B2', 'o'),
              'best rule': ('#E69F00', 's')}
    for label, (loader, budgets) in METHODS.items():
        c, mk = MSTYLE[label]
        for fam, ls in (('Setup', '-'), ('Other', (0, (4, 2)))):
            if budgets is None:
                y = bvals[(label, None, fam)]
                bx.axhline(y, color=c, ls=ls, lw=1.1, alpha=0.9, zorder=2)
                continue
            xs = budgets
            ys = [bvals[(label, b, fam)] for b in xs]
            bx.plot(xs, ys, ls=ls, color=c, marker=mk, ms=3.8, mew=0.6,
                    markeredgecolor='white', lw=1.3, zorder=4)
    bx.set_xscale('log')
    # The axis has to start below the policy's own measured latency, or its
    # marker is clipped onto the left spine and reads as a wrong x. A 0.4 s
    # lower limit with a labelled 0.5 rung puts the marker inside the axes at
    # its true value, and the measured rungs 1, 2, 5, 10, 30 keep their labels.
    bx.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 30]))
    bx.xaxis.set_major_formatter(
        FixedFormatter(['0.5', '1', '2', '5', '10', '30']))
    bx.xaxis.set_minor_locator(FixedLocator([]))
    bx.set_xlim(0.40, 35)
    bx.set_xlabel('time budget per instance (s), log scale', labelpad=1)
    # Same discipline as panel (a), but the band goes on top: below the curves
    # there is no room (CP-SAT without setups runs to -4.7 at the last rung),
    # while nothing rises above +2.5, so one legend row clears every curve.
    bx.set_ylim(-5.0, 5.4)
    bx.set_ylabel('gap to\nPACT (%)')
    bx.set_title('(a) alternatives vs. PACT at matched budgets',
                 fontsize=8, pad=6)
    bx.grid(axis='y', lw=0.3, alpha=0.4)
    bx.set_axisbelow(True)
    # The legend names the per-instance best of the four textbook rules by
    # the term the manuscript defines (Section 6.2), not by the internal key.
    SHOWN = {'best rule': 'rule pool'}
    mh = [Line2D([0], [0], color=MSTYLE[k][0], marker=MSTYLE[k][1], ls='-',
                 lw=1.3, ms=3.8, mew=0.6, markeredgecolor='white',
                 label=SHOWN.get(k, k))
          for k in MSTYLE]
    # The solid/dashed key lives in the caption, not here: six entries need
    # three rows, and three rows do not fit under the curves without covering
    # them.
    mh += [Line2D([0], [0], color='black', ls='none', marker='v', ms=5,
                  label=f'PACT ({_psec:.2f} s)')]
    # One row, sized so the box stays INSIDE the axes: a legend wider than
    # its panel pushes the saved canvas past the 3.5 in column, and the whole
    # figure is then scaled down in the manuscript, printing every label below
    # its stated point size.
    leg2 = bx.legend(handles=mh, ncol=4, loc='upper left', fontsize=5.9,
                     handlelength=1.0, handletextpad=0.3, columnspacing=0.45,
                     labelspacing=0.25, borderpad=0.25, borderaxespad=0.25,
                     framealpha=0.92, edgecolor='0.75')
    leg2.get_frame().set_linewidth(0.4)

    # ---------------- panel (b): the SAME frontier at three shop sizes, pooled
    # over all eight regimes. Panel (a) is the training size alone, where the
    # solver eventually passes the policy; this panel is what happens to that
    # crossing when the shop grows, which is the condition a module factory
    # actually runs at.
    cx.axhline(0, color='0.35', lw=0.6, zorder=1)
    SSTYLE = {10: ('#999999', 'o', (0, (4, 2))),
              20: ('#0072B2', 's', '-'),
              30: ('#D55E00', '^', '-'),
              50: ('#009E73', 'D', '-'),
              80: ('#CC79A7', 'v', '-')}
    for size, _res, _n, _w, jobs, _c in SCALE:
        col, mk, ls = SSTYLE[jobs]
        xs = [b for b in SCALE_RUNGS if (size, b) in spooled]
        cx.plot(xs, [spooled[(size, b)] for b in xs], ls=ls, color=col,
                marker=mk, ms=3.8, mew=0.6, markeredgecolor='white',
                lw=1.5 if jobs > 10 else 1.2, zorder=4 if jobs > 10 else 3,
                label=f'{jobs} modules')
    for size, _res, jobs, _w in TRAJ_SIZES:
        col, mk, ls = SSTYLE[jobs]
        cx.plot(TRAJ_RUNGS, [traj[(size, b)] for b in TRAJ_RUNGS], ls=ls,
                color=col, marker=mk, ms=3.8, mew=0.6, markeredgecolor='white',
                lw=1.5, zorder=5, label=f'{jobs} modules')
    cx.set_xscale('log')
    _ticks = [1, 5, 30, 120, 600, 1800]
    cx.xaxis.set_major_locator(FixedLocator(_ticks))
    cx.xaxis.set_major_formatter(FixedFormatter([str(b) for b in _ticks]))
    cx.xaxis.set_minor_locator(FixedLocator([]))
    cx.set_xlim(0.55, 2200)
    # Room for one legend row above every curve: the 30-module lead peaks at
    # +3.4 and the 10-module curve bottoms at -2.6, so the axis runs to +6.6 and
    # the legend sits in empty space rather than over the data.
    cx.set_ylim(-4.0, 8.2)
    cx.set_xlabel('time budget per instance (s), log scale')
    cx.set_ylabel('PACT lead over\ncredited solver (%)')
    cx.set_title('(b) the same frontier as the shop grows, all regimes pooled',
                 fontsize=8, pad=6)
    cx.grid(axis='y', lw=0.3, alpha=0.4)
    cx.set_axisbelow(True)
    leg3 = cx.legend(ncol=5, loc='upper left', fontsize=5.9, handlelength=1.0,
                     handletextpad=0.3, columnspacing=0.45, labelspacing=0.25,
                     borderpad=0.25, borderaxespad=0.25, framealpha=0.92,
                     edgecolor='0.75')
    leg3.get_frame().set_linewidth(0.4)

    fig.tight_layout(pad=0.3, h_pad=0.9)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f5_anytime.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F5 written (macro assertions passed: panel (a) all pooled gaps vs '
          'macros_gal; panel (b) all '
          f'{len(SCALE) * len(ANY_REGIMES)} size x regime crossings vs '
          'macros_anytime_scale)')


def f23_parity_zeroshot(joint='10x25+family+jointblind-v1',
                        cond='10x25+family+joint-v1', margin=1.0):
    # PROTAGONIST SWAP (2026-08-11): panel (a)'s measured policy and panel
    # (b)'s zero-shot row are the RELEASED token-blind joint; the conditioned
    # joint appears in panel (b) as the evaluated pathway.
    """E1 and E2 in one double-column float: parity forest, then zero-shot ranks.

    The two panels answer one question in two steps: does one policy match the
    specialists on the regimes it trained on (a), and does it still match them
    on compositions it never saw (b). Sharing a float saves the page budget a
    caption and a float separation, and the panels are read together anyway.
    """
    regimes = ['N', 'L', 'S', 'W', 'LS', 'LW']
    los, his, mids = [], [], []
    for r in regimes:
        j = _load_joint(joint, r)
        sp = _load_spec(r)
        if j is None or sp is None:
            _stale('f23_parity_zeroshot', f'joint or specialist missing for {r}')
            return
        d = 100.0 * (j - sp) / sp
        lo, hi = _paired_boot(d)
        los.append(lo); his.append(hi); mids.append(float(d.mean()))

    def _verdict(lo, hi):
        if hi < 0:
            return 'superior'          # whole CI below zero; at these widths
                                       # it also sits inside the margin band
        if -margin < lo and hi < margin:
            return 'equivalent'
        if lo > margin:
            return 'inferior'
        return 'inconclusive'

    STYLE = {'superior':     ('#009E73', '^'),
             'equivalent':   (C_JOINT,   'o'),
             'inconclusive': ('#999999', 's'),
             'inferior':     ('#D55E00', 'D')}

    # Model bases, NOT 'greedy+...' tags: every learned row is seed-pooled through
    # _pooled(), so panel (b) compares three seeds against three seeds, as its
    # caption says and as panel (a) already did. The rules carry no seed.
    rows = [('zero-shot joint', joint, None),
            ('conditioned variant', cond, None),
            ('composition specialist', '10x25+family+spec-{R}', None),
            ('best dispatching rule', None, ('FIFO', 'MOR', 'SPT', 'MWKR'))]
    held = ['SW', 'LSW']
    M = np.full((len(rows), len(held)), np.nan)
    for c, reg in enumerate(held):
        for r, (label, base, pdrs) in enumerate(rows):
            if pdrs:
                vals = [_load(pp, reg) for pp in pdrs]
                vals = [v.mean() for v in vals if v is not None]
                M[r, c] = min(vals) if vals else np.nan
            else:
                v = _pooled(base.replace('{R}', reg), reg)
                M[r, c] = v.mean() if v is not None else np.nan
    if np.isnan(M).any():
        _stale('f23_parity_zeroshot', 'a method has no result on SW/LSW')
        return
    shade = np.zeros_like(M)
    for c in range(M.shape[1]):
        col = M[:, c]
        lo_, hi_ = np.nanmin(col), np.nanmax(col)
        shade[:, c] = (col - lo_) / (hi_ - lo_ + 1e-9)

    # Sized to print at 0.90\textwidth (6.44 in): the canvas IS the print size,
    # so every font below lands on paper at its stated point size.
    # Full text width (7.04 in = \textwidth - 9pt keyline padding) so the frame
    # spans the column band 1:1; height held so widening does not overflow.
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(7.04, 1.34), dpi=300,
        gridspec_kw={'width_ratios': [1.35, 1.0], 'wspace': 0.50})

    y = np.arange(len(regimes))[::-1]
    ax.axvspan(-margin, margin, color=C_SPEC, alpha=0.16, lw=0,
               label=f'±{margin:g}% margin')
    for xv in (-margin, margin):
        ax.axvline(xv, color=C_SPEC, lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.axvline(0, color='0.35', lw=0.6, zorder=1)
    seen = set()
    for yi, lo, hi, m in zip(y, los, his, mids):
        v = _verdict(lo, hi)
        c, mk = STYLE[v]
        ax.plot([lo, hi], [yi, yi], lw=1.4, color=c, solid_capstyle='round', zorder=3)
        ax.plot([m], [yi], mk, ms=3.4, color=c, zorder=4,
                label=v if v not in seen else None)
        seen.add(v)
    ax.set_yticks(y); ax.set_yticklabels(regimes)
    ax.set_ylim(-0.7, len(regimes) - 0.3)
    # Left headroom so the legend sits in empty space: every CI starts at >= -0.3,
    # so the band left of -0.4 carries no data for the legend to cover.
    ax.set_xlim(-1.5, 1.9)
    ax.set_xlabel('paired makespan gap, PACT policy − specialist (%)')
    ax.set_ylabel('training regime')
    ax.grid(axis='x', lw=0.3, alpha=0.4); ax.set_axisbelow(True)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0)
    # Small legend tucked into the empty lower-left (all markers sit at x>=-0.3).
    leg = ax.legend(loc='lower left', ncol=1, handlelength=0.9,
                    handletextpad=0.4, fontsize=6.3, labelspacing=0.2,
                    borderpad=0.3, framealpha=0.92, edgecolor='0.75')
    leg.get_frame().set_linewidth(0.4)
    ax.set_title('(a) Specialist Parity (E1)', fontsize=9, pad=13)

    _cells(bx, shade, 'Blues', 0, 1.15)
    for c in range(M.shape[1]):
        best = int(np.nanargmin(M[:, c]))
        for r in range(M.shape[0]):
            bx.text(c, r, f'{M[r, c]:.1f}', ha='center', va='center',
                    fontsize=8, fontweight='bold' if r == best else 'normal',
                    color='white' if shade[r, c] > 0.62 else '0.15')
    bx.set_xticks(range(len(held)))
    bx.set_xticklabels([f'{h}\n(held out)' for h in held])
    bx.set_yticks(range(len(rows)))
    bx.set_yticklabels([r[0] for r in rows])
    bx.tick_params(length=0)
    for sp_ in bx.spines.values():
        sp_.set_visible(False)
    bx.set_title('(b) Compositional Zero-Shot (E2): Mean Makespan (h)',
                 fontsize=9, pad=13)

    fig.tight_layout(pad=0.3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f23_parity_zeroshot.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('F23 written')


def f6_load(src='results/load_axis.json'):
    """The load axis: the policy's lead over both rivals as the shop fills.

    Everything plotted comes from the file scripts/p53_load_axis.py writes, so
    the figure and the macros the text prints are one computation rather than
    two that can drift. Positive is the policy ahead, which is the reverse of
    p53's internal sign, and the flip happens once, here.
    """
    if not os.path.exists(src):
        _stale('f6_load', 'run scripts/p53_load_axis.py first')
        return
    with open(src) as f:
        D = json.load(f)
    jobs = [D['jobs'][s] for s in D['sizes']]
    spec = {r['size']: r for r in D['spec']}
    rule = {r['size']: r for r in D['rule']}

    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)

    # the eight regimes individually, so the concession is visible in the same
    # panel as the win rather than in a separate exhibit. The best and the worst
    # are labelled where they end, so the reader never has to consult the
    # caption to learn which line is which.
    ends = {}
    for reg in D['regimes']:
        y = [-next(c['mean'] for c in D['cells']
                   if c['size'] == s and c['regime'] == reg)
             for s in D['sizes']]
        ax.plot(jobs, y, '-', color='#9DC7E0', lw=0.6, zorder=1)
        ends[reg] = y[-1]
    for reg in (max(ends, key=ends.get), min(ends, key=ends.get)):
        ax.annotate(f'${reg}$', (jobs[-1], ends[reg]), textcoords='offset points',
                    xytext=(3, 0), va='center', ha='left', fontsize=6.4,
                    color='black')

    for key, src_d, color, style, label in (
            ('spec', spec, C_JOINT, '-', 'over the regime specialist'),
            ('rule', rule, C_PDR, '--', 'over SPT')):
        m = [-src_d[s]['mean'] for s in D['sizes']]
        lo = [-src_d[s]['hi'] for s in D['sizes']]
        hi = [-src_d[s]['lo'] for s in D['sizes']]
        ax.fill_between(jobs, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
        ax.plot(jobs, m, style, color=color, lw=1.6, marker='o', ms=3.4,
                zorder=3, label=label)

    ax.axhline(0, color='black', lw=0.7, zorder=2)

    # the documented floor batch: 3034 modules over two 56-storey towers
    # (United Tec Construction project record, Avenue South Residence)
    floor = 3034 / (2 * 56)
    ax.axvline(floor, color='#777777', lw=0.8, ls=(0, (4, 2)), zorder=2)
    lo_y, hi_y = ax.get_ylim()
    ax.text(floor + 0.4, lo_y + 0.62 * (hi_y - lo_y),
            'per-floor batch\n(one floor of the\ndocumented tower)', ha='left', va='center',
            fontsize=6.2, color='black', linespacing=1.2)

    ax.set_xticks(jobs)
    ax.set_xlim(jobs[0] - 1.2, jobs[-1] + 2.2)
    ax.set_xlabel(f'modules scheduled together '
                  f'({D["n_machines"]} stations throughout)', fontsize=7.2)
    ax.set_ylabel('PACT ahead by (%)', fontsize=7.2)
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', color='#DDDDDD', lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=6.6, frameon=False, loc='upper left',
              handlelength=1.8, borderaxespad=0.2)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}/f6_load.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('  wrote f6_load')


def _assert_times_only():
    """Fail loudly if any figure PDF embeds a face that is not the Times clone.

    One `$\\pm$` is enough to pull STIXGeneral into a figure, and nothing about
    the rendered plot looks wrong when it happens: the paper just ships two
    typefaces. pdffonts is the only place it shows, so check it here rather
    than trusting the next reader to run it.
    """
    import subprocess
    allowed = _SERIF.replace(' ', '')          # pdffonts drops the space
    bad = []
    for pdf in sorted(glob.glob(f'{OUT}/*.pdf')):
        try:
            out = subprocess.run(['pdffonts', pdf], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            print('font check skipped (pdffonts unavailable)')
            return
        for line in out.splitlines()[2:]:
            name = line.split()[0].split('+')[-1] if line.split() else ''
            if name and not name.startswith(allowed):
                bad.append(f'{os.path.basename(pdf)}: {name}')
    if bad:
        raise SystemExit('NON-TIMES FONT IN A FIGURE:\n  ' + '\n  '.join(bad))
    print(f'font check: every figure PDF is {_SERIF} only')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    only = None
    for i, a in enumerate(_ARGV):
        if a == '--only':
            only = _ARGV[i + 1]
    if only in (None, 'f2'):
        f2_parity()
    if only in (None, 'f3'):
        f3_zeroshot()
    if only in (None, 'f23'):
        f23_parity_zeroshot()
    if only in (None, 'f4'):
        f4_causality()
    if only in (None, 'f5'):
        f5_anytime()
    if only in (None, 'f6'):
        f6_load()
    _assert_times_only()
