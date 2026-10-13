"""Process-route figure: one reinforced-concrete wet-area module (RC-wet).

Draws the 22 operations of ROUTES['RC-wet'] in ppvc_instance_generator.py as
one horizontal chain of boxes, grouped into the four route segments (shell,
MEP fit-out, wet-area finishing, assembly and dispatch) and shaded by the
generator's operation type. The mandatory waits (time lags) are drawn as
hatched gaps after the operations that carry them, and each wait type's lag
range is read from the generator and printed once in the key, never per box.

Every operation name, type and lag range comes from the generator, so the
figure cannot drift from the benchmark.

Output: paper/figures/f7_route.pdf (and .png), 7.0 x 2.2 in, 8 pt Times.
Usage: python scripts/make_route_figure.py   (run from the repository root)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ppvc_instance_generator import ROUTES, _RC_SHELL, _MEP, _WET_FINISH, _ASSEMBLY

OUT = 'paper/figures'

# Same Times face as the manuscript body (see make_family_figures.py and
# make_termes_ttf.py for why the TrueType conversion is used).
from matplotlib import font_manager as _fm
_FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
for _f in glob.glob(os.path.join(_FONTDIR, 'TeXGyreTermes-*.ttf')):
    _fm.fontManager.addfont(_f)
_SERIF = 'TeX Gyre Termes'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [_SERIF],
    'mathtext.fontset': 'custom', 'mathtext.rm': _SERIF,
    'mathtext.it': f'{_SERIF}:italic', 'mathtext.bf': f'{_SERIF}:bold',
    'font.size': 8, 'pdf.fonttype': 42, 'hatch.linewidth': 0.5,
})
FS = 8

ROUTE = ROUTES['RC-wet']
assert len(ROUTE) == 22
SEGMENTS = [('Structural shell', len(_RC_SHELL)),
            ('MEP fit-out', len(_MEP)),
            ('Wet-area finishing', len(_WET_FINISH)),
            ('Assembly and dispatch', len(_ASSEMBLY))]
assert sum(n for _, n in SEGMENTS) == len(ROUTE)

# Plain operation names for the generator's identifiers.
NAMES = {
    'mould_setup': 'mould set-up', 'rebar_cage': 'rebar cage',
    'mep_cast_in': 'MEP cast-in items', 'pre_pour_QC': 'pre-pour check',
    'concrete_pour': 'concrete pour', 'demould_inspect': 'demould, inspect',
    'trial_assembly': 'trial assembly', 'plumbing': 'plumbing',
    'electrical': 'electrical', 'HVAC_ducting': 'HVAC ducting',
    'mep_test_QC': 'MEP tests', 'internal_partitions': 'internal partitions',
    'waterproofing': 'waterproofing', 'screed_tiling': 'screed, tiling',
    'plaster_skim': 'plaster skim', 'paint_coat_1': 'paint coat 1',
    'paint_coat_2': 'paint coat 2', 'finishing_QC': 'finishing check',
    'doors_windows': 'doors, windows', 'fitout_fixtures': 'fit-out fixtures',
    'final_QC': 'final check', 'wrap_protect_ship': 'wrap and ship',
}
# Wait type named after the operation that precedes it.
WAIT_OF = {'concrete_pour': 'curing', 'waterproofing': 'ponding test',
           'paint_coat_1': 'paint drying', 'paint_coat_2': 'paint drying'}

# Pastel fills by operation type (house palette); QC stays unfilled.
FILL = {'structural': '#c3cbd3', 'MEP': '#a8c6e3', 'finishing': '#f3cf8f',
        'assembly': '#eda9b0', 'QC': '#ffffff'}
TYPE_LABEL = [('structural', 'structural'), ('MEP', 'MEP'),
              ('finishing', 'finishing'), ('assembly', 'assembly'),
              ('QC', 'quality check')]

# Lag ranges per wait type, read from the generator.
RANGES = {}
for name, _s, _t, _d, (lo, hi) in ROUTE:
    if hi > 0:
        w = WAIT_OF[name]
        assert RANGES.get(w, (lo, hi)) == (lo, hi)
        RANGES[w] = (lo, hi)
assert set(RANGES) == {'curing', 'ponding test', 'paint drying'}
assert sum(1 for r in ROUTE if r[4][1] > 0) == 4

# ---- geometry, in inches (axes coordinates equal inches) --------------------
W_IN, H_IN = 7.0, 2.2
BOX_W, STEP, WAIT_W, SEG_GAP = 0.225, 0.048, 0.22, 0.10
BOX_B, BOX_T = 0.66, 1.72           # box bottom / top
X0 = 0.10

fig = plt.figure(figsize=(W_IN, H_IN))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W_IN)
ax.set_ylim(0, H_IN)
ax.axis('off')

x = X0
seg_edges = []
i = 0
wait_spans = []
prev_right = None
for s, (seg_name, n) in enumerate(SEGMENTS):
    if s > 0:
        x += SEG_GAP
    left = x
    for k in range(n):
        name, _st, otype, _d, (lo, hi) = ROUTE[i]
        if prev_right is not None:   # chain link from the previous box
            ax.plot([prev_right, x], [(BOX_B + BOX_T) / 2] * 2, color='black',
                    lw=0.6, zorder=1)
        ax.add_patch(Rectangle((x, BOX_B), BOX_W, BOX_T - BOX_B,
                               facecolor=FILL[otype], edgecolor='black',
                               lw=0.6, zorder=2))
        ax.text(x + BOX_W / 2, (BOX_B + BOX_T) / 2, NAMES[name], rotation=90,
                ha='center', va='center', fontsize=FS, zorder=3)
        x += BOX_W
        prev_right = x
        if hi > 0:
            wait_spans.append((x + 0.03, x + WAIT_W - 0.03, WAIT_OF[name]))
            x += WAIT_W
        elif k < n - 1:
            x += STEP
        i += 1
    seg_edges.append((left, prev_right, seg_name))
right_end = prev_right
assert right_end < W_IN - 0.02, right_end

# waits: a hatched band spanning the box height, then its name underneath
for a, b, w in wait_spans:
    ax.add_patch(Rectangle((a, BOX_B), b - a, BOX_T - BOX_B, facecolor='white',
                           edgecolor='#5a5a5a', hatch='////', lw=0.0, zorder=2))
    ax.plot([a, a], [BOX_B, BOX_T], color='#5a5a5a', lw=0.5, zorder=2)
    ax.plot([b, b], [BOX_B, BOX_T], color='#5a5a5a', lw=0.5, zorder=2)
    # 'drying' under each paint gap: the two gaps sit one box apart, too close
    # for the full wait name twice; the key gives the full name once.
    lab = 'drying' if w == 'paint drying' else w
    ax.text((a + b) / 2, BOX_B - 0.07, lab, ha='center', va='top', fontsize=FS)

# segment brackets above the chain
for left, right, seg_name in seg_edges:
    yb = BOX_T + 0.07
    ax.plot([left, left, right, right], [yb, yb + 0.06, yb + 0.06, yb],
            color='black', lw=0.6)
    t = ax.text((left + right) / 2, yb + 0.11, seg_name, ha='center',
                va='bottom', fontsize=FS, fontweight='bold')
    fig.canvas.draw()
    tb = t.get_window_extent().transformed(ax.transData.inverted())
    assert 0.0 < tb.x0 and tb.x1 < W_IN - 0.02 and tb.y1 < H_IN - 0.02, seg_name

# ---- key: operation types, then the wait patch with each lag range ----------
ky = 0.33
kx = X0
sw = 0.16
for t, lab in TYPE_LABEL:
    ax.add_patch(Rectangle((kx, ky - 0.055), sw, 0.11, facecolor=FILL[t],
                           edgecolor='black', lw=0.6))
    txt = ax.text(kx + sw + 0.05, ky, lab, ha='left', va='center', fontsize=FS)
    fig.canvas.draw()
    bb = txt.get_window_extent().transformed(ax.transData.inverted())
    kx = bb.x1 + 0.16
# second key row: the wait patch and every lag range, printed once
ky -= 0.20
kx = X0
ax.add_patch(Rectangle((kx, ky - 0.055), sw, 0.11, facecolor='white',
                       edgecolor='#5a5a5a', hatch='////', lw=0.5))
waits = ', '.join(f'{w} {lo}–{hi} h' for w, (lo, hi) in
                  (('curing', RANGES['curing']),
                   ('ponding test', RANGES['ponding test']),
                   ('paint drying', RANGES['paint drying'])))
txt = ax.text(kx + sw + 0.05, ky,
              f'mandatory wait, station free ({waits} per coat)',
              ha='left', va='center', fontsize=FS)
fig.canvas.draw()
bb = txt.get_window_extent().transformed(ax.transData.inverted())
assert bb.x1 < W_IN - 0.02, bb.x1

os.makedirs(OUT, exist_ok=True)
fig.savefig(f'{OUT}/f7_route.pdf')
fig.savefig(f'{OUT}/f7_route.png', dpi=300)
plt.close(fig)
print(f'wrote {OUT}/f7_route.pdf; chain ends at {right_end:.2f} in of {W_IN} in')
