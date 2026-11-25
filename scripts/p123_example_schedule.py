"""Example schedule of the policy on one test instance, for the main text.

The policy 10x25+family+xmask-v1-s304 builds one schedule of the first test
instance of data/FAMILY/10x25 under the combination with no timing rule active
(N) and one under the combination with all three rules active (LSW). The
script rolls the policy out greedily on the CPU with one torch thread, records
both schedules during the roll-out, checks them, and draws them as one Gantt
chart with a shared time axis.

Evaluation settings. eval_family.py is imported with the arguments that
scripts/run_eval_newseeds.sh gives the evaluation of this run:
  --family_model 10x25+family+xmask-v1-s304 --data data/FAMILY/10x25
  --base data/PPVC/10x25+ppvc-mixed --methods greedy --device cpu
  --device_id "" --torch_threads 1 --first_mask exact --mask_ties exact
  --tag _xmask, with PACT_FIRST_MASK and PACT_MASK_TOL unset.
Its load_model restores the network keys of the training configuration and
loads the checkpoint, and its mask_options resolves the two mask options. The
environment is built as eval_family.py builds it for a policy method: mask_mode
and pair_extra of the training configuration, both mask options, the delta
envelope of resolve_delta_mode, the instance's type lists, and the descriptor
vector computed from the descriptor and attached to the state.

Recording and checks follow scripts/p74_gantt.py. Before each step the script
reads the chosen operation and machine, the changeover the machine will pay
(its last routing class into the operation's class) and the environment's own
processing start on the hour axis (_earliest_start, a pure function of the
pre-step state). After the step it checks that this start plus the processing
time equals the completion the environment stored, and that the block of
changeover and processing starts no earlier than the job and the machine are
ready. The schedule then passes
  * p74_gantt.check_schedule: machine occupancy with the changeover and the
    processing as one block, job precedence with the minimum time lag, no
    block overlapping an unavailability window, and the independent validator
    constraint_family/reference_sim.validate_schedule on the recorded times;
  * the validator called as eval_family.py calls it, on the environment's
    completion times and the processing start ct - p;
  * the recorded completions against the environment's completions, and the
    largest of them against the environment's makespan;
  * the makespan against the value stored for instance index 0 in
    test_results/FAMILY/10x25_xmask/Result_greedy+<model>_<combination>.npy.
Any failure stops the script with the values involved; a makespan that
differs from the stored value stops it with the two values. The JSON and the
figure are written only when every check passes.

Figure: two panels with one shared time axis in hours, (a) no rule active and
(b) all three rules active. Rows are the 25 stations, grouped by station type
and labelled once per group. Bars are filled by product route, the changeover
is the hatched part at the start of a block, an unavailability window is a
grey band on the station's row, a minimum time lag is a thin line under the
bar of the operation that carries it, from its completion to the time the
module's next operation may start, closed by a short tick so that lags that
overlap on one row stay apart, and the makespan is a dashed vertical line. The drawing checks its own layout (no text overlaps another text or the
plot area, the legend clears the panels, the height stays within 4.7 in) and
pdffonts must list only the Times face.

Outputs:
  out/exact/results/example_schedule.json
      recorded schedules, unavailability windows, makespans, stored values and
      the result of every check
  manuscript/sections_new/figures/fq0_example_schedule.pdf and .png (300 dpi)
      6.324 in wide, the inner width of the manuscript's figure frame, so the
      figure is included unscaled and 8 pt prints at 8 pt

Usage (repository root):
  taskset -c 3 env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
      python -B scripts/p123_example_schedule.py [--check | --dry [--dry_dir DIR]]
  --check  list missing inputs; exit non-zero when any is missing
  --dry    check and draw a small synthetic schedule in a temporary folder
           (or DIR), without loading a model or importing torch
"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import numpy as np

MODEL = '10x25+family+xmask-v1-s304'
DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
TAG = '_xmask'
STORED = 'test_results/FAMILY/10x25' + TAG + '/Result_greedy+{model}_{combo}.npy'
COMBOS = ('N', 'LSW')
ACTIVE = {'N': (False, False, False), 'LSW': (True, True, True)}
INDEX = 0                                   # first descriptor in sorted order
OUT_FIG = 'manuscript/sections_new/figures/fq0_example_schedule'
OUT_JSON = 'out/exact/results/example_schedule.json'
OUT_TEX = 'manuscript/macros_example.tex'
MACRO_NAMES = {'N': 'ExmMakespanN', 'LSW': 'ExmMakespanLSW'}
FONT_FILES = ('scripts/fonts/TeXGyreTermes-Regular.ttf',
              'scripts/fonts/TeXGyreTermes-Bold.ttf')
TOL = 1e-6

# The arguments scripts/run_eval_newseeds.sh passes to eval_family.py for this
# run (job exact:10x25:xmask-v1-s304), restricted to the two combinations.
EVAL_ARGV = ['--family_model', MODEL, '--data', DATA, '--base', BASE,
             '--regimes', *COMBOS, '--methods', 'greedy',
             '--device', 'cpu', '--device_id', '', '--torch_threads', '1',
             '--first_mask', 'exact', '--mask_ties', 'exact', '--tag', TAG]


# ----------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------

def descriptor_paths(combo):
    return sorted(glob.glob(os.path.join(DATA, 'desc', combo, '*.desc.json')))


def missing_inputs():
    """Every input of the roll-out and the figure that is absent or unusable."""
    miss = []
    for p, what in (('eval_family.py', 'evaluation script'),
                    ('scripts/p74_gantt.py', 'recording and check functions'),
                    (f'trained_network/FAMILY/{MODEL}.pth', 'policy checkpoint'),
                    (f'train_log/FAMILY/config_{MODEL}.json',
                     'training configuration')):
        if not os.path.exists(p):
            miss.append(f'{p} ({what})')
    names = set()
    for combo in COMBOS:
        paths = descriptor_paths(combo)
        if len(paths) <= INDEX:
            miss.append(f'{DATA}/desc/{combo}/*.desc.json (descriptors of {combo})')
        else:
            names.add(os.path.basename(paths[INDEX]).replace('.desc.json', ''))
        sp = STORED.format(model=MODEL, combo=combo)
        if not os.path.exists(sp):
            miss.append(f'{sp} (stored greedy makespans of {combo})')
        else:
            a = np.load(sp)
            if a.ndim != 2 or a.shape[0] <= INDEX:
                miss.append(f'{sp} (no row {INDEX}; shape {a.shape})')
    for name in sorted(names):
        for ext in ('.fjs', '.meta.json'):
            p = os.path.join(BASE, name + ext)
            if not os.path.exists(p):
                miss.append(f'{p} (base instance)')
    for p in FONT_FILES:
        if not os.path.exists(p):
            miss.append(f'{p} (Times face of the figure)')
    if shutil.which('pdffonts') is None:
        miss.append('pdffonts (font check of the figure)')
    return miss


# ----------------------------------------------------------------------------
# roll-out and checks
# ----------------------------------------------------------------------------

def import_eval_family():
    """eval_family.py with the evaluation arguments of this run; the module
    parses them (and params.py its share) on import."""
    for var in ('PACT_FIRST_MASK', 'PACT_MASK_TOL'):
        os.environ.pop(var, None)            # the queue runs with both unset
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    saved = sys.argv
    sys.argv = [saved[0]] + list(EVAL_ARGV)
    try:
        import eval_family as EF
    finally:
        sys.argv = saved
    if (EF.ARG_MODEL, EF.ARG_DATA, EF.ARG_BASE, EF.ARG_TAG, EF.ARG_TORCH_THREADS,
            EF.ARG_TOKEN_MODE, EF.ARG_BLIND) != (MODEL, DATA, BASE, TAG, 1,
                                                 'true', ''):
        sys.exit('eval_family.py did not take the evaluation arguments')
    if EF.configs.device != 'cpu' or EF.configs.device_id != '':
        sys.exit(f'device {EF.configs.device!r}, id {EF.configs.device_id!r}; '
                 f'want cpu and an empty id')
    return EF


def stop(msg):
    raise SystemExit(f'p123: {msg}')


def rollout(EF, ppo, desc, jl, pt, meta, first_mask, mask_ties):
    """Greedy roll-out as eval_family.py runs it; returns the recorded schedule,
    the makespan and the unavailability windows per station."""
    import torch
    from common_utils import greedy_select_action
    from constraint_family.descriptor import compute_token
    from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                    resolve_delta_mode)
    from constraint_family.reference_sim import validate_schedule
    configs = EF.configs
    if getattr(desc, 'setup_mult', None) is not None or \
            getattr(desc, 'release', None) is not None:
        stop('the descriptor carries a setup multiplier or release dates')
    N, M = pt.shape
    env = FJSPEnvFamily(jl.shape[0], M, family_mode=True, blind_channels=(),
                        mask_mode=getattr(configs, 'mask_mode', 'nondelay'),
                        pair_extra=getattr(configs, 'pair_extra', 'none'),
                        first_mask=first_mask, mask_ties=mask_ties)
    dm = resolve_delta_mode(configs)
    env.delta_mode = dm
    env.reward_delta_env = (dm != 'zero')
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()), M)])
    env._attach_token()
    state = env.state

    first_op = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    amch = np.full(N, -1, dtype=int)
    rec = []
    while not env.done().all():
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, token=state.token_tensor)
        act = greedy_select_action(pi).cpu().numpy()
        job, mch = int(act[0]) // M, int(act[0]) % M
        op = int(env.candidate[0, job])
        amch[op] = mch

        # pre-step reads: changeover from the machine's last class, the job's
        # ready time, the machine's free time, and the environment's own
        # processing start on the hour axis
        setup = float(env.true_setup[0, env.mch_last_class[0, mch],
                                     env.op_class[0, op]] * env.c_bits[0, 1])
        job_ready = float(env.true_candidate_free_time[0, job])
        mch_free = float(env.true_mch_free_time[0, mch])
        st_pred = float(env._earliest_start(
            np.array([job]), np.array([mch]), np.array([op]),
            true_side=True)[0])

        state, _, _ = env.step(act)

        ct = float(env.true_op_ct[0, op])
        p = float(env.true_op_pt[0, op, mch])
        lag = float(env.true_op_lag[0, op])
        if abs(ct - p - st_pred) > TOL:
            stop(f'op {op}: start {st_pred} + processing {p} != completion {ct}')
        block = st_pred - setup
        if block < max(job_ready, mch_free) - TOL:
            stop(f'op {op}: block starts at {block}, before the job '
                 f'({job_ready}) or the machine ({mch_free}) is ready')
        rec.append({'step': len(rec), 'job': job, 'op': op,
                    'op_in_job': op - int(first_op[job]), 'machine': mch,
                    'route_class': int(env.op_class[0, op]),
                    'block_start': block, 'setup': setup,
                    'proc_start': st_pred, 'proc_time': p, 'completion': ct,
                    'lag': lag, 'successor_ready': ct + lag})

    if len(rec) != N:
        stop(f'{len(rec)} decisions for {N} operations')
    ms = float(env.current_makespan[0])
    op_ct = env.true_op_ct[0]
    by_op = {r['op']: r for r in rec}
    if any(abs(by_op[o]['completion'] - float(op_ct[o])) > TOL for o in range(N)):
        stop('a recorded completion differs from the environment')
    if abs(max(r['completion'] for r in rec) - ms) > TOL:
        stop(f'largest completion {max(r["completion"] for r in rec)} != '
             f'makespan {ms}')
    # the validator as eval_family.py calls it
    op_st = op_ct - pt[np.arange(N), amch]
    viol = validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    if viol:
        stop(f'validator (eval_family call): {viol[:5]}')

    outages = [[] for _ in range(M)]
    if env.n_win:
        for m in range(M):
            outages[m] = [(float(s), float(e)) for s, e in
                          zip(env.true_win_s[0, m], env.true_win_e[0, m])
                          if np.isfinite(s)]
    return rec, ms, outages


def check_schedule(rec, outages, jl, pt, desc):
    """p74_gantt.check_schedule: occupancy, precedence with the lag, windows,
    and the independent validator on the recorded times."""
    import p74_gantt as P74
    try:
        P74.check_schedule(rec, outages, jl, pt, desc)
    except AssertionError as e:
        stop(f'schedule check failed: {e}')


def compare_stored(combo, ms, stored_ms):
    if abs(ms - stored_ms) > TOL:
        stop(f'{combo}: makespan of the roll-out {ms:g} h differs from the '
             f'stored value {stored_ms:g} h; nothing written')


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

# Station types in the words of the benchmark table (make_route_figure.py).
STATION_WORDS = {'A': 'formwork', 'B': 'concrete pour', 'C': 'MEP fit-out',
                 'D': 'finishing', 'E': 'painting', 'F': 'assembly',
                 'G': 'dispatch', 'H': 'steel fabrication',
                 'Q': 'quality control'}
ROUTE_LABEL = ['RC-wet module', 'RC-dry module', 'steel-wet module',
               'steel-dry module']
# Fills of the route figure (Fig. 4): cool tints for the reinforced-concrete
# routes, warm tints for the steel routes.
ROUTE_FILL = ['#a8c6e3', '#c3cbd3', '#eda9b0', '#f3cf8f']
EDGE = '#2b2b2b'
WINDOW_FILL = '#dcdcdc'
LAG_COLOR = '#3f6f9a'
SEP_COLOR = '#a6a6a6'
FS = 8

W_IN = 455.33 / 72                 # inner width of the figure frame
H_MAX = 4.7
LEFT, RIGHT = 0.98, 0.12           # label column, room for the last tick
TOP, LEG_H, LEG_GAP = 0.01, 0.33, 0.06
TITLE_H, PANEL_GAP, BOTTOM = 0.16, 0.08, 0.35
PANEL_H = 1.77
GROUP_GAP = 0.7                    # extra row units between station groups
BAR_LO, BAR_HI = -0.38, 0.20       # bar extent about the row centre (y down)
LAG_Y = 0.36                       # lag line under the bar
LAG_TICK = 0.10                    # half-height of the end tick of a lag line
WIN_HALF = 0.47                    # half-height of a window band
LAG_LW, EDGE_LW, SEP_LW = 0.6, 0.35, 0.4


def station_groups(mch_type):
    """[(type letter, first row, last row)] of consecutive stations of one type,
    in row order."""
    from ppvc_instance_generator import STATION_TYPES
    groups = []
    for i, t in enumerate(np.asarray(mch_type).tolist()):
        if groups and groups[-1][0] == STATION_TYPES[t] and groups[-1][2] == i - 1:
            groups[-1] = (groups[-1][0], groups[-1][1], i)
        else:
            groups.append((STATION_TYPES[t], i, i))
    letters = [g[0] for g in groups]
    if len(set(letters)) != len(letters):
        stop(f'stations of one type are not adjacent: {letters}')
    return groups


def row_positions(groups):
    """y position of every station row, with GROUP_GAP between groups."""
    y = {}
    for gi, (_t, a, b) in enumerate(groups):
        for r in range(a, b + 1):
            y[r] = r + gi * GROUP_GAP
    return y


def _bbox_in(fig, artist):
    r = fig.canvas.get_renderer()
    return artist.get_window_extent(r).transformed(fig.dpi_scale_trans.inverted())


def _overlap(a, b, pad=0.0):
    return not (a.x1 + pad <= b.x0 or b.x1 + pad <= a.x0 or
                a.y1 + pad <= b.y0 or b.y1 + pad <= a.y0)


def draw(panels, mch_type, out_stem):
    """panels: [(title, {'operations', 'outages', 'makespan'})] top to bottom.
    Writes out_stem.pdf and out_stem.png; returns the layout record."""
    import p74_gantt as P74
    plt = P74.setup_fonts()                  # Times face of the manuscript
    plt.rcParams.update({'hatch.linewidth': 0.4, 'font.size': FS,
                         'axes.linewidth': 0.6, 'xtick.major.size': 2.5,
                         'xtick.major.pad': 2.0})
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import MultipleLocator

    groups = station_groups(mch_type)
    ypos = row_positions(groups)
    M = len(mch_type)
    y_last = ypos[M - 1]
    xmax = 20.0 * np.ceil(1.03 * max(p['makespan'] for _t, p in panels) / 20.0)

    n = len(panels)
    H = TOP + LEG_H + LEG_GAP + n * (TITLE_H + PANEL_H) + (n - 1) * PANEL_GAP \
        + BOTTOM
    if H > H_MAX + 1e-9:
        stop(f'figure height {H:.3f} in exceeds {H_MAX} in')
    fig = plt.figure(figsize=(W_IN, H))
    axw = W_IN - LEFT - RIGHT
    axes = []
    top = H - TOP - LEG_H - LEG_GAP
    for k in range(n):
        y0 = top - TITLE_H - PANEL_H
        axes.append(fig.add_axes([LEFT / W_IN, y0 / H, axw / W_IN, PANEL_H / H]))
        top = y0 - PANEL_GAP

    titles, labels = [], []
    for k, (ax, (title, res)) in enumerate(zip(axes, panels)):
        for m, spans in enumerate(res['outages']):
            for ws, we in spans:
                if ws >= xmax:
                    continue
                ax.add_patch(Rectangle((ws, ypos[m] - WIN_HALF),
                                       min(we, xmax) - ws, 2 * WIN_HALF,
                                       facecolor=WINDOW_FILL, edgecolor='none',
                                       lw=0, zorder=1))
        for r in res['operations']:
            y, fill = ypos[r['machine']], ROUTE_FILL[r['route_class']]
            if r['setup'] > 0:
                ax.add_patch(Rectangle((r['block_start'], y + BAR_LO),
                                       r['setup'], BAR_HI - BAR_LO,
                                       facecolor=fill, edgecolor=EDGE,
                                       lw=EDGE_LW, hatch='//////', zorder=3))
            ax.add_patch(Rectangle((r['proc_start'], y + BAR_LO),
                                   r['proc_time'], BAR_HI - BAR_LO,
                                   facecolor=fill, edgecolor=EDGE, lw=EDGE_LW,
                                   zorder=3))
            if r['lag'] > 0:
                # a line from the completion to the earliest start of the
                # module's next operation, closed by a short tick so that
                # lags that overlap on one row stay apart
                t0, t1 = r['completion'], r['successor_ready']
                ax.plot([t0, t1, np.nan, t1, t1],
                        [y + LAG_Y, y + LAG_Y, np.nan, y + LAG_Y - LAG_TICK,
                         y + LAG_Y + LAG_TICK],
                        color=LAG_COLOR, lw=LAG_LW, solid_capstyle='butt',
                        zorder=4)
        for gi in range(1, len(groups)):
            ysep = (ypos[groups[gi][1]] + ypos[groups[gi - 1][2]]) / 2
            ax.axhline(ysep, color=SEP_COLOR, lw=SEP_LW, zorder=0)
        ax.axvline(res['makespan'], color='black', lw=0.8, ls=(0, (4, 2)),
                   zorder=5)
        ax.set_xlim(0, xmax)
        ax.set_ylim(y_last + 0.5, -0.5)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.tick_params(axis='x', labelsize=FS, labelbottom=(k == n - 1))
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        titles.append(ax.text(0.0, 1.0 + 0.045 / PANEL_H, title,
                              transform=ax.transAxes, ha='left', va='bottom',
                              fontsize=FS))
        for t, a, b in groups:            # one label per group, centred
            labels.append(ax.text(-0.06 / axw, (ypos[a] + ypos[b]) / 2,
                                  STATION_WORDS[t],
                                  transform=ax.get_yaxis_transform(),
                                  ha='right', va='center', fontsize=FS))
    axes[-1].set_xlabel('Time (h)', fontsize=FS, labelpad=3)

    handles = [Patch(facecolor=ROUTE_FILL[c], edgecolor=EDGE, lw=EDGE_LW,
                     label=ROUTE_LABEL[c]) for c in range(4)]
    extra = [Patch(facecolor='white', edgecolor=EDGE, lw=EDGE_LW,
                   hatch='//////', label='changeover'),
             Patch(facecolor=WINDOW_FILL, edgecolor='none',
                   label='unavailability window'),
             Line2D([], [], color=LAG_COLOR, lw=LAG_LW * 1.3,
                    label='minimum time lag'),
             Line2D([], [], color='black', lw=0.8, ls=(0, (4, 2)),
                    label='makespan')]
    # two rows of four: the routes on top, the timing marks below
    order = [h for pair in zip(handles, extra) for h in pair]
    leg_y = (H - TOP - LEG_H) / H
    leg = fig.legend(handles=order, loc='upper left', ncol=4, mode='expand',
                     bbox_to_anchor=(LEFT / W_IN, leg_y, axw / W_IN, LEG_H / H),
                     frameon=True, fancybox=False, fontsize=FS,
                     handlelength=1.8, handleheight=0.8, handletextpad=0.5,
                     borderpad=0.35, labelspacing=0.3, borderaxespad=0.0)
    leg.get_frame().set_edgecolor('0.6')
    leg.get_frame().set_linewidth(0.5)

    # layout checks on the drawn figure
    fig.canvas.draw()
    fig_box = fig.bbox.transformed(fig.dpi_scale_trans.inverted())
    ax_boxes = [_bbox_in(fig, ax.patch) for ax in axes]
    leg_box = _bbox_in(fig, leg)
    ticks = [t for t in axes[-1].get_xticklabels()
             if t.get_visible() and t.get_text()
             and -1e-9 <= t.get_position()[0] <= xmax + 1e-9]
    texts = labels + titles + [axes[-1].xaxis.label] + ticks
    boxes = [_bbox_in(fig, t) for t in texts]
    for t, b in zip(texts, boxes):
        if not (b.x0 >= 0.005 and b.x1 <= fig_box.x1 - 0.005 and
                b.y0 >= 0.005 and b.y1 <= fig_box.y1 - 0.005):
            stop(f'text {t.get_text()!r} leaves the figure')
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if _overlap(boxes[i], boxes[j], pad=0.01):
                stop(f'texts {texts[i].get_text()!r} and '
                     f'{texts[j].get_text()!r} touch')
    for t, b in zip(labels + titles, [_bbox_in(fig, t) for t in labels + titles]):
        for ab in ax_boxes:
            if _overlap(b, ab, pad=0.01):
                stop(f'text {t.get_text()!r} touches a plot area')
    tick_len = plt.rcParams['xtick.major.size'] / 72.0
    for k in range(1, n):                    # tick marks of the panel above
        above = ax_boxes[k - 1]
        if _bbox_in(fig, titles[k]).y1 + 0.02 > above.y0 - tick_len:
            stop(f'title {titles[k].get_text()!r} touches the ticks above it')
    if not (leg_box.x0 >= -1e-6 and leg_box.x1 <= fig_box.x1 + 1e-6 and
            leg_box.y1 <= fig_box.y1):
        stop('the legend leaves the figure')
    title_top = max(_bbox_in(fig, titles[0]).y1, ax_boxes[0].y1)
    if leg_box.y0 - title_top < 0.04:
        stop(f'the legend is {leg_box.y0 - title_top:.3f} in above panel (a)')
    lab_gap = min(boxes[i].y0 - boxes[i + 1].y1
                  for i in range(len(groups) - 1))   # panel (a), top to bottom
    x0s = [ax.get_position().x0 for ax in axes]
    x1s = [ax.get_position().x1 for ax in axes]
    if max(x0s) - min(x0s) > 1e-9 or max(x1s) - min(x1s) > 1e-9 or \
            len({ax.get_xlim() for ax in axes}) != 1:
        stop('the panels do not share the time axis')

    os.makedirs(os.path.dirname(out_stem) or '.', exist_ok=True)
    fig.savefig(out_stem + '.pdf')
    fig.savefig(out_stem + '.png', dpi=300)
    plt.close(fig)
    try:
        P74.check_fonts(out_stem + '.pdf')   # only TeX Gyre Termes embedded
    except AssertionError as e:
        stop(f'font check failed: {e}')
    return {'width_in': round(W_IN, 4), 'height_in': round(H, 4),
            'time_axis_h': [0.0, float(xmax)],
            'legend_to_panel_in': round(float(leg_box.y0 - title_top), 4),
            'min_gap_between_group_labels_in': round(float(lab_gap), 4),
            'station_groups': [[STATION_WORDS[t], a, b] for t, a, b in groups]}


# ----------------------------------------------------------------------------
# synthetic schedule (--dry)
# ----------------------------------------------------------------------------

def synthetic_case(seed=7):
    """A small instance on twelve stations of all nine types, four modules of
    the four routes, and its schedule under no rule and under all three rules,
    computed by constraint_family/reference_sim.simulate."""
    from constraint_family.descriptor import ConstraintDescriptor, N_CLASSES
    from constraint_family.reference_sim import simulate
    rng = np.random.default_rng(seed)
    mch_type = np.array([0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 8])
    M = len(mch_type)
    routes = [[0, 1, 2, 3, 4, 8], [0, 1, 2, 3, 5, 6], [7, 2, 3, 4, 8, 6],
              [7, 2, 4, 5, 8, 6]]                     # station type per op
    jl = np.array([len(r) for r in routes])
    N = int(jl.sum())
    pt = np.zeros((N, M), dtype=int)
    o = 0
    for r in routes:
        for t in r:
            for m in np.flatnonzero(mch_type == t):
                pt[o, m] = int(rng.integers(3, 9))
            o += 1
    op_class = np.repeat(np.arange(4), jl)
    lag = np.zeros(N)
    lag[[1, 3, 9, 15]] = [6, 4, 5, 3]
    setup = rng.integers(1, 4, size=(N_CLASSES + 1, N_CLASSES))
    windows = [[] for _ in range(M)]
    for m in range(M):
        s = int(rng.integers(5, 30))
        windows[m] = [(s, s + int(rng.integers(2, 5)))]
    descs = {'N': ConstraintDescriptor((False, False, False), np.zeros(N),
                                       op_class, horizon=60),
             'LSW': ConstraintDescriptor((True, True, True), lag, op_class,
                                         setup=setup, windows=windows,
                                         horizon=60)}
    # decisions: modules in turn, each operation on the first compatible
    # station of its type that is free earliest in this simple sequence
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    actions, nxt, free = [], first.copy(), np.zeros(M)
    for _k in range(int(jl.max())):
        for j in range(len(jl)):
            if nxt[j] < first[j] + jl[j]:
                ms = np.flatnonzero(pt[nxt[j]] > 0)
                m = int(ms[np.argmin(free[ms])])
                free[m] += pt[nxt[j], m]
                actions.append((j, m))
                nxt[j] += 1
    out = {}
    for combo, d in descs.items():
        sim = simulate(jl, pt, d, actions)
        out[combo] = (records_from_simulation(jl, pt, d, actions, sim), sim)
    return jl, pt, mch_type, descs, out


def records_from_simulation(jl, pt, desc, actions, sim):
    """Records in the roll-out's format from a reference_sim schedule."""
    from constraint_family.descriptor import IDLE_CLASS
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    nxt = first.copy()
    last = np.full(pt.shape[1], IDLE_CLASS, dtype=int)
    rec = []
    for j, m in actions:
        op = int(nxt[j])
        s = float(desc.setup[last[m], desc.op_class[op]]) if desc.has_setup else 0.0
        st, ct = float(sim['op_st'][op]), float(sim['op_ct'][op])
        lag = float(desc.lag[op]) if desc.has_lag else 0.0
        rec.append({'step': len(rec), 'job': int(j), 'op': op,
                    'op_in_job': op - int(first[j]), 'machine': int(m),
                    'route_class': int(desc.op_class[op]),
                    'block_start': st - s, 'setup': s, 'proc_start': st,
                    'proc_time': float(pt[op, m]), 'completion': ct,
                    'lag': lag, 'successor_ready': ct + lag})
        last[m] = desc.op_class[op]
        nxt[j] += 1
    return rec


def windows_of(desc, M):
    if desc.has_window and desc.windows is not None:
        return [[(float(s), float(e)) for s, e in w] for w in desc.windows]
    return [[] for _ in range(M)]


PANEL_TITLE = {'N': '(a) No rule active',
               'LSW': '(b) LAG, SETUP and WINDOW active'}


def dry(out_dir=None):
    out_dir = out_dir or tempfile.mkdtemp(prefix='p123_dry_')
    os.makedirs(out_dir, exist_ok=True)
    jl, pt, mch_type, descs, cases = synthetic_case()
    panels, doc = [], {'synthetic': True, 'combinations': {}}
    for combo in COMBOS:
        rec, sim = cases[combo]
        outages = windows_of(descs[combo], len(mch_type))
        check_schedule(rec, outages, jl, pt, descs[combo])
        ms = float(sim['makespan'])
        compare_stored(combo, ms, ms)
        panels.append((PANEL_TITLE[combo], {'operations': rec,
                                            'outages': outages,
                                            'makespan': ms}))
        doc['combinations'][combo] = {'makespan': ms, 'operations': rec,
                                      'windows': outages}
    doc['figure'] = draw(panels, mch_type, os.path.join(out_dir, 'dry_schedule'))
    with open(os.path.join(out_dir, 'dry_schedule.json'), 'w') as f:
        json.dump(doc, f, indent=1)
    print(f'p123 --dry: synthetic schedules checked and drawn in {out_dir}')
    return out_dir


# ----------------------------------------------------------------------------

def run():
    miss = missing_inputs()
    if miss:
        stop('inputs missing, nothing computed:\n  ' + '\n  '.join(miss))
    EF = import_eval_family()
    import torch
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.public_adapter import load_base_instance
    from ppvc_instance_generator import ROUTING_CLASSES

    picks = {}
    for combo in COMBOS:
        dp = descriptor_paths(combo)[INDEX]
        with open(dp) as f:
            raw = json.load(f)
        desc = ConstraintDescriptor.from_json(raw, setup_convention='attached')
        if tuple(desc.active) != ACTIVE[combo]:
            stop(f'{dp}: active rules {desc.active}, want {ACTIVE[combo]}')
        picks[combo] = {'path': dp,
                        'name': os.path.basename(dp).replace('.desc.json', ''),
                        'base_instance': raw.get('base_instance'), 'desc': desc}
    names = {p['name'] for p in picks.values()}
    bases = {p['base_instance'] for p in picks.values()}
    if len(names) != 1 or len(bases) != 1:
        stop(f'the descriptors name different instances: {names}, {bases}')
    name = names.pop()
    stem = os.path.join(EF.ARG_BASE, name)       # as eval_family.py builds it
    base = bases.pop()
    if base is not None and os.path.normpath(base) != os.path.normpath(stem):
        stop(f'descriptor base {base} != {stem}')
    print(f'instance {name}: both descriptors name {stem}')

    jl, pt, meta = load_base_instance(stem)
    mch_type = np.asarray(meta['mch_type'])
    rc = np.asarray(meta['routing_class'])
    for combo in COMBOS:
        if not np.array_equal(np.repeat(rc, jl), picks[combo]['desc'].op_class):
            stop(f'{combo}: descriptor op_class differs from the routes')

    ppo, snap = EF.load_model()
    masks = EF.mask_options(snap)
    fm, mt = masks['first_mask'][0], masks['mask_ties'][0]
    if (fm, mt) != ('exact', 'exact'):
        stop(f'mask options {masks}, want exact and exact')
    if torch.get_num_threads() != 1:
        stop(f'{torch.get_num_threads()} torch threads, want 1')
    print('mask: ' + ', '.join(f'{k}={v} ({s})' for k, (v, s) in masks.items())
          + f'; mask_mode={EF.configs.mask_mode}, '
          f'pair_extra={EF.configs.pair_extra}; torch threads 1; device cpu')

    result, panels = {}, []
    for combo in COMBOS:
        desc = picks[combo]['desc']
        rec, ms, outages = rollout(EF, ppo, desc, jl, pt, meta, fm, mt)
        check_schedule(rec, outages, jl, pt, desc)
        stored = np.load(STORED.format(model=MODEL, combo=combo))
        stored_ms = float(stored[INDEX, 0])
        compare_stored(combo, ms, stored_ms)
        print(f'[{combo:>3}] makespan {ms:g} h, stored {stored_ms:g} h: equal; '
              f'{len(rec)} operations, schedule checks passed')
        result[combo] = {
            'descriptor': picks[combo]['path'],
            'active_rules': dict(zip(('lag', 'setup', 'window'),
                                     map(bool, desc.active))),
            'makespan': ms, 'stored_makespan': stored_ms,
            'equal_to_stored': True,
            'checks': {'start_plus_processing_equals_completion': True,
                       'block_after_job_and_station_ready': True,
                       'p74_occupancy_precedence_lag_windows': True,
                       'validator_on_recorded_times': True,
                       'validator_as_eval_family_calls_it': True,
                       'recorded_completions_equal_environment': True,
                       'largest_completion_equals_makespan': True},
            'operations': rec, 'windows': outages}
        panels.append((PANEL_TITLE[combo], {'operations': rec,
                                            'outages': outages,
                                            'makespan': ms}))

    doc = {'model': MODEL,
           'evaluation': {'method': 'greedy, batch of one',
                          'device': 'cpu', 'torch_threads': 1,
                          'first_mask': fm, 'mask_ties': mt,
                          'mask_mode': EF.configs.mask_mode,
                          'pair_extra': EF.configs.pair_extra,
                          'eval_family_arguments': EVAL_ARGV},
           'instance': stem, 'instance_index': INDEX, 'time_unit': 'h',
           'stored_results': STORED.format(model=MODEL, combo='<combination>'),
           'route_names': list(ROUTING_CLASSES),
           'module_route': rc.tolist(), 'station_type': mch_type.tolist(),
           'combinations': result, 'figure': None}
    # the record is complete once every schedule check has passed; the figure
    # entry is added when the drawing has passed its layout checks
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f'-> {OUT_JSON}')
    fig = draw(panels, mch_type, OUT_FIG)
    doc['figure'] = dict(fig, file=OUT_FIG + '.pdf')
    with open(OUT_JSON, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f'-> {OUT_FIG}.pdf, {OUT_FIG}.png '
          f'({fig["width_in"]:.3f} x {fig["height_in"]:.3f} in)')
    write_macros(doc)


def macro_lines(doc):
    """The macro file of the two example makespans, from a record of run(). A makespan
    that the record does not mark as equal to the stored value is refused."""
    L = ['% Generated by scripts/p123_example_schedule.py -- do not hand-edit',
         f'% greedy schedules of the selected training run on test instance '
         f'{doc["instance_index"]} of the training size; makespans in hours']
    for combo in COMBOS:
        c = doc['combinations'][combo]
        if c.get('equal_to_stored') is not True or c['makespan'] != c['stored_makespan']:
            stop(f'{combo}: the record does not hold a makespan equal to the stored value')
        L.append(f'\\newcommand{{\\{MACRO_NAMES[combo]}}}{{{c["makespan"]:g}}}'
                 f'  % makespan of the example schedule under {combo}, hours')
    return L


def write_macros(doc, path=OUT_TEX):
    lines = macro_lines(doc)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'-> {path}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--macros', action='store_true',
                    help='write the macro file from the stored record; no rollout, no drawing')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--dry_dir', default=None)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    if args.check:
        miss = missing_inputs()
        if miss:
            print('p123 --check: missing inputs:\n  ' + '\n  '.join(miss))
            return 1
        print('p123 --check: every input present')
        return 0
    if args.macros:
        with open(OUT_JSON) as f:
            write_macros(json.load(f))
        return 0
    if args.dry:
        dry(args.dry_dir)
        return 0
    run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
