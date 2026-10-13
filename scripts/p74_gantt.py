"""Gantt chart of the released policy on one benchmark instance, two regimes.

Rolls out the released policy (10x25+family+jointblind-v1, greedy, batch of
one) on the same 10-module base instance under two descriptors: regime N (no
rule active) and regime LSW (time lags, sequence-dependent setups and
unavailability windows all active). Each regime uses the first descriptor of
data/FAMILY/10x25/desc/<regime>/ in sorted order; both must name the same base
instance, which is asserted.

The environment keeps completion times only (true_op_ct), so the schedule is
recorded during the rollout. At every decision the script reads the chosen
(job, operation, machine), the changeover the machine will pay (its last
routing class into the operation's class, read before the step updates it) and
the environment's own true-axis processing start (_earliest_start, a pure
function of the pre-step state). After the step it reads the completion the
environment stored and checks start + p == completion. The recorded schedule
is then re-validated three ways: an in-script check of machine occupancy
(changeover and processing are one contiguous block), job precedence with the
time lag, and outage avoidance; the independent validator in
constraint_family/reference_sim.py, called as eval_family.py calls it; and the
largest recorded completion against env.current_makespan.

The rollout mirrors eval_family.py (same architecture keys restored from the
config snapshot, same delta envelope on the observation, true token), but runs
on the CPU with one torch thread. Each regime's makespan must equal the value
stored for the same instance index in
test_results/FAMILY/10x25/Result_greedy+10x25+family+jointblind-v1_<regime>.npy.
The JSON is written in either case and records the comparison; the figure is
written only when both makespans match.

Outputs:
  paper/figures/f8_gantt.pdf, paper/figures/f8_gantt.png (300 dpi), 7.0 in wide
  results/gantt_example.json   recorded schedules, outages and makespans

Usage (from the repository root):
  taskset -c 9 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/p74_gantt.py
"""
import glob
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1]              # params.py parses argv on import
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np

MODEL = '10x25+family+jointblind-v1'
DESC_ROOT = 'data/FAMILY/10x25/desc'
BASE_ROOT = 'data/PPVC/10x25+ppvc-mixed'
STORED = 'test_results/FAMILY/10x25/Result_greedy+{model}_{regime}.npy'
REGIMES = ('N', 'LSW')
OUT_FIG = 'paper/figures/f8_gantt'
OUT_JSON = 'results/gantt_example.json'

# Same key set eval_family.py restores from the config snapshot, so the policy
# sees the observation convention it was trained with.
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode',
             'mask_mode', 'pair_extra')

TOL = 1e-6


# ----------------------------------------------------------------------------
# model and rollout
# ----------------------------------------------------------------------------

def load_policy():
    from params import configs
    with open(f'./train_log/FAMILY/config_{MODEL}.json') as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v in ('True', 'False'):
                v = v == 'True'
            setattr(configs, k, v)
    # must precede the environment import: EnvState binds its device at class
    # definition time
    configs.device = 'cpu'

    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/FAMILY/{MODEL}.pth',
        map_location=torch.device('cpu')))
    ppo.policy.eval()
    return ppo, configs


def first_descriptor(regime):
    paths = sorted(glob.glob(os.path.join(DESC_ROOT, regime, '*.desc.json')))
    assert paths, f'no descriptors under {DESC_ROOT}/{regime}'
    return paths


def rollout(ppo, configs, desc, jl, pt, meta):
    """Greedy rollout; returns the recorded per-operation schedule."""
    import torch
    from common_utils import greedy_select_action
    from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                    resolve_delta_mode)

    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        mask_mode=getattr(configs, 'mask_mode', 'nondelay'),
                        pair_extra=getattr(configs, 'pair_extra', 'none'))
    dm = resolve_delta_mode(configs)
    env.delta_mode = dm
    env.reward_delta_env = (dm != 'zero')
    state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                 op_type_list=[meta['op_type']],
                                 mch_type_list=[meta['mch_type']])

    M = pt.shape[1]
    first_op = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
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
        a = int(act[0])
        job, mch = a // M, a % M
        op = int(env.candidate[0, job])

        # pre-step reads: changeover from the machine's last class, the job's
        # ready time, the machine's free time, and the environment's own
        # processing start on the true (hour) axis
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
        assert abs(ct - p - st_pred) < TOL, (op, ct, p, st_pred)
        block = st_pred - setup
        assert block >= max(job_ready, mch_free) - TOL, (op, block)
        rec.append({'step': len(rec), 'job': job, 'op': op,
                    'op_in_job': op - int(first_op[job]), 'machine': mch,
                    'route_class': int(env.op_class[0, op]),
                    'block_start': block, 'setup': setup,
                    'proc_start': st_pred, 'proc_time': p, 'completion': ct,
                    'lag': lag, 'successor_ready': ct + lag})

    assert len(rec) == pt.shape[0], (len(rec), pt.shape[0])
    ms = float(env.current_makespan[0])
    assert abs(max(r['completion'] for r in rec) - ms) < TOL, ms

    outages = []
    if env.n_win:
        for m in range(M):
            spans = [(float(s), float(e)) for s, e in
                     zip(env.true_win_s[0, m], env.true_win_e[0, m])
                     if np.isfinite(s)]
            outages.append(spans)
    else:
        outages = [[] for _ in range(M)]
    return rec, ms, outages


def check_schedule(rec, outages, jl, pt, desc):
    """Feasibility of the recorded schedule, in-script and by the validator."""
    from constraint_family.reference_sim import validate_schedule
    by_op = {r['op']: r for r in rec}
    # machine occupancy: changeover plus processing is one contiguous block
    M = pt.shape[1]
    for m in range(M):
        blocks = sorted((r['block_start'], r['completion'])
                        for r in rec if r['machine'] == m)
        for (s0, c0), (s1, _c1) in zip(blocks, blocks[1:]):
            assert s1 >= c0 - TOL, f'machine {m} overlap {c0} > {s1}'
        # no block may overlap an unavailability window
        for s, c in blocks:
            for ws, we in outages[m]:
                assert not (s < we - TOL and c > ws + TOL), \
                    f'machine {m} block [{s},{c}] meets outage [{ws},{we}]'
    # job precedence with the time lag
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    for j, f0 in enumerate(first):
        for o in range(f0, f0 + int(jl[j]) - 1):
            assert by_op[o + 1]['block_start'] >= \
                by_op[o]['successor_ready'] - TOL, f'job {j} op {o}'
    N = pt.shape[0]
    op_st = np.array([by_op[o]['proc_start'] for o in range(N)])
    op_ct = np.array([by_op[o]['completion'] for o in range(N)])
    amch = np.array([by_op[o]['machine'] for o in range(N)])
    viol = validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    assert not viol, viol[:5]


# ----------------------------------------------------------------------------
# figure
# ----------------------------------------------------------------------------

# Plain station names for the generator's station types (A..Q).
STATION_LABEL = {'A': 'formwork', 'B': 'concrete pour', 'C': 'MEP',
                 'D': 'finishing', 'E': 'painting', 'F': 'assembly',
                 'G': 'dispatch', 'H': 'steel fabrication',
                 'Q': 'quality check'}

# Pastel fills by route class (house palette: cyan, blue-grey, rose,
# lavender; no red and green pair). Steel-dry keeps the medium violet stroke
# of its hue, the other classes the dark border.
ROUTE_FILL = ['#9fd3e3', '#c6d0dc', '#eaa9b2', '#d9d1ea']
EDGE = '#2b2b2b'
ROUTE_EDGE = [EDGE, EDGE, EDGE, '#7a63a8']
LAG_LW = 1.1
OUTAGE = '#d4d4d4'
LAG_LINE = '#3f7aa8'


def station_labels(mch_type):
    from ppvc_instance_generator import STATION_TYPES
    counts = {t: int(np.sum(mch_type == t)) for t in set(mch_type.tolist())}
    seen, out = {}, []
    for t in mch_type.tolist():
        seen[t] = seen.get(t, 0) + 1
        name = STATION_LABEL[STATION_TYPES[t]]
        out.append(name if counts[t] == 1 else f'{name} {seen[t]}')
    return out


def setup_fonts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    ttfs = glob.glob(os.path.join(fdir, 'TeXGyreTermes-*.ttf'))
    assert ttfs, f'TeX Gyre Termes TrueType files missing under {fdir}'
    for f in ttfs:
        fm.fontManager.addfont(f)
    serif = 'TeX Gyre Termes'
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': [serif],
        'mathtext.fontset': 'custom', 'mathtext.rm': serif,
        'mathtext.it': f'{serif}:italic', 'mathtext.bf': f'{serif}:bold',
        'font.size': 8, 'pdf.fonttype': 42, 'hatch.linewidth': 0.45,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6,
        'ytick.major.width': 0.6, 'xtick.major.size': 3,
        'ytick.major.size': 0, 'axes.edgecolor': 'black',
        'text.color': 'black', 'xtick.color': 'black',
        'ytick.color': 'black', 'axes.labelcolor': 'black',
    })
    return plt


def draw(result, labels, mch_type, class_names):
    plt = setup_fonts()
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    M = len(labels)
    xmax = 20.0 * np.ceil(1.03 * max(result[r]['makespan']
                                     for r in REGIMES) / 20.0)

    # geometry in inches; axes placed explicitly so the figure is exactly
    # 7.0 in wide at print size
    W, LEFT, RIGHT = 7.0, 1.05, 0.12
    TOP_LEG, PANEL, TICKS, GAP, BOTTOM = 0.50, 2.85, 0.22, 0.22, 0.45
    H = TOP_LEG + 2 * PANEL + TICKS + GAP + BOTTOM
    fig = plt.figure(figsize=(W, H))
    axw = W - LEFT - RIGHT
    y_b = BOTTOM
    y_a = BOTTOM + PANEL + TICKS + GAP
    axes = [fig.add_axes([LEFT / W, y_a / H, axw / W, PANEL / H]),
            fig.add_axes([LEFT / W, y_b / H, axw / W, PANEL / H])]

    BAR_LO, BAR_HI = -0.34, 0.22     # bar extent about the row centre
    LAG_Y = 0.34                     # lag line below the bar (y is inverted)
    OUT_HALF = 0.46                  # outage band half-height

    group_edges = [i for i in range(1, M) if mch_type[i] != mch_type[i - 1]]
    for ax, regime, tag in zip(axes, REGIMES, ('(a)', '(b)')):
        res = result[regime]
        for m, spans in enumerate(res['outages']):
            for ws, we in spans:
                if ws >= xmax:
                    continue
                ax.add_patch(Rectangle((ws, m - OUT_HALF), min(we, xmax) - ws,
                                       2 * OUT_HALF, facecolor=OUTAGE,
                                       edgecolor='none', zorder=1))
        for r in res['operations']:
            m, fill = r['machine'], ROUTE_FILL[r['route_class']]
            edge = ROUTE_EDGE[r['route_class']]
            if r['setup'] > 0:
                ax.add_patch(Rectangle(
                    (r['block_start'], m + BAR_LO), r['setup'],
                    BAR_HI - BAR_LO, facecolor=fill, edgecolor=edge,
                    lw=0.35, hatch='//////', zorder=3))
            ax.add_patch(Rectangle(
                (r['proc_start'], m + BAR_LO), r['proc_time'],
                BAR_HI - BAR_LO, facecolor=fill, edgecolor=edge, lw=0.35,
                zorder=3))
            if r['lag'] > 0:
                ax.plot([r['completion'], r['successor_ready']],
                        [m + LAG_Y] * 2, color=LAG_LINE, lw=LAG_LW,
                        solid_capstyle='butt', zorder=4)
                ax.plot([r['successor_ready']] * 2,
                        [m + LAG_Y - 0.09, m + LAG_Y + 0.09],
                        color=LAG_LINE, lw=LAG_LW, solid_capstyle='butt',
                        zorder=4)
        for g in group_edges:
            ax.axhline(g - 0.5, color='#bfbfbf', lw=0.4, zorder=0)
        ax.axvline(res['makespan'], color='black', lw=0.9,
                   ls=(0, (4, 2)), zorder=5)

        ax.set_xlim(0, xmax)
        ax.set_ylim(M - 0.5, -0.5)
        ax.set_yticks(range(M))
        ax.set_yticklabels(labels, fontsize=7)
        ax.xaxis.set_major_locator(MultipleLocator(20))
        ax.tick_params(axis='x', labelsize=7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.text(-LEFT / axw + 0.004, 1.0, tag, transform=ax.transAxes,
                ha='left', va='bottom', fontsize=8, fontweight='bold')
    axes[1].set_xlabel('Time (h)', fontsize=8)

    handles = [Patch(facecolor=ROUTE_FILL[c], edgecolor=ROUTE_EDGE[c], lw=0.35,
                     label=f'{class_names[c]} module')
               for c in range(len(class_names))]
    handles += [
        Patch(facecolor='white', edgecolor=EDGE, lw=0.35, hatch='//////',
              label='changeover'),
        Patch(facecolor=OUTAGE, edgecolor='none',
              label='unavailability window'),
        Line2D([], [], color=LAG_LINE, lw=LAG_LW, label='time lag'),
        Line2D([], [], color='black', lw=0.9, ls=(0, (4, 2)),
               label='makespan'),
    ]
    leg = fig.legend(handles=handles, loc='upper center', ncol=4,
                     bbox_to_anchor=(0.5 + (LEFT - RIGHT) / (2 * W),
                                     1.0 - 0.04 / H),
                     frameon=True, fancybox=False, fontsize=7,
                     handlelength=1.8, columnspacing=1.6,
                     borderpad=0.4, labelspacing=0.35)
    leg.get_frame().set_edgecolor('black')
    leg.get_frame().set_linewidth(0.5)

    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG + '.pdf')
    fig.savefig(OUT_FIG + '.png', dpi=300)
    plt.close(fig)


def check_fonts(pdf):
    """Only TeX Gyre Termes may be embedded (no DejaVu or other leak)."""
    if not shutil.which('pdffonts'):
        print('pdffonts not found: font family not verified')
        return
    out = subprocess.run(['pdffonts', pdf], capture_output=True,
                         text=True, check=True).stdout.splitlines()[2:]
    names = [ln.split()[0] for ln in out if ln.strip()]
    bad = [n for n in names if 'TeXGyreTermes' not in n]
    assert names and not bad, f'non-Termes fonts in {pdf}: {bad}'
    print(f'fonts in {pdf}: {sorted(set(names))}')


# ----------------------------------------------------------------------------

def main():
    from constraint_family.descriptor import ConstraintDescriptor
    from ppvc_instance_generator import ROUTING_CLASSES, load_instance

    # the same base instance under both regimes
    picks = {}
    for regime in REGIMES:
        paths = first_descriptor(regime)
        dp = paths[0]
        name = os.path.basename(dp).replace('.desc.json', '')
        with open(dp) as f:
            raw = json.load(f)
        picks[regime] = {'path': dp, 'name': name, 'index': 0,
                         'base_instance': raw.get('base_instance'),
                         'desc': ConstraintDescriptor.from_json(raw)}
    names = {p['name'] for p in picks.values()}
    bases = {p['base_instance'] for p in picks.values()}
    assert len(names) == 1 and len(bases) == 1, (names, bases)
    name = names.pop()
    stem = os.path.join(BASE_ROOT, name)
    base = bases.pop()
    assert base is None or os.path.normpath(base) == os.path.normpath(stem), \
        (base, stem)
    print(f'instance {name}: both descriptors point at {stem}')

    jl, pt, meta = load_instance(stem)
    mch_type = np.asarray(meta['mch_type'])
    rc = np.asarray(meta['routing_class'])
    assert np.array_equal(np.repeat(rc, jl), picks['LSW']['desc'].op_class), \
        'descriptor op_class differs from the instance routing classes'

    ppo, configs = load_policy()
    result = {}
    for regime in REGIMES:
        desc = picks[regime]['desc']
        rec, ms, outages = rollout(ppo, configs, desc, jl, pt, meta)
        check_schedule(rec, outages, jl, pt, desc)
        stored = np.load(STORED.format(model=MODEL, regime=regime))
        stored_ms = float(stored[picks[regime]['index'], 0])
        result[regime] = {
            'descriptor': picks[regime]['path'],
            'active_rules': {'lag': bool(desc.has_lag),
                             'setup': bool(desc.has_setup),
                             'window': bool(desc.has_window)},
            'makespan': ms, 'stored_makespan': stored_ms,
            'matches_stored': abs(ms - stored_ms) < TOL,
            'operations': rec, 'outages': outages}
        print(f'[{regime:>3}] makespan {ms:.1f} h, stored {stored_ms:.1f} h, '
              f'{"match" if result[regime]["matches_stored"] else "MISMATCH"}')

    labels = station_labels(mch_type)
    doc = {'model': MODEL, 'method': 'greedy, batch of one, CPU',
           'instance': stem, 'time_unit': 'h',
           'stored_results': STORED.format(model=MODEL, regime='<regime>'),
           'instance_index': 0,
           'routing_class_names': ROUTING_CLASSES,
           'job_routing_class': rc.tolist(),
           'machine_type': mch_type.tolist(), 'machine_label': labels,
           'regimes': result}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f'-> {OUT_JSON}')

    bad = [r for r in REGIMES if not result[r]['matches_stored']]
    assert not bad, (f'makespan differs from the stored greedy result for '
                     f'{bad}; figure not written (see {OUT_JSON})')

    draw(result, labels, mch_type, ROUTING_CLASSES)
    check_fonts(OUT_FIG + '.pdf')
    print(f'-> {OUT_FIG}.pdf, {OUT_FIG}.png')


if __name__ == '__main__':
    main()
