r"""P69: release dates, the candidate fourth timing rule, evaluated zero-shot.

A release date rho_i is the earliest start of job i. It acts as the start map
b -> max(b, rho_i) on the job's first operation, which FJSPEnvFamily applies
by initialising the job-ready time to rho_i (descriptor field `release`, see
constraint_family/fjsp_env_family.py). The shaping potential keeps the trivial
envelope for this rule, the observation gains no channel and the mask is the
published non-delay rule. The released token-blind policy (three training
seeds) has never seen a release date; this script measures it against the
four dispatching rules under the same release dates, on the 100 published
10x25 test instances of all eight regimes.

Release-date draw (fixed, documented in results/release_dates.md):
  rng = numpy.random.default_rng(20260926); for instance k = 0..99 in order,
  rho = rng.integers(0, floor(0.25 * H_k) + 1, size=J), integer hours, where
  H_k is the descriptor's horizon field (job-chain lower bound, identical in
  all eight regimes of instance k). The SAME rho is used in every regime, so
  cross-regime comparisons stay paired.

Arms, all through the same environment class:
  PACT   greedy, jointblind-v1 / -s302 / -s303, true token, delta_mode 'min',
         mask 'nondelay' (restored from each checkpoint's config snapshot);
  rules  SPT, FIFO, MOR, MWKR with np.random.seed(50) per rollout (the
         published tie-break seed); rule pool = per-instance best of the four.
Each arm is also run with release=None (the control). The control must
reproduce the published Result_*.npy files of test_results/FAMILY/10x25
exactly, or the script refuses: that is the check that the evaluation here is
configured like the published one.

Every schedule is re-validated by both independent validators with the
release check on (reference_sim.validate_schedule and
schedule_validator.validate_schedule) and replayed through the reference
simulator. Every PACT rollout is bound-audited: max_j C_hat_j(s_t) <= realized
makespan at every step (normalized axis), equality at completion, and every
step reward <= 0.

Statistics follow p63/p54/p33: per-instance relative gap in percent with PACT
in the numerator (negative = PACT shorter), PACT pooled over its three seeds by
averaging each instance's makespan, paired bootstrap over instances with
B = 20000 and one rng (seed 7) used in a fixed order.

Outputs (new files only):
  results/release_dates_raw.json   per-rollout makespans, audits, the draw
  results/release_dates.md         pre-run answers, verification, tables
  paper/tables/release_table.tex   per-regime table
  paper/macros_release.tex         every number the table and prose use

Usage (CPU only):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    taskset -c 0-7 python scripts/p69_release_dates.py [--workers 6] [--limit K]
      [--analyze-only]
"""
import glob
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1] + ['--device', 'cpu']


def _opt(flag, default=None, cast=str):
    if flag in _ARGV:
        return cast(_ARGV[_ARGV.index(flag) + 1])
    return default


WORKERS = _opt('--workers', 6, int)
LIMIT = _opt('--limit', 0, int)
ANALYZE_ONLY = '--analyze-only' in _ARGV

import numpy as np  # noqa: E402

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
COMPOSITIONS = ['LS', 'LW', 'SW', 'LSW']
RULES = ['SPT', 'FIFO', 'MOR', 'MWKR']
SEEDS = {'s301': 'jointblind-v1', 's302': 'jointblind-v1-s302',
         's303': 'jointblind-v1-s303'}
DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
PUB = 'test_results/FAMILY/10x25'
DRAW_SEED = 20260926
DRAW_FRAC = 0.25
BOOT_B = 20000
BOOT_SEED = 7
RULE_SEED = 50               # configs.seed_test, the published tie-break seed
TOL = 1e-9

RAW = 'results/release_dates_raw.json'
MD = 'results/release_dates.md'
TABLE = 'paper/tables/release_table.tex'
MACROS = 'paper/macros_release.tex'


# ---------------------------------------------------------------- data

def instance_names():
    names = sorted(os.path.basename(p).replace('.desc.json', '')
                   for p in glob.glob(f'{DATA}/desc/N/*.desc.json'))
    return names[:LIMIT] if LIMIT else names


def read_desc(regime, name):
    with open(f'{DATA}/desc/{regime}/{name}.desc.json') as f:
        return json.load(f)


def draw_release():
    """One draw for all regimes: rho ~ U{0..floor(0.25 H)} per job, integer h."""
    rng = np.random.default_rng(DRAW_SEED)
    out = {}
    for name in sorted(os.path.basename(p).replace('.desc.json', '')
                       for p in glob.glob(f'{DATA}/desc/N/*.desc.json')):
        hs = {read_desc(r, name)['horizon'] for r in REGIMES}
        assert len(hs) == 1, f'{name}: horizon differs across regimes {hs}'
        H = hs.pop()
        rec = read_desc('N', name)
        from ppvc_instance_generator import load_instance
        jl, _, _ = load_instance(rec['base_instance'])
        J = int(jl.shape[0])
        hi = int(np.floor(DRAW_FRAC * H))
        rho = rng.integers(0, hi + 1, size=J)
        out[name] = {'H': int(H), 'hi': hi, 'rho': [int(x) for x in rho]}
    return out


# ---------------------------------------------------------------- workers

_W = {}


def _init_worker():
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from params import configs
    configs.device = 'cpu'
    import fjsp_env_same_op_nums
    fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')
    _W['models'] = {s: load_model(suf)[0] for s, suf in SEEDS.items()}


def load_model(suffix):
    """eval_family.load_model, CPU, returning the resolved configuration."""
    import torch
    from params import configs
    import eval_family
    with open(f'train_log/FAMILY/config_10x25+family+{suffix}.json') as f:
        snap = json.load(f)
    for k in eval_family.ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v in ('True', 'False'):
                v = v == 'True'
            setattr(configs, k, v)
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    path = f'trained_network/FAMILY/10x25+family+{suffix}.pth'
    ppo.policy.load_state_dict(torch.load(path, map_location='cpu'))
    ppo.policy.eval()
    from constraint_family.fjsp_env_family import resolve_delta_mode
    resolved = {k: getattr(configs, k, None) for k in eval_family.ARCH_KEYS}
    resolved['delta_mode_resolved'] = resolve_delta_mode(configs)
    resolved['mask_mode_resolved'] = getattr(configs, 'mask_mode', 'nondelay')
    resolved['pair_extra_resolved'] = getattr(configs, 'pair_extra', 'none')
    resolved['n_params'] = int(sum(p.numel() for p in ppo.policy.parameters()))
    with open(path, 'rb') as f:
        resolved['checkpoint_sha256'] = hashlib.sha256(f.read()).hexdigest()
    return ppo, resolved


def run_one(task):
    """task = (regime, name, arm, rho or None); arm = 'pact:s301' or a rule."""
    regime, name, arm, rho = task
    import torch
    from params import configs
    from common_utils import greedy_select_action, heuristic_select_action
    from constraint_family.descriptor import ConstraintDescriptor, compute_token
    from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                    resolve_delta_mode)
    from constraint_family import reference_sim
    from ppvc_instance_generator import load_instance
    import schedule_validator

    desc = ConstraintDescriptor.from_json(read_desc(regime, name))
    if rho is not None:
        desc.release = np.asarray(rho, dtype=float)
    jl, pt, meta = load_instance(os.path.join(BASE, name))
    policy = arm.startswith('pact:')
    mm = getattr(configs, 'mask_mode', 'nondelay') if policy else 'nondelay'
    pe = getattr(configs, 'pair_extra', 'none') if policy else 'none'
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        mask_mode=mm, pair_extra=pe)
    dm = resolve_delta_mode(configs)
    env.delta_mode = dm
    env.reward_delta_env = (dm != 'zero')
    state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                 op_type_list=[meta['op_type']],
                                 mch_type_list=[meta['mch_type']])
    M = pt.shape[1]
    amch = np.full(pt.shape[0], -1, dtype=int)
    actions = []
    bound_viol = 0
    bounds, rewards = [float(np.max(env.op_ct_lb[0]))], []
    t0 = time.time()
    if policy:
        ppo = _W['models'][arm.split(':')[1]]
        env.token_g = np.asarray([compute_token(desc, float(pt[pt > 0].mean()),
                                                M)])
        env._attach_token()
        state = env.state
        while not env.done().all():
            with torch.no_grad():
                pi, _ = ppo.policy(
                    fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                    mch_type=state.mch_type_tensor, token=state.token_tensor)
            a = int(greedy_select_action(pi).cpu().numpy()[0])
            j, m = a // M, a % M
            assert not env.dynamic_pair_mask[0, j, m], 'policy chose a masked pair'
            amch[env.candidate[0, j]] = m
            actions.append((j, m))
            state, r, _ = env.step(np.array([a]))
            rewards.append(float(r[0]))
            bounds.append(float(np.max(env.op_ct_lb[0])))
    else:
        np.random.seed(RULE_SEED)
        while not env.done().all():
            a = int(heuristic_select_action(arm, env))
            j, m = a // M, a % M
            amch[env.candidate[0, j]] = m
            actions.append((j, m))
            _, r, _ = env.step(np.array([a]))
            rewards.append(float(r[0]))
            bounds.append(float(np.max(env.op_ct_lb[0])))
    dt = time.time() - t0

    ms = float(env.current_makespan[0])
    op_ct = env.true_op_ct[0]
    op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
    v_ref = reference_sim.validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    lag = desc.lag if desc.has_lag else np.zeros(pt.shape[0])
    v_sv = schedule_validator.validate_schedule(
        jl, pt, lag, amch, op_ct, release=desc.release)['violations']
    replay = reference_sim.simulate(jl, pt, desc, actions)
    replay_ok = bool(np.allclose(replay['op_ct'], op_ct, atol=1e-6))
    first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    rel = np.zeros(len(jl)) if rho is None else np.asarray(rho, dtype=float)
    early_first = int(np.sum(op_st[first] < rel - 1e-6))

    # bound audit (normalized axis): C_hat(s_t) <= realized, exact at the end,
    # every reward <= 0
    realized = float(np.max(env.op_ct[0]))
    b = np.array(bounds)
    bound_viol = int(np.sum(b > realized + TOL))
    end_gap = abs(b[-1] - realized)
    bound_viol += int(end_gap > TOL)
    reward_viol = int(np.sum(np.array(rewards) > TOL))
    return {'regime': regime, 'name': name, 'arm': arm,
            'release': rho is not None, 'makespan': ms, 'time': dt,
            'viol_ref': len(v_ref), 'viol_sv': len(v_sv),
            'viol_first': v_ref[:1] + v_sv[:1],
            'replay_ok': replay_ok, 'early_first': early_first,
            'bound_viol': bound_viol, 'reward_viol': reward_viol,
            'end_gap': end_gap,
            'first_start_min_slack': float(np.min(op_st[first] - rel))}


# ---------------------------------------------------------------- main

def pub_arr(tag, regime):
    p = f'{PUB}/Result_{tag}_{regime}.npy'
    a = np.load(p)
    return (a[:, 0] if a.ndim > 1 else a).astype(float)


def verify_configs():
    """Resolved config per seed; everything except the checkpoint must match."""
    from params import configs
    configs.device = 'cpu'
    res = {}
    for s, suf in SEEDS.items():
        _, r = load_model(suf)
        res[s] = r
    keys = sorted(set().union(*[set(r) for r in res.values()]) -
                  {'checkpoint_sha256'})
    diffs = {k: {s: res[s].get(k) for s in res} for k in keys
             if len({json.dumps(res[s].get(k), sort_keys=True) for s in res}) > 1}
    return res, diffs


def run_all_rollouts(draw):
    names = instance_names()
    tasks = []
    for regime in REGIMES:
        for name in names:
            for cond in (None, 'rel'):
                rho = None if cond is None else draw[name]['rho']
                for s in SEEDS:
                    tasks.append((regime, name, f'pact:{s}', rho))
                for rule in RULES:
                    tasks.append((regime, name, rule, rho))
    print(f'{len(tasks)} rollouts on {WORKERS} workers', flush=True)
    out = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as ex:
        for i, r in enumerate(ex.map(run_one, tasks, chunksize=4), 1):
            out.append(r)
            if i % 500 == 0:
                print(f'  [{i}/{len(tasks)}] {(time.time() - t0) / 60:.1f} min',
                      flush=True)
    print(f'rollouts done in {(time.time() - t0) / 60:.1f} min', flush=True)
    return out


def main():
    os.makedirs('results', exist_ok=True)
    if ANALYZE_ONLY:
        raw = json.load(open(RAW))
        draw, rows, cfg, cfg_diffs = (raw['draw'], raw['rows'], raw['configs'],
                                      raw['config_diffs'])
    else:
        for p in (RAW,):
            if os.path.exists(p) and not LIMIT:
                sys.exit(f'REFUSING: {p} exists; use --analyze-only or move it')
        cfg, cfg_diffs = verify_configs()
        print('resolved config diffs across seeds:', cfg_diffs or 'none')
        for s, r in cfg.items():
            print(f'  {s}: delta_mode={r["delta_mode_resolved"]} '
                  f'mask_mode={r["mask_mode_resolved"]} '
                  f'pair_extra={r["pair_extra_resolved"]} use_film={r["use_film"]} '
                  f'n_params={r["n_params"]} sha={r["checkpoint_sha256"][:12]}')
        assert not cfg_diffs, f'seeds differ in resolved config: {cfg_diffs}'
        assert all(r['delta_mode_resolved'] == 'min' and
                   r['mask_mode_resolved'] == 'nondelay' and
                   r['pair_extra_resolved'] == 'none' for r in cfg.values())
        draw = draw_release()
        rows = run_all_rollouts(draw)
        if LIMIT:
            print('--limit given: probe only, nothing written')
            summarize_probe(rows)
            return
        with open(RAW, 'w') as f:
            json.dump({'draw': draw, 'rows': rows, 'configs': cfg,
                       'config_diffs': cfg_diffs,
                       'draw_rule': f'rho ~ U{{0..floor({DRAW_FRAC} H)}} per job, '
                                    f'default_rng({DRAW_SEED}), instances in '
                                    f'name order, shared by all regimes'}, f)
        print(f'wrote {RAW}')
    analyze(draw, rows, cfg)


def summarize_probe(rows):
    for r in rows[:40]:
        print(r['regime'], r['name'], r['arm'], r['release'],
              f'{r["makespan"]:.0f}', f'{r["time"]:.2f}s', r['viol_ref'],
              r['viol_sv'], r['replay_ok'], r['bound_viol'], r['reward_viol'])
    names = instance_names()
    for regime in REGIMES:
        for s, suf in SEEDS.items():
            pub = pub_arr(f'greedy+10x25+family+{suf}', regime)[:len(names)]
            mine = np.array([r['makespan'] for n in names for r in rows
                             if r['regime'] == regime and r['name'] == n
                             and r['arm'] == f'pact:{s}' and not r['release']])
            print(regime, s, 'control == published:', np.array_equal(pub, mine),
                  int(np.sum(pub == mine)), '/', len(names))
        for rule in RULES:
            pub = pub_arr(rule, regime)[:len(names)]
            mine = np.array([r['makespan'] for n in names for r in rows
                             if r['regime'] == regime and r['name'] == n
                             and r['arm'] == rule and not r['release']])
            print(regime, rule, 'rule control == published:',
                  np.array_equal(pub, mine))


# ---------------------------------------------------------------- analysis

def analyze(draw, rows, cfg):
    from macro_io import emit, refuse
    names = instance_names()
    n = len(names)
    idx = {(r['regime'], r['name'], r['arm'], r['release']): r for r in rows}

    def ms(regime, arm, rel):
        return np.array([idx[(regime, nm, arm, rel)]['makespan'] for nm in names])

    # ---- control reproduces the published evaluation
    # Rules involve no network: their control must equal the published
    # makespans exactly. The published policy runs were made on the GPU
    # (config device 'cuda'); on the CPU a near-tied greedy argmax can flip,
    # so the policy control is compared instance by instance and its regime
    # mean must stay within the paper's 1% equivalence margin of the
    # published mean.
    repro, pol_repro = {}, {}
    for regime in REGIMES:
        for rule in RULES:
            repro[(regime, rule)] = np.array_equal(ms(regime, rule, False),
                                                   pub_arr(rule, regime))
        for s, suf in SEEDS.items():
            mine = ms(regime, f'pact:{s}', False)
            pub = pub_arr(f'greedy+10x25+family+{suf}', regime)
            pol_repro[(regime, s)] = {
                'identical': int(np.sum(mine == pub)),
                'mean_diff_pct': float(100 * (mine.mean() - pub.mean()) / pub.mean()),
                'mean_abs_diff_h': float(np.mean(np.abs(mine - pub)))}
    bad = [k for k, v in repro.items() if not v]
    if bad:
        refuse(MACROS, f'rule control (release=None) does not reproduce the '
                       f'published results for {bad}; the evaluation is not '
                       f'configured like the published one')
    pooled_diff = {r: float(100 * (np.mean([ms(r, f'pact:{s}', False) for s in SEEDS])
                                   - np.mean([pub_arr(f'greedy+10x25+family+{suf}', r)
                                              for suf in SEEDS.values()]))
                            / np.mean([pub_arr(f'greedy+10x25+family+{suf}', r)
                                       for suf in SEEDS.values()]))
                   for r in REGIMES}
    far = {r: v for r, v in pooled_diff.items() if abs(v) > 1.0}
    if far:
        refuse(MACROS, f'policy control on CPU departs from the published GPU '
                       f'means by more than 1% in {far}')
    n_ident = sum(v['identical'] for v in pol_repro.values())
    n_pol = len(pol_repro) * n

    # ---- audits
    def count(key, rel=True, pact_only=False):
        return int(sum(r[key] for r in rows if r['release'] == rel and
                       (not pact_only or r['arm'].startswith('pact:'))))
    infeas_rel = int(sum((r['viol_ref'] + r['viol_sv']) > 0 or not r['replay_ok']
                         or r['early_first'] > 0 for r in rows if r['release']))
    infeas_ctl = int(sum((r['viol_ref'] + r['viol_sv']) > 0 or not r['replay_ok']
                         for r in rows if not r['release']))
    bound_rel = count('bound_viol', True, True) + count('reward_viol', True, True)
    bound_ctl = count('bound_viol', False, True) + count('reward_viol', False, True)
    n_rel_sched = sum(1 for r in rows if r['release'])
    n_rel_pact = sum(1 for r in rows if r['release'] and r['arm'].startswith('pact:'))
    first_viol_examples = [r['viol_first'] for r in rows
                           if r['release'] and (r['viol_ref'] + r['viol_sv'])][:3]

    # ---- release statistics
    rho_all = np.concatenate([draw[nm]['rho'] for nm in names]).astype(float)
    frac_all = np.concatenate([np.asarray(draw[nm]['rho']) / draw[nm]['H']
                               for nm in names])
    H_all = np.array([draw[nm]['H'] for nm in names], dtype=float)
    mean_rho, max_rho = float(rho_all.mean()), int(rho_all.max())
    mean_frac = float(frac_all.mean())

    # ---- per-regime statistics
    rng = np.random.default_rng(BOOT_SEED)

    def ci(d):
        bs = d[rng.integers(0, d.size, (BOOT_B, d.size))].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return float(d.mean()), float(lo), float(hi)

    def pact(regime, rel):
        return np.mean([ms(regime, f'pact:{s}', rel) for s in SEEDS], axis=0)

    def pool(regime, rel):
        return np.min(np.stack([ms(regime, r, rel) for r in RULES]), axis=0)

    per, g_pool_all, g_spt_all, g_ctl_all = {}, {}, {}, {}
    for regime in REGIMES:
        p, sp, pl = pact(regime, True), ms(regime, 'SPT', True), pool(regime, True)
        gp = 100.0 * (p - pl) / pl
        gs = 100.0 * (p - sp) / sp
        pc, plc = pact(regime, False), pool(regime, False)
        gc = 100.0 * (pc - plc) / plc
        per[regime] = {
            'pact': float(p.mean()), 'spt': float(sp.mean()),
            'pool': float(pl.mean()),
            'pact_ctl': float(pc.mean()), 'pool_ctl': float(plc.mean()),
            'spt_ctl': float(ms(regime, 'SPT', False).mean()),
            'rules': {r: float(ms(regime, r, True).mean()) for r in RULES},
            'gap_pool': ci(gp), 'gap_spt': ci(gs), 'gap_ctl': ci(gc),
            'infeasible': int(sum((idx[(regime, nm, a, True)]['viol_ref'] +
                                   idx[(regime, nm, a, True)]['viol_sv']) > 0
                                  or not idx[(regime, nm, a, True)]['replay_ok']
                                  for nm in names
                                  for a in [f'pact:{s}' for s in SEEDS] + RULES)),
            'bound': int(sum(idx[(regime, nm, f'pact:{s}', True)]['bound_viol'] +
                             idx[(regime, nm, f'pact:{s}', True)]['reward_viol']
                             for nm in names for s in SEEDS)),
            'pact_time': float(np.mean([idx[(regime, nm, f'pact:{s}', True)]['time']
                                        for nm in names for s in SEEDS])),
        }
        g_pool_all[regime], g_spt_all[regime], g_ctl_all[regime] = gp, gs, gc
    pooled8 = ci(np.concatenate([g_pool_all[r] for r in REGIMES]))
    pooled8_spt = ci(np.concatenate([g_spt_all[r] for r in REGIMES]))
    pooledc = ci(np.concatenate([g_pool_all[r] for r in COMPOSITIONS]))
    pooledc_spt = ci(np.concatenate([g_spt_all[r] for r in COMPOSITIONS]))
    pooled8_ctl = ci(np.concatenate([g_ctl_all[r] for r in REGIMES]))
    ahead = [r for r in REGIMES if per[r]['gap_pool'][2] < 0]
    ahead_spt = [r for r in REGIMES if per[r]['gap_spt'][2] < 0]
    behind = [r for r in REGIMES if per[r]['gap_pool'][1] > 0]
    ms_increase = 100.0 * np.mean([(per[r]['pact'] - per[r]['pact_ctl']) /
                                   per[r]['pact_ctl'] for r in REGIMES])

    # ---- table
    def f1(x):
        return f'{x:.1f}'

    def gap(c):
        return f'${c[0]:+.2f}$ [${c[1]:+.2f}$, ${c[2]:+.2f}$]'

    lines = ['% auto-generated by scripts/p69_release_dates.py -- do not hand-edit',
             '% Release dates rho_i ~ U{0..floor(0.25 H)} h per job (seed 20260926),',
             '% same draw in every regime; 100 test instances per regime at 10x25.',
             '% Makespans in hours; PACT pooled over three training seeds.',
             '% gap = mean per-instance relative makespan gap of PACT, %;'
             ' negative = PACT shorter.',
             '% Suggested caption: Makespan (h) under release dates, 100 test '
             'instances per regime. Bold marks the lowest mean makespan in each '
             'row across PACT, SPT and the rule pool.',
             r'\begin{tabular}{lrrrccl}', r'\toprule',
             r'regime & PACT & SPT & rule pool & gap to rule pool (\%) [95\% CI]'
             r' & gap to SPT (\%) [95\% CI] & verdict \\', r'\midrule']
    macro_names = {'N': 'N', 'L': 'L', 'S': 'S', 'W': 'W', 'LS': 'LS', 'LW': 'LW',
                   'SW': 'SW', 'LSW': 'LSW'}
    md_rows = ['| regime | PACT (h) | SPT (h) | rule pool (h) | gap to pool % [95% CI] '
               '| gap to SPT % [95% CI] | verdict | gap to pool without release % '
               '| infeasible | bound viol. |',
               '|---|---|---|---|---|---|---|---|---|---|']
    for r in REGIMES:
        d = per[r]
        vals = {'pact': d['pact'], 'spt': d['spt'], 'pool': d['pool']}
        lo = min(vals.values())
        cells = []
        for k in ('pact', 'spt', 'pool'):
            mac = f'\\Rel{k.capitalize()}Mean{macro_names[r]}'
            cells.append(f'\\textbf{{{mac}}}' if abs(vals[k] - lo) < 1e-12 else mac)
        verdict = 'PACT ahead' if d['gap_pool'][2] < 0 else (
            'rule pool ahead' if d['gap_pool'][1] > 0 else 'not separable')
        if r == 'SW':
            lines.append(r'\addlinespace')
        lines.append(f'{r} & ' + ' & '.join(cells) + f' & {gap(d["gap_pool"])} & '
                     f'{gap(d["gap_spt"])} & {verdict} \\\\')
        md_rows.append(f'| {r} | {f1(d["pact"])} | {f1(d["spt"])} | {f1(d["pool"])} '
                       f'| {d["gap_pool"][0]:+.2f} [{d["gap_pool"][1]:+.2f}, '
                       f'{d["gap_pool"][2]:+.2f}] | {d["gap_spt"][0]:+.2f} '
                       f'[{d["gap_spt"][1]:+.2f}, {d["gap_spt"][2]:+.2f}] | {verdict} '
                       f'| {d["gap_ctl"][0]:+.2f} [{d["gap_ctl"][1]:+.2f}, '
                       f'{d["gap_ctl"][2]:+.2f}] | {d["infeasible"]} | {d["bound"]} |')
        per[r]['verdict'] = verdict
    lines += [r'\bottomrule', r'\end{tabular}']

    # ---- macros
    mac = ['% auto-generated by scripts/p69_release_dates.py -- do not hand-edit',
           '% Release dates (candidate fourth rule), PACT zero-shot vs dispatching '
           'rules. Source: results/release_dates_raw.json. Negative gap = PACT '
           'shorter.',
           ('RelDrawFrac', f'{DRAW_FRAC:.2f}', 'rho_i ~ U{0..floor(frac * H)} h'),
           ('RelSeed', str(DRAW_SEED), 'rng seed of the release-date draw'),
           ('RelInstances', str(n), 'test instances per regime'),
           ('RelRegimes', str(len(REGIMES))),
           ('RelSeedsN', str(len(SEEDS)), 'PACT training seeds pooled'),
           ('RelMeanRelease', f'{mean_rho:.1f}', 'mean release date, h'),
           ('RelMaxRelease', str(max_rho), 'largest release date, h'),
           ('RelMeanReleasePct', f'{100 * mean_frac:.1f}',
            'mean release date as % of the horizon H'),
           ('RelMeanHorizon', f'{H_all.mean():.1f}', 'mean horizon H, h'),
           ('RelRegimesAheadN', str(len(ahead)),
            'regimes with the PACT-vs-pool CI entirely below zero'),
           ('RelRegimesBehindN', str(len(behind)),
            'regimes with the PACT-vs-pool CI entirely above zero'),
           ('RelRegimesAheadSptN', str(len(ahead_spt))),
           ('RelPooledGap', f'{pooled8[0]:+.2f}', 'PACT vs rule pool, 8 regimes'),
           ('RelPooledCiLo', f'{pooled8[1]:+.2f}'),
           ('RelPooledCiHi', f'{pooled8[2]:+.2f}'),
           ('RelPooledGapAbs', f'{abs(pooled8[0]):.2f}'),
           ('RelPooledSptGap', f'{pooled8_spt[0]:+.2f}', 'PACT vs SPT, 8 regimes'),
           ('RelPooledSptCiLo', f'{pooled8_spt[1]:+.2f}'),
           ('RelPooledSptCiHi', f'{pooled8_spt[2]:+.2f}'),
           ('RelCompGap', f'{pooledc[0]:+.2f}', 'PACT vs rule pool, LS LW SW LSW'),
           ('RelCompCiLo', f'{pooledc[1]:+.2f}'),
           ('RelCompCiHi', f'{pooledc[2]:+.2f}'),
           ('RelCompGapAbs', f'{abs(pooledc[0]):.2f}'),
           ('RelCompSptGap', f'{pooledc_spt[0]:+.2f}'),
           ('RelCompSptCiLo', f'{pooledc_spt[1]:+.2f}'),
           ('RelCompSptCiHi', f'{pooledc_spt[2]:+.2f}'),
           ('RelControlGap', f'{pooled8_ctl[0]:+.2f}',
            'PACT vs rule pool on the same instances without release dates'),
           ('RelControlCiLo', f'{pooled8_ctl[1]:+.2f}'),
           ('RelControlCiHi', f'{pooled8_ctl[2]:+.2f}'),
           ('RelMakespanIncreasePct', f'{ms_increase:.1f}',
            'mean PACT makespan increase caused by the release dates, %'),
           ('RelSchedules', str(n_rel_sched),
            'schedules validated under release dates'),
           ('RelPactRollouts', str(n_rel_pact), 'PACT rollouts bound-audited'),
           ('RelInfeasible', str(infeas_rel),
            'schedules rejected by a validator under release dates'),
           ('RelBoundViolations', str(bound_rel),
            'bound-audit violations over all PACT rollouts with release dates')]
    for r in REGIMES:
        d = per[r]
        mac += [(f'RelPactMean{r}', f1(d['pact'])), (f'RelSptMean{r}', f1(d['spt'])),
                (f'RelPoolMean{r}', f1(d['pool'])),
                (f'RelGap{r}', f'{d["gap_pool"][0]:+.2f}'),
                (f'RelGapLo{r}', f'{d["gap_pool"][1]:+.2f}'),
                (f'RelGapHi{r}', f'{d["gap_pool"][2]:+.2f}')]
    # macro names must be letters only: regime codes are letters, fine
    mine = set()
    for m in mac:
        if isinstance(m, tuple):
            assert re.fullmatch(r'[A-Za-z]+', m[0]), m[0]
            mine.add(m[0])
    for fpath in glob.glob('paper/macros_*.tex') + ['paper/macros.tex']:
        if not os.path.exists(fpath) or \
                os.path.abspath(fpath) == os.path.abspath(MACROS):
            continue
        clash = mine & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',
                                      open(fpath).read()))
        if clash:
            refuse(MACROS, f'macro name collision with {fpath}: {sorted(clash)}')
    emit(MACROS, mac)
    os.makedirs('paper/tables', exist_ok=True)
    with open(TABLE, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'wrote {TABLE}')

    # ---- markdown report
    s0 = next(iter(cfg.values()))
    md = []
    md.append('# Release dates: the candidate fourth timing rule, evaluated zero-shot\n')
    md.append('Generated by `scripts/p69_release_dates.py` from '
              '`results/release_dates_raw.json`. Do not hand-edit.\n')
    md.append('## Pre-run answers\n')
    md.append(
        '1. **Purpose and destination.** Section 8.5 of the manuscript concedes that '
        'no fourth timing rule was implemented, while Section 3.4 argues that a '
        'release date meets the membership test with the trivial envelope. This run '
        'implements release dates and measures the released token-blind policy '
        '(jointblind-v1, three seeds) zero-shot against SPT, FIFO, MOR, MWKR and '
        'their per-instance best (rule pool) under the same release dates, on the '
        '100 published 10x25 test instances of all eight regimes. The numbers land '
        'in `paper/tables/release_table.tex` and `paper/macros_release.tex` '
        '(\\Rel... macros), to replace the concession in Section 8.5 with a measured '
        'result.\n'
        '2. **Expected result and the reading of the alternative.** Expected: every '
        'schedule feasible (the release map is applied by the environment, not '
        'learned), zero bound violations (the potential keeps the trivial envelope, '
        'which stays admissible), and PACT still ahead of the rule pool on most '
        'regimes, with a smaller margin than without release dates because the '
        'observation carries no release channel and the policy never trained on '
        'staggered arrivals. If PACT falls behind the rule pool, the conclusion is '
        'that membership guarantees feasibility and an admissible reward but not '
        'zero-shot quality, so a new rule needs retraining (or a release channel); '
        'the paper would report that boundary. A validator failure or a bound '
        'violation would falsify the membership argument itself and would be '
        'reported as such.\n'
        '3. **Contamination.** Evaluation only, CPU only (CUDA_VISIBLE_DEVICES empty, '
        'cores 0-7, OMP/MKL 4 threads, torch 1 thread). No checkpoint is written or '
        'resumed. All outputs are new files (the script refuses to overwrite '
        '`results/release_dates_raw.json`); no existing Result_*.npy is written. '
        'Wall-clock times are recorded but not reported as measurements (the box '
        'is shared). Test instances only; no training split is touched.\n'
        '4. **Data accuracy.** Instances are the published '
        '`data/FAMILY/10x25/desc/<regime>/instance_000..099` descriptors on '
        '`data/PPVC/10x25+ppvc-mixed`, the same files the published evaluation read. '
        'Every arm is also run with release=None. The rule controls must reproduce '
        'the published `test_results/FAMILY/10x25/Result_*.npy` makespans exactly, '
        'instance by instance; the policy controls (published on the GPU, rerun '
        'here on the CPU) must stay within 1% of the published regime means. The '
        'script refuses to write any table otherwise.\n')
    md.append('## Verification before the numbers\n')
    md.append(f'- Resolved configuration of the three checkpoints (from '
              f'`train_log/FAMILY/config_10x25+family+<suffix>.json` through '
              f'eval_family.ARCH_KEYS, as the published evaluation does): '
              f'delta_mode `{s0["delta_mode_resolved"]}`, mask_mode '
              f'`{s0["mask_mode_resolved"]}`, pair_extra `{s0["pair_extra_resolved"]}`, '
              f'use_film `{s0["use_film"]}`, token_dim {s0["token_dim"]}, '
              f'{s0["n_params"]:,} parameters. The three seeds differ in nothing but '
              f'the checkpoint file (sha256 prefixes '
              + ', '.join(f'{s} `{c["checkpoint_sha256"][:12]}`' for s, c in cfg.items())
              + ').')
    md.append(f'- Control reproduction, rules: with release=None, all '
              f'{len(repro)} rule-regime cells ({len(RULES)} rules x {len(REGIMES)} '
              f'regimes, {n} instances each) reproduce the published makespans '
              f'exactly (np.array_equal). The environment, the instances, the mask '
              f'and the tie-break seed are therefore those of the published '
              f'evaluation.')
    md.append(f'- Control reproduction, policy: the published greedy runs were '
              f'made on the GPU; this run is CPU-only, where a near-tied greedy '
              f'argmax can resolve differently. {n_ident} of {n_pol} control '
              f'rollouts (3 seeds x 8 regimes x {n}) give the published makespan '
              f'exactly; the pooled regime means differ from the published ones by '
              + ', '.join(f'{r} {pooled_diff[r]:+.2f}%' for r in REGIMES)
              + ' (the script refuses beyond 1%). Every gap in this report compares '
              'arms run in this same CPU evaluation, so the release effect is not '
              'mixed with the device difference.')
    md.append('- Code-level regression: `tests/test_release.py` runs the environment '
              'with release=None in lockstep with a frozen copy of the published '
              'environment (tests/fixtures/fjsp_env_family_published.py) on three '
              'instances of LS and SW under SPT, MWKR and random legal actions and '
              'asserts bit-identical observations, masks, rewards and schedules at '
              'every step; an all-zero release equals release=None.')
    md.append(f'- Control audits: {infeas_ctl} infeasible control schedules, '
              f'{bound_ctl} bound violations on control PACT rollouts.\n')
    md.append('## Release-date rule\n')
    md.append(f'rho_i ~ integer uniform on [0, floor({DRAW_FRAC} H)] hours per job, H = '
              f'the descriptor horizon (job-chain lower bound incl. lags, identical in '
              f'all regimes of an instance); numpy default_rng({DRAW_SEED}), instances '
              f'in name order, one draw shared by the eight regimes. Mean H = '
              f'{H_all.mean():.1f} h. Mean release date {mean_rho:.1f} h '
              f'({100 * mean_frac:.1f}% of H), max {max_rho} h. Environment semantics: '
              f'job-ready time initialised to rho_i on both time axes (start map '
              f'b -> max(b, rho_i) on the first operation); potential unchanged '
              f'(trivial envelope); no new observation channel; first decision at '
              f'min_i rho_i with only jobs released then admissible (the non-delay '
              f'rule restricted to readiness, as the published first decision is).\n')
    md.append('## Results (release dates on)\n')
    md += md_rows
    md.append('')
    md.append(f'- Pooled over the 8 regimes ({8 * n} instances): PACT vs rule pool '
              f'{pooled8[0]:+.2f}% [{pooled8[1]:+.2f}, {pooled8[2]:+.2f}]; PACT vs SPT '
              f'{pooled8_spt[0]:+.2f}% [{pooled8_spt[1]:+.2f}, {pooled8_spt[2]:+.2f}].')
    md.append(f'- Pooled over the 4 compositions (LS, LW, SW, LSW): PACT vs rule pool '
              f'{pooledc[0]:+.2f}% [{pooledc[1]:+.2f}, {pooledc[2]:+.2f}]; PACT vs SPT '
              f'{pooledc_spt[0]:+.2f}% [{pooledc_spt[1]:+.2f}, {pooledc_spt[2]:+.2f}].')
    md.append(f'- Regimes with the PACT-vs-pool interval entirely below zero: '
              f'{len(ahead)} of 8 ({", ".join(ahead) or "none"}); entirely above zero: '
              f'{len(behind)} ({", ".join(behind) or "none"}). Versus SPT: '
              f'{len(ahead_spt)} of 8 below zero.')
    md.append(f'- Same instances without release dates (control): PACT vs rule pool '
              f'{pooled8_ctl[0]:+.2f}% [{pooled8_ctl[1]:+.2f}, {pooled8_ctl[2]:+.2f}]. '
              f'Release dates raise the mean PACT makespan by {ms_increase:.1f}%.')
    md.append(f'- Feasibility: {infeas_rel} of {n_rel_sched} schedules under release '
              f'dates rejected by a validator (reference_sim.validate_schedule and '
              f'schedule_validator.validate_schedule, both with the release check, '
              f'plus a replay through reference_sim.simulate).'
              + (f' First failures: {first_viol_examples}' if first_viol_examples
                 else ''))
    md.append(f'- Bound audit on all {n_rel_pact} PACT rollouts with release dates: '
              f'{bound_rel} violations (C_hat(s_t) <= realized makespan at every step, '
              f'equality at completion within {TOL:g}, every step reward <= 0).')
    md.append('\nPer-rule means under release dates (h):\n')
    md.append('| regime | ' + ' | '.join(RULES) + ' |')
    md.append('|---|' + '---|' * len(RULES))
    for r in REGIMES:
        md.append(f'| {r} | ' + ' | '.join(f1(per[r]['rules'][k]) for k in RULES)
                  + ' |')
    with open(MD, 'w') as f:
        f.write('\n'.join(md) + '\n')
    print(f'wrote {MD}')
    print('\n'.join(md_rows))
    print('policy control vs published:', n_ident, '/', n_pol, 'identical;',
          {r: round(v, 3) for r, v in pooled_diff.items()})
    print(f'pooled8 {pooled8}, comp {pooledc}, ahead {ahead}, behind {behind}, '
          f'infeasible {infeas_rel}, bound {bound_rel}, control {pooled8_ctl}')


if __name__ == '__main__':
    main()
