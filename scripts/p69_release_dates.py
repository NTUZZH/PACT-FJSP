r"""P69: release dates, the candidate fourth timing rule, evaluated zero-shot.

A release date rho_i is a job-side delay: job i is not present before rho_i,
so neither the setup nor the processing of its first operation starts
earlier. FJSPEnvFamily applies it by initialising the job-ready time to rho_i
(descriptor field `release`, see constraint_family/fjsp_env_family.py). The
makespan estimate behind the shaping potential is unchanged: it starts every
job chain at time zero, so it stays a lower bound. The observation gains no
channel and the mask is the published non-delay rule. The released token-blind policy (three training
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

Seeds: the active seeds of scripts/arm_registry.py (three in the submitted
analysis); outputs go to the run's output root (arm_registry.out). In the
legacy environment the rollouts build the environment with the published
(legacy) mask; everything above describes that environment.

EXACT ENVIRONMENT (PACT_ENV=exact; the active seeds, 301 to 305 in the paper)
  Rollouts. Every rollout, the policy's and the dispatching rules', builds the
  environment with the mask options each checkpoint's configuration records
  (first_mask and mask_ties); the five configurations must agree with each other
  and with arm_registry.MASK_OPTIONS of the environment, the options of the stored
  arrays, or the script refuses. The four rules act on the legal pairs of the
  non-delay mask (common_utils.masked_rule_action), as the rules job of the
  evaluation plan of this environment does. Release dates, the tie-break seed and
  the arms are the ones above, at every active seed of the policy
  (arm_registry.seeds()).
  Evaluation and analysis are separate phases:
    plan       units, estimated core-hours and the commands of K parts;
    evaluate   the rollouts of the units of one part (--part i --parts K, unit j goes
               to part j mod K), one thread, CPU; a unit is one combination and one test
               instance, the 18 rollouts of the policy seeds and the four rules with
               and without release dates, written to <root>/10x25/<R>/<instance>.json
               under a temporary name and renamed; an existing unit file of the same
               design (environment, models, checkpoint hashes, mask options, rules,
               seeds of the draw and of the rules) is skipped, one of another design
               refuses;
    check      read only: every unit exists and is of the design, and the rollouts
               without release dates reproduce the stored arrays of the environment
               (arm_registry.res_dir('10x25'): the policy at every seed and the four
               rules) exactly, instance by instance; exit 0 only then;
    analyze    the check, then the statistics; any failure removes the outputs.
  Statistics, the protocol of the paper (scripts/p88_family_inference.py): per
  combination d = 100 (policy - rule pool) / rule pool per training run and test
  instance, the rule pool being the per-instance best of the four rules under the same
  condition; the crossed hierarchical bootstrap with the training run as the unit
  (stats_family, B = --boot, fresh default_rng(11) per quantity); the combined verdict
  ahead / behind / not distinguishable of p88.superiority_block (one-sided seed-level t
  and hierarchical bootstrap p-values, Holm over the eight combinations, each direction
  its own family); pooled rows over the eight combinations and over the four
  compositions LS, LW, SW, LSW (hierarchical, base-instance clusters, interval and
  direction only). The same comparison without release dates beside it, with its own
  Holm families. Counts of validated schedules, of schedules a validator rejects and of
  makespan-estimate violations (policy and rules), per condition. Intervals print with
  two decimals (p88 f2, sign kept); verdicts and directions use the unrounded values.
  Outputs (arm_registry.out): manuscript/macros_release.tex (names Rel...),
  manuscript/tables/release_table.tex, results/release_dates.json,
  results/release_dates.md. --n N limits the instances (smoke runs; the outputs are
  marked as a development run).

Usage (CPU only):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    taskset -c 0-7 python scripts/p69_release_dates.py [--workers 6] [--limit K]
      [--analyze-only]
  PACT_ENV=exact PACT_SEEDS=301-305 python scripts/p69_release_dates.py plan [--parts K]
  PACT_ENV=exact PACT_SEEDS=301-305 taskset -c <core> python scripts/p69_release_dates.py \
      evaluate [--part i --parts K] [--root DIR] [--n N] [--combos N,L]
  PACT_ENV=exact PACT_SEEDS=301-305 python scripts/p69_release_dates.py check [--root DIR]
  PACT_ENV=exact PACT_SEEDS=301-305 PACT_OUT=<folder> python scripts/p69_release_dates.py \
      analyze [--root DIR] [--boot B]
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

import arm_registry as REG  # noqa: E402  (pops --pact-* options from sys.argv)
EXACT = REG.env() == 'exact'
REG.banner()
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
# the released policy at the active seeds: key s<seed> -> run name
SEEDS = {f's{s}': REG.model('released', None, s) for s in REG.seeds()}
KWORD = {3: 'three', 5: 'five'}.get(len(SEEDS), str(len(SEEDS)))
DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
PUB = REG.res_dir('10x25')
DRAW_SEED = 20260926
DRAW_FRAC = 0.25
BOOT_B = 20000
BOOT_SEED = 7
RULE_SEED = 50               # configs.seed_test, the published tie-break seed
TOL = 1e-9

if not EXACT:
    RAW = REG.out('results/release_dates_raw.json')
    MD = REG.out('results/release_dates.md')
    TABLE = REG.out('paper/tables/release_table.tex')
    MACROS = REG.out('paper/macros_release.tex')
# mask options of every rollout: none in the legacy environment (the published
# mask); in the exact environment the options of the checkpoints' configurations
MASK_KW = {}
# the dispatching rules: common_utils.heuristic_select_action in the legacy
# environment; in the exact environment the rules act on the legal pairs of the
# non-delay mask (common_utils.masked_rule_action), as the rules job of the
# evaluation plan of that environment does (eval_family.py --rule_set masked)
RULE_SET = 'masked' if EXACT else 'classical'


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
    if EXACT:
        for k in MASK_NAMES:
            resolved[k] = snap.get(k)
    return ppo, resolved


def run_one(task):
    """task = (regime, name, arm, rho or None); arm = 'pact:s301' or a rule."""
    regime, name, arm, rho = task
    import torch
    from params import configs
    from common_utils import (greedy_select_action, heuristic_select_action,
                              masked_rule_action)
    from constraint_family.descriptor import ConstraintDescriptor, compute_token
    from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                    resolve_delta_mode)
    from constraint_family import reference_sim
    from ppvc_instance_generator import load_instance
    import schedule_validator

    desc = ConstraintDescriptor.from_json(read_desc(regime, name), setup_convention='attached')
    if rho is not None:
        desc.release = np.asarray(rho, dtype=float)
    jl, pt, meta = load_instance(os.path.join(BASE, name))
    policy = arm.startswith('pact:')
    mm = getattr(configs, 'mask_mode', 'nondelay') if policy else 'nondelay'
    pe = getattr(configs, 'pair_extra', 'none') if policy else 'none'
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        mask_mode=mm, pair_extra=pe, **MASK_KW)
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
        rule_action = (masked_rule_action if RULE_SET == 'masked'
                       else heuristic_select_action)
        while not env.done().all():
            a = int(rule_action(arm, env))
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
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
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
             f'% Makespans in hours; PACT pooled over {KWORD} training seeds.',
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
                os.path.basename(fpath) == os.path.basename(MACROS):
            continue
        clash = mine & set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}',
                                      open(fpath).read()))
        if clash:
            refuse(MACROS, f'macro name collision with {fpath}: {sorted(clash)}')
    emit(MACROS, mac)
    os.makedirs(os.path.dirname(TABLE), exist_ok=True)
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
        'release date, a job-side delay that leaves the makespan estimate unchanged, '
        'meets the membership test. This run '
        'implements release dates and measures the released token-blind policy '
        f'(jointblind-v1, {KWORD} seeds) zero-shot against SPT, FIFO, MOR, MWKR and '
        'their per-instance best (rule pool) under the same release dates, on the '
        '100 published 10x25 test instances of all eight regimes. The numbers land '
        'in `paper/tables/release_table.tex` and `paper/macros_release.tex` '
        '(\\Rel... macros), to replace the concession in Section 8.5 with a measured '
        'result.\n'
        '2. **Expected result and the reading of the alternative.** Expected: every '
        'schedule feasible (the release dates are applied by the environment, not '
        'learned), zero bound violations (the estimate starts every job chain at '
        'time zero, so it stays a lower bound), and PACT still ahead of the rule pool on most '
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
    md.append(f'- Resolved configuration of the {KWORD} checkpoints (from '
              f'`train_log/FAMILY/config_10x25+family+<suffix>.json` through '
              f'eval_family.ARCH_KEYS, as the published evaluation does): '
              f'delta_mode `{s0["delta_mode_resolved"]}`, mask_mode '
              f'`{s0["mask_mode_resolved"]}`, pair_extra `{s0["pair_extra_resolved"]}`, '
              f'use_film `{s0["use_film"]}`, token_dim {s0["token_dim"]}, '
              f'{s0["n_params"]:,} parameters. The {KWORD} seeds differ in nothing but '
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
              f'rollouts ({len(SEEDS)} seeds x 8 regimes x {n}) give the published makespan '
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
              f'a job-side delay, the job-ready time initialised to rho_i on both time '
              f'axes, so neither the setup nor the processing of the first operation '
              f'starts earlier; potential unchanged (the estimate starts every job '
              f'chain at time zero and stays a lower bound); no new observation '
              f'channel; first decision at '
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


# ================================================================ exact environment
MASK_NAMES = ('first_mask', 'mask_ties')
EXACT_ROOT = 'test_results/FAMILY/release_p69'
EXACT_SIZE = '10x25'
EXACT_OUT = (('macros', 'manuscript/macros_release.tex'),
             ('table', 'manuscript/tables/release_table.tex'),
             ('json', 'results/release_dates.json'),
             ('md', 'results/release_dates.md'))
# Seconds of CPU time per rollout at 10x25 with one thread, validation included (planning
# estimates of the plan phase): 6.8 and 8.2 s per unit of 18 rollouts on N and LSW,
# measured on one core of a machine that other work shared.
COST_POLICY_S = 0.60
COST_RULE_S = 0.25
CONDITIONS = ((False, 'without release dates'), (True, 'with release dates'))
FAMILIES = (('Pooled', REGIMES, 'the eight combinations'),
            ('Comp', COMPOSITIONS, 'the four compositions LS, LW, SW, LSW'))


class Refused(Exception):
    """An input of the exact-environment phases is missing or does not fit."""


def environment_mask_options():
    """{first_mask, mask_ties} of arm_registry.MASK_OPTIONS of the active environment: the
    options the stored arrays of the environment were evaluated with."""
    o = list(REG.MASK_OPTIONS[REG.env()])
    opts = {o[i].lstrip('-'): o[i + 1] for i in range(0, len(o), 2)}
    if set(opts) != set(MASK_NAMES):
        raise Refused(f'arm_registry.MASK_OPTIONS of {REG.env()} is {o}, want both of '
                      f'{MASK_NAMES}')
    return opts


def light_design():
    """The design every unit file must carry, from the configurations and the checkpoint
    files (no network is loaded): environment, size, models, checkpoint hashes, mask
    options, rules, the seeds of the rules and of the draw."""
    want = environment_mask_options()
    sha, cfg_opts = {}, {}
    for s, m in SEEDS.items():
        cp = f'train_log/FAMILY/config_{REG.PREFIX}{m}.json'
        ck = f'trained_network/FAMILY/{REG.PREFIX}{m}.pth'
        for p in (cp, ck):
            if not os.path.exists(p):
                raise Refused(f'missing {p}')
        snap = json.load(open(cp))
        cfg_opts[s] = {k: snap.get(k) for k in MASK_NAMES}
        with open(ck, 'rb') as f:
            sha[s] = hashlib.sha256(f.read()).hexdigest()
    bad = {s: v for s, v in cfg_opts.items() if v != want}
    if bad:
        raise Refused(f'the configurations of {sorted(bad)} record the mask options '
                      f'{next(iter(bad.values()))}, the stored arrays of the {REG.env()} '
                      f'environment were evaluated with {want}')
    return {'environment': REG.env(), 'size': EXACT_SIZE, 'models': dict(SEEDS),
            'checkpoint_sha256': sha, 'mask_options': want, 'rules': list(RULES),
            'rule_set': RULE_SET, 'rule_seed': RULE_SEED, 'draw_seed': DRAW_SEED,
            'draw_frac': DRAW_FRAC}


def exact_names(n=None):
    names = sorted(os.path.basename(p).replace('.desc.json', '')
                   for p in glob.glob(f'{DATA}/desc/N/*.desc.json'))
    return names[:n] if n else names


def exact_units(combos, names):
    """(combination, instance) in combination-major order."""
    return [(r, nm) for r in combos for nm in names]


def part_units(units, part, parts):
    """Unit j belongs to part j mod parts."""
    if parts < 1 or part is None and parts != 1 or part is not None and not 0 <= part < parts:
        raise Refused(f'--part {part} --parts {parts}: want 0 <= part < parts')
    return units if part is None else [u for j, u in enumerate(units) if j % parts == part]


def unit_path(root, combo, name):
    return os.path.join(root, EXACT_SIZE, combo, f'{name}.json')


def expected_arms():
    return [f'pact:{s}' for s in SEEDS] + list(RULES)


def read_unit(root, combo, name, design):
    """The record of one unit, or a Refused naming why it does not count."""
    p = unit_path(root, combo, name)
    if not os.path.exists(p):
        raise Refused(f'missing {p}')
    rec = json.load(open(p))
    if rec.get('design') != design:
        diff = sorted(k for k in set(design) | set(rec.get('design') or {})
                      if (rec.get('design') or {}).get(k) != design.get(k))
        raise Refused(f'{p} is of another design (fields {", ".join(diff)})')
    have = sorted((r['arm'], r['release']) for r in rec.get('rows', []))
    want = sorted((a, rel) for a in expected_arms() for rel in (False, True))
    if rec.get('combination') != combo or rec.get('instance') != name or have != want:
        raise Refused(f'{p} does not hold the rollouts of {combo}/{name}')
    return rec


def write_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f'{path}.tmp{os.getpid()}'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def exact_plan(a):
    units = exact_units(a.combos, exact_names(a.n))
    per = 2 * (len(SEEDS) * COST_POLICY_S + len(RULES) * COST_RULE_S)
    try:
        design = light_design()
        done = sum(1 for c, nm in units if os.path.exists(unit_path(a.root, c, nm)))
    except Refused as e:
        design, done = None, 0
        print(f'[p69 plan] the design cannot be read yet: {e}')
    total = len(units) * per
    sizes = [len(part_units(units, i, a.parts)) for i in range(a.parts)]
    print(f'[p69 plan] {REG.describe()}; {len(units)} units (combination x test instance), '
          f'{done} done; each {2 * (len(SEEDS) + len(RULES))} rollouts ({len(SEEDS)} policy '
          f'seeds and {len(RULES)} rules, with and without release dates)')
    print(f'[p69 plan] estimated {total / 3600:.2f} core-hours of CPU time '
          f'({COST_POLICY_S:g} s per policy rollout, {COST_RULE_S:g} s per rule rollout, one '
          f'thread); {a.parts} parts of {min(sizes) * per / 3600:.2f} to '
          f'{max(sizes) * per / 3600:.2f} h')
    if design:
        print(f'[p69 plan] mask options {design["mask_options"]}; root {a.root}')
    seeds = f'{min(REG.seeds())}-{max(REG.seeds())}'
    for i in range(a.parts):
        print(f'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PACT_ENV={REG.env()} '
              f'PACT_SEEDS={seeds} taskset -c <CORE_{i}> {sys.executable} -B '
              f'scripts/p69_release_dates.py evaluate --part {i} --parts {a.parts}'
              + ('' if a.root == EXACT_ROOT else f' --root {a.root}'))
    return dict(units=len(units), done=done, core_hours=total / 3600)


def exact_evaluate(a):
    global MASK_KW
    design = light_design()
    cfg, diffs = verify_configs()
    if diffs:
        raise Refused(f'the policy seeds differ in their resolved configuration: {diffs}')
    for s, r in cfg.items():
        if (r['delta_mode_resolved'], r['mask_mode_resolved'], r['pair_extra_resolved']) != \
                ('min', 'nondelay', 'none') or r['checkpoint_sha256'] != \
                design['checkpoint_sha256'][s]:
            raise Refused(f'{s}: delta_mode {r["delta_mode_resolved"]}, mask_mode '
                          f'{r["mask_mode_resolved"]}, pair_extra {r["pair_extra_resolved"]}')
    MASK_KW = dict(design['mask_options'])
    units = part_units(exact_units(a.combos, exact_names(a.n)), a.part, a.parts)
    draw = draw_release()
    _init_worker()
    print(f'[p69 evaluate] {len(units)} units, mask options {MASK_KW}, root {a.root}',
          flush=True)
    t_all = time.time()
    counts = {'done': 0, 'exists': 0}
    for k, (combo, name) in enumerate(units, 1):
        p = unit_path(a.root, combo, name)
        if os.path.exists(p):
            read_unit(a.root, combo, name, design)          # refuses a foreign file
            counts['exists'] += 1
            continue
        c0 = time.process_time()
        rows = []
        for rel, _ in CONDITIONS:
            rho = draw[name]['rho'] if rel else None
            for arm in expected_arms():
                rows.append(run_one((combo, name, arm, rho)))
        write_atomic(p, {'design': design, 'combination': combo, 'instance': name,
                         'rho': draw[name]['rho'], 'horizon': draw[name]['H'], 'rows': rows,
                         'cpu_s': time.process_time() - c0,
                         'written': time.strftime('%Y-%m-%d %H:%M:%S')})
        counts['done'] += 1
        print(f'[p69 evaluate {k}/{len(units)}] {combo:4s} {name} cpu '
              f'{time.process_time() - c0:6.1f}s elapsed {time.time() - t_all:7.0f}s',
              flush=True)
    print(f'[p69 evaluate] {counts["done"]} units written, {counts["exists"]} existed')
    return counts


def stored(tag, combo, n):
    a = REG.read_path(f'{REG.res_dir(EXACT_SIZE)}/Result_{tag}_{combo}.npy')
    if n and len(a) < n or not n and len(a) != len(exact_names()):
        raise Refused(f'{REG.res_dir(EXACT_SIZE)}/Result_{tag}_{combo}.npy holds {len(a)} rows')
    return a[:len(exact_names(n))]


def collect(root, combos, n):
    """Every unit of the design and the control reproduction. Returns (records by unit,
    reproduction summary); raises Refused on a missing or foreign unit or on any rollout
    without release dates that differs from the stored array."""
    design = light_design()
    names = exact_names(n)
    recs, miss = {}, []
    for combo, name in exact_units(combos, names):
        try:
            recs[(combo, name)] = read_unit(root, combo, name, design)
        except Refused as e:
            miss.append(str(e))
    if miss:
        raise Refused(f'{len(miss)} of {len(combos) * len(names)} units do not count; first: '
                      f'{miss[0]}' + (f'; last: {miss[-1]}' if len(miss) > 1 else ''))
    bad, compared = [], 0
    for combo in combos:
        for arm in expected_arms():
            tag = f'greedy+{REG.PREFIX}{SEEDS[arm[5:]]}' if arm.startswith('pact:') else arm
            ref = stored(tag, combo, n)
            mine = np.array([next(r['makespan'] for r in recs[(combo, nm)]['rows']
                                  if r['arm'] == arm and not r['release']) for nm in names])
            compared += len(names)
            if not np.array_equal(mine, ref):
                i = int(np.flatnonzero(mine != ref)[0])
                bad.append(f'{combo} {arm} {names[i]}: {mine[i]:g} against the stored {ref[i]:g}')
    if bad:
        raise Refused(f'the rollouts without release dates differ from the stored arrays of '
                      f'{REG.res_dir(EXACT_SIZE)} on {len(bad)} (combination, arm) pairs; '
                      f'first: {bad[0]}')
    return design, recs, dict(compared=compared, identical=compared, folder=REG.res_dir(EXACT_SIZE))


def exact_check(a):
    try:
        _, recs, rep = collect(a.root, a.combos, a.n)
    except (Refused, REG.MissingResult) as e:
        print(f'p69 check: not ready: {e}')
        return 1
    print(f'p69 check: {len(recs)} units of the design present under {a.root}; the '
          f'{rep["compared"]} rollouts without release dates reproduce {rep["folder"]} exactly')
    return 0


def audit_counts(rows):
    """Schedules, validator rejections and estimate violations of a list of rows."""
    def rejected(r):
        return ((r['viol_ref'] + r['viol_sv']) > 0 or not r['replay_ok']
                or r['early_first'] > 0)
    pol = [r for r in rows if r['arm'].startswith('pact:')]
    rul = [r for r in rows if not r['arm'].startswith('pact:')]
    return dict(schedules=len(rows), rejected=int(sum(rejected(r) for r in rows)),
                policy_rollouts=len(pol), rule_rollouts=len(rul),
                policy_estimate=int(sum(r['bound_viol'] + r['reward_viol'] for r in pol)),
                rule_estimate=int(sum(r['bound_viol'] + r['reward_viol'] for r in rul)),
                first_rejections=[r['viol_first'] for r in rows if rejected(r)][:3])


def exact_statistics(recs, combos, names, boot):
    import p88_family_inference as F88
    import stats_family as SF
    F88.configure(boot=boot, alpha=0.05)
    F88.verify_bases(EXACT_SIZE, len(names))

    def ms(combo, arm, rel):
        return np.array([next(r['makespan'] for r in recs[(combo, nm)]['rows']
                              if r['arm'] == arm and r['release'] == rel) for nm in names])

    def pool(combo, rel):
        return np.min(np.stack([ms(combo, q, rel) for q in RULES]), axis=0)

    out = {}
    for rel, what in CONDITIONS:
        D = {c: np.stack([SF.rel_gap(ms(c, f'pact:{s}', rel), pool(c, rel)) for s in SEEDS])
             for c in combos}
        sup = F88.superiority_block(D)
        pooled = {}
        for fam, regs, text in FAMILIES:
            regs = [r for r in regs if r in combos]
            if len(regs) < 2:
                continue
            hb = SF.hier_boot(np.stack([D[r] for r in regs], axis=2), B=F88.B,
                              seed=F88.FRESH_SEED, alpha=F88.ALPHA)
            pooled[fam] = dict(mean=hb['mean'], lo=hb['lo'], hi=hb['hi'], combinations=regs,
                               direction=SF.direction(hb['lo'], hb['hi']), text=text)
        means = {c: dict(policy=float(np.mean([ms(c, f'pact:{s}', rel) for s in SEEDS])),
                         pool=float(pool(c, rel).mean()),
                         rules={q: float(ms(c, q, rel).mean()) for q in RULES}) for c in combos}
        rows = [r for c in combos for nm in names for r in recs[(c, nm)]['rows']
                if r['release'] == rel]
        out['release' if rel else 'control'] = dict(
            what=what, rows=sup, pooled=pooled, means=means, audit=audit_counts(rows),
            seed_means={c: sup[c]['seed_means'] for c in combos})
    inc = 100.0 * np.mean([(out['release']['means'][c]['policy'] -
                            out['control']['means'][c]['policy']) /
                           out['control']['means'][c]['policy'] for c in combos])
    return out, dict(B=F88.B, alpha=F88.ALPHA, fresh_seed=F88.FRESH_SEED,
                     verdict_words=list(F88.VERDICT_WORDS)), float(inc)


def exact_macros(res, draw, names, combos, settings, dev):
    import p88_family_inference as F88
    f2 = F88.f2
    R, C = res['release'], res['control']
    words = settings['verdict_words']

    def count(blk, w):
        return sum(1 for x in blk['rows'].values() if x['verdict'] == w)
    rho = np.concatenate([draw[nm]['rho'] for nm in names]).astype(float)
    frac = np.concatenate([np.asarray(draw[nm]['rho']) / draw[nm]['H'] for nm in names])
    H = np.array([draw[nm]['H'] for nm in names], dtype=float)
    head = [f'% Generated by scripts/p69_release_dates.py ({REG.describe()}) -- do not hand-edit',
            '% Release dates as a fourth rule type, the policy against the rule pool (per-instance '
            'best of SPT, FIFO, MOR, MWKR) under the same release dates, and without them.',
            f'% Differences: policy minus rule pool over rule pool, %, per training run and test '
            f'instance; hierarchical 95% interval (training runs x base instances, B = '
            f'{settings["B"]}); combined verdict with Holm over the {len(combos)} combinations '
            f'(p88_family_inference.superiority_block). Bounds print with two decimals; '
            f'verdicts and directions use the unrounded values.']
    if dev:
        head.append(f'% DEVELOPMENT RUN on {len(names)} instances per combination')
    L = [('RelDrawFrac', f'{DRAW_FRAC:.2f}', 'rho_i ~ U{0..floor(frac * H)} h'),
         ('RelSeed', str(DRAW_SEED), 'seed of the release-date draw'),
         ('RelInstances', str(len(names)), 'test instances per combination'),
         ('RelRegimes', str(len(combos)), 'combinations'),
         ('RelSeedsN', str(len(SEEDS)), 'training runs of the policy'),
         ('RelBoot', str(settings['B']), 'bootstrap replicates'),
         ('RelMeanRelease', f'{rho.mean():.1f}', 'mean release date, h'),
         ('RelMaxRelease', str(int(rho.max())), 'largest release date, h'),
         ('RelMeanReleasePct', f'{100 * frac.mean():.1f}', 'mean release date, % of H'),
         ('RelMeanHorizon', f'{H.mean():.1f}', 'mean horizon H, h'),
         ('RelRegimesAheadN', str(count(R, words[0])),
          'combinations with the combined verdict ahead, release dates'),
         ('RelRegimesBehindN', str(count(R, words[1])),
          'combinations with the combined verdict behind, release dates'),
         ('RelNotDistinguishableN', str(count(R, words[2])),
          'combinations not distinguishable, release dates'),
         ('RelControlAheadN', str(count(C, words[0])), 'ahead, without release dates'),
         ('RelControlBehindN', str(count(C, words[1])), 'behind, without release dates'),
         ('RelControlNotDistinguishableN', str(count(C, words[2])),
          'not distinguishable, without release dates')]
    for key, blk in (('Pooled', R), ('Control', C)):
        for fam, _, text in FAMILIES:
            x = blk['pooled'].get(fam)
            if x is None:
                continue
            nm = {('Pooled', 'Pooled'): 'RelPooled', ('Pooled', 'Comp'): 'RelComp',
                  ('Control', 'Pooled'): 'RelControl', ('Control', 'Comp'): 'RelControlComp'}[
                      (key, fam)]
            L += [(f'{nm}Gap', f2(x['mean']), f'{blk["what"]}, pooled over {text}'),
                  (f'{nm}CiLo', f2(x['lo']), f'interval end {x["lo"]:+.4f}'),
                  (f'{nm}CiHi', f2(x['hi']), f'interval end {x["hi"]:+.4f}'),
                  (f'{nm}GapAbs', f'{abs(x["mean"]):.2f}', 'absolute value'),
                  (f'{nm}Dir', x['direction'], 'direction of the unrounded interval')]
    L += [('RelMakespanIncreasePct', f'{res["increase"]:.1f}',
           'mean policy makespan increase caused by the release dates, %')]
    for key, blk in (('', R), ('Control', C)):
        a = blk['audit']
        L += [(f'Rel{key}Schedules', str(a['schedules']), f'schedules validated, {blk["what"]}'),
              (f'Rel{key}PactRollouts', str(a['policy_rollouts']), 'policy rollouts audited'),
              (f'Rel{key}RuleRollouts', str(a['rule_rollouts']), 'rule rollouts audited'),
              (f'Rel{key}Infeasible', str(a['rejected']), 'schedules a validator rejects'),
              (f'Rel{key}BoundViolations', str(a['policy_estimate']),
               'makespan-estimate violations over the policy rollouts'),
              (f'Rel{key}RuleBoundViolations', str(a['rule_estimate']),
               'makespan-estimate violations over the rule rollouts')]
    for c in combos:
        m, x, y = R['means'][c], R['rows'][c], C['rows'][c]
        L += [(f'RelPactMean{c}', f'{m["policy"]:.1f}', 'policy, mean over runs and instances, h'),
              (f'RelSptMean{c}', f'{m["rules"]["SPT"]:.1f}', 'SPT, h'),
              (f'RelPoolMean{c}', f'{m["pool"]:.1f}', 'rule pool, h'),
              (f'RelGap{c}', f2(x['mean']), 'policy minus rule pool, release dates, %'),
              (f'RelGapLo{c}', f2(x['lo']), f'interval end {x["lo"]:+.4f}'),
              (f'RelGapHi{c}', f2(x['hi']), f'interval end {x["hi"]:+.4f}'),
              (f'RelVerdict{c}', x['verdict'],
               f'Holm p faster {x["p_t_holm"]:.2g} (t), {x["p_boot_holm"]:.2g} (bootstrap); '
               f'Holm p slower {x["p_t_greater_holm"]:.2g} (t), '
               f'{x["p_boot_greater_holm"]:.2g} (bootstrap)'),
              (f'RelControlGap{c}', f2(y['mean']), 'policy minus rule pool, without release dates'),
              (f'RelControlGapLo{c}', f2(y['lo']), f'interval end {y["lo"]:+.4f}'),
              (f'RelControlGapHi{c}', f2(y['hi']), f'interval end {y["hi"]:+.4f}'),
              (f'RelControlVerdict{c}', y['verdict'],
               f'Holm p faster {y["p_t_holm"]:.2g} (t), {y["p_boot_holm"]:.2g} (bootstrap); '
               f'Holm p slower {y["p_t_greater_holm"]:.2g} (t), '
               f'{y["p_boot_greater_holm"]:.2g} (bootstrap)')]
    return head, L


def exact_table(res, combos, names, settings, dev):
    import p88_family_inference as F88
    f2 = F88.f2
    R, C = res['release'], res['control']
    lines = ['% Generated by scripts/p69_release_dates.py -- do not hand-edit',
             f'% {REG.describe()}; {len(names)} test instances per combination at 10x25; '
             f'makespans in hours, policy averaged over its training runs.',
             '% Differences: policy minus rule pool over rule pool, %, hierarchical 95% interval; '
             'verdict: combined verdict, Holm over the combinations. Bold: the lowest mean '
             'makespan of the row across the policy, SPT and the rule pool.']
    if dev:
        lines.append(f'% DEVELOPMENT RUN on {len(names)} instances per combination')
    lines += [r'\begin{tabular}{lrrrclcl}', r'\toprule',
              r'combination & policy & SPT & rule pool & \multicolumn{2}{c}{with release dates}'
              r' & \multicolumn{2}{c}{without release dates} \\',
              r'\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
              r' & & & & difference (\%) [95\% CI] & verdict & difference (\%) [95\% CI] '
              r'& verdict \\', r'\midrule']
    for c in combos:
        m = R['means'][c]
        vals = {'Pact': m['policy'], 'Spt': m['rules']['SPT'], 'Pool': m['pool']}
        lo = f'{min(vals.values()):.1f}'           # the printed values are compared
        cells = [f'\\textbf{{\\Rel{k}Mean{c}}}' if f'{v:.1f}' == lo else
                 f'\\Rel{k}Mean{c}' for k, v in vals.items()]
        x, y = R['rows'][c], C['rows'][c]
        if c == 'SW':
            lines.append(r'\addlinespace')
        lines.append(f'{c} & ' + ' & '.join(cells) +
                     f' & ${f2(x["mean"])}$ [${f2(x["lo"])}$, ${f2(x["hi"])}$] & {x["verdict"]}'
                     f' & ${f2(y["mean"])}$ [${f2(y["lo"])}$, ${f2(y["hi"])}$] & {y["verdict"]} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return lines


def exact_markdown(res, design, rep, combos, names, settings, draw, dev):
    import p88_family_inference as F88
    f2 = F88.f2
    md = ['# Release dates as a fourth rule type, exact environment', '',
          f'Generated by `scripts/p69_release_dates.py analyze` ({REG.describe()}). '
          + (f'DEVELOPMENT RUN on {len(names)} instances per combination. ' if dev else ''),
          '',
          f'Rollouts: the policy at {len(SEEDS)} training runs '
          f'({", ".join(design["models"].values())}) and {", ".join(RULES)} (tie-break seed '
          f'{RULE_SEED}), mask options {design["mask_options"]} for every rollout, with and '
          f'without release dates, on {len(names)} test instances of each of '
          f'{len(combos)} combinations at 10x25.',
          f'Release dates: rho_i ~ integer uniform on [0, floor({DRAW_FRAC} H)] h per job, '
          f'default_rng({DRAW_SEED}), instances in name order, one draw shared by the '
          f'combinations.',
          f'Control reproduction: the {rep["compared"]} rollouts without release dates equal '
          f'the stored arrays of `{rep["folder"]}` instance by instance (policy at every '
          f'training run and the four rules).', '',
          f'Statistics: d = 100 (policy - rule pool) / rule pool per training run and test '
          f'instance; hierarchical bootstrap over training runs and base instances, B = '
          f'{settings["B"]}, fresh default_rng({settings["fresh_seed"]}) per quantity; combined '
          f'verdict of p88 (ahead: interval below zero and both Holm-adjusted one-sided p for '
          f'"faster" below {settings["alpha"]}; behind mirrored, its own Holm family; not '
          f'distinguishable otherwise), Holm over the {len(combos)} combinations per '
          f'condition.', '']
    for key in ('release', 'control'):
        blk = res[key]
        md += [f'## Policy against the rule pool, {blk["what"]}', '',
               '| combination | policy (h) | rule pool (h) | difference % [95% CI] | Holm p faster '
               '(t / boot) | Holm p slower (t / boot) | verdict |',
               '|---|---|---|---|---|---|---|']
        for c in combos:
            x, m = blk['rows'][c], blk['means'][c]
            md.append(f'| {c} | {m["policy"]:.1f} | {m["pool"]:.1f} | {f2(x["mean"])} '
                      f'[{f2(x["lo"])}, {f2(x["hi"])}] | {x["p_t_holm"]:.2g} / '
                      f'{x["p_boot_holm"]:.2g} | {x["p_t_greater_holm"]:.2g} / '
                      f'{x["p_boot_greater_holm"]:.2g} | {x["verdict"]} |')
        md.append('')
        for fam, x in blk['pooled'].items():
            md.append(f'- Pooled over {x["text"]}: {f2(x["mean"])} [{f2(x["lo"])}, '
                      f'{f2(x["hi"])}], direction {x["direction"]}.')
        a = blk['audit']
        md += [f'- Schedules validated: {a["schedules"]} ({a["policy_rollouts"]} policy, '
               f'{a["rule_rollouts"]} rule rollouts); rejected by a validator '
               f'(reference_sim.validate_schedule and schedule_validator.validate_schedule with '
               f'the release check, replay through reference_sim.simulate, first starts not '
               f'before the release date): {a["rejected"]}'
               + (f', first {a["first_rejections"]}' if a['first_rejections'] else '') + '.',
               f'- Makespan-estimate audit (estimate at most the realized makespan at every '
               f'step, equal at completion within {TOL:g}, every reward at most 0): '
               f'{a["policy_estimate"]} violations over the policy rollouts, '
               f'{a["rule_estimate"]} over the rule rollouts.', '']
    md.append(f'Release dates raise the mean policy makespan by {res["increase"]:.1f}% '
              f'(mean over the combinations).')
    rho = np.concatenate([draw[nm]['rho'] for nm in names]).astype(float)
    md.append(f'Mean release date {rho.mean():.1f} h, largest {int(rho.max())} h.')
    md += ['', 'Mean makespan of each rule with release dates (h):', '',
           '| combination | ' + ' | '.join(RULES) + ' |', '|---|' + '---|' * len(RULES)]
    for c in combos:
        md.append(f'| {c} | ' + ' | '.join(f'{res["release"]["means"][c]["rules"][q]:.1f}'
                                          for q in RULES) + ' |')
    return md


def defined_macros(own):
    """{name: file} of every macro defined in the macro files of manuscript/ and its
    folders and in manuscript/sections_new/*.tex, the files named like the output left
    out."""
    rx = re.compile(r'\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)')
    out = {}
    files = set(glob.glob('manuscript/macros*.tex') + glob.glob('manuscript/*/macros*.tex')
                + glob.glob('manuscript/sections_new/*.tex'))
    for f in sorted(files):
        if os.path.basename(f) == os.path.basename(own):
            continue
        text = '\n'.join(re.sub(r'(?<!\\)%.*', '', x)
                         for x in open(f, errors='replace').read().splitlines())
        for m in rx.findall(text):
            out.setdefault(m, f)
    return out


def exact_outputs():
    return {k: REG.out(p) for k, p in EXACT_OUT}


def exact_analyze(a):
    from macro_io import emit
    if os.environ.get('PACT_OUT') is None:
        raise Refused('set PACT_OUT (the output root of the chain, e.g. out/exact)')
    outs = exact_outputs()
    for p in outs.values():                        # a refusal leaves no stale output
        if os.path.exists(p):
            os.remove(p)
    try:
        design, recs, rep = collect(a.root, a.combos, a.n)
        names = exact_names(a.n)
        draw = draw_release()
        res, settings, inc = exact_statistics(recs, a.combos, names, a.boot)
        res['increase'] = inc
        dev = bool(a.n) or list(a.combos) != list(REGIMES)
        head, L = exact_macros(res, draw, names, a.combos, settings, dev)
        names_out = [n for n, _, _ in L]
        bad = [n for n in names_out if not re.fullmatch(r'Rel[A-Za-z]+', n)]
        if bad or len(set(names_out)) != len(names_out):
            raise Refused(f'macro names must be Rel plus letters and unique: {bad[:5]}')
        have = defined_macros(outs['macros'])
        clash = sorted(n for n in names_out if n in have)
        if clash:
            raise Refused('macro names already defined: '
                          + ', '.join(f'\\{n} ({have[n]})' for n in clash[:5]))
        emit(outs['macros'], head + [(n, v, c) for n, v, c in L])
        os.makedirs(os.path.dirname(outs['table']), exist_ok=True)
        with open(outs['table'], 'w') as f:
            f.write('\n'.join(exact_table(res, a.combos, names, settings, dev)) + '\n')
        md = exact_markdown(res, design, rep, a.combos, names, settings, draw, dev)
        with open(outs['md'], 'w') as f:
            f.write('\n'.join(md) + '\n')
        import p88_family_inference as F88
        with open(outs['json'], 'w') as f:
            json.dump(F88.jsonable(dict(
                generator='scripts/p69_release_dates.py analyze', setting=REG.describe(),
                development=dev, design=design, root=a.root, combinations=list(a.combos),
                instances=names, reproduction=rep, statistics=settings, results=res,
                macros={n: v for n, v, _ in L})), f, indent=1)
    except BaseException:
        for p in outs.values():
            if os.path.exists(p):
                os.remove(p)
        raise
    print('\n'.join(md))
    print('wrote ' + ', '.join(outs.values()))
    return 0


def exact_main(argv):
    import argparse
    ap = argparse.ArgumentParser(description='release dates in the exact environment')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('plan', 'evaluate', 'check', 'analyze'):
        p = sub.add_parser(name)
        p.add_argument('--root', default=EXACT_ROOT, help='folder of the unit files')
        p.add_argument('--n', type=int, default=None, help='first N instances (smoke runs)')
        p.add_argument('--combos', default=','.join(REGIMES))
        if name in ('plan', 'evaluate'):
            p.add_argument('--parts', type=int, default=1)
        if name == 'evaluate':
            p.add_argument('--part', type=int, default=None)
        if name == 'analyze':
            p.add_argument('--boot', type=int, default=BOOT_B)
    a = ap.parse_args(argv)
    a.combos = [c for c in a.combos.split(',') if c]
    if [c for c in a.combos if c not in REGIMES] or len(set(a.combos)) != len(a.combos):
        ap.error(f'--combos: want distinct combinations of {",".join(REGIMES)}')
    try:
        if a.cmd == 'plan':
            exact_plan(a)
            return 0
        if a.cmd == 'evaluate':
            exact_evaluate(a)
            return 0
        if a.cmd == 'check':
            return exact_check(a)
        return exact_analyze(a)
    except (Refused, REG.MissingResult) as e:
        print(f'*** REFUSING ***\n{e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    if EXACT:
        sys.exit(exact_main(_ARGV))
    main()
