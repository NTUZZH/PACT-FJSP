"""P40: three probes around E3 (token-intervention causality).

The main paper's E3 shows that feeding the policy a FALSIFIED constraint token,
while the ENVIRONMENT stays correctly configured, degrades makespan on exactly
one regime after Holm correction: W, +4.18 h (q=1.9e-06), on a single-seed probe
(model joint-v1, seed 301). Three questions E3 leaves open; this script answers all
three from ONE fresh rollout campaign on this machine, written under
test_results/FAMILY/10x25_misconfig/ so it never mixes with the published probe's
numbers (greedy rollouts are deterministic within a machine but not across
machines/environments, so every arm compared here is produced by the same code
path on the same box).

  (i)   REPLICATION ACROSS SEEDS. Rerun the full E3 token-intervention family
        (3 interventions x 5 regimes = 15 one-sided paired Wilcoxon tests, Holm
        across the whole family, exactly as scripts/p17_e3_multiplicity.py) for
        each training seed 301/302/303, and read off the W conflicting-token cell.
        Does the W effect replicate on all three seeds?

  (ii)  ENVIRONMENT MISCONFIGURATION vs a wrong token. A wrong token (E3) leaves
        the shop correct and only biases the schedule the policy builds; a wrong
        ENVIRONMENT makes the policy build its schedule under the wrong rules.
        Four cases on the 100-instance test set, two true/false regime pairs:
          A env=true , token=true      (control)
          B env=true , token=false     (= E3: wrong description, correct shop)
          C env=false, token=false     (fully misconfigured software stack)
          D env=false, token=true      (inconsistent setup)
        Every resulting schedule is validated against the TRUE descriptor by the
        independently implemented reference simulator
        (constraint_family.reference_sim.validate_schedule). We report mean
        makespan, feasible/infeasible counts under the truth, and the violation
        types the validator catches (setup-block overlap, window overlap, lag).

  (iii) FIRST-DISPATCH EXCEPTION. Supplement Sec. S-II.C: at the first decision
        instant (t=0) the legality mask degenerates to static compatibility, so
        the first dispatched operation is the one place a chosen op can begin
        processing AFTER t=0 (a window in progress at t=0 pushes it forward).
        From the case-A rollouts, per regime (all 8), the fraction of instances
        whose first dispatched block starts strictly after t=0, and the max such
        offset in hours.

Emits paper/macros_misconfig.tex and rollout artifacts under
test_results/FAMILY/10x25_misconfig/.

Device: CPU only. Pin with `taskset -c 16-23` and OMP/MKL threads = 4.
Usage: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 taskset -c 16-23 \
       python3 scripts/p40_misconfig.py [--limit N]
"""
import os
import sys
import json
import glob

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
torch.set_num_threads(4)
from scipy.stats import wilcoxon                       # noqa: E402

from params import configs                             # noqa: E402
# CPU only. The env's State class captures torch.device(configs.device) at class
# definition time, so this must be set BEFORE the env modules are imported.
configs.device = 'cpu'
from constraint_family.descriptor import (             # noqa: E402
    ConstraintDescriptor, compute_token, TOKEN_DIM, IDLE_CLASS)
from constraint_family.fjsp_env_family import (        # noqa: E402
    FJSPEnvFamily, resolve_delta_mode)
from constraint_family.reference_sim import validate_schedule  # noqa: E402
from ppvc_instance_generator import load_instance      # noqa: E402
from common_utils import greedy_select_action          # noqa: E402

DATA = 'data/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
OUT = 'test_results/FAMILY/10x25_misconfig'
MACRO = 'paper/macros_misconfig.tex'
TOL = 1e-6
B = 20000                                              # bootstrap resamples
BOOT_RNG = np.random.default_rng(7)                    # p11/p31's stream

# ---- E3 protocol constants (identical to scripts/p2_gate_g1.py / p17) --------
E3_REGIMES = ['L', 'S', 'W', 'LS', 'LW']
CF = {'L': 'S', 'S': 'L', 'W': 'LS', 'LS': 'W', 'LW': 'S'}   # conflicting token
CH = {'L': ('lag',), 'S': ('setup',), 'W': ('window',),
      'LS': ('lag', 'setup'), 'LW': ('lag', 'window')}       # full-blind channels
SEED_MODELS = [('301', '10x25+family+joint-v1'),
               ('302', '10x25+family+joint-v1-s302'),
               ('303', '10x25+family+joint-v1-s303')]
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode')

LIMIT = int(_ARGV[_ARGV.index('--limit') + 1]) if '--limit' in _ARGV else 0


# ---------------------------------------------------------------------------
def load_policy(model_tag):
    """Load a family checkpoint on CPU, restoring its trained architecture."""
    with open(f'train_log/FAMILY/config_{model_tag}.json') as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v in ('True', 'False'):
                v = v == 'True'
            setattr(configs, k, v)
    configs.device = 'cpu'
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/FAMILY/{model_tag}.pth', map_location='cpu'))
    ppo.policy.eval()
    return ppo


def names(regime):
    ps = sorted(glob.glob(os.path.join(DATA, 'desc', regime, '*.desc.json')))
    if LIMIT:
        ps = ps[:LIMIT]
    return [os.path.basename(p).replace('.desc.json', '') for p in ps]


def load_desc(regime, name):
    with open(os.path.join(DATA, 'desc', regime, f'{name}.desc.json')) as f:
        return ConstraintDescriptor.from_json(json.load(f))


def rollout_batch(ppo, insts, env_descs, token_vecs, blind=()):
    """Greedy rollout of `ppo` over a whole batch of instances at once.

    Every FAMILY instance is 10x25 (50 operations), so the SameOpNums env
    processes all of them in one vectorized episode: identical arithmetic to a
    batch-of-1 rollout per instance (each env is independent; there is no
    cross-instance coupling), validated against the batch-of-1 path.

    `insts` is a list of (job_length, op_pt, meta); `env_descs` and `token_vecs`
    are one per instance. Returns per-instance makespan, op start/completion
    (true hours), machine assignment, and the first DISPATCHED (op, machine).
    """
    E = len(insts)
    jls = [x[0] for x in insts]
    pts = [x[1] for x in insts]
    metas = [x[2] for x in insts]
    N, M = pts[0].shape
    env = FJSPEnvFamily(jls[0].shape[0], M, family_mode=True, blind_channels=blind)
    dm = resolve_delta_mode(configs)
    env.delta_mode = dm
    env.reward_delta_env = (dm != 'zero')
    env.set_initial_data(jls, pts, descriptor_list=list(env_descs),
                         op_type_list=[m['op_type'] for m in metas],
                         mch_type_list=[m['mch_type'] for m in metas])
    env.token_g = np.asarray(token_vecs)
    env._attach_token()
    state = env.state
    amch = np.full((E, N), -1, dtype=int)
    first_op = np.full(E, -1, dtype=int)
    first_m = np.full(E, -1, dtype=int)
    ar = np.arange(E)
    step_i = 0
    while not env.done().all():
        done_before = np.asarray(env.done()).reshape(-1).astype(bool).copy()
        with torch.no_grad():
            pi, _ = ppo.policy(
                fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor, op_type=state.op_type_tensor,
                mch_type=state.mch_type_tensor, token=state.token_tensor)
        acts = greedy_select_action(pi).cpu().numpy()
        cj, cm = acts // M, acts % M
        ops = env.candidate[ar, cj]
        live = ~done_before                      # freeze already-finished envs
        if step_i == 0:
            first_op[live], first_m[live] = ops[live], cm[live]
        sel = ar[live]
        amch[sel, ops[sel]] = cm[sel]
        state, _, _ = env.step(acts)
        step_i += 1
    op_ct = np.asarray(env.true_op_ct, dtype=float).copy()          # [E, N]
    pt_arr = np.stack(pts)                                          # [E, N, M]
    chosen_pt = np.take_along_axis(pt_arr, amch[:, :, None], axis=2)[:, :, 0]
    op_st = op_ct - chosen_pt
    ms = np.asarray(env.current_makespan, dtype=float).reshape(-1).copy()
    return dict(ms=ms, op_st=op_st, op_ct=op_ct, amch=amch,
                first_op=first_op, first_m=first_m)


def token_of(desc, pt):
    return compute_token(desc, float(pt[pt > 0].mean()), pt.shape[1])


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (order preserved). Matches p17."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (m - i) * pvals[k])
        adj[k] = min(1.0, running)
    return adj


def classify(viols):
    """Bucket reference_sim violation strings into deployment-relevant types."""
    c = dict(lag=0, setup=0, window=0, other=0)
    for s in viols:
        if 'pred ct+lag' in s:
            c['lag'] += 1
        elif 'setup+proc block overlaps' in s:
            c['setup'] += 1
        elif 'overlaps outage' in s:
            c['window'] += 1
        else:
            c['other'] += 1
    return c


def rmac(name, val):
    return f'\\newcommand{{\\{name}}}{{{val}}}'


# ---------------------------------------------------------------------------
def experiment_i(macros):
    """(i) E3 token-intervention family replicated across three training seeds."""
    print('=' * 72)
    print('(i) E3 TOKEN INTERVENTION ACROSS SEEDS 301/302/303')
    print('    conflicting / withheld / full-blind x L,S,W,LS,LW = 15 tests,')
    print('    one-sided paired Wilcoxon, Holm across the family (as p17)')
    print('=' * 72)

    # cache base rollouts per (seed, regime); descriptors/instances are shared
    w_deltas, w_qs, w_replicate = {}, {}, 0
    w_boot = {}
    for seed, model in SEED_MODELS:
        ppo = load_policy(model)
        rows, raw = [], []           # rows: (intervention, regime, delta, p)
        w_base_vec = w_conf_vec = None
        for regime in E3_REGIMES:
            nm = names(regime)
            true_descs = [load_desc(regime, n) for n in nm]
            cf_descs = [load_desc(CF[regime], n) for n in nm]
            insts = [load_instance(os.path.join(BASE, n)) for n in nm]
            pts = [x[1] for x in insts]
            tok_true = [token_of(td, pt) for td, pt in zip(true_descs, pts)]
            tok_conf = [token_of(cd, pt) for cd, pt in zip(cf_descs, pts)]
            tok_zero = [np.zeros(TOKEN_DIM)] * len(insts)
            base = rollout_batch(ppo, insts, true_descs, tok_true)['ms']
            conf = rollout_batch(ppo, insts, true_descs, tok_conf)['ms']
            zero = rollout_batch(ppo, insts, true_descs, tok_zero)['ms']
            fbld = rollout_batch(ppo, insts, true_descs, tok_zero,
                                 blind=CH[regime])['ms']
            np.save(f'{OUT}/Result_greedy+{model}_{regime}.npy', base)
            np.save(f'{OUT}/Result_greedy+{model}+cf-{CF[regime]}_{regime}.npy', conf)
            np.save(f'{OUT}/Result_greedy+{model}+withhold_{regime}.npy', zero)
            np.save(f'{OUT}/Result_greedy+{model}+fullblind_{regime}.npy', fbld)
            for nmv, vec in (('conflicting', conf), ('withheld', zero),
                             ('full-blind', fbld)):
                p = wilcoxon(vec, base, alternative='greater').pvalue \
                    if np.any(vec != base) else 1.0
                rows.append((nmv, regime, float(np.mean(vec - base)), p))
                raw.append(p)
            if regime == 'W':
                w_base_vec, w_conf_vec = base, conf
        adj = holm(np.array(raw))
        print(f'\nseed {seed} ({model}):')
        print(f'  {"interv":11s} {"reg":4s} {"delta(h)":>9s} {"Holm q":>10s}')
        for (nmv, regime, d, _), q in zip(rows, adj):
            mark = ''
            if nmv == 'conflicting' and regime == 'W':
                w_deltas[seed], w_qs[seed] = d, q
                if q < 0.05 and d > 0:
                    w_replicate += 1
                    mark = '  <-- W conflicting (published cell)'
                else:
                    mark = '  <-- W conflicting (published cell) DID NOT REPLICATE'
            print(f'  {nmv:11s} {regime:4s} {d:+9.2f} {q:10.2e}{mark}')
        # paired-bootstrap CI on the W conflicting delta for this seed
        diff = w_conf_vec - w_base_vec
        idx = BOOT_RNG.integers(0, len(diff), (B, len(diff)))
        bs = diff[idx].mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        w_boot[seed] = (float(diff.mean()), float(lo), float(hi))
        print(f'  W conflicting delta {diff.mean():+.2f} h, 95% CI '
              f'[{lo:+.2f},{hi:+.2f}] (paired bootstrap B={B})')

    print(f'\nW conflicting-token effect replicates (Holm q<0.05 AND delta>0) on '
          f'{w_replicate}/3 seeds.')

    macros.append('% (i) E3 conflicting-token W-regime cell, per training seed')
    for seed, key in (('301', 'One'), ('302', 'Two'), ('303', 'Three')):
        m, lo, hi = w_boot[seed]
        macros.append(
            rmac(f'MisWCfSeed{key}Delta', f'{w_deltas[seed]:+.2f}') +
            rmac(f'MisWCfSeed{key}Q', f'{w_qs[seed]:.1e}') +
            rmac(f'MisWCfSeed{key}Lo', f'{lo:+.2f}') +
            rmac(f'MisWCfSeed{key}Hi', f'{hi:+.2f}') +
            f'  % seed {seed}: W conflicting-token delta(h), Holm q, 95% CI')
    macros.append(rmac('MisWCfReplicate', w_replicate) +
                  '  % of 3 seeds where the W conflicting-token effect survives '
                  'Holm (q<0.05) with delta>0')
    macros.append(rmac('MisNSeeds', 3))


# ---------------------------------------------------------------------------
PAIRS = [('LSW', 'N', 'Lsw'),     # true=LSW, false=N: all rules missing
         ('SW', 'LS', 'Sw')]      # true=SW, false=LS: one rule swapped for another
CASES = ['A', 'B', 'C', 'D']


def experiment_ii(macros):
    """(ii) environment misconfiguration, validated by the independent simulator."""
    print('\n' + '=' * 72)
    print('(ii) ENVIRONMENT MISCONFIGURATION (seed 301 checkpoint)')
    print('     A env=true/tok=true  B env=true/tok=false  '
          'C env=false/tok=false  D env=false/tok=true')
    print('     schedules validated against the TRUE descriptor by reference_sim')
    print('=' * 72)
    ppo = load_policy(SEED_MODELS[0][1])           # seed 301
    model = SEED_MODELS[0][1]

    for true_reg, false_reg, plabel in PAIRS:
        nm = names(true_reg)
        insts = [load_instance(os.path.join(BASE, n)) for n in nm]
        tdesc = [load_desc(true_reg, n) for n in nm]
        fdesc = [load_desc(false_reg, n) for n in nm]
        print(f'\npair: true={true_reg}  false={false_reg}  '
              f'({len(nm)} instances)')
        print(f'  {"case":4s} {"env":5s} {"token":6s} {"mean ms":>8s} '
              f'{"dMs%":>7s} {"infeas":>7s}  violations(setup/window/lag)')
        base_mean = None
        pts = [x[1] for x in insts]
        for case in CASES:
            env_desc = tdesc if case in ('A', 'B') else fdesc
            tok_desc = tdesc if case in ('A', 'D') else fdesc
            tok = [token_of(d, pt) for d, pt in zip(tok_desc, pts)]
            r = rollout_batch(ppo, insts, env_desc, tok)
            feas_ct, vt = 0, dict(lag=0, setup=0, window=0, other=0)
            for i, ((jl, pt, meta), truth) in enumerate(zip(insts, tdesc)):
                viols = validate_schedule(jl, pt, truth, r['op_st'][i],
                                          r['op_ct'][i], r['amch'][i])
                if not viols:
                    feas_ct += 1
                for k, v in classify(viols).items():
                    vt[k] += v
            ms = r['ms']
            np.save(f'{OUT}/Result_greedy+{model}+misconfig-{plabel}-{case}_'
                    f'{true_reg}.npy', ms)
            if case == 'A':
                base_mean = ms.mean()
            dms = 100.0 * (ms.mean() - base_mean) / base_mean
            infeas = len(nm) - feas_ct
            edlab = true_reg if case in ('A', 'B') else false_reg
            tklab = true_reg if case in ('A', 'D') else false_reg
            print(f'  {case:4s} {edlab:5s} {tklab:6s} {ms.mean():8.2f} '
                  f'{dms:+7.2f} {infeas:5d}/{len(nm)}  '
                  f'{vt["setup"]}/{vt["window"]}/{vt["lag"]}'
                  + (f' (+{vt["other"]} other)' if vt['other'] else ''))
            macros.append(
                rmac(f'Mis{plabel}{case}Ms', f'{ms.mean():.2f}') +
                rmac(f'Mis{plabel}{case}dMs', f'{dms:+.2f}') +
                rmac(f'Mis{plabel}{case}Infeas', infeas) +
                rmac(f'Mis{plabel}{case}Win', vt['window']) +
                rmac(f'Mis{plabel}{case}Set', vt['setup']) +
                rmac(f'Mis{plabel}{case}Lag', vt['lag']) +
                f'  % pair {true_reg}/{false_reg} case {case}: mean ms, dMs% vs A,'
                f' #infeasible under truth, window/setup/lag violation counts')
        macros.append(rmac(f'Mis{plabel}True', true_reg) +
                      rmac(f'Mis{plabel}False', false_reg) +
                      f'  % pair {plabel}: true and false regime labels')


# ---------------------------------------------------------------------------
ALL8 = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']


def experiment_iii(macros):
    """(iii) first-dispatch offset: how often the t=0 mask degeneracy bites."""
    print('\n' + '=' * 72)
    print('(iii) FIRST-DISPATCH EXCEPTION (case A, seed 301, all 8 regimes)')
    print('      fraction of instances whose first dispatched block starts > t=0')
    print('=' * 72)
    ppo = load_policy(SEED_MODELS[0][1])
    print(f'  {"regime":6s} {"n>0/N":>9s} {"frac":>7s} {"max offset(h)":>14s}')
    worst_frac, worst_off, worst_reg = 0.0, 0.0, 'N'
    for regime in ALL8:
        nm = names(regime)
        insts = [load_instance(os.path.join(BASE, n)) for n in nm]
        descs = [load_desc(regime, n) for n in nm]
        pts = [x[1] for x in insts]
        r = rollout_batch(ppo, insts, descs, [token_of(d, pt)
                                              for d, pt in zip(descs, pts)])
        offsets = []
        for i, d in enumerate(descs):
            op = int(r['first_op'][i])
            s = (d.setup[IDLE_CLASS, d.op_class[op]] if d.has_setup else 0.0)
            offsets.append(float(r['op_st'][i][op] - s))   # block start vs t=0
        offsets = np.array(offsets)
        n_after = int((offsets > TOL).sum())
        frac = n_after / len(nm)
        mx = float(offsets.max()) if len(offsets) else 0.0
        print(f'  {regime:6s} {n_after:4d}/{len(nm):<4d} {frac:7.3f} {mx:14.2f}')
        if frac > worst_frac:
            worst_frac, worst_reg = frac, regime
        worst_off = max(worst_off, mx)
        macros.append(
            rmac(f'MisFdFrac{regime}', f'{frac:.3f}') +
            rmac(f'MisFdPct{regime}', f'{100.0 * frac:.0f}') +
            rmac(f'MisFdMax{regime}', f'{mx:.2f}') +
            f'  % regime {regime}: fraction (and %) of first dispatches starting'
            f' after t=0, max offset (h)')
    print(f'\n  worst regime by fraction: {worst_reg} '
          f'({worst_frac:.3f}); largest first-dispatch offset anywhere '
          f'{worst_off:.2f} h')
    macros.append(rmac('MisFdWorstReg', worst_reg) +
                  rmac('MisFdWorstFrac', f'{worst_frac:.3f}') +
                  rmac('MisFdWorstPct', f'{100.0 * worst_frac:.0f}') +
                  rmac('MisFdWorstOff', f'{worst_off:.2f}') +
                  '  % regime with the largest first-dispatch fraction, that '
                  'fraction (and %), and the largest offset across all regimes (h)')


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    macros = ['% auto-generated by scripts/p40_misconfig.py -- do not hand-edit',
              '% Probes around E3 (token-intervention causality).',
              '% Fresh rollout campaign under test_results/FAMILY/10x25_misconfig/;',
              '% never mixed with the published E3 probe (rollouts are not',
              '% reproducible across machines).']
    experiment_i(macros)
    experiment_ii(macros)
    experiment_iii(macros)
    with open(MACRO, 'w') as f:
        f.write('\n'.join(macros) + '\n')
    print(f'\nwrote {MACRO}')
    print(f'wrote rollout artifacts under {OUT}/')


if __name__ == '__main__':
    main()
