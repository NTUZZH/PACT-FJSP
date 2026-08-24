"""Evaluate a family policy / PDR heuristics on the FAMILY benchmark (this paper).

Mirrors eval_ppvc.py's protocol (greedy + optional sampling + PDRs, batch-of-1
per instance, results as .npy + markdown summary) on the constraint-family
datasets, with test-time TOKEN INTERVENTIONS for the causality experiments:

  --token_mode true                 real token (default)
  --token_mode withhold             zero token (token-blind at test)
  --token_mode shuffle              tokens permuted across instances (seeded
                                    derangement-ish shift by 1)
  --token_mode counterfactual:REG   token computed from regime REG's
                                    descriptor of the SAME base instance

Usage:
  python eval_family.py --family_model 10x25+family+joint-v1 \
         --regimes N L S W LS LW SW LSW --methods greedy MWKR \
         [--token_mode true] [--data data/FAMILY/10x25] [--sample_times 100]
         [--torch_threads 1]
"""
import glob
import json
import os
import sys
import time

# consume private flags BEFORE params parses argv
def _pop(flag, default=None, has_val=True):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1] if has_val else True
        del sys.argv[i:i + 2 if has_val else i + 1]
        return v
    return default


ARG_MODEL = _pop('--family_model', '10x25+family+joint-v1')
ARG_DATA = _pop('--data', 'data/FAMILY/10x25')
ARG_BASE = _pop('--base', 'data/PPVC/10x25+ppvc-mixed')
ARG_TOKEN_MODE = _pop('--token_mode', 'true')
ARG_TAG = _pop('--tag', '')
ARG_LIMIT = int(_pop('--limit', '0'))     # 0 = all instances
ARG_BLIND = _pop('--blind_channels', '')  # e.g. 'window' / 'lag,setup' (E3)
# Torch thread count for the policy forward. Default 0 = leave torch alone, so
# every published evaluation keeps the threading it was run under. Pass 1 on a
# large instance: this is a batch-of-1 forward on one graph, too little
# arithmetic to parallelize, and per-operator thread barriers dominate above one
# thread (measured on the companion study, 1.66 ms at one thread against 141.88
# at eight on a contended box). Without the flag torch sizes its pool from
# OMP_NUM_THREADS, which the caller sets for the numpy environment code, so the
# two settings silently become one.
ARG_TORCH_THREADS = int(_pop('--torch_threads', '0'))
_regimes = []
if '--regimes' in sys.argv:
    i = sys.argv.index('--regimes')
    del sys.argv[i]
    while i < len(sys.argv) and not sys.argv[i].startswith('--'):
        _regimes.append(sys.argv[i])
        del sys.argv[i]
_methods = []
if '--methods' in sys.argv:
    i = sys.argv.index('--methods')
    del sys.argv[i]
    while i < len(sys.argv) and not sys.argv[i].startswith('--'):
        _methods.append(sys.argv[i])
        del sys.argv[i]

import numpy as np
import torch as _torch

if ARG_TORCH_THREADS:
    _torch.set_num_threads(ARG_TORCH_THREADS)
    try:
        _torch.set_num_interop_threads(ARG_TORCH_THREADS)
    except RuntimeError:
        pass          # already started; the intra-op cap is the one that matters

from params import configs

# reward_delta_env / delta_mode must be restored on the evaluation env: they
# select the delay envelope inside _bound_min_term, and that bound is not only
# the reward potential -- it is ALSO operation feature channel #2
# (construct_op_features). Restoring them here evaluates each policy on the
# observation convention it was trained with. The env's dynamics and the
# realized makespan do not depend on these flags, so makespans remain
# comparable across arms.
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode',
             # revision arms: the admissible-pair mask is the policy's action
             # space and the pair width is the actor's input, so both are
             # restored from the arm's training config. They are applied ONLY
             # to the env the policy rolls out in (see below); PDR baselines
             # always run under the published non-delay mask.
             'mask_mode', 'pair_extra')

REGIMES = _regimes or ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
METHODS = _methods or ['greedy', 'MWKR']
PDRS = ('FIFO', 'MOR', 'SPT', 'MWKR')


def load_model():
    snap_path = f'./train_log/FAMILY/config_{ARG_MODEL}.json'
    with open(snap_path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v in ('True', 'False'):
                v = v == 'True'
            setattr(configs, k, v)
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from model.PPO import PPO_initialize
    device = torch.device(configs.device)
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/FAMILY/{ARG_MODEL}.pth', map_location=device))
    ppo.policy.eval()
    return ppo


def make_token_override(regime, stems, descs, tokens):
    """Return per-instance token vectors after the intervention."""
    import copy
    from constraint_family.descriptor import ConstraintDescriptor, compute_token
    from ppvc_instance_generator import load_instance
    mode = ARG_TOKEN_MODE
    if mode == 'true':
        return tokens
    if mode == 'withhold':
        return [np.zeros_like(t) for t in tokens]
    if mode == 'shuffle':
        # deterministic cyclic shift: every instance gets another instance's
        # token from the SAME regime (magnitude mismatch, semantics kept)
        return [tokens[(i + 1) % len(tokens)] for i in range(len(tokens))]
    if mode.startswith('counterfactual:'):
        reg2 = mode.split(':', 1)[1].upper()
        out = []
        for i, stem in enumerate(stems):
            name = os.path.basename(stem)
            with open(os.path.join(ARG_DATA, 'desc', reg2,
                                   f'{name}.desc.json')) as f:
                d2 = ConstraintDescriptor.from_json(json.load(f))
            jl, pt, meta = load_instance(stem)
            mp = float(pt[pt > 0].mean())
            out.append(compute_token(d2, mp, pt.shape[1]))
        return out
    sys.exit(f'unknown token_mode {mode}')


def main():
    from constraint_family.descriptor import ConstraintDescriptor, compute_token
    from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                    resolve_delta_mode)
    from constraint_family.reference_sim import validate_schedule
    from ppvc_instance_generator import load_instance
    from common_utils import (greedy_select_action, heuristic_select_action,
                              sample_action)
    import torch

    ppo = None
    if any(m in ('greedy', 'sampling') for m in METHODS):
        ppo = load_model()

    size_tag = os.path.basename(ARG_DATA.rstrip('/'))
    blind = tuple(b.strip() for b in ARG_BLIND.split(',') if b.strip())
    mode_tag = ('' if ARG_TOKEN_MODE == 'true'
                else '+' + ARG_TOKEN_MODE.replace(':', '-'))
    if blind:
        mode_tag += '+chblind-' + '-'.join(sorted(blind))
    out_root = f'./test_results/FAMILY/{size_tag}{ARG_TAG}'
    os.makedirs(out_root, exist_ok=True)

    summary_rows = []
    for regime in REGIMES:
        descs_paths = sorted(glob.glob(os.path.join(
            ARG_DATA, 'desc', regime, '*.desc.json')))
        if ARG_LIMIT:
            descs_paths = descs_paths[:ARG_LIMIT]
        stems, descs, tokens = [], [], []
        for dp in descs_paths:
            name = os.path.basename(dp).replace('.desc.json', '')
            stem = os.path.join(ARG_BASE, name)
            with open(dp) as f:
                desc = ConstraintDescriptor.from_json(json.load(f))
            jl, pt, meta = load_instance(stem)
            stems.append(stem)
            descs.append(desc)
            tokens.append(compute_token(desc, float(pt[pt > 0].mean()),
                                        pt.shape[1]))
        tok_override = make_token_override(regime, stems, descs, tokens)

        # CP-SAT references if available
        refs = {}
        for rp in glob.glob(f'./or_solution/FAMILY/{size_tag}/{regime}/*.json'):
            with open(rp) as f:
                r = json.load(f)
            if r.get('objective') is not None:
                refs[r['instance']] = r['objective']

        for method in METHODS:
            ms_list, t_list, feas = [], [], 0
            # The policy is evaluated on the action space and pair channels it
            # trained with (restored via ARCH_KEYS from its config snapshot).
            # PDR methods keep the published defaults: their numbers must not
            # move with the arm under evaluation.
            _policy_method = method in ('greedy', 'sampling')
            _mm = (getattr(configs, 'mask_mode', 'nondelay')
                   if _policy_method else 'nondelay')
            _pe = (getattr(configs, 'pair_extra', 'none')
                   if _policy_method else 'none')
            for i, stem in enumerate(stems):
                jl, pt, meta = load_instance(stem)
                env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                                    blind_channels=blind,
                                    mask_mode=_mm, pair_extra=_pe)
                # Give the policy the observation it was TRAINED on. The bound is
                # feature channel #2 as well as the reward potential, so a model
                # trained with the trivial envelope must be evaluated with it too,
                # or the delta=0 ablation measures an observation shift instead of
                # a reward. Dynamics and makespan do not depend on this flag.
                # resolve_delta_mode, NOT a default of 'min': an old checkpoint
                # carries reward_delta_env only, and defaulting delta_mode here
                # would override it and evaluate the ablation as the baseline.
                _dm = resolve_delta_mode(configs)
                env.delta_mode = _dm
                env.reward_delta_env = (_dm != 'zero')
                state = env.set_initial_data(
                    [jl], [pt], descriptor_list=[descs[i]],
                    op_type_list=[meta['op_type']],
                    mch_type_list=[meta['mch_type']])
                if ppo is not None:
                    env.token_g = np.asarray([tok_override[i]])
                    env._attach_token()
                    state = env.state
                amch = np.full(pt.shape[0], -1, dtype=int)
                t0 = time.time()
                if method == 'greedy':
                    while not env.done().all():
                        with torch.no_grad():
                            pi, _ = ppo.policy(
                                fea_j=state.fea_j_tensor,
                                op_mask=state.op_mask_tensor,
                                candidate=state.candidate_tensor,
                                fea_m=state.fea_m_tensor,
                                mch_mask=state.mch_mask_tensor,
                                comp_idx=state.comp_idx_tensor,
                                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                fea_pairs=state.fea_pairs_tensor,
                                op_type=state.op_type_tensor,
                                mch_type=state.mch_type_tensor,
                                token=state.token_tensor)
                        a = greedy_select_action(pi)
                        job = int(a.cpu().numpy()[0]) // pt.shape[1]
                        amch[env.candidate[0, job]] = int(a.cpu().numpy()[0]) % pt.shape[1]
                        state, _, _ = env.step(a.cpu().numpy())
                elif method == 'sampling':
                    # sample_times parallel copies of this instance; keep best
                    from common_utils import sample_action
                    K = int(configs.sample_times)
                    senv = FJSPEnvFamily(jl.shape[0], pt.shape[1],
                                         family_mode=True,
                                         mask_mode=_mm, pair_extra=_pe)
                    sstate = senv.set_initial_data(
                        [jl] * K, [pt] * K, descriptor_list=[descs[i]] * K,
                        op_type_list=[meta['op_type']] * K,
                        mch_type_list=[meta['mch_type']] * K)
                    senv.token_g = np.tile(np.asarray([tok_override[i]]),
                                           (K, 1))
                    senv._attach_token()
                    sstate = senv.state
                    torch.manual_seed(configs.seed_test)
                    amch_b = np.full((K, pt.shape[0]), -1, dtype=int)
                    t0 = time.time()
                    while not senv.done().all():
                        with torch.no_grad():
                            pi, _ = ppo.policy(
                                fea_j=sstate.fea_j_tensor,
                                op_mask=sstate.op_mask_tensor,
                                candidate=sstate.candidate_tensor,
                                fea_m=sstate.fea_m_tensor,
                                mch_mask=sstate.mch_mask_tensor,
                                comp_idx=sstate.comp_idx_tensor,
                                dynamic_pair_mask=sstate.dynamic_pair_mask_tensor,
                                fea_pairs=sstate.fea_pairs_tensor,
                                op_type=sstate.op_type_tensor,
                                mch_type=sstate.mch_type_tensor,
                                token=sstate.token_tensor)
                        action_envs, _ = sample_action(pi)
                        acts = action_envs.cpu().numpy()
                        cj, cm = acts // pt.shape[1], acts % pt.shape[1]
                        ops = senv.candidate[np.arange(K), cj]
                        amch_b[np.arange(K), ops] = cm
                        sstate, _, done = senv.step(acts)
                        if done.all():
                            break
                    best = int(np.argmin(senv.current_makespan))
                    env = senv          # downstream reads true_op_ct/makespan
                    amch = amch_b[best]
                    # narrow batched arrays to the best sample for validation
                    env.true_op_ct = senv.true_op_ct[best:best + 1]
                    env.current_makespan = senv.current_makespan[best:best + 1]
                elif method in PDRS:
                    np.random.seed(configs.seed_test)
                    while not env.done().all():
                        a = heuristic_select_action(method, env)
                        job = a // pt.shape[1]
                        amch[env.candidate[0, job]] = a % pt.shape[1]
                        env.step(np.array([a]))
                else:
                    sys.exit(f'unknown method {method}')
                dt = time.time() - t0
                ms = float(env.current_makespan[0])
                # independent feasibility re-validation from realized times
                op_ct = env.true_op_ct[0]
                s_arr = np.zeros(pt.shape[0])
                # recompute per-op setup from machine sequences for op_st
                # (validator recomputes internally; pass st = ct - pt)
                op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
                viol = validate_schedule(jl, pt, descs[i], op_st, op_ct, amch)
                if not viol:
                    feas += 1
                ms_list.append(ms)
                t_list.append(dt)
            ms_arr = np.array(ms_list)
            res = np.stack([ms_arr, np.array(t_list)], axis=1)
            mname = method if method not in ('greedy', 'sampling') \
                else f'{method}+{ARG_MODEL}{mode_tag}'
            np.save(os.path.join(out_root,
                                 f'Result_{mname}_{regime}.npy'), res)
            gaps = [100.0 * (ms_list[i] - refs[os.path.basename(stems[i])])
                    / refs[os.path.basename(stems[i])]
                    for i in range(len(stems))
                    if os.path.basename(stems[i]) in refs]
            row = (regime, mname, ms_arr.mean(), ms_arr.std(),
                   f'{feas}/{len(stems)}',
                   f'{np.mean(gaps):.2f}% ({len(gaps)})' if gaps else 'n/a')
            summary_rows.append(row)
            print(f'[{regime:>3}] {mname:<40} mean={ms_arr.mean():7.1f} '
                  f'std={ms_arr.std():5.1f}  feas={feas}/{len(stems)} '
                  f'gap={row[5]}', flush=True)

    sfx = ARG_MODEL + mode_tag + ARG_TAG
    with open(os.path.join(out_root, f'summary_{sfx}.md'), 'w') as f:
        f.write(f'# FAMILY eval — {sfx}\n\n')
        f.write('| regime | method | mean | std | feasible | gap(cov) |\n')
        f.write('|---|---|---|---|---|---|\n')
        for r in summary_rows:
            f.write(f'| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.1f} | {r[4]} '
                    f'| {r[5]} |\n')
    print(f'wrote {out_root}/summary_{sfx}.md', flush=True)


if __name__ == '__main__':
    main()
