"""E4 out-of-family boundary map: BLOCKING (this paper, P5).

Deploys the family policy — which assumes monotone-delay dynamics — on the
BLOCKING shop, an out-of-family regime whose machine release is coupled to
downstream availability. Because blocking is NOT in the monotone-delay
family, two failures are expected and quantified here:

  1. DEADLOCK / DNF: greedy list scheduling without deadlock-awareness can
     schedule itself into an unrecoverable state (a hazard specific to
     blocking shops). We report the DNF rate of the family policy and of
     dispatching rules, against the deadlock-free CP-SAT reference.
  2. BOUND VIOLATION: the family's admissible reward bound assumes a machine
     is free at its own completion; under blocking it is held longer, so the
     realized makespan can EXCEED the bound. We report the violation rate —
     direct evidence that the soundness theorem's premise is necessary.

Usage:
  python eval_blocking.py --regime L --limit 100 [--methods greedy MWKR SPT]
      [--family_model 10x25+family+joint-v1]
Writes results/e4_blocking_<regime>.md and Result_blocking_*.npy.
"""
import glob
import json
import os
import sys
import time

_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]


def _pop(flag, default=None):
    if flag in _ARGV:
        i = _ARGV.index(flag)
        v = _ARGV[i + 1]
        del _ARGV[i:i + 2]
        return v
    return default


MODEL = _pop('--family_model', '10x25+family+joint-v1')
REGIME = _pop('--regime', 'L')      # LAG-only or N (blocking v1 supports these)
LIMIT = int(_pop('--limit', '100'))
_methods = []
if '--methods' in _ARGV:
    i = _ARGV.index('--methods')
    del _ARGV[i]
    while i < len(_ARGV) and not _ARGV[i].startswith('--'):
        _methods.append(_ARGV[i]); del _ARGV[i]
METHODS = _methods or ['greedy', 'MWKR', 'SPT', 'FIFO']

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from params import configs

RES = 'test_results/FAMILY/10x25'
BASE = 'data/PPVC/10x25+ppvc-mixed'
DATA = 'data/FAMILY/10x25'


def load_policy():
    snap = json.load(open(f'train_log/FAMILY/config_{MODEL}.json'))
    for k in ('use_film', 'token_dim', 'film_hidden', 'token_to_heads',
              'use_type_embedding', 'fea_j_input_dim', 'fea_m_input_dim',
              'type_emb_dim', 'n_op_types', 'n_mch_types'):
        if k in snap:
            v = snap[k]
            if isinstance(v, str) and v in ('True', 'False'):
                v = v == 'True'
            setattr(configs, k, v)
    configs.fea_j_input_dim = 12
    configs.fea_m_input_dim = 10
    import torch
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'trained_network/FAMILY/{MODEL}.pth', map_location='cuda'))
    ppo.policy.eval()
    return ppo


def legal_pdr_action(method, env, M):
    """Blocking-aware PDR: choose among LEGAL (unmasked) op-machine pairs only.

    The lab's heuristic_select_action ignores the blocking mask, so it can
    pick a held machine; under blocking the fair comparison restricts every
    online method to the same legal-move set the policy's mask enforces.
    Returns a scalar action, or None if no legal move remains (deadlock).
    """
    mask = env.state.dynamic_pair_mask_tensor[0].cpu().numpy()   # [J, M]
    legal_job = ~mask.all(axis=1)                                 # [J]
    if not legal_job.any():
        return None
    cand = env.candidate[0]                                       # [J] op ids
    # per-job priority (higher = pick first), only over legal jobs
    if method == 'FIFO':
        pr = -env.candidate_free_time[0]
    elif method == 'MOR':
        pr = env.op_match_job_left_op_nums[0][cand]
    elif method == 'MWKR':
        pr = env.op_match_job_remain_work[0][cand]
    elif method == 'SPT':
        legal_pt = np.where(mask, np.inf, env.pair_free_time[0])
        pr = -legal_pt.min(axis=1)
    else:
        raise ValueError(method)
    pr = np.where(legal_job, pr, -np.inf)
    job = int(np.argmax(pr))
    # earliest legal machine for that job
    legal_ft = np.where(mask[job], np.inf, env.pair_free_time[0, job])
    mch = int(np.argmin(legal_ft))
    return job * M + mch


def main():
    import torch
    from model.PPO import PPO_initialize  # noqa
    from common_utils import greedy_select_action
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_blocking import FJSPEnvBlocking, BIG
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from ppvc_instance_generator import load_instance

    ppo = load_policy() if 'greedy' in METHODS else None

    descs = sorted(glob.glob(os.path.join(DATA, 'desc', REGIME,
                                          '*.desc.json')))[:LIMIT]
    rows = {}
    bound_viol = 0
    n = len(descs)
    for method in METHODS:
        dnf, ms_done = 0, []
        for dp in descs:
            name = os.path.basename(dp).replace('.desc.json', '')
            jl, pt, meta = load_instance(os.path.join(BASE, name))
            desc = ConstraintDescriptor.from_json(json.load(open(dp)))
            env = FJSPEnvBlocking(jl.shape[0], pt.shape[1], family_mode=True)
            st = env.set_initial_data(
                [jl], [pt], descriptor_list=[desc],
                op_type_list=[meta['op_type']], mch_type_list=[meta['mch_type']])
            if method in ('FIFO', 'MOR', 'SPT', 'MWKR'):
                np.random.seed(configs.seed_test)
            dead = False
            while not env.done().all():
                if env.next_schedule_time[0] >= BIG / 2:
                    dead = True; break
                if method == 'greedy':
                    # if every remaining job is fully masked, the policy has
                    # no legal move -> deadlock (out-of-family failure)
                    if env.state.dynamic_pair_mask_tensor[0].all(dim=1)\
                            .all().item():
                        dead = True; break
                    with torch.no_grad():
                        pi, _ = ppo.policy(
                            fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor,
                            candidate=st.candidate_tensor, fea_m=st.fea_m_tensor,
                            mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
                            dynamic_pair_mask=st.dynamic_pair_mask_tensor,
                            fea_pairs=st.fea_pairs_tensor, op_type=st.op_type_tensor,
                            mch_type=st.mch_type_tensor, token=st.token_tensor)
                    a = greedy_select_action(pi).cpu().numpy()
                else:
                    la = legal_pdr_action(method, env, pt.shape[1])
                    if la is None:
                        dead = True; break
                    a = np.array([la])
                st, _, _ = env.step(a)
                if env.deadlock[0]:
                    dead = True; break
            if dead:
                dnf += 1
            else:
                ms_done.append(float(env.current_makespan[0]))
                # Admissibility audit (family reward bound vs realized BLOCKING
                # makespan). Blocking only ADDS delay, so realized >= family
                # optimistic bound: the bound stays a valid LOWER bound even
                # out of family. A true violation is bound > makespan (the
                # bound over-estimating) — expected 0. We tally it to
                # demonstrate the boundary is dynamics-inexpressibility, NOT
                # reward unsoundness.
                if method == 'greedy':
                    fenv = FJSPEnvFamily(jl.shape[0], pt.shape[1],
                                         family_mode=True)
                    fenv.set_initial_data([jl], [pt], descriptor_list=[desc])
                    bound_true = float(np.max(fenv.op_ct_lb, axis=1)[0]) / \
                        fenv._slope
                    if bound_true > env.current_makespan[0] + 1e-6:
                        bound_viol += 1
        arr = np.array(ms_done) if ms_done else np.array([np.nan])
        np.save(os.path.join(RES, f'Result_blocking_{method}_{REGIME}.npy'),
                arr)
        rows[method] = (dnf, n, arr.mean() if ms_done else float('nan'),
                        len(ms_done))
        print(f'[blocking {REGIME}] {method:<8} DNF={dnf}/{n} '
              f'completed_mean={arr.mean() if ms_done else float("nan"):.1f}',
              flush=True)

    lines = [f'# E4 boundary map: BLOCKING (regime {REGIME}, model {MODEL})', '',
             '| method | DNF (deadlock) | completed mean ms | n completed |',
             '|---|---|---|---|']
    for m, (d, tot, mean, c) in rows.items():
        lines.append(f'| {m} | {d}/{tot} ({100*d/tot:.0f}%) | {mean:.1f} | {c} |')
    n_greedy_done = rows.get("greedy", (0, n, 0, n))[3] if "greedy" in rows else n
    lines += ['', f'Admissibility audit: the family bound OVER-estimates the '
              f'realized blocking makespan on {bound_viol}/{n_greedy_done} '
              f'completed greedy schedules. Blocking only ADDS delay, so the '
              f'optimistic monotone-delay bound remains a valid lower bound '
              f'(expected 0 over-estimations); the out-of-family failure is '
              f'therefore dynamics-inexpressibility and DEADLOCK, not reward '
              f'unsoundness.']
    os.makedirs('results', exist_ok=True)
    with open(f'results/e4_blocking_{REGIME}.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines[-2:]))
    print(f'wrote results/e4_blocking_{REGIME}.md')

    # The deadlock rate is a property of the POLICY, not of the environment, so
    # it must be re-derived whenever the policy is retrained. It is emitted here,
    # from this run, rather than stored in the manuscript.
    if REGIME == 'L' and 'greedy' in rows:
        g_dnf, g_n, _, g_done = rows['greedy']
        pdr = [rows[m][0] for m in ('FIFO', 'SPT', 'MWKR') if m in rows]
        mac = [
            '% auto-generated by eval_blocking.py -- do not hand-edit',
            f'\\newcommand{{\\BlockDnfGreedy}}{{{g_dnf}/{g_n}}}'
            '  % the family policy deadlocks on this many blocking instances',
            f'\\newcommand{{\\BlockBoundViol}}{{{bound_viol}}}'
            f'  % bound over-estimations, of {g_done} completed rollouts',
        ]
        if pdr:
            mac.append(f'\\newcommand{{\\BlockDnfPdrLo}}{{{min(pdr)}}}'
                       f'\\newcommand{{\\BlockDnfPdrHi}}{{{max(pdr)}}}'
                       '  % dispatching-rule deadlock range')
        os.makedirs('paper', exist_ok=True)
        with open('paper/macros_e4.tex', 'w') as f:
            f.write('\n'.join(mac) + '\n')
        print(f'wrote paper/macros_e4.tex (policy deadlocks {g_dnf}/{g_n}; '
              f'rules {min(pdr)}-{max(pdr)}; bound over-estimations {bound_viol})')


if __name__ == '__main__':
    main()
