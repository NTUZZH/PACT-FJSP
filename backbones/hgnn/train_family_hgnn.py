"""
Train the HGNN scheduler (Song et al., IEEE TII 2023) on PACT's constraint
family, with the upstream architecture and PPO hyperparameters of config.json
(not tuned here; they are copied into the resolved configuration).

Training data are drawn exactly as train_family.py draws them: a fresh batch of
--num_envs instances every --parallel_iter iterations (both 20, which are also
PACT's num_envs and reset_env_timestep), regimes balanced over
--family_regimes, a new generator seed and a new descriptor seed per instance,
from the global numpy and random streams seeded by --seed
(pact_bridge.sample_family_batch, checked against train_family.py by the
tests). With the same seed as a PACT run, the instance sequence is the same.

Validation follows PACT's protocol: every --save_timestep iterations, greedy
decoding on 20 fixed base instances per size in --vali_sizes and per training
regime (generator seed 100 + i, descriptor seed 100000 + i); the score is the
mean over (size, regime) of mean makespan / mean MWKR makespan, and the best
score's weights are kept. --vali_sizes 10 reproduces PACT's selection rule;
--vali_sizes 10,20 adds the 20-module sets.

Outputs (relative to --out_root, default the repository root):
    trained_network/FAMILY/<name>.pth        best validation weights
    train_log/FAMILY/config_<name>.json      resolved configuration
    train_log/FAMILY/valiquality_<name>.txt  validation curve
    train_log/FAMILY/reward_<name>.txt       per-iteration [iteration, reward, loss, seconds]
with <name> = <n_j>x<n_m>+family+hgnn-<reward>-<regime | joint>-s<seed>[-<suffix>]
('joint' = the six training regimes N,L,S,W,LS,LW).

Examples:
    python backbones/hgnn/train_family_hgnn.py --family_regimes L --hgnn_reward certified --seed 301 --device cuda
    python backbones/hgnn/train_family_hgnn.py --family_regimes N,L,S,W,LS,LW --hgnn_reward published \
        --seed 301 --device cuda
"""
import argparse
import copy
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppvc_hgnn as P  # noqa: E402  (sets sys.path and the compatibility modules)
import pact_bridge as B  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

TRAIN_REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW')


def parse_args(argv=None):
    cfg = P.load_config()
    tp, ep = cfg['train_paras'], cfg['env_paras']
    ap = argparse.ArgumentParser(description='Train HGNN on the constraint family')
    ap.add_argument('--family_regimes', type=str, default='N,L,S,W,LS,LW')
    ap.add_argument('--hgnn_reward', type=str, default='certified', choices=['published', 'certified'],
                    help="'published': HGNN's mean-processing-time estimate, blind to lags, setups and windows "
                         "(not a lower bound); 'certified': PACT's certified estimate C-hat")
    ap.add_argument('--hgnn_mask', type=str, default='pact', choices=['pact', 'exact'],
                    help="legal pairs: 'pact' = PACT's non-delay mask (identical action set); 'exact' = "
                         "non-delay set on the integer-hour axis")
    ap.add_argument('--n_j', type=int, default=10, help='modules per training instance')
    ap.add_argument('--factory', type=str, default='default', choices=list(B.FACTORY_NAMES))
    ap.add_argument('--mix', type=str, default='mixed')
    ap.add_argument('--num_envs', type=int, default=ep['batch_size'])
    ap.add_argument('--max_iterations', type=int, default=tp['max_iterations'])
    ap.add_argument('--parallel_iter', type=int, default=tp['parallel_iter'])
    ap.add_argument('--update_timestep', type=int, default=tp['update_timestep'])
    ap.add_argument('--save_timestep', type=int, default=tp['save_timestep'])
    ap.add_argument('--vali_sizes', type=str, default='10',
                    help="comma-separated validation sizes in modules, e.g. '10,20'")
    ap.add_argument('--vali_size', type=int, default=20, help='validation instances per size and regime (max 20)')
    ap.add_argument('--seed_vali', type=int, default=100)
    ap.add_argument('--lag_scale', type=float, default=1.0,
                    help='multiply every training and validation lag by this factor (rounded); evaluation uses '
                         'the true lags')
    ap.add_argument('--lag_features', action='store_true',
                    help='append the static-lag and remaining-lag operation features (in_size_ope 8)')
    ap.add_argument('--seed', type=int, default=300)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--threads', type=int, default=1, help='torch CPU threads')
    ap.add_argument('--model_suffix', type=str, default='')
    ap.add_argument('--no_checks', action='store_true',
                    help='skip the on-the-fly transition and estimate checks of the environment')
    ap.add_argument('--out_root', type=str, default=B.REPO_ROOT)
    ap.add_argument('--force', action='store_true', help='overwrite an existing checkpoint of the same name')
    args = ap.parse_args(argv)
    args.regimes = [r.strip().upper() for r in args.family_regimes.split(',') if r.strip()]
    assert args.regimes and all(r in ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW') for r in args.regimes)
    args.vali_size_list = [int(x) for x in args.vali_sizes.split(',') if x.strip()]
    return args, cfg


def model_name(args, n_m):
    if len(args.regimes) == 1:
        tag = args.regimes[0]
    elif sorted(args.regimes) == sorted(TRAIN_REGIMES):
        tag = 'joint'
    else:
        tag = '_'.join(args.regimes)
    name = '{0}x{1}+family+hgnn-{2}-{3}-s{4}'.format(args.n_j, n_m, args.hgnn_reward, tag, args.seed)
    if args.model_suffix:
        name += '-' + args.model_suffix
    return name


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def greedy(env, policy):
    '''Greedy decoding over the whole batch (upstream validate.py logic); returns realized makespans.'''
    import PPO_model
    memory = PPO_model.Memory()
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, _, dones = env.step(actions)
        done = dones.all()
    viol = env.validate_family()
    assert all(len(v) == 0 for v in viol), 'infeasible validation schedule: {0}'.format(
        [v[:2] for v in viol if v][:2])
    ms = env.realized_makespan().cpu().numpy().copy()
    env.reset()
    return ms


def main(argv=None):
    from env.fjsp_env_family import FJSPEnvFamilyHGNN
    args, cfg = parse_args(argv)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)

    n_m = B.n_machines(args.factory)
    name = model_name(args, n_m)
    net_dir = os.path.join(args.out_root, 'trained_network', 'FAMILY')
    log_dir = os.path.join(args.out_root, 'train_log', 'FAMILY')
    os.makedirs(net_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    ckpt = os.path.join(net_dir, name + '.pth')
    if os.path.exists(ckpt) and not args.force:
        sys.exit('{0} exists; pass --force to overwrite'.format(ckpt))

    train_paras = copy.deepcopy(cfg['train_paras'])
    train_paras.update(max_iterations=args.max_iterations, parallel_iter=args.parallel_iter,
                       update_timestep=args.update_timestep, save_timestep=args.save_timestep)
    env_paras, model_paras = P.make_paras(cfg, device, 'aware', args.lag_features, args.num_envs, args.n_j, n_m)
    checks = not args.no_checks

    # validation sets first: their MWKR norms reseed the numpy stream, as in train_family.py
    t0 = time.time()
    vali = B.family_vali_sets(args.regimes, args.vali_size_list, args.mix, args.factory,
                              seed_vali=args.seed_vali, vali_size=args.vali_size, lag_scale=args.lag_scale)
    vali_envs = {k: FJSPEnvFamilyHGNN(v[0], v[1], v[2], env_paras, reward=args.hgnn_reward,
                                      mask=args.hgnn_mask, checks=checks)
                 for k, v in vali.items()}
    vali_norm = {k: v[3] for k, v in vali.items()}
    print('[vali] MWKR norms: ' + '  '.join('{0}@{1}={2:.1f}'.format(r, s, vali_norm[(s, r)])
                                           for (s, r) in vali_norm) +
          '  ({0:.0f}s)'.format(time.time() - t0), flush=True)

    setup_seed(args.seed)
    import PPO_model
    memories = PPO_model.Memory()
    model = P.build_model(model_paras, train_paras, num_envs=args.num_envs)
    n_params = sum(p.numel() for p in model.policy.parameters())
    print('[model] params total={0}'.format(n_params), flush=True)

    snapshot = {'model_name': name, 'backbone': 'hgnn', 'n_params': n_params,
                'args': {k: v for k, v in vars(args).items()},
                'hgnn_reward': args.hgnn_reward, 'hgnn_mask': args.hgnn_mask,
                'n_j': args.n_j, 'n_m': n_m, 'lag_features': bool(args.lag_features),
                'env_paras': {k: v for k, v in env_paras.items() if k != 'device'},
                'model_paras': {k: v for k, v in model_paras.items() if k != 'device'},
                'train_paras': train_paras, 'upstream_config': cfg,
                'vali_norm': {'{0}@{1}'.format(r, s): v for (s, r), v in vali_norm.items()}}
    with open(os.path.join(log_dir, 'config_{0}.json'.format(name)), 'w') as f:
        json.dump(snapshot, f, indent=1, default=str)
    print('[train_family_hgnn] model={0} regimes={1} reward={2} mask={3} iterations={4} batch={5} device={6}'
          .format(name, args.regimes, args.hgnn_reward, args.hgnn_mask, args.max_iterations, args.num_envs,
                  device), flush=True)

    # same stream position as train_family.py at the start of its loop
    setup_seed(args.seed)
    log, vali_log = [], []
    best = float('inf')
    start = time.time()
    env = None
    for i in range(1, args.max_iterations + 1):
        t_it = time.time()
        if (i - 1) % args.parallel_iter == 0:
            jls, pts, descs, _, _ = B.sample_family_batch(args.num_envs, args.regimes, args.n_j, n_m, args.mix,
                                                          args.factory)
            descs = [B.scale_lags(d, args.lag_scale) for d in descs]
            env = FJSPEnvFamilyHGNN(jls, pts, descs, env_paras, reward=args.hgnn_reward, mask=args.hgnn_mask,
                                    checks=checks)

        state = env.state
        done = False
        dones = env.done_batch
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
        ep_ms = float(env.realized_makespan().mean())
        env.reset()

        loss, reward = float('nan'), float('nan')
        if i % args.update_timestep == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            memories.clear_memory()
        secs = time.time() - t_it
        log.append([i, float(reward), float(loss), secs])
        print('Ep {0}\t reward {1:.3f}\t ms {2:.1f}\t loss {3:.6f}\t {4:.1f}s'.format(i, reward, ep_ms, loss, secs),
              flush=True)

        if i % args.save_timestep == 0:
            scores = {k: float(greedy(e, model.policy_old).mean()) for k, e in vali_envs.items()}
            rel = float(np.mean([scores[k] / vali_norm[k] for k in scores]))
            vali_log.append(rel)
            if rel < best:
                best = rel
                torch.save(model.policy.state_dict(), ckpt)
            with open(os.path.join(log_dir, 'valiquality_{0}.txt'.format(name)), 'w') as f:
                f.write(str(vali_log))
            print('[vali @{0}] rel={1:.4f} (best {2:.4f})  '.format(i, rel, best) +
                  '  '.join('{0}@{1}={2:.1f}'.format(r, s, scores[(s, r)]) for (s, r) in scores), flush=True)
        with open(os.path.join(log_dir, 'reward_{0}.txt'.format(name)), 'w') as f:
            f.write(str(log))
    print('[train_family_hgnn] done in {0:.2f} h; best rel={1:.4f}; checkpoint {2}'.format(
        (time.time() - start) / 3600, best, ckpt), flush=True)


if __name__ == '__main__':
    main()
