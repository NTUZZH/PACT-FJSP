"""Measure the greedy policy's per-instance decision latency, honestly.

Why this needs its own script. The latency baked into the E5 result files
(column 1) was recorded during a bulk evaluation that shared the box with other
jobs: on 10x25, where every instance has exactly the same operation count and
therefore the same number of sequential decisions, it ranges from 2.3 s to
13.9 s. A 6x spread on a fixed-size workload is contention, not computation.
E5's whole argument is a time axis, so the policy's position on that axis cannot
be a number measured while eight training lanes were on the same GPU.

This replays the SAME rollout eval_family.py performs (batch-of-1, greedy
argmax, model already resident, which is the deployment condition: a controller
does not reload weights per instance) and times it on a quiet box, the first
instance of each regime dropped as warm-up.

Threads. A batch-1 forward pass of this network (32,786 parameters) is too
little arithmetic to spread over threads, so the measurement uses ONE torch
thread on one otherwise idle core: --torch_threads (default 1) sets
torch.set_num_threads and torch.set_num_interop_threads as the first torch
calls; --numpy_threads (default 1) sets OMP_NUM_THREADS, MKL_NUM_THREADS and
OPENBLAS_NUM_THREADS of the numpy environment code before numpy is imported.
The two never share one flag. The earlier records (policy_latency_cpu.json,
policy_latency_cpu_4core_blind.json: OMP_NUM_THREADS=4 on four pinned cores,
torch threads from that variable) are reproduced by --torch_threads 4
--numpy_threads 4 under taskset -c 0-3.

Checkpoint, environment and idle cores follow scripts/p75_deploy_cost.py
(shared helpers): the released policy of the active environment through
arm_registry (--seed 301 by default, or --seed select; --model names one
directly), refused with the missing file named when it is not on disk; the
rollout environment carries the environment's mask options, which must match
the checkpoint's training configuration; the run stops before measuring when
another process uses more than 5% of a core of the affinity mask, unless
--allow_busy (tests only). The record stores the torch thread counts (and, as
'cores', the torch thread count that p13_e5_table_fair.py reads), the numpy
thread variables, the affinity mask, the load average at start and end, the
idle-core samples before and after, the checkpoint, and cpu_share (CPU seconds
of the process over the timed rollouts / their wall seconds / torch threads;
at one thread below 0.9 marks the record invalid).

Output: results/policy_latency_<device>_<T>thread<out_tag>.json through
arm_registry.out (outside the default setting under PACT_OUT); an existing file
is refused without --force. No macro file is written here.

Usage:
  taskset -c 20 python scripts/p16_policy_latency.py [--device cpu|cuda]
      [--n 30] [--regime L] [--seed 301|select] [--model M]
      [--torch_threads 1] [--numpy_threads 1] [--out_tag TAG] [--force]
      [--allow_busy]
  Exact environment: PACT_ENV=exact PACT_OUT=<folder> before the command.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHELL_OMP = os.environ.get('OMP_NUM_THREADS')


def _early(argv, flag, default):
    v = default
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            v = argv[i + 1]
        elif a.startswith(flag + '='):
            v = a.split('=', 1)[1]
    return v


# before numpy (imported by arm_registry) and torch are imported
_K = _early(sys.argv, '--numpy_threads', '1')
if _K.isdigit() and int(_K) >= 1:
    for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[_v] = _K
if _early(sys.argv, '--device', 'cpu') == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import arm_registry as REG  # noqa: E402  (pops --pact-* options; imports numpy)
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]
import p75_deploy_cost as P75  # noqa: E402  (shared helpers; chdir to ROOT)

REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
ARCH_KEYS = P75.ARCH_KEYS


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--n', type=int, default=30,
                   help='measured instances per regime; one more is run and dropped')
    p.add_argument('--regime', default=None, help='one regime (default all eight)')
    p.add_argument('--model', default=None,
                   help='checkpoint name; default the released policy of the '
                        'active environment (arm_registry)')
    p.add_argument('--seed', default='301',
                   help="training seed of the released policy, or 'select'")
    p.add_argument('--out_tag', default='',
                   help='suffix of the output file name')
    p.add_argument('--torch_threads', type=int, default=1)
    p.add_argument('--numpy_threads', type=int, default=1)
    p.add_argument('--force', action='store_true',
                   help='allow replacing an existing output')
    p.add_argument('--allow_busy', action='store_true',
                   help='measure although the cores are not idle (tests only)')
    a = p.parse_args(argv)
    if a.torch_threads < 1 or a.numpy_threads < 1:
        p.error('thread counts must be at least 1')
    if a.regime is not None and a.regime not in REGIMES:
        p.error(f'--regime must be one of {REGIMES}')
    return a


def main():
    args = parse(_ARGV)
    dev = args.device
    regimes = [args.regime] if args.regime else list(REGIMES)
    REG.banner()
    name, ckpt_rel, cfg_rel = P75.resolve_checkpoint(args.model, args.seed)
    dst = os.path.join(ROOT, REG.out(
        f'results/policy_latency_{dev}_{args.torch_threads}thread{args.out_tag}.json'))
    if os.path.exists(dst) and not args.force:
        sys.exit(f'{dst} exists; pass --force to replace it')
    cores = P75.affinity()
    if SHELL_OMP not in (None, str(args.numpy_threads)):
        print(f'note: OMP_NUM_THREADS={SHELL_OMP} in the environment; this process '
              f'uses --numpy_threads {args.numpy_threads}', flush=True)

    import torch
    P75.set_torch_threads(torch, args.torch_threads)   # the first torch calls

    from params import configs
    with open(os.path.join(ROOT, cfg_rel)) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            setattr(configs, k, (v == 'True') if isinstance(v, str) and
                    v in ('True', 'False') else v)
    for k, dflt in P75.ENV_DEFAULTS.items():
        v = snap.get(k, dflt)
        if isinstance(v, str) and v in ('True', 'False'):
            v = v == 'True'
        if v != dflt:
            sys.exit(f'{name}: snapshot sets {k}={v!r}; this rollout uses the '
                     f'published default {dflt!r}. Time it with eval_family.py '
                     f'conventions instead.')
    configs.device = dev         # must precede the env import (class attribute)

    from model.PPO import PPO_initialize
    from common_utils import greedy_select_action
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from ppvc_instance_generator import load_instance
    masks = P75.mask_options(snap, name)

    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        os.path.join(ROOT, ckpt_rel), map_location=torch.device(dev)))
    ppo.policy.eval()
    n_param = int(sum(p.numel() for p in ppo.policy.parameters()))
    threads = torch.get_num_threads()
    print(f'model {name} ({REG.env()} environment, {n_param} parameters); '
          f'device {dev}; torch threads {threads} (inter-op '
          f'{torch.get_num_interop_threads()}); numpy threads {args.numpy_threads}; '
          f'affinity {cores}', flush=True)

    started = time.strftime('%Y-%m-%d %H:%M:%S')
    la_start = P75.loadavg()
    before = P75.idle_check(cores, args.allow_busy, 'before')
    out = {}
    cpu_all, wall_all = 0.0, 0.0
    for regime in regimes:
        descs = sorted(glob.glob(
            f'data/FAMILY/10x25/desc/{regime}/*.desc.json'))[:args.n + 1]
        lat, n_ops = [], 0
        cpu_r, wall_r = 0.0, 0.0
        for k, dp in enumerate(descs):
            nm = os.path.basename(dp).replace('.desc.json', '')
            jl, pt, meta = load_instance(f'data/PPVC/10x25+ppvc-mixed/{nm}')
            with open(dp) as f:
                desc = ConstraintDescriptor.from_json(json.load(f), setup_convention='attached')
            env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True, **masks)
            state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                         op_type_list=[meta['op_type']],
                                         mch_type_list=[meta['mch_type']])
            n_ops = int(pt.shape[0])
            if dev == 'cuda':
                torch.cuda.synchronize()
            c0 = time.process_time()
            t0 = time.perf_counter()
            while not env.done().all():
                with torch.no_grad():
                    pi, _ = ppo.policy(
                        fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                        candidate=state.candidate_tensor,
                        fea_m=state.fea_m_tensor,
                        mch_mask=state.mch_mask_tensor,
                        comp_idx=state.comp_idx_tensor,
                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                        fea_pairs=state.fea_pairs_tensor,
                        op_type=state.op_type_tensor,
                        mch_type=state.mch_type_tensor, token=state.token_tensor)
                act = greedy_select_action(pi)
                state, _, _ = env.step(act.cpu().numpy())
            if dev == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            dc = time.process_time() - c0
            if k > 0:                   # drop the first: it pays lazy-init costs
                lat.append(dt)
                cpu_r += dc
                wall_r += dt
        if not lat:
            sys.exit(f'{regime}: fewer than two instances, nothing measured')
        lat.sort()
        med = statistics.median(lat)
        share = cpu_r / (wall_r * threads)
        cpu_all += cpu_r
        wall_all += wall_r
        out[regime] = {'median_s': med, 'p05_s': lat[int(.05 * len(lat))],
                       'p95_s': lat[-1 - int(.05 * len(lat))], 'n': len(lat),
                       'decisions': n_ops, 'cpu_share': share}
        print(f'{regime:4s} n={len(lat):3d}  median {med*1000:7.1f} ms  '
              f'({med*1000/n_ops:.2f} ms/decision)  '
              f'spread {lat[0]*1000:.0f}-{lat[-1]*1000:.0f} ms  '
              f'cpu_share {share:.3f}', flush=True)
    la_end = P75.loadavg()
    after = P75.idle_check(cores, True, 'after', record_only=True)

    share = cpu_all / (wall_all * threads)
    valid = before['idle'] and after['idle'] and (threads != 1 or share >= P75.SHARE_MIN)
    worst = max(v['median_s'] for v in out.values())
    rec = {'device': dev,
           'cores': threads,     # torch threads; read as \PolicyCores by p13
           'model': name, 'checkpoint': ckpt_rel, 'config': cfg_rel,
           'environment': REG.env(), 'mask': masks or 'legacy (no options)',
           'seed_option': args.seed, 'parameter_numel': n_param,
           **P75.thread_record(torch),
           'affinity': cores, 'cpu_count': os.cpu_count(), **P75.core_info(cores),
           'hostname': os.uname().nodename,
           'started': started, 'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
           'loadavg_start': la_start, 'loadavg_end': la_end,
           'idle_before': before, 'idle_after': after,
           'allow_busy': args.allow_busy,
           'cpu_seconds': cpu_all, 'wall_seconds': wall_all, 'cpu_share': share,
           'valid': valid,
           'per_regime': out, 'worst_regime_median_s': worst}
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, 'w') as f:
        json.dump(rec, f, indent=1)
    print(f'\nworst-regime median: {worst:.3f} s  cpu_share {share:.3f}  '
          f'{"valid" if valid else "INVALID"}  -> {dst}')
    if not valid:
        sys.exit('INVALID measurement: the cores were not idle or cpu_share is below '
                 f'{P75.SHARE_MIN} at one thread; repeat it on an idle core.')


if __name__ == '__main__':
    main()
