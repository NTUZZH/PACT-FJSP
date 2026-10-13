"""Deployment cost of the released policy across shop sizes, on CPU.

What a plant would pay to run the released policy as a dispatching controller:
wall-clock per instance, decisions per instance, time per decision, peak
resident memory, checkpoint size on disk and parameter count, from the
training size (10 modules) to the production batch (80 modules).

The rollout is the one p16_policy_latency.py times for the published 0.49 s
figure: batch of one, greedy argmax, model already resident, the same config
snapshot restore and the same first-instance warm-up drop. Only the size loop,
the memory and checkpoint bookkeeping and the generated artifacts are new.

Threads follow the published convention: OMP_NUM_THREADS=4 on four pinned
cores, with torch.set_num_threads called explicitly from that variable so the
new numbers sit on the same footing as results/policy_latency_cpu_4core_blind.json.

Peak resident set size (ru_maxrss) is a high-water mark for the whole process,
so a second size measured in the same process would report the larger of the
two. Each size therefore runs in its own child process, and the child also
records the high-water mark just after the model is loaded, so the rollout's
own increment is visible.

CPU only. CUDA is hidden from the process before torch is imported.

Usage:
  taskset -c 0-3 env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
      python scripts/p75_deploy_cost.py [--size 10x25,20x25,30x25,50x25,80x25]
      [--n N] [--model M] [--torch_threads T] [--out_tag TAG] [--force]
  python scripts/p75_deploy_cost.py --smoke          (prints only, writes nothing)
  python scripts/p75_deploy_cost.py --from_json results/deploy_cost_cpu_4core.json
                                                     (regenerates macros and table)
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]          # params.py parses sys.argv at import

MODEL = '10x25+family+jointblind-v1'
REGIMES = ['N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW']
DEFAULT_SIZES = '10x25,20x25,30x25,50x25,80x25'
# Same restore list as p16_policy_latency.py.
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads')
# p16 builds the env with the published defaults (non-delay mask, no extra pair
# channels, min envelope). eval_family.py restores these from the snapshot
# instead; for a model trained under the defaults the two rollouts coincide.
# A snapshot that departs from them would be timed on the wrong action space,
# so it is refused here.
ENV_DEFAULTS = {'mask_mode': 'nondelay', 'pair_extra': 'none',
                'delta_mode': None, 'reward_delta_env': True}
SIZE_WORD = {10: 'Ten', 20: 'Twenty', 30: 'Thirty', 40: 'Forty', 50: 'Fifty',
             60: 'Sixty', 70: 'Seventy', 80: 'Eighty', 100: 'Hundred'}
MARK = '@@P75RESULT '
MIB = 1024.0 * 1024.0           # MB in every output below means 2**20 bytes


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--size', default=DEFAULT_SIZES)
    p.add_argument('--n', type=int, default=None,
                   help='measured instances per regime (default 30 below 50 '
                        'modules, 10 from 50 modules); one more is run and '
                        'dropped as warm-up')
    p.add_argument('--model', default=MODEL)
    p.add_argument('--torch_threads', type=int, default=None,
                   help='override the thread count read from OMP_NUM_THREADS')
    p.add_argument('--out_tag', default='')
    p.add_argument('--force', action='store_true',
                   help='allow replacing an existing deploy_cost output')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--from_json', default=None)
    # internal: one size per child process
    p.add_argument('--_child', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_regimes', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_take', type=int, default=None, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def n_modules(size):
    return int(size.split('x')[0])


def default_n(size):
    return 10 if n_modules(size) >= 50 else 30


def paths_for(size, regime):
    return (os.path.join(ROOT, 'data', 'FAMILY', size, 'desc', regime),
            os.path.join(ROOT, 'data', 'PPVC', f'{size}+ppvc-mixed'))


def select_instances(size, regime, take):
    """First `take` descriptors of a regime, each checked against its instance."""
    ddir, base = paths_for(size, regime)
    descs = sorted(glob.glob(os.path.join(ddir, '*.desc.json')))
    if not descs:
        sys.exit(f'no descriptors under {ddir}')
    out = []
    for dp in descs[:take]:
        name = os.path.basename(dp).replace('.desc.json', '')
        stem = os.path.join(base, name)
        for ext in ('.fjs', '.meta.json'):
            if not os.path.isfile(stem + ext):
                sys.exit(f'instance file missing: {stem + ext}')
        out.append((dp, stem))
    return out, len(descs)


def thread_count(args):
    omp = os.environ.get('OMP_NUM_THREADS')
    if args.torch_threads is not None:
        return args.torch_threads, omp
    if omp is None:
        print('WARNING: OMP_NUM_THREADS is unset. The published 0.49 s was '
              'measured with OMP_NUM_THREADS=4 on four pinned cores; set it '
              'or pass --torch_threads.', file=sys.stderr)
        return None, omp
    return int(omp), omp


def loadavg():
    with open('/proc/loadavg') as f:
        return [float(x) for x in f.read().split()[:3]]


def core_busy(cores, window=1.0):
    """Busy fraction of each given core over `window` seconds, from /proc/stat."""
    def snap():
        rows = {}
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and line[3].isdigit():
                    parts = line.split()
                    v = [int(x) for x in parts[1:]]
                    idle = v[3] + (v[4] if len(v) > 4 else 0)
                    rows[int(parts[0][3:])] = (sum(v), idle)
        return rows
    a = snap()
    time.sleep(window)
    b = snap()
    out = {}
    for c in cores:
        dt = b[c][0] - a[c][0]
        out[c] = round(1.0 - (b[c][1] - a[c][1]) / dt, 3) if dt else 0.0
    return out


def maxrss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KiB on Linux


# --------------------------------------------------------------------------
# child: one size, one process
# --------------------------------------------------------------------------
def run_child(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''      # never touch the GPU
    size = args._child
    regimes = args._regimes.split(',')
    take = args._take
    threads, omp = thread_count(args)
    rss_start = maxrss_mb()

    from params import configs
    with open(os.path.join(ROOT, 'train_log', 'FAMILY',
                           f'config_{args.model}.json')) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            v = snap[k]
            setattr(configs, k, (v == 'True') if isinstance(v, str) and
                    v in ('True', 'False') else v)
    for k, dflt in ENV_DEFAULTS.items():
        v = snap.get(k, dflt)
        if isinstance(v, str) and v in ('True', 'False'):
            v = v == 'True'
        if v != dflt:
            sys.exit(f'{args.model}: snapshot sets {k}={v!r}; this rollout uses '
                     f'the published default {dflt!r}. Time it with '
                     f'eval_family.py conventions instead.')
    configs.device = 'cpu'       # must precede the env import (class attribute)

    import torch
    if threads is not None:
        torch.set_num_threads(threads)
    from model.PPO import PPO_initialize
    from common_utils import greedy_select_action
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from ppvc_instance_generator import load_instance

    ckpt = os.path.join(ROOT, 'trained_network', 'FAMILY', f'{args.model}.pth')
    ppo = PPO_initialize()
    sd = torch.load(ckpt, map_location=torch.device('cpu'))
    ppo.policy.load_state_dict(sd)
    ppo.policy.eval()
    n_state = int(sum(t.numel() for t in sd.values()))
    n_param = int(sum(p.numel() for p in ppo.policy.parameters()))
    rss_model = maxrss_mb()

    per_regime, smoke_rows = {}, []
    for regime in regimes:
        sel, n_avail = select_instances(size, regime, take)
        lat, dec, msd = [], [], []
        for k, (dp, stem) in enumerate(sel):
            jl, pt, meta = load_instance(stem)
            with open(dp) as f:
                desc = ConstraintDescriptor.from_json(json.load(f))
            env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
            state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                         op_type_list=[meta['op_type']],
                                         mch_type_list=[meta['mch_type']])
            n_ops = int(pt.shape[0])
            steps = 0
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
                steps += 1
            dt = time.perf_counter() - t0
            if steps != n_ops:
                sys.exit(f'{stem}: {steps} decisions for {n_ops} operations')
            name = os.path.basename(stem)
            if args.smoke:
                tag = 'warm-up, dropped' if (k == 0 and len(sel) > 1) else (
                    'only instance, includes first-call costs'
                    if len(sel) == 1 else 'measured')
                print(f'  {size} {regime:4s} {name}: {dt:7.3f} s  '
                      f'{steps} decisions  {dt * 1000 / steps:.2f} ms/decision'
                      f'  ({tag})', flush=True)
                smoke_rows.append({'regime': regime, 'instance': name,
                                   'seconds': dt, 'decisions': steps,
                                   'warmup': k == 0 and len(sel) > 1})
            if k > 0 or len(sel) == 1:  # drop the first: it pays lazy-init costs
                lat.append(dt)
                dec.append(steps)
                msd.append(dt * 1000.0 / steps)
        lat.sort()
        m = len(lat)
        per_regime[regime] = {
            'median_s': statistics.median(lat),
            'p05_s': lat[int(.05 * m)], 'p95_s': lat[-1 - int(.05 * m)],
            'min_s': lat[0], 'max_s': lat[-1], 'n': m,
            'descriptors_available': n_avail,
            'decisions_median': statistics.median(dec),
            'decisions_min': min(dec), 'decisions_max': max(dec),
            'ms_per_decision_median': statistics.median(msd)}
        r = per_regime[regime]
        if not args.smoke:
            print(f'  {size} {regime:4s} n={m:3d}  median {r["median_s"]:7.3f} s'
                  f'  p05-p95 {r["p05_s"]:.3f}-{r["p95_s"]:.3f} s  '
                  f'{r["decisions_median"]:.0f} decisions  '
                  f'{r["ms_per_decision_median"]:.2f} ms/decision', flush=True)

    rss_peak = maxrss_mb()
    rec = {'size': size, 'modules': n_modules(size), 'per_regime': per_regime,
           'peak_rss_mb': rss_peak, 'rss_after_import_mb': rss_start,
           'rss_after_model_load_mb': rss_model,
           'rollout_rss_increment_mb': rss_peak - rss_model,
           'state_dict_numel': n_state, 'parameter_numel': n_param,
           'torch_num_threads': torch.get_num_threads(),
           'torch_num_interop_threads': torch.get_num_interop_threads(),
           'OMP_NUM_THREADS': omp,
           'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'smoke_rows': smoke_rows}
    print(MARK + json.dumps(rec), flush=True)


# --------------------------------------------------------------------------
# parent: one child per size, then the artifacts
# --------------------------------------------------------------------------
def spawn(args, size, regimes, take):
    cmd = [sys.executable, os.path.abspath(__file__), '--_child', size,
           '--_regimes', ','.join(regimes), '--_take', str(take),
           '--model', args.model]
    if args.torch_threads is not None:
        cmd += ['--torch_threads', str(args.torch_threads)]
    if args.smoke:
        cmd.append('--smoke')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                            text=True)
    rec = None
    for line in proc.stdout:
        if line.startswith(MARK):
            rec = json.loads(line[len(MARK):])
        else:
            print(line, end='', flush=True)
    if proc.wait() != 0 or rec is None:
        sys.exit(f'child for {size} failed (exit {proc.returncode})')
    return rec


def contention_check(cores):
    la = loadavg()
    busy = core_busy(cores)
    hot = {c: b for c, b in busy.items() if b > 0.10}
    print(f'load average {la[0]:.2f} {la[1]:.2f} {la[2]:.2f} on '
          f'{os.cpu_count()} cores; pinned cores busy: '
          + ', '.join(f'{c}:{b:.0%}' for c, b in busy.items()), flush=True)
    if hot:
        print(f'WARNING: pinned cores {sorted(hot)} are already busy. A '
              'measurement under contention is not a measurement.', flush=True)
    return {'loadavg': la, 'pinned_core_busy': busy}


def worst_regime(rec):
    return max(rec['per_regime'].items(), key=lambda kv: kv[1]['median_s'])


def size_word(modules):
    if modules not in SIZE_WORD:
        sys.exit(f'no macro word for {modules} modules; extend SIZE_WORD')
    return SIZE_WORD[modules]


def guarded_write(path, text, force):
    if os.path.exists(path) and not force:
        sys.exit(f'{path} exists; pass --force to replace it')
    with open(path, 'w') as f:
        f.write(text)
    print(f'wrote {path}')


def write_artifacts(rec, src, force):
    sizes = sorted(rec['sizes'].values(), key=lambda r: r['modules'])
    mac = ['% auto-generated by scripts/p75_deploy_cost.py -- do not hand-edit',
           f'% source: {src}',
           f'% model {rec["model"]}, CPU, torch threads '
           f'{rec["torch_num_threads"]} (OMP_NUM_THREADS={rec["OMP_NUM_THREADS"]}),'
           ' batch of one, greedy decoding, first instance per regime dropped.',
           '% latency and ms per decision: median over instances in the regime '
           'with the largest median latency. MB = 2^20 bytes.',
           '% peak RSS: ru_maxrss of a fresh process holding the model and '
           'rolling out that size only.']
    for r in sizes:
        w = size_word(r['modules'])
        reg, wr = worst_regime(r)
        mac.append(f'\\newcommand{{\\DeployLatency{w}}}{{{wr["median_s"]:.2f}}}'
                   f'  % s, worst regime {reg}')
        mac.append(f'\\newcommand{{\\DeployMsPerDecision{w}}}'
                   f'{{{wr["ms_per_decision_median"]:.1f}}}')
        mac.append(f'\\newcommand{{\\DeployPeakRssMB{w}}}{{{r["peak_rss_mb"]:.0f}}}')
    mac.append(f'\\newcommand{{\\DeployCheckpointMB}}'
               f'{{{rec["checkpoint_bytes"] / MIB:.1f}}}')
    mac.append(f'\\newcommand{{\\DeployCheckpointKB}}'
               f'{{{rec["checkpoint_bytes"] / 1024.0:.0f}}}  % KiB')
    mac.append(f'\\newcommand{{\\DeployParams}}{{{rec["state_dict_numel"]}}}'
               '  % numel summed over the checkpoint state dict')
    guarded_write(os.path.join(ROOT, 'paper', 'macros_deploy.tex'),
                  '\n'.join(mac) + '\n', force)

    tab = ['% auto-generated by scripts/p75_deploy_cost.py -- do not hand-edit',
           f'% source: {src}',
           f'% CPU, {rec["torch_num_threads"]} torch threads, batch of one, '
           'greedy decoding; latency and ms per decision from the regime with '
           'the largest median latency.',
           '% peak RSS: fresh process per size. tabular only: caption and label '
           'live in the manuscript.',
           '\\begin{tabular}{rrrrr}',
           '\\toprule',
           'modules & decisions per instance & median latency (s) & '
           'ms per decision & peak RSS (MB) \\\\',
           '\\midrule']
    for r in sizes:
        reg, wr = worst_regime(r)
        tab.append(f'{r["modules"]} & {wr["decisions_median"]:.0f} & '
                   f'{wr["median_s"]:.2f} & {wr["ms_per_decision_median"]:.1f} & '
                   f'{r["peak_rss_mb"]:.0f} \\\\')
    tab += ['\\bottomrule', '\\end{tabular}']
    os.makedirs(os.path.join(ROOT, 'paper', 'tables'), exist_ok=True)
    guarded_write(os.path.join(ROOT, 'paper', 'tables', 'deploy_table.tex'),
                  '\n'.join(tab) + '\n', force)


def project_wallclock(smoke):
    """Scale the smoke's per-instance times to the default full run."""
    a = smoke.get('10x25')
    b = smoke.get('50x25')
    if not a or not b:
        return
    ta = [x['seconds'] for x in a['smoke_rows'] if not x['warmup']]
    tb = [x['seconds'] for x in b['smoke_rows']]
    if not ta or not tb:
        return
    import math
    ta, tb = statistics.median(ta), statistics.median(tb)
    expo = math.log(tb / ta) / math.log(50 / 10)
    total = 0.0
    for size in DEFAULT_SIZES.split(','):
        n = n_modules(size)
        avail = len(glob.glob(os.path.join(paths_for(size, 'N')[0],
                                           '*.desc.json')))
        runs = min(default_n(size) + 1, avail) * len(REGIMES)
        t = ta * (n / 10) ** expo
        total += runs * t
        print(f'  {size}: ~{t:6.2f} s/instance x {runs} instances')
    print(f'projected default run: ~{total / 60:.0f} min of rollouts '
          f'(per-instance time grows as modules^{expo:.2f} between 10 and 50; '
          '80 modules is extrapolated)')


def main():
    args = parse(_ARGV)
    if args._child:
        return run_child(args)

    if args.from_json:
        with open(args.from_json) as f:
            rec = json.load(f)
        return write_artifacts(rec, os.path.relpath(args.from_json, ROOT),
                               args.force)

    threads, omp = thread_count(args)
    cores = sorted(os.sched_getaffinity(0))
    if threads is not None and threads > len(cores):
        print(f'WARNING: {threads} threads on {len(cores)} pinned cores.',
              flush=True)
    print(f'model {args.model}; OMP_NUM_THREADS={omp}; torch threads '
          f'{threads if threads is not None else "torch default"}; '
          f'affinity {cores}', flush=True)
    print('Run on an otherwise idle machine: a measurement under contention is '
          'not a measurement.', flush=True)

    if args.smoke:
        plan = [('10x25', ['LSW'], 2), ('50x25', ['LSW'], 1)]
    else:
        plan = []
        for size in [s.strip() for s in args.size.split(',') if s.strip()]:
            n = args.n if args.n is not None else default_n(size)
            plan.append((size, REGIMES, n + 1))

    # verify every descriptor and instance before any timing starts
    for size, regimes, take in plan:
        for regime in regimes:
            sel, avail = select_instances(size, regime, take)
            if len(sel) < take:
                print(f'note: {size}/{regime} has {avail} descriptors; '
                      f'{len(sel) - 1} measured instead of {take - 1}',
                      flush=True)

    # refuse an existing output now, not after an hour of measurement
    dst = None
    if not args.smoke:
        tag = threads if threads is not None else 'default'
        dst = os.path.join(ROOT, 'results',
                           f'deploy_cost_cpu_{tag}core{args.out_tag}.json')
        for path in (dst, os.path.join(ROOT, 'paper', 'macros_deploy.tex'),
                     os.path.join(ROOT, 'paper', 'tables', 'deploy_table.tex')):
            if os.path.exists(path) and not args.force:
                sys.exit(f'{path} exists; pass --force to replace it')

    ckpt = os.path.join(ROOT, 'trained_network', 'FAMILY', f'{args.model}.pth')
    rec = {'device': 'cpu', 'model': args.model,
           'checkpoint': os.path.relpath(ckpt, ROOT),
           'checkpoint_bytes': os.path.getsize(ckpt),
           'OMP_NUM_THREADS': omp,
           'MKL_NUM_THREADS': os.environ.get('MKL_NUM_THREADS'),
           'affinity': cores, 'cpu_count': os.cpu_count(),
           'hostname': os.uname().nodename,
           'started': time.strftime('%Y-%m-%d %H:%M:%S'),
           'sizes': {}, 'contention': {}}
    t_all = time.perf_counter()
    for size, regimes, take in plan:
        print(f'\n== {size}', flush=True)
        before = contention_check(cores)
        t_size = time.perf_counter()
        r = spawn(args, size, regimes, take)
        r['wall_s'] = time.perf_counter() - t_size
        after = {'loadavg': loadavg()}
        rec['contention'][size] = {'before': before, 'after': after}
        rec['sizes'][size] = r
        reg, wr = worst_regime(r)
        spread = max(v['p95_s'] / v['p05_s'] for v in r['per_regime'].values())
        print(f'{size}: worst-regime median {wr["median_s"]:.3f} s ({reg}), '
              f'{wr["ms_per_decision_median"]:.2f} ms/decision, peak RSS '
              f'{r["peak_rss_mb"]:.0f} MB (model loaded '
              f'{r["rss_after_model_load_mb"]:.0f} MB)', flush=True)
        if not args.smoke and spread > 1.5:
            print(f'WARNING: p95/p05 reaches {spread:.2f} within a regime at '
                  f'{size}; suspect the machine before the code.', flush=True)
    rec['elapsed_s'] = time.perf_counter() - t_all

    first = next(iter(rec['sizes'].values()))
    for k in ('torch_num_threads', 'torch_num_interop_threads',
              'state_dict_numel', 'parameter_numel'):
        rec[k] = first[k]
    print(f'\ncheckpoint {rec["checkpoint_bytes"]} bytes '
          f'({rec["checkpoint_bytes"] / MIB:.2f} MB); state-dict numel '
          f'{rec["state_dict_numel"]}, parameter numel {rec["parameter_numel"]}; '
          f'torch threads {rec["torch_num_threads"]}; elapsed '
          f'{rec["elapsed_s"]:.0f} s', flush=True)

    if args.smoke:
        project_wallclock(rec['sizes'])
        print('smoke: nothing written')
        return

    if os.path.basename(dst).startswith('policy_latency'):
        sys.exit('refusing to write over a policy_latency result')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    guarded_write(dst, json.dumps(rec, indent=1) + '\n', args.force)
    write_artifacts(rec, os.path.relpath(dst, ROOT), args.force)


if __name__ == '__main__':
    main()
