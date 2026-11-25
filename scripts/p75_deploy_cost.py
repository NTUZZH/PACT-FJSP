"""Deployment cost of the released policy across shop sizes, on CPU.

What a plant would pay to run the released policy as a dispatching controller:
wall-clock per instance, decisions per instance, time per decision, peak
resident memory, checkpoint size on disk and parameter count, from the
training size (10 modules) to the production batch (80 modules).

The rollout is the one p16_policy_latency.py times: batch of one, greedy
argmax, model already resident, the same config snapshot restore and the same
first-instance warm-up drop. Only the size loop, the memory and checkpoint
bookkeeping and the generated artifacts are new.

Threads. A batch-1 forward pass of this network (32,786 parameters) is too
little arithmetic to spread over threads, so the measurement uses ONE torch
thread on one otherwise idle core. Two separate settings, never one flag for
both:
  --torch_threads T  (default 1) torch.set_num_threads(T) and
                     torch.set_num_interop_threads(T), the first torch calls of
                     the measuring process;
  --numpy_threads K  (default 1) OMP_NUM_THREADS, MKL_NUM_THREADS and
                     OPENBLAS_NUM_THREADS of the measuring process, set before
                     numpy is imported (the environment code is numpy).
The earlier four-thread convention (results/deploy_cost_cpu_4core_pcore0to3.json)
is --torch_threads 4 --numpy_threads 4 under taskset -c 0-3.

Checkpoint. The released policy of the active environment, through
scripts/arm_registry.py: --seed S (default 301) or --seed select (the registry's
selection rule over the active seeds, PACT_SEEDS); legacy
10x25+family+jointblind-v1, exact 10x25+family+xmask-v1-s<seed>. --model NAME
names a checkpoint directly. A missing checkpoint or configuration is refused
with the file named. The checkpoint's training configuration must carry the
mask options of the active environment (absent = legacy), and the rollout
environment is built with them (exact: first_mask='exact', mask_ties='exact').

Idle cores. Before each size, the cores of the process's affinity mask are
sampled for IDLE_WINDOW seconds: a thread of another process that used more
than 5% of one of these cores, or a process whose lifetime CPU share (ps pcpu)
is above 5% and that last ran on one of them, stops the run before the
measurement starts, unless --allow_busy (tests only). The same sample is taken
after each size and stored. Each size's record also stores the measuring
process's CPU seconds over its timed rollouts, divided by their wall seconds
and the torch thread count (cpu_share); at one thread a share below 0.9 marks
the record invalid. Macros and table are written only when every size is valid;
an invalid run still writes its JSON record (with --allow_busy stated) and exits
non-zero.

Peak resident set size (ru_maxrss) is a high-water mark for the whole process,
so a second size measured in the same process would report the larger of the
two. Each size therefore runs in its own child process, and the child also
records the high-water mark just after the model is loaded, so the rollout's
own increment is visible.

Every record stores the torch thread counts, the numpy thread variables, the
affinity mask of the measuring process, the load average at its start and end,
the idle-core samples, the environment and the checkpoint. Outputs go through
arm_registry.out (outside the default setting under PACT_OUT):
results/deploy_cost_cpu_<T>thread<out_tag>.json, paper/macros_deploy.tex and
paper/tables/deploy_table.tex. An existing output is refused without --force.

CPU only. CUDA is hidden from the process before torch is imported.

Usage:
  taskset -c 20 python scripts/p75_deploy_cost.py
      [--size 10x25,20x25,30x25,50x25,80x25] [--n N] [--seed 301|select]
      [--model M] [--torch_threads 1] [--numpy_threads 1] [--out_tag TAG]
      [--force] [--allow_busy]
  python scripts/p75_deploy_cost.py --smoke          (prints only, writes nothing)
  python scripts/p75_deploy_cost.py --from_json results/deploy_cost_cpu_4core.json
                                                     (regenerates macros and table)
  Exact environment: PACT_ENV=exact PACT_OUT=<folder> before any of these.
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
NUMPY_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
SHELL_OMP = os.environ.get('OMP_NUM_THREADS')


def early_numpy_threads(argv):
    """Set the numpy thread variables from --numpy_threads (default 1) before
    numpy is first imported (arm_registry imports it). A value that is not a
    positive integer is left to the option parser to refuse."""
    k = '1'
    for i, a in enumerate(argv):
        if a == '--numpy_threads' and i + 1 < len(argv):
            k = argv[i + 1]
        elif a.startswith('--numpy_threads='):
            k = a.split('=', 1)[1]
    if k.isdigit() and int(k) >= 1:
        for v in NUMPY_VARS:
            os.environ[v] = k


if __name__ == '__main__':
    early_numpy_threads(sys.argv)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)
import arm_registry as REG  # noqa: E402  (pops --pact-* options; imports numpy)
_ARGV = sys.argv[1:]
sys.argv = sys.argv[:1]          # params.py parses sys.argv at import

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
BUSY_PCT = 5.0                  # a core is not idle above 5% of it
IDLE_WINDOW = 2.0               # seconds of the idle-core sample
SHARE_MIN = 0.9                 # cpu_share below this at one thread: invalid


# --------------------------------------------------------------------------
# helpers shared with p16_policy_latency.py
# --------------------------------------------------------------------------
def set_numpy_threads(k):
    """Thread variables of numpy's numerical libraries; effective only when
    set before numpy is imported in the process (or passed to a child)."""
    for v in NUMPY_VARS:
        os.environ[v] = str(int(k))


def set_torch_threads(torch, t):
    """Intra-op and inter-op thread counts of torch, before any torch work."""
    torch.set_num_threads(int(t))
    try:
        torch.set_num_interop_threads(int(t))
    except RuntimeError:
        # inter-op work has already started in this process; the record
        # stores the count torch reports
        pass


def thread_record(torch):
    return {'torch_num_threads': torch.get_num_threads(),
            'torch_num_interop_threads': torch.get_num_interop_threads(),
            **{v: os.environ.get(v) for v in NUMPY_VARS}}


def loadavg():
    with open('/proc/loadavg') as f:
        return [float(x) for x in f.read().split()[:3]]


def affinity():
    return sorted(os.sched_getaffinity(0))


def core_info(cores):
    """Processor model and the maximum frequency (MHz) of each given core: on
    a hybrid processor the core type changes a one-thread latency."""
    model = None
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                    break
    except OSError:
        pass
    mhz = {}
    for c in cores:
        try:
            with open(f'/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq') as f:
                mhz[c] = int(f.read().strip()) // 1000
        except (OSError, ValueError):
            mhz[c] = None
    return {'cpu_model': model, 'core_max_mhz': mhz}


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


def _own_pids():
    """This process and its ancestors (the launching shells)."""
    out, pid = set(), os.getpid()
    while pid > 1:
        out.add(pid)
        try:
            with open(f'/proc/{pid}/stat') as f:
                pid = int(f.read().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def _thread_times():
    """{'/proc/<pid>/task/<tid>': (utime + stime in ticks, last processor)}."""
    snap = {}
    for tdir in glob.glob('/proc/[0-9]*/task/[0-9]*'):
        try:
            with open(tdir + '/stat') as f:
                s = f.read()
        except OSError:
            continue
        fields = s[s.rfind(')') + 2:].split()
        snap[tdir] = (int(fields[11]) + int(fields[12]), int(fields[36]))
    return snap


def _cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().replace(b'\0', b' ').decode(errors='replace').strip()[:120]
    except OSError:
        return '?'


def _ps_rows():
    """(rows of `ps -eo pid,psr,pcpu,args`, pid of that ps process)."""
    ps = subprocess.Popen(['ps', '-eo', 'pid,psr,pcpu,args', '--no-headers'],
                          stdout=subprocess.PIPE, text=True)
    return ps.communicate()[0].splitlines(), ps.pid


def busy_processes(cores, window=IDLE_WINDOW):
    """Other processes on the given cores: threads that used more than
    BUSY_PCT of a core over the window and last ran on one of the cores, and
    processes whose lifetime CPU share (ps pcpu) is above BUSY_PCT and that
    last ran there. This process, its ancestors and the ps call itself are
    not counted."""
    cores, mine = set(cores), _own_pids()
    a = _thread_times()
    time.sleep(window)
    b = _thread_times()
    hz = os.sysconf('SC_CLK_TCK')
    busy = {}
    for k, (ticks, cpu) in b.items():
        pid = int(k.split('/')[2])
        if cpu not in cores or pid in mine:
            continue
        pct = 100.0 * (ticks - a.get(k, (ticks, cpu))[0]) / hz / window
        if pct > BUSY_PCT:
            busy[pid] = (f'pid {pid} thread {k.split("/")[4]} core {cpu} '
                         f'{pct:.0f}% over {window:g} s: {_cmdline(pid)}')
    rows, ps_pid = _ps_rows()
    for line in rows:
        f = line.split(None, 3)
        if len(f) < 4:
            continue
        pid, psr, pcpu = int(f[0]), int(f[1]), float(f[2])
        if pid == ps_pid or pid in mine or pid in busy:
            continue
        if pcpu > BUSY_PCT and psr in cores:
            busy[pid] = f'pid {pid} core {psr} {pcpu:.0f}% (lifetime): {f[3][:120]}'
    return sorted(busy.values())


def idle_check(cores, allow_busy, label, window=IDLE_WINDOW, record_only=False):
    """Sample the cores; stop unless they are idle or allow_busy is set
    (record_only: the sample after a measurement, which never stops).
    Returns the evidence stored in the record."""
    la = loadavg()
    busy = busy_processes(cores, window)
    per_core = core_busy(cores, window=1.0)
    print(f'[{label}] load average {la[0]:.2f} {la[1]:.2f} {la[2]:.2f} on '
          f'{os.cpu_count()} cores; affinity {cores}; core busy '
          + ', '.join(f'{c}:{b:.0%}' for c, b in per_core.items())
          + f'; other processes above {BUSY_PCT:g}%: {len(busy)}', flush=True)
    for x in busy[:10]:
        print(f'    {x}', flush=True)
    if busy and record_only:
        print(f'[{label}] the cores were not idle at the end of the measurement; '
              f'the record is marked invalid', flush=True)
    elif busy and not allow_busy:
        sys.exit(f'REFUSING to measure: {len(busy)} other process(es) use more than '
                 f'{BUSY_PCT:g}% of a core in the affinity mask {cores}. A timing '
                 f'under contention is not a measurement. Free the cores or pin '
                 f'the run elsewhere (--allow_busy only for tests).')
    elif busy:
        print(f'[{label}] cores not idle, measuring anyway (--allow_busy or smoke); '
              f'the record is marked invalid', flush=True)
    return {'loadavg': la, 'busy_processes': busy, 'core_busy': per_core,
            'idle': not busy, 'window_s': window}


def resolve_checkpoint(model=None, seed='301'):
    """(model name, checkpoint path, config path) of the released policy in the
    active environment (arm_registry), or of an explicitly named model. Refuses
    with the missing file named."""
    if model:
        name = model
    else:
        try:
            s = (REG.select_seed('released')[0] if str(seed) == 'select'
                 else int(seed))
            name = REG.PREFIX + REG.model('released', None, s)
        except (REG.MissingResult, ValueError) as e:
            sys.exit(f'REFUSING ({REG.describe()}): {e}')
    ckpt = f'trained_network/FAMILY/{name}.pth'
    cfg = f'train_log/FAMILY/config_{name}.json'
    missing = [p for p in (ckpt, cfg) if not os.path.isfile(os.path.join(ROOT, p))]
    if missing:
        sys.exit(f'REFUSING ({REG.describe()}): the checkpoint of {name} is not on '
                 'disk: missing ' + ', '.join(missing))
    return name, ckpt, cfg


def mask_options(snap, name):
    """FJSPEnvFamily options of the active environment, checked against the
    mask options in the checkpoint's training configuration `snap` (absent =
    legacy). legacy: {} (the stored rollout); exact: first_mask and mask_ties
    'exact' through resolve_mask_option, which refuses a contradicting
    variable. Import after configs.device is set."""
    from constraint_family.fjsp_env_family import (mask_option_from_env,
                                                    resolve_mask_option)
    env = REG.env()
    if env == 'legacy':
        asked = {n: mask_option_from_env(n) for n in ('first_mask', 'mask_ties')}
        if 'exact' in asked.values():
            sys.exit(f'REFUSING: the legacy environment is active but the mask '
                     f'variables ask for {asked}; unset PACT_FIRST_MASK and '
                     f'PACT_MASK_TOL, or set PACT_ENV=exact.')
        opts, want = {}, {'first_mask': 'legacy', 'mask_ties': 'legacy'}
    else:
        flags = REG.MASK_OPTIONS[env]
        opts = {f.lstrip('-'): resolve_mask_option(f.lstrip('-'), v)
                for f, v in zip(flags[0::2], flags[1::2])}
        want = dict(opts)
    trained = {k: (snap.get(k) or 'legacy') for k in want}
    if trained != want:
        sys.exit(f'REFUSING: {name} was trained with {trained}, the {env} '
                 f'environment rolls out with {want}')
    return opts


# --------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------
def parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--size', default=DEFAULT_SIZES)
    p.add_argument('--n', type=int, default=None,
                   help='measured instances per regime (default 30 below 50 '
                        'modules, 10 from 50 modules); one more is run and '
                        'dropped as warm-up')
    p.add_argument('--model', default=None,
                   help='checkpoint name; default the released policy of the '
                        'active environment (arm_registry)')
    p.add_argument('--seed', default='301',
                   help="training seed of the released policy, or 'select'")
    p.add_argument('--torch_threads', type=int, default=1,
                   help='torch intra-op and inter-op threads (default 1)')
    p.add_argument('--numpy_threads', type=int, default=1,
                   help='OMP/MKL/OpenBLAS threads of the numpy environment code '
                        '(default 1)')
    p.add_argument('--out_tag', default='')
    p.add_argument('--force', action='store_true',
                   help='allow replacing an existing deploy_cost output')
    p.add_argument('--allow_busy', action='store_true',
                   help='measure although the cores are not idle (tests only)')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--from_json', default=None)
    # internal: one size per child process
    p.add_argument('--_child', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_regimes', default=None, help=argparse.SUPPRESS)
    p.add_argument('--_take', type=int, default=None, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    if a.torch_threads < 1 or a.numpy_threads < 1:
        p.error('thread counts must be at least 1')
    return a


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


def maxrss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KiB on Linux


# --------------------------------------------------------------------------
# child: one size, one process
# --------------------------------------------------------------------------
def run_child(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''      # never touch the GPU
    set_numpy_threads(args.numpy_threads)        # before numpy is imported
    la_start = loadavg()
    rss_start = maxrss_mb()
    import torch
    set_torch_threads(torch, args.torch_threads)  # the first torch calls
    size = args._child
    regimes = args._regimes.split(',')
    take = args._take
    name, ckpt_rel, cfg_rel = resolve_checkpoint(args.model, args.seed)

    from params import configs
    with open(os.path.join(ROOT, cfg_rel)) as f:
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
            sys.exit(f'{name}: snapshot sets {k}={v!r}; this rollout uses '
                     f'the published default {dflt!r}. Time it with '
                     f'eval_family.py conventions instead.')
    configs.device = 'cpu'       # must precede the env import (class attribute)

    from model.PPO import PPO_initialize
    from common_utils import greedy_select_action
    from constraint_family.descriptor import ConstraintDescriptor
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from ppvc_instance_generator import load_instance
    masks = mask_options(snap, name)

    ckpt = os.path.join(ROOT, ckpt_rel)
    ppo = PPO_initialize()
    sd = torch.load(ckpt, map_location=torch.device('cpu'))
    ppo.policy.load_state_dict(sd)
    ppo.policy.eval()
    n_state = int(sum(t.numel() for t in sd.values()))
    n_param = int(sum(p.numel() for p in ppo.policy.parameters()))
    rss_model = maxrss_mb()

    per_regime, smoke_rows = {}, []
    cpu_all, wall_all = 0.0, 0.0
    for regime in regimes:
        sel, n_avail = select_instances(size, regime, take)
        lat, dec, msd = [], [], []
        cpu_r, wall_r = 0.0, 0.0
        for k, (dp, stem) in enumerate(sel):
            jl, pt, meta = load_instance(stem)
            with open(dp) as f:
                desc = ConstraintDescriptor.from_json(json.load(f),
                                                      setup_convention='attached')
            env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True, **masks)
            state = env.set_initial_data([jl], [pt], descriptor_list=[desc],
                                         op_type_list=[meta['op_type']],
                                         mch_type_list=[meta['mch_type']])
            n_ops = int(pt.shape[0])
            steps = 0
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
                steps += 1
            dt = time.perf_counter() - t0
            dc = time.process_time() - c0
            if steps != n_ops:
                sys.exit(f'{stem}: {steps} decisions for {n_ops} operations')
            name_i = os.path.basename(stem)
            if args.smoke:
                tag = 'warm-up, dropped' if (k == 0 and len(sel) > 1) else (
                    'only instance, includes first-call costs'
                    if len(sel) == 1 else 'measured')
                print(f'  {size} {regime:4s} {name_i}: {dt:7.3f} s  '
                      f'{steps} decisions  {dt * 1000 / steps:.2f} ms/decision'
                      f'  ({tag})', flush=True)
                smoke_rows.append({'regime': regime, 'instance': name_i,
                                   'seconds': dt, 'decisions': steps,
                                   'warmup': k == 0 and len(sel) > 1})
            if k > 0 or len(sel) == 1:  # drop the first: it pays lazy-init costs
                lat.append(dt)
                dec.append(steps)
                msd.append(dt * 1000.0 / steps)
                cpu_r += dc
                wall_r += dt
        lat.sort()
        m = len(lat)
        cpu_all += cpu_r
        wall_all += wall_r
        per_regime[regime] = {
            'median_s': statistics.median(lat),
            'p05_s': lat[int(.05 * m)], 'p95_s': lat[-1 - int(.05 * m)],
            'min_s': lat[0], 'max_s': lat[-1], 'n': m,
            'descriptors_available': n_avail,
            'decisions_median': statistics.median(dec),
            'decisions_min': min(dec), 'decisions_max': max(dec),
            'ms_per_decision_median': statistics.median(msd),
            'cpu_share': cpu_r / (wall_r * torch.get_num_threads())}
        r = per_regime[regime]
        if not args.smoke:
            print(f'  {size} {regime:4s} n={m:3d}  median {r["median_s"]:7.3f} s'
                  f'  p05-p95 {r["p05_s"]:.3f}-{r["p95_s"]:.3f} s  '
                  f'{r["decisions_median"]:.0f} decisions  '
                  f'{r["ms_per_decision_median"]:.2f} ms/decision  '
                  f'cpu_share {r["cpu_share"]:.3f}', flush=True)

    rss_peak = maxrss_mb()
    rec = {'size': size, 'modules': n_modules(size), 'per_regime': per_regime,
           'peak_rss_mb': rss_peak, 'rss_after_import_mb': rss_start,
           'rss_after_model_load_mb': rss_model,
           'rollout_rss_increment_mb': rss_peak - rss_model,
           'state_dict_numel': n_state, 'parameter_numel': n_param,
           **thread_record(torch),
           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'affinity': affinity(), 'loadavg_start': la_start,
           'loadavg_end': loadavg(), 'model': name, 'checkpoint': ckpt_rel,
           'environment': REG.env(), 'mask': masks or 'legacy (no options)',
           'cpu_seconds': cpu_all, 'wall_seconds': wall_all,
           'cpu_share': cpu_all / (wall_all * torch.get_num_threads()),
           'smoke_rows': smoke_rows}
    print(MARK + json.dumps(rec), flush=True)


# --------------------------------------------------------------------------
# parent: one child per size, then the artifacts
# --------------------------------------------------------------------------
def spawn(args, size, regimes, take, model):
    cmd = [sys.executable, os.path.abspath(__file__), '--_child', size,
           '--_regimes', ','.join(regimes), '--_take', str(take),
           '--model', model, '--torch_threads', str(args.torch_threads),
           '--numpy_threads', str(args.numpy_threads)]
    if args.smoke:
        cmd.append('--smoke')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
               **{v: str(args.numpy_threads) for v in NUMPY_VARS})
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


def out_paths(json_name=None):
    """Outputs through arm_registry.out (refuses outside the default setting
    unless PACT_OUT is set): (json or None, macros, table)."""
    return (os.path.join(ROOT, REG.out(f'results/{json_name}')) if json_name else None,
            os.path.join(ROOT, REG.out('paper/macros_deploy.tex')),
            os.path.join(ROOT, REG.out('paper/tables/deploy_table.tex')))


def write_artifacts(rec, src, force):
    _, mac_path, tab_path = out_paths()
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
    guarded_write(mac_path, '\n'.join(mac) + '\n', force)

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
    os.makedirs(os.path.dirname(tab_path), exist_ok=True)
    guarded_write(tab_path, '\n'.join(tab) + '\n', force)


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

    REG.banner()
    name, ckpt_rel, cfg_rel = resolve_checkpoint(args.model, args.seed)
    cores = affinity()
    if SHELL_OMP not in (None, str(args.numpy_threads)):
        print(f'note: OMP_NUM_THREADS={SHELL_OMP} in the environment; the measuring '
              f'process uses --numpy_threads {args.numpy_threads}', flush=True)
    if args.torch_threads > len(cores):
        print(f'WARNING: {args.torch_threads} torch threads on {len(cores)} '
              f'pinned cores.', flush=True)
    print(f'model {name} ({REG.env()} environment); torch threads '
          f'{args.torch_threads}; numpy threads {args.numpy_threads}; '
          f'affinity {cores}', flush=True)

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
        dst, mac_path, tab_path = out_paths(
            f'deploy_cost_cpu_{args.torch_threads}thread{args.out_tag}.json')
        for path in (dst, mac_path, tab_path):
            if os.path.exists(path) and not args.force:
                sys.exit(f'{path} exists; pass --force to replace it')

    ckpt = os.path.join(ROOT, ckpt_rel)
    rec = {'device': 'cpu', 'model': name, 'checkpoint': ckpt_rel,
           'config': cfg_rel, 'environment': REG.env(), 'seed_option': args.seed,
           'checkpoint_bytes': os.path.getsize(ckpt),
           'torch_threads_option': args.torch_threads,
           'numpy_threads_option': args.numpy_threads,
           'OMP_NUM_THREADS': str(args.numpy_threads),
           'MKL_NUM_THREADS': str(args.numpy_threads),
           'affinity': cores, 'cpu_count': os.cpu_count(), **core_info(cores),
           'hostname': os.uname().nodename,
           'started': time.strftime('%Y-%m-%d %H:%M:%S'),
           'loadavg_start': loadavg(), 'allow_busy': args.allow_busy,
           'sizes': {}, 'contention': {}}
    t_all = time.perf_counter()
    valid = True
    for size, regimes, take in plan:
        print(f'\n== {size}', flush=True)
        # the smoke test writes nothing, so a busy core only warns there
        before = idle_check(cores, args.allow_busy or args.smoke, f'{size} before')
        t_size = time.perf_counter()
        r = spawn(args, size, regimes, take, name)
        r['wall_s'] = time.perf_counter() - t_size
        after = idle_check(cores, True, f'{size} after', record_only=True)
        rec['contention'][size] = {'before': before, 'after': after}
        rec['sizes'][size] = r
        ok = before['idle'] and after['idle'] and (
            r['torch_num_threads'] != 1 or r['cpu_share'] >= SHARE_MIN)
        r['valid'] = ok
        valid = valid and ok
        reg, wr = worst_regime(r)
        spread = max(v['p95_s'] / v['p05_s'] for v in r['per_regime'].values())
        print(f'{size}: worst-regime median {wr["median_s"]:.3f} s ({reg}), '
              f'{wr["ms_per_decision_median"]:.2f} ms/decision, peak RSS '
              f'{r["peak_rss_mb"]:.0f} MB (model loaded '
              f'{r["rss_after_model_load_mb"]:.0f} MB), cpu_share '
              f'{r["cpu_share"]:.3f}{"" if ok else "  INVALID"}', flush=True)
        if not args.smoke and spread > 1.5:
            print(f'WARNING: p95/p05 reaches {spread:.2f} within a regime at '
                  f'{size}; suspect the machine before the code.', flush=True)
    rec['elapsed_s'] = time.perf_counter() - t_all
    rec['loadavg_end'] = loadavg()
    rec['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    rec['valid'] = valid

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
    if not valid:
        sys.exit(f'INVALID measurement (busy cores or cpu_share below {SHARE_MIN} '
                 f'at one thread); {dst} records it, macros and table not written')
    write_artifacts(rec, os.path.relpath(dst, ROOT), args.force)


if __name__ == '__main__':
    main()
