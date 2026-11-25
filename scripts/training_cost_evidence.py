"""One training run with the GPU and its cores to itself, and the record that proves it.

scripts/run_training_cost.sh calls this module for plan, sample and run; only rejudge is
started by hand. A measurement repeats one finished run of the exact-mask campaign under
a new name with the same resolved configuration, and writes one evidence record:

  campaign run (arm_registry)      new name                    attempt k = 1, 2, ...
  xmask-v1-s<seed>                 xcost-v1-s<seed>-r<k>
  xmask-solo-<R>-s<seed>           xcost-solo-<R>-s<seed>-r<k>

Files of the new run, all under train_log/training_cost/:
  <name>.log           the trainer's output (outside the train_log/FAMILY_*.log glob)
  <name>.samples.csv   one row per sample of the GPU and of the reserved cores
  <name>.json          the evidence record (kind 'training_cost')
  <name>.first.json    the record as the run wrote it, kept by rejudge (no reader takes it)
  <name>/              the trainer's checkpoint, configuration, validation and reward
                       files, moved out of trained_network/FAMILY and train_log/FAMILY
                       once the run has ended, so that no glob over those folders sees them

The record is always written once the trainer has started. It is valid only when
  - no sample shows another process with GPU compute activity (SM utilization above
    FOREIGN_COMPUTE_SM_MAX) or another process above p75's BUSY_PCT on the reserved
    cores (the idle-core sampler of scripts/p75_deploy_cost.py, imported); the processes
    that the sampler itself starts are not other processes;
  - the sample before the start shows at most FOREIGN_UTIL_MAX percent SM utilization
    summed over the other processes, so that a run never starts on a busy device;
  - during the run, that summed utilization averaged over the run is at most
    FOREIGN_UTIL_MEAN_MAX percent. Each run sample with GPU rows enters with its largest
    one-second value, weighted by its window, so the average is an upper bound. A process
    that holds the GPU for a share f of the run delays the trainer by at most f of the
    wall time;
  - the samples cover the run (at least wall / interval - 1 samples while it ran);
  - the resolved configuration differs from the campaign run's only in ALLOWED_DIFF;
  - the log reports PARAMS_EXPECTED parameters, the exact mask, the model banner, every
    update and the closing 'done in' line, and the trainer exited with status 0.
A record that fails any rule carries valid=false and the reasons; no reader may use it.

Commands (each prints what it does; only 'run' and 'rejudge' write files):
  plan     the names, the command, the files, the checks and the preconditions (DRY=1)
  sample   the sample taken before a start, and its verdict (SAMPLE_ONLY=1)
  run      the pre-start sample, the run, the in-run samples and the record
  rejudge  <record.json> [--dry]: the verdict of a stored record under the rule in force,
           from its stored files (samples, log, configurations) and the facts the record
           holds. No measured quantity changes. The record is copied once to
           <name>.first.json, and the field 'rejudged' keeps the first verdict. A missing
           file, a stored file that no longer gives the record's facts, and a record of
           a stub trainer are refused. --dry prints the two verdicts and writes nothing.
           Usage: python scripts/training_cost_evidence.py rejudge <record.json> [--dry]

CPU only for this process: it starts no CUDA context. The trainer gets
CUDA_VISIBLE_DEVICES=0, as in scripts/run_exactmask_queue.sh.
"""
import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

PARAMS_EXPECTED = 32786
ALLOWED_DIFF = ('model_suffix',)       # the run name; no other field depends on it
FOREIGN_UTIL_MAX = 5.0                 # % SM of the other processes summed, sample before start
FOREIGN_UTIL_MEAN_MAX = 1.0            # % SM, the same sum averaged over the run samples
FOREIGN_COMPUTE_SM_MAX = 0.0           # % SM of any other compute process, per sample
WINDOW_DEFAULT = 30                    # s, sample before the start
INTERVAL_DEFAULT = 60                  # s, one sample per interval during the run
RECORD_DIR = 'train_log/training_cost'
FIRST_SUFFIX = '.first.json'           # <name>.first.json: the record as the run wrote it
STOPPED = 'stopped by the launcher: '  # opening of the reason of a stopped trainer
HOLD = 'train_log/GPU_HOLD'
NAME_STEM = 'xcost'
ARMS = ('released', 'specialist')
# Options of the launch line of scripts/run_exactmask_queue.sh; every other option is a
# default of params.py. The configuration check proves the resolved configuration equal.
QUEUE_OPTS = ('--data_source', 'FAMILY', '--n_j', '10', '--use_type_embedding', 'True',
              '--validate_timestep', '25', '--use_film', 'False')
MASK_OPTS = ('--first_mask', 'exact', '--mask_ties', 'exact')
TRAIN_ENV = {'CUDA_VISIBLE_DEVICES': '0',
             'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
CSV_FIELDS = ('phase', 'time', 't_rel_s', 'window_s', 'gpu_blocks', 'gpu_foreign_sm_max',
              'gpu_foreign_compute', 'gpu_memory_holders', 'gpu_own_sm_max', 'cpu_foreign_n',
              'cpu_foreign_pct_max', 'cpu_foreign', 'cpu_kernel_threads', 'loadavg1',
              'trainer_alive')

PARAMS_RX = re.compile(r'^\[model\] params total=(\d+)', re.M)
MASK_RX = re.compile(r'^\[train_family\] MASK: first_mask=(\S+) mask_ties=(\S+)', re.M)
MODEL_RX = re.compile(r"^\[train_family\] model=(\S+) regimes=\[([^\]]*)\] updates=(\d+)",
                      re.M)
DONE_RX = re.compile(r'^\[train_family\] done in ([\d.]+) h; best rel=([\d.]+)', re.M)
EP_RX = re.compile(r'^Ep (\d+)\t.*\t ([\d.]+)s$', re.M)
# the two line formats of p75_deploy_cost.busy_processes
BUSY_RX = re.compile(r'^pid (\d+) (?:thread \d+ )?core (\d+) (\d+)% (over|\(lifetime\))')


class Refusal(SystemExit):
    """A precondition or a check failed; the message names it."""


# ------------------------------------------------------------------ names and paths
def run_names(arm, combo, seed, rep):
    """(campaign run, new run) for an arm, combination, seed and attempt number."""
    import arm_registry as REG
    if arm not in ARMS:
        raise Refusal(f'REFUSING: arm {arm!r}; a training-cost run repeats one of {ARMS}')
    if arm == 'released' and combo:
        raise Refusal('REFUSING: the released policy takes no combination')
    if int(rep) < 1:
        raise Refusal(f'REFUSING: attempt number {rep}; attempts count from 1')
    try:
        campaign = REG.model(arm, combo or None, int(seed), environment='exact')
    except (ValueError, KeyError) as e:
        raise Refusal(f'REFUSING: {e}')
    part = 'v1' if arm == 'released' else f'solo-{combo}'
    return campaign, f'{NAME_STEM}-{part}-s{int(seed)}-r{int(rep)}'


def campaign_files(campaign):
    """Stored files of a campaign run (paths relative to the root)."""
    import arm_registry as REG
    m = REG.PREFIX + campaign
    return {'config': f'train_log/FAMILY/config_{m}.json',
            'checkpoint': f'trained_network/FAMILY/{m}.pth',
            'log': f'train_log/FAMILY_{campaign}.log'}


def new_files(name):
    """Files of a measurement run: those the trainer writes and those of the launcher."""
    import arm_registry as REG
    m = REG.PREFIX + name
    return {'checkpoint': f'trained_network/FAMILY/{m}.pth',
            'config': f'train_log/FAMILY/config_{m}.json',
            'vali': f'train_log/FAMILY/valiquality_{m}.txt',
            'reward': f'train_log/FAMILY/reward_{m}.txt',
            'log': f'{RECORD_DIR}/{name}.log',
            'samples': f'{RECORD_DIR}/{name}.samples.csv',
            'record': f'{RECORD_DIR}/{name}.json',
            'archive': f'{RECORD_DIR}/{name}'}


TRAINER_FILES = ('checkpoint', 'config', 'vali', 'reward')


def parse_cores(text):
    """'4-7' or '4,5,7' or '3' -> [4, 5, 6, 7]."""
    out = []
    for part in str(text).replace(' ', '').split(','):
        if not part:
            continue
        a, _, b = part.partition('-')
        out.extend(range(int(a), int(b or a) + 1))
    if not out or len(set(out)) != len(out):
        raise Refusal(f'REFUSING: core list {text!r}')
    return sorted(out)


# ------------------------------------------------------------------ the log and the configuration
def log_facts(text):
    """What the trainer's log states, or None per item when the line is missing."""
    p = PARAMS_RX.search(text)
    mk = MASK_RX.search(text)
    md = MODEL_RX.search(text)
    dn = DONE_RX.search(text)
    eps = EP_RX.findall(text)
    return {'params': int(p.group(1)) if p else None,
            'mask': (mk.group(1), mk.group(2)) if mk else None,
            'model': md.group(1) if md else None,
            'banner_updates': int(md.group(3)) if md else None,
            'done_in_h': float(dn.group(1)) if dn else None,
            'best_rel': float(dn.group(2)) if dn else None,
            'updates_logged': len(eps),
            'update_seconds_sum': round(sum(float(s) for _, s in eps), 3),
            'last_update': int(eps[-1][0]) if eps else 0}


def config_diff(ref, new):
    """[(key, campaign value, new value)] for every field that differs."""
    absent = '<absent>'
    return [(k, ref.get(k, absent), new.get(k, absent)) for k in sorted(set(ref) | set(new))
            if ref.get(k, absent) != new.get(k, absent)]


def config_problems(ref, new, campaign, name):
    """Reasons the new configuration is not the campaign's under a new name."""
    probs = []
    for k, rv, nv in config_diff(ref, new):
        if k == 'model_suffix' and rv == campaign and nv == name:
            continue
        probs.append(f'configuration field {k} is {nv!r}, the campaign run has {rv!r}')
    for k in ('first_mask', 'mask_ties'):
        if new.get(k) != 'exact':
            probs.append(f'configuration records {k}={new.get(k)!r}, want exact')
    return probs


def campaign_problems(ref, arm, combo, seed):
    """Reasons the stored configuration is not the campaign run of this arm."""
    import arm_registry as REG
    want = ','.join(REG.TRAIN_COMBOS) if arm == 'released' else combo
    probs = []
    if ref.get('family_regimes') != want:
        probs.append(f"campaign configuration trains {ref.get('family_regimes')!r}, want {want!r}")
    if int(ref.get('seed_train', -1)) != int(seed):
        probs.append(f"campaign configuration has seed_train={ref.get('seed_train')}, want {seed}")
    for k in ('first_mask', 'mask_ties'):
        if ref.get(k) != 'exact':
            probs.append(f'campaign configuration records {k}={ref.get(k)!r}, want exact')
    return probs


def train_command(py, cores, ref, name):
    """The launch line of scripts/run_exactmask_queue.sh with the campaign run's regimes,
    updates and seed, read from its stored configuration."""
    return (['taskset', '-c', cores, py, 'train_family.py', *QUEUE_OPTS,
             '--family_regimes', str(ref['family_regimes']),
             '--max_updates', str(int(ref['max_updates'])),
             '--seed_train', str(int(ref['seed_train'])), *MASK_OPTS,
             '--model_suffix', name])


# ------------------------------------------------------------------ GPU samples
def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_pmon(text):
    """Rows of `nvidia-smi pmon -s um` output grouped into one block per sample.
    Columns are read from the header line, so a driver that adds a column is read
    correctly. A block ends where a process id repeats."""
    cols, blocks, cur, seen = None, [], [], set()
    for line in text.splitlines():
        if line.startswith('#'):
            toks = line[1:].split()
            if cols is None and 'pid' in toks:
                cols = toks
            continue
        if not line.strip() or cols is None:
            continue
        toks = line.split()
        k = len(cols) - 1
        row = dict(zip(cols[:k], toks[:k]))
        row['command'] = ' '.join(toks[k:])
        pid = row.get('pid')
        if pid in seen:
            blocks.append(cur)
            cur, seen = [], set()
        seen.add(pid)
        cur.append(row)
    if cur:
        blocks.append(cur)
    return blocks


def gpu_summary(blocks, own_pids):
    """Per-sample figures of a list of pmon blocks: the largest SM utilization summed over
    the other processes, the other compute processes with SM activity, the other compute
    processes that only hold memory, and the trainer's own SM utilization."""
    out = {'gpu_blocks': len(blocks), 'gpu_foreign_sm_max': 0.0, 'gpu_foreign_compute': [],
           'gpu_memory_holders': [], 'gpu_own_sm_max': 0.0}
    holders, compute = {}, {}
    for b in blocks:
        foreign = own = 0.0
        for r in b:
            pid = r.get('pid', '-')
            if not pid.isdigit():
                continue
            pid = int(pid)
            sm = _num(r.get('sm')) or 0.0
            if pid in own_pids:
                own += sm
                continue
            foreign += sm
            if 'C' in r.get('type', ''):
                if sm > FOREIGN_COMPUTE_SM_MAX:
                    compute[pid] = (max(sm, compute.get(pid, (0.0,))[0]), r['command'])
                else:
                    holders[pid] = f'{pid}:{_num(r.get("fb")) or 0:g}MB:{r["command"]}'
        out['gpu_foreign_sm_max'] = max(out['gpu_foreign_sm_max'], foreign)
        out['gpu_own_sm_max'] = max(out['gpu_own_sm_max'], own)
    out['gpu_foreign_compute'] = [f'{pid}:{sm:g}%:{cmd}' for pid, (sm, cmd) in sorted(compute.items())]
    out['gpu_memory_holders'] = sorted(holders.values())
    return out


def pmon_cmd(count):
    return ['nvidia-smi', 'pmon', '-s', 'um', '-c', str(int(count)), '-d', '1']


# ------------------------------------------------------------------ CPU samples
def proc_table():
    """{pid: ppid} of every process, from one pass over /proc/<pid>/stat."""
    table = {}
    for d in os.listdir('/proc'):
        if not d.isdigit():
            continue
        try:
            with open(f'/proc/{d}/stat') as f:
                s = f.read()
            table[int(d)] = int(s[s.rfind(')') + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return table


def descendants(table, roots):
    """The roots and every process below them."""
    kids = {}
    for pid, ppid in table.items():
        kids.setdefault(ppid, []).append(pid)
    out, stack = set(), [r for r in roots if r]
    while stack:
        p = stack.pop()
        if p in out:
            continue
        out.add(p)
        stack.extend(kids.get(p, ()))
    return out


def split_busy(entries, own, table):
    """Split the lines of p75.busy_processes into other processes, the launcher's and
    trainer's own processes, and kernel threads (children of kthreadd, pid 2). A line
    that does not parse counts as another process."""
    foreign, kernel = [], []
    for e in entries:
        m = BUSY_RX.match(e)
        if not m:
            foreign.append((100.0, f'unparsed: {e}'))
            continue
        pid, pct = int(m.group(1)), float(m.group(3))
        if pid in own:
            continue
        if pid == 2 or table.get(pid) == 2:
            kernel.append((pct, e))
            continue
        foreign.append((pct, e))
    return foreign, kernel


def cpu_summary(foreign, kernel):
    return {'cpu_foreign_n': len(foreign),
            'cpu_foreign_pct_max': max([p for p, _ in foreign], default=0.0),
            'cpu_foreign': [e[:160] for _, e in foreign][:10],
            'cpu_kernel_threads': [e[:160] for _, e in kernel][:10]}


def take_sample(cores, window, phase, trainer_pid=None, t0=None, gpu_count=None,
                alive=None):
    """One sample: p75's idle-core sampler over `window` seconds on the reserved cores and
    one pmon call (gpu_count samples one second apart, started together with the core
    sampler; one sample when None, taken after it). The processes this sampler starts
    (the pmon call) count as the launcher's own, also after they have exited."""
    import p75_deploy_cost as P75      # imported here: its import changes argv and cwd
    pm = None
    helpers = []                       # pids of the processes this sample starts
    if gpu_count:
        pm = subprocess.Popen(pmon_cmd(gpu_count), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        helpers.append(pm.pid)
    t_wall = time.time()
    entries = P75.busy_processes(cores, window)
    if pm is None:
        pm = subprocess.Popen(pmon_cmd(1), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True)
        helpers.append(pm.pid)
    try:
        text = pm.communicate(timeout=60 + 2 * (gpu_count or 1))[0]
    except subprocess.TimeoutExpired:
        pm.kill()
        text = pm.communicate()[0]
    # The process table is read only when the core sampler flagged a process, so that a
    # quiet minute costs one pmon call and the two snapshots of the core sampler. It is
    # read after the pmon call has ended, so the helpers enter `own` by their pids, not
    # through the table.
    table = proc_table() if entries else None
    own = (descendants(table, [os.getpid(), trainer_pid]) | set(helpers)
           if table is not None else set())
    foreign, kernel = split_busy(entries, own, table or {})
    gpu_own = set()
    if trainer_pid:
        gpu_own = descendants(table, [trainer_pid]) if table is not None else {trainer_pid}
    row = {'phase': phase, 'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t_wall)),
           't_rel_s': round(t_wall - t0, 1) if t0 else 0.0, 'window_s': window,
           **gpu_summary(parse_pmon(text) if pm.returncode == 0 else [], gpu_own),
           **cpu_summary(foreign, kernel),
           'loadavg1': P75.loadavg()[0],
           'trainer_alive': bool(alive()) if alive else False}
    return row


def sample_problems(row):
    """Reasons one sample shows that the device or the cores were not the run's alone.
    The limit FOREIGN_UTIL_MAX applies to the sample before the start; the utilization of
    the other processes during the run is judged by its average (gpu_run_average)."""
    probs = []
    where = f"{row['phase']} sample at {row['time']}"
    if int(row['gpu_blocks']) < 1:
        probs.append(f'{where}: the GPU sample failed (no pmon rows)')
    if row['gpu_foreign_compute']:
        probs.append(f'{where}: other GPU compute ' + ', '.join(row['gpu_foreign_compute']))
    if row['phase'] != 'run' and float(row['gpu_foreign_sm_max']) > FOREIGN_UTIL_MAX:
        probs.append(f"{where}: other processes use {float(row['gpu_foreign_sm_max']):g}% of "
                     f'the GPU (limit {FOREIGN_UTIL_MAX:g}%)')
    if int(row['cpu_foreign_n']) > 0:
        probs.append(f"{where}: {row['cpu_foreign_n']} other process(es) on the reserved "
                     f"cores, up to {float(row['cpu_foreign_pct_max']):g}%: "
                     + '; '.join(row['cpu_foreign'][:3]))
    return probs


def _csv_row(r):
    return {k: (' | '.join(r[k]) if isinstance(r[k], list) else r[k]) for k in CSV_FIELDS}


def write_samples(path, rows):
    """A new samples file with these rows."""
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(_csv_row(r))


def append_sample(path, row):
    """One more row at the end of a samples file."""
    with open(path, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(_csv_row(row))


def read_samples(path):
    """Rows of a samples file with the list fields split and the numbers parsed."""
    rows = []
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            for k in ('gpu_foreign_compute', 'gpu_memory_holders', 'cpu_foreign',
                      'cpu_kernel_threads'):
                r[k] = [x for x in r[k].split(' | ') if x] if r[k] else []
            for k in ('t_rel_s', 'window_s', 'gpu_foreign_sm_max', 'gpu_own_sm_max',
                      'cpu_foreign_pct_max', 'loadavg1'):
                r[k] = float(r[k])
            for k in ('gpu_blocks', 'cpu_foreign_n'):
                r[k] = int(r[k])
            r['trainer_alive'] = r['trainer_alive'] in ('True', 'true', '1')
            rows.append(r)
    return rows


def gpu_run_average(run):
    """(average, samples above FOREIGN_UTIL_MAX) of the SM utilization summed over the other
    processes during the run, in percent. The average is the sum over the run samples with
    GPU rows of gpu_foreign_sm_max * window_s divided by the sum of their window_s; each
    sample enters with its largest one-second value, so the average is an upper bound. A
    sample without GPU rows fails the record by itself and does not enter the average. The
    average is None when no run sample has GPU rows."""
    held = [r for r in run if int(r['gpu_blocks']) >= 1]
    weight = sum(float(r['window_s']) for r in held)
    mean = (sum(float(r['gpu_foreign_sm_max']) * float(r['window_s']) for r in held) / weight
            if weight > 0 else None)
    above = sum(1 for r in run if float(r['gpu_foreign_sm_max']) > FOREIGN_UTIL_MAX)
    return mean, above


def assess_samples(rows, wall_seconds, interval):
    """(problems, summary) of the samples of one run."""
    probs = []
    pre = [r for r in rows if r['phase'] == 'pre']
    run = [r for r in rows if r['phase'] == 'run']
    if not pre:
        probs.append('no sample before the start')
    for r in rows:
        probs.extend(sample_problems(r))
    mean, above = gpu_run_average(run)
    if mean is not None and mean > FOREIGN_UTIL_MEAN_MAX:
        probs.append(f'other processes used {mean:.4g}% of the GPU on average over the run '
                     f'(limit {FOREIGN_UTIL_MEAN_MAX:.1f}%), above {FOREIGN_UTIL_MAX:g}% in '
                     f'{above} of {len(run)} samples')
    need = max(1, int(wall_seconds // interval) - 1)
    if len(run) < need:
        probs.append(f'{len(run)} samples during the run, {need} needed for '
                     f'{wall_seconds:.0f} s at one per {interval:g} s')
    summary = {'n': len(rows), 'pre': len(pre), 'run': len(run), 'needed_run': need,
               'foreign_gpu_util_max': max([float(r['gpu_foreign_sm_max']) for r in rows],
                                           default=None),
               'foreign_gpu_util_mean': round(mean, 4) if mean is not None else None,
               'foreign_gpu_samples_above_limit': above,
               'foreign_gpu_compute': sorted({x for r in rows for x in r['gpu_foreign_compute']}),
               'gpu_memory_holders': sorted({x for r in rows for x in r['gpu_memory_holders']}),
               'foreign_cpu_pct_max': max([float(r['cpu_foreign_pct_max']) for r in rows],
                                          default=None),
               'kernel_threads': sorted({x for r in rows for x in r['cpu_kernel_threads']})[:20],
               'own_gpu_util_max': max([float(r['gpu_own_sm_max']) for r in run], default=None)}
    return probs, summary


# ------------------------------------------------------------------ the record
def assess_run(*, campaign, name, ref_cfg, new_cfg, log_text, exit_code, wall_seconds,
               samples, interval, checkpoint_exists, abort_reason=None, stub=None):
    """(valid, reasons, facts, sample summary) of one finished or stopped run."""
    reasons = []
    if stub:
        reasons.append(f'test run with the stub trainer {stub}; never a measurement')
    if abort_reason:
        reasons.append(f'{STOPPED}{abort_reason}')
    if exit_code != 0:
        reasons.append(f'trainer exit status {exit_code}')
    f = log_facts(log_text)
    if f['params'] is None:
        reasons.append("log has no '[model] params total=' line")
    elif f['params'] != PARAMS_EXPECTED:
        reasons.append(f"log reports {f['params']} parameters, want {PARAMS_EXPECTED}")
    if f['mask'] is None:
        reasons.append("log has no '[train_family] MASK:' line")
    elif f['mask'] != ('exact', 'exact'):
        reasons.append(f"log reports the mask {f['mask']}, want ('exact', 'exact')")
    if f['model'] is None:
        reasons.append("log has no '[train_family] model=' line")
    if f['done_in_h'] is None:
        reasons.append("log has no '[train_family] done in' line")
    if new_cfg is None:
        reasons.append('the trainer wrote no configuration')
    else:
        reasons.extend(config_problems(ref_cfg, new_cfg, campaign, name))
        upd = int(new_cfg.get('max_updates', -1))
        if f['banner_updates'] != upd or f['updates_logged'] != upd or f['last_update'] != upd:
            reasons.append(f"updates: configuration {upd}, banner {f['banner_updates']}, "
                           f"logged {f['updates_logged']} (last {f['last_update']})")
    if not checkpoint_exists:
        reasons.append('the trainer saved no checkpoint')
    sprobs, ssum = assess_samples(samples, wall_seconds, interval)
    reasons.extend(sprobs)
    return not reasons, reasons, f, ssum


def thresholds(busy_pct):
    """The limits a record is judged by, as the record states them."""
    return {'foreign_gpu_util_mean_max_pct': FOREIGN_UTIL_MEAN_MAX,
            'foreign_gpu_util_mean': 'run samples with GPU rows, each at its largest '
                                     'one-second value, weighted by its window',
            'foreign_gpu_util_pre_max_pct': FOREIGN_UTIL_MAX,
            'foreign_compute_sm_max_pct': FOREIGN_COMPUTE_SM_MAX,
            'foreign_cpu_busy_pct': busy_pct,
            'samples_needed': 'wall seconds / interval - 1, at least 1'}


def check_record(rec):
    """Reasons a training-cost record may not be used; empty when it may. A record that
    rejudge has rewritten is read like any other."""
    probs = []
    if rec.get('kind') != 'training_cost':
        return [f"kind {rec.get('kind')!r}, want 'training_cost'"]
    if rec.get('valid') is not True:
        probs.append('record is invalid: ' + '; '.join(rec.get('reasons') or ['no reason given']))
    if rec.get('stub'):
        probs.append('record of a stub trainer')
    log = (rec.get('files') or {}).get('log', '')
    if not log.startswith(RECORD_DIR + '/'):
        probs.append(f'log {log!r} is not a measurement log under {RECORD_DIR}/')
    rep = rec.get('repeats') or {}
    if rep.get('arm') not in ARMS:
        probs.append(f"repeats an arm {rep.get('arm')!r}")
    for k in ('wall_seconds', 'seconds_per_update'):
        if not isinstance(rec.get(k), (int, float)) or not rec[k] > 0:
            probs.append(f'{k} = {rec.get(k)!r}')
    if not isinstance(rec.get('updates'), int) or rec['updates'] < 1:
        probs.append(f"updates = {rec.get('updates')!r}")
    return probs


# ------------------------------------------------------------------ host facts
def host_facts(cores):
    """GPU model and driver from nvidia-smi, the torch and CUDA versions of the trainer's
    environment (read from the installed package, torch is not imported), the processor."""
    import importlib.metadata as md
    import importlib.util
    import p75_deploy_cost as P75
    gpu = {}
    try:
        line = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader'], capture_output=True, text=True,
                              timeout=30).stdout.splitlines()[0]
        gpu = dict(zip(('gpu', 'driver', 'gpu_memory'), [x.strip() for x in line.split(',')]))
    except (OSError, IndexError, subprocess.SubprocessError):
        gpu = {'gpu': None, 'driver': None, 'gpu_memory': None}
    torch_v = cuda_v = None
    try:
        torch_v = md.version('torch')
        spec = importlib.util.find_spec('torch')
        vfile = os.path.join(list(spec.submodule_search_locations)[0], 'version.py')
        m = re.search(r"^cuda\s*(?::\s*\S+)?\s*=\s*'([^']*)'", open(vfile).read(), re.M)
        cuda_v = m.group(1) if m else None
    except (md.PackageNotFoundError, OSError, TypeError, AttributeError):
        pass
    return {**gpu, 'torch': torch_v, 'torch_cuda': cuda_v,
            'python': sys.version.split()[0], 'hostname': os.uname().nodename,
            **P75.core_info(cores)}


# ------------------------------------------------------------------ plan
def make_plan(a):
    root = os.path.abspath(a.root)
    campaign, name = run_names(a.arm, a.combo, a.seed, a.rep)
    cf, nf = campaign_files(campaign), new_files(name)
    cores = parse_cores(a.cores)
    ref = None
    pre = []                                   # (ok, text)
    env_set = os.environ.get('PACT_ENV')
    pre.append((env_set in (None, '', 'exact'),
                f'environment exact (PACT_ENV={env_set!r})'))
    ok_cfg = os.path.isfile(os.path.join(root, cf['config']))
    ok_ckpt = os.path.isfile(os.path.join(root, cf['checkpoint']))
    lp = os.path.join(root, cf['log'])
    ok_log = os.path.isfile(lp) and 'done in' in open(lp, errors='replace').read()
    pre.append((ok_cfg, f"campaign configuration {cf['config']}"))
    pre.append((ok_ckpt, f"campaign checkpoint {cf['checkpoint']}"))
    pre.append((ok_log, f"campaign log {cf['log']} with its 'done in' line"))
    if ok_cfg:
        ref = json.load(open(os.path.join(root, cf['config'])))
        probs = campaign_problems(ref, a.arm, a.combo, a.seed)
        pre.append((not probs, 'campaign configuration is the ' +
                    ('released recipe' if a.arm == 'released' else f'{a.combo} specialist') +
                    ' under the exact mask' + (': ' + '; '.join(probs) if probs else '')))
    taken = [nf[k] for k in (*TRAINER_FILES, 'log', 'samples', 'record', 'archive')
             if os.path.exists(os.path.join(root, nf[k]))]
    pre.append((not taken, f'new name {name} unused' + (': ' + ', '.join(taken) if taken else '')))
    pre.append((not os.path.exists(os.path.join(root, HOLD)), f'{HOLD} absent'))
    pre.append((shutil.which('nvidia-smi') is not None, 'nvidia-smi available'))
    pre.append((1 <= int(a.threads) <= len(cores),
                f'threads {a.threads} on {len(cores)} reserved core(s)'))
    if ref is not None:
        cmd = train_command(a.py, a.cores, ref, name)
    else:
        cmd = ['taskset', '-c', a.cores, a.py, 'train_family.py', *QUEUE_OPTS,
               '--family_regimes', '<from the campaign configuration>',
               '--max_updates', '<from the campaign configuration>',
               '--seed_train', str(a.seed), *MASK_OPTS, '--model_suffix', name]
    env = {**TRAIN_ENV, 'OMP_NUM_THREADS': str(a.threads), 'MKL_NUM_THREADS': str(a.threads)}
    return {'root': root, 'campaign': campaign, 'name': name, 'arm': a.arm,
            'combo': a.combo or None, 'seed': int(a.seed), 'rep': int(a.rep),
            'campaign_files': cf, 'files': nf, 'cores': a.cores, 'core_list': cores,
            'threads': int(a.threads), 'command': cmd, 'train_env': env, 'ref': ref,
            'preconditions': pre, 'window': a.window, 'interval': a.interval}


def print_plan(p):
    import arm_registry as REG
    say = print
    say('[training-cost] DRY RUN: nothing is started and nothing is written')
    say('environment      exact')
    say(f"repeats          {p['campaign']} (arm {p['arm']}"
        + (f", combination {p['combo']}" if p['combo'] else '') + f", seed {p['seed']})")
    for k, v in p['campaign_files'].items():
        say(f'  {k:<14} {v}')
    say(f"new run          {p['name']} (model {REG.PREFIX}{p['name']}, attempt {p['rep']})")
    say(f"cores / threads  {p['cores']} / {p['threads']}")
    say('command          ' + ' '.join(f'{k}={v}' for k, v in p['train_env'].items()) + ' '
        + ' '.join(shlex.quote(x) for x in p['command']))
    say('files written    (all under the root)')
    for k in ('log', 'samples', 'record'):
        say(f"  {k:<14} {p['files'][k]}")
    for k in TRAINER_FILES:
        say(f"  {k:<14} {p['files'][k]}  (by the trainer; moved to {p['files']['archive']}/ "
            'after the run)')
    say('checks')
    say(f"  before the start  {p['window']} s: one pmon call of {p['window']} one-second samples "
        f"and p75's idle-core sampler on cores {p['cores']}; a run starts only when no other "
        f'process has GPU compute activity (SM above {FOREIGN_COMPUTE_SM_MAX:g}%), the other '
        f'processes use at most {FOREIGN_UTIL_MAX:g}% of the GPU and the cores are idle')
    say(f"  during the run    every {p['interval']} s the same two samples over the interval "
        '(the GPU every second), appended to the samples file; no sample may show other GPU '
        'compute or a busy reserved core, and the GPU use of the other processes averaged '
        f'over the run may be at most {FOREIGN_UTIL_MEAN_MAX:.1f}%')
    say(f'  at the start      the log must report {PARAMS_EXPECTED} parameters and the exact '
        f'mask, and the configuration must equal the campaign run\'s except {ALLOWED_DIFF}; '
        'otherwise the trainer is stopped')
    say("  at the end        exit status 0, the 'done in' line, every update logged, the "
        'checkpoint saved, the samples cover the run; the record states valid and the reasons')
    say('preconditions')
    for ok, text in p['preconditions']:
        say(f"  {'OK  ' if ok else 'FAIL'} {text}")


def failing(p):
    return [t for ok, t in p['preconditions'] if not ok]


# ------------------------------------------------------------------ commands
def cmd_plan(a):
    p = make_plan(a)
    print_plan(p)
    bad = failing(p)
    if bad:
        print(f'DRY: {len(bad)} precondition(s) fail, so a run would be refused')
        return 3
    print('DRY: every precondition holds')
    return 0


def pre_start(cores, window):
    row = take_sample(cores, window, 'pre', gpu_count=window)
    probs = sample_problems(row)
    print(f"[training-cost] sample before the start ({window} s, cores {cores}): GPU other "
          f"processes {row['gpu_foreign_sm_max']:g}% SM, other compute "
          f"{row['gpu_foreign_compute'] or 'none'}, memory holders "
          f"{row['gpu_memory_holders'] or 'none'}; cores: {row['cpu_foreign_n']} other "
          f"process(es), kernel threads {len(row['cpu_kernel_threads'])}; load "
          f"{row['loadavg1']:.2f}", flush=True)
    for x in probs:
        print(f'  NOT IDLE: {x}', flush=True)
    return row, probs


def cmd_sample(a):
    cores = parse_cores(a.cores)
    _, probs = pre_start(cores, a.window)
    print('[training-cost] verdict: ' + ('idle, a run could start' if not probs else
                                         'not idle, a run would be refused'))
    return 0 if not probs else 1


def _read(path):
    try:
        return open(path, errors='replace').read()
    except OSError:
        return ''


def cmd_run(a):
    p = make_plan(a)
    root, files, name = p['root'], p['files'], p['name']
    stub = a.stub_trainer or None
    if stub and os.path.realpath(root) == os.path.realpath(ROOT):
        raise Refusal('REFUSING: a stub trainer runs only under a test root, never in the '
                      'repository')
    bad = failing(p)
    if bad:
        print_plan(p)
        raise Refusal('REFUSING to start: ' + '; '.join(bad))
    if a.allow_busy and not stub:
        raise Refusal('REFUSING: --allow_busy is for tests with a stub trainer only')
    cores = p['core_list']
    pre_row, probs = pre_start(cores, a.window)
    if probs and not a.allow_busy:
        raise Refusal('REFUSING to start: the device or the cores are not idle; nothing '
                      'was started and nothing was written')
    os.makedirs(os.path.join(root, RECORD_DIR), exist_ok=True)
    host = host_facts(cores)
    cmd = list(p['command'])
    env = dict(os.environ, **p['train_env'])
    for k in ('PACT_FIRST_MASK', 'PACT_MASK_TOL'):
        env.pop(k, None)
    if stub:
        cmd = [a.py, os.path.abspath(stub), *cmd[cmd.index('train_family.py') + 1:]]
        env['CUDA_VISIBLE_DEVICES'] = ''
    log_path = os.path.join(root, files['log'])
    cfg_path = os.path.join(root, files['config'])
    print(f"[training-cost] TRAIN {name} (repeats {p['campaign']}) on cores {p['cores']}, "
          f"{p['threads']} thread(s); log {files['log']}", flush=True)
    samples_path = os.path.join(root, files['samples'])
    write_samples(samples_path, [pre_row])
    state = {'abort': None}
    logf = open(log_path, 'w')
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=root, env=env,
                            start_new_session=True)
    done = threading.Event()
    res = {}

    def waiter():
        _, status, ru = os.wait4(proc.pid, 0)
        res['t1'] = time.time()
        res['code'] = os.waitstatus_to_exitcode(status)
        res['cpu_s'] = ru.ru_utime + ru.ru_stime
        proc.returncode = res['code']
        done.set()

    threading.Thread(target=waiter, daemon=True).start()

    def stop(reason):
        if state['abort'] is None:
            state['abort'] = reason
            print(f'[training-cost] STOPPING the trainer: {reason}', flush=True)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if not done.wait(30):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def on_signal(signum, _frame):
        stop(f'signal {signum} to the launcher')

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    checked = {'params': False, 'config': False}
    next_t = t0
    try:
        while not done.is_set():
            next_t += a.interval               # one sample per interval, on the clock
            win = max(1, a.interval - 2)       # the GPU is sampled every second of it
            append_sample(samples_path, take_sample(cores, win, 'run', proc.pid, t0,
                                                    gpu_count=win,
                                                    alive=lambda: not done.is_set()))
            text = _read(log_path)
            if not checked['params']:
                m = PARAMS_RX.search(text)
                if m:
                    checked['params'] = True
                    if int(m.group(1)) != PARAMS_EXPECTED:
                        stop(f'{m.group(1)} parameters, want {PARAMS_EXPECTED}')
            if not checked['config'] and os.path.isfile(cfg_path):
                try:
                    new = json.load(open(cfg_path))
                except ValueError:
                    new = None                 # still being written; read at the next sample
                if new is not None:
                    checked['config'] = True
                    cp = config_problems(p['ref'], new, p['campaign'], name)
                    if cp:
                        stop('configuration drift: ' + '; '.join(cp))
            done.wait(max(0.0, next_t - time.time()))
    except Exception as e:                     # the record is still written, and invalid
        stop(f'launcher error {e!r}')
    finally:
        if not done.is_set():
            stop(state['abort'] or 'the launcher ended before the trainer')
            done.wait()
        logf.close()
    wall = res['t1'] - t0
    text = _read(log_path)
    new_cfg = json.load(open(cfg_path)) if os.path.isfile(cfg_path) else None
    ck = os.path.isfile(os.path.join(root, files['checkpoint']))
    samples = read_samples(samples_path)
    valid, reasons, facts, ssum = assess_run(
        campaign=p['campaign'], name=name, ref_cfg=p['ref'], new_cfg=new_cfg, log_text=text,
        exit_code=res['code'], wall_seconds=wall, samples=samples, interval=a.interval,
        checkpoint_exists=ck, abort_reason=state['abort'], stub=stub)
    # move the trainer's files out of the folders that other scripts glob
    arch = os.path.join(root, files['archive'])
    os.makedirs(arch, exist_ok=True)
    moved = {}
    for k in TRAINER_FILES:
        src = os.path.join(root, files[k])
        if os.path.exists(src):
            dst = os.path.join(arch, os.path.basename(src))
            shutil.move(src, dst)
            moved[k] = os.path.relpath(dst, root)
    upd = int(new_cfg['max_updates']) if new_cfg else None
    rec = {
        'kind': 'training_cost', 'generator': 'scripts/run_training_cost.sh',
        'name': name, 'environment': 'exact',
        'repeats': {'arm': p['arm'], 'combination': p['combo'], 'seed': p['seed'],
                    'run': p['campaign'], 'config': p['campaign_files']['config']},
        'attempt': p['rep'], 'command': [home_masked(x) for x in cmd],
        'train_env': p['train_env'],
        'cores': p['cores'], 'affinity': cores, 'threads': p['threads'],
        'host': host, 'started': started,
        'finished': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(res['t1'])),
        'wall_seconds': round(wall, 3), 'hours': round(wall / 3600.0, 6),
        'done_in_hours': facts['done_in_h'], 'best_rel': facts['best_rel'],
        'updates': upd, 'episodes': upd * int(new_cfg['num_envs']) if new_cfg else None,
        'num_envs': int(new_cfg['num_envs']) if new_cfg else None,
        'update_seconds_sum': facts['update_seconds_sum'],
        'seconds_per_update': (round(facts['update_seconds_sum'] / upd, 4)
                               if upd and facts['updates_logged'] == upd else None),
        'trainer_cpu_seconds': round(res.get('cpu_s', 0.0), 1),
        'params_total': facts['params'], 'mask': facts['mask'],
        'exit_status': res['code'],
        'config_check': {'reference': p['campaign_files']['config'],
                         'differences': [list(d) for d in
                                         (config_diff(p['ref'], new_cfg) if new_cfg else [])],
                         'allowed': list(ALLOWED_DIFF)},
        'samples': {**ssum, 'file': files['samples'], 'interval_s': a.interval,
                    'pre_window_s': a.window},
        'thresholds': thresholds(_busy_pct()),
        'files': {'log': files['log'], 'samples': files['samples'], **moved},
        'stub': stub, 'allow_busy': bool(a.allow_busy), 'valid': valid, 'reasons': reasons,
    }
    rp = os.path.join(root, files['record'])
    with open(rp, 'w') as f:
        json.dump(rec, f, indent=1)
    print(f"[training-cost] {name}: {wall / 3600:.3f} h wall, "
          f"{rec['seconds_per_update']} s per update, valid={valid}", flush=True)
    for r in reasons:
        print(f'  INVALID: {r}', flush=True)
    print(f'[training-cost] wrote {files["record"]}', flush=True)
    return 0 if valid else 1


# ------------------------------------------------------------------ rejudge
# entries of the sample summary that follow from the rule and not from the samples file alone
RULE_SUMMARY = ('needed_run', 'foreign_gpu_util_mean', 'foreign_gpu_samples_above_limit')


def first_record_path(path):
    """<name>.first.json beside <name>.json."""
    return path[:-len('.json')] + FIRST_SUFFIX if path.endswith('.json') else path + FIRST_SUFFIX


def _stored(root, rel, what):
    """Absolute path of a stored file named by the record; refuses when it is missing."""
    p = os.path.join(root, rel) if rel else ''
    if not rel or not os.path.isfile(p):
        raise Refusal(f'REFUSING: {what} {rel!r} is missing; nothing was written')
    return p


def _field(rec, *keys):
    """rec[k1][k2]...; refuses when the record lacks the field."""
    v = rec
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            raise Refusal(f"REFUSING: the record has no field {'.'.join(keys)}; nothing was "
                          'written')
        v = v[k]
    return v


def rejudge_record(root, path):
    """(new record, first verdict, verdict before this call) of a stored record judged again
    under the rule in force, from its stored files and the facts the record holds. Every
    field other than valid, reasons, samples (the summary), thresholds and rejudged is
    copied unchanged. Refuses a record of a stub trainer, a record that is not
    <root>/RECORD_DIR/<name>.json, a missing stored file, and stored files that no longer
    give the facts of the record (log, configurations, sample summary)."""
    root, path = os.path.abspath(root), os.path.abspath(path)
    if not os.path.isfile(path):
        raise Refusal(f'REFUSING: no record {path}')
    try:
        with open(path) as f:
            rec = json.load(f)
    except ValueError as e:
        raise Refusal(f'REFUSING: {path} is not a JSON record ({e})')
    if not isinstance(rec, dict) or rec.get('kind') != 'training_cost':
        raise Refusal(f'REFUSING: {path} is not a training-cost record')
    if rec.get('stub'):
        raise Refusal(f"REFUSING: {path} is a record of a stub trainer ({rec.get('stub')}); "
                      'never a measurement')
    if rec.get('allow_busy'):
        raise Refusal(f'REFUSING: {path} is a record of a test start on busy cores '
                      '(--allow_busy); never a measurement')
    name = str(rec.get('name') or '')
    want = os.path.join(root, RECORD_DIR, f'{name}.json')
    if not name or os.path.realpath(path) != os.path.realpath(want):
        raise Refusal(f'REFUSING: {path} is not the record {RECORD_DIR}/{name}.json under the '
                      f'root {root}')
    files = _field(rec, 'files')
    for k in ('log', 'samples'):
        _field(rec, 'files', k)
    paths = {k: _stored(root, v, f'the stored file {k}') for k, v in sorted(files.items())}
    ref_path = _stored(root, _field(rec, 'repeats', 'config'), 'the campaign configuration')
    with open(ref_path) as f:
        ref_cfg = json.load(f)
    new_cfg = None
    if 'config' in files:
        with open(paths['config']) as f:
            new_cfg = json.load(f)
    log_text = _read(paths['log'])
    samples = read_samples(paths['samples'])
    old_summary = _field(rec, 'samples')
    interval = _field(rec, 'samples', 'interval_s')
    first_reasons = list(rec.get('reasons') or [])
    stopped = [r[len(STOPPED):] for r in first_reasons if r.startswith(STOPPED)]
    valid, reasons, facts, ssum = assess_run(
        campaign=_field(rec, 'repeats', 'run'), name=name, ref_cfg=ref_cfg, new_cfg=new_cfg,
        log_text=log_text, exit_code=_field(rec, 'exit_status'),
        wall_seconds=float(_field(rec, 'wall_seconds')), samples=samples, interval=interval,
        checkpoint_exists='checkpoint' in files, abort_reason=stopped[0] if stopped else None)
    # the stored files must give the facts the record holds; otherwise they are not its files
    given = {'params_total': facts['params'],
             'mask': list(facts['mask']) if facts['mask'] else None,
             'done_in_hours': facts['done_in_h'], 'best_rel': facts['best_rel'],
             'update_seconds_sum': facts['update_seconds_sum']}
    if new_cfg is not None:
        given['updates'] = int(new_cfg['max_updates'])
    diffs = [list(d) for d in config_diff(ref_cfg, new_cfg)] if new_cfg is not None else []
    mism = [f'{k}: the record holds {rec.get(k)!r}, the stored files give {v!r}'
            for k, v in given.items() if rec.get(k) != v]
    stored_diffs = (rec.get('config_check') or {}).get('differences')
    if stored_diffs != diffs:
        mism.append(f'configuration differences: the record holds {stored_diffs!r}, the '
                    f'stored files give {diffs!r}')
    mism += [f'samples.{k}: the record holds {old_summary[k]!r}, the samples file gives {v!r}'
             for k, v in ssum.items()
             if k not in RULE_SUMMARY and k in old_summary and old_summary[k] != v]
    if mism:
        raise Refusal('REFUSING: the stored files do not give the facts of the record; '
                      'nothing was written: ' + '; '.join(mism))
    before = {'valid': rec.get('valid'), 'reasons': first_reasons}
    first = rec.get('rejudged') or before
    busy = (rec.get('thresholds') or {}).get('foreign_cpu_busy_pct')
    new = dict(rec)
    new['samples'] = {**ssum, 'file': old_summary.get('file', files['samples']),
                      'interval_s': interval, 'pre_window_s': old_summary.get('pre_window_s')}
    new['thresholds'] = thresholds(_busy_pct() if busy is None else busy)
    new['valid'], new['reasons'] = valid, reasons
    new['rejudged'] = {'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                       'first_record': os.path.relpath(first_record_path(path), root),
                       'valid': first.get('valid'), 'reasons': list(first.get('reasons') or [])}
    return new, first, before


def _say_verdict(label, valid, reasons):
    print(f'  {label} valid={valid}', flush=True)
    for r in reasons or []:
        print(f'    INVALID: {r}', flush=True)


def cmd_rejudge(a):
    root, path = os.path.abspath(a.root), os.path.abspath(a.record)
    new, first, before = rejudge_record(root, path)
    rel = os.path.relpath(path, root)
    print(f'[training-cost] rejudge {rel}'
          + (' (DRY: nothing is written)' if a.dry else ''), flush=True)
    _say_verdict('first verdict:          ', first.get('valid'), first.get('reasons'))
    if before is not first:
        _say_verdict('verdict before this call:', before['valid'], before['reasons'])
    s = new['samples']
    pct = lambda x: 'none measured' if x is None else f'{x:g}%'
    print(f"  GPU use of the other processes: {pct(s['foreign_gpu_util_mean'])} on average "
          f'over the run (limit {FOREIGN_UTIL_MEAN_MAX:.1f}%), largest sample '
          f"{pct(s['foreign_gpu_util_max'])}, above {FOREIGN_UTIL_MAX:g}% in "
          f"{s['foreign_gpu_samples_above_limit']} of {s['run']} run samples", flush=True)
    _say_verdict('new verdict:            ', new['valid'], new['reasons'])
    if a.dry:
        print('[training-cost] DRY: nothing written', flush=True)
        return 0 if new['valid'] else 1
    first_path = first_record_path(path)
    if not os.path.exists(first_path):
        shutil.copy2(path, first_path)             # once, with its time; never overwritten
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(new, f, indent=1)
    os.replace(tmp, path)
    print(f'[training-cost] wrote {rel}; the record as the run wrote it is '
          f'{os.path.relpath(first_path, root)}', flush=True)
    return 0 if new['valid'] else 1


def home_masked(s):
    """A path under the user's home directory written from '~', so that no record holds
    the home folder of this machine."""
    home = os.path.expanduser('~')
    s = str(s)
    return '~' + s[len(home):] if home not in ('', '/') and s.startswith(home + os.sep) else s


def _busy_pct():
    import p75_deploy_cost as P75
    return P75.BUSY_PCT


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('command', choices=('plan', 'sample', 'run', 'rejudge'))
    ap.add_argument('record', nargs='?', default='',
                    help='rejudge: the record <root>/train_log/training_cost/<name>.json')
    ap.add_argument('--dry', action='store_true',
                    help='rejudge: print the first and the new verdict, write nothing')
    ap.add_argument('--arm', default='released')
    ap.add_argument('--combo', default='')
    ap.add_argument('--seed', type=int, default=301)
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--cores', default='4-7')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--window', type=int, default=WINDOW_DEFAULT)
    ap.add_argument('--interval', type=int, default=INTERVAL_DEFAULT)
    ap.add_argument('--py', default=sys.executable)
    ap.add_argument('--root', default=ROOT, help='tests only: a root other than the repository')
    ap.add_argument('--stub_trainer', default='', help='tests only, with --root')
    ap.add_argument('--allow_busy', action='store_true',
                    help='tests only, with --stub_trainer: start although the sample before '
                         'the start is not idle')
    a = ap.parse_args(argv)
    if a.window < 1 or a.interval < 3:
        ap.error('window must be at least 1 s and interval at least 3 s')
    if a.command == 'rejudge' and not a.record:
        ap.error('rejudge needs the record: rejudge <record.json> [--dry]')
    if a.command != 'rejudge' and (a.record or a.dry):
        ap.error(f'a record and --dry belong to rejudge, not to {a.command}')
    return a


def main(argv=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''     # this process starts no CUDA context
    try:
        return {'plan': cmd_plan, 'sample': cmd_sample, 'run': cmd_run,
                'rejudge': cmd_rejudge}[a.command](a)
    except Refusal as e:
        print(e, file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
