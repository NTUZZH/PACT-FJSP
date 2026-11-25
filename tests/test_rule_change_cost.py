"""Tests for the tools of the question "What does a rule change cost in practice?":
scripts/run_training_cost.sh with scripts/training_cost_evidence.py (one training run with
the GPU and its cores to itself, and its evidence record) and scripts/p99_rule_change_cost.py
(checks, count, infer).

No test starts a GPU process. The launcher is tested in its DRY and SAMPLE_ONLY modes, and
end to end with a stand-in trainer under a temporary root, whose records are always invalid.
The record validity rules are tested on synthetic samples, logs and configurations, the
sampler's own helper process with a stand-in pmon call and core sampler, and rejudge on
stored records built under a temporary root.
Read-only nvidia-smi queries run in the sampling tests. Every output goes to a temporary
folder. Run from the repository root on one core:
  taskset -c 3 python -m pytest -p no:cacheprovider --basetemp=<scratch> tests/test_rule_change_cost.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import types

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import training_cost_evidence as TCE  # noqa: E402

PY = sys.executable
LAUNCHER = os.path.join(ROOT, 'scripts', 'run_training_cost.sh')
P99 = os.path.join(ROOT, 'scripts', 'p99_rule_change_cost.py')
R8 = list(REG.COMBOS)
CORE = str(sorted(os.sched_getaffinity(0))[0])     # the core the tests were pinned to
LEGACY_SPEC_CFG = os.path.join(ROOT, 'train_log', 'FAMILY', 'config_10x25+family+spec-N-s303.json')
LEGACY_POL_CFG = os.path.join(ROOT, 'train_log', 'FAMILY',
                              'config_10x25+family+jointblind-v1-s303.json')

STUB = '''"""Stand-in trainer: the banners and files of train_family.py, no training."""
import json, os, sys, time
args = sys.argv[1:]
def opt(k):
    return args[args.index(k) + 1]
name, upd = opt('--model_suffix'), int(opt('--max_updates'))
ref = [f for f in os.listdir('train_log/FAMILY') if f.startswith('config_') and 'xmask' in f][0]
cfg = json.load(open(os.path.join('train_log/FAMILY', ref)))
cfg['model_suffix'] = name
if os.environ.get('STUB_DRIFT'):
    cfg['lr'] = 0.001
print('[train_family] MASK: first_mask=exact mask_ties=exact', flush=True)
print('[model] params total=32786  film=0 (0.00%)', flush=True)
json.dump(cfg, open(f'train_log/FAMILY/config_10x25+family+{name}.json', 'w'), indent=1)
print(f"[train_family] model=10x25+family+{name} regimes=['N'] updates={upd}", flush=True)
for i in range(upd):
    time.sleep(float(os.environ.get('STUB_SLEEP', '0.5')))
    print(f'Ep {i + 1}\\t reward -1.000\\t ms 100.0\\t loss 0.100000\\t 0.5s', flush=True)
open(f'trained_network/FAMILY/10x25+family+{name}.pth', 'wb').write(b'x')
print('[train_family] done in 0.00 h; best rel=0.9000', flush=True)
'''


def env_for(**kw):
    e = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
             PY=PY)
    for k in ('PACT_ENV', 'PACT_OUT', 'PACT_SEEDS', 'PACT_RESULTS', 'ROOT_DIR', 'STUB_TRAINER',
              'ALLOW_BUSY', 'DRY', 'SAMPLE_ONLY', 'ARM', 'COMBO', 'STUB_DRIFT', 'STUB_SLEEP'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kw.items()})
    return e


def run(cmd, timeout=300, **kw):
    return subprocess.run(cmd, cwd=ROOT, env=env_for(**kw), capture_output=True, text=True,
                          timeout=timeout)


# ================================================================== synthetic inputs
def log_text(params=32786, mask=True, model=True, done=True, updates=3, name='xcost-solo-N-s301-r1'):
    lines = []
    if mask:
        lines.append('[train_family] MASK: first_mask=exact mask_ties=exact')
    lines.append('[vali] MWKR norms: N=170.0')
    if params is not None:
        lines.append(f'[model] params total={params}  film=0 (0.00%)')
    if model:
        lines.append(f"[train_family] model=10x25+family+{name} regimes=['N'] updates={updates}")
    lines += [f'Ep {i + 1}\t reward -1.000\t ms 100.0\t loss 0.100000\t 2.7s'
              for i in range(updates)]
    if done:
        lines.append('[train_family] done in 0.00 h; best rel=0.9000')
    return '\n'.join(lines) + '\n'


def configs(updates=3):
    ref = json.load(open(LEGACY_SPEC_CFG))
    ref.update(model_suffix='xmask-solo-N-s301', seed_train=301, first_mask='exact',
               mask_ties='exact', max_updates=updates)
    new = dict(ref, model_suffix='xcost-solo-N-s301-r1')
    return ref, new


def sample(phase='run', foreign_sm=0.0, compute=(), cpu_n=0, cpu_pct=0.0, blocks=1, t='t'):
    return {'phase': phase, 'time': t, 't_rel_s': 0.0, 'window_s': 58, 'gpu_blocks': blocks,
            'gpu_foreign_sm_max': foreign_sm, 'gpu_foreign_compute': list(compute),
            'gpu_memory_holders': ['4242:13816MB:python'], 'gpu_own_sm_max': 40.0,
            'cpu_foreign_n': cpu_n, 'cpu_foreign_pct_max': cpu_pct,
            'cpu_foreign': ['pid 99 core 5 40% (lifetime): x'] * cpu_n, 'cpu_kernel_threads': [],
            'loadavg1': 1.0, 'trainer_alive': True}


def samples_file(tmp_path, rows):
    p = tmp_path / 'samples.csv'
    TCE.write_samples(str(p), rows)
    return TCE.read_samples(str(p))


def assess(tmp_path, rows=None, wall=150.0, interval=60, ref_new=None, text=None, code=0,
           ckpt=True, **kw):
    ref, new = ref_new or configs()
    rows = rows if rows is not None else [sample('pre'), sample(), sample(), sample()]
    return TCE.assess_run(campaign='xmask-solo-N-s301', name='xcost-solo-N-s301-r1',
                          ref_cfg=ref, new_cfg=new, log_text=text or log_text(),
                          exit_code=code, wall_seconds=wall,
                          samples=samples_file(tmp_path, rows), interval=interval,
                          checkpoint_exists=ckpt, **kw)


# ================================================================== record validity
def test_clean_record_is_valid(tmp_path):
    valid, reasons, facts, summ = assess(tmp_path)
    assert valid and not reasons
    assert facts['params'] == 32786 and facts['mask'] == ('exact', 'exact')
    assert facts['updates_logged'] == 3 and facts['update_seconds_sum'] == pytest.approx(8.1)
    assert summ['run'] == 3 and summ['pre'] == 1
    assert summ['gpu_memory_holders'] == ['4242:13816MB:python']   # recorded, allowed


def test_foreign_gpu_compute_invalidates(tmp_path):
    rows = [sample('pre'), sample(), sample(foreign_sm=3.0, compute=['777:3%:python']), sample()]
    valid, reasons, _, summ = assess(tmp_path, rows)
    assert not valid and any('other GPU compute 777:3%:python' in r for r in reasons)
    assert summ['foreign_gpu_util_max'] == 3.0


def run_rows(n=88, high=(), level=7.0):
    """The sample before the start and n run samples r0 ... r<n-1>; the run samples whose
    index is in `high` show `level` percent SM of the other processes (display processes)."""
    return [sample('pre')] + [sample(foreign_sm=level if i in high else 0.0, t=f'r{i}')
                              for i in range(n)]


def test_one_display_second_in_88_run_samples_leaves_the_record_valid(tmp_path):
    valid, reasons, _, summ = assess(tmp_path, run_rows(88, high={40}), wall=88 * 60.0)
    assert valid and not reasons, reasons
    assert summ['foreign_gpu_util_mean'] == round(7.0 / 88, 4) == 0.0795
    assert summ['foreign_gpu_samples_above_limit'] == 1
    assert summ['foreign_gpu_util_max'] == 7.0


def test_many_run_samples_at_ten_percent_fail_with_the_average(tmp_path):
    valid, reasons, _, summ = assess(tmp_path, run_rows(88, high=set(range(12)), level=10.0),
                                     wall=88 * 60.0)
    assert not valid and summ['foreign_gpu_util_mean'] == round(120.0 / 88, 4)
    assert reasons == ['other processes used 1.364% of the GPU on average over the run '
                       '(limit 1.0%), above 5% in 12 of 88 samples']
    # at the limit: eight samples at 11% average to 1.0%, and the record stays valid
    valid, reasons, _, summ = assess(tmp_path, run_rows(88, high=set(range(8)), level=11.0),
                                     wall=88 * 60.0)
    assert valid and not reasons and summ['foreign_gpu_util_mean'] == 1.0
    assert summ['foreign_gpu_samples_above_limit'] == 8


def test_the_run_average_weights_each_sample_by_its_window(tmp_path):
    rows = [sample('pre'), dict(sample(foreign_sm=3.0), window_s=10), dict(sample(), window_s=50)]
    valid, reasons, _, summ = assess(tmp_path, rows)       # 3 x 10 / 60; unweighted 1.5
    assert valid and summ['foreign_gpu_util_mean'] == 0.5
    rows = [sample('pre'), sample(foreign_sm=1.2), sample(blocks=0, t='lost')]
    valid, reasons, _, summ = assess(tmp_path, rows)       # a sample without GPU rows is left out
    assert summ['foreign_gpu_util_mean'] == 1.2
    assert reasons == ['run sample at lost: the GPU sample failed (no pmon rows)',
                       'other processes used 1.2% of the GPU on average over the run '
                       '(limit 1.0%), above 5% in 0 of 2 samples']


def test_other_compute_in_one_run_sample_of_88_fails(tmp_path):
    rows = run_rows(88)
    rows[30] = sample(foreign_sm=1.0, compute=['777:1%:python'], t='r29')
    valid, reasons, _, summ = assess(tmp_path, rows, wall=88 * 60.0)
    assert not valid and reasons == ['run sample at r29: other GPU compute 777:1%:python']
    assert summ['foreign_gpu_util_mean'] < TCE.FOREIGN_UTIL_MEAN_MAX


def test_the_sample_before_the_start_keeps_the_per_sample_limit(tmp_path, monkeypatch):
    rows = [sample('pre', foreign_sm=7.0)] + run_rows(3)[1:]
    valid, reasons, _, _ = assess(tmp_path, rows)
    assert not valid
    assert reasons == ['pre sample at t: other processes use 7% of the GPU (limit 5%)']
    assert TCE.sample_problems(sample('pre', foreign_sm=5.0)) == []      # at the limit
    assert TCE.sample_problems(sample('run', foreign_sm=7.0)) == []      # judged by the average
    # the start is refused and nothing is written
    root, stub = campaign_root(tmp_path)
    monkeypatch.delenv('PACT_ENV', raising=False)
    monkeypatch.setattr(TCE, 'take_sample', lambda *a, **k: sample('pre', foreign_sm=7.0))
    a = TCE.parse(['run', '--arm', 'specialist', '--combo', 'N', '--root', str(root),
                   '--stub_trainer', str(stub), '--cores', CORE, '--threads', '1',
                   '--window', '1', '--interval', '3'])
    with pytest.raises(TCE.Refusal, match='not idle'):
        TCE.cmd_run(a)
    assert not (root / TCE.RECORD_DIR).exists()


def test_foreign_cpu_invalidates(tmp_path):
    rows = [sample('pre'), sample(), sample(cpu_n=1, cpu_pct=40.0), sample()]
    valid, reasons, _, summ = assess(tmp_path, rows)
    assert not valid and any('other process(es) on the reserved cores' in r for r in reasons)
    assert summ['foreign_cpu_pct_max'] == 40.0


def test_busy_cores_before_the_start_invalidate(tmp_path):
    rows = [sample('pre', cpu_n=2, cpu_pct=90.0), sample(), sample(), sample()]
    valid, reasons, _, _ = assess(tmp_path, rows)
    assert not valid and any(r.startswith('pre sample') for r in reasons)


def test_failed_gpu_sample_invalidates(tmp_path):
    rows = [sample('pre'), sample(blocks=0), sample(), sample()]
    valid, reasons, _, _ = assess(tmp_path, rows)
    assert not valid and any('GPU sample failed' in r for r in reasons)


def test_too_few_samples_invalidate(tmp_path):
    valid, reasons, _, _ = assess(tmp_path, wall=600.0)        # 3 samples, 9 needed
    assert not valid and any('9 needed' in r for r in reasons)
    valid, reasons, _, _ = assess(tmp_path, rows=[sample(), sample()])
    assert not valid and 'no sample before the start' in reasons


@pytest.mark.parametrize('field,value', [('lr', 0.001), ('device', 'cpu'), ('device_id', '1'),
                                         ('num_envs', 10), ('first_mask', 'legacy')])
def test_configuration_difference_invalidates(tmp_path, field, value):
    ref, new = configs()
    new[field] = value
    valid, reasons, _, _ = assess(tmp_path, ref_new=(ref, new))
    assert not valid and any(f'configuration field {field}' in r for r in reasons)


def test_only_the_name_may_differ(tmp_path):
    ref, new = configs()
    assert TCE.config_problems(ref, new, 'xmask-solo-N-s301', 'xcost-solo-N-s301-r1') == []
    assert TCE.config_problems(ref, new, 'xmask-solo-N-s301', 'xcost-solo-N-s301-r2')
    extra = dict(new, dr_shift='')                   # a field the campaign run lacks
    assert TCE.config_problems(ref, extra, 'xmask-solo-N-s301', 'xcost-solo-N-s301-r1')


@pytest.mark.parametrize('missing,reason', [
    ('done', "no '[train_family] done in' line"),
    ('params', "no '[model] params total=' line"),
    ('mask', "no '[train_family] MASK:' line"),
    ('model', "no '[train_family] model=' line")])
def test_missing_banner_invalidates(tmp_path, missing, reason):
    kw = {'params': None} if missing == 'params' else {missing: False}
    valid, reasons, _, _ = assess(tmp_path, text=log_text(**kw))
    assert not valid and any(reason in r for r in reasons)


def test_wrong_parameter_count_and_missing_updates_invalidate(tmp_path):
    valid, reasons, _, _ = assess(tmp_path, text=log_text(params=29602))
    assert not valid and any('29602 parameters' in r for r in reasons)
    valid, reasons, _, _ = assess(tmp_path, text=log_text(updates=2))
    assert not valid and any(r.startswith('updates:') for r in reasons)
    valid, reasons, _, _ = assess(tmp_path, code=1, ckpt=False)
    assert not valid and 'trainer exit status 1' in reasons
    assert 'the trainer saved no checkpoint' in reasons


def test_stub_and_stopped_runs_are_never_valid(tmp_path):
    valid, reasons, _, _ = assess(tmp_path, stub='/x/stub.py')
    assert not valid and 'stub trainer' in reasons[0]
    valid, reasons, _, _ = assess(tmp_path, abort_reason='configuration drift: lr')
    assert not valid and 'stopped by the launcher' in reasons[0]


# ================================================================== parsing the samplers
PMON = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0       2062     G      3      -      -      -      -      -     58      0    gnome-shell
    0       4242     C      -      -      -      -      -      -  13816      0    python
    0       5000     C     41     10      -      -      -      -    300      0    python
    0       2062     G      -      -      -      -      -      -     58      0    gnome-shell
    0       4242     C      2      1      -      -      -      -  13816      0    python
    0       5000     C     38     10      -      -      -      -    300      0    python
"""


def test_pmon_blocks_and_gpu_summary():
    blocks = TCE.parse_pmon(PMON)
    assert len(blocks) == 2 and [r['pid'] for r in blocks[0]] == ['2062', '4242', '5000']
    s = TCE.gpu_summary(blocks, own_pids={5000})
    assert s['gpu_own_sm_max'] == 41.0
    assert s['gpu_foreign_sm_max'] == 3.0                  # display 3 + idle holder 0
    assert s['gpu_foreign_compute'] == ['4242:2%:python']   # sm 2 in the second sample
    assert s['gpu_memory_holders'] == ['4242:13816MB:python']
    assert TCE.parse_pmon('') == []


def test_busy_lines_are_split_by_owner():
    entries = ['pid 10 thread 11 core 5 30% over 58 s: python train_family.py',
               'pid 20 core 6 12% (lifetime): /usr/bin/foo',
               'pid 30 thread 30 core 4 9% over 58 s: [irq/150-nvidia]',
               'something else']
    table = {10: 1, 20: 1, 30: 2}
    foreign, kernel = TCE.split_busy(entries, own={10}, table=table)
    assert [p for p, _ in foreign] == [12.0, 100.0]
    assert [p for p, _ in kernel] == [9.0]
    s = TCE.cpu_summary(foreign, kernel)
    assert s['cpu_foreign_n'] == 2 and s['cpu_foreign_pct_max'] == 100.0


def test_descendants_cover_the_trainer_tree():
    table = {1: 0, 50: 1, 51: 50, 52: 51, 60: 1}
    assert TCE.descendants(table, [50]) == {50, 51, 52}
    assert TCE.descendants(table, [None, 60]) == {60}


PMON_DISPLAY = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa     fb   ccpm    command
# Idx           #    C/G      %      %      %      %      %      %     MB     MB    name
    0       2062     G      0      -      -      -      -      -     58      0    gnome-shell
"""


@pytest.mark.parametrize('other', [False, True])
def test_the_sampler_helper_is_not_another_process(monkeypatch, other):
    """The pmon call of a sample has ended when the process table is read, so the table no
    longer lists it; its pid still belongs to the launcher, and another busy process on the
    reserved cores still counts."""
    started = []

    class Helper:                      # stands in for subprocess.Popen of the pmon call
        def __init__(self, cmd, **kw):
            self.pid, self.returncode = 900001 + len(started), 0
            started.append((self.pid, cmd))

        def communicate(self, timeout=None):
            return PMON_DISPLAY, None

        def kill(self):
            pass

    def busy(cores, window):           # stands in for p75's core sampler
        lines = [f'pid {pid} core {cores[0]} 5% (lifetime): nvidia-smi pmon -s um -c 30 -d 1'
                 for pid, _ in started]
        if other:
            lines.append(f'pid 900100 core {cores[0]} 40% (lifetime): python other.py')
        return lines

    monkeypatch.setitem(sys.modules, 'p75_deploy_cost',
                        types.SimpleNamespace(busy_processes=busy, loadavg=lambda: (0.5,) * 3))
    monkeypatch.setattr(TCE.subprocess, 'Popen', Helper)
    monkeypatch.setattr(TCE, 'proc_table', lambda: {os.getpid(): 1, 900100: 1})
    row = TCE.take_sample([int(CORE)], 30, 'pre', gpu_count=30)
    assert [cmd for _, cmd in started] == [TCE.pmon_cmd(30)]
    assert not any('nvidia-smi' in x for x in row['cpu_foreign'])
    if other:
        assert row['cpu_foreign_n'] == 1 and 'other.py' in row['cpu_foreign'][0]
        assert any('other process(es) on the reserved cores' in x
                   for x in TCE.sample_problems(row))
    else:
        assert row['cpu_foreign_n'] == 0 and TCE.sample_problems(row) == []


def test_check_record_rules():
    rec = {'kind': 'training_cost', 'valid': True, 'stub': None,
           'files': {'log': 'train_log/training_cost/xcost-v1-s301-r1.log'},
           'repeats': {'arm': 'released'}, 'wall_seconds': 5400.0,
           'seconds_per_update': 2.7, 'updates': 2000}
    assert TCE.check_record(rec) == []
    assert TCE.check_record(dict(rec, valid=False, reasons=['x']))
    assert TCE.check_record(dict(rec, stub='/x'))
    assert TCE.check_record(dict(rec, files={'log': 'train_log/FAMILY_xmask-v1-s301.log'}))
    assert TCE.check_record(dict(rec, kind='other'))
    assert TCE.check_record(dict(rec, repeats={'arm': 'control'}))


def test_names_and_core_lists():
    assert TCE.run_names('released', '', 301, 1) == ('xmask-v1-s301', 'xcost-v1-s301-r1')
    assert TCE.run_names('specialist', 'LSW', 303, 2) == ('xmask-solo-LSW-s303', 'xcost-solo-LSW-s303-r2')
    for bad in (('control', '', 301, 1), ('released', 'N', 301, 1), ('released', '', 309, 1),
                ('released', '', 301, 0)):
        with pytest.raises(TCE.Refusal):
            TCE.run_names(*bad)
    assert TCE.parse_cores('4-7') == [4, 5, 6, 7] and TCE.parse_cores('3') == [3]
    assert TCE.home_masked(os.path.expanduser('~/a/b')) == '~/a/b'


def new_run_paths(name):
    f = TCE.new_files(name)
    final = [f['log'], f['samples'], f['record']] + [
        f"{f['archive']}/{os.path.basename(f[k])}" for k in TCE.TRAINER_FILES]
    return final


def script_globs():
    """Wildcard patterns over train_log/ and trained_network/ in the code base (Python
    and shell), with f-string and shell placeholders read as '*'. A pattern of a find
    exclusion (-not -path) selects nothing and is left out."""
    rx = re.compile(r"""(?:train_log|trained_network)[^\s'"`),\]]*\*[^\s'"`),\]]*""")
    pats = set()
    own = ('p99_rule_change_cost.py', 'training_cost_evidence.py', 'run_training_cost.sh')
    for p in glob.glob(os.path.join(ROOT, 'scripts', '*.py')) + \
            glob.glob(os.path.join(ROOT, 'scripts', '*.sh')) + glob.glob(os.path.join(ROOT, '*.py')):
        if os.path.basename(p) in own:       # the writer and the reader of the new files
            continue
        for line in open(p, errors='replace'):
            if '-not -path' in line:
                continue
            for m in rx.findall(line):
                pats.add(re.sub(r'\$\{[^}]*\}|\$[A-Za-z_]+|\{[^}]*\}', '*', m))
    return sorted(pats)


def test_new_run_files_escape_every_glob(tmp_path):
    pats = script_globs()
    assert any(p.startswith('train_log/FAMILY') for p in pats)   # the extraction finds real patterns
    names = ['xcost-v1-s301-r1'] + [f'xcost-solo-{R}-s301-r1' for R in R8]
    for n in names:
        for rel in new_run_paths(n):
            q = tmp_path / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_text('x')
    hits = {p: glob.glob(str(tmp_path / p)) for p in pats}
    assert not {p: h for p, h in hits.items() if h}
    users = [p for p in glob.glob(os.path.join(ROOT, 'scripts', '*'))
             if os.path.isfile(p) and 'xcost' in open(p, errors='replace').read()]
    assert sorted(os.path.basename(p) for p in users) == ['run_training_cost.sh',
                                                          'training_cost_evidence.py']


# ================================================================== the launcher
def campaign_root(tmp_path, updates=3):
    root = tmp_path / 'root'
    (root / 'train_log' / 'FAMILY').mkdir(parents=True)
    (root / 'trained_network' / 'FAMILY').mkdir(parents=True)
    ref, _ = configs(updates)
    json.dump(ref, open(root / 'train_log' / 'FAMILY' / 'config_10x25+family+xmask-solo-N-s301.json', 'w'))
    (root / 'trained_network' / 'FAMILY' / '10x25+family+xmask-solo-N-s301.pth').write_bytes(b'x')
    (root / 'train_log' / 'FAMILY_xmask-solo-N-s301.log').write_text(
        '[train_family] done in 0.70 h; best rel=0.8000\n')
    stub = tmp_path / 'stub_trainer.py'
    stub.write_text(STUB)
    return root, stub


def launcher(**kw):
    return run(['taskset', '-c', CORE, LAUNCHER], CORES=CORE, **kw)


def tree(path):
    return sorted(os.path.relpath(p, path) for p in glob.glob(os.path.join(path, '**'), recursive=True))


def test_dry_run_in_the_repository_writes_nothing():
    before = tree(os.path.join(ROOT, TCE.RECORD_DIR))
    r = launcher(ARM='released', DRY=1)
    assert r.returncode in (0, 3), r.stderr
    out = r.stdout
    assert 'DRY RUN: nothing is started and nothing is written' in out
    assert 'xcost-v1-s301-r1' in out and 'repeats          xmask-v1-s301' in out
    assert '--first_mask exact --mask_ties exact' in out
    assert 'train_log/training_cost/xcost-v1-s301-r1.json' in out
    if not os.path.isfile(os.path.join(ROOT, 'train_log/FAMILY/config_10x25+family+xmask-v1-s301.json')):
        assert r.returncode == 3 and 'FAIL campaign configuration' in out
    if os.path.exists(os.path.join(ROOT, TCE.HOLD)):
        assert 'FAIL train_log/GPU_HOLD absent' in out
    assert tree(os.path.join(ROOT, TCE.RECORD_DIR)) == before


def test_dry_run_reads_the_campaign_configuration(tmp_path):
    root, _ = campaign_root(tmp_path)
    r = launcher(ARM='specialist', COMBO='N', DRY=1, ROOT_DIR=root)
    assert r.returncode == 0, r.stdout + r.stderr
    assert '--family_regimes N --max_updates 3 --seed_train 301 --first_mask exact' in r.stdout
    assert 'DRY: every precondition holds' in r.stdout
    assert tree(root / 'train_log') == tree(campaign_root(tmp_path / 'again')[0] / 'train_log')


def test_launcher_refusals(tmp_path):
    root, stub = campaign_root(tmp_path)
    (root / 'train_log' / 'GPU_HOLD').write_text('')
    r = launcher(ARM='specialist', COMBO='N', ROOT_DIR=root, STUB_TRAINER=stub, ALLOW_BUSY=1,
                 WINDOW=1, INTERVAL=3)
    assert r.returncode == 2 and 'GPU_HOLD absent' in r.stderr + r.stdout
    assert not (root / 'train_log' / 'training_cost').exists()
    (root / 'train_log' / 'GPU_HOLD').unlink()
    r = launcher(ARM='specialist', COMBO='N', ROOT_DIR=root, ALLOW_BUSY=1, WINDOW=1, INTERVAL=3)
    assert r.returncode == 2 and 'stub trainer only' in r.stderr
    r = launcher(ARM='control', DRY=1, ROOT_DIR=root)
    assert r.returncode == 2 and 'REFUSING' in r.stderr
    r = launcher(ARM='released', DRY=1, PACT_ENV='legacy')
    assert r.returncode == 1 and 'exact-mask runs' in r.stdout
    r = launcher(ARM='specialist', COMBO='N', ROOT_DIR=ROOT, STUB_TRAINER=stub, ALLOW_BUSY=1,
                 WINDOW=1, INTERVAL=3)
    assert r.returncode == 2 and 'never in the repository' in r.stderr


def test_end_to_end_with_a_stub_trainer(tmp_path):
    root, stub = campaign_root(tmp_path)
    r = launcher(ARM='specialist', COMBO='N', ROOT_DIR=root, STUB_TRAINER=stub, ALLOW_BUSY=1,
                 WINDOW=1, INTERVAL=3, STUB_SLEEP=1)
    assert r.returncode == 1, r.stdout + r.stderr          # a stub record is invalid
    name = 'xcost-solo-N-s301-r1'
    rec = json.load(open(root / TCE.RECORD_DIR / f'{name}.json'))
    assert rec['valid'] is False and 'stub trainer' in rec['reasons'][0]
    assert rec['repeats']['run'] == 'xmask-solo-N-s301' and rec['exit_status'] == 0
    assert rec['params_total'] == 32786 and rec['mask'] == ['exact', 'exact']
    assert rec['updates'] == 3 and rec['episodes'] == 60 and rec['seconds_per_update'] == 0.5
    assert rec['config_check']['differences'] == [['model_suffix', 'xmask-solo-N-s301', name]]
    assert rec['wall_seconds'] >= 3.0 and rec['done_in_hours'] == 0.0
    assert rec['host']['gpu'] and rec['host']['torch'] and rec['allow_busy'] is True
    assert not any(x.startswith(os.path.expanduser('~') + os.sep) for x in rec['command'][:1])
    # the trainer's files left the globbed folders
    assert not glob.glob(str(root / 'trained_network' / 'FAMILY' / f'*{name}*'))
    assert not glob.glob(str(root / 'train_log' / 'FAMILY' / f'*{name}*'))
    assert sorted(os.listdir(root / TCE.RECORD_DIR / name)) == [
        f'10x25+family+{name}.pth', f'config_10x25+family+{name}.json']
    rows = TCE.read_samples(str(root / TCE.RECORD_DIR / f'{name}.samples.csv'))
    assert rows[0]['phase'] == 'pre' and sum(r['phase'] == 'run' for r in rows) >= 1
    assert TCE.check_record(rec)                            # never usable
    assert rec['thresholds']['foreign_gpu_util_mean_max_pct'] == TCE.FOREIGN_UTIL_MEAN_MAX
    assert rec['thresholds']['foreign_gpu_util_pre_max_pct'] == TCE.FOREIGN_UTIL_MAX
    assert {'foreign_gpu_util_mean', 'foreign_gpu_samples_above_limit',
            'foreign_gpu_util_max'} <= set(rec['samples'])
    data = (root / TCE.RECORD_DIR / f'{name}.json').read_bytes()
    r = rejudge(root, root / TCE.RECORD_DIR / f'{name}.json')
    assert r.returncode == 2 and 'stub trainer' in r.stderr
    assert (root / TCE.RECORD_DIR / f'{name}.json').read_bytes() == data
    assert not (root / TCE.RECORD_DIR / f'{name}{TCE.FIRST_SUFFIX}').exists()
    r = launcher(ARM='specialist', COMBO='N', DRY=1, ROOT_DIR=root)
    assert r.returncode == 3 and 'FAIL new name xcost-solo-N-s301-r1 unused' in r.stdout


def test_a_drifted_configuration_stops_the_trainer(tmp_path):
    root, stub = campaign_root(tmp_path, updates=40)
    r = launcher(ARM='specialist', COMBO='N', ROOT_DIR=root, STUB_TRAINER=stub, ALLOW_BUSY=1,
                 WINDOW=1, INTERVAL=3, STUB_DRIFT=1, STUB_SLEEP=1)
    assert r.returncode == 1
    rec = json.load(open(root / TCE.RECORD_DIR / 'xcost-solo-N-s301-r1.json'))
    assert any(x.startswith('stopped by the launcher: configuration drift') for x in rec['reasons'])
    assert rec['exit_status'] != 0 and rec['wall_seconds'] < 30
    ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    assert str(stub) not in ps


# ================================================================== rejudge
NAME = 'xcost-solo-N-s301-r1'
GPU_SAMPLE_REASON = 'run sample at r40: other processes use 7% of the GPU (limit 5%)'
EVIDENCE = os.path.join(ROOT, 'scripts', 'training_cost_evidence.py')


def rejudge(root, record, *extra):
    return run([PY, '-B', EVIDENCE, 'rejudge', str(record), '--root', str(root), *extra])


def stored_record(tmp_path, rows=None, reasons=None, code=0, wall=88 * 60.0):
    """A stored record of a specialist run with its files under a test root: the campaign
    configuration, the log, the samples file and the trainer's files in the archive folder.
    The facts are those of the stored files, the sample summary holds the entries read from
    the samples file, and the verdict is invalid with the given reasons (by default one run
    sample above the per-sample GPU limit)."""
    root = tmp_path / 'root'
    rows = run_rows(88, high={40}) if rows is None else rows
    ref, new = configs()
    (root / 'train_log' / 'FAMILY').mkdir(parents=True)
    campaign_cfg = 'train_log/FAMILY/config_10x25+family+xmask-solo-N-s301.json'
    json.dump(ref, open(root / campaign_cfg, 'w'))
    f = TCE.new_files(NAME)
    (root / f['archive']).mkdir(parents=True)
    text = log_text()
    (root / f['log']).write_text(text)
    TCE.write_samples(str(root / f['samples']), rows)
    moved = {}
    for k in TCE.TRAINER_FILES:
        rel = f"{f['archive']}/{os.path.basename(f[k])}"
        (root / rel).write_text(json.dumps(new) if k == 'config' else 'x')
        moved[k] = rel
    facts = TCE.log_facts(text)
    _, summ = TCE.assess_samples(TCE.read_samples(str(root / f['samples'])), wall, 60)
    summ = {k: v for k, v in summ.items() if k not in TCE.RULE_SUMMARY[1:]}
    rec = {'kind': 'training_cost', 'generator': 'scripts/run_training_cost.sh', 'name': NAME,
           'environment': 'exact',
           'repeats': {'arm': 'specialist', 'combination': 'N', 'seed': 301,
                       'run': 'xmask-solo-N-s301', 'config': campaign_cfg},
           'attempt': 1, 'host': {'gpu': 'NVIDIA RTX PRO 5000 Blackwell'},
           'wall_seconds': wall, 'hours': round(wall / 3600.0, 6),
           'done_in_hours': facts['done_in_h'], 'best_rel': facts['best_rel'],
           'updates': int(new['max_updates']), 'num_envs': int(new['num_envs']),
           'episodes': int(new['max_updates']) * int(new['num_envs']),
           'update_seconds_sum': facts['update_seconds_sum'],
           'seconds_per_update': round(facts['update_seconds_sum'] / int(new['max_updates']), 4),
           'params_total': facts['params'], 'mask': list(facts['mask']), 'exit_status': code,
           'config_check': {'reference': campaign_cfg,
                            'differences': [list(d) for d in TCE.config_diff(ref, new)],
                            'allowed': list(TCE.ALLOWED_DIFF)},
           'samples': {**summ, 'file': f['samples'], 'interval_s': 60, 'pre_window_s': 30},
           'thresholds': {'foreign_gpu_util_max_pct': 5.0, 'foreign_compute_sm_max_pct': 0.0,
                          'foreign_cpu_busy_pct': 5.0,
                          'samples_needed': 'wall seconds / interval - 1, at least 1'},
           'files': {'log': f['log'], 'samples': f['samples'], **moved},
           'stub': None, 'allow_busy': False, 'valid': False,
           'reasons': [GPU_SAMPLE_REASON] if reasons is None else list(reasons)}
    p = root / f['record']
    with open(p, 'w') as fh:
        json.dump(rec, fh, indent=1)
    return root, p


def test_rejudge_makes_a_record_that_failed_one_gpu_sample_valid(tmp_path):
    root, p = stored_record(tmp_path)
    first_bytes = p.read_bytes()
    first = json.loads(first_bytes)
    out = tmp_path / 'out'
    (out / 'results').mkdir(parents=True)
    json.dump(count_record(), open(out / 'results' / 'rule_change_count.json', 'w'))
    r = infer(root, out)
    assert r.returncode == 2 and 'record is invalid' in r.stderr
    r = rejudge(root, p)
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.search(r'first verdict: +valid=False', r.stdout) and GPU_SAMPLE_REASON in r.stdout
    assert re.search(r'new verdict: +valid=True', r.stdout)
    assert '0.0795% on average over the run (limit 1.0%)' in r.stdout
    fp = root / TCE.RECORD_DIR / f'{NAME}{TCE.FIRST_SUFFIX}'
    assert fp.read_bytes() == first_bytes                   # the record as first written
    rec = json.load(open(p))
    assert rec['valid'] is True and rec['reasons'] == []
    assert rec['rejudged']['valid'] is False and rec['rejudged']['reasons'] == [GPU_SAMPLE_REASON]
    assert rec['rejudged']['first_record'] == f'{TCE.RECORD_DIR}/{NAME}{TCE.FIRST_SUFFIX}'
    assert re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d', rec['rejudged']['time'])
    assert {k: v for k, v in rec.items() if k not in ('valid', 'reasons', 'samples',
                                                       'thresholds', 'rejudged')} == \
        {k: v for k, v in first.items() if k not in ('valid', 'reasons', 'samples', 'thresholds')}
    S = rec['samples']
    assert {k: S[k] for k in first['samples']} == first['samples']   # measured entries unchanged
    assert S['foreign_gpu_util_mean'] == 0.0795 and S['foreign_gpu_samples_above_limit'] == 1
    assert rec['thresholds'] == TCE.thresholds(5.0)
    assert TCE.check_record(rec) == []
    # the reader of p99 takes the rewritten record and leaves the first copy alone
    r = infer(root, out)
    assert r.returncode == 0, r.stderr
    _, M = macros(out)
    assert M['RccSpecMeasuredN'] == '1' and M['RccSpecHoursN'] == '1.47'
    # a second call keeps the first copy and the first verdict
    r = rejudge(root, p)
    assert r.returncode == 0 and 'verdict before this call: valid=True' in r.stdout
    assert fp.read_bytes() == first_bytes
    again = json.load(open(p))
    assert again['valid'] is True and again['rejudged']['reasons'] == [GPU_SAMPLE_REASON]


def test_rejudge_dry_prints_both_verdicts_and_writes_nothing(tmp_path):
    root, p = stored_record(tmp_path)
    before, data = tree(root), p.read_bytes()
    r = rejudge(root, p, '--dry')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'DRY: nothing is written' in r.stdout and GPU_SAMPLE_REASON in r.stdout
    assert re.search(r'new verdict: +valid=True', r.stdout)
    assert tree(root) == before and p.read_bytes() == data


def test_rejudge_refuses_a_missing_file_and_files_that_do_not_give_the_facts(tmp_path):
    root, p = stored_record(tmp_path)
    data = p.read_bytes()
    first = root / TCE.RECORD_DIR / f'{NAME}{TCE.FIRST_SUFFIX}'
    samples = root / TCE.new_files(NAME)['samples']
    kept = samples.read_bytes()
    samples.unlink()
    r = rejudge(root, p)
    assert r.returncode == 2 and "the stored file samples 'train_log/training_cost/" in r.stderr
    assert 'is missing' in r.stderr and p.read_bytes() == data and not first.exists()
    samples.write_bytes(kept)
    log = root / TCE.new_files(NAME)['log']
    log.write_text(log.read_text().replace('2.7s', '2.8s', 1))      # one update time changed
    r = rejudge(root, p)
    assert r.returncode == 2 and 'do not give the facts of the record' in r.stderr
    assert 'update_seconds_sum' in r.stderr and p.read_bytes() == data and not first.exists()
    log.write_text(log_text())
    r = rejudge(root, root / 'elsewhere.json')
    assert r.returncode == 2 and 'no record' in r.stderr
    copy = root / TCE.RECORD_DIR / 'copy.json'
    copy.write_bytes(data)                              # a record under another name
    r = rejudge(root, copy)
    assert r.returncode == 2 and 'is not the record' in r.stderr
    for change, message in ((dict(stub='/x/stub.py'), 'stub trainer'),
                            (dict(allow_busy=True), 'busy cores')):
        p.write_text(json.dumps(dict(json.loads(data), **change)))
        r = rejudge(root, p)
        assert r.returncode == 2 and message in r.stderr and not first.exists()
    p.write_bytes(data)
    assert rejudge(root, p, '--dry').returncode == 0     # the restored files give the facts


@pytest.mark.parametrize('case', ['busy core', 'stopped'])
def test_rejudge_leaves_a_record_with_another_reason_invalid(tmp_path, case):
    rows, code = run_rows(88, high={40}), 0
    if case == 'busy core':
        rows[10] = sample(cpu_n=1, cpu_pct=40.0, t='r9')
        other = TCE.sample_problems(rows[10])
    else:
        code = -15
        other = [f'{TCE.STOPPED}signal 15 to the launcher', 'trainer exit status -15']
    root, p = stored_record(tmp_path, rows=rows, code=code,
                            reasons=other + [GPU_SAMPLE_REASON])
    r = rejudge(root, p)
    assert r.returncode == 1, r.stdout + r.stderr
    rec = json.load(open(p))
    assert rec['valid'] is False and rec['reasons'] == other
    assert rec['rejudged']['reasons'] == other + [GPU_SAMPLE_REASON]
    assert TCE.check_record(rec)
    assert (root / TCE.RECORD_DIR / f'{NAME}{TCE.FIRST_SUFFIX}').exists()


def test_sample_only_prints_a_verdict():
    r = launcher(SAMPLE_ONLY=1, WINDOW=1)
    assert r.returncode in (0, 1)
    assert 'sample before the start (1 s' in r.stdout and '[training-cost] verdict:' in r.stdout


def test_host_facts_read_the_versions():
    r = run([PY, '-c', 'import sys; sys.path.insert(0, "scripts"); import training_cost_evidence as T, '
             'json; print(json.dumps(T.host_facts([%s])))' % CORE])
    h = json.loads(r.stdout.strip().splitlines()[-1])
    assert re.match(r'^\d+\.\d+', h['torch']) and re.match(r'^\d+\.\d+$', h['torch_cuda'])
    assert 'core_max_mhz' in h


# ================================================================== p99 checks and count
def test_checks_on_two_instances(tmp_path):
    out = tmp_path / 'out'
    args = ['taskset', '-c', CORE, PY, P99, 'checks', '--size', '10x25', '--combos', 'LSW',
            '--n', '2', '--repeats', '3', '--schedules', 'rule', '--allow_busy']
    r = run(args, PACT_ENV='exact', PACT_OUT=out)
    assert r.returncode in (0, 2), r.stderr
    rec = json.load(open(out / 'results' / 'rule_change_checks.json'))
    S = rec['sizes']['10x25']
    assert rec['allow_busy'] is True and rec['schedules'] == 'rule' and rec['repeats'] == 3
    assert rec['timed'] == {'descriptor': ['check_schema_and_values', 'check_family_membership',
                                           'check_magnitude_range'],
                            'schedule': 'check_schedule_validation'}
    assert rec['schedule_folders']['10x25'] == 'or_solution/FAMILY/warmstart_cache_xmask_10x25'
    assert S['instances'] == 2 and [x['instance'] for x in S['per_instance']] == ['instance_000',
                                                                                    'instance_001']
    assert all(x['C1'] == x['C3'] == x['C6'] == 'PASS' for x in S['per_instance'])
    assert 0 < S['descriptor_ms_median'] <= S['descriptor_ms_max']
    assert 0 < S['schedule_ms_median'] <= S['schedule_ms_max']
    assert rec['threads']['OMP_NUM_THREADS'] == '1' and rec['affinity'] == [int(CORE)]
    assert rec['valid'] == S['valid'] and (r.returncode == 0) == rec['valid']
    r = run(args, PACT_ENV='exact', PACT_OUT=out)
    assert r.returncode == 2 and 'pass --force' in r.stderr
    r = run(args[:-1], PACT_OUT=tmp_path / 'out_legacy')          # legacy: no rule caches
    assert r.returncode == 2 and 'PACT_ENV=exact' in r.stderr


def test_checks_policy_schedules_name_their_source(tmp_path):
    out = tmp_path / 'out'
    base = ['taskset', '-c', CORE, PY, P99, 'checks', '--combos', 'LSW', '--n', '1',
            '--repeats', '1', '--seed', '301', '--allow_busy']
    pol = [os.path.isdir(os.path.join(ROOT, REG.schedule_cache(301, s, 'exact')))
           for s in ('10x25', '50x25n30', '80x25n30')]
    if any(pol):
        pytest.skip('policy schedules of the exact-mask campaign exist')
    r = run(base, PACT_ENV='exact', PACT_OUT=out)                    # default sizes
    assert r.returncode == 2 and '--schedules rule' in r.stderr
    r = run(base + ['--size', '10x25,50x25n30', '--rule_sizes', '10x25'], PACT_ENV='exact',
            PACT_OUT=out)
    assert r.returncode == 2 and 'no exported policy schedules at 50x25n30' in r.stderr
    assert '--rule_sizes 50x25n30,10x25' in r.stderr
    r = run(base + ['--schedules', 'rule', '--rule_sizes', '10x25'], PACT_ENV='exact', PACT_OUT=out)
    assert r.returncode == 2 and 'applies to --schedules policy' in r.stderr
    r = run(base + ['--size', '10x25', '--rule_sizes', '80x25n30'], PACT_ENV='exact', PACT_OUT=out)
    assert r.returncode == 2 and 'must be among --size' in r.stderr
    assert not (out / 'results' / 'rule_change_checks.json').exists()
    r = run(base + ['--size', '10x25', '--rule_sizes', '10x25'], PACT_ENV='exact', PACT_OUT=out)
    assert r.returncode in (0, 2), r.stderr
    rec = json.load(open(out / 'results' / 'rule_change_checks.json'))
    assert rec['schedules'] == 'policy' and rec['schedule_sources'] == {'10x25': 'rule'}
    assert rec['sizes']['10x25']['schedules'] == 'rule' and rec['policy_seed'] is None
    assert rec['schedule_folders']['10x25'] == 'or_solution/FAMILY/warmstart_cache_xmask_10x25'


def test_count_in_the_earlier_environment(tmp_path):
    out = tmp_path / 'out'
    r = run([PY, P99, 'count', '--seed', '301'], PACT_OUT=out)
    assert r.returncode == 0, r.stderr
    K = json.load(open(out / 'results' / 'rule_change_count.json'))
    P, S = K['policy'], K['specialists']
    assert K['environment'] == 'legacy'
    assert (P['models'], S['models']) == (1, 8)
    assert P['params_per_model'] == S['params_per_model'] == [32786]
    assert (P['config_files'], S['config_files']) == (1, 8)
    assert P['validation_pairs'] == S['validation_pairs'] == 8
    assert P['validation_rollouts'] == S['validation_rollouts'] == 160
    assert K['validation_equal'] is True and 'the two sides are equal' in r.stdout
    assert K['checks_by_kind'] == {'computed': ['C1', 'C3', 'C4', 'C6'],
                                   'declared': ['C2', 'C5'], 'organizational': ['C7']}
    assert [c['id'] for c in K['checks']] == [f'C{i}' for i in range(1, 8)]
    assert S['checkpoint_bytes'] == sum(
        os.path.getsize(os.path.join(ROOT, REG.checkpoint('specialist', R, 301, 'legacy'))) for R in R8)


def test_count_in_the_exact_environment_names_the_missing_file(tmp_path):
    r = run([PY, P99, 'count'], PACT_ENV='exact', PACT_OUT=tmp_path / 'out')
    if os.path.isfile(os.path.join(ROOT, 'train_log/FAMILY/valiquality_10x25+family+xmask-v1-s301.txt')):
        pytest.skip('the exact-mask campaign has trained the released policy')
    assert r.returncode == 2
    assert 'missing' in r.stderr and re.search(r'train_log/FAMILY/\S*xmask\S*', r.stderr)
    assert not (tmp_path / 'out' / 'results' / 'rule_change_count.json').exists()


def test_check_kinds_equal_run_all():
    code = ('import sys; sys.argv = sys.argv[:1]; sys.path.insert(0, "scripts"); '
            'import p99_rule_change_cost as P; print(P.verify_check_kinds())')
    r = run([PY, '-c', code])
    assert r.returncode == 0, r.stderr
    import p99_rule_change_cost as P
    assert r.stdout.strip() == str([(i, n) for i, n, _ in P.CHECK_KINDS])


def test_ms_decimals():
    import p99_rule_change_cost as P
    assert [P.ms_decimals(x) for x in (0.03, 0.4, 3.0, 0.0004, 0)] == [2, 1, 0, 3, 3]


# ================================================================== p99 infer
HOURS = {'N': 0.74, 'L': 0.75, 'S': 0.77, 'W': 0.76, 'LS': 0.78, 'LW': 0.77, 'SW': 0.79,
         'LSW': 0.80}


def make_record(root, arm, combo=None, hours=1.5, env='exact', **over):
    campaign, name = TCE.run_names(arm, combo or '', 301, 1)
    upd = 2000 if arm == 'released' else 1000
    arch = f'{TCE.RECORD_DIR}/{name}'
    cfgp = root / arch / f'config_10x25+family+{name}.json'
    cfgp.parent.mkdir(parents=True, exist_ok=True)
    base = json.load(open(LEGACY_POL_CFG if arm == 'released' else LEGACY_SPEC_CFG))
    base.update(model_suffix=name, max_updates=upd, first_mask='exact', mask_ties='exact')
    json.dump(base, open(cfgp, 'w'))
    rec = {'kind': 'training_cost', 'name': name, 'environment': env,
           'repeats': {'arm': arm, 'combination': combo, 'seed': 301, 'run': campaign,
                       'config': f'train_log/FAMILY/config_10x25+family+{campaign}.json'},
           'host': {'gpu': 'NVIDIA RTX PRO 5000 Blackwell'}, 'wall_seconds': hours * 3600.0,
           'seconds_per_update': round(hours * 3600.0 / upd, 4), 'updates': upd,
           'done_in_hours': round(hours - 0.01, 2), 'stub': None, 'valid': True, 'reasons': [],
           'files': {'log': f'{TCE.RECORD_DIR}/{name}.log',
                     'config': f'{arch}/config_10x25+family+{name}.json'}}
    rec.update(over)
    p = root / TCE.RECORD_DIR / f'{name}.json'
    json.dump(rec, open(p, 'w'))
    return p


def checks_record(env='exact', sizes=('10x25', '50x25n30', '80x25n30'), valid=True):
    S = {}
    for s in sizes:
        m = int(s.split('x')[0])
        S[s] = {'instances': 240, 'modules': m, 'descriptor_ms_median': 0.2 + m / 1000,
                'descriptor_ms_max': 0.5 + m / 1000, 'descriptor_spread_ms': 0.004,
                'schedule_ms_median': m / 10.0, 'schedule_ms_max': m / 8.0,
                'schedule_spread_ms': 0.03, 'valid': valid, 'reasons': [] if valid else ['busy']}
    return {'kind': 'rule_change_checks', 'environment': env, 'valid': valid,
            'schedules': 'rule', 'sizes': S}


def count_record(env='exact'):
    side = lambda n: {'models': n, 'params_per_model': [32786], 'checkpoint_bytes': 156257 * n,
                      'config_files': n, 'validation_pairs': 8, 'validation_rollouts': 160,
                      'validation_instances': [20]}
    return {'kind': 'rule_change_count', 'environment': env, 'policy': side(1),
            'specialists': side(8), 'validation_equal': True,
            'checks_by_kind': {'computed': ['C1', 'C3', 'C4', 'C6'], 'declared': ['C2', 'C5'],
                               'organizational': ['C7']}}


def infer_tree(tmp_path, combos=R8, checks=True, count=True):
    root = tmp_path / 'root'
    (root / TCE.RECORD_DIR).mkdir(parents=True)
    make_record(root, 'released', hours=1.5)
    for R in combos:
        make_record(root, 'specialist', R, hours=HOURS[R])
    out = tmp_path / 'out'
    (out / 'results').mkdir(parents=True)
    if checks:
        json.dump(checks_record(), open(out / 'results' / 'rule_change_checks.json', 'w'))
    if count:
        json.dump(count_record(), open(out / 'results' / 'rule_change_count.json', 'w'))
    return root, out


def infer(root, out, *extra):
    return run([PY, P99, 'infer', '--root', str(root), *extra], PACT_ENV='exact', PACT_OUT=out)


def macros(out):
    txt = open(out / 'manuscript' / 'macros_rulechange_cost.tex').read()
    return txt, dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', txt))


def test_infer_with_every_input(tmp_path):
    root, out = infer_tree(tmp_path)
    r = infer(root, out, '--require', 'all')
    assert r.returncode == 0, r.stderr
    txt, M = macros(out)
    assert txt.startswith('% Generated by scripts/p99_rule_change_cost.py -- do not hand-edit')
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert all(re.fullmatch(r'Rcc[A-Za-z]+', n) for n in names) and len(names) == len(set(names))
    assert M['RccPolicyHours'] == '1.50' and M['RccPolicyUpdates'] == '2000'
    assert M['RccPolicyEpisodes'] == '40000' and M['RccPolicySecPerUpdate'] == '2.70'
    assert M['RccCatalogueHours'] == f'{sum(HOURS.values()):.2f}'
    assert M['RccSpecMedianHours'] == '0.77' and M['RccSpecMeasuredN'] == '8'
    assert M['RccCatalogueUpdates'] == '8000' and M['RccCatalogueOverPolicyUpdates'] == '4.0'
    assert M['RccCatalogueOverPolicyHours'] == f'{sum(HOURS.values()) / 1.5:.1f}'
    assert M['RccSpecHoursLSW'] == '0.80' and M['RccSpecUpdates'] == '1000'
    hs = sorted(HOURS.values())
    assert M['RccSpecMinHours'] == f'{hs[0]:.2f}' and M['RccSpecMaxHours'] == f'{hs[-1]:.2f}'
    above = next(k for k in range(1, 9) if sum(hs[:k]) > 1.5)
    below = max(k for k in range(1, 9) if sum(hs[8 - k:]) < 1.5)
    assert M['RccSpecSetAbovePolicyN'] == str(above) and M['RccSpecSetBelowPolicyN'] == str(below)
    assert below < above
    assert M['RccDescChecksMsEighty'] == '0.280' and M['RccScheduleCheckMsTen'] == '1.00'
    assert M['RccScheduleCheckMaxMsEighty'] == '10.00' and M['RccChecksInstancesFifty'] == '240'
    assert (M['RccPolicyModels'], M['RccSpecModels'], M['RccParams']) == ('1', '8', '32786')
    assert M['RccPolicyValiRollouts'] == M['RccSpecValiRollouts'] == '160'
    assert (M['RccChecksComputed'], M['RccChecksDeclared'], M['RccChecksOrganizational']) == ('4', '2', '1')
    res = json.load(open(out / 'results' / 'rule_change_cost.json'))
    assert res['missing'] == [] and res['count_validation_equal'] is True
    assert (out / 'results' / 'rule_change_cost.md').read_text().count('`\\Rcc') == len(names)


def test_fewer_than_eight_specialists_never_print_the_catalogue(tmp_path):
    root, out = infer_tree(tmp_path, combos=('N', 'L', 'S', 'W', 'LS'))
    r = infer(root, out)
    assert r.returncode == 0, r.stderr
    txt, M = macros(out)
    assert M['RccSpecMeasuredN'] == '5' and 'RccSpecHoursLS' in M and 'RccSpecHoursSW' not in M
    assert not [n for n in M if 'Catalogue' in n or 'Median' in n]
    assert 'Blocks skipped: catalogue sum, median and ratios: 5 of 8 specialists measured' in txt
    r = infer(root, out, '--require', 'all')
    assert r.returncode == 2 and 'specialists LW,SW,LSW' in r.stderr
    assert not glob.glob(str(out / 'manuscript' / '*')) and not (out / 'results' / 'rule_change_cost.json').exists()


def test_infer_writes_only_the_blocks_with_inputs(tmp_path):
    root, out = infer_tree(tmp_path, combos=(), checks=False, count=True)
    r = infer(root, out)
    assert r.returncode == 0, r.stderr
    txt, M = macros(out)
    assert 'RccPolicyHours' in M and 'RccPolicyModels' in M and 'RccDescChecksMsTen' not in M
    assert 'checks: no results/rule_change_checks.json' in txt


@pytest.mark.parametrize('change,message', [
    (dict(valid=False, reasons=['run sample: other GPU compute 1:3%:python']), 'record is invalid'),
    (dict(stub='tests/stub.py'), 'stub trainer'),
    (dict(files={'log': 'train_log/FAMILY_xmask-solo-W-s301.log'}), 'not a measurement log'),
    (dict(environment='legacy'), 'mixed environments'),
    (dict(host={'gpu': 'another GPU'}), 'records from different GPUs')])
def test_infer_refuses_a_record(tmp_path, change, message):
    root, out = infer_tree(tmp_path)
    make_record(root, 'specialist', 'W', hours=0.76, **change)
    (out / 'manuscript').mkdir()
    (out / 'manuscript' / 'macros_rulechange_cost.tex').write_text('% stale\n')
    r = infer(root, out)
    assert r.returncode == 2 and message in r.stderr
    assert not (out / 'manuscript' / 'macros_rulechange_cost.tex').exists()


def test_infer_refuses_duplicates_and_other_environments(tmp_path):
    root, out = infer_tree(tmp_path)
    p = make_record(root, 'specialist', 'W', hours=0.76)
    dup = root / TCE.RECORD_DIR / 'xcost-solo-W-s301-r2.json'
    rec = json.load(open(p))
    rec['name'] = 'xcost-solo-W-s301-r2'
    json.dump(rec, open(dup, 'w'))
    r = infer(root, out)
    assert r.returncode == 2 and 'both measure' in r.stderr
    dup.unlink()
    json.dump(count_record(env='legacy'), open(out / 'results' / 'rule_change_count.json', 'w'))
    r = infer(root, out)
    assert r.returncode == 2 and 'mixed environments' in r.stderr
    json.dump(count_record(), open(out / 'results' / 'rule_change_count.json', 'w'))
    json.dump(checks_record(valid=False), open(out / 'results' / 'rule_change_checks.json', 'w'))
    r = infer(root, out)
    assert r.returncode == 2 and 'rule_change_checks.json is invalid' in r.stderr
    r = run([PY, P99, 'infer', '--root', str(root)], PACT_OUT=out)      # legacy reading
    assert r.returncode == 2 and 'mixed environments' in r.stderr


def test_infer_records_named_on_the_command_line(tmp_path):
    root, out = infer_tree(tmp_path)
    one = root / TCE.RECORD_DIR / 'xcost-v1-s301-r1.json'
    r = infer(root, out, '--records', str(one))
    assert r.returncode == 0, r.stderr
    _, M = macros(out)
    assert 'RccPolicyHours' in M and 'RccSpecMeasuredN' not in M
