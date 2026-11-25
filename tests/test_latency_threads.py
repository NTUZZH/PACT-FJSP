"""Tests for the thread, checkpoint and idle-core handling of
scripts/p75_deploy_cost.py and scripts/p16_policy_latency.py.

  * the numpy thread variables follow --numpy_threads (default 1) before numpy
    is imported, and torch reports one intra-op and one inter-op thread after
    set_torch_threads(torch, 1) (subprocess);
  * the checkpoint comes from arm_registry: jointblind-v1 in the legacy
    environment, and in the exact environment a clean refusal naming
    trained_network/FAMILY/10x25+family+xmask-v1-s301.pth while it is missing
    (function and p16 command line);
  * the mask options of the rollout must match the checkpoint's training
    configuration, and a legacy run refuses a mask variable that asks for the
    exact correction;
  * the idle-core check flags another process's thread above 5% of a core in
    the mask (synthetic /proc and ps tables), ignores other cores and this
    process, and stops a measurement unless the override is given.

Run from the repository root:  python3 -m pytest tests/test_latency_threads.py -q
CPU only; a few seconds.
"""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None, PACT_FIRST_MASK=None,
                                        PACT_MASK_TOL=None)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402  (parses sys.argv once, at import)
    spec = importlib.util.spec_from_file_location(
        'p75_under_test', os.path.join(ROOT, 'scripts', 'p75_deploy_cost.py'))
    P75 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(P75)
finally:
    sys.argv = _argv
configs.device = 'cpu'
CLEAN = {k: v for k, v in os.environ.items()
         if not k.startswith('PACT_') and k not in P75.NUMPY_VARS}


def _run(code_or_args, env_extra=None, script=None):
    env = dict(CLEAN, CUDA_VISIBLE_DEVICES='', **(env_extra or {}))
    cmd = ([sys.executable, os.path.join(ROOT, script)] + code_or_args if script
           else [sys.executable, '-c', code_or_args])
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                          timeout=100)


def test_early_numpy_threads(monkeypatch):
    for v in P75.NUMPY_VARS:
        monkeypatch.delenv(v, raising=False)
    P75.early_numpy_threads(['x'])
    assert all(os.environ[v] == '1' for v in P75.NUMPY_VARS)
    P75.early_numpy_threads(['x', '--numpy_threads', '4'])
    assert all(os.environ[v] == '4' for v in P75.NUMPY_VARS)
    P75.early_numpy_threads(['x', '--numpy_threads=2'])
    assert os.environ['OMP_NUM_THREADS'] == '2'


def test_torch_one_thread_subprocess():
    code = ('import sys; sys.argv=[sys.argv[0]]; '
            'sys.path.insert(0, "scripts"); import p75_deploy_cost as P; '
            'import torch; P.set_torch_threads(torch, 1); '
            'print(torch.get_num_threads(), torch.get_num_interop_threads())')
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ['1', '1']


def test_checkpoint_legacy():
    name, ckpt, cfg = P75.resolve_checkpoint()
    assert name == '10x25+family+jointblind-v1'
    assert ckpt == 'trained_network/FAMILY/10x25+family+jointblind-v1.pth'
    assert cfg == 'train_log/FAMILY/config_10x25+family+jointblind-v1.json'


def test_checkpoint_exact_missing(monkeypatch, tmp_path):
    # an empty root stands in for the repository, so the checkpoint is missing whatever
    # the repository holds
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setattr(P75, 'ROOT', str(tmp_path))
    with pytest.raises(SystemExit, match=r'trained_network/FAMILY/10x25\+family\+'
                                         r'xmask-v1-s301\.pth'):
        P75.resolve_checkpoint()


def test_p16_refuses_missing_exact_checkpoint(tmp_path):
    # p16 run as __main__ after its helper module p75 was imported with an empty root in
    # place of the repository, so the checkpoint is missing whatever the repository holds
    empty, out = tmp_path / 'root', tmp_path / 'out'
    empty.mkdir()
    code = ('import os, runpy, sys\n'
            'script = os.path.join("scripts", "p16_policy_latency.py")\n'
            'sys.path.insert(0, "scripts")\n'
            'sys.argv = [script]\n'
            'import p75_deploy_cost as P75\n'
            f'P75.ROOT = {str(empty)!r}\n'
            'sys.argv = [script, "--n", "1", "--regime", "S"]\n'
            'runpy.run_path(script, run_name="__main__")\n')
    r = _run(code, env_extra={'PACT_ENV': 'exact', 'PACT_OUT': str(out)})
    assert r.returncode != 0
    assert 'trained_network/FAMILY/10x25+family+xmask-v1-s301.pth' in r.stderr
    assert not any(p.is_file() for p in tmp_path.rglob('*'))


def test_mask_options(monkeypatch):
    assert P75.mask_options({}, 'legacy-run') == {}
    monkeypatch.setenv('PACT_FIRST_MASK', '1')
    with pytest.raises(SystemExit, match='legacy environment is active'):
        P75.mask_options({}, 'legacy-run')
    monkeypatch.delenv('PACT_FIRST_MASK')
    monkeypatch.setenv('PACT_ENV', 'exact')
    exact = {'first_mask': 'exact', 'mask_ties': 'exact'}
    assert P75.mask_options(dict(exact), 'exact-run') == exact
    with pytest.raises(SystemExit, match='was trained with'):
        P75.mask_options({}, 'legacy-run')


def _fake_proc(monkeypatch, rows_a, rows_b, ps_rows=()):
    snaps = iter([rows_a, rows_b])
    monkeypatch.setattr(P75, '_thread_times', lambda: next(snaps))
    monkeypatch.setattr(P75, '_ps_rows', lambda: (list(ps_rows), 999999))
    monkeypatch.setattr(P75.time, 'sleep', lambda s: None)


def test_busy_processes(monkeypatch):
    hz = os.sysconf('SC_CLK_TCK')
    me = os.getpid()
    a = {'/proc/111/task/111': (0, 5), '/proc/222/task/222': (0, 9),
         f'/proc/{me}/task/{me}': (0, 5), '/proc/333/task/334': (0, 5)}
    b = {'/proc/111/task/111': (int(0.5 * hz * 2), 5),     # 50% on core 5
         '/proc/222/task/222': (int(0.9 * hz * 2), 9),     # busy, other core
         f'/proc/{me}/task/{me}': (int(1.0 * hz * 2), 5),  # this process
         '/proc/333/task/334': (int(0.02 * hz * 2), 5)}    # 2%: idle enough
    _fake_proc(monkeypatch, a, b, ps_rows=['  444   5  12.0 sleeper --lifetime',
                                           '  555   7  80.0 elsewhere'])
    busy = P75.busy_processes([4, 5], window=2.0)
    pids = sorted(int(x.split()[1]) for x in busy)
    assert pids == [111, 444], busy


def test_idle_check_stops_unless_override(monkeypatch):
    monkeypatch.setattr(P75, 'busy_processes', lambda cores, window: ['pid 1 core 5 50%'])
    monkeypatch.setattr(P75, 'core_busy', lambda cores, window=1.0: {c: 0.5 for c in cores})
    with pytest.raises(SystemExit, match='REFUSING to measure'):
        P75.idle_check([5], False, 'test')
    ev = P75.idle_check([5], True, 'test')
    assert ev['idle'] is False and ev['busy_processes']
    assert P75.idle_check([5], False, 'after', record_only=True)['idle'] is False
    monkeypatch.setattr(P75, 'busy_processes', lambda cores, window: [])
    assert P75.idle_check([5], False, 'test')['idle'] is True
