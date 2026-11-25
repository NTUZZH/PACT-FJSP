"""Memory gate of scripts/p85_cpsat_arms.py: the hold read from another
project's state file and the runner's own state lines."""
import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope='module')
def p85():
    spec = importlib.util.spec_from_file_location(
        'p85_cpsat_arms', os.path.join(ROOT, 'scripts', 'p85_cpsat_arms.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_no_variable_no_hold(p85, monkeypatch):
    monkeypatch.delenv('PACT_PEER_MEMORY_STATE', raising=False)
    assert p85.peer_hold() is False


def test_missing_file_no_hold(p85, monkeypatch, tmp_path):
    monkeypatch.setenv('PACT_PEER_MEMORY_STATE', str(tmp_path / 'absent.txt'))
    assert p85.peer_hold() is False


@pytest.mark.parametrize('lines, hold', [
    (['STEADY pass 1 training'], False),
    (['LOADING pass 2 file'], True),
    (['LOADING pass 2 file', 'STEADY pass 2 training'], False),
    (['STEADY pass 2 training', 'LOADING pass 3 file', ''], True),
    (['EVALUATION 4B running'], False),
    (['LOADING pass 2 file | written 2026-11-12 03:31:00',
      'NOTE load of pass 2 starts during a large solve | written 2026-11-12 03:31:05'], True),
    (['IDLE pass 1 done', 'NOTE remark'], False),
    ([], False),
])
def test_last_line_decides(p85, monkeypatch, tmp_path, lines, hold):
    f = tmp_path / 'state.txt'
    f.write_text('\n'.join(lines) + '\n')
    monkeypatch.setenv('PACT_PEER_MEMORY_STATE', str(f))
    monkeypatch.delenv('PACT_PEER_HOLD_WORDS', raising=False)
    assert p85.peer_hold() is hold


def test_hold_words_from_environment(p85, monkeypatch, tmp_path):
    f = tmp_path / 'state.txt'
    f.write_text('EVALUATION 4B running\n')
    monkeypatch.setenv('PACT_PEER_MEMORY_STATE', str(f))
    monkeypatch.setenv('PACT_PEER_HOLD_WORDS', 'LOADING EVALUATION')
    assert p85.peer_hold() is True


def test_announce_appends_and_is_silent_without_variable(p85, monkeypatch, tmp_path):
    monkeypatch.delenv('PACT_MEMORY_STATE', raising=False)
    p85.announce('BIG up to 40 GB')          # no file, no error
    f = tmp_path / 'own.txt'
    monkeypatch.setenv('PACT_MEMORY_STATE', str(f))
    p85.announce('BIG up to 40 GB, ref 80x25n30 S instance_000, until about 03:10')
    p85.announce('IDLE no large solve')
    lines = f.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('BIG up to 40 GB') and lines[1].startswith('IDLE')
    assert ' | written ' in lines[1]


def test_only_the_80_module_full_models_are_large(p85):
    assert p85.needs_gb('ref', '80x25n30', 'S')[0] >= p85.PEER_MIN_GB
    assert p85.needs_gb('compact', '80x25n30', 'LSW')[0] >= p85.PEER_MIN_GB
    assert p85.needs_gb('ref', '50x25n30', 'S')[0] < p85.PEER_MIN_GB
    assert p85.needs_gb('lns', '80x25n30', 'S')[0] < p85.PEER_MIN_GB
    assert p85.needs_gb('ref', '80x25n30', 'N')[0] < p85.PEER_MIN_GB


def test_worker_waits_for_the_peer_and_announces(p85, monkeypatch, tmp_path):
    import json
    import multiprocessing as mp
    peer = tmp_path / 'peer.txt'
    peer.write_text('LOADING pass 2 file\n')
    own = tmp_path / 'own.txt'
    monkeypatch.setenv('PACT_PEER_MEMORY_STATE', str(peer))
    monkeypatch.setenv('PACT_MEMORY_STATE', str(own))
    monkeypatch.delenv('PACT_PEER_HOLD_WORDS', raising=False)
    monkeypatch.setattr(p85, 'needs_gb', lambda arm, size, regime: (40.0, False))
    monkeypatch.setattr(p85, 'mem_available_gb', lambda: 64.0)

    def fake_solve(t, cores):
        with open(t['out'], 'w') as f:
            json.dump({'status': 'FEASIBLE', 'objective': 1, 'warm_makespan': 1,
                       'cpu_share': 1.0}, f)
    monkeypatch.setattr(p85, 'solve_task', fake_solve)
    waits = []

    def fake_sleep(seconds):
        waits.append(seconds)
        peer.write_text('LOADING pass 2 file\nSTEADY pass 2 training\n')
    monkeypatch.setattr(p85.time, 'sleep', fake_sleep)

    q = mp.Queue()
    q.put({'size': '80x25n30', 'regime': 'S', 'name': 'instance_000', 'arm': 'ref',
           'budget': 600, 'out': str(tmp_path / 'rec.json')})
    p85.worker(None, q, str(tmp_path / '_big.lock'))
    # one wait ended by the peer's new line, then the pause after the solve
    # that lets the peer's poll see the IDLE line
    assert waits == [20, p85.PEER_GAP_S]
    lines = own.read_text().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith('BIG up to 40 GB, ref 80x25n30 S instance_000, until about ')
    assert lines[1].startswith('IDLE')
    assert (tmp_path / 'rec.json').exists()


def test_worker_writes_nothing_for_a_small_solve(p85, monkeypatch, tmp_path):
    import json
    import multiprocessing as mp
    peer = tmp_path / 'peer.txt'
    peer.write_text('LOADING pass 2 file\n')   # a small solve does not wait for it
    own = tmp_path / 'own.txt'
    monkeypatch.setenv('PACT_PEER_MEMORY_STATE', str(peer))
    monkeypatch.setenv('PACT_MEMORY_STATE', str(own))
    monkeypatch.setattr(p85, 'needs_gb', lambda arm, size, regime: (4.0, False))
    monkeypatch.setattr(p85, 'mem_available_gb', lambda: 64.0)

    def fake_solve(t, cores):
        with open(t['out'], 'w') as f:
            json.dump({'status': 'FEASIBLE', 'objective': 1, 'warm_makespan': 1,
                       'cpu_share': 1.0}, f)
    monkeypatch.setattr(p85, 'solve_task', fake_solve)
    q = mp.Queue()
    q.put({'size': '50x25n30', 'regime': 'N', 'name': 'instance_000', 'arm': 'lns',
           'budget': 600, 'out': str(tmp_path / 'rec.json')})
    p85.worker(None, q, str(tmp_path / '_big.lock'))
    assert not own.exists()
    assert (tmp_path / 'rec.json').exists()
