"""
Lanes of the three GPU training queues, checked through LIST_ONLY, which prints
the tasks an invocation would train and exits before any probe:

    scripts/run_exactmask_queue.sh        principal arms, seeds 301-305
    scripts/run_exactmask_extra_queue.sh  further arms, seeds 301-303
    scripts/run_hgnn_exact_queue.sh       PHASE=tune, panel, panel2 and solo

    python -m pytest tests/test_gpu_queue_lanes.py -q

No torch, no training and no file written in the repository. The scripts run
with HOME pointed at an empty folder, so the trainer's Python cannot start.

  * the default (LANE=0/1, NO_GATE=0) lists the tasks of the queue in the order
    the queue builds them; the expected lists below are written out
    independently of the scripts;
  * for n = 2, 3, 4 the lanes are pairwise disjoint, their union is the default
    list, and lane i holds exactly the tasks at index i modulo n of the full
    list, in order; with ONLY the same holds for the matching tasks, because
    the index is taken before ONLY;
  * LIST_ONLY=1 skips the finished runs (and, in the HGNN queue, the runs with
    files of an unfinished run) and writes nothing. It runs on a copy of each
    script in an empty repository where such runs are planted, so the result
    does not depend on which runs this machine has finished;
  * in the HGNN queue the order of the phases holds under LIST_ONLY=1 with and
    without NO_GATE=1: panel and panel2 need the selection of the tuning runs and
    its LR (panel2 not the finished panel), and solo needs the finished panel;
  * a LANE, NO_GATE or LIST_ONLY value that does not parse, and a CORES the
    launcher cannot pin itself to, stop the script before it lists anything.
"""
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRINCIPAL = 'run_exactmask_queue.sh'
EXTRA = 'run_exactmask_extra_queue.sh'
HGNN = 'run_hgnn_exact_queue.sh'
SCRIPTS = (PRINCIPAL, EXTRA, HGNN)
# the launcher pins itself to CORES: give it the cores this test already runs on
CORES = ','.join(str(c) for c in sorted(os.sched_getaffinity(0)))
REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
RATES = ('1e-4', '2e-4', '4e-4')


# ------------------------------------------------------------------ expected task lists

def principal_tasks():
    out = []
    for tier in ((301, 302, 303), (304, 305)):
        out += ['xmask-v1-s%d' % s for s in tier]
        out += ['xmask-ctl-s%d' % s for s in tier]
        out += ['xmask-solo-%s-s%d' % (r, s) for s in tier for r in REGIMES]
    return out


def extra_tasks():
    arms = ('eq', 'sparse', 'dzero', 'dmax', 'wthird', 'film', 'dr', 'all8', 'pair', 'active')
    return ['xmask-%s-s%d' % (a, s) for a in arms for s in (301, 302, 303)]


def lr_tag(lr):
    return '' if lr == '2e-4' else '-lr' + lr


def hgnn_tasks(phase, lr=None):
    if phase == 'tune':
        return ['hgnn-certified-joint-s301-xmask' + lr_tag(r) for r in RATES]
    if phase == 'panel':
        return (['hgnn-certified-joint-s%d-xmask%s' % (s, lr_tag(lr)) for s in (302, 303)]
                + ['hgnn-published-joint-s%d-xmask%s' % (s, lr_tag(lr)) for s in (301, 302, 303)])
    if phase == 'panel2':
        return ['hgnn-certified-joint-s%d-xmask%s' % (s, lr_tag(lr)) for s in (304, 305)]
    return ['hgnn-certified-%s-s301-xmask%s' % (r, lr_tag(lr)) for r in REGIMES]


# (id, script, environment, full task list, ONLY regex matching some but not all tasks)
CASES = [
    ('principal', PRINCIPAL, {}, principal_tasks(), r'solo-(S|LSW)-|v1-s30[45]'),
    ('extra', EXTRA, {}, extra_tasks(), r'-(sparse|film)-|s303'),
    ('hgnn-tune', HGNN, {'PHASE': 'tune'}, hgnn_tasks('tune'), r'lr'),
    ('hgnn-panel-1e-4', HGNN, {'PHASE': 'panel', 'LR': '1e-4'}, hgnn_tasks('panel', '1e-4'), r'published'),
    ('hgnn-panel-2e-4', HGNN, {'PHASE': 'panel', 'LR': '2e-4'}, hgnn_tasks('panel', '2e-4'), r'-s30[23]-'),
    ('hgnn-solo-4e-4', HGNN, {'PHASE': 'solo', 'LR': '4e-4'}, hgnn_tasks('solo', '4e-4'), r'-(N|LSW)-'),
    ('hgnn-panel2-2e-4', HGNN, {'PHASE': 'panel2', 'LR': '2e-4'}, hgnn_tasks('panel2', '2e-4'), r'-s305-'),
    ('hgnn-panel2-1e-4', HGNN, {'PHASE': 'panel2', 'LR': '1e-4'}, hgnn_tasks('panel2', '1e-4'), r'-s304-'),
]
IDS = [c[0] for c in CASES]
PARAMS = [c[1:] for c in CASES]
# the cases whose LIST_ONLY=1 list needs no earlier phase
FIRST = [c[1:4] for c in CASES[:3]]
FIRST_IDS = IDS[:3]


# ------------------------------------------------------------------ helpers

def run(script, env, home, repo=REPO):
    full = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'HOME': str(home),
            'CORES': CORES, 'THREADS': '1'}
    full.update(env)
    return subprocess.run(['bash', os.path.join(str(repo), 'scripts', script)], env=full,
                          capture_output=True, text=True, timeout=120)


def tasks_of(out):
    return [line.split(' ', 1)[1] for line in out.splitlines() if line.startswith('TASK ')]


def listed(script, env, home, repo=REPO):
    r = run(script, env, home, repo)
    assert r.returncode == 0, r.stdout + r.stderr
    return tasks_of(r.stdout)


def lane_of(full, i, n, keep=None):
    return [t for k, t in enumerate(full) if k % n == i and (keep is None or re.search(keep, t))]


@pytest.fixture
def home(tmp_path):
    h = tmp_path / 'home'   # no miniconda3 under it: the trainer's Python cannot start
    h.mkdir()
    return h


def make_repo(tmp_path, selection=False):
    repo = tmp_path / 'repo'
    (repo / 'scripts').mkdir(parents=True)
    for s in SCRIPTS + (('hgnn_select_lr.py',) if selection else ()):
        shutil.copy2(os.path.join(REPO, 'scripts', s), str(repo / 'scripts' / s))
    return repo


def write(repo, rel, text=''):
    p = os.path.join(str(repo), rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(text)


def plant_run(repo, name, pth=True, done=True):
    if pth:
        write(repo, 'trained_network/FAMILY/10x25+family+%s.pth' % name, 'x')
    write(repo, 'train_log/FAMILY_%s.log' % name,
          '[train_family] done in 2.00 h; best rel=0.8000\n' if done else 'Ep 1\n')


def snapshot(root):
    out = {}
    for d, dirs, files in os.walk(str(root)):
        for f in dirs + files:
            p = os.path.join(d, f)
            st = os.lstat(p)
            out[os.path.relpath(p, str(root))] = (st.st_size, st.st_mtime_ns)
    return out


# ------------------------------------------------------------------ default and partition

@pytest.mark.parametrize('script', SCRIPTS)
def test_bash_syntax(script):
    r = subprocess.run(['bash', '-n', os.path.join(REPO, 'scripts', script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


@pytest.mark.parametrize('script,env,full,only', PARAMS, ids=IDS)
def test_default_lists_the_queue_in_order(script, env, full, only, home):
    assert len(full) == len(set(full))
    for extra in ({}, {'LANE': '0/1'}, {'NO_GATE': '0'}, {'NO_GATE': '1'}):
        assert listed(script, dict(env, LIST_ONLY='all', **extra), home) == full, extra
    keep = [t for t in full if re.search(only, t)]
    assert 0 < len(keep) < len(full)
    assert listed(script, dict(env, LIST_ONLY='all', ONLY=only), home) == keep


@pytest.mark.parametrize('with_only', [False, True], ids=['full', 'ONLY'])
@pytest.mark.parametrize('script,env,full,only', PARAMS, ids=IDS)
def test_lanes_partition_the_list(script, env, full, only, with_only, home):
    keep = only if with_only else None
    want = [t for t in full if keep is None or re.search(keep, t)]
    extra = {'ONLY': only} if with_only else {}
    for n in (2, 3, 4):
        lanes = [listed(script, dict(env, LIST_ONLY='all', LANE='%d/%d' % (i, n), **extra), home)
                 for i in range(n)]
        for i, got in enumerate(lanes):
            assert got == lane_of(full, i, n, keep), (n, i)
        for a in range(n):
            for b in range(a + 1, n):
                assert not set(lanes[a]) & set(lanes[b]), (n, a, b)
        assert sorted(t for lane in lanes for t in lane) == sorted(want), n


# ------------------------------------------------------------------ LIST_ONLY=1 on a planted repository

PLANTED = {
    # name: (checkpoint, completion line); finished = both
    PRINCIPAL: {'xmask-v1-s302': (True, True), 'xmask-solo-LSW-s305': (True, True),
                'xmask-ctl-s301': (False, True), 'xmask-solo-N-s301': (True, False)},
    EXTRA: {'xmask-film-s302': (True, True), 'xmask-eq-s301': (True, False)},
    HGNN: {'hgnn-certified-joint-s301-xmask': (True, True)},
}


@pytest.mark.parametrize('script,env,full', FIRST, ids=FIRST_IDS)
def test_list_only_skips_finished_runs_and_writes_nothing(script, env, full, tmp_path, home):
    repo = make_repo(tmp_path)
    assert listed(script, dict(env, LIST_ONLY='1'), home, repo) == full   # nothing finished yet
    skipped = set()
    for name, (pth, done) in PLANTED[script].items():
        plant_run(repo, name, pth, done)
        if pth and done:
            skipped.add(name)
    if script == HGNN:   # files of an unfinished run: refused, so not listed by LIST_ONLY=1
        partial = 'hgnn-certified-joint-s301-xmask-lr4e-4'
        write(repo, 'train_log/FAMILY/config_10x25+family+%s.json' % partial, '{}')
        skipped.add(partial)
    before = snapshot(repo)
    assert listed(script, dict(env, LIST_ONLY='1'), home, repo) == [t for t in full if t not in skipped]
    assert listed(script, dict(env, LIST_ONLY='all'), home, repo) == full
    for i in range(2):
        got = listed(script, dict(env, LIST_ONLY='1', LANE='%d/2' % i), home, repo)
        assert got == [t for t in lane_of(full, i, 2) if t not in skipped]
    assert snapshot(repo) == before


# ------------------------------------------------------------------ HGNN phase order

NORMS = {'N@10': 156.6, 'L@10': 226.95, 'S@10': 235.3, 'W@10': 177.65, 'LS@10': 304.4, 'LW@10': 243.4}


def plant_tuning(repo, scores):
    for lr, score in scores.items():
        name = 'hgnn-certified-joint-s301-xmask' + lr_tag(lr)
        cfg = {'model_name': '10x25+family+' + name, 'hgnn_mask': 'exact', 'hgnn_reward': 'certified',
               'n_params': 204250, 'train_paras': {'lr': float(lr)}, 'vali_norm': NORMS,
               'args': {'seed': 301, 'max_iterations': 2000, 'lr': float(lr),
                        'family_regimes': 'N,L,S,W,LS,LW'}}
        curve = [1.0] * 200
        curve[100] = score
        write(repo, 'train_log/FAMILY/config_10x25+family+%s.json' % name, json.dumps(cfg))
        write(repo, 'train_log/FAMILY/valiquality_10x25+family+%s.txt' % name, str(curve))
        write(repo, 'train_log/FAMILY_%s.log' % name,
              '[train_family_hgnn] done in 3.10 h; best rel=%.4f; checkpoint x\n' % score)


def test_hgnn_phase_order_holds_under_list_only(tmp_path):
    repo = make_repo(tmp_path, selection=True)
    home = tmp_path / 'home'
    py = home / 'miniconda3' / 'envs' / 'fjsp' / 'bin' / 'python'
    py.parent.mkdir(parents=True)
    py.write_text('#!/bin/sh\nexec %s "$@"\n' % sys.executable)   # the selection is plain Python
    py.chmod(0o755)

    def go(phase, lr, gate):
        return run(HGNN, {'PHASE': phase, 'LR': lr, 'LIST_ONLY': '1', 'NO_GATE': gate}, home, repo)

    for gate in ('0', '1'):
        for phase in ('panel', 'panel2'):
            r = go(phase, '1e-4', gate)
            assert r.returncode == 1 and 'selection is not available' in r.stdout and not tasks_of(r.stdout)
    plant_tuning(repo, {'1e-4': 0.8810, '2e-4': 0.8850, '4e-4': 0.8900})
    for gate in ('0', '1'):
        for phase in ('panel', 'panel2'):    # panel2 does not wait for the panel
            r = go(phase, '2e-4', gate)
            assert r.returncode == 1 and 'the selection chose 1e-4' in r.stdout and not tasks_of(r.stdout)
            r = go(phase, '1e-4', gate)
            assert r.returncode == 0 and tasks_of(r.stdout) == hgnn_tasks(phase, '1e-4'), r.stdout
        r = go('solo', '1e-4', gate)
        assert r.returncode == 1 and 'PHASE=solo runs last' in r.stdout and not tasks_of(r.stdout)
    for name in hgnn_tasks('panel', '1e-4'):
        write(repo, 'train_log/FAMILY_%s.log' % name, '[train_family_hgnn] done in 3.00 h; best rel=0.9\n')
    before = snapshot(repo)
    for gate in ('0', '1'):
        r = go('solo', '1e-4', gate)
        assert r.returncode == 0 and tasks_of(r.stdout) == hgnn_tasks('solo', '1e-4'), r.stdout
    assert snapshot(repo) == before


# ------------------------------------------------------------------ refusals

BAD = [('LANE', v) for v in ('2/2', '1/0', '0/0', 'a/2', '01/4', '1', '-1/2', '1/2/3')] \
    + [('NO_GATE', 'yes'), ('LIST_ONLY', 'yes'), ('CORES', '9999')]


@pytest.mark.parametrize('var,value', BAD, ids=['%s=%s' % b for b in BAD])
@pytest.mark.parametrize('script', SCRIPTS)
def test_bad_options_stop_the_launcher(script, var, value, home):
    env = {'PHASE': 'tune', 'LIST_ONLY': 'all'}
    env[var] = value
    r = run(script, env, home)
    want = 'could not pin the launcher' if var == 'CORES' else 'ABORT: %s must be' % var
    assert r.returncode == 1 and want in r.stdout and not tasks_of(r.stdout), r.stdout + r.stderr
