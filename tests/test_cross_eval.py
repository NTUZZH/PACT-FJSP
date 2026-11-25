"""The off-combination evaluation of the specialists and the two secondary arms that
train on all eight combinations (xmask-all8) and read the pair changeover channels
(xmask-pair); the arm with the active-schedule mask (xmask-active), appended after them,
is checked here wherever the registry, the plan or the extra queue lists every arm.

  * plan: the group cross holds one job per specialist run with its seven other
    combinations, writes into the folder of its own, and asks for the files its check
    asks for; no cross array has the path of an array of another group;
  * regression: against the scripts of commit BASE, the registry returns the same
    names, paths and seeds for every arm it had, and every job and expected file of
    every group it had is unchanged (the only additions are the jobs of the three arms
    added since);
  * queue: the three arms added to scripts/run_exactmask_extra_queue.sh resolve to the
    released recipe plus their one option, and the parameter count the queue asserts
    for every arm is the count of the network built on the CPU (no training);
  * analysis: scripts/p100_cross_eval.py on synthetic result trees with a known
    answer, its refusals and its macro names; the staged table uses its macros only.

Run from the repository root:
    python -m pytest tests/test_cross_eval.py -q
CPU only. The plan, regression and queue tests read the stored results, data and
training configurations and are skipped without them; nothing is written in the
repository.
"""
import contextlib
import importlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402

R8 = list(REG.COMBOS)
SIX = ','.join(REG.TRAIN_COMBOS)
ALL8 = ','.join(R8)
BASE = 'd92ff1526142212c7c372fbaa3c3a4351d5185ad'     # the scripts before the two arms and the group
NEW_ARMS = ('xmask-all8', 'xmask-pair')
LATER_ARMS = ('xmask-active',)                         # appended to the registry after NEW_ARMS
ADDED = NEW_ARMS + LATER_ARMS                          # every arm the registry has beyond BASE
# names of the active-rule arm of the public family, added to the registry after BASE
PUB_ACT_CONSTS = ('PUB_RULE_WORD', 'PUB_ACT_RULE', 'PUB_ACT_STEM', 'PUB_ACT_SEEDS')
# names of the public family under the active rule, added to the registry after them
PUB_ACTFAM_CONSTS = ('PUB_RULE_STEM', 'PUB_ACTFAM_SEEDS')
EXTRA = os.path.join(ROOT, 'scripts', 'run_exactmask_extra_queue.sh')
PRINCIPAL_Q = os.path.join(ROOT, 'scripts', 'run_exactmask_queue.sh')
REF_CONFIG = 'train_log/FAMILY/config_10x25+family+jointblind-v1-s303.json'
TABLE = os.path.join(ROOT, 'manuscript', 'sections_new', 'tables', 'ts_cross_eval.tex')
needs_table = pytest.mark.skipif(
    not os.path.isfile(TABLE),
    reason='needs the table of the manuscript manuscript/sections_new/tables/ts_cross_eval.tex, '
           'which the release does not hold')
PY = sys.executable

needs_results = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'test_results/FAMILY')) or
    not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25')),
    reason='needs the stored results and the instance data')
GROUP_ENV = ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'PUBACT', 'LEGACY45', 'CROSS',
             'HGNN', 'HGNN_LR', 'HGNN_SOLO')
ALL_GROUPS = 'principal secondary n30 rules public pubsens pubeq legacy45 hgnn'


def clean_env():
    env = {k: v for k, v in os.environ.items() if k not in GROUP_ENV and not k.startswith('PACT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='')
    return env


@contextlib.contextmanager
def groups(**env):
    """The plan module reloaded under env (the group switches cleared first), with the
    completeness check run once; reloaded with the defaults afterwards."""
    os.chdir(ROOT)
    keep = {k: os.environ.pop(k, None) for k in GROUP_ENV}
    import eval_newseeds_plan as P
    try:
        os.environ.update(env)
        P = importlib.reload(P)
        P.EXPECTED.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            P.check()
        yield P
    finally:
        for k in GROUP_ENV:
            os.environ.pop(k, None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(P)


# ================================================================== plan: group cross
@needs_results
def test_cross_group_holds_one_job_per_specialist_run():
    with groups(EVAL_GROUPS='cross') as P:
        js = P.jobs()
        expected = list(P.EXPECTED)
    assert len(js) == 40 and all(j['group'] == 'cross' for j in js)
    names = set()
    for j in js:
        name = j['model'][len(REG.PREFIX):]
        m = re.fullmatch(r'xmask-solo-([A-Z]+)-s(30[1-5])', name)
        assert m, name
        r, s = m.group(1), int(m.group(2))
        names.add((r, s))
        others = [c for c in R8 if c != r]
        assert j['regimes'].split() == others and j['n'] == 100
        assert (j['env'], j['device'], j['omp'], j['tt'], j['methods'], j['limit']) == \
            ('exact', 'cpu', 1, 1, 'greedy', 0)
        assert (j['tag'], j['folder'], j['data'], j['out_root']) == \
            ('_cross_xmask', '10x25_cross_xmask', 'data/FAMILY/10x25', 'test_results/FAMILY')
        assert j['files'] == ['test_results/FAMILY/10x25_cross_xmask/Result_greedy+10x25+family+%s_%s.npy'
                              % (name, c) for c in others] + \
            ['test_results/FAMILY/10x25_cross_xmask/summary_10x25+family+%s_cross_xmask.md' % name]
        assert (j['ckpt'], j['log']) == (REG.checkpoint('specialist', r, s, 'exact'),
                                         REG.train_log('specialist', r, s, 'exact'))
        assert P.preflight_of(j) == 'exact' and j['id'] == 'exact:10x25_cross:' + name
    assert names == {(r, s) for r in R8 for s in (301, 302, 303, 304, 305)}
    arrays = [f for j in js for f in j['files'] if f.endswith('.npy')]
    assert len(arrays) == len(set(arrays)) == 280
    planned = {f for j in js for f in j['files']}
    assert len(expected) == len(set(expected)) and set(expected) == planned


@needs_results
def test_cross_arrays_never_take_the_path_of_another_groups_file(monkeypatch):
    """The off-combination arrays share 19 names per seed with arrays of the principal
    10x25 campaign; in their own folder no path coincides with a file any other group
    plans, and no registry path of a principal arm points into that folder."""
    with groups(EVAL_GROUPS=ALL_GROUPS + ' cross', HGNN_LR='2e-4', HGNN_SOLO='1') as P:
        js = P.jobs()
    cross = {f for j in js if j['group'] == 'cross' for f in j['files']}
    other = {f for j in js if j['group'] != 'cross' for f in j['files']}
    assert cross and not cross & other
    principal_10 = {f for j in js if j['group'] == 'principal' and j['folder'] == '10x25_xmask'
                    for f in j['files']}
    same_name = {os.path.basename(f) for f in cross} & {os.path.basename(f) for f in principal_10}
    assert len(same_name) == 19 * 5          # the reason for the folder of its own
    monkeypatch.setenv('PACT_ENV', 'exact')
    try:
        for r in R8:
            for s in REG.PLANNED_SEEDS['exact']:
                for c in R8:
                    assert REG.result_path('specialist', r, s, '10x25', c) not in cross
                    assert REG.result_path('specialist', r, s, REG.CROSS_FOLDER, c) in cross or c == r
        assert REG.res_dir('10x25') != REG.res_dir(REG.CROSS_FOLDER)
        assert REG.summary_tag(REG.CROSS_FOLDER) == '_cross_xmask' and REG.summary_tag('10x25') == '_xmask'
    finally:
        monkeypatch.delenv('PACT_ENV', raising=False)
    # the completeness check skips every folder ending in _xmask when it reads the
    # stored legacy campaigns, so the new folder is never taken for a legacy campaign
    assert (REG.CROSS_FOLDER + REG.FOLDER_SUFFIX['exact']).endswith(P.XS)


@needs_results
def test_cross_group_is_off_by_default_and_adds_only_its_jobs():
    with groups() as P:
        before, pre_before, groups_before = P.jobs(), P.preflights(), list(P.GROUPS)
    assert 'cross' not in groups_before and not any(j['group'] == 'cross' for j in before)
    with groups(CROSS='1') as P:
        after, pre = P.jobs(), P.preflights()
        counts = io.StringIO()
        with contextlib.redirect_stdout(counts):
            P.counts()
    assert [j for j in after if j['group'] != 'cross'] == before and pre == pre_before
    line = next(ln for ln in counts.getvalue().splitlines() if ln.startswith('cross') and '10x25_cross' in ln)
    jobs_n, files_n, rollouts, hours = line.split()[-4:]
    assert (int(jobs_n), int(files_n), int(rollouts)) == (40, 320, 28000)
    assert abs(float(hours) - 40 * (0.741 * 700 + 5.0) / 3600) < 0.06
    with groups(EVAL_GROUPS='cross') as P:
        assert [p['name'] for p in P.preflights()] == ['xdet', 'xcross_policy', 'xcross_rules', 'xdet_rules']


def build_cmd(*args):
    """The evaluation queue's build_cmd function run alone: the command it builds."""
    with open(os.path.join(ROOT, 'scripts', 'run_eval_newseeds.sh')) as f:
        fn = re.search(r'^build_cmd\(\) \{.*?^\}\n', f.read(), re.S | re.M).group(0)
    script = ('PY=/py; MASK_OPTS="--first_mask exact --mask_ties exact"\n' + fn
              + 'build_cmd "$@"; printf "%s\\n" "${CMD[@]}"\n')
    r = subprocess.run(['bash', '-c', script, 'build_cmd'] + [str(a) for a in args],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.split('\n')[:-1]


@needs_results
def test_queue_runs_the_cross_job_into_the_folder_the_plan_names():
    with groups(EVAL_GROUPS='cross') as P:
        j = next(x for x in P.jobs() if x['id'] == 'exact:10x25_cross:xmask-solo-N-s302')
    cmd = build_cmd('7', j['device'], j['omp'], j['tt'], j['extra'], j['model'], j['methods'], j['data'],
                    j['base'], j['tag'], j['limit'], j['regimes'], j['token_mode'], j['blind'],
                    j['best_known'], j['sched'], 'exact', '/stage/out', j['rule_set'])
    assert cmd == ['env', '-u', 'PACT_FIRST_MASK', '-u', 'PACT_MASK_TOL', 'OMP_NUM_THREADS=1',
                   'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'taskset', '-c', '7', '/py',
                   'eval_family.py', '--family_model', '10x25+family+xmask-solo-N-s302', '--data',
                   'data/FAMILY/10x25', '--base', 'data/PPVC/10x25+ppvc-mixed', '--regimes',
                   'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW', '--methods', 'greedy', '--out_root',
                   '/stage/out', '--device', 'cpu', '--device_id', '', '--torch_threads', '1',
                   '--tag', '_cross_xmask', '--first_mask', 'exact', '--mask_ties', 'exact']
    # eval_family.py writes <out_root>/<basename of --data><--tag>/Result_<method>+<model>_<R>.npy
    # and summary_<model><--tag>.md, and the queue moves <stage>/out/<path> to <out_root>/<path>
    written = ['/stage/out/10x25_cross_xmask/Result_greedy+10x25+family+xmask-solo-N-s302_%s.npy' % c
               for c in R8[1:]] + ['/stage/out/10x25_cross_xmask/summary_10x25+family+xmask-solo-N-s302_cross_xmask.md']
    assert [f.replace(j['out_root'] + '/', '/stage/out/', 1) for f in j['files']] == written


# ================================================================== regression against BASE
def base_scripts(tmp_path):
    """arm_registry.py, eval_newseeds_plan.py and hgnn_select_lr.py of commit BASE in a
    folder of their own; skipped when git or the commit is not available."""
    d = tmp_path / 'base_scripts'
    d.mkdir()
    for name in ('arm_registry.py', 'eval_newseeds_plan.py', 'hgnn_select_lr.py'):
        r = subprocess.run(['git', '-C', ROOT, 'show', f'{BASE}:scripts/{name}'],
                           capture_output=True, text=True)
        if r.returncode != 0:
            pytest.skip(f'commit {BASE[:7]} is not available: {r.stderr.strip()}')
        (d / name).write_text(r.stdout)
    return str(d)


REGISTRY_DUMP = r'''
import json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import arm_registry as R
out = {}
FOLDERS = ['10x25', '10x25_cpuref', '10x25_blindref', '10x25_oodsetup', '20x25', '30x25', '50x25',
           '80x25', '50x25n30', '80x25n30', '10x25_firstmask', '50x25_pactwarm']
for e in R.ENVIRONMENTS:
    os.environ['PACT_ENV'] = e
    for arm in R.ARMS:
        for combo in (R.COMBOS if arm == 'specialist' else (None,)):
            for s in R.PLANNED_SEEDS[e]:
                k = f'{e}|{arm}|{combo}|{s}'
                out[k] = [R.model(arm, combo, s, e), R.suffix(arm, s, e), R.tag(arm, combo, s, 'greedy', e),
                          R.checkpoint(arm, combo, s, e), R.config_path(arm, combo, s, e),
                          R.train_log(arm, combo, s, e), R.vali_log(arm, combo, s, e)]
                out[k + '|paths'] = [R.result_path(arm, combo, s, f, r) for f in FOLDERS for r in R.COMBOS]
    for s in R.PLANNED_SEEDS[e]:
        if e == 'exact' or s in (301, 302, 303, 304, 305):
            out[f'{e}|cache|{s}'] = [R.schedule_cache(s, z, e) for z in ('50x25', '80x25n30')]
    out[f'{e}|folders'] = [[R.res_dir(f), R.summary_tag(f), R.rule_dir(f), R.fixed_dir(f)] for f in FOLDERS]
for stem in R.SECONDARY_ARMS:
    for s in R.SECONDARY_SEEDS:
        out[f'secondary|{stem}|{s}'] = [R.secondary_model(stem, s), list(R.secondary_paths(stem, s))]
for stem, (seeds, _) in R.OTHER_ARMS.items():
    for s in seeds:
        out[f'other|{stem}|{s}'] = [R.other_model(stem, s), R.other_tag(stem, s)]
for arm in R.ARMS:
    for combo in (R.COMBOS if arm == 'specialist' else (None,)):
        for s in R.PUB_SEEDS:
            out[f'pub|{arm}|{combo}|{s}'] = [R.pub_model(arm, combo, s), list(R.pub_paths(arm, combo, s)),
                                             R.pub_vali_log(arm, combo, s)]
for s in R.PUB_SEEDS:
    out[f'pubeq|{s}'] = [R.pub_eq_model(s), list(R.pub_eq_paths(s))]
if hasattr(R, 'pub_act_model'):                 # the active-rule arm of the public family
    for s in R.PUB_ACT_SEEDS:
        out[f'pubact|{s}'] = [R.pub_act_model(s), list(R.pub_act_paths(s))]
if hasattr(R, 'pub_rule_model'):                # the public family under the active rule
    for arm in R.PUB_RULE_STEM:
        for combo in (R.COMBOS if arm == 'specialist' else (None,)):
            for s in R.PUB_ACTFAM_SEEDS:
                out[f'pubrule|{arm}|{combo}|{s}'] = [R.pub_rule_model(arm, combo, s),
                                                     list(R.pub_rule_paths(arm, combo, s)),
                                                     R.pub_rule_counterpart(arm, combo, s)]
for name in ('COMBOS', 'TRAIN_COMBOS', 'HELD_OUT', 'CONTROL_COMBOS', 'PREFIX', 'ENVIRONMENTS',
             'DEFAULT_ENV', 'DEFAULT_SEEDS', 'PLANNED_SEEDS', 'ARMS', 'FOLDER_SUFFIX', 'MASK_OPTIONS',
             'FOLDER_ALIAS', 'LEGACY_TAG', 'SECONDARY_ARMS', 'SECONDARY_SEEDS', 'PUB_PREFIX', 'PUB_STEM',
             'PUB_SEEDS', 'PUB_EQ_STEM', 'PUB_EQ_UPDATES', 'PUB_RESULTS', 'PUB_SETS', 'PUB_SENS_SETS',
             'SPEC_EVAL_10x25', 'OTHER_ARMS', 'SEED_WORDS') + tuple(
        n for n in ('PUB_RULE_WORD', 'PUB_ACT_RULE', 'PUB_ACT_STEM', 'PUB_ACT_SEEDS',
                    'PUB_RULE_STEM', 'PUB_ACTFAM_SEEDS') if hasattr(R, n)):
    out['const|' + name] = json.loads(json.dumps(getattr(R, name), default=list))
os.environ['PACT_ENV'] = 'legacy'
os.environ['PACT_SEEDS'] = '301-305'
out['seeds|legacy'] = [list(R.seeds()), R.describe(), R.suffixes('released'), R.tags('specialist', 'LS')]
print(json.dumps(out))
'''


def dump(code, scripts, env_extra=None):
    env = clean_env()
    env.update(env_extra or {})
    r = subprocess.run([PY, '-c', code, scripts], cwd=ROOT, env=env, capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.splitlines()[-1])


def test_registry_returns_the_same_names_paths_and_seeds_for_every_existing_arm(tmp_path):
    old = dump(REGISTRY_DUMP, base_scripts(tmp_path))
    new = dump(REGISTRY_DUMP, os.path.join(ROOT, 'scripts'))
    assert len(old) > 300
    added = sorted(set(new) - set(old))
    pubrule = [f'pubrule|{a}|{c}|{s}' for a in ('released', 'control', 'specialist', 'eq')
               for c in (R8 if a == 'specialist' else [None]) for s in (301, 302, 303, 304, 305)]
    assert added == sorted([f'secondary|{a}|{s}' for a in ADDED for s in (301, 302, 303)]
                           + [f'pubact|{s}' for s in (301, 302, 303)]
                           + ['const|' + n for n in PUB_ACT_CONSTS]
                           + pubrule + ['const|' + n for n in PUB_ACTFAM_CONSTS])
    assert new['pubact|302'] == ['pubx-act-s302', [
        'trained_network/FAMILY/10x5+pubfamily+pubx-act-s302.pth', 'train_log/PUBFAMILY_pubx-act-s302.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-act-s302.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-act-s302.json']]
    assert [new['const|' + n] for n in PUB_ACT_CONSTS] == [{'active': 'act'}, 'active', 'pubx-act',
                                                           [301, 302, 303]]
    assert new['pubrule|control|None|302'] == ['pubx-ctlact-s302', [
        'trained_network/FAMILY/10x5+pubfamily+pubx-ctlact-s302.pth',
        'train_log/PUBFAMILY_pubx-ctlact-s302.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-ctlact-s302.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-ctlact-s302.json'], 'pubx-ctl-s302']
    # the policy of seeds 301-303 under the active rule is the active-rule arm itself
    for s in (301, 302, 303):
        assert new[f'pubrule|released|None|{s}'][:2] == new[f'pubact|{s}']
    assert [new['const|' + n] for n in PUB_ACTFAM_CONSTS] == [
        {'released': 'pubx-{W}', 'control': 'pubx-ctl{W}', 'specialist': 'pubx-solo{W}-{R}',
         'eq': 'pubx-eq{W}'}, [301, 302, 303, 304, 305]]
    changed = sorted(k for k in old if old[k] != new[k])
    assert changed == ['const|SECONDARY_ARMS']
    o, n = old['const|SECONDARY_ARMS'], new['const|SECONDARY_ARMS']
    assert {k: v for k, v in n.items() if k in o} == o and sorted(set(n) - set(o)) == sorted(ADDED)
    assert n['xmask-all8'][1] == 'joint-all8' and n['xmask-pair'][1] == 'jointpair'
    assert n['xmask-active'][1] == 'jointact'


PLAN_DUMP = r'''
import contextlib, io, json, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import eval_newseeds_plan as P
with contextlib.redirect_stdout(io.StringIO()):
    P.check()
js = P.jobs()
for j in js:
    j['pre'] = P.preflight_of(j)
print(json.dumps(dict(
    jobs=['|'.join(str(j[k]) for k in P.KEYS_JOB) + '|' + ' '.join(j['files']) + '|%.6f' % j['cost']
          for j in js],
    ids=[j['id'] for j in js], groups=[j['group'] for j in js],
    expected=list(P.EXPECTED),
    pre=['|'.join(str(p[k]) for k in P.KEYS_PRE) for p in P.preflights()])))
'''


def is_new(text):
    return any(a + '-' in text for a in ADDED)


@needs_results
@pytest.mark.parametrize('setting', ['default', 'all'])
def test_every_existing_job_and_expected_file_is_unchanged(tmp_path, setting):
    """The jobs (every field and file, in order) and the expected files of every group the
    plan had are those of commit BASE; the only additions are the jobs and files of the
    three arms added since (three seeds each, eight combinations and a summary per job)."""
    env = {} if setting == 'default' else dict(EVAL_GROUPS=ALL_GROUPS, HGNN_LR='2e-4', HGNN_SOLO='1')
    old = dump(PLAN_DUMP, base_scripts(tmp_path), env)
    new = dump(PLAN_DUMP, os.path.join(ROOT, 'scripts'), env)
    assert len(old['jobs']) > 300
    assert [ln for ln in new['jobs'] if not is_new(ln.split('|')[0])] == old['jobs']
    added = [ln for ln in new['jobs'] if is_new(ln.split('|')[0])]
    assert sorted(ln.split('|')[0] for ln in added) == sorted(
        f'exact:{a}_10x25:{a}-s{s}' for a in ADDED for s in (301, 302, 303))
    assert all(ln.split('|')[1] == 'secondary' for ln in added)
    assert [f for f in new['expected'] if not is_new(f)] == old['expected']
    new_files = [f for f in new['expected'] if is_new(f)]
    assert len(new_files) == len(set(new_files)) == len(ADDED) * 3 * 9
    assert all(f.startswith('test_results/FAMILY/10x25_xmask/') for f in new_files)
    assert new['pre'] == old['pre']
    assert 'cross' not in new['groups']


# ================================================================== queue: the two new arms
def queue_arms():
    """{arm: (regimes, updates, parameters, minutes, own JSON, flags)} of the extra queue."""
    text = open(EXTRA).read()
    var = dict(re.findall(r'^([A-Z0-9_]+)="([^"$]*)"$', text, re.M))
    body = re.search(r'^ARMS=\(\n(.*?)^\)', text, re.S | re.M).group(1)
    out = {}
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln.startswith('"'):
            continue
        ln = ln[1:-1].replace('\\"', '"')
        for k, v in var.items():
            ln = ln.replace('$' + k, v)
        arm, regs, upd, params, est, own, flags = ln.split('|')
        out[arm] = (regs, int(upd), int(params), int(est), own, flags)
    return out


def train_args(tmp_path, regs, upd, seed, name, flags, *extra):
    """The arguments the queue's train_cmd passes to train_family.py."""
    text = open(EXTRA).read()
    fn = re.search(r'^train_cmd\(\) \{.*?^\}\n', text, re.S | re.M).group(0)
    dr = re.search(r"^DR_SHIFT='(.*)'$", text, re.M).group(1)
    exe = tmp_path / 'args.sh'
    exe.write_text('#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done\n')
    exe.chmod(0o755)
    script = (fn + f'PY={exe} MASK_OPTS="--first_mask exact --mask_ties exact" DR_SHIFT={shlex.quote(dr)} '
              f'train_cmd "$@"\n')
    r = subprocess.run(['bash', '-c', script, 'train_cmd', regs, str(upd), str(seed), name, flags] + list(extra),
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.split('\n')[:-1]
    assert lines[0] == 'train_family.py'
    return lines[1:]


# The trainer's configuration without training. params.py parses the arguments, and the
# configuration edits of train_family.py are applied (at module level the input widths and
# the data source, in FamilyTrainer the station count of the factory and the two mask
# options). With build, the network is built on the CPU and its parameters counted.
RESOLVE = r'''
import json, os, sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
argsets = json.loads(sys.argv[1])
sys.argv = ['train_family.py']
import params
from params import configs
from ppvc_instance_generator import DEFAULT_FACTORY, SMALL_FACTORY, TIGHT_FACTORY, build_factory
from constraint_family.fjsp_env_family import resolve_mask_option
import torch
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cpu')
from model.PPO import PPO_initialize
out = []
for args, build in argsets:
    ns = params.parser.parse_args(args)
    configs.__dict__.clear()
    configs.__dict__.update(vars(ns))
    configs.fea_j_input_dim, configs.fea_m_input_dim = 12, 10
    if configs.data_source != 'FAMILY':
        configs.data_source = 'FAMILY'
    fac = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY, 'tight': TIGHT_FACTORY}[configs.ppvc_factory]
    configs.n_m = len(build_factory(fac)[0])
    configs.first_mask = resolve_mask_option('first_mask', configs.first_mask)
    configs.mask_ties = resolve_mask_option('mask_ties', configs.mask_ties)
    cfg = json.loads(json.dumps(vars(configs), default=str))
    n = sum(p.numel() for p in PPO_initialize().policy.parameters()) if build else None
    out.append(dict(config=cfg, params=n))
print(json.dumps(out))
'''

needs_configs = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, REF_CONFIG)),
                                   reason='needs the stored configuration of the released recipe')


def resolve(argsets):
    env = clean_env()
    env.pop('PACT_FIRST_MASK', None)
    env.pop('PACT_MASK_TOL', None)
    r = subprocess.run([PY, '-c', RESOLVE, json.dumps(argsets)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.splitlines()[-1])


def test_new_arms_are_appended_to_the_extra_queue():
    arms = queue_arms()
    assert list(arms)[-3:] == list(ADDED)
    assert arms['xmask-all8'][:3] == (ALL8, 2000, 32786)
    assert json.loads(arms['xmask-all8'][4]) == {'family_regimes': ALL8} and arms['xmask-all8'][5] == ''
    assert arms['xmask-pair'][:3] == (SIX, 2000, 32914)
    assert json.loads(arms['xmask-pair'][4]) == {'pair_extra': 'sw'} and arms['xmask-pair'][5] == '--pair_extra sw'
    assert arms['xmask-active'][:3] == (SIX, 2000, 32786)
    assert json.loads(arms['xmask-active'][4]) == {'mask_mode': 'active'}
    assert arms['xmask-active'][5] == '--mask_mode active'
    assert set(arms) == set(REG.SECONDARY_ARMS)


@needs_configs
def test_queue_parameter_counts_and_probe_diffs_hold_without_training(tmp_path):
    """For every arm of the extra queue, the probe's arguments (one update on the CPU)
    resolve to a network with the parameter count the queue asserts, and the queue's own
    configuration check (the Python of its probe) accepts the resolved configuration,
    with the differences from the released recipe named below."""
    arms = queue_arms()
    sets = [(train_args(tmp_path, regs, 1, 301, '_xeprobe', flags, '--device', 'cpu'), True)
            for regs, _, _, _, _, flags in arms.values()]
    res = dict(zip(arms, resolve(sets)))
    text = open(EXTRA).read()
    check = re.search(r"<< 'PYEOF' \|\| return 1\n(.*?)^PYEOF$", text, re.S | re.M).group(1)
    dr = re.search(r"^DR_SHIFT='(.*)'$", text, re.M).group(1)
    repo = tmp_path / 'probe_repo' / 'train_log' / 'FAMILY'
    repo.mkdir(parents=True)
    (repo / os.path.basename(REF_CONFIG)).write_text(open(os.path.join(ROOT, REF_CONFIG)).read())
    base = {'device', 'first_mask', 'mask_ties', 'max_updates', 'model_suffix', 'seed_train'}
    own_key = {'xmask-eq': None, 'xmask-sparse': 'reward_mode', 'xmask-dzero': 'delta_mode',
               'xmask-dmax': 'delta_mode', 'xmask-wthird': 'family_regimes', 'xmask-film': 'use_film',
               'xmask-dr': 'dr_shift', 'xmask-all8': 'family_regimes', 'xmask-pair': 'pair_extra',
               'xmask-active': 'mask_mode'}
    for arm, (regs, _, params, _, own, _) in arms.items():
        assert res[arm]['params'] == params, (arm, res[arm]['params'], params)
        (repo / 'config_10x25+family+_xeprobe.json').write_text(json.dumps(res[arm]['config']))
        env = dict(clean_env(), ARM=arm, OWN=own, DR_SHIFT=dr, WANT_REG=regs, SUF='_xeprobe')
        r = subprocess.run([PY, '-c', check], cwd=str(tmp_path / 'probe_repo'), env=env,
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, (arm, r.stdout, r.stderr)
        diff = set(json.loads(re.search(r'only in (\[.*\])', r.stdout).group(1).replace("'", '"')))
        assert diff == base | ({own_key[arm]} if own_key[arm] else set()), (arm, diff)
    assert res['xmask-pair']['params'] - res['xmask-eq']['params'] == 2 * 64   # two channels into 64 units


@needs_configs
def test_new_arms_differ_from_the_released_policys_training_run_in_their_option_only(tmp_path):
    """The training command of xmask-v1 (principal queue) and those of the three arms
    added to the extra queue, each resolved as the trainer writes its configuration: the only
    differences are the run name and the arm's own option."""
    text = open(PRINCIPAL_Q).read()
    m = re.search(r'\$tmo taskset -c \$CORES \$PY (train_family\.py .*?)> "\$log"', text, re.S)
    cmd = m.group(1).replace('\\\n', ' ')
    for k, v in (('"$regs"', SIX), ('"$upd"', '2000'), ('"$seed"', '301'), ('"$name"', 'xmask-v1-s301'),
                 ('$MASK_OPTS', '--first_mask exact --mask_ties exact')):
        cmd = cmd.replace(k, v)
    v1 = shlex.split(cmd)[1:]
    arms = queue_arms()
    sets = [(v1, False)] + [(train_args(tmp_path, arms[a][0], arms[a][1], 301, f'{a}-s301', arms[a][5]), False)
                            for a in ADDED]
    res = resolve(sets)
    ref = res[0]['config']
    assert ref['family_regimes'] == SIX and ref['max_updates'] == 2000 and ref['use_film'] is False
    for a, r in zip(ADDED, res[1:]):
        cfg = r['config']
        diff = {k: (ref.get(k), cfg.get(k)) for k in set(ref) | set(cfg) if ref.get(k) != cfg.get(k)}
        want = {'model_suffix': ('xmask-v1-s301', f'{a}-s301')}
        want.update({'xmask-all8': {'family_regimes': (SIX, ALL8)}, 'xmask-pair': {'pair_extra': ('none', 'sw')},
                     'xmask-active': {'mask_mode': ('nondelay', 'active')}}[a])
        assert diff == want, (a, diff)


# ================================================================== analysis: p100 on synthetic trees
N_INST = 6
BOOT = '300'
T0 = time.time() - 3600               # checkpoints are older than every array
SEEDS5 = (301, 302, 303, 304, 305)


def rules(c):
    return set() if c == 'N' else set(c)


def effect(r, c):
    """Known difference in percent of specialist r run under c from the policy."""
    if r == c:
        return 0.0
    if r == 'LSW':
        return 0.3                    # within 1%, above zero
    return 2.0 if rules(c) - rules(r) else -0.6


def base_ms(seed, combo):
    rng = np.random.default_rng(seed * 100 + R8.index(combo))
    k = 10.0 if seed > 303 else 1.0   # a comparison across seeds would show
    return k * rng.integers(100, 160, N_INST).astype(float)


def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(text)
    return p


def save(root, rel, ms):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.stack([np.asarray(ms, float), np.full(len(ms), 0.5)], axis=1))


def write_run(root, run, cfg):
    ck = write(root, run.ckpt, 'checkpoint')
    os.utime(ck, (T0, T0))
    write(root, run.log, '[train_family] done in 1.00 h; best rel=0.900000\n')
    write(root, run.config, json.dumps(cfg))


def cfg(seed, updates, regimes):
    return dict(seed_train=seed, max_updates=updates, num_envs=20, family_regimes=regimes,
                first_mask='exact', mask_ties='exact')


@pytest.fixture
def x(monkeypatch):
    """p100 in the exact environment with seeds 301-305 and outputs under the tree."""
    for v in ('PACT_ENV', 'PACT_RESULTS', 'PACT_SEEDS', 'PACT_OUT'):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    import p100_cross_eval as X
    return X


def build(root, monkeypatch, X):
    monkeypatch.setenv('PACT_OUT', os.path.join(root, 'out'))
    for r in R8:
        for i in range(N_INST):
            write(root, f'data/FAMILY/10x25/desc/{r}/instance_{i:03d}.desc.json',
                  json.dumps({'base_instance': f'data/PPVC/10x25+ppvc-mixed/instance_{i:03d}', 'regime': r}))
    main = {combo: set(regs) for arm, combo, regs in X.PLAN.principal_runs('10x25') if arm == 'specialist'}
    for s in SEEDS5:
        run = X.policy_run(s)
        write_run(root, run, cfg(s, 2000, SIX))
        for c in R8:
            save(root, run.array(c), base_ms(s, c))
        for r in R8:
            own = X.specialist_run(r, s, X.FOLDER)
            write_run(root, own, cfg(s, 1000, r))
            for c in main[r]:          # own combination and the principal campaign's others
                save(root, own.array(c), base_ms(s, c) * (1 + effect(r, c) / 100))
            cross = X.specialist_run(r, s, X.CROSS)
            for c in X.off(r):
                save(root, cross.array(c), base_ms(s, c) * (1 + effect(r, c) / 100))


def run_x(X, root):
    return X.main(['--root', root, '--boot', BOOT])


def macros(path):
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(path).read()))


def out_path(root, i=0):
    return os.path.join(root, 'out', ('manuscript/macros_crosseval.tex', 'results/cross_eval.json',
                                      'results/cross_eval.md')[i])


def test_known_matrix_counts_and_pooled_rows(tmp_path, monkeypatch, x):
    root = str(tmp_path)
    build(root, monkeypatch, x)
    res = run_x(x, root)
    m = macros(out_path(root))
    for r in R8:
        for c in R8:
            want = x.f2(effect(r, c))
            name = f'Xev{r}On{c}'
            assert (m[name], m[name + 'Lo'], m[name + 'Hi']) == (want, want, want), name
            d = effect(r, c)
            assert m[name + 'Dir'] == ('behind' if d > 0 else 'ahead' if d < 0 else 'neither')
            assert m[name + 'VOne'] == ('equivalent' if abs(d) < 1 else
                                        'inferior' if d > 0 else 'non-inferior')
        pooled = np.mean([effect(r, c) for c in R8 if c != r])
        assert m[f'Xev{r}Off'] == x.f2(pooled) and m[f'Xev{r}OffLo'] == x.f2(pooled)
    off = [effect(r, c) for r in R8 for c in R8 if c != r]
    assert (m['XevOffN'], m['XevSeedN'], m['XevInstN'], m['XevBoot']) == ('56', '5', str(N_INST), BOOT)
    assert m['XevOffBehindN'] == str(sum(d > 0 for d in off))
    assert m['XevOffAheadN'] == str(sum(d < 0 for d in off))
    assert m['XevOffEqOneN'] == str(sum(abs(d) < 1 for d in off))
    assert (m['XevLSWOffBehindN'], m['XevLSWOffEqOneN'], m['XevLSWOffAheadN']) == ('7', '7', '0')
    assert m['XevLSWOff'] == '+0.30' and m['XevLSWOffVOne'] == 'equivalent'
    # 19 off-combination arrays per seed are compared with the principal campaign
    assert res['compared'] == 19 * 5 and res['seeds'] == list(SEEDS5)
    j = json.load(open(out_path(root, 1)))
    assert j['cells']['LSW|N']['k'] == 5 and j['cells']['LSW|N']['n'] == N_INST
    assert j['lsw_pooled_off']['c'] == 7 and j['counts']['lsw']['cells'] == 7
    md = open(out_path(root, 2)).read()
    assert '| LSW | +0.30 [+0.30, +0.30]' in md and '95 off-combination arrays' in md


ADD = {'Lag': [('N', 'L'), ('S', 'LS'), ('W', 'LW'), ('SW', 'LSW')],
       'Setup': [('N', 'S'), ('L', 'LS'), ('W', 'SW'), ('LW', 'LSW')],
       'Window': [('N', 'W'), ('L', 'LW'), ('S', 'SW'), ('LS', 'LSW')]}


def test_single_rule_changes_grouped_by_rule_and_direction(tmp_path, monkeypatch, x):
    """The 24 cells whose combinations differ by one rule: four per rule and direction, the
    twelve of a direction equal to the rule-change pairs of the family analysis, and the
    pooled means and counts of the known tree (a rule the specialist has not met costs
    2.0, a rule it has met and that is absent -0.6, the LSW specialist +0.3)."""
    G = x.change_groups()
    assert list(G) == ['AddLag', 'DropLag', 'AddSetup', 'DropSetup', 'AddWindow', 'DropWindow',
                       'AddAll', 'DropAll']
    for rule, pairs in ADD.items():
        assert G[f'Add{rule}'][0] == pairs
        assert G[f'Drop{rule}'][0] == [(b, a) for a, b in pairs]
    up = [p for rule in ('Lag', 'Setup', 'Window') for p in ADD[rule]]
    assert G['AddAll'][0] == up and G['DropAll'][0] == [(b, a) for a, b in up]
    assert len(set(up)) == 12 and all(rules(b) - rules(a) and not rules(a) - rules(b)
                                      and len(rules(b)) == len(rules(a)) + 1 for a, b in up)
    import p88_family_inference as F
    assert set(F.PAIRS) == set(up)

    root = str(tmp_path)
    build(root, monkeypatch, x)
    res = run_x(x, root)
    m = macros(out_path(root))
    assert m['XevChgRulePairN'] == '4'

    def g_of(r, c):
        """The policy minus the specialist in percent of the specialist's makespan."""
        e = effect(r, c)
        return -100 * e / (100 + e)

    for g, (keys, _) in G.items():
        want = float(np.mean([g_of(r, c) for r, c in keys]))
        pk = res['groups'][g]
        assert abs(pk['mean'] - want) < 1e-9 and abs(pk['lo'] - want) < 1e-9 \
            and abs(pk['hi'] - want) < 1e-9, g
        assert pk['c'] == len(keys) and pk['k'] == 5 and pk['n'] == N_INST
        assert pk['pairs'] == [list(k) for k in keys]
    for g in ('AddLag', 'AddSetup', 'AddWindow', 'AddAll'):
        name = f'XevChg{g}'
        assert (m[name], m[name + 'Lo'], m[name + 'Hi']) == ('-1.96', '-1.96', '-1.96')
        assert m[name + 'VOne'] == 'non-inferior' and m[name + 'Dir'] == 'ahead'
    # a rule removed: three specialists 0.6% shorter than the policy, the LSW specialist 0.3% longer
    assert res['groups']['DropLag']['mean'] > 0 and m['XevChgDropAllDir'] == 'behind'
    assert not res['family_check']['compared']          # the tree holds no family record
    j = json.load(open(out_path(root, 1)))
    assert j['groups']['DropWindow']['pairs'] == [['W', 'N'], ['LW', 'L'], ['SW', 'S'], ['LSW', 'LS']]
    md = open(out_path(root, 2)).read()
    assert '## Single-rule changes grouped by the rule and the direction' in md
    assert '| AddLag | N>L, S>LS, W>LW, SW>LSW | -1.96 [-1.96, -1.96]' in md
    assert 'AddAll was not compared with the family record' in md


def test_grouped_changes_are_checked_against_the_family_record(tmp_path, monkeypatch, x):
    """With a family record of the same seeds under PACT_OUT, the pooled mean of its twelve
    rule-change pairs must equal AddAll; another mean or other pairs are refused, and a
    record of other seeds is left out of the comparison."""
    root = str(tmp_path)
    build(root, monkeypatch, x)
    res = run_x(x, root)
    mean = res['groups']['AddAll']['mean']
    pairs = {f'{a}To{b}': [a, b] for a, b in res['groups']['AddAll']['pairs']}
    rec = os.path.join(root, 'out', 'results', 'family_inference.json')

    def record(mean, pairs=pairs, seeds=SEEDS5):
        write(root, os.path.relpath(rec, root), json.dumps(dict(
            seeds=list(seeds), boot=int(BOOT),
            rule_change=dict(Pol=dict(pairs=pairs, pooled=dict(mean=mean, lo=mean, hi=mean))))))

    record(mean)
    res = run_x(x, root)
    assert res['family_check']['compared'] and res['family_check']['same_replicates']
    assert 'AddAll holds the twelve rule-change pairs of the family record' in open(out_path(root, 2)).read()
    record(mean, seeds=(301, 302, 303))
    assert not run_x(x, root)['family_check']['compared']
    record(mean + 0.01)
    with pytest.raises(SystemExit, match=r'^refused: the twelve rule-change pairs pooled'):
        run_x(x, root)
    assert not os.path.exists(out_path(root))
    record(mean, pairs=dict(list(pairs.items())[:11]))
    with pytest.raises(SystemExit, match=r'^refused: the rule-change pairs of'):
        run_x(x, root)


def test_pooled_row_keeps_the_combinations_of_a_base_instance_together(tmp_path, monkeypatch, x):
    """Specialist N gets +z_i on L and -z_i on S for base instance i (z varies by instance
    and seed), zero elsewhere: the pooled row has mean zero and a zero-width interval,
    while the two cells have wide intervals."""
    root = str(tmp_path)
    build(root, monkeypatch, x)
    for s in SEEDS5:
        z = np.linspace(0.5, 3.0, N_INST) * (1 + 0.1 * (s - 303))
        for c, sign in (('L', 1.0), ('S', -1.0)):
            ms = base_ms(s, c) * (1 + sign * z / 100)
            for folder in (x.CROSS, x.FOLDER):
                p = os.path.join(root, x.specialist_run('N', s, folder).array(c))
                if os.path.exists(p):
                    save(root, x.specialist_run('N', s, folder).array(c), ms)
        for c in ('W', 'LS', 'LW', 'SW', 'LSW'):
            save(root, x.specialist_run('N', s, x.CROSS).array(c), base_ms(s, c))
        save(root, x.specialist_run('N', s, x.FOLDER).array('W'), base_ms(s, 'W'))
    res = run_x(x, root)
    row = res['rows']['N']
    assert abs(row['mean']) < 1e-9 and abs(row['lo']) < 1e-9 and abs(row['hi']) < 1e-9
    assert res['cells']['N|L']['hi'] - res['cells']['N|L']['lo'] > 0.5


@pytest.mark.parametrize('damage', ['missing cross array', 'missing seed', 'principal differs',
                                    'wrong rows', 'other seed in configuration', 'other mask',
                                    'older than checkpoint'])
def test_refuses_and_leaves_no_output(tmp_path, monkeypatch, x, damage):
    root = str(tmp_path)
    build(root, monkeypatch, x)
    run_x(x, root)                                      # outputs of a complete tree exist
    assert os.path.exists(out_path(root))
    cross = x.specialist_run('LSW', 303, x.CROSS)
    p = os.path.join(root, cross.array('N'))
    if damage == 'missing cross array':
        os.remove(p)
    elif damage == 'missing seed':
        for c in x.off('W'):
            os.remove(os.path.join(root, x.specialist_run('W', 305, x.CROSS).array(c)))
    elif damage == 'principal differs':
        save(root, x.specialist_run('LSW', 303, x.FOLDER).array('N'), base_ms(303, 'N') * 1.01)
    elif damage == 'wrong rows':
        np.save(p, np.ones((N_INST - 1, 2)) * 100)
    elif damage == 'other seed in configuration':
        write(root, cross.config, json.dumps(cfg(302, 1000, 'LSW')))
    elif damage == 'other mask':
        c = cfg(303, 1000, 'LSW')
        c['mask_ties'] = 'legacy'
        write(root, cross.config, json.dumps(c))
    else:
        os.utime(p, (T0 - 60, T0 - 60))
    with pytest.raises(SystemExit) as e:
        run_x(x, root)
    msg = str(e.value)
    assert msg.startswith('refused'), msg
    assert {'missing cross array': 'missing', 'missing seed': 'xmask-solo-W-s305',
            'principal differs': 'differ', 'wrong rows': 'shape', 'other seed in configuration': 'seed 302',
            'other mask': 'mask_ties', 'older than checkpoint': 'older'}[damage] in msg
    for i in range(3):
        assert not os.path.exists(out_path(root, i))


def test_refuses_outside_the_exact_environment_and_without_an_output_root(tmp_path, monkeypatch, x):
    root = str(tmp_path)
    build(root, monkeypatch, x)
    monkeypatch.delenv('PACT_OUT')
    with pytest.raises(SystemExit, match=r'^refused: .*PACT_OUT'):
        run_x(x, root)
    monkeypatch.setenv('PACT_OUT', os.path.join(root, 'out'))
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(SystemExit, match=r'^refused: .*exact environment'):
        run_x(x, root)


@needs_table
def test_macro_names_prefix_and_collisions(tmp_path, monkeypatch, x):
    root = str(tmp_path)
    build(root, monkeypatch, x)
    res = run_x(x, root)
    text = open(out_path(root)).read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    # 64 cells, 8 pooled rows, 4 design numbers, 6 counts, 8 groups of single-rule changes
    # and the number of cells of a rule and a direction
    assert len(names) == len(set(names)) == res['n_macros'] == \
        64 * 6 + 8 * 6 + 4 + 6 + 8 * 6 + 1
    assert all(re.fullmatch(r'Xev[A-Za-z]+', n) for n in names)
    assert text.startswith('% Generated by scripts/p100_cross_eval.py -- do not hand-edit')
    vals = re.findall(r'\\newcommand\{\\[A-Za-z]+\}\{([^}]*)\}', text)
    assert not any(re.search(r'(?<![A-Za-z-])(nan|inf)(?![A-Za-z-])', v) for v in vals)
    # the staged table uses macros of this script only: every mean and bound of the 64 cells
    # and of the 8 pooled rows (its last column)
    used = set(re.findall(r'\\(Xev[A-Za-z]+)', open(TABLE).read()))
    assert used == ({f'Xev{r}On{c}{s}' for r in R8 for c in R8 for s in ('', 'Lo', 'Hi')}
                    | {f'Xev{r}Off{s}' for r in R8 for s in ('', 'Lo', 'Hi')})
    assert used <= set(names)
    write(root, 'manuscript/sections_new/tables/other.tex', '\\newcommand{\\XevOther}{1}\n')
    with pytest.raises(SystemExit, match=r'^refused: the prefix Xev is in use'):
        run_x(x, root)


def test_prefix_is_free_in_the_repository():
    """No macro file or staged section of the repository defines a name with the prefix."""
    import p100_cross_eval as X
    other, n_files = X.defined_elsewhere(ROOT, {os.path.abspath(os.path.join(ROOT, X.OUT_FILES[0]))})
    assert n_files > 10 and not [k for k in other if k.startswith(X.PREFIX)]


@needs_table
def test_staged_table_layout():
    t = open(TABLE).read()
    assert '\\footnotesize' in t and '\\toprule' in t and '\\bottomrule' in t
    assert '{\\linewidth}' in t and '\\label{tab:cross-eval}' in t
    assert not re.search(r'\\(bfseries|textbf|boldmath)', t)
    cap = re.search(r'\\caption\{(.*?)\}\n\\label', t, re.S).group(1)
    assert len(re.findall(r'\.(\s|$)', cap)) <= 4
    for word in ('xmask', 'solo', 'PACT', 'arm', 'regime'):
        assert word not in cap


@needs_results
def test_analysis_reads_only_files_the_plan_writes(monkeypatch, x):
    monkeypatch.chdir(ROOT)
    with groups(EVAL_GROUPS='principal cross') as P:
        planned = {f for j in P.jobs() for f in j['files']}
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-305')
    x.check_layout()
    read = [run.array(c) for run, combos, _ in x.reads(SEEDS5) for c in combos]
    assert len(read) == len(set(read)) == 5 * (8 + 8 + 56)
    assert set(read) <= planned, sorted(set(read) - planned)[:5]
