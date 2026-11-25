"""The job plan of scripts/run_eval_newseeds.sh and its completeness check are
derived independently (the plan from the campaign tables, the check from the
stored legacy files). They must ask for the same files, in every group.

The hgnn group (HGNN=1) adds only its own jobs and pre-flights, its jobs run
backbones/hgnn/eval_family_hgnn.py with the settings of the plan into the files
the plan names, and the commands of the other jobs do not change (the queue's
build_cmd function is run alone, so no command is executed).

Run from the repository root:  python -m pytest tests/test_eval_newseeds_plan.py -q
Reads test_results/FAMILY and data/FAMILY; skipped when they are absent.
"""
import contextlib
import glob
import importlib
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import arm_registry as REG  # noqa: E402

R8 = list(REG.COMBOS)
QUEUE = os.path.join(ROOT, 'scripts', 'run_eval_newseeds.sh')
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'test_results/FAMILY')) or
    not os.path.isdir(os.path.join(ROOT, 'data/FAMILY')),
    reason='needs the stored results and the instance data')

# Stored files of the earlier implementation that the check of some groups lists to derive
# what it expects; the release holds only those of the default groups.
TOKEN_ARRAYS = 'test_results/FAMILY/10x25/Result_greedy+10x25+family+joint-v1+*_*.npy'
LEGACY_SUMMARY = 'test_results/FAMILY/10x25/summary_10x25+family+jointblind-nlsw-s303.md'
HGNN_SOLO_ARRAY = 'test_results/FAMILY/10x25/Result_hgnn-certified-N-s301_N.npy'
HGNN_CONFIGS = tuple(f'train_log/FAMILY/config_10x25+family+hgnn-{r}-joint-s301.json'
                     for r in ('certified', 'published'))


def needs(paths, what):
    """Skip unless every path (a glob pattern matches at least one file) exists."""
    absent = [p for p in paths if not glob.glob(os.path.join(ROOT, p))]
    return pytest.mark.skipif(bool(absent), reason=f'needs {what} '
                              f'({", ".join(absent)}), which the release does not hold')


needs_tokens = needs([TOKEN_ARRAYS], 'the stored arrays of the token interventions of the '
                     'prior implementation')
needs_legacy = needs([LEGACY_SUMMARY], 'the stored records of seed 303 of the prior '
                     'implementation, from which the check of the group legacy45 derives '
                     'its files')
needs_hgnn_solo = needs([HGNN_SOLO_ARRAY], 'the stored arrays of the HGNN specialists of the '
                        'prior implementation, from which the check derives the specialists '
                        'of the group hgnn')
needs_hgnn_configs = needs(HGNN_CONFIGS, 'the stored configurations of the HGNN runs of the prior '
                           'implementation')


HGNN_ENV = ('HGNN', 'HGNN_LR', 'HGNN_SOLO')


@pytest.fixture(scope='module', params=['default', 'all', 'hgnn'])
def plan(request):
    os.chdir(ROOT)
    for k in ('EVAL_GROUPS', 'PUBLIC', 'LEGACY45', 'CROSS') + HGNN_ENV:
        os.environ.pop(k, None)
    if request.param == 'all':
        os.environ['PUBLIC'] = '1'
        os.environ['LEGACY45'] = '1'
    if request.param == 'hgnn':      # the default groups and the HGNN runs, specialists included
        os.environ.update(HGNN='1', HGNN_LR='2e-4', HGNN_SOLO='1')
    import eval_newseeds_plan as P
    P = importlib.reload(P)
    P.EXPECTED.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        P.check()
    yield P
    for k in ('PUBLIC', 'LEGACY45') + HGNN_ENV:
        os.environ.pop(k, None)
    importlib.reload(P)              # later importers see the default groups


@pytest.mark.parametrize('plan', ['default', pytest.param('all', marks=needs_legacy),
                                  pytest.param('hgnn', marks=needs_hgnn_solo)], indirect=True)
def test_plan_and_check_ask_for_the_same_files(plan):
    planned = {f for j in plan.jobs() for f in j['files']}
    # the rule jobs also write a summary file, which no stored campaign has
    planned_cmp = {f for f in planned if not os.path.basename(f).startswith('summary_rules')}
    expected = set(plan.EXPECTED)
    assert len(expected) == len(plan.EXPECTED), 'the check asks for a file twice'
    assert sorted(planned_cmp - expected)[:5] == []
    assert sorted(expected - planned_cmp)[:5] == []


def test_default_plan_is_exact_only_on_one_cpu_thread(plan):
    for j in plan.jobs():
        if j['env'] == 'exact':
            assert (j['device'], j['omp'], j['tt']) == ('cpu', 1, 1), j['id']
            assert j['tag'] == '_xmask' and j['folder'].endswith('_xmask'), j['id']
        else:
            assert 'legacy45' in plan.GROUPS and '_xmask' not in j['folder'], j['id']
    if 'legacy45' not in plan.GROUPS:
        assert all(j['env'] == 'exact' for j in plan.jobs())
        assert all(j['group'] != 'public' for j in plan.jobs())


def test_no_job_can_touch_a_stored_file(plan):
    """Every planned file is in an exact-mask folder, in a 30-instance folder, in a
    new schedule folder, or belongs to a legacy run with seed 304 or 305."""
    for j in plan.jobs():
        for f in j['files']:
            parts = f.split('/')
            ok = (any(p.endswith('_xmask') or p.endswith('n30') for p in parts)
                  or any(p.startswith('pactwarm_cache_xmask_') for p in parts)
                  or '-s304' in f or '-s305' in f
                  or any(p.startswith(('pactwarm_cache_s304', 'pactwarm_cache_s305')) for p in parts))
            assert ok, f'{j["id"]} plans {f}'


def test_job_ids_unique(plan):
    js = plan.jobs()
    assert len({j['id'] for j in js}) == len(js)


@needs_tokens
def test_token_modes_round_trip(plan):
    modes = plan.e3_modes()
    assert len(modes) == 17
    for tag in modes:
        assert plan.mode_tag(*plan.parse_mode(tag)) == tag


def test_rule_set_masked_on_every_exact_rules_job(plan):
    """Exact rules jobs (rules and public groups) use the masked rules; legacy rule
    jobs and every policy job keep the default, so their commands are unchanged."""
    js = plan.jobs()
    assert any(j['model'] == 'rules' for j in js)
    for j in js:
        want = 'masked' if (j['model'] == 'rules' and j['env'] == 'exact') else 'classical'
        assert j['rule_set'] == want, j['id']
        line = '|'.join(str({**j, 'pre': plan.preflight_of(j)}[k]) for k in plan.KEYS_JOB)
        assert line.split('|')[-1] == want and plan.KEYS_JOB[-1] == 'rule_set'
    pre = {p['name']: p for p in plan.preflights()}
    x = pre['xdet_rules']
    assert (x['rule_set'], x['kind'], x['mask'], x['model']) == ('masked', 'twice', 'exact', 'rules')
    assert all(p['rule_set'] == 'classical' for n, p in pre.items() if n != 'xdet_rules')


def test_sensitivity_group_mirrors_the_public_jobs():
    """Each pubsens job is the public job of its test set with only the record folder,
    the result folder and the group changed; plan and check ask for the same files."""
    import arm_registry as REG
    st, (base, _, _) = next(iter(REG.PUB_SENS_SETS.items()))
    if not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family', st)):
        pytest.skip(f'{st} records are not built')
    os.chdir(ROOT)
    keep = {k: os.environ.pop(k, None) for k in ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'LEGACY45')}
    import eval_newseeds_plan as P
    try:
        os.environ['EVAL_GROUPS'] = 'public pubsens'
        P = importlib.reload(P)
        P.EXPECTED.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            P.check()
        js = P.jobs()
        sens = [j for j in js if j['group'] == 'pubsens']
        pub = {j['id']: j for j in js if j['group'] == 'public'}
        assert len(sens) == len(REG.PUB_SEEDS) * 10 + 1
        for j in sens:
            k = pub[j['id'].replace(f'public_{st}', f'public_{base}')]
            assert j['data'] == k['data'].replace(base, st) and j['folder'] == st + P.XS
            assert j['files'] == [f.replace(f'/{base}{P.XS}/', f'/{st}{P.XS}/') for f in k['files']]
            assert P.preflight_of(j) == P.preflight_of(k)
            same = set(P.KEYS_JOB) - {'id', 'group', 'campaign', 'data', 'folder', 'pre'}
            assert {x: j[x] for x in same} == {x: k[x] for x in same}, j['id']
        planned = {f for j in sens for f in j['files']
                   if not os.path.basename(f).startswith('summary_rules')}
        assert planned == {f for f in P.EXPECTED if f'/{st}{P.XS}/' in f}
    finally:
        os.environ.pop('EVAL_GROUPS', None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(P)


# ------------------------------------------------------------------ the hgnn group
@contextlib.contextmanager
def groups(**env):
    """The plan module reloaded under env (the group switches cleared first), with the
    completeness check run once; reloaded with the defaults afterwards."""
    os.chdir(ROOT)
    names = ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'PUBACT', 'LEGACY45', 'CROSS') + HGNN_ENV
    keep = {k: os.environ.pop(k, None) for k in names}
    import eval_newseeds_plan as P
    try:
        os.environ.update(env)
        P = importlib.reload(P)
        P.EXPECTED.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            P.check()
        yield P
    finally:
        for k in names:
            os.environ.pop(k, None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(P)


def hgnn_names(rate, solo):
    """The run names of scripts/run_hgnn_exact_queue.sh, written out independently."""
    tag = '' if rate == '2e-4' else '-lr' + rate
    out = ['hgnn-certified-joint-s%d-xmask%s' % (s, tag) for s in (301, 302, 303, 304, 305)]
    out += ['hgnn-published-joint-s%d-xmask%s' % (s, tag) for s in (301, 302, 303)]
    if solo:
        out += ['hgnn-certified-%s-s301-xmask%s' % (r, tag) for r in R8]
    return out


@pytest.mark.parametrize('solo', ['0', pytest.param('1', marks=needs_hgnn_solo)])
def test_hgnn_group_adds_only_its_jobs_and_pre_flights(solo):
    with groups() as P:
        before, pre_before = P.jobs(), P.preflights()
    with groups(HGNN='1', HGNN_LR='1e-4', HGNN_SOLO=solo) as P:
        after, pre = P.jobs(), P.preflights()
        hj = [j for j in after if j['group'] == 'hgnn']
        assert [j for j in after if j['group'] != 'hgnn'] == before
        assert pre[:len(pre_before)] == pre_before
        assert [p['name'] for p in pre[len(pre_before):]] == ['hgnn_cross', 'hgnn_cross_pub']
        assert sorted(j['model'][len(REG.PREFIX):] for j in hj) == sorted(hgnn_names('1e-4', solo == '1'))
        for j in hj:
            name = j['model'][len(REG.PREFIX):]
            regs = R8 if '-joint-' in name else [name.split('-')[2]]
            assert (j['methods'], j['env'], j['device'], j['omp'], j['tt'], j['tag'], j['limit']) == \
                ('hgnn', 'exact', 'cpu', 1, 1, '_xmask', 0), j['id']
            assert (j['data'], j['n'], j['regimes']) == ('data/FAMILY/10x25', 100, ' '.join(regs))
            assert j['files'] == ['test_results/FAMILY/10x25_xmask/Result_%s_%s.npy' % (name, r)
                                  for r in regs]
            assert (j['ckpt'], j['log']) == ('trained_network/FAMILY/10x25+family+%s.pth' % name,
                                             'train_log/FAMILY_%s.log' % name)
            assert P.preflight_of(j) == 'hgnn'
        assert {f for j in hj for f in j['files']} == {f for f in P.EXPECTED if '/Result_hgnn-' in f}
    with groups(EVAL_GROUPS='hgnn', HGNN_LR='1e-4', HGNN_SOLO=solo) as P:
        assert [j['id'] for j in P.jobs()] == [j['id'] for j in sorted(hj, key=lambda j: (-j['cost'], j['id']))]
        assert [p['name'] for p in P.preflights()] == ['hgnn_cross', 'hgnn_cross_pub']


@needs_hgnn_configs
def test_hgnn_pre_flights_compare_two_stored_instances_per_reward():
    with groups(EVAL_GROUPS='hgnn', HGNN_LR='2e-4') as P:
        pre = {p['name']: p for p in P.preflights()}
    for name, reward in (('hgnn_cross', 'certified'), ('hgnn_cross_pub', 'published')):
        p, run = pre[name], 'hgnn-%s-joint-s301' % reward
        assert (p['kind'], p['device'], p['omp'], p['tt'], p['methods'], p['mask'], p['rule_set']) == \
            ('stored', 'cpu', 1, 1, 'hgnn', 'legacy', 'classical')
        assert (p['model'], p['cmp_tag'], p['tag'], p['limit'], p['regimes']) == \
            ('10x25+family+' + run, run, '-', 2, 'S LSW')
        assert (p['data'], p['folder'], p['stored']) == ('data/FAMILY/10x25', '10x25', '10x25')
        for r in ('S', 'LSW'):     # the stored arrays the pre-flight compares with
            assert np.load(os.path.join(ROOT, 'test_results/FAMILY/10x25/Result_%s_%s.npy' % (run, r))).shape[0] == 100
        with open(os.path.join(ROOT, 'train_log/FAMILY/config_10x25+family+%s.json' % run)) as f:
            cfg = json.load(f)
        assert (cfg['hgnn_mask'], cfg['hgnn_reward']) == ('pact', reward)


def test_hgnn_rate_and_specialist_options(tmp_path, monkeypatch):
    import eval_newseeds_plan as P
    for k in HGNN_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('HGNN_LR', '3e-4')
    with pytest.raises(SystemExit, match='not one of'):
        P.hgnn_rate()
    monkeypatch.delenv('HGNN_LR')

    def unavailable(root=None):
        raise SystemExit('REFUSING: a tuning run is missing')
    monkeypatch.setattr(P.HLR, 'select', unavailable)
    with pytest.raises(SystemExit, match='HGNN_LR'):
        P.hgnn_rate()
    monkeypatch.setattr(P.HLR, 'select', lambda root=None: ('4e-4', {}))
    assert P.hgnn_rate() == '4e-4'
    assert P.hgnn_name('certified', 'joint', 304, '4e-4') == 'hgnn-certified-joint-s304-xmask-lr4e-4'
    assert P.hgnn_name('published', 'joint', 301, '2e-4') == 'hgnn-published-joint-s301-xmask'
    monkeypatch.setenv('HGNN_SOLO', 'yes')
    with pytest.raises(SystemExit, match='HGNN_SOLO'):
        P.hgnn_solo_on('2e-4', str(tmp_path))
    monkeypatch.setenv('HGNN_SOLO', 'auto')
    assert not P.hgnn_solo_on('2e-4', str(tmp_path))
    log = tmp_path / 'train_log' / 'FAMILY_hgnn-certified-LS-s301-xmask.log'
    log.parent.mkdir(parents=True)
    log.write_text('Ep 1\n')
    assert P.hgnn_solo_on('2e-4', str(tmp_path))         # one specialist started: all eight
    assert not P.hgnn_solo_on('1e-4', str(tmp_path))     # the names of another rate
    monkeypatch.setenv('HGNN_SOLO', '0')
    assert not P.hgnn_solo_on('2e-4', str(tmp_path))


def build_cmd(*args):
    """The queue's build_cmd function run alone: the command array it builds."""
    with open(QUEUE) as f:
        fn = re.search(r'^build_cmd\(\) \{.*?^\}\n', f.read(), re.S | re.M).group(0)
    script = ('PY=/py; MASK_OPTS="--first_mask exact --mask_ties exact"\n' + fn
              + 'build_cmd "$@"; printf "%s\\n" "${CMD[@]}"\n')
    r = subprocess.run(['bash', '-c', script, 'build_cmd'] + [str(a) for a in args],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.split('\n')[:-1]


def job_args(j, core='7', root='/stage/out'):
    return [core, j['device'], j['omp'], j['tt'], j['extra'], j['model'], j['methods'], j['data'],
            j['base'], j['tag'], j['limit'], j['regimes'], j['token_mode'], j['blind'],
            j['best_known'], j['sched'], 'exact' if j['env'] == 'exact' else 'legacy', root,
            j['rule_set']]


def test_queue_builds_the_hgnn_command_and_keeps_the_others():
    with groups(EVAL_GROUPS='principal hgnn', HGNN_LR='2e-4', HGNN_SOLO='1') as P:
        js, pre = P.jobs(), {p['name']: p for p in P.preflights()}
    hj = [j for j in js if j['group'] == 'hgnn']
    assert len(hj) == 16
    for j in hj:
        name, regs = j['model'][len(REG.PREFIX):], j['regimes'].split()
        assert build_cmd(*job_args(j)) == [
            'env', '-u', 'PACT_FIRST_MASK', '-u', 'PACT_MASK_TOL', 'OMP_NUM_THREADS=1',
            'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'taskset', '-c', '7', '/py',
            'backbones/hgnn/eval_family_hgnn.py', '--family_model', j['model'],
            '--data_root', 'data/FAMILY', '--sizes', '10x25', '--regimes', ','.join(regs),
            '--device', 'cpu', '--threads', '1', '--out_root', '/stage/out', '--tag', '_xmask'], j['id']
        # the evaluator writes <out_root>/<size><tag>/Result_<model without its prefix>_<R>.npy,
        # and the queue moves <stage>/out/<path> to <out_root of the job>/<path>
        assert [f.replace(j['out_root'] + '/', '/stage/out/', 1) for f in j['files']] == \
            ['/stage/out/10x25_xmask/Result_%s_%s.npy' % (name, r) for r in regs]
    p = pre['hgnn_cross']
    cmd = build_cmd('7', p['device'], p['omp'], p['tt'], p['extra'], p['model'], p['methods'],
                    p['data'], p['base'], p['tag'], p['limit'], p['regimes'], 'true', '-', '-', '-',
                    p['mask'], '/pre/a', p['rule_set'])
    assert cmd[cmd.index('--regimes') + 1] == 'S,LSW' and '--tag' not in cmd
    assert cmd[-6:] == ['--threads', '1', '--out_root', '/pre/a', '--limit', '2']
    # an eval_family.py job keeps its command
    j = next(j for j in js if j['id'] == 'exact:10x25:xmask-v1-s301')
    assert build_cmd(*job_args(j)) == [
        'env', '-u', 'PACT_FIRST_MASK', '-u', 'PACT_MASK_TOL', 'OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'CUDA_VISIBLE_DEVICES=', 'taskset', '-c', '7', '/py', 'eval_family.py', '--family_model',
        '10x25+family+xmask-v1-s301', '--data', 'data/FAMILY/10x25', '--base',
        'data/PPVC/10x25+ppvc-mixed', '--regimes'] + R8 + [
        '--methods', 'greedy', '--out_root', '/stage/out', '--device', 'cpu', '--device_id', '',
        '--torch_threads', '1', '--tag', '_xmask', '--first_mask', 'exact', '--mask_ties', 'exact']
