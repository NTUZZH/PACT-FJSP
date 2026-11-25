"""The evaluation of the specialists at the production batches: its job plan and its runner.

  * scripts/production_specialists_plan.py: 80 jobs, each the evaluation plan's n30 job of the
    policy on the same set and seed with only the model, its checkpoint and log, the combination
    (R alone), the schedule export (none), the tag and folder (_xmask_spec) and the bookkeeping
    keys exchanged; the lanes i/K partition the jobs; a bad lane is refused;
  * the result folder <set>_xmask_spec is none of the folders scripts/p120_lower_bound_reference.py
    sweeps for feasible makespans (arm_registry.res_dir of the set and of its EXTRA_FOLDERS), and
    the file names match no reference name of the legacy sweep of eval_newseeds_plan.py check;
  * scripts/run_production_specialists_eval.sh LIST_ONLY=1 on an empty result tree: every printed
    command is build_cmd of scripts/run_eval_newseeds.sh on the n30 policy job of the same set
    and seed with exactly --family_model, --regimes, --tag and --save_schedules changed (and the
    staging --out_root); the pre-flight is the S specialist of seed 301 on two instances of
    50x25n30; nothing is written;
  * the runner skips a finished job, refuses a job whose array holds another number of rows and a
    job whose files partly exist (fake files in a temporary result tree), refuses a bad LANE and a
    missing CORES;
  * the pre-flight comparison accepts the stored makespans and refuses a changed one.

Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      tests/test_production_specialists_eval.py
"""
import os
import re
import shlex
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None, RESULTS=None,
                                        SEEDS=None, PACT_FIRST_MASK=None, PACT_MASK_TOL=None,
                                        EVAL_GROUPS=None, EXACT_SEEDS=None, PACT_ENV='exact',
                                        PACT_SEEDS='301-305')

import arm_registry as REG  # noqa: E402
import eval_newseeds_plan as PLAN  # noqa: E402
import production_specialists_plan as PS  # noqa: E402

RUNNER = os.path.join(ROOT, 'scripts', 'run_production_specialists_eval.sh')
PY = os.path.join(os.path.expanduser('~'), 'miniconda3', 'envs', 'fjsp', 'bin', 'python')
READ_JOB = ('IFS="|" read -r id group camp envn device omp tt extra model methods data base tag '
            'limit regimes token blind bk sched oroot folder n ckpt log pre rset files')
CHANGED = {'--family_model', '--regimes', '--tag', '--save_schedules'}

pytestmark = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(ROOT, 'data', 'FAMILY', s, 'dataset_meta.json'))
            or os.path.isdir(os.path.join(ROOT, 'data', 'FAMILY', s, 'desc')) for s in PS.SETS),
    reason='the production sets are not present')


def env(**kv):
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS', 'EVAL_GROUPS', 'EXACT_SEEDS'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kv.items()})
    return e


# ----------------------------------------------------------------- the plan
def test_jobs_are_the_n30_policy_jobs_with_the_specialist_exchanged():
    js = PS.jobs()
    assert len(js) == 80 and len({j['id'] for j in js}) == 80
    seen = set()
    for j in js:
        name = j['campaign'][:-len('_spec')]
        seed = int(j['model'].rsplit('-s', 1)[1])
        r = j['regimes']
        t = PS.template(name, seed)
        diff = {k for k in set(j) | set(t) if j.get(k) != t.get(k)}
        assert diff <= set(PS.EXCHANGED), diff - set(PS.EXCHANGED)
        assert t['model'] == REG.PREFIX + REG.model('released', None, seed, 'exact')
        assert t['regimes'].split() == list(REG.COMBOS) and t['sched'] != '-'
        assert j['model'] == REG.PREFIX + REG.model('specialist', r, seed, 'exact')
        assert j['ckpt'] == REG.checkpoint('specialist', r, seed, 'exact')
        assert j['log'] == REG.train_log('specialist', r, seed, 'exact')
        assert j['sched'] == '-' and j['tag'] == PS.TAG == '_xmask_spec'
        assert j['folder'] == name + PS.TAG and j['n'] == 30
        assert (j['data'], j['base']) == PLAN.N30[name] == (t['data'], t['base'])
        out = f'{PLAN.RESULTS}/{name}{PS.TAG}'
        assert j['files'] == [f'{out}/Result_greedy+{j["model"]}_{r}.npy',
                              f'{out}/summary_{j["model"]}{PS.TAG}.md']
        seen.add((name, r, seed))
    assert seen == {(s, r, x) for s in PS.SETS for r in REG.COMBOS for x in PS.SEEDS}
    costs = [j['cost'] for j in js]
    assert costs == sorted(costs, reverse=True)               # longest first


@pytest.mark.parametrize('k', [1, 8, 16])
def test_lanes_partition_the_jobs(k):
    ids = [j['id'] for j in PS.jobs()]
    lanes = [[j['id'] for j in PS.lane(f'{i}/{k}')] for i in range(k)]
    flat = [x for ln in lanes for x in ln]
    assert sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
    assert max(map(len, lanes)) - min(map(len, lanes)) <= 1


@pytest.mark.parametrize('bad', ['8/8', '-1/4', '1', 'a/b', '0/0'])
def test_a_bad_lane_is_refused(bad):
    with pytest.raises(PS.Refused):
        PS.parse_lane(bad)


def test_the_result_folder_is_not_swept_by_a_policy_analysis():
    import p120_lower_bound_reference as P120
    names = PLAN.ref_names()
    for s in PS.SETS:
        swept = {REG.res_dir(f) for f in (s,) + tuple(P120.EXTRA_FOLDERS.get(s, ()))}
        assert PS.folder(s) not in swept
        assert os.path.basename(PS.folder(s)) == s + '_xmask_spec'
    for j in PS.jobs():
        for f in j['files']:
            m = PLAN.RX.match(os.path.basename(f)) or PLAN.RXS.match(os.path.basename(f))
            assert m and m.group(1) not in names


# ----------------------------------------------------------------- the runner
def untrained():
    return sorted({j['model'] for j in PS.jobs() if PS.trained(j['model'])})


def options(tokens):
    i = tokens.index('eval_family.py')
    d, key = {}, None
    for t in tokens[i + 1:]:
        if t.startswith('--'):
            key, d[t] = t, ()
        else:
            d[key] += (t,)
    return tokens[:i + 1], d


def template_commands(tmp_path):
    """{job id: (prefix, options)} of build_cmd on the n30 policy job of each job."""
    lines = [f'{j["id"]}\t' + PS.job_line(PS.template_of(j['id'])) for j in PS.jobs()]
    src = tmp_path / 'templates.txt'
    src.write_text('\n'.join(lines) + '\n')
    script = f"""
set -u
cd {ROOT}
PY={PY}
eval "$(sed -n '/^build_cmd() {{/,/^}}/p; /^show() {{.*}}$/p; /^MASK_OPTS=/p' scripts/run_eval_newseeds.sh)"
while IFS=$'\\t' read -r jid line; do
  {READ_JOB} <<< "$line"
  build_cmd "<core>" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \\
    "$limit" "$regimes" "$token" "$blind" "$bk" "$sched" exact "STAGE/$jid/out" "$rset"
  echo "$jid	$(show)"
done < {src}
"""
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env(),
                         check=True).stdout
    return {ln.split('\t', 1)[0]: options(shlex.split(ln.split('\t', 1)[1]))
            for ln in out.splitlines()}


def folder_state(folders):
    out = []
    for d in folders:
        if not os.path.exists(d):
            out.append(None)
            continue
        paths = [d] + sorted(os.path.join(p, n) for p, ds, fs in os.walk(d) for n in ds + fs)
        out.append([(p, os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths])
    return out


def test_runner_commands_are_the_policy_commands_with_four_options_changed(tmp_path):
    if untrained():
        pytest.skip(f'needs the trained specialists; not trained: {", ".join(untrained()[:3])}')
    folders = [os.path.join(ROOT, 'test_results', '.eval_staging', 'prodspec'),
               os.path.join(ROOT, 'train_log', 'production_specialists_eval'),
               os.path.join(ROOT, 'train_log', 'production_specialists_eval_locks')]
    before = folder_state(folders)
    r = subprocess.run(['bash', RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(LIST_ONLY=1, TMPDIR=tmp_path, RESULTS=tmp_path / 'results'))
    assert r.returncode == 0, r.stdout + r.stderr
    got = {}
    for ln in r.stdout.splitlines():
        m = re.match(r'JOB (\S+) \(2 files, 30 instances x 1 combinations x 1 methods\): (.*)$', ln)
        if m:
            got[m.group(1)] = options(shlex.split(m.group(2)))
    want = template_commands(tmp_path)
    assert set(got) == set(want) and len(got) == 80
    for jid, (pre, opt) in got.items():
        j = [x for x in PS.jobs() if x['id'] == jid][0]
        tpre, topt = want[jid]
        assert pre == tpre
        assert opt['--out_root'] == (f'test_results/.eval_staging/prodspec/{jid}/out',)
        diff = {k for k in set(opt) | set(topt) if k != '--out_root' and opt.get(k) != topt.get(k)}
        assert diff == CHANGED, (jid, diff)
        assert opt['--family_model'] == (j['model'],) and opt['--regimes'] == (j['regimes'],)
        assert opt['--tag'] == ('_xmask_spec',) and '--save_schedules' not in opt
        assert topt['--tag'] == ('_xmask',)
        assert opt['--first_mask'] == ('exact',) and opt['--mask_ties'] == ('exact',)
        assert opt['--device'] == ('cpu',) and opt['--torch_threads'] == ('1',)
    pre = [ln for ln in r.stdout.splitlines() if ln.startswith('PREFLIGHT: ')][0]
    assert ('--family_model 10x25+family+xmask-solo-S-s301 --data data/FAMILY/50x25n30 '
            '--base data/PPVC/50x25+ppvc-mixed-n30 --regimes S ') in pre
    assert '--limit 2 ' in pre and '--first_mask exact --mask_ties exact' in pre
    assert PS.pre_stored() in r.stdout
    assert 'production specialists: 80 jobs, 2400 rollouts' in r.stdout
    assert folder_state(folders) == before                        # nothing written


def test_runner_skips_finished_and_refuses_wrong_or_partial(tmp_path):
    results = tmp_path / 'results'
    js = [j for j in PS.jobs() if j['campaign'] == '50x25n30_spec' and j['regimes'] in ('N', 'L', 'S')
          and j['model'].endswith('-s301')]
    fin, short, part = sorted(js, key=lambda j: j['regimes'])     # L, N, S

    def place(f, rows):
        p = results / os.path.relpath(f, PLAN.RESULTS)
        p.parent.mkdir(parents=True, exist_ok=True)
        if f.endswith('.npy'):
            np.save(p, np.zeros((rows, 2)))
        else:
            p.write_text('summary\n')

    for f in fin['files']:
        place(f, 30)
    place(short['files'][0], 29)
    place(short['files'][1], 30)
    place(part['files'][0], 30)
    before = sorted(str(p) for p in results.rglob('*'))
    ids = [fin['id'], short['id'], part['id']]
    r = subprocess.run(['bash', RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(LIST_ONLY=1, TMPDIR=tmp_path, RESULTS=results,
                               ONLY='^(' + '|'.join(re.escape(i) for i in ids) + ')$',
                               STAGE_ROOT=tmp_path / 'stage', LOCKDIR=tmp_path / 'lock',
                               LOGDIR=tmp_path / 'log'))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert f'SKIP {ids[0]} (finished)' in out
    assert re.search(rf'REFUSING {re.escape(ids[1])}: its files exist but an array does not '
                     r'hold 30 rows', out)
    assert re.search(rf'REFUSING {re.escape(ids[2])}: 1 of 2 files exist in place', out)
    assert 'JOB ' not in out
    assert sorted(str(p) for p in results.rglob('*')) == before
    for d in ('stage', 'lock', 'log'):
        assert not (tmp_path / d).exists()


def test_runner_refuses_a_bad_lane_and_a_missing_core(tmp_path):
    r = subprocess.run(['bash', RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(LIST_ONLY=1, LANE='3/3', TMPDIR=tmp_path))
    assert r.returncode == 2 and 'REFUSING: LANE must be' in r.stdout
    r = subprocess.run(['bash', RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(TMPDIR=tmp_path, LOGDIR=tmp_path / 'log'))
    assert r.returncode == 2 and 'CORES=<one core> is required' in r.stdout
    assert not (tmp_path / 'log').exists()


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, PS.pre_stored())),
                    reason='the default level of p103 is not stored')
def test_preflight_comparison_accepts_the_stored_array_and_refuses_a_change(tmp_path, capsys,
                                                                          monkeypatch):
    monkeypatch.chdir(ROOT)
    j, stored = PS.preflight()
    assert j['limit'] == PS.PRE_N == 2 and j['regimes'] == 'S'
    assert j['model'] == REG.PREFIX + REG.model('specialist', 'S', 301, 'exact')
    with np.load(stored) as z:
        ms = np.asarray(z['makespan'], dtype=float)[:2]
    p = tmp_path / j['folder'] / os.path.basename(j['files'][0])
    p.parent.mkdir(parents=True)
    np.save(p, np.stack([ms, np.ones(2)], axis=1))
    assert PS.compare(str(tmp_path)) == 0
    assert 'preflight identical: 2 makespans' in capsys.readouterr().out
    np.save(p, np.stack([ms + np.array([0.0, 1.0]), np.ones(2)], axis=1))
    assert PS.compare(str(tmp_path)) == 1
    assert 'PREFLIGHT MISMATCH' in capsys.readouterr().out
