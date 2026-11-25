"""The two further test sets: their builders, the job plan and the runner of their evaluation.

  * scripts/further_sets_plan.py: every job of a set is the evaluation plan's job of the same
    run on the model set (20x25 for 27x25, 10x25_oodsetup for 10x25_asymsetup) with only the
    id, campaign, data, base, result folder, file names, instance count, cost and pre-flight
    group exchanged; every file lies in arm_registry.res_dir(<set>); a second seed (306 to 310)
    is the job of seed 305 with the run name exchanged in id, model, checkpoint, log and files;
  * scripts/run_further_sets_eval.sh LIST_ONLY=1 on an empty result tree (so the listing does
    not depend on how far the stored evaluation of the set has come): each printed command
    equals build_cmd of scripts/run_eval_newseeds.sh on the plan's job of the model set, with
    the data and base folders of the set in place of those of the model set; nothing is
    written; skipped while a seed-301 run of the set is not trained;
  * the runner skips a finished job, refuses a job whose arrays have another length and a job
    whose files partly exist (fake files in a temporary result tree);
  * the runner refuses an unknown set and a missing CORE outside LIST_ONLY;
  * scripts/p57_build_scale_data.py --sizes 27 --n 50, scripts/p1_build_family_data.py on 27x25
    and scripts/p29_ood.py --build --sets 10x25_asymsetup verify the built sets and write
    nothing (the folder digests are unchanged).

Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      tests/test_further_sets_eval.py
"""
import hashlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None, RESULTS=None,
                                        SEEDS=None, PACT_FIRST_MASK=None,
                                        PACT_MASK_TOL=None, PACT_ENV='exact',
                                        PACT_SEEDS='301-305')

import arm_registry as REG  # noqa: E402
import eval_newseeds_plan as PLAN  # noqa: E402
import further_sets_plan as F  # noqa: E402

PY = sys.executable
RUNNER = os.path.join(ROOT, 'scripts', 'run_further_sets_eval.sh')
EXCHANGED = {'id', 'campaign', 'data', 'base', 'folder', 'files', 'n', 'cost', 'pre'}
N_OF = {'27x25': 50, '10x25_asymsetup': 100}
READ_JOB = ('IFS="|" read -r id group camp envn device omp tt extra model methods data base tag '
            'limit regimes token blind bk sched oroot folder n ckpt log pre rset files')

pytestmark = pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(ROOT, s['data'], 'dataset_meta.json'))
            for s in F.SETS.values()),
    reason='the further sets are not built (scripts/p57_build_scale_data.py, p1, p29)')


def model_set_jobs(name, seed):
    """The plan's jobs on the model set as its 'jobs' subcommand prints them (with pre)."""
    return {m: dict(j, pre=PLAN.preflight_of(j)) for m, j in F.model_jobs(name, seed).items()}


def env(**kv):
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kv.items()})
    return e


def digest(folder):
    h = hashlib.sha256()
    for d, _, fs in sorted(os.walk(os.path.join(ROOT, folder))):
        for f in sorted(fs):
            p = os.path.join(d, f)
            h.update(os.path.relpath(p, ROOT).encode())
            h.update(open(p, 'rb').read())
    return h.hexdigest()


# ----------------------------------------------------------------- the job plan
@pytest.mark.parametrize('name', list(F.SETS))
def test_jobs_are_the_model_set_jobs_with_the_set_exchanged(name):
    s = F.SETS[name]
    folder = os.path.basename(s['data']) + PLAN.XS
    res = os.path.normpath(REG.res_dir(name))
    assert res == os.path.normpath(os.path.join(PLAN.RESULTS, folder))
    n_jobs = 0
    for seed in F.FIRST:
        model = model_set_jobs(name, seed)
        mine = F.jobs(name, [seed])
        assert sorted(j['model'] for j in mine) == sorted(model)
        assert set(model) == {'rules', REG.PREFIX + REG.model('released', None, seed)} | {
            REG.PREFIX + REG.model('specialist', r, seed) for r in REG.COMBOS}
        for j in mine:
            m = model[j['model']]
            changed = {k for k in m if m[k] != j[k]}
            assert changed <= EXCHANGED, (j['id'], changed - EXCHANGED)
            assert set(j) == set(m)
            assert j['data'] == s['data'] and j['base'] == s['base']
            assert j['campaign'] == name and j['folder'] == folder and j['n'] == N_OF[name]
            assert j['id'] == f'exact:{name}:' + m['id'][len(f'exact:{s["model"]}:'):]
            old = f'{m["out_root"]}/{m["folder"]}/'
            assert j['files'] == [f'{PLAN.RESULTS}/{folder}/' + f[len(old):] for f in m['files']]
            assert all(os.path.normpath(os.path.dirname(f)) == res for f in j['files'])
            assert j['pre'] == f'further_{name}' and j['sched'] == '-' and j['env'] == 'exact'
            n_jobs += 1
    assert n_jobs == 5 * 10            # policy, eight specialists, the rules; per seed


@pytest.mark.parametrize('name', list(F.SETS))
def test_second_seed_is_the_seed_305_job_renamed(name):
    a = {j['id']: j for j in F.jobs(name, [305])}
    b = {j['id']: j for j in F.jobs(name, [306])}
    assert len(a) == len(b) == 10
    for k, j in b.items():
        i = a[k.replace('s306', 's305')]
        if j['model'] == 'rules':
            assert i == j
            continue
        for key in i:
            if key == 'files':
                assert j[key] == [f.replace('s305', 's306') for f in i[key]]
            elif key in ('id', 'model', 'ckpt', 'log'):
                assert j[key] == i[key].replace('s305', 's306') != i[key]
            else:
                assert j[key] == i[key], key


def test_seeds_setting_refuses_other_seeds(monkeypatch):
    monkeypatch.setenv('SEEDS', '301 311')
    with pytest.raises(F.Refused):
        F.seeds()
    monkeypatch.setenv('SEEDS', '302 301')
    with pytest.raises(F.Refused):
        F.seeds()


# ----------------------------------------------------------------- the runner, LIST_ONLY
def expected_commands(name, tmp_path):
    """{new id: command} from build_cmd of run_eval_newseeds.sh on the plan's model-set jobs."""
    s = F.SETS[name]
    stage = f'test_results/.eval_staging/further_{name}'
    lines = []
    for seed in (301,):
        model = model_set_jobs(name, seed)
        for j in F.jobs(name, [seed]):
            lines.append(j['id'] + '\t' + F.job_line(model[j['model']]))
    src = tmp_path / f'{name}_jobs.txt'
    src.write_text('\n'.join(lines) + '\n')
    script = f"""
set -u
cd {ROOT}
PY=$HOME/miniconda3/envs/fjsp/bin/python
eval "$(sed -n '/^build_cmd() {{/,/^}}/p; /^show() {{.*}}$/p; /^MASK_OPTS=/p' scripts/run_eval_newseeds.sh)"
while IFS=$'\\t' read -r newid line; do
  {READ_JOB} <<< "$line"
  build_cmd "<core>" "$device" "$omp" "$tt" "$extra" "$model" "$methods" "$data" "$base" "$tag" \\
    "$limit" "$regimes" "$token" "$blind" "$bk" - exact "{stage}/$newid/out" "$rset"
  echo "$newid	$(show)"
done < {src}
"""
    out = subprocess.run(['bash', '-c', script], capture_output=True, text=True, env=env(),
                         check=True).stdout
    md, mb = PLAN.FACTORY[s['model']]
    cmds = {}
    for ln in out.splitlines():
        newid, cmd = ln.split('\t', 1)
        assert f'--data {md} ' in cmd and f'--base {mb} ' in cmd
        cmds[newid] = cmd.replace(f'--data {md} ', f'--data {s["data"]} ').replace(
            f'--base {mb} ', f'--base {s["base"]} ')
    return cmds


def folder_state(folders):
    """Per folder: None when it is absent, else every path under it with its size and time of
    last change."""
    out = []
    for d in folders:
        if not os.path.exists(d):
            out.append(None)
            continue
        paths = [d] + sorted(os.path.join(p, n) for p, ds, fs in os.walk(d) for n in ds + fs)
        out.append([(p, os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths])
    return out


def untrained(name):
    """The seed-301 runs of the set's jobs that further_sets_plan.py does not count as trained
    (checkpoint and completion line of the training log); the runner lists no job for them."""
    runs = sorted({j['model'] for j in F.jobs(name, [301])} - {'rules'})
    return [m for m in runs if F.trained(m)]


@pytest.mark.parametrize('name', list(F.SETS))
def test_runner_commands_equal_build_cmd_on_the_model_set_jobs(name, tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    missing = untrained(name)
    if missing:
        pytest.skip(f'needs the trained seed-301 runs of the set; not trained: {", ".join(missing)}')
    want = expected_commands(name, tmp_path)
    # the runner's staging and log folders: a real evaluation of the set leaves them behind
    # (empty), so the listing must leave them as they were, present or not
    folders = [os.path.join(ROOT, 'test_results', '.eval_staging', f'further_{name}'),
               os.path.join(ROOT, 'train_log', 'further_sets_eval', name)]
    before = folder_state(folders)
    # an empty result tree (RESULTS): the listing holds every job of seed 301 whether or not the
    # stored evaluation of the set has started or finished (a finished job would be listed as
    # SKIP, a partly written one refused)
    r = subprocess.run([RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(SET=name, LIST_ONLY=1, TMPDIR=tmp_path, ONLY='s301|rules$',
                               RESULTS=tmp_path / 'results'))
    assert r.returncode == 0, r.stdout + r.stderr
    got = {}
    for ln in r.stdout.splitlines():
        m = re.match(r'JOB (\S+) \(\d+ files, (\d+) instances x \d+ combinations x \d+ '
                     r'methods\): (.*)$', ln)
        if m:
            got[m.group(1)] = m.group(3)
            assert int(m.group(2)) == N_OF[name]
    assert got == want
    assert len(got) == 10
    pre = [ln for ln in r.stdout.splitlines() if ln.startswith('PREFLIGHT: ')][0]
    md, mb = PLAN.FACTORY[F.SETS[name]['model']]
    assert f'--data {md} --base {mb} --regimes S LSW ' in pre and '--limit 3 ' in pre
    assert '--first_mask exact --mask_ties exact' in pre
    assert folder_state(folders) == before                       # nothing written


def test_runner_skips_finished_and_refuses_wrong_or_partial(tmp_path):
    name, n = '10x25_asymsetup', N_OF['10x25_asymsetup']
    results = tmp_path / 'results'
    js = {j['model']: j for j in F.jobs(name, [301])}
    pol = REG.PREFIX + REG.model('released', None, 301)
    spec = REG.PREFIX + REG.model('specialist', 'N', 301)

    def place(f, rows):
        p = results / os.path.relpath(f, PLAN.RESULTS)
        p.parent.mkdir(parents=True, exist_ok=True)
        if f.endswith('.npy'):
            np.save(p, np.zeros((rows, 2)))
        else:
            p.write_text('summary\n')

    for f in js['rules']['files']:
        place(f, n)                                     # finished
    for k, f in enumerate(js[pol]['files']):
        place(f, n - 1 if k == 0 else n)                # one array one row short
    place(js[spec]['files'][0], n)                      # one of its files only
    before = sorted(str(p) for p in results.rglob('*'))
    ids = [js[m]['id'] for m in ('rules', pol, spec)]
    r = subprocess.run([RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(SET=name, LIST_ONLY=1, TMPDIR=tmp_path, RESULTS=results,
                               ONLY='^(' + '|'.join(re.escape(i) for i in ids) + ')$',
                               STAGE_ROOT=tmp_path / 'stage', LOCKDIR=tmp_path / 'lock',
                               LOGDIR=tmp_path / 'log'))
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert f'SKIP {ids[0]} (finished)' in out
    assert re.search(rf'REFUSING {re.escape(ids[1])}: its files exist but an array does not '
                     rf'hold {n} rows', out)
    assert re.search(rf'REFUSING {re.escape(ids[2])}: 1 of {len(js[spec]["files"])} files exist '
                     r'in place', out)
    assert 'JOB ' not in out
    assert f'results in {results}/{name}{PLAN.XS}' in out
    assert sorted(str(p) for p in results.rglob('*')) == before
    for d in ('stage', 'lock', 'log'):
        assert not (tmp_path / d).exists()


def test_runner_refuses_unknown_set_and_missing_core(tmp_path):
    r = subprocess.run([RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(SET='20x25', LIST_ONLY=1, TMPDIR=tmp_path))
    assert r.returncode == 2 and 'REFUSING: SET=' in r.stdout
    r = subprocess.run([RUNNER], capture_output=True, text=True, cwd=ROOT,
                       env=env(SET='27x25', TMPDIR=tmp_path))
    assert r.returncode == 2 and 'CORE=<list> is required' in r.stdout


# ----------------------------------------------------------------- the builders
BUILDS = [
    (['scripts/p57_build_scale_data.py', '--sizes', '27', '--n', '50'],
     'data/PPVC/27x25+ppvc-mixed',
     r'data/PPVC/27x25\+ppvc-mixed: already built and identical \(50 instances\)'),
    (['scripts/p1_build_family_data.py', '--base', 'data/PPVC/27x25+ppvc-mixed', '--out',
      'data/FAMILY/27x25'],
     'data/FAMILY/27x25',
     r'data/FAMILY/27x25: already built and identical \(401 files compared byte for byte\)'),
    (['scripts/p29_ood.py', '--build', '--sets', '10x25_asymsetup'],
     'data/FAMILY/10x25_asymsetup',
     r'data/FAMILY/10x25_asymsetup: already built; 800 descriptors equal the generator in every '
     r'compared field, 800 of them byte-identical; nothing written'),
]


@pytest.mark.parametrize('cmd,folder,message', BUILDS, ids=['p57', 'p1', 'p29'])
def test_builders_verify_an_existing_set_and_write_nothing(cmd, folder, message):
    before = digest(folder)
    r = subprocess.run([PY, '-B'] + cmd, capture_output=True, text=True, cwd=ROOT, env=env())
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.search(message, r.stdout), r.stdout
    assert digest(folder) == before
