"""Counts, file facts and the check of the design constants typed by hand (scripts/p111_model_facts.py).

  * parameter counts from synthetic training logs: equal logs give the count; logs on disk that
    disagree are refused; a log without the line or a missing log is refused where the count is
    required and named and left out otherwise; one log with two counts is refused;
  * the parsing of a small .fjs file (header and every compatible processing time);
  * the constants check on synthetic macro files: equal values pass after removing thousands
    separators, a different value and a name defined nowhere are differences;
  * training episodes per combination and nested fields of synthetic configurations;
  * manuscript/macros_design.tex defines exactly the checked constants, in their order, each with
    a comment naming its source;
  * the macro file leaves out a count whose runs are not on disk and says so; the kept names
    may be defined in the output of p28 only;
  * on the stored runs (when present): the script writes its files under a temporary output
    root, and the check exits 0.

Run from the repository root:  python -m pytest tests/test_model_facts.py -q
CPU only; every file is written under a temporary folder.
"""
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import p111_model_facts as P  # noqa: E402

PY = sys.executable
LINE = '[model] params total={}  film=0 (0.00%)\n'

needs_runs = pytest.mark.skipif(
    not os.path.exists(os.path.join(ROOT, 'train_log/FAMILY_xmask-v1-s301.log')),
    reason='needs the training logs of the exact environment')


def put(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)
    return str(path)


def test_count_from_equal_logs(tmp_path):
    logs = [put(tmp_path / f'a{i}.log', 'start\n' + LINE.format(32786) + 'done in 1 h\n')
            for i in range(3)]
    got = P.count(logs, 'policy', True)
    assert got['value'] == 32786 and len(got['runs']) == 3 and got['missing'] == []


def test_count_refuses_logs_that_disagree(tmp_path):
    logs = [put(tmp_path / 'a.log', LINE.format(32786)), put(tmp_path / 'b.log', LINE.format(32914))]
    with pytest.raises(P.Refused, match='disagree'):
        P.count(logs, 'policy', False)


def test_count_missing_logs_required_and_optional(tmp_path):
    present = put(tmp_path / 'a.log', LINE.format(71838))
    empty = put(tmp_path / 'b.log', 'no banner here\n')
    absent = str(tmp_path / 'c.log')
    with pytest.raises(P.Refused, match='no training log'):
        P.count([present, absent], 'policy', True)
    with pytest.raises(P.Refused, match='has no line'):
        P.count([present, empty], 'policy', True)
    got = P.count([present, empty, absent], 'descriptor', False)
    assert got['value'] == 71838 and len(got['missing']) == 2
    none = P.count([absent], 'descriptor', False)
    assert none['value'] is None and none['missing']


def test_one_log_with_two_counts_is_refused(tmp_path):
    p = put(tmp_path / 'a.log', LINE.format(1) + LINE.format(2))
    with pytest.raises(P.Refused, match='more than one'):
        P.log_params(p)


def test_fjs_parsing(tmp_path):
    # two jobs on three machines: job 0 has two operations (machines 1,2 with 4,6 h; machine 3
    # with 2 h), job 1 one operation (machine 1 with 5 h)
    p = put(tmp_path / 'x.fjs', '2\t3\t1.3\n2 2 1 4 2 6 1 3 2\n1 1 1 5\n')
    nj, nm, vals = P.fjs_times(p)
    assert (nj, nm) == (2, 3) and vals == [4, 6, 2, 5]


def test_number_parsing():
    assert P.number('2,000') == 2000 and P.number('2{,}000') == 2000
    assert P.number('6\\,000') == 6000 and P.number('0.9') == 0.9 and P.number('n/a') is None


def test_constants_check_on_synthetic_files(tmp_path, monkeypatch):
    put(tmp_path / 'manuscript' / 'a.tex', '\\newcommand{\\JointUpdates}{2,000}  % comment\n'
                                         '\\newcommand{\\SpecUpdates}{1{,}000}\n')
    put(tmp_path / 'manuscript' / 'b.tex', '\\newcommand{\\ExpEnvs}{24}\n'
                                         '% \\newcommand{\\CpuShareMin}{0.9}\n')
    monkeypatch.setattr(P, 'ROOT', str(tmp_path))
    monkeypatch.setattr(P, 'CONSTANT_FILES', ('manuscript/a.tex', 'manuscript/b.tex'))
    monkeypatch.setattr(P, 'CONSTANTS', (('JointUpdates', 'updates of the policy', 'p'),
                                         ('SpecUpdates', 'updates of a specialist', 's'),
                                         ('ExpEnvs', 'episodes per update', 'e'),
                                         ('CpuShareMin', 'minimum CPU share', 'c')))
    force = {'p': (2000, ['cfg']), 's': (1000, ['cfg']), 'e': (20, ['cfg']), 'c': (0.9, ['tv'])}
    rows, bad = P.check_constants(force)
    assert [r['macro'] for r in rows] == ['JointUpdates', 'SpecUpdates', 'ExpEnvs', 'CpuShareMin']
    assert rows[0]['definitions'][0]['equal'] and rows[1]['definitions'][0]['equal']
    assert len(bad) == 2
    assert "\\ExpEnvs (episodes per update) is '24' in manuscript/b.tex" in bad[0]
    assert '\\CpuShareMin (minimum CPU share) is defined in none of' in bad[1]   # commented out


def test_episodes_per_combination_and_nested_fields(tmp_path):
    assert P.episodes_per_combination(dict(max_updates=2000, num_envs=20,
                                           family_regimes='N,L,S,W,LS,LW')) == 6667
    assert P.episodes_per_combination(dict(max_updates=1000, num_envs=20,
                                           family_regimes='LS')) == 20000
    with pytest.raises(P.Refused, match='lacks'):
        P.episodes_per_combination(dict(max_updates=1000, num_envs=20))
    a = put(tmp_path / 'a.json', json.dumps(dict(n_params=204250, env_paras=dict(batch_size=20))))
    b = put(tmp_path / 'b.json', json.dumps(dict(n_params=204250, env_paras=dict(batch_size=10))))
    assert P.config_value([a], 'env_paras.batch_size', 'batch')[0] == 20
    assert P.config_value([a, b], 'n_params', 'params')[0] == 204250
    with pytest.raises(P.Refused, match='disagree'):
        P.config_value([a, b], 'env_paras.batch_size', 'batch')
    c = put(tmp_path / 'c.json', json.dumps(dict(max_updates=6000, num_envs=20,
                                                 family_regimes='N,L,S,W,LS,LW')))
    d = put(tmp_path / 'd.json', json.dumps(dict(max_updates=2000, num_envs=20,
                                                 family_regimes='N,L,S,W,LS,LW')))
    assert P.episodes_value([c], 'eq')[0] == 20000
    with pytest.raises(P.Refused, match='disagree'):
        P.episodes_value([c, d], 'mixed')


def test_constant_file_defines_exactly_the_checked_constants():
    assert P.CONSTANT_FILES == ('manuscript/macros_design.tex',)
    lines = open(os.path.join(ROOT, P.CONSTANT_FILES[0])).read().splitlines()
    defs = [ln for ln in lines if ln.startswith('\\newcommand')]
    names = [ln.split('{\\', 1)[1].split('}', 1)[0] for ln in defs]
    assert names == [c[0] for c in P.CONSTANTS]
    assert len({c[2] for c in P.CONSTANTS}) == len(P.CONSTANTS)     # one source per constant
    for ln in defs:
        assert '  % ' in ln and ':' in ln.split('  % ', 1)[1], ln    # meaning: source
    assert all(ln.startswith(('%', '\\newcommand')) or not ln.strip() for ln in lines)


def fake_result(descriptor=None, hgnn=None):
    def cnt(v):
        return dict(value=v, runs={'x.log': v} if v else {},
                    missing=[] if v else ['no training log train_log/FAMILY_y.log'])
    return dict(setting='environment exact, synthetic',
                facts=dict(policy=cnt(32786), descriptor=cnt(descriptor), hgnn=cnt(hgnn),
                           checkpoint=dict(seed=304, path='ckpt.pth', bytes=155957,
                                           kib=155957 / 1024.0),
                           op_duration=dict(value=4.0, pairs=10, instances=2, base_dir='b')))


def test_macro_file_leaves_out_counts_not_on_disk():
    lines, left = P.macro_lines(fake_result())
    text = '\n'.join(lines)
    assert '\\newcommand{\\BlindParams}{32786}' in text
    assert '\\newcommand{\\PolicyCheckpointKB}{152}' in text
    assert '\\newcommand{\\OpDurMedian}{4}' in text
    assert 'DescriptorParams}{' not in text and 'HgnnNetworkParams}{' not in text
    assert len(left) == 2 and text.count('% Left out, runs not on disk:') == 2
    lines, left = P.macro_lines(fake_result(71838, 204250))
    text = '\n'.join(lines)
    assert '\\newcommand{\\DescriptorParams}{71838}' in text and not left
    assert '\\newcommand{\\HgnnNetworkParams}{204250}' in text


def test_names_kept_only_beside_the_output_of_p28(monkeypatch):
    monkeypatch.setattr(P, 'defined_macros', lambda: {
        'BlindParams': ['manuscript/macros_exposure.tex'],
        'OpDurMedian': ['manuscript/macros_exposure.tex'],
        'ParamsTotal': ['manuscript/macros.tex']})
    P.check_names(['BlindParams', 'OpDurMedian', 'DescriptorParams'])
    with pytest.raises(P.Refused, match='also defined in manuscript/macros.tex'):
        P.check_names(['ParamsTotal'])
    monkeypatch.setattr(P, 'defined_macros', lambda: {
        'BlindParams': ['manuscript/macros_exposure.tex', 'manuscript/macros_blind.tex']})
    with pytest.raises(P.Refused, match='macros_blind'):
        P.check_names(['BlindParams'])


def test_new_names_are_free_under_manuscript():
    have = P.defined_macros()
    for x in ('DescriptorParams', 'HgnnNetworkParams', 'PolicyCheckpointKB'):
        assert x not in have, (x, have.get(x))
    for x, f in P.KEPT.items():
        assert set(have.get(x, [])) <= {f}


@needs_runs
def test_run_and_check_on_the_stored_runs(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=str(tmp_path / 'out'))
    r = subprocess.run([PY, '-B', 'scripts/p111_model_facts.py', '--check'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0 and 'inputs present' in r.stdout, r.stdout + r.stderr
    r = subprocess.run([PY, '-B', 'scripts/p111_model_facts.py'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    text = open(tmp_path / 'out' / P.OUT_FILES['macros']).read()
    assert '\\newcommand{\\BlindParams}{32786}' in text
    rec = json.load(open(tmp_path / 'out' / P.OUT_FILES['json']))
    assert rec['differences'] == [] and len(rec['constants']) == len(P.CONSTANTS)
    c = rec['facts']['checkpoint']
    assert c['seed'] in (301, 302, 303, 304, 305) and c['path'].endswith(f'-s{c["seed"]}.pth')
    assert '\\newcommand{\\PolicyCheckpointKB}{%.0f}' % c['kib'] in text


def test_other_environment_is_refused(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PACT_OUT=str(tmp_path / 'out'))
    r = subprocess.run([PY, '-B', 'scripts/p111_model_facts.py'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode != 0 and 'PACT_ENV=exact' in (r.stdout + r.stderr)
