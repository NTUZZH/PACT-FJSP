"""Selection of the configuration of the memetic algorithm (scripts/p106_memetic_selection.py):
the score arithmetic and the counts of best cells on synthetic records with a known answer, a
tie for the shortest makespan, the bold markers, the refusals (nothing written, non-zero exit),
that the folder H is never opened, that no output holds an absolute path and nothing is written
outside --out, the rebuild from the JSON alone, the macro names, and that the supplement uses
only macros the script writes. Synthetic record folders written by the tests; no search runs."""
import builtins
import json
import os
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import p106_memetic_selection as M  # noqa: E402

SUPPLEMENT = os.path.join(REPO, 'manuscript', 'supp_new', 'sn05_baselines.tex')
NUMS = [n for _, n, _ in M.CANDIDATES]
FOLDER = {n: f for f, n, _ in M.CANDIDATES}
N_OPS = {'50x25n30': 320, '80x25n30': 360}          # above 300, as at the real sizes
COMMON = {'decoder': 'insert', 'pop_size': 20, 'tournament': 2, 'p_cross': 0.9,
          'p_mut_os': 0.2, 'p_mut_ma': 0.2, 'mut_ma_ops': 2, 'init_split': [0.6, 0.3, 0.1],
          'ts_n7': True, 'ts_cap_n7': 5, 'tie_break': 'ncrit', 'tenure_min_base': 10,
          'stall_gens': 15, 'restart_keep': 2, 'its_kick': 4, 'its_accept': 'not_worse'}
DIFF = {1: dict(n_offspring=20, ts_scale=5.0, init_frac=None, its_share=0.0, ts_cap=10),
        2: dict(n_offspring=20, ts_scale=5.0, init_frac=None, its_share=0.0, ts_cap=50),
        3: dict(n_offspring=5, ts_scale=0.5, init_frac=0.1, its_share=0.9, ts_cap=10)}
OUT_FILES = [p for _, p in M.OUTPUTS]


def config(num, n_ops):
    cfg = dict(COMMON, **DIFF[num])
    cfg.update(M.derived_values(cfg, n_ops))
    return cfg


def record(num, cell, makespan):
    size, combo, i = cell
    n = N_OPS[size]
    folder = f'or_solution/FAMILY/warmstart_cache_{size[:5]}'
    rules = {'FIFO': 1200.0, 'MOR': 1300.0, 'SPT': 1100.0, 'MWKR': 1250.0}
    return dict(instance=f'instance_{i:03d}', regime=combo, size=size, seed=1, budget_s=300.0,
                budget_clock='cpu', max_decodes=None, seed_schedules=True, warm_cache=folder,
                seed_from=[], seeds=[dict(folder=folder, rule='SPT', makespan=1100,
                                          rule_makespans=rules, source='warm_cache')],
                config=config(num, n), makespan=int(makespan), objective=float(makespan),
                cpu_s=300.001, wall_s=300.2, cpu_share=0.9994,
                n_generations=0 if num == 1 else 3, validated=True, violations=[],
                schedule=dict(op_mch=[0] * n, op_st=[0] * n, op_ct=[1] * n),
                code_sha256_16='0123456789abcdef')


def write_world(tmp_path, value, skip=(), edit=None):
    """Records of the three candidates on the 24 cells; value(num, cell) is the makespan and
    edit(num, cell, record) may change a record before it is written."""
    root = tmp_path / 'records'
    for num in NUMS:
        for cell in M.CELLS:
            if (num, cell) in skip:
                continue
            rec = record(num, cell, value(num, cell))
            if edit:
                edit(num, cell, rec)
            p = root / M.record_rel(FOLDER[num], cell)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec))
    return root


SPECIAL = ('80x25n30', 'LSW', 2)


def known_value(num, cell):
    """Configuration 3 at 100, 1 at 102, 2 at 104; on SPECIAL 1 at 100 and 3 at 101."""
    if cell == SPECIAL:
        return {1: 100, 2: 104, 3: 101}[num]
    return {1: 102, 2: 104, 3: 100}[num]


def run(tmp_path, root, out=None, extra=()):
    out = out or tmp_path / 'out'
    rc = M.main(['--records', str(root), '--out', str(out), *extra])
    return rc, out


def macros(out):
    text = (out / 'manuscript' / M.MACRO_FILE).read_text()
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  %', text))


def result(out):
    return json.loads((out / 'results' / 'memetic_selection.json').read_text())


def outputs(out):
    return sorted(os.path.relpath(os.path.join(d, f), out)
                  for d, _, fs in os.walk(out) for f in fs)


# ---------------------------------------------------------------------------

def test_scores_and_best_counts_on_a_known_answer(tmp_path):
    rc, out = run(tmp_path, write_world(tmp_path, known_value))
    assert rc == 0
    mac, R = macros(out), result(out)
    # 23 cells at 2, 4 and 0 percent; on SPECIAL the best is 100, so 0, 4 and 1 percent
    assert (mac['MselOneScore'], mac['MselTwoScore'], mac['MselThreeScore']) == \
        ('1.92', '4.00', '0.04')
    assert (mac['MselOneBest'], mac['MselTwoBest'], mac['MselThreeBest']) == ('1', '0', '23')
    cand = {x['number']: x for x in R['candidates']}
    assert cand[1]['score']['all'] == pytest.approx(46 / 24)
    assert cand[3]['score']['all'] == pytest.approx(1 / 24)
    assert cand[1]['score']['80x25n30'] == pytest.approx(22 / 12)
    assert cand[1]['score']['50x25n30'] == pytest.approx(2.0)
    assert cand[1]['score']['LSW'] == pytest.approx(10 / 6)
    assert cand[3]['score']['LSW'] == pytest.approx(1 / 6)
    assert cand[2]['score']['N'] == pytest.approx(4.0)
    assert mac['MselThreeScoreBold'] == '\\bfseries' and mac['MselOneScoreBold'] == ''
    assert mac['MselThreeBestBold'] == '\\bfseries' and mac['MselOneBestBold'] == ''
    assert mac['MselSelected'] == '3' and mac['MselTieRuns'] == '0' and R['ties'] == []
    assert (mac['MselRuns'], mac['MselInstances'], mac['MselBudget'], mac['MselSeed']) == \
        ('24', '3', '300', '1')
    assert mac['MselOneNoGenerationRuns'] == '24' and mac['MselThreeNoGenerationRuns'] == '0'
    assert len(R['runs']) == 72 and all(len(r['sha256']) == 64 for r in R['runs'])


def test_differing_values_come_from_the_records(tmp_path):
    rc, out = run(tmp_path, write_world(tmp_path, known_value))
    assert rc == 0
    mac, R = macros(out), result(out)
    assert sorted(R['differing_keys']) == sorted(k for k, _, _ in M.PRINTED)
    assert (mac['MselOneChildren'], mac['MselThreeChildren']) == ('20', '5')
    assert (mac['MselOneInitShare'], mac['MselThreeInitShare']) == ('no limit', '10\\%')
    assert (mac['MselOneMachineMoves'], mac['MselTwoMachineMoves']) == ('10', '50')
    assert (mac['MselOneTabuScale'], mac['MselThreeTabuScale']) == ('5', '0.5')
    assert (mac['MselTwoItsShare'], mac['MselThreeItsShare']) == ('0\\%', '90\\%')
    assert R['common_config']['its_kick'] == 4 and 'its_kick' not in R['differing_keys']
    assert R['candidates'][2]['derived']['80x25n30'] == dict(
        M.derived_values(dict(ts_scale=0.5), 360), operations=360)


def test_a_tie_for_the_shortest_makespan_counts_for_both_and_is_reported(tmp_path):
    tie = ('50x25n30', 'N', 0)

    def value(num, cell):
        return {1: 100, 2: 104, 3: 100}[num] if cell == tie else known_value(num, cell)

    rc, out = run(tmp_path, write_world(tmp_path, value))
    assert rc == 0
    mac, R = macros(out), result(out)
    assert (mac['MselOneBest'], mac['MselTwoBest'], mac['MselThreeBest']) == ('2', '0', '23')
    assert mac['MselTieRuns'] == '1'
    assert R['ties'] == [dict(size='50x25n30', combination='N', instance=0, candidates=[1, 3])]
    assert '(tie)' in (out / 'results' / 'memetic_selection.md').read_text()


def test_bold_marks_every_one_of_a_tie_for_the_largest_count(tmp_path):
    def value(num, cell):
        # configuration 1 shortest at 80 modules, 3 at 50 modules, with different excess
        if num == 2:
            return 110
        fifty = cell[0] == '50x25n30'
        return {1: 101 if fifty else 100, 3: 100 if fifty else 102}[num]

    rc, out = run(tmp_path, write_world(tmp_path, value))
    assert rc == 0
    mac = macros(out)
    assert (mac['MselOneBest'], mac['MselThreeBest']) == ('12', '12')
    assert mac['MselOneBestBold'] == mac['MselThreeBestBold'] == '\\bfseries'
    assert mac['MselTwoBestBold'] == ''
    assert (mac['MselOneScore'], mac['MselThreeScore']) == ('0.50', '1.00')
    assert mac['MselOneScoreBold'] == '\\bfseries' and mac['MselThreeScoreBold'] == ''


def _bad_seed(num, cell, rec):
    if num == 2 and cell == SPECIAL:
        rec['seed'] = 2


def _mixed(num, cell, rec):
    if num == 2 and cell == ('50x25n30', 'S', 1):
        rec['config']['ts_cap'] = 30


def _invalid(num, cell, rec):
    if num == 3 and cell == ('80x25n30', 'N', 0):
        rec['validated'] = False


def _violation(num, cell, rec):
    if num == 1 and cell == ('80x25n30', 'SW', 1):
        rec['violations'] = ['C6: overlap on machine 3']


def _short_budget(num, cell, rec):
    if num == 3 and cell == SPECIAL:
        rec['cpu_s'] = 299.9


def _two_starts(num, cell, rec):
    if num == 1 and cell == SPECIAL:
        rec['seed_from'] = ['or_solution/FAMILY/pactwarm_cache_80x25']
        rec['seeds'].append(dict(folder=rec['seed_from'][0], makespan=1000, source='seed_from'))


def _derived(num, cell, rec):
    if num == 3 and cell == SPECIAL:
        rec['config']['ts_no_improve'] = 100


def _not_best_rule(num, cell, rec):
    if num == 2 and cell == SPECIAL:
        rec['seeds'][0]['makespan'] = 1200


REFUSALS = {
    'missing cell': (dict(skip={(2, ('80x25n30', 'N', 1))}), 'missing record'),
    'wrong seed': (dict(edit=_bad_seed), 'seed 2'),
    'mixed configuration': (dict(edit=_mixed), 'configuration differs'),
    'invalid record': (dict(edit=_invalid), 'validated is False'),
    'violation': (dict(edit=_violation), 'violations'),
    'budget not reached': (dict(edit=_short_budget), 'does not reach the budget'),
    'second starting schedule': (dict(edit=_two_starts), 'starting schedules'),
    'derived value off the rule': (dict(edit=_derived), 'do not follow the rule'),
    'start not the best rule': (dict(edit=_not_best_rule), 'best dispatching rule'),
}


@pytest.mark.parametrize('case', sorted(REFUSALS))
def test_a_refused_selection_writes_nothing_and_exits_nonzero(tmp_path, capsys, case):
    kw, words = REFUSALS[case]
    root = write_world(tmp_path, known_value, **kw)
    rc, out = run(tmp_path, root)
    err = capsys.readouterr().err
    assert rc != 0 and 'REFUSING' in err and words in err, err
    assert not out.exists() or outputs(out) == []


def test_a_record_outside_the_design_and_a_tie_for_the_lowest_score_are_refused(tmp_path,
                                                                                 capsys):
    root = write_world(tmp_path, known_value)
    extra = root / 'A' / '50x25n30' / 'N' / 'instance_003_seed1.json'
    extra.write_text('{}')
    rc, out = run(tmp_path, root)
    assert rc != 0 and 'outside the design' in capsys.readouterr().err
    extra.unlink()

    def even(num, cell):
        if num == 2:
            return 110
        fifty = cell[0] == '50x25n30'
        return {1: 101 if fifty else 100, 3: 100 if fifty else 101}[num]

    root2 = write_world(tmp_path / 'b', even)
    rc, out = run(tmp_path, root2)
    assert rc != 0 and 'share the lowest' in capsys.readouterr().err
    assert not out.exists() or outputs(out) == []


def test_a_refusal_removes_earlier_outputs(tmp_path):
    root = write_world(tmp_path, known_value)
    rc, out = run(tmp_path, root)
    assert rc == 0 and outputs(out) == sorted(OUT_FILES)
    os.remove(root / M.record_rel('C', ('50x25n30', 'SW', 2)))
    rc, out = run(tmp_path, root)
    assert rc != 0 and outputs(out) == []


def test_the_folder_H_is_never_opened(tmp_path, monkeypatch):
    root = write_world(tmp_path, known_value)
    h = root / 'H' / '50x25n30' / 'N'
    h.mkdir(parents=True)
    (h / 'instance_000_seed1.json').write_text('not a record')
    opened, walked = [], []
    real_open, real_walk = builtins.open, os.walk

    def tracked_open(p, *a, **k):
        opened.append(str(p))
        return real_open(p, *a, **k)

    def tracked_walk(p, *a, **k):
        walked.append(str(p))
        return real_walk(p, *a, **k)

    monkeypatch.setattr(M, 'open', tracked_open, raising=False)
    monkeypatch.setattr(M.os, 'walk', tracked_walk)
    rc, out = run(tmp_path, root)
    monkeypatch.undo()
    assert rc == 0
    hdir = str(root / 'H')
    assert opened and not any(p.startswith(hdir) for p in opened + walked)
    assert sorted(walked) == sorted(str(root / f) for f, _, _ in M.CANDIDATES)


def test_no_absolute_path_in_any_output_and_nothing_written_outside_out(tmp_path):
    root = write_world(tmp_path, known_value)
    repo_outs = [os.path.join(REPO, p) for p in OUT_FILES]
    before = [os.path.exists(p) and os.stat(p).st_mtime_ns for p in repo_outs]
    top = sorted(os.listdir(REPO))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    script = os.path.join(REPO, 'scripts', 'p106_memetic_selection.py')
    out = tmp_path / 'out'
    for extra in ([], ['--from-json']):
        p = subprocess.run([sys.executable, '-B', script, '--records', 'records', '--out', 'out',
                            *extra], cwd=str(tmp_path), env=env, capture_output=True, text=True,
                           timeout=60)
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        assert str(tmp_path) not in p.stdout
    assert [os.path.exists(p) and os.stat(p).st_mtime_ns for p in repo_outs] == before
    assert sorted(os.listdir(REPO)) == top
    written = [os.path.join(d, f) for d, _, fs in os.walk(str(tmp_path)) for f in fs]
    assert all(f.startswith((str(root) + os.sep, str(out) + os.sep)) for f in written)
    assert outputs(out) == sorted(OUT_FILES)
    for f in OUT_FILES:
        text = (out / f).read_text()
        for bad in (REPO, str(tmp_path), '/home/', '/tmp/'):
            assert bad not in text, f'{f} holds {bad}'
    R = result(out)
    assert R['records_root'] is None
    assert all(not os.path.isabs(r['record']) for r in R['runs'])


def test_from_json_rebuilds_the_markdown_and_the_macros_without_records(tmp_path):
    root = write_world(tmp_path, known_value)
    rc, out = run(tmp_path, root)
    assert rc == 0
    keep = {f: (out / f).read_text() for f in OUT_FILES}
    for f in OUT_FILES:
        if not f.endswith('.json'):
            (out / f).unlink()
    for d, _, fs in os.walk(str(root), topdown=False):
        for f in fs:
            os.remove(os.path.join(d, f))
        os.rmdir(d)
    rc, out = run(tmp_path, root, extra=['--from-json'])
    assert rc == 0
    assert {f: (out / f).read_text() for f in OUT_FILES} == keep


def test_from_json_refuses_an_edited_json_and_keeps_it(tmp_path, capsys):
    rc, out = run(tmp_path, write_world(tmp_path, known_value))
    assert rc == 0
    js = out / 'results' / 'memetic_selection.json'
    R = json.loads(js.read_text())
    R['runs'][0]['makespan'] += 7
    js.write_text(json.dumps(R))
    rc, out = run(tmp_path, tmp_path / 'records', extra=['--from-json'])
    assert rc != 0 and 'differ from the JSON' in capsys.readouterr().err
    assert outputs(out) == ['results/memetic_selection.json']


def test_macro_names_are_unique_and_every_line_says_what_it_is(tmp_path):
    rc, out = run(tmp_path, write_world(tmp_path, known_value))
    assert rc == 0
    lines = (out / 'manuscript' / M.MACRO_FILE).read_text().splitlines()
    defs = [ln for ln in lines if ln.startswith('\\newcommand')]
    names = [re.match(r'\\newcommand\{\\([^}]*)\}', ln).group(1) for ln in defs]
    assert len(names) == len(set(names)) == len(lines) - 1
    assert all(re.fullmatch(r'Msel[A-Za-z]+', n) for n in names)
    assert all(re.search(r'\}  % \S', ln) for ln in defs)
    L = M.macro_lines(result(out))
    with pytest.raises(M.Refusal):
        M.check_macro_names(L + [L[0]], str(out / 'manuscript' / M.MACRO_FILE))
    with pytest.raises(M.Refusal):
        M.check_macro_names([('MselBad1', '2', 'digit in the name')],
                            str(out / 'manuscript' / M.MACRO_FILE))


@pytest.mark.skipif(not os.path.isfile(SUPPLEMENT),
                    reason='needs the source of the manuscript '
                           'manuscript/supp_new/sn05_baselines.tex, which the release '
                           'does not hold')
def test_the_supplement_uses_only_macros_the_script_writes(tmp_path):
    rc, out = run(tmp_path, write_world(tmp_path, known_value))
    assert rc == 0
    body = '\n'.join(ln for ln in open(SUPPLEMENT).read().splitlines()
                     if not ln.lstrip().startswith('%'))
    used = set(re.findall(r'\\(Msel[A-Za-z]+)', body))
    assert used and used <= set(macros(out)), sorted(used - set(macros(out)))
