"""scripts/p138_decision_rows.py: the paired difference behind the lowest entry of each row of
the decision table, with a 95% interval.

  * on the synthetic trees of p124 and p137 and a training-size world with p94's own full run on
    it: the 20 rows, the printed values of every row equal to the records of p124 (600 s), p137
    (60 s, both memetic columns from the 60-s runs; 300 s, both from the 300-s runs of its
    five-minute record, the other four columns p124's) and p94 (the table on the common
    instances), every value equal to the one built into the tree, the cells of every row, and
    the macro file (Drow plus letters, one comment per macro, a dagger or nothing per row, the
    flag of a lowest search started from the policy's schedule, the counts per reading time);
  * L, R, P and O on planted rows; a planted tie (two columns equal in every cell, or two columns
    whose difference changes sign across the instances) is not decided and carries the dagger; a
    planted decided row;
  * the interval of a paired difference equals p94.compare_at on the same per-cell differences;
  * the counts per reading time (60, 300, 600 s) and over the twelve rows; the count of rows
    whose lowest column uses the policy at each time equals its record, and a planted difference
    refuses;
  * the five-minute record is required: without it the run refuses and --check returns 1, with
    --no-five-minute the rows at 300 s are p124's, the outputs say so in their first lines and no
    macro file is written; a preview (--min-seeds) passes the preview of the five-minute record
    through on the complete seeds of each 300-s set, says so in its first lines and writes no
    macro file, and is refused with too few complete seeds; the full run refuses that preview;
  * refusals: a dry record read outside the dry mode, a preview of p124 or of either record of
    p137, records of
    different training runs, instance counts or bootstrap counts, a row of a record whose columns
    are means over different cells, a printed value that differs from a record (the refusal
    names the row and the column); every refusal leaves no output and --check returns 1;
  * the macro names are unique against the other macro files and the prefix is refused once
    another .tex file uses it;
  * the options of the five-minute sets and of the preview;
  * the chain step decision_rows after memetic_five_minute.

Run from the repository root on one core:  python -B -m pytest -q tests/test_p138_decision_rows.py
CPU only; every file is written under the pytest temporary folder.
"""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import p138_decision_rows as M  # noqa: E402

T = M.T
P124 = M.P124
P137 = M.P137
SF = M.SF
BOOT = 300
COMBOS4 = ('N', 'L', 'W', 'LW')


# ----------------------------------------------------------------- the dry tree
@pytest.fixture(scope='module')
def dry(tmp_path_factory):
    tree = str(tmp_path_factory.mktemp('p138') / 'tree')
    os.makedirs(tree)
    res, names, paths, a = M.dry_run(tree, boot=BOOT)
    return dict(tree=tree, res=res, names=names, paths=paths, a=a)


def record_files(tree):
    return dict(p124=os.path.join(tree, 'out', M.P124_JSON),
                p137=os.path.join(tree, 'out', M.P137_JSON),
                p137_five=os.path.join(tree, 'out', M.P137_FIVE_JSON),
                p94=os.path.join(tree, 'p94', M.TIMED_JSON))


def no_outputs(tree):
    return not any(os.path.exists(os.path.join(tree, 'out', p)) for p in M.OUT_FILES.values())


def refusal(dry, which, change, dry_flag=True):
    """Run the analysis of the dry tree with one record changed by change(record) in place; the
    record is put back afterwards. Returns the refusal texts and whether outputs remain."""
    path = record_files(dry['tree'])[which] if which else None
    text = open(path).read() if path else None
    try:
        if path:
            J = json.loads(text)
            change(J)
            with open(path, 'w') as f:
                json.dump(J, f)
        with P124.dry_setting(dry['tree']):
            with pytest.raises(M.Refusal) as e:
                M.run(dry['a'], sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False,
                      dry=dry_flag)
            gone = no_outputs(dry['tree'])
            rc = M.check(dry['a'], sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False,
                         dry=dry_flag)
    finally:
        if path:
            with open(path, 'w') as f:
                f.write(text)
    return e.value.args[0], gone, rc


def rows_of(res, part=None):
    return [r for r in res['rows'] if part is None or r['part'] == part]


# ----------------------------------------------------------------- the whole path
def test_twenty_rows_equal_the_records_and_the_built_values(dry):
    res = dry['res']
    assert [(r['size'], r['budget'], r['entry']) for r in res['rows']] == \
        [(s, b, k) for s in M.PROD_SIZES for b in M.PROD_BUDGETS for k in M.ENTRIES] + \
        [(M.TRAIN_SIZE, b, k) for b in M.TRAIN_BUDGETS for k in M.ENTRIES]
    files = record_files(dry['tree'])
    J124, J137, J5, J94 = (json.load(open(files[k])) for k in ('p124', 'p137', 'p137_five',
                                                                 'p94'))
    for r in rows_of(res, 'prod'):
        if r['budget'] in M.SHORT_BUDGETS:
            J = J137 if r['budget'] == M.ONE_MINUTE else J5
            one = J['decision']['rows'][r['size']][r['entry']]
            assert r['values'] == one['after_values']
            assert sorted(r['lowest']) == sorted(one['after'])
            assert list(r['cells']) == [one['instances'], one['combinations']]
            assert r['source'] == f'p137 at {r["budget"]} s'
        else:
            rec = J124['decision_table']['sizes'][r['size']]['rows'][str(r['budget'])][r['entry']]
            assert r['values'] == rec['values'] and sorted(r['lowest']) == sorted(rec['bold'])
            assert r['cells'][0] == rec['instances']['rules']
    dc = J94['results'][M.TRAIN_SIZE]['decision_common']
    assert dc['instances'] == list(range(6))
    for r in rows_of(res, 'train'):
        rec = dc['rows'][str(r['budget'])][r['entry']]
        assert r['values'] == rec['values'] and sorted(r['lowest']) == sorted(rec['bold'])
        assert r['values']['memetic_policy'] is None and r['values']['cpsat_policy'] is None
        assert r['best_with'] == ['policy_sample'] and r['cells'][0] == list(range(6))
    assert M.dry_verify(res) == []


def test_differences_are_paired_on_the_cells_of_the_row(dry):
    for r in dry['res']['rows']:
        for key, first, second in (('lowest_vs_runner_up', 'lowest_column', 'runner_up_column'),
                                   ('with_vs_without', 'with_column', 'without_column')):
            x = r[key]
            assert abs(x['mean'] - (r['means'][r[first]] - r['means'][r[second]])) < 1e-9
            assert x['n'] == len(r['cells'][0]) and x['c'] == len(r['cells'][1]) == 4
            assert x['direction'] == SF.direction(x['lo'], x['hi'])
        assert r['lowest_column'] in r['lowest'] and r['runner_up_column'] in r['runner_up']
        assert r['with_column'] in M.WITH_POLICY and r['without_column'] in M.WITHOUT_POLICY


def test_counts_per_reading_time_and_the_records(dry):
    """Built: at 60 s three of the four rows have a search from the policy's schedule lowest (the
    60-s runs from the rule schedule win at 50 modules with setups), at 300 s two (the 300-s runs
    from the policy's schedule win at 50 modules with setups, those from the rule schedule both
    rows at 80 modules), at 600 s all four; the count over the twelve rows is no longer p137's
    count over its twelve rows (eleven), and each count per time equals its record."""
    res = dry['res']
    files = record_files(dry['tree'])
    J124, J137, J5 = (json.load(open(files[k])) for k in ('p124', 'p137', 'p137_five'))
    c = res['counts']['prod']
    assert c['rows'] == 12 and res['counts']['train']['rows'] == 8
    at = res['counts']['prod_at']
    assert list(at) == ['60', '300', '600']
    assert [at[b]['lowest_pol_start'] for b in at] == [3, 2, 4]
    assert [at[b]['lowest_uses_policy'] for b in at] == [3, 2, 4]
    assert c['lowest_pol_start'] == c['lowest_uses_policy'] == 9
    assert J137['decision']['all_rows']['uses_policy_after'] == 11 != c['lowest_uses_policy']
    assert at['60']['lowest_uses_policy'] == J137['decision']['counts']['uses_policy_after']
    assert at['300']['lowest_uses_policy'] == J5['decision']['counts']['uses_policy_after']
    for key in ('rows', 'decided', 'tied', 'with_ahead', 'with_neither', 'with_behind',
                'lowest_pol_start', 'lowest_uses_policy'):
        assert sum(at[b][key] for b in at) == c[key], key
    assert c['decided'] + c['tied'] == 12
    assert c['with_ahead'] + c['with_neither'] + c['with_behind'] == 12
    assert c['lowest_pol_start_ahead'] == sum(
        1 for r in rows_of(res, 'prod') if set(r['lowest']) & set(M.POLICY_START)
        and r['with_vs_without']['direction'] == 'ahead')
    rows = rows_of(res, 'prod')
    short = {60: (J137, 'p137'), 300: (J5, 'p137 five')}
    assert M.count_problems(rows, J124, short, 'p124') == []
    J5b = json.loads(json.dumps(J5))
    J5b['decision']['counts']['uses_policy_after'] = 3
    bad = M.count_problems(rows, J124, {60: (J137, 'p137'), 300: (J5b, 'p137 five')}, 'p124')
    assert bad == ['rows at 300 s whose lowest column uses the policy: 2 here, 3 in p137 five']


def test_macro_file(dry):
    path = dry['paths']['macros']
    txt = open(path).read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert names == dry['names'] and len(names) == len(set(names))
    assert all(re.fullmatch(r'Drow[A-Za-z]+', n) for n in names)
    for line in txt.splitlines():
        if line.startswith('\\newcommand'):
            assert '  % ' in line, line
    for r in dry['res']['rows']:
        base = f'Drow{T.SIZE_WORD[r["size"]]}{M.TIME_WORD[r["budget"]]}{r["entry"]}'
        for w in ('Lowest', 'RunnerUp', 'BestWith', 'BestWithout', 'LowestVsRunnerUp',
                  'LowestVsRunnerUpLo', 'LowestVsRunnerUpHi', 'LowestVsRunnerUpDir',
                  'WithVsWithout', 'WithVsWithoutLo', 'WithVsWithoutHi', 'WithVsWithoutDir', 'Tie'):
            assert base + w in names, base + w
        tie = re.search(r'\\newcommand\{\\' + base + r'Tie\}\{(.*)\}  %', txt).group(1)
        assert tie == ('' if r['decided'] else M.TIE_MARK)
        low = re.search(r'\\newcommand\{\\' + base + r'Lowest\}\{([^}]*)\}', txt).group(1)
        assert low == P137.lowest_word(r['lowest'])
    for part in ('Prod', 'Train'):
        for w in ('RowN', 'DecidedN', 'TiedN', 'WithAheadN', 'WithNeitherN', 'WithBehindN',
                  'LowestPolStartN', 'LowestPolStartAheadN'):
            assert f'Drow{part}{w}' in names
    assert 'DrowBoot' in names and 'DrowProdLowestUsesPolicyN' in names
    for t in ('Sixty', 'ThreeHundred', 'SixHundred'):
        for w in ('RowN', 'DecidedN', 'TiedN', 'LowestPolStartN', 'LowestUsesPolicyN',
                  'WithAheadN', 'WithNeitherN', 'WithBehindN'):
            assert f'DrowProd{t}{w}' in names, (t, w)
    for t, b in (('Sixty', '60'), ('ThreeHundred', '300'), ('SixHundred', '600')):
        got = re.search(r'\\newcommand\{\\DrowProd' + t + r'LowestPolStartN\}\{(\d+)\}', txt)
        assert got.group(1) == str(dry['res']['counts']['prod_at'][b]['lowest_pol_start'])
    for r in dry['res']['rows']:
        base = f'Drow{T.SIZE_WORD[r["size"]]}{M.TIME_WORD[r["budget"]]}{r["entry"]}'
        flag = re.search(r'\\newcommand\{\\' + base + r'LowestIsPolStart\}\{(\d)\}', txt).group(1)
        assert flag == str(int(bool(set(r['lowest']) & set(M.POLICY_START))))
    assert re.search(r'\\newcommand\{\\DrowEightyThreeHundredSetupLowestIsPolStart\}\{0\}', txt)
    assert re.search(r'\\newcommand\{\\DrowFiftyThreeHundredSetupLowestIsPolStart\}\{1\}', txt)
    # unique against every other macro file of the repository
    assert M.check_names(M.macro_lines(dry['res']),
                         os.path.join(ROOT, M.OUT_FILES['macros'])) == names


def test_prefix_used_by_another_file_refuses(dry, tmp_path):
    own = tmp_path / 'manuscript' / 'macros_decision_rows.tex'
    own.parent.mkdir()
    (own.parent / 'other.tex').write_text('\\newcommand{\\DrowSomething}{1}\n')
    with pytest.raises(M.Refusal) as e:
        M.check_names(M.macro_lines(dry['res']), str(own))
    assert 'the prefix Drow is used by \\DrowSomething' in e.value.args[0][0]


def test_check_passes_on_the_dry_tree(dry, capsys):
    with P124.dry_setting(dry['tree']):
        rc = M.check(dry['a'], sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False, dry=True)
    assert rc == 0 and 'p138: every input present' in capsys.readouterr().out


# ----------------------------------------------------------------- planted rows
def planted(n=9, c=4, seed=3, **cols):
    """Per-cell matrices of the six columns: cols {column: offset or matrix}; a column left out
    has no number."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n, c))
    out = {k: None for k in M.COLUMNS}
    for k, v in cols.items():
        out[k] = np.asarray(v, dtype=float) if np.ndim(v) else base + v
    return out


def test_lowest_runner_up_best_with_and_without_on_a_planted_row():
    mats = planted(rules=3.0, policy_sample=-1.0, memetic_rule=-2.0, cpsat_rule=-2.5,
                   memetic_policy=-4.0, cpsat_policy=-3.0)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert r['lowest'] == ['memetic_policy'] and r['runner_up'] == ['cpsat_policy']
    assert r['best_with'] == ['memetic_policy'] and r['best_without'] == ['cpsat_rule']
    assert abs(r['lowest_vs_runner_up']['mean'] + 1.0) < 1e-12
    assert abs(r['with_vs_without']['mean'] + 1.5) < 1e-12
    assert r['decided'] and r['lowest_vs_runner_up']['direction'] == 'ahead'
    # the lowest column does not use the policy: P and O on the two sides
    mats = planted(rules=3.0, policy_sample=-1.0, memetic_rule=-2.0, cpsat_rule=-5.0,
                   memetic_policy=-4.0, cpsat_policy=-3.0)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert r['lowest'] == ['cpsat_rule'] and r['runner_up'] == ['memetic_policy']
    assert r['best_with'] == ['memetic_policy'] and r['best_without'] == ['cpsat_rule']
    assert r['with_vs_without']['direction'] == 'behind'
    # the training size: no search started from the policy's schedule, P is policy_sample
    mats = planted(rules=3.0, policy_sample=-1.0, memetic_rule=-2.0, cpsat_rule=-0.5)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert r['best_with'] == ['policy_sample'] and r['best_without'] == ['memetic_rule']
    assert r['values']['memetic_policy'] is None and r['lowest'] == ['memetic_rule']


def test_a_planted_tie_is_not_decided_and_carries_the_dagger():
    # two columns equal in every cell: one printed value, both lowest
    mats = planted(rules=3.0, policy_sample=-1.0, memetic_rule=-2.0, cpsat_rule=-2.5,
                   memetic_policy=-4.0, cpsat_policy=-4.0)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert sorted(r['lowest']) == ['cpsat_policy', 'memetic_policy'] and r['printed_tie']
    assert r['runner_up'] == ['cpsat_rule'] and not r['decided']
    # two columns whose difference changes sign across the instances: the interval holds zero
    n = 9
    base = planted(n=n, rules=0.0)['rules']
    swing = np.repeat(np.array([0.6, -0.6, 0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.3])[:, None], 4,
                      axis=1)
    mats = planted(n=n, rules=base + 3.0, policy_sample=base - 1.0, memetic_rule=base - 2.0,
                   cpsat_rule=base - 2.5, memetic_policy=base - 4.0,
                   cpsat_policy=base - 4.0 + swing)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert len(r['lowest']) == 1 and not r['printed_tie']
    x = r['lowest_vs_runner_up']
    assert x['lo'] < 0 < x['hi'] and x['direction'] == 'neither' and not r['decided']
    res = dict(boot=BOOT, alpha=0.05, setting='-', selected_run=1, cpsat_configuration='-',
               rows=[dict(r, size=M.PROD_SIZES[0], budget=300, entry='NoSetup',
                          cells=[list(range(3, 12)), list(COMBOS4)])],
               counts={p: M.count_rows([]) for p in ('prod', 'train')})
    txt = '\n'.join(M.macro_lines(res))
    assert re.search(r'\\newcommand\{\\DrowFiftyThreeHundredNoSetupTie\}\{\\textsuperscript'
                     r'\{\$\\dagger\$\}\}', txt)
    assert re.search(r'\\newcommand\{\\DrowFiftyThreeHundredNoSetupLowestVsRunnerUpDir\}'
                     r'\{neither\}', txt)


def test_a_planted_decided_row():
    mats = planted(rules=3.0, policy_sample=-1.0, memetic_rule=-2.0, cpsat_rule=-2.5,
                   memetic_policy=-2.8, cpsat_policy=-3.0)
    r = M.analyse_row(mats, BOOT, 0.05)
    assert r['lowest'] == ['cpsat_policy'] and r['runner_up'] == ['memetic_policy']
    assert r['decided'] and r['lowest_vs_runner_up']['hi'] < 0
    c = M.count_rows([dict(r, part='prod')])
    assert c['decided'] == 1 and c['tied'] == 0 and c['lowest_pol_start'] == 1
    assert c['with_ahead'] == 1 and c['lowest_pol_start_ahead'] == 1


def test_interval_equals_compare_at_on_the_same_differences():
    rng = np.random.default_rng(11)
    for n in (6, 9):
        d = rng.normal(-0.3, 0.8, (n, len(COMBOS4)))
        mine = M.stat(d, 2000, 0.05)
        first, second = 1.0 + d / 100.0, np.ones_like(d)
        rows = T.compare_at(first, second, np.ones_like(d, dtype=bool), list(COMBOS4), 2000, 0.05)
        for key in ('Pool', 'NoSetup'):
            for f in ('mean', 'lo', 'hi'):
                assert abs(rows[key][f] - mine[f]) < 1e-9, (n, key, f)
            assert rows[key]['direction'] == mine['direction']
            assert (rows[key]['n'], rows[key]['c']) == (mine['n'], mine['c'])


def test_row_matrices_refuse_a_row_over_different_cells():
    n, C = 4, len(COMBOS4)
    first = np.full((n, C), 101.0)
    g = np.full((n, C), 100.0)
    pres = np.ones((n, C), dtype=bool)
    row = dict(values={c: (None if c in ('memetic_policy', 'cpsat_policy') else '+1.00')
                       for c in M.COLUMNS},
               instances={c: [0, 1, 2, 3] for c in M.COLUMNS},
               combinations_used={c: list(COMBOS4) for c in M.COLUMNS})
    row['instances']['cpsat_rule'] = [1, 2, 3]
    S = dict(decision={'300': {'NoSetup': row}})
    mats, cells, bad = M.row_matrices('row', S, 300, 'NoSetup', {
        c: (first, g, pres) for c in M.COLUMNS}, [0, 1, 2, 3], list(COMBOS4))
    assert mats is None and 'means over different cells' in bad[0]


# ----------------------------------------------------------------- refusals
def test_a_dry_record_outside_the_dry_mode_refuses(dry):
    bad, gone, rc = refusal(dry, None, None, dry_flag=False)
    assert gone and rc == 1
    assert any('dry True, this run reads False' in x for x in bad)
    assert any(M.P137_JSON in x and 'dry True' in x for x in bad)


@pytest.mark.parametrize('which', ['p124', 'p137', 'p137_five'])
def test_a_preview_refuses(dry, which):
    def change(J):
        J['run'] = 'preview'
        J['warning'] = 'PREVIEW: not for the paper'
    bad, gone, rc = refusal(dry, which, change)
    assert gone and rc == 1
    assert any("run 'preview', this run reads 'full'" in x for x in bad)


def test_records_of_different_training_runs_refuse(dry):
    def change(J):
        J['selection']['seed'] = 303
    bad, gone, rc = refusal(dry, 'p94', change)
    assert gone and rc == 1 and any('records of different training runs' in x for x in bad)


def test_records_of_different_instance_counts_refuse(dry):
    def change(J):
        J['instances'] = J['instances'][:-1]
    bad, gone, rc = refusal(dry, 'p137', change)
    assert gone and rc == 1 and any('instances [3, 4, 5, 6, 7, 8, 9, 10]' in x for x in bad)


def test_records_of_different_bootstrap_counts_refuse(dry):
    def change(J):
        J['boot'] = 2 * BOOT
    bad, gone, rc = refusal(dry, 'p94', change)
    assert gone and rc == 1 and any('records of different bootstrap counts' in x for x in bad)


def test_a_record_row_over_different_cells_refuses(dry):
    def change(J):
        row = J['decision_table']['sizes'][M.PROD_SIZES[0]]['rows']['300']['NoSetup']
        row['instances']['cpsat_rule'] = row['instances']['cpsat_rule'][1:]
    bad, gone, rc = refusal(dry, 'p124', change)
    assert gone and rc == 1
    assert any(f'{M.PROD_SIZES[0]}, combinations without setups, 300 s: the columns of the row '
               'of' in x and 'means over different cells' in x for x in bad)


@pytest.mark.parametrize('which, size, b, key, col', [
    ('p124', '80x25n30', 600, 'Setup', 'memetic_policy'),
    ('p137', '50x25n30', 60, 'NoSetup', 'memetic_rule'),
    ('p137_five', '80x25n30', 300, 'Setup', 'memetic_rule'),
    ('p137_five', '50x25n30', 300, 'Setup', 'memetic_policy'),
    ('p94', '10x25', 10, 'Setup', 'cpsat_rule')])
def test_a_printed_value_that_differs_from_the_record_refuses(dry, which, size, b, key, col):
    def change(J):
        if which == 'p124':
            J['decision_table']['sizes'][size]['rows'][str(b)][key]['values'][col] = '-9.99'
        elif which in ('p137', 'p137_five'):
            J['decision']['rows'][size][key]['after_values'][col] = '-9.99'
        else:
            J['results'][size]['decision_common']['rows'][str(b)][key]['values'][col] = '-9.99'
    bad, gone, rc = refusal(dry, which, change)
    assert gone and rc == 1
    want = f'{size}, {T.entry_text(key)}, {b} s, {col}: '
    assert any(x.startswith(want) and x.endswith('-9.99 in ' + M.shown(
        record_files(dry['tree'])[which])) for x in bad), bad


# ----------------------------------------------------------------- the chain step
def test_chain_step():
    src = open(os.path.join(ROOT, 'scripts', 'run_analysis.sh')).read()
    args = ('--ref-slots "${TIMED_SLOT_LIST[@]}" --pol-slots "${TIMED_SLOT_LIST[@]}" '
            '--five-slots "${MFIVE_SLOT_LIST[@]}" --five-pol-slots "${MFIVE_POL_SLOT_LIST[@]}"')
    assert f'run_cmd scripts/p138_decision_rows.py {args} || return 1' in src
    assert f'check_cmd scripts/p138_decision_rows.py --check {args}' in src
    steps = re.search(r'STEPS\+=\(rulecmp_factory(.*?)\)', src, re.S).group(1).split()
    assert steps[steps.index('memetic_one_minute') + 1] == 'memetic_five_minute'
    assert steps[steps.index('memetic_five_minute') + 1] == 'decision_rows'
    assert re.search(r'^    decision_rows\) echo "memetic_parallel memetic_one_minute '
                     r'memetic_five_minute";;', src, re.M)
    assert 'echo "$PACT_OUT/manuscript/macros_decision_rows.tex ' \
           'manuscript/macros_decision_rows.tex"' in src
    assert "'manuscript/macros_decision_rows.tex'" in src


# ----------------------------------------------------------------- the five-minute record
def test_rows_at_300_take_both_memetic_columns_from_the_300s_runs(dry):
    """At 300 s the two memetic columns are the 300-s runs of p137's record at 300 s, the other
    four columns p124's record at 300 s; at 600 s every column is p124's."""
    files = record_files(dry['tree'])
    J124, J5 = (json.load(open(files[k])) for k in ('p124', 'p137_five'))
    for r in rows_of(dry['res'], 'prod'):
        rec = J124['decision_table']['sizes'][r['size']]['rows'][str(r['budget'])][r['entry']]
        if r['budget'] != M.FIVE:
            continue
        five = J5['decision']['rows'][r['size']][r['entry']]
        for c in M.COLUMNS:
            if c in ('memetic_rule', 'memetic_policy'):
                assert r['values'][c] == five['after_values'][c] != rec['values'][c], (r, c)
            else:
                assert r['values'][c] == rec['values'][c]
    assert dry['res']['records']['p137_five'] == M.shown(files['p137_five'])
    assert dry['res']['five_seeds'] == {'rule': list(range(1, 9)), 'policy': list(range(1, 9))}


def test_without_the_five_minute_record_refuses_unless_asked(dry, tmp_path):
    path = record_files(dry['tree'])['p137_five']
    keep = path + '.keep'
    os.rename(path, keep)
    try:
        with P124.dry_setting(dry['tree']):
            with pytest.raises(M.Refusal) as e:
                M.run(dry['a'], sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False, dry=True)
            assert no_outputs(dry['tree'])
            assert any('no record of p137 at 300 s (step memetic_five_minute' in x
                       for x in e.value.args[0])
            assert M.check(dry['a'], sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False,
                           dry=True) == 1
        res, names, paths, _ = M.dry_run(dry['tree'], boot=BOOT, build=False,
                                         extra=['--no-five-minute', '--out', str(tmp_path / 'nf')])
    finally:
        os.rename(keep, path)
    assert names is None and not os.path.exists(paths['macros'])
    assert res['run'] == 'without the five-minute record'
    assert res['warning'] == M.NO_FIVE_WARNING and res['five_seeds'] is None
    text = open(paths['json']).read()
    assert text.splitlines()[1].startswith(' "warning": "WITHOUT THE FIVE-MINUTE RECORD')
    md = open(paths['md']).read().splitlines()
    assert 'without the five-minute record' in md[0] and md[2].startswith('**WITHOUT THE FIVE')
    J124 = json.load(open(record_files(dry['tree'])['p124']))
    for r in rows_of(res, 'prod'):
        if r['budget'] == M.FIVE:
            rec = J124['decision_table']['sizes'][r['size']]['rows']['300'][r['entry']]
            assert r['values'] == rec['values'] and r['source'] == 'p124'
    assert M.dry_verify(res) == []
    assert 'p137_five' not in res['records']


def test_preview_passes_the_five_minute_preview_through(tmp_path, capsys):
    tree = tmp_path / 'tree'
    pol_drop = [(k, '80x25n30', 'SW', 7) for k in (1, 3)]
    out = tmp_path / 'p138preview'
    res, names, paths, _ = M.dry_run(tree, boot=BOOT, five=dict(min_seeds=2, pol_drop=pol_drop),
                                     extra=['--min-seeds', '2', '--out', str(out)])
    assert res['run'] == 'preview' and names is None and not os.path.exists(paths['macros'])
    assert res['five_seeds'] == {'rule': list(range(1, 9)), 'policy': [2] + list(range(4, 9))}
    assert res['warning'].startswith('PREVIEW') and '2,4-8' in res['warning']
    assert open(paths['json']).read().splitlines()[1].startswith(' "warning": "PREVIEW')
    md = open(paths['md']).read().splitlines()
    assert '(preview run)' in md[0] and md[2].startswith('**PREVIEW')
    assert M.dry_verify(res) == []
    J5 = json.load(open(tree / 'preview' / M.P137_FIVE_JSON))
    assert J5['run'] == 'preview'
    for r in rows_of(res, 'prod'):
        if r['budget'] == M.FIVE:
            assert r['values'] == J5['decision']['rows'][r['size']][r['entry']]['after_values']
    # the preview record is refused by the full run, and a preview needs enough complete seeds
    with P124.dry_setting(str(tree)):
        a = M.build_parser().parse_args(['--result-root', str(tree / 'rr'), '--data-root',
                                         str(tree / 'data'), '--bars', str(tree / 'bars.json'),
                                         '--boot', str(BOOT), '--timed-json',
                                         str(tree / 'p94' / M.TIMED_JSON), '--p137-five-json',
                                         str(tree / 'preview' / M.P137_FIVE_JSON)])
        assert M.check_options(a) == []
        with pytest.raises(M.Refusal) as e:
            M.run(a, sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False, dry=True)
        assert any("run 'preview', this run reads 'full'" in x for x in e.value.args[0])
        a = M.build_parser().parse_args(['--result-root', str(tree / 'rr'), '--data-root',
                                         str(tree / 'data'), '--bars', str(tree / 'bars.json'),
                                         '--boot', str(BOOT), '--timed-json',
                                         str(tree / 'p94' / M.TIMED_JSON), '--p137-five-json',
                                         str(tree / 'preview' / M.P137_FIVE_JSON),
                                         '--min-seeds', '7', '--out', str(tmp_path / 'x')])
        assert M.check_options(a) == []
        with pytest.raises(M.Refusal) as e:
            M.run(a, sel=P124.DRY_SEL, cpsat_arm=P124.DRY_CPSAT, chosen=False, dry=True)
        assert any('--min-seeds 7' in x and 'in each set' in x for x in e.value.args[0])


def test_options_of_the_preview_and_of_the_five_minute_sets(tmp_path):
    ap = M.build_parser()
    a = ap.parse_args([])
    assert (a.five_slots, a.five_pol_slots, a.no_five_minute, a.min_seeds) == \
        (['16-23'], ['8-15'], False, None)
    assert any('needs --out' in x for x in M.check_options(ap.parse_args(['--min-seeds', '1'])))
    assert any('give one' in x for x in M.check_options(
        ap.parse_args(['--min-seeds', '1', '--no-five-minute', '--out', str(tmp_path / 'x')])))
    assert any("core set 'x-y'" in x for x in M.check_options(
        ap.parse_args(['--five-pol-slots', 'x-y'])))
    a = ap.parse_args(['--five-slots', '8-15', '16-23', '--five-pol-slots', '16-23'])
    M.check_options(a)
    assert (a.a5.budget, a.a5.slots, a.a5.pol_slots) == (300, ['8-15', '16-23'], ['16-23'])
    assert a.a5.pa.budgets == [300] and a.a137.pa.budgets == [60]
    assert P137.BUDGET == 60                                  # p137 reads 60 s again after
    a = ap.parse_args(['--no-five-minute'])
    M.check_options(a)
    assert a.a5 is None
