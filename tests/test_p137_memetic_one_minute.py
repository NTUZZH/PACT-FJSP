"""scripts/p137_memetic_one_minute.py: the memetic algorithm with a budget of 60 s, from the rule
schedule and from the policy's schedule, read one minute after the request beside the 600-s
runs, the policy and CP-SAT.

  * the record names of the 60-s runs are the planner's (Campaign.record of memetic_rule@60 and
    memetic_policy@60 with --memetic-seed k, folders <size>_memetic60s and
    <size>_memetic60s_pactwarm_s<run>, budget 60 s and the start caches in the expected fields);
  * on p124's synthetic tree with eight seeds of 60-s records of each start: every difference
    equals the one built into the tree, the best of eight is the shortest run of each cell (also
    when another seed than the usual one is the shortest), the runs from the policy's schedule are
    read on the o_pol clock (a step the o_rule clock would show at 60 s is not read), the spreads
    and the smallest cpu_share of each set are the built ones, and every macro is Mmin plus
    letters with a comment;
  * a missing 60-s record or an invalid one of either set (low cpu_share, a core outside the
    slots, two threads), or one whose start cache differs from the plan and the 600-s record: the
    run refuses and leaves no output, --check lists the cell and returns 1, a rejected record
    prints one line with the planner's reason; a configuration, code hash or start cache that
    differs from the 600-s record refuses;
  * p124's smp_vs_best at 60 s, the two memetic columns of its decision table, the bold columns of
    its twelve rows and its count of rows whose lowest column uses the policy must equal p124's
    record: a planted difference refuses and leaves no output;
  * the one-minute rows of the decision table: the lowest column before, with the 60-s value from
    the rule schedule alone and with both 60-s values, on hand-made rows (each 60-s value below,
    above and equal to the lowest) and on the tree (the second column becomes the lowest of one
    row and not of the others; with the policy's sampled schedules made the lowest of two rows,
    one keeps them); the bold markers and the lowest-column words after the replacement; the
    count over the twelve rows;
  * --min-seeds: a preview on every complete seed of each set, the two sets with different seeds,
    marked in every output, no macro file, refused without --out, with an output folder inside
    the repository, or with fewer complete seeds in a set;
  * the prefix Mmin is refused once another .tex file defines a macro with it;
  * --budget 300, on a tree with the 300-s records beside the 60-s records: the planner's record
    names (<size>_memetic300s, <size>_memetic300s_pactwarm_s<run>), the names of the module
    (prefix Mfiv, keys m300_vs_m600 and the others, macro words ThreeHundred, outputs
    memetic_five_minute, core sets 16-23 and 8-15) and their return to 60 s; every difference at
    300 s equal to the built one (a step at 200 s counts, a step at 294 s of a run from the
    policy's schedule does not); the four five-minute rows before and after both 300-s values;
    no count over the twelve rows; a step planted in a 300-s record changes the five-minute
    reading and leaves the one-minute reading byte for byte the same; a missing, invalid or
    foreign 300-s record, and a difference from p124's record at 300 s, refuse; the preview; the
    macro names of both readings unique against each other and every other macro file.

Run from the repository root on one core:  python -B -m pytest -q tests/test_p137_memetic_one_minute.py
CPU only; every file is written under the pytest temporary folder.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import p137_memetic_one_minute as M  # noqa: E402

T = M.T
P124 = M.P124
BOOT = 200
PY = sys.executable
INST = M.P.parse_instances(M.INSTANCES)
SETUP = T.GROUPS[1][2]


def macro_names(path):
    return re.findall(r'\\newcommand\{\\([^}]*)\}', open(path).read())


def macro_value(text, name):
    m = re.search(r'\\newcommand\{\\' + name + r'\}\{([^}]*)\}', text)
    assert m, name
    return m.group(1)


def no_outputs(tree):
    return not any(os.path.exists(os.path.join(tree, 'out', p)) for p in M.OUT_FILES.values())


def test_record_names_are_the_planners(tmp_path):
    for k in (1, 4, 8):
        a = T.build_parser().parse_args(['--result-root', str(tmp_path), '--memetic-seed', str(k)])
        for token, folder, policy in (('memetic_rule@60', '80x25n30_memetic60s', False),
                                      ('memetic_policy@60', '80x25n30_memetic60s_pactwarm_s304',
                                       True)):
            camp = T.make_campaign(['80x25n30'], [f'{token}:3-4'], a, 304, 'lns')
            for seg in camp.segments:
                for c in camp.cells(seg):
                    p = camp.record(c)
                    assert p == os.path.join(str(tmp_path), 'or_solution', 'FAMILY', folder,
                                             c.combo, f'instance_{c.idx:03d}_seed{k}.json')
                    e = camp.expected_fields(c)
                    assert e['seed'] == k and e['budget_s'] == 60.0
                    assert bool(e['seed_from']) == policy and e['warm_cache']
        camp = T.make_campaign(['80x25n30'], ['memetic_rule@60:3-4'], a, 304, 'lns')
        c = next(c for seg in camp.segments for c in camp.cells(seg))
        assert camp.record(c) == M.path60(str(tmp_path), c.size, c.combo, c.idx, k)
        camp = T.make_campaign(['80x25n30'], ['memetic_policy@60:3-4'], a, P124.DRY_SEL, 'lns')
        c = next(c for seg in camp.segments for c in camp.cells(seg))
        assert camp.record(c) == M.path60(str(tmp_path), c.size, c.combo, c.idx, k, policy=True)


def test_dry_numbers_equal_the_built_ones(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    assert res['run'] == 'full' and M.dry_verify(res) == []
    assert res['cells_valid'] == res['cells_total'] == 2304
    for name in M.SETS:
        assert res['set_states'][name]['cells_valid'] == 1152
    assert res['seeds'] == res['pol_seeds'] == list(M.SEEDS)
    assert res['init_phase_s'] == pytest.approx(6.0) and res['ref_init_phase_s'] == pytest.approx(60.0)
    assert res['set_states']['rule']['share_min'] == pytest.approx(0.991)
    assert res['set_states']['policy']['share_min'] == pytest.approx(0.990)
    assert res['decision']['counts'] == M.DRY_COUNTS
    for p in paths.values():
        assert os.path.exists(p)
    S = res['results']['50x25n30']
    assert S['cells_used'] == S['pol_cells_used'] == 72
    assert S['records']['cores']['1'] == {'16-23': 72}
    assert S['pol_records']['cores']['1'] == {'16-23': 72}       # the two slots of the second set
    assert S['pol_records']['cores']['8'] == {'8-15': 72}
    # built: the best of eight 60-s runs (seed 8) is shorter than the run of seed 1
    for j, r in enumerate(M.COMBOS):
        e = [M.dry_expected('50x25n30', i, j) for i in INST]
        assert all(x['best60'] < x['one60'] for x in e)
    # p124's smp_vs_best at 60 s, recomputed and equal to its record
    J = json.load(open(os.path.join(tmp_path, 'w', 'out', 'results', 'memetic_parallel.json')))
    rec = J['results']['50x25n30']['comparisons']['smp_vs_best']['60']
    assert M.printed_entry(S['p124_smp_vs_best']['Pool']) == M.printed_entry(rec['Pool'])
    for key in M.VERDICT_KEYS:
        fam = S['verdicts'][key]
        assert sum(fam['counts'].values()) == 8 and not fam['held']
    names = macro_names(paths['macros'])
    assert names and all(re.fullmatch(r'Mmin[A-Za-z]+', x) for x in names)
    assert len(names) == len(set(names))
    text = open(paths['macros']).read()
    for name, value in (('MminSeedN', '8'), ('MminBudgetS', '60'), ('MminInitPhaseS', '6'),
                        ('MminRefBudgetS', '600'), ('MminRefInitPhaseS', '60'),
                        ('MminCellN', '1152'), ('MminCellValidN', '1152'), ('MminInstN', '9'),
                        ('MminPolStartCellN', '1152'), ('MminPolStartCellValidN', '1152'),
                        ('MminPolStartShareMin', '0.9900'), ('MminShareMin', '0.9910'),
                        ('MminRowN', '4'), ('MminRowLowestChangeN', '2'),
                        ('MminRuleOnlyRowLowestChangeN', '1'), ('MminRowBelowLowestN', '1'),
                        ('MminRowLowestPolSmpBeforeN', '0'), ('MminRowLowestPolSmpAfterN', '0'),
                        ('MminRuleOnlyRowLowestPolSmpAfterN', '0'),
                        ('MminRowUsesPolicyBeforeN', '4'), ('MminRowUsesPolicyAfterN', '3'),
                        ('MminBothDecisionRowN', '12'),
                        ('MminBothDecisionLowestUsesPolicyN', '11')):
        assert macro_value(text, name) == value, name
    for sw in ('Fifty', 'Eighty'):
        for word in ('MSixtyVsMSixHundred', 'MpSixtyVsMpSixHundred', 'MpSixtyVsMSixty'):
            for suffix in ('Pool', 'PoolLo', 'PoolHi', 'PoolDir', 'NoSetup', 'Setup', 'LSW', 'NDir'):
                assert f'Mmin{sw}{word}{suffix}' in names
        for word in ('SmpVsMSixty', 'SmpVsMpSixty'):
            for suffix in ('Pool', 'PoolLo', 'PoolHi', 'PoolDir', 'LSWVerdict', 'VerdictAheadN',
                           'VerdictBehindN', 'VerdictNdN'):
                assert f'Mmin{sw}{word}{suffix}' in names
        for word in ('MSixtyVsPol', 'MpSixtyVsPol'):
            assert f'Mmin{sw}{word}NoSetup' in names and f'Mmin{sw}{word}Setup' in names
            assert f'Mmin{sw}{word}Pool' not in names            # the two rows of the table only
            assert f'Mmin{sw}{word}SetupLo' not in names
        assert f'Mmin{sw}PolVsMSixtyN' not in names               # pooled entries only
        for suffix in ('Pool', 'NoSetup', 'LSW'):
            assert f'Mmin{sw}Spread{suffix}' in names and f'Mmin{sw}PolStartSpread{suffix}' in names
        assert f'Mmin{sw}PolStartCellValidN' in names
        for g in ('NoSetup', 'Setup'):
            # the table's one-minute names: p124's with Mmin in the place of Mpar
            assert macro_value(text, f'Mmin{sw}MemRuleStartVsPolSixty{g}Sel') == \
                macro_value(text, f'Mmin{sw}MSixtyVsPol{g}')
            assert macro_value(text, f'Mmin{sw}MemPolStartVsPolSixty{g}') == \
                macro_value(text, f'Mmin{sw}MpSixtyVsPol{g}')
    assert all(' % ' in line for line in text.splitlines() if 'newcommand' in line)
    md = open(paths['md']).read()
    assert '## Decision table on eight cores' in md and "Spread over the 60-s runs started from " \
        "the policy's schedule" in md


def test_bold_markers_and_lowest_words_after_the_replacement(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    text = open(paths['macros']).read()
    # built lowest columns after both replacements: CP-SAT from the policy's schedule without
    # setups at both sizes, the 60-s runs from the rule schedule with setups at 50 modules, the
    # 60-s runs from the policy's schedule with setups at 80 modules (before: CP-SAT from the
    # policy's schedule in all four rows)
    want = {('Fifty', 'NoSetup'): 'cpsat_policy', ('Fifty', 'Setup'): 'memetic_rule',
            ('Eighty', 'NoSetup'): 'cpsat_policy', ('Eighty', 'Setup'): 'memetic_policy'}
    for (sw, g), col in want.items():
        names_row = {c: tuple('Mmin' + x[len('Tmd'):] for x in v)
                     for c, v in T.decision_names(sw, 'Sixty', g).items()}
        for c, (_, bold) in names_row.items():
            assert macro_value(text, bold) == (T.BOLD if c == col else ''), (sw, g, c)
        assert macro_value(text, f'Mmin{sw}Sixty{g}Lowest') == M.LOWEST_WORD[col]
    rows = res['decision']['rows']
    assert rows['80x25n30']['Setup']['before'] == ['cpsat_policy']
    assert rows['80x25n30']['Setup']['rule_only_after'] == ['cpsat_policy']
    assert rows['80x25n30']['Setup']['after'] == ['memetic_policy']
    assert rows['80x25n30']['Setup']['below_lowest_policy'] and rows['80x25n30']['Setup']['changed']
    for size, key in (('50x25n30', 'NoSetup'), ('80x25n30', 'NoSetup')):
        r = rows[size][key]                      # the second column does not become the lowest
        assert r['after'] == r['before'] == ['cpsat_policy'] and not r['changed']
        assert not r['below_lowest_policy']
    a12 = res['decision']['all_rows']
    assert (a12['rows'], a12['uses_policy_before'], a12['uses_policy_after']) == (12, 12, 11)
    assert a12['record_uses_policy'] == 12 and a12['budgets'] == [60, 300, 600]


def test_best_of_eight_is_the_shortest_run_of_each_cell(tmp_path):
    def shorter(root):
        p = M.path60(root, '80x25n30', 'W', 5, 3)
        rec = json.load(open(p))
        rec['anytime'].append([30.0, 31.0, 31.0, rec['makespan'] - 70])
        rec['makespan'] -= 70
        json.dump(rec, open(p, 'w'))
    res, *_ = M.dry_run(tmp_path / 'w', boot=BOOT, before=shorter)
    S = res['results']['80x25n30']
    j = M.COMBOS.index('W')
    e = [M.dry_expected('80x25n30', i, j) for i in INST]
    best = [x['best60'] if i != 5 else M.P124.r0(5, j) - 13 - 3 - 70 for i, x in zip(INST, e)]
    want = np.mean([100 * (b - x['best600']) / x['best600'] for b, x in zip(best, e)])
    assert S['comparisons']['m60_vs_m600']['W']['mean'] == pytest.approx(want, abs=1e-9)
    cell = next(c for c in S['spread']['cells'] if (c['combination'], c['instance']) == ('W', 5))
    assert cell['smallest'] == M.P124.r0(5, j) - 86 and cell['largest'] == M.P124.r0(5, j) - 14
    # the run of seed 1 is unchanged
    assert S['comparisons']['one60_vs_one600']['Pool']['mean'] == pytest.approx(np.mean(
        [100 * (x['one60'] - x['one600']) / x['one600']
         for j2 in range(8) for x in (M.dry_expected('80x25n30', i, j2) for i in INST)]),
        abs=1e-9)


def test_policy_start_is_read_on_the_o_pol_clock(tmp_path):
    """A run from the policy's schedule is read at run time 60 - max(o_pol, o_rule): a step of
    seed 3 at search time 40 s (run time 41 s, request time 49 s) counts at 60 s, a step of seed 2
    at search time 53 s (run time 54 s, request time 62 s with o_pol, 56 s with o_rule) does not;
    so does the step at DRY_LATE_S of every run."""
    assert P124.DRY_O_POL > P124.DRY_O_RULE
    j = M.COMBOS.index('N')
    g = P124.dry_policy(P124.DRY_SEL, 6, j)

    def steps(root):
        for k, t, v in ((3, 40.0, g - 70), (2, 53.0, g - 200)):
            p = M.path60(root, '50x25n30', 'N', 6, k, policy=True)
            rec = json.load(open(p))
            rec['anytime'] = sorted(rec['anytime'] + [[t, t + 1.0, t + 1.0, v]])
            json.dump(rec, open(p, 'w'))
    res, *_ = M.dry_run(tmp_path / 'w', boot=BOOT, before=steps)
    S = res['results']['50x25n30']
    cell = next(c for c in S['pol_spread']['cells'] if (c['combination'], c['instance']) == ('N', 6))
    assert cell['smallest'] == g - 70                  # the step at 40 s, not the one at 53 s
    assert cell['largest'] == g - 13                   # seed 1: g - 12 - 1 after 20 s
    e = [M.dry_expected('50x25n30', i, j) for i in INST]
    assert all(x['bestp60'] == P124.dry_policy(P124.DRY_SEL, i, j) - 20 for i, x in zip(INST, e))
    best = [x['bestp60'] if i != 6 else g - 70 for i, x in zip(INST, e)]
    want = np.mean([100 * (b - x['greedy']) / x['greedy'] for b, x in zip(best, e)])
    assert S['comparisons']['mp60_vs_pol']['N']['mean'] == pytest.approx(want, abs=1e-9)
    want = np.mean([100 * (b - x['bestp600']) / x['bestp600'] for b, x in zip(best, e)])
    assert S['comparisons']['mp60_vs_mp600']['N']['mean'] == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize('policy, drop, line', [
    (False, (3, '50x25n30', 'LS', 7), 'without a valid record 3: LS 7'),
    (True, (6, '80x25n30', 'SW', 4), 'without a valid record 6: SW 4')])
def test_missing_record_refuses_and_check_lists_it(tmp_path, capsys, policy, drop, line):
    kw = dict(pol_drop=[drop]) if policy else dict(drop=[drop])
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'a', boot=BOOT, **kw)
    text = M.SETS['policy' if policy else 'rule'][2]
    assert any('fewer than 8 valid 60-s runs' in x and text in x for x in e.value.args[0])
    assert no_outputs(tmp_path / 'a')
    rc, *_ = M.dry_run(tmp_path / 'b', check_only=True, **kw)
    out = capsys.readouterr().out
    assert rc == 1
    assert line in out and 'p137: incomplete' in out
    token = M.POL_TOKEN if policy else M.TOKEN
    other = M.TOKEN if policy else M.POL_TOKEN
    seeds = '1-5,7-8' if policy else '1-2,4-8'
    assert f'p137: {token}: seeds with every cell valid: {seeds}' in out
    assert f'p137: {other}: seeds with every cell valid: 1-8' in out
    rc, *_ = M.dry_run(tmp_path / 'c', check_only=True)
    assert rc == 0 and 'p137: every input present' in capsys.readouterr().out


@pytest.mark.parametrize('policy, field, value, reason', [
    (False, 'cpu_share', 0.5, 'invalid (share 0.5 < 0.9)'),
    (False, 'affinity', [8], 'invalid (affinity [8])'),
    (False, 'threads', 2, 'invalid (threads 2)'),
    (True, 'cpu_share', 0.5, 'invalid (share 0.5 < 0.9)'),
    (True, 'affinity', [3], 'invalid (affinity [3])'),
    (True, 'seed_from', ['or_solution/FAMILY/another_cache'], 'foreign (seed_from=')])
def test_invalid_record_prints_the_planners_reason_and_refuses(tmp_path, capsys, policy, field,
                                                               value, reason):
    def spoil(root):
        p = M.path60(root, '80x25n30', 'N', 3, 5, policy)
        rec = json.load(open(p))
        rec[field] = value
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, before=spoil)
    assert any('records of the 60-s runs rejected' in x for x in e.value.args[0])
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith('p137: rejected')]
    assert len(lines) == 1
    token = M.POL_TOKEN if policy else M.TOKEN
    assert lines[0].startswith(f'p137: rejected {token} seed 5/80x25n30/N/003: {reason}')
    assert no_outputs(tmp_path / 'w')
    rc, *_ = M.dry_run(tmp_path / 'v', check_only=True, before=spoil)
    assert rc == 1


@pytest.mark.parametrize('policy, field, value, match', [
    (False, 'config', {'init_frac': 0.2, 'pop_size': 20}, 'configuration differs'),
    (True, 'config', {'init_frac': 0.2, 'pop_size': 20}, 'configuration differs'),
    (True, 'code_sha256_16', 'another', 'code code_sha256_16')])
def test_another_configuration_refuses(tmp_path, policy, field, value, match):
    def other(root):
        p = M.path60(root, '50x25n30', 'S', 4, 2, policy)
        rec = json.load(open(p))
        rec[field] = value
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, before=other)
    assert any(match in x for x in e.value.args[0]), e.value.args[0]
    assert no_outputs(tmp_path / 'w')


def test_a_start_cache_that_differs_from_the_600s_record_refuses(tmp_path):
    """The planner calls a 60-s record with a start cache other than the plan's foreign (above);
    a 600-s record of the same cell with another start cache makes the comparison refuse."""
    x = dict(record='a60.json', fields=dict(config={'init_frac': 0.1}, code_sha256_16='c',
                                            warm_cache='cache/rule', seed_from=['cache/policy']))
    ref = tmp_path / 'r600.json'
    rec = dict(x['fields'])
    ref.write_text(json.dumps(rec))
    assert M.record_differences(x, dict(record=str(ref))) == []
    for field, value in (('seed_from', ['cache/other']), ('warm_cache', 'cache/other')):
        ref.write_text(json.dumps(dict(rec, **{field: value})))
        bad = M.record_differences(x, dict(record=str(ref)))
        assert len(bad) == 1 and 'start cache' in bad[0] and field in bad[0], bad


def plant(J, size, path, delta):
    """A record hook: add delta to an entry of p124's record."""
    if path in ('memetic_rule', 'memetic_policy'):
        row = J['decision_table']['sizes'][size]['rows']['60']['Setup']
        row['values'][path] = T.f2(float(row['values'][path]) + delta)
    else:
        J['results'][size]['comparisons']['smp_vs_best']['60'][path]['mean'] += delta


@pytest.mark.parametrize('where, delta, match', [
    ('Pool', 0.01, 'smp_vs_best at 60 s'), ('LSW', -0.02, 'smp_vs_best at 60 s'),
    ('memetic_rule', 0.01, 'memetic column started from the rule schedule'),
    ('memetic_policy', 0.01, "memetic column started from the policy's schedule")])
def test_a_difference_from_p124s_record_refuses(tmp_path, where, delta, match):
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, record=lambda J: plant(J, '80x25n30', where, delta))
    assert any(match in x for x in e.value.args[0]), e.value.args[0]
    assert no_outputs(tmp_path / 'w')
    # a rounding difference below the printed digit is no difference
    res, *_ = M.dry_run(tmp_path / 'v', boot=BOOT,
                        record=lambda J: plant(J, '80x25n30', 'Pool', 1e-7))
    assert res['run'] == 'full'


def test_p124_record_of_another_setting_refuses(tmp_path):
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, record=lambda J: J.update(boot=BOOT + 1))
    assert any('boot' in x for x in e.value.args[0])
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'v', boot=BOOT, record=lambda J: J.pop('decision_table'))
    assert any('no decision table' in x for x in e.value.args[0])


def test_twelve_row_count_must_equal_p124s_record(tmp_path):
    def count(J):
        J['decision_table']['counts']['both']['lowest_uses_policy'] = 11
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, record=count)
    assert any('11 of 12 rows whose lowest column uses the policy in the record, 12 of 12 '
               'recomputed' in x for x in e.value.args[0]), e.value.args[0]
    assert no_outputs(tmp_path / 'w')

    def late_bold(J):                       # a row at 600 s whose bold is not its lowest value
        J['decision_table']['sizes']['50x25n30']['rows']['600']['NoSetup']['bold'] = ['rules']
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'v', boot=BOOT, record=late_bold)
    assert any('600 s: bold' in x and 'lowest of its values' in x for x in e.value.args[0])


def test_row_change_on_hand_made_rows():
    base = dict(rules='+3.00', policy_sample='-2.50', memetic_rule='+1.00', cpsat_rule='+2.00',
                memetic_policy='-1.00', cpsat_policy='-0.50')
    below = M.row_change(base, '-3.10')
    assert below['before'] == ['policy_sample'] and below['after'] == ['memetic_rule']
    assert below['below_lowest'] and below['changed'] and below['changed_rule_only']
    above = M.row_change(base, '-1.20')
    assert above['after'] == ['policy_sample'] and not above['below_lowest']
    assert not above['changed']
    tie = M.row_change(base, '-2.50')
    assert tie['after'] == ['policy_sample', 'memetic_rule'] and not tie['below_lowest']
    assert tie['changed'] and tie['lowest_word'] == 'policy samples and memetic from rule'
    other = M.row_change(dict(base, cpsat_policy='-4.00'), '-3.00')
    assert other['before'] == other['after'] == ['cpsat_policy'] and not other['changed']
    # both 60-s values: the second column becomes the lowest, or it does not
    second = M.row_change(base, '-1.20', '-2.80')
    assert second['rule_only_after'] == ['policy_sample'] and second['after'] == ['memetic_policy']
    assert second['changed'] and not second['changed_rule_only'] and second['below_lowest_policy']
    assert second['uses_policy_before'] and second['uses_policy_after']
    assert second['lowest_word'] == 'memetic from policy'
    not_second = M.row_change(base, '-3.10', '-2.80')
    assert not_second['after'] == ['memetic_rule'] and not not_second['uses_policy_after']
    later = M.row_change(base, '-1.20', '+0.50')         # the 60-s value above the 600-s value
    assert later['after'] == ['policy_sample'] and later['after_values']['memetic_policy'] == '+0.50'
    c = M.row_counts({'a': {'NoSetup': below, 'Setup': above}, 'b': {'NoSetup': tie,
                                                                     'Setup': other}})
    assert c == dict(rows=4, changed=2, changed_rule_only=2, below_lowest=1,
                     below_lowest_policy=0, pol_smp_before=3, pol_smp_after=2,
                     pol_smp_after_rule_only=2, uses_policy_before=4, uses_policy_after=3)
    c = M.row_counts({'a': {'NoSetup': second, 'Setup': not_second}})
    assert (c['changed'], c['changed_rule_only'], c['uses_policy_before'],
            c['uses_policy_after'], c['pol_smp_after'], c['pol_smp_after_rule_only']) == \
        (2, 1, 2, 1, 0, 1)


def test_rows_on_the_tree_with_the_sampled_schedules_lowest(tmp_path):
    def sampled_lowest(J):
        for key in ('NoSetup', 'Setup'):
            row = J['decision_table']['sizes']['80x25n30']['rows']['60'][key]
            row['values']['policy_sample'] = '-9.00'
            row['bold'] = T.lowest(row['values'])
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT, record=sampled_lowest)
    rows = res['decision']['rows']
    assert rows['50x25n30']['NoSetup']['after'] == ['cpsat_policy']
    assert rows['50x25n30']['Setup']['after'] == ['memetic_rule']
    r = rows['80x25n30']['NoSetup']           # the 60-s values stay above the sampled schedules
    assert r['before'] == r['rule_only_after'] == r['after'] == ['policy_sample']
    assert not r['changed']
    r = rows['80x25n30']['Setup']             # the 60-s runs from the policy's schedule go below
    assert r['before'] == r['rule_only_after'] == ['policy_sample']
    assert r['after'] == ['memetic_policy'] and r['changed'] and not r['changed_rule_only']
    assert res['decision']['counts'] == dict(
        rows=4, changed=2, changed_rule_only=1, below_lowest=1, below_lowest_policy=1,
        pol_smp_before=2, pol_smp_after=1, pol_smp_after_rule_only=2, uses_policy_before=4,
        uses_policy_after=3)
    text = open(paths['macros']).read()
    for name, value in (('MminRowLowestPolSmpBeforeN', '2'), ('MminRowLowestPolSmpAfterN', '1'),
                        ('MminRuleOnlyRowLowestPolSmpAfterN', '2'),
                        ('MminEightyPolSmpVsPolSixtyNoSetupBold', T.BOLD),
                        ('MminEightyPolSmpVsPolSixtySetupBold', ''),
                        ('MminEightyMemPolStartVsPolSixtySetupBold', T.BOLD),
                        ('MminEightySixtyNoSetupLowest', 'policy samples')):
        assert macro_value(text, name) == value, name
    # a record whose bold columns are not the lowest of its values refuses
    def stale_bold(J):
        sampled_lowest(J)
        J['decision_table']['sizes']['80x25n30']['rows']['60']['Setup']['bold'] = ['cpsat_policy']
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'v', boot=BOOT, record=stale_bold)
    assert any('lowest of its values' in x for x in e.value.args[0])


def test_preview_with_different_seeds_in_the_two_sets(tmp_path, capsys):
    drop = [(3, '80x25n30', 'SW', 9)]
    pol_drop = [(k, '50x25n30', 'L', 4) for k in (1, 2, 3, 4, 5)]
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT, min_seeds=2, drop=drop,
                                     pol_drop=pol_drop)
    assert res['run'] == 'preview' and names is None
    assert res['seeds'] == [1, 2, 4, 5, 6, 7, 8] and res['pol_seeds'] == [6, 7, 8]
    assert res['set_states']['rule']['complete_seeds'] == [1, 2, 4, 5, 6, 7, 8]
    assert res['set_states']['policy']['complete_seeds'] == [6, 7, 8]
    out = capsys.readouterr().out
    assert 'NOT FOR THE PAPER' in out
    assert f'seeds 1-2,4-8 of {M.TOKEN} and seeds 6-8 of {M.POL_TOKEN}' in out
    assert not os.path.exists(paths['macros'])
    assert str(tmp_path / 'w' / 'preview') in paths['json']
    J = json.load(open(paths['json']))
    assert J['warning'].startswith('PREVIEW') and '6-8 of memetic_policy@60' in J['warning']
    md = open(paths['md']).read()
    assert 'PREVIEW' in md.splitlines()[2]
    assert f'{M.POL_TOKEN}, seeds 6-8' in md
    assert M.dry_verify(res) == []            # each side of a comparison on the seeds of its set
    S = res['results']['50x25n30']
    assert S['records']['share_min'] == pytest.approx(0.991)
    assert S['pol_records']['share_min'] == pytest.approx(0.995)
    assert list(S['pol_records']['per_seed']) == ['6', '7', '8']
    # every cell of the set enters, the cell of the dropped records with seeds 6 to 8
    assert S['pol_cells_used'] == 72
    # fewer complete seeds than asked in one set, or seed 1 missing from the rule start, refuse
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'v', boot=BOOT, min_seeds=4, pol_drop=pol_drop)
    assert any('in each set' in x for x in e.value.args[0])
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'u', boot=BOOT, min_seeds=1, drop=[(1, '50x25n30', 'N', 3)])
    assert any('seed 1 among those of' in x for x in e.value.args[0])


def test_options_for_a_preview(tmp_path):
    ap = M.build_parser()
    a = ap.parse_args(['--min-seeds', '2'])
    assert any('needs --out' in x for x in M.check_options(a))
    for folder in ('results', 'manuscript', 'or_solution', 'out'):
        a = ap.parse_args(['--min-seeds', '2', '--out', os.path.join(ROOT, folder, 'x')])
        assert any('lies inside or around' in x for x in M.check_options(a))
    a = ap.parse_args(['--min-seeds', '9', '--out', str(tmp_path / 'x')])
    assert any('between 1 and 8' in x for x in M.check_options(a))
    a = ap.parse_args(['--min-seeds', '2', '--check'])
    assert any('give one' in x for x in M.check_options(a))
    a = ap.parse_args(['--pol-slots', '16-23', 'x-y'])
    assert any("core set 'x-y'" in x for x in M.check_options(a))
    assert ap.parse_args([]).pol_slots == ['16-23', '8-15']


def test_prefix_is_checked(tmp_path):
    own = tmp_path / 'manuscript' / 'macros_memetic_minute.tex'
    own.parent.mkdir()
    lines = ['\\newcommand{\\MminSeedN}{8}  % x']
    assert M.check_names(lines, str(own)) == ['MminSeedN']
    (own.parent / 'other.tex').write_text('\\newcommand{\\MminFoo}{1}\n')
    with pytest.raises(T.Refusal):
        M.check_names(lines, str(own))
    with pytest.raises(T.Refusal):
        M.check_names(['\\newcommand{\\MminA1}{8}'], str(tmp_path / 'x.tex'))
    with pytest.raises(T.Refusal):
        M.check_names(lines * 2, str(tmp_path / 'y' / 'x.tex'))


def test_command_line_dry(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    p = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p137_memetic_one_minute.py'),
                        '--dry', str(tmp_path / 'w'), '--boot', str(BOOT)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    assert 'p137 self-test on synthetic records passed' in p.stdout
    assert 'twelve rows using the policy 12 before, 11 after' in p.stdout


# ----------------------------------------------------------------- the five-minute reading
def test_budget_names_and_the_planners_record_names_at_300(tmp_path):
    """--budget 300 reads memetic_rule@300 and memetic_policy@300 (folders <size>_memetic300s and
    <size>_memetic300s_pactwarm_s<run>) at 300 s, prefix Mfiv, its own outputs and core sets; the
    names of the module return to the one-minute reading after the block."""
    with M.at_budget(300):
        assert (M.BUDGET, M.READ_AT, M.PREFIX) == (300, 300, 'Mfiv')
        assert (M.TOKEN, M.POL_TOKEN) == ('memetic_rule@300', 'memetic_policy@300')
        assert M.OUT_FILES == {'macros': 'manuscript/macros_memetic_five.tex',
                               'json': 'results/memetic_five_minute.json',
                               'md': 'results/memetic_five_minute.md'}
        assert M.TABLE_KEY == {'memetic_rule': 'm300_vs_pol', 'memetic_policy': 'mp300_vs_pol'}
        assert [c[1] for c in M.COMPARISONS][:2] == ['MThreeHundredVsMSixHundred',
                                                     'OneThreeHundredVsOneSixHundred']
        ap = M.build_parser()
        a = ap.parse_args([])
        assert (a.budget, a.slots, a.pol_slots) == (300, ['16-23'], ['8-15'])
        for k in (1, 8):
            pa = T.build_parser().parse_args(['--result-root', str(tmp_path), '--memetic-seed',
                                              str(k)])
            for token, folder, policy in (
                    ('memetic_rule@300', '50x25n30_memetic300s', False),
                    ('memetic_policy@300', '50x25n30_memetic300s_pactwarm_s304', True)):
                camp = T.make_campaign(['50x25n30'], [f'{token}:3-4'], pa, 304, 'lns')
                c = next(c for seg in camp.segments for c in camp.cells(seg))
                assert camp.record(c) == os.path.join(str(tmp_path), 'or_solution', 'FAMILY',
                                                      folder, c.combo,
                                                      f'instance_{c.idx:03d}_seed{k}.json')
                assert camp.expected_fields(c)['budget_s'] == 300.0
            camp = T.make_campaign(['50x25n30'], ['memetic_policy@300:3-4'], pa, P124.DRY_SEL,
                                   'lns')
            c = next(c for seg in camp.segments for c in camp.cells(seg))
            assert camp.record(c) == M.path60(str(tmp_path), c.size, c.combo, c.idx, k, policy=True)
    assert (M.BUDGET, M.PREFIX, M.TOKEN) == (60, 'Mmin', 'memetic_rule@60')
    assert M.build_parser().parse_args([]).budget == 60
    a = M.build_parser().parse_args(['--budget', '300'])
    assert any('--budget 300' in x for x in M.check_options(a))     # the module reads 60 s


def five(tmp_path, name='w', **kw):
    """The five-minute reading of a dry tree with the records of both budgets."""
    with M.at_budget(300):
        return M.dry_run(tmp_path / name, boot=BOOT, **kw)


def test_five_minute_numbers_equal_the_built_ones(tmp_path):
    res, names, paths, _ = five(tmp_path)
    with M.at_budget(300):
        assert res['run'] == 'full' and M.dry_verify(res) == []
        assert res['decision']['counts'] == M.DRY_COUNTS
        e = [M.dry_expected('80x25n30', i, 0) for i in INST]
        # the step at 200 s of search counts at 300 s; the step at DRY_LATE_S (294 s) of the
        # runs from the policy's schedule does not (o_pol clock)
        assert all(x['best60'] == P124.r0(i, 0) - 68 for i, x in zip(INST, e))
        assert all(x['bestp60'] == P124.dry_policy(P124.DRY_SEL, i, 0) - 22
                   for i, x in zip(INST, e))
        assert M.DRY_LATE_S == 294.0
    assert (res['budget_s'], res['read_at_s'], res['ref_budget_s']) == (300, 300, 600)
    assert res['init_phase_s'] == pytest.approx(30.0)
    assert res['cells_valid'] == res['cells_total'] == 2304
    assert 'all_rows' not in res['decision']               # the twelve rows are p138's
    for p in paths.values():
        assert os.path.exists(p) and ('five' in os.path.basename(p))
    assert not os.path.exists(os.path.join(tmp_path, 'w', 'out', 'results',
                                           'memetic_one_minute.json'))
    S = res['results']['50x25n30']
    assert S['records']['cores']['1'] == {'16-23': 72}
    assert S['pol_records']['cores']['1'] == S['pol_records']['cores']['8'] == {'8-15': 72}
    assert set(S['comparisons']) == {
        'm300_vs_m600', 'one300_vs_one600', 'smp_vs_m300', 'pol_vs_m300', 'cp_vs_m300',
        'm300_vs_pol', 'mp300_vs_mp600', 'smp_vs_mp300', 'mp300_vs_m300', 'mp300_vs_pol'}
    for key in ('smp_vs_m300', 'smp_vs_mp300'):
        assert S['verdicts'][key]['held']                    # the policy record ends at 120 s
    J = json.load(open(os.path.join(tmp_path, 'w', 'out', 'results', 'memetic_parallel.json')))
    rec = J['results']['50x25n30']['comparisons']['smp_vs_best']['300']
    assert M.printed_entry(S['p124_smp_vs_best']['Pool']) == M.printed_entry(rec['Pool'])
    text = open(paths['macros']).read()
    assert names and all(re.fullmatch(r'Mfiv[A-Za-z]+', x) for x in names)
    assert len(names) == len(set(names)) and 'MfivBothDecisionRowN' not in names
    for name, value in (('MfivSeedN', '8'), ('MfivBudgetS', '300'), ('MfivInitPhaseS', '30'),
                        ('MfivRefBudgetS', '600'), ('MfivRefInitPhaseS', '60'),
                        ('MfivCellN', '1152'), ('MfivPolStartCellValidN', '1152'),
                        ('MfivRowN', '4'), ('MfivRowLowestChangeN', '3'),
                        ('MfivRuleOnlyRowLowestChangeN', '2'), ('MfivRowBelowLowestN', '2'),
                        ('MfivRowUsesPolicyBeforeN', '4'), ('MfivRowUsesPolicyAfterN', '2')):
        assert macro_value(text, name) == value, name
    for sw in ('Fifty', 'Eighty'):
        for suffix in ('Pool', 'PoolLo', 'PoolHi', 'PoolDir', 'NoSetup', 'LSW'):
            assert f'Mfiv{sw}MThreeHundredVsMSixHundred{suffix}' in names
            assert f'Mfiv{sw}MpThreeHundredVsMSixHundred{suffix}' not in names
            assert f'Mfiv{sw}MpThreeHundredVsMThreeHundred{suffix}' in names
        for g in ('NoSetup', 'Setup'):
            assert macro_value(text, f'Mfiv{sw}MemRuleStartVsPolThreeHundred{g}Sel') == \
                macro_value(text, f'Mfiv{sw}MThreeHundredVsPol{g}')
            assert macro_value(text, f'Mfiv{sw}MemPolStartVsPolThreeHundred{g}') == \
                macro_value(text, f'Mfiv{sw}MpThreeHundredVsPol{g}')
    assert all(' % ' in line for line in text.splitlines() if 'newcommand' in line)
    assert '60-s' not in text and 'one-minute' not in text
    md = open(paths['md']).read()
    assert md.startswith('# The memetic algorithm with a budget of 300 s, read 300 s after')
    assert "Spread over the 300-s runs started from the policy's schedule" in md
    assert 'Over the 12 rows' not in md


def test_five_minute_rows_before_and_after(tmp_path):
    res, names, paths, _ = five(tmp_path)
    rows = res['decision']['rows']
    for by in rows.values():
        for r in by.values():
            assert r['before'] == ['cpsat_policy'] and set(r) >= {'m300', 'mp300'}
            assert 'm60' not in r
    assert rows['50x25n30']['NoSetup']['after'] == ['cpsat_policy']
    r = rows['50x25n30']['Setup']         # the 300-s runs from the policy's schedule go below
    assert r['rule_only_after'] == ['cpsat_policy'] and r['after'] == ['memetic_policy']
    assert r['changed'] and not r['changed_rule_only'] and r['below_lowest_policy']
    for key in ('NoSetup', 'Setup'):     # the 300-s runs from the rule schedule go below
        r = rows['80x25n30'][key]
        assert r['rule_only_after'] == r['after'] == ['memetic_rule'] and r['below_lowest']
    text = open(paths['macros']).read()
    want = {('Fifty', 'NoSetup'): 'cpsat_policy', ('Fifty', 'Setup'): 'memetic_policy',
            ('Eighty', 'NoSetup'): 'memetic_rule', ('Eighty', 'Setup'): 'memetic_rule'}
    for (sw, g), col in want.items():
        names_row = {c: tuple('Mfiv' + x[len('Tmd'):] for x in v)
                     for c, v in T.decision_names(sw, 'ThreeHundred', g).items()}
        for c, (_, bold) in names_row.items():
            assert macro_value(text, bold) == (T.BOLD if c == col else ''), (sw, g, c)
        assert macro_value(text, f'Mfiv{sw}ThreeHundred{g}Lowest') == M.LOWEST_WORD[col]
    # the 600-s columns of the rows are p124's at 300 s
    J = json.load(open(os.path.join(tmp_path, 'w', 'out', 'results', 'memetic_parallel.json')))
    for size, by in rows.items():
        for key, r in by.items():
            assert r['values'] == J['decision_table']['sizes'][size]['rows']['300'][key]['values']


def test_each_reading_reads_the_runs_of_its_own_budget(tmp_path):
    """A step planted in a 300-s record changes the five-minute reading and not the one-minute
    reading of the same tree; a step at search time 296 s of a run from the policy's schedule
    (request time 305 s on the o_pol clock) is not read at 300 s."""
    j = M.COMBOS.index('W')

    def plant(root):
        with M.at_budget(300):
            for policy, t, dv in ((False, 250.0, 90), (True, 296.0, 300)):
                p = M.path60(root, '50x25n30', 'W', 5, 3, policy=policy)
                rec = json.load(open(p))
                rec['anytime'] = sorted(rec['anytime'] + [[t, t + 1.0, t + 1.0,
                                                           rec['makespan'] - dv]])
                json.dump(rec, open(p, 'w'))
    res60, _, paths60, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    text60 = open(paths60['json']).read()
    res, *_ = five(tmp_path, build=False, before=plant)
    res60b, _, paths60b, _ = M.dry_run(tmp_path / 'w', boot=BOOT, build=False)
    assert open(paths60b['json']).read() == text60            # the 60-s reading is unchanged
    S = res['results']['50x25n30']
    cell = next(c for c in S['spread']['cells'] if (c['combination'], c['instance']) == ('W', 5))
    R, g = P124.r0(5, j), P124.dry_policy(P124.DRY_SEL, 5, j)
    # seed 3 ends its 300-s run at R - 21; the planted step at 250 s of search counts at 300 s
    assert cell['smallest'] == R - 21 - 90 and cell['largest'] == R - 19
    pcell = next(c for c in S['pol_spread']['cells']
                 if (c['combination'], c['instance']) == ('W', 5))
    assert pcell['smallest'] == g - 22                   # the step at 296 s is not read
    with M.at_budget(300):
        e = [M.dry_expected('50x25n30', i, j) for i in INST]
    best = [x['best60'] if i != 5 else R - 111 for i, x in zip(INST, e)]
    want = np.mean([100 * (b - x['greedy']) / x['greedy'] for b, x in zip(best, e)])
    assert S['comparisons']['m300_vs_pol']['W']['mean'] == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize('policy, drop, line', [
    (False, (2, '80x25n30', 'L', 9), 'without a valid record 2: L 9'),
    (True, (7, '50x25n30', 'LS', 3), 'without a valid record 7: LS 3')])
def test_five_minute_missing_record_refuses_and_check_lists_it(tmp_path, capsys, policy, drop,
                                                               line):
    kw = dict(pol_drop=[drop]) if policy else dict(drop=[drop])
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'a', **kw)
    with M.at_budget(300):
        text = M.SETS['policy' if policy else 'rule'][2]
        assert any('fewer than 8 valid 300-s runs' in x and text in x for x in e.value.args[0])
        assert no_outputs(tmp_path / 'a')
    rc, *_ = five(tmp_path, 'b', check_only=True, **kw)
    out = capsys.readouterr().out
    assert rc == 1 and line in out and 'p137: incomplete' in out
    token = 'memetic_policy@300' if policy else 'memetic_rule@300'
    assert f'p137: {token}: seeds with every cell valid: ' in out


def test_five_minute_invalid_or_foreign_record_refuses(tmp_path, capsys):
    def spoil(root):
        with M.at_budget(300):
            p = M.path60(root, '80x25n30', 'N', 3, 5, True)
        rec = json.load(open(p))
        rec['affinity'] = [16]                         # outside the core set 8-15 of the set
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'w', before=spoil)
    assert any('records of the 300-s runs rejected' in x for x in e.value.args[0])
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith('p137: rejected')]
    assert len(lines) == 1 and lines[0].startswith(
        'p137: rejected memetic_policy@300 seed 5/80x25n30/N/003: invalid (affinity [16])')

    def other(root):
        with M.at_budget(300):
            p = M.path60(root, '50x25n30', 'S', 4, 2)
        rec = json.load(open(p))
        rec['config'] = {'init_frac': 0.2, 'pop_size': 20}
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'v', before=other)
    assert any('configuration differs' in x for x in e.value.args[0])


def test_five_minute_difference_from_p124s_record_refuses(tmp_path):
    def planted(J):
        row = J['decision_table']['sizes']['80x25n30']['rows']['300']['NoSetup']
        row['values']['memetic_policy'] = T.f2(float(row['values']['memetic_policy']) + 0.01)
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'w', record=planted)
    assert any("300 s: memetic column started from the policy's schedule" in x
               for x in e.value.args[0]), e.value.args[0]
    with M.at_budget(300):
        assert no_outputs(tmp_path / 'w')

    def smp(J):
        J['results']['50x25n30']['comparisons']['smp_vs_best']['300']['Pool']['mean'] += 0.01
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'v', record=smp)
    assert any('smp_vs_best at 300 s' in x for x in e.value.args[0])


def test_five_minute_preview(tmp_path, capsys):
    pol_drop = [(k, '80x25n30', 'S', 6) for k in (1, 2)]
    res, names, paths, _ = five(tmp_path, min_seeds=3, pol_drop=pol_drop)
    assert res['run'] == 'preview' and names is None
    assert res['seeds'] == list(range(1, 9)) and res['pol_seeds'] == list(range(3, 9))
    out = capsys.readouterr().out
    assert 'NOT FOR THE PAPER' in out and 'seeds 3-8 of memetic_policy@300' in out
    assert not os.path.exists(paths['macros'])
    J = json.load(open(paths['json']))
    assert J['warning'].startswith('PREVIEW') and J['budget_s'] == 300
    assert 'PREVIEW' in open(paths['md']).read().splitlines()[2]
    with M.at_budget(300):
        assert M.dry_verify(res) == []
    with pytest.raises(T.Refusal) as e:
        five(tmp_path, 'v', min_seeds=1, drop=[(1, '50x25n30', 'N', 3)])
    assert any('seed 1 among those of memetic_rule@300' in x for x in e.value.args[0])


def test_macro_names_of_both_readings_are_unique(tmp_path):
    """The two macro files of the dry tree: prefixes Mmin and Mfiv, no name in both, each checked
    against the other (and every macro file of the repository) by check_names."""
    res60, names60, paths60, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    res, names, paths, _ = five(tmp_path, build=False)
    assert not set(names60) & set(names)
    assert os.path.dirname(paths60['macros']) == os.path.dirname(paths['macros'])
    lines = open(paths['macros']).read().splitlines()
    with M.at_budget(300):
        assert M.check_names(lines, paths['macros']) == names
        (tmp_path / 'w' / 'out' / 'manuscript' / 'stray.tex').write_text(
            '\\newcommand{\\MfivSeedN}{8}\n')
        with pytest.raises(T.Refusal) as e:
            M.check_names(lines, paths['macros'])
        assert 'the prefix Mfiv is used by \\MfivSeedN' in e.value.args[0][0]


def test_command_line_dry_at_300(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    p = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p137_memetic_one_minute.py'),
                        '--budget', '300', '--dry', str(tmp_path / 'w'), '--boot', str(BOOT)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert p.returncode == 0, p.stderr[-2000:]
    assert 'p137 self-test on synthetic records passed' in p.stdout
    assert 'pooled at 300 s: best of the 300-s runs minus best of the 600-s runs' in p.stdout
    assert 'twelve rows' not in p.stdout
