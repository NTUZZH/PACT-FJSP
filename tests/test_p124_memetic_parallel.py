"""scripts/p124_memetic_parallel.py: eight memetic runs read as one parallel method.

  * the record name of a seed is the planner's (Campaign.record with --memetic-seed k);
  * on a synthetic tree (descriptors, arrays, eight seeds of memetic records, CP-SAT records,
    rule and policy records) every difference equals the one built into the tree, the best of
    eight is never longer than seed 1, the spread is the built one, and every macro name is
    Mpar plus letters;
  * a cell with fewer than eight valid runs: the run refuses and leaves no output, --check
    lists the cell and returns 1, a rejected record prints one line with the planner's reason;
  * --min-seeds below eight computes, warns that the output is not for the paper, writes no
    macro file and needs --out-root outside the repository's result and paper folders;
  * the prefix Mpar is refused once another .tex file defines a macro with it;
  * the second set (eight runs started from the policy's schedule): the record name is the
    planner's; with every seed complete its differences equal the built ones and come after
    the first set's outputs, which stay as they are; while it is incomplete the three outputs
    are those without it, byte for byte, and the run and --check print the valid runs per seed;
    a seed whose every cell has a record of which one is rejected makes the run refuse and
    --check exit 1; a policy start schedule that differs from the policy array is refused;
  * the decision table on eight cores: values and bold markers on a hand-made case with a known
    lowest column per row and a tie; the counts; p94's four columns checked against its record
    (a planted difference refuses and leaves no output); the table file's macro names are
    written; the curves file has p94's columns, its own method names, n_runs 8 and the built
    best of eight; the curve rows equal those of p94's analyse_request on the same series;
  * --preview-first-set: the first set alone, marked in every output, no macro file, refused
    with an output folder inside the repository.

Run from the repository root on one core:  python -B -m pytest -q tests/test_p124_memetic_parallel.py
CPU only; every file is written under the pytest temporary folder.
"""
import csv
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

import p124_memetic_parallel as M  # noqa: E402

T = M.T
BOOT = 200
PY = sys.executable


def test_record_name_is_the_planners(tmp_path):
    for k in (1, 3, 8):
        a = T.build_parser().parse_args(['--result-root', str(tmp_path), '--memetic-seed', str(k)])
        camp = T.make_campaign(['50x25n30'], ['memetic_rule@600:3-4'], a, 304, 'lns')
        for seg in camp.segments:
            for c in camp.cells(seg):
                assert camp.record(c) == M.memetic_path(str(tmp_path), c.size, c.combo, c.idx, k)
                assert camp.expected_fields(c)['seed'] == k


def test_dry_numbers_equal_the_built_ones(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    assert res['run'] == 'full' and M.dry_verify(res) == []
    assert names and all(n.startswith('Mpar') and n.isalpha() for n in names)
    for p in paths.values():
        assert os.path.exists(p)
    S = res['results']['50x25n30']
    assert S['cells_used'] == 72
    assert S['core_sets']['1'] == {'16-23': 72} and S['core_sets']['8'] == {'8-15': 72}
    held = S['comparisons']['smp_vs_best']
    assert not held['60']['Pool']['held'] and held['300']['Pool']['held']
    v = S['verdicts']['pol_vs_best']
    assert sum(v['counts'].values()) == 8
    text = open(paths['macros']).read()
    assert '\\newcommand{\\MparFiftyPolVsBestSixHundredPool}' in text
    assert '\\newcommand{\\MparEightyBestVsOneSixtyPool}' in text
    assert 'MparFiftyPolVsBestSixtyN}' not in text          # combinations at the stated budget only


def test_incomplete_cell_refuses_and_check_lists_it(tmp_path, capsys):
    drop = [(3, '50x25n30', 'LS', 7)]
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'a', boot=BOOT, drop=drop)
    assert any('fewer than 8 valid runs' in x for x in e.value.args[0])
    assert not os.path.exists(tmp_path / 'a' / 'out' / 'results' / 'memetic_parallel.json')
    assert not os.path.exists(tmp_path / 'a' / 'out' / 'manuscript' / 'macros_memetic_parallel.tex')
    rc, *_ = M.dry_run(tmp_path / 'b', drop=drop, check_only=True)
    out = capsys.readouterr().out
    assert rc == 1
    assert 'without a valid record 3: LS 7' in out and 'p124: incomplete' in out
    rc, *_ = M.dry_run(tmp_path / 'c', check_only=True)
    assert rc == 0 and 'every input present' in capsys.readouterr().out


def test_rejected_record_prints_the_planners_reason(tmp_path, capsys):
    def low_share(root):
        p = M.memetic_path(root, '80x25n30', 'N', 3, 5)
        rec = json.load(open(p))
        rec['cpu_share'] = 0.5
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal):
        M.dry_run(tmp_path / 'w', boot=BOOT, before=low_share)
    lines = [x for x in capsys.readouterr().out.splitlines() if 'rejected' in x]
    assert len(lines) == 1
    assert lines[0].startswith('p124: rejected seed 5/80x25n30/N/003: invalid (share 0.5 < 0.9)')


def test_partial_writes_no_macros(tmp_path, capsys):
    drop = [(2, '50x25n30', 'N', 3), (6, '50x25n30', 'N', 3)]
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT, min_seeds=6, drop=drop)
    assert res['run'] == 'partial' and names is None
    assert 'NOT FOR THE PAPER' in capsys.readouterr().out
    assert not os.path.exists(paths['macros']) and os.path.exists(paths['json'])
    assert str(tmp_path / 'w' / 'partial') in paths['json']
    S = res['results']['50x25n30']
    assert S['cells_used'] == 72 and S['valid_seeds_per_cell']['6'] == 1
    cell = next(c for c in S['spread']['cells'] if (c['combination'], c['instance']) == ('N', 3))
    assert cell['seeds'] == 6


def test_options_for_partial_output():
    ap = M.build_parser()
    bad = M.check_options(ap.parse_args(['--min-seeds', '3']))
    assert any('needs --out-root' in x for x in bad)
    for folder in ('results', 'manuscript', 'or_solution'):
        bad = M.check_options(ap.parse_args(['--min-seeds', '3', '--out-root',
                                             os.path.join(ROOT, folder, 'x')]))
        assert any('lies inside or around' in x for x in bad)
    assert any('seed 1' in x for x in M.check_options(ap.parse_args(['--seeds', '2-8'])))
    assert any('--primary' in x for x in M.check_options(ap.parse_args(['--budgets', '60', '300'])))


def test_prefix_is_checked(tmp_path):
    own = tmp_path / 'manuscript' / 'macros_memetic_parallel.tex'
    own.parent.mkdir()
    lines = ['\\newcommand{\\MparSeedN}{8}  % x']
    assert M.check_names(lines, str(own)) == ['MparSeedN']
    (own.parent / 'other.tex').write_text('\\newcommand{\\MparFoo}{1}\n')
    with pytest.raises(T.Refusal):
        M.check_names(lines, str(own))
    with pytest.raises(T.Refusal):
        M.check_names(['\\newcommand{\\MparA1}{8}'], str(tmp_path / 'x.tex'))


def test_command_line_dry(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    p = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p124_memetic_parallel.py'),
                        '--dry', str(tmp_path / 'w'), '--boot', str(BOOT)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert 'self-test on synthetic records passed' in p.stdout


# ----------------------------------------------------------------- the second set

def outputs(paths, tree):
    """The text of the outputs that exist, the folder of the synthetic tree replaced by TREE."""
    return {k: open(p).read().replace(str(tree), 'TREE') for k, p in paths.items()
            if os.path.exists(p)}


def low_share_at(size, combo, idx, seed):
    """A before-hook: the record of the second set at this cell gets cpu_share 0.5."""
    def hook(root):
        p = M.memetic_path(root, size, combo, idx, seed, run=M.DRY_SEL)
        rec = json.load(open(p))
        rec['cpu_share'] = 0.5
        json.dump(rec, open(p, 'w'))
    return hook


def test_policy_record_name_is_the_planners(tmp_path):
    for k in (1, 4, 8):
        a = T.build_parser().parse_args(['--result-root', str(tmp_path), '--memetic-seed', str(k)])
        camp = T.make_campaign(['80x25n30'], ['memetic_policy@600:3-4'], a, 304, 'lns')
        for seg in camp.segments:
            for c in camp.cells(seg):
                assert camp.record(c) == M.memetic_path(str(tmp_path), c.size, c.combo, c.idx, k,
                                                        run=304)
                e = camp.expected_fields(c)
                assert e['seed'] == k
                assert e['seed_from'] == ['or_solution/FAMILY/pactwarm_cache_xmask_s304_80x25n30']


def test_second_set_numbers_equal_the_built_ones(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    assert M.dry_verify(res) == []
    p = res['policy_start']
    assert p['state'] == 'complete' and p['rejected'] == [] and p['missing'] == []
    assert p['counts']['50x25n30']['1'] == dict(valid=72, rejected=0, missing=0,
                                                cores={'8-15': 72})
    assert p['counts']['80x25n30']['8']['cores'] == {'16-23': 72}
    S = p['results']['50x25n30']
    assert S['cells_used'] == 72
    # built: the runs started from the policy's schedule are shorter at every budget
    assert all(S['comparisons']['bestp_vs_best'][str(b)]['Pool']['hi'] < 0 for b in res['budgets'])
    assert S['comparisons']['smp_vs_bestp']['300']['Pool']['held']
    text = open(paths['macros']).read()
    for name in ('MparFiftyPolStartCellN', 'MparFiftyBestPolStartVsBestSixHundredPool',
                 'MparEightyBestVsPolSixtyNoSetup', 'MparEightyBestPolStartVsPolThreeHundredSetupDir',
                 'MparFiftyPolSmpVsBestPolStartSixHundredLSWShorter',
                 'MparFiftyCpPolStartVsBestPolStartSixtyPoolLo'):
        assert f'\\newcommand{{\\{name}}}' in text
    assert 'MparFiftyBestVsPolSixHundredN}' not in text           # decision form: pooled only
    assert 'MparFiftyCpPolStartVsBestPolStartSixtyN}' not in text  # combinations at 600 s only
    assert all(n.startswith('Mpar') and n.isalpha() for n in names)
    # every comparison macro of the second set names what, the instances, the runs, the clock
    words = [c[1] for c in M.COMPARISONS_POLICY]
    new = [x for x in text.splitlines()
           if any(f'{{\\Mpar{sw}{w}' in x for sw in ('Fifty', 'Eighty') for w in words)]
    assert len(new) == 2 * (119 + 68 + 119 + 36 + 36)
    assert all('instances 3-11' in x and 'request clock' in x and 'of memetic_' in x for x in new)
    # the first set's outputs come first, as they are without the second set
    _, _, base, _ = M.dry_run(tmp_path / 'none', boot=BOOT, policy_seeds=())
    old, now = outputs(base, tmp_path / 'none'), outputs(paths, tmp_path / 'w')
    assert now['macros'].startswith(old['macros'].rstrip('\n'))
    assert now['md'].startswith(old['md'].rstrip('\n'))
    assert '## Second set' in now['md'] and '## Second set' not in old['md']
    a, b = json.loads(old['json']), json.loads(now['json'])
    assert set(b) - set(a) == {'policy_start', 'decision_table'} and all(b[k] == a[k] for k in a)
    assert list(b)[-2:] == ['policy_start', 'decision_table']
    assert 'curves' not in old and 'curves' in now


def test_incomplete_second_set_leaves_the_outputs_unchanged(tmp_path, capsys):
    _, _, base, _ = M.dry_run(tmp_path / 'none', boot=BOOT, policy_seeds=())
    want = outputs(base, tmp_path / 'none')
    assert 'PolStart' not in want['macros'] and 'policy_start' not in json.loads(want['json'])
    assert set(want) == {'macros', 'json', 'md'} and 'Decision' not in want['macros']
    capsys.readouterr()
    # seeds 1-3 only; every seed but one record, and a rejected record in that same seed (a seed
    # with a cell still to run is incomplete, not refused)
    for name, kw in (('part', dict(policy_seeds=(1, 2, 3))),
                     ('drop', dict(policy_drop=[(4, '80x25n30', 'SW', 9)],
                                   before=low_share_at('50x25n30', 'L', 5, 4)))):
        res, _, paths, _ = M.dry_run(tmp_path / name, boot=BOOT, **kw)
        assert 'policy_start' not in res and 'decision_table' not in res
        assert outputs(paths, tmp_path / name) == want
        assert not os.path.exists(paths['curves'])
    out = capsys.readouterr().out
    assert out.count("p124: second set, eight runs started from the policy's schedule "
                     '(memetic_policy@600, seeds 1-8, run s302): not complete; the outputs are '
                     'those of the first set alone') == 2
    assert ('p124:   second set 80x25n30: valid runs per seed (of 72 cells): seed 1 72 (8-15 72), '
            'seed 2 72 (8-15 72), seed 3 72 (8-15 72), seed 4 0, seed 5 0') in out
    assert 'seed 4 71 (8-15 71; 1 rejected)' in out
    assert ('p124: second set: rejected memetic_policy seed 4/50x25n30/L/005: invalid '
            '(share 0.5 < 0.9)') in out
    # --check: the first set decides the exit code while the second set is incomplete
    rc, *_ = M.dry_run(tmp_path / 'chk', check_only=True, policy_drop=[(4, '80x25n30', 'SW', 9)])
    out = capsys.readouterr().out
    assert rc == 0 and out.rstrip().endswith('p124: every input present')
    assert 'not complete' in out and 'seed 4 71 (8-15 71)' in out


def test_second_set_partly_invalid_refuses(tmp_path, capsys):
    hook = low_share_at('80x25n30', 'N', 3, 5)
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, before=hook)
    assert any(x.startswith('second set: seed 5: every one of its 144 cells has a record, 1 of '
                            'them not usable') for x in e.value.args[0])
    lines = [x for x in capsys.readouterr().out.splitlines()
             if x.startswith('p124: second set: rejected')]
    assert len(lines) == 1
    assert lines[0].startswith('p124: second set: rejected memetic_policy seed 5/80x25n30/N/003: '
                               'invalid (share 0.5 < 0.9)')
    assert not os.path.exists(tmp_path / 'w' / 'out' / 'results' / 'memetic_parallel.json')
    assert not os.path.exists(tmp_path / 'w' / 'out' / 'manuscript' / 'macros_memetic_parallel.tex')
    rc, *_ = M.dry_run(tmp_path / 'c', before=hook, check_only=True)
    out = capsys.readouterr().out
    assert rc == 1 and '): REFUSED' in out
    assert out.rstrip().endswith('the second set is not usable')


def test_second_set_policy_start_is_checked(tmp_path):
    def other_start(root):
        p = M.memetic_path(root, '50x25n30', 'S', 6, 2, run=M.DRY_SEL)
        rec = json.load(open(p))
        rec['seeds'][1]['makespan'] += 1.0
        json.dump(rec, open(p, 'w'))
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, before=other_start)
    assert any('instance_006_seed2.json: policy start schedule' in x for x in e.value.args[0])
    assert not os.path.exists(tmp_path / 'w' / 'out' / 'results' / 'memetic_parallel.json')


# ----------------------------------------------------------------- decision table, eight cores

def hand_made():
    """Options and arrays of one size with two instances: every cell of a column holds one value
    per budget (60, 300, 600 s), in percent of the greedy schedule 100. Rows: at 60 s the memetic
    algorithm started from the policy's schedule is lowest, at 300 s the memetic algorithm
    started from the rule schedule, at 600 s without setups the two searches started from the
    rule schedule tie, at 600 s with setups (and pooled) CP-SAT started from the policy's
    schedule is lowest."""
    a = M.build_parser().parse_args(['--instances', '3-4', '--boot', str(BOOT)])
    assert M.check_options(a) == []
    n, C = len(a.inst), len(M.COMBOS)
    setup = np.array([r in T.GROUPS[1][2] for r in M.COMBOS])

    def cols(*v):
        return np.stack([np.full((n, C), x, dtype=float) for x in v])
    cpp = cols(97.5, 95.5, 94.5)
    cpp[2][:, setup] = 93.0
    D = dict(rule=np.full((n, C), 104.0), pol_sel=np.full((n, C), 100.0),
             o_rule=np.full((n, C), 2.0), o_pol=np.full((n, C), 8.0), smp=cols(98, 98, 98),
             sp=np.ones((n, C), bool), CP=cols(99, 97, 94), CPv=np.ones((n, C), bool),
             policy_budget_s=120.0)
    use = np.ones((n, C), bool)
    return a, D, cols(101, 95, 94), use, cols(97, 96, 95), (cpp, use)


def test_decision_table_values_bold_and_tie():
    a, D, best, use, bestp, cp = hand_made()
    S = M.decision_size(a, D, best, use, bestp, use, cp)
    rows = S['decision']
    assert list(rows) == ['60', '300', '600'] and list(rows['60']) == ['Pool', 'NoSetup', 'Setup']
    for key in ('Pool', 'NoSetup', 'Setup'):
        assert rows['60'][key]['values'] == dict(
            rules='+4.00', policy_sample='-2.00', memetic_rule='+1.00', cpsat_rule='-1.00',
            memetic_policy='-3.00', cpsat_policy='-2.50')
        assert rows['60'][key]['bold'] == ['memetic_policy']
        assert rows['300'][key]['bold'] == ['memetic_rule']
        assert rows['300'][key]['same_instances'] is True
    assert rows['600']['NoSetup']['bold'] == ['memetic_rule', 'cpsat_rule']        # the tie
    assert rows['600']['Setup']['values']['cpsat_policy'] == '-7.00'
    assert rows['600']['Setup']['bold'] == ['cpsat_policy']
    assert rows['600']['Pool']['values']['cpsat_policy'] == '-6.25'
    assert rows['600']['Pool']['bold'] == ['cpsat_policy']
    assert T.decision_mismatches({'50x25n30': dict(decision=rows)}) == []
    # without the runs started from the policy's schedule: no value there, bold over five
    S1 = M.decision_size(a, D, best, use, None, None, cp)
    r1 = S1['decision']['60']['NoSetup']
    assert r1['values']['memetic_policy'] is None and r1['bold'] == ['cpsat_policy']
    # a column over other cells: p94's check of the row fires
    fewer = use.copy()
    fewer[0, 0] = False
    S2 = M.decision_size(a, D, best, fewer, bestp, use, cp)
    bad = T.decision_mismatches({'50x25n30': dict(decision=S2['decision'])})
    assert bad and all('not means over one set of instances' in x for x in bad)


def test_decision_counts():
    a, D, best, use, bestp, cp = hand_made()
    rows = M.decision_size(a, D, best, use, bestp, use, cp)['decision']
    c = M.decision_counts({'50x25n30': rows, '80x25n30': rows}, a.budgets)
    one = c['50x25n30']
    assert one['rows'] == 6 and one['ties'] == 1 and one['without_number'] == 0
    assert one['lowest'] == dict(rules=0, policy_sample=0, memetic_rule=3, cpsat_rule=1,
                                 memetic_policy=2, cpsat_policy=1)
    assert one['late_rows'] == 4 and one['late_lowest_policy_start'] == 1
    assert one['lowest_uses_policy'] == 3
    both = c['both']
    assert both['rows'] == 12 and both['lowest']['memetic_rule'] == 6 and both['ties'] == 2
    assert both['late_lowest_policy_start'] == 2 and both['lowest_uses_policy'] == 6
    assert set(c) == {'50x25n30', '80x25n30', 'both'}


def test_p94_columns_are_checked():
    a, D, best, use, bestp, cp = hand_made()
    rows = M.decision_size(a, D, best, use, bestp, use, cp)['decision']
    J = dict(run='full', origin='request', selection=dict(seed=7), results={'50x25n30': dict(
        decision={b: {k: dict(values=dict(r['values'])) for k, r in br.items()}
                  for b, br in rows.items()})})
    tables = {'50x25n30': rows}
    assert M.p94_check(a, 7, tables, J, 'p94.json') == []
    J['results']['50x25n30']['decision']['600']['Setup']['values']['memetic_rule'] = '+9.99'
    assert M.p94_check(a, 7, tables, J, 'p94.json') == []          # replaced by the best of eight
    J['results']['50x25n30']['decision']['300']['NoSetup']['values']['policy_sample'] = '-2.01'
    bad = M.p94_check(a, 7, tables, J, 'p94.json')
    assert bad == ['50x25n30, 300 s, combinations without setups, policy_sample: -2.00 here, '
                   '-2.01 in p94.json']
    assert any('selected run' in x for x in M.p94_check(a, 8, tables, J, 'p94.json'))


def test_planted_difference_from_p94_refuses(tmp_path):
    def plant(rec):
        rec['results']['80x25n30']['decision']['600']['Setup']['values']['cpsat_rule'] = '+0.00'
    with pytest.raises(T.Refusal) as e:
        M.dry_run(tmp_path / 'w', boot=BOOT, timed=plant)
    assert any(x.startswith('80x25n30, 600 s, combinations with setups, cpsat_rule:')
               for x in e.value.args[0])
    for f in ('results/memetic_parallel.json', 'results/memetic_parallel_curves.csv',
              'manuscript/macros_memetic_parallel.tex'):
        assert not os.path.exists(tmp_path / 'w' / 'out' / f)


@pytest.mark.skipif(not os.path.isfile(os.path.join(ROOT, 'manuscript', 'sections_new',
                                                    'tables', 'tq3_decision_parallel.tex')),
                    reason='needs the source of the manuscript '
                           'manuscript/sections_new/tables/tq3_decision_parallel.tex, which the release '
                           'does not hold')
def test_decision_macros_and_table_file(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    assert M.dry_verify(res, paths['curves']) == []
    d = res['decision_table']
    text = open(paths['macros']).read()
    row = d['sizes']['50x25n30']['rows']['600']['NoSetup']
    assert (f'\\newcommand{{\\MparFiftyMemRuleStartVsPolSixHundredNoSetupSel}}'
            f'{{{row["values"]["memetic_rule"]}}}') in text
    assert '\\newcommand{\\MparEightyCpPolStartVsPolSixHundredSetupBold}{\\bfseries}' in text
    assert '\\newcommand{\\MparBothDecisionRowN}{12}' in text
    assert '\\newcommand{\\MparFiftyDecisionLowestCpPolStartN}{6}' in text
    decision = [x for x in text.splitlines() if x.startswith('\\newcommand')
                and ('VsPol' in x.split('}')[0] and 'Best' not in x.split('}')[0]
                     or 'Decision' in x.split('}')[0])]
    assert decision and all('request clock' in x for x in decision)
    assert all('instances 3-11' in x for x in decision)
    # every Mpar name of the table file is a macro the script writes (the memetic cells of the
    # rows of one and of five minutes take the macros of p137, prefixes Mmin and Mfiv)
    tab = open(os.path.join(ROOT, 'manuscript', 'sections_new', 'tables',
                            'tq3_decision_parallel.tex')).read()
    used = set(re.findall(r'\\(Mpar[A-Za-z]+)', '\n'.join(
        x for x in tab.splitlines() if not x.lstrip().startswith('%'))))
    assert len(used) == 74 and used <= set(names)
    # the first set's outputs and the second set's come first, as they are without the table
    md = open(paths['md']).read()
    assert md.index('## Second set') < md.index('## Decision table on eight cores')


def test_curves_file(tmp_path):
    res, names, paths, _ = M.dry_run(tmp_path / 'w', boot=BOOT)
    rows = list(csv.DictReader(open(paths['curves'])))
    with open(paths['curves']) as f:
        assert tuple(next(csv.reader(f))) == T.CSV_COLUMNS_REQUEST
    methods = {x['method'] for x in rows}
    assert methods == {'memetic_rule_best8', 'memetic_policy_best8'}
    assert not methods & (set(T.P.ARMS) | {T.POLICY_ARM})
    assert {x['n_runs'] for x in rows} == {'8'} and {x['row'] for x in rows} == {'Sel'}
    assert {x['run'] for x in rows} == {'full'} and {x['origin'] for x in rows} == {'request'}
    assert {x['group'] for x in rows} == {'all', 'no_setups', 'setups'}
    # hand-computed: at 100 s since the request a run started from the rule schedule has
    # searched 100 - 2 - 1 = 97 s, so seed 8 holds r0 - 18; the greedy schedule is r0 - 15
    r0 = np.array([[M.r0(i, j) for j in range(8)] for i in range(3, 12)])
    for method, want in (('memetic_rule_best8', -300.0 / (r0 - 15)),
                         ('memetic_policy_best8', -1800.0 / (r0 - 15))):
        x = next(x for x in rows if x['size'] == '50x25n30' and x['combination'] == 'Pool'
                 and x['method'] == method and x['t_s'] == '100')
        assert abs(float(x['mean_pct']) - want.mean()) < 1e-6
        assert (x['n_instances'], x['n_combinations']) == ('9', '8')
    grid = {float(x['t_s']) for x in rows}
    assert grid <= set(res['decision_table']['curves']['grid'])


def test_curve_rows_equal_p94s(tmp_path):
    """curve_rows repeats p94's loop in analyse_request: the same rows for the same series."""
    combos = list(T.COMBOS)
    n, C = 3, len(combos)
    grid = [0.0, 1.0, 2.0, 5.0, 10.0, 60.0]
    rng = np.random.default_rng(3)
    pol = 100.0 + rng.uniform(0, 5, (n, C))
    vals = np.stack([pol + 8 - t ** 0.5 + rng.uniform(0, 1, (n, C)) for t in grid])
    vals[:2] = np.nan                   # no schedule before 2 s
    vals[2][:, 2] = np.nan              # combination S from 5 s
    pres = np.ones((n, C), bool)
    pres[1, 4] = False                  # a cell without a record
    D = dict(A={'memetic_rule': vals}, A_present={'memetic_rule': pres}, rule=pol + 4.0,
             pol_sel=pol, all_pol=None, o_rule=np.full((n, C), 1.0),
             o_pol=np.full((n, C), 0.5), S=np.full((len(grid), n, C), np.nan),
             S_present=np.zeros((n, C), bool), policy_budget_s=None, budgets=[10, 60],
             instances=list(range(n)), arms=['memetic_rule'], n_descriptors=n,
             seeds_with_arrays=[], o_res=np.full((n, C), np.nan))
    R = T.analyse_request({'50x25n30': D}, combos, BOOT, 0.05, grid, [0.0, 1.0], grid,
                          'selected training run s1', primary=60)
    S = R['50x25n30']
    curves, group_curves, first, g_first = M.curve_rows(
        '50x25n30', 'memetic_rule', 'memetic', 'rule', vals, pres, pol, combos, grid, BOOT,
        0.05, 1)
    assert curves and group_curves
    assert curves == S['curves'] and group_curves == S['group_curves']
    assert first == S['curve_start_s']['memetic_rule'] == 5.0
    assert g_first == {k: v['memetic_rule'] for k, v in S['group_curve_start_s'].items()}


# ----------------------------------------------------------------- --preview-first-set

def test_preview_options():
    ap = M.build_parser()
    bad = M.check_options(ap.parse_args(['--preview-first-set']))
    assert any('--preview-first-set needs --out-root' in x for x in bad)
    for folder in ('results', 'manuscript', 'out', 'or_solution'):
        bad = M.check_options(ap.parse_args(['--preview-first-set', '--out-root',
                                             os.path.join(ROOT, folder, 'x')]))
        assert any('lies inside or around' in x for x in bad)
    assert any('--check' in x for x in M.check_options(ap.parse_args(
        ['--preview-first-set', '--check', '--out-root', '/nonexistent/x'])))
    assert any('request clock' in x for x in M.check_options(ap.parse_args(
        ['--preview-first-set', '--origin', 'start', '--out-root', '/nonexistent/x'])))
    assert M.check_options(ap.parse_args(['--preview-first-set', '--out-root',
                                          '/nonexistent/x'])) == []


def test_preview_first_set(tmp_path, capsys):
    for name, kw in (('alone', dict(policy_seeds=(), cp_policy=True)), ('with', {})):
        res, names, paths, _ = M.dry_run(tmp_path / name, boot=BOOT, preview=True, **kw)
        assert M.dry_verify(res, paths['curves']) == []
        assert res['run'] == 'preview' and names is None and 'policy_start' not in res
        assert str(tmp_path / name / 'preview') in paths['json']
        assert not os.path.exists(paths['macros'])
        assert not os.path.exists(tmp_path / name / 'out' / 'results' / 'memetic_parallel.json')
        d = res['decision_table']
        assert 'not for the paper' in d['warning'] and 'not for the paper' in res['warning']
        for part in d['sizes'].values():
            for brow in part['rows'].values():
                for row in brow.values():
                    assert row['values']['memetic_policy'] is None
                    assert row['values']['cpsat_policy'] is not None
                    assert 'memetic_policy' not in row['bold']
                    assert row['bold'] == T.lowest(row['values'])
        assert 'not for the paper' in open(paths['md']).read().split('\n')[2]
        rows = list(csv.DictReader(open(paths['curves'])))
        assert {x['method'] for x in rows} == {'memetic_rule_best8'}
        assert {x['run'] for x in rows} == {M.PREVIEW_RUN} and 'not for the paper' in M.PREVIEW_RUN
    assert 'NOT FOR THE PAPER' in capsys.readouterr().out
