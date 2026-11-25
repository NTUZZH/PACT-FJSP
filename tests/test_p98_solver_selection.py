"""Selection of the solver's configuration (scripts/p98_solver_selection.py): the scores
against a direct recomputation, the tie rule of the selection, the count of shortest
makespans with ties, the improvement over the warm start, the bold markers on the printed
scores, the variable counts of the full model with a missing count, the refusals (nothing
written, non-zero exit), the macro names, and that the two supplement tables use only
macros the script writes. Synthetic record folders written by the tests; no solver runs."""
import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, 'scripts'))
sys.path.insert(0, REPO)
import p98_solver_selection as M  # noqa: E402

P = M.P
ARMS = list(P.PILOT_ARMS)
SIZES = list(P.SIZES)
REGIMES = ('S', 'N')
INSTANCES = (0, 1)
CELLS = [(s, r, i) for s in SIZES for r in REGIMES for i in INSTANCES]
SLOT_A = list(range(8, 16))
TABLES = os.path.join(REPO, 'manuscript', 'sections_new', 'tables')


def warm(size, regime, i):
    return 500.0 + 10 * SIZES.index(size) + 5 * (regime == 'S') + i


def variables(arm, size, regime, i):
    return 1000 * (1 + SIZES.index(size)) + (50000 if regime in P.SETUP else 0) + 7 * i


def log_head(n):
    return ['', 'Starting CP-SAT solver v9.14.6206', '',
            f"Initial optimization model '': (model_fingerprint: 0x1)\n#Variables: "
            f"{n:,} (#ints: 1 in objective)".replace(',', "'"), '',
            "Presolved optimization model '': (model_fingerprint: 0x2)\n#Variables: 12 (#ints: 1)"]


def write_world(tmp_path, value, skip=(), share=0.95, affinity=SLOT_A, edit=None):
    """Records of the eight candidates on CELLS; value(arm, cell) is the objective.
    edit(arm, cell, record) may change a record before it is written."""
    root = tmp_path / 'records'
    for arm in ARMS:
        for c in CELLS:
            if (arm, c) in skip:
                continue
            size, r, i = c
            d = root / f'{size}_cparms' / 'pilot' / arm / r
            d.mkdir(parents=True, exist_ok=True)
            v = float(value(arm, c))
            rec = dict(hint='complete', validated=True, status='FEASIBLE', cpu_share=share,
                       affinity=affinity, objective=v, warm_makespan=warm(*c), budget_s=600.0,
                       workers=8, arm=arm, build_s=0.5 + i + 10 * (r in P.SETUP),
                       hint_source=f'cache_{size}', anytime=[[5.0, v + 2, None], [400.0, v, None]],
                       development_run=False, peak_rss_gb=1.0 + ARMS.index(arm) + 0.25 * i,
                       solver_log_head=log_head(variables(arm, *c)))
            if edit:
                edit(arm, c, rec)
            json.dump(rec, open(d / f'instance_{i:03d}.json', 'w'))
    bars = tmp_path / 'bars.json'
    bars.write_text('{}')
    return root, bars


def run(tmp_path, root, bars, out=None, extra=()):
    out = out or tmp_path / 'out'
    rc = M.main(['--out-root', str(out), '--records-root', str(root), '--bars', str(bars),
                 '--regimes', *REGIMES, '--instances', *map(str, INSTANCES), *extra])
    return rc, out


def macros(out):
    text = open(out / 'manuscript' / M.MACRO_FILE).read()
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}  %', text))


def result(out):
    return json.load(open(out / 'results' / 'solver_selection.json'))


def outputs(out):
    return sorted(os.path.relpath(os.path.join(d, f), out)
                  for d, _, fs in os.walk(out) for f in fs)


def default_value(arm, c):
    """lns shortest everywhere; every other row k above the warm start's level by k."""
    size, r, i = c
    k = ARMS.index(arm)
    return warm(*c) - 20 - 3 * (r == 'S') if arm == 'lns' else warm(*c) - 10 + k + i


def parts_of(c):
    size, r, _ = c
    return ['all', size, 'setup' if r in P.SETUP else 'nosetup']


PART_WORD = {'all': 'All', SIZES[0]: 'Fifty', SIZES[1]: 'Eighty', 'setup': 'Setup',
             'nosetup': 'NoSetup'}


# ---------------------------------------------------------------------------

def test_scores_equal_a_direct_recomputation(tmp_path):
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    for j, arm in enumerate(ARMS):
        for part, word in PART_WORD.items():
            cs = [c for c in CELLS if part in parts_of(c)]
            want = np.mean([100.0 * (default_value(arm, c) - min(default_value(b, c) for b in ARMS))
                            / min(default_value(b, c) for b in ARMS) for c in cs])
            assert R['selection']['scores'][arm][part] == pytest.approx(want, abs=1e-12)
            assert mac[f'CselRow{M.ROW_WORDS[j]}Score{word}'] == f'{want:.2f}'
    others = [R['selection']['scores'][a]['all'] for a in ARMS if a != 'lns']
    assert mac['CselScoreOthersMin'] == f'{min(others):.2f}'
    assert mac['CselScoreOthersMax'] == f'{max(others):.2f}'
    assert mac['CselScoreLowest'] == '0.00'
    assert mac['CselRuns'] == str(len(CELLS)) and mac['CselInstances'] == str(len(INSTANCES))
    assert mac['CselBudgetMin'] == '10' and mac['CselTieMargin'] == '0.1'
    assert R['checks']['same_warm_start_per_run'] is True


def test_the_simpler_candidate_within_the_margin_is_selected(tmp_path):
    # lns lowest; compact (row 5) above it by 0.05 points on average: compact is selected
    def value(arm, c):
        base = 1000.0
        return base * (1 + 0.0005 * (arm == 'compact')) if arm in ('lns', 'compact') else base + 30
    root, bars = write_world(tmp_path, value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    assert R['selection']['scores']['compact']['all'] == pytest.approx(0.05)
    assert R['selection']['within_tie'] == ['compact', 'lns']
    assert R['selection']['chosen'] == 'compact' and mac['CselSelectedRow'] == '5'
    assert mac['CselSelectedConfiguration'] == M.CANDIDATE_TEXT['compact']
    # 0.2 points behind: outside the margin, the lowest score wins
    root, bars = write_world(tmp_path / 'b', lambda a, c: 1000.0 * (1 + 0.002 * (a == 'compact'))
                             if a in ('lns', 'compact') else 1030.0)
    rc, out = run(tmp_path / 'b', root, bars)
    assert rc == 0 and macros(out)['CselSelectedRow'] == '8'


def test_shortest_makespans_count_every_candidate_of_a_tie(tmp_path):
    # on the runs of instance 0, radius ties lns; elsewhere lns alone is shortest
    def value(arm, c):
        if arm == 'lns' or (arm == 'radius' and c[2] == 0):
            return 900.0
        return 950.0 + ARMS.index(arm)
    root, bars = write_world(tmp_path, value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    count = {r['candidate']: r['shortest'] for r in R['rows']}
    assert count['lns'] == len(CELLS)
    assert count['radius'] == len([c for c in CELLS if c[2] == 0])
    assert all(count[a] == 0 for a in ARMS if a not in ('lns', 'radius'))
    assert mac['CselRowSevenShortest'] == str(count['radius'])
    assert mac['CselRowEightShortest'] == str(len(CELLS))


def test_improvement_equals_a_direct_recomputation(tmp_path):
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    for j, arm in enumerate(ARMS):
        row = R['rows'][j]
        for part, word in PART_WORD.items():
            cs = [c for c in CELLS if part in parts_of(c)]
            want = np.mean([100.0 * (warm(*c) - default_value(arm, c)) / warm(*c) for c in cs])
            assert row['improve'][part] == pytest.approx(want, abs=1e-12)
            assert mac[f'CselRow{M.ROW_WORDS[j]}Improve{word}'] == M.pct_text(want)
        for size in SIZES:
            for g, gw in (('setup', 'Setup'), ('nosetup', 'NoSetup')):
                cs = [c for c in CELLS if c[0] == size and parts_of(c)[2] == g]
                want = np.mean([100.0 * (warm(*c) - default_value(arm, c)) / warm(*c) for c in cs])
                assert row['improve_group'][f'{size}|{g}'] == pytest.approx(want, abs=1e-12)
                assert mac[f'CselRow{M.ROW_WORDS[j]}Improve{M.SIZE_WORD[size]}{gw}'] == \
                    M.pct_text(want)


def test_bold_marks_the_lowest_printed_score_of_each_column(tmp_path):
    # radius exceeds lns by 0.004 % on one run of 50 modules only: its printed scores equal
    # those of lns (0.00) in every column, so both are bold everywhere
    def value(arm, c):
        if arm == 'lns':
            return 1000.0
        if arm == 'radius':
            return 1000.04 if c == (SIZES[0], 'S', 0) else 1000.0
        return 1010.0
    root, bars = write_world(tmp_path, value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    assert 0 < R['selection']['scores']['radius']['all'] < 0.005
    for word in PART_WORD.values():
        for j, arm in enumerate(ARMS):
            bold = mac[f'CselRow{M.ROW_WORDS[j]}Score{word}Bold']
            assert bold == (M.BOLD if arm in ('lns', 'radius') else '')


def test_variable_counts_of_the_full_model_and_a_missing_count(tmp_path):
    dropped = (SIZES[1], 'S', 1)

    def edit(arm, c, rec):
        if arm == M.FULL_MODEL and c == dropped:
            rec['solver_log_head'] = []
    root, bars = write_world(tmp_path, default_value, edit=edit)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    R, mac = result(out), macros(out)
    for size in SIZES:
        for r, gw in (('S', 'Setup'), ('N', 'NoSetup')):
            g = f'{M.SIZE_WORD[size]}{gw}'
            have = [variables(M.FULL_MODEL, size, r, i) for i in INSTANCES
                    if (size, r, i) != dropped]
            assert mac[f'CselModelVars{g}'] == M.P94.count_text(float(np.median(have)))
            assert mac[f'CselModelVarsMissing{g}'] == ('1' if (size, r) == dropped[:2] else '0')
            build = [0.5 + i + 10 * (r in P.SETUP) for i in INSTANCES]
            assert mac[f'CselModelBuild{g}'] == f'{np.median(build):.2f}'
    fm = R['full_model']['groups'][f'{SIZES[1]}|setup']
    assert len(fm['variables_missing']) == 1 and 'no solver_log_head' in \
        fm['variables_missing'][0]['reason']
    assert '{,}' in mac['CselModelVarsFiftySetup']          # 51{,}000: grouped digits


@pytest.mark.parametrize('case', ['missing', 'invalid', 'no_bars'])
def test_a_refused_selection_writes_nothing_and_exits_nonzero(tmp_path, capsys, case):
    skip = {('lns', (SIZES[1], 'N', 1))} if case == 'missing' else ()
    aff = list(range(0, 8)) if case == 'invalid' else SLOT_A
    root, bars = write_world(tmp_path, default_value, skip=skip, affinity=aff)
    if case == 'no_bars':
        bars.unlink()
    out = tmp_path / 'out'
    rc, _ = run(tmp_path, root, bars, out=out)
    err = capsys.readouterr().err
    assert rc != 0 and 'REFUSING' in err
    assert not out.exists() or outputs(out) == []
    assert {'missing': 'missing ', 'invalid': 'invalid ', 'no_bars': 'bar file'}[case] in err


def test_a_refusal_removes_earlier_outputs(tmp_path):
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0 and len(outputs(out)) == 3
    os.remove(root / f'{SIZES[0]}_cparms' / 'pilot' / 'compact' / 'N' / 'instance_000.json')
    rc, out = run(tmp_path, root, bars)
    assert rc != 0 and outputs(out) == []


def test_macro_names_are_unique_and_every_line_says_what_it_is(tmp_path):
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    lines = open(out / 'manuscript' / M.MACRO_FILE).read().splitlines()
    defs = [ln for ln in lines if ln.startswith('\\newcommand')]
    names = [re.match(r'\\newcommand\{\\([^}]*)\}', ln).group(1) for ln in defs]
    assert len(names) == len(set(names)) and len(names) > 300
    assert all(re.fullmatch(r'Csel[A-Za-z]+', n) for n in names)
    assert all(re.search(r'\}  % \S', ln) for ln in defs)
    # the check refuses a repeated name
    L = M.macro_lines(result(out))
    with pytest.raises(M.Refusal):
        M.check_macro_names(L + [L[0]], str(out / 'manuscript' / M.MACRO_FILE))


@pytest.mark.skipif(
    not all(os.path.isfile(os.path.join(TABLES, f))
            for f in ('ts_solver_selection.tex', 'ts_solver_model.tex')),
    reason='needs the tables of the manuscript ts_solver_selection.tex and ts_solver_model.tex '
           'under manuscript/sections_new/tables, which the release does not hold')
def test_the_two_tables_use_only_macros_the_script_writes(tmp_path):
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0
    written = set(macros(out))
    for f in ('ts_solver_selection.tex', 'ts_solver_model.tex'):
        body = '\n'.join(ln for ln in open(os.path.join(TABLES, f)).read().splitlines()
                         if not ln.lstrip().startswith('%'))
        used = set(re.findall(r'\\(Csel[A-Za-z]+)', body))
        assert used and used <= written, sorted(used - written)


def test_the_working_directory_and_command_line_are_kept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root, bars = write_world(tmp_path, default_value)
    rc, out = run(tmp_path, root, bars)
    assert rc == 0 and os.getcwd() == str(tmp_path)
