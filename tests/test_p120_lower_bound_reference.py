"""scripts/p120_lower_bound_reference.py: the three lower bounds, the self-check and the gaps.

  * heads, tails and the machine-load bound on small instances computed by hand: lags, the
    minimum-setup term, the ceiling under integer times and its absence otherwise, the station
    types, and the extra machine sets when the eligible sets do not partition the operations;
  * the machine-load bound never exceeds the makespan of a feasible schedule of the reference
    simulator on a small instance with lags, setups and unavailability windows (random orders);
  * on a stored instance of every combination, the longest job chain of shortest blocks equals the
    environment's makespan estimate at the empty schedule (bound (i), p113's function);
  * stored solver records: a schedule that today's instance accepts is used, one that it rejects
    and a record without a schedule or a warm start are not verifiable, a record naming another
    cell is rejected, and a bound above a validated feasible makespan is excluded;
  * --dry recovers the planted gaps exactly, writes letters-only macros with the prefix Lbr, and
    every pooled macro value follows from the JSON;
  * a feasible makespan below a bound stops the script with nothing written; an incomplete input
    is refused without --partial, and --partial writes only to --out with a PARTIAL first line;
    --check exits 0 on the complete synthetic tree and 1 when an array is missing;
  * the timed comparison as launched: without --campaign every mode refuses and names the
    descriptions to pass; the expected cells are those of the launched 600 s tokens of the two
    rule-started searches (CP-SAT on two instances, the memetic algorithm on all three), so the
    other instances never count as missing, while a missing launched cell does; other tokens are
    read past, an unknown token or a size outside the analysis is refused, and a set without a
    token for a search has no column for it; every method's instance count and the policy's gap
    on each search's instances are macros;
  * a stored record whose file was modified before the date of the rule gives no bound, keeps
    its validated schedule as a feasible makespan, and is counted (JSON, macros); an earlier
    date keeps it; the folders of the records used carry their budgets, warm starts and file
    times; a descriptor with release dates or a setup multiplier is outside the solver model;
  * cells proved optimal: a hand-made set with a known count and the tolerance edge; the
    per-combination macros of the training size follow the naming scheme (a combination is
    always followed by a word that starts with G or O), no name is a prefix of a macro under
    manuscript/ or has one as its prefix; the table parses (two comment lines, booktabs, the
    column count of every row, the dash of a method without records, the instance counts of the
    heading read from the data, values equal to the JSON, no bold), and --partial writes it with
    the PARTIAL first line.

Synthetic arrays and records under the pytest temporary folder, real descriptors of data/FAMILY
(read only); nothing in the repository is written.
Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -B -m pytest -q tests/test_p120_lower_bound_reference.py
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

import p120_lower_bound_reference as P  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from constraint_family.reference_sim import simulate  # noqa: E402

needs_data = pytest.mark.skipif(
    not all(os.path.isdir(os.path.join(ROOT, f'data/FAMILY/{s}/desc')) for s in P.DRY_SETS),
    reason='needs the descriptors of data/FAMILY')


def desc(lag=None, setup=None, windows=None, op_class=None, n=5):
    return ConstraintDescriptor(active=(lag is not None, setup is not None, windows is not None),
                                lag=np.zeros(n) if lag is None else np.asarray(lag, float),
                                op_class=np.zeros(n, int) if op_class is None else op_class,
                                setup=setup, windows=windows, horizon=100)


# three jobs on three machines: machines 0 and 1 form station type 0, machine 2 type 1
JL = np.array([2, 2, 1])
PT = np.array([[3, 4, 0], [0, 0, 5], [2, 2, 0], [0, 0, 6], [5, 5, 0]], float)
MT = np.array([0, 0, 1])
CLS = np.array([0, 0, 1, 1, 0])
# four classes and the cold-start row (index 4); the column minima are 2 for class 0, 3 for 1
S = np.array([[2, 4, 5, 5], [4, 3, 5, 5], [5, 5, 5, 5], [5, 5, 5, 5], [2, 3, 5, 5]], float)


def test_heads_and_tails_by_hand():
    h, q = P.heads_tails([3, 2], [2, 3, 4, 5, 1], [1, 0, 2, 3, 9])
    assert h.tolist() == [0, 3, 6, 0, 8]
    assert q.tolist() == [8, 4, 0, 4, 0]


def test_machine_load_bound_by_hand():
    # lag 1 after op 0: d = 3 5 2 6 5; type 0 holds ops 0 2 4: 0 + ceil(10 / 2) + 0 = 5;
    # type 1 holds ops 1 3: min h = min(4, 2) = 2, 2 + 11 + 0 = 13; chains 9, 8, 5
    x = P.machine_load_bound(JL, PT, desc(lag=[1, 0, 0, 0, 0]), MT)
    assert (x['value'], x['kind'], x['group'], x['chain'], x['integral']) == (13.0, 'type', 1, 9.0,
                                                                            True)
    assert {g['group']: g['value'] for g in x['groups']} == {0: 5.0, 1: 13.0}
    # with setups the shortest blocks gain the column minima (2 for class 0, 3 for class 1):
    # d = 5 7 5 9 7, h = 0 6 0 5 0, q = 8 0 9 0 0; type 0: 0 + ceil(17 / 2) + 0 = 9;
    # type 1: 5 + 16 + 0 = 21; chains 13, 14, 7
    d = desc(lag=[1, 0, 0, 0, 0], setup=S, op_class=CLS)
    x = P.machine_load_bound(JL, PT, d, MT)
    assert (x['value'], x['chain']) == (21.0, 14.0)
    assert {g['group']: g['value'] for g in x['groups']} == {0: 9.0, 1: 21.0}
    # a processing time of 5.5 h: no ceiling, type 0 holds 0 + 17.5 / 2 + 0 = 8.75
    pt = PT.copy()
    pt[4, :2] = 5.5
    x = P.machine_load_bound(JL, pt, d, MT)
    assert not x['integral'] and {g['group']: g['value'] for g in x['groups']}[0] == 8.75


def test_eligible_sets_that_do_not_partition():
    # ops 0 and 2 may use machine 0 only: the eligible set {0} adds 0 + 3 + 2 = 5 + q
    pt = PT.copy()
    pt[0, 1] = pt[2, 1] = 0
    st = P.station_structure(pt, MT)
    assert not st['partition'] and st['one_type'] == 5 and st['whole_type'] == 3
    x = P.machine_load_bound(JL, pt, desc(), MT)
    sets = {tuple(g['group']) if isinstance(g['group'], list) else g['group']: g['value']
            for g in x['groups']}
    # q of op 0 is 5 (op 1), of op 2 is 6 (op 3): the set {0} gives 0 + 5 + 5 = 10; type 0
    # gives 0 + ceil(10 / 2) + 0 = 5 and type 1 min(3, 2) + 11 + 0 = 13
    assert sets[(0,)] == 10.0 and sets[0] == 5.0 and x['value'] == 13.0
    assert P.station_structure(PT, MT)['partition']


def test_bound_below_every_simulated_schedule():
    rng = np.random.default_rng(5)
    d = desc(lag=[1, 0, 2, 0, 0], setup=S, op_class=CLS,
             windows=[[(4, 6)], [(1, 3), (9, 11)], [(7, 8)]])
    lb = P.machine_load_bound(JL, PT, d, MT)['value']
    best = np.inf
    for _ in range(300):
        left = {j: int(JL[j]) for j in range(len(JL))}
        first = np.concatenate([[0], np.cumsum(JL)[:-1]])
        acts = []
        while any(left.values()):
            j = int(rng.choice([k for k, v in left.items() if v]))
            o = int(first[j] + JL[j] - left[j])
            acts.append((j, int(rng.choice(np.where(PT[o] > 0)[0]))))
            left[j] -= 1
        best = min(best, simulate(JL, PT, d, acts)['makespan'])
    assert lb <= best


@needs_data
def test_chain_equals_the_environment_estimate():
    P.analysis_instances('10x25')
    for c in P.R8:
        jl, pt, meta, d, _ = P.cell_data('10x25', c, 'instance_000')
        x = P.machine_load_bound(jl, pt, d, meta['mch_type'])
        assert x['chain'] == pytest.approx(P.instance_bound('10x25', c, 'instance_000', {}),
                                           abs=1e-6)
        assert x['value'] >= 0 and x['integral']


# ----------------------------------------------------------------- the synthetic tree
@pytest.fixture(scope='module')
def built(tmp_path_factory):
    tree = tmp_path_factory.mktemp('p120_tree')
    with P.dry_context(str(tree)) as b:
        pass
    return str(tree), tuple(b)


@pytest.fixture
def tree(built, tmp_path):
    src, b = built
    dst = tmp_path / 'tree'
    shutil.copytree(src, dst)
    return str(dst), b


@needs_data
def test_stored_records(tree):
    t, b = tree
    root = os.path.join(t, 'or_solution', '10x25_dry600s')
    with P.dry_context(t, b):
        names = P.descriptor_names('10x25')
        rec = json.load(open(f'{root}/S/{names[0]}.json'))
        bad = dict(rec, op_st=[x - 1.0 for x in rec['op_st']])      # a schedule today rejects
        json.dump(bad, open(f'{root}/S/{names[0]}.json', 'w'))
        rec = json.load(open(f'{root}/L/{names[0]}.json'))
        json.dump(dict(rec, instance=names[1]), open(f'{root}/L/{names[0]}.json', 'w'))
        analysis = {}
        for size in P.INPUTS.sets:
            nm, idx = P.analysis_instances(size)
            analysis[size] = dict(names=nm, idx=idx, index={n: i for i, n in idx})
        feas = P.Feasible()
        cand, counts, ex = P.stored_bounds(analysis, feas)
        assert counts['wrong_fields'] == 1
        assert any('violates' in e for e in ex['unverifiable'])
        assert any('neither a schedule nor a warm start' in e for e in ex['unverifiable'])
        used, excluded = P.apply_exclusion(cand, feas, counts)
        assert all(x['bound'] > feas.best(x['cell'])[0] for x in excluded)
        assert all(x['bound'] <= feas.best(x['cell'])[0] for x in used)
        assert len(excluded) == 16 and len(used) == 16 - 2


@needs_data
def test_dry_run(tree):
    t, b = tree
    with P.dry_context(t, b) as (_, ref):
        res, names, outs, md = P.run(boot=P.DRY_BOOT)
    assert P.dry_ok(res, ref) and not res['partial']
    assert all(re.fullmatch(r'Lbr[A-Za-z]+', n) for n in names) and len(names) == len(set(names))
    text = open(outs['macros']).read()
    js = json.load(open(outs['json']))
    for size, S in js['per'].items():
        w = P.PREFIX + js['sets'][size]['word']
        for m, word in P.METHOD_WORD.items():
            g = S['gaps'].get(m, {}).get('All')
            if g:
                assert f'\\newcommand{{\\{w}{word}Gap}}{{{g["mean"]:.1f}}}' in text
                assert f'\\newcommand{{\\{w}{word}N}}{{{g["n"]}}}' in text
        for m in ('memetic', 'solver'):
            g = S['matched'][m]['All']
            k = f'{w}PolOn{P.METHOD_WORD[m]}'
            assert f'\\newcommand{{\\{k}Gap}}{{{g["mean"]:.1f}}}' in text
            assert f'\\newcommand{{\\{k}N}}{{{g["n"]}}}' in text
        assert f'\\newcommand{{\\{w}LoadShare}}{{{S["shares"]["All"]["load"]["largest"]:.1f}}}' \
            in text
    # CP-SAT rests on its two launched instances at 10x25, the memetic algorithm on three
    assert '\\newcommand{\\LbrTenSolverN}{2}' in text and '\\newcommand{\\LbrTenMemN}{3}' in text
    assert '\\newcommand{\\LbrTenPolOnSolverN}{2}' in text
    assert js['campaign']['read_past'] == ['memetic_policy@600', 'policy_sample@120']
    segs = {(s['token'], s['size']): s['instances'] for s in js['campaign']['segments']}
    assert segs == {('memetic_rule@600', '50x25n30'): '3-29', ('cpsat_rule@600', '50x25n30'): '3-29',
                    ('memetic_rule@600', '10x25'): '0-29', ('cpsat_rule@600', '10x25'): '0-1'}
    for r in js['cells']:
        assert r['reference'] == max(v for v in (r['chain'], r['load'], r['stored'])
                                     if v is not None)
        assert r['best_known'] >= r['reference'] - 1e-9


@needs_data
def test_self_check_refuses(tree):
    t, b = tree
    with P.dry_context(t, b):
        names = P.descriptor_names('10x25')
        p = f'{P.REG.rule_dir("10x25")}/Result_SSPT_N.npy'      # a further rule of the folder
        col = np.full(len(names), 1e5)
        col[0] = 10.0                                            # below every bound of the cell
        np.save(p, np.stack([col, np.zeros(len(names))], axis=1))
        with pytest.raises(P.Refused, match='below bound'):
            P.run(boot=P.DRY_BOOT)
        assert not any(os.path.exists(x) for x in P.outputs().values())


@needs_data
def test_incomplete_inputs(tree, tmp_path):
    t, b = tree
    with P.dry_context(t, b):
        assert P.check() == 0
        os.remove(P.REG.result_path('released', None, 302, '50x25n30', 'SW'))
        assert P.check() == 1
        with pytest.raises(P.Refused, match='incomplete'):
            P.run(boot=P.DRY_BOOT)
        assert not any(os.path.exists(x) for x in P.outputs().values())
        with pytest.raises(P.Refused, match='real outputs'):
            P.run(partial=True, out_dir=os.path.dirname(P.outputs()['json']), boot=P.DRY_BOOT)
        out = tmp_path / 'partial'
        res, names, outs, md = P.run(partial=True, out_dir=str(out), boot=P.DRY_BOOT)
        assert sorted(os.listdir(out)) == sorted(os.path.basename(x) for x in outs.values())
        assert open(outs['macros']).readline().startswith('% PARTIAL')
        assert open(outs['table']).readline().startswith('% PARTIAL')
        assert md[0].startswith('**PARTIAL')
        assert res['partial'] and any('s302' in m for m in res['missing'])
        assert 'SW' not in res['per']['50x25n30']['gaps'].get('policy_mean', {})
        assert 'All' not in res['per']['50x25n30']['gaps'].get('policy_mean', {})


def test_proved_optimal_count_and_tolerance():
    ref = 100.0
    edge = P.TOL * ref

    def row(best, largest):
        return dict(reference=ref, best_known=best, largest=largest)
    rows = [row(ref, ['stored']),                      # equal: optimal
            row(ref + 0.5 * edge, ['chain', 'stored']),  # inside the tolerance: optimal, a tie
            row(ref + 2.0 * edge, ['load']),           # outside: not optimal
            row(ref + 1.0, ['stored']),                # an hour above: not optimal
            row(None, ['chain'])]                      # no feasible makespan: not optimal
    o = P.proved_optimal(rows)
    assert (o['count'], o['cells']) == (2, 5)
    assert o['attained'] == {'chain': 1, 'load': 0, 'stored': 2}
    assert P.f1(-6.7e-15) == '0.0' and P.f1(-0.04) == '-0.0' and P.f1(19.66) == '19.7'


@needs_data
def test_stored_bounds_written_before_the_date(tree):
    t, b = tree
    root = os.path.join(t, 'or_solution', '10x25_dry600s')
    with P.dry_context(t, b):
        names = P.descriptor_names('10x25')
        old = f'{root}/N/{names[0]}.json'                     # a used record (bound attained)
        stamp = P.since_time('2026-07-07T23:59')
        os.utime(old, (stamp, stamp))
        analysis = {}
        for size in P.INPUTS.sets:
            nm, idx = P.analysis_instances(size)
            analysis[size] = dict(names=nm, idx=idx, index={n: i for i, n in idx})
        feas = P.Feasible()
        cand, counts, _ = P.stored_bounds(analysis, feas)
        assert counts['before_date'] == 1 and counts['stored_since'] == P.STORED_SINCE
        assert counts['by_folder']['10x25_dry600s']['before_date'] == 1
        assert not any(x['record'].endswith(f'N/{names[0]}.json') and '10x25_' in x['record']
                       for x in cand)
        assert any(src.startswith('10x25_dry600s') for src, _ in
                   feas.by_cell.get(('10x25', 'N', 0), ()))          # its schedule still counts
        res, _, outs, md = P.run(boot=P.DRY_BOOT)
        assert res['stored']['counts']['before_date'] == 1
        assert res['per']['10x25']['stored']['before_date'] == 1
        assert '\\newcommand{\\LbrStoredBefore}{1}' in open(outs['macros']).read()
        assert '\\newcommand{\\LbrTenStoredBefore}{1}' in open(outs['macros']).read()
        r = [x for x in res['cells'] if (x['size'], x['combo'], x['index']) == ('10x25', 'N', 0)][0]
        assert r['stored'] is None and r['reference'] == max(r['chain'], r['load'])
        f = res['per']['10x25']['stored']['folders']['10x25_dry600s']
        assert f['budgets'] == {'600': f['records']} and f['warm_start'] == 0
        assert f['oldest'] >= '2026-07-08'
        # an earlier date keeps the record
        with P.inputs(stored_since='2026-07-07'):
            cand, counts, _ = P.stored_bounds(analysis, P.Feasible())
        assert counts['before_date'] == 0
        with pytest.raises(P.Refused, match='YYYY-MM-DD'):
            P.since_time('07/08/2026')


def test_model_mismatch():
    class D:
        pass
    d = D()
    assert P.model_mismatch(d) is None
    d.release = np.zeros(3)
    assert P.model_mismatch(d) == 'release dates'
    d = D()
    d.setup_mult = np.ones(2)
    assert 'setup multiplier' in P.model_mismatch(d)


def _table_rows(text):
    body = text.split('\\toprule', 1)[1].split('\\bottomrule', 1)[0]
    return [x.strip() for x in body.splitlines() if x.strip() and not x.strip().startswith('\\midrule')]


@needs_data
def test_table_and_new_macros(tree, monkeypatch):
    t, b = tree
    with P.dry_context(t, b):
        res, names, outs, md = P.run(boot=P.DRY_BOOT)
    text = open(outs['table']).read()
    js = json.load(open(outs['json']))
    lines = text.splitlines()
    assert lines[0].startswith('% Generated by') and lines[1].startswith('% environment exact')
    assert '\\textbf' not in text and text.count('\\begin{tabular}{@{}') == 1
    spec = ' '.join('l' if m in P.TABLE_INTERVALS else 'r' for m in P.COMB_METHODS)
    assert f'\\begin{{tabular}}{{@{{}}l {spec} rr@{{}}}}' in text
    cols = len(P.COMB_METHODS) + 3
    head, body = text.split('\\toprule', 1)[1].split('\\midrule', 1)
    hrows = [x.strip() for x in head.splitlines() if x.strip()]
    assert hrows[0].count('&') == 3 and f'\\multicolumn{{{len(P.COMB_METHODS)}}}{{c}}' in hrows[0]
    assert hrows[1].startswith('\\cmidrule') and all(x.count('&') == cols - 1 for x in hrows[2:])
    assert len(hrows) == 4 and hrows[3].startswith('combination &')
    rows = _table_rows('\\toprule' + body)
    for r in rows:
        assert r.endswith('\\\\') and r.count('{') == r.count('}') and r.count('$') % 2 == 0
        if r.startswith('\\multicolumn'):
            assert r.startswith(f'\\multicolumn{{{cols}}}{{@{{}}l}}{{\\textit{{')
        else:
            assert r.count('&') - r.count('\\&') == cols - 1
    data = {r.split(' & ')[0]: r[:-2].strip().split(' & ') for r in rows
            if not r.startswith('\\multicolumn')}
    T = js['per']['10x25']

    def shown(g, m):
        s = f'${g["mean"]:.1f}$'
        return s + (f' [${g["lo"]:.1f}$, ${g["hi"]:.1f}$]' if m in P.TABLE_INTERVALS else '')
    for c in P.R8 + ('All',):
        cells = data['pooled' if c == 'All' else f'${c}$']
        for j, m in enumerate(P.COMB_METHODS):
            assert cells[1 + j] == shown(T['gaps'][m][c], m)
        assert cells[-1] == str(T['optimal'][c]['count'])
        assert cells[-2] == f'${T["over_best"][c]["reference"]["mean"]:.1f}$'
    # the instance counts of the heading lines come from the data: 3 instances, CP-SAT on 2
    h10 = [r for r in rows if '10 modules' in r][0]
    assert '$n = 3$ instances per combination' in h10 and 'solver $n = 2$' in h10 \
        and 'memetic algorithm $n = 3$' in h10
    assert [r for r in rows if 'pooled over the eight' in r][0].endswith(
        '$3$ instances at 50 modules}} \\\\')
    prod = data['50 modules']
    assert prod[2] == '\\textendash'                     # no specialist at 50 modules
    assert prod[4] == shown(js['per']['50x25n30']['gaps']['memetic']['All'], 'memetic')
    # the narrower form prints the mean alone; a method on other instances than the row says so
    monkeypatch.setattr(P, 'TABLE_INTERVALS', ('policy',))
    g = dict(mean=1.25, lo=-6.7e-15, hi=2.0, n=2)
    assert P.table_cell(g, 'specialist') == '$1.2$'
    assert P.table_cell(g, 'policy', n_row=3) == '$1.2$ [$0.0$, $2.0$] ($n = 2$)'
    assert P.table_cell(None, 'solver') == '\\textendash'
    monkeypatch.undo()
    # the new macros: per combination at the training size, per set the optimal counts and ratios
    for c in P.R8:
        for m in P.COMB_METHODS:
            for s in ('', 'Lo', 'Hi', 'N'):
                assert f'LbrTenComb{c}Gap{P.METHOD_WORD[m]}{s}' in names
        for s in ('Opt', 'OptCells', 'OverBestRef', 'OverBestChain', 'OverBestLoad'):
            assert f'LbrTenComb{c}{s}' in names
    for w in ('Ten', 'Fifty'):
        for s in ('Opt', 'OptCells', 'ChainOverBest', 'LoadOverBest', 'RefOverBest'):
            assert f'Lbr{w}{s}' in names
    assert not any(n.startswith('LbrFiftyComb') for n in names)
    # after Comb<R> always a word starting with G or O, so a combination never runs into Lo or Hi
    combos = '|'.join(sorted(P.R8, key=len, reverse=True))
    for n in names:
        if n.startswith('LbrTenComb'):
            assert re.fullmatch(rf'LbrTenComb({combos})(Gap(Pol|Spec|Rule|Mem|Solver)(Lo|Hi|N)?'
                                r'|Opt|OptCells|OverBest(Ref|Chain|Load))', n), n
    macros = open(outs['macros']).read()
    assert P.NAME_SCHEME in macros
    g = T['gaps']['solver']['LS']
    assert f'\\newcommand{{\\LbrTenCombLSGapSolverN}}{{{g["n"]}}}' in macros
    # no emitted name is a prefix of a macro defined under manuscript/ or has one as its prefix
    have = P.defined_macros()
    assert not [(n, h) for n in names for h in have if h.startswith(n) or n.startswith(h)]
    # a macro of the prefix defined elsewhere stops the output
    monkeypatch.setattr(P, 'defined_macros', lambda: {'LbrTenN': 'manuscript/other.tex'})
    with pytest.raises(P.Refused, match='prefix Lbr'):
        P.check_names(['\\newcommand{\\LbrTenN}{1}'])


@needs_data
def test_campaign_as_launched(tree, tmp_path):
    t, b = tree
    table, ref = b
    with P.dry_context(t, b):
        # every mode refuses without a description and names the two to pass
        with P.inputs(campaigns=None):
            for call in (P.check, lambda: P.run(boot=P.DRY_BOOT)):
                with pytest.raises(P.Refused, match='as it was launched') as e:
                    call()
                assert all(s in str(e.value) for s in P.AS_LAUNCHED)
            with pytest.raises(SystemExit, match='as it was launched'):
                P.main(['--check'])
        assert not any(os.path.exists(x) for x in P.outputs().values())
        for bad, why in (('10x25 = memetic_rule@600 cpsat_rul@600', 'unknown token'),
                         ('20x25 40x25 = memetic_rule@600', 'not a test set'),
                         ('10x25 memetic_rule@600', 'want')):
            with P.inputs(campaigns=[bad]), pytest.raises(P.Refused, match=why):
                P.check()
        # complete as launched: the CP-SAT cells of instance 2 at 10x25 were never launched
        assert P.check() == 0
        # a set without a token for a search has no column for it and misses nothing
        with P.inputs(campaigns=['50x25n30 = memetic_rule@600', '10x25 = memetic_rule@600']):
            assert P.check() == 0
            res, names, outs, md = P.run(boot=P.DRY_BOOT)
            assert not res['partial'] and 'solver' not in res['per']['10x25']['gaps']
            assert not any(n.endswith('SolverGap') or 'PolOnSolver' in n for n in names)
            assert res['per']['50x25n30']['gaps']['memetic']['All']['n'] == 3
    # a launched cell that is missing is an incomplete input
    t2 = tmp_path / 'tree2'
    shutil.copytree(t, t2)
    gone = dict(table)
    gone[('cpsat_rule', '10x25', 'LS', 1)] = None
    with P.dry_context(str(t2), (gone, ref)):
        assert P.check() == 1
        with pytest.raises(P.Refused, match='CP-SAT started from the best dispatching rule, '
                                            '10x25/LS: cells 1 missing'):
            P.run(boot=P.DRY_BOOT)
        res, _, _, _ = P.run(partial=True, out_dir=str(tmp_path / 'partial'), boot=P.DRY_BOOT)
        assert res['per']['10x25']['gaps']['solver']['All']['n'] == 1
        assert res['per']['10x25']['gaps']['solver']['LS']['n'] == 1
