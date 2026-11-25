"""The timed comparison on its nine instances against all 27 evidence instances: the option
--tag of scripts/p94_timed_inference.py and scripts/p130_timed_instance_sets.py.

  * p94 --tag on a synthetic campaign (the world of tests/test_p94_timed_inference.py): the
    run writes results/timed_inference_<tag>.{json,md} and results/timed_curves_<tag>.csv and
    no macro file; every untagged output under the same root stays byte-identical; its JSON
    and CSV equal those of the untagged run with the same options (the JSON apart from the
    field tag); a refusal removes the tagged files only; the request clock, partial mode and a
    tag that is not letters are refused before anything is read;
  * p130 on synthetic records: every mean, interval, direction and shift equals the built one,
    and the counts of equal direction words and of flips and the largest shifts equal a
    recomputation from the built values; macro names are Tis plus letters, unique, and define
    every macro the supplement table uses;
  * every guard refuses (a different selected run, clock, budget set, arm, a design that is not
    a proper subset of the evidence instances, evidence that are not the evidence instances,
    instances that differ between sizes, a partial or stand-in record, a wrong tag, a missing
    entry or record) and leaves no output;
  * end to end: two tagged runs of p94 on the synthetic campaign (instances 3-11 and 3-29) read
    by p130;
  * the prefix Tis is refused once another .tex file under manuscript/ defines it.

Run from the repository root on one core:
  python -B -m pytest -q tests/test_p130_timed_instance_sets.py
CPU only; every file is written under the pytest temporary folder.
"""
import csv
import hashlib
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.join(ROOT, 'tests'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=None,
                                        PACT_RESULTS=None)

import arm_registry as REG  # noqa: E402
import p130_timed_instance_sets as M  # noqa: E402
import stats_family as SF  # noqa: E402
import test_p94_timed_inference as W  # noqa: E402

T = M.T
P = T.P
SIZES = ('50x25n30', '80x25n30')
DESIGN, EVIDENCE = list(range(3, 12)), list(range(3, 30))
TABLE = os.path.join(ROOT, 'manuscript', 'sections_new', 'tables', 'ts_timed_instance_sets.tex')
FLIP = ('80x25n30', 'cpsat', 600, 'W')          # the one built flip: ahead against behind


# ---------------------------------------------------------------------------
# synthetic records of p94 (the fields p130 reads)

def built(key, size, method, budget, entry):
    """(mean, lo, hi) of an entry: evidence mean from the position of the entry, the design
    mean shifted by a built amount; one entry flips, every tenth has a different width."""
    k = (SIZES.index(size) * 1000 + M.METHODS.index(method) * 100 + M.BUDGETS.index(budget) * 20
         + M.ENTRIES.index(entry))
    ev = -3.0 + (k % 13) * 0.5
    shift = 0.1 * ((k % 7) - 3)
    if (size, method, budget, entry) == FLIP:
        return (-3.0, -4.0, -2.0) if key == 'design' else (3.0, 2.0, 4.0)
    mean = ev + shift if key == 'design' else ev
    half = 1.5 if k % 10 == 0 else 0.8
    return mean, mean - half, mean + half


def record(key, tag, inst, sizes=SIZES, origin='start', seed=304, budgets=(60, 300, 600),
           arms=M.ARMS):
    results = {}
    for s in sizes:
        S = dict(instances=list(inst), arms=list(arms),
                 design={a: dict(instances=list(inst), range_given=True) for a in arms}, i={})
        for m in M.METHODS:
            for b in budgets:
                rows = {}
                for r in M.ENTRIES:
                    mean, lo, hi = built(key, s, m, b, r) if b in M.BUDGETS else (0.0, -1.0, 1.0)
                    rows[r] = dict(mean=mean, lo=lo, hi=hi, direction=SF.direction(lo, hi),
                                   n=len(inst), c=M.n_combinations(r), shorter=0.0)
                S['i'].setdefault(m, {})[str(b)] = dict(selected=rows, all_seeds={})
        results[s] = S
    rng = P.compress(inst)
    return dict(generator='scripts/p94_timed_inference.py', run='full', partial=False,
                stand_in=False, tag=tag, setting=REG.describe(), seeds=list(REG.seeds()),
                origin=origin, origin_text=T.ORIGIN_TEXT[origin], budgets=list(budgets),
                boot=200, alpha=0.05, selection=dict(seed=seed), policy=f'selected training run '
                f's{seed}', cpsat_arm='lns',
                campaigns=[f'{" ".join(sizes)} = ' + ' '.join(f'{a}@600:{rng}' for a in arms)],
                combinations=list(T.COMBOS), results=results)


def write_records(root, design=None, evidence=None):
    """The two records under the default tags of p130, whatever their field tag says."""
    d = design or record('design', 'startnine', DESIGN)
    e = evidence or record('evidence', 'startall', EVIDENCE)
    for J, tag in ((d, 'startnine'), (e, 'startall')):
        p = M.record_path(str(root), tag)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump(J, open(p, 'w'))
    return d, e


def opts(root, *extra):
    return M.parse(['--records-root', str(root), '--out-root', str(root / 'out'), *extra])


def macro_values(path):
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(path).read()))


# ---------------------------------------------------------------------------
# p130 on synthetic records

def test_values_shifts_and_counts_equal_the_built_ones(tmp_path):
    write_records(tmp_path)
    res, names, paths = M.run(opts(tmp_path))
    mac = macro_values(paths['macros'])
    assert len(names) == len(set(names)) and all(re.fullmatch(r'Tis[A-Za-z]+', n) for n in names)
    expect = {lv: dict(entries=0, same=0, flips=0, top=0.0) for lv, _, _ in M.LEVELS}
    for s in SIZES:
        for m in M.METHODS:
            for b in M.BUDGETS:
                for r in M.ENTRIES:
                    d, e = built('design', s, m, b, r), built('evidence', s, m, b, r)
                    x = res['results'][s][m][str(b)][r]
                    assert (x['design']['mean'], x['design']['lo'], x['design']['hi']) == d
                    assert (x['evidence']['mean'], x['evidence']['lo'], x['evidence']['hi']) == e
                    assert x['shift'] == pytest.approx(d[0] - e[0])
                    dd, de = SF.direction(d[1], d[2]), SF.direction(e[1], e[2])
                    base = f'Tis{T.SIZE_WORD[s]}{T.METHOD_WORD[m]}{T.number_word(b)}{r}'
                    assert mac[base + 'Design'] == T.f2(d[0])
                    assert mac[base + 'EvidenceLo'] == T.f2(e[1])
                    assert mac[base + 'DesignHi'] == T.f2(d[2])
                    assert mac[base + 'EvidenceDir'] == de and mac[base + 'DesignDir'] == dd
                    assert mac[base + 'Shift'] == T.f2(d[0] - e[0])
                    for lv, _, members in M.LEVELS:
                        if r in members:
                            q = expect[lv]
                            q['entries'] += 1
                            q['same'] += dd == de
                            q['flips'] += {dd, de} == {'ahead', 'behind'}
                            q['top'] = max(q['top'], abs(d[0] - e[0]))
    for lv, word, _ in M.LEVELS:
        sm, q = res['summary'][lv], expect[lv]
        assert (sm['entries'], sm['same_direction'], sm['flips']) == \
            (q['entries'], q['same'], q['flips'])
        assert sm['largest_abs_shift'] == pytest.approx(q['top'])
        assert mac[f'Tis{word}EntryN'] == str(q['entries'])
        assert mac[f'Tis{word}SameDirN'] == str(q['same'])
        assert mac[f'Tis{word}FlipN'] == str(q['flips'])
        assert mac[f'Tis{word}ShiftMax'] == f'{q["top"]:.2f}'
    assert expect['all']['entries'] == 2 * 2 * 3 * 11 and expect['pooled']['entries'] == 36
    assert res['summary']['single']['flips'] == 1 and res['summary']['pooled']['flips'] == 0
    assert res['summary']['single']['largest_at']['entry'] == 'W'      # the flip, 6 points
    assert mac['TisDesignInstN'] == '9' and mac['TisEvidenceInstN'] == '27'
    assert (mac['TisBudgetShortMin'], mac['TisBudgetMidMin'], mac['TisBudgetLongMin']) == \
        ('1', '5', '10')
    J = json.load(open(paths['json']))
    assert J['records']['design']['tag'] == 'startnine'
    assert J['records']['evidence']['path'] == 'timed_inference_startall.json'
    assert len(J['records']['design']['sha256']) == 64
    assert J['instances']['design'] == {s: DESIGN for s in SIZES}
    assert any(x['flip'] for x in J['differing'])
    md = open(paths['md']).read()
    assert '| 600 | W |' in md and '(flip)' in md
    text = open(paths['macros']).read()
    assert text.startswith(f'% Generated by {M.GENERATOR} -- do not hand-edit')
    assert 'Clock origin start' in text and '/tmp' not in text and ROOT not in text


@pytest.mark.skipif(not os.path.isfile(TABLE),
                    reason='needs the source of the manuscript '
                           'manuscript/sections_new/tables/ts_timed_instance_sets.tex, which the release '
                           'does not hold')
def test_the_supplement_table_uses_only_macros_the_script_writes(tmp_path):
    write_records(tmp_path)
    _, names, _ = M.run(opts(tmp_path))
    body = re.sub(r'(?<!\\)%.*', '', open(TABLE).read())
    used = set(re.findall(r'\\(Tis[A-Za-z]+)', body))
    assert used and used <= set(names)
    assert {f'Tis{sw}{mw}{bw}{r}{st}{end}' for sw in ('Fifty', 'Eighty') for mw in ('Mem', 'Cp')
            for bw in ('Sixty', 'ThreeHundred', 'SixHundred') for st in ('Design', 'Evidence')
            for r, end in (('Pool', ''), ('Pool', 'Lo'), ('Pool', 'Hi'), ('NoSetup', ''),
                           ('Setup', ''))} <= used
    assert '\\label{tab:sn-timed-sets}' in body
    assert not re.search(r'\\newcommand\*?\{\\Tis', body)


def mutate(fn):
    """A pair of records changed by fn(design, evidence)."""
    d, e = record('design', 'startnine', DESIGN), record('evidence', 'startall', EVIDENCE)
    fn(d, e)
    return d, e


def _drop_entry(d, e):
    del d['results']['80x25n30']['i']['cpsat']['300']['selected']['Setup']


def _n_wrong(d, e):
    d['results']['50x25n30']['i']['memetic']['60']['selected']['Pool']['n'] = 8


def _direction_wrong(d, e):
    d['results']['50x25n30']['i']['memetic']['60']['selected']['N']['direction'] = 'ahead'
    d['results']['50x25n30']['i']['memetic']['60']['selected']['N'].update(lo=0.5, hi=1.0)


def _arm_wrong(d, e):
    for J in (d, e):
        for S in J['results'].values():
            S['arms'] = ['memetic_rule', 'memetic_policy']


def _instances_between_sizes(d, e):
    S = d['results']['80x25n30']
    S['instances'] = list(range(4, 13))
    for a in M.ARMS:
        S['design'][a]['instances'] = list(range(4, 13))


@pytest.mark.parametrize('change,reason', [
    (lambda d, e: e['selection'].update(seed=302), 'selected training run differs'),
    (lambda d, e: e.update(policy='selected training run s302'), 'policy differs'),
    (lambda d, e: [J.update(origin='search', origin_text=T.ORIGIN_TEXT['search'])
                   for J in (e,)], 'origin differs'),
    (lambda d, e: [J.update(origin='request') for J in (d, e)], 'clock'),
    (lambda d, e: d.update(budgets=[60, 600]), 'lack [300]'),
    (lambda d, e: d.update(budgets=[1, 60, 300, 600]), 'budgets differs'),
    (lambda d, e: e.update(cpsat_arm='ref'), 'cpsat_arm differs'),
    (lambda d, e: d.update(campaigns=['50x25n30 80x25n30 = memetic_rule@300:3-11 '
                                      'cpsat_rule@600:3-11']), 'arm budgets differ'),
    (_arm_wrong, 'want [\'memetic_rule\', \'cpsat_rule\']'),
    (lambda d, e: d.update(partial=True), 'partial is True'),
    (lambda d, e: e.update(stand_in=True), 'stand_in is True'),
    (lambda d, e: d.update(run='partial'), "run is 'partial'"),
    (lambda d, e: d.update(tag='other'), "tag is 'other'"),
    (lambda d, e: d.update(setting='environment exact, training seeds 301,302,303 (3 runs per '
                                   'arm)'), 'setting is'),
    (lambda d, e: e['results'].pop('80x25n30'), 'sizes differ'),
    (_drop_entry, 'entry or a field'),
    (_n_wrong, 'n 8'),
    (_direction_wrong, 'is not that of'),
    (_instances_between_sizes, 'differ between the sizes'),
])
def test_every_guard_refuses_and_leaves_no_output(tmp_path, change, reason):
    d, e = mutate(change)
    write_records(tmp_path, d, e)
    a = opts(tmp_path)
    for p in M.output_paths(a).values():
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, 'w').write('stale\n')
    with pytest.raises(T.Refusal) as err:
        M.run(a)
    assert any(reason in x for x in err.value.args[0]), err.value.args[0]
    assert not any(os.path.exists(p) for p in M.output_paths(a).values())


@pytest.mark.parametrize('design,evidence,reason', [
    (list(range(3, 30)), list(range(3, 30)), 'not a proper subset'),
    (list(range(3, 12)) + [30], list(range(3, 30)), 'not a proper subset'),
    (list(range(3, 12)), list(range(3, 29)), 'the evidence instances are 3-29'),
])
def test_instance_sets_are_refused_unless_nine_within_the_evidence(tmp_path, design, evidence,
                                                                   reason):
    write_records(tmp_path, record('design', 'startnine', design),
                  record('evidence', 'startall', evidence))
    with pytest.raises(T.Refusal) as err:
        M.run(opts(tmp_path))
    assert any(reason in x for x in err.value.args[0]), err.value.args[0]


def test_a_missing_record_refuses_and_check_says_not_ready(tmp_path, capsys):
    d, _ = write_records(tmp_path)
    os.remove(M.record_path(str(tmp_path), 'startall'))
    with pytest.raises(T.Refusal) as err:
        M.run(opts(tmp_path))
    assert any('timed_inference_startall.json missing' in x for x in err.value.args[0])
    assert M.main(['--records-root', str(tmp_path), '--check']) == 1
    assert 'p130: not ready' in capsys.readouterr().out
    write_records(tmp_path, d)
    assert M.main(['--records-root', str(tmp_path), '--check']) == 0
    out = capsys.readouterr().out
    assert 'both records present and consistent' in out and '50x25n30 3-11' in out
    assert not os.path.exists(tmp_path / 'out')                    # --check writes nothing


def test_tags_must_be_letters_and_distinct(tmp_path):
    for extra in (['--design-tag', 'start9'], ['--evidence-tag', 'start_all'],
                  ['--design-tag', 'same', '--evidence-tag', 'same']):
        with pytest.raises(SystemExit):
            opts(tmp_path, *extra)


def test_prefix_is_checked(tmp_path):
    own = tmp_path / 'manuscript' / 'macros_timedsets.tex'
    own.parent.mkdir()
    lines = ['\\newcommand{\\TisDesignInstN}{9}  % x']
    assert M.check_names(lines, str(own)) == ['TisDesignInstN']
    (own.parent / 'other.tex').write_text('\\newcommand{\\TisFoo}{1}\n')
    with pytest.raises(T.Refusal):
        M.check_names(lines, str(own))
    with pytest.raises(T.Refusal):
        M.check_names(['\\newcommand{\\TisA1}{8}'], str(tmp_path / 'x.tex'))


# ---------------------------------------------------------------------------
# p94 --tag on the synthetic campaign of the p94 tests

@pytest.fixture(scope='module')
def base_world(tmp_path_factory):
    return W.write_world(tmp_path_factory.mktemp('world'))


@pytest.fixture
def world(base_world, tmp_path, monkeypatch):
    return W.enter(base_world, tmp_path / 'w', monkeypatch)


RULE_ARMS = 'memetic_rule@600 cpsat_rule@600'
LONG = ('--budgets', '60', '300', '600', '--primary', '600')


def files_under(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, root)] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return out


def tagged(root, tag):
    return {k: os.path.join(root, 'out', p.format(tag=tag)) for k, p in T.TAG_OUTPUTS}


def test_a_tagged_run_leaves_every_untagged_output_byte_identical(world):
    assert T.main(W.args(world, arms=RULE_ARMS)) == 0
    before = files_under(world / 'out')
    assert sorted(before) == ['manuscript/macros_timed.tex', 'results/timed_curves.csv',
                              'results/timed_inference.json', 'results/timed_inference.md']
    assert T.main(W.args(world, '--tag', 'startnine', arms=RULE_ARMS)) == 0
    after = files_under(world / 'out')
    assert {k: v for k, v in after.items() if k in before} == before
    paths = tagged(world, 'startnine')
    assert sorted(set(after) - set(before)) == sorted(
        os.path.relpath(p, world / 'out') for p in paths.values())
    J = json.load(open(paths['json']))
    U = json.load(open(world / 'out' / 'results' / 'timed_inference.json'))
    assert J.pop('tag') == 'startnine' and 'tag' not in U and J == U
    assert open(paths['csv']).read() == open(world / 'out' / 'results' / 'timed_curves.csv').read()
    md, mdu = open(paths['md']).read().splitlines(), \
        open(world / 'out' / 'results' / 'timed_inference.md').read().splitlines()
    assert md[0] == '# Timed campaign against the policy (full, tag startnine)'
    assert mdu[0] == '# Timed campaign against the policy (full)' and md[1:] == mdu[1:]


def test_a_tagged_refusal_removes_the_tagged_files_only(world, capsys):
    assert T.main(W.args(world, arms=RULE_ARMS)) == 0
    assert T.main(W.args(world, '--tag', 'startnine', arms=RULE_ARMS)) == 0
    before = {k: v for k, v in files_under(world / 'out').items() if 'start' not in k}
    camp = W.camp_of(world)
    os.remove(camp.record(W.cell(camp, 'memetic_rule', 'LS', 7)))
    assert T.main(W.args(world, '--tag', 'startnine', arms=RULE_ARMS)) == 1
    assert 'REFUSING' in capsys.readouterr().err
    assert not any(os.path.exists(p) for p in tagged(world, 'startnine').values())
    assert files_under(world / 'out') == before


@pytest.mark.parametrize('extra,reason', [
    (('--origin', 'request'), 'start and search clocks only'),
    (('--origin', 'start', '--tag', 'start9'), 'letters only'),
    (('--origin', 'start', '--partial', '--out', 'PART'), 'partial mode writes into --out'),
])
def test_a_tag_refuses_the_request_clock_partial_mode_and_other_characters(tmp_path, capsys,
                                                                            extra, reason):
    argv = ['--tag', 'startnine', '--out-root', str(tmp_path / 'out')] + \
        [str(tmp_path / 'part') if x == 'PART' else x for x in extra]
    assert T.main(argv) == 2
    assert reason in capsys.readouterr().err
    assert not os.path.exists(tmp_path / 'out') and not os.path.exists(tmp_path / 'part')


def test_two_tagged_runs_read_by_p130(world):
    for tag, rng in (('startnine', '3-11'), ('startall', '3-29')):
        arms = ' '.join(f'{a}@600:{rng}' for a in ('memetic_rule', 'cpsat_rule'))
        assert T.main(W.args(world, *LONG, '--tag', tag, arms=arms)) == 0
    assert not os.path.exists(world / 'out' / 'manuscript')           # no macro file of p94
    res, names, paths = M.run(M.parse(['--records-root', str(world / 'out'),
                                       '--out-root', str(world / 'p130')]))
    D = json.load(open(tagged(world, 'startnine')['json']))
    E = json.load(open(tagged(world, 'startall')['json']))
    assert res['instances'] == {'design': {W.SIZE: DESIGN}, 'evidence': {W.SIZE: EVIDENCE}}
    assert res['selected_run'] == W.SEL and res['origin'] == 'start'
    for m in M.METHODS:
        for b in M.BUDGETS:
            for r in M.ENTRIES:
                x = res['results'][W.SIZE][m][str(b)][r]
                d = D['results'][W.SIZE]['i'][m][str(b)]['selected'][r]
                e = E['results'][W.SIZE]['i'][m][str(b)]['selected'][r]
                assert x['design'] == {f: d[f] for f in M.ENTRY_FIELDS}
                assert x['evidence'] == {f: e[f] for f in M.ENTRY_FIELDS}
                assert (x['design']['n'], x['evidence']['n']) == (9, 27)
    # the synthetic searches end below the policy on every cell: behind on both sets
    pool = res['results'][W.SIZE]['memetic']['600']['Pool']
    assert pool['design']['direction'] == pool['evidence']['direction'] == 'behind'
    assert res['summary']['all']['flips'] == 0
    mac = macro_values(paths['macros'])
    assert mac['TisFiftyMemSixHundredPoolDesign'] == T.f2(pool['design']['mean'])
    rows = list(csv.DictReader(open(tagged(world, 'startall')['csv'])))
    assert {x['origin'] for x in rows} == {'start'}


def test_the_records_of_p94_carry_every_field_p130_reads(world):
    assert T.main(W.args(world, *LONG, '--tag', 'startall',
                         arms='memetic_rule@600:3-29 cpsat_rule@600:3-29')) == 0
    J = json.load(open(tagged(world, 'startall')['json']))
    E = record('evidence', 'startall', EVIDENCE, sizes=(W.SIZE,))
    missing = [k for k in E if k not in J]
    assert missing == []
    S, SE = J['results'][W.SIZE], E['results'][W.SIZE]
    assert set(SE) <= set(S)
    x = S['i']['cpsat']['300']['selected']['NoSetup']
    assert set(M.ENTRY_FIELDS) <= set(x) and x['c'] == 4 and x['n'] == 27
    assert J['campaigns'] == [f'{W.SIZE} = memetic_rule@600:3-29 cpsat_rule@600:3-29']
