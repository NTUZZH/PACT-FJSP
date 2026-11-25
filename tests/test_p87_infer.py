"""Tests for the infer phase of scripts/p87_sensitivity.py, on synthetic result trees.

  * gaps: first-named method first, seed s against seed s, per instance; the best
    dispatching rule is the per-instance minimum of FIFO, MOR, SPT and MWKR;
  * intervals: stats_family.hier_boot with a fresh default_rng(11) per quantity, the
    pooled rows on [k, n, 8] (a base instance brings its eight combinations);
  * against the best dispatching rule: Holm over the combinations a level reaches and
    the combined verdict there; the combinations it does not reach and the pooled
    rows carry the interval and the direction only;
  * a combination a level does not reach reads the default level;
  * the project-mix levels are not compared with the default level (other instances);
  * known effects give the expected verdicts and directions;
  * the counts of evaluate --trace_counts (count_decision, count_setup on a stand-in
    environment) and the two measured variables computed from them by hand, the tie
    rule of the best rule's schedule, and "not computed" when a file lacks them;
  * plan puts --trace_counts into its commands only when asked, and evaluate refuses
    a part launched otherwise than its plan;
  * the rule set: infer (and its --check) refuses a rule file whose record states another
    rule set than the mask's (in the exact mask, one without rule_set masked); plan
    --only_rules plans the rule units alone in plan_rules.json, and --dry writes nothing;
    evaluate of such a part skips the files that state masked and refuses, before any
    unit, a rule file of the earlier set; archive-rules moves the earlier rule files of
    the exact mask with their manifest, keeps the policy files and the masked rule files,
    refuses on an existing target, and moves nothing with --dry;
  * refusals (a missing seed file, arms on different instances, a schedule that
    failed check C6, every level without --partial, one seed, a guarded output
    folder, in-place writes in the earlier environment), with no output left;
  * --partial computes the complete cells only and marks its outputs;
  * macro names are Sens plus letters, unique, and a collision is refused; only the
    default level keeps per-combination macros; the supplement table and the macro
    comments use plain words; the tidy CSV has one row per level, comparison and row.

Run from the repository root:  python3 -m pytest tests/test_p87_infer.py -q
Synthetic arrays only, CPU, a few seconds.
"""
import csv
import importlib.util
import json
import os
import shutil
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    _spec = importlib.util.spec_from_file_location(
        'p87_infer_under_test', os.path.join(ROOT, 'scripts', 'p87_sensitivity.py'))
    P87 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(P87)
finally:
    sys.argv = _argv

import stats_family as SF  # noqa: E402
from p89_public_family import superiority_verdict  # noqa: E402

SEEDS = [301, 302, 303]
N = 6                      # instances per combination (the mix levels: N - 2)
B = 200
MASK = 'legacy'


def names_of(level, n=None):
    if P87.LEVELS[level]['family'] == 'mix':
        return [f'mix_{i:03d}' for i in range(N - 2)]
    return [f'instance_{i:03d}' for i in range(N)]


WHO = ('pact', 'spec', 'single') + tuple(P87.RULES)


def noisy(level, regime, who, seed, n):
    """Makespans with a mild effect per method and noise per seed and instance."""
    base = 100.0 + 10.0 * np.arange(n) + 3.0 * P87.R8.index(regime)
    eff = {'pact': 0.99, 'spec': 1.0, 'single': 1.01, 'FIFO': 1.10, 'MOR': 1.12,
           'SPT': 1.02, 'MWKR': 1.05}[who]
    g = np.random.default_rng([list(P87.LEVELS).index(level), P87.R8.index(regime),
                               WHO.index(who), seed or 0])
    return np.round(base * eff * (1 + 0.02 * g.standard_normal(n)), 3)


def write_unit(root, level, regime, label, ms, names, arm='rule', seed=None, c6_ok=None,
               mask=MASK, trace=None, rule_set=None):
    """rule_set: None writes a rule record without the field (as every rule file written
    before the field existed), else the field."""
    p = P87.result_path(root, mask, level, regime, label)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    n = len(ms)
    np.savez(p, makespan=np.asarray(ms, float), wall=np.zeros(n), cpu=np.zeros(n),
             lb0=np.ones(n), util=np.zeros(n), util_max_type=np.zeros(n),
             c6_ok=np.ones(n, bool) if c6_ok is None else np.asarray(c6_ok, bool),
             amch=np.zeros((n, 3), np.int8), **(trace or {}))
    rec = {'arm': arm, 'seed': seed, 'label': label, 'level': level,
           'regime': regime, 'mask': mask, 'instances': names, 'n': n}
    if rule_set is not None and arm == 'rule':
        rec['rule_set'] = rule_set
    with open(p[:-4] + '.json', 'w') as f:
        json.dump(rec, f)


def build_tree(root, gen=noisy, levels=None, seeds=SEEDS, mask=MASK, trace=None,
               rule_set=None):
    """trace(level, regime, who, seed, n) -> dict of the TRACE_FIELDS arrays, or None;
    rule_set: the field of the rule records (None: no field)."""
    tr = trace or (lambda *a: None)
    for level in levels or list(P87.LEVELS):
        names = names_of(level)
        n = len(names)
        for regime in P87.reached(level):
            for arm in ('pact', 'spec', 'single'):
                for s in seeds:
                    write_unit(root, level, regime, 'greedy+' + P87.model_name(arm, regime, s),
                               gen(level, regime, arm, s, n), names, arm, s,
                               mask=mask, trace=tr(level, regime, arm, s, n))
            for r in P87.RULES:
                write_unit(root, level, regime, r, gen(level, regime, r, None, n),
                           names, mask=mask, trace=tr(level, regime, r, None, n),
                           rule_set=rule_set)


def args_for(root, out, **kw):
    a = dict(root=str(root), out=None if out is None else str(out), mask=MASK,
             seeds=list(SEEDS), levels=[],
             partial=False, boot=B)
    a.update(kw)
    return types.SimpleNamespace(**a)


def load(root, level, regime, label):
    with np.load(P87.result_path(str(root), MASK, level, regime, label)) as z:
        return z['makespan'].astype(float)


def arrays(root, level, regime):
    lv = level if regime in P87.reached(level) else 'default'
    P = np.stack([load(root, lv, regime, 'greedy+' + P87.model_name('pact', regime, s))
                  for s in SEEDS])
    S = np.stack([load(root, lv, regime, 'greedy+' + P87.model_name('spec', regime, s))
                  for s in SEEDS])
    C = np.stack([load(root, lv, regime, 'greedy+' + P87.model_name('single', regime, s))
                  for s in SEEDS])
    R = np.stack([load(root, lv, regime, r) for r in P87.RULES])
    return P, S, C, R


@pytest.fixture(autouse=True)
def _names(monkeypatch):
    monkeypatch.setattr(P87, 'level_names', names_of)
    monkeypatch.setitem(P87.ARM_NAMES, 'scheme', 'legacy')


@pytest.fixture(scope='module')
def tree(tmp_path_factory):
    root = tmp_path_factory.mktemp('sens')
    build_tree(str(root))
    return root


@pytest.fixture(scope='module')
def full(tree, tmp_path_factory):
    out = tmp_path_factory.mktemp('out')
    mp = pytest.MonkeyPatch()
    mp.setattr(P87, 'level_names', names_of)
    try:
        res = P87.infer(args_for(tree, out))
    finally:
        mp.undo()
    return res, out


def test_every_level_has_a_letters_only_word_and_a_parameter():
    assert set(P87.LEVEL_WORDS) == set(P87.LEVELS)
    assert all(w.isalpha() for w in P87.LEVEL_WORDS.values())
    assert len(set(P87.LEVEL_WORDS.values())) == len(P87.LEVELS)
    for level in P87.LEVELS:
        par, val = P87.level_parameter(level)
        assert par and (level == 'default' or val != '')


def test_gaps_intervals_and_pairing(full, tree):
    res, _ = full
    x = res['levels']['default']
    for regime in ('S', 'LSW'):
        P, S, C, R = arrays(tree, 'default', regime)
        best = R.min(axis=0)
        for key, d in (('pol_spec', 100 * (P - S) / S), ('ctl_spec', 100 * (C - S) / S),
                       ('pol_rule', 100 * (P - best) / best)):
            hb = SF.hier_boot(d, B=B, seed=11)
            c = x['rows'][key][regime]
            assert c['mean'] == pytest.approx(d.mean(), abs=1e-12)
            assert (c['lo'], c['hi']) == (hb['lo'], hb['hi'])
            assert c['k'] == 3 and c['n'] == N and c['scheme'] == 'hierarchical'
        # seed s against seed s: a permutation of the specialist seeds changes the mean
        dperm = 100 * (P - S[[1, 2, 0]]) / S[[1, 2, 0]]
        assert x['rows']['pol_spec'][regime]['mean'] != pytest.approx(dperm.mean(), abs=1e-9)


def test_pooled_rows_keep_base_instances_together(full, tree):
    res, _ = full
    x = res['levels']['default']
    d3 = np.stack([100 * (a[0] - a[1]) / a[1] for a in
                   (arrays(tree, 'default', r) for r in P87.R8)], axis=2)
    hb = SF.hier_boot(d3, B=B, seed=11)
    c = x['pools']['pool']['pol_spec']
    assert (c['mean'], c['lo'], c['hi'], c['c']) == (pytest.approx(hb['mean']), hb['lo'],
                                                     hb['hi'], 8)
    lag = res['levels']['lag_x050']
    assert lag['pools']['reached']['combos'] == ['L', 'LS', 'LW', 'LSW']
    assert 'reached' not in res['levels']['dur_cv10']['pools']


def test_holm_over_the_reached_combinations(full):
    res, _ = full
    lv = res['levels']['setup_x050']
    rows, rch = lv['rows']['pol_rule'], ['S', 'LS', 'SW', 'LSW']
    assert lv['holm_family'] == rch
    for col in ('p_t', 'p_boot', 'p_t_greater', 'p_boot_greater'):
        adj = SF.holm([rows[r][col] for r in rch])
        assert [rows[r][col + '_holm'] for r in rch] == adj
    for r in rch:
        c = rows[r]
        assert c['holm_family'] == rch
        assert c['verdict'] == superiority_verdict(c['lo'], c['hi'], c['p_t_holm'],
                                                   c['p_boot_holm'], c['p_t_greater_holm'],
                                                   c['p_boot_greater_holm'], 0.05)
    # the combinations the level does not reach: interval and direction only
    base = res['levels']['default']['rows']['pol_rule']
    for r in ('N', 'L', 'W', 'LW'):
        assert 'verdict' not in rows[r] and 'p_t_holm' not in rows[r]
        assert (rows[r]['lo'], rows[r]['hi']) == (base[r]['lo'], base[r]['hi'])
        assert rows[r]['direction'] == SF.direction(rows[r]['lo'], rows[r]['hi'])
    # the default level reaches all eight: Holm over the eight
    adj = SF.holm([base[r]['p_t'] for r in P87.R8])
    assert [base[r]['p_t_holm'] for r in P87.R8] == adj


def test_pooled_rows_against_the_rule_carry_no_verdict(full):
    res, _ = full
    for level in ('default', 'lag_x050', 'mix_steel'):
        for P in res['levels'][level]['pools'].values():
            c = P['pol_rule']
            assert 'verdict' not in c and 'p_t_holm' not in c
            assert c['direction'] == SF.direction(c['lo'], c['hi'])
            assert {'v0.5', 'v1.0'} <= set(P['pol_spec'])


def test_combinations_not_reached_read_the_default_level(full):
    res, _ = full
    lag, base = res['levels']['lag_x050'], res['levels']['default']
    assert lag['not_reached'] == ['N', 'S', 'W', 'SW']
    for r in lag['not_reached']:
        for key in ('pol_spec', 'ctl_spec'):
            assert lag['rows'][key][r]['mean'] == base['rows'][key][r]['mean']
            assert lag['rows'][key][r]['lo'] == base['rows'][key][r]['lo']
    assert lag['counted_over'] == ['L', 'LS', 'LW', 'LSW']
    assert res['levels']['mix_rc']['n']['N'] == N - 2


def test_mix_levels_are_not_compared_with_the_default_level(full):
    res, out = full
    for level in ('mix_rc', 'mix_steel'):
        x = res['levels'][level]
        assert x['changed'] is None and x['changed_note'] == P87.MIX_NOTE
    assert res['changed_not_compared'] == {'mix_rc': P87.MIX_NOTE, 'mix_steel': P87.MIX_NOTE}
    assert 'mix_rc' not in res['changed_over'] and len(res['changed_over']) == 20
    assert all(isinstance(res['levels'][lv]['changed'], dict) for lv in res['changed_over'])
    mac = (out / 'manuscript' / 'macros_sensitivity.tex').read_text()
    assert '\\SensMixConcretePolSpecChangedHalfCount' not in mac
    assert '\\SensLagHalfPolSpecChangedHalfCount' in mac
    assert '% mix_rc: verdict changes from the default level not computed' in mac
    assert '\\newcommand{\\SensChangedOverN}{20}' in mac
    md = (out / 'results' / 'sensitivity_inference.md').read_text()
    assert '- mix_steel: not computed' in md


def test_outputs_names_and_csv(full):
    res, out = full
    mac = (out / 'manuscript' / 'macros_sensitivity.tex').read_text()
    assert mac.startswith('% Generated by scripts/p87_sensitivity.py infer -- do not hand-edit')
    import re
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', mac)
    assert names and all(re.fullmatch(r'Sens[A-Za-z]+', n) for n in names)
    assert len(names) == len(set(names))
    # per-combination rows: all eight for the default level, the reached combinations
    # for a changed level
    for x in ('SensDefaultPolRuleLSWVerdict', 'SensDefaultPolSpecNVOne',
              'SensDefaultCtlSpecSWLo', 'SensLagHalfPolSpecL', 'SensLagHalfPolRuleLSWVerdict',
              'SensMixConcretePolSpecNLo', 'SensSetupHalfCtlSpecSWHi'):
        assert x in names
    for x in ('SensLagHalfPolSpecN', 'SensLagHalfPolRuleSWVerdict', 'SensSetupHalfPolSpecL',
              'SensWinHalfPolSpecLS'):
        assert x not in names
    # pooled rows: verdicts against the specialist, interval and direction against the rule
    for x in ('SensDefaultPolSpecPoolVHalf', 'SensLagHalfPolSpecReachedVOne',
              'SensLagHalfPolRuleReachedDir', 'SensLagHalfPolRulePoolHi',
              'SensMixSteelCtlSpecPoolDir', 'SensLagHalfPolSpecEqOneCount',
              'SensLagHalfPolRuleAheadCount', 'SensPolSpecChangedOneLevels',
              'SensPolRuleChangedLevels'):
        assert x in names
    assert not [x for x in names if re.search(r'PolRule(Pool|Reached)Verdict$', x)]
    assert not [x for x in names if 'PolRule' in x and re.search(r'(Pool|Reached)V', x)]
    assert 'nan' not in mac.lower().replace('non-inferior', '')
    for word in ('placebo', 'regime', 'PACT', ' cell'):
        assert word not in mac
    assert '% The two measured variables are not computed' in mac
    js = json.loads((out / 'results' / 'sensitivity_inference.json').read_text())
    assert js['partial'] is False and js['B'] == B and js['seeds'] == SEEDS
    assert js['measured_variables']['computed'] is False
    assert 'carry no trace counts' in js['measured_variables']['reason']
    with open(out / 'results' / 'sensitivity_inference.csv') as f:
        rows = list(csv.DictReader(f))
    want = sum(3 * (len(x['combos']) + len(x['pools'])) for x in res['levels'].values())
    assert len(rows) == want and set(rows[0]) == set(P87.CSV_FIELDS)
    assert {r['comparison'] for r in rows} == set(P87.COMPARISONS)
    lagN = [r for r in rows if r['level'] == 'lag_x050' and r['row'] == 'N']
    assert {r['reached'] for r in lagN} == {'0'}
    assert all(r['pol_one_pair'] == '' for r in rows)
    assert all(r['verdict'] == '' for r in rows if r['row'] in ('pool', 'reached'))
    assert len(res['levels']) == len(P87.LEVELS)
    assert (out / 'results' / 'sensitivity_inference.md').exists()


def test_supplement_table(full):
    res, out = full
    tab = (out / 'manuscript' / 'tables' / 'sensitivity_table.tex').read_text()
    assert tab.startswith('% auto-generated by scripts/p87_sensitivity.py infer')
    body = [ln for ln in tab.splitlines() if ln.endswith('\\\\') and '&' in ln]
    assert len(body) == 2 + len(P87.LEVELS)          # two header rows, one row per level
    for word in ('regime', 'Regime', 'PACT', 'placebo', 'cell', 'textbf', 'bfseries'):
        assert word not in tab
    for word in ('combination', 'policy', 'specialists', 'best dispatching rule'):
        assert word in tab.lower()
    lag = res['levels']['lag_x050']
    a, b = lag['pools']['reached']['pol_spec'], lag['pools']['reached']['pol_rule']
    row = [ln for ln in body if ln.startswith('lag scale & 0.50 &')]
    assert len(row) == 1
    f2 = P87._stats()[2]
    assert f'${f2(a["mean"])}$ $[{f2(a["lo"])},\\,{f2(a["hi"])}]$' in row[0]
    assert f'${f2(b["mean"])}$ $[{f2(b["lo"])},\\,{f2(b["hi"])}]$' in row[0]
    eq = lag['counts']['pol_spec|1.0']['equivalent']
    assert f'& {eq} of 4 &' in row[0] and row[0].count(' of 4') == 2
    assert [ln for ln in body if ln.startswith('stations charged with setups & 7 of 25 &')]
    assert [ln for ln in body if ln.startswith('project mix & steel &')]
    dflt = [ln for ln in body if ln.startswith('default benchmark & -- &')]
    assert len(dflt) == 1 and ' of 8' in dflt[0]


def _effect(level, regime, who, seed, n):
    """Policy 0.2% shorter than the specialist (seed 301; the other seeds shorter
    still, see test_known_effects), control 2% longer, SPT the best rule at the
    policy's makespan over 0.97, the other rules longer."""
    base = 100.0 + 10.0 * np.arange(n)
    if who == 'spec':
        return base
    if who == 'pact':
        return base * 0.998
    if who == 'single':
        return base * 1.02
    best = base * 0.998 / 0.97
    return {'SPT': best, 'FIFO': best * 1.1, 'MOR': best * 1.2, 'MWKR': best * 1.3}[who]


def test_known_effects(tmp_path):
    root, out = tmp_path / 'tree', tmp_path / 'out'

    def gen(level, regime, who, seed, n):
        x = _effect(level, regime, who, seed, n)
        if who == 'pact':                       # seed-dependent lead over the rule
            k = seed - 301
            x = x * (1 - 0.005 * k)
        return x
    build_tree(str(root), gen=gen, levels=['default'])
    res = P87.infer(args_for(root, out, levels=['default'], partial=True))
    x = res['levels']['default']
    for r in P87.R8:
        assert x['rows']['ctl_spec'][r]['v0.5'] == 'inferior'
        assert x['rows']['ctl_spec'][r]['direction'] == 'behind'
        assert x['rows']['pol_rule'][r]['verdict'] == 'ahead'
        assert x['rows']['pol_rule'][r]['mean'] < -2.9
    # seed 301 is exactly 0.2% shorter than its specialist, the others shorter still
    assert x['rows']['pol_spec']['N']['direction'] == 'ahead'
    assert x['counts']['pol_rule']['ahead'] == 8


def test_refuses_missing_seed_file_and_leaves_no_output(tree, tmp_path):
    root, out = tmp_path / 'tree', tmp_path / 'out'
    shutil.copytree(tree, root)
    p = P87.result_path(str(root), MASK, 'setup_x150', 'LS',
                        'greedy+' + P87.model_name('spec', 'LS', 302))
    os.remove(p)
    stale = out / 'manuscript' / 'macros_sensitivity.tex'
    stale.parent.mkdir(parents=True)
    stale.write_text('% stale\n')
    with pytest.raises(P87.InferRefusal, match='result files missing'):
        P87.infer(args_for(root, out))
    assert not any(q.is_file() for q in out.rglob('*'))
    res = P87.infer(args_for(root, out, partial=True))
    assert res['partial'] and res['missing'] == [os.path.relpath(p, str(root))]
    lv = res['levels']['setup_x150']
    assert 'LS' not in lv['combos'] and 'pool' not in lv['pools'] and 'reached' not in lv['pools']
    assert '% PARTIAL' in (out / 'manuscript' / 'macros_sensitivity.tex').read_text()


def test_refuses_mismatched_instances(tree, tmp_path):
    root = tmp_path / 'tree'
    shutil.copytree(tree, root)
    p = P87.result_path(str(root), MASK, 'tight15', 'W', 'MOR')[:-4] + '.json'
    rec = json.load(open(p))
    rec['instances'] = rec['instances'][::-1]
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P87.InferRefusal, match='different instances'):
        P87.infer(args_for(root, tmp_path / 'out'))


def test_refuses_failed_schedule_check(tree, tmp_path):
    root = tmp_path / 'tree'
    shutil.copytree(tree, root)
    names = names_of('win_x200')
    write_unit(str(root), 'win_x200', 'SW', 'FIFO', np.full(N, 150.0), names,
               c6_ok=[True] * (N - 1) + [False])
    with pytest.raises(P87.InferRefusal, match='check C6'):
        P87.infer(args_for(root, tmp_path / 'out'))
    res = P87.infer(args_for(root, tmp_path / 'out', partial=True))
    assert res['c6_fail_combinations'] == [['win_x200', 'SW']]


def test_record_must_match_its_file(tree, tmp_path):
    root = tmp_path / 'tree'
    shutil.copytree(tree, root)
    p = P87.result_path(str(root), MASK, 'default', 'N', 'SPT')[:-4] + '.json'
    rec = json.load(open(p))
    rec['mask'] = 'exact'
    json.dump(rec, open(p, 'w'))
    with pytest.raises(P87.InferRefusal, match='does not match the file'):
        P87.infer(args_for(root, tmp_path / 'out'))


@pytest.mark.parametrize('kw,msg', [
    (dict(levels=['default', 'lag_x050']), 'needs --partial'),
    (dict(seeds=[301]), 'at least two training seeds'),
    (dict(out=None), 'write only to --out'),
    (dict(out=None, partial=True), 'write only to --out'),
    (dict(out=ROOT), 'repository root'),
    (dict(out=os.path.join(ROOT, 'results', 'sens')), 'paper, results'),
    (dict(out=os.path.join(ROOT, 'manuscript', 'x'), partial=True), 'paper, results'),
    (dict(out=os.path.join(ROOT, 'test_results', 'x'), mask='exact'), 'paper, results'),
])
def test_refusals_before_reading(tree, kw, msg):
    kw = dict({'out': '/nonexistent-never-used'}, **kw)
    out = kw.pop('out')
    with pytest.raises(P87.InferRefusal, match=msg):
        P87.infer(args_for(tree, out, **kw))
    assert not os.path.exists('/nonexistent-never-used')


def test_one_seed_partial_uses_the_instance_bootstrap(tree, tmp_path):
    res = P87.infer(args_for(tree, tmp_path / 'out', seeds=[302], partial=True,
                             levels=['default']))
    c = res['levels']['default']['rows']['pol_spec']['S']
    P, S, _, _ = arrays(tree, 'default', 'S')
    d = 100 * (P[1] - S[1]) / S[1]
    ib = SF.instance_boot(d, B=B, seed=7)
    assert c['scheme'] == 'instance' and (c['lo'], c['hi']) == (ib['lo'], ib['hi'])


def test_partial_skips_levels_with_only_placebo_cells(tree, tmp_path):
    root = tmp_path / 'tree'
    shutil.copytree(tree, root)
    shutil.rmtree(os.path.join(str(root), 'results', MASK, 'lag_x075'))
    res = P87.infer(args_for(root, tmp_path / 'out', partial=True))
    assert 'lag_x075' not in res['levels'] and 'lag_x050' in res['levels']


def test_macro_name_collision_is_refused(tmp_path, monkeypatch):
    (tmp_path / 'manuscript').mkdir()
    (tmp_path / 'manuscript' / 'macros_other.tex').write_text('\\newcommand{\\SensFoo}{1}\n')
    monkeypatch.setattr(P87, 'ROOT_DIR', str(tmp_path))
    outs = P87.infer_outputs(str(tmp_path / 'out'))
    assert P87.infer_check_names(['SensBar'], outs) >= 1
    with pytest.raises(P87.InferRefusal, match='collision'):
        P87.infer_check_names(['SensFoo'], outs)
    with pytest.raises(P87.InferRefusal, match='duplicate'):
        P87.infer_check_names(['SensBar', 'SensBar'], outs)
    with pytest.raises(P87.InferRefusal, match='letters'):
        P87.infer_check_names(['SensBar2'], outs)


def test_check_reads_only(tree, tmp_path, capsys):
    assert P87.infer(args_for(tree, None, check=True))['complete'] is True
    root = tmp_path / 'tree'
    shutil.copytree(tree, root)
    p = P87.result_path(str(root), MASK, 'mix_steel', 'LW', 'SPT')
    os.remove(p)
    r = P87.infer(args_for(root, None, check=True))
    assert r['complete'] is False and r['missing'] == [p]
    assert 'NOT complete' in capsys.readouterr().out
    with pytest.raises(P87.InferRefusal, match='reads only'):
        P87.infer(args_for(root, tmp_path / 'out', check=True))
    assert not (tmp_path / 'out').exists()


# ------------------------------------------------- evaluate --trace_counts and plan
def _fake_env(masked, setup_mult=None):
    """Stand-in for the environment fields the two counters read: one job (0) whose
    candidate is operation 1 of class 2; machine 1 last held class 0."""
    env = types.SimpleNamespace(
        dynamic_pair_mask=np.array([masked]), candidate=np.array([[1, 0]]),
        mch_last_class=np.array([[3, 0]]), op_class=np.array([[0, 2]]),
        true_setup=np.arange(16, dtype=float).reshape(1, 4, 4),
        c_bits=np.array([[1, 1, 0]]))
    if setup_mult is not None:
        env._has_setup_mult = True
        env.setup_mult = np.array([setup_mult])
    return env


def test_counters_on_a_stand_in_environment():
    c = {'decisions': 0, 'one_pair': 0, 'setup': 0.0}
    P87.count_decision(_fake_env([[True, False], [True, True]]), c)    # one admitted pair
    P87.count_decision(_fake_env([[False, False], [True, True]]), c)   # two
    P87.count_decision(_fake_env([[True, True], [False, True]]), c)    # one
    assert (c['decisions'], c['one_pair']) == (3, 2)
    P87.count_setup(_fake_env([[False]]), 0, 1, c)              # true_setup[0, 0, 2] = 2
    assert c['setup'] == 2.0
    P87.count_setup(_fake_env([[False]], setup_mult=[1.0, 0.0]), 0, 1, c)   # factor 0
    P87.count_setup(_fake_env([[False]], setup_mult=[1.0, 1.0]), 0, 0, c)   # class 3 -> 2
    assert c['setup'] == 2.0 + 0.0 + 14.0
    env = _fake_env([[False]])
    env.c_bits = np.array([[1, 0, 0]])                          # no setups charged
    P87.count_setup(env, 0, 1, c)
    assert c['setup'] == 16.0


def _trace(level, regime, who, seed, n):
    """Counts that differ by level, combination, method, seed and instance."""
    i = np.arange(n)
    k = P87.R8.index(regime) + 3 * WHO.index(who) + (seed or 0) % 7
    dec = 40 + 2 * i + k
    return {'n_decisions': dec.astype(np.int64),
            'n_one_pair': ((dec * (k + 1)) % 17).astype(np.int64),
            'setup_sum': 5.0 + 1.5 * i + 0.25 * k,
            'machines_used': (20 + (i + k) % 5).astype(np.int64)}


def _tie(level, regime, who, seed, n):
    """noisy() with FIFO tied with the best rule (SPT) on instance 0."""
    x = noisy(level, regime, who, seed, n)
    if who == 'FIFO':
        x = x.copy()
        x[0] = noisy(level, regime, 'SPT', None, n)[0]
    return x


def test_measured_variables_by_hand(tmp_path):
    root, out = tmp_path / 'tree', tmp_path / 'out'
    build_tree(str(root), gen=_tie, levels=['default', 'setup_x050'], trace=_trace)
    res = P87.infer(args_for(root, out, levels=['setup_x050'], partial=True))
    assert res['measured_variables']['computed'] is True
    for level, regime in (('default', 'N'), ('setup_x050', 'LS')):
        m = res['levels'][level]['measured'][regime]
        P, _, _, R = arrays(root, level, regime)
        T = [_trace(level, regime, 'pact', s, N) for s in SEEDS]
        op = [100 * t['n_one_pair'].sum() / t['n_decisions'].sum() for t in T]
        ss = [100 * np.mean(t['setup_sum'] / (t['machines_used'] * P[k]))
              for k, t in enumerate(T)]
        assert m['pol_one_pair'] == pytest.approx(np.mean(op), abs=1e-12)
        assert m['pol_setup_share'] == pytest.approx(np.mean(ss), abs=1e-12)
        assert m['pol_one_pair_range'] == pytest.approx([min(op), max(op)])
        # the best rule's schedule per instance; the tie on instance 0 goes to FIFO
        b = [next(r for j, r in enumerate(P87.RULES) if R[j, i] == R[:, i].min())
             for i in range(N)]
        assert R[0, 0] == R[2, 0] == R[:, 0].min() and b[0] == 'FIFO'
        TR = {r: _trace(level, regime, r, None, N) for r in P87.RULES}
        dec = sum(TR[b[i]]['n_decisions'][i] for i in range(N))
        one = sum(TR[b[i]]['n_one_pair'][i] for i in range(N))
        sh = np.mean([TR[b[i]]['setup_sum'][i] / (TR[b[i]]['machines_used'][i]
                                                  * R[P87.RULES.index(b[i]), i])
                      for i in range(N)])
        assert m['rule_one_pair'] == pytest.approx(100 * one / dec, abs=1e-12)
        assert m['rule_setup_share'] == pytest.approx(100 * sh, abs=1e-12)
        assert m['best_rule_instances'] == {r: b.count(r) for r in P87.RULES}
    u1 = P87._stats()[5]
    mac = (out / 'manuscript' / 'macros_sensitivity.tex').read_text()
    m = res['levels']['setup_x050']['measured']['LS']
    assert f'\\newcommand{{\\SensSetupHalfPolOnePairLS}}{{{u1(m["pol_one_pair"])}}}' in mac
    assert '\\SensSetupHalfRuleSetupShareLSW' in mac and '\\SensDefaultRuleOnePairN' in mac
    assert '\\SensSetupHalfPolOnePairN}' not in mac          # N is not reached
    assert 'The two measured variables are not computed' not in mac
    with open(out / 'results' / 'sensitivity_inference.csv') as f:
        rows = list(csv.DictReader(f))
    r = [x for x in rows if x['level'] == 'setup_x050' and x['row'] == 'LS'][0]
    assert float(r['rule_setup_share']) == pytest.approx(m['rule_setup_share'], abs=1e-4)
    assert all(x['pol_one_pair'] == '' for x in rows if x['row'] in ('pool', 'reached'))
    assert '## Measured variables' in (out / 'results' / 'sensitivity_inference.md').read_text()


def test_measured_variables_not_computed_when_a_file_lacks_the_counts(tmp_path):
    root, out = tmp_path / 'tree', tmp_path / 'out'
    build_tree(str(root), levels=['default'], trace=_trace)
    names = names_of('default')
    write_unit(str(root), 'default', 'W', 'MOR', noisy('default', 'W', 'MOR', None, N), names)
    res = P87.infer(args_for(root, out, levels=['default'], partial=True))
    mv = res['measured_variables']
    assert mv['computed'] is False and mv['reason'].startswith('1 of the 104 result files')
    assert 'measured' not in res['levels']['default']
    mac = (out / 'manuscript' / 'macros_sensitivity.tex').read_text()
    assert 'OnePair' not in mac and '% PARTIAL The two measured variables are not computed' in mac
    # a count array that does not cover the instances is refused
    p = P87.result_path(str(root), MASK, 'default', 'S', 'SPT')
    write_unit(str(root), 'default', 'S', 'SPT', noisy('default', 'S', 'SPT', None, N), names,
               trace={k: v[:-1] for k, v in _trace('default', 'S', 'SPT', None, N).items()})
    with pytest.raises(P87.InferRefusal, match='trace counts do not cover'):
        P87.infer(args_for(root, out, levels=['default'], partial=True))
    assert os.path.exists(p)


def _plan_args(root, **kw):
    a = dict(root=str(root), levels=['default'], arms=['pact', 'spec', 'single', 'rules'],
             seeds=list(SEEDS), timing='', n=None, parts=2, mask='exact',
             arm_names='xmask', trace_counts=False)
    a.update(kw)
    return types.SimpleNamespace(**a)


def test_plan_prints_the_option_only_when_asked(tmp_path, capsys):
    P87.plan(_plan_args(tmp_path / 'a'))
    off = capsys.readouterr().out
    plan = json.loads((tmp_path / 'a' / 'plan.json').read_text())
    assert 'trace_counts' not in plan and '--trace_counts' not in off
    P87.plan(_plan_args(tmp_path / 'b', trace_counts=True))
    on = capsys.readouterr().out
    plan_on = json.loads((tmp_path / 'b' / 'plan.json').read_text())
    assert plan_on.pop('trace_counts') is True
    cmds = [ln for ln in on.splitlines() if 'p87_sensitivity.py evaluate' in ln]
    assert len(cmds) == 2 and all('--trace_counts ' in ln for ln in cmds)
    plan_on['cost_source'] = plan['cost_source']
    assert plan_on == plan                     # the option changes nothing else


@pytest.mark.parametrize('planned,asked', [(True, False), (False, True)])
def test_evaluate_refuses_a_part_launched_otherwise_than_its_plan(tmp_path, monkeypatch,
                                                                  planned, asked):
    P87.plan(_plan_args(tmp_path, trace_counts=planned, parts=1))
    monkeypatch.setattr(P87, 'Evaluator', lambda mask: None)
    a = types.SimpleNamespace(root=str(tmp_path), part=0, parts=1, mask='exact',
                              arm_names='xmask', trace_counts=asked, force=False)
    with pytest.raises(SystemExit, match='--trace_counts; launch every part the same way'):
        P87.evaluate(a)


# ------------------------------------------------- the rule set of the exact mask
def _rec_without_rule_set(root, mask, level, regime, rule, value=None):
    """Remove the field rule_set of a rule record, or set it to value."""
    p = P87.result_path(str(root), mask, level, regime, rule)[:-4] + '.json'
    rec = json.load(open(p))
    rec.pop('rule_set', None)
    if value is not None:
        rec['rule_set'] = value
    json.dump(rec, open(p, 'w'))


def test_infer_refuses_a_rule_file_of_another_rule_set(tmp_path):
    root, out = tmp_path / 'tree', tmp_path / 'out'
    build_tree(str(root), levels=['default'], mask='exact', rule_set='masked')
    res = P87.infer(args_for(root, out, levels=['default'], partial=True, mask='exact'))
    assert (res['rule_set'], res['rule_function']) == ('masked',
                                                       'common_utils.masked_rule_action')
    _rec_without_rule_set(root, 'exact', 'default', 'LW', 'MWKR')
    with pytest.raises(P87.InferRefusal, match='rule set classical'):
        P87.infer(args_for(root, out, levels=['default'], partial=True, mask='exact'))
    assert not any(q.is_file() for q in out.rglob('*'))
    with pytest.raises(P87.InferRefusal, match='rule set classical'):
        P87.infer(args_for(root, None, mask='exact', check=True))
    # the legacy mask keeps the classical rules: a record without the field passes, one
    # that states masked is refused
    leg = tmp_path / 'legacy'
    build_tree(str(leg), levels=['default'])
    P87.infer(args_for(leg, tmp_path / 'out2', levels=['default'], partial=True))
    _rec_without_rule_set(leg, MASK, 'default', 'N', 'SPT', value='masked')
    with pytest.raises(P87.InferRefusal, match='rule set masked'):
        P87.infer(args_for(leg, tmp_path / 'out2', levels=['default'], partial=True))


def test_plan_of_the_rule_units(tmp_path, capsys):
    kw = dict(levels=['default', 'lag_x050'], parts=3, trace_counts=True, only_rules=True)
    P87.plan(_plan_args(tmp_path, dry=True, **kw))
    dry = capsys.readouterr().out
    assert not any(tmp_path.iterdir())                       # --dry writes nothing
    P87.plan(_plan_args(tmp_path, dry=False, **kw))
    wet = capsys.readouterr().out
    cmds = [ln for ln in wet.splitlines() if 'p87_sensitivity.py evaluate' in ln]
    assert cmds == [ln for ln in dry.splitlines() if 'p87_sensitivity.py evaluate' in ln]
    assert len(cmds) == 3
    assert all('--only_rules --trace_counts ' in ln and f'rules_part_{i}.log' in ln
               for i, ln in enumerate(cmds))
    assert not (tmp_path / 'plan.json').exists()            # the full plan is not replaced
    plan = json.loads((tmp_path / 'plan_rules.json').read_text())
    units = [u for v in plan['assignment'].values() for u in v]
    assert plan['only_rules'] is True and plan['rule_set'] == 'masked'
    assert plan['arms'] == ['rules'] and len(units) == plan['n_units'] == 4 * (8 + 4)
    assert {u[2] for u in units} == {'rule'} and {u[4] for u in units} == set(P87.RULES)
    with pytest.raises(SystemExit, match='--only_rules plans the rule units alone'):
        P87.plan(_plan_args(tmp_path, arms=['pact', 'rules'], **kw))


def test_evaluate_of_the_rule_units_resumes_and_refuses_the_earlier_set(tmp_path, monkeypatch,
                                                                        capsys):
    monkeypatch.setattr(P87, 'Evaluator', lambda mask: types.SimpleNamespace(
        mask=mask, rule_set=P87.RULE_SET[mask]))
    # a skipped unit reads only the instance names of its cell (no instance is loaded)
    monkeypatch.setattr(P87, 'load_cell', lambda root, level, regime, n: [
        (x, None, None, None, None) for x in names_of(level)])
    P87.plan(_plan_args(tmp_path, parts=1, trace_counts=True, only_rules=True))
    names = names_of('default')
    for regime in P87.R8:
        for r in P87.RULES:
            write_unit(str(tmp_path), 'default', regime, r, noisy('default', regime, r, None, N),
                       names, mask='exact', trace=_trace('default', regime, r, None, N),
                       rule_set='masked')
    capsys.readouterr()
    a = types.SimpleNamespace(root=str(tmp_path), part=0, parts=1, mask='exact',
                              arm_names='xmask', trace_counts=True, force=False,
                              only_rules=True)
    P87.evaluate(a)
    out = capsys.readouterr().out
    assert out.count(' skip wall=') == 32 and ' done ' not in out
    with pytest.raises(SystemExit, match='plan.json not found'):
        P87.evaluate(types.SimpleNamespace(**dict(vars(a), only_rules=False)))
    _rec_without_rule_set(tmp_path, 'exact', 'default', 'W', 'MOR')
    with pytest.raises(SystemExit,
                       match='(?s)^REFUSING: 1 rule result files .*rule set classical'):
        P87.evaluate(a)


def _files(root):
    return sorted(os.path.relpath(os.path.join(d, f), str(root))
                  for d, _, fs in os.walk(str(root)) for f in fs)


def test_archive_moves_the_earlier_rule_files(tmp_path, capsys):
    root = tmp_path / 'tree'
    build_tree(str(root), levels=['default', 'lag_x050'], mask='exact')   # no rule_set field
    write_unit(str(root), 'lag_x050', 'L', 'SPT', noisy('lag_x050', 'L', 'SPT', None, N),
               names_of('lag_x050'), mask='exact', rule_set='masked')
    def arch(**kw):
        return types.SimpleNamespace(**dict(dict(root=str(root), mask='exact', dry=False), **kw))
    src, dst = os.path.join('results', 'exact'), os.path.join('results', P87.ARCHIVE_RULES)
    before = _files(root)
    n_rules = 4 * (8 + 4) - 1
    r = P87.archive_rules(arch(dry=True))
    assert _files(root) == before                            # --dry moves nothing
    assert r['dry'] and len(r['moved']) == 2 * n_rules and r['kept_masked'] == 1
    assert f'would move {2 * n_rules} files' in capsys.readouterr().out
    r = P87.archive_rules(arch())
    after = _files(root)
    gone = [f for f in before if f not in after]
    assert len(gone) == 2 * n_rules and f'moved {2 * n_rules} files' in capsys.readouterr().out
    for f in gone:
        assert f.startswith(src + os.sep)
        assert os.path.basename(f) in {r + e for r in P87.RULES for e in ('.npz', '.json')}
        assert os.path.join(dst, os.path.relpath(f, src)) in after
    man = json.load(open(root / dst / 'manifest.json'))
    assert sorted(e['from'] for e in man['files']) == gone
    for e in man['files']:
        assert P87.sha256_file(str(root / e['to'])) == e['sha256']
    left = [f for f in after if f.startswith(src + os.sep)]
    assert len(left) == len(before) - 2 * n_rules               # policy files and masked SPT
    assert os.path.join(src, 'lag_x050', 'L', 'SPT.npz') in left
    assert P87.archive_rules(arch())['moved'] == []             # nothing left to move
    # a file of the earlier set whose target exists: refused, nothing moved
    write_unit(str(root), 'default', 'N', 'FIFO', noisy('default', 'N', 'FIFO', None, N),
               names_of('default'), mask='exact')
    before = _files(root)
    for dry in (True, False):
        with pytest.raises(SystemExit, match='target files exist.*nothing was moved'):
            P87.archive_rules(arch(dry=dry))
    assert _files(root) == before
    with pytest.raises(SystemExit, match='--mask exact'):
        P87.archive_rules(arch(mask='legacy'))
