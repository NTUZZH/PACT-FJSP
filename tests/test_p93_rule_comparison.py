"""Tests for scripts/p93_rule_comparison.py and its plan, scripts/rulecmp_plan.py,
on synthetic arrays only (no rollout, no solver).

Run from the repository root:  python -m pytest tests/test_p93_rule_comparison.py -q
Every file is written under a temporary folder.
"""
import ast
import itertools
import json
import os
import re
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import p93_rule_comparison as P93  # noqa: E402
import rulecmp_plan as PLAN  # noqa: E402
from p90_public_rule_comparison import best_of_k, p_shorter_equal  # noqa: E402

N_INST = 4
OTHER_SET, OTHER_NAMES = 'pub_Hurink_edata', ['abz5', 'abz6', 'ft06']   # orb7 left out
VALI_SIZE = 2           # the runs validate on the first 2 of the N_INST validation records
TEST_K = 32
R8 = list(REG.COMBOS)
RULES = PLAN.RULES
B = 200


# ----------------------------------------------------------------- estimator
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_best_of_k_equals_brute_force(seed):
    """The order-statistic estimate equals the mean of the minimum over every
    k-subset of the K draws, ties included; so do the shares shorter and equal."""
    rng = np.random.default_rng(seed)
    x = rng.integers(10, 16, 8).astype(float)          # K = 8 with ties
    for k in range(1, len(x) + 1):
        subsets = [min(c) for c in itertools.combinations(x, k)]
        assert best_of_k(x, k) == pytest.approx(np.mean(subsets), abs=1e-12)
        for r in (11.0, 12.5, 13.0):
            lt, eq = p_shorter_equal(x, k, r)
            assert lt == pytest.approx(np.mean([m < r for m in subsets]), abs=1e-12)
            assert eq == pytest.approx(np.mean([m == r for m in subsets]), abs=1e-12)


def test_best_k_and_greedy_plus():
    """best_k applies the estimator per seed and instance; the greedy schedule plus
    k sampled ones is the best of k of min(sample, greedy)."""
    rng = np.random.default_rng(3)
    S = rng.integers(10, 30, (2, 3, 6)).astype(float)
    g = rng.integers(10, 30, (2, 3)).astype(float)
    bk = P93.best_k(S, 3)
    assert bk.shape == (2, 3)
    assert bk[1, 2] == pytest.approx(best_of_k(S[1, 2], 3))
    gp = P93.best_k(np.minimum(S, g[:, :, None]), 3)
    brute = np.mean([min(min(c), g[0, 1]) for c in itertools.combinations(S[0, 1], 3)])
    assert gp[0, 1] == pytest.approx(brute)


# ----------------------------------------------------------------- selection
def test_select_rule_lowest_mean_first_on_tie():
    assert RULES[P93.select_rule([5.0, 4.0, 6.0, 4.5])] == 'MOR'
    assert RULES[P93.select_rule([5.0, 4.0, 4.0, 4.0])] == 'MOR'      # tie: first in order
    assert RULES[P93.select_rule([3.0, 3.0, 3.0, 3.0])] == 'FIFO'


def test_trace_definitions():
    """Single-pair share pooled over decisions; setup share per instance over
    (machines used x makespan), averaged over instances."""
    dec = [np.array([[1, 1, 1], [2, 2, 1], [1, 1, 1], [3, 2, 2]]),
           np.array([[1, 1, 1], [4, 2, 2], [2, 1, 2], [1, 1, 1]])]
    # machine, processing, setup, start, previous class, class, job
    ops = [np.array([[0, 3, 2, 0, 4, 0, 0], [0, 2, 0, 5, 0, 0, 1], [1, 4, 1, 0, 4, 1, 0],
                     [1, 1, 0, 5, 1, 1, 1]], float),
           np.array([[0, 3, 0, 0, 4, 0, 0], [2, 2, 0, 0, 4, 0, 1], [2, 4, 0, 2, 0, 1, 0],
                     [0, 1, 0, 3, 0, 1, 1]], float)]
    z = dict(decisions=np.array(dec, dtype=object), ops=np.array(ops, dtype=object),
             makespan=np.array([10.0, 8.0]))
    t = P93.trace_stats(z)
    assert t['one_pair'] == pytest.approx(100 * 4 / 8)
    assert t['setup_share'] == pytest.approx(100 * np.mean([3 / (2 * 10), 0 / (2 * 8)]))


def test_k_setting(monkeypatch):
    """K comes from the argument, else RULECMP_K, else 32; it must be an integer of
    at least 16."""
    monkeypatch.delenv(PLAN.K_ENV, raising=False)
    assert PLAN.k_setting() == PLAN.K_DEFAULT == 32
    monkeypatch.setenv(PLAN.K_ENV, '100')
    assert PLAN.k_setting() == 100 and PLAN.k_setting('20') == 20
    for bad in ('8', '15', 'x', '32.5'):
        with pytest.raises(SystemExit):
            PLAN.k_setting(bad)
    assert os.path.basename(PLAN.Family('public', K=20).sample_path(
        PLAN.Family('public', K=20).runs[0], 'S')).startswith('sample20+')


def test_verdict_rule():
    """Ahead needs the interval below zero and both Holm-adjusted p for shorter below
    0.05; behind is the mirror image; everything else is not distinguishable."""
    base = dict(lo=-2.0, hi=-0.5, p_t_holm=0.01, p_boot_holm=0.02, p_t_greater_holm=1.0,
                p_boot_greater_holm=1.0)
    assert P93.verdict(base) == 'ahead'
    assert P93.verdict({**base, 'p_boot_holm': 0.18}) == 'not distinguishable'
    assert P93.verdict({**base, 'p_t_holm': 0.06}) == 'not distinguishable'
    assert P93.verdict({**base, 'hi': 0.3}) == 'not distinguishable'
    beh = dict(lo=0.4, hi=2.0, p_t_holm=1.0, p_boot_holm=1.0, p_t_greater_holm=0.001,
               p_boot_greater_holm=0.004)
    assert P93.verdict(beh) == 'behind'
    assert P93.verdict({**beh, 'p_t_greater_holm': 0.2}) == 'not distinguishable'
    assert P93.verdict({**beh, 'lo': -0.1}) == 'not distinguishable'


# ----------------------------------------------------------------- synthetic tree
def write_summary(folder, arrays):
    n = len(next(iter(arrays.values())))
    L = ['# FAMILY eval — rules_xmask', '', 'mask: first_mask=exact mask_ties=exact', '',
         '| regime | method | mean | std | feasible | gap(cov) | gap to best known (N only) |',
         '|---|---|---|---|---|---|---|']
    L += [f'| {R} | {q} | {v.mean():.1f} | {v.std():.1f} | {n}/{n} | n/a | n/a |'
          for (R, q), v in arrays.items()]
    with open(os.path.join(folder, 'summary_rules_xmask.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def save_result(path, ms):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([np.asarray(ms, float), np.zeros(len(ms))], axis=1))


def trace_file(path, names, ms, rng):
    dec, ops = [], []
    for _ in names:
        dec.append(np.stack([rng.integers(1, 4, 5), np.ones(5), np.ones(5)], axis=1))
        o = np.zeros((5, 7))
        o[:, 0] = [0, 1, 0, 2, 1]
        o[:, 1] = rng.integers(1, 9, 5)
        o[:, 2] = rng.integers(0, 3, 5)
        ops.append(o)
    np.savez(path, instances=np.array(names), makespan=np.asarray(ms, float),
             seconds=np.zeros(len(names)), decisions=np.array(dec, dtype=object),
             ops=np.array(ops, dtype=object))


def write_records(root, data, names, old, excluded=()):
    for R in R8:
        d = os.path.join(root, data, 'desc', R)
        os.makedirs(d)
        for nm in names:
            p = os.path.join(d, nm + '.desc.json')
            with open(p, 'w') as f:
                f.write('{}')
            os.utime(p, (old, old))
    with open(os.path.join(root, data, 'dataset_meta.json'), 'w') as f:
        json.dump(dict(n_instances=len(names), instances=list(names),
                       excluded=[dict(instance=x + '.fjs') for x in excluded]), f)


def build_tree(tmp_path, setitem):
    """A complete synthetic public-family tree: records, checkpoints, stored arrays
    and every rulecmp_* output of the headline set, with SPT the lowest validation
    mean in every combination and the records and checkpoints older than every
    array. `setitem(dict, key, value)` sets the expected record counts of the
    synthetic sets."""
    setitem(PLAN.PUB_SET_INFO, PLAN.HEADLINE, ('Sdten', N_INST, ()))
    setitem(PLAN.PUB_SET_INFO, OTHER_SET, ('Edata', len(OTHER_NAMES), ('orb7',)))
    fam = PLAN.Family('public', K=TEST_K)
    names = [f'inst{i:02d}' for i in range(N_INST)]
    old = time.time() - 10000
    root = str(tmp_path / 'repo')
    write_records(root, fam.data, names, old)
    write_records(root, fam.vali_data, names, old)
    for r in fam.runs:
        for p in (r.checkpoint, r.config, r.log):
            p = os.path.join(root, p)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                f.write(json.dumps(vali_config(fam)) if p.endswith('.json') else '')
            os.utime(p, (old + 10, old + 10))
    rng = np.random.default_rng(0)
    base = os.path.join(root, fam.folder)
    stored, greedy = fill_set(root, fam, names, rng)
    d = fam.vali_folder(base)
    arrays = {}
    for R in R8:
        for q in RULES:
            arrays[(R, q)] = (90 if q == 'SPT' else 120) + rng.integers(0, 5, N_INST)
            if R == 'W':
                # SPT is lowest on the first VALI_SIZE records, MOR on all of them
                arrays[(R, q)] = np.array({'SPT': [90, 90, 200, 200], 'MOR': [120, 120, 100, 100]}
                                          .get(q, [130, 130, 130, 130]))
            save_result(os.path.join(d, f'Result_{q}_{R}.npy'), arrays[(R, q)])
    write_summary(d, arrays)
    return dict(root=root, out=str(tmp_path / 'out'), fam=fam, base=base,
                stored=stored, greedy=greedy, old=old)


def fill_set(root, fam, names, rng):
    """Stored rule and greedy arrays of one public test set and every rulecmp_*
    output except the validation rules: sampled schedules of fam.runs, traces of the
    released seeds and the rules, rule arrays with the tie-break seeds 51-53."""
    n = len(names)
    base = os.path.join(root, fam.folder)
    stored, greedy = {}, {}
    for R in R8:
        for q in RULES:
            stored[(q, R)] = 100 + 10 * np.arange(n) + rng.integers(0, 20, n)
            save_result(fam.rule_path(q, R, root), stored[(q, R)])
    for r in fam.runs:
        for R in r.regimes:
            g = 100 + 10 * np.arange(n) + rng.integers(-5, 25, n)
            greedy[(r.model, R)] = g
            save_result(fam.greedy_path(r, R, root), g)
            S = np.maximum(1, g[:, None] + rng.integers(-15, 15, (n, fam.K)))
            p = fam.sample_path(r, R, base)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, instances=np.array(names), makespan=S.astype(float),
                     feasible=np.ones(S.shape, bool), seconds=np.zeros(n), K=fam.K,
                     seed_test=PLAN.STORED_TIE_SEED)
    os.makedirs(fam.sub('trace', base))
    for r in fam.released():
        for R in R8:
            trace_file(fam.trace_path(r.model, R, base), names, greedy[(r.model, R)], rng)
    for q in RULES:
        for R in R8:
            trace_file(fam.trace_path(q, R, base), names, stored[(q, R)], rng)
    for t in PLAN.TIE_SEEDS[1:]:
        d = fam.ties_dir(t, base, root)
        arrays = {}
        for R in R8:
            for q in RULES:
                arrays[(R, q)] = stored[(q, R)] + rng.integers(-3, 4, n)
                save_result(os.path.join(d, f'Result_{q}_{R}.npy'), arrays[(R, q)])
        write_summary(d, arrays)
    return stored, greedy


@pytest.fixture
def tree(tmp_path, monkeypatch):
    return build_tree(tmp_path, monkeypatch.setitem)


@pytest.fixture
def two_sets(tree):
    """The headline tree plus a second public test set (released policy only), with
    records that leave orb7 out as the Hurink sets do."""
    fam2 = PLAN.Family('public', K=TEST_K, set_name=OTHER_SET)
    write_records(tree['root'], fam2.data, OTHER_NAMES, tree['old'], excluded=('orb7',))
    stored2, greedy2 = fill_set(tree['root'], fam2, OTHER_NAMES, np.random.default_rng(5))
    return dict(tree, fam2=fam2, stored2=stored2, greedy2=greedy2)


def vali_config(fam, **change):
    c = dict(family_base='sd2', n_j=10, n_m=5, vali_size=VALI_SIZE, seed_train_vali_datagen=100,
             ppvc_mix='mixed', ppvc_factory='default', dr_shift='', pub_vali_dir=fam.vali_data)
    c.update(change)
    return c


def run93(t, *extra):
    return P93.main(['--family', 'public', '--root', t['root'], '--out', t['out'],
                     '--boot', str(B), '--K', str(TEST_K), *extra])


def macro_file(t):
    return os.path.join(t['out'], 'manuscript', 'macros_rulecmp_public.tex')


def test_complete_tree_writes_valid_macros(tree):
    assert run93(tree) == 0
    txt = open(macro_file(tree)).read()
    assert txt.startswith('% Generated by scripts/p93_rule_comparison.py -- do not hand-edit')
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert names and len(names) == len(set(names))
    assert all(re.fullmatch(r'Rcp[A-Za-z]+', n) for n in names), \
        [n for n in names if not re.fullmatch(r'Rcp[A-Za-z]+', n)][:5]
    for ln in txt.splitlines():
        code = ln.split('  %')[0]
        assert code.count('{') == code.count('}')
    vals = dict(re.findall(r'\\newcommand\{\\(Rcp[A-Za-z]+)\}\{([^}]*)\}', txt))
    for R in R8:
        assert vals[f'RcpValiRule{R}'] == 'SPT'          # on the first VALI_SIZE records
    assert vals['RcpValiN'] == str(VALI_SIZE) and vals['RcpSampleK'] == str(TEST_K)
    verdicts = {v for k, v in vals.items() if k.endswith('Verdict')}
    assert verdicts and verdicts <= set(P93.VERDICTS)
    for nm in ('PT', 'PBoot', 'PHolm', 'PBootHolm', 'PTBehind', 'PBootBehind', 'PHolmBehind',
               'PBootHolmBehind'):
        assert f'RcpPolGreedyValiRuleS{nm}' in vals
    n_v = sum(int(vals[f'RcpPolSampleFour{w}N']) for w in ('Ahead', 'Behind', 'NotDist'))
    assert n_v == 8
    # form (a) of the policy equals the hand computation
    fam = tree['fam']
    g = np.stack([tree['greedy'][(r.model, 'LS')] for r in fam.released()]).astype(float)
    best4 = np.min(np.stack([tree['stored'][(q, 'LS')] for q in RULES]), axis=0)
    want = (100.0 * (g - best4) / best4).mean()
    assert vals['RcpPolGreedyBestFourLS'] == P93.f2(want)
    with open(os.path.join(tree['out'], 'results', 'rulecmp_public.json')) as f:
        js = json.load(f)
    assert js['settings']['boot'] == B and js['settings']['n_test'] == N_INST
    assert js['settings']['K'] == TEST_K
    assert js['holm']['released|sample16']['family_size'] == 8
    vw = js['combos']['W']['vali']
    assert vw['rule'] == 'SPT' and vw['n'] == VALI_SIZE
    assert vw['all_records']['rule'] == 'MOR' and vw['all_records']['n'] == N_INST
    # form (c) uses the rule of the first VALI_SIZE records
    g = np.stack([tree['greedy'][(r.model, 'W')] for r in fam.released()]).astype(float)
    spt = tree['stored'][('SPT', 'W')].astype(float)
    assert js['combos']['W']['arms']['released']['greedy_vali']['mean'] == \
        pytest.approx((100.0 * (g - spt) / spt).mean())


def test_refuses_on_missing_array(tree):
    assert run93(tree) == 0
    fam = tree['fam']
    r = fam.run_of('control', None, 303)
    os.remove(fam.sample_path(r, 'LW', tree['base']))
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'REFUSING' in str(e.value) and 'missing' in str(e.value)
    assert not os.path.exists(macro_file(tree))          # the earlier file is removed


def test_refuses_on_wrong_shape(tree):
    fam = tree['fam']
    r = fam.run_of('specialist', 'SW', 302)
    save_result(fam.greedy_path(r, 'SW', tree['root']), np.arange(1, N_INST))
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'shape' in str(e.value)
    assert not os.path.exists(macro_file(tree))


def test_refuses_on_array_older_than_checkpoint(tree):
    fam = tree['fam']
    r = fam.run_of('released', None, 305)
    p = fam.sample_path(r, 'S', tree['base'])
    os.utime(p, (tree['old'] + 5, tree['old'] + 5))
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'older than its checkpoint' in str(e.value)


def test_refuses_on_trace_that_does_not_reproduce(tree):
    fam = tree['fam']
    p = fam.trace_path('MWKR', 'LSW', tree['base'])
    z = dict(np.load(p, allow_pickle=True))
    z['makespan'] = z['makespan'] + 1
    np.savez(p, **z)
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'differ from the stored array' in str(e.value)


@pytest.mark.parametrize('change', [dict(pub_vali_dir='data/PUBLIC/family/other_vali'),
                                    dict(vali_size=3)])
def test_refuses_when_a_run_validated_elsewhere(tree, change):
    fam = tree['fam']
    r = fam.run_of('control', None, 302)
    with open(os.path.join(tree['root'], r.config), 'w') as f:
        json.dump(vali_config(fam, **change), f)
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'do not share one validation set' in str(e.value)
    assert next(iter(change)) in str(e.value)


def test_refuses_when_every_run_validated_on_another_folder(tree):
    fam = tree['fam']
    for r in fam.runs:
        with open(os.path.join(tree['root'], r.config), 'w') as f:
            json.dump(vali_config(fam, pub_vali_dir='data/PUBLIC/family/other_vali'), f)
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'validated on' in str(e.value)


def test_refuses_on_sampled_K_mismatch(tree):
    """A sampled file under the name of K = 32 that holds 100 draws per instance."""
    fam = tree['fam']
    r = fam.run_of('released', None, 302)
    p = fam.sample_path(r, 'LS', tree['base'])
    z = dict(np.load(p))
    z['makespan'] = np.tile(z['makespan'], (1, 4))[:, :100]
    z['feasible'] = np.ones(z['makespan'].shape, bool)
    z['K'] = 100
    np.savez(p, **z)
    with pytest.raises(SystemExit) as e:
        run93(tree)
    assert 'K = 100' in str(e.value) and 'not K = 32' in str(e.value)
    assert not os.path.exists(macro_file(tree))
    with pytest.raises(SystemExit) as e:          # asked for K = 8: refused before any read
        P93.main(['--family', 'public', '--root', tree['root'], '--out', tree['out'],
                  '--K', '8'])
    assert 'at least 16' in str(e.value)


def test_partial_skips_and_writes_no_macros(tree):
    fam = tree['fam']
    import shutil
    shutil.rmtree(os.path.dirname(fam.ties_dir(52, tree['base'], tree['root'])))
    assert run93(tree, '--partial') == 0
    assert not os.path.exists(macro_file(tree))
    with open(os.path.join(tree['out'], 'results', 'rulecmp_public.json')) as f:
        js = json.load(f)
    assert js['missing'] and 'sample16' not in js['combos']['S']['arms']['released']
    assert 'sample4' in js['combos']['S']['arms']['released']


# ----------------------------------------------------------------- plan
def test_plan_statuses(tree, monkeypatch):
    monkeypatch.setattr(PLAN, 'ROOT', tree['root'])
    fam = tree['fam']
    js = PLAN.jobs(fam, ratio=1.0)
    count = {k: sum(j.kind == k for j in js) for k in PLAN.KINDS}
    assert count == {'sample': 120, 'trace': 9, 'ties': 3, 'vali': 1}
    assert all(j.status == 'done' for j in js), [(j.id, j.reason) for j in js if j.status != 'done'][:3]
    r = fam.run_of('released', None, 301)
    os.remove(fam.sample_path(r, 'N'))
    bad = fam.sample_path(r, 'L')
    np.savez(bad, instances=np.array(['x']), makespan=np.ones((1, 3)))
    st = {j.id: j for j in PLAN.jobs(fam, ratio=1.0, kinds=('sample',))}
    assert st[f'sample/{r.model}_N'].status == 'open'
    j = st[f'sample/{r.model}_L']
    assert j.status == 'invalid'
    PLAN.move_invalid(j)
    assert not os.path.exists(bad)
    assert os.listdir(os.path.join(os.path.dirname(bad), 'invalid'))


def test_plan_blocks_without_checkpoint(tree, monkeypatch):
    monkeypatch.setattr(PLAN, 'ROOT', tree['root'])
    fam = tree['fam']
    r = fam.run_of('specialist', 'LW', 304)
    os.remove(os.path.join(tree['root'], r.checkpoint))
    os.remove(fam.sample_path(r, 'LW'))
    st = {j.id: j for j in PLAN.jobs(fam, ratio=1.0, kinds=('sample',))}
    assert st[f'sample/{r.model}_LW'].status == 'blocked'
    miss = PLAN.missing_inputs(fam)
    assert miss['checkpoints'] == [r.checkpoint]


# ----------------------------------------------------------------- public test sets
def test_set_names_and_tags():
    """Every public test set: label, macro prefix with the p89 tag, own result
    folder, released policy only by default; the headline set as before."""
    head = PLAN.Family('public', K=TEST_K)
    assert (head.label, head.macro, head.arms) == ('public', 'Rcp', PLAN.ARMS)
    assert head.folder == f'test_results/PUBFAMILY/{PLAN.HEADLINE}_xmask'
    for st, tag in (('pub_SD2_test_20x10+mix', 'Sdtwenty'), ('pub_Brandimarte', 'Brand'),
                    ('pub_Hurink_edata', 'Edata'), ('pub_Hurink_rdata', 'Rdata'),
                    ('pub_Hurink_vdata', 'Vdata')):
        f = PLAN.Family('public', K=TEST_K, set_name=st)
        assert (f.label, f.macro, f.arms) == (f'public_{tag.lower()}', f'Rcp{tag}', ('released',))
        assert f.folder == f'test_results/PUBFAMILY/{st}_xmask'
        assert {r.arm for r in f.runs} == {'released'} and len(f.runs) == 5
        assert f.vali_folder() == PLAN.Family('public', K=TEST_K).vali_folder()
        assert PLAN.label_of('public', st) == f.label
        g = PLAN.Family('public', K=TEST_K, set_name=st, arms=['released', 'specialist'])
        assert len(g.runs) == 45 and g.arms == ('released', 'specialist')
    for bad in (dict(set_name='pub_SD2_vali_10x5+mix'), dict(arms=['control'])):
        with pytest.raises(SystemExit):
            PLAN.Family('public', K=TEST_K, **bad)


def test_set_table_matches_p89():
    """The tags, record counts and exclusions are the SETS table of p89 (read from
    its source, not imported) and cover arm_registry.PUB_SETS."""
    src = open(os.path.join(ROOT, 'scripts', 'p89_public_family.py')).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and
                any(getattr(t, 'id', None) == 'SETS' for t in n.targets))
    p89 = ast.literal_eval(node.value)
    assert set(p89) == set(PLAN.PUB_SET_INFO) == set(REG.PUB_SETS)
    for st, (tag, _, n, excl) in p89.items():
        assert PLAN.PUB_SET_INFO[st] == (tag, n, tuple(excl))


def test_second_set_macros_without_collision(two_sets):
    """Both macro files written into one manuscript folder: the second set's names all
    carry its tag, hold no control or specialist macro, reuse the headline selection
    of the validation rule on the set's own rule arrays, and collide with nothing."""
    t = two_sets
    t = dict(t, out=t['root'])             # p93 checks collisions against <root>/manuscript
    assert run93(t) == 0
    assert run93(t, '--set', OTHER_SET) == 0
    a = open(os.path.join(t['root'], 'manuscript', 'macros_rulecmp_public.tex')).read()
    b = open(os.path.join(t['root'], 'manuscript', 'macros_rulecmp_public_edata.tex')).read()
    na = re.findall(r'\\newcommand\{\\([^}]*)\}', a)
    nb = re.findall(r'\\newcommand\{\\([^}]*)\}', b)
    assert nb and not set(na) & set(nb) and len(nb) == len(set(nb))
    assert all(re.fullmatch(r'RcpEdata[A-Za-z]+', n) for n in nb)
    assert not any(n.startswith(('RcpEdataCtl', 'RcpEdataSpec')) for n in nb)
    assert any(n.startswith('RcpEdataPol') for n in nb)
    vb = dict(re.findall(r'\\newcommand\{\\(RcpEdata[A-Za-z]+)\}\{([^}]*)\}', b))
    assert all(vb[f'RcpEdataValiRule{R}'] == 'SPT' for R in R8)
    assert vb['RcpEdataInstN'] == str(len(OTHER_NAMES))
    with open(os.path.join(t['root'], 'results', 'rulecmp_public_edata.json')) as f:
        js = json.load(f)
    assert js['settings']['test_set'] == OTHER_SET and js['settings']['arms'] == ['released']
    g = np.stack([t['greedy2'][(r.model, 'W')] for r in t['fam2'].released()]).astype(float)
    spt = t['stored2'][('SPT', 'W')].astype(float)
    assert js['combos']['W']['arms']['released']['greedy_vali']['mean'] == \
        pytest.approx((100.0 * (g - spt) / spt).mean())
    # a name shared with another macro file is refused
    with open(os.path.join(t['root'], 'manuscript', 'macros_other.tex'), 'w') as f:
        f.write('\\newcommand{\\RcpEdataSeedN}{5}\n')
    with pytest.raises(SystemExit) as e:
        run93(t, '--set', OTHER_SET)
    assert 'collision' in str(e.value)


def test_second_set_refuses_on_records_and_headline_validation(two_sets):
    t = two_sets
    meta = os.path.join(t['root'], t['fam2'].data, 'dataset_meta.json')
    m = json.load(open(meta))
    json.dump(dict(m, excluded=[]), open(meta, 'w'))
    with pytest.raises(SystemExit) as e:
        run93(t, '--set', OTHER_SET)
    assert 'exclusions' in str(e.value)
    json.dump(m, open(meta, 'w'))
    import shutil
    shutil.rmtree(t['fam'].vali_folder(t['base']))      # the headline validation rules
    with pytest.raises(SystemExit) as e:
        run93(t, '--set', OTHER_SET)
    assert 'missing' in str(e.value) and 'rulecmp_vali' in str(e.value)


def test_plan_second_set(two_sets, monkeypatch):
    monkeypatch.setattr(PLAN, 'ROOT', two_sets['root'])
    fam2 = two_sets['fam2']
    js = PLAN.jobs(fam2, ratio=1.0)
    count = {k: sum(j.kind == k for j in js) for k in PLAN.KINDS}
    assert count == {'sample': 40, 'trace': 9, 'ties': 3, 'vali': 0}
    assert all(j.status == 'done' for j in js), [(j.id, j.reason) for j in js if j.status != 'done'][:3]
    assert all(fam2.folder in PLAN.rel(j.outputs[0]) for j in js)
    vr, _ = PLAN.validation_rules(fam2)
    assert vr == {R: 'SPT' for R in R8}
