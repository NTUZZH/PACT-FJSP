"""Tests for scripts/p141_mask_choice_validation.py, on small synthetic inputs.

  * the validation set is rebuilt from the seeds as train_family.py builds it: base instance i
    from the seed seed_train_vali_datagen + i, the descriptor of each combination from the seed
    seed_train_vali_datagen * 1000 + i, at most 20 instances; the public set takes the first
    names of the sorted records of N, the base instance from the record of N and the descriptor
    of each combination from its own record; train_family.py holds the construction this
    rebuild follows;
  * the best of four is the per-instance minimum of the four rule makespans, on cases whose
    rollouts return fixed makespans, and a block rule that differs from nondelay without setups
    refuses;
  * the direction words, and the words same and different;
  * the MWKR check passes when every rebuilt mean prints as its logged value and refuses
    otherwise; the logged values are read from the first line of the training log;
  * the test-instance values are read from the record of p107 and refuse when the macro file
    of p107 disagrees;
  * the macro names are the prefix plus letters, unique, every line carries a comment, and a
    name defined in another macro file refuses.

Run from the repository root:
  python -B -m pytest tests/test_p141_mask_choice_validation.py -q -p no:cacheprovider
CPU only; nothing is written outside the temporary folder of pytest.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

from conftest import module_environment  # noqa: E402

MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact')

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401  (parses argv at import)
    _spec = importlib.util.spec_from_file_location(
        'p141_mask_choice_validation',
        os.path.join(ROOT, 'scripts', 'p141_mask_choice_validation.py'))
    M = importlib.util.module_from_spec(_spec)
    sys.modules['p141_mask_choice_validation'] = M
    _spec.loader.exec_module(M)
finally:
    sys.argv = _argv
configs.device = 'cpu'


@pytest.fixture(scope='module', autouse=True)
def _cpu():
    M.P101._cpu_torch()


def factory_cfg(**kw):
    cfg = dict(family_base='ppvc', n_j=2, n_m=25, vali_size=3, seed_train_vali_datagen=7,
               seed_test=50, ppvc_mix='mixed', ppvc_factory='default', dr_shift='',
               pub_vali_dir='')
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# the validation set
# ---------------------------------------------------------------------------

def test_factory_set_is_rebuilt_from_the_seeds_of_the_training_code():
    from constraint_family.descriptor import make_descriptor
    from ppvc_instance_generator import DEFAULT_FACTORY, ppvc_instance_generator
    names, base, descs = M.factory_validation(factory_cfg(), ('N', 'LSW'))
    assert names == ['vali_seed7', 'vali_seed8', 'vali_seed9']
    for i, (jl, pt, meta) in enumerate(base):
        jl0, pt0, meta0 = ppvc_instance_generator(n_modules=2, class_mix='mixed',
                                                  station_counts=DEFAULT_FACTORY, seed=7 + i)
        assert np.array_equal(jl, jl0) and np.array_equal(pt, pt0)
        assert np.array_equal(meta['op_type'], meta0['op_type'])
        for R in ('N', 'LSW'):
            want = make_descriptor(R, jl0, pt0, meta0, 25, seed=7000 + i)
            assert json.dumps(descs[R][i].to_json(), sort_keys=True, default=str) == \
                json.dumps(want.to_json(), sort_keys=True, default=str)
    # instance i of seed 7 is instance i - 1 of seed 8: the base seed is seed + i
    _, base8, _ = M.factory_validation(factory_cfg(seed_train_vali_datagen=8, vali_size=2),
                                       ('N',))
    assert np.array_equal(base8[0][1], base[1][1])
    # the descriptor seed is seed * 1000 + i: another base seed draws other setups
    assert json.dumps(M.factory_validation(factory_cfg(seed_train_vali_datagen=8, vali_size=1),
                                           ('S',))[2]['S'][0].to_json(), default=str) != \
        json.dumps(M.factory_validation(factory_cfg(vali_size=1), ('S',))[2]['S'][0].to_json(),
                   default=str)


def test_at_most_twenty_validation_instances():
    assert M.vali_size(factory_cfg(vali_size=100)) == 20
    assert M.vali_size(factory_cfg(vali_size=5)) == 5
    names, base, _ = M.factory_validation(factory_cfg(vali_size=100, n_j=1), ('N',))
    assert len(names) == len(base) == 20


def test_train_family_holds_the_construction_this_rebuild_follows():
    src = open(os.path.join(ROOT, 'train_family.py')).read()
    for line in ('self.vali_size = min(config.vali_size, 20)',
                 'station_counts=self.ppvc_factory, seed=vseed + i)',
                 'seed=vseed * 1000 + i,',
                 "names = sorted(os.listdir(os.path.join(stored, 'desc', 'N')))",
                 'names = names[:self.vali_size]',
                 "first_mask='legacy', mask_ties='legacy')",
                 'np.random.seed(config.seed_test)',
                 "a = heuristic_select_action('MWKR', henv)",
                 "'  '.join(f'{r}={self.vali_norm[r]:.1f}' for r in self.regimes)"):
        assert line in src, line
    assert M.VALI_CAP == 20


def test_public_set_takes_the_first_sorted_records(tmp_path, monkeypatch):
    names = ['c.desc.json', 'a.desc.json', 'd.desc.json', 'b.desc.json']
    for R in ('N', 'S'):
        (tmp_path / 'desc' / R).mkdir(parents=True)
        for k, nm in enumerate(names):
            (tmp_path / 'desc' / R / nm).write_text(json.dumps(
                dict(regime=R, base_instance=f'base/{nm[0]}', value=10 * k + (R == 'S'))))

    def fake(path):
        rec = json.load(open(path))
        return np.array([rec['value']]), np.array([[1]]), (rec['regime'], rec['value']), rec
    monkeypatch.setattr(M, 'load_family_record', fake)
    cfg = dict(pub_vali_dir=str(tmp_path), vali_size=3)
    got, base, descs = M.public_validation(cfg, ('S',))
    assert got == ['a', 'b', 'c']
    assert [int(b[0][0]) for b in base] == [10, 30, 0]          # from the records of N
    assert all(b[2] == {'op_type': None, 'mch_type': None} for b in base)
    assert descs['S'] == [('S', 11), ('S', 31), ('S', 1)]         # from the records of S
    # a record of S that names another base instance refuses
    p = tmp_path / 'desc' / 'S' / 'b.desc.json'
    p.write_text(json.dumps(dict(regime='S', base_instance='base/x', value=0)))
    with pytest.raises(M.Refused, match='base instance'):
        M.public_validation(cfg, ('S',))


# ---------------------------------------------------------------------------
# the best of four
# ---------------------------------------------------------------------------

def fixed_rollouts(monkeypatch, table, setups=True, block_moves=False):
    """Rollouts of p107 replaced by fixed makespans: table[(mask, rule)] -> makespan."""
    def rollout(rule, jl, pt, meta, desc, adm):
        ms = table[(adm, rule)]
        ct = np.full(3, ms) + (1.0 if (block_moves and adm == 'block') else 0.0)
        return dict(makespan=ms, amch=np.zeros(3, dtype=int), op_ct=ct,
                    counts=np.ones(3, dtype=np.int32), differ=0)
    monkeypatch.setattr(M.P107, 'rule_rollout', rollout)
    monkeypatch.setattr(M.P107, 'check_schedule', lambda *a, **k: None)
    monkeypatch.setattr(M.P107, 'has_setups', lambda desc, combo: setups)


def test_best_of_four_is_the_per_instance_minimum(monkeypatch):
    table = {(a, r): float(v) for a, vals in (('nondelay', (10, 9, 12, 11)),
                                              ('block', (8, 13, 9, 14)),
                                              ('active', (15, 16, 7, 12)))
             for r, v in zip(M.RULES, vals)}
    fixed_rollouts(monkeypatch, table)
    ms, setups = M.evaluate_case(None, None, None, None, 'S', 'case')
    assert setups and ms['nondelay'] == [10.0, 9.0, 12.0, 11.0]
    best = {a: M.best_of_four(ms[a]) for a in M.ADM}
    assert best == {'nondelay': 9.0, 'block': 8.0, 'active': 7.0}
    arr = np.array([[[3, 2, 5, 4], [1, 1, 1, 1]], [[9, 8, 7, 6], [2, 3, 4, 0.5]]])
    assert np.array_equal(M.best_of_four(arr), [[2, 1], [6, 0.5]])


def test_block_that_differs_without_setups_refuses(monkeypatch):
    table = {(a, r): 10.0 for a in M.ADM for r in M.RULES}
    fixed_rollouts(monkeypatch, table, setups=False, block_moves=True)
    with pytest.raises(M.Refused, match='without setups the block rule differs'):
        M.evaluate_case(None, None, None, None, 'N', 'case')
    fixed_rollouts(monkeypatch, table, setups=False, block_moves=False)
    M.evaluate_case(None, None, None, None, 'N', 'case')


def test_pooled_and_per_combination_rows_use_the_bootstrap_of_p107():
    rng = np.random.default_rng(3)
    nd = rng.uniform(100, 200, (6, 8))
    best = {'nondelay': nd, 'active': nd * 0.9, 'block': nd * (1 + rng.normal(0, 0.01, (6, 8)))}
    C = M.comparison(best)
    d = 100 * (best['active'] - nd) / nd
    want = M.P107.SF.instance_boot(d, B=M.B, seed=7)
    assert C['active']['All']['mean'] == pytest.approx(want['mean'])
    assert (C['active']['All']['lo'], C['active']['All']['hi']) == (want['lo'], want['hi'])
    assert C['active']['All']['c'] == 8 and C['active']['rows']['S']['c'] == 1
    assert C['active']['All']['direction'] == 'shorter'
    assert C['active']['counts']['shorter'] == list(M.R8)
    assert C['active']['rows']['N']['shorter_cases'] == 6


# ---------------------------------------------------------------------------
# words
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lo, hi, word', [(-2, -1, 'shorter'), (1, 2, 'longer'),
                                          (-1, 1, 'neither'), (0.0, 0.0, 'neither'),
                                          (-1, 0.0, 'neither'), (0.0, 1, 'neither')])
def test_direction_words(lo, hi, word):
    assert M.direction(lo, hi) == word


@pytest.mark.parametrize('pooled, uses, word', [
    ('shorter', 'active', 'same'), ('longer', 'active', 'different'),
    ('neither', 'active', 'different'), ('longer', 'nondelay', 'same'),
    ('shorter', 'nondelay', 'different'), ('neither', 'nondelay', 'different')])
def test_same_and_different(pooled, uses, word):
    assert M.choice_word(pooled, uses) == word


# ---------------------------------------------------------------------------
# the MWKR check
# ---------------------------------------------------------------------------

def test_mwkr_check_passes_on_the_printed_digit_and_refuses_otherwise(monkeypatch):
    vals = iter([10.0, 10.08, 20.0, 20.0])
    monkeypatch.setattr(M, 'mwkr_makespan', lambda jl, pt, d, cfg: next(vals))
    base = [(None, None, None)] * 2
    descs = {'N': [0, 1], 'S': [0, 1]}
    cover = {'N': ('run', 'log'), 'S': ('run', 'log')}
    got = M.norm_check(base, descs, {}, {'N': '10.0', 'S': '20.0'}, cover)
    assert got['N']['printed'] == '10.0' and got['S']['equal']
    vals = iter([10.0, 10.2, 20.0, 20.0])
    monkeypatch.setattr(M, 'mwkr_makespan', lambda jl, pt, d, cfg: next(vals))
    with pytest.raises(M.Refused, match=r'N: rebuilt 10\.1, log prints 10\.0'):
        M.norm_check(base, descs, {}, {'N': '10.0', 'S': '20.0'}, cover)


def test_logged_norms_are_read_from_the_first_line(tmp_path):
    p = tmp_path / 'train.log'
    p.write_text('[train_family] MASK: first_mask=exact\n'
                 '[vali] MWKR norms: N=156.6  LSW=331.9\n[vali] MWKR norms: N=1.0\n')
    assert M.logged_norms(str(p), 'run') == {'N': '156.6', 'LSW': '331.9'}
    q = tmp_path / 'empty.log'
    q.write_text('Ep 1\n')
    with pytest.raises(M.Refused, match='MWKR norms'):
        M.logged_norms(str(q), 'run')
    with pytest.raises(M.Refused, match='missing input'):
        M.logged_norms(str(tmp_path / 'absent.log'), 'run')


# ---------------------------------------------------------------------------
# the test-instance values and the macros
# ---------------------------------------------------------------------------

def synthetic_record(mean=-5.0, lo=-6.0, hi=-4.0):
    def g(m, a, b, c=1):
        return dict(mean=m, lo=a, hi=b, n=100, c=c)
    rows = {R: dict(diff={x: dict(best_rule=g(mean, lo, hi)) for x in M.ALT}, n=100)
            for R in M.R8}
    rows['All'] = dict(diff={x: dict(best_rule=g(mean, lo, hi, 8)) for x in M.ALT}, n=100)
    return dict(sets={'pub10x5': dict(rows=rows), '10x25': dict(rows=rows)})


def synthetic_macros(mean=-5.0, lo=-6.0, hi=-4.0):
    out = {}
    for w in ('Ten', 'Pub'):
        for x in M.ALT:
            nm = M.test_macro_names(w)[x]
            out.update({nm['mean']: M.f2(mean), nm['Lo']: M.f2(lo), nm['Hi']: M.f2(hi)})
    return out


def test_test_values_are_read_from_the_record_and_checked_against_its_macros():
    T = M.test_values(synthetic_record(), synthetic_macros(), 'public', 'active')
    assert T['choice'] == 'same' and T['chosen'] == 'active' and T['n'] == 100
    assert T['comparison']['active']['All']['mean'] == -5.0
    assert len(T['comparison']['block']['counts']['shorter']) == 8
    assert M.test_values(synthetic_record(), synthetic_macros(), 'factory',
                         'nondelay')['choice'] == 'different'
    assert M.test_macro_names('Ten')['active'] == {
        'mean': 'MrulDiffActBestRuleTenAll', 'Lo': 'MrulDiffActBestRuleLoTenAll',
        'Hi': 'MrulDiffActBestRuleHiTenAll'}
    with pytest.raises(M.Refused, match='MrulDiffActBestRuleHiPubAll'):
        M.test_values(synthetic_record(), synthetic_macros(hi=-3.0), 'public', 'active')


def synthetic_result():
    rng = np.random.default_rng(5)
    out = {}
    for bench, uses in (('factory', 'nondelay'), ('public', 'active')):
        nd = rng.uniform(100, 200, (4, 8))
        best = {'nondelay': nd, 'active': nd * 1.05, 'block': nd * 1.02}
        comp = M.comparison(best)
        T = M.test_values(synthetic_record(), synthetic_macros(), bench, uses)
        out[bench] = dict(n=4, uses=uses, comparison=comp,
                          choice=M.choice_word(comp['active']['All']['direction'], uses),
                          test=T)
    return dict(benchmarks=out)


def test_macro_names_are_unique_new_and_commented(tmp_path):
    lines = M.macro_lines(synthetic_result())
    names = M.macro_names(lines)
    assert len(names) == len(set(names))
    assert all(n.startswith('Mvl') and n.isalpha() for n in names)
    assert all('  % ' in ln for ln in lines if ln.startswith('\\newcommand'))
    for w in ('Ten', 'Pub'):
        for n in (f'MvlN{w}', f'MvlTestN{w}', f'MvlDiffAct{w}', f'MvlDiffActLo{w}',
                  f'MvlTestDiffBlkHi{w}', f'MvlDirAct{w}', f'MvlTestDirBlk{w}',
                  f'MvlShorterAct{w}', f'MvlTestNeitherBlk{w}', f'MvlChoice{w}',
                  f'MvlTestChoice{w}'):
            assert n in names, n
    assert '\\newcommand{\\MvlChoiceTen}{same}' in '\n'.join(lines)
    assert '\\newcommand{\\MvlChoicePub}{different}' in '\n'.join(lines)
    assert M.collision_check(names, str(tmp_path / 'macros_mask_validation.tex')) > 0
    with pytest.raises(M.Refused, match='duplicate'):
        M.collision_check(names + names[:1], str(tmp_path / 'm.tex'))
    with pytest.raises(M.Refused, match='plus letters'):
        M.collision_check(names + ['MvlDiff2'], str(tmp_path / 'm.tex'))


def test_a_name_defined_in_another_macro_file_refuses(tmp_path, monkeypatch):
    (tmp_path / 'manuscript').mkdir()
    (tmp_path / 'manuscript' / 'macros_other.tex').write_text('\\newcommand{\\MvlBoot}{1}\n')
    (tmp_path / 'manuscript' / 'macros_mask_validation.tex').write_text(
        '\\newcommand{\\MvlBoot}{20000}\n')
    monkeypatch.setattr(M, 'ROOT', str(tmp_path))
    out = tmp_path / 'out' / 'macros_mask_validation.tex'
    with pytest.raises(M.Refused, match='macros_other.tex'):
        M.collision_check(['MvlBoot'], str(out))
    (tmp_path / 'manuscript' / 'macros_other.tex').unlink()
    assert M.collision_check(['MvlBoot'], str(out)) == 1       # its own copy is skipped
