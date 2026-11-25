"""Tests for the perturbation rules of scripts/p87_sensitivity.py.

  * a setup scale of 1.0 and a window scale of 1.0, drawn through the same
    make_descriptor path as the shifted levels, reproduce the stored fields;
  * setup scales are monotone entry by entry, integer, at least 1 h;
  * lag scaling keeps zeros and integer hours;
  * every level leaves the combinations it cannot reach unchanged, and the
    build's placebo check accepts the setup convention the loader states and
    nothing else;
  * the build writes perturbed records that state their setup convention and
    pass checks C1, C2, C3 and C5 as the evaluation loads them;
  * duration noise keeps compatibility, integer hours >= 1, the mean within a
    few percent, and uses common random numbers across levels;
  * the station-type scale touches only finishing and assembly rows;
  * the capacity-halved factory keeps the first stations of each type with
    their durations and calendars;
  * the setup subsets charge the stated station types;
  * the verdict rule equals the paper's;
  * the rule set of each mask: the exact mask rolls the rules out with
    common_utils.masked_rule_action and reproduces the stored arrays of
    eval_family.py --rule_set masked on the first instances, the legacy mask with
    heuristic_select_action; each rule record states its rule set, a rule file of the
    mask's set is skipped (resume), and one of the other set is refused, never
    overwritten.

Run from the repository root:  python3 -m pytest tests/test_p87_sensitivity.py -q
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401  (parses argv at import)
    import data_utils  # noqa: E402,F401
    _spec = importlib.util.spec_from_file_location(
        'p87', os.path.join(ROOT, 'scripts', 'p87_sensitivity.py'))
    p87 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(p87)
finally:
    sys.argv = _argv

from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from ppvc_instance_generator import (STATION_TYPES, TIGHT_FACTORY,  # noqa: E402
                                     load_instance)

NAMES = ['instance_000', 'instance_017', 'instance_063']


def stored(regime, name):
    rec = p87.stored_record(regime, name)
    jl, pt, meta = load_instance(rec['base_instance'])
    return rec, jl, pt, meta


def core(d):
    j = d.to_json()
    return {k: json.dumps(j[k]) for k in p87.CORE_KEYS}


@pytest.fixture
def temp_levels():
    added = {'_t_setup_x100': dict(family='setup', kind='scale', scale=1.0),
             '_t_win_x100': dict(family='win', scale=1.0)}
    p87.LEVELS.update(added)
    yield
    for k in added:
        p87.LEVELS.pop(k)


def test_unit_scales_reproduce_stored(temp_levels):
    for name in NAMES:
        for regime in ('S', 'LSW'):
            rec, jl, pt, meta = stored(regime, name)
            d = p87.perturb_desc('_t_setup_x100', regime, rec, jl, pt, meta)
            assert json.dumps(d.to_json()['setup']) == json.dumps(rec['setup'])
        for regime in ('W', 'SW'):
            rec, jl, pt, meta = stored(regime, name)
            d = p87.perturb_desc('_t_win_x100', regime, rec, jl, pt, meta)
            assert json.dumps(d.to_json()['windows']) == json.dumps(rec['windows'])


def test_setup_scales_monotone():
    for name in NAMES:
        rec, jl, pt, meta = stored('LS', name)
        S0 = np.asarray(rec['setup'], float)
        lo = p87.perturb_desc('setup_x050', 'LS', rec, jl, pt, meta).setup
        hi = p87.perturb_desc('setup_x150', 'LS', rec, jl, pt, meta).setup
        assert np.all(lo <= S0) and np.all(S0 <= hi)
        for S in (lo, hi):
            assert np.all(S >= 1) and np.all(S == np.round(S))
        z = p87.perturb_desc('setup_same0', 'LS', rec, jl, pt, meta).setup
        assert np.all(np.diag(z[:4]) == 0)
        off = ~np.eye(4, dtype=bool)
        assert np.array_equal(z[:4][off], S0[:4][off])
        assert np.array_equal(z[4], S0[4])


def test_lag_scale():
    rec, jl, pt, meta = stored('LSW', 'instance_005')
    lag0 = np.asarray(rec['lag'], float)
    for lv, s in (('lag_x050', 0.5), ('lag_x150', 1.5)):
        lag = p87.perturb_desc(lv, 'LSW', rec, jl, pt, meta).lag
        assert np.array_equal(lag == 0, lag0 == 0)
        assert np.all(lag == np.round(lag))
        assert np.all(np.abs(lag - s * lag0) <= 0.5)


@pytest.mark.parametrize('level', [k for k in p87.LEVELS
                                   if p87.LEVELS[k]['family'] in ('lag', 'setup', 'win')])
def test_placebo_cells_unchanged(level):
    for regime in p87.placebo(level):
        rec, jl, pt, meta = stored(regime, 'instance_042')
        d = p87.perturb_desc(level, regime, rec, jl, pt, meta)
        assert core(d) == {k: json.dumps(rec[k]) for k in p87.CORE_KEYS}
        assert getattr(d, 'setup_mult', None) is None


@pytest.mark.parametrize('level', ('lag_x050', 'setup_x050', 'win_x150'))
def test_placebo_check_accepts_the_stated_convention_only(level):
    # placebo cells with setups (S, SW of a lag level) carry the convention
    # the loader stated; the build's check accepts it and nothing else
    n_setup = 0
    for regime in p87.placebo(level):
        rec, jl, pt, meta = stored(regime, 'instance_042')
        d = p87.perturb_desc(level, regime, rec, jl, pt, meta)
        out = d.to_json()
        assert 'setup_convention' not in rec
        assert out.get('setup_convention') == ('attached' if d.has_setup else None)
        assert p87.placebo_unchanged(d, rec)
        if d.has_setup:
            n_setup += 1
            d.setup_convention = 'detached'
            assert not p87.placebo_unchanged(d, rec)
            d.setup_convention = 'attached'
            d.setup = d.setup + 1.0
            assert not p87.placebo_unchanged(d, rec)
    assert n_setup == (0 if level.startswith('setup') else 2)


def test_build_writes_cells_that_pass_their_checks(tmp_path):
    # the build path itself: perturbed and placebo cells of a level that
    # changes setups, one that charges them on a subset of stations, and one
    # that leaves them alone; the build refuses on any C1, C2, C3 or C5 failure
    import argparse
    from constraint_family import checks
    levels = ['lag_x050', 'setup_x050', 'setup_sub_wide']
    p87.build(argparse.Namespace(root=str(tmp_path), levels=levels, n=2))
    for level in levels:
        man = json.load(open(tmp_path / 'data' / level / 'manifest.json'))
        assert man['placebo_cells_checked'] == 2 * len(p87.placebo(level))
        for regime in p87.R8:
            for name, jl, pt, meta, d in p87.load_cell(str(tmp_path), level, regime, 2):
                rep = {r.id: r for r in checks.run_all(d, pt, False, job_length=jl)}
                assert all(rep[c].status != checks.FAIL
                           for c in ('C1', 'C2', 'C3', 'C5')), (level, regime, name)
                if regime in p87.reached(level):
                    rec = json.load(open(tmp_path / 'data' / level / 'desc' / regime
                                         / f'{name}.desc.json'))
                    assert rec.get('setup_convention') == ('attached' if 'S' in regime
                                                           else None)
        if level == 'setup_sub_wide':
            assert man['setup_machines_charged'] == [21]


def test_duration_noise():
    rec, jl, pt, meta = stored('N', 'instance_011')
    outs = {}
    for lv in ('dur_cv10', 'dur_cv20', 'dur_cv30'):
        jl2, pt2, meta2 = p87.perturb_base(lv, 11, jl, pt, meta)
        assert np.array_equal(pt2 > 0, pt > 0)
        assert np.all(pt2[pt2 > 0] >= 1) and pt2.dtype.kind == 'i'
        assert abs(pt2[pt > 0].mean() / pt[pt > 0].mean() - 1) < 0.03
        outs[lv] = np.abs(pt2[pt > 0] - pt[pt > 0]).mean()
    assert outs['dur_cv10'] < outs['dur_cv20'] < outs['dur_cv30']
    # common random numbers: the sign of the deviation agrees across levels
    a = p87.perturb_base('dur_cv10', 11, jl, pt, meta)[1].astype(float) - pt
    c = p87.perturb_base('dur_cv30', 11, jl, pt, meta)[1].astype(float) - pt
    both = (a != 0) & (c != 0)
    assert np.mean(np.sign(a[both]) == np.sign(c[both])) > 0.95
    # deterministic
    again = p87.perturb_base('dur_cv20', 11, jl, pt, meta)[1]
    assert np.array_equal(again, p87.perturb_base('dur_cv20', 11, jl, pt, meta)[1])


def test_type_scale_touches_only_long_types():
    rec, jl, pt, meta = stored('N', 'instance_003')
    st = np.asarray(meta['op_station'])
    rows = np.isin(st, [STATION_TYPES.index(t) for t in p87.LONG_TYPES])
    for lv, s in (('dur_long_m25', 0.75), ('dur_long_p25', 1.25)):
        pt2 = p87.perturb_base(lv, 3, jl, pt, meta)[1]
        assert np.array_equal(pt2[~rows], pt[~rows])
        m = pt[rows] > 0
        assert np.all(np.abs(pt2[rows][m] - np.maximum(1, s * pt[rows][m])) <= 0.5)


def test_tight_factory():
    rec, jl, pt, meta = stored('LSW', 'instance_009')
    jl2, pt2, meta2 = p87.perturb_base('tight15', 9, jl, pt, meta)
    keep = p87.kept_machines('tight15')
    assert pt2.shape == (pt.shape[0], 15) and len(keep) == 15
    assert np.array_equal(pt2, pt[:, keep])
    mt = np.asarray(meta2['mch_type'])
    for i, t in enumerate(STATION_TYPES):
        assert (mt == i).sum() == TIGHT_FACTORY[t]
    d = p87.perturb_desc('tight15', 'LSW', rec, jl, pt, meta)
    assert d.windows == [ConstraintDescriptor.from_json(rec).windows[k] for k in keep]
    assert json.dumps(d.to_json()['setup']) == json.dumps(rec['setup'])


def test_setup_subsets():
    rec, jl, pt, meta = stored('S', 'instance_001')
    mt = np.asarray(meta['mch_type'])
    for lv, types, n in (('setup_sub_strict', p87.SUB_STRICT, 7),
                         ('setup_sub_wide', p87.SUB_WIDE, 21)):
        u = p87.perturb_desc(lv, 'S', rec, jl, pt, meta).setup_mult
        assert int(u.sum()) == n
        on = {STATION_TYPES[t] for t in mt[u == 1]}
        assert on == set(types)
    assert {'Q', 'G'}.isdisjoint(p87.SUB_WIDE)


def test_verdict_rule():
    v = p87.verdict
    assert v(-0.8, -0.1, 1.0) == 'SUPERIOR'
    assert v(-0.6, 0.4, 1.0) == 'EQUIVALENT'
    assert v(-0.6, 0.4, 0.5) == 'non-inferior'
    assert v(-0.4, 0.6, 0.5) == 'INCONCLUSIVE'
    assert v(-1.5, 0.4, 1.0) == 'non-inferior'
    assert v(1.2, 2.0, 1.0) == 'INFERIOR'
    assert v(0.2, 1.4, 1.0) == 'INCONCLUSIVE'


def test_model_names():
    assert p87.model_name('pact', 'S', 301) == '10x25+family+jointblind-v1'
    assert p87.model_name('pact', 'S', 303) == '10x25+family+jointblind-v1-s303'
    assert p87.model_name('spec', 'LSW', 302) == '10x25+family+spec-LSW-s302'
    assert p87.model_name('single', 'N', 301) == '10x25+family+jointblind-nlsw-s301'
    u = p87.units_for('setup_x050', 'S', ['pact', 'spec', 'single', 'rules'],
                      [301, 302, 303])
    assert len(u) == 13


# ------------------------------------------------- the rule set of each mask
def bare_evaluator(mask):
    """An Evaluator with only its mask set: a rule rollout loads no policy, and
    Evaluator.__init__ would change the torch settings of the whole test process."""
    import torch
    ev = p87.Evaluator.__new__(p87.Evaluator)
    ev.mask = mask
    ev.torch = torch                 # the record reads its version and thread count
    return ev


def test_rule_set_by_mask(monkeypatch):
    import common_utils
    assert p87.RULE_SET == {'legacy': 'classical', 'exact': 'masked'}
    assert p87.rule_function('exact') is common_utils.masked_rule_action
    assert p87.rule_function('legacy') is common_utils.heuristic_select_action
    assert bare_evaluator('exact').rule_set == 'masked'
    calls = []
    for name in ('masked_rule_action', 'heuristic_select_action'):
        f = getattr(common_utils, name)
        monkeypatch.setattr(common_utils, name,
                            lambda m, env, f=f, name=name: calls.append(name) or f(m, env))
    (_, jl, pt, meta, desc), = p87.load_cell(ROOT, 'default', 'LS', 1)
    for mask, want in (('exact', 'masked_rule_action'), ('legacy', 'heuristic_select_action')):
        calls.clear()
        bare_evaluator(mask).rollout('MOR', jl, pt, meta, desc)
        assert calls and set(calls) == {want}


@pytest.mark.parametrize('regime', ['N', 'LSW'])
def test_exact_rules_reproduce_the_stored_arrays(regime):
    # the first instances of the default level against eval_family.py --rule_set masked
    cell = p87.load_cell(ROOT, 'default', regime, 3)
    ev = bare_evaluator('exact')
    for r in p87.RULES:
        stored = np.load(os.path.join(ROOT, p87.STORED_XMASK, f'Result_{r}_{regime}.npy'))
        counts = {'decisions': 0, 'one_pair': 0, 'setup': 0.0}
        got = [ev.rollout(r, jl, pt, meta, desc, counts=counts)[0]
               for _, jl, pt, meta, desc in cell]
        assert got == stored[:3, 0].tolist(), r
        assert counts['decisions'] == sum(pt.shape[0] for _, jl, pt, meta, desc in cell)


def test_rule_record_states_the_rule_set(tmp_path, monkeypatch):
    monkeypatch.setattr(p87, 'git_commit', lambda: 'test')
    cell = p87.load_cell(ROOT, 'default', 'S', 2)
    root = str(tmp_path)
    for mask, want in (('exact', 'masked'), ('legacy', 'classical')):
        ev = bare_evaluator(mask)
        assert p87.evaluate_unit(ev, root, 'default', 'S', 'rule', None, 'FIFO', cell,
                                 trace=True)[0] == 'done'
        p = p87.result_path(root, mask, 'default', 'S', 'FIFO')
        rec = json.load(open(p[:-4] + '.json'))
        assert (rec['rule_set'], rec['rule_function'], rec['rule_seed']) == (
            want, p87.RULE_FUNCTION[want], 50)
        # a file of the mask's rule set is skipped, so a stopped part resumes
        assert p87.evaluate_unit(ev, root, 'default', 'S', 'rule', None, 'FIFO', cell,
                                 trace=True)[0] == 'skip'
    # a rule file of the earlier set in the exact mask is refused, never overwritten
    p = p87.result_path(root, 'exact', 'default', 'S', 'FIFO')
    rec = json.load(open(p[:-4] + '.json'))
    rec.pop('rule_set')
    json.dump(rec, open(p[:-4] + '.json', 'w'))
    before = open(p, 'rb').read()
    for force in (False, True):
        with pytest.raises(SystemExit, match='rule set classical.*archive-rules'):
            p87.evaluate_unit(bare_evaluator('exact'), root, 'default', 'S', 'rule', None,
                              'FIFO', cell, force=force, trace=True)
    assert open(p, 'rb').read() == before
