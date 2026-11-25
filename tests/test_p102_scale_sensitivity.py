"""Tests for scripts/p102_scale_sensitivity.py.

  * the window rule is the generator's: draw_windows with the stored seed and the
    stored horizon reproduces every stored calendar;
  * level win_full keeps lags, setups, classes and activation bits, puts every window
    inside [0, H') with H' the stored SPT makespan without windows, keeps durations in
    the generator's range, and gives per-machine densities in the stored rule's range,
    with a mean between the rule's target mean and the stored mean (the last window's
    excess over the target is a smaller share of a longer horizon); checks C1 to C5
    pass;
  * the window seeds are a documented stream distinct from the stored seeds;
  * the default level's rules (a classical one and a descriptor-reading one)
    reproduce the stored exact-mask arrays;
  * the five-class probe builds a [6, 5] matrix with the cold-start row last and the
    stored block unchanged, a quarter of the modules in class 4, and reports that
    check C1 and the environment refuse it;
  * infer refuses a missing result with a message that starts with "refused", and on
    synthetic policy results produces the comparisons, the shifts and unique,
    prefixed macros.

Run from the repository root:
  python -m pytest tests/test_p102_scale_sensitivity.py -q -p no:cacheprovider
CPU only.
"""
import importlib.util
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401
    import data_utils  # noqa: E402,F401
    mods = {}
    for name in ('p101_mask_controls', 'p102_scale_sensitivity'):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(ROOT, 'scripts', f'{name}.py'))
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name]
        spec.loader.exec_module(mods[name])
finally:
    sys.argv = _argv
configs.device = 'cpu'
P101, P102 = mods['p101_mask_controls'], mods['p102_scale_sensitivity']

from constraint_family import checks  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor, draw_windows  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

NAMES = ('instance_000', 'instance_007', 'instance_021', 'instance_029')


def stored_rec(size, combo, name):
    with open(P102.stored_desc_path(size, combo, name)) as f:
        return json.load(f)


@pytest.mark.parametrize('size', P102.SIZES)
def test_window_rule_is_the_generators(size):
    for name in NAMES:
        for combo in P102.WIN_COMBOS:
            rec = stored_rec(size, combo, name)
            rng = np.random.default_rng(np.random.SeedSequence([rec['desc_seed'],
                                                                P102.WINDOW_TAG]))
            dens, dur = P102.window_ranges()
            w = draw_windows(len(rec['windows']), rec['horizon'], rng, density_range=dens,
                             dur_range=dur)
            assert json.dumps([[list(x) for x in m] for m in w]) == json.dumps(rec['windows'])
    assert P102.window_ranges() == ((0.05, 0.15), (2, 8))


@pytest.mark.parametrize('size', P102.SIZES)
def test_full_level_descriptors(size):
    dens_new, dens_old = [], []
    row = {nm: i for i, nm in enumerate(P101.names_of(size))}
    for name in NAMES:
        jl, pt, _ = load_instance(os.path.join(ROOT, P101.SIZES[size]['base'], name))
        for combo in P102.WIN_COMBOS:
            spt, _ = P102.stored_spt(size, P102.NO_WINDOW[combo])
            H = int(spt[row[name]])
            d, rec = P102.full_descriptor(size, combo, name, H)
            j = d.to_json()
            for k in ('active', 'lag', 'op_class', 'setup'):
                assert json.dumps(j[k]) == json.dumps(rec[k]), k
            assert j['horizon'] == H > rec['horizon']
            assert len(d.windows) == pt.shape[1]
            for m in d.windows:
                prev = 0
                for s, e in m:
                    assert 0 <= s < e < H and 2 <= e - s <= 8
                    assert s >= prev
                    prev = e
                dens_new.append(sum(e - s for s, e in m) / H)
            dens_old += [sum(e - s for s, e in m) / rec['horizon'] for m in rec['windows']]
            assert max(e for m in d.windows for _, e in m) > 0.8 * H
            rep = {r.id: r for r in checks.run_all(d, pt, False, job_length=jl)}
            assert all(rep[c].status == checks.PASS for c in ('C1', 'C2', 'C3', 'C5'))
            assert rep['C4'].status == checks.PASS, rep['C4'].messages
    # the rule draws a target density U[0.05, 0.15] and stops at the first window that
    # reaches it, so a density exceeds its target by less than one window (8 h) over the
    # horizon; the longer horizon shrinks that excess, so the new mean lies between the
    # target mean 0.10 and the stored mean
    (lo, hi), (_, dmax) = P102.window_ranges()
    h_min = min(int(P102.stored_spt(size, c)[0].min()) for c in P102.NO_WINDOW.values())
    assert min(dens_new) >= lo and max(dens_new) <= hi + dmax / h_min
    assert 0.5 * (lo + hi) - 0.01 < np.mean(dens_new) <= np.mean(dens_old) + 0.002
    assert np.mean(dens_old) - np.mean(dens_new) < 0.015


def test_window_seeds():
    names = P101.names_of('50x25n30')
    seeds = [P102.window_seed(n) for n in names]
    assert seeds == [20261113 + i for i in range(30)]
    stored = {stored_rec('50x25n30', 'W', n)['desc_seed'] for n in names}
    assert not stored & set(seeds)


def test_default_level_reproduces_stored_rules():
    cell = P102.load_level_cell(None, 'default', '50x25n30', 'SW', 1)
    _, jl, pt, meta, desc = cell[0]
    for rule in ('MWKR', 'ECT'):
        r = (P101.rule_rollout(rule, jl, pt, meta, desc) if rule in P102.RULES4
             else P102.aware_rollout(rule, jl, pt, meta, desc))
        stored = P101.stored_rule('50x25n30', rule, 'SW', 1)
        assert r['makespan'] == stored[0]


def test_five_class_probe():
    d5, jl, pt, meta, pick = P102.five_class_descriptor()
    rec = stored_rec('10x25', 'S', 'instance_000')
    S4 = np.asarray(rec['setup'], float)
    assert d5.setup.shape == (6, 5)
    assert np.array_equal(d5.setup[:4, :4], S4[:4])
    assert np.array_equal(d5.setup[5, :4], S4[4])            # cold-start row last
    assert np.array_equal(d5.setup[:5, :5], d5.setup[:5, :5].T)
    mp = float(pt[pt > 0].mean())
    assert all(max(1, round(0.5 * mp)) <= d5.setup[a, 4] <= max(1, round(1.0 * mp))
               for a in range(4))
    assert len(pick) == len(jl) // 4
    job_of_op = np.repeat(np.arange(len(jl)), jl)
    assert np.array_equal(d5.op_class == 4, np.isin(job_of_op, pick))
    out = P102.five_class_probe()
    assert out['check_C1'].startswith('FAIL')
    assert out['environment'].startswith('refused: ValueError')


def test_infer_refuses_a_missing_result(tmp_path):
    args = _args(str(tmp_path / 'empty'), str(tmp_path / 'out'), ['rules'])
    with pytest.raises(SystemExit) as e:
        P102.infer(args)
    assert str(e.value).startswith('refused'), str(e.value)


def _args(root, out, arms, n=3, seeds=(301, 302)):
    class A:
        pass
    a = A()
    a.root, a.out, a.partial, a.n, a.boot = root, out, True, n, 200
    a.levels, a.sizes, a.combos = list(P102.LEVELS), ['50x25n30'], list(P102.WIN_COMBOS)
    a.arms, a.seeds = list(arms), list(seeds)
    return a


def _write_unit(root, level, combo, label, ms, names):
    p = P102.result_path(root, level, '50x25n30', combo, label)
    n = len(ms)
    P101.save_npz_atomic(p, makespan=np.asarray(ms, float), lb0=np.ones(n),
                         util=np.full(n, 0.2), ms_over_lb0=np.ones(n),
                         cov_end=np.full(n, 0.5 if level == 'default' else 0.9),
                         cov_time=np.full(n, 0.05), last_window_end=np.full(n, 100.0),
                         horizon=np.full(n, 200.0), amch=np.zeros((n, 4), dtype=np.int16))
    P101.write_json_atomic(p[:-4] + '.json', {'level': level, 'size': '50x25n30',
                                              'regime': combo, 'label': label,
                                              'instances': names})


def test_infer_on_synthetic_policy_results(tmp_path, monkeypatch):
    """Default-level rules are the stored arrays (infer checks them); level win_full
    multiplies them by 1.05; the policy is 1% shorter than the specialist on the
    default level and 0.5% longer on win_full. The synthetic policy arrays are not the
    exact campaign's, so the check against its stored arrays is switched off here."""
    monkeypatch.setattr(P102, 'check_stored_policy', lambda *a: {'synthetic': 'unchecked'})
    root, n, seeds = str(tmp_path / 'r'), 3, (301, 302)
    names = P101.names_of('50x25n30', n)
    rng = np.random.default_rng(0)
    for combo in P102.WIN_COMBOS:
        for level, f in (('default', 1.0), ('win_full', 1.05)):
            for r in P102.RULES7:
                _write_unit(root, level, combo, r,
                            f * P101.stored_rule('50x25n30', r, combo, n), names)
            base = f * P101.stored_rule('50x25n30', 'SPT', combo, n)
            for s in seeds:
                spec = base * (1 + 0.01 * rng.random(n))
                pol = spec * (0.99 if level == 'default' else 1.005)
                for arm, ms in (('spec', spec), ('pact', pol), ('single', spec * 1.02)):
                    _write_unit(root, level, combo,
                                'greedy+' + P102.model_name(arm, combo, s), ms, names)
    mp = os.path.join(root, 'data', 'win_full', '50x25n30', 'manifest.json')
    P101.write_json_atomic(mp, {'per_combination': {c: {
        'horizon_new_mean_h': 400.0, 'horizon_stored_mean_h': 210.0,
        'window_density_mean': 0.1, 'windows_per_machine_mean': 9.0,
        'windows_per_machine_stored_mean': 5.0, 'c4_warn_share': 0.0}
        for c in P102.WIN_COMBOS}})
    res = P102.infer(_args(root, str(tmp_path / 'out'), list(P102.ARM_CHOICES), n, seeds))
    S = res['sizes']['50x25n30']
    P = S['policy']
    assert P['default']['pooled']['pol_spec']['mean'] == pytest.approx(-1.0, abs=1e-6)
    assert P['win_full']['pooled']['pol_spec']['mean'] == pytest.approx(0.5, abs=1e-6)
    assert P['shift']['pol_spec']['pooled']['mean'] == pytest.approx(1.5, abs=1e-3)
    assert set(P['win_full']['rows']['pol_rule']['LSW']) >= {'verdict', 'p_t_holm'}
    for c in P102.WIN_COMBOS:
        assert S['combos'][c]['rule_shift']['spt']['mean'] == pytest.approx(5.0)
        assert all(v == 'equal' for v in
                   S['combos'][c]['levels']['default']['stored_check'].values())
    txt = open(tmp_path / 'out' / 'manuscript' / 'macros_scalesens.tex').read()
    mnames = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(mnames) == len(set(mnames)) > 100
    assert all(re.fullmatch(r'Psens[A-Za-z]+', x) for x in mnames)
