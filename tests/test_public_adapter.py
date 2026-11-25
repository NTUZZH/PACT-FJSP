"""Tests for the public-instance adapter and the public-family rule generator.

  * every collected instance parses; job, machine and operation counts match
    the published sizes; the adapter agrees with data_utils.text_to_matrix;
  * the best-known tables cover every instance and are internally
    consistent; a simple dispatching rule on regime N never beats the
    best-known value, and its schedule passes both validators;
  * the rule generator is deterministic, keeps lags, classes, setup matrix,
    windows and horizon identical across the regimes that activate them,
    stays inside its declared ranges and reduces to the factory's ranges on
    factory instances;
  * random feasible dispatch sequences, simulated with the reference
    simulator under each regime, pass both validators, and for a fixed
    sequence the makespan never decreases when a rule is switched on;
  * FJSPEnvFamily runs on public instances (unequal operation counts,
    arbitrary eligibility, machines no operation uses) and its schedules
    equal the reference simulator's.

Run from the repository root:  python3 -m pytest tests/test_public_adapter.py -q
CPU only; reads data/PUBLIC and, for the factory checks, data/PPVC.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # stored records name their base instance relative to the root

# params parses sys.argv at import time; hide pytest's arguments and force CPU
_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402
    import data_utils  # noqa: E402
finally:
    sys.argv = _argv
configs.device = 'cpu'

import torch  # noqa: E402

import fjsp_env_same_op_nums  # noqa: E402
fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')

from common_utils import heuristic_select_action  # noqa: E402
from constraint_family import checks, reference_sim  # noqa: E402
from constraint_family.checks import FAIL, PASS  # noqa: E402
from constraint_family.descriptor import (REGIMES_ALL,  # noqa: E402
                                          ConstraintDescriptor,
                                          job_chain_horizon)
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from constraint_family.public_adapter import (SETUP_SENTINEL,  # noqa: E402
                                              UnrepresentableInstance,
                                              list_instances, load_set,
                                              parse_fjs_text, read_fjs,
                                              sd2_instance, setup_structure,
                                              to_fjs_text)
from constraint_family.public_rules import (DEFAULT_PARAMS,  # noqa: E402
                                            FACTORY_MEAN_PT, SET_SEED0,
                                            draw_rules, factory_lag_bands,
                                            factory_lag_fractions,
                                            load_family_record,
                                            magnitude_violations,
                                            public_horizon,
                                            window_duration_range)
from conftest import instances_present, needs_public_instances  # noqa: E402

PUB = 'data/PUBLIC'
STANDARD_SETS = ['Brandimarte', 'Hurink_edata', 'Hurink_rdata', 'Hurink_vdata',
                 'SD2/test/10x5+mix', 'SD2/test/20x10+mix',
                 'SD2/vali/10x5+mix', 'SD2/vali/20x10+mix']
FAMILY_DIRS = {k: f'family/pub_{k}' for k in (
    'Brandimarte', 'Hurink_edata', 'Hurink_rdata', 'Hurink_vdata',
    'SD2_test_10x5+mix', 'SD2_test_20x10+mix', 'SD2_vali_10x5+mix',
    'SD2_vali_20x10+mix')}
BKS_CSV = {'Brandimarte': 'Brandimarte/best_known.csv',
           'Hurink_edata': 'Hurink_edata/best_known.csv',
           'Hurink_rdata': 'Hurink_rdata/best_known.csv',
           'Hurink_vdata': 'Hurink_vdata/best_known.csv',
           'SD2/test/10x5+mix': 'SD2/best_known_test_10x5+mix.csv',
           'SD2/test/20x10+mix': 'SD2/best_known_test_20x10+mix.csv'}

# Published sizes. Brandimarte: Behnke and Geiger (2012), Table 4 (jobs,
# declared machines, operations per job, maximum eligible machines per
# operation). Hurink et al. (1994) reuse the classical job-shop instances, so
# every job has M operations: Fisher and Thompson (mt), Lawrence (la), Adams,
# Balas and Zawack (abz), Carlier and Pinson (car, sizes from Behnke and
# Geiger 2012, Table 10) and Applegate and Cook (orb).
BRANDIMARTE = {'Mk01': (10, 6, (5, 7), 3), 'Mk02': (10, 6, (5, 7), 6),
               'Mk03': (15, 8, (10, 10), 5), 'Mk04': (15, 8, (3, 10), 3),
               'Mk05': (15, 4, (5, 10), 2), 'Mk06': (10, 15, (15, 15), 5),
               'Mk07': (20, 5, (5, 5), 5), 'Mk08': (20, 10, (5, 15), 2),
               'Mk09': (20, 10, (10, 15), 5), 'Mk10': (20, 15, (10, 15), 5)}
HURINK = {'mt06': (6, 6), 'mt10': (10, 10), 'mt20': (20, 5),
          'abz5': (10, 10), 'abz6': (10, 10), 'abz7': (20, 15),
          'abz8': (20, 15), 'abz9': (20, 15),
          'car1': (11, 5), 'car2': (13, 4), 'car3': (12, 5), 'car4': (14, 4),
          'car5': (10, 6), 'car6': (8, 9), 'car7': (7, 7), 'car8': (8, 8)}
HURINK.update({f'orb{k}': (10, 10) for k in range(1, 11)})
for _k in range(1, 41):
    HURINK[f'la{_k:02d}'] = [(10, 5), (15, 5), (20, 5), (10, 10), (15, 10),
                             (20, 10), (30, 10), (15, 15)][(_k - 1) // 5]
# Fattahi, Mehrabad and Jolai (2007) via Behnke and Geiger (2012), Table 18:
# jobs, machines, maximum operations per job.
FATTAHI = [(2, 2, 2), (2, 2, 2), (3, 2, 2), (3, 2, 2), (3, 2, 2), (3, 2, 3),
           (3, 5, 3), (3, 4, 3), (3, 3, 3), (4, 5, 3), (5, 6, 3), (5, 7, 3),
           (6, 7, 3), (7, 7, 3), (7, 7, 3), (8, 7, 3), (8, 7, 4), (9, 8, 4),
           (11, 8, 4), (12, 8, 4)]
# Fattahi_setup_06 declares and uses 3 machines; Table 18 lists SFJS6 with 2.
FATTAHI_FILE_DIFFERS = {5: (3, 3, 3)}
# Machines that no operation of the file can use (declared M, used M).
UNUSED_MACHINES = {'Mk06': (15, 10), 'Mk08': (10, 9), 'Mk10': (15, 11)}
ZERO_PT_FILES = {'orb7'}


def _set_instances(d):
    insts, skipped = load_set(os.path.join(PUB, d), skip_unrepresentable=True)
    return insts, skipped


def _read_csv(rel):
    with open(os.path.join(PUB, rel)) as f:
        return list(csv.DictReader(f))


def ect_schedule(job_length, op_pt):
    """Earliest-completion-time dispatching on regime N: repeatedly schedule
    the (next operation, eligible machine) pair that completes first; ties go
    to the lower job index, then the lower machine index."""
    job_length = np.asarray(job_length, dtype=int)
    J, M = len(job_length), op_pt.shape[1]
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]])
    nxt = first.copy()
    end = first + job_length
    jr, mf = np.zeros(J), np.zeros(M)
    actions = []
    for _ in range(int(job_length.sum())):
        best = None
        for j in range(J):
            if nxt[j] == end[j]:
                continue
            op = nxt[j]
            for m in np.flatnonzero(op_pt[op]):
                c = max(jr[j], mf[m]) + op_pt[op, m]
                if best is None or c < best[0]:
                    best = (c, j, m)
        c, j, m = best
        actions.append((j, int(m)))
        jr[j] = mf[m] = c
        nxt[j] += 1
    return actions


def random_actions(job_length, op_pt, rng):
    """A random feasible dispatch sequence: a random job with operations left,
    then a random eligible machine for its next operation."""
    job_length = np.asarray(job_length, dtype=int)
    left = job_length.copy()
    first = np.concatenate([[0], np.cumsum(job_length)[:-1]])
    acts = []
    while left.sum():
        j = int(rng.choice(np.flatnonzero(left)))
        op = first[j] + job_length[j] - left[j]
        m = int(rng.choice(np.flatnonzero(op_pt[op])))
        acts.append((j, m))
        left[j] -= 1
    return acts


def validate(jl, pt, desc, sim):
    r = checks.check_schedule_validation(desc, jl, pt, sim['op_mch'],
                                         sim['op_ct'], sim['op_st'])
    assert r.status == PASS, r.messages


# ----------------------------------------------------------------------
# Part 1: parsing and sizes
# ----------------------------------------------------------------------

@needs_public_instances()
@pytest.mark.parametrize('d', STANDARD_SETS)
def test_parse_and_agree_with_existing_loader(d):
    paths = list_instances(os.path.join(PUB, d))
    assert paths, d
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if name in ZERO_PT_FILES:
            with pytest.raises(UnrepresentableInstance):
                read_fjs(p)
            continue
        inst = read_fjs(p)
        assert inst.setup is None
        with open(p) as f:
            jl, pt = data_utils.text_to_matrix(f.readlines())
        assert np.array_equal(jl, inst.job_length), p
        assert np.array_equal(pt, inst.op_pt), p
        assert (inst.eligible_counts >= 1).all()
        assert inst.has_incompatible_pair, p
        # round trip through the writer
        back = parse_fjs_text(to_fjs_text(*inst.env_arrays()))
        assert np.array_equal(back.op_pt, inst.op_pt)
        assert np.array_equal(back.job_length, inst.job_length)


@needs_public_instances()
def test_brandimarte_sizes():
    insts, skipped = _set_instances('Brandimarte')
    assert not skipped and [i.name for i in insts] == sorted(BRANDIMARTE)
    for inst in insts:
        n, m, (hlo, hhi), fmax = BRANDIMARTE[inst.name]
        assert (inst.n_jobs, inst.n_machines) == (n, m), inst.name
        assert hlo <= inst.job_length.min() and inst.job_length.max() <= hhi
        assert inst.eligible_counts.max() <= fmax, inst.name
        used = len(inst.used_machines)
        if inst.name in UNUSED_MACHINES:
            assert (inst.n_machines, used) == UNUSED_MACHINES[inst.name]
        else:
            assert used == inst.n_machines, inst.name
    ops = {i.name: i.n_ops for i in insts}
    assert ops == {'Mk01': 55, 'Mk02': 58, 'Mk03': 150, 'Mk04': 90,
                   'Mk05': 106, 'Mk06': 150, 'Mk07': 100, 'Mk08': 225,
                   'Mk09': 240, 'Mk10': 240}


@needs_public_instances()
@pytest.mark.parametrize('data', ['edata', 'rdata', 'vdata'])
def test_hurink_sizes(data):
    insts, skipped = _set_instances(f'Hurink_{data}')
    assert [s[0] for s in skipped] == ['orb7.fjs']
    assert sorted(i.name for i in insts) == sorted(set(HURINK) - ZERO_PT_FILES)
    flex = []
    for inst in insts:
        n, m = HURINK[inst.name]
        assert (inst.n_jobs, inst.n_machines) == (n, m), inst.name
        assert (inst.job_length == m).all() and inst.n_ops == n * m
        # machine-independent processing times (multi-purpose machines)
        for row in inst.op_pt:
            assert len(set(row[row > 0])) == 1, inst.name
        kmax = int(inst.eligible_counts.max())
        if data == 'edata':
            assert kmax <= (2 if m <= 6 else 3), inst.name
        elif data == 'rdata':
            assert kmax <= 3, inst.name
        else:
            assert kmax <= m, inst.name
        flex.append(inst.flexibility / (m if data == 'vdata' else 1))
    lo, hi = {'edata': (1.10, 1.20), 'rdata': (1.85, 2.10),
              'vdata': (0.40, 0.55)}[data]
    assert lo <= np.mean(flex) <= hi, (data, np.mean(flex))


@needs_public_instances()
@pytest.mark.parametrize('d,n,m', [('SD2/test/10x5+mix', 10, 5),
                                   ('SD2/test/20x10+mix', 20, 10),
                                   ('SD2/vali/10x5+mix', 10, 5),
                                   ('SD2/vali/20x10+mix', 20, 10)])
def test_sd2_sizes(d, n, m):
    insts, skipped = _set_instances(d)
    assert len(insts) == 100 and not skipped
    for inst in insts:
        assert (inst.n_jobs, inst.n_machines, inst.n_ops) == (n, m, n * m)
        assert (inst.job_length == m).all()
        assert inst.op_pt[inst.op_pt > 0].min() >= 1 and inst.op_pt.max() <= 99
        assert inst.eligible_counts.max() <= m


@needs_public_instances()
def test_sd2_generator_matches_published_distribution():
    pub, _ = _set_instances('SD2/test/10x5+mix')
    rng = np.random.default_rng(123)
    gen = [sd2_instance(10, 5, rng) for _ in range(100)]
    a = np.array([[i.flexibility, i.mean_pt] for i in pub])
    b = np.array([[(pt > 0).sum(1).mean(), pt[pt > 0].mean()] for _, pt in gen])
    assert all(jl.tolist() == [5] * 10 and pt.shape == (50, 5) for jl, pt in gen)
    assert abs(a[:, 0].mean() - b[:, 0].mean()) < 0.1     # both about 3
    assert abs(a[:, 1].mean() - b[:, 1].mean()) < 1.5     # both about 50
    jl1, pt1 = sd2_instance(10, 5, np.random.default_rng(7))
    jl2, pt2 = sd2_instance(10, 5, np.random.default_rng(7))
    assert np.array_equal(pt1, pt2)


@needs_public_instances('Fattahi_setup')
def test_fattahi_setup_variant():
    insts = load_set(os.path.join(PUB, 'Fattahi_setup'))
    assert len(insts) == 20
    for k, inst in enumerate(insts):
        n, m, hmax = FATTAHI_FILE_DIFFERS.get(k, FATTAHI[k])
        assert (inst.n_jobs, inst.n_machines) == (n, m), inst.name
        assert inst.job_length.max() <= hmax
        assert inst.setup is not None
        assert inst.setup.shape == (m, inst.n_ops, inst.n_ops)
        st = setup_structure(inst)
        assert st['sentinel_matches_eligibility'], inst.name
        assert st['triangle_violations'] == 0, inst.name
        assert not st['exact_in_descriptor'] and st['reasons_not_exact']
        if inst.n_machines > 1 and st['used_entries']:
            assert not st['machine_independent'], inst.name
    # total flexibility: no ineligible pair, the family env's normalization
    # assertion would fail on these two alone
    assert [i.name for i in insts if not i.has_incompatible_pair] == \
        ['Fattahi_setup_01', 'Fattahi_setup_05']


def test_parser_rejects_malformed_text():
    with pytest.raises(ValueError):
        parse_fjs_text('2 2\n1 1 1 5\n')                      # job 2 missing
    with pytest.raises(ValueError):
        parse_fjs_text('1 2\n1 1 3 5\n')                      # machine 3 > M
    with pytest.raises(ValueError):
        parse_fjs_text('1 2\n1 2 1 5 1 6\n')                  # machine twice
    with pytest.raises(ValueError):
        parse_fjs_text('1 2\n1 1 1 5\n7 7 7\n')               # stray integers
    with pytest.raises(UnrepresentableInstance):
        parse_fjs_text('1 2\n1 1 1 0\n')                      # zero time
    inst = parse_fjs_text('1 2\n2 1 1\n 5 1 2 6\n')          # wrapped record
    assert inst.op_pt.tolist() == [[5, 0], [0, 6]]


# ----------------------------------------------------------------------
# Part 1: best-known tables
# ----------------------------------------------------------------------

@needs_public_instances()
@pytest.mark.parametrize('d', list(BKS_CSV))
def test_best_known_tables(d):
    rows = _read_csv(BKS_CSV[d])
    names = {os.path.splitext(os.path.basename(p))[0]
             for p in list_instances(os.path.join(PUB, d))}
    assert {r['instance'] for r in rows} == names
    for r in rows:
        assert r['source'], r
        bks = int(r['best_known_makespan'])
        if r['lower_bound_if_known']:
            lb = int(r['lower_bound_if_known'])
            assert lb <= bks, r
            if r['optimal'] == 'yes':
                assert lb == bks, r


@needs_public_instances()
@pytest.mark.parametrize('d', list(BKS_CSV))
def test_simple_rule_not_below_best_known(d):
    """ECT dispatching on regime N, validated, is never below the best-known
    makespan (an upper bound) and never below a known lower bound."""
    rows = {r['instance']: r for r in _read_csv(BKS_CSV[d])}
    insts, _ = _set_instances(d)
    gaps = []
    for inst in insts:
        jl, pt = inst.env_arrays()
        desc = draw_rules(jl, pt, 0).descriptor('N')
        sim = reference_sim.simulate(jl, pt, desc, ect_schedule(jl, pt))
        validate(jl, pt, desc, sim)
        r = rows[inst.name]
        bks = int(r['best_known_makespan'])
        if r['lower_bound_if_known']:
            assert sim['makespan'] >= int(r['lower_bound_if_known'])
        if not d.startswith('SD2') or r['optimal'].startswith('yes'):
            assert sim['makespan'] >= bks, (inst.name, sim['makespan'], bks)
        gaps.append(sim['makespan'] / bks - 1.0)
    assert len(gaps) == len(insts)


# ----------------------------------------------------------------------
# Part 3: rule generator
# ----------------------------------------------------------------------

def test_template_constants():
    assert factory_lag_bands() == [(12, 24), (24, 48)]
    fr = factory_lag_fractions()
    assert fr[(12, 24)] == pytest.approx(2 / 22)
    assert fr[(24, 48)] == pytest.approx(1.25 / 22)
    p = DEFAULT_PARAMS
    assert np.allclose(np.array(p.lag_short_pbar) * FACTORY_MEAN_PT, (12, 24))
    assert np.allclose(np.array(p.lag_long_pbar) * FACTORY_MEAN_PT, (24, 48))
    assert window_duration_range(FACTORY_MEAN_PT) == (2, 8)
    assert (p.setup_same, p.setup_cross, p.window_density) == \
        ((0.25, 0.50), (0.50, 1.00), (0.05, 0.15))


def test_factory_reduction():
    """On factory instances the public conventions give the factory's values:
    the measured p_bar, and a horizon equal to the job-chain bound."""
    from ppvc_instance_generator import load_instance
    stems = sorted(s[:-4] for s in glob.glob('data/PPVC/10x25+ppvc-mixed/instance_*.fjs'))
    if len(stems) != 100:
        pytest.skip('factory test set not present')
    pb = []
    for k, s in enumerate(stems):
        jl, pt, meta = load_instance(s)
        pb.append(pt[pt > 0].mean())
        if k < 20:
            lag = np.asarray(meta['time_lag'])
            assert public_horizon(jl, pt, lag) == job_chain_horizon(jl, pt, lag)
    assert np.mean(pb) == pytest.approx(FACTORY_MEAN_PT, abs=1e-9)


@needs_public_instances()
def test_rules_deterministic_and_seeded():
    inst = read_fjs(os.path.join(PUB, 'Brandimarte', 'Mk04.fjs'))
    jl, pt = inst.env_arrays()
    a, b, c = draw_rules(jl, pt, 5), draw_rules(jl, pt, 5), draw_rules(jl, pt, 6)
    assert json.dumps(a.to_json()) == json.dumps(b.to_json())
    assert json.dumps(a.to_json()) != json.dumps(c.to_json())
    counts = np.bincount(a.job_class, minlength=4)
    assert counts.max() - counts.min() <= 1


def test_lag_density_matches_factory():
    target = sum(factory_lag_fractions().values())
    for key in ('SD2_test_10x5+mix', 'SD2_test_20x10+mix'):
        dens = []
        for p in sorted(glob.glob(os.path.join(PUB, FAMILY_DIRS[key], 'rules', '*.json'))):
            with open(p) as f:
                dens.append(np.mean(np.asarray(json.load(f)['lag']) > 0))
        assert len(dens) == 100
        assert abs(np.mean(dens) - target) < 0.01, (key, np.mean(dens), target)


@needs_public_instances()
@pytest.mark.parametrize('key', list(FAMILY_DIRS))
def test_stored_family_records(key):
    """Every stored record: C1 to C3 pass, C4 never fails, members shared
    across regimes, equal to a fresh draw from its seed, inside the ranges."""
    fam = os.path.join(PUB, FAMILY_DIRS[key])
    with open(os.path.join(fam, 'dataset_meta.json')) as f:
        meta = json.load(f)
    assert meta['regimes'] == list(REGIMES_ALL)
    assert meta['desc_seed0'] == SET_SEED0[key]
    for name in meta['instances']:
        recs = {}
        for regime in REGIMES_ALL:
            jl, pt, desc, rec = load_family_record(
                os.path.join(fam, 'desc', regime, f'{name}.desc.json'))
            recs[regime] = (desc, rec)
            assert checks.check_schema_and_values(desc, pt, jl).status == PASS
            assert checks.check_setup_convention(desc).status == PASS
            assert checks.check_family_membership(desc).status == PASS
            assert checks.check_magnitude_range(desc, pt).status != FAIL
        seed = recs['N'][1]['desc_seed']
        rules = draw_rules(jl, pt, seed)
        assert not magnitude_violations(rules, jl, pt), name
        for regime, (desc, rec) in recs.items():
            fresh = rules.descriptor(regime).to_json()
            # a fresh draw states its convention; the stored records predate
            # the field and are read with the loader's statement
            assert fresh.pop('setup_convention', None) == \
                ('attached' if 'S' in regime else None), (name, regime)
            assert 'setup_convention' not in rec
            assert {k: rec[k] for k in fresh} == fresh, (name, regime)
            assert rec['horizon'] == rules.horizon
            assert np.array_equal(desc.op_class, rules.op_class)
            if 'L' in regime:
                assert np.array_equal(desc.lag, rules.lag)
            else:
                assert not desc.lag.any()
            if 'S' in regime:
                assert np.array_equal(desc.setup, rules.setup)
            else:
                assert desc.setup is None
            if 'W' in regime:
                assert desc.windows == [list(map(tuple, w)) for w in rules.windows]
            else:
                assert desc.windows is None


@needs_public_instances()
@pytest.mark.parametrize('key', ('SD2_vali_10x5+mix', 'SD2_test_20x10+mix'))
def test_magnitude_violations_report_setups_below_the_range(key):
    """Setups scaled down to a quarter of their draw (rounded, at least 1 h)
    fall below the declared ranges and are reported, as check C4 reports them
    on the factory benchmark; the draw itself is inside."""
    import copy
    fam = os.path.join(PUB, FAMILY_DIRS[key])
    with open(os.path.join(fam, 'dataset_meta.json')) as f:
        names = json.load(f)['instances'][:10]
    for name in names:
        jl, pt, _, rec = load_family_record(
            os.path.join(fam, 'desc', 'N', f'{name}.desc.json'))
        rules = draw_rules(jl, pt, rec['desc_seed'])
        assert not magnitude_violations(rules, jl, pt), name
        low = copy.copy(rules)
        low.setup = np.maximum(1, np.floor(rules.setup * 0.25 + 0.5)).astype(int)
        out = magnitude_violations(low, jl, pt)
        assert out and all(v.startswith('setup[') for v in out), (name, out)


def _sample_instances():
    """A small sample covering every base set, unequal operation counts and
    machines that no operation uses."""
    picks = [('Brandimarte', 'Mk01'), ('Brandimarte', 'Mk06'),
             ('Brandimarte', 'Mk10'), ('Hurink_edata', 'la01'),
             ('Hurink_rdata', 'car3'), ('Hurink_vdata', 'la16'),
             ('Hurink_vdata', 'abz5'), ('SD2/test/10x5+mix', '10x5+mix_001'),
             ('SD2/test/20x10+mix', '20x10+mix_001')]
    if not instances_present():
        return [pytest.param(None, id='instances', marks=needs_public_instances())]
    return [read_fjs(os.path.join(PUB, d, f'{n}.fjs')) for d, n in picks]


@pytest.mark.parametrize('inst', _sample_instances(), ids=lambda i: i.name)
def test_random_dispatch_valid_and_monotone(inst):
    jl, pt = inst.env_arrays()
    rules = draw_rules(jl, pt, 11)
    rng = np.random.default_rng(3)
    chains = [('N', 'L', 'LS', 'LSW'), ('N', 'S', 'SW', 'LSW'),
              ('N', 'W', 'LW', 'LSW')]
    for _ in range(3):
        acts = random_actions(jl, pt, rng)
        ms = {}
        for regime in REGIMES_ALL:
            desc = rules.descriptor(regime)
            sim = reference_sim.simulate(jl, pt, desc, acts)
            validate(jl, pt, desc, sim)
            ms[regime] = sim['makespan']
        for ch in chains:
            vals = [ms[r] for r in ch]
            assert vals == sorted(vals), (inst.name, ch, vals)


@pytest.mark.parametrize('inst', _sample_instances(), ids=lambda i: i.name)
def test_family_env_matches_reference(inst):
    """FJSPEnvFamily on a public instance: random legal actions from its mask
    and MWKR; the realized schedule equals the reference simulator's replay
    and passes both validators, under every regime."""
    jl, pt = inst.env_arrays()
    rules = draw_rules(jl, pt, 11)
    M = pt.shape[1]
    for regime in REGIMES_ALL:
        desc = rules.descriptor(regime)
        for policy in ('random', 'MWKR'):
            env = FJSPEnvFamily(jl.shape[0], M, family_mode=True)
            env.set_initial_data([jl], [pt], descriptor_list=[desc])
            rng = np.random.default_rng(1)
            np.random.seed(50)
            acts, amch = [], np.full(pt.shape[0], -1)
            while not env.done().all():
                if policy == 'random':
                    a = int(rng.choice(np.flatnonzero(~env.dynamic_pair_mask[0].reshape(-1))))
                else:
                    a = int(heuristic_select_action('MWKR', env))
                j, m = a // M, a % M
                amch[env.candidate[0, j]] = m
                acts.append((j, m))
                env.step(np.array([a]))
            sim = reference_sim.simulate(jl, pt, desc, acts)
            assert np.allclose(sim['op_ct'], env.true_op_ct[0]), (inst.name, regime)
            assert env.current_makespan[0] == sim['makespan']
            assert np.array_equal(sim['op_mch'], amch)
            validate(jl, pt, desc, sim)
