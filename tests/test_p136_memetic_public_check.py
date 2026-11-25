"""The memetic algorithm on the Brandimarte instances without rules
(scripts/p136_memetic_public_check.py): the cells are the published instances, the start
schedule is the best valid masked rule, the validity rule of a record, and the gaps of the
analysis on synthetic records. No timed run."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=None)

import p136_memetic_public_check as M  # noqa: E402

PUBLIC_FILES = [os.path.join(ROOT, 'data', 'PUBLIC', 'Brandimarte', f'{n}.fjs')
                for n in ('Mk01', 'Mk06', 'Mk10')]
needs_instances = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in PUBLIC_FILES),
    reason='needs the Brandimarte instance files under data/PUBLIC/Brandimarte, which the '
           'release does not hold (scripts/fetch_public_instances.py --sets Brandimarte '
           'fetches them)')


def cpu_params():
    from params import configs
    configs.device = 'cpu'


@needs_instances
def test_every_cell_is_a_published_instance_with_a_best_known_value():
    bk = M.best_known()
    assert sorted(bk) == list(M.NAMES) and len(bk) == 10
    assert bk['Mk01'] == 40 and bk['Mk10'] == 197
    cpu_params()
    for name in ('Mk01', 'Mk06', 'Mk10'):
        jl, pt, desc, inst, drec = M.load_cell(name)
        assert drec['regime'] == 'N' and not any(drec['active'])
        assert drec['base_instance'] == f'data/PUBLIC/Brandimarte/{name}'
        assert inst.lag.sum() == 0 and inst.rel_op.sum() == 0
        assert np.asarray(inst.S).sum() == 0 and np.asarray(inst.nw).sum() == 0
        assert inst.N == int(np.sum(jl))


@needs_instances
def test_start_schedule_is_the_best_valid_masked_rule():
    cpu_params()
    from baselines.memetic.validate import validate
    jl, pt, desc, inst, _ = M.load_cell('Mk01')
    amch, ct, rule, per = M.rule_start(jl, pt, desc, inst)
    assert set(per) == set(M.RULE_ORDER) and all(v is not None for v in per.values())
    assert per[rule] == min(per.values())
    assert rule == min(M.RULE_ORDER, key=lambda r: (per[r], M.RULE_ORDER.index(r)))
    ok, _, ms = validate(inst, amch, ct - inst.pt[np.arange(inst.N), amch], ct)
    assert ok and ms == per[rule]


def good_record(**kw):
    rec = dict(validated=True, budget_clock='wall', threads=1, workers=1, affinity=[9],
               run_cores=list(range(8, 18)), cpu_share=0.97, development_run=False,
               idle_before={'idle': True, 'busy_processes': []})
    rec.update(kw)
    return rec


def test_validity_rule_of_a_record():
    assert M.classify(good_record()) == (True, 'share')
    assert not M.classify(good_record(development_run=True))[0]
    assert not M.classify(good_record(idle_before={'idle': False, 'busy_processes': ['x']}))[0]
    assert not M.classify(good_record(cpu_share=0.85))[0]
    assert not M.classify(good_record(affinity=[3]))[0]
    assert not M.classify(good_record(affinity=[8, 9]))[0]
    assert not M.classify(good_record(threads=2))[0]
    assert not M.classify(good_record(budget_clock='cpu'))[0]
    assert not M.classify(good_record(run_cores=[]))[0]
    assert M.classify(good_record(affinity=[3]), cores=[3])[0]


def write_world(root, names, seeds):
    """Records whose makespans give gaps of 2 s % at 60 s and (s - 1) % at 600 s for seed s,
    and a start schedule 10 % above the best known."""
    bk = M.best_known()
    for name in names:
        for s in seeds:
            b = bk[name]
            rec = good_record(instance=name, seed=s, valid=True,
                              start_schedule={'makespan': 1.1 * b},
                              readings={'60': b * (1 + 0.02 * s), '600': b * (1 + 0.01 * (s - 1))})
            p = M.record_path(root, 600.0, 'rule', name, s)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'w') as f:
                json.dump(rec, f)


def args(*extra):
    a = M.build_parser().parse_args(['infer', *extra])
    a.root = os.path.abspath(a.root)
    a.out = os.path.abspath(a.out) if a.out else None
    a.names = [x for x in a.instances.split(',') if x]
    a.seeds = M.TCP.parse_instances(a.seeds)
    a.cores_allowed = None
    return a


def test_infer_gives_the_gaps_of_hand_computed_records(tmp_path):
    root, out = tmp_path / 'root', tmp_path / 'out'
    write_world(str(root), M.NAMES, M.SEEDS)
    M.infer(args('--root', str(root), '--out', str(out)))
    res = json.load(open(out / M.OUT_FILES['json']))
    assert res['run'] == 'full' and res['cells_valid'] == 30
    P = res['results']['pooled']
    assert P['60']['mean_gap'] == pytest.approx(4.0) and P['60']['at_best_known'] == 0
    assert P['600']['mean_gap'] == pytest.approx(1.0) and P['600']['at_best_known'] == 10
    y = res['results']['instances']['Mk07']
    assert y['min_gap_60'] == pytest.approx(2.0) and y['min_gap_600'] == pytest.approx(0.0)
    assert res['results']['start_mean_gap'] == pytest.approx(10.0)
    tex = open(out / M.OUT_FILES['macros']).read()
    assert '\\newcommand{\\MpcMeanGapSixty}{4.00}' in tex
    assert '\\newcommand{\\MpcMeanGapSixHundred}{1.00}' in tex
    assert '\\newcommand{\\MpcMkTenMinGapSixHundred}{0.00}' in tex
    assert 'PARTIAL' not in tex


def test_infer_refuses_a_full_write_with_a_missing_cell_and_marks_a_partial_run(tmp_path,
                                                                                monkeypatch):
    root, out = tmp_path / 'root', tmp_path / 'out'
    write_world(str(root), M.NAMES, M.SEEDS)
    os.unlink(M.record_path(str(root), 600.0, 'rule', 'Mk04', 2))
    with pytest.raises(SystemExit, match='set PACT_OUT'):
        M.infer(args('--root', str(root)))
    monkeypatch.setenv('PACT_OUT', str(tmp_path / 'pact'))
    with pytest.raises(SystemExit, match='1 of 30 cells without a valid record'):
        M.infer(args('--root', str(root)))
    assert not (tmp_path / 'pact').exists()
    monkeypatch.delenv('PACT_OUT')
    M.infer(args('--root', str(root), '--out', str(out)))
    res = json.load(open(out / M.OUT_FILES['json']))
    assert res['run'] == 'partial' and res['cells_valid'] == 29
    assert res['results']['instances']['Mk04']['n_60'] == 2
    assert 'PARTIAL' in open(out / M.OUT_FILES['macros']).read()


def test_record_folder_is_outside_the_folders_the_analysis_reads():
    p = M.record_path(ROOT, 600.0, 'rule', 'Mk01', 1)
    rel = os.path.relpath(p, ROOT)
    assert rel.startswith(os.path.join('or_solution', 'PUBFAMILY', 'memetic_p136') + os.sep)
    assert not rel.startswith(os.path.join('or_solution', 'FAMILY'))
