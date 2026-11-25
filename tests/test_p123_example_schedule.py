"""scripts/p123_example_schedule.py: example schedules of the policy for the main text.

  * --dry checks and draws two synthetic schedules in a given folder, without importing torch
    or params.py, and writes the figure (PDF with the Times face only, PNG) and its record;
    the drawing is 6.324 in wide and at most 4.7 in tall, the group labels and the legend
    keep their distances;
  * the synthetic schedules pass the schedule checks of p74_gantt and the independent
    validator; a precedence violation, a block inside an unavailability window and an overlap
    on a station are each refused;
  * a makespan that differs from the stored value stops the script and names both values;
  * the stations are grouped by type in row order, and types that are not adjacent are refused;
  * --check lists a missing checkpoint, configuration and stored arrays and exits non-zero;
  * the evaluation arguments are those of the queue's job for this run
    (scripts/eval_newseeds_plan.py jobs, read only).

Synthetic schedules under the pytest temporary folder; nothing in the repository is written and
no model is loaded. Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 3 python -B -m pytest -q \
      tests/test_p123_example_schedule.py
"""
import copy
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_FIRST_MASK=None, PACT_MASK_TOL=None)

import p123_example_schedule as P  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
FACTORY_TYPES = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 8]


@pytest.fixture(scope='module')
def case():
    return P.synthetic_case()


def test_dry_draws_without_torch(tmp_path):
    out = tmp_path / 'dry'
    code = ('import sys; sys.path.insert(0, "scripts"); import p123_example_schedule as P; '
            f'P.main(["--dry", "--dry_dir", {str(out)!r}]); '
            'print("TORCH", "torch" in sys.modules, "PARAMS", "params" in sys.modules)')
    r = subprocess.run([PY, '-B', '-c', code], cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'TORCH False PARAMS False' in r.stdout, r.stdout
    for f in ('dry_schedule.pdf', 'dry_schedule.png', 'dry_schedule.json'):
        assert (out / f).exists(), f
    doc = json.loads((out / 'dry_schedule.json').read_text())
    fig = doc['figure']
    assert abs(fig['width_in'] - 455.33 / 72) < 1e-3
    assert fig['height_in'] <= 4.7 + 1e-9
    assert fig['legend_to_panel_in'] >= 0.04
    assert fig['min_gap_between_group_labels_in'] > 0.01
    assert [g[0] for g in fig['station_groups']] == [
        'formwork', 'concrete pour', 'MEP fit-out', 'finishing', 'painting', 'assembly',
        'dispatch', 'steel fabrication', 'quality control']
    if shutil.which('pdffonts'):
        fonts = subprocess.run(['pdffonts', str(out / 'dry_schedule.pdf')],
                               capture_output=True, text=True, check=True).stdout
        names = [ln.split()[0] for ln in fonts.splitlines()[2:] if ln.strip()]
        assert names and all('TeXGyreTermes' in n for n in names), names


def test_dry_command_line(tmp_path):
    r = subprocess.run([PY, '-B', 'scripts/p123_example_schedule.py', '--dry', '--dry_dir',
                        str(tmp_path)], cwd=ROOT, env=ENV, capture_output=True, text=True,
                       timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (tmp_path / 'dry_schedule.pdf').exists()


def test_synthetic_schedules_pass(case):
    from constraint_family.reference_sim import validate_schedule
    jl, pt, mch_type, descs, cases = case
    for combo in P.COMBOS:
        rec, sim = cases[combo]
        P.check_schedule(rec, P.windows_of(descs[combo], len(mch_type)), jl, pt, descs[combo])
        by_op = {r['op']: r for r in rec}
        n = pt.shape[0]
        st = np.array([by_op[o]['proc_start'] for o in range(n)])
        ct = np.array([by_op[o]['completion'] for o in range(n)])
        mch = np.array([by_op[o]['machine'] for o in range(n)])
        assert validate_schedule(jl, pt, descs[combo], st, ct, mch) == []
        assert max(r['completion'] for r in rec) == sim['makespan']
    # all three rules leave a trace in the synthetic LSW schedule
    rec = cases['LSW'][0]
    assert any(r['setup'] > 0 for r in rec) and any(r['lag'] > 0 for r in rec)


def _shift(r, d):
    for k in ('block_start', 'proc_start', 'completion', 'successor_ready'):
        r[k] += d


def test_precedence_violation_refused(case):
    jl, pt, mch_type, descs, cases = case
    rec = copy.deepcopy(cases['LSW'][0])
    by_op = {r['op']: r for r in rec}
    lagged = [r for r in rec if r['lag'] > 0][0]
    succ = by_op[lagged['op'] + 1]
    _shift(succ, lagged['successor_ready'] - succ['block_start'] - 1.0)   # 1 h too early
    with pytest.raises(SystemExit):
        P.check_schedule(rec, P.windows_of(descs['LSW'], len(mch_type)), jl, pt, descs['LSW'])


def test_window_violation_refused(case):
    jl, pt, mch_type, descs, cases = case
    rec = cases['N'][0]
    outages = [[] for _ in mch_type]
    r = rec[0]
    outages[r['machine']] = [(r['proc_start'], r['proc_start'] + 1.0)]
    with pytest.raises(SystemExit):
        P.check_schedule(rec, outages, jl, pt, descs['N'])


def test_station_overlap_refused(case):
    jl, pt, mch_type, descs, cases = case
    rec = copy.deepcopy(cases['N'][0])
    by_m = {}
    for r in rec:
        by_m.setdefault(r['machine'], []).append(r)
    two = next(v for v in by_m.values() if len(v) >= 2)
    a, b = sorted(two, key=lambda r: r['block_start'])[:2]
    _shift(b, a['completion'] - b['block_start'] - 1.0)   # starts 1 h before a completes
    with pytest.raises(SystemExit):
        P.check_schedule(rec, [[] for _ in mch_type], jl, pt, descs['N'])


def test_stored_makespan_comparison():
    P.compare_stored('LSW', 289.0, 289.0)
    with pytest.raises(SystemExit) as e:
        P.compare_stored('LSW', 289.0, 291.0)
    assert '289' in str(e.value) and '291' in str(e.value)


def test_station_groups():
    groups = P.station_groups(np.array(FACTORY_TYPES))
    assert [(P.STATION_WORDS[t], a, b) for t, a, b in groups] == [
        ('formwork', 0, 2), ('concrete pour', 3, 4), ('MEP fit-out', 5, 8),
        ('finishing', 9, 12), ('painting', 13, 15), ('assembly', 16, 18), ('dispatch', 19, 19),
        ('steel fabrication', 20, 21), ('quality control', 22, 24)]
    y = P.row_positions(groups)
    assert y[3] - y[2] == pytest.approx(1 + P.GROUP_GAP) and y[1] - y[0] == 1
    with pytest.raises(SystemExit):
        P.station_groups(np.array([0, 1, 0]))


def test_check_lists_missing(monkeypatch, capsys):
    monkeypatch.setattr(P, 'MODEL', 'no-such-policy-s000')
    assert P.main(['--check']) == 1
    out = capsys.readouterr().out
    assert 'trained_network/FAMILY/no-such-policy-s000.pth' in out
    assert 'train_log/FAMILY/config_no-such-policy-s000.json' in out
    assert out.count('stored greedy makespans') == len(P.COMBOS)


def test_run_refuses_missing_inputs(monkeypatch):
    monkeypatch.setattr(P, 'MODEL', 'no-such-policy-s000')
    with pytest.raises(SystemExit) as e:
        P.main([])
    assert 'missing' in str(e.value)


def test_evaluation_arguments_follow_the_queue():
    r = subprocess.run([PY, '-B', 'scripts/eval_newseeds_plan.py', 'jobs'], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    job = [ln.split('|') for ln in r.stdout.splitlines()
           if ln.startswith(f'exact:10x25:{P.MODEL.split("+")[-1]}|')]
    assert len(job) == 1, job
    (jid, _g, _c, envn, device, omp, tt, extra, model, methods, data, base, tag, limit,
     regimes, token, blind, *_rest) = job[0]
    assert (envn, device, omp, tt, extra) == ('exact', 'cpu', '1', '1', '-')
    assert (model, methods, data, base, tag, token, blind) == (
        P.MODEL, 'greedy', P.DATA, P.BASE, P.TAG, 'true', '-')
    assert set(P.COMBOS) <= set(regimes.split())
    a = P.EVAL_ARGV
    for flag, val in (('--family_model', model), ('--data', data), ('--base', base),
                      ('--methods', methods), ('--device', 'cpu'), ('--device_id', ''),
                      ('--torch_threads', tt), ('--first_mask', 'exact'),
                      ('--mask_ties', 'exact'), ('--tag', tag)):
        assert a[a.index(flag) + 1] == val, flag


def test_macro_lines_hold_the_two_makespans_and_refuse_an_unequal_record():
    doc = {'model': 'm', 'instance_index': 0, 'instance': 'data/x/instance_000',
           'combinations': {c: {'makespan': ms, 'stored_makespan': ms, 'equal_to_stored': True}
                            for c, ms in (('N', 125.0), ('LSW', 289.5))}}
    L = P.macro_lines(doc)
    assert L[0].startswith('% Generated by scripts/p123_example_schedule.py')
    assert '\\newcommand{\\ExmMakespanN}{125}' in L[2]
    assert '\\newcommand{\\ExmMakespanLSW}{289.5}' in L[3]
    bad = copy.deepcopy(doc)
    bad['combinations']['LSW']['stored_makespan'] = 290.0
    with pytest.raises(SystemExit):
        P.macro_lines(bad)
