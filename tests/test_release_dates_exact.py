"""Release dates in the exact environment (scripts/p69_release_dates.py, phases plan, evaluate,
check and analyze).

  * one real unit (combination N, the first test instance; every policy seed and the four
    dispatching rules, with and without release dates) evaluated as part 0 of two: the unit of
    part 1 is not written, a second call skips the existing unit, every schedule passes the
    validators and no makespan lies below the makespan estimate, and the check confirms that
    the rollouts without release dates reproduce the stored arrays exactly;
  * the check refuses while a unit of the design is missing;
  * the analysis on synthetic units built from the stored arrays (release dates add 10% to
    every policy makespan and 5% to every rule makespan): the four outputs are written, the
    increase of the policy's mean makespan is 10%, every macro name is Rel plus letters, and a
    missing or foreign unit stops the analysis with no output left behind;
  * plan prints one command per part.

Run from the repository root:  python -m pytest tests/test_release_dates_exact.py -q
CPU only, one thread; about 20 s, nearly all of it the one real unit.
"""
import copy
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402

PY = sys.executable
SCRIPT = 'scripts/p69_release_dates.py'
SEEDS = (301, 302, 303, 304, 305)
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
R8 = list(REG.COMBOS)

needs_runs = pytest.mark.skipif(
    not all(os.path.exists(os.path.join(ROOT, f'trained_network/FAMILY/10x25+family+xmask-v1-s{s}.pth'))
            for s in SEEDS) or not os.path.isdir(os.path.join(ROOT, 'test_results/FAMILY/10x25_xmask')),
    reason='needs the policy checkpoints and the stored arrays of the exact environment')


def env_of(out):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=str(out), CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def p69(out, *args, timeout=600):
    return subprocess.run([PY, '-B', SCRIPT, *args], cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout, env=env_of(out))


@pytest.fixture(scope='module')
def real_unit(tmp_path_factory):
    base = tmp_path_factory.mktemp('p69')
    root = base / 'units'
    r = p69(base / 'out', 'evaluate', '--n', '1', '--combos', 'N,L', '--parts', '2',
            '--part', '0', '--root', str(root))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return base, root, r.stdout


@needs_runs
def test_part_zero_writes_its_unit_only_and_skips_it_the_second_time(real_unit):
    base, root, out = real_unit
    assert (root / '10x25' / 'N' / 'instance_000.json').exists()
    assert not (root / '10x25' / 'L').exists()
    assert '1 units written, 0 existed' in out
    r = p69(base / 'out', 'evaluate', '--n', '1', '--combos', 'N,L', '--parts', '2',
            '--part', '0', '--root', str(root))
    assert r.returncode == 0 and '0 units written, 1 existed' in r.stdout, r.stdout + r.stderr


@needs_runs
def test_validators_and_estimate_on_the_real_unit(real_unit):
    _, root, _ = real_unit
    rec = json.load(open(root / '10x25' / 'N' / 'instance_000.json'))
    assert rec['design']['mask_options'] == {'first_mask': 'exact', 'mask_ties': 'exact'}
    assert rec['design']['rule_set'] == 'masked'
    assert sorted({r['arm'] for r in rec['rows']}) == sorted([f'pact:s{s}' for s in SEEDS] + list(RULES))
    assert len(rec['rows']) == 2 * (len(SEEDS) + len(RULES))
    for r in rec['rows']:
        assert r['replay_ok'] and r['viol_ref'] == 0 and r['viol_sv'] == 0 and r['early_first'] == 0
        assert r['bound_viol'] == 0 and r['reward_viol'] == 0


@needs_runs
def test_check_reproduces_the_stored_arrays_and_refuses_a_missing_unit(real_unit):
    base, root, _ = real_unit
    r = p69(base / 'out', 'check', '--n', '1', '--combos', 'N', '--root', str(root))
    assert r.returncode == 0 and 'reproduce' in r.stdout and 'exactly' in r.stdout, r.stdout + r.stderr
    r = p69(base / 'out', 'check', '--n', '1', '--combos', 'N,L', '--root', str(root))
    assert r.returncode == 1 and 'missing' in r.stdout, r.stdout + r.stderr


def stored(tag, combo, n):
    folder = os.path.join(ROOT, 'test_results/FAMILY/10x25_xmask')
    a = np.load(os.path.join(folder, f'Result_{tag}_{combo}.npy'))
    return (a[:, 0] if a.ndim > 1 else a)[:n].astype(float)


def synthetic_units(template, root, n, skip=None, foreign=None):
    """Units of the first n instances on the eight combinations, rows from the stored arrays:
    without release dates the stored makespan, with release dates 10% more for the policy and
    5% more for a rule."""
    names = [f'instance_{i:03d}' for i in range(n)]
    proto = template['rows'][0]
    for combo in R8:
        for i, nm in enumerate(names):
            if (combo, nm) == skip:
                continue
            rows = []
            for rel in (False, True):
                for arm in [f'pact:s{s}' for s in SEEDS] + list(RULES):
                    tag = (f'greedy+10x25+family+xmask-v1-s{arm[6:]}' if arm.startswith('pact:')
                           else arm)
                    m = stored(tag, combo, n)[i]
                    if rel:
                        m = round(m * (1.10 if arm.startswith('pact:') else 1.05), 6)
                    row = dict(proto, regime=combo, name=nm, arm=arm, release=rel, makespan=m)
                    rows.append(row)
            design = copy.deepcopy(template['design'])
            if (combo, nm) == foreign:
                design['draw_seed'] = -1
            p = root / '10x25' / combo / f'{nm}.json'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(dict(template, design=design, combination=combo, instance=nm,
                                         rows=rows)))


@needs_runs
def test_analysis_on_synthetic_units(real_unit, tmp_path):
    _, root, _ = real_unit
    template = json.load(open(root / '10x25' / 'N' / 'instance_000.json'))
    synth = tmp_path / 'synth'
    synthetic_units(template, synth, 3)
    out = tmp_path / 'out'
    r = p69(out, 'analyze', '--n', '3', '--boot', '300', '--root', str(synth))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    for p in ('manuscript/macros_release.tex', 'manuscript/tables/release_table.tex',
              'results/release_dates.json', 'results/release_dates.md'):
        assert (out / p).exists(), p
    rec = json.load(open(out / 'results/release_dates.json'))
    assert rec['results']['increase'] == pytest.approx(10.0, abs=1e-6)
    assert rec['reproduction']['compared'] == 3 * 8 * (len(SEEDS) + len(RULES))
    assert rec['results']['release']['audit']['rejected'] == 0
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', (out / 'manuscript/macros_release.tex').read_text())
    assert names and all(re.fullmatch(r'Rel[A-Za-z]+', x) for x in names)


@needs_runs
@pytest.mark.parametrize('kind', ['missing', 'foreign'])
def test_analysis_refuses_a_missing_or_foreign_unit(real_unit, tmp_path, kind):
    _, root, _ = real_unit
    template = json.load(open(root / '10x25' / 'N' / 'instance_000.json'))
    synth = tmp_path / 'synth'
    if kind == 'missing':
        synthetic_units(template, synth, 3, skip=('W', 'instance_001'))
    else:
        synthetic_units(template, synth, 3, foreign=('W', 'instance_001'))
    out = tmp_path / 'out'
    r = p69(out, 'analyze', '--n', '3', '--boot', '300', '--root', str(synth))
    assert r.returncode == 2 and 'do not count' in r.stderr, r.stdout[-2000:] + r.stderr[-2000:]
    assert ('missing' if kind == 'missing' else 'another design') in r.stderr
    assert not (out / 'manuscript/macros_release.tex').exists()


@needs_runs
def test_plan_prints_one_command_per_part(tmp_path):
    r = p69(tmp_path / 'out', 'plan', '--parts', '3')
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [x for x in r.stdout.splitlines() if 'scripts/p69_release_dates.py evaluate' in x]
    assert len(cmds) == 3
    assert all(f'--part {i} --parts 3' in c for i, c in enumerate(cmds))
    assert '800 units' in r.stdout
