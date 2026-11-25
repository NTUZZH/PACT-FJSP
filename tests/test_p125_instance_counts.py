"""scripts/p125_instance_counts.py: the principal comparison in counts and in hours.

  * on p110's synthetic tree with an incomplete second five seeds: the five seeds of p88 are
    used and said to be used, the five-seed record is reproduced, the pooled hour mean and
    the pair shares equal those of the arrays, every share triple sums to 100;
  * with the second five seeds complete and p110's output present: the ten seeds of p110;
  * a five-seed record that the arrays do not reproduce refuses and leaves no output;
  * --check reports a missing array and returns 1;
  * per-instance means below 0.5 h count as equal; the prefix Cnt is refused once another .tex
    file defines a macro with it.

Run from the repository root on one core:  python -B -m pytest -q tests/test_p125_instance_counts.py
CPU only; every file is written under the pytest temporary folder.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_ENV=None, PACT_SEEDS=None, PACT_OUT=None,
                                        PACT_RESULTS=None)

import p125_instance_counts as C  # noqa: E402

P110, SE, F88 = C.P110, C.SE, C.F88
BOOT = 300
PY = sys.executable


def test_five_seed_branch(tmp_path):
    res, names, outs = C.dry_run(tmp_path / 'w', boot=BOOT, ten=False)
    assert res['branch'] == 'five' and res['seeds'] == list(SE.FIRST_SEEDS)
    assert 'not complete' in res['seeds_reason']
    assert res['reproduction']['five']['compared'] > 100 and 'ten' not in res['reproduction']
    assert C.dry_verify(res, str(tmp_path / 'w')) == []
    for r, x in res['rows'].items():
        for kind in ('pairs', 'instances'):
            c = x[kind]
            assert c['n_shorter'] + c['n_equal'] + c['n_longer'] == c['n']
        assert x['pairs']['n'] == 5 * 100 * len(x['combinations'])
    assert names and all(n.startswith('Cnt') and n.isalpha() for n in names)
    for p in outs.values():
        assert os.path.exists(p)
    text = open(outs['macros']).read()
    assert '\\newcommand{\\CntHoursAll}' in text and '\\newcommand{\\CntInstEqualHeld}' in text
    assert 'five seeds of p88' in open(outs['md']).read()


def test_ten_seed_branch(tmp_path):
    res, names, _ = C.dry_run(tmp_path / 'w', boot=BOOT, ten=True)
    assert res['branch'] == 'ten' and res['seeds'] == list(SE.ALL_SEEDS)
    assert res['reproduction']['ten']['identical']
    assert C.dry_verify(res, str(tmp_path / 'w')) == []
    assert res['rows']['All']['pairs']['n'] == 10 * 100 * 8


def test_record_that_does_not_reproduce_refuses(tmp_path):
    tree = str(tmp_path / 'w')
    keep = (F88.B, F88.ALPHA)
    try:
        with P110.environment(**P110.dry_paths(tree)):
            P110.build_synthetic(tree, boot=BOOT)
            p = SE.record_path()
            rec = json.load(open(p))
            rec['parity']['components']['L']['mean'] += 0.05
            json.dump(rec, open(p, 'w'))
            with pytest.raises(SE.Refused) as e:
                C.run(root=tree, dry=True)
            assert 'does not reproduce' in str(e.value)
            for q in C.outputs().values():
                assert not os.path.exists(q)
    finally:
        F88.configure(*keep)


def test_check_reports_a_missing_array(tmp_path, capsys):
    tree = str(tmp_path / 'w')
    keep = (F88.B, F88.ALPHA)
    try:
        with P110.environment(**P110.dry_paths(tree)):
            P110.build_synthetic(tree, boot=BOOT)
            os.remove(SE.result_path('specialist', 'W', SE.SECOND_SEEDS[0], 'W'))
            assert C.check(root=tree) == 0
            os.remove(SE.result_path('released', None, SE.FIRST_SEEDS[1], 'LS'))
            assert C.check(root=tree) == 1
    finally:
        F88.configure(*keep)
    out = capsys.readouterr().out
    assert 'every input present' in out and 'seed 302 on LS' in out and 'p125: incomplete' in out


def test_threshold_and_shares():
    assert C.threshold([0.4, -0.4, 0.5, -0.6, 0.0]).tolist() == [0.0, 0.0, 0.5, -0.6, 0.0]
    c = C.counts(np.array([[-1.0, 0.0], [2.0, 3.0]]))
    assert (c['n_shorter'], c['n_equal'], c['n_longer']) == (1, 1, 2)
    assert c['shorter'] == 25.0 and c['longer'] == 50.0


def test_prefix_is_checked(tmp_path):
    own = tmp_path / 'manuscript' / 'macros_counts.tex'
    own.parent.mkdir()
    lines = ['\\newcommand{\\CntSeedN}{5}  % x']
    assert C.check_names(lines, str(own)) == ['CntSeedN']
    (own.parent / 'other.tex').write_text('\\newcommand{\\CntFoo}{1}\n')
    with pytest.raises(SE.Refused):
        C.check_names(lines, str(own))


def test_command_line_dry(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    p = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p125_instance_counts.py'),
                        '--dry', str(tmp_path / 'w'), '--boot', str(BOOT)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert 'self-test on synthetic arrays passed' in p.stdout
