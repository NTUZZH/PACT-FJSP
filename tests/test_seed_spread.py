"""The spread of the mean makespan across training seeds (scripts/p112_seed_spread.py).

  * synthetic arrays with a known per-seed deviation of each arm: the script returns the
    spread of every arm on every combination, for the first five seeds and, once the 45 runs
    of the second five are complete, for the second five and for all ten;
  * while the second five are incomplete the two later seed sets are left out, and the macro
    file, the JSON and the Markdown say so;
  * the paired test of equal variances of Pitman and Morgan against a value computed by hand;
  * the macro names: letters only, one prefix, no clash with a macro defined under manuscript/;
  * the read-only check: exit 0 on the stored arrays of the first five seeds, 1 on a folder
    without them.

Run from the repository root:  python -m pytest tests/test_seed_spread.py -q
CPU only, nothing is trained or evaluated; every file is written under a temporary folder.
"""
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import p112_seed_spread as P  # noqa: E402
import seed_extension as SE  # noqa: E402

R8 = list(REG.COMBOS)
PY = sys.executable

needs_data = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, 'data/FAMILY/10x25/desc')),
                                reason='needs the descriptors of data/FAMILY/10x25')


@pytest.fixture(autouse=True)
def at_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def test_pitman_morgan_against_a_hand_computed_case():
    # x = (1, 2, 3, 5), y = (2, 1, 4, 3): sums u = (3, 3, 7, 8), differences w = (-1, 1, -1, 2);
    # centred u = (-2.25, -2.25, 1.75, 2.75), centred w = (-1.25, 0.75, -1.25, 1.75);
    # sum of products 3.75, sums of squares 20.75 and 6.75, so r = 3.75 / sqrt(140.0625).
    # With n - 2 = 2 degrees of freedom the two-sided p of t = r sqrt(2 / (1 - r^2)) is 1 - |r|.
    r = 3.75 / math.sqrt(20.75 * 6.75)
    t = r * math.sqrt(2.0 / (1.0 - r * r))
    got = P.pitman_morgan([1, 2, 3, 5], [2, 1, 4, 3])
    assert got['df'] == 2
    assert got['r'] == pytest.approx(r, abs=1e-12)
    assert got['t'] == pytest.approx(t, abs=1e-12)
    assert got['p'] == pytest.approx(1.0 - r, abs=1e-12)
    # equal variances: x and y with the same spread give r = 0 and p = 1
    eq = P.pitman_morgan([1, 2, 3, 4], [4, 1, 3, 2])
    assert eq['r'] == pytest.approx(0.0, abs=1e-12) and eq['p'] == pytest.approx(1.0)


def test_pitman_morgan_refuses_too_few_pairs():
    with pytest.raises(P.Refused):
        P.pitman_morgan([1, 2], [2, 1])


def test_spread_and_relative_values():
    v = P.relative([100.0, 102.0, 98.0])
    assert v.tolist() == pytest.approx([0.0, 2.0, -2.0])
    sd, lo, hi = P.spread(v)
    assert sd == pytest.approx(2.0) and lo == pytest.approx(-2.0) and hi == pytest.approx(2.0)


def read_outputs(tree):
    out = os.path.join(tree, 'out')
    macros = open(os.path.join(out, P.OUT_FILES['macros'])).read()
    res = json.load(open(os.path.join(out, P.OUT_FILES['json'])))
    md = open(os.path.join(out, P.OUT_FILES['md'])).read()
    return macros, res, md


@needs_data
def test_known_spread_with_ten_seeds(tmp_path):
    res, names, _ = P.dry_run(str(tmp_path / 'tree'), 10)
    assert list(res['sets']) == ['First', 'Second', 'Ten'] and not res['left_out']
    for word, blk in res['sets'].items():
        want = P.dry_expected(blk['seeds'])
        assert len(blk['seeds']) == (10 if word == 'Ten' else 5)
        for r in R8:
            for arm in ('policy', 'specialist'):
                assert blk['rows'][r][arm]['spread'] == pytest.approx(want[arm], abs=1e-9)
            # the per-seed values are the synthetic deviations themselves
            idx = [list(SE.ALL_SEEDS).index(s) for s in blk['seeds']]
            assert blk['rows'][r]['specialist_values'] == pytest.approx(
                np.asarray(P.DRY_EFFECT['specialist'])[idx], abs=1e-9)
        assert blk['rms']['specialist'] == pytest.approx(want['specialist'], abs=1e-9)
    macros, rec, md = read_outputs(str(tmp_path / 'tree'))
    assert '\\newcommand{\\SsprTenSpecW}{%.2f}' % P.dry_expected(SE.ALL_SEEDS)['specialist'] in macros
    assert rec['sets']['Second']['seeds'] == list(SE.SECOND_SEEDS)
    assert 'seeds 306 to 310' in md and 'Left out' not in md


@needs_data
def test_later_seed_sets_left_out_while_second_five_incomplete(tmp_path):
    res, names, _ = P.dry_run(str(tmp_path / 'tree'), 5)
    assert list(res['sets']) == ['First']
    assert 'Second and Ten are left out' in res['left_out']
    macros, rec, md = read_outputs(str(tmp_path / 'tree'))
    assert 'SsprSecond' not in macros and 'SsprTen' not in macros
    assert '% Left out: the second five seeds are not complete' in macros
    assert '\\newcommand{\\SsprSetsN}{1}' in macros
    assert rec['left_out'] and 'Left out' in md
    want = P.dry_expected(SE.FIRST_SEEDS)
    assert res['sets']['First']['rows']['W']['policy']['spread'] == pytest.approx(want['policy'])


@needs_data
def test_macro_names_and_the_clash_check(tmp_path):
    res, names, _ = P.dry_run(str(tmp_path / 'tree'), 10)
    assert names and all(re.fullmatch(r'Sspr[A-Za-z]+', x) for x in names)
    assert len(names) == len(set(names))
    have = P.defined_macros()
    assert not set(names) & set(have)


def test_clash_and_name_checks(monkeypatch):
    monkeypatch.setattr(P, 'defined_macros', lambda: {'SsprFirstPolW': 'manuscript/other.tex'})
    with pytest.raises(P.Refused, match='already defined'):
        P.check_names(['\\newcommand{\\SsprFirstPolW}{1}'])
    with pytest.raises(P.Refused, match='letters only'):
        P.check_names(['\\newcommand{\\SsprFirst2}{1}'])


@needs_data
def test_dry_mode_from_the_command_line(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    r = subprocess.run([PY, '-B', 'scripts/p112_seed_spread.py', '--dry', str(tmp_path / 't'),
                        '--dry-seeds', '5'], cwd=ROOT, capture_output=True, text=True,
                       timeout=300, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert 'p112 dry run on synthetic arrays passed: seed sets First,' in r.stdout
    assert 'left out' in r.stdout


@needs_data
def test_check_exit_codes(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PACT_ENV='exact', PACT_SEEDS='301-305', PACT_OUT=str(tmp_path / 'out'),
               PACT_RESULTS=str(tmp_path / 'empty'))
    r = subprocess.run([PY, '-B', 'scripts/p112_seed_spread.py', '--check'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 1 and 'do not count' in r.stdout, r.stdout + r.stderr
    env['PACT_SEEDS'] = '301-303'
    r = subprocess.run([PY, '-B', 'scripts/p112_seed_spread.py', '--check'], cwd=ROOT,
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode != 0 and 'PACT_SEEDS=301-305' in (r.stdout + r.stderr)
