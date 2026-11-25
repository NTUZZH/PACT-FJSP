"""scripts/p126_best_of_seven.py: the policy against the per-instance best of seven rules.

  * on synthetic arrays of the five sizes: the pooled differences against the best of seven and
    against the best of four and the shortest-rule shares equal the built ones; the summary of
    p72 for 10x25 is reproduced; every macro name is Bsv plus letters;
  * a summary record or a Markdown row of p72 that the arrays do not reproduce refuses and
    leaves no output; a missing rule array is listed by --check and refuses the run;
  * ties count for each rule; the prefix Bsv is refused once another .tex file defines a
    macro with it.

Run from the repository root on one core:  python -B -m pytest -q tests/test_p126_best_of_seven.py
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

import p126_best_of_seven as B7  # noqa: E402

BOOT = 300
PY = sys.executable


def test_dry_numbers_equal_the_built_ones(tmp_path):
    out, names, outs = B7.dry_run(tmp_path / 'w', boot=BOOT)
    assert B7.dry_verify(out) == []
    rp = out['reproduction']['summary']
    assert rp['identical'] and rp['compared'] == 55
    assert names and all(n.startswith('Bsv') and n.isalpha() for n in names)
    for p in outs.values():
        assert os.path.exists(p)
    text = open(outs['macros']).read()
    for name in ('BsvTenSevenPool', 'BsvEightyFourPoolDir', 'BsvTenShortestLlf',
                 'BsvFiftyHierSevenPool'):
        assert f'\\newcommand{{\\{name}}}' in text
    x = out['sizes']['10x25']
    assert x['seeds'] == [301, 302, 303, 304, 305] and len(x['files']) == 5


def _tamper_record(tree):
    p = os.path.join(tree, 'out', 'results', 'aware_rules.json')
    rec = json.load(open(p))
    rec['sizes']['10x25']['regimes']['LS']['gap_PACT_LLF'][1] += 1e-9
    json.dump(rec, open(p, 'w'))


def _tamper_markdown(tree):
    p = os.path.join(tree, 'out', 'results', 'aware_rules.md')
    lines = open(p).read().splitlines()
    cells = lines[-1].split(' | ')
    cells[-1] = '+9.99 [+9.98, +9.99] |'
    lines[-1] = ' | '.join(cells)
    open(p, 'w').write('\n'.join(lines) + '\n')


@pytest.mark.parametrize('tamper', [_tamper_record, _tamper_markdown])
def test_summary_that_does_not_reproduce_refuses(tmp_path, tamper):
    tree = tmp_path / 'w'
    with pytest.raises(B7.Refused) as e:
        B7.dry_run(tree, boot=BOOT, before=tamper)
    assert 'does not reproduce' in str(e.value)
    for name in B7.OUT_FILES.values():
        assert not os.path.exists(tree / 'out' / name)


def test_missing_rule_array(tmp_path, capsys):
    tree = str(tmp_path / 'w')

    def drop(t):
        os.remove(os.path.join(t, 'res', '30x25_xmask', 'Result_SSPT_LW.npy'))
        assert B7.check() == 1
    with pytest.raises(B7.Refused) as e:
        B7.dry_run(tree, boot=BOOT, before=drop)
    assert 'inputs missing' in str(e.value)
    assert 'Result_SSPT_LW.npy' in capsys.readouterr().out


def test_ties_count_for_each_rule():
    rules = {'A': np.array([1.0, 2.0, 3.0]), 'B': np.array([1.0, 1.0, 4.0]),
             'C': np.array([2.0, 1.0, 3.0])}
    best = np.min(list(rules.values()), axis=0)
    sh = B7.shortest_shares(rules, best)
    assert sh['A'] == pytest.approx(200 / 3) and sh['B'] == pytest.approx(200 / 3)
    assert sh['C'] == pytest.approx(200 / 3)


def test_prefix_is_checked(tmp_path):
    own = tmp_path / 'manuscript' / 'macros_bestseven.tex'
    own.parent.mkdir()
    lines = ['\\newcommand{\\BsvBoot}{20000}  % x']
    assert B7.check_names(lines, str(own)) == ['BsvBoot']
    (own.parent / 'other.tex').write_text('\\newcommand{\\BsvFoo}{1}\n')
    with pytest.raises(B7.Refused):
        B7.check_names(lines, str(own))


def test_command_line_dry(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    p = subprocess.run([PY, '-B', os.path.join(ROOT, 'scripts', 'p126_best_of_seven.py'),
                        '--dry', str(tmp_path / 'w'), '--boot', str(BOOT)],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
    assert 'self-test on synthetic arrays passed' in p.stdout
