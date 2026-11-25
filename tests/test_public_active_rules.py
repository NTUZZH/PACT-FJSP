"""scripts/p109_public_active_rules.py: the four dispatching rules of the public benchmark
rolled out under a mask.

On small samples of the stored records: under the non-delay mask the rollouts reproduce the
stored rule arrays bit for bit; on SD2 10x5 the rollouts under both masks reproduce the cases
of scripts/p107_mask_rule_variants.py; every schedule is validated (an infeasible one refuses);
the arrays, the summary and the meta record are written in the layout p89 reads, under a root
of their own, without an absolute path; the refusals of the output root and of a partial job.

Run from the repository root:  python -m pytest tests/test_public_active_rules.py -q
CPU only, one thread; every file is written under a temporary folder.
"""
import argparse
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

import arm_registry as REG  # noqa: E402
import p109_public_active_rules as P109  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

R8 = list(REG.COMBOS)
HEAD = 'pub_SD2_test_10x5+mix'
pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family')) or
    not os.path.isdir(os.path.join(ROOT, P109.p107_root())),
    reason='needs the public-family records and the cases of p107')


def ns(**kw):
    base = dict(sets='', combos='', mask='active', out_root='', budget_s=0.0, n=0)
    return argparse.Namespace(**dict(base, **kw))


@pytest.fixture
def few_records(monkeypatch):
    """records() limited to the first k records of every set and combination."""
    real = P109.records

    def first(k):
        monkeypatch.setattr(P109, 'records', lambda st, combo, root=ROOT: real(st, combo, root)[:k])
    return first


@needs_public_instances()
def test_nondelay_mask_reproduces_the_stored_arrays():
    sets = 'pub_Brandimarte,pub_Hurink_vdata,pub_SD2_test_20x10+mix,pub_SD2_test_10x5+mix_lag050'
    lines = []
    tot, bad = P109.check_nondelay(ns(sets=sets, combos='N,LSW', n=2), out=lines.append)
    assert not bad, bad
    assert tot == 4 * 2 * 4 * 2 and 'all equal' in lines[0]


@needs_public_instances()
def test_both_masks_reproduce_the_cases_of_p107():
    lines = []
    tot, bad = P109.check_p107(ns(combos='S,SW', n=3), out=lines.append)
    assert not bad, bad
    assert tot == 2 * 2 * 4 * 3 and 'all equal' in lines[0]


@needs_public_instances()
def test_the_active_mask_changes_the_rollouts():
    """A control of the two checks above: on the same records the two masks differ."""
    a, names = P109.rollout_combo(HEAD, 'N', 'active', n=4)
    d, names_d = P109.rollout_combo(HEAD, 'N', 'nondelay', n=4)
    assert names == names_d
    assert any(not np.array_equal(a[q][:, 0], d[q][:, 0]) for q in P109.RULES)


@needs_public_instances()
def test_an_infeasible_schedule_refuses(monkeypatch):
    P109._setup()
    monkeypatch.setitem(P109._CONF, 'validate', lambda *a: {'precedence': 1})
    rec = P109.records(HEAD, 'N')[0][1]
    with pytest.raises(SystemExit) as e:
        P109.rollout(rec, 'SPT', 'active')
    assert 'REFUSING' in str(e.value) and 'infeasible' in str(e.value)


@needs_public_instances()
def test_run_writes_the_layout_and_reproduces_the_stored_active_arrays(tmp_path, few_records):
    few_records(2)
    out = tmp_path / 'rules'
    done = P109.main(['run', '--sets', HEAD, '--combos', 'N,LS', '--out-root', str(out)])
    assert done == [(HEAD, 'N'), (HEAD, 'LS')]
    d = out / (HEAD + P109.XS)
    assert sorted(os.listdir(d)) == sorted(P109.array_name(q, r) for q in P109.RULES for r in ('N', 'LS'))
    stored = os.path.join(ROOT, P109.ACTIVE_ROOT, HEAD + P109.XS)
    for q in P109.RULES:
        for r in ('N', 'LS'):
            v = np.load(d / P109.array_name(q, r))
            assert v.shape == (2, 2)
            p = os.path.join(stored, P109.array_name(q, r))
            if os.path.exists(p):                  # the stored rollout of the set, row for row
                assert np.array_equal(v[:, 0], np.load(p)[:2, 0])
    # a finished job is skipped; the summary and the meta record wait for all 32 arrays
    assert P109.main(['run', '--sets', HEAD, '--combos', 'N', '--out-root', str(out)]) == []
    assert not (d / P109.SUMMARY).exists() and not (d / P109.META).exists()
    P109.main(['run', '--sets', HEAD, '--out-root', str(out)])
    summ = (d / P109.SUMMARY).read_text()
    meta = json.loads((d / P109.META).read_text())
    assert 'pairs: mask_mode=active' in summ and summ.count('| 2/2 |') == 32
    assert meta['mask_mode'] == 'active' and meta['n'] == 2 and meta['feasible'] == '64 of 64 schedules'
    for text in (summ, json.dumps(meta)):
        assert str(tmp_path) not in text and ROOT not in text and '/home/' not in text and '/tmp/' not in text


def test_summary_layout_is_the_stored_one():
    """Under the non-delay mask and with the stored arrays, the summary is the stored rule
    summary of eval_family.py plus the line of the mask."""
    st = HEAD
    d = os.path.join(ROOT, REG.PUB_RESULTS, st + P109.XS)
    names = [nm for nm, _ in P109.records(st, 'N')]
    arrays = {(q, r): np.load(os.path.join(d, P109.array_name(q, r))) for q in P109.RULES for r in R8}
    lines = P109.summary_lines(st, 'nondelay', arrays, names)
    i = lines.index('pairs: mask_mode=nondelay')
    assert lines[i + 1] == ''
    want = open(os.path.join(d, P109.SUMMARY)).read().rstrip('\n').split('\n')
    assert lines[:i] + lines[i + 2:] == want


def test_output_root_refusals(tmp_path, few_records):
    few_records(1)
    for root in (os.path.join(ROOT, REG.PUB_RESULTS), os.path.join(ROOT, 'test_results')):
        with pytest.raises(SystemExit) as e:
            P109.check_out_root(root, 'active')
        assert 'holds the stored rule arrays' in str(e.value)
    with pytest.raises(SystemExit) as e:
        P109.check_out_root(os.path.join(ROOT, 'test_results', 'PUBFAMILY', 'rules_nd'), 'nondelay')
    assert 'outside test_results' in str(e.value)
    P109.check_out_root(str(tmp_path / 'nd'), 'nondelay')          # allowed outside test_results
    # a partial job refuses
    out = tmp_path / 'rules'
    d = out / ('pub_Brandimarte' + P109.XS)
    d.mkdir(parents=True)
    np.save(d / P109.array_name('SPT', 'W'), np.ones((1, 2)))
    with pytest.raises(SystemExit) as e:
        P109.main(['run', '--sets', 'pub_Brandimarte', '--combos', 'W', '--out-root', str(out)])
    assert 'some of the four arrays exist' in str(e.value)
    with pytest.raises(SystemExit) as e:
        P109.main(['run', '--sets', 'pub_Nowhere', '--out-root', str(out)])
    assert 'unknown sets' in str(e.value)


def test_plan_lists_every_set(tmp_path, capsys):
    tot = P109.main(['plan', '--out-root', str(tmp_path / 'rules')])
    text = capsys.readouterr().out
    assert tot > 0 and text.count('done 0 of 8 combinations') == len(P109.SETS) == 7
    assert str(tmp_path) in text.splitlines()[0]          # the root given, shown as given
