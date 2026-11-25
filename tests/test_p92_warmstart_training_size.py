"""Warm-start cache of the training size (scripts/p92_warmstart_xmask_training_size.py):
the metadata against the production caches, the guards, and the written cache
against its checksum list and against a fresh rollout of two cells (eight
dispatching-rule rollouts of a 10-module instance, a few seconds)."""
import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts'))
import p92_warmstart_xmask_training_size as W  # noqa: E402

CACHE = os.path.join(W.ROOT, W.CACHE)
need_cache = pytest.mark.skipif(not os.path.isdir(CACHE), reason='no training-size cache')


def test_metadata_has_the_keys_of_the_production_caches():
    new, prod = json.loads(W.meta_text()), json.loads(W.p86.masked_meta_text(50))
    assert set(new) - set(prod) == {'instances'} and set(prod) <= set(new)
    for k in ('rule_set', 'first_mask', 'mask_ties', 'rules', 'tie_rule', 'files', 'regimes'):
        assert new[k] == prod[k], k
    assert new['set'] == 'data/FAMILY/10x25' and new['n_instances'] == 30
    assert new['base_dir'] == 'data/PPVC/10x25+ppvc-mixed'
    assert 'SPT included' in new['procedure'] and 'p92' in new['generator']


def test_a_written_cache_holds_instances_0_to_29(tmp_path):
    with pytest.raises(SystemExit):
        W.parse_args(['--instances', '0-5'])
    assert W.parse_args(['--instances', '0-5', '--check-only']).idx == list(range(6))
    assert W.parse_args([]).idx == list(range(30))
    with pytest.raises(SystemExit):
        W.parse_args(['--check-only', '--stage', os.path.join(W.ROOT, 'or_solution', 'x')])


def test_a_stage_of_other_code_is_refused(tmp_path):
    st = tmp_path / 'stage'
    W.open_stage(str(st))
    W.open_stage(str(st))                               # same code: accepted
    mark = st / 'STAGE.json'
    code = json.loads(mark.read_text())
    code['common_utils.py'] = '0' * 64
    mark.write_text(json.dumps(code))
    with pytest.raises(SystemExit):
        W.open_stage(str(st))
    other = tmp_path / 'other'
    other.mkdir()
    (other / 'x.npz').write_bytes(b'')
    with pytest.raises(SystemExit):                     # not empty and no STAGE.json
        W.open_stage(str(other))


@need_cache
@pytest.mark.skipif(not os.path.isfile(os.path.join(W.ROOT, W.SUMS)),
                    reason=f'needs the checksum list {W.SUMS} of the cache, which the release '
                           f'does not hold')
def test_cache_equals_its_checksum_list():
    lines = open(os.path.join(W.ROOT, W.SUMS)).read().split('\n')
    lines = [x for x in lines if x]
    assert len(lines) == 242
    files = sorted(os.path.relpath(os.path.join(d, f), W.ROOT)
                   for d, _, fs in os.walk(CACHE) for f in fs)
    assert sorted(x.split('  ', 1)[1] for x in lines) == files
    for x in lines:
        h, p = x.split('  ', 1)
        assert hashlib.sha256(open(os.path.join(W.ROOT, p), 'rb').read()).hexdigest() == h, p


@need_cache
@pytest.mark.parametrize('combo,idx', [('N', 0), ('LSW', 29)])
def test_cell_rebuilds_identically(combo, idx):
    jl, pt, desc = W.load_cell(combo, idx)
    z = np.load(os.path.join(CACHE, combo, f'instance_{idx:03d}.npz'), allow_pickle=True)
    got = {r: W.p86.masked_rollout(r, jl, pt, desc) for r in W.RULES}
    assert [got[r][2] for r in W.RULES] == list(z['rule_ms'])
    best = str(z['rule'])
    assert np.array_equal(got[best][0], z['amch']) and np.array_equal(got[best][1], z['ct'])
    stored = np.load(os.path.join(W.ROOT, W.STORED, f'Result_{best}_{combo}.npy'))
    assert stored[idx, 0] == float(z['ms'])
