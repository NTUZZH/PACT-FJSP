"""scripts/p108_public_active_arm.py with five seeds: the policy runs pubx-act-s304 and -s305
of the public family under the active mask join the three runs of the active-rule arm.

--seeds auto reads seeds 301 to 305 once a file of seed 304 or 305 exists and seeds 301 to
303 before; --seeds 301-303 reproduces the three-seed outputs byte for byte whatever else
exists; a partial five-seed arm refuses. Synthetic trees of test_pubact_arm.

Run from the repository root:  python -m pytest tests/test_pubact_five_seeds.py -q
CPU only; every file is written under a temporary folder.
"""
import json
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p108_public_active_arm as A  # noqa: E402
import p89_public_family as P  # noqa: E402
import stats_family as SF  # noqa: E402
import test_pubact_arm as TA  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances per set)
from test_public_references import bases, write_refs  # noqa: E402

R8 = list(REG.COMBOS)
FIVE = (301, 302, 303, 304, 305)
LATER = (304, 305)
E5 = {301: -0.4, 302: 0.0, 303: 0.4, 304: -0.2, 305: 0.2}     # mean zero over the five seeds
B = T.B
ST = A.HEADLINE


def write_later(root, seeds=LATER, only_checkpoint=()):
    """pubx-v1 checkpoints and configurations, and the pubx-act runs of seeds 304 and 305
    (names and paths of arm_registry.pub_rule_model), written as test_pubact_arm writes
    seeds 301 to 303; a seed in only_checkpoint gets its checkpoint and nothing else."""
    os.makedirs(os.path.join(root, 'trained_network', 'FAMILY'), exist_ok=True)
    for s in seeds:
        ck, _, _, cp = A.run_paths('policy', s)
        if not os.path.exists(os.path.join(root, cp)):
            open(os.path.join(root, ck), 'w').close()
            os.utime(os.path.join(root, ck), (TA.OLD, TA.OLD))
            json.dump(dict(device='cpu', family_regimes=TA.SIX, max_updates=2000, seed_train=s,
                           model_suffix=f'pubx-v1-s{s}', first_mask='exact', mask_ties='exact',
                           mask_mode='nondelay', lr=0.0003, family_base='sd2', n_m=5,
                           use_type_embedding=False, reward_mode='dense'),
                      open(os.path.join(root, cp), 'w'))
    base = bases(root)
    out = os.path.join(root, REG.PUB_RESULTS, ST + P.XS)
    for s in seeds:
        name = A.run_name('active', s)
        assert name == f'pubx-act-s{s}'
        ck, lp, vp, cp = A.run_paths('active', s)
        assert (ck, lp, vp, cp) == REG.pub_rule_paths('released', None, s)
        open(os.path.join(root, ck), 'w').close()
        os.utime(os.path.join(root, ck), (TA.OLD, TA.OLD))
        if s in only_checkpoint:
            continue
        rows = []
        for r in R8:
            v = TA.pol_of(root, s, r) * (1 + (TA.C[r] + E5[s]) / 100)
            np.save(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'),
                    np.stack([v, np.full(len(v), 0.1)], axis=1))
            rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
        T.write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'), name, rows,
                        len(base), base * 0.7)
        lines = ['[train_family] MASK: first_mask=exact mask_ties=exact', A.BANNER,
                 '[vali] MWKR norms: ' + '  '.join(f'{r}=500.0' for r in P.TRAIN),
                 f'[model] params total={P.N_PARAMS}  film=0 (0.00%)']
        best = 9.0
        for k, sc in enumerate(TA.ACT_SCORES):
            best = min(best, sc)
            lines.append(f'[vali @{25 * (k + 1)}] rel={sc:.4f} (best {best:.4f})  '
                         + '  '.join(f'{r}={500 * sc:.1f}' for r in P.TRAIN))
        lines.append(f'[train_family] done in 0.90 h; best rel={min(TA.ACT_SCORES):.4f}')
        open(os.path.join(root, lp), 'w').write('\n'.join(lines) + '\n')
        open(os.path.join(root, vp), 'w').write(repr(TA.ACT_SCORES))
        ref = json.load(open(os.path.join(root, A.run_paths('policy', s)[3])))
        json.dump(dict(ref, model_suffix=name, mask_mode='active', reward_mode='dense',
                       pub_type_source='none'), open(os.path.join(root, cp), 'w'))


def three_tree(tmp_path):
    """The three-seed tree of test_pubact_arm, with E5 equal to its E on seeds 301 to 303."""
    assert all(E5[s] == TA.E[s] for s in TA.SEEDS)
    return TA.complete_tree(tmp_path)


def run(root, out, *extra):
    return A.main(['infer', '--root', str(root), '--out', str(out), '--boot', str(B)] + list(extra))


def read_outputs(out):
    return [open(os.path.join(str(out), p), 'rb').read() for p in A.OUT_FILES]


def test_five_seeds_hand_computed(tmp_path):
    root = three_tree(tmp_path)
    write_later(root)
    plan = A.main(['plan', '--root', root])
    assert plan['seeds'] == list(FIVE) and plan['arm']['state'] == 'complete'
    assert plan['arm']['expected'] == 5 * 13 and plan['ready']
    res = run(root, tmp_path / 'out')
    assert res['seeds'] == list(FIVE) and A.SEEDS == tuple(REG.PUB_ACT_SEEDS)   # restored
    n = T.N_INST
    for r in R8:
        d = np.stack([np.full(n, TA.C[r] + E5[s]) for s in FIVE])
        hb = SF.hier_boot(d, B=B, seed=11)
        c = res['components'][r]
        assert c['mean'] == pytest.approx(TA.C[r], abs=1e-9)
        assert (c['lo'], c['hi']) == pytest.approx((hb['lo'], hb['hi']), abs=1e-9)
        assert c['seed_means'] == pytest.approx([TA.C[r] + E5[s] for s in FIVE], abs=1e-9)
    regs = A.FAMILIES['All']
    d3 = np.stack([np.stack([np.full(n, TA.C[r] + E5[s]) for s in FIVE]) for r in regs], axis=2)
    hb = SF.hier_boot(d3, B=B, seed=11)
    x = res['pooled']['All']
    assert x['mean'] == pytest.approx(-0.6, abs=1e-9)
    assert (x['lo'], x['hi']) == pytest.approx((hb['lo'], hb['hi']), abs=1e-9)
    assert res['per_seed']['All'] == pytest.approx({f's{s}': -0.6 + E5[s] for s in FIVE}, abs=1e-9)
    assert res['files_read'] == 2 * 5 * 8
    assert res['config_differs_in'] == {f's{s}': ['mask_mode', 'model_suffix'] for s in FIVE}
    js, md, mac = (t.decode() for t in read_outputs(tmp_path / 'out'))
    for macro, val in (('PubActSeedN', '5'), ('PubActSeedFour', '-0.80'), ('PubActSeedFive', '-0.40'),
                       ('PubActDiffPoolAll', '-0.60'), ('PubActDiffN', '-2.00')):
        assert f'\\newcommand{{\\{macro}}}{{{val}}}' in mac, macro
    assert 'training seeds 301-305' in mac and '(4 df)' in md and 'no makespan of the ten runs' in md
    for text in (js, md, mac):
        assert str(tmp_path) not in text and '/home/' not in text and '/tmp/' not in text


def test_three_seed_reading_reproduced_byte_for_byte(tmp_path):
    root = three_tree(tmp_path)
    res3 = run(root, tmp_path / 'auto3')
    assert res3['seeds'] == list(TA.SEEDS)
    three = read_outputs(tmp_path / 'auto3')
    write_later(root)
    run(root, tmp_path / 'explicit3', '--seeds', '301-303')
    assert read_outputs(tmp_path / 'explicit3') == three
    res5 = run(root, tmp_path / 'auto5')
    assert res5['seeds'] == list(FIVE) and read_outputs(tmp_path / 'auto5') != three
    run(root, tmp_path / 'explicit5', '--seeds', '301-305')
    assert read_outputs(tmp_path / 'explicit5') == read_outputs(tmp_path / 'auto5')


@pytest.mark.parametrize('seeds,refuses', [('auto', True), ('301-305', True), ('301-303', False)])
def test_partial_five_seed_arm(tmp_path, seeds, refuses):
    """Seed 305 in training (checkpoint only): the five-seed reading refuses, the three-seed
    reading does not see it."""
    root = three_tree(tmp_path)
    write_later(root, only_checkpoint=(305,))
    out = tmp_path / 'out'
    if refuses:
        with pytest.raises(SystemExit) as e:
            run(root, out, '--seeds', seeds)
        msg = str(e.value)
        assert 'REFUSING' in msg and 'incomplete' in msg and 'seeds 301-305' in msg, msg
        assert not any(os.path.exists(os.path.join(str(out), p)) for p in A.OUT_FILES)
    else:
        assert run(root, out, '--seeds', seeds)['seeds'] == list(TA.SEEDS)


def test_unknown_seed_choice_refused(tmp_path):
    root = three_tree(tmp_path)
    with pytest.raises(SystemExit) as e:
        run(root, tmp_path / 'out', '--seeds', '301-304')
    assert 'REFUSING' in str(e.value) and '301-304' in str(e.value)
    assert re.search(r'auto, 301-303 or 301-305', str(e.value))
