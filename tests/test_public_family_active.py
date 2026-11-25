"""scripts/p89_public_family.py --family active on synthetic trees: the public family trained
under the active mask (arm_registry.pub_rule_model) and its rules under that mask
(scripts/p109_public_active_rules.py layout), read by the code of the default family.

Hand-computed values for one comparison with a specialist, one pooled row and the
comparison across masks; refusals of a partial or wrong family; the partial report; nothing
written outside --out; the outputs of the default family unchanged by the presence of the
family under the active mask.

Run from the repository root:  python -m pytest tests/test_public_family_active.py -q
CPU only; every file is written under a temporary folder.
"""
import json
import os
import re
import shutil
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p109_public_active_rules as P109  # noqa: E402
import p89_public_family as P  # noqa: E402
import stats_family as SF  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances)
from test_public_references import write_refs  # noqa: E402

R8 = list(REG.COMBOS)
SIX = ','.join(REG.TRAIN_COMBOS)
B = T.B
ST = 'pub_SD2_test_10x5+mix'
ALL_SETS = list(P.SETS) + list(REG.PUB_SENS_SETS)
# makespan under the active mask = non-delay makespan of the same run x (1 + K / 100); the
# control also moves by E[seed] / 100 (mean zero over the seeds)
K = {'released': {'N': -1.0, 'L': -2.0, 'S': -3.0, 'W': 0.5, 'LS': -2.5, 'LW': 0.0, 'SW': -1.5,
                  'LSW': -2.0},
     'control': {r: -1.0 for r in R8},
     'specialist': {'N': -0.5, 'L': -1.0, 'S': -2.0, 'W': 1.0, 'LS': -2.0, 'LW': 0.5, 'SW': -1.0,
                    'LSW': -2.5}}
E = {301: -0.4, 302: -0.2, 303: 0.0, 304: 0.2, 305: 0.4}
# each rule under the active mask = the same rule under the non-delay mask x G
G = {'FIFO': 1.03, 'MOR': 0.98, 'SPT': 1.0, 'MWKR': 1.0}
BANNER = '[train_family] OPTIONS: mask_mode=active pair_extra=none'
FAM = P.Fam('active')


def factor(arm, r, s):
    return 1 + (K[arm][r] + (E[s] if arm == 'control' else 0.0)) / 100


def write_cfg(root, path, **kw):
    base = dict(device='cpu', first_mask='exact', mask_ties='exact', lr=0.0003, family_base='sd2',
                n_m=5, use_type_embedding=False)
    with open(os.path.join(root, path), 'w') as f:
        json.dump(dict(base, **kw), f)


def build_active(root, skip=(), rule_summary_line=True):
    """The family under the active mask on a tree of build_tree: arrays and summaries of every
    principal run in every test and sensitivity folder, training records with the trainer's
    line of the active mask, configurations of each run and of its non-delay counterpart,
    and the four rules under the active mask in the rules_act folder. skip: runs left out."""
    nd = P.Fam('nondelay')
    for st in ALL_SETS:
        out = os.path.join(root, REG.PUB_RESULTS, st + P.XS)
        base = np.load(os.path.join(
            out, f'Result_greedy+{REG.PUB_PREFIX}{REG.pub_model("specialist", "N", 301)}_N.npy'))[:, 0]
        n, bk = len(base), base * 0.7
        for arm, combo, regs in P.Inputs(root).runs():
            for s in P.SEEDS:
                name = FAM.model(arm, combo, s)
                if name in skip:
                    continue
                rows = []
                for r in regs:
                    v = np.load(os.path.join(
                        out, f'Result_greedy+{REG.PUB_PREFIX}{nd.model(arm, combo, s)}_{r}.npy'))[:, 0]
                    v = v * factor(arm, r, s)
                    np.save(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'),
                            np.stack([v, np.full(n, 0.1)], axis=1))
                    rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
                T.write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'),
                                name, rows, n, bk)
        rd = os.path.join(root, FAM.rule_root, st + P.XS)
        os.makedirs(rd, exist_ok=True)
        rows = []
        for r in R8:
            for q in P.RULES:
                v = np.load(os.path.join(out, f'Result_{q}_{r}.npy'))[:, 0] * G[q]
                np.save(os.path.join(rd, f'Result_{q}_{r}.npy'), np.stack([v, np.full(n, 0.1)], axis=1))
                rows.append((r, q, v))
        T.write_summary(os.path.join(rd, f'summary_rules{P.XS}.md'), 'rules', rows, n, bk)
        if rule_summary_line:
            p = os.path.join(rd, f'summary_rules{P.XS}.md')
            txt = open(p).read().replace('\n\n|', '\n\npairs: mask_mode=active\n\n|', 1)
            open(p, 'w').write(txt)
    os.makedirs(os.path.join(root, 'trained_network', 'FAMILY'), exist_ok=True)
    for arm, combo, _ in P.Inputs(root).runs():
        regs = {'released': SIX, 'control': 'N,L,S,W'}.get(arm, combo)
        upd = 1000 if arm == 'specialist' else 2000
        for s in P.SEEDS:
            name = FAM.model(arm, combo, s)
            ck, lp, vp, cp = FAM.paths(arm, combo, s)
            _, nlp, nvp, ncp = nd.paths(arm, combo, s)
            write_cfg(root, ncp, family_regimes=regs, max_updates=upd, seed_train=s,
                      model_suffix=nd.model(arm, combo, s), mask_mode='nondelay')
            if name in skip:
                continue
            txt = open(os.path.join(root, nlp)).read()
            open(os.path.join(root, lp), 'w').write(BANNER + '\n' + txt)
            shutil.copy(os.path.join(root, nvp), os.path.join(root, vp))
            open(os.path.join(root, ck), 'w').close()
            write_cfg(root, cp, family_regimes=regs, max_updates=upd, seed_train=s,
                      model_suffix=name, mask_mode='active', reward_mode='dense')
    return root


def tree(tmp_path, **kw):
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    return build_active(root, **kw)


def run(root, out, *extra):
    return P.main(['--family', 'active', '--root', str(root), '--out', str(out), '--boot', str(B)]
                  + list(extra))


def outputs(out):
    return [out / p for p in FAM.out_files]


def snapshot(root):
    return {os.path.relpath(os.path.join(d, f), root): (os.stat(os.path.join(d, f)).st_size,
                                                        os.stat(os.path.join(d, f)).st_mtime_ns)
            for d, _, fs in os.walk(root) for f in fs}


# ------------------------------------------------------------------ names and layout
def test_family_names_and_files():
    assert FAM.out_files == ('results/public_family_act.json', 'results/public_family_act.md',
                             'manuscript/macros_pubfam_act.tex')
    assert FAM.prefix == 'PubFamAct' and FAM.rule_root == 'test_results/PUBFAMILY/rules_act'
    assert FAM.rule_root == P109.ACTIVE_ROOT
    assert FAM.model('control', None, 302) == 'pubx-ctlact-s302'
    assert FAM.model('specialist', 'LS', 305) == 'pubx-soloact-LS-s305'
    assert FAM.model('released', None, 301) == REG.pub_act_model(301)
    assert FAM.eq_stem == 'pubx-eqact' and FAM.eq_model(303) == 'pubx-eqact-s303'
    nd = P.Fam('nondelay')
    assert nd.out_files == P.OUT_FILES and nd.prefix == 'Pub' and nd.rule_root == REG.PUB_RESULTS
    assert nd.paths('specialist', 'W', 304)[:3] == REG.pub_paths('specialist', 'W', 304) + (
        REG.pub_vali_log('specialist', 'W', 304),)
    # the rules folder is matched by no glob of the default family (check_duplicates: <set>*)
    for st in ALL_SETS:
        assert not os.path.basename(FAM.rule_root).startswith(st)


# ------------------------------------------------------------------ complete family
def test_hand_computed_values_and_outputs(tmp_path):
    root = tree(tmp_path)
    before = snapshot(root)
    out = tmp_path / 'out'
    res = run(root, out)
    assert snapshot(root) == before                       # nothing written outside --out
    assert sorted(snapshot(str(out))) == sorted(FAM.out_files)
    n = T.N_INST
    # one comparison with a specialist: under the active mask, the policy on L is its
    # non-delay value x 0.98 and the specialist's x 0.99; in the tree the policy is the
    # specialist x 1.004, so d = 100 (1.004 x 0.98 / 0.99 - 1) on every seed and instance
    want = 100 * (1.004 * 0.98 / 0.99 - 1)
    c = res['sets'][ST]['parity']['components']['L']
    assert c['mean'] == pytest.approx(want, abs=1e-9) == pytest.approx(-0.614141414, abs=1e-6)
    assert (c['lo'], c['hi']) == pytest.approx((want, want), abs=1e-9)
    assert c['seed_means'] == pytest.approx([want] * 5, abs=1e-9)
    # one pooled row: the mean of the eight constant differences
    per = [100 * (1.004 * factor('released', r, 301) / factor('specialist', r, 301) - 1) for r in R8]
    x = res['sets'][ST]['parity']['pooled']['All']
    assert x['mean'] == pytest.approx(np.mean(per), abs=1e-9)
    assert (x['lo'], x['hi']) == pytest.approx((np.mean(per), np.mean(per)), abs=1e-9)
    # across masks: the policy on L is exactly 2% shorter; the control moves by -1 + E[s]
    M = res['mask_comparison']['sets'][ST]
    p = M['arms']['policy']['components']['L']
    assert (p['mean'], p['lo'], p['hi']) == pytest.approx((-2.0, -2.0, -2.0), abs=1e-9)
    assert p['direction'] == 'ahead' and p['shorter'] == 100.0
    ctl = M['arms']['control']['components']['W']
    d = np.stack([np.full(n, -1.0 + E[s]) for s in P.SEEDS])
    hb = SF.hier_boot(d, B=B, seed=P.HIER_SEED)
    assert ctl['mean'] == pytest.approx(-1.0, abs=1e-9)
    assert ctl['seed_means'] == pytest.approx([-1.0 + E[s] for s in P.SEEDS], abs=1e-9)
    assert (ctl['lo'], ctl['hi']) == pytest.approx((hb['lo'], hb['hi']), abs=1e-9)
    assert M['arms']['specialist']['components']['LSW']['mean'] == pytest.approx(-2.5, abs=1e-9)
    assert M['arms']['policy']['pooled']['Setup']['mean'] == pytest.approx(
        np.mean([K['released'][r] for r in ('S', 'LS', 'SW', 'LSW')]), abs=1e-9)
    assert M['arms']['specialist']['pooled']['NoSetup']['mean'] == pytest.approx(
        np.mean([K['specialist'][r] for r in ('N', 'L', 'W', 'LW')]), abs=1e-9)
    # rules: FIFO is the best rule under the non-delay mask (factor 1.00 of the base cost),
    # MOR under the active mask (1.01 x 0.98 = 0.9898), so the best rule moves by -1.02%
    br = M['rules']['best']['components']['S']
    assert br['mean'] == pytest.approx(100 * (1.01 * 0.98 - 1), abs=1e-9)
    assert M['rules']['FIFO']['pooled']['All']['mean'] == pytest.approx(3.0, abs=1e-9)
    assert M['best_rule_by_mask']['active|S']['lowest_mean'] == 'MOR'
    assert M['best_rule_by_mask']['nondelay|S']['lowest_mean'] == 'FIFO'
    # the rules of section 3 are those of the active mask
    best = res['sets'][ST]['rules']['S']['best_rule_share']
    assert best == pytest.approx({'FIFO': 0.0, 'MOR': 100.0, 'SPT': 0.0, 'MWKR': 0.0})
    # configurations differ from their counterparts in the name and the mask only
    assert set(map(tuple, res['family']['config_differs_in'].values())) == {('mask_mode', 'model_suffix')}
    assert len(res['family']['config_differs_in']) == 10 * 5
    # outputs: every macro carries the prefix, no absolute path, the family in the texts
    js, md, mac = (q.read_text() for q in outputs(out))
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', mac)
    assert names and len(names) == len(set(names)) and all(x.startswith('PubFamAct') for x in names)
    assert len(names) == res['n_macros'] + res['n_macros_verdict'] + res['n_macros_dir'] + \
        res['n_macros_mask']
    for text in (js, md, mac):
        assert str(tmp_path) not in text and ROOT not in text and '/home/' not in text
    assert f'\\newcommand{{\\PubFamActSdtenParL}}{{{P.f2(want)}}}' in mac
    assert '\\newcommand{\\PubFamActMaskSdtenPolL}{-2.00}' in mac
    # the halved-lag set joins section 10 after the six test sets, with the same definitions
    assert list(res['mask_comparison']['sets']) == list(P.SETS) + list(REG.PUB_SENS_SETS)
    lag = res['mask_comparison']['sets']['pub_SD2_test_10x5+mix_lag050']
    assert lag['arms']['policy']['components']['L']['mean'] == pytest.approx(-2.0, abs=1e-9)
    assert lag['rules']['FIFO']['pooled']['All']['mean'] == pytest.approx(3.0, abs=1e-9)
    assert '\\newcommand{\\PubFamActMaskLagHalfPolL}{-2.00}' in mac
    assert '\\newcommand{\\PubFamActMaskLagHalfSpecLSW}{-2.50}' in mac
    assert 'PubFamActMaskLagHalfLowRuleActS' not in mac        # the headline set only
    assert '### pub_SD2_test_10x5+mix_lag050 (sensitivity of' in md
    assert '\\newcommand{\\PubFamActMaskSdtenLowRuleActS}{MOR}' in mac
    assert '\\newcommand{\\PubFamActMaskSdtenLowRuleNdS}{FIFO}' in mac
    assert 'active mask' in mac.splitlines()[1] and mac.startswith('% Generated by scripts/p89')
    assert md.startswith('# Public-instance family under the active mask:')
    assert '## 10. The active mask against the non-delay mask' in md
    assert json.loads(js)['generator'] == 'scripts/p89_public_family.py --family active'
    # every macro of the default family has its counterpart of the same name after the prefix
    nd_out = tmp_path / 'nd'
    P.main(['--root', str(root), '--out', str(nd_out), '--boot', str(B)])
    nd_names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', (nd_out / P.OUT_FILES[2]).read_text())
    assert {'PubFamAct' + x[3:] for x in nd_names} == {x for x in names if 'FamActMask' not in x}


def test_default_family_unchanged_by_the_active_family(tmp_path):
    """The default run gives the same bytes with and without the family under the active
    mask in the tree (its rules folder included)."""
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    a, b = tmp_path / 'a', tmp_path / 'b'
    P.main(['--root', root, '--out', str(a), '--boot', str(B)])
    build_active(root)
    P.main(['--root', root, '--out', str(b), '--boot', str(B)])
    for f in P.OUT_FILES:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


# ------------------------------------------------------------------ refusals and partial report
@pytest.mark.parametrize('damage', ['missing run', 'rules folder', 'no rule line', 'no banner',
                                    'config drift', 'nondelay config', 'counterpart under the mask'])
def test_wrong_or_partial_family_refuses(tmp_path, damage):
    out = tmp_path / 'out'
    for q in outputs(out):                      # stale outputs of an earlier run
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_text('stale')
    msg = 'REFUSING'
    if damage == 'missing run':
        root = tree(tmp_path, skip=('pubx-soloact-LS-s305',))
        msg = 'family is incomplete'
    elif damage == 'no rule line':
        root = tree(tmp_path, rule_summary_line=False)
        msg = 'pairs: mask_mode=active'
    else:
        root = tree(tmp_path)
        if damage == 'rules folder':
            os.remove(os.path.join(root, FAM.rule_root, 'pub_Brandimarte' + P.XS, 'Result_MOR_LW.npy'))
            msg = 'rule arrays of the active family are incomplete'
        elif damage == 'no banner':
            lp = os.path.join(root, FAM.paths('control', None, 303)[1])
            txt = open(lp).read()
            open(lp, 'w').write(txt.replace(BANNER + '\n', ''))
            msg = 'does not state the active mask'
        elif damage in ('config drift', 'nondelay config', 'counterpart under the mask'):
            cp = FAM.paths('specialist', 'SW', 302)[3] if damage != 'counterpart under the mask' \
                else P.Fam('nondelay').paths('specialist', 'SW', 302)[3]
            c = json.load(open(os.path.join(root, cp)))
            c.update({'config drift': {'lr': 0.001}, 'nondelay config': {'mask_mode': 'nondelay'},
                      'counterpart under the mask': {'mask_mode': 'active'}}[damage])
            json.dump(c, open(os.path.join(root, cp), 'w'))
            msg = 'is not the recipe' if damage == 'config drift' else 'mask_mode'
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'REFUSING' in str(e.value) and msg in str(e.value), str(e.value)
    assert not any(q.exists() for q in outputs(out))


def test_partial_report(tmp_path):
    root = tree(tmp_path, skip=('pubx-soloact-LS-s305', 'pubx-ctlact-s304'))
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:                  # refused without the flag
        run(root, out)
    assert 'family is incomplete' in str(e.value) and 'complete seeds: [301, 302, 303]' in str(e.value)
    with pytest.raises(SystemExit) as e:                  # never into the repository's outputs
        P.main(['--family', 'active', '--partial', '--root', root, '--out', ROOT, '--boot', str(B)])
    assert 'never written into the repository root' in str(e.value)
    res = run(root, out, '--partial')
    assert res['partial']['seeds'] == [301, 302, 303] and res['seeds'] == (301, 302, 303)
    # training records (4) and, in the seven folders, 8 arrays and a summary (control) or 1
    assert res['partial']['left_out'] == {'s304': 4 + 7 * 9, 's305': 4 + 7 * 2}
    js, md, mac = (q.read_text() for q in outputs(out))
    assert '"PARTIAL REPORT' in js and md.startswith('# PARTIAL REPORT') and mac.startswith('% PARTIAL REPORT')
    c = res['sets'][ST]['parity']['components']['L']
    assert c['mean'] == pytest.approx(100 * (1.004 * 0.98 / 0.99 - 1), abs=1e-9)
    assert len(c['seed_means']) == 3
    assert res['mask_comparison']['seeds'] == [301, 302, 303]
    assert P.SEEDS == tuple(REG.PUB_SEEDS)              # the module's seeds are restored
    # the same seeds give the same numbers as a complete run restricted to them
    with pytest.raises(SystemExit) as e:                  # fewer than two complete seeds
        shutil.rmtree(tmp_path / 'out')
        for s in (301, 302):
            for f in P.run_files(P.Inputs(root, FAM), 'control', None, s)[1:2]:
                os.remove(os.path.join(root, f))
        run(root, out, '--partial')
    assert 'needs two seeds' in str(e.value)


def test_macro_collision_refused(tmp_path):
    root = tree(tmp_path)
    os.makedirs(os.path.join(root, 'manuscript'), exist_ok=True)
    with open(os.path.join(root, 'manuscript', 'macros_other.tex'), 'w') as f:
        f.write('\\newcommand{\\PubFamActMaskBrandPolPoolAll}{1}\n')
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'collision with manuscript/macros_other.tex' in str(e.value)
    os.remove(os.path.join(root, 'manuscript', 'macros_other.tex'))
    # an earlier output of the family under another output root is not a collision
    with open(os.path.join(root, 'manuscript', 'macros_pubfam_act.tex'), 'w') as f:
        f.write('\\newcommand{\\PubFamActMaskBrandPolPoolAll}{1}\n')
    assert run(root, out)['n_macros'] > 0


def test_equal_experience_recipe_under_the_mask(tmp_path):
    """Section 8 under the active mask: pubx-eqact-s<seed>, its configuration against pubx-eq
    of the same seed, its arrays the pubx-eq arrays x 0.99."""
    import test_pubeq_arm as Q
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    Q.write_eq(root)
    build_active(root)
    for st in ALL_SETS:
        out = os.path.join(root, REG.PUB_RESULTS, st + P.XS)
        n = None
        for s in P.SEEDS:
            name, rows = FAM.eq_model(s), []
            for r in R8:
                v = np.load(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{REG.pub_eq_model(s)}_{r}.npy'))
                v = v[:, 0] * 0.99
                n = len(v)
                np.save(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'),
                        np.stack([v, np.full(n, 0.1)], axis=1))
                rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
            base = np.load(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}pubx-solo-N-s301_N.npy'))[:, 0]
            T.write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'), name, rows, n,
                            base * 0.7)
    for s in P.SEEDS:
        ck, lp, vp, cp = FAM.eq_paths(s)
        _, nlp, nvp, ncp = REG.pub_eq_paths(s)
        txt = open(os.path.join(root, nlp)).read()
        open(os.path.join(root, lp), 'w').write(BANNER + '\n' + txt)
        shutil.copy(os.path.join(root, nvp), os.path.join(root, vp))
        open(os.path.join(root, ck), 'w').close()
        c = json.load(open(os.path.join(root, ncp)))
        json.dump(dict(c, model_suffix=FAM.eq_model(s), mask_mode='active'), open(os.path.join(root, cp), 'w'))
    out = tmp_path / 'out'
    res = run(root, out)
    E8 = res['equal_experience']
    assert E8['arm'] == 'pubx-eqact' and E8['config_reference'] == 'pubx-eq of the same seed'
    assert E8['config_differs_in'] == {f's{s}': ['mask_mode', 'model_suffix'] for s in P.SEEDS}
    c = E8['sets'][ST]['parity']['components']['L']
    assert c['mean'] == pytest.approx(100 * (Q.EQ_FACTOR * 0.99 / factor('specialist', 'L', 301) - 1),
                                      abs=1e-9)
    mac = outputs(out)[2].read_text()
    assert '\\newcommand{\\PubFamActEqUpdates}{6000}' in mac
    assert 'pubx-eq of the same seed except for mask_mode, model_suffix' in outputs(out)[1].read_text()
    # a configuration of the recipe under the mask with other updates refuses
    cp = os.path.join(root, FAM.eq_paths(304)[3])
    json.dump(dict(json.load(open(cp)), max_updates=2000), open(cp, 'w'))
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'is not the recipe' in str(e.value) and 'max_updates' in str(e.value)


# ------------------------------------------------------------------ analysis chain
CHAIN = os.path.join(ROOT, 'scripts', 'run_analysis.sh')


def test_chain_step_and_its_precondition_text():
    src = open(CHAIN).read()
    steps = re.search(r'^STEPS=\(([^)]*)\)', src, re.M).group(1).split()
    assert steps[steps.index('public'):steps.index('public') + 3] == ['public', 'pubact', 'public_act']
    assert len(steps) == len(set(steps))
    assert re.search(r'^unset [^\n]*\bPUBACTFAM PUBACTSENS PUBACTEQ\b', src, re.M)
    outs = re.search(r'^    public_act\) printf .%s\\n. ([^;]*);;', src, re.M).group(1)
    assert outs.replace('\\', ' ').split() == [FAM.out_files[2], FAM.out_files[0], FAM.out_files[1]]
    assert 'run_cmd scripts/p89_public_family.py --family active || return 1' in src
    assert "'pubactfam-state': pubactfam_state" in src
    assert "'manuscript/macros_pubfam_act.tex'," in src            # in CHAIN of the macro table
    assert re.search(r'^    public_act\)\n      helper pubactfam-state \|\| ok=1\n', src, re.M)


def chain_helper(monkeypatch, root):
    """The Python helper of run_analysis.sh, its functions only, reading the tree root."""
    src = open(CHAIN).read()
    code = re.search(r"read -r -d '' HELPER <<'PYEOF'\n(.*?)\nPYEOF", src, re.S).group(1)
    code = code.rsplit('\nmain()', 1)[0]
    ns = {'__name__': 'figures_helper'}
    exec(compile(code, 'run_analysis.sh', 'exec'), ns)
    monkeypatch.setattr(P, 'REPO', str(root))
    return ns


@pytest.mark.parametrize('state', ['none', 'policy runs only', 'partial', 'no rules', 'complete'])
def test_chain_state_every_file_or_none(tmp_path, monkeypatch, capsys, state):
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    if state == 'policy runs only':                # the three runs of step pubact, nothing else
        keep = {FAM.model('released', None, s) for s in REG.PUB_ACT_SEEDS}
        build_active(root, skip={FAM.model(a, c, s) for a, c, _ in P.Inputs(root).runs()
                                 for s in P.SEEDS} - keep)
    elif state == 'partial':
        build_active(root, skip={'pubx-soloact-LW-s305'})
    elif state in ('no rules', 'complete'):
        build_active(root)
        if state == 'no rules':
            os.remove(os.path.join(root, FAM.rule_root, 'pub_Brandimarte' + P.XS, 'Result_MOR_SW.npy'))
    ns = chain_helper(monkeypatch, root)
    ok = ns['pubactfam_state']()
    text = capsys.readouterr().out
    assert ok == (state == 'complete'), text
    if state in ('none', 'policy runs only'):
        assert 'the step waits for the family' in text
    elif state == 'partial':
        assert 'partial' in text and 'pubx-soloact-LW-s305' in text and 's305 ' in text
    elif state == 'no rules':
        assert 'rule arrays under the active mask: 1 files missing' in text
        assert 'rules_act/pub_Brandimarte_xmask/Result_MOR_SW.npy' in text
    else:
        assert 'every file of the principal runs of seeds 301 to 305 present' in text
        assert 'pubx-eqact: no file present, section 8 is left out' in text
    assert str(tmp_path) not in text
