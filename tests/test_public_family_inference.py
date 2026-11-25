"""Tests for scripts/p89_public_family.py on synthetic result trees.

Run from the repository root:  python -m pytest tests/test_public_family_inference.py -q
CPU only; every file is written under a temporary folder.
"""
import csv
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import p89_public_family as P  # noqa: E402

N_INST = 6
B = 200
R8 = list(REG.COMBOS)
SEED_FACTOR = {301: 1.0, 302: 1.3, 303: 0.8, 304: 1.1, 305: 0.9}


@pytest.fixture(autouse=True)
def small_sets(monkeypatch):
    """Six instances per set (Hurink sets still leave out orb7)."""
    monkeypatch.setattr(P, 'SETS', {k: (w, role, N_INST, ex)
                                    for k, (w, role, _, ex) in P.SETS.items()})


def names_of(st):
    return [f'inst{i:02d}' for i in range(N_INST)]


def write_summary(path, title, rows, n, bk):
    """The table eval_family.py writes; rows = [(regime, method, makespans)]."""
    with open(path, 'w') as f:
        f.write(f'# FAMILY eval — {title}\n\nmask: first_mask=exact mask_ties=exact\n\n')
        f.write('| regime | method | mean | std | feasible | gap(cov) | gap to best known (N only) |\n')
        f.write('|---|---|---|---|---|---|---|\n')
        for r, m, v in rows:
            g = f'{np.mean(100 * (v - bk) / bk):.2f}% ({n})' if r == 'N' else 'n/a'
            f.write(f'| {r} | {m} | {v.mean():.1f} | {v.std():.1f} | {n}/{n} | n/a | {g} |\n')


def default_spec(base, r, s):
    return base * (1.0 + 0.1 * R8.index(r)) * SEED_FACTOR[s]


def build_tree(root, pol=None, ctl=None, rules=None, sens_pol=None):
    """A complete synthetic public family. pol(spec, st, r, s, base) and ctl(...) give
    the policy and control makespans from the specialist's; rules(base, st, r) the
    four rule vectors; sens_pol(...) the policy on the combinations with L of each
    sensitivity set (everything else there equals its test set)."""
    pol = pol or (lambda sp, st, r, s, base: sp * 1.004)
    ctl = ctl or (lambda sp, st, r, s, base: sp * 1.002)
    rules = rules or (lambda base, st, r: {q: base * (1.2 + 0.1 * R8.index(r)) * (1 + 0.01 * j)
                                           for j, q in enumerate(P.RULES)})
    sens_pol = sens_pol or (lambda sp, st, r, s, base: sp * 1.010)
    g = np.random.default_rng(5)
    bases = {}
    for st, (_, _, n, excl) in P.SETS.items():
        names = names_of(st)
        fam = os.path.join(root, 'data', 'PUBLIC', 'family', st)
        for r in R8:
            os.makedirs(os.path.join(fam, 'desc', r), exist_ok=True)
            for i, nm in enumerate(names):
                with open(os.path.join(fam, 'desc', r, f'{nm}.desc.json'), 'w') as f:
                    json.dump(dict(regime=r, rules_version=P.RULES_VERSION, desc_seed=1000 + i,
                                   base_instance=f'data/PUBLIC/{st}/{nm}'), f)
        with open(os.path.join(fam, 'dataset_meta.json'), 'w') as f:
            json.dump(dict(instances=names, n_instances=n, desc_seed0=1000,
                           excluded=[{'instance': f'{x}.fjs', 'reason': 'test'} for x in excl],
                           params=dict(lag_scale=1.0)), f)
        base = g.uniform(100, 200, size=n).round()
        bases[st] = base
        bk = base * 0.7
        csvp = os.path.join(root, REG.PUB_SETS[st])
        os.makedirs(os.path.dirname(csvp), exist_ok=True)
        with open(csvp, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['instance', 'best_known_makespan', 'lower_bound_if_known', 'optimal'])
            for nm, b in zip(names, bk):
                w.writerow([nm, b, b * 0.9, 'no'])
            for x in excl:
                w.writerow([x, 100, 90, 'no'])
        write_results(root, st, base, pol, ctl, rules)
    for st, (bst, _, change) in REG.PUB_SENS_SETS.items():
        src = os.path.join(root, 'data', 'PUBLIC', 'family', bst)
        dst = os.path.join(root, 'data', 'PUBLIC', 'family', st)
        for r in R8:
            os.makedirs(os.path.join(dst, 'desc', r), exist_ok=True)
            for nm in names_of(bst):
                rec = json.load(open(os.path.join(src, 'desc', r, f'{nm}.desc.json')))
                rec.update(horizon=60, setup_convention='attached')
                if 'L' in r:
                    rec['lag'] = [5, 0]
                if 'W' in r:
                    rec['windows'] = [[[1, 2]]]
                json.dump(rec, open(os.path.join(dst, 'desc', r, f'{nm}.desc.json'), 'w'))
        meta = json.load(open(os.path.join(src, 'dataset_meta.json')))
        meta['params'] = dict(meta['params'], **change)
        json.dump(meta, open(os.path.join(dst, 'dataset_meta.json'), 'w'))
        write_results(root, st, bases[bst],
                      lambda sp, s_, r, s, base: (sens_pol if 'L' in r else pol)(sp, s_, r, s, base),
                      ctl, rules)
    write_training_records(root)
    return root


def write_results(root, st, base, pol, ctl, rules):
    n = len(base)
    bk = base * 0.7
    out = os.path.join(root, REG.PUB_RESULTS, st + P.XS)
    os.makedirs(out, exist_ok=True)

    def save(tag, r, v):
        np.save(os.path.join(out, f'Result_{tag}_{r}.npy'),
                np.stack([v, np.full(n, 0.1)], axis=1))

    for s in P.SEEDS:
        for arm, combo, regs in [('released', None, R8), ('control', None, R8)] + \
                [('specialist', r, [r]) for r in R8]:
            name = REG.pub_model(arm, combo, s)
            rows = []
            for r in regs:
                sp = default_spec(base, r, s)
                v = {'released': pol, 'control': ctl}[arm](sp, st, r, s, base) \
                    if arm != 'specialist' else sp
                save(f'greedy+{REG.PUB_PREFIX}{name}', r, v)
                rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
            write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'),
                          name, rows, n, bk)
    rows = []
    for r in R8:
        for q, v in rules(base, st, r).items():
            save(q, r, v)
            rows.append((r, q, v))
    write_summary(os.path.join(out, f'summary_rules{P.XS}.md'), 'rules', rows, n, bk)


def write_training_records(root):
    os.makedirs(os.path.join(root, 'train_log', 'FAMILY'), exist_ok=True)
    for arm, combo, _ in P.Inputs(root).runs():
        regs = {'released': P.TRAIN, 'control': P.CTL_SEEN}.get(arm, [combo])
        for s in P.SEEDS:
            best = 0.9 - (0.01 if (arm == 'released' and s == 303) else 0.0)
            scores = [1.0, best + 0.02, best, best + 0.01]
            lines = ['[train_family] MASK: first_mask=exact mask_ties=exact',
                     '[vali] MWKR norms: ' + '  '.join(f'{r}=500.0' for r in regs),
                     f'[model] params total={P.N_PARAMS}  film=0 (0.00%)']
            run_best = 9.0
            for k, sc in enumerate(scores):
                run_best = min(run_best, sc)
                lines.append(f'[vali @{25 * (k + 1)}] rel={sc:.4f} (best {run_best:.4f})  '
                             + '  '.join(f'{r}={500 * sc:.1f}' for r in regs))
            lines.append(f'[train_family] done in 0.10 h; best rel={min(scores):.4f}')
            with open(os.path.join(root, REG.pub_paths(arm, combo, s)[1]), 'w') as f:
                f.write('\n'.join(lines) + '\n')
            with open(os.path.join(root, REG.pub_vali_log(arm, combo, s)), 'w') as f:
                f.write(repr(scores))


def run(root, tmp_path):
    out = tmp_path / 'out'
    return P.main(['--root', str(root), '--out', str(out), '--boot', str(B)]), out


def outputs(out):
    return [out / p for p in P.OUT_FILES]


# ------------------------------------------------------------------ tests
def test_pairs_seed_with_seed(tmp_path):
    """Specialist seeds differ by up to 30%; the policy is its own seed's specialist
    plus 0.4%. Only pairing seed s with seed s gives d = 0.4 everywhere."""
    root = build_tree(str(tmp_path / 'tree'))
    res, out = run(root, tmp_path)
    for st in P.SETS:
        for r in R8:
            c = res['sets'][st]['parity']['components'][r]
            assert c['mean'] == pytest.approx(0.4, abs=1e-9)
            assert c['lo'] == pytest.approx(0.4, abs=1e-9) and c['hi'] == pytest.approx(0.4, abs=1e-9)
            assert c['seed_means'] == pytest.approx([0.4] * 5, abs=1e-9)
            assert (c['v0.5'], c['v1.0'], c['direction']) == ('equivalent', 'equivalent', 'behind')
            pc = res['sets'][st]['control']['policy_vs_control'][r]
            assert pc['mean'] == pytest.approx(100 * (1.004 - 1.002) / 1.002, abs=1e-9)
    assert res['selection']['applied'] and res['selection']['seed'] == 303
    assert all(p.exists() for p in outputs(out))
    js = json.load(open(out / 'results' / 'public_family_inference.json'))
    assert js['boot'] == B and js['seeds'] == list(P.SEEDS)


def test_verdicts_and_joint_claims(tmp_path):
    spread = np.array([3.0, -3.0, 3.0, -3.0, 3.0, -3.0])
    d = {'N': 0.4, 'L': 0.7, 'S': -0.7, 'W': 2.0, 'LS': None, 'LW': 0.0, 'SW': -0.2, 'LSW': 0.3}

    def pol(sp, st, r, s, base):
        x = spread * (1 if s % 2 else -1) if d[r] is None else d[r]
        return sp * (1 + np.asarray(x) / 100)

    res, _ = run(build_tree(str(tmp_path / 'tree'), pol=pol), tmp_path)
    comp = res['sets']['pub_SD2_test_10x5+mix']['parity']['components']
    want = {'N': ('equivalent', 'equivalent'), 'L': ('inferior', 'equivalent'),
            'S': ('non-inferior only', 'equivalent'), 'W': ('inferior', 'inferior'),
            'LS': ('inconclusive', 'inconclusive'), 'LW': ('equivalent', 'equivalent'),
            'SW': ('equivalent', 'equivalent'), 'LSW': ('equivalent', 'equivalent')}
    for r, (v05, v1) in want.items():
        assert (comp[r]['v0.5'], comp[r]['v1.0']) == (v05, v1), r
    assert comp['S']['direction'] == 'ahead' and comp['LW']['direction'] == 'neither'
    iut = res['sets']['pub_SD2_test_10x5+mix']['parity']['iut']
    assert iut['Held|1.0|equivalent']['holds']
    assert not iut['Train|1.0|equivalent']['holds']
    assert sorted(iut['Train|1.0|equivalent']['failing']) == ['LS', 'W']
    assert not iut['All|1.0|non-inferior']['holds']


def test_best_rule_and_shares(tmp_path):
    def rules(base, st, r):
        odd = np.arange(len(base)) % 2 == 1
        return {'FIFO': base * 1.10, 'MOR': np.where(odd, base * 1.20, base * 1.05),
                'SPT': base * 1.15, 'MWKR': base * 1.08}

    def pol(sp, st, r, s, base):    # makespan independent of seed and combination
        v = base * 1.06
        v[0] = base[0] * 1.05       # instance 0: equal to the best rule (MOR)
        return v

    res, _ = run(build_tree(str(tmp_path / 'tree'), pol=pol, rules=rules), tmp_path)
    x = res['sets']['pub_Brandimarte']['rules']['L']
    odd = np.arange(N_INST) % 2 == 1
    d = np.where(odd, 100 * (1.06 - 1.08) / 1.08, 100 * (1.06 - 1.05) / 1.05)
    d[0] = 0.0
    assert x['best']['mean'] == pytest.approx(d.mean(), abs=1e-9)
    assert x['best']['shorter'] == pytest.approx(100 * np.mean(d < 0))
    assert x['best']['equal'] == pytest.approx(100 / N_INST)
    assert x['best']['longer'] == pytest.approx(100 * np.mean(d > 0))
    assert x['best_rule_share'] == pytest.approx({'FIFO': 0.0, 'MOR': 50.0, 'SPT': 0.0, 'MWKR': 50.0})
    assert x['best_selected']['mean'] == pytest.approx(d.mean(), abs=1e-9)
    fifo = np.full(N_INST, 100 * (1.06 - 1.10) / 1.10)
    fifo[0] = 100 * (1.05 - 1.10) / 1.10
    assert x['rules']['FIFO']['mean'] == pytest.approx(fifo.mean(), abs=1e-9)
    assert x['rules']['FIFO']['shorter'] == pytest.approx(100.0)
    a = res['sets']['pub_Brandimarte']['anchor']['arms']
    assert a['best rule']['gap'] == pytest.approx(
        np.mean(np.where(odd, 108 / 70 - 1, 105 / 70 - 1)) * 100, abs=1e-6)


def test_refuses_and_removes_outputs_on_a_missing_file(tmp_path):
    root = build_tree(str(tmp_path / 'tree'))
    out = tmp_path / 'out'
    for p in outputs(out):                      # stale outputs of an earlier run
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('stale')
    os.remove(os.path.join(root, REG.PUB_RESULTS, 'pub_Hurink_rdata' + P.XS,
                           f'Result_greedy+{REG.PUB_PREFIX}pubx-solo-LS-s304_LS.npy'))
    with pytest.raises(SystemExit) as e:
        P.main(['--root', root, '--out', str(out), '--boot', str(B)])
    assert 'REFUSING' in str(e.value) and 'pubx-solo-LS-s304' in str(e.value)
    assert not any(p.exists() for p in outputs(out))


def test_refuses_on_rows_out_of_record_order(tmp_path):
    """A permuted array keeps its mean but not the regime-N gap the summary recorded."""
    root = build_tree(str(tmp_path / 'tree'))
    p = os.path.join(root, REG.PUB_RESULTS, 'pub_SD2_test_20x10+mix' + P.XS,
                     f'Result_greedy+{REG.PUB_PREFIX}pubx-ctl-s302_N.npy')
    a = np.load(p)
    np.save(p, a[::-1])
    with pytest.raises(SystemExit) as e:
        run(root, tmp_path)
    assert 'row order' in str(e.value)


def test_refuses_on_nan_and_on_a_record_mismatch(tmp_path):
    root = build_tree(str(tmp_path / 'tree'))
    p = os.path.join(root, REG.PUB_RESULTS, 'pub_Brandimarte' + P.XS, 'Result_SPT_W.npy')
    a = np.load(p)
    a[2, 0] = np.nan
    np.save(p, a)
    with pytest.raises(SystemExit) as e:
        run(root, tmp_path)
    assert 'NaN' in str(e.value)
    root2 = build_tree(str(tmp_path / 'tree2'))
    q = os.path.join(root2, 'data', 'PUBLIC', 'family', 'pub_Hurink_vdata', 'desc', 'LW',
                     'inst03.desc.json')
    rec = json.load(open(q))
    rec['desc_seed'] += 1
    json.dump(rec, open(q, 'w'))
    with pytest.raises(SystemExit) as e:
        run(root2, tmp_path)
    assert 'descriptor seeds' in str(e.value)


def test_macro_names_unique_and_collision_refused(tmp_path):
    root = build_tree(str(tmp_path / 'tree'))
    res, out = run(root, tmp_path)
    text = (out / 'manuscript' / 'macros_public.tex').read_text()
    import re
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', text)
    assert len(names) == len(set(names)) == \
        res['n_macros'] + res['n_macros_verdict'] + res['n_macros_dir']
    assert res['n_macros_verdict'] == len(P.SETS) * (len(R8) + 2)
    assert sum('Verdict' in n for n in names) == res['n_macros_verdict']
    assert all(n.startswith('Pub') for n in names)
    assert '\\newcommand{\\PubSdtenParLSWVOne}{equivalent}' in text
    os.makedirs(os.path.join(root, 'paper'), exist_ok=True)
    with open(os.path.join(root, 'paper', 'macros_other.tex'), 'w') as f:
        f.write('\\newcommand{\\PubBrandRuleSW}{1}\n')
    with pytest.raises(SystemExit) as e:
        run(root, tmp_path)
    assert 'collision' in str(e.value) and 'PubBrandRuleSW' in str(e.value)
    assert not any(p.exists() for p in outputs(out))


SENS = 'pub_SD2_test_10x5+mix_lag050'


def test_sensitivity_section_and_free_check(tmp_path):
    """At lag scale 0.5 the policy is its specialist plus 1.0% on the combinations
    with L (plus 0.4% at scale 1.0); N and S equal the test set."""
    res, out = run(build_tree(str(tmp_path / 'tree')), tmp_path)
    X = res['sensitivity'][SENS]
    fc = X['checks']['free_check']
    assert fc['identical'] and fc['arrays'] == 5 * 2 * 2 + 5 * 2 + 4 * 2
    assert X['checks']['fields_changed']['N|horizon'] == N_INST
    for r in P.L_COMBOS:
        C = X['combos'][r]
        assert C['half']['policy_vs_specialist']['mean'] == pytest.approx(1.0, abs=1e-9)
        assert C['one']['policy_vs_specialist']['mean'] == pytest.approx(0.4, abs=1e-9)
        assert C['change']['par']['mean'] == pytest.approx(0.6, abs=1e-9)
        assert C['half']['policy_vs_control']['mean'] == pytest.approx(
            100 * (1.010 - 1.002) / 1.002, abs=1e-9)
        assert C['half']['control_vs_specialist']['mean'] == pytest.approx(0.2, abs=1e-9)
        assert (C['half']['policy_vs_specialist']['v1.0'],
                C['one']['policy_vs_specialist']['v1.0']) == ('inferior', 'equivalent')
    assert not X['iut']['half']['L|1.0|equivalent']['holds']
    assert X['iut']['one']['L|1.0|equivalent']['holds']
    text = (out / 'manuscript' / 'macros_public.tex').read_text()
    for name, val in (('PubLagHalfParL', '+1.00'), ('PubLagHalfBaseParL', '+0.40'),
                      ('PubLagHalfDiffParLSW', '+0.60')):
        assert f'\\newcommand{{\\{name}}}{{{val}}}' in text, name
    assert '## 7. Sensitivity' in (out / 'results' / 'public_family_inference.md').read_text()


def test_sensitivity_refusals(tmp_path):
    root = build_tree(str(tmp_path / 'tree'))
    p = os.path.join(root, REG.PUB_RESULTS, SENS + P.XS,
                     f'Result_greedy+{REG.PUB_PREFIX}pubx-v1-s302_S.npy')
    a = np.load(p)
    a[1, 0] += 0.01                       # inside the summary's rounding, not bit for bit
    np.save(p, a)
    with pytest.raises(SystemExit) as e:
        run(root, tmp_path)
    assert 'free check failed' in str(e.value)
    root2 = build_tree(str(tmp_path / 'tree2'))
    for r in R8:
        q = os.path.join(root2, 'data', 'PUBLIC', 'family', SENS, 'desc', r, 'inst02.desc.json')
        rec = json.load(open(q))
        rec['setup'] = [[1]]
        json.dump(rec, open(q, 'w'))
        os.utime(q, (1e9, 1e9))          # older than the results, as a record must be
    with pytest.raises(SystemExit) as e:
        run(root2, tmp_path)
    assert 'differs from' in str(e.value) and "'setup'" in str(e.value)
