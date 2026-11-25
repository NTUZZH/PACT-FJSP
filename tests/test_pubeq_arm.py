"""The equal-experience arm of the public family (pubx-eq, 6000 updates): its names,
the queue's task lists and configuration check, the evaluation jobs, and section 8
of scripts/p89_public_family.py on synthetic result trees (complete, absent, partial).

Run from the repository root:  python -m pytest tests/test_pubeq_arm.py -q
CPU only. The queue is run only in its listing mode (no Python, no file written);
the planner tests read data/PUBLIC/family and are skipped without it; every p89 tree
is written under a temporary folder.
"""
import contextlib
import importlib
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p89_public_family as P  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances per set)

QUEUE = os.path.join(ROOT, 'scripts', 'run_pubfamily_queue.sh')
R8 = list(REG.COMBOS)
SIX = 'N,L,S,W,LS,LW'


# ------------------------------------------------------------------ names
def test_registry_names_are_additive():
    assert REG.PUB_EQ_STEM == 'pubx-eq' and REG.PUB_EQ_UPDATES == 6000
    assert REG.pub_eq_model(302) == 'pubx-eq-s302'
    assert REG.pub_eq_paths(305) == (
        'trained_network/FAMILY/10x5+pubfamily+pubx-eq-s305.pth',
        'train_log/PUBFAMILY_pubx-eq-s305.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-eq-s305.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-eq-s305.json')
    with pytest.raises(ValueError):
        REG.pub_eq_model(306)
    # the principal arms of the public family are unchanged
    assert REG.PUB_STEM == {'released': 'pubx-v1', 'control': 'pubx-ctl', 'specialist': 'pubx-solo-{R}'}
    assert REG.ARMS == ('released', 'control', 'specialist')
    with pytest.raises(Exception):
        REG.pub_model('eq', None, 301)


# ------------------------------------------------------------------ training queue
def queue(**env):
    e = {k: v for k, v in os.environ.items() if k not in
         ('MASK', 'TYPES', 'ARMS', 'LANE', 'ONLY', 'LIST_ONLY', 'PROBE_ONLY')}
    e.update(env)
    return subprocess.run(['bash', QUEUE], env=e, capture_output=True, text=True, timeout=60)


def default_names(prefix):
    out = []
    for tier in ((301, 302, 303), (304, 305)):
        out += [f'{prefix}-v1-s{s}' for s in tier] + [f'{prefix}-ctl-s{s}' for s in tier]
        out += [f'{prefix}-solo-{r}-s{s}' for s in tier for r in R8]
    return out


def listed(p):
    assert p.returncode == 0, p.stdout + p.stderr
    return [ln.split(' ', 1)[1] for ln in p.stdout.splitlines() if ln.startswith('TASK ')]


def test_default_task_list_is_unchanged():
    assert listed(queue(LIST_ONLY='all')) == default_names('pubx')
    assert listed(queue(LIST_ONLY='all', MASK='legacy')) == default_names('publ')
    assert listed(queue(LIST_ONLY='all', TYPES='class')) == default_names('pubc')
    assert listed(queue(LIST_ONLY='all', ARMS='')) == default_names('pubx')
    lanes = [listed(queue(LIST_ONLY='all', LANE=f'{i}/3')) for i in range(3)]
    assert sorted(sum(lanes, [])) == sorted(default_names('pubx'))
    assert lanes[1] == default_names('pubx')[1::3]
    assert not any('eq' in n for n in default_names('pubx'))


def test_eq_task_list_lanes_and_refusals():
    names = [f'pubx-eq-s{s}' for s in REG.PUB_SEEDS]
    assert listed(queue(LIST_ONLY='all', ARMS='eq')) == names
    assert [listed(queue(LIST_ONLY='all', ARMS='eq', LANE=f'{i}/5')) for i in range(5)] == \
        [[n] for n in names]
    assert listed(queue(LIST_ONLY='all', ARMS='eq', ONLY='s30[45]')) == names[3:]
    for env, msg in ((dict(ARMS='eq', MASK='legacy'), 'MASK=exact TYPES=none only'),
                     (dict(ARMS='eq', TYPES='class'), 'MASK=exact TYPES=none only'),
                     (dict(ARMS='v1'), 'ARMS must be empty'),
                     (dict(LIST_ONLY='yes'), 'LIST_ONLY must be'),
                     (dict(LANE='5/5'), 'LANE must be')):
        p = queue(LIST_ONLY=env.pop('LIST_ONLY', 'all'), **env)
        assert p.returncode == 1 and msg in p.stdout, (env, p.stdout)


def eq_check(tmp_path, cfg, seed, upd, kind):
    """Run the queue's eq_check function alone, in tmp_path, on its stored files."""
    src = open(QUEUE).read()
    fn = re.search(r'^eq_check\(\) \{.*?^PYEOF\n\}\n', src, re.S | re.M).group(0)
    script = (f'PY={sys.executable}; SIX={SIX}; EQ_UPDATES=6000; WANT_PARAMS=29602\n{fn}\n'
              f'cd {tmp_path} && eq_check {cfg} {seed} {upd} {kind}\n')
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=60)


def write_cfgs(tmp_path):
    d = tmp_path / 'train_log' / 'FAMILY'
    d.mkdir(parents=True)
    base = dict(device='cpu', family_regimes=SIX, max_updates=2000, first_mask='exact',
                mask_ties='exact', family_base='sd2', use_type_embedding=False, lr=0.0003,
                n_m=5, pub_vali_dir='data/PUBLIC/family/pub_SD2_vali_10x5+mix')
    for s in REG.PUB_SEEDS:
        c = dict(base, model_suffix=f'pubx-v1-s{s}', seed_train=s)
        if s != 301:
            c['reward_mode'] = 'dense'      # added to the trainer between the stored runs
        json.dump(c, open(d / f'config_10x5+pubfamily+pubx-v1-s{s}.json', 'w'))
    for s in (301, 302, 303):
        json.dump(dict(family_regimes=SIX, max_updates=6000),
                  open(d / f'config_10x25+family+jointblind-eq-s{s}.json', 'w'))
    return d, base


def test_queue_config_check(tmp_path):
    d, base = write_cfgs(tmp_path)
    good = dict(base, model_suffix='pubx-eq-s302', seed_train=302, max_updates=6000,
                reward_mode='dense', pub_type_source='none')
    json.dump(good, open(tmp_path / 'run.json', 'w'))
    p = eq_check(tmp_path, 'run.json', 302, 6000, 'run')
    assert p.returncode == 0 and "differs only in ['max_updates', 'model_suffix']" in p.stdout, p.stderr
    probe = dict(good, model_suffix='_pubprobe_eq_0', seed_train=301, max_updates=1)
    json.dump(probe, open(tmp_path / 'probe.json', 'w'))
    p = eq_check(tmp_path, 'probe.json', 301, 1, 'probe')
    assert p.returncode == 0 and 'pubx-v1-s301..305 are one recipe' in p.stdout, p.stderr
    for change, msg in ((dict(lr=0.001), 'configuration drift'),
                        (dict(max_updates=2000), 'max_updates=2000'),
                        (dict(seed_train=303), 'configuration drift'),
                        (dict(pub_type_source='class'), 'configuration drift')):
        json.dump(dict(good, **change), open(tmp_path / 'bad.json', 'w'))
        p = eq_check(tmp_path, 'bad.json', 302, 6000, 'run')
        assert p.returncode != 0 and msg in (p.stdout + p.stderr), change
    # a reference that is not one recipe with the others refuses the probe
    c = json.load(open(d / 'config_10x5+pubfamily+pubx-v1-s304.json'))
    json.dump(dict(c, lr=0.001), open(d / 'config_10x5+pubfamily+pubx-v1-s304.json', 'w'))
    p = eq_check(tmp_path, 'probe.json', 301, 1, 'probe')
    assert p.returncode != 0 and 'pubx-v1-s304 differs from pubx-v1-s301' in p.stderr


# ------------------------------------------------------------------ evaluation plan
needs_public = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family')),
    reason='needs the public-family records')


@contextlib.contextmanager
def planner(groups):
    os.chdir(ROOT)
    keep = {k: os.environ.pop(k, None) for k in ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'LEGACY45')}
    import eval_newseeds_plan as EP
    try:
        os.environ['EVAL_GROUPS'] = groups
        EP = importlib.reload(EP)
        EP.EXPECTED.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            EP.check()
        yield EP
    finally:
        os.environ.pop('EVAL_GROUPS', None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(EP)


@needs_public
def test_eval_jobs_are_the_released_jobs_renamed():
    with planner('public pubsens') as EP:
        before = EP.jobs()
        pre_before = EP.preflights()
    with planner('public pubsens pubeq') as EP:
        after = EP.jobs()
        eqj = [j for j in after if j['group'] == 'pubeq']
        assert [j for j in after if j['group'] != 'pubeq'] == before
        assert EP.preflights() == pre_before
        sets = list(REG.PUB_SETS) + list(REG.PUB_SENS_SETS)
        assert len(eqj) == len(sets) * len(REG.PUB_SEEDS)
        assert len({j['id'] for j in after}) == len(after)
        rel = {j['id']: j for j in before}
        for j in eqj:
            k = rel[j['id'].replace('pubx-eq-', 'pubx-v1-')]
            assert sorted(x for x in EP.KEYS_JOB if x != 'pre' and j[x] != k[x]) == \
                ['ckpt', 'group', 'id', 'log', 'model'], j['id']
            assert j['files'] == [f.replace('pubx-v1-', 'pubx-eq-') for f in k['files']]
            assert j['regimes'] == ' '.join(R8) and EP.preflight_of(j) == 'pubdet'
            s = int(j['id'][-3:])
            assert (j['ckpt'], j['log']) == REG.pub_eq_paths(s)[:2]
        planned = {f for j in eqj for f in j['files']}
        assert planned == {f for f in EP.EXPECTED if 'pubx-eq-' in f}
    with planner('pubeq') as EP:
        assert [p['name'] for p in EP.preflights()] == ['xdet', 'xcross_policy', 'xcross_rules',
                                                         'xdet_rules', 'pubdet']
        assert [j['id'] for j in EP.jobs()] == [j['id'] for j in sorted(
            eqj, key=lambda j: (-j['cost'], j['id']))]


# ------------------------------------------------------------------ p89, section 8
EQ_FACTOR = 1.001        # the equal-experience policy: its seed's specialist plus 0.1%


def write_eq(root, factor=EQ_FACTOR):
    """The five pubx-eq runs on a tree of build_tree: arrays and summaries in every
    test and sensitivity folder, training records, configurations and checkpoints,
    and the configurations of the pubx-v1 runs they are compared with."""
    for st in list(P.SETS) + list(REG.PUB_SENS_SETS):
        out = os.path.join(root, REG.PUB_RESULTS, st + P.XS)
        # the seed-301 N specialist of build_tree is the base makespan vector itself
        base = np.load(os.path.join(
            out, f'Result_greedy+{REG.PUB_PREFIX}{REG.pub_model("specialist", "N", 301)}_N.npy'))[:, 0]
        bk = base * 0.7
        n = len(base)
        for s in P.SEEDS:
            name = REG.pub_eq_model(s)
            rows = []
            for r in R8:
                v = T.default_spec(base, r, s) * factor
                np.save(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'),
                        np.stack([v, np.full(n, 0.1)], axis=1))
                rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
            T.write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'),
                            name, rows, n, bk)
    os.makedirs(os.path.join(root, 'trained_network', 'FAMILY'), exist_ok=True)
    for s in P.SEEDS:
        ck, lp, vp, cp = REG.pub_eq_paths(s)
        scores = [1.0, 0.88, 0.87, 0.875]
        lines = ['[train_family] MASK: first_mask=exact mask_ties=exact',
                 '[vali] MWKR norms: ' + '  '.join(f'{r}=500.0' for r in P.TRAIN),
                 f'[model] params total={P.N_PARAMS}  film=0 (0.00%)']
        best = 9.0
        for k, sc in enumerate(scores):
            best = min(best, sc)
            lines.append(f'[vali @{1500 * (k + 1)}] rel={sc:.4f} (best {best:.4f})  '
                         + '  '.join(f'{r}={500 * sc:.1f}' for r in P.TRAIN))
        lines.append(f'[train_family] done in 0.30 h; best rel={min(scores):.4f}')
        open(os.path.join(root, lp), 'w').write('\n'.join(lines) + '\n')
        open(os.path.join(root, vp), 'w').write(repr(scores))
        open(os.path.join(root, ck), 'w').close()
        ref = dict(device='cpu', family_regimes=SIX, max_updates=2000, seed_train=s,
                   model_suffix=f'pubx-v1-s{s}', first_mask='exact', mask_ties='exact')
        json.dump(ref, open(os.path.join(root, 'train_log', 'FAMILY',
                                         f'config_{REG.PUB_PREFIX}pubx-v1-s{s}.json'), 'w'))
        json.dump(dict(ref, model_suffix=REG.pub_eq_model(s), max_updates=6000, reward_mode='dense'),
                  open(os.path.join(root, cp), 'w'))


def read_out(out):
    return [open(p).read() for p in T.outputs(out)]


def test_absent_arm_changes_nothing_and_complete_arm_adds_only(tmp_path):
    root = T.build_tree(str(tmp_path / 'tree'))
    res0, out0 = T.run(root, tmp_path / 'a')
    assert 'equal_experience' not in res0
    js0, md0, mac0 = read_out(out0)
    assert 'PubEq' not in mac0 and '## 8.' not in md0
    write_eq(root)
    res1, out1 = T.run(root, tmp_path / 'b')
    js1, md1, mac1 = read_out(out1)
    # section 9 (the references) is the last block of both files, so the arm's section 8
    # is inserted before it: everything ahead of section 9 is a prefix, and section 9 itself
    # is the same with and without the arm
    cut_mac = lambda t: t.index('% Section 9')
    cut_md = lambda t: t.index('## 9.')
    assert mac1.startswith(mac0[:cut_mac(mac0)]) and md1.startswith(md0[:cut_md(md0)].rstrip('\n'))
    assert mac1[cut_mac(mac1):] == mac0[cut_mac(mac0):]
    assert md1[cut_md(md1):].split('\n')[:3] == md0[cut_md(md0):].split('\n')[:3]
    mac0 = mac0[:cut_mac(mac0)]
    # the combined-verdict macros and section precede section 8, so the arm only appends
    assert 'RuleVerdictAheadN' in mac0 and '## Combined verdict' in md0
    # the pooled directions of sections 1, 2 and 7 precede section 8; those of section 8
    # are written with it (d = 0.1 exactly against the specialists, so behind)
    assert '\\newcommand{\\PubSdtenParPoolAllDir}{behind}' in mac0
    assert '\\newcommand{\\PubEqSdtenParPoolAllDir}{behind}' in mac1[len(mac0):]
    assert 'PubEqBrandVsPolPoolHeldDir' in mac1 and 'PubEq' not in mac0
    j0, j1 = json.loads(js0), json.loads(js1)
    E = j1.pop('equal_experience')
    assert sorted(k for k in set(j0) | set(j1) if j0.get(k) != j1.get(k)) == ['n_macros']
    # seed s against seed s: d = 0.1 exactly on every combination and set
    for st in P.SETS:
        S = E['sets'][st]
        for r in R8:
            c = S['parity']['components'][r]
            assert c['mean'] == pytest.approx(0.1, abs=1e-9) and c['seed_means'] == pytest.approx([0.1] * 5)
            assert (c['v0.5'], c['v1.0']) == ('equivalent', 'equivalent')
            want = 100 * (EQ_FACTOR - 1.004) / 1.004          # build_tree's policy: specialist + 0.4%
            assert S['vs_policy']['components'][r]['mean'] == pytest.approx(want, abs=1e-9)
        for fam in ('All', 'Train', 'Held'):
            assert S['parity']['pooled'][fam]['mean'] == pytest.approx(0.1, abs=1e-9)
        assert S['parity']['iut']['All|0.5|equivalent']['holds']
    assert E['config_differs_in']['s301'] == ['max_updates', 'model_suffix']
    assert E['checks'][T.SENS]['free_check']['arrays'] == 10
    assert E['files_read'] == (len(P.SETS) + len(REG.PUB_SENS_SETS)) * 5 * 8
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', mac1[len(mac0):])
    assert names and all(n.startswith('PubEq') for n in names)
    assert len(set(names)) == len(names)
    assert '\\newcommand{\\PubEqSdtenParN}{+0.10}' in mac1
    assert '\\newcommand{\\PubEqUpdates}{6000}' in mac1
    assert any(n.startswith('PubEqBrandParPool') for n in names)
    # the other sets carry regime N alone; the remaining combinations are headline-set macros
    assert any(n.startswith('PubEqBrandParN') for n in names)
    assert not any(re.match(r'PubEqBrandPar(LSW|LS|LW|SW|L|S|W)(?![a-z])', n) for n in names)
    assert '## 8. Equal-experience policy' in md1


def test_same_quantity_same_number(tmp_path):
    """An equal-experience arm identical to the released policy reproduces section 1
    bit for bit (same estimator, same fresh stream per quantity)."""
    root = T.build_tree(str(tmp_path / 'tree'))
    write_eq(root, factor=1.004)
    res, out = T.run(root, tmp_path / 'a')
    E = res['equal_experience']
    for st in P.SETS:
        for r in R8:
            a = E['sets'][st]['parity']['components'][r]
            b = res['sets'][st]['parity']['components'][r]
            assert (a['mean'], a['lo'], a['hi'], a['t_lo'], a['t_hi']) == \
                (b['mean'], b['lo'], b['hi'], b['t_lo'], b['t_hi'])
        for fam in ('All', 'Train', 'Held'):
            a, b = E['sets'][st]['parity']['pooled'][fam], res['sets'][st]['parity']['pooled'][fam]
            assert (a['mean'], a['lo'], a['hi']) == (b['mean'], b['lo'], b['hi'])


@pytest.mark.parametrize('damage', ['missing array', 'missing summary', 'unfinished run',
                                    'sixth seed', 'checkpoint only', 'config drift'])
def test_partial_or_wrong_arm_refuses(tmp_path, damage):
    root = T.build_tree(str(tmp_path / 'tree'))
    out = tmp_path / 'out'
    for p in T.outputs(out):                      # stale outputs of an earlier run
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('stale')
    if damage == 'checkpoint only':               # a run in training, nothing evaluated yet
        os.makedirs(os.path.join(root, 'trained_network', 'FAMILY'), exist_ok=True)
        open(os.path.join(root, REG.pub_eq_paths(303)[0]), 'w').close()
        msg = 'incomplete'
    else:
        write_eq(root)
        msg = 'incomplete'
        if damage == 'missing array':
            os.remove(os.path.join(root, REG.PUB_RESULTS, 'pub_Hurink_rdata' + P.XS,
                                   f'Result_greedy+{REG.PUB_PREFIX}pubx-eq-s304_SW.npy'))
        elif damage == 'missing summary':
            os.remove(os.path.join(root, REG.PUB_RESULTS, 'pub_SD2_test_10x5+mix_lag050' + P.XS,
                                   f'summary_{REG.PUB_PREFIX}pubx-eq-s301{P.XS}.md'))
        elif damage == 'unfinished run':
            lp = os.path.join(root, REG.pub_eq_paths(302)[1])
            txt = open(lp).read()
            open(lp, 'w').write(txt.replace('[train_family] done in', '[train_family] stopped at'))
        elif damage == 'sixth seed':
            open(os.path.join(root, 'trained_network', 'FAMILY',
                              f'{REG.PUB_PREFIX}pubx-eq-s306.pth'), 'w').close()
        elif damage == 'config drift':
            cp = os.path.join(root, REG.pub_eq_paths(305)[3])
            c = json.load(open(cp))
            json.dump(dict(c, lr=0.001), open(cp, 'w'))
            msg = 'is not the recipe'
    with pytest.raises(SystemExit) as e:
        P.main(['--root', root, '--out', str(out), '--boot', str(T.B)])
    assert 'REFUSING' in str(e.value) and msg in str(e.value), str(e.value)
    assert not any(p.exists() for p in T.outputs(out))
