"""The arm of the public family under the active mask (pubx-act, the released recipe under
--mask_mode active, seeds 301-303): its names, the queue's task lists, training arguments
and configuration check, the evaluation jobs, the analysis step of the chain, and
scripts/p108_public_active_arm.py on synthetic result trees (complete, absent, partial).

Run from the repository root:  python -m pytest tests/test_pubact_arm.py -q
CPU only. The queue is run only in its listing mode and its functions alone (no
training); the planner tests read data/PUBLIC/family and are skipped without it; every
p108 tree is written under a temporary folder.
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
import p108_public_active_arm as A  # noqa: E402
import p89_public_family as P  # noqa: E402
import stats_family as SF  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances per set)
from test_public_references import bases, ref_value, write_refs  # noqa: E402
from test_pubeq_arm import default_names, listed, queue  # noqa: E402

QUEUE = os.path.join(ROOT, 'scripts', 'run_pubfamily_queue.sh')
CHAIN = os.path.join(ROOT, 'scripts', 'run_analysis.sh')
R8 = list(REG.COMBOS)
SIX = 'N,L,S,W,LS,LW'
SEEDS = (301, 302, 303)
B = T.B
ST = A.HEADLINE


# ------------------------------------------------------------------ names
def test_registry_names_are_additive():
    assert (REG.PUB_ACT_RULE, REG.PUB_ACT_STEM, REG.PUB_ACT_SEEDS) == ('active', 'pubx-act', SEEDS)
    assert REG.PUB_RULE_WORD == {'active': 'act'}
    assert REG.pub_act_model(302) == 'pubx-act-s302'
    assert REG.pub_act_paths(303) == (
        'trained_network/FAMILY/10x5+pubfamily+pubx-act-s303.pth',
        'train_log/PUBFAMILY_pubx-act-s303.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-act-s303.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-act-s303.json')
    for bad in (300, 304, 305):
        with pytest.raises(ValueError):
            REG.pub_act_model(bad)
    # the principal arms and the equal-experience arm of the public family are unchanged
    assert REG.PUB_STEM == {'released': 'pubx-v1', 'control': 'pubx-ctl', 'specialist': 'pubx-solo-{R}'}
    assert (REG.PUB_EQ_STEM, REG.PUB_EQ_UPDATES, REG.PUB_SEEDS) == ('pubx-eq', 6000, (301, 302, 303, 304, 305))
    with pytest.raises(Exception):
        REG.pub_model('act', None, 301)
    # p108 reads the registry's paths for the arm and p89's for the comparator
    assert A.run_paths('active', 302) == REG.pub_act_paths(302)
    assert A.run_paths('policy', 302) == (
        'trained_network/FAMILY/10x5+pubfamily+pubx-v1-s302.pth', 'train_log/PUBFAMILY_pubx-v1-s302.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-v1-s302.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-v1-s302.json')


# ------------------------------------------------------------------ training queue
def test_task_lists_lanes_and_refusals():
    names = [f'pubx-act-s{s}' for s in SEEDS]
    assert listed(queue(LIST_ONLY='all', ARMS='act')) == names
    assert [listed(queue(LIST_ONLY='all', ARMS='act', LANE=f'{i}/3')) for i in range(3)] == \
        [[n] for n in names]
    assert listed(queue(LIST_ONLY='all', ARMS='act', ONLY='s30[23]')) == names[1:]
    # the default and eq lists are unchanged and hold no run of the arm
    assert listed(queue(LIST_ONLY='all')) == default_names('pubx')
    assert listed(queue(LIST_ONLY='all', ARMS='eq')) == [f'pubx-eq-s{s}' for s in REG.PUB_SEEDS]
    assert not any('act' in n for n in default_names('pubx'))
    for env, msg in ((dict(ARMS='act', MASK='legacy'), 'ARMS=act is defined for MASK=exact TYPES=none only'),
                     (dict(ARMS='act', TYPES='class'), 'ARMS=act is defined for MASK=exact TYPES=none only'),
                     (dict(ARMS='active'), 'ARMS must be empty'),
                     (dict(ARMS='v1'), 'ARMS must be empty')):
        p = queue(LIST_ONLY='all', **env)
        assert p.returncode == 1 and msg in p.stdout, (env, p.stdout)


def queue_fn(name):
    src = open(QUEUE).read()
    return re.search(r'^%s\(\) \{.*?^\}\n' % name, src, re.S | re.M).group(0)


def test_training_arguments_carry_the_rule_only_for_the_rule_arm(tmp_path):
    fn = queue_fn('train_args')
    script = (f"COMMON='--data_source FAMILY --n_j 10'\n{fn}\n"
              'train_args "$@"; printf "%s\\n" "${TRAIN_ARGS[@]}"\n')

    def args(*a):
        r = subprocess.run(['bash', '-c', script, 'x'] + list(a), capture_output=True, text=True,
                           timeout=30)
        assert r.returncode == 0, r.stderr
        return r.stdout.splitlines()
    base = ['--data_source', 'FAMILY', '--n_j', '10', '--family_regimes', SIX, '--max_updates', '2000',
            '--seed_train', '302', '--model_suffix']
    assert args(SIX, '2000', '302', 'pubx-v1-s302', '') == base + ['pubx-v1-s302']
    assert args(SIX, '2000', '302', 'pubx-v1-s302') == base + ['pubx-v1-s302']
    assert args(SIX, '2000', '302', 'pubx-act-s302', 'active') == \
        base + ['pubx-act-s302', '--mask_mode', 'active']
    src = open(QUEUE).read()
    # the probe and every run train through train_args, the probe with the arm's rule
    assert 'train_args "$SIX" 1 301 "$suf" "$RULE"' in src
    assert 'train_args "$regs" "$upd" "$seed" "$name" "$rule"' in src
    assert src.count('train_family.py "${TRAIN_ARGS[@]}"') == 2
    assert 'RULE_ARMS=("active|act")' in src


def rule_check(tmp_path, cfg, seed, upd, kind, rule='active'):
    """Run the queue's rule_check function alone, in tmp_path, on its stored files."""
    fn = re.search(r'^rule_check\(\) \{.*?^PYEOF\n\}\n', open(QUEUE).read(), re.S | re.M).group(0)
    script = (f'PY={sys.executable}; SIX={SIX}; RULE={rule}; ARMS=act; RULE_SEEDS="301 302 303"; '
              f'RULE_UPDATES=2000; WANT_PARAMS=29602\n{fn}\n'
              f'cd {tmp_path} && rule_check {cfg} {seed} {upd} {kind}\n')
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=60)


def write_cfgs(tmp_path):
    d = tmp_path / 'train_log' / 'FAMILY'
    d.mkdir(parents=True)
    base = dict(device='cpu', family_regimes=SIX, max_updates=2000, first_mask='exact',
                mask_ties='exact', family_base='sd2', use_type_embedding=False, lr=0.0003,
                n_m=5, pub_vali_dir='data/PUBLIC/family/pub_SD2_vali_10x5+mix', mask_mode='nondelay')
    for s in REG.PUB_SEEDS:
        c = dict(base, model_suffix=f'pubx-v1-s{s}', seed_train=s)
        if s != 301:
            c['reward_mode'] = 'dense'      # added to the trainer between the stored runs
        json.dump(c, open(d / f'config_10x5+pubfamily+pubx-v1-s{s}.json', 'w'))
    return d, base


def test_queue_config_check(tmp_path):
    d, base = write_cfgs(tmp_path)
    good = dict(base, model_suffix='pubx-act-s302', seed_train=302, mask_mode='active',
                reward_mode='dense', pub_type_source='none')
    json.dump(good, open(tmp_path / 'run.json', 'w'))
    p = rule_check(tmp_path, 'run.json', 302, 2000, 'run')
    assert p.returncode == 0 and "differs only in ['mask_mode', 'model_suffix']" in p.stdout, p.stderr
    probe = dict(good, model_suffix='_pubprobe_act_0', seed_train=301, max_updates=1)
    probe.pop('reward_mode')
    json.dump(probe, open(tmp_path / 'probe.json', 'w'))
    p = rule_check(tmp_path, 'probe.json', 301, 1, 'probe')
    assert p.returncode == 0 and "differs only in ['mask_mode', 'max_updates', 'model_suffix']" in p.stdout \
        and 'pubx-v1-s301..303 are one recipe' in p.stdout, p.stderr
    for change, kind, upd, msg in ((dict(lr=0.001), 'run', 2000, 'configuration drift'),
                                   (dict(mask_mode='nondelay'), 'run', 2000, "mask_mode='nondelay'"),
                                   (dict(max_updates=6000), 'run', 2000, 'max_updates=6000'),
                                   (dict(max_updates=6000), 'run', 6000, 'configuration drift'),
                                   (dict(seed_train=303), 'run', 2000, 'configuration drift'),
                                   (dict(pub_type_source='class'), 'run', 2000, 'configuration drift'),
                                   (dict(pair_extra='sw'), 'run', 2000, 'configuration drift'),
                                   (dict(max_updates=2000), 'probe', 1, 'max_updates=2000')):
        json.dump(dict(good, **change), open(tmp_path / 'bad.json', 'w'))
        p = rule_check(tmp_path, 'bad.json', 302, upd, kind)
        assert p.returncode != 0 and msg in (p.stdout + p.stderr), change
    # a rule arm under the non-delay rule is no arm
    json.dump(dict(good, mask_mode='nondelay'), open(tmp_path / 'nd.json', 'w'))
    p = rule_check(tmp_path, 'nd.json', 302, 2000, 'run', rule='nondelay')
    assert p.returncode != 0 and 'want a rule other than nondelay' in p.stderr
    # a reference under another rule, or references that are not one recipe, refuse
    c = json.load(open(d / 'config_10x5+pubfamily+pubx-v1-s302.json'))
    json.dump(dict(c, mask_mode='active'), open(d / 'config_10x5+pubfamily+pubx-v1-s302.json', 'w'))
    p = rule_check(tmp_path, 'run.json', 302, 2000, 'run')
    assert p.returncode != 0 and 'want the non-delay rule' in p.stderr
    json.dump(c, open(d / 'config_10x5+pubfamily+pubx-v1-s302.json', 'w'))
    c = json.load(open(d / 'config_10x5+pubfamily+pubx-v1-s303.json'))
    json.dump(dict(c, lr=0.001), open(d / 'config_10x5+pubfamily+pubx-v1-s303.json', 'w'))
    p = rule_check(tmp_path, 'probe.json', 301, 1, 'probe')
    assert p.returncode != 0 and 'pubx-v1-s303 differs from pubx-v1-s301' in p.stderr
    # seeds 304 and 305 are no reference of the arm
    os.remove(d / 'config_10x5+pubfamily+pubx-v1-s305.json')
    json.dump(c, open(d / 'config_10x5+pubfamily+pubx-v1-s303.json', 'w'))
    assert rule_check(tmp_path, 'probe.json', 301, 1, 'probe').returncode == 0


# ------------------------------------------------------------------ evaluation plan
needs_public = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family')),
    reason='needs the public-family records')
PLAN_ENV = ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'PUBACT', 'LEGACY45', 'CROSS')


@contextlib.contextmanager
def planner(groups=None, **env):
    os.chdir(ROOT)
    keep = {k: os.environ.pop(k, None) for k in PLAN_ENV}
    import eval_newseeds_plan as EP
    try:
        if groups is not None:
            os.environ['EVAL_GROUPS'] = groups
        os.environ.update(env)
        EP = importlib.reload(EP)
        EP.EXPECTED.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            EP.check()
        yield EP
    finally:
        for k in PLAN_ENV:
            os.environ.pop(k, None)
        for k, v in keep.items():
            if v is not None:
                os.environ[k] = v
        importlib.reload(EP)


@needs_public
def test_eval_jobs_are_the_released_jobs_renamed():
    with planner('public pubsens pubeq') as EP:
        before = EP.jobs()
        pre_before = EP.preflights()
    with planner('public pubsens pubeq pubact') as EP:
        after = EP.jobs()
        actj = [j for j in after if j['group'] == 'pubact']
        assert [j for j in after if j['group'] != 'pubact'] == before
        assert EP.preflights() == pre_before
        assert EP.PUB_ACT_SETS == ('pub_SD2_test_10x5+mix', 'pub_SD2_test_20x10+mix')
        assert len(actj) == 2 * 3 and len({j['id'] for j in after}) == len(after)
        rel = {j['id']: j for j in before}
        for j in actj:
            k = rel[j['id'].replace('pubx-act-', 'pubx-v1-')]
            assert sorted(x for x in EP.KEYS_JOB if x != 'pre' and j[x] != k[x]) == \
                ['ckpt', 'group', 'id', 'log', 'model'], j['id']
            assert j['files'] == [f.replace('pubx-v1-', 'pubx-act-') for f in k['files']]
            assert j['regimes'] == ' '.join(R8) and EP.preflight_of(j) == 'pubdet'
            assert j['rule_set'] == 'classical' and j['extra'] == '-'   # the rule comes from the config
            s = int(j['id'][-3:])
            assert s in SEEDS and (j['ckpt'], j['log']) == REG.pub_act_paths(s)[:2]
        planned = {f for j in actj for f in j['files']}
        assert planned == {f for f in EP.EXPECTED if 'pubx-act-' in f}
        # p108 reads exactly the planned files of the headline set
        head = {f for f in planned if f'/{ST}_xmask/' in f}
        assert len(head) == 3 * 9
        assert head == {f for f in A.expected('active') if f.startswith(REG.PUB_RESULTS)}
    with planner('pubact') as EP:
        assert [p['name'] for p in EP.preflights()] == ['xdet', 'xcross_policy', 'xcross_rules',
                                                         'xdet_rules', 'pubdet']
        assert [j['id'] for j in EP.jobs()] == [j['id'] for j in sorted(
            actj, key=lambda j: (-j['cost'], j['id']))]
    with planner('public', PUBACT='1') as EP:             # the switch adds the group
        assert 'pubact' in EP.GROUPS and {j['id'] for j in EP.jobs() if j['group'] == 'pubact'} == \
            {j['id'] for j in actj}


def test_eval_queue_and_chain_name_the_group_and_the_step():
    q = open(os.path.join(ROOT, 'scripts', 'run_eval_newseeds.sh')).read()
    assert 'PUBACT=1' in q
    src = open(CHAIN).read()
    steps = re.search(r'^STEPS=\(([^)]*)\)', src, re.M).group(1).split()
    assert steps[steps.index('public') + 1] == 'pubact'
    assert re.search(r'^unset [^\n]*\bPUBACT\b', src, re.M)
    outs = re.search(r'^    pubact\) printf .%s\\n. ([^;]*);;', src, re.M).group(1)
    outs = outs.replace('\\', ' ').split()
    assert outs == [A.OUT_FILES[2], A.OUT_FILES[0], A.OUT_FILES[1]]
    assert 'run_cmd scripts/p108_public_active_arm.py infer || return 1' in src
    assert "'pubact-state': pubact_state" in src and 'manuscript/macros_pubact.tex' in src


# ------------------------------------------------------------------ p108 on synthetic trees
# d = C[r] + E[s] exactly: the policy under the active mask is the policy of its seed times
# (1 + (C[r] + E[s]) / 100) on every instance
C = {'N': -2.0, 'L': -1.0, 'S': -3.0, 'W': 1.0, 'LS': -0.5, 'LW': 0.2, 'SW': 2.0, 'LSW': -1.5}
E = {301: -0.4, 302: 0.0, 303: 0.4}
ACT_SCORES = [1.0, 0.89, 0.88, 0.885]          # kept checkpoint at the third validation
OLD = 1e9                                       # checkpoint times, before every array


def pol_of(root, s, r):
    return np.load(os.path.join(root, A.array_path('policy', s, r)))[:, 0]


def write_comparator(root):
    """What build_tree leaves out for pubx-v1 of seeds 301-303: checkpoints and
    configurations (seed 301 without the reward_mode key, as stored)."""
    os.makedirs(os.path.join(root, 'trained_network', 'FAMILY'), exist_ok=True)
    for s in SEEDS:
        ck, _, _, cp = A.run_paths('policy', s)
        open(os.path.join(root, ck), 'w').close()
        os.utime(os.path.join(root, ck), (OLD, OLD))
        c = dict(device='cpu', family_regimes=SIX, max_updates=2000, seed_train=s,
                 model_suffix=f'pubx-v1-s{s}', first_mask='exact', mask_ties='exact',
                 mask_mode='nondelay', lr=0.0003, family_base='sd2', n_m=5, use_type_embedding=False)
        if s != 301:
            c['reward_mode'] = 'dense'
        json.dump(c, open(os.path.join(root, cp), 'w'))


def write_act(root, tweak=None, seeds=SEEDS):
    """The three pubx-act runs: arrays and summaries on the headline set, training records,
    configurations and checkpoints. tweak(s, r, v) may change an array before it and its
    summary are written."""
    base = bases(root)
    bk = base * 0.7
    out = os.path.join(root, REG.PUB_RESULTS, ST + P.XS)
    for s in seeds:
        name = REG.pub_act_model(s)
        rows = []
        for r in R8:
            v = pol_of(root, s, r) * (1 + (C[r] + E[s]) / 100)
            if tweak:
                v = tweak(s, r, v.copy())
            np.save(os.path.join(out, f'Result_greedy+{REG.PUB_PREFIX}{name}_{r}.npy'),
                    np.stack([v, np.full(len(v), 0.1)], axis=1))
            rows.append((r, f'greedy+{REG.PUB_PREFIX}{name}', v))
        T.write_summary(os.path.join(out, f'summary_{REG.PUB_PREFIX}{name}{P.XS}.md'), name, rows,
                        len(base), bk)
        ck, lp, vp, cp = REG.pub_act_paths(s)
        lines = ['[train_family] MASK: first_mask=exact mask_ties=exact',
                 A.BANNER,
                 '[vali] MWKR norms: ' + '  '.join(f'{r}=500.0' for r in P.TRAIN),
                 f'[model] params total={P.N_PARAMS}  film=0 (0.00%)']
        best = 9.0
        for k, sc in enumerate(ACT_SCORES):
            best = min(best, sc)
            lines.append(f'[vali @{25 * (k + 1)}] rel={sc:.4f} (best {best:.4f})  '
                         + '  '.join(f'{r}={500 * sc:.1f}' for r in P.TRAIN))
        lines.append(f'[train_family] done in 0.90 h; best rel={min(ACT_SCORES):.4f}')
        open(os.path.join(root, lp), 'w').write('\n'.join(lines) + '\n')
        open(os.path.join(root, vp), 'w').write(repr(ACT_SCORES))
        os.makedirs(os.path.dirname(os.path.join(root, ck)), exist_ok=True)
        open(os.path.join(root, ck), 'w').close()
        os.utime(os.path.join(root, ck), (OLD, OLD))
        ref = json.load(open(os.path.join(root, A.run_paths('policy', s)[3])))
        json.dump(dict(ref, model_suffix=name, mask_mode='active', reward_mode='dense',
                       pub_type_source='none'), open(os.path.join(root, cp), 'w'))


def complete_tree(tmp_path, **kw):
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    write_comparator(root)
    write_act(root, **kw)
    return root


def run(root, out):
    return A.main(['infer', '--root', str(root), '--out', str(out), '--boot', str(B)])


def outputs(out):
    return [out / p for p in A.OUT_FILES]


def edit(path, old, new):
    """Replace old by new in a text file; old must occur in it."""
    with open(path) as f:
        txt = f.read()
    assert old in txt, (path, old)
    with open(path, 'w') as f:
        f.write(txt.replace(old, new))


def snapshot(root):
    out = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[os.path.relpath(p, root)] = (st.st_size, st.st_mtime_ns)
    return out


def test_complete_arm_hand_computed_differences_and_outputs(tmp_path):
    root = complete_tree(tmp_path)
    before = snapshot(root)
    out = tmp_path / 'out'
    res = run(root, out)
    # nothing is written outside --out, and only the three outputs inside it
    assert snapshot(root) == before
    assert sorted(snapshot(str(out))) == sorted(A.OUT_FILES)
    assert not any(os.path.exists(os.path.join(ROOT, p)) and
                   os.path.getmtime(os.path.join(ROOT, p)) >= os.path.getmtime(str(out / p))
                   for p in A.OUT_FILES)
    n = T.N_INST
    for r in R8:
        d = np.stack([np.full(n, C[r] + E[s]) for s in SEEDS])
        c = res['components'][r]
        hb = SF.hier_boot(d, B=B, seed=11)
        assert c['mean'] == pytest.approx(C[r], abs=1e-9)
        assert (c['lo'], c['hi']) == pytest.approx((hb['lo'], hb['hi']), abs=1e-9)
        assert c['seed_means'] == pytest.approx([C[r] + E[s] for s in SEEDS], abs=1e-9)
        sh = [100.0 * (C[r] + E[s] < 0) for s in SEEDS]       # every instance of a seed on one side
        assert (c['shorter'], c['longer'], c['equal']) == pytest.approx(
            (np.mean(sh), 100.0 - np.mean(sh), 0.0))
    assert res['components']['LW']['shorter'] == pytest.approx(100 / 3)
    assert res['below'] == ['N', 'L', 'S', 'LS', 'LSW'] and res['above'] == ['W', 'SW']
    assert (res['below_n'], res['above_n']) == (5, 2)
    assert res['components']['LW']['direction'] == 'neither'
    for fam, regs in A.FAMILIES.items():
        x = res['pooled'][fam]
        want = float(np.mean([C[r] for r in regs]))
        d3 = np.stack([np.stack([np.full(n, C[r] + E[s]) for s in SEEDS]) for r in regs], axis=2)
        hb = SF.hier_boot(d3, B=B, seed=11)
        assert x['mean'] == pytest.approx(want, abs=1e-9) and x['c'] == len(regs)
        assert (x['lo'], x['hi']) == pytest.approx((hb['lo'], hb['hi']), abs=1e-9)
        assert res['per_seed'][fam] == pytest.approx({f's{s}': want + E[s] for s in SEEDS}, abs=1e-9)
    assert res['pooled']['Setup']['combinations'] == ['S', 'LS', 'SW', 'LSW']
    assert res['pooled']['NoSetup']['combinations'] == ['N', 'L', 'W', 'LW']
    assert res['pooled']['All']['mean'] == pytest.approx(-0.6, abs=1e-9)
    # gaps to the references: hand-computed means, the LSW reference of instance 0 at 1.6 base
    base = bases(root)
    for arm in A.ARMS:
        cells = []
        for r in R8:
            ref = ref_value(r, base).copy()
            if r == 'LSW':
                ref[0] = 1.6 * base[0]
            g = np.stack([100 * (pol_of(root, s, r) * ((1 + (C[r] + E[s]) / 100) if arm == 'active' else 1)
                                 - ref) / ref for s in SEEDS])
            cells.append(g)
            assert res['references']['arms'][arm]['components'][r]['mean'] == pytest.approx(g.mean(), abs=1e-9)
        assert res['references']['arms'][arm]['pooled']['mean'] == pytest.approx(
            np.mean(np.stack(cells, axis=2)), abs=1e-9)
    assert res['references']['counted'] == 8 * n
    # validation: build_tree's policy keeps 0.90 (0.89 at seed 303), the arm 0.88
    pol_best = {301: 0.90, 302: 0.90, 303: 0.89}
    want = float(np.mean([100 * (0.88 - pol_best[s]) / pol_best[s] for s in SEEDS]))
    for r in P.TRAIN:
        assert res['validation'][r]['mean'] == pytest.approx(want, abs=1e-6)
    assert res['validation']['score']['mean'] == pytest.approx(want, abs=1e-6)
    assert res['validation']['kept_update']['active'] == [75, 75, 75]
    assert res['config_differs_in'] == {f's{s}': ['mask_mode', 'model_suffix'] for s in SEEDS}
    assert res['files_read'] == 2 * 3 * 8
    # outputs: macro names, one comment per macro line, no absolute path
    js, md, mac = (p.read_text() for p in outputs(out))
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', mac)
    assert names and len(names) == len(set(names)) == res['n_macros']
    assert all(re.fullmatch(r'PubAct[A-Za-z]+', x) for x in names)
    assert all('  % ' in ln for ln in mac.splitlines() if '\\newcommand' in ln)
    for text in (js, md, mac):
        assert str(tmp_path) not in text and ROOT not in text and '/home/' not in text and '/tmp/' not in text
    for macro, val in (('PubActDiffN', '-2.00'), ('PubActDiffLSW', '-1.50'), ('PubActDiffNDir', 'ahead'),
                       ('PubActDiffSWDir', 'behind'), ('PubActDiffLWDir', 'neither'),
                       ('PubActDiffPoolAll', '-0.60'), ('PubActDiffPoolSetup', '-0.75'),
                       ('PubActDiffPoolNoSetup', '-0.45'), ('PubActDiffPoolHeld', '+0.25'),
                       ('PubActDiffPoolTrain', '-0.88'),
                       ('PubActBelowN', '5'), ('PubActAboveN', '2'), ('PubActSeedOne', '-1.00'),
                       ('PubActSeedTwo', '-0.60'), ('PubActSeedThree', '-0.20'),
                       ('PubActInstN', str(n)), ('PubActSeedN', '3'), ('PubActBoot', str(B))):
        assert f'\\newcommand{{\\{macro}}}{{{val}}}' in mac, macro
    for stem in ('PubActRefActN', 'PubActRefPolLSW', 'PubActRefActPool', 'PubActRefPolPool',
                 'PubActValiN', 'PubActValiLW', 'PubActValiScore', 'PubActValiScoreAct',
                 'PubActDiffPoolTrainDir'):
        assert stem in names, stem
    assert '## 1. Paired difference per combination' in md and '## 3. Gap to the references' in md
    assert json.loads(js)['generator'] == A.GENERATOR


def test_same_runs_give_zero_difference(tmp_path):
    """An arm identical to the policy gives d = 0 exactly on every instance."""
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    write_comparator(root)
    write_act(root, tweak=lambda s, r, v: pol_of(root, s, r))
    res = run(root, tmp_path / 'out')
    for r in R8:
        c = res['components'][r]
        assert (c['mean'], c['lo'], c['hi'], c['direction'], c['equal']) == (0.0, 0.0, 0.0, 'neither', 100.0)
    a, p = res['references']['arms']['active'], res['references']['arms']['policy']
    assert (a['pooled']['mean'], a['pooled']['lo'], a['pooled']['hi']) == \
        (p['pooled']['mean'], p['pooled']['lo'], p['pooled']['hi'])
    assert res['below_n'] == res['above_n'] == 0


def test_absent_arm_refuses_and_plan_lists_every_file(tmp_path, capsys):
    root = T.build_tree(str(tmp_path / 'tree'))
    write_refs(root)
    write_comparator(root)
    x = A.main(['plan', '--root', root])
    assert x['arm']['state'] == 'none' and len(x['arm']['missing']) == 39 and not x['ready']
    assert not x['comparator_missing'] and x['references']['complete']
    text = capsys.readouterr().out
    assert 'arm: none, 0 of 39 files' in text and text.count('  missing ') == 39
    assert 'infer: not ready' in text
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'REFUSING' in str(e.value) and 'no file of pubx-act exists' in str(e.value)
    assert not out.exists() or not any(p.exists() for p in outputs(out))


@pytest.mark.parametrize('damage', ['missing array', 'missing summary', 'unfinished run', 'fourth seed',
                                    'checkpoint only', 'config drift', 'wrong rule', 'no banner',
                                    'comparator file', 'array older than checkpoint',
                                    'below a reference bound', 'reference missing', 'other norms'])
def test_partial_or_wrong_arm_refuses(tmp_path, damage):
    out = tmp_path / 'out'
    for p in outputs(out):                      # stale outputs of an earlier run
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('stale')
    msg = 'incomplete'
    if damage == 'checkpoint only':             # a run in training, nothing evaluated yet
        root = T.build_tree(str(tmp_path / 'tree'))
        write_refs(root)
        write_comparator(root)
        open(os.path.join(root, REG.pub_act_paths(302)[0]), 'w').close()
    elif damage == 'below a reference bound':   # SW of instance 1 below its counted bound
        def tweak(s, r, v):
            if (s, r) == (302, 'SW'):
                v[1] = 1.1 * bases(str(tmp_path / 'tree'))[1]
            return v
        root = complete_tree(tmp_path, tweak=tweak)
        msg = 'below the reference bound'
    else:
        root = complete_tree(tmp_path)
        hp = os.path.join(root, REG.PUB_RESULTS, ST + P.XS)
        if damage == 'missing array':
            os.remove(os.path.join(hp, f'Result_greedy+{REG.PUB_PREFIX}pubx-act-s302_LS.npy'))
        elif damage == 'missing summary':
            os.remove(os.path.join(hp, f'summary_{REG.PUB_PREFIX}pubx-act-s303{P.XS}.md'))
        elif damage == 'unfinished run':
            lp = os.path.join(root, REG.pub_act_paths(301)[1])
            edit(lp, '[train_family] done in', '[train_family] at')
        elif damage == 'fourth seed':
            open(os.path.join(root, 'trained_network', 'FAMILY', f'{REG.PUB_PREFIX}pubx-act-s304.pth'),
                 'w').close()
        elif damage in ('config drift', 'wrong rule'):
            cp = os.path.join(root, REG.pub_act_paths(303)[3])
            c = json.load(open(cp))
            json.dump(dict(c, **({'lr': 0.001} if damage == 'config drift' else {'mask_mode': 'nondelay'})),
                      open(cp, 'w'))
            msg = 'is not the recipe' if damage == 'config drift' else "mask_mode 'nondelay'"
        elif damage == 'no banner':
            lp = os.path.join(root, REG.pub_act_paths(302)[1])
            edit(lp, A.BANNER + '\n', '')
            msg = 'does not state the active mask'
        elif damage == 'comparator file':
            os.remove(os.path.join(root, A.run_paths('policy', 303)[3]))
            msg = 'comparator pubx-v1 is incomplete'
        elif damage == 'array older than checkpoint':
            p = os.path.join(hp, f'Result_greedy+{REG.PUB_PREFIX}pubx-act-s301_W.npy')
            os.utime(p, (OLD - 10, OLD - 10))
            msg = 'older than the checkpoint'
        elif damage == 'reference missing':
            folder = os.path.join(root, P.Inputs(root).ref_folder())
            os.remove(os.path.join(folder, 'LW', 'inst02.json'))
            msg = 'references'
        elif damage == 'other norms':
            lp = os.path.join(root, REG.pub_act_paths(302)[1])
            edit(lp, '[vali] MWKR norms: N=500.0', '[vali] MWKR norms: N=501.0')
            msg = 'not on one scale'
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'REFUSING' in str(e.value) and msg in str(e.value), str(e.value)
    assert not any(p.exists() for p in outputs(out))


def test_macro_collision_refused(tmp_path):
    root = complete_tree(tmp_path)
    os.makedirs(os.path.join(root, 'manuscript'), exist_ok=True)
    with open(os.path.join(root, 'manuscript', 'macros_other.tex'), 'w') as f:
        f.write('\\newcommand{\\PubActDiffPoolAll}{1}\n')
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        run(root, out)
    assert 'collision with manuscript/macros_other.tex' in str(e.value) and 'PubActDiffPoolAll' in str(e.value)
    assert not any(p.exists() for p in outputs(out))
    # a file of the output's own name is an earlier output of this script, not a collision
    os.remove(os.path.join(root, 'manuscript', 'macros_other.tex'))
    with open(os.path.join(root, 'manuscript', 'macros_pubact.tex'), 'w') as f:
        f.write('\\newcommand{\\PubActDiffPoolAll}{1}\n')
    assert run(root, out)['n_macros'] > 0
