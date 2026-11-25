"""The public family under the active rule (scripts/run_pubfamily_queue.sh ARMS=actfam and
ARMS=acteq): the names of its runs in the registry, the queue's task lists and their order,
the protection of the finished runs pubx-act-s301..303, the configuration check against
the non-delay counterpart, the queue's probes and runs with a stand-in trainer on a
synthetic tree, and the globs and name patterns of p89, p108 and the evaluation plan, which
take none of the new control, specialist and eq runs.

Run from the repository root:  python -m pytest tests/test_pubactfam_arm.py -q
CPU only, no training: the queue runs in its listing mode on the repository, and in full
only on a synthetic tree under a temporary folder, where an interpreter stand-in answers
for train_family.py and passes every other call to the real interpreter. The planner test
reads data/PUBLIC/family and is skipped without it.
"""
import fnmatch
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p108_public_active_arm as A  # noqa: E402
import p89_public_family as P  # noqa: E402
from test_pubeq_arm import default_names, listed, queue  # noqa: E402

QUEUE = os.path.join(ROOT, 'scripts', 'run_pubfamily_queue.sh')
R8 = list(REG.COMBOS)
SIX = 'N,L,S,W,LS,LW'
SEEDS = (301, 302, 303, 304, 305)
ACT_SEEDS = (301, 302, 303)
# recipe -> (registry arm, combination, regimes, updates)
RECIPES = [('v1', 'released', None, SIX, 2000), ('ctl', 'control', None, 'N,L,S,W', 2000)] + \
          [(f'solo-{r}', 'specialist', r, r, 1000) for r in R8]


def family_names():
    """The task list of ARMS=actfam in the order the queue builds it."""
    out = []
    for tier in ((301, 302, 303), (304, 305)):
        out += [REG.pub_rule_model('released', None, s) for s in tier]
        out += [REG.pub_rule_model('control', None, s) for s in tier]
        out += [REG.pub_rule_model('specialist', r, s) for s in tier for r in R8]
    return out


def finished_eq_seeds(root=ROOT):
    """Seeds whose non-delay run pubx-eq-s<seed> has a checkpoint and a completed log."""
    out = []
    for s in SEEDS:
        ck, lp = REG.pub_eq_paths(s)[:2]
        if os.path.exists(os.path.join(root, ck)) and os.path.exists(os.path.join(root, lp)) \
                and 'done in' in open(os.path.join(root, lp), errors='replace').read():
            out.append(s)
    return out


# ------------------------------------------------------------------ names
def test_registry_names_and_paths():
    assert REG.PUB_RULE_STEM == {'released': 'pubx-{W}', 'control': 'pubx-ctl{W}',
                                 'specialist': 'pubx-solo{W}-{R}', 'eq': 'pubx-eq{W}'}
    assert REG.PUB_ACTFAM_SEEDS == SEEDS == REG.PUB_SEEDS
    assert REG.pub_rule_model('control', None, 302) == 'pubx-ctlact-s302'
    assert REG.pub_rule_model('specialist', 'LS', 301) == 'pubx-soloact-LS-s301'
    assert REG.pub_rule_model('eq', None, 303) == 'pubx-eqact-s303'
    assert REG.pub_rule_model('released', None, 304) == 'pubx-act-s304'
    assert REG.pub_rule_model('control', None, 305, word='act') == 'pubx-ctlact-s305'
    assert REG.pub_rule_paths('specialist', 'LSW', 305) == (
        'trained_network/FAMILY/10x5+pubfamily+pubx-soloact-LSW-s305.pth',
        'train_log/PUBFAMILY_pubx-soloact-LSW-s305.log',
        'train_log/FAMILY/valiquality_10x5+pubfamily+pubx-soloact-LSW-s305.txt',
        'train_log/FAMILY/config_10x5+pubfamily+pubx-soloact-LSW-s305.json')
    # the policy of seeds 301-303 is the active-rule arm, whose helpers keep their seeds
    for s in ACT_SEEDS:
        assert REG.pub_rule_model('released', None, s) == REG.pub_act_model(s)
        assert REG.pub_rule_paths('released', None, s) == REG.pub_act_paths(s)
    assert REG.PUB_ACT_SEEDS == ACT_SEEDS and (A.SEEDS, A.STEM) == (ACT_SEEDS, 'pubx-act')
    for bad in (304, 305):
        with pytest.raises(ValueError):
            REG.pub_act_model(bad)
    # counterparts: the principal run of the same arm, combination and seed, or pubx-eq
    assert REG.pub_rule_counterpart('released', None, 302) == 'pubx-v1-s302'
    assert REG.pub_rule_counterpart('control', None, 304) == 'pubx-ctl-s304'
    assert REG.pub_rule_counterpart('specialist', 'SW', 305) == 'pubx-solo-SW-s305'
    assert REG.pub_rule_counterpart('eq', None, 301) == 'pubx-eq-s301'
    for args, err in ((('control', None, 300), ValueError), (('control', None, 306), ValueError),
                      (('control', None, None), ValueError), (('specialist', None, 301), ValueError),
                      (('specialist', 'X', 301), ValueError), (('control', 'N', 301), ValueError),
                      (('v1', None, 301), REG.NotInEnvironment)):
        with pytest.raises(err):
            REG.pub_rule_model(*args)
    with pytest.raises(ValueError):
        REG.pub_rule_model('control', None, 301, word='active')
    # every name is new except the three runs of the active-rule arm, and none is principal
    new = [REG.pub_rule_model(a, c, s) for a in REG.PUB_RULE_STEM
           for c in (R8 if a == 'specialist' else [None]) for s in SEEDS]
    assert len(new) == len(set(new)) == 5 * 11
    old = {REG.pub_model(a, c, s) for a in REG.ARMS for c in (R8 if a == 'specialist' else [None])
           for s in SEEDS} | {REG.pub_eq_model(s) for s in SEEDS} | {REG.pub_act_model(s) for s in ACT_SEEDS}
    assert set(new) & old == {REG.pub_act_model(s) for s in ACT_SEEDS}
    assert REG.PUB_STEM == {'released': 'pubx-v1', 'control': 'pubx-ctl', 'specialist': 'pubx-solo-{R}'}
    with pytest.raises(Exception):
        REG.pub_model('eq', None, 301)


# ------------------------------------------------------------------ task lists
def test_task_lists_their_order_and_the_unchanged_lists():
    names = family_names()
    assert listed(queue(LIST_ONLY='all', ARMS='actfam')) == names
    assert len(names) == 50 and len(set(names)) == 50
    # seeds 301-303 first: their three policy runs are the finished active-rule runs, so the
    # 27 runs to train of a complete three-seed family come before the 20 of seeds 304-305
    first, rest = names[:30], names[30:]
    assert all(n.endswith(('-s301', '-s302', '-s303')) for n in first)
    assert all(n.endswith(('-s304', '-s305')) for n in rest) and len(rest) == 20
    assert first[:3] == [REG.pub_act_model(s) for s in ACT_SEEDS]
    assert rest[:4] == ['pubx-act-s304', 'pubx-act-s305', 'pubx-ctlact-s304', 'pubx-ctlact-s305']
    lanes = [listed(queue(LIST_ONLY='all', ARMS='actfam', LANE=f'{i}/8')) for i in range(8)]
    assert lanes == [names[i::8] for i in range(8)]
    assert listed(queue(LIST_ONLY='all', ARMS='actfam', ONLY='soloact-LS-')) == \
        [n for n in names if 'soloact-LS-' in n]
    # the equal-experience recipe: the seeds of the finished pubx-eq runs on disk
    eq_seeds = finished_eq_seeds()
    assert eq_seeds, 'no finished pubx-eq run on disk'
    assert listed(queue(LIST_ONLY='all', ARMS='acteq')) == [REG.pub_rule_model('eq', None, s) for s in eq_seeds]
    # the default, eq and act lists are unchanged and hold no run of the new arms
    assert listed(queue(LIST_ONLY='all')) == default_names('pubx')
    assert listed(queue(LIST_ONLY='all', MASK='legacy')) == default_names('publ')
    assert listed(queue(LIST_ONLY='all', TYPES='class')) == default_names('pubc')
    assert listed(queue(LIST_ONLY='all', ARMS='eq')) == [f'pubx-eq-s{s}' for s in SEEDS]
    assert listed(queue(LIST_ONLY='all', ARMS='act')) == [f'pubx-act-s{s}' for s in ACT_SEEDS]
    assert not set(names[3:]) & set(default_names('pubx'))
    for env, msg in ((dict(ARMS='actfam', MASK='legacy'), 'ARMS=actfam is defined for MASK=exact TYPES=none only'),
                     (dict(ARMS='acteq', TYPES='class'), 'ARMS=acteq is defined for MASK=exact TYPES=none only'),
                     (dict(ARMS='fam'), 'ARMS must be empty'),
                     (dict(ARMS='actfamx'), 'ARMS must be empty'),
                     (dict(ARMS='activefam'), 'ARMS must be empty')):
        p = queue(LIST_ONLY='all', **env)
        assert p.returncode == 1 and msg in p.stdout, (env, p.stdout)


def test_finished_active_runs_are_listed_and_never_tasks_to_train():
    p = queue(LIST_ONLY='1', ARMS='actfam')
    todo = listed(p)
    protected = [REG.pub_act_model(s) for s in ACT_SEEDS]
    assert not set(todo) & set(protected)
    assert set(todo) <= set(family_names()[3:])
    for n in protected:
        assert f'SKIP {n} (finished; a run of ARMS=act, never trained by ARMS=actfam)' in p.stderr
    # every name the queue would train is trained at most once over the eight lanes
    lanes = [listed(queue(LIST_ONLY='1', ARMS='actfam', LANE=f'{i}/8')) for i in range(8)]
    assert sorted(sum(lanes, [])) == sorted(todo)


# ------------------------------------------------------------------ configuration check
def fam_check(tmp_path, cfg, name, ref, regs, recipe_upd, want_upd, kind, seeds='', rule='active'):
    """Run the queue's fam_check function alone, in tmp_path, on its stored files."""
    fn = re.search(r'^fam_check\(\) \{.*?^PYEOF\n\}\n', open(QUEUE).read(), re.S | re.M).group(0)
    script = (f'PY={sys.executable}; RULE={rule}; RWORD=act\n{fn}\n'
              f'cd {tmp_path} && fam_check {cfg} {name} {ref} {regs} {recipe_upd} {want_upd} {kind} '
              f'"{seeds}" 29602\n')
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True, timeout=60)


BASE = dict(device='cpu', first_mask='exact', mask_ties='exact', family_base='sd2',
            use_type_embedding=False, lr=0.0003, n_m=5,
            pub_vali_dir='data/PUBLIC/family/pub_SD2_vali_10x5+mix')


def stored_cfg(name, regs, upd, seed, rule='nondelay', **extra):
    return dict(BASE, model_suffix=name, family_regimes=regs, max_updates=upd, seed_train=seed,
                mask_mode=rule, **extra)


def write_counterparts(d):
    for stem, regs, upd in (('pubx-ctl', 'N,L,S,W', 2000), ('pubx-solo-LS', 'LS', 1000),
                            ('pubx-eq', SIX, 6000)):
        for s in SEEDS:
            # the seed-301 run predates the reward_mode key, as the stored pubx-v1-s301
            extra = {} if s == 301 else {'reward_mode': 'dense'}
            json.dump(stored_cfg(f'{stem}-s{s}', regs, upd, s, **extra),
                      open(d / f'config_10x5+pubfamily+{stem}-s{s}.json', 'w'))


@pytest.mark.parametrize('recipe', ['ctl', 'solo-LS', 'eq'])
def test_config_check_accepts_the_name_and_the_rule_only(tmp_path, recipe):
    d = tmp_path / 'train_log' / 'FAMILY'
    d.mkdir(parents=True)
    write_counterparts(d)
    stem, regs, upd, name = {'ctl': ('pubx-ctl', 'N,L,S,W', 2000, 'pubx-ctlact-s302'),
                             'solo-LS': ('pubx-solo-LS', 'LS', 1000, 'pubx-soloact-LS-s302'),
                             'eq': ('pubx-eq', SIX, 6000, 'pubx-eqact-s302')}[recipe]
    good = stored_cfg(name, regs, upd, 302, rule='active', reward_mode='dense', pub_type_source='none')
    json.dump(good, open(tmp_path / 'run.json', 'w'))
    p = fam_check(tmp_path, 'run.json', name, f'{stem}-s302', regs, upd, upd, 'run')
    assert p.returncode == 0 and "differs only in ['mask_mode', 'model_suffix']" in p.stdout, p.stderr
    probe = stored_cfg('_pubprobe_x', regs, 1, 301, rule='active', pub_type_source='none')
    json.dump(probe, open(tmp_path / 'probe.json', 'w'))
    p = fam_check(tmp_path, 'probe.json', '_pubprobe_x', f'{stem}-s301', regs, upd, 1, 'probe', '301 302 303 304 305')
    assert p.returncode == 0 and "differs only in ['mask_mode', 'max_updates', 'model_suffix']" in p.stdout \
        and f'{stem}-s301/302/303/304/305 are one recipe' in p.stdout, p.stderr
    for change, kind, want, msg in ((dict(lr=0.001), 'run', upd, 'configuration drift'),
                                    (dict(mask_mode='nondelay'), 'run', upd, "mask_mode='nondelay'"),
                                    (dict(max_updates=upd + 1), 'run', upd, f'max_updates={upd + 1}'),
                                    (dict(max_updates=upd + 1), 'run', upd + 1, 'configuration drift'),
                                    (dict(seed_train=303), 'run', upd, 'configuration drift'),
                                    (dict(family_regimes='N'), 'run', upd, 'configuration drift'),
                                    (dict(pub_type_source='class'), 'run', upd, 'configuration drift'),
                                    (dict(pair_extra='sw'), 'run', upd, 'configuration drift'),
                                    (dict(model_suffix='pubx-other'), 'run', upd, "model_suffix='pubx-other'"),
                                    (dict(max_updates=upd), 'probe', 1, f'max_updates={upd}')):
        json.dump(dict(good, **change), open(tmp_path / 'bad.json', 'w'))
        p = fam_check(tmp_path, 'bad.json', name, f'{stem}-s302', regs, upd, want, kind, '302')
        assert p.returncode != 0 and msg in (p.stdout + p.stderr), change
    # the counterpart must be the recipe: its regimes, its updates, the non-delay rule
    for regs_x, upd_x, msg in ((SIX if recipe != 'eq' else 'N', upd, 'family_regimes'),
                               (regs, upd + 1, 'max_updates')):
        p = fam_check(tmp_path, 'run.json', name, f'{stem}-s302', regs_x, upd_x, upd_x, 'run')
        assert p.returncode != 0 and f'the counterpart {stem}-s302 records {msg}' in p.stderr
    c = json.load(open(d / f'config_10x5+pubfamily+{stem}-s302.json'))
    json.dump(dict(c, mask_mode='active'), open(d / f'config_10x5+pubfamily+{stem}-s302.json', 'w'))
    p = fam_check(tmp_path, 'run.json', name, f'{stem}-s302', regs, upd, upd, 'run')
    assert p.returncode != 0 and 'want the non-delay rule' in p.stderr
    json.dump(c, open(d / f'config_10x5+pubfamily+{stem}-s302.json', 'w'))
    # counterparts that are not one recipe refuse the probe; a seed outside the list does not
    c = json.load(open(d / f'config_10x5+pubfamily+{stem}-s304.json'))
    json.dump(dict(c, lr=0.001), open(d / f'config_10x5+pubfamily+{stem}-s304.json', 'w'))
    p = fam_check(tmp_path, 'probe.json', '_pubprobe_x', f'{stem}-s301', regs, upd, 1, 'probe', '301 302 303 304 305')
    assert p.returncode != 0 and f'{stem}-s304 differs from {stem}-s301' in p.stderr
    assert fam_check(tmp_path, 'probe.json', '_pubprobe_x', f'{stem}-s301', regs, upd, 1, 'probe',
                     '301 302 303').returncode == 0
    # a rule word that names the non-delay rule is no arm
    p = fam_check(tmp_path, 'run.json', name, f'{stem}-s302', regs, upd, upd, 'run', rule='nondelay')
    assert p.returncode != 0 and 'want a rule other than nondelay' in p.stderr


# ------------------------------------------------------------------ the queue on a synthetic tree
FAKE_TRAINER = r'''
import json, os, sys
a = sys.argv[1:]


def opt(k, d=None):
    return a[a.index(k) + 1] if k in a else d


name, regs = opt('--model_suffix'), opt('--family_regimes')
upd, seed, rule = int(opt('--max_updates')), int(opt('--seed_train')), opt('--mask_mode', 'nondelay')
with open('trainer_calls.txt', 'a') as f:
    f.write(' '.join(a) + '\n')
cfg = dict(device=opt('--device'), first_mask=opt('--first_mask'), mask_ties=opt('--mask_ties'),
           family_base=opt('--family_base'), use_type_embedding=opt('--use_type_embedding') == 'True',
           lr=0.0003, n_m=int(opt('--n_m')), pub_vali_dir=opt('--pub_vali_dir'), model_suffix=name,
           family_regimes=regs, max_updates=upd, seed_train=seed, mask_mode=rule,
           reward_mode='dense', pub_type_source='none')
if os.environ.get('FAKE_DRIFT') == name:
    cfg['lr'] = 0.001
with open(f'train_log/FAMILY/config_10x5+pubfamily+{name}.json', 'w') as f:
    json.dump(cfg, f)
if rule != 'nondelay':
    print(f'[train_family] OPTIONS: mask_mode={rule} pair_extra=none')
print(f"[train_family] MASK: first_mask={cfg['first_mask']} mask_ties={cfg['mask_ties']}")
print('[model] params total=29602  film=0 (0.00%)')
with open(f'train_log/FAMILY/reward_10x5+pubfamily+{name}.txt', 'w') as f:
    f.write('[]')
if upd > 1:
    open(f'trained_network/FAMILY/10x5+pubfamily+{name}.pth', 'w').close()
    print('[train_family] done in 0.00 h; best rel=0.9000')
'''


def write_run(root, name, regs, upd, seed, rule='nondelay', done=True):
    """A stored run: configuration, training log and checkpoint."""
    json.dump(stored_cfg(name, regs, upd, seed, rule=rule, **({} if seed == 301 else {'reward_mode': 'dense'})),
              open(os.path.join(root, 'train_log', 'FAMILY', f'config_10x5+pubfamily+{name}.json'), 'w'))
    lines = ['[train_family] MASK: first_mask=exact mask_ties=exact', '[model] params total=29602  film=0 (0.00%)']
    if rule != 'nondelay':
        lines.insert(0, f'[train_family] OPTIONS: mask_mode={rule} pair_extra=none')
    if done:
        lines.append('[train_family] done in 0.50 h; best rel=0.9000')
    open(os.path.join(root, 'train_log', f'PUBFAMILY_{name}.log'), 'w').write('\n'.join(lines) + '\n')
    open(os.path.join(root, 'trained_network', 'FAMILY', f'10x5+pubfamily+{name}.pth'), 'w').write(name)


@pytest.fixture
def tree(tmp_path):
    """A copy of the queue in a tree holding the finished non-delay runs of the family, the
    finished pubx-eq runs and pubx-act-s301..303; the interpreter of the queue is a stand-in
    that answers for train_family.py (writes the configuration, the log lines and, after
    more than one update, a checkpoint) and runs everything else in the real interpreter."""
    root = tmp_path / 'repo'
    (root / 'scripts').mkdir(parents=True)
    shutil.copy(QUEUE, root / 'scripts' / 'run_pubfamily_queue.sh')
    for d in ('train_log/FAMILY', 'trained_network/FAMILY'):
        (root / d).mkdir(parents=True)
    home = tmp_path / 'home'
    py = home / 'miniconda3' / 'envs' / 'fjsp' / 'bin' / 'python'
    py.parent.mkdir(parents=True)
    trainer = tmp_path / 'fake_trainer.py'
    trainer.write_text(FAKE_TRAINER)
    py.write_text('#!/bin/bash\nif [ "${1:-}" = train_family.py ]; then shift; '
                  f'exec {sys.executable} -B {trainer} "$@"; fi\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    for _, arm, combo, regs, upd in RECIPES:
        for s in SEEDS:
            write_run(str(root), REG.pub_model(arm, combo, s), regs, upd, s)
    for s in SEEDS:
        write_run(str(root), REG.pub_eq_model(s), SIX, 6000, s)
    for s in ACT_SEEDS:
        write_run(str(root), REG.pub_act_model(s), SIX, 2000, s, rule='active')
    return root, home


def run_tree(tree, **env):
    root, home = tree
    e = {k: v for k, v in os.environ.items() if k not in
         ('MASK', 'TYPES', 'ARMS', 'LANE', 'ONLY', 'LIST_ONLY', 'PROBE_ONLY', 'FAKE_DRIFT')}
    e.update(HOME=str(home), CORES='3', THREADS='1', PYTHONDONTWRITEBYTECODE='1', **env)
    return subprocess.run(['bash', str(root / 'scripts' / 'run_pubfamily_queue.sh')], env=e,
                          capture_output=True, text=True, timeout=300)


def calls(root):
    p = root / 'trainer_calls.txt'
    return p.read_text().splitlines() if p.exists() else []


def opt(call, k):
    a = call.split()
    return a[a.index(k) + 1]


def stray_probe_files(root):
    return [f for d, _, fs in os.walk(root) for f in fs if '_pubprobe_' in f or 'pubfamily_probe_' in f]


def snapshot(root, names):
    out = {}
    for n in names:
        for p in (f'trained_network/FAMILY/10x5+pubfamily+{n}.pth', f'train_log/PUBFAMILY_{n}.log',
                  f'train_log/FAMILY/config_10x5+pubfamily+{n}.json'):
            st = os.stat(root / p)
            out[p] = (st.st_mtime_ns, (root / p).read_bytes())
    return out


def test_probes_build_every_recipe_once_and_leave_nothing(tree):
    root, _ = tree
    p = run_tree(tree, ARMS='actfam', PROBE_ONLY='1')
    assert p.returncode == 0, p.stdout + p.stderr
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith('[pubfamily] act ')]
    stems = ['pubx-ctl'] + [f'pubx-solo-{r}' for r in R8] + ['pubx-v1']
    assert [ln.split()[2].rstrip(':') for ln in lines] == stems
    for ln, st in zip(lines, stems):
        assert f'29602 parameters as {st}-s301, exact mask, mask_mode active' in ln
        assert "differs only in ['mask_mode', 'max_updates', 'model_suffix']" in ln
        assert f'{st}-s301/302/303/304/305 are one recipe under the non-delay rule' in ln
    c = calls(root)
    assert len(c) == 10 and all(opt(x, '--max_updates') == '1' and opt(x, '--seed_train') == '301'
                                and opt(x, '--mask_mode') == 'active' for x in c)
    assert sorted(opt(x, '--family_regimes') for x in c) == sorted([SIX, 'N,L,S,W'] + R8)
    assert 'PROBE_ONLY=1: not starting the queue' in p.stdout and not stray_probe_files(root)
    # the eq recipe: one probe against pubx-eq-s301
    p = run_tree(tree, ARMS='acteq', PROBE_ONLY='1')
    assert p.returncode == 0 and '[pubfamily] act pubx-eq: 29602 parameters as pubx-eq-s301' in p.stdout
    assert len(calls(root)) == 11 and not stray_probe_files(root)
    # a counterpart with another parameter count stops the probe
    lp = root / 'train_log' / 'PUBFAMILY_pubx-solo-W-s301.log'
    lp.write_text(lp.read_text().replace('params total=29602', 'params total=32786'))
    p = run_tree(tree, ARMS='actfam', PROBE_ONLY='1', ONLY='soloact-W-')
    assert p.returncode == 1 and '32786 in pubx-solo-W-s301' in p.stdout


def test_runs_train_only_new_names_and_check_each_against_its_counterpart(tree):
    root, _ = tree
    write_run(str(root), 'pubx-ctlact-s301', 'N,L,S,W', 2000, 301, rule='active')    # finished before
    protected = [REG.pub_act_model(s) for s in ACT_SEEDS]
    before = snapshot(root, protected + ['pubx-ctlact-s301'])
    p = run_tree(tree, ARMS='actfam', ONLY=r'^pubx-(act-s30[15]|ctlact-s301|soloact-LS-s302)$')
    assert p.returncode == 0, p.stdout + p.stderr
    out = p.stdout
    assert 'SKIP pubx-act-s301 (a run of ARMS=act, never trained by ARMS=actfam)' in out
    assert 'SKIP pubx-ctlact-s301 (finished)' in out
    assert 'TRAIN pubx-act-s305 (regimes N,L,S,W,LS,LW, 2000 updates, seed 305, rule active, ' \
           'counterpart pubx-v1-s305)' in out
    assert 'TRAIN pubx-soloact-LS-s302 (regimes LS, 1000 updates, seed 302, rule active, ' \
           'counterpart pubx-solo-LS-s302)' in out
    assert "[pubfamily] pubx-soloact-LS-s302: configuration against pubx-solo-LS-s302 differs only in " \
           "['mask_mode', 'model_suffix'] (mask_mode active)" in out
    assert 'trained pubx-act-s305' in out and 'queue finished' in out
    runs = [x for x in calls(root) if opt(x, '--max_updates') != '1']
    assert [(opt(x, '--model_suffix'), opt(x, '--family_regimes'), opt(x, '--max_updates'),
             opt(x, '--seed_train'), opt(x, '--mask_mode')) for x in runs] == [
        ('pubx-soloact-LS-s302', 'LS', '1000', '302', 'active'),
        ('pubx-act-s305', SIX, '2000', '305', 'active')]
    # the probes of this lane: the recipes of its tasks, the runs of the rule arm left out
    probes = [x for x in calls(root) if opt(x, '--max_updates') == '1']
    assert sorted(opt(x, '--family_regimes') for x in probes) == sorted(['N,L,S,W', 'LS', SIX])
    assert not any(n in x for x in calls(root) for n in protected)
    assert snapshot(root, protected + ['pubx-ctlact-s301']) == before
    # a second pass trains nothing
    n = len(calls(root))
    p = run_tree(tree, ARMS='actfam', ONLY=r'^pubx-(act-s30[15]|ctlact-s301|soloact-LS-s302)$')
    assert p.returncode == 0 and 'TRAIN ' not in p.stdout and 'SKIP pubx-act-s305 (finished)' in p.stdout
    assert len([x for x in calls(root)[n:] if opt(x, '--max_updates') != '1']) == 0
    # the eq recipe under the rule: 6000 updates against pubx-eq of the same seed
    p = run_tree(tree, ARMS='acteq', ONLY='s303$')
    assert p.returncode == 0 and 'counterpart pubx-eq-s303' in p.stdout, p.stdout + p.stderr
    last = calls(root)[-1]
    assert (opt(last, '--model_suffix'), opt(last, '--max_updates'), opt(last, '--mask_mode')) == \
        ('pubx-eqact-s303', '6000', 'active')


def test_a_run_that_drifts_from_its_counterpart_stops_the_queue(tree):
    root, _ = tree
    p = run_tree(tree, ARMS='actfam', ONLY='soloact-N-s303$', FAKE_DRIFT='pubx-soloact-N-s303')
    assert p.returncode == 1
    assert 'ABORT pubx-soloact-N-s303: its configuration is not the pubx-solo-N-s303 recipe under the ' \
           'active rule' in p.stdout
    assert 'configuration drift against pubx-solo-N-s303' in p.stderr


def test_eq_seeds_come_from_the_stored_files_and_the_active_runs_must_be_finished(tree):
    root, _ = tree
    lp = root / 'train_log' / 'PUBFAMILY_pubx-eq-s302.log'
    lp.write_text(lp.read_text().replace('done in', 'at'))
    os.remove(root / 'trained_network' / 'FAMILY' / '10x5+pubfamily+pubx-eq-s304.pth')
    assert finished_eq_seeds(str(root)) == [301, 303, 305]
    assert listed(run_tree(tree, ARMS='acteq', LIST_ONLY='all')) == \
        ['pubx-eqact-s301', 'pubx-eqact-s303', 'pubx-eqact-s305']
    for s in (301, 303, 305):
        os.remove(root / 'trained_network' / 'FAMILY' / f'10x5+pubfamily+pubx-eq-s{s}.pth')
    p = run_tree(tree, ARMS='acteq', LIST_ONLY='all')
    assert p.returncode == 1 and 'no finished run pubx-eq-s<seed>' in p.stdout
    # the family arm refuses while a run of the rule arm is not finished
    lp = root / 'train_log' / 'PUBFAMILY_pubx-act-s302.log'
    lp.write_text(lp.read_text().replace('done in', 'at'))
    for mode in ('all', '1', '0'):
        p = run_tree(tree, ARMS='actfam', LIST_ONLY=mode, PROBE_ONLY='1')
        assert p.returncode == 1 and 'pubx-act-s302 is a run of ARMS=act and is not finished' in p.stdout
    assert calls(root) == []


def test_hold_file_stops_the_family_arm_before_a_run(tree):
    root, home = tree
    (root / 'train_log' / 'PUB_HOLD').write_text('')
    e = {k: v for k, v in os.environ.items() if k not in ('MASK', 'TYPES', 'ARMS', 'LANE', 'ONLY',
                                                          'LIST_ONLY', 'PROBE_ONLY', 'FAKE_DRIFT')}
    e.update(HOME=str(home), CORES='3', THREADS='1', ARMS='actfam', ONLY='ctlact-s302$')
    log = root / 'queue.out'
    with open(log, 'w') as f:
        q = subprocess.Popen(['bash', str(root / 'scripts' / 'run_pubfamily_queue.sh')], env=e,
                             stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        t0 = time.time()
        while 'all recipes verified' not in log.read_text() and time.time() - t0 < 60:
            time.sleep(0.2)
        time.sleep(2)
        assert 'all recipes verified' in log.read_text() and 'TRAIN ' not in log.read_text()
        assert q.poll() is None
        assert [opt(x, '--max_updates') for x in calls(root)] == ['1']
    finally:
        os.killpg(q.pid, signal.SIGKILL)
        q.wait()


# ------------------------------------------------------------------ patterns of other analyses
def new_runs():
    """Every name of the family under the active rule that is not a run of the rule arm."""
    return [REG.pub_rule_model(a, c, s) for a in REG.PUB_RULE_STEM
            for c in (R8 if a == 'specialist' else [None]) for s in SEEDS
            if not (a == 'released' and s in ACT_SEEDS)]


def files_of(name):
    """Training files and, in every public test and sensitivity folder, the arrays and the
    summary an evaluation of the run would write."""
    m = REG.PUB_PREFIX + name
    out = [f'trained_network/FAMILY/{m}.pth', f'train_log/PUBFAMILY_{name}.log',
           f'train_log/FAMILY/valiquality_{m}.txt', f'train_log/FAMILY/config_{m}.json']
    for st in list(REG.PUB_SETS) + list(REG.PUB_SENS_SETS):
        d = f'{REG.PUB_RESULTS}/{st}{P.XS}'
        out += [f'{d}/Result_greedy+{m}_{r}.npy' for r in R8] + [f'{d}/summary_{m}{P.XS}.md']
    return out


def test_no_pattern_of_another_analysis_takes_a_new_run(tmp_path):
    root = tmp_path / 'tree'
    names = new_runs()
    policy_new = ['pubx-act-s304', 'pubx-act-s305']
    others = [n for n in names if n not in policy_new]
    assert len(others) == 5 * 11 - 5
    for n in names:
        for f in files_of(n):
            os.makedirs(root / os.path.dirname(f), exist_ok=True)
            (root / f).write_text('')
    # p89, section 8: the glob pubx-eq-* of the equal-experience arm finds no file
    assert P.Inputs(str(root)).eq_state() is False
    # p108: its globs pubx-act-* take the policy runs of seeds 304 and 305, which carry the
    # name of the rule arm, and no control, specialist or eq run: the five-seed reading
    # (chosen once a file of seed 304 or 305 exists) expects exactly their files, the
    # three-seed reading neither reads them nor counts them as unexpected
    want = {f for n in policy_new for f in files_of(n)
            if f.startswith(('trained_network', 'train_log')) or f'/{A.HEADLINE}{P.XS}/' in f}
    assert len(want) == 2 * (4 + 9)
    try:
        assert A.choose_seeds(str(root), 'auto') == SEEDS
        st = A.arm_state(str(root))
        assert not st['unexpected'] and st['found'] == len(want)
        assert want <= set(A.expected('active'))
        assert A.choose_seeds(str(root), '301-303') == ACT_SEEDS
        st = A.arm_state(str(root))
        assert st['state'] == 'none' and st['found'] == 0 and not st['unexpected']
        for n in policy_new:
            for f in files_of(n):
                os.remove(root / f)
        assert A.choose_seeds(str(root), 'auto') == ACT_SEEDS
        st = A.arm_state(str(root))
        assert st['state'] == 'none' and st['found'] == 0
    finally:
        A.SEEDS = ACT_SEEDS
    # the evaluation plan's patterns of stored files read the factory family only
    import eval_newseeds_plan as EP
    for n in names:
        for f in files_of(n):
            assert not EP.RX.match(os.path.basename(f)) and not EP.RXS.match(os.path.basename(f))
    # the globs a reader of one non-delay arm or of the rule arm would use take no run of
    # another arm (a bare *act-s<seed>* would take pubx-ctlact-s<seed>; no script has one)
    for n in others:
        for pat in ['pubx-act-*', 'pubx-eq-*', 'pubx-v1-*', 'pubx-ctl-*', 'pubx-solo-*'] + \
                [f'*solo-{r}-s{s}*' for r in R8 for s in SEEDS] + \
                [f'*{w}-s{s}*' for w in ('ctl', 'eq', 'v1') for s in SEEDS]:
            assert not fnmatch.fnmatchcase(n, pat), (n, pat)
    for n in policy_new:
        assert fnmatch.fnmatchcase(n, 'pubx-act-*')


needs_public = pytest.mark.skipif(
    not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family')),
    reason='needs the public-family records')


@needs_public
def test_the_existing_evaluation_groups_plan_no_new_run():
    from test_pubact_arm import planner
    names = new_runs()
    with planner('public pubsens pubeq pubact') as EP:
        jobs = EP.jobs()
        expected = list(EP.EXPECTED)
    text = [str(j.get(k, '')) for j in jobs for k in ('id', 'model', 'ckpt', 'log')] + \
        [f for j in jobs for f in j['files']] + expected
    pat = re.compile('|'.join(re.escape(REG.PUB_PREFIX + n) + r'(?![\w-])' for n in names))
    assert not [t for t in text if pat.search(t)]
    assert not [t for t in text if re.search(r'pubx-(ctlact|soloact|eqact)', t)]
