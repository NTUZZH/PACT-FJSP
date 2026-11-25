"""scripts/eval_newseeds_plan.py, groups pubactfam, pubactsens and pubacteq: the public family
under the active mask evaluated exactly as its non-delay counterparts in groups public,
pubsens and pubeq.

Each job is the job of the counterpart run (arm_registry.pub_rule_counterpart) with the run
name, the checkpoint, the log and the group changed and nothing else; the jobs of group
pubact (pubx-act-s301..303 on SD2 10x5 and 20x10) are not planned again; the jobs of every
other group are unchanged; the files the completeness check expects are the files of the jobs.

Run from the repository root:  python -m pytest tests/test_pubactfam_eval_plan.py -q
CPU only; nothing is written.
"""
import contextlib
import importlib
import io
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family')),
                                reason='needs the public-family records')
NEW = ('pubactfam', 'pubactsens', 'pubacteq')
PLAN_ENV = ('EVAL_GROUPS', 'PUBLIC', 'PUBSENS', 'PUBEQ', 'PUBACT', 'PUBACTFAM', 'PUBACTSENS',
            'PUBACTEQ', 'LEGACY45', 'CROSS')


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


def name_of(j):
    return j['model'][len(REG.PUB_PREFIX):]


def counterpart_id(j):
    """The id of the counterpart job: the run name of the counterpart, the rest unchanged."""
    name = name_of(j)
    arm, combo, seed = parse(name)
    return j['id'][:-len(name)] + REG.pub_rule_counterpart(arm, combo, seed)


def parse(name):
    """(arm, combo, seed) of a run name of the family under the active mask."""
    for arm in ('released', 'control', 'eq'):
        for s in REG.PUB_ACTFAM_SEEDS:
            if name == REG.pub_rule_model(arm, None, s):
                return arm, None, s
    for combo in REG.COMBOS:
        for s in REG.PUB_ACTFAM_SEEDS:
            if name == REG.pub_rule_model('specialist', combo, s):
                return 'specialist', combo, s
    raise AssertionError(f'{name} is not a run of the family under the active mask')


def test_jobs_are_the_counterpart_jobs_renamed():
    with planner('public pubsens pubeq pubact') as EP:
        before = EP.jobs()
        pre_before = EP.preflights()
        actj = [j for j in before if j['group'] == 'pubact']
    with planner('public pubsens pubeq pubact ' + ' '.join(NEW)) as EP:
        after = EP.jobs()
        assert [j for j in after if j['group'] not in NEW] == before       # nothing else moves
        assert EP.preflights() == pre_before
        assert len({j['id'] for j in after}) == len(after)
        new = [j for j in after if j['group'] in NEW]
        by_group = {g: [j for j in new if j['group'] == g] for g in NEW}
        n_runs = len(EP.PUB_RUNS) * len(REG.PUB_ACTFAM_SEEDS)     # policy, control, 8 specialists
        assert len(by_group['pubactfam']) == len(REG.PUB_SETS) * n_runs - len(actj) == 294
        assert len(by_group['pubactsens']) == len(REG.PUB_SENS_SETS) * n_runs == 50
        assert len(by_group['pubacteq']) == len(EP.pub_eq_sets()) * len(REG.PUB_SEEDS) == 35
        rel = {j['id']: j for j in before}
        for j in new:
            k = rel[counterpart_id(j)]
            assert k['group'] == EP.PUB_ACTFAM_COUNTERPART[j['group']]
            diff = sorted(x for x in EP.KEYS_JOB if x != 'pre' and j[x] != k[x])
            assert diff == ['ckpt', 'group', 'id', 'log', 'model'], (j['id'], diff)
            assert j['files'] == [f.replace(k['model'], j['model']) for f in k['files']]
            arm, combo, s = parse(name_of(j))
            assert (j['ckpt'], j['log']) == REG.pub_rule_paths(arm, combo, s)[:2]
            assert EP.preflight_of(j) == 'pubdet' and j['rule_set'] == k['rule_set']
        # nothing of another arm: only runs of the family under the active mask, never a rule job
        assert all(j['model'] != 'rules' for j in new)
        # the jobs of group pubact are not planned again, and together with them the family
        # covers every job of the public counterpart once
        assert not {j['id'] for j in actj} & {j['id'] for j in new}
        assert not {f for j in actj for f in j['files']} & {f for j in new for f in j['files']}
        pol = {counterpart_id(j) for j in actj + by_group['pubactfam']}
        assert pol == {j['id'] for j in before if j['group'] == 'public' and j['model'] != 'rules'}
        for j in actj:
            assert j['campaign'].split('_', 1)[1] in EP.PUB_ACT_SETS


def test_check_expects_exactly_the_files_of_the_jobs():
    with planner(' '.join(NEW)) as EP:
        jobs = EP.jobs()
        assert {j['group'] for j in jobs} == set(NEW)
        planned = {f for j in jobs for f in j['files']}
        assert planned == set(EP.EXPECTED)
        assert [p['name'] for p in EP.preflights()] == ['xdet', 'xcross_policy', 'xcross_rules',
                                                         'xdet_rules', 'pubdet']
        assert all(REG.PUB_PREFIX + 'pubx-' in f for f in planned)
        assert not any(f'+{REG.PUB_PREFIX}pubx-v1-' in f or f'+{REG.PUB_PREFIX}pubx-ctl-' in f or
                       f'+{REG.PUB_PREFIX}pubx-solo-' in f or f'+{REG.PUB_PREFIX}pubx-eq-' in f
                       for f in planned)


def test_switches_add_the_groups():
    with planner('public') as EP:
        base = {j['id'] for j in EP.jobs()}
    with planner('public', PUBACTFAM='1', PUBACTSENS='1', PUBACTEQ='1') as EP:
        assert all(g in EP.GROUPS for g in NEW)
        got = EP.jobs()
        assert {j['id'] for j in got if j['group'] == 'public'} == base
        assert {j['group'] for j in got} == {'public', *NEW}
    with planner(' '.join(NEW)) as EP:
        assert sorted(j['id'] for j in EP.jobs()) == sorted(j['id'] for j in got if j['group'] in NEW)


def test_queue_documents_the_groups():
    q = open(os.path.join(ROOT, 'scripts', 'run_eval_newseeds.sh')).read()
    assert 'PUBACTFAM=1, PUBACTSENS=1, PUBACTEQ=1' in q
