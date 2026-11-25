"""Tests for scripts/p131_rc_trained_policy.py (the policy trained on reinforced-concrete
instances alone: evaluation, inference, contamination check), in the manner of
tests/test_p129_project_mix.py:

  * design: two levels, three sizes, eight combinations and seeds 301-303 give 144 result files
    of the arm and 6 control units, under the study's own folder, with the cell sizes of p129
    (50, 30, 30 instances on mix_rc; 100, 30, 30 on default);
  * the replay of the trainer's seed stream equals the global numpy stream and the trainer's
    own sample_batch; the contamination check refuses a planted seed collision and a planted
    equal instance, and passes on the real test sets;
  * on a hand-made case infer recovers the four comparisons (the arm against the best rules,
    against the policy, against the specialist, and the policy against the best rules on the
    same seeds), their verdicts and counts, and writes unique macro names, letters only with the
    prefix Rct, one comment per macro;
  * infer refuses a run that records another ppvc_mix, a configuration that differs from the
    policy's, a result written by a network with another parameter count or by another
    checkpoint, a missing and a short result file; evaluate refuses a schedule that fails the
    schedule check and writes nothing, and a run without its completion line or with another
    parameter count, and it verifies the runs of its part before it rolls out its units;
  * at the level default the stored arrays are read where p129 reads them, the control of the
    evaluation path compares bit for bit, and the rule means of the stored results equal p129's
    record;
  * plan partitions the units, prints one evaluate command per part and skips a finished unit;
  * no existing script reads the new folder, and the macro prefix is unused under manuscript/.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      -p no:cacheprovider tests/test_p131_rc_trained_policy.py
"""
import fnmatch
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from conftest import module_environment  # noqa: E402
MODULE_ENVIRONMENT = module_environment(PACT_OUT=None, PACT_RESULTS=None, RESULTS=None,
                                        SEEDS=None, PACT_FIRST_MASK=None, PACT_MASK_TOL=None,
                                        PACT_ENV='exact', PACT_SEEDS='301-305')

_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    import p131_rc_trained_policy as P  # noqa: E402  (sets sys.argv for params.py at import)
finally:
    sys.argv = _argv

SCRIPT = os.path.join(ROOT, 'scripts', 'p131_rc_trained_policy.py')
PY = sys.executable
NEW_FOLDER = os.path.basename(P.DEFAULT_ROOT)
SEEDS = (301, 302, 303)
POLICY_CFG = os.path.join(ROOT, 'train_log', 'FAMILY', 'config_10x25+family+xmask-v1-s301.json')


def env(**kv):
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kv.items()})
    return e


# ---------------------------------------------------------------------------
# design
# ---------------------------------------------------------------------------

def test_design_units_and_folders():
    assert P.SEEDS == SEEDS and P.LEVELS == ('mix_rc', 'default')
    assert P.SIZES == ('10x25', '50x25n30', '80x25n30') and P.EXPECT_PARAMS == 32786
    assert P.EXPECT_MIX == 'rc_project' and P.RULES7 == P.P129.RULES7
    units = P.all_units(P.LEVELS, P.SIZES, P.R8, P.SEEDS)
    assert len(units) == len(set(units)) == 2 * 3 * 8 * 3 + 3 * 2 == 150
    arm = [u for u in units if u[3] == 'rct']
    assert {u[5] for u in arm} == {f'greedy+10x25+family+xmask-rc-s{s}' for s in SEEDS}
    assert {(u[0], u[1]): P.unit_n(u) for u in arm} == {
        ('mix_rc', '10x25'): 50, ('mix_rc', '50x25n30'): 30, ('mix_rc', '80x25n30'): 30,
        ('default', '10x25'): 100, ('default', '50x25n30'): 30, ('default', '80x25n30'): 30}
    ctl = [u for u in units if u[0] == 'control']
    assert {u[5] for u in ctl} == {'greedy+10x25+family+xmask-v1-s301'}
    assert sorted({u[2] for u in ctl}) == ['LSW', 'N'] and all(P.unit_n(u) == 3 for u in ctl)
    assert not [u for u in P.all_units(['mix_rc'], P.SIZES, P.R8, P.SEEDS) if u[0] == 'control']
    assert P.P129.result_path(P.DEFAULT_ROOT, 'default', '80x25n30', 'LS', arm[0][5]) == (
        'test_results/FAMILY/rc_trained_p131/results/exact/default/80x25n30/LS/'
        'greedy+10x25+family+xmask-rc-s301.npz')
    assert P.policy_model(302) == '10x25+family+xmask-v1-s302'
    assert set(P.COMPARISONS) == {'rct_rule', 'rct_rule7', 'rct_pol', 'rct_spec', 'pol_rule',
                                  'pol_rule7'}
    assert set(P.SUPERIORITY) == {'rct_rule', 'rct_rule7', 'pol_rule', 'pol_rule7'}
    # the planned cost: p129's per-instance rates
    assert sum(P.unit_cost(u) for u in units) == pytest.approx(
        3 * 8 * (50 * 1.3 + 100 * 1.3 + 2 * 30 * 9.0 + 2 * 30 * 20.0) + 2 * 3 * (1.3 + 9.0 + 20.0))


def test_macro_prefix_is_unused_under_manuscript():
    have = P.P129.defined_macros([os.path.join(ROOT, P.OUT_FILES['macros'])])
    assert len(have) > 1000 and not any(x.startswith(P.PREFIX) for x in have)


# ---------------------------------------------------------------------------
# the seed stream and the contamination check
# ---------------------------------------------------------------------------

def test_replay_equals_the_global_numpy_stream():
    state = np.random.get_state()
    try:
        np.random.seed(302)
        want = [int(np.random.randint(0, 2 ** 31 - 1)) for _ in range(2 * 20 * 3)]
    finally:
        np.random.set_state(state)
    inst, desc = P.stream_seeds(302, 60, 20, 20)
    assert len(inst) == len(desc) == 60
    assert [x for pair in zip(inst, desc) for x in pair] == want


@pytest.mark.skipif(not os.path.exists(POLICY_CFG), reason='the policy configuration is absent')
def test_replay_equals_the_trainer_sampler():
    settings = {s: P.stream_settings(s) for s in (301, 303)}
    assert settings[301]['updates'] == 2000 and settings[301]['reset'] == 20
    assert settings[301]['envs'] == 20 and settings[301]['vseed'] == 100
    assert settings[301]['vali_n'] == 20 and settings[301]['modules'] == 10
    got = P.trainer_draws([301, 303], 1, settings)
    for s in (301, 303):
        inst, desc = P.stream_seeds(s, 20, 20, 20)
        want = []
        for a, b in zip(inst, desc):
            want += [['inst', 10, 'rc_project', a], ['desc', b]]
        assert got[s] == want


def small_settings(monkeypatch, updates=40):
    real = P.stream_settings

    def fake(seed):
        return dict(real(seed), updates=updates)
    monkeypatch.setattr(P, 'stream_settings', fake)


@pytest.mark.skipif(not os.path.exists(POLICY_CFG), reason='the policy configuration is absent')
def test_contamination_refuses_a_planted_collision(monkeypatch):
    small_settings(monkeypatch)
    rec = P.contamination((301,), batches=0, content=True)
    assert rec['compared'] == {'generated': 40 + 20, 'stored': 150}
    assert not rec['seed_hits'] and not rec['content_hits']
    first = P.stream_seeds(301, 40, 20, 20)[0][0]
    blocks = P.heldout_blocks(P.STORED_DATA)
    planted = dict(blocks)
    planted['planted 50x25n30'] = dict(seed0=first, n=1, modules=50, mix='rc_project', base='x')
    monkeypatch.setattr(P, 'heldout_blocks', lambda data_root: planted)
    P.contamination((301,), batches=0, content=False)           # another module count: no hit
    planted['planted 10x25'] = dict(seed0=first, n=1, modules=10, mix='rc_project', base='x')
    with pytest.raises(P.Refused, match='may equal a test instance'):
        P.contamination((301,), batches=0, content=False)
    planted['planted 10x25'] = dict(seed0=105, n=10, modules=10, mix='rc_project', base='x')
    with pytest.raises(P.Refused, match='"validation"'):         # validation seeds 100-119
        P.contamination((301,), batches=0, content=False)
    # an equal instance in content, whatever its seed
    monkeypatch.setattr(P, 'heldout_blocks', lambda data_root: blocks)
    from ppvc_instance_generator import DEFAULT_FACTORY, ppvc_instance_generator
    jl, pt, _ = ppvc_instance_generator(n_modules=10, class_mix='rc_project',
                                        station_counts=DEFAULT_FACTORY, seed=first)
    monkeypatch.setattr(P, 'stored_instance_keys',
                        lambda b, m: {P.instance_key(jl, pt): 'planted/instance_000'})
    with pytest.raises(P.Refused, match='planted/instance_000'):
        P.contamination((301,), batches=0, content=True)


# ---------------------------------------------------------------------------
# a hand-made case on synthetic results
# ---------------------------------------------------------------------------

FACTOR = {'FIFO': 1.05, 'MOR': 1.06, 'SPT': 1.07, 'MWKR': 1.08, 'ECT': 1.02, 'SSPT': 1.09,
          'LLF': 1.09}
OFFSET = dict(zip(SEEDS, (-0.0005, 0.0, 0.0005)))


def write_unit(root, level, size, combo, label, ms, names, policy=None):
    p = P.P129.result_path(root, level, size, combo, label)
    P.P101.save_npz_atomic(p, makespan=np.asarray(ms, float))
    rec = {'level': level, 'size': size, 'regime': combo, 'label': label, 'instances': names,
           'validated': len(names), 'violations': 0}
    if policy is not None:
        rec['policy'] = policy
    P.write_json_atomic(p[:-4] + '.json', rec)


def policy_config(seed):
    if os.path.exists(POLICY_CFG):
        c = json.load(open(POLICY_CFG))
    else:
        c = {'ppvc_mix': 'mixed', 'first_mask': 'exact', 'mask_ties': 'exact', 'lr': 2e-4,
             'max_updates': 2000, 'reset_env_timestep': 20, 'num_envs': 20, 'n_j': 10,
             'seed_train_vali_datagen': 100, 'vali_size': 100, 'ppvc_factory': 'default',
             'family_regimes': 'N,L,S,W,LS,LW', 'ppvc_size_mix': ''}
    return dict(c, seed_train=seed, model_suffix=f'xmask-v1-s{seed}')


def plant_runs(tmp_path, monkeypatch, change=None):
    """Configurations and checkpoints of the arm and the policy in a scratch folder; returns
    {seed: the policy record a result file of the arm carries}."""
    cfg = tmp_path / 'runs' / 'train_log' / 'FAMILY'
    ck = tmp_path / 'runs' / 'trained_network' / 'FAMILY'
    cfg.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    monkeypatch.setitem(P.RUN_DIRS, 'config', str(cfg))
    monkeypatch.setitem(P.RUN_DIRS, 'checkpoint', str(ck))
    monkeypatch.setitem(P.RUN_DIRS, 'log', str(tmp_path / 'runs' / 'train_log'))
    out = {}
    for s in SEEDS:
        ref = policy_config(s)
        mine = dict(ref, model_suffix=f'xmask-rc-s{s}', ppvc_mix='rc_project')
        mine.update(change or {})
        (cfg / f'config_{P.policy_model(s)}.json').write_text(json.dumps(ref))
        (cfg / f'config_{P.rct_model(s)}.json').write_text(json.dumps(mine))
        (ck / f'{P.rct_model(s)}.pth').write_bytes(b'checkpoint %d' % s)
        out[s] = {'model': P.rct_model(s), 'params': 32786,
                  'snapshot_sha256': P.P101.sha256_file(str(cfg / f'config_{P.rct_model(s)}.json')),
                  'checkpoint_sha256': P.P101.sha256_file(str(ck / f'{P.rct_model(s)}.pth'))}
    return out


def plant_case(tmp_path, monkeypatch, size='50x25n30'):
    """mix_rc at 50 modules: the specialist is random, the policy 1% longer and the arm 3%
    shorter than it (seed offsets -0.05, 0, +0.05 points), the four classical rules 5 to 8%
    longer, ECT 2% longer, SSPT and LLF 9% longer."""
    root, stored, data = str(tmp_path / 'r'), str(tmp_path / 'stored'), str(tmp_path / 'data')
    lv, n = 'mix_rc', P.P129.N_NEW
    names = [P.P129.inst(i) for i in range(n)]
    d = os.path.join(P.P129.desc_dir(data, lv, size), 'desc', 'N')
    os.makedirs(d)
    for x in names:
        open(os.path.join(d, x + '.desc.json'), 'w').close()
    pol = plant_runs(tmp_path, monkeypatch)
    rng = np.random.default_rng(0)
    for combo in P.R8:
        spec = 1000.0 * (1 + 0.1 * rng.random(n))
        for r in P.RULES7:
            write_unit(stored, lv, size, combo, r, FACTOR[r] * spec, names)
        for s in SEEDS:
            write_unit(stored, lv, size, combo, 'greedy+' + P.P103.model_name('spec', combo, s),
                       spec, names)
            write_unit(stored, lv, size, combo, 'greedy+' + P.P103.model_name('pact', combo, s),
                       (1.01 + OFFSET[s]) * spec, names)
            write_unit(root, lv, size, combo, P.rct_label(s), (0.97 + OFFSET[s]) * spec, names,
                       policy=pol[s])
    monkeypatch.setattr(P, 'contamination', lambda *a, **k: {
        'compared': {}, 'seed_hits': [], 'content_hits': [], 'trainer_check': None})
    return root, stored, data, names


def infer_args(root, stored, data, out, levels, sizes, n=None, boot=400):
    class A:
        pass
    a = A()
    a.root, a.stored_root, a.data_root, a.out, a.partial, a.n, a.boot = (
        root, stored, data, out, True, n, boot)
    a.levels, a.sizes, a.combos, a.seeds = list(levels), list(sizes), list(P.R8), list(SEEDS)
    return a


def test_infer_on_a_hand_made_case(tmp_path, monkeypatch):
    root, stored, data, names = plant_case(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    res = P.infer(infer_args(root, stored, data, str(out), ['mix_rc'], ['50x25n30']))
    S = res['sizes']['50x25n30']
    B = S['comparisons']['mix_rc']
    pooled = B['pooled']
    assert pooled['rct_rule']['mean'] == pytest.approx(100 * (0.97 / 1.05 - 1), abs=1e-9)
    assert pooled['rct_rule7']['mean'] == pytest.approx(100 * (0.97 / 1.02 - 1), abs=1e-9)
    assert pooled['rct_pol']['mean'] == pytest.approx(100 * (0.97 / 1.01 - 1), abs=1e-4)
    assert pooled['rct_spec']['mean'] == pytest.approx(-3.0, abs=1e-9)
    assert pooled['pol_rule']['mean'] == pytest.approx(100 * (1.01 / 1.05 - 1), abs=1e-9)
    assert pooled['pol_rule7']['mean'] == pytest.approx(100 * (1.01 / 1.02 - 1), abs=1e-9)
    for key in P.SUPERIORITY:
        assert B['counts'][key]['ahead'] == 8, key
        assert set(B['rows'][key]['LSW']) >= {'verdict', 'p_t_holm', 'p_boot_holm'}
    for key in P.EQUIVALENCE:
        assert B['counts'][f'{key}|direction']['ahead'] == 8
        assert B['counts'][f'{key}|1.0']['non-inferior only'] == 8
    assert B['rows']['rct_pol']['N']['scheme'] == 'hierarchical' and B['rows']['rct_pol']['N']['k'] == 3
    assert S['validated'] == 8 * (7 + 2 * 3 + 3) * len(names)
    C = S['levels']['mix_rc']['combos']['N']
    assert C['policy_ms']['rct'] == pytest.approx(0.97 * C['policy_ms']['spec'], rel=1e-9)
    assert C['best_rule7_ms'] == pytest.approx(C['rule_ms']['ECT'])
    assert res['controls']['record'] == 'not compared (partial run)'
    assert res['runs']['301']['differs_from_policy'] == ['model_suffix', 'ppvc_mix']
    txt = open(out / P.OUT_FILES['macros']).read()
    lines = [ln for ln in txt.splitlines() if ln.startswith('\\newcommand')]
    mnames = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(mnames) == len(set(mnames)) == len(lines) > 300
    assert all(re.fullmatch(r'Rct[A-Za-z]+', x) for x in mnames)
    seed_line = [ln for ln in lines if ln.startswith('\\newcommand{\\RctSeedN}')]
    assert len(seed_line) == 1 and '{3}' in seed_line[0] and '301,302,303' in seed_line[0]
    assert all('  % ' in ln and 'level mix_rc, 50x25n30, seeds 301,302,303' in ln
               for ln in lines if ln not in seed_line)
    for want in ('RctRcRuleConcreteFiftyPool', 'RctRcRuleVerdictConcreteFiftyLSW',
                 'RctRcPolVHalfConcreteFiftyN', 'RctRcSpecDirConcreteFiftyPool',
                 'RctPolRuleSevenAheadConcreteFifty', 'RctMsRcConcreteFiftyN', 'RctNConcreteFifty'):
        assert want in mnames, want
    assert '\\RctRcRuleVerdictConcreteFiftyN}{ahead}' in txt and 'PARTIAL' in txt
    md = open(out / P.OUT_FILES['md']).read()
    assert '## 50x25n30, mix_rc' in md and 'PARTIAL' in md and '(i) arm - best rule' in md
    for f in (out / P.OUT_FILES['json'], out / P.OUT_FILES['md']):
        assert '/home/' not in open(f).read().replace(str(tmp_path), '')


@pytest.mark.parametrize('damage,want', [
    ('mix', "records ppvc_mix='mixed'"),
    ('drift', 'configuration differs from xmask-v1-s301 in lr'),
    ('params', 'has 32000 parameters'),
    ('checkpoint', 'written by another checkpoint'),
    ('missing', 'is missing'),
    ('short', 'holds 29 makespans'),
])
def test_infer_refuses(tmp_path, monkeypatch, damage, want):
    root, stored, data, names = plant_case(tmp_path, monkeypatch)
    label = P.rct_label(302)
    p = P.P129.result_path(root, 'mix_rc', '50x25n30', 'SW', label)
    if damage in ('mix', 'drift'):
        plant_runs(tmp_path, monkeypatch, {'ppvc_mix': 'mixed'} if damage == 'mix' else {'lr': 0.5})
    elif damage == 'params':
        rec = json.load(open(p[:-4] + '.json'))
        rec['policy']['params'] = 32000
        P.write_json_atomic(p[:-4] + '.json', rec)
    elif damage == 'checkpoint':
        with open(os.path.join(P.RUN_DIRS['checkpoint'], P.rct_model(302) + '.pth'), 'ab') as f:
            f.write(b'retrained')
    elif damage == 'missing':
        os.remove(p)
    elif damage == 'short':
        P.P101.save_npz_atomic(p, makespan=np.ones(len(names) - 1))
    out = tmp_path / 'out'
    with pytest.raises(P.Refused, match=re.escape(want)):
        P.infer(infer_args(root, stored, data, str(out), ['mix_rc'], ['50x25n30']))
    assert not (out / P.OUT_FILES['json']).exists() and not (out / P.OUT_FILES['macros']).exists()


def test_macro_name_check(tmp_path, monkeypatch):
    paths = {k: str(tmp_path / v) for k, v in P.OUT_FILES.items()}
    with pytest.raises(P.Refused, match='macro names that are not Rct plus letters'):
        P.check_macro_names([('RctA1', 1, 'x')], paths)
    with pytest.raises(P.Refused, match='duplicate macro names'):
        P.check_macro_names([('RctA', 1, 'x'), ('RctA', 2, 'y')], paths)
    with monkeypatch.context() as m:
        m.setattr(P.P129, 'defined_macros', lambda own: {'RctRcPolMixedTenN': 'manuscript/x.tex'})
        with pytest.raises(P.Refused, match=r'already defined: \\RctRcPolMixedTenN'):
            P.check_macro_names([('RctRcPolMixedTenN', 1, 'x')], paths)
        with pytest.raises(P.Refused, match='the prefix Rct is used'):
            P.check_macro_names([('RctOther', 1, 'x')], paths)
    assert P.check_macro_names([('RctA', 1, 'x'), '% comment'], paths) == ['RctA']


# ---------------------------------------------------------------------------
# evaluate: refusals before and during the rollouts
# ---------------------------------------------------------------------------

class Args:
    def __init__(self, root, **kw):
        self.root, self.hold, self.force, self.n, self.data_root = root, '/nonexistent/hold', False, None, P.STORED_DATA
        self.__dict__.update(kw)


def test_a_schedule_that_fails_the_check_stops_the_run_and_writes_nothing(tmp_path, monkeypatch):
    jl, pt = np.array([2]), np.ones((2, 3), dtype=int)
    cell = [('instance_000', jl, pt, {'op_type': None}, None)]
    monkeypatch.setattr(P.P129, 'load_cell', lambda *a, **k: cell)
    monkeypatch.setattr(P.P103, 'rollout', lambda *a: dict(makespan=2.0, amch=np.zeros(2, int),
                                                          op_ct=np.array([1.0, 2.0]), lb0=1.0))
    monkeypatch.setattr(P.P103, 'schedule_check', lambda *a: ['two operations overlap on a machine'])
    root = str(tmp_path / 'r')
    u = ('mix_rc', '10x25', 'N', 'rct', 301, P.rct_label(301))
    with pytest.raises(P.Refused, match='is infeasible'):
        P.run_units(Args(root), [u], runner=None)
    assert not os.path.exists(os.path.join(root, 'results'))


def test_evaluate_refuses_an_unfinished_run_and_another_parameter_count(tmp_path, monkeypatch):
    plant_runs(tmp_path, monkeypatch)

    class Runner:
        def __init__(self, n):
            self.n = n

        def load(self, model):
            return None, {'params': self.n}
    with pytest.raises(P.Refused, match='has no completion line'):
        P.verify_runs([301], Runner(32786))
    log = tmp_path / 'runs' / 'train_log' / 'FAMILY_xmask-rc-s301.log'
    log.write_text('[train_family] done in 3.00 h; best rel=0.85\n')
    assert P.verify_runs([301], Runner(32786))[301]['params'] == 32786
    with pytest.raises(P.Refused, match='has 32914 parameters'):
        P.verify_runs([301], Runner(32914))


def test_evaluate_verifies_the_runs_of_its_part_and_rolls_out_its_units(tmp_path, monkeypatch):
    plant_runs(tmp_path, monkeypatch)
    for s in SEEDS:
        (tmp_path / 'runs' / 'train_log' / f'FAMILY_xmask-rc-s{s}.log').write_text('done in 3.0 h\n')
    loaded, got = [], []

    class Runner:
        def load(self, model):
            loaded.append(model)
            return None, {'params': 32786}
    monkeypatch.setattr(P.P101, 'PolicyRunner', Runner)
    monkeypatch.setattr(P, 'run_units', lambda args, units, runner: got.append(units))
    units = P.all_units(P.LEVELS, P.SIZES, P.R8, SEEDS)
    parts, _ = P.assign_parts(units, 2)
    for i in range(2):
        loaded.clear()
        P.evaluate(Args(str(tmp_path / 'r'), levels=list(P.LEVELS), sizes=list(P.SIZES),
                        combos=list(P.R8), seeds=list(SEEDS), part=i, parts=2))
        assert got[-1] == [units[j] for j in parts[i]]
        assert loaded == [P.rct_model(s) for s in sorted({units[j][4] for j in parts[i]
                                                           if units[j][3] == 'rct'})]


# ---------------------------------------------------------------------------
# the level default and p129's record, on the stored results
# ---------------------------------------------------------------------------

STORED_POLICY = P.REG.result_path('released', None, 301, '10x25', 'N')
RECORD = os.path.join(ROOT, P.RECORD_P129)


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, STORED_POLICY)),
                    reason='the stored arrays of the exact campaign are absent')
def test_default_level_reads_the_stored_arrays_and_checks_the_evaluation_path(tmp_path, monkeypatch):
    """The arm 1% shorter than the stored policy of the same seed on every instance: the pooled
    gap to the policy is -1.0 exactly, and (i) equals (iv) shifted by the same factor."""
    root, size = str(tmp_path / 'r'), '10x25'
    pol = plant_runs(tmp_path, monkeypatch)
    monkeypatch.setattr(P, 'contamination', lambda *a, **k: {
        'compared': {}, 'seed_hits': [], 'content_hits': [], 'trainer_check': None})
    names = P.P101.names_of(size)
    for combo in P.R8:
        for s in SEEDS:
            stored = P.REG.read_path(P.REG.result_path('released', None, s, size, combo))
            write_unit(root, 'default', size, combo, P.rct_label(s), 0.99 * stored, names,
                       policy=pol[s])
    for combo in P.CONTROL_COMBOS:
        stored = P.REG.read_path(P.REG.result_path('released', None, 301, size, combo))
        write_unit(root, 'control', size, combo, P.control_label(), stored[:3], names[:3])
    out = str(tmp_path / 'out')
    res = P.infer(infer_args(root, None, P.STORED_DATA, out, ['default'], [size]))
    B = res['sizes'][size]['comparisons']['default']
    assert B['pooled']['rct_pol']['mean'] == pytest.approx(-1.0, abs=1e-9)
    assert B['pooled']['rct_pol']['lo'] == pytest.approx(-1.0, abs=1e-9)
    for r in P.R8:
        a, b = B['rows']['rct_rule'][r]['mean'], B['rows']['pol_rule'][r]['mean']
        assert a < b
    assert res['controls']['evaluation_path'] == {f'{size} {c}': [3, 3] for c in P.CONTROL_COMBOS}
    assert res['sizes'][size]['levels']['default']['n'] == 100
    # one makespan of a control unit off: refused
    p = P.P129.result_path(root, 'control', size, 'LSW', P.control_label())
    with np.load(p) as z:
        ms = np.asarray(z['makespan'], float)
    ms[1] += 0.5
    P.P101.save_npz_atomic(p, makespan=ms)
    with pytest.raises(P.Refused, match=r'the control 10x25/LSW: 1 of 3 makespans'):
        P.infer(infer_args(root, None, P.STORED_DATA, out, ['default'], [size]))


@pytest.mark.skipif(not os.path.exists(RECORD), reason="p129's record is absent")
def test_rule_means_of_the_stored_results_equal_p129_record(tmp_path):
    res = {'sizes': {}}
    for size in P.SIZES:
        S = res['sizes'][size] = {'levels': {}}
        for lv in P.LEVELS:
            names = P.P129.level_names(lv, size, P.STORED_DATA)
            Lv = S['levels'][lv] = {'n': len(names), 'combos': {}}
            for combo in ('N', 'LSW'):
                rules, _ = P.stored_cell(lv, size, combo, SEEDS, names, P.STORED_ROOT,
                                         {'validated': 0})
                b4 = np.min(np.stack([rules[r] for r in P.RULES4]), axis=0)
                b7 = np.min(np.stack([rules[r] for r in P.RULES7]), axis=0)
                Lv['combos'][combo] = dict(rule_ms={r: float(rules[r].mean()) for r in P.RULES7},
                                           best_rule_ms=float(b4.mean()),
                                           best_rule7_ms=float(b7.mean()))
    assert P.record_control(res)['values'] == 3 * 2 * 2 * 9
    res['sizes']['50x25n30']['levels']['mix_rc']['combos']['N']['rule_ms']['SPT'] += 0.01
    with pytest.raises(P.Refused, match='rule means differ'):
        P.record_control(res)


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('K', [1, 2, 7])
def test_plan_partition(K):
    units = P.all_units(P.LEVELS, P.SIZES, P.R8, P.SEEDS)
    parts, loads = P.assign_parts(units, K)
    assert sorted(i for p in parts for i in p) == list(range(len(units)))
    assert max(loads) - min(loads) <= max(P.unit_cost(u) for u in units) + 1e-9
    assert sum(loads) == pytest.approx(sum(P.unit_cost(u) for u in units))


def test_plan_prints_one_command_per_part_and_skips_finished(tmp_path):
    root = tmp_path / 'r'
    u = ('mix_rc', '10x25', 'N', 'rct', 301, P.rct_label(301))
    write_unit(str(root), *u[:3], u[5], np.ones(50), [f'instance_{i:03d}' for i in range(50)])
    r = subprocess.run([PY, '-B', SCRIPT, 'plan', '--parts', '2', '--root', str(root)],
                       capture_output=True, text=True, cwd=ROOT, env=env())
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cmds = [x for x in r.stdout.splitlines() if 'p131_rc_trained_policy.py evaluate' in x]
    assert len(cmds) == 2
    for i, c in enumerate(cmds):
        assert c.startswith('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ')
        assert f'taskset -c <CORE_{i}> ' in c and c.endswith(
            f'evaluate --part {i} --parts 2 --root {root}')
    assert '150 result files (1 finished)' in r.stdout
    assert 'total 12.95 core-hours' in r.stdout
    assert sorted(os.listdir(root / 'results' / 'exact')) == ['mix_rc']   # plan writes nothing


# ---------------------------------------------------------------------------
# the folder
# ---------------------------------------------------------------------------

def _scanned_files():
    out = []
    for top in ('.', 'scripts', 'tests', 'constraint_family', 'baselines', 'backbones', 'model'):
        d = os.path.join(ROOT, top)
        if not os.path.isdir(d):
            continue
        if top == '.':
            out += [os.path.join(d, f) for f in os.listdir(d) if f.endswith(('.py', '.sh'))]
            continue
        for dp, dns, fns in os.walk(d):
            dns[:] = [x for x in dns if not x.startswith(('.', '__'))]
            out += [os.path.join(dp, f) for f in fns if f.endswith(('.py', '.sh'))]
    return sorted(set(os.path.normpath(p) for p in out))


def test_new_folder_is_read_by_no_existing_script():
    """The new folder holds greedy+<run>.npz and their JSON records at depth 6 below
    test_results/FAMILY. No other script names it, and no literal path pattern below
    test_results/FAMILY with a wildcard matches one of its file paths."""
    own = {os.path.normpath(os.path.join(ROOT, p)) for p in (
        'scripts/p131_rc_trained_policy.py', 'tests/test_p131_rc_trained_policy.py')}
    base = 'test_results/FAMILY/' + NEW_FOLDER
    examples = [base + '/results/exact/mix_rc/10x25/LSW/greedy+10x25+family+xmask-rc-s301.npz',
                base + '/results/exact/default/80x25n30/S/greedy+10x25+family+xmask-rc-s303.json',
                base + '/results/exact/control/50x25n30/N/greedy+10x25+family+xmask-v1-s301.npz']
    rx = re.compile('test_results' + r'/FAMILY/[^\'"\s)]*')
    patterns = {}
    for f in _scanned_files():
        if f in own:
            continue
        txt = open(f, errors='replace').read()
        assert NEW_FOLDER not in txt, f'{f} names {NEW_FOLDER}'
        for m in rx.findall(txt):
            first = m.split('/')[2] if m.count('/') >= 2 else ''
            if any(ch in first for ch in '*?['):
                patterns.setdefault(m, f)
    assert patterns, 'the scan found no wildcard pattern; the scan itself is broken'
    for pat, f in patterns.items():
        for ex in examples:
            assert not fnmatch.fnmatch(ex, pat), f'{f}: {pat} matches {ex}'
    import arm_registry as REG
    suffixes = set(REG.FOLDER_SUFFIX.values()) | set(REG.LEGACY_TAG.values()) - {''}
    assert not re.match(r'\d+x\d+', NEW_FOLDER)
    assert not any(s and NEW_FOLDER.endswith(s) for s in suffixes)
    assert NEW_FOLDER not in REG.FOLDER_ALIAS['exact'] and NEW_FOLDER not in REG.LEGACY_TAG
