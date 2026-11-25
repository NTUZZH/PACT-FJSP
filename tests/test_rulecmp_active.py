"""The comparison with dispatching rules of the public family under the active mask
(scripts/rulecmp_plan.py --mask active, scripts/p93_rule_comparison.py --mask active, the
rule side through scripts/p109_public_active_rules.py, the policy side through
scripts/p90_public_rule_comparison.py, and the step rulecmp_public_act of
scripts/run_analysis.sh).

  * the plan under the active mask lists exactly the jobs of the plan under the non-delay
    mask, with the run names, the folders and the mask changed, on every public test set;
  * the rule jobs under both masks on small samples of the stored records: the non-delay
    rollouts reproduce the stored tie-break-seed and validation arrays bit for bit, the
    rollouts under the active mask reproduce the stored arrays of p109, a rule trace under
    the active mask reproduces them and records its mask;
  * p93 on a synthetic tree that holds both families: hand-computed values of one
    difference in each of the four forms, the counterpart macros equal to the default macro
    set, the refusal of a partial campaign, a partial report only under --out and marked,
    nothing written outside --out, the default outputs byte-identical, no absolute path;
  * the chain step and its every-file-or-none precondition.

Run from the repository root:  python -m pytest tests/test_rulecmp_active.py -q
CPU only, one thread; every file is written under a temporary folder.
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, ROOT)

import arm_registry as REG  # noqa: E402
import p109_public_active_rules as P109  # noqa: E402
import p93_rule_comparison as P93  # noqa: E402
import rulecmp_plan as PLAN  # noqa: E402
from conftest import needs_public_instances  # noqa: E402

R8 = list(REG.COMBOS)
RULES = PLAN.RULES
HEAD = PLAN.HEADLINE
VALI_SET = P109.VALI_SETS[0]
N_INST = 4
VALI_SIZE = 2
TEST_K = 32
B = 200
DONE = '[train_family] done in 1 s\n'
CHAIN = os.path.join(ROOT, 'scripts', 'run_analysis.sh')
HAS_DATA = os.path.isdir(os.path.join(ROOT, 'data/PUBLIC/family', HEAD, 'desc'))
needs_data = pytest.mark.skipif(not HAS_DATA, reason='needs the public-family records')


# ================================================================== the plan under both masks
def model_map(nd, ac):
    """{non-delay run name: active run name} over the runs of two families, matched by arm,
    combination and seed."""
    key = {(r.arm, r.combo, r.seed): r for r in ac.runs}
    assert sorted(key) == sorted((r.arm, r.combo, r.seed) for r in nd.runs)
    out = {}
    for r in nd.runs:
        a = key[(r.arm, r.combo, r.seed)]
        out[REG.pub_model(r.arm, r.combo, r.seed)] = REG.pub_rule_model(r.arm, r.combo, r.seed,
                                                                       word='act')
        assert a.model == r.model.replace(REG.pub_model(r.arm, r.combo, r.seed),
                                          out[REG.pub_model(r.arm, r.combo, r.seed)])
    return out


def translator(nd, ac, root):
    """The non-delay names of jobs, paths and commands in the names of the active family: the
    run names, and the rule folder and its rulecmp_* subfolders (a greedy array stays in the
    result folder both families share)."""
    names = model_map(nd, ac)
    rx = re.compile('|'.join(re.escape(k) for k in sorted(names, key=len, reverse=True)))
    old = os.path.join(root, nd.campaign) + os.sep
    new = os.path.join(root, ac.campaign) + os.sep
    old_rel, new_rel = nd.campaign + '/', ac.campaign + '/'

    def T(s):
        s = rx.sub(lambda m: names[m.group(0)], s)
        if os.path.basename(s).startswith('Result_greedy+'):
            return s
        if s.startswith(old):
            return new + s[len(old):]
        if s.startswith(old_rel):
            return new_rel + s[len(old_rel):]
        return s
    return T


@pytest.mark.parametrize('set_name', list(PLAN.PUB_SET_INFO))
def test_plan_under_the_active_mask_lists_the_jobs_of_the_non_delay_plan(tmp_path, monkeypatch,
                                                                       set_name):
    monkeypatch.setattr(PLAN, 'ROOT', str(tmp_path))
    nd = PLAN.Family('public', set_name=set_name)
    ac = PLAN.Family('public', set_name=set_name, mask='active')
    assert (nd.mask, nd.word, ac.mask, ac.word) == ('nondelay', '', 'active', 'act')
    assert ac.label == ('public_act' if set_name == HEAD else
                        f'public_act_{PLAN.PUB_SET_INFO[set_name][0].lower()}')
    assert ac.label != nd.label and ac.macro == 'RcAct' + nd.tag and nd.macro == 'Rcp' + nd.tag
    assert ac.folder == nd.folder and ac.rule_folder == ac.campaign == \
        f'{P109.RULE_ROOT["active"]}/{set_name}{PLAN.XS}'
    T = translator(nd, ac, str(tmp_path))
    jn, ja = PLAN.jobs(nd, ratio=1.0), PLAN.jobs(ac, ratio=1.0)
    assert [(j.kind, T(j.id), [T(p) for p in j.outputs], [T(p) for p in j.needs], j.status)
            for j in jn] == \
        [(j.kind, j.id, j.outputs, j.needs, j.status) for j in ja]
    # the headline set: 5 released runs and 5 controls on 8 combinations and 40 specialists on
    # their own; every other set: the released runs only; 5 policy and 4 rule traces; 3 seeds
    assert len(ja) == (120 + 9 + 3 + 1 if set_name == HEAD else 40 + 9 + 3)
    for a, b in zip(jn, ja):
        cn, ca = PLAN.command(nd, a, '<tmp>'), PLAN.command(ac, b, '<tmp>')
        if a.kind == 'sample' or (a.kind == 'trace' and a.run is not None):
            assert ca == [T(x) for x in cn] + ['--expect_mask', 'active'], a.id
        elif a.kind == 'trace':
            i = cn.index('--rule') + 2
            assert ca == cn[:i] + ['--rule_mask', 'active'] + cn[i:] + ['--expect_mask', 'active']
        elif a.kind == 'ties':
            assert ca == [PLAN.py(), 'scripts/p109_public_active_rules.py', 'run', '--mask',
                          'active', '--sets', set_name, '--seed', str(b.t), '--out-root', '<tmp>']
        else:
            assert ca == [PLAN.py(), 'scripts/p109_public_active_rules.py', 'run', '--mask',
                          'active', '--sets', VALI_SET, '--out-root', '<tmp>']
    # the jobs of the rules alone: the rule traces, the tie-break seeds and the validation job
    jr = PLAN.jobs(ac, ratio=1.0, rules_only=True)
    assert [j.id for j in jr] == [j.id for j in ja if j.kind != 'sample' and
                                  getattr(j, 'run', None) is None]


def test_the_validation_set_of_p109_is_the_one_the_runs_validated_on():
    for mask in PLAN.MASKS:
        fam = PLAN.Family('public', mask=mask)
        assert os.path.basename(fam.vali_data) == VALI_SET and fam.vali_dir == VALI_SET + PLAN.XS
    assert VALI_SET not in P109.SETS and VALI_SET in P109.ALL_SETS


def test_masks_of_the_output_files():
    """A file without the key mask_mode counts as the non-delay mask."""
    class Z(dict):
        files = property(lambda self: list(self))
    assert PLAN.mask_problem(Z(), 'nondelay') is None
    assert 'records no mask' in PLAN.mask_problem(Z(), 'active')
    assert PLAN.mask_problem(Z(mask_mode=np.array('active')), 'active') is None
    assert 'built under the active mask' in PLAN.mask_problem(Z(mask_mode=np.array('active')),
                                                              'nondelay')
    assert PLAN.mask_problem(Z(), None) is None
    with pytest.raises(SystemExit):
        PLAN.label_of('factory', None, 'active')


# ================================================================== the rule jobs, real records
def ns(**kw):
    import argparse
    base = dict(sets='', combos='', mask='active', out_root='', budget_s=0.0, n=0, seed=None,
                ref_root='')
    return argparse.Namespace(**dict(base, **kw))


@needs_public_instances()
@needs_data
def test_non_delay_rollouts_reproduce_the_stored_tie_break_and_validation_arrays():
    fam = PLAN.Family('public', K=PLAN.K_MIN)
    ties = os.path.join(fam.ties_dir(51), f'Result_SPT_S.npy')
    if not os.path.exists(ties) or not os.path.isdir(fam.vali_folder()):
        pytest.skip('needs the stored arrays of the comparison under the non-delay mask')
    lines = []
    tot, bad = P109.check_nondelay(ns(sets=HEAD, combos='S,LSW', n=3, seed=51), out=lines.append)
    assert not bad, bad
    assert tot == 2 * 4 * 3 and 'all equal' in lines[0] and 'tie-break seed 51' in lines[0]
    tot, bad = P109.check_nondelay(ns(sets=VALI_SET, combos='N,W', n=3), out=lines.append)
    assert not bad, bad
    assert tot == 2 * 4 * 3 and 'all equal' in lines[1]
    assert P109.nondelay_folder(VALI_SET, 50) == fam.vali_folder()
    assert P109.nondelay_folder(HEAD, 52) == fam.ties_dir(52)
    with pytest.raises(SystemExit, match='tie-break seed 50 only'):
        P109.nondelay_folder(VALI_SET, 51)


@needs_public_instances()
@needs_data
def test_active_rollouts_reproduce_the_arrays_of_p109_and_the_seed_reaches_the_rollout():
    if not os.path.exists(os.path.join(ROOT, P109.folder(P109.ACTIVE_ROOT, HEAD), 'Result_SPT_S.npy')):
        pytest.skip('needs the rule arrays of p109 under the active mask')
    lines = []
    tot, bad = P109.check_arrays(ns(sets=HEAD, combos='S', n=3, seed=50,
                                    ref_root=P109.ACTIVE_ROOT), out=lines.append)
    assert not bad, bad
    assert tot == 4 * 3 and 'all equal' in lines[0]
    # the tie-break seed reaches the stream of every rollout: one seeding per rule and record
    calls = []
    real = np.random.seed

    def seeded(s=None):
        calls.append(s)
        return real(s)
    try:
        np.random.seed = seeded
        P109.rollout_combo(HEAD, 'N', 'active', n=1, seed=52)
    finally:
        np.random.seed = real
    assert calls.count(52) == len(RULES) and P109.STORED_SEED not in calls


def test_p109_refusals_of_a_seed(tmp_path):
    with pytest.raises(SystemExit, match='needs --out-root'):
        P109.run(ns(sets=HEAD, seed=51))
    root = tmp_path / 'r'
    d = P109.folder(str(root), HEAD)
    os.makedirs(d)
    with open(os.path.join(d, P109.META), 'w') as f:
        json.dump(dict(mask_mode='active', tie_break_seed=51), f)
    with pytest.raises(SystemExit, match='tie-break seed 51, this run the active mask and seed 52'):
        P109.run(ns(sets=HEAD, seed=52, out_root=str(root)))
    with pytest.raises(SystemExit, match='check-arrays needs --ref-root'):
        P109.check_arrays(ns(sets=HEAD))
    assert sorted(os.listdir(d)) == [P109.META]


def run_p90(*args):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    return subprocess.run([sys.executable, '-B', 'scripts/p90_public_rule_comparison.py', 'rollout',
                           *args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


@needs_public_instances()
@needs_data
def test_a_rule_trace_under_the_active_mask_reproduces_the_arrays_of_p109(tmp_path):
    fam = PLAN.Family('public', mask='active')
    stored = fam.rule_path('MOR', 'S')
    if not os.path.exists(stored):
        pytest.skip('needs the rule arrays of p109 under the active mask')
    j = next(j for j in PLAN.jobs(fam, ratio=1.0, rules_only=True) if j.id == 'trace/MOR')
    j.regimes = ['S']
    cmd = PLAN.command(fam, j, str(tmp_path))
    r = subprocess.run(cmd[:1] + ['-B'] + cmd[1:] + ['--limit', '2'], cwd=ROOT,
                       env=PLAN.job_env(), capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    p = os.path.join(str(tmp_path), os.path.basename(fam.trace_path('MOR', 'S')))
    names = fam.names('S')[:2]
    assert PLAN.check_trace(p, names, PLAN.load_ms(stored)[:2], mask='active') is None
    assert 'built under the active mask' in PLAN.check_trace(p, names, mask='nondelay')
    # the switches refuse a schedule under another mask
    r = run_p90('--mode', 'trace', '--family_model', fam.runs[0].model, '--rule_mask', 'active',
                '--data', fam.data, '--base', fam.base, '--regimes', 'S', '--out', str(tmp_path))
    assert r.returncode == 2 and '--rule_mask' in r.stderr
    assert sorted(os.listdir(str(tmp_path))) == [os.path.basename(p)]


@needs_data
def test_a_policy_of_the_non_delay_mask_refuses_the_active_campaign(tmp_path):
    """The command of a job of the active family carries --expect_mask active, so a run whose
    configuration records another mask refuses before any rollout or output."""
    r0 = PLAN.Family('public').runs[0]
    if not all(os.path.exists(os.path.join(ROOT, p)) for p in (r0.checkpoint, r0.config)):
        pytest.skip('needs the checkpoint and configuration of a run of the non-delay mask')
    out = tmp_path / 'out'
    r = run_p90('--mode', 'trace', '--family_model', r0.model, '--data', PLAN.Family('public').data,
                '--base', 'record', '--regimes', 'S', '--out', str(out), '--limit', '1',
                '--torch_threads', '1', '--device', 'cpu', '--device_id', '', *PLAN.MASK,
                '--expect_mask', 'active')
    assert r.returncode != 0
    assert f'the schedules of {r0.model} would be built under the nondelay mask, not the active ' \
           f'mask' in r.stdout + r.stderr
    assert not out.exists()


# ================================================================== synthetic tree for p93
def save_result(path, ms):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([np.asarray(ms, float), np.zeros(len(ms))], axis=1))


def write_summary(folder, arrays, mask=None):
    n = len(next(iter(arrays.values())))
    L = ['# FAMILY eval — rules_xmask', '', 'mask: first_mask=exact mask_ties=exact', '']
    if mask:
        L += ['rules: rule_set=masked', '', f'pairs: mask_mode={mask}', '']
    L += ['| regime | method | mean | std | feasible | gap(cov) | gap to best known (N only) |',
          '|---|---|---|---|---|---|---|']
    L += [f'| {R} | {q} | {v.mean():.1f} | {v.std():.1f} | {n}/{n} | n/a | n/a |'
          for (R, q), v in arrays.items()]
    with open(os.path.join(folder, 'summary_rules_xmask.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')


def write_meta(folder, mask, seed):
    with open(os.path.join(folder, P109.META), 'w') as f:
        json.dump(dict(generator='scripts/p109_public_active_rules.py', mask_mode=mask,
                       tie_break_seed=seed), f)


def rule_folder(folder, arrays, mask, seed):
    for (R, q), v in arrays.items():
        save_result(os.path.join(folder, f'Result_{q}_{R}.npy'), v)
    write_summary(folder, arrays, mask if mask != 'nondelay' else None)
    if mask != 'nondelay':
        write_meta(folder, mask, seed)


def trace_file(path, names, ms, rng, mask):
    dec, ops = [], []
    for _ in names:
        dec.append(np.stack([rng.integers(1, 4, 5), np.ones(5), np.ones(5)], axis=1))
        o = np.zeros((5, 7))
        o[:, 0] = [0, 1, 0, 2, 1]
        o[:, 1] = rng.integers(1, 9, 5)
        o[:, 2] = rng.integers(0, 3, 5)
        ops.append(o)
    extra = {} if mask == 'nondelay' else dict(mask_mode=mask)
    np.savez(path, instances=np.array(names), makespan=np.asarray(ms, float),
             seconds=np.zeros(len(names)), decisions=np.array(dec, dtype=object),
             ops=np.array(ops, dtype=object), **extra)


def write_records(root, data, names, old):
    for R in R8:
        d = os.path.join(root, data, 'desc', R)
        os.makedirs(d, exist_ok=True)
        for nm in names:
            p = os.path.join(d, nm + '.desc.json')
            with open(p, 'w') as f:
                f.write('{}')
            os.utime(p, (old, old))
    with open(os.path.join(root, data, 'dataset_meta.json'), 'w') as f:
        json.dump(dict(n_instances=len(names), instances=list(names), excluded=[]), f)


def vali_values(R, q):
    """Validation arrays: on the first VALI_SIZE records MWKR is lowest in LS and SPT in every
    other combination; over all records FIFO is lowest in LS."""
    if R == 'LS':
        return np.array({'MWKR': [80, 80, 300, 300], 'FIFO': [120, 120, 90, 90]}.get(
            q, [130, 130, 130, 130]), float)
    return np.array([90, 91, 92, 93] if q == 'SPT' else [120, 121, 122, 123], float)


def build_family(root, fam, names, seed, old):
    """Every input of p93 for one family of the public headline set: runs (finished, with the
    family's mask in the configuration), stored rule and greedy arrays, sampled schedules,
    traces, the rule folders of the tie-break seeds and of the validation records, each
    under the family's mask. Returns the arrays written."""
    rng = np.random.default_rng(seed)
    n = len(names)
    for r in fam.runs:
        for p in (r.checkpoint, r.config, r.log):
            q = os.path.join(root, p)
            os.makedirs(os.path.dirname(q), exist_ok=True)
            with open(q, 'w') as f:
                if p.endswith('.json'):
                    json.dump(dict(family_base='sd2', n_j=10, n_m=5, vali_size=VALI_SIZE,
                                   seed_train_vali_datagen=100, ppvc_mix='mixed',
                                   ppvc_factory='default', dr_shift='', pub_vali_dir=fam.vali_data,
                                   **({} if fam.mask == 'nondelay' else dict(mask_mode=fam.mask))),
                              f)
                else:
                    f.write(DONE)
            os.utime(q, (old + 10, old + 10))
    base = os.path.join(root, fam.campaign)
    out = dict(stored={}, greedy={}, sample={}, ties={})
    arrays = {}
    for R in R8:
        for q in RULES:
            out['stored'][(q, R)] = arrays[(R, q)] = \
                (100 + 10 * np.arange(n) + rng.integers(0, 20, n)).astype(float)
    if fam.word:
        rule_folder(os.path.join(root, fam.rule_folder), arrays, fam.mask, PLAN.STORED_TIE_SEED)
    else:
        for (R, q), v in arrays.items():
            save_result(fam.rule_path(q, R, root), v)
    for r in fam.runs:
        for R in r.regimes:
            g = (100 + 10 * np.arange(n) + rng.integers(-5, 25, n)).astype(float)
            out['greedy'][(r.model, R)] = g
            save_result(fam.greedy_path(r, R, root), g)
            S = np.maximum(1, g[:, None] + rng.integers(-15, 15, (n, fam.K))).astype(float)
            out['sample'][(r.model, R)] = S
            p = fam.sample_path(r, R, base)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            np.savez(p, instances=np.array(names), makespan=S, feasible=np.ones(S.shape, bool),
                     seconds=np.zeros(n), K=fam.K, seed_test=PLAN.STORED_TIE_SEED,
                     **({} if fam.mask == 'nondelay' else dict(mask_mode=fam.mask)))
    os.makedirs(fam.sub('trace', base))
    for r in fam.released():
        for R in R8:
            trace_file(fam.trace_path(r.model, R, base), names, out['greedy'][(r.model, R)], rng,
                       fam.mask)
    for q in RULES:
        for R in R8:
            trace_file(fam.trace_path(q, R, base), names, out['stored'][(q, R)], rng, fam.mask)
    for t in PLAN.TIE_SEEDS[1:]:
        arrays = {(R, q): out['stored'][(q, R)] + rng.integers(-3, 4, n) for R in R8 for q in RULES}
        out['ties'][t] = arrays
        rule_folder(fam.ties_dir(t, base, root), arrays, fam.mask, t)
    rule_folder(fam.vali_folder(base), {(R, q): vali_values(R, q) for R in R8 for q in RULES},
                fam.mask, PLAN.STORED_TIE_SEED)
    return out


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A root with the records of the headline set and of the validation set and both
    families: the default one (non-delay) and the one under the active mask."""
    monkeypatch.setitem(PLAN.PUB_SET_INFO, HEAD, ('Sdten', N_INST, ()))
    root = str(tmp_path / 'repo')
    old = time.time() - 10000
    names = [f'inst{i:02d}' for i in range(N_INST)]
    nd = PLAN.Family('public', K=TEST_K)
    ac = PLAN.Family('public', K=TEST_K, mask='active')
    write_records(root, nd.data, names, old)
    write_records(root, nd.vali_data, names, old)
    return dict(root=root, tmp=tmp_path, nd=nd, ac=ac, names=names, old=old)


def with_default(t):
    t['dflt'] = build_family(t['root'], t['nd'], t['names'], 0, t['old'])
    return t


def with_active(t):
    t['act'] = build_family(t['root'], t['ac'], t['names'], 1, t['old'])
    return t


def run93(t, out, *extra, mask='active'):
    return P93.main(['--family', 'public', '--root', t['root'], '--out', str(out), '--boot', str(B),
                     '--K', str(TEST_K), *(['--mask', mask] if mask != 'nondelay' else []), *extra])


def outputs(out, label):
    return [os.path.join(str(out), p.format(f=label)) for p in P93.OUT_REL]


def macro_values(path):
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(path).read()))


def best_of_k(x, k):
    """Expected minimum of k of the K values drawn without replacement, from the order
    statistics: sum over j of x_(j) C(K - j, k - 1) / C(K, k)."""
    x = np.sort(np.asarray(x, float))
    K = len(x)
    return sum(x[j - 1] * math.comb(K - j, k - 1) for j in range(1, K - k + 2)) / math.comb(K, k)


def snapshot(top, skip):
    out = {}
    for d, _, fs in os.walk(top):
        if os.path.abspath(d).startswith(os.path.abspath(skip)):
            continue
        for f in fs:
            p = os.path.join(d, f)
            st = os.stat(p)
            out[p] = (st.st_size, st.st_mtime_ns)
    return out


def test_four_forms_by_hand_counterpart_macros_and_no_absolute_path(tree):
    t = with_active(with_default(tree))
    ac, R = t['ac'], 'LS'
    out = t['tmp'] / 'out'
    assert run93(t, out) == 0
    mac, js, md = outputs(out, 'public_act')
    vals = macro_values(mac)
    names = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(mac).read()))
    assert names and all(re.fullmatch(r'RcAct[A-Za-z]+', x) for x in names)
    A = t['act']
    rel = [r.model for r in ac.released()]
    g = np.stack([A['greedy'][(m, R)] for m in rel])
    S = [A['sample'][(m, R)] for m in rel]
    best4 = np.min([A['stored'][(q, R)] for q in RULES], axis=0)
    best16 = np.min([A['stored'][(q, R)] for q in RULES] +
                    [A['ties'][s][(R, q)] for s in PLAN.TIE_SEEDS[1:] for q in RULES], axis=0)
    vali = A['stored'][('MWKR', R)]              # lowest on the first VALI_SIZE records
    e4 = np.array([[best_of_k(s[i], 4) for i in range(N_INST)] for s in S])
    e16 = np.array([[best_of_k(s[i], 16) for i in range(N_INST)] for s in S])
    want = {'GreedyBestFour': 100 * (g - best4) / best4, 'GreedyValiRule': 100 * (g - vali) / vali,
            'SampleFour': 100 * (e4 - best4) / best4, 'SampleSixteen': 100 * (e16 - best16) / best16}
    for form, d in want.items():
        assert vals[f'RcActPol{form}{R}'] == P93.f2(d.mean()), form
    assert vals[f'RcActValiRule{R}'] == 'MWKR' and vals['RcActValiN'] == str(VALI_SIZE)
    assert vals['RcActSeedN'] == str(len(REG.PUB_ACTFAM_SEEDS)) and vals['RcActInstN'] == str(N_INST)
    res = json.load(open(js))
    assert res['settings']['mask'] == 'active' and res['settings']['label'] == 'public_act'
    assert res['settings']['rule_folder'] == ac.rule_folder and 'partial_report' not in res['settings']
    assert res['combos'][R]['vali']['all_records']['rule'] == 'FIFO'
    assert '-- do not hand-edit\n% public family under the active mask' in open(mac).read()
    assert open(md).read().startswith('# Comparison with dispatching rules, public family under '
                                      'the active mask')
    # the counterpart of every default macro: the same set with Rcp replaced by RcAct
    assert run93(t, out, mask='nondelay') == 0
    dflt = set(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', open(outputs(out, 'public')[0]).read()))
    assert {x.replace('RcAct', 'Rcp', 1) for x in names} == dflt
    for p in outputs(out, 'public_act'):
        text = open(p).read()
        assert str(t['tmp']) not in text and ROOT not in text
        assert '/home/' not in text and '/tmp/' not in text
        assert 'admissib' not in text and 'mask mode' not in text


def test_default_outputs_unchanged_by_the_family_under_the_active_mask(tree):
    t = with_default(tree)
    a, b = t['tmp'] / 'before', t['tmp'] / 'after'
    assert run93(t, a, mask='nondelay') == 0
    with_active(t)
    assert run93(t, b, mask='nondelay') == 0
    first = {os.path.basename(p): open(p, 'rb').read() for p in outputs(b, 'public')}
    assert {os.path.basename(p): open(p, 'rb').read() for p in outputs(a, 'public')} == first
    keep = snapshot(str(b), str(t['tmp'] / 'none'))
    assert run93(t, b) == 0
    after = snapshot(str(b), str(t['tmp'] / 'none'))
    assert {p: v for p, v in after.items() if p in keep} == keep
    assert sorted(set(after) - set(keep)) == sorted(outputs(b, 'public_act'))
    # the plan and the analysis of the default family see no file of the other family
    js = json.load(open(outputs(b, 'public')[1]))
    assert 'rules_act' not in json.dumps(js) and 'pubx-act' not in json.dumps(js)


def test_a_partial_campaign_refuses_and_the_partial_report_stays_under_out(tree):
    t = with_active(tree)
    ac = t['ac']
    base = os.path.join(t['root'], ac.campaign)
    out = t['tmp'] / 'out'
    assert run93(t, out) == 0
    shutil.rmtree(os.path.dirname(ac.ties_dir(52, base, t['root'])))
    with pytest.raises(SystemExit) as e:
        run93(t, out)
    assert 'REFUSING' in str(e.value) and 'missing' in str(e.value) and 'seed52' in str(e.value)
    assert not any(os.path.exists(p) for p in outputs(out, 'public_act'))   # removed
    for inside in (os.path.join(t['root'], 'results'), os.path.join(t['root'], 'manuscript'),
                   t['root']):
        with pytest.raises(SystemExit, match='never written into the repository root'):
            run93(t, inside, '--partial')
    assert not os.path.exists(os.path.join(t['root'], 'results'))
    before = snapshot(str(t['tmp']), str(t['tmp'] / 'part'))
    assert run93(t, t['tmp'] / 'part', '--partial') == 0
    assert snapshot(str(t['tmp']), str(t['tmp'] / 'part')) == before
    mac, js, md = outputs(t['tmp'] / 'part', 'public_act')
    assert not os.path.exists(mac)
    res = json.load(open(js))
    assert res['settings']['partial_report'] == P93.PARTIAL_NOTE and res['settings']['partial']
    assert any('seed52' in x for x in res['missing'])
    assert open(md).read().startswith(f'# {P93.PARTIAL_NOTE}\n')
    assert sorted(os.listdir(t['tmp'] / 'part')) == ['results']


def test_an_unfinished_run_refuses_and_the_partial_report_names_it(tree):
    t = with_active(tree)
    r = t['ac'].run_of('specialist', 'SW', 304)
    with open(os.path.join(t['root'], r.log), 'w') as f:
        f.write('[train_family] at update 900\n')
    with pytest.raises(SystemExit, match=f'the training of {re.escape(r.model)} has not finished'):
        run93(t, t['tmp'] / 'out')
    assert run93(t, t['tmp'] / 'part', '--partial') == 0
    res = json.load(open(outputs(t['tmp'] / 'part', 'public_act')[1]))
    assert f'{r.log} (training not finished)' in res['missing']


@pytest.mark.parametrize('damage', ['sample without mask', 'trace under the non-delay mask',
                                    'ties summary without mask', 'meta of another seed',
                                    'configuration of the non-delay mask'])
def test_a_file_of_another_mask_refuses(tree, damage):
    t = with_active(tree)
    ac = t['ac']
    base = os.path.join(t['root'], ac.campaign)
    if damage == 'sample without mask':
        p = ac.sample_path(ac.run_of('control', None, 302), 'LW', base)
        z = dict(np.load(p))
        z.pop('mask_mode')
        np.savez(p, **z)
        msg = 'records no mask'
    elif damage == 'trace under the non-delay mask':
        p = ac.trace_path('SPT', 'N', base)
        z = dict(np.load(p, allow_pickle=True))
        z['mask_mode'] = np.array('nondelay')
        np.savez(p, **z)
        msg = 'built under the nondelay mask'
    elif damage == 'ties summary without mask':
        p = os.path.join(ac.ties_dir(53, base, t['root']), 'summary_rules_xmask.md')
        open(p, 'w').write(open(p).read().replace('pairs: mask_mode=active\n', ''))
        msg = "does not record 'pairs: mask_mode=active'"
    elif damage == 'meta of another seed':
        write_meta(ac.ties_dir(52, base, t['root']), 'active', 51)
        msg = 'tie-break seed 51'
    else:
        r = ac.run_of('released', None, 305)
        p = os.path.join(t['root'], r.config)
        c = json.load(open(p))
        c['mask_mode'] = 'nondelay'
        json.dump(c, open(p, 'w'))
        msg = 'records the nondelay mask'
    with pytest.raises(SystemExit) as e:
        run93(t, t['tmp'] / 'out')
    assert 'REFUSING' in str(e.value) and msg in str(e.value), str(e.value)[:400]
    assert not any(os.path.exists(p) for p in outputs(t['tmp'] / 'out', 'public_act'))


def test_plan_statuses_on_the_synthetic_tree(tree, monkeypatch):
    t = with_active(tree)
    monkeypatch.setattr(PLAN, 'ROOT', t['root'])
    ac = PLAN.Family('public', K=TEST_K, mask='active')
    js = PLAN.jobs(ac, ratio=1.0)
    assert {j.status for j in js} == {'done'} and len(js) == 120 + 9 + 3 + 1
    assert PLAN.validation_rules(ac)[0]['LS'] == 'MWKR'
    write_meta(ac.ties_dir(51), 'active', 50)
    p = os.path.join(t['root'], ac.run_of('released', None, 303).log)
    open(p, 'w').write('[train_family] at update 10\n')
    st = {j.id: (j.status, getattr(j, 'reason', '')) for j in PLAN.jobs(ac, ratio=1.0)}
    assert st['ties/seed51'][0] == 'invalid' and 'tie-break seed 50' in st['ties/seed51'][1]
    assert st[f'sample/{ac.run_of("released", None, 303).model}_S'][0] == 'done'
    # a job that has not run waits for the end of its run's training
    os.remove(ac.sample_path(ac.run_of('released', None, 303), 'S'))
    st = {j.id: (j.status, getattr(j, 'reason', '')) for j in PLAN.jobs(ac, ratio=1.0)}
    assert st[f'sample/{ac.run_of("released", None, 303).model}_S'] == \
        ('blocked', f'the training of {ac.run_of("released", None, 303).model} has not finished')


# ================================================================== the chain step
def helper(root, monkeypatch):
    """The Python helper of run_analysis.sh, its functions only, reading the tree root."""
    src = open(CHAIN).read()
    code = re.search(r"read -r -d '' HELPER <<'PYEOF'\n(.*?)\nPYEOF", src, re.S).group(1)
    code = code.rsplit('\nmain()', 1)[0]
    ns_ = {'__name__': 'figures_helper'}
    exec(compile(code, 'run_analysis.sh', 'exec'), ns_)
    monkeypatch.setattr(PLAN, 'ROOT', root)
    return ns_


def test_chain_step_names_the_six_analyses_and_their_outputs():
    src = open(CHAIN).read()
    steps = re.search(r'^STEPS=\(([^)]*)\)', src, re.M).group(1).split()
    assert steps[steps.index('public_act') + 1] == 'rulecmp_public_act'
    tags = re.search(r'^PUB_TAGS=\(([^)]*)\)', src, re.M).group(1).split()
    assert [t.lower() for t in tags] == [PLAN.PUB_SET_INFO[s][0].lower()
                                         for s in PLAN.PUB_SET_INFO if s != HEAD]
    assert 'run_cmd scripts/p93_rule_comparison.py --family public --mask active || return 1' in src
    assert 'run_cmd scripts/p93_rule_comparison.py --family public --mask active --set ' \
           '"${PUB_SET[$t]}" || return 1' in src
    labels = ['public_act'] + [f'public_act_{t}' for t in tags]
    # every case statement names the step before the glob of the other public sets
    for fn in ('title_of', 'deps_of', 'pre_text', 'check_of', 'cmds_of', 'outs_of'):
        body = re.search('^' + fn + r'\(\) \{[^\n]*\n(.*?)\n\}\n', src, re.S | re.M).group(1)
        assert body.index('rulecmp_public_act)') < body.index('rulecmp_public_*)'), fn
    assert re.search(r'^    rulecmp_public_act\) echo "public_act";;', src, re.M)
    assert "'rulecmp-act-state': rulecmp_act_state" in src
    assert "'manuscript/macros_rulecmp_public_act.tex'" in src
    assert "f'manuscript/macros_rulecmp_public_act_{t}.tex'" in src
    # the dry run prints the outputs (its precondition reads the repository, whatever it holds)
    out = subprocess.run(['bash', CHAIN], cwd=ROOT, capture_output=True, text=True, timeout=120,
                         env=dict(os.environ, DRY='1', ONLY='rulecmp_public_act'))
    assert '] rulecmp_public_act: rule comparison in three forms under the active mask' \
        in out.stdout, out.stdout[-2000:] + out.stderr[-2000:]
    writes = re.search(r'    writes: (.*)\n', out.stdout).group(1).split()
    assert writes == [p for lab in labels for p in (f'manuscript/macros_rulecmp_{lab}.tex',
                                                    f'results/rulecmp_{lab}.json',
                                                    f'results/rulecmp_{lab}.md')]
    block = out.stdout.split('] rulecmp_public_act: ', 1)[1].split('\n\n', 1)[0]
    assert 'admissib' not in block and 'rule mask' not in block


def test_chain_state_every_file_or_none(tree, monkeypatch, capsys):
    t = with_default(tree)
    monkeypatch.setattr(PLAN, 'PUB_SET_INFO', {HEAD: ('Sdten', N_INST, ())})
    H = helper(t['root'], monkeypatch)
    assert H['rulecmp_act_state']() is False
    assert 'no output of a job on any of the six public sets' in capsys.readouterr().out
    with_active(t)
    assert H['rulecmp_act_state']() is True
    assert 'every job of the six public sets done' in capsys.readouterr().out
    ac = t['ac']
    shutil.rmtree(os.path.dirname(ac.ties_dir(53, os.path.join(t['root'], ac.campaign), t['root'])))
    assert H['rulecmp_act_state']() is False
    text = capsys.readouterr().out
    assert 'partial, 1 of 1 sets not ready' in text and 'ties 2 done, 1 open' in text
