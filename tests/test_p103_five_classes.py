"""Tests for scripts/p103_five_classes.py.

Every computation at five classes runs in a subprocess, so this process keeps the four
classes of the shared modules throughout.

  * four-class identity: level default, evaluated by the script's code path with the
    class count set to 4, reproduces the stored exact-mask arrays of all seven rules bit
    for bit (10x25, S and LSW, three instances; 50x25n30, LSW, one instance);
  * a hand-checked five-class case on two machines, where the changeover between
    class 0 and the fifth class decides the makespan: the environment's makespans, the
    changeover hours and the validator's verdicts equal the hand values, and a schedule
    that shortens the fifth-class changeover is rejected;
  * a process at four classes refuses the five-class hand descriptor in check C1 and
    in the environment, so a missing override fails loudly;
  * no leak: a fresh process imports four classes after the five-class runs, the
    override refuses a module that binds another class count and a module that copied
    the count at import time, and every shared module file equals its committed blob;
  * the legacy policy jointblind-v1 (32786 parameters) reads the five-class input with
    the input shapes of the four-class input; only the setup entries of the token and
    the setup inputs change, and its schedule passes the five-class validator;
  * the build rule: a [6, 5] matrix with the stored block and the stored cold-start row
    kept (the cold-start row last), symmetric, the new entries in the generator's
    ranges, floor(J / 4) whole modules in class 4, one matrix per instance across the
    four combinations, every other field unchanged, and the written files equal to the
    rule;
  * infer refuses a missing result and, on synthetic results, produces the comparisons,
    the shifts and unique, prefixed macros, with the default-level policy read from the
    script's files or from stored arrays;
  * no existing script reads the new folder, and the frozen instance and descriptor
    folders and the stored rule arrays are unchanged by build and evaluate.

Run from the repository root:
  python -m pytest tests/test_p103_five_classes.py -q -p no:cacheprovider
CPU only.
"""
import fnmatch
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(ROOT, 'scripts'))
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402,F401  (parses argv at import)
    import data_utils  # noqa: E402,F401
    mods = {}
    for name in ('p101_mask_controls', 'p102_scale_sensitivity', 'p103_five_classes'):
        if name in sys.modules:
            mods[name] = sys.modules[name]
            continue
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(ROOT, 'scripts', f'{name}.py'))
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name]
        spec.loader.exec_module(mods[name])
finally:
    sys.argv = _argv
configs.device = 'cpu'
P101, P102, P103 = (mods['p101_mask_controls'], mods['p102_scale_sensitivity'],
                    mods['p103_five_classes'])

from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402

PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONDONTWRITEBYTECODE='1')
ENV.pop('PACT_ENV', None)
LEGACY = '10x25+family+jointblind-v1'
FROZEN = ('data/FAMILY/10x25', 'data/FAMILY/50x25n30', 'data/FAMILY/80x25n30',
          'data/FAMILY/50x25', 'data/FAMILY/80x25', 'data/PPVC/10x25+ppvc-mixed',
          'data/PPVC/50x25+ppvc-mixed-n30', 'data/PPVC/80x25+ppvc-mixed-n30')
STORED = ('10x25', '50x25n30', '80x25n30')
SHARED = ('constraint_family/descriptor.py', 'constraint_family/fjsp_env_family.py',
          'constraint_family/checks.py', 'constraint_family/reference_sim.py',
          'constraint_family/cpsat_family.py', 'constraint_family/cpsat_arms.py',
          'constraint_family/public_rules.py', 'constraint_family/public_types.py',
          'baselines/memetic/instance.py', 'eval_ga_family.py', 'eval_family.py',
          'params.py', 'common_utils.py', 'scripts/p101_mask_controls.py',
          'scripts/p102_scale_sensitivity.py', 'scripts/p72_aware_rules.py')
NEW_FOLDER = 'five_p103'


def _run(args, timeout=900):
    r = subprocess.run([PY, '-B'] + args, cwd=ROOT, env=ENV, capture_output=True,
                       text=True, timeout=timeout)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    return r.stdout


def _check_json(out):
    line = [x for x in out.splitlines() if x.startswith('CHECK ')]
    assert len(line) == 1, out[-2000:]
    return json.loads(line[0][len('CHECK '):])


def _tree_digest(rel):
    h = hashlib.sha256()
    base = os.path.join(ROOT, rel)
    n = 0
    for dp, dns, fns in os.walk(base):
        dns.sort()
        for fn in sorted(fns):
            p = os.path.join(dp, fn)
            h.update(os.path.relpath(p, base).encode())
            with open(p, 'rb') as fh:
                h.update(fh.read())
            n += 1
    return h.hexdigest(), n


def _stored_rule_digest():
    """sha256 of the stored rule arrays the study reads (the policy arrays of the exact
    campaign may still be written while the tests run, so they are left out)."""
    out = {}
    for size in STORED:
        for r in P103.RULES7:
            for c in P101.R8:
                p = os.path.join(ROOT, 'test_results', 'FAMILY', f'{size}_xmask',
                                 f'Result_{r}_{c}.npy')
                out[p] = P101.sha256_file(p)
    return out


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """build, evaluate at both levels and check at both class counts, each in its own
    process, with the frozen digests before and after."""
    root = str(tmp_path_factory.mktemp('p103'))
    before = ({d: _tree_digest(d) for d in FROZEN}, _stored_rule_digest())
    _run(['scripts/p103_five_classes.py', 'build', '--root', root, '--sizes', '10x25',
          '--n', '3'])
    for level in ('default', 'five'):
        _run(['scripts/p103_five_classes.py', 'evaluate', '--level', level, '--root', root,
              '--sizes', '10x25', '--combos', 'S,LSW', '--n', '3', '--arms', 'rules'])
    _run(['scripts/p103_five_classes.py', 'evaluate', '--level', 'default', '--root', root,
          '--sizes', '50x25n30', '--combos', 'LSW', '--n', '1', '--arms', 'rules'])
    chk = {K: _check_json(_run(['scripts/p103_five_classes.py', 'check', '--classes',
                                str(K), '--policy', LEGACY])) for K in (4, 5)}
    after = ({d: _tree_digest(d) for d in FROZEN}, _stored_rule_digest())
    return dict(root=root, before=before, after=after, check=chk)


def _unit(root, level, size, combo, label):
    p = P103.result_path(root, 'exact', level, size, combo, label)
    with np.load(p) as z:
        arr = {k: np.asarray(z[k]) for k in z.files}
    with open(p[:-4] + '.json') as f:
        return arr, json.load(f)


# ---------------------------------------------------------------------------
# 1. four-class identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('size,combo,n', [('10x25', 'S', 3), ('10x25', 'LSW', 3),
                                          ('50x25n30', 'LSW', 1)])
def test_four_class_identity(runs, size, combo, n):
    for r in P103.RULES7:
        arr, rec = _unit(runs['root'], 'default', size, combo, r)
        stored = P101.stored_rule(size, r, combo, n)
        assert np.array_equal(arr['makespan'], stored), (size, combo, r)
        assert rec['class_count'] == 4 and rec['validated'] == n and rec['violations'] == 0
        assert np.all(arr['setup_new_h'] == 0) and np.all(arr['setup_h'] > 0)


def test_five_level_runs_at_five_classes(runs):
    for combo in ('S', 'LSW'):
        for r in P103.RULES7:
            arr, rec = _unit(runs['root'], 'five', '10x25', combo, r)
            assert rec['class_count'] == 5 and rec['validated'] == 3
            assert rec['violations'] == 0
            assert np.all(arr['setup_new_h'] > 0)
            assert np.all(arr['setup_new_h'] < arr['setup_h'])


# ---------------------------------------------------------------------------
# 2. the hand-checked five-class case
# ---------------------------------------------------------------------------

def test_hand_case_at_five_classes(runs):
    """Classes: op0 class 0 (m0, 2 h), op1 class 4 (m0, 3 h), op2 class 4 (m1, 1 h).
    Setups: cold start 1 h into class 0 and 2 h into class 4, class 0 to class 4 and
    back c h. By hand, machine 0 in the order op0, op1:
        setup [0, 1), op0 [1, 3), setup [3, 3 + c), op1 [3 + c, 6 + c)
    and machine 1: setup [0, 2), op2 [2, 3). Makespan 6 + c, which the fifth-class
    changeover c decides: 11 h at c = 5, 8 h at c = 2. Changeover hours 1 + c + 2,
    of which c + 2 involve class 4. In the order op1, op0: op1 [2, 5), op0
    [5 + c, 7 + c), makespan 7 + c, changeover 2 + c + 2, all of it involving class 4.
    Shortening the class 0 to class 4 changeover by 2 h (op1 on [6, 9) at c = 5) puts
    its setup block on [1, 6), inside op0's processing, so the validator must reject it."""
    h = runs['check'][5]['hand']
    assert h['setup_shape'] == [6, 5]
    for c in (5, 2):
        rows = h[f'cross{c}']
        assert rows, 'the mask admits no complete schedule'
        for row in rows:
            assert row['violations'] == 0
            if row['m0_order'] == [0, 1]:
                assert row['makespan'] == 6 + c
                assert row['changeover_h'] == [1 + c + 2, c + 2]
                assert row['op_ct'] == [3.0, 6.0 + c, 3.0]
            else:
                assert row['m0_order'] == [1, 0] and row['makespan'] == 7 + c
        assert min(r['makespan'] for r in rows) == 6 + c
        rev = h[f'reverse{c}']
        assert rev['makespan'] == 7 + c and rev['violations'] == []
        assert rev['changeover_h'] == [2 + c + 2, 2 + c + 2]
    assert len(h['shortened']) == 1
    assert 'mch 0: op 1 setup+proc block overlaps predecessor' in h['shortened'][0]
    assert len(h['reverse_shortened']) == 1
    assert 'mch 0: op 0 setup+proc block overlaps predecessor' in h['reverse_shortened'][0]
    ov = runs['check'][5]['override']
    assert ov['class_count'] == 5
    assert ov['probe'] == {'setup_shape': [6, 5], 'cold_start_row': 5, 'c1': 'PASS',
                           'c1_other_count': 'FAIL', 'spt_makespan': 11.0, 'violations': 0}
    assert all(v == 5 for d in runs['check'][5]['constants'].values() for v in d.values())


def test_four_class_process_refuses_the_five_class_hand_case(runs):
    h = runs['check'][4]['hand']
    assert h['c1_five'] == 'FAIL'
    assert h['env_five'] == 'refused: ValueError'
    assert runs['check'][4]['override']['probe']['setup_shape'] == [5, 4]


# ---------------------------------------------------------------------------
# 3. no leak
# ---------------------------------------------------------------------------

def test_fresh_process_has_four_classes(runs):
    code = ('import sys; sys.argv = sys.argv[:1] + ["--device", "cpu"]; '
            'import json; from constraint_family import descriptor, checks, reference_sim, '
            'fjsp_env_family as fe; print(json.dumps([descriptor.N_CLASSES, '
            'descriptor.IDLE_CLASS, checks.N_CLASSES, checks.IDLE_CLASS, '
            'reference_sim.IDLE_CLASS, fe.N_CLASSES, fe.IDLE_CLASS]))')
    out = _run(['-c', code])
    assert json.loads(out.strip().splitlines()[-1]) == [4] * 7
    for m, names in P103.PATCHED.items():
        mod = sys.modules.get(m)
        if mod is not None:
            assert all(getattr(mod, k) == 4 for k in names), m


def test_override_refuses_foreign_bindings():
    code = r'''
import sys, types, json
sys.path.insert(0, "scripts")
sys.argv = sys.argv[:1]
import p103_five_classes as P
out = {}
for case in ("foreign", "captured", "clean"):
    for m, names in P.PATCHED.items():
        mod = sys.modules.get(m)
        if mod is not None:
            for k in names:
                setattr(mod, k, 4)
    P._SAVED.clear()
    sys.modules.pop("fake_holder", None)
    sys.modules.pop("constraint_family.public_types", None)
    if case == "foreign":
        fake = types.ModuleType("fake_holder"); fake.N_CLASSES = 4
        sys.modules["fake_holder"] = fake
    if case == "captured":
        sys.modules["constraint_family.public_types"] = types.ModuleType("public_types")
    try:
        with P.class_count(5) as info:
            out[case] = "accepted " + str(info["class_count"])
    except SystemExit as e:
        out[case] = str(e)
    sys.modules.pop("fake_holder", None)
    sys.modules.pop("constraint_family.public_types", None)
    out[case + "_after"] = [getattr(sys.modules[m], k) for m, ns in P.PATCHED.items()
                            for k in ns]
print("CHECK " + json.dumps(out))
'''
    out = _check_json(_run(['-c', code]))
    assert out['foreign'].startswith('REFUSING: modules bind another class count'), out
    assert 'fake_holder' in out['foreign']
    assert out['captured'].startswith('REFUSING:') and 'public_types' in out['captured']
    assert out['clean'] == 'accepted 5'
    for case in ('foreign', 'captured', 'clean'):
        assert out[case + '_after'] == [4] * 7


def has_commit(rev):
    """Whether the folder of the repository is the top of a git work tree that holds the
    commit rev (False without git, outside a work tree, or without that commit)."""
    try:
        top = subprocess.run(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT,
                             capture_output=True, text=True)
        if top.returncode or os.path.realpath(top.stdout.strip()) != os.path.realpath(ROOT):
            return False
        return subprocess.run(['git', 'cat-file', '-e', rev + '^{commit}'], cwd=ROOT,
                              capture_output=True).returncode == 0
    except OSError:
        return False


def _blob(path):
    with open(os.path.join(ROOT, path), 'rb') as f:
        data = f.read()
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


@pytest.mark.skipif(not has_commit('HEAD'),
                    reason='needs the git history of the repository (the files of HEAD), which '
                           'the release does not hold')
def test_shared_files_equal_their_committed_blobs(runs):
    hg = subprocess.run(['git', 'ls-files', 'backbones/hgnn'], cwd=ROOT, capture_output=True,
                        text=True, check=True).stdout.split()
    paths = list(SHARED) + [p for p in hg if p.endswith('.py')]
    tree = subprocess.run(['git', 'ls-tree', '-r', 'HEAD', '--'] + paths, cwd=ROOT,
                          capture_output=True, text=True, check=True).stdout
    head = {ln.split('\t')[1]: ln.split()[2] for ln in tree.splitlines()}
    assert set(head) == set(paths)
    changed = [p for p in paths if _blob(p) != head[p]]
    assert not changed, changed


# ---------------------------------------------------------------------------
# 4. the policy reads the five-class input
# ---------------------------------------------------------------------------

def test_policy_accepts_five_classes(runs):
    p4, p5 = runs['check'][4]['policy'], runs['check'][5]['policy']
    assert p4['params'] == p5['params'] == P101.EXPECT_PARAMS == 32786
    assert p4['shapes'] == p5['shapes']
    assert p4['setup_shape'] == [5, 4] and p5['setup_shape'] == [6, 5]
    assert p4['violations'] == 0 and p5['violations'] == 0
    assert p5['makespan'] > 0
    t4, t5 = np.array(p4['token']), np.array(p5['token'])
    other = [i for i in range(len(t4)) if i not in (6, 7, 8)]
    assert np.array_equal(t4[other], t5[other])
    assert not np.array_equal(t4[6:9], t5[6:9])
    assert len(p4['mch_setup_channel']) == len(p5['mch_setup_channel'])


# ---------------------------------------------------------------------------
# 5. the build rule
# ---------------------------------------------------------------------------

def _one(size, combo, name):
    from ppvc_instance_generator import load_instance
    with open(P102.stored_desc_path(size, combo, name)) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(os.path.join(ROOT, P101.SIZES[size]['base'], name))
    return name, jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


@pytest.mark.parametrize('size,name', [('10x25', 'instance_000'), ('50x25n30', 'instance_007'),
                                       ('80x25n30', 'instance_029')])
def test_build_rule(size, name):
    cells = {c: _one(size, c, name) for c in P103.COMBOS}
    first = None
    for combo, (_, jl, pt, meta, d4) in cells.items():
        d5, info = P103.five_class_descriptor(d4, jl, pt, size, name)
        again, _ = P103.five_class_descriptor(d4, jl, pt, size, name)
        S4, S5 = np.asarray(d4.setup), d5.setup
        assert S5.shape == (6, 5)
        assert np.array_equal(S5[:4, :4], S4[:4])
        assert np.array_equal(S5[5, :4], S4[4])
        assert np.array_equal(S5[:5, :5], S5[:5, :5].T)
        pb = float(pt[pt > 0].mean())
        assert all(max(1, round(0.5 * pb)) <= S5[a, 4] <= max(1, round(1.0 * pb))
                   for a in range(4))
        for v in (S5[4, 4], S5[5, 4]):
            assert max(1, round(0.25 * pb)) <= v <= max(1, round(0.5 * pb))
        J = len(jl)
        assert len(info['new_class_jobs']) == J // 4
        job_of_op = np.repeat(np.arange(J), jl)
        assert np.array_equal(d5.op_class == 4, np.isin(job_of_op, info['new_class_jobs']))
        keep = d5.op_class != 4
        assert np.array_equal(d5.op_class[keep], d4.op_class[keep])
        assert d5.active == d4.active and np.array_equal(d5.lag, d4.lag)
        assert d5.windows == d4.windows and d5.horizon == d4.horizon
        assert d5.to_json() == again.to_json()
        P103.check_unchanged(d4, d5, info)
        if first is None:
            first = (S5, info['new_class_jobs'])
        assert np.array_equal(first[0], S5) and first[1] == info['new_class_jobs']
    assert P103.five_stream(size, name) == [20261113, 5, P101.SIZES[size]['code'],
                                            P101.instance_index(name)]


def test_written_descriptors_equal_the_rule(runs):
    root = runs['root']
    with open(P103.manifest_path(root, '10x25')) as f:
        man = json.load(f)
    assert man['n_classes'] == 5 and man['new_class_modules'] == 2
    for combo in P103.COMBOS:
        for name, jl, pt, meta, d4 in P101.load_cell('10x25', combo, 3):
            with open(P103.five_desc_path(root, '10x25', combo, name)) as f:
                rec = json.load(f)
            d5, info = P103.five_class_descriptor(d4, jl, pt, '10x25', name)
            want = d5.to_json()
            assert {k: rec[k] for k in want} == want
            assert rec['level']['new_class_jobs'] == info['new_class_jobs']
            assert rec['level']['source_sha256'] == P101.sha256_file(
                P102.stored_desc_path('10x25', combo, name))


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

def _args(root, out, arms, n=3, seeds=(301, 302), default_policy='run'):
    class A:
        pass
    a = A()
    a.root, a.out, a.partial, a.n, a.boot = root, out, True, n, 200
    a.levels, a.sizes, a.combos = list(P103.LEVELS), ['10x25'], list(P103.COMBOS)
    a.arms, a.seeds = list(arms), list(seeds)
    a.environment, a.default_policy = 'exact', default_policy
    return a


def _write_unit(root, level, combo, label, ms, names):
    p = P103.result_path(root, 'exact', level, '10x25', combo, label)
    n = len(ms)
    P101.save_npz_atomic(p, makespan=np.asarray(ms, float), lb0=np.ones(n),
                         util=np.full(n, 0.2), ms_over_lb0=np.ones(n),
                         setup_h=np.full(n, 100.0),
                         setup_new_h=np.full(n, 25.0 if level == 'five' else 0.0),
                         amch=np.zeros((n, 4), dtype=np.int16))
    P101.write_json_atomic(p[:-4] + '.json', {
        'level': level, 'size': '10x25', 'regime': combo, 'label': label,
        'instances': names, 'class_count': P103.LEVEL_K[level], 'validated': n,
        'violations': 0})


def _manifest(root, names):
    P101.write_json_atomic(P103.manifest_path(root, '10x25'), {
        'instances': names, 'new_class_modules': 2, 'modules': 10,
        'cross_new_mean_h': 3.3, 'same_new_mean_h': 1.0, 'cold_new_mean_h': 2.0,
        'c4_warn_share': {c: 0.0 for c in P103.COMBOS}})


def test_infer_refuses_a_missing_result(tmp_path):
    args = _args(str(tmp_path / 'empty'), str(tmp_path / 'out'), ['rules'])
    _manifest(args.root, P101.names_of('10x25', 3))
    with pytest.raises(SystemExit) as e:
        P103.infer(args)
    assert str(e.value).startswith('refused'), str(e.value)


@pytest.mark.parametrize('default_policy', ['run', 'stored'])
def test_infer_on_synthetic_results(tmp_path, monkeypatch, default_policy):
    """Default-level rules are the stored arrays (infer checks them); level five
    multiplies them by 1.03; the policy is 1% shorter than the specialist at four
    classes and 0.5% longer at five. The synthetic default-level policy arrays are read
    from the script's files or, with default_policy stored, the released policy's from a
    stand-in for the exact campaign's arrays and the specialist's from the files."""
    root, n, seeds = str(tmp_path / 'r'), 3, (301, 302)
    names = P101.names_of('10x25', n)
    rng = np.random.default_rng(0)
    stand_in = {}
    for combo in P103.COMBOS:
        for level, f in (('default', 1.0), ('five', 1.03)):
            for r in P103.RULES7:
                _write_unit(root, level, combo, r, f * P101.stored_rule('10x25', r, combo, n),
                            names)
            base = f * P101.stored_rule('10x25', 'SPT', combo, n)
            for s in seeds:
                spec = base * (1 + 0.01 * rng.random(n))
                pol = spec * (0.99 if level == 'default' else 1.005)
                for arm, ms in (('spec', spec), ('pact', pol)):
                    label = 'greedy+' + P103.model_name(arm, combo, s)
                    if level == 'default' and default_policy == 'stored' and arm == 'pact':
                        stand_in[(combo, label)] = ms
                    else:
                        _write_unit(root, level, combo, label, ms, names)
    _manifest(root, names)
    monkeypatch.setattr(P103, 'stored_policy_array',
                        lambda size, combo, label, env, n: stand_in.get((combo, label)))
    res = P103.infer(_args(root, str(tmp_path / 'out'), list(P103.ARM_CHOICES), n, seeds,
                           default_policy))
    S = res['sizes']['10x25']
    P = S['policy']
    assert P['default']['pooled']['pol_spec']['mean'] == pytest.approx(-1.0, abs=1e-6)
    assert P['five']['pooled']['pol_spec']['mean'] == pytest.approx(0.5, abs=1e-6)
    assert P['shift']['pol_spec']['pooled']['mean'] == pytest.approx(1.5, abs=1e-3)
    assert set(P['five']['rows']['pol_rule']['LSW']) >= {'verdict', 'p_t_holm'}
    for c in P103.COMBOS:
        assert S['combos'][c]['rule_shift']['spt']['mean'] == pytest.approx(3.0)
        assert all(v == 'equal' for v in
                   S['combos'][c]['levels']['default']['stored_check'].values())
        assert S['combos'][c]['levels']['five']['setup_new_share_spt'] == pytest.approx(25.0)
    per_unit = 2 * len(P103.COMBOS) * len(P103.RULES7) * n
    pol_units = (3 if default_policy == 'stored' else 4) * len(P103.COMBOS) * len(seeds) * n
    src = S['combos']['LSW']['levels']['default']['policy_source']
    assert src['pact'] == ['stored' if default_policy == 'stored' else 'run'] * len(seeds)
    assert src['spec'] == ['run'] * len(seeds)
    assert S['validated'] == per_unit + pol_units and S['violations'] == 0
    txt = open(tmp_path / 'out' / 'manuscript' / 'macros_fiveclass.tex').read()
    mnames = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(mnames) == len(set(mnames)) > 100
    assert all(re.fullmatch(r'Fcls[A-Za-z]+', x) for x in mnames)


def test_units_needed_leave_out_stored_default_runs(tmp_path, monkeypatch):
    have = str(tmp_path / 'stored.npy')
    np.save(have, np.zeros(3))
    pact = 'greedy+' + P103.model_name('pact', 'S', 301)
    monkeypatch.setattr(P103, 'stored_policy_path', lambda size, combo, label, env:
                        have if label == pact else str(tmp_path / 'missing.npy'))

    class A:
        default_policy, environment = 'stored', 'exact'
    every = P103.all_units(['default', 'five'], ['10x25'], ['S'], ['pact', 'spec', 'rules'],
                           [301, 302])
    kept = P103.units_needed(every, A)
    assert [u for u in every if u not in kept] == [
        ('default', '10x25', 'S', 'pact', 301, pact)]
    A.default_policy = 'run'
    assert P103.units_needed(every, A) == every


# ---------------------------------------------------------------------------
# folders
# ---------------------------------------------------------------------------

def _scanned_files():
    out = []
    for top in ('.', 'scripts', 'tests', 'constraint_family', 'baselines', 'backbones',
                'model'):
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
    """The new folder holds <rule>.npz, greedy+<run>.npz, their JSON records and
    descriptor JSON files at depth 4 to 7 below test_results/FAMILY. (1) No other script
    names it. (2) Every literal path pattern below test_results/FAMILY with a wildcard
    fails to match any of its file paths (fnmatch, whose * also crosses '/'). (3) The
    name starts with no size and ends in no registry suffix, so a folder built from a
    size key or a registry folder plus a suffix never equals it."""
    own = {os.path.normpath(os.path.join(ROOT, p)) for p in (
        'scripts/p103_five_classes.py', 'tests/test_p103_five_classes.py')}
    base = 'test_results/FAMILY/' + NEW_FOLDER
    examples = [base + '/results/exact/default/10x25/LSW/SPT.npz',
                base + '/results/exact/five/80x25n30/S/SPT.json',
                base + '/results/exact/five/50x25n30/SW/greedy+10x25+family+xmask-v1-s301.npz',
                base + '/results/legacy/five/10x25/LS/greedy+10x25+family+spec-LS.npz',
                base + '/data/five/50x25n30/desc/LSW/instance_000.desc.json',
                base + '/data/five/80x25n30/manifest.json']
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


def test_frozen_folders_unchanged_by_a_run(runs):
    (fb, rb), (fa, ra) = runs['before'], runs['after']
    assert all(n > 0 for _, n in fb.values())
    assert fa == fb
    assert ra == rb and len(rb) == len(STORED) * len(P103.RULES7) * 8
