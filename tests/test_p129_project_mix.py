"""Tests for scripts/p129_project_mix.py (single-shell project mixes, evaluation only).

  * levels, sizes, seeds and folders: the two single-shell levels at 10 modules are p87's
    levels mix_rc and mix_steel, the new sets at 50 and 80 modules use the presets of the
    stored single-shell sets, 30 instances each, and seed blocks that collide with no seed
    block in use; the design has 816 result files for seeds 301-305, and a colliding seed
    block is refused;
  * infer refuses when one makespan of a 10-module unit differs from p87's stored unit, and
    counts 50 of 50 equal makespans on an unchanged copy;
  * no existing script reads the new folder;
  * on synthetic results infer recovers the planted differences, gives the verdicts, counts
    every validated schedule and writes unique macro names, letters only with the prefix Pmx;
    the name check refuses a clash with a macro of manuscript/ and a name that is not letters;
  * plan partitions the units into K parts (each unit in exactly one part, for K = 1, 8 and
    16), balances them within one unit's cost, prints one evaluate command per part and skips
    a finished unit.

Run from the repository root on one core:
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 taskset -c 3 python -m pytest -q \
      -p no:cacheprovider tests/test_p129_project_mix.py
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
    import p129_project_mix as P  # noqa: E402  (sets sys.argv for params.py at import)
finally:
    sys.argv = _argv

SCRIPT = os.path.join(ROOT, 'scripts', 'p129_project_mix.py')
PY = sys.executable
NEW_FOLDER = os.path.basename(P.DEFAULT_ROOT)


def env(**kv):
    e = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-305', CUDA_VISIBLE_DEVICES='',
             OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    for k in ('PACT_OUT', 'PACT_RESULTS', 'RESULTS', 'SEEDS'):
        e.pop(k, None)
    e.update({k: str(v) for k, v in kv.items()})
    return e


# ---------------------------------------------------------------------------
# levels, seeds and folders
# ---------------------------------------------------------------------------

def test_levels_sizes_seeds_and_folders():
    assert P.LEVELS == ('default', 'mix_rc', 'mix_steel')
    assert P.SIZES == ('10x25', '50x25n30', '80x25n30')
    assert P.SEEDS == (301, 302, 303, 304, 305)
    assert P.RULES7 == ('FIFO', 'MOR', 'SPT', 'MWKR', 'ECT', 'SSPT', 'LLF')
    # the 10-module levels are p87's, on the stored single-shell sets
    for lv in P.MIX_LEVELS:
        base = P.P87.LEVELS[lv]['base']
        assert base == f'data/PPVC/10x25+ppvc-{P.MIX[lv]}'
        meta = json.load(open(os.path.join(ROOT, base, 'dataset_meta.json')))
        assert meta['ppvc_mix'] == P.MIX[lv] and meta['n_instances'] == P.N_STORED_MIX
    # four new blocks of 30 seeds, none overlapping a block in use or each other
    assert sorted(P.SEED0) == sorted((lv, sz) for lv in P.MIX_LEVELS for sz in P.NEW_SIZES)
    rec = P.check_seed_blocks()
    assert len(rec['new']) == 4 and all(v[1] == 30 for v in rec['new'].values())
    assert 'data/PPVC/10x25+ppvc-rc_project' in rec['used']
    # every result file of the design, under the study's own folder
    units = P.all_units(P.MIX_LEVELS, P.SIZES, P.R8, P.ARM_CHOICES, P.SEEDS)
    assert len(units) == len(set(units)) == 2 * 3 * 8 * (2 * 5 + 7) == 816
    assert all(P.unit_n(u) == (50 if u[1] == '10x25' else 30) for u in units)
    assert P.result_path(P.DEFAULT_ROOT, 'mix_rc', '50x25n30', 'LS', 'SPT') == \
        'test_results/FAMILY/mix_p129/results/exact/mix_rc/50x25n30/LS/SPT.npz'
    assert P.base_dir(P.DEFAULT_DATA, 'mix_steel', '80x25n30') == \
        'test_results/FAMILY/mix_p129/data/PPVC/80x25+ppvc-steel_project'
    assert P.desc_dir(P.DEFAULT_DATA, 'mix_rc', '50x25n30') == \
        'test_results/FAMILY/mix_p129/data/FAMILY/50x25_mix_rc'
    labels = {u[5] for u in units if u[3] != 'rule'}
    assert 'greedy+10x25+family+xmask-v1-s305' in labels
    assert 'greedy+10x25+family+xmask-solo-LSW-s301' in labels
    # level default is read, never evaluated
    assert not P.all_units(['default'], P.SIZES, P.R8, P.ARM_CHOICES, P.SEEDS)


def test_a_colliding_seed_block_is_refused(monkeypatch):
    monkeypatch.setitem(P.SEED0, ('mix_rc', '50x25n30'), 90010)       # inside rc_project
    with pytest.raises(P.Refused, match='new seed blocks collide'):
        P.check_seed_blocks()
    monkeypatch.setitem(P.SEED0, ('mix_rc', '50x25n30'), 155010)      # inside steel at 50
    with pytest.raises(P.Refused, match='mix_rc 50x25n30 and mix_steel 50x25n30'):
        P.check_seed_blocks()


# ---------------------------------------------------------------------------
# the control against p87
# ---------------------------------------------------------------------------

P87_FIFO = P.P87.result_path(P.P87.DEFAULT_ROOT, 'exact', 'mix_rc', 'N', 'FIFO')


def _copy_p87_unit(root, level, combo, label, change=None):
    src = P.P87.result_path(P.P87.DEFAULT_ROOT, 'exact', level, combo, label)
    with np.load(src) as z:
        ms = np.asarray(z['makespan'], dtype=float)
    names = json.load(open(src[:-4] + '.json'))['instances']
    if change is not None:
        ms[change] += 1.0
    dst = P.result_path(root, level, '10x25', combo, label)
    P.P101.save_npz_atomic(dst, makespan=ms)
    P.write_json_atomic(dst[:-4] + '.json', {
        'level': level, 'size': '10x25', 'regime': combo, 'label': label, 'instances': names,
        'validated': len(names), 'violations': 0})
    return dst


def _infer_args(root, out, data_root, levels, sizes, seeds=(301, 302), n=None, boot=200):
    class A:
        pass
    a = A()
    a.root, a.out, a.data_root, a.partial, a.n, a.boot = root, out, data_root, True, n, boot
    a.levels, a.sizes, a.combos = list(levels), list(sizes), list(P.R8)
    a.arms, a.rules, a.seeds = list(P.ARM_CHOICES), list(P.RULES7), list(seeds)
    return a


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, P87_FIFO)),
                    reason='the stored results of p87 are absent')
def test_infer_refuses_when_a_ten_module_unit_differs_from_p87(tmp_path):
    root = str(tmp_path / 'r')
    _copy_p87_unit(root, 'mix_rc', 'N', 'FIFO')
    counts, diffs, absent = P.compare_p87(root, [301])
    assert counts == {('rule FIFO', 'mix_rc', 'N'): [50, 50]} and not diffs
    assert len(absent) == 2 * 8 * (2 + 4) - 1
    _copy_p87_unit(root, 'mix_rc', 'N', 'FIFO', change=17)
    with pytest.raises(P.Refused, match=r'10-module units differ from p87 \(first: FIFO '
                                        r'mix_rc/N: 1 of 50 makespans differ'):
        P.infer(_infer_args(root, str(tmp_path / 'out'), str(tmp_path / 'data'),
                            ['mix_rc'], ['10x25'], seeds=(301,)))
    assert not (tmp_path / 'out' / P.OUT_FILES['json']).exists()
    # a full run needs every stored unit of p87
    _copy_p87_unit(root, 'mix_rc', 'N', 'FIFO')
    _, diffs, _ = P.compare_p87(root, [301], need_all=True)
    assert diffs and all('missing (p87 stored this unit)' in d for d in diffs)


# ---------------------------------------------------------------------------
# the folder
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
    """The new folder holds <rule>.npz, greedy+<run>.npz, their JSON records, base instances,
    descriptor JSON files and manifests at depth 4 to 8 below test_results/FAMILY. (1) No
    other script names it. (2) Every literal path pattern below test_results/FAMILY with a
    wildcard fails to match any of its file paths (fnmatch, whose * also crosses '/'). (3)
    The name starts with no size and ends in no registry suffix."""
    own = {os.path.normpath(os.path.join(ROOT, p)) for p in (
        'scripts/p129_project_mix.py', 'tests/test_p129_project_mix.py')}
    base = 'test_results/FAMILY/' + NEW_FOLDER
    examples = [base + '/results/exact/mix_rc/10x25/LSW/SPT.npz',
                base + '/results/exact/mix_steel/80x25n30/S/LLF.json',
                base + '/results/exact/mix_rc/50x25n30/SW/greedy+10x25+family+xmask-v1-s301.npz',
                base + '/results/exact/mix_steel/50x25n30/N/'
                       'greedy+10x25+family+xmask-solo-N-s305.json',
                base + '/data/PPVC/50x25+ppvc-rc_project/instance_000.fjs',
                base + '/data/PPVC/80x25+ppvc-steel_project/instance_029.meta.json',
                base + '/data/PPVC/80x25+ppvc-steel_project/dataset_meta.json',
                base + '/data/FAMILY/50x25_mix_rc/desc/LSW/instance_000.desc.json',
                base + '/data/FAMILY/80x25_mix_steel/manifest.json']
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


# ---------------------------------------------------------------------------
# infer on synthetic results, macro names
# ---------------------------------------------------------------------------

def _write_unit(root, level, size, combo, label, ms, names):
    p = P.result_path(root, level, size, combo, label)
    P.P101.save_npz_atomic(p, makespan=np.asarray(ms, float))
    P.write_json_atomic(p[:-4] + '.json', {
        'level': level, 'size': size, 'regime': combo, 'label': label, 'instances': names,
        'validated': len(names), 'violations': 0})


def test_infer_on_synthetic_results(tmp_path, monkeypatch):
    """mix_rc at 50 modules, three seeds: the specialist is random, the policy 1% shorter
    with a seed offset of -0.05, 0 and +0.05 points, the four classical rules 5% to 8% longer
    than the specialist, ECT 2% longer and SSPT, LLF 9% longer. Pooled: policy - specialist
    -1.0%, policy - best of four 100 (0.99/1.05 - 1), policy - best of seven
    100 (0.99/1.02 - 1); the policy is ahead of both on every combination."""
    root, data, seeds = str(tmp_path / 'r'), str(tmp_path / 'data'), (301, 302, 303)
    lv, size, n = 'mix_rc', '50x25n30', P.N_NEW
    names = [P.inst(i) for i in range(n)]
    d = os.path.join(P.desc_dir(data, lv, size), 'desc', 'N')
    os.makedirs(d)
    for x in names:
        open(os.path.join(d, x + '.desc.json'), 'w').close()
    P.write_json_atomic(P.manifest_path(data, lv, size), {'level': lv, 'size': size,
                                                          'instances': names})
    rng = np.random.default_rng(0)
    factor = {'FIFO': 1.05, 'MOR': 1.06, 'SPT': 1.07, 'MWKR': 1.08, 'ECT': 1.02, 'SSPT': 1.09,
              'LLF': 1.09}
    for combo in P.R8:
        spec = 1000.0 * (1 + 0.1 * rng.random(n))
        for r in P.RULES7:
            _write_unit(root, lv, size, combo, r, factor[r] * spec, names)
        for s, off in zip(seeds, (-0.0005, 0.0, 0.0005)):
            for arm, f in (('spec', 1.0), ('pact', 0.99 + off)):
                label = 'greedy+' + P.P103.model_name(arm, combo, s)
                _write_unit(root, lv, size, combo, label, f * spec, names)
    out = tmp_path / 'out'
    res = P.infer(_infer_args(root, str(out), data, [lv], [size], seeds, boot=400))
    S = res['sizes'][size]
    pol = S['policy'][lv]
    assert pol['pooled']['pol_spec']['mean'] == pytest.approx(-1.0, abs=1e-9)
    assert pol['pooled']['pol_rule']['mean'] == pytest.approx(100 * (0.99 / 1.05 - 1), abs=1e-9)
    assert pol['pooled']['pol_rule7']['mean'] == pytest.approx(100 * (0.99 / 1.02 - 1), abs=1e-9)
    assert pol['counts']['pol_rule']['ahead'] == 8 and pol['counts']['pol_rule7']['ahead'] == 8
    assert pol['counts']['pol_spec|0.5'] and set(pol['rows']['pol_rule']['LSW']) >= {
        'verdict', 'p_t_holm', 'p_boot_holm'}
    assert S['validated'] == 8 * (7 + 2 * len(seeds)) * n
    C = S['levels'][lv]['combos']['N']
    assert C['best_rule_ms'] == pytest.approx(_ms(root, lv, size, 'N', 'FIFO'))
    assert C['best_rule7_ms'] == pytest.approx(_ms(root, lv, size, 'N', 'ECT'))
    assert res['controls']['records'] == 'not compared (partial run)'
    txt = open(out / P.OUT_FILES['macros']).read()
    mnames = re.findall(r'\\newcommand\{\\([^}]*)\}', txt)
    assert len(mnames) == len(set(mnames)) > 100
    assert all(re.fullmatch(r'Pmx[A-Za-z]+', x) for x in mnames)
    assert 'PmxPolSpecConcreteFiftyPool' in mnames and 'PmxPolRuleVerdictConcreteFiftyLSW' in mnames
    assert 'PARTIAL' in txt
    md = open(out / P.OUT_FILES['md']).read()
    assert '## 50x25n30' in md and 'PARTIAL' in md
    # the name check: a digit, a duplicate and a clash with a macro of manuscript/ are refused
    paths = {k: str(out / v) for k, v in P.OUT_FILES.items()}
    with pytest.raises(P.Refused, match='macro names that are not Pmx plus letters'):
        P.check_macro_names(['\\newcommand{\\PmxA1}{1}'], paths)
    with pytest.raises(P.Refused, match='duplicate macro names'):
        P.check_macro_names(['\\newcommand{\\PmxA}{1}\\newcommand{\\PmxA}{2}'], paths)
    real = P.defined_macros([paths['macros'], os.path.join(ROOT, P.OUT_FILES['macros'])])
    assert len(real) > 1000 and not any(x.startswith('Pmx') for x in real)
    with monkeypatch.context() as m:
        m.setattr(P, 'defined_macros', lambda own: {'PmxPolSpecConcreteFiftyPool':
                                                    'manuscript/macros_other.tex'})
        with pytest.raises(P.Refused, match=r'already defined: \\PmxPolSpecConcreteFiftyPool'):
            P.check_macro_names(['\\newcommand{\\PmxPolSpecConcreteFiftyPool}{1}'], paths)


def _ms(root, level, size, combo, label):
    with np.load(P.result_path(root, level, size, combo, label)) as z:
        return np.asarray(z['makespan'], dtype=float).mean()


def test_infer_refuses_a_missing_result(tmp_path):
    data = str(tmp_path / 'data')
    names = [P.inst(i) for i in range(P.N_NEW)]
    d = os.path.join(P.desc_dir(data, 'mix_steel', '80x25n30'), 'desc', 'N')
    os.makedirs(d)
    for x in names:
        open(os.path.join(d, x + '.desc.json'), 'w').close()
    P.write_json_atomic(P.manifest_path(data, 'mix_steel', '80x25n30'), {'instances': names})
    with pytest.raises(P.Refused, match='result mix_steel/80x25n30/N/FIFO is missing'):
        P.infer(_infer_args(str(tmp_path / 'r'), str(tmp_path / 'out'), data, ['mix_steel'],
                            ['80x25n30']))


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('K', [1, 8, 16])
def test_plan_partition(K):
    units = P.all_units(P.MIX_LEVELS, P.SIZES, P.R8, P.ARM_CHOICES, P.SEEDS)
    parts, loads = P.assign_parts(units, K)
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(units)))                         # each unit exactly once
    assert max(loads) - min(loads) <= max(P.unit_cost(u) for u in units) + 1e-9
    assert sum(loads) == pytest.approx(sum(P.unit_cost(u) for u in units))
    # the stated rates: about 4.5 and 10 minutes per combination of 30 instances
    pol = [u for u in units if u[3] == 'pact']
    assert {sz: P.unit_cost(next(u for u in pol if u[1] == sz)) / 60 for sz in P.NEW_SIZES} == \
        {'50x25n30': 4.5, '80x25n30': 10.0}


def test_plan_prints_one_command_per_part_and_skips_finished(tmp_path):
    root = tmp_path / 'r'
    u = ('mix_rc', '10x25', 'N', 'rule', None, 'FIFO')
    _write_unit(str(root), *u[:3], u[5], np.ones(50), [f'instance_{i:03d}' for i in range(50)])
    r = subprocess.run([PY, '-B', SCRIPT, 'plan', '--parts', '16', '--root', str(root)],
                       capture_output=True, text=True, cwd=ROOT, env=env())
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    cmds = [x for x in r.stdout.splitlines() if 'p129_project_mix.py evaluate' in x]
    assert len(cmds) == 16
    for i, c in enumerate(cmds):
        assert f'taskset -c <CORE_{i}> ' in c and c.endswith(
            f'evaluate --part {i} --parts 16 --root {root}')
        assert c.startswith('CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ')
    assert '816 result files (1 finished)' in r.stdout
    assert not (root / 'results' / 'exact' / 'mix_steel').exists()  # plan writes nothing
