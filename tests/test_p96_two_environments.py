"""Tests for scripts/p96_two_environments.py: the environment of the paper against the prior
implementation, on synthetic trees that hold both environments (10x25, seeds 301 to 303).

Run from the repository root:  python -m pytest tests/test_p96_two_environments.py -q
CPU only; every tree is written under a temporary folder.
"""
import json
import os
import re
import subprocess
import sys
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import p96_two_environments as P96  # noqa: E402
import stats_family as SF  # noqa: E402

P96_PATH = os.path.join(ROOT, 'scripts', 'p96_two_environments.py')
R8 = list(REG.COMBOS)
SEEDS = (301, 302, 303)
N = 12
BT = 300
SEED_FACTOR = {s: 1 + 0.05 * (s - 302) for s in SEEDS}   # large, so a wrong pairing shows
# environment of the paper = prior implementation times this factor (per arm)
ARM_FACTOR = {'released': 0.98, 'specialist': 1.01, 'control': 1.0}
OUTS = ('manuscript/macros_environments.tex', 'results/environments.json', 'results/environments.md')


def noise(*key):
    return 1 + 0.002 * np.random.default_rng(zlib.crc32(repr(key).encode())).standard_normal(N)


def save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([np.asarray(x, dtype=float), np.zeros(len(x))], axis=1))


def rule_factor(q, e):
    """Rules: the environment of the paper shortens rule q by (j + 1)% on even instances."""
    j = F88.RULES.index(q)
    return np.where(np.arange(N) % 2 == 0, 1 - 0.01 * (j + 1), 1.0) if e == 'exact' else 1.0


def two_environment_tree(root):
    """Descriptors and the arrays of both environments under root/test_results/FAMILY."""
    for r in R8:
        d = root / 'data' / 'FAMILY' / '10x25' / 'desc' / r
        d.mkdir(parents=True)
        for i in range(N):
            json.dump({'base_instance': f'10x25/base_{i:03d}'}, open(d / f'instance_{i:03d}.desc.json', 'w'))
    cwd = os.getcwd()
    os.chdir(root)
    try:
        for r in R8:
            base = 400 + 200 * np.random.default_rng(zlib.crc32(r.encode())).random(N)
            rules = {'FIFO': 1.05, 'MOR': 1.10, 'SPT': np.where(np.arange(N) % 2 == 0, 1.08, 1.04),
                     'MWKR': 1.12}
            for e in ('legacy', 'exact'):
                with P96.environment(e):
                    for q in F88.RULES:
                        save(f'{REG.rule_dir("10x25")}/Result_{q}_{r}.npy',
                             base * rules[q] * rule_factor(q, e))
                    for arm in ('released', 'specialist', 'control'):
                        for s in SEEDS:
                            x = base * SEED_FACTOR[s] * noise(arm, r, s)
                            if e == 'exact':
                                x = x * ARM_FACTOR[arm] * noise(arm, r, s, 'exact')
                            save(REG.result_path(arm, r if arm == 'specialist' else None, s, '10x25', r), x)
    finally:
        os.chdir(cwd)


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.delenv('PACT_RESULTS', raising=False)
    monkeypatch.delenv('PACT_OUT', raising=False)
    monkeypatch.setenv('PACT_ENV', 'exact')
    monkeypatch.setenv('PACT_SEEDS', '301-303')
    monkeypatch.setattr(F88, 'B', F88.B)            # main() calls configure(); restored after
    monkeypatch.setattr(F88, 'ALPHA', F88.ALPHA)
    root = tmp_path / 'root'
    root.mkdir()
    two_environment_tree(root)
    (root / 'manuscript').mkdir()
    (root / 'manuscript' / 'macros_unrelated.tex').write_text('\\newcommand{\\SomethingElse}{1}\n')
    monkeypatch.chdir(tmp_path)                     # main() changes into root; restored after
    return root


def run_main(root, out, *extra):
    rc = P96.main(['--root', str(root), '--out', str(out), '--boot', str(BT), *extra])
    js = json.load(open(out / OUTS[1])) if rc == 0 else None
    return rc, js


def read(root, e, arm, r, s):
    cwd = os.getcwd()
    os.chdir(root)
    try:
        with P96.environment(e):
            return REG.load(arm, r if arm == 'specialist' else None, s, '10x25', r)
    finally:
        os.chdir(cwd)


def test_p96_sign_and_same_seed_pairing(tree, tmp_path):
    rc, js = run_main(tree, tmp_path / 'out')
    assert rc == 0
    A = js['arms']
    for key, arm, want, dirn in (('Pol', 'released', -2.0, 'ahead'), ('Spec', 'specialist', 1.0, 'behind'),
                                 ('Ctl', 'control', 0.0, 'neither')):
        for r in R8:
            x = A[key]['rows'][r]
            d = np.stack([SF.rel_gap(read(tree, 'exact', arm, r, s), read(tree, 'legacy', arm, r, s))
                          for s in SEEDS])                     # paper minus prior over prior, seed s vs s
            assert x['mean'] == pytest.approx(d.mean(), abs=1e-12)
            assert x['mean'] == pytest.approx(want, abs=0.2) and x['direction'] == dirn
            # the 5% seed factor cancels only when seed s meets seed s
            assert x['seed_means'] == pytest.approx([want] * 3, abs=0.2)
            draws = SF.hier_draws_stream(d, BT, np.random.default_rng(11))
            assert (x['lo'], x['hi']) == SF.percentile_ci(draws, 0.05)
    assert set(A['Pol']['pooled']) == set(A['Spec']['pooled']) == {'All', 'Train', 'Held'}
    assert set(A['Ctl']['pooled']) == {'All', 'Seen', 'Unseen'}
    regs = {'All': R8, 'Seen': list(REG.CONTROL_COMBOS),
            'Unseen': [r for r in R8 if r not in REG.CONTROL_COMBOS]}
    for fam, rs in regs.items():                               # pooled: a base instance with all
        d3 = np.stack([np.stack([SF.rel_gap(read(tree, 'exact', 'control', r, s),   # its combinations
                                            read(tree, 'legacy', 'control', r, s)) for s in SEEDS])
                       for r in rs], axis=2)
        hb = SF.hier_boot(d3, B=BT, seed=11)
        x = A['Ctl']['pooled'][fam]
        assert (x['mean'], x['lo'], x['hi'], x['c']) == (hb['mean'], hb['lo'], hb['hi'], len(rs))
    assert A['Pol']['pooled']['All']['direction'] == 'ahead'


def test_p96_rules_and_their_best_with_the_instance_bootstrap(tree, tmp_path):
    rc, js = run_main(tree, tmp_path / 'out')
    assert rc == 0
    cwd = os.getcwd()
    os.chdir(tree)
    try:
        arr = {}
        for e in ('legacy', 'exact'):
            with P96.environment(e):
                arr[e] = {q: {r: REG.read_path(f'{REG.rule_dir("10x25")}/Result_{q}_{r}.npy')
                              for r in R8} for q in F88.RULES}
                arr[e]['Best'] = {r: np.min([arr[e][q][r] for q in F88.RULES], axis=0) for r in R8}
    finally:
        os.chdir(cwd)
    for q in F88.RULES + ('Best',):
        cells = np.stack([SF.rel_gap(arr['exact'][q][r], arr['legacy'][q][r]) for r in R8], axis=1)
        for j, r in enumerate(R8):
            ib = SF.instance_boot(cells[:, j], B=BT, seed=7)
            x = js['rules'][q]['rows'][r]
            assert (x['mean'], x['lo'], x['hi']) == (ib['mean'], ib['lo'], ib['hi'])
            assert x['direction'] == SF.direction(ib['lo'], ib['hi'])
        ib = SF.instance_boot(cells, B=BT, seed=7)
        x = js['rules'][q]['pooled']
        assert (x['mean'], x['lo'], x['hi'], x['c']) == (ib['mean'], ib['lo'], ib['hi'], 8)
    # FIFO 1% shorter on half the instances: mean -0.5; the best is SPT (even) or FIFO (odd)
    assert js['rules']['FIFO']['pooled']['mean'] == pytest.approx(-0.5, abs=1e-9)
    assert js['rules']['FIFO']['rows']['N']['direction'] == 'ahead'


def test_p96_verdict_tables_are_p88_section_A_in_each_environment(tree, tmp_path):
    rc, js = run_main(tree, tmp_path / 'out')
    assert rc == 0
    F88.configure(boot=BT)
    os.chdir(tree)
    for e, key in (('legacy', 'prior'), ('exact', 'paper')):
        with P96.environment(e):
            D, C = F88.parity_components('10x25', SEEDS)
            iut, pooled = F88.iut_block(C, F88.PAR_FAM), F88.pooled_block(D, F88.PAR_FAM)
        vt = js['verdict_tables'][key]
        for r in R8:
            for k in ('mean', 'lo', 'hi', 'sim_lo', 'sim_hi', 't_lo', 't_hi', 'v0.5', 'v1.0',
                      'direction'):
                assert vt['components'][r][k] == C[r][k]
        assert vt['iut'] == json.loads(json.dumps(iut))
        assert {f: (x['lo'], x['hi']) for f, x in vt['pooled'].items()} == \
            {f: (x['lo'], x['hi']) for f, x in pooled.items()}
    dif = js['verdict_tables']['differences']
    for mg in F88.MARGINS:
        assert dif[f'v{mg}'] == [r for r in R8 if js['verdict_tables']['prior']['components'][r][f'v{mg}']
                                 != js['verdict_tables']['paper']['components'][r][f'v{mg}']]
    text = (tmp_path / 'out' / OUTS[0]).read_text()
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    assert mac['EnvParDiffNHalf'] == str(len(dif['v0.5'])) and mac['EnvParDirDiffN'] == str(len(dif['direction']))
    for r in R8:
        assert mac[f'EnvParPrior{r}VHalf'] == SF.TEX[js['verdict_tables']['prior']['components'][r]['v0.5']]
        assert mac[f'EnvParExact{r}Dir'] == SF.TEX[js['verdict_tables']['paper']['components'][r]['direction']]


def test_p96_writes_under_the_output_root_with_env_names_and_plain_words(tree, tmp_path):
    out = tmp_path / 'out'
    rc, js = run_main(tree, out)
    assert rc == 0
    assert sorted(os.listdir(tree / 'manuscript')) == ['macros_unrelated.tex']   # inputs untouched
    assert not (tree / 'results').exists()
    text = (out / OUTS[0]).read_text()
    md = (out / OUTS[2]).read_text()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == len(set(names)) == js['n_macros']
    assert all(re.fullmatch(r'Env[A-Za-z]+', n) for n in names)
    for word in ('legacy', 'old', 'xmask'):
        assert not re.search(rf'\b{word}\b', text + md, flags=re.I), word
    assert 'prior implementation' in text and 'implementation released with the prior study' in text
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    assert mac['EnvSeedN'] == '3' and mac['EnvInstN'] == str(N) and mac['EnvBoot'] == str(BT)
    assert mac['EnvPolNDir'] == 'ahead' and mac['EnvSpecNDir'] == 'behind'
    assert mac['EnvPolN'] == P96.f2(js['arms']['Pol']['rows']['N']['mean'])
    assert {'EnvRuleFIFON', 'EnvBestPool', 'EnvParPriorIutAllHalfEq', 'EnvParExactPoolHeldVOne',
            'EnvCtlPoolUnseen'} <= set(mac)
    # no name of this file is defined by a macro file of either manuscript in the repository
    existing = set()
    for d in ('paper', 'manuscript', os.path.join('manuscript', 'sections_new'),
              os.path.join('manuscript', 'supp_new')):
        p = os.path.join(ROOT, d)
        if not os.path.isdir(p):         # a manuscript folder that the release does not hold
            continue
        for f in os.listdir(p):
            if re.fullmatch(r'macros.*\.tex', f) and f != 'macros_environments.tex':
                existing |= {a or b for a, b in P96.DEF_RX.findall(open(os.path.join(p, f), errors='replace').read())}
    assert not set(names) & existing


def test_p96_collision_check_skips_only_its_own_file_name(tree, tmp_path):
    (tree / 'manuscript' / 'macros_environments.tex').write_text('\\newcommand{\\EnvPolN}{x}\n')
    rc, _ = run_main(tree, tmp_path / 'out')
    assert rc == 0                                    # the file of the output name is skipped
    (tree / 'manuscript' / 'macros_other.tex').write_text('\\newcommand{\\EnvPolN}{x}\n')
    rc, _ = run_main(tree, tmp_path / 'out')
    assert rc == 1
    assert not any((tmp_path / 'out' / p).exists() for p in OUTS)   # a refusal removes them


def run_script(*args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('PACT_')}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable, P96_PATH, *args], env=env, capture_output=True,
                          text=True, timeout=300)


def test_p96_refuses_a_missing_or_short_array_of_either_environment(tree, tmp_path):
    out = tmp_path / 'out'
    base = ['--root', str(tree), '--out', str(out), '--boot', '100']
    run = run_script(*base, '--check')
    assert run.returncode == 0 and not out.exists()                    # --check writes nothing
    assert 'prior implementation: 0 problem(s)' in run.stdout
    assert run_script(*base).returncode == 0 and all((out / p).exists() for p in OUTS)
    cwd = os.getcwd()
    os.chdir(tree)
    try:
        with P96.environment('exact'):
            gone = REG.result_path('control', None, 302, '10x25', 'SW')
        with P96.environment('legacy'):
            short = REG.result_path('specialist', 'LS', 303, '10x25', 'LS')
    finally:
        os.chdir(cwd)
    os.rename(tree / gone, tmp_path / 'kept.npy')
    run = run_script(*base)
    assert run.returncode == 1 and not any((out / p).exists() for p in OUTS)
    assert 'environment of the paper: 1' in run.stderr and f'missing {gone}' in run.stderr
    assert 'prior implementation: 0' in run.stderr
    assert run_script(*base, '--check').returncode == 1
    os.rename(tmp_path / 'kept.npy', tree / gone)
    save(str(tree / short), np.ones(N - 1))
    run = run_script(*base)
    assert run.returncode == 1 and 'prior implementation: 1' in run.stderr
    assert f'{short}: {N - 1} rows, the test set has {N}' in run.stderr
    run = run_script(*base, '--seeds', '301-306')
    assert run.returncode != 0 and 'not planned in both environments' in run.stderr
