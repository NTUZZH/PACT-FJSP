"""Tests for the combined verdict against a rule quantity (ahead / behind / not
distinguishable) in scripts/p89_public_family.py and scripts/p88_family_inference.py,
for the production rows of p88 under the paper's sign convention (policy minus
rival over rival), and for p88's sections F (rule-change pairs), G (the LSW specialist
on every combination) and H (larger shops).

The earlier tests load p88's functions from the source file (ast) into a namespace that
holds the names they read; the bodies are the script's own. p88 runs its analysis only as a
script, so the tests of F, G and H import it, and run it as a script on a synthetic tree
of the exact environment.

Run from the repository root:  python -m pytest tests/test_sign_and_verdict.py -q
CPU only; the tests write synthetic result trees under a temporary folder.
"""
import ast
import json
import os
import re
import subprocess
import sys
import types
import zlib

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import arm_registry as REG  # noqa: E402
import p88_family_inference as F88  # noqa: E402
import p89_public_family as P  # noqa: E402
import stats_family as SF  # noqa: E402
import test_public_family_inference as T  # noqa: E402
from test_public_family_inference import small_sets  # noqa: E402,F401  (autouse: six instances per set)

R8 = list(REG.COMBOS)
P88 = os.path.join(ROOT, 'scripts', 'p88_family_inference.py')
AH, BH, ND = 'ahead', 'behind', 'not distinguishable'


def p88_functions(names, **namespace):
    """The named top-level functions of p88, compiled from its source into a namespace
    that holds np, SF, the shared verdict rule and the given names."""
    tree = ast.parse(open(P88).read(), P88)
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(d.name for d in defs) == sorted(names)
    g = dict(np=np, SF=SF, superiority_verdict=P.superiority_verdict,
             VERDICT_WORDS=P.VERDICT_WORDS, **namespace)
    exec(compile(ast.Module(body=defs, type_ignores=[]), P88, 'exec'), g)
    return g


# ------------------------------------------------------------------ the rule itself
def test_verdict_words_and_branches():
    assert P.VERDICT_WORDS == (AH, BH, ND)
    v = P.superiority_verdict
    assert v(-2.0, -0.5, 0.001, 0.004, 1.0, 1.0) == AH
    assert v(0.5, 2.0, 1.0, 1.0, 0.001, 0.004) == BH
    assert v(-2.0, 0.5, 0.001, 0.004, 1.0, 1.0) == ND     # seed-level t and bootstrap say
    assert v(-2.0, 0.5, 0.001, 0.30, 1.0, 1.0) == ND      # ahead, the interval contains zero
    assert v(-2.0, -0.5, 0.001, 0.06, 1.0, 1.0) == ND     # bootstrap p not below alpha
    assert v(-2.0, -0.5, 0.06, 0.001, 1.0, 1.0) == ND     # seed-level t p not below alpha
    assert v(-2.0, 0.0, 0.001, 0.001, 1.0, 1.0) == ND     # an interval ending at zero
    assert v(0.0, 2.0, 1.0, 1.0, 0.001, 0.001) == ND
    assert v(0.5, 2.0, 0.001, 0.001, 1.0, 1.0) == ND      # ahead p-values, interval above zero
    assert v(-2.0, -0.5, float('nan'), 0.001, 1.0, 1.0) == ND
    assert v(-2.0, -0.5, 0.049, 0.049, 1.0, 1.0) == AH
    assert v(-2.0, -0.5, 0.049, 0.049, 1.0, 1.0, alpha=0.01) == ND
    assert v(-2.0, -0.5, 0.05, 0.001, 1.0, 1.0) == ND     # strictly below alpha


def test_mirror_p_values_use_the_same_means_and_draws():
    """component(behind=True) adds the H1: d > 0 p-values and changes nothing else."""
    g = np.random.default_rng(3)
    d = g.normal(0.3, 2.0, size=(5, 40)) + np.array([0.1, -0.2, 0.0, 0.3, -0.1])[:, None]
    a, b = P.component(d, 300), P.component(d, 300, behind=True)
    assert set(b) - set(a) == {'p_t_greater', 'p_boot_greater'}
    assert all(np.array_equal(a[k], b[k]) if isinstance(a[k], list) else a[k] == b[k]
               for k in a)
    hb = SF.hier_boot(d, B=300, seed=P.HIER_SEED, return_draws=True)
    assert b['p_boot_greater'] == SF.boot_p_one_sided(-hb['draws'], 'less')
    assert b['p_t_greater'] == pytest.approx(SF.seed_t_p(-d, 'less'), rel=1e-12)
    ties = int((hb['draws'] == 0).sum())
    assert b['p_boot_less'] + b['p_boot_greater'] == pytest.approx((302 + ties) / 301)


# ------------------------------------------------------------------ p89 end to end
def rule_vectors(base, st, r):
    """Four rules; FIFO is the per-instance best."""
    return {q: base * (1.2 + 0.1 * R8.index(r)) * (1 + 0.01 * j) for j, q in enumerate(P.RULES)}


PATTERN = np.array([20.0, -21.0, 20.0, -21.0, 20.0, -21.0])     # mean -0.5, wide over instances


def d_policy(r, s):
    """d of the policy against the best rule in percent, per instance (six instances)."""
    jitter = 0.01 * (s - 303)
    return {'N': np.full(6, -3.0 + jitter),            # ahead
            'L': np.full(6, 3.0 + jitter),             # behind
            'S': PATTERN + jitter,                     # seed-level t says ahead, CI contains 0
            'W': np.full(6, jitter)}.get(r, np.full(6, -3.0 + jitter))


def policy(sp, st, r, s, base):
    best = rule_vectors(base, st, r)['FIFO']
    return best * (1 + d_policy(r, s) / 100)


def test_p89_combined_verdict(tmp_path):
    root = T.build_tree(str(tmp_path / 'tree'), pol=policy, rules=rule_vectors)
    res, out = T.run(root, tmp_path)
    for st, S in res['sets'].items():
        x = {r: S['rules'][r]['best'] for r in R8}
        assert x['N']['verdict'] == AH and x['N']['hi'] < 0
        assert x['L']['verdict'] == BH and x['L']['lo'] > 0
        assert x['L']['p_t_greater_holm'] < 0.05 and x['L']['p_boot_greater_holm'] < 0.05
        assert x['L']['p_t_holm'] > 0.05                        # the ahead family does not fire
        # the seed-level t alone says ahead after Holm, the interval contains zero
        assert x['S']['p_t_holm_ahead'] and x['S']['lo'] < 0 < x['S']['hi']
        assert x['S']['verdict'] == ND
        assert x['W']['verdict'] == ND
        assert [x[r]['verdict'] for r in ('LS', 'LW', 'SW', 'LSW')] == [AH] * 4
        # the mirror family is Holm over the same eight combinations
        want = SF.holm([x[r]['p_t_greater'] for r in R8])
        assert [x[r]['p_t_greater_holm'] for r in R8] == pytest.approx(want)
        # every existing quantity of the row keeps its meaning
        d = np.stack([d_policy('S', s) for s in P.SEEDS])
        assert x['S']['mean'] == pytest.approx(d.mean(), abs=1e-9)
        assert x['S']['p_t'] == pytest.approx(SF.seed_t_p(d, 'less'), rel=1e-6)
        assert x['S']['p_t_greater'] == pytest.approx(SF.seed_t_p(d, 'greater'), rel=1e-6)
    text = (out / 'manuscript' / 'macros_public.tex').read_text()
    assert '\\newcommand{\\PubSdtenRuleNVerdict}{ahead}' in text
    assert '\\newcommand{\\PubSdtenRuleLVerdict}{behind}' in text
    assert '\\newcommand{\\PubSdtenRuleSVerdict}{not distinguishable}' in text
    assert '\\newcommand{\\PubBrandRuleVerdictAheadN}{5}' in text
    assert '\\newcommand{\\PubBrandRuleVerdictBehindN}{1}' in text
    names = re.findall(r'\\newcommand\{\\([A-Za-z]+)\}', text)
    assert len(names) == len(set(names)) == \
        res['n_macros'] + res['n_macros_verdict'] + res['n_macros_dir']
    md = (out / 'results' / 'public_family_inference.md').read_text()
    assert '## Combined verdict against the per-instance best rule' in md
    assert 'Ahead on 5, behind on 1, not distinguishable on 2 of 8 combinations.' in md


# ------------------------------------------------------------------ pooled directions
HELD_PATTERN = np.array([3.0, -3.0, 3.0, -3.0, 3.0, -3.0])


def policy_for_directions(sp, st, r, s, base):
    """Against its specialist: -0.003% on the six training combinations (an interval
    below zero whose bound rounds to zero), +-3% by instance on the two held-out ones."""
    return sp * (1 + (HELD_PATTERN if r in P.HELD else np.full(len(base), -0.003)) / 100)


def test_p89_pooled_directions(tmp_path):
    root = T.build_tree(str(tmp_path / 'tree'), pol=policy_for_directions)
    res, out = T.run(root, tmp_path)
    text = (out / 'manuscript' / 'macros_public.tex').read_text()
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', text))
    rows = {}
    for st, S in res['sets'].items():
        w = P.SETS[st][0]
        rows.update({f'Pub{w}ParPool{f}': S['parity']['pooled'][f] for f in ('All', 'Train', 'Held')})
        for f in ('Unseen', 'Pair', 'Held'):
            rows[f'Pub{w}PolCtlPool{f}'] = S['control']['pooled_policy_vs_control'][f]
            rows[f'Pub{w}CtlSpecPool{f}'] = S['control']['pooled_control_vs_specialist'][f]
    for st, X in res['sensitivity'].items():
        for lab, pre in (('half', ''), ('one', 'Base')):
            for k, nm in (('par', 'Par'), ('polctl', 'PolCtl'), ('ctlspec', 'CtlSpec'), ('rule', 'Rule')):
                rows[f'Pub{P.SENS_WORD[st]}{pre}{nm}PoolL'] = X['pooled'][lab][k]
    for name, x in rows.items():                    # the per-combination rule, unrounded ends
        assert mac[name + 'Dir'] == SF.TEX[SF.direction(x['lo'], x['hi'])] == x['direction']
    pooled = {m[:-2] for m in mac if m.endswith('Lo') and 'Pool' in m}
    assert pooled == set(rows) and res['n_macros_dir'] == len(rows) == len(P.SETS) * 9 + 8
    # a bound below zero that rounds to zero keeps its sign, and the direction says ahead
    assert mac['PubSdtenParPoolTrainHi'] == '-0.00' and mac['PubSdtenParPoolTrainDir'] == 'ahead'
    assert mac['PubSdtenParPoolTrainLo'] == '-0.00' and mac['PubSdtenParPoolTrain'] == '-0.00'
    assert mac['PubSdtenParPoolHeldDir'] == 'neither'
    assert mac['PubSdtenCtlSpecPoolPairDir'] == 'behind'      # control = specialist + 0.2%
    assert mac['PubSdtenPolCtlPoolPairDir'] == 'ahead'
    assert set(mac[n + 'Dir'] for n in rows) == {'ahead', 'behind', 'neither'}
    # written after the verdict macros, before section 8; nothing above them changed
    names = list(mac)
    first_dir = names.index('PubSdtenParPoolAllDir')
    assert names[first_dir - 1] == 'PubVdataRuleVerdictBehindN'
    assert all(n.endswith('Dir') for n in names[first_dir:])
    assert res['sets']['pub_SD2_test_10x5+mix']['parity']['pooled']['Train']['hi'] < 0


def test_p88_pooled_directions():
    ns = p88_functions(['dir_macros', 'tex', 'f2'])
    block = {'All': dict(lo=-0.6745, hi=-0.0023), 'Train': dict(lo=-0.1, hi=0.2),
             'Held': dict(lo=0.3, hi=0.9)}
    lines = ns['dir_macros']('FamParPool', block, 'test rows')
    vals = [re.match(r'\\newcommand\{\\(\w+)\}\{([a-z]+)\}', x).groups() for x in lines]
    assert vals == [('FamParPoolAllDir', 'ahead'), ('FamParPoolTrainDir', 'neither'),
                    ('FamParPoolHeldDir', 'behind')]
    assert '[-0.6745, -0.0023]' in lines[0]
    # the interval macros keep the sign of the unrounded value; an exact zero prints +0.00
    assert [ns['f2'](x) for x in (-0.0023, 0.0023, 0.0, -0.0, -0.005, 0.004999, -1.236, 2.0)] == \
        ['-0.00', '+0.00', '+0.00', '+0.00', '-0.01', '+0.00', '-1.24', '+2.00']
    assert [P.f2(x) for x in (-0.0023, 0.0, -0.0, np.float64(-0.004))] == \
        ['-0.00', '+0.00', '+0.00', '-0.00']


# ------------------------------------------------------------------ p88
def test_p88_superiority_block():
    k, n = 3, 40
    g = np.random.default_rng(9)
    jitter = np.array([-0.01, 0.0, 0.01])[:, None]
    alt = np.where(np.arange(n) % 2 == 0, 20.0, -21.0)
    D = {'N': -3.0 + 0.2 * g.standard_normal((k, n)) + jitter,
         'L': 3.0 + 0.2 * g.standard_normal((k, n)) + jitter,
         'S': alt[None, :] + jitter,                  # mean -0.5, wide over instances
         'W': 0.2 * g.standard_normal((k, n)) + 5 * jitter}
    for r in R8[4:]:
        D[r] = -2.0 + 0.2 * g.standard_normal((k, n)) + jitter
    ns = p88_functions(['superiority_block'], ALPHA=0.05,
                       fresh=lambda d: SF.hier_draws_stream(d, 400, np.random.default_rng(11)))
    rows = ns['superiority_block'](D)
    assert rows['N']['verdict'] == AH and rows['L']['verdict'] == BH
    assert rows['S']['p_t_holm_ahead'] and rows['S']['lo'] < 0 < rows['S']['hi']
    assert rows['S']['verdict'] == ND and rows['W']['verdict'] == ND
    assert [rows[r]['verdict'] for r in R8[4:]] == [AH] * 4
    assert list(rows['N'])[:11] == ['mean', 'lo', 'hi', 'p_t', 'p_boot', 'seed_means', 'n',
                                    'p_t_holm', 'p_t_holm_ahead', 'p_boot_holm',
                                    'p_boot_holm_ahead']               # existing keys first
    draws = SF.hier_draws_stream(D['L'], 400, np.random.default_rng(11))
    assert rows['L']['p_boot_greater'] == SF.boot_p_one_sided(draws, 'greater')
    assert rows['L']['p_t_greater'] == pytest.approx(SF.seed_t_p(D['L'], 'greater'))
    assert [rows[r]['p_boot_greater_holm'] for r in R8] == pytest.approx(
        SF.holm([rows[r]['p_boot_greater'] for r in R8]))


def p88_production(seed_factor, B=300):
    """lead_block and diff_block of p88 on synthetic makespans: rival r_i per regime,
    policy seed s = rival times a per-instance factor."""
    n, seeds = 30, (301, 302, 303)
    g = np.random.default_rng(4)
    rival = {r: g.uniform(400, 600, n) for r in R8}
    pol = {s: {r: rival[r] * seed_factor[s] * (1 + g.uniform(-0.03, 0.03, n)) for r in R8}
           for s in seeds}
    reg = types.SimpleNamespace(pooled=lambda arm, combo, size, r, n_: np.mean(
        [pol[s][r][:n_] for s in seeds], axis=0))
    ns = p88_functions(['lead_block', 'diff_block'], SEEDS=seeds, SEL=302, B=B, INST_SEED=7,
                       ALPHA=0.05, REG=reg, released=lambda s, size, r, n_: pol[s][r][:n_])

    def rel(a, b):
        return 100.0 * (a - b) / b
    old = ns['lead_block'](lambda src: np.stack([rel(rival[r], src('50x25', r, n)) for r in R8], 1))
    new = ns['diff_block'](lambda src: np.stack([rel(src('50x25', r, n), rival[r]) for r in R8], 1))
    return old, new, rival, pol, rel, n


def test_p88_production_sign_convention():
    """Policy 4% shorter (s301, s302) and 3% longer (s303): the new rows are
    100 (policy - rival) / rival, the opposite sign of the lead, with the rival's
    makespan as the denominator."""
    old, new, rival, pol, rel, n = p88_production({301: 0.96, 302: 0.96, 303: 1.03})
    for lab in ('s301', 's302', 's303', 'mean'):
        src = (lambda r: pol[int(lab[1:])][r]) if lab != 'mean' else \
            (lambda r: np.mean([pol[s][r] for s in pol], axis=0))
        cells = np.stack([rel(src(r), rival[r]) for r in R8], 1)
        lead = np.stack([rel(rival[r], src(r)) for r in R8], 1)
        assert np.allclose(cells, -lead / (1 + lead / 100), atol=1e-12)
        assert new[lab]['mean'] == pytest.approx(cells.mean(), abs=1e-12)
        assert np.sign(new[lab]['mean']) == -np.sign(old[lab]['mean'])
        assert abs(new[lab]['mean']) != pytest.approx(abs(old[lab]['mean']), abs=1e-3)
        ib = SF.instance_boot(cells, B=300, seed=7)
        assert (new[lab]['lo'], new[lab]['hi']) == (ib['lo'], ib['hi'])
    for lab in ('s301', 's302'):           # policy shorter: negative, smaller magnitude
        assert new[lab]['hi'] < 0 and new[lab]['direction'] == 'ahead'
        assert -new[lab]['mean'] < old[lab]['mean']
        assert new[lab]['regimes_ahead'] == 8 and new[lab]['regimes_behind'] == 0
    assert new['s303']['lo'] > 0 and new['s303']['direction'] == 'behind'
    assert new['s303']['mean'] > -old['s303']['mean'] > 0            # policy longer: larger
    assert new['s303']['regimes_behind'] == 8
    assert new['seed_summary'] == dict(min=min(new['s301']['mean'], new['s302']['mean']),
                                       max=new['s303']['mean'], clear=2, tie=0, behind=1)
    assert old['seed_summary']['clear'] == 2 and old['seed_summary']['behind'] == 1
    assert new['selected'] == dict(new['s302'], seed=302)


# ------------------------------------------------------------------ p88, production sets
PROD_FUNCTIONS = ['production', 'lead_block', 'diff_block', 'pool', 'released', 'load', 'rel',
                  'verify_bases', 'empty_comparisons']


def production_tree(root, env, n=12, solver_sizes=('50x25', '80x25')):
    """Descriptors, rule and policy arrays of the two production sets under the active
    environment, and 1800 s solver records (first 10 instances) for solver_sizes."""
    import json
    seeds = REG.seeds()
    g = np.random.default_rng(8)
    for size in ('50x25', '80x25'):
        os.makedirs(REG.rule_dir(size), exist_ok=True)
        for r in R8:
            d = root / 'data' / 'FAMILY' / size / 'desc' / r
            d.mkdir(parents=True)
            for i in range(n):
                json.dump({'base_instance': f'{size}/base_{i:03d}'},
                          open(d / f'instance_{i:03d}.desc.json', 'w'))
            rule = g.uniform(500, 600, n)
            for j, q in enumerate(('FIFO', 'MOR', 'SPT', 'MWKR')):
                np.save(f'{REG.rule_dir(size)}/Result_{q}_{r}.npy',
                        np.stack([rule + 10 * j, np.zeros(n)], axis=1))
            for s in seeds:
                np.save(REG.result_path('released', None, s, size, r),
                        np.stack([rule * g.uniform(0.9, 0.95, n), np.zeros(n)], axis=1))
            if size in solver_sizes:
                o = root / 'or_solution' / 'FAMILY' / f'{size}_fairwarm1800s' / r
                o.mkdir(parents=True)
                for i in range(10):
                    json.dump(dict(instance=f'instance_{i:03d}', warm_makespan=float(rule[i]),
                                   anytime=[[5.0, float(rule[i]) - 30, 0.0]]),
                              open(o / f'instance_{i:03d}.json', 'w'))


def production_namespace(seen, solver_reader):
    import glob as real_glob
    import json
    return p88_functions(
        PROD_FUNCTIONS, REG=REG, PROD=(('50x25', 'Fifty'), ('80x25', 'Eighty')),
        PROD_N30={'50x25': '50x25n30', '80x25': '80x25n30'}, R8=R8,
        RULES=('FIFO', 'MOR', 'SPT', 'MWKR'), SEEDS=REG.seeds(), SEL=302, B=200, INST_SEED=7,
        ALPHA=0.05, SOLVER_DIR='or_solution/FAMILY/{size}_fairwarm1800s', p82=solver_reader,
        glob=types.SimpleNamespace(glob=lambda p: seen.append(p) or real_glob.glob(p)),
        json=json, os=os, sys=sys)


@pytest.mark.parametrize('env', ['exact', 'legacy'])
def test_p88_production_rows_skip_the_solver_in_the_exact_environment(tmp_path, monkeypatch, env):
    """In the exact environment p88's production rows never read the 1800 s solver folders
    (they exist in the tree and would be read in the legacy environment) and hold the
    rule-pool rows only; in the legacy environment both."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PACT_ENV', env)
    monkeypatch.setenv('PACT_SEEDS', '301-303')
    monkeypatch.setenv('PACT_RESULTS', str(tmp_path / 'res'))
    production_tree(tmp_path, env)
    import p82_best3
    calls, seen = [], []
    reader = types.SimpleNamespace(cp_1800=lambda size, r: calls.append((size, r)) or
                                   p82_best3.cp_1800(size, r))
    ns = production_namespace(seen, reader)
    solver_rows = env != 'exact'                       # p88's SOLVER_ROWS
    prodr, prodd = ns['production'](solver_rows, False)
    keys = [('50x25', 'cpsat1800'), ('50x25', 'pool'), ('80x25', 'cpsat1800'), ('80x25', 'pool')]
    want = keys if solver_rows else [k for k in keys if k[1] == 'pool']
    assert list(prodr) == want and list(prodd) == want
    touched = [p for p in seen if 'fairwarm1800s' in p]
    if solver_rows:
        assert len(calls) == 16 and touched
        assert prodr[('50x25', 'cpsat1800')]['selected']['n'] == 10
    else:
        assert calls == [] and touched == []
    for k in want:                                    # finite numbers only, no '+nan' path
        assert all(np.isfinite(prodd[k][lab]['mean']) for lab in ('s301', 'mean', 'selected'))
    assert prodr[('50x25', 'pool')]['selected']['n'] == 12
    assert prodd[('50x25', 'pool')]['selected']['hi'] < 0          # policy below the rules
    # the empty-comparison check: nothing to report on a complete tree
    assert ns['empty_comparisons'](solver_rows, False) == []


def test_p88_empty_comparisons_are_found_before_any_output_is_removed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PACT_ENV', 'legacy')
    monkeypatch.delenv('PACT_SEEDS', raising=False)
    monkeypatch.setenv('PACT_RESULTS', str(tmp_path / 'res'))
    production_tree(tmp_path, 'legacy', solver_sizes=('50x25',))
    seen = []
    ns = production_namespace(seen, None)
    found = ns['empty_comparisons'](True, False)
    assert found == ['80x25: no 1800 s solver record in or_solution/FAMILY/80x25_fairwarm1800s/'
                     '<regime> for ' + ', '.join(R8)]
    assert ns['empty_comparisons'](False, False) == []       # the exact environment reads none
    np.save(f'{REG.rule_dir("50x25")}/Result_SPT_LW.npy', np.zeros((0, 2)))
    found = ns['empty_comparisons'](False, False)
    assert found == [f'{REG.rule_dir("50x25")}/Result_SPT_LW.npy holds no instance']
    # the script itself: refuses with the reason and leaves the existing outputs in place
    import subprocess
    outs = [tmp_path / 'out' / 'paper' / 'macros_family.tex',
            tmp_path / 'out' / 'results' / 'family_inference.json',
            tmp_path / 'out' / 'results' / 'family_inference.md']
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('existing\n')
    envv = dict(os.environ, PACT_OUT=str(tmp_path / 'out'), PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    run = subprocess.run([sys.executable, P88], cwd=tmp_path, env=envv, capture_output=True,
                         text=True, timeout=120)
    assert run.returncode != 0
    assert 'REFUSING: a production comparison would hold no instance' in run.stderr
    assert '80x25: no 1800 s solver record' in run.stderr and 'holds no instance' in run.stderr
    assert all(p.read_text() == 'existing\n' for p in outs)


# ------------------------------------------------------------------ p88, sections F, G, H
SEEDS3 = (301, 302, 303)
BT = 500     # replicates: Holm over twelve pairs needs 12 / (B + 1) < 0.05 for a verdict
SIZES_N = {'10x25': 100, '20x25': 10, '30x25': 10, '50x15': 10, '80x15': 10, '50x25': 10,
           '80x25': 10}
# makespan of the specialist of c on r = released policy on r times FACTOR (1 elsewhere), so
#   F, policy vs specialist of A on B:       d = 100 (1 / FACTOR[A, B] - 1)
#   F, specialist of B vs specialist of A:   d = 100 (FACTOR[B, B] / FACTOR[A, B] - 1)
#   G, LSW specialist vs policy on r:        d = 100 (FACTOR[LSW, r] - 1)
FACTOR = {('10x25', 'N', 'L'): 1 / 0.97,        # F NToL: policy 3% shorter, specialist of L too
          ('10x25', 'L', 'LS'): 1 / 1.03,       # F LToLS: 3% longer
          ('10x25', 'LSW', 'N'): 1.04,          # G: the LSW specialist 4% longer on N
          ('20x25', 'N', 'N'): 1 / 0.98}        # H: policy 2% shorter than its specialist on N
SEED_FACTOR = {s: 1 + 0.005 * (s - 302) for s in SEEDS3}   # the same for both arms of a seed


def noise(n, *key):
    return 1 + 0.002 * np.random.default_rng(zlib.crc32(repr(key).encode())).standard_normal(n)


def save(path, x):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.stack([np.asarray(x, dtype=float), np.zeros(len(x))], axis=1))


def family_tree(root, seeds=SEEDS3):
    """A complete tree of the exact environment for p88 under root (cwd = root and
    PACT_RESULTS = root/res set by the caller): descriptors, arrays, training records.
    Rules: FIFO 5% above the base on even instances, SPT on odd ones, so the per-instance
    best is 5% above the base everywhere and no single rule is."""
    for folder, n in SIZES_N.items():
        for r in R8:
            d = root / 'data' / 'FAMILY' / folder / 'desc' / r
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                json.dump({'base_instance': f'{folder}/base_{i:03d}'},
                          open(d / f'instance_{i:03d}.desc.json', 'w'))
            base = 400 + 200 * np.random.default_rng(zlib.crc32(f'{folder}{r}'.encode())).random(n)
            even = np.arange(n) % 2 == 0
            rules = {'FIFO': np.where(even, 1.05, 1.08), 'MOR': 1.10,
                     'SPT': np.where(even, 1.08, 1.05), 'MWKR': 1.12}
            for q in F88.RULES:
                save(f'{REG.rule_dir(folder)}/Result_{q}_{r}.npy', base * rules[q])
            specs = R8 if folder == '10x25' else ([r] if folder in ('20x25', '30x25') else [])
            for s in seeds:
                pol = base * SEED_FACTOR[s] * noise(n, 'rel', folder, r, s)
                save(REG.result_path('released', None, s, folder, r), pol)
                if folder == '10x25':
                    save(REG.result_path('control', None, s, folder, r),
                         pol * 1.001 * noise(n, 'ctl', r, s))
                for c in specs:
                    save(REG.result_path('specialist', c, s, folder, r),
                         pol * FACTOR.get((folder, c, r), 1.0) * noise(n, 'spec', folder, c, r, s))
    (root / 'train_log' / 'FAMILY').mkdir(parents=True, exist_ok=True)
    for s, vals in zip(seeds, ([0.95, 0.91], [0.9, 0.86, 0.88], [0.93, 0.92])):   # selects 302
        (root / REG.vali_log('released', None, s)).write_text(repr(vals))
        (root / REG.train_log('released', None, s)).write_text(
            f'[vali] MWKR norms: N=1.0 L=1.0\nbest rel={min(vals):.6f}\n')


def published_files(out, seeds=SEEDS3):
    """macros_seed.tex as p27 writes it (p88 reproduces it with the same replicates) and
    macros_eq.tex with an interval for every combination."""
    _, C = F88.parity_components('10x25', seeds)
    words = [REG.SEED_WORDS[s] for s in seeds]
    seed = []
    for r in R8:
        seed.append(f'\\newcommand{{\\Sd{r}Lo}}{{{C[r]["lo"]:+.2f}}}')
        seed.append(f'\\newcommand{{\\Sd{r}Hi}}{{{C[r]["hi"]:+.2f}}}')
        seed += [f'\\newcommand{{\\Sd{r}{w}}}{{{m:+.2f}}}' for w, m in zip(words, C[r]['seed_means'])]
    eq = [f'\\newcommand{{\\Eq{k}{e}}}{{{v:+.2f}}}'
          for k in [f'Eone{r}' for r in F88.TRAIN] + ['Sw', 'Lsw'] for e, v in (('Lo', -0.3), ('Hi', 0.2))]
    (out / 'paper').mkdir(parents=True, exist_ok=True)
    (out / 'paper' / 'macros_seed.tex').write_text('\n'.join(seed) + '\n')
    (out / 'paper' / 'macros_eq.tex').write_text('\n'.join(eq) + '\n')


def tree_env(mp, root):
    mp.chdir(root)
    mp.setenv('PACT_ENV', 'exact')
    mp.setenv('PACT_SEEDS', '301-303')
    mp.setenv('PACT_RESULTS', str(root / 'res'))
    mp.setattr(F88, 'B', BT)
    mp.setattr(F88, 'ALPHA', 0.05)


@pytest.fixture(scope='module')
def family_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('family')
    with pytest.MonkeyPatch.context() as mp:
        tree_env(mp, root)
        family_tree(root)
        published_files(root / 'out')
    return root


@pytest.fixture
def in_tree(family_root, monkeypatch):
    tree_env(monkeypatch, family_root)
    return family_root


def paired(fa, fb, folder, r):
    """d [k, n] computed here: seed s of fa against seed s of fb."""
    return np.stack([SF.rel_gap(fa(s, folder, r), fb(s, folder, r)) for s in SEEDS3])


def test_p88_rule_change_pairs_are_twelve_one_rule_additions():
    assert len(F88.PAIRS) == 12
    for a, b in F88.PAIRS:
        assert set(F88.rules_of(a)) < set(F88.rules_of(b))
        assert len(F88.rules_of(b)) == len(F88.rules_of(a)) + 1
    assert ('N', 'L') in F88.PAIRS and ('LW', 'LSW') in F88.PAIRS
    assert F88.PAIR_NAME[('L', 'LS')] == 'LToLS'


def test_p88_section_F_sign_pairing_holm_and_pooling(in_tree):
    res = F88.section_F(SEEDS3)
    for key in ('Pol', 'Spec'):
        assert list(res[key]['rows']) == [F88.PAIR_NAME[p] for p in F88.PAIRS]
    pol, spec = res['Pol']['rows'], res['Spec']['rows']
    # sign: first-named method first; the policy 3% shorter on NToL, 3% longer on LToLS
    assert pol['NToL']['mean'] == pytest.approx(-3.0, abs=0.05) and pol['NToL']['hi'] < 0
    assert pol['NToL']['verdict'] == AH and pol['NToL']['direction'] == 'ahead'
    assert pol['LToLS']['mean'] == pytest.approx(3.0, abs=0.05) and pol['LToLS']['verdict'] == BH
    assert spec['NToL']['mean'] == pytest.approx(-3.0, abs=0.05) and spec['NToL']['verdict'] == AH
    assert spec['LToLS']['verdict'] == BH
    # pairing: seed s against seed s (the seed factor of 0.5% cancels), per pair as computed here
    D = {'Pol': {}, 'Spec': {}}
    for (a, b), name in F88.PAIR_NAME.items():
        D['Pol'][name] = paired(F88.released, F88.specialist_of(a), '10x25', b)
        D['Spec'][name] = paired(F88.specialist_of(b), F88.specialist_of(a), '10x25', b)
        for key in D:
            x = res[key]['rows'][name]
            assert x['mean'] == pytest.approx(D[key][name].mean(), abs=1e-12)
            assert x['seed_means'] == pytest.approx(D[key][name].mean(axis=1).tolist(), abs=1e-12)
            for mg in F88.MARGINS:
                assert x[f'v{mg}'] == SF.verdict(x['lo'], x['hi'], mg)
    assert all(abs(m) < 0.1 for m in pol['NToS']['seed_means'])
    for key in D:
        rows = [res[key]['rows'][F88.PAIR_NAME[p]] for p in F88.PAIRS]
        # Holm over the twelve pairs, each direction and each comparison its own family
        for col in ('p_t', 'p_boot'):
            assert [x[col + '_holm'] for x in rows] == pytest.approx(SF.holm([x[col] for x in rows]))
            assert [x[col + '_greater_holm'] for x in rows] == pytest.approx(
                SF.holm([x[col + '_greater'] for x in rows]))
        # the pooled row: one hierarchical bootstrap of [k, n, 12], a base instance with its pairs
        hb = SF.hier_boot(np.stack(list(D[key].values()), axis=2), B=BT, seed=11)
        pr = res[key]['pooled']
        assert (pr['mean'], pr['lo'], pr['hi'], pr['c']) == (hb['mean'], hb['lo'], hb['hi'], 12)
        assert sum(res[key]['counts'].values()) == 12
    assert res['Pol']['counts'][AH] >= 1 and res['Pol']['counts'][BH] >= 1


def test_p88_section_G_sign_combinations_and_pooling(in_tree):
    res = F88.section_G(SEEDS3)
    assert res['Pol']['combinations'] == R8
    assert res['Spec']['combinations'] == [r for r in R8 if r != 'LSW']      # not against itself
    for key, other in (('Pol', F88.released), ('Spec', F88.specialist)):
        for r in res[key]['combinations']:
            d = paired(F88.specialist_of('LSW'), other, '10x25_blindref', r)
            x = res[key]['rows'][r]
            assert x['mean'] == pytest.approx(d.mean(), abs=1e-12)
            draws = SF.hier_draws_stream(d, BT, np.random.default_rng(11))
            assert (x['lo'], x['hi']) == SF.percentile_ci(draws, 0.05)
            m = len(res[key]['combinations'])                     # Bonferroni over this family
            assert (x['sim_lo'], x['sim_hi']) == SF.percentile_ci(draws, 0.05 / m)
            assert x['direction'] == SF.direction(x['lo'], x['hi'])
    # the LSW specialist 4% longer on N: positive and behind, against both
    for key in ('Pol', 'Spec'):
        x = res[key]['rows']['N']
        assert x['mean'] == pytest.approx(4.0, abs=0.05) and x['direction'] == 'behind'
        assert x['v0.5'] == x['v1.0'] == 'inferior'
    assert set(res['Pol']['pooled']) == {'All', 'Other'} and set(res['Spec']['pooled']) == {'Other'}
    for key, fam, regs in (('Pol', 'All', R8), ('Pol', 'Other', R8[:-1]), ('Spec', 'Other', R8[:-1])):
        d3 = np.stack([paired(F88.specialist_of('LSW'),
                              F88.released if key == 'Pol' else F88.specialist, '10x25_blindref', r)
                       for r in regs], axis=2)
        hb = SF.hier_boot(d3, B=BT, seed=11)
        x = res[key]['pooled'][fam]
        assert (x['mean'], x['lo'], x['hi'], x['c']) == (hb['mean'], hb['lo'], hb['hi'], len(regs))


def test_p88_section_H_rule_pool_selected_run_and_specialists(in_tree):
    res = F88.section_H(302, SEEDS3)
    assert list(res) == ['20x25', '30x25', '50x15', '80x15']
    for size, x in res.items():
        assert x['n'] == 10
        # the per-instance best of the four rules is 5% above the base; the policy about 4.8%
        # below it on every combination: ahead, Holm over the eight combinations of the size
        for r in R8:
            want = np.stack([SF.rel_gap(F88.released(s, size, r),
                                        np.min([F88.REG.read_path(f'{REG.rule_dir(size)}/Result_{q}_{r}.npy')
                                                for q in F88.RULES], axis=0)) for s in SEEDS3])
            assert x['rule_pool'][r]['mean'] == pytest.approx(want.mean(), abs=1e-12)
            assert x['rule_pool'][r]['mean'] == pytest.approx(100 * (1 / 1.05 - 1), abs=0.2)
        rows = [x['rule_pool'][r] for r in R8]
        assert [y['p_t_holm'] for y in rows] == pytest.approx(SF.holm([y['p_t'] for y in rows]))
        assert [y['verdict'] for y in rows] == [AH] * 8
        assert x['rule_pool_counts'] == {AH: 8, BH: 0, ND: 0}
        assert x['rule_pool_pooled']['c'] == 8 and x['rule_pool_pooled']['hi'] < 0
        if size in ('20x25', '30x25'):
            assert 'selected' not in x and set(x['parity']) == set(R8)
            assert set(x['parity_pooled']) == {'All', 'Train', 'Held'}
            assert len(x['parity_iut']) == 3 * 2 * 2
        else:
            assert 'parity' not in x and x['selected']['seed'] == 302
            cells = np.stack([SF.rel_gap(F88.released(302, size, r), F88.pool(size, r))
                              for r in R8], axis=1)
            sel = x['selected']
            for j, r in enumerate(R8):
                ib = SF.instance_boot(cells[:, j], B=BT, seed=7, return_draws=True)
                y = sel['rows'][r]
                assert (y['mean'], y['lo'], y['hi']) == (ib['mean'], ib['lo'], ib['hi'])
                assert y['p_boot'] == SF.boot_p_one_sided(ib['draws'], 'less')
            for col in ('p_boot', 'p_boot_greater'):
                assert [sel['rows'][r][col + '_holm'] for r in R8] == pytest.approx(
                    SF.holm([sel['rows'][r][col] for r in R8]))
            ib = SF.instance_boot(cells, B=BT, seed=7)                 # pooled: a base instance
            assert (sel['pooled']['lo'], sel['pooled']['hi'], sel['pooled']['c']) == \
                (ib['lo'], ib['hi'], 8)                                 # with its eight cells
            assert sel['counts'][AH] == 8
    par = res['20x25']['parity']
    assert par['N']['mean'] == pytest.approx(-2.0, abs=0.15) and par['N']['direction'] == 'ahead'
    assert res['30x25']['parity']['N']['direction'] == 'neither'


def test_p88_section_state_skip_wrong_length_and_required(tmp_path, monkeypatch):
    tree_env(monkeypatch, tmp_path)
    family_tree(tmp_path)
    st = {sec: F88.section_state(sec, SEEDS3) for sec in F88.SECTIONS}
    assert all(x['ready'] and x['reason'] is None for x in st.values())
    assert st['F']['arrays'] == 78 and st['G']['arrays'] == 69 and st['H']['arrays'] == 272
    gone = REG.result_path('specialist', 'N', 301, '10x25', 'L')          # read by F only
    os.remove(gone)
    short = REG.result_path('released', None, 303, '50x15', 'SW')          # read by H only
    save(short, np.ones(9))
    st = {sec: F88.section_state(sec, SEEDS3) for sec in F88.SECTIONS}
    assert not st['F']['ready'] and st['F']['missing'] == [gone]
    assert st['F']['reason'] == f'1 of 78 input arrays missing (first {gone})'
    assert st['G']['ready']
    assert not st['H']['ready'] and st['H']['missing'] == []
    assert st['H']['wrong_length'] == [f'{short}: 9 rows, the test set has 10']
    assert F88.required_sections('none') == [] and F88.required_sections('all') == ['F', 'G', 'H']
    assert F88.required_sections('H,F') == ['F', 'H']
    with pytest.raises(SystemExit, match='unknown section'):
        F88.required_sections('F,X')


def test_p88_section_macro_names(in_tree):
    res = {'F': F88.section_F(SEEDS3), 'G': F88.section_G(SEEDS3), 'H': F88.section_H(302, SEEDS3)}
    state = {sec: F88.section_state(sec, SEEDS3) for sec in F88.SECTIONS}
    lines = F88.section_lines(state, res)
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', '\n'.join(lines))
    assert len(names) == len(set(names)) and all(re.fullmatch(r'Fam[A-Za-z]+', n) for n in names)
    mac = dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', '\n'.join(lines)))
    assert mac['FamChgPairN'] == '12'
    assert mac['FamChgPolNToLVerdict'] == AH and mac['FamChgPolLToLSVerdict'] == BH
    assert mac['FamChgPolNToL'] == F88.f2(res['F']['Pol']['rows']['NToL']['mean'])
    assert mac['FamChgPolNToLVHalf'] == 'non-inferior' and mac['FamChgPolPoolDir'] in (
        'ahead', 'behind', 'neither')
    assert int(mac['FamChgPolVerdictAheadN']) + int(mac['FamChgPolVerdictBehindN']) + \
        int(mac['FamChgPolVerdictNdN']) == 12
    assert mac['FamFullPolN'].startswith('+4.') and mac['FamFullPolNDir'] == 'behind'
    assert 'FamFullSpecLSW' not in mac and 'FamFullSpecPoolOther' in mac
    assert mac['FamShopTwentyN'] == '10' and mac['FamShopTwentyParNDir'] == 'ahead'
    assert mac['FamShopFiftyFifteenSelVerdictAheadN'] == '8' and 'FamShopFiftyFifteenParN' not in mac
    assert 'FamShopThirtyParIutAllHalfEq' in mac and 'FamShopEightyFifteenSelPoolLo' in mac
    # every name of F, G and H is new to the macro files of both manuscripts (the output itself
    # excluded, as the script's own collision check does)
    files = []
    for d in ('paper', 'manuscript', os.path.join('manuscript', 'sections_new'),
              os.path.join('manuscript', 'supp_new')):
        p = os.path.join(ROOT, d)
        if os.path.isdir(p):             # a manuscript folder that the release does not hold
            files += [os.path.join(p, x) for x in os.listdir(p)
                      if re.fullmatch(r'macros.*\.tex', x)]
    existing = set()
    for f in files:
        if os.path.basename(f) != 'macros_family.tex':
            existing |= set(re.findall(r'\\(?:newcommand|renewcommand|providecommand)\*?\{?\\([A-Za-z]+)',
                                       open(f, errors='replace').read()))
    assert not set(names) & existing
    # a skipped section is one comment line and no macro
    lines = F88.section_lines(dict(state, G=dict(state['G'], reason='1 of 69 input arrays missing')),
                              dict(res, G=None))
    skipped = [x for x in lines if 'Section G' in x]
    assert skipped == ['% Section G (specialist of the full combination LSW, 10x25) skipped: '
                       '1 of 69 input arrays missing']
    assert not any('FamFull' in x for x in lines)


def run_p88(root, *args):
    env = dict(os.environ, PACT_ENV='exact', PACT_SEEDS='301-303', PACT_RESULTS=str(root / 'res'),
               PACT_OUT=str(root / 'out'), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable, P88, '--boot', str(BT), *args], cwd=root, env=env,
                          capture_output=True, text=True, timeout=600)


P88_OUTS = ('paper/macros_family.tex', 'results/family_inference.json', 'results/family_inference.md')


def test_p88_script_computes_skips_and_refuses_sections(tmp_path, monkeypatch):
    tree_env(monkeypatch, tmp_path)
    family_tree(tmp_path)
    published_files(tmp_path / 'out')
    out = tmp_path / 'out'
    run = run_p88(tmp_path, '--require-sections', 'all')
    assert run.returncode == 0, run.stderr[-2000:]
    full = (out / P88_OUTS[0]).read_text()
    js = json.load(open(out / P88_OUTS[1]))
    assert all(js['sections'][s]['computed'] for s in F88.SECTIONS)
    assert js['rule_change']['Pol']['rows']['NToL']['verdict'] == AH
    assert js['selection']['seed'] == 302 and js['larger_shops']['50x15']['selected']['seed'] == 302
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', full)
    assert len(names) == len(set(names)) and all(re.fullmatch(r'Fam[A-Za-z]+', n) for n in names)
    head = full.split('\n% F. Rule-change pairs')[0]
    assert names.index('FamChgPairN') == len(re.findall(r'\\newcommand', head))   # new names last
    # F and G lose one input each: both skipped with the reason, H computed, the script finishes
    os.remove(REG.result_path('specialist', 'N', 301, '10x25', 'L'))
    os.remove(REG.result_path('specialist', 'LSW', 302, '10x25_blindref', 'N'))
    run = run_p88(tmp_path)
    assert run.returncode == 0, run.stderr[-2000:]
    text = (out / P88_OUTS[0]).read_text()
    assert text.split('\n% Section F')[0] == head                 # the existing part unchanged
    assert '% Section F (rule-change pairs, 10x25) skipped: 1 of 78 input arrays missing' in text
    assert '% Section G (' in text and 'FamChg' not in text and 'FamFull' not in text
    assert '\\newcommand{\\FamShopTwentyN}{10}' in text
    js = json.load(open(out / P88_OUTS[1]))
    sec = js['sections']
    assert (sec['F']['computed'], sec['G']['computed'], sec['H']['computed']) == (False, False, True)
    assert sec['F']['missing_n'] == 1 and sec['F']['reason'].startswith('1 of 78 input arrays missing')
    assert js['rule_change'] is None and js['full_combination_specialist'] is None
    md = (out / P88_OUTS[2]).read_text()
    assert '- F, rule-change pairs, 10x25: skipped: 1 of 78 input arrays missing' in md
    assert '- H, larger shops: 20x25, 30x25, 50x15, 80x15: computed' in md
    assert '## H. Larger shops' in md and '## F. Rule-change' not in md
    run = run_p88(tmp_path, '--require-sections', 'H')
    assert run.returncode == 0, run.stderr[-2000:]
    # a required section that would be skipped refuses, and no output is left behind
    for arg, named in (('all', ('F (', 'G (')), ('F,H', ('F (',))):
        run = run_p88(tmp_path, '--require-sections', arg)
        assert run.returncode != 0
        assert f'REFUSING: --require-sections {arg}' in run.stderr
        assert all(x in run.stderr for x in named) and 'H (' not in run.stderr
        assert not any((out / p).exists() for p in P88_OUTS)
    run = run_p88(tmp_path, '--require-sections', 'Q')
    assert run.returncode != 0 and 'unknown section' in run.stderr
