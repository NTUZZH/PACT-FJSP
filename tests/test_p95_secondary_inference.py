"""scripts/p95_secondary_inference.py on synthetic result trees: sign, pairing by seed
number, clustering of the pooled rows by base instance, the optional block, the
refusals, the check on the earlier environment and the macro names.

Run from the repository root:
    python -m pytest tests/test_p95_secondary_inference.py -q
CPU only, a few seconds. Every tree is written under a temporary folder (six test
instances per combination); the repository is read only by the test that compares
the paths with the evaluation plan (skipped without data/FAMILY/10x25).
"""
import json
import os
import re
import sys
import time

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import arm_registry as REG  # noqa: E402
import eval_newseeds_plan as PLAN  # noqa: E402
import hgnn_select_lr as HLR  # noqa: E402
import p95_secondary_inference as P  # noqa: E402
import stats_family as SF  # noqa: E402

R8 = list(REG.COMBOS)
SIX = ','.join(REG.TRAIN_COMBOS)
N_INST = 6
BOOT = '400'
RATE = '1e-4'                 # the tuning records below make this the selected rate
HGNN_FACTOR = {301: 0.968, 302: 0.970, 303: 0.972, 304: 0.966, 305: 0.974}   # policy = HGNN x factor
T0 = time.time() - 3600       # checkpoints are older than every array


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ('PACT_ENV', 'PACT_RESULTS', 'PACT_SEEDS', 'PACT_OUT'):
        monkeypatch.delenv(v, raising=False)


# ------------------------------------------------------------------ tree builder
def write(root, rel, text):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(text)
    return p


def save(root, rel, ms):
    p = os.path.join(root, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, np.stack([np.asarray(ms, float), np.full(len(ms), 0.5)], axis=1))


def write_data(root):
    for r in R8:
        for i in range(N_INST):
            write(root, f'data/FAMILY/10x25/desc/{r}/instance_{i:03d}.desc.json',
                  json.dumps({'base_instance': f'data/PPVC/10x25+ppvc-mixed/instance_{i:03d}',
                              'regime': r}))


def base(seed, combo):
    rng = np.random.default_rng(seed * 100 + R8.index(combo))
    return rng.integers(100, 160, N_INST).astype(float)


def write_run(root, run, values, cfg):
    ck = write(root, run.ckpt, 'checkpoint')
    os.utime(ck, (T0, T0))
    write(root, run.log, '[train] done in 1.00 h; best rel=0.900000\n')
    write(root, run.config, json.dumps(cfg))
    for c, ms in values.items():
        save(root, run.array(c), ms)


def family_cfg(run, updates, regimes, **extra):
    c = dict(seed_train=run.seed, max_updates=updates, num_envs=20, family_regimes=regimes,
             first_mask='exact', mask_ties='exact')
    c.update(extra)
    return c


def hgnn_cfg(run, reward, iterations, regimes, rate):
    return dict(hgnn_reward=reward, hgnn_mask='exact', n_params=HLR.PARAMS,
                args=dict(seed=run.seed, max_iterations=iterations, num_envs=20,
                          family_regimes=regimes, lr=float(rate)),
                train_paras=dict(lr=float(rate)), vali_norm={'N@10': 150.0})


def write_tuning(root):
    """The three tuning runs of scripts/hgnn_select_lr.py; RATE scores best. The run at
    RATE is also the seed-301 run of the PACT-reward HGNN, written by write_hgnn."""
    for lr in HLR.RATES:
        name = HLR.run_name(lr)
        run = P.Run(name, 'hgnn', 301, P.FOLDER)
        best = 0.9 if lr == RATE else 0.95
        curve = [best + 0.01 * (i % 7) for i in range(HLR.ITERATIONS // 10)]
        write(root, run.config, json.dumps(hgnn_cfg(run, 'certified', HLR.ITERATIONS, SIX, lr)))
        write(root, f'train_log/FAMILY/valiquality_{REG.PREFIX}{name}.txt', repr(curve))
        write(root, run.log, f'done in 3.10 h; best rel={best:.6f}\n')


ALL8 = ','.join(R8)


def build(root, arms=('eq', 'sparse', 'dzero', 'dmax', 'film', 'dr', 'wthird', 'all8', 'pair'),
          hgnn=True, solo=False, clustered=False, hgnn_pact_seeds=(301, 302, 303), factors=None):
    """An exact-environment tree: policy and specialists on seeds 301-305, the named
    supporting arms on 301-303, HGNN (PACT reward on hgnn_pact_seeds, published reward
    on 301-303, the selected rate) and, with solo, the HGNN specialists (seed 301).
    Seeds 304 and 305 of the policy have makespans ten times larger, so a comparison
    that read them against another seed would show it; the HGNN runs of seeds 304 and
    305 are ten times larger too, so that seed s against seed s gives the factor.
    factors {block name: f} makes that arm f times the policy's base makespans."""
    factors = factors or {}
    write_data(root)
    with P.environment_set('exact'):
        pol = P.principal('released', None, 'exact', P.FOLDER)
        for s, run in pol.runs.items():
            k = 10.0 if s > 303 else 1.0
            write_run(root, run, {r: k * base(s, r) for r in R8}, family_cfg(run, 2000, SIX))
        for r in R8:
            spec = P.principal('specialist', r, 'exact', P.FOLDER)
            combos = sorted(set([r] + (list(dict(PLAN.SECONDARY_EVAL[P.WTHIRD])[P.FOLDER])
                                       if r == P.WTHIRD_COMBO else [])))
            for s, run in spec.runs.items():
                f = 1.0 + 0.003 * (s - 302)
                write_run(root, run, {c: f * base(s, c) for c in combos}, family_cfg(run, 1000, r))
        for stem in REG.SECONDARY_ARMS:
            name = P.block_name(stem)
            if name not in arms:
                continue
            arm = P.secondary(stem, 'exact', P.FOLDER)
            combos = dict(PLAN.SECONDARY_EVAL[stem])[P.FOLDER]
            for s, run in arm.runs.items():
                if name in factors:
                    vals = {r: factors[name] * base(s, r) for r in combos}
                elif name == 'eq':
                    vals = {r: 1.02 * base(s, r) for r in combos}
                elif name == 'sparse' and clustered:
                    # d = a_R z_i with sum over the combinations of a_R = 0 within every
                    # family: the mean over the combinations of a base instance is zero
                    z = np.linspace(0.5, 3.0, N_INST) * (1 + 0.1 * (s - 302))
                    a = {r: (1.0 if j % 2 == 0 else -1.0) for j, r in enumerate(R8)}
                    vals = {r: base(s, r) * (1 + a[r] * z / 100.0) for r in combos}
                else:
                    vals = {r: (1 + 0.001 * (s - 300)) * base(s, r) for r in combos}
                if name == 'wthird':
                    cfg = family_cfg(run, 333, 'W')
                elif name == 'all8':
                    cfg = family_cfg(run, 2000, ALL8)
                elif name == 'pair':
                    cfg = family_cfg(run, 2000, SIX, pair_extra='sw')
                else:
                    cfg = family_cfg(run, 6000 if name == 'eq' else 2000, SIX)
                write_run(root, run, vals, cfg)
        if hgnn:
            write_tuning(root)
            tag = P.lr_tag(RATE)
            for reward, extra, seeds in (('certified', 0.0, hgnn_pact_seeds),
                                         ('published', 0.01, (301, 302, 303))):
                arm = P.hgnn_arm(reward, 'joint', seeds, tag, 'exact', P.FOLDER)
                for s, run in arm.runs.items():
                    k = 10.0 if s > 303 else 1.0
                    vals = {r: k * base(s, r) / (HGNN_FACTOR[s] - extra) for r in R8}
                    write_run(root, run, vals, hgnn_cfg(run, reward, 2000, SIX, RATE))
            if solo:
                for r in R8:
                    arm = P.hgnn_arm('certified', r, P.HGNN_SOLO_SEEDS, tag, 'exact', P.FOLDER)
                    for s, run in arm.runs.items():
                        write_run(root, run, {r: 1.004 * base(s, r) / HGNN_FACTOR[s]},
                                  hgnn_cfg(run, 'certified', 1000, r, RATE))


def run_main(root, *extra):
    out = os.path.join(root, 'out')
    res = P.main(['--root', root, '--out', out, '--boot', BOOT] + list(extra))
    return res, out


def macros(path):
    return dict(re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}', open(path).read()))


def mac_path(out, check=False):
    return os.path.join(out, 'manuscript', 'macros_secondary_check.tex' if check
                        else 'macros_secondary.tex')


# ------------------------------------------------------------------ paths
def test_paths_follow_the_registry_and_hgnn_layout():
    P.check_layout()
    with P.environment_set('exact'):
        r = P.secondary('xmask-eq', 'exact', P.FOLDER).runs[301]
        assert r.array('N') == 'test_results/FAMILY/10x25_xmask/Result_greedy+10x25+family+xmask-eq-s301_N.npy'
        assert (r.ckpt, r.log, r.config) == (
            'trained_network/FAMILY/10x25+family+xmask-eq-s301.pth', 'train_log/FAMILY_xmask-eq-s301.log',
            'train_log/FAMILY/config_10x25+family+xmask-eq-s301.json')
        h = P.hgnn_arm('certified', 'joint', (301, 302, 303, 304, 305), P.lr_tag('1e-4'), 'exact',
                       P.FOLDER).runs
        assert h[302].array('LS') == 'test_results/FAMILY/10x25_xmask/Result_hgnn-certified-joint-s302-xmask-lr1e-4_LS.npy'
        assert h[305].name == 'hgnn-certified-joint-s305-xmask-lr1e-4'
        assert P.hgnn_arm('published', 'joint', (301, 302, 303), P.lr_tag('2e-4'), 'exact',
                          P.FOLDER).runs[301].name == 'hgnn-published-joint-s301-xmask'
    with P.environment_set('legacy'):
        w = P.secondary('xmask-wthird', 'legacy', P.FOLDER).runs
        assert w[301].array('W') == 'test_results/FAMILY/10x25/Result_greedy+10x25+family+spec-W-third_W.npy'
        assert w[302].name == 'spec-W-third-s302'
        assert P.principal('released', None, 'legacy', '10x25_cpuref').runs[301].array('S') == \
            'test_results/FAMILY/10x25_cpuref/Result_greedy+10x25+family+jointblind-v1_S.npy'
        assert P.hgnn_arm('certified', 'N', (301,), '', 'legacy', P.FOLDER).runs[301].name == \
            'hgnn-certified-N-s301'
    assert os.environ.get('PACT_ENV') is None            # restored after each use


def test_arrays_read_are_the_arrays_the_plan_writes(monkeypatch):
    if not os.path.isdir(os.path.join(ROOT, 'data', 'FAMILY', '10x25', 'desc')):
        pytest.skip('no data/FAMILY/10x25 in this checkout')
    monkeypatch.chdir(ROOT)
    if not {'principal', 'secondary'} <= set(PLAN.GROUPS):
        pytest.skip('EVAL_GROUPS leaves out the principal or secondary group')
    planned = {f for j in PLAN.exact_jobs() if j['group'] in ('principal', 'secondary')
               for f in j['files']}
    with P.environment_set('exact'):
        blocks, _ = P.build_blocks(ROOT, 'exact', None, 'no rate')
    read = [run.array(c) for b in blocks for run, combos, _, _ in b.reads for c in combos]
    assert read and set(read) <= planned, sorted(set(read) - planned)[:5]
    # the HGNN blocks read what the plan's group hgnn writes, for every seed set
    monkeypatch.setenv('HGNN_LR', '2e-4')
    monkeypatch.setenv('HGNN_SOLO', '1')
    planned_h = {f for j in PLAN.hgnn_jobs() for f in j['files']}
    with P.environment_set('exact'):
        blocks, _ = P.build_blocks(ROOT, 'exact', '2e-4', '')
        own = [run.array(c) for b in blocks if b.name in ('hgnn', 'hgnnpub', 'hgnnsolo')
               for run, combos, o, _ in b.reads if o for c in combos]
        five = P.hgnn_arm('certified', 'joint', P.HGNN_SEED_SETS['certified'][0], P.lr_tag('2e-4'),
                          'exact', P.FOLDER)
        own += [run.array(c) for run in five.runs.values() for c in R8]
    assert own and set(own) <= planned_h, sorted(set(own) - planned_h)[:5]
    assert len(planned_h) == 8 * 8 + 8


# ------------------------------------------------------------------ sign and pairing
def test_sign_pairing_and_superiority(tmp_path):
    root = str(tmp_path)
    build(root)
    res, out = run_main(root, '--require', 'all')
    assert res['refused'] == {} and 'hgnnsolo' in res['skipped']
    m = macros(mac_path(out))
    # eq = 1.02 x policy on seeds 301-303; the x10 seeds 304-305 of the policy are not read
    for r in R8:
        assert m[f'SecEqVsPol{r}'] == '+2.00' and m[f'SecEqVsPol{r}Dir'] == 'behind'
        assert m[f'SecEqVsPol{r}Lo'] == '+2.00' and m[f'SecEqVsPol{r}Hi'] == '+2.00'
    assert m['SecEqVsPolPoolAll'] == '+2.00'
    # eq against the specialist: 1.02 / (1 + 0.003 (s - 302)) - 1, mean over the seeds
    want = np.mean([100 * (1.02 / (1 + 0.003 * (s - 302)) - 1) for s in (301, 302, 303)])
    assert m['SecEqParW'] == P.f2(want)
    # policy minus HGNN over HGNN: factor - 1 per seed, negative, ahead
    want = np.mean([100 * (HGNN_FACTOR[s] - 1) for s in (301, 302, 303)])
    for r in R8:
        assert m[f'SecPolHgnn{r}'] == P.f2(want) and m[f'SecPolHgnn{r}Dir'] == 'ahead'
        assert m[f'SecPolHgnn{r}Verdict'] == 'ahead'
    assert m['SecPolHgnnVerdictAheadN'] == '8' and m['SecPolHgnnVerdictBehindN'] == '0'
    assert m['SecPolHgnnSeedN'] == '3' and m['SecPolHgnnPubSeedN'] == '3'
    assert r'\newcommand{\SecHgnnLr}{1\times10^{-4}}' in open(mac_path(out)).read()
    assert m['SecHgnnIterations'] == '2,000'
    assert m['SecEqUpdates'] == '6,000' and m['SecEqEpisodes'] == '20,000'
    assert m['SecWthirdUpdates'] == '333' and m['SecWthirdEpisodes'] == '6,660'
    assert m['SecPolEpisodes'] == '6,667' and m['SecSpecEpisodes'] == '20,000'
    j = json.load(open(os.path.join(out, 'results', 'secondary_inference.json')))
    assert j['hgnn_learning_rate'] == RATE and j['blocks']['eq']['result']['seeds'] == [301, 302, 303]
    assert j['check'] is False
    h = j['blocks']['hgnn']
    assert h['result']['seeds'] == [301, 302, 303] and h['title'] == 'policy vs HGNN (PACT reward)'
    assert (h['result']['reward'], h['result']['hgnn_reward']) == ('PACT', 'certified')
    # the text says 'PACT reward'; 'certified' stays only in run names and configuration values
    texts = [open(p).read() for p in (mac_path(out), os.path.join(out, 'results', 'secondary_inference.md'),
                                      os.path.join(out, 'results', 'secondary_inference.json'))]
    for t in texts:
        assert 'PACT reward' in t and 'certified reward' not in t and '(certified' not in t
    md = texts[1]
    assert 'PACT reward; seeds 301, 302, 303 (HGNN seed s against policy seed s' in md


def test_pairs_use_the_seed_number_not_the_position():
    a = P.Arm('a', {s: P.Run(f'a{s}', 'family', s, P.FOLDER) for s in (302, 303)})
    b = P.Arm('b', {s: P.Run(f'b{s}', 'family', s, P.FOLDER) for s in (301, 302, 303)})
    cache = {}
    for s in (301, 302, 303):
        cache[(f'b{s}', P.FOLDER, 'N')] = np.full(4, float(s))
    for s in (302, 303):
        cache[(f'a{s}', P.FOLDER, 'N')] = np.full(4, float(s) * 1.01)
    seeds = P.common(a, b)
    assert seeds == (302, 303)
    d = P.pairs(a, b, seeds, 'N', cache)
    np.testing.assert_allclose(d, 1.0)                  # seed 302 meets 302, 303 meets 303
    assert P.common(a, b, within=(303, 304)) == (303,)


def test_hgnn_pact_reward_reads_five_seeds_when_all_five_exist(tmp_path):
    root = str(tmp_path)
    build(root, hgnn_pact_seeds=(301, 302, 303, 304, 305))
    res, out = run_main(root, '--require', 'all')
    assert res['refused'] == {}
    x = res['blocks']['hgnn']['result']
    assert x['seeds'] == [301, 302, 303, 304, 305] and x['components']['N']['k'] == 5
    assert x['seed_sets'] == [[301, 302, 303, 304, 305], [301, 302, 303]]
    assert res['blocks']['hgnnpub']['result']['seeds'] == [301, 302, 303]
    m = macros(mac_path(out))
    # policy and HGNN seeds 304 and 305 are both x10: seed s against seed s gives factor - 1
    want = np.mean([100 * (HGNN_FACTOR[s] - 1) for s in (301, 302, 303, 304, 305)])
    for r in R8:
        assert m[f'SecPolHgnn{r}'] == P.f2(want) and m[f'SecPolHgnn{r}Verdict'] == 'ahead'
    assert m['SecPolHgnnSeedN'] == '5' and m['SecPolHgnnPubSeedN'] == '3'
    md = open(os.path.join(out, 'results', 'secondary_inference.md')).read()
    assert 'PACT reward; seeds 301, 302, 303, 304, 305 (HGNN seed s against policy seed s' in md
    assert '| hgnn | computed | 301,302,303,304,305 |' in md
    line = next(ln for ln in open(mac_path(out)) if '\\SecPolHgnnN}' in ln)
    assert 'PACT reward, seeds 301-305' in line and 'over 5 seed pairs' in line


@pytest.mark.parametrize('extra', ['one more seed', 'one seed started', 'both started'])
def test_hgnn_pact_reward_refuses_another_set_of_runs(tmp_path, extra):
    root = str(tmp_path)
    build(root, hgnn_pact_seeds=(301, 302, 303, 304) if extra == 'one more seed' else (301, 302, 303))
    with P.environment_set('exact'):
        for s in {'one seed started': (304,), 'both started': (304, 305)}.get(extra, ()):
            run = P.hgnn_arm('certified', 'joint', (s,), P.lr_tag(RATE), 'exact', P.FOLDER).runs[s]
            write(root, run.log, 'Ep 1\n')            # training started, nothing evaluated
    res, out = run_main(root)
    assert list(res['refused']) == ['hgnn'] and 'hgnnpub' in res['computed']
    why = ' '.join(res['refused']['hgnn'])
    if extra == 'both started':                         # the five-seed set, incomplete
        assert 'missing' in why and 'completion' in why and 'has runs of seeds' not in why
    else:
        assert 'has runs of seeds 301,302,303,304; the block reads seeds 301-305 or 301-303' in why
    assert not any(k.startswith('SecPolHgnn') and not k.startswith('SecPolHgnnPub')
                   for k in macros(mac_path(out)))
    with pytest.raises(SystemExit, match='required blocks not computed'):
        run_main(root, '--require', 'all')


# ------------------------------------------------------------------ clustering
def test_pooled_interval_keeps_the_combinations_of_a_base_instance_together():
    rng = np.random.default_rng(3)
    z = rng.uniform(0.5, 3.0, (3, 40))
    d3 = np.stack([z, -z], axis=2)                       # [k, n, c]; mean over c is zero
    c, _ = P.component(d3, 500)
    assert abs(c['lo']) < 1e-12 and abs(c['hi']) < 1e-12 and c['c'] == 2
    flat = SF.hier_boot(d3.reshape(3, -1), B=500, seed=11)   # combinations as instances
    assert flat['hi'] - flat['lo'] > 0.1
    one, _ = P.component(d3[:1], 500)                    # one run: instance bootstrap
    assert one['scheme'] == 'instance' and abs(one['hi'] - one['lo']) < 1e-12


def test_pooled_rows_of_a_block_cluster_by_base_instance(tmp_path):
    root = str(tmp_path)
    build(root, arms=('sparse',), hgnn=False, clustered=True)
    res, out = run_main(root)
    x = res['blocks']['sparse']['result']['vs_policy']
    for fam in ('All', 'Train', 'Held'):
        assert abs(x['pooled'][fam]['lo']) < 1e-9 and abs(x['pooled'][fam]['hi']) < 1e-9
    assert all(x['components'][r]['hi'] - x['components'][r]['lo'] > 0.5 for r in R8)
    m = macros(mac_path(out))
    assert m['SecSparseVsPolPoolAllDir'] == 'neither'


# ------------------------------------------------------------------ all eight combinations, pair channels
def test_all_eight_and_pair_arms_against_the_policy(tmp_path):
    """all8 = 0.995 x policy and pair = 1.004 x policy on every combination and seed: the
    per-combination rows and the pooled rows over the six training combinations, over SW
    and LSW and over all eight give exactly -0.50 and +0.40; the effort counts eight
    combinations for all8 (2000 x 20 / 8 episodes each)."""
    root = str(tmp_path)
    build(root, hgnn=True, factors={'all8': 0.995, 'pair': 1.004})
    res, out = run_main(root, '--require', 'all')
    assert res['refused'] == {} and {'all8', 'pair'} <= set(res['computed'])
    m = macros(mac_path(out))
    for w, want in (('AllEight', '-0.50'), ('Pair', '+0.40')):
        for r in R8:
            assert m[f'Sec{w}VsPol{r}'] == want and m[f'Sec{w}VsPol{r}Lo'] == want
        for fam in ('Train', 'Held', 'All'):
            assert m[f'Sec{w}VsPolPool{fam}'] == want and m[f'Sec{w}VsPolPool{fam}Hi'] == want
            assert m[f'Sec{w}VsPolPool{fam}Dir'] == ('ahead' if w == 'AllEight' else 'behind')
        assert f'Sec{w}ParPoolHeld' in m and f'Sec{w}ParIutAllOneEq' in m
    assert (m['SecAllEightUpdates'], m['SecAllEightEpisodes'], m['SecAllEightCombos']) == ('2,000', '5,000', '8')
    assert (m['SecPairUpdates'], m['SecPairEpisodes'], m['SecPairCombos']) == ('2,000', '6,667', '6')
    x = res['blocks']['all8']['result']
    assert x['vs_policy']['pooled']['Held']['combos'] == ['SW', 'LSW'] and x['seeds'] == [301, 302, 303]
    assert x['effort']['per_seed']['s301']['combos'] == R8
    assert all(re.fullmatch(r'Sec[A-Za-z]+', k) for k in m)


@pytest.mark.parametrize('arm,key,bad', [('pair', 'pair_extra', 'none'),
                                         ('all8', 'family_regimes', SIX)])
def test_all_eight_and_pair_runs_must_record_their_option(tmp_path, arm, key, bad):
    root = str(tmp_path)
    build(root, arms=('eq', arm), hgnn=False)
    with P.environment_set('exact'):
        run = P.secondary(f'xmask-{arm}', 'exact', P.FOLDER).runs[302]
    cfg = json.load(open(os.path.join(root, run.config)))
    cfg[key] = bad
    write(root, run.config, json.dumps(cfg))
    res, out = run_main(root)
    assert list(res['refused']) == [arm] and 'eq' in res['computed']
    assert any(f'{key}={bad!r}' in p for p in res['refused'][arm])
    w = P.word(f'xmask-{arm}')
    assert not any(k.startswith(f'Sec{w}') for k in macros(mac_path(out)))


# ------------------------------------------------------------------ optional block
def test_hgnn_specialists_are_optional_and_use_the_one_run_interval(tmp_path):
    root = str(tmp_path)
    build(root, solo=True)
    res, out = run_main(root, '--require', 'all')
    assert 'hgnnsolo' in res['computed']
    m = macros(mac_path(out))
    want = 100 * (1 / 1.004 - 1)
    assert m['SecHgnnParN'] == P.f2(want) and m['SecHgnnSpecIterations'] == '1,000'
    line = next(ln for ln in open(mac_path(out)) if '\\SecHgnnParN}' in ln)
    assert 'one training run' in line
    assert res['blocks']['hgnnsolo']['result']['seeds'] == [301]
    assert res['blocks']['hgnnsolo']['result']['components']['N']['scheme'] == 'instance'


def test_absent_arms_are_skipped_and_required_blocks_refuse(tmp_path):
    root = str(tmp_path)
    build(root, arms=('eq',), hgnn=False)
    res, out = run_main(root)
    assert res['computed'] == ['policy', 'eq']
    assert {'sparse', 'dr', 'wthird', 'hgnn', 'hgnnpub', 'hgnnsolo'} <= set(res['skipped'])
    assert 'no result array' in res['skipped']['dr']
    assert 'learning rate not selected' in res['skipped']['hgnn']
    with pytest.raises(SystemExit, match='required blocks not computed'):
        run_main(root, '--require', 'all')
    assert not os.path.exists(mac_path(out))             # a refusal leaves no output
    res, _ = run_main(root, '--require', 'policy,eq')
    assert res['refused'] == {}
    with pytest.raises(SystemExit, match='unknown blocks'):
        run_main(root, '--require', 'eq,nosuchblock')


# ------------------------------------------------------------------ refusals
def eq_run(seed):
    with P.environment_set('exact'):
        r = P.secondary('xmask-eq', 'exact', P.FOLDER).runs[seed]
        return r, {c: r.array(c) for c in R8}


@pytest.mark.parametrize('damage', ['missing seed', 'wrong shape', 'older than checkpoint',
                                    'no completion line', 'other mask'])
def test_an_inconsistent_block_refuses(tmp_path, damage):
    root = str(tmp_path)
    build(root, arms=('eq', 'dzero'), hgnn=False)
    run, arrays = eq_run(303)
    p = os.path.join(root, arrays['W'])
    if damage == 'missing seed':
        os.remove(p)
    elif damage == 'wrong shape':
        np.save(p, np.ones((N_INST - 1, 2)) * 100)
    elif damage == 'older than checkpoint':
        os.utime(p, (T0 - 60, T0 - 60))
    elif damage == 'no completion line':
        write(root, run.log, 'training interrupted\n')
    else:
        cfg = json.load(open(os.path.join(root, run.config)))
        cfg['first_mask'] = None
        write(root, run.config, json.dumps(cfg))
    res, out = run_main(root)
    assert list(res['refused']) == ['eq'] and 'dzero' in res['computed']
    why = ' '.join(res['refused']['eq'])
    assert {'missing seed': 'missing', 'wrong shape': 'shape', 'older than checkpoint': 'older',
            'no completion line': 'completion', 'other mask': 'first_mask'}[damage] in why
    m = macros(mac_path(out))
    assert not any(k.startswith('SecEq') for k in m) and 'SecDzeroVsPolN' in m
    with pytest.raises(SystemExit, match='required blocks not computed'):
        run_main(root, '--require', 'eq')
    assert not os.path.exists(mac_path(out))


def test_hgnn_arrays_without_a_learning_rate_refuse(tmp_path):
    root = str(tmp_path)
    build(root, arms=(), hgnn=False)
    save(root, f'test_results/FAMILY/10x25_xmask/Result_hgnn-certified-joint-s302-xmask_N.npy',
         base(302, 'N'))
    res, _ = run_main(root)
    assert set(res['refused']) == {'hgnn', 'hgnnpub', 'hgnnsolo'}
    res, _ = run_main(root, '--hgnn-lr', '2e-4')        # named, but its other arrays are missing
    assert 'hgnn' in res['refused'] and any('missing' in p for p in res['refused']['hgnn'])
    with pytest.raises(SystemExit, match='is not one of'):
        run_main(root, '--hgnn-lr', '3e-4')


def test_check_mode_writes_only_outside_the_paper_and_results_folders(tmp_path):
    root = str(tmp_path)
    write_data(root)
    for out in (root, os.path.join(root, 'manuscript'), os.path.join(root, 'paper', 'x'),
                os.path.join(root, 'results'), os.path.join(root, 'test_results', 'x'), ROOT,
                os.path.join(ROOT, 'manuscript')):
        with pytest.raises(SystemExit, match='outside'):
            P.main(['--earlier-check', '--root', root, '--out', out, '--boot', BOOT])
    with pytest.raises(SystemExit, match='needs --out'):
        P.main(['--earlier-check', '--root', root])
    assert not os.path.exists(os.path.join(root, 'manuscript'))
    assert not os.path.exists(os.path.join(ROOT, 'manuscript', 'macros_secondary_check.tex'))


def test_check_mode_reads_the_stored_counterparts(tmp_path):
    root, out = str(tmp_path / 'repo'), str(tmp_path / 'check')
    write_data(root)
    with P.environment_set('legacy'):
        pol = P.principal('released', None, 'legacy', P.FOLDER)
        for s in (301, 302, 303):
            run = pol.runs[s]
            write_run(root, run, {r: base(s, r) for r in R8}, dict(seed_train=s, max_updates=2000,
                                                                   num_envs=20, family_regimes=SIX))
        for r in R8:
            for s in (301, 302, 303):
                run = P.principal('specialist', r, 'legacy', P.FOLDER).runs[s]
                write_run(root, run, {r: base(s, r)}, dict(seed_train=s, max_updates=1000,
                                                          num_envs=20, family_regimes=r))
        for s, run in P.secondary('xmask-eq', 'legacy', P.FOLDER).runs.items():
            write_run(root, run, {r: 0.99 * base(s, r) for r in R8},
                      dict(seed_train=s, max_updates=6000, num_envs=20, family_regimes=SIX))
    assert pol.runs[301].name == 'jointblind-v1'          # the bare seed-301 name
    res = P.main(['--earlier-check', '--root', root, '--out', out, '--boot', BOOT])
    assert res['check'] is True and res['computed'] == ['policy', 'eq']
    assert 'sparse' in res['not_applicable']
    path = mac_path(out, check=True)
    m = macros(path)
    assert m['SecEqVsPolN'] == '-1.00'
    lines = open(path).read().splitlines()
    assert 'CHECK ONLY' in lines[1]
    assert all('CHECK' in ln for ln in lines if ln.startswith('\\newcommand'))
    assert os.path.exists(os.path.join(out, 'results', 'secondary_inference_check.json'))
    assert open(os.path.join(out, 'results', 'secondary_inference_check.md')).readline().startswith('# CHECK')
    assert os.environ.get('PACT_ENV') is None


def test_the_environment_variable_must_not_contradict_the_mode(tmp_path, monkeypatch):
    root = str(tmp_path)
    write_data(root)
    monkeypatch.setenv('PACT_ENV', 'legacy')
    with pytest.raises(SystemExit, match='PACT_ENV'):
        run_main(root)


# ------------------------------------------------------------------ macro names
def test_macro_names_are_valid_and_unique(tmp_path):
    root = str(tmp_path)
    build(root, solo=True)
    res, out = run_main(root, '--require', 'all')
    text = open(mac_path(out)).read()
    names = re.findall(r'\\newcommand\{\\([^}]*)\}', text)
    assert len(names) == res['n_macros'] > 900
    assert all(re.fullmatch(r'Sec[A-Za-z]+', n) for n in names)
    assert len(set(names)) == len(names)
    assert text.startswith('% Generated by scripts/p95_secondary_inference.py -- do not hand-edit')
    vals = re.findall(r'\\newcommand\{\\[A-Za-z]+\}\{([^}]*)\}', text)
    assert not any(re.search(r'(?<![A-Za-z-])(nan|inf)(?![A-Za-z-])', v) for v in vals)
    with pytest.raises(SystemExit, match='duplicate'):
        P.check_names(['SecA', 'SecA'], root, [])
    with pytest.raises(SystemExit, match='not Sec plus letters'):
        P.check_names(['SecA1'], root, [])
    write(root, 'manuscript/macros_other.tex', '\\newcommand{\\SecEqVsPolN}{0}\n')
    with pytest.raises(SystemExit, match='collision'):
        run_main(root)
