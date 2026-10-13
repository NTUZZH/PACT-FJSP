"""
Tests for the HGNN port to PACT's constraint family (env/fjsp_env_family.py,
pact_bridge.py, train_family_hgnn.py, eval_family_hgnn.py). CPU only:

    CUDA_VISIBLE_DEVICES= python backbones/hgnn/tests/test_family_hgnn.py
    CUDA_VISIBLE_DEVICES= python -m pytest backbones/hgnn/tests/test_family_hgnn.py

F1 acceptance (the check of the ReSched port, on this backbone): random
   dispatch sequences over random instances of all eight regimes, five seeds
   per regime, both mask modes. The environment reproduces
   constraint_family.reference_sim.simulate's machine, start and end times
   exactly, operation by operation; every completed schedule passes
   reference_sim.validate_schedule; the certified estimate never exceeds the
   realized makespan along the rollout and equals it at termination.
F2 zero-rule equivalence: in regime N, under the exact mask and the published
   reward, every state field, reward and legal pair equals the upstream
   environment env/fjsp_env.py, step by step, on random sequences and under the
   upstream checkpoint (greedy); sequences drawn from PACT's mask replay
   identically in the upstream environment; in regime L the certified reward
   equals the lag-aware environment with the minimum-processing-time estimate.
F3 action sets: the policy's legal set equals PACT's mask at every decision;
   counts of decisions where PACT's set is strictly smaller than the exact
   non-delay set, and where the upstream job-idle-and-machine-idle rule would
   admit pairs that cannot start at the decision time.
F4 sampler: pact_bridge.sample_family_batch draws the same instances and
   descriptors as train_family.FamilyTrainer.sample_batch for the same seed.
F5 smoke: two training iterations on the CPU, then evaluation of the
   checkpoint on two instances of two regimes.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
HGNN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HGNN_DIR)
import ppvc_hgnn as P  # noqa: E402
import pact_bridge as B  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_device('cpu')

from env.fjsp_env import FJSPEnv  # noqa: E402
from env.fjsp_env_lag import FJSPEnvLag  # noqa: E402
from env.fjsp_env_family import FJSPEnvFamilyHGNN  # noqa: E402

REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
DEVICE = torch.device('cpu')
CFG = P.load_config()
CKPT = os.path.join(HGNN_DIR, 'model', 'save_10_5.pt')
PY = sys.executable
PACT = B.pact()


def env_paras():
    return P.make_paras(CFG, DEVICE, 'aware', False, 1, 1, 1)[0]


def generated(regime, seeds, n_modules=4):
    mch_type, _ = PACT.G.build_factory(PACT.G.DEFAULT_FACTORY)
    n_m = len(mch_type)
    jls, pts, descs = [], [], []
    for seed in seeds:
        jl, pt, meta = PACT.G.ppvc_instance_generator(n_modules=n_modules, class_mix='mixed',
                                                       station_counts=PACT.G.DEFAULT_FACTORY, seed=seed)
        jls.append(jl)
        pts.append(pt)
        descs.append(PACT.D.make_descriptor(regime, jl, pt, meta, n_m, seed=seed * 3 + 1))
    return jls, pts, descs


def from_disk(size, regime, i):
    with open(os.path.join(B.REPO_ROOT, 'data', 'FAMILY', size, 'desc', regime,
                           'instance_{0:03d}.desc.json'.format(i))) as f:
        desc = PACT.D.ConstraintDescriptor.from_json(json.load(f))
    stem = os.path.join(B.REPO_ROOT, 'data', 'PPVC', size + '+ppvc-mixed', 'instance_{0:03d}'.format(i))
    jl, pt, _ = PACT.G.load_instance(stem)
    return jl, pt, desc, stem + '.fjs'


def random_actions(env, rng):
    '''One uniformly random legal (operation, machine, job) per live instance.'''
    step = torch.minimum(env.ope_step_batch, env.end_ope_biases_batch)
    opes, mas, jobs = [], [], []
    for b in env.batch_idxes.tolist():
        pairs = torch.nonzero(env.pair_eligible[b]).tolist()
        j, m = pairs[int(rng.integers(len(pairs)))]
        opes.append(int(step[b, j]))
        mas.append(m)
        jobs.append(j)
    return torch.tensor([opes, mas, jobs])


def policy_eligible(env):
    '''The legal set the HGNN actor computes (PPO_model.get_action_prob), [B, J, M].'''
    step = torch.minimum(env.ope_step_batch, env.end_ope_biases_batch)
    adj = env.ope_ma_adj_batch.gather(1, step[..., None].expand(-1, -1, env.num_mas)) == 1
    job_ok = ~(env.mask_job_procing_batch | env.mask_job_finish_batch)
    ma_ok = ~env.mask_ma_procing_batch
    upstream = adj & job_ok[:, :, None] & ma_ok[:, None, :]
    pair = getattr(env.state, 'pair_eligible_batch', None)
    return upstream if pair is None else upstream & pair, upstream


# ---------------------------------------------------------------------------------------------------------------
# F1 acceptance
# ---------------------------------------------------------------------------------------------------------------

def _acceptance(mask, reward, seeds=(201, 202, 203, 204, 205)):
    rng = np.random.default_rng(7)
    lines = []
    total_ops = total_mis = 0
    for regime in REGIMES:
        jls, pts, descs = generated(regime, seeds)
        env = FJSPEnvFamilyHGNN(jls, pts, descs, env_paras(), reward=reward, mask=mask, checks=True)
        acts = [[] for _ in seeds]
        est = [[float(env.makespan_batch[b])] for b in range(len(seeds))]
        while not bool(env.done_batch.all()):
            a = random_actions(env, rng)
            for k, b in enumerate(env.batch_idxes.tolist()):
                acts[b].append((int(a[2, k]), int(a[1, k])))
            env.step(a)
            for b in range(len(seeds)):
                est[b].append(float(env.makespan_batch[b]))
        ops = mis = 0
        for b in range(len(seeds)):
            ref = PACT.R.simulate(jls[b], pts[b], descs[b], acts[b])
            st, ct, mch = env.schedule_arrays(b)
            ops += len(st)
            mis += int(np.sum((st != ref['op_st']) | (ct != ref['op_ct']) | (mch != ref['op_mch'])))
            viol = PACT.R.validate_schedule(jls[b], pts[b], descs[b], st, ct, mch)
            assert not viol, (regime, viol[:3])
            final = float(ref['makespan'])
            if reward == 'certified':
                assert max(est[b]) <= final + 1e-6, (regime, max(est[b]), final)
            assert abs(est[b][-1] - final) < 1e-6, (regime, est[b][-1], final)
        total_ops += ops
        total_mis += mis
        lines.append('F1 mask={0:5s} reward={1:9s} regime {2:>3}: {3} instances, {4} operations compared, '
                     'mismatches {5}, validator violations 0'.format(mask, reward, regime, len(seeds), ops, mis))
        assert mis == 0, (regime, mis)
    print('\n'.join(lines))
    return total_ops, total_mis


def test_f1_acceptance_pact_mask_certified():
    ops, mis = _acceptance('pact', 'certified')
    print('F1 pact/certified: {0} operations, mismatches {1}'.format(ops, mis))


def test_f1_acceptance_exact_mask_published():
    ops, mis = _acceptance('exact', 'published')
    print('F1 exact/published: {0} operations, mismatches {1}'.format(ops, mis))


# ---------------------------------------------------------------------------------------------------------------
# F2 zero-rule equivalence with the upstream environment
# ---------------------------------------------------------------------------------------------------------------

FIELDS = ('feat_opes_batch', 'feat_mas_batch', 'proc_times_batch', 'ope_ma_adj_batch', 'mask_job_procing_batch',
          'mask_job_finish_batch', 'mask_ma_procing_batch', 'time', 'makespan_batch', 'schedules_batch',
          'ope_step_batch', 'done_batch')


def _same(a, b, where):
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert torch.equal(x, y), '{0}: field {1} differs'.format(where, f)
    assert torch.equal(a.machines_batch[:, :, :2], b.machines_batch[:, :, :2]), where + ': machines'
    assert torch.equal(a.machines_batch[:, :, 2:], b.machines_batch[:, :, 2:]), where + ': machines'


def _replay(ref, env, actions, compare_eligible):
    _same(ref, env, 'step 0')
    for k, (o, m, j) in enumerate(actions):
        ref_ok = policy_eligible(ref)[0]
        assert bool(ref_ok[0, j, m]), 'action not eligible upstream'
        if compare_eligible:
            assert torch.equal(policy_eligible(env)[0], ref_ok), 'legal sets differ at step {0}'.format(k)
        a = torch.tensor([[o], [m], [j]])
        _, r1, _ = ref.step(a)
        _, r2, _ = env.step(a)
        assert torch.equal(r1, r2), 'reward differs at step {0}'.format(k)
        _same(ref, env, 'step {0}'.format(k + 1))


def _random_sequence(env, rng):
    seq = []
    while not bool(env.done_batch.all()):
        a = random_actions(env, rng)
        seq.append((int(a[0, 0]), int(a[1, 0]), int(a[2, 0])))
        env.step(a)
    return seq


def test_f2_zero_rule_equivalence():
    rng = np.random.default_rng(11)
    model = P.build_model(P.make_paras(CFG, DEVICE, 'aware', False, 1, 1, 1)[1], CFG['train_paras'], ckpt=CKPT)
    for i in range(5):
        jl, pt, desc, fjs = from_disk('10x25', 'N', i)
        # (a) exact mask + published reward against upstream, random sequence drawn from the upstream rule
        probe = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras(), reward='published', mask='exact')
        seq = _random_sequence(probe, rng)
        ref = P.make_file_env(fjs, env_paras(), env_cls=FJSPEnv)
        env = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras(), reward='published', mask='exact')
        _replay(ref, env, seq, compare_eligible=True)
        # (b) greedy under the upstream checkpoint
        ref = P.make_file_env(fjs, env_paras(), env_cls=FJSPEnv)
        env = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras(), reward='published', mask='exact')
        ms_ref = P.greedy_rollout(ref, model)
        ms = P.greedy_rollout(env, model)
        assert torch.equal(ms_ref, ms) and torch.equal(ref.schedules_batch, env.schedules_batch)
        # (c) PACT mask: a sequence from PACT's legal set replays identically upstream
        probe = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras(), reward='published', mask='pact')
        seq_p = _random_sequence(probe, rng)
        ref = P.make_file_env(fjs, env_paras(), env_cls=FJSPEnv)
        env = FJSPEnvFamilyHGNN([jl], [pt], [desc], env_paras(), reward='published', mask='pact')
        _replay(ref, env, seq_p, compare_eligible=False)
        # (d) regime L, certified reward against the lag-aware environment with the min-pt estimate
        jlL, ptL, descL, fjsL = from_disk('10x25', 'L', i)
        probe = FJSPEnvFamilyHGNN([jlL], [ptL], [descL], env_paras(), reward='certified', mask='exact')
        seq_l = _random_sequence(probe, rng)
        ref = P.make_file_env(fjsL, dict(env_paras(), bound_pt='min'), time_lags=[descL.lag])
        env = FJSPEnvFamilyHGNN([jlL], [ptL], [descL], env_paras(), reward='certified', mask='exact')
        _replay(ref, env, seq_l, compare_eligible=True)
        print('F2 instance_{0:03d}: N exact/published random {1} steps identical to upstream; greedy upstream '
              'checkpoint {2:.0f} = {3:.0f}; N PACT-mask sequence replays identically upstream; L certified '
              'identical to the lag-aware min-pt environment ({4} steps)'.format(
                  i, len(seq), float(ms_ref[0]), float(ms[0]), len(seq_l)))


# ---------------------------------------------------------------------------------------------------------------
# F3 action sets
# ---------------------------------------------------------------------------------------------------------------

def test_f3_action_sets():
    rng = np.random.default_rng(3)
    tot = {'decisions': 0, 'pact_smaller': 0, 'first_pact_larger': 0, 'upstream_rule_larger': 0}
    for regime in REGIMES:
        jls, pts, descs = generated(regime, range(301, 306), n_modules=10)
        env = FJSPEnvFamilyHGNN(jls, pts, descs, env_paras(), reward='certified', mask='pact')
        c = dict.fromkeys(tot, 0)
        first = True
        while not bool(env.done_batch.all()):
            pol, upstream = policy_eligible(env)
            pact = torch.as_tensor(~env.shadow.state.dynamic_pair_mask_tensor.numpy())
            assert torch.equal(pol, pact), 'policy legal set differs from PACT mask'
            exact = env.exact_eligible
            for b in range(env.batch_size):
                c['decisions'] += 1
                if first:
                    c['first_pact_larger'] += int(bool((pact[b] & ~exact[b]).any()))
                else:
                    assert not bool((pact[b] & ~exact[b]).any())
                    c['pact_smaller'] += int(bool((exact[b] & ~pact[b]).any()))
                    c['upstream_rule_larger'] += int(bool((upstream[b] & ~exact[b]).any()))
            env.step(random_actions(env, rng))
            first = False
        print('F3 regime {0:>3}: {1} decisions; policy set == PACT mask at every decision; PACT set strictly '
              'smaller than the exact non-delay set at {2} (ties lost to rounding); at the first decision PACT '
              'admits pairs that start later at {3}; the upstream idle-job-and-idle-machine rule would admit '
              'pairs that cannot start at t* at {4}'.format(regime, c['decisions'], c['pact_smaller'],
                                                           c['first_pact_larger'], c['upstream_rule_larger']))
        for k in tot:
            tot[k] += c[k]
    print('F3 total: {0}'.format(tot))


# ---------------------------------------------------------------------------------------------------------------
# F4 sampler parity with train_family.py
# ---------------------------------------------------------------------------------------------------------------

def _digest(batches):
    h = hashlib.sha256()
    for jls, pts, descs, opts, mchts in batches:
        for jl, pt, d, ot, mt in zip(jls, pts, descs, opts, mchts):
            h.update(np.asarray(jl).tobytes())
            h.update(np.asarray(pt).tobytes())
            h.update(json.dumps(d.to_json(), sort_keys=True).encode())
            h.update(np.asarray(ot).tobytes())
            h.update(np.asarray(mt).tobytes())
    return h.hexdigest()


REF_SNIPPET = r'''
import sys, types, random, json, hashlib
sys.argv = [sys.argv[0], '--device', 'cpu']
import numpy as np
import train_family as T
from ppvc_instance_generator import DEFAULT_FACTORY
regimes = REGIMES
self = types.SimpleNamespace(num_envs=20, regimes=regimes, n_j=10, n_m=25, desc_shift=None,
                             ppvc_factory=DEFAULT_FACTORY, config=types.SimpleNamespace(ppvc_mix='mixed'))
from common_utils import setup_seed
setup_seed(SEED)
batches = [T.FamilyTrainer.sample_batch(self) for _ in range(2)]
h = hashlib.sha256()
for jls, pts, descs, opts, mchts in batches:
    for jl, pt, d, ot, mt in zip(jls, pts, descs, opts, mchts):
        h.update(np.asarray(jl).tobytes()); h.update(np.asarray(pt).tobytes())
        h.update(json.dumps(d.to_json(), sort_keys=True).encode())
        h.update(np.asarray(ot).tobytes()); h.update(np.asarray(mt).tobytes())
print('DIGEST', h.hexdigest())
'''


def test_f4_sampler_matches_train_family():
    import random
    for regimes, seed in ((['N', 'L', 'S', 'W', 'LS', 'LW'], 301), (['L'], 302)):
        code = REF_SNIPPET.replace('REGIMES', repr(regimes)).replace('SEED', str(seed))
        r = subprocess.run([PY, '-c', code], cwd=B.REPO_ROOT, capture_output=True, text=True,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        assert r.returncode == 0, r.stderr[-2000:]
        ref = re.search(r'DIGEST (\w+)', r.stdout).group(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        ours = _digest([B.sample_family_batch(20, regimes, 10, 25, 'mixed', 'default') for _ in range(2)])
        print('F4 regimes {0} seed {1}: train_family {2}... ours {3}...'.format(regimes, seed, ref[:16], ours[:16]))
        assert ours == ref


# ---------------------------------------------------------------------------------------------------------------
# F5 smoke
# ---------------------------------------------------------------------------------------------------------------

def test_f5_smoke_training():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [PY, os.path.join(HGNN_DIR, 'train_family_hgnn.py'), '--family_regimes', 'N,L,S,W,LS,LW',
               '--hgnn_reward', 'certified', '--num_envs', '6', '--max_iterations', '2', '--save_timestep', '2',
               '--vali_size', '2', '--seed', '301', '--device', 'cpu', '--threads', '4', '--out_root', tmp]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(r.stdout[-2500:])
        assert r.returncode == 0, r.stderr[-3000:]
        n = int(re.search(r'\[model\] params total=(\d+)', r.stdout).group(1))
        name = '10x25+family+hgnn-certified-joint-s301'
        assert os.path.exists(os.path.join(tmp, 'trained_network', 'FAMILY', name + '.pth'))
        with open(os.path.join(tmp, 'train_log', 'FAMILY', 'config_{0}.json'.format(name))) as f:
            assert json.load(f)['n_params'] == n
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode != 0 and 'exists' in (r.stdout + r.stderr), 'overwrite was not refused'
        out = os.path.join(tmp, 'test_results')
        cmd = [PY, os.path.join(HGNN_DIR, 'eval_family_hgnn.py'), '--family_model', name, '--sizes', '10x25',
               '--regimes', 'N,LSW', '--limit', '2', '--model_root', tmp, '--out_root', out, '--device', 'cpu']
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        print(r.stdout[-1500:])
        assert r.returncode == 0, r.stderr[-3000:]
        for regime in ('N', 'LSW'):
            res = np.load(os.path.join(out, '10x25', 'Result_hgnn-certified-joint-s301_{0}.npy'.format(regime)))
            assert res.shape == (2, 2) and np.isfinite(res).all()
        r = subprocess.run(cmd, capture_output=True, text=True, env=env)
        assert r.returncode != 0 and 'exists' in (r.stdout + r.stderr), 'result overwrite was not refused'
        print('F5 params {0}; two iterations, checkpoint and config written; evaluation files of shape (2, 2); '
              'overwrites refused'.format(n))


if __name__ == '__main__':
    failed = []
    only = sys.argv[1:]
    for name in sorted(k for k in dict(globals()) if k.startswith('test_f')):
        if only and not any(o in name for o in only):
            continue
        print('=' * 20, name, flush=True)
        try:
            globals()[name]()
            print('PASS', name, flush=True)
        except AssertionError as e:
            failed.append(name)
            print('FAIL', name, repr(e)[:800], flush=True)
    print('failed:', failed if failed else 'none')
    sys.exit(1 if failed else 0)
