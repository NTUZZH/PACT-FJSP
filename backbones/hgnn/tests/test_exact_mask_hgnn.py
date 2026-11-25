"""
Action sets of the HGNN family environment under the exact mask against the
policy environment under the exact options.

Reference: constraint_family.fjsp_env_family.FJSPEnvFamily with
first_mask='exact' and mask_ties='exact'. Its legal (job, machine) pairs are
~dynamic_pair_mask; the policy reads the same array as
state.dynamic_pair_mask_tensor, and both are compared.

Compared: FJSPEnvFamilyHGNN(mask='exact'). Its legal set is the one the HGNN
actor uses (PPO_model.get_action_prob): the upstream job-idle, machine-idle and
compatibility mask intersected with state.pair_eligible_batch.

Both environments are driven by the same action sequence, step by step:
  random   one uniformly random pair per decision from the pairs legal in both
           environments, which is the reference set when the two agree (numpy
           default_rng with a fixed seed), batch of all instances;
  SPT      the shortest-processing-time rule of common_utils, computed in the
           reference environment, one instance at a time. SPT chooses among the
           pairs of the reference set (it reads dynamic_pair_mask). FIFO, MOR and
           MWKR choose a ready job first and fall back to its earliest free
           compatible machine when none of its pairs is legal, so their pair can
           lie outside the set and they cannot drive an environment that
           enforces it.
At every decision the two legal sets, the policy-side tensor and the decision
instant (in hours) are compared. After the episode the HGNN schedule is compared
operation by operation with constraint_family.reference_sim.simulate and with
the reference environment's completion times, and it is checked by
reference_sim.validate_schedule.

Data: data/FAMILY/10x25 (instances 0-19) and data/FAMILY/50x25 (instances 0-2)
for all eight combinations, read only.

    CUDA_VISIBLE_DEVICES= python -m pytest -q backbones/hgnn/tests/test_exact_mask_hgnn.py -s
    CUDA_VISIBLE_DEVICES= python backbones/hgnn/tests/test_exact_mask_hgnn.py      (full report)
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HGNN_DIR = os.path.dirname(HERE)
sys.path.insert(0, HGNN_DIR)
import ppvc_hgnn as P  # noqa: E402
import pact_bridge as B  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_device('cpu')

from env.fjsp_env_family import FJSPEnvFamilyHGNN  # noqa: E402

REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
DEVICE = torch.device('cpu')
CFG = P.load_config()
PACT = B.pact()
SCOPE = (('10x25', 20), ('50x25', 3))
RANDOM_SEED = 20261111
RULE = 'SPT'
RULE_SEED = 50


def env_paras():
    return P.make_paras(CFG, DEVICE, 'aware', False, 1, 1, 1)[0]


def load_instances(size, regime, n):
    '''(job lengths, processing times, descriptors) of the first n instances of size/regime, read as
    eval_family_hgnn.py reads them.'''
    root = os.path.join(B.REPO_ROOT, 'data', 'FAMILY', size)
    with open(os.path.join(root, 'dataset_meta.json')) as f:
        base_dir = os.path.join(B.REPO_ROOT, json.load(f)['base_dir'])
    jls, pts, descs = [], [], []
    for i in range(n):
        with open(os.path.join(root, 'desc', regime, 'instance_{0:03d}.desc.json'.format(i))) as f:
            descs.append(PACT.D.ConstraintDescriptor.from_json(json.load(f), setup_convention='attached'))
        jl, pt, _ = PACT.G.load_instance(os.path.join(base_dir, 'instance_{0:03d}'.format(i)))
        jls.append(jl)
        pts.append(pt)
    return jls, pts, descs


def reference_env(jls, pts, descs, option='exact'):
    '''The policy environment with both mask options set to `option` ('exact', or 'legacy' for the control).'''
    env = PACT.F.FJSPEnvFamily(len(jls[0]), pts[0].shape[1], family_mode=True,
                               first_mask=option, mask_ties=option)
    env.set_initial_data(list(jls), list(pts), descriptor_list=list(descs))
    assert env.mask_setting('first_mask') == option and env.mask_setting('mask_ties') == option
    return env


def hgnn_legal_set(env):
    '''[B, J, M] bool: the pairs the HGNN actor can choose (PPO_model.get_action_prob).'''
    step = torch.minimum(env.ope_step_batch, env.end_ope_biases_batch)
    adj = env.ope_ma_adj_batch.gather(1, step[..., None].expand(-1, -1, env.num_mas)) == 1
    job_ok = ~(env.mask_job_procing_batch | env.mask_job_finish_batch)
    ma_ok = ~env.mask_ma_procing_batch
    return (adj & job_ok[:, :, None] & ma_ok[:, None, :] & env.state.pair_eligible_batch).cpu().numpy()


def _new_counts():
    return dict(instances=0, decisions=0, set_mismatch=0, ref_only_pairs=0, hgnn_only_pairs=0,
                tensor_mismatch=0, time_mismatch=0, empty_intersection=0, operations=0,
                schedule_mismatch_ref_sim=0, completion_mismatch_ref_env=0, validator_violations=0,
                legal_pairs=0, first_decisions_with_choice=0)


def rollout(jls, pts, descs, driver, rng=None, counts=None, examples=None, option='exact'):
    '''Drive the reference environment and the HGNN environment with one action sequence and compare them at
    every decision. driver: 'random' (pairs drawn by rng from the pairs legal in both environments) or a rule
    name (batch of one). Returns the updated counts.'''
    c = counts if counts is not None else _new_counts()
    ref = reference_env(jls, pts, descs, option)
    hg = FJSPEnvFamilyHGNN(jls, pts, descs, env_paras(), reward='certified', mask='exact', checks=True)
    Bn, M = len(jls), pts[0].shape[1]
    if driver != 'random':
        assert Bn == 1, 'a dispatching rule reads instance 0 of the environment only'
        np.random.seed(RULE_SEED)
    acts = [[] for _ in range(Bn)]
    step = 0
    while not bool(ref.done().all()):
        ref_set = ~ref.dynamic_pair_mask
        tensor_set = ~ref.state.dynamic_pair_mask_tensor.cpu().numpy()
        hg_set = hgnn_legal_set(hg)
        t_ref = np.rint(ref.next_schedule_time / ref._slope)
        t_hg = hg.time.double().cpu().numpy()
        chosen = []
        for b in range(Bn):
            c['decisions'] += 1
            c['legal_pairs'] += int(ref_set[b].sum())
            if step == 0 and int(ref_set[b].sum()) > 1:
                c['first_decisions_with_choice'] += 1
            if not np.array_equal(tensor_set[b], ref_set[b]):
                c['tensor_mismatch'] += 1
            if t_ref[b] != t_hg[b]:
                c['time_mismatch'] += 1
            if not np.array_equal(ref_set[b], hg_set[b]):
                c['set_mismatch'] += 1
                c['ref_only_pairs'] += int((ref_set[b] & ~hg_set[b]).sum())
                c['hgnn_only_pairs'] += int((hg_set[b] & ~ref_set[b]).sum())
                if examples is not None and len(examples) < 20:
                    examples.append(dict(step=step, instance=b, t_ref=float(t_ref[b]), t_hgnn=float(t_hg[b]),
                                         ref_only=np.argwhere(ref_set[b] & ~hg_set[b]).tolist()[:5],
                                         hgnn_only=np.argwhere(hg_set[b] & ~ref_set[b]).tolist()[:5]))
            both = ref_set[b] & hg_set[b]
            if driver == 'random':
                pool = np.argwhere(both if both.any() else ref_set[b])
                j, m = (int(x) for x in pool[int(rng.integers(len(pool)))])
            else:
                a = int(PACT.U.heuristic_select_action(driver, ref))
                j, m = divmod(a, M)
                assert ref_set[b][j, m], 'the rule chose a pair outside the reference set'
            if not both[j, m]:
                c['empty_intersection'] += 1
                raise AssertionError('step {0}, instance {1}: the chosen pair ({2}, {3}) is not legal in both '
                                     'environments'.format(step, b, j, m))
            chosen.append((j, m))
        opes = [int(torch.minimum(hg.ope_step_batch, hg.end_ope_biases_batch)[b, j]) for b, (j, m) in
                enumerate(chosen)]
        hg.step(torch.tensor([opes, [m for _, m in chosen], [j for j, _ in chosen]]))
        ref.step(np.array([j * M + m for j, m in chosen]))
        for b, a in enumerate(chosen):
            acts[b].append(a)
        step += 1
    assert bool(hg.done_batch.all()), 'the HGNN episode did not finish with the reference episode'
    viol = hg.validate_family()
    for b in range(Bn):
        c['instances'] += 1
        st, ct, mch = hg.schedule_arrays(b)
        c['operations'] += len(st)
        sim = PACT.R.simulate(jls[b], pts[b], descs[b], acts[b])
        c['schedule_mismatch_ref_sim'] += int(np.sum((st != sim['op_st']) | (ct != sim['op_ct']) |
                                                    (mch != sim['op_mch'])))
        c['completion_mismatch_ref_env'] += int(np.sum(ct != np.asarray(ref.true_op_ct[b], dtype=float)))
        c['validator_violations'] += len(viol[b])
    return c


def compare_scope(scope=SCOPE, regimes=REGIMES, drivers=('random', RULE), log=print):
    '''Counts per (size, combination, driver) and totals; examples of mismatching decisions.'''
    rows, examples = {}, []
    total = _new_counts()
    for size, n in scope:
        for regime in regimes:
            jls, pts, descs = load_instances(size, regime, n)
            for driver in drivers:
                c = _new_counts()
                if driver == 'random':
                    rng = np.random.default_rng(RANDOM_SEED + REGIMES.index(regime))
                    rollout(jls, pts, descs, 'random', rng=rng, counts=c, examples=examples)
                else:
                    for i in range(n):
                        rollout([jls[i]], [pts[i]], [descs[i]], driver, counts=c, examples=examples)
                rows[(size, regime, driver)] = c
                for k in total:
                    total[k] += c[k]
                log('{0} {1:>3} {2:6s}: {3} instances, {4} decisions ({5} legal pairs), set mismatches {6}, '
                    'policy-tensor mismatches {7}, decision-time mismatches {8}; {9} operations, schedule '
                    'mismatches against reference_sim {10}, completion mismatches against the reference '
                    'environment {11}, validator violations {12}; first decisions with more than one legal pair '
                    '{13}'.format(size, regime, driver, c['instances'], c['decisions'], c['legal_pairs'],
                                  c['set_mismatch'], c['tensor_mismatch'], c['time_mismatch'], c['operations'],
                                  c['schedule_mismatch_ref_sim'], c['completion_mismatch_ref_env'],
                                  c['validator_violations'], c['first_decisions_with_choice']))
    log('EXACT total: {0}'.format(total))
    return rows, total, examples


def test_comparison_detects_the_legacy_mask():
    '''Control: against the policy environment with the legacy options the same comparison must find the two
    known differences (every compatible pair legal at the first decision; tied pairs dropped by rounding).'''
    c = _new_counts()
    for regime in ('S', 'LSW'):
        jls, pts, descs = load_instances('10x25', regime, 5)
        rollout(jls, pts, descs, 'random', rng=np.random.default_rng(RANDOM_SEED), counts=c, option='legacy')
    print('control against the legacy options: {0}'.format(c))
    assert c['set_mismatch'] > 0 and c['ref_only_pairs'] > 0 and c['hgnn_only_pairs'] > 0


def test_exact_mask_action_sets_equal():
    rows, total, examples = compare_scope()
    assert total['set_mismatch'] == 0, examples[:5]
    assert total['tensor_mismatch'] == 0
    assert total['time_mismatch'] == 0
    assert total['schedule_mismatch_ref_sim'] == 0
    assert total['completion_mismatch_ref_env'] == 0
    assert total['validator_violations'] == 0
    assert total['instances'] == 2 * len(REGIMES) * sum(n for _, n in SCOPE)


if __name__ == '__main__':
    _, tot, ex = compare_scope()
    if ex:
        print('examples of mismatching decisions:')
        for e in ex:
            print(' ', e)
    ok = all(tot[k] == 0 for k in ('set_mismatch', 'tensor_mismatch', 'time_mismatch', 'schedule_mismatch_ref_sim',
                                   'completion_mismatch_ref_env', 'validator_violations'))
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
