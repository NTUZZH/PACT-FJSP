"""Tests for the masked dispatching rules (common_utils.masked_rule_action) and
eval_family.py --rule_set.

  * masked SPT gives the same actions and makespans as classical SPT, under
    the legacy and the exact mask, in all eight regimes (three instances each);
  * every action of masked FIFO, MOR, SPT and MWKR lies inside
    ~env.dynamic_pair_mask at every decision of the rollout, in all eight
    regimes under both masks, and every schedule passes
    reference_sim.validate_schedule;
  * the classical rules keep their published arrays: eval_family.py with the
    default and with --rule_set classical reproduces the stored 10x25 rule
    files on a sample, and heuristic_select_action is untouched;
  * eval_family.py --rule_set masked writes the arrays of the in-test masked
    rollout, records the rule set in the summary and prints it in the log
    header, and refuses to run without --tag or with an unknown value.

Run from the repository root:  python3 -m pytest tests/test_masked_rules.py -q
CPU only, one thread.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs  # noqa: E402
finally:
    sys.argv = _argv
configs.device = 'cpu'

import torch  # noqa: E402

import fjsp_env_same_op_nums  # noqa: E402
fjsp_env_same_op_nums.EnvState.device = torch.device('cpu')

from common_utils import heuristic_select_action, masked_rule_action  # noqa: E402
from constraint_family import reference_sim  # noqa: E402
from constraint_family.descriptor import ConstraintDescriptor  # noqa: E402
from constraint_family.fjsp_env_family import FJSPEnvFamily  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402

DATA = 'data/FAMILY/10x25'
STORED = 'test_results/FAMILY/10x25'
REGIMES = ('N', 'L', 'S', 'W', 'LS', 'LW', 'SW', 'LSW')
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR')
INSTANCES = (0, 1, 2)
MASKS = {'legacy': {}, 'exact': dict(first_mask='exact', mask_ties='exact')}
MASK_VARS = ('PACT_FIRST_MASK', 'PACT_MASK_TOL')


@pytest.fixture(autouse=True)
def _no_mask_vars(monkeypatch):
    for v in MASK_VARS:
        monkeypatch.delenv(v, raising=False)


def load(regime, idx):
    with open(os.path.join(DATA, 'desc', regime, f'instance_{idx:03d}.desc.json')) as f:
        rec = json.load(f)
    jl, pt, meta = load_instance(rec['base_instance'])
    return jl, pt, meta, ConstraintDescriptor.from_json(rec, setup_convention='attached')


def rollout(select, method, regime, idx, mask, seed):
    """One rule rollout as eval_family.py runs it: the env of the PDR methods
    (non-delay mask, no pair channels), np.random seeded once per instance.
    Returns makespan, actions, whether each action was inside the mask, and
    the validator's violations."""
    jl, pt, meta, desc = load(regime, idx)
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True,
                        mask_mode='nondelay', pair_extra='none', **MASKS[mask])
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']],
                         mch_type_list=[meta['mch_type']])
    np.random.seed(seed)
    M = pt.shape[1]
    amch = np.full(pt.shape[0], -1, dtype=int)
    acts, inside = [], []
    while not env.done().all():
        a = int(select(method, env))
        inside.append(bool(~env.dynamic_pair_mask[0].reshape(-1)[a]))
        acts.append(a)
        amch[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    op_ct = env.true_op_ct[0]
    op_st = op_ct - pt[np.arange(pt.shape[0]), amch]
    viol = reference_sim.validate_schedule(jl, pt, desc, op_st, op_ct, amch)
    return float(env.current_makespan[0]), acts, inside, viol


@pytest.mark.parametrize('mask', sorted(MASKS))
def test_masked_spt_equals_classical_spt(mask):
    for regime in REGIMES:
        for idx in INSTANCES:
            ms_c, acts_c, _, _ = rollout(heuristic_select_action, 'SPT', regime, idx, mask, 200)
            ms_m, acts_m, _, _ = rollout(masked_rule_action, 'SPT', regime, idx, mask, 200)
            assert acts_m == acts_c, (mask, regime, idx)
            assert ms_m == ms_c, (mask, regime, idx)


@pytest.mark.parametrize('mask', sorted(MASKS))
@pytest.mark.parametrize('method', RULES)
def test_masked_rules_act_inside_the_mask_and_validate(mask, method):
    for regime in REGIMES:
        for idx in INSTANCES:
            ms, acts, inside, viol = rollout(masked_rule_action, method, regime, idx, mask, 200)
            assert all(inside), (f'{method} {mask} {regime} instance {idx}: '
                                 f'{len(inside) - sum(inside)} of {len(inside)} '
                                 'decisions outside the mask')
            assert not viol, (method, mask, regime, idx, viol[:3])
            assert np.isfinite(ms) and ms > 0


def test_classical_two_stage_rules_leave_the_mask_under_the_exact_mask():
    """The reason for the masked set: classical MWKR commits jobs outside the
    exact mask on a setup regime (a control on the measurement above)."""
    out = sum(len(inside) - sum(inside) for idx in INSTANCES for inside in
              [rollout(heuristic_select_action, 'MWKR', 'S', idx, 'exact', 200)[2]])
    assert out > 0


def test_masked_rule_action_refuses_unknown_method():
    jl, pt, meta, desc = load('N', 0)
    env = FJSPEnvFamily(jl.shape[0], pt.shape[1], family_mode=True)
    env.set_initial_data([jl], [pt], descriptor_list=[desc],
                         op_type_list=[meta['op_type']], mch_type_list=[meta['mch_type']])
    with pytest.raises(ValueError):
        masked_rule_action('LPT', env)


def _eval(args, out_root):
    env = {k: v for k, v in os.environ.items() if k not in MASK_VARS}
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable, 'eval_family.py', '--family_model', 'rules',
                           '--data', DATA, '--base', 'data/PPVC/10x25+ppvc-mixed',
                           '--device', 'cpu', '--device_id', '', '--out_root', str(out_root)]
                          + args, cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.mark.parametrize('extra', ([], ['--rule_set', 'classical']))
def test_classical_default_reproduces_the_stored_rule_arrays(extra, tmp_path):
    n, regs = 4, ('S', 'LSW')
    r = _eval(['--regimes', *regs, '--methods', *RULES, '--limit', str(n)] + extra, tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'RULES: rule_set=classical' in r.stdout
    for rule in RULES:
        for reg in regs:
            new = np.load(tmp_path / '10x25' / f'Result_{rule}_{reg}.npy')[:, 0]
            old = np.load(f'{STORED}/Result_{rule}_{reg}.npy')
            old = (old[:, 0] if old.ndim > 1 else old)[:n]
            assert np.array_equal(new, old), (rule, reg, new, old)
    summary = (tmp_path / '10x25' / 'summary_rules.md').read_text()
    assert 'rule_set' not in summary and 'mask:' not in summary


def test_eval_family_masked_matches_the_in_test_rollout(tmp_path):
    n, regs = 2, ('N', 'S')
    r = _eval(['--regimes', *regs, '--methods', *RULES, '--limit', str(n),
               '--rule_set', 'masked', '--first_mask', 'exact', '--mask_ties', 'exact',
               '--tag', '_t'], tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'RULES: rule_set=masked' in r.stdout
    out = tmp_path / '10x25_t'
    summary = (out / 'summary_rules_t.md').read_text()
    assert 'rules: rule_set=masked' in summary and 'first_mask=exact' in summary
    for rule in RULES:
        for reg in regs:
            got = np.load(out / f'Result_{rule}_{reg}.npy')[:, 0]
            want = [rollout(masked_rule_action, rule, reg, i, 'exact', configs.seed_test)[0]
                    for i in range(n)]
            assert np.array_equal(got, np.array(want)), (rule, reg, got, want)


def test_eval_family_refuses_masked_without_tag_or_unknown_value(tmp_path):
    r = _eval(['--regimes', 'S', '--methods', 'SPT', '--limit', '1',
               '--rule_set', 'masked'], tmp_path)
    assert r.returncode != 0 and 'needs --tag' in (r.stdout + r.stderr), r.stdout + r.stderr
    r = _eval(['--regimes', 'S', '--methods', 'SPT', '--limit', '1',
               '--rule_set', 'repaired', '--tag', '_t'], tmp_path)
    assert r.returncode != 0 and 'unknown --rule_set' in (r.stdout + r.stderr)
    assert not any(tmp_path.iterdir()), 'a refused run wrote a result folder'
