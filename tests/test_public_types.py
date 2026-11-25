"""Class input of the public family through the type embedding (--pub_type_source).

Checks:
  * type_lists: 'none' gives no type input; 'class' gives the job class of each
    operation and one constant station type;
  * the class reaches the network input: two descriptors of one base instance
    that differ only by a relabelling of the classes (setup matrix relabelled
    with them, so the instance is physically the same) give identical
    observations and policy outputs without the class input, and different
    type inputs and policy outputs with it;
  * parameter counts: 29,602 without and 32,714 with the class input;
  * the trainer's default path is unchanged: the same draws, no type lists, the
    class-blind network; with 'class' the same draws and the class lists;
  * evaluation refuses a checkpoint whose type source differs from the option,
    and evaluates a class-input checkpoint with the matching option.

Tiny instances only (one SD2 10x5 record, two validation instances).
Run: taskset -c 14 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
       python -m pytest tests/test_public_types.py -q -p no:cacheprovider
"""
import copy
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# params parses sys.argv at import time; hide pytest's arguments and force CPU
_argv, sys.argv = sys.argv, sys.argv[:1] + ['--device', 'cpu']
try:
    from params import configs
    import torch
    from constraint_family.descriptor import ConstraintDescriptor, IDLE_CLASS, N_CLASSES
    from constraint_family.fjsp_env_family import FJSPEnvFamily
    from constraint_family.public_rules import load_family_record
    from constraint_family.public_types import (CLASS_N_MCH_TYPES, CLASS_N_OP_TYPES,
                                                PUBC_PARAMS, PUBX_PARAMS,
                                                refuse_mismatch, snapshot_source,
                                                type_lists)
    from model.main_model import DANIEL
finally:
    sys.argv = _argv
from conftest import needs_public_instances  # noqa: E402

REC = os.path.join(ROOT, 'data/PUBLIC/family/pub_SD2_test_10x5+mix/desc/{R}/10x5+mix_001.desc.json')
SNAP = os.path.join(ROOT, 'train_log/FAMILY/config_10x5+pubfamily+pubx-v1-s301.json')
CKPT = os.path.join(ROOT, 'trained_network/FAMILY/10x5+pubfamily+pubx-v1-s301.pth')
ARCH_KEYS = ('use_lag_features', 'use_type_embedding', 'fea_j_input_dim',
             'fea_m_input_dim', 'type_emb_dim', 'n_op_types', 'n_mch_types',
             'n_j', 'n_m', 'n_op', 'num_heads_OAB', 'num_heads_MAB',
             'layer_fea_output_dim', 'num_mlp_layers_actor',
             'hidden_dim_actor', 'num_mlp_layers_critic', 'hidden_dim_critic',
             'dropout_prob', 'use_film', 'token_dim', 'film_hidden',
             'token_to_heads', 'reward_delta_env', 'delta_mode',
             'mask_mode', 'pair_extra')

needs_data = pytest.mark.skipif(not (os.path.exists(REC.format(R='N')) and os.path.exists(SNAP)),
                                reason='public records or the pubx snapshot are not on disk')


def recipe(source):
    """Network configuration of the public recipe (the pubx snapshot), with the
    type tables of `source`."""
    with open(SNAP) as f:
        snap = json.load(f)
    c = copy.deepcopy(configs)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(c, k, snap[k])
    if source == 'class':
        c.use_type_embedding = True
        c.n_op_types, c.n_mch_types = CLASS_N_OP_TYPES, CLASS_N_MCH_TYPES
    c.device = 'cpu'
    return c


def relabel(desc, perm):
    """The same physical descriptor with class a renamed perm[a]."""
    perm = np.asarray(perm)
    setup = None
    if desc.setup is not None:
        S = np.asarray(desc.setup)
        setup = np.zeros_like(S)
        for a in range(N_CLASSES):
            for b in range(N_CLASSES):
                setup[perm[a], perm[b]] = S[a, b]
        for b in range(N_CLASSES):
            setup[IDLE_CLASS, perm[b]] = S[IDLE_CLASS, b]
    return ConstraintDescriptor(active=desc.active, lag=desc.lag,
                                op_class=perm[np.asarray(desc.op_class)], setup=setup,
                                windows=desc.windows, horizon=desc.horizon,
                                setup_convention='attached')


def state_of(jl, pt, desc, source):
    env = FJSPEnvFamily(len(jl), pt.shape[1], family_mode=True,
                        first_mask='exact', mask_ties='exact')
    ot, mt = type_lists(source, [desc], pt.shape[1])
    env.set_initial_data([jl], [pt], descriptor_list=[desc], op_type_list=ot,
                         mch_type_list=mt)
    return env.state


def forward(model, st):
    with torch.no_grad():
        pi, _ = model(fea_j=st.fea_j_tensor, op_mask=st.op_mask_tensor,
                      candidate=st.candidate_tensor, fea_m=st.fea_m_tensor,
                      mch_mask=st.mch_mask_tensor, comp_idx=st.comp_idx_tensor,
                      dynamic_pair_mask=st.dynamic_pair_mask_tensor,
                      fea_pairs=st.fea_pairs_tensor, op_type=st.op_type_tensor,
                      mch_type=st.mch_type_tensor, token=None)
    return pi


@needs_public_instances()
@needs_data
def test_type_lists():
    jl, pt, desc, _ = load_family_record(REC.format(R='S'))
    assert type_lists('none', [desc], pt.shape[1]) == (None, None)
    ot, mt = type_lists('class', [desc, desc], pt.shape[1])
    assert len(ot) == len(mt) == 2
    np.testing.assert_array_equal(ot[0], desc.op_class)
    np.testing.assert_array_equal(mt[1], np.zeros(pt.shape[1], dtype=int))
    with pytest.raises(ValueError):
        type_lists('job', [desc], pt.shape[1])
    bad = copy.deepcopy(desc)
    bad.op_class = np.full_like(bad.op_class, N_CLASSES)
    with pytest.raises(ValueError):
        type_lists('class', [bad], pt.shape[1])


@needs_public_instances()
@needs_data
@pytest.mark.parametrize('regime', ['N', 'W', 'S', 'LSW'])
def test_every_combination_carries_the_class(regime):
    """Combinations without setups carry the same classes as those with setups."""
    _, _, d, _ = load_family_record(REC.format(R=regime))
    _, _, s, _ = load_family_record(REC.format(R='S'))
    np.testing.assert_array_equal(d.op_class, s.op_class)
    assert set(np.unique(d.op_class)) <= set(range(N_CLASSES))


@needs_public_instances()
@needs_data
@pytest.mark.parametrize('regime', ['N', 'S'])
def test_class_reaches_the_network_input(regime):
    jl, pt, desc, _ = load_family_record(REC.format(R=regime))
    other = relabel(desc, [2, 3, 0, 1])
    assert not np.array_equal(other.op_class, desc.op_class)
    torch.manual_seed(0)
    blind = DANIEL(recipe('none')).eval()
    torch.manual_seed(0)
    seeing = DANIEL(recipe('class')).eval()
    outs = {}
    for source, model in (('none', blind), ('class', seeing)):
        a, b = state_of(jl, pt, desc, source), state_of(jl, pt, other, source)
        # the relabelled instance is physically the same: same observation
        for name in ('fea_j_tensor', 'fea_m_tensor', 'fea_pairs_tensor',
                     'dynamic_pair_mask_tensor', 'candidate_tensor'):
            assert torch.equal(getattr(a, name), getattr(b, name)), (source, name)
        if source == 'none':
            assert a.op_type_tensor is None and b.op_type_tensor is None
        else:
            assert not torch.equal(a.op_type_tensor, b.op_type_tensor)
            assert torch.equal(a.mch_type_tensor, torch.zeros(pt.shape[1], dtype=torch.long)[None])
        outs[source] = (forward(model, a), forward(model, b))
    assert torch.equal(*outs['none'])
    assert not torch.allclose(*outs['class'])


@needs_data
def test_parameter_counts():
    assert sum(p.numel() for p in DANIEL(recipe('none')).parameters()) == PUBX_PARAMS == 29602
    m = DANIEL(recipe('class'))
    assert sum(p.numel() for p in m.parameters()) == PUBC_PARAMS == 32714
    assert tuple(m.op_type_embedding.weight.shape) == (4, 8)
    assert tuple(m.mch_type_embedding.weight.shape) == (1, 8)


def test_snapshot_source_and_refusal():
    assert snapshot_source({}) == 'none'
    assert snapshot_source({'pub_type_source': 'class'}) == 'class'
    refuse_mismatch('none', 'none')
    refuse_mismatch('class', 'class')
    with pytest.raises(SystemExit):
        refuse_mismatch('class', 'none')
    with pytest.raises(SystemExit):
        refuse_mismatch('none', 'class')


@pytest.fixture
def trainer_module():
    """train_family with the public recipe on the global configs (restored after)."""
    saved = copy.deepcopy(vars(configs))
    saved_env = os.environ.get('CUDA_VISIBLE_DEVICES')
    import train_family
    yield train_family
    for k in list(vars(configs)):
        if k not in saved:
            delattr(configs, k)
    for k, v in saved.items():
        setattr(configs, k, v)
    if saved_env is None:
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = saved_env


def _trainer(tf, source):
    c = configs
    c.data_source, c.family_base, c.n_j, c.n_m = 'FAMILY', 'sd2', 10, 5
    c.family_regimes, c.vali_size, c.pub_vali_dir = 'S', 2, ''
    c.use_film, c.first_mask, c.mask_ties = False, 'exact', 'exact'
    c.seed_train, c.num_envs, c.device = 301, 4, 'cpu'
    c.n_op_types, c.n_mch_types = 5, 9
    c.use_type_embedding = (source == 'class')
    c.pub_type_source = source
    return tf.FamilyTrainer(c)


@needs_data
def test_trainer_default_path_unchanged(trainer_module):
    tf = trainer_module
    out = {}
    for source in ('none', 'class'):
        t = _trainer(tf, source)
        n = sum(p.numel() for p in t.ppo.policy.parameters())
        assert n == (PUBX_PARAMS if source == 'none' else PUBC_PARAMS)
        v = t.vali_envs['S'].state.op_type_tensor
        assert (v is None) == (source == 'none')
        random.seed(5)
        np.random.seed(5)
        out[source] = t.sample_batch()
    (jn, pn, dn, on, mn), (jc, pc, dc, oc, mc) = out['none'], out['class']
    assert on is None and mn is None
    for a, b in zip(jn + pn, jc + pc):
        np.testing.assert_array_equal(a, b)
    for a, b, o in zip(dn, dc, oc):
        np.testing.assert_array_equal(a.op_class, b.op_class)
        np.testing.assert_array_equal(np.asarray(a.setup), np.asarray(b.setup))
        np.testing.assert_array_equal(o, a.op_class)
    assert all(np.array_equal(m, np.zeros(5, dtype=int)) for m in mc)
    with pytest.raises(SystemExit):        # class needs the embedding
        configs.use_type_embedding = False
        configs.pub_type_source = 'class'
        tf.FamilyTrainer(configs)


def _run_eval(cwd, model, extra):
    cmd = [sys.executable, os.path.join(ROOT, 'eval_family.py'), '--family_model', model,
           '--data', 'data/PUBLIC/family/pub_SD2_test_10x5+mix', '--base', 'record',
           '--regimes', 'N', '--methods', 'greedy', '--limit', '1', '--out_root', 'out',
           '--tag', '_typetest', '--device', 'cpu', '--device_id', '', '--torch_threads', '1',
           '--first_mask', 'exact', '--mask_ties', 'exact'] + extra
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    env.pop('PACT_FIRST_MASK', None)
    env.pop('PACT_MASK_TOL', None)
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=300)


@needs_public_instances()
@needs_data
@pytest.mark.skipif(not os.path.exists(CKPT), reason='the pubx checkpoint is not on disk')
def test_evaluation_refuses_a_type_mismatch(tmp_path):
    """eval_family.py in a scratch folder: data linked read-only, a random
    class-input checkpoint written there, the stored class-blind one linked."""
    os.makedirs(tmp_path / 'train_log' / 'FAMILY')
    os.makedirs(tmp_path / 'trained_network' / 'FAMILY')
    os.symlink(os.path.join(ROOT, 'data'), tmp_path / 'data')
    with open(SNAP) as f:
        snap = json.load(f)
    snap.update(use_type_embedding=True, n_op_types=4, n_mch_types=1, pub_type_source='class')
    name = '10x5+pubfamily+pubc-typetest'
    with open(tmp_path / 'train_log' / 'FAMILY' / f'config_{name}.json', 'w') as f:
        json.dump(snap, f)
    torch.manual_seed(0)
    torch.save(DANIEL(recipe('class')).state_dict(),
               tmp_path / 'trained_network' / 'FAMILY' / f'{name}.pth')
    blind = '10x5+pubfamily+pubx-v1-s301'
    os.symlink(SNAP, tmp_path / 'train_log' / 'FAMILY' / f'config_{blind}.json')
    os.symlink(CKPT, tmp_path / 'trained_network' / 'FAMILY' / f'{blind}.pth')

    r = _run_eval(tmp_path, name, [])                                 # class checkpoint, no option
    assert r.returncode != 0 and 'REFUSING' in (r.stdout + r.stderr)
    r = _run_eval(tmp_path, blind, ['--pub_type_source', 'class'])    # blind checkpoint, class option
    assert r.returncode != 0 and 'REFUSING' in (r.stdout + r.stderr)
    r = _run_eval(tmp_path, name, ['--pub_type_source', 'class'])     # matching option
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'feas=1/1' in r.stdout
    out = tmp_path / 'out' / 'pub_SD2_test_10x5+mix_typetest'
    assert (out / f'Result_greedy+{name}_N.npy').exists()
    assert 'types: pub_type_source=class' in (out / f'summary_{name}_typetest.md').read_text()
