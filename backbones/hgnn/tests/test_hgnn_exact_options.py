"""
Tests of the trainer and evaluator options used by the exact-mask HGNN runs
(scripts/run_hgnn_exact_queue.sh) and of the learning-rate selection
(scripts/hgnn_select_lr.py). CPU only:

    CUDA_VISIBLE_DEVICES= python -m pytest -q backbones/hgnn/tests/test_hgnn_exact_options.py

  * --lr defaults to the upstream value of config.json, is recorded as args.lr and
    train_paras.lr, and is the learning rate of the optimizer;
  * the run name carries the suffix after the seed;
  * the trainer refuses to start while PACT_FIRST_MASK or PACT_MASK_TOL is set;
  * the evaluator reads the mask from the checkpoint's configuration, refuses an
    exact-mask checkpoint without --tag, writes to <size><tag>, and gives the same
    makespans on two CPU runs;
  * the selection picks the lowest validation score, breaks a tie toward 2e-4, and
    refuses runs whose validation norms differ or whose configuration is not the
    tuning recipe.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HGNN_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(HGNN_DIR))
sys.path.insert(0, HGNN_DIR)
import ppvc_hgnn as P  # noqa: E402
import train_family_hgnn as T  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location('hgnn_select_lr', os.path.join(REPO, 'scripts', 'hgnn_select_lr.py'))
SEL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SEL)

PY = sys.executable
CPU_ENV = {k: v for k, v in os.environ.items() if k not in ('PACT_FIRST_MASK', 'PACT_MASK_TOL')}
CPU_ENV['CUDA_VISIBLE_DEVICES'] = ''


def test_lr_option_default_and_value():
    cfg = P.load_config()
    args, _ = T.parse_args(['--device', 'cpu'])
    assert args.lr == cfg['train_paras']['lr'] == 2e-4
    args, _ = T.parse_args(['--device', 'cpu', '--lr', '1e-4'])
    assert args.lr == 1e-4


def test_run_name_carries_the_suffix_after_the_seed():
    a, _ = T.parse_args(['--hgnn_reward', 'certified', '--seed', '301', '--model_suffix', 'xmask-lr1e-4'])
    assert T.model_name(a, 25) == '10x25+family+hgnn-certified-joint-s301-xmask-lr1e-4'
    a, _ = T.parse_args(['--family_regimes', 'SW', '--hgnn_reward', 'certified', '--seed', '301',
                         '--model_suffix', 'xmask'])
    assert T.model_name(a, 25) == '10x25+family+hgnn-certified-SW-s301-xmask'


@pytest.mark.parametrize('var', ['PACT_FIRST_MASK', 'PACT_MASK_TOL'])
def test_trainer_refuses_mask_variables(tmp_path, var):
    r = subprocess.run([PY, os.path.join(HGNN_DIR, 'train_family_hgnn.py'), '--device', 'cpu',
                        '--out_root', str(tmp_path)], capture_output=True, text=True,
                       env=dict(CPU_ENV, **{var: '1'}))
    assert r.returncode != 0 and 'REFUSING' in (r.stdout + r.stderr) and var in (r.stdout + r.stderr)
    assert not os.path.exists(os.path.join(str(tmp_path), 'train_log'))


@pytest.fixture(scope='module')
def exact_run(tmp_path_factory):
    '''One-iteration exact-mask run at lr 1e-4 on the CPU; returns (root, name, stdout).'''
    root = str(tmp_path_factory.mktemp('hgnn_exact'))
    cmd = [PY, '-c', LR_PROBE, root]
    r = subprocess.run(cmd, capture_output=True, text=True, env=CPU_ENV, cwd=REPO)
    assert r.returncode == 0, r.stderr[-3000:]
    return root, '10x25+family+hgnn-certified-joint-s301-t', r.stdout


# runs train_family_hgnn.main in a child process and reports the optimizer's learning rate
LR_PROBE = r'''
import sys
root = sys.argv[1]
sys.path.insert(0, 'backbones/hgnn')
import ppvc_hgnn as P
import train_family_hgnn as T
seen = []
build = P.build_model
def spy(model_paras, train_paras, **kw):
    m = build(model_paras, train_paras, **kw)
    seen.append((train_paras['lr'], m.optimizer.param_groups[0]['lr']))
    return m
P.build_model = spy
T.main(['--family_regimes', 'N,L,S,W,LS,LW', '--hgnn_reward', 'certified', '--hgnn_mask', 'exact',
        '--num_envs', '2', '--max_iterations', '1', '--save_timestep', '1', '--vali_size', '1',
        '--seed', '301', '--lr', '1e-4', '--model_suffix', 't', '--device', 'cpu', '--threads', '1',
        '--out_root', root])
print('OPTIMIZER_LR', seen)
'''


def test_lr_recorded_and_used(exact_run):
    root, name, out = exact_run
    assert "OPTIMIZER_LR [(0.0001, 0.0001)]" in out, out[-1500:]
    assert 'mask=exact iterations=1 ' in out and 'lr=0.0001' in out
    with open(os.path.join(root, 'train_log', 'FAMILY', 'config_{0}.json'.format(name))) as f:
        cfg = json.load(f)
    assert cfg['args']['lr'] == 1e-4 and cfg['train_paras']['lr'] == 1e-4
    assert cfg['upstream_config']['train_paras']['lr'] == 2e-4
    assert cfg['hgnn_mask'] == 'exact' and cfg['n_params'] == 204250
    assert os.path.exists(os.path.join(root, 'trained_network', 'FAMILY', name + '.pth'))


def _eval(root, name, out, extra, env=None):
    cmd = [PY, os.path.join(HGNN_DIR, 'eval_family_hgnn.py'), '--family_model', name, '--sizes', '10x25',
           '--regimes', 'N,LSW', '--limit', '2', '--model_root', root, '--out_root', out, '--device', 'cpu',
           '--threads', '1'] + extra
    return subprocess.run(cmd, capture_output=True, text=True, env=env or CPU_ENV)


def test_evaluator_exact_mask_campaign_and_determinism(exact_run, tmp_path):
    root, name, _ = exact_run
    out = str(tmp_path / 'res')
    r = _eval(root, name, out, [])
    assert r.returncode != 0 and 'pass --tag' in (r.stdout + r.stderr)
    assert not os.path.exists(out)
    r = _eval(root, name, out, ['--tag', '_t'])
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'mask=exact' in r.stdout and 'campaign=_t' in r.stdout
    first = {R: np.load(os.path.join(out, '10x25_t', 'Result_hgnn-certified-joint-s301-t_{0}.npy'.format(R)))
             for R in ('N', 'LSW')}
    assert all(a.shape == (2, 2) for a in first.values())
    assert not os.path.exists(os.path.join(out, '10x25'))
    r = _eval(root, name, out, ['--tag', '_t'])
    assert r.returncode != 0 and 'exists' in (r.stdout + r.stderr)
    r = _eval(root, name, out, ['--tag', '_t', '--force'])
    assert r.returncode == 0, r.stderr[-2000:]
    for R, a in first.items():
        b = np.load(os.path.join(out, '10x25_t', 'Result_hgnn-certified-joint-s301-t_{0}.npy'.format(R)))
        assert np.array_equal(a[:, 0], b[:, 0]), (R, a[:, 0], b[:, 0])


def test_evaluator_refuses_mask_variables_for_the_pact_mask(exact_run, tmp_path):
    root, name, _ = exact_run
    cfg_p = os.path.join(root, 'train_log', 'FAMILY', 'config_{0}.json'.format(name))
    with open(cfg_p) as f:
        cfg = json.load(f)
    pname = name + 'p'
    cfg.update(model_name=pname, hgnn_mask='pact')
    with open(os.path.join(root, 'train_log', 'FAMILY', 'config_{0}.json'.format(pname)), 'w') as f:
        json.dump(cfg, f)
    src = os.path.join(root, 'trained_network', 'FAMILY', name + '.pth')
    dst = os.path.join(root, 'trained_network', 'FAMILY', pname + '.pth')
    if not os.path.exists(dst):
        os.symlink(src, dst)
    r = _eval(root, pname, str(tmp_path / 'res'), [], env=dict(CPU_ENV, PACT_MASK_TOL='1'))
    assert r.returncode != 0 and 'PACT_MASK_TOL' in (r.stdout + r.stderr)


# --------------------------------------------------------------------------------------------- selection

NORMS = {'N@10': 156.6, 'L@10': 226.95, 'S@10': 235.3, 'W@10': 177.65, 'LS@10': 304.4, 'LW@10': 243.4}


def _tuning_run(root, lr, curve, norms=NORMS, **over):
    name = SEL.PREFIX + SEL.run_name(lr)
    d = os.path.join(root, 'train_log', 'FAMILY')
    os.makedirs(d, exist_ok=True)
    cfg = {'model_name': name, 'hgnn_mask': 'exact', 'hgnn_reward': 'certified', 'n_params': 204250,
           'args': {'seed': 301, 'max_iterations': 2000, 'lr': float(lr), 'family_regimes': 'N,L,S,W,LS,LW'},
           'train_paras': {'lr': float(lr)}, 'vali_norm': dict(norms)}
    for k, v in over.items():
        cfg['args'][k] = v
    with open(os.path.join(d, 'config_{0}.json'.format(name)), 'w') as f:
        json.dump(cfg, f)
    with open(os.path.join(d, 'valiquality_{0}.txt'.format(name)), 'w') as f:
        f.write(str(curve))
    with open(os.path.join(root, 'train_log', 'FAMILY_{0}.log'.format(SEL.run_name(lr))), 'w') as f:
        f.write('[train_family_hgnn] done in 3.10 h; best rel={0:.4f}; checkpoint x\n'.format(min(curve)))


def _curve(best, at):
    c = [1.0] * 200
    c[at] = best
    return c


def test_selection_picks_the_lowest_score(tmp_path):
    root = str(tmp_path)
    _tuning_run(root, '1e-4', _curve(0.8810, 150))
    _tuning_run(root, '2e-4', _curve(0.8850, 120))
    _tuning_run(root, '4e-4', _curve(0.8900, 90))
    chosen, scores = SEL.select(root)
    assert chosen == '1e-4' and scores['1e-4'] == (0.8810, 1510)
    _tuning_run(root, '4e-4', _curve(0.8810, 90))
    _tuning_run(root, '2e-4', _curve(0.8810, 10))
    assert SEL.select(root)[0] == '2e-4'


def test_selection_refuses_other_norms_or_recipe(tmp_path):
    root = str(tmp_path)
    for lr in SEL.RATES:
        _tuning_run(root, lr, _curve(0.88, 5))
    _tuning_run(root, '4e-4', _curve(0.87, 5), norms=dict(NORMS, **{'S@10': 240.4}))
    with pytest.raises(SystemExit):
        SEL.select(root)
    _tuning_run(root, '4e-4', _curve(0.87, 5), max_iterations=1000)
    with pytest.raises(SystemExit):
        SEL.select(root)
