"""
Shared helpers for running the HGNN scheduler (Song et al., IEEE TII 2023) on
PPVC instances with post-operation time-lags.

  * setup_paths()           puts this folder first on sys.path (for the HGNN
                            packages env/, graph/, utils/) and the repository
                            root after it (for the PPVC generator, the schedule
                            validator and the right-shift repair)
  * PPVCCaseGenerator       instance source for FJSPEnvLag(data_source='case'),
                            backed by ppvc_instance_generator
  * load_config / make_paras / build_model
  * make_file_env / greedy_rollout / extract_schedule
"""
import copy
import json
import os
import sys

HGNN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HGNN_DIR))


def setup_paths():
    if HGNN_DIR not in sys.path:
        sys.path.insert(0, HGNN_DIR)
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    import compat
    compat.install()


setup_paths()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from env.load_data import matrix_to_lines, nums_detec  # noqa: E402
from env.fjsp_env_lag import FJSPEnvLag  # noqa: E402
import ppvc_instance_generator as pig  # noqa: E402

FACTORIES = {'default': pig.DEFAULT_FACTORY, 'small': pig.SMALL_FACTORY,
             'tight': pig.TIGHT_FACTORY}


class PPVCCaseGenerator:
    '''
    Instance source with the interface FJSPEnv expects from its CaseGenerator:
    get_case(i) returns (lines, num_jobs, num_jobs); get_lags(i) returns the
    time-lag vector of the instance built by the last get_case(i) call.
    Instance i uses generator seed seeds[i].
    '''
    def __init__(self, n_modules, seeds, factory='default', class_mix='mixed', lag_scale=1.0):
        self.n_modules = n_modules
        self.lag_scale = float(lag_scale)  # training-time lag scaling, as train.py --lag_scale (rounded)
        self.seeds = list(seeds)
        self.station_counts = FACTORIES[factory]
        self.class_mix = class_mix
        self.lags = {}
        self.instances = {}

    def get_case(self, idx=0):
        jl, pt, meta = pig.ppvc_instance_generator(
            n_modules=self.n_modules, class_mix=self.class_mix,
            station_counts=self.station_counts, seed=int(self.seeds[idx]))
        if self.lag_scale != 1.0:
            meta = dict(meta, time_lag=np.rint(np.asarray(meta['time_lag']) * self.lag_scale).astype(int))
        self.lags[idx] = np.asarray(meta['time_lag'])
        self.instances[idx] = (jl, pt, meta)
        return matrix_to_lines(jl, pt), self.n_modules, self.n_modules

    def get_lags(self, idx):
        return self.lags[idx]


def load_config():
    with open(os.path.join(HGNN_DIR, 'config.json')) as f:
        return json.load(f)


def make_paras(cfg, device, lag_mode, lag_features, batch_size, num_jobs, num_mas, bound_pt='mean'):
    env_paras = copy.deepcopy(cfg['env_paras'])
    model_paras = copy.deepcopy(cfg['model_paras'])
    env_paras.update(device=device, lag_mode=lag_mode, lag_features=bool(lag_features),
                     batch_size=batch_size, num_jobs=num_jobs, num_mas=num_mas, bound_pt=bound_pt)
    if lag_features:
        env_paras['ope_feat_dim'] = 8
        model_paras['in_size_ope'] = 8
    model_paras['device'] = device
    model_paras['actor_in_dim'] = model_paras['out_size_ma'] * 2 + model_paras['out_size_ope'] * 2
    model_paras['critic_in_dim'] = model_paras['out_size_ma'] + model_paras['out_size_ope']
    return env_paras, model_paras


def build_model(model_paras, train_paras, num_envs=None, ckpt=None):
    import PPO_model
    model = PPO_model.PPO(model_paras, train_paras, num_envs=num_envs)
    if ckpt is not None:
        state = torch.load(ckpt, map_location=model_paras['device'])
        model.policy.load_state_dict(state)
        model.policy_old.load_state_dict(state)
    return model


def make_file_env(fjs_path, env_paras, env_cls=None, time_lags=None):
    '''One-instance environment built from an .fjs file (lags from its .meta.json).'''
    with open(fjs_path) as f:
        lines = f.readlines()
    num_jobs, num_mas, _ = nums_detec(lines)
    paras = dict(env_paras, num_jobs=num_jobs, num_mas=num_mas, batch_size=1)
    if env_cls is None:
        return FJSPEnvLag(case=[fjs_path], env_paras=paras, data_source='file', time_lags=time_lags)
    return env_cls(case=[fjs_path], env_paras=paras, data_source='file')


def greedy_rollout(env, model, memories=None, observer=None):
    '''Greedy decoding as in the upstream test.py; observer(env) is called after every step.'''
    import PPO_model
    memories = memories or PPO_model.Memory()
    state = env.state
    dones = env.done_batch
    done = False
    while ~done:
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=False, flag_train=False)
        state, _, dones = env.step(actions)
        done = dones.all()
        if observer is not None:
            observer(env)
    return copy.deepcopy(env.makespan_batch)


def extract_schedule(env, b=0):
    '''(job_length [J], op_pt [N, M], assigned_mch [N], op_ct [N]) of instance b, as numpy arrays.'''
    n = int(env.nums_opes[b])
    jl = env.nums_ope_batch[b].cpu().numpy().astype(int)
    pt = env.old_proc_times_batch[b, :n].cpu().numpy()
    sched = env.schedules_batch[b, :n].cpu().numpy()
    assert (sched[:, 0] == 1).all(), 'unscheduled operation left'
    return jl, pt, sched[:, 1].astype(int), sched[:, 3].astype(float)
