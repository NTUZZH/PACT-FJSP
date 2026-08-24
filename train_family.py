"""Joint constraint-family PPO training (this paper).

One policy over randomized regimes: each env in the batch draws a regime
(balanced over --family_regimes) and fresh member parameters every resample;
the reward is the family-admissible telescoped bound (built into
FJSPEnvFamily); the policy is FiLM-conditioned on the token g.

Usage (locked defaults, Appendix B):
  python train_family.py --data_source FAMILY --n_j 10 --use_film True \
         --use_type_embedding True --max_updates 2000 --seed_train 301 \
         --model_suffix joint-v1
Specialist training reuses this script with a single regime:
  python train_family.py ... --family_regimes L --max_updates 1000 \
         --model_suffix spec-L
"""
import json
import os
import random
import sys
import time
from copy import deepcopy

import numpy as np

from params import configs

# family mode fixes the feature widths BEFORE PPO_initialize reads them
configs.fea_j_input_dim = 12
configs.fea_m_input_dim = 10
if configs.data_source != 'FAMILY':
    print('[train_family] forcing data_source=FAMILY')
    configs.data_source = 'FAMILY'

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

from common_utils import (setup_seed, sample_action, greedy_select_action,
                          heuristic_select_action, strToSuffix)
from constraint_family.descriptor import make_descriptor
from constraint_family.fjsp_env_family import (FJSPEnvFamily,
                                                resolve_delta_mode)
from ppvc_instance_generator import (ppvc_instance_generator, build_factory,
                                     DEFAULT_FACTORY, SMALL_FACTORY,
                                     TIGHT_FACTORY)
from model.PPO import PPO_initialize, Memory

device = torch.device(configs.device)
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


class FamilyTrainer:

    def __init__(self, config):
        self.config = config
        self.n_j = config.n_j
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs
        self.regimes = [r.strip().upper() for r in
                        config.family_regimes.split(',') if r.strip()]

        # Domain-randomized arm: one shift dict applied to EVERY training and
        # validation descriptor draw. make_descriptor rejects unknown keys, so
        # a typo dies here rather than silently training the published arm
        # under a randomized name. Test descriptors are read from disk and are
        # never affected.
        self.desc_shift = (json.loads(config.dr_shift)
                           if getattr(config, 'dr_shift', '') else None)
        if self.desc_shift is not None:
            print(f'[train_family] DOMAIN-RANDOMIZED descriptor draws: '
                  f'{self.desc_shift}', flush=True)

        self.ppvc_factory = {'default': DEFAULT_FACTORY, 'small': SMALL_FACTORY,
                             'tight': TIGHT_FACTORY}[config.ppvc_factory]
        mch_type, _ = build_factory(self.ppvc_factory)
        self.n_m = len(mch_type)
        config.n_m = self.n_m

        tag = ''.join(self.regimes) if len(self.regimes) == 1 else 'joint'
        self.data_name = f'{self.n_j}x{self.n_m}+family'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        os.makedirs('./trained_network/FAMILY', exist_ok=True)
        os.makedirs('./train_log/FAMILY', exist_ok=True)

        torch.set_default_dtype(torch.float32)
        if device.type == 'cuda':
            torch.set_default_device('cuda')
        else:
            torch.set_default_device('cpu')

        setup_seed(config.seed_train)
        # Revision arms: the admissible-pair mask and the two extra pair
        # channels are properties of the MDP the policy trains on, so both
        # come from the config and are stamped into the banner below.
        self.mask_mode = getattr(config, 'mask_mode', 'nondelay')
        self.pair_extra = getattr(config, 'pair_extra', 'none')
        self.env = FJSPEnvFamily(self.n_j, self.n_m, family_mode=True,
                                 mask_mode=self.mask_mode,
                                 pair_extra=self.pair_extra)
        if self.mask_mode != 'nondelay' or self.pair_extra != 'none':
            print(f'[train_family] REVISION ARM: mask_mode={self.mask_mode} '
                  f'pair_extra={self.pair_extra}', flush=True)
        # The shaping potential's machine-side envelope. min = admissible+tight
        # (the paper's), zero = admissible+loose, max = INADMISSIBLE and still
        # terminally exact. See FJSPEnvFamily._bound_min_term.
        # The banner below records which envelope actually trained, so downstream
        # tooling can verify a run against its name.
        _GLOSS = {'min': "admissible, tight (the paper's)",
                  'zero': 'admissible, loose',
                  'max': 'INADMISSIBLE, tight'}
        self.delta_mode = resolve_delta_mode(config)
        self.env.delta_mode = self.delta_mode
        self.env.reward_delta_env = (self.delta_mode != 'zero')
        print(f'[train_family] shaping envelope: delta_mode={self.delta_mode} '
              f'({_GLOSS[self.delta_mode]})', flush=True)

        # ---- fixed per-regime validation sets (20 shared base instances) ----
        self.vali_size = min(config.vali_size, 20)
        self.vali_envs = {}
        self.vali_norm = {}
        vseed = config.seed_train_vali_datagen
        base = []
        for i in range(self.vali_size):
            jl, pt, meta = ppvc_instance_generator(
                n_modules=self.n_j, class_mix=config.ppvc_mix,
                station_counts=self.ppvc_factory, seed=vseed + i)
            base.append((jl, pt, meta))
        for regime in self.regimes:
            descs = [make_descriptor(regime, jl, pt, meta, self.n_m,
                                     seed=vseed * 1000 + i,
                                     shift=self.desc_shift)
                     for i, (jl, pt, meta) in enumerate(base)]
            # The validation env must use the SAME delay envelope, mask mode
            # and pair channels as the training env: the bound is operation
            # feature channel #2, the mask is the action space, and the pair
            # width is the actor's input, so checkpoint selection must see
            # the conventions the policy trains with.
            env = FJSPEnvFamily(self.n_j, self.n_m, family_mode=True,
                                mask_mode=self.mask_mode,
                                pair_extra=self.pair_extra)
            env.delta_mode = self.delta_mode
            env.reward_delta_env = (self.delta_mode != 'zero')
            env.set_initial_data([b[0] for b in base], [b[1] for b in base],
                                 descriptor_list=descs,
                                 op_type_list=[b[2]['op_type'] for b in base],
                                 mch_type_list=[b[2]['mch_type'] for b in base])
            self.vali_envs[regime] = env
            # per-regime scale: deterministic MWKR mean on this exact set
            # (heuristic_select_action is batch-of-1, so loop instances)
            ms = []
            for i, (jl, pt, meta) in enumerate(base):
                # The MWKR norm env stays at the DEFAULT mask and channels on
                # purpose: the norm is a fixed per-regime scale shared by
                # every arm, so it must not move with the arm's action space.
                henv = FJSPEnvFamily(self.n_j, self.n_m, family_mode=True)
                henv.set_initial_data([jl], [pt], descriptor_list=[descs[i]])
                np.random.seed(config.seed_test)
                while not henv.done().all():
                    a = heuristic_select_action('MWKR', henv)
                    henv.step(np.array([a]))
                ms.append(float(henv.current_makespan[0]))
            self.vali_norm[regime] = float(np.mean(ms))
        print('[vali] MWKR norms: ' +
              '  '.join(f'{r}={self.vali_norm[r]:.1f}' for r in self.regimes),
              flush=True)

        self.ppo = PPO_initialize()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)
        n_film = (sum(p.numel() for p in self.ppo.policy.film.parameters())
                  if getattr(self.ppo.policy, 'use_film', False) else 0)
        n_all = sum(p.numel() for p in self.ppo.policy.parameters())
        print(f'[model] params total={n_all}  film={n_film} '
              f'({100.0 * n_film / max(n_all, 1):.2f}%)', flush=True)

    def sample_batch(self):
        """num_envs fresh instances, regimes balanced, members redrawn.

        num_envs need not be divisible by the number of regimes (the locked
        defaults are 20 envs over 6 regimes), so each update has a remainder to
        place. It is dealt to a RANDOM subset of the regimes, which is what makes
        every regime's expected share exactly num_envs / len(regimes).

        Placing the remainder deterministically (e.g. giving it to the first
        regimes in the list) would hand those regimes a permanent surplus on
        every update -- 4/20 against 3/20 here, a standing 33% edge. Shuffling
        afterwards does not fix that: it permutes positions within the batch and
        leaves the multiset of counts exactly as skewed.

        Specialist training is unaffected: with one regime the remainder is zero.
        """
        k, rem = divmod(self.num_envs, len(self.regimes))
        regs = list(self.regimes) * k
        if rem:
            regs += random.sample(list(self.regimes), rem)
        random.shuffle(regs)
        jls, pts, descs, opts, mchts = [], [], [], [], []
        for r in regs:
            inst_seed = int(np.random.randint(0, 2 ** 31 - 1))
            jl, pt, meta = ppvc_instance_generator(
                n_modules=self.n_j, class_mix=self.config.ppvc_mix,
                station_counts=self.ppvc_factory, seed=inst_seed)
            desc_seed = int(np.random.randint(0, 2 ** 31 - 1))
            descs.append(make_descriptor(r, jl, pt, meta, self.n_m,
                                         seed=desc_seed,
                                         shift=self.desc_shift))
            jls.append(jl)
            pts.append(pt)
            opts.append(meta['op_type'])
            mchts.append(meta['mch_type'])
        return jls, pts, descs, opts, mchts

    def policy_rollout_kwargs(self, state):
        return dict(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor,
                    comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor,
                    op_type=state.op_type_tensor,
                    mch_type=state.mch_type_tensor,
                    token=getattr(state, 'token_tensor', None))

    def validate(self):
        self.ppo.policy.eval()
        scores = {}
        for regime, env in self.vali_envs.items():
            state = env.reset()
            while not env.done().all():
                with torch.no_grad():
                    pi, _ = self.ppo.policy(**self.policy_rollout_kwargs(state))
                action = greedy_select_action(pi)
                state, _, _ = env.step(action.cpu().numpy())
            scores[regime] = float(env.current_makespan.mean())
        self.ppo.policy.train()
        rel = np.mean([scores[r] / self.vali_norm[r] for r in self.regimes])
        return rel, scores

    def train(self):
        setup_seed(self.config.seed_train)
        self.log, self.validation_log = [], []
        self.record = float('inf')
        with open(f'./train_log/FAMILY/config_{self.model_name}.json', 'w') as f:
            json.dump({k: v for k, v in vars(self.config).items()}, f,
                      indent=1, default=str)
        print(f'[train_family] model={self.model_name} regimes={self.regimes} '
              f'updates={self.max_updates}', flush=True)

        t0 = time.time()
        for i_update in range(self.max_updates):
            ep_st = time.time()
            if i_update % self.reset_env_timestep == 0:
                jls, pts, descs, opts, mchts = self.sample_batch()
                state = self.env.set_initial_data(
                    jls, pts, descriptor_list=descs, op_type_list=opts,
                    mch_type_list=mchts)
            else:
                state = self.env.reset()

            ep_rewards = -deepcopy(self.env.init_quality)
            while True:
                self.memory.push(state)
                with torch.no_grad():
                    pi_envs, vals_envs = self.ppo.policy_old(
                        **self.policy_rollout_kwargs(state))
                action_envs, action_logprob_envs = sample_action(pi_envs)
                state, reward, done = self.env.step(
                    actions=action_envs.cpu().numpy())
                ep_rewards += reward
                self.memory.done_seq.append(
                    torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(
                    torch.from_numpy(reward).to(device))
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))
                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()
            self.log.append([i_update, float(np.mean(ep_rewards))])

            if (i_update + 1) % self.validate_timestep == 0:
                rel, scores = self.validate()
                if rel < self.record:
                    self.record = rel
                    torch.save(self.ppo.policy.state_dict(),
                               f'./trained_network/FAMILY/{self.model_name}.pth')
                self.validation_log.append(rel)
                with open(f'./train_log/FAMILY/valiquality_{self.model_name}.txt',
                          'w') as f:
                    f.write(str(self.validation_log))
                print(f'[vali @{i_update + 1}] rel={rel:.4f} (best '
                      f'{self.record:.4f})  ' +
                      '  '.join(f'{r}={scores[r]:.1f}' for r in self.regimes),
                      flush=True)

            print(f'Ep {i_update + 1}\t reward {np.mean(ep_rewards):.3f}\t '
                  f'ms {np.mean(self.env.current_makespan):.1f}\t '
                  f'loss {loss:.6f}\t {time.time() - ep_st:.1f}s', flush=True)

        with open(f'./train_log/FAMILY/reward_{self.model_name}.txt', 'w') as f:
            f.write(str(self.log))
        print(f'[train_family] done in {(time.time() - t0) / 3600:.2f} h; '
              f'best rel={self.record:.4f}', flush=True)


if __name__ == '__main__':
    trainer = FamilyTrainer(configs)
    trainer.train()
