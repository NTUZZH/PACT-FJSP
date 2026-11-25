'''
HGNN environment for PACT's constraint family: every combination of minimum
time lags (L), sequence-dependent attached setups (S) and machine
unavailability windows (W), switched per instance by the descriptor's
activation bits.

The class keeps the observation of the lag-aware HGNN environment
(env/fjsp_env_lag.py, itself the upstream env/fjsp_env.py with lags): the same
operation and machine features, the same graph tensors, the same state
fields. Three things change.

1. Transition. The two hooks of PACT's family transition, on the integer-hour
   axis:
     job side      r_i <- C_ij + a_L * l_ij
     machine side  b'  = push(max(r_i, f_k), sigma + p, a_W * W_k),
                   sigma = a_S * S[lambda_k, tau_ij],
                   C_ij = b' + sigma + p,  f_k <- C_ij,  lambda_k <- tau_ij
   with half-open outage intervals and no preemption (the setup and the
   processing form one block that must avoid every outage). The processing
   start of the dispatched pair is b' + sigma. The reference is
   constraint_family/reference_sim.simulate; tests/test_family_hgnn.py checks
   operation-by-operation equality.

2. Decision time and legal pairs. The upstream environment advances its
   clock to the next machine release or job release and then admits every
   pair whose job and machine are both idle. Under setups and windows that
   admits pairs that cannot start at that time (a machine inside an outage, or
   a changeover that delays the processing start). The decision time is
   therefore PACT's: t* = the earliest processing start over all compatible
   (job, machine) pairs, and the legal pairs are published per pair in
   state.pair_eligible_batch, which the policy intersects with its own
   job/machine mask (PPO_model.py). Two mask modes:
     'pact'  (default) the admissible pairs of PACT's own environment, read
             from a shadow FJSPEnvFamily stepped with the same actions. This
             is the mask PACT's policy and its dispatching rules use, so the
             action sets are identical. PACT's mask is evaluated on its
             normalized time axis, where a pair whose start ties t* exactly
             in hours can differ by one rounding unit and drop out; and at the
             first decision PACT admits every compatible pair. Both effects
             are inherited unchanged.
     'exact' the non-delay set on the integer-hour axis, at every decision
             including the first. In regime N it equals the upstream HGNN
             eligibility (job idle and machine idle at the decision time).
   With every rule inactive both the clock and the eligible set reduce to the
   upstream ones; the tests replay random sequences against env/fjsp_env.py.

3. Reward. The completion estimate HGNN propagates along job chains drives
   both the reward (decrease of the estimated makespan) and operation
   features 4 and 5 (estimated job completion and start), as upstream.
     'published'  HGNN's own estimate: mean processing time over compatible
                  machines for unscheduled operations, no lags, no setups,
                  no windows. It is NOT a lower bound on the makespan, even
                  with every rule inactive (the mean exceeds the minimum), so
                  intermediate rewards can be positive. It is exact at
                  termination, so the return still telescopes to the initial
                  estimate minus the realized makespan.
     'certified'  PACT's estimate C-hat (paper Eq. (recursion)): realized
                  completions for scheduled operations, then along each job
                  chain the predecessor's lag plus m_ij = min_k p_ijk + a_S *
                  min over source classes of S[., tau_ij]. m_ij is evaluated by
                  PACT's FJSPEnvFamily._bound_min_term (pact_bridge). HGNN's
                  chain propagation maintains the recursion because it is
                  additive along jobs. Admissible at every step and exact at
                  termination; both are asserted on every rollout when
                  checks=True (rewards never positive, final estimate equals
                  the realized makespan) and, in 'pact' mask mode, the
                  estimate is compared with PACT's own at every step.

All instances of a batch must have the same operation count (PPVC routes are
fixed at 22 operations per module, and PACT's environment requires it).
'''
import copy

import numpy as np
import torch

from env.fjsp_env_lag import FJSPEnvLag, EnvState, convert_feat_job_2_ope
from env.load_data import load_fjs, nums_detec, matrix_to_lines

REWARDS = ('published', 'certified')
MASKS = ('pact', 'exact')


class FJSPEnvFamilyHGNN(FJSPEnvLag):
    '''
    :param job_length_list: list of [J] int arrays (operations per job)
    :param op_pt_list: list of [N, M] processing-time arrays in hours (0 = incompatible)
    :param descriptor_list: list of constraint_family.descriptor.ConstraintDescriptor (hours)
    :param env_paras: HGNN env parameters (show_mode, ope_feat_dim, ma_feat_dim, device, lag_features)
    :param reward: 'published' or 'certified' (module docstring)
    :param mask: 'pact' or 'exact' (module docstring)
    :param checks: assert the transition, admissibility and terminal exactness on the fly
    '''

    def __init__(self, job_length_list, op_pt_list, descriptor_list, env_paras, reward='certified',
                 mask='pact', checks=True):
        import pact_bridge
        assert reward in REWARDS, reward
        assert mask in MASKS, mask
        self.reward_mode = reward
        self.mask_mode = mask
        self.checks = checks
        B = len(job_length_list)
        assert len(op_pt_list) == B and len(descriptor_list) == B
        self.show_mode = env_paras["show_mode"]
        self.batch_size = B
        self.num_jobs = int(len(job_length_list[0]))
        self.num_mas = int(op_pt_list[0].shape[1])
        self.paras = dict(env_paras, batch_size=B, num_jobs=self.num_jobs, num_mas=self.num_mas)
        self.device = env_paras["device"]
        num_mas = self.num_mas
        num_jobs = self.num_jobs

        # instances through the upstream text format and loader, so the graph tensors are the upstream ones
        lines = [matrix_to_lines(np.asarray(jl), np.asarray(pt)) for jl, pt in zip(job_length_list, op_pt_list)]
        counts = [nums_detec(l)[2] for l in lines]
        assert len(set(counts)) == 1, 'every instance of a batch must have the same number of operations'
        self.num_opes = counts[0]
        num_data = 8
        tensors = [[] for _ in range(num_data)]
        for i in range(B):
            load_data = load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # ---- descriptor (static, hours) -------------------------------------------------------------------
        self.descriptors = list(descriptor_list)
        self.job_length_list = [np.asarray(jl, dtype=int) for jl in job_length_list]
        self.op_pt_list = [np.asarray(pt, dtype=float) for pt in op_pt_list]
        for d in self.descriptors:
            assert getattr(d, 'release', None) is None, 'release dates are not part of this port'
            assert d.setup_convention == 'attached', 'only the attached setup convention is admissible'
        P = pact_bridge.pact()
        C, idle = P.D.N_CLASSES, P.D.IDLE_CLASS
        self.idle_class = idle
        f64 = dict(dtype=torch.float64)
        self.bits = torch.tensor([[d.has_lag, d.has_setup, d.has_window] for d in self.descriptors], **f64)
        self.pt64 = torch.as_tensor(np.stack(self.op_pt_list), **f64)
        lag = np.zeros((B, self.num_opes))
        cls = np.zeros((B, self.num_opes), dtype=np.int64)
        setup = np.zeros((B, C + 1, C))
        for e, d in enumerate(self.descriptors):
            if d.has_lag:
                lag[e, :len(d.lag)] = np.asarray(d.lag, dtype=float)
            cls[e, :len(d.op_class)] = np.asarray(d.op_class, dtype=int)
            if d.has_setup and d.setup is not None:
                setup[e] = np.asarray(d.setup, dtype=float)
        lag = torch.as_tensor(lag, **f64)
        # a lag after the last operation of a job delays nothing
        lag.scatter_(1, self.end_ope_biases_batch, 0.0)
        self.lag64 = lag
        self.op_class = torch.as_tensor(cls)
        self.setup64 = torch.as_tensor(setup, **f64)
        K = max([max([len(w) for w in d.windows] + [0]) if (d.has_window and d.windows is not None) else 0
                 for d in self.descriptors] + [0])
        self.num_win = K
        win_s = np.full((B, num_mas, max(K, 1)), np.inf)
        win_e = np.full((B, num_mas, max(K, 1)), np.inf)
        for e, d in enumerate(self.descriptors):
            if d.has_window and d.windows is not None:
                for m, spans in enumerate(d.windows):
                    for k, (s, en) in enumerate(spans):
                        win_s[e, m, k] = float(s)
                        win_e[e, m, k] = float(en)
        self.win_s = torch.as_tensor(win_s, **f64)
        self.win_e = torch.as_tensor(win_e, **f64)

        # lags as HGNN sees them (float32), for the lag features and the chain estimate
        self.lag_mode = 'aware'
        self.lag_features = env_paras.get("lag_features", False)
        assert (not self.lag_features) or self.paras["ope_feat_dim"] == 8, "lag features need ope_feat_dim 8"
        self.lag_batch = lag.float()
        pos_times = torch.where(self.proc_times_batch > 0, self.proc_times_batch, float('inf'))
        self.min_proc_batch = torch.min(pos_times, dim=2)[0]
        self.min_proc_batch = torch.where(torch.isinf(self.min_proc_batch), 0.0, self.min_proc_batch)
        if reward == 'certified':
            m = pact_bridge.bound_min_term(self.op_pt_list, self.descriptors)
            self.est_min_term = torch.as_tensor(m, dtype=torch.float32)
            self.est_lag = self.lag_batch
        else:
            self.est_min_term = None
            self.est_lag = torch.zeros_like(self.lag_batch)

        # ---- features (upstream definitions) --------------------------------------------------------------
        feat_opes_batch = torch.zeros(size=(B, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(B, self.paras["ma_feat_dim"], num_mas))
        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        est_proc = feat_opes_batch[:, 2, :] if reward == 'published' else self.est_min_term
        feat_opes_batch[:, 5, :] = torch.bmm((est_proc + self.est_lag).unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze(1)
        end_time_batch = (feat_opes_batch[:, 5, :] + est_proc).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        if self.lag_features:
            feat_opes_batch[:, 6, :] = self.lag_batch
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.shadow = None
        self._init_dynamic()
        self.old_state = copy.deepcopy(self.state)

    # ------------------------------------------------------------------------------------------------------
    # dynamic state
    # ------------------------------------------------------------------------------------------------------

    def _init_dynamic(self):
        B, J, M = self.batch_size, self.num_jobs, self.num_mas
        self.batch_idxes = torch.arange(B)
        self.time = torch.zeros(B)
        self.N = torch.zeros(B).int()
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(B, J), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(B, J), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(B, M), dtype=torch.bool, fill_value=False)
        # family dynamics on the integer-hour axis
        self.job_ready = torch.zeros(B, J, dtype=torch.float64)
        self.mch_free = torch.zeros(B, M, dtype=torch.float64)
        self.mch_last = torch.full((B, M), self.idle_class, dtype=torch.long)
        self.op_start = torch.full((B, self.num_opes), -1.0, dtype=torch.float64)
        self.op_end = torch.full((B, self.num_opes), -1.0, dtype=torch.float64)
        self.op_mch = torch.full((B, self.num_opes), -1, dtype=torch.long)
        self.decisions = 0
        # kept for the inherited helpers (render, validate_gantt of the lag environment)
        self.job_ready_batch = self.job_ready.float()
        self.schedules_batch = torch.zeros(size=(B, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(B, M, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(B, M))
        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = bool(self.done_batch.all())

        if self.mask_mode == 'pact':
            if self.shadow is None:
                import pact_bridge
                self.shadow = pact_bridge.make_shadow(self.job_length_list, self.op_pt_list, self.descriptors)
                self._shadow_scale = (np.max(np.stack(self.op_pt_list)) - np.min(np.stack(self.op_pt_list))
                                      + 1e-8)
            else:
                self.shadow.reset()
        start, compat = self._pair_start()
        tstar = self._tstar(start, compat)
        self.time = tstar.float()
        self._set_masks(start, compat, tstar, first=True)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)
        self.state.pair_eligible_batch = self.pair_eligible

    def _pair_start(self):
        '''Processing start of every (candidate operation, machine) pair, [B, J, M] hours, through the two
        hooks; +inf for incompatible pairs and finished jobs. Returns (start, compat).'''
        B, J, M = self.batch_size, self.num_jobs, self.num_mas
        step = torch.minimum(self.ope_step_batch, self.end_ope_biases_batch)            # [B, J]
        p = self.pt64.gather(1, step[:, :, None].expand(-1, -1, M))                    # [B, J, M]
        compat = (p > 0) & ~self.mask_job_finish_batch[:, :, None]
        cls = self.op_class.gather(1, step)                                              # [B, J]
        b = torch.arange(B)[:, None, None]
        sigma = self.setup64[b, self.mch_last[:, None, :], cls[:, :, None]]              # [B, J, M]
        block = torch.maximum(self.job_ready[:, :, None], self.mch_free[:, None, :])
        if self.num_win:
            dur = sigma + p
            for k in range(self.num_win):
                ws = self.win_s[:, None, :, k]
                we = self.win_e[:, None, :, k]
                overlap = (block < we) & (block + dur > ws)
                block = torch.where(overlap, we, block)
        start = torch.where(compat, block + sigma, torch.full_like(block, float('inf')))
        return start, compat

    def _tstar(self, start, compat):
        '''Decision time per instance: earliest processing start over compatible pairs; the current time for
        finished instances.'''
        t = start.flatten(1).min(dim=1)[0]
        return torch.where(torch.isinf(t), self.time.double(), t)

    def _set_masks(self, start, compat, tstar, first=False):
        '''Upstream job/machine masks at the decision time and the pair-level legal set.'''
        self.pair_start = start
        exact = compat & (start <= tstar[:, None, None])
        self.exact_eligible = exact
        if self.mask_mode == 'exact':
            self.pair_eligible = exact
        else:
            # the mask PACT's policy reads (state tensor; PACT's reset() restores it, the env-side array not)
            pact = torch.as_tensor(~self.shadow.state.dynamic_pair_mask_tensor.cpu().numpy())
            if self.checks and not first:
                # PACT's set never exceeds the exact non-delay set (it can only lose ties to rounding)
                assert not bool((pact & ~exact).any()), 'PACT mask admits a pair that cannot start at t*'
            self.pair_eligible = pact & compat
        t = self.time.double()[:, None]
        self.mask_job_procing_batch = self.job_ready > t
        self.mask_ma_procing_batch = self.mch_free > t
        self.machines_batch[:, :, 0] = (~self.mask_ma_procing_batch).float()

    # ------------------------------------------------------------------------------------------------------
    # transition
    # ------------------------------------------------------------------------------------------------------

    def step(self, actions):
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        bi = self.batch_idxes
        self.N += 1

        if self.checks:
            assert bool(self.pair_eligible[bi, jobs, mas].all()), 'action outside the legal pair set'
            assert bool((torch.minimum(self.ope_step_batch, self.end_ope_biases_batch)[bi, jobs] == opes).all())
        start = self.pair_start[bi, jobs, mas]
        p64 = self.pt64[bi, opes, mas]
        if self.checks and (self.mask_mode == 'exact' or self.decisions > 0):
            assert bool((start == self.time[bi].double()).all()), 'dispatched pair does not start at t*'
        end = start + p64
        self.op_start[bi, opes] = start
        self.op_end[bi, opes] = end
        self.op_mch[bi, opes] = mas
        self.job_ready[bi, jobs] = end + self.lag64[bi, opes]
        self.mch_free[bi, mas] = end
        self.mch_last[bi, mas] = self.op_class[bi, opes]
        self.decisions += 1

        # Removed unselected O-M arcs of the scheduled operations (upstream)
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[bi, mas] = 1
        self.ope_ma_adj_batch[bi, opes] = remain_ope_ma_adj[bi, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        proc_times = self.proc_times_batch[bi, opes, mas]
        self.feat_opes_batch[bi, :3, opes] = torch.stack((torch.ones(bi.size(0), dtype=torch.float),
                                                          torch.ones(bi.size(0), dtype=torch.float),
                                                          proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[bi, jobs], self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[bi, last_opes, :] = 0

        start_ope = self.num_ope_biases_batch[bi, jobs]
        end_ope = self.end_ope_biases_batch[bi, jobs]
        for i in range(bi.size(0)):
            self.feat_opes_batch[bi[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

        # realized start of the dispatched operation, then the chain estimate of the unscheduled ones
        self.feat_opes_batch[bi, 5, opes] = start.float()
        is_scheduled = self.feat_opes_batch[bi, 0, :]
        proc_est = self.feat_opes_batch[bi, 2, :]
        if self.reward_mode == 'certified':
            proc_est = torch.where(is_scheduled == 1, proc_est, self.est_min_term[bi, :])
        start_times = self.feat_opes_batch[bi, 5, :] * is_scheduled
        un_scheduled = 1 - is_scheduled
        estimate_times = torch.bmm((start_times + proc_est + self.est_lag[bi, :]).unsqueeze(1),
                                   self.cal_cumul_adj_batch[bi, :, :]).squeeze(1) * un_scheduled
        self.feat_opes_batch[bi, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[bi, 5, :] + proc_est).gather(1, self.end_ope_biases_batch[bi, :])
        self.feat_opes_batch[bi, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[bi, :])

        self.schedules_batch[bi, opes, :2] = torch.stack((torch.ones(bi.size(0)), mas), dim=1)
        self.schedules_batch[bi, :, 2] = self.feat_opes_batch[bi, 5, :]
        self.schedules_batch[bi, :, 3] = self.feat_opes_batch[bi, 5, :] + self.feat_opes_batch[bi, 2, :]
        self.machines_batch[bi, mas, 0] = torch.zeros(bi.size(0))
        self.machines_batch[bi, mas, 1] = end.float()
        self.machines_batch[bi, mas, 2] += proc_times
        self.machines_batch[bi, mas, 3] = jobs.float()
        self.job_ready_batch = self.job_ready.float()

        self.feat_mas_batch[bi, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[bi, :, :], dim=1).float()
        self.feat_mas_batch[bi, 1, mas] = end.float()
        utiliz = self.machines_batch[bi, :, 2]
        cur_time = self.time[bi, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[bi, None] + 1e-9)
        self.feat_mas_batch[bi, 2, :] = utiliz

        self.ope_step_batch[bi, jobs] += 1
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max

        # PACT's environment takes the same decisions (it steps every instance of its batch)
        if self.mask_mode == 'pact':
            assert bi.size(0) == self.batch_size, 'PACT mask needs every instance of the batch to be live'
            a = (jobs * self.num_mas + mas).cpu().numpy()
            self.shadow.step(a)

        # next decision time: the earliest processing start over the remaining pairs
        start_all, compat = self._pair_start()
        tstar = self._tstar(start_all, compat)
        trans = (tstar > self.time.double()) & ~self.done_batch
        if bool(trans.any()):
            self.time = torch.where(trans, tstar, self.time.double()).float()
            # upstream next_time() recomputes the utilization of every instance with a 1e-5 guard
            utiliz = self.machines_batch[:, :, 2]
            cur_time = self.time[:, None].expand_as(utiliz)
            utiliz = torch.minimum(utiliz, cur_time)
            utiliz = utiliz.div(self.time[:, None] + 1e-5)
            self.feat_mas_batch[:, 2, :] = utiliz
        self._set_masks(start_all, compat, tstar)

        if self.lag_features:
            self.update_remain_lag()

        if self.checks:
            self._check_step(bi, opes, end)

        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch, self.ope_step_batch, self.time)
        self.state.pair_eligible_batch = self.pair_eligible
        return self.state, self.reward_batch, self.done_batch

    def _check_step(self, bi, opes, end):
        '''On-the-fly checks: realized completion equals PACT's, the certified estimate never decreases and
        equals PACT's, and every estimate is exact at termination.'''
        if self.mask_mode == 'pact':
            sh = self.shadow
            ct = np.asarray(sh.true_op_ct[bi.cpu().numpy(), opes.cpu().numpy()], dtype=float)
            assert np.array_equal(ct, end.cpu().numpy()), 'completion differs from PACT environment'
            if self.reward_mode == 'certified':
                chat = sh.op_ct_lb.max(axis=1) * self._shadow_scale
                ours = self.makespan_batch.double().cpu().numpy()
                assert np.allclose(ours, chat, rtol=1e-6, atol=1e-4), \
                    'certified estimate differs from PACT: {0} vs {1}'.format(ours.tolist(), chat.tolist())
        if self.reward_mode == 'certified':
            assert bool((self.reward_batch <= 1e-4).all()), 'certified estimate decreased (reward > 0)'
        if bool(self.done_batch.any()):
            d = self.done_batch
            realized = self.op_end.max(dim=1)[0]
            assert torch.allclose(self.makespan_batch[d].double().cpu(), realized[d].cpu(), atol=1e-4), \
                'estimate not exact at termination'

    # ------------------------------------------------------------------------------------------------------
    # reset and validation
    # ------------------------------------------------------------------------------------------------------

    def reset(self):
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self._init_dynamic()
        return self.state

    def realized_makespan(self):
        return self.op_end.max(dim=1)[0]

    def schedule_arrays(self, b):
        '''(op_st, op_ct, op_mch) of instance b as numpy arrays, hours.'''
        n = int(self.nums_opes[b])
        return (self.op_start[b, :n].cpu().numpy(), self.op_end[b, :n].cpu().numpy(),
                self.op_mch[b, :n].cpu().numpy())

    def validate_family(self):
        '''Violation lists of every instance under its descriptor, by PACT's independent validator.'''
        import pact_bridge
        R = pact_bridge.pact().R
        out = []
        for b in range(self.batch_size):
            st, ct, mch = self.schedule_arrays(b)
            if (mch < 0).any():
                out.append(['unscheduled operations left'])
                continue
            out.append(R.validate_schedule(self.job_length_list[b], self.op_pt_list[b], self.descriptors[b],
                                           st, ct, mch))
        return out

    def validate_gantt(self):
        ok = all(len(v) == 0 for v in self.validate_family())
        return ok, self.schedules_batch
