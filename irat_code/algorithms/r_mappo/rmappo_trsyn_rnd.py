import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch.distributions import kl_divergence
from onpolicy.algorithms.utils.distributions import FixedCategorical, FixedNormal, FixedBernoulli
import copy


class RMappoTrSynRnd:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        # self.idv_policy = idv_policy
        self.policy = policy

        self.clip_param = args.clip_param

        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length

        self.value_loss_coef = args.value_loss_coef
        self.team_value_loss_coef = args.team_value_loss_coef

        self.entropy_coef = args.entropy_coef
        self.entropy_end_coef = args.entropy_end_coef
        self.entropy_anneal_gap = (self.entropy_coef - self.entropy_end_coef) / args.entropy_change_episode
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.adv_use_surgery = args.adv_use_surgery
        self.gradient_use_surgery = args.gradient_use_surgery
        self.idv_critic_ratio = args.idv_critic_ratio
        self.idv_critic_end_ratio = args.idv_critic_end_ratio
        self.idv_critic_gap = (self.idv_critic_ratio - self.idv_critic_end_ratio) / args.idv_critic_episode
        self.team_critic_ratio = args.team_critic_ratio
        self.team_critic_end_ratio = args.team_critic_end_ratio
        self.team_ciritic_gap = (self.team_critic_ratio - self.team_critic_end_ratio) / args.team_critic_episode

        self.ep_adv_surgery = args.ep_adv_surgery
        self.ep_adv_use_ratio = args.ep_adv_use_ratio

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.idv_value_normalizer = self.policy.idv_critic.v_out
        elif self._use_valuenorm:
            self.idv_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.idv_value_normalizer = None

        if self._use_popart:
            self.team_value_normalizer = self.policy.team_critic.v_out
        elif self._use_valuenorm:
            self.team_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.team_value_normalizer = None

    def update_entropy_coef(self):
        tmp = self.entropy_coef - self.entropy_anneal_gap
        if self.entropy_anneal_gap >= 0:
            self.entropy_coef = max(self.entropy_end_coef, tmp)
        else:
            self.entropy_coef = min(self.entropy_end_coef, tmp)

    def update_critic_ratio(self):
        tmp = self.idv_critic_ratio - self.idv_critic_gap
        if self.idv_critic_gap >= 0:
            self.idv_critic_ratio = max(self.idv_critic_end_ratio, tmp)
        else:
            self.idv_critic_ratio = min(self.idv_critic_end_ratio, tmp)

        tmp = self.team_critic_ratio - self.team_ciritic_gap
        if self.team_ciritic_gap >= 0:
            self.team_critic_ratio = max(self.team_critic_end_ratio, tmp)
        else:
            self.team_critic_ratio = min(self.team_critic_end_ratio, tmp)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, value_normalizer):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, episode, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        idv_share_obs_batch, team_share_obs_batch, obs_batch, \
        idv_rnn_states_batch, team_rnn_states_batch, idv_rnn_states_critic_batch, team_rnn_states_critic_batch,\
        actions_batch, idv_act_dists_batch, team_act_dists_batch, \
        idv_value_preds_batch, team_value_preds_batch, idv_return_batch, team_return_batch,\
        masks_batch, active_masks_batch, idv_action_log_probs_batch, team_action_log_probs_batch, \
        idv_adv_targ, team_adv_targ, available_actions_batch = sample

        idv_action_log_probs_batch = check(idv_action_log_probs_batch).to(**self.tpdv).detach()
        team_action_log_probs_batch = check(team_action_log_probs_batch).to(**self.tpdv).detach()

        idv_adv_targ = check(idv_adv_targ).to(**self.tpdv)
        team_adv_targ = check(team_adv_targ).to(**self.tpdv)

        idv_value_preds_batch = check(idv_value_preds_batch).to(**self.tpdv)
        team_value_preds_batch = check(team_value_preds_batch).to(**self.tpdv)

        idv_return_batch = check(idv_return_batch).to(**self.tpdv)
        team_return_batch = check(team_return_batch).to(**self.tpdv)

        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        idv_new_values, team_new_values, idv_new_action_log_probs, idv_new_dist_entropy, idv_new_act_dists = \
            self.policy.evaluate_actions(idv_share_obs_batch,
                                         team_share_obs_batch,
                                         obs_batch,
                                         idv_rnn_states_batch,
                                         idv_rnn_states_critic_batch,
                                         team_rnn_states_critic_batch,
                                         actions_batch,
                                         masks_batch,
                                         available_actions_batch,
                                         active_masks_batch)

        # individual actor update
        imp_weights = torch.exp(idv_new_action_log_probs - idv_action_log_probs_batch)

        adv_targ = self.idv_critic_ratio * idv_adv_targ + self.team_critic_ratio * team_adv_targ
        # print(idv_adv_targ.shape, team_adv_targ.shape)
        adv_scale = (idv_adv_targ * team_adv_targ).sum()
        idv_adv_scale = adv_scale / (team_adv_targ ** 2).sum()
        team_adv_scale = adv_scale / (idv_adv_targ ** 2).sum()
        # print((team_adv_targ ** 2).sum() == (idv_adv_targ ** 2).sum())
        # print(idv_adv_scale.item(), team_adv_scale.item())
        idv_adv_surgery = idv_adv_targ - idv_adv_scale * team_adv_targ
        team_adv_surgery = team_adv_targ - team_adv_scale * idv_adv_targ
        if self.adv_use_surgery and adv_scale < 0:
            idv_adv_targ, team_adv_targ = idv_adv_surgery, team_adv_surgery
            adv_targ = self.idv_critic_ratio * idv_adv_surgery + self.team_critic_ratio * team_adv_surgery

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        # clp = torch.clamp(so_weights, 1.0 - self.idv_clip_ratio, 1.0 + self.idv_clip_ratio)
        # surr3 = clp * idv_adv_targ
        # print(so_weights.shape, imp_weights.shape, idv_adv_targ.shape)
        # print(surr1.shape, surr2.shape, surr3.shape)

        tn = imp_weights.clone().view(-1, 1).shape[0]
        tg = imp_weights >= (1.0 - self.clip_param)
        tl = imp_weights <= (1.0 + self.clip_param)
        tcr = (tl & tg).float().sum() / tn
        ts = surr1 <= surr2
        tsr = ts.float().sum() / tn
        tsl = surr1.clone()[ts].mean()

        idv_min = torch.min(surr1, surr2)

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(idv_min,
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(idv_min, dim=-1, keepdim=True).mean()
        idv_ppo_abs = torch.sum(torch.abs(idv_min), dim=-1, keepdim=True).mean().detach()

        idv_surr1 = imp_weights * idv_adv_targ
        idv_surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * idv_adv_targ

        team_surr1 = imp_weights * team_adv_targ
        team_surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * team_adv_targ
        if self._use_policy_active_masks:
            idv_cur_loss = (-torch.sum(torch.min(idv_surr1, idv_surr2),
                                       dim=-1,
                                       keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
            team_cur_loss = (-torch.sum(torch.min(team_surr1, team_surr2),
                                        dim=-1,
                                        keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            idv_cur_loss = -torch.sum(torch.min(idv_surr1, idv_surr2), dim=-1, keepdim=True).mean()
            team_cur_loss = -torch.sum(torch.min(team_surr1, team_surr2), dim=-1, keepdim=True).mean()
        idv_cur_loss = self.idv_critic_ratio * idv_cur_loss
        team_cur_loss = self.team_critic_ratio * team_cur_loss

        idv_policy_loss = policy_action_loss
        idv_loss = idv_policy_loss - idv_new_dist_entropy * self.entropy_coef
        idv_entropy_abs = idv_new_dist_entropy.clone().detach() * self.entropy_coef
        loss_abs = idv_ppo_abs + idv_entropy_abs + 1e-7
        policy_loss_prop = idv_ppo_abs / loss_abs
        entropy_prop = idv_entropy_abs / loss_abs

        # individual critic loss
        idv_value_loss = self.cal_value_loss(idv_new_values, idv_value_preds_batch, idv_return_batch,
                                             active_masks_batch, self.idv_value_normalizer)

        # team critic loss
        team_value_loss = self.cal_value_loss(team_new_values, team_value_preds_batch, team_return_batch,
                                              active_masks_batch, self.team_value_normalizer)

        # update actor
        if self.gradient_use_surgery:
            actor_params = list(self.policy.actor.parameters())
            self.policy.actor_optimizer.zero_grad()
            idv_cur_loss.backward(retain_graph=True)
            idv_grads = [copy.deepcopy(p.grad) for p in actor_params if p.grad is not None]

            self.policy.actor_optimizer.zero_grad()
            team_cur_loss.backward(retain_graph=True)
            team_grads = [copy.deepcopy(p.grad) for p in actor_params if p.grad is not None]

            # print(self.policy.actor.state_dict())
            # for gi in range(len(idv_grads)):
            #     print(actor_params[gi])
            #     print(idv_grads[gi])
            idv_grads_vec = torch.cat([copy.deepcopy(g).reshape(-1) for g in idv_grads])
            team_grads_vec = torch.cat([copy.deepcopy(g).reshape(-1) for g in team_grads])
            grad_scale = (team_grads_vec * idv_grads_vec).sum()
            team_scale = grad_scale / (team_grads_vec ** 2).sum()
            idv_scale = grad_scale / (idv_grads_vec ** 2).sum()
            if grad_scale < 0:
                for gi in range(len(idv_grads)):
                    idv_gd = copy.deepcopy(idv_grads[gi])
                    team_gd = copy.deepcopy(team_grads[gi])
                    team_grads[gi] -= idv_scale * idv_gd
                    idv_grads[gi] -= team_scale * team_gd

            self.policy.actor_optimizer.zero_grad()
            (-self.entropy_coef * idv_new_dist_entropy).backward()
            gi = 0
            for pi in range(len(actor_params)):
                if actor_params[pi].grad is not None:
                    actor_params[pi].grad += idv_grads[gi] + team_grads[gi]
                    gi += 1
            if self._use_max_grad_norm:
                idv_actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            else:
                idv_actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            self.policy.actor_optimizer.step()

        else:
            self.policy.actor_optimizer.zero_grad()
            if update_actor:
                idv_loss.backward()
            if self._use_max_grad_norm:
                idv_actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
            else:
                idv_actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
            self.policy.actor_optimizer.step()

        # update individual critic
        self.policy.idv_critic_optimizer.zero_grad()
        (idv_value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            idv_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.idv_critic.parameters(), self.max_grad_norm)
        else:
            idv_critic_grad_norm = get_gard_norm(self.policy.idv_critic.parameters())
        self.policy.idv_critic_optimizer.step()

        # update team critic
        self.policy.team_critic_optimizer.zero_grad()
        (team_value_loss * self.team_value_loss_coef).backward()
        if self._use_max_grad_norm:
            team_critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.team_critic.parameters(), self.max_grad_norm)
        else:
            team_critic_grad_norm = get_gard_norm(self.policy.team_critic.parameters())
        self.policy.team_critic_optimizer.step()

        return idv_value_loss, team_value_loss, idv_critic_grad_norm, team_critic_grad_norm,\
               idv_policy_loss, idv_new_dist_entropy, idv_actor_grad_norm, \
               imp_weights, tcr, tsr, tsl, idv_loss, idv_ppo_abs, policy_loss_prop, entropy_prop, adv_targ, \
               idv_adv_scale, idv_adv_surgery, team_adv_scale, team_adv_surgery

    def train(self, buffer, episode, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            idv_advantages = buffer.idv_returns[:-1] - self.idv_value_normalizer.denormalize(buffer.idv_value_preds[:-1])
        else:
            idv_advantages = buffer.idv_returns[:-1] - buffer.idv_value_preds[:-1]
        idv_advantages_copy = idv_advantages.copy()
        idv_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        idv_mean_advantages = np.nanmean(idv_advantages_copy)
        idv_std_advantages = np.nanstd(idv_advantages_copy)
        idv_advantages = (idv_advantages - idv_mean_advantages) / (idv_std_advantages + 1e-5)

        if self._use_popart or self._use_valuenorm:
            team_advantages = buffer.team_returns[:-1] - self.team_value_normalizer.denormalize(buffer.team_value_preds[:-1])
        else:
            team_advantages = buffer.team_returns[:-1] - buffer.team_value_preds[:-1]
        team_advantages_copy = team_advantages.copy()
        team_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        team_mean_advantages = np.nanmean(team_advantages_copy)
        team_std_advantages = np.nanstd(team_advantages_copy)
        team_advantages = (team_advantages - team_mean_advantages) / (team_std_advantages + 1e-5)
        # print(idv_advantages.shape, team_advantages.shape)        # episode_length, n_threads, n_agents, 1

        if self.ep_adv_surgery:
            tmp_idv_adv, tmp_team_adv = [], []
            n_threads, n_agents = idv_advantages.shape[1:3]
            for i in range(n_threads):
                tia, tta = [], []
                for j in range(n_agents):
                    idv_adv_targ, team_adv_targ = copy.deepcopy(idv_advantages[:, i, j, :]), \
                                                  copy.deepcopy(team_advantages[:, i, j, :])
                    adv_scale = (idv_adv_targ * team_adv_targ).sum()
                    idv_adv_scale = adv_scale / (team_adv_targ ** 2).sum()
                    team_adv_scale = adv_scale / (idv_adv_targ ** 2).sum()
                    idv_ratio, team_ratio = 1.0, 1.0
                    if self.ep_adv_use_ratio:
                        idv_ratio, team_ratio = self.idv_critic_ratio, self.team_critic_ratio
                    idv_adv_surgery = idv_adv_targ - idv_ratio * idv_adv_scale * team_adv_targ
                    team_adv_surgery = team_adv_targ - team_ratio * team_adv_scale * idv_adv_targ
                    if adv_scale < 0:
                        idv_adv_targ, team_adv_targ = idv_adv_surgery, team_adv_surgery
                    # print(idv_adv_targ)
                    # idv_adv_targ = torch.tensor(idv_adv_targ).to(**self.tpdv)
                    # team_adv_targ = torch.tensor(team_adv_targ).to(**self.tpdv)
                    tia.append(idv_adv_targ)
                    tta.append(team_adv_targ)
                tia = np.array(tia).transpose(1, 0, 2)
                tta = np.array(tta).transpose(1, 0, 2)
                tmp_idv_adv.append(tia)
                tmp_team_adv.append(tta)
            idv_advantages = np.array(tmp_idv_adv).transpose(1, 0, 2, 3)
            team_advantages = np.array(tmp_team_adv).transpose(1, 0, 2, 3)
            # print(idv_advantages.shape, team_advantages.shape)

        train_info = {}

        train_info['Aa_idv_actor_loss'] = 0

        train_info['Ab_policy_loss'] = 0
        train_info['Ac_idv_ppo_loss_abs'] = 0
        train_info['Ad_idv_ppo_prop'] = 0

        train_info['Ae_eta'] = 0
        train_info['Af_noclip_proportion'] = 0
        train_info['Ag_update_proportion'] = 0
        train_info['Ah_update_loss'] = 0

        train_info['Ao_idv_entropy_prop'] = 0
        train_info['Ap_dist_entropy'] = 0

        train_info['Au_value_loss'] = 0
        train_info['Av_advantages'] = 0
        train_info['Aw_idv_actor_norm'] = 0
        train_info['Ax_idv_critic_norm'] = 0

        train_info['Va_update_advantages'] = 0
        train_info['Vb_idv_adv_scale'] = 0
        train_info['Vc_idv_adv_surgery'] = 0
        train_info['Vd_team_adv_scale'] = 0
        train_info['Ve_team_adv_surgery'] = 0

        train_info['Tq_team_value_loss'] = 0
        train_info['Tr_team_advantages'] = 0
        train_info['Tt_team_critic_norm'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(idv_advantages,
                                                            team_advantages,
                                                            self.num_mini_batch,
                                                            self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(idv_advantages,
                                                                  team_advantages,
                                                                  self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(idv_advantages,
                                                               team_advantages,
                                                               self.num_mini_batch)

            for sample in data_generator:
                idv_value_loss, team_value_loss, idv_critic_grad_norm, team_critic_grad_norm, \
                idv_policy_loss, idv_new_dist_entropy, idv_actor_grad_norm, \
                imp_weights, tcr, tsr, tsl, idv_loss, idv_ppo_abs, policy_loss_prop, entropy_prop, adv_targ, \
                idv_adv_scale, idv_adv_surgery, team_adv_scale, team_adv_surgery \
                    = self.ppo_update(sample, episode, update_actor)

                train_info['Aa_idv_actor_loss'] += idv_loss.item()
                train_info['Ab_policy_loss'] += idv_policy_loss.item()
                train_info['Ac_idv_ppo_loss_abs'] += idv_ppo_abs.item()
                train_info['Ad_idv_ppo_prop'] += policy_loss_prop.item()
                train_info['Ae_eta'] += imp_weights.mean()
                train_info['Af_noclip_proportion'] += tcr
                train_info['Ag_update_proportion'] += tsr
                train_info['Ah_update_loss'] += tsl
                # train_info['Aj_idv_sigma'] += so_weights.mean()
                # train_info['Ak_idv_clip(sigma, 1-epislon\', 1+epislon\')'] += clp.mean()
                # train_info['Al_idv_noclip_proportion'] += tcrs
                # train_info['Am_idv_(sigma*A)update_proportion'] += ts3
                # train_info['An_idv_(sigma*A)update_loss'] += tsl3
                train_info['Ao_idv_entropy_prop'] += entropy_prop.item()
                train_info['Ap_dist_entropy'] += idv_new_dist_entropy.item()
                # train_info['Aq_idv_kl_prop'] += idv_kl_prop.item()
                # train_info['As_idv_kl_loss'] += idv_kl_loss.item()
                # train_info['At_idv_cross_entropy'] += idv_cross_entropy.item()
                train_info['Au_value_loss'] += idv_value_loss.item()
                train_info['Aw_idv_actor_norm'] += idv_actor_grad_norm
                train_info['Ax_idv_critic_norm'] += idv_critic_grad_norm

                train_info['Va_update_advantages'] += adv_targ.mean()
                train_info['Vb_idv_adv_scale'] += idv_adv_scale.item()
                train_info['Vc_idv_adv_surgery'] += idv_adv_surgery.mean()
                train_info['Vd_team_adv_scale'] += team_adv_scale.item()
                train_info['Ve_team_adv_surgery'] += team_adv_surgery.mean()

                train_info['Tq_team_value_loss'] += team_value_loss.item()
                train_info['Tt_team_critic_norm'] += team_critic_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info['Av_advantages'] = np.nanmean(idv_advantages)

        train_info['Tr_team_advantages'] = np.nanmean(team_advantages)

        self.update_entropy_coef()
        self.update_critic_ratio()

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.idv_critic.train()
        self.policy.team_critic.train()

    # def team_prep_training(self):
    #     self.team_policy.actor.train()
    #     self.team_policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.idv_critic.eval()
        self.policy.team_critic.eval()

    # def team_prep_rollout(self):
    #     self.team_policy.actor.eval()
    #     self.team_policy.critic.eval()
