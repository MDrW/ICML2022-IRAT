import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch.distributions import kl_divergence
from onpolicy.algorithms.utils.distributions import FixedCategorical, FixedNormal, FixedBernoulli
# torch.autograd.set_detect_anomaly(True)


class RMappoTrSyn:
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 idv_policy,
                 team_policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.idv_policy = idv_policy
        self.team_policy = team_policy

        self.clip_param = args.clip_param

        self.idv_clip_ratio = args.idv_clip_ratio
        self.idv_end_clip_ratio = args.idv_end_clip_ratio
        self.idv_clip_gap = (self.idv_clip_ratio - self.idv_end_clip_ratio) / args.idv_clip_episodes

        self.idv_kl_coef = args.idv_kl_coef
        self.idv_kl_end_coef = args.idv_kl_end_coef
        self.idv_kl_anneal_gap = (self.idv_kl_coef - self.idv_kl_end_coef) / args.idv_kl_episodes

        self.team_clip_ratio = args.team_clip_ratio
        self.team_end_clip_ratio = args.team_end_clip_ratio
        self.team_clip_gap = (self.team_clip_ratio - self.team_end_clip_ratio) / args.team_clip_episodes

        self.team_kl_coef = args.team_kl_coef
        self.team_kl_end_coef = args.team_kl_end_coef
        self.team_kl_anneal_gap = (self.team_kl_coef - self.team_kl_end_coef) / args.team_kl_episodes

        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
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

        self.idv_use_two_clip = args.idv_use_two_clip
        self.idv_use_kl_loss = args.idv_use_kl_loss

        self.team_use_clip = args.team_use_clip
        self.team_use_kl_loss = args.team_use_kl_loss

        self.idv_kl_loss_use_present = args.idv_kl_loss_use_present
        self.team_kl_loss_use_present = args.team_kl_loss_use_present
        self.idv_clip_use_present = args.idv_clip_use_present
        self.team_clip_use_present = args.team_clip_use_present

        self.idv_use_cross_entropy = args.idv_use_cross_entropy
        self.team_use_cross_entropy = args.team_use_cross_entropy

        self.change_reward = args.change_reward
        self.change_reward_episode = args.change_reward_episode
        self.change_use_policy = args.change_use_policy

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.idv_value_normalizer = self.idv_policy.critic.v_out
        elif self._use_valuenorm:
            self.idv_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.idv_value_normalizer = None

        if self._use_popart:
            self.team_value_normalizer = self.team_policy.critic.v_out
        elif self._use_valuenorm:
            self.team_value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.team_value_normalizer = None

    def update_ratio_by_gap(self, ratio, gap, end_ratio):
        tmp = ratio - gap
        if gap >= 0:
            ratio = max(tmp, end_ratio)
        else:
            ratio = min(tmp, end_ratio)
        return ratio

    def update_idv_clip_ratio(self):
        self.idv_clip_ratio = self.update_ratio_by_gap(self.idv_clip_ratio,
                                                       self.idv_clip_gap, self.idv_end_clip_ratio)
        # tmp = self.idv_clip_ratio - self.idv_clip_gap
        # self.idv_clip_ratio = max(self.idv_end_clip_ratio, tmp)

    def update_team_clip_ratio(self):
        self.team_clip_ratio = self.update_ratio_by_gap(self.team_clip_ratio,
                                                        self.team_clip_gap, self.team_end_clip_ratio)
        # tmp = self.team_clip_ratio - self.team_clip_gap
        # self.team_clip_ratio = min(self.team_end_clip_ratio, tmp)

    def update_idv_kl_coef(self):
        self.idv_kl_coef = self.update_ratio_by_gap(self.idv_kl_coef,
                                                    self.idv_kl_anneal_gap, self.idv_kl_end_coef)
        # tmp = self.idv_kl_coef - self.idv_kl_anneal_gap
        # self.idv_kl_coef = min(self.idv_kl_end_coef, tmp)

    def update_team_kl_coef(self):
        self.team_kl_coef = self.update_ratio_by_gap(self.team_kl_coef,
                                                     self.team_kl_anneal_gap, self.team_kl_end_coef)
        # tmp = self.team_kl_coef - self.team_kl_anneal_gap
        # self.team_kl_coef = max(self.team_kl_end_coef, tmp)

    def update_entropy_coef(self):
        self.entropy_coef = self.update_ratio_by_gap(self.entropy_coef,
                                                     self.entropy_anneal_gap, self.entropy_end_coef)
        # tmp = self.entropy_coef - self.entropy_anneal_gap
        # if self.entropy_anneal_gap >= 0:
        #     self.entropy_coef = max(self.entropy_end_coef, tmp)
        # else:
        #     self.entropy_coef = min(self.entropy_end_coef, tmp)

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

        idv_action_log_probs_batch[idv_action_log_probs_batch < -20.] = -20.
        team_action_log_probs_batch[team_action_log_probs_batch < -20.] = -20.

        # Reshape to do in a single forward pass for all steps
        idv_new_values, idv_new_action_log_probs, idv_new_dist_entropy, idv_new_act_dists = \
            self.idv_policy.evaluate_actions(idv_share_obs_batch,
                                             obs_batch,
                                             idv_rnn_states_batch,
                                             idv_rnn_states_critic_batch,
                                             actions_batch,
                                             masks_batch,
                                             available_actions_batch,
                                             active_masks_batch)

        team_new_values, team_new_action_log_probs, team_new_dist_entropy, team_new_act_dists = \
            self.team_policy.evaluate_actions(team_share_obs_batch,
                                              obs_batch,
                                              team_rnn_states_batch,
                                              team_rnn_states_critic_batch,
                                              actions_batch,
                                              masks_batch,
                                              available_actions_batch,
                                              active_masks_batch)

        # print(idv_act_dists_batch.shape, len(idv_new_act_dists))
        # print(idv_new_act_dists)
        # print(idv_act_dists_batch)
        idv_kl_loss = 0
        idv_cross_entropy = torch.zeros(1)
        team_entropy = 0
        for ai in range(len(team_new_act_dists)):
            if type(idv_new_act_dists[ai]) == FixedCategorical:
                if self.idv_kl_loss_use_present:
                    tmp_probs = team_new_act_dists[ai].probs.clone().detach()
                else:
                    tmp_other_dists = team_act_dists_batch[:, ai]
                    tmp_probs = []
                    for tp in tmp_other_dists:
                        tmp_probs.append(tp.probs.clone().detach())
                    # print(tmp_probs)
                    tmp_probs = torch.stack(tmp_probs).to(**self.tpdv)
                other_dists = type(idv_new_act_dists[ai])(probs=tmp_probs)
            elif type(idv_new_act_dists[ai]) == FixedNormal:
                if self.idv_kl_loss_use_present:
                    tmp_mu = team_new_act_dists[ai].loc.clone().detach()
                    tmp_sigma = team_new_act_dists[ai].scale.clone().detach()
                else:
                    tmp_other_dists = team_act_dists_batch[:, ai]
                    tmp_mu, tmp_sigma = [], []
                    for tp in tmp_other_dists:
                        tmp_mu.append(tp.loc.clone().detach())
                        tmp_sigma.append(tp.scale.clone().detach())
                    tmp_mu = torch.stack(tmp_mu).to(**self.tpdv)
                    tmp_sigma = torch.stack(tmp_sigma).to(**self.tpdv)
                other_dists = type(idv_new_act_dists[ai])(loc=tmp_mu, scale=tmp_sigma)
                # print(ai, other_dists)
            else:
                raise NotImplementedError
            idv_kl_loss += kl_divergence(other_dists, idv_new_act_dists[ai]).mean()
            if self.idv_use_cross_entropy:
                idv_cross_entropy -= (other_dists.probs * torch.log(idv_new_act_dists[ai].probs)).sum(dim=1).mean()
            team_entropy += other_dists.entropy().mean()

        team_kl_loss = 0
        team_cross_entropy = torch.zeros(1)
        idv_entropy = 0
        for ai in range(len(idv_new_act_dists)):
            if type(team_new_act_dists[ai]) == FixedCategorical:
                if self.team_kl_loss_use_present:
                    tmp_probs = idv_new_act_dists[ai].probs.clone().detach()
                else:
                    tmp_other_dists = idv_act_dists_batch[:, ai]
                    tmp_probs = []
                    for tp in tmp_other_dists:
                        tmp_probs.append(tp.probs.clone().detach())
                    # print(tmp_probs)
                    tmp_probs = torch.stack(tmp_probs).to(**self.tpdv)
                other_dists = type(team_new_act_dists[ai])(probs=tmp_probs)
            elif type(team_new_act_dists[ai]) == FixedNormal:
                if self.team_kl_loss_use_present:
                    tmp_mu = idv_new_act_dists[ai].loc.clone().detach()
                    tmp_sigma = idv_new_act_dists[ai].scale.clone().detach()
                else:
                    tmp_other_dists = idv_act_dists_batch[:, ai]
                    tmp_mu, tmp_sigma = [], []
                    for tp in tmp_other_dists:
                        tmp_mu.append(tp.loc.clone().detach())
                        tmp_sigma.append(tp.scale.clone().detach())
                    tmp_mu = torch.stack(tmp_mu).to(**self.tpdv)
                    tmp_sigma = torch.stack(tmp_sigma).to(**self.tpdv)
                other_dists = type(team_new_act_dists[ai])(loc=tmp_mu, scale=tmp_sigma)
            else:
                raise NotImplementedError
            team_kl_loss += kl_divergence(other_dists, team_new_act_dists[ai]).mean()
            if self.team_use_cross_entropy:
                team_cross_entropy -= (other_dists.probs * torch.log(team_new_act_dists[ai].probs)).sum(dim=1).mean()
            idv_entropy += other_dists.entropy().mean()

        # print(idv_cross_entropy, team_entropy, idv_kl_loss,
        # (idv_cross_entropy - team_entropy).item(), team_new_dist_entropy)
        # print(team_cross_entropy, idv_entropy, team_kl_loss,
        # (team_cross_entropy - idv_entropy).item(), idv_new_dist_entropy)
        # print(idv_entropy.item(), idv_new_dist_entropy.item(), team_entropy.item(), team_new_dist_entropy.item())

        # if self.team_kl_loss_use_present:
        #     assert abs(idv_entropy.item() - idv_new_dist_entropy.item()) < 0.001, \
        #         str(idv_entropy.item()) + ", " + str(idv_new_dist_entropy.item())
        # if self.idv_kl_loss_use_present:
        #     assert abs(team_entropy.item() - team_new_dist_entropy.item()) < 0.001, \
        #         str(team_entropy.item()) + ", " + str(team_new_dist_entropy.item())
        #
        # assert abs((idv_cross_entropy - team_entropy).item() - idv_kl_loss.item()) < 0.001, \
        #     str((idv_cross_entropy - team_entropy).item()) + ", " + str(idv_kl_loss.item())
        # assert abs((team_cross_entropy - idv_entropy).item() - team_kl_loss.item()) < 0.001, \
        #     str((team_cross_entropy - idv_entropy).item()) + ", " + str(team_kl_loss.item())

        # individual actor update
        imp_weights = torch.exp(idv_new_action_log_probs - idv_action_log_probs_batch)

        # individual actions probs / team actions probs
        if self.idv_clip_use_present:
            so_weights = torch.exp(idv_new_action_log_probs - team_new_action_log_probs.clone().detach())
        else:
            so_weights = torch.exp(idv_new_action_log_probs - team_action_log_probs_batch)
        # print("idv_action_log_probs", torch.sum(idv_new_action_log_probs))
        # print(idv_new_action_log_probs)
        # print("team_action_log_probs", torch.sum(team_action_log_probs_batch))
        # print(team_action_log_probs_batch)
        # print(idv_new_action_log_probs.shape, idv_action_log_probs_batch.shape, team_action_log_probs_batch.shape)
        # if torch.isnan(imp_weights).any():
        #     print("imp_weights has nan")
        #     print(imp_weights)
        #     print("----------------------------------")
        #     print(idv_share_obs_batch)
        #     print(torch.isnan(torch.tensor(idv_share_obs_batch)).any())
        #     print("----------------------------------")
        #     print(obs_batch)
        #     print(torch.isnan(torch.tensor(obs_batch)).any())
        #     print("-----------------------------------")
        #     print(idv_rnn_states_batch)
        #     print(torch.isnan(torch.tensor(idv_rnn_states_batch)).any())
        #     print("----------------------------------")
        #     print(idv_rnn_states_critic_batch)
        #     print(torch.isnan(torch.tensor(idv_rnn_states_critic_batch)).any())
        #     print("----------------------------------")
        #     print(actions_batch)
        #     print(torch.isnan(torch.tensor(actions_batch)).any())
        #     print("----------------------------------")
        #     print(self.idv_policy.actor.parameters)
        #     print(torch.isnan(torch.tensor(self.idv_policy.actor.parameters)))
        #     print("----------------------------------")
        #     print(self.team_policy.actor.parameters)
        #     print(torch.isnan(torch.tensor(self.team_policy.actor.parameters)))
        #     print("-----------------------------------")
        #     imp_weights[torch.isnan(imp_weights)] = 1.0 + self.clip_param
        # if torch.isnan(imp_weights).any():
        #     print("imp_weights has nan")
        #     print(imp_weights)
        # if torch.isnan(so_weights).any():
        #     print("so_weights has nan")
        #     print(so_weights)
        #     so_weights[torch.isnan(so_weights)] = 1.0 + self.idv_clip_ratio

        surr1 = imp_weights * idv_adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * idv_adv_targ
        clp = torch.clamp(so_weights, 1.0 - self.idv_clip_ratio, 1.0 + self.idv_clip_ratio)
        # clp = so_weights.clamp(min=1.0 - self.idv_clip_ratio, max=1.0 + self.idv_clip_ratio)
        # clp = so_weights.clone()
        # clp[clp < (1.0 - self.idv_clip_ratio)] = 1.0 - self.idv_clip_ratio
        # clp[clp > (1.0 + self.idv_clip_ratio)] = 1.0 + self.idv_clip_ratio
        surr3 = clp * idv_adv_targ
        # print(so_weights.shape, imp_weights.shape, idv_adv_targ.shape)
        # print(surr1.shape, surr2.shape, surr3.shape, clp.shape)
        # print(idv_new_action_log_probs.shape, idv_action_log_probs_batch.shape)
        # if torch.isnan(idv_adv_targ).any():
        #     print("idv_adv_targ has nan")
        #     print(idv_adv_targ)
        # if torch.isnan(surr1).any():
        #     print("surr1 has nan")
        #     print(surr1)
        # if torch.isnan(surr2).any():
        #     print("surr2 has nan")
        #     print(surr2)
        # if torch.isnan(surr3).any():
        #     print("surr3 has nan")
        # print("so_weights", torch.isnan(so_weights).any(), torch.sum(so_weights))
        # print(so_weights)
        # print("imp_weights", torch.isnan(imp_weights).any(), torch.sum(imp_weights))
        # print(imp_weights)
        # print("surr2", torch.isnan(surr2).any(), torch.sum(surr2))
        # print(surr2)
        # print("surr3", torch.isnan(surr3).any(), torch.sum(surr3))
        # print(surr3)

        tn = imp_weights.clone().detach().view(-1, 1).shape[0]
        tg = imp_weights >= (1.0 - self.clip_param)
        tl = imp_weights <= (1.0 + self.clip_param)
        tcr = (tl & tg).float().sum() / tn
        tgs = so_weights >= (1.0 - self.idv_clip_ratio)
        tls = so_weights <= (1.0 + self.idv_clip_ratio)
        tcrs = (tgs & tls).float().sum() / tn

        ts12 = surr1 <= surr2
        ts13 = surr1 <= surr3
        ts1 = (ts12 & ts13).float().sum() / tn
        tsl1 = surr1.clone().detach()[ts12 & ts13].mean()
        ts31 = surr3 <= surr1
        ts32 = surr3 <= surr2
        t3 = ts31 & ts32 & tgs & tls
        ts3 = t3.float().sum() / tn
        tsl3 = surr3.clone().detach()[t3].mean()

        idv_min = torch.min(surr1, surr2)
        if self.idv_use_two_clip:
            idv_min = torch.min(idv_min, surr3)

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(idv_min,
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(idv_min, dim=-1, keepdim=True).mean()
        idv_ppo_abs = torch.sum(torch.abs(idv_min.clone().detach()), dim=-1, keepdim=True).mean().clone().detach()

        idv_policy_loss = policy_action_loss
        idv_loss = idv_policy_loss - idv_new_dist_entropy * self.entropy_coef
        idv_entropy_abs = idv_new_dist_entropy.clone().detach() * self.entropy_coef
        if self.idv_use_kl_loss:
            idv_loss += self.idv_kl_coef * idv_kl_loss
            idv_kl_abs = self.idv_kl_coef * idv_kl_loss.clone().detach()
        elif self.idv_use_cross_entropy:
            idv_loss += self.idv_kl_coef * idv_cross_entropy
            idv_kl_abs = self.idv_kl_coef * idv_cross_entropy.clone().detach()
        else:
            idv_kl_abs = torch.zeros(1)

        # print("Clip", torch.sum(clp), self.idv_clip_ratio)
        # print(clp)

        # idv_policy_loss_prop = idv_policy_loss / idv_loss
        # idv_entropy_prop = idv_new_dist_entropy * self.entropy_coef / idv_loss
        # if self.idv_use_kl_loss:
        #     idv_klce_prop = self.idv_kl_coef * idv_kl_loss / idv_loss
        # elif self.idv_use_cross_entropy:
        #     idv_klce_prop = self.idv_kl_coef * idv_cross_entropy / idv_loss
        # else:
        #     idv_klce_prop = torch.zeros(1)

        # individual critic loss
        idv_value_loss = self.cal_value_loss(idv_new_values, idv_value_preds_batch, idv_return_batch,
                                             active_masks_batch, self.idv_value_normalizer)

        if self.change_reward and episode > self.change_reward_episode:
            if self.change_use_policy == "team":
                idv_loss = torch.zeros(1, requires_grad=True)
                idv_value_loss = torch.zeros(1, requires_grad=True)
                idv_ppo_abs = torch.zeros(1)
                idv_entropy_abs = torch.zeros(1)
                idv_kl_abs = torch.zeros(1)
            else:
                idv_min = torch.min(surr1, surr2)
                if self._use_policy_active_masks:
                    policy_action_loss = (-torch.sum(idv_min,
                                                     dim=-1,
                                                     keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    policy_action_loss = -torch.sum(idv_min, dim=-1, keepdim=True).mean()
                idv_policy_loss = policy_action_loss
                idv_loss = idv_policy_loss - idv_new_dist_entropy * self.entropy_coef
                idv_ppo_abs = torch.sum(torch.abs(idv_min.clone().detach()), dim=-1, keepdim=True).mean().clone().detach()
                idv_kl_abs = torch.zeros(1)
        else:
            pass

        idv_loss_abs = idv_ppo_abs + idv_entropy_abs + idv_kl_abs + 1e-7
        idv_ppo_prop = idv_ppo_abs / idv_loss_abs
        idv_entropy_prop = idv_entropy_abs / idv_loss_abs
        idv_kl_prop = idv_kl_abs / idv_loss_abs

        # update individual actor
        if torch.isnan(idv_loss).any():
            print("idv loss has nan")
        if torch.isinf(idv_loss).any():
            print("idv loss has inf")
        if torch.isnan(so_weights).any():
            print("so_weights has nan")
        if torch.isinf(so_weights).any():
            print("so_weights has inf")
        if torch.isnan(imp_weights).any():
            print("imp_weights has nan")
        if torch.isinf(imp_weights).any():
            print("imp_weights has inf")
        self.idv_policy.actor_optimizer.zero_grad()
        if update_actor:
            # with torch.autograd.detect_anomaly():
            idv_loss.backward()
            # if torch.isinf(so_weights).any():
            #     for p in list(self.idv_policy.actor.parameters()):
            #         print(p.grad)
        if self._use_max_grad_norm:
            idv_actor_grad_norm = nn.utils.clip_grad_norm_(self.idv_policy.actor.parameters(), self.max_grad_norm)
        else:
            idv_actor_grad_norm = get_gard_norm(self.idv_policy.actor.parameters())
        self.idv_policy.actor_optimizer.step()

        # update individual critic
        self.idv_policy.critic_optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        (idv_value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            idv_critic_grad_norm = nn.utils.clip_grad_norm_(self.idv_policy.critic.parameters(), self.max_grad_norm)
        else:
            idv_critic_grad_norm = get_gard_norm(self.idv_policy.critic.parameters())
        self.idv_policy.critic_optimizer.step()

        # team actor update
        if self.team_clip_use_present:
            team_imp_weights = torch.exp(team_new_action_log_probs - idv_new_action_log_probs.clone().detach())
        else:
            team_imp_weights = torch.exp(team_new_action_log_probs - idv_action_log_probs_batch)

        # if torch.isnan(team_imp_weights).any():
        #     print("team_imp_weights has nan")
        #     team_imp_weights[torch.isnan(team_imp_weights)] = 1.0 + self.team_clip_ratio
        team_surr1 = team_imp_weights * team_adv_targ
        tclp = torch.clamp(team_imp_weights, 1.0 - self.team_clip_ratio, 1.0 + self.team_clip_ratio)
        team_surr2 = tclp * team_adv_targ

        ttn = team_imp_weights.clone().detach().view(-1, 1).shape[0]
        ttg = team_imp_weights >= (1.0 - self.team_clip_ratio)
        ttl = team_imp_weights <= (1.0 + self.team_clip_ratio)
        ttcr = (ttl & ttg).float().sum() / ttn
        tts = team_surr1 <= team_surr2
        tsr = tts.float().sum() / ttn
        tsl = team_surr1.clone().detach()[tts].mean()

        team_min = team_surr1
        if self.team_use_clip:
            team_min = torch.min(team_surr1, team_surr2)
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(team_min,
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(team_min, dim=-1, keepdim=True).mean()
        team_ppo_abs = torch.sum(torch.abs(team_min.clone().detach()), dim=-1, keepdim=True).mean().clone().detach()
        team_policy_loss = policy_action_loss
        team_loss = team_policy_loss - team_new_dist_entropy * self.entropy_coef
        team_entropy_abs = team_new_dist_entropy.clone().detach() * self.entropy_coef
        if self.team_use_kl_loss:
            team_loss += self.team_kl_coef * team_kl_loss
            team_kl_abs = self.team_kl_coef * team_kl_loss.clone().detach()
        elif self.team_use_cross_entropy:
            team_loss += self.team_kl_coef * team_cross_entropy
            team_kl_abs = self.team_kl_coef * team_cross_entropy.clone().detach()
        else:
            team_kl_abs = torch.zeros(1)

        team_value_loss = self.cal_value_loss(team_new_values, team_value_preds_batch, team_return_batch,
                                              active_masks_batch, self.team_value_normalizer)

        if self.change_reward and episode > self.change_reward_episode:
            if self.change_use_policy == "team":
                team_imp_weights = torch.exp(team_new_action_log_probs - team_action_log_probs_batch)
                team_surr1 = team_imp_weights * team_adv_targ
                team_surr2 = torch.clamp(team_imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * team_adv_targ
                team_min = torch.min(team_surr1, team_surr2)
                if self._use_policy_active_masks:
                    policy_action_loss = (-torch.sum(team_min,
                                                     dim=-1,
                                                     keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    policy_action_loss = -torch.sum(team_min, dim=-1, keepdim=True).mean()
                team_policy_loss = policy_action_loss
                team_loss = team_policy_loss - team_new_dist_entropy * self.entropy_coef
                team_ppo_abs = torch.sum(torch.abs(team_min.clone().detach()), dim=-1, keepdim=True).mean().clone().detach()
                team_kl_abs = torch.zeros(1)
            else:
                team_loss = torch.zeros(1, requires_grad=True)
                team_value_loss = torch.zeros(1, requires_grad=True)
                team_ppo_abs = torch.zeros(1)
                team_entropy_abs = torch.zeros(1)
                team_kl_abs = torch.zeros(1)
        else:
            pass

        team_loss_abs = team_ppo_abs + team_entropy_abs + team_kl_abs + 1e-7
        team_ppo_prop = team_ppo_abs / team_loss_abs
        team_entropy_prop = team_entropy_abs / team_loss_abs
        team_kl_prop = team_kl_abs / team_loss_abs

        # update team actor
        if torch.isnan(team_loss).any():
            print("team loss has nan")
        if torch.isinf(team_loss).any():
            print("team loss has inf")
        if torch.isnan(team_imp_weights).any():
            print("team has nan")
        if torch.isinf(team_imp_weights).any():
            print("team has inf")
        self.team_policy.actor_optimizer.zero_grad()
        if update_actor:
            # with torch.autograd.detect_anomaly():
            team_loss.backward()
        if self._use_max_grad_norm:
            team_actor_grad_norm = nn.utils.clip_grad_norm_(self.team_policy.actor.parameters(), self.max_grad_norm)
        else:
            team_actor_grad_norm = get_gard_norm(self.team_policy.actor.parameters())
        self.team_policy.actor_optimizer.step()

        # update team critic
        self.team_policy.critic_optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        (team_value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            team_critic_grad_norm = nn.utils.clip_grad_norm_(self.team_policy.critic.parameters(), self.max_grad_norm)
        else:
            team_critic_grad_norm = get_gard_norm(self.team_policy.critic.parameters())
        self.team_policy.critic_optimizer.step()

        # if torch.isnan(idv_loss).any():
        #     print("idv_loss has nan")
        # if torch.isnan(team_loss).any():
        #     print("team_loss has nan")

        return idv_value_loss, team_value_loss, idv_critic_grad_norm, team_critic_grad_norm,\
               idv_policy_loss, team_policy_loss, idv_new_dist_entropy, team_new_dist_entropy, \
               idv_kl_loss, team_kl_loss, idv_cross_entropy, team_cross_entropy, idv_actor_grad_norm, team_actor_grad_norm,\
               imp_weights, so_weights, team_imp_weights, clp, tcr, tcrs, ts1, tsl1, ts3, tsl3, tclp, ttcr, tsr, tsl, \
               idv_loss, idv_ppo_abs, idv_ppo_prop, idv_entropy_prop, idv_kl_prop, \
               team_loss, team_ppo_abs, team_ppo_prop, team_entropy_prop, team_kl_prop

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

        train_info = {}

        train_info['Aa_idv_actor_loss'] = 0

        train_info['Ab_policy_loss'] = 0
        train_info['Ac_idv_ppo_loss_abs'] = 0
        train_info['Ad_idv_ppo_prop'] = 0

        train_info['Ae_eta'] = 0
        train_info['Af_noclip_proportion'] = 0
        train_info['Ag_update_proportion'] = 0
        train_info['Ah_update_loss'] = 0

        train_info['Ai_idv_epsilon\''] = 0
        train_info['Aj_idv_sigma'] = 0
        train_info['Ak_idv_clip(sigma, 1-epislon\', 1+epislon\')'] = 0
        train_info['Al_idv_noclip_proportion'] = 0
        train_info['Am_idv_(sigma*A)update_proportion'] = 0
        train_info['An_idv_(sigma*A)update_loss'] = 0

        train_info['Ao_idv_entropy_prop'] = 0
        train_info['Ap_dist_entropy'] = 0

        train_info['Aq_idv_kl_prop'] = 0
        train_info['Ar_idv_kl_coef'] = 0
        train_info['As_idv_kl_loss'] = 0
        train_info['At_idv_cross_entropy'] = 0

        train_info['Au_value_loss'] = 0
        train_info['Av_advantages'] = 0
        train_info['Aw_idv_actor_norm'] = 0
        train_info['Ax_idv_critic_norm'] = 0

        train_info['Ta_team_actor_loss'] = 0

        train_info['Tb_team_policy_loss'] = 0
        train_info['Tc_team_ppo_loss_abs'] = 0
        train_info['Td_team_ppo_prop'] = 0

        train_info['Te_team_epsilon^'] = 0
        train_info['Tf_team_sigma^'] = 0
        train_info['Tg_team_clip(sigma^, 1-epislon^\', 1+epislon^\')'] = 0
        train_info['Th_team_noclip_proportion'] = 0
        train_info['Ti_team_(sigma^*A)update_proportion'] = 0
        train_info['Tj_team_(sigma^*A)update_loss'] = 0

        train_info['Tk_team_entropy_prop'] = 0
        train_info['Tl_team_dist_entropy'] = 0

        train_info['Tm_team_kl_prop'] = 0
        train_info['Tn_team_kl_coef'] = 0
        train_info['To_team_kl_loss'] = 0
        train_info['Tp_team_cross_entropy'] = 0

        train_info['Tq_team_value_loss'] = 0
        train_info['Tr_team_advantages'] = 0
        train_info['Ts_team_actor_norm'] = 0
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
                # print("in", ppi)
                idv_value_loss, team_value_loss, idv_critic_grad_norm, team_critic_grad_norm, \
                idv_policy_loss, team_policy_loss, idv_new_dist_entropy, team_new_dist_entropy, \
                idv_kl_loss, team_kl_loss, idv_cross_entropy, team_cross_entropy, idv_actor_grad_norm, team_actor_grad_norm, \
                imp_weights, so_weights, team_imp_weights, clp, tcr, tcrs, ts1, tsl1, ts3, tsl3, tclp, ttcr, tsr, tsl,\
                idv_loss, idv_ppo_abs, idv_ppo_prop, idv_entropy_prop, idv_kl_prop,\
                team_loss, team_ppo_abs, team_ppo_prop, team_entropy_prop, team_kl_prop\
                    = self.ppo_update(sample, episode, update_actor)

                train_info['Aa_idv_actor_loss'] += idv_loss.item()
                train_info['Ab_policy_loss'] += idv_policy_loss.item()
                train_info['Ac_idv_ppo_loss_abs'] += idv_ppo_abs.item()
                train_info['Ad_idv_ppo_prop'] += idv_ppo_prop.item()
                train_info['Ae_eta'] += imp_weights.mean()
                train_info['Af_noclip_proportion'] += tcr
                train_info['Ag_update_proportion'] += ts1
                train_info['Ah_update_loss'] += tsl1
                train_info['Aj_idv_sigma'] += so_weights.mean()
                train_info['Ak_idv_clip(sigma, 1-epislon\', 1+epislon\')'] += clp.mean()
                train_info['Al_idv_noclip_proportion'] += tcrs
                train_info['Am_idv_(sigma*A)update_proportion'] += ts3
                train_info['An_idv_(sigma*A)update_loss'] += tsl3
                train_info['Ao_idv_entropy_prop'] += idv_entropy_prop.item()
                train_info['Ap_dist_entropy'] += idv_new_dist_entropy.item()
                train_info['Aq_idv_kl_prop'] += idv_kl_prop.item()
                train_info['As_idv_kl_loss'] += idv_kl_loss.item()
                train_info['At_idv_cross_entropy'] += idv_cross_entropy.item()
                train_info['Au_value_loss'] += idv_value_loss.item()
                train_info['Aw_idv_actor_norm'] += idv_actor_grad_norm
                train_info['Ax_idv_critic_norm'] += idv_critic_grad_norm

                train_info['Ta_team_actor_loss'] += team_loss.item()
                train_info['Tb_team_policy_loss'] += team_policy_loss.item()
                train_info['Tc_team_ppo_loss_abs'] += team_ppo_abs.item()
                train_info['Td_team_ppo_prop'] += team_ppo_prop.item()
                train_info['Tf_team_sigma^'] += team_imp_weights.mean()
                train_info['Tg_team_clip(sigma^, 1-epislon^\', 1+epislon^\')'] += tclp.mean()
                train_info['Th_team_noclip_proportion'] += ttcr
                train_info['Ti_team_(sigma^*A)update_proportion'] += tsr
                train_info['Tj_team_(sigma^*A)update_loss'] += tsl
                train_info['Tk_team_entropy_prop'] += team_entropy_prop.item()
                train_info['Tl_team_dist_entropy'] += team_new_dist_entropy.item()
                train_info['Tm_team_kl_prop'] += team_kl_prop.item()
                train_info['To_team_kl_loss'] += team_kl_loss.item()
                train_info['Tp_team_cross_entropy'] += team_cross_entropy.item()
                train_info['Tq_team_value_loss'] += team_value_loss.item()
                train_info['Ts_team_actor_norm'] += team_actor_grad_norm
                train_info['Tt_team_critic_norm'] += team_critic_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        train_info['Ai_idv_epsilon\''] = self.idv_clip_ratio
        train_info['Ar_idv_kl_coef'] = self.idv_kl_coef
        train_info['Av_advantages'] = np.nanmean(idv_advantages)

        train_info['Te_team_epsilon^'] = self.team_clip_ratio
        train_info['Tn_team_kl_coef'] = self.team_kl_coef
        train_info['Tr_team_advantages'] = np.nanmean(team_advantages)

        self.update_entropy_coef()
        if self.idv_use_two_clip:
            self.update_idv_clip_ratio()
        if self.idv_use_kl_loss or self.idv_use_cross_entropy:
            self.update_idv_kl_coef()

        if self.team_use_clip:
            self.update_team_clip_ratio()
        if self.team_use_kl_loss or self.team_use_cross_entropy:
            self.update_team_kl_coef()

        return train_info

    def idv_prep_training(self):
        self.idv_policy.actor.train()
        self.idv_policy.critic.train()

    def team_prep_training(self):
        self.team_policy.actor.train()
        self.team_policy.critic.train()

    def idv_prep_rollout(self):
        self.idv_policy.actor.eval()
        self.idv_policy.critic.eval()

    def team_prep_rollout(self):
        self.team_policy.actor.eval()
        self.team_policy.critic.eval()
