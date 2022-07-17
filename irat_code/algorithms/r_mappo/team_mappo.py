import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
from torch.distributions import kl_divergence


class Team_RMAPPO():
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
        self.policy = policy

        self.clip_param = args.clip_param
        self.team_clip_ratio = args.team_clip_ratio
        self.team_end_clip_ratio = args.team_end_clip_ratio
        # episodes = int(args.num_env_steps) // args.episode_length // args.n_rollout_threads
        self.clip_gap = (self.team_clip_ratio - self.team_end_clip_ratio) / args.team_clip_episodes
        self.kl_coef = args.team_kl_coef
        self.kl_end_coef = args.team_kl_end_coef
        self.kl_anneal_gap = (self.kl_coef - self.kl_end_coef) / args.team_kl_episodes
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
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

        self.team_use_kl_loss = args.team_use_kl_loss

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def update_so_clip_ratio(self):
        tmp = self.team_clip_ratio - self.clip_gap
        self.team_clip_ratio = min(tmp, self.team_end_clip_ratio)

    def update_kl_coef(self):
        tmp = self.kl_coef - self.kl_anneal_gap
        self.kl_coef = max(self.kl_end_coef, tmp)

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
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
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
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

    def ppo_update(self, sample, update_actor=True):
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
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, other_act_dists_batch,\
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        other_log_probs, adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        other_log_probs = check(other_log_probs).to(**self.tpdv).detach()
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy, act_dists = self.policy.evaluate_actions(share_obs_batch,
                                                                                         obs_batch,
                                                                                         rnn_states_batch,
                                                                                         rnn_states_critic_batch,
                                                                                         actions_batch,
                                                                                         masks_batch,
                                                                                         available_actions_batch,
                                                                                         active_masks_batch)
        # print("team", other_act_dists_batch.shape, len(act_dists))
        kl_loss = 0
        for ai in range(len(act_dists)):
            tmp_other_dists = other_act_dists_batch[:, ai]
            tmp_probs = []
            for tp in tmp_other_dists:
                tmp_probs.append(tp.probs)
            # print(tmp_probs)
            tmp_probs = torch.stack(tmp_probs).to(**self.tpdv)
            other_dists = type(act_dists[ai])(probs=tmp_probs)
            kl_loss += kl_divergence(other_dists, act_dists[ai]).mean()

        # actor update
        imp_weights = torch.exp(action_log_probs - other_log_probs)

        surr1 = imp_weights * adv_targ
        tclp = torch.clamp(imp_weights, 1.0 - self.team_clip_ratio, 1.0 + self.team_clip_ratio)
        surr2 = tclp * adv_targ

        tn = imp_weights.clone().view(-1, 1).shape[0]
        tg = imp_weights >= (1.0 - self.team_clip_ratio)
        tl = imp_weights <= (1.0 + self.team_clip_ratio)
        tcr = (tl & tg).float().sum() / tn
        ts = surr1 <= surr2
        tsr = ts.float().sum() / tn
        tsl = surr1.clone()[ts].mean()

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            loss = policy_loss - dist_entropy * self.entropy_coef
            if self.team_use_kl_loss:
                loss += self.kl_coef * kl_loss
            loss.backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, tclp, tcr, tsr, tsl, kl_loss

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['team_value_loss'] = 0
        train_info['team_policy_loss'] = 0
        train_info['team_dist_entropy'] = 0
        # train_info['team_actor_grad_norm'] = 0
        # train_info['team_critic_grad_norm'] = 0
        train_info['team_sigma^'] = 0
        train_info['team_clip(sigma^, 1-epislon^\', 1+epislon^\')'] = 0
        train_info['team_noclip_proportion'] = 0
        train_info['team_(sigma^*A)update_proportion'] = 0
        train_info['team_(sigma^*A)update_loss'] = 0
        train_info['team_kl_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, tclp, tcr, tsr, tsl, kl_loss \
                    = self.ppo_update(sample, update_actor)

                train_info['team_value_loss'] += value_loss.item()
                train_info['team_policy_loss'] += policy_loss.item()
                train_info['team_dist_entropy'] += dist_entropy.item()
                # train_info['team_actor_grad_norm'] += actor_grad_norm
                # train_info['team_critic_grad_norm'] += critic_grad_norm
                train_info['team_noclip_proportion'] += tcr
                train_info['team_(sigma^*A)update_proportion'] += tsr
                train_info['team_(sigma^*A)update_loss'] += tsl
                train_info['team_sigma^'] += imp_weights.mean()
                train_info['team_clip(sigma^, 1-epislon^\', 1+epislon^\')'] += tclp.mean()
                train_info['team_kl_loss'] += kl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
        
        train_info['team_epsilon^'] = self.team_clip_ratio
        train_info['team_advantages'] = np.mean(advantages)
        train_info['team_kl_coef'] = self.kl_coef

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        action = check(action).to(**self.tpdv)
        value, action_log_prob, _, _ = self.policy.evaluate_actions(
            cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions, active_masks
        )
        return value, action_log_prob
