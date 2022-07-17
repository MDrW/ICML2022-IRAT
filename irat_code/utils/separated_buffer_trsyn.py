import torch
import numpy as np
from collections import defaultdict

from onpolicy.utils.util import check, get_shape_from_obs_space, get_shape_from_act_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 0, 2).reshape(-1, *x.shape[2:])


class SeparatedReplayBuffer(object):
    def __init__(self, args, obs_space, idv_share_obs_space, team_share_obs_space, act_space):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        idv_share_obs_shape = get_shape_from_obs_space(idv_share_obs_space)
        team_share_obs_shape = get_shape_from_obs_space(team_share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(idv_share_obs_shape[-1]) == list:
            idv_share_obs_shape = idv_share_obs_shape[:1]

        if type(team_share_obs_shape[-1]) == list:
            team_share_obs_shape = team_share_obs_shape[:1]

        self.idv_share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *idv_share_obs_shape), dtype=np.float32)
        self.team_share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *team_share_obs_shape), dtype=np.float32)

        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, *obs_shape), dtype=np.float32)

        self.idv_rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.idv_rnn_states_critic = np.zeros_like(self.idv_rnn_states)

        self.team_rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.team_rnn_states_critic = np.zeros_like(self.team_rnn_states)

        self.idv_value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.idv_returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        self.team_value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.team_returns = np.zeros((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)

        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)
        # print(act_space, ",", act_shape)

        self.actions = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.idv_actions_dists = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=object)
        self.team_actions_dists = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=object)
        self.idv_action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.team_action_log_probs = np.zeros((self.episode_length, self.n_rollout_threads, act_shape), dtype=np.float32)

        self.idv_rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        self.team_rewards = np.zeros((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)

        self.step = 0

    def insert(self, idv_share_obs, team_share_obs, obs, idv_rnn_states, team_rnn_states,
               idv_rnn_states_critic, team_rnn_states_critic, actions, idv_actions_dists, team_actions_dists,
               idv_action_log_probs, team_action_log_probs, idv_value_preds, team_value_preds,
               idv_rewards, team_rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.idv_share_obs[self.step + 1] = idv_share_obs.copy()
        self.team_share_obs[self.step + 1] = team_share_obs.copy()

        self.obs[self.step + 1] = obs.copy()

        self.idv_rnn_states[self.step + 1] = idv_rnn_states.copy()
        self.team_rnn_states[self.step + 1] = team_rnn_states.copy()

        self.idv_rnn_states_critic[self.step + 1] = idv_rnn_states_critic.copy()
        self.team_rnn_states_critic[self.step + 1] = team_rnn_states_critic.copy()

        self.actions[self.step] = actions.copy()

        self.idv_actions_dists[self.step] = idv_actions_dists.copy()
        self.team_actions_dists[self.step] = team_actions_dists.copy()

        self.idv_action_log_probs[self.step] = idv_action_log_probs.copy()
        self.team_action_log_probs[self.step] = team_action_log_probs.copy()

        self.idv_value_preds[self.step] = idv_value_preds.copy()
        self.team_value_preds[self.step] = team_value_preds.copy()

        self.idv_rewards[self.step] = idv_rewards.copy()
        self.team_rewards[self.step] = team_rewards.copy()

        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.idv_share_obs[0] = self.idv_share_obs[-1].copy()
        self.team_share_obs[0] = self.team_share_obs[-1].copy()

        self.obs[0] = self.obs[-1].copy()

        self.idv_rnn_states[0] = self.idv_rnn_states[-1].copy()
        self.team_rnn_states[0] = self.team_rnn_states[-1].copy()

        self.idv_rnn_states_critic[0] = self.idv_rnn_states_critic[-1].copy()
        self.team_rnn_states_critic[0] = self.team_rnn_states_critic[-1].copy()

        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()

    def idv_compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.idv_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.idv_rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.idv_rewards[step] + self.gamma * \
                                value_normalizer.denormalize(self.idv_value_preds[step + 1]) * \
                                self.masks[step + 1] - value_normalizer.denormalize(self.idv_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.idv_returns[step] = gae + value_normalizer.denormalize(self.idv_value_preds[step])
                    else:
                        delta = self.idv_rewards[step] + self.gamma * self.idv_value_preds[step + 1] * \
                                self.masks[step + 1] - self.idv_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.idv_returns[step] = gae + self.idv_value_preds[step]
            else:
                self.idv_returns[-1] = next_value
                for step in reversed(range(self.idv_rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.idv_returns[step] = (self.idv_returns[step + 1] * self.gamma * self.masks[step + 1] +
                                                  self.idv_rewards[step]) * self.bad_masks[step + 1] \
                                                 + (1 - self.bad_masks[step + 1]) \
                                                 * value_normalizer.denormalize(self.idv_value_preds[step])
                    else:
                        self.idv_returns[step] = (self.idv_returns[step + 1] * self.gamma * self.masks[step + 1]
                                                  + self.idv_rewards[step]) * self.bad_masks[step + 1] \
                                                 + (1 - self.bad_masks[step + 1]) * self.idv_value_preds[step]
        else:
            if self._use_gae:
                self.idv_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.idv_rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.idv_rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.idv_value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(
                            self.idv_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.idv_returns[step] = gae + value_normalizer.denormalize(self.idv_value_preds[step])
                    else:
                        delta = self.idv_rewards[step] + self.gamma * self.idv_value_preds[step + 1] * \
                                self.masks[step + 1] - self.idv_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.idv_returns[step] = gae + self.idv_value_preds[step]
            else:
                self.idv_returns[-1] = next_value
                for step in reversed(range(self.idv_rewards.shape[0])):
                    self.idv_returns[step] = self.idv_returns[step + 1] * self.gamma * self.masks[step + 1] \
                                             + self.idv_rewards[step]
    
    def team_compute_returns(self, next_value, value_normalizer=None):
        if self._use_proper_time_limits:
            if self._use_gae:
                self.team_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.team_rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.team_rewards[step] + self.gamma * \
                                value_normalizer.denormalize(self.team_value_preds[step + 1]) * \
                                self.masks[step + 1] - value_normalizer.denormalize(self.team_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.team_returns[step] = gae + value_normalizer.denormalize(self.team_value_preds[step])
                    else:
                        delta = self.team_rewards[step] + self.gamma * self.team_value_preds[step + 1] \
                                * self.masks[step + 1] - self.team_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.team_returns[step] = gae + self.team_value_preds[step]
            else:
                self.team_returns[-1] = next_value
                for step in reversed(range(self.team_rewards.shape[0])):
                    if self._use_popart:
                        self.team_returns[step] = (self.team_returns[step + 1] * self.gamma * self.masks[step + 1] 
                                                   + self.team_rewards[step]) * self.bad_masks[step + 1] + \
                                                  (1 - self.bad_masks[step + 1]) * \
                                                  value_normalizer.denormalize(self.team_value_preds[step])
                    else:
                        self.team_returns[step] = (self.team_returns[step + 1] * self.gamma * self.masks[step + 1] 
                                                   + self.team_rewards[step]) * self.bad_masks[step + 1] \
                                                  + (1 - self.bad_masks[step + 1]) * self.team_value_preds[step]
        else:
            if self._use_gae:
                self.team_value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.team_rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.team_rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.team_value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(
                            self.team_value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.team_returns[step] = gae + value_normalizer.denormalize(self.team_value_preds[step])
                    else:
                        delta = self.team_rewards[step] + self.gamma * self.team_value_preds[step + 1] * \
                                self.masks[step + 1] - self.team_value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.team_returns[step] = gae + self.team_value_preds[step]
            else:
                self.team_returns[-1] = next_value
                for step in reversed(range(self.team_rewards.shape[0])):
                    self.team_returns[step] = self.team_returns[step + 1] * self.gamma * \
                                              self.masks[step + 1] + self.team_rewards[step]

    def feed_forward_generator(self, idv_advantages, team_advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.idv_rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        idv_share_obs = self.idv_share_obs[:-1].reshape(-1, *self.idv_share_obs.shape[2:])
        team_share_obs = self.team_share_obs[:-1].reshape(-1, *self.team_share_obs.shape[2:])

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])

        idv_rnn_states = self.idv_rnn_states[:-1].reshape(-1, *self.idv_rnn_states.shape[2:])
        team_rnn_states = self.team_rnn_states[:-1].reshape(-1, *self.team_rnn_states.shape[2:])

        idv_rnn_states_critic = self.idv_rnn_states_critic[:-1].reshape(-1, *self.idv_rnn_states_critic.shape[2:])
        team_rnn_states_critic = self.team_rnn_states_critic[:-1].reshape(-1, *self.team_rnn_states_critic.shape[2:])

        actions = self.actions.reshape(-1, self.actions.shape[-1])

        idv_actions_dists = self.idv_actions_dists.reshape(-1, self.idv_actions_dists.shape[-1])
        team_actions_dists = self.team_actions_dists.reshape(-1, self.team_actions_dists.shape[-1])

        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])

        idv_value_preds = self.idv_value_preds[:-1].reshape(-1, 1)
        team_value_preds = self.team_value_preds[:-1].reshape(-1, 1)

        idv_returns = self.idv_returns[:-1].reshape(-1, 1)
        team_returns = self.team_returns[:-1].reshape(-1, 1)

        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)

        idv_action_log_probs = self.idv_action_log_probs.reshape(-1, self.idv_action_log_probs.shape[-1])
        team_action_log_probs = self.team_action_log_probs.reshape(-1, self.team_action_log_probs.shape[-1])

        idv_advantages = idv_advantages.reshape(-1, 1)
        team_advantages = team_advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            idv_share_obs_batch = idv_share_obs[indices]
            team_share_obs_batch = team_share_obs[indices]

            obs_batch = obs[indices]

            idv_rnn_states_batch = idv_rnn_states[indices]
            team_rnn_states_batch = team_rnn_states[indices]

            idv_rnn_states_critic_batch = idv_rnn_states_critic[indices]
            team_rnn_states_critic_batch = team_rnn_states_critic[indices]

            actions_batch = actions[indices]

            idv_actions_dists_batch = idv_actions_dists[indices]
            team_actions_dists_batch = team_actions_dists[indices]

            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None

            idv_value_preds_batch = idv_value_preds[indices]
            team_value_preds_batch = team_value_preds[indices]

            idv_return_batch = idv_returns[indices]
            team_return_batch = team_returns[indices]

            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]

            idv_action_log_probs_batch = idv_action_log_probs[indices]
            team_action_log_probs_batch = team_action_log_probs[indices]

            idv_adv_targ = idv_advantages[indices]
            team_adv_targ = team_advantages[indices]

            yield idv_share_obs_batch, team_share_obs_batch, obs_batch, \
                  idv_rnn_states_batch, team_rnn_states_batch, idv_rnn_states_critic_batch, team_rnn_states_critic_batch,\
                  actions_batch, idv_actions_dists_batch, team_actions_dists_batch, \
                  idv_value_preds_batch, team_value_preds_batch, idv_return_batch, team_return_batch, \
                  masks_batch, active_masks_batch, idv_action_log_probs_batch, team_action_log_probs_batch, \
                  idv_adv_targ, team_adv_targ, available_actions_batch

    def naive_recurrent_generator(self, idv_advantages, team_advantages, num_mini_batch):
        n_rollout_threads = self.idv_rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))

        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()

        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            idv_share_obs_batch = []
            team_share_obs_batch = []

            obs_batch = []

            idv_rnn_states_batch = []
            team_rnn_states_batch = []

            idv_rnn_states_critic_batch = []
            team_rnn_states_critic_batch = []

            actions_batch = []

            idv_actions_dists_batch = []
            team_actions_dists_batch = []

            available_actions_batch = []

            idv_value_preds_batch = []
            team_value_preds_batch = []

            idv_return_batch = []
            team_return_batch = []

            masks_batch = []
            active_masks_batch = []

            idv_action_log_probs_batch = []
            team_action_log_probs_batch = []

            idv_adv_targ = []
            team_adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                idv_share_obs_batch.append(self.idv_share_obs[:-1, ind])
                team_share_obs_batch.append(self.team_share_obs[:-1, ind])

                obs_batch.append(self.obs[:-1, ind])

                idv_rnn_states_batch.append(self.idv_rnn_states[0:1, ind])
                team_rnn_states_batch.append(self.team_rnn_states[0:1, ind])

                idv_rnn_states_critic_batch.append(self.idv_rnn_states_critic[0:1, ind])
                team_rnn_states_critic_batch.append(self.team_rnn_states_critic[0:1, ind])

                actions_batch.append(self.actions[:, ind])

                idv_actions_dists_batch.append(self.idv_actions_dists[:, ind])
                team_actions_dists_batch.append(self.team_actions_dists[:, ind])

                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])

                idv_value_preds_batch.append(self.idv_value_preds[:-1, ind])
                team_value_preds_batch.append(self.team_value_preds[:-1, ind])

                idv_return_batch.append(self.idv_returns[:-1, ind])
                team_return_batch.append(self.team_returns[:-1, ind])

                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])

                idv_action_log_probs_batch.append(self.idv_action_log_probs[:, ind])
                team_action_log_probs_batch.append(self.team_action_log_probs[:, ind])

                idv_adv_targ.append(idv_advantages[:, ind])
                team_adv_targ.append(team_advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            idv_share_obs_batch = np.stack(idv_share_obs_batch, 1)
            team_share_obs_batch = np.stack(team_share_obs_batch, 1)

            obs_batch = np.stack(obs_batch, 1)

            actions_batch = np.stack(actions_batch, 1)

            idv_actions_dists_batch = np.stack(idv_actions_dists_batch, 1)
            team_actions_dists_batch = np.stack(team_actions_dists_batch, 1)

            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)

            idv_value_preds_batch = np.stack(idv_value_preds_batch, 1)
            team_value_preds_batch = np.stack(team_value_preds_batch, 1)

            idv_return_batch = np.stack(idv_return_batch, 1)
            team_return_batch = np.stack(team_return_batch, 1)

            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)

            idv_action_log_probs_batch = np.stack(idv_action_log_probs_batch, 1)
            team_action_log_probs_batch = np.stack(team_action_log_probs_batch, 1)

            idv_adv_targ = np.stack(idv_adv_targ, 1)
            team_adv_targ = np.stack(team_adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            idv_rnn_states_batch = np.stack(idv_rnn_states_batch, 1).reshape(N, *self.idv_rnn_states.shape[2:])
            team_rnn_states_batch = np.stack(team_rnn_states_batch, 1).reshape(N, *self.team_rnn_states.shape[2:])

            idv_rnn_states_critic_batch = np.stack(idv_rnn_states_critic_batch, 1).reshape(
                N, *self.idv_rnn_states_critic.shape[2:])
            team_rnn_states_critic_batch = np.stack(team_rnn_states_critic_batch, 1).reshape(
                N, *self.team_rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            idv_share_obs_batch = _flatten(T, N, idv_share_obs_batch)
            team_share_obs_batch = _flatten(T, N, team_share_obs_batch)

            obs_batch = _flatten(T, N, obs_batch)

            actions_batch = _flatten(T, N, actions_batch)

            idv_actions_dists_batch = _flatten(T, N, idv_actions_dists_batch)
            team_actions_dists_batch = _flatten(T, N, team_actions_dists_batch)

            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None

            idv_value_preds_batch = _flatten(T, N, idv_value_preds_batch)
            team_value_preds_batch = _flatten(T, N, team_value_preds_batch)

            idv_return_batch = _flatten(T, N, idv_return_batch)
            team_return_batch = _flatten(T, N, team_return_batch)

            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)

            idv_action_log_probs_batch = _flatten(T, N, idv_action_log_probs_batch)
            team_action_log_probs_batch = _flatten(T, N, team_action_log_probs_batch)

            idv_adv_targ = _flatten(T, N, idv_adv_targ)
            team_adv_targ = _flatten(T, N, team_adv_targ)

            yield idv_share_obs_batch, team_share_obs_batch, obs_batch, \
                  idv_rnn_states_batch, team_rnn_states_batch, idv_rnn_states_critic_batch, team_rnn_states_critic_batch,\
                  actions_batch, idv_actions_dists_batch, team_actions_dists_batch, \
                  idv_value_preds_batch, team_value_preds_batch, idv_return_batch, team_return_batch,\
                  masks_batch, active_masks_batch, idv_action_log_probs_batch, team_action_log_probs_batch, \
                  idv_adv_targ, team_adv_targ, available_actions_batch

    def recurrent_generator(self, idv_advantages, team_advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.idv_rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.idv_share_obs.shape) > 3:
            idv_share_obs = self.idv_share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.idv_share_obs.shape[2:])
            team_share_obs = self.team_share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.team_share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            idv_share_obs = _cast(self.idv_share_obs[:-1])
            team_share_obs = _cast(self.team_share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)

        idv_actions_dists = _cast(self.idv_actions_dists)
        team_actions_dists = _cast(self.team_actions_dists)

        idv_action_log_probs = _cast(self.idv_action_log_probs)
        team_action_log_probs = _cast(self.team_action_log_probs)

        idv_advantages = _cast(idv_advantages)
        team_advantages = _cast(team_advantages)

        idv_value_preds = _cast(self.idv_value_preds[:-1])
        team_value_preds = _cast(self.team_value_preds[:-1])

        idv_returns = _cast(self.idv_returns[:-1])
        team_returns = _cast(self.team_returns[:-1])

        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])

        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        idv_rnn_states = self.idv_rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.idv_rnn_states.shape[2:])
        team_rnn_states = self.team_rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.team_rnn_states.shape[2:])

        idv_rnn_states_critic = self.idv_rnn_states_critic[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.idv_rnn_states_critic.shape[2:])
        team_rnn_states_critic = self.team_rnn_states_critic[:-1].transpose(
            1, 0, 2, 3).reshape(-1, *self.team_rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            idv_share_obs_batch, team_share_obs_batch = [], []
            obs_batch = []
            idv_rnn_states_batch, team_rnn_states_batch = [], []
            idv_rnn_states_critic_batch, team_rnn_states_critic_batch = [], []
            actions_batch = []
            idv_actions_dists_batch, team_actions_dists_batch = [], []
            available_actions_batch = []
            idv_value_preds_batch, team_value_preds_batch = [], []
            idv_return_batch, team_return_batch = [], []
            masks_batch = []
            active_masks_batch = []
            idv_action_log_probs_batch, team_action_log_probs_batch = [], []
            idv_adv_targ, team_adv_targ = [], []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                idv_share_obs_batch.append(idv_share_obs[ind:ind + data_chunk_length])
                team_share_obs_batch.append(team_share_obs[ind:ind + data_chunk_length])

                obs_batch.append(obs[ind:ind + data_chunk_length])

                actions_batch.append(actions[ind:ind + data_chunk_length])

                idv_actions_dists_batch.append(idv_actions_dists[ind:ind + data_chunk_length])
                team_actions_dists_batch.append(team_actions_dists[ind:ind + data_chunk_length])

                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])

                idv_value_preds_batch.append(idv_value_preds[ind:ind + data_chunk_length])
                team_value_preds_batch.append(team_value_preds[ind:ind + data_chunk_length])

                idv_return_batch.append(idv_returns[ind:ind + data_chunk_length])
                team_return_batch.append(team_returns[ind:ind + data_chunk_length])

                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])

                idv_action_log_probs_batch.append(idv_action_log_probs[ind:ind + data_chunk_length])
                team_action_log_probs_batch.append(team_action_log_probs[ind:ind + data_chunk_length])

                idv_adv_targ.append(idv_advantages[ind:ind + data_chunk_length])
                team_adv_targ.append(team_advantages[ind:ind + data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                idv_rnn_states_batch.append(idv_rnn_states[ind])
                team_rnn_states_batch.append(team_rnn_states[ind])

                idv_rnn_states_critic_batch.append(idv_rnn_states_critic[ind])
                team_rnn_states_critic_batch.append(team_rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            idv_share_obs_batch = np.stack(idv_share_obs_batch)
            team_share_obs_batch = np.stack(team_share_obs_batch)

            obs_batch = np.stack(obs_batch)

            actions_batch = np.stack(actions_batch)

            idv_actions_dists_batch = np.stack(idv_actions_dists_batch)
            team_actions_dists_batch = np.stack(team_actions_dists_batch)

            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch)

            idv_value_preds_batch = np.stack(idv_value_preds_batch)
            team_value_preds_batch = np.stack(team_value_preds_batch)

            idv_return_batch = np.stack(idv_return_batch)
            team_return_batch = np.stack(team_return_batch)

            masks_batch = np.stack(masks_batch)
            active_masks_batch = np.stack(active_masks_batch)

            idv_action_log_probs_batch = np.stack(idv_action_log_probs_batch)
            team_action_log_probs_batch = np.stack(team_action_log_probs_batch)

            idv_adv_targ = np.stack(idv_adv_targ)
            team_adv_targ = np.stack(team_adv_targ)

            # States is just a (N, -1) from_numpy
            idv_rnn_states_batch = np.stack(idv_rnn_states_batch).reshape(N, *self.idv_rnn_states.shape[2:])
            team_rnn_states_batch = np.stack(team_rnn_states_batch).reshape(N, *self.team_rnn_states.shape[2:])

            idv_rnn_states_critic_batch = np.stack(idv_rnn_states_critic_batch).reshape(
                N, *self.idv_rnn_states_critic.shape[2:])
            team_rnn_states_critic_batch = np.stack(team_rnn_states_critic_batch).reshape(
                N, *self.team_rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            idv_share_obs_batch = _flatten(L, N, idv_share_obs_batch)
            team_share_obs_batch = _flatten(L, N, team_share_obs_batch)

            obs_batch = _flatten(L, N, obs_batch)

            actions_batch = _flatten(L, N, actions_batch)

            idv_actions_dists_batch = _flatten(L, N, idv_actions_dists_batch)
            team_actions_dists_batch = _flatten(L, N, team_actions_dists_batch)

            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None

            idv_value_preds_batch = _flatten(L, N, idv_value_preds_batch)
            team_value_preds_batch = _flatten(L, N, team_value_preds_batch)

            idv_return_batch = _flatten(L, N, idv_return_batch)
            team_return_batch = _flatten(L, N, team_return_batch)

            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)

            idv_action_log_probs_batch = _flatten(L, N, idv_action_log_probs_batch)
            team_action_log_probs_batch = _flatten(L, N, team_action_log_probs_batch)

            idv_adv_targ = _flatten(L, N, idv_adv_targ)
            team_adv_targ = _flatten(L, N, team_adv_targ)

            yield idv_share_obs_batch, team_share_obs_batch, obs_batch, \
                  idv_rnn_states_batch, team_rnn_states_batch, idv_rnn_states_critic_batch, team_rnn_states_critic_batch,\
                  actions_batch, idv_actions_dists_batch, team_actions_dists_batch, \
                  idv_value_preds_batch, team_value_preds_batch, idv_return_batch, team_return_batch,\
                  masks_batch, active_masks_batch, idv_action_log_probs_batch, team_action_log_probs_batch,\
                  idv_adv_targ, team_adv_targ, available_actions_batch
