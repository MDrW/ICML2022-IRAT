import time
import wandb
import numpy as np
from functools import reduce
import torch
from onpolicy.runner.shared.base_runner_trsyn import Runner
from onpolicy.algorithms.utils.distributions import FixedNormal, FixedCategorical


def _t2n(x):
    return x.detach().cpu().numpy()


class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(SMACRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.idv_policy.lr_decay(episode, episodes)
                self.trainer.team_policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists = \
                    self.idv_collect(step)

                # Get data using Team Policy
                team_values, team_actions, team_log_probs, team_rnn, team_rnn_critic, team_act_dists = \
                    self.team_collect(step)

                if self.all_args.change_reward and episode > self.all_args.change_reward_episode and \
                        self.all_args.change_use_policy == "team":
                    actions = team_actions
                    values, action_log_probs = self.evaluate_actions("idv", step, actions)
                else:
                    team_values, team_log_probs = self.evaluate_actions("team", step, actions)

                # Obser reward and next obs
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists, \
                       team_values, team_log_probs, team_rnn, team_rnn_critic, team_act_dists

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train(episode)

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Map {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.map_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "StarCraft2":
                    battles_won = []
                    battles_game = []
                    incre_battles_won = []
                    incre_battles_game = []

                    for i, info in enumerate(infos):
                        if 'battles_won' in info[0].keys():
                            battles_won.append(info[0]['battles_won'])
                            incre_battles_won.append(info[0]['battles_won'] - last_battles_won[i])
                        if 'battles_game' in info[0].keys():
                            battles_game.append(info[0]['battles_game'])
                            incre_battles_game.append(info[0]['battles_game'] - last_battles_game[i])

                    incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(
                        incre_battles_game) > 0 else 0.0
                    print("incre win rate is {}.".format(incre_win_rate))
                    if self.use_wandb:
                        wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                    else:
                        self.writter.add_scalars("incre_win_rate", {"incre_win_rate": incre_win_rate}, total_num_steps)

                    last_battles_game = battles_game
                    last_battles_won = battles_won

                train_infos['dead_ratio'] = 1 - self.buffer.active_masks.sum() / reduce(lambda x, y: x * y, list(
                    self.buffer.active_masks.shape))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, "team_policy")
                self.eval(total_num_steps, "idv_policy")

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            team_share_obs = obs.copy()
        else:
            team_share_obs = share_obs.copy()

        if not self.idv_use_shared_obs:
            idv_share_obs = obs.copy()
        else:
            idv_share_obs = share_obs.copy()

        self.buffer.idv_share_obs[0] = idv_share_obs.copy()
        self.buffer.team_share_obs[0] = team_share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = available_actions.copy()

    @torch.no_grad()
    def idv_collect(self, step):
        self.trainer.idv_prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, act_dist \
            = self.trainer.idv_policy.get_actions(np.concatenate(self.buffer.idv_share_obs[step]),
                                                  np.concatenate(self.buffer.obs[step]),
                                                  np.concatenate(self.buffer.idv_rnn_states[step]),
                                                  np.concatenate(self.buffer.idv_rnn_states_critic[step]),
                                                  np.concatenate(self.buffer.masks[step]),
                                                  np.concatenate(self.buffer.available_actions[step]))

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        act_dists = []
        for dist in act_dist:
            if type(dist) == FixedCategorical:
                tmp_probs = dist.probs.detach()
                tps = []
                for tp in tmp_probs:
                    tps.append(type(dist)(probs=tp))
            elif type(dist) == FixedNormal:
                tmp_mu = dist.loc.detach()
                tmp_sigma = dist.scale.detach()
                tps = []
                for tm, ts in zip(tmp_mu, tmp_sigma):
                    tps.append(type(dist)(loc=tm, scale=ts))
            else:
                raise NotImplementedError
            act_dists.append(tps)
        act_dists = np.array(np.split(np.array(act_dists).transpose((1, 0)), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists

    @torch.no_grad()
    def team_collect(self, step):
        self.trainer.team_prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic, act_dist \
            = self.trainer.team_policy.get_actions(np.concatenate(self.buffer.team_share_obs[step]),
                                                   np.concatenate(self.buffer.obs[step]),
                                                   np.concatenate(self.buffer.team_rnn_states[step]),
                                                   np.concatenate(self.buffer.team_rnn_states_critic[step]),
                                                   np.concatenate(self.buffer.masks[step]),
                                                   np.concatenate(self.buffer.available_actions[step]),
                                                   deterministic=True)

        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        act_dists = []
        for dist in act_dist:
            if type(dist) == FixedCategorical:
                tmp_probs = dist.probs.detach()
                tps = []
                for tp in tmp_probs:
                    tps.append(type(dist)(probs=tp))
            elif type(dist) == FixedNormal:
                tmp_mu = dist.loc.detach()
                tmp_sigma = dist.scale.detach()
                tps = []
                for tm, ts in zip(tmp_mu, tmp_sigma):
                    tps.append(type(dist)(loc=tm, scale=ts))
            else:
                raise NotImplementedError
            act_dists.append(tps)
        act_dists = np.array(np.split(np.array(act_dists).transpose((1, 0)), self.n_rollout_threads))

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists

    @torch.no_grad()
    def evaluate_actions(self, policy: str, step, actions):
        if policy == "team":
            self.trainer.team_prep_rollout()
            exc_policy = self.trainer.team_policy
            share_obs = np.concatenate(self.buffer.team_share_obs[step])
            obs = np.concatenate(self.buffer.obs[step])
            rnn_states = np.concatenate(self.buffer.team_rnn_states[step])
            rnn_states_critic = np.concatenate(self.buffer.team_rnn_states_critic[step])
            masks = np.concatenate(self.buffer.masks[step])
            available_actions = np.concatenate(self.buffer.available_actions[step])
            active_masks = np.concatenate(self.buffer.active_masks[step])
        else:
            self.trainer.idv_prep_rollout()
            exc_policy = self.trainer.idv_policy
            share_obs = np.concatenate(self.buffer.idv_share_obs[step])
            obs = np.concatenate(self.buffer.obs[step])
            rnn_states = np.concatenate(self.buffer.idv_rnn_states[step])
            rnn_states_critic = np.concatenate(self.buffer.idv_rnn_states_critic[step])
            masks = np.concatenate(self.buffer.masks[step])
            available_actions = np.concatenate(self.buffer.available_actions[step])
            active_masks = np.concatenate(self.buffer.active_masks[step])

            # print(type(policy), policy.evaluate_actions)
        value, action_log_prob, _, _ \
            = exc_policy.evaluate_actions(share_obs, obs, rnn_states, rnn_states_critic,
                                          np.concatenate(actions), masks, available_actions, active_masks)

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        return values, action_log_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists, \
        team_values, team_log_probs, team_rnn, team_rnn_critic, team_act_dists = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer.idv_rnn_states_critic.shape[3:]), dtype=np.float32)

        team_rnn[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents,
                                                self.recurrent_N, self.hidden_size),
                                               dtype=np.float32)
        team_rnn_critic[dones_env == True] = np.zeros(((dones_env == True).sum(),
                                                      self.num_agents, *self.buffer.team_rnn_states_critic.shape[3:]),
                                                      dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array(
            [[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in
             infos])

        if not self.use_centralized_V:
            team_share_obs = obs.copy()
        else:
            team_share_obs = share_obs.copy()

        if not self.idv_use_shared_obs:
            idv_share_obs = obs.copy()
        else:
            idv_share_obs = share_obs.copy()

        idv_rewards, team_rewards = [], []
        for info in infos:
            irw, trw = [], []
            for i in range(self.num_agents):
                irw.append([info[i]["individual_reward"]])
                trw.append([info[i]["team_reward"]])
            idv_rewards.append(irw)
            team_rewards.append(trw)
        idv_rewards = np.array(idv_rewards)  # n_rollout_thread, n_agent, 1
        team_rewards = np.array(team_rewards)

        self.buffer.insert(idv_share_obs, team_share_obs, obs,
                           rnn_states, team_rnn, rnn_states_critic, team_rnn_critic,
                           actions, act_dists, team_act_dists, action_log_probs, team_log_probs,
                           values, team_values, idv_rewards, team_rewards, masks,
                           bad_masks, active_masks, available_actions)

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_team_rewards"] = np.mean(self.buffer.team_rewards)
        train_infos["average_step_rewards"] = np.mean(self.buffer.idv_rewards)
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps, title):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        eval_episode_team_rewards = []
        one_episode_team_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            if title == "team_policy":
                self.trainer.team_prep_rollout()
                policy = self.trainer.team_policy
            else:
                self.trainer.idv_prep_rollout()
                policy = self.trainer.idv_policy
            eval_actions, eval_rnn_states = \
                policy.act(np.concatenate(eval_obs),
                           np.concatenate(eval_rnn_states),
                           np.concatenate(eval_masks),
                           np.concatenate(eval_available_actions),
                           deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(
                eval_actions)
            one_episode_rewards.append(eval_rewards)

            # tirw, ttrw = [], []
            ttrw = []
            for info in eval_infos:
                ttrw.append(info[0]["team_reward"])
            eval_episode_team_rewards.append(np.array(ttrw))

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    eval_episode_team_rewards.append(np.sum(one_episode_team_rewards, axis=0))
                    one_episode_rewards = []
                    one_episode_team_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_episode_team_rewards = np.array(eval_episode_team_rewards)
                eval_env_infos = {title + '_eval_average_episode_rewards': eval_episode_rewards,
                                  title + '_eval_average_episode_team_rewards': eval_episode_team_rewards}
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print(title + " eval win rate is {}.".format(eval_win_rate))
                if self.use_wandb:
                    wandb.log({title + "_eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalars(title + "_eval_win_rate", {title + "_eval_win_rate": eval_win_rate},
                                             total_num_steps)
                break
