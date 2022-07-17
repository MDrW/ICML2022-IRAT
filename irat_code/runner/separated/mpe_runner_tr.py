import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner_tr import Runner
import imageio
from torch.distributions import kl_divergence as kld


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        # 热启动 把环境重置得到obv,存到buffer里面
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.idv_trainer[agent_id].policy.lr_decay(episode, episodes)
                    self.team_trainer[agent_id].policy.lr_decay(episode, episodes)
            #n thread rollout 的收集数据
            for step in range(self.episode_length):
                # print(episode, step, "start")
                # 返回 n个环境的一个step数据
                # Sample actions 直接从buffer里面取了obv
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, act_dists = \
                    self.idv_collect(step)

                # Get data using Team Policy
                team_values, team_actions, team_log_probs, team_rnn, team_rnn_critic, team_act_dists \
                    = self.team_collect(step, actions)
                # print(len(act_dists[0]), act_dists[0])
                # t = kld(act_dists[0][0], team_act_dists[0][0])
                # t.requires_grad_(True)
                # print("t", t.requires_grad)
                # print(episode, step, "end")

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # insert data into individual buffer
                data = obs, rewards, dones, infos, values, actions, action_log_probs, team_log_probs, rnn_states, rnn_states_critic, team_act_dists
                self.idv_insert(data)

                # insert data into team buffer
                data = obs, rewards, dones, infos, team_values, team_actions, team_log_probs, action_log_probs, team_rnn, team_rnn_critic, act_dists
                self.team_insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        # idv_rews = []
                        # for info in infos:
                        #     if 'individual_reward' in info[agent_id].keys():
                        #         idv_rews.append(info[agent_id]['individual_reward'])
                        # train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update(
                            {'average_step_individual_rewards': np.mean(self.idv_buffer[agent_id].rewards[-1])})
                        train_infos[agent_id].update(
                            {"average_episode_team_rewards": np.mean(self.team_buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, self.team_trainer, "team_policy")
                self.eval(total_num_steps, self.idv_trainer, "idv_policy")

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        idv_share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
            idv_share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        idv_share_obs = np.array(idv_share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.team_buffer[agent_id].share_obs[0] = share_obs.copy()
            self.team_buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

            if not self.idv_use_shared_obs:
                idv_share_obs = np.array(list(obs[:, agent_id]))
            self.idv_buffer[agent_id].share_obs[0] = idv_share_obs.copy()
            self.idv_buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def idv_collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        act_dists = []

        for agent_id in range(self.num_agents):
            # 设置为eval()
            self.idv_trainer[agent_id].prep_rollout()
            # self.idv_trainer[agent_id].prep_training()
            value, action, action_log_prob, rnn_state, rnn_state_critic, act_dist \
                = self.idv_trainer[agent_id].policy.get_actions(self.idv_buffer[agent_id].share_obs[step],
                                                                self.idv_buffer[agent_id].obs[step],
                                                                self.idv_buffer[agent_id].rnn_states[step],
                                                                self.idv_buffer[agent_id].rnn_states_critic[step],
                                                                self.idv_buffer[agent_id].masks[step])

            # print(act_dist[0].probs)
            tmp_act_dist = []
            for dist in act_dist:
                tmp_probs = dist.probs.detach()
                tps = []
                for tp in tmp_probs:
                    tps.append(type(dist)(probs=tp))
                tmp_act_dist.append(tps)
            act_dists.append(tmp_act_dist)

            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action 和环境接受动作对齐
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        # 把数据转为buffer格式
        values = np.array(values).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        action_log_probs = np.array(action_log_probs).transpose((1, 0, 2))
        rnn_states = np.array(rnn_states).transpose((1, 0, 2, 3))
        rnn_states_critic = np.array(rnn_states_critic).transpose((1, 0, 2, 3))
        # print(act_dists[0][0][0].probs, act_dists[1][0][2].probs)
        # print(np.array(act_dists), np.array(act_dists).shape)        # 2, 1, 4 n_agents, n_actions, n_threads
        act_dists = np.array(act_dists).transpose((2, 0, 1))
        # print(act_dists[0][0][0].probs, act_dists[2][1][0].probs)
        # print(act_dists, act_dists.shape)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env, act_dists

    @torch.no_grad()
    def team_collect(self, step, idv_actions):
        values = []
        actions = []
        # temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []
        act_dists = []

        for agent_id in range(self.num_agents):
            self.team_trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, act_dist \
                = self.team_trainer[agent_id].policy.get_actions(self.team_buffer[agent_id].share_obs[step],
                                                                 self.team_buffer[agent_id].obs[step],
                                                                 self.team_buffer[agent_id].rnn_states[step],
                                                                 self.team_buffer[agent_id].rnn_states_critic[step],
                                                                 self.team_buffer[agent_id].masks[step])
            value, action_log_prob \
                = self.team_trainer[agent_id].evaluate_actions(self.team_buffer[agent_id].share_obs[step],
                                                               self.team_buffer[agent_id].obs[step],
                                                               self.team_buffer[agent_id].rnn_states[step],
                                                               self.team_buffer[agent_id].rnn_states_critic[step],
                                                               idv_actions[:, agent_id],
                                                               self.team_buffer[agent_id].masks[step])
            # act_dists.append(act_dist)
            tmp_act_dist = []
            for dist in act_dist:
                tmp_probs = dist.probs.detach()
                tps = []
                for tp in tmp_probs:
                    tps.append(type(dist)(probs=tp))
                tmp_act_dist.append(tps)

            act_dists.append(tmp_act_dist)

            # [agents, envs, dim]
            values.append(_t2n(value))
            action = idv_actions[:, agent_id]
            # rearrange action
            # if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
            #     for i in range(self.envs.action_space[agent_id].shape):
            #         uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
            #         if i == 0:
            #             action_env = uc_action_env
            #         else:
            #             action_env = np.concatenate((action_env, uc_action_env), axis=1)
            # elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
            #     action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            # else:
            #     raise NotImplementedError

            actions.append(action)
            # temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        # actions_env = []
        # for i in range(self.n_rollout_threads):
        #     one_hot_action_env = []
        #     for temp_action_env in temp_actions_env:
        #         one_hot_action_env.append(temp_action_env[i])
        #     actions_env.append(one_hot_action_env)

        values = np.array(values).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        action_log_probs = np.array(action_log_probs).transpose((1, 0, 2))
        rnn_states = np.array(rnn_states).transpose((1, 0, 2, 3))
        rnn_states_critic = np.array(rnn_states_critic).transpose((1, 0, 2, 3))
        act_dists = np.array(act_dists).transpose((2, 0, 1))
        # print(act_dists.shape)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists

    def idv_insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, other_log_probs, rnn_states, rnn_states_critic, other_act_dists = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        idv_rewards = []
        for info in infos:
            trw = []
            for i in range(self.num_agents):
                trw.append([info[i]["individual_reward"]])
            idv_rewards.append(trw)
        idv_rewards = np.array(idv_rewards)

        for agent_id in range(self.num_agents):
            if not self.idv_use_shared_obs:
                share_obs = np.array(list(obs[:, agent_id]))

            self.idv_buffer[agent_id].insert(share_obs,
                                             np.array(list(obs[:, agent_id])),
                                             rnn_states[:, agent_id],
                                             rnn_states_critic[:, agent_id],
                                             actions[:, agent_id],
                                             other_act_dists[:, agent_id],
                                             action_log_probs[:, agent_id],
                                             other_log_probs[:, agent_id],
                                             values[:, agent_id],
                                             idv_rewards[:, agent_id],
                                             masks[:, agent_id])

    def team_insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, other_log_probs, rnn_states, rnn_states_critic, other_act_dists = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                    dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        team_rewards = []
        for info in infos:
            trw = []
            for i in range(self.num_agents):
                trw.append([info[i]["team_reward"]])
            team_rewards.append(trw)
        team_rewards = np.array(team_rewards)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.team_buffer[agent_id].insert(share_obs,
                                              np.array(list(obs[:, agent_id])),
                                              rnn_states[:, agent_id],
                                              rnn_states_critic[:, agent_id],
                                              actions[:, agent_id],
                                              other_act_dists[:, agent_id],
                                              action_log_probs[:, agent_id],
                                              other_log_probs[:, agent_id],
                                              values[:, agent_id],
                                              team_rewards[:, agent_id],
                                              masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps, trainer, title):
        idv_episode_rewards, team_episode_rewards = [], []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                   dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        print_eval_infos = [[] for _ in range(self.n_eval_rollout_threads)]
        catch_infos = []

        for eval_step in range(self.all_args.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                           eval_rnn_states[:, agent_id],
                                                                           eval_masks[:, agent_id],
                                                                           deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            team_rewards, idv_rewards, tmp_catch = [], [], []
            for ti, info in enumerate(eval_infos):
                tinfo = ""
                if "detail_infos" in info[0].keys():
                    tinfo += info[0]["detail_infos"]
                if "additional_infos" in info[0].keys():
                    tinfo += info[0]["additional_infos"]
                if tinfo != "":
                    print_eval_infos[ti].append(tinfo)

                trw, irw, tc = [], [], []
                for i in range(self.num_agents):
                    trw.append(info[i]["team_reward"])
                    irw.append(info[i]["individual_reward"])
                    if "catch_infos" in info[i].keys():
                        tc.append(info[i]["catch_infos"])
                team_rewards.append(trw)
                idv_rewards.append(irw)
                if len(tc) > 0:
                    tmp_catch.append(tc)
            idv_episode_rewards.append(idv_rewards)  # episode_length, n_threads, n_agents
            team_episode_rewards.append(team_rewards)
            if len(tmp_catch) > 0:
                catch_infos.append(tmp_catch)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        # self.eval_obs = eval_obs
        # eval_episode_rewards = np.array(eval_episode_rewards)
        idv_episode_rewards = np.array(idv_episode_rewards)
        team_episode_rewards = np.array(team_episode_rewards)
        if len(catch_infos) > 0:
            catch_infos = np.array(catch_infos)
        else:
            catch_infos = None

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            tinfos = {}
            eval_average_episode_rewards = np.mean(idv_episode_rewards[:, :, agent_id])
            tinfos[title + '_eval_average_step_individual_rewards'] = eval_average_episode_rewards
            print(title + " eval average step individual rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

            eval_average_episode_rewards = np.mean(np.sum(team_episode_rewards[:, :, agent_id], axis=0))
            tinfos[title + '_eval_average_episode_team_rewards'] = eval_average_episode_rewards
            print(title + " eval average team episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))
            if catch_infos is not None:
                tinfos[title + "_eval_idv_catch_total_num"] = np.sum(catch_infos[:, :, agent_id, 0])
                print(title + " eval idv catch total num of agent%i: " % agent_id + str(np.sum(catch_infos[:, :, agent_id, 0])))
                tinfos[title + "_eval_team_catch_total_num"] = np.sum(catch_infos[:, :, agent_id, 1])
                print(title + " eval team catch total num: " + str(np.sum(catch_infos[:, :, agent_id, 1])))
            eval_train_infos.append(tinfos)
        self.log_train(eval_train_infos, total_num_steps)

        with open(str(self.eval_log_dir), 'a+') as f:
            for fi, infos in enumerate(print_eval_infos):
                if len(infos) > 0:
                    f.write("*****total training steps %s thread %s infos*****\n" % (total_num_steps, fi))
                    if catch_infos is not None:
                        for agent_id in range(self.num_agents):
                            f.write("eval catch total num of agent %i: " % agent_id
                                    + str(np.sum(catch_infos[:, fi, agent_id, 0])) + "\n")
                        f.write("eval team catch total num: " + str(np.sum(catch_infos[:, fi, 0, 1])) + "\n")
                    for ti, tfs in enumerate(infos):
                        f.write("step % s\n %s" % (ti, tfs))
                    f.write("-------------------------------------------------\n")

    @torch.no_grad()
    def policy_render(self, trainer, title):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            idv_episode_rewards, team_episode_rewards, catch_infos = [], [], []
            print_render_infos = [[] for _ in range(self.n_rollout_threads)]
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    trainer[agent_id].prep_rollout()
                    action, rnn_state = trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                     rnn_states[:, agent_id],
                                                                     masks[:, agent_id],
                                                                     deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                team_rewards, idv_rewards, tmp_catch = [], [], []
                for ti, info in enumerate(infos):
                    tinfo = ""
                    if "detail_infos" in info[0].keys():
                        tinfo += info[0]["detail_infos"]
                    if "additional_infos" in info[0].keys():
                        tinfo += info[0]["additional_infos"]
                    if tinfo != "":
                        print_render_infos[ti].append(tinfo)
                    trw, irw, tc = [], [], []
                    for i in range(self.num_agents):
                        trw.append(info[i]["team_reward"])
                        irw.append(info[i]["individual_reward"])
                        if "catch_infos" in info[i].keys():
                            tc.append(info[i]["catch_infos"])
                    team_rewards.append(trw)
                    idv_rewards.append(irw)
                    if len(tc) > 0:
                        tmp_catch.append(tc)
                idv_episode_rewards.append(idv_rewards)  # episode_length, n_threads, n_agents
                team_episode_rewards.append(team_rewards)
                if len(tmp_catch) > 0:
                    catch_infos.append(tmp_catch)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                                     dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            idv_episode_rewards = np.array(idv_episode_rewards)
            team_episode_rewards = np.array(team_episode_rewards)
            if len(catch_infos) > 0:
                catch_infos = np.array(catch_infos)
            else:
                catch_infos = None
            with open(str(self.gif_dir) + '/' + title + 'info.txt', "a+") as f:
                f.write("episode: " + str(episode) + " start....\n")
                for agent_id in range(self.num_agents):
                    average_idv_rewards = np.mean(idv_episode_rewards[:, :, agent_id])
                    f.write("render average step rewards of agent%i: " % agent_id + str(average_idv_rewards) + "\n")
                    if catch_infos is not None:
                        f.write("render catch total num of agent %i: " % agent_id
                                + str(np.sum(catch_infos[:, :, agent_id, 0])) + "\n")
                if catch_infos is not None:
                    f.write("render team catch total num: " + str(np.sum(catch_infos[:, :, 0, 1])) + "\n")
                average_team_rewards = np.mean(np.sum(team_episode_rewards, axis=0))
                f.write("render average episode team rewards: " + str(average_team_rewards) + "\n")
                f.write("---------------------------------------------------------------\n")

            with open(str(self.eval_log_dir), 'a+') as f:
                for fi, infos in enumerate(print_render_infos):
                    if len(infos) > 0:
                        f.write("*****episode %s thread %s infos*****\n" % (episode, fi))
                        if catch_infos is not None:
                            for agent_id in range(self.num_agents):
                                f.write("render catch total num of agent %i: " % agent_id
                                        + str(np.sum(catch_infos[:, fi, agent_id, 0])) + "\n")
                            f.write("render team catch total num: " + str(np.sum(catch_infos[:, fi, 0, 1])) + "\n")
                        for ti, tfs in enumerate(infos):
                            f.write("step % s\n %s" % (ti, tfs))
                        f.write("-------------------------------------------------\n")

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/' + title + 'render.gif', all_frames, duration=self.all_args.ifi)

    @torch.no_grad()
    def render(self):
        self.policy_render(self.team_trainer, "team_")
        self.policy_render(self.idv_trainer, "individual_")
