import time
import numpy as np
from itertools import chain
import torch
from copy import deepcopy

from onpolicy.runner.shared.base_runner_trsyn_rnd import Runner
import imageio
from onpolicy.algorithms.utils.distributions import FixedNormal, FixedCategorical, FixedBernoulli


def _t2n(x):
    return x.detach().cpu().numpy()


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                # self.trainer.team_policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions using Individual Policy
                values, team_values, actions, action_log_probs, rnn_states, \
                rnn_states_critic, team_rnn_critic, actions_env, act_dists = \
                    self.collect(step)

                # values, team_values, action_log_probs = self.evaluate_actions(step, actions)
                team_log_probs = deepcopy(action_log_probs)
                team_rnn = deepcopy(rnn_states)
                team_act_dists = deepcopy(act_dists)

                # else:
                #     team_values, team_log_probs = self.evaluate_actions("team", step, actions)
                # if step == 0:
                #     print(list(self.trainer[0].team_policy.actor.base.parameters()))
                #     print(list(self.trainer[1].idv_policy.actor.base.parameters()))
                #     print("individual actions", actions.shape, actions)
                #     print(act_dists[0][0][0].probs)
                #     print(team_values)
                #     print(team_log_probs)
                #     raise NotImplementedError

                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # insert data into buffer
                data = obs, rewards, dones, infos, \
                       values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists, \
                       team_values, team_log_probs, team_rnn, team_rnn_critic, team_act_dists
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
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.env_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                # if self.env_name == "MPE":
                agent_infos = [{} for _ in range(self.num_agents)]
                for agent_id in range(self.num_agents):
                    agent_infos[agent_id].update(
                        {'average_step_individual_rewards': np.mean(self.buffer.idv_rewards[:, :, agent_id])})
                    agent_infos[agent_id].update(
                        {"average_episode_team_rewards": np.mean(np.sum(
                            self.buffer.team_rewards[:, :, agent_id], axis=0))})
                self.log_agent(agent_infos, total_num_steps)

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, "team_policy")
                self.eval(total_num_steps, "idv_policy")

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        if self.idv_use_shared_obs:
            idv_share_obs = obs.reshape(self.n_rollout_threads, -1)
            idv_share_obs = np.expand_dims(idv_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            idv_share_obs = obs

        if self.use_centralized_V:
            team_share_obs = obs.reshape(self.n_rollout_threads, -1)
            team_share_obs = np.expand_dims(team_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            team_share_obs = obs

        self.buffer.obs[0] = obs.copy()
        self.buffer.idv_share_obs[0] = idv_share_obs.copy()
        self.buffer.team_share_obs[0] = team_share_obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        idv_value, team_value, action, action_log_prob, rnn_states, \
        idv_rnn_states_critic, team_rnn_states_critic, act_dist = \
            self.trainer.policy.get_actions(np.concatenate(self.buffer.idv_share_obs[step]),
                                            np.concatenate(self.buffer.team_share_obs[step]),
                                            np.concatenate(self.buffer.obs[step]),
                                            np.concatenate(self.buffer.idv_rnn_states[step]),
                                            np.concatenate(self.buffer.idv_rnn_states_critic[step]),
                                            np.concatenate(self.buffer.team_rnn_states_critic[step]),
                                            np.concatenate(self.buffer.masks[step]))

        # [self.envs, agents, dim]
        idv_values = np.array(np.split(_t2n(idv_value), self.n_rollout_threads))
        team_values = np.array(np.split(_t2n(team_value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        idv_rnn_states_critic = np.array(np.split(_t2n(idv_rnn_states_critic), self.n_rollout_threads))
        team_rnn_states_critic = np.array(np.split(_t2n(team_rnn_states_critic), self.n_rollout_threads))
        # if self.envs.action_space[0].__class__.__name__ == 'Box':
        #     actions = np.clip(actions, self.envs.action_space[0].low[0], self.envs.action_space[0].high[0])

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

        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        elif self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions
        else:
            raise NotImplementedError

        return idv_values, team_values, actions, action_log_probs, rnn_states, \
               idv_rnn_states_critic, team_rnn_states_critic, actions_env, act_dists

    @torch.no_grad()
    def evaluate_actions(self, step, actions):
        self.trainer.prep_rollout()

        idv_values, team_values, action_log_probs, _, _ \
            = self.trainer.policy.evaluate_actions(np.concatenate(self.buffer.idv_share_obs[step]),
                                                   np.concatenate(self.buffer.team_share_obs[step]),
                                                   np.concatenate(self.buffer.obs[step]),
                                                   np.concatenate(self.buffer.idv_rnn_states[step]),
                                                   np.concatenate(self.buffer.idv_rnn_states_critic[step]),
                                                   np.concatenate(self.buffer.team_rnn_states_critic[step]),
                                                   np.concatenate(actions),
                                                   np.concatenate(self.buffer.masks[step]))

        idv_values = np.array(np.split(_t2n(idv_values), self.n_rollout_threads))
        team_values = np.array(np.split(_t2n(team_values), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))

        return idv_values, team_values, action_log_probs

    def insert(self, data):
        obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic, act_dists, \
        team_values, team_log_probs, team_rnn, team_rnn_critic, team_act_dists = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(),
                                                     *self.buffer.idv_rnn_states_critic.shape[3:]),
                                                    dtype=np.float32)

        team_rnn[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                           dtype=np.float32)
        team_rnn_critic[dones == True] = np.zeros(((dones == True).sum(),
                                                   *self.buffer.team_rnn_states_critic.shape[3:]),
                                                  dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.idv_use_shared_obs:
            idv_share_obs = obs.reshape(self.n_rollout_threads, -1)
            idv_share_obs = np.expand_dims(idv_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            idv_share_obs = obs

        if self.use_centralized_V:
            team_share_obs = obs.reshape(self.n_rollout_threads, -1)
            team_share_obs = np.expand_dims(team_share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            team_share_obs = obs

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
                           values, team_values, idv_rewards, team_rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps, title):
        idv_episode_rewards, team_episode_rewards = [], []
        eval_obs = self.eval_envs.reset()

        # if title == "team_policy":
        #     eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.team_rnn_states.shape[2:]),
        #                                dtype=np.float32)
        # else:
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.idv_rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        print_eval_infos = [[] for _ in range(self.n_eval_rollout_threads)]
        catch_infos = []

        for eval_step in range(self.all_args.episode_length):
            if title == "team_policy":
                self.trainer.prep_rollout()
                policy = self.trainer.policy
            else:
                self.trainer.prep_rollout()
                policy = self.trainer.policy
            eval_action, eval_rnn_states = policy.act(np.concatenate(eval_obs),
                                                      np.concatenate(eval_rnn_states),
                                                      np.concatenate(eval_masks),
                                                      deterministic=True)

            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[0].shape):
                    uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = uc_actions_env
                    else:
                        eval_actions_env = np.concatenate((eval_actions_env, uc_actions_env), axis=2)
            elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                eval_actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[eval_actions], 2)
            elif self.envs.action_space[0].__class__.__name__ == 'Box':
                eval_actions_env = eval_actions
            else:
                raise NotImplementedError

            # if self.eval_envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            #     for i in range(self.eval_envs.action_space[0].shape):
            #         eval_uc_actions_env = np.eye(self.eval_envs.action_space[0].high[i] + 1)[eval_actions[:, :, i]]
            #         if i == 0:
            #             eval_actions_env = eval_uc_actions_env
            #         else:
            #             eval_actions_env = np.concatenate((eval_actions_env, eval_uc_actions_env), axis=2)
            # elif self.eval_envs.action_space[0].__class__.__name__ == 'Discrete':
            #     eval_actions_env = np.squeeze(np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2)
            # else:
            #     raise NotImplementedError

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

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
            print(title + " eval average step individual rewards of agent%i: " % agent_id + str(
                eval_average_episode_rewards))

            eval_average_episode_rewards = np.mean(np.sum(team_episode_rewards[:, :, agent_id], axis=0))
            tinfos[title + '_eval_average_episode_team_rewards'] = eval_average_episode_rewards
            print(title + " eval average team episode rewards of agent%i: " % agent_id + str(
                eval_average_episode_rewards))
            if catch_infos is not None:
                tinfos[title + "_eval_idv_catch_total_num"] = np.sum(catch_infos[:, :, agent_id, 0])
                print(title + " eval idv catch total num of agent%i: " % agent_id + str(
                    np.sum(catch_infos[:, :, agent_id, 0])))
                tinfos[title + "_eval_team_catch_total_num"] = np.sum(catch_infos[:, :, agent_id, 1])
                print(title + " eval team catch total num: " + str(np.sum(catch_infos[:, :, agent_id, 1])))
            eval_train_infos.append(tinfos)
        self.log_agent(eval_train_infos, total_num_steps)

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
    def policy_render(self, title):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            idv_episode_rewards, team_episode_rewards, catch_infos = [], [], []
            print_render_infos = [[] for _ in range(self.n_rollout_threads)]
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0]
                all_frames.append(image)
            else:
                self.envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                                  dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                # for agent_id in range(self.num_agents):
                if title == "individual_":
                    self.trainer.prep_rollout()
                    policy = self.trainer.policy
                else:
                    self.trainer.prep_rollout()
                    policy = self.trainer.policy
                action, rnn_states = policy.act(np.concatenate(obs),
                                                np.concatenate(rnn_states),
                                                np.concatenate(masks),
                                                deterministic=True)

                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.envs.action_space[0].shape):
                        uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
                elif self.envs.action_space[0].__class__.__name__ == 'Box':
                    actions_env = actions
                else:
                    raise NotImplementedError
                # if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                #     for i in range(self.envs.action_space[0].shape):
                #         uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                #         if i == 0:
                #             actions_env = uc_actions_env
                #         else:
                #             actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                # elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                #     actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
                # else:
                #     raise NotImplementedError

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
                    image = self.envs.render('rgb_array')[0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    self.envs.render("human")

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
                f.write("-------------------------------------------------------------------\n")

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
        self.policy_render("team_")
        self.policy_render("individual_")
