    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

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
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            idv_rewards = []
            team_rewards = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic, episode
                
                # insert data into buffer
                self.insert(data)

                for info in infos:
                    irw, trw = [], []
                    for i in range(self.num_agents):
                        irw.append(info[i]["individual_reward"])
                        trw.append(info[i]["team_reward"])
                    idv_rewards.append(irw)
                    team_rewards.append(trw)

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
                        idv_rewards = np.array(idv_rewards)
                        team_rewards = np.array(team_rewards)
                        train_infos[agent_id].update(
                            {'average_step_individual_rewards': np.mean(idv_rewards[-1])})
                        train_infos[agent_id].update(
                            {"average_episode_team_rewards": np.mean(team_rewards) * self.episode_length})
                        # idv_rews = []
                        # for info in infos:
                        #     if 'individual_reward' in info[agent_id].keys():
                        #         idv_rews.append(info[agent_id]['individual_reward'])
                        # train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        # train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps, "team_policy")
                self.eval(total_num_steps, "idv_policy")

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, _ \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
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
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, episode = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        idv_rewards, team_rewards = [], []
        for info in infos:
            trw, irw = [], []
            for i in range(self.num_agents):
                irw.append([info[i]["individual_reward"]])
                trw.append([info[i]["team_reward"]])
            idv_rewards.append(irw)
            team_rewards.append(trw)
        idv_rewards = np.array(idv_rewards)
        team_rewards = np.array(team_rewards)

        if self.all_args.change_reward:
            if episode > self.all_args.change_reward_episode:
                rewards = team_rewards
            else:
                rewards = idv_rewards

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps, title: str):
        idv_episode_rewards, team_episode_rewards = [], []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        print_eval_infos = [[] for _ in range(self.n_eval_rollout_threads)]
        catch_infos = []

        for eval_step in range(self.all_args.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
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
            # eval_episode_rewards.append(eval_rewards)
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
                catch_infos.append(tmp_catch)       # episode_length, n_threads, n_agents, n_catchs

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
        # self.eval_obs = eval_obs

        # eval_episode_rewards = np.array(eval_episode_rewards)

        # eval_train_infos = []
        # for agent_id in range(self.num_agents):
        #     tinfos = {}
        #     eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
        #     tinfos['eval_average_episode_rewards'] = eval_average_episode_rewards
        #     eval_train_infos.append(tinfos)
        #     print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))
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
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            idv_episode_rewards, team_episode_rewards, catch_infos = [], [], []
            print_render_infos = [[] for _ in range(self.n_rollout_threads)]
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
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

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
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
            with open(str(self.gif_dir) + '/info.txt', "a+") as f:
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
                f.write("-------------------------------------------------------\n")

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
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
