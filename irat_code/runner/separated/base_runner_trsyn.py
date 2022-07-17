import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import copy

from onpolicy.utils.separated_buffer_trsyn import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.idv_use_shared_obs = self.all_args.idv_use_shared_obs
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            import imageio
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = config["run_dir"]
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

        self.eval_log_dir = str(self.run_dir / 'eval_logs.txt')
        # if not os.path.exists(self.eval_log_dir):
        #     os.makedirs(self.eval_log_dir)

        # from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        self.team_policy = []
        self.idv_policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.idv_use_shared_obs else \
                self.envs.observation_space[agent_id]
            ipo = Policy(self.all_args,
                         self.envs.observation_space[agent_id],
                         share_observation_space,
                         self.envs.action_space[agent_id],
                         device=self.device)
            self.idv_policy.append(ipo)

        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V \
                else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.team_policy.append(po)
        # print(list(self.idv_policy[1].actor.base.parameters()))

        if self.model_dir is not None:
            self.restore()

        from onpolicy.algorithms.r_mappo.rmappo_trsyn import RMappoTrSyn as TrainAlgo

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # Trainer
            tr = TrainAlgo(self.all_args, self.idv_policy[agent_id], self.team_policy[agent_id], device=self.device)
            self.trainer.append(tr)

            # Buffer
            idv_share_obs_space = self.envs.share_observation_space[agent_id] if self.idv_use_shared_obs else \
                self.envs.observation_space[agent_id]
            team_share_obs_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       idv_share_obs_space,
                                       team_share_obs_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)

        # print(list(self.trainer[0].team_policy.actor.base.parameters()))
        # print(list(self.trainer[0].idv_policy.actor.base.parameters()))

        # self.eval_obs = self.eval_envs.reset()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def idv_collect(self, step):
        raise NotImplementedError

    def team_collect(self, step):
        raise NotImplementedError

    def idv_insert(self, data):
        raise NotImplementedError

    def team_insert(self, data):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].idv_prep_rollout()
            idv_next_value = self.trainer[agent_id].idv_policy.get_values(self.buffer[agent_id].idv_share_obs[-1],
                                                                          self.buffer[agent_id].idv_rnn_states_critic[-1],
                                                                          self.buffer[agent_id].masks[-1])
            idv_next_value = _t2n(idv_next_value)
            self.buffer[agent_id].idv_compute_returns(idv_next_value, self.trainer[agent_id].idv_value_normalizer)

            self.trainer[agent_id].team_prep_rollout()
            team_next_value = self.trainer[agent_id].team_policy.get_values(self.buffer[agent_id].team_share_obs[-1],
                                                                            self.buffer[agent_id].team_rnn_states_critic[-1],
                                                                            self.buffer[agent_id].masks[-1])
            team_next_value = _t2n(team_next_value)
            self.buffer[agent_id].team_compute_returns(team_next_value, self.trainer[agent_id].team_value_normalizer)

    def train(self, episode):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].idv_prep_training()
            self.trainer[agent_id].team_prep_training()

            train_info = self.trainer[agent_id].train(self.buffer[agent_id], episode)

            if self.all_args.idv_use_two_clip:
                self.trainer[agent_id].update_idv_clip_ratio()
            if self.all_args.idv_use_kl_loss:
                self.trainer[agent_id].update_idv_kl_coef()

            if self.all_args.team_use_clip:
                self.trainer[agent_id].update_team_clip_ratio()
            if self.all_args.team_use_kl_loss:
                self.trainer[agent_id].update_team_kl_coef()

            self.buffer[agent_id].after_update()

            train_infos.append(train_info)
        # print(train_infos)
        # print("policy_loss", train_infos[0]["policy_loss"])
        # print("team_policy_loss", train_infos[0]["team_policy_loss"])
        # print("value_loss", train_infos[0]["value_loss"])
        # print("team_value_loss", train_infos[0]["team_value_loss"])
        # print("eta", train_infos[0]["eta"])
        # print("idv_sigma", train_infos[0]["idv_sigma"])
        # print("team_sigma^", train_infos[0]["team_sigma^"])
        # if episode > 7:
        #     raise NotImplementedError

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].idv_policy.actor
            torch.save(policy_actor.state_dict(),
                       str(self.save_dir) + "/individual_actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].idv_policy.critic
            torch.save(policy_critic.state_dict(),
                       str(self.save_dir) + "/individual_critic_agent" + str(agent_id) + ".pt")

            policy_actor = self.trainer[agent_id].team_policy.actor
            torch.save(policy_actor.state_dict(),
                       str(self.save_dir) + "/team_actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.trainer[agent_id].team_policy.critic
            torch.save(policy_critic.state_dict(),
                       str(self.save_dir) + "/team_critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/individual_actor_agent' + str(agent_id) + '.pt')
            self.idv_policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/individual_critic_agent' + str(agent_id) + '.pt')
            # print(policy_critic_state_dict)
            # print(self.idv_policy[agent_id].critic.state_dict())
            # print(list(self.idv_policy[agent_id].critic.parameters()))
            self.idv_policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/team_actor_agent' + str(agent_id) + '.pt')
            self.team_policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/team_critic_agent' + str(agent_id) + '.pt')
            self.team_policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
