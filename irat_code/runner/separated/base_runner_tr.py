import time
import wandb
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import copy

from onpolicy.utils.separated_buffer_tr import SeparatedReplayBuffer
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
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
            self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.team_policy.append(po)

            share_observation_space = self.envs.share_observation_space[agent_id] if self.idv_use_shared_obs else \
                self.envs.observation_space[agent_id]
            ipo = Policy(self.all_args,
                         self.envs.observation_space[agent_id],
                         share_observation_space,
                         self.envs.action_space[agent_id],
                         device=self.device)
            self.idv_policy.append(ipo)

        if self.model_dir is not None:
            self.restore()

        from onpolicy.algorithms.r_mappo.idv_mappo import Idv_RMAPPO as IdvTrainAlgo
        from onpolicy.algorithms.r_mappo.team_mappo import Team_RMAPPO as TeamTrainAlgo
        self.idv_trainer = []
        self.team_trainer = []
        self.idv_buffer = []
        self.team_buffer = []
        for agent_id in range(self.num_agents):
            # Individual Trainer
            itr = IdvTrainAlgo(self.all_args, self.idv_policy[agent_id], device=self.device)
            self.idv_trainer.append(itr)

            # Individual Buffer
            idv_share_obs_space = self.envs.share_observation_space[agent_id] if self.idv_use_shared_obs else \
                self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       idv_share_obs_space,
                                       self.envs.action_space[agent_id])
            self.idv_buffer.append(bu)

            # Team Trainer
            ttr = TeamTrainAlgo(self.all_args, self.team_policy[agent_id], device=self.device)
            self.team_trainer.append(ttr)

            # Team Buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else \
                self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.team_buffer.append(bu)
        # self.eval_obs = self.eval_envs.reset()

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def idv_collect(self, step):
        raise NotImplementedError

    def team_collect(self, step, idv_actions):
        raise NotImplementedError

    def idv_insert(self, data):
        raise NotImplementedError

    def team_insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.idv_trainer[agent_id].prep_rollout()
            next_value = self.idv_trainer[agent_id].policy.get_values(self.idv_buffer[agent_id].share_obs[-1],
                                                                      self.idv_buffer[agent_id].rnn_states_critic[-1],
                                                                      self.idv_buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.idv_buffer[agent_id].compute_returns(next_value, self.idv_trainer[agent_id].value_normalizer)

            self.team_trainer[agent_id].prep_rollout()
            next_value = self.team_trainer[agent_id].policy.get_values(self.team_buffer[agent_id].share_obs[-1],
                                                                       self.team_buffer[agent_id].rnn_states_critic[-1],
                                                                       self.team_buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.team_buffer[agent_id].compute_returns(next_value, self.team_trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            # team_policy = copy.deepcopy(self.team_trainer[agent_id].policy)
            # idv_policy = copy.deepcopy(self.idv_trainer[agent_id].policy)
            self.idv_trainer[agent_id].prep_training()
            # 更新epsilon idv
            self.idv_trainer[agent_id].update_so_clip_ratio()
            # idv policy train
            train_info = self.idv_trainer[agent_id].train(self.idv_buffer[agent_id])
            # 更新KL散度权重
            if self.all_args.idv_use_kl_loss:
                self.idv_trainer[agent_id].update_kl_coef()
            # train_infos.append(train_info)
            # 无限步长，一个episode结束后存储当前状态作为下一个episode的开始状态，
            # 相当于一个环境一直继续下去，相当于以任意状态作为起始状态
            self.idv_buffer[agent_id].after_update()
            # team policy的train
            self.team_trainer[agent_id].prep_training()
            self.team_trainer[agent_id].update_so_clip_ratio()
            train_info.update(self.team_trainer[agent_id].train(self.team_buffer[agent_id]))
            if self.all_args.team_use_kl_loss:
                self.team_trainer[agent_id].update_kl_coef()
            self.team_buffer[agent_id].after_update()

            train_infos.append(train_info)

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.idv_trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(),
                       str(self.save_dir) + "/individual_actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.idv_trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(),
                       str(self.save_dir) + "/individual_critic_agent" + str(agent_id) + ".pt")

            policy_actor = self.team_trainer[agent_id].policy.actor
            torch.save(policy_actor.state_dict(),
                       str(self.save_dir) + "/team_actor_agent" + str(agent_id) + ".pt")
            policy_critic = self.team_trainer[agent_id].policy.critic
            torch.save(policy_critic.state_dict(),
                       str(self.save_dir) + "/team_critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(
                str(self.model_dir) + '/individual_actor_agent' + str(agent_id) + '.pt')
            self.idv_policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + '/individual_critic_agent' + str(agent_id) + '.pt')
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
