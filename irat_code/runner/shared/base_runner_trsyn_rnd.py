import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter

from onpolicy.utils.shared_buffer_trsyn import SharedReplayBuffer


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']

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
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
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

        from onpolicy.algorithms.r_mappo.algorithm.rMAPPORNDPolicy import R_MAPPO_RND_Policy as Policy
        idv_share_observation_space = self.envs.share_observation_space[0] if self.idv_use_shared_obs \
            else self.envs.observation_space[0]
        team_share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V \
            else self.envs.observation_space[0]
        self.policy = Policy(self.all_args,
                             self.envs.observation_space[0],
                             idv_share_observation_space,
                             team_share_observation_space,
                             self.envs.action_space[0],
                             device=self.device)
        # team_share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V \
        #     else self.envs.observation_space[0]
        # self.team_policy = Policy(self.all_args,
        #                           self.envs.observation_space[0],
        #                           team_share_observation_space,
        #                           self.envs.action_space[0],
        #                           device=self.device)

        if self.model_dir is not None:
            self.restore()

        from onpolicy.algorithms.r_mappo.rmappo_trsyn_rnd import RMappoTrSynRnd as TrainAlgo

        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)

        self.buffer = SharedReplayBuffer(self.all_args,
                                         self.num_agents,
                                         self.envs.observation_space[0],
                                         idv_share_observation_space,
                                         team_share_observation_space,
                                         self.envs.action_space[0])

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def idv_insert(self, data):
        raise NotImplementedError

    def team_insert(self, data):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        idv_next_values, team_next_values = self.trainer.policy.get_values(
            np.concatenate(self.buffer.idv_share_obs[-1]),
            np.concatenate(self.buffer.team_share_obs[-1]),
            np.concatenate(self.buffer.idv_rnn_states_critic[-1]),
            np.concatenate(self.buffer.team_rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1])
        )
        idv_next_values = np.array(np.split(_t2n(idv_next_values), self.n_rollout_threads))
        team_next_values = np.array(np.split(_t2n(team_next_values), self.n_rollout_threads))
        self.buffer.idv_compute_returns(idv_next_values, self.trainer.idv_value_normalizer)
        self.buffer.team_compute_returns(team_next_values, self.trainer.team_value_normalizer)

    def train(self, episode):
        self.trainer.prep_training()

        train_infos = self.trainer.train(self.buffer, episode)

        # if self.all_args.idv_use_two_clip:
        #     self.trainer.update_idv_clip_ratio()
        # if self.all_args.idv_use_kl_loss or self.all_args.idv_use_cross_entropy:
        #     self.trainer.update_idv_kl_coef()
        #
        # if self.all_args.team_use_clip:
        #     self.trainer.update_team_clip_ratio()
        # if self.all_args.team_use_kl_loss or self.all_args.team_use_cross_entropy:
        #     self.trainer.update_team_kl_coef()

        self.buffer.after_update()
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
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent.pt")
        idv_policy_critic = self.trainer.policy.idv_critic
        torch.save(idv_policy_critic.state_dict(), str(self.save_dir) + "/individual_critic_agent.pt")
        team_policy_critic = self.trainer.policy.team_critic
        torch.save(team_policy_critic.state_dict(), str(self.save_dir) + "/team_critic_agent.pt")

    def restore(self):
        policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent.pt')
        self.policy.actor.load_state_dict(policy_actor_state_dict)
        idv_policy_critic_state_dict = torch.load(str(self.model_dir) + '/individual_critic_agent.pt')
        self.policy.idv_critic.load_state_dict(idv_policy_critic_state_dict)
        team_policy_critic_state_dict = torch.load(str(self.model_dir) + '/team_critic_agent.pt')
        self.policy.team_critic.load_state_dict(team_policy_critic_state_dict)

    def log_agent(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
