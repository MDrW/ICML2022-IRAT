import wandb
import os
import numpy as np
import torch
from tensorboardX import SummaryWriter
from onpolicy.utils.shared_buffer_tr import SharedReplayBuffer


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """

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

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
        from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy

        team_share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V \
            else self.envs.observation_space[0]

        # team policy network
        self.team_policy = Policy(self.all_args,
                                  self.envs.observation_space[0],
                                  team_share_observation_space,
                                  self.envs.action_space[0],
                                  device=self.device)

        idv_share_observation_space = self.envs.share_observation_space[0] if self.idv_use_shared_obs \
            else self.envs.observation_space[0]

        # individual policy network
        self.idv_policy = Policy(self.all_args,
                                 self.envs.observation_space[0],
                                 idv_share_observation_space,
                                 self.envs.action_space[0],
                                 device=self.device)

        if self.model_dir is not None:
            self.restore()

        from onpolicy.algorithms.r_mappo.idv_mappo import Idv_RMAPPO as IdvTrainAlgo
        from onpolicy.algorithms.r_mappo.team_mappo import Team_RMAPPO as TeamTrainAlgo

        # algorithm
        self.team_trainer = TeamTrainAlgo(self.all_args, self.team_policy, device=self.device)
        self.idv_trainer = IdvTrainAlgo(self.all_args, self.idv_policy, device=self.device)

        # buffer
        self.team_buffer = SharedReplayBuffer(self.all_args,
                                              self.num_agents,
                                              self.envs.observation_space[0],
                                              team_share_observation_space,
                                              self.envs.action_space[0])
        self.idv_buffer = SharedReplayBuffer(self.all_args,
                                             self.num_agents,
                                             self.envs.observation_space[0],
                                             idv_share_observation_space,
                                             self.envs.action_space[0])

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def idv_collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def team_collect(self, step, idv_actions):
        raise NotImplementedError

    def idv_insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    def team_insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                     np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                     np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)
        self.buffer.after_update()
        return train_infos

    def save(self):
        """Save policy's actor and critic networks."""
        team_policy_actor = self.team_trainer.policy.actor
        torch.save(team_policy_actor.state_dict(), str(self.save_dir) + "/team_actor.pt")
        team_policy_critic = self.team_trainer.policy.critic
        torch.save(team_policy_critic.state_dict(), str(self.save_dir) + "/team_critic.pt")

        idv_policy_actor = self.idv_trainer.policy.actor
        torch.save(idv_policy_actor.state_dict(), str(self.save_dir) + "/idv_actor.pt")
        idv_policy_critic = self.idv_trainer.policy.critic
        torch.save(idv_policy_critic.state_dict(), str(self.save_dir) + "/idv_critic.pt")

    def restore(self):
        """Restore policy's networks from a saved model."""
        team_policy_actor = torch.load(str(self.model_dir) + '/team_actor.pt')
        self.team_policy.actor.load_state_dict(team_policy_actor)
        if not self.all_args.use_render:
            team_policy_critic = torch.load(str(self.model_dir) + '/team_critic.pt')
            self.team_policy.critic.load_state_dict(team_policy_critic)

        idv_policy_actor = torch.load(str(self.model_dir) + '/idv_actor.pt')
        self.idv_policy.actor.load_state_dict(idv_policy_actor)
        if not self.all_args.use_render:
            idv_policy_critic = torch.load(str(self.model_dir) + '/idv_critic.pt')
            self.idv_policy.critic.load_state_dict(idv_policy_critic)

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
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
