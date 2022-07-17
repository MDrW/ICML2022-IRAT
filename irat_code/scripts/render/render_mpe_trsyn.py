#!/usr/bin/env python
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from onpolicy.config import get_config

from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "MPE":
                env = MPEEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='simple_spread', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int,
                        default=2, help="number of players")
    parser.add_argument("--num_good_agents", type=int, default=1, help="number of good agents")
    parser.add_argument("--num_adversaries", type=int, default=3, help="number of adversaries")
    parser.add_argument("--collaborative", action='store_false', default=True,
                        help="whether agents use individual or team reward")
    parser.add_argument("--reward_shaping", action='store_true', default=False)
    parser.add_argument("--agent_view_radius", type=float, default=1.0)
    parser.add_argument("--use_partial_obs", action='store_true', default=False)
    parser.add_argument("--rew_bound", action='store_true', default=False)
    parser.add_argument("--game_mode", type=str, default="hard")

    parser.add_argument("--wandb_group", type=str, default="NotDefined", help="wandb group")
    parser.add_argument("--wandb_exp_name", type=str, default="DefaultName", help="wandb name")
    parser.add_argument("--wandb_project", type=str, default="DefaultProject", help="wandb project")
    parser.add_argument("--eval_episode_length", type=int, default=50)

    parser.add_argument("--change_reward", action="store_true", default=False)
    parser.add_argument("--change_reward_episode", type=int, default=10000)
    parser.add_argument("--change_use_policy", type=str, default="team")

    # Algorithm's parameters
    parser.add_argument("--idv_use_shared_obs", action='store_true', default=False,
                        help="whether individual policy use shared observation")
    parser.add_argument("--idv_clip_ratio", type=float, default=10., help="individual clip ratio")
    parser.add_argument("--idv_end_clip_ratio", type=float, default=0.2, help="individual end clip ratio")
    parser.add_argument("--idv_clip_episodes", type=float, default=1000000, help="individual clip episodes")
    parser.add_argument("--team_clip_ratio", type=float, default=0., help="team clip ratio")
    parser.add_argument("--team_end_clip_ratio", type=float, default=10., help="team end clip ratio")
    parser.add_argument("--team_clip_episodes", type=float, default=1000000, help="team clip episodes")

    parser.add_argument("--idv_use_two_clip", action='store_false', default=True,
                        help="whether individual reward policy use two clip")
    parser.add_argument("--idv_use_kl_loss", action='store_true', default=False,
                        help="whether individual reward policy use kl loss")
    parser.add_argument("--idv_use_cross_entropy", action='store_true', default=False)
    parser.add_argument("--idv_kl_coef", type=float, default=0.)
    parser.add_argument("--idv_kl_end_coef", type=float, default=1.)
    parser.add_argument("--idv_kl_episodes", type=float, default=1000000)

    parser.add_argument("--team_use_clip", action='store_false', default=True)
    parser.add_argument("--team_use_kl_loss", action='store_true', default=False)
    parser.add_argument("--team_use_cross_entropy", action='store_true', default=False)
    parser.add_argument("--team_kl_coef", type=float, default=1.)
    parser.add_argument("--team_kl_end_coef", type=float, default=0.)
    parser.add_argument("--team_kl_episodes", type=float, default=1000000)

    parser.add_argument("--idv_kl_loss_use_present", action='store_true', default=False)
    parser.add_argument("--team_kl_loss_use_present", action='store_true', default=False)
    parser.add_argument("--idv_clip_use_present", action='store_true', default=False)
    parser.add_argument("--team_clip_use_present", action='store_true', default=False)

    parser.add_argument("--scenario_has_diff_rewards", action="store_true", default=False)
    parser.add_argument("--sparse_reward", action="store_true", default=False)
    parser.add_argument("--agent_policy", type=str, default="prey")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappotrsyn":
        assert (
                all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappotrsyn":
        assert (all_args.use_recurrent_policy and all_args.use_naive_recurrent_policy) == False, (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    assert all_args.use_render, ("u need to set use_render be True")
    assert not (all_args.model_dir == None or all_args.model_dir == ""), ("set model_dir first")
    assert all_args.n_rollout_threads == 1, ("only support to use 1 env to render.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
                              str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
        all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_render_env(all_args)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.mpe_runner_trsyn import MPERunner as Runner
    else:
        from onpolicy.runner.separated.mpe_runner_trsyn import MPERunner as Runner

    runner = Runner(config)
    runner.render()

    # post process
    envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])
