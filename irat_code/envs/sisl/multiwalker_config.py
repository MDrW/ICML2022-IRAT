import argparse


def get_multiwalker_config(parser):
    """
    n_walkers = 2, position_noise = 1e-3, angle_noise = 1e-3, reward_mech = 'local',
    forward_reward = 1.0, fall_reward = -100.0, drop_reward = -100.0, terminate_on_fall = False,
    one_hot = False
    """
    # parser = argparse.ArgumentParser(description='MultiWalker', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_walkers", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--position_noise", type=float, default=1e-3)
    parser.add_argument("--angle_noise", type=float, default=1e-3)
    parser.add_argument("--reward_mech", type=str, default='local')
    parser.add_argument("--forward_reward", type=float, default=1.0)
    parser.add_argument("--fall_reward", type=float, default=-100.0)
    parser.add_argument("--drop_reward", type=float, default=-100.0)
    parser.add_argument("--terminate_on_fall", action='store_true', default=False)
    parser.add_argument("--ir_use_pos", action='store_true', default=False)
    parser.add_argument("--one_hot", action='store_true', default=False)
    return parser
