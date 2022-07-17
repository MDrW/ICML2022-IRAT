import argparse


def get_pursuit_config(parser):
    """
    n_walkers = 2, position_noise = 1e-3, angle_noise = 1e-3, reward_mech = 'local',
    forward_reward = 1.0, fall_reward = -100.0, drop_reward = -100.0, terminate_on_fall = False,
    one_hot = False
    """
    # parser = argparse.ArgumentParser(description='Pursuit', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--x_size", type=int, default=16)
    parser.add_argument("--y_size", type=int, default=16)
    parser.add_argument("--reward_mech", type=str, default="local")
    parser.add_argument("--max_cycles", type=int, default=200)
    parser.add_argument("--n_evaders", type=int, default=30)
    parser.add_argument("--n_pursuers", type=int, default=8)
    parser.add_argument("--num_agents", type=int, default=8)
    parser.add_argument("--obs_range", type=int, default=7)
    parser.add_argument("--n_catch", type=int, default=2)
    parser.add_argument("--catchr", type=float, default=0.01)
    parser.add_argument("--term_pursuit", type=float, default=5.0)
    parser.add_argument("--urgency_reward", type=float, default=0.0)
    parser.add_argument("--team_dec", action='store_true', default=False)
    parser.add_argument("--flatten", action='store_false', default=True)
    return parser


def convert_puisuit(config):
    r = dict()
    # r["x_size"] = config.x_size
    # r["y_size"] = config.y_size
    r["reward_mech"] = config.reward_mech
    r["max_cycles"] = config.max_cycles
    r["n_evaders"] = config.n_evaders
    r["n_pursuers"] = config.n_pursuers
    r["obs_range"] = config.obs_range
    r["n_catch"] = config.n_catch
    r["catchr"] = config.catchr
    r["term_pursuit"] = config.term_pursuit
    r["urgency_reward"] = config.urgency_reward
    r["team_dec"] = config.team_dec
    r['flatten'] = config.flatten
    r['seed'] = config.seed
    return r
