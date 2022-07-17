import argparse


def get_waterworld_config(parser):
    """
    n_pursuers, n_evaders, n_coop=2, n_poison=10, radius=0.015,
    obstacle_radius=0.2, obstacle_loc=np.array([0.5, 0.5]), ev_speed=0.01,
    poison_speed=0.01, n_sensors=30, sensor_range=0.2, action_scale=0.01,
    poison_reward=-1., food_reward=1., encounter_reward=.05, control_penalty=-.5,
    reward_mech='local', addid=True, speed_features=True, **kwargs
    """
    # parser = argparse.ArgumentParser(description='MultiWalker', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_pursuers", type=int, default=16)
    parser.add_argument("--num_agents", type=int, default=16)
    parser.add_argument("--n_evaders", type=int, default=5)
    parser.add_argument("--n_coop", type=int, default=4)
    parser.add_argument("--n_poison", type=int, default=10)
    parser.add_argument("--radius", type=float, default=0.015)
    parser.add_argument("--obstacle_radius", type=float, default=0.2)
    parser.add_argument("--obstacle_loc", type=list, default=[0.5, 0.5])
    parser.add_argument("--ev_speed", type=float, default=0.01)
    parser.add_argument("--poison_speed", type=float, default=0.01)
    parser.add_argument("--n_sensors", type=int, default=30)
    parser.add_argument("--sensor_range", type=float, default=0.2)
    parser.add_argument("--action_scale", type=float, default=0.01)
    parser.add_argument("--poison_reward", type=float, default=-1.0)
    parser.add_argument("--food_reward", type=float, default=1.0)
    parser.add_argument("--encounter_reward", type=float, default=0.05)
    parser.add_argument("--control_penalty", type=float, default=-0.5)
    parser.add_argument("--reward_mech", type=str, default="local")
    parser.add_argument("--addid", action='store_false', default=True)
    parser.add_argument("--speed_features", action='store_false', default=True)
    parser.add_argument("--max_cycles", type=int, default=1000)
    parser.add_argument("--idv_use_caught_food", action='store_true', default=False)
    return parser
