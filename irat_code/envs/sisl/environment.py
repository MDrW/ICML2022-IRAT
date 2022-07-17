from onpolicy.envs.sisl.walker.multi_walker import MultiWalkerEnv
from onpolicy.envs.sisl.pursuit.pursuit_evade import PursuitEvade
from onpolicy.envs.sisl.pursuit_config import convert_puisuit
from onpolicy.envs.sisl.pursuit.waterworld import MAWaterWorld


def get_sisl_envs(config):
    if config.env_name == "MultiWalker":
        env = MultiWalkerEnv(n_walkers=config.n_walkers, position_noise=config.position_noise,
                             angle_noise=config.angle_noise, reward_mech=config.reward_mech,
                             forward_reward=config.forward_reward, fall_reward=config.fall_reward,
                             drop_reward=config.drop_reward, terminate_on_fall=config.terminate_on_fall,
                             ir_use_pos=config.ir_use_pos, one_hot=config.one_hot)
    elif config.env_name == "Pursuit":
        config_dict = convert_puisuit(config)
        env = PursuitEvade(x_size=config.x_size, y_size=config.y_size, **config_dict)
    elif config.env_name == "WaterWorld":
        env = MAWaterWorld(n_pursuers=config.n_pursuers, n_evaders=config.n_evaders, n_coop=config.n_coop,
                           n_poison=config.n_poison, radius=config.radius, obstacle_radius=config.obstacle_radius,
                           obstacle_loc=config.obstacle_loc, ev_speed=config.ev_speed, poison_speed=config.poison_speed,
                           n_sensors=config.n_sensors, sensor_range=config.sensor_range, action_scale=config.action_scale,
                           poison_reward=config.poison_reward, food_reward=config.food_reward,
                           encounter_reward=config.encounter_reward, control_penalty=config.control_penalty,
                           reward_mech=config.reward_mech, addid=config.addid, speed_features=config.speed_features,
                           max_cycles=config.max_cycles, idv_use_caught_food=config.idv_use_caught_food)
    else:
        print("Can not support the " +
              config.env_name + "environment.")
        raise NotImplementedError
    return env
