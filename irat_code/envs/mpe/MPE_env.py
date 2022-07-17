from .environment import MultiAgentEnv
from .scenarios import load


def MPEEnv(args):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''

    # load scenario from script
    scenario = load(args.scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world(args)
    if args.use_partial_obs:
        obs_callback = scenario.partial_observation
    else:
        obs_callback = scenario.observation

    if hasattr(scenario, 'done_callback'):
        done_call = scenario.done_callback
    else:
        done_call = None
    # create multiagent environment
    if args.scenario_has_diff_rewards:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.individual_reward, obs_callback, scenario.info,
                            team_reward_callback=scenario.team_reward, sparse_reward=args.sparse_reward,
                            reward_shaping=args.reward_shaping, discrete_action=args.discrete_action,
                            done_callback=done_call)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, obs_callback, scenario.info,
                            discrete_action=args.discrete_action,
                            done_callback=done_call)

    return env
