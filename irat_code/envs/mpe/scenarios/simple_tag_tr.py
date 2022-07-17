import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        world.world_length = args.episode_length
        num_good_agents = args.num_good_agents
        num_adversaries = args.num_adversaries
        num_agents = num_adversaries + num_good_agents  # deactivate "good" agent
        num_landmarks = args.num_landmarks
        world.collaborative = args.collaborative
        world.rew_bound = getattr(args, "rew_bound", False)

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False     # last agent is good agent
            agent.size = 0.075 if agent.adversary else 0.05
            agent.accel = 3.0 if agent.adversary else 4.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            if i < num_adversaries:
                agent.action_callback = None
            else:
                if args.agent_policy == "random":
                    agent.action_callback = self.random_policy
                elif args.agent_policy == "prey":
                    agent.action_callback = self.prey_policy
                else:
                    print("agent policy %s hasn't completed, agent policy set to random policy" % args.agent_policy)
                    agent.action_callback = self.random_policy
            agent.view_radius = getattr(args, "agent_view_radius", -1)
            # print("AGENT VIEW RADIUS set to: {}".format(agent.view_radius))
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        self.score_function = getattr(args, "score_function", "sum")
        return world

    def random_policy(self, agent, world):
        if agent.movable:
            agent.action.u = (np.random.random(world.dim_p) * 2 - 1)
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity

        agent.action.c = np.zeros(world.dim_c)
        return agent.action

    def prey_policy(self, agent, world):
        action = None
        n = 1000         # number of positions sampled
        # sample actions randomly from a target circle
        length = np.sqrt(np.random.uniform(0, 1, n))
        angle = np.pi * np.random.uniform(0, 2, n)
        x = length * np.cos(angle)
        y = length * np.sin(angle)

        # evaluate score for each position
        # check whether positions are reachable
        # sample a few evenly spaced points on the way and see if they collide with anything
        scores = np.zeros(n, dtype=np.float32)
        n_iter = 5

        if self.score_function == "sum":
            for i in range(n_iter):
                waypoints_length = (length / float(n_iter)) * (i + 1)
                x_wp = waypoints_length * np.cos(angle)
                y_wp = waypoints_length * np.sin(angle)
                proj_pos = np.vstack((x_wp, y_wp)).transpose() + agent.state.p_pos
                idx = ((proj_pos[:, 0] > 1.75).astype(int) + (proj_pos[:, 1] > 1.75).astype(int) + (
                            proj_pos[:, 0] < -1.75).astype(int) +
                       (proj_pos[:, 1] < -1.75).astype(int)) > 0
                for _agent in world.agents:
                    if _agent.name != agent.name:
                        delta_pos = _agent.state.p_pos - proj_pos
                        dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
                        dist_min = _agent.size + agent.size
                        scores[dist < dist_min] = -9999999
                        scores[idx] = -9999999
                        if i == n_iter - 1 and _agent.movable:
                            scores += dist
        elif self.score_function == "min":
            rel_dis = []
            adv_names = []
            adversaries = self.adversaries(world)
            proj_pos = np.vstack((x, y)).transpose() + agent.state.p_pos # the position of the 100 sampled points.
            idx = ((proj_pos[:, 0] > 1.75).astype(int) + (proj_pos[:, 1] > 1.75).astype(int) + (proj_pos[:, 0] < -1.75).astype(int) +
                   (proj_pos[:, 1] < -1.75).astype(int)) > 0
            for adv in adversaries:
                rel_dis.append(np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos))))
                adv_names.append(adv.name)
            min_dis_adv_name = adv_names[np.argmin(rel_dis)]
            for adv in adversaries:
                delta_pos = adv.state.p_pos - proj_pos
                dist = np.sqrt(np.sum(np.square(delta_pos), axis=1))
                dist_min = adv.size + agent.size
                scores[dist < dist_min] = -9999999
                scores[idx] = -9999999
                if adv.name == min_dis_adv_name:
                    scores += dist
        else:
            raise Exception("Unknown score function {}".format(self.score_function))

        # move to best position
        best_idx = np.argmax(scores)
        chosen_action = np.array([x[best_idx], y[best_idx]], dtype=np.float32)
        if scores[best_idx] < 0:
            chosen_action *= np.random.uniform(-1, 1, 2)    # cannot go anywhere
        agent.action.u = chosen_action
        sensitivity = 5.0
        if agent.accel is not None:
            sensitivity = agent.accel
        agent.action.u *= sensitivity
        # print(agent.action.u)
        agent.action.c = np.zeros(world.dim_c)
        return agent.action

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        shape = False
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                rew += 0.1 * np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    rew -= 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew += 10
        return rew

    def individual_reward(self, adv, world):
        rew = 0
        agents = self.good_agents(world)
        rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if adv.collide:
            for a in agents:
                if self.is_collision(a, adv):
                    rew += 5

        def bound(x):
            if x < 1.75:
                return 0
            else:
                return min(np.exp(x - 1.75), 3)
        if world.rew_bound:
            for p in range(world.dim_p):
                x = abs(adv.state.p_pos[p])
                rew -= bound(x)

        return rew

    def team_reward(self, world, sparse_reward=False):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        rew = 0
        for a in agents:
            n = 0
            for adv in adversaries:
                if self.is_collision(a, adv):
                    n += 1
            if n >= 2:
                rew += 20
        return rew

    def info(self, agent, world):
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)

        catch_infos = ""
        total_infos = ""
        catch_n = [0, 0]
        for ai, a in enumerate(agents):
            n = 0
            for di, adv in enumerate(adversaries):
                if self.is_collision(a, adv):
                    catch_infos += "adversary%i catch good_agent%i\n" % (di, ai)
                    n += 1
            total_infos += "There are %i adversaries caught good_agent%i\n" % (n, ai)
            if n >= 2:
                catch_n[1] += 1

        for a in agents:
            if self.is_collision(a, agent):
                catch_n[0] += 1
        infos = {'detail_infos': catch_infos, "additional_infos": total_infos, 'catch_infos': catch_n}
        return infos

    def partial_observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            dist = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
            if not entity.boundary and (agent.view_radius >= 0) and dist <= agent.view_radius:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            else:
                entity_pos.append(np.array([0., 0.]))
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
            if agent.view_radius >= 0 and dist <= agent.view_radius:
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                if not other.adversary:
                    other_vel.append(other.state.p_vel)
            else:
                other_pos.append(np.array([0., 0.]))
                if not other.adversary:
                    other_vel.append(np.array([0., 0.]))
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            if not other.adversary:
                other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)