import logging
from collections import deque

import numpy as np
from gym import spaces
from gym.utils import seeding

import ode
import vapory as vap
from onpolicy.envs.sisl import AbstractMAEnv, Agent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Object constants
LENGTH = 0.6  # object's length
WIDTH = 0.2
HEIGHT = 0.05  # object's height
MASS = 1  # object's mass

# environment constants
MU = 0  #0.5      # the global mu to use # this parameter is discarded, use FRIC instead
GRAVITY = 10  #9.81  #0.5  # the global gravity to use
FRIC = 5  # friction
MU_V = 0.2  # coefficient of the viscous force, which is proportional to velocity

# wall constants
nWall = 6
WALL_THICK = 0.04
WALL_TALL = 0.3

# robot constants
#nRobot = 12
FMAX = 1.4
TMAX = 2
ROBOT_RADIUS = 0.03  # used to visualize the robots, which are drew as spheres

# other constants
TIME_STEP = 0.01
TIME_INTERVAL = 50

WALL_LENGTH = np.array([3, 2.5, 3, 3, 2, 2.5])
WALL_POS = np.array([[5.5, 2], [5.25, 3], [7, 3.5], [6.5, 4.5], [8, 5], [7.75, 6]])
WALL_DIR = [0, 0, 1, 1, 0, 0]  # 1: y-axis wall

ROBOT_REL_POS = np.array([[0.3, 0.1], [0.1, 0.1], [-0.1, 0.1], [-0.3, 0.1], [-0.3, 0.033],
                          [-0.3, -0.033], [-0.3, -0.1], [-0.1, -0.1], [0.1, -0.1], [0.3, -0.1],
                          [0.3, -0.033], [0.3, 0.033]])


class OdeObj(object):

    def __new__(cls, *args, **kwargs):
        obj = super(OdeObj, cls).__new__(cls)
        return obj

    @property
    def body(self):
        raise NotImplementedError()

    @property
    def geom(self):
        raise NotImplementedError()

    @property
    def rendered(self):
        raise NotImplementedError()

    def setPosition(self, *args):
        self.body.setPosition(args)

    def setQuat(self, *args):
        self.body.setQuaternion(args)

    def getPosition(self):
        return self.body.getPosition()

    def getQuat(self):
        return self.body.getQuaternion()


class Box(OdeObj):

    def __init__(self, space, world, size, mass, color=None):
        self._size = size
        self._color = color
        assert len(size) == 3
        self._odebody = ode.Body(world)
        if mass:
            self._odemass = ode.Mass()
            self._odemass.setBox(1, *size)
            self._odemass.adjust(mass)
            self._odebody.setMass(self._odemass)

        self._odegeom = ode.GeomBox(space, size)
        self._odegeom.setBody(self._odebody)

    @property
    def body(self):
        return self._odebody

    @property
    def geom(self):
        return self._odegeom

    @property
    def rendered(self):
        return vap.Box(
            [-s / 2 for s in self._size], [s / 2 for s in self._size],
            vap.Texture('T_Ruby_Glass' if not self._color else vap.Pigment('color', self._color)),
            vap.Interior('ior', 4), 'matrix', self.body.getRotation() + self.body.getPosition())


class SphereRobot(OdeObj, Agent):

    def __init__(self, space, world, radius, mass, color=None):
        self._radius = radius
        self._color = color
        self._odemass = ode.Mass()
        self._odemass.setSphereTotal(0.00001, radius)
        self._odemass.adjust(mass)
        self._odebody = ode.Body(world)
        self._odebody.setMass(self._odemass)

        self._odegeom = ode.GeomSphere(space, radius)
        self._odegeom.setBody(self._odebody)

        # Observation
        self._neighbors = 2
        self._obs_dim = 2 * self._neighbors + 2  # Acceleration of n nearest neighbors + Velocity of object

    @property
    def body(self):
        return self._odebody

    @property
    def geom(self):
        return self._odegeom

    @property
    def rendered(self):
        return vap.Sphere(
            list(self.body.getPosition()), self._radius,
            vap.Texture(vap.Pigment('color', self._color if self._color else [1, 0, 1])))

    @property
    def observation_space(self):
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self._obs_dim,))

    @property
    def action_space(self):
        return spaces.Box(low=0, high=10, shape=(2,))


def axisangle_to_quat(axis, angle):
    norm = np.linalg.norm(axis)
    axis = axis / norm
    angle /= 2
    x, y, z = axis
    w = np.cos(angle)
    x *= np.sin(angle)
    y *= np.sin(angle)
    z *= np.sin(angle)
    return [w, x, y, z]


class BoxCarrying(AbstractMAEnv):

    def __init__(self, n_enemybots=12):
        self.n_robots = 12
        self.n_enemybots = n_enemybots
        self.world = ode.World()
        self.world.setGravity((0, 0, -GRAVITY))
        self.space = ode.HashSpace()
        self.space.enable()
        self.ground = ode.GeomPlane(self.space, (0, 0, 1), 0)
        self.contactgroup = ode.JointGroup()

        self.obj = Box(self.space, self.world, (LENGTH, WIDTH, HEIGHT), MASS)

        self.wall = [None for _ in range(nWall)]

        self.robot = [SphereRobot(self.space, self.world, ROBOT_RADIUS, MASS)
                      for _ in range(self.n_robots)]

        self.joint = [None for _ in range(self.n_robots)]

        self.enemy_bot = [SphereRobot(self.space, self.world, ROBOT_RADIUS, MASS, color=[0, 0, 1])
                          for _ in range(self.n_enemybots)]
        self.seed()

        self.objv = deque(maxlen=3)
        [self.objv.append(np.zeros(2)) for _ in range(3)]
        self.result_force = np.zeros(2)
        self.objacc = np.zeros(2)
        self._is_static = True
        self.result_torque = 0
        self.count = 0
        self.drift_count = 0
        self.sim_time = 0

        self.fricdir = np.zeros(2)

        self.stage = 0
        self.sum_err_f_mag_pid = 0
        self.time_f_mag_pid = 0

    def _init_force(self):
        self.force_NR_2 = self.np_random.rand(self.n_robots, 2) * FMAX / 2
        return self.force_NR_2

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        return [seed1]

    def reset(self):
        # Object
        self.obj.setPosition(0, 0, 0)
        # Wall
        for i in range(nWall):
            self.wall[i] = Box(self.space, self.world, (WALL_LENGTH[i], WALL_THICK, WALL_TALL),
                               None, [0.3, 0.7, 0.1])
            self.wall[i].setPosition(WALL_POS[i, 0], WALL_POS[i, 1], WALL_TALL / 2)
            if WALL_DIR[i] == 1:
                Q = axisangle_to_quat(np.array([0, 0, 1]), np.pi / 2)
                self.wall[i].setQuat(*Q)
        # Robots
        for i in range(4):
            self.robot[i].setPosition(0.33 - i * 0.22, 0.13, ROBOT_RADIUS)
        for i in range(4, 8):
            self.robot[i].setPosition(0.33 - (i - 4) * 0.22, -0.13, ROBOT_RADIUS)
        for i in range(8, 10):
            self.robot[i].setPosition(0.33, 0.047 - (i - 8) * 0.087, ROBOT_RADIUS)
        for i in range(10, 12):
            self.robot[i].setPosition(-0.33, 0.047 - (i - 10) * 0.087, ROBOT_RADIUS)

        for i in range(self.n_robots):
            self.joint[i] = ode.FixedJoint(self.world)
            self.joint[i].attach(self.obj.body, self.robot[i].body)
            self.joint[i].setFixed()

        # Enemy bot
        for ebot in self.enemy_bot:
            loc = (self.np_random.rand(), self.np_random.rand())
            ebot.setPosition(loc[0], loc[1], ROBOT_RADIUS)

        # self.force_NR_2 = self._init_force()
        # self._add_force(self.force_NR_2)

    def _check_static_fric(self):
        self._is_static = False
        if np.linalg.norm(self.objv[-1]) < 0.01:
            if np.linalg.norm(self.objacc) < 0.8:
                self._is_static = True

        if np.linalg.norm(self.objacc) < 0.01:
            if np.linalg.norm(self.objv[-1]) < 0.5:
                self._is_static = True

    def _update_fric_dir(self):
        spdd = np.array(list(self.obj.body.getLinearVel())[:2])
        if not self._is_static:
            self.fricdir = spdd / np.linalg.norm(spdd)
        else:
            self.fricdir = self.robot_sum_force / np.linalg.norm(self.robot_sum_force)

    def _add_force(self, force_NR_2):
        self.robot_sum_force = np.zeros(2)
        for i in range(self.n_robots):
            self.robot_sum_force += force_NR_2[i, :]

        self._update_fric_dir()

        if self._is_static:
            if np.linalg.norm(self.robot_sum_force) <= FRIC:
                if np.linalg.norm(self.objv[-1]) >= 0.07:
                    vel = self.objv[-1] / np.linalg.norm(self.objv[-1])
                    self.result_force = -FRIC * vel[:2]
                else:
                    self.result_force = np.zeros(2)
            else:
                self.result_force = self.robot_sum_force - FRIC * self.fricdir
        else:  # Dynamic friction
            self.result_force = self.robot_sum_force - FRIC * self.fricdir - MU_V * self.objv[-1][:
                                                                                                  2]

        self.obj.body.addForce((self.result_force[0], self.result_force[1], 0))

    def _add_torque(self, force_NR_2):
        if self._is_static:
            return

        self.result_torque = 0
        robot_torque = 0
        fric_torque = 0

        for i in range(self.n_robots):
            f_body = self.obj.body.vectorFromWorld((force_NR_2[i, 0], force_NR_2[i, 1], 0))
            r = np.array([ROBOT_REL_POS[i, 0], ROBOT_REL_POS[i, 1], 0])
            f = np.array([f_body[0], f_body[1]])
            c = np.cross(r, f)
            t = c[2]
            robot_torque += t

        # Torque by friction

        divide_x = LENGTH * 100
        #20.0;
        divide_y = WIDTH * 100
        #20.0;
        offset_x = LENGTH / divide_x / 2
        offset_y = WIDTH / divide_y / 2

        cur_speed = np.linalg.norm(self.objv[-1])
        if cur_speed < 0.3:
            kp = 3
        elif cur_speed >= 0.3 and cur_speed < 0.5:
            kp = 2
        else:
            kp = 1

        fric_sum = np.zeros(2)
        for x in range(int(divide_x)):
            for y in range(int(divide_y)):
                body_point = (-LENGTH / 2) + np.array(
                    [x / divide_x * LENGTH, y / divide_y * WIDTH]) + np.array([offset_x, offset_y])
                body_point_vel = np.array(
                    list(self.obj.body.getRelPointVel((body_point[0], body_point[1], 0))))
                f_world = -kp * FRIC / divide_x / divide_y * body_point_vel / np.linalg.norm(
                    body_point_vel)
                f_body = self.obj.body.vectorFromWorld((f_world[0], f_world[1], 0))

                r = np.array([body_point[0], body_point[1], 0])
                f = np.array([f_body[0], f_body[1], 0])
                c = np.cross(r, f)
                fric_torque += c[2]

                fric_sum += f_world[:2]

            # Viscous torque
        ang_vel = self.obj.body.getAngularVel()
        vis_torque = -0.05 * ang_vel[2]

        self.result_torque = robot_torque + fric_torque + vis_torque

    def _get_acc(self):
        objv = np.array(self.objv)
        assert objv.shape == (3, 2), objv.shape
        dv = objv[1:] - objv[:-1]
        acc = dv.mean(axis=1) * 1 / TIME_STEP
        assert acc.shape == (2,), acc.shape
        return acc

    def _near_callback(self, _, geom1, geom2):
        g1 = (geom1 == self.ground)
        g2 = (geom2 == self.ground)
        if not (g1 ^ g2):
            return

        b1 = geom1.getBody()
        b2 = geom2.getBody()

        contact = ode.collide(geom1, geom2)
        if contact:
            for con in contact[:3]:
                con.setMode(ode.ContactSoftCFM | ode.ContactApprox1)
                con.setMu(MU)
                con.setSoftCFM(0.01)
                j = ode.ContactJoint(self.world, self.contactgroup, con)
                j.attach(b1, b2)

    @property
    def is_terminal(self):
        pass

    def _info(self):
        pos = self.obj.getPosition()
        logger.info("-" * 20)
        logger.info("Rbt Force = {}, sum = {}, stage = {}, is_static = {}".format(
            self.robot_sum_force, np.linalg.norm(self.robot_sum_force), self.stage, 'yes' if
            self._is_static else 'no'))
        logger.info("End Force = {}, sum = {}".format(self.result_force, np.linalg.norm(
            self.result_force)))
        logger.info("Pos: {}".format(pos))
        logger.info("Vel: {}, Acc: {}".format(self.objv[-1], self.objacc))
        logger.info("Abs Vel: {}".format(np.linalg.norm(self.objv[-1])))
        logger.info("Simtime: {}".format(self.sim_time))

    def step(self, force_NR_2):
        self.force_NR_2 = force_NR_2
        self.count += 1
        self.sim_time += TIME_STEP

        ################################
        # Strategy
        self._intelligent_leader_strategy()
        self._baseline()
        ################################

        self._add_force(self.force_NR_2)

        self._add_torque(self.force_NR_2)
        self.obj.body.addTorque((0, 0, self.result_torque))

        self.space.collide(None, self._near_callback)
        self.world.step(TIME_STEP)
        self.contactgroup.empty()

        speed = self.obj.body.getLinearVel()[:2]
        self.objv.append(np.array(list(speed)))
        self.objacc = self._get_acc()

        if self.count == TIME_INTERVAL:
            self._info()
            self.count = 0
            if any(self.objv[-1] == 0) and any(self.result_force == 0):
                self.drift_count += 1
            if self.drift_count == 2:
                self.drift_count = 0
                self.obj.body.setLinearVel((0, 0, 0))

            # TODO
            # obs?: distance and id of nearest robots? distance and id of walls
            # rew?
            # terminal?

    def render(self, screen_size):
        light = vap.LightSource([3, 3, 3], 'color', [3, 3, 3], 'parallel', 'point_at', [0, 0, 0])
        camera = vap.Camera('location', [0.5 * 1, -2 * 1, 3 * 1], 'look_at', [0, 0, 0], 'rotate',
                            [20, 0, 0])
        ground = vap.Plane([0, 0, 1], 0, vap.Texture('T_Stone33'))
        walls = [wall.rendered for wall in self.wall]
        robots = [bot.rendered for bot in self.robot]
        obj = self.obj.rendered
        obj_pos_str = '\"{:2.2f}, {:2.2f}, {:2.2f}\"'.format(*self.obj.body.getPosition())
        for ir, robot in enumerate(self.robot):
            logger.info('{} - {:2.2f}, {:2.2f}, {:2.2f} - {st}'.format(ir, *robot.body.getPosition(
            ), st=self.sim_time))
        logger.info('{} - {}'.format(obj_pos_str, self.sim_time))
        # obj_pos = vap.Text('ttf', '\"timrom.ttf\"', obj_pos_str, 0.1, '0.1 * x', 'rotate',
        #                    '<100,0,10>', 'translate', '-3*x', 'finish',
        #                    '{ reflection .25 specular 1  diffuse 0.1}', 'scale', [0.25, 0.25, 0.25])
        scene = vap.Scene(
            camera, [light, ground, vap.Background('color', [0.2, 0.2, 0.3]), obj] + robots + walls,
            included=["colors.inc", "textures.inc", "glass.inc", "stones.inc"])
        return scene.render(height=screen_size, width=screen_size, antialiasing=0.01,
                            remove_temp=False)

    def _f_mag_pid(self, set_speed):
        kp_f_mag_pid = 6
        ki_f_mag_pid = 0.05
        cur_speed = np.linalg.norm(self.objv[-1])
        err = set_speed - cur_speed
        self.sum_err_f_mag_pid += err  # Integral control
        out = kp_f_mag_pid * err + ki_f_mag_pid * self.sum_err_f_mag_pid + FRIC / self.n_robots
        if out > FMAX:
            out = FMAX

        if out < FRIC / self.n_robots:
            out = FRIC / self.n_robots
        self.time_f_mag_pid += 1
        return out

    def _f_ang_pid(self, set_angle):
        kp_f_ang_pid = 1.5
        max_turn_angle = np.pi / 3
        cur_angle = self._vec_ang(self.objv[-1])
        err = set_angle - cur_angle
        if err > np.pi:
            err = err - 2 * np.pi
        if err < -np.pi:
            err = err + 2 * np.pi

        if err > max_turn_angle / kp_f_ang_pid:
            out = max_turn_angle + cur_angle
        elif err < -max_turn_angle / kp_f_ang_pid:
            out = -max_turn_angle + cur_angle
        else:
            out = kp_f_ang_pid * err + cur_angle

        return out

    def _f_leader_set(self, set_speed, set_angle):
        mag = self._f_mag_pid(set_speed)
        ang = self._f_ang_pid(set_angle)
        self.force_NR_2[0, :] = mag * np.array([np.cos(ang), np.sin(ang)])

    def _vec_ang(self, vec):
        # ARGHHHHHH
        assert vec.shape == (2,)
        if vec[0] == 0:
            if vec[1] > 0:
                return np.pi / 2
            elif vec[1] < 0:
                return 2 * np.pi / 2
            else:
                return 0
        else:
            if vec[1] == 0:
                if vec[0] > 0:
                    return 0
                else:
                    return np.pi
            else:
                temp = np.absolute(vec[1] / vec[0])
                if vec[0] > 0 and vec[1] > 0:
                    return np.arctan(temp)
                elif vec[0] < 0 and vec[1] > 0:
                    return np.pi - np.arctan(temp)
                elif vec[0] < 0 and vec[1] < 0:
                    return np.pi + np.arctan(temp)
                else:
                    return 2 * np.pi - np.arctan(temp)

    def _intelligent_leader_strategy(self):

        def rotate(vec, angle):
            assert vec.shape == (2,)
            return np.array([np.cos(angle) * vec[0] - np.sin(angle) * vec[1],
                             np.sin(angle) * vec[0] + np.cos(angle) * vec[1]])

        if self.stage == 0:
            if self.count == TIME_INTERVAL:
                self.force_NR_2 = self._init_force()
            # Moved?
            if self.count == TIME_INTERVAL - 2:
                if np.linalg.norm(self.objv[-1]) > 0.02:
                    if all(self.robot_sum_force * self.objacc >= 0):
                        self.stage += 1
                        self._is_static = False
                        logger.info('Started Moving, initial_acc: {}'.format(self.objacc))
            return

        self._check_static_fric()

        path_stage = 0
        if path_stage == 0:
            pos = np.array(self.obj.getPosition()[:2])
            vn = pos - np.array([2.5, 0])  # Circle center?
            v_to_path = vn
            vn = (vn * 2.5) / np.linalg.norm(vn)
            v_to_path = vn - v_to_path
            v_path = rotate(vn, -np.pi / 2)
            v_syn = v_path + 4. * v_to_path
            v_syn /= np.linalg.norm(v_syn)

            self._f_leader_set(np.linalg.norm(v_syn), self._vec_ang(v_syn))
            if pos[0] >= 2.5:
                path_stage = 1

    def _default_strategy(self, force_2):
        df = (MASS * self.objacc - self.n_robots * force_2 + FRIC * self.fricdir[:2] + MU_V *
              self.objv[-1]) * TIME_STEP
        temp = df + force_2
        if np.linalg.norm(temp) > FMAX:
            force_2 = FMAX * temp / np.linalg.norm(temp)
        else:
            force_2 = temp

        return force_2

    def _baseline(self):
        for i in range(1, self.n_robots):
            self.force_NR_2[i, :] = self._default_strategy(self.force_NR_2[i, :])
        return self.force_NR_2


if __name__ == '__main__':
    env = BoxCarrying()
    env.reset()
    print('n:{}'.format(env.space.getNumGeoms()))
    print('g:{}'.format(env.world.getGravity()))
    print('o:{}'.format(env.obj.getPosition()))
    for i in range(env.n_robots):
        print(env.robot[i].getPosition())
        print('---')

    env.render(800)
    count = 0
    while True:
        env.step(env._init_force())
        count += 1
        if count % TIME_INTERVAL == 0:
            env.render(800)
    # def make_frame(t):
    #     env.step(env._init_force())
    #     return env.render(800)
    # import moviepy.editor as mpy
    # clip = mpy.VideoClip(make_frame, duration=100)
    # clip.write_videofile("ode.avi", codec="png", fps=20)
    # print('o:{}'.format(env.obj.getPos()))
